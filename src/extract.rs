//! Pull planes back out of a cube.
//!
//! Only the requested plane is read. The output keeps the cube's axes, with the
//! FREQ/TIME axis of length 1, and takes its beam from the `BEAMS` table.
use std::path::{Path, PathBuf};

use atfits_rs::raw::{Pixel, disable_scaling, read_pixels, write_pixels};
use atfits_rs::{
    HeaderGeom, TargetAxis, copy_hdu_header, delete_key, key_bool, resize_image, update_key_f64,
    update_key_i64,
};
use fitsio::errors::check_status;
use fitsio::{FitsFile, sys};
use rayon::prelude::*;

use crate::beams::{Beam, read_beam_table};
use crate::combine::{check_exists, in_pool};
use crate::error::{FitsCubeError, Result};
use crate::partial::PartialFile;
use crate::pixels::with_pixel_type;
use crate::wcs::{find_axis, world};

/// Which plane of which HDU to extract, and where to.
#[derive(Debug, Clone, Default)]
pub struct ExtractOptions {
    pub hdu_index: usize,
    /// Frequency channel. Exactly one of this and `time_index` must be set.
    pub channel_index: Option<usize>,
    pub time_index: Option<usize>,
    pub overwrite: bool,
    /// Defaults to `cube.channel-N.fits` (or `timestep-N`) beside the cube.
    pub output_path: Option<PathBuf>,
}

/// Options for [`extract_planes_from_cube`].
#[derive(Debug, Clone)]
pub struct ExtractPlanesOptions {
    pub hdu_index: usize,
    pub overwrite: bool,
    /// One path per channel. Defaults to `cube.channel-N.fits` beside the cube.
    pub output_paths: Option<Vec<PathBuf>>,
    /// Planes written at once. `None` uses every core.
    pub max_workers: Option<usize>,
}

impl Default for ExtractPlanesOptions {
    fn default() -> Self {
        Self {
            hdu_index: 0,
            overwrite: false,
            output_paths: None,
            max_workers: Some(4),
        }
    }
}

/// The axis to extract along, and the index along it.
struct TargetIndex {
    axis_name: &'static str,
    axis_index: usize,
    output_name: &'static str,
}

fn create_target_index(opts: &ExtractOptions) -> Result<TargetIndex> {
    match (opts.channel_index, opts.time_index) {
        (Some(_), Some(_)) => Err(FitsCubeError::InvalidSpec(
            "Both time and channel index are set. Only one may be set.".to_string(),
        )),
        (None, None) => Err(FitsCubeError::InvalidSpec(
            "Both channel index and time index are None. One needs to be set.".to_string(),
        )),
        (Some(c), None) => Ok(TargetIndex {
            axis_name: "FREQ",
            axis_index: c,
            output_name: "channel",
        }),
        (None, Some(t)) => Ok(TargetIndex {
            axis_name: "TIME",
            axis_index: t,
            output_name: "timestep",
        }),
    }
}

/// `cube.fits` to `cube.channel-5.fits`.
fn default_output_path(input: &Path, output_name: &str, index: usize) -> PathBuf {
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    let stem = input.file_stem().unwrap_or_default().to_string_lossy();
    let ext = input.extension().unwrap_or_default().to_string_lossy();
    parent.join(format!("{stem}.{output_name}-{index}.{ext}"))
}

/// Extract one plane of `cube` into a new image and return its path.
pub fn extract_plane_from_cube(cube: &Path, opts: &ExtractOptions) -> Result<PathBuf> {
    let target = create_target_index(opts)?;
    let out_path = opts
        .output_path
        .clone()
        .unwrap_or_else(|| default_output_path(cube, target.output_name, target.axis_index));
    let (mut fptr, source) = Cube::open(cube, opts.hdu_index, target.axis_name)?;
    source.write_plane(&mut fptr, target.axis_index, &out_path, opts.overwrite)?;
    Ok(out_path)
}

/// Extract every channel of `cube` into its own image and return their paths.
///
/// The cube and its beam table are read once, rather than once per plane.
pub fn extract_planes_from_cube(cube: &Path, opts: &ExtractPlanesOptions) -> Result<Vec<PathBuf>> {
    let (_, source) = Cube::open(cube, opts.hdu_index, "FREQ")?;
    let n = source.n_planes();
    let out_paths = match &opts.output_paths {
        None => (0..n)
            .map(|i| default_output_path(cube, "channel", i))
            .collect(),
        Some(paths) if paths.len() == n => paths.clone(),
        Some(paths) => {
            return Err(FitsCubeError::ShapeMismatch(format!(
                "Have {} output paths for a cube of {n} channels",
                paths.len()
            )));
        }
    };
    let pool = opts
        .max_workers
        .map(|n| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n.max(1))
                .build()
        })
        .transpose()?;
    in_pool(pool.as_ref(), || {
        out_paths.par_iter().enumerate().try_for_each_init(
            || None,
            |fptr: &mut Option<FitsFile>, (i, out)| {
                let fptr = match fptr {
                    Some(fptr) => fptr,
                    None => fptr.insert(source.reopen()?),
                };
                source.write_plane(fptr, i, out, opts.overwrite)
            },
        )
    })?;
    Ok(out_paths)
}

/// The parts of a cube that every extracted plane needs.
struct Cube {
    path: PathBuf,
    hdu_index: usize,
    geom: HeaderGeom,
    axis: TargetAxis,
    /// The header says beams are in a table (CASAMBM).
    casambm: bool,
    beams: Option<Vec<Beam>>,
}

impl Cube {
    fn open(path: &Path, hdu_index: usize, axis_name: &str) -> Result<(FitsFile, Self)> {
        check_exists(path)?;
        let mut fptr = FitsFile::open(path)?;
        fptr.hdu(hdu_index)?;
        let geom = HeaderGeom::from_file(&mut fptr)?;
        let axis = find_axis(&mut fptr, axis_name)?.ok_or_else(|| {
            FitsCubeError::TargetAxisMissing(format!("Did not find the {axis_name} axis"))
        })?;
        let casambm = key_bool(&mut fptr, "CASAMBM")?.unwrap_or(false);
        let beams = if casambm {
            let beams = read_beam_table(&mut fptr)?;
            if beams.is_none() {
                tracing::warn!("{} sets CASAMBM but has no BEAMS table", path.display());
            }
            fptr.hdu(hdu_index)?;
            beams
        } else {
            None
        };
        let cube = Self {
            path: path.to_path_buf(),
            hdu_index,
            geom,
            axis,
            casambm,
            beams,
        };
        Ok((fptr, cube))
    }

    fn reopen(&self) -> Result<FitsFile> {
        let mut fptr = FitsFile::open(&self.path)?;
        fptr.hdu(self.hdu_index)?;
        Ok(fptr)
    }

    fn n_planes(&self) -> usize {
        self.geom.dims[self.axis.fits_idx - 1]
    }

    /// Write plane `index` to `out`, read through `fptr`.
    fn write_plane(
        &self,
        fptr: &mut FitsFile,
        index: usize,
        out: &Path,
        overwrite: bool,
    ) -> Result<()> {
        let n = self.n_planes();
        if index >= n {
            return Err(FitsCubeError::ChannelMissing(format!(
                "index {index} outside cube axis of length {n}"
            )));
        }
        let beam = match &self.beams {
            Some(beams) => Some(*beams.get(index).ok_or_else(|| {
                FitsCubeError::ChannelMissing(format!(
                    "beam table has {} rows, requested index {index}",
                    beams.len()
                ))
            })?),
            None => None,
        };
        if out.exists() {
            if same_file(&self.path, out) {
                return Err(FitsCubeError::OutputIsInput(out.to_path_buf()));
            }
            if !overwrite {
                return Err(FitsCubeError::OutputExists(out.to_path_buf()));
            }
        }
        let partial = PartialFile::create(out)?;
        with_pixel_type!(
            self.geom.bitpix,
            copy_plane(self, fptr, index, beam, &partial.path)
        )?;
        partial.persist(out)
    }
}

/// Copy plane `index` of the cube open at `src`, with the cube's header
/// adjusted for one plane, into a new file at `path`.
fn copy_plane<T: Pixel>(
    cube: &Cube,
    src: &mut FitsFile,
    index: usize,
    beam: Option<Beam>,
    path: &Path,
) -> Result<()> {
    let fi = cube.axis.fits_idx;
    let mut ranges: Vec<(usize, usize)> = cube.geom.dims.iter().map(|&n| (0, n)).collect();
    ranges[fi - 1] = (index, index + 1);
    let mut plane: Vec<T> = Vec::new();
    // Stored values are copied as they are, under the cube's BSCALE and BZERO
    disable_scaling(src)?;
    read_pixels(src, Some(&ranges), None, &mut plane)?;

    let mut out = copy_hdu_header(src, path)?;
    let mut dims = cube.geom.dims.clone();
    dims[fi - 1] = 1;
    resize_image(&mut out, cube.geom.bitpix, &dims)?;
    update_key_i64(&mut out, &format!("CRPIX{fi}"), 1)?;
    update_key_f64(&mut out, &format!("CRVAL{fi}"), world(&cube.axis, index))?;
    if let Some(b) = beam {
        update_key_f64(&mut out, "BMAJ", b.major_deg)?;
        update_key_f64(&mut out, "BMIN", b.minor_deg)?;
        update_key_f64(&mut out, "BPA", b.pa_deg)?;
    }
    if cube.casambm {
        delete_key(&mut out, "CASAMBM")?;
    }
    disable_scaling(&mut out)?;
    write_pixels(&mut out, 0, &plane)?;
    let mut status = 0;
    // SAFETY: `out` is open; flushing reports write errors that close hides.
    unsafe { sys::ffflus(out.as_raw(), &mut status) };
    check_status(status)?;
    Ok(())
}

fn same_file(a: &Path, b: &Path) -> bool {
    matches!((a.canonicalize(), b.canonicalize()), (Ok(a), Ok(b)) if a == b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_path_default() {
        let p = default_output_path(Path::new("/data/cube.fits"), "channel", 5);
        assert_eq!(p, PathBuf::from("/data/cube.channel-5.fits"));
    }

    #[test]
    fn both_indices_errors() {
        let opts = ExtractOptions {
            channel_index: Some(1),
            time_index: Some(2),
            ..Default::default()
        };
        assert!(create_target_index(&opts).is_err());
    }

    #[test]
    fn neither_index_errors() {
        let opts = ExtractOptions::default();
        assert!(create_target_index(&opts).is_err());
    }
}
