//! Pull planes back out of a cube.
//!
//! Only the requested plane is read. The output keeps the cube's axes, with the
//! FREQ/TIME axis of length 1, and takes its beam from the `BEAMS` table.
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use atfits_rs::raw::{Pixel, as_bytes, disable_scaling, read_pixels, swap_to_be};
use atfits_rs::{
    HeaderGeom, TargetAxis, delete_key, extract_header_layout, key_bool, update_key_f64,
    update_key_i64,
};
use fitsio::FitsFile;
use rayon::prelude::*;

use crate::beams::{Beam, read_beam_table};
use crate::combine::{check_exists, in_pool};
use crate::error::{FitsCubeError, Result};
use crate::header::plain_header;
use crate::partial::{partial_file, same_file};
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

/// `cube.fits` (or `cube.fits.gz`, `cube.fits.fz`) to `cube.channel-5.fits`.
fn default_output_path(input: &Path, output_name: &str, index: usize) -> PathBuf {
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    // Planes are written uncompressed, so name them as plain FITS
    let input = match input.extension().and_then(|e| e.to_str()) {
        Some("gz" | "fz") => Path::new(input.file_stem().unwrap_or_default()),
        _ => input,
    };
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

/// Extract every plane of `cube` into its own image and return their paths.
///
/// Planes are taken along the FREQ or TIME axis, whichever has more than one
/// plane. The cube and its beam table are read once, rather than once per plane.
pub fn extract_planes_from_cube(cube: &Path, opts: &ExtractPlanesOptions) -> Result<Vec<PathBuf>> {
    let (axis_name, output_name) = find_plane_axis(cube, opts.hdu_index)?;
    let (_, source) = Cube::open(cube, opts.hdu_index, axis_name)?;
    let n = source.n_planes();
    let out_paths = match &opts.output_paths {
        None => (0..n)
            .map(|i| default_output_path(cube, output_name, i))
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

/// The axis to split a cube along: FREQ or TIME, whichever has more than one
/// plane (a time cube made from images keeps their one-plane FREQ axis).
fn find_plane_axis(cube: &Path, hdu_index: usize) -> Result<(&'static str, &'static str)> {
    check_exists(cube)?;
    let mut fptr = FitsFile::open(cube)?;
    fptr.hdu(hdu_index)?;
    let dims = HeaderGeom::from_file(&mut fptr)?.dims;
    let mut found = Vec::new();
    for (axis_name, output_name) in [("FREQ", "channel"), ("TIME", "timestep")] {
        if let Some(axis) = find_axis(&mut fptr, axis_name)? {
            found.push((axis_name, output_name, dims[axis.fits_idx - 1]));
        }
    }
    let long: Vec<_> = found.iter().filter(|(_, _, n)| *n > 1).collect();
    match (long.as_slice(), found.first()) {
        ([(axis_name, output_name, _)], _) => Ok((axis_name, output_name)),
        ([], Some((axis_name, output_name, _))) => Ok((axis_name, output_name)),
        ([], None) => Err(FitsCubeError::TargetAxisMissing(
            "Did not find a FREQ or TIME axis".to_string(),
        )),
        _ => Err(FitsCubeError::InvalidSpec(
            "Both the FREQ and TIME axes have more than one plane, so cannot pick one".to_string(),
        )),
    }
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
        let partial = partial_file(out)?;
        with_pixel_type!(
            self.geom.bitpix,
            copy_plane(self, fptr, index, beam, &partial)
        )?;
        partial.persist(out).map_err(|e| e.error)?;
        Ok(())
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

    let mut dims = cube.geom.dims.clone();
    dims[fi - 1] = 1;
    let mut header = plain_header(src, cube.geom.bitpix, &dims)?;
    update_key_i64(&mut header, &format!("CRPIX{fi}"), 1)?;
    update_key_f64(&mut header, &format!("CRVAL{fi}"), world(&cube.axis, index))?;
    if let Some(b) = beam {
        update_key_f64(&mut header, "BMAJ", b.major_deg)?;
        update_key_f64(&mut header, "BMIN", b.minor_deg)?;
        update_key_f64(&mut header, "BPA", b.pa_deg)?;
    }
    if cube.casambm {
        delete_key(&mut header, "CASAMBM")?;
    }
    let layout = extract_header_layout(&mut header, &dims)?;

    swap_to_be(&mut plane);
    let data = as_bytes(&plane);
    let mut out = File::create(path)?;
    out.write_all(&layout.header)?;
    out.write_all(data)?;
    out.write_all(&vec![0u8; (2880 - data.len() % 2880) % 2880])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_path_default() {
        let p = default_output_path(Path::new("/data/cube.fits"), "channel", 5);
        assert_eq!(p, PathBuf::from("/data/cube.channel-5.fits"));
        for compressed in ["/data/cube.fits.gz", "/data/cube.fits.fz"] {
            let p = default_output_path(Path::new(compressed), "channel", 5);
            assert_eq!(p, PathBuf::from("/data/cube.channel-5.fits"));
        }
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
