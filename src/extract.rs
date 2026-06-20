//! Extract a single plane (channel or timestep) from a FITS cube.
//!
//! Port of `fitscube.extract`. Finds the FREQ/TIME axis, slices the requested
//! index out of the cube (keeping the axis with length 1, as `np.take` +
//! `np.expand_dims` do), rewrites the axis WCS for the extracted plane, copies
//! the per-channel beam from the `BEAMS` table when present, and writes a new
//! FITS image.
use std::path::{Path, PathBuf};

use fitsio::FitsFile;
use ndarray::{ArrayD, Axis, IxDyn};

use crate::error::{FitsCubeError, Result};
use crate::fits_io::{
    PixelType, copy_header_only, delete_key, resize_image, update_key_f64, update_key_i64,
    update_key_str,
};

/// What to pull out of the cube.
#[derive(Debug, Clone, Default)]
pub struct ExtractOptions {
    pub hdu_index: usize,
    pub channel_index: Option<usize>,
    pub time_index: Option<usize>,
    pub overwrite: bool,
    pub output_path: Option<PathBuf>,
}

/// The resolved extraction target.
struct TargetIndex {
    axis_name: &'static str,
    axis_index: usize,
    output_name: &'static str,
}

/// Resolve the target axis/index, enforcing exactly one of channel/time.
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

/// Build the default output path: `cube.fits` → `cube.channel-5.fits`. Mirrors
/// `get_output_path`.
fn default_output_path(input: &Path, target: &TargetIndex) -> PathBuf {
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    let stem = input.file_stem().unwrap_or_default().to_string_lossy();
    let ext = input.extension().unwrap_or_default().to_string_lossy();
    parent.join(format!(
        "{stem}.{}-{}.{ext}",
        target.output_name, target.axis_index
    ))
}

/// FREQ/TIME axis location within a cube header (read from a given HDU).
struct AxisWcs {
    fits_idx: usize,
    ctype: String,
    crval: f64,
    cdelt: f64,
    cunit: Option<String>,
}

fn find_axis(fptr: &mut FitsFile, hdu_index: usize, name: &str) -> Result<AxisWcs> {
    let hdu = fptr.hdu(hdu_index)?;
    let naxis: i64 = hdu.read_key(fptr, "NAXIS")?;
    for axis in 1..=naxis {
        let ctype: String = match hdu.read_key(fptr, &format!("CTYPE{axis}")) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if ctype.contains(name) {
            let crval: f64 = hdu.read_key(fptr, &format!("CRVAL{axis}")).unwrap_or(0.0);
            let cdelt: f64 = hdu.read_key(fptr, &format!("CDELT{axis}")).unwrap_or(1.0);
            let cunit: Option<String> = hdu.read_key(fptr, &format!("CUNIT{axis}")).ok();
            return Ok(AxisWcs {
                fits_idx: axis as usize,
                ctype,
                crval,
                cdelt,
                cunit,
            });
        }
    }
    Err(FitsCubeError::TargetAxisMissing(format!(
        "Did not find the {name} axis"
    )))
}

/// Does the header advertise a CASA beam table (`CASAMBM = T`)?
fn contains_beam_table(fptr: &mut FitsFile, hdu_index: usize) -> bool {
    let Ok(hdu) = fptr.hdu(hdu_index) else {
        return false;
    };
    if let Ok(b) = hdu.read_key::<bool>(fptr, "CASAMBM") {
        return b;
    }
    if let Ok(s) = hdu.read_key::<String>(fptr, "CASAMBM") {
        return matches!(s.trim(), "T" | "TRUE");
    }
    false
}

/// Read the dims (FITS order) and BITPIX of the chosen HDU.
fn read_geom(fptr: &mut FitsFile, hdu_index: usize) -> Result<(Vec<usize>, i64)> {
    let hdu = fptr.hdu(hdu_index)?;
    let naxis: i64 = hdu.read_key(fptr, "NAXIS")?;
    let bitpix: i64 = hdu.read_key(fptr, "BITPIX")?;
    let mut dims = Vec::with_capacity(naxis as usize);
    for i in 1..=naxis {
        let n: i64 = hdu.read_key(fptr, &format!("NAXIS{i}"))?;
        dims.push(n as usize);
    }
    Ok((dims, bitpix))
}

/// Beam (degrees) for a channel, read from the `BEAMS` binary table.
struct BeamRow {
    bmaj_deg: f64,
    bmin_deg: f64,
    bpa_deg: f64,
}

fn extract_beam_row(path: &Path, index: usize) -> Result<BeamRow> {
    let mut fptr = FitsFile::open(path.to_string_lossy().as_ref())?;
    let hdu = fptr
        .hdu("BEAMS")
        .map_err(|_| FitsCubeError::Other("Beam table was not found".to_string()))?;
    let bmaj: Vec<f32> = hdu.read_col(&mut fptr, "BMAJ")?;
    let bmin: Vec<f32> = hdu.read_col(&mut fptr, "BMIN")?;
    let bpa: Vec<f32> = hdu.read_col(&mut fptr, "BPA")?;
    if index >= bmaj.len() {
        return Err(FitsCubeError::ChannelMissing(format!(
            "beam table has {} rows, requested index {index}",
            bmaj.len()
        )));
    }
    // Table stores BMAJ/BMIN in arcsec, BPA in deg.
    Ok(BeamRow {
        bmaj_deg: bmaj[index] as f64 / 3600.0,
        bmin_deg: bmin[index] as f64 / 3600.0,
        bpa_deg: bpa[index] as f64,
    })
}

/// Extract a plane buffer at `axis_index` along the cube's `array_idx`
/// (numpy-order) axis, keeping that axis with length 1.
fn take_plane<T: Clone + 'static>(
    flat: Vec<T>,
    dims_fits: &[usize],
    array_idx: usize,
    axis_index: usize,
) -> Result<Vec<T>> {
    // numpy (C-order) shape is the reverse of the FITS dim order.
    let shape_np: Vec<usize> = dims_fits.iter().rev().copied().collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&shape_np), flat)?;
    let plane = arr.index_axis(Axis(array_idx), axis_index); // axis removed
    let plane = plane.insert_axis(Axis(array_idx)); // restore length-1 axis
    let std = plane.as_standard_layout();
    Ok(std.iter().cloned().collect())
}

#[allow(clippy::too_many_arguments)]
fn write_plane<T: CubeImage>(
    out_path: &Path,
    cube: &Path,
    hdu_index: usize,
    dims_out_fits: &[usize],
    bitpix: i64,
    plane: &[T],
    axis: &AxisWcs,
    axis_index: usize,
    beam: Option<&BeamRow>,
    drop_casambm: bool,
    overwrite: bool,
) -> Result<()> {
    if out_path.exists() {
        if overwrite {
            std::fs::remove_file(out_path)?;
        } else {
            return Err(FitsCubeError::OutputExists(out_path.to_path_buf()));
        }
    }
    // Copy the (chosen-HDU) header, resize to the plane shape, rewrite the axis.
    copy_header_only_hdu(cube, out_path, hdu_index)?;
    let mut fptr = FitsFile::edit(out_path.to_string_lossy().as_ref())?;
    fptr.primary_hdu()?;
    resize_image(&mut fptr, bitpix, dims_out_fits)?;

    let fi = axis.fits_idx;
    let new_crval = axis.crval + (axis_index as f64) * axis.cdelt;
    update_key_str(&mut fptr, &format!("CTYPE{fi}"), &axis.ctype)?;
    update_key_i64(&mut fptr, &format!("CRPIX{fi}"), 1)?;
    update_key_f64(&mut fptr, &format!("CRVAL{fi}"), new_crval)?;
    update_key_f64(&mut fptr, &format!("CDELT{fi}"), axis.cdelt)?;
    if let Some(cunit) = &axis.cunit {
        update_key_str(&mut fptr, &format!("CUNIT{fi}"), cunit)?;
    }

    if let Some(b) = beam {
        update_key_f64(&mut fptr, "BMAJ", b.bmaj_deg)?;
        update_key_f64(&mut fptr, "BMIN", b.bmin_deg)?;
        update_key_f64(&mut fptr, "BPA", b.bpa_deg)?;
    }
    if drop_casambm {
        delete_key(&mut fptr, "CASAMBM")?;
    }

    T::write_primary(&mut fptr, plane)?;
    Ok(())
}

/// Copy only the header of a specific input HDU into a fresh primary HDU.
fn copy_header_only_hdu(input: &Path, output: &Path, hdu_index: usize) -> Result<()> {
    if hdu_index == 0 {
        return copy_header_only(input, output);
    }
    // For non-primary HDUs, fall back to copying via the safe API path: open the
    // HDU, then reuse the primary-copy machinery is not applicable, so error
    // clearly (cubes are stored in the primary HDU in practice).
    Err(FitsCubeError::Other(format!(
        "extracting from non-primary HDU index {hdu_index} is not supported"
    )))
}

/// Minimal write helper so extract can be generic over pixel precision.
trait CubeImage: Clone + ndarray::ScalarOperand + 'static {
    fn read_primary(fptr: &mut FitsFile, hdu_index: usize) -> Result<Vec<Self>>;
    fn write_primary(fptr: &mut FitsFile, data: &[Self]) -> Result<()>;
}

macro_rules! impl_cube_image {
    ($t:ty) => {
        impl CubeImage for $t {
            fn read_primary(fptr: &mut FitsFile, hdu_index: usize) -> Result<Vec<Self>> {
                let hdu = fptr.hdu(hdu_index)?;
                Ok(hdu.read_image(fptr)?)
            }
            fn write_primary(fptr: &mut FitsFile, data: &[Self]) -> Result<()> {
                let hdu = fptr.primary_hdu()?;
                hdu.write_image(fptr, data)?;
                Ok(())
            }
        }
    };
}
impl_cube_image!(f32);
impl_cube_image!(f64);

#[allow(clippy::too_many_arguments)]
fn extract_typed<T: CubeImage>(
    cube: &Path,
    opts: &ExtractOptions,
    target: &TargetIndex,
    out_path: &Path,
    dims: &[usize],
    bitpix: i64,
    axis: AxisWcs,
    beam: Option<BeamRow>,
    drop_casambm: bool,
) -> Result<()> {
    let array_idx = dims.len() - axis.fits_idx; // numpy axis = len - fits_idx
    if target.axis_index >= dims[array_idx] {
        return Err(FitsCubeError::ChannelMissing(format!(
            "index {} outside cube axis of length {}",
            target.axis_index, dims[array_idx]
        )));
    }

    let mut fptr = FitsFile::open(cube.to_string_lossy().as_ref())?;
    let flat: Vec<T> = T::read_primary(&mut fptr, opts.hdu_index)?;
    drop(fptr);

    let plane = take_plane(flat, dims, array_idx, target.axis_index)?;

    let mut dims_out = dims.to_vec();
    dims_out[axis.fits_idx - 1] = 1;

    write_plane::<T>(
        out_path,
        cube,
        opts.hdu_index,
        &dims_out,
        bitpix,
        &plane,
        &axis,
        target.axis_index,
        beam.as_ref(),
        drop_casambm,
        opts.overwrite,
    )
}

/// Extract a plane from `cube`, writing a new FITS image and returning its path.
pub fn extract_plane_from_cube(cube: &Path, opts: &ExtractOptions) -> Result<PathBuf> {
    let target = create_target_index(opts)?;
    let out_path = opts
        .output_path
        .clone()
        .unwrap_or_else(|| default_output_path(cube, &target));

    let mut fptr = FitsFile::open(cube.to_string_lossy().as_ref())?;
    let axis = find_axis(&mut fptr, opts.hdu_index, target.axis_name)?;
    let (dims, bitpix) = read_geom(&mut fptr, opts.hdu_index)?;
    let has_beam_table = contains_beam_table(&mut fptr, opts.hdu_index);
    drop(fptr);

    let beam = if has_beam_table {
        extract_beam_row(cube, target.axis_index).ok()
    } else {
        None
    };
    let drop_casambm = has_beam_table;

    match PixelType::from_bitpix(bitpix) {
        PixelType::F32 => extract_typed::<f32>(
            cube,
            opts,
            &target,
            &out_path,
            &dims,
            bitpix,
            axis,
            beam,
            drop_casambm,
        )?,
        PixelType::F64 => extract_typed::<f64>(
            cube,
            opts,
            &target,
            &out_path,
            &dims,
            bitpix,
            axis,
            beam,
            drop_casambm,
        )?,
    }
    Ok(out_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_path_default() {
        let target = TargetIndex {
            axis_name: "FREQ",
            axis_index: 5,
            output_name: "channel",
        };
        let p = default_output_path(Path::new("/data/cube.fits"), &target);
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

    #[test]
    fn take_middle_plane_of_cube() {
        // FITS dims [nx=2, ny=2, nchan=3]; numpy shape [3,2,2].
        // Channel values: chan c filled with c.
        let dims_fits = vec![2usize, 2, 3];
        let mut flat = vec![0.0f64; 12];
        // numpy C-order index = ((c)*2 + r)*2 + col, with shape [3,2,2].
        for c in 0..3 {
            for r in 0..2 {
                for col in 0..2 {
                    flat[(c * 2 + r) * 2 + col] = c as f64;
                }
            }
        }
        // FREQ axis is fits_idx 3 → array_idx = 3 - 3 = 0.
        let plane = take_plane(flat, &dims_fits, 0, 1).unwrap();
        assert_eq!(plane, vec![1.0; 4]);
    }
}
