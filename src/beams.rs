//! Per-channel restoring-beam handling.
//!
//! Port of the beam logic in `fitscube.combine_fits`: read BMAJ/BMIN/BPA (all in
//! degrees) from each input header, decide whether they are all the same, and —
//! when they vary — emit a CASA-style `BEAMS` binary-table extension.
use std::path::{Path, PathBuf};

use fitsio::FitsFile;
use fitsio::tables::{ColumnDataType, ColumnDescription};

use crate::error::Result;
use crate::fits_io::{find_target_axis, read_key_f64};

/// A restoring beam in degrees. Any field may be NaN (no beam in header).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Beam {
    pub major_deg: f64,
    pub minor_deg: f64,
    pub pa_deg: f64,
}

impl Beam {
    pub fn is_nan(&self) -> bool {
        self.major_deg.is_nan() || self.minor_deg.is_nan() || self.pa_deg.is_nan()
    }
}

/// Read BMAJ/BMIN/BPA (degrees) from a header. Missing → a NaN beam, matching
/// the `NoBeamException` fallback in the original.
pub fn read_beam(path: &Path) -> Result<Beam> {
    let major_deg = read_key_f64(path, "BMAJ")?.unwrap_or(f64::NAN);
    let minor_deg = read_key_f64(path, "BMIN")?.unwrap_or(f64::NAN);
    let pa_deg = read_key_f64(path, "BPA")?.unwrap_or(f64::NAN);
    Ok(Beam {
        major_deg,
        minor_deg,
        pa_deg,
    })
}

/// Read one beam per file.
pub fn parse_beams(file_list: &[PathBuf]) -> Result<Vec<Beam>> {
    file_list.iter().map(|p| read_beam(p)).collect()
}

/// `numpy.isclose` with the default tolerances (`rtol = 1e-5`, `atol = 1e-8`).
/// NaNs are never close (matching numpy with `equal_nan=False`).
fn isclose(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
}

/// Whether every beam matches the first on major, minor and PA.
///
/// Mirrors the `single_beam` test in the original (compares shape, not area).
/// If beams are all-NaN this returns `false`, since NaN is never close to NaN.
pub fn is_single_beam(beams: &[Beam]) -> bool {
    if beams.is_empty() {
        return false;
    }
    let first = beams[0];
    beams.iter().all(|b| {
        isclose(first.major_deg, b.major_deg)
            && isclose(first.minor_deg, b.minor_deg)
            && isclose(first.pa_deg, b.pa_deg)
    })
}

/// FITS polarisation index: `CRPIX − 1` of the STOKES axis, or 0 if absent.
/// Mirrors `get_polarisation`.
pub fn get_polarisation(path: &Path) -> Result<i32> {
    match find_target_axis(path, "STOKES") {
        Ok(axis) => Ok((axis.crpix - 1.0) as i32),
        Err(_) => Ok(0),
    }
}

/// Append a CASA `BEAMS` binary-table extension to an open output cube.
///
/// Columns: BMAJ/BMIN [arcsec], BPA [deg], CHAN, POL. NaN beam values are
/// replaced with `f32::MIN_POSITIVE` (numpy `finfo(float32).tiny`) to keep CASA
/// happy, exactly as `make_beam_table` does.
pub fn write_beam_table(fptr: &mut FitsFile, beams: &[Beam], pol: i32) -> Result<()> {
    let tiny = f32::MIN_POSITIVE;
    let nchan = beams.len();

    let nan_to_tiny = |v: f32| if v.is_nan() { tiny } else { v };

    let bmaj: Vec<f32> = beams
        .iter()
        .map(|b| nan_to_tiny((b.major_deg * 3600.0) as f32))
        .collect();
    let bmin: Vec<f32> = beams
        .iter()
        .map(|b| nan_to_tiny((b.minor_deg * 3600.0) as f32))
        .collect();
    let bpa: Vec<f32> = beams.iter().map(|b| nan_to_tiny(b.pa_deg as f32)).collect();
    let chan: Vec<i32> = (0..nchan as i32).collect();
    let pols: Vec<i32> = vec![pol; nchan];

    let col_bmaj = ColumnDescription::new("BMAJ")
        .with_type(ColumnDataType::Float)
        .create()?;
    let col_bmin = ColumnDescription::new("BMIN")
        .with_type(ColumnDataType::Float)
        .create()?;
    let col_bpa = ColumnDescription::new("BPA")
        .with_type(ColumnDataType::Float)
        .create()?;
    let col_chan = ColumnDescription::new("CHAN")
        .with_type(ColumnDataType::Int)
        .create()?;
    let col_pol = ColumnDescription::new("POL")
        .with_type(ColumnDataType::Int)
        .create()?;

    let table_hdu =
        fptr.create_table("BEAMS", &[col_bmaj, col_bmin, col_bpa, col_chan, col_pol])?;
    table_hdu.write_col(fptr, "BMAJ", &bmaj)?;
    table_hdu.write_col(fptr, "BMIN", &bmin)?;
    table_hdu.write_col(fptr, "BPA", &bpa)?;
    table_hdu.write_col(fptr, "CHAN", &chan)?;
    table_hdu.write_col(fptr, "POL", &pols)?;

    let beam_hdu = fptr.hdu("BEAMS")?;
    beam_hdu.write_key(fptr, "NCHAN", nchan as i64)?;
    beam_hdu.write_key(fptr, "NPOL", 1i64)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(maj: f64, min: f64, pa: f64) -> Beam {
        Beam {
            major_deg: maj,
            minor_deg: min,
            pa_deg: pa,
        }
    }

    #[test]
    fn identical_beams_are_single() {
        let beams = vec![b(1.0, 0.5, 10.0); 4];
        assert!(is_single_beam(&beams));
    }

    #[test]
    fn varying_beams_are_not_single() {
        let beams = vec![b(1.0, 0.5, 10.0), b(1.1, 0.5, 10.0)];
        assert!(!is_single_beam(&beams));
    }

    #[test]
    fn all_nan_is_not_single() {
        let beams = vec![b(f64::NAN, f64::NAN, f64::NAN); 3];
        assert!(!is_single_beam(&beams));
    }
}
