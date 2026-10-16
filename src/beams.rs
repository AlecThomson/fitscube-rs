//! Restoring beams, and the CASA `BEAMS` table that holds them when they vary
//! across the cube.
use fitsio::FitsFile;
use fitsio::tables::{ColumnDataType, ColumnDescription};

use crate::checks::InputHeader;
use crate::error::{FitsCubeError, Result};

/// A restoring beam in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Beam {
    pub major_deg: f64,
    pub minor_deg: f64,
    pub pa_deg: f64,
}

impl Beam {
    /// No beam: a blank channel, or an input without BMAJ.
    pub const NAN: Beam = Beam {
        major_deg: f64::NAN,
        minor_deg: f64::NAN,
        pa_deg: f64::NAN,
    };

    pub fn is_nan(&self) -> bool {
        self.major_deg.is_nan() || self.minor_deg.is_nan() || self.pa_deg.is_nan()
    }
}

/// One beam per input, NaN for inputs without one, or `None` if no input has
/// a beam.
pub fn input_beams(headers: &[InputHeader]) -> Option<Vec<Beam>> {
    headers.iter().any(|h| h.beam.is_some()).then(|| {
        headers
            .iter()
            .map(|h| h.beam.unwrap_or(Beam::NAN))
            .collect()
    })
}

/// Spread one beam per input over the cube channels, NaN for `missing` ones.
pub fn expand_beams(beams: &[Beam], missing: &[bool]) -> Result<Vec<Beam>> {
    let populated = missing.iter().filter(|&&m| !m).count();
    if populated != beams.len() {
        return Err(FitsCubeError::ChannelMissing(format!(
            "Have {} beams for the {populated} populated channels of {}",
            beams.len(),
            missing.len()
        )));
    }
    let mut present = beams.iter();
    Ok(missing
        .iter()
        .map(|&m| match m {
            true => Beam::NAN,
            false => *present.next().unwrap_or(&Beam::NAN),
        })
        .collect())
}

/// Channels whose beam has a zero major or minor axis.
///
/// wsclean writes such a beam for a plane without a fitted PSF, such as a
/// `-fit-spectral-pol` model image. A zero position angle is fine.
pub fn find_zero_beams(beams: &[Beam]) -> Vec<bool> {
    beams
        .iter()
        .map(|b| b.major_deg == 0.0 || b.minor_deg == 0.0)
        .collect()
}

/// `beams` with the flagged ones replaced by [`Beam::NAN`].
pub fn nan_zero_beams(beams: &[Beam], zero_beam: &[bool]) -> Vec<Beam> {
    beams
        .iter()
        .zip(zero_beam)
        .map(|(&b, &zero)| if zero { Beam::NAN } else { b })
        .collect()
}

/// `numpy.isclose` with its default tolerances. NaN is never close.
fn isclose(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
}

/// Whether every beam matches the first. False when the beams are NaN.
pub fn is_single_beam(beams: &[Beam]) -> bool {
    let Some(first) = beams.first() else {
        return false;
    };
    beams.iter().all(|b| {
        isclose(first.major_deg, b.major_deg)
            && isclose(first.minor_deg, b.minor_deg)
            && isclose(first.pa_deg, b.pa_deg)
    })
}

/// 0-based plane indices along the Stokes axis, the POL values of the beam
/// table. No Stokes axis counts as one plane.
pub fn get_polarisation(stokes_len: Option<usize>) -> Vec<i32> {
    (0..stokes_len.unwrap_or(1) as i32).collect()
}

/// The POL value of every beam table row. Only single-Stokes cubes are
/// supported.
pub fn beam_table_pol(stokes_idx: &[i32]) -> Result<i32> {
    match stokes_idx {
        [pol] => Ok(*pol),
        _ => Err(FitsCubeError::NotImplemented(format!(
            "Only single-Stokes cubes are supported - found {} Stokes planes",
            stokes_idx.len()
        ))),
    }
}

/// Append a CASA `BEAMS` table (BMAJ, BMIN in arcsec, BPA in deg, CHAN, POL) to
/// an open cube.
///
/// CASA cannot read NaN beams, so NaN and zero-sized beams are written as
/// `f32::MIN_POSITIVE` (numpy's `finfo(float32).tiny`).
pub fn write_beam_table(fptr: &mut FitsFile, beams: &[Beam], stokes_idx: &[i32]) -> Result<()> {
    let pol = beam_table_pol(stokes_idx)?;
    let nchan = beams.len();
    let beams = nan_zero_beams(beams, &find_zero_beams(beams));
    let column = |value: fn(&Beam) -> f64| -> Vec<f32> {
        beams
            .iter()
            .map(|b| match value(b) as f32 {
                v if v.is_nan() => f32::MIN_POSITIVE,
                v => v,
            })
            .collect()
    };
    let bmaj = column(|b| b.major_deg * 3600.0);
    let bmin = column(|b| b.minor_deg * 3600.0);
    let bpa = column(|b| b.pa_deg);
    let chan: Vec<i32> = (0..nchan as i32).collect();
    let pols: Vec<i32> = vec![pol; nchan];

    let col = |name: &str, typ| ColumnDescription::new(name).with_type(typ).create();
    let columns = [
        col("BMAJ", ColumnDataType::Float)?,
        col("BMIN", ColumnDataType::Float)?,
        col("BPA", ColumnDataType::Float)?,
        col("CHAN", ColumnDataType::Int)?,
        col("POL", ColumnDataType::Int)?,
    ];
    let hdu = fptr.create_table("BEAMS", &columns)?;
    hdu.write_col(fptr, "BMAJ", &bmaj)?;
    hdu.write_col(fptr, "BMIN", &bmin)?;
    hdu.write_col(fptr, "BPA", &bpa)?;
    hdu.write_col(fptr, "CHAN", &chan)?;
    hdu.write_col(fptr, "POL", &pols)?;
    hdu.write_key(fptr, "NCHAN", nchan as i64)?;
    hdu.write_key(fptr, "NPOL", stokes_idx.len() as i64)?;
    // casacore reads the beam units from TUNITn
    for (col, unit) in [(1, "arcsec"), (2, "arcsec"), (3, "deg")] {
        hdu.write_key(fptr, &format!("TUNIT{col}"), unit)?;
    }
    Ok(())
}

/// The beams in the `BEAMS` table of an open file, or `None` if it has none.
pub fn read_beam_table(fptr: &mut FitsFile) -> Result<Option<Vec<Beam>>> {
    let hdu = match fptr.hdu("BEAMS") {
        Ok(hdu) => hdu,
        Err(fitsio::errors::Error::Fits(e)) if e.status == fitsio::sys::BAD_HDU_NUM as i32 => {
            return Ok(None);
        }
        Err(e) => return Err(e.into()),
    };
    let bmaj: Vec<f32> = hdu.read_col(fptr, "BMAJ")?;
    let bmin: Vec<f32> = hdu.read_col(fptr, "BMIN")?;
    let bpa: Vec<f32> = hdu.read_col(fptr, "BPA")?;
    let beams = bmaj
        .iter()
        .zip(&bmin)
        .zip(&bpa)
        .map(|((&maj, &min), &pa)| Beam {
            major_deg: maj as f64 / 3600.0,
            minor_deg: min as f64 / 3600.0,
            pa_deg: pa as f64,
        })
        .collect();
    Ok(Some(beams))
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
    fn polarisation_is_axis_index() {
        assert_eq!(get_polarisation(None), vec![0]);
        assert_eq!(get_polarisation(Some(1)), vec![0]);
        assert_eq!(get_polarisation(Some(3)), vec![0, 1, 2]);
    }

    #[test]
    fn beam_table_rejects_multi_stokes() {
        assert_eq!(beam_table_pol(&get_polarisation(Some(1))).unwrap(), 0);
        let err = beam_table_pol(&get_polarisation(Some(3))).unwrap_err();
        assert!(matches!(err, FitsCubeError::NotImplemented(_)), "{err:?}");
    }

    #[test]
    fn expand_beams_blanks_missing_channels() {
        let beams = vec![b(1.0, 1.0, 0.0), b(2.0, 2.0, 0.0)];
        let expanded = expand_beams(&beams, &[false, true, false]).unwrap();
        assert_eq!(expanded[0], beams[0]);
        assert!(expanded[1].is_nan());
        assert_eq!(expanded[2], beams[1]);
    }

    #[test]
    fn expand_beams_checks_count() {
        let beams = vec![b(1.0, 1.0, 0.0)];
        let err = expand_beams(&beams, &[false, true, false]).unwrap_err();
        assert!(matches!(err, FitsCubeError::ChannelMissing(_)), "{err:?}");
    }

    #[test]
    fn find_zero_beams_ignores_nan_and_zero_pa() {
        let beams = vec![
            b(1.0, 1.0, 0.0),
            b(0.0, 0.0, 0.0),
            b(2.0, 0.0, 0.0),
            Beam::NAN,
        ];
        assert_eq!(find_zero_beams(&beams), vec![false, true, true, false]);
    }

    #[test]
    fn nan_zero_beams_blanks_only_flagged() {
        let beams = vec![b(1.0, 1.0, 10.0), b(0.0, 0.0, 0.0)];
        let blanked = nan_zero_beams(&beams, &[false, true]);
        assert_eq!(blanked[0], beams[0]);
        assert!(blanked[1].major_deg.is_nan());
        assert!(blanked[1].minor_deg.is_nan());
        assert!(blanked[1].pa_deg.is_nan());
    }

    #[test]
    fn all_nan_is_not_single() {
        let beams = vec![Beam::NAN; 3];
        assert!(!is_single_beam(&beams));
    }
}
