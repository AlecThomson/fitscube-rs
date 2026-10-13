//! Read every input header once, and check that the inputs can share the cube
//! header, which is taken from the first of them.
use std::fmt::Debug;
use std::path::{Path, PathBuf};

use atfits_rs::{HeaderGeom, TargetAxis, key_exists, key_f64, key_i64, key_string};
use fitsio::FitsFile;
use rayon::prelude::*;

use crate::beams::Beam;
use crate::error::{FitsCubeError, Result};
use crate::specs::utc_to_mjdsec;
use crate::wcs::{find_axis, hz_per_unit, world};

/// What combine needs from the primary header of one input.
#[derive(Debug, Clone)]
pub struct InputHeader {
    /// NAXISn in FITS order.
    pub dims: Vec<usize>,
    /// CTYPEn, "" where absent.
    pub ctypes: Vec<String>,
    /// FITS Stokes code of each plane along the STOKES axis.
    pub stokes_codes: Option<Vec<i64>>,
    pub storage: Storage,
    /// `None` without BMAJ. BMIN and BPA are NaN when absent.
    pub beam: Option<Beam>,
    pub reffreq: Option<f64>,
    pub date_obs: Option<String>,
    pub has_cd: bool,
    pub has_pc: bool,
    pub freq_axis: Option<TargetAxis>,
}

/// BITPIX, and the BSCALE, BZERO and BLANK cards (`None` where absent).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Storage {
    pub bitpix: i64,
    pub bscale: Option<f64>,
    pub bzero: Option<f64>,
    pub blank: Option<i64>,
}

impl InputHeader {
    pub fn read(path: &Path) -> Result<Self> {
        let mut fptr = FitsFile::open(path)?;
        fptr.primary_hdu()?;
        let f = &mut fptr;
        let geom = HeaderGeom::from_file(f)?;
        if geom.naxis < 2 {
            return Err(FitsCubeError::UnsupportedNaxis(geom.naxis as i64));
        }
        let ctypes = (1..=geom.naxis)
            .map(|n| {
                let ctype = key_string(f, &format!("CTYPE{n}"))?;
                Ok(ctype.map(|c| c.trim().to_string()).unwrap_or_default())
            })
            .collect::<Result<Vec<_>>>()?;

        let stokes_codes = match ctypes.iter().position(|c| c == "STOKES") {
            None => None,
            Some(i) => {
                let n = i + 1;
                let crval = key_f64(f, &format!("CRVAL{n}"))?.unwrap_or(0.0);
                let cdelt = key_f64(f, &format!("CDELT{n}"))?.unwrap_or(1.0);
                let crpix = key_f64(f, &format!("CRPIX{n}"))?.unwrap_or(0.0);
                let codes = (1..=geom.dims[i])
                    .map(|pix| (crval + cdelt * (pix as f64 - crpix)).round() as i64)
                    .collect();
                Some(codes)
            }
        };

        let storage = Storage {
            bitpix: geom.bitpix,
            bscale: key_f64(f, "BSCALE")?,
            bzero: key_f64(f, "BZERO")?,
            blank: key_i64(f, "BLANK")?,
        };
        let beam = match key_f64(f, "BMAJ")? {
            None => None,
            Some(major_deg) => Some(Beam {
                major_deg,
                minor_deg: key_f64(f, "BMIN")?.unwrap_or(f64::NAN),
                pa_deg: key_f64(f, "BPA")?.unwrap_or(f64::NAN),
            }),
        };
        Ok(Self {
            stokes_codes,
            storage,
            beam,
            reffreq: key_f64(f, "REFFREQ")?,
            date_obs: key_string(f, "DATE-OBS")?,
            has_cd: key_exists(f, "CD1_1")?,
            has_pc: key_exists(f, "PC1_1")?,
            freq_axis: find_axis(f, "FREQ")?,
            dims: geom.dims,
            ctypes,
        })
    }

    /// (NAXIS1, NAXIS2)
    pub fn shape(&self) -> (usize, usize) {
        (self.dims[0], self.dims[1])
    }

    pub fn is_2d(&self) -> bool {
        self.dims.len() == 2
    }

    /// Frequency (Hz) or time (MJD seconds) of the image at `path`.
    ///
    /// Time comes from DATE-OBS. Frequency comes from REFFREQ for a 2D image,
    /// otherwise from the first pixel of the FREQ axis.
    pub fn spec(&self, path: &Path, time_domain_mode: bool) -> Result<f64> {
        if time_domain_mode {
            let date_obs = self.date_obs.as_deref().ok_or_else(|| {
                FitsCubeError::MissingKeyword(format!(
                    "DATE-OBS not in header of {} (needed for time-domain mode)",
                    path.display()
                ))
            })?;
            return utc_to_mjdsec(date_obs);
        }
        if self.is_2d() {
            return self.reffreq.ok_or_else(|| {
                FitsCubeError::MissingKeyword(format!(
                    "REFFREQ not in header of {}. Cannot combine 2D images without \
                     frequency information.",
                    path.display()
                ))
            });
        }
        let axis = self.freq_axis.as_ref().ok_or_else(|| {
            FitsCubeError::TargetAxisMissing(format!(
                "No FREQ axis found in WCS of {}. Cannot combine N-D images without \
                 frequency information.",
                path.display()
            ))
        })?;
        let scale = hz_per_unit(axis)
            .map_err(|e| FitsCubeError::InvalidSpec(format!("{}: {e}", path.display())))?;
        Ok(world(axis, 0) * scale)
    }
}

/// [`InputHeader`] of every input, in `file_list` order, read in parallel.
pub fn read_input_headers(file_list: &[PathBuf]) -> Result<Vec<InputHeader>> {
    file_list.par_iter().map(|p| InputHeader::read(p)).collect()
}

/// FITS Stokes symbols (WCS paper III, table 7), indexed by `code + 8`.
const STOKES_SYMBOLS: [&str; 13] = [
    "YX", "XY", "YY", "XX", "LR", "RL", "LL", "RR", "", "I", "Q", "U", "V",
];

fn stokes_symbol(code: i64) -> String {
    usize::try_from(code + 8)
        .ok()
        .and_then(|i| STOKES_SYMBOLS.get(i))
        .filter(|s| !s.is_empty())
        .map_or_else(|| format!("?{code}"), |s| (*s).to_string())
}

/// Check that every input has the same value as the first, and return it.
/// `error` wraps the message listing the inputs that differ.
pub fn check_values_match<T: PartialEq + Clone>(
    file_list: &[PathBuf],
    values: &[T],
    describe: impl Fn(&T) -> String,
    requirement: &str,
    error: impl FnOnce(String) -> FitsCubeError,
) -> Result<T> {
    let expected = &values[0];
    let offenders: Vec<String> = file_list
        .iter()
        .zip(values)
        .filter(|(_, value)| *value != expected)
        .map(|(path, value)| format!("{} has {}", path.display(), describe(value)))
        .collect();
    if !offenders.is_empty() {
        const MAX_SHOWN: usize = 10;
        let mut shown: Vec<String> = offenders.iter().take(MAX_SHOWN).cloned().collect();
        if offenders.len() > shown.len() {
            shown.push(format!("...and {} more", offenders.len() - shown.len()));
        }
        return Err(error(format!(
            "All input images must share {requirement}. Expected {} from {}, but found:\n{}",
            describe(expected),
            file_list[0].display(),
            shown.join("\n")
        )));
    }
    Ok(expected.clone())
}

fn describe_list<T: Debug>(values: &[T]) -> String {
    format!("{values:?}")
}

/// Check that every input has the same NAXIS1 and NAXIS2, and return them.
///
/// Planes go into the cube at a fixed offset per channel, so a differing grid
/// would shift every later plane.
pub fn check_matching_shapes(
    file_list: &[PathBuf],
    headers: &[InputHeader],
) -> Result<(usize, usize)> {
    let shapes: Vec<(usize, usize)> = headers.iter().map(InputHeader::shape).collect();
    check_values_match(
        file_list,
        &shapes,
        |shape| format!("(NAXIS1, NAXIS2)={shape:?}"),
        "the same pixel grid",
        FitsCubeError::ShapeMismatch,
    )
}

/// Check that every input has the same axes and Stokes parameter.
pub fn check_matching_axes(file_list: &[PathBuf], headers: &[InputHeader]) -> Result<()> {
    let ctypes: Vec<Vec<String>> = headers.iter().map(|h| h.ctypes.clone()).collect();
    check_values_match(
        file_list,
        &ctypes,
        |ctypes| format!("axes {}", describe_list(ctypes)),
        "the same axes, in the same order",
        FitsCubeError::AxisMismatch,
    )?;
    let stokes: Vec<Option<Vec<i64>>> = headers.iter().map(|h| h.stokes_codes.clone()).collect();
    check_values_match(
        file_list,
        &stokes,
        |codes| match codes {
            None => "no Stokes axis".to_string(),
            Some(codes) => {
                let symbols: Vec<String> = codes.iter().map(|&c| stokes_symbol(c)).collect();
                format!(
                    "Stokes {} (codes {})",
                    describe_list(&symbols),
                    describe_list(codes)
                )
            }
        },
        "the same Stokes parameter",
        FitsCubeError::StokesMismatch,
    )?;
    Ok(())
}

/// Check that every input stores pixels with the same BITPIX, BSCALE, BZERO
/// and BLANK. An integer cube copies the stored values under one set of them.
pub fn check_matching_storage(file_list: &[PathBuf], headers: &[InputHeader]) -> Result<()> {
    let storage: Vec<Storage> = headers.iter().map(|h| h.storage).collect();
    check_values_match(
        file_list,
        &storage,
        |s| format!("{s:?}"),
        "the same BITPIX and integer scaling for an integer cube (pass float_length=32 or 64, \
         --floating, to write decoded values instead)",
        FitsCubeError::ScalingMismatch,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stokes_symbols_follow_fits_codes() {
        assert_eq!(stokes_symbol(1), "I");
        assert_eq!(stokes_symbol(4), "V");
        assert_eq!(stokes_symbol(-1), "RR");
        assert_eq!(stokes_symbol(-5), "XX");
        assert_eq!(stokes_symbol(-8), "YX");
        assert_eq!(stokes_symbol(0), "?0");
        assert_eq!(stokes_symbol(9), "?9");
    }

    #[test]
    fn values_match_lists_offenders() {
        let files: Vec<PathBuf> = (0..13).map(|i| PathBuf::from(format!("f{i}"))).collect();
        let mut values = vec![1; 13];
        let check = |values: &[i32]| {
            check_values_match(
                &files,
                values,
                |v| v.to_string(),
                "x",
                FitsCubeError::InvalidSpec,
            )
        };
        assert_eq!(check(&values).unwrap(), 1);

        for v in values.iter_mut().skip(1) {
            *v = 2;
        }
        let err = check(&values).unwrap_err().to_string();
        assert!(err.contains("Expected 1 from f0"), "{err}");
        assert!(err.contains("f10 has 2"), "{err}");
        assert!(!err.contains("f11 has 2"), "{err}");
        assert!(err.contains("...and 2 more"), "{err}");
    }
}
