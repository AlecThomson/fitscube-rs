//! Header-only checks that every input image can share one cube header.
//!
//! Port of `check_matching_shapes` / `check_matching_axes` in
//! `fitscube.combine_fits`. The cube's header is taken from the first input, so
//! an input with another pixel grid, other axes (e.g. RA/DEC swapped, or an
//! extra axis) or another Stokes parameter would be silently mislabelled.
use std::fmt::Debug;
use std::path::{Path, PathBuf};

use fitsio::FitsFile;
use rayon::prelude::*;

use crate::error::{FitsCubeError, Result};

/// The header cards of one input that must match across all inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct InputAxes {
    /// (NAXIS1, NAXIS2)
    pub shape: (i64, i64),
    /// One CTYPE per axis, in FITS axis order, "" where an axis has none.
    pub ctypes: Vec<String>,
    /// FITS Stokes code of each plane along the Stokes axis, or `None` if
    /// there is no Stokes axis.
    pub stokes_codes: Option<Vec<i64>>,
}

/// FITS Stokes symbols (FITS WCS paper III, table 7), indexed by `code + 8`.
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

/// Read the shape, axis types and Stokes codes of one image, without its data.
pub fn read_input_axes(path: &Path) -> Result<InputAxes> {
    let mut fptr = FitsFile::open(path.to_string_lossy().as_ref())?;
    let hdu = fptr.primary_hdu()?;
    let naxis: i64 = hdu.read_key(&mut fptr, "NAXIS")?;
    let shape = (
        hdu.read_key(&mut fptr, "NAXIS1")?,
        hdu.read_key(&mut fptr, "NAXIS2")?,
    );
    let ctypes: Vec<String> = (1..=naxis)
        .map(|n| {
            hdu.read_key::<String>(&mut fptr, &format!("CTYPE{n}"))
                .map(|c| c.trim().to_string())
                .unwrap_or_default()
        })
        .collect();

    let stokes_codes = match ctypes.iter().position(|c| c == "STOKES") {
        None => None,
        Some(i) => {
            let n = i + 1;
            let len: i64 = hdu.read_key(&mut fptr, &format!("NAXIS{n}"))?;
            // FITS defaults for an absent card
            let key = |fptr: &mut FitsFile, key: &str, default: f64| {
                hdu.read_key::<f64>(fptr, &format!("{key}{n}"))
                    .unwrap_or(default)
            };
            let crval = key(&mut fptr, "CRVAL", 0.0);
            let cdelt = key(&mut fptr, "CDELT", 1.0);
            let crpix = key(&mut fptr, "CRPIX", 0.0);
            Some(
                (1..=len)
                    .map(|pix| (crval + cdelt * (pix as f64 - crpix)).round() as i64)
                    .collect(),
            )
        }
    };

    Ok(InputAxes {
        shape,
        ctypes,
        stokes_codes,
    })
}

/// Read [`InputAxes`] of every input, in `file_list` order.
pub fn read_inputs_axes(file_list: &[PathBuf]) -> Result<Vec<InputAxes>> {
    file_list.par_iter().map(|p| read_input_axes(p)).collect()
}

/// Check that every input has the same value as the first, returning it.
///
/// Mirrors `check_values_match`. `error` wraps the message in the error
/// variant to return when any value differs from the first.
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
        // Keep the message readable when a whole run is mismatched
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

/// Confirm every input image shares the same NAXIS1/NAXIS2 pixel grid.
///
/// Planes are written to the cube at a fixed byte offset per channel, so inputs
/// of differing shape would silently slide against each other and produce a
/// scrambled cube with a self-consistent header.
pub fn check_matching_shapes(file_list: &[PathBuf], axes: &[InputAxes]) -> Result<(i64, i64)> {
    let shapes: Vec<(i64, i64)> = axes.iter().map(|a| a.shape).collect();
    check_values_match(
        file_list,
        &shapes,
        |shape| format!("(NAXIS1, NAXIS2)={shape:?}"),
        "the same pixel grid",
        FitsCubeError::ShapeMismatch,
    )
}

/// Confirm every input image has the same axes and Stokes parameter.
pub fn check_matching_axes(file_list: &[PathBuf], axes: &[InputAxes]) -> Result<()> {
    let ctypes: Vec<Vec<String>> = axes.iter().map(|a| a.ctypes.clone()).collect();
    check_values_match(
        file_list,
        &ctypes,
        |ctypes| format!("axes {}", describe_list(ctypes)),
        "the same axes, in the same order",
        FitsCubeError::AxisMismatch,
    )?;
    let stokes: Vec<Option<Vec<i64>>> = axes.iter().map(|a| a.stokes_codes.clone()).collect();
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
        assert_eq!(
            check_values_match(
                &files,
                &values,
                |v| v.to_string(),
                "x",
                FitsCubeError::Other
            )
            .unwrap(),
            1
        );

        for v in values.iter_mut().skip(1) {
            *v = 2;
        }
        let err = check_values_match(
            &files,
            &values,
            |v| v.to_string(),
            "x",
            FitsCubeError::Other,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("Expected 1 from f0"), "{err}");
        assert!(err.contains("f10 has 2"), "{err}");
        assert!(!err.contains("f11 has 2"), "{err}");
        assert!(err.contains("...and 2 more"), "{err}");
    }
}
