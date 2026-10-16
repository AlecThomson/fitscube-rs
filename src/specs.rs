//! The frequency or time axis of the cube ("spequency" covers both).
use std::path::{Path, PathBuf};

use hifitime::Epoch;

use crate::checks::InputHeader;
use crate::error::{FitsCubeError, Result};

/// The axis values of the inputs and of the cube.
#[derive(Debug, Clone)]
pub struct SpecInfo {
    /// One value per input (Hz or s), in input order.
    pub file_specs: Vec<f64>,
    /// One value per cube channel. Longer than `file_specs` with blanks.
    pub specs: Vec<f64>,
    /// Channels with no input, written as NaN.
    pub missing: Vec<bool>,
}

/// ISO-8601 DATE-OBS to MJD seconds, as `Time(t, format="isot").mjd * 86400`.
pub fn utc_to_mjdsec(utc_time: &str) -> Result<f64> {
    const SEC_PER_DAY: f64 = 86400.0;
    let trimmed = utc_time.trim().trim_matches('\'').trim();
    let epoch = Epoch::from_gregorian_str(trimmed)
        .or_else(|_| trimmed.parse::<Epoch>())
        .map_err(|e| FitsCubeError::TimeParse {
            value: utc_time.to_string(),
            msg: e.to_string(),
        })?;
    Ok(epoch.to_mjd_utc_days() * SEC_PER_DAY)
}

/// Frequency (Hz) or time (MJD seconds) of one image. See [`InputHeader::spec`].
pub fn read_spec_from_header(path: &Path, time_domain_mode: bool) -> Result<f64> {
    InputHeader::read(path)?.spec(path, time_domain_mode)
}

/// For each element, whether it is close to any of `test`: about 0.5 s for MJD
/// seconds and 1 Hz at 1 GHz.
pub fn isin_close(elements: &[f64], test: &[f64], time_domain_mode: bool) -> Vec<bool> {
    let (atol, rtol) = if time_domain_mode {
        (1e-9, 1e-10)
    } else {
        (1e-5, 1e-9)
    };
    elements
        .iter()
        .map(|&e| test.iter().any(|&b| (e - b).abs() <= atol + rtol * b.abs()))
        .collect()
}

/// `np.arange(start, stop, step)`, nudged so rounding does not drop `stop`.
fn np_arange_fix(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = (stop - start) / step + 1.0;
    let x = n - n.trunc();
    let stop = if x < 0.5 {
        stop + step * f64::max(0.1, x)
    } else {
        stop
    };
    let mut out = Vec::new();
    let mut v = start;
    while v < stop {
        out.push(v);
        v += step;
    }
    out
}

/// Smallest step `grid_step` accepts, as a fraction of the smallest diff.
const MAX_BLANKS_PER_GAP: f64 = 16.0;

/// Most grid points `even_spacing` builds per input value.
const MAX_GRID_PER_INPUT: f64 = 1000.0;

/// Channel width of a regular grid with missing channels: the gcd of the
/// diffs, so channels 0, 3, 5 give 1 rather than `min(diffs) = 2`. Falls back
/// to `min(diffs)` when the gcd does not divide every diff or is implausibly
/// small.
fn grid_step(diffs: &[f64]) -> f64 {
    let min_diff = diffs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_diff = diffs.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
    let tol = max_diff * 1e-6;

    let mut g = diffs[0].abs();
    for &d in &diffs[1..] {
        let (mut a, mut b) = if g >= d.abs() {
            (g, d.abs())
        } else {
            (d.abs(), g)
        };
        // Euclid, stopping once the remainder is rounding noise
        while b > tol {
            let r = a % b;
            a = b;
            b = r;
        }
        g = a;
    }

    let divides_all = diffs.iter().all(|&d| {
        let q = d.abs() / g;
        (q - q.round()).abs() <= 1e-3
    });
    // Diffs of 1e6 and 1e6 + 2 have a gcd of 2
    if g <= tol || !divides_all || g < min_diff / MAX_BLANKS_PER_GAP {
        min_diff
    } else {
        g
    }
}

/// Put `specs` on a regular grid and flag the grid points with no input.
///
/// Returns [`FitsCubeError::IrregularSpacing`] if any input would fall between
/// grid points or share one with another input.
pub fn even_spacing(specs: &[f64], time_domain_mode: bool) -> Result<(Vec<f64>, Vec<bool>)> {
    check_finite(specs)?;
    let mut sorted = specs.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len() < 2 {
        let n = sorted.len();
        return Ok((sorted, vec![false; n]));
    }

    let diffs: Vec<f64> = sorted.windows(2).map(|w| w[1] - w[0]).collect();
    let step = grid_step(&diffs);
    let span = sorted[sorted.len() - 1] - sorted[0];
    if span / step > MAX_GRID_PER_INPUT * sorted.len() as f64 {
        return Err(FitsCubeError::IrregularSpacing(format!(
            "A regular grid of step {step} over {} values spanning {span} would need \
             {:.0} channels. Combine without blank channels.",
            sorted.len(),
            span / step + 1.0
        )));
    }
    let new_specs = np_arange_fix(sorted[0], sorted[sorted.len() - 1], step);
    let present = isin_close(&new_specs, &sorted, time_domain_mode);
    let missing: Vec<bool> = present.iter().map(|&p| !p).collect();

    let populated = present.iter().filter(|&&p| p).count();
    if populated != sorted.len() {
        let (spequencies, unit) = if time_domain_mode {
            ("times", "s")
        } else {
            ("frequencies", "Hz")
        };
        return Err(FitsCubeError::IrregularSpacing(format!(
            "Cannot place all {} {spequencies} on a regular grid of {step} {unit} - only \
             {populated} of {} grid points are matched, so blank channels would drop \
             inputs. Combine without blank channels, or regrid the inputs onto a common step.",
            sorted.len(),
            new_specs.len()
        )));
    }
    Ok((new_specs, missing))
}

fn check_finite(specs: &[f64]) -> Result<()> {
    match specs.iter().position(|v| !v.is_finite()) {
        Some(i) => Err(FitsCubeError::InvalidSpec(format!(
            "value {i} of the frequency/time axis is {}",
            specs[i]
        ))),
        None => Ok(()),
    }
}

/// Axis values of the inputs and the cube.
///
/// The values come from `spec_list`, from `spec_file` (one per line), or from
/// each input's header.
pub fn parse_specs(
    file_list: &[PathBuf],
    headers: &[InputHeader],
    spec_file: Option<&Path>,
    spec_list: Option<&[f64]>,
    ignore_spec: bool,
    create_blanks: bool,
    time_domain_mode: bool,
) -> Result<SpecInfo> {
    let n = file_list.len();

    if ignore_spec {
        tracing::info!("Ignoring frequency/time information");
        let seq: Vec<f64> = (0..n).map(|i| i as f64).collect();
        return Ok(SpecInfo {
            file_specs: seq.clone(),
            specs: seq,
            missing: vec![false; n],
        });
    }

    let file_specs: Vec<f64> = match (spec_file, spec_list) {
        (Some(_), Some(_)) => {
            return Err(FitsCubeError::InvalidSpec(
                "Must specify either spec_file or spec_list, not both".to_string(),
            ));
        }
        (Some(path), None) => {
            tracing::info!("Reading spec values from {}", path.display());
            let text = std::fs::read_to_string(path)?;
            let vals: Vec<f64> = text
                .split_whitespace()
                .map(str::parse)
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| {
                    FitsCubeError::InvalidSpec(format!("parsing {}: {e}", path.display()))
                })?;
            if vals.len() != n {
                return Err(FitsCubeError::InvalidSpec(format!(
                    "Number of values in {} ({}) does not match number of images ({n})",
                    path.display(),
                    vals.len()
                )));
            }
            vals
        }
        (None, Some(list)) => {
            if list.len() != n {
                return Err(FitsCubeError::InvalidSpec(format!(
                    "Number of --specs values ({}) does not match number of images ({n})",
                    list.len()
                )));
            }
            list.to_vec()
        }
        (None, None) => file_list
            .iter()
            .zip(headers)
            .map(|(path, header)| header.spec(path, time_domain_mode))
            .collect::<Result<_>>()?,
    };
    check_finite(&file_specs)?;

    let (specs, missing) = if create_blanks {
        tracing::info!("Creating an evenly-spaced axis with interpolated blanks");
        even_spacing(&file_specs, time_domain_mode)?
    } else {
        (file_specs.clone(), vec![false; n])
    };
    Ok(SpecInfo {
        file_specs,
        specs,
        missing,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn one_second_time_steps() {
        let times: Vec<f64> = [0.0, 1.0, 3.0, 4.0].iter().map(|t| 5.18e9 + t).collect();
        let (new, missing) = even_spacing(&times, true).unwrap();
        assert_eq!(new.len(), 5);
        assert_eq!(missing, vec![false, false, true, false, false]);
    }

    #[test]
    fn even_axis_has_no_missing() {
        let specs = vec![1.0e9, 2.0e9, 3.0e9, 4.0e9];
        let (new, missing) = even_spacing(&specs, false).unwrap();
        assert_eq!(new.len(), 4);
        assert!(missing.iter().all(|&m| !m));
    }

    #[test]
    fn gap_is_flagged_missing() {
        let specs = vec![1.0e9, 2.0e9, 4.0e9];
        let (new, missing) = even_spacing(&specs, false).unwrap();
        assert_eq!(new.len(), 4);
        assert_eq!(missing.iter().filter(|&&m| m).count(), 1);
        assert!(missing[2]);
    }

    #[test]
    fn non_adjacent_gaps_recover_true_step() {
        // min(diffs) == 2 would grid to [0, 2, 4] and drop channels 3 and 5
        let specs = vec![0.0, 3.0, 5.0];
        let (new, missing) = even_spacing(&specs, false).unwrap();
        assert_eq!(new.len(), 6);
        assert_eq!(missing.iter().filter(|&&m| m).count(), 3);
        let expected = [false, true, true, false, true, false];
        assert_eq!(missing, expected);
    }

    #[test]
    fn unsorted_values_are_sorted_first() {
        let (new, missing) = even_spacing(&[4.0e9, 1.0e9, 2.0e9], false).unwrap();
        assert_eq!(new, vec![1.0e9, 2.0e9, 3.0e9, 4.0e9]);
        assert_eq!(missing, vec![false, false, true, false]);
    }

    #[test]
    fn fewer_than_two_values_are_even() {
        assert_eq!(even_spacing(&[], false).unwrap(), (vec![], vec![]));
        assert_eq!(
            even_spacing(&[1.0e9], false).unwrap(),
            (vec![1.0e9], vec![false])
        );
    }

    #[test]
    fn inputs_off_the_grid_are_refused() {
        // No common step divides both 1 and sqrt(2)
        let err = even_spacing(&[0.0, 1.0, 1.0 + std::f64::consts::SQRT_2], false).unwrap_err();
        assert!(matches!(err, FitsCubeError::IrregularSpacing(_)), "{err:?}");
        // Duplicates collapse onto one grid point
        let err = even_spacing(&[0.0, 0.0, 2.0], false).unwrap_err();
        assert!(matches!(err, FitsCubeError::IrregularSpacing(_)), "{err:?}");
    }

    #[test]
    fn float_gcd_does_not_explode_the_grid() {
        // The gcd of 1e6 and 1e6 + 2 is 2, a grid of a million channels
        let err = even_spacing(&[0.0, 1e6, 2e6 + 2.0], false).unwrap_err();
        assert!(matches!(err, FitsCubeError::IrregularSpacing(_)), "{err:?}");
        assert_eq!(grid_step(&[1e6, 1e6 + 2.0]), 1e6);
        // Missing channels 1, 2 and 4 still give the unit step
        assert_eq!(grid_step(&[3.0, 2.0]), 1.0);
    }

    #[test]
    fn huge_grids_are_refused() {
        let err = even_spacing(&[0.0, 1.0, 1e9], false).unwrap_err();
        assert!(matches!(err, FitsCubeError::IrregularSpacing(_)), "{err:?}");
    }

    #[test]
    fn non_finite_specs_are_refused() {
        for bad in [f64::NAN, f64::INFINITY] {
            let err = even_spacing(&[1.0, bad, 3.0], false).unwrap_err();
            assert!(matches!(err, FitsCubeError::InvalidSpec(_)), "{err:?}");
        }
    }

    #[test]
    fn mjdsec_round_number() {
        let s = utc_to_mjdsec("1858-11-17T00:00:00").unwrap();
        assert_relative_eq!(s, 0.0, epsilon = 1e-3);
    }

    #[test]
    fn mjdsec_one_day() {
        let s = utc_to_mjdsec("1858-11-18T00:00:00").unwrap();
        assert_relative_eq!(s, 86400.0, epsilon = 1e-3);
    }
}
