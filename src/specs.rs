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
    /// Largest distance of an input from its channel, when blanks were made.
    pub spec_error: Option<f64>,
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

/// How far `value` may sit from a grid point and still be on it: about 0.5 s
/// for MJD seconds and 1 Hz at 1 GHz.
fn match_tolerance(value: f64, time_domain_mode: bool) -> f64 {
    let (atol, rtol) = if time_domain_mode {
        (1e-9, 1e-10)
    } else {
        (1e-5, 1e-9)
    };
    atol + rtol * value.abs()
}

/// A regular grid through a set of sorted values.
#[derive(Debug)]
struct GridFit {
    start: f64,
    step: f64,
    /// Grid point of each value.
    index: Vec<usize>,
    /// Distance of each value from its grid point.
    residuals: Vec<f64>,
}

/// The widest regular grid that holds every sorted, distinct value, or `None`
/// if no grid of at most 1000 points per value does.
///
/// The step divides the smallest gap, so steps of that gap over 1, 2, 3... are
/// tried in turn. Each is fitted to all the values, and the first that puts
/// every value on a grid point within the match tolerance wins.
fn fit_grid(specs: &[f64], time_domain_mode: bool) -> Option<GridFit> {
    let offsets: Vec<f64> = specs.iter().map(|&v| v - specs[0]).collect();
    let span = offsets[offsets.len() - 1];
    let min_gap = specs
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f64::INFINITY, f64::min);
    let max_channels = 1000.0 * specs.len() as f64;
    let n = specs.len() as f64;

    let mut divisor = 1.0;
    while span * divisor / min_gap < max_channels {
        let index: Vec<f64> = offsets
            .iter()
            .map(|&o| (o * divisor / min_gap).round())
            .collect();
        // Least-squares line through (index, offset)
        let mean_i = index.iter().sum::<f64>() / n;
        let mean_o = offsets.iter().sum::<f64>() / n;
        let (mut sxy, mut sxx) = (0.0, 0.0);
        for (&i, &o) in index.iter().zip(&offsets) {
            sxy += (i - mean_i) * (o - mean_o);
            sxx += (i - mean_i) * (i - mean_i);
        }
        let step = sxy / sxx;
        let start = mean_o - step * mean_i;
        let residuals: Vec<f64> = index
            .iter()
            .zip(&offsets)
            .map(|(&i, &o)| (o - start - i * step).abs())
            .collect();
        let on_grid = residuals
            .iter()
            .zip(specs)
            .all(|(&r, &v)| r <= match_tolerance(v, time_domain_mode));
        if on_grid {
            return Some(GridFit {
                start: specs[0] + start,
                step,
                index: index.iter().map(|&i| i as usize).collect(),
                residuals,
            });
        }
        divisor += 1.0;
    }
    None
}

/// Put `specs` on a regular grid and flag the grid points with no input.
///
/// Returns [`FitsCubeError::IrregularSpacing`] if no grid of at most 1000
/// points per input holds every input. Inputs that sit on the grid only within
/// the match tolerance are kept, with a warning.
pub fn even_spacing(specs: &[f64], time_domain_mode: bool) -> Result<(Vec<f64>, Vec<bool>)> {
    check_finite(specs)?;
    let mut sorted = specs.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len() < 2 {
        let n = sorted.len();
        return Ok((sorted, vec![false; n]));
    }

    let (spequencies, unit) = if time_domain_mode {
        ("times", "s")
    } else {
        ("frequencies", "Hz")
    };
    let distinct = sorted.windows(2).all(|w| w[1] > w[0]);
    let Some(grid) = distinct
        .then(|| fit_grid(&sorted, time_domain_mode))
        .flatten()
    else {
        return Err(FitsCubeError::IrregularSpacing(format!(
            "No regular grid of at most {} points holds all {} {spequencies}, so blank \
             channels would drop inputs. Combine without blank channels, or regrid the \
             inputs onto a common step.",
            1000 * sorted.len(),
            sorted.len()
        )));
    };

    let (worst, offset) = grid
        .residuals
        .iter()
        .copied()
        .enumerate()
        .fold((0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
    let largest = sorted.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if offset > 1e-12 * largest {
        tracing::warn!(
            "The {spequencies} are not exactly on a regular grid of {} {unit}: {} {unit} \
             is {offset} {unit} from its plane. The cube labels each plane with its grid \
             value, not the input's own.",
            grid.step,
            sorted[worst]
        );
    }

    let n_chan = grid.index[grid.index.len() - 1] + 1;
    let new_specs: Vec<f64> = (0..n_chan)
        .map(|i| grid.start + i as f64 * grid.step)
        .collect();
    let mut missing = vec![true; n_chan];
    for &i in &grid.index {
        missing[i] = false;
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
            spec_error: None,
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

    if !create_blanks {
        return Ok(SpecInfo {
            specs: file_specs.clone(),
            file_specs,
            missing: vec![false; n],
            spec_error: None,
        });
    }

    tracing::info!("Creating an evenly-spaced axis with interpolated blanks");
    let (specs, missing) = even_spacing(&file_specs, time_domain_mode)?;
    let mut sorted = file_specs.clone();
    sorted.sort_by(f64::total_cmp);
    let populated = specs.iter().zip(&missing).filter(|&(_, &m)| !m);
    let spec_error = sorted
        .iter()
        .zip(populated)
        .fold(0.0_f64, |e, (&f, (&s, _))| e.max((f - s).abs()));
    Ok(SpecInfo {
        file_specs,
        specs,
        missing,
        spec_error: Some(spec_error),
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
        for (got, want) in new.iter().zip([1.0e9, 2.0e9, 3.0e9, 4.0e9]) {
            assert_relative_eq!(*got, want, epsilon = 1e-3);
        }
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
        // Irrational offsets with no common step, as in combine_guards.rs
        use std::f64::consts::{E, LN_10, PI, SQRT_2};
        let irregular: Vec<f64> = [0.0, LN_10, E, SQRT_2 * 2.0, PI, 3.0 * E / 2.0]
            .iter()
            .map(|x| 1e9 + x * 1e7)
            .collect();
        let err = even_spacing(&irregular, false).unwrap_err();
        assert!(matches!(err, FitsCubeError::IrregularSpacing(_)), "{err:?}");
        // Duplicates collapse onto one grid point
        let err = even_spacing(&[0.0, 0.0, 2.0], false).unwrap_err();
        assert!(matches!(err, FitsCubeError::IrregularSpacing(_)), "{err:?}");
    }

    #[test]
    fn offsets_within_tolerance_keep_the_coarse_grid() {
        // 2 Hz off a 1 MHz grid at 1 GHz is within tolerance, not a 2 Hz grid
        let (new, missing) = even_spacing(&[1e9, 1e9 + 1e6, 1e9 + 2e6 + 2.0], false).unwrap();
        assert_eq!(new.len(), 3);
        assert!(missing.iter().all(|&m| !m));
    }

    /// Channel index on the native grid, and centre frequency, of every
    /// RACS-all channel: low3 799.5-1087.5, mid 1295.5-1439.5 and high
    /// 1511.5-1799.5 MHz.
    fn racs_all_channels(width: f64) -> (Vec<i64>, Vec<f64>) {
        let per_mhz = (1e6 / width).round() as i64;
        let index: Vec<i64> = [(0, 288), (496, 144), (712, 288)]
            .iter()
            .flat_map(|&(start, count)| (start * per_mhz)..((start + count) * per_mhz))
            .collect();
        let freqs = index
            .iter()
            .map(|&i| 799.5e6 + width / 2.0 + i as f64 * width)
            .collect();
        (index, freqs)
    }

    fn gcd(a: i64, b: i64) -> i64 {
        if b == 0 { a } else { gcd(b, a % b) }
    }

    /// wsclean's frequency for each image of channels `a..=b`, and the planes
    /// of the exact grid, found with integers on the half-channel grid.
    fn check_bins(index: &[i64], freqs: &[f64], bins: &[(usize, usize)]) {
        let centres: Vec<f64> = bins
            .iter()
            .map(|&(a, b)| (freqs[a] + freqs[b]) / 2.0)
            .collect();
        let halves: Vec<i64> = bins.iter().map(|&(a, b)| index[a] + index[b]).collect();
        let step = halves.windows(2).fold(0, |g, w| gcd(g, w[1] - w[0]));
        let planes = ((halves[halves.len() - 1] - halves[0]) / step + 1) as usize;

        let (new, missing) = even_spacing(&centres, false).unwrap();
        assert_eq!(new.len(), planes, "{bins:?}");
        assert_eq!(missing.iter().filter(|&&m| !m).count(), centres.len());
    }

    #[test]
    fn racs_all_uneven_bins() {
        // Bin sizes cycle through -25%..+25% of the nominal size
        let jitter = [1.0, 0.75, 1.25, 0.9, 1.1, 0.8, 1.2];
        for width in [0.5e6, 1e6, 2e6] {
            let (index, freqs) = racs_all_channels(width);
            for size in 2..=40 {
                let mut bins = Vec::new();
                let mut start = 0;
                while start < index.len() {
                    let len = (size as f64 * jitter[bins.len() % jitter.len()]).round() as usize;
                    let end = (start + len.max(1)).min(index.len());
                    bins.push((start, end - 1));
                    start = end;
                }
                check_bins(&index, &freqs, &bins);
            }
        }
    }

    #[test]
    fn racs_all_wsclean_default_splits() {
        let (index, freqs) = racs_all_channels(1e6);
        let n = index.len();
        for channels_out in 2..=84 {
            let bins: Vec<(usize, usize)> = (0..channels_out)
                .map(|i| (i * n / channels_out, (i + 1) * n / channels_out - 1))
                .collect();
            check_bins(&index, &freqs, &bins);
        }
    }

    #[test]
    fn racs_all_fine_channels() {
        // ASKAP's native 18.5 kHz channels, imaged 54 at a time with a few odd bins
        let (index, freqs) = racs_all_channels(1e6 / 54.0);
        let mut bins = Vec::new();
        let mut start = 0;
        while start < index.len() {
            let len = if bins.len() % 7 == 3 { 50 } else { 54 };
            let end = (start + len).min(index.len());
            bins.push((start, end - 1));
            start = end;
        }
        check_bins(&index, &freqs, &bins);
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
