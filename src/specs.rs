//! Spectral/temporal axis parsing ("spequency" = frequency *or* time).
//!
//! Port of the metadata-parsing half of `fitscube.combine_fits`: pull a
//! frequency (Hz) or time (MJD seconds) value out of each input image, optionally
//! synthesise an evenly-spaced axis with gaps flagged as missing channels.
use std::path::Path;

use hifitime::Epoch;

use crate::error::{FitsCubeError, Result};
use crate::fits_io::{HeaderGeom, find_target_axis, read_key_f64, read_key_string};

/// Result of parsing the spectral/temporal axis.
#[derive(Debug, Clone)]
pub struct SpecInfo {
    /// One value per input file (Hz or s), used to sort the files.
    pub file_specs: Vec<f64>,
    /// Output-axis values (Hz or s); longer than `file_specs` when blanks are
    /// interpolated.
    pub specs: Vec<f64>,
    /// `true` for each output channel that has no backing file (a blank plane).
    pub missing: Vec<bool>,
}

/// Convert a CUNIT string for a frequency axis to a factor that scales the axis
/// value to Hz. Defaults to 1.0 (assume Hz) for unknown/absent units.
fn freq_unit_to_hz(cunit: Option<&str>) -> f64 {
    match cunit.map(|s| s.trim().to_ascii_uppercase()) {
        Some(ref u) if u == "HZ" => 1.0,
        Some(ref u) if u == "KHZ" => 1e3,
        Some(ref u) if u == "MHZ" => 1e6,
        Some(ref u) if u == "GHZ" => 1e9,
        _ => 1.0,
    }
}

/// Convert an ISO-8601 (`isot`) DATE-OBS timestamp to MJD seconds.
///
/// Matches `Time(utc_time, format="isot").mjd * 86400` from the original.
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

/// Read the frequency (Hz) or time (s) of a single image from its header.
///
/// * time mode: `DATE-OBS` → MJD seconds (2D or N-D).
/// * frequency mode, 2D: the `REFFREQ` keyword (Hz).
/// * frequency mode, N-D: the spectral world value at pixel 0,
///   `CRVAL + (1 − CRPIX)·CDELT`, scaled to Hz by CUNIT.
pub fn read_spec_from_header(path: &Path, time_domain_mode: bool) -> Result<f64> {
    if time_domain_mode {
        let date_obs = read_key_string(path, "DATE-OBS")?.ok_or_else(|| {
            FitsCubeError::MissingKeyword(format!(
                "DATE-OBS not in header of {} (needed for time-domain mode)",
                path.display()
            ))
        })?;
        return utc_to_mjdsec(&date_obs);
    }

    let geom = HeaderGeom::read(path)?;
    if geom.is_2d() {
        let reffreq = read_key_f64(path, "REFFREQ")?.ok_or_else(|| {
            FitsCubeError::MissingKeyword(format!(
                "REFFREQ not in header of {}. Cannot combine 2D images without \
                 frequency information.",
                path.display()
            ))
        })?;
        return Ok(reffreq);
    }

    // N-D: locate the FREQ axis and evaluate its world coordinate at pixel 0.
    let axis = find_target_axis(path, "FREQ").map_err(|_| {
        FitsCubeError::TargetAxisMissing(format!(
            "No FREQ axis found in WCS of {}. Cannot combine N-D images without \
             frequency information.",
            path.display()
        ))
    })?;
    let world = axis.crval + (1.0 - axis.crpix) * axis.cdelt;
    Ok(world * freq_unit_to_hz(axis.cunit.as_deref()))
}

/// `isin_close`: for each element, is it close to *any* test element?
///
/// NOTE: the original calls `np.isclose(element, test, atol, rtol)` with the
/// `rtol`/`atol` positions swapped, so the effective tolerances differ from the
/// variable names. We replicate the *effective* behaviour for bit-for-bit
/// parity with `fitscube`:
///   * frequency: `atol = 1e-5`, `rtol = 1e-9`
///   * time:      `atol = 1e-10`, `rtol = 1e-9`
///
/// Closeness test: `|e − b| ≤ atol + rtol·|b|`.
pub fn isin_close(elements: &[f64], test: &[f64], time_domain_mode: bool) -> Vec<bool> {
    let (atol, rtol) = if time_domain_mode {
        (1e-10, 1e-9)
    } else {
        (1e-5, 1e-9)
    };
    elements
        .iter()
        .map(|&e| test.iter().any(|&b| (e - b).abs() <= atol + rtol * b.abs()))
        .collect()
}

/// `np_arange_fix`: `np.arange` with a nudge so the stop endpoint is included
/// when floating-point rounding would otherwise drop it.
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
    // Match numpy: stop is exclusive.
    while v < stop {
        out.push(v);
        v += step;
    }
    out
}

/// Fundamental channel step: the gcd of the observed diffs.
///
/// On a regular grid with missing channels every diff is an integer multiple
/// of the channel width, so the gcd recovers that width even when no two
/// surviving channels are adjacent — the case where `min(diffs)` overestimates
/// the step (e.g. present channels 0, 3, 5 give `min = 2` but the true step is
/// 1). Falls back to `min(diffs)` if the gcd is numerically unstable or does
/// not divide every diff, so the common case is bit-for-bit unchanged (any
/// surviving adjacent pair makes the gcd equal the minimum diff exactly).
fn grid_step(diffs: &[f64]) -> f64 {
    let min_diff = diffs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_diff = diffs.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
    let tol = max_diff * 1e-6;

    let mut g = diffs[0].abs();
    for &d in &diffs[1..] {
        let (mut a, mut b) = if g >= d.abs() { (g, d.abs()) } else { (d.abs(), g) };
        // Tolerant Euclid: stop once the remainder is within the noise floor.
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
    if g <= tol || !divides_all {
        min_diff
    } else {
        g
    }
}

/// Build an evenly-spaced axis from possibly-uneven input values, flagging the
/// interpolated gaps as missing channels. Mirrors `even_spacing`.
pub fn even_spacing(specs: &[f64], time_domain_mode: bool) -> (Vec<f64>, Vec<bool>) {
    assert!(specs.len() >= 2, "need at least two values to space evenly");
    let diffs: Vec<f64> = specs.windows(2).map(|w| w[1] - w[0]).collect();
    let step = grid_step(&diffs);
    let new_specs = np_arange_fix(specs[0], specs[specs.len() - 1], step);
    let present = isin_close(&new_specs, specs, time_domain_mode);
    let missing: Vec<bool> = present.iter().map(|&p| !p).collect();
    (new_specs, missing)
}

/// Parse the spectral/temporal axis for a set of input files.
///
/// Mirrors `parse_specs_coro`. `spec_list` (the CLI `--specs`) supplies the
/// per-file values directly; `spec_file` reads them one-per-line from a text
/// file; otherwise they are read from each header.
pub fn parse_specs(
    file_list: &[std::path::PathBuf],
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

    if spec_file.is_some() && spec_list.is_some() {
        return Err(FitsCubeError::InvalidSpec(
            "Must specify either spec_file or spec_list, not both".to_string(),
        ));
    }

    let mut file_specs: Vec<f64> = if let Some(path) = spec_file {
        tracing::info!("Reading spec values from {}", path.display());
        let text = std::fs::read_to_string(path)?;
        let vals: Vec<f64> = text
            .split_whitespace()
            .map(|t| t.parse::<f64>())
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| FitsCubeError::InvalidSpec(format!("parsing {}: {e}", path.display())))?;
        if vals.len() != n {
            return Err(FitsCubeError::InvalidSpec(format!(
                "Number of values in {} ({}) does not match number of images ({n})",
                path.display(),
                vals.len()
            )));
        }
        vals
    } else if let Some(list) = spec_list {
        if list.len() != n {
            return Err(FitsCubeError::InvalidSpec(format!(
                "Number of --specs values ({}) does not match number of images ({n})",
                list.len()
            )));
        }
        list.to_vec()
    } else {
        file_list
            .iter()
            .map(|p| read_spec_from_header(p, time_domain_mode))
            .collect::<Result<Vec<f64>>>()?
    };

    let (specs, missing) = if create_blanks {
        tracing::info!("Creating an evenly-spaced axis with interpolated blanks");
        even_spacing(&file_specs, time_domain_mode)
    } else {
        (file_specs.clone(), vec![false; file_specs.len()])
    };

    // Keep `file_specs` length == number of files.
    file_specs.truncate(n);

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
    fn even_axis_has_no_missing() {
        let specs = vec![1.0e9, 2.0e9, 3.0e9, 4.0e9];
        let (new, missing) = even_spacing(&specs, false);
        assert_eq!(new.len(), 4);
        assert!(missing.iter().all(|&m| !m));
    }

    #[test]
    fn gap_is_flagged_missing() {
        // Missing the 3 GHz channel.
        let specs = vec![1.0e9, 2.0e9, 4.0e9];
        let (new, missing) = even_spacing(&specs, false);
        assert_eq!(new.len(), 4);
        // The interpolated 3 GHz slot is the only missing one.
        assert_eq!(missing.iter().filter(|&&m| m).count(), 1);
        assert!(missing[2]);
    }

    #[test]
    fn non_adjacent_gaps_recover_true_step() {
        // Present channels 0, 3, 5 on a unit grid (1, 2, 4 missing). No two
        // surviving channels are adjacent, so `min(diffs) == 2` would mis-grid
        // the axis to [0, 2, 4] and drop the real channels. The gcd of the
        // diffs [3, 2] recovers the true step of 1.
        let specs = vec![0.0, 3.0, 5.0];
        let (new, missing) = even_spacing(&specs, false);
        assert_eq!(new.len(), 6, "should rebuild the full 0..=5 grid");
        assert_eq!(missing.iter().filter(|&&m| m).count(), 3);
        let expected = [false, true, true, false, true, false];
        assert_eq!(missing, expected);
    }

    #[test]
    fn mjdsec_round_number() {
        // 1858-11-17T00:00:00 is MJD 0.
        let s = utc_to_mjdsec("1858-11-17T00:00:00").unwrap();
        assert_relative_eq!(s, 0.0, epsilon = 1e-3);
    }

    #[test]
    fn mjdsec_one_day() {
        let s = utc_to_mjdsec("1858-11-18T00:00:00").unwrap();
        assert_relative_eq!(s, 86400.0, epsilon = 1e-3);
    }
}
