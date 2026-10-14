//! WCS lookups on the current HDU of an open file.
use atfits_rs::{AtfitsError, TargetAxis, find_target_axis_in, key_exists, key_f64};
use fitsio::FitsFile;

use crate::error::{FitsCubeError, Result};

/// The first axis whose CTYPE contains `name`, or `None`.
///
/// atfits ignores PCi_i, so it is applied to the step here.
pub(crate) fn find_axis(fptr: &mut FitsFile, name: &str) -> Result<Option<TargetAxis>> {
    let mut axis = match find_target_axis_in(fptr, name) {
        Ok(axis) => axis,
        Err(AtfitsError::TargetAxisMissing(_)) => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let i = axis.fits_idx;
    if !key_exists(fptr, &format!("CD{i}_{i}"))? {
        axis.cdelt *= key_f64(fptr, &format!("PC{i}_{i}"))?.unwrap_or(1.0);
    }
    Ok(Some(axis))
}

/// World value of the 0-based pixel `index` along `axis`.
pub(crate) fn world(axis: &TargetAxis, index: usize) -> f64 {
    axis.world(index as f64 + 1.0)
}

/// Factor converting the frequency unit of `axis` to Hz.
pub(crate) fn hz_per_unit(axis: &TargetAxis) -> Result<f64> {
    atfits_rs::spectral_unit_scale(axis.cunit.as_deref()).ok_or_else(|| {
        FitsCubeError::InvalidSpec(format!(
            "unknown frequency unit CUNIT{}={:?}; expected Hz, kHz, MHz or GHz",
            axis.fits_idx,
            axis.cunit.as_deref().unwrap_or("")
        ))
    })
}
