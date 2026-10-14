//! WCS lookups on the current HDU of an open file.
use atfits_rs::{AtfitsError, TargetAxis, find_target_axis_in};
use fitsio::FitsFile;

use crate::error::{FitsCubeError, Result};

/// The first axis whose CTYPE contains `name`, or `None`.
pub(crate) fn find_axis(fptr: &mut FitsFile, name: &str) -> Result<Option<TargetAxis>> {
    match find_target_axis_in(fptr, name) {
        Ok(axis) => Ok(Some(axis)),
        Err(AtfitsError::TargetAxisMissing(_)) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// World value of the 0-based pixel `index` along `axis`.
pub(crate) fn world(axis: &TargetAxis, index: usize) -> f64 {
    axis.world(index as f64 + 1.0)
}

/// Factor converting the frequency unit of `axis` to Hz.
pub(crate) fn hz_per_unit(axis: &TargetAxis) -> Result<f64> {
    axis.unit_scale().ok_or_else(|| {
        FitsCubeError::InvalidSpec(format!(
            "unknown frequency unit CUNIT{}={:?}; expected Hz, kHz, MHz, GHz or THz",
            axis.fits_idx,
            axis.cunit.as_deref().unwrap_or("")
        ))
    })
}
