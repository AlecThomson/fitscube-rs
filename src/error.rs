//! Error types for fitscube-rs.
//!
//! Mirrors the exception hierarchy of the original `fitscube` Python package
//! ([`fitscube.exceptions`](https://github.com/AlecThomson/fitscube)): a base
//! [`FitsCubeError`] with dedicated variants for a missing target (FREQ/TIME)
//! axis and an inaccessible channel, plus the I/O and parsing errors that arise
//! in the Rust implementation.
use std::path::PathBuf;

use thiserror::Error;

/// Top-level error for all fitscube-rs operations.
///
/// `TargetAxisMissing` and `ChannelMissing` correspond to the Python
/// `TargetAxisMissingException` and `ChannelMissingException`.
#[derive(Debug, Error)]
pub enum FitsCubeError {
    /// Underlying cfitsio error.
    #[error("FITS I/O error: {0}")]
    Fits(#[from] fitsio::errors::Error),

    /// Filesystem I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// ndarray shape mismatch when reshaping a flat plane buffer.
    #[error("shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),

    /// A required header keyword was absent.
    #[error("missing header keyword: {0}")]
    MissingKeyword(String),

    /// The FREQ/TIME (or other requested) axis was not found in the WCS.
    ///
    /// Equivalent to Python `TargetAxisMissingException`.
    #[error("target axis missing: {0}")]
    TargetAxisMissing(String),

    /// A requested channel/timestep could not be accessed.
    ///
    /// Equivalent to Python `ChannelMissingException`.
    #[error("channel missing: {0}")]
    ChannelMissing(String),

    /// The output file already exists and `overwrite` was not set.
    #[error("output file {0} already exists (use overwrite)")]
    OutputExists(PathBuf),

    /// Mutually exclusive spectral-input options were combined, or a count
    /// mismatch was detected.
    #[error("invalid spectral input: {0}")]
    InvalidSpec(String),

    /// A FITS image had an unexpected dimensionality.
    #[error("unsupported NAXIS={0}")]
    UnsupportedNaxis(i64),

    /// Failed to parse a DATE-OBS timestamp.
    #[error("could not parse DATE-OBS {value:?}: {msg}")]
    TimeParse { value: String, msg: String },

    /// Catch-all for invariant violations.
    #[error("{0}")]
    Other(String),
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, FitsCubeError>;
