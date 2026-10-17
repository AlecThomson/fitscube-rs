//! Error type. Variants named after a `fitscube.exceptions` class are raised
//! as that class by the Python bindings.
use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum FitsCubeError {
    #[error("FITS I/O error: {0}")]
    Fits(#[from] fitsio::errors::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("thread pool: {0}")]
    ThreadPool(#[from] rayon::ThreadPoolBuildError),

    /// Error from an atfits-rs helper with no closer match here.
    #[error("{0}")]
    Atfits(String),

    #[error("missing header keyword: {0}")]
    MissingKeyword(String),

    /// `TargetAxisMissingException`
    #[error("target axis missing: {0}")]
    TargetAxisMissing(String),

    /// `ChannelMissingException`
    #[error("channel missing: {0}")]
    ChannelMissing(String),

    /// `ShapeMismatchException`
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// `AxisMismatchException`
    #[error("axis mismatch: {0}")]
    AxisMismatch(String),

    /// `StokesMismatchException`
    #[error("Stokes mismatch: {0}")]
    StokesMismatch(String),

    /// `AxisOrderException`
    #[error("axis order: {0}")]
    AxisOrder(String),

    /// `IrregularSpacingException`
    #[error("irregular spacing: {0}")]
    IrregularSpacing(String),

    /// Integer inputs differ in BITPIX, BSCALE, BZERO or BLANK.
    #[error("scaling mismatch: {0}")]
    ScalingMismatch(String),

    /// Python `NotImplementedError`
    #[error("not implemented: {0}")]
    NotImplemented(String),

    /// Python `FileExistsError`
    #[error("output file {0} already exists (use overwrite)")]
    OutputExists(PathBuf),

    #[error("output {0} is the input file")]
    OutputIsInput(PathBuf),

    #[error("invalid spectral input: {0}")]
    InvalidSpec(String),

    #[error("float_length={0} is not a FITS float width; use 32 or 64")]
    InvalidFloatLength(u8),

    #[error("no input images given")]
    NoInputs,

    #[error(
        "blank channels and invalidated zeros are written as NaN, which integer output \
         (BITPIX={0}) cannot hold; pass float_length=32 or 64 (--floating)"
    )]
    BlanksNeedFloat(i64),

    #[error("no input image has any finite pixels to bound")]
    NoValidData,

    #[error("unsupported BITPIX={0}")]
    UnsupportedBitpix(i64),

    #[error("unsupported NAXIS={0}")]
    UnsupportedNaxis(i64),

    #[error("could not parse DATE-OBS {value:?}: {msg}")]
    TimeParse { value: String, msg: String },

    #[error("RICE_1 cannot store floats losslessly; use GZIP_2, or a quantize_level above 0")]
    RiceNeedsQuantize,

    #[error("{0} is already compressed")]
    AlreadyCompressed(PathBuf),

    #[error("{path} holds integers (BITPIX={bitpix}); only float cubes are supported")]
    IntegerTiles { path: PathBuf, bitpix: i64 },

    #[error(
        "{0} was already quantized once (FCLOSSY=T), and quantizing again adds a new error \
         each time. Use lossless 'tile-gzip', or set allow_requantize to accept the extra error."
    )]
    AlreadyQuantized(PathBuf),

    #[error("{0} holds no tile-compressed image")]
    NotTileCompressed(PathBuf),

    #[error("Cannot derive an output name for {0}; pass output")]
    NoOutputName(PathBuf),

    #[error("a worker thread panicked")]
    WorkerPanic,

    /// The plane writer stopped, so the readers gave up. The writer's own
    /// error is the one returned.
    #[error("cancelled after another error")]
    Cancelled,
}

impl From<atfits_rs::AtfitsError> for FitsCubeError {
    fn from(e: atfits_rs::AtfitsError) -> Self {
        use atfits_rs::AtfitsError as A;
        match e {
            A::Fits(e) => Self::Fits(e),
            A::Io(e) => Self::Io(e),
            A::MissingKeyword(s) => Self::MissingKeyword(s),
            A::TargetAxisMissing(s) => Self::TargetAxisMissing(s),
            A::UnsupportedNaxis(n) => Self::UnsupportedNaxis(n),
            A::Other(s) => Self::Atfits(s),
        }
    }
}

pub type Result<T> = std::result::Result<T, FitsCubeError>;
