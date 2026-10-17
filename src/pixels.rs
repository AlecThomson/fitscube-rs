//! Pixel helpers on top of `atfits_rs::raw`.
use crate::bounding_box::BoundingBox;

/// Per-axis ranges selecting `bbox` from every plane of an image of `dims`.
pub(crate) fn bbox_ranges(dims: &[usize], bbox: &BoundingBox) -> Vec<(usize, usize)> {
    let mut ranges: Vec<(usize, usize)> = dims.iter().map(|&n| (0, n)).collect();
    ranges[0] = (bbox.ymin, bbox.ymax);
    ranges[1] = (bbox.xmin, bbox.xmax);
    ranges
}

/// Run `f` with the Rust type of the stored pixels of a `bitpix` image.
macro_rules! with_pixel_type {
    ($bitpix:expr, $f:ident ( $($arg:expr),* )) => {
        match $bitpix {
            8 => $f::<u8>($($arg),*),
            16 => $f::<i16>($($arg),*),
            32 => $f::<i32>($($arg),*),
            64 => $f::<i64>($($arg),*),
            -32 => $f::<f32>($($arg),*),
            -64 => $f::<f64>($($arg),*),
            other => Err($crate::error::FitsCubeError::UnsupportedBitpix(other)),
        }
    };
}
pub(crate) use with_pixel_type;

/// What blank pixels read as: NaN for floats, and as stored for integers.
pub(crate) trait NullValue: Sized {
    fn null_value() -> Option<Self> {
        None
    }
}
impl NullValue for u8 {}
impl NullValue for i16 {}
impl NullValue for i32 {}
impl NullValue for i64 {}
impl NullValue for f32 {
    fn null_value() -> Option<Self> {
        Some(Self::NAN)
    }
}
impl NullValue for f64 {
    fn null_value() -> Option<Self> {
        Some(Self::NAN)
    }
}
