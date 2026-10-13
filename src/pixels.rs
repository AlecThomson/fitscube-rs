//! Pixel reads and writes in the image's own types, straight through cfitsio.
use fitsio::FitsFile;
use fitsio::errors::check_status;
use fitsio::sys;

use crate::bounding_box::BoundingBox;
use crate::error::Result;

/// A FITS pixel type.
///
/// # Safety
/// `DATATYPE` must be the cfitsio type code of `Self`, and `Self` must be a
/// plain number with no padding, so a slice of it can be viewed as bytes.
pub(crate) unsafe trait Pixel: Copy + Default + Send + Sync + 'static {
    const DATATYPE: u32;
    fn to_be(self) -> Self;
}

macro_rules! int_pixel {
    ($($t:ty => $code:ident),*) => {$(
        unsafe impl Pixel for $t {
            const DATATYPE: u32 = sys::$code;
            fn to_be(self) -> Self {
                <$t>::to_be(self)
            }
        }
    )*};
}
int_pixel!(u8 => TBYTE, i16 => TSHORT, i32 => TINT, i64 => TLONGLONG);

unsafe impl Pixel for f32 {
    const DATATYPE: u32 = sys::TFLOAT;
    fn to_be(self) -> Self {
        f32::from_bits(self.to_bits().to_be())
    }
}

unsafe impl Pixel for f64 {
    const DATATYPE: u32 = sys::TDOUBLE;
    fn to_be(self) -> Self {
        f64::from_bits(self.to_bits().to_be())
    }
}

/// Swap `data` to the big-endian byte order FITS stores, in place.
pub(crate) fn swap_to_be<T: Pixel>(data: &mut [T]) {
    for v in data.iter_mut() {
        *v = v.to_be();
    }
}

/// View pixels as raw bytes.
pub(crate) fn as_bytes<T: Pixel>(data: &[T]) -> &[u8] {
    // SAFETY: `Pixel` types are plain numbers without padding.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data)) }
}

/// Read and write stored values as they are, ignoring BSCALE and BZERO.
pub(crate) fn disable_scaling(fptr: &mut FitsFile) -> Result<()> {
    let mut status = 0;
    // SAFETY: `fptr` is an open file positioned on an image HDU.
    unsafe { sys::ffpscl(fptr.as_raw(), 1.0, 0.0, &mut status) };
    check_status(status)?;
    Ok(())
}

/// Read pixels of the current image HDU into `out`, converted to `T`.
///
/// `dims` is the image shape in FITS order. `ranges` picks a 0-based,
/// end-exclusive range per axis, or every pixel when `None`. With `nulval`,
/// BLANK (and float NaN) pixels are set to it.
pub(crate) fn read_pixels<T: Pixel>(
    fptr: &mut FitsFile,
    dims: &[usize],
    ranges: Option<&[(usize, usize)]>,
    nulval: Option<T>,
    out: &mut Vec<T>,
) -> Result<()> {
    let n: usize = match ranges {
        Some(r) => r.iter().map(|(a, b)| b - a).product(),
        None => dims.iter().product(),
    };
    out.clear();
    out.resize(n, T::default());
    let mut nulval = nulval;
    let nul_ptr = nulval
        .as_mut()
        .map_or(std::ptr::null_mut(), |v| (v as *mut T).cast());
    let mut anynul = 0;
    let mut status = 0;
    match ranges {
        None => unsafe {
            // SAFETY: `out` holds the `n` pixels requested, of type `T`.
            sys::ffgpv(
                fptr.as_raw(),
                T::DATATYPE as _,
                1,
                n as _,
                nul_ptr,
                out.as_mut_ptr().cast(),
                &mut anynul,
                &mut status,
            );
        },
        Some(ranges) => {
            let mut first: Vec<_> = ranges.iter().map(|&(a, _)| (a + 1) as _).collect();
            let mut last: Vec<_> = ranges.iter().map(|&(_, b)| b as _).collect();
            let mut inc = vec![1; ranges.len()];
            // SAFETY: `out` holds the product of the range lengths, one pixel
            // per element of the subset.
            unsafe {
                sys::ffgsv(
                    fptr.as_raw(),
                    T::DATATYPE as _,
                    first.as_mut_ptr(),
                    last.as_mut_ptr(),
                    inc.as_mut_ptr(),
                    nul_ptr,
                    out.as_mut_ptr().cast(),
                    &mut anynul,
                    &mut status,
                );
            }
        }
    }
    check_status(status)?;
    Ok(())
}

/// Per-axis ranges selecting `bbox` from every plane of an image of `dims`.
pub(crate) fn bbox_ranges(dims: &[usize], bbox: &BoundingBox) -> Vec<(usize, usize)> {
    let mut ranges: Vec<(usize, usize)> = dims.iter().map(|&n| (0, n)).collect();
    ranges[0] = (bbox.ymin, bbox.ymax);
    ranges[1] = (bbox.xmin, bbox.xmax);
    ranges
}

/// Write all of `data` to the current image HDU, converting from `T`.
pub(crate) fn write_pixels<T: Pixel>(fptr: &mut FitsFile, data: &[T]) -> Result<()> {
    let mut status = 0;
    // SAFETY: cfitsio reads `data.len()` values of type `T` and does not
    // modify them.
    unsafe {
        sys::ffppr(
            fptr.as_raw(),
            T::DATATYPE as _,
            1,
            data.len() as _,
            data.as_ptr() as *mut _,
            &mut status,
        );
    }
    check_status(status)?;
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn big_endian_bytes() {
        let mut ints = [1i16, 2];
        swap_to_be(&mut ints);
        assert_eq!(as_bytes(&ints), &[0, 1, 0, 2]);
        let mut floats = [1.0f32];
        swap_to_be(&mut floats);
        assert_eq!(as_bytes(&floats), &[0x3f, 0x80, 0, 0]);
    }
}
