//! Low-level FITS helpers shared by the combine and extract pipelines.
//!
//! The original `fitscube` leans on astropy's WCS machinery; here we work
//! directly against cfitsio (via the `fitsio` crate, with a few raw `fitsio::sys`
//! calls where the safe API is insufficient). The two non-obvious needs are:
//!
//! * **Update-in-place keywords.** `fitsio`'s `write_key` calls `ffpky*`, which
//!   *appends* a card even when the keyword already exists, leaving duplicates
//!   that cfitsio then reads stale. We use `ffuky*` (`fits_update_key`) so the
//!   WCS cards we rewrite (CRVAL/CDELT/CTYPE/…) overwrite in place.
//! * **Resize an image HDU.** Promoting a 2D image to an N-D cube (and changing
//!   BITPIX) is done with `ffrsim` (`fits_resize_img`), which keeps every other
//!   header card — all the spatial WCS we want to preserve.
use std::ffi::CString;
use std::path::Path;

use fitsio::FitsFile;
use fitsio::errors::check_status;

use crate::error::{FitsCubeError, Result};

/// Pixel precision a cube is read/written in, derived from FITS `BITPIX`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelType {
    F32,
    F64,
}

impl PixelType {
    /// `-64` → f64; everything else (`-32` and integer BITPIX) → f32, matching
    /// the working precision of the original package.
    pub fn from_bitpix(bitpix: i64) -> Self {
        if bitpix == -64 {
            PixelType::F64
        } else {
            PixelType::F32
        }
    }
}

/// Pixel element types a plane can be streamed in: `f32` or `f64`.
///
/// Bundles the cfitsio section read/write so the streaming pipeline can be
/// generic over precision while keeping the cfitsio calls monomorphic.
pub trait CubeElem: Copy + Send + Sync + 'static {
    /// FITS `BITPIX` written for this element type.
    const BITPIX: i64;
    fn read_full(fptr: &mut FitsFile) -> Result<Vec<Self>>;
    fn read_section(fptr: &mut FitsFile, start: usize, end: usize) -> Result<Vec<Self>>;
    fn write_section(fptr: &mut FitsFile, start: usize, end: usize, data: &[Self]) -> Result<()>;
}

macro_rules! impl_cube_elem {
    ($t:ty, $bitpix:expr) => {
        impl CubeElem for $t {
            const BITPIX: i64 = $bitpix;
            fn read_full(fptr: &mut FitsFile) -> Result<Vec<Self>> {
                let hdu = fptr.primary_hdu()?;
                Ok(hdu.read_image(fptr)?)
            }
            fn read_section(fptr: &mut FitsFile, start: usize, end: usize) -> Result<Vec<Self>> {
                let hdu = fptr.primary_hdu()?;
                Ok(hdu.read_section(fptr, start, end)?)
            }
            fn write_section(
                fptr: &mut FitsFile,
                start: usize,
                end: usize,
                data: &[Self],
            ) -> Result<()> {
                let hdu = fptr.primary_hdu()?;
                hdu.write_section(fptr, start, end, data)?;
                Ok(())
            }
        }
    };
}
impl_cube_elem!(f32, -32);
impl_cube_elem!(f64, -64);

// ── Header geometry ─────────────────────────────────────────────────────────

/// Shape and pixel metadata of a FITS primary image, in FITS axis order
/// (`dims[0]` = NAXIS1, the fastest-varying axis).
#[derive(Debug, Clone)]
pub struct HeaderGeom {
    pub naxis: usize,
    /// `dims[i]` = NAXIS(i+1).
    pub dims: Vec<usize>,
    pub bitpix: i64,
}

impl HeaderGeom {
    /// Read NAXIS, every NAXISn, and BITPIX from the primary HDU.
    pub fn read(path: &Path) -> Result<Self> {
        let mut fptr = FitsFile::open(path.to_string_lossy().as_ref())?;
        let hdu = fptr.primary_hdu()?;
        let naxis: i64 = hdu.read_key(&mut fptr, "NAXIS")?;
        let bitpix: i64 = hdu.read_key(&mut fptr, "BITPIX")?;
        let mut dims = Vec::with_capacity(naxis as usize);
        for i in 1..=naxis {
            let n: i64 = hdu.read_key(&mut fptr, &format!("NAXIS{i}"))?;
            dims.push(n as usize);
        }
        Ok(Self {
            naxis: naxis as usize,
            dims,
            bitpix,
        })
    }

    /// Whether the image is two-dimensional (a single plane).
    pub fn is_2d(&self) -> bool {
        self.naxis == 2
    }

    /// Number of pixels per plane = NAXIS1 × NAXIS2.
    pub fn plane_len(&self) -> usize {
        self.dims.first().copied().unwrap_or(1) * self.dims.get(1).copied().unwrap_or(1)
    }
}

/// Location of a target (FREQ/TIME) axis within a header's WCS.
#[derive(Debug, Clone)]
pub struct TargetAxis {
    /// FITS axis number (1-based), e.g. 3 for NAXIS3.
    pub fits_idx: usize,
    /// numpy/array index (0-based, axis order reversed), as used by `np.take`.
    pub array_idx: usize,
    pub ctype: String,
    pub crpix: f64,
    pub crval: f64,
    pub cdelt: f64,
    pub cunit: Option<String>,
}

/// Search a header for the axis whose CTYPE contains `name` ("FREQ" or "TIME").
///
/// Mirrors `find_target_axis` / the WCS lookup in `create_output_cube`. Returns
/// [`FitsCubeError::TargetAxisMissing`] when absent.
pub fn find_target_axis(path: &Path, name: &str) -> Result<TargetAxis> {
    let mut fptr = FitsFile::open(path.to_string_lossy().as_ref())?;
    let hdu = fptr.primary_hdu()?;
    let naxis: i64 = hdu.read_key(&mut fptr, "NAXIS")?;

    for axis in 1..=naxis {
        let ctype: String = match hdu.read_key(&mut fptr, &format!("CTYPE{axis}")) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if ctype.contains(name) {
            let crpix: f64 = hdu
                .read_key(&mut fptr, &format!("CRPIX{axis}"))
                .unwrap_or(1.0);
            let crval: f64 = hdu
                .read_key(&mut fptr, &format!("CRVAL{axis}"))
                .unwrap_or(0.0);
            let cdelt: f64 = hdu
                .read_key(&mut fptr, &format!("CDELT{axis}"))
                .unwrap_or(1.0);
            let cunit: Option<String> = hdu.read_key(&mut fptr, &format!("CUNIT{axis}")).ok();
            return Ok(TargetAxis {
                fits_idx: axis as usize,
                array_idx: (naxis - axis) as usize,
                ctype,
                crpix,
                crval,
                cdelt,
                cunit,
            });
        }
    }
    Err(FitsCubeError::TargetAxisMissing(format!(
        "No {name} axis found in WCS of {}",
        path.display()
    )))
}

/// Read a header keyword as `f64`, or `None` if it is absent.
pub fn read_key_f64(path: &Path, key: &str) -> Result<Option<f64>> {
    let mut fptr = FitsFile::open(path.to_string_lossy().as_ref())?;
    let hdu = fptr.primary_hdu()?;
    Ok(hdu.read_key::<f64>(&mut fptr, key).ok())
}

/// Read a header keyword as `String`, or `None` if it is absent.
pub fn read_key_string(path: &Path, key: &str) -> Result<Option<String>> {
    let mut fptr = FitsFile::open(path.to_string_lossy().as_ref())?;
    let hdu = fptr.primary_hdu()?;
    Ok(hdu.read_key::<String>(&mut fptr, key).ok())
}

/// True if the keyword is present in the primary header.
pub fn has_key(path: &Path, key: &str) -> Result<bool> {
    let mut fptr = FitsFile::open(path.to_string_lossy().as_ref())?;
    let hdu = fptr.primary_hdu()?;
    Ok(hdu.read_key::<String>(&mut fptr, key).is_ok()
        || hdu.read_key::<f64>(&mut fptr, key).is_ok())
}

// ── Raw cfitsio header editing (update-in-place) ──────────────────────────────

fn cstr(s: &str) -> Result<CString> {
    CString::new(s).map_err(|e| FitsCubeError::Other(format!("invalid C string {s:?}: {e}")))
}

/// `fits_update_key_lng` — update/insert an integer keyword in place.
pub fn update_key_i64(fptr: &mut FitsFile, name: &str, value: i64) -> Result<()> {
    let c_name = cstr(name)?;
    let mut status = 0;
    unsafe {
        fitsio::sys::ffukyj(
            fptr.as_raw(),
            c_name.as_ptr(),
            value,
            std::ptr::null_mut(),
            &mut status,
        );
    }
    check_status(status)?;
    Ok(())
}

/// `fits_update_key_dbl` — update/insert a double keyword in place.
///
/// `decimals = -15` asks cfitsio for the shortest decimal that round-trips.
pub fn update_key_f64(fptr: &mut FitsFile, name: &str, value: f64) -> Result<()> {
    let c_name = cstr(name)?;
    let mut status = 0;
    unsafe {
        fitsio::sys::ffukyd(
            fptr.as_raw(),
            c_name.as_ptr(),
            value,
            -15,
            std::ptr::null_mut(),
            &mut status,
        );
    }
    check_status(status)?;
    Ok(())
}

/// `fits_update_key_str` — update/insert a string keyword in place.
pub fn update_key_str(fptr: &mut FitsFile, name: &str, value: &str) -> Result<()> {
    let c_name = cstr(name)?;
    let c_val = cstr(value)?;
    let mut status = 0;
    unsafe {
        fitsio::sys::ffukys(
            fptr.as_raw(),
            c_name.as_ptr(),
            c_val.as_ptr(),
            std::ptr::null_mut(),
            &mut status,
        );
    }
    check_status(status)?;
    Ok(())
}

/// `fits_update_key_log` — update/insert a logical (boolean) keyword in place.
pub fn update_key_logical(fptr: &mut FitsFile, name: &str, value: bool) -> Result<()> {
    let c_name = cstr(name)?;
    let mut status = 0;
    unsafe {
        fitsio::sys::ffukyl(
            fptr.as_raw(),
            c_name.as_ptr(),
            value as std::os::raw::c_int,
            std::ptr::null_mut(),
            &mut status,
        );
    }
    check_status(status)?;
    Ok(())
}

/// `fits_delete_key` — remove a keyword if present (absence is not an error).
pub fn delete_key(fptr: &mut FitsFile, name: &str) -> Result<()> {
    let c_name = cstr(name)?;
    let mut status = 0;
    unsafe {
        fitsio::sys::ffdkey(fptr.as_raw(), c_name.as_ptr(), &mut status);
    }
    // KEY_NO_EXIST (202) is fine — nothing to delete.
    if status == 202 {
        return Ok(());
    }
    check_status(status)?;
    Ok(())
}

/// `fits_write_comment` — append a COMMENT card.
pub fn write_comment(fptr: &mut FitsFile, comment: &str) -> Result<()> {
    let c_comment = cstr(comment)?;
    let mut status = 0;
    unsafe {
        fitsio::sys::ffpcom(fptr.as_raw(), c_comment.as_ptr(), &mut status);
    }
    check_status(status)?;
    Ok(())
}

/// `fits_resize_img` — change the BITPIX and shape of the primary image in
/// place, preserving all other header cards.
///
/// `dims` is in FITS order (`dims[0]` = NAXIS1). cfitsio zero-fills any new
/// pixels (sparsely) when the file is closed.
pub fn resize_image(fptr: &mut FitsFile, bitpix: i64, dims: &[usize]) -> Result<()> {
    let mut naxes: Vec<std::os::raw::c_long> =
        dims.iter().map(|&d| d as std::os::raw::c_long).collect();
    let mut status = 0;
    unsafe {
        fitsio::sys::ffrsim(
            fptr.as_raw(),
            bitpix as std::os::raw::c_int,
            naxes.len() as std::os::raw::c_int,
            naxes.as_mut_ptr(),
            &mut status,
        );
    }
    check_status(status)?;
    Ok(())
}

/// Keywords that describe the on-disk array structure; these are set by
/// `fits_create_img` for the new cube and must NOT be copied from the template.
fn is_structural_keyword(name: &str) -> bool {
    matches!(
        name,
        "SIMPLE"
            | "BITPIX"
            | "NAXIS"
            | "EXTEND"
            | "PCOUNT"
            | "GCOUNT"
            | "END"
            | "BSCALE"
            | "BZERO"
            | "BLANK"
    ) || (name.starts_with("NAXIS") && name[5..].chars().all(|c| c.is_ascii_digit()))
}

/// Map a FITS `BITPIX` to the `fitsio` image element type.
fn bitpix_to_image_type(bitpix: i64) -> fitsio::images::ImageType {
    use fitsio::images::ImageType;
    match bitpix {
        8 => ImageType::UnsignedByte,
        16 => ImageType::Short,
        32 => ImageType::Long,
        64 => ImageType::LongLong,
        -64 => ImageType::Double,
        _ => ImageType::Float, // -32 and anything unexpected
    }
}

/// Create `output` as a fresh image of shape `dims` (FITS order, NAXIS1 first)
/// and `bitpix`, copy every non-structural header card from `input`, and return
/// the **open** handle.
///
/// Keeping the handle open is the key to performance: cfitsio writes the (large)
/// data unit to disk only when the file is closed, so if the caller streams all
/// planes into this handle before closing, the data is written exactly once.
/// `copy_header_only` + [`resize_image`] instead closes after resizing, forcing
/// cfitsio to write a full pass of zeros that every plane then overwrites —
/// doubling the I/O for big cubes.
pub fn create_cube_open(
    input: &Path,
    output: &Path,
    bitpix: i64,
    dims: &[usize],
) -> Result<FitsFile> {
    use fitsio::images::ImageDescription;

    if let (Ok(in_canon), Ok(out_canon)) = (input.canonicalize(), output.canonicalize())
        && in_canon == out_canon
    {
        return Err(FitsCubeError::Other(format!(
            "output {} resolves to the input image",
            output.display()
        )));
    }
    if output.exists() {
        std::fs::remove_file(output)?;
    }

    // `ImageDescription::dimensions` is C-order (row-major), the reverse of the
    // FITS NAXIS order.
    let c_dims: Vec<usize> = dims.iter().rev().copied().collect();
    let desc = ImageDescription {
        data_type: bitpix_to_image_type(bitpix),
        dimensions: &c_dims,
    };
    let mut out = FitsFile::create(output)
        .with_custom_primary(&desc)
        .overwrite()
        .open()?;
    out.primary_hdu()?;

    // Copy every non-structural card from the template's primary header.
    let mut in_fptr = FitsFile::open(input.to_string_lossy().as_ref())?;
    in_fptr.primary_hdu()?;
    let mut status = 0;
    unsafe {
        let mut nkeys: std::os::raw::c_int = 0;
        let mut morekeys: std::os::raw::c_int = 0;
        fitsio::sys::ffghsp(in_fptr.as_raw(), &mut nkeys, &mut morekeys, &mut status);
        check_status(status)?;

        let mut card = [0i8; 81];
        for i in 1..=nkeys {
            card.fill(0);
            fitsio::sys::ffgrec(in_fptr.as_raw(), i, card.as_mut_ptr(), &mut status);
            if check_status(status).is_err() {
                break;
            }
            let card_str = std::ffi::CStr::from_ptr(card.as_ptr()).to_string_lossy();
            let name = card_str.split([' ', '=']).next().unwrap_or("").trim();
            if is_structural_keyword(name) {
                continue;
            }
            fitsio::sys::ffprec(out.as_raw(), card.as_ptr(), &mut status);
            check_status(status)?;
        }
    }
    Ok(out)
}

/// Create `output` containing only the primary-HDU header of `input` — no pixel
/// data read or copied.
///
/// Uses `ffinit` + `ffcphd`. Adapted from the convolve-rs implementation: a
/// plain `FitsFile::create` eagerly writes an empty NAXIS=0 primary, which makes
/// `ffcphd` a no-op; `ffinit` gives a truly empty file so the header copies into
/// the primary HDU directly.
pub fn copy_header_only(input: &Path, output: &Path) -> Result<()> {
    if let (Ok(in_canon), Ok(out_canon)) = (input.canonicalize(), output.canonicalize())
        && in_canon == out_canon
    {
        return Err(FitsCubeError::Other(format!(
            "output {} resolves to the input image",
            output.display()
        )));
    }

    let mut in_fptr = FitsFile::open(input.to_string_lossy().as_ref())?;
    in_fptr.primary_hdu()?;

    if output.exists() {
        std::fs::remove_file(output)?;
    }
    let out_name = cstr(output.to_string_lossy().as_ref())?;

    let mut status = 0;
    let mut raw_out: *mut fitsio::sys::fitsfile = std::ptr::null_mut();
    unsafe {
        fitsio::sys::ffinit(&mut raw_out, out_name.as_ptr(), &mut status);
        check_status(status)?;

        fitsio::sys::ffcphd(in_fptr.as_raw(), raw_out, &mut status);
        let copy_status = check_status(status);

        let mut close_status = 0;
        fitsio::sys::ffclos(raw_out, &mut close_status);
        copy_status?;
        check_status(close_status)?;
    }
    Ok(())
}
