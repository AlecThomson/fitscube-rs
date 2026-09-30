//! Combine single-plane FITS images into a cube.
//!
//! Port of `fitscube.combine_fits`. The pipeline: parse the spectral/temporal
//! axis ([`crate::specs`]), read beams ([`crate::beams`]), sort the inputs,
//! optionally compute a common bounding box, build the output header, then
//! stream each plane into the cube with raw I/O (reading/decoding in parallel,
//! writing on a single thread). The data unit bypasses cfitsio entirely to avoid
//! its zero-fill pass — see [`mem_header`] and [`write_cube_raw`].
use std::fs::OpenOptions;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::mpsc::sync_channel;

use fitsio::FitsFile;
use ndarray::ArrayView2;
use rayon::prelude::*;

use crate::beams::{self, Beam};
use crate::bounding_box::{BoundingBox, get_common_bounding_box};
use crate::checks::{check_matching_axes, check_matching_shapes, read_inputs_axes};
use crate::error::{FitsCubeError, Result};
use crate::fits_io::{
    CubeElem, CubeLayout, HeaderGeom, create_mem_cube, delete_key, extract_header_layout,
    find_target_axis, has_key, read_key_f64, update_key_f64, update_key_i64, update_key_logical,
    update_key_str, write_comment,
};
use crate::progress::{progress_bar, spinner};
use crate::specs::parse_specs;

/// FITS records are 2880 bytes; headers and the data unit are each padded up to
/// a whole number of these blocks.
const FITS_BLOCK: u64 = 2880;

fn round_up_to_block(n: u64) -> u64 {
    n.div_ceil(FITS_BLOCK) * FITS_BLOCK
}

/// A pixel value serialisable to big-endian FITS byte order.
///
/// The combine writer bypasses cfitsio for the data unit (see [`write_cube_raw`]),
/// so it byte-swaps each plane itself — exactly as the Python reference does with
/// `ndarray.astype(">f4")` before a raw `tofile`.
trait BeBytes: Copy {
    /// On-disk width in bytes (FITS `|BITPIX|/8`).
    const WIDTH: usize;
    fn extend_be(self, buf: &mut Vec<u8>);
}

impl BeBytes for f32 {
    const WIDTH: usize = 4;
    fn extend_be(self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_bits().to_be_bytes());
    }
}

impl BeBytes for f64 {
    const WIDTH: usize = 8;
    fn extend_be(self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_bits().to_be_bytes());
    }
}

macro_rules! impl_be_bytes_int {
    ($($t:ty),*) => {$(
        impl BeBytes for $t {
            const WIDTH: usize = std::mem::size_of::<$t>();
            fn extend_be(self, buf: &mut Vec<u8>) {
                buf.extend_from_slice(&self.to_be_bytes());
            }
        }
    )*};
}
impl_be_bytes_int!(i16, i32, i64);

/// Options for [`combine_fits`], mirroring the keyword arguments of the Python
/// `combine_fits`.
#[derive(Debug, Clone)]
pub struct CombineOptions {
    pub spec_file: Option<PathBuf>,
    pub spec_list: Option<Vec<f64>>,
    pub ignore_spec: bool,
    pub create_blanks: bool,
    pub overwrite: bool,
    pub max_workers: Option<usize>,
    pub time_domain_mode: bool,
    pub bounding_box: bool,
    /// A bounding box to trim to, used as is (it takes precedence over
    /// `bounding_box`), so that separate cubes — e.g. an image cube and its
    /// weights cube — can be forced onto a common pixel grid. See
    /// [`crate::bounding_box::get_common_bounding_box`].
    pub supplied_bounding_box: Option<BoundingBox>,
    pub invalidate_zeros: bool,
    /// Output floating-point precision in bits. Only 32 and 64 are valid FITS
    /// float widths (BITPIX −32 / −64); other values are rejected.
    pub float_length: Option<u8>,
    /// Draw progress bars/spinners to stderr. The CLI sets this; the Python
    /// bindings leave it off so importing the module stays silent.
    pub progress: bool,
    /// Blank (NaN) any input image whose restoring beam is exactly zero.
    /// wsclean writes such a beam when a plane holds no fitted PSF — e.g. the
    /// model image `-fit-spectral-pol` plants into a channel — and those planes
    /// are not comparable to the rest of the cube. Defaults to `true`.
    pub blank_zero_beams: bool,
}

impl Default for CombineOptions {
    fn default() -> Self {
        Self {
            spec_file: None,
            spec_list: None,
            ignore_spec: false,
            create_blanks: false,
            overwrite: false,
            max_workers: None,
            time_domain_mode: false,
            bounding_box: false,
            supplied_bounding_box: None,
            invalidate_zeros: false,
            float_length: None,
            progress: false,
            blank_zero_beams: true,
        }
    }
}

/// Validate `float_length` and map it to a FITS BITPIX, or `None` to inherit
/// the input precision.
fn float_length_to_bitpix(float_length: Option<u8>) -> Result<Option<i64>> {
    match float_length {
        None => Ok(None),
        Some(32) => Ok(Some(-32)),
        Some(64) => Ok(Some(-64)),
        Some(other) => Err(FitsCubeError::Other(format!(
            "floating={other} is not a valid FITS float precision; use 32 or 64 \
             (FITS defines only −32 and −64 bit IEEE floats)"
        ))),
    }
}

fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn std_dev(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    var.sqrt()
}

/// `argsort`: indices that would sort `keys` ascending (stable).
fn argsort(keys: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..keys.len()).collect();
    idx.sort_by(|&a, &b| keys[a].partial_cmp(&keys[b]).unwrap());
    idx
}

/// Decide whether the axis is evenly spaced enough to encode as a regular
/// CDELT, mirroring the `even_spec` test in `create_output_cube`.
fn is_evenly_spaced(specs: &[f64], time_domain_mode: bool) -> bool {
    if specs.len() < 2 {
        return true;
    }
    let mut sorted = specs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let diff: Vec<f64> = sorted.windows(2).map(|w| w[1] - w[0]).collect();

    if time_domain_mode {
        // Constrain the accumulated deviation of the second-order differences:
        // small running total ⇒ close enough to regular spacing to encode.
        if diff.len() < 2 {
            return true;
        }
        let diff_diff: Vec<f64> = diff.windows(2).map(|w| w[1] - w[0]).collect();
        let mut cumsum = 0.0;
        let mut max_dev = 0.0_f64;
        for d in &diff_diff {
            cumsum += d;
            max_dev = max_dev.max(cumsum.abs());
        }
        let mean_diff = diff.iter().sum::<f64>() / diff.len() as f64;
        max_dev < mean_diff * 0.02
    } else {
        std_dev(&diff) < 1e-4
    }
}

/// Where the spectral/temporal axis lives in the output cube.
struct AxisPlacement {
    fits_idx: usize,
}

/// Result of initialising the output cube.
struct InitResult {
    /// BITPIX of the output data unit.
    bitpix: i64,
    /// Pixels in one output channel: the product of every axis below the
    /// combine axis (NAXIS1 × NAXIS2 after any bounding-box trim, times any
    /// Stokes or other axes of the inputs).
    plane_len: usize,
}

/// Build the complete primary header for the output cube and return its on-disk
/// byte layout ([`CubeLayout`]) — no data unit is written here (see
/// [`crate::mem_header`]).
#[allow(clippy::too_many_arguments)]
fn create_output_cube(
    template: &Path,
    out_cube: &Path,
    specs: &[f64],
    ignore_spec: bool,
    has_beams: bool,
    single_beam: bool,
    overwrite: bool,
    time_domain_mode: bool,
    bbox: Option<&BoundingBox>,
    float_length: Option<u8>,
) -> Result<(InitResult, CubeLayout)> {
    if out_cube.exists() && !overwrite {
        return Err(FitsCubeError::OutputExists(out_cube.to_path_buf()));
    }

    let unit = if time_domain_mode { "s" } else { "Hz" };
    let ctype = if time_domain_mode { "TIME" } else { "FREQ" };

    let geom = HeaderGeom::read(template)?;
    let n_chan = specs.len();
    let even_spec = is_evenly_spaced(specs, time_domain_mode);
    if !even_spec {
        tracing::warn!(
            "{} are not evenly spaced; encoding axis as CHAN",
            if time_domain_mode {
                "Times"
            } else {
                "Frequencies"
            }
        );
    }

    // Locate the spectral axis (existing axis for a cube input, or a new one).
    let placement = if geom.is_2d() {
        AxisPlacement { fits_idx: 3 }
    } else {
        match find_target_axis(template, ctype) {
            Ok(axis) => AxisPlacement {
                fits_idx: axis.fits_idx,
            },
            Err(_) => AxisPlacement {
                fits_idx: geom.naxis + 1,
            },
        }
    };
    let fi = placement.fits_idx;

    // Output dimensions in FITS order (NAXIS1 fastest).
    let mut dims = geom.dims.clone();
    if geom.is_2d() {
        dims.push(n_chan); // new NAXIS3
    } else if fi <= dims.len() {
        dims[fi - 1] = n_chan;
    } else {
        dims.resize(fi, 1);
        dims[fi - 1] = n_chan;
    }
    if let Some(bb) = bbox {
        // NAXIS1 (fast/cols) ← y-span, NAXIS2 (slow/rows) ← x-span.
        dims[0] = bb.y_span;
        dims[1] = bb.x_span;
    }

    // Planes are written at an offset of (plane bytes × channel), which only
    // lands on the right plane when the combine axis is the slowest-varying one
    // — i.e. every axis above it in the cube is degenerate.
    let trailing: Vec<String> = dims
        .iter()
        .enumerate()
        .skip(fi)
        .filter(|&(_, &n)| n != 1)
        .map(|(i, n)| format!("NAXIS{}={n}", i + 1))
        .collect();
    if !trailing.is_empty() {
        return Err(FitsCubeError::AxisOrder(format!(
            "The {ctype} axis (NAXIS{fi}) must be the slowest-varying axis of the \
             output cube, but non-degenerate axes sit above it: {}. Reorder the axes \
             of the input images before combining.",
            trailing.join(", ")
        )));
    }

    let in_bitpix = geom.bitpix;
    let out_bitpix = float_length_to_bitpix(float_length)?.unwrap_or(in_bitpix);

    // Detect transform-matrix presence before we clobber anything.
    let has_cd = has_key(template, "CD1_1")?;
    let has_pc = has_key(template, "PC1_1")?;

    // Build the header in memory (no disk, so cfitsio never zero-fills the data
    // unit) at its final shape/BITPIX, copying the template's WCS cards. The
    // caller writes the header bytes and streams planes with raw I/O, so the data
    // unit is written exactly once and its untouched tail stays sparse — see
    // [`crate::mem_header`] and [`write_cube_raw`].
    // Build the header in memory (no disk, so cfitsio never zero-fills the data
    // unit) at its final shape/BITPIX. `create_mem_cube` allocates size-1 dummy
    // axes internally — `extract_header_layout` stamps the real NAXISn into the
    // serialised header — so no cube-sized buffer is ever allocated in RAM.
    let mut fptr = create_mem_cube(template, out_bitpix, &dims)?;

    // Integer cubes copy the inputs' stored values (see
    // [`process_plane_raw_int`]), so they keep the scaling that decodes them;
    // float cubes hold decoded values and must not.
    if out_bitpix > 0 {
        for key in ["BSCALE", "BZERO"] {
            if let Some(value) = read_key_f64(template, key)? {
                update_key_f64(&mut fptr, key, value)?;
            }
        }
    }

    // Spectral/temporal axis cards.
    update_key_i64(&mut fptr, &format!("CRPIX{fi}"), 1)?;
    update_key_f64(&mut fptr, &format!("CRVAL{fi}"), specs[0])?;
    let cdelt = if n_chan > 1 {
        let diffs: Vec<f64> = specs.windows(2).map(|w| w[1] - w[0]).collect();
        median(&diffs)
    } else {
        1.0
    };
    update_key_f64(&mut fptr, &format!("CDELT{fi}"), cdelt)?;
    update_key_str(&mut fptr, &format!("CUNIT{fi}"), unit)?;
    update_key_str(&mut fptr, &format!("CTYPE{fi}"), ctype)?;

    // Diagonal transform term for the new axis, for consistency.
    if (has_cd || has_pc) && fi != 1 {
        let kind = if has_cd { "CD" } else { "PC" };
        update_key_f64(&mut fptr, &format!("{kind}{fi}_{fi}"), 1.0)?;
    }

    // Unevenly spaced or ignored ⇒ encode a plain channel index.
    if ignore_spec || !even_spec {
        update_key_f64(&mut fptr, &format!("CDELT{fi}"), 1.0)?;
        delete_key(&mut fptr, &format!("CUNIT{fi}"))?;
        update_key_str(&mut fptr, &format!("CTYPE{fi}"), "CHAN")?;
        update_key_f64(&mut fptr, &format!("CRVAL{fi}"), 1.0)?;
    }

    // Varying beams ⇒ drop the single-beam keywords; the BEAMS table holds
    // the per-channel values.
    if has_beams && !single_beam {
        let tiny = f32::MIN_POSITIVE;
        update_key_logical(&mut fptr, "CASAMBM", true)?;
        write_comment(&mut fptr, "The PSF in each image plane varies.")?;
        write_comment(
            &mut fptr,
            "Full beam information is stored in the second FITS extension.",
        )?;
        write_comment(
            &mut fptr,
            &format!("The value '{tiny}' represents a NaN PSF in the beamtable."),
        )?;
        delete_key(&mut fptr, "BMAJ")?;
        delete_key(&mut fptr, "BMIN")?;
        delete_key(&mut fptr, "BPA")?;
    }

    // Bounding box shifts the spatial reference pixel.
    if let Some(bb) = bbox {
        let hdu = fptr.primary_hdu()?;
        let crpix1: f64 = hdu.read_key(&mut fptr, "CRPIX1").unwrap_or(1.0);
        let crpix2: f64 = hdu.read_key(&mut fptr, "CRPIX2").unwrap_or(1.0);
        update_key_f64(&mut fptr, "CRPIX1", crpix1 - bb.ymin as f64)?;
        update_key_f64(&mut fptr, "CRPIX2", crpix2 - bb.xmin as f64)?;
    }

    let layout = extract_header_layout(&mut fptr, &dims)?;
    Ok((
        InitResult {
            bitpix: out_bitpix,
            plane_len: dims[..fi - 1].iter().product(),
        },
        layout,
    ))
}

/// Slice every 2D plane of an input (rows xmin:xmax, cols ymin:ymax, matching
/// numpy `[..., x, y]`) down to the bounding box.
fn crop_to_box<T: Clone>(
    flat: &[T],
    nrows: usize,
    ncols: usize,
    bb: &BoundingBox,
) -> Result<Vec<T>> {
    let mut out = Vec::with_capacity(flat.len() / (nrows * ncols) * bb.x_span * bb.y_span);
    for plane in flat.chunks(nrows * ncols) {
        let view = ArrayView2::from_shape((nrows, ncols), plane)?;
        let sub = view.slice(ndarray::s![bb.xmin..bb.xmax, bb.ymin..bb.ymax]);
        out.extend(sub.iter().cloned());
    }
    Ok(out)
}

/// NAXIS2 (rows) and NAXIS1 (cols) of the open image, for a bounding-box crop.
fn plane_dims(fptr: &mut FitsFile) -> Result<(usize, usize)> {
    let hdu = fptr.primary_hdu()?;
    // FITS order: NAXIS1 = cols (fast), NAXIS2 = rows.
    let ncols: i64 = hdu.read_key(fptr, "NAXIS1")?;
    let nrows: i64 = hdu.read_key(fptr, "NAXIS2")?;
    Ok((nrows as usize, ncols as usize))
}

/// Read one input image as type `T`, apply bounding box / zero-invalidation, and
/// return the flat (row-major) buffer of every plane in it.
fn process_plane<T: CubeElem + num_traits::Float>(
    path: &Path,
    bbox: Option<&BoundingBox>,
    invalidate_zeros: bool,
) -> Result<Vec<T>> {
    // Single open per plane. Only read the spatial dims (extra header keys) when
    // a bounding box actually needs them.
    let mut fptr = FitsFile::open(path.to_string_lossy().as_ref())?;
    let dims = bbox.map(|_| plane_dims(&mut fptr)).transpose()?;
    let flat: Vec<T> = T::read_full(&mut fptr)?;

    let mut plane: Vec<T> = match (bbox, dims) {
        (Some(bb), Some((nrows, ncols))) => crop_to_box(&flat, nrows, ncols, bb)?,
        _ => flat,
    };

    if invalidate_zeros {
        let zero = T::zero();
        let nan = T::nan();
        for v in &mut plane {
            if *v == zero {
                *v = nan;
            }
        }
    }
    Ok(plane)
}

/// Read one integer input image as its raw stored values (BSCALE/BZERO not
/// applied), cropped to any bounding box.
///
/// The cube keeps the first input's BITPIX, BSCALE and BZERO, so copying the
/// stored integers is lossless, where decoding through floats and re-encoding
/// would not be.
fn process_plane_raw_int(path: &Path, bbox: Option<&BoundingBox>) -> Result<Vec<i64>> {
    let mut fptr = FitsFile::open(path.to_string_lossy().as_ref())?;
    let dims = plane_dims(&mut fptr)?;
    let n: usize = HeaderGeom::read(path)?.dims.iter().product();
    let mut flat = vec![0i64; n];
    let mut anynul = 0;
    let mut status = 0;
    // SAFETY: `fptr` is an open image positioned on its primary HDU, and `flat`
    // holds exactly the `n` elements requested.
    unsafe {
        fitsio::sys::ffpscl(fptr.as_raw(), 1.0, 0.0, &mut status);
        fitsio::sys::ffgpvjj(
            fptr.as_raw(),
            0,
            1,
            n as _,
            0,
            flat.as_mut_ptr(),
            &mut anynul,
            &mut status,
        );
    }
    if status != 0 {
        return Err(FitsCubeError::Other(format!(
            "cfitsio error {status} reading raw integers from {}",
            path.display()
        )));
    }
    match bbox {
        Some(bb) => crop_to_box(&flat, dims.0, dims.1, bb),
        None => Ok(flat),
    }
}

/// Serialise a plane to big-endian FITS bytes.
fn encode_be<T: BeBytes>(data: &[T]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(data.len() * T::WIDTH);
    for &v in data {
        v.extend_be(&mut buf);
    }
    buf
}

/// Serialise raw integers to big-endian FITS bytes at the width of `bitpix`.
fn encode_int_be(data: &[i64], bitpix: i64) -> Vec<u8> {
    match bitpix {
        8 => data.iter().map(|&v| v as u8).collect(),
        16 => encode_be(&data.iter().map(|&v| v as i16).collect::<Vec<_>>()),
        32 => encode_be(&data.iter().map(|&v| v as i32).collect::<Vec<_>>()),
        _ => encode_be(data),
    }
}

/// Stream all channels into the output cube using raw I/O.
///
/// Bypasses cfitsio for the data unit: the file is created with the prebuilt
/// header ([`CubeLayout`]) and sparsely extended to its final length, then each
/// channel's big-endian bytes from `encode_channel` are written at its offset.
/// This mirrors the Python reference (`astype(">f4")` + raw `tofile`), which is
/// markedly faster than cfitsio's per-block write path and never pays the
/// zero-fill pass cfitsio does on close.
///
/// Channels are decoded and byte-swapped by the rayon pool (parallel readers)
/// and written on this single thread; `write_all_at` is positional, so
/// out-of-order arrival is fine.
fn write_cube_raw(
    out_cube: &Path,
    layout: &CubeLayout,
    n_chan: usize,
    plane_bytes: u64,
    encode_channel: impl Fn(usize) -> Result<Vec<u8>> + Sync,
    max_workers: Option<usize>,
    progress: bool,
) -> Result<()> {
    // Lay down the header and size the file. `set_len` past the header leaves the
    // data unit (and its 2880-padded tail) sparse — zero-backed on demand — so no
    // zeros are physically written; the planes below cover the real data.
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(out_cube)?;
    file.write_all_at(&layout.header, 0)?;
    let data_len = plane_bytes * n_chan as u64;
    file.set_len(layout.datastart + round_up_to_block(data_len))?;

    // Buffer enough decoded planes that the parallel readers stay ahead of the
    // single (serial) writer instead of blocking on a tiny queue.
    let default_bound = std::thread::available_parallelism()
        .map(|n| n.get() * 2)
        .unwrap_or(8);
    let bound = max_workers.unwrap_or(default_bound).max(1);
    let (tx, rx) = sync_channel::<(usize, Vec<u8>)>(bound);

    std::thread::scope(|scope| -> Result<()> {
        let encode_channel = &encode_channel;
        let producer = scope.spawn(move || -> Result<()> {
            let res = (0..n_chan)
                .into_par_iter()
                .try_for_each(|new_chan| -> Result<()> {
                    let bytes = encode_channel(new_chan)?;
                    if bytes.len() as u64 != plane_bytes {
                        return Err(FitsCubeError::Other(format!(
                            "channel {new_chan} holds {} bytes, but the cube expects {plane_bytes}",
                            bytes.len()
                        )));
                    }
                    tx.send((new_chan, bytes))
                        .map_err(|e| FitsCubeError::Other(format!("channel send failed: {e}")))?;
                    Ok(())
                });
            drop(tx); // close the channel so the writer loop below ends
            res
        });

        // Writer (this thread): write each channel at its offset.
        let pb = progress.then(|| {
            let bar = progress_bar(n_chan as u64);
            bar.set_message("writing planes");
            bar
        });
        for (chan, bytes) in rx {
            let offset = layout.datastart + chan as u64 * plane_bytes;
            file.write_all_at(&bytes, offset)?;
            if let Some(bar) = &pb {
                bar.inc(1);
            }
        }
        if let Some(bar) = &pb {
            bar.finish_with_message("planes written");
        }

        producer
            .join()
            .map_err(|_| FitsCubeError::Other("reader thread panicked".to_string()))?
    })
}

/// Encode one float channel: the input plane in `T`, or NaNs for a blank one.
fn encode_float_channel<T: CubeElem + num_traits::Float + BeBytes>(
    input: Option<&Path>,
    plane_len: usize,
    bbox: Option<&BoundingBox>,
    invalidate_zeros: bool,
) -> Result<Vec<u8>> {
    Ok(match input {
        Some(path) => encode_be(&process_plane::<T>(path, bbox, invalidate_zeros)?),
        None => encode_be(&vec![T::nan(); plane_len]),
    })
}

/// Combine `file_list` into the cube at `out_cube`. Returns the output-axis
/// values (Hz for frequency mode, MJD seconds for time mode).
pub fn combine_fits(
    file_list: &[PathBuf],
    out_cube: &Path,
    options: &CombineOptions,
) -> Result<Vec<f64>> {
    if file_list.is_empty() {
        return Err(FitsCubeError::Other("file_list is empty".to_string()));
    }
    // Validate precision early.
    float_length_to_bitpix(options.float_length)?;

    // The cube's header comes from the first input, so every input must share
    // its pixel grid, axes and Stokes parameter.
    let input_axes = read_inputs_axes(file_list)?;
    check_matching_shapes(file_list, &input_axes)?;
    check_matching_axes(file_list, &input_axes)?;

    let spec_info = parse_specs(
        file_list,
        options.spec_file.as_deref(),
        options.spec_list.as_deref(),
        options.ignore_spec,
        options.create_blanks,
        options.time_domain_mode,
    )?;

    // Sort files by their per-file value; sort the output axis independently.
    let old_sort = argsort(&spec_info.file_specs);
    let sorted_files: Vec<PathBuf> = old_sort.iter().map(|&i| file_list[i].clone()).collect();

    let new_sort = argsort(&spec_info.specs);
    let specs: Vec<f64> = new_sort.iter().map(|&i| spec_info.specs[i]).collect();
    let missing: Vec<bool> = new_sort.iter().map(|&i| spec_info.missing[i]).collect();

    // Beams are parsed after sorting so that they follow the channel order of
    // the output cube, not the order the files were given in. Any input with a
    // beam counts, not just the first.
    let has_beams = beams::check_for_any_beam(&sorted_files)?;
    let mut zero_beam = vec![false; specs.len()];
    let (beams_vec, single_beam): (Option<Vec<Beam>>, bool) = if has_beams {
        let mut beams = beams::expand_beams(&beams::parse_beams(&sorted_files)?, &missing);
        let found_zero_beams = beams::find_zero_beams(&beams);
        if found_zero_beams.iter().any(|&z| z) {
            // Keep the message readable when a whole run has zero beams
            let all_zero_chans: Vec<usize> = found_zero_beams
                .iter()
                .enumerate()
                .filter_map(|(chan, &z)| z.then_some(chan))
                .collect();
            let zero_chans = if all_zero_chans.len() > 10 {
                format!(
                    "{:?} (and {} more)",
                    &all_zero_chans[..10],
                    all_zero_chans.len() - 10
                )
            } else {
                format!("{all_zero_chans:?}")
            };
            if options.blank_zero_beams {
                tracing::warn!(
                    "Channels {zero_chans} have a restoring beam of exactly zero. These \
                     planes carry no real PSF (e.g. a wsclean model image from \
                     -fit-spectral-pol) and are being blanked with NaNs. Pass \
                     --no-blank-zero-beams (blank_zero_beams=False) to keep them."
                );
                beams = beams::nan_zero_beams(&beams, &found_zero_beams);
                zero_beam = found_zero_beams;
            } else {
                tracing::warn!(
                    "Channels {zero_chans} have a restoring beam of exactly zero, but \
                     blank_zero_beams is off, so they are being kept as-is."
                );
            }
        }
        let single = beams::is_single_beam(&beams);
        (Some(beams), single)
    } else {
        (None, false)
    };

    // Optional common bounding box. A caller supplied box is used as is, so that
    // separate cubes can be forced onto a common pixel grid.
    let final_bbox: Option<BoundingBox> = if let Some(bb) = options.supplied_bounding_box {
        let (ncols, nrows) = input_axes[0].shape;
        let grid = (nrows as usize, ncols as usize);
        if bb.original_shape != grid || bb.xmax > grid.0 || bb.ymax > grid.1 {
            return Err(FitsCubeError::ShapeMismatch(format!(
                "The supplied bounding box {bb:?} does not fit the input images, whose \
                 planes have shape (NAXIS2, NAXIS1)={grid:?}"
            )));
        }
        Some(bb)
    } else if options.bounding_box {
        let spin = options
            .progress
            .then(|| spinner("solving for common bounding box"));
        let bb = get_common_bounding_box(&sorted_files, options.invalidate_zeros)?;
        if let Some(spin) = spin {
            spin.finish_and_clear();
        }
        Some(bb)
    } else {
        None
    };
    if let Some(bb) = &final_bbox {
        tracing::info!("The final bounding box is: {bb:?}");
    }

    // Lay out the beam table's POL column before writing any data, so an
    // unsupported cube (e.g. multi-Stokes) fails before the planes are copied
    // in. Stokes is never the combine axis, so the cube keeps the inputs' one.
    let stokes_idx = beams::get_polarisation(input_axes[0].stokes_codes.as_ref().map(Vec::len));
    if has_beams && !single_beam {
        beams::beam_table_pol(&stokes_idx)?;
    }

    // Build the output header (in memory) and its on-disk byte layout.
    let (init, layout) = create_output_cube(
        &sorted_files[0],
        out_cube,
        &specs,
        options.ignore_spec,
        has_beams,
        single_beam,
        options.overwrite,
        options.time_domain_mode,
        final_bbox.as_ref(),
        options.float_length,
    )?;

    // Map each output channel to an input index (None ⇒ blank/missing plane).
    let mut new_to_old: Vec<Option<usize>> = Vec::with_capacity(specs.len());
    let mut next_old = 0usize;
    for &is_missing in &missing {
        if is_missing {
            new_to_old.push(None);
        } else {
            new_to_old.push(Some(next_old));
            next_old += 1;
        }
    }
    if next_old != sorted_files.len() {
        return Err(FitsCubeError::ChannelMissing(format!(
            "channel/file count mismatch: {} present channels for {} files",
            next_old,
            sorted_files.len()
        )));
    }

    // Blank channels, zero-beam planes and invalidated zeros are all NaNs,
    // which an integer cube cannot hold.
    let blank_channels = missing.iter().zip(&zero_beam).any(|(&m, &z)| m || z);
    if init.bitpix > 0 && (blank_channels || options.invalidate_zeros) {
        return Err(FitsCubeError::Other(format!(
            "Blank channels and invalidated zeros are written as NaNs, which integer \
             output data (BITPIX={}) cannot hold. Pass float_length=32 or 64 (--floating).",
            init.bitpix
        )));
    }

    // Input image of each output channel; None ⇒ a blank (NaN) plane.
    let inputs: Vec<Option<&Path>> = new_to_old
        .iter()
        .zip(&zero_beam)
        .map(|(old, &z)| old.filter(|_| !z).map(|i| sorted_files[i].as_path()))
        .collect();
    let bbox = final_bbox.as_ref();
    let plane_len = init.plane_len;
    let width = (init.bitpix.unsigned_abs() / 8) as usize;
    let plane_bytes = (plane_len * width) as u64;
    let write = |encode: &(dyn Fn(usize) -> Result<Vec<u8>> + Sync)| {
        write_cube_raw(
            out_cube,
            &layout,
            inputs.len(),
            plane_bytes,
            encode,
            options.max_workers,
            options.progress,
        )
    };

    // Stream planes in the output precision.
    match init.bitpix {
        -32 => write(&|chan| {
            encode_float_channel::<f32>(inputs[chan], plane_len, bbox, options.invalidate_zeros)
        })?,
        -64 => write(&|chan| {
            encode_float_channel::<f64>(inputs[chan], plane_len, bbox, options.invalidate_zeros)
        })?,
        8 | 16 | 32 | 64 => write(&|chan| {
            // Integer cubes never have blank channels (checked above)
            let path = inputs[chan].expect("integer cubes have no blank channels");
            Ok(encode_int_be(
                &process_plane_raw_int(path, bbox)?,
                init.bitpix,
            ))
        })?,
        other => {
            return Err(FitsCubeError::Other(format!(
                "unsupported output BITPIX={other}"
            )));
        }
    }

    // Append the per-channel beam table when beams vary.
    if has_beams
        && !single_beam
        && let Some(beams) = beams_vec
    {
        let mut fptr = FitsFile::edit(out_cube.to_string_lossy().as_ref())?;
        beams::write_beam_table(&mut fptr, &beams, &stokes_idx)?;
    }

    Ok(specs)
}
