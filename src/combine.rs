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
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::beams::{self, Beam};
use crate::bounding_box::{BoundingBox, create_bound_box_plane, extract_common_bounding_box};
use crate::error::{FitsCubeError, Result};
use crate::fits_io::{
    CubeElem, CubeLayout, HeaderGeom, PixelType, create_mem_cube, delete_key,
    extract_header_layout, find_target_axis, has_key, update_key_f64, update_key_i64,
    update_key_logical, update_key_str, write_comment,
};
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

/// Options for [`combine_fits`], mirroring the keyword arguments of the Python
/// `combine_fits`.
#[derive(Debug, Clone, Default)]
pub struct CombineOptions {
    pub spec_file: Option<PathBuf>,
    pub spec_list: Option<Vec<f64>>,
    pub ignore_spec: bool,
    pub create_blanks: bool,
    pub overwrite: bool,
    pub max_workers: Option<usize>,
    pub time_domain_mode: bool,
    pub bounding_box: bool,
    pub invalidate_zeros: bool,
    /// Output floating-point precision in bits. Only 32 and 64 are valid FITS
    /// float widths (BITPIX −32 / −64); other values are rejected.
    pub float_length: Option<u8>,
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
    pixel_type: PixelType,
    /// Output plane length (NAXIS1 × NAXIS2 after any bounding-box trim).
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
    let mut fptr = create_mem_cube(template, out_bitpix, &dims)?;

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
            &format!("The value '{tiny}' repsenents a NaN PSF in the beamtable."),
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

    let plane_len = dims[0] * dims.get(1).copied().unwrap_or(1);
    let layout = extract_header_layout(&mut fptr)?;
    Ok((
        InitResult {
            pixel_type: PixelType::from_bitpix(out_bitpix),
            plane_len,
        },
        layout,
    ))
}

/// Read one input plane as type `T`, apply bounding box / zero-invalidation, and
/// return the flat (row-major) plane buffer ready for `write_section`.
fn process_plane<T: CubeElem + num_traits::Float>(
    path: &Path,
    bbox: Option<&BoundingBox>,
    invalidate_zeros: bool,
) -> Result<Vec<T>> {
    // Single open per plane. Only read the spatial dims (extra header keys) when
    // a bounding box actually needs them.
    let mut fptr = FitsFile::open(path.to_string_lossy().as_ref())?;
    let dims = if bbox.is_some() {
        let hdu = fptr.primary_hdu()?;
        // FITS order: NAXIS1 = cols (fast), NAXIS2 = rows.
        let ncols: i64 = hdu.read_key(&mut fptr, "NAXIS1")?;
        let nrows: i64 = hdu.read_key(&mut fptr, "NAXIS2")?;
        Some((nrows as usize, ncols as usize))
    } else {
        None
    };
    let flat: Vec<T> = T::read_full(&mut fptr)?;

    let mut plane: Vec<T> = if let Some(bb) = bbox {
        let (nrows, ncols) = dims.expect("dims read when bbox is set");
        let view: ArrayView2<T> = ArrayView2::from_shape((nrows, ncols), &flat)?;
        // Slice rows xmin:xmax, cols ymin:ymax (matches numpy `[..., x, y]`).
        let sub = view.slice(ndarray::s![bb.xmin..bb.xmax, bb.ymin..bb.ymax]);
        let owned: Array2<T> = sub.to_owned();
        owned.into_raw_vec_and_offset().0
    } else {
        flat
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

/// Stream all channels into the output cube using raw I/O.
///
/// Bypasses cfitsio for the data unit: the file is created with the prebuilt
/// header ([`CubeLayout`]) and sparsely extended to its final length, then each
/// decoded plane is byte-swapped to big-endian ([`BeBytes`]) and written at its
/// offset. This mirrors the Python reference (`astype(">f4")` + raw `tofile`),
/// which is markedly faster than cfitsio's per-block write path and never pays
/// the zero-fill pass cfitsio does on close.
///
/// Planes are decoded by the rayon pool (parallel readers) and written on this
/// single thread; `write_all_at` is positional, so out-of-order arrival is fine.
#[allow(clippy::too_many_arguments)]
fn write_cube_raw<T: CubeElem + num_traits::Float + BeBytes>(
    out_cube: &Path,
    layout: &CubeLayout,
    file_list: &[PathBuf],
    new_to_old: &[Option<usize>],
    plane_len: usize,
    bbox: Option<&BoundingBox>,
    invalidate_zeros: bool,
    max_workers: Option<usize>,
) -> Result<()> {
    let n_chan = new_to_old.len();
    let plane_bytes = (plane_len * T::WIDTH) as u64;

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
    let (tx, rx) = sync_channel::<(usize, Vec<T>)>(bound);

    std::thread::scope(|scope| -> Result<()> {
        let producer = scope.spawn(move || -> Result<()> {
            let res = (0..n_chan)
                .into_par_iter()
                .try_for_each(|new_chan| -> Result<()> {
                    let plane = match new_to_old[new_chan] {
                        Some(old) => process_plane::<T>(&file_list[old], bbox, invalidate_zeros)?,
                        None => vec![T::nan(); plane_len], // missing → blank plane
                    };
                    tx.send((new_chan, plane))
                        .map_err(|e| FitsCubeError::Other(format!("channel send failed: {e}")))?;
                    Ok(())
                });
            drop(tx); // close the channel so the writer loop below ends
            res
        });

        // Writer (this thread): byte-swap each plane and write it at its offset.
        let mut buf: Vec<u8> = Vec::with_capacity(plane_bytes as usize);
        for (chan, data) in rx {
            buf.clear();
            for v in &data {
                v.extend_be(&mut buf);
            }
            let offset = layout.datastart + chan as u64 * plane_bytes;
            file.write_all_at(&buf, offset)?;
        }

        producer
            .join()
            .map_err(|_| FitsCubeError::Other("reader thread panicked".to_string()))?
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

    let spec_info = parse_specs(
        file_list,
        options.spec_file.as_deref(),
        options.spec_list.as_deref(),
        options.ignore_spec,
        options.create_blanks,
        options.time_domain_mode,
    )?;

    // Beams (parsed in input order, matching the original).
    let has_beams = has_key(&file_list[0], "BMAJ")?;
    let (beams_vec, single_beam): (Option<Vec<Beam>>, bool) = if has_beams {
        let beams = beams::parse_beams(file_list)?;
        let single = beams::is_single_beam(&beams);
        (Some(beams), single)
    } else {
        (None, false)
    };

    // Sort files by their per-file value; sort the output axis independently.
    let old_sort = argsort(&spec_info.file_specs);
    let sorted_files: Vec<PathBuf> = old_sort.iter().map(|&i| file_list[i].clone()).collect();

    let new_sort = argsort(&spec_info.specs);
    let specs: Vec<f64> = new_sort.iter().map(|&i| spec_info.specs[i]).collect();
    let missing: Vec<bool> = new_sort.iter().map(|&i| spec_info.missing[i]).collect();

    // Optional common bounding box (computed from the sorted files).
    let final_bbox: Option<BoundingBox> = if options.bounding_box {
        let boxes: Vec<Option<BoundingBox>> = sorted_files
            .par_iter()
            .map(|p| -> Result<Option<BoundingBox>> {
                let plane = process_plane::<f64>(p, None, options.invalidate_zeros)?;
                let geom = HeaderGeom::read(p)?;
                let ncols = geom.dims.first().copied().unwrap_or(1);
                let nrows = geom.dims.get(1).copied().unwrap_or(1);
                let view = ArrayView2::from_shape((nrows, ncols), &plane)?;
                Ok(create_bound_box_plane(&view))
            })
            .collect::<Result<Vec<_>>>()?;
        let bb = extract_common_bounding_box(&boxes)?;
        tracing::info!("The final bounding box is: {bb:?}");
        Some(bb)
    } else {
        None
    };

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

    // Stream planes in the output precision.
    match init.pixel_type {
        PixelType::F32 => write_cube_raw::<f32>(
            out_cube,
            &layout,
            &sorted_files,
            &new_to_old,
            init.plane_len,
            final_bbox.as_ref(),
            options.invalidate_zeros,
            options.max_workers,
        )?,
        PixelType::F64 => write_cube_raw::<f64>(
            out_cube,
            &layout,
            &sorted_files,
            &new_to_old,
            init.plane_len,
            final_bbox.as_ref(),
            options.invalidate_zeros,
            options.max_workers,
        )?,
    }

    // Append the per-channel beam table when beams vary.
    if has_beams
        && !single_beam
        && let Some(beams) = beams_vec
    {
        let pol = beams::get_polarisation(&sorted_files[0])?;
        let mut fptr = FitsFile::edit(out_cube.to_string_lossy().as_ref())?;
        beams::write_beam_table(&mut fptr, &beams, pol)?;
    }

    Ok(specs)
}
