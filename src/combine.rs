//! Combine single-plane FITS images into a cube.
//!
//! The cube header is built in memory from the first input. Planes are read
//! and byte-swapped in parallel and written by one thread at their offsets in
//! a temporary file, which replaces the output once everything is written.
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::mpsc::sync_channel;

use atfits_rs::raw::{Pixel, as_bytes, disable_scaling, read_pixels, swap_to_be};
use atfits_rs::{
    CubeLayout, create_mem_cube, delete_key, extract_header_layout, key_f64, update_key_f64,
    update_key_i64, update_key_logical, update_key_str, write_comment,
};
use fitsio::FitsFile;
use num_traits::Float;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::beams::{self, Beam};
use crate::bounding_box::{BoundingBox, get_common_bounding_box};
use crate::checks::{
    InputHeader, check_matching_axes, check_matching_shapes, check_matching_storage,
    read_input_headers,
};
use crate::error::{FitsCubeError, Result};
use crate::partial::partial_file;
use crate::pixels::bbox_ranges;
use crate::progress::{progress_bar, spinner};
use crate::specs::parse_specs;

/// FITS headers and data units are padded to 2880-byte blocks.
const FITS_BLOCK: u64 = 2880;

/// Options for [`combine_fits`]. Field names follow fitscube's arguments.
#[derive(Debug, Clone)]
pub struct CombineOptions {
    /// Text file of frequencies (Hz) or times (MJD s), one per input.
    pub spec_file: Option<PathBuf>,
    /// Frequencies (Hz) or times (MJD s), one per input.
    pub spec_list: Option<Vec<f64>>,
    /// Stack in input order and label the axis with channel numbers.
    pub ignore_spec: bool,
    /// Fill gaps in a regular axis with NaN channels.
    pub create_blanks: bool,
    pub overwrite: bool,
    /// Threads used to read input images. `None` uses every core.
    pub max_workers: Option<usize>,
    /// Combine along time (DATE-OBS) instead of frequency.
    pub time_domain_mode: bool,
    /// Trim to the box around the finite pixels of all inputs.
    pub bounding_box: bool,
    /// Trim to this box. Takes precedence over `bounding_box`.
    pub supplied_bounding_box: Option<BoundingBox>,
    /// Treat pixels that are exactly zero as NaN.
    pub invalidate_zeros: bool,
    /// Output float width in bits, 32 or 64. `None` keeps the input BITPIX.
    pub float_length: Option<u8>,
    /// Draw progress bars on stderr.
    pub progress: bool,
    /// Blank (NaN) inputs whose beam has a zero axis.
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

/// Combine `file_list` into the cube at `out_cube` and return the values of
/// its axis (Hz, or MJD seconds in time mode).
pub fn combine_fits(
    file_list: &[PathBuf],
    out_cube: &Path,
    options: &CombineOptions,
) -> Result<Vec<f64>> {
    if file_list.is_empty() {
        return Err(FitsCubeError::NoInputs);
    }
    let float_bitpix = float_length_to_bitpix(options.float_length)?;
    for path in file_list {
        check_exists(path)?;
    }
    if out_cube.exists() && !options.overwrite {
        return Err(FitsCubeError::OutputExists(out_cube.to_path_buf()));
    }
    let pool = options
        .max_workers
        .map(|n| ThreadPoolBuilder::new().num_threads(n.max(1)).build())
        .transpose()?;
    let pool = pool.as_ref();

    let headers = in_pool(pool, || read_input_headers(file_list))?;
    check_matching_shapes(file_list, &headers)?;
    check_matching_axes(file_list, &headers)?;
    let first = &headers[0];
    let bitpix = float_bitpix.unwrap_or(first.storage.bitpix);
    if bitpix > 0 {
        check_matching_storage(file_list, &headers)?;
    }

    let channels = plan_channels(file_list, &headers, options)?;
    let bbox = bounding_box(file_list, first, options, pool)?;
    if let Some(bb) = &bbox {
        tracing::info!("The final bounding box is: {bb:?}");
    }

    let stokes_idx = beams::get_polarisation(first.stokes_codes.as_ref().map(Vec::len));
    let varying_beams = channels
        .beams
        .as_ref()
        .is_some_and(|b| !beams::is_single_beam(b));
    // Fail before any data is written
    if varying_beams {
        beams::beam_table_pol(&stokes_idx)?;
    }
    let has_blanks = channels.inputs.iter().any(Option::is_none);
    if bitpix > 0 && (has_blanks || options.invalidate_zeros) {
        return Err(FitsCubeError::BlanksNeedFloat(bitpix));
    }

    let header = CubeHeader {
        template: &file_list[0],
        first,
        specs: &channels.specs,
        ignore_spec: options.ignore_spec,
        varying_beams,
        time_domain_mode: options.time_domain_mode,
        bbox: bbox.as_ref(),
        bitpix,
    };
    let (layout, plane_len) = header.build()?;

    let partial = partial_file(out_cube)?;
    let file = File::options().read(true).write(true).open(&partial)?;
    let source = PlaneSource {
        files: file_list,
        headers: &headers,
        inputs: &channels.inputs,
        bbox: bbox.as_ref(),
        invalidate_zeros: options.invalidate_zeros,
        plane_len,
    };
    let writer = CubeWriter {
        file: &file,
        layout: &layout,
        n_chan: channels.inputs.len(),
        plane_len,
        pool,
        progress: options.progress,
    };
    match bitpix {
        -32 => writer.write(|c, buf| source.read_float::<f32>(c, buf))?,
        -64 => writer.write(|c, buf| source.read_float::<f64>(c, buf))?,
        8 => writer.write(|c, buf| source.read_raw::<u8>(c, buf))?,
        16 => writer.write(|c, buf| source.read_raw::<i16>(c, buf))?,
        32 => writer.write(|c, buf| source.read_raw::<i32>(c, buf))?,
        64 => writer.write(|c, buf| source.read_raw::<i64>(c, buf))?,
        other => return Err(FitsCubeError::UnsupportedBitpix(other)),
    }
    drop(file);

    if varying_beams && let Some(beams) = &channels.beams {
        let mut fptr = FitsFile::edit(&partial)?;
        beams::write_beam_table(&mut fptr, beams, &stokes_idx)?;
    }
    partial.persist(out_cube).map_err(|e| e.error)?;
    Ok(channels.specs)
}

fn float_length_to_bitpix(float_length: Option<u8>) -> Result<Option<i64>> {
    match float_length {
        None => Ok(None),
        Some(32) => Ok(Some(-32)),
        Some(64) => Ok(Some(-64)),
        Some(other) => Err(FitsCubeError::InvalidFloatLength(other)),
    }
}

/// A missing input as an io `NotFound` error, which cfitsio reports only as a
/// failure to open.
pub(crate) fn check_exists(path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("{} does not exist", path.display()),
    )
    .into())
}

/// Run `f` in `pool`, or in rayon's global pool when there is none.
pub(crate) fn in_pool<R: Send>(pool: Option<&ThreadPool>, f: impl FnOnce() -> R + Send) -> R {
    match pool {
        Some(pool) => pool.install(f),
        None => f(),
    }
}

/// The channels of the cube in axis order.
struct Channels {
    specs: Vec<f64>,
    /// Index of each channel's input, `None` for a NaN plane.
    inputs: Vec<Option<usize>>,
    /// Beam of each channel, if any input has one.
    beams: Option<Vec<Beam>>,
}

fn plan_channels(
    file_list: &[PathBuf],
    headers: &[InputHeader],
    options: &CombineOptions,
) -> Result<Channels> {
    let spec_info = parse_specs(
        file_list,
        headers,
        options.spec_file.as_deref(),
        options.spec_list.as_deref(),
        options.ignore_spec,
        options.create_blanks,
        options.time_domain_mode,
    )?;
    let file_order = argsort(&spec_info.file_specs);
    let chan_order = argsort(&spec_info.specs);
    let specs: Vec<f64> = chan_order.iter().map(|&i| spec_info.specs[i]).collect();
    let missing: Vec<bool> = chan_order.iter().map(|&i| spec_info.missing[i]).collect();

    let mut sorted_inputs = file_order.iter().copied();
    let mut inputs: Vec<Option<usize>> = missing
        .iter()
        .map(|&m| if m { None } else { sorted_inputs.next() })
        .collect();
    let placed = inputs.iter().flatten().count();
    if placed != file_list.len() {
        return Err(FitsCubeError::ChannelMissing(format!(
            "{placed} channels have an input, but there are {} inputs",
            file_list.len()
        )));
    }

    let beams = match beams::input_beams(headers) {
        None => None,
        Some(per_input) => {
            let sorted: Vec<Beam> = file_order.iter().map(|&i| per_input[i]).collect();
            let beams = beams::expand_beams(&sorted, &missing)?;
            Some(blank_zero_beams(
                beams,
                &mut inputs,
                options.blank_zero_beams,
            ))
        }
    };
    Ok(Channels {
        specs,
        inputs,
        beams,
    })
}

/// Warn about channels with a zero beam and, if `blank`, make them NaN planes.
fn blank_zero_beams(beams: Vec<Beam>, inputs: &mut [Option<usize>], blank: bool) -> Vec<Beam> {
    let zero = beams::find_zero_beams(&beams);
    let chans: Vec<usize> = (0..zero.len()).filter(|&c| zero[c]).collect();
    if chans.is_empty() {
        return beams;
    }
    let shown = match chans.len() {
        n if n > 10 => format!("{:?} (and {} more)", &chans[..10], n - 10),
        _ => format!("{chans:?}"),
    };
    if !blank {
        tracing::warn!(
            "Channels {shown} have a restoring beam of exactly zero, but \
             blank_zero_beams is off, so they are being kept as-is."
        );
        return beams;
    }
    tracing::warn!(
        "Channels {shown} have a restoring beam of exactly zero. These planes carry no \
         real PSF (e.g. a wsclean model image from -fit-spectral-pol) and are being \
         blanked with NaNs. Pass --no-blank-zero-beams (blank_zero_beams=False) to keep them."
    );
    for &c in &chans {
        inputs[c] = None;
    }
    beams::nan_zero_beams(&beams, &zero)
}

/// The box to trim the planes to, if any.
fn bounding_box(
    file_list: &[PathBuf],
    first: &InputHeader,
    options: &CombineOptions,
    pool: Option<&ThreadPool>,
) -> Result<Option<BoundingBox>> {
    if let Some(bb) = options.supplied_bounding_box {
        let (ncols, nrows) = first.shape();
        if bb.original_shape != (nrows, ncols) || bb.xmax > nrows || bb.ymax > ncols {
            return Err(FitsCubeError::ShapeMismatch(format!(
                "The supplied bounding box {bb:?} does not fit the input images, whose \
                 planes have shape (NAXIS2, NAXIS1)={:?}",
                (nrows, ncols)
            )));
        }
        return Ok(Some(bb));
    }
    if !options.bounding_box {
        return Ok(None);
    }
    let spin = options
        .progress
        .then(|| spinner("solving for common bounding box"));
    let bb = in_pool(pool, || {
        get_common_bounding_box(file_list, options.invalidate_zeros)
    })?;
    if let Some(spin) = spin {
        spin.finish_and_clear();
    }
    Ok(Some(bb))
}

/// Indices that sort `keys` ascending (stable).
fn argsort(keys: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..keys.len()).collect();
    idx.sort_by(|&a, &b| keys[a].total_cmp(&keys[b]));
    idx
}

fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    match v.len() {
        0 => f64::NAN,
        n if n % 2 == 1 => v[n / 2],
        n => 0.5 * (v[n / 2 - 1] + v[n / 2]),
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

/// Whether the axis is regular enough to describe with CRVAL and CDELT.
fn is_evenly_spaced(specs: &[f64], time_domain_mode: bool) -> bool {
    if specs.len() < 2 {
        return true;
    }
    let mut sorted = specs.to_vec();
    sorted.sort_by(f64::total_cmp);
    let diff: Vec<f64> = sorted.windows(2).map(|w| w[1] - w[0]).collect();

    if !time_domain_mode {
        return std_dev(&diff) < 1e-4;
    }
    // ASKAP scan times jitter, so allow uneven steps whose deviations cancel
    // out along the axis
    let mut cumsum = 0.0;
    let mut max_dev = 0.0_f64;
    for w in diff.windows(2) {
        cumsum += w[1] - w[0];
        max_dev = max_dev.max(cumsum.abs());
    }
    let mean_diff = diff.iter().sum::<f64>() / diff.len() as f64;
    diff.len() < 2 || max_dev < mean_diff * 0.02
}

/// What the cube header is built from.
struct CubeHeader<'a> {
    template: &'a Path,
    first: &'a InputHeader,
    specs: &'a [f64],
    ignore_spec: bool,
    varying_beams: bool,
    time_domain_mode: bool,
    bbox: Option<&'a BoundingBox>,
    bitpix: i64,
}

impl CubeHeader<'_> {
    /// The header bytes, and the number of pixels in one channel.
    fn build(&self) -> Result<(CubeLayout, usize)> {
        let first = self.first;
        let (ctype, unit) = if self.time_domain_mode {
            ("TIME", "s")
        } else {
            ("FREQ", "Hz")
        };
        let n_chan = self.specs.len();
        let even_spec = is_evenly_spaced(self.specs, self.time_domain_mode);
        if !even_spec {
            tracing::warn!(
                "{} are not evenly spaced; labelling the axis with channel numbers",
                if self.time_domain_mode {
                    "Times"
                } else {
                    "Frequencies"
                }
            );
        }

        // 1-based FITS number of the combine axis
        let fi = match first.ctypes.iter().position(|c| c.contains(ctype)) {
            _ if first.is_2d() => 3,
            Some(i) => i + 1,
            None => first.dims.len() + 1,
        };
        let mut dims = first.dims.clone();
        dims.resize(dims.len().max(fi), 1);
        dims[fi - 1] = n_chan;
        if let Some(bb) = self.bbox {
            dims[0] = bb.y_span;
            dims[1] = bb.x_span;
        }

        // Planes go in at (channel x plane size), so every axis above the
        // combine axis must have length 1
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

        let mut fptr = create_mem_cube(self.template, self.bitpix, &dims)?;
        let as_channels = self.ignore_spec || !even_spec;
        let (crval, cdelt) = if as_channels {
            (1.0, 1.0)
        } else if n_chan > 1 {
            let diffs: Vec<f64> = self.specs.windows(2).map(|w| w[1] - w[0]).collect();
            (self.specs[0], median(&diffs))
        } else {
            (self.specs[0], 1.0)
        };
        update_key_i64(&mut fptr, &format!("CRPIX{fi}"), 1)?;
        update_key_f64(&mut fptr, &format!("CRVAL{fi}"), crval)?;
        update_key_f64(&mut fptr, &format!("CDELT{fi}"), cdelt)?;
        if as_channels {
            delete_key(&mut fptr, &format!("CUNIT{fi}"))?;
            update_key_str(&mut fptr, &format!("CTYPE{fi}"), "CHAN")?;
        } else {
            update_key_str(&mut fptr, &format!("CUNIT{fi}"), unit)?;
            update_key_str(&mut fptr, &format!("CTYPE{fi}"), ctype)?;
        }
        if (first.has_cd || first.has_pc) && fi != 1 {
            let kind = if first.has_cd { "CD" } else { "PC" };
            update_key_f64(&mut fptr, &format!("{kind}{fi}_{fi}"), 1.0)?;
        }

        if self.varying_beams {
            update_key_logical(&mut fptr, "CASAMBM", true)?;
            write_comment(&mut fptr, "The PSF in each image plane varies.")?;
            write_comment(
                &mut fptr,
                "Full beam information is stored in the second FITS extension.",
            )?;
            write_comment(
                &mut fptr,
                &format!(
                    "The value '{}' represents a NaN PSF in the beamtable.",
                    f32::MIN_POSITIVE
                ),
            )?;
            for key in ["BMAJ", "BMIN", "BPA"] {
                delete_key(&mut fptr, key)?;
            }
        }

        if let Some(bb) = self.bbox {
            let crpix1 = key_f64(&mut fptr, "CRPIX1")?.unwrap_or(0.0);
            let crpix2 = key_f64(&mut fptr, "CRPIX2")?.unwrap_or(0.0);
            update_key_f64(&mut fptr, "CRPIX1", crpix1 - bb.ymin as f64)?;
            update_key_f64(&mut fptr, "CRPIX2", crpix2 - bb.xmin as f64)?;
        }

        let layout = extract_header_layout(&mut fptr, &dims)?;
        Ok((layout, dims[..fi - 1].iter().product()))
    }
}

/// Reads the image behind each channel.
struct PlaneSource<'a> {
    files: &'a [PathBuf],
    headers: &'a [InputHeader],
    inputs: &'a [Option<usize>],
    bbox: Option<&'a BoundingBox>,
    invalidate_zeros: bool,
    plane_len: usize,
}

impl PlaneSource<'_> {
    fn open(&self, i: usize) -> Result<FitsFile> {
        let mut fptr = FitsFile::open(&self.files[i])?;
        fptr.primary_hdu()?;
        Ok(fptr)
    }

    fn ranges(&self, i: usize) -> Option<Vec<(usize, usize)>> {
        self.bbox.map(|bb| bbox_ranges(&self.headers[i].dims, bb))
    }

    /// Channel `chan` decoded to `T`, with BLANK pixels and blank channels NaN.
    fn read_float<T: Pixel + Float>(&self, chan: usize, buf: &mut Vec<T>) -> Result<()> {
        let Some(i) = self.inputs[chan] else {
            buf.clear();
            buf.resize(self.plane_len, T::nan());
            return Ok(());
        };
        let mut fptr = self.open(i)?;
        // Only integer images need BLANK mapped; NaN checks slow float reads
        let s = self.headers[i].storage;
        let nulval = (s.bitpix > 0 && s.blank.is_some()).then(T::nan);
        read_pixels(&mut fptr, self.ranges(i).as_deref(), nulval, buf)?;
        if self.invalidate_zeros {
            for v in buf.iter_mut().filter(|v| v.is_zero()) {
                *v = T::nan();
            }
        }
        Ok(())
    }

    /// Channel `chan` as stored, without BSCALE/BZERO applied.
    fn read_raw<T: Pixel>(&self, chan: usize, buf: &mut Vec<T>) -> Result<()> {
        let i = self.inputs[chan]
            .ok_or_else(|| FitsCubeError::ChannelMissing(format!("no input for channel {chan}")))?;
        let mut fptr = self.open(i)?;
        disable_scaling(&mut fptr)?;
        read_pixels(&mut fptr, self.ranges(i).as_deref(), None, buf)?;
        Ok(())
    }
}

/// Writes the header and the planes of a cube to `file`.
struct CubeWriter<'a> {
    file: &'a File,
    layout: &'a CubeLayout,
    n_chan: usize,
    plane_len: usize,
    pool: Option<&'a ThreadPool>,
    progress: bool,
}

impl CubeWriter<'_> {
    /// Write each channel `c` as filled into a buffer by `read_plane(c, buf)`.
    fn write<T: Pixel>(
        &self,
        read_plane: impl Fn(usize, &mut Vec<T>) -> Result<()> + Sync,
    ) -> Result<()> {
        let plane_bytes = (self.plane_len * size_of::<T>()) as u64;
        let data_len = plane_bytes * self.n_chan as u64;
        write_at(self.file, &self.layout.header, 0)?;
        // Size the file without writing the data unit; the OS backs the gap
        // with zeros until the planes land
        self.file
            .set_len(self.layout.datastart + data_len.div_ceil(FITS_BLOCK) * FITS_BLOCK)?;

        let threads = self
            .pool
            .map_or_else(rayon::current_num_threads, ThreadPool::current_num_threads);
        let (tx, rx) = sync_channel::<(usize, Vec<T>)>(threads * 2);
        // Written buffers go back to the readers
        let spare: Mutex<Vec<Vec<T>>> = Mutex::new(Vec::new());

        std::thread::scope(|scope| -> Result<()> {
            let read_plane = &read_plane;
            let spare = &spare;
            let producer = scope.spawn(move || -> Result<()> {
                let res = in_pool(self.pool, || {
                    (0..self.n_chan).into_par_iter().try_for_each(|chan| {
                        let mut buf = spare
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .pop()
                            .unwrap_or_default();
                        read_plane(chan, &mut buf)?;
                        if buf.len() != self.plane_len {
                            return Err(FitsCubeError::ShapeMismatch(format!(
                                "channel {chan} has {} pixels, but the cube expects {}",
                                buf.len(),
                                self.plane_len
                            )));
                        }
                        swap_to_be(&mut buf);
                        tx.send((chan, buf)).map_err(|_| FitsCubeError::Cancelled)
                    })
                });
                drop(tx);
                res
            });

            let pb = self.progress.then(|| {
                let bar = progress_bar(self.n_chan as u64);
                bar.set_message("writing planes");
                bar
            });
            for (chan, buf) in rx {
                let offset = self.layout.datastart + chan as u64 * plane_bytes;
                write_at(self.file, as_bytes(&buf), offset)?;
                spare.lock().unwrap_or_else(|e| e.into_inner()).push(buf);
                if let Some(bar) = &pb {
                    bar.inc(1);
                }
            }
            if let Some(bar) = &pb {
                bar.finish_with_message("planes written");
            }
            producer.join().map_err(|_| FitsCubeError::WorkerPanic)?
        })
    }
}

#[cfg(unix)]
fn write_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buf, offset)
}

#[cfg(windows)]
fn write_at(file: &File, mut buf: &[u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_write(buf, offset)? {
            0 => return Err(std::io::ErrorKind::WriteZero.into()),
            n => {
                buf = &buf[n..];
                offset += n as u64;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_pool_limits_threads() {
        let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        assert_eq!(in_pool(Some(&pool), rayon::current_num_threads), 2);
        assert_eq!(
            in_pool(None, rayon::current_num_threads),
            rayon::current_num_threads()
        );
    }
}
