//! Compress a finished cube with gzip or FITS tile compression (as fpack
//! does), and turn a compressed cube back into plain FITS.
use std::ffi::{CStr, c_char, c_int, c_void};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use atfits_rs::{
    Pixel, as_bytes, disable_scaling, extract_header_layout, image_shape, is_structural_keyword,
    key_f64, key_i64, key_string, read_pixels, swap_to_be, update_key_f64, update_key_logical,
};
use fitsio::FitsFile;
use fitsio::errors::check_status;
use fitsio::sys::{self, LONGLONG};
use flate2::Compression;
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use num_traits::Float;
use rayon::prelude::*;

use crate::combine::in_pool;
use crate::error::{FitsCubeError, Result};
use crate::partial::partial_file;
use crate::pixels::with_pixel_type;
use crate::tile::{
    TileCodec, TileSettings, copy_other_hdus, has_compressed_hdu, image_kind, tile_compress,
};

/// How to compress a cube.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum CompressMethod {
    /// Gzip the whole file to `.fits.gz`.
    #[default]
    Gzip,
    /// As `Gzip`, compressing chunks across threads.
    Pgzip,
    /// Lossless GZIP_2 tile compression to `.fits.fz`.
    TileGzip,
    /// Lossy RICE_1 tile compression to `.fits.fz`, after quantising.
    TileRice,
}

/// Options for [`compress_cube`].
#[derive(Clone, Debug)]
pub struct CompressOptions {
    pub method: CompressMethod,
    /// Threads for `Pgzip` and the tile methods. Defaults to one per core.
    pub max_workers: Option<usize>,
    /// Quantisation steps per noise sigma for `TileRice`.
    pub quantize_level: f32,
    /// (rows, columns) of a tile. Defaults to bands of 64 full-width rows.
    pub tile_shape: Option<(usize, usize)>,
    /// ZDITHER0, 1 to 10000. Defaults to one derived from the data.
    pub dither_seed: Option<u32>,
    /// zlib level of `TileGzip` tiles.
    pub gzip_level: u32,
    /// Let `TileRice` quantise a cube that was already quantised once.
    pub allow_requantize: bool,
    /// Delete the input once the compressed cube is written.
    pub remove: bool,
}

impl Default for CompressOptions {
    fn default() -> Self {
        Self {
            method: CompressMethod::Gzip,
            max_workers: None,
            quantize_level: 4.0,
            tile_shape: None,
            dither_seed: None,
            gzip_level: 6,
            allow_requantize: false,
            remove: true,
        }
    }
}

/// Options for [`decompress_cube`].
#[derive(Clone, Debug, Default)]
pub struct DecompressOptions {
    /// Where to write. Defaults to the input without its `.gz`/`.fz` suffix.
    pub output: Option<PathBuf>,
    pub overwrite: bool,
    /// Delete the compressed input once the output is written.
    pub remove: bool,
}

/// Whether a cube is gzipped or tile-compressed.
pub fn is_compressed(cube: &Path) -> Result<bool> {
    Ok(has_compressed_suffix(cube) || has_compressed_hdu(cube)?)
}

fn has_compressed_suffix(path: &Path) -> bool {
    path.extension().is_some_and(|e| e == "gz" || e == "fz")
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// Compress `cube` to `cube.gz` or, for the tile methods, `cube.fz`. Returns
/// the new path.
pub fn compress_cube(cube: &Path, opts: &CompressOptions) -> Result<PathBuf> {
    let pool = opts
        .max_workers
        .map(|n| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n.max(1))
                .build()
        })
        .transpose()?;
    let output = in_pool(pool.as_ref(), || match opts.method {
        CompressMethod::Gzip | CompressMethod::Pgzip => gzip_cube(cube, opts),
        CompressMethod::TileGzip | CompressMethod::TileRice => tile_compress_cube(cube, opts),
    })?;
    if opts.remove {
        std::fs::remove_file(cube)?;
    }
    Ok(output)
}

fn gzip_cube(cube: &Path, opts: &CompressOptions) -> Result<PathBuf> {
    let output = with_suffix(cube, ".gz");
    tracing::info!(
        "Compressing {} to {} via {:?}",
        cube.display(),
        output.display(),
        opts.method
    );
    let partial = partial_file(&output)?;
    let mut src = File::open(cube)?;
    let mut dst = BufWriter::new(File::create(&partial)?);
    // Python's gzip.open defaults to level 9
    let level = Compression::best();
    if opts.method == CompressMethod::Pgzip {
        pgzip(&mut src, &mut dst, level)?;
    } else {
        let mut gz = GzEncoder::new(dst, level);
        std::io::copy(&mut BufReader::new(src), &mut gz)?;
        dst = gz.finish()?;
    }
    dst.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    partial.persist(&output).map_err(|e| e.error)?;
    Ok(output)
}

/// Gzip `src` as a series of gzip members, compressed in parallel. Any gzip
/// reader reads them back as one stream.
fn pgzip(src: &mut File, dst: &mut impl Write, level: Compression) -> Result<()> {
    let chunk = 8 << 20;
    let batch = rayon::current_num_threads();
    let mut written = false;
    loop {
        let mut chunks = Vec::with_capacity(batch);
        for _ in 0..batch {
            let mut buf = Vec::with_capacity(chunk);
            Read::by_ref(src).take(chunk as u64).read_to_end(&mut buf)?;
            if buf.is_empty() {
                break;
            }
            chunks.push(buf);
        }
        if chunks.is_empty() {
            // An empty input still needs one member to be valid gzip
            if !written {
                dst.write_all(&GzEncoder::new(Vec::new(), level).finish()?)?;
            }
            return Ok(());
        }
        let members = chunks
            .par_iter()
            .map(|c| -> Result<Vec<u8>> {
                let mut gz = GzEncoder::new(Vec::with_capacity(c.len() / 2), level);
                gz.write_all(c)?;
                Ok(gz.finish()?)
            })
            .collect::<Result<Vec<_>>>()?;
        for member in &members {
            dst.write_all(member)?;
        }
        written = true;
    }
}

fn tile_compress_cube(cube: &Path, opts: &CompressOptions) -> Result<PathBuf> {
    let rice = opts.method == CompressMethod::TileRice;
    if rice && opts.quantize_level <= 0.0 {
        return Err(FitsCubeError::RiceNeedsQuantize);
    }
    if is_compressed(cube)? {
        return Err(FitsCubeError::AlreadyCompressed(cube.to_path_buf()));
    }
    let (bitpix, lossy) = image_kind(cube)?;
    if bitpix > 0 {
        return Err(FitsCubeError::IntegerTiles {
            path: cube.to_path_buf(),
            bitpix,
        });
    }
    if rice && lossy && !opts.allow_requantize {
        return Err(FitsCubeError::AlreadyQuantized(cube.to_path_buf()));
    }

    let output = with_suffix(cube, ".fz");
    tracing::info!(
        "Tile-compressing {} to {} via {:?}",
        cube.display(),
        output.display(),
        opts.method
    );
    let settings = TileSettings {
        codec: if rice {
            TileCodec::Rice1 {
                level: opts.quantize_level,
            }
        } else {
            TileCodec::Gzip2 {
                level: opts.gzip_level,
            }
        },
        tile_shape: opts.tile_shape,
        dither_seed: opts.dither_seed,
    };
    let partial = partial_file(&output)?;
    tile_compress(cube, &partial, &settings, rayon::current_num_threads())?;
    partial.persist(&output).map_err(|e| e.error)?;
    Ok(output)
}

/// Write a gzipped or tile-compressed cube back out as plain FITS, a plane at
/// a time. A cube that was quantised is marked `FCLOSSY = T`, so that later
/// compression knows not to quantise it again.
pub fn decompress_cube(cube: &Path, opts: &DecompressOptions) -> Result<PathBuf> {
    let output = match &opts.output {
        Some(output) => output.clone(),
        None if has_compressed_suffix(cube) => cube.with_extension(""),
        None => return Err(FitsCubeError::NoOutputName(cube.to_path_buf())),
    };
    if output.exists() && !opts.overwrite {
        return Err(FitsCubeError::OutputExists(output));
    }
    tracing::info!("Decompressing {} to {}", cube.display(), output.display());
    let partial = partial_file(&output)?;
    if cube.extension().is_some_and(|e| e == "gz") {
        let mut src = MultiGzDecoder::new(BufReader::new(File::open(cube)?));
        let mut dst = BufWriter::new(File::create(&partial)?);
        std::io::copy(&mut src, &mut dst)?;
        dst.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    } else {
        untile_cube(cube, &partial)?;
    }
    partial.persist(&output).map_err(|e| e.error)?;
    if opts.remove {
        std::fs::remove_file(cube)?;
    }
    Ok(output)
}

/// Write the tile-compressed image in `cube` to `output` as a plain primary image.
fn untile_cube(cube: &Path, output: &Path) -> Result<()> {
    let mut fptr = FitsFile::open(cube)?;
    let index = crate::tile::compressed_hdu(&mut fptr)?
        .ok_or_else(|| FitsCubeError::NotTileCompressed(cube.to_path_buf()))?;
    fptr.hdu(index)?;

    let mut bitpix: c_int = 0;
    let mut status = 0;
    unsafe { sys::ffgidt(fptr.as_raw(), &mut bitpix, &mut status) };
    check_status(status)?;
    let bitpix = i64::from(bitpix);
    let dims = image_shape(&mut fptr)?;

    let mut header = image_header(&mut fptr, bitpix, &dims)?;
    // Quantised tiles always carry a scale, whatever the dithering
    let n_fields = key_i64(&mut fptr, "TFIELDS")?.unwrap_or(0);
    let mut quantised = false;
    for i in 1..=n_fields {
        quantised |= key_string(&mut fptr, &format!("TTYPE{i}"))?.as_deref() == Some("ZSCALE");
    }
    if quantised {
        let mut noisebit = None;
        for i in 1..10 {
            if key_string(&mut fptr, &format!("ZNAME{i}"))?.as_deref() == Some("NOISEBIT") {
                noisebit = key_f64(&mut fptr, &format!("ZVAL{i}"))?;
            }
        }
        let codec = key_string(&mut fptr, "ZCMPTYPE")?.unwrap_or_default();
        update_key_logical(&mut header, "FCLOSSY", true)?;
        if let Some(level) = noisebit {
            update_key_f64(&mut header, "FCQLEVEL", level)?;
        }
        let level = noisebit.map_or_else(|| "None".to_string(), |l| format!("{l:?}"));
        let history = std::ffi::CString::new(format!(
            "fitscube: decompressed from lossy {codec}, quantize level {level}. Values are not the originals."
        ))
        .map_err(|e| FitsCubeError::Atfits(e.to_string()))?;
        unsafe { sys::ffphis(header.as_raw(), history.as_ptr(), &mut status) };
        check_status(status)?;
    }
    let layout = extract_header_layout(&mut header, &dims)?;

    let mut out = BufWriter::with_capacity(8 << 20, File::create(output)?);
    out.write_all(&layout.header)?;
    let written = with_pixel_type!(bitpix, write_planes(&mut fptr, &dims, &mut out))?;
    out.write_all(&vec![0u8; (2880 - (written % 2880) as usize) % 2880])?;
    out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    drop(fptr);

    copy_other_hdus(cube, output, &[0, index])
}

/// A new in-memory primary image with the uncompressed header of the current
/// (compressed) HDU of `fptr`.
fn image_header(fptr: &mut FitsFile, bitpix: i64, dims: &[usize]) -> Result<FitsFile> {
    let mut status = 0;
    let mut text: *mut c_char = std::ptr::null_mut();
    let mut nkeys: c_int = 0;
    unsafe {
        sys::ffcnvthdr2str(
            fptr.as_raw(),
            0,
            std::ptr::null_mut(),
            0,
            &mut text,
            &mut nkeys,
            &mut status,
        )
    };
    check_status(status)?;
    let cards = unsafe { CStr::from_ptr(text) }.to_bytes().to_vec();
    unsafe { sys::fffree(text as *mut c_void, &mut status) };

    let mut raw: *mut sys::fitsfile = std::ptr::null_mut();
    unsafe { sys::ffinit(&mut raw, c"mem://".as_ptr(), &mut status) };
    check_status(status)?;
    let mut header = unsafe { FitsFile::from_raw(raw, fitsio::FileOpenMode::READWRITE)? };
    let mut naxes: Vec<LONGLONG> = vec![1; dims.len()];
    unsafe {
        sys::ffcrimll(
            header.as_raw(),
            bitpix as c_int,
            naxes.len() as c_int,
            naxes.as_mut_ptr(),
            &mut status,
        )
    };
    check_status(status)?;
    for card in cards.chunks(80) {
        let name = std::str::from_utf8(&card[..8]).unwrap_or("").trim_end();
        if is_structural_keyword(name)
            || matches!(name, "XTENSION" | "EXTNAME" | "CHECKSUM" | "DATASUM")
        {
            continue;
        }
        let card =
            std::ffi::CString::new(card).map_err(|e| FitsCubeError::Atfits(e.to_string()))?;
        unsafe { sys::ffprec(header.as_raw(), card.as_ptr(), &mut status) };
        check_status(status)?;
    }
    // Re-read the header so CFITSIO applies any copied BSCALE/BZERO/BLANK
    unsafe { sys::ffrdef(header.as_raw(), &mut status) };
    check_status(status)?;
    Ok(header)
}

/// Decompress every plane of the current image of `fptr` to `out`, as stored
/// big-endian values. Returns the bytes written.
fn write_planes<T: Pixel + MaybeFloat>(
    fptr: &mut FitsFile,
    dims: &[usize],
    out: &mut impl Write,
) -> Result<u64> {
    disable_scaling(fptr)?;
    let n_planes: usize = dims[2.min(dims.len())..].iter().product();
    let mut buf: Vec<T> = Vec::new();
    let mut written = 0;
    for plane in 0..n_planes {
        let mut ranges: Vec<(usize, usize)> = dims.iter().map(|&n| (0, n)).collect();
        let mut rest = plane;
        for (axis, &n) in dims.iter().enumerate().skip(2) {
            ranges[axis] = (rest % n, rest % n + 1);
            rest /= n;
        }
        read_pixels(fptr, Some(&ranges), T::nan_or_none(), &mut buf)?;
        swap_to_be(&mut buf);
        out.write_all(as_bytes(&buf))?;
        written += size_of_val(buf.as_slice()) as u64;
    }
    Ok(written)
}

/// Blank pixels read as NaN for float images, and as stored for integers.
trait MaybeFloat: Sized {
    fn nan_or_none() -> Option<Self> {
        None
    }
}
impl MaybeFloat for u8 {}
impl MaybeFloat for i16 {}
impl MaybeFloat for i32 {}
impl MaybeFloat for i64 {}
impl MaybeFloat for f32 {
    fn nan_or_none() -> Option<Self> {
        Some(Float::nan())
    }
}
impl MaybeFloat for f64 {
    fn nan_or_none() -> Option<Self> {
        Some(Float::nan())
    }
}

#[cfg(test)]
mod tests {
    use flate2::read::GzDecoder;

    use super::*;

    #[test]
    fn gzip_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cube.fits");
        let data: Vec<u8> = (0..10_000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let gz = compress_cube(&path, &CompressOptions::default()).unwrap();
        assert_eq!(gz, dir.path().join("cube.fits.gz"));
        assert!(!path.exists());
        let mut back = Vec::new();
        GzDecoder::new(File::open(gz).unwrap())
            .read_to_end(&mut back)
            .unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn pgzip_writes_members_any_reader_joins() {
        let data: Vec<u8> = (0..20_000_000u32).map(|i| (i % 251) as u8).collect();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("in");
        std::fs::write(&path, &data).unwrap();
        let mut packed = Vec::new();
        pgzip(
            &mut File::open(&path).unwrap(),
            &mut packed,
            Compression::fast(),
        )
        .unwrap();
        let mut back = Vec::new();
        MultiGzDecoder::new(packed.as_slice())
            .read_to_end(&mut back)
            .unwrap();
        assert_eq!(back, data);

        std::fs::write(&path, b"").unwrap();
        let mut packed = Vec::new();
        pgzip(
            &mut File::open(&path).unwrap(),
            &mut packed,
            Compression::fast(),
        )
        .unwrap();
        let mut back = Vec::new();
        MultiGzDecoder::new(packed.as_slice())
            .read_to_end(&mut back)
            .unwrap();
        assert!(back.is_empty());
    }
}
