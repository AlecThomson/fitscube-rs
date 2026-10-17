//! FITS tile compression (`.fits.fz`), in the layout fitscube and fpack write.
//!
//! Planes are read as raw bytes, their tiles compressed across threads, and the
//! table and heap written in plane order, so the output does not depend on the
//! thread count.
use std::ffi::{CStr, c_char, c_double, c_float, c_int, c_long};
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::mpsc::sync_channel;

use atfits_rs::{image_shape, is_scaling_keyword, is_structural_keyword, key_bool};
use fitsio::FitsFile;
use fitsio::errors::check_status;
use fitsio::sys::{self, LONGLONG};
use flate2::{Compression, GzBuilder};
use rayon::prelude::*;

use crate::error::{FitsCubeError, Result};

unsafe extern "C" {
    fn fits_init_randoms() -> c_int;
    fn fits_quantize_float(
        row: c_long,
        fdata: *mut c_float,
        nxpix: c_long,
        nypix: c_long,
        nullcheck: c_int,
        in_null_value: c_float,
        qlevel: c_float,
        dither_method: c_int,
        idata: *mut c_int,
        bscale: *mut c_double,
        bzero: *mut c_double,
        iminval: *mut c_int,
        imaxval: *mut c_int,
    ) -> c_int;
    fn fits_quantize_double(
        row: c_long,
        fdata: *mut c_double,
        nxpix: c_long,
        nypix: c_long,
        nullcheck: c_int,
        in_null_value: c_double,
        qlevel: c_float,
        dither_method: c_int,
        idata: *mut c_int,
        bscale: *mut c_double,
        bzero: *mut c_double,
        iminval: *mut c_int,
        imaxval: *mut c_int,
    ) -> c_int;
    fn fits_rcomp(a: *mut c_int, nx: c_int, c: *mut u8, clen: c_int, nblock: c_int) -> c_int;
}

/// How the tiles are compressed.
#[derive(Clone, Copy, Debug)]
pub(crate) enum TileCodec {
    /// Lossless: shuffled bytes, gzipped.
    Gzip2 { level: u32 },
    /// Lossy: quantised to `level` steps per noise sigma, then Rice coded.
    Rice1 { level: f32 },
}

/// The settings of one tile compression run.
pub(crate) struct TileSettings {
    pub codec: TileCodec,
    /// (rows, columns) of a tile. Defaults to bands of 64 full-width rows.
    pub tile_shape: Option<(usize, usize)>,
    pub dither_seed: Option<u32>,
}

/// The float types a tile may hold.
trait TileFloat: Copy + Send + Sync {
    const BITPIX: i64;
    /// CFITSIO's stand-in for NaN while quantising.
    const NULL: Self;
    fn from_be(bytes: &[u8]) -> Self;
    fn is_nan(self) -> bool;
    /// `fits_quantize_float` or `fits_quantize_double`: its status, scale and zero.
    unsafe fn quantize(
        row: c_long,
        data: &mut [Self],
        nx: c_long,
        ny: c_long,
        nullcheck: c_int,
        level: f32,
        idata: &mut [c_int],
    ) -> (c_int, f64, f64);
}

impl TileFloat for f32 {
    const BITPIX: i64 = -32;
    const NULL: Self = -9.11912E-36;
    fn from_be(bytes: &[u8]) -> Self {
        Self::from_be_bytes(bytes.try_into().unwrap())
    }
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    unsafe fn quantize(
        row: c_long,
        data: &mut [Self],
        nx: c_long,
        ny: c_long,
        nullcheck: c_int,
        level: f32,
        idata: &mut [c_int],
    ) -> (c_int, f64, f64) {
        let (mut imin, mut imax) = (0, 0);
        let (mut scale, mut zero) = (0.0, 0.0);
        let status = unsafe {
            fits_quantize_float(
                row,
                data.as_mut_ptr(),
                nx,
                ny,
                nullcheck,
                Self::NULL,
                level,
                SUBTRACTIVE_DITHER_2,
                idata.as_mut_ptr(),
                &mut scale,
                &mut zero,
                &mut imin,
                &mut imax,
            )
        };
        (status, scale, zero)
    }
}

impl TileFloat for f64 {
    const BITPIX: i64 = -64;
    const NULL: Self = -9.1191291391491E-36;
    fn from_be(bytes: &[u8]) -> Self {
        Self::from_be_bytes(bytes.try_into().unwrap())
    }
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    unsafe fn quantize(
        row: c_long,
        data: &mut [Self],
        nx: c_long,
        ny: c_long,
        nullcheck: c_int,
        level: f32,
        idata: &mut [c_int],
    ) -> (c_int, f64, f64) {
        let (mut imin, mut imax) = (0, 0);
        let (mut scale, mut zero) = (0.0, 0.0);
        let status = unsafe {
            fits_quantize_double(
                row,
                data.as_mut_ptr(),
                nx,
                ny,
                nullcheck,
                Self::NULL,
                level,
                SUBTRACTIVE_DITHER_2,
                idata.as_mut_ptr(),
                &mut scale,
                &mut zero,
                &mut imin,
                &mut imax,
            )
        };
        (status, scale, zero)
    }
}

const SUBTRACTIVE_DITHER_2: c_int = 2;

/// One compressed tile: a row of the table.
#[derive(Default)]
struct Tile {
    data: Vec<u8>,
    /// Gzipped raw values, for a tile that could not be quantised.
    gzip_data: Vec<u8>,
    scale: f64,
    zero: f64,
}

/// The cube being compressed.
struct Source {
    file: File,
    /// Byte offset of the data unit.
    data_start: u64,
    /// `dims[0]` = NAXIS1.
    dims: Vec<usize>,
    bitpix: i64,
    /// Header cards worth keeping, 80 bytes each.
    cards: Vec<u8>,
    extend: bool,
}

impl Source {
    fn plane_bytes(&self) -> usize {
        self.dims[0] * self.dims[1] * self.bitpix.unsigned_abs() as usize / 8
    }

    fn n_planes(&self) -> usize {
        self.dims[2..].iter().product()
    }

    fn read_plane(&self, index: usize, buf: &mut Vec<u8>) -> Result<()> {
        buf.resize(self.plane_bytes(), 0);
        let offset = self.data_start + (index * self.plane_bytes()) as u64;
        read_at(&self.file, buf, offset)?;
        Ok(())
    }
}

/// The tiles of one plane in table order, and their (rows, columns).
struct Layout {
    ny: usize,
    nx: usize,
    tile: (usize, usize),
}

impl Layout {
    fn tiles(&self) -> impl Iterator<Item = (usize, usize, usize, usize)> + '_ {
        (0..self.ny).step_by(self.tile.0).flat_map(move |y| {
            (0..self.nx).step_by(self.tile.1).map(move |x| {
                (
                    y,
                    (y + self.tile.0).min(self.ny),
                    x,
                    (x + self.tile.1).min(self.nx),
                )
            })
        })
    }

    fn tiles_per_plane(&self) -> usize {
        self.ny.div_ceil(self.tile.0) * self.nx.div_ceil(self.tile.1)
    }
}

/// Header facts for the compression guards: BITPIX and FCLOSSY of the image.
pub(crate) fn image_kind(cube: &Path) -> Result<(i64, bool)> {
    let mut fptr = FitsFile::open(cube)?;
    fptr.primary_hdu()?;
    let mut bitpix: c_int = 0;
    let mut status = 0;
    unsafe { sys::ffgidt(fptr.as_raw(), &mut bitpix, &mut status) };
    check_status(status)?;
    Ok((
        i64::from(bitpix),
        key_bool(&mut fptr, "FCLOSSY")?.unwrap_or(false),
    ))
}

/// Whether any HDU of `cube` is a tile-compressed image.
pub(crate) fn has_compressed_hdu(cube: &Path) -> Result<bool> {
    Ok(compressed_hdu(&mut FitsFile::open(cube)?)?.is_some())
}

/// The index of the first tile-compressed HDU.
pub(crate) fn compressed_hdu(fptr: &mut FitsFile) -> Result<Option<usize>> {
    let mut n: c_int = 0;
    let mut status = 0;
    unsafe { sys::ffthdu(fptr.as_raw(), &mut n, &mut status) };
    check_status(status)?;
    for i in 0..n as usize {
        fptr.hdu(i)?;
        if unsafe { sys::fits_is_compressed_image(fptr.as_raw(), &mut status) } != 0 {
            return Ok(Some(i));
        }
        check_status(status)?;
    }
    Ok(None)
}

/// Tile-compress the float image in the primary HDU of `cube` into `output`,
/// then copy the other HDUs after it.
pub(crate) fn tile_compress(
    cube: &Path,
    output: &Path,
    settings: &TileSettings,
    threads: usize,
) -> Result<()> {
    let source = open_source(cube)?;
    if source.dims.len() < 2 {
        return Err(FitsCubeError::UnsupportedNaxis(source.dims.len() as i64));
    }
    let (ny, nx) = (source.dims[1], source.dims[0]);
    let layout = Layout {
        ny,
        nx,
        tile: settings.tile_shape.unwrap_or((64.min(ny), nx)),
    };
    let n_planes = source.n_planes();
    let n_rows = n_planes * layout.tiles_per_plane();
    let seed = match settings.dither_seed {
        Some(seed) => seed,
        None => {
            // Seeded from the data, so output is reproducible, yet recompressing
            // changed (e.g. already quantised) data dithers afresh. A fixed seed
            // would line up the errors of repeated quantisation.
            let mut middle = Vec::new();
            source.read_plane(n_planes / 2, &mut middle)?;
            let mut crc = flate2::Crc::new();
            crc.update(&middle);
            crc.sum() % 10000 + 1
        }
    };
    if matches!(settings.codec, TileCodec::Rice1 { .. }) {
        // Fills CFITSIO's shared table of random numbers, before threads race to
        if unsafe { fits_init_randoms() } != 0 {
            return Err(FitsCubeError::Atfits(
                "CFITSIO could not set up its dither table".into(),
            ));
        }
    }
    let table = TableHeader {
        source: &source,
        layout: &layout,
        codec: settings.codec,
        n_rows,
        seed,
    };

    let mut out = File::create(output)?;
    out.write_all(&primary_header())?;
    let table_start = out.stream_position()?;
    let mut descriptors = vec![0u8; n_rows * table.row_width()];
    let mut max_lengths = [0usize; 2];
    out.write_all(&table.header(0, max_lengths))?;
    out.write_all(&descriptors)?;

    let batch = 2 * threads.max(1);
    let mut heap_size = 0usize;
    std::thread::scope(|scope| -> Result<()> {
        let (tx, rx) = sync_channel::<Vec<Vec<Tile>>>(1);
        let source = &source;
        let layout = &layout;
        let producer = scope.spawn(move || -> Result<()> {
            let mut start = 0;
            while start < n_planes {
                let end = (start + batch).min(n_planes);
                let planes = (start..end)
                    .into_par_iter()
                    .map_init(Vec::new, |buf, plane| {
                        source.read_plane(plane, buf)?;
                        compress_plane(buf, layout, settings.codec, plane, seed, source.bitpix)
                    })
                    .collect::<Result<Vec<_>>>()?;
                if tx.send(planes).is_err() {
                    return Err(FitsCubeError::Cancelled);
                }
                start = end;
            }
            Ok(())
        });

        let mut row = 0;
        let width = table.row_width();
        let writer = (|| -> Result<()> {
            for planes in rx {
                for tile in planes.into_iter().flatten() {
                    let cells = &mut descriptors[row * width..(row + 1) * width];
                    for (col, data) in [&tile.data, &tile.gzip_data].into_iter().enumerate() {
                        if col == 1 && !table.is_rice() {
                            break;
                        }
                        let offset = if data.is_empty() { 0 } else { heap_size };
                        cells[col * 16..col * 16 + 8]
                            .copy_from_slice(&(data.len() as i64).to_be_bytes());
                        cells[col * 16 + 8..col * 16 + 16]
                            .copy_from_slice(&(offset as i64).to_be_bytes());
                        max_lengths[col] = max_lengths[col].max(data.len());
                        out.write_all(data)?;
                        heap_size += data.len();
                    }
                    if table.is_rice() {
                        cells[32..40].copy_from_slice(&tile.scale.to_be_bytes());
                        cells[40..48].copy_from_slice(&tile.zero.to_be_bytes());
                    }
                    row += 1;
                }
            }
            Ok(())
        })();
        // A failed writer drops the channel, which cancels the producer
        let produced = producer.join().map_err(|_| FitsCubeError::WorkerPanic)?;
        writer?;
        produced
    })?;

    let end = out.stream_position()?;
    out.write_all(&vec![0u8; (2880 - end as usize % 2880) % 2880])?;
    out.seek(SeekFrom::Start(table_start))?;
    // Same size as the placeholder: only fixed-width values change
    out.write_all(&table.header(heap_size, max_lengths))?;
    out.write_all(&descriptors)?;
    out.sync_all()?;
    drop(out);

    copy_other_hdus(cube, output, &[0])
}

fn open_source(cube: &Path) -> Result<Source> {
    let mut fptr = FitsFile::open(cube)?;
    fptr.primary_hdu()?;
    let dims = image_shape(&mut fptr)?;
    let (bitpix, _) = image_kind(cube)?;
    let (mut head, mut data, mut end): (LONGLONG, LONGLONG, LONGLONG) = (0, 0, 0);
    let mut status = 0;
    unsafe { sys::ffghadll(fptr.as_raw(), &mut head, &mut data, &mut end, &mut status) };
    check_status(status)?;

    let mut nkeys: c_int = 0;
    unsafe { sys::ffghsp(fptr.as_raw(), &mut nkeys, std::ptr::null_mut(), &mut status) };
    check_status(status)?;
    let mut cards = Vec::new();
    let mut card = [0 as c_char; 81];
    for i in 1..=nkeys {
        unsafe { sys::ffgrec(fptr.as_raw(), i, card.as_mut_ptr(), &mut status) };
        check_status(status)?;
        let text = unsafe { CStr::from_ptr(card.as_ptr()) }.to_bytes();
        if keep_card(card_name(text)) {
            cards.extend_from_slice(&pad_card(text));
        }
    }
    Ok(Source {
        file: File::open(cube)?,
        data_start: data as u64,
        dims,
        bitpix,
        cards,
        extend: key_bool(&mut fptr, "EXTEND")?.unwrap_or(false),
    })
}

/// Whether a card of the image belongs in the compressed table header too.
fn keep_card(name: &str) -> bool {
    let reserved = [
        "ZIMAGE", "ZSIMPLE", "ZEXTEND", "ZBITPIX", "ZCMPTYPE", "ZQUANTIZ", "ZDITHER0", "ZBLANK",
        "ZSCALE", "ZZERO", "ZTENSION", "ZPCOUNT", "ZGCOUNT", "ZHECKSUM", "ZDATASUM", "XTENSION",
        "EXTNAME", "TFIELDS", "THEAP", "CHECKSUM", "DATASUM",
    ];
    let numbered = ["ZNAXIS", "ZTILE", "ZNAME", "ZVAL", "TTYPE", "TFORM"];
    !(is_structural_keyword(name)
        || is_scaling_keyword(name)
        || reserved.contains(&name)
        || numbered.iter().any(|p| {
            name.strip_prefix(p)
                .is_some_and(|n| n.bytes().all(|c| c.is_ascii_digit()))
        }))
}

/// Keyword name of a card: the text before the first space or `=`.
fn card_name(card: &[u8]) -> &str {
    let end = card
        .iter()
        .position(|&b| b == b' ' || b == b'=')
        .unwrap_or(card.len());
    std::str::from_utf8(&card[..end]).unwrap_or("")
}

fn pad_card(text: &[u8]) -> [u8; 80] {
    let mut card = [b' '; 80];
    let n = text.len().min(80);
    card[..n].copy_from_slice(&text[..n]);
    card
}

/// A card with a fixed-format value: numbers and logicals right-justified,
/// strings quoted from column 11.
fn card(name: &str, value: CardValue) -> [u8; 80] {
    let text = match value {
        CardValue::Int(v) => format!("{name:<8}= {v:>20}"),
        CardValue::Float(v) => format!("{name:<8}= {:>20}", format!("{v:?}").to_uppercase()),
        CardValue::Bool(v) => format!("{name:<8}= {:>20}", if v { "T" } else { "F" }),
        CardValue::Str(v) => format!("{name:<8}= '{:<8}'", v.replace('\'', "''")),
    };
    pad_card(text.as_bytes())
}

enum CardValue<'a> {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(&'a str),
}

/// Cards, then END, padded to a whole number of blocks.
fn finish(mut cards: Vec<u8>) -> Vec<u8> {
    cards.extend_from_slice(&pad_card(b"END"));
    cards.resize(cards.len().div_ceil(2880) * 2880, b' ');
    cards
}

/// An empty primary HDU, as the compressed image goes in an extension.
fn primary_header() -> Vec<u8> {
    use CardValue::*;
    let mut cards = Vec::new();
    cards.extend(card("SIMPLE", Bool(true)));
    cards.extend(card("BITPIX", Int(8)));
    cards.extend(card("NAXIS", Int(0)));
    cards.extend(card("EXTEND", Bool(true)));
    finish(cards)
}

/// The binary-table header of the compressed image.
struct TableHeader<'a> {
    source: &'a Source,
    layout: &'a Layout,
    codec: TileCodec,
    n_rows: usize,
    seed: u32,
}

impl TableHeader<'_> {
    fn is_rice(&self) -> bool {
        matches!(self.codec, TileCodec::Rice1 { .. })
    }

    /// Two 64-bit descriptors per heap column, then ZSCALE and ZZERO.
    fn row_width(&self) -> usize {
        if self.is_rice() { 48 } else { 16 }
    }

    fn header(&self, heap_size: usize, max_lengths: [usize; 2]) -> Vec<u8> {
        use CardValue::*;
        let mut columns = vec![("COMPRESSED_DATA", format!("1QB({})", max_lengths[0]))];
        if self.is_rice() {
            columns.push(("GZIP_COMPRESSED_DATA", format!("1QB({})", max_lengths[1])));
            columns.push(("ZSCALE", "1D".into()));
            columns.push(("ZZERO", "1D".into()));
        }
        let mut c: Vec<[u8; 80]> = vec![
            card("XTENSION", Str("BINTABLE")),
            card("BITPIX", Int(8)),
            card("NAXIS", Int(2)),
            card("NAXIS1", Int(self.row_width() as i64)),
            card("NAXIS2", Int(self.n_rows as i64)),
            card("PCOUNT", Int(heap_size as i64)),
            card("GCOUNT", Int(1)),
            card("TFIELDS", Int(columns.len() as i64)),
        ];
        for (i, (name, form)) in columns.iter().enumerate() {
            c.push(card(&format!("TTYPE{}", i + 1), Str(name)));
            c.push(card(&format!("TFORM{}", i + 1), Str(form)));
        }
        c.push(card("ZIMAGE", Bool(true)));
        c.push(card("ZSIMPLE", Bool(true)));
        c.push(card("ZBITPIX", Int(self.source.bitpix)));
        c.push(card("ZNAXIS", Int(self.source.dims.len() as i64)));
        for (i, &n) in self.source.dims.iter().enumerate() {
            c.push(card(&format!("ZNAXIS{}", i + 1), Int(n as i64)));
        }
        let (ty, tx) = self.layout.tile;
        for i in 0..self.source.dims.len() {
            let n = match i {
                0 => tx,
                1 => ty,
                _ => 1,
            };
            c.push(card(&format!("ZTILE{}", i + 1), Int(n as i64)));
        }
        if self.source.extend {
            c.push(card("ZEXTEND", Bool(true)));
        }
        match self.codec {
            TileCodec::Gzip2 { .. } => c.push(card("ZCMPTYPE", Str("GZIP_2"))),
            TileCodec::Rice1 { level } => {
                c.push(card("ZCMPTYPE", Str("RICE_1")));
                c.push(card("ZNAME1", Str("BLOCKSIZE")));
                c.push(card("ZVAL1", Int(32)));
                c.push(card("ZNAME2", Str("BYTEPIX")));
                c.push(card("ZVAL2", Int(4)));
                c.push(card("ZNAME3", Str("NOISEBIT")));
                c.push(card("ZVAL3", Float(f64::from(level))));
                c.push(card("ZQUANTIZ", Str("SUBTRACTIVE_DITHER_2")));
                c.push(card("ZDITHER0", Int(i64::from(self.seed))));
                // What CFITSIO quantises a blank pixel to
                c.push(card("ZBLANK", Int(-2147483647)));
            }
        }
        c.push(card("EXTNAME", Str("COMPRESSED_IMAGE")));
        let mut bytes: Vec<u8> = c.concat();
        bytes.extend_from_slice(&self.source.cards);
        finish(bytes)
    }
}

/// The tiles of one plane, given as big-endian bytes.
fn compress_plane(
    plane: &[u8],
    layout: &Layout,
    codec: TileCodec,
    index: usize,
    seed: u32,
    bitpix: i64,
) -> Result<Vec<Tile>> {
    let bytepix = bitpix.unsigned_abs() as usize / 8;
    let first_row = index * layout.tiles_per_plane();
    layout
        .tiles()
        .enumerate()
        .map(|(i, (y0, y1, x0, x1))| {
            let mut raw = Vec::with_capacity((y1 - y0) * (x1 - x0) * bytepix);
            for y in y0..y1 {
                let start = (y * layout.nx + x0) * bytepix;
                raw.extend_from_slice(&plane[start..start + (x1 - x0) * bytepix]);
            }
            match codec {
                TileCodec::Gzip2 { level } => Ok(Tile {
                    data: gzip(&shuffle(&raw, bytepix), level)?,
                    ..Tile::default()
                }),
                TileCodec::Rice1 { level } => {
                    // CFITSIO's own convention for the dither of table row `row`
                    let row = (first_row + i) as c_long + c_long::from(seed);
                    let (nx, ny) = ((x1 - x0) as c_long, (y1 - y0) as c_long);
                    if bitpix == -32 {
                        rice_tile::<f32>(&raw, row, nx, ny, level)
                    } else {
                        rice_tile::<f64>(&raw, row, nx, ny, level)
                    }
                }
            }
        })
        .collect()
}

/// GZIP_2 keeps the first byte of every value together, then the second, and so on.
fn shuffle(raw: &[u8], bytepix: usize) -> Vec<u8> {
    let n = raw.len() / bytepix;
    let mut out = vec![0; raw.len()];
    for (i, value) in raw.chunks_exact(bytepix).enumerate() {
        for (b, &byte) in value.iter().enumerate() {
            out[b * n + i] = byte;
        }
    }
    out
}

fn gzip(data: &[u8], level: u32) -> Result<Vec<u8>> {
    let mut gz = GzBuilder::new()
        .mtime(0)
        .write(Vec::with_capacity(data.len() / 2), Compression::new(level));
    gz.write_all(data)?;
    Ok(gz.finish()?)
}

fn rice_tile<T: TileFloat>(
    raw: &[u8],
    row: c_long,
    nx: c_long,
    ny: c_long,
    level: f32,
) -> Result<Tile> {
    let bytepix = (-T::BITPIX) as usize / 8;
    let mut values: Vec<T> = raw.chunks_exact(bytepix).map(T::from_be).collect();
    let mut nullcheck = 0;
    for v in values.iter_mut().filter(|v| v.is_nan()) {
        *v = T::NULL;
        nullcheck = 1;
    }
    let mut ints: Vec<c_int> = vec![0; values.len()];
    let (quantised, scale, zero) =
        unsafe { T::quantize(row, &mut values, nx, ny, nullcheck, level, &mut ints) };
    match quantised {
        1 => {
            let mut packed = vec![0u8; ints.len() * 4 + ints.len() / 8 + 64];
            let n = unsafe {
                fits_rcomp(
                    ints.as_mut_ptr(),
                    ints.len() as c_int,
                    packed.as_mut_ptr(),
                    packed.len() as c_int,
                    32,
                )
            };
            if n < 0 {
                return Err(FitsCubeError::Atfits(
                    "CFITSIO could not Rice-code a tile".into(),
                ));
            }
            packed.truncate(n as usize);
            Ok(Tile {
                data: packed,
                gzip_data: Vec::new(),
                scale,
                zero,
            })
        }
        // No noise to quantise against (e.g. a constant tile), so store it exactly
        0 => Ok(Tile {
            gzip_data: gzip(raw, 6)?,
            ..Tile::default()
        }),
        status => Err(FitsCubeError::Fits(fitsio::errors::Error::Fits(
            fitsio::errors::FitsError {
                status,
                message: atfits_rs::status_message(status),
            },
        ))),
    }
}

/// Append every HDU of `src` except `skip` to `dst`.
pub(crate) fn copy_other_hdus(src: &Path, dst: &Path, skip: &[usize]) -> Result<()> {
    let mut input = FitsFile::open(src)?;
    let mut n: c_int = 0;
    let mut status = 0;
    unsafe { sys::ffthdu(input.as_raw(), &mut n, &mut status) };
    check_status(status)?;
    if (0..n as usize).all(|i| skip.contains(&i)) {
        return Ok(());
    }
    let mut output = FitsFile::edit(dst)?;
    for i in (0..n as usize).filter(|i| !skip.contains(i)) {
        input.hdu(i)?;
        unsafe { sys::ffcopy(input.as_raw(), output.as_raw(), 0, &mut status) };
        check_status(status)?;
    }
    Ok(())
}

#[cfg(unix)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(windows)]
fn read_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_read(buf, offset)? {
            0 => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            n => {
                buf = &mut buf[n..];
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
    fn shuffle_groups_bytes() {
        assert_eq!(shuffle(&[1, 2, 3, 4, 5, 6], 2), vec![1, 3, 5, 2, 4, 6]);
    }

    #[test]
    fn cards_are_fixed_format() {
        let c = card("NAXIS1", CardValue::Int(48));
        assert_eq!(&c[..30], b"NAXIS1  =                   48");
        let c = card("TFORM1", CardValue::Str("1QB(12)"));
        assert_eq!(&c[..20], b"TFORM1  = '1QB(12) '");
        assert_eq!(
            card("ZVAL3", CardValue::Float(4.0))[..30],
            *b"ZVAL3   =                  4.0"
        );
    }

    #[test]
    fn bands_cover_the_plane() {
        let layout = Layout {
            ny: 100,
            nx: 30,
            tile: (64, 30),
        };
        let tiles: Vec<_> = layout.tiles().collect();
        assert_eq!(tiles, vec![(0, 64, 0, 30), (64, 100, 0, 30)]);
        assert_eq!(layout.tiles_per_plane(), 2);
    }

    #[test]
    fn keeps_user_cards_only() {
        assert!(keep_card("BUNIT"));
        assert!(keep_card("HISTORY"));
        assert!(keep_card("ZOOM"));
        assert!(!keep_card("NAXIS3"));
        assert!(!keep_card("ZNAXIS2"));
        assert!(!keep_card("BSCALE"));
        assert!(!keep_card("CHECKSUM"));
    }
}
