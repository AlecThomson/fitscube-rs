//! I/O micro-probe: compare cfitsio section I/O against the raw seek + byteswap
//! + write that the Python `fitscube` uses, to see where the per-byte cost is.
//!
//! Run: `cargo run --release --example io_probe`
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::time::Instant;

use fitscube_rs::fits_io::{CubeElem, create_cube_open};
use fitsio::FitsFile;
use fitsio::images::{ImageDescription, ImageType};

const NCHAN: usize = 400;
const SIZE: usize = 512;

fn datastart(path: &str) -> i64 {
    let mut f = FitsFile::open(path).unwrap();
    f.primary_hdu().unwrap();
    let (mut hs, mut ds, mut de, mut status) = (0i64, 0i64, 0i64, 0);
    unsafe {
        fitsio::sys::ffghad(f.as_raw(), &mut hs, &mut ds, &mut de, &mut status);
    }
    ds
}

fn main() {
    let dir = std::env::temp_dir().join("fitscube_io_probe");
    std::fs::create_dir_all(&dir).unwrap();
    let template = dir.join("template.fits");
    let cube = dir.join("cube.fits");

    // 2D template.
    {
        let desc = ImageDescription {
            data_type: ImageType::Float,
            dimensions: &[SIZE, SIZE],
        };
        let mut f = FitsFile::create(&template)
            .with_custom_primary(&desc)
            .overwrite()
            .open()
            .unwrap();
        let hdu = f.primary_hdu().unwrap();
        hdu.write_image(&mut f, &vec![0.0f32; SIZE * SIZE]).unwrap();
        hdu.write_key(&mut f, "REFFREQ", 1.0e9).unwrap();
    }

    let plane_elems = SIZE * SIZE;
    let plane_bytes = plane_elems * 4;
    let plane: Vec<f32> = (0..plane_elems).map(|i| i as f32 * 0.001).collect();

    // Pre-build the cube skeleton (final dims), closed.
    let dims = vec![SIZE, SIZE, NCHAN];
    drop(create_cube_open(&template, &cube, -32, &dims).unwrap());
    let ds = datastart(cube.to_str().unwrap());
    println!("datastart = {ds} bytes; plane = {plane_bytes} bytes; {NCHAN} planes\n");

    // ── WRITE A: cfitsio write_section, single open handle ──────────────────
    let t = Instant::now();
    {
        let mut f = FitsFile::edit(cube.to_str().unwrap()).unwrap();
        f.primary_hdu().unwrap();
        for c in 0..NCHAN {
            let start = c * plane_elems;
            f32::write_section(&mut f, start, start + plane_elems, &plane).unwrap();
        }
    }
    println!("WRITE cfitsio write_section : {:?}", t.elapsed());

    // ── WRITE B: raw seek + big-endian swap + write, single fd (Python-style) ─
    let t = Instant::now();
    {
        let mut fd = OpenOptions::new().write(true).open(&cube).unwrap();
        let mut buf = vec![0u8; plane_bytes];
        for c in 0..NCHAN {
            for (i, &v) in plane.iter().enumerate() {
                buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_bits().to_be_bytes());
            }
            fd.seek(SeekFrom::Start((ds as u64) + (c * plane_bytes) as u64))
                .unwrap();
            fd.write_all(&buf).unwrap();
        }
        fd.flush().unwrap();
    }
    println!("WRITE raw seek+swap+write   : {:?}", t.elapsed());

    // ── READ A: cfitsio read_image, open-per-plane (matches process_plane) ──
    let t = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..NCHAN {
        let mut f = FitsFile::open(template.to_str().unwrap()).unwrap();
        let v: Vec<f32> = f32::read_full(&mut f).unwrap();
        sink += v[0];
    }
    println!("READ  cfitsio read_image    : {:?}", t.elapsed());

    // ── READ B: raw read + big-endian swap, open-per-plane ──────────────────
    let tds = datastart(template.to_str().unwrap());
    let t = Instant::now();
    for _ in 0..NCHAN {
        let mut fd = std::fs::File::open(&template).unwrap();
        fd.seek(SeekFrom::Start(tds as u64)).unwrap();
        let mut buf = vec![0u8; plane_bytes];
        fd.read_exact(&mut buf).unwrap();
        let mut v = vec![0.0f32; plane_elems];
        for (i, out) in v.iter_mut().enumerate() {
            *out = f32::from_be_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap());
        }
        sink += v[0];
    }
    println!("READ  raw seek+read+swap    : {:?}", t.elapsed());

    println!("\n(sink={sink})");
}
