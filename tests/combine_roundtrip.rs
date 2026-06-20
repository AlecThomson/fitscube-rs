//! End-to-end Rust tests: write synthetic single-plane images, combine them
//! into a cube, and read the cube back to check data and WCS. Also exercises
//! plane extraction. These are self-contained (no Python / no test data) and run
//! under plain `cargo test`.
use std::path::{Path, PathBuf};

use fitscube_rs::combine::{CombineOptions, combine_fits};
use fitscube_rs::extract::{ExtractOptions, extract_plane_from_cube};
use fitsio::FitsFile;
use fitsio::images::{ImageDescription, ImageType};

/// Write a 2D image (ny × nx) filled with `fill`, tagged with REFFREQ and a
/// minimal celestial WCS.
fn write_image(path: &Path, ny: usize, nx: usize, reffreq: f64, fill: f32, beam: Option<f64>) {
    let desc = ImageDescription {
        data_type: ImageType::Float,
        dimensions: &[ny, nx],
    };
    let mut f = FitsFile::create(path)
        .with_custom_primary(&desc)
        .overwrite()
        .open()
        .unwrap();
    let hdu = f.primary_hdu().unwrap();
    let data = vec![fill; ny * nx];
    hdu.write_image(&mut f, &data).unwrap();
    hdu.write_key(&mut f, "REFFREQ", reffreq).unwrap();
    hdu.write_key(&mut f, "CTYPE1", "RA---SIN").unwrap();
    hdu.write_key(&mut f, "CTYPE2", "DEC--SIN").unwrap();
    hdu.write_key(&mut f, "CRPIX1", 1.0_f64).unwrap();
    hdu.write_key(&mut f, "CRPIX2", 1.0_f64).unwrap();
    hdu.write_key(&mut f, "CRVAL1", 0.0_f64).unwrap();
    hdu.write_key(&mut f, "CRVAL2", 0.0_f64).unwrap();
    hdu.write_key(&mut f, "CDELT1", -1.0_f64).unwrap();
    hdu.write_key(&mut f, "CDELT2", 1.0_f64).unwrap();
    if let Some(b) = beam {
        hdu.write_key(&mut f, "BMAJ", b).unwrap();
        hdu.write_key(&mut f, "BMIN", b).unwrap();
        hdu.write_key(&mut f, "BPA", 0.0_f64).unwrap();
    }
}

fn read_string_key(path: &Path, key: &str) -> String {
    let mut f = FitsFile::open(path.to_string_lossy().as_ref()).unwrap();
    let hdu = f.primary_hdu().unwrap();
    hdu.read_key::<String>(&mut f, key).unwrap()
}

fn read_f64_key(path: &Path, key: &str) -> f64 {
    let mut f = FitsFile::open(path.to_string_lossy().as_ref()).unwrap();
    let hdu = f.primary_hdu().unwrap();
    hdu.read_key::<f64>(&mut f, key).unwrap()
}

fn read_i64_key(path: &Path, key: &str) -> i64 {
    let mut f = FitsFile::open(path.to_string_lossy().as_ref()).unwrap();
    let hdu = f.primary_hdu().unwrap();
    hdu.read_key::<i64>(&mut f, key).unwrap()
}

#[test]
fn combine_three_frequency_images() {
    let dir = tempfile::tempdir().unwrap();
    let (ny, nx) = (4usize, 5usize);
    let files: Vec<PathBuf> = (0..3)
        .map(|i| {
            let p = dir.path().join(format!("img{i}.fits"));
            let freq = 1.0e9 * (i as f64 + 1.0);
            write_image(&p, ny, nx, freq, 10.0 * (i as f32 + 1.0), None);
            p
        })
        .collect();

    let out = dir.path().join("cube.fits");
    let specs = combine_fits(&files, &out, &CombineOptions::default()).unwrap();

    assert_eq!(specs, vec![1.0e9, 2.0e9, 3.0e9]);

    // Header: a 3-axis cube with a regular FREQ axis.
    assert_eq!(read_i64_key(&out, "NAXIS"), 3);
    assert_eq!(read_i64_key(&out, "NAXIS3"), 3);
    assert_eq!(read_string_key(&out, "CTYPE3"), "FREQ");
    assert_eq!(read_string_key(&out, "CUNIT3"), "Hz");
    assert!((read_f64_key(&out, "CRVAL3") - 1.0e9).abs() < 1.0);
    assert!((read_f64_key(&out, "CDELT3") - 1.0e9).abs() < 1.0);

    // Data: each plane holds its fill value.
    let mut f = FitsFile::open(out.to_string_lossy().as_ref()).unwrap();
    let hdu = f.primary_hdu().unwrap();
    let plane = ny * nx;
    for (chan, expect) in [(0usize, 10.0f32), (1, 20.0), (2, 30.0)] {
        let start = chan * plane;
        let data: Vec<f32> = hdu.read_section(&mut f, start, start + plane).unwrap();
        assert!(
            data.iter().all(|&v| (v - expect).abs() < 1e-6),
            "channel {chan} expected {expect}"
        );
    }
}

#[test]
fn combine_then_extract_roundtrips_plane() {
    let dir = tempfile::tempdir().unwrap();
    let (ny, nx) = (3usize, 3usize);
    let files: Vec<PathBuf> = (0..4)
        .map(|i| {
            let p = dir.path().join(format!("f{i}.fits"));
            write_image(&p, ny, nx, 1.0e9 + 1.0e8 * i as f64, i as f32, None);
            p
        })
        .collect();

    let cube = dir.path().join("c.fits");
    combine_fits(&files, &cube, &CombineOptions::default()).unwrap();

    let opts = ExtractOptions {
        channel_index: Some(2),
        ..Default::default()
    };
    let plane = extract_plane_from_cube(&cube, &opts).unwrap();
    assert_eq!(plane, dir.path().join("c.channel-2.fits"));

    // Extracted plane holds channel 2's data and the right reference value.
    let mut f = FitsFile::open(plane.to_string_lossy().as_ref()).unwrap();
    let hdu = f.primary_hdu().unwrap();
    let data: Vec<f32> = hdu.read_image(&mut f).unwrap();
    assert!(data.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    assert!((read_f64_key(&plane, "CRVAL3") - (1.0e9 + 2.0e8)).abs() < 1.0);
}

#[test]
fn varying_beams_emit_beam_table() {
    let dir = tempfile::tempdir().unwrap();
    let files: Vec<PathBuf> = (0..3)
        .map(|i| {
            let p = dir.path().join(format!("b{i}.fits"));
            // Beams differ per channel ⇒ a BEAMS table must be written.
            write_image(
                &p,
                2,
                2,
                1.0e9 * (i as f64 + 1.0),
                1.0,
                Some(0.01 * (i as f64 + 1.0)),
            );
            p
        })
        .collect();

    let cube = dir.path().join("beamcube.fits");
    combine_fits(&files, &cube, &CombineOptions::default()).unwrap();

    // CASAMBM set, primary beam keywords gone, BEAMS extension present.
    assert!(read_string_key(&cube, "CASAMBM").contains('T') || read_i64_key(&cube, "CASAMBM") == 1);
    let mut f = FitsFile::open(cube.to_string_lossy().as_ref()).unwrap();
    let beams = f.hdu("BEAMS").expect("BEAMS extension present");
    let bmaj: Vec<f32> = beams.read_col(&mut f, "BMAJ").unwrap();
    assert_eq!(bmaj.len(), 3);
    // First channel BMAJ ≈ 0.01 deg = 36 arcsec.
    assert!((bmaj[0] - 36.0).abs() < 1e-3);
}
