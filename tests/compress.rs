//! Compression and decompression, from fitscube's test_compress.py.
use std::path::{Path, PathBuf};
use std::process::Command;

use atfits_rs::{key_bool, key_f64, key_i64, update_key_logical, update_key_str};
use fitscube_rs::compress::{
    CompressMethod, CompressOptions, DecompressOptions, compress_cube, decompress_cube,
};
use fitscube_rs::error::FitsCubeError;
use fitsio::FitsFile;
use fitsio::images::{ImageDescription, ImageType};
use fitsio::tables::{ColumnDataType, ColumnDescription};

const SIGMA: f32 = 1e-3;

/// Gaussian noise from a fixed seed (xorshift and Box-Muller).
fn noise(n: usize) -> Vec<f32> {
    let mut state: u64 = 42;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u, v) = (uniform(), uniform());
            ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32 * SIGMA
        })
        .collect()
}

/// A float cube of noise (numpy shape) with a blanked strip, an all-blank
/// plane, a constant plane and a BEAMS table. Returns the data in file order.
fn write_noise_cube(path: &Path, shape: &[usize]) -> Vec<f32> {
    let (ny, nx) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    let plane = ny * nx;
    let mut data = noise(shape.iter().product());
    for p in data.chunks_mut(plane) {
        p[..8 * nx].fill(f32::NAN);
    }
    data[plane..2 * plane].fill(f32::NAN);
    data[2 * plane..3 * plane].fill(0.5);

    let desc = ImageDescription {
        data_type: ImageType::Float,
        dimensions: shape,
    };
    let mut f = FitsFile::create(path)
        .with_custom_primary(&desc)
        .open()
        .unwrap();
    let hdu = f.primary_hdu().unwrap();
    hdu.write_image(&mut f, &data).unwrap();
    hdu.write_key(&mut f, "BUNIT", "Jy/beam").unwrap();
    hdu.write_key(&mut f, "CTYPE3", "FREQ").unwrap();
    let col = ColumnDescription::new("BMAJ")
        .with_type(ColumnDataType::Float)
        .create()
        .unwrap();
    let beams = f.create_table("BEAMS", &[col]).unwrap();
    let n_beams = shape[shape.len() - 3];
    beams
        .write_col(
            &mut f,
            "BMAJ",
            &(0..n_beams).map(|i| i as f32).collect::<Vec<_>>(),
        )
        .unwrap();
    data
}

fn cube(dir: &Path) -> (PathBuf, Vec<f32>) {
    let path = dir.join("cube.fits");
    let data = write_noise_cube(&path, &[2, 5, 40, 48]);
    (path, data)
}

/// Pixels of image HDU `hdu`, which CFITSIO decompresses if need be.
fn read_image(path: &Path, hdu: usize) -> Vec<f32> {
    let mut f = FitsFile::open(path).unwrap();
    f.hdu(hdu).unwrap();
    // CFITSIO 3.49 fails to decompress a read spanning planes of a 4D cube, so
    // read a plane at a time
    let shape = atfits_rs::image_shape(&mut f).unwrap();
    let mut out = Vec::new();
    for plane in 0..shape[2..].iter().product() {
        let mut ranges: Vec<_> = shape.iter().map(|&n| (0, n)).collect();
        let mut rest = plane;
        for (axis, &n) in shape.iter().enumerate().skip(2) {
            ranges[axis] = (rest % n, rest % n + 1);
            rest /= n;
        }
        let mut buf = Vec::new();
        atfits_rs::read_pixels(&mut f, Some(&ranges), Some(f32::NAN), &mut buf).unwrap();
        out.extend(buf);
    }
    out
}

fn read_beams(path: &Path) -> Vec<f32> {
    let mut f = FitsFile::open(path).unwrap();
    f.hdu("BEAMS").unwrap().read_col(&mut f, "BMAJ").unwrap()
}

fn raw_key_i64(path: &Path, hdu: usize, key: &str) -> Option<i64> {
    let mut f = FitsFile::open(path).unwrap();
    f.hdu(hdu).unwrap();
    key_i64(&mut f, key).unwrap()
}

fn options(method: CompressMethod) -> CompressOptions {
    CompressOptions {
        method,
        ..CompressOptions::default()
    }
}

/// Equal, with NaN equal to NaN.
fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

#[test]
fn lossless_round_trips() {
    for method in [
        CompressMethod::Gzip,
        CompressMethod::Pgzip,
        CompressMethod::TileGzip,
    ] {
        let dir = tempfile::tempdir().unwrap();
        let (path, data) = cube(dir.path());
        let original = std::fs::read(&path).unwrap();

        let compressed = compress_cube(&path, &options(method)).unwrap();
        let suffix = if method == CompressMethod::TileGzip {
            "fz"
        } else {
            "gz"
        };
        assert_eq!(compressed, dir.path().join(format!("cube.fits.{suffix}")));
        assert!(!path.exists());
        if method == CompressMethod::TileGzip {
            assert!(same(&read_image(&compressed, 1), &data));
            assert_eq!(read_beams(&compressed), [0.0, 1.0, 2.0, 3.0, 4.0]);
        }

        let restored = decompress_cube(&compressed, &DecompressOptions::default()).unwrap();
        assert_eq!(restored, path);
        assert!(compressed.exists());
        assert!(same(&read_image(&restored, 0), &data));
        assert_eq!(read_beams(&restored), [0.0, 1.0, 2.0, 3.0, 4.0]);
        if method != CompressMethod::TileGzip {
            assert_eq!(std::fs::read(&restored).unwrap(), original);
        }
    }
}

#[test]
fn tile_rice_error_is_a_fraction_of_the_noise() {
    let dir = tempfile::tempdir().unwrap();
    let (path, data) = cube(dir.path());
    let compressed = compress_cube(&path, &options(CompressMethod::TileRice)).unwrap();
    let got = read_image(&compressed, 1);

    let mut f = FitsFile::open(&compressed).unwrap();
    f.hdu(1).unwrap();
    assert_eq!(
        atfits_rs::key_string(&mut f, "BUNIT").unwrap().as_deref(),
        Some("Jy/beam")
    );
    assert_eq!(
        atfits_rs::key_string(&mut f, "ZQUANTIZ")
            .unwrap()
            .as_deref(),
        Some("SUBTRACTIVE_DITHER_2")
    );

    let mut sq = 0.0;
    let mut n = 0;
    let plane = 40 * 48;
    for (i, (&a, &b)) in data.iter().zip(&got).enumerate() {
        assert_eq!(a.is_nan(), b.is_nan(), "pixel {i}");
        if (2 * plane..3 * plane).contains(&i) {
            // The constant plane has no noise to quantise against, so is kept exactly
            assert_eq!(a, b);
        } else if a.is_finite() {
            sq += f64::from(a - b).powi(2);
            n += 1;
        }
    }
    // One step is sigma / 4, so the rms error is about step / sqrt(12)
    let rms = (sq / f64::from(n)).sqrt();
    assert!(
        rms < f64::from(SIGMA) / 4.0 / 12f64.sqrt() * 1.2,
        "rms {rms}"
    );
}

#[test]
fn three_axis_cubes_compress() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cube.fits");
    let data = write_noise_cube(&path, &[4, 70, 30]);
    let compressed = compress_cube(&path, &options(CompressMethod::TileGzip)).unwrap();
    assert!(same(&read_image(&compressed, 1), &data));
    // Bands of 64 full-width rows
    assert_eq!(raw_key_i64(&compressed, 1, "ZTILE1"), Some(30));
    assert_eq!(raw_key_i64(&compressed, 1, "ZTILE2"), Some(64));
    assert_eq!(raw_key_i64(&compressed, 1, "NAXIS2"), Some(8));
}

#[test]
fn tile_compress_is_the_same_for_any_worker_count() {
    let outputs: Vec<Vec<u8>> = [1, 4]
        .into_iter()
        .map(|workers| {
            let dir = tempfile::tempdir().unwrap();
            let (path, _) = cube(dir.path());
            let opts = CompressOptions {
                method: CompressMethod::TileRice,
                max_workers: Some(workers),
                ..CompressOptions::default()
            };
            std::fs::read(compress_cube(&path, &opts).unwrap()).unwrap()
        })
        .collect();
    assert_eq!(outputs[0], outputs[1]);
}

#[test]
fn small_tiles_decode() {
    let dir = tempfile::tempdir().unwrap();
    let (path, data) = cube(dir.path());
    let opts = CompressOptions {
        method: CompressMethod::TileGzip,
        tile_shape: Some((16, 16)),
        ..CompressOptions::default()
    };
    let compressed = compress_cube(&path, &opts).unwrap();
    assert!(same(&read_image(&compressed, 1), &data));
}

#[test]
fn tile_compress_refuses_compressed_input() {
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = cube(dir.path());
    let opts = CompressOptions {
        remove: false,
        ..options(CompressMethod::TileGzip)
    };
    let compressed = compress_cube(&path, &opts).unwrap();
    let err = compress_cube(&compressed, &opts).unwrap_err();
    assert!(matches!(err, FitsCubeError::AlreadyCompressed(_)), "{err}");
    assert!(err.to_string().ends_with("is already compressed"));

    // Found by its content too, not just its name
    let renamed = dir.path().join("renamed.fits");
    std::fs::rename(&compressed, &renamed).unwrap();
    let err = compress_cube(&renamed, &opts).unwrap_err();
    assert!(matches!(err, FitsCubeError::AlreadyCompressed(_)), "{err}");
}

#[test]
fn tile_rice_needs_a_quantize_level() {
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = cube(dir.path());
    let opts = CompressOptions {
        quantize_level: 0.0,
        ..options(CompressMethod::TileRice)
    };
    let err = compress_cube(&path, &opts).unwrap_err();
    assert!(err.to_string().contains("RICE_1"), "{err}");
    assert!(path.exists());
}

#[test]
fn tile_compress_refuses_integers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("int.fits");
    let desc = ImageDescription {
        data_type: ImageType::Short,
        dimensions: &[2, 8, 8],
    };
    let mut f = FitsFile::create(&path)
        .with_custom_primary(&desc)
        .open()
        .unwrap();
    f.primary_hdu()
        .unwrap()
        .write_image(&mut f, &[1i16; 128])
        .unwrap();
    drop(f);
    let err = compress_cube(&path, &options(CompressMethod::TileGzip)).unwrap_err();
    assert!(
        matches!(err, FitsCubeError::IntegerTiles { bitpix: 16, .. }),
        "{err}"
    );
}

#[test]
fn decompress_refuses_to_overwrite() {
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = cube(dir.path());
    let compressed = compress_cube(&path, &options(CompressMethod::TileGzip)).unwrap();
    std::fs::write(&path, b"").unwrap();
    let err = decompress_cube(&compressed, &DecompressOptions::default()).unwrap_err();
    assert!(matches!(err, FitsCubeError::OutputExists(_)), "{err}");

    let opts = DecompressOptions {
        overwrite: true,
        remove: true,
        ..DecompressOptions::default()
    };
    decompress_cube(&compressed, &opts).unwrap();
    assert!(!compressed.exists());
}

#[test]
fn decompress_needs_an_output_name() {
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = cube(dir.path());
    let err = decompress_cube(&path, &DecompressOptions::default()).unwrap_err();
    assert!(matches!(err, FitsCubeError::NoOutputName(_)), "{err}");
    let err = decompress_cube(
        &path,
        &DecompressOptions {
            output: Some(dir.path().join("out.fits")),
            ..DecompressOptions::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, FitsCubeError::NotTileCompressed(_)), "{err}");
}

#[test]
fn requantizing_a_lossy_cube_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = cube(dir.path());
    let remove = DecompressOptions {
        remove: true,
        ..DecompressOptions::default()
    };
    let compressed = compress_cube(&path, &options(CompressMethod::TileRice)).unwrap();
    let restored = decompress_cube(&compressed, &remove).unwrap();
    let lossy = read_image(&restored, 0);
    let mut f = FitsFile::open(&restored).unwrap();
    assert_eq!(key_bool(&mut f, "FCLOSSY").unwrap(), Some(true));
    assert_eq!(key_f64(&mut f, "FCQLEVEL").unwrap(), Some(4.0));
    drop(f);

    let err = compress_cube(&restored, &options(CompressMethod::TileRice)).unwrap_err();
    assert!(err.to_string().contains("already quantized"), "{err}");

    // Lossless compression of the quantised values is fine
    let lossless = compress_cube(&restored, &options(CompressMethod::TileGzip)).unwrap();
    assert!(same(&read_image(&lossless, 1), &lossy));

    let restored = decompress_cube(&lossless, &remove).unwrap();
    let mut f = FitsFile::open(&restored).unwrap();
    // Still marked: decompressing GZIP_2 neither adds nor drops the mark
    assert_eq!(key_bool(&mut f, "FCLOSSY").unwrap(), Some(true));
    drop(f);
    let opts = CompressOptions {
        allow_requantize: true,
        ..options(CompressMethod::TileRice)
    };
    assert!(compress_cube(&restored, &opts).unwrap().exists());
}

#[test]
fn requantizing_dithers_afresh() {
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = cube(dir.path());
    let first = compress_cube(&path, &options(CompressMethod::TileRice)).unwrap();
    let seed = raw_key_i64(&first, 1, "ZDITHER0").unwrap();
    assert!((1..=10000).contains(&seed));

    let restored = decompress_cube(
        &first,
        &DecompressOptions {
            remove: true,
            ..DecompressOptions::default()
        },
    )
    .unwrap();
    let opts = CompressOptions {
        allow_requantize: true,
        ..options(CompressMethod::TileRice)
    };
    let second = compress_cube(&restored, &opts).unwrap();
    assert_ne!(raw_key_i64(&second, 1, "ZDITHER0").unwrap(), seed);
}

#[test]
fn user_cards_survive_and_reserved_ones_do_not_clash() {
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = cube(dir.path());
    let mut f = FitsFile::edit(&path).unwrap();
    update_key_str(&mut f, "OBJECT", "it's a test").unwrap();
    update_key_logical(&mut f, "ZIMAGE", false).unwrap();
    drop(f);
    let compressed = compress_cube(&path, &options(CompressMethod::TileGzip)).unwrap();
    let restored = decompress_cube(&compressed, &DecompressOptions::default()).unwrap();
    let mut f = FitsFile::open(&restored).unwrap();
    assert_eq!(
        atfits_rs::key_string(&mut f, "OBJECT").unwrap().as_deref(),
        Some("it's a test")
    );
    assert_eq!(key_bool(&mut f, "FCLOSSY").unwrap(), None);
}

#[test]
fn compress_and_decompress_cli() {
    let dir = tempfile::tempdir().unwrap();
    let (path, data) = cube(dir.path());
    let bin = env!("CARGO_BIN_EXE_fitscubers");

    let status = Command::new(bin)
        .arg("compress")
        .arg(&path)
        .status()
        .unwrap();
    assert!(status.success());
    let compressed = dir.path().join("cube.fits.fz");
    assert!(path.exists(), "compress keeps its input unless asked");
    assert!(compressed.exists());

    std::fs::remove_file(&path).unwrap();
    let status = Command::new(bin)
        .arg("decompress")
        .arg(&compressed)
        .arg("--remove")
        .status()
        .unwrap();
    assert!(status.success());
    assert!(!compressed.exists());
    assert!(same(&read_image(&path, 0), &data));
}
