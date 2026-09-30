//! End-to-end combines of image pairs across input axis layouts.
//!
//! Port of `tests/test_combine_layouts.py` in fitscube. Each case combines two
//! single-plane images and states the cube it expects in the notation
//! `(RA, DEC, FREQ) + (RA, DEC, FREQ) -> (RA, DEC, FREQ)`, or the error it
//! expects. Axes are named by CTYPE. A Stokes symbol (e.g. `I` or `XX`) is a
//! STOKES axis holding that parameter, and `FOO` is an axis type fitscube knows
//! nothing about.
use std::path::{Path, PathBuf};

use fitscube_rs::combine::{CombineOptions, combine_fits};
use fitscube_rs::error::FitsCubeError;
use fitsio::FitsFile;
use fitsio::images::{ImageDescription, ImageType};
use hifitime::Epoch;

/// FITS Stokes codes (FITS WCS paper III, table 7).
const STOKES_CODES: &[(&str, f64)] = &[
    ("I", 1.0),
    ("Q", 2.0),
    ("U", 3.0),
    ("V", 4.0),
    ("RR", -1.0),
    ("LL", -2.0),
    ("RL", -3.0),
    ("LR", -4.0),
    ("XX", -5.0),
    ("YY", -6.0),
    ("XY", -7.0),
    ("YX", -8.0),
];

fn stokes_code(axis: &str) -> Option<f64> {
    STOKES_CODES
        .iter()
        .find(|(symbol, _)| *symbol == axis)
        .map(|&(_, code)| code)
}

/// Write the `index`-th single-plane image with the given axes.
///
/// Plane `index` holds the value `index` and its own beam, is one step later in
/// frequency and time than the plane before, and each non-celestial axis has
/// length one.
fn write_plane(dir: &Path, axes: &[&str], index: usize) -> PathBuf {
    let i = index as f64;
    let mjd = 60000.0 + i * 10.0 / 86400.0;
    let isot = Epoch::from_mjd_utc(mjd).to_isoformat();

    // FITS order is NAXIS1 fastest; fitsio takes C order (slowest first).
    let mut dims = vec![1usize; axes.len() - 2];
    dims.extend([10, 10]);
    let desc = ImageDescription {
        data_type: ImageType::Double,
        dimensions: &dims,
    };
    let path = dir.join(format!("plane_{index}.fits"));
    let mut f = FitsFile::create(&path)
        .with_custom_primary(&desc)
        .overwrite()
        .open()
        .unwrap();
    let hdu = f.primary_hdu().unwrap();
    hdu.write_image(&mut f, &vec![i; 100]).unwrap();

    for (n, axis) in axes.iter().enumerate() {
        let n = n + 1;
        let (ctype, crval, cdelt, crpix, cunit): (&str, f64, f64, f64, Option<&str>) =
            match (*axis, stokes_code(axis)) {
                (_, Some(code)) => ("STOKES", code, 1.0, 1.0, None),
                ("RA", _) => ("RA---SIN", 0.0, -1e-3, 5.0, None),
                ("DEC", _) => ("DEC--SIN", 0.0, 1e-3, 5.0, None),
                ("FREQ", _) => ("FREQ", 1e9 + i * 1e6, 1e6, 1.0, Some("Hz")),
                ("TIME", _) => ("TIME", mjd * 86400.0, 10.0, 1.0, Some("s")),
                ("FOO", _) => ("FOO", 1.0, 1.0, 1.0, None),
                (other, _) => panic!("unknown test axis {other}"),
            };
        hdu.write_key(&mut f, &format!("CTYPE{n}"), ctype).unwrap();
        hdu.write_key(&mut f, &format!("CRVAL{n}"), crval).unwrap();
        hdu.write_key(&mut f, &format!("CDELT{n}"), cdelt).unwrap();
        hdu.write_key(&mut f, &format!("CRPIX{n}"), crpix).unwrap();
        if let Some(cunit) = cunit {
            hdu.write_key(&mut f, &format!("CUNIT{n}"), cunit).unwrap();
        }
    }
    hdu.write_key(&mut f, "DATE-OBS", isot.as_str()).unwrap();
    hdu.write_key(&mut f, "MJD-OBS", mjd).unwrap();
    if axes.len() == 2 {
        hdu.write_key(&mut f, "REFFREQ", 1e9 + i * 1e6).unwrap();
    }
    // Vary the beam per plane so a beam table is written
    hdu.write_key(&mut f, "BMAJ", 1e-3 * (1.0 + i)).unwrap();
    hdu.write_key(&mut f, "BMIN", 1e-3).unwrap();
    hdu.write_key(&mut f, "BPA", 0.0).unwrap();
    path
}

fn combine_pair(
    dir: &Path,
    time_domain_mode: bool,
    first: &[&str],
    second: &[&str],
) -> (PathBuf, fitscube_rs::error::Result<Vec<f64>>) {
    let files = vec![write_plane(dir, first, 0), write_plane(dir, second, 1)];
    let out_cube = dir.join("cube.fits");
    let options = CombineOptions {
        time_domain_mode,
        overwrite: true,
        ..Default::default()
    };
    let result = combine_fits(&files, &out_cube, &options);
    (out_cube, result)
}

/// first + second -> expected, with the planes in order and a valid BEAMS table
fn check_pair(time_domain_mode: bool, first: &[&str], second: &[&str], expected: &[&str]) {
    let dir = tempfile::tempdir().unwrap();
    let (out_cube, result) = combine_pair(dir.path(), time_domain_mode, first, second);
    result.unwrap_or_else(|e| panic!("{first:?} + {second:?} failed: {e}"));

    let mut f = FitsFile::open(out_cube.to_string_lossy().as_ref()).unwrap();
    let hdu = f.primary_hdu().unwrap();
    let n_axes: usize = hdu.read_key::<i64>(&mut f, "NAXIS").unwrap() as usize;
    let ctypes: Vec<String> = (1..=n_axes)
        .map(|n| {
            let ctype: String = hdu.read_key(&mut f, &format!("CTYPE{n}")).unwrap();
            ctype.split('-').next().unwrap().to_string()
        })
        .collect();
    assert_eq!(ctypes, expected, "{first:?} + {second:?}");

    let combine_ctype = if time_domain_mode { "TIME" } else { "FREQ" };
    let combine_axis = ctypes.iter().position(|c| c == combine_ctype).unwrap() + 1;
    let naxes: Vec<i64> = (1..=n_axes)
        .map(|n| hdu.read_key(&mut f, &format!("NAXIS{n}")).unwrap())
        .collect();
    let expected_naxes: Vec<i64> = ctypes
        .iter()
        .enumerate()
        .map(|(n, ctype)| match ctype.as_str() {
            "RA" | "DEC" => 10,
            _ if n + 1 == combine_axis => 2,
            _ => 1,
        })
        .collect();
    assert_eq!(naxes, expected_naxes, "{first:?} + {second:?}");

    if let Some(code) = first.iter().find_map(|axis| stokes_code(axis)) {
        let stokes_axis = ctypes.iter().position(|c| c == "STOKES").unwrap() + 1;
        let crval: f64 = hdu
            .read_key(&mut f, &format!("CRVAL{stokes_axis}"))
            .unwrap();
        assert_eq!(crval, code, "{first:?} + {second:?}");
    }

    // Every non-celestial axis but the combine axis has length one, so plane i
    // along the combine axis is simply the i-th block of 100 pixels.
    let data: Vec<f64> = hdu.read_image(&mut f).unwrap();
    assert_eq!(data.len(), 200);
    assert!(
        data[..100].iter().all(|&v| v == 0.0),
        "{first:?} + {second:?}"
    );
    assert!(
        data[100..].iter().all(|&v| v == 1.0),
        "{first:?} + {second:?}"
    );

    let beams = f.hdu("BEAMS").unwrap();
    assert_eq!(beams.read_key::<i64>(&mut f, "NCHAN").unwrap(), 2);
    assert_eq!(beams.read_key::<i64>(&mut f, "NPOL").unwrap(), 1);
    let chan: Vec<i32> = beams.read_col(&mut f, "CHAN").unwrap();
    let pol: Vec<i32> = beams.read_col(&mut f, "POL").unwrap();
    assert_eq!(chan, vec![0, 1], "{first:?} + {second:?}");
    assert_eq!(pol, vec![0, 0], "{first:?} + {second:?}");
}

/// first + second -> error, before any output is written
fn check_pair_raises(
    time_domain_mode: bool,
    first: &[&str],
    second: &[&str],
    matches: impl Fn(&FitsCubeError) -> bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let (out_cube, result) = combine_pair(dir.path(), time_domain_mode, first, second);
    match result {
        Ok(_) => panic!("{first:?} + {second:?} should have failed"),
        Err(e) => assert!(matches(&e), "{first:?} + {second:?} gave {e:?}"),
    }
    assert!(!out_cube.exists(), "{first:?} + {second:?} wrote output");
}

fn axis_mismatch(e: &FitsCubeError) -> bool {
    matches!(e, FitsCubeError::AxisMismatch(_))
}

fn stokes_mismatch(e: &FitsCubeError) -> bool {
    matches!(e, FitsCubeError::StokesMismatch(_))
}

// Frequency cubes

#[test]
fn freq_2d() {
    check_pair(
        false,
        &["RA", "DEC"],
        &["RA", "DEC"],
        &["RA", "DEC", "FREQ"],
    );
}

#[test]
fn freq_3d() {
    let axes = ["RA", "DEC", "FREQ"];
    check_pair(false, &axes, &axes, &axes);
}

#[test]
fn freq_3d_dec_ra() {
    let axes = ["DEC", "RA", "FREQ"];
    check_pair(false, &axes, &axes, &axes);
}

#[test]
fn freq_foo_before_freq() {
    let axes = ["RA", "DEC", "FOO", "FREQ"];
    check_pair(false, &axes, &axes, &axes);
}

#[test]
fn freq_foo_after_freq() {
    let axes = ["RA", "DEC", "FREQ", "FOO"];
    check_pair(false, &axes, &axes, &axes);
}

#[test]
fn freq_wsclean_order() {
    let axes = ["RA", "DEC", "FREQ", "I"];
    check_pair(false, &axes, &axes, &["RA", "DEC", "FREQ", "STOKES"]);
}

#[test]
fn freq_casa_order() {
    let axes = ["RA", "DEC", "U", "FREQ"];
    check_pair(false, &axes, &axes, &["RA", "DEC", "STOKES", "FREQ"]);
}

#[test]
fn freq_linear_feed_stokes() {
    let axes = ["RA", "DEC", "FREQ", "XX"];
    check_pair(false, &axes, &axes, &["RA", "DEC", "FREQ", "STOKES"]);
}

#[test]
fn freq_with_time_axis() {
    let axes = ["RA", "DEC", "TIME", "FREQ"];
    check_pair(false, &axes, &axes, &axes);
}

#[test]
fn freq_5d() {
    let axes = ["RA", "DEC", "FREQ", "TIME", "V"];
    check_pair(
        false,
        &axes,
        &axes,
        &["RA", "DEC", "FREQ", "TIME", "STOKES"],
    );
}

// Time cubes. A missing TIME axis is appended.

#[test]
fn time_2d() {
    check_pair(true, &["RA", "DEC"], &["RA", "DEC"], &["RA", "DEC", "TIME"]);
}

#[test]
fn time_3d() {
    let axes = ["RA", "DEC", "TIME"];
    check_pair(true, &axes, &axes, &axes);
}

#[test]
fn time_appended_to_freq() {
    let axes = ["RA", "DEC", "FREQ"];
    check_pair(true, &axes, &axes, &["RA", "DEC", "FREQ", "TIME"]);
}

#[test]
fn time_appended_to_freq_stokes() {
    let axes = ["RA", "DEC", "FREQ", "I"];
    check_pair(true, &axes, &axes, &["RA", "DEC", "FREQ", "STOKES", "TIME"]);
}

#[test]
fn time_with_stokes() {
    let axes = ["RA", "DEC", "Q", "TIME"];
    check_pair(true, &axes, &axes, &["RA", "DEC", "STOKES", "TIME"]);
}

#[test]
fn time_foo_after_time() {
    let axes = ["RA", "DEC", "TIME", "FOO"];
    check_pair(true, &axes, &axes, &axes);
}

// Pairs that must be rejected

#[test]
fn raises_ra_dec_swapped() {
    check_pair_raises(
        false,
        &["RA", "DEC", "FREQ"],
        &["DEC", "RA", "FREQ"],
        axis_mismatch,
    );
}

#[test]
fn raises_2d_with_3d() {
    check_pair_raises(false, &["RA", "DEC"], &["RA", "DEC", "FREQ"], axis_mismatch);
}

#[test]
fn raises_extra_axis() {
    check_pair_raises(
        false,
        &["RA", "DEC", "FREQ"],
        &["RA", "DEC", "FOO", "FREQ"],
        axis_mismatch,
    );
}

#[test]
fn raises_stokes_freq_order() {
    check_pair_raises(
        false,
        &["RA", "DEC", "FREQ", "I"],
        &["RA", "DEC", "I", "FREQ"],
        axis_mismatch,
    );
}

#[test]
fn raises_stokes_missing() {
    check_pair_raises(
        false,
        &["RA", "DEC", "FREQ", "I"],
        &["RA", "DEC", "FREQ"],
        axis_mismatch,
    );
}

#[test]
fn raises_stokes_differ() {
    check_pair_raises(
        false,
        &["RA", "DEC", "FREQ", "I"],
        &["RA", "DEC", "FREQ", "Q"],
        stokes_mismatch,
    );
}

#[test]
fn raises_linear_stokes_differ() {
    check_pair_raises(
        false,
        &["RA", "DEC", "FREQ", "XX"],
        &["RA", "DEC", "FREQ", "YY"],
        stokes_mismatch,
    );
}

/// Frequency mode needs a FREQ axis, or REFFREQ on a 2D image
#[test]
fn raises_no_freq() {
    check_pair_raises(false, &["RA", "DEC", "FOO"], &["RA", "DEC", "FOO"], |_| {
        true
    });
}

#[test]
fn raises_time_vs_freq() {
    check_pair_raises(
        true,
        &["RA", "DEC", "TIME"],
        &["RA", "DEC", "FREQ"],
        axis_mismatch,
    );
}

#[test]
fn raises_time_stokes_differ() {
    check_pair_raises(
        true,
        &["RA", "DEC", "TIME", "I"],
        &["RA", "DEC", "TIME", "Q"],
        stokes_mismatch,
    );
}

/// A multi-Stokes cube with varying beams fails before any plane is written
#[test]
fn rejects_multi_stokes_beams() {
    let dir = tempfile::tempdir().unwrap();
    let files: Vec<PathBuf> = (0..2)
        .map(|index| {
            let i = index as f64;
            let path = dir.path().join(format!("plane_{index}.fits"));
            let desc = ImageDescription {
                data_type: ImageType::Double,
                dimensions: &[1, 2, 10, 10],
            };
            let mut f = FitsFile::create(&path)
                .with_custom_primary(&desc)
                .open()
                .unwrap();
            let hdu = f.primary_hdu().unwrap();
            hdu.write_image(&mut f, &vec![1.0 + i; 200]).unwrap();
            hdu.write_key(&mut f, "CTYPE3", "STOKES").unwrap();
            hdu.write_key(&mut f, "CRVAL3", 1.0).unwrap();
            hdu.write_key(&mut f, "CDELT3", 1.0).unwrap();
            hdu.write_key(&mut f, "CRPIX3", 1.0).unwrap();
            hdu.write_key(&mut f, "CTYPE4", "FREQ").unwrap();
            hdu.write_key(&mut f, "CRVAL4", 1e9 + i * 1e6).unwrap();
            hdu.write_key(&mut f, "CDELT4", 1e6).unwrap();
            hdu.write_key(&mut f, "CRPIX4", 1.0).unwrap();
            hdu.write_key(&mut f, "CUNIT4", "Hz").unwrap();
            hdu.write_key(&mut f, "BMAJ", 1e-3 * (1.0 + i)).unwrap();
            hdu.write_key(&mut f, "BMIN", 1e-3).unwrap();
            hdu.write_key(&mut f, "BPA", 0.0).unwrap();
            path
        })
        .collect();

    let out_cube = dir.path().join("cube.fits");
    let options = CombineOptions {
        overwrite: true,
        ..Default::default()
    };
    let err = combine_fits(&files, &out_cube, &options).unwrap_err();
    assert!(
        matches!(err, FitsCubeError::NotImplemented(ref msg) if msg.contains("single-Stokes")),
        "{err:?}"
    );
    assert!(!out_cube.exists());
}
