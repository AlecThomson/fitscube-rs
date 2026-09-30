//! Regression tests for the cube-scrambling, blank-channel and beam bugs.
//!
//! Port of `tests/test_combine_guards.py`, `tests/test_zero_beams.py` and the
//! beam tests of `tests/test_combine.py` in fitscube.
use std::path::{Path, PathBuf};
use std::process::Command;

use fitscube_rs::bounding_box::{BoundingBox, get_common_bounding_box};
use fitscube_rs::checks::{check_matching_axes, check_matching_shapes, read_inputs_axes};
use fitscube_rs::combine::{CombineOptions, combine_fits};
use fitscube_rs::error::FitsCubeError;
use fitscube_rs::fits_io::update_key_f64;
use fitsio::FitsFile;
use fitsio::images::{ImageDescription, ImageType};

/// `numpy.finfo(numpy.float32).tiny`, the NaN-PSF sentinel of the BEAMS table.
const TINY: f32 = f32::MIN_POSITIVE;

/// Four channels 1 MHz apart from 1 GHz.
fn specs() -> Vec<f64> {
    (0..4).map(|i| 1e9 + i as f64 * 1e6).collect()
}

/// How [`make_plane`] writes an image.
#[derive(Clone, Copy)]
struct Plane {
    /// numpy (C-order) shape.
    shape: &'static [usize],
    value: f64,
    /// (STOKES, FREQ, DEC, RA) instead of (FREQ, STOKES, DEC, RA).
    pol_outer: bool,
    beam: Option<f64>,
    /// Write BITPIX 16 instead of -32.
    int16: bool,
}

impl Default for Plane {
    fn default() -> Self {
        Self {
            shape: &[1, 1, 8, 8],
            value: 1.0,
            pol_outer: false,
            beam: None,
            int16: false,
        }
    }
}

/// Write a single-channel image with a FREQ axis at `spec`.
///
/// By default the axes are (FREQ, STOKES, DEC, RA), the usual ASKAP ordering.
/// With `pol_outer` they are (STOKES, FREQ, DEC, RA), which puts a
/// non-degenerate STOKES axis above FREQ in the output cube.
fn make_plane(path: &Path, spec: f64, plane: Plane) -> PathBuf {
    let desc = ImageDescription {
        data_type: if plane.int16 {
            ImageType::Short
        } else {
            ImageType::Float
        },
        dimensions: plane.shape,
    };
    let mut f = FitsFile::create(path)
        .with_custom_primary(&desc)
        .overwrite()
        .open()
        .unwrap();
    let hdu = f.primary_hdu().unwrap();
    let n: usize = plane.shape.iter().product();
    if plane.int16 {
        hdu.write_image(&mut f, &vec![plane.value as i16; n])
            .unwrap();
    } else {
        hdu.write_image(&mut f, &vec![plane.value as f32; n])
            .unwrap();
    }
    let (spec_axis, pol_axis) = if plane.pol_outer { (3, 4) } else { (4, 3) };
    let cards: [(&str, &str); 5] = [
        ("CTYPE1", "RA---SIN"),
        ("CUNIT1", "deg"),
        ("CTYPE2", "DEC--SIN"),
        ("CUNIT2", "deg"),
        ("", ""),
    ];
    for (key, value) in cards.iter().filter(|(k, _)| !k.is_empty()) {
        hdu.write_key(&mut f, key, *value).unwrap();
    }
    for (key, value) in [
        ("CRPIX1", 1.0),
        ("CRVAL1", 0.0),
        ("CDELT1", -1e-3),
        ("CRPIX2", 1.0),
        ("CRVAL2", 0.0),
        ("CDELT2", 1e-3),
    ] {
        hdu.write_key(&mut f, key, value).unwrap();
    }
    hdu.write_key(&mut f, &format!("CTYPE{spec_axis}"), "FREQ")
        .unwrap();
    hdu.write_key(&mut f, &format!("CRPIX{spec_axis}"), 1.0)
        .unwrap();
    hdu.write_key(&mut f, &format!("CRVAL{spec_axis}"), spec)
        .unwrap();
    hdu.write_key(&mut f, &format!("CDELT{spec_axis}"), 1e6)
        .unwrap();
    hdu.write_key(&mut f, &format!("CUNIT{spec_axis}"), "Hz")
        .unwrap();
    hdu.write_key(&mut f, &format!("CTYPE{pol_axis}"), "STOKES")
        .unwrap();
    hdu.write_key(&mut f, &format!("CRPIX{pol_axis}"), 1.0)
        .unwrap();
    hdu.write_key(&mut f, &format!("CRVAL{pol_axis}"), 1.0)
        .unwrap();
    hdu.write_key(&mut f, &format!("CDELT{pol_axis}"), 1.0)
        .unwrap();
    if let Some(beam) = plane.beam {
        hdu.write_key(&mut f, "BMAJ", beam).unwrap();
        hdu.write_key(&mut f, "BMIN", beam / 2.0).unwrap();
        hdu.write_key(&mut f, "BPA", 0.0).unwrap();
    }
    path.to_path_buf()
}

/// One plane per spec, named `{stem}_{i}.fits`, with `plane(i)` options.
fn make_planes(
    dir: &Path,
    stem: &str,
    specs: &[f64],
    plane: impl Fn(usize) -> Plane,
) -> Vec<PathBuf> {
    specs
        .iter()
        .enumerate()
        .map(|(i, &spec)| make_plane(&dir.join(format!("{stem}_{i}.fits")), spec, plane(i)))
        .collect()
}

/// The default file list: four planes holding their index.
fn file_list(dir: &Path) -> Vec<PathBuf> {
    make_planes(dir, "plane", &specs(), |i| Plane {
        value: i as f64,
        ..Default::default()
    })
}

/// Update a card in place (`write_key` would append a duplicate card).
fn set_key(path: &Path, key: &str, value: f64) {
    let mut f = FitsFile::edit(path.to_string_lossy().as_ref()).unwrap();
    update_key_f64(&mut f, key, value).unwrap();
}

fn options() -> CombineOptions {
    CombineOptions {
        overwrite: true,
        ..Default::default()
    }
}

fn read_i64(path: &Path, key: &str) -> i64 {
    let mut f = FitsFile::open(path.to_string_lossy().as_ref()).unwrap();
    let hdu = f.primary_hdu().unwrap();
    hdu.read_key(&mut f, key).unwrap()
}

/// The cube's planes, one `Vec` per channel (all non-spatial axes but the
/// combine axis are degenerate in these tests).
fn read_planes(path: &Path) -> Vec<Vec<f64>> {
    let mut f = FitsFile::open(path.to_string_lossy().as_ref()).unwrap();
    let hdu = f.primary_hdu().unwrap();
    let naxis = read_i64(path, "NAXIS");
    let n_chan = read_i64(path, &format!("NAXIS{naxis}")) as usize;
    let data: Vec<f64> = hdu.read_image(&mut f).unwrap();
    data.chunks(data.len() / n_chan)
        .map(<[f64]>::to_vec)
        .collect()
}

fn read_beam_col(path: &Path, col: &str) -> Vec<f32> {
    let mut f = FitsFile::open(path.to_string_lossy().as_ref()).unwrap();
    let hdu = f.hdu("BEAMS").unwrap();
    hdu.read_col(&mut f, col).unwrap()
}

fn all_eq(plane: &[f64], value: f64) -> bool {
    plane
        .iter()
        .all(|&v| (v - value).abs() <= 1e-6 * value.abs().max(1.0))
}

fn all_nan(plane: &[f64]) -> bool {
    plane.iter().all(|v| v.is_nan())
}

fn close(a: f32, b: f64) -> bool {
    (a as f64 - b).abs() <= 1e-5 * b.abs()
}

// Input checks

/// Differing NAXIS1/NAXIS2 used to slide the planes against each other.
#[test]
fn mismatched_shapes_raise() {
    let dir = tempfile::tempdir().unwrap();
    let mut files = file_list(dir.path());
    files.push(make_plane(
        &dir.path().join("plane_odd.fits"),
        specs()[3] + 1e6,
        Plane {
            shape: &[1, 1, 6, 6],
            ..Default::default()
        },
    ));
    let out_cube = dir.path().join("cube.fits");
    let err = combine_fits(&files, &out_cube, &options()).unwrap_err();
    assert!(
        matches!(err, FitsCubeError::ShapeMismatch(ref msg) if msg.contains("plane_odd")),
        "{err:?}"
    );
    assert!(!out_cube.exists());
}

#[test]
fn check_matching_shapes_returns_grid() {
    let dir = tempfile::tempdir().unwrap();
    let files = file_list(dir.path());
    let axes = read_inputs_axes(&files).unwrap();
    assert_eq!(check_matching_shapes(&files, &axes).unwrap(), (8, 8));
}

#[test]
fn check_matching_axes_accepts_common_stokes() {
    let dir = tempfile::tempdir().unwrap();
    let files = file_list(dir.path());
    for path in &files {
        set_key(path, "CRVAL3", 3.0); // Stokes U
    }
    check_matching_axes(&files, &read_inputs_axes(&files).unwrap()).unwrap();
}

/// CRVAL3=2 at CRPIX3=2 is still Stokes I at the (only) first pixel
#[test]
fn check_matching_axes_compares_stokes_codes_not_keywords() {
    let dir = tempfile::tempdir().unwrap();
    let files = file_list(dir.path());
    set_key(&files[3], "CRVAL3", 2.0);
    set_key(&files[3], "CRPIX3", 2.0);
    check_matching_axes(&files, &read_inputs_axes(&files).unwrap()).unwrap();
}

/// A Q plane among I planes would be mislabelled as I in the cube
#[test]
fn mismatched_stokes_raise() {
    let dir = tempfile::tempdir().unwrap();
    let files = file_list(dir.path());
    set_key(&files[2], "CRVAL3", 2.0); // Stokes Q
    let out_cube = dir.path().join("cube.fits");
    let err = combine_fits(&files, &out_cube, &options()).unwrap_err();
    assert!(
        matches!(err, FitsCubeError::StokesMismatch(ref msg) if msg.contains("plane_2.fits")),
        "{err:?}"
    );
    assert!(!out_cube.exists());
}

/// A non-degenerate axis above FREQ breaks the per-plane seek.
#[test]
fn spectral_axis_must_be_slowest() {
    let dir = tempfile::tempdir().unwrap();
    let files = make_planes(dir.path(), "pol", &specs(), |_| Plane {
        shape: &[2, 1, 8, 8],
        pol_outer: true,
        ..Default::default()
    });
    let out_cube = dir.path().join("cube.fits");
    let err = combine_fits(&files, &out_cube, &options()).unwrap_err();
    assert!(
        matches!(err, FitsCubeError::AxisOrder(ref msg) if msg.contains("NAXIS4")),
        "{err:?}"
    );
    assert!(!out_cube.exists());
}

/// Several Stokes planes per input below FREQ are copied whole, not truncated
/// to one 2D plane per channel.
#[test]
fn multi_plane_inputs_below_spectral_axis() {
    let dir = tempfile::tempdir().unwrap();
    let files = make_planes(dir.path(), "stokes", &specs(), |i| Plane {
        shape: &[1, 2, 8, 8],
        value: i as f64,
        ..Default::default()
    });
    let out_cube = dir.path().join("cube.fits");
    combine_fits(&files, &out_cube, &options()).unwrap();
    assert_eq!(read_i64(&out_cube, "NAXIS3"), 2);
    let planes = read_planes(&out_cube);
    assert_eq!(planes.len(), 4);
    for (chan, plane) in planes.iter().enumerate() {
        assert_eq!(plane.len(), 128);
        assert!(all_eq(plane, chan as f64), "channel {chan}");
    }
}

// Output precision

#[test]
fn cube_keeps_input_precision() {
    let dir = tempfile::tempdir().unwrap();
    let files = file_list(dir.path());
    let out_cube = dir.path().join("cube.fits");
    combine_fits(&files, &out_cube, &options()).unwrap();
    assert_eq!(read_i64(&out_cube, "BITPIX"), -32);
    assert_eq!(read_i64(&out_cube, "NAXIS4"), 4);
    for (chan, plane) in read_planes(&out_cube).iter().enumerate() {
        assert!(all_eq(plane, chan as f64), "channel {chan}");
    }
}

#[test]
fn float_length_is_respected() {
    let dir = tempfile::tempdir().unwrap();
    let files = file_list(dir.path());
    let out_cube = dir.path().join("cube.fits");
    let opts = CombineOptions {
        float_length: Some(64),
        ..options()
    };
    combine_fits(&files, &out_cube, &opts).unwrap();
    assert_eq!(read_i64(&out_cube, "BITPIX"), -64);
    // 4 planes of 64 pixels at 8 bytes, after one 2880-byte header block
    let len = std::fs::metadata(&out_cube).unwrap().len();
    assert!(len >= 2880 + 4 * 64 * 8, "{len}");
}

/// Integer inputs are written as integers, not as floats under an integer
/// BITPIX.
#[test]
fn integer_input_is_not_cast_to_float() {
    let dir = tempfile::tempdir().unwrap();
    let files = make_planes(dir.path(), "int", &specs(), |i| Plane {
        value: i as f64,
        int16: true,
        ..Default::default()
    });
    let out_cube = dir.path().join("cube.fits");
    combine_fits(&files, &out_cube, &options()).unwrap();
    assert_eq!(read_i64(&out_cube, "BITPIX"), 16);
    for (chan, plane) in read_planes(&out_cube).iter().enumerate() {
        assert!(
            all_eq(plane, chan as f64),
            "channel {chan}: {:?}",
            &plane[..4]
        );
    }
}

/// NaN blanks cannot be stored in an integer cube.
#[test]
fn blanks_need_float_output() {
    let dir = tempfile::tempdir().unwrap();
    let s = specs();
    let files = make_planes(dir.path(), "int_gap", &[s[0], s[2], s[3]], |_| Plane {
        int16: true,
        ..Default::default()
    });
    let out_cube = dir.path().join("cube.fits");
    let opts = CombineOptions {
        create_blanks: true,
        ..options()
    };
    let err = combine_fits(&files, &out_cube, &opts).unwrap_err();
    assert!(err.to_string().contains("float_length"), "{err:?}");
    assert!(!out_cube.exists());

    let opts = CombineOptions {
        float_length: Some(32),
        ..opts
    };
    combine_fits(&files, &out_cube, &opts).unwrap();
    assert!(all_nan(&read_planes(&out_cube)[1]));
}

// Spectral axis

#[test]
fn blank_channels_are_created() {
    let dir = tempfile::tempdir().unwrap();
    let s = specs();
    let files = make_planes(dir.path(), "gap", &[s[0], s[2], s[3]], |i| Plane {
        value: i as f64,
        ..Default::default()
    });
    let out_cube = dir.path().join("cube.fits");
    let opts = CombineOptions {
        create_blanks: true,
        ..options()
    };
    let out_specs = combine_fits(&files, &out_cube, &opts).unwrap();
    assert_eq!(out_specs.len(), 4);
    let planes = read_planes(&out_cube);
    assert!(all_eq(&planes[0], 0.0));
    assert!(all_nan(&planes[1]));
    assert!(all_eq(&planes[2], 1.0));
    assert!(all_eq(&planes[3], 2.0));
}

/// even_spacing builds the grid from the end points, so it must sort first.
#[test]
fn unsorted_input_with_blanks() {
    let dir = tempfile::tempdir().unwrap();
    let s = specs();
    let gapped = [s[3], s[0], s[2]];
    let files = make_planes(dir.path(), "unsorted", &gapped, |i| Plane {
        value: gapped[i],
        ..Default::default()
    });
    let out_cube = dir.path().join("cube.fits");
    let opts = CombineOptions {
        create_blanks: true,
        ..options()
    };
    let out_specs = combine_fits(&files, &out_cube, &opts).unwrap();
    for (got, want) in out_specs.iter().zip(&s) {
        assert!((got - want).abs() < 1.0, "{out_specs:?}");
    }
    let planes = read_planes(&out_cube);
    assert!(all_nan(&planes[1]));
    for chan in [0, 2, 3] {
        assert!(
            all_eq(&planes[chan], s[chan] as f32 as f64),
            "channel {chan}"
        );
    }
}

#[test]
fn spec_list_is_used() {
    let dir = tempfile::tempdir().unwrap();
    let files = file_list(dir.path());
    let spec_list = vec![2e9, 2.1e9, 2.2e9, 2.3e9];
    let opts = CombineOptions {
        spec_list: Some(spec_list.clone()),
        ..options()
    };
    let out_specs = combine_fits(&files, &dir.path().join("cube.fits"), &opts).unwrap();
    assert_eq!(out_specs, spec_list);
}

#[test]
fn spec_file_is_used() {
    let dir = tempfile::tempdir().unwrap();
    let files = file_list(dir.path());
    let spec_file = dir.path().join("specs.txt");
    std::fs::write(&spec_file, "2e9\n2.1e9\n2.2e9\n2.3e9\n").unwrap();
    let opts = CombineOptions {
        spec_file: Some(spec_file),
        ..options()
    };
    let out_specs = combine_fits(&files, &dir.path().join("cube.fits"), &opts).unwrap();
    assert_eq!(out_specs, vec![2e9, 2.1e9, 2.2e9, 2.3e9]);
}

/// A grid that cannot hold every input used to write a cube missing channels.
#[test]
fn irregular_spacing_never_drops_inputs() {
    let dir = tempfile::tempdir().unwrap();
    // Irregular, with no common step that a grid could recover (irrational
    // offsets, like the random floats of the Python test)
    use std::f64::consts::{E, LN_10, PI, SQRT_2};
    let irregular: Vec<f64> = [
        0.0,
        LN_10,
        E,
        SQRT_2 * 2.0,
        PI,
        3.0 * E / 2.0,
        12f64.sqrt() * 2.0,
    ]
    .iter()
    .map(|x| 1e9 + x * 1e7)
    .collect();
    let files = make_planes(dir.path(), "irregular", &irregular, |i| Plane {
        value: i as f64,
        ..Default::default()
    });
    let out_cube = dir.path().join("cube.fits");
    let opts = CombineOptions {
        create_blanks: true,
        ..options()
    };
    let err = combine_fits(&files, &out_cube, &opts).unwrap_err();
    assert!(
        matches!(err, FitsCubeError::IrregularSpacing(ref msg) if msg.contains("would drop inputs")),
        "{err:?}"
    );
    assert!(!out_cube.exists());

    // Without blanks the irregular axis is kept and every input is written
    let out_specs = combine_fits(&files, &out_cube, &options()).unwrap();
    assert_eq!(out_specs.len(), irregular.len());
    for (got, want) in out_specs.iter().zip(&irregular) {
        // Header cards round the frequencies to ~15 significant digits
        assert!((got - want).abs() < 1e-3, "{out_specs:?}");
    }
    for (chan, plane) in read_planes(&out_cube).iter().enumerate() {
        assert!(all_eq(plane, chan as f64), "channel {chan}");
    }
}

/// Two inputs collapsing onto one grid point would lose one of them.
#[test]
fn duplicate_frequencies_never_drop_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let s = specs();
    let files = make_planes(dir.path(), "duplicate", &[s[0], s[0], s[2]], |i| Plane {
        value: i as f64,
        ..Default::default()
    });
    let opts = CombineOptions {
        create_blanks: true,
        ..options()
    };
    let err = combine_fits(&files, &dir.path().join("cube.fits"), &opts).unwrap_err();
    assert!(matches!(err, FitsCubeError::IrregularSpacing(_)), "{err:?}");
}

/// A single input with create_blanks is its own (trivially even) grid.
#[test]
fn single_input_with_blanks() {
    let dir = tempfile::tempdir().unwrap();
    let files = make_planes(dir.path(), "single", &specs()[..1], |_| Plane::default());
    let opts = CombineOptions {
        create_blanks: true,
        ..options()
    };
    let out_specs = combine_fits(&files, &dir.path().join("cube.fits"), &opts).unwrap();
    assert_eq!(out_specs, vec![specs()[0]]);
}

// Beams

/// The beam table used to be one row per file, so beams slid past the gap.
#[test]
fn beam_table_follows_blank_channels() {
    let dir = tempfile::tempdir().unwrap();
    let s = specs();
    let files = make_planes(dir.path(), "gap", &[s[0], s[2], s[3]], |i| Plane {
        value: i as f64,
        beam: Some(1e-3 * (i as f64 + 1.0)),
        ..Default::default()
    });
    let out_cube = dir.path().join("cube.fits");
    let opts = CombineOptions {
        create_blanks: true,
        ..options()
    };
    combine_fits(&files, &out_cube, &opts).unwrap();

    assert_eq!(read_i64(&out_cube, "NAXIS4"), 4);
    let chan = read_beam_col(&out_cube, "CHAN");
    assert_eq!(chan, vec![0.0, 1.0, 2.0, 3.0]);
    let majors = read_beam_col(&out_cube, "BMAJ");
    let expected = 1e-3 * 3600.0;
    assert!(close(majors[0], expected), "{majors:?}");
    assert_eq!(majors[1], TINY, "the blank channel");
    assert!(close(majors[2], 2.0 * expected), "{majors:?}");
    assert!(close(majors[3], 3.0 * expected), "{majors:?}");
}

/// Beams follow the channel order of the cube, not the order the files were
/// given in.
#[test]
fn beam_table_follows_sorted_channels() {
    let dir = tempfile::tempdir().unwrap();
    let s = specs();
    let unsorted = [s[2], s[0], s[3], s[1]];
    let files = make_planes(dir.path(), "unsorted", &unsorted, |i| Plane {
        value: i as f64,
        // The beam tracks the frequency, so it must end up sorted too
        beam: Some(1e-3 * (1.0 + (unsorted[i] - s[0]) / 1e6)),
        ..Default::default()
    });
    let out_cube = dir.path().join("cube.fits");
    combine_fits(&files, &out_cube, &options()).unwrap();
    let majors = read_beam_col(&out_cube, "BMAJ");
    for (chan, &major) in majors.iter().enumerate() {
        assert!(close(major, 3.6 * (chan as f64 + 1.0)), "{majors:?}");
    }
}

/// A beam only on the fourth input should still reach the output cube
#[test]
fn beam_not_in_first_file() {
    let dir = tempfile::tempdir().unwrap();
    let files = make_planes(dir.path(), "onebeam", &specs(), |i| Plane {
        value: i as f64,
        beam: (i == 3).then_some(1.0),
        ..Default::default()
    });
    let out_cube = dir.path().join("cube.fits");
    combine_fits(&files, &out_cube, &options()).unwrap();
    let majors = read_beam_col(&out_cube, "BMAJ");
    // Only plane 3 carries a beam; the rest are the NaN sentinel
    assert!(close(majors[3], 3600.0), "{majors:?}");
    assert!(majors[..3].iter().all(|&m| m == TINY), "{majors:?}");
}

// Zero beams: wsclean writes BMAJ = BMIN = 0 when a plane holds no fitted PSF.
// With `-fit-spectral-pol` that plane is the model image, which looks like real
// data but is not comparable to the rest of the cube.

/// Four planes holding `i + 1`, where the second carries a zero beam.
fn zero_beam_file_list(dir: &Path) -> Vec<PathBuf> {
    make_planes(dir, "plane", &specs(), |i| Plane {
        value: i as f64 + 1.0,
        beam: Some(if i == 1 { 0.0 } else { 1e-3 * (i as f64 + 1.0) }),
        ..Default::default()
    })
}

/// The zero-beam plane is NaN-ed out; every other plane is untouched.
#[test]
fn zero_beam_plane_is_blanked() {
    let dir = tempfile::tempdir().unwrap();
    let files = zero_beam_file_list(dir.path());
    let out_cube = dir.path().join("cube.fits");
    combine_fits(&files, &out_cube, &options()).unwrap();

    let planes = read_planes(&out_cube);
    assert!(all_nan(&planes[1]));
    assert!(all_eq(&planes[0], 1.0));
    assert!(all_eq(&planes[2], 3.0));
    assert!(all_eq(&planes[3], 4.0));

    // The blanked beam is stored with the same NaN sentinel as any other NaN PSF
    let majors = read_beam_col(&out_cube, "BMAJ");
    let minors = read_beam_col(&out_cube, "BMIN");
    assert_eq!(majors[1], TINY);
    assert_eq!(minors[1], TINY);
    assert!(close(majors[0], 1e-3 * 3600.0), "{majors:?}");
}

/// With blank_zero_beams = false the image data is kept.
#[test]
fn zero_beam_opt_out() {
    let dir = tempfile::tempdir().unwrap();
    let files = zero_beam_file_list(dir.path());
    let out_cube = dir.path().join("cube.fits");
    let opts = CombineOptions {
        blank_zero_beams: false,
        ..options()
    };
    combine_fits(&files, &out_cube, &opts).unwrap();
    assert!(all_eq(&read_planes(&out_cube)[1], 2.0));
    // The data survives, but a literal zero must never reach the beam table
    assert_eq!(read_beam_col(&out_cube, "BMAJ")[1], TINY);
}

/// A zero BMAJ/BMIN always becomes the sentinel, whichever way we are called.
#[test]
fn beam_table_never_holds_a_zero_beam() {
    let dir = tempfile::tempdir().unwrap();
    let files = zero_beam_file_list(dir.path());
    for blank in [true, false] {
        let out_cube = dir.path().join(format!("cube_{blank}.fits"));
        let opts = CombineOptions {
            blank_zero_beams: blank,
            ..options()
        };
        combine_fits(&files, &out_cube, &opts).unwrap();
        let bmaj = read_beam_col(&out_cube, "BMAJ");
        let bmin = read_beam_col(&out_cube, "BMIN");
        let bpa = read_beam_col(&out_cube, "BPA");
        assert!(!bmaj.contains(&0.0), "{blank}: {bmaj:?}");
        assert!(!bmin.contains(&0.0), "{blank}: {bmin:?}");
        assert_eq!((bmaj[1], bmin[1], bpa[1]), (TINY, TINY, TINY), "{blank}");
        // A zero position angle on a real beam is legitimate and must survive,
        // which proves the sentinel is applied per-beam, not per-column.
        assert_eq!([bpa[0], bpa[2], bpa[3]], [0.0; 3], "{blank}");
    }
}

/// Cubes without a zero beam behave exactly as before.
#[test]
fn no_zero_beams_are_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let files = make_planes(dir.path(), "plane", &specs(), |i| Plane {
        value: i as f64 + 1.0,
        beam: Some(1e-3 * (i as f64 + 1.0)),
        ..Default::default()
    });
    let out_cube = dir.path().join("cube.fits");
    combine_fits(&files, &out_cube, &options()).unwrap();
    assert!(read_planes(&out_cube).iter().flatten().all(|v| !v.is_nan()));
}

/// A zero beam and a create_blanks gap must not shift each other.
#[test]
fn zero_beam_and_blank_channels() {
    let dir = tempfile::tempdir().unwrap();
    let s = specs();
    let beams = [1e-3, 0.0, 3e-3];
    let files = make_planes(dir.path(), "gap", &[s[0], s[2], s[3]], |i| Plane {
        value: i as f64 + 1.0,
        beam: Some(beams[i]),
        ..Default::default()
    });
    let out_cube = dir.path().join("cube.fits");
    let opts = CombineOptions {
        create_blanks: true,
        ..options()
    };
    combine_fits(&files, &out_cube, &opts).unwrap();

    // Channel 1 is the missing channel, channel 2 is the zero-beam plane
    let planes = read_planes(&out_cube);
    assert!(all_eq(&planes[0], 1.0));
    assert!(all_nan(&planes[1]));
    assert!(all_nan(&planes[2]));
    assert!(all_eq(&planes[3], 3.0));
    let majors = read_beam_col(&out_cube, "BMAJ");
    assert_eq!(majors[1], TINY);
    assert_eq!(majors[2], TINY);
    assert!(close(majors[3], 3e-3 * 3600.0), "{majors:?}");
}

/// Zero-beam planes are blanked with NaNs, so an integer cube cannot hold them.
#[test]
fn zero_beams_need_float_output() {
    let dir = tempfile::tempdir().unwrap();
    let files = make_planes(dir.path(), "int", &specs(), |i| Plane {
        beam: Some(if i == 1 { 0.0 } else { 1e-3 }),
        int16: true,
        ..Default::default()
    });
    let out_cube = dir.path().join("cube.fits");
    let err = combine_fits(&files, &out_cube, &options()).unwrap_err();
    assert!(err.to_string().contains("float_length"), "{err:?}");
    assert!(!out_cube.exists());
}

fn fitscubers(files: &[PathBuf], out_cube: &Path, extra: &[&str]) {
    let status = Command::new(env!("CARGO_BIN_EXE_fitscubers"))
        .arg("combine")
        .args(files)
        .arg(out_cube)
        .args(extra)
        .status()
        .unwrap();
    assert!(status.success());
}

#[test]
fn cli_no_blank_zero_beams() {
    let dir = tempfile::tempdir().unwrap();
    let files = zero_beam_file_list(dir.path());
    let out_cube = dir.path().join("cube.fits");
    fitscubers(&files, &out_cube, &["--no-blank-zero-beams"]);
    assert!(all_eq(&read_planes(&out_cube)[1], 2.0));
}

#[test]
fn cli_blanks_zero_beams_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let files = zero_beam_file_list(dir.path());
    let out_cube = dir.path().join("cube.fits");
    fitscubers(&files, &out_cube, &[]);
    assert!(all_nan(&read_planes(&out_cube)[1]));
}

// Bounding boxes

/// Blank the first `rows` rows of every plane of an image.
fn blank_rows(path: &Path, rows: usize) {
    let mut f = FitsFile::edit(path.to_string_lossy().as_ref()).unwrap();
    let hdu = f.primary_hdu().unwrap();
    let mut data: Vec<f32> = hdu.read_image(&mut f).unwrap();
    for row in data.chunks_mut(8).take(rows) {
        row.fill(f32::NAN);
    }
    hdu.write_image(&mut f, &data).unwrap();
}

/// A caller-supplied box is used as is, so two cubes can share a grid.
#[test]
fn bounding_box_can_be_supplied() {
    let dir = tempfile::tempdir().unwrap();
    // Images blanked to different extents, as per-channel linmos mosaics are
    let images = make_planes(dir.path(), "image", &specs(), |i| Plane {
        value: i as f64,
        ..Default::default()
    });
    for (i, image) in images.iter().enumerate() {
        blank_rows(image, i + 1);
    }
    let weights = make_planes(dir.path(), "weight", &specs(), |_| Plane::default());

    let common_box = get_common_bounding_box(&images, false).unwrap();
    assert_eq!(common_box.x_span, 8 - 1); // first row blanked in every image

    let image_cube = dir.path().join("image_cube.fits");
    let weight_cube = dir.path().join("weight_cube.fits");
    for (files, out_cube) in [(&images, &image_cube), (&weights, &weight_cube)] {
        let opts = CombineOptions {
            supplied_bounding_box: Some(common_box),
            ..options()
        };
        combine_fits(files, out_cube, &opts).unwrap();
    }
    for key in ["NAXIS1", "NAXIS2", "NAXIS3", "NAXIS4"] {
        assert_eq!(
            read_i64(&image_cube, key),
            read_i64(&weight_cube, key),
            "{key}"
        );
    }
    for key in ["CRPIX1", "CRPIX2"] {
        let crpix = |path: &Path| {
            let mut f = FitsFile::open(path.to_string_lossy().as_ref()).unwrap();
            let hdu = f.primary_hdu().unwrap();
            hdu.read_key::<f64>(&mut f, key).unwrap()
        };
        assert_eq!(crpix(&image_cube), crpix(&weight_cube), "{key}");
    }
    assert_eq!(read_i64(&image_cube, "NAXIS2"), common_box.x_span as i64);
    assert!(
        read_planes(&weight_cube)
            .iter()
            .flatten()
            .all(|&v| v == 1.0)
    );

    // The weights alone would have given the full, untrimmed grid
    assert_eq!(get_common_bounding_box(&weights, false).unwrap().x_span, 8);
}

/// A supplied box from another pixel grid is refused, not sliced out of range.
#[test]
fn supplied_bounding_box_must_fit() {
    let dir = tempfile::tempdir().unwrap();
    let files = file_list(dir.path());
    let out_cube = dir.path().join("cube.fits");
    let opts = CombineOptions {
        supplied_bounding_box: Some(BoundingBox::new(0, 10, 0, 10, (10, 10))),
        ..options()
    };
    let err = combine_fits(&files, &out_cube, &opts).unwrap_err();
    assert!(matches!(err, FitsCubeError::ShapeMismatch(_)), "{err:?}");
    assert!(!out_cube.exists());
}

/// Integers stored with BZERO/BSCALE (e.g. unsigned 16-bit) are copied as their
/// stored values, so the cube decodes to the same physical values.
#[test]
fn scaled_integer_input_is_copied_losslessly() {
    let dir = tempfile::tempdir().unwrap();
    let files = make_planes(dir.path(), "uint", &specs(), |i| Plane {
        value: i as f64 * 1000.0 - 30000.0,
        int16: true,
        ..Default::default()
    });
    for path in &files {
        set_key(path, "BZERO", 32768.0);
        set_key(path, "BSCALE", 2.0);
    }
    let out_cube = dir.path().join("cube.fits");
    combine_fits(&files, &out_cube, &options()).unwrap();
    assert_eq!(read_i64(&out_cube, "BITPIX"), 16);
    for (chan, plane) in read_planes(&out_cube).iter().enumerate() {
        let physical = 32768.0 + 2.0 * (chan as f64 * 1000.0 - 30000.0);
        assert!(all_eq(plane, physical), "channel {chan}: {:?}", &plane[..2]);
    }
}
