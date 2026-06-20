#![allow(unsafe_op_in_unsafe_fn)]
//! Python bindings (`fitscube_rs._fitscube_rs`).
//!
//! Thin wrappers over [`crate::combine`] and [`crate::extract`]; all the FITS
//! work happens in Rust against file paths.
use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
#[cfg(feature = "stubgen")]
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::combine::{CombineOptions, combine_fits as rust_combine_fits};
use crate::error::FitsCubeError;
use crate::extract::{ExtractOptions, extract_plane_from_cube as rust_extract};

fn to_py_err(e: FitsCubeError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Combine single-plane FITS images into a cube.
///
/// Args:
///     file_list (list[str]): Paths of the FITS images to combine.
///     out_cube (str): Output cube path.
///     spec_file (str, optional): File of frequencies (Hz) / times (MJD s),
///         one per line. Mutually exclusive with ``spec_list``.
///     spec_list (list[float], optional): Frequencies/times supplied directly.
///         Mutually exclusive with ``spec_file``.
///     ignore_spec (bool): Ignore frequency/time info and just stack.
///     create_blanks (bool): Interpolate an evenly-spaced axis, blanking gaps.
///     overwrite (bool): Overwrite the output cube if it exists.
///     max_workers (int, optional): Concurrency bound for in-flight planes.
///     time_domain_mode (bool): Combine along time (DATE-OBS) instead of FREQ.
///     bounding_box (bool): Trim blank padding via a common bounding box.
///     invalidate_zeros (bool): Treat exactly-zero pixels as NaN.
///     float_length (int, optional): Output precision in bits (32 or 64).
///
/// Returns:
///     list[float]: The output-axis values (Hz for frequency, MJD s for time).
///
/// Raises:
///     ValueError: On invalid input or a FITS error.
#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (
    file_list,
    out_cube,
    spec_file=None,
    spec_list=None,
    ignore_spec=false,
    create_blanks=false,
    overwrite=false,
    max_workers=None,
    time_domain_mode=false,
    bounding_box=false,
    invalidate_zeros=false,
    float_length=None,
))]
#[allow(clippy::too_many_arguments)]
fn combine_fits(
    file_list: Vec<PathBuf>,
    out_cube: PathBuf,
    spec_file: Option<PathBuf>,
    spec_list: Option<Vec<f64>>,
    ignore_spec: bool,
    create_blanks: bool,
    overwrite: bool,
    max_workers: Option<usize>,
    time_domain_mode: bool,
    bounding_box: bool,
    invalidate_zeros: bool,
    float_length: Option<u8>,
) -> PyResult<Vec<f64>> {
    let options = CombineOptions {
        spec_file,
        spec_list,
        ignore_spec,
        create_blanks,
        overwrite,
        max_workers,
        time_domain_mode,
        bounding_box,
        invalidate_zeros,
        float_length,
    };
    rust_combine_fits(&file_list, &out_cube, &options).map_err(to_py_err)
}

/// Extract a single plane (channel or timestep) from a FITS cube.
///
/// Args:
///     fits_cube (str): The cube to extract from.
///     channel_index (int, optional): Frequency channel to extract. Mutually
///         exclusive with ``time_index``.
///     time_index (int, optional): Timestep to extract. Mutually exclusive with
///         ``channel_index``.
///     hdu_index (int): HDU index of the cube data (default 0).
///     overwrite (bool): Overwrite the output file if it exists.
///     output_path (str, optional): Output path; generated from the cube name
///         if omitted.
///
/// Returns:
///     str: Path of the written plane image.
///
/// Raises:
///     ValueError: On invalid options or a FITS error.
#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (
    fits_cube,
    channel_index=None,
    time_index=None,
    hdu_index=0,
    overwrite=false,
    output_path=None,
))]
fn extract_plane_from_cube(
    fits_cube: PathBuf,
    channel_index: Option<usize>,
    time_index: Option<usize>,
    hdu_index: usize,
    overwrite: bool,
    output_path: Option<PathBuf>,
) -> PyResult<String> {
    let options = ExtractOptions {
        hdu_index,
        channel_index,
        time_index,
        overwrite,
        output_path,
    };
    rust_extract(&fits_cube, &options)
        .map(|p| p.to_string_lossy().into_owned())
        .map_err(to_py_err)
}

#[cfg(feature = "stubgen")]
pyo3_stub_gen::define_stub_info_gatherer!(stub_info);

#[cfg(feature = "stubgen")]
#[pyfunction]
fn _generate_stubs() -> PyResult<()> {
    if std::env::var("CARGO_MANIFEST_DIR").is_err() {
        let cwd = std::env::current_dir()
            .map_err(|e| PyValueError::new_err(format!("cannot get cwd: {e}")))?;
        #[allow(unused_unsafe)]
        unsafe {
            std::env::set_var("CARGO_MANIFEST_DIR", cwd);
        }
    }
    stub_info()
        .and_then(|s| s.generate())
        .map_err(|e| PyValueError::new_err(format!("stub generation failed: {e}")))
}

#[pymodule]
pub fn _fitscube_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(combine_fits, m)?)?;
    m.add_function(wrap_pyfunction!(extract_plane_from_cube, m)?)?;
    #[cfg(feature = "stubgen")]
    m.add_function(wrap_pyfunction!(_generate_stubs, m)?)?;
    Ok(())
}
