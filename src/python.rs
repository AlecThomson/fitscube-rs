#![allow(unsafe_op_in_unsafe_fn)]
//! Python bindings (`fitscube_rs._fitscube_rs`).
//!
//! Thin wrappers over [`crate::combine`] and [`crate::extract`]; all the FITS
//! work happens in Rust against file paths.
use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
#[cfg(feature = "stubgen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::bounding_box::{BoundingBox, get_common_bounding_box as rust_common_bounding_box};
use crate::combine::{CombineOptions, combine_fits as rust_combine_fits};
use crate::error::FitsCubeError;
use crate::extract::{ExtractOptions, extract_plane_from_cube as rust_extract};

fn to_py_err(e: FitsCubeError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Pixel bounds of the valid data of an image, as used to trim a cube.
///
/// ``x`` and ``y`` are the *numpy* axes, the reverse of the FITS ``NAXIS``
/// convention: ``x`` is image rows (``NAXIS2``) and ``y`` is image columns
/// (``NAXIS1``). Minimum values are inclusive and maximum values exclusive, so
/// a plane is sliced as ``data[..., xmin:xmax, ymin:ymax]``.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "BoundingBox", module = "fitscube_rs._fitscube_rs", frozen, eq)]
#[derive(Clone, PartialEq)]
struct PyBoundingBox(BoundingBox);

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyBoundingBox {
    /// A box over rows ``xmin:xmax`` and columns ``ymin:ymax`` of a plane of
    /// shape ``original_shape`` (rows, columns).
    #[new]
    fn new(
        xmin: usize,
        xmax: usize,
        ymin: usize,
        ymax: usize,
        original_shape: (usize, usize),
    ) -> PyResult<Self> {
        if xmin >= xmax || ymin >= ymax {
            return Err(PyValueError::new_err(format!(
                "empty bounding box: rows {xmin}:{xmax}, columns {ymin}:{ymax}"
            )));
        }
        Ok(Self(BoundingBox::new(
            xmin,
            xmax,
            ymin,
            ymax,
            original_shape,
        )))
    }

    /// Minimum row pixel (numpy axis -2, FITS NAXIS2). Inclusive.
    #[getter]
    fn xmin(&self) -> usize {
        self.0.xmin
    }

    /// Maximum row pixel (numpy axis -2, FITS NAXIS2). Exclusive.
    #[getter]
    fn xmax(&self) -> usize {
        self.0.xmax
    }

    /// Minimum column pixel (numpy axis -1, FITS NAXIS1). Inclusive.
    #[getter]
    fn ymin(&self) -> usize {
        self.0.ymin
    }

    /// Maximum column pixel (numpy axis -1, FITS NAXIS1). Exclusive.
    #[getter]
    fn ymax(&self) -> usize {
        self.0.ymax
    }

    /// Shape (rows, columns) of the plane the box was built from.
    #[getter]
    fn original_shape(&self) -> (usize, usize) {
        self.0.original_shape
    }

    /// The span between ymax and ymin (i.e. the trimmed NAXIS1).
    #[getter]
    fn y_span(&self) -> usize {
        self.0.y_span
    }

    /// The span between xmax and xmin (i.e. the trimmed NAXIS2).
    #[getter]
    fn x_span(&self) -> usize {
        self.0.x_span
    }

    fn __repr__(&self) -> String {
        let b = &self.0;
        format!(
            "BoundingBox(xmin={}, xmax={}, ymin={}, ymax={}, original_shape={:?})",
            b.xmin, b.xmax, b.ymin, b.ymax, b.original_shape
        )
    }
}

/// The ``bounding_box`` argument of ``combine_fits``: a flag, or a box to use
/// as is.
#[derive(FromPyObject, IntoPyObject)]
enum BoundingBoxArg {
    Box(PyBoundingBox),
    Flag(bool),
}

#[cfg(feature = "stubgen")]
impl pyo3_stub_gen::PyStubType for BoundingBoxArg {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::builtin("bool") | PyBoundingBox::type_output()
    }
}

/// Compute the single bounding box that encompasses the valid data of every
/// image in ``file_list``.
///
/// This is the box ``combine_fits`` computes internally when
/// ``bounding_box=True``. Compute it once with this function and pass the
/// result to ``combine_fits(bounding_box=...)`` when several cubes (e.g. an
/// image cube and its weights cube) must land on an identical pixel grid.
///
/// Args:
///     file_list (list[str]): The FITS images to consider.
///     invalidate_zeros (bool): Treat exactly-zero pixels as invalid.
///
/// Returns:
///     BoundingBox: The smallest bounding box that contains all valid data.
///
/// Raises:
///     ValueError: If no image has valid data, or on a FITS error.
#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (file_list, invalidate_zeros=false))]
fn get_common_bounding_box(
    file_list: Vec<PathBuf>,
    invalidate_zeros: bool,
) -> PyResult<PyBoundingBox> {
    rust_common_bounding_box(&file_list, invalidate_zeros)
        .map(PyBoundingBox)
        .map_err(to_py_err)
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
///     bounding_box (bool | BoundingBox): Trim blank padding via a common
///         bounding box. A ``BoundingBox`` is used as is (see
///         ``get_common_bounding_box``) to force several cubes onto an
///         identical pixel grid.
///     invalidate_zeros (bool): Treat exactly-zero pixels as NaN.
///     float_length (int, optional): Output precision in bits (32 or 64).
///     blank_zero_beams (bool): Blank (NaN) any input image whose restoring
///         beam is exactly zero, e.g. a wsclean ``-fit-spectral-pol`` model
///         plane (default True).
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
    bounding_box=BoundingBoxArg::Flag(false),
    invalidate_zeros=false,
    float_length=None,
    blank_zero_beams=true,
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
    bounding_box: BoundingBoxArg,
    invalidate_zeros: bool,
    float_length: Option<u8>,
    blank_zero_beams: bool,
) -> PyResult<Vec<f64>> {
    let (bounding_box, supplied_bounding_box) = match bounding_box {
        BoundingBoxArg::Flag(flag) => (flag, None),
        BoundingBoxArg::Box(bb) => (true, Some(bb.0)),
    };
    let options = CombineOptions {
        spec_file,
        spec_list,
        ignore_spec,
        create_blanks,
        overwrite,
        max_workers,
        time_domain_mode,
        bounding_box,
        supplied_bounding_box,
        invalidate_zeros,
        float_length,
        // No stderr progress bars when driven from Python.
        progress: false,
        blank_zero_beams,
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
    m.add_class::<PyBoundingBox>()?;
    m.add_function(wrap_pyfunction!(combine_fits, m)?)?;
    m.add_function(wrap_pyfunction!(get_common_bounding_box, m)?)?;
    m.add_function(wrap_pyfunction!(extract_plane_from_cube, m)?)?;
    #[cfg(feature = "stubgen")]
    m.add_function(wrap_pyfunction!(_generate_stubs, m)?)?;
    Ok(())
}
