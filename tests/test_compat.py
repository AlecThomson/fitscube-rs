"""``fitscube_rs.compat`` against the fitscube package it stands in for."""

from __future__ import annotations

import inspect
import pickle
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from test_parity import _make_freq_images, _write_noise_cube

import fitscube_rs
from fitscube_rs import compat

fitscube = pytest.importorskip("fitscube")
fc_bbox = pytest.importorskip("fitscube.bounding_box")
fc_combine = pytest.importorskip("fitscube.combine_fits")
fc_compress = pytest.importorskip("fitscube.compress")
fc_extract = pytest.importorskip("fitscube.extract")
fc_exceptions = pytest.importorskip("fitscube.exceptions")

SHARED = [
    (fc_combine, "combine_fits"),
    (fc_combine, "even_spacing"),
    (fc_compress, "compress_cube"),
    (fc_extract, "extract_plane_from_cube"),
    (fc_extract, "extract_planes_from_cube"),
    (fc_extract, "find_target_axis"),
    (fc_extract, "count_cube_planes"),
    (fc_bbox, "extract_common_bounding_box"),
    (fc_bbox, "get_common_bounding_box"),
]


@pytest.mark.parametrize(("module", "name"), SHARED)
def test_same_parameters(module, name):
    """Each function takes fitscube's parameters, in order, with its defaults."""
    want = inspect.signature(getattr(module, name)).parameters
    if not want or "args" in want:
        pytest.skip(f"fitscube's {name} hides its signature")
    got = inspect.signature(getattr(compat, name)).parameters
    assert list(got) == list(want)
    for key, p in want.items():
        assert got[key].default == p.default, key


@pytest.mark.parametrize("name", compat.__all__)
def test_functions_are_python_functions(name):
    """Prefect and other tools that wrap functions need plain Python ones."""
    obj = getattr(compat, name)
    if callable(obj) and not isinstance(obj, type):
        assert inspect.isfunction(obj), name


@pytest.mark.parametrize(
    "freqs",
    [
        [1.0e9, 1.1e9, 1.2e9],
        [1.0e9, 1.1e9, 1.4e9, 1.5e9],
        [1.5e9, 1.0e9, 1.2e9],
        [1.0e9],
    ],
)
def test_even_spacing(freqs):
    ref = fc_combine.even_spacing(np.array(freqs) * u.Hz)
    rs = compat.even_spacing(np.array(freqs) * u.Hz)
    assert rs.specs.unit == ref.specs.unit
    np.testing.assert_array_equal(rs.specs.value, ref.specs.value)
    np.testing.assert_array_equal(rs.missing_chan_idx, ref.missing_chan_idx)
    assert rs.missing_chan_idx.dtype == bool


def test_even_spacing_keeps_unit():
    rs = compat.even_spacing([1.0, 1.1, 1.3] * u.GHz)
    assert rs.specs.unit == u.GHz
    assert len(rs.specs) == 4


def _askap_times(steps: np.ndarray) -> np.ndarray:
    """Start times of ASKAP integrations, in MJD seconds to 0.1 s as wsclean
    writes DATE-OBS."""
    return np.round(5.2597e9 + 9.95328 * np.asarray(steps), 1)


_RUN = np.arange(1564)
_OFF_CADENCE = _askap_times(_RUN)
_OFF_CADENCE[700] += 1.0

# Times that grid, and to how many planes
GRIDDED_TIMES = {
    "rounded": (_askap_times(_RUN), 1564),
    "rounded, 13 missing": (
        _askap_times(np.delete(_RUN, [100, 500, 501, *range(800, 810)])),
        1564,
    ),
    "long hole": (_askap_times(np.r_[0:50, 150:200]), 200),
}
REFUSED_TIMES = {
    "1 s off the cadence": _OFF_CADENCE,
    "gaps of 2 and 3 steps": 5.18e9 + np.array([0.0, 20.0, 50.0]),
    "span beyond float64": np.array([-1e308, 1e308]),
}

# fitscube#78 (2.12.2) added number_gaps when it kept time grids at the cadence
needs_time_grid_fix = pytest.mark.skipif(
    not hasattr(fc_combine, "number_gaps"),
    reason=f"fitscube {fitscube.__version__} puts imprecise times on a ~1 s grid",
)


@needs_time_grid_fix
@pytest.mark.parametrize(
    ("times", "planes"), list(GRIDDED_TIMES.values()), ids=list(GRIDDED_TIMES)
)
def test_even_spacing_times(times: np.ndarray, planes: int) -> None:
    """Times rounded to 0.1 s keep their 9.95 s cadence (fitscube#77)."""
    ref = fc_combine.even_spacing(times * u.s, time_domain_mode=True)
    rs = compat.even_spacing(times * u.s, time_domain_mode=True)
    assert len(ref.specs) == len(rs.specs) == planes
    np.testing.assert_array_equal(rs.missing_chan_idx, ref.missing_chan_idx)
    np.testing.assert_allclose(rs.specs.value, ref.specs.value, rtol=1e-15, atol=0)


@needs_time_grid_fix
@pytest.mark.parametrize("times", list(REFUSED_TIMES.values()), ids=list(REFUSED_TIMES))
def test_even_spacing_refused_times(times: np.ndarray) -> None:
    """Times off a single cadence are refused, not put on a finer grid."""
    with (
        np.errstate(over="ignore"),
        pytest.raises(fc_exceptions.IrregularSpacingException),
    ):
        fc_combine.even_spacing(times * u.s, time_domain_mode=True)
    with pytest.raises(compat.IrregularSpacingException):
        compat.even_spacing(times * u.s, time_domain_mode=True)


def test_even_spacing_irregular_raises():
    specs = np.array([1.0e9, 1.0e9 + np.pi, 1.0e9 + 1e7]) * u.Hz
    with pytest.raises(fc_exceptions.IrregularSpacingException):
        fc_combine.even_spacing(specs)
    with pytest.raises(compat.IrregularSpacingException):
        compat.even_spacing(specs)


def _header(**cards) -> fits.Header:
    h = fits.Header()
    h["NAXIS"] = 3
    h["NAXIS1"], h["NAXIS2"], h["NAXIS3"] = 4, 4, 7
    h["CTYPE1"], h["CTYPE2"] = "RA---SIN", "DEC--SIN"
    for k, v in cards.items():
        h[k] = v
    return h


@pytest.mark.parametrize(
    "cards",
    [
        {"CTYPE3": "FREQ", "CRPIX3": 1.0, "CRVAL3": 1e9, "CDELT3": 1e6},
        {"CTYPE3": "FREQ", "CDELT3": 2e6, "PC3_3": 0.5, "CUNIT3": "Hz"},
        {"CTYPE3": "FREQ", "CDELT3": 2e6, "CD3_3": 3e6},
        {"CTYPE3": "FREQ"},
    ],
)
def test_find_target_axis(cards):
    h = _header(**cards)
    assert (
        compat.find_target_axis(h).__dict__ == fc_extract.find_target_axis(h).__dict__
    )
    assert compat.count_cube_planes(h) == fc_extract.count_cube_planes(h) == 7


def test_find_target_axis_by_name_and_index():
    h = _header(CTYPE3="TIME", CDELT3=10.0)
    assert compat.find_target_axis(h, "TIME").axis == 3
    index = compat.TargetIndex(axis_name="TIME", axis_index=0, output_name="timestep")
    assert compat.find_target_axis(h, index).cdelt == 10.0
    with pytest.raises(compat.TargetAxisMissingException):
        compat.find_target_axis(h)
    with pytest.raises(fc_exceptions.TargetAxisMissingException):
        fc_extract.find_target_axis(h)


BOXES = [(3, 20, 5, 40), None, (1, 30, 8, 45)]


def _fields(box) -> tuple:
    return (box.xmin, box.xmax, box.ymin, box.ymax, tuple(box.original_shape))


def test_extract_common_bounding_box():
    shape = (50, 60)
    ref = fc_bbox.extract_common_bounding_box(
        [b and fc_bbox.BoundingBox(*b, shape, b[3] - b[2], b[1] - b[0]) for b in BOXES]
    )
    rs = compat.extract_common_bounding_box(
        [b and compat.BoundingBox(*b, shape) for b in BOXES]
    )
    assert _fields(rs) == _fields(ref)
    assert (rs.x_span, rs.y_span) == (ref.x_span, ref.y_span)


def test_extract_common_bounding_box_errors():
    a = compat.BoundingBox(0, 5, 0, 5, (10, 10))
    b = compat.BoundingBox(0, 5, 0, 5, (10, 12))
    with pytest.raises(compat.ShapeMismatchException):
        compat.extract_common_bounding_box([a, b])
    with pytest.raises(ValueError):
        compat.extract_common_bounding_box([None, None])


def test_bounding_box_pickles():
    box = fitscube_rs.BoundingBox(2, 9, 3, 11, (20, 30))
    again = pickle.loads(pickle.dumps(box))
    assert again == box
    assert again.y_span == 8


def _bordered(d: Path) -> list[Path]:
    files = _make_freq_images(d, [1.0e9, 1.1e9, 1.2e9])
    for f in files:
        with fits.open(f, mode="update") as hdul:
            hdul[0].data[:4] = np.nan
            hdul[0].data[:, -3:] = np.nan
    return files


def test_get_common_bounding_box(tmp_path):
    files = _bordered(tmp_path)
    ref = fc_bbox.get_common_bounding_box(files)
    rs = compat.get_common_bounding_box(files, max_workers=1)
    assert _fields(rs) == _fields(ref)


@pytest.mark.parametrize("create_blanks", [False, True])
def test_combine_fits_returns_quantity(tmp_path, create_blanks):
    freqs = [1.0e9, 1.1e9, 1.3e9]
    specs = []
    for name, module in (("ref", fc_combine), ("rs", compat)):
        files = _make_freq_images(tmp_path / name, freqs)
        specs.append(
            module.combine_fits(
                file_list=files,
                out_cube=tmp_path / f"{name}.fits",
                create_blanks=create_blanks,
            )
        )
    ref, rs = specs
    assert isinstance(rs, u.Quantity)
    assert rs.unit == ref.unit
    np.testing.assert_allclose(rs.value, ref.value)


def test_extract_planes(tmp_path):
    files = _make_freq_images(tmp_path / "in", [1.0e9, 1.1e9])
    cube = tmp_path / "cube.fits"
    compat.combine_fits(files, cube)
    outputs = [tmp_path / "a.fits", tmp_path / "b.fits"]
    paths = compat.extract_planes_from_cube(
        cube, compat.ExtractPlanesOptions(output_paths=outputs, overwrite=True)
    )
    assert paths == outputs
    path = compat.extract_plane_from_cube(
        cube, compat.ExtractOptions(channel_index=1, output_path=tmp_path / "c.fits")
    )
    assert path == tmp_path / "c.fits"
    np.testing.assert_array_equal(fits.getdata(path), fits.getdata(outputs[1]))


def test_compress_cube_out_cube(tmp_path):
    cube = tmp_path / "noise.fits"
    _write_noise_cube(cube)
    out = compat.compress_cube(out_cube=cube, method="tile-gzip", remove=False)
    assert isinstance(out, Path)
    assert out.exists()


def test_exceptions_match_fitscube_names():
    for name in dir(fc_exceptions):
        if name.endswith("Exception") and not name.startswith("_"):
            assert issubclass(getattr(compat, name), compat.FITSCubeException), name
