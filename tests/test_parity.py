"""Parity tests: run the original `fitscube` and this Rust port on identical
inputs and assert the output cubes agree (pixel data, the spectral/temporal WCS
cards, and the per-channel BEAMS table).

The original package is the reference implementation, so any disagreement is a
bug in `fitscube_rs`.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from packaging.version import Version

import fitscube_rs

# The reference implementation. Skip the whole module if it is not installed.
fitscube = pytest.importorskip("fitscube")
fitscube_extract = pytest.importorskip("fitscube.extract")
fitscube_compress = pytest.importorskip("fitscube.compress")


def _write_image(
    path: Path,
    fill: float,
    *,
    reffreq: float | None = None,
    date_obs: str | None = None,
    beam: float | None = None,
    # Above fitscube's 1801-element small-cube path, which corrupted data
    # before v2.3.0
    ny: int = 30,
    nx: int = 30,
) -> None:
    """Write a 2D image with a minimal celestial WCS and optional spectral/beam
    metadata."""
    data = np.full((ny, nx), fill, dtype=np.float32)
    header = fits.Header()
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CDELT1"] = -1.0 / 3600.0
    header["CDELT2"] = 1.0 / 3600.0
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    if reffreq is not None:
        header["REFFREQ"] = reffreq
    if date_obs is not None:
        header["DATE-OBS"] = date_obs
    if beam is not None:
        header["BMAJ"] = beam
        header["BMIN"] = beam
        header["BPA"] = 0.0
    fits.writeto(path, data, header, overwrite=True)


def _make_freq_images(d: Path, freqs, beams=None) -> list[Path]:
    d.mkdir(parents=True, exist_ok=True)
    files = []
    for i, freq in enumerate(freqs):
        p = d / f"img_{i}.fits"
        beam = None if beams is None else beams[i]
        _write_image(p, fill=float(i + 1), reffreq=freq, beam=beam)
        files.append(p)
    return files


# WCS cards that define the new cube axis; these must match exactly (numerically).
SPECTRAL_CARDS = ["NAXIS", "NAXIS3", "CTYPE3", "CRPIX3", "CRVAL3", "CDELT3", "CSYER3"]


def _compare_cubes(ref: Path, rs: Path) -> None:
    with fits.open(ref) as ref_hdul, fits.open(rs) as rs_hdul:
        ref_data = ref_hdul[0].data
        rs_data = rs_hdul[0].data
        assert ref_data.shape == rs_data.shape, (
            f"shape {rs_data.shape} != reference {ref_data.shape}"
        )
        np.testing.assert_allclose(
            np.nan_to_num(rs_data), np.nan_to_num(ref_data), rtol=1e-6, atol=1e-6
        )

        ref_h = ref_hdul[0].header
        rs_h = rs_hdul[0].header
        for card in SPECTRAL_CARDS:
            if card in ref_h:
                assert card in rs_h, f"{card} missing from fitscube_rs output"
                if isinstance(ref_h[card], str):
                    assert rs_h[card] == ref_h[card], card
                else:
                    assert rs_h[card] == pytest.approx(
                        ref_h[card], rel=1e-9, abs=1e-3
                    ), f"{card}: {rs_h[card]} != {ref_h[card]}"


def test_even_frequency_combine(tmp_path):
    freqs = [1.0e9, 2.0e9, 3.0e9, 4.0e9]
    ref_files = _make_freq_images(tmp_path / "ref", freqs)
    rs_files = _make_freq_images(tmp_path / "rs", freqs)

    ref_cube = tmp_path / "ref_cube.fits"
    rs_cube = tmp_path / "rs_cube.fits"

    fitscube.combine_fits(file_list=ref_files, out_cube=ref_cube, overwrite=True)
    fitscube_rs.combine_fits([str(f) for f in rs_files], str(rs_cube), overwrite=True)

    _compare_cubes(ref_cube, rs_cube)


def test_uneven_frequency_combine_with_blanks(tmp_path):
    # 3 GHz channel missing; create_blanks should interpolate a blank plane.
    freqs = [1.0e9, 2.0e9, 4.0e9]
    ref_files = _make_freq_images(tmp_path / "ref", freqs)
    rs_files = _make_freq_images(tmp_path / "rs", freqs)

    ref_cube = tmp_path / "ref_cube.fits"
    rs_cube = tmp_path / "rs_cube.fits"

    fitscube.combine_fits(
        file_list=ref_files, out_cube=ref_cube, overwrite=True, create_blanks=True
    )
    fitscube_rs.combine_fits(
        [str(f) for f in rs_files], str(rs_cube), overwrite=True, create_blanks=True
    )

    _compare_cubes(ref_cube, rs_cube)


@pytest.mark.parametrize("jitter", [0.0, 0.25])
def test_racs_all_uneven_bins(tmp_path, jitter):
    """Images of uneven width over RACS-low3, -mid and -high, with the gaps
    between the bands, as wsclean labels them. fitscube 2.10 refused these."""
    starts, counts = [0, 496, 712], [288, 144, 288]
    index = np.concatenate(
        [s + np.arange(c) for s, c in zip(starts, counts, strict=True)]
    )
    channels = 799.5e6 + 0.5e6 + index * 1e6
    rng = np.random.default_rng(0)
    bins, start = [], 0
    while start < len(channels):
        end = min(len(channels), start + int(rng.integers(15, 26)))
        bins.append((start, end - 1))
        start = end
    freqs = [(channels[a] + channels[b]) / 2 for a, b in bins]
    freqs += rng.uniform(-jitter, jitter, len(freqs))

    files = []
    for i, freq in enumerate(freqs):
        files.append(tmp_path / f"img_{i}.fits")
        _write_image(files[-1], fill=float(i + 1), reffreq=freq, ny=4, nx=4)
    rs_cube = tmp_path / "rs_cube.fits"
    fitscube_rs.combine_fits(
        [str(f) for f in files], str(rs_cube), overwrite=True, create_blanks=True
    )
    header = fits.getheader(rs_cube)
    assert header["CDELT3"] == pytest.approx(0.5e6, abs=1e-3)
    assert (header["CSYER3"] > 0.01) == (jitter > 0)

    if not hasattr(importlib.import_module("fitscube.combine_fits"), "fit_grid"):
        pytest.skip(f"fitscube {fitscube.__version__} refuses these grids")
    ref_cube = tmp_path / "ref_cube.fits"
    fitscube.combine_fits(
        file_list=files, out_cube=ref_cube, overwrite=True, create_blanks=True
    )
    _compare_cubes(ref_cube, rs_cube)


def test_varying_beams_beam_table(tmp_path):
    freqs = [1.0e9, 2.0e9, 3.0e9]
    beams = [10.0 / 3600.0, 11.0 / 3600.0, 12.0 / 3600.0]  # arcsec → deg, varying
    ref_files = _make_freq_images(tmp_path / "ref", freqs, beams=beams)
    rs_files = _make_freq_images(tmp_path / "rs", freqs, beams=beams)

    ref_cube = tmp_path / "ref_cube.fits"
    rs_cube = tmp_path / "rs_cube.fits"

    fitscube.combine_fits(file_list=ref_files, out_cube=ref_cube, overwrite=True)
    fitscube_rs.combine_fits([str(f) for f in rs_files], str(rs_cube), overwrite=True)

    _compare_cubes(ref_cube, rs_cube)

    # Both must carry a BEAMS table with matching BMAJ/BMIN/BPA.
    with fits.open(ref_cube) as ref_hdul, fits.open(rs_cube) as rs_hdul:
        ref_beams = ref_hdul["BEAMS"].data
        rs_beams = rs_hdul["BEAMS"].data
        for col in ("BMAJ", "BMIN", "BPA"):
            np.testing.assert_allclose(
                rs_beams[col], ref_beams[col], rtol=1e-4, atol=1e-4
            )
        rs_units = [c.unit for c in rs_hdul["BEAMS"].columns]
        assert rs_units == [c.unit for c in ref_hdul["BEAMS"].columns]
        assert rs_units == ["arcsec", "arcsec", "deg", None, None]


def _make_stokes_images(d: Path, freqs, beams, stokes_code: float) -> list[Path]:
    """wsclean-order (RA, DEC, FREQ, STOKES) single-plane images."""
    d.mkdir(parents=True, exist_ok=True)
    files = []
    for i, (freq, beam) in enumerate(zip(freqs, beams, strict=True)):
        p = d / f"img_{i}.fits"
        _write_image(p, fill=float(i + 1), beam=beam)
        with fits.open(p) as hdul:
            header = hdul[0].header
            data = hdul[0].data[np.newaxis, np.newaxis]
        header["CTYPE3"], header["CRVAL3"] = "FREQ", freq
        header["CDELT3"], header["CRPIX3"], header["CUNIT3"] = 1.0e6, 1.0, "Hz"
        header["CTYPE4"], header["CRVAL4"] = "STOKES", stokes_code
        header["CDELT4"], header["CRPIX4"] = 1.0, 1.0
        fits.writeto(p, data, header, overwrite=True)
        files.append(p)
    return files


@pytest.mark.parametrize("stokes_code", [1.0, 3.0, -5.0])  # I, U, XX
def test_single_stokes_beam_table_pol(tmp_path, stokes_code):
    """BEAMS CHAN/POL are 0-based axis indices, not FITS Stokes codes."""
    freqs = [1.0e9, 2.0e9, 3.0e9]
    beams = [10.0 / 3600.0, 11.0 / 3600.0, 12.0 / 3600.0]
    rs_files = _make_stokes_images(tmp_path / "rs", freqs, beams, stokes_code)
    rs_cube = tmp_path / "rs_cube.fits"
    fitscube_rs.combine_fits([str(f) for f in rs_files], str(rs_cube), overwrite=True)

    with fits.open(rs_cube) as hdul:
        assert hdul[0].header["CRVAL4"] == stokes_code
        assert hdul["BEAMS"].header["NPOL"] == 1
        assert hdul["BEAMS"].data["POL"].tolist() == [0, 0, 0]
        assert hdul["BEAMS"].data["CHAN"].tolist() == [0, 1, 2]

    ref_files = _make_stokes_images(tmp_path / "ref", freqs, beams, stokes_code)
    ref_cube = tmp_path / "ref_cube.fits"
    fitscube.combine_fits(file_list=ref_files, out_cube=ref_cube, overwrite=True)
    with fits.open(ref_cube) as ref_hdul, fits.open(rs_cube) as rs_hdul:
        np.testing.assert_allclose(
            np.nan_to_num(rs_hdul[0].data), np.nan_to_num(ref_hdul[0].data)
        )
        for key in ("NCHAN", "NPOL"):
            assert rs_hdul["BEAMS"].header[key] == ref_hdul["BEAMS"].header[key]
        for col in ("CHAN", "POL"):
            assert (
                rs_hdul["BEAMS"].data[col].tolist()
                == ref_hdul["BEAMS"].data[col].tolist()
            )


def test_time_domain_combine(tmp_path):
    # Evenly spaced 10s steps; combine along the TIME axis (DATE-OBS).
    dates = [
        "2020-01-01T00:00:00.000",
        "2020-01-01T00:00:10.000",
        "2020-01-01T00:00:20.000",
        "2020-01-01T00:00:30.000",
    ]
    ref_dir = tmp_path / "ref"
    rs_dir = tmp_path / "rs"
    ref_dir.mkdir()
    rs_dir.mkdir()
    ref_files, rs_files = [], []
    for i, date in enumerate(dates):
        rp = ref_dir / f"t_{i}.fits"
        sp = rs_dir / f"t_{i}.fits"
        _write_image(rp, fill=float(i + 1), date_obs=date)
        _write_image(sp, fill=float(i + 1), date_obs=date)
        ref_files.append(rp)
        rs_files.append(sp)

    ref_cube = tmp_path / "ref_cube.fits"
    rs_cube = tmp_path / "rs_cube.fits"

    fitscube.combine_fits(
        file_list=ref_files, out_cube=ref_cube, overwrite=True, time_domain_mode=True
    )
    fitscube_rs.combine_fits(
        [str(f) for f in rs_files], str(rs_cube), overwrite=True, time_domain_mode=True
    )

    _compare_cubes(ref_cube, rs_cube)
    with fits.open(rs_cube) as hdul:
        assert hdul[0].header["CTYPE3"] == "TIME"
        assert hdul[0].header["CUNIT3"] == "s"


def _make_bordered_images(d: Path, freqs) -> list[Path]:
    """Images with a NaN border; valid block rows 5..24 (20), cols 8..21 (14)."""
    d.mkdir(parents=True, exist_ok=True)
    files = []
    for i, freq in enumerate(freqs):
        data = np.full((30, 30), np.nan, dtype=np.float32)
        data[5:25, 8:22] = float(i + 1)
        h = fits.Header()
        h["CTYPE1"] = "RA---SIN"
        h["CTYPE2"] = "DEC--SIN"
        h["CRPIX1"] = 1.0
        h["CRPIX2"] = 1.0
        h["CRVAL1"] = 0.0
        h["CRVAL2"] = 0.0
        h["CDELT1"] = -1.0 / 3600.0
        h["CDELT2"] = 1.0 / 3600.0
        h["REFFREQ"] = freq
        p = d / f"b_{i}.fits"
        fits.writeto(p, data, h, overwrite=True)
        files.append(p)
    return files


# The lossless bounding box (keeping the last valid row/col) landed after the
# v2.3.0 tag (fitscube PR #51). Released v2.3.0 has an off-by-one that drops a
# row and column of real data. fitscube_rs implements the correct, lossless
# behaviour, so the cross-check against fitscube only holds from that fix on.
_BBOX_FIX_VERSION = Version("2.3.1")
_fitscube_has_bbox_fix = Version(
    fitscube.__version__.split("+")[0].split(".dev")[0]
) >= (_BBOX_FIX_VERSION)


def test_bounding_box_trim(tmp_path):
    freqs = [1.0e9, 2.0e9, 3.0e9]
    rs_files = _make_bordered_images(tmp_path / "rs", freqs)
    rs_cube = tmp_path / "rs_cube.fits"
    fitscube_rs.combine_fits(
        [str(f) for f in rs_files], str(rs_cube), overwrite=True, bounding_box=True
    )

    # Correctness (always): fitscube_rs trims losslessly to the full valid block.
    with fits.open(rs_cube) as hdul:
        assert hdul[0].data.shape == (3, 20, 14)
        assert hdul[0].header["CRPIX1"] == pytest.approx(1.0 - 8.0)  # -= ymin
        assert hdul[0].header["CRPIX2"] == pytest.approx(1.0 - 5.0)  # -= xmin
        for c in range(3):
            np.testing.assert_allclose(hdul[0].data[c], float(c + 1))

    # Parity (only when the reference carries the lossless fix, i.e. > v2.3.0).
    if not _fitscube_has_bbox_fix:
        pytest.skip(
            f"fitscube {fitscube.__version__} predates the lossless bounding-box "
            f"fix (>= {_BBOX_FIX_VERSION}); skipping cross-check"
        )
    ref_files = _make_bordered_images(tmp_path / "ref", freqs)
    ref_cube = tmp_path / "ref_cube.fits"
    fitscube.combine_fits(
        file_list=ref_files, out_cube=ref_cube, overwrite=True, bounding_box=True
    )
    with fits.open(ref_cube) as ref_hdul, fits.open(rs_cube) as rs_hdul:
        assert rs_hdul[0].data.shape == ref_hdul[0].data.shape
        for card in ("NAXIS1", "NAXIS2", "CRPIX1", "CRPIX2"):
            assert rs_hdul[0].header[card] == pytest.approx(ref_hdul[0].header[card]), (
                card
            )
        np.testing.assert_allclose(
            np.nan_to_num(rs_hdul[0].data), np.nan_to_num(ref_hdul[0].data), rtol=1e-6
        )


def test_extract_matches_input_plane(tmp_path):
    freqs = [1.0e9, 2.0e9, 3.0e9, 4.0e9]
    rs_files = _make_freq_images(tmp_path / "rs", freqs)
    cube = tmp_path / "cube.fits"
    fitscube_rs.combine_fits([str(f) for f in rs_files], str(cube), overwrite=True)

    out = fitscube_rs.extract_plane_from_cube(
        str(cube), channel_index=2, overwrite=True
    )
    with fits.open(out) as hdul:
        data = np.squeeze(hdul[0].data)
        # Channel 2 was filled with value 3.0 (fill = i + 1).
        np.testing.assert_allclose(data, 3.0, rtol=1e-6)


def _write_plane(
    path: Path,
    freq: float,
    fill: float,
    *,
    beam: float | None = None,
    dtype=np.float32,
) -> Path:
    """A (FREQ, STOKES, DEC, RA) single-channel image, the usual ASKAP layout."""
    header = fits.Header()
    header["CTYPE1"], header["CRPIX1"], header["CRVAL1"] = "RA---SIN", 1.0, 0.0
    header["CDELT1"], header["CUNIT1"] = -1.0 / 3600.0, "deg"
    header["CTYPE2"], header["CRPIX2"], header["CRVAL2"] = "DEC--SIN", 1.0, 0.0
    header["CDELT2"], header["CUNIT2"] = 1.0 / 3600.0, "deg"
    header["CTYPE3"], header["CRPIX3"], header["CRVAL3"] = "STOKES", 1.0, 1.0
    header["CDELT3"] = 1.0
    header["CTYPE4"], header["CRPIX4"], header["CRVAL4"] = "FREQ", 1.0, freq
    header["CDELT4"], header["CUNIT4"] = 1.0e6, "Hz"
    if beam is not None:
        header["BMAJ"], header["BMIN"], header["BPA"] = beam, beam / 2, 0.0
    data = np.full((1, 1, 30, 30), fill, dtype=dtype)
    fits.PrimaryHDU(data, header=header).writeto(path, overwrite=True)
    return path


def _combine_both(tmp_path: Path, make, **kwargs) -> tuple[Path, Path]:
    """Combine the same inputs with fitscube and fitscube_rs."""
    cubes = []
    for name, combine in (
        ("ref", fitscube.combine_fits),
        ("rs", fitscube_rs.combine_fits),
    ):
        d = tmp_path / name
        d.mkdir()
        cube = tmp_path / f"{name}_cube.fits"
        # The reference takes Paths; fitscube_rs takes any path-like
        combine(
            file_list=[Path(f) for f in make(d)],
            out_cube=cube,
            overwrite=True,
            **kwargs,
        )
        cubes.append(cube)
    return cubes[0], cubes[1]


def _compare_full(ref: Path, rs: Path) -> None:
    """Data (NaNs included), dtype, and every BEAMS column and count."""
    with fits.open(ref) as ref_hdul, fits.open(rs) as rs_hdul:
        ref_data, rs_data = ref_hdul[0].data, rs_hdul[0].data
        assert rs_data.shape == ref_data.shape
        assert rs_data.dtype == ref_data.dtype
        np.testing.assert_array_equal(np.isnan(rs_data), np.isnan(ref_data))
        np.testing.assert_allclose(
            np.nan_to_num(rs_data), np.nan_to_num(ref_data), rtol=1e-6
        )
        ref_names = [hdu.name for hdu in ref_hdul]
        assert [hdu.name for hdu in rs_hdul] == ref_names
        if "BEAMS" not in ref_names:
            return
        ref_beams, rs_beams = ref_hdul["BEAMS"], rs_hdul["BEAMS"]
        for key in ("NCHAN", "NPOL", "TUNIT1", "TUNIT2", "TUNIT3"):
            assert rs_beams.header[key] == ref_beams.header[key], key
        assert len(rs_beams.columns) == len(ref_beams.columns)
        for rs_col, ref_col in zip(rs_beams.columns, ref_beams.columns, strict=True):
            assert (rs_col.name, rs_col.unit) == (ref_col.name, ref_col.unit)
        for col in ("BMAJ", "BMIN", "BPA"):
            np.testing.assert_allclose(
                rs_beams.data[col], ref_beams.data[col], rtol=1e-6, err_msg=col
            )
        for col in ("CHAN", "POL"):
            assert rs_beams.data[col].tolist() == ref_beams.data[col].tolist(), col


def test_unsorted_beams_follow_channels_and_blanks(tmp_path):
    """Beams are sorted with the planes and padded over blank channels."""
    freqs = [1.004e9, 1.0e9, 1.003e9, 1.001e9]  # unsorted, 1.002 GHz missing

    def make(d: Path) -> list[str]:
        return [
            str(_write_plane(d / f"p{i}.fits", f, float(i), beam=(f - 0.99e9) / 1e10))
            for i, f in enumerate(freqs)
        ]

    ref, rs = _combine_both(tmp_path, make, create_blanks=True)
    _compare_full(ref, rs)


def test_beam_not_in_first_file(tmp_path):
    def make(d: Path) -> list[str]:
        return [
            str(
                _write_plane(
                    d / f"p{i}.fits", 1e9 + i * 1e6, float(i), beam=1e-3 if i else None
                )
            )
            for i in range(4)
        ]

    ref, rs = _combine_both(tmp_path, make)
    _compare_full(ref, rs)


@pytest.mark.parametrize("blank_zero_beams", [True, False])
def test_zero_beams(tmp_path, blank_zero_beams):
    """wsclean -fit-spectral-pol model planes carry an exactly-zero beam."""

    def make(d: Path) -> list[str]:
        return [
            str(
                _write_plane(
                    d / f"p{i}.fits",
                    1e9 + i * 1e6,
                    float(i + 1),
                    beam=0.0 if i == 1 else 1e-3 * (i + 1),
                )
            )
            for i in range(4)
        ]

    ref, rs = _combine_both(tmp_path, make, blank_zero_beams=blank_zero_beams)
    _compare_full(ref, rs)


def test_integer_input(tmp_path):
    def make(d: Path) -> list[str]:
        return [
            str(_write_plane(d / f"p{i}.fits", 1e9 + i * 1e6, i, dtype=np.int16))
            for i in range(4)
        ]

    ref, rs = _combine_both(tmp_path, make)
    assert fits.getheader(rs)["BITPIX"] == 16
    _compare_full(ref, rs)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        # Irregular frequencies cannot be gridded without dropping inputs
        ({"create_blanks": True}, "would drop inputs"),
    ],
)
def test_irregular_spacing_raises(tmp_path, kwargs, error):
    freqs = 1e9 + np.array([0.0, 2.302585, 2.718281, 2.828427, 3.141592]) * 1e7

    def make(d: Path) -> list[str]:
        return [
            str(_write_plane(d / f"p{i}.fits", f, float(i)))
            for i, f in enumerate(freqs)
        ]

    for combine in (fitscube.combine_fits, fitscube_rs.combine_fits):
        d = tmp_path / combine.__module__
        d.mkdir()
        with pytest.raises(Exception, match=error):
            combine(
                file_list=[Path(f) for f in make(d)],
                out_cube=tmp_path / "c.fits",
                **kwargs,
            )


def test_supplied_bounding_box(tmp_path):
    """A common box forces an image cube and its weights onto one grid."""

    def make_images(d: Path) -> list[str]:
        files = []
        for i in range(4):
            path = _write_plane(d / f"image_{i}.fits", 1e9 + i * 1e6, float(i))
            with fits.open(path, mode="update") as hdul:
                hdul[0].data[..., : i + 3, :] = np.nan
                hdul[0].data[..., :, 25:] = np.nan
            files.append(str(path))
        return files

    (tmp_path / "images").mkdir()
    images = make_images(tmp_path / "images")
    ref_box = fitscube.get_common_bounding_box(file_list=[Path(f) for f in images])
    rs_box = fitscube_rs.get_common_bounding_box(images)
    for attr in ("xmin", "xmax", "ymin", "ymax", "x_span", "y_span"):
        assert getattr(rs_box, attr) == getattr(ref_box, attr), attr
    assert rs_box.original_shape == tuple(ref_box.original_shape)
    assert rs_box == fitscube_rs.BoundingBox(
        rs_box.xmin, rs_box.xmax, rs_box.ymin, rs_box.ymax, rs_box.original_shape
    )

    def make_weights(d: Path) -> list[str]:
        return [
            str(_write_plane(d / f"weight_{i}.fits", 1e9 + i * 1e6, 1.0))
            for i in range(4)
        ]

    ref_cube = tmp_path / "ref_weights.fits"
    rs_cube = tmp_path / "rs_weights.fits"
    (tmp_path / "ref").mkdir()
    (tmp_path / "rs").mkdir()
    fitscube.combine_fits(
        file_list=[Path(f) for f in make_weights(tmp_path / "ref")],
        out_cube=ref_cube,
        bounding_box=ref_box,
    )
    fitscube_rs.combine_fits(
        make_weights(tmp_path / "rs"), rs_cube, bounding_box=rs_box
    )
    _compare_full(ref_cube, rs_cube)
    for key in ("NAXIS1", "NAXIS2", "CRPIX1", "CRPIX2"):
        assert fits.getheader(rs_cube)[key] == fits.getheader(ref_cube)[key], key
    # The weights alone would not have been trimmed
    assert (
        fitscube_rs.get_common_bounding_box(make_weights(tmp_path / "rs")).x_span == 30
    )


def test_extract_planes_from_cube(tmp_path):
    """Every plane, beam included, matches fitscube's extraction."""
    freqs = [1.0e9, 2.0e9, 3.0e9]
    beams = [10.0 / 3600.0, 11.0 / 3600.0, 12.0 / 3600.0]
    files = _make_freq_images(tmp_path / "in", freqs, beams=beams)
    cube = tmp_path / "cube.fits"
    fitscube_rs.combine_fits(files, cube)

    ref_paths = fitscube_extract.extract_planes_from_cube(
        cube,
        fitscube_extract.ExtractPlanesOptions(
            output_paths=[tmp_path / f"ref{i}.fits" for i in range(3)]
        ),
    )
    rs_paths = fitscube_rs.extract_planes_from_cube(
        cube, output_paths=[tmp_path / f"rs{i}.fits" for i in range(3)]
    )
    for ref, rs in zip(ref_paths, rs_paths, strict=True):
        with fits.open(ref) as ref_hdul, fits.open(rs) as rs_hdul:
            np.testing.assert_array_equal(rs_hdul[0].data, ref_hdul[0].data)
            ref_h, rs_h = ref_hdul[0].header, rs_hdul[0].header
            for key in ("NAXIS3", "CRPIX3", "CRVAL3", "CDELT3", "BMAJ", "BMIN", "BPA"):
                assert rs_h[key] == pytest.approx(ref_h[key], rel=1e-6), key
            assert "CASAMBM" not in rs_h


def _write_noise_cube(path: Path, shape=(2, 5, 40, 48)) -> np.ndarray:
    """Float noise with a blanked strip, an all-blank plane, a constant plane and beams."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1e-3, shape).astype(np.float32)
    data[..., :8, :] = np.nan
    data[0, 1] = np.nan
    data[1, 2] = 0.5
    header = fits.Header()
    header["BUNIT"] = "Jy/beam"
    header["CTYPE3"] = "FREQ"
    beams = fits.BinTableHDU.from_columns(
        [fits.Column(name="BMAJ", format="E", array=np.arange(shape[-3], dtype=float))],
        name="BEAMS",
    )
    fits.HDUList([fits.PrimaryHDU(data, header), beams]).writeto(path)
    return data


def _compress_both(tmp_path: Path, method: str) -> tuple[Path, Path, np.ndarray]:
    """The same cube compressed by fitscube and fitscube_rs."""
    outputs = []
    for name, compress in (
        ("ref", fitscube_compress.compress_cube),
        ("rs", fitscube_rs.compress_cube),
    ):
        cube = tmp_path / name / "cube.fits"
        cube.parent.mkdir()
        data = _write_noise_cube(cube)
        outputs.append(compress(cube, method=method))
    return outputs[0], outputs[1], data


def test_tile_gzip_decodes_exactly(tmp_path):
    """astropy reads the Rust GZIP_2 cube back exactly, beams included."""
    ref, rs, data = _compress_both(tmp_path, "tile-gzip")
    with fits.open(rs) as rs_hdul, fits.open(ref) as ref_hdul:
        np.testing.assert_array_equal(rs_hdul[1].data, data)
        assert rs_hdul[1].header["BUNIT"] == ref_hdul[1].header["BUNIT"]
        np.testing.assert_array_equal(
            rs_hdul["BEAMS"].data["BMAJ"], ref_hdul["BEAMS"].data["BMAJ"]
        )


def test_tile_rice_matches_fitscube(tmp_path):
    """Same blanks, same dither seed and an error as small as fitscube's.

    Values differ slightly: astropy fills blanks before estimating the noise,
    while CFITSIO skips them.
    """
    ref, rs, data = _compress_both(tmp_path, "tile-rice")
    ref_data, rs_data = fits.getdata(ref, 1), fits.getdata(rs, 1)
    np.testing.assert_array_equal(np.isnan(rs_data), np.isnan(ref_data))
    np.testing.assert_array_equal(rs_data[1, 2], data[1, 2])
    finite = np.isfinite(data)
    bound = 1e-3 / 4 / np.sqrt(12) * 1.2
    assert np.std(rs_data[finite] - data[finite]) < bound
    assert np.std(ref_data[finite] - data[finite]) < bound
    raw = {"disable_image_compression": True}
    assert (
        fits.getheader(rs, 1, **raw)["ZDITHER0"]
        == fits.getheader(ref, 1, **raw)["ZDITHER0"]
    )


@pytest.mark.parametrize("method", ["gzip", "tile-gzip", "tile-rice"])
def test_decompress_matches_fitscube(tmp_path, method):
    """Each decompresses the other's cubes to the same data and marks."""
    ref, rs, _ = _compress_both(tmp_path, method)
    out = {}
    for name, compressed in (("ref", ref), ("rs", rs)):
        out[name] = (
            fitscube_compress.decompress_cube(
                compressed, output=tmp_path / f"{name}_by_ref.fits"
            ),
            fitscube_rs.decompress_cube(
                compressed, output=tmp_path / f"{name}_by_rs.fits"
            ),
        )
    for by_ref, by_rs in out.values():
        np.testing.assert_array_equal(fits.getdata(by_rs), fits.getdata(by_ref))
        ref_h, rs_h = fits.getheader(by_ref), fits.getheader(by_rs)
        for key in ("BUNIT", "CTYPE3", "NAXIS4", "FCLOSSY", "FCQLEVEL"):
            assert rs_h.get(key) == ref_h.get(key), key
        np.testing.assert_array_equal(
            fits.getdata(by_rs, "BEAMS")["BMAJ"], fits.getdata(by_ref, "BEAMS")["BMAJ"]
        )


def test_pgzip_reads_as_gzip(tmp_path):
    """Rust pgzip writes gzip members that fitscube's plain gzip reader joins."""
    cube = tmp_path / "cube.fits"
    data = _write_noise_cube(cube)
    original = cube.read_bytes()
    compressed = fitscube_rs.compress_cube(cube, method="pgzip", max_workers=4)
    restored = fitscube_compress.decompress_cube(compressed)
    assert restored.read_bytes() == original
    np.testing.assert_array_equal(fits.getdata(restored), data)


def test_cd_matrix_carries_the_step(tmp_path):
    """A CD matrix input keeps the spectral step in CD3_3, as fitscube does."""

    def make(d: Path) -> list[Path]:
        files = _make_freq_images(d, [1.0e9, 1.1e9, 1.2e9])
        for f in files:
            with fits.open(f, mode="update") as hdul:
                h = hdul[0].header
                h["CD1_1"], h["CD2_2"] = h.pop("CDELT1"), h.pop("CDELT2")
        return files

    ref, rs = _combine_both(tmp_path, make)
    ref_h, rs_h = fits.getheader(ref), fits.getheader(rs)
    assert rs_h["CD3_3"] == pytest.approx(ref_h["CD3_3"])
    assert rs_h["CD3_3"] == pytest.approx(1e8)


def test_time_axis_reference_is_set(tmp_path):
    """Time cubes are in UTC MJD seconds from MJD 0, whatever the inputs said."""

    def make(d: Path) -> list[Path]:
        d.mkdir(parents=True, exist_ok=True)
        files = []
        for i in range(3):
            p = d / f"img_{i}.fits"
            _write_image(p, fill=float(i), date_obs=f"2024-01-01T00:0{i}:00")
            with fits.open(p, mode="update") as hdul:
                hdul[0].header["MJDREFI"] = 50000
                hdul[0].header["TIMESYS"] = "TAI"
            files.append(p)
        return files

    ref, rs = _combine_both(tmp_path, make, time_domain_mode=True)
    ref_h, rs_h = fits.getheader(ref), fits.getheader(rs)
    for key in ("MJDREF", "TIMESYS"):
        assert rs_h[key] == ref_h[key], key
    assert "MJDREFI" not in rs_h
    assert "MJDREFI" not in ref_h
