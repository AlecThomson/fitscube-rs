"""Parity tests: run the original `fitscube` and this Rust port on identical
inputs and assert the output cubes agree (pixel data, the spectral/temporal WCS
cards, and the per-channel BEAMS table).

The original package is the reference implementation, so any disagreement is a
bug in `fitscube_rs`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from packaging.version import Version

import fitscube_rs

# The reference implementation. Skip the whole module if it is not installed.
fitscube = pytest.importorskip("fitscube")


def _write_image(
    path: Path,
    fill: float,
    *,
    reffreq: float | None = None,
    date_obs: str | None = None,
    beam: float | None = None,
    # 30x30 planes keep the cube well above fitscube's 1801-element small-cube
    # threshold. (Older fitscube builds corrupted data on the in-memory
    # small-cube path; v2.3.0 fixed it, but staying large keeps the tests robust
    # across reference versions.)
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
SPECTRAL_CARDS = ["NAXIS", "NAXIS3", "CTYPE3", "CRPIX3", "CRVAL3", "CDELT3"]


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
                    assert rs_h[card] == pytest.approx(ref_h[card], rel=1e-9, abs=1e-3), (
                        f"{card}: {rs_h[card]} != {ref_h[card]}"
                    )


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
            np.testing.assert_allclose(rs_beams[col], ref_beams[col], rtol=1e-4, atol=1e-4)


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

    fitscube.combine_fits(file_list=ref_files, out_cube=ref_cube, overwrite=True, time_domain_mode=True)
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
_fitscube_has_bbox_fix = Version(fitscube.__version__.split("+")[0].split(".dev")[0]) >= (
    _BBOX_FIX_VERSION
)


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
    fitscube.combine_fits(file_list=ref_files, out_cube=ref_cube, overwrite=True, bounding_box=True)
    with fits.open(ref_cube) as ref_hdul, fits.open(rs_cube) as rs_hdul:
        assert rs_hdul[0].data.shape == ref_hdul[0].data.shape
        for card in ("NAXIS1", "NAXIS2", "CRPIX1", "CRPIX2"):
            assert rs_hdul[0].header[card] == pytest.approx(ref_hdul[0].header[card]), card
        np.testing.assert_allclose(
            np.nan_to_num(rs_hdul[0].data), np.nan_to_num(ref_hdul[0].data), rtol=1e-6
        )


def test_extract_matches_input_plane(tmp_path):
    freqs = [1.0e9, 2.0e9, 3.0e9, 4.0e9]
    rs_files = _make_freq_images(tmp_path / "rs", freqs)
    cube = tmp_path / "cube.fits"
    fitscube_rs.combine_fits([str(f) for f in rs_files], str(cube), overwrite=True)

    out = fitscube_rs.extract_plane_from_cube(str(cube), channel_index=2, overwrite=True)
    with fits.open(out) as hdul:
        data = np.squeeze(hdul[0].data)
        # Channel 2 was filled with value 3.0 (fill = i + 1).
        np.testing.assert_allclose(data, 3.0, rtol=1e-6)
