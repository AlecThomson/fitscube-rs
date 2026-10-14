"""Errors from the Rust layer arrive as the matching Python exceptions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from astropy.io import fits

import fitscube_rs
from fitscube_rs import (
    ChannelMissingException,
    FITSCubeException,
    IrregularSpacingException,
    ShapeMismatchException,
)

if TYPE_CHECKING:
    from pathlib import Path


def _image(path: Path, freq: float, shape=(8, 8)) -> str:
    header = fits.Header()
    header["REFFREQ"] = freq
    fits.writeto(path, np.ones(shape, dtype=np.float32), header, overwrite=True)
    return str(path)


@pytest.fixture
def images(tmp_path):
    return [_image(tmp_path / f"p{i}.fits", 1e9 + i * 1e6) for i in range(3)]


def test_exceptions_share_a_base():
    for exc in (ChannelMissingException, IrregularSpacingException):
        assert issubclass(exc, FITSCubeException)


def test_output_exists(images, tmp_path):
    cube = tmp_path / "cube.fits"
    fitscube_rs.combine_fits(images, cube)
    with pytest.raises(FileExistsError):
        fitscube_rs.combine_fits(images, cube)


def test_missing_input(images, tmp_path):
    with pytest.raises(FileNotFoundError):
        fitscube_rs.combine_fits([*images, tmp_path / "nope.fits"], tmp_path / "c.fits")
    with pytest.raises(FileNotFoundError):
        fitscube_rs.extract_plane_from_cube(tmp_path / "nope.fits", channel_index=0)


def test_shape_mismatch(images, tmp_path):
    odd = _image(tmp_path / "odd.fits", 2e9, shape=(8, 9))
    with pytest.raises(ShapeMismatchException):
        fitscube_rs.combine_fits([*images, odd], tmp_path / "c.fits")


def test_irregular_spacing(tmp_path):
    offsets = [0.0, 1.0, 1.0 + np.sqrt(2)]
    files = [
        _image(tmp_path / f"p{i}.fits", 1e9 + 1e7 * f) for i, f in enumerate(offsets)
    ]
    with pytest.raises(IrregularSpacingException):
        fitscube_rs.combine_fits(files, tmp_path / "c.fits", create_blanks=True)


def test_channel_missing(images, tmp_path):
    cube = tmp_path / "cube.fits"
    fitscube_rs.combine_fits(images, cube)
    with pytest.raises(ChannelMissingException):
        fitscube_rs.extract_plane_from_cube(cube, channel_index=3)


def test_other_errors_are_value_errors(images, tmp_path):
    with pytest.raises(ValueError, match="32 or 64"):
        fitscube_rs.combine_fits(images, tmp_path / "c.fits", float_length=16)
