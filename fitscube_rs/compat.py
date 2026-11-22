"""The API of the Python ``fitscube`` package, over the Rust implementation.

Names, signatures and return types match fitscube, so code written against it
switches by changing its imports from ``fitscube.*`` to ``fitscube_rs.compat``.
Needs the ``compat`` extra: ``pip install fitscube-rs[compat]``.

>>> from fitscube_rs.compat import even_spacing
>>> import astropy.units as u
>>> even_spacing([1.0, 3.0, 4.0] * u.Hz).missing_chan_idx.tolist()
[False, True, False, False]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np

try:
    import astropy.units as u
except ImportError as e:
    msg = "fitscube_rs.compat needs astropy: pip install 'fitscube-rs[compat]'"
    raise ImportError(msg) from e

from fitscube_rs import (
    AxisMismatchException,
    AxisOrderException,
    BoundingBox,
    ChannelMissingException,
    FITSCubeException,
    IrregularSpacingException,
    ShapeMismatchException,
    StokesMismatchException,
    TargetAxisMissingException,
)
from fitscube_rs import _fitscube_rs as _rs

if TYPE_CHECKING:
    from astropy.io import fits
    from numpy.typing import NDArray

__all__ = [
    "AxisMismatchException",
    "AxisOrderException",
    "BoundingBox",
    "ChannelMissingException",
    "ExtractOptions",
    "ExtractPlanesOptions",
    "FITSCubeException",
    "IrregularSpacingException",
    "ShapeMismatchException",
    "SpequencyInfo",
    "StokesMismatchException",
    "TargetAxisMissingException",
    "TargetIndex",
    "TargetWCS",
    "combine_fits",
    "compress_cube",
    "count_cube_planes",
    "even_spacing",
    "extract_common_bounding_box",
    "extract_plane_from_cube",
    "extract_planes_from_cube",
    "find_target_axis",
    "get_common_bounding_box",
]


class SpequencyInfo(NamedTuple):
    """Frequency/time information."""

    specs: u.Quantity
    """Frequencies/Times"""
    missing_chan_idx: NDArray[np.bool_]
    """Missing channel indices"""


@dataclass
class ExtractOptions:
    """Basic options to extract a plane of data"""

    hdu_index: int = 0
    """The HDU in the fits cube to access (e.g. for header and data)"""
    channel_index: int | None = None
    """The channel of the cube to extract. Defaults to None"""
    time_index: int | None = None
    """The timestep of the cube to extract. Defaults to None"""
    overwrite: bool = False
    """overwrite the output file, if it exists"""
    output_path: Path | None = None
    """The output path of the new file. If None it is generated from the name of base fits cube."""


@dataclass
class ExtractPlanesOptions:
    """Options to extract every plane of a cube at once"""

    hdu_index: int = 0
    """The HDU in the fits cube to access (e.g. for header and data)"""
    overwrite: bool = False
    """overwrite the output files, if they exist"""
    output_paths: list[Path] | None = None
    """The output path of each plane, in channel order. If None they are generated from the name of the base fits cube."""
    max_workers: int | None = 4
    """The number of planes written at once. None places no limit."""


@dataclass
class TargetWCS:
    """The WCS of the target axis, taken straight from the FITS header."""

    axis: int
    """The axis count that the frequency corresponds to"""
    ctype: str
    """The FITS WCS key name"""
    crpix: float
    """The reference index position (FITS 1-based)"""
    crval: float
    """The reference value stored in the header"""
    cdelt: float
    """The step between planes, with any CDi_i or PCi_i applied"""
    cunit: str
    """the unit of the target axis"""


@dataclass
class TargetIndex:
    """Simple container to capture the mode to extract."""

    axis_name: str
    """The name of the axis to search for (e.g. TIME or FREQ)"""
    axis_index: int
    """The index (e.g. channel or timestep) to extract from the cube"""
    output_name: str
    """The output name to put in output files"""


def combine_fits(
    file_list: list[Path],
    out_cube: Path,
    spec_file: Path | None = None,
    spec_list: list[float] | None = None,
    ignore_spec: bool = False,
    create_blanks: bool = False,
    overwrite: bool = False,
    max_workers: int | None = None,
    time_domain_mode: bool = False,
    bounding_box: bool | BoundingBox = False,
    invalidate_zeros: bool = False,
    float_length: Literal[32, 64] | None = None,
    blank_zero_beams: bool = True,
) -> u.Quantity:
    """Combine FITS files into a cube; returns its frequencies (Hz) or times (s)."""
    specs = _rs.combine_fits(
        file_list,
        out_cube,
        spec_file=spec_file,
        spec_list=spec_list,
        ignore_spec=ignore_spec,
        create_blanks=create_blanks,
        overwrite=overwrite,
        max_workers=max_workers,
        time_domain_mode=time_domain_mode,
        bounding_box=bounding_box,
        invalidate_zeros=invalidate_zeros,
        float_length=float_length,
        blank_zero_beams=blank_zero_beams,
    )
    return np.array(specs) * (u.s if time_domain_mode else u.Hz)


def get_common_bounding_box(
    file_list: list[Path],
    invalidate_zeros: bool = False,
    max_workers: int | None = None,
) -> BoundingBox:
    """The smallest box that holds the finite pixels of every image."""
    return _rs.get_common_bounding_box(
        file_list, invalidate_zeros=invalidate_zeros, max_workers=max_workers
    )


def extract_common_bounding_box(
    bounding_boxes: list[BoundingBox | None],
) -> BoundingBox:
    """The smallest box that holds every given box; ``None`` entries are skipped."""
    return _rs.extract_common_bounding_box(bounding_boxes)


def even_spacing(specs: u.Quantity, time_domain_mode: bool = False) -> SpequencyInfo:
    """Put frequencies or times on a regular grid and flag the empty grid points."""
    values, missing = _rs.even_spacing(
        np.asarray(specs.value, dtype=np.float64).ravel().tolist(), time_domain_mode
    )
    return SpequencyInfo(np.array(values) * specs.unit, np.array(missing, dtype=bool))


def compress_cube(
    out_cube: Path,
    method: str = "gzip",
    max_workers: int | None = None,
    quantize_level: float = 4.0,
    tile_shape: tuple[int, int] | None = None,
    allow_requantize: bool = False,
    remove: bool = True,
) -> Path:
    """Compress a finished cube and return the path of the compressed one."""
    return Path(
        _rs.compress_cube(
            out_cube,
            method=method,
            max_workers=max_workers,
            quantize_level=quantize_level,
            tile_shape=tile_shape,
            allow_requantize=allow_requantize,
            remove=remove,
        )
    )


def extract_plane_from_cube(fits_cube: Path, extract_options: ExtractOptions) -> Path:
    """Write one channel or timestep of a cube to its own file."""
    return Path(
        _rs.extract_plane_from_cube(
            fits_cube,
            channel_index=extract_options.channel_index,
            time_index=extract_options.time_index,
            hdu_index=extract_options.hdu_index,
            overwrite=extract_options.overwrite,
            output_path=extract_options.output_path,
        )
    )


def extract_planes_from_cube(
    fits_cube: Path, extract_planes_options: ExtractPlanesOptions | None = None
) -> list[Path]:
    """Write every channel of a cube to its own file, in channel order."""
    opts = extract_planes_options or ExtractPlanesOptions()
    paths = _rs.extract_planes_from_cube(
        fits_cube,
        hdu_index=opts.hdu_index,
        overwrite=opts.overwrite,
        output_paths=opts.output_paths,
        max_workers=opts.max_workers,
    )
    return [Path(p) for p in paths]


def find_target_axis(
    header: fits.Header, target_index: TargetIndex | str = "FREQ"
) -> TargetWCS:
    """Find the axis whose CTYPE contains ``target_index`` (e.g. FREQ or TIME)."""
    axis_name = (
        target_index if isinstance(target_index, str) else target_index.axis_name
    )
    for axis in range(1, header["NAXIS"] + 1):
        if axis_name in header.get(f"CTYPE{axis}", ""):
            # CDi_i replaces CDELTi, PCi_i scales it. Absent keys take FITS defaults.
            cdelt = header.get(
                f"CD{axis}_{axis}",
                header.get(f"CDELT{axis}", 1.0) * header.get(f"PC{axis}_{axis}", 1.0),
            )
            return TargetWCS(
                axis=axis,
                ctype=header[f"CTYPE{axis}"],
                crpix=header.get(f"CRPIX{axis}", 0.0),
                crval=header.get(f"CRVAL{axis}", 0.0),
                cdelt=cdelt,
                cunit=header.get(f"CUNIT{axis}", ""),
            )
    msg = f"Did not find the {axis_name} axis"
    raise TargetAxisMissingException(msg)


def count_cube_planes(header: fits.Header) -> int:
    """The number of channels along the frequency axis of a cube."""
    return int(header[f"NAXIS{find_target_axis(header=header).axis}"])
