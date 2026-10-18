"""fitscube-rs: combine single-plane FITS images into a cube (Rust-backed).

A Rust port of the `fitscube <https://github.com/AlecThomson/fitscube>`_ Python
package. The work is done in the compiled extension
``fitscube_rs._fitscube_rs``; this module re-exports its public API and
defines the exceptions it raises.
"""

from __future__ import annotations

from fitscube_rs._fitscube_rs import (
    BoundingBox,
    combine_fits,
    compress_cube,
    decompress_cube,
    extract_plane_from_cube,
    extract_planes_from_cube,
    get_common_bounding_box,
)

__all__ = [
    "AxisMismatchException",
    "AxisOrderException",
    "BoundingBox",
    "ChannelMissingException",
    "FITSCubeException",
    "IrregularSpacingException",
    "ShapeMismatchException",
    "StokesMismatchException",
    "TargetAxisMissingException",
    "combine_fits",
    "compress_cube",
    "decompress_cube",
    "extract_plane_from_cube",
    "extract_planes_from_cube",
    "get_common_bounding_box",
]


# The exception names and hierarchy of fitscube.exceptions
class FITSCubeException(Exception):
    """Base class of the fitscube exceptions."""


class TargetAxisMissingException(FITSCubeException):
    """The FREQ or TIME axis is missing."""


class ChannelMissingException(FITSCubeException):
    """A channel cannot be accessed."""


class ShapeMismatchException(FITSCubeException):
    """Input images do not share a pixel grid."""


class AxisMismatchException(FITSCubeException):
    """Input images do not share the same axes, in the same order."""


class StokesMismatchException(FITSCubeException):
    """Input images do not share the same Stokes parameter."""


class AxisOrderException(FITSCubeException):
    """The combine axis is not the slowest-varying axis of the cube."""


class IrregularSpacingException(FITSCubeException):
    """A regular grid through the inputs would not hold every input."""
