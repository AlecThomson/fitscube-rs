"""fitscube-rs: combine single-plane FITS images into a cube (Rust-backed).

A Rust port of the `fitscube <https://github.com/AlecThomson/fitscube>`_ Python
package. The heavy lifting happens in the compiled extension
``fitscube_rs._fitscube_rs``; this module re-exports its public API.
"""

from __future__ import annotations

from fitscube_rs._fitscube_rs import (
    BoundingBox,
    combine_fits,
    extract_plane_from_cube,
    get_common_bounding_box,
)

__all__ = [
    "BoundingBox",
    "combine_fits",
    "extract_plane_from_cube",
    "get_common_bounding_box",
]
