//! Combine single-frequency (or single-time) FITS images into a cube, and
//! extract planes from one.
//!
//! A Rust port of the [`fitscube`](https://github.com/AlecThomson/fitscube)
//! Python package, also built as a Python module (`fitscube_rs`) and a CLI
//! (`fitscubers`).
//!
//! - [`combine_fits`] stacks images along frequency or time, with optional
//!   blank channels, per-channel beams and bounding-box trimming.
//! - [`extract_plane_from_cube`] and [`extract_planes_from_cube`] pull planes
//!   back out.
//!
//! The inputs must share a pixel grid, axes and Stokes parameter (checked).
//! Each image's frequency comes from its FREQ axis or REFFREQ, and its time
//! from DATE-OBS.
pub mod beams;
pub mod bounding_box;
pub mod checks;
pub mod combine;
pub mod compress;
pub mod error;
pub mod extract;
mod header;
mod partial;
mod pixels;
pub mod progress;
pub mod specs;
mod tile;
mod wcs;

pub use beams::Beam;
pub use bounding_box::{
    BoundingBox, create_bound_box_plane, extract_common_bounding_box, get_common_bounding_box,
};
pub use combine::{CombineOptions, combine_fits};
pub use error::{FitsCubeError, Result};
pub use extract::{
    ExtractOptions, ExtractPlanesOptions, extract_plane_from_cube, extract_planes_from_cube,
};

#[cfg(feature = "python")]
mod python;
#[cfg(feature = "python")]
pub use python::_fitscube_rs;
