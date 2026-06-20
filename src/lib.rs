//! Combine single-frequency (or single-time) FITS images into a FITS cube.
//!
//! A Rust port of the [`fitscube`](https://github.com/AlecThomson/fitscube)
//! Python package. Also available as a Python package (`fitscube_rs`) and a CLI
//! tool (`fitscube`).
//!
//! # Overview
//!
//! - [`combine_fits`]: stack a list of images into a cube along the frequency
//!   or time axis ([`combine`] module), handling uneven spacing, blank-channel
//!   interpolation, per-channel beams, and bounding-box trimming.
//! - [`extract_plane_from_cube`]: pull one channel/timestep back out of a cube
//!   ([`extract`] module).
//!
//! Assumptions (inherited from the original): all inputs share the same WCS and
//! pixel grid, and the frequency/time of each image is available either as a WCS
//! axis, the `REFFREQ` keyword, or the `DATE-OBS` keyword (time mode).
pub mod beams;
pub mod bounding_box;
pub mod combine;
pub mod error;
pub mod extract;
pub mod fits_io;
pub mod specs;

pub use beams::Beam;
pub use bounding_box::{BoundingBox, create_bound_box_plane, extract_common_bounding_box};
pub use combine::{CombineOptions, combine_fits};
pub use error::{FitsCubeError, Result};
pub use extract::{ExtractOptions, extract_plane_from_cube};

#[cfg(feature = "python")]
mod python;
#[cfg(feature = "python")]
pub use python::_fitscube_rs;
