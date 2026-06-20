//! Low-level FITS helpers shared by the combine and extract pipelines.
//!
//! These were extracted into the [`atfits_rs`] crate (shared with `convolve-rs`):
//! update-in-place keyword editing, header geometry / WCS axis lookup, the
//! `f32`/`f64` [`CubeElem`] section I/O, and image-HDU creation/resize. This
//! module re-exports them so the combine/extract code keeps importing from
//! `crate::fits_io`.
pub use atfits_rs::{
    CubeElem, CubeLayout, HeaderGeom, PixelType, TargetAxis, bitpix_to_image_type,
    copy_header_only, copy_header_only_open, create_cube_open, create_mem_cube, delete_key,
    extract_header_layout, find_target_axis, has_key, read_key_f64, read_key_string, resize_image,
    update_key_f64, update_key_i64, update_key_logical, update_key_str, write_comment,
};
