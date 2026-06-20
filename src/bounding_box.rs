//! Bounding-box geometry for trimming blank padding from images.
//!
//! Port of `fitscube.bounding_box`. A [`BoundingBox`] captures the pixel extent
//! of the finite (non-NaN) data in a 2D image so that padded regions can be
//! clipped when building a cube. The naming follows the Python original, where
//! `x` is the slow (row / axis-0) direction and `y` is the fast (column /
//! axis-1) direction.
use ndarray::ArrayView2;
use num_traits::Float;

use crate::error::{FitsCubeError, Result};

/// Pixel bounds of the valid (finite) region of a 2D image.
///
/// `*max` bounds are exclusive and may be used directly as the upper end of a
/// slice (matching the Python `dataclass`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundingBox {
    /// Minimum x pixel (row / axis-0), inclusive.
    pub xmin: usize,
    /// Maximum x pixel (row / axis-0), exclusive.
    pub xmax: usize,
    /// Minimum y pixel (column / axis-1), inclusive.
    pub ymin: usize,
    /// Maximum y pixel (column / axis-1), exclusive.
    pub ymax: usize,
    /// Shape `(nrows, ncols)` of the plane the box was built from.
    pub original_shape: (usize, usize),
    /// `ymax - ymin`.
    pub y_span: usize,
    /// `xmax - xmin`.
    pub x_span: usize,
}

impl BoundingBox {
    fn new(
        xmin: usize,
        xmax: usize,
        ymin: usize,
        ymax: usize,
        original_shape: (usize, usize),
    ) -> Self {
        Self {
            xmin,
            xmax,
            ymin,
            ymax,
            original_shape,
            y_span: ymax - ymin,
            x_span: xmax - xmin,
        }
    }
}

/// Build a bounding box around the finite pixels of a 2D image.
///
/// Returns `None` if no pixel is finite (nothing to bound). Mirrors
/// `create_bound_box_plane`: `x` indexes rows (axis 0), `y` indexes columns
/// (axis 1); the `*max` values are made exclusive (`+1`).
pub fn create_bound_box_plane<T: Float>(image: &ArrayView2<T>) -> Option<BoundingBox> {
    let (nrows, ncols) = image.dim();

    let mut xmin: Option<usize> = None;
    let mut xmax = 0usize;
    let mut ymin: Option<usize> = None;
    let mut ymax = 0usize;
    let mut any_valid = false;

    for r in 0..nrows {
        for c in 0..ncols {
            if image[(r, c)].is_finite() {
                any_valid = true;
                if xmin.is_none() {
                    xmin = Some(r);
                }
                xmax = r;
                if ymin.is_none_or(|y| c < y) {
                    ymin = Some(c);
                }
                if c > ymax {
                    ymax = c;
                }
            }
        }
    }

    if !any_valid {
        tracing::info!("No pixels to create bounding box for");
        return None;
    }

    // xmin is set once we have seen any valid pixel; xmax tracks the last valid
    // row. ymin/ymax track first/last valid column. `+1` makes the max bounds
    // exclusive, matching numpy slicing in the Python original.
    let xmin = xmin.expect("any_valid implies xmin set");
    let ymin = ymin.expect("any_valid implies ymin set");
    Some(BoundingBox::new(
        xmin,
        xmax + 1,
        ymin,
        ymax + 1,
        (nrows, ncols),
    ))
}

/// Smallest bounding box that encompasses all the given boxes.
///
/// `None` entries (returned for fully blank images) are skipped. Errors if no
/// valid box remains or if the boxes disagree on `original_shape`. Mirrors
/// `extract_common_bounding_box`.
pub fn extract_common_bounding_box(boxes: &[Option<BoundingBox>]) -> Result<BoundingBox> {
    let valid: Vec<&BoundingBox> = boxes.iter().filter_map(|b| b.as_ref()).collect();

    if valid.is_empty() {
        return Err(FitsCubeError::Other(
            "No valid input boxes to consider".to_string(),
        ));
    }

    let shape0 = valid[0].original_shape;
    if !valid.iter().all(|b| b.original_shape == shape0) {
        return Err(FitsCubeError::Other(
            "Different shapes, and not sure this is really supported or meaningful".to_string(),
        ));
    }

    let xmin = valid.iter().map(|b| b.xmin).min().unwrap();
    let xmax = valid.iter().map(|b| b.xmax).max().unwrap();
    let ymin = valid.iter().map(|b| b.ymin).min().unwrap();
    let ymax = valid.iter().map(|b| b.ymax).max().unwrap();

    Ok(BoundingBox::new(xmin, xmax, ymin, ymax, shape0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn nan() -> f64 {
        f64::NAN
    }

    #[test]
    fn full_image_is_bounded_by_itself() {
        let img = Array2::<f64>::ones((4, 5));
        let bb = create_bound_box_plane(&img.view()).unwrap();
        assert_eq!(bb.xmin, 0);
        assert_eq!(bb.xmax, 4);
        assert_eq!(bb.ymin, 0);
        assert_eq!(bb.ymax, 5);
        assert_eq!(bb.x_span, 4);
        assert_eq!(bb.y_span, 5);
        assert_eq!(bb.original_shape, (4, 5));
    }

    #[test]
    fn nan_border_is_trimmed() {
        let mut img = Array2::<f64>::from_elem((5, 5), nan());
        // Valid block rows 1..=3, cols 2..=3.
        for r in 1..=3 {
            for c in 2..=3 {
                img[(r, c)] = 1.0;
            }
        }
        let bb = create_bound_box_plane(&img.view()).unwrap();
        assert_eq!((bb.xmin, bb.xmax), (1, 4));
        assert_eq!((bb.ymin, bb.ymax), (2, 4));
        assert_eq!(bb.x_span, 3);
        assert_eq!(bb.y_span, 2);
    }

    #[test]
    fn all_nan_returns_none() {
        let img = Array2::<f64>::from_elem((3, 3), nan());
        assert!(create_bound_box_plane(&img.view()).is_none());
    }

    #[test]
    fn common_box_is_union() {
        let a = create_bound_box_plane(
            &{
                let mut m = Array2::<f64>::from_elem((6, 6), nan());
                m[(1, 1)] = 1.0;
                m
            }
            .view(),
        )
        .unwrap();
        let b = create_bound_box_plane(
            &{
                let mut m = Array2::<f64>::from_elem((6, 6), nan());
                m[(4, 4)] = 1.0;
                m
            }
            .view(),
        )
        .unwrap();
        let common = extract_common_bounding_box(&[Some(a), None, Some(b)]).unwrap();
        assert_eq!((common.xmin, common.xmax), (1, 5));
        assert_eq!((common.ymin, common.ymax), (1, 5));
    }

    #[test]
    fn no_valid_boxes_errors() {
        assert!(extract_common_bounding_box(&[None, None]).is_err());
    }

    #[test]
    fn shape_mismatch_errors() {
        let a = BoundingBox::new(0, 1, 0, 1, (4, 4));
        let b = BoundingBox::new(0, 1, 0, 1, (5, 5));
        assert!(extract_common_bounding_box(&[Some(a), Some(b)]).is_err());
    }
}
