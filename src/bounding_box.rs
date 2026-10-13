//! The box around the finite pixels of an image, used to trim blank padding.
//!
//! As in fitscube, `x` is the row (numpy axis 0, FITS NAXIS2) and `y` the
//! column (numpy axis 1, FITS NAXIS1).
use std::path::{Path, PathBuf};

use fitsio::FitsFile;
use ndarray::ArrayView2;
use num_traits::Float;
use rayon::prelude::*;

use crate::error::{FitsCubeError, Result};
use crate::pixels::{Pixel, read_pixels};

/// Pixel bounds of the finite data of a plane. Maxima are exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundingBox {
    pub xmin: usize,
    pub xmax: usize,
    pub ymin: usize,
    pub ymax: usize,
    /// `(nrows, ncols)` of the plane.
    pub original_shape: (usize, usize),
    /// `ymax - ymin`, the trimmed NAXIS1.
    pub y_span: usize,
    /// `xmax - xmin`, the trimmed NAXIS2.
    pub x_span: usize,
}

impl BoundingBox {
    pub fn new(
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

/// Box around the finite pixels of `image`, or `None` if there are none.
pub fn create_bound_box_plane<T: Float>(image: &ArrayView2<T>) -> Option<BoundingBox> {
    let (nrows, ncols) = image.dim();
    let mut rows: Option<(usize, usize)> = None;
    let (mut ymin, mut ymax) = (ncols, 0);
    for (r, row) in image.rows().into_iter().enumerate() {
        let Some(first) = row.iter().position(|v| v.is_finite()) else {
            continue;
        };
        let last = row.iter().rposition(|v| v.is_finite()).unwrap_or(first);
        rows = Some((rows.map_or(r, |(xmin, _)| xmin), r));
        ymin = ymin.min(first);
        ymax = ymax.max(last);
    }
    let Some((xmin, xmax)) = rows else {
        tracing::info!("No pixels to create bounding box for");
        return None;
    };
    Some(BoundingBox::new(
        xmin,
        xmax + 1,
        ymin,
        ymax + 1,
        (nrows, ncols),
    ))
}

/// Box around the finite pixels of one image, or `None` if there are none.
///
/// The image must hold one plane once degenerate axes are dropped.
pub fn get_bounding_box_for_fits(
    path: &Path,
    invalidate_zeros: bool,
) -> Result<Option<BoundingBox>> {
    let mut fptr = FitsFile::open(path)?;
    let hdu = fptr.primary_hdu()?;
    let (dims, bitpix): (Vec<usize>, i64) = match &hdu.info {
        fitsio::hdu::HduInfo::ImageInfo { shape, .. } => (
            shape.iter().rev().copied().collect(),
            hdu.read_key(&mut fptr, "BITPIX")?,
        ),
        _ => (vec![], 0),
    };
    let squeezed: Vec<usize> = dims.iter().copied().filter(|&n| n != 1).collect();
    let [ncols, nrows] = squeezed[..] else {
        return Err(FitsCubeError::ShapeMismatch(format!(
            "Only two-dimensional images can be bounded, but {} has shape {dims:?}",
            path.display(),
        )));
    };
    if bitpix == -64 {
        plane_box::<f64>(&mut fptr, &dims, (nrows, ncols), bitpix, invalidate_zeros)
    } else {
        plane_box::<f32>(&mut fptr, &dims, (nrows, ncols), bitpix, invalidate_zeros)
    }
}

fn plane_box<T: Pixel + Float>(
    fptr: &mut FitsFile,
    dims: &[usize],
    shape: (usize, usize),
    bitpix: i64,
    invalidate_zeros: bool,
) -> Result<Option<BoundingBox>> {
    let mut data = Vec::new();
    // BLANK pixels of an integer image are not data
    read_pixels(fptr, dims, None, (bitpix > 0).then(T::nan), &mut data)?;
    if invalidate_zeros {
        for v in data.iter_mut().filter(|v| v.is_zero()) {
            *v = T::nan();
        }
    }
    let view = ArrayView2::from_shape(shape, &data)
        .map_err(|e| FitsCubeError::ShapeMismatch(e.to_string()))?;
    Ok(create_bound_box_plane(&view))
}

/// The smallest box holding the finite pixels of every image in `file_list`.
///
/// Pass it to several combines as `supplied_bounding_box` to put them on one
/// pixel grid, such as an image cube and its weights.
pub fn get_common_bounding_box(
    file_list: &[PathBuf],
    invalidate_zeros: bool,
) -> Result<BoundingBox> {
    let boxes = file_list
        .par_iter()
        .map(|p| get_bounding_box_for_fits(p, invalidate_zeros))
        .collect::<Result<Vec<_>>>()?;
    extract_common_bounding_box(&boxes)
}

/// The smallest box holding all of `boxes`, skipping `None`s.
pub fn extract_common_bounding_box(boxes: &[Option<BoundingBox>]) -> Result<BoundingBox> {
    let valid: Vec<&BoundingBox> = boxes.iter().flatten().collect();
    let Some(first) = valid.first() else {
        return Err(FitsCubeError::NoValidData);
    };
    if valid
        .iter()
        .any(|b| b.original_shape != first.original_shape)
    {
        return Err(FitsCubeError::ShapeMismatch(
            "bounding boxes come from planes of different shapes".to_string(),
        ));
    }
    Ok(BoundingBox::new(
        valid.iter().map(|b| b.xmin).min().unwrap_or(0),
        valid.iter().map(|b| b.xmax).max().unwrap_or(0),
        valid.iter().map(|b| b.ymin).min().unwrap_or(0),
        valid.iter().map(|b| b.ymax).max().unwrap_or(0),
        first.original_shape,
    ))
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
