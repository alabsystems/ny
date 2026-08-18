// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sliding window extraction (im2col) for image tensors.
//!
//! Provides [`inplace_unfold`] which extracts overlapping patches from spatial
//! dimensions, producing an output compatible with the Patches mode 6D format
//! used in CROWN backward propagation for CNNs.
//!
//! # Reference
//!
//! alpha-beta-CROWN `auto_LiRPA/patches.py::inplace_unfold()` uses
//! `torch.as_strided` for zero-copy views. This implementation allocates the
//! output tensor since `ny-tensor` forbids `unsafe` code. The allocation
//! cost is acceptable: unfold is called once per backward pass at termination
//! points (Linear layer) or for slope unfolding in activation layers.

use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};

/// Sliding window extraction (im2col) for image tensors.
///
/// Extracts overlapping patches from the spatial (H, W) dimensions of an image
/// tensor. Each output position `(oh, ow)` contains the receptive field window
/// of size `(kH, kW)` centered (with stride/padding) at that position.
///
/// # Arguments
///
/// * `image` - Input tensor of shape `(batch, C, H, W)` or `(C, H, W)`.
/// * `kernel_size` - `(kH, kW)` window dimensions.
/// * `stride` - `(sH, sW)` step between successive windows.
/// * `padding` - `(left, right, top, bottom)` zero-padding applied before extraction.
///
/// # Returns
///
/// `ArrayD<f32>` of shape:
/// - `(batch, out_H, out_W, C, kH, kW)` for 4D input
/// - `(out_H, out_W, C, kH, kW)` for 3D input
///
/// Where:
/// ```text
///   out_H = (H + pad_top + pad_bottom - kH) / sH + 1
///   out_W = (W + pad_left + pad_right - kW) / sW + 1
/// ```
///
/// # Errors
///
/// Returns an error if the input is not 3D or 4D, the kernel or stride contains
/// a zero dimension, the padded/output shape overflows, or the padded spatial
/// dimensions are smaller than the kernel.
///
/// # Reference
///
/// alpha-beta-CROWN `auto_LiRPA/patches.py::inplace_unfold()`.
/// PyTorch equivalent: `torch.nn.Unfold` / `F.unfold`.
pub fn inplace_unfold(
    image: &ArrayD<f32>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize, usize, usize),
) -> Result<ArrayD<f32>> {
    let ndim = image.ndim();
    if ndim != 3 && ndim != 4 {
        return Err(NyError::ShapeMismatch {
            expected: vec![3, 4],
            got: vec![ndim],
        });
    }

    let has_batch = ndim == 4;
    let (batch, channels, height, width) = if has_batch {
        (
            image.shape()[0],
            image.shape()[1],
            image.shape()[2],
            image.shape()[3],
        )
    } else {
        (1, image.shape()[0], image.shape()[1], image.shape()[2])
    };

    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (pad_left, pad_right, pad_top, pad_bottom) = padding;

    if kh == 0 || kw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "inplace_unfold requires a non-empty kernel, got {kernel_size:?}"
        )));
    }
    if sh == 0 || sw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "inplace_unfold requires non-zero stride, got {stride:?}"
        )));
    }

    let padded_h = height
        .checked_add(pad_top)
        .and_then(|value| value.checked_add(pad_bottom))
        .ok_or_else(|| {
            NyError::InvalidSpec("inplace_unfold vertical padding overflows usize".to_string())
        })?;
    let padded_w = width
        .checked_add(pad_left)
        .and_then(|value| value.checked_add(pad_right))
        .ok_or_else(|| {
            NyError::InvalidSpec("inplace_unfold horizontal padding overflows usize".to_string())
        })?;

    if padded_h < kh || padded_w < kw {
        return Err(NyError::ShapeMismatch {
            expected: vec![kh, kw],
            got: vec![padded_h, padded_w],
        });
    }

    let out_h = (padded_h - kh)
        .checked_div(sh)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec("inplace_unfold output height overflows usize".to_string())
        })?;
    let out_w = (padded_w - kw)
        .checked_div(sw)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec("inplace_unfold output width overflows usize".to_string())
        })?;

    // Output shape: (batch, out_h, out_w, C, kH, kW) or (out_h, out_w, C, kH, kW)
    let out_shape: Vec<usize> = if has_batch {
        vec![batch, out_h, out_w, channels, kh, kw]
    } else {
        vec![out_h, out_w, channels, kh, kw]
    };
    let total_elems = checked_shape_product(&out_shape).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "inplace_unfold output shape product overflows: {out_shape:?}"
        ))
    })?;
    if total_elems == 0 {
        return ArrayD::from_shape_vec(IxDyn(&out_shape), Vec::new()).map_err(|e| {
            NyError::InternalError(format!("inplace_unfold shape construction failed: {e}"))
        });
    }
    let mut data = Vec::new();
    data.try_reserve_exact(total_elems).map_err(|error| {
        NyError::InvalidSpec(format!(
            "inplace_unfold output allocation failed for {total_elems} elements: {error}"
        ))
    })?;

    // Fill the output tensor by iterating over all window positions.
    // Memory layout: row-major (C-order) matching ndarray default.
    for b in 0..batch {
        for oh in 0..out_h {
            for ow in 0..out_w {
                for c in 0..channels {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            // Compute source position in the (possibly padded) image
                            let ih_padded = oh * sh + ki;
                            let iw_padded = ow * sw + kj;

                            // Map back to original image coordinates without signed
                            // casts, which would wrap for padding above isize::MAX.
                            let value = match (
                                ih_padded.checked_sub(pad_top),
                                iw_padded.checked_sub(pad_left),
                            ) {
                                (Some(ih), Some(iw)) if ih < height && iw < width => {
                                    if has_batch {
                                        image[[b, c, ih, iw].as_slice()]
                                    } else {
                                        image[[c, ih, iw].as_slice()]
                                    }
                                }
                                _ => 0.0, // Zero-padding
                            };

                            // The loop order is exactly ndarray's row-major output
                            // order, so appending avoids a second set of overflow-prone
                            // flat-index products.
                            data.push(value);
                        }
                    }
                }
            }
        }
    }
    debug_assert_eq!(data.len(), total_elems);

    ArrayD::from_shape_vec(IxDyn(&out_shape), data).map_err(|e| {
        NyError::InternalError(format!("inplace_unfold shape construction failed: {e}"))
    })
}

/// Compute the output spatial dimensions for a sliding window operation.
///
/// Returns `(out_H, out_W)` given the input size, kernel, stride, and padding.
pub fn unfold_output_size(
    height: usize,
    width: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize, usize, usize),
) -> (usize, usize) {
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (pad_left, pad_right, pad_top, pad_bottom) = padding;

    fn output_dim(
        input: usize,
        kernel: usize,
        stride: usize,
        pad_before: usize,
        pad_after: usize,
    ) -> usize {
        if kernel == 0 || stride == 0 {
            return 0;
        }
        let Some(padded) = input
            .checked_add(pad_before)
            .and_then(|value| value.checked_add(pad_after))
        else {
            return 0;
        };
        if padded < kernel {
            return 0;
        }
        (padded - kernel) / stride + 1
    }

    let out_h = output_dim(height, kh, sh, pad_top, pad_bottom);
    let out_w = output_dim(width, kw, sw, pad_left, pad_right);
    (out_h, out_w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_unfold_identity_no_padding() {
        // 1-channel 3x3 image, 1x1 kernel, stride 1, no padding
        // Should extract each pixel as a 1x1 patch
        let image =
            Array3::from_shape_vec((1, 3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap()
                .into_dyn();

        let result = inplace_unfold(&image, (1, 1), (1, 1), (0, 0, 0, 0)).unwrap();
        // Output shape: (3, 3, 1, 1, 1) = (out_h, out_w, C, kH, kW)
        assert_eq!(result.shape(), &[3, 3, 1, 1, 1]);
        // Each patch should be the corresponding pixel
        assert_eq!(result[[0, 0, 0, 0, 0]], 1.0);
        assert_eq!(result[[0, 1, 0, 0, 0]], 2.0);
        assert_eq!(result[[1, 1, 0, 0, 0]], 5.0);
        assert_eq!(result[[2, 2, 0, 0, 0]], 9.0);
    }

    #[test]
    fn test_unfold_3x3_kernel() {
        // 1-channel 4x4 image, 3x3 kernel, stride 1, no padding
        // Output: (2, 2) spatial positions
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let image = Array3::from_shape_vec((1, 4, 4), data).unwrap().into_dyn();

        let result = inplace_unfold(&image, (3, 3), (1, 1), (0, 0, 0, 0)).unwrap();
        assert_eq!(result.shape(), &[2, 2, 1, 3, 3]);

        // Top-left patch: rows 0-2, cols 0-2
        assert_eq!(result[[0, 0, 0, 0, 0]], 1.0);
        assert_eq!(result[[0, 0, 0, 0, 1]], 2.0);
        assert_eq!(result[[0, 0, 0, 0, 2]], 3.0);
        assert_eq!(result[[0, 0, 0, 1, 0]], 5.0);
        assert_eq!(result[[0, 0, 0, 2, 2]], 11.0);

        // Bottom-right patch: rows 1-3, cols 1-3
        assert_eq!(result[[1, 1, 0, 0, 0]], 6.0);
        assert_eq!(result[[1, 1, 0, 2, 2]], 16.0);
    }

    #[test]
    fn test_unfold_with_padding() {
        // 1-channel 2x2 image, 3x3 kernel, stride 1, padding 1 on all sides
        // Padded size: 4x4, output: (2, 2) positions
        let image = Array3::from_shape_vec((1, 2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .into_dyn();

        let result = inplace_unfold(&image, (3, 3), (1, 1), (1, 1, 1, 1)).unwrap();
        assert_eq!(result.shape(), &[2, 2, 1, 3, 3]);

        // Top-left patch includes padding zeros
        assert_eq!(result[[0, 0, 0, 0, 0]], 0.0); // pad
        assert_eq!(result[[0, 0, 0, 0, 1]], 0.0); // pad
        assert_eq!(result[[0, 0, 0, 1, 0]], 0.0); // pad
        assert_eq!(result[[0, 0, 0, 1, 1]], 1.0); // image[0,0]
        assert_eq!(result[[0, 0, 0, 1, 2]], 2.0); // image[0,1]
        assert_eq!(result[[0, 0, 0, 2, 1]], 3.0); // image[1,0]
        assert_eq!(result[[0, 0, 0, 2, 2]], 4.0); // image[1,1]
    }

    #[test]
    fn test_unfold_with_stride() {
        // 1-channel 4x4 image, 2x2 kernel, stride 2, no padding
        // Output: (2, 2) positions — non-overlapping windows
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let image = Array3::from_shape_vec((1, 4, 4), data).unwrap().into_dyn();

        let result = inplace_unfold(&image, (2, 2), (2, 2), (0, 0, 0, 0)).unwrap();
        assert_eq!(result.shape(), &[2, 2, 1, 2, 2]);

        // Top-left: rows 0-1, cols 0-1
        assert_eq!(result[[0, 0, 0, 0, 0]], 1.0);
        assert_eq!(result[[0, 0, 0, 0, 1]], 2.0);
        assert_eq!(result[[0, 0, 0, 1, 0]], 5.0);
        assert_eq!(result[[0, 0, 0, 1, 1]], 6.0);

        // Bottom-right: rows 2-3, cols 2-3
        assert_eq!(result[[1, 1, 0, 0, 0]], 11.0);
        assert_eq!(result[[1, 1, 0, 0, 1]], 12.0);
        assert_eq!(result[[1, 1, 0, 1, 0]], 15.0);
        assert_eq!(result[[1, 1, 0, 1, 1]], 16.0);
    }

    #[test]
    fn test_unfold_multi_channel() {
        // 2-channel 3x3 image, 2x2 kernel, stride 1, no padding
        // Output: (2, 2, 2, 2, 2)
        #[rustfmt::skip]
        let data = vec![
            // Channel 0
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            // Channel 1
            10.0, 20.0, 30.0,
            40.0, 50.0, 60.0,
            70.0, 80.0, 90.0,
        ];
        let image = Array3::from_shape_vec((2, 3, 3), data).unwrap().into_dyn();

        let result = inplace_unfold(&image, (2, 2), (1, 1), (0, 0, 0, 0)).unwrap();
        assert_eq!(result.shape(), &[2, 2, 2, 2, 2]);

        // Position (0,0), channel 0
        assert_eq!(result[[0, 0, 0, 0, 0]], 1.0);
        assert_eq!(result[[0, 0, 0, 0, 1]], 2.0);
        assert_eq!(result[[0, 0, 0, 1, 0]], 4.0);
        assert_eq!(result[[0, 0, 0, 1, 1]], 5.0);

        // Position (0,0), channel 1
        assert_eq!(result[[0, 0, 1, 0, 0]], 10.0);
        assert_eq!(result[[0, 0, 1, 0, 1]], 20.0);
        assert_eq!(result[[0, 0, 1, 1, 0]], 40.0);
        assert_eq!(result[[0, 0, 1, 1, 1]], 50.0);
    }

    #[test]
    fn test_unfold_batched_4d() {
        // Batch=2, 1-channel 2x2, 2x2 kernel, stride 1, no padding → (2, 1, 1, 1, 2, 2)
        let data = vec![
            // Batch 0: channel 0
            1.0, 2.0, 3.0, 4.0, // Batch 1: channel 0
            10.0, 20.0, 30.0, 40.0,
        ];
        let image = ArrayD::from_shape_vec(IxDyn(&[2, 1, 2, 2]), data).unwrap();

        let result = inplace_unfold(&image, (2, 2), (1, 1), (0, 0, 0, 0)).unwrap();
        assert_eq!(result.shape(), &[2, 1, 1, 1, 2, 2]);

        // Batch 0
        assert_eq!(result[[0, 0, 0, 0, 0, 0]], 1.0);
        assert_eq!(result[[0, 0, 0, 0, 0, 1]], 2.0);
        assert_eq!(result[[0, 0, 0, 0, 1, 0]], 3.0);
        assert_eq!(result[[0, 0, 0, 0, 1, 1]], 4.0);

        // Batch 1
        assert_eq!(result[[1, 0, 0, 0, 0, 0]], 10.0);
        assert_eq!(result[[1, 0, 0, 0, 1, 1]], 40.0);
    }

    #[test]
    fn test_unfold_output_size() {
        assert_eq!(
            unfold_output_size(4, 4, (3, 3), (1, 1), (0, 0, 0, 0)),
            (2, 2)
        );
        assert_eq!(
            unfold_output_size(4, 4, (2, 2), (2, 2), (0, 0, 0, 0)),
            (2, 2)
        );
        assert_eq!(
            unfold_output_size(3, 3, (3, 3), (1, 1), (1, 1, 1, 1)),
            (3, 3)
        );
        assert_eq!(
            unfold_output_size(5, 5, (3, 3), (2, 2), (0, 0, 0, 0)),
            (2, 2)
        );
    }

    #[test]
    fn test_unfold_invalid_shape() {
        let image = ArrayD::from_shape_vec(IxDyn(&[3, 3]), vec![0.0; 9]).unwrap();
        assert!(inplace_unfold(&image, (2, 2), (1, 1), (0, 0, 0, 0)).is_err());
    }

    #[test]
    fn test_unfold_kernel_too_large() {
        let image = Array3::from_shape_vec((1, 2, 2), vec![1.0; 4])
            .unwrap()
            .into_dyn();
        // 5x5 kernel on 2x2 image with no padding → impossible
        assert!(inplace_unfold(&image, (5, 5), (1, 1), (0, 0, 0, 0)).is_err());
    }

    #[test]
    fn test_unfold_rejects_zero_kernel_and_stride_without_panicking() {
        let image = Array3::from_elem((1, 1, 1), 1.0).into_dyn();

        assert!(inplace_unfold(&image, (0, 1), (1, 1), (0, 0, 0, 0)).is_err());
        assert!(inplace_unfold(&image, (1, 1), (0, 1), (0, 0, 0, 0)).is_err());
        assert_eq!(
            unfold_output_size(1, 1, (0, 1), (1, 1), (0, 0, 0, 0)),
            (0, 1)
        );
        assert_eq!(
            unfold_output_size(1, 1, (1, 1), (0, 1), (0, 0, 0, 0)),
            (0, 1)
        );
    }

    #[test]
    fn test_unfold_rejects_padding_and_output_shape_overflow() {
        let image = Array3::from_elem((1, 1, 1), 1.0).into_dyn();
        assert!(inplace_unfold(&image, (1, 1), (1, 1), (usize::MAX, 0, 0, 0)).is_err());

        let two_channels = Array3::from_elem((2, 1, 1), 1.0).into_dyn();
        assert!(inplace_unfold(&two_channels, (1, 1), (1, 1), (0, usize::MAX - 1, 0, 0),).is_err());

        assert_eq!(
            unfold_output_size(1, 1, (1, 1), (1, 1), (usize::MAX, 0, 0, 0)),
            (1, 0)
        );
    }

    #[test]
    fn test_unfold_large_padding_does_not_wrap_through_isize() {
        let image = Array3::from_elem((1, 1, 1), 7.0).into_dyn();
        let large_pad = isize::MAX as usize + 1;

        let result = inplace_unfold(&image, (1, 1), (usize::MAX, 1), (0, 0, large_pad, 0))
            .expect("large valid padding must not wrap through a signed cast");

        assert_eq!(result.shape(), &[1, 1, 1, 1, 1]);
        assert_eq!(result[[0, 0, 0, 0, 0]], 0.0);
    }

    #[test]
    fn test_unfold_zero_volume_returns_before_large_coordinate_loop() {
        let image = Array3::<f32>::zeros((0, 1, 1)).into_dyn();
        let result = inplace_unfold(&image, (1, 1), (1, 1), (0, 1_000_000, 0, 0))
            .expect("zero-channel output is a valid empty tensor");

        assert_eq!(result.shape(), &[1, 1_000_001, 0, 1, 1]);
        assert!(result.is_empty());
    }
}
