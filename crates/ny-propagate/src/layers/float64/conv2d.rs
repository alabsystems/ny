// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Double-precision Conv2D layer for f64 propagation.
//!
//! IBP: standard positive/negative kernel splitting with f64 arithmetic.
//! CROWN backward: transposed convolution on coefficient matrices.
//!
//! Reference: alpha-beta-CROWN `auto_lirpa/operators/convolution.py`.

use ndarray::{s, Array1, Array2, Array4};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor64;

use crate::bounds::LinearBounds64;

/// Conv2D parameters (shared between IBP and CROWN paths).
#[derive(Debug, Clone)]
pub struct Conv2dParams {
    /// Stride (height, width).
    pub stride: (usize, usize),
    /// Padding (height, width).
    pub padding: (usize, usize),
    /// Input spatial dimensions (height, width). Required for CROWN backward.
    pub input_hw: (usize, usize),
}

/// Compute Conv2D output spatial dimensions.
fn conv2d_output_hw(
    input_hw: (usize, usize),
    kernel_hw: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> (usize, usize) {
    let out_h = (input_hw.0 + 2 * padding.0 - kernel_hw.0) / stride.0 + 1;
    let out_w = (input_hw.1 + 2 * padding.1 - kernel_hw.1) / stride.1 + 1;
    (out_h, out_w)
}

/// Single-channel convolution in f64 (no batching).
///
/// input: (in_h, in_w), kernel: (kh, kw) -> output: (out_h, out_w)
fn conv2d_single_f64(
    input: &ndarray::ArrayView2<f64>,
    kernel: &ndarray::ArrayView2<f64>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Array2<f64> {
    let (in_h, in_w) = (input.nrows(), input.ncols());
    let (kh, kw) = (kernel.nrows(), kernel.ncols());
    let (out_h, out_w) = conv2d_output_hw((in_h, in_w), (kh, kw), stride, padding);

    let mut output = Array2::<f64>::zeros((out_h, out_w));

    for oh in 0..out_h {
        for ow in 0..out_w {
            let mut sum = 0.0f64;
            for khi in 0..kh {
                for kwi in 0..kw {
                    let ih = oh * stride.0 + khi;
                    let iw = ow * stride.1 + kwi;
                    // Account for padding
                    let ih_actual = ih as isize - padding.0 as isize;
                    let iw_actual = iw as isize - padding.1 as isize;
                    if ih_actual >= 0
                        && ih_actual < in_h as isize
                        && iw_actual >= 0
                        && iw_actual < in_w as isize
                    {
                        sum += input[[ih_actual as usize, iw_actual as usize]] * kernel[[khi, kwi]];
                    }
                }
            }
            output[[oh, ow]] = sum;
        }
    }
    output
}

/// IBP propagation for Conv2D in f64.
///
/// Input shape: (C_in, H, W). Kernel shape: (C_out, C_in, KH, KW).
///
/// Uses W+/W- splitting:
///   lower = conv(x_lower, W+) + conv(x_upper, W-) + bias
///   upper = conv(x_upper, W+) + conv(x_lower, W-) + bias
pub(crate) fn propagate_conv2d_ibp_f64(
    kernel: &Array4<f64>,
    bias: &Array1<f64>,
    input: &BoundedTensor64,
    params: &Conv2dParams,
) -> Result<BoundedTensor64> {
    let input_shape = input.shape();
    if input_shape.len() != 3 {
        return Err(NyError::InvalidSpec(format!(
            "Conv2D f64 IBP expects 3D input (C,H,W), got shape {:?}",
            input_shape
        )));
    }

    let c_in = input_shape[0];
    let in_h = input_shape[1];
    let in_w = input_shape[2];
    let c_out = kernel.shape()[0];
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];

    if kernel.shape()[1] != c_in {
        return Err(NyError::shape_mismatch(vec![c_in], vec![kernel.shape()[1]]));
    }

    let (out_h, out_w) = conv2d_output_hw((in_h, in_w), (kh, kw), params.stride, params.padding);

    // Reshape input bounds to 3D views
    let in_l = input
        .lower()
        .clone()
        .into_shape_with_order((c_in, in_h, in_w))
        .map_err(|e| NyError::InvalidSpec(format!("Conv2D f64: input reshape failed: {e}")))?;
    let in_u = input
        .upper()
        .clone()
        .into_shape_with_order((c_in, in_h, in_w))
        .map_err(|e| NyError::InvalidSpec(format!("Conv2D f64: input reshape failed: {e}")))?;

    let mut lower = ndarray::Array3::<f64>::zeros((c_out, out_h, out_w));
    let mut upper = ndarray::Array3::<f64>::zeros((c_out, out_h, out_w));

    for oc in 0..c_out {
        // Add bias
        lower.slice_mut(s![oc, .., ..]).fill(bias[oc]);
        upper.slice_mut(s![oc, .., ..]).fill(bias[oc]);

        for ic in 0..c_in {
            let k_slice = kernel.slice(s![oc, ic, .., ..]);
            // Split kernel into positive and negative parts
            let k_pos = k_slice.mapv(|v| v.max(0.0));
            let k_neg = k_slice.mapv(|v| v.min(0.0));

            let in_l_ch = in_l.slice(s![ic, .., ..]);
            let in_u_ch = in_u.slice(s![ic, .., ..]);

            // lower += conv(x_l, K+) + conv(x_u, K-)
            let conv_l_pos =
                conv2d_single_f64(&in_l_ch, &k_pos.view(), params.stride, params.padding);
            let conv_u_neg =
                conv2d_single_f64(&in_u_ch, &k_neg.view(), params.stride, params.padding);
            // upper += conv(x_u, K+) + conv(x_l, K-)
            let conv_u_pos =
                conv2d_single_f64(&in_u_ch, &k_pos.view(), params.stride, params.padding);
            let conv_l_neg =
                conv2d_single_f64(&in_l_ch, &k_neg.view(), params.stride, params.padding);

            lower
                .slice_mut(s![oc, .., ..])
                .zip_mut_with(&(&conv_l_pos + &conv_u_neg), |a, &b| *a += b);
            upper
                .slice_mut(s![oc, .., ..])
                .zip_mut_with(&(&conv_u_pos + &conv_l_neg), |a, &b| *a += b);
        }
    }

    BoundedTensor64::new(lower.into_dyn(), upper.into_dyn())
}

/// CROWN backward propagation for Conv2D in f64.
///
/// The backward pass through conv is a transposed convolution:
///   new_A = A_reshaped * conv_transpose(kernel)
///   new_b = A @ bias + b_old
///
/// Each coefficient row is reshaped to (C_out, out_H, out_W), passed through
/// transposed convolution, then flattened back.
pub(crate) fn propagate_conv2d_crown_backward_f64(
    kernel: &Array4<f64>,
    bias: &Array1<f64>,
    bounds: &LinearBounds64,
    params: &Conv2dParams,
) -> Result<LinearBounds64> {
    let m = bounds.num_outputs();
    let c_out = kernel.shape()[0];
    let c_in = kernel.shape()[1];
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];

    let (in_h, in_w) = params.input_hw;
    let (out_h, out_w) = conv2d_output_hw((in_h, in_w), (kh, kw), params.stride, params.padding);
    let n_out_flat = c_out * out_h * out_w;
    let n_in_flat = c_in * in_h * in_w;

    if bounds.num_inputs() != n_out_flat {
        return Err(NyError::shape_mismatch(
            vec![n_out_flat],
            vec![bounds.num_inputs()],
        ));
    }

    let mut new_lower_a = Array2::<f64>::zeros((m, n_in_flat));
    let mut new_upper_a = Array2::<f64>::zeros((m, n_in_flat));
    let mut new_lower_b = bounds.lower_b().clone();
    let mut new_upper_b = bounds.upper_b().clone();

    // For each output row, compute transposed convolution
    for i in 0..m {
        // Accumulate bias: b_new[i] += A[i,:] @ bias_expanded
        let mut bias_dot_l = 0.0f64;
        let mut bias_dot_u = 0.0f64;
        for oc in 0..c_out {
            let bias_val = bias[oc];
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let idx = oc * out_h * out_w + oh * out_w + ow;
                    bias_dot_l += bounds.lower_a()[[i, idx]] * bias_val;
                    bias_dot_u += bounds.upper_a()[[i, idx]] * bias_val;
                }
            }
        }
        new_lower_b[i] += bias_dot_l;
        new_upper_b[i] += bias_dot_u;

        // Transposed convolution: for each input pixel, accumulate contributions
        for ic in 0..c_in {
            for ih in 0..in_h {
                for iw in 0..in_w {
                    let in_idx = ic * in_h * in_w + ih * in_w + iw;
                    let mut sum_l = 0.0f64;
                    let mut sum_u = 0.0f64;

                    // Find which output pixels this input pixel contributes to
                    for khi in 0..kh {
                        for kwi in 0..kw {
                            let oh_with_pad = ih + params.padding.0;
                            let ow_with_pad = iw + params.padding.1;

                            // Check if this kernel position reaches an output pixel
                            if oh_with_pad >= khi
                                && ow_with_pad >= kwi
                                && (oh_with_pad - khi).is_multiple_of(params.stride.0)
                                && (ow_with_pad - kwi).is_multiple_of(params.stride.1)
                            {
                                let oh = (oh_with_pad - khi) / params.stride.0;
                                let ow = (ow_with_pad - kwi) / params.stride.1;
                                if oh < out_h && ow < out_w {
                                    for oc in 0..c_out {
                                        let out_idx = oc * out_h * out_w + oh * out_w + ow;
                                        let k_val = kernel[[oc, ic, khi, kwi]];
                                        sum_l += bounds.lower_a()[[i, out_idx]] * k_val;
                                        sum_u += bounds.upper_a()[[i, out_idx]] * k_val;
                                    }
                                }
                            }
                        }
                    }

                    new_lower_a[[i, in_idx]] = sum_l;
                    new_upper_a[[i, in_idx]] = sum_u;
                }
            }
        }
    }

    LinearBounds64::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
}
