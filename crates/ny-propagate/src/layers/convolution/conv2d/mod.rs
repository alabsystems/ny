// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! 2D convolution layers for bound propagation.

use ndarray::{Array2, ArrayD};
use ny_core::{GemmEngine, Result};

mod bound;
mod bound_patches;
mod bound_transpose;
mod bound_transpose_patches;
mod ops;
mod ops_gemm;
mod ops_ibp_fwd;
mod ops_ibp_gemm;
mod ops_transpose_fwd;
mod ops_transpose_gemm;
mod types;

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use ops::conv2d_single_grouped;
#[cfg(test)]
pub(crate) use ops::conv2d_transpose;
pub(crate) use ops::{conv2d_single, conv2d_transpose_grouped};
pub(crate) use ops_gemm::conv2d_forward_backward_coeff_f64;
pub(crate) use ops_gemm::conv2d_forward_backward_coeff_f64_pair_with_deadline;
pub(crate) use ops_gemm::conv2d_forward_backward_coeff_f64_with_deadline;
pub(crate) use ops_gemm::conv2d_forward_batched_gemm;
pub(crate) use ops_transpose_fwd::conv2d_transpose_forward;
pub(crate) use ops_transpose_gemm::conv2d_transpose_backward_coeff_f64;
pub(crate) use ops_transpose_gemm::conv2d_transpose_batched_gemm_grouped_with_deadline;
pub(crate) use ops_transpose_gemm::conv2d_transpose_pair_batched_gemm_grouped_with_deadline;
pub use types::{Conv2dLayer, ConvTranspose2dLayer};

/// Batched conv2d transpose via GEMM for CROWN backward pass with groups support.
///
/// Replaces N calls to `conv2d_transpose` with GEMM + col2im scatter.
/// For groups=1, uses a single GEMM. For groups>1, uses per-group GEMMs.
///
/// Kernel shape: (out_c, in_c_per_group, kh, kw) where in_c_per_group = total_in_c / groups.
///
/// Reference: alpha-beta-CROWN `auto_LiRPA/operators/convolution.py:85-115`.
/// Design doc: `designs/2026-03-06-conv-crown-backward-gemm.md` (#3382).
// Justification: Conv2d backward GEMM needs all conv parameters (kernel, stride,
// padding, sizes, channels, groups) plus the engine; grouping into a struct would lose
// clarity for this internal function.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_transpose_batched_gemm(
    a_coefficients: &Array2<f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    output_size: (usize, usize),
    grad_size: (usize, usize),
    out_channels: usize,
    engine: Option<&dyn GemmEngine>,
) -> Result<Array2<f32>> {
    conv2d_transpose_batched_gemm_grouped(
        a_coefficients,
        kernel,
        stride,
        padding,
        dilation,
        output_size,
        grad_size,
        out_channels,
        1,
        1,
        engine,
    )
}

/// Batched conv2d transpose via GEMM with explicit groups parameter.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_transpose_batched_gemm_grouped(
    a_coefficients: &Array2<f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    output_size: (usize, usize),
    grad_size: (usize, usize),
    out_channels: usize,
    groups: usize,
    resident_result_buffers: usize,
    engine: Option<&dyn GemmEngine>,
) -> Result<Array2<f32>> {
    conv2d_transpose_batched_gemm_grouped_with_deadline(
        a_coefficients,
        kernel,
        stride,
        padding,
        dilation,
        output_size,
        grad_size,
        out_channels,
        groups,
        resident_result_buffers,
        engine,
        None,
    )
}
