// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! 1D convolution layers for bound propagation.

mod bound;
mod ops;
mod ops_gemm;
mod types;

#[cfg(test)]
mod tests;

pub(crate) use ops::{conv1d_single, conv1d_transpose, conv1d_transpose_forward};
pub(crate) use ops_gemm::{
    conv1d_forward_backward_coeff_f64, conv1d_forward_batched_gemm,
    conv1d_transpose_backward_coeff_f64, conv1d_transpose_batched_gemm,
};
pub use types::{Conv1dLayer, ConvTranspose1dLayer};
