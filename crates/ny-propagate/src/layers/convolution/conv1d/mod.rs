// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! 1D convolution layers for bound propagation.

mod bound;
mod ops;
mod ops_gemm;
mod ops_ibp_cert;
mod ops_ibp_fwd;
mod types;

#[cfg(test)]
mod tests;

pub(crate) use ops::{conv1d_single, conv1d_transpose, conv1d_transpose_forward};
pub(crate) use ops_gemm::{
    conv1d_forward_backward_coeff_f64, conv1d_forward_batched_gemm,
    conv1d_transpose_backward_coeff_f64, conv1d_transpose_batched_gemm,
};
pub(crate) use ops_ibp_cert::{
    conv1d_ibp_certified_forward, conv1d_transpose_ibp_certified_forward,
};
pub(crate) use ops_ibp_fwd::{
    conv1d_ibp_forward_with_deadline, conv1d_transpose_ibp_forward_with_deadline,
};
pub use types::{Conv1dLayer, ConvTranspose1dLayer};
