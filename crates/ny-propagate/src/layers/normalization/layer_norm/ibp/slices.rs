// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared last-axis batch-prefix iteration and slice extraction helpers
//! for LayerNorm IBP.
//!
//! Centralizes the repeated flat-batch decoding and last-axis slice
//! collection previously duplicated across forward-mode, MeanOnly, and
//! Standard interval paths.

use ndarray::ArrayD;
use ny_core::{checked_dim_product, Result};

use super::common::ln_internal_err;

/// Compute the total number of batch elements from the non-norm dimensions.
///
/// `shape[..ndim-1]` is the batch prefix; the last dimension is `norm_size`.
pub(super) fn batch_size(shape: &[usize]) -> Result<usize> {
    let ndim = shape.len();
    checked_dim_product(&shape[..ndim - 1], "LayerNorm IBP batch dimensions")
}

/// Decode a flat batch index into an existing buffer (zero-alloc).
///
/// Writes the prefix indices into `buf[..ndim-1]`. The caller is responsible
/// for ensuring `buf` has at least `shape.len() - 1` elements.
/// Part of #2237: eliminates per-batch-element heap allocations.
pub(super) fn decode_batch_prefix_into(shape: &[usize], batch_idx: usize, buf: &mut [usize]) {
    let prefix_len = shape.len() - 1;
    let mut remaining = batch_idx;
    for d in (0..prefix_len).rev() {
        buf[d] = remaining % shape[d];
        remaining /= shape[d];
    }
}

/// Collect a 1D slice along the last axis at the given batch prefix.
///
/// Uses a stack-allocated index buffer to avoid per-element heap allocations.
/// Part of #2237.
pub(super) fn collect_last_axis_slice(
    arr: &ArrayD<f32>,
    prefix: &[usize],
    norm_size: usize,
) -> Vec<f32> {
    let prefix_len = prefix.len();
    let mut full_idx = [0usize; 8];
    full_idx[..prefix_len].copy_from_slice(prefix);
    (0..norm_size)
        .map(|i| {
            full_idx[prefix_len] = i;
            arr[&full_idx[..=prefix_len]]
        })
        .collect()
}

/// Look up a scalar mean value at the given batch prefix.
///
/// Handles the 0-d / multi-d mean array shape difference: ndarray's
/// `mean_axis` produces a 0-d array for 1-batch inputs and a multi-d
/// array for batched inputs.
pub(super) fn mean_value_at(mean: &ArrayD<f32>, prefix: &[usize], ctx: &str) -> Result<f32> {
    if mean.ndim() == 0 {
        mean.first().copied().ok_or_else(|| ln_internal_err(ctx))
    } else {
        Ok(mean[prefix])
    }
}
