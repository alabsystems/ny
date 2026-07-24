// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sparse CROWN merging utilities for IBP/CROWN-IBP intersection.

use crate::contiguous_flat_slice;
use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{BoundedTensor, RepairStrategy};

/// Merge sparse CROWN bounds with IBP bounds.
///
/// CROWN backward in sparse mode only computes bounds for unstable neurons.
/// Stable neurons have [0, 0] in the CROWN output (from zero rows in the
/// dense matrix). This function replaces those zero-row entries with the
/// original IBP bounds, which are already tight for stable neurons.
///
/// Part of #2613 Phase 4 step 19
pub(super) fn merge_sparse_crown_with_ibp(
    crown: &BoundedTensor,
    ibp: &BoundedTensor,
) -> Result<BoundedTensor> {
    let crown_lower = crown.lower();
    let crown_upper = crown.upper();
    let ibp_lower = ibp.lower();
    let ibp_upper = ibp.upper();

    if crown_lower.len() != ibp_lower.len() {
        return Err(NyError::ShapeMismatch {
            expected: ibp_lower.shape().to_vec(),
            got: crown_lower.shape().to_vec(),
        });
    }

    // For each output position: use CROWN bounds if they're non-trivial
    // (CROWN lower > -Inf or CROWN upper < Inf), else use IBP bounds.
    // Stable neurons in sparse CROWN get zero rows → concretize → [0, 0],
    // which is trivially narrower than IBP bounds. So we detect sparse-zero
    // rows by checking if both lower and upper are exactly 0.
    let mut merged_lower = crown_lower.clone();
    let mut merged_upper = crown_upper.clone();

    for (i, ((cl, cu), (il, iu))) in crown_lower
        .iter()
        .zip(crown_upper.iter())
        .zip(ibp_lower.iter().zip(ibp_upper.iter()))
        .enumerate()
    {
        // Sparse CROWN zero row: both bounds are exactly 0. Replace with IBP.
        // Also handles the case where CROWN gave wider bounds than IBP by
        // taking the tighter of the two (intersection).
        if *cl == 0.0 && *cu == 0.0 {
            merged_lower
                .as_slice_mut()
                .ok_or_else(|| NyError::InternalError("Non-contiguous lower array".into()))?[i] =
                *il;
            merged_upper
                .as_slice_mut()
                .ok_or_else(|| NyError::InternalError("Non-contiguous upper array".into()))?[i] =
                *iu;
        }
    }

    BoundedTensor::new(merged_lower, merged_upper)
}

/// Find flat indices of unstable neurons in Dense CROWN-IBP output bounds.
///
/// A neuron at flat index i is unstable if `lower[i] < 0 < upper[i]` — the
/// pre-activation bounds cross zero, so the next ReLU relaxation is non-trivial
/// and CROWN backward can potentially tighten bounds beyond IBP.
///
/// Returns `Some(indices)` when the fraction of stable neurons exceeds
/// `min_sparsity` (e.g., 0.9 = >90% stable), making a sparse seed worthwhile.
/// Returns `None` if all neurons are unstable, stability is below threshold,
/// or the bounds are non-contiguous.
///
/// Reference: Same instability criterion as `UnstableIdx::from_ibp_bounds`
/// (bounds/patches.rs) but for flat Dense tensors instead of spatial (C,H,W).
///
/// Part of #3599 Phase 2
pub(super) fn find_unstable_dense_indices(
    lower: &[f32],
    upper: &[f32],
    min_sparsity: f32,
) -> Option<Vec<usize>> {
    let total = lower.len();
    if total == 0 || total != upper.len() {
        return None;
    }

    let mut indices = Vec::new();
    for i in 0..total {
        if lower[i] < 0.0 && upper[i] > 0.0 {
            indices.push(i);
        }
    }

    if indices.is_empty() {
        // All neurons are stable — IBP bounds are already optimal, skip CROWN.
        return None;
    }

    let stable_frac = 1.0 - (indices.len() as f32 / total as f32);
    if stable_frac < min_sparsity {
        return None;
    }

    Some(indices)
}

/// Scatter sparse CROWN bounds (one per unstable neuron) into a full-sized
/// BoundedTensor, using IBP bounds for stable neurons.
///
/// The CROWN backward only computed bounds for `unstable_indices.len()` neurons.
/// For stable neurons, IBP bounds are already optimal (exact ReLU relaxation),
/// so we use IBP directly. The caller will intersect the returned tensor with
/// IBP bounds, which is a no-op for stable positions (intersection of IBP with
/// itself = IBP) and tightens unstable positions.
///
/// Part of #3599 Phase 2
pub(super) fn scatter_sparse_crown_into_ibp(
    crown_lower: &[f32],
    crown_upper: &[f32],
    ibp: &BoundedTensor,
    unstable_indices: &[usize],
    output_shape: &[usize],
) -> Result<BoundedTensor> {
    let ibp_lower = ibp.lower();
    let ibp_upper = ibp.upper();
    let ibp_lower_slice = contiguous_flat_slice(ibp_lower);
    let ibp_upper_slice = contiguous_flat_slice(ibp_upper);

    // Start with IBP bounds for all neurons.
    let mut merged_lower = ibp_lower_slice.to_vec();
    let mut merged_upper = ibp_upper_slice.to_vec();

    // Overwrite unstable positions with CROWN-computed bounds.
    for (sparse_idx, &output_idx) in unstable_indices.iter().enumerate() {
        merged_lower[output_idx] = crown_lower[sparse_idx];
        merged_upper[output_idx] = crown_upper[sparse_idx];
    }

    let lower_arr = ArrayD::from_shape_vec(IxDyn(output_shape), merged_lower)
        .map_err(|e| NyError::InvalidSpec(format!("scatter_sparse_crown reshape: {e}")))?;
    let upper_arr = ArrayD::from_shape_vec(IxDyn(output_shape), merged_upper)
        .map_err(|e| NyError::InvalidSpec(format!("scatter_sparse_crown reshape: {e}")))?;

    // Use Widen repair for minor f32 rounding that may invert bounds.
    BoundedTensor::new_repaired(lower_arr, upper_arr, RepairStrategy::Widen)
}
