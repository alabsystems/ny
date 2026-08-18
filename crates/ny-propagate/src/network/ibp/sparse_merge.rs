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
    tracked_flat: &[usize],
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

    // POSITIONAL, not value-based. Start from IBP everywhere and keep the CROWN
    // value only at positions the sparse seed actually TRACKED.
    //
    // This used to identify untracked rows by their VALUE:
    //
    //     if *cl == 0.0 && *cu == 0.0 { take IBP }
    //
    // An untracked (stable) neuron gets a zero COEFFICIENT row, but its
    // concretized bound is `[bias, bias]` — only `[0, 0]` when the bias happens
    // to be zero. With any nonzero bias the value test missed the row and the
    // spurious point interval `[b, b]` was published as the neuron's bound in
    // place of IBP's real, wider one. Narrower than the truth is the false-proof
    // direction, and a degenerate interval is as narrow as it gets: downstream
    // it reads as a provably-stable neuron, so the ReLU is never split and the
    // property can verify at the root on a bound that was never established.
    //
    // Value-based identification also cannot distinguish "untracked" from
    // "tracked, and legitimately [0, 0]" — the two are the same bits. Only the
    // seed's index list knows, which is why the sibling
    // `scatter_sparse_crown_into_ibp` in this file has always taken it. This is
    // now the same shape as that function.
    let ibp_lower_slice = ibp_lower
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Non-contiguous IBP lower array".into()))?;
    let ibp_upper_slice = ibp_upper
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Non-contiguous IBP upper array".into()))?;
    let crown_lower_slice = crown_lower
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Non-contiguous CROWN lower array".into()))?;
    let crown_upper_slice = crown_upper
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Non-contiguous CROWN upper array".into()))?;

    let mut merged_lower = ibp_lower.clone();
    let mut merged_upper = ibp_upper.clone();
    {
        let out_lower = merged_lower
            .as_slice_mut()
            .ok_or_else(|| NyError::InternalError("Non-contiguous lower array".into()))?;
        let out_upper = merged_upper
            .as_slice_mut()
            .ok_or_else(|| NyError::InternalError("Non-contiguous upper array".into()))?;
        for &position in tracked_flat {
            // Out-of-range indices mean the seed and the output disagree about
            // shape. Fail closed rather than tightening a neuron by accident.
            if position >= out_lower.len() {
                return Err(NyError::InternalError(format!(
                    "sparse CROWN merge: tracked index {position} outside output of length {}",
                    out_lower.len()
                )));
            }
            // Intersect rather than overwrite: CROWN is not uniformly tighter
            // than IBP, and the merge must never widen a neuron that IBP already
            // bounded better. Both arms enclose, so the intersection does too.
            out_lower[position] = crown_lower_slice[position].max(ibp_lower_slice[position]);
            out_upper[position] = crown_upper_slice[position].min(ibp_upper_slice[position]);
        }
    }

    // `Widen` repair: the intersection above can invert a pair by a few ULPs
    // when CROWN and IBP straddle each other, exactly as in
    // `scatter_sparse_crown_into_ibp`.
    BoundedTensor::new_repaired(merged_lower, merged_upper, RepairStrategy::Widen)
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
