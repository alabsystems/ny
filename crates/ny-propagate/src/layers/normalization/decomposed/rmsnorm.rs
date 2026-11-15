// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Decomposed RmsNorm CROWN backward propagation.
//!
//! Propagates CROWN backward through a decomposed RmsNorm chain:
//!   x → x² → mean(x²) → var+eps → sqrt(var+eps) → 1/rms → x*inv_rms → γ·norm

use ndarray::Array1;
use ny_core::Result;
use ny_tensor::BoundedTensor;

use super::DecomposedNormBackwardResult;
use crate::BatchedLinearBounds;

/// A per-row narrowing of the internal `inv_rms = 1/sqrt(mean(x²)+eps)` range,
/// supplied by GenBaB norm branching (#norm-genbab).
///
/// `decomposed_rms_norm_crown_backward` ordinarily derives the `inv_rms`
/// interval from the input box `x_ibp` by sound interval arithmetic. That
/// interval is very wide for wide inputs (e.g. `[1, 316]` for `x ∈ [-1,1]`,
/// `eps=1e-5`), so the fixed-midpoint reciprocal/sqrt relaxations are loose and
/// the row collapses to the fused-RmsNorm IBP in `validate_norm_against_fused_ibp`.
///
/// When a [`InvRmsOverride`] is supplied, the IBP-derived `[inv_rms_l,
/// inv_rms_u]` is **intersected** with `[lo, hi]` (per batch row). Intersection
/// only ever narrows the interval, so the resulting reciprocal/sqrt relaxation
/// is a sound over-approximation of the function on the narrowed set — which is
/// exactly the input subregion `{x : inv_rms(x) ∈ [lo,hi]}` that the requesting
/// GenBaB child subdomain is responsible for. The sibling child covers the
/// complementary range, so the two children union-cover the parent `inv_rms`
/// range (hence the full input box) and the combined BaB verdict is sound.
///
/// SOUNDNESS: the override may only be a *sub-interval* of the IBP-derived
/// range. We intersect (not replace) so a malformed/too-wide override can never
/// widen the certified range; a too-narrow override is still sound because the
/// sibling subdomain reclaims the excluded portion.
///
/// PER-GROUP (#norm-genbab soundness): the window is keyed by normalization
/// group (batch row `b`). A split narrows the `inv_rms` of ONE group; the other
/// groups keep their full IBP range. This is essential: a single shared window
/// would create a join gap — an input `x` whose group 0 has `inv_rms ≤ mid` but
/// group 1 has `inv_rms ≥ mid` would fall in NEITHER sibling child. Splitting
/// one group at a time keeps the union-cover argument valid (the constrained
/// group is partitioned; unconstrained groups span their full range in both
/// children).
#[derive(Debug, Clone)]
pub(crate) struct InvRmsOverride {
    /// Per-group `inv_rms` clamp window, indexed by normalization group (batch
    /// row). `windows[b] = Some((lo, hi))` narrows group `b`; `None` leaves it
    /// at the full IBP range. A length shorter than the batch count leaves the
    /// trailing groups unconstrained.
    pub windows: Vec<Option<(f32, f32)>>,
}

impl InvRmsOverride {
    /// A single-group window: clamp group `group_idx` to `[lo, hi]`, leaving all
    /// other groups at their full IBP range.
    pub(crate) fn single_group(group_idx: usize, lo: f32, hi: f32) -> Self {
        let mut windows = vec![None; group_idx + 1];
        windows[group_idx] = Some((lo, hi));
        Self { windows }
    }
}

/// Decomposed RmsNorm CROWN backward propagation.
///
/// Propagates CROWN backward through a decomposed RmsNorm chain:
///   x → x² → mean(x²) → var+eps → sqrt(var+eps) → 1/rms → x*inv_rms → γ·norm
///
/// Simpler than LayerNorm decomposition: no mean subtraction, no beta offset.
/// Fan-out at x (not d): x feeds both the product path (x*inv_rms) and the
/// variance path (x² → mean(x²) → sqrt → 1/rms).
///
/// Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization," NeurIPS 2019.
/// alpha-beta-CROWN decomposes via `_split_complex` (normalization.py:303-331).
pub(crate) fn decomposed_rms_norm_crown_backward(
    a_output: &BatchedLinearBounds,
    ny: &Array1<f32>,
    eps: f32,
    x_ibp: &BoundedTensor,
) -> Result<DecomposedNormBackwardResult> {
    decomposed_rms_norm_crown_backward_with_override(a_output, ny, eps, x_ibp, None)
}

/// [`decomposed_rms_norm_crown_backward`] with an optional per-row `inv_rms`
/// range override from GenBaB norm branching (#norm-genbab).
///
/// `inv_rms_override`, when `Some`, applies the SAME narrowing to every batch
/// row (the GenBaB split point is one scalar shared across the normalization
/// groups of a node). It is intersected with the IBP-derived range; see
/// [`InvRmsOverride`] for the soundness argument.
pub(crate) fn decomposed_rms_norm_crown_backward_with_override(
    a_output: &BatchedLinearBounds,
    ny: &Array1<f32>,
    eps: f32,
    x_ibp: &BoundedTensor,
    inv_rms_override: Option<InvRmsOverride>,
) -> Result<DecomposedNormBackwardResult> {
    crate::layers::normalization::radial::baseline::checked_baseline(
        a_output,
        ny,
        None,
        eps,
        x_ibp,
        false,
        inv_rms_override.as_ref().map(|ov| ov.windows.as_slice()),
    )
}
