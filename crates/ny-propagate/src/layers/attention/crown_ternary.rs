// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Fused attention CROWN adapter. Production uses direct property-direction
//! contraction with checked arithmetic, oscillation remainders and weighted
//! score-box supports; it never builds dense output-by-input Jacobians.
//! All input bounds have zero bias, with constants and numeric error charged
//! through the shared Nary bias channel. Unsupported masks/shapes refuse to
//! the existing forward/IBP route rather than using an unchecked affine bound.

use super::{AttentionMask, SelfAttentionLayer};
use crate::LinearBounds;
use ndarray::Array1;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

pub(crate) type AttnTernaryResult = (Vec<Option<LinearBounds>>, Array1<f32>, Array1<f32>);

/// Original key tensor storage only. Scores and center/radius caches ALWAYS
/// retain canonical [key, channel] indices; this is no mask or proof profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum KeyStorage {
    Canonical,
    AuthoredKt,
}

impl KeyStorage {
    pub(super) fn column(
        self,
        slice: usize,
        key: usize,
        channel: usize,
        sk: usize,
        dk: usize,
    ) -> usize {
        let base = slice * sk * dk;
        match self {
            Self::Canonical => base + key * dk + channel,
            Self::AuthoredKt => base + channel * sk + key,
        }
    }
}

pub(super) struct SliceDims {
    pub(super) n_slices: usize,
    pub(super) sq: usize,
    pub(super) sk: usize,
    pub(super) dk: usize,
    pub(super) dv: usize,
}

impl SelfAttentionLayer {
    /// Compose a recognized original region on the caller's existing query
    /// ledger and earlier node deadline. This creates no new allowance.
    pub(crate) fn propagate_crown_ternary_with_budget_and_deadline(
        &self,
        node_lb: &LinearBounds,
        q: &BoundedTensor,
        k: &BoundedTensor,
        v: &BoundedTensor,
        budget: &crate::certify::work::WorkBudget,
        deadline: Option<std::time::Instant>,
    ) -> Result<AttnTernaryResult> {
        super::directional::propagate_with_budget_and_deadline(
            self, node_lb, q, k, v, budget, deadline,
        )
    }

    /// Checked original Kt storage [prefix, channel, key]. Only the region
    /// recognizer routes this private adapter; returned K columns bind that
    /// SAME original producer output. No endpoint or coefficient transpose.
    pub(crate) fn propagate_crown_ternary_with_authored_kt_and_budget_and_deadline(
        &self,
        node_lb: &LinearBounds,
        q: &BoundedTensor,
        kt: &BoundedTensor,
        v: &BoundedTensor,
        budget: &crate::certify::work::WorkBudget,
        deadline: Option<std::time::Instant>,
    ) -> Result<AttnTernaryResult> {
        super::directional::propagate_with_authored_kt_and_budget_and_deadline(
            self, node_lb, q, kt, v, budget, deadline,
        )
    }

    #[cfg(test)]
    pub(crate) fn propagate_crown_ternary_with_budget(
        &self,
        node_lb: &LinearBounds,
        q: &BoundedTensor,
        k: &BoundedTensor,
        v: &BoundedTensor,
        budget: &crate::certify::work::WorkBudget,
    ) -> Result<AttnTernaryResult> {
        super::directional::propagate_with_budget(self, node_lb, q, k, v, budget)
    }
    pub(crate) fn propagate_crown_ternary(
        &self,
        node_lb: &LinearBounds,
        q_bounds: &BoundedTensor,
        k_bounds: &BoundedTensor,
        v_bounds: &BoundedTensor,
    ) -> Result<AttnTernaryResult> {
        super::directional::propagate(self, node_lb, q_bounds, k_bounds, v_bounds)
    }
}

pub(super) fn parse_slice_dims(
    layer: &SelfAttentionLayer,
    q: &BoundedTensor,
    k: &BoundedTensor,
    v: &BoundedTensor,
) -> Result<SliceDims> {
    parse_slice_dims_with_key_storage(layer, q, k, v, KeyStorage::Canonical)
}

pub(super) fn parse_slice_dims_with_key_storage(
    layer: &SelfAttentionLayer,
    q: &BoundedTensor,
    k: &BoundedTensor,
    v: &BoundedTensor,
    key_storage: KeyStorage,
) -> Result<SliceDims> {
    let qs = q.shape();
    let ks = k.shape();
    let vs = v.shape();
    if qs.len() < 2 || ks.len() != qs.len() || vs.len() != qs.len() {
        return Err(NyError::UnsupportedOp(format!(
            "SelfAttention ternary CROWN: need matching rank ≥2 Q/K/V, got {qs:?}/{ks:?}/{vs:?}"
        )));
    }
    let nd = qs.len();
    let sq = qs[nd - 2];
    let dk = qs[nd - 1];
    let (sk, dk_k) = match key_storage {
        KeyStorage::Canonical => (ks[nd - 2], ks[nd - 1]),
        KeyStorage::AuthoredKt => (ks[nd - 1], ks[nd - 2]),
    };
    let sk_v = vs[nd - 2];
    let dv = vs[nd - 1];

    if dk_k != dk {
        return Err(NyError::UnsupportedOp(format!(
            "SelfAttention ternary CROWN: Q head_dim {dk} != K head_dim {dk_k}"
        )));
    }
    if sk_v != sk {
        return Err(NyError::UnsupportedOp(format!(
            "SelfAttention ternary CROWN: K key-seq {sk} != V key-seq {sk_v}"
        )));
    }
    // Leading batch/head dims must match across Q/K/V (asymmetric cross-attention
    // with differing leading dims is rejected to the IBP fallback).
    if qs[..nd - 2] != ks[..nd - 2] || qs[..nd - 2] != vs[..nd - 2] {
        return Err(NyError::UnsupportedOp(
            "SelfAttention ternary CROWN: mismatched leading (batch/head) dims".to_string(),
        ));
    }
    let n_slices = ny_core::checked_shape_product(&qs[..nd - 2])
        .ok_or_else(|| NyError::UnsupportedOp("attention slice count overflow".into()))?;

    if n_slices == 0 || sq == 0 || sk == 0 || dk == 0 || dv == 0 {
        return Err(NyError::UnsupportedOp(
            "SelfAttention ternary CROWN: zero dimension".to_string(),
        ));
    }

    // === MASK-SOUNDNESS GATE (false-proof fix) ===
    // The CROWN ternary surrogate AND its T1/T2/T3 error margin both restrict the
    // softmax/V sum to the keys for which `crown_visible(i,k)` is true (the
    // `visible` closure in `build_slice_block`):
    //   Standard:        all k.
    //   Causal:          k <= i  AND (window ⇒ i − k <= w).
    // The SOUND forward / IBP (`CausalSoftmaxLayer::active_range`) instead attends
    //   active_end   = min(i+1, sk)
    //   active_start = window ? max(0, active_end − (w+1)) : 0
    // i.e. keys `[active_start, active_end)`. A CROWN bound is only sound when
    // crown_visible(i,k) PROVABLY equals active_range for EVERY (i,k); otherwise
    // CROWN drops genuinely-visible keys from BOTH the surrogate and the margin and
    // can certify a too-narrow interval that EXCLUDES the reachable true output
    // (a FALSE certificate). The two predicates are provably IDENTICAL except for
    // WINDOWED-causal with sq > sk:
    //
    //   * Standard mask                        — crown=all, forward=all.            MATCH.
    //   * Causal, windowless (window=None)      — both = {k : 0 ≤ k ≤ min(i, sk−1)}. MATCH.
    //   * Causal, windowed, every row i < sk    — both = {k : max(0,i−w) ≤ k
    //                                              ≤ min(i,sk−1)} (active_end=i+1).  MATCH.
    //   * Causal, windowed, some row i ≥ sk     — active_end clamps to sk so forward
    //                                              start = max(0, sk−(w+1)) but crown
    //                                              start = max(0, i−w) > forward start
    //                                              ⇒ crown DROPS keys [forward_start,
    //                                              i−w). MISMATCH (UNSOUND).
    //
    // A row i ≥ sk exists iff sq > sk. So the ONLY non-provably-matched class is
    // WINDOWED-causal cross-attention with sq > sk. For it we RETURN UnsupportedOp
    // so the dispatch falls back to the sound IBP (which uses active_range and
    // correctly encloses the truth) rather than emit a bound we cannot prove
    // encloses it. Standard, any sq==sk, and windowless-causal keep the win.
    //
    // DEFENSE IN DEPTH: today the SOUND forward `CausalSoftmaxLayer` itself only
    // supports seq_q ≤ seq_k, so `propagate_ibp_ternary` (called below) already
    // refuses sq>sk causal with `InvalidSpec` — meaning the original code aborted
    // rather than shipping the false [1,1]. This gate does NOT rely on that
    // incidental refusal: it makes crown explicitly return the GRACEFUL
    // `UnsupportedOp` (IBP fallback) BEFORE the IBP call, and — crucially — if the
    // forward's seq_q≤seq_k restriction is ever relaxed, this gate STILL blocks the
    // false crown bound for the one class where `visible() ≠ active_range`, instead
    // of crown then happily emitting [1,1]. The gate is the durable soundness
    // boundary; the forward constraint is not. (#attn-crown-mask-gate)
    if layer.mask == AttentionMask::Causal && layer.window_size.is_some() && sq > sk {
        return Err(NyError::UnsupportedOp(format!(
            "SelfAttention ternary CROWN: windowed-causal with sq>sk (sq={sq}, sk={sk}, \
             window={:?}) — crown visible() ≠ forward active_range for rows i≥sk; \
             falling back to sound IBP",
            layer.window_size,
        )));
    }

    Ok(SliceDims {
        n_slices,
        sq,
        sk,
        dk,
        dv,
    })
}

#[cfg(test)]
mod tests;
