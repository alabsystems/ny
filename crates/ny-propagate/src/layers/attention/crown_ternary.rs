// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound, memory-bounded ternary CROWN backward for fused self-attention.
//!
//! For `Y = softmax(scale · Q Kᵀ) @ V`, this produces three `LinearBounds`
//! (one per input Q, K, V) plus a shared bias channel, in the
//! [`BackwardDispatchResult::Nary`](crate::network::backward_dispatch::BackwardDispatchResult)
//! shape. It is the fused alternative to the de-fused MatMul+Softmax+MatMul
//! sub-graph; it is used by `nn-verify` for large attention
//! (`num_heads·seq² > de-fuse budget`, e.g. table_transformer `S=64`,
//! DETR-medium `S=32`) where the de-fused per-head CROWN backward materializes
//! `O(num_heads · seq²)` McCormick / softmax-Jacobian coefficient tensors
//! simultaneously and OOMs.
//!
//! # Method — center-point linearization + directional O(radius²) margin
//!
//! Linearize the attention map about the box center (a single mean-value
//! surrogate `Ŷ`), then add a DIRECTIONAL bias margin `[−m_below, +m_above]`
//! derived from a SOUND, closed-form enclosure of ONLY the relaxation error
//! `e(x) = Y_true(x) − Ŷ(x)`. The error is SECOND-ORDER in the box radius
//! (softmax curvature + the two bilinear `P·V` and `Q·Kᵀ` remainders), so the
//! margin shrinks like `O(radius²)` while the simplex-aware IBP envelope shrinks
//! only like `O(radius)`. On the small/medium robustness boxes verification
//! actually uses the margin is therefore strictly inside the IBP gap and the
//! concretized bound is strictly TIGHTER than IBP; on large boxes the margin is
//! CLAMPED to the IBP gap, so the result ties IBP and is never worse.
//!
//! This replaces the original SYMMETRIC, IBP-sized margin
//! (`m_below = ŷ_hi − ibp_lo`, `m_above = ibp_hi − ŷ_lo`, sized to the full
//! `O(radius)` IBP envelope) which only ever TIED IBP after the framework's
//! CROWN∩IBP intersection. The surrogate `Ŷ` and its exact center-point
//! Jacobians are UNCHANGED; only the constant/bias channel changes (the
//! lower/upper coefficient matrices remain the single affine slope — sound
//! because a single affine `Ŷ` with the error enclosed in `[−m_below, +m_above]`
//! gives `Ŷ − m_below ≤ Y ≤ Ŷ + m_above` pointwise).
//!
//! Concretely, per `(batch, head)` slice with `S` query/key positions and head
//! dims `d_k` (Q/K) and `d_v` (V):
//!
//! 1. Centers `Qc, Kc, Vc` (box midpoints). Forward: `Pc = softmax(scale·Qc Kcᵀ)`
//!    (row `i` over keys), `Yc = Pc @ Vc`.
//! 2. Analytic Jacobians at the center of `Y[i, j]`:
//!      - `∂Y[i,j]/∂V[k,j]   = Pc[i,k]`              (V is linear in `Y`)
//!      - `∂Y[i,j]/∂(score[i,k]) = Pc[i,k]·(Vc[k,j] − Yc[i,j])`
//!        via the softmax Jacobian `∂P[i,k]/∂score[i,m] = P[i,k](δ_{km} − P[i,m])`,
//!        and `score[i,k] = scale · Σ_d Q[i,d] K[k,d]` gives
//!      - `∂Y[i,j]/∂Q[i,d]  = scale · Σ_k g[i,k,j] · Kc[k,d]`
//!      - `∂Y[i,j]/∂K[k,d]  = scale · g[i,k,j] · Qc[i,d]`,
//!        where `g[i,k,j] = Pc[i,k]·(Vc[k,j] − Yc[i,j])`.
//! 3. Affine surrogate
//!    `Ŷ(Q,K,V) = J_Q·(Q−Qc) + J_K·(K−Kc) + J_V·(V−Vc) + Yc`.
//! 4. DIRECTIONAL O(radius²) margin from a sound error enclosure. Decompose the
//!    relaxation error EXACTLY (no shared-ξ assumption) as `e = T1 + T2 + T3`
//!    using `P = Pc + dP`, `V = Vc + dV`, `s = sc + ds` with `Σ_k dP[k] = 0`:
//!      - `T1 = Σ_k (Vc[k,j] − Yc)·R_P[k]` — softmax 2nd-order remainder,
//!        `R_P[k] = P[k] − Pc[k] − Σ_m J_P[k,m]·ds[m]`. Bounded by
//!        `|R_P[k]| ≤ Ph[k]·(Σ_m|ds[m]|)²` (sound softmax-Hessian bound, `Ph`
//!        from the per-row softmax-of-box range). The recenter by `Yc` is legal
//!        because `Σ_k R_P[k] = 0`, and shrinks the V weighting to `|Vc−Yc|`.
//!      - `T2 = Σ_k dP[k]·dV[k,j]` — the dropped `P·V` bilinear remainder,
//!        bounded by `Σ_k max(|Pl[k]−Pc[k]|,|Ph[k]−Pc[k]|)·radV[k,j]`.
//!      - `T3 = Σ_m g[m]·(ds[m] − ds_lin[m])` — the `Q·Kᵀ` bilinear remainder,
//!        `ds[m] − ds_lin[m] = scale·Σ_d dQ[i,d]·dK[m,d]`, bounded by
//!        `|scale|·Σ_d radQ[i,d]·radK[m,d]`, weighted by `|g[m]|`.
//!
//!    All three are `O(radius²)`. Let `E = |T1| + |T2| + |T3|` (outward-rounded).
//!    Set `m_below = min(E, max(0, ŷ_hi − ref_lo))`,
//!    `m_above = min(E, max(0, ref_hi − ŷ_lo))`, where `[ref_lo, ref_hi]` is a
//!    SOUND output envelope — the WIDER of the external simplex-aware IBP and a
//!    self-computed `box ∩ simplex` water-filling envelope built from THIS node's
//!    own sound per-key softmax range `[p_lo, p_hi]` and the V box. Widening the
//!    reference to the self-envelope guarantees soundness even when the external
//!    softmax-IBP is narrow-unsound in the large-score-gap / underflow regime
//!    (a separate epsilon-vs-underflow issue), while on normal boxes the two
//!    references agree so the clamp still gives never-worse-than-IBP and the
//!    end-to-end win is preserved. Lower bias pushed DOWN by `m_below`, upper UP by
//!    `m_above` (sign-aware through the upstream coefficients). See the proof in
//!    `build_slice_block`.
//!
//! # Soundness (enclosure argument)
//!
//! The surrogate `Ŷ` is a single affine function with EXACT (interval) Jacobian
//! coefficients. The error decomposition `e = T1 + T2 + T3` is an algebraic
//! IDENTITY (it telescopes the true output `Σ_k P[k]V[k,j]` against the
//! center-point linearization), so `|e(x)| ≤ E` for every `x` in the box once
//! each `Tι` is enclosed by its sound interval. Hence, pointwise,
//! `Ŷ(x) − m_below ≤ Ŷ(x) − E ≤ Y_true(x) ≤ Ŷ(x) + E ≤ Ŷ(x) + m_above` for
//! every `x` (the `m_below,m_above ≤ E` only TIGHTENS via the IBP clamp, since
//! `[ibp_lo,ibp_hi] ∋ Y_true(x)` and the clamp is an UPPER cap on the margin —
//! it can only reduce the widening, never under-approximate). The lower/upper
//! coefficient matrices are the SAME affine slope; all asymmetry lives in the
//! bias, sound because a single affine `Ŷ` is its own lower/upper bound up to
//! `±E`. Every softmax-Hessian / bilinear bound is a PROVEN over-approximation
//! computed in f64 with OUTWARD rounding (the margin magnitude is rounded UP);
//! a non-finite intermediate degrades the row to `[−∞, +∞]`. The framework
//! later intersects the concretized output with the sound IBP forward bound
//! (`tighten_crown_output_with_provenance`), so the shipped result is always
//! tighter-or-equal to IBP.
//!
//! # Memory boundedness
//!
//! Work proceeds one `(batch, head)` slice at a time. Per slice the only `O(S²)`
//! allocations are the dense per-slice attention Jacobian blocks reused across
//! slices; the full-tensor (`num_heads · seq²`) materialization that OOMs the
//! de-fused path is never formed. The OUTPUT matrices are `(d_spec, S·d)` per
//! input — inherent to CROWN at this node and independent of method. Peak extra
//! RSS is `O(S² + S·d_v + S·d_k)` per slice, i.e. far below the de-fused blowup.

use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::{AttentionMask, SelfAttentionLayer};
use crate::LinearBounds;

/// Return type for the attention ternary CROWN backward:
/// `(per-input LinearBounds [Q, K, V], bias_lower, bias_upper)`.
/// Mirrors `normalization::adain::crown_ternary::TernaryCrownResult`; the
/// `Nary` wrapping happens in the dispatch layer.
pub(crate) type AttnTernaryResult = (Vec<Option<LinearBounds>>, Array1<f32>, Array1<f32>);

/// Per-`(batch, head)` slice dimensions parsed from Q/K/V shapes.
struct SliceDims {
    /// Number of leading `(batch·head)` slices.
    n_slices: usize,
    /// Query sequence length (rows of `Y` per slice).
    sq: usize,
    /// Key/value sequence length (softmax width per slice).
    sk: usize,
    /// Q/K head dimension.
    dk: usize,
    /// V head / output dimension.
    dv: usize,
}

impl SelfAttentionLayer {
    /// Ternary CROWN backward for fused self-attention.
    ///
    /// Returns `(per-input LinearBounds [Q, K, V], bias_lower, bias_upper)`.
    /// Each `LinearBounds` carries zero bias; all bias flows through the shared
    /// channel (the `Nary` contract). On any unsupported shape / non-finite
    /// intermediate this returns `Err(UnsupportedOp)` so the caller cleanly
    /// falls back to IBP (never an unsound bound).
    pub(crate) fn propagate_crown_ternary(
        &self,
        node_lb: &LinearBounds,
        q_bounds: &BoundedTensor,
        k_bounds: &BoundedTensor,
        v_bounds: &BoundedTensor,
    ) -> Result<AttnTernaryResult> {
        let dims = parse_slice_dims(self, q_bounds, k_bounds, v_bounds)?;
        let n_out = node_lb.num_outputs();
        let n_y = dims.n_slices * dims.sq * dims.dv; // flattened Y size
        let n_q = dims.n_slices * dims.sq * dims.dk;
        let n_k = dims.n_slices * dims.sk * dims.dk;
        let n_v = dims.n_slices * dims.sk * dims.dv;

        if node_lb.num_inputs() != n_y {
            return Err(NyError::UnsupportedOp(format!(
                "SelfAttention ternary CROWN: node_lb num_inputs {} != flattened Y size {}",
                node_lb.num_inputs(),
                n_y
            )));
        }

        let scale = self.resolve_scale(q_bounds)?;

        // Flattened, contiguous input box slices. Non-contiguous inputs are
        // rejected to the IBP fallback (sound, just looser).
        let (ql, qu) = lower_upper_flat(q_bounds)?;
        let (kl, ku) = lower_upper_flat(k_bounds)?;
        let (vl, vu) = lower_upper_flat(v_bounds)?;

        // Sound fused IBP bounds (simplex-aware) used to certify the margins.
        let ibp = self.propagate_ibp_ternary(q_bounds, k_bounds, v_bounds)?;
        let ibp_flat = ibp.flatten();
        let ibp_lo = contiguous(ibp_flat.lower())?;
        let ibp_hi = contiguous(ibp_flat.upper())?;
        if ibp_lo.len() != n_y {
            return Err(NyError::UnsupportedOp(format!(
                "SelfAttention ternary CROWN: IBP size {} != flattened Y size {}",
                ibp_lo.len(),
                n_y
            )));
        }

        // Output coefficient matrices (inherent O(n_out · n_in) — the Nary result
        // shape). These are the ONLY full-size allocations; the per-slice Jacobian
        // blocks below are transient and reused, so peak extra memory stays
        // O(per-slice Jacobian) = O(sq·dv · sk·dk), never O(n_y · n_in).
        let mut acc = OutputAccumulator::new(node_lb, n_out, n_q, n_k, n_v);

        // Stream over (batch·head) slices. The attention Jacobian is
        // block-diagonal across slices (head h's output depends only on head h's
        // Q/K/V), so each slice is composed independently and its Jacobian block
        // discarded before the next slice — this is the memory-bounded core.
        for slice in 0..dims.n_slices {
            let block = build_slice_block(
                self,
                slice,
                &dims,
                scale,
                &ql,
                &qu,
                &kl,
                &ku,
                &vl,
                &vu,
                ibp_lo.as_slice(),
                ibp_hi.as_slice(),
            )?;
            acc.compose_slice(node_lb, slice, &dims, &block);
        }

        acc.finalize(n_out, n_q, n_k, n_v)
    }
}

/// Parse the per-slice attention dimensions from Q/K/V shapes.
///
/// Supports 2-D `[S, d]` (single slice) and ≥3-D `[..., S, d]` (leading dims
/// folded into `n_slices`). Q and K must share `(S_kv? , d_k)`; V supplies
/// `(S_kv, d_v)`. Standard, windowless-causal, and windowed-causal masking with
/// `sq <= sk` are supported (the crown `visible(i,k)` predicate provably equals
/// the sound forward `active_range`). Windowed-causal with `sq > sk` (the only
/// class where the two predicates differ — see the mask-soundness gate below)
/// returns `UnsupportedOp` so the framework falls back to the sound IBP.
fn parse_slice_dims(
    layer: &SelfAttentionLayer,
    q: &BoundedTensor,
    k: &BoundedTensor,
    v: &BoundedTensor,
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
    let sk = ks[nd - 2];
    let dk_k = ks[nd - 1];
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
    let n_slices: usize = qs[..nd - 2].iter().product::<usize>().max(1);

    if sq == 0 || sk == 0 || dk == 0 || dv == 0 {
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

/// One `(batch, head)` slice's surrogate, stored as DENSE per-slice Jacobian
/// blocks — NOT the full `(n_y × n_in)` matrices. Each block is reused across
/// slices, so peak memory is `O(sq·dv · sk·dk)`, not `O(n_slices²·…)`.
struct SliceBlock {
    /// `jq[(i*dv+j), (i'*dk+d)]` — `∂Y_slice[i,j]/∂Q_slice[i',d]`. Only `i'==i`
    /// rows are nonzero (a query attends with its own row), but we keep the full
    /// `(sq·dv × sq·dk)` block for a simple dense compose; it is small per slice.
    jq: Array2<f32>,
    /// `jk[(i*dv+j), (k*dk+d)]`.
    jk: Array2<f32>,
    /// `jv[(i*dv+j), (k*dv+j')]`.
    jv: Array2<f32>,
    /// Margin-adjusted constants per slice output element `(sq·dv,)`.
    const_lo: Array1<f64>,
    const_hi: Array1<f64>,
}

#[allow(clippy::too_many_arguments)]
fn build_slice_block(
    layer: &SelfAttentionLayer,
    slice: usize,
    dims: &SliceDims,
    scale: f32,
    ql: &[f32],
    qu: &[f32],
    kl: &[f32],
    ku: &[f32],
    vl: &[f32],
    vu: &[f32],
    ibp_lo: &[f32],
    ibp_hi: &[f32],
) -> Result<SliceBlock> {
    let SliceDims { sq, sk, dk, dv, .. } = *dims;
    let scale_f64 = scale as f64;

    // Slice base offsets into the flattened input/output buffers.
    let q_base = slice * sq * dk;
    let k_base = slice * sk * dk;
    let v_base = slice * sk * dv;
    let y_base = slice * sq * dv;

    let qc = center_slice(ql, qu, q_base, sq * dk);
    let kc = center_slice(kl, ku, k_base, sk * dk);
    let vc = center_slice(vl, vu, v_base, sk * dv);
    // Box radii (half-width) for the directional O(radius²) error margin.
    let qr = radius_slice(ql, qu, q_base, sq * dk);
    let kr = radius_slice(kl, ku, k_base, sk * dk);
    let vr = radius_slice(vl, vu, v_base, sk * dv);

    let yn = sq * dv;
    let mut jq = Array2::<f32>::zeros((yn, sq * dk));
    let mut jk = Array2::<f32>::zeros((yn, sk * dk));
    let mut jv = Array2::<f32>::zeros((yn, sk * dv));
    let mut const_lo = Array1::<f64>::zeros(yn);
    let mut const_hi = Array1::<f64>::zeros(yn);

    let mask = layer.mask;
    let window = layer.window_size;
    // SOUNDNESS: this `visible(i,k)` set must equal the forward `active_range`
    // (`CausalSoftmaxLayer::active_range`) for every (i,k); otherwise the surrogate
    // and the error margin drop genuinely-visible keys and the certificate is false.
    // `parse_slice_dims` PROVES the equality for the configs that reach here
    // (Standard; windowless-causal; windowed-causal with sq<=sk) and gates the one
    // non-matching class (windowed-causal, sq>sk) to the IBP fallback. Do NOT widen
    // the supported classes without re-proving `visible == active_range` there.
    let visible = |i: usize, k: usize| -> bool {
        match mask {
            AttentionMask::Standard => true,
            AttentionMask::Causal => k <= i && window.map(|w| i - k <= w).unwrap_or(true),
        }
    };

    let mut scores = vec![0.0f64; sk];
    let mut pc = vec![0.0f64; sk];
    // Per-row directional-margin scratch (reused across rows):
    //  - `ds_abs[k]` = sound bound on `|s[k] − sc[k]|` over the box (for T1, T3),
    //  - `p_hi[k]` = sound upper bound on `P[k]` over the box (for T1),
    //  - `dp_mag[k]` = sound bound on `|P[k] − Pc[k]|` over the box (for T2),
    //  - `score_remainder[m]` = sound bound on `|ds[m] − ds_lin[m]|` (for T3).
    let mut ds_abs = vec![0.0f64; sk];
    let mut p_hi = vec![0.0f64; sk];
    let mut p_lo = vec![0.0f64; sk];
    let mut dp_mag = vec![0.0f64; sk];
    let mut score_remainder = vec![0.0f64; sk];
    for i in 0..sq {
        // scores[k] = scale · Σ_d Qc[i,d] Kc[k,d]
        let mut any_visible = false;
        let mut max_score = f64::NEG_INFINITY;
        for k in 0..sk {
            if !visible(i, k) {
                scores[k] = f64::NEG_INFINITY;
                continue;
            }
            any_visible = true;
            let mut dot = 0.0f64;
            for d in 0..dk {
                dot += qc[i * dk + d] * kc[k * dk + d];
            }
            let s = scale_f64 * dot;
            scores[k] = s;
            if s > max_score {
                max_score = s;
            }
        }
        if !any_visible || !max_score.is_finite() {
            return Err(NyError::NumericalInstability(
                "SelfAttention ternary CROWN: no visible key or non-finite score".to_string(),
            ));
        }
        let mut sum_exp = 0.0f64;
        for k in 0..sk {
            if scores[k].is_finite() {
                let e = (scores[k] - max_score).exp();
                pc[k] = e;
                sum_exp += e;
            } else {
                pc[k] = 0.0;
            }
        }
        if sum_exp <= 0.0 || !sum_exp.is_finite() {
            return Err(NyError::NumericalInstability(
                "SelfAttention ternary CROWN: softmax sum non-finite".to_string(),
            ));
        }
        for k in 0..sk {
            pc[k] /= sum_exp;
        }

        // === Per-row directional-margin precompute (sound, O(radius²)) ===
        // Compute, over the WHOLE box (not the center): the per-key score
        // deviation bound `|s[k] − sc[k]|`, the score bilinear remainder
        // `|ds[m] − ds_lin[m]| = |scale|·Σ_d radQ·radK`, and the per-key softmax
        // RANGE `[p_lo, p_hi]` (sound, monotone softmax-of-box). These feed the
        // T1/T2/T3 enclosures below. All are independent of the V output dim `j`.
        let mut sum_ds_abs = 0.0f64; // Σ_m |ds[m]|_max  (for the T1 (Σ|ds|)² factor)
        for m in 0..sk {
            // SOUNDNESS (false-proof fix): gate the error margin on the ACTUAL
            // causal/window mask — NOT on `pc[m] == 0.0`. A genuinely MASKED key is
            // P[m]=0 for EVERY input in the box (sound to drop). But a VISIBLE key
            // whose CENTER softmax prob underflowed to exactly 0.0 in f64 (its
            // `score[m] − max_score < ≈ −745`) is NOT masked: over the box its score
            // can RISE until it WINS the softmax (P[m] → 1), so it MUST contribute to
            // the error margin. Using `pc[m]==0.0` here dropped such keys from
            // T1/T2/T3 and produced a too-narrow, FALSE certificate (see the
            // `false_proof_underflowed_visible_key_*` regression tests). The surrogate
            // Jacobian may still treat `pc[m]≈0` keys as ≈0 (their center derivative
            // is ∝ pc[m]); only the MARGIN must include every visible key.
            if !visible(i, m) {
                // Masked / non-visible key: it never contributes to any output.
                ds_abs[m] = 0.0;
                score_remainder[m] = 0.0;
                continue;
            }
            // Interval score s[m] = scale·⟨Q[i,:],K[m,:]⟩ around the center.
            // Linear part ds_lin = scale·Σ_d (Qc·dK + Kc·dQ); the deviation
            // |ds[m]| ≤ |scale|·Σ_d (|Qc[i,d]|·radK + |Kc[m,d]|·radQ + radQ·radK).
            // The pure-bilinear remainder is |scale|·Σ_d radQ·radK.
            let mut dev = 0.0f64;
            let mut rem = 0.0f64;
            for d in 0..dk {
                let qcd = qc[i * dk + d].abs();
                let kcd = kc[m * dk + d].abs();
                let rq = qr[i * dk + d];
                let rk = kr[m * dk + d];
                dev += qcd * rk + kcd * rq + rq * rk;
                rem += rq * rk;
            }
            let sc_abs = scale_f64.abs();
            ds_abs[m] = sc_abs * dev;
            score_remainder[m] = sc_abs * rem;
            sum_ds_abs += ds_abs[m];
        }

        // Per-key softmax RANGE over the box: P[m] is increasing in s[m] and
        // decreasing in every s[m'≠m]. With sc[k] the center score and ds_abs the
        // sound score half-deviation, su[k]=sc[k]+ds_abs[k], sl[k]=sc[k]−ds_abs[k]
        // (visible keys), so a sound per-key envelope is the standard softmax
        // monotone bound, computed with a shared max-shift for stability:
        //   p_hi[k] = e^{su[k]} / (e^{su[k]} + Σ_{m≠k} e^{sl[m]})
        //   p_lo[k] = e^{sl[k]} / (e^{sl[k]} + Σ_{m≠k} e^{su[m]})
        // We only need p_hi[k] (T1) and dp_mag[k]=max(|p_lo−Pc|,|p_hi−Pc|) (T2).
        {
            // Per-key upper/lower softmax probabilities computed in a numerically
            // ROBUST, sound way. We need, for each key k:
            //   p_hi[k] = e^{su[k]} / (e^{su[k]} + Σ_{m≠k} e^{sl[m]})   (k rises, rest fall)
            //   p_lo[k] = e^{sl[k]} / (e^{sl[k]} + Σ_{m≠k} e^{su[m]})   (k falls, rest rise)
            // where su[m]=sc[m]+ds_abs[m], sl[m]=sc[m]−ds_abs[m] for VISIBLE keys.
            //
            // OVERFLOW: with the underflow fix a visible key may have a HUGE score
            // range (su[m]−max_score ≫ 0 when its center underflowed but its box
            // score can rise), so a single shared `exp(·−max_score)` overflows to
            // +inf and the ratios become inf/inf = NaN. We instead form each ratio by
            // shifting the exponents by THAT ratio's own per-term max, so every
            // exponent argument is ≤ 0 (e^x ∈ (0,1], no overflow). Masked keys are
            // excluded (the `visible` gate), as P[m]=0 for every input.
            for k in 0..sk {
                if !visible(i, k) {
                    p_hi[k] = 0.0;
                    p_lo[k] = 0.0;
                    dp_mag[k] = 0.0;
                    continue;
                }
                // --- p_hi[k]: numerator su[k], others at sl[m] ---
                let num_hi_exp = scores[k] + ds_abs[k];
                let mut max_hi = num_hi_exp;
                for m in 0..sk {
                    if m == k || !visible(i, m) {
                        continue;
                    }
                    let e = scores[m] - ds_abs[m];
                    if e > max_hi {
                        max_hi = e;
                    }
                }
                let p_h = if max_hi.is_finite() {
                    let num = (num_hi_exp - max_hi).exp();
                    let mut den = num;
                    for m in 0..sk {
                        if m == k || !visible(i, m) {
                            continue;
                        }
                        den += (scores[m] - ds_abs[m] - max_hi).exp();
                    }
                    if den > 0.0 && den.is_finite() {
                        (num / den).clamp(0.0, 1.0)
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };
                // --- p_lo[k]: numerator sl[k], others at su[m] ---
                let num_lo_exp = scores[k] - ds_abs[k];
                let mut max_lo = num_lo_exp;
                for m in 0..sk {
                    if m == k || !visible(i, m) {
                        continue;
                    }
                    let e = scores[m] + ds_abs[m];
                    if e > max_lo {
                        max_lo = e;
                    }
                }
                let p_l = if max_lo.is_finite() {
                    let num = (num_lo_exp - max_lo).exp();
                    let mut den = num;
                    for m in 0..sk {
                        if m == k || !visible(i, m) {
                            continue;
                        }
                        den += (scores[m] + ds_abs[m] - max_lo).exp();
                    }
                    if den > 0.0 && den.is_finite() {
                        (num / den).clamp(0.0, 1.0)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                // Outward (sound) widening: p_hi up, p_lo down.
                let p_h = (p_h + 1e-7).min(1.0);
                let p_l = (p_l - 1e-7).max(0.0);
                p_hi[k] = p_h;
                p_lo[k] = p_l;
                // dp_mag must remain a sound bound on |P[k]−Pc[k]|; Pc[k] (which may
                // have underflowed to 0) lies in [p_l, p_h] so this max-deviation is
                // a valid over-bound either way.
                let dm = (pc[k] - p_l).abs().max((p_h - pc[k]).abs());
                dp_mag[k] = dm;
            }
        }

        for j in 0..dv {
            let yrow = i * dv + j; // local slice output index
            let y_glob = y_base + yrow; // global Y index (for IBP lookup)

            let mut yc_ij = 0.0f64;
            for k in 0..sk {
                yc_ij += pc[k] * vc[k * dv + j];
            }
            if !yc_ij.is_finite() {
                return Err(NyError::NumericalInstability(
                    "SelfAttention ternary CROWN: Yc non-finite".to_string(),
                ));
            }

            // ∂Y[i,j]/∂V[k,j] = Pc[i,k]
            for k in 0..sk {
                if pc[k] != 0.0 {
                    jv[[yrow, k * dv + j]] = pc[k] as f32;
                }
            }
            // ∂Y[i,j]/∂Q[i,d] = scale · Σ_k g·Kc[k,d];  ∂Y[i,j]/∂K[k,d] = scale·g·Qc[i,d]
            // with g[i,k,j] = Pc[i,k]·(Vc[k,j] − Yc[i,j]).
            for d in 0..dk {
                let mut q_grad = 0.0f64;
                for k in 0..sk {
                    if pc[k] == 0.0 {
                        continue;
                    }
                    let g = pc[k] * (vc[k * dv + j] - yc_ij);
                    q_grad += g * kc[k * dk + d];
                }
                let q_coeff = scale_f64 * q_grad;
                if q_coeff != 0.0 {
                    jq[[yrow, i * dk + d]] = q_coeff as f32;
                }
            }
            for k in 0..sk {
                if pc[k] == 0.0 {
                    continue;
                }
                let g = pc[k] * (vc[k * dv + j] - yc_ij);
                let kf = scale_f64 * g;
                for d in 0..dk {
                    let k_coeff = kf * qc[i * dk + d];
                    if k_coeff != 0.0 {
                        jk[[yrow, k * dk + d]] = k_coeff as f32;
                    }
                }
            }

            // Constant c = Yc − J·center, and box-concretization of the surrogate.
            // `c_acc` = J·center (subtracted to form the constant); `lo_acc`/`hi_acc`
            // = interval-arithmetic box-concretization of the J·box linear part.
            let mut c_acc = 0.0f64;
            let mut lo_acc = 0.0f64;
            let mut hi_acc = 0.0f64;
            for d in 0..dk {
                let coeff = jq[[yrow, i * dk + d]] as f64;
                c_acc += coeff * qc[i * dk + d];
                let (lo, hi) = (
                    ql[q_base + i * dk + d] as f64,
                    qu[q_base + i * dk + d] as f64,
                );
                if coeff >= 0.0 {
                    lo_acc += coeff * lo;
                    hi_acc += coeff * hi;
                } else {
                    lo_acc += coeff * hi;
                    hi_acc += coeff * lo;
                }
            }
            // K contribution.
            for k in 0..sk {
                for d in 0..dk {
                    let coeff = jk[[yrow, k * dk + d]] as f64;
                    if coeff == 0.0 {
                        continue;
                    }
                    c_acc += coeff * kc[k * dk + d];
                    let (lo, hi) = (
                        kl[k_base + k * dk + d] as f64,
                        ku[k_base + k * dk + d] as f64,
                    );
                    if coeff >= 0.0 {
                        lo_acc += coeff * lo;
                        hi_acc += coeff * hi;
                    } else {
                        lo_acc += coeff * hi;
                        hi_acc += coeff * lo;
                    }
                }
            }
            // V contribution.
            for k in 0..sk {
                let coeff = jv[[yrow, k * dv + j]] as f64;
                if coeff == 0.0 {
                    continue;
                }
                c_acc += coeff * vc[k * dv + j];
                let (lo, hi) = (
                    vl[v_base + k * dv + j] as f64,
                    vu[v_base + k * dv + j] as f64,
                );
                if coeff >= 0.0 {
                    lo_acc += coeff * lo;
                    hi_acc += coeff * hi;
                } else {
                    lo_acc += coeff * hi;
                    hi_acc += coeff * lo;
                }
            }
            let c = yc_ij - c_acc;
            let approx_lo = c + lo_acc;
            let approx_hi = c + hi_acc;

            // === Sound, closed-form O(radius²) relaxation-error enclosure ===
            // The relaxation error e(x) = Y_true(x) − Ŷ(x) decomposes EXACTLY
            // (algebraic identity, no shared-ξ assumption) as e = T1 + T2 + T3,
            // using P = Pc+dP, V = Vc+dV, s = sc+ds with Σ_k dP[k]=0:
            //   T1 = Σ_k (Vc[k,j]−Yc)·R_P[k]   (softmax 2nd-order remainder)
            //   T2 = Σ_k dP[k]·dV[k,j]         (dropped P·V bilinear remainder)
            //   T3 = Σ_m g[m]·(ds[m]−ds_lin[m]) (Q·Kᵀ bilinear remainder)
            // Each is bounded by a SOUND interval, all O(radius²). See module doc.
            //
            // T1: |R_P[k]| ≤ p_hi[k]·(Σ_m|ds[m]|)²; recenter the V weight by Yc
            // (legal since Σ_k R_P[k]=0). T2: |dP[k]|≤dp_mag[k], |dV[k,j]|≤radV.
            // T3: |ds[m]−ds_lin[m]| ≤ score_remainder[m], weighted by |g[m]|.
            let sum_ds_sq = sum_ds_abs * sum_ds_abs;
            let mut e_t1 = 0.0f64;
            let mut e_t2 = 0.0f64;
            let mut e_t3 = 0.0f64;
            for k in 0..sk {
                // Include EVERY visible key (mask gate, not `pc[k]==0.0`). For an
                // underflowed-center visible key Pc[k]≈0 and J_P[k,·]≈0, so its
                // surrogate Jacobian is ≈0 and the FULL first-order swing P[k]·(Vc−Yc)
                // lives unmodeled in R_P[k]≈P[k]; the T1 bound
                // |R_P[k]| ≤ p_hi[k]·(Σ|ds|)² (p_hi[k]→1 here) encloses it. On a
                // pathological large-gap box this T1 is large and the sound-envelope
                // clamp below absorbs it (ties the simplex envelope, sound); on normal
                // boxes pc[k]≈0 keys are not underflowed-and-rising so this stays
                // O(radius²) and the win is preserved.
                if !visible(i, k) {
                    continue;
                }
                let vkj = vc[k * dv + j];
                // T1: weight |Vc[k,j] − Yc| · p_hi[k] · (Σ|ds|)².
                e_t1 += (vkj - yc_ij).abs() * p_hi[k] * sum_ds_sq;
                // T2: |dP[k]| · radV[k,j].
                e_t2 += dp_mag[k] * vr[k * dv + j];
                // T3: |g[m]| · score_remainder[m], g[m] = Pc[m]·(Vc[m,j]−Yc).
                let g = pc[k] * (vkj - yc_ij);
                e_t3 += g.abs() * score_remainder[k];
            }
            // Outward (sound) magnitude of the error: round the sum UP into f32
            // precision by inflating with a tiny relative+absolute slack, so the
            // f64→f32 cast of the constant cannot shrink the enclosure.
            let e_mag = e_t1 + e_t2 + e_t3;
            let e_mag = next_up_f64(e_mag);

            // === Directional, clamped margin ===
            // e(x) ∈ [−e_mag, +e_mag] pointwise, so Ŷ−e_mag ≤ Y ≤ Ŷ+e_mag.
            // CLAMP each side to a SOUND output envelope so the certified interval
            // can never exceed that envelope — guaranteeing tighter-or-equal and a
            // graceful tie on large boxes. The clamp is an UPPER cap on the margin:
            // it only TIGHTENS, and as long as the reference ENCLOSES Y_true it can
            // never under-approximate.
            //
            // SOUNDNESS (false-proof fix, part 2): we do NOT clamp to the external
            // IBP ALONE. In the large-score-gap / underflow regime the passed-in
            // simplex-aware IBP can itself be NARROW-UNSOUND (a separate softmax-IBP
            // epsilon-vs-underflow issue), and clamping to a too-narrow reference
            // would re-introduce a false certificate. Instead we clamp to a SOUND
            // envelope computed HERE from this node's OWN sound per-key softmax range
            // `[p_lo[k], p_hi[k]]` (per-ratio-shifted, underflow-robust) and the V
            // box, via the exact box∩simplex water-filling LP — then take the WIDER
            // of {this envelope, the external IBP}. The widening side (`min` on the
            // lower reference, `max` on the upper) guarantees the reference always
            // encloses Y_true even if the external IBP is unsoundly narrow; on normal
            // boxes the two references agree, so tightness (the end-to-end win) is
            // preserved. The framework still intersects downstream with its IBP.
            let ibp_l = ibp_lo[y_glob] as f64;
            let ibp_h = ibp_hi[y_glob] as f64;
            // Sound self-computed simplex envelope for Y[i,j] over the V box and the
            // sound P range. min/max of Σ_k p_k·V[k,j] over {p_lo≤p≤p_hi, Σp=1}.
            let (self_lo, self_hi) = simplex_v_envelope(
                &p_lo,
                &p_hi,
                sk,
                |k| vl[v_base + k * dv + j] as f64,
                |k| vu[v_base + k * dv + j] as f64,
            );
            // Widen the clamp reference to the SOUND envelope (never trust a
            // narrower-than-sound external IBP).
            let ref_lo = ibp_l.min(self_lo);
            let ref_hi = ibp_h.max(self_hi);
            let gap_below = (approx_hi - ref_lo).max(0.0);
            let gap_above = (ref_hi - approx_lo).max(0.0);
            let m_below = e_mag.max(0.0).min(gap_below);
            let m_above = e_mag.max(0.0).min(gap_above);

            if !c.is_finite() || !m_below.is_finite() || !m_above.is_finite() || !e_mag.is_finite()
            {
                // Degrade: zero this output's Jacobian row, set ±∞ constant.
                for d in 0..dk {
                    jq[[yrow, i * dk + d]] = 0.0;
                }
                for k in 0..sk {
                    for d in 0..dk {
                        jk[[yrow, k * dk + d]] = 0.0;
                    }
                    jv[[yrow, k * dv + j]] = 0.0;
                }
                const_lo[yrow] = f64::NEG_INFINITY;
                const_hi[yrow] = f64::INFINITY;
            } else {
                const_lo[yrow] = c - m_below;
                const_hi[yrow] = c + m_above;
            }
        }
    }

    Ok(SliceBlock {
        jq,
        jk,
        jv,
        const_lo,
        const_hi,
    })
}

/// Accumulates the per-input output coefficient matrices (in f64) and the shared
/// bias channel, composing one slice at a time. Holds the full `(n_out × n_in)`
/// outputs (inherent) but only ever one transient `SliceBlock` at a time.
struct OutputAccumulator {
    aq_lo: Array2<f64>,
    aq_hi: Array2<f64>,
    ak_lo: Array2<f64>,
    ak_hi: Array2<f64>,
    av_lo: Array2<f64>,
    av_hi: Array2<f64>,
    bias_lo: Array1<f64>,
    bias_hi: Array1<f64>,
    /// Per-output-row degradation flag (any slice produced a non-finite const).
    degraded: Vec<bool>,
}

impl OutputAccumulator {
    fn new(node_lb: &LinearBounds, n_out: usize, n_q: usize, n_k: usize, n_v: usize) -> Self {
        Self {
            aq_lo: Array2::zeros((n_out, n_q)),
            aq_hi: Array2::zeros((n_out, n_q)),
            ak_lo: Array2::zeros((n_out, n_k)),
            ak_hi: Array2::zeros((n_out, n_k)),
            av_lo: Array2::zeros((n_out, n_v)),
            av_hi: Array2::zeros((n_out, n_v)),
            bias_lo: node_lb.lower_b().mapv(|v| v as f64),
            bias_hi: node_lb.upper_b().mapv(|v| v as f64),
            degraded: vec![false; n_out],
        }
    }

    /// Compose one slice's Jacobian block into the output matrices.
    ///
    /// For each output row `o` and each of this slice's Y outputs `y` (global
    /// index `y_base + yrow`): `A_input[o, in] += node_lb_A[o, y] · J[yrow, in]`
    /// and `bias[o] += node_lb_A[o, y] · const[yrow]`. The Jacobian is the same
    /// for the lower/upper relaxation (the affine surface); the upstream
    /// coefficient sign selects which constant. f64 accumulation throughout.
    fn compose_slice(
        &mut self,
        node_lb: &LinearBounds,
        slice: usize,
        dims: &SliceDims,
        block: &SliceBlock,
    ) {
        let SliceDims { sq, sk, dk, dv, .. } = *dims;
        let n_out = self.aq_lo.nrows();
        let q_off = slice * sq * dk;
        let k_off = slice * sk * dk;
        let v_off = slice * sk * dv;
        let y_off = slice * sq * dv;
        let yn = sq * dv;
        let lower_a = node_lb.lower_a();
        let upper_a = node_lb.upper_a();

        for o in 0..n_out {
            if self.degraded[o] {
                continue;
            }
            let mut row_degraded = false;
            for yrow in 0..yn {
                let y = y_off + yrow;
                let la = lower_a[[o, y]] as f64;
                let ua = upper_a[[o, y]] as f64;
                if la == 0.0 && ua == 0.0 {
                    continue;
                }
                let cl = if la >= 0.0 {
                    block.const_lo[yrow]
                } else {
                    block.const_hi[yrow]
                };
                let cu = if ua >= 0.0 {
                    block.const_hi[yrow]
                } else {
                    block.const_lo[yrow]
                };
                if !cl.is_finite() || !cu.is_finite() {
                    row_degraded = true;
                    break;
                }
                self.bias_lo[o] += la * cl;
                self.bias_hi[o] += ua * cu;

                for (c, &jval) in block.jq.row(yrow).indexed_iter() {
                    if jval != 0.0 {
                        let jv64 = jval as f64;
                        self.aq_lo[[o, q_off + c]] += la * jv64;
                        self.aq_hi[[o, q_off + c]] += ua * jv64;
                    }
                }
                for (c, &jval) in block.jk.row(yrow).indexed_iter() {
                    if jval != 0.0 {
                        let jv64 = jval as f64;
                        self.ak_lo[[o, k_off + c]] += la * jv64;
                        self.ak_hi[[o, k_off + c]] += ua * jv64;
                    }
                }
                for (c, &jval) in block.jv.row(yrow).indexed_iter() {
                    if jval != 0.0 {
                        let jv64 = jval as f64;
                        self.av_lo[[o, v_off + c]] += la * jv64;
                        self.av_hi[[o, v_off + c]] += ua * jv64;
                    }
                }
            }
            if row_degraded {
                self.degraded[o] = true;
            }
        }
    }

    /// Finalize: cast to f32 with directed bias rounding, zero degraded rows.
    fn finalize(
        self,
        n_out: usize,
        n_q: usize,
        n_k: usize,
        n_v: usize,
    ) -> Result<AttnTernaryResult> {
        let mut aq_lo = self.aq_lo.mapv(|v| v as f32);
        let mut aq_hi = self.aq_hi.mapv(|v| v as f32);
        let mut ak_lo = self.ak_lo.mapv(|v| v as f32);
        let mut ak_hi = self.ak_hi.mapv(|v| v as f32);
        let mut av_lo = self.av_lo.mapv(|v| v as f32);
        let mut av_hi = self.av_hi.mapv(|v| v as f32);
        let mut bias_lo = Array1::<f32>::zeros(n_out);
        let mut bias_hi = Array1::<f32>::zeros(n_out);

        for o in 0..n_out {
            let blo = self.bias_lo[o];
            let bhi = self.bias_hi[o];
            if self.degraded[o] || !blo.is_finite() || !bhi.is_finite() {
                bias_lo[o] = f32::NEG_INFINITY;
                bias_hi[o] = f32::INFINITY;
                for c in 0..n_q {
                    aq_lo[[o, c]] = 0.0;
                    aq_hi[[o, c]] = 0.0;
                }
                for c in 0..n_k {
                    ak_lo[[o, c]] = 0.0;
                    ak_hi[[o, c]] = 0.0;
                }
                for c in 0..n_v {
                    av_lo[[o, c]] = 0.0;
                    av_hi[[o, c]] = 0.0;
                }
            } else {
                bias_lo[o] = next_down_f32(blo as f32);
                bias_hi[o] = next_up_f32(bhi as f32);
            }
        }

        let z = || Array1::<f32>::zeros(n_out);
        let lb_q = LinearBounds::new_or_conservative(aq_lo, z(), aq_hi, z())?;
        let lb_k = LinearBounds::new_or_conservative(ak_lo, z(), ak_hi, z())?;
        let lb_v = LinearBounds::new_or_conservative(av_lo, z(), av_hi, z())?;
        Ok((vec![Some(lb_q), Some(lb_k), Some(lb_v)], bias_lo, bias_hi))
    }
}

/// Center (midpoint) of a flat box slice in f64.
fn center_slice(lo: &[f32], hi: &[f32], base: usize, len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| {
            let l = lo[base + i] as f64;
            let u = hi[base + i] as f64;
            l + (u - l) * 0.5
        })
        .collect()
}

/// Half-width (radius) of a flat box slice in f64, rounded UP so every per-coord
/// radius is a sound over-bound of the true `(u−l)/2` (the directional error
/// margin must never under-state the box extent). Clamped to be non-negative.
fn radius_slice(lo: &[f32], hi: &[f32], base: usize, len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| {
            let l = lo[base + i] as f64;
            let u = hi[base + i] as f64;
            next_up_f64(((u - l) * 0.5).max(0.0))
        })
        .collect()
}

/// SOUND output envelope for one `Y[i,j] = Σ_k P[k]·V[k,j]` over the box∩simplex
/// `{p_lo ≤ p ≤ p_hi, Σ_k p_k = 1}` and the V box `vlo[k] ≤ V[k,j] ≤ vhi[k]`.
///
/// Returns `(lo, hi)` with `lo ≤ Y[i,j] ≤ hi` for EVERY feasible `(P, V)`. This is
/// the same sum-to-1-aware ("simplex water-filling") LP as the IBP softmax-V lever
/// (`softmax::simplex_v`), but recomputed here from THIS node's own sound per-key
/// softmax range so the clamp reference never depends on a possibly-narrow-unsound
/// external IBP (the underflow false-proof fix).
///
/// `p_lo`/`p_hi` are the sound per-key softmax bounds (independent, so they may
/// over/under-sum 1). Water-filling is exact when `Σ p_lo ≤ 1 ≤ Σ p_hi`; otherwise
/// we FALL BACK to the plain convex hull of the V box rows (`[min vlo, max vhi]`),
/// which is always a sound enclosure of any convex combination. Computed in f64.
#[inline]
fn simplex_v_envelope(
    p_lo: &[f64],
    p_hi: &[f64],
    sk: usize,
    vlo: impl Fn(usize) -> f64,
    vhi: impl Fn(usize) -> f64,
) -> (f64, f64) {
    // Plain convex-hull fallback over visible (p_hi>0) rows — always sound.
    let mut hull_lo = f64::INFINITY;
    let mut hull_hi = f64::NEG_INFINITY;
    let mut any = false;
    for k in 0..sk {
        if p_hi[k] <= 0.0 {
            continue; // masked / impossible key (P[k]=0 for all inputs)
        }
        any = true;
        hull_lo = hull_lo.min(vlo(k));
        hull_hi = hull_hi.max(vhi(k));
    }
    if !any || !hull_lo.is_finite() || !hull_hi.is_finite() {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    let sum_lo: f64 = (0..sk).map(|k| p_lo[k]).sum();
    let sum_hi: f64 = (0..sk).map(|k| p_hi[k]).sum();
    // Water-filling needs the box to bracket the simplex hyperplane. With a small
    // tolerance for f64 noise; otherwise use the (sound) hull.
    if sum_lo > 1.0 + 1e-9 || sum_hi < 1.0 - 1e-6 {
        return (hull_lo, hull_hi);
    }
    // MIN of Σ p_k·vlo[k]: start at p_lo, pour residual mass (1−Σp_lo) into the
    // SMALLEST-coefficient (most-negative vlo) coordinates first, up to p_hi.
    // MAX of Σ p_k·vhi[k]: pour into the LARGEST-coefficient (vhi) coordinates.
    let env = |coeff: &dyn Fn(usize) -> f64, maximize: bool| -> f64 {
        // Start every coordinate at its floor p_lo[k]; each is poured at most once
        // (greedy order), so p[k] stays at p_lo[k] until it is filled.
        let mut base = 0.0f64;
        for k in 0..sk {
            base += p_lo[k] * coeff(k);
        }
        let mut budget = (1.0 - sum_lo).max(0.0);
        let mut order: Vec<usize> = (0..sk).filter(|&k| p_hi[k] > 0.0).collect();
        order.sort_by(|&a, &b| {
            let (ca, cb) = (coeff(a), coeff(b));
            if maximize {
                cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
            }
        });
        for &k in &order {
            if budget <= 0.0 {
                break;
            }
            let room = (p_hi[k] - p_lo[k]).max(0.0);
            let add = room.min(budget);
            base += add * coeff(k);
            budget -= add;
        }
        base
    };
    let lo = env(&|k| vlo(k), false);
    let hi = env(&|k| vhi(k), true);
    // Intersect with the always-sound hull (water-filling result lies inside it,
    // but guard f64 noise) and round OUTWARD.
    let lo = next_down_f64(lo.max(hull_lo).min(hull_hi));
    let hi = next_up_f64(hi.min(hull_hi).max(hull_lo));
    (lo, hi)
}

/// Largest f64 strictly LESS than a finite `x` by one ULP (outward DOWN rounding).
#[inline]
fn next_down_f64(x: f64) -> f64 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude >= f64::INFINITY.to_bits() {
        return x;
    }
    if magnitude == 0 {
        return -f64::from_bits(1);
    }
    if bits & 0x8000_0000_0000_0000 == 0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

/// Smallest f64 strictly greater than a finite non-negative `x` by one ULP,
/// for OUTWARD rounding of error-margin magnitudes. Non-finite / negative input
/// is returned unchanged (callers guard non-finite separately). `0.0` maps to
/// the smallest positive subnormal so a zero magnitude still rounds outward.
#[inline]
fn next_up_f64(x: f64) -> f64 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude >= f64::INFINITY.to_bits() || (bits & 0x8000_0000_0000_0000 != 0 && magnitude != 0)
    {
        return x;
    }
    if magnitude == 0 {
        return f64::from_bits(1);
    }
    f64::from_bits(bits + 1)
}

/// Flattened, contiguous lower/upper as owned `Vec<f32>` (rejects non-contiguous).
fn lower_upper_flat(t: &BoundedTensor) -> Result<(Vec<f32>, Vec<f32>)> {
    let (l, u) = t.lower_upper();
    let ls = l.as_slice().ok_or_else(|| {
        NyError::UnsupportedOp("SelfAttention CROWN: non-contiguous input".into())
    })?;
    let us = u.as_slice().ok_or_else(|| {
        NyError::UnsupportedOp("SelfAttention CROWN: non-contiguous input".into())
    })?;
    Ok((ls.to_vec(), us.to_vec()))
}

/// Contiguous flat slice of an `ArrayD`, as an owned `Vec<f32>`.
fn contiguous(a: &ndarray::ArrayD<f32>) -> Result<Vec<f32>> {
    a.as_slice()
        .map(|s| s.to_vec())
        .ok_or_else(|| NyError::UnsupportedOp("SelfAttention CROWN: non-contiguous IBP".into()))
}

#[cfg(test)]
mod tests;
