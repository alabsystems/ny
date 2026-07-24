// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Convex-combination ("simplex envelope") tightening for a row-stochastic
//! Softmax output contracted against a constant weight vector/matrix.
//!
//! # The DFL (Distribution-Focal-Loss / expectation-decode) pattern
//!
//! A common decode head computes
//!
//! ```text
//! p = softmax(logits)            // along the bin axis, |bins| = K
//! y = p · w   (MatMul / Linear)  // w = constant bin-index weights, e.g. [0, 1, .., K-1]
//! ```
//!
//! Each output element is a **convex combination** of the contracted constants:
//! `y_o = sum_k p_k * w_{k,o}` with `p_k >= 0` and `sum_k p_k = 1`. A convex
//! combination provably lies in the range of the combined constants, so
//!
//! ```text
//! min_k w_{k,o}  <=  y_o  <=  max_k w_{k,o}.
//! ```
//!
//! # Why term-wise IBP is loose here
//!
//! Interval IBP for the softmax yields `p_k in [0, 1]` but **drops the simplex
//! constraint** `sum_k p_k = 1`. The downstream MatMul/Linear term-wise IBP then
//! computes the upper bound as `sum_k 1 * w_{k,o}` — e.g. `K(K-1)/2 = 120` for
//! 16 bins `[0..15]`, versus the true maximum of `15`. This module restores the
//! dropped simplex information by intersecting each output element's IBP
//! interval with the convex-combination envelope `[min_k w, max_k w]`.
//!
//! # Soundness
//!
//! The envelope `[min_k w_{k,o}, max_k w_{k,o}]` is a sound enclosure of `y_o`
//! for *any* point in the simplex, so intersecting the existing (also-sound) IBP
//! interval with it can only **tighten** — never widen — the bound. The result
//! `[max(l_ibp, min_k w), min(u_ibp, max_k w)]` therefore remains a sound
//! enclosure. We only apply it when:
//!   1. the contracted operand is the output of a `Softmax`/`CausalSoftmax`
//!      producer node (so `sum_k p_k = 1` provably holds), and
//!   2. the softmax axis equals the contraction axis.
//!
//! When any precondition fails we return `None` and the caller keeps the
//! existing term-wise bound (never widening).
//!
//! # Perturbed weight (attention `softmax @ V`)
//!
//! When the weight operand is a compile-time CONSTANT (`lower == upper`), the
//! envelope is the fixed range `[min_k w, max_k w]` (the DFL/decode case above).
//!
//! When the weight operand is itself PERTURBED — e.g. the value matrix `V` in
//! self-attention `softmax(QKᵀ) @ V`, where `V` is a projection of the perturbed
//! input — the plain `[min_k Bl, max_k Bh]` envelope is still sound but almost
//! always looser than the term-wise IBP (attention probabilities are small, so
//! the IBP dot product is tight). The sum-to-1 constraint is recovered instead
//! by an exact LP over the box-intersected simplex
//! `{p : Pl <= p <= Ph, sum p = 1}`, solved by greedy water-filling
//! ([`GraphNetwork::try_softmax_v_simplex_lp`], restricted to the 2-D
//! non-batched MatMul). This is strictly tighter than the plain envelope and
//! gives multi-x tightening on realistic attention blocks (DETR/SVTR/table
//! transformer encoder self-attention).

use ndarray::{ArrayD, Axis, Zip};
use ny_core::Result;
use ny_tensor::BoundedTensor;

use crate::layers::common::resolve_axis_i32;
use crate::layers::softmax::{simplex_lp_max, simplex_lp_min};
use crate::layers::Layer;

use super::super::{GraphNetwork, GraphNode, NETWORK_INPUT};

impl GraphNetwork {
    /// Resolve a node-input name to its already-computed bounds.
    ///
    /// Mirrors [`Self::bounds_ref`] but does not require borrowing the
    /// network input separately — the simplex-envelope check never inspects the
    /// raw network input (the softmax producer is always an interior node).
    fn dfl_bounds<'a>(
        &self,
        name: &str,
        input: &'a BoundedTensor,
        cache: &'a std::collections::HashMap<String, BoundedTensor>,
    ) -> Option<&'a BoundedTensor> {
        if name == NETWORK_INPUT {
            Some(input)
        } else {
            cache.get(name)
        }
    }

    /// Whether the producer of `name` is a row-stochastic Softmax-family node,
    /// and if so the (positive, resolved-against-`ndim`) softmax axis.
    fn softmax_producer_axis(&self, name: &str, ndim: usize) -> Option<usize> {
        let producer = self.nodes.get(name)?;
        let axis_i32 = match &producer.layer {
            Layer::Softmax(s) => s.axis,
            Layer::CausalSoftmax(s) => s.axis,
            _ => return None,
        };
        // A Softmax/CausalSoftmax is provably row-stochastic along its axis.
        resolve_axis_i32(axis_i32, ndim, "DFL-simplex-envelope").ok()
    }

    /// If `node` contracts a Softmax output against a constant weight along the
    /// softmax axis, intersect `output_bounds` with the convex-combination
    /// envelope and return the tightened tensor; otherwise return `None`.
    ///
    /// SOUNDNESS: see the module-level proof. Intersection with a sound envelope
    /// can only tighten, so this is sound for every shape/axis where the
    /// preconditions hold, and a no-op (`None`) otherwise.
    pub(crate) fn try_dfl_simplex_envelope(
        &self,
        node: &GraphNode,
        output_bounds: &BoundedTensor,
        input: &BoundedTensor,
        cache: &std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<Option<BoundedTensor>> {
        match &node.layer {
            // y = W x + b, x = softmax output. W is the layer's own constant.
            Layer::Linear(linear) => {
                // Bias breaks the pure convex-combination structure; only the
                // unbiased (or zero-bias) decode is a convex combination of W.
                if linear
                    .bias
                    .as_ref()
                    .is_some_and(|b| b.iter().any(|&v| v != 0.0))
                {
                    return Ok(None);
                }
                let Ok(x_name) = node.require_unary_input() else {
                    return Ok(None);
                };
                let Some(x_bounds) = self.dfl_bounds(x_name, input, cache) else {
                    return Ok(None);
                };
                let x_ndim = x_bounds.shape().len();
                // Linear contracts the LAST axis (in_features). Require the
                // softmax axis to be that same last axis.
                let Some(sm_axis) = self.softmax_producer_axis(x_name, x_ndim) else {
                    return Ok(None);
                };
                if x_ndim == 0 || sm_axis != x_ndim - 1 {
                    return Ok(None);
                }
                // Per output feature o: envelope over contracted weights W[o, ..].
                // weight shape is (out_features, in_features).
                let weight = &linear.weight;
                let out_features = weight.nrows();
                let (mut env_min, mut env_max) = (
                    vec![f32::INFINITY; out_features],
                    vec![f32::NEG_INFINITY; out_features],
                );
                for o in 0..out_features {
                    for &w in weight.row(o).iter() {
                        if w.is_nan() {
                            // A NaN weight makes the envelope meaningless; bail
                            // out and keep the (conservatively repaired) IBP bound.
                            return Ok(None);
                        }
                        env_min[o] = env_min[o].min(w);
                        env_max[o] = env_max[o].max(w);
                    }
                }
                // Output last axis indexes the out_features; intersect per o.
                Ok(Some(intersect_last_axis_envelope(
                    output_bounds,
                    &env_min,
                    &env_max,
                )))
            }
            // y = A @ B (optionally B^T). One operand is the softmax output and
            // the other is a constant weight; contraction is over A's last axis.
            Layer::MatMul(matmul) => {
                let Ok((a_name, b_name)) = node.require_binary_inputs() else {
                    return Ok(None);
                };
                let (Some(a_bounds), Some(b_bounds)) = (
                    self.dfl_bounds(a_name, input, cache),
                    self.dfl_bounds(b_name, input, cache),
                ) else {
                    return Ok(None);
                };
                // The softmax operand must be A (the left factor): A @ B contracts
                // A's last axis against B's contraction axis, and `A`'s last axis
                // is the simplex axis for the per-row distribution.
                let a_ndim = a_bounds.shape().len();
                let Some(sm_axis) = self.softmax_producer_axis(a_name, a_ndim) else {
                    return Ok(None);
                };
                if a_ndim < 2 || sm_axis != a_ndim - 1 {
                    return Ok(None);
                }
                let b_shape = b_bounds.shape();
                if b_shape.len() != 2 {
                    return Ok(None);
                }
                let k = a_bounds.shape()[a_ndim - 1];

                // PERTURBED-V SIMPLEX BOUND (#softmax-V-lever).
                //
                // When B (= V) is NOT a compile-time constant, the plain
                // [min_k B, max_k B] envelope is still sound, but it is almost
                // always LOOSER than the term-wise IBP for `softmax @ V` (the
                // attention probabilities are small/near-uniform, so the IBP dot
                // product is tight). The win comes from the SUM-TO-1 constraint:
                // since `sum_k p[i,k] = 1` and `p[i,k] >= 0`, each output
                //   y[i,j] = sum_k p[i,k] * B_kj   with B_kj in [Bl_kj, Bh_kj]
                // is bounded by the maximum/minimum of `sum_k p[i,k] * B*_kj` over
                // the box-intersected simplex `{pl<=p<=ph, sum p = 1}`. That LP is
                // solved exactly by greedy water-filling (see `simplex_lp_*`),
                // which exploits both the per-row probability intervals AND
                // sum-to-1, and is strictly tighter than the plain envelope.
                //
                // Restricted to the 2-D non-batched case (`a_ndim == 2`); the
                // batched cases keep the existing constant-only behaviour (no
                // regression — returns None below if B is non-constant).
                if !is_constant(b_bounds) {
                    if a_ndim == 2 {
                        return Ok(self.try_softmax_v_simplex_lp(
                            a_bounds,
                            b_bounds,
                            output_bounds,
                            matmul.transpose_b,
                            matmul.scale,
                            k,
                        ));
                    }
                    return Ok(None);
                }
                let b = b_bounds.lower(); // == upper (constant)
                                          // For non-transposed B: shape (K, N), contract over rows (axis 0),
                                          //   output column j envelope = [min_k B[k,j], max_k B[k,j]].
                                          // For transposed B (B^T, B stored as (N, K)): contract over B's
                                          //   last axis, output column j envelope = [min_k B[j,k], max_k B[j,k]].
                let (n, contract_dim) = if matmul.transpose_b {
                    (b_shape[0], b_shape[1])
                } else {
                    (b_shape[1], b_shape[0])
                };
                if contract_dim != k {
                    return Ok(None);
                }
                let (mut env_min, mut env_max) =
                    (vec![f32::INFINITY; n], vec![f32::NEG_INFINITY; n]);
                for j in 0..n {
                    for kk in 0..k {
                        let w = if matmul.transpose_b {
                            b[[j, kk]]
                        } else {
                            b[[kk, j]]
                        };
                        if w.is_nan() {
                            return Ok(None);
                        }
                        env_min[j] = env_min[j].min(w);
                        env_max[j] = env_max[j].max(w);
                    }
                }
                // The MatMul `scale` (e.g. attention 1/sqrt(d)) linearly rescales
                // the convex combination, so the envelope scales by the same
                // factor (flipping when negative). Apply it so the envelope stays
                // a sound bound on the *scaled* output.
                if let Some(scale) = matmul.scale {
                    for j in 0..n {
                        let (lo, hi) = (env_min[j] * scale, env_max[j] * scale);
                        env_min[j] = lo.min(hi);
                        env_max[j] = lo.max(hi);
                    }
                }
                // Output last axis indexes N; intersect per j.
                Ok(Some(intersect_last_axis_envelope(
                    output_bounds,
                    &env_min,
                    &env_max,
                )))
            }
            _ => Ok(None),
        }
    }

    /// Sound, sum-to-1-aware bound for `Y = P @ V` (optionally `P @ V^T`) where
    /// `P` is a row-stochastic Softmax output (so each row is a probability
    /// distribution: `P[i,k] >= 0`, `sum_k P[i,k] = 1`) and `V` is a perturbed
    /// (interval) operand. Intersects the result with `output_bounds`.
    ///
    /// Shapes (2-D, non-batched): `P` is `(M, K)` from cache (`a_bounds`), `V`
    /// is `(K, N)` (or `(N, K)` if `transpose_b`). Output is `(M, N)`.
    ///
    /// # Soundness
    ///
    /// For every reachable point `(P, V)`:
    ///   * `P[i, :]` is a true softmax row, so `P[i,k] in [Pl[i,k], Ph[i,k]]`
    ///     (sound IBP interval) AND `sum_k P[i,k] = 1` exactly, i.e. `P[i,:]` lies
    ///     in the box-intersected simplex `S_i = {p : Pl[i,:] <= p <= Ph[i,:],
    ///     sum_k p_k = 1}`.
    ///   * `V[k,j] in [Vl[k,j], Vh[k,j]]` (sound IBP interval).
    ///
    /// Since `p_k >= 0`:
    ///   `Y[i,j] = sum_k p_k V[k,j] <= sum_k p_k Vh[k,j] <= max_{p in S_i} sum_k p_k Vh[k,j]`,
    ///   `Y[i,j] = sum_k p_k V[k,j] >= sum_k p_k Vl[k,j] >= min_{p in S_i} sum_k p_k Vl[k,j]`.
    /// The two extremal LPs over `S_i` (a box intersected with one hyperplane) are
    /// solved EXACTLY by greedy water-filling — see [`simplex_lp_max`]. We round
    /// the f64 LP value OUTWARD on the f32 cast (`next_down`/`next_up`) so the
    /// stored interval still encloses the true real value. Intersecting the
    /// resulting per-element envelope with the (also-sound) `output_bounds` can
    /// only tighten, never widen. If a row's simplex is infeasible after the IBP
    /// over-approximation (`sum Pl > 1` or `sum Ph < 1`), we skip that row's
    /// tightening (keep the existing bound) — never widening.
    ///
    /// Returns `None` when shapes/preconditions do not match (caller keeps the
    /// existing term-wise IBP bound).
    #[allow(clippy::too_many_arguments)]
    fn try_softmax_v_simplex_lp(
        &self,
        p_bounds: &BoundedTensor,
        v_bounds: &BoundedTensor,
        output_bounds: &BoundedTensor,
        transpose_b: bool,
        scale: Option<f32>,
        k: usize,
    ) -> Option<BoundedTensor> {
        let p_shape = p_bounds.shape();
        let v_shape = v_bounds.shape();
        let out_shape = output_bounds.shape();
        if p_shape.len() != 2 || v_shape.len() != 2 || out_shape.len() != 2 {
            return None;
        }
        let (m, kp) = (p_shape[0], p_shape[1]);
        if kp != k {
            return None;
        }
        // V is (K, N) normally, or (N, K) when transpose_b (V^T pattern).
        let (n, contract_dim) = if transpose_b {
            (v_shape[0], v_shape[1])
        } else {
            (v_shape[1], v_shape[0])
        };
        if contract_dim != k || out_shape[0] != m || out_shape[1] != n {
            return None;
        }
        if k == 0 || m == 0 || n == 0 {
            return None;
        }

        let scale_f64 = scale.unwrap_or(1.0) as f64;
        if !scale_f64.is_finite() {
            return None;
        }

        let (pl, ph) = p_bounds.lower_upper();
        let (vl, vh) = v_bounds.lower_upper();
        let (out_l, out_u) = output_bounds.lower_upper();

        let mut new_lower = out_l.clone();
        let mut new_upper = out_u.clone();

        // Per-output-column V ranges as contiguous (over k) buffers.
        // vcol_lo[j][k] = Vl[k,j] (or Vl[j,k] if transpose_b), etc.
        let mut vcol_lo = vec![vec![0.0f32; k]; n];
        let mut vcol_hi = vec![vec![0.0f32; k]; n];
        for j in 0..n {
            for kk in 0..k {
                let (lo, hi) = if transpose_b {
                    (vl[[j, kk]], vh[[j, kk]])
                } else {
                    (vl[[kk, j]], vh[[kk, j]])
                };
                if !lo.is_finite() || !hi.is_finite() {
                    // A non-finite V entry makes the LP unusable; abandon
                    // tightening entirely (keep IBP bound) for safety.
                    return None;
                }
                vcol_lo[j][kk] = lo;
                vcol_hi[j][kk] = hi;
            }
        }

        // Per-row probability interval buffers.
        let mut p_lo = vec![0.0f32; k];
        let mut p_hi = vec![0.0f32; k];
        for i in 0..m {
            let mut bad = false;
            for kk in 0..k {
                let lo = pl[[i, kk]];
                let hi = ph[[i, kk]];
                // Softmax IBP already clamps to [0,1]; defensively reject
                // anything non-finite or negative (would break the LP's p>=0
                // premise) and skip this row.
                if !lo.is_finite() || !hi.is_finite() || lo < 0.0 || hi < lo {
                    bad = true;
                    break;
                }
                p_lo[kk] = lo;
                p_hi[kk] = hi;
            }
            if bad {
                continue; // keep existing bound for this row (no widening)
            }
            // Feasibility of the box-intersected simplex for this row. Strict on
            // the lower side (sum pl > 1 would put the LP floor above the
            // simplex; see `simplex_lp_max` soundness note); tolerant on the
            // upper side (only rounding noise makes sum ph < 1).
            let sum_lo: f64 = p_lo.iter().map(|&x| x as f64).sum();
            let sum_hi: f64 = p_hi.iter().map(|&x| x as f64).sum();
            if sum_lo > 1.0 || sum_hi < 1.0 - 1e-5 {
                continue; // infeasible after over-approx; skip (no widening)
            }

            for j in 0..n {
                let hi = simplex_lp_max(&p_lo, &p_hi, &vcol_hi[j]);
                let lo = simplex_lp_min(&p_lo, &p_hi, &vcol_lo[j]);
                let (Some(mut hi), Some(mut lo)) = (hi, lo) else {
                    continue;
                };
                hi *= scale_f64;
                lo *= scale_f64;
                if scale_f64 < 0.0 {
                    std::mem::swap(&mut hi, &mut lo);
                }
                // Outward f64->f32 rounding so the envelope encloses the true
                // real value despite the cast.
                let env_hi = ny_tensor::next_up_f32(hi as f32);
                let env_lo = ny_tensor::next_down_f32(lo as f32);
                if !env_hi.is_finite() || !env_lo.is_finite() {
                    continue;
                }
                // Intersect (tighten only). If disjoint after intersection the
                // envelope is the authoritative sound range (the true value lies
                // in it), so clamp to it.
                let cur_l = new_lower[[i, j]];
                let cur_u = new_upper[[i, j]];
                let tl = cur_l.max(env_lo);
                let tu = cur_u.min(env_hi);
                if tl <= tu {
                    new_lower[[i, j]] = tl;
                    new_upper[[i, j]] = tu;
                } else {
                    new_lower[[i, j]] = env_lo;
                    new_upper[[i, j]] = env_hi;
                }
            }
        }

        Some(rebuild(new_lower, new_upper, output_bounds))
    }
}

/// Whether a tensor is a compile-time constant (lower == upper everywhere).
fn is_constant(t: &BoundedTensor) -> bool {
    let (l, u) = t.lower_upper();
    l.iter().zip(u.iter()).all(|(&a, &b)| a == b)
}

/// Intersect each output element's `[lower, upper]` with the per-last-axis
/// envelope `[env_min[idx], env_max[idx]]`, where `idx` is the position along
/// the output's last axis.
///
/// SOUNDNESS: this is a pure intersection (`max` on the lower, `min` on the
/// upper) against a sound enclosure, so the result encloses every value the
/// original IBP interval enclosed that is also a feasible convex combination —
/// i.e. it never drops a reachable output value, and never widens. If the
/// intersection were to invert (`l > u`) due to a disjoint-but-still-sound
/// over-approximation we clamp to the envelope endpoint, which stays sound
/// because the true value lies in the envelope.
fn intersect_last_axis_envelope(
    bounds: &BoundedTensor,
    env_min: &[f32],
    env_max: &[f32],
) -> BoundedTensor {
    let (lower, upper) = bounds.lower_upper();
    let shape = lower.shape().to_vec();
    let ndim = shape.len();
    let last = if ndim == 0 { 0 } else { shape[ndim - 1] };

    // Defensive: if the envelope length does not match the last axis, do not
    // tighten (return a clone). This keeps the function total and sound.
    if ndim == 0 || last != env_min.len() || last != env_max.len() {
        return bounds.clone();
    }

    let mut new_lower = lower.clone();
    let mut new_upper = upper.clone();

    // Iterate over lanes along the last (contraction-output) axis: for every
    // lane of length `last`, intersect element j with envelope[j].
    let axis = Axis(ndim - 1);
    Zip::from(new_lower.lanes_mut(axis))
        .and(new_upper.lanes_mut(axis))
        .for_each(|mut lane_l, mut lane_u| {
            for j in 0..last {
                let lo = lane_l[j].max(env_min[j]);
                let hi = lane_u[j].min(env_max[j]);
                // Intersection (tighten only): the new interval must enclose the
                // true value, which lies in BOTH the IBP interval and the
                // envelope. If they are disjoint (l > u after intersection) the
                // envelope is the authoritative sound range, so clamp to it.
                if lo <= hi {
                    lane_l[j] = lo;
                    lane_u[j] = hi;
                } else {
                    lane_l[j] = env_min[j];
                    lane_u[j] = env_max[j];
                }
            }
        });

    rebuild(new_lower, new_upper, bounds)
}

/// Rebuild a `BoundedTensor` from tightened arrays, falling back to the
/// original bounds if (defensively) construction fails. Tightening must never
/// introduce NaN/Inf because intersection only moves endpoints inward.
fn rebuild(lower: ArrayD<f32>, upper: ArrayD<f32>, original: &BoundedTensor) -> BoundedTensor {
    match BoundedTensor::new(lower, upper) {
        Ok(t) => t,
        Err(_) => original.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::common::BoundPropagation;
    use crate::layers::{Layer, MatMulLayer, SoftmaxLayer};
    use ndarray::{Array2, ArrayD, IxDyn};
    use proptest::prelude::*;

    // (LP-exactness vs brute-force is proptested in
    // `layers::softmax::simplex_v::tests`; here we cover the GRAPH integration.)

    /// Build a graph `input --Softmax--> probs ; input --(reshape via MatMul I)-->`
    /// is awkward; instead drive the private method directly through a 2-node
    /// graph `probs = Softmax(input)` and a binary MatMul whose B operand we
    /// supply as a perturbed bound via the cache. We construct the cache
    /// manually and call `try_softmax_v_simplex_lp`.
    fn make_softmax_matmul_graph(
        transpose_b: bool,
        scale: Option<f32>,
    ) -> (GraphNetwork, GraphNode) {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "probs",
            Layer::Softmax(SoftmaxLayer::new(-1)),
        ));
        let node = GraphNode::binary(
            "out",
            Layer::MatMul(MatMulLayer::new(transpose_b, scale)),
            "probs",
            "vval",
        );
        graph.add_node(node.clone());
        graph.set_output("out");
        (graph, node)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(120))]

        /// End-to-end soundness: the perturbed-V simplex bound (intersected with
        /// the term-wise IBP output) must ENCLOSE every true `softmax(logits) @ V`
        /// for logits/V sampled from their boxes. Monte-Carlo check.
        #[test]
        fn softmax_v_simplex_lp_is_sound_vs_monte_carlo(
            seed in 0u64..100_000,
            seq in 2usize..5,
            kdim in 2usize..5,
            ndim in 1usize..4,
            logit_r in 0.1f32..3.0,
            v_lo in -2.0f32..0.0,
            v_w in 0.1f32..2.0,
        ) {
            use rand::{RngExt, SeedableRng};
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

            // Logits box (seq, kdim).
            let mut llo = Array2::<f32>::zeros((seq, kdim));
            let mut lhi = Array2::<f32>::zeros((seq, kdim));
            for i in 0..seq {
                for k in 0..kdim {
                    let c: f32 = rng.random_range(-1.0..1.0);
                    llo[[i, k]] = c - logit_r;
                    lhi[[i, k]] = c + logit_r;
                }
            }
            let logits = BoundedTensor::new(llo.clone().into_dyn(), lhi.clone().into_dyn()).unwrap();

            // V box. transpose_b=false: V is (kdim, ndim). We test that case.
            let mut vlo = Array2::<f32>::zeros((kdim, ndim));
            let mut vhi = Array2::<f32>::zeros((kdim, ndim));
            for a in 0..kdim {
                for b in 0..ndim {
                    let lo = v_lo + rng.random_range(0.0..1.0);
                    vlo[[a, b]] = lo;
                    vhi[[a, b]] = lo + v_w;
                }
            }
            let vval = BoundedTensor::new(vlo.clone().into_dyn(), vhi.clone().into_dyn()).unwrap();

            // probs = softmax(logits) IBP bound.
            let probs = SoftmaxLayer::new(-1).propagate_ibp(&logits).unwrap();

            // term-wise IBP output bound.
            let scale = None;
            let out_ibp = MatMulLayer::new(false, scale)
                .propagate_ibp_binary(&probs, &vval)
                .unwrap();

            let (graph, _node) = make_softmax_matmul_graph(false, scale);
            let _ = &graph; // graph only needed for the method receiver
            let tightened = graph
                .try_softmax_v_simplex_lp(&probs, &vval, &out_ibp, false, scale, kdim)
                .expect("2-D softmax@V tightening should produce Some");

            // Monte-Carlo: sample logits & V from boxes, compute true output,
            // assert it lies within `tightened` (the claimed sound bound).
            let (tl, tu) = tightened.lower_upper();
            for _ in 0..200 {
                let mut ls = Array2::<f32>::zeros((seq, kdim));
                for i in 0..seq { for k in 0..kdim {
                    ls[[i,k]] = rng.random_range(llo[[i,k]]..=lhi[[i,k]]);
                }}
                let mut vs = Array2::<f32>::zeros((kdim, ndim));
                for a in 0..kdim { for b in 0..ndim {
                    vs[[a,b]] = rng.random_range(vlo[[a,b]]..=vhi[[a,b]]);
                }}
                // softmax rows of ls
                let mut p = ls.clone();
                for mut row in p.rows_mut() {
                    let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut s = 0.0f32;
                    for x in row.iter_mut() { *x = (*x - mx).exp(); s += *x; }
                    for x in row.iter_mut() { *x /= s; }
                }
                let y = p.dot(&vs);
                for ((i,j), &val) in y.indexed_iter() {
                    let lo = tl[[i,j]];
                    let hi = tu[[i,j]];
                    prop_assert!(
                        val >= lo - 1e-3 && val <= hi + 1e-3,
                        "UNSOUND: true {val} outside [{lo},{hi}] at ({i},{j})"
                    );
                }
            }
        }
    }

    /// The tightening is tighten-or-equal: every element of the simplex-LP bound
    /// must be within the term-wise IBP bound (never wider).
    #[test]
    fn softmax_v_simplex_lp_never_widens() {
        let seq = 4;
        let kdim = 4;
        let ndim = 3;
        // Wide logits => probs spread => term-wise IBP loose.
        let llo = ArrayD::from_elem(IxDyn(&[seq, kdim]), -2.0f32);
        let lhi = ArrayD::from_elem(IxDyn(&[seq, kdim]), 2.0f32);
        let logits = BoundedTensor::new(llo, lhi).unwrap();
        let probs = SoftmaxLayer::new(-1).propagate_ibp(&logits).unwrap();

        let vlo = ArrayD::from_elem(IxDyn(&[kdim, ndim]), -1.0f32);
        let vhi = ArrayD::from_elem(IxDyn(&[kdim, ndim]), 1.0f32);
        let vval = BoundedTensor::new(vlo, vhi).unwrap();

        let out_ibp = MatMulLayer::new(false, None)
            .propagate_ibp_binary(&probs, &vval)
            .unwrap();
        let (graph, _n) = make_softmax_matmul_graph(false, None);
        let tightened = graph
            .try_softmax_v_simplex_lp(&probs, &vval, &out_ibp, false, None, kdim)
            .expect("tightening should be Some");

        let (il, iu) = out_ibp.lower_upper();
        let (tl, tu) = tightened.lower_upper();
        let mut tightened_somewhere = false;
        for idx in 0..il.len() {
            let il = il.as_slice().unwrap()[idx];
            let iu = iu.as_slice().unwrap()[idx];
            let tl = tl.as_slice().unwrap()[idx];
            let tu = tu.as_slice().unwrap()[idx];
            assert!(tl >= il - 1e-4, "lower widened: {tl} < {il}");
            assert!(tu <= iu + 1e-4, "upper widened: {tu} > {iu}");
            if tu - tl < (iu - il) - 1e-4 {
                tightened_somewhere = true;
            }
        }
        assert!(
            tightened_somewhere,
            "simplex-LP must tighten at least one element for wide logits/V"
        );
    }
}
