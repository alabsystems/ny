// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Standard interval arithmetic IBP propagation for MatMul.

use ndarray::{ArrayD, IxDyn};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};

use super::shape::{decode_batch_index_into, parse_matmul_dims};
use super::{MatMulLayer, Result};
use crate::layers::linear::crown_single_gamma_n_f64;

impl MatMulLayer {
    /// Propagate IBP bounds through matrix multiplication using standard
    /// interval arithmetic.
    ///
    /// For each element-wise product in the sum, computes min/max over the four
    /// corner products. NaN from 0*inf is widened during accumulation; final
    /// non-finite repair happens at the type boundary via `new_repaired`.
    /// Reference: alpha-beta-CROWN interval_bound.py:73-93.
    ///
    /// # Certified f64 accumulation residual (#matmul-ibp-accumulation-charge)
    ///
    /// Each corner product is EXACT in f64 (an f32×f32 significand fits in
    /// 48 < 53 bits and the exponent range of f64 covers every f32 product,
    /// subnormals included), so the products themselves carry no error. The
    /// `k`-term running sum, however, is plain round-to-nearest f64, and a
    /// directed-rounded final `f64→f32` store certifies only the **cast** — not
    /// that accumulation. Under cancellation the residual is unbounded relative
    /// to the result: with `a = [2^40, 1, 2^40]`, `b = [2^40, 1, -2^40]` the f64
    /// sum evaluates to exactly `0.0` while the true value is `1.0`, so a box
    /// built from the cast alone EXCLUDES the true product — a wrong `unsat` if
    /// such a row ever reached a verdict.
    ///
    /// This is the MatMul analogue of the AveragePool gap named at
    /// `network/graph_ibp.rs` (#avgpool-1ulp-arm) and fixed in
    /// `layers::pooling::average::propagate_ibp_sound`, whose structure this
    /// mirrors. Higham (Accuracy and Stability, Thm 3.1) bounds recursive
    /// summation of `k` terms by `γ⁶⁴_{k−1}·Σ|x_l|`, a MAGNITUDE-sum charge that
    /// survives cancellation. So alongside the interval sum we accumulate
    ///
    /// ```text
    /// S = Σ_l max(|p1|,|p2|,|p3|,|p4|)  ≥  Σ_l |prod_min_l|,  Σ_l |prod_max_l|
    /// ```
    ///
    /// and widen both endpoints by `γ⁶⁴_{k+1}·S_safe` BEFORE the directed store.
    /// `S` is itself summed in round-to-nearest f64 over non-negative terms, so
    /// it can fall SHORT of the true magnitude sum by its own `γ⁶⁴_{k−1}·S`;
    /// `S_safe = S/(1−γ⁶⁴_k)` inflates it outward (mirrors conv's `s_inflate`).
    /// Charging index `k+1` rather than the counted `k−1` leaves two units of
    /// documented margin, which is what pays for the optional `scale` multiply
    /// (one extra rounding: `γ_k − γ_{k−1} = u(1+γ_{k−1})/(1−ku) ≥ u(1+γ_{k−1})`,
    /// exactly the relative error a single f64 multiply can add) — the error
    /// budget `S` is scaled by `|scale|` with it. The remaining sub-ULP slop in
    /// forming the charge (four f64 roundings, ≤ 4·2⁻⁵³, plus the f32 cast's
    /// half-ULP 2⁻²⁴) is covered by the `next_up_f32` on the charge itself,
    /// which moves a FULL f32 ULP (2⁻²³).
    ///
    /// SOUND, and looser only: the published box strictly encloses the true real
    /// interval, so the cost of this term is at worst a Timeout, never a false
    /// proof.
    ///
    /// HONEST SCOPE — this closes a gap in the CERTIFICATE; it is not a patch for
    /// an observed wrong verdict. No shipped VNN-COMP row is known to exercise
    /// it. The uncharged residual `γ⁶⁴_k·S ≈ k·2⁻⁵³·S` only escapes what the
    /// pre-existing directed store already absorbed (`|y|·2⁻²⁴` for the cast's
    /// round-to-nearest half-ULP, `|y|·2⁻²³` for the full directed step) once
    /// `S/|y|` exceeds roughly `2²⁹/k` … `2³⁰/k` — about `7e5 … 1.4e6` at
    /// `k = 768`. The finder of this gap stated plainly that they could NOT
    /// construct such a row on a shipped benchmark, and neither can this change;
    /// it is fixed because a preset or a caller can make a latent hazard live.
    pub(super) fn propagate_ibp_standard(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        let dims = parse_matmul_dims(self.transpose_b, input_a.shape(), input_b.shape())?;

        let mut out_shape = dims.batch_dims.clone();
        out_shape.push(dims.m);
        out_shape.push(dims.n);

        let mut out_lower = ArrayD::zeros(IxDyn(&out_shape));
        let mut out_upper = ArrayD::zeros(IxDyn(&out_shape));

        let batch_size = dims.batch_size()?;
        let bd_len = dims.batch_dims.len();

        // Higham constants for the f64 dot-product accumulation; they depend only
        // on the contraction width, so hoist them out of the element loops.
        // `crown_single_gamma_n_f64` saturates to +inf for pathologically wide
        // contractions (`k·2^-53 ≥ 1`), which drives the charge below to +inf and
        // the row to `[-inf, +inf]` — degraded, never under-counted.
        let gamma = crown_single_gamma_n_f64(dims.k.saturating_add(1));
        let gamma_k = crown_single_gamma_n_f64(dims.k);
        let s_inflate = if gamma_k < 1.0 {
            1.0 / (1.0 - gamma_k)
        } else {
            f64::INFINITY
        };

        // Reusable index buffers: batch prefix + 2 suffix slots.
        // Eliminates batch_size × m × n × k clone allocations.
        let mut batch_scratch = Vec::with_capacity(bd_len);
        let mut idx_buf = Vec::with_capacity(bd_len + 2);

        for batch_idx in 0..batch_size {
            decode_batch_index_into(batch_idx, &dims.batch_dims, &mut batch_scratch)?;

            for i in 0..dims.m {
                for j in 0..dims.n {
                    // SOUNDNESS (#vnncomp-aw-soundness,
                    // #matmul-ibp-accumulation-charge): accumulate the interval
                    // dot product in f64 alongside the magnitude sum `S` that
                    // certifies the accumulation, then charge `γ⁶⁴_{k+1}·S_safe`
                    // outward and directed-round the final f64→f32 store. The
                    // directed store covers the CAST ONLY; the f64 accumulation
                    // residual is paid by the Higham term below. See the method
                    // doc comment for the derivation.
                    let mut sum_lower = 0.0_f64;
                    let mut sum_upper = 0.0_f64;
                    // Σ_l max_c |p_c|: dominates both Σ|prod_min| and Σ|prod_max|
                    // termwise. Non-negative addends, so this can never become NaN
                    // and never underflows to 0 unless every product is exactly 0.
                    let mut abs_sum = 0.0_f64;

                    for l in 0..dims.k {
                        // Get A[..., i, l]
                        idx_buf.clear();
                        idx_buf.extend_from_slice(&batch_scratch);
                        idx_buf.push(i);
                        idx_buf.push(l);
                        let a_l = input_a.lower()[idx_buf.as_slice()];
                        let a_u = input_a.upper()[idx_buf.as_slice()];

                        // Get B[..., l, j] or B[..., j, l] if transposed
                        idx_buf.clear();
                        idx_buf.extend_from_slice(&batch_scratch);
                        if self.transpose_b {
                            idx_buf.push(j);
                            idx_buf.push(l);
                        } else {
                            idx_buf.push(l);
                            idx_buf.push(j);
                        }
                        let b_l = input_b.lower()[idx_buf.as_slice()];
                        let b_u = input_b.upper()[idx_buf.as_slice()];

                        // Interval multiplication: [a_l, a_u] * [b_l, b_u].
                        // EXACT f32→f64 widening before multiplying, so each f64
                        // product equals the true real product (no rounding); the
                        // accumulation then rounds only in f64, and that rounding
                        // is charged via `abs_sum` below.
                        let p1 = (a_l as f64) * (b_l as f64);
                        let p2 = (a_l as f64) * (b_u as f64);
                        let p3 = (a_u as f64) * (b_l as f64);
                        let p4 = (a_u as f64) * (b_u as f64);

                        // NaN-safe min/max: if ANY product is NaN (e.g., 0*inf),
                        // conservatively widen to [-inf, +inf] AND make the error
                        // budget +inf, so the charge below refuses rather than
                        // silently dropping the NaN term out of `abs_sum` (f64::max
                        // returns the non-NaN operand, which would UNDER-count).
                        let (prod_min, prod_max, prod_abs) =
                            if p1.is_nan() || p2.is_nan() || p3.is_nan() || p4.is_nan() {
                                (f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY)
                            } else {
                                let lo = p1.min(p2).min(p3.min(p4));
                                let hi = p1.max(p2).max(p3.max(p4));
                                // max_c |p_c| == max(|min_c p_c|, |max_c p_c|).
                                (lo, hi, lo.abs().max(hi.abs()))
                            };

                        abs_sum += prod_abs;

                        // NaN-safe accumulation: inf + -inf -> conservatively widen.
                        let new_lower = sum_lower + prod_min;
                        let new_upper = sum_upper + prod_max;
                        sum_lower = if new_lower.is_nan() {
                            f64::NEG_INFINITY
                        } else {
                            new_lower
                        };
                        sum_upper = if new_upper.is_nan() {
                            f64::INFINITY
                        } else {
                            new_upper
                        };
                    }

                    // Apply optional scaling (NaN-safe). Scale is exact-widened to
                    // f64; the f64 multiply is one additional rounding, covered by
                    // charging γ_{k+1} instead of the counted γ_{k-1} (see the doc
                    // comment). The error budget is scaled by |scale| with the
                    // endpoints so the charge stays in the OUTPUT's units.
                    if let Some(scale) = self.scale {
                        let scale = scale as f64;
                        if scale >= 0.0 {
                            sum_lower *= scale;
                            sum_upper *= scale;
                        } else {
                            let tmp = sum_lower;
                            sum_lower = sum_upper * scale;
                            sum_upper = tmp * scale;
                        }
                        if sum_lower.is_nan() {
                            sum_lower = f64::NEG_INFINITY;
                        }
                        if sum_upper.is_nan() {
                            sum_upper = f64::INFINITY;
                        }
                        abs_sum *= scale.abs();
                    }

                    // Certified f64-accumulation charge, applied OUTWARD to both
                    // endpoints before the directed store. `next_up_f32` on the
                    // f32 charge buys a full f32 ULP, which dominates the handful
                    // of 2^-53 roundings used to form it and the f32 cast's own
                    // half-ULP. A non-finite charge (overflow, `abs_sum = inf`
                    // from a NaN/inf product, or `0 * inf` when `s_inflate`
                    // saturated) degrades the row to [-inf, +inf] — fail-closed.
                    let err = next_up_f32((gamma * (abs_sum * s_inflate)) as f32);
                    if err.is_finite() {
                        let err = err as f64;
                        sum_lower -= err;
                        sum_upper += err;
                    } else {
                        sum_lower = f64::NEG_INFINITY;
                        sum_upper = f64::INFINITY;
                    }

                    // Directed-round OUTWARD at the f64→f32 store: `next_down_f32`
                    // for the lower endpoint, `next_up_f32` for the upper, so the
                    // stored f32 interval encloses the widened f64 interval despite
                    // the f64→f32 cast (both pass ±inf through unchanged). A full
                    // directed ULP also absorbs the round-to-nearest f64 subtract /
                    // add that applied the charge above (half an f64 ULP, 2^30×
                    // smaller). Non-finite endpoints stored as-is; repaired at type
                    // boundary.
                    idx_buf.clear();
                    idx_buf.extend_from_slice(&batch_scratch);
                    idx_buf.push(i);
                    idx_buf.push(j);
                    out_lower[idx_buf.as_slice()] = next_down_f32(sum_lower as f32);
                    out_upper[idx_buf.as_slice()] = next_up_f32(sum_upper as f32);
                }
            }
        }

        // Repair NaN/Inf at the type boundary instead of per-element.
        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }
}

#[cfg(test)]
mod accumulation_charge_tests {
    use super::{ArrayD, BoundedTensor, IxDyn, MatMulLayer};

    fn concrete(shape: &[usize], values: &[f32]) -> BoundedTensor {
        let array = ArrayD::from_shape_vec(IxDyn(shape), values.to_vec()).expect("shape");
        BoundedTensor::new(array.clone(), array).expect("concrete bounds")
    }

    /// The published box must contain the TRUE real dot product even when the
    /// f64 accumulation cancels catastrophically (#matmul-ibp-accumulation-charge).
    ///
    /// `a = [2^40, 1, 2^40]`, `b = [2^40, 1, -2^40]` (all f32-exact, all products
    /// f64-exact): the true value is `2^80 + 1 - 2^80 = 1.0`, but recursive f64
    /// summation gives `fl(2^80 + 1) = 2^80` (the ULP at `2^80` is `2^28`), then
    /// `2^80 - 2^80 = 0.0`. WITHOUT the Higham charge the row publishes
    /// `[next_down_f32(0), next_up_f32(0)] = [-1.4e-45, +1.4e-45]`, which
    /// EXCLUDES 1.0 — this assertion fails on that code. With the charge,
    /// `S ≈ 2^81` and `γ⁶⁴_4·S ≈ 1.1e9` swamps the residual.
    ///
    /// Calls `propagate_ibp_standard` directly so the economic-IBP dispatch in
    /// `propagate_ibp_binary` cannot silently move the row to another lane.
    #[ntest::timeout(10000)]
    #[test]
    fn ibp_standard_box_contains_catastrophically_cancelling_dot_product() {
        let big = 2.0_f32.powi(40);
        let matmul = MatMulLayer::new(false, None);
        let a = concrete(&[1, 3], &[big, 1.0, big]);
        let b = concrete(&[3, 1], &[big, 1.0, -big]);

        let out = matmul
            .propagate_ibp_standard(&a, &b)
            .expect("standard IBP must succeed");

        let truth = 1.0_f32;
        let lo = out.lower()[[0, 0]];
        let hi = out.upper()[[0, 0]];
        assert!(
            lo <= truth,
            "lower bound {lo} excludes the true product {truth}: the f64 \
             accumulation residual is uncharged",
        );
        assert!(
            truth <= hi,
            "upper bound {hi} excludes the true product {truth}: the f64 \
             accumulation residual is uncharged",
        );
    }

    /// The same charge must stay negligible on a well-conditioned row, so the
    /// fix cannot be mistaken for (or degrade into) a blanket widening.
    #[ntest::timeout(10000)]
    #[test]
    fn ibp_standard_charge_is_negligible_without_cancellation() {
        let matmul = MatMulLayer::new(false, None);
        let a = concrete(&[1, 2], &[1.0, 2.0]);
        let b = concrete(&[2, 1], &[3.0, 4.0]);

        let out = matmul
            .propagate_ibp_standard(&a, &b)
            .expect("standard IBP must succeed");

        let truth = 11.0_f32;
        let lo = out.lower()[[0, 0]];
        let hi = out.upper()[[0, 0]];
        assert!(
            lo <= truth && truth <= hi,
            "box [{lo}, {hi}] must hold 11.0"
        );
        assert!(
            hi - lo < 1e-3,
            "well-conditioned row widened to [{lo}, {hi}]; the Higham charge \
             should be far below one f32 ULP here",
        );
    }
}
