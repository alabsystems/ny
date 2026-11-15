// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Economic IBP propagation for MatMul (memory-efficient, potentially looser).
//!
//! "Economic" is about the SHAPE of the working set, not its absolute size: the
//! corner products are folded through a handful of `Array2::dot` GEMMs so the
//! per-`(i, j, l)` product tensor is never materialised. The Higham certificate
//! added in [`MatMulLayer::propagate_ibp_economic`] respects that: it costs one
//! extra array of each operand shape (`m×k` and `k×n`, on top of the ~12 of each
//! the loop already builds), one extra output-shaped array (`m×n`, on top of 3),
//! and one extra GEMM (8 rather than 7) — an `O(1)` factor on the existing terms,
//! with no `m·k·n` term introduced. Counted from the bodies below, not measured
//! on a box.

use ndarray::{Array2, ArrayD, IxDyn};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};

use super::helpers::{apply_scale, view_batch_2d};
use super::shape::{decode_batch_index_into, MatMulDims};
use super::{MatMulLayer, NyError, Result};
use crate::bounds::{nan_propagating_max, nan_propagating_max_zero, nan_propagating_min_zero};
use crate::layers::linear::crown_single_gamma_n_f32;

/// Bound on `(Σ over one endpoint's whole chain of |term|) / Σ_l Xm[i,l]·Ym[l,j]`,
/// where `Xm = max(|x_l|, |x_u|)` and `Ym = max(|y_l|, |y_u|)`.
///
/// Termwise (per `l`, so masks are honoured pointwise): `|x_l| , |y_l| ≤ Xm, Ym`;
/// an unmasked width is `dx = fl(x_u − x_l) ≤ 2·Xm·(1+u)`; and a masked width is
/// only `Xm·(1+u)`, because `mask_xn[l] = 1` forces `x_u < 0` (so
/// `dx ≤ |x_l| ≤ Xm`) and `mask_xp[l] = 1` forces `x_l > 0` (so `dx ≤ x_u ≤ Xm`)
/// — and symmetrically for `dy` under `mask_yn` / `mask_yp`. Masks are exact
/// `{0, 1}`, and `|mask_xpn| ≤ 1` even on an inverted box (where it is `−1`), so
/// masking never raises a term above its unmasked magnitude. Summing the six
/// dots that reach each endpoint:
///
/// ```text
/// lower:  base 1 + dx·y_l_neg 2 + dx_xn·y_l_ypn 1 + x_l_neg·dy 2
///                             + x_l_xpn·dy_yn 1 + dx_xn·dy_yn 1  =  8
/// upper:  base 1 + dx·y_l_pos 2 + dx_xp·y_l_ypn 1 + x_l_pos·dy 2
///                             + x_l_xpn·dy_yp 1 + dx_xpn_xp·dy_ypn_yp 4  =  11
/// ```
///
/// with a further `(1+u)²` for the two f32 subtractions forming `dx`/`dy`.
/// `11·(1+2^-24)² < 16` — the power of two is chosen for margin, not tightness.
///
/// # Why this also makes OVERFLOW fail closed
///
/// Every term that can move an endpoint INWARD carries factor exactly `1`: the
/// two subtracted dots (`dx_xn·y_l_ypn`, `x_l_xpn·dy_yn`) and the two dots added
/// to `upper` whose sign is forced non-positive (`dx_xp·y_l_ypn`,
/// `x_l_xpn·dy_yp`) are all masked on both operands, and `dx_xn·dy_yn` — the one
/// term added to `lower` that is non-negative — is masked on both too. So if any
/// inward-moving term overflows f32, its `Xm·Ym` counterpart is at least as
/// large and overflows with it, driving `abs_sum` to `+inf` and the element to
/// `[-inf, +inf]`. The factor-2 and factor-4 terms are exactly the ones that can
/// only ever move an endpoint OUTWARD, where an `inf` is safe by itself.
const ECONOMIC_TERM_MAGNITUDE_FACTOR: f64 = 16.0;

/// Higham index charged ON TOP of the `k` roundings that `γ_k` already covers
/// for one width-`k` f32 dot product.
///
/// Counted: `+5` for the five f32 add/sub that fold the six dot results into one
/// endpoint, `+1` for the `dx` subtraction, `+1` for the `dy` subtraction, `+1`
/// for the optional `scale` multiply — nine. Sixteen are charged, leaving seven
/// units of documented margin on top of the 16-vs-13 slack above.
const ECONOMIC_GAMMA_INDEX_MARGIN: usize = 16;

impl MatMulLayer {
    /// Economic (memory-efficient) interval forward for a two-perturbed-input
    /// MatMul, with a certified f32-accumulation error term.
    ///
    /// # Certified f32 accumulation charge (#matmul-ibp-economic-charge)
    ///
    /// Unlike the sibling [`MatMulLayer::propagate_ibp_standard`], which
    /// accumulates each output in f64 and pays only for its `f64→f32` cast, this
    /// path accumulates in **f32** — it folds the corner products into a handful
    /// of `Array2::dot` GEMMs precisely so it never materialises the per-`(i, j,
    /// l)` product tensor. `ndarray`'s f32 GEMM (and any BLAS `sgemm` behind it)
    /// keeps an f32 accumulator, so a width-`k` dot deviates from the true real
    /// sum by up to the Higham bound (Accuracy and Stability, Thm 3.1)
    ///
    /// ```text
    /// |fl(Σ_l t_l) − Σ_l t_l|  ≤  γ³²_k · Σ_l |t_l|,    γ³²_n = n·2^-24/(1 − n·2^-24)
    /// ```
    ///
    /// which is a MAGNITUDE-scaled charge and therefore survives cancellation —
    /// it is `2^29×` the f64 factor the standard path pays. Before this change
    /// the endpoints were stored with **no widening at all**: no `f64`, no
    /// directed rounding, no error term. Under cancellation (`x_l = [2^40, 1,
    /// 2^40]`, `y_l = [2^40, 1, −2^40]`: true `1.0`, f32 `0.0`) the published box
    /// EXCLUDED the true product, which a downstream ReLU stability test can turn
    /// into a false `unsat`.
    ///
    /// ## The magnitude sum
    ///
    /// `S[i,j] = Σ_l |X|max[i,l] · |Y|max[l,j]` with `|X|max = max(|x_l|, |x_u|)`
    /// dominates `Σ_l |t_l|` for EVERY dot in the chain up to the constant
    /// [`ECONOMIC_TERM_MAGNITUDE_FACTOR`]; see that constant for the term-by-term
    /// accounting. It is obtained by ONE extra GEMM on two extra operand-shaped
    /// magnitude arrays — the same "run the forward on the degenerate `max(|l|,
    /// |u|)` box" device used by `pooling::average::propagate_ibp_sound` and
    /// `conv2d`'s `|kernel|` pass. **The `|A|·|B|` product is never materialised**,
    /// so the property this path exists for is preserved: the loop already holds
    /// ~12 arrays of each operand shape and 3 of the output shape, and this adds
    /// one of each (see the module docs for the measured cost).
    ///
    /// `S` is itself computed by a round-to-nearest f32 GEMM over NON-NEGATIVE
    /// terms, so it can fall SHORT of the true magnitude sum by its own
    /// `γ³²_k·S`; `S_safe = S/(1 − γ³²_k)` inflates it outward (mirrors conv's
    /// `s_inflate`). `|X|max`/`|Y|max` themselves are exact — `abs` and `max`
    /// introduce no rounding.
    ///
    /// ## The charge
    ///
    /// Both endpoints are widened outward by
    /// `next_up_f32(γ³²_{k+16} · S_safe · 16 · |scale|)` AFTER `apply_scale`, so
    /// the charge lives in the OUTPUT's units. The index margin is accounted in
    /// [`ECONOMIC_GAMMA_INDEX_MARGIN`]; the `scale` multiply's own rounding
    /// (`u·|y|`, result-scaled) is dominated because the charge is at least
    /// `16·γ³²_16·S ≥ 256u·S ≫ 11u·S ≥ u·|y|`. The final `next_down_f32` /
    /// `next_up_f32` each move a FULL f32 ULP, which absorbs the round-to-nearest
    /// subtract/add that applies the charge (half an ULP) and the sub-ULP slop in
    /// forming it.
    ///
    /// A non-finite charge — GEMM overflow to `+inf`, a NaN operand, or `0·inf`
    /// when `γ`/`s_inflate` saturated on a pathologically wide contraction —
    /// degrades that element to `[-inf, +inf]`. **Fail-closed**: an uncertifiable
    /// element publishes the trivial box, never a bare f32 endpoint. This is what
    /// closes the pre-existing OVERFLOW hole as well: a chain that overflowed to
    /// `+inf` on the LOWER endpoint used to publish `[+inf, +inf]`, a box that
    /// excludes every real number and whose `lower > 0` would pin a downstream
    /// ReLU active, and `RepairStrategy::Conservative` does not catch it (it
    /// repairs NaN and swaps inversions; `[+inf, +inf]` is neither). See
    /// [`ECONOMIC_TERM_MAGNITUDE_FACTOR`] for why every inward-moving term is
    /// dominated by `Xm·Ym` with factor 1, so its overflow forces `abs_sum` to
    /// overflow too.
    ///
    /// SOUND, and looser only: the published box strictly encloses the true real
    /// interval, so the cost is at worst a Timeout, never a false proof.
    ///
    /// # Honest scope
    ///
    /// This closes a gap in the CERTIFICATE. `MatMulIbpMode::Economic` is set
    /// nowhere outside tests today (`MatMulLayer::new`/`try_new` default to
    /// `Standard`, and `should_use_economic_ibp` is a MODE check, not a size
    /// heuristic), so no shipped VNN-COMP row is known to reach this code. It is
    /// fixed because a preset or a caller can make a latent hazard live, and
    /// because an uncertified path must not sit first in a live dispatch.
    // Justification: Economic IBP for two-perturbed-input matmul requires both input bound
    // tensors, parsed dimensions struct, and output shape — these represent distinct
    // mathematical/structural parameters of the batched matmul computation.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn propagate_ibp_economic(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
        dims: &MatMulDims,
        out_shape: &[usize],
    ) -> Result<BoundedTensor> {
        let mut out_lower = ArrayD::zeros(IxDyn(out_shape));
        let mut out_upper = ArrayD::zeros(IxDyn(out_shape));

        let batch_size = dims.batch_size()?;

        // Higham constants for the f32 dot chain (#matmul-ibp-economic-charge).
        // They depend only on the contraction width, so hoist them out of the
        // batch loop. `crown_single_gamma_n_f32` saturates to +inf once
        // `n·2^-24 ≥ 1`, which drives the charge to +inf and every element of
        // such a row to [-inf, +inf] — degraded, never under-counted.
        let gamma = crown_single_gamma_n_f32(dims.k.saturating_add(ECONOMIC_GAMMA_INDEX_MARGIN));
        let gamma_k = crown_single_gamma_n_f32(dims.k);
        let s_inflate = if gamma_k < 1.0 {
            1.0 / (1.0 - gamma_k)
        } else {
            f64::INFINITY
        };
        // `apply_scale` multiplies both endpoints by `scale` (swapping them when
        // it is negative); the error budget is in the same units, so it is scaled
        // by |scale| with them.
        let scale_abs = self.scale.map_or(1.0_f64, |s| (s as f64).abs());

        // Reusable batch index buffer — eliminates batch_size clone allocations.
        let mut batch_scratch = Vec::with_capacity(dims.batch_dims.len());
        let mut idx_buf = Vec::with_capacity(dims.batch_dims.len() + 2);

        for batch_idx in 0..batch_size {
            decode_batch_index_into(batch_idx, &dims.batch_dims, &mut batch_scratch)?;

            let (a_lower, a_upper) =
                view_batch_2d(input_a, &batch_scratch, "MatMul IBP economic A")?;
            let (b_lower, b_upper) =
                view_batch_2d(input_b, &batch_scratch, "MatMul IBP economic B")?;

            if a_lower.nrows() != dims.m || a_lower.ncols() != dims.k {
                return Err(NyError::shape_mismatch(
                    vec![dims.m, dims.k],
                    vec![a_lower.nrows(), a_lower.ncols()],
                ));
            }

            let (b_rows, b_cols) = if self.transpose_b {
                (dims.n, dims.k)
            } else {
                (dims.k, dims.n)
            };
            if b_lower.nrows() != b_rows || b_lower.ncols() != b_cols {
                return Err(NyError::shape_mismatch(
                    vec![b_rows, b_cols],
                    vec![b_lower.nrows(), b_lower.ncols()],
                ));
            }

            let x_l = a_lower.to_owned();
            let x_u = a_upper.to_owned();
            let (y_l, y_u) = if self.transpose_b {
                (b_lower.t().to_owned(), b_upper.t().to_owned())
            } else {
                (b_lower.to_owned(), b_upper.to_owned())
            };

            // S[i,j] = Σ_l max(|x_l|,|x_u|)·max(|y_l|,|y_u|), the magnitude sum
            // that certifies the f32 dot chain below (#matmul-ibp-economic-charge).
            // `abs` and `max` are EXACT, so the ONLY rounding here is the f32 GEMM
            // itself, repaid by `s_inflate` at the charge site. Cost: two extra
            // operand-shaped arrays and one extra output-shaped GEMM — the
            // per-(i,j,l) |A|·|B| product is NEVER materialised, so this keeps the
            // memory shape the economic path exists for.
            //
            // `nan_propagating_max`, not `f32::max`: the latter returns the
            // non-NaN operand, which would silently UNDER-count a NaN endpoint
            // out of the budget. Propagating the NaN makes the charge non-finite
            // and the element degrade to [-inf, +inf] instead.
            let mut x_mag = x_l.mapv(f32::abs);
            ndarray::Zip::from(&mut x_mag)
                .and(&x_u)
                .for_each(|m, &u| *m = nan_propagating_max(*m, u.abs()));
            let mut y_mag = y_l.mapv(f32::abs);
            ndarray::Zip::from(&mut y_mag)
                .and(&y_u)
                .for_each(|m, &u| *m = nan_propagating_max(*m, u.abs()));
            let abs_sum = x_mag.dot(&y_mag);

            // NaN-propagating max(0) so NaN bounds poison rather than silently vanish.
            let dx = (&x_u - &x_l).mapv(nan_propagating_max_zero);
            let dy = (&y_u - &y_l).mapv(nan_propagating_max_zero);
            let base = x_l.dot(&y_l);

            let mask_xp = x_l.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
            let mask_xn = x_u.mapv(|v| if v < 0.0 { 1.0 } else { 0.0 });
            let mask_xpn = Array2::<f32>::from_elem(x_l.raw_dim(), 1.0) - &mask_xp - &mask_xn;

            let mask_yp = y_l.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
            let mask_yn = y_u.mapv(|v| if v < 0.0 { 1.0 } else { 0.0 });
            let mask_ypn = Array2::<f32>::from_elem(y_l.raw_dim(), 1.0) - &mask_yp - &mask_yn;

            let mut lower = base.clone();
            let mut upper = base;

            let y_l_neg = y_l.mapv(nan_propagating_min_zero);
            let y_l_pos = y_l.mapv(nan_propagating_max_zero);
            lower += &dx.dot(&y_l_neg);
            upper += &dx.dot(&y_l_pos);

            let dx_xn = &dx * &mask_xn;
            let dx_xp = &dx * &mask_xp;
            let y_l_ypn = &y_l * &mask_ypn;
            lower -= &dx_xn.dot(&y_l_ypn);
            upper += &dx_xp.dot(&y_l_ypn);

            let x_l_neg = x_l.mapv(nan_propagating_min_zero);
            let x_l_pos = x_l.mapv(nan_propagating_max_zero);
            lower += &x_l_neg.dot(&dy);
            upper += &x_l_pos.dot(&dy);

            let x_l_xpn = &x_l * &mask_xpn;
            let dy_yn = &dy * &mask_yn;
            let dy_yp = &dy * &mask_yp;
            lower -= &x_l_xpn.dot(&dy_yn);
            upper += &x_l_xpn.dot(&dy_yp);

            lower += &dx_xn.dot(&dy_yn);
            let dx_xpn_xp = &dx * (&mask_xpn + &mask_xp);
            let dy_ypn_yp = &dy * (&mask_ypn + &mask_yp);
            upper += &dx_xpn_xp.dot(&dy_ypn_yp);

            apply_scale(&mut lower, &mut upper, self.scale);

            // Certified f32-accumulation charge (#matmul-ibp-economic-charge),
            // applied OUTWARD to both endpoints AFTER `apply_scale` so it is in
            // the output's units. Every accumulation above reaches a published
            // endpoint through exactly this pair, so one charge per (i, j) covers
            // the whole chain — see the method doc comment for the derivation.
            // `next_down_f32`/`next_up_f32` move a full f32 ULP, which absorbs
            // the round-to-nearest subtract/add applying the charge. A non-finite
            // charge degrades that element to [-inf, +inf]: fail-closed, never a
            // bare uncertified endpoint.
            ndarray::Zip::from(&mut lower)
                .and(&mut upper)
                .and(&abs_sum)
                .for_each(|lo, up, &s| {
                    let s_safe =
                        (s as f64) * s_inflate * ECONOMIC_TERM_MAGNITUDE_FACTOR * scale_abs;
                    let err = next_up_f32((gamma * s_safe) as f32);
                    if err.is_finite() {
                        *lo = next_down_f32(*lo - err);
                        *up = next_up_f32(*up + err);
                    } else {
                        *lo = f32::NEG_INFINITY;
                        *up = f32::INFINITY;
                    }
                });

            for i in 0..dims.m {
                for j in 0..dims.n {
                    idx_buf.clear();
                    idx_buf.extend_from_slice(&batch_scratch);
                    idx_buf.push(i);
                    idx_buf.push(j);
                    out_lower[idx_buf.as_slice()] = lower[[i, j]];
                    out_upper[idx_buf.as_slice()] = upper[[i, j]];
                }
            }
        }

        // Repair NaN/Inf at the type boundary instead of per-batch.
        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }
}

#[cfg(test)]
mod economic_accumulation_charge_tests {
    use super::{ArrayD, BoundedTensor, IxDyn, MatMulLayer};
    use crate::layers::binary_ops::MatMulIbpMode;

    fn bounded(shape: &[usize], lower: &[f32], upper: &[f32]) -> BoundedTensor {
        let lo = ArrayD::from_shape_vec(IxDyn(shape), lower.to_vec()).expect("lower shape");
        let up = ArrayD::from_shape_vec(IxDyn(shape), upper.to_vec()).expect("upper shape");
        BoundedTensor::new(lo, up).expect("valid bounds")
    }

    fn economic() -> MatMulLayer {
        MatMulLayer::new_with_ibp_mode(false, None, MatMulIbpMode::Economic)
    }

    /// The published box must contain the TRUE real product even when the f32
    /// GEMM that forms `base = x_l · y_l` cancels catastrophically
    /// (#matmul-ibp-economic-charge).
    ///
    /// `k = 64` with `x_l = [2^40, 1×62, 2^40]` and `y_l = [2^40, 1×62, −2^40]`:
    /// every product is f32-EXACT, and their exact sum is `2^80 + 62 − 2^80 = 62`.
    /// An f32 accumulator cannot hold it: the ULP at `2^80` is `2^57`, so each of
    /// the 62 unit terms that lands in an accumulator already holding `±2^80` is
    /// swallowed whole. Sequentially the sum is `0.0`; even a 16-way blocked
    /// kernel (only the two accumulators holding the big terms lose their units)
    /// still returns `≤ 55`. The two big terms sit at the ENDS so no blocking can
    /// pair them off early.
    ///
    /// Both operands carry a `2^-10` perturbation on a UNIT entry — enough to
    /// take the economic branch (`is_perturbed` on both sides), small enough that
    /// the interval terms add only `2·2^-10`. So before the Higham charge the row
    /// published `[base, base + 2^-9]` ⊆ `[0, 55.002]`, which EXCLUDES 62.0 — and
    /// `62.0` is the value at the input box's own lower corner, so this is a live
    /// exclusion of a reachable point, not an approximation quibble. This
    /// assertion fails on the uncertified code.
    ///
    /// The ASSERTION is unconditionally correct — `62.0` is in the box's image
    /// whatever the backend does — but the demonstration that it fails today
    /// rests on `ndarray`'s GEMM accumulating in f32 in some blocked order. A
    /// hypothetical f64-accumulating or compensated backend would keep it green
    /// while proving nothing.
    /// [`economic_overflowing_magnitude_sum_fails_closed`] is the companion that
    /// does not depend on summation order at all.
    #[ntest::timeout(10000)]
    #[test]
    fn economic_box_contains_catastrophically_cancelling_dot_product() {
        const K: usize = 64;
        let big = (1u64 << 40) as f32;
        let delta = 1.0_f32 / 1024.0;

        let mut a_lower = vec![1.0_f32; K];
        a_lower[0] = big;
        a_lower[K - 1] = big;
        let mut a_upper = a_lower.clone();
        a_upper[1] = 1.0 + delta;

        let mut b_lower = vec![1.0_f32; K];
        b_lower[0] = big;
        b_lower[K - 1] = -big;
        let mut b_upper = b_lower.clone();
        b_upper[2] = 1.0 + delta;

        // True value at the input box's LOWER corner, summed in an order that is
        // exact in f64: the two ±2^80 products cancel to exactly 0, leaving the
        // 62 unit products. Both asserts pin the fixture, so a future edit that
        // breaks the construction fails loudly instead of weakening the test.
        let big_pair = (a_lower[0] as f64) * (b_lower[0] as f64)
            + (a_lower[K - 1] as f64) * (b_lower[K - 1] as f64);
        let small: f64 = a_lower[1..K - 1]
            .iter()
            .zip(&b_lower[1..K - 1])
            .map(|(&x, &y)| (x as f64) * (y as f64))
            .sum();
        assert_eq!(big_pair, 0.0, "fixture: the two big products must cancel");
        assert_eq!(big_pair + small, 62.0, "fixture: true product must be 62");
        let truth = 62.0_f32;

        let a = bounded(&[1, K], &a_lower, &a_upper);
        let b = bounded(&[K, 1], &b_lower, &b_upper);
        let layer = economic();
        assert!(
            layer.should_use_economic_ibp(&a, &b),
            "fixture must actually take the economic branch",
        );

        let out = layer
            .propagate_ibp_binary(&a, &b)
            .expect("economic IBP must succeed");
        let lo = out.lower()[[0, 0]];
        let hi = out.upper()[[0, 0]];

        assert!(
            lo <= truth,
            "lower bound {lo} excludes the true product {truth}: the f32 \
             accumulation residual is uncharged",
        );
        assert!(
            truth <= hi,
            "upper bound {hi} excludes the true product {truth}: the f32 \
             accumulation residual is uncharged",
        );
    }

    /// The charge must stay far below the interval width on a well-conditioned
    /// row, so the fix cannot be mistaken for (or decay into) a blanket widening.
    ///
    /// `x ∈ [1,1]×[2,2.5]`, `y ∈ [3,3.5]×[4,4]`: the exact product range is
    /// `[1·3 + 2·4, 1·3.5 + 2.5·4] = [11, 13.5]`, and the economic decomposition
    /// reproduces it exactly. `S = 1·3.5 + 2.5·4 = 13.5`, so the charge is
    /// `γ³²_18 · 16 · 13.5 ≈ 2.3e-4` per endpoint — four orders of magnitude
    /// under the 2.5-wide box.
    #[ntest::timeout(10000)]
    #[test]
    fn economic_charge_is_negligible_without_cancellation() {
        let a = bounded(&[1, 2], &[1.0, 2.0], &[1.0, 2.5]);
        let b = bounded(&[2, 1], &[3.0, 4.0], &[3.5, 4.0]);
        let layer = economic();
        assert!(layer.should_use_economic_ibp(&a, &b));

        let out = layer
            .propagate_ibp_binary(&a, &b)
            .expect("economic IBP must succeed");
        let lo = out.lower()[[0, 0]];
        let hi = out.upper()[[0, 0]];

        assert!(lo <= 11.0, "lower {lo} must enclose the true minimum 11.0");
        assert!(hi >= 13.5, "upper {hi} must enclose the true maximum 13.5");
        assert!(
            hi - lo < 2.501,
            "well-conditioned row widened to [{lo}, {hi}]; the Higham charge \
             should be ~2.3e-4 per endpoint here, not a blanket widening",
        );
    }

    /// A magnitude sum that OVERFLOWS f32 must fail closed to `[-inf, +inf]`.
    ///
    /// With `x_l = y_l = [1e38, 1e38]` every product overflows, so the
    /// uncertified `base = x_l · y_l` is `+inf` and BOTH endpoints were published
    /// as `+inf` — a box that excludes every real number, and whose `lower > 0`
    /// would pin a downstream ReLU active. `RepairStrategy::Conservative` does
    /// not catch it: it repairs NaN and swaps inversions, and `[+inf, +inf]` is
    /// neither. The charge's own `is_finite` guard is what makes this safe.
    #[ntest::timeout(10000)]
    #[test]
    fn economic_overflowing_magnitude_sum_fails_closed() {
        let huge = 1.0e38_f32;
        let wide = huge * 1.001;
        assert!(wide.is_finite(), "fixture: operands must be finite");

        let a = bounded(&[1, 2], &[huge, huge], &[huge, wide]);
        let b = bounded(&[2, 1], &[huge, huge], &[huge, wide]);
        let layer = economic();
        assert!(layer.should_use_economic_ibp(&a, &b));

        let out = layer
            .propagate_ibp_binary(&a, &b)
            .expect("economic IBP must succeed");
        let lo = out.lower()[[0, 0]];
        let hi = out.upper()[[0, 0]];

        assert!(
            lo == f32::NEG_INFINITY && hi == f32::INFINITY,
            "an uncertifiable row must degrade to [-inf, +inf], got [{lo}, {hi}]",
        );
    }
}
