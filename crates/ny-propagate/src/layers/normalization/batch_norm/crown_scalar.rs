// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Scalar CROWN backward propagation for BatchNorm.

use ndarray::{Array2, Axis, Zip};
use ny_core::Result;
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::math::detect_input_layout;
use super::types::BatchNormLayer;
#[cfg(test)]
use crate::bounds::safe_mul_for_bounds;
use crate::bounds::safe_mul_for_bounds_f64;
use crate::LinearBounds;

/// One unit roundoff for round-to-nearest f32 (`2^-24`); the fresh multiply-error
/// factor for a single `safe_mul_for_bounds` column-scaling product.
const F32_U: f32 = 1.0 / (1u32 << 24) as f32;

impl BatchNormLayer {
    /// CROWN backward propagation through BatchNorm with shape information.
    ///
    /// BatchNorm is a linear operation: y_i = scale[c(i)] * x_i + bias[c(i)]
    /// where c(i) is the channel index for flattened position i.
    ///
    /// For CROWN backward:
    /// - Scale coefficient columns by scale[c(i)]
    /// - Add bias contribution: new_b = b + A @ bias_expanded
    /// - No swap for negative scale (CROWN composes by substitution, not IBP)
    ///
    /// # Vectorization (bit-identical to the scalar reference)
    ///
    /// This is the profiled cGAN/ConvTranspose CROWN-IBP hotspot. The previous
    /// implementation was a double scalar loop with `A[[out, i]]` 2D-indexed
    /// access repeated several times per `(out, i)` cell. This version computes
    /// the **exact same** bounds (bit-for-bit — see
    /// `test_propagate_linear_vectorized_matches_scalar_reference`) but:
    ///   * the column scaling `new_A = safe_mul(A, scale)` is a broadcast
    ///     multiply + zero-mask (SIMD, no per-cell branch dependency),
    ///   * the fresh/propagated coefficient-error pass is an element-wise
    ///     broadcast `Zip`,
    ///   * the bias-fold + OUTWARD widen accumulation reads each output row via a
    ///     single contiguous row view instead of re-indexing the 2D matrix, and
    ///     keeps the identical f64 accumulation ORDER and `safe_mul_for_bounds_f64`
    ///     / directed-rounding used by the scalar loop.
    ///
    /// SOUNDNESS: the outward error/margin terms (`scale_err`/`bias_err`/`w_err`,
    /// the coefficient-error widen, and the fresh multiply-rounding) are
    /// accumulated in f64 in exactly the same order with the same
    /// `safe_mul_for_bounds_f64` (0·∞ = 0) and the same OUTWARD `next_down_f32` /
    /// `next_up_f32` rounding as before. The result is bit-identical, so the
    /// bound is neither tightened nor loosened.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        let shape = pre_activation.shape();
        let num_inputs = bounds.num_inputs();
        let num_outputs = bounds.num_outputs();
        let layout = detect_input_layout(shape, self.num_channels, Some(num_inputs))?;
        let (expanded_scale, expanded_bias) = self.expand_scale_bias(&layout);

        // Per-input-position error weight `w_err[i] = scale_err·xmag_i + bias_err`,
        // folded OUTWARD into the bias below so the bound stays sound against the
        // *real* batchnorm despite the f32 precompute error baked into
        // `scale`/`bias`. The column scaling alone uses the f32-rounded scale; over
        // the input box (`pre_activation`, |x_in_i| ≤ xmag_i) the coefficient error
        // `|A_i|·scale_err·xmag_i` plus the bias error `|A_i|·bias_err` bounds the
        // total effect on the output, and folding it as a constant is sound through
        // all further backward composition (#batchnorm-ibp-directed-rounding, CROWN
        // counterpart). safe_mul keeps 0·∞ = 0 for degenerate channels.
        let (expanded_scale_err, expanded_bias_err) = self.expand_errs(&layout);
        let pre_l: Vec<f32> = pre_activation.lower().iter().copied().collect();
        let pre_u: Vec<f32> = pre_activation.upper().iter().copied().collect();
        let w_err: Vec<f64> = (0..num_inputs)
            .map(|i| {
                let xmag = pre_l
                    .get(i)
                    .copied()
                    .unwrap_or(0.0)
                    .abs()
                    .max(pre_u.get(i).copied().unwrap_or(0.0).abs());
                safe_mul_for_bounds_f64(xmag as f64, expanded_scale_err[i] as f64)
                    + expanded_bias_err[i] as f64
            })
            .collect();

        // Per-input-column f64 constants, hoisted out of the row loop. These are
        // bit-identical to reading `expanded_bias[i] as f64` (and its abs) inside
        // the loop, so the accumulation below is unchanged.
        let bias_f64: Vec<f64> = expanded_bias.iter().map(|&b| b as f64).collect();
        let abs_bias_f64: Vec<f64> = bias_f64.iter().map(|&b| b.abs()).collect();

        // Compute bias contributions BEFORE scaling (using original matrices).
        // Accumulate in f64 to avoid cancellation when many channels contribute.
        //
        // #cgan-conv-err-compose: with INCOMING certified coefficient error `e`
        // (the true incoming coefficient lies in `[a−e, a+e]`), two bias terms
        // must widen by the coefficient uncertainty:
        //   - the `A @ bias` contribution uses the stored `a`; the true
        //     contribution can differ by up to `e·|bias_i|`;
        //   - the precompute-error margin `|a|·w_err` under-counts the true
        //     magnitude by up to `e·w_err`.
        // Both are covered by using `(|a| + e)` / adding `e·|bias_i|` below.
        let in_lower_err = bounds.lower_a_err();
        let in_upper_err = bounds.upper_a_err();
        let lower_a = bounds.lower_a();
        let upper_a = bounds.upper_a();
        let mut new_lower_b_f64 = bounds.lower_b().mapv(|x| x as f64);
        let mut new_upper_b_f64 = bounds.upper_b().mapv(|x| x as f64);

        // Per-output-row f64 accumulation. Row-slice access (contiguous for the
        // C-order coefficient matrices) replaces the original `A[[out, i]]`
        // double-indexing; the arithmetic — accumulation ORDER,
        // `safe_mul_for_bounds_f64` (0·∞ = 0), f64 promotion, and the OUTWARD
        // widen (`-=` lower / `+=` upper) — is byte-for-byte the scalar loop.
        for out_row in 0..num_outputs {
            let la = lower_a.row(out_row);
            let ua = upper_a.row(out_row);
            let el_row = in_lower_err.map(|e| e.row(out_row));
            let eu_row = in_upper_err.map(|e| e.row(out_row));

            let mut lower_acc = new_lower_b_f64[out_row];
            let mut upper_acc = new_upper_b_f64[out_row];
            let mut widen_lower = 0.0f64;
            let mut widen_upper = 0.0f64;
            for i in 0..num_inputs {
                let la_i = la[i] as f64;
                let ua_i = ua[i] as f64;
                // safe_mul_for_bounds_f64 gives 0*inf=0 so a degenerate Inf/NaN
                // bias (BatchNorm channel with var+eps ~= 0) does not turn a zero
                // coefficient into NaN. Nonzero coefficient times an Inf bias
                // yields a ±Inf bias contribution, which concretize handles
                // soundly via the Inf-bias short-circuit.
                lower_acc += safe_mul_for_bounds_f64(la_i, bias_f64[i]);
                upper_acc += safe_mul_for_bounds_f64(ua_i, bias_f64[i]);

                let el = el_row.as_ref().map_or(0.0, |e| e[i] as f64);
                let eu = eu_row.as_ref().map_or(0.0, |e| e[i] as f64);
                let abs_bias = abs_bias_f64[i];
                // Outward precompute-error margin over the true-coefficient
                // magnitude (|a|+e), plus the bias-contribution uncertainty
                // e·|bias_i| (0·∞ = 0 safe).
                widen_lower += safe_mul_for_bounds_f64(la_i.abs() + el, w_err[i])
                    + safe_mul_for_bounds_f64(el, abs_bias);
                widen_upper += safe_mul_for_bounds_f64(ua_i.abs() + eu, w_err[i])
                    + safe_mul_for_bounds_f64(eu, abs_bias);
            }
            new_lower_b_f64[out_row] = lower_acc - widen_lower;
            new_upper_b_f64[out_row] = upper_acc + widen_upper;
        }

        // Scale coefficient matrices column-wise by scale (affine substitution).
        // No swap for s < 0: CROWN backward composes by substitution, not IBP.
        // Negative s just flips coefficient sign; downstream nonlinear relaxations
        // already branch on coefficient sign.
        // Reference: designs/2026-01-29-crown-affine-negative-scale.md
        //
        // Vectorized as a broadcast multiply with a zero-mask reproducing
        // `safe_mul_for_bounds` EXACTLY: whenever `a == 0` OR `s == 0` the product
        // is forced to `+0.0` (0*inf=0 and 0*NaN=0, and canonical +0.0 sign),
        // otherwise the plain `a * s` is the same single round-to-nearest f32
        // multiply the scalar path used. A degenerate Inf scale over a nonzero
        // coefficient yields ±Inf (concretize widens via CROWN_COEFF_MAX/Inf); a
        // NaN scale/coeff over a nonzero partner propagates NaN — identical to
        // `safe_mul_for_bounds`.
        let scale_row = expanded_scale.view().insert_axis(Axis(0));
        let scale_b = scale_row
            .broadcast((num_outputs, num_inputs))
            .expect("scale (1, num_inputs) broadcasts over output rows");
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_inputs));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_inputs));
        Zip::from(&mut new_lower_a)
            .and(lower_a)
            .and(scale_b)
            .for_each(|out, &a, &s| {
                *out = if a == 0.0 || s == 0.0 { 0.0 } else { a * s };
            });
        Zip::from(&mut new_upper_a)
            .and(upper_a)
            .and(scale_b)
            .for_each(|out, &a, &s| {
                *out = if a == 0.0 || s == 0.0 { 0.0 } else { a * s };
            });

        // SOUND BatchNorm coefficient error (#vnncomp-aw-soundness). The column
        // scaling `A_new[j,i] = A[j,i]·scale[c(i)]` is a single round-to-nearest f32
        // product (`safe_mul_for_bounds`), carrying a relative error of at most one
        // unit roundoff `u = 2^-24`: `|A_new - exact| ≤ |A_new|·u`. This is SEPARATE
        // from the precompute-error margin already folded into the bias above (which
        // covers the f32-rounded `scale`/`bias` *values*, not the multiply).
        //
        // #cgan-conv-err-compose: BatchNorm now PROPAGATES incoming certified
        // coefficient error (it is listed in `propagates_coeff_err`, query.rs).
        // The backward substitution is an EXACT per-column scaling, so the true
        // incoming coefficient interval `[a−e, a+e]` maps to
        // `[a·s − e·|s|, a·s + e·|s|]`: the propagated error is exactly `e·|s|`,
        // evaluated here in f64 (exact for f32 operands) and rounded OUTWARD
        // together with the fresh multiply-rounding term `|A_new|·u`. This replaces
        // the dispatcher-side discharge `Σ_p max(|y_p|)·e_p` over BatchNorm's own
        // output box — sound but needlessly loose: on cGAN-class conv→BN stacks
        // that discharge converted a u-scale relative coefficient error into an
        // absolute width penalty at intermediate-box magnitude (the 2.05× BN_5 /
        // 404× Conv_19 gap vs the exact affine composition). Carrying `e·|s|`
        // instead defers the fold to the network-input concretize, where the
        // penalty is `Σ_j e_j·max(|x_l|,|x_u|)` over the (small) input box — the
        // same enclosure, discharged over the provably tightest available box.
        // 0·∞ = 0 (safe_mul): an ∞-poisoned incoming err over a zero scale
        // contributes exactly 0 (the composed coefficient is exactly 0).
        //
        // Vectorized as an element-wise broadcast `Zip` (`|scale|` broadcast over
        // rows); bit-identical to the prior per-cell loop.
        let f32_u_f64 = F32_U as f64;
        let abs_scale_f64 = expanded_scale.mapv(|s| (s as f64).abs());
        let abs_scale_row = abs_scale_f64.view().insert_axis(Axis(0));
        let fresh_err = |a_new: &Array2<f32>, in_err: Option<&Array2<f32>>| -> Array2<f32> {
            let mut err = Array2::<f32>::zeros(a_new.raw_dim());
            let s_b = abs_scale_row
                .broadcast(a_new.raw_dim())
                .expect("|scale| (1, num_inputs) broadcasts over output rows");
            match in_err {
                Some(ie) => {
                    Zip::from(&mut err).and(a_new).and(s_b).and(ie).for_each(
                        |e, &a_new_ji, &abs_s, &in_err_ji| {
                            let fresh = (a_new_ji.abs() as f64) * f32_u_f64;
                            let prop = safe_mul_for_bounds_f64(in_err_ji as f64, abs_s);
                            *e = next_up_f32((fresh + prop) as f32);
                        },
                    );
                }
                None => {
                    Zip::from(&mut err).and(a_new).for_each(|e, &a_new_ji| {
                        let fresh = (a_new_ji.abs() as f64) * f32_u_f64;
                        *e = next_up_f32(fresh as f32);
                    });
                }
            }
            err
        };
        let lower_err = fresh_err(&new_lower_a, in_lower_err);
        let upper_err = fresh_err(&new_upper_a, in_upper_err);

        LinearBounds::new_or_conservative_with_err(
            new_lower_a,
            new_lower_b_f64.mapv(|x| next_down_f32(x as f32)),
            new_upper_a,
            new_upper_b_f64.mapv(|x| next_up_f32(x as f32)),
            lower_err,
            upper_err,
        )
    }

    /// Original scalar double-loop reference for `propagate_linear_with_bounds`.
    ///
    /// Retained verbatim as the differential oracle for
    /// `test_propagate_linear_vectorized_matches_scalar_reference`: the
    /// production path is a bit-identical vectorization of this body, so any drift
    /// (main terms or the outward error/margin terms) is caught by that test.
    #[cfg(test)]
    pub(super) fn propagate_linear_with_bounds_scalar_reference(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        let shape = pre_activation.shape();
        let num_inputs = bounds.num_inputs();
        let num_outputs = bounds.num_outputs();
        let layout = detect_input_layout(shape, self.num_channels, Some(num_inputs))?;
        let (expanded_scale, expanded_bias) = self.expand_scale_bias(&layout);

        let (expanded_scale_err, expanded_bias_err) = self.expand_errs(&layout);
        let pre_l: Vec<f32> = pre_activation.lower().iter().copied().collect();
        let pre_u: Vec<f32> = pre_activation.upper().iter().copied().collect();
        let w_err: Vec<f64> = (0..num_inputs)
            .map(|i| {
                let xmag = pre_l
                    .get(i)
                    .copied()
                    .unwrap_or(0.0)
                    .abs()
                    .max(pre_u.get(i).copied().unwrap_or(0.0).abs());
                safe_mul_for_bounds_f64(xmag as f64, expanded_scale_err[i] as f64)
                    + expanded_bias_err[i] as f64
            })
            .collect();

        let in_lower_err = bounds.lower_a_err();
        let in_upper_err = bounds.upper_a_err();
        let mut new_lower_b_f64 = bounds.lower_b().mapv(|x| x as f64);
        let mut new_upper_b_f64 = bounds.upper_b().mapv(|x| x as f64);
        for out_row in 0..num_outputs {
            let mut widen_lower = 0.0f64;
            let mut widen_upper = 0.0f64;
            for i in 0..num_inputs {
                new_lower_b_f64[out_row] += safe_mul_for_bounds_f64(
                    bounds.lower_a()[[out_row, i]] as f64,
                    expanded_bias[i] as f64,
                );
                new_upper_b_f64[out_row] += safe_mul_for_bounds_f64(
                    bounds.upper_a()[[out_row, i]] as f64,
                    expanded_bias[i] as f64,
                );
                let el = in_lower_err.map_or(0.0, |e| e[[out_row, i]] as f64);
                let eu = in_upper_err.map_or(0.0, |e| e[[out_row, i]] as f64);
                let abs_bias = (expanded_bias[i] as f64).abs();
                widen_lower += safe_mul_for_bounds_f64(
                    (bounds.lower_a()[[out_row, i]] as f64).abs() + el,
                    w_err[i],
                ) + safe_mul_for_bounds_f64(el, abs_bias);
                widen_upper += safe_mul_for_bounds_f64(
                    (bounds.upper_a()[[out_row, i]] as f64).abs() + eu,
                    w_err[i],
                ) + safe_mul_for_bounds_f64(eu, abs_bias);
            }
            new_lower_b_f64[out_row] -= widen_lower;
            new_upper_b_f64[out_row] += widen_upper;
        }

        let mut new_lower_a = bounds.lower_a().clone();
        let mut new_upper_a = bounds.upper_a().clone();

        for i in 0..num_inputs {
            let s = expanded_scale[i];
            for j in 0..num_outputs {
                new_lower_a[[j, i]] = safe_mul_for_bounds(new_lower_a[[j, i]], s);
                new_upper_a[[j, i]] = safe_mul_for_bounds(new_upper_a[[j, i]], s);
            }
        }

        let fresh_err = |a_new: &Array2<f32>, in_err: Option<&Array2<f32>>| -> Array2<f32> {
            let mut err = Array2::<f32>::zeros(a_new.raw_dim());
            for j in 0..num_outputs {
                for i in 0..num_inputs {
                    let fresh = (a_new[[j, i]].abs() as f64) * (F32_U as f64);
                    let prop = in_err.map_or(0.0, |e| {
                        safe_mul_for_bounds_f64(e[[j, i]] as f64, (expanded_scale[i] as f64).abs())
                    });
                    err[[j, i]] = next_up_f32((fresh + prop) as f32);
                }
            }
            err
        };
        let lower_err = fresh_err(&new_lower_a, in_lower_err);
        let upper_err = fresh_err(&new_upper_a, in_upper_err);

        LinearBounds::new_or_conservative_with_err(
            new_lower_a,
            new_lower_b_f64.mapv(|x| next_down_f32(x as f32)),
            new_upper_a,
            new_upper_b_f64.mapv(|x| next_up_f32(x as f32)),
            lower_err,
            upper_err,
        )
    }
}
