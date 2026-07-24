// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN backward propagation for BatchNorm.

use ndarray::{ArrayD, Axis, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::math::detect_input_layout;
use super::types::BatchNormLayer;
use crate::bounds::{safe_mul_for_bounds, safe_mul_for_bounds_f64};
use crate::BatchedLinearBounds;

impl BatchNormLayer {
    /// Batched CROWN backward propagation through BatchNorm.
    ///
    /// BatchNorm is a linear operation: y_i = scale[c(i)] * x_i + bias[c(i)]
    /// where c(i) is the channel index for flattened position i.
    ///
    /// For batched CROWN backward with incoming bounds A shape [batch.., out_dim, in_dim]:
    /// - Scale: new_A[.., :, i] = A[.., :, i] * scale[c(i)]  (column-wise)
    /// - Bias: new_b = b + sum_i(A[.., :, i] * bias[c(i)])  (f64 accumulation)
    ///
    /// Since BatchNorm is purely affine, the scale/bias mapping is identical
    /// across all batch positions — no per-batch loop needed.
    ///
    /// No upper/lower swap for negative scale: CROWN backward composes by
    /// substitution, not IBP. Downstream nonlinear relaxations branch on
    /// coefficient sign. Reference: designs/2026-01-29-crown-affine-negative-scale.md
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        let shape = pre_activation.shape();

        let a_shape = bounds.lower_a.shape();
        if a_shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        let in_dim = a_shape[a_shape.len() - 1];
        let a_ndim = a_shape.len();

        // Build expanded scale and bias for the flattened in_dim, reusing
        // the same channel-axis heuristic as the scalar CROWN path.
        let layout = detect_input_layout(shape, self.num_channels, Some(in_dim))?;
        let (expanded_scale, expanded_bias) = self.expand_scale_bias(&layout);

        // Column-wise scaling: new_A[.., :, i] = A[.., :, i] * scale[c(i)]
        // ndarray broadcasts [in_dim] against [batch.., out_dim, in_dim] on last axis.
        let scale_view = ArrayD::from_shape_vec(IxDyn(&[in_dim]), expanded_scale.to_vec())
            .map_err(|_| {
                NyError::InternalError("BatchNorm: cannot create scale broadcast".to_string())
            })?;
        // Use safe_mul_for_bounds (0*inf=0) instead of plain `*` so a degenerate
        // Inf scale (BatchNorm channel with var+eps ~= 0) does not turn a zero
        // coefficient into NaN. A zero incoming coefficient composes to exactly 0
        // regardless of scale; a nonzero coefficient times Inf yields a ±Inf
        // coefficient that concretize widens to ±inf via the CROWN_COEFF_MAX/Inf
        // short-circuit. Keeps the bound sound and avoids a NaN abort.
        let scale_b = scale_view
            .broadcast(bounds.lower_a.raw_dim())
            .ok_or_else(|| {
                NyError::InternalError("BatchNorm: cannot broadcast scale to A shape".to_string())
            })?;
        let mut new_lower_a = bounds.lower_a.clone();
        let mut new_upper_a = bounds.upper_a.clone();
        ndarray::Zip::from(&mut new_lower_a)
            .and(&scale_b)
            .for_each(|a, &s| *a = safe_mul_for_bounds(*a, s));
        ndarray::Zip::from(&mut new_upper_a)
            .and(&scale_b)
            .for_each(|a, &s| *a = safe_mul_for_bounds(*a, s));

        // Bias contribution: new_b = b + sum_col(A * bias), accumulated in f64.
        // Compute A @ bias along the last axis (in_dim), producing shape [batch.., out_dim].
        let bias_view =
            ArrayD::from_shape_vec(IxDyn(&[in_dim]), expanded_bias.to_vec()).map_err(|_| {
                NyError::InternalError("BatchNorm: cannot create bias broadcast".to_string())
            })?;
        let lower_a_f64 = bounds.lower_a.mapv(|x| x as f64);
        let upper_a_f64 = bounds.upper_a.mapv(|x| x as f64);
        let bias_f64 = bias_view.mapv(|x| x as f64);

        // safe_mul_for_bounds_f64 (0*inf=0) so a degenerate Inf/NaN bias does not
        // produce NaN from a zero coefficient. Mirrors the scale handling above.
        let bias_b = bias_f64.broadcast(lower_a_f64.raw_dim()).ok_or_else(|| {
            NyError::InternalError("BatchNorm: cannot broadcast bias to A shape".to_string())
        })?;
        let mut lower_bias_terms = lower_a_f64.clone();
        let mut upper_bias_terms = upper_a_f64.clone();
        ndarray::Zip::from(&mut lower_bias_terms)
            .and(&bias_b)
            .for_each(|a, &b| *a = safe_mul_for_bounds_f64(*a, b));
        ndarray::Zip::from(&mut upper_bias_terms)
            .and(&bias_b)
            .for_each(|a, &b| *a = safe_mul_for_bounds_f64(*a, b));
        let bias_contrib_lower = lower_bias_terms.sum_axis(Axis(a_ndim - 1));
        let bias_contrib_upper = upper_bias_terms.sum_axis(Axis(a_ndim - 1));

        // INCOMING certified coefficient error (may be absent). The true incoming
        // coefficient at [.., j, i] lies in `[a − el, a + el]`. Materialize as f64
        // arrays (zeros when absent) so the bias widenings and the propagated
        // `e·|scale|` term below can be computed elementwise exactly as the scalar
        // path (#cgan-conv-err-compose).
        let el_lower = bounds
            .lower_a_err
            .as_ref()
            .map(|e| e.mapv(|x| x as f64))
            .unwrap_or_else(|| ArrayD::<f64>::zeros(lower_a_f64.raw_dim()));
        let el_upper = bounds
            .upper_a_err
            .as_ref()
            .map(|e| e.mapv(|x| x as f64))
            .unwrap_or_else(|| ArrayD::<f64>::zeros(upper_a_f64.raw_dim()));

        // Outward fold of the f32 precompute error baked into `scale`/`bias`.
        // The stored coefficient `A_i·scale[c(i)]` differs from the exact real
        // affine by up to `|A_i|·scale_err[c(i)]` (on the x_in-relative coeff), and
        // the bias contribution by `|A_i|·bias_err[c(i)]`. The total effect on the
        // network output, over the input box (`pre_activation`, |x_in_i| ≤ xmag_i),
        // is bounded per output row by `Σ_i |A_i|·(scale_err·xmag_i + bias_err)`.
        // Folding that constant into the bias is sound through all further backward
        // composition (biases only add), so the bound stays sound against the real
        // batchnorm — the column scaling alone uses the f32-rounded scale and is
        // otherwise ~ulp-unsound at large |x| (#batchnorm-ibp-directed-rounding,
        // CROWN counterpart). safe_mul keeps 0·∞ = 0 for degenerate channels.
        //
        // #cgan-conv-err-compose: with INCOMING certified coefficient error `el`
        // (true incoming coeff in `[a − el, a + el]`), two bias terms widen by the
        // coefficient uncertainty, exactly as crown_scalar.rs: the precompute-error
        // margin uses the true-coefficient magnitude `(|a| + el)` instead of `|a|`,
        // and the `A @ bias` contribution can differ by up to `el·|bias_i|`.
        let (expanded_scale_err, expanded_bias_err) = self.expand_errs(&layout);
        let mut w_err = vec![0.0f64; in_dim];
        {
            let pre_l: Vec<f32> = pre_activation.lower().iter().copied().collect();
            let pre_u: Vec<f32> = pre_activation.upper().iter().copied().collect();
            for (i, w) in w_err.iter_mut().enumerate() {
                let xmag = pre_l
                    .get(i)
                    .copied()
                    .unwrap_or(0.0)
                    .abs()
                    .max(pre_u.get(i).copied().unwrap_or(0.0).abs());
                let scale_term = safe_mul_for_bounds_f64(xmag as f64, expanded_scale_err[i] as f64);
                *w = scale_term + expanded_bias_err[i] as f64;
            }
        }
        let w_err_arr = ArrayD::from_shape_vec(IxDyn(&[in_dim]), w_err).map_err(|_| {
            NyError::InternalError("BatchNorm: cannot create err-weight broadcast".to_string())
        })?;
        let w_err_b = w_err_arr.broadcast(lower_a_f64.raw_dim()).ok_or_else(|| {
            NyError::InternalError("BatchNorm: cannot broadcast err-weight to A shape".to_string())
        })?;
        // `|bias_i|` broadcast on the last (in_dim) axis, for the `el·|bias_i|`
        // bias-contribution-uncertainty term (mirrors crown_scalar.rs `abs_bias`).
        let abs_bias_arr = bias_f64.mapv(|b| b.abs());
        let abs_bias_b = abs_bias_arr
            .broadcast(lower_a_f64.raw_dim())
            .ok_or_else(|| {
                NyError::InternalError(
                    "BatchNorm: cannot broadcast abs-bias to A shape".to_string(),
                )
            })?;
        // Per-coeff widen term = (|a| + el)·w_err + el·|bias_i|. With el ≡ 0
        // (no incoming err) this collapses to the original |a|·w_err. 0·∞ = 0 safe.
        let mut widen_lower_terms = ArrayD::<f64>::zeros(lower_a_f64.raw_dim());
        ndarray::Zip::from(&mut widen_lower_terms)
            .and(&lower_a_f64)
            .and(&w_err_b)
            .and(&el_lower)
            .and(&abs_bias_b)
            .for_each(|out, &a, &w, &el, &ab| {
                *out = safe_mul_for_bounds_f64(a.abs() + el, w) + safe_mul_for_bounds_f64(el, ab);
            });
        let mut widen_upper_terms = ArrayD::<f64>::zeros(upper_a_f64.raw_dim());
        ndarray::Zip::from(&mut widen_upper_terms)
            .and(&upper_a_f64)
            .and(&w_err_b)
            .and(&el_upper)
            .and(&abs_bias_b)
            .for_each(|out, &a, &w, &el, &ab| {
                *out = safe_mul_for_bounds_f64(a.abs() + el, w) + safe_mul_for_bounds_f64(el, ab);
            });
        let widen_lower = widen_lower_terms.sum_axis(Axis(a_ndim - 1));
        let widen_upper = widen_upper_terms.sum_axis(Axis(a_ndim - 1));

        let new_lower_b = (&bounds.lower_b.mapv(|x| x as f64) + &bias_contrib_lower - &widen_lower)
            .mapv(|x| next_down_f32(x as f32));
        let new_upper_b = (&bounds.upper_b.mapv(|x| x as f64) + &bias_contrib_upper + &widen_upper)
            .mapv(|x| next_up_f32(x as f32));

        // SOUND BatchNorm coefficient error (#vnncomp-aw-soundness). Two ADDED
        // (never replaced) terms, rounded OUTWARD, mirroring crown_scalar.rs:
        //   - FRESH multiply-rounding: the column scaling `new_A = A·scale` is a
        //     single round-to-nearest f32 product (`safe_mul_for_bounds`), so
        //     `|new_A − exact| ≤ |new_A|·u`, u = 2^-24. SEPARATE from the
        //     precompute-error margin folded into the bias above.
        //   - PROPAGATED incoming err (#cgan-conv-err-compose): the backward
        //     substitution is an EXACT per-column scaling, so the incoming interval
        //     `[a − el, a + el]·s` maps EXACTLY to `[a·s − el·|s|, a·s + el·|s|]`;
        //     the propagated error is exactly `el·|s|`. Adding it here (instead of
        //     discharging incoming err over BN's output box in the dispatcher — the
        //     old loose path this replaces) defers the fold to the network-input
        //     concretize, matching the scalar path and the cGAN tightness fix. It is
        //     evaluated in f64 (exact for f32 operands). 0·∞ = 0 (safe_mul): an
        //     ∞-poisoned incoming err over a zero scale contributes exactly 0.
        const F32_U: f32 = 1.0 / (1u32 << 24) as f32;
        // `|scale|` broadcast on the last (in_dim) axis for the `el·|s|` prop term.
        let abs_scale_vals: Vec<f64> = expanded_scale.iter().map(|s| (*s as f64).abs()).collect();
        let abs_scale_arr =
            ArrayD::from_shape_vec(IxDyn(&[in_dim]), abs_scale_vals).map_err(|_| {
                NyError::InternalError("BatchNorm: cannot create abs-scale broadcast".to_string())
            })?;
        // Broadcast is shared by the lower and upper err loops: sound because
        // new_lower_a and new_upper_a are guaranteed the same shape (both cloned
        // from bounds.*_a, then scaled in place), so the neutral `_b` name.
        let abs_scale_b = abs_scale_arr
            .broadcast(new_lower_a.raw_dim())
            .ok_or_else(|| {
                NyError::InternalError(
                    "BatchNorm: cannot broadcast abs-scale to A shape".to_string(),
                )
            })?;
        let mut lower_err = ArrayD::<f32>::zeros(new_lower_a.raw_dim());
        ndarray::Zip::from(&mut lower_err)
            .and(&new_lower_a)
            .and(&el_lower)
            .and(&abs_scale_b)
            .for_each(|out, &a_new, &el, &abss| {
                let fresh = (a_new.abs() as f64) * (F32_U as f64);
                let prop = safe_mul_for_bounds_f64(el, abss);
                *out = next_up_f32((fresh + prop) as f32);
            });
        let mut upper_err = ArrayD::<f32>::zeros(new_upper_a.raw_dim());
        ndarray::Zip::from(&mut upper_err)
            .and(&new_upper_a)
            .and(&el_upper)
            .and(&abs_scale_b)
            .for_each(|out, &a_new, &el, &abss| {
                let fresh = (a_new.abs() as f64) * (F32_U as f64);
                let prop = safe_mul_for_bounds_f64(el, abss);
                *out = next_up_f32((fresh + prop) as f32);
            });

        let mut out = BatchedLinearBounds::new_or_conservative(
            new_lower_a,
            new_lower_b,
            new_upper_a,
            new_upper_b,
            bounds.input_shape.clone(),
            bounds.output_shape.clone(),
        )?;
        out.set_coeff_err(lower_err, upper_err);
        Ok(out)
    }
}
