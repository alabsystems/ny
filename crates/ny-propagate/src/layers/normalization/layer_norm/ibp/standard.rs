// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Standard interval bound propagation for LayerNorm.
//!
//! Handles `LayerNormMode::Standard` where `y_i = ny_i * (x_i - mean) / std + beta_i`.
//! Uses variance bounds, 4-corner normalized output bounds, and the theoretical
//! `[-sqrt(n-1), sqrt(n-1)]` clamping for tighter intervals.

use ndarray::Axis;
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};

use super::super::super::math_common::square_interval_bounds;
use super::super::types::LayerNormLayer;
use super::common::{mean_axis_f64_lower, mean_axis_f64_upper, IbpShapeContext};
use super::slices;
use crate::bounds::{nan_propagating_max, nan_propagating_min};

/// Standard interval propagation for LayerNorm.
///
/// Computes bounds on `y_i = ny_i * (x_i - mean) / std + beta_i` using
/// interval arithmetic over mean, variance, and normalized output ranges.
pub(super) fn propagate_interval(
    layer: &LayerNormLayer,
    input: &BoundedTensor,
    ctx: &IbpShapeContext,
) -> Result<BoundedTensor> {
    let shape = input.shape();
    let ndim = ctx.ndim;
    let norm_size = ctx.norm_size;

    // Compute bounds on mean using f64 accumulation with directed rounding.
    // Part of #2423.
    let mean_lower = mean_axis_f64_lower(input.lower(), Axis(ndim - 1)).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "LayerNorm: mean_axis failed for axis {} on {}D input",
            ndim - 1,
            ndim
        ))
    })?;
    let mean_upper = mean_axis_f64_upper(input.upper(), Axis(ndim - 1)).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "LayerNorm: mean_axis failed for axis {} on {}D input",
            ndim - 1,
            ndim
        ))
    })?;

    let has_nonfinite_mean = mean_lower
        .iter()
        .chain(mean_upper.iter())
        .any(|&v| !v.is_finite());
    if has_nonfinite_mean {
        return layer.fallback_output_bounds(shape);
    }

    let mut out_lower = input.lower().clone();
    let mut out_upper = input.upper().clone();

    if ndim == 1 {
        let mean_l = slices::mean_value_at(&mean_lower, &[], "standard 1D: mean_lower empty")?;
        let mean_u = slices::mean_value_at(&mean_upper, &[], "standard 1D: mean_upper empty")?;

        let lower_slice: Vec<f32> = (0..norm_size).map(|i| input.lower()[[i]]).collect();
        let upper_slice: Vec<f32> = (0..norm_size).map(|i| input.upper()[[i]]).collect();
        standard_ibp_1d_slice(
            layer,
            &lower_slice,
            &upper_slice,
            mean_l,
            mean_u,
            &mut |i, lo, hi| {
                out_lower[[i]] = lo;
                out_upper[[i]] = hi;
            },
        );
    } else {
        // Multi-dimensional case: normalize along last axis
        let bs = slices::batch_size(shape)?;
        let prefix_len = ndim - 1;
        let mut full_idx = [0usize; 8]; // stack buffer, part of #2237

        for batch_idx in 0..bs {
            slices::decode_batch_prefix_into(shape, batch_idx, &mut full_idx[..prefix_len]);

            let mean_l = slices::mean_value_at(
                &mean_lower,
                &full_idx[..prefix_len],
                "standard: mean_lower empty",
            )?;
            let mean_u = slices::mean_value_at(
                &mean_upper,
                &full_idx[..prefix_len],
                "standard: mean_upper empty",
            )?;

            let lower_slice =
                slices::collect_last_axis_slice(input.lower(), &full_idx[..prefix_len], norm_size);
            let upper_slice =
                slices::collect_last_axis_slice(input.upper(), &full_idx[..prefix_len], norm_size);

            // Capture prefix snapshot for the emit closure since full_idx is
            // reused across iterations. Part of #2237.
            let prefix_snapshot: [usize; 8] = full_idx;

            standard_ibp_1d_slice(
                layer,
                &lower_slice,
                &upper_slice,
                mean_l,
                mean_u,
                &mut |i, lo, hi| {
                    let mut emit_idx = prefix_snapshot;
                    emit_idx[prefix_len] = i;
                    out_lower[&emit_idx[..=prefix_len]] = lo;
                    out_upper[&emit_idx[..=prefix_len]] = hi;
                },
            );
        }
    }

    // Post-check: reject NaN in computed bounds. This should not happen with
    // finite inputs (eps > 0 prevents zero std), but catches unforeseen arithmetic
    // edge cases rather than silently replacing NaN with arbitrary finite values.
    // Category B per domain validation policy: NaN -> NumericalInstability error.
    if out_lower.iter().any(|v| v.is_nan()) || out_upper.iter().any(|v| v.is_nan()) {
        return Err(NyError::NumericalInstability(
            "LayerNorm IBP: NaN in computed bounds (possible 0/0 or inf/inf \
             in normalization arithmetic)"
                .to_string(),
        ));
    }

    // Repair non-finite endpoints only; keep valid finite bounds unchanged (#2549).
    BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
}

/// Core standard-mode IBP computation for a single 1D normalization slice.
///
/// Given input lower/upper slices, mean bounds, ny/beta, and eps:
/// 1. Compute variance bounds from `(x - mean)^2` intervals
/// 2. Compute std bounds = `sqrt(var + eps)`
/// 3. For each element, compute 4-corner normalized bounds
/// 4. Apply ny scaling and beta offset
///
/// Calls `emit(i, lower_i, upper_i)` for each output coordinate.
///
/// Extracted in #3043 / #2535: the 1D and multi-D paths previously inlined
/// identical variance/std/corners logic, differing only in array indexing.
fn standard_ibp_1d_slice(
    layer: &LayerNormLayer,
    input_lower: &[f32],
    input_upper: &[f32],
    mean_l: f32,
    mean_u: f32,
    emit: &mut impl FnMut(usize, f32, f32),
) {
    let norm_size = input_lower.len();
    let nf = norm_size as f32;

    // Variance bounds: Var = sum((x - mean)^2) / n
    // Uses shared square_interval_bounds for consistency with InstanceNorm/RmsNorm.
    // f64 accumulation for variance sum to prevent precision loss for large
    // norm_size (768+). Catastrophic cancellation in (xi - mean)^2 when
    // xi ~ mean is the primary concern. Part of #2423.
    let mut var_lower_f64 = 0.0_f64;
    let mut var_upper_f64 = 0.0_f64;

    for i in 0..norm_size {
        // Directed rounding on intermediate deviation: feeds variance via
        // square_interval_bounds -- too-narrow domain -> unsound. Part of #3344.
        let diff_l = next_down_f32(input_lower[i] - mean_u);
        let diff_u = next_up_f32(input_upper[i] - mean_l);
        let (sq_l, sq_u) = square_interval_bounds(diff_l, diff_u);
        var_lower_f64 += sq_l as f64;
        var_upper_f64 += sq_u as f64;
    }
    // Directed rounding: var_lower is denominator bound -> round down,
    // var_upper is range bound -> round up (#3270).
    let var_lower = next_down_f32((var_lower_f64 / nf as f64) as f32);
    let var_upper = next_up_f32((var_upper_f64 / nf as f64) as f32);

    // Directed rounding on sqrt: std_lower is denominator -> round DOWN,
    // std_upper is range bound -> round UP (wider interval = sound).
    // f64 intermediate for precision. Pattern from GroupNorm #3327, fix for #3332.
    let std_lower = next_down_f32(((var_lower as f64 + layer.eps as f64).sqrt()) as f32);
    let std_upper = next_up_f32(((var_upper as f64 + layer.eps as f64).sqrt()) as f32);

    // Max magnitude of x over the normalized group. Drives the f32 cancellation
    // margin below (the ULP of the largest |x| bounds the absolute error of the
    // centered numerator fl(x_i - mean) and of mean itself).
    let max_abs_x = input_lower
        .iter()
        .zip(input_upper.iter())
        .map(|(&l, &u)| l.abs().max(u.abs()))
        .fold(0.0_f32, f32::max);

    // Theoretical bound: in EXACT real arithmetic the centered-normalized values
    // satisfy |z_i| <= sqrt(n-1), because sum_i (x_i - mean) = 0 removes one
    // degree of freedom (the extremum is one outlier against n-1 equal others).
    // The interval-arithmetic over-approximation (treating x_i, mean, std as
    // independent) can exceed this, so clamping the corner ratios to
    // [-sqrt(n-1), sqrt(n-1)] tightens the bound (#3196).
    //
    // SOUNDNESS OF THE FLOAT MARGIN (the verifier must cover the *f32* inference
    // program, whose z deviates from the exact z, so the bare next_up(sqrt(n-1))
    // ceiling is too tight by ~ULP and can chop a reachable f32 output):
    //
    // The downstream kernel computes z_i = fl( fl(x_i - mean) / std ). A
    // spec-compliant naive f32 mean is a sequential sum of n terms of magnitude
    // ~M=max_abs_x, accumulating ~(n-1)*0.5*ulp(n*M) rounding; after /n that is
    // ~0.5*(n-1)*ulp(M), and the (x_i - mean) subtraction adds ~ulp(M). So the
    // centered-numerator absolute error is ~0.5*(n+1)*ulp(M), NOT ulp(M): a bare
    // ulp(max_abs_x) (no n factor) UNDER-counts it and is UNSOUND (validated
    // counterexample at n=10 realized numerator error 0.14375 vs budget
    // ulp(max)=0.0625). The (n+2) factor dominates the (n-1)-term summation plus
    // the subtraction. Dividing by std (>= std_lower) amplifies this by at most
    // 1/std_lower, base*EPSILON covers the final divide's relative error at
    // |z|<=base, and the divide rounding is <= 0.5*ulp(|z|) <= 0.5*ulp(sqrt(n-1)).
    // Hence
    //
    //   |z_i^{f32}| <= sqrt(n-1) + (n+2)*ulp(max_abs_x)/std_lower
    //                            + base*EPSILON + 0.5*ulp(sqrt(n-1)).
    //
    // The MEAN-SUBTRACTION (x-mean) cancellation is exactly why LayerNorm needs
    // the (n+2)*ulp(max_abs_x)/std_lower term; RMSNorm (no centering) does NOT,
    // and uses sqrt(n) with a denominator-relative margin instead. Every step
    // below rounds strictly OUTWARD:
    //   - std_lower is already next_down'd, so 1/std_lower over-estimates 1/std;
    //   - ulp_x, ulp_b are computed via next_up (>= the true ULPs);
    //   - their combination is next_up'd into delta;
    //   - base + delta is next_up'd into max_norm.
    // Therefore max_norm >= the true f32 ceiling, so clamping to
    // [-max_norm, max_norm] cannot exclude any reachable output => SOUND. It is
    // still strictly tighter than the unclamped 4-corner interval in the
    // high-|x| / low-variance regime, retaining the #3196 tightening benefit.
    //
    // Non-finite guard: if max_abs_x is Inf/NaN (Inf inputs), skip clamping by
    // setting max_norm = +INF so the unclamped interval is kept and handed to
    // the existing non-finite/NaN repair paths -- never tightened.
    let max_norm = if norm_size > 1 {
        if max_abs_x.is_finite() {
            let base = (((norm_size as f64) - 1.0).sqrt()) as f32; // norm_size > 1
            let ulp_x = next_up_f32(max_abs_x) - max_abs_x; // 1 ULP at max_abs_x
            let ulp_b = next_up_f32(base) - base; // 1 ULP at sqrt(n-1)
                                                  // (n+2) ULP(max|x|): a spec-compliant naive f32 mean is a sequential
                                                  // sum of n terms of magnitude ~M, accumulating ~(n-1)*0.5*ulp(n*M)
                                                  // error; after /n that is ~0.5*(n-1)*ulp(M), and the (x_i - mean)
                                                  // subtraction adds another ~ulp(M). A single ulp(max|x|) (no n
                                                  // factor) UNDER-counts this and is unsound. The (n+2) factor
                                                  // dominates the (n-1)-term mean summation plus the subtraction.
                                                  // base*EPSILON covers the final divide's relative error at |z|<=base;
                                                  // 0.5*ulp(base) the sqrt rounding.
            let delta =
                next_up_f32((nf + 2.0) * ulp_x / std_lower + base * f32::EPSILON + 0.5 * ulp_b);
            next_up_f32(base + delta)
        } else {
            f32::INFINITY
        }
    } else {
        0.0
    };

    // 4-corner normalized output bounds. The numerator (x_i - mean) ranges over
    // [centered_lower, centered_upper]; the denominator (std) ranges over
    // [std_lower, std_upper] with std > 0. For f(a, c) = a/c with c > 0,
    // extrema occur at the 4 corners of the (a, c) box, matching the pattern
    // used by InstanceNorm. The 8-corner approach previously used here evaluated
    // 4 redundant interior points of the numerator interval.
    for i in 0..norm_size {
        // Directed rounding on intermediate deviation: if centered_lower
        // rounds UP, the 4-corner domain is too narrow -> unsound. Part of #3344.
        let centered_lower = next_down_f32(input_lower[i] - mean_u);
        let centered_upper = next_up_f32(input_upper[i] - mean_l);

        let corners = [
            centered_lower / std_upper,
            centered_lower / std_lower,
            centered_upper / std_upper,
            centered_upper / std_lower,
        ];

        // NaN-propagating fold: if any corner is NaN (e.g., from 0/0 or Inf-Inf),
        // the bound must be NaN so downstream guards produce conservative bounds.
        // IEEE 754 f32::min/max silently absorb NaN -- see #2577.
        let norm_l = corners
            .iter()
            .fold(f32::INFINITY, |a, &b| nan_propagating_min(a, b));
        let norm_u = corners
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| nan_propagating_max(a, b));

        // Clamp to [-sqrt(n-1), sqrt(n-1)]. NaN must propagate through the clamp
        // so downstream guards fire correctly.
        let norm_l = nan_propagating_max(norm_l, -max_norm);
        let norm_u = nan_propagating_min(norm_u, max_norm);

        // Apply ny and beta
        let g = layer.ny[i];
        let b = layer.beta[i];

        // Directed rounding on final bounds: bare f32 mul+add from
        // corner division results. Part of #3344.
        if g >= 0.0 {
            emit(
                i,
                next_down_f32(g * norm_l + b),
                next_up_f32(g * norm_u + b),
            );
        } else {
            emit(
                i,
                next_down_f32(g * norm_u + b),
                next_up_f32(g * norm_l + b),
            );
        }
    }
}
