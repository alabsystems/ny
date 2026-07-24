// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Property-based soundness tests for bound propagation.
//!
//! These tests verify that IBP bounds are sound: for any concrete input x
//! within the input bounds, the function output f(x) is within the output bounds.
//!
//! Note: A small tolerance (FP_TOLERANCE) is used to account for floating-point
//! precision errors. For strict mathematical soundness, use directed rounding
//! via `propagate_ibp_sound()` which applies `next_down/next_up` to bounds.
//!
//! ## Proptest case counts (issue #1716)
//!
//! Case counts are inversely proportional to per-case compute cost:
//! - 1000: cheap elementwise activations (Exp, Log, Sigmoid, etc.)
//! - 500: moderate (CROWN backward, linear, reductions)
//! - 300: heavier (batchnorm, transformer blocks)
//! - 200: multi-layer (network, add_constant)
//! - 100: directed rounding (exact arithmetic, few cases needed)
//!
//! ## Shrink time cap (issue #4141)
//!
//! All proptest configs set `max_shrink_time: 5000` (5 seconds). Without this,
//! proptest's default unlimited shrinking can spend minutes minimizing a
//! failing case, causing `ntest::timeout` to fire and hiding the original
//! failure message.

use ndarray::{Array1, Array2};
use proptest::prelude::*;

use crate::layers::activations::LinearRelaxation;

/// Tolerance for floating-point precision in soundness checks.
/// This accounts for FP rounding in both bound computation and function evaluation.
/// For strict soundness guarantees, use directed rounding (`propagate_ibp_sound`).
pub(crate) const FP_TOLERANCE: f32 = 1e-5;

/// Strategy to generate valid interval bounds [lower, upper] where lower <= upper.
/// Constrained to avoid extreme values that could cause overflow.
pub(crate) fn valid_interval(range: f32) -> impl Strategy<Value = (f32, f32)> {
    (-range..=range)
        .prop_flat_map(move |a| (-range..=range).prop_map(move |b| (a.min(b), a.max(b))))
}

/// Sample points within an interval for soundness verification.
/// Returns at least the interval endpoints.
pub(crate) fn sample_points(lower: f32, upper: f32, num_samples: usize) -> Vec<f32> {
    let (lower, upper) = if lower <= upper {
        (lower, upper)
    } else {
        (upper, lower)
    };
    if lower == upper {
        return vec![lower];
    }
    let samples = num_samples.max(2);
    let denom = (samples - 1) as f32;
    (0..samples)
        .map(|i| {
            let t = i as f32 / denom;
            let sample = lower + (upper - lower) * t;
            sample.clamp(lower, upper)
        })
        .collect()
}

/// Helper: compute softmax over a vector.
pub(crate) fn softmax(x: &Array1<f32>) -> Array1<f32> {
    let max_x = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_x: Array1<f32> = x.mapv(|xi| (xi - max_x).exp());
    let sum_exp = exp_x.sum();
    assert!(
        sum_exp.is_finite() && sum_exp > 0.0,
        "softmax normalization failed: sum_exp={}",
        sum_exp
    );
    exp_x.mapv(|ei| ei / sum_exp)
}

/// Helper: compute log-softmax over a vector.
/// log_softmax(x)_i = x_i - log(sum(exp(x)))
/// Uses log-sum-exp trick for numerical stability.
pub(crate) fn logsoftmax(x: &Array1<f32>) -> Array1<f32> {
    let max_x = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_shifted: Array1<f32> = x.mapv(|xi| (xi - max_x).exp());
    let lse = max_x + exp_shifted.sum().ln();
    x.mapv(|xi| xi - lse)
}

/// Helper: compute causal softmax over a 2D matrix [seq_q, seq_k].
/// Row i computes softmax over positions 0..=i, with zeros for j > i.
/// Matches the causal attention pattern where position i can only attend to past positions.
/// Uses SOFTMAX_EPSILON = 1e-12 for consistency with CausalSoftmaxLayer::eval_row.
pub(crate) fn causal_softmax(x: &Array2<f32>) -> Array2<f32> {
    let (seq_q, seq_k) = (x.nrows(), x.ncols());
    let mut out = Array2::zeros((seq_q, seq_k));
    for i in 0..seq_q {
        let active_len = (i + 1).min(seq_k);
        if active_len == 0 {
            continue;
        }
        let mut max_val = f32::NEG_INFINITY;
        for j in 0..active_len {
            max_val = max_val.max(x[[i, j]]);
        }
        let mut sum_exp = 0.0_f32;
        for j in 0..active_len {
            let e = (x[[i, j]] - max_val).exp();
            out[[i, j]] = e;
            sum_exp += e;
        }
        let inv_sum = 1.0 / (sum_exp + 1e-12);
        for j in 0..active_len {
            out[[i, j]] *= inv_sum;
        }
    }
    out
}

/// Helper: compute layer normalization over a vector.
pub(crate) fn layernorm(
    x: &Array1<f32>,
    ny: &Array1<f32>,
    beta: &Array1<f32>,
    eps: f32,
) -> Array1<f32> {
    let mean = x.mean().unwrap_or(0.0);
    let var = x.mapv(|xi| (xi - mean).powi(2)).mean().unwrap_or(0.0);
    let std = (var + eps).sqrt();
    x.mapv(|xi| (xi - mean) / std) * ny + beta
}

/// Helper: compute mean-only layer normalization over a vector.
///
/// MeanOnly LayerNorm: y_i = ny_i * (x_i - mean(x)) + beta_i
/// No variance normalization (no division by std).
pub(crate) fn layernorm_mean_only(
    x: &Array1<f32>,
    ny: &Array1<f32>,
    beta: &Array1<f32>,
) -> Array1<f32> {
    let mean = x.mean().unwrap_or(0.0);
    x.mapv(|xi| xi - mean) * ny + beta
}

/// Helper: compute RMS normalization over a vector.
///
/// RMSNorm: y_i = ny_i * x_i / sqrt(mean(x^2) + eps)
pub(crate) fn rms_norm(x: &Array1<f32>, ny: &Array1<f32>, eps: f32) -> Array1<f32> {
    let n = x.len() as f32;
    let mean_sq = x.iter().map(|&xi| xi * xi).sum::<f32>() / n;
    let rms = (mean_sq + eps).sqrt();
    x.iter()
        .zip(ny.iter())
        .map(|(&xi, &g)| g * xi / rms)
        .collect()
}

/// Helper: compute instance normalization for a single channel.
///
/// InstanceNorm1d (per channel): y_t = ny * (x_t - mean(x)) / sqrt(var(x) + eps) + beta
pub(crate) fn instance_norm_channel(
    x: &Array1<f32>,
    ny_c: f32,
    beta_c: f32,
    eps: f32,
) -> Array1<f32> {
    let mean = x.mean().unwrap_or(0.0);
    let n = x.len() as f32;
    let var = x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f32>() / n;
    let std = (var + eps).sqrt();
    x.mapv(|xi| ny_c * (xi - mean) / std + beta_c)
}

/// Helper: compute AdaIN1d evaluation for a single channel.
///
/// AdaIN1d: y = style_gamma * InstanceNorm(x, ny, beta, eps) + style_beta
pub(crate) fn adain_eval_channel(
    x: &Array1<f32>,
    ny_c: f32,
    beta_c: f32,
    style_ny_c: f32,
    style_beta_c: f32,
    eps: f32,
) -> Array1<f32> {
    let normed = instance_norm_channel(x, ny_c, beta_c, eps);
    normed.mapv(|z| style_ny_c * z + style_beta_c)
}

/// Helper: compute GroupNorm for a single group.
///
/// `group_vals` has length `cpg * time_len`, laid out as
/// [c0_t0, c0_t1, ..., c0_tT, c1_t0, ...].
/// `gammas`/`betas` have length `cpg` (per-channel within group).
/// Returns output of same length with per-element ny/beta applied.
pub(crate) fn group_norm_group(
    group_vals: &[f32],
    gammas: &[f32],
    betas: &[f32],
    cpg: usize,
    time_len: usize,
    eps: f32,
) -> Vec<f32> {
    let n = group_vals.len();
    let nf = n as f32;
    let mean: f32 = group_vals.iter().sum::<f32>() / nf;
    let var: f32 = group_vals
        .iter()
        .map(|&xi| (xi - mean).powi(2))
        .sum::<f32>()
        / nf;
    let std = (var + eps).sqrt();
    let mut output = vec![0.0_f32; n];
    for c_offset in 0..cpg {
        let g = gammas[c_offset];
        let b = betas[c_offset];
        for t in 0..time_len {
            let idx = c_offset * time_len + t;
            output[idx] = g * (group_vals[idx] - mean) / std + b;
        }
    }
    output
}

/// Helper: compute batch normalization for a single sample.
/// BatchNorm: y = ny * (x - mean) / std + beta
/// For inference, mean and std are the running statistics.
pub(crate) fn batchnorm(
    x: &Array1<f32>,
    ny: &Array1<f32>,
    beta: &Array1<f32>,
    running_mean: &Array1<f32>,
    running_var: &Array1<f32>,
    eps: f32,
) -> Array1<f32> {
    // Reference value computed in f64 to approximate the TRUE (real-arithmetic)
    // batchnorm y = gamma*(x-mean)/sqrt(var+eps) + beta. The IBP/CROWN soundness
    // contract is to enclose that real affine (and ny's f32 forward, which uses the
    // precomputed `x*scale+bias` factorization — see batch_norm/math.rs). An all-f32
    // reference in the DIFFERENT `(x-mean)/std*gamma+beta` factorization deviates from
    // the true value by several f32 ULPs (a property of that f32 proxy, NOT an IBP
    // under-widening: the bound encloses the real value and ny's forward to ~0.3 ULP,
    // while this proxy drifts ~2 ULP). Computing in f64 tests the bound against the
    // value it must actually bound. (#batchnorm-ibp-directed-rounding self-audit.)
    Array1::from_shape_fn(x.len(), |i| {
        let std = ((running_var[i] as f64) + eps as f64).sqrt();
        (((x[i] as f64 - running_mean[i] as f64) / std) * (ny[i] as f64) + beta[i] as f64) as f32
    })
}

/// SELU constants
pub(crate) const SELU_ALPHA: f32 = 1.673_263_2;
pub(crate) const SELU_LAMBDA: f32 = 1.050_701;

/// Softplus: ln(1 + exp(x))
/// Uses numerically stable computation to avoid overflow.
pub(crate) fn softplus_eval(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0_f32 + x.exp()).ln()
    }
}

/// Mish: x * tanh(softplus(x))
/// Uses numerically stable softplus computation to avoid overflow.
pub(crate) fn mish_eval(x: f32) -> f32 {
    x * softplus_eval(x).tanh()
}

/// HardSwish: x * max(0, min(1, (x + 3) / 6))
pub(crate) fn hardswish_eval(x: f32) -> f32 {
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        x
    } else {
        x * (x + 3.0) / 6.0
    }
}

/// SiLU (Swish): x * sigmoid(x)
pub(crate) fn silu_eval(x: f32) -> f32 {
    let s = sigmoid_eval(x);
    x * s
}

/// Sigmoid: 1 / (1 + exp(-x))
/// Uses the numerically stable two-branch formulation.
pub(crate) fn sigmoid_eval(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
/// Equivalent to 2*sigmoid(2x) - 1, uses std library for stability.
pub(crate) fn tanh_eval(x: f32) -> f32 {
    x.tanh()
}

/// GELU (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
/// Reference: Hendrycks & Gimpel 2016, "Gaussian Error Linear Units."
pub(crate) fn gelu_tanh_eval(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// GELU (exact via erf): 0.5 * x * (1 + erf(x / sqrt(2))).
pub(crate) fn gelu_erf_eval(x: f32) -> f32 {
    let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
    0.5 * x * (1.0 + libm::erff(x * inv_sqrt2))
}

/// Tolerance for CROWN relaxation envelope checks.
/// Relaxation functions may include small epsilon margins for numerical safety,
/// so we use a slightly looser tolerance than the raw FP_TOLERANCE.
pub(super) const CROWN_TOLERANCE: f32 = 1e-4;

/// Test a standalone relaxation function envelope over sampled points.
/// `f` is the true function, `relaxation_fn` returns a `LinearRelaxation`.
pub(super) fn assert_relaxation_envelope<F, R>(
    l: f32,
    u: f32,
    f: F,
    relaxation_fn: R,
    name: &str,
    tol: f32,
) -> Result<(), TestCaseError>
where
    F: Fn(f32) -> f32,
    R: Fn(f32, f32) -> LinearRelaxation,
{
    let relax = relaxation_fn(l, u);
    let (ls, li, us, ui) = (
        relax.lower_slope,
        relax.lower_intercept,
        relax.upper_slope,
        relax.upper_intercept,
    );

    for x in sample_points(l, u, 100) {
        let fx = f(x);
        let lower = ls * x + li;
        let upper = us * x + ui;

        // Scale tolerance with magnitude
        let scale_tol = tol * fx.abs().max(1.0);

        prop_assert!(
            lower <= fx + scale_tol,
            "{name} lower envelope violated on [{l}, {u}] at x={x}: \
             lower={lower} > f(x)={fx} (tol={scale_tol})"
        );
        prop_assert!(
            upper + scale_tol >= fx,
            "{name} upper envelope violated on [{l}, {u}] at x={x}: \
             upper={upper} < f(x)={fx} (tol={scale_tol})"
        );
    }
    Ok(())
}

/// Verify CROWN backward soundness for a single neuron using identity bounds.
///
/// With identity A and zero b, the CROWN output gives:
///   lower = lower_slope * x + lower_intercept (per neuron)
///   upper = upper_slope * x + upper_intercept (per neuron)
/// which is equivalent to testing the relaxation envelope directly.
pub(super) fn assert_crown_backward_sound<F>(
    l: f32,
    u: f32,
    f: F,
    crown_result: &crate::LinearBounds,
    name: &str,
    tol: f32,
) -> Result<(), TestCaseError>
where
    F: Fn(f32) -> f32,
{
    let ls = crown_result.lower_a[[0, 0]];
    let li = crown_result.lower_b[0];
    let us = crown_result.upper_a[[0, 0]];
    let ui = crown_result.upper_b[0];

    for x in sample_points(l, u, 100) {
        let fx = f(x);
        let lower = ls * x + li;
        let upper = us * x + ui;

        let scale_tol = tol * fx.abs().max(1.0);

        prop_assert!(
            lower <= fx + scale_tol,
            "{name} CROWN lower envelope violated on [{l}, {u}] at x={x}: \
             lower={lower} > f(x)={fx} (tol={scale_tol})"
        );
        prop_assert!(
            upper + scale_tol >= fx,
            "{name} CROWN upper envelope violated on [{l}, {u}] at x={x}: \
             upper={upper} < f(x)={fx} (tol={scale_tol})"
        );
    }
    Ok(())
}

/// Generic CROWN backward soundness check for 2-neuron elementwise layers.
///
/// `propagate_fn` takes (incoming, pre_activation) and returns LinearBounds.
/// `eval_fn` is the true scalar function for this activation.
pub(super) fn assert_crown_negative_coeff_sound<P, E>(
    pre_lower: [f32; 2],
    pre_upper: [f32; 2],
    incoming: &crate::LinearBounds,
    propagate_fn: P,
    eval_fn: E,
    name: &str,
    tol: f32,
) -> Result<(), TestCaseError>
where
    P: Fn(&crate::LinearBounds, &ny_tensor::BoundedTensor) -> crate::Result<crate::LinearBounds>,
    E: Fn(f32) -> f32,
{
    let [l0, l1] = pre_lower;
    let [u0, u1] = pre_upper;

    let pre_activation = ny_tensor::BoundedTensor::new(
        ndarray::arr1(&[l0, l1]).into_dyn(),
        ndarray::arr1(&[u0, u1]).into_dyn(),
    )
    .unwrap();

    let result = propagate_fn(incoming, &pre_activation)
        .map_err(|e| TestCaseError::fail(format!("{name} propagate failed: {e}")))?;

    let samples_0 = sample_points(l0, u0, 20);
    let samples_1 = sample_points(l1, u1, 20);

    for &x0 in &samples_0 {
        for &x1 in &samples_1 {
            let fx0 = eval_fn(x0);
            let fx1 = eval_fn(x1);
            // Compute reference in f64 to avoid f32 catastrophic cancellation
            // when large intermediate terms nearly cancel (e.g. PReLU with
            // slope=38, negative coefficients). The CROWN bounds must contain
            // the true f64 output, not just an f32 approximation of it.
            let incoming_lower_f64 = (incoming.lower_a[[0, 0]] as f64) * (fx0 as f64)
                + (incoming.lower_a[[0, 1]] as f64) * (fx1 as f64)
                + (incoming.lower_b[0] as f64);
            let incoming_upper_f64 = (incoming.upper_a[[0, 0]] as f64) * (fx0 as f64)
                + (incoming.upper_a[[0, 1]] as f64) * (fx1 as f64)
                + (incoming.upper_b[0] as f64);
            let incoming_lower = incoming_lower_f64 as f32;
            let incoming_upper = incoming_upper_f64 as f32;

            // Concretize CROWN result in f64 too — f32 accumulation of large
            // opposite-sign terms loses precision, masking real bound quality.
            let lb_f64 = (result.lower_a[[0, 0]] as f64) * (x0 as f64)
                + (result.lower_a[[0, 1]] as f64) * (x1 as f64)
                + (result.lower_b[0] as f64);
            let ub_f64 = (result.upper_a[[0, 0]] as f64) * (x0 as f64)
                + (result.upper_a[[0, 1]] as f64) * (x1 as f64)
                + (result.upper_b[0] as f64);
            let lb = lb_f64 as f32;
            let ub = ub_f64 as f32;

            // Scale tolerance by maximum intermediate magnitude, not output
            // magnitude. When CROWN backward-pass coefficients (stored in f32)
            // multiply by inputs producing large terms that nearly cancel, the
            // f32 coefficient rounding error propagates proportional to the
            // *largest term*, not the cancelled result.
            // Known limitation: CROWN backward pass uses f32 coefficient
            // products which lose precision with large slopes. Filed as a
            // separate bug for f64 backward accumulation.
            let max_intermediate = (result.lower_a[[0, 0]].abs() * x0.abs())
                .max(result.lower_a[[0, 1]].abs() * x1.abs())
                .max(result.upper_a[[0, 0]].abs() * x0.abs())
                .max(result.upper_a[[0, 1]].abs() * x1.abs())
                .max(1.0);
            let scale_tol = tol * max_intermediate;

            prop_assert!(
                lb <= incoming_lower + scale_tol,
                "{name} lower bound violated at ({x0}, {x1}): \
                 lb={lb} > incoming_lower={incoming_lower} (tol={scale_tol})"
            );
            prop_assert!(
                ub + scale_tol >= incoming_upper,
                "{name} upper bound violated at ({x0}, {x1}): \
                 ub={ub} < incoming_upper={incoming_upper} (tol={scale_tol})"
            );
        }
    }
    Ok(())
}

mod add_constant;
mod additional_activations;
mod advanced_activations;
mod aw_directed_rounding_soundness;
mod batched_compose_blas_soundness;
mod batchnorm;
mod clip_algorithms;
mod composition_mixer;
mod concretize_directed_soundness;
mod crown_activation_aw_soundness;
mod crown_alpha_coeff_err_carry;
mod crown_arithmetic;
mod crown_avgpool_aw_soundness;
mod crown_batched_equiv;
mod crown_beta_split_aw_soundness;
mod crown_bilinear;
mod crown_bilinear_asymmetric;
mod crown_binary_ops;
mod crown_budget_guards;
mod crown_compose;
mod crown_concat;
mod crown_conv1d_extended;
mod crown_conv_aw_soundness;
mod crown_convolution;
mod crown_cut_segment;
mod crown_decomposed_normalization;
mod crown_decomposed_rmsnorm;
mod crown_div_graph;
mod crown_domain_guard;
mod crown_domain_guard_batched;
mod crown_elementwise;
mod crown_elu_family;
mod crown_linear_aw_soundness;
mod crown_linear_multidomain_aw_soundness;
mod crown_logsumexp;
mod crown_maxpool_aw_soundness;
mod crown_merge_aw_soundness;
mod crown_multivariate;
mod crown_multivariate_asymmetric;
mod crown_normalization;
mod crown_normalization_asymmetric;
mod crown_normalization_batched;
mod crown_normalization_batched_asymmetric;
mod crown_normalization_batched_negcoeff;
mod crown_normalization_groupnorm;
mod crown_normalization_layernorm;
mod crown_obj_chunk;
mod crown_patches;
mod crown_patches_composition;
mod crown_patches_convtranspose;
mod crown_patches_pooling;
mod crown_piecewise;
mod crown_piecewise_asymmetric;
mod crown_piecewise_constant;
mod crown_piecewise_dual_alpha;
mod crown_piecewise_negcoeff;
mod crown_pooling;
mod crown_precomputed_ibp;
mod crown_rope;
mod crown_s_shaped;
mod crown_sincos;
mod crown_snake;
mod crown_softmax;
mod crown_softmax_family_batched;
mod crown_softmax_sliding;
mod crown_trig_reciprocal;
mod cumsum;
mod directed_rounding;
mod edge_case_ibp;
mod elementwise;
mod ibp_binary_ops;
mod infinite_bounds;
mod l2_constraint_soundness;
mod linear;
mod matmul;
mod matmul_crown_extended;
mod misc_layers;
mod nan_ibp;
mod network;
mod normalization_ibp;
mod normalization_ibp_extended;
mod normalization_ibp_groupnorm;
mod reductions;
mod reductions_extremum;
mod resize;
mod s_shaped_trig_ibp;
mod scatter_bounded;
mod self_attention;
mod sub_constant;
mod transformer;
mod transforms;
