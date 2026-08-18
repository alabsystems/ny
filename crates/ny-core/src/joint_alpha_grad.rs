// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! TRUE joint α-gradient for the sound resnet CROWN backward
//! (`docs/BATCHED_BAB_JOINT_ALPHA_GRADIENT.md`).
//!
//! The wide/batched CROWN backward folds a batch of BaB sub-domains' lower-bound
//! coefficient frontiers through a shared-weight resnet; the per-ReLU lower
//! relaxation slope `α ∈ [0,1]` is the only free variable steering the lower
//! bound. To re-optimize `α` per sub-domain we need `∂(bound)/∂α`. NY has no
//! autograd, so this module computes the gradient by the hand-derived reverse-mode
//! adjoint (§2 of the design doc), FD-validated in `scripts/validate_joint_*`.
//!
//! **This is a pure, non-soundness-critical computation.** It reads the network
//! (shared weights + per-ReLU slopes baked into the `Activation` layers) and the
//! input box, and returns a gradient. The gradient only proposes the *next* `α`;
//! every `α ∈ [0,1]` is a valid ReLU lower relaxation and the verdict bound is
//! always recomputed by the sound fold, so a wrong gradient can only loosen — never
//! unsound (design doc §4). Hence the coefficient-channel-only adjoint here
//! (certified-error channel deliberately omitted) is safe.
//!
//! ## What it computes (design doc §1–§2)
//!
//! Forward, output→input over the backward-order segments, tracking the lower
//! coefficient `A` (num_specs × dim) and bias `b`:
//! - Linear `A' = A·W`, `b += A·bias`.
//! - Conv2d `A' = A ⊛ Wᵀ` (transposed conv), `b += A·bias_expanded`.
//! - ReLU: per `(s,i)` σ = lower_slope if `A[s,i] ≥ 0` else upper_slope; τ =
//!   lower_intercept / upper_intercept correspondingly. `A'[s,i] = A[s,i]·σ`,
//!   `b[s] += A[s,i]·τ`. α = the lower_slope.
//! - Residual `A_in = A_F + A_skip`; ResidualProj `A_in = A_F + A_P`.
//! - Concretize `bound[s] = b[s] + Σ_j φ(A⁰[s,j])`, φ = `A·x_l` if `A≥0` else `A·x_u`.
//!
//! Adjoint, input→output (reverse of the fold), carrying `Ā` (same shape as `A`)
//! and the constant bias adjoint `adj_b = 1`:
//! - Seed `Ā⁰[s,j] = ξ[s,j] = (A⁰[s,j] ≥ 0 ? x_l[j] : x_u[j])`.
//! - Linear `Ā_in = Ā_out·Wᵀ + bias` (the `+ bias` is the **bias channel**).
//! - Conv2d `Ā_in = Ā_out ⊛ W + bias_expanded`.
//! - ReLU: harvest `grad_α[i] += Σ_s Ā_out[s,i]·max(A_preᵏ[s,i], 0)`, then
//!   propagate `Ā_in[s,i] = Ā_out[s,i]·σ + τ` (`+ τ` = the bias channel).
//! - Residual: `Ā_out = Ā_in + adjoint_F(Ā_in)` (skip fan-out). ResidualProj:
//!   `Ā_out = adjoint_F(Ā_in) + adjoint_P(Ā_in)`.
//!
//! **Dropping the bias channel** (the `+ bias` / `+ τ` terms) yields a
//! systematically-wrong (≈0.7× in a 2-ReLU test) gradient — still sound but a poor
//! direction. It is toggled by [`JointGradConfig::bias_channel`] purely so the
//! FD unit test can demonstrate the degradation.

use crate::gemm::{GpuCrownLayer, GpuResnetSegment};

/// Configuration for [`joint_alpha_gradient`].
#[derive(Clone, Copy, Debug)]
pub struct JointGradConfig {
    /// Include the bias channel (`+ bias` after each linear/conv adjoint, `+ τ`
    /// after each ReLU adjoint). MUST be `true` in production — `false` only for
    /// the FD degradation test (design doc §2).
    pub bias_channel: bool,
}

impl Default for JointGradConfig {
    fn default() -> Self {
        Self { bias_channel: true }
    }
}

/// A frozen per-ReLU forward record (the reverse-mode checkpoint of design doc
/// §"Intermediates to store"): the PRE-transform lower coefficient `A_preᵏ`
/// (num_specs × nn) and the sign-selected slope/intercept (σ, τ).
struct ReluRec {
    a_pre: Vec<f32>, // num_specs * nn, row-major
    sigma: Vec<f32>, // num_specs * nn
    tau: Vec<f32>,   // num_specs * nn
    nn: usize,
}

/// The frozen forward fold: the per-ReLU checkpoints (fold order) plus the folded
/// input-level lower coefficient `A⁰` and the running bias (used for the optional
/// self-concretized bound in tests).
struct Forward {
    relus: Vec<ReluRec>,
    a0: Vec<f32>, // num_specs * input_dim
    b: Vec<f32>,  // num_specs (bias accumulator after the whole fold)
}

/// Compute the TRUE joint α-gradient for one domain's resnet: `∂(lower_bound)/∂α`
/// for every ReLU neuron, returned per ReLU in **fold order** (the same order the
/// backward walk / extractor enumerates `Activation` layers), each a `Vec<f32>` of
/// length `num_neurons`.
///
/// `segments` are the backward-order (output→input) resnet segments with this
/// domain's α baked into the `Activation` layers' `lower_slope`. `seed_lower_a`
/// (num_specs × output_dim, row-major) + `seed_lower_b` (num_specs) are the spec
/// frontier at the network output (the same seed the sound fold starts from).
/// `input_lower/upper` are this domain's input box.
///
/// The gradient matches central finite differences of the sound fold's lower bound
/// (coefficient channel; the small certified-error channel is omitted — sound per
/// design doc §4). Returns `None` on any shape inconsistency (caller falls back to
/// the local rule / no α step — never unsound).
pub fn joint_alpha_gradient(
    segments: &[GpuResnetSegment],
    seed_lower_a: &[f32],
    seed_lower_b: &[f32],
    num_specs: usize,
    output_dim: usize,
    input_lower: &[f32],
    input_upper: &[f32],
    cfg: JointGradConfig,
) -> Option<Vec<Vec<f32>>> {
    if num_specs == 0 || output_dim == 0 {
        return None;
    }
    if seed_lower_a.len() != num_specs * output_dim || seed_lower_b.len() != num_specs {
        return None;
    }
    let input_dim = input_lower.len();
    if input_dim == 0 || input_upper.len() != input_dim {
        return None;
    }

    // ---- forward fold (coefficient + bias, no certified error) ----
    let mut fwd = Forward {
        relus: Vec::new(),
        a0: Vec::new(),
        b: seed_lower_b.to_vec(),
    };
    let mut a = seed_lower_a.to_vec();
    let mut dim = output_dim;
    for seg in segments {
        let (na, nd) = fold_segment_forward(seg, &a, &mut fwd.b, num_specs, dim, &mut fwd.relus)?;
        a = na;
        dim = nd;
    }
    if dim != input_dim {
        return None; // fold did not reach the input box
    }
    fwd.a0 = a;

    // ---- seed the adjoint at the input: ξ ----
    let mut abar = vec![0.0f32; num_specs * input_dim];
    for s in 0..num_specs {
        for j in 0..input_dim {
            let a0 = fwd.a0[s * input_dim + j];
            abar[s * input_dim + j] = if a0 >= 0.0 {
                input_lower[j]
            } else {
                input_upper[j]
            };
        }
    }

    // ---- adjoint pass (input→output), harvesting each ReLU's gradient ----
    let mut grads: Vec<Vec<f32>> = fwd.relus.iter().map(|r| vec![0.0f32; r.nn]).collect();
    // ReLU records are consumed in EXACT reverse of the forward fold order; the
    // cursor walks down from the end as the reversed structural walk reaches each
    // ReLU (see `adjoint_*`).
    let mut cursor = fwd.relus.len();
    let _out = adjoint_segments(
        segments,
        abar,
        num_specs,
        input_dim,
        &fwd.relus,
        &mut cursor,
        &mut grads,
        cfg,
    )?;
    debug_assert_eq!(cursor, 0, "every ReLU record must be consumed exactly once");
    if cursor != 0 {
        return None;
    }
    Some(grads)
}

/// The joint fold's own concretized LOWER bound (coefficient channel only, no
/// certified error). This is the exact function whose α-gradient
/// [`joint_alpha_gradient`] computes; exposed so the GPU FD test can cross-check
/// that this fold tracks the sound GPU serial bound (up to the small omitted error
/// channel) and thus that the frozen signs the adjoint uses are the GPU's. Returns
/// one lower bound per spec row, or `None` on a shape inconsistency.
pub fn joint_lower_bound_debug(
    segments: &[GpuResnetSegment],
    seed_lower_a: &[f32],
    seed_lower_b: &[f32],
    num_specs: usize,
    output_dim: usize,
    input_lower: &[f32],
    input_upper: &[f32],
) -> Option<Vec<f32>> {
    if num_specs == 0 || output_dim == 0 {
        return None;
    }
    if seed_lower_a.len() != num_specs * output_dim || seed_lower_b.len() != num_specs {
        return None;
    }
    let input_dim = input_lower.len();
    if input_dim == 0 || input_upper.len() != input_dim {
        return None;
    }
    let mut b = seed_lower_b.to_vec();
    let mut a = seed_lower_a.to_vec();
    let mut dim = output_dim;
    let mut relus = Vec::new();
    for seg in segments {
        let (na, nd) = fold_segment_forward(seg, &a, &mut b, num_specs, dim, &mut relus)?;
        a = na;
        dim = nd;
    }
    if dim != input_dim {
        return None;
    }
    let mut out = b;
    for s in 0..num_specs {
        for j in 0..input_dim {
            let av = a[s * input_dim + j];
            out[s] += if av >= 0.0 {
                av * input_lower[j]
            } else {
                av * input_upper[j]
            };
        }
    }
    Some(out)
}

/// f64 twin of [`joint_lower_bound_debug`]: the SAME error-free forward fold +
/// concretize, but every accumulator is `f64` (weights/inputs read as f32 and
/// promoted). This is the IDEAL-arithmetic reference used to diagnose the sound
/// GPU fold's certified-error tax — `f64_fold − f32_fold` is the ACTUAL f32
/// rounding drift of the whole backward, and `f64_fold − gpu_lb` is the total
/// sound conservatism. Forward-only (no ReLU checkpoints / adjoint) — it returns
/// only the concretized lower bound per spec row. `None` on shape inconsistency.
pub fn joint_lower_bound_debug_f64(
    segments: &[GpuResnetSegment],
    seed_lower_a: &[f32],
    seed_lower_b: &[f32],
    num_specs: usize,
    output_dim: usize,
    input_lower: &[f32],
    input_upper: &[f32],
) -> Option<Vec<f64>> {
    if num_specs == 0 || output_dim == 0 {
        return None;
    }
    if seed_lower_a.len() != num_specs * output_dim || seed_lower_b.len() != num_specs {
        return None;
    }
    let input_dim = input_lower.len();
    if input_dim == 0 || input_upper.len() != input_dim {
        return None;
    }
    let mut b: Vec<f64> = seed_lower_b.iter().map(|&x| f64::from(x)).collect();
    let mut a: Vec<f64> = seed_lower_a.iter().map(|&x| f64::from(x)).collect();
    let mut dim = output_dim;
    for seg in segments {
        let (na, nd) = fold_segment_forward_f64(seg, &a, &mut b, num_specs, dim)?;
        a = na;
        dim = nd;
    }
    if dim != input_dim {
        return None;
    }
    let mut out = b;
    for s in 0..num_specs {
        for j in 0..input_dim {
            let av = a[s * input_dim + j];
            out[s] += if av >= 0.0 {
                av * f64::from(input_lower[j])
            } else {
                av * f64::from(input_upper[j])
            };
        }
    }
    Some(out)
}

/// f64 forward fold of one segment (twin of [`fold_segment_forward`], no ReLU
/// checkpoints — the ideal-arithmetic reference bound only).
fn fold_segment_forward_f64(
    seg: &GpuResnetSegment,
    a: &[f64],
    b: &mut [f64],
    num_specs: usize,
    dim: usize,
) -> Option<(Vec<f64>, usize)> {
    match seg {
        GpuResnetSegment::Chain(layers) => fold_chain_forward_f64(layers, a, b, num_specs, dim),
        GpuResnetSegment::Residual(f) => {
            let a_skip = a.to_vec();
            let (a_f, dim_f) = fold_chain_forward_f64(f, a, b, num_specs, dim)?;
            if dim_f != dim {
                return None;
            }
            let merged: Vec<f64> = a_skip.iter().zip(a_f.iter()).map(|(x, y)| x + y).collect();
            Some((merged, dim))
        }
        GpuResnetSegment::ResidualProj(f, p) => {
            let b_in = b.to_vec();
            let mut b_f = b_in.clone();
            let (a_f, dim_f) = fold_chain_forward_f64(f, a, &mut b_f, num_specs, dim)?;
            let mut b_p = b_in.clone();
            let (a_p, dim_p) = fold_chain_forward_f64(p, a, &mut b_p, num_specs, dim)?;
            if dim_f != dim_p {
                return None;
            }
            for s in 0..num_specs {
                b[s] = b_f[s] + b_p[s] - b_in[s];
            }
            let merged: Vec<f64> = a_f.iter().zip(a_p.iter()).map(|(x, y)| x + y).collect();
            Some((merged, dim_f))
        }
    }
}

fn fold_chain_forward_f64(
    layers: &[GpuCrownLayer],
    a: &[f64],
    b: &mut [f64],
    num_specs: usize,
    dim: usize,
) -> Option<(Vec<f64>, usize)> {
    let mut cur = a.to_vec();
    let mut cur_dim = dim;
    for layer in layers {
        let (na, nd) = fold_layer_forward_f64(layer, &cur, b, num_specs, cur_dim)?;
        cur = na;
        cur_dim = nd;
    }
    Some((cur, cur_dim))
}

fn fold_layer_forward_f64(
    layer: &GpuCrownLayer,
    a: &[f64],
    b: &mut [f64],
    num_specs: usize,
    dim: usize,
) -> Option<(Vec<f64>, usize)> {
    match layer {
        GpuCrownLayer::Linear {
            weight,
            bias,
            out_features,
            in_features,
            ..
        } => {
            if *out_features != dim || weight.len() != out_features * in_features {
                return None;
            }
            let din = *in_features;
            let mut out = vec![0.0f64; num_specs * din];
            for s in 0..num_specs {
                if let Some(bs) = bias {
                    if bs.len() != *out_features {
                        return None;
                    }
                    let mut acc = 0.0f64;
                    for i in 0..dim {
                        acc += a[s * dim + i] * f64::from(bs[i]);
                    }
                    b[s] += acc;
                }
                for i in 0..dim {
                    let av = a[s * dim + i];
                    if av == 0.0 {
                        continue;
                    }
                    let wrow = i * din;
                    let orow = s * din;
                    for j in 0..din {
                        out[orow + j] += av * f64::from(weight[wrow + j]);
                    }
                }
            }
            Some((out, din))
        }
        GpuCrownLayer::Conv2d {
            weight_col,
            bias_expanded,
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
            in_h,
            in_w,
            ..
        } => {
            let oc = *out_channels;
            let ic = *in_channels;
            let (kh, kw) = (*kernel_h, *kernel_w);
            let (oh, ow) = (*out_h, *out_w);
            let (ih, iw) = (*in_h, *in_w);
            let out_dim = oc * oh * ow;
            let in_dim = ic * ih * iw;
            if out_dim != dim || weight_col.len() != oc * ic * kh * kw {
                return None;
            }
            let mut out = vec![0.0f64; num_specs * in_dim];
            for s in 0..num_specs {
                if let Some(be) = bias_expanded {
                    if be.len() != out_dim {
                        return None;
                    }
                    let mut acc = 0.0f64;
                    for k in 0..out_dim {
                        acc += a[s * dim + k] * f64::from(be[k]);
                    }
                    b[s] += acc;
                }
                for c in 0..oc {
                    for y in 0..oh {
                        for x in 0..ow {
                            let av = a[s * dim + (c * oh + y) * ow + x];
                            if av == 0.0 {
                                continue;
                            }
                            for ky in 0..kh {
                                let iy = y * stride_h + ky;
                                if iy < *pad_h {
                                    continue;
                                }
                                let iyy = iy - pad_h;
                                if iyy >= ih {
                                    continue;
                                }
                                for kx in 0..kw {
                                    let ix = x * stride_w + kx;
                                    if ix < *pad_w {
                                        continue;
                                    }
                                    let ixx = ix - pad_w;
                                    if ixx >= iw {
                                        continue;
                                    }
                                    for cin in 0..ic {
                                        let wv = weight_col
                                            [c * (ic * kh * kw) + cin * (kh * kw) + ky * kw + kx];
                                        out[s * in_dim + (cin * ih + iyy) * iw + ixx] +=
                                            av * f64::from(wv);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Some((out, in_dim))
        }
        GpuCrownLayer::Activation {
            lower_slope,
            upper_slope,
            lower_intercept,
            upper_intercept,
            num_neurons,
        } => {
            let nn = *num_neurons;
            if nn != dim
                || lower_slope.len() != nn
                || upper_slope.len() != nn
                || lower_intercept.len() != nn
                || upper_intercept.len() != nn
            {
                return None;
            }
            let mut out = vec![0.0f64; num_specs * nn];
            for s in 0..num_specs {
                for i in 0..nn {
                    let ap = a[s * nn + i];
                    let (sig, ta) = if ap >= 0.0 {
                        (f64::from(lower_slope[i]), f64::from(lower_intercept[i]))
                    } else {
                        (f64::from(upper_slope[i]), f64::from(upper_intercept[i]))
                    };
                    b[s] += ap * ta;
                    out[s * nn + i] = ap * sig;
                }
            }
            Some((out, nn))
        }
        GpuCrownLayer::ActivationReluDualAlpha { .. } | GpuCrownLayer::MaxPool2d { .. } => None,
    }
}

/// Fold one segment forward (output→input). Returns `(A', new_dim)` and pushes each
/// ReLU's frozen checkpoint (fold order) into `relus`; accumulates the bias in `b`.
fn fold_segment_forward(
    seg: &GpuResnetSegment,
    a: &[f32],
    b: &mut [f32],
    num_specs: usize,
    dim: usize,
    relus: &mut Vec<ReluRec>,
) -> Option<(Vec<f32>, usize)> {
    match seg {
        GpuResnetSegment::Chain(layers) => fold_chain_forward(layers, a, b, num_specs, dim, relus),
        GpuResnetSegment::Residual(f) => {
            // out = F(z) + z. Skip = identity. F and skip share the bias accumulator
            // `b`; skip adds no bias. A_in = A_skip + A_F (design doc §1).
            let a_skip = a.to_vec();
            let (a_f, dim_f) = fold_chain_forward(f, a, b, num_specs, dim, relus)?;
            if dim_f != dim {
                return None; // identity residual must map dim → dim
            }
            let merged: Vec<f32> = a_skip.iter().zip(a_f.iter()).map(|(x, y)| x + y).collect();
            Some((merged, dim))
        }
        GpuResnetSegment::ResidualProj(f, p) => {
            // out = F(z) + P(z). Incoming bias counted ONCE: b_in = b_F + b_P − b.
            let b_in = b.to_vec();
            let mut b_f = b_in.clone();
            let (a_f, dim_f) = fold_chain_forward(f, a, &mut b_f, num_specs, dim, relus)?;
            let mut b_p = b_in.clone();
            let (a_p, dim_p) = fold_chain_forward(p, a, &mut b_p, num_specs, dim, relus)?;
            if dim_f != dim_p {
                return None;
            }
            for s in 0..num_specs {
                b[s] = b_f[s] + b_p[s] - b_in[s];
            }
            let merged: Vec<f32> = a_f.iter().zip(a_p.iter()).map(|(x, y)| x + y).collect();
            Some((merged, dim_f))
        }
    }
}

fn fold_chain_forward(
    layers: &[GpuCrownLayer],
    a: &[f32],
    b: &mut [f32],
    num_specs: usize,
    dim: usize,
    relus: &mut Vec<ReluRec>,
) -> Option<(Vec<f32>, usize)> {
    let mut cur = a.to_vec();
    let mut cur_dim = dim;
    for layer in layers {
        let (na, nd) = fold_layer_forward(layer, &cur, b, num_specs, cur_dim, relus)?;
        cur = na;
        cur_dim = nd;
    }
    Some((cur, cur_dim))
}

fn fold_layer_forward(
    layer: &GpuCrownLayer,
    a: &[f32],
    b: &mut [f32],
    num_specs: usize,
    dim: usize,
    relus: &mut Vec<ReluRec>,
) -> Option<(Vec<f32>, usize)> {
    match layer {
        GpuCrownLayer::Linear {
            weight,
            bias,
            out_features,
            in_features,
            ..
        } => {
            if *out_features != dim || weight.len() != out_features * in_features {
                return None;
            }
            let din = *in_features;
            let mut out = vec![0.0f32; num_specs * din];
            for s in 0..num_specs {
                // b += A·bias
                if let Some(bs) = bias {
                    if bs.len() != *out_features {
                        return None;
                    }
                    let mut acc = 0.0f32;
                    for i in 0..dim {
                        acc += a[s * dim + i] * bs[i];
                    }
                    b[s] += acc;
                }
                // A' = A·W
                for i in 0..dim {
                    let av = a[s * dim + i];
                    if av == 0.0 {
                        continue;
                    }
                    let wrow = i * din;
                    let orow = s * din;
                    for j in 0..din {
                        out[orow + j] += av * weight[wrow + j];
                    }
                }
            }
            Some((out, din))
        }
        GpuCrownLayer::Conv2d {
            weight_col,
            bias_expanded,
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
            in_h,
            in_w,
            ..
        } => {
            let oc = *out_channels;
            let ic = *in_channels;
            let (kh, kw) = (*kernel_h, *kernel_w);
            let (oh, ow) = (*out_h, *out_w);
            let (ih, iw) = (*in_h, *in_w);
            let out_dim = oc * oh * ow;
            let in_dim = ic * ih * iw;
            if out_dim != dim || weight_col.len() != oc * ic * kh * kw {
                return None;
            }
            let mut out = vec![0.0f32; num_specs * in_dim];
            for s in 0..num_specs {
                // b += A·bias_expanded
                if let Some(be) = bias_expanded {
                    if be.len() != out_dim {
                        return None;
                    }
                    let mut acc = 0.0f32;
                    for k in 0..out_dim {
                        acc += a[s * dim + k] * be[k];
                    }
                    b[s] += acc;
                }
                // A' = A ⊛ Wᵀ (transposed conv): scatter each (oc,oh,ow) onto its
                // receptive (ic,ih,iw) window.
                for c in 0..oc {
                    for y in 0..oh {
                        for x in 0..ow {
                            let av = a[s * dim + (c * oh + y) * ow + x];
                            if av == 0.0 {
                                continue;
                            }
                            for ky in 0..kh {
                                let iy = y * stride_h + ky;
                                if iy < *pad_h {
                                    continue;
                                }
                                let iyy = iy - pad_h;
                                if iyy >= ih {
                                    continue;
                                }
                                for kx in 0..kw {
                                    let ix = x * stride_w + kx;
                                    if ix < *pad_w {
                                        continue;
                                    }
                                    let ixx = ix - pad_w;
                                    if ixx >= iw {
                                        continue;
                                    }
                                    for cin in 0..ic {
                                        let wv = weight_col
                                            [c * (ic * kh * kw) + cin * (kh * kw) + ky * kw + kx];
                                        out[s * in_dim + (cin * ih + iyy) * iw + ixx] += av * wv;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Some((out, in_dim))
        }
        GpuCrownLayer::Activation {
            lower_slope,
            upper_slope,
            lower_intercept,
            upper_intercept,
            num_neurons,
        } => {
            let nn = *num_neurons;
            if nn != dim
                || lower_slope.len() != nn
                || upper_slope.len() != nn
                || lower_intercept.len() != nn
                || upper_intercept.len() != nn
            {
                return None;
            }
            let mut out = vec![0.0f32; num_specs * nn];
            let mut sigma = vec![0.0f32; num_specs * nn];
            let mut tau = vec![0.0f32; num_specs * nn];
            for s in 0..num_specs {
                for i in 0..nn {
                    let ap = a[s * nn + i];
                    // LOWER stream: positive coeff → lower relaxation (slope=α),
                    // negative coeff → upper relaxation (chord).
                    let (sig, ta) = if ap >= 0.0 {
                        (lower_slope[i], lower_intercept[i])
                    } else {
                        (upper_slope[i], upper_intercept[i])
                    };
                    sigma[s * nn + i] = sig;
                    tau[s * nn + i] = ta;
                    b[s] += ap * ta;
                    out[s * nn + i] = ap * sig;
                }
            }
            relus.push(ReluRec {
                a_pre: a.to_vec(),
                sigma,
                tau,
                nn,
            });
            Some((out, nn))
        }
        // Not supported by the joint gradient (the wide α path gates these out
        // upstream: dual-alpha / maxpool are not wide-batchable). Bail → caller
        // falls back to the local rule for this domain (sound).
        GpuCrownLayer::ActivationReluDualAlpha { .. } | GpuCrownLayer::MaxPool2d { .. } => None,
    }
}

/// Adjoint over segments, walked in REVERSE (input→output). `abar` enters at the
/// input side (ξ seed) and is returned at the output side. `cursor` walks down from
/// `relus.len()`; each ReLU adjoint consumes `relus[cursor-1]` so the reversed walk
/// pairs with the forward fold order exactly.
#[allow(clippy::too_many_arguments)]
fn adjoint_segments(
    segments: &[GpuResnetSegment],
    mut abar: Vec<f32>,
    num_specs: usize,
    dim: usize,
    relus: &[ReluRec],
    cursor: &mut usize,
    grads: &mut [Vec<f32>],
    cfg: JointGradConfig,
) -> Option<Vec<f32>> {
    let mut cur_dim = dim;
    for seg in segments.iter().rev() {
        let (na, nd) = adjoint_segment(seg, abar, num_specs, cur_dim, relus, cursor, grads, cfg)?;
        abar = na;
        cur_dim = nd;
    }
    Some(abar)
}

#[allow(clippy::too_many_arguments)]
fn adjoint_segment(
    seg: &GpuResnetSegment,
    abar: Vec<f32>,
    num_specs: usize,
    dim: usize,
    relus: &[ReluRec],
    cursor: &mut usize,
    grads: &mut [Vec<f32>],
    cfg: JointGradConfig,
) -> Option<(Vec<f32>, usize)> {
    match seg {
        GpuResnetSegment::Chain(layers) => {
            adjoint_chain(layers, abar, num_specs, dim, relus, cursor, grads, cfg)
        }
        GpuResnetSegment::Residual(f) => {
            // Ā_out = Ā_in + adjoint_F(Ā_in) (skip fan-out; identity carries no bias).
            let abar_f_in = abar.clone();
            let (abar_f, dim_f) =
                adjoint_chain(f, abar_f_in, num_specs, dim, relus, cursor, grads, cfg)?;
            if dim_f != dim {
                return None;
            }
            let out: Vec<f32> = abar.iter().zip(abar_f.iter()).map(|(x, y)| x + y).collect();
            Some((out, dim))
        }
        GpuResnetSegment::ResidualProj(f, p) => {
            // Ā_out = adjoint_F(Ā_in) + adjoint_P(Ā_in). To keep the ReLU cursor the
            // EXACT reverse of the forward fold (which folded F THEN P), the adjoint
            // must consume P's records BEFORE F's — run P's adjoint first.
            let (abar_p, dim_p) =
                adjoint_chain(p, abar.clone(), num_specs, dim, relus, cursor, grads, cfg)?;
            let (abar_f, dim_f) =
                adjoint_chain(f, abar, num_specs, dim, relus, cursor, grads, cfg)?;
            if dim_f != dim_p {
                return None;
            }
            let out: Vec<f32> = abar_f
                .iter()
                .zip(abar_p.iter())
                .map(|(x, y)| x + y)
                .collect();
            Some((out, dim_f))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn adjoint_chain(
    layers: &[GpuCrownLayer],
    mut abar: Vec<f32>,
    num_specs: usize,
    dim: usize,
    relus: &[ReluRec],
    cursor: &mut usize,
    grads: &mut [Vec<f32>],
    cfg: JointGradConfig,
) -> Option<(Vec<f32>, usize)> {
    let mut cur_dim = dim;
    for layer in layers.iter().rev() {
        let (na, nd) = adjoint_layer(layer, abar, num_specs, cur_dim, relus, cursor, grads, cfg)?;
        abar = na;
        cur_dim = nd;
    }
    Some((abar, cur_dim))
}

#[allow(clippy::too_many_arguments)]
fn adjoint_layer(
    layer: &GpuCrownLayer,
    abar: Vec<f32>,
    num_specs: usize,
    dim: usize,
    relus: &[ReluRec],
    cursor: &mut usize,
    grads: &mut [Vec<f32>],
    cfg: JointGradConfig,
) -> Option<(Vec<f32>, usize)> {
    match layer {
        GpuCrownLayer::Linear {
            weight,
            bias,
            out_features,
            in_features,
            ..
        } => {
            // Forward: A_out(din) = A_in(dout)·W ; b += A_in·bias.
            // Adjoint: Ā_in[i] = Σ_j Ā_out[j]·W[i,j] + bias[i]  (Ā_out has dim din).
            let dout = *out_features;
            let din = *in_features;
            if din != dim || weight.len() != dout * din {
                return None;
            }
            let mut out = vec![0.0f32; num_specs * dout];
            for s in 0..num_specs {
                for i in 0..dout {
                    let mut acc = 0.0f32;
                    let wrow = i * din;
                    for j in 0..din {
                        acc += abar[s * din + j] * weight[wrow + j];
                    }
                    if cfg.bias_channel {
                        if let Some(bs) = bias {
                            acc += bs[i];
                        }
                    }
                    out[s * dout + i] = acc;
                }
            }
            Some((out, dout))
        }
        GpuCrownLayer::Conv2d {
            weight_col,
            bias_expanded,
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
            in_h,
            in_w,
            ..
        } => {
            // Forward: A_out(IC·IH·IW) = A_in(OC·OH·OW) ⊛ Wᵀ ; b += A_in·bias_exp.
            // Adjoint: gather — Ā_in[oc,oh,ow] = Σ_{ic,kh,kw} Ā_out[ic,ih,iw]·W + bias.
            let oc = *out_channels;
            let ic = *in_channels;
            let (kh, kw) = (*kernel_h, *kernel_w);
            let (oh, ow) = (*out_h, *out_w);
            let (ih, iw) = (*in_h, *in_w);
            let out_dim = oc * oh * ow; // Ā_in dim
            let in_dim = ic * ih * iw; // Ā_out (incoming abar) dim
            if in_dim != dim || weight_col.len() != oc * ic * kh * kw {
                return None;
            }
            let mut out = vec![0.0f32; num_specs * out_dim];
            for s in 0..num_specs {
                for c in 0..oc {
                    for y in 0..oh {
                        for x in 0..ow {
                            let mut acc = 0.0f32;
                            for ky in 0..kh {
                                let iy = y * stride_h + ky;
                                if iy < *pad_h {
                                    continue;
                                }
                                let iyy = iy - pad_h;
                                if iyy >= ih {
                                    continue;
                                }
                                for kx in 0..kw {
                                    let ix = x * stride_w + kx;
                                    if ix < *pad_w {
                                        continue;
                                    }
                                    let ixx = ix - pad_w;
                                    if ixx >= iw {
                                        continue;
                                    }
                                    for cin in 0..ic {
                                        let wv = weight_col
                                            [c * (ic * kh * kw) + cin * (kh * kw) + ky * kw + kx];
                                        acc += abar[s * in_dim + (cin * ih + iyy) * iw + ixx] * wv;
                                    }
                                }
                            }
                            if cfg.bias_channel {
                                if let Some(be) = bias_expanded {
                                    acc += be[(c * oh + y) * ow + x];
                                }
                            }
                            out[s * out_dim + (c * oh + y) * ow + x] = acc;
                        }
                    }
                }
            }
            Some((out, out_dim))
        }
        GpuCrownLayer::Activation { num_neurons, .. } => {
            let nn = *num_neurons;
            if nn != dim || *cursor == 0 {
                return None;
            }
            *cursor -= 1;
            let rec = &relus[*cursor];
            if rec.nn != nn {
                return None;
            }
            let g = &mut grads[*cursor];
            let mut out = vec![0.0f32; num_specs * nn];
            for i in 0..nn {
                let mut gi = 0.0f32;
                for s in 0..num_specs {
                    let ab = abar[s * nn + i];
                    // harvest (uses Ā_out = incoming abar, α enters only positive coeff)
                    gi += ab * rec.a_pre[s * nn + i].max(0.0);
                    // propagate to A_in side
                    let mut v = ab * rec.sigma[s * nn + i];
                    if cfg.bias_channel {
                        v += rec.tau[s * nn + i];
                    }
                    out[s * nn + i] = v;
                }
                g[i] += gi;
            }
            Some((out, nn))
        }
        GpuCrownLayer::ActivationReluDualAlpha { .. } | GpuCrownLayer::MaxPool2d { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Deterministic LCG in [-1,1) (no rand dep), mirroring the Python validators.
    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    fn lin(rng: &mut Lcg, out: usize, inp: usize, scale: f32, bias: bool) -> GpuCrownLayer {
        let weight: Vec<f32> = (0..out * inp).map(|_| rng.f() * scale).collect();
        let b: Option<Arc<[f32]>> = if bias {
            Some((0..out).map(|_| rng.f()).collect::<Vec<_>>().into())
        } else {
            None
        };
        GpuCrownLayer::Linear {
            weight: weight.into(),
            bias: b,
            out_features: out,
            in_features: inp,
            cert_err: Default::default(),
        }
    }

    // ReLU relaxation for pre-activation bounds l<0<u: lower_slope=α, upper_slope=chord.
    fn relu(alpha: &[f32], l: &[f32], u: &[f32]) -> GpuCrownLayer {
        let nn = alpha.len();
        let ubar: Vec<f32> = (0..nn).map(|i| u[i] / (u[i] - l[i])).collect();
        let t: Vec<f32> = (0..nn).map(|i| -ubar[i] * l[i]).collect();
        GpuCrownLayer::Activation {
            lower_slope: alpha.to_vec(),
            upper_slope: ubar,
            lower_intercept: vec![0.0; nn],
            upper_intercept: t,
            num_neurons: nn,
        }
    }

    /// Self-concretized lower bound of the forward fold (coefficient channel only)
    /// — the exact function whose α-gradient the adjoint computes. Used to FD the
    /// module against itself (no GPU), mirroring `validate_joint_alpha_gradient.py`.
    fn fold_bound(
        segments: &[GpuResnetSegment],
        seed_a: &[f32],
        seed_b: &[f32],
        num_specs: usize,
        output_dim: usize,
        x_l: &[f32],
        x_u: &[f32],
    ) -> Vec<f32> {
        let mut b = seed_b.to_vec();
        let mut a = seed_a.to_vec();
        let mut dim = output_dim;
        let mut relus = Vec::new();
        for seg in segments {
            let (na, nd) =
                fold_segment_forward(seg, &a, &mut b, num_specs, dim, &mut relus).unwrap();
            a = na;
            dim = nd;
        }
        let input_dim = x_l.len();
        assert_eq!(dim, input_dim);
        let mut out = b;
        for s in 0..num_specs {
            for j in 0..input_dim {
                let av = a[s * input_dim + j];
                out[s] += if av >= 0.0 { av * x_l[j] } else { av * x_u[j] };
            }
        }
        out
    }

    fn central_fd(
        segments: &[GpuResnetSegment],
        seed_a: &[f32],
        seed_b: &[f32],
        num_specs: usize,
        output_dim: usize,
        x_l: &[f32],
        x_u: &[f32],
        relu_fold_idx: usize, // index of the Activation among fold-order relus
        neuron: usize,
        eps: f32,
    ) -> f32 {
        // Perturb the target Activation's lower_slope[neuron] by ±eps and central-diff
        // the summed lower bound. For multiple specs this is the joint scalar
        // objective whose adjoint sums the per-row alpha gradients.
        let perturb = |delta: f32| -> Vec<GpuResnetSegment> {
            let mut segs = segments.to_vec();
            let mut seen = 0usize;
            for seg in segs.iter_mut() {
                let branches: Vec<&mut Vec<GpuCrownLayer>> = match seg {
                    GpuResnetSegment::Chain(l) | GpuResnetSegment::Residual(l) => vec![l],
                    GpuResnetSegment::ResidualProj(f, p) => vec![f, p],
                };
                for layers in branches {
                    for l in layers.iter_mut() {
                        if let GpuCrownLayer::Activation { lower_slope, .. } = l {
                            if seen == relu_fold_idx {
                                lower_slope[neuron] += delta;
                            }
                            seen += 1;
                        }
                    }
                }
            }
            segs
        };
        let sp = perturb(eps);
        let sm = perturb(-eps);
        let bp: f32 = fold_bound(&sp, seed_a, seed_b, num_specs, output_dim, x_l, x_u)
            .iter()
            .sum();
        let bm: f32 = fold_bound(&sm, seed_a, seed_b, num_specs, output_dim, x_l, x_u)
            .iter()
            .sum();
        (bp - bm) / (2.0 * eps)
    }

    fn relerr(a: f32, b: f32) -> f32 {
        (a - b).abs() / a.abs().max(b.abs()).max(1e-6)
    }

    #[test]
    fn joint_matches_self_fd_two_relu_chain() {
        // 2-ReLU chain (the regime where local ≠ joint): W3·ReLU2·W2·ReLU1·W1.
        let mut rng = Lcg(0xA1FA_C0DE);
        let (d0, d1, d2, d3) = (4usize, 5usize, 5usize, 3usize);
        let l1: Vec<f32> = (0..d1).map(|i| -1.0 - 0.3 * i as f32).collect();
        let u1: Vec<f32> = (0..d1).map(|i| 1.0 + 0.2 * i as f32).collect();
        let l2: Vec<f32> = (0..d2).map(|i| -1.2 - 0.2 * i as f32).collect();
        let u2: Vec<f32> = (0..d2).map(|i| 0.9 + 0.3 * i as f32).collect();
        let x_l = vec![-1.0f32; d0];
        let x_u = vec![1.0f32; d0];

        for (a1v, a2v) in [(0.5f32, 0.5f32), (0.3, 0.7), (0.8, 0.2), (0.1, 0.4)] {
            let a1 = vec![a1v; d1];
            let a2 = vec![a2v; d2];
            // fold order (output→input): W3, ReLU2, W2, ReLU1, W1
            let w3 = lin(&mut rng, d3, d2, 1.0, true);
            let r2 = relu(&a2, &l2, &u2);
            let w2 = lin(&mut rng, d2, d1, 1.0, true);
            let r1 = relu(&a1, &l1, &u1);
            let w1 = lin(&mut rng, d1, d0, 1.0, true);
            let segs = vec![GpuResnetSegment::Chain(vec![w3, r2, w2, r1, w1])];
            // seed = identity spec (num_specs=d3), b=0
            let mut seed_a = vec![0.0f32; d3 * d3];
            for i in 0..d3 {
                seed_a[i * d3 + i] = 1.0;
            }
            let seed_b = vec![0.0f32; d3];

            let g = joint_alpha_gradient(
                &segs,
                &seed_a,
                &seed_b,
                d3,
                d3,
                &x_l,
                &x_u,
                JointGradConfig::default(),
            )
            .expect("gradient");
            assert_eq!(g.len(), 2, "two relus in fold order");
            // relu fold idx 0 == ReLU2 (nn d2), idx 1 == ReLU1 (nn d1)
            let mut worst = 0.0f32;
            for (fold_idx, nn) in [(0usize, d2), (1usize, d1)] {
                for neuron in 0..nn {
                    let fd = central_fd(
                        &segs, &seed_a, &seed_b, d3, d3, &x_l, &x_u, fold_idx, neuron, 1e-3,
                    );
                    worst = worst.max(relerr(g[fold_idx][neuron], fd));
                }
            }
            assert!(
                worst < 2e-2,
                "2-relu joint vs self-FD worst rel.err {worst}"
            );
        }
    }

    #[test]
    fn joint_matches_self_fd_residual_block() {
        // x→W1→ReLU1→a1→[a1 + Wg·ReLUf·Wf(a1)]→Wout→out (a Residual block).
        let mut rng = Lcg(0x5EED_BEEF);
        let (d0, dh, dout) = (3usize, 4usize, 2usize);
        let l1: Vec<f32> = (0..dh).map(|i| -1.0 - 0.3 * i as f32).collect();
        let u1: Vec<f32> = (0..dh).map(|i| 1.0 + 0.2 * i as f32).collect();
        let lf: Vec<f32> = (0..dh).map(|i| -1.1 - 0.2 * i as f32).collect();
        let uf: Vec<f32> = (0..dh).map(|i| 0.8 + 0.3 * i as f32).collect();
        let x_l = vec![-1.0f32; d0];
        let x_u = vec![1.0f32; d0];

        for (a1v, afv) in [(0.5f32, 0.5f32), (0.3, 0.7), (0.8, 0.2), (0.2, 0.9)] {
            let a1 = vec![a1v; dh];
            let af = vec![afv; dh];
            // Trunk (fold order): Wout(dout×dh), then residual block over dh, then
            // ReLU1(dh), then W1(dh×d0). Residual F fold order: Wg, ReLUf, Wf.
            let wout = lin(&mut rng, dout, dh, 1.0, true);
            let wg = lin(&mut rng, dh, dh, 1.0, true);
            let rf = relu(&af, &lf, &uf);
            let wf = lin(&mut rng, dh, dh, 1.0, true);
            let r1 = relu(&a1, &l1, &u1);
            let w1 = lin(&mut rng, dh, d0, 1.0, true);
            let segs = vec![
                GpuResnetSegment::Chain(vec![wout]),
                GpuResnetSegment::Residual(vec![wg, rf, wf]),
                GpuResnetSegment::Chain(vec![r1, w1]),
            ];
            let mut seed_a = vec![0.0f32; dout * dout];
            for i in 0..dout {
                seed_a[i * dout + i] = 1.0;
            }
            let seed_b = vec![0.0f32; dout];

            let g = joint_alpha_gradient(
                &segs,
                &seed_a,
                &seed_b,
                dout,
                dout,
                &x_l,
                &x_u,
                JointGradConfig::default(),
            )
            .expect("gradient");
            // fold order relus: ReLUf (idx 0, inside residual), ReLU1 (idx 1, trunk)
            assert_eq!(g.len(), 2);
            let mut worst = 0.0f32;
            let mut worst_branch = 0.0f32;
            for (fold_idx, is_branch) in [(0usize, true), (1usize, false)] {
                for neuron in 0..dh {
                    let fd = central_fd(
                        &segs, &seed_a, &seed_b, dout, dout, &x_l, &x_u, fold_idx, neuron, 1e-3,
                    );
                    let e = relerr(g[fold_idx][neuron], fd);
                    worst = worst.max(e);
                    if is_branch {
                        worst_branch = worst_branch.max(e);
                    }
                }
            }
            assert!(
                worst < 2e-2,
                "residual joint vs self-FD worst rel.err {worst}"
            );
            // Confirm the residual-branch fan-out is genuinely exercised (nonzero grad).
            let branch_nz = g[0].iter().any(|&v| v.abs() > 1e-4);
            assert!(branch_nz, "residual branch grad must be nonzero");
            let _ = worst_branch;
        }
    }

    #[test]
    fn joint_matches_self_fd_residual_proj_block() {
        // Projection residual out = F(z) + P(z) (a 1×1-conv/linear skip at a stage
        // transition). Exercises the ResidualProj fan-out + the F-then-P ReLU cursor.
        let mut rng = Lcg(0xC0FF_EE42);
        let (d0, din, dblk, dout) = (3usize, 4usize, 5usize, 2usize);
        let lf: Vec<f32> = (0..dblk).map(|i| -1.0 - 0.2 * i as f32).collect();
        let uf: Vec<f32> = (0..dblk).map(|i| 0.9 + 0.2 * i as f32).collect();
        let lp: Vec<f32> = (0..dblk).map(|i| -0.8 - 0.15 * i as f32).collect();
        let up: Vec<f32> = (0..dblk).map(|i| 1.0 + 0.1 * i as f32).collect();
        let l1: Vec<f32> = (0..din).map(|i| -1.1 - 0.1 * i as f32).collect();
        let u1: Vec<f32> = (0..din).map(|i| 0.7 + 0.2 * i as f32).collect();
        let x_l = vec![-1.0f32; d0];
        let x_u = vec![1.0f32; d0];

        for (afv, apv, a1v) in [(0.5f32, 0.4f32, 0.6f32), (0.3, 0.7, 0.2), (0.8, 0.2, 0.5)] {
            let af = vec![afv; dblk];
            let ap = vec![apv; dblk];
            let a1 = vec![a1v; din];
            // Trunk (fold order): Wout(dout×dblk), ResidualProj block (dblk from din),
            // then W_in(din×d0)·ReLU1 on the block input side. F: Wf1(dblk×din)+ReLUf(dblk)
            // ... wait, both F and P map block-input(din) → block-output(dblk).
            // F fold order (output→input): ReLUf(dblk), Wf(dblk×din).
            // P fold order: ReLUp(dblk), Wp(dblk×din).
            let wout = lin(&mut rng, dout, dblk, 1.0, true);
            let rf = relu(&af, &lf, &uf);
            let wf = lin(&mut rng, dblk, din, 1.0, true);
            let rp = relu(&ap, &lp, &up);
            let wp = lin(&mut rng, dblk, din, 1.0, true);
            let r1 = relu(&a1, &l1, &u1);
            let w1 = lin(&mut rng, din, d0, 1.0, true);
            let segs = vec![
                GpuResnetSegment::Chain(vec![wout]),
                GpuResnetSegment::ResidualProj(vec![rf, wf], vec![rp, wp]),
                GpuResnetSegment::Chain(vec![r1, w1]),
            ];
            let mut seed_a = vec![0.0f32; dout * dout];
            for i in 0..dout {
                seed_a[i * dout + i] = 1.0;
            }
            let seed_b = vec![0.0f32; dout];

            let g = joint_alpha_gradient(
                &segs,
                &seed_a,
                &seed_b,
                dout,
                dout,
                &x_l,
                &x_u,
                JointGradConfig::default(),
            )
            .expect("gradient");
            // fold-order relus: ReLUf (idx 0, F branch), ReLUp (idx 1, P branch),
            // ReLU1 (idx 2, trunk).
            assert_eq!(g.len(), 3);
            let mut worst = 0.0f32;
            for (fold_idx, nn) in [(0usize, dblk), (1, dblk), (2, din)] {
                for neuron in 0..nn {
                    let fd = central_fd(
                        &segs, &seed_a, &seed_b, dout, dout, &x_l, &x_u, fold_idx, neuron, 1e-3,
                    );
                    worst = worst.max(relerr(g[fold_idx][neuron], fd));
                }
            }
            assert!(
                worst < 2e-2,
                "residual-proj joint vs self-FD worst rel.err {worst}"
            );
            // Both branches must carry nonzero gradient (fan-out genuinely exercised).
            assert!(
                g[0].iter().any(|&v| v.abs() > 1e-4),
                "F branch grad must be nonzero"
            );
            assert!(
                g[1].iter().any(|&v| v.abs() > 1e-4),
                "P branch grad must be nonzero"
            );
        }
    }

    #[test]
    fn positive_weighted_nine_row_joint_matches_chunk_sum_and_fd() {
        // Exercise the production MW decomposition boundary with one row more
        // than the resident eight-row cap. Every base objective is scaled by a
        // strictly positive simplex weight before it enters the adjoint.
        let mut rng = Lcg(0x9A11_C0DE);
        let (d0, d1, d2, output_dim) = (3usize, 4usize, 4usize, 3usize);
        let l1: Vec<f32> = (0..d1).map(|i| -1.1 - 0.2 * i as f32).collect();
        let u1: Vec<f32> = (0..d1).map(|i| 0.9 + 0.15 * i as f32).collect();
        let l2: Vec<f32> = (0..d2).map(|i| -0.8 - 0.25 * i as f32).collect();
        let u2: Vec<f32> = (0..d2).map(|i| 1.2 + 0.1 * i as f32).collect();
        let x_l = vec![-1.0f32; d0];
        let x_u = vec![1.0f32; d0];
        let segs = vec![GpuResnetSegment::Chain(vec![
            lin(&mut rng, output_dim, d2, 0.8, true),
            relu(&[0.62, 0.47, 0.71, 0.38], &l2, &u2),
            lin(&mut rng, d2, d1, 0.8, true),
            relu(&[0.41, 0.58, 0.36, 0.69], &l1, &u1),
            lin(&mut rng, d1, d0, 0.8, true),
        ])];

        const ROWS: usize = 9;
        const CHUNK_ROWS: usize = 8;
        let normalizer = (1..=ROWS).sum::<usize>() as f32;
        let weights: Vec<f32> = (1..=ROWS).map(|row| row as f32 / normalizer).collect();
        assert!(weights
            .iter()
            .all(|weight| weight.is_finite() && *weight > 0.0));
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        let mut weighted_seed = Vec::with_capacity(ROWS * output_dim);
        for (row, &weight) in weights.iter().enumerate() {
            for column in 0..output_dim {
                // Alternating signs force distinct per-row relaxation choices;
                // nonzero magnitudes also make positive-scale preservation
                // observable rather than vacuous.
                let sign = if (row + column).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
                let coefficient = sign * (0.35 + 0.06 * row as f32 + 0.04 * column as f32);
                weighted_seed.push(weight * coefficient);
            }
        }
        let seed_bias = vec![0.0f32; ROWS];

        let full = joint_alpha_gradient(
            &segs,
            &weighted_seed,
            &seed_bias,
            ROWS,
            output_dim,
            &x_l,
            &x_u,
            JointGradConfig::default(),
        )
        .expect("nine-row weighted joint gradient");
        assert_eq!(full.iter().map(Vec::len).collect::<Vec<_>>(), vec![d2, d1]);

        let mut chunk_sum: Option<Vec<Vec<f32>>> = None;
        for chunk in weighted_seed.chunks(CHUNK_ROWS * output_dim) {
            let chunk_rows = chunk.len() / output_dim;
            assert!(chunk_rows > 0 && chunk_rows <= CHUNK_ROWS);
            assert_eq!(chunk.len(), chunk_rows * output_dim);
            let chunk_gradient = joint_alpha_gradient(
                &segs,
                chunk,
                &vec![0.0; chunk_rows],
                chunk_rows,
                output_dim,
                &x_l,
                &x_u,
                JointGradConfig::default(),
            )
            .expect("bounded weighted chunk gradient");
            match &mut chunk_sum {
                Some(accumulated) => {
                    assert_eq!(accumulated.len(), chunk_gradient.len());
                    for (total, addend) in accumulated.iter_mut().zip(chunk_gradient) {
                        assert_eq!(total.len(), addend.len());
                        for (total, addend) in total.iter_mut().zip(addend) {
                            *total += addend;
                        }
                    }
                }
                None => chunk_sum = Some(chunk_gradient),
            }
        }
        let chunk_sum = chunk_sum.expect("8+1 decomposition must produce gradients");

        let mut max_chunk_error = 0.0f32;
        let mut max_fd_error = 0.0f32;
        let mut max_scale = 0.0f32;
        let mut nonzero = false;
        for (fold_idx, neuron_count) in [(0usize, d2), (1usize, d1)] {
            for neuron in 0..neuron_count {
                let joint = full[fold_idx][neuron];
                nonzero |= joint.abs() > 1e-4;
                let chunked = chunk_sum[fold_idx][neuron];
                max_chunk_error = max_chunk_error.max((joint - chunked).abs());
                let fd = central_fd(
                    &segs,
                    &weighted_seed,
                    &seed_bias,
                    ROWS,
                    output_dim,
                    &x_l,
                    &x_u,
                    fold_idx,
                    neuron,
                    1e-3,
                );
                max_fd_error = max_fd_error.max((joint - fd).abs());
                max_scale = max_scale.max(joint.abs()).max(fd.abs()).max(chunked.abs());
            }
        }
        assert!(
            nonzero,
            "weighted fixture must exercise a nonzero alpha gradient"
        );
        assert!(
            max_chunk_error <= 2e-6 + 2e-5 * max_scale,
            "full nine-row joint vs 8+1 chunk sum max error {max_chunk_error} at scale {max_scale}"
        );
        assert!(
            max_fd_error <= 5e-4 + 3e-2 * max_scale,
            "weighted nine-row joint vs central FD max error {max_fd_error} at scale {max_scale}"
        );
    }

    #[test]
    fn bias_channel_off_degrades() {
        // Dropping the bias channel must degrade the 2-ReLU-chain gradient (≈0.7×
        // wrong per design doc §2), while the full adjoint matches FD.
        let mut rng = Lcg(0xD00D_5EED);
        let (d0, d1, d2, d3) = (4usize, 5usize, 5usize, 3usize);
        let l1: Vec<f32> = (0..d1).map(|i| -1.0 - 0.3 * i as f32).collect();
        let u1: Vec<f32> = (0..d1).map(|i| 1.0 + 0.2 * i as f32).collect();
        let l2: Vec<f32> = (0..d2).map(|i| -1.2 - 0.2 * i as f32).collect();
        let u2: Vec<f32> = (0..d2).map(|i| 0.9 + 0.3 * i as f32).collect();
        let x_l = vec![-1.0f32; d0];
        let x_u = vec![1.0f32; d0];
        let a1 = vec![0.5f32; d1];
        let a2 = vec![0.5f32; d2];
        let w3 = lin(&mut rng, d3, d2, 1.0, true);
        let r2 = relu(&a2, &l2, &u2);
        let w2 = lin(&mut rng, d2, d1, 1.0, true);
        let r1 = relu(&a1, &l1, &u1);
        let w1 = lin(&mut rng, d1, d0, 1.0, true);
        let segs = vec![GpuResnetSegment::Chain(vec![w3, r2, w2, r1, w1])];
        let mut seed_a = vec![0.0f32; d3 * d3];
        for i in 0..d3 {
            seed_a[i * d3 + i] = 1.0;
        }
        let seed_b = vec![0.0f32; d3];

        let g_full = joint_alpha_gradient(
            &segs,
            &seed_a,
            &seed_b,
            d3,
            d3,
            &x_l,
            &x_u,
            JointGradConfig { bias_channel: true },
        )
        .unwrap();
        let g_nobias = joint_alpha_gradient(
            &segs,
            &seed_a,
            &seed_b,
            d3,
            d3,
            &x_l,
            &x_u,
            JointGradConfig {
                bias_channel: false,
            },
        )
        .unwrap();
        // FD reference (deep layer ReLU1 == fold idx 1, where the bias channel bites).
        let mut full_worst = 0.0f32;
        let mut nobias_worst = 0.0f32;
        for neuron in 0..d1 {
            let fd = central_fd(&segs, &seed_a, &seed_b, d3, d3, &x_l, &x_u, 1, neuron, 1e-3);
            full_worst = full_worst.max(relerr(g_full[1][neuron], fd));
            nobias_worst = nobias_worst.max(relerr(g_nobias[1][neuron], fd));
        }
        assert!(
            full_worst < 2e-2,
            "full adjoint must match FD (worst {full_worst})"
        );
        assert!(
            nobias_worst > 0.1,
            "no-bias adjoint must visibly diverge from FD (worst {nobias_worst})"
        );
    }
}
