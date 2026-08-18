// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound GPU CROWN backward — composition of the verified per-layer error
//! primitives (increment 5 of task #15, host-orchestrated form).
//!
//! Composes [`GemmEngine::crown_aw_error_step`] (linear A·W + error),
//! [`crown_activation_error_step`] (relaxation + error), and
//! [`WgpuDevice::concretize_sound_gpu`] (sound concretize) into a complete,
//! SOUND CROWN backward whose heavy GEMMs run on the GPU. It carries the
//! certified coefficient error `(lower_a_err, upper_a_err)` and the accumulated
//! bias error through every layer — the bias/intercept folds charge BOTH the
//! propagated coefficient error AND their own f64 accumulation rounding
//! (`γ_{n+1}·(|acc| + Σ|a·b|)`, the same term the resident shader carries) —
//! so the final concretized bounds are a sound enclosure under round-to-nearest
//! f32 — the property the `sound_gpu_crown_required` gate currently forces onto
//! the CPU.
//!
//! This host-orchestrated version still round-trips A-matrices between layers
//! (so it is not yet faster than the CPU-error dense path); its job is to prove
//! the primitives COMPOSE into a correct sound backward. The fully GPU-resident
//! dispatch (no round-trip — the actual cifar100/tinyimagenet speedup) reuses
//! exactly this math inside the ping-pong shader loop.

use ny_core::{
    crown_activation_error_step, ConvTranspose2dParams, GemmEngine, GpuCrownLayer, NyError, Result,
};

use super::super::WgpuDevice;
use super::gemm::WgpuDiagnosticGemm;
use super::sentinel_taint_selfcheck::PRODUCTION_GUARDS_CONSULT_TAINT_WORD;

/// f64 unit roundoff, 2⁻⁵³ (exact).
const U64: f64 = f64::from_bits(0x3CA0_0000_0000_0000);
/// Smallest positive f64 subnormal, 2⁻¹⁰⁷⁴ — additive underflow floor.
const ETA64: f64 = f64::from_bits(0x0000_0000_0000_0001);

/// `γ_k = k·u / (1 − k·u)` for an f64 length-`k` accumulation (`u = 2⁻⁵³`).
#[inline]
fn gamma_k_f64(k: usize) -> f64 {
    let ku = (k as f64) * U64;
    if ku < 0.5 {
        ku / (1.0 - ku)
    } else {
        2.0 * ku
    }
}

/// SOUND fold of a per-output bias into a running f64 bound accumulator:
/// `acc[s] += Σ_k a[s,k]·b_k`, certifying BOTH the propagated coefficient error
/// (`a_err·|b_k|`) AND the fold's own f64 rounding. Each f32×f32 product is
/// exact in f64, but the sum chains `n` adds onto the incoming `acc[s]`, so the
/// fold error is bounded by Higham's `γ_{n+1}·(|acc| + Σ|a·b|)`, charged over
/// the COMPUTED magnitudes with a `(1+2γ)` slack + underflow floor (the same
/// accounting as the resident shader's `γ_k·Σ|a·bias|` term).
fn bias_fold_f64(
    num_specs: usize,
    n: usize,
    a: &[f32],
    a_err: &[f32],
    bias: &[f32],
    acc: &mut [f64],
    acc_err: &mut [f64],
) {
    let gamma = gamma_k_f64(n + 1);
    let real_factor = 1.0 + 2.0 * gamma;
    let additive = 8.0 * ((n + 1) as f64) * ETA64;
    for s in 0..num_specs {
        let mut mag = acc[s].abs();
        for k in 0..n {
            let bk = f64::from(bias[k]);
            let prod = f64::from(a[s * n + k]) * bk;
            acc[s] += prod;
            mag += prod.abs();
            acc_err[s] += f64::from(a_err[s * n + k]) * bk.abs();
        }
        acc_err[s] += gamma * mag * real_factor + additive;
    }
}

/// #u4 taint companion of [`bias_fold_f64`] (TAINT_GUARD_AUDIT.md §4 C1,
/// "plumbed from"): OR the per-coefficient `a`/`err` taint words into the
/// per-spec bias-taint accumulator, one channel (lower or upper) per call,
/// mirroring the value fold's `acc[s] += Σ_k a[s,k]·b_k` /
/// `acc_err[s] += Σ_k a_err[s,k]·|b_k|` indexing exactly.
///
/// Canon rule: `taint_out = OR over inputs of (taint_in AND its multiplicative
/// partner != 0)`. Both `a[s,k]` and `a_err[s,k]` multiply `bias[k]`, so
/// `bias[k] == 0.0` (either sign of zero) annihilates both words for that `k`
/// (`R·0 == 0` for every finite real the sentinel stands for). The fold's own
/// saturation term is absent by construction: it accumulates in f64 (which
/// never clamps to the finite sentinel) and any non-finite escape is refused by
/// the concretize host preflight bit tests (crown_concretize_sound.rs, G5).
///
/// TEST-REFERENCE STATUS (2026-08-10): the resident walk no longer calls this
/// — its Linear bias-fold transport now runs ON-DEVICE
/// (`TAINT_ROW_OR_SHADER`, per-COLUMN partner = `bias_buf`, same `bias[k] !=
/// 0` conjunct bit-for-bit; crown_backward_sound_resident.rs). This fn stays
/// as the committed CPU statement of those semantics, exercised by the
/// `cpu_tests` below and pinned through the walk by
/// `taint_walk_bias_conjunct_annihilates_on_device`.
/// `PRODUCTION_GUARDS_CONSULT_TAINT_WORD` (ops/sentinel_taint_selfcheck.rs) is
/// armed; this helper remains a test reference for the corresponding on-device
/// fold and is not a production escape around the consult.
#[allow(dead_code)] // test-reference: the walk's transport moved on-device
pub(super) fn bias_fold_taint(
    num_specs: usize,
    n: usize,
    a_taint: &[u32],
    a_err_taint: &[u32],
    bias: &[f32],
    acc_taint: &mut [u32],
) {
    for s in 0..num_specs {
        let mut word = 0u32;
        for k in 0..n {
            if bias[k] != 0.0 {
                word |= a_taint[s * n + k] | a_err_taint[s * n + k];
            }
        }
        acc_taint[s] |= word;
    }
}

/// #u4 taint companion of the Activation INTERCEPT fold in
/// [`WgpuDevice::crown_backward_sound_host`] (the sign-routed
/// `lb[s] += la·li` / `ub[s] += ua·ui` loop plus its `err·(|li|+|ui|)` charge):
/// OR the per-coefficient `a`/`err` taint into the per-spec bias-taint, one
/// channel per call.
///
/// The value fold routes each `a` to `lower_intercept[i]` or
/// `upper_intercept[i]` by the SIGN of `a` — but a tainted `a` has an
/// untrustworthy sign, so its true partner may be EITHER intercept; and the
/// err term multiplies `|li| + |ui|`, nonzero whenever either is. Annihilation
/// is therefore only sound when BOTH intercepts are exactly zero (the common
/// ReLU `lower_intercept == 0` alone does NOT annihilate): conjunct
/// `lower_intercept[i] != 0.0 || upper_intercept[i] != 0.0` for both words.
///
/// TEST-REFERENCE STATUS (2026-08-10): the resident walk no longer calls this
/// — its Activation intercept-fold transport now runs ON-DEVICE as TWO
/// `TAINT_ROW_OR_SHADER` per-COLUMN-partner dispatches per word buffer (one
/// with `lint_buf`, one with `uint_buf`: a word survives iff EITHER dispatch
/// keeps it, exactly this fn's `li != 0 || ui != 0` disjunction;
/// crown_backward_sound_resident.rs, single-domain layout — batched-domain
/// keeps the unconditional row-OR fallback). This fn stays as the committed
/// CPU statement of those semantics, exercised by the `cpu_tests` below.
#[allow(dead_code)] // test-reference: the walk's transport moved on-device
pub(super) fn intercept_fold_taint(
    num_specs: usize,
    num_neurons: usize,
    a_taint: &[u32],
    err_taint: &[u32],
    lower_intercept: &[f32],
    upper_intercept: &[f32],
    acc_taint: &mut [u32],
) {
    for s in 0..num_specs {
        let mut word = 0u32;
        for i in 0..num_neurons {
            if lower_intercept[i] != 0.0 || upper_intercept[i] != 0.0 {
                word |= a_taint[s * num_neurons + i] | err_taint[s * num_neurons + i];
            }
        }
        acc_taint[s] |= word;
    }
}

/// #u4 taint companion of the Conv2d row-max coefficient-error over-bound in
/// [`WgpuDevice::crown_backward_sound_host`] (the
/// `γ·max_k|a[s,k]|·‖W‖₁ + max_k|err[s,k]|·‖W‖₁` fold — host analogue of
/// `CROWN_CONV_ERROR_ROWMAX_SHADER`): OR taint ACROSS the maxed row into one
/// per-spec word, one channel per call.
///
/// The value fold's output is a single row-constant error built from row
/// MAXIMA, so any tainted element in the row taints the whole broadcast row —
/// the max may have selected the tainted element, or a laundered (shrunk)
/// taint may have kept it from being selected, which is the exact under-count
/// this word exists to catch. Multiplicative partner: `‖W‖₁` (`kernel_l1`);
/// an exactly-zero kernel annihilates (every product in the fold is exactly
/// 0). The per-spec output word stands for all `new_dim` broadcast copies of
/// the row error, mirroring the value fold's constant `el`/`eu` fill.
#[allow(dead_code)]
fn conv_rowmax_error_taint(
    num_specs: usize,
    out_dim: usize,
    a_taint: &[u32],
    err_taint: &[u32],
    kernel_l1: f64,
    row_err_taint: &mut [u32],
) {
    if kernel_l1 == 0.0 {
        return;
    }
    for s in 0..num_specs {
        let mut word = 0u32;
        for k in 0..out_dim {
            word |= a_taint[s * out_dim + k] | err_taint[s * out_dim + k];
        }
        row_err_taint[s] |= word;
    }
}

/// Reshape a CROWN coefficient `(num_specs × OC·OH·OW)` row-major into the
/// `(num_specs·OH·OW × OC)` layout `conv_transpose_2d` expects.
fn reshape_for_conv(a: &[f32], num_specs: usize, oc: usize, oh: usize, ow: usize) -> Vec<f32> {
    let spatial = oh * ow;
    let mut out = vec![0.0f32; num_specs * spatial * oc];
    for s in 0..num_specs {
        for c in 0..oc {
            for p in 0..spatial {
                out[(s * spatial + p) * oc + c] = a[s * (oc * spatial) + c * spatial + p];
            }
        }
    }
    out
}

/// Round an `f64` DOWN / UP to `f32` (outward), so a bias lower bound never
/// rounds up and an upper never rounds down.
fn down(x: f64) -> f32 {
    let n = x as f32;
    if n.is_finite() && f64::from(n) > x {
        f32::from_bits(if n > 0.0 {
            n.to_bits() - 1
        } else {
            n.to_bits() + 1
        })
    } else {
        n
    }
}
fn up(x: f64) -> f32 {
    let n = x as f32;
    if n.is_finite() && f64::from(n) < x {
        f32::from_bits(if n > 0.0 {
            n.to_bits() + 1
        } else {
            n.to_bits() - 1
        })
    } else {
        n
    }
}

impl WgpuDevice {
    /// Sound CROWN backward over admitted Linear/Activation layers (backward
    /// order), with GPU GEMMs. Returns `(lower, upper)`, one sound bound per
    /// spec row.
    ///
    /// `spec` is the initial coefficient matrix `(num_specs × output_dim)`
    /// row-major (the network-output selector C). With the C1 word consult
    /// armed, Conv2d refuses before dispatch: its fused GEMM-to-col2im interior
    /// has no word transport, so a boundary-only host sweep cannot observe a
    /// sentinel that is created and then cancelled internally. MaxPool and
    /// dual-alpha layers are likewise not handled by this host form.
    ///
    /// Increment 5 of task #15.
    #[allow(dead_code)]
    pub(crate) fn crown_backward_sound_host(
        &self,
        layers: &[GpuCrownLayer],
        spec: &[f32],
        num_specs: usize,
        output_dim: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if spec.len() != num_specs * output_dim {
            return Err(NyError::shape_mismatch(
                vec![num_specs, output_dim],
                vec![spec.len()],
            ));
        }
        // #flush-charge Lane A (guard-coverage audit, route R5): this host
        // driver charges Higham/γ terms derived for round-to-nearest IEEE f32
        // with gradual underflow. NONE of them are audited against the charged
        // flush model, and the raw diagnostic GEMM it dispatches carries no
        // DAZ cover of its own — so a CHARGED-flush device must never run this
        // route. (Today it also has no production caller; this refusal makes
        // that structural rather than incidental.)
        if self.charged_flush_authority_cached().is_some() {
            return Err(NyError::UnsupportedOp(
                "#flush-charge: crown_backward_sound_host is not audited \
                 against the charged flush model — refusing under \
                 charged-flush authority (fail-closed)"
                    .into(),
            ));
        }
        // #cert-err fail-closed: this host walk charges only its own f32/f64
        // rounding against weights it treats as EXACT. A layer carrying a
        // BN-fold `CertifiedWeightError` needs the extra `w_rel`/`bias_abs_err`
        // terms the resident walk charges; publishing without them would be a
        // radius that omits a real error source.
        ny_core::refuse_uncharged_certified_weight_error(layers, "crown_backward_sound_host")?;
        if PRODUCTION_GUARDS_CONSULT_TAINT_WORD
            && layers
                .iter()
                .any(|layer| matches!(layer, GpuCrownLayer::Conv2d { .. }))
        {
            return Err(NyError::UnsupportedOp(
                "crown_backward_sound_host: armed taint-word authority refuses Conv2d because \
                 the fused GEMM-to-col2im interior has no word transport"
                    .into(),
            ));
        }
        let mut dim = output_dim;
        let mut lower_a = spec.to_vec();
        let mut upper_a = spec.to_vec();
        let mut lower_err = vec![0.0f32; num_specs * dim];
        let mut upper_err = vec![0.0f32; num_specs * dim];
        // Bias and its certified error, accumulated in f64. The error covers the
        // propagated coefficient error AND each fold's own f64 rounding.
        let mut lb = vec![0.0f64; num_specs];
        let mut ub = vec![0.0f64; num_specs];
        let mut lb_err = vec![0.0f64; num_specs];
        let mut ub_err = vec![0.0f64; num_specs];
        // This host-orchestrated arithmetic oracle is crate-private.  Route
        // its certified default helpers through the equally private raw-GEMM
        // adapter; the public `GemmEngine for WgpuDevice` remains fail-closed.
        let diagnostic_gemm = WgpuDiagnosticGemm::new(self);

        // #u4 C1 wording for the admitted HOST form: every layer's outputs live in
        // host Vecs, so a saturation (the GPU value GEMM's nan_safe_clamp
        // writes exactly ±FALLBACK_BOUND, and NaN is preserved by contract)
        // is HOST-VISIBLE at each step boundary BEFORE any subsequent op can
        // launder it — a per-step G13 sweep is therefore a complete, honest
        // transport for Linear/Activation (no twin dispatches needed). Conv2d
        // refuses above because its fused interior has no host-visible boundary.
        // Words are OR-only; the entry sweep covers the seed.
        let mut taint_rows = vec![0u32; num_specs];
        let word_rows =
            |rows: &mut Vec<u32>, la: &[f32], ua: &[f32], le: &[f32], ue: &[f32], d: usize| {
                for s in 0..num_specs {
                    let mut w = 0u32;
                    for j in 0..d {
                        for v in [la[s * d + j], ua[s * d + j], le[s * d + j], ue[s * d + j]] {
                            w |= u32::from(!v.is_finite() || v.abs() >= ny_core::CROWN_COEFF_MAX);
                        }
                    }
                    rows[s] |= w;
                }
            };
        word_rows(
            &mut taint_rows,
            &lower_a,
            &upper_a,
            &lower_err,
            &upper_err,
            dim,
        );
        for layer in layers {
            // #u4: word the CURRENT frontier before this layer transforms it
            // (first iteration duplicates the entry sweep — OR is idempotent).
            word_rows(
                &mut taint_rows,
                &lower_a,
                &upper_a,
                &lower_err,
                &upper_err,
                dim,
            );
            match layer {
                GpuCrownLayer::Linear {
                    weight,
                    bias,
                    out_features,
                    in_features,
                    ..
                } => {
                    if dim != *out_features {
                        return Err(NyError::shape_mismatch(vec![*out_features], vec![dim]));
                    }
                    // Bias contribution (uses the CURRENT coefficient, before A·W).
                    if let Some(bias) = bias {
                        bias_fold_f64(
                            num_specs,
                            *out_features,
                            &lower_a,
                            &lower_err,
                            bias,
                            &mut lb,
                            &mut lb_err,
                        );
                        bias_fold_f64(
                            num_specs,
                            *out_features,
                            &upper_a,
                            &upper_err,
                            bias,
                            &mut ub,
                            &mut ub_err,
                        );
                    }
                    // A_new = A @ W with certified error propagation, on the GPU.
                    let (nla, nle) = diagnostic_gemm.crown_aw_error_step(
                        num_specs,
                        *out_features,
                        *in_features,
                        &lower_a,
                        &lower_err,
                        weight,
                    )?;
                    let (nua, nue) = diagnostic_gemm.crown_aw_error_step(
                        num_specs,
                        *out_features,
                        *in_features,
                        &upper_a,
                        &upper_err,
                        weight,
                    )?;
                    lower_a = nla;
                    upper_a = nua;
                    lower_err = nle;
                    upper_err = nue;
                    dim = *in_features;
                }
                GpuCrownLayer::Activation {
                    lower_slope,
                    upper_slope,
                    lower_intercept,
                    upper_intercept,
                    num_neurons,
                } => {
                    if dim != *num_neurons {
                        return Err(NyError::shape_mismatch(vec![*num_neurons], vec![dim]));
                    }
                    // Intercept contribution to bias (uses current coefficient),
                    // sign-routed. As in `bias_fold_f64`, the fold's own f64
                    // rounding is certified by `γ_{n+1}` over the computed
                    // magnitudes (incoming bound included), on top of the
                    // propagated coefficient error (`err·(|li|+|ui|)`, covering a
                    // sign flip of `a` under its error).
                    let gamma = gamma_k_f64(*num_neurons + 1);
                    let real_factor = 1.0 + 2.0 * gamma;
                    let additive = 8.0 * ((*num_neurons + 1) as f64) * ETA64;
                    for s in 0..num_specs {
                        let mut lmag = lb[s].abs();
                        let mut umag = ub[s].abs();
                        for i in 0..*num_neurons {
                            let la = lower_a[s * num_neurons + i];
                            let ua = upper_a[s * num_neurons + i];
                            let li = if la >= 0.0 {
                                lower_intercept[i]
                            } else {
                                upper_intercept[i]
                            };
                            let ui = if ua >= 0.0 {
                                upper_intercept[i]
                            } else {
                                lower_intercept[i]
                            };
                            let lprod = f64::from(la) * f64::from(li);
                            let uprod = f64::from(ua) * f64::from(ui);
                            lb[s] += lprod;
                            ub[s] += uprod;
                            lmag += lprod.abs();
                            umag += uprod.abs();
                            let int_sum = f64::from(lower_intercept[i]).abs()
                                + f64::from(upper_intercept[i]).abs();
                            lb_err[s] += f64::from(lower_err[s * num_neurons + i]) * int_sum;
                            ub_err[s] += f64::from(upper_err[s * num_neurons + i]) * int_sum;
                        }
                        lb_err[s] += gamma * lmag * real_factor + additive;
                        ub_err[s] += gamma * umag * real_factor + additive;
                    }
                    let (nla, nua, nle, nue) = crown_activation_error_step(
                        num_specs,
                        *num_neurons,
                        &lower_a,
                        &upper_a,
                        &lower_err,
                        &upper_err,
                        lower_slope,
                        upper_slope,
                    )?;
                    lower_a = nla;
                    upper_a = nua;
                    lower_err = nle;
                    upper_err = nue;
                    // dim unchanged (activation is elementwise).
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
                    let out_dim = out_channels * out_h * out_w;
                    if dim != out_dim {
                        return Err(NyError::shape_mismatch(vec![out_dim], vec![dim]));
                    }
                    // Bias contribution (uses the current coefficient over OC·OH·OW).
                    if let Some(bias) = bias_expanded {
                        bias_fold_f64(
                            num_specs,
                            out_dim,
                            &lower_a,
                            &lower_err,
                            bias,
                            &mut lb,
                            &mut lb_err,
                        );
                        bias_fold_f64(
                            num_specs,
                            out_dim,
                            &upper_a,
                            &upper_err,
                            bias,
                            &mut ub,
                            &mut ub_err,
                        );
                    }
                    // A_new = conv_transpose(A, W) on the GPU (reshape → GEMM → col2im).
                    let params = ConvTranspose2dParams {
                        num_specs,
                        out_channels: *out_channels,
                        in_channels: *in_channels,
                        out_h: *out_h,
                        out_w: *out_w,
                        in_h: *in_h,
                        in_w: *in_w,
                        kernel_h: *kernel_h,
                        kernel_w: *kernel_w,
                        stride_h: *stride_h,
                        stride_w: *stride_w,
                        pad_h: *pad_h,
                        pad_w: *pad_w,
                    };
                    let rl = reshape_for_conv(&lower_a, num_specs, *out_channels, *out_h, *out_w);
                    let ru = reshape_for_conv(&upper_a, num_specs, *out_channels, *out_h, *out_w);
                    let nla = self.conv_transpose_2d(&rl, weight_col, &params)?;
                    let nua = self.conv_transpose_2d(&ru, weight_col, &params)?;
                    let new_dim = in_channels * in_h * in_w;

                    // Certified conv-transpose coefficient error (CPU over-bound,
                    // 0/6M-trial validated): per row,
                    //   γ_{OC·KH·KW}·max_k|a[s,k]|·‖W‖₁ + max_k|err[s,k]|·‖W‖₁,
                    // constant over the IC·IH·IW outputs (a sound, ULP-scale-looser
                    // bound that avoids a second conv pass).
                    const U: f64 = f64::from_bits(0x3E70_0000_0000_0000); // 2^-24
                    let n_contraction = (out_channels * kernel_h * kernel_w) as f64;
                    let nu = n_contraction * U;
                    let gamma = if nu < 0.5 { nu / (1.0 - nu) } else { 2.0 * nu };
                    let kernel_l1: f64 = weight_col.iter().map(|v| f64::from(*v).abs()).sum();
                    let row_max = |buf: &[f32], s: usize| -> f64 {
                        let mut m = 0.0f64;
                        for k in 0..out_dim {
                            let v = f64::from(buf[s * out_dim + k]).abs();
                            if v > m {
                                m = v;
                            }
                        }
                        m
                    };
                    let mut nle = vec![0.0f32; num_specs * new_dim];
                    let mut nue = vec![0.0f32; num_specs * new_dim];
                    for s in 0..num_specs {
                        let el = up(gamma * row_max(&lower_a, s) * kernel_l1
                            + row_max(&lower_err, s) * kernel_l1);
                        let eu = up(gamma * row_max(&upper_a, s) * kernel_l1
                            + row_max(&upper_err, s) * kernel_l1);
                        for p in 0..new_dim {
                            nle[s * new_dim + p] = el;
                            nue[s * new_dim + p] = eu;
                        }
                    }
                    lower_a = nla;
                    upper_a = nua;
                    lower_err = nle;
                    upper_err = nue;
                    dim = new_dim;
                }
                _ => {
                    return Err(NyError::UnsupportedOp(
                        "crown_backward_sound_host: Linear/Activation/Conv2d only (host form)"
                            .into(),
                    ));
                }
            }
        }

        // (per-step #u4 sweep for the LAST layer's outputs.)
        word_rows(
            &mut taint_rows,
            &lower_a,
            &upper_a,
            &lower_err,
            &upper_err,
            dim,
        );

        // Fold the certified bias error into the bias passed to the concretize
        // (lower widened DOWN, upper widened UP, with outward f32 rounding). The
        // concretize then widens by the coefficient error + its own dot rounding.
        let bias_lower: Vec<f32> = (0..num_specs).map(|s| down(lb[s] - lb_err[s])).collect();
        let bias_upper: Vec<f32> = (0..num_specs).map(|s| up(ub[s] + ub_err[s])).collect();

        self.concretize_sound_gpu(
            num_specs,
            dim,
            &lower_a,
            &upper_a,
            &lower_err,
            &upper_err,
            input_lower,
            input_upper,
            &bias_lower,
            &bias_upper,
            // #u4 C1 (armed 2026-08-11 UTC): the per-step host G13 sweep above is
            // this driver's complete word transport (host-visible step
            // boundaries — see the sweep's doc). Bias words are covered by the
            // sweeps of the a/err arrays the folds consumed.
            Some(&taint_rows),
        )
    }
}

// CPU-only unit tests for the #u4 host taint-fold companions (no GPU device
// required): annihilation on exactly-zero partners, OR accumulation into
// pre-set words, and indexing cross-checked against the value fold on a
// hand-computed 2-spec-row example.
#[cfg(test)]
mod taint_fold_tests {
    use super::{bias_fold_f64, bias_fold_taint, conv_rowmax_error_taint, intercept_fold_taint};

    /// `bias[k] == 0.0` (either sign of zero) annihilates BOTH the `a` and the
    /// `err` word for that column (canon: `R·0 == 0`).
    #[test]
    fn bias_fold_taint_zero_bias_annihilates() {
        let a_taint = vec![0xffff_ffff_u32; 3];
        let a_err_taint = vec![0xffff_ffff_u32; 3];
        let bias = vec![0.0f32, -0.0, 0.0];
        let mut acc_taint = vec![0u32];
        bias_fold_taint(1, 3, &a_taint, &a_err_taint, &bias, &mut acc_taint);
        assert_eq!(acc_taint, vec![0], "exact-zero partners must annihilate");
    }

    /// Hand-computed 2-spec-row example cross-checked against the REAL value
    /// fold: with unique bits per element, the value fold's `acc` (exact —
    /// each product is a small integer) pins which `bias[k]` each `a[s,k]`
    /// multiplied, and the companion's word must be the OR of exactly the
    /// bits whose partner was nonzero, accumulated into the pre-set word.
    #[test]
    fn bias_fold_taint_matches_value_fold_indexing_and_accumulates() {
        let (num_specs, n) = (2usize, 3usize);
        let bias = vec![1.0f32, 0.0, 2.0];
        // Row 0: a = [1,1,1] → acc = 1·1 + 1·0 + 1·2 = 3.
        // Row 1: a = [1,2,3] → acc = 1·1 + 2·0 + 3·2 = 7.
        let a = vec![1.0f32, 1.0, 1.0, 1.0, 2.0, 3.0];
        let a_err = vec![0.0f32; num_specs * n];
        let mut acc = vec![0.0f64; num_specs];
        let mut acc_err = vec![0.0f64; num_specs];
        bias_fold_f64(num_specs, n, &a, &a_err, &bias, &mut acc, &mut acc_err);
        assert_eq!(acc, vec![3.0, 7.0], "value fold pins the s*n+k partner map");

        // a bits 0..5, err bits 8..13 (element i ⇒ a bit i, err bit 8+i).
        let a_taint: Vec<u32> = (0..num_specs * n).map(|i| 1u32 << i).collect();
        let a_err_taint: Vec<u32> = (0..num_specs * n).map(|i| 1u32 << (8 + i)).collect();
        let mut acc_taint = vec![0x8000_0000_u32, 0x4000_0000];
        bias_fold_taint(num_specs, n, &a_taint, &a_err_taint, &bias, &mut acc_taint);
        // Columns 0 and 2 survive (bias 1.0 / 2.0); column 1 annihilates.
        assert_eq!(
            acc_taint[0],
            0x8000_0000 | (1 << 0) | (1 << 2) | (1 << 8) | (1 << 10)
        );
        assert_eq!(
            acc_taint[1],
            0x4000_0000 | (1 << 3) | (1 << 5) | (1 << 11) | (1 << 13)
        );
    }

    /// Annihilation at the intercept fold needs BOTH intercepts exactly zero:
    /// a tainted `a` has an untrustworthy sign, so either intercept may be its
    /// true partner (the common ReLU `lower_intercept == 0` alone keeps the
    /// word). OR accumulates into the pre-set per-spec word.
    #[test]
    fn intercept_fold_taint_needs_both_intercepts_zero_to_annihilate() {
        let (num_specs, neurons) = (2usize, 4usize);
        // Neuron: 0 → both zero (annihilate), 1 → lower only nonzero,
        // 2 → upper only nonzero, 3 → both zero via -0.0 (annihilate).
        let li = vec![0.0f32, 0.25, 0.0, -0.0];
        let ui = vec![0.0f32, 0.0, 0.5, 0.0];
        let a_taint: Vec<u32> = (0..num_specs * neurons).map(|i| 1u32 << i).collect();
        let err_taint: Vec<u32> = (0..num_specs * neurons).map(|i| 1u32 << (16 + i)).collect();
        let mut acc_taint = vec![0x8000_0000_u32, 0x4000_0000];
        intercept_fold_taint(
            num_specs,
            neurons,
            &a_taint,
            &err_taint,
            &li,
            &ui,
            &mut acc_taint,
        );
        // Row 0 = elements 0..3, row 1 = elements 4..7; neurons 1 and 2 keep.
        assert_eq!(
            acc_taint[0],
            0x8000_0000 | (1 << 1) | (1 << 2) | (1 << 17) | (1 << 18)
        );
        assert_eq!(
            acc_taint[1],
            0x4000_0000 | (1 << 5) | (1 << 6) | (1 << 21) | (1 << 22)
        );
    }

    /// An exactly-zero kernel L1 annihilates the whole conv row-max fold
    /// (every product is exactly 0); the words must not move.
    #[test]
    fn conv_rowmax_taint_zero_kernel_annihilates() {
        let a_taint = vec![0xffu32; 6];
        let err_taint = vec![0xff00u32; 6];
        let mut row_taint = vec![0x1u32, 0x2];
        conv_rowmax_error_taint(2, 3, &a_taint, &err_taint, 0.0, &mut row_taint);
        assert_eq!(row_taint, vec![0x1, 0x2], "zero kernel must annihilate");
    }

    /// With a nonzero kernel, ANY tainted element in a spec row taints that
    /// row's single broadcast word (the row MAX may have selected — or been
    /// hidden by — it), rows stay independent, and OR accumulates.
    #[test]
    fn conv_rowmax_taint_spreads_across_its_own_row_only() {
        let (num_specs, out_dim) = (2usize, 3usize);
        // Row 0: a-taint at column 1 only. Row 1: err-taint at column 2 only.
        let mut a_taint = vec![0u32; num_specs * out_dim];
        let mut err_taint = vec![0u32; num_specs * out_dim];
        a_taint[1] = 0x2;
        err_taint[out_dim + 2] = 0x20; // row 1, column 2
        let mut row_taint = vec![0x1u32, 0x0];
        conv_rowmax_error_taint(
            num_specs,
            out_dim,
            &a_taint,
            &err_taint,
            2.5,
            &mut row_taint,
        );
        assert_eq!(row_taint[0], 0x1 | 0x2, "row 0 word ORs its own taint");
        assert_eq!(row_taint[1], 0x20, "row 1 taint must not leak into row 0");
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod tests {
    use super::*;
    use crate::wgpu_device::test_support::{gpu_test_serial_guard, require_device};
    use std::sync::Arc;

    /// Linear-only network: the composed function is exactly affine, so CROWN is
    /// exact and the output range over the box is `Σ |w_total|·radius` around the
    /// center. The sound backward's `(lower, upper)` must enclose every sampled
    /// output (a necessary soundness condition that catches over-tight bounds).
    #[test]
    fn crown_backward_sound_host_linear_only_is_sound() {
        let _g = gpu_test_serial_guard();
        let device = require_device();
        let mut state: u64 = 0x50FA_1234;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        for _ in 0..6 {
            let (din, dh, dout) = (4usize, 5usize, 3usize);
            // Backward order: output layer first. Net = L2(relu-free)(L1(x)).
            // L1: din->dh, L2: dh->dout. Backward processes L2 then L1.
            let w1: Vec<f32> = (0..dh * din).map(|_| rng() * 0.8).collect(); // (dh × din)
            let b1: Vec<f32> = (0..dh).map(|_| rng() * 0.5).collect();
            let w2: Vec<f32> = (0..dout * dh).map(|_| rng() * 0.8).collect(); // (dout × dh)
            let b2: Vec<f32> = (0..dout).map(|_| rng() * 0.5).collect();

            let layers = vec![
                GpuCrownLayer::Linear {
                    weight: Arc::from(w2.clone().into_boxed_slice()),
                    bias: Some(Arc::from(b2.clone().into_boxed_slice())),
                    out_features: dout,
                    in_features: dh,
                    cert_err: Default::default(),
                },
                GpuCrownLayer::Linear {
                    weight: Arc::from(w1.clone().into_boxed_slice()),
                    bias: Some(Arc::from(b1.clone().into_boxed_slice())),
                    out_features: dh,
                    in_features: din,
                    cert_err: Default::default(),
                },
            ];
            // spec = identity (dout × dout): bound each output neuron.
            let mut spec = vec![0.0f32; dout * dout];
            for i in 0..dout {
                spec[i * dout + i] = 1.0;
            }
            let xc: Vec<f32> = (0..din).map(|_| rng()).collect();
            let xr: Vec<f32> = (0..din).map(|_| (rng() * 0.3).abs() + 0.05).collect();
            let xl: Vec<f32> = (0..din).map(|i| xc[i] - xr[i]).collect();
            let xu: Vec<f32> = (0..din).map(|i| xc[i] + xr[i]).collect();

            let (lo, hi) = device
                .crown_backward_sound_host(&layers, &spec, dout, dout, &xl, &xu)
                .expect("sound backward");

            // Forward-evaluate the affine net at many sampled inputs.
            let eval = |x: &[f32]| -> Vec<f32> {
                let mut h = vec![0.0f32; dh];
                for j in 0..dh {
                    let mut s = b1[j];
                    for i in 0..din {
                        s += w1[j * din + i] * x[i];
                    }
                    h[j] = s;
                }
                let mut o = vec![0.0f32; dout];
                for j in 0..dout {
                    let mut s = b2[j];
                    for i in 0..dh {
                        s += w2[j * dh + i] * h[i];
                    }
                    o[j] = s;
                }
                o
            };
            for t in 0..200 {
                let x: Vec<f32> = (0..din)
                    .map(|i| {
                        let f = ((t * 31 + i * 17) % 100) as f32 / 99.0;
                        xl[i] + f * (xu[i] - xl[i])
                    })
                    .collect();
                let o = eval(&x);
                for k in 0..dout {
                    assert!(
                        lo[k] <= o[k] + 1e-3 && o[k] <= hi[k] + 1e-3,
                        "UNSOUND: output[{k}]={} not in [{}, {}]",
                        o[k],
                        lo[k],
                        hi[k]
                    );
                }
            }
        }
    }

    /// The armed host proof seam must refuse Conv2d before its unworded fused
    /// GEMM-to-col2im interior. The older numerical oracle remains below the
    /// gate for any future implementation that adds complete internal words.
    #[test]
    fn crown_backward_sound_host_conv_refuses_without_internal_words() {
        let _g = gpu_test_serial_guard();
        let device = require_device();
        let mut state: u64 = 0xC0_5151;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        // IC=1, OC=2, KH=KW=2, IH=IW=3 -> OH=OW=2 (stride 1, pad 0).
        let (ic, oc, kh, kw, ih, iw) = (1usize, 2usize, 2usize, 2usize, 3usize, 3usize);
        let (oh, ow) = (ih - kh + 1, iw - kw + 1);
        let out_dim = oc * oh * ow; // 8
        let in_dim = ic * ih * iw; // 9
        for _ in 0..5 {
            let weight_col: Vec<f32> = (0..oc * ic * kh * kw).map(|_| rng() * 0.9).collect(); // (oc × ic*kh*kw)

            let layers = vec![GpuCrownLayer::Conv2d {
                weight_col: Arc::from(weight_col.clone().into_boxed_slice()),
                bias_expanded: None,
                out_channels: oc,
                in_channels: ic,
                kernel_h: kh,
                kernel_w: kw,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
                out_h: oh,
                out_w: ow,
                in_h: ih,
                in_w: iw,
                cert_err: Default::default(),
            }];
            // spec = identity over the out_dim output neurons.
            let mut spec = vec![0.0f32; out_dim * out_dim];
            for i in 0..out_dim {
                spec[i * out_dim + i] = 1.0;
            }
            let xc: Vec<f32> = (0..in_dim).map(|_| rng()).collect();
            let xl: Vec<f32> = xc.iter().map(|&c| c - 0.2).collect();
            let xu: Vec<f32> = xc.iter().map(|&c| c + 0.2).collect();

            let result =
                device.crown_backward_sound_host(&layers, &spec, out_dim, out_dim, &xl, &xu);
            if PRODUCTION_GUARDS_CONSULT_TAINT_WORD {
                let error = result.expect_err("armed host Conv2d must fail closed");
                assert!(
                    matches!(&error, NyError::UnsupportedOp(message) if message.contains("GEMM-to-col2im")),
                    "unexpected host Conv2d refusal: {error:?}"
                );
                return;
            }
            let (lo, hi) = result.expect("sound conv backward");

            // conv forward: out[oc,oh,ow] = Σ_{kh,kw} W[oc, kh*KW+kw] · x[(oh+kh)*IW + (ow+kw)]
            let eval = |x: &[f32]| -> Vec<f32> {
                let mut out = vec![0.0f32; out_dim];
                for c in 0..oc {
                    for yy in 0..oh {
                        for xx in 0..ow {
                            let mut s = 0.0f32;
                            for a in 0..kh {
                                for b in 0..kw {
                                    let wv = weight_col[c * (ic * kh * kw) + a * kw + b];
                                    s += wv * x[(yy + a) * iw + (xx + b)];
                                }
                            }
                            out[c * oh * ow + yy * ow + xx] = s;
                        }
                    }
                }
                out
            };
            for t in 0..200 {
                let x: Vec<f32> = (0..in_dim)
                    .map(|i| {
                        let f = ((t * 37 + i * 13) % 100) as f32 / 99.0;
                        xl[i] + f * (xu[i] - xl[i])
                    })
                    .collect();
                let o = eval(&x);
                for k in 0..out_dim {
                    assert!(
                        lo[k] <= o[k] + 1e-3 && o[k] <= hi[k] + 1e-3,
                        "UNSOUND conv: output[{k}]={} not in [{}, {}]",
                        o[k],
                        lo[k],
                        hi[k]
                    );
                }
            }
        }
    }
}
