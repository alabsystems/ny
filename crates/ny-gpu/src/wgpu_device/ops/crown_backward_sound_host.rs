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
    /// Sound CROWN backward over Linear/Activation layers (backward order), with
    /// GPU GEMMs. Returns `(lower, upper)`, one sound bound per spec row.
    ///
    /// `spec` is the initial coefficient matrix `(num_specs × output_dim)`
    /// row-major (the network-output selector C). Conv2d / MaxPool / dual-alpha
    /// layers are not handled by this host form yet (the resident dispatch will).
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

        for layer in layers {
            match layer {
                GpuCrownLayer::Linear {
                    weight,
                    bias,
                    out_features,
                    in_features,
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
                    let (nla, nle) = self.crown_aw_error_step(
                        num_specs,
                        *out_features,
                        *in_features,
                        &lower_a,
                        &lower_err,
                        weight,
                    )?;
                    let (nua, nue) = self.crown_aw_error_step(
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
        )
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
                },
                GpuCrownLayer::Linear {
                    weight: Arc::from(w1.clone().into_boxed_slice()),
                    bias: Some(Arc::from(b1.clone().into_boxed_slice())),
                    out_features: dh,
                    in_features: din,
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

    /// Single Conv2d layer (affine): the sound backward's per-output-neuron bounds
    /// must enclose the conv forward value over the input box. Validates the conv
    /// reshape + conv_transpose_2d + the over-bound coefficient error.
    #[test]
    fn crown_backward_sound_host_single_conv_is_sound() {
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
            }];
            // spec = identity over the out_dim output neurons.
            let mut spec = vec![0.0f32; out_dim * out_dim];
            for i in 0..out_dim {
                spec[i * out_dim + i] = 1.0;
            }
            let xc: Vec<f32> = (0..in_dim).map(|_| rng()).collect();
            let xl: Vec<f32> = xc.iter().map(|&c| c - 0.2).collect();
            let xu: Vec<f32> = xc.iter().map(|&c| c + 0.2).collect();

            let (lo, hi) = device
                .crown_backward_sound_host(&layers, &spec, out_dim, out_dim, &xl, &xu)
                .expect("sound conv backward");

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
