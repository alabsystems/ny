// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #flush-charge Lane B: end-to-end ENCLOSURE-PARITY acceptance harness on the
//! TEST-SCOPED charged device (the evidence that closed the enclosure-parity
//! row on the charged gate's audit ledger in `ops/sound_authority.rs`; the
//! row was CLOSED by the 2026-08-13 opening review and this harness remains
//! the regression oracle).
//!
//! The device under test is built by the PRODUCTION constructor
//! [`WgpuDevice::new_for_verdict_flush_charged`] (reachable since the
//! 2026-08-13 opening review): the full production charged admission, so
//! `FlushChargePolicy::production()` is armed, `charged_walk_guard` runs at
//! walk entry, and every audited widening (`w_l1 ×4`, concretize `×8`,
//! bias-combine `×4`, act-bias `×4`) is LIVE on the exact production-armed
//! path. (Pre-flip, the harness ran on the test-scoped twin — the production
//! predicate minus only the source gate; the twin remains for
//! gate-independent evidence.)
//!
//! # The acceptance claim measured here
//!
//! For every admitted fixture, the charged GPU backward + concretize
//! (`crown_backward_gpu_sound`) must return an enclosure that CONTAINS the
//! near-exact CPU f64 reference of the SAME CROWN fold elementwise:
//!
//! ```text
//! gpu_lower[i] <= f64_lower[i]  AND  gpu_upper[i] >= f64_upper[i]
//! ```
//!
//! with NO tolerance: the charged covers claim to pay every f32 rounding and
//! DAZ-flush loss outward, and the f64 reference's own rounding (~2^-52
//! relative) is orders of magnitude inside those charges. Any breach is a
//! finding and is reported VERBATIM.
//!
//! Fixture matrix: >= 200 seeded random fixtures across Linear/ReLU chains,
//! general activations with nonzero NORMAL intercepts (the §E re-admission),
//! and Conv2d chains — plus a structured ADMITTED-BOUNDARY set (input-box
//! endpoints, weights, and intercepts at exactly ±2^-126 = `f32::MIN_POSITIVE`
//! and tiny-normal multiples, mixed signs). Strictly-subnormal
//! weights/bias/slopes/intercepts/inputs are REFUSED by the charge policy, so
//! the boundary set sits exactly at the smallest admitted magnitudes.

use std::sync::{Arc, OnceLock};

use super::test_support::gpu_test_serial_guard;
use super::WgpuDevice;
use ny_core::{GpuCrownBackward, GpuCrownLayer};

static CHARGED_DEVICE: OnceLock<Result<Arc<WgpuDevice>, String>> = OnceLock::new();

/// Outcome of the acceptance-device construction attempt in THIS process
/// environment.
enum ChargedAcceptance {
    /// The TEST-SCOPED device armed: run the acceptance body.
    Armed(Arc<WgpuDevice>),
    /// The ONE recognized environmental precondition failure
    /// (admission-config): the user explicitly pinned
    /// `NY_GPU_DENORM_PRESERVE=1`, which the charged constructors typed-refuse
    /// (env wins — the pinned passthrough configuration is not the one the
    /// oracle charges model). Under the DEFAULT env the twin builds its
    /// device with the plain-WGSL path FORCED per-device, so the historical
    /// Metal AUTO passthrough poison can no longer refuse it: the acceptance
    /// body now runs with no env at all.
    EnvRequiredPinnedRefusal(String),
}

/// The shared charged acceptance device, or the verified environmental
/// refusal. ANY refusal other than the recognized explicit
/// `NY_GPU_DENORM_PRESERVE=1` pin is a hard failure — never a skip.
///
/// Since the 2026-08-13 opening review this is the PRODUCTION charged
/// constructor: the harness measures the exact production-armed path (the
/// test-scoped twin remains for gate-independent evidence and the ny-cli
/// wall-clock harness).
fn charged_acceptance() -> ChargedAcceptance {
    let outcome = CHARGED_DEVICE.get_or_init(|| {
        WgpuDevice::new_for_verdict_flush_charged(super::WgpuChargedVerdictRequest::new())
            .map(Arc::new)
            .map_err(|error| format!("{error}"))
    });
    match outcome {
        Ok(device) => ChargedAcceptance::Armed(Arc::clone(device)),
        Err(error) => {
            assert!(
                error.contains("NY_GPU_DENORM_PRESERVE"),
                "charged acceptance device REFUSED on this adapter for a \
                 reason OTHER than the recognized explicit \
                 NY_GPU_DENORM_PRESERVE=1 pin (the forced plain-WGSL device \
                 must arm under the default env on this box): {error}"
            );
            println!(
                "[flush-charge acceptance] PRECONDITION NOT MET in this process \
                 environment: NY_GPU_DENORM_PRESERVE=1 explicitly pins the \
                 passthrough loading path, which the charges cannot cover — \
                 VERIFIED typed refusal: {error}\n\
                 [flush-charge acceptance] unset NY_GPU_DENORM_PRESERVE (or set \
                 auto/0) and re-run: cargo test -p ny-gpu --features gpu-tests \
                 --lib flush_charge_acceptance_gpu_tests -- --nocapture"
            );
            ChargedAcceptance::EnvRequiredPinnedRefusal(error.clone())
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic RNG (xorshift64*), fixed seeds — reproducible fixture matrix.
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [lo, hi).
    fn f32_in(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * unit
    }

    /// Uniform integer in [lo, hi] inclusive.
    fn usize_in(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() as usize) % (hi - lo + 1)
    }
}

/// Zero any strictly-subnormal f32 (the charge policy refuses them; exact
/// zeros and normals — including ±2^-126 — are admitted). Fixture generation
/// with the magnitudes below cannot produce subnormals, but this keeps the
/// admitted-fixture invariant structural rather than probabilistic.
fn drop_subnormals(values: &mut [f32]) {
    for v in values {
        if *v != 0.0 && v.abs() < f32::MIN_POSITIVE {
            *v = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// CPU f64 reference: the SAME CROWN fold (identical slopes/intercepts and
// branch rules as the GPU walk's value lane), accumulated in f64.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn f64_linear_backward(
    a_l: &mut Vec<f64>,
    a_u: &mut Vec<f64>,
    b_l: &mut [f64],
    b_u: &mut [f64],
    weight: &[f32],
    bias: Option<&[f32]>,
    num_specs: usize,
    out_f: usize,
    in_f: usize,
) {
    if let Some(layer_bias) = bias {
        for s in 0..num_specs {
            let (mut lb, mut ub) = (0.0f64, 0.0f64);
            for j in 0..out_f {
                lb += a_l[s * out_f + j] * f64::from(layer_bias[j]);
                ub += a_u[s * out_f + j] * f64::from(layer_bias[j]);
            }
            b_l[s] += lb;
            b_u[s] += ub;
        }
    }
    let mut new_l = vec![0.0f64; num_specs * in_f];
    let mut new_u = vec![0.0f64; num_specs * in_f];
    for s in 0..num_specs {
        for c in 0..in_f {
            let (mut sl, mut su) = (0.0f64, 0.0f64);
            for k in 0..out_f {
                let w = f64::from(weight[k * in_f + c]);
                sl += a_l[s * out_f + k] * w;
                su += a_u[s * out_f + k] * w;
            }
            new_l[s * in_f + c] = sl;
            new_u[s * in_f + c] = su;
        }
    }
    *a_l = new_l;
    *a_u = new_u;
}

#[allow(clippy::too_many_arguments)]
fn f64_activation_backward(
    a_l: &mut [f64],
    a_u: &mut [f64],
    b_l: &mut [f64],
    b_u: &mut [f64],
    ls: &[f32],
    us: &[f32],
    li: &[f32],
    ui: &[f32],
    num_specs: usize,
    n: usize,
) {
    for s in 0..num_specs {
        let (mut lb, mut ub) = (0.0f64, 0.0f64);
        for j in 0..n {
            let idx = s * n + j;
            let (al, au) = (a_l[idx], a_u[idx]);
            if al >= 0.0 {
                a_l[idx] = al * f64::from(ls[j]);
                lb += al * f64::from(li[j]);
            } else {
                a_l[idx] = al * f64::from(us[j]);
                lb += al * f64::from(ui[j]);
            }
            if au >= 0.0 {
                a_u[idx] = au * f64::from(us[j]);
                ub += au * f64::from(ui[j]);
            } else {
                a_u[idx] = au * f64::from(ls[j]);
                ub += au * f64::from(li[j]);
            }
        }
        b_l[s] += lb;
        b_u[s] += ub;
    }
}

/// f64 reference for the whole backward + concretize over `layers`
/// (backward order), for the Linear/Activation/Conv2d kinds the charged walk
/// admits.
fn f64_crown_backward(
    layers: &[GpuCrownLayer],
    spec: &[f32],
    num_specs: usize,
    input_lower: &[f32],
    input_upper: &[f32],
) -> (Vec<f64>, Vec<f64>) {
    let mut a_l: Vec<f64> = spec.iter().map(|&v| f64::from(v)).collect();
    let mut a_u = a_l.clone();
    let mut b_l = vec![0.0f64; num_specs];
    let mut b_u = vec![0.0f64; num_specs];
    let mut dim = a_l.len() / num_specs;

    for layer in layers {
        match layer {
            GpuCrownLayer::Linear {
                weight,
                bias,
                out_features,
                in_features,
                ..
            } => {
                f64_linear_backward(
                    &mut a_l,
                    &mut a_u,
                    &mut b_l,
                    &mut b_u,
                    weight,
                    bias.as_deref(),
                    num_specs,
                    *out_features,
                    *in_features,
                );
                dim = *in_features;
            }
            GpuCrownLayer::Activation {
                lower_slope,
                upper_slope,
                lower_intercept,
                upper_intercept,
                num_neurons,
            } => {
                f64_activation_backward(
                    &mut a_l,
                    &mut a_u,
                    &mut b_l,
                    &mut b_u,
                    lower_slope,
                    upper_slope,
                    lower_intercept,
                    upper_intercept,
                    num_specs,
                    *num_neurons,
                );
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
                let spatial = out_h * out_w;
                let kernel_cols = in_channels * kernel_h * kernel_w;
                let flat_input_dim = in_channels * in_h * in_w;

                if let Some(expanded_bias) = bias_expanded {
                    for s in 0..num_specs {
                        let (mut lb, mut ub) = (0.0f64, 0.0f64);
                        for j in 0..(out_channels * spatial) {
                            lb += a_l[s * out_channels * spatial + j] * f64::from(expanded_bias[j]);
                            ub += a_u[s * out_channels * spatial + j] * f64::from(expanded_bias[j]);
                        }
                        b_l[s] += lb;
                        b_u[s] += ub;
                    }
                }

                let mut new_l = vec![0.0f64; num_specs * flat_input_dim];
                let mut new_u = vec![0.0f64; num_specs * flat_input_dim];
                for s in 0..num_specs {
                    for ic in 0..*in_channels {
                        for ih in 0..*in_h {
                            for iw_pos in 0..*in_w {
                                let flat_idx = ic * in_h * in_w + ih * in_w + iw_pos;
                                let (mut sum_l, mut sum_u) = (0.0f64, 0.0f64);
                                for ki in 0..*kernel_h {
                                    let ih_plus_ph = ih + pad_h;
                                    if ih_plus_ph < ki {
                                        continue;
                                    }
                                    let num_h = ih_plus_ph - ki;
                                    if num_h % stride_h != 0 {
                                        continue;
                                    }
                                    let gy = num_h / stride_h;
                                    if gy >= *out_h {
                                        continue;
                                    }
                                    for kj in 0..*kernel_w {
                                        let iw_plus_pw = iw_pos + pad_w;
                                        if iw_plus_pw < kj {
                                            continue;
                                        }
                                        let num_w = iw_plus_pw - kj;
                                        if num_w % stride_w != 0 {
                                            continue;
                                        }
                                        let gx = num_w / stride_w;
                                        if gx >= *out_w {
                                            continue;
                                        }
                                        for oc in 0..*out_channels {
                                            let w = f64::from(
                                                weight_col[oc * kernel_cols
                                                    + ic * kernel_h * kernel_w
                                                    + ki * kernel_w
                                                    + kj],
                                            );
                                            let src = s * out_channels * spatial
                                                + oc * spatial
                                                + gy * out_w
                                                + gx;
                                            sum_l += a_l[src] * w;
                                            sum_u += a_u[src] * w;
                                        }
                                    }
                                }
                                new_l[s * flat_input_dim + flat_idx] = sum_l;
                                new_u[s * flat_input_dim + flat_idx] = sum_u;
                            }
                        }
                    }
                }
                a_l = new_l;
                a_u = new_u;
                dim = flat_input_dim;
            }
            _ => panic!("f64 reference: unsupported fixture layer kind"),
        }
    }

    let mut lower = vec![0.0f64; num_specs];
    let mut upper = vec![0.0f64; num_specs];
    for s in 0..num_specs {
        let (mut lb, mut ub) = (b_l[s], b_u[s]);
        for j in 0..dim {
            let (al, au) = (a_l[s * dim + j], a_u[s * dim + j]);
            let (xl, xu) = (f64::from(input_lower[j]), f64::from(input_upper[j]));
            lb += al.max(0.0) * xl + al.min(0.0) * xu;
            ub += au.max(0.0) * xu + au.min(0.0) * xl;
        }
        lower[s] = lb;
        upper[s] = ub;
    }
    (lower, upper)
}

// ---------------------------------------------------------------------------
// Fixture builders (all values sanitized to the ADMITTED domain).
// ---------------------------------------------------------------------------

fn identity_spec(dim: usize) -> Vec<f32> {
    let mut spec = vec![0.0f32; dim * dim];
    for i in 0..dim {
        spec[i * dim + i] = 1.0;
    }
    spec
}

/// f32 IBP forward through one linear layer (pre-activation bounds).
fn ibp_linear(
    weight: &[f32],
    bias: &[f32],
    inp_l: &[f32],
    inp_u: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut pre_l = vec![0.0f32; out_dim];
    let mut pre_u = vec![0.0f32; out_dim];
    for j in 0..out_dim {
        let (mut lb, mut ub) = (bias[j], bias[j]);
        for k in 0..in_dim {
            let w = weight[j * in_dim + k];
            if w >= 0.0 {
                lb += w * inp_l[k];
                ub += w * inp_u[k];
            } else {
                lb += w * inp_u[k];
                ub += w * inp_l[k];
            }
        }
        pre_l[j] = lb;
        pre_u[j] = ub;
    }
    (pre_l, pre_u)
}

/// ReLU relaxation (matches ny-propagate's `relu_linear_relaxation`), with the
/// intercept sanitized to the admitted domain.
fn relu_relaxation_admitted(l: f32, u: f32) -> (f32, f32, f32, f32) {
    let (ls, us, li, mut ui) = if l >= 0.0 {
        (1.0, 1.0, 0.0, 0.0)
    } else if u <= 0.0 {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        let us = u / (u - l);
        (if u > -l { 1.0 } else { 0.0 }, us, 0.0, -us * l)
    };
    if ui != 0.0 && ui.abs() < f32::MIN_POSITIVE {
        ui = 0.0;
    }
    let mut us_s = us;
    if us_s != 0.0 && us_s.abs() < f32::MIN_POSITIVE {
        us_s = 0.0;
    }
    (ls, us_s, li, ui)
}

struct Fixture {
    label: String,
    layers: Vec<GpuCrownLayer>,
    num_specs: usize,
    input_lower: Vec<f32>,
    input_upper: Vec<f32>,
}

/// Family A: Linear -> ReLU -> Linear with random normal-magnitude values.
fn linear_relu_fixture(rng: &mut Rng, index: usize) -> Fixture {
    let in_dim = rng.usize_in(2, 8);
    let hidden = rng.usize_in(2, 12);
    let out_dim = rng.usize_in(1, 5);

    let mut w1: Vec<f32> = (0..hidden * in_dim)
        .map(|_| rng.f32_in(-2.0, 2.0))
        .collect();
    let mut b1: Vec<f32> = (0..hidden).map(|_| rng.f32_in(-1.0, 1.0)).collect();
    let mut w2: Vec<f32> = (0..out_dim * hidden)
        .map(|_| rng.f32_in(-2.0, 2.0))
        .collect();
    let mut b2: Vec<f32> = (0..out_dim).map(|_| rng.f32_in(-1.0, 1.0)).collect();
    drop_subnormals(&mut w1);
    drop_subnormals(&mut b1);
    drop_subnormals(&mut w2);
    drop_subnormals(&mut b2);

    let mut input_lower = Vec::with_capacity(in_dim);
    let mut input_upper = Vec::with_capacity(in_dim);
    for _ in 0..in_dim {
        let center = rng.f32_in(-2.0, 2.0);
        let radius = rng.f32_in(0.01, 1.0);
        input_lower.push(center - radius);
        input_upper.push(center + radius);
    }
    drop_subnormals(&mut input_lower);
    drop_subnormals(&mut input_upper);

    let (pre_l, pre_u) = ibp_linear(&w1, &b1, &input_lower, &input_upper, hidden, in_dim);
    let mut ls = Vec::new();
    let mut us = Vec::new();
    let mut li = Vec::new();
    let mut ui = Vec::new();
    for j in 0..hidden {
        let (a, b, c, d) = relu_relaxation_admitted(pre_l[j], pre_u[j]);
        ls.push(a);
        us.push(b);
        li.push(c);
        ui.push(d);
    }

    Fixture {
        label: format!("A/linear-relu-{index} ({in_dim}->{hidden}->{out_dim})"),
        layers: vec![
            GpuCrownLayer::Linear {
                weight: w2.into(),
                bias: Some(b2.into()),
                out_features: out_dim,
                in_features: hidden,
                cert_err: Default::default(),
            },
            GpuCrownLayer::Activation {
                lower_slope: ls,
                upper_slope: us,
                lower_intercept: li,
                upper_intercept: ui,
                num_neurons: hidden,
            },
            GpuCrownLayer::Linear {
                weight: w1.into(),
                bias: Some(b1.into()),
                out_features: hidden,
                in_features: in_dim,
                cert_err: Default::default(),
            },
        ],
        num_specs: out_dim,
        input_lower,
        input_upper,
    }
}

/// Family B: Linear -> general activation with NONZERO NORMAL intercepts
/// (the §E re-admission under the widened `ActBiasParams.slack`) -> Linear.
fn general_activation_fixture(rng: &mut Rng, index: usize) -> Fixture {
    let in_dim = rng.usize_in(2, 6);
    let hidden = rng.usize_in(2, 10);
    let out_dim = rng.usize_in(1, 4);

    let mut w1: Vec<f32> = (0..hidden * in_dim)
        .map(|_| rng.f32_in(-1.0, 1.0))
        .collect();
    let mut b1: Vec<f32> = (0..hidden).map(|_| rng.f32_in(-0.5, 0.5)).collect();
    let mut w2: Vec<f32> = (0..out_dim * hidden)
        .map(|_| rng.f32_in(-1.0, 1.0))
        .collect();
    let mut b2: Vec<f32> = (0..out_dim).map(|_| rng.f32_in(-0.5, 0.5)).collect();
    drop_subnormals(&mut w1);
    drop_subnormals(&mut b1);
    drop_subnormals(&mut w2);
    drop_subnormals(&mut b2);

    // Ordered slopes in [0,1]; NONZERO normal intercepts with li <= ui.
    let mut ls = Vec::with_capacity(hidden);
    let mut us = Vec::with_capacity(hidden);
    let mut li = Vec::with_capacity(hidden);
    let mut ui = Vec::with_capacity(hidden);
    for _ in 0..hidden {
        let l = rng.f32_in(0.0, 1.0);
        let u = l + (1.0 - l) * rng.f32_in(0.0, 1.0);
        let a = rng.f32_in(-0.5, 0.5);
        let b = a + rng.f32_in(0.01, 0.5);
        ls.push(l);
        us.push(u);
        li.push(if a == 0.0 { 0.01 } else { a });
        ui.push(b);
    }
    drop_subnormals(&mut ls);
    drop_subnormals(&mut us);
    drop_subnormals(&mut li);
    drop_subnormals(&mut ui);

    let mut input_lower = Vec::with_capacity(in_dim);
    let mut input_upper = Vec::with_capacity(in_dim);
    for _ in 0..in_dim {
        let center = rng.f32_in(-1.0, 1.0);
        let radius = rng.f32_in(0.01, 0.5);
        input_lower.push(center - radius);
        input_upper.push(center + radius);
    }
    drop_subnormals(&mut input_lower);
    drop_subnormals(&mut input_upper);

    Fixture {
        label: format!("B/general-act-{index} ({in_dim}->{hidden}->{out_dim})"),
        layers: vec![
            GpuCrownLayer::Linear {
                weight: w2.into(),
                bias: Some(b2.into()),
                out_features: out_dim,
                in_features: hidden,
                cert_err: Default::default(),
            },
            GpuCrownLayer::Activation {
                lower_slope: ls,
                upper_slope: us,
                lower_intercept: li,
                upper_intercept: ui,
                num_neurons: hidden,
            },
            GpuCrownLayer::Linear {
                weight: w1.into(),
                bias: Some(b1.into()),
                out_features: hidden,
                in_features: in_dim,
                cert_err: Default::default(),
            },
        ],
        num_specs: out_dim,
        input_lower,
        input_upper,
    }
}

/// f32 IBP forward through a Conv2d (direct interval convolution) for the
/// ReLU relaxation of family C.
#[allow(clippy::too_many_arguments)]
fn ibp_conv2d(
    weight_col: &[f32],
    bias: &[f32],
    inp_l: &[f32],
    inp_u: &[f32],
    in_c: usize,
    in_h: usize,
    in_w: usize,
    out_c: usize,
    k: usize,
    stride: usize,
    pad: usize,
    out_h: usize,
    out_w: usize,
) -> (Vec<f32>, Vec<f32>) {
    let kernel_cols = in_c * k * k;
    let mut pre_l = vec![0.0f32; out_c * out_h * out_w];
    let mut pre_u = vec![0.0f32; out_c * out_h * out_w];
    for oc in 0..out_c {
        for oy in 0..out_h {
            for ox in 0..out_w {
                let (mut lb, mut ub) = (bias[oc], bias[oc]);
                for ic in 0..in_c {
                    for ky in 0..k {
                        for kx in 0..k {
                            let iy = oy * stride + ky;
                            let ix = ox * stride + kx;
                            if iy < pad || ix < pad {
                                continue;
                            }
                            let (iy, ix) = (iy - pad, ix - pad);
                            if iy >= in_h || ix >= in_w {
                                continue;
                            }
                            let w = weight_col[oc * kernel_cols + ic * k * k + ky * k + kx];
                            let x_l = inp_l[ic * in_h * in_w + iy * in_w + ix];
                            let x_u = inp_u[ic * in_h * in_w + iy * in_w + ix];
                            if w >= 0.0 {
                                lb += w * x_l;
                                ub += w * x_u;
                            } else {
                                lb += w * x_u;
                                ub += w * x_l;
                            }
                        }
                    }
                }
                let idx = oc * out_h * out_w + oy * out_w + ox;
                pre_l[idx] = lb;
                pre_u[idx] = ub;
            }
        }
    }
    (pre_l, pre_u)
}

/// Family C: Conv2d -> ReLU -> Linear.
fn conv_fixture(rng: &mut Rng, index: usize) -> Fixture {
    let in_c = rng.usize_in(1, 3);
    let out_c = rng.usize_in(1, 4);
    let k = rng.usize_in(1, 3);
    let stride = rng.usize_in(1, 2);
    let pad = rng.usize_in(0, 1);
    let in_h = rng.usize_in(k.max(3), 6);
    let in_w = rng.usize_in(k.max(3), 6);
    let out_h = (in_h + 2 * pad - k) / stride + 1;
    let out_w = (in_w + 2 * pad - k) / stride + 1;
    let conv_flat = out_c * out_h * out_w;
    let out_dim = rng.usize_in(1, 4);

    let kernel_cols = in_c * k * k;
    let mut weight_col: Vec<f32> = (0..out_c * kernel_cols)
        .map(|_| rng.f32_in(-1.0, 1.0))
        .collect();
    let mut conv_bias: Vec<f32> = (0..out_c).map(|_| rng.f32_in(-0.5, 0.5)).collect();
    drop_subnormals(&mut weight_col);
    drop_subnormals(&mut conv_bias);
    let mut bias_expanded = vec![0.0f32; conv_flat];
    for oc in 0..out_c {
        for pos in 0..out_h * out_w {
            bias_expanded[oc * out_h * out_w + pos] = conv_bias[oc];
        }
    }

    let flat_in = in_c * in_h * in_w;
    let mut input_lower = Vec::with_capacity(flat_in);
    let mut input_upper = Vec::with_capacity(flat_in);
    for _ in 0..flat_in {
        let center = rng.f32_in(-1.0, 1.0);
        let radius = rng.f32_in(0.01, 0.5);
        input_lower.push(center - radius);
        input_upper.push(center + radius);
    }
    drop_subnormals(&mut input_lower);
    drop_subnormals(&mut input_upper);

    let (pre_l, pre_u) = ibp_conv2d(
        &weight_col,
        &conv_bias,
        &input_lower,
        &input_upper,
        in_c,
        in_h,
        in_w,
        out_c,
        k,
        stride,
        pad,
        out_h,
        out_w,
    );
    let mut ls = Vec::new();
    let mut us = Vec::new();
    let mut li = Vec::new();
    let mut ui = Vec::new();
    for j in 0..conv_flat {
        let (a, b, c, d) = relu_relaxation_admitted(pre_l[j], pre_u[j]);
        ls.push(a);
        us.push(b);
        li.push(c);
        ui.push(d);
    }

    let mut w2: Vec<f32> = (0..out_dim * conv_flat)
        .map(|_| rng.f32_in(-1.0, 1.0))
        .collect();
    let mut b2: Vec<f32> = (0..out_dim).map(|_| rng.f32_in(-0.5, 0.5)).collect();
    drop_subnormals(&mut w2);
    drop_subnormals(&mut b2);

    Fixture {
        label: format!(
            "C/conv-{index} ({in_c}x{in_h}x{in_w} k{k}s{stride}p{pad} -> {out_c}x{out_h}x{out_w} -> {out_dim})"
        ),
        layers: vec![
            GpuCrownLayer::Linear {
                weight: w2.into(),
                bias: Some(b2.into()),
                out_features: out_dim,
                in_features: conv_flat,
                cert_err: Default::default(),
            },
            GpuCrownLayer::Activation {
                lower_slope: ls,
                upper_slope: us,
                lower_intercept: li,
                upper_intercept: ui,
                num_neurons: conv_flat,
            },
            GpuCrownLayer::Conv2d {
                weight_col: weight_col.into(),
                bias_expanded: Some(bias_expanded.into()),
                out_channels: out_c,
                in_channels: in_c,
                kernel_h: k,
                kernel_w: k,
                stride_h: stride,
                stride_w: stride,
                pad_h: pad,
                pad_w: pad,
                out_h,
                out_w,
                in_h,
                in_w,
                cert_err: Default::default(),
            },
        ],
        num_specs: out_dim,
        input_lower,
        input_upper,
    }
}

/// Family D: the structured ADMITTED-BOUNDARY set. Every value is at or just
/// above the smallest admitted magnitude: `2^-126` exactly
/// (`f32::MIN_POSITIVE`), tiny-normal multiples, mixed signs, exact zeros.
fn boundary_fixtures() -> Vec<Fixture> {
    const TINY: f32 = f32::MIN_POSITIVE; // 2^-126 exactly: the admitted boundary.
    let tiny15 = 1.5 * TINY; // tiny-normal
    let tiny25 = 2.5 * TINY;

    let boxes: Vec<(&str, Vec<f32>, Vec<f32>)> = vec![
        (
            "box=[2^-126,2^-126]^3 (degenerate at the boundary)",
            vec![TINY; 3],
            vec![TINY; 3],
        ),
        (
            "box=[-2^-126,2^-126]^3 (mixed-sign boundary)",
            vec![-TINY; 3],
            vec![TINY; 3],
        ),
        (
            "box=[0,2^-126]^3 (zero to boundary)",
            vec![0.0; 3],
            vec![TINY; 3],
        ),
        (
            "box=[-1.5*2^-126,2.5*2^-126]^3 (tiny-normal asymmetric)",
            vec![-tiny15; 3],
            vec![tiny25; 3],
        ),
        (
            "box=[-2^-126,1.0]x[?] (boundary against unit magnitude)",
            vec![-TINY, -1.0, TINY],
            vec![1.0, TINY, 1.0],
        ),
        (
            "box=[-1,1]^3 (normal box; boundary lives in weights/intercepts)",
            vec![-1.0; 3],
            vec![1.0; 3],
        ),
    ];

    // Weight matrices: normal magnitudes, tiny-normal magnitudes, mixed.
    let weight_sets: Vec<(&str, Vec<f32>, Vec<f32>)> = vec![
        (
            "weights normal",
            vec![
                0.5, -0.25, 1.0, -1.0, 0.75, 0.125, -0.5, 0.25, -0.125, 1.5, -0.75, 0.375,
            ],
            vec![0.5, -1.0, 0.25, -0.25, 1.0, -0.5, 0.75, -0.375],
        ),
        (
            "weights at +-2^-126 and mixed magnitude",
            vec![
                TINY, -TINY, 1.0, -1.0, tiny15, -tiny25, 0.5, TINY, -0.5, -TINY, tiny15, 1.0,
            ],
            vec![-TINY, TINY, 0.5, -1.0, tiny25, -tiny15, TINY, -0.5],
        ),
    ];

    // Activation parameter sets (4 hidden neurons): slopes in [0,1] including
    // exact 0/1 and the tiny-normal boundary; intercepts nonzero NORMAL
    // including exactly +-2^-126 (admitted; strictly-subnormal is refused).
    let act_sets: Vec<(&str, [f32; 4], [f32; 4], [f32; 4], [f32; 4])> = vec![
        (
            "intercepts at +-2^-126 exactly",
            [0.0, 1.0, 0.5, TINY],
            [1.0, 1.0, 0.75, 0.5],
            [-TINY, 0.0, TINY, -TINY],
            [TINY, 0.0, tiny25, tiny15],
        ),
        (
            "intercepts normal-magnitude, slopes at the boundary",
            [TINY, 0.0, 1.0, 0.25],
            [0.5, TINY, 1.0, 0.75],
            [-0.25, -0.125, 0.0625, -0.5],
            [0.25, 0.125, 0.5, 0.0625],
        ),
    ];

    let mut fixtures = Vec::new();
    for (box_label, inp_l, inp_u) in &boxes {
        for (w_label, w1, w2) in &weight_sets {
            for (a_label, ls, us, li, ui) in &act_sets {
                // 3 -> 4 -> 2 net.
                fixtures.push(Fixture {
                    label: format!("D/boundary [{box_label}; {w_label}; {a_label}]"),
                    layers: vec![
                        GpuCrownLayer::Linear {
                            weight: w2.clone().into(),
                            bias: Some(vec![TINY, -TINY].into()),
                            out_features: 2,
                            in_features: 4,
                            cert_err: Default::default(),
                        },
                        GpuCrownLayer::Activation {
                            lower_slope: ls.to_vec(),
                            upper_slope: us.to_vec(),
                            lower_intercept: li.to_vec(),
                            upper_intercept: ui.to_vec(),
                            num_neurons: 4,
                        },
                        GpuCrownLayer::Linear {
                            weight: w1.clone().into(),
                            bias: Some(vec![0.0, TINY, -0.5, 0.25].into()),
                            out_features: 4,
                            in_features: 3,
                            cert_err: Default::default(),
                        },
                    ],
                    num_specs: 2,
                    input_lower: inp_l.clone(),
                    input_upper: inp_u.clone(),
                });
            }
        }
    }
    fixtures
}

// ---------------------------------------------------------------------------
// The containment check.
// ---------------------------------------------------------------------------

struct ParityStats {
    fixtures: usize,
    spec_rows: usize,
    breaches: Vec<String>,
    degenerate_rows: usize,
    max_lower_margin: f64,
    max_upper_margin: f64,
}

impl ParityStats {
    fn new() -> Self {
        Self {
            fixtures: 0,
            spec_rows: 0,
            breaches: Vec::new(),
            degenerate_rows: 0,
            max_lower_margin: 0.0,
            max_upper_margin: 0.0,
        }
    }

    fn finish(self, family: &str, expect_no_degenerate: bool) {
        println!(
            "[flush-charge acceptance/{family}] fixtures={} spec_rows={} \
             breaches={} degenerate_rows={} max_charge_margin(lower={:.3e}, upper={:.3e})",
            self.fixtures,
            self.spec_rows,
            self.breaches.len(),
            self.degenerate_rows,
            self.max_lower_margin,
            self.max_upper_margin,
        );
        assert!(
            self.breaches.is_empty(),
            "ENCLOSURE-PARITY BREACH ({family}): the charged GPU enclosure \
             failed to contain the CPU f64 reference on {} row(s) (verbatim):\n{}",
            self.breaches.len(),
            self.breaches.join("\n")
        );
        if expect_no_degenerate {
            assert_eq!(
                self.degenerate_rows, 0,
                "{family}: well-conditioned fixtures must not publish \
                 FALLBACK-degenerate rows"
            );
        }
    }
}

/// Run one fixture through the charged GPU walk and the f64 reference and
/// record containment. A walk refusal is a hard failure (these fixtures are
/// built inside the admitted domain).
fn check_fixture(device: &WgpuDevice, fixture: &Fixture, stats: &mut ParityStats) {
    let spec = identity_spec(fixture.num_specs);
    let gpu = device
        .crown_backward_gpu_sound(
            &fixture.layers,
            &spec,
            fixture.num_specs,
            &fixture.input_lower,
            &fixture.input_upper,
        )
        .unwrap_or_else(|error| {
            panic!(
                "charged GPU walk REFUSED an admitted fixture ({}): {error}",
                fixture.label
            )
        });
    let (cpu_l, cpu_u) = f64_crown_backward(
        &fixture.layers,
        &spec,
        fixture.num_specs,
        &fixture.input_lower,
        &fixture.input_upper,
    );

    stats.fixtures += 1;
    for i in 0..fixture.num_specs {
        stats.spec_rows += 1;
        let gl = f64::from(gpu.lower_bounds[i]);
        let gu = f64::from(gpu.upper_bounds[i]);
        if gpu.lower_bounds[i].abs() >= 1e10 || gpu.upper_bounds[i].abs() >= 1e10 {
            stats.degenerate_rows += 1;
        }
        // CONTAINMENT, no tolerance: [gl, gu] must contain [cpu_l, cpu_u].
        if gl > cpu_l[i] {
            stats.breaches.push(format!(
                "  {} row {i}: LOWER breach gpu={:?} (={gl:e}) > f64_ref={:e} \
                 (excess {:e})",
                fixture.label,
                gpu.lower_bounds[i],
                cpu_l[i],
                gl - cpu_l[i],
            ));
        }
        if gu < cpu_u[i] {
            stats.breaches.push(format!(
                "  {} row {i}: UPPER breach gpu={:?} (={gu:e}) < f64_ref={:e} \
                 (deficit {:e})",
                fixture.label,
                gpu.upper_bounds[i],
                cpu_u[i],
                cpu_u[i] - gu,
            ));
        }
        stats.max_lower_margin = stats.max_lower_margin.max(cpu_l[i] - gl);
        stats.max_upper_margin = stats.max_upper_margin.max(gu - cpu_u[i]);
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

/// Run the acceptance body only on an armed device; under the verified
/// explicit env-pin refusal, the refusal itself has already been asserted and
/// loudly reported by [`charged_acceptance`].
fn with_armed_device(test_name: &str, body: impl FnOnce(&WgpuDevice)) {
    match charged_acceptance() {
        ChargedAcceptance::Armed(device) => body(&device),
        ChargedAcceptance::EnvRequiredPinnedRefusal(reason) => println!(
            "[flush-charge acceptance] {test_name}: acceptance body NOT RUN \
             (verified explicit NY_GPU_DENORM_PRESERVE=1 refusal: {reason})"
        ),
    }
}

/// The TEST-SCOPED device arms with the production policy, is NOT fully
/// qualified, and the PRODUCTION charged constructor (gate OPEN since the
/// 2026-08-13 review) runs its own genuine admission on this same adapter —
/// admitting with charged (never full) authority on a pure-flush box, or
/// refusing with a typed measured reason.
#[test]
fn charged_acceptance_device_arms_with_production_policy_and_grants_no_full_authority() {
    let _serial = gpu_test_serial_guard();

    // The PRODUCTION constructor measures its own forced plain-WGSL device.
    match WgpuDevice::new_for_verdict_flush_charged(super::WgpuChargedVerdictRequest::new()) {
        Ok(device) => {
            assert!(
                device.charged_flush_authority_cached().is_some(),
                "an admitted production charged device must carry the policy"
            );
            assert!(
                !device.sound_gpu_authority_cached(),
                "charged admission must never masquerade as full qualification"
            );
        }
        Err(error) => {
            let message = error.source_error().to_string();
            assert!(
                message.contains("NY_GPU_DENORM_PRESERVE")
                    || message.contains("not PURE-FLUSH")
                    || message.contains("five-rung ladder")
                    || message.contains("HAZARDOUS"),
                "an open-gate refusal must be the typed env pin or a measured \
                 ladder refusal, got: {message}"
            );
        }
    }

    with_armed_device("authority pins", |device| {
        let policy = device
            .charged_flush_authority_cached()
            .copied()
            .expect("acceptance device must carry the armed charge policy");
        assert_eq!(
            policy,
            super::ops::sound_authority::FlushChargePolicy::production()
        );
        assert!(
            !device.sound_gpu_authority_cached(),
            "charged acceptance authority must never masquerade as full qualification"
        );
        assert!(matches!(
            device.verdict_authority(),
            super::WgpuVerdictAuthority::QualifiedWithFlushCharge(_)
        ));
        let report = device
            .verdict_report()
            .expect("stored (non-qualified) report");
        assert!(!report.qualified());
        println!(
            "[flush-charge acceptance] armed PRODUCTION charged device: adapter={:?} report_reason={:?}",
            report.adapter(),
            report.reason()
        );
    });
}

/// Family A: >= 80 seeded random Linear/ReLU/Linear fixtures.
#[test]
fn enclosure_parity_linear_relu_chains() {
    let _serial = gpu_test_serial_guard();
    with_armed_device("family A", |device| {
        let mut rng = Rng::new(0x5EED_1A0E_B001_0001);
        let mut stats = ParityStats::new();
        for index in 0..80 {
            let fixture = linear_relu_fixture(&mut rng, index);
            check_fixture(device, &fixture, &mut stats);
        }
        stats.finish("A linear-relu x80", true);
    });
}

/// Family B: >= 70 seeded random general-activation fixtures with NONZERO
/// NORMAL intercepts (the §E re-admission paying the widened act-bias slack).
#[test]
fn enclosure_parity_general_activation_nonzero_normal_intercepts() {
    let _serial = gpu_test_serial_guard();
    with_armed_device("family B", |device| {
        let mut rng = Rng::new(0x5EED_1A0E_B001_0002);
        let mut stats = ParityStats::new();
        for index in 0..70 {
            let fixture = general_activation_fixture(&mut rng, index);
            check_fixture(device, &fixture, &mut stats);
        }
        stats.finish("B general-act x70", true);
    });
}

/// Family C: >= 60 seeded random Conv2d/ReLU/Linear fixtures.
#[test]
fn enclosure_parity_conv2d_chains() {
    let _serial = gpu_test_serial_guard();
    with_armed_device("family C", |device| {
        let mut rng = Rng::new(0x5EED_1A0E_B001_0003);
        let mut stats = ParityStats::new();
        for index in 0..60 {
            let fixture = conv_fixture(&mut rng, index);
            check_fixture(device, &fixture, &mut stats);
        }
        stats.finish("C conv2d x60", true);
    });
}

/// Family D: the structured admitted-boundary set (exact 2^-126 endpoints,
/// tiny-normal weights/intercepts, mixed signs). Degenerate FALLBACK rows are
/// tolerated here (a valid, useless bound is not a soundness breach) but
/// containment is still mandatory.
#[test]
fn enclosure_parity_admitted_boundary_set() {
    let _serial = gpu_test_serial_guard();
    with_armed_device("family D", |device| {
        let fixtures = boundary_fixtures();
        assert!(fixtures.len() >= 20, "boundary set must stay substantial");
        let mut stats = ParityStats::new();
        for fixture in &fixtures {
            check_fixture(device, fixture, &mut stats);
        }
        stats.finish("D admitted-boundary", false);
    });
}

/// The charged walk guard is LIVE on the acceptance device: a
/// strictly-subnormal input-box endpoint (refused, unchargeable) must produce
/// a typed refusal, not a bound.
#[test]
fn charged_guard_refuses_strictly_subnormal_input_endpoint_live() {
    let _serial = gpu_test_serial_guard();
    with_armed_device("subnormal-input refusal", |device| {
        let mut rng = Rng::new(0x5EED_1A0E_B001_0004);
        let mut fixture = linear_relu_fixture(&mut rng, 0);
        fixture.input_upper[0] = 1.0e-40; // strictly subnormal: refused
        fixture.input_lower[0] = -1.0;
        let spec = identity_spec(fixture.num_specs);
        let error = device
            .crown_backward_gpu_sound(
                &fixture.layers,
                &spec,
                fixture.num_specs,
                &fixture.input_lower,
                &fixture.input_upper,
            )
            .expect_err("subnormal input endpoint must be refused under charged authority");
        let message = error.to_string();
        assert!(
            message.contains("SUBNORMAL"),
            "refusal must name the subnormal channel, got: {message}"
        );
    });
}

/// The charged walk guard refuses a strictly-subnormal activation intercept
/// live (the §E permanent refusal), while exactly 2^-126 stays admitted —
/// pinning the boundary from both sides on the real device.
#[test]
fn charged_guard_boundary_is_exactly_min_positive_for_intercepts_live() {
    let _serial = gpu_test_serial_guard();
    with_armed_device("intercept boundary", |device| {
        let mut rng = Rng::new(0x5EED_1A0E_B001_0005);
        let fixture = general_activation_fixture(&mut rng, 0);
        let spec = identity_spec(fixture.num_specs);

        // Admitted side: run unchanged (nonzero normal intercepts).
        device
            .crown_backward_gpu_sound(
                &fixture.layers,
                &spec,
                fixture.num_specs,
                &fixture.input_lower,
                &fixture.input_upper,
            )
            .expect("nonzero NORMAL intercepts are admitted under the §E charge");

        // Refused side: poison one intercept to strictly-subnormal.
        let mut layers = fixture.layers.clone();
        if let GpuCrownLayer::Activation {
            upper_intercept, ..
        } = &mut layers[1]
        {
            upper_intercept[0] = f32::MIN_POSITIVE / 2.0; // strictly subnormal
        } else {
            panic!("fixture layer 1 must be the activation");
        }
        let error = device
            .crown_backward_gpu_sound(
                &layers,
                &spec,
                fixture.num_specs,
                &fixture.input_lower,
                &fixture.input_upper,
            )
            .expect_err("strictly-subnormal intercept must be refused (permanent §E refusal)");
        assert!(
            error.to_string().contains("SUBNORMAL"),
            "refusal must name the subnormal channel, got: {error}"
        );
    });
}
