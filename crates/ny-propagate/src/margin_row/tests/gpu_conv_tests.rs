// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #mr-gpu-eval soundness, precision and dark-default tests.
//!
//! The device is emulated by [`ReorderFmaEngine`], an f64 `GemmEngine` that
//! sums every dot product in a DIFFERENT order than the CPU kernel — reversed
//! contraction, four interleaved FMA accumulators, a tree combine: the shape
//! of a blocked / split-K cuBLAS `Dgemm`. The certificate under test is
//! order-independent, so passing against one adversarial order is evidence
//! for the argument, and the CPU parity checks pin that the route keeps f64
//! precision (an f32 route's radius would fail them by ~6 orders).
//!
//! * `gpu_conv_route_is_dark_by_default` — declaration, gate, route, engine
//!   default and a dark `offload` that touches nothing.
//! * `gpu_conv_tap_table_matches_back_taps_and_refuses_transposed`.
//! * `gpu_conv_gemm_matches_cpu_kernel_within_gamma` — random and real
//!   cifar100-resnet conv shapes, signed and `|W|`.
//! * `gpu_conv_certified_interval_contains_exact_rational_product` — THE
//!   operator-level containment test: `[lnew - en, lnew + en]` (the engine's
//!   own charges, device-computed) contains the EXACT-RATIONAL product for the
//!   stored operand and for exact operands anywhere in its error box.
//! * `gpu_conv_routed_root_eval_never_exceeds_exact_margin` — the published
//!   root bound under the routed engine (per-column AND envelope lanes) never
//!   exceeds the exact margin at sampled points, point boxes included.
//! * `gpu_conv_beta_passes_stay_below_exact_lagrangian` — beta: uniform
//!   passes with shared AND per-column beta, and a domain-stacked pass, are
//!   checked against the EXACT Lagrangian `y - sum beta*s*z` (lower) /
//!   `y + sum beta*s*z` (upper), valid over the whole box for any sign.
//! * `gpu_conv_refusals_fall_back_bit_identically` — every engine failure
//!   mode publishes the dark pass bit-for-bit.
//! * `gpu_conv_stop_line_has_fixed_schema`.
//!
//! `#bi-gpu` (the backward-interm / joint-refresh prefix pairs on a scoped
//! route of this seam) is pinned in the child module `bi_gpu_tests`, which
//! reuses the emulated devices and oracles below.

use super::*;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use num_rational::BigRational;
use ny_core::{GemmEngine, NyError};

use crate::margin_row::engine::{BetaSplit, Collect, PassOut, PcBetaCol, Seed};
use crate::margin_row::errlane::ErrLane;
use crate::margin_row::gpu_conv::{self, ConvLane, GpuConvRoute, Refusal};
use crate::margin_row::net::{conv_apply_backward, ConvOp, TwinOp};
use crate::margin_row::rounding::{certify_up, gamma_n, next_up};

// ---------------------------------------------------------------------------
//  Emulated devices
// ---------------------------------------------------------------------------

/// f64 engine with a deliberately different summation order (see module docs).
#[derive(Default)]
struct ReorderFmaEngine {
    calls: AtomicUsize,
}

impl ReorderFmaEngine {
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl GemmEngine for ReorderFmaEngine {
    fn backend_provenance(&self) -> &'static str {
        "test-reorder-fma-f64"
    }

    fn gemm_f32(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        Err(NyError::UnsupportedOp("f64-only test engine".into()))
    }

    fn gemm_f64_with_deadline(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f64],
        b: &[f64],
        deadline: Instant,
        _max_dispatch_macs: usize,
    ) -> ny_core::Result<Vec<f64>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(
                "test engine: budget expired".into(),
            ));
        }
        assert_eq!(a.len(), m * k, "lhs shape");
        assert_eq!(b.len(), k * n, "rhs shape");
        let mut c = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = [0.0f64; 4];
                for (step, l) in (0..k).rev().enumerate() {
                    acc[step % 4] = a[i * k + l].mul_add(b[l * n + j], acc[step % 4]);
                }
                c[i * n + j] = (acc[0] + acc[2]) + (acc[1] + acc[3]);
            }
        }
        Ok(c)
    }
}

/// How [`FailingEngine`] fails.
#[derive(Clone, Copy, Debug)]
enum Failure {
    Error,
    Deadline,
    NonFinite,
    WrongLength,
}

/// An engine that fails every call in one fixed way.
struct FailingEngine {
    mode: Failure,
    calls: AtomicUsize,
}

impl GemmEngine for FailingEngine {
    fn backend_provenance(&self) -> &'static str {
        "test-failing-f64"
    }

    fn gemm_f32(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        Err(NyError::UnsupportedOp("f64-only test engine".into()))
    }

    fn gemm_f64_with_deadline(
        &self,
        m: usize,
        _k: usize,
        n: usize,
        _a: &[f64],
        _b: &[f64],
        _deadline: Instant,
        _max_dispatch_macs: usize,
    ) -> ny_core::Result<Vec<f64>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Failure::Error => Err(NyError::InternalError("test engine failure".into())),
            Failure::Deadline => Err(NyError::DeadlineExceeded("test engine".into())),
            Failure::NonFinite => {
                let mut c = vec![0.0f64; m * n];
                if let Some(last) = c.last_mut() {
                    *last = f64::NAN;
                }
                Ok(c)
            }
            Failure::WrongLength => Ok(vec![0.0f64; m * n + 1]),
        }
    }
}

// ---------------------------------------------------------------------------
//  Fixtures and oracles
// ---------------------------------------------------------------------------

fn uniform(rng: &mut StdRng, n: usize, s: f64) -> Vec<f64> {
    (0..n).map(|_| rng.random_range(-s..s)).collect()
}

#[allow(clippy::too_many_arguments)]
fn conv_op(
    input: usize,
    weight: Vec<f64>,
    bias: Vec<f64>,
    kernel: (usize, usize, usize, usize),
    stride: (usize, usize),
    pad: usize,
    ishape: (usize, usize, usize),
    oshape: (usize, usize, usize),
) -> TwinOpSpec {
    TwinOpSpec::Conv {
        input,
        weight,
        bias,
        bias_err: vec![0.0; kernel.0],
        weight_rel_err: 0.0,
        kernel,
        stride,
        pads: (pad, pad, pad, pad),
        ishape,
        oshape,
    }
}

/// A resnet-shaped twin net over a `(ci, h, w)` input: 3x3 stem conv, a
/// residual block, a stride-2 3x3 downsample, a 1x1 conv and the
/// `Gemm -> Relu -> Gemm` head. `c1`/`c2` are the stage widths.
fn resnetish_spec(
    rng: &mut StdRng,
    (ci, h, w): (usize, usize, usize),
    c1: usize,
    c2: usize,
) -> TwinSpec {
    let h2 = (h - 1) / 2 + 1;
    let w2 = (w - 1) / 2 + 1;
    let s_stem = 1.5 / ((ci * 9) as f64).sqrt();
    let s_blk = 1.5 / ((c1 * 9) as f64).sqrt();
    let s_one = 1.5 / (c2 as f64).sqrt();
    let n_flat = c2 * h2 * w2;
    let n_y = 6;
    let n_out = 4;
    let ops = vec![
        conv_op(
            0,
            uniform(rng, c1 * ci * 9, s_stem),
            uniform(rng, c1, 0.2),
            (c1, ci, 3, 3),
            (1, 1),
            1,
            (ci, h, w),
            (c1, h, w),
        ), // t1
        TwinOpSpec::Relu { input: 1 }, // t2 (trunk relu 0)
        conv_op(
            2,
            uniform(rng, c1 * c1 * 9, s_blk),
            uniform(rng, c1, 0.2),
            (c1, c1, 3, 3),
            (1, 1),
            1,
            (c1, h, w),
            (c1, h, w),
        ), // t3
        TwinOpSpec::Add { lhs: 3, rhs: 2 }, // t4 (residual)
        TwinOpSpec::Relu { input: 4 }, // t5 (trunk relu 1)
        conv_op(
            5,
            uniform(rng, c2 * c1 * 9, s_blk),
            uniform(rng, c2, 0.2),
            (c2, c1, 3, 3),
            (2, 2),
            1,
            (c1, h, w),
            (c2, h2, w2),
        ), // t6 (stride-2 downsample)
        TwinOpSpec::Relu { input: 6 }, // t7 (trunk relu 2)
        conv_op(
            7,
            uniform(rng, c2 * c2, s_one),
            uniform(rng, c2, 0.2),
            (c2, c2, 1, 1),
            (1, 1),
            0,
            (c2, h2, w2),
            (c2, h2, w2),
        ), // t8 (1x1)
        TwinOpSpec::Relu { input: 8 }, // t9 (trunk relu 3)
        TwinOpSpec::Flatten { input: 9 }, // t10
        TwinOpSpec::Gemm {
            input: 10,
            weight: uniform(rng, n_y * n_flat, 2.0 / (n_flat as f64).sqrt()),
            bias: uniform(rng, n_y, 0.2),
            bias_err: Vec::new(),
            weight_rel_err: 0.0,
            shape: (n_y, n_flat),
        }, // t11 (y)
        TwinOpSpec::Relu { input: 11 }, // t12 (head relu)
        TwinOpSpec::Gemm {
            input: 12,
            weight: uniform(rng, n_out * n_y, 1.0),
            bias: uniform(rng, n_out, 0.2),
            bias_err: Vec::new(),
            weight_rel_err: 0.0,
            shape: (n_out, n_y),
        }, // t13
    ];
    TwinSpec {
        n_in: ci * h * w,
        ops,
    }
}

/// Every conv op of a compiled net.
fn convs(net: &TwinNet) -> Vec<&ConvOp> {
    net.ops
        .iter()
        .filter_map(|op| match op {
            TwinOp::Conv(cv) => Some(&**cv),
            _ => None,
        })
        .collect()
}

/// A `(rows, r)` matrix of mixed-sign, mixed-EXPONENT (2^-30 .. 2^30) values:
/// the dynamic range that makes two summation orders round differently.
fn wild_matrix(rng: &mut StdRng, rows: usize, r: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, r), |_| {
        let e: i32 = rng.random_range(-30..31);
        let m: f64 = rng.random_range(-1.0..1.0);
        m * 2f64.powi(e)
    })
}

/// A dense deterministic seed without an error lane.
fn plain_seed(n_y: usize, r: usize, salt: usize) -> Seed {
    let s = Array2::from_shape_fn((n_y, r), |(j, c)| {
        ((1 + salt * 17 + j * 11 + c * 5) as f64 * 0.311).sin()
    });
    Seed { s, e: None }
}

fn q(v: f64) -> BigRational {
    BigRational::from_float(v).expect("finite f64 is an exact dyadic")
}

/// EXACT dyadic-rational forward over a raw TwinSpec (Conv / Relu / Add /
/// Flatten / Gemm), returning EVERY tensor (index = tensor id).
fn exact_tensors(spec: &TwinSpec, x: &[f64]) -> Vec<Vec<BigRational>> {
    let zero = q(0.0);
    let mut t: Vec<Vec<BigRational>> = Vec::with_capacity(spec.ops.len() + 1);
    t.push(x.iter().map(|&v| q(v)).collect());
    for op in &spec.ops {
        let out: Vec<BigRational> = match op {
            TwinOpSpec::Conv {
                input,
                weight,
                bias,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
                ..
            } => {
                let src = &t[*input];
                let (co, ci, kh, kw) = *kernel;
                let (_, ih, iw) = *ishape;
                let (_, oh, ow) = *oshape;
                let mut out = Vec::with_capacity(co * oh * ow);
                for oc in 0..co {
                    for oy in 0..oh {
                        for ox in 0..ow {
                            let mut acc = q(bias[oc]);
                            for c in 0..ci {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let iy = (oy * stride.0 + ky) as isize - pads.0 as isize;
                                        let ix = (ox * stride.1 + kx) as isize - pads.1 as isize;
                                        if iy >= 0
                                            && (iy as usize) < ih
                                            && ix >= 0
                                            && (ix as usize) < iw
                                        {
                                            let wv = weight[((oc * ci + c) * kh + ky) * kw + kx];
                                            if wv != 0.0 {
                                                let s = &src[c * ih * iw
                                                    + (iy as usize) * iw
                                                    + ix as usize];
                                                acc += q(wv) * s;
                                            }
                                        }
                                    }
                                }
                            }
                            out.push(acc);
                        }
                    }
                }
                out
            }
            TwinOpSpec::Relu { input } => t[*input]
                .iter()
                .map(|v| if *v > zero { v.clone() } else { zero.clone() })
                .collect(),
            TwinOpSpec::Add { lhs, rhs } => {
                t[*lhs].iter().zip(&t[*rhs]).map(|(a, b)| a + b).collect()
            }
            TwinOpSpec::Flatten { input } => t[*input].clone(),
            // `y = scale[ch] * x + shift[ch]` on the STORED f64 parameters (the
            // engine's certified fold errors only widen around them).
            TwinOpSpec::ChannelAffine {
                input,
                scale,
                shift,
                shape,
                ..
            } => {
                let hw = shape.1 * shape.2;
                t[*input]
                    .iter()
                    .enumerate()
                    .map(|(i, v)| q(scale[i / hw]) * v + q(shift[i / hw]))
                    .collect()
            }
            TwinOpSpec::Gemm {
                input,
                weight,
                bias,
                shape,
                ..
            } => {
                let src = &t[*input];
                let (no, ni) = *shape;
                (0..no)
                    .map(|o| {
                        let mut acc = q(bias[o]);
                        for (i, s) in src.iter().enumerate().take(ni) {
                            let wv = weight[o * ni + i];
                            if wv != 0.0 {
                                acc += q(wv) * s;
                            }
                        }
                        acc
                    })
                    .collect()
            }
            other => panic!("exact_tensors: fixture op {other:?} is outside this oracle"),
        };
        t.push(out);
    }
    t
}

/// EXACT `out[ic*ip + isp, col]` of the conv backward of `src` (optionally with
/// `|W|`), summed over exactly the listed back taps.
fn exact_backward_entry(
    cv: &ConvOp,
    src: &[BigRational],
    r: usize,
    row: usize,
    col: usize,
    abs_w: bool,
) -> BigRational {
    let (co, _ci, kh, kw) = cv.kernel;
    let ip = cv.ishape.1 * cv.ishape.2;
    let op_ = cv.oshape.1 * cv.oshape.2;
    let (ic, isp) = (row / ip, row % ip);
    let mut acc = q(0.0);
    for &(ky, kx, osp) in &cv.back_taps[isp] {
        for oc in 0..co {
            let w0 = cv.wt[((ic * kh + ky) * kw + kx) * co + oc];
            let wv = if abs_w { w0.abs() } else { w0 };
            acc += q(wv) * &src[(oc * op_ + osp) * r + col];
        }
    }
    acc
}

/// Sample points of a box: center, both corners, random corners and interior.
fn box_points(rng: &mut StdRng, lo: &[f64], hi: &[f64], extra: usize) -> Vec<Vec<f64>> {
    let n = lo.len();
    let mut pts = vec![
        (0..n).map(|i| f64::midpoint(lo[i], hi[i])).collect(),
        lo.to_vec(),
        hi.to_vec(),
    ];
    for _ in 0..extra {
        pts.push(
            (0..n)
                .map(|i| {
                    if rng.random_range(0.0..1.0) < 0.5 {
                        lo[i]
                    } else {
                        hi[i]
                    }
                })
                .collect(),
        );
        pts.push(
            (0..n)
                .map(|i| lo[i] + rng.random_range(0.0..1.0) * (hi[i] - lo[i]))
                .collect(),
        );
    }
    pts
}

fn far_deadline() -> Instant {
    Instant::now() + Duration::from_secs(600)
}

/// The two small fixtures plus a resnet-shaped net, all inside the oracle's
/// op set. `(name, spec, box half-width)`.
fn net_fixtures() -> Vec<(&'static str, TwinSpec, f64)> {
    let mut rng = StdRng::seed_from_u64(0x6C0F_0001);
    let tiny_a = tiny_spec(&mut rng, 0.6);
    let tiny_b = tiny_spec(&mut rng, 0.9);
    let resnetish = resnetish_spec(&mut rng, (3, 10, 10), 6, 8);
    vec![
        ("tiny_0.6", tiny_a, 0.3),
        ("tiny_0.9", tiny_b, 0.25),
        ("resnetish_3x10x10", resnetish, 0.2),
    ]
}

/// Real cifar100-resnet conv shapes: 16ch 32x32 3x3 stem and block, a 16->32
/// stride-2 downsample, a 32ch 16x16 1x1 conv.
fn real_shape_convs() -> TwinNet {
    let mut rng = StdRng::seed_from_u64(0x6C0F_0002);
    let spec = resnetish_spec(&mut rng, (16, 32, 32), 16, 32);
    TwinNet::compile(&spec).expect("real-shape net compiles")
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

/// RETIRED (ENGINE_PROGRAM R1 kill line + R2): a request through the only
/// route left (the environment; the typed key is refused at preset
/// validation) still RESOLVES in the declaration — resolved here from a
/// simulated environment, never the process one, so no other test's
/// once-per-process request latch can see it — but the lane route NEVER arms,
/// and the lane-start line is the fixed-schema refusal instead of `armed`.
#[test]
fn gpu_conv_is_retired_and_a_request_is_refused_not_armed() {
    let decl = &ny_levers::decls::margin_row::MARGIN_ROW_GPU_CONV;
    let requested = ny_levers::read_over_config_with(
        decl,
        |name| (name == decl.name).then(|| "1".to_string()),
        None,
    )
    .expect("the retired switch still parses");
    assert!(
        requested.value.as_bool(),
        "VOID: the simulated request did not resolve"
    );
    assert!(!gpu_conv::armed(), "the retired route is never armed");
    assert!(gpu_conv::lane_route().is_off());
    assert_eq!(gpu_conv::announce_line(false), "[mr-gpu-conv] dark");
    let refused = gpu_conv::announce_line(true);
    assert!(
        refused.starts_with("[mr-gpu-conv] refused reason=retired(r1_kill_c2050_c1590"),
        "{refused}"
    );
    assert!(refused.ends_with("route=cpu"), "{refused}");
}

/// DARK BY DEFAULT: the declaration defaults false, the governed gate resolves
/// false with no preset and no env, the lane route is `Off`, a fresh engine
/// carries `Off`, and a dark `offload` refuses without touching `dst`.
#[test]
fn gpu_conv_route_is_dark_by_default() {
    assert!(matches!(
        ny_levers::decls::margin_row::MARGIN_ROW_GPU_CONV.default,
        ny_levers::DefaultSpec::Bool(false)
    ));
    assert!(
        !crate::preset_gates::margin_row_gpu_conv_enabled(),
        "the gate must resolve dark with no preset cell and no environment"
    );
    assert!(!gpu_conv::armed(), "the process latch must be dark");
    assert!(GpuConvRoute::default().is_off());
    assert!(gpu_conv::lane_route().is_off());
    assert!(GpuConvRoute::from_armed(false).is_off());
    assert!(matches!(
        GpuConvRoute::from_armed(true),
        GpuConvRoute::Global
    ));
    assert!(gpu_conv::backend_admitted("cuda-cublas-rn"));
    assert!(!gpu_conv::backend_admitted("faer-cpu"));
    assert!(!gpu_conv::backend_admitted("accelerate-veclib-cpu"));

    let (_, spec, rad) = net_fixtures().remove(0);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-rad; spec.n_in];
    let hi = vec![rad; spec.n_in];
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let eng = BackwardEngine::new(&net, &root);
    assert!(
        eng.gpu_conv_route().is_off(),
        "a fresh engine must carry the dark route"
    );

    let cv = convs(&net)[0];
    let r = 3;
    let src = Array2::from_shape_fn((cv.kernel.0 * cv.oshape.1 * cv.oshape.2, r), |(i, c)| {
        (i * 3 + c) as f64 * 0.125 - 1.0
    });
    let rows = cv.kernel.1 * cv.ishape.1 * cv.ishape.2;
    let sentinel = -1234.5f64;
    let mut dst = Array2::from_elem((rows, r), sentinel);
    let offloaded = gpu_conv::offload(
        &GpuConvRoute::Off,
        cv,
        &src,
        &mut dst,
        false,
        ConvLane::Signed,
        true,
    );
    assert!(!offloaded, "the dark route must never offload");
    assert!(
        dst.iter().all(|v| v.to_bits() == sentinel.to_bits()),
        "a dark offload must not touch dst"
    );
}

/// The dense back-tap table reproduces `back_taps` exactly (one slot per
/// listed tap, no duplicates), and a transposed conv is refused.
#[test]
fn gpu_conv_tap_table_matches_back_taps_and_refuses_transposed() {
    let mut nets: Vec<TwinNet> = net_fixtures()
        .into_iter()
        .map(|(_, spec, _)| TwinNet::compile(&spec).expect("compile"))
        .collect();
    nets.push(real_shape_convs());
    let mut checked = 0usize;
    for net in &nets {
        for cv in convs(net) {
            let (_, _, kh, kw) = cv.kernel;
            let taps = kh * kw;
            let tab = gpu_conv::tap_table(cv).expect("plain conv tap table");
            for (isp, list) in cv.back_taps.iter().enumerate() {
                let slots = &tab[isp * taps..(isp + 1) * taps];
                let live = slots.iter().filter(|&&o| o != usize::MAX).count();
                assert_eq!(live, list.len(), "isp {isp}: tap count");
                for &(ky, kx, osp) in list {
                    assert_eq!(slots[ky * kw + kx], osp, "isp {isp}: tap ({ky},{kx})");
                }
            }
            checked += 1;
        }
    }
    assert!(checked >= 10, "only {checked} convs checked");

    // A ConvTranspose (cgan_like_spec) is refused before any engine call.
    let mut rng = StdRng::seed_from_u64(0x6C0F_0003);
    let net = TwinNet::compile(&cgan_like_spec(&mut rng, 0.6)).expect("compile");
    let transposed: Vec<&ConvOp> = convs(&net).into_iter().filter(|cv| cv.transposed).collect();
    assert!(!transposed.is_empty(), "fixture lost its ConvTranspose");
    let engine = ReorderFmaEngine::default();
    for cv in transposed {
        let r = 2;
        let src = Array2::<f64>::zeros((cv.kernel.0 * cv.oshape.1 * cv.oshape.2, r));
        let mut dst = Array2::<f64>::zeros((cv.kernel.1 * cv.ishape.1 * cv.ishape.2, r));
        assert_eq!(
            gpu_conv::conv_backward_on_engine(&engine, cv, &src, &mut dst, false, far_deadline()),
            Err(Refusal::Transposed)
        );
    }
    assert_eq!(
        engine.calls(),
        0,
        "a refused conv must never reach the engine"
    );
}

/// PRECISION PARITY: the reordered-FMA device product agrees with the CPU
/// kernel within `2 * gamma(k_bwd + 2) * (|W|^T |L|)` entrywise (both are
/// within `gamma_n` of the same exact sum) on random and real conv shapes,
/// signed and `|W|`, and genuinely differs bitwise somewhere (non-vacuous).
#[test]
fn gpu_conv_gemm_matches_cpu_kernel_within_gamma() {
    let mut rng = StdRng::seed_from_u64(0x6C0F_0004);
    let mut nets: Vec<(TwinNet, Vec<usize>)> = net_fixtures()
        .into_iter()
        .map(|(_, spec, _)| (TwinNet::compile(&spec).expect("compile"), vec![1, 3, 17]))
        .collect();
    nets.push((real_shape_convs(), vec![1, 2]));
    let engine = ReorderFmaEngine::default();
    let mut entries = 0usize;
    let mut differ = 0usize;
    for (net, widths) in &nets {
        for cv in convs(net) {
            let n_out = cv.kernel.0 * cv.oshape.1 * cv.oshape.2;
            let n_in = cv.kernel.1 * cv.ishape.1 * cv.ishape.2;
            for &r in widths {
                let src = wild_matrix(&mut rng, n_out, r);
                let abs_src = src.mapv(f64::abs);
                let mut mag = Array2::<f64>::zeros((n_in, r));
                conv_apply_backward(cv, &abs_src, &mut mag, true);
                let bound_gamma = 2.0 * gamma_n(cv.k_bwd + 2) * (1.0 + 1e-6);
                for abs_w in [false, true] {
                    let mut cpu = Array2::<f64>::zeros((n_in, r));
                    conv_apply_backward(cv, &src, &mut cpu, abs_w);
                    let mut dev = Array2::<f64>::from_elem((n_in, r), f64::NAN);
                    let macs = gpu_conv::conv_backward_on_engine(
                        &engine,
                        cv,
                        &src,
                        &mut dev,
                        abs_w,
                        far_deadline(),
                    )
                    .expect("device conv");
                    assert_eq!(macs, gpu_conv::conv_macs(cv, r), "MAC accounting");
                    for ((ij, &d), (&c, &m)) in dev.indexed_iter().zip(cpu.iter().zip(mag.iter())) {
                        assert!(d.is_finite(), "device entry {ij:?} left unwritten");
                        let tol = bound_gamma * m;
                        assert!(
                            (d - c).abs() <= tol,
                            "k_bwd={} r={r} abs_w={abs_w} {ij:?}: |dev-cpu| = {:e} > {tol:e}",
                            cv.k_bwd,
                            (d - c).abs()
                        );
                        if d.to_bits() != c.to_bits() {
                            differ += 1;
                        }
                        entries += 1;
                    }
                }
            }
        }
    }
    assert!(entries > 100_000, "only {entries} entries compared");
    assert!(
        differ > 0,
        "the reordered device sum never differed from the CPU kernel — the \
         order-independence is untested (vacuous)"
    );
    eprintln!("[mr-gpu-conv-test] cpu parity: {entries} entries, {differ} bitwise-different");
}

/// THE CONTAINMENT TEST. For every conv (random fixtures AND real cifar100
/// resnet shapes), wild operands `L` with an incoming error box `E`, the
/// engine's own charges evaluated ON THE DEVICE ROUTE —
/// `lnew = dev(W^T L)`, `en = certify_up(dev(|W|^T (E + g(|L| + E))),
/// gamma(k_bwd + 8))` with `g = gamma(k_bwd + 2) + rel + gamma*rel` — give an
/// interval `[lnew - en, lnew + en]` that CONTAINS the EXACT-RATIONAL product
/// `W^T L_ex` for `L_ex = L` and for exact corners `L_ex = L + sigma*E`.
/// Equivalently the certified radius is never tighter than the true error.
#[test]
fn gpu_conv_certified_interval_contains_exact_rational_product() {
    let mut rng = StdRng::seed_from_u64(0x6C0F_0005);
    let mut nets: Vec<(TwinNet, Vec<usize>, usize)> = net_fixtures()
        .into_iter()
        .map(|(_, spec, _)| {
            (
                TwinNet::compile(&spec).expect("compile"),
                vec![1, 3],
                usize::MAX,
            )
        })
        .collect();
    // Real shapes: exact-check a random sample of 256 entries per product.
    nets.push((real_shape_convs(), vec![2], 256));
    let engine = ReorderFmaEngine::default();
    let mut checks = 0usize;
    let mut min_slack_ratio = f64::INFINITY;
    for (net, widths, sample) in &nets {
        for cv in convs(net) {
            let n_out = cv.kernel.0 * cv.oshape.1 * cv.oshape.2;
            let n_in = cv.kernel.1 * cv.ishape.1 * cv.ishape.2;
            let g = next_up(
                gamma_n(cv.k_bwd + 2)
                    + cv.weight_rel_err
                    + gamma_n(cv.k_bwd + 2) * cv.weight_rel_err,
            );
            let g2 = gamma_n(cv.k_bwd + 8);
            for &r in widths {
                let l = wild_matrix(&mut rng, n_out, r);
                // Incoming certified error: zero on a third of the entries (the
                // pure-rounding case), tiny relative elsewhere.
                let e = Array2::from_shape_fn((n_out, r), |(i, c)| {
                    if (i + c) % 3 == 0 {
                        0.0
                    } else {
                        l[[i, c]].abs() * 1e-13 * (1 + (i * 7 + c) % 5) as f64
                    }
                });
                // The engine's `combined_err` expression.
                let comb = Array2::from_shape_fn((n_out, r), |(i, c)| {
                    let (lv, ev) = (l[[i, c]], e[[i, c]]);
                    ev + g * (lv.abs() + ev)
                });
                let mut lnew = Array2::<f64>::zeros((n_in, r));
                let mut en = Array2::<f64>::zeros((n_in, r));
                gpu_conv::conv_backward_on_engine(
                    &engine,
                    cv,
                    &l,
                    &mut lnew,
                    false,
                    far_deadline(),
                )
                .expect("device signed product");
                gpu_conv::conv_backward_on_engine(
                    &engine,
                    cv,
                    &comb,
                    &mut en,
                    true,
                    far_deadline(),
                )
                .expect("device |W| product");
                en.mapv_inplace(|v| certify_up(v, g2));

                // Exact operands: the stored L, and two exact corners of its box.
                let l_exact: Vec<BigRational> = l.iter().map(|&v| q(v)).collect();
                let corners: Vec<Vec<BigRational>> = (0..2)
                    .map(|_| {
                        l.iter()
                            .zip(e.iter())
                            .map(|(&lv, &ev)| {
                                if rng.random_range(0.0..1.0) < 0.5 {
                                    q(lv) + q(ev)
                                } else {
                                    q(lv) - q(ev)
                                }
                            })
                            .collect()
                    })
                    .collect();
                let total = n_in * r;
                let picks: Vec<usize> = if *sample >= total {
                    (0..total).collect()
                } else {
                    (0..*sample).map(|_| rng.random_range(0..total)).collect()
                };
                for idx in picks {
                    let (row, col) = (idx / r, idx % r);
                    let mid = lnew[[row, col]];
                    let rad = en[[row, col]];
                    let lo_q = q(mid) - q(rad);
                    let hi_q = q(mid) + q(rad);
                    for (which, src) in std::iter::once(&l_exact).chain(corners.iter()).enumerate()
                    {
                        let exact = exact_backward_entry(cv, src, r, row, col, false);
                        assert!(
                            lo_q <= exact && exact <= hi_q,
                            "CONTAINMENT FALSIFIED: k_bwd={} r={r} entry ({row},{col}) operand \
                             {which}: device lnew={mid:e} en={rad:e} does not enclose the \
                             exact product",
                            cv.k_bwd
                        );
                        checks += 1;
                        if which == 0 && rad > 0.0 {
                            use num_traits::ToPrimitive;
                            let err = (exact - q(mid)).to_f64().unwrap_or(0.0).abs();
                            if err > 0.0 {
                                min_slack_ratio = min_slack_ratio.min(rad / err);
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(checks > 3_000, "only {checks} exact checks ran");
    eprintln!(
        "[mr-gpu-conv-test] exact containment: {checks} checks, 0 violations; smallest \
         certified-radius / true-error ratio on the stored operand = {min_slack_ratio:e}"
    );
}

/// THE ROUTED-PASS FALSIFIER. Every published root bound `dj[k]` computed by
/// an engine whose conv products ALL run on the emulated device (per-column
/// AND row-envelope lanes; the envelope's `R = 1` product included) is `<=`
/// the EXACT margin at sampled points of its box — point boxes included,
/// where only certified rounding separates the bound from the truth — and its
/// y-row passes agree with the CPU-routed ones to f64 precision.
#[test]
fn gpu_conv_routed_root_eval_never_exceeds_exact_margin() {
    let engine = Arc::new(ReorderFmaEngine::default());
    let route = GpuConvRoute::Engine(engine.clone());
    let mut rng = StdRng::seed_from_u64(0x6C0F_0006);
    let mut checks = 0usize;
    for (name, spec, rad0) in net_fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        let n_in = spec.n_in;
        let center: Vec<f64> = (0..n_in).map(|_| rng.random_range(-rad0..rad0)).collect();
        for rad in [0.0, 1e-6, 1e-3, 0.05] {
            let lo: Vec<f64> = center.iter().map(|c| c - rad).collect();
            let hi: Vec<f64> = center.iter().map(|c| c + rad).collect();
            let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
            let (t, _, adv) = tiny_targets(&net, &lo, &hi);
            let extra = if rad > 0.0 { 2 } else { 0 };
            let exact: Vec<Vec<BigRational>> = box_points(&mut rng, &lo, &hi, extra)
                .iter()
                .map(|x| {
                    let s = exact_tensors(&spec, x).pop().expect("scores");
                    adv.iter().map(|&j| &s[t] - &s[j]).collect()
                })
                .collect();
            for lane in [ErrLane::PerCol, ErrLane::Envelope] {
                let cpu = BackwardEngine::new(&net, &root)
                    .with_err_lane(lane)
                    .with_err_min_cols(1);
                let dev = BackwardEngine::new(&net, &root)
                    .with_err_lane(lane)
                    .with_err_min_cols(1)
                    .with_gpu_conv_route(route.clone());
                // Precision parity on the BRANCH-FREE y-row passes (the root
                // eval itself branches on a via-y sign, so its dj may take a
                // different — equally certified — path at a threshold).
                let (al_c, au_c) = cpu.y_rows(None).expect("cpu y rows");
                let (al_d, au_d) = dev.y_rows(None).expect("dev y rows");
                for (lower, oc, od) in [(true, &al_c, &al_d), (false, &au_c, &au_d)] {
                    let (bc, bd) = if lower {
                        (cpu.concretize_lower(oc), dev.concretize_lower(od))
                    } else {
                        (cpu.concretize_upper(oc), dev.concretize_upper(od))
                    };
                    for (j, (&x, &y)) in bc.iter().zip(&bd).enumerate() {
                        assert!(
                            (x - y).abs() <= 1e-9 * (1.0 + x.abs()),
                            "{name} rad {rad:e} {lane:?}: routed y-row {j} bound {y:e} vs cpu \
                             {x:e} — the route lost f64 precision"
                        );
                    }
                }
                let re_dev =
                    super::super::bab::root_eval(&dev, &net, t, &adv).expect("dev root_eval");
                for (p, m) in exact.iter().enumerate() {
                    for (k, mk) in m.iter().enumerate() {
                        let dj = q(re_dev.dj[k]);
                        assert!(
                            &dj <= mk,
                            "FALSIFIED: {name} rad {rad:e} lane {lane:?}: routed dj[{k}] = {:e} \
                             exceeds the EXACT margin at point {p}",
                            re_dev.dj[k]
                        );
                        checks += 1;
                    }
                }
            }
        }
    }
    assert!(checks > 100, "only {checks} exact checks ran");
    assert!(
        engine.calls() > 0,
        "the routed engine never reached the device: VOID, not a pass"
    );
    eprintln!(
        "[mr-gpu-conv-test] routed falsifier: {checks} exact checks, 0 violations, {} device calls",
        engine.calls()
    );
}

/// Domain gates carrying shared beta terms at every trunk layer and, when
/// `per_column`, per-column terms on two columns. Root gates otherwise (no
/// split overrides), so the Lagrangian bound is valid over the WHOLE box.
fn beta_domain(root: &RootGates, r: usize, per_column: bool) -> DomainGates {
    let mut dom = DomainGates::default();
    for (li, rec) in root.layers.iter().enumerate() {
        let n = rec.n;
        dom.beta.insert(
            li,
            vec![
                BetaSplit {
                    neuron: (li * 3) % n,
                    sign: 1,
                    beta: 0.375,
                },
                BetaSplit {
                    neuron: (li * 3 + 1) % n,
                    sign: -1,
                    beta: 1.25,
                },
            ],
        );
        if per_column {
            dom.beta_pc.insert(
                li,
                (0..r.min(2))
                    .map(|c| PcBetaCol {
                        col: (c + 1) % r,
                        terms: vec![BetaSplit {
                            neuron: (li * 5 + c + 2) % n,
                            sign: if c % 2 == 0 { 1 } else { -1 },
                            beta: 0.5 + 0.25 * c as f64,
                        }],
                    })
                    .collect(),
            );
        }
    }
    dom
}

/// Exact `sum_terms beta * s * z(x)` of `terms` at trunk layer `li`.
fn exact_beta_sum(
    net: &TwinNet,
    root: &RootGates,
    tensors: &[Vec<BigRational>],
    li: usize,
    terms: &[BetaSplit],
) -> BigRational {
    let TwinOp::Relu { input, .. } = &net.ops[root.layers[li].op] else {
        panic!("layer {li} is not a relu");
    };
    let mut acc = q(0.0);
    for term in terms {
        acc += q(term.beta * f64::from(term.sign)) * &tensors[*input][term.neuron];
    }
    acc
}

/// The exact Lagrangian objective of column `col` of a pass seeded by `seed`
/// under `dom`: `seed[:, col] . y(x) -/+ sum beta*s*z(x)`.
#[allow(clippy::too_many_arguments)]
fn exact_lagrangian(
    net: &TwinNet,
    root: &RootGates,
    tensors: &[Vec<BigRational>],
    seed: &Seed,
    dom: &DomainGates,
    col: usize,
    lower: bool,
) -> BigRational {
    let y = &tensors[net.i_gemm1 + 1];
    let mut obj = q(0.0);
    for (j, yj) in y.iter().enumerate() {
        let s = seed.s[[j, col]];
        if s != 0.0 {
            obj += q(s) * yj;
        }
    }
    let mut lag = q(0.0);
    for (&li, terms) in &dom.beta {
        lag += exact_beta_sum(net, root, tensors, li, terms);
    }
    for (&li, cols) in &dom.beta_pc {
        for pc in cols.iter().filter(|pc| pc.col == col) {
            lag += exact_beta_sum(net, root, tensors, li, &pc.terms);
        }
    }
    if lower {
        obj - lag
    } else {
        obj + lag
    }
}

/// BETA IS HANDLED: device-routed passes with shared AND per-column beta at
/// every trunk layer (identity seed) and a device-routed DOMAIN-STACKED pass
/// (two blocks, one with shared beta) publish lower bounds `<=` and upper
/// bounds `>=` the EXACT Lagrangian at sampled box points, and agree with the
/// CPU-routed passes to f64 precision. The moved op is linear; beta stays in
/// the host ReLU arm, and this pins that the composition is right.
#[test]
fn gpu_conv_beta_passes_stay_below_exact_lagrangian() {
    let engine = Arc::new(ReorderFmaEngine::default());
    let route = GpuConvRoute::Engine(engine.clone());
    let mut rng = StdRng::seed_from_u64(0x6C0F_0007);
    let mut checks = 0usize;
    let mut moved = false;
    for (name, spec, rad0) in net_fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        let n_in = spec.n_in;
        let center: Vec<f64> = (0..n_in).map(|_| rng.random_range(-rad0..rad0)).collect();
        for rad in [0.0, 0.02] {
            let lo: Vec<f64> = center.iter().map(|c| c - rad).collect();
            let hi: Vec<f64> = center.iter().map(|c| c + rad).collect();
            let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
            let cpu = BackwardEngine::new(&net, &root);
            let dev = BackwardEngine::new(&net, &root).with_gpu_conv_route(route.clone());
            let extra = if rad > 0.0 { 2 } else { 0 };
            let pts = box_points(&mut rng, &lo, &hi, extra);
            let tensors: Vec<Vec<Vec<BigRational>>> =
                pts.iter().map(|x| exact_tensors(&spec, x)).collect();

            // (a) uniform passes: identity seed, shared + per-column beta.
            let seed = dev.identity_seed();
            let r = seed.s.ncols();
            let dom = beta_domain(&root, r, true);
            for dir in [LaneDir::Lower, LaneDir::Upper] {
                let lower = matches!(dir, LaneDir::Lower);
                let out_cpu = cpu
                    .run_collect(&seed, Some(&dom), dir, None, Collect::default())
                    .expect("cpu beta pass");
                let out_dev = dev
                    .run_collect(&seed, Some(&dom), dir, None, Collect::default())
                    .expect("device beta pass");
                let plain = dev
                    .run_collect(&seed, None, dir, None, Collect::default())
                    .expect("device plain pass");
                moved |= out_dev.a.iter().zip(plain.a.iter()).any(|(a, b)| a != b);
                check_pass(
                    &format!("{name}/rad={rad:e}/uniform/{dir:?}"),
                    &cpu,
                    &dev,
                    &out_cpu,
                    &out_dev,
                    lower,
                    |col, x| exact_lagrangian(&net, &root, &tensors[x], &seed, &dom, col, lower),
                    tensors.len(),
                    &mut checks,
                );
            }

            // (b) a domain-stacked pass: block A carries shared beta (the
            // stacked arm refuses per-column beta), block B is the root.
            let dom_a = beta_domain(&root, 3, false);
            let dom_b = DomainGates::default();
            let no_exc = Exceptions::default();
            let stacked_seed = plain_seed(net.n_y, 5, 3);
            let blocks = vec![
                RowDomainGateBlock {
                    columns: 0..3,
                    gates: &dom_a,
                    exceptions: &no_exc,
                },
                RowDomainGateBlock {
                    columns: 3..5,
                    gates: &dom_b,
                    exceptions: &no_exc,
                },
            ];
            for dir in [LaneDir::Lower, LaneDir::Upper] {
                let lower = matches!(dir, LaneDir::Lower);
                let out_cpu = cpu
                    .run_domain_stacked(&stacked_seed, &blocks, dir)
                    .expect("cpu stacked pass");
                let out_dev = dev
                    .run_domain_stacked(&stacked_seed, &blocks, dir)
                    .expect("device stacked pass");
                check_pass(
                    &format!("{name}/rad={rad:e}/stacked/{dir:?}"),
                    &cpu,
                    &dev,
                    &out_cpu,
                    &out_dev,
                    lower,
                    |col, x| {
                        let gates = if col < 3 { &dom_a } else { &dom_b };
                        exact_lagrangian(&net, &root, &tensors[x], &stacked_seed, gates, col, lower)
                    },
                    tensors.len(),
                    &mut checks,
                );
            }
        }
    }
    assert!(
        moved,
        "beta never moved a device-routed coefficient (vacuous)"
    );
    assert!(checks > 200, "only {checks} exact Lagrangian checks ran");
    assert!(
        engine.calls() > 0,
        "the routed engine never reached the device: VOID, not a pass"
    );
    eprintln!(
        "[mr-gpu-conv-test] beta Lagrangian oracle: {checks} exact checks, 0 violations, {} \
         device calls",
        engine.calls()
    );
}

/// Shared body of the beta test: precision parity against the CPU-routed
/// pass, then exact containment of every column at every sample point.
#[allow(clippy::too_many_arguments)]
fn check_pass(
    what: &str,
    cpu: &BackwardEngine<'_>,
    dev: &BackwardEngine<'_>,
    out_cpu: &PassOut,
    out_dev: &PassOut,
    lower: bool,
    exact: impl Fn(usize, usize) -> BigRational,
    points: usize,
    checks: &mut usize,
) {
    let (bc, bd) = if lower {
        (cpu.concretize_lower(out_cpu), dev.concretize_lower(out_dev))
    } else {
        (cpu.concretize_upper(out_cpu), dev.concretize_upper(out_dev))
    };
    assert_eq!(bc.len(), bd.len(), "{what}: column count");
    for (c, (&x, &y)) in bc.iter().zip(&bd).enumerate() {
        assert!(
            (x - y).abs() <= 1e-9 * (1.0 + x.abs()),
            "{what}: column {c} routed bound {y:e} vs cpu {x:e} — the route lost f64 precision"
        );
    }
    for p in 0..points {
        for (c, &b) in bd.iter().enumerate() {
            let truth = exact(c, p);
            let bq = q(b);
            assert!(
                if lower { bq <= truth } else { bq >= truth },
                "LAGRANGIAN FALSIFIED: {what}: column {c} point {p}: routed bound {b:e} is on \
                 the wrong side of the exact objective"
            );
            *checks += 1;
        }
    }
}

/// FAIL-CLOSED: under every engine failure mode (error, deadline, a NaN in the
/// output, a wrong-length output) the routed engine publishes the DARK pass
/// bit-for-bit — the CPU kernel overwrites every element — and the failing
/// engine was really consulted.
#[test]
fn gpu_conv_refusals_fall_back_bit_identically() {
    let (name, spec, rad) = net_fixtures().remove(2);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-rad; spec.n_in];
    let hi = vec![rad; spec.n_in];
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let dark = BackwardEngine::new(&net, &root);
    let seeds = [dark.identity_seed(), plain_seed(net.n_y, 4, 9)];
    let dom = beta_domain(&root, 4, true);
    for mode in [
        Failure::Error,
        Failure::Deadline,
        Failure::NonFinite,
        Failure::WrongLength,
    ] {
        let failing = Arc::new(FailingEngine {
            mode,
            calls: AtomicUsize::new(0),
        });
        for lane in [ErrLane::PerCol, ErrLane::Envelope] {
            let dark = BackwardEngine::new(&net, &root)
                .with_err_lane(lane)
                .with_err_min_cols(1);
            let routed = BackwardEngine::new(&net, &root)
                .with_err_lane(lane)
                .with_err_min_cols(1)
                .with_gpu_conv_route(GpuConvRoute::Engine(failing.clone()));
            for (si, seed) in seeds.iter().enumerate() {
                let d = if seed.s.ncols() == 4 {
                    Some(&dom)
                } else {
                    None
                };
                for dir in [LaneDir::Lower, LaneDir::Upper] {
                    let a = dark
                        .run_collect(seed, d, dir, None, Collect::default())
                        .expect("dark pass");
                    let b = routed
                        .run_collect(seed, d, dir, None, Collect::default())
                        .expect("fallback pass");
                    let what = format!("{name}/{mode:?}/{lane:?}/seed{si}/{dir:?}");
                    assert_pass_bits_equal(&a, &b, &what);
                }
            }
        }
        assert!(
            failing.calls.load(Ordering::SeqCst) > 0,
            "{mode:?}: the failing engine was never consulted (vacuous)"
        );
    }
}

fn assert_pass_bits_equal(a: &PassOut, b: &PassOut, what: &str) {
    assert_eq!(a.a.raw_dim(), b.a.raw_dim(), "{what}: shape");
    for (i, (x, y)) in a.a.iter().zip(b.a.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: coefficient {i}");
    }
    match (a.e.as_ref(), b.e.as_ref()) {
        (Some(ea), Some(eb)) => {
            for (i, (x, y)) in ea.iter().zip(eb.iter()).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "{what}: error {i}");
            }
        }
        (None, None) => {}
        _ => panic!("{what}: error lanes differ in presence"),
    }
    for (i, (x, y)) in a.b.iter().zip(&b.b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: bias {i}");
    }
    for (i, (x, y)) in a.eb.iter().zip(&b.eb).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: bias error {i}");
    }
}

/// The stop receipt carries EVERY field on every line (fixed schema), and the
/// lane-start line is the dark one in a process that never armed the key.
#[test]
fn gpu_conv_stop_line_has_fixed_schema() {
    let line = gpu_conv::stop_line();
    assert!(line.starts_with("[mr-gpu-conv] done armed="), "{line}");
    for field in [
        " convs=",
        " gpu=",
        " gpu_signed=",
        " gpu_abs=",
        " gpu_env=",
        " gpu_macs=",
        " gpu_secs=",
        " fallback=",
        " small=",
        " parity=",
        " transposed=",
        " shape=",
        " nonfinite_in=",
        " cpu_only=",
        " no_engine=",
        " backend_refused=",
        " engine_err=",
        " deadline=",
        " bad_output=",
        " backoff=",
        " backend=",
        " verdict=",
    ] {
        assert!(line.contains(field), "stop line lacks {field:?}: {line}");
    }
    let census = gpu_conv::census_fields();
    assert!(census.contains(" gpuconv_armed="), "{census}");
    assert!(census.contains(" gpuconv_gpu="), "{census}");
    assert!(census.contains(" gpuconv_fallback="), "{census}");
    if !gpu_conv::armed() {
        assert!(
            line.contains(" armed=0 ") || line.contains("armed=0 "),
            "{line}"
        );
        assert!(line.ends_with("verdict=dark"), "{line}");
        assert_eq!(gpu_conv::announce(), "[mr-gpu-conv] dark");
    }
}

/// #bi-gpu: the backward-interm / joint-refresh prefix pairs on a scoped route
/// of this seam — exact-rational containment (operator and pair level, random
/// and real cct CNN7 shapes), CPU parity of the routed phase, bit-identical
/// refusal fallback (device and RSS), dark default and the receipt schema.
mod bi_gpu_tests;

/// `#bi-patch` (per-column receptive-field prefix passes): the patch walk's
/// coefficients equal the dense pass's and its error lane is dominated, its
/// bounds are never looser and contain the EXACT pre-activations (point boxes
/// included), frames cover the dense support, the crossover and the fused pair
/// are exact, refusals fall back bit-identically, and the phase is dark by
/// default. A child of this module to reuse the exact-rational oracle above.
mod patch_tests;

#[test]
fn gpu_conv_shared_dispatch_keeps_dark_precision_lazy_and_calls_armed_precision_once() {
    let _env = crate::tests::lock_env_shared();
    let (_, spec, _) = net_fixtures().remove(0);
    let net = TwinNet::compile(&spec).expect("compile original convolution fixture");
    let cv = convs(&net)[0];
    let columns = 3;
    let src = Array2::from_shape_fn(
        (cv.kernel.0 * cv.oshape.1 * cv.oshape.2, columns),
        |(i, column)| (i * 3 + column) as f64 * 0.125 - 1.0,
    );
    let shape = (cv.kernel.1 * cv.ishape.1 * cv.ishape.2, columns);
    let dark_scope = gpu_conv::ScopedRoute::new(GpuConvRoute::Off);
    let failing = Arc::new(FailingEngine {
        mode: Failure::Error,
        calls: AtomicUsize::new(0),
    });
    let armed_scope = gpu_conv::ScopedRoute::new(GpuConvRoute::Engine(failing.clone()));
    for abs_w in [false, true] {
        let lane = if abs_w {
            ConvLane::Abs
        } else {
            ConvLane::Signed
        };
        let mut expected = Array2::zeros(shape);
        conv_apply_backward(cv, &src, &mut expected, abs_w);
        for route in [GpuConvRoute::Off, GpuConvRoute::Scoped(dark_scope.clone())] {
            let mut actual = Array2::from_elem(shape, -1234.5);
            BackwardEngine::conv_backward(&route, cv, &src, &mut actual, abs_w, lane, || {
                panic!("dark dispatch must not inspect the precision operand");
            });
            assert_eq!(actual.raw_dim(), expected.raw_dim());
            assert!(actual
                .iter()
                .zip(expected.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        assert_eq!(dark_scope.tally().convs, 0);
        for route in [
            GpuConvRoute::Engine(failing.clone()),
            GpuConvRoute::Scoped(armed_scope.clone()),
        ] {
            let precision_calls = AtomicUsize::new(0);
            let before = failing.calls.load(Ordering::SeqCst);
            let mut actual = Array2::from_elem(shape, -1234.5);
            BackwardEngine::conv_backward(&route, cv, &src, &mut actual, abs_w, lane, || {
                precision_calls.fetch_add(1, Ordering::SeqCst);
                true
            });
            assert_eq!(precision_calls.load(Ordering::SeqCst), 1);
            assert_eq!(failing.calls.load(Ordering::SeqCst), before + 1);
            assert_eq!(actual.raw_dim(), expected.raw_dim());
            assert!(actual
                .iter()
                .zip(expected.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }
    assert_eq!(armed_scope.tally().convs, 2);
    assert_eq!(armed_scope.tally().gpu, 0);
}

/// An intentionally incomplete emulated device transaction: a visible first
/// tile marker, then refusal. No complete call or certificate accepts that
/// marker as mathematical evidence; the test requires a whole CPU overwrite.
struct FirstTileThenRefusal {
    calls: AtomicUsize,
}

impl GemmEngine for FirstTileThenRefusal {
    fn backend_provenance(&self) -> &'static str {
        "test-partial-tile-then-refusal"
    }

    fn gemm_f32(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        Err(NyError::UnsupportedOp(
            "f64-only incomplete test transaction".into(),
        ))
    }

    fn gemm_f64_with_deadline(
        &self,
        m: usize,
        _k: usize,
        n: usize,
        _a: &[f64],
        _b: &[f64],
        _deadline: Instant,
        _max_dispatch_macs: usize,
    ) -> ny_core::Result<Vec<f64>> {
        match self.calls.fetch_add(1, Ordering::SeqCst) {
            0 => Ok(vec![42.0; m * n]),
            _ => Err(NyError::InternalError("second tile refused".into())),
        }
    }
}

#[test]
fn gpu_conv_shared_dispatch_overwrites_an_observed_first_tile_after_later_refusal() {
    let _env = crate::tests::lock_env_shared();
    const CHANNELS: usize = 128;
    let channels = CHANNELS;
    let columns = 1024;
    let weights: Vec<_> = (0..channels * 9)
        .map(|i| {
            if i % 2 == 0 {
                1.0 / 128.0
            } else {
                -1.0 / 256.0
            }
        })
        .collect();
    let bias = [0.0; CHANNELS];
    let cv = crate::margin_row::net::compile_conv(
        0,
        &weights,
        &bias,
        &bias,
        0.0,
        (channels, 1, 3, 3),
        (1, 1),
        (1, 1, 1, 1),
        (1, 2, 2),
        (channels, 2, 2),
    );
    let input_positions = 4;
    let k = cv.kernel.0 * cv.kernel.2 * cv.kernel.3;
    let first_positions = (gpu_conv::GPU_CONV_TILE_ELEMS / (k * columns)).clamp(1, input_positions);
    assert_eq!(first_positions, 3);
    assert_eq!(input_positions - first_positions, 1);
    let first_cells = first_positions * columns;
    let src = Array2::from_shape_fn((channels * input_positions, columns), |(i, j)| {
        ((i + j) % 11) as f64 / 1024.0 - 5.0 / 1024.0
    });
    let shape = (input_positions, columns);
    for abs_w in [false, true] {
        let lane = if abs_w {
            ConvLane::Abs
        } else {
            ConvLane::Signed
        };
        let probe = FirstTileThenRefusal {
            calls: AtomicUsize::new(0),
        };
        let mut partial = Array2::from_elem(shape, -1234.5);
        assert_eq!(
            gpu_conv::conv_backward_on_engine(
                &probe,
                &cv,
                &src,
                &mut partial,
                abs_w,
                Instant::now() + Duration::from_secs(30)
            ),
            Err(Refusal::Engine),
        );
        assert_eq!(probe.calls.load(Ordering::SeqCst), 2);
        let partial = partial.as_slice().unwrap();
        assert!(partial[..first_cells].iter().all(|&value| value == 42.0));
        assert!(partial[first_cells..].iter().all(|&value| value == -1234.5));
        let mut expected = Array2::zeros(shape);
        conv_apply_backward(&cv, &src, &mut expected, abs_w);
        assert!(expected
            .iter()
            .all(|&value| value != 42.0 && value != -1234.5));
        for scoped in [false, true] {
            let engine = Arc::new(FirstTileThenRefusal {
                calls: AtomicUsize::new(0),
            });
            let route = GpuConvRoute::Engine(engine.clone());
            let route = if scoped {
                GpuConvRoute::Scoped(gpu_conv::ScopedRoute::new(route))
            } else {
                route
            };
            let precision_calls = AtomicUsize::new(0);
            let mut actual = Array2::from_elem(shape, -1234.5);
            BackwardEngine::conv_backward(&route, &cv, &src, &mut actual, abs_w, lane, || {
                precision_calls.fetch_add(1, Ordering::SeqCst);
                true
            });
            assert_eq!(precision_calls.load(Ordering::SeqCst), 1);
            assert_eq!(engine.calls.load(Ordering::SeqCst), 2);
            assert!(actual
                .iter()
                .zip(expected.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()));
            if let GpuConvRoute::Scoped(scope) = route {
                assert_eq!(scope.tally().convs, 1);
                assert_eq!(scope.tally().gpu, 0);
            }
        }
    }
}
