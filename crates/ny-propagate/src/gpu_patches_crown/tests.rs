// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#vgg-gpu-crown` soundness, fallback and dark-default tests.
//!
//! The device is emulated by [`EmuEngine`]: f64 AND f32 GEMMs that sum every
//! dot product in a DIFFERENT order than the CPU product (reversed
//! contraction, four interleaved FMA accumulators, a tree combine — the shape
//! of a blocked cuBLAS GEMM). It implements the plain `gemm_f32` and only the
//! DEADLINE-BOUNDED f64 method, so the f64 calls also exercise the
//! plain -> deadline-method fallback of the product seam.
//!
//! * `certified_linear_product_contains_exact_rational` /
//!   `certified_conv_product_contains_exact_rational` — THE operator-level
//!   containment: `[mu - err, mu + err]` contains the EXACT-rational
//!   `W^T lambda` for wild-exponent `lambda`, on the CPU route and on the
//!   emulated device in both precisions, random and real (VGG16 block) shapes,
//!   strides 1 and 2; and no in-image coefficient ever falls outside the
//!   clipped backward window.
//! * `gpu_certified_crown_bound_never_above_sampled_minimum` — the whole
//!   pass on a small conv net (conv / relu / pool / stride-2 conv / flatten /
//!   linear / relu / linear): every certified box contains every sampled
//!   activation, and every objective's certified lower bound is below its
//!   sampled minimum, for the CPU route and the emulated GPU in f64 and f32,
//!   with the GPU route demonstrably engaged.
//! * `truncated_depth_is_sound` — the same containment for `depth` 1 and 2.
//! * `refusal_fallback_is_bit_identical` — every engine failure mode, on the
//!   certificate GEMM or the signed GEMM, publishes the CPU route bit-for-bit.
//! * `chunking_is_bit_identical` — the row chunk size never changes a bit.
//! * `gpu_patches_crown_is_dark_by_default`, `config_validation_refuses_*`,
//!   `observable_line_has_fixed_schema`, `plan_refuses_*`,
//!   `admission_refuses_*`, `chunk_rows_for_budget_*`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use num_rational::BigRational;
use ny_core::{GemmEngine, NyError};
use ny_tensor::BoundedTensor;

use super::backward::back_window;
use super::boxes::Boxes;
use super::plan::{ChainPlan, ConvSpec, LinearSpec, OpKind, Shape3};
use super::product::{conv_product, linear_product, ConvGeom, Ctx, Route, Stats};
use super::{
    chunk_rows_for_budget, run_inner, run_root_with_route, GpuPatchesCrownConfig,
    GpuPatchesCrownOutcome, GpuPatchesPrecision, Inner, Tally,
};
use crate::layers::{
    Conv2dLayer, FlattenLayer, Layer, LinearLayer, MaxPool2dLayer, ReLULayer, SigmoidLayer,
};
use crate::{BetaCrownConfig, GraphNetwork, GraphNode};

// ---------------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------------

/// Deterministic xorshift.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `[-1, 1)`.
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// Uniform in `[0, 1)`.
    fn frac(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    fn f32s(&mut self, n: usize, scale: f64) -> Vec<f32> {
        (0..n).map(|_| (self.unit() * scale) as f32).collect()
    }

    /// Mixed-sign, mixed-EXPONENT (2^-30 .. 2^30) value, zero 1 time in 8:
    /// the dynamic range that makes two summation orders round differently.
    fn wild(&mut self) -> f64 {
        if self.below(8) == 0 {
            return 0.0;
        }
        let e = self.below(61) as i32 - 30;
        self.unit() * 2f64.powi(e)
    }
}

fn q(v: f64) -> BigRational {
    BigRational::from_float(v).expect("finite f64 is an exact dyadic")
}

fn far_deadline() -> Instant {
    Instant::now() + Duration::from_secs(600)
}

/// f64 AND f32 GEMMs in a deliberately different summation order.
#[derive(Default)]
struct EmuEngine {
    calls: AtomicUsize,
}

fn emu_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "lhs shape");
    assert_eq!(b.len(), k * n, "rhs shape");
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = [0.0f32; 4];
            for (step, l) in (0..k).rev().enumerate() {
                acc[step % 4] = a[i * k + l].mul_add(b[l * n + j], acc[step % 4]);
            }
            c[i * n + j] = (acc[0] + acc[2]) + (acc[1] + acc[3]);
        }
    }
    c
}

fn emu_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
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
    c
}

impl GemmEngine for EmuEngine {
    fn backend_provenance(&self) -> &'static str {
        "test-emu-reorder-fma"
    }

    fn gemm_f32(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(emu_f32(m, k, n, a, b))
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
            return Err(NyError::DeadlineExceeded("emu engine".into()));
        }
        Ok(emu_f64(m, k, n, a, b))
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

/// Fails every f32 and/or every f64 call in one fixed way (the other width
/// behaves like [`EmuEngine`]).
struct FailingEngine {
    f32_mode: Option<Failure>,
    f64_mode: Option<Failure>,
    calls: AtomicUsize,
}

fn fail_with<T: Copy>(mode: Failure, len: usize, zero: T, bad: T) -> ny_core::Result<Vec<T>> {
    match mode {
        Failure::Error => Err(NyError::InternalError("test engine failure".into())),
        Failure::Deadline => Err(NyError::DeadlineExceeded("test engine".into())),
        Failure::NonFinite => {
            let mut c = vec![zero; len];
            if let Some(last) = c.last_mut() {
                *last = bad;
            }
            Ok(c)
        }
        Failure::WrongLength => Ok(vec![zero; len + 1]),
    }
}

impl GemmEngine for FailingEngine {
    fn backend_provenance(&self) -> &'static str {
        "test-failing"
    }

    fn gemm_f32(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.f32_mode {
            Some(mode) => fail_with(mode, m * n, 0.0f32, f32::NAN),
            None => Ok(emu_f32(m, k, n, a, b)),
        }
    }

    fn gemm_f64_with_deadline(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f64],
        b: &[f64],
        _deadline: Instant,
        _max_dispatch_macs: usize,
    ) -> ny_core::Result<Vec<f64>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.f64_mode {
            Some(mode) => fail_with(mode, m * n, 0.0f64, f64::NAN),
            None => Ok(emu_f64(m, k, n, a, b)),
        }
    }
}

fn ctx<'a>(route: &'a Route, stats: &'a Stats, precision: GpuPatchesPrecision) -> Ctx<'a> {
    Ctx {
        route,
        stats,
        precision,
        tile_bytes: 256 << 20,
        deadline: far_deadline(),
    }
}

/// The three routes every soundness oracle runs under.
fn routes() -> Vec<(&'static str, Route, GpuPatchesPrecision)> {
    vec![
        ("cpu", Route::Cpu, GpuPatchesPrecision::F64),
        (
            "emu-f64",
            Route::Engine(Arc::new(EmuEngine::default())),
            GpuPatchesPrecision::F64,
        ),
        (
            "emu-f32",
            Route::Engine(Arc::new(EmuEngine::default())),
            GpuPatchesPrecision::F32,
        ),
    ]
}

fn assert_contains(what: &str, mu: f64, err: f64, exact: &BigRational) {
    assert!(err.is_finite() && err >= 0.0, "{what}: bad err {err}");
    let (m, e) = (q(mu), q(err));
    assert!(
        &(&m - &e) <= exact && exact <= &(&m + &e),
        "{what}: exact {exact} escapes [{mu:e} -+ {err:e}]"
    );
}

// ---------------------------------------------------------------------------
//  Operator-level: exact-rational containment
// ---------------------------------------------------------------------------

#[test]
fn certified_linear_product_contains_exact_rational() {
    let mut rng = Rng::new(0x5EED_0001);
    for &(rows, nout, nin) in &[(3usize, 37usize, 29usize), (2, 256, 64), (1, 7, 300)] {
        let w = rng.f32s(nout * nin, 0.8);
        let spec = LinearSpec::new(nout, nin, &w, vec![0.0; nout]).expect("spec");
        let lam: Vec<f64> = (0..rows * nout).map(|_| rng.wild()).collect();
        for (name, route, precision) in routes() {
            let stats = Stats::default();
            let c = ctx(&route, &stats, precision);
            let cert = linear_product(&c, &spec, rows, &lam).expect("product");
            if route.is_gpu() {
                assert!(
                    stats.gpu_passes.load(Ordering::SeqCst) > 0,
                    "{name}: the device route did not engage (VOID)"
                );
            }
            for r in 0..rows {
                for j in 0..nin {
                    let mut exact = q(0.0);
                    for o in 0..nout {
                        let l = lam[r * nout + o];
                        if l != 0.0 {
                            exact += q(l) * q(spec.weight(o, j));
                        }
                    }
                    assert_contains(
                        &format!("{name} linear {rows}x{nout}x{nin} r={r} j={j}"),
                        cert.mu[r * nin + j],
                        cert.err[r * nin + j],
                        &exact,
                    );
                }
            }
        }
    }
}

/// One conv-product containment case (see the module docs).
#[allow(clippy::too_many_arguments)]
fn check_conv_case(
    rng: &mut Rng,
    cout: usize,
    cin: usize,
    k: usize,
    stride: usize,
    pad: usize,
    inp: (usize, usize),
    win: (usize, usize),
    rows: usize,
    entry_stride: usize,
) {
    let (h, w) = inp;
    let oh = (h + 2 * pad - k) / stride + 1;
    let ow = (w + 2 * pad - k) / stride + 1;
    let (qh, qw) = (win.0.min(oh), win.1.min(ow));
    let wt: Vec<f64> = rng
        .f32s(cout * cin * k * k, 0.5)
        .into_iter()
        .map(f64::from)
        .collect();
    let spec = ConvSpec::new(
        cout,
        cin,
        k,
        k,
        (stride, stride),
        (pad, pad),
        wt,
        vec![0.0; cout],
        false,
    )
    .expect("spec");
    let oy: Vec<usize> = (0..rows).map(|_| rng.below(oh - qh + 1)).collect();
    let ox: Vec<usize> = (0..rows).map(|_| rng.below(ow - qw + 1)).collect();
    let lam: Vec<f64> = (0..rows * cout * qh * qw).map(|_| rng.wild()).collect();
    let (qh2, oy2) = back_window(qh, &oy, stride, pad, k, h);
    let (qw2, ox2) = back_window(qw, &ox, stride, pad, k, w);

    // Exact dense W^T lambda over the WHOLE input, by the forward definition
    // `Y2 = s*Y - p + ky` (independent of the gather under test).
    let mut dense: Vec<Vec<BigRational>> = Vec::with_capacity(rows);
    let mut touched: Vec<Vec<bool>> = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut d = vec![q(0.0); cin * h * w];
        let mut t = vec![false; cin * h * w];
        for c in 0..cout {
            for y in 0..qh {
                for x in 0..qw {
                    let l = lam[((r * cout + c) * qh + y) * qw + x];
                    let (yy, xx) = (oy[r] + y, ox[r] + x);
                    for ky in 0..k {
                        for kx in 0..k {
                            let y2 = (stride * yy + ky) as isize - pad as isize;
                            let x2 = (stride * xx + kx) as isize - pad as isize;
                            if y2 < 0 || y2 >= h as isize || x2 < 0 || x2 >= w as isize {
                                continue;
                            }
                            for ci in 0..cin {
                                let idx = (ci * h + y2 as usize) * w + x2 as usize;
                                t[idx] = true;
                                if l != 0.0 {
                                    d[idx] += q(l) * q(spec.weight(c, ci, ky, kx));
                                }
                            }
                        }
                    }
                }
            }
        }
        dense.push(d);
        touched.push(t);
    }

    for (name, route, precision) in routes() {
        let stats = Stats::default();
        let c = ctx(&route, &stats, precision);
        let geom = ConvGeom {
            rows,
            qh,
            qw,
            oy: &oy,
            ox: &ox,
            qh2,
            qw2,
            oy2: &oy2,
            ox2: &ox2,
        };
        let cert = conv_product(&c, &spec, &geom, &lam).expect("product");
        if route.is_gpu() {
            assert!(
                stats.gpu_passes.load(Ordering::SeqCst) > 0,
                "{name}: the device route did not engage (VOID)"
            );
        }
        let plane2 = qh2 * qw2;
        for r in 0..rows {
            // No in-image coefficient outside the clipped window.
            for ci in 0..cin {
                for yy in 0..h {
                    for xx in 0..w {
                        let inside =
                            yy >= oy2[r] && yy < oy2[r] + qh2 && xx >= ox2[r] && xx < ox2[r] + qw2;
                        assert!(
                            inside || !touched[r][(ci * h + yy) * w + xx],
                            "{name}: coefficient at ({ci},{yy},{xx}) dropped by the window"
                        );
                    }
                }
            }
            let mut e = 0usize;
            for ci in 0..cin {
                for y2 in 0..qh2 {
                    for x2 in 0..qw2 {
                        e += 1;
                        if !e.is_multiple_of(entry_stride) {
                            continue;
                        }
                        let i = (r * cin + ci) * plane2 + y2 * qw2 + x2;
                        let exact = &dense[r][(ci * h + oy2[r] + y2) * w + ox2[r] + x2];
                        assert_contains(
                            &format!(
                                "{name} conv co={cout} ci={cin} s={stride} r={r} ({ci},{y2},{x2})"
                            ),
                            cert.mu[i],
                            cert.err[i],
                            exact,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn certified_conv_product_contains_exact_rational() {
    let mut rng = Rng::new(0x5EED_0002);
    // Random small blocks, stride 1 and 2, windows touching the borders.
    check_conv_case(&mut rng, 5, 4, 3, 1, 1, (7, 6), (3, 2), 3, 1);
    check_conv_case(&mut rng, 3, 2, 3, 2, 1, (9, 8), (2, 3), 4, 1);
    check_conv_case(&mut rng, 4, 3, 2, 2, 0, (8, 8), (4, 4), 2, 1);
    check_conv_case(&mut rng, 2, 3, 3, 1, 1, (5, 5), (5, 5), 2, 1);
    // Real VGG16 block shape: 64 -> 64, 3x3, pad 1 (contraction 576), and a
    // full-image window at 14x14 with 512-channel-like depth scaled down.
    check_conv_case(&mut rng, 64, 64, 3, 1, 1, (14, 14), (3, 3), 2, 7);
    check_conv_case(&mut rng, 32, 16, 3, 1, 1, (7, 7), (7, 7), 1, 5);
}

// ---------------------------------------------------------------------------
//  Whole pass on a small conv net
// ---------------------------------------------------------------------------

/// conv(3->4, s1 p1) / relu / pool 2x2 / conv(4->5, s2 p1) / relu / flatten /
/// linear(->6) / relu / linear(->3) on a `3 x h x w` input.
fn tiny_net(rng: &mut Rng, h: usize, w: usize) -> GraphNetwork {
    let conv1 = Conv2dLayer::new(
        ArrayD::from_shape_vec(IxDyn(&[4, 3, 3, 3]), rng.f32s(4 * 3 * 9, 0.45)).unwrap(),
        Some(Array1::from_vec(rng.f32s(4, 0.1))),
        (1, 1),
        (1, 1),
    )
    .unwrap();
    let conv2 = Conv2dLayer::new(
        ArrayD::from_shape_vec(IxDyn(&[5, 4, 3, 3]), rng.f32s(5 * 4 * 9, 0.4)).unwrap(),
        Some(Array1::from_vec(rng.f32s(5, 0.1))),
        (2, 2),
        (1, 1),
    )
    .unwrap();
    let (ph, pw) = (h / 2, w / 2);
    let (oh, ow) = ((ph + 2 - 3) / 2 + 1, (pw + 2 - 3) / 2 + 1);
    let flat = 5 * oh * ow;
    let fc1 = LinearLayer::new(
        Array2::from_shape_vec((6, flat), rng.f32s(6 * flat, 0.5)).unwrap(),
        Some(Array1::from_vec(rng.f32s(6, 0.1))),
    )
    .unwrap();
    let fc2 = LinearLayer::new(
        Array2::from_shape_vec((3, 6), rng.f32s(18, 0.6)).unwrap(),
        Some(Array1::from_vec(rng.f32s(3, 0.1))),
    )
    .unwrap();
    let nodes = vec![
        GraphNode::from_input("conv1", Layer::Conv2d(conv1)),
        GraphNode::new("relu1", Layer::ReLU(ReLULayer), vec!["conv1".into()]),
        GraphNode::new(
            "pool1",
            Layer::MaxPool2d(MaxPool2dLayer::new((2, 2), (2, 2), (0, 0))),
            vec!["relu1".into()],
        ),
        GraphNode::new("conv2", Layer::Conv2d(conv2), vec!["pool1".into()]),
        GraphNode::new("relu2", Layer::ReLU(ReLULayer), vec!["conv2".into()]),
        GraphNode::new(
            "flat",
            Layer::Flatten(FlattenLayer::new(1)),
            vec!["relu2".into()],
        ),
        GraphNode::new("fc1", Layer::Linear(fc1), vec!["flat".into()]),
        GraphNode::new("relu3", Layer::ReLU(ReLULayer), vec!["fc1".into()]),
        GraphNode::new("fc2", Layer::Linear(fc2), vec!["relu3".into()]),
    ];
    let mut g = GraphNetwork::new();
    for n in nodes {
        g.add_node(n);
    }
    g.set_output("fc2");
    g
}

fn input_box(rng: &mut Rng, h: usize, w: usize, eps: f32) -> BoundedTensor {
    let n = 3 * h * w;
    let base = rng.f32s(n, 1.0);
    let lo: Vec<f32> = base.iter().map(|&v| v - eps).collect();
    let hi: Vec<f32> = base.iter().map(|&v| v + eps).collect();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3, h, w]), lo).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3, h, w]), hi).unwrap(),
    )
    .unwrap()
}

/// f64 reference forward over the admitted plan, every value returned.
fn forward_all(plan: &ChainPlan<'_>, x: &[f64]) -> Vec<Vec<f64>> {
    let mut vals = vec![x.to_vec()];
    for op in &plan.ops {
        let cur = vals.last().expect("value");
        let (inp, out) = (op.input, op.output);
        let next: Vec<f64> = match &op.kind {
            OpKind::Conv(s) => {
                let mut y = vec![0.0; out.numel()];
                for co in 0..out.c {
                    for oy in 0..out.h {
                        for ox in 0..out.w {
                            let mut acc = s.bias[co];
                            for ci in 0..inp.c {
                                for ky in 0..s.kh {
                                    for kx in 0..s.kw {
                                        let iy = (oy * s.sh + ky) as isize - s.ph as isize;
                                        let ix = (ox * s.sw + kx) as isize - s.pw as isize;
                                        if iy < 0
                                            || ix < 0
                                            || iy >= inp.h as isize
                                            || ix >= inp.w as isize
                                        {
                                            continue;
                                        }
                                        acc += s.weight(co, ci, ky, kx)
                                            * cur[(ci * inp.h + iy as usize) * inp.w + ix as usize];
                                    }
                                }
                            }
                            y[(co * out.h + oy) * out.w + ox] = acc;
                        }
                    }
                }
                y
            }
            OpKind::Linear(s) => (0..s.nout)
                .map(|o| {
                    let mut acc = s.bias[o];
                    for (j, &v) in cur.iter().enumerate() {
                        acc += s.weight(o, j) * v;
                    }
                    acc
                })
                .collect(),
            OpKind::Relu => cur.iter().map(|v| v.max(0.0)).collect(),
            OpKind::MaxPool(p) => {
                let mut y = vec![f64::NEG_INFINITY; out.numel()];
                for c in 0..out.c {
                    for oy in 0..out.h {
                        for ox in 0..out.w {
                            let o = (c * out.h + oy) * out.w + ox;
                            for dy in 0..p.kh {
                                for dx in 0..p.kw {
                                    let j = (c * inp.h + oy * p.sh + dy) * inp.w + ox * p.sw + dx;
                                    y[o] = y[o].max(cur[j]);
                                }
                            }
                        }
                    }
                }
                y
            }
            OpKind::Reshape => cur.clone(),
        };
        vals.push(next);
    }
    vals
}

/// Center, both corners, random vertices and random interior points.
fn samples(rng: &mut Rng, input: &BoundedTensor, count: usize) -> Vec<Vec<f64>> {
    let lo: Vec<f64> = input.lower().iter().map(|&v| f64::from(v)).collect();
    let hi: Vec<f64> = input.upper().iter().map(|&v| f64::from(v)).collect();
    let mut pts = vec![
        lo.iter()
            .zip(&hi)
            .map(|(&l, &h)| 0.5 * l + 0.5 * h)
            .collect(),
        lo.clone(),
        hi.clone(),
    ];
    for i in 0..count {
        pts.push(
            lo.iter()
                .zip(&hi)
                .map(|(&l, &h)| {
                    if i % 2 == 0 {
                        if rng.below(2) == 0 {
                            l
                        } else {
                            h
                        }
                    } else {
                        l + (h - l) * rng.frac()
                    }
                })
                .collect(),
        );
    }
    pts
}

fn objectives() -> Vec<Vec<f32>> {
    vec![
        vec![1.0, -1.0, 0.0],
        vec![-0.5, 0.25, 1.0],
        vec![0.0, 0.0, -1.0],
    ]
}

fn test_cfg() -> GpuPatchesCrownConfig {
    GpuPatchesCrownConfig {
        min_perturbed: 0,
        chunk_rows: 7,
        ..GpuPatchesCrownConfig::default()
    }
}

fn run_pass(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    cfg: &GpuPatchesCrownConfig,
    route: &Route,
) -> (Inner, Stats) {
    let stats = Stats::default();
    let mut tally = Tally::default();
    let inner = run_inner(
        graph,
        input,
        &objectives(),
        cfg,
        far_deadline(),
        route,
        &stats,
        &mut tally,
    )
    .unwrap_or_else(|why| panic!("the pass must run on the fixture: {why}"));
    // `tally.layers` counts every tightened VALUE, even one whose screen left
    // no neuron crossing, so it cannot tell "CROWN rows were walked" from
    // "nothing happened". Every fixture seed leaves conv1 neurons crossing
    // under the screen (20..211 of 256, computed offline for seeds 11-13, 21,
    // 31, 41, 51), so zero rows here would make every containment check below
    // a check of the one-step screen alone: VOID, not a pass.
    assert!(
        tally.rows > 0,
        "no CROWN row was walked: the tightening never engaged (VOID)"
    );
    (inner, stats)
}

/// Every box contains every sampled activation (to 1e-12 relative: the
/// sample itself is an f64 evaluation), and every objective lower bound is
/// below every sampled objective value.
fn assert_sound(what: &str, plan: &ChainPlan<'_>, boxes: &Boxes, lower: &[f64], pts: &[Vec<f64>]) {
    let objs = objectives();
    for (p, x) in pts.iter().enumerate() {
        let vals = forward_all(plan, x);
        for (v, xs) in vals.iter().enumerate() {
            let view = boxes.view(v);
            assert_eq!(view.len(), xs.len(), "{what}: value {v} arity");
            for (j, &val) in xs.iter().enumerate() {
                let tol = 1e-12 * (1.0 + val.abs());
                assert!(
                    view.lo(j) <= val + tol && val - tol <= view.hi(j),
                    "{what}: sample {p} value {v} elem {j}: {val} outside [{}, {}]",
                    view.lo(j),
                    view.hi(j)
                );
            }
        }
        let y = vals.last().expect("output");
        for (i, obj) in objs.iter().enumerate() {
            let value: f64 = obj.iter().zip(y).map(|(&c, &v)| f64::from(c) * v).sum();
            assert!(lower[i].is_finite(), "{what}: objective {i} not finite");
            assert!(
                lower[i] <= value + 1e-12 * (1.0 + value.abs()),
                "{what}: objective {i} certified lower {} ABOVE sampled value {value} (sample {p})",
                lower[i]
            );
        }
    }
}

#[test]
fn gpu_certified_crown_bound_never_above_sampled_minimum() {
    for (seed, eps) in [(11u64, 0.08f32), (12, 0.2), (13, 0.02)] {
        let mut rng = Rng::new(seed);
        let (h, w) = (8usize, 8usize);
        let graph = tiny_net(&mut rng, h, w);
        let input = input_box(&mut rng, h, w, eps);
        let plan = ChainPlan::build(&graph, Shape3 { c: 3, h, w }).expect("the fixture is a chain");
        let pts = samples(&mut rng, &input, 60);
        let cfg = test_cfg();
        let mut cpu_lower = Vec::new();
        for (name, route, precision) in routes() {
            let cfg = GpuPatchesCrownConfig { precision, ..cfg };
            let (inner, stats) = run_pass(&graph, &input, &cfg, &route);
            if route.is_gpu() {
                assert!(
                    stats.gpu_passes.load(Ordering::SeqCst) > 0,
                    "{name}: the GPU route never engaged (VOID, not a pass)"
                );
                assert_eq!(
                    stats.cpu_fallback.load(Ordering::SeqCst),
                    0,
                    "{name}: no refusal expected"
                );
            }
            assert_sound(
                &format!("seed {seed} eps {eps} {name}"),
                &plan,
                &inner.boxes,
                &inner.lower,
                &pts,
            );
            if name == "cpu" {
                cpu_lower = inner.lower;
            } else if precision == GpuPatchesPrecision::F64 {
                // Same algorithm, reordered f64 sums: equal to far below any margin.
                for (a, b) in inner.lower.iter().zip(&cpu_lower) {
                    assert!(
                        (a - b).abs() <= 1e-9 * (1.0 + b.abs()),
                        "{name}: {a} vs cpu {b}"
                    );
                }
            }
        }
    }
}

#[test]
fn truncated_depth_is_sound() {
    let mut rng = Rng::new(21);
    let (h, w) = (8usize, 8usize);
    let graph = tiny_net(&mut rng, h, w);
    let input = input_box(&mut rng, h, w, 0.1);
    let plan = ChainPlan::build(&graph, Shape3 { c: 3, h, w }).expect("chain");
    let pts = samples(&mut rng, &input, 40);
    for depth in [1usize, 2, 3] {
        for screen in [true, false] {
            let cfg = GpuPatchesCrownConfig {
                depth,
                screen,
                ..test_cfg()
            };
            let (inner, _) = run_pass(&graph, &input, &cfg, &Route::Cpu);
            assert_sound(
                &format!("depth {depth} screen {screen}"),
                &plan,
                &inner.boxes,
                &inner.lower,
                &pts,
            );
        }
    }
}

fn assert_bit_identical(what: &str, a: &Inner, b: &Inner, values: usize) {
    assert_eq!(a.lower.len(), b.lower.len(), "{what}: objective count");
    for (x, y) in a.lower.iter().zip(&b.lower) {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: objective bound differs");
    }
    for v in 0..values {
        let (va, vb) = (a.boxes.view(v), b.boxes.view(v));
        assert_eq!(va.len(), vb.len(), "{what}: value {v} arity");
        for j in 0..va.len() {
            assert_eq!(
                va.lo(j).to_bits(),
                vb.lo(j).to_bits(),
                "{what}: value {v} lo[{j}]"
            );
            assert_eq!(
                va.hi(j).to_bits(),
                vb.hi(j).to_bits(),
                "{what}: value {v} hi[{j}]"
            );
        }
    }
}

#[test]
fn refusal_fallback_is_bit_identical() {
    let mut rng = Rng::new(31);
    let (h, w) = (8usize, 8usize);
    let graph = tiny_net(&mut rng, h, w);
    let input = input_box(&mut rng, h, w, 0.1);
    let values = ChainPlan::build(&graph, Shape3 { c: 3, h, w })
        .expect("chain")
        .ops
        .len()
        + 1;
    for precision in [GpuPatchesPrecision::F64, GpuPatchesPrecision::F32] {
        let cfg = GpuPatchesCrownConfig {
            precision,
            ..test_cfg()
        };
        let (dark, dark_stats) = run_pass(&graph, &input, &cfg, &Route::Cpu);
        assert!(
            dark_stats.cpu_products.load(Ordering::SeqCst) > 0,
            "the dark route computes on the CPU"
        );
        assert_eq!(dark_stats.gpu_passes.load(Ordering::SeqCst), 0);
        for mode in [
            Failure::Error,
            Failure::Deadline,
            Failure::NonFinite,
            Failure::WrongLength,
        ] {
            // The f32 signed route never issues an f64 GEMM for intermediate
            // rows, so an f64-only failure is not a refusal scenario there.
            let combos: Vec<(Option<Failure>, Option<Failure>)> = match precision {
                GpuPatchesPrecision::F64 => vec![
                    (Some(mode), None),
                    (None, Some(mode)),
                    (Some(mode), Some(mode)),
                ],
                GpuPatchesPrecision::F32 => vec![(Some(mode), None), (Some(mode), Some(mode))],
            };
            for &(f32_mode, f64_mode) in &combos {
                let engine = Arc::new(FailingEngine {
                    f32_mode,
                    f64_mode,
                    calls: AtomicUsize::new(0),
                });
                let route = Route::Engine(engine.clone());
                let (fell_back, stats) = run_pass(&graph, &input, &cfg, &route);
                let what = format!("{precision:?} f32={f32_mode:?} f64={f64_mode:?}");
                assert!(
                    engine.calls.load(Ordering::SeqCst) > 0,
                    "{what}: engine never consulted"
                );
                assert!(
                    stats.cpu_fallback.load(Ordering::SeqCst) > 0,
                    "{what}: no counted fallback"
                );
                assert_eq!(
                    stats.gpu_passes.load(Ordering::SeqCst),
                    0,
                    "{what}: a failing product was published"
                );
                assert_bit_identical(&what, &dark, &fell_back, values);
            }
        }
    }
}

#[test]
fn chunking_is_bit_identical() {
    let mut rng = Rng::new(41);
    let (h, w) = (8usize, 8usize);
    let graph = tiny_net(&mut rng, h, w);
    let input = input_box(&mut rng, h, w, 0.12);
    let values = ChainPlan::build(&graph, Shape3 { c: 3, h, w })
        .expect("chain")
        .ops
        .len()
        + 1;
    let one = GpuPatchesCrownConfig {
        chunk_rows: 1,
        ..test_cfg()
    };
    let many = GpuPatchesCrownConfig {
        chunk_rows: 4096,
        ..test_cfg()
    };
    let (a, _) = run_pass(&graph, &input, &one, &Route::Cpu);
    let (b, _) = run_pass(&graph, &input, &many, &Route::Cpu);
    assert_bit_identical("chunk 1 vs 4096", &a, &b, values);
}

// ---------------------------------------------------------------------------
//  Dark default, config, observable, admission
// ---------------------------------------------------------------------------

#[test]
fn gpu_patches_crown_is_dark_by_default() {
    assert_eq!(BetaCrownConfig::default().gpu_patches_crown, None);
    let absent: BetaCrownConfig =
        serde_json::from_value(serde_json::json!({})).expect("empty config");
    assert_eq!(absent.gpu_patches_crown, None);
    let armed: BetaCrownConfig =
        serde_json::from_value(serde_json::json!({ "gpu_patches_crown": {} })).expect("armed");
    assert_eq!(
        armed.gpu_patches_crown,
        Some(GpuPatchesCrownConfig::default())
    );
    let explicit: BetaCrownConfig = serde_json::from_value(serde_json::json!({
        "gpu_patches_crown": { "precision": "f32", "depth": 3, "chunk_rows": 64 }
    }))
    .expect("explicit");
    let pass = explicit.gpu_patches_crown.expect("armed");
    assert_eq!(pass.precision, GpuPatchesPrecision::F32);
    assert_eq!((pass.depth, pass.chunk_rows), (3, 64));
    assert!(
        serde_json::from_value::<BetaCrownConfig>(serde_json::json!({
            "gpu_patches_crown": { "chunk_row": 64 }
        }))
        .is_err(),
        "a misspelt sub-key must be refused, not defaulted"
    );
    // The default arming is itself admissible.
    assert_eq!(GpuPatchesCrownConfig::default().validate(), Ok(()));
    assert_eq!(
        GpuPatchesCrownConfig::default().precision,
        GpuPatchesPrecision::F64
    );
}

#[test]
fn config_validation_refuses_out_of_range_values() {
    let d = GpuPatchesCrownConfig::default();
    for bad in [
        GpuPatchesCrownConfig { chunk_rows: 0, ..d },
        GpuPatchesCrownConfig {
            chunk_rows: GpuPatchesCrownConfig::MAX_CHUNK_ROWS + 1,
            ..d
        },
        GpuPatchesCrownConfig { max_mb: 0, ..d },
        GpuPatchesCrownConfig {
            headroom_frac: 0.0,
            ..d
        },
        GpuPatchesCrownConfig {
            headroom_frac: 1.5,
            ..d
        },
        GpuPatchesCrownConfig {
            headroom_frac: f64::NAN,
            ..d
        },
        GpuPatchesCrownConfig {
            gpu_tile_mb: 0,
            ..d
        },
        GpuPatchesCrownConfig {
            depth: GpuPatchesCrownConfig::MAX_DEPTH + 1,
            ..d
        },
        GpuPatchesCrownConfig {
            budget_frac: 0.0,
            ..d
        },
        GpuPatchesCrownConfig {
            budget_frac: 0.99,
            ..d
        },
        GpuPatchesCrownConfig { max_secs: 0.0, ..d },
        GpuPatchesCrownConfig {
            max_secs: f64::INFINITY,
            ..d
        },
    ] {
        let why = bad.validate().expect_err("must be refused");
        assert!(why.starts_with("gpu_patches_crown."), "{why}");
    }
}

#[test]
fn observable_line_has_fixed_schema() {
    let mut rng = Rng::new(51);
    let graph = tiny_net(&mut rng, 8, 8);
    let input = input_box(&mut rng, 8, 8, 0.05);
    let thresholds = vec![-1.0e9f32, -1.0e9, -1.0e9];
    let out = run_root_with_route(
        &graph,
        &input,
        &objectives(),
        &test_cfg(),
        far_deadline(),
        &Route::Cpu,
    );
    assert!(out.refused.is_none(), "{:?}", out.refused);
    assert!(out.certifies(&thresholds), "a -1e9 threshold must certify");
    assert!(!out.certifies(&[1.0e9f32, 1.0e9, 1.0e9]));
    assert!(
        !out.certifies(&[-1.0e9f32]),
        "arity mismatch never certifies"
    );
    let line = out.observable_line(&thresholds);
    assert!(
        line.starts_with("[gpu-patches-crown] armed=1 layers="),
        "{line}"
    );
    let mut at = 0usize;
    for field in [
        " gpu_passes=",
        " cpu_fallback=",
        " secs=",
        " peak_mb=",
        " route=cpu",
        " backend=none",
        " precision=f64",
        " depth=0",
        " screen=1",
        " rows=",
        " unstable_before=",
        " unstable_after=",
        " gpu_calls=",
        " gpu_secs=",
        " cpu_products=",
        " boxes_mb=",
        " fb_small=",
        " fb_bad_output=",
        " refused=none",
        " verdict=verified",
        " lower=",
        " threshold=",
    ] {
        let found = line[at..]
            .find(field)
            .unwrap_or_else(|| panic!("field `{field}` missing or out of order in {line}"));
        at += found + field.len();
    }
    assert!(
        out.rows > 0 && out.unstable_before > 0,
        "the fixture has crossing neurons to tighten (`layers` alone counts a tightened \
         value even when no row was walked)"
    );
}

/// The CLI's up-front refusals (spec shape, config, objectives) print the SAME
/// fixed-schema line as a run: a parser keyed on the schema must never meet a
/// shorter line from an armed row.
#[test]
fn refused_before_run_keeps_the_fixed_schema() {
    let out = GpuPatchesCrownOutcome::refused_before_run(2, "spec_shape", &test_cfg());
    assert!(
        !out.certifies(&[-1.0e30f32, -1.0e30]),
        "a refusal never certifies"
    );
    assert!(out.lower.iter().all(|v| *v == f64::NEG_INFINITY));
    assert_eq!(out.gpu_passes, 0);
    let line = out.observable_line(&[]);
    let mut at = 0usize;
    for field in [
        "[gpu-patches-crown] armed=1 layers=0",
        " gpu_passes=0",
        " cpu_fallback=0",
        " route=global",
        " backend=none",
        " precision=f64",
        " fb_small=0",
        " fb_bad_output=0",
        " refused=spec_shape",
        " verdict=refused",
        " lower=",
        " threshold=",
    ] {
        let found = line[at..]
            .find(field)
            .unwrap_or_else(|| panic!("field `{field}` missing or out of order in {line}"));
        at += found + field.len();
    }
}

#[test]
fn admission_refuses_sparse_inputs_and_reports_why() {
    let mut rng = Rng::new(61);
    let graph = tiny_net(&mut rng, 8, 8);
    let input = input_box(&mut rng, 8, 8, 0.05);
    let cfg = GpuPatchesCrownConfig {
        min_perturbed: 1_000_000,
        ..test_cfg()
    };
    let out = run_root_with_route(
        &graph,
        &input,
        &objectives(),
        &cfg,
        far_deadline(),
        &Route::Cpu,
    );
    let why = out.refused.clone().expect("admission must refuse");
    assert!(why.starts_with("admission"), "{why}");
    assert!(out.lower.iter().all(|v| *v == f64::NEG_INFINITY));
    assert!(
        !out.certifies(&[-1.0e30f32, -1.0e30, -1.0e30]),
        "a refusal never certifies"
    );
    assert!(out
        .observable_line(&[0.0, 0.0, 0.0])
        .contains(" verdict=refused"));

    let expired = run_root_with_route(
        &graph,
        &input,
        &objectives(),
        &test_cfg(),
        Instant::now(),
        &Route::Cpu,
    );
    assert!(expired.refused.is_some(), "an expired deadline refuses");
}

#[test]
fn plan_refuses_unsupported_graphs() {
    let mut rng = Rng::new(71);
    let conv = Conv2dLayer::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3, 3, 3]), rng.f32s(54, 0.3)).unwrap(),
        None,
        (1, 1),
        (1, 1),
    )
    .unwrap();
    let mut sig = GraphNetwork::new();
    sig.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv.clone())));
    sig.add_node(GraphNode::new(
        "sig",
        Layer::Sigmoid(SigmoidLayer),
        vec!["conv".into()],
    ));
    sig.set_output("sig");
    assert!(ChainPlan::build(&sig, Shape3 { c: 3, h: 4, w: 4 }).is_err());

    let mut padded = GraphNetwork::new();
    padded.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv.clone())));
    padded.add_node(GraphNode::new(
        "pool",
        Layer::MaxPool2d(MaxPool2dLayer::new((2, 2), (2, 2), (1, 1))),
        vec!["conv".into()],
    ));
    padded.set_output("pool");
    assert!(ChainPlan::build(&padded, Shape3 { c: 3, h: 4, w: 4 }).is_err());

    let mut overlap = GraphNetwork::new();
    overlap.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    overlap.add_node(GraphNode::new(
        "pool",
        Layer::MaxPool2d(MaxPool2dLayer::new((3, 3), (1, 1), (0, 0))),
        vec!["conv".into()],
    ));
    overlap.set_output("pool");
    assert!(ChainPlan::build(&overlap, Shape3 { c: 3, h: 4, w: 4 }).is_err());

    // Wrong channel count is refused, never mis-indexed.
    let ok = tiny_net(&mut rng, 8, 8);
    assert!(ChainPlan::build(&ok, Shape3 { c: 4, h: 8, w: 8 }).is_err());
    assert!(ChainPlan::build(&ok, Shape3 { c: 3, h: 8, w: 8 }).is_ok());
}

#[test]
fn chunk_rows_for_budget_respects_every_cap() {
    // budget, fixed, per_row, chunk_rows, want
    assert_eq!(chunk_rows_for_budget(1000, 0, 100, 256, 1000), 10);
    assert_eq!(chunk_rows_for_budget(1000, 500, 100, 256, 1000), 5);
    assert_eq!(chunk_rows_for_budget(1000, 0, 100, 3, 1000), 3);
    assert_eq!(chunk_rows_for_budget(1000, 0, 100, 256, 2), 2);
    assert_eq!(
        chunk_rows_for_budget(1000, 950, 100, 256, 10),
        0,
        "one row does not fit"
    );
    assert_eq!(
        chunk_rows_for_budget(100, 200, 1, 256, 10),
        0,
        "tiles alone exceed the budget"
    );
}
