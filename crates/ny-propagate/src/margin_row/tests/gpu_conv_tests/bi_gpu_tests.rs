// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #bi-gpu soundness, precision, fallback and dark-default tests.
//!
//! The device is [`PatientEngine`]: the parent module's `ReorderFmaEngine`
//! arithmetic (a DIFFERENT summation order from the CPU kernel: reversed
//! contraction, four interleaved FMA accumulators, a tree combine) that never
//! abandons a call on the per-call wall budget, so a loaded test host cannot
//! turn a routed product into a (sound, but assertion-breaking) deadline
//! fallback. It is reached through the phase's own SCOPED route, so every test
//! exercises the new dispatch path (`GpuConvRoute::Scoped` ->
//! `ScopedRoute::offload` -> `attempt`) and its per-pair accounting. The
//! deadline refusal itself is pinned by the `FailingEngine` fallback test.
//!
//! * `bi_gpu_is_dark_by_default` — declaration, gate, policy, phase state,
//!   engine route and the RSS admission rule.
//! * `bi_gpu_pick_routes_admitted_refused_and_dark_pairs`.
//! * `bi_gpu_scoped_route_counts_every_call_exactly_once` — the scope's tally,
//!   scopes never nest, a scope over the dark route is dark.
//! * `bi_gpu_certified_interval_contains_exact_rational_product` — THE
//!   operator-level containment test through the scoped route, on random
//!   fixture convs (one-hot identity-seed sources and wild sources) and the
//!   REAL cct CNN7 trunk conv shapes: `[lnew - en, lnew + en]` contains the
//!   exact-rational product (never tighter than the true value).
//! * `bi_gpu_routed_prefix_pairs_contain_exact_preactivations` — every routed
//!   prefix pair (dense per-column, dense envelope, `#bi-window` crop) brackets
//!   the EXACT pre-activation at sampled box points, point boxes included (only
//!   certified rounding separates the bound from the truth there), and agrees
//!   with the CPU pair to f64 precision.
//! * `bi_gpu_phase_matches_the_cpu_phase_and_contains_exact_preactivations` —
//!   the routed `tighten_layer` (dense and cropped, with `#interm-alpha`
//!   candidates, per-column and envelope lanes) publishes shrink-only boxes
//!   that agree with the CPU phase to f64 precision and contain the exact
//!   pre-activations; every pair is carried by the device (non-vacuous).
//! * `bi_gpu_joint_refresh_matches_the_cpu_refresh_and_stays_contained`.
//! * `bi_gpu_refusals_fall_back_bit_identically` — every device failure mode
//!   and the RSS admission refusal publish the DARK phase / refresh bit-for-bit.
//! * `bi_gpu_policy_armed_root_build_is_contained_and_off_cuda_bit_identical` —
//!   the real policy route (`gpu: true` -> global route) end to end, under the
//!   TEST-ONLY device admission (the route is retired in production, R2).
//! * `bi_gpu_policy_armed_is_refused_outside_tests_bit_identically` — R2: without
//!   that admission an armed policy builds no phase state, prints the refusal
//!   and publishes the dark root bit for bit.
//! * `bi_gpu_receipt_has_fixed_schema`.

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array2;
use num_rational::BigRational;
use ny_core::GemmEngine;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::margin_row::backward_interm::window_tests::chain_spec;
use crate::margin_row::backward_interm::{refresh_root_intermediates, window, BiState};
use crate::margin_row::bi_gpu::{self, BiGpu, BiGpuScope, BI_GPU_ADMIT_BYTES};
use crate::margin_row::engine::{BackwardEngine, LaneDir, Seed};
use crate::margin_row::errlane::ErrLane;
use crate::margin_row::gpu_conv::{self, ConvLane, GpuConvRoute, ScopedRoute};
use crate::margin_row::net::{TwinNet, TwinOp};
use crate::margin_row::root::{LayerGates, RootGates};
use crate::margin_row::rounding::{certify_up, gamma_n, next_up, RoundMode};
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use crate::margin_row::{BackwardIntermRuntimePolicy, RootPrecisionRuntimePolicy};

use super::super::tiny_spec;
use super::{
    box_points, conv_op, convs, exact_backward_entry, exact_tensors, far_deadline, q,
    resnetish_spec, uniform, wild_matrix, FailingEngine, Failure, ReorderFmaEngine,
};

// ---------------------------------------------------------------------------
//  Fixtures and helpers
// ---------------------------------------------------------------------------

/// An injected phase state over `engine` whose RSS admission always admits
/// (the admission itself is pinned separately, with a fixed probe).
fn injected(scope: BiGpuScope, engine: Arc<dyn GemmEngine>) -> BiGpu {
    BiGpu::with_engine(scope, engine).with_headroom(|| None)
}

/// `ReorderFmaEngine`'s arithmetic without its entry-time deadline check (see
/// the module docs): every call it receives it carries.
#[derive(Default)]
struct PatientEngine {
    inner: ReorderFmaEngine,
}

impl PatientEngine {
    fn calls(&self) -> usize {
        self.inner.calls()
    }
}

impl GemmEngine for PatientEngine {
    fn backend_provenance(&self) -> &'static str {
        "test-patient-reorder-fma-f64"
    }

    fn gemm_f32(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        Err(ny_core::NyError::UnsupportedOp(
            "f64-only test engine".into(),
        ))
    }

    fn gemm_f64_with_deadline(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f64],
        b: &[f64],
        _deadline: Instant,
        max_dispatch_macs: usize,
    ) -> ny_core::Result<Vec<f64>> {
        self.inner
            .gemm_f64_with_deadline(m, k, n, a, b, far_deadline(), max_dispatch_macs)
    }
}

/// The emulated device, and the same engine as a trait object.
fn reorder_engine() -> (Arc<PatientEngine>, Arc<dyn GemmEngine>) {
    let engine = Arc::new(PatientEngine::default());
    let shared: Arc<dyn GemmEngine> = engine.clone();
    (engine, shared)
}

/// `(name, spec, box half-width)`: a croppable conv/affine/relu chain (with a
/// ChannelAffine input normalization), the residual tiny net and a
/// resnet-shaped net (both fall back to the dense pair under `#bi-window`).
fn fixtures() -> Vec<(&'static str, TwinSpec, f64)> {
    let mut rng = StdRng::seed_from_u64(0xB16_0001);
    let chain = chain_spec(&mut rng, 0.6);
    let tiny = tiny_spec(&mut rng, 0.6);
    let resnetish = resnetish_spec(&mut rng, (3, 10, 10), 6, 8);
    vec![
        ("chain_2x8x8", chain, 0.35),
        ("tiny_2x6x6", tiny, 0.3),
        ("resnetish_3x10x10", resnetish, 0.2),
    ]
}

/// The REAL cct CNN7 trunk (`challenging_certified_training_2026`, narrow
/// CNN7; trunk relu widths 65536 / 65536 / 32768 / 32768 / 32768 per
/// scratchpad/vbs/diag6/cct.md): 3x32x32 -> conv 64 -> conv 64 -> conv 128
/// (stride 2) -> conv 128 -> conv 128, with a small head so it compiles.
fn cnn7_real_net(rng: &mut StdRng) -> TwinNet {
    /// 3x3, pad 1 conv `ci -> co` over an `ih x iw` input.
    fn conv(
        rng: &mut StdRng,
        input: usize,
        (co, ci): (usize, usize),
        stride: usize,
        (ih, iw): (usize, usize),
    ) -> TwinOpSpec {
        let (oh, ow) = ((ih - 1) / stride + 1, (iw - 1) / stride + 1);
        conv_op(
            input,
            uniform(rng, co * ci * 9, 1.5 / ((ci * 9) as f64).sqrt()),
            uniform(rng, co, 0.2),
            (co, ci, 3, 3),
            (stride, stride),
            1,
            (ci, ih, iw),
            (co, oh, ow),
        )
    }
    let n_flat = 128 * 16 * 16;
    let ops = vec![
        conv(rng, 0, (64, 3), 1, (32, 32)),    // t1
        TwinOpSpec::Relu { input: 1 },         // t2
        conv(rng, 2, (64, 64), 1, (32, 32)),   // t3
        TwinOpSpec::Relu { input: 3 },         // t4
        conv(rng, 4, (128, 64), 2, (32, 32)),  // t5
        TwinOpSpec::Relu { input: 5 },         // t6
        conv(rng, 6, (128, 128), 1, (16, 16)), // t7
        TwinOpSpec::Relu { input: 7 },         // t8
        conv(rng, 8, (128, 128), 1, (16, 16)), // t9
        TwinOpSpec::Relu { input: 9 },         // t10
        TwinOpSpec::Flatten { input: 10 },     // t11
        TwinOpSpec::Gemm {
            input: 11,
            weight: uniform(rng, 2 * n_flat, 1.0 / (n_flat as f64).sqrt()),
            bias: uniform(rng, 2, 0.1),
            bias_err: Vec::new(),
            weight_rel_err: 0.0,
            shape: (2, n_flat),
        }, // t12
        TwinOpSpec::Relu { input: 12 },        // t13
        TwinOpSpec::Gemm {
            input: 13,
            weight: uniform(rng, 4, 1.0),
            bias: uniform(rng, 2, 0.1),
            bias_err: Vec::new(),
            weight_rel_err: 0.0,
            shape: (2, 2),
        }, // t14
    ];
    TwinNet::compile(&TwinSpec {
        n_in: 3 * 32 * 32,
        ops,
    })
    .expect("the real CNN7 trunk compiles")
}

fn boxed(center: &[f64], rad: f64) -> (Vec<f64>, Vec<f64>) {
    (
        center.iter().map(|c| c - rad).collect(),
        center.iter().map(|c| c + rad).collect(),
    )
}

/// Exact tensors of `spec` at each point.
fn exact_at(spec: &TwinSpec, points: &[Vec<f64>]) -> Vec<Vec<Vec<BigRational>>> {
    points.iter().map(|x| exact_tensors(spec, x)).collect()
}

/// Tensor id of trunk relu op `relu_op`'s pre-activation.
fn pre_tensor(net: &TwinNet, relu_op: usize) -> usize {
    let TwinOp::Relu { input, .. } = &net.ops[relu_op] else {
        panic!("op {relu_op} is not a relu");
    };
    *input
}

/// One dense identity-seeded Lower+Upper prefix pair on `eng`.
fn dense_pair(
    eng: &BackwardEngine<'_>,
    relu_op: usize,
    n_pre: usize,
    cols: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    let mut s = Array2::<f64>::zeros((n_pre, cols.len()));
    for (c, &j) in cols.iter().enumerate() {
        s[[j, c]] = 1.0;
    }
    let seed = Seed { s, e: None };
    let lower = eng
        .run_prefix(&seed, relu_op, LaneDir::Lower)
        .expect("certified lower prefix pass");
    let upper = eng
        .run_prefix(&seed, relu_op, LaneDir::Upper)
        .expect("certified upper prefix pass");
    (eng.concretize_lower(&lower), eng.concretize_upper(&upper))
}

/// `[lb[k], ub[k]]` contains the EXACT pre-activation `z[cols[k]]` at every
/// sample point.
fn assert_contains(
    what: &str,
    lb: &[f64],
    ub: &[f64],
    cols: &[usize],
    exact: &[Vec<Vec<BigRational>>],
    tensor: usize,
    checks: &mut usize,
) {
    assert_eq!(lb.len(), cols.len(), "{what}: lower bound count");
    assert_eq!(ub.len(), cols.len(), "{what}: upper bound count");
    for (p, tensors) in exact.iter().enumerate() {
        let z = &tensors[tensor];
        for (k, &j) in cols.iter().enumerate() {
            let (lo_q, hi_q) = (q(lb[k]), q(ub[k]));
            assert!(
                lo_q <= z[j] && z[j] <= hi_q,
                "CONTAINMENT FALSIFIED: {what}: neuron {j} at point {p}: routed bounds \
                 [{:e}, {:e}] do not enclose the exact pre-activation",
                lb[k],
                ub[k]
            );
            *checks += 1;
        }
    }
}

/// Routed vs CPU agree to f64 precision (both are certified; they differ only
/// by the ulp-level reordering of a device sum).
fn assert_close(what: &str, cpu: &[f64], dev: &[f64]) {
    assert_eq!(cpu.len(), dev.len(), "{what}: length");
    for (k, (&x, &y)) in cpu.iter().zip(dev).enumerate() {
        assert!(
            (x - y).abs() <= 1e-9 * (1.0 + x.abs().max(y.abs())),
            "{what}: entry {k}: routed {y:e} vs cpu {x:e} — the route lost f64 precision"
        );
    }
}

fn assert_bits(what: &str, a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}: entry {k}: {x:e} vs {y:e}"
        );
    }
}

/// A fixed, fully deterministic phase policy (no wall-clock chunk sizing, no
/// budget cut). `gpu` stays false: routed states are INJECTED with
/// `BiState::with_bi_gpu` so the device is the emulated one.
fn phase_policy(window: bool, interm_alpha: usize) -> BackwardIntermRuntimePolicy {
    BackwardIntermRuntimePolicy {
        enabled: true,
        secs: 1_000.0,
        chunk: 16,
        chunk_secs: 0.0,
        topk: 4_096,
        interm_alpha,
        window,
        gpu: false,
        patch: crate::margin_row::BiPatchRuntimePolicy::default(),
    }
}

/// The root's frozen gates of layers `0..li` (the prefix `tighten_layer`
/// consumes), without clip rows.
fn prefix_gates(root: &RootGates, li: usize) -> Vec<LayerGates> {
    root.layers[..li]
        .iter()
        .map(|g| LayerGates {
            op: g.op,
            n: g.n,
            l: g.l.clone(),
            u: g.u.clone(),
            alpha: g.alpha.clone(),
            s: g.s.clone(),
            c: g.c.clone(),
            ms: g.ms.clone(),
            unst: g.unst.clone(),
            clip_rows: None,
        })
        .collect()
}

/// Run the phase on trunk layer `li` of `root` and return its published box.
fn run_layer(st: &mut BiState, net: &TwinNet, root: &RootGates, li: usize) -> (Vec<f64>, Vec<f64>) {
    let mut prefix = prefix_gates(root, li);
    let (mut l, mut u) = (root.layers[li].l.clone(), root.layers[li].u.clone());
    st.tighten_layer(
        net,
        RoundMode::Outward,
        &root.mid,
        &root.rad,
        &root.xabs,
        &root.lo,
        &root.hi,
        &mut prefix,
        root.layers[li].op,
        &mut l,
        &mut u,
    );
    assert_eq!(prefix.len(), li, "the prefix gates must be handed back");
    (l, u)
}

/// Trunk layers past the affine-only layer 0 that have an unstable neuron.
fn working_layers(root: &RootGates) -> Vec<usize> {
    (1..root.layers.len())
        .filter(|&li| {
            let layer = &root.layers[li];
            layer
                .l
                .iter()
                .zip(&layer.u)
                .any(|(&a, &b)| a < 0.0 && b > 0.0)
        })
        .collect()
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

/// DARK BY DEFAULT: the declaration defaults false, the compatibility gate
/// resolves false with no preset and no env, the default policy builds no
/// route, a disabled phase never builds one even when the flag is set, the
/// dark engine helper IS `BackwardEngine::new`, and the RSS admission rule is
/// exactly `headroom / 2 >= BI_GPU_ADMIT_BYTES` (unobservable admits).
#[test]
fn bi_gpu_is_dark_by_default() {
    assert!(matches!(
        ny_levers::decls::margin_row::MARGIN_ROW_BI_GPU.default,
        ny_levers::DefaultSpec::Bool(false)
    ));
    assert!(
        !crate::preset_gates::margin_row_bi_gpu_enabled(),
        "the gate must resolve dark with no preset cell and no environment"
    );
    assert!(!BackwardIntermRuntimePolicy::default().gpu);

    let dark = BiState::new(
        BackwardIntermRuntimePolicy {
            enabled: true,
            ..BackwardIntermRuntimePolicy::default()
        },
        None,
    );
    assert!(dark.bi_gpu_stats().is_none(), "#bi-gpu must ship dark");
    let disabled = BiState::new(
        BackwardIntermRuntimePolicy {
            enabled: false,
            gpu: true,
            ..BackwardIntermRuntimePolicy::default()
        },
        None,
    );
    assert!(
        disabled.bi_gpu_stats().is_none(),
        "a disabled phase must never build a device route"
    );
    // R2: outside the test-only admission the armed policy is REFUSED (the
    // route is retired in production) and builds no state; admitted, it
    // reaches the phase state (global route; the injected tests below replace
    // it with the emulated device).
    let armed_policy = BackwardIntermRuntimePolicy {
        enabled: true,
        gpu: true,
        ..BackwardIntermRuntimePolicy::default()
    };
    assert!(
        BiState::new(armed_policy, None).bi_gpu_stats().is_none(),
        "R2: an armed policy must not build a device route outside the test admission"
    );
    let armed =
        bi_gpu::with_device_certification_admitted_for_test(|| BiState::new(armed_policy, None));
    assert_eq!(
        armed.bi_gpu_stats().map(|s| s.pairs),
        Some(0),
        "the admitted armed policy must build the phase's route"
    );

    // A dark phase run builds no route and never offers a product.
    let mut rng = StdRng::seed_from_u64(0xB16_0010);
    let spec = chain_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let (lo, hi) = (vec![-0.4; 128], vec![0.35; 128]);
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let li = *working_layers(&root)
        .first()
        .expect("the fixture has no unstable neuron past layer 0 (VOID)");
    let mut st = BiState::new(phase_policy(true, 1), None);
    let _ = run_layer(&mut st, &net, &root, li);
    assert!(st.bi_gpu_stats().is_none());
    assert!(
        bi_gpu::engine(&net, &root, None).gpu_conv_route().is_off(),
        "the dark helper must be BackwardEngine::new (dark lane route)"
    );

    assert!(bi_gpu::admits(None), "an unobservable headroom admits");
    assert!(bi_gpu::admits(Some(2 * BI_GPU_ADMIT_BYTES)));
    assert!(!bi_gpu::admits(Some(2 * BI_GPU_ADMIT_BYTES - 1)));
    assert!(!bi_gpu::admits(Some(0)));
}

/// Which engine a pair runs on: the scoped device engine when admitted, the
/// CPU-only engine (route `Off`, even if the lane route were armed) when the
/// armed phase's RSS admission refused, and the phase's own dark engine with
/// no route override when `#bi-gpu` is dark.
#[test]
fn bi_gpu_pick_routes_admitted_refused_and_dark_pairs() {
    let (_, spec, _) = fixtures().remove(0);
    let net = TwinNet::compile(&spec).expect("compile");
    let (lo, hi) = (vec![-0.4; 128], vec![0.35; 128]);
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let (_, shared) = reorder_engine();
    let gpu = injected(BiGpuScope::BackwardInterm, shared);
    let route = gpu.route();
    let off = GpuConvRoute::Off;
    let dark = BackwardEngine::new(&net, &root);
    let routed = bi_gpu::engine(&net, &root, Some(&route));
    let cpu = bi_gpu::engine(&net, &root, Some(&off));

    let (r, e) = bi_gpu::pick(true, Some(&route), Some(&routed), &off, Some(&cpu), &dark);
    assert!(
        r.is_some_and(|r| std::ptr::eq(r, &raw const route)) && std::ptr::eq(e, &raw const routed)
    );
    assert!(
        !e.gpu_conv_route().is_off(),
        "an admitted pair must be routed"
    );

    let (r, e) = bi_gpu::pick(false, Some(&route), Some(&routed), &off, Some(&cpu), &dark);
    assert!(r.is_some_and(|r| std::ptr::eq(r, &raw const off)) && std::ptr::eq(e, &raw const cpu));
    assert!(
        e.gpu_conv_route().is_off(),
        "a refused pair must run on the CPU"
    );

    for admitted in [false, true] {
        let (r, e) = bi_gpu::pick(admitted, None, None, &off, None, &dark);
        assert!(
            r.is_none() && std::ptr::eq(e, &raw const dark),
            "dark must be the dark engine"
        );
    }
}

/// The scope counts every call exactly once, in itself: device calls by lane,
/// MACs, and refusals by reason; a scope built over another scope does not
/// nest (the inner tally stays untouched); a scope over the dark route is dark.
#[test]
fn bi_gpu_scoped_route_counts_every_call_exactly_once() {
    assert!(GpuConvRoute::Scoped(ScopedRoute::new(GpuConvRoute::Off)).is_off());
    let (engine, shared) = reorder_engine();
    let inner = ScopedRoute::new(GpuConvRoute::Engine(shared));
    let outer = ScopedRoute::new(GpuConvRoute::Scoped(inner.clone()));
    let route = GpuConvRoute::Scoped(outer.clone());
    assert!(!route.is_off());

    let (_, spec, _) = fixtures().remove(2);
    let net = TwinNet::compile(&spec).expect("compile");
    let cv = convs(&net)[1];
    let r = 3;
    let n_out = cv.kernel.0 * cv.oshape.1 * cv.oshape.2;
    let n_in = cv.kernel.1 * cv.ishape.1 * cv.ishape.2;
    let src = Array2::from_shape_fn((n_out, r), |(i, c)| ((i * 5 + c) as f64 * 0.37).sin());
    let mut dst = Array2::<f64>::zeros((n_in, r));
    for (abs_w, lane) in [
        (false, ConvLane::Signed),
        (true, ConvLane::Abs),
        (true, ConvLane::EnvRow),
    ] {
        assert!(gpu_conv::offload(
            &route, cv, &src, &mut dst, abs_w, lane, true
        ));
    }
    // A Parity pass always refuses (and is counted as such).
    assert!(!gpu_conv::offload(
        &route,
        cv,
        &src,
        &mut dst,
        false,
        ConvLane::Signed,
        false
    ));
    let t = outer.tally();
    assert_eq!(t.convs, 4);
    assert_eq!(t.gpu, 3);
    assert_eq!((t.gpu_signed, t.gpu_abs, t.gpu_env), (1, 1, 1));
    assert_eq!(t.macs, 3 * gpu_conv::conv_macs(cv, r));
    assert_eq!(t.fallback(), 1);
    assert_eq!(t.refused(gpu_conv::Refusal::Parity), 1);
    assert_eq!(
        inner.tally(),
        gpu_conv::TallySnapshot::default(),
        "scopes never nest: the inner scope must count nothing"
    );
    assert_eq!(engine.calls(), 3);
}

/// THE OPERATOR-LEVEL CONTAINMENT TEST, through the scoped route. For every
/// conv — random fixtures (one-hot identity-seed sources and wild sources, R in
/// {1, 3}) AND the real cct CNN7 trunk (wild sources, R = 1, 48 sampled entries
/// per product) — the engine's own charges evaluated on the ROUTED products,
/// `lnew = dev(W^T L)`, `en = certify_up(dev(|W|^T (E + g(|L| + E))),
/// gamma(k_bwd + 8))`, give `[lnew - en, lnew + en]` containing the
/// EXACT-RATIONAL product for the stored `L` and an exact corner of its error
/// box: the certified radius is never tighter than the true error.
#[test]
fn bi_gpu_certified_interval_contains_exact_rational_product() {
    let mut rng = StdRng::seed_from_u64(0xB16_0003);
    let (engine, shared) = reorder_engine();
    let scope = ScopedRoute::new(GpuConvRoute::Engine(shared));
    let route = GpuConvRoute::Scoped(scope.clone());
    // (name, net, widths, sampled entries per product, one-hot sources too)
    let mut nets: Vec<(&str, TwinNet, Vec<usize>, usize, bool)> = fixtures()
        .into_iter()
        .map(|(name, spec, _)| {
            (
                name,
                TwinNet::compile(&spec).expect("compile"),
                vec![1, 3],
                usize::MAX,
                true,
            )
        })
        .collect();
    nets.push(("cct_cnn7_real", cnn7_real_net(&mut rng), vec![1], 48, false));
    let mut checks = 0usize;
    let mut products = 0u64;
    let mut real_convs = 0usize;
    for (name, net, widths, sample, one_hot_too) in &nets {
        for cv in convs(net) {
            if *name == "cct_cnn7_real" {
                real_convs += 1;
            }
            let n_out = cv.kernel.0 * cv.oshape.1 * cv.oshape.2;
            let n_in = cv.kernel.1 * cv.ishape.1 * cv.ishape.2;
            let g = next_up(
                gamma_n(cv.k_bwd + 2)
                    + cv.weight_rel_err
                    + gamma_n(cv.k_bwd + 2) * cv.weight_rel_err,
            );
            let g2 = gamma_n(cv.k_bwd + 8);
            for &r in widths {
                let kinds: &[bool] = if *one_hot_too {
                    &[true, false]
                } else {
                    &[false]
                };
                for &one_hot in kinds {
                    // One-hot columns are the identity seed a prefix pair starts
                    // from (exact, no error lane); wild ones are a deep pass's.
                    let l = if one_hot {
                        Array2::from_shape_fn((n_out, r), |(i, c)| {
                            if i == (c * 7_919 + 13) % n_out {
                                1.0
                            } else {
                                0.0
                            }
                        })
                    } else {
                        wild_matrix(&mut rng, n_out, r)
                    };
                    let e = Array2::from_shape_fn((n_out, r), |(i, c)| {
                        if one_hot || (i + c) % 3 == 0 {
                            0.0
                        } else {
                            l[[i, c]].abs() * 1e-13 * (1 + (i * 7 + c) % 5) as f64
                        }
                    });
                    let comb = Array2::from_shape_fn((n_out, r), |(i, c)| {
                        let (lv, ev) = (l[[i, c]], e[[i, c]]);
                        ev + g * (lv.abs() + ev)
                    });
                    let mut lnew = Array2::<f64>::zeros((n_in, r));
                    let mut en = Array2::<f64>::zeros((n_in, r));
                    assert!(
                        gpu_conv::offload(&route, cv, &l, &mut lnew, false, ConvLane::Signed, true),
                        "{name}: the scoped route refused a signed product"
                    );
                    assert!(
                        gpu_conv::offload(&route, cv, &comb, &mut en, true, ConvLane::Abs, true),
                        "{name}: the scoped route refused a |W| product"
                    );
                    products += 2;
                    en.mapv_inplace(|v| certify_up(v, g2));

                    let stored: Vec<BigRational> = l.iter().map(|&v| q(v)).collect();
                    let corner: Vec<BigRational> = l
                        .iter()
                        .zip(e.iter())
                        .map(|(&lv, &ev)| {
                            if rng.random_range(0.0..1.0) < 0.5 {
                                q(lv) + q(ev)
                            } else {
                                q(lv) - q(ev)
                            }
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
                        let (mid, rad) = (lnew[[row, col]], en[[row, col]]);
                        let (lo_q, hi_q) = (q(mid) - q(rad), q(mid) + q(rad));
                        for (which, src) in [&stored, &corner].into_iter().enumerate() {
                            let exact = exact_backward_entry(cv, src, r, row, col, false);
                            assert!(
                                lo_q <= exact && exact <= hi_q,
                                "CONTAINMENT FALSIFIED: {name} k_bwd={} r={r} one_hot={one_hot} \
                                 entry ({row},{col}) operand {which}: routed lnew={mid:e} \
                                 en={rad:e} does not enclose the exact product",
                                cv.k_bwd
                            );
                            checks += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(real_convs, 5, "the real CNN7 trunk lost a conv");
    let t = scope.tally();
    assert_eq!(t.gpu, products, "every product must be device-carried");
    assert_eq!(t.fallback(), 0, "the emulated device must never be refused");
    assert_eq!(t.gpu_signed, t.gpu_abs);
    assert!(checks > 3_000, "only {checks} exact checks ran");
    assert!(engine.calls() > 0);
    eprintln!("[bi-gpu-test] operator containment: {checks} exact checks, {products} products");
}

/// PAIR-LEVEL CONTAINMENT. On every fixture and every trunk layer >= 1, at
/// radii {0 (a point box), 1e-6, 1e-3, fixture}, a routed identity-seeded
/// prefix pair — dense per-column, dense ENVELOPE (`err_min_cols = 1`) and, on
/// the croppable chain, the `#bi-window` crop — brackets the EXACT
/// pre-activation at sampled points and agrees with the CPU pair to f64
/// precision. At a point box only certified rounding separates the bounds from
/// the truth, so a missing charge would falsify here first. The crop's shape
/// gate (`window::prefix_shapes`) admits exactly the Add-free chain prefixes:
/// every residual prefix is refused there (not at the crop), and every chain
/// prefix it admits must crop.
#[test]
fn bi_gpu_routed_prefix_pairs_contain_exact_preactivations() {
    let (engine, shared) = reorder_engine();
    let mut gpu = injected(BiGpuScope::BackwardInterm, shared);
    let route = gpu.route();
    let mut rng = StdRng::seed_from_u64(0xB16_0004);
    let mut checks = 0usize;
    let mut cropped = 0usize;
    let mut gated = 0usize;
    for (name, spec, rad0) in fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        let center: Vec<f64> = (0..spec.n_in)
            .map(|_| rng.random_range(-rad0..rad0))
            .collect();
        for rad in [0.0, 1e-6, 1e-3, rad0] {
            let (lo, hi) = boxed(&center, rad);
            let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
            let extra = if rad > 0.0 { 2 } else { 0 };
            let exact = exact_at(&spec, &box_points(&mut rng, &lo, &hi, extra));
            for li in 1..root.layers.len() {
                let (relu_op, n) = (root.layers[li].op, root.layers[li].n);
                let tensor = pre_tensor(&net, relu_op);
                let stride = (n / 24).max(1);
                let cols: Vec<usize> = (0..n).step_by(stride).collect();
                for lane in [ErrLane::PerCol, ErrLane::Envelope] {
                    let what = format!("{name} rad={rad:e} layer={li} {lane:?} dense");
                    let cpu = BackwardEngine::new(&net, &root)
                        .with_err_lane(lane)
                        .with_err_min_cols(1);
                    let dev = bi_gpu::engine(&net, &root, Some(&route))
                        .with_err_lane(lane)
                        .with_err_min_cols(1);
                    let (cl, cu) = dense_pair(&cpu, relu_op, n, &cols);
                    let mark = gpu.open_pair().expect("an injected state admits");
                    let (dl, du) = dense_pair(&dev, relu_op, n, &cols);
                    gpu.close_pair(mark);
                    assert_close(&format!("{what} lower"), &cl, &dl);
                    assert_close(&format!("{what} upper"), &cu, &du);
                    assert_contains(&what, &dl, &du, &cols, &exact, tensor, &mut checks);
                }
                // `prefix_shapes` is the crop's STRUCTURAL gate: a residual
                // prefix (tiny layer 1, resnetish layers 1-3 — at 2 and 3 a
                // downstream conv pins every shape, the relu back-filling the
                // `Add` output) is refused HERE, once per layer, and runs as the
                // dense pair both lanes above have just checked; an Add-free
                // chain prefix must pass it, and then the crop must express it.
                let residual = net.ops[..=relu_op]
                    .iter()
                    .any(|op| matches!(op, TwinOp::Add { .. }));
                let Some(shapes) = window::prefix_shapes(&net, relu_op) else {
                    assert!(
                        residual,
                        "{name} rad={rad:e} layer={li}: the shape gate refused an Add-free chain"
                    );
                    gated += 1;
                    continue;
                };
                assert!(
                    !residual,
                    "{name} rad={rad:e} layer={li}: a residual prefix passed the crop's shape gate"
                );
                let (ordered, ends) = window::tile_order(&cols, shapes[tensor], 8);
                let part = &ordered[..ends[0]];
                let what = format!("{name} rad={rad:e} layer={li} crop");
                let cpu = window::prefix_bounds(&net, &root, relu_op, &shapes, part, None, None)
                    .expect("a chain chunk is expressible")
                    .expect("the cropped CPU pair is certified");
                let mark = gpu.open_pair().expect("an injected state admits");
                let dev =
                    window::prefix_bounds(&net, &root, relu_op, &shapes, part, None, Some(&route))
                        .expect("a chain chunk is expressible")
                        .expect("the cropped routed pair is certified");
                gpu.close_pair(mark);
                assert_close(&format!("{what} lower"), &cpu.lb, &dev.lb);
                assert_close(&format!("{what} upper"), &cpu.ub, &dev.ub);
                assert_contains(&what, &dev.lb, &dev.ub, part, &exact, tensor, &mut checks);
                cropped += 1;
            }
        }
    }
    let stats = gpu.stats();
    assert!(checks > 5_000, "only {checks} exact checks ran");
    assert!(cropped > 0, "the crop was never exercised (VOID)");
    assert!(
        gated > 0,
        "no residual prefix reached the shape gate, so its refusal is VOID"
    );
    assert!(stats.pairs > 0 && stats.tally.gpu > 0, "{stats:?}");
    assert_eq!(
        stats.pairs, stats.pairs_gpu,
        "every routed pair must be wholly device-carried: {stats:?}"
    );
    assert!(engine.calls() > 0);
    eprintln!(
        "[bi-gpu-test] pair containment: {checks} exact checks, {} pairs ({cropped} cropped, \
         {gated} residual prefixes gated), {} device products",
        stats.pairs, stats.tally.gpu
    );
}

/// PHASE PARITY + CONTAINMENT. The routed `tighten_layer` — dense and cropped
/// (`#bi-window`), with one `#interm-alpha` candidate per chunk, per-column and
/// envelope error lanes — publishes, on every working trunk layer of every
/// fixture, a box that is shrink-only against the forward box, agrees with the
/// CPU phase to f64 precision and contains the EXACT pre-activation at sampled
/// points; and every one of its pairs was device-carried (non-vacuous). On a
/// RESIDUAL layer the armed phase is the dark phase bit for bit, refused once
/// at the crop's shape gate (never tile-capped dense chunks behind a crop that
/// every chunk refuses).
#[test]
fn bi_gpu_phase_matches_the_cpu_phase_and_contains_exact_preactivations() {
    let (engine, shared) = reorder_engine();
    let mut rng = StdRng::seed_from_u64(0xB16_0005);
    let mut checks = 0usize;
    let mut routed_pairs = 0usize;
    let mut routed_windowed = 0usize;
    let mut residual_armed = 0usize;
    for (name, spec, rad) in fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        let center: Vec<f64> = (0..spec.n_in)
            .map(|_| rng.random_range(-rad..rad))
            .collect();
        let (lo, hi) = boxed(&center, rad);
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let exact = exact_at(&spec, &box_points(&mut rng, &lo, &hi, 3));
        let layers = working_layers(&root);
        assert!(
            !layers.is_empty(),
            "{name}: no working layer (VOID fixture)"
        );
        // Both error lanes on the croppable chain; the per-column lane (the
        // shipped default) on the residual fixtures, which bounds the runtime.
        let lanes: &[ErrLane] = if name == "chain_2x8x8" {
            &[ErrLane::PerCol, ErrLane::Envelope]
        } else {
            &[ErrLane::PerCol]
        };
        // The dark phase's routed boxes per (lane, layer), for the residual
        // layers' armed arm to be compared against bit for bit.
        let mut dark: std::collections::BTreeMap<(usize, usize), (Vec<f64>, Vec<f64>)> =
            std::collections::BTreeMap::new();
        for window in [false, true] {
            for (lane_ix, &lane) in lanes.iter().enumerate() {
                for &li in &layers {
                    let what = format!("{name} layer={li} window={window} {lane:?}");
                    let mut cpu = BiState::new(phase_policy(window, 1), None).with_err_lane(lane);
                    let mut dev = BiState::new(phase_policy(window, 1), None)
                        .with_err_lane(lane)
                        .with_bi_gpu(Some(injected(BiGpuScope::BackwardInterm, shared.clone())));
                    let (cl, cu) = run_layer(&mut cpu, &net, &root, li);
                    let (dl, du) = run_layer(&mut dev, &net, &root, li);
                    assert_close(&format!("{what} lower"), &cl, &dl);
                    assert_close(&format!("{what} upper"), &cu, &du);
                    let (l0, u0) = (&root.layers[li].l, &root.layers[li].u);
                    for j in 0..dl.len() {
                        assert!(
                            dl[j] >= l0[j] && du[j] <= u0[j],
                            "{what}: the routed phase widened neuron {j}"
                        );
                    }
                    let all: Vec<usize> = (0..dl.len()).collect();
                    let tensor = pre_tensor(&net, root.layers[li].op);
                    assert_contains(&what, &dl, &du, &all, &exact, tensor, &mut checks);
                    let stats = dev.bi_gpu_stats().expect("the injected route");
                    assert!(stats.pairs > 0, "{what}: no pair was routed (VOID)");
                    assert_eq!(
                        stats.pairs, stats.pairs_gpu,
                        "{what}: a routed pair fell back: {stats:?}"
                    );
                    assert_eq!(stats.rss_refused, 0, "{what}: {stats:?}");
                    assert_eq!(
                        dev.window_counts(),
                        cpu.window_counts(),
                        "{what}: the route changed which pairs were cropped"
                    );
                    routed_pairs += stats.pairs;
                    if window {
                        routed_windowed += dev.window_counts().0;
                    }
                    // #bi-window on a RESIDUAL layer: refused once at the shape
                    // gate, so the armed layer IS the dark layer — same column
                    // order, same chunk widths, same passes, bit for bit — and
                    // never a per-chunk crop attempt behind tile-capped chunks.
                    let relu_op = root.layers[li].op;
                    if net.ops[..=relu_op]
                        .iter()
                        .any(|op| matches!(op, TwinOp::Add { .. }))
                    {
                        if window {
                            let (bl, bu) = dark
                                .get(&(lane_ix, li))
                                .expect("the dark arm ran this layer first");
                            assert_bits(&format!("{what} lower vs dark"), bl, &dl);
                            assert_bits(&format!("{what} upper vs dark"), bu, &du);
                            assert_eq!(
                                dev.window_counts(),
                                (0, 1),
                                "{what}: a residual layer must be refused once, at the shape gate"
                            );
                            residual_armed += 1;
                        } else {
                            dark.insert((lane_ix, li), (dl.clone(), du.clone()));
                        }
                    }
                }
            }
        }
    }
    assert!(routed_windowed > 0, "no cropped pair was routed (VOID)");
    assert!(
        residual_armed > 0,
        "no residual layer ran armed, so its dark-identity pin is VOID"
    );
    assert!(checks > 5_000, "only {checks} exact checks ran");
    assert!(engine.calls() > 0);
    eprintln!(
        "[bi-gpu-test] phase parity: {checks} exact checks, {routed_pairs} routed pairs \
         ({routed_windowed} cropped), {} device calls",
        engine.calls()
    );
}

/// `#joint-refresh` on the route: the refresh's routed pairs (dense and
/// cropped) publish boxes that agree with the CPU refresh to f64 precision and
/// contain the exact pre-activations, and every pair was device-carried.
#[test]
fn bi_gpu_joint_refresh_matches_the_cpu_refresh_and_stays_contained() {
    let (engine, shared) = reorder_engine();
    // The `#joint-refresh` suite's own fixture (`rebake.rs` joint_refresh_tests:
    // seed 0x7F_0001, scale 0.6), where the refresh is pinned to tighten.
    let spec = chain_spec(&mut StdRng::seed_from_u64(0x7F_0001), 0.6);
    let mut rng = StdRng::seed_from_u64(0xB16_0006);
    let net = TwinNet::compile(&spec).expect("compile");
    let (lo, hi) = (vec![-0.4; 128], vec![0.35; 128]);
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let exact = exact_at(&spec, &box_points(&mut rng, &lo, &hi, 3));
    let mut checks = 0usize;
    for window in [false, true] {
        let (cpu, cs) = refresh_root_intermediates(&net, &root, 4_096, 16, window, None, None)
            .expect("cpu refresh");
        let mut gpu = injected(BiGpuScope::JointRefresh, shared.clone());
        let (dev, ds) =
            refresh_root_intermediates(&net, &root, 4_096, 16, window, None, Some(&mut gpu))
                .expect("routed refresh");
        assert!(
            ds.tightened > 0,
            "window={window}: the refresh tightened nothing (VOID)"
        );
        assert_eq!(
            (ds.windowed_pairs, ds.dense_pairs),
            (cs.windowed_pairs, cs.dense_pairs),
            "window={window}: the route changed which pairs ran"
        );
        let stats = gpu.stats();
        assert_eq!(
            stats.pairs,
            ds.windowed_pairs + ds.dense_pairs,
            "window={window}: every refresh pair must be routed: {stats:?}"
        );
        assert_eq!(stats.pairs, stats.pairs_gpu, "window={window}: {stats:?}");
        for (li, (lc, ld)) in cpu.layers.iter().zip(&dev.layers).enumerate() {
            let what = format!("joint refresh window={window} layer={li}");
            assert_close(&format!("{what} lower"), &lc.l, &ld.l);
            assert_close(&format!("{what} upper"), &lc.u, &ld.u);
            let all: Vec<usize> = (0..ld.n).collect();
            let tensor = pre_tensor(&net, ld.op);
            assert_contains(&what, &ld.l, &ld.u, &all, &exact, tensor, &mut checks);
        }
    }
    assert!(checks > 1_000, "only {checks} exact checks ran");
    assert!(engine.calls() > 0);
}

/// FAIL-CLOSED. Under every device failure mode (error, deadline, a NaN in the
/// output, a wrong-length output) the routed phase — dense and cropped, with
/// `#interm-alpha` candidates — and the routed joint refresh publish the DARK
/// boxes bit-for-bit (the CPU kernel overwrites every refused product), every
/// pair is counted `pairs_cpu`, and the failing engine was really consulted.
/// An RSS admission refusal runs every pair on the CPU kernel: bit-identical,
/// counted `rss_refused`, and the device is never called.
#[test]
fn bi_gpu_refusals_fall_back_bit_identically() {
    // The `#joint-refresh` suite's own fixture (see the refresh test above).
    let spec = chain_spec(&mut StdRng::seed_from_u64(0x7F_0001), 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let (lo, hi) = (vec![-0.4; 128], vec![0.35; 128]);
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let layers = working_layers(&root);
    assert!(!layers.is_empty(), "no working layer (VOID fixture)");

    let failing_states = [
        Failure::Error,
        Failure::Deadline,
        Failure::NonFinite,
        Failure::WrongLength,
    ]
    .into_iter()
    .map(|mode| {
        let failing = Arc::new(FailingEngine {
            mode,
            calls: AtomicUsize::new(0),
        });
        (format!("{mode:?}"), failing)
    })
    .collect::<Vec<_>>();

    for window in [false, true] {
        for &li in &layers {
            let mut dark = BiState::new(phase_policy(window, 1), None);
            let (bl, bu) = run_layer(&mut dark, &net, &root, li);
            for (mode, failing) in &failing_states {
                let shared: Arc<dyn GemmEngine> = failing.clone();
                let mut routed = BiState::new(phase_policy(window, 1), None)
                    .with_bi_gpu(Some(injected(BiGpuScope::BackwardInterm, shared)));
                let (rl, ru) = run_layer(&mut routed, &net, &root, li);
                let what = format!("{mode} window={window} layer={li}");
                assert_bits(&format!("{what} lower"), &bl, &rl);
                assert_bits(&format!("{what} upper"), &bu, &ru);
                let stats = routed.bi_gpu_stats().expect("the injected route");
                assert!(stats.pairs > 0, "{what}: no pair was routed (VOID)");
                assert_eq!(stats.pairs, stats.pairs_cpu, "{what}: {stats:?}");
                assert_eq!(stats.tally.gpu, 0, "{what}: {stats:?}");
                assert!(stats.tally.fallback() > 0, "{what}: {stats:?}");
            }
            // RSS refusal: the device is never reached.
            let (probe, shared) = reorder_engine();
            let mut refused = BiState::new(phase_policy(window, 1), None).with_bi_gpu(Some(
                BiGpu::with_engine(BiGpuScope::BackwardInterm, shared).with_headroom(|| Some(0)),
            ));
            let (rl, ru) = run_layer(&mut refused, &net, &root, li);
            let what = format!("rss window={window} layer={li}");
            assert_bits(&format!("{what} lower"), &bl, &rl);
            assert_bits(&format!("{what} upper"), &bu, &ru);
            let stats = refused.bi_gpu_stats().expect("the injected route");
            assert!(stats.rss_refused > 0, "{what}: {stats:?}");
            assert_eq!(stats.pairs, 0, "{what}: {stats:?}");
            assert_eq!(
                probe.calls(),
                0,
                "{what}: an RSS-refused pair reached the device"
            );
        }
    }
    for (mode, failing) in &failing_states {
        assert!(
            failing.calls.load(std::sync::atomic::Ordering::SeqCst) > 0,
            "{mode}: the failing engine was never consulted (vacuous)"
        );
    }

    // The joint refresh, dense and cropped, under a failing device.
    for window in [false, true] {
        let (dark, _) = refresh_root_intermediates(&net, &root, 4_096, 16, window, None, None)
            .expect("dark refresh");
        let (_, failing) = &failing_states[0];
        let shared: Arc<dyn GemmEngine> = failing.clone();
        let mut gpu = injected(BiGpuScope::JointRefresh, shared);
        let (routed, _) =
            refresh_root_intermediates(&net, &root, 4_096, 16, window, None, Some(&mut gpu))
                .expect("fallback refresh");
        for (li, (a, b)) in dark.layers.iter().zip(&routed.layers).enumerate() {
            let what = format!("joint refresh window={window} layer={li}");
            assert_bits(&format!("{what} lower"), &a.l, &b.l);
            assert_bits(&format!("{what} upper"), &a.u, &b.u);
            assert_eq!(a.unst, b.unst, "{what}: unstable set");
        }
        let stats = gpu.stats();
        assert!(
            stats.pairs > 0 && stats.pairs == stats.pairs_cpu,
            "{stats:?}"
        );
    }
}

/// THE REAL POLICY ROUTE, end to end: a root build whose frozen policy arms
/// `gpu: true` goes through `BiGpu::armed` (the GLOBAL sound-f64 route, with
/// its full admission: MAC floor, CPU-only scope, backoff, engine slot, CUDA
/// provenance). Its published boxes contain the exact pre-activations, and a
/// refused product runs the CPU kernel.
///
/// Engagement is PROVEN, not inferred from equal outputs (they would match
/// even if the policy flag never reached the phase): the armed build must leave exactly one `backward_interm` receipt
/// whose pairs offered conv products to the global route, and the dark build
/// none. Every product of this small chain is under the global MAC floor, so
/// the receipt must show every one refused `small` — which is what licenses
/// asserting the armed build BIT-IDENTICAL to the dark build on ANY host, CUDA
/// or not.
#[test]
fn bi_gpu_policy_armed_root_build_is_contained_and_off_cuda_bit_identical() {
    // Drain anything an earlier state on this thread recorded.
    let _ = bi_gpu::take_thread_receipts();
    let mut rng = StdRng::seed_from_u64(0xB16_0008);
    let spec = chain_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let (lo, hi) = (vec![-0.4; 128], vec![0.35; 128]);
    let build = |gpu: bool| {
        RootGates::build_retaining_with_policy(
            &net,
            &lo,
            &hi,
            RoundMode::Outward,
            None,
            None,
            &[],
            RootPrecisionRuntimePolicy::default(),
            BackwardIntermRuntimePolicy {
                gpu,
                ..phase_policy(true, 1)
            },
            None,
        )
        .expect("root")
        .0
    };
    let dark = build(false);
    let dark_receipts = bi_gpu::take_thread_receipts();
    assert!(
        dark_receipts.is_empty(),
        "a dark build must build no #bi-gpu state: {dark_receipts:?}"
    );
    // The route is retired in production (R2); the TEST-ONLY admission keeps
    // its end-to-end policy path exercised.
    let armed = bi_gpu::with_device_certification_admitted_for_test(|| build(true));
    let receipts = bi_gpu::take_thread_receipts();
    assert_eq!(
        receipts.len(),
        1,
        "the policy-armed root build must build exactly one phase state: {receipts:?}"
    );
    let (scope, stats) = receipts[0];
    assert_eq!(scope, BiGpuScope::BackwardInterm, "{stats:?}");
    assert!(
        stats.pairs + stats.rss_refused > 0,
        "the policy-armed phase never consulted its route (VOID): {stats:?}"
    );
    if stats.pairs > 0 {
        assert!(
            stats.tally.convs > 0,
            "admitted pairs offered no conv product to the global route (VOID): {stats:?}"
        );
    }
    assert_eq!(
        stats.tally.refused(gpu_conv::Refusal::Small),
        stats.tally.convs,
        "every product of this chain is under the global MAC floor: {stats:?}"
    );
    assert_eq!(stats.tally.gpu, 0, "{stats:?}");
    let exact = exact_at(&spec, &box_points(&mut rng, &lo, &hi, 3));
    let mut checks = 0usize;
    for (ld, la) in dark.layers.iter().zip(&armed.layers) {
        let what = format!("policy-armed layer op={}", la.op);
        assert_close(&format!("{what} lower"), &ld.l, &la.l);
        assert_close(&format!("{what} upper"), &ld.u, &la.u);
        let all: Vec<usize> = (0..la.n).collect();
        assert_contains(
            &what,
            &la.l,
            &la.u,
            &all,
            &exact,
            pre_tensor(&net, la.op),
            &mut checks,
        );
        // Every offered product was refused (receipt above), so each ran the
        // unchanged CPU kernel: bit-identical on every host.
        assert_bits(&format!("{what} lower"), &ld.l, &la.l);
        assert_bits(&format!("{what} upper"), &ld.u, &la.u);
    }
    assert!(checks > 1_000, "only {checks} exact checks ran");
}

/// R2 ("the device never certifies"): `margin_row.bi_gpu` is RETIRED, so an
/// armed policy — the environment is the only route left — builds NO phase
/// state outside the test-only admission: no `[bi-gpu] done` receipt, no
/// device route, and the published root is the DARK root bit for bit. The
/// refusal line is the documented fixed schema.
#[test]
fn bi_gpu_policy_armed_is_refused_outside_tests_bit_identically() {
    let _ = bi_gpu::take_thread_receipts();
    let mut rng = StdRng::seed_from_u64(0xB16_0009);
    let spec = chain_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let (lo, hi) = (vec![-0.4; 128], vec![0.35; 128]);
    let build = |gpu: bool| {
        RootGates::build_retaining_with_policy(
            &net,
            &lo,
            &hi,
            RoundMode::Outward,
            None,
            None,
            &[],
            RootPrecisionRuntimePolicy::default(),
            BackwardIntermRuntimePolicy {
                gpu,
                ..phase_policy(true, 1)
            },
            None,
        )
        .expect("root")
        .0
    };
    let dark = build(false);
    let armed = build(true);
    let receipts = bi_gpu::take_thread_receipts();
    assert!(
        receipts.is_empty(),
        "R2: the refused armed build must build no #bi-gpu state: {receipts:?}"
    );
    assert!(
        armed.backward_interm_execution.columns_completed > 0,
        "VOID: the backward-interm phase never ran, so the refusal protected nothing"
    );
    for (ld, la) in dark.layers.iter().zip(&armed.layers) {
        let what = format!("refused layer op={}", la.op);
        assert_bits(&format!("{what} lower"), &ld.l, &la.l);
        assert_bits(&format!("{what} upper"), &ld.u, &la.u);
    }
    for scope in [BiGpuScope::BackwardInterm, BiGpuScope::JointRefresh] {
        let line = bi_gpu::refused_line(scope);
        assert!(
            line.starts_with(&format!(
                "[bi-gpu] refused reason=r2_device_never_certifies scope={} route=cpu",
                scope.tag()
            )),
            "{line}"
        );
    }
    assert!(
        BiGpu::armed(BiGpuScope::BackwardInterm).is_none(),
        "outside the admission the armed state never exists"
    );
}

/// The receipts carry every field on every line (fixed schema); an armed state
/// that never engaged says `verdict=VOID`, one that did says `engaged`.
#[test]
fn bi_gpu_receipt_has_fixed_schema() {
    let (_, shared) = reorder_engine();
    let mut gpu = injected(BiGpuScope::JointRefresh, shared);
    assert!(
        gpu.armed_line().starts_with(
            "[bi-gpu] armed scope=joint_refresh route=sound_f64_engine backend_req=cuda \
             admit_bytes="
        ),
        "{}",
        gpu.armed_line()
    );
    let idle = gpu.done_line();
    assert!(
        idle.starts_with("[bi-gpu] done armed=1 pairs=0 gpu=0 cpu_fallback=0 secs="),
        "{idle}"
    );
    for field in [
        " scope=joint_refresh ",
        " pairs_gpu=",
        " pairs_mixed=",
        " pairs_cpu=",
        " pairs_noconv=",
        " rss_refused=",
        " convs=",
        " gpu_signed=",
        " gpu_abs=",
        " gpu_env=",
        " gpu_macs=",
        " gpu_secs=",
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
    ] {
        assert!(idle.contains(field), "receipt lacks {field:?}: {idle}");
    }
    assert!(idle.ends_with("verdict=VOID"), "{idle}");

    let (_, spec, _) = fixtures().remove(0);
    let net = TwinNet::compile(&spec).expect("compile");
    let cv = convs(&net)[1];
    let n_out = cv.kernel.0 * cv.oshape.1 * cv.oshape.2;
    let n_in = cv.kernel.1 * cv.ishape.1 * cv.ishape.2;
    let src = Array2::from_shape_fn((n_out, 2), |(i, c)| (i + c) as f64 * 0.25 - 1.0);
    let mut dst = Array2::<f64>::zeros((n_in, 2));
    let mark = gpu.open_pair().expect("an injected state admits");
    assert!(gpu_conv::offload(
        &gpu.route(),
        cv,
        &src,
        &mut dst,
        false,
        ConvLane::Signed,
        true
    ));
    gpu.close_pair(mark);
    let line = gpu.done_line();
    assert!(
        line.starts_with("[bi-gpu] done armed=1 pairs=1 gpu=1 cpu_fallback=0 secs="),
        "{line}"
    );
    assert!(line.contains(" pairs_gpu=1 "), "{line}");
    assert!(line.ends_with("verdict=engaged"), "{line}");
}
