// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

// ADVERSARIAL AUDIT — the SPEEDUP, isolated fold and end-to-end.
//
// This implementation is included by the explicit `audit_speedup` example.
// It is deliberately not a test: wall-clock comparisons are measurements,
// while the default test suite contains only deterministic assertions.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use ndarray::{Array1, Array2};
use ny_accelerate::AccelerateGemmEngine;
use ny_core::{GemmEngine, NyError, Result};
use ny_propagate::prelude::*;

static ARMED: AtomicBool = AtomicBool::new(false);
static SEAM_NANOS: AtomicU64 = AtomicU64::new(0);
static SEAM_MACS: AtomicU64 = AtomicU64::new(0);
static SEAM_CALLS: AtomicU64 = AtomicU64::new(0);
static SHADOW: AtomicBool = AtomicBool::new(false);
static SHADOW_NANOS: AtomicU64 = AtomicU64::new(0);
static IN_RAYON: AtomicU64 = AtomicU64::new(0);
static SHAPES: std::sync::Mutex<Vec<(usize, usize, usize, u64, u64)>> =
    std::sync::Mutex::new(Vec::new());

/// Toggling engine. Disarmed, `gemm_f64` returns `Err`, which is exactly the
/// signal `aw_f64_with_abssum_unbounded` / `conv_group_col_flat_f64` use to fall
/// through to their faer paths — the UNARMED arithmetic, bit for bit.
struct Toggle(AccelerateGemmEngine);

impl GemmEngine for Toggle {
    fn backend_provenance(&self) -> &'static str {
        "audit-toggle-accelerate"
    }
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.0.gemm_f32(m, k, n, a, b)
    }
    fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        if !ARMED.load(Ordering::Relaxed) {
            return Err(NyError::UnsupportedOp("audit: seam disarmed".into()));
        }
        let t = Instant::now();
        let r = self.0.gemm_f64(m, k, n, a, b);
        let dt = t.elapsed().as_nanos() as u64;
        if r.is_ok() {
            SEAM_NANOS.fetch_add(dt, Ordering::Relaxed);
            SEAM_MACS.fetch_add((m * k * n) as u64, Ordering::Relaxed);
            SEAM_CALLS.fetch_add(1, Ordering::Relaxed);
            if SHADOW.load(Ordering::Relaxed) {
                // Time the kernel this dispatch DISPLACED, on the same thread,
                // with the same `current_par()` policy the call site would see.
                let par = if rayon::current_thread_index().is_some() {
                    Par::Seq
                } else {
                    faer::get_global_parallelism()
                };
                if rayon::current_thread_index().is_some() {
                    IN_RAYON.fetch_add(1, Ordering::Relaxed);
                }
                let t2 = Instant::now();
                std::hint::black_box(faer_gemm(m, k, n, a, b, par));
                let dt2 = t2.elapsed().as_nanos() as u64;
                SHADOW_NANOS.fetch_add(dt2, Ordering::Relaxed);
                SHAPES.lock().expect("shapes").push((m, k, n, dt, dt2));
            }
        }
        r
    }
}

fn faer_gemm(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], par: Par) -> Vec<f64> {
    let a = MatRef::from_row_major_slice(a, m, k);
    let b = MatRef::from_row_major_slice(b, k, n);
    let mut out = vec![0.0f64; m * n];
    {
        let dst = MatMut::from_row_major_slice_mut(&mut out, m, n);
        matmul(dst, Accum::Replace, a, b, 1.0, par);
    }
    out
}

fn operands(m: usize, k: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut s = 0x1357_9BDF_0246_8ACEu64;
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        f64::from(((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.5)
    };
    (
        (0..m * k).map(|_| next()).collect(),
        (0..k * n).map(|_| next()).collect(),
    )
}

fn time_it(reps: usize, mut f: impl FnMut()) -> f64 {
    f(); // warm
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() / reps as f64
}

fn isolated_fold_gflops() {
    let eng = AccelerateGemmEngine::new_with_gates(true, false).expect("engine");
    println!("{}", eng.install_summary());
    println!(
        "{:>18} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "m x k x n",
        "acc ms",
        "acc GF/s",
        "faerSeq ms",
        "faerSeq GF",
        "faerPar ms",
        "vs Seq",
        "vs Par"
    );
    // Shapes: the one production actually dispatched (cifar100 α-CROWN:
    // m=100 k=100 n=2048), the crown_seam_reachability shape, the plan's
    // benchmark shapes, and a tall/thin CROWN strip.
    for &(m, k, n) in &[
        (100usize, 100usize, 2048usize),
        (64, 512, 512),
        (512, 64, 512),
        (1024, 128, 1024),
        (2048, 256, 2048),
        (4096, 512, 4096),
        (1, 8192, 8192),
        (99, 512, 512),
    ] {
        let (a, b) = operands(m, k, n);
        let flops = 2.0 * (m * k * n) as f64;
        let reps = if m * k * n > 1 << 26 { 5 } else { 50 };
        let acc = time_it(reps, || {
            std::hint::black_box(eng.gemm_f64(m, k, n, &a, &b).expect("acc"));
        });
        let fs = time_it(reps, || {
            std::hint::black_box(faer_gemm(m, k, n, &a, &b, Par::Seq));
        });
        let fp = time_it(reps, || {
            std::hint::black_box(faer_gemm(m, k, n, &a, &b, faer::get_global_parallelism()));
        });
        println!(
            "{:>18} {:>10.4} {:>10.1} {:>10.4} {:>10.1} {:>10.4} {:>8.2}x {:>8.2}x",
            format!("{m}x{k}x{n}"),
            acc * 1e3,
            flops / acc / 1e9,
            fs * 1e3,
            flops / fs / 1e9,
            fp * 1e3,
            fs / acc,
            fp / acc
        );
    }
}

/// Where does `cblas_dgemm` LOSE to the incumbent? The engine only declines
/// below a MAC floor; it has no aspect-ratio guard, so a thin-`m` CROWN strip
/// (one spec row against a wide contraction) is dispatched at full size.
fn thin_m_regression_sweep() {
    let eng = AccelerateGemmEngine::new_with_gates(true, false).expect("engine");
    println!(
        "{:>18} {:>10} {:>10} {:>10} {:>9} {:>9}",
        "m x k x n", "acc ms", "faerSeq ms", "faerPar ms", "vs Seq", "vs Par"
    );
    for &(m, k, n) in &[
        (1usize, 1024usize, 1024usize),
        (1, 4096, 4096),
        (1, 8192, 8192),
        (2, 4096, 4096),
        (4, 4096, 4096),
        (8, 4096, 4096),
        (16, 4096, 4096),
        (32, 4096, 4096),
        (64, 4096, 4096),
        (128, 4096, 4096),
        (4096, 4096, 1),
        (4096, 4096, 8),
        (4096, 4096, 64),
    ] {
        let (a, b) = operands(m, k, n);
        let reps = if m * k * n > 1 << 26 { 5 } else { 30 };
        let acc = time_it(reps, || {
            std::hint::black_box(eng.gemm_f64(m, k, n, &a, &b).expect("acc"));
        });
        let fs = time_it(reps, || {
            std::hint::black_box(faer_gemm(m, k, n, &a, &b, Par::Seq));
        });
        let fp = time_it(reps, || {
            std::hint::black_box(faer_gemm(m, k, n, &a, &b, faer::get_global_parallelism()));
        });
        let flag = if fs / acc < 1.0 {
            "  <-- SLOWER than the incumbent"
        } else {
            ""
        };
        println!(
            "{:>18} {:>10.4} {:>10.4} {:>10.4} {:>8.2}x {:>8.2}x{flag}",
            format!("{m}x{k}x{n}"),
            acc * 1e3,
            fs * 1e3,
            fp * 1e3,
            fs / acc,
            fp / acc
        );
    }
}

fn weights(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut s = seed | 1;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (((s >> 40) as f32) / (1u64 << 24) as f32 - 0.5) * 0.05
    })
}

fn mlp(inn: usize, hid: usize, depth: usize, out: usize) -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(weights(hid, inn, 1), Some(Array1::zeros(hid))).expect("l"),
    ));
    for d in 0..depth {
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(
            LinearLayer::new(weights(hid, hid, 2 + d as u64), Some(Array1::zeros(hid))).expect("l"),
        ));
    }
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(weights(out, hid, 99), Some(Array1::zeros(out))).expect("l"),
    ));
    network
}

fn certify_once(network: &Network, inn: usize, method: PropagationMethod) -> f64 {
    let input = BoundedTensor::new(
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[inn]), -0.05f32),
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[inn]), 0.05f32),
    )
    .expect("input");
    let config = PropagationConfig {
        method,
        ..PropagationConfig::default()
    };
    let t = Instant::now();
    let r = Verifier::new(config)
        .certify_network_bounds("speed", network, &input, None)
        .expect("certify");
    std::hint::black_box(&r);
    t.elapsed().as_secs_f64()
}

fn end_to_end_propagation_armed_vs_unarmed() {
    let eng = AccelerateGemmEngine::new_with_gates(true, false).expect("engine");
    ny_propagate::sound_f64_gemm::set_sound_f64_gemm_engine(
        Arc::new(Toggle(eng)) as Arc<dyn GemmEngine>
    );

    for &(inn, hid, depth, out) in &[
        (512usize, 512usize, 1usize, 64usize),
        (1024, 1024, 2, 100),
        (2048, 2048, 1, 100),
    ] {
        let net = mlp(inn, hid, depth, out);
        for method in [PropagationMethod::Crown, PropagationMethod::AlphaCrown] {
            ARMED.store(false, Ordering::Relaxed);
            let mut off = f64::INFINITY;
            for _ in 0..3 {
                off = off.min(certify_once(&net, inn, method));
            }

            ARMED.store(true, Ordering::Relaxed);
            SEAM_NANOS.store(0, Ordering::Relaxed);
            SEAM_MACS.store(0, Ordering::Relaxed);
            SEAM_CALLS.store(0, Ordering::Relaxed);
            let mut on = f64::INFINITY;
            for _ in 0..3 {
                on = on.min(certify_once(&net, inn, method));
            }
            let calls = SEAM_CALLS.load(Ordering::Relaxed) / 3;
            let seam_s = SEAM_NANOS.load(Ordering::Relaxed) as f64 / 3e9;
            let macs = SEAM_MACS.load(Ordering::Relaxed) as f64 / 3.0;
            println!(
                "{inn}-{hid}x{depth}-{out} {method:?}: unarmed {:.3}s  armed {:.3}s  \
                 speedup {:.3}x | seam: {calls} dispatches, {:.3}s inside cblas_dgemm \
                 ({:.1}% of armed wall, {:.1} GFLOP/s)",
                off,
                on,
                off / on,
                seam_s,
                100.0 * seam_s / on,
                2.0 * macs / seam_s / 1e9,
            );
        }
    }
}

/// Does the seam ever get dispatched from a RAYON WORKER? That is the only
/// context where the incumbent is `Par::Seq` faer (~59 GFLOP/s), i.e. the only
/// context where `cblas_dgemm`'s single-threaded ~390 GFLOP/s is a real win.
fn beta_crown_dispatch_context() {
    let eng = AccelerateGemmEngine::new_with_gates(true, false).expect("engine");
    ny_propagate::sound_f64_gemm::set_sound_f64_gemm_engine(
        Arc::new(Toggle(eng)) as Arc<dyn GemmEngine>
    );
    ARMED.store(true, Ordering::Relaxed);
    SHADOW.store(true, Ordering::Relaxed);

    let inn = 512usize;
    let net = mlp(inn, 512, 1, 64);
    let spec = VerificationSpec::new(
        (0..inn).map(|_| Bound::new(-0.05, 0.05)).collect(),
        (0..64).map(|_| Bound::new(-1e6, 1e6)).collect(),
    )
    .expect("spec")
    .with_timeout_ms(30_000);

    // Isolate the DEADLINE as the variable: same entry point, same network,
    // timeout None vs Some.
    for timeout in [None, Some(30_000u64)] {
        SEAM_CALLS.store(0, Ordering::Relaxed);
        SHADOW.store(false, Ordering::Relaxed);
        let input = BoundedTensor::new(
            ndarray::ArrayD::from_elem(ndarray::IxDyn(&[inn]), -0.05f32),
            ndarray::ArrayD::from_elem(ndarray::IxDyn(&[inn]), 0.05f32),
        )
        .expect("input");
        let config = PropagationConfig {
            method: PropagationMethod::AlphaCrown,
            ..PropagationConfig::default()
        };
        let _ = Verifier::new(config)
            .certify_network_bounds("deadline", &net, &input, timeout)
            .expect("certify");
        println!(
            "certify_network_bounds AlphaCrown timeout_ms={timeout:?}: {} dgemm dispatches",
            SEAM_CALLS.load(Ordering::Relaxed)
        );
        SHADOW.store(true, Ordering::Relaxed);
    }

    for method in [PropagationMethod::BetaCrown, PropagationMethod::AlphaCrown] {
        SEAM_NANOS.store(0, Ordering::Relaxed);
        SHADOW_NANOS.store(0, Ordering::Relaxed);
        SEAM_CALLS.store(0, Ordering::Relaxed);
        IN_RAYON.store(0, Ordering::Relaxed);
        let config = PropagationConfig {
            method,
            ..PropagationConfig::default()
        };
        let t = Instant::now();
        let r = Verifier::new(config).verify(&net, &spec);
        println!(
            "{method:?}: {:?} in {:.2}s — {} dispatches, {} on rayon workers; dgemm {:.4}s vs \
             displaced faer {:.4}s",
            r.as_ref().map(std::mem::discriminant),
            t.elapsed().as_secs_f64(),
            SEAM_CALLS.load(Ordering::Relaxed),
            IN_RAYON.load(Ordering::Relaxed),
            SEAM_NANOS.load(Ordering::Relaxed) as f64 / 1e9,
            SHADOW_NANOS.load(Ordering::Relaxed) as f64 / 1e9,
        );
    }
}

/// The decisive per-dispatch question: at the shapes PRODUCTION actually sends,
/// is `cblas_dgemm` faster than the faer call it displaced? Measured on the same
/// thread, same operands, same `current_par()` policy the call site would use.
fn per_dispatch_shadow_comparison() {
    let eng = AccelerateGemmEngine::new_with_gates(true, false).expect("engine");
    ny_propagate::sound_f64_gemm::set_sound_f64_gemm_engine(
        Arc::new(Toggle(eng)) as Arc<dyn GemmEngine>
    );
    ARMED.store(true, Ordering::Relaxed);
    SHADOW.store(true, Ordering::Relaxed);

    for &(inn, hid, depth, out) in &[
        (512usize, 512usize, 1usize, 64usize),
        (1024, 1024, 2, 100),
        (2048, 2048, 1, 100),
    ] {
        let net = mlp(inn, hid, depth, out);
        for method in [PropagationMethod::Crown, PropagationMethod::AlphaCrown] {
            SEAM_NANOS.store(0, Ordering::Relaxed);
            SHADOW_NANOS.store(0, Ordering::Relaxed);
            SEAM_CALLS.store(0, Ordering::Relaxed);
            IN_RAYON.store(0, Ordering::Relaxed);
            SHAPES.lock().expect("shapes").clear();
            certify_once(&net, inn, method);
            let acc = SEAM_NANOS.load(Ordering::Relaxed) as f64 / 1e9;
            let fae = SHADOW_NANOS.load(Ordering::Relaxed) as f64 / 1e9;
            let calls = SEAM_CALLS.load(Ordering::Relaxed);
            let in_rayon = IN_RAYON.load(Ordering::Relaxed);
            println!(
                "{inn}-{hid}x{depth}-{out} {method:?}: {calls} dispatches ({in_rayon} on rayon \
                 workers) | cblas_dgemm {acc:.4}s vs displaced faer {fae:.4}s => {:.3}x",
                fae / acc
            );
            let mut by_shape: std::collections::BTreeMap<(usize, usize, usize), (u64, u64, u64)> =
                std::collections::BTreeMap::new();
            for &(m, k, n, a, f) in SHAPES.lock().expect("shapes").iter() {
                let e = by_shape.entry((m, k, n)).or_insert((0, 0, 0));
                e.0 += 1;
                e.1 += a;
                e.2 += f;
            }
            for ((m, k, n), (c, a, f)) in by_shape {
                println!(
                    "    {m}x{k}x{n} x{c}: acc {:.4}ms faer {:.4}ms => {:.2}x",
                    a as f64 / c as f64 / 1e6,
                    f as f64 / c as f64 / 1e6,
                    f as f64 / a as f64
                );
            }
        }
    }
}

pub(crate) fn run() {
    let command = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!(
            "usage: cargo run --release -p ny-accelerate --example audit_speedup -- \
             <isolated|thin|end-to-end|dispatch-context|shadow|all>"
        );
        std::process::exit(2);
    });
    match command.as_str() {
        "isolated" => isolated_fold_gflops(),
        "thin" => thin_m_regression_sweep(),
        "end-to-end" => end_to_end_propagation_armed_vs_unarmed(),
        "dispatch-context" => beta_crown_dispatch_context(),
        "shadow" => per_dispatch_shadow_comparison(),
        "all" => {
            isolated_fold_gflops();
            thin_m_regression_sweep();
            end_to_end_propagation_armed_vs_unarmed();
            beta_crown_dispatch_context();
            per_dispatch_shadow_comparison();
        }
        _ => {
            eprintln!("unknown audit_speedup command: {command}");
            std::process::exit(2);
        }
    }
}
