// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

// ADVERSARIAL AUDIT — ATTRIBUTION. Arming this seam changes TWO things at once:
//
//   (A) `A·W` is computed by `cblas_dgemm` instead of faer, AND
//   (B) installing ANY engine in `sound_f64_gemm` routes the CROWN backward
//       into `aw_via_engine`, which builds the abs-sum `S` from the CHEAP
//       f32 seam (`crown_single.rs:703`) instead of a second FULL f64 GEMM.
//
// Effect (B) requires no Accelerate at all. This measurement separates them by
// running three arms through one process-global engine slot. It lives behind
// an explicit example rather than masquerading as an ignored correctness test.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Instant;

use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use ndarray::{Array1, Array2};
use ny_accelerate::AccelerateGemmEngine;
use ny_core::{GemmEngine, NyError, Result};
use ny_propagate::prelude::*;

const OFF: u8 = 0;
const FAER: u8 = 1;
const ACCEL: u8 = 2;

static MODE: AtomicU8 = AtomicU8::new(OFF);

struct TriEngine(AccelerateGemmEngine);

fn faer_gemm(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    // Same policy the displaced call site would use.
    let par = if rayon::current_thread_index().is_some() {
        Par::Seq
    } else {
        faer::get_global_parallelism()
    };
    let a = MatRef::from_row_major_slice(a, m, k);
    let b = MatRef::from_row_major_slice(b, k, n);
    let mut out = vec![0.0f64; m * n];
    {
        let dst = MatMut::from_row_major_slice_mut(&mut out, m, n);
        matmul(dst, Accum::Replace, a, b, 1.0, par);
    }
    out
}

impl GemmEngine for TriEngine {
    fn backend_provenance(&self) -> &'static str {
        "audit-attribution"
    }
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.0.gemm_f32(m, k, n, a, b)
    }
    fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        match MODE.load(Ordering::Relaxed) {
            OFF => Err(NyError::UnsupportedOp("audit: engine off".into())),
            FAER => {
                if a.len() != m * k || b.len() != k * n {
                    return Err(NyError::UnsupportedOp("audit: shape".into()));
                }
                Ok(faer_gemm(m, k, n, a, b))
            }
            _ => self.0.gemm_f64(m, k, n, a, b),
        }
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

fn certify_once(network: &Network, inn: usize, method: PropagationMethod) -> (f64, Vec<f32>) {
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
        .certify_network_bounds("attr", network, &input, None)
        .expect("certify");
    let dt = t.elapsed().as_secs_f64();
    let flat: Vec<f32> = match r {
        BoundCertificationResult::Certified(cert) => {
            let ob = cert.output_bounds();
            ob.lower()
                .iter()
                .copied()
                .chain(ob.upper().iter().copied())
                .collect()
        }
        BoundCertificationResult::Timeout { .. } => panic!("unexpected timeout"),
    };
    (dt, flat)
}

fn best_of(n: usize, network: &Network, inn: usize, method: PropagationMethod) -> (f64, Vec<f32>) {
    let mut best = f64::INFINITY;
    let mut bounds = Vec::new();
    for _ in 0..n {
        let (dt, b) = certify_once(network, inn, method);
        best = best.min(dt);
        bounds = b;
    }
    (best, bounds)
}

pub(crate) fn run() {
    let eng = AccelerateGemmEngine::new_with_gates(true, false).expect("engine");
    ny_propagate::sound_f64_gemm::set_sound_f64_gemm_engine(
        Arc::new(TriEngine(eng)) as Arc<dyn GemmEngine>
    );

    println!(
        "{:>22} {:>12} {:>9} {:>9} {:>9} {:>9} {:>10}",
        "network / method", "OFF s", "FAER s", "ACCEL s", "FAER/OFF", "ACC/OFF", "ACC/FAER"
    );
    for &(inn, hid, depth, out) in &[
        (512usize, 512usize, 1usize, 64usize),
        (1024, 1024, 2, 100),
        (2048, 2048, 1, 100),
    ] {
        let net = mlp(inn, hid, depth, out);
        for method in [PropagationMethod::Crown, PropagationMethod::AlphaCrown] {
            MODE.store(OFF, Ordering::Relaxed);
            let (t_off, b_off) = best_of(3, &net, inn, method);
            MODE.store(FAER, Ordering::Relaxed);
            let (t_faer, b_faer) = best_of(3, &net, inn, method);
            MODE.store(ACCEL, Ordering::Relaxed);
            let (t_acc, b_acc) = best_of(3, &net, inn, method);

            let eq_faer = b_off == b_faer;
            let eq_acc = b_off == b_acc;
            println!(
                "{:>22} {:>12.3} {:>9.3} {:>9.3} {:>8.3}x {:>8.3}x {:>9.3}x  \
                 bounds: FAER==OFF {eq_faer}, ACCEL==OFF {eq_acc}",
                format!("{inn}-{hid}x{depth}-{out} {method:?}"),
                t_off,
                t_faer,
                t_acc,
                t_off / t_faer,
                t_off / t_acc,
                t_faer / t_acc,
            );
        }
    }
}
