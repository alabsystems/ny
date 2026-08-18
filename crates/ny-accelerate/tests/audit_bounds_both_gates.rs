// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ADVERSARIAL AUDIT — published bounds with BOTH gates armed.
//!
//! Separate binary because `sound_f64_gemm`'s engine slot is a process-global
//! `OnceLock`: this file installs the `(f64=true, f32=true)` engine, so
//! `aw_via_engine`'s abs-sum base `S` is a `cblas_sgemm` result rather than a
//! faer f32 one. `S` feeds the certified radius `γ_k·S` directly, so a SMALLER
//! `S` is a TIGHTER published bound.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ndarray::{Array1, Array2};
use ny_accelerate::AccelerateGemmEngine;
use ny_core::{GemmEngine, NyError, Result};
use ny_propagate::prelude::*;

static ARMED: AtomicBool = AtomicBool::new(false);

struct Toggle(AccelerateGemmEngine);

impl GemmEngine for Toggle {
    fn backend_provenance(&self) -> &'static str {
        "audit-toggle-accelerate-both"
    }
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.0.gemm_f32(m, k, n, a, b)
    }
    fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        if !ARMED.load(Ordering::SeqCst) {
            return Err(NyError::UnsupportedOp("audit: seam disarmed".into()));
        }
        self.0.gemm_f64(m, k, n, a, b)
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

fn net(inn: usize, hid: usize, out: usize, seed: u64) -> Network {
    let mut n = Network::new();
    n.add_layer(Layer::Linear(
        LinearLayer::new(weights(hid, inn, seed + 1), Some(Array1::zeros(hid))).expect("l1"),
    ));
    n.add_layer(Layer::ReLU(ReLULayer));
    n.add_layer(Layer::Linear(
        LinearLayer::new(weights(hid, hid, seed + 2), Some(Array1::zeros(hid))).expect("l2"),
    ));
    n.add_layer(Layer::ReLU(ReLULayer));
    n.add_layer(Layer::Linear(
        LinearLayer::new(weights(out, hid, seed + 3), Some(Array1::zeros(out))).expect("l3"),
    ));
    n
}

fn certify(network: &Network, inn: usize, r: f32, method: PropagationMethod) -> Vec<(f32, f32)> {
    let input = BoundedTensor::new(
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[inn]), -r),
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[inn]), r),
    )
    .expect("input");
    let config = PropagationConfig {
        method,
        ..PropagationConfig::default()
    };
    match Verifier::new(config)
        .certify_network_bounds("audit", network, &input, None)
        .expect("certify")
    {
        BoundCertificationResult::Certified(c) => c
            .output_bounds()
            .lower()
            .iter()
            .zip(c.output_bounds().upper().iter())
            .map(|(l, u)| (*l, *u))
            .collect(),
        BoundCertificationResult::Timeout { .. } => panic!("timeout"),
    }
}

#[test]
fn both_gates_armed_never_tightens_the_published_bound() {
    let eng = AccelerateGemmEngine::new_with_gates(true, true).expect("engine");
    println!("{}", eng.install_summary());
    assert!(
        eng.f32_via_accelerate(),
        "sgemm probe refused — this audit needs the f32 seam live"
    );
    ny_propagate::sound_f64_gemm::set_sound_f64_gemm_engine(
        Arc::new(Toggle(eng)) as Arc<dyn GemmEngine>
    );

    let (mut total, mut same, mut wider, mut tighter, mut reached) = (0usize, 0, 0, 0, 0usize);
    let mut worst = 0.0f64;
    for &(inn, hid, out) in &[(512usize, 512usize, 64usize), (1024, 1024, 100)] {
        let n = net(inn, hid, out, inn as u64);
        for &r in &[0.005f32, 0.05, 0.2] {
            for method in [PropagationMethod::Crown, PropagationMethod::AlphaCrown] {
                ARMED.store(false, Ordering::SeqCst);
                let off = certify(&n, inn, r, method);
                let t1 = ny_accelerate::telemetry();
                ARMED.store(true, Ordering::SeqCst);
                let on = certify(&n, inn, r, method);
                let t2 = ny_accelerate::telemetry();
                if t2.f64_calls > t1.f64_calls {
                    reached += 1;
                }
                assert!(
                    t2.f32_accelerate_calls > t1.f32_accelerate_calls,
                    "the armed run never routed an abs-sum through cblas_sgemm"
                );
                for (&(ul, uu), &(al, au)) in off.iter().zip(on.iter()) {
                    total += 1;
                    if ul.to_bits() == al.to_bits() && uu.to_bits() == au.to_bits() {
                        same += 1;
                    } else if al <= ul && au >= uu {
                        wider += 1;
                    } else {
                        tighter += 1;
                        let d = (f64::from(al) - f64::from(ul)).max(f64::from(uu) - f64::from(au));
                        worst = worst.max(d / (f64::from(uu) - f64::from(ul)).abs().max(1e-30));
                    }
                }
            }
        }
    }
    println!(
        "both gates: {reached} seam-reaching cases, {total} published bounds — \
         bit-identical={same} wider={wider} TIGHTER={tighter} (worst {worst:.3e} of width); \
         verdict-path sgemm calls total = {}",
        ny_accelerate::telemetry().f32_accelerate_calls
    );
    assert!(reached > 0);
    assert_eq!(
        tighter, 0,
        "STOP THE LINE: both gates armed produced a TIGHTER published bound"
    );
}
