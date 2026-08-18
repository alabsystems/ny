// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ADVERSARIAL AUDIT — the PUBLISHED bound, ARMED vs UNARMED, same process.
//!
//! `sound_f64_gemm`'s engine slot is a `OnceLock`, so a process cannot un-install
//! an engine. This test instead installs a TOGGLE engine: when the toggle is off
//! its `gemm_f64` returns `Err`, which is exactly the signal
//! `crown_single::aw_f64_with_abssum_unbounded` uses to fall through to its faer
//! path — i.e. toggle-off reproduces the UNARMED arithmetic bit-for-bit while
//! holding everything else (allocation, threading, layout) fixed.
//!
//! Reported: the certified output bounds of the same network under both
//! settings, compared in raw f32 bits, plus a containment check of the true
//! network outputs over sampled inputs.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ndarray::{Array1, Array2};
use ny_accelerate::AccelerateGemmEngine;
use ny_core::{GemmEngine, NyError, Result};
use ny_propagate::prelude::*;

static ARMED: AtomicBool = AtomicBool::new(false);
/// Set when the engine under test also has the f32 gate armed. NOTE: with both
/// gates armed the SAME engine object serves `sound_f64_gemm`, so
/// `aw_via_engine`'s abs-sum base `S` — a VERDICT-feeding quantity — is computed
/// by `cblas_sgemm`, not by the "non-verdict IBP/PGD/BaB" traffic the gate is
/// documented to serve.
static F32_GATE: AtomicBool = AtomicBool::new(false);

struct Toggle(AccelerateGemmEngine);

impl GemmEngine for Toggle {
    fn backend_provenance(&self) -> &'static str {
        "audit-toggle-accelerate"
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

/// Standalone process (`--test-threads=1` + a separate binary invocation is not
/// needed: this test uses its OWN engine, installed into the same global slot by
/// whichever test runs first — so it asserts only on what it can observe).
#[test]
fn f32_gate_puts_cblas_sgemm_on_the_verdict_abs_sum() {
    // The claim under audit: "NY_ACCELERATE_F32 = the non-verdict IBP/PGD/BaB
    // free-rider". `aw_via_engine` calls `eng.gemm_f32` for the abs-sum base S
    // whenever k < 2^23, and the CLI installs ONE engine object into BOTH
    // factories — so with both gates armed, S is a cblas_sgemm result.
    let both = AccelerateGemmEngine::new_with_gates(true, true).expect("engine");
    println!("both gates: {}", both.install_summary());
    F32_GATE.store(both.f32_via_accelerate(), Ordering::SeqCst);
    assert!(
        both.f32_via_accelerate(),
        "sgemm probe refused; the rest of this observation does not apply"
    );
    // 100x100x2048 is the shape cifar100 actually dispatched; it is above the
    // engine's 2^15 MAC floor, so `gemm_f32` routes to cblas_sgemm.
    let (m, k, n) = (100usize, 100usize, 2048usize);
    let a = vec![0.5f32; m * k];
    let b = vec![0.25f32; k * n];
    let before = ny_accelerate::telemetry().f32_accelerate_calls;
    let _ = both.gemm_f32(m, k, n, &a, &b).expect("sgemm");
    let after = ny_accelerate::telemetry().f32_accelerate_calls;
    assert!(
        after > before,
        "gemm_f32 did not reach cblas_sgemm at a CROWN-sized shape"
    );
    println!(
        "CONFIRMED: with NY_ACCELERATE_F32=1 the engine installed into sound_f64_gemm answers \
         gemm_f32 from cblas_sgemm — and aw_via_engine uses gemm_f32 for the verdict-feeding \
         abs-sum base S (crown_single.rs `let s32 = eng.gemm_f32(...)`)"
    );
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

fn biases(n: usize, seed: u64) -> Array1<f32> {
    let mut s = seed | 1;
    Array1::from_shape_fn(n, |_| {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (((s >> 40) as f32) / (1u64 << 24) as f32 - 0.5) * 0.1
    })
}

struct Net {
    w: Vec<Array2<f32>>,
    b: Vec<Array1<f32>>,
}

impl Net {
    fn eval(&self, x: &[f32]) -> Vec<f32> {
        let mut v: Vec<f32> = x.to_vec();
        for (li, w) in self.w.iter().enumerate() {
            let mut out = vec![0.0f32; w.nrows()];
            for i in 0..w.nrows() {
                let mut acc = f64::from(self.b[li][i]);
                for (j, item) in v.iter().enumerate() {
                    acc += f64::from(w[[i, j]]) * f64::from(*item);
                }
                out[i] = acc as f32;
            }
            if li + 1 < self.w.len() {
                for o in out.iter_mut() {
                    *o = o.max(0.0);
                }
            }
            v = out;
        }
        v
    }
}

fn build(inn: usize, hid: usize, out: usize) -> (Network, Net) {
    let (w1, b1) = (weights(hid, inn, 1), biases(hid, 11));
    let (w2, b2) = (weights(hid, hid, 2), biases(hid, 22));
    let (w3, b3) = (weights(out, hid, 3), biases(out, 33));
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(w1.clone(), Some(b1.clone())).expect("l1"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(w2.clone(), Some(b2.clone())).expect("l2"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(w3.clone(), Some(b3.clone())).expect("l3"),
    ));
    (
        network,
        Net {
            w: vec![w1, w2, w3],
            b: vec![b1, b2, b3],
        },
    )
}

fn certify(
    network: &Network,
    inn: usize,
    radius: f32,
    method: PropagationMethod,
) -> Vec<(f32, f32)> {
    let input = BoundedTensor::new(
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[inn]), -radius),
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[inn]), radius),
    )
    .expect("input tensor");
    let config = PropagationConfig {
        method,
        ..PropagationConfig::default()
    };
    let result = Verifier::new(config)
        .certify_network_bounds("audit", network, &input, None)
        .expect("certification runs");
    match result {
        BoundCertificationResult::Certified(cert) => {
            let ob = cert.output_bounds();
            ob.lower()
                .iter()
                .zip(ob.upper().iter())
                .map(|(l, u)| (*l, *u))
                .collect()
        }
        BoundCertificationResult::Timeout { .. } => panic!("unexpected timeout"),
    }
}

#[test]
fn published_output_bounds_armed_vs_unarmed() {
    const IN: usize = 512;
    const HID: usize = 512;
    const OUT: usize = 64;
    const RADIUS: f32 = 0.05;

    let eng = AccelerateGemmEngine::new_with_gates(true, false).expect("engine");
    println!("{}", eng.install_summary());
    ny_propagate::sound_f64_gemm::set_sound_f64_gemm_engine(
        Arc::new(Toggle(eng)) as Arc<dyn GemmEngine>
    );

    let (network, plain) = build(IN, HID, OUT);

    for method in [PropagationMethod::Crown, PropagationMethod::AlphaCrown] {
        ARMED.store(false, Ordering::SeqCst);
        let t0 = ny_accelerate::telemetry();
        let unarmed = certify(&network, IN, RADIUS, method);
        let t1 = ny_accelerate::telemetry();
        assert_eq!(
            t1.f64_calls, t0.f64_calls,
            "the disarmed run still reached cblas_dgemm"
        );

        ARMED.store(true, Ordering::SeqCst);
        let armed = certify(&network, IN, RADIUS, method);
        let t2 = ny_accelerate::telemetry();
        println!(
            "{method:?}: dgemm dispatches in the armed run = {}",
            t2.f64_calls - t1.f64_calls
        );
        assert!(
            t2.f64_calls > t1.f64_calls,
            "{method:?}: the armed run never reached the Accelerate seam"
        );

        let mut identical = 0usize;
        let mut wider = 0usize;
        let mut tighter = 0usize;
        let mut worst_tighten_ulps = 0i64;
        let mut worst_tighten_abs = 0.0f64;
        let mut worst_width_ratio: f64 = 1.0;
        for (i, (&(ul, uu), &(al, au))) in unarmed.iter().zip(armed.iter()).enumerate() {
            let same = ul.to_bits() == al.to_bits() && uu.to_bits() == au.to_bits();
            if same {
                identical += 1;
                continue;
            }
            // "wider" = armed lower <= unarmed lower AND armed upper >= unarmed upper.
            if al <= ul && au >= uu {
                wider += 1;
            } else {
                tighter += 1;
                let d_lo = f64::from(al) - f64::from(ul); // > 0 means tighter below
                let d_hi = f64::from(uu) - f64::from(au); // > 0 means tighter above
                let worst = d_lo.max(d_hi);
                if worst > worst_tighten_abs {
                    worst_tighten_abs = worst;
                    worst_tighten_ulps =
                        ulps_between(if d_lo >= d_hi { (ul, al) } else { (uu, au) });
                    println!(
                        "  out[{i}]: unarmed [{ul:e},{uu:e}] armed [{al:e},{au:e}] \
                         (Δlo={d_lo:e} Δhi={d_hi:e})"
                    );
                }
            }
            let wu = f64::from(uu) - f64::from(ul);
            let wa = f64::from(au) - f64::from(al);
            if wu > 0.0 {
                worst_width_ratio = worst_width_ratio.min(wa / wu);
            }
        }
        println!(
            "{method:?}: {} outputs — bit-identical={identical} wider={wider} tighter={tighter} \
             worst tighten={worst_tighten_abs:e} ({worst_tighten_ulps} f32 ulps) \
             min width ratio armed/unarmed={worst_width_ratio:.12}",
            unarmed.len()
        );

        // Soundness of the ARMED bound against the actual network.
        let mut s = 0x1234_5678_9ABC_DEF0u64;
        let mut viol = 0usize;
        for trial in 0..2000 {
            let x: Vec<f32> = (0..IN)
                .map(|_| {
                    s = s
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    match trial % 3 {
                        0 => {
                            if (s >> 33) & 1 == 1 {
                                RADIUS
                            } else {
                                -RADIUS
                            }
                        }
                        _ => ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * RADIUS,
                    }
                })
                .collect();
            for (i, y) in plain.eval(&x).into_iter().enumerate() {
                if y < armed[i].0 || y > armed[i].1 {
                    viol += 1;
                }
            }
        }
        assert_eq!(
            viol, 0,
            "STOP THE LINE: the ARMED published bound excluded a real network output"
        );
        println!("{method:?}: 2000 sampled inputs (incl. box corners) — 0 containment violations");
    }
}

fn ulps_between((a, b): (f32, f32)) -> i64 {
    i64::from(a.to_bits() as i32) - i64::from(b.to_bits() as i32)
}

/// Broad hunt for a TIGHTER published bound: many networks, radii, and both
/// methods. A single tighter output bound is the stop-the-line event.
#[test]
fn hunt_for_a_tighter_published_bound() {
    let eng = AccelerateGemmEngine::new_with_gates(true, false).expect("engine");
    ny_propagate::sound_f64_gemm::set_sound_f64_gemm_engine(
        Arc::new(Toggle(eng)) as Arc<dyn GemmEngine>
    );

    let mut total = 0usize;
    let mut identical = 0usize;
    let mut wider = 0usize;
    let mut tighter = 0usize;
    let mut worst_tighten = 0.0f64;
    let mut cases = 0usize;
    let mut reached = 0usize;

    for &(inn, hid, out) in &[
        (512usize, 512usize, 64usize),
        (600, 700, 100),
        (1024, 1024, 100),
        (256, 2048, 50),
    ] {
        let (network, _) = build_seeded(inn, hid, out, inn as u64 * 7 + hid as u64);
        for &radius in &[0.001f32, 0.01, 0.05, 0.2] {
            for method in [PropagationMethod::Crown, PropagationMethod::AlphaCrown] {
                cases += 1;
                ARMED.store(false, Ordering::SeqCst);
                let unarmed = certify(&network, inn, radius, method);
                let t1 = ny_accelerate::telemetry();
                ARMED.store(true, Ordering::SeqCst);
                let armed = certify(&network, inn, radius, method);
                let t2 = ny_accelerate::telemetry();
                if t2.f64_calls > t1.f64_calls {
                    reached += 1;
                }
                for (&(ul, uu), &(al, au)) in unarmed.iter().zip(armed.iter()) {
                    total += 1;
                    if ul.to_bits() == al.to_bits() && uu.to_bits() == au.to_bits() {
                        identical += 1;
                    } else if al <= ul && au >= uu {
                        wider += 1;
                    } else {
                        tighter += 1;
                        let d = (f64::from(al) - f64::from(ul)).max(f64::from(uu) - f64::from(au));
                        let scale = (f64::from(uu) - f64::from(ul)).abs().max(1e-30);
                        worst_tighten = worst_tighten.max(d / scale);
                    }
                }
            }
        }
    }
    println!(
        "hunt: {cases} (net,radius,method) cases, {reached} reached the seam, {total} published \
         output bounds — bit-identical={identical} wider={wider} TIGHTER={tighter} \
         (worst relative tightening {worst_tighten:.3e} of the bound width)"
    );
    assert!(
        reached > 0,
        "no case reached the seam — the hunt proved nothing"
    );
    assert_eq!(
        tighter, 0,
        "STOP THE LINE: arming the Accelerate seam produced a TIGHTER published bound"
    );
}

fn build_seeded(inn: usize, hid: usize, out: usize, seed: u64) -> (Network, Net) {
    let (w1, b1) = (weights(hid, inn, seed + 1), biases(hid, seed + 11));
    let (w2, b2) = (weights(hid, hid, seed + 2), biases(hid, seed + 22));
    let (w3, b3) = (weights(out, hid, seed + 3), biases(out, seed + 33));
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(w1.clone(), Some(b1.clone())).expect("l1"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(w2.clone(), Some(b2.clone())).expect("l2"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(w3.clone(), Some(b3.clone())).expect("l3"),
    ));
    (
        network,
        Net {
            w: vec![w1, w2, w3],
            b: vec![b1, b2, b3],
        },
    )
}
