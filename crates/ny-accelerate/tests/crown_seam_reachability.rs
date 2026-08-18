// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Does the sound CROWN backward ACTUALLY reach this engine?
//!
//! Every other test in this crate proves the engine is correct. This one proves
//! it is CONNECTED: it installs the engine into the real process-global
//! `ny_propagate::sound_f64_gemm` slot, runs a real CROWN verification through
//! `ny-propagate`'s public API on a network whose `A·W` crosses the seam's
//! `2^24`-MAC dispatch floor, and asserts the engine's own call counter moved.
//!
//! Without this, "the engine is wired" would be an assertion about code I read,
//! not a measurement.
//!
//! Run with `--release`; the network is deliberately large enough to cross the
//! threshold and a debug build spends minutes in the f32 forward pass.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::sync::Arc;

use ndarray::{Array1, Array2};
use ny_accelerate::AccelerateGemmEngine;
use ny_propagate::prelude::*;

/// Deterministic small weights: the point is the SHAPE of the contraction, not
/// the numerics (those are covered by `engine_soundness.rs`).
fn weights(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut s = seed | 1;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (((s >> 40) as f32) / (1u64 << 24) as f32 - 0.5) * 0.05
    })
}

#[test]
fn crown_backward_reaches_the_accelerate_engine_and_stays_sound() {
    // 512 -> 512 -> 512 -> 64. The CROWN backward's second contraction is
    // (64 specs) x (512) x (512) = 16_777_216 MACs, exactly the
    // `SOUND_F64_GEMM_MIN_MACS = 1 << 24` dispatch floor.
    const IN: usize = 512;
    const HID: usize = 512;
    const OUT: usize = 64;

    let engine = AccelerateGemmEngine::new_with_gates(true, false).expect("engine constructs");
    println!("{}", engine.install_summary());
    ny_propagate::sound_f64_gemm::set_sound_f64_gemm_engine(
        Arc::new(engine) as Arc<dyn ny_core::GemmEngine>
    );
    assert!(ny_propagate::sound_f64_gemm::is_installed());

    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(weights(HID, IN, 1), Some(Array1::zeros(HID))).expect("l1"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(weights(HID, HID, 2), Some(Array1::zeros(HID))).expect("l2"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(weights(OUT, HID, 3), Some(Array1::zeros(OUT))).expect("l3"),
    ));

    let spec = VerificationSpec::new(
        (0..IN).map(|_| Bound::new(-0.05, 0.05)).collect(),
        (0..OUT).map(|_| Bound::new(-1e6, 1e6)).collect(),
    )
    .expect("spec");

    let before = ny_accelerate::telemetry();
    let config = PropagationConfig {
        method: PropagationMethod::Crown,
        ..PropagationConfig::default()
    };
    let result = Verifier::new(config)
        .verify(&network, &spec)
        .expect("verification runs");
    let after = ny_accelerate::telemetry();

    println!("verdict: {result:?}");
    println!("engine telemetry before: {before:?}");
    println!("engine telemetry after:  {after:?}");
    assert!(
        after.f64_calls > before.f64_calls,
        "the CROWN backward never reached the Accelerate f64 seam — the engine is \
         correct but NOT connected"
    );
    // Guards must not have fired: this domain is f32-widened, so G2 is
    // unreachable by construction and every shape is well inside the LP64 ABI.
    assert_eq!(
        after.declined_underflow_domain,
        before.declined_underflow_domain
    );
    assert_eq!(after.declined_non_finite, before.declined_non_finite);
    assert_eq!(after.declined_shape, before.declined_shape);
}
