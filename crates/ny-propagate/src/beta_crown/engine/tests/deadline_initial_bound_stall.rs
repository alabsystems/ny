// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the initial-bound-pass deadline (VNN-COMP no-JSON
//! timeout fix).
//!
//! The categories `vggnet16_2022`, `yolo_2023`, `tinyimagenet_2024`, and
//! `soundnessbench` were being OS-killed (exit 124) with no JSON verdict
//! because the large-model *initial* α-CROWN bound pass ran past the
//! `--timeout` deadline before any branch-and-bound domain was explored.
//!
//! Unlike `deadline_explicit_caps.rs` (which passes an already-expired
//! deadline and uses `use_alpha_crown: false`), these tests:
//!   * enable α-CROWN with a high iteration count, so the *initial* bound pass
//!     is genuinely expensive, and
//!   * pass a deadline that is in the FUTURE at the start of `verify` but
//!     expires *while the initial pass is still running*.
//!
//! The contract: `verify` must return a graceful (non-`Verified`) verdict
//! promptly after the deadline — never hang until an OS wall-clock kill.

use std::time::Instant;

use super::prelude::*;

/// Build a deep + wide ReLU MLP whose unbounded α-CROWN initial pass takes a
/// non-trivial amount of wall time (well over the short deadlines used below).
///
/// `width` hidden units per layer, `depth` hidden ReLU blocks. The weights are
/// deterministic (a cheap LCG) so the test is reproducible and the bounds do
/// not trivially verify at the root.
fn deep_wide_relu_mlp(input_dim: usize, width: usize, depth: usize) -> Network {
    // Simple deterministic pseudo-random fill in [-0.5, 0.5).
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / u32::MAX as f32) - 0.5
    };

    let mut network = Network::new();

    // Input projection: input_dim -> width.
    let w_in: Vec<f32> = (0..width * input_dim).map(|_| next()).collect();
    let w_in = Array2::from_shape_vec((width, input_dim), w_in).expect("w_in shape");
    network.add_layer(Layer::Linear(LinearLayer::new(w_in, None).expect("w_in")));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Hidden blocks: width -> width, ReLU.
    for _ in 0..depth {
        let w: Vec<f32> = (0..width * width).map(|_| next()).collect();
        let w = Array2::from_shape_vec((width, width), w).expect("w shape");
        network.add_layer(Layer::Linear(LinearLayer::new(w, None).expect("hidden")));
        network.add_layer(Layer::ReLU(ReLULayer));
    }

    // Output projection: width -> 1.
    let w_out: Vec<f32> = (0..width).map(|_| next()).collect();
    let w_out = Array2::from_shape_vec((1, width), w_out).expect("w_out shape");
    network.add_layer(Layer::Linear(LinearLayer::new(w_out, None).expect("w_out")));

    network
}

fn unit_box_input(dim: usize) -> BoundedTensor {
    BoundedTensor::new(
        Array1::from_elem(dim, -1.0_f32).into_dyn(),
        Array1::from_elem(dim, 1.0_f32).into_dyn(),
    )
    .expect("valid bounded input")
}

/// A large α-CROWN initial pass whose deadline expires while it is still
/// running must abort promptly and return a graceful (non-`Verified`) verdict
/// — not hang. This is the core VNN-COMP exit-124 fix.
#[ntest::timeout(30000)]
#[test]
fn alpha_crown_initial_pass_future_deadline_aborts_promptly() {
    let input_dim = 32;
    let network = deep_wide_relu_mlp(input_dim, 128, 24);
    let input = unit_box_input(input_dim);

    // Generous configured timeout, but a SHORT wall-clock deadline that fires
    // during the initial bound pass. With many α-CROWN iterations + fixed
    // intermediate-bound recomputation, the unbounded initial pass takes well
    // over 250ms on this network.
    let mut config = BetaCrownConfig {
        use_alpha_crown: true,
        use_crown_ibp: true,
        enable_cuts: false,
        enable_pgd_attack: false,
        branching_heuristic: BranchingHeuristic::BoundImpact,
        timeout: Duration::from_mins(5),
        max_domains: 100_000,
        max_depth: 200,
        batch_size: 16,
        ..Default::default()
    };
    config.alpha_config.iterations = 2_000;
    config.alpha_config.fix_interm_bounds = false; // recompute interm bounds → expensive

    // Use a threshold the root cannot trivially verify so we actually enter the
    // expensive α-CROWN pass rather than the IBP/CROWN fast path.
    let threshold = 1.0e9_f32;

    let deadline = Some(Instant::now() + Duration::from_millis(300));
    let start = Instant::now();
    let result = BetaCrownVerifier::new(config)
        .verify_with_engine(&network, &input, threshold, None, deadline)
        .expect("future-deadline α-CROWN verify should return cleanly, not error");
    let elapsed = start.elapsed();

    // Must return WELL before any plausible OS wall-clock kill. The internal
    // deadline was 300ms; allow generous slack for one in-flight α-CROWN
    // iteration / CROWN backward pass to finish, but it must not hang.
    assert!(
        elapsed < Duration::from_secs(12),
        "initial α-CROWN pass must abort promptly after the deadline, took {elapsed:?}"
    );

    // Soundness: aborting on the deadline must NOT report Verified. A graceful
    // Timeout / Unknown / PotentialViolation is the only acceptable outcome.
    assert!(
        !matches!(result.result, BabVerificationStatus::Verified),
        "deadline abort during initial bounds must not claim Verified, got {:?}",
        result.result
    );
}

/// Same expensive network, but the deadline is far in the future: the verifier
/// must NOT abort early (it should make real progress / hit the domain or
/// timeout budget normally). Guards against an over-eager abort that would
/// turn solvable instances into spurious timeouts.
#[ntest::timeout(60000)]
#[test]
fn alpha_crown_initial_pass_distant_deadline_does_not_abort_early() {
    let input_dim = 16;
    let network = deep_wide_relu_mlp(input_dim, 48, 6);
    let input = unit_box_input(input_dim);

    let mut config = BetaCrownConfig {
        use_alpha_crown: true,
        use_crown_ibp: true,
        enable_cuts: false,
        enable_pgd_attack: false,
        branching_heuristic: BranchingHeuristic::BoundImpact,
        timeout: Duration::from_mins(5),
        max_domains: 50,
        max_depth: 10,
        batch_size: 8,
        ..Default::default()
    };
    config.alpha_config.iterations = 20;

    // Threshold low enough that the root verifies easily — exercises the normal
    // (non-aborting) path with a distant deadline.
    let threshold = -1.0e9_f32;
    let deadline = Some(Instant::now() + Duration::from_mins(2));

    let result = BetaCrownVerifier::new(config)
        .verify_with_engine(&network, &input, threshold, None, deadline)
        .expect("distant-deadline verify should return cleanly");

    // Lower bound is hugely above the threshold → must verify, proving the
    // deadline machinery did not abort a trivially-solvable instance.
    assert_eq!(
        result.result,
        BabVerificationStatus::Verified,
        "distant deadline must not prevent a trivially-verifiable root from verifying"
    );
}
