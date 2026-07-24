// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Straight-through-estimator Sign surrogate regression tests
//! (#surrogate-sign, `attack: surrogate_sign_gradient`).
//!
//! The legacy tanh(β·x) smooth relaxation (#3769) saturates to a zero
//! finite-difference once Sign pre-activations leave `[-1, 1]` scale — the
//! traffic_signs BNN regime (QConv pre-activations in the tens to hundreds).
//! The STE surrogate (sign(x) → x during ATTACK gradient probes only) keeps a
//! gradient signal at any scale. Violation checks always use the TRUE Sign
//! forward, so these are attack-quality tests, not soundness tests.
//!
//! CAVEAT on end-to-end negative controls: `attack()` with the default
//! AdamClipping optimizer is NOT gradient-starved by a zero gradient. Its
//! direction is `(m / (sqrt(v) + eps)).signum()` and `(+0.0_f32).signum()`
//! is `+1.0`, so a saturated (all-zero) gradient still steps `+lr` in EVERY
//! dim, deterministically walking each restart onto the upper-bound corner of
//! the input box — and any single-Sign violation region is a half-space that
//! contains a box corner. A multi-restart "standard PGD must fail" assertion
//! is therefore unwinnable by construction on corner-touching pockets. The
//! end-to-end test below removes every non-gradient signal instead (single
//! restart, plain scaled-gradient steps, no stuck-resampling, no dense sweep)
//! so the tanh-vs-STE difference it asserts is purely the gradient mechanism.

use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::layers::*;
use crate::pgd_attack::attacker::PgdAttacker;
use crate::pgd_attack::config::PgdConfig;
use crate::pgd_attack::optimizer::{PgdAlphaMode, PgdOptimizer};
use crate::Network;

/// 1-D BNN-style layer scale: z = 100·x − 90, Sign. The Sign flips at
/// x = 0.9; everywhere else tanh(10·z) is fully saturated in f32.
fn saturated_sign_1d_network() -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[100.0_f32]]), Some(arr1(&[-90.0]))).unwrap(),
    ));
    network.add_layer(Layer::Sign(SignLayer::new()));
    network
}

/// 2-D corner-pocket BNN: out = sign(100·x0 − 99) + sign(100·x1 − 99).
/// out ≥ 1.5 requires BOTH dims > 0.99 — a 0.25%×0.25% pocket of [-1, 1]²
/// touching the (1, 1) corner. Off the flip lines x_i = 0.99 the tanh(10·z)
/// relaxation is fully saturated in f32, so its SPSA finite difference is
/// exactly zero, while the STE linearization 100·(x0 + x1) − 198 has gradient
/// (100, 100) everywhere. NOTE: because the pocket contains a box corner, any
/// schedule that probes corners (dense sweep) or drifts to them on zero
/// gradient (AdamClipping, see module doc) finds it WITHOUT a gradient — the
/// gradient-only config below removes those signals.
fn saturated_sign_pocket_network() -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[[100.0_f32, 0.0], [0.0, 100.0]]),
            Some(arr1(&[-99.0, -99.0])),
        )
        .unwrap(),
    ));
    network.add_layer(Layer::Sign(SignLayer::new()));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32, 1.0]]), None).unwrap(),
    ));
    network
}

fn unit_box_2d() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0_f32, -1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0_f32, 1.0]).unwrap(),
    )
    .unwrap()
}

/// Gradient-only attack config: every signal except the SPSA gradient is
/// removed, so the tanh-vs-STE comparison isolates the surrogate mechanism.
///
/// - `num_restarts: 1`, `parallel: false`: sequential path; the only TRUE-Sign
///   violation check is the final iterate of the single restart — no corner,
///   center, or extra-sample evaluation anywhere in the schedule.
/// - `SignedGradient` + scalar alpha: the update is `x + α·g`, so a zero
///   gradient is a genuine no-op. The default AdamClipping steps
///   `signum(0/eps) = +1` per dim on a zero gradient, drifting onto the
///   corner-touching pocket with no gradient signal at all (module doc).
/// - `restart_when_stuck: false`: a stuck iterate is NOT resampled; frozen
///   means frozen.
/// - `dense_low_dim_sweep: false`: no deterministic grid pre-phase (this box
///   has only 2 varying dims, which would otherwise arm it if enabled).
fn pocket_config(surrogate_sign_gradient: bool) -> PgdConfig {
    PgdConfig {
        num_restarts: 1,
        num_steps: 80,
        parallel: false,
        seed: 42,
        optimizer: PgdOptimizer::SignedGradient,
        alpha_mode: PgdAlphaMode::Scalar(0.05),
        restart_when_stuck: false,
        dense_low_dim_sweep: false,
        surrogate_sign_gradient,
        ..PgdConfig::default()
    }
}

/// At BNN activation scale the tanh smooth relaxation is saturated: the SPSA
/// finite difference is exactly zero, while the STE surrogate recovers the
/// exact linearized gradient (here d out/dx = 100 → SPSA estimate 100·p² = 100).
#[ntest::timeout(10000)]
#[test]
fn test_ste_surrogate_gradient_nonzero_where_tanh_saturates() {
    let network = saturated_sign_1d_network();
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0_f32]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0_f32]).unwrap(),
    )
    .unwrap();
    let x = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0_f32]).unwrap();

    let standard = PgdAttacker::new(pocket_config(false));
    let mut rng = StdRng::seed_from_u64(42);
    let (tanh_grad, _) = standard
        .estimate_gradient_spsa_with_bounds(&network, &x, &bounds, 0, &mut rng)
        .unwrap();
    assert_eq!(
        tanh_grad[[0]],
        0.0,
        "tanh(10·(100x−90)) is saturated at x=0: smooth-Sign SPSA must be exactly zero"
    );

    let ste = PgdAttacker::new(pocket_config(true));
    let mut rng = StdRng::seed_from_u64(42);
    let (ste_grad, evals) = ste
        .estimate_gradient_spsa_with_bounds(&network, &x, &bounds, 0, &mut rng)
        .unwrap();
    assert_eq!(evals, 2, "STE surrogate keeps the 2-eval smooth-Sign path");
    assert!(
        (ste_grad[[0]] - 100.0).abs() < 1e-2,
        "STE surrogate recovers the linearized gradient 100, got {}",
        ste_grad[[0]]
    );
}

/// Gradient-only end-to-end duel from the SAME seed-42 interior start (see
/// `pocket_config` for how every non-gradient signal is removed):
///
/// - tanh smooth Sign: saturated at the start point, so the SPSA gradient is
///   exactly zero, the `x + α·g` update is a no-op, and the single iterate
///   ends where it began — on the all-negative plateau (TRUE-Sign value −2),
///   far from the ≥ 1.5 pocket. The attack must report no counterexample.
/// - STE surrogate: the linearization 100·(x0 + x1) − 198 gives SPSA a
///   constant ascent direction, PGD climbs into the (0.99, 1]² pocket, and
///   the attack loop confirms the violation on the TRUE Sign forward.
///
/// This does NOT claim standard multi-restart PGD misses the pocket — with
/// the default AdamClipping optimizer it provably cannot miss it (zero-grad
/// corner drift, module doc). The claim is the mechanism: gradient signal
/// alone, tanh has none here and STE does.
#[ntest::timeout(60000)]
#[test]
fn test_tanh_zero_gradient_freezes_pgd_ste_surrogate_reaches_sign_pocket() {
    let network = saturated_sign_pocket_network();
    let bounds = unit_box_2d();

    let standard = PgdAttacker::new(pocket_config(false));
    let result = standard
        .attack(&network, &bounds, 0, 1.5, true)
        .expect("standard attack should run");
    assert!(
        !result.found_counterexample,
        "gradient is the only signal in this config and the saturated tanh gradient \
         is zero: standard PGD must not reach the (0.99, 1]² pocket (best={})",
        result.best_output_value
    );
    assert_eq!(
        result.best_output_value, -2.0,
        "zero gradient must freeze the single iterate on the start plateau \
         (TRUE-Sign value −2); movement would show up as −2 → 0 or 2"
    );

    let ste = PgdAttacker::new(pocket_config(true));
    let result = ste
        .attack(&network, &bounds, 0, 1.5, true)
        .expect("STE attack should run");
    assert!(
        result.found_counterexample,
        "STE surrogate gradient should drive PGD into the corner pocket (best={})",
        result.best_output_value
    );
    assert!(
        result.best_output_value >= 1.5,
        "violation must hold on the TRUE Sign forward, got {}",
        result.best_output_value
    );
    let ce = result.counterexample.expect("counterexample present");
    for (i, v) in ce.iter().enumerate() {
        assert!(
            (-1.0..=1.0).contains(v),
            "counterexample dim {i} out of box: {v}"
        );
        assert!(
            *v > 0.99,
            "counterexample dim {i} must sit in the pocket: {v}"
        );
    }
}
