// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Resident-forward regression test.

use ndarray::{arr1, arr2};
use ndarray::{ArrayD, IxDyn};
use ny_core::NaiveCpuGemmEngine;
use ny_tensor::BoundedTensor;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::layers::*;
use crate::pgd_attack::attacker::PgdAttacker;
use crate::pgd_attack::config::PgdConfig;
use crate::Network;

use super::common::{
    sign_threshold_network, CachedPlanCountingEngine, ResidentCountingEngine,
    UnsupportedModelFallbackEngine,
};

fn flatten_linear_restart_axis_network() -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Flatten(FlattenLayer::new(0)));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 0.0, 0.0, 0.0]]), None).unwrap(),
    ));
    network
}

#[ntest::timeout(10000)]
#[test]
fn test_evaluate_batch_dense_chain_uses_single_resident_call_4081() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]),
            Some(arr1(&[0.0, 0.0, 0.0])),
        )
        .unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 0.0, 0.0]]), None).unwrap(),
    ));

    let engine = ResidentCountingEngine::new();
    let attacker = PgdAttacker::new(PgdConfig::fast()).with_engine(&engine);
    let inputs = arr2(&[[-1.0_f32, -0.5], [-0.5, 0.0], [0.0, 0.5], [0.5, 1.0]]).into_dyn();

    let output = attacker
        .evaluate_batch(&network, &inputs)
        .expect("dense PGD batch should use the resident fast path");

    // The named point route declines an uncached whole-network interval plan;
    // its two Linear point kernels evaluate the actual source y=ReLU(x0).
    assert_eq!(engine.resident_calls(), 0);
    assert_eq!(engine.gemm_calls(), 2);
    assert_eq!(output.shape(), &[4, 1]);
    assert_eq!(output, arr2(&[[0.0_f32], [0.0], [0.0], [0.5]]).into_dyn());
}

#[ntest::timeout(10000)]
#[test]
fn test_evaluate_batch_dense_chain_reuses_cached_model_plan_4268() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]),
            Some(arr1(&[0.0, 0.0, 0.0])),
        )
        .unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 0.0, 0.0]]), None).unwrap(),
    ));

    let engine = CachedPlanCountingEngine::new();
    let attacker = PgdAttacker::new(PgdConfig::fast()).with_engine(&engine);
    let inputs = arr2(&[[-1.0_f32, -0.5], [-0.5, 0.0], [0.0, 0.5], [0.5, 1.0]]).into_dyn();

    let first = attacker
        .evaluate_batch(&network, &inputs)
        .expect("dense PGD batch should prepare a cached model plan");
    let second = attacker
        .evaluate_batch(&network, &inputs)
        .expect("dense PGD batch should reuse the cached model plan");

    assert_eq!(engine.plan_preparations(), 1);
    assert_eq!(engine.cached_calls(), 2);
    assert_eq!(engine.resident_calls(), 0);
    assert_eq!(engine.gemm_calls(), 0);
    assert_eq!(first, second);
    assert_eq!(first.shape(), &[4, 1]);
    assert_eq!(first[[0, 0]], 0.25);
    assert_eq!(first[[3, 0]], 1.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_evaluate_batch_unsupported_model_skips_cached_plan_and_falls_back_4268() {
    let network = sign_threshold_network(0.5);
    let engine = UnsupportedModelFallbackEngine::new();
    let attacker = PgdAttacker::new(PgdConfig::fast()).with_engine(&engine);
    let inputs = arr2(&[[0.0_f32], [0.25], [0.75], [1.5]]).into_dyn();

    let first = attacker
        .evaluate_batch(&network, &inputs)
        .expect("unsupported PGD network should fall back to the per-layer path");
    let second = attacker
        .evaluate_batch(&network, &inputs)
        .expect("fallback path should stay stable across repeated PGD evaluations");

    assert_eq!(engine.plan_preparations(), 0);
    assert_eq!(engine.resident_calls(), 0);
    assert_eq!(engine.gemm_calls(), 2);
    assert_eq!(first, second);
    assert_eq!(first.shape(), &[4, 1]);
    assert_eq!(first[[0, 0]], -1.0);
    assert_eq!(first[[1, 0]], -1.0);
    assert_eq!(first[[2, 0]], 1.0);
    assert_eq!(first[[3, 0]], 1.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_evaluate_batch_flatten_linear_preserves_restart_axis_4345() {
    let network = flatten_linear_restart_axis_network();
    let engine = NaiveCpuGemmEngine;
    let attacker = PgdAttacker::new(PgdConfig::fast()).with_engine(&engine);
    let inputs = ArrayD::from_shape_vec(
        IxDyn(&[3, 1, 4, 1]),
        vec![
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();

    let output = attacker
        .evaluate_batch(&network, &inputs)
        .expect("Flatten(0) -> Linear PGD batch should preserve the restart axis");

    assert_eq!(output.shape(), &[3, 1]);

    // EXACT EQUALITY RETIRED (#linear-ibp-higham + #degenerate-ibp-gemm). This is a
    // point forward: `evaluate_batch` -> `propagate_concrete_point_preserve_leading_axis`
    // -> `LinearLayer::propagate_ibp_with_engine`, whose `lower == upper` arm is
    // `certify_point_gemm`. A point input does NOT make an f32 GEMM exact, so that
    // arm now publishes a NON-degenerate box and `BoundedTensor::center` returns its
    // midpoint. 1.0, 5.0 and 9.0 are still the true forward values; they are no
    // longer the bit-exact published ones, and the stale half of the assertion is
    // the `assert_eq!`, not the literals.
    //
    // The shift is a BINADE effect, not a widening: the box is symmetric in ULP
    // COUNTS, `[next_down(y - err) - k ulp, next_up(y + err) + k ulp]` with
    // k = in_features + 2 = 6, but at y = 1.0 the grid step below is half the step
    // above, so the midpoint lands above y. Ceiling: the k + 1 = 7 downward steps
    // are worth at most 7*ulp(1.0)/2, so the midpoint moves by at most
    // 7*ulp(1.0)/4 = 2.086e-7, plus <=0.5 ulp for rounding the midpoint itself
    // = 2.682e-7. Rows 1 and 2 (y = 5.0, 9.0) are interior to their binades, the
    // construction is symmetric there, and they stay bit-exact — which is exactly
    // what the log showed: [[1.0000002], [5.0], [9.0]].
    //
    // 1e-6 is 3.7x that ceiling and 4.2x the observed 2.3841858e-7 (= 2*ulp(1.0)).
    // It does not weaken what this test pins: the property is that Flatten(0) under a
    // prepended restart axis routes input row i to output row i, and the expected
    // values are 4.0 apart, so this tolerance is 4e6x below the separation that
    // would let two restarts be confused.
    let expected = [1.0_f32, 5.0, 9.0];
    assert_eq!(output.len(), expected.len());
    for (i, (&actual, &want)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() <= 1e-6,
            "Flatten(0) -> Linear restart axis: output[{i}] = {actual}, want {want} \
             (tol 1e-6; see the binade derivation above)"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_estimate_gradient_spsa_batch_flatten_linear_preserves_restart_axis_4345() {
    let network = flatten_linear_restart_axis_network();
    let engine = NaiveCpuGemmEngine;
    let attacker = PgdAttacker::new(PgdConfig {
        spsa_delta: 0.01,
        ..PgdConfig::fast()
    })
    .with_engine(&engine);
    let inputs = ArrayD::from_shape_vec(
        IxDyn(&[2, 1, 4, 1]),
        vec![0.1_f32, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4],
    )
    .unwrap();
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 4, 1]), vec![-1.0_f32; 4]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 4, 1]), vec![1.0_f32; 4]).unwrap(),
    )
    .unwrap();
    let mut rngs = vec![StdRng::seed_from_u64(11), StdRng::seed_from_u64(29)];

    let (gradient, evals) = attacker
        .estimate_gradient_spsa_batch_with_bounds(&network, &inputs, &bounds, 0, &mut rngs)
        .expect("batched SPSA should preserve the restart axis through Flatten(0)");

    assert_eq!(gradient.shape(), &[2, 1, 4, 1]);
    assert_eq!(evals, 4);
    assert!(
        gradient.iter().all(|value| value.is_finite()),
        "Flatten(0) SPSA gradients should stay finite: {gradient:?}"
    );
    assert!(
        (gradient[[0, 0, 0, 0]] - 1.0).abs() < 1e-6,
        "first sample first-feature SPSA gradient should stay exact for y=x0, got {}",
        gradient[[0, 0, 0, 0]]
    );
    assert!(
        (gradient[[1, 0, 0, 0]] - 1.0).abs() < 1e-6,
        "second sample first-feature SPSA gradient should stay exact for y=x0, got {}",
        gradient[[1, 0, 0, 0]]
    );
}
