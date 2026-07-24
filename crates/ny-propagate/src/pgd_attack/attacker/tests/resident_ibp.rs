// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Resident-forward regression test for #4081.

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

    assert_eq!(engine.resident_calls(), 1);
    assert_eq!(engine.gemm_calls(), 0);
    assert_eq!(output.shape(), &[4, 1]);
    assert_eq!(output[[0, 0]], 0.25);
    assert_eq!(output[[3, 0]], 1.0);
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
    assert_eq!(
        output,
        ArrayD::from_shape_vec(IxDyn(&[3, 1]), vec![1.0_f32, 5.0, 9.0]).unwrap()
    );
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
