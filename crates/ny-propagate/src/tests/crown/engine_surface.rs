// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential CROWN engine-threading regressions for public `Network` surfaces.

use super::helpers::{assert_bounds_finite, CountingGemmEngine};
use super::*;
use crate::tests::assert_linear_bounds_close;
use ndarray::{arr1, arr2};
use ny_test_utils::assert_bounded_tensor_close;

fn build_two_linear_relu_network() -> (Network, BoundedTensor) {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[[0.9_f32, -0.4], [0.3, 0.8], [-0.2, 0.6]]),
            Some(arr1(&[0.05_f32, -0.1, 0.2])),
        )
        .expect("valid first Linear layer"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[[0.7_f32, -0.3, 0.5], [-0.4, 0.2, 0.6]]),
            Some(arr1(&[0.0_f32, 0.15])),
        )
        .expect("valid second Linear layer"),
    ));

    let input = BoundedTensor::new(
        arr1(&[-0.75_f32, -0.25]).into_dyn(),
        arr1(&[1.0_f32, 0.8]).into_dyn(),
    )
    .expect("valid sequential CROWN input");

    (network, input)
}

#[ntest::timeout(10000)]
#[test]
fn test_propagate_crown_with_engine_matches_baseline_and_threads_backward_3959() {
    tests::with_crown_dense_budget_mb("2048", || {
        let (network, input) = build_two_linear_relu_network();
        let baseline = network
            .propagate_crown(&input)
            .expect("#3959 baseline sequential CROWN should succeed");

        let collection_engine = CountingGemmEngine::new();
        network
            .collect_crown_ibp_bounds_with_engine_and_deadline(
                &input,
                Some(&collection_engine),
                None,
            )
            .expect("#3959 sequential CROWN-IBP collection should succeed");
        let collection_calls = collection_engine.gemm_calls();
        assert!(
            collection_calls > 0,
            "#3959 regression: sequential CROWN-IBP collection should already exercise GemmEngine"
        );

        let engine = CountingGemmEngine::new();
        let with_engine = network
            .propagate_crown_with_engine(&input, Some(&engine))
            .expect("#3959 engine-aware sequential CROWN should succeed");

        assert_bounds_finite(&with_engine, "propagate_crown_with_engine output");
        assert_bounded_tensor_close(
            &with_engine,
            &baseline,
            1e-5,
            "#3959 sequential propagate_crown_with_engine parity",
        );
        let total_calls = engine.gemm_calls();
        assert!(
            total_calls > collection_calls,
            "#3959 regression: propagate_crown_with_engine should add backward GemmEngine calls \
             beyond CROWN-IBP collection (total={total_calls}, collection={collection_calls})"
        );
    });
}

#[ntest::timeout(10000)]
#[test]
fn test_propagate_crown_batched_with_engine_matches_baseline_3959() {
    tests::with_crown_dense_budget_mb("2048", || {
        let (network, input) = build_two_linear_relu_network();
        let baseline = network
            .propagate_crown_batched(&input)
            .expect("#3959 baseline sequential batched CROWN should succeed");

        let engine = CountingGemmEngine::new();
        let with_engine = network
            .propagate_crown_batched_with_engine(&input, Some(&engine))
            .expect("#3959 engine-aware sequential batched CROWN should succeed");

        assert_bounds_finite(&with_engine, "propagate_crown_batched_with_engine output");
        assert_bounded_tensor_close(
            &with_engine,
            &baseline,
            1e-5,
            "#3959 sequential propagate_crown_batched_with_engine parity",
        );
        let calls = engine.gemm_calls();
        assert!(
            calls > 0,
            "#3959 regression: propagate_crown_batched_with_engine should hit GemmEngine, got {calls} calls"
        );
    });
}

#[ntest::timeout(10000)]
#[test]
fn test_propagate_crown_with_linear_and_engine_matches_baseline_3959() {
    tests::with_crown_dense_budget_mb("2048", || {
        let (network, input) = build_two_linear_relu_network();
        let (baseline_bounds, baseline_linear) = network
            .propagate_crown_with_linear(&input)
            .expect("#3959 baseline fast CROWN with linear should succeed");

        let engine = CountingGemmEngine::new();
        let (with_engine_bounds, with_engine_linear) = network
            .propagate_crown_with_linear_and_engine(&input, Some(&engine))
            .expect("#3959 engine-aware fast CROWN with linear should succeed");

        assert_bounds_finite(
            &with_engine_bounds,
            "propagate_crown_with_linear_and_engine output",
        );
        assert_bounded_tensor_close(
            &with_engine_bounds,
            &baseline_bounds,
            1e-5,
            "#3959 fast CROWN with linear concrete parity",
        );
        assert_linear_bounds_close(
            &with_engine_linear,
            &baseline_linear,
            1e-5,
            "#3959 fast CROWN with linear relaxation parity",
        );
        let calls = engine.gemm_calls();
        assert!(
            calls > 0,
            "#3959 regression: propagate_crown_with_linear_and_engine should hit GemmEngine, got {calls} calls"
        );
    });
}
