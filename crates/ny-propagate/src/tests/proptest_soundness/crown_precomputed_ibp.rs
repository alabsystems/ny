// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Property-based verification of pre-computed IBP bounds reuse (#3397).
//!
//! Verifies:
//! 1. CROWN with pre-computed IBP produces identical bounds to CROWN without
//! 2. Pre-computed IBP bounds are sound (contain all concrete outputs)
//! 3. CROWN-IBP tightening invariant: CROWN bounds are at least as tight as IBP

use crate::{Layer, LinearLayer, Network, ReLULayer};
use ndarray::arr1;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;
use std::time::{Duration, Instant};

use super::{sample_points, valid_interval, FP_TOLERANCE};

fn assert_tensor_matches(lhs: &BoundedTensor, rhs: &BoundedTensor, label: &str) {
    assert_eq!(
        lhs.shape(),
        rhs.shape(),
        "{label}: shape mismatch: {:?} vs {:?}",
        lhs.shape(),
        rhs.shape()
    );

    for (idx, (left, right)) in lhs.lower().iter().zip(rhs.lower().iter()).enumerate() {
        assert!(
            (left - right).abs() < FP_TOLERANCE,
            "{label}: lower[{idx}] mismatch: lhs={left}, rhs={right}"
        );
    }

    for (idx, (left, right)) in lhs.upper().iter().zip(rhs.upper().iter()).enumerate() {
        assert!(
            (left - right).abs() < FP_TOLERANCE,
            "{label}: upper[{idx}] mismatch: lhs={left}, rhs={right}"
        );
    }
}

fn build_elapsed_deadline_output_network_3397() -> (Network, BoundedTensor) {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            ndarray::arr2(&[[1.0, -0.5], [0.25, 2.0]]),
            Some(arr1(&[0.1, -0.2])),
        )
        .unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(ndarray::arr2(&[[0.75, -1.25]]), Some(arr1(&[0.3]))).unwrap(),
    ));

    let input =
        BoundedTensor::new(arr1(&[-1.0, -0.5]).into_dyn(), arr1(&[1.5, 2.0]).into_dyn()).unwrap();
    (network, input)
}

fn build_elapsed_deadline_crown_ibp_network_3397() -> (Network, BoundedTensor) {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            ndarray::arr2(&[[1.0, 0.5], [-0.5, 1.0], [0.3, -0.7]]),
            Some(arr1(&[0.1, -0.1, 0.0])),
        )
        .unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(ndarray::arr2(&[[0.5, -0.3, 0.8]]), Some(arr1(&[0.0]))).unwrap(),
    ));

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    (network, input)
}

fn assert_deadline_layer_bounds_match(
    plain_deadline_bounds: &[BoundedTensor],
    precomputed_deadline_bounds: &[BoundedTensor],
    ibp_layer_bounds: &[BoundedTensor],
) {
    assert_eq!(
        plain_deadline_bounds.len(),
        ibp_layer_bounds.len(),
        "plain deadline bounds length should match IBP layer count"
    );
    assert_eq!(
        precomputed_deadline_bounds.len(),
        ibp_layer_bounds.len(),
        "precomputed deadline bounds length should match IBP layer count"
    );

    for (layer_idx, ((plain_layer, precomputed_layer), ibp_layer)) in plain_deadline_bounds
        .iter()
        .zip(precomputed_deadline_bounds.iter())
        .zip(ibp_layer_bounds.iter())
        .enumerate()
    {
        assert_tensor_matches(
            plain_layer,
            ibp_layer,
            &format!("plain deadline layer {layer_idx} vs IBP"),
        );
        assert_tensor_matches(
            precomputed_layer,
            ibp_layer,
            &format!("precomputed deadline layer {layer_idx} vs IBP"),
        );
        assert_tensor_matches(
            precomputed_layer,
            plain_layer,
            &format!("precomputed deadline layer {layer_idx} vs plain deadline"),
        );
    }
}

// =============================================================================
// PRE-COMPUTED IBP EQUIVALENCE
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// CROWN with pre-computed IBP produces identical bounds to CROWN without.
    ///
    /// The pre-computed IBP optimization (#3397) is purely a performance
    /// optimization: it should not change the mathematical result. This test
    /// generates random 3-layer networks and verifies that both paths produce
    /// identical output bounds.
    #[ntest::timeout(10000)]
    #[test]
    fn precomputed_ibp_matches_fresh_crown(
        w1_vec in prop::collection::vec(-2.0f32..2.0, 4),
        b1_vec in prop::collection::vec(-2.0f32..2.0, 2),
        w2_vec in prop::collection::vec(-2.0f32..2.0, 2),
        b2 in -2.0f32..2.0,
        (l1, u1) in valid_interval(3.0),
        (l2, u2) in valid_interval(3.0),
    ) {
        let w1 = ndarray::Array2::from_shape_vec((2, 2), w1_vec).unwrap();
        let b1 = ndarray::Array1::from_vec(b1_vec);
        let w2 = ndarray::Array2::from_shape_vec((1, 2), w2_vec).unwrap();
        let b2_arr = ndarray::Array1::from_vec(vec![b2]);

        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(w1, Some(b1)).unwrap(),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(
            LinearLayer::new(w2, Some(b2_arr)).unwrap(),
        ));

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn(),
        )
        .unwrap();

        // Path A: CROWN without pre-computed IBP (fresh computation).
        let fresh_output = network.propagate_crown(&input).unwrap();

        // Path B: CROWN with pre-computed IBP.
        let ibp_layer_bounds = network.collect_ibp_bounds(&input).unwrap();
        let precomputed_output = network
            .propagate_crown_with_precomputed_ibp(&input, ibp_layer_bounds, None, None)
            .unwrap();

        // Both paths must produce identical bounds (same math, same input).
        for i in 0..fresh_output.lower().len() {
            prop_assert!(
                (fresh_output.lower()[[i]] - precomputed_output.lower()[[i]]).abs() < FP_TOLERANCE,
                "Lower bound mismatch at [{}]: fresh={}, precomputed={}",
                i,
                fresh_output.lower()[[i]],
                precomputed_output.lower()[[i]]
            );
            prop_assert!(
                (fresh_output.upper()[[i]] - precomputed_output.upper()[[i]]).abs() < FP_TOLERANCE,
                "Upper bound mismatch at [{}]: fresh={}, precomputed={}",
                i,
                fresh_output.upper()[[i]],
                precomputed_output.upper()[[i]]
            );
        }
    }

    /// CROWN with pre-computed IBP is sound (all concrete outputs within bounds).
    ///
    /// Even if the equivalence test above passes, this independently verifies
    /// that the pre-computed path produces sound bounds by sampling concrete inputs.
    #[ntest::timeout(10000)]
    #[test]
    fn precomputed_ibp_crown_is_sound(
        w1_vec in prop::collection::vec(-2.0f32..2.0, 4),
        b1_vec in prop::collection::vec(-2.0f32..2.0, 2),
        w2_vec in prop::collection::vec(-2.0f32..2.0, 2),
        b2 in -2.0f32..2.0,
        (l1, u1) in valid_interval(3.0),
        (l2, u2) in valid_interval(3.0),
    ) {
        let w1 = ndarray::Array2::from_shape_vec((2, 2), w1_vec).unwrap();
        let b1 = ndarray::Array1::from_vec(b1_vec);
        let w2 = ndarray::Array2::from_shape_vec((1, 2), w2_vec).unwrap();
        let b2_arr = ndarray::Array1::from_vec(vec![b2]);

        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap(),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(
            LinearLayer::new(w2.clone(), Some(b2_arr.clone())).unwrap(),
        ));

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn(),
        )
        .unwrap();

        let ibp_layer_bounds = network.collect_ibp_bounds(&input).unwrap();
        let crown_output = network
            .propagate_crown_with_precomputed_ibp(&input, ibp_layer_bounds, None, None)
            .unwrap();

        for x1 in sample_points(l1, u1, 4) {
            for x2 in sample_points(l2, u2, 4) {
                let x = arr1(&[x1, x2]);
                let y1 = w1.dot(&x) + &b1;
                let relu_out = y1.mapv(|v| v.max(0.0));
                let final_out = w2.dot(&relu_out) + &b2_arr;

                prop_assert!(
                    crown_output.lower()[[0]] - FP_TOLERANCE <= final_out[0]
                        && final_out[0] <= crown_output.upper()[[0]] + FP_TOLERANCE,
                    "Pre-computed IBP CROWN soundness violation: output={} not in [{}, {}]",
                    final_out[0],
                    crown_output.lower()[[0]],
                    crown_output.upper()[[0]]
                );
            }
        }
    }

    /// CROWN with pre-computed IBP is at least as tight as IBP.
    ///
    /// The CROWN-IBP tightening step intersects CROWN backward bounds with IBP
    /// forward bounds. The result should never be looser than IBP alone.
    #[ntest::timeout(10000)]
    #[test]
    fn precomputed_ibp_crown_tighter_than_ibp(
        w1_vec in prop::collection::vec(-2.0f32..2.0, 4),
        b1_vec in prop::collection::vec(-2.0f32..2.0, 2),
        w2_vec in prop::collection::vec(-2.0f32..2.0, 2),
        b2 in -2.0f32..2.0,
        (l1, u1) in valid_interval(3.0),
        (l2, u2) in valid_interval(3.0),
    ) {
        let w1 = ndarray::Array2::from_shape_vec((2, 2), w1_vec).unwrap();
        let b1 = ndarray::Array1::from_vec(b1_vec);
        let w2 = ndarray::Array2::from_shape_vec((1, 2), w2_vec).unwrap();
        let b2_arr = ndarray::Array1::from_vec(vec![b2]);

        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(w1, Some(b1)).unwrap(),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(
            LinearLayer::new(w2, Some(b2_arr)).unwrap(),
        ));

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn(),
        )
        .unwrap();

        let ibp_output = network.propagate_ibp(&input).unwrap();
        let ibp_layer_bounds = network.collect_ibp_bounds(&input).unwrap();
        let crown_output = network
            .propagate_crown_with_precomputed_ibp(&input, ibp_layer_bounds, None, None)
            .unwrap();

        // CROWN lower bound >= IBP lower bound (tighter).
        prop_assert!(
            crown_output.lower()[[0]] >= ibp_output.lower()[[0]] - FP_TOLERANCE,
            "Pre-computed IBP CROWN lower ({}) looser than IBP lower ({})",
            crown_output.lower()[[0]],
            ibp_output.lower()[[0]]
        );
        // CROWN upper bound <= IBP upper bound (tighter).
        prop_assert!(
            crown_output.upper()[[0]] <= ibp_output.upper()[[0]] + FP_TOLERANCE,
            "Pre-computed IBP CROWN upper ({}) looser than IBP upper ({})",
            crown_output.upper()[[0]],
            ibp_output.upper()[[0]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_precomputed_ibp_elapsed_deadline_matches_plain_elapsed_deadline_3397() {
    let (network, input) = build_elapsed_deadline_output_network_3397();

    let expired_deadline = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());

    let plain_deadline_output = network
        .propagate_crown_with_engine_and_deadline(&input, None, expired_deadline)
        .unwrap();
    let ibp_layer_bounds = network.collect_ibp_bounds(&input).unwrap();
    let precomputed_deadline_output = network
        .propagate_crown_with_precomputed_ibp(&input, ibp_layer_bounds, None, expired_deadline)
        .unwrap();
    let ibp_output = network.propagate_ibp(&input).unwrap();

    assert_tensor_matches(
        &plain_deadline_output,
        &ibp_output,
        "plain deadline fallback vs IBP",
    );
    assert_tensor_matches(
        &precomputed_deadline_output,
        &ibp_output,
        "precomputed deadline fallback vs IBP",
    );
    assert_tensor_matches(
        &precomputed_deadline_output,
        &plain_deadline_output,
        "precomputed deadline fallback vs plain deadline fallback",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_precomputed_crown_ibp_elapsed_deadline_matches_plain_crown_ibp_3397() {
    let (network, input) = build_elapsed_deadline_crown_ibp_network_3397();

    let ibp_layer_bounds = network.collect_ibp_bounds(&input).unwrap();
    let expired_deadline = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());

    let plain_deadline_bounds = network
        .collect_crown_ibp_bounds_with_engine_and_deadline(&input, None, expired_deadline)
        .unwrap();
    let precomputed_deadline_bounds = network
        .collect_crown_ibp_bounds_with_precomputed_ibp(
            &input,
            ibp_layer_bounds.clone(),
            None,
            expired_deadline,
        )
        .unwrap();
    assert_deadline_layer_bounds_match(
        &plain_deadline_bounds,
        &precomputed_deadline_bounds,
        &ibp_layer_bounds,
    );
}
