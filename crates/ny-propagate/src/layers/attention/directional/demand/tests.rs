// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::propagate_dense_queries_for_test;
use super::*;
use crate::layers::attention::crown_ternary::{parse_slice_dims, AttnTernaryResult};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use std::time::Duration;

fn bounded(shape: &[usize], center: &[f32], radius: f32) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), center.iter().map(|&x| x - radius).collect()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(shape), center.iter().map(|&x| x + radius).collect()).unwrap(),
    )
    .unwrap()
}

fn equal_bits(left: &AttnTernaryResult, right: &AttnTernaryResult) {
    for (left, right) in left.0.iter().zip(&right.0) {
        let (left, right) = (left.as_ref().unwrap(), right.as_ref().unwrap());
        for (a, b) in [
            (left.lower_a(), right.lower_a()),
            (left.upper_a(), right.upper_a()),
        ] {
            assert_eq!(a.shape(), b.shape());
            assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        assert_eq!(left.lower_b(), right.lower_b());
        assert_eq!(left.upper_b(), right.upper_b());
        assert!(left.lower_a_err().is_none() && left.upper_a_err().is_none());
    }
    for (a, b) in [(&left.1, &right.1), (&left.2, &right.2)] {
        assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}

#[test]
fn mixed_query_head_demand_has_bitwise_dense_control_parity_with_strided_carriers() {
    let shape = [2, 4, 2];
    let q_values: Vec<_> = (0..16).map(|i| (i as f32 - 8.0) * 0.03).collect();
    let k_values: Vec<_> = (0..16)
        .map(|i| ((i * 3 % 11) as f32 - 5.0) * 0.04)
        .collect();
    let v_values: Vec<_> = (0..16).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1).collect();
    let q = bounded(&shape, &q_values, 0.01);
    let k = bounded(&shape, &k_values, 0.015);
    let v = bounded(&shape, &v_values, 0.02);
    // F layout forces the scan to use typed indexing rather than flattening A.
    let strided = || Array2::<f32>::zeros((16, 3)).reversed_axes();
    let (mut lower, mut upper, mut le, mut ue) = (strided(), strided(), strided(), strided());
    lower[[0, 0]] = 1.0;
    lower[[0, 1]] = -0.5;
    upper[[0, 6]] = -0.75;
    lower[[1, 8]] = -0.9;
    upper[[1, 14]] = 0.25;
    // Query/head IDs 2 and 5 are error-only; IDs 1 and 6 are fully inactive.
    le[[0, 4]] = 0.125;
    ue[[1, 10]] = 0.2;
    let lb = Array1::from_vec(vec![-0.3, 0.2, -0.0]);
    let ub = Array1::from_vec(vec![0.4, 0.7, 0.125]);
    let node =
        LinearBounds::new_or_conservative_with_err(lower, lb.clone(), upper, ub.clone(), le, ue)
            .unwrap();
    assert!(node.lower_a().as_slice().is_none());
    for layer in [
        SelfAttentionLayer::standard(),
        SelfAttentionLayer::causal(),
        SelfAttentionLayer::causal().with_window_size(1),
    ] {
        let result = layer.propagate_crown_ternary(&node, &q, &k, &v).unwrap();
        let dense = propagate_dense_queries_for_test(&layer, &node, &q, &k, &v).unwrap();
        equal_bits(&result, &dense);
        assert_eq!(result.1[2].to_bits(), lb[2].to_bits());
        assert_eq!(result.2[2].to_bits(), ub[2].to_bits());
        for bound in &result.0 {
            let bound = bound.as_ref().unwrap();
            assert!(bound.lower_a().row(2).iter().all(|&x| x == 0.0));
            assert!(bound.upper_a().row(2).iter().all(|&x| x == 0.0));
        }
        let q_bound = result.0[0].as_ref().unwrap();
        for query in [1, 6] {
            for row in 0..3 {
                for channel in 0..2 {
                    assert_eq!(q_bound.lower_a()[[row, query * 2 + channel]], 0.0);
                    assert_eq!(q_bound.upper_a()[[row, query * 2 + channel]], 0.0);
                }
            }
        }
    }
}

#[test]
fn fully_zero_demand_preserves_incoming_bias_bits_and_zero_carriers() {
    let box_ = bounded(&[3, 2], &[0.25; 6], 0.1);
    let node = LinearBounds::new(
        Array2::from_elem((2, 6), -0.0),
        Array1::from_vec(vec![-0.0, -1.25]),
        Array2::zeros((2, 6)),
        Array1::from_vec(vec![0.0, 2.5]),
    )
    .unwrap();
    let result = SelfAttentionLayer::causal()
        .propagate_crown_ternary(&node, &box_, &box_, &box_)
        .unwrap();
    for row in 0..2 {
        assert_eq!(result.1[row].to_bits(), node.lower_b()[row].to_bits());
        assert_eq!(result.2[row].to_bits(), node.upper_b()[row].to_bits());
    }
    for bound in result.0 {
        let bound = bound.unwrap();
        assert!(bound
            .lower_a()
            .iter()
            .chain(bound.upper_a())
            .all(|&a| a == 0.0));
    }
}

#[test]
fn error_only_query_remains_active_and_pays_both_sided_error() {
    let q = bounded(&[1, 1], &[0.2], 0.01);
    let v = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![4.0]).unwrap(),
    )
    .unwrap();
    let node = LinearBounds::new_or_conservative_with_err(
        Array2::zeros((1, 1)),
        Array1::zeros(1),
        Array2::zeros((1, 1)),
        Array1::zeros(1),
        Array2::from_elem((1, 1), 0.25),
        Array2::from_elem((1, 1), 0.5),
    )
    .unwrap();
    let layer = SelfAttentionLayer::standard();
    let dims = parse_slice_dims(&layer, &q, &q, &v).unwrap();
    let demand = Demand::scan(&layer, &node, &dims, None, None, true).unwrap();
    assert_eq!(demand.summary.active_queries, 1);
    let result = layer.propagate_crown_ternary(&node, &q, &q, &v).unwrap();
    assert!(result.1[0] <= -1.0 && result.2[0] >= 2.0);
    let dense = propagate_dense_queries_for_test(&layer, &node, &q, &q, &v).unwrap();
    equal_bits(&result, &dense);
}

#[test]
fn exact_demand_counts_visible_mask_pairs_and_retains_subnormal_carriers() {
    let box_ = bounded(&[2, 4, 2], &[0.0; 16], 0.01);
    let mut a = Array2::zeros((1, 16));
    for query in [1, 3, 6] {
        a[[0, query * 2]] = 1.0;
    }
    let node = LinearBounds::symmetric(a, Array1::zeros(1)).unwrap();
    for (layer, pairs, max_keys) in [
        (SelfAttentionLayer::standard(), 12, 4),
        (SelfAttentionLayer::causal(), 9, 4),
        (SelfAttentionLayer::causal().with_window_size(1), 6, 2),
    ] {
        let dims = parse_slice_dims(&layer, &box_, &box_, &box_).unwrap();
        let demand = Demand::scan(&layer, &node, &dims, None, None, true).unwrap();
        assert_eq!(demand.summary.active_queries, 3);
        assert_eq!(demand.summary.visible_pairs, pairs);
        assert_eq!(demand.summary.max_visible_keys, max_keys);
        assert_eq!(demand.summary.map_bytes, 8);
    }
    let mut lower = Array2::zeros((1, 16));
    lower[[0, 0]] = f32::from_bits(1);
    let mut upper_error = Array2::zeros((1, 16));
    upper_error[[0, 2]] = f32::from_bits(1);
    let node = LinearBounds::new_or_conservative_with_err(
        lower,
        Array1::zeros(1),
        Array2::zeros((1, 16)),
        Array1::zeros(1),
        Array2::zeros((1, 16)),
        upper_error,
    )
    .unwrap();
    let layer = SelfAttentionLayer::standard();
    let dims = parse_slice_dims(&layer, &box_, &box_, &box_).unwrap();
    let demand = Demand::scan(&layer, &node, &dims, None, None, true).unwrap();
    assert_eq!(demand.summary.active_queries, 2);
    assert!(demand.active(0, 0, 4) && demand.active(0, 1, 4));
}

#[test]
fn demand_scan_admits_work_and_checkpoints_before_allocating_map() {
    let box_ = bounded(&[4, 1], &[0.0; 4], 0.01);
    let layer = SelfAttentionLayer::standard();
    let node = LinearBounds::symmetric(Array2::zeros((1, 4)), Array1::zeros(1)).unwrap();
    let dims = parse_slice_dims(&layer, &box_, &box_, &box_).unwrap();
    for budget in [
        WorkBudget::new(None, 1, 0, u64::MAX, 1, Duration::ZERO).unwrap(),
        WorkBudget::new(None, 1 << 20, 0, 0, 1, Duration::ZERO).unwrap(),
        WorkBudget::new(
            Some(
                Instant::now()
                    .checked_sub(Duration::from_millis(1))
                    .unwrap(),
            ),
            1 << 20,
            0,
            u64::MAX,
            1,
            Duration::ZERO,
        )
        .unwrap(),
    ] {
        assert!(Demand::scan(&layer, &node, &dims, None, Some(&budget), true).is_err());
        let receipt = budget.receipt().unwrap();
        assert_eq!(receipt.admitted, 0);
        assert_eq!(receipt.work_reserved, 0);
        assert_eq!(receipt.live_scratch_bytes, 0);
    }
}

#[test]
fn zero_demand_retains_finite_input_and_contiguous_layout_profile() {
    let node = LinearBounds::symmetric(Array2::zeros((1, 6)), Array1::zeros(1)).unwrap();
    let layer = SelfAttentionLayer::standard();
    let infinite = BoundedTensor::new_conservative(&[3, 2]);
    let result = layer.propagate_crown_ternary(&node, &infinite, &infinite, &infinite);
    assert!(matches!(
        result,
        Err(NyError::UnsupportedOp(message)) if message.contains("finite input")
    ));
    let noncontiguous = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 3]), -0.1).reversed_axes(),
        ArrayD::from_elem(IxDyn(&[2, 3]), 0.1).reversed_axes(),
    )
    .unwrap();
    assert!(noncontiguous.lower().as_slice().is_none());
    assert!(matches!(
        layer.propagate_crown_ternary(&node, &noncontiguous, &noncontiguous, &noncontiguous),
        Err(NyError::UnsupportedOp(message)) if message.contains("non-contiguous input")
    ));
}

#[test]
fn sparse_last_query_crosses_default_work_admission_with_real_projection_and_envelopes() {
    const SEQ: usize = 1024;
    const WIDTH: usize = 16;
    let q_center: Vec<_> = (0..SEQ * WIDTH)
        .map(|i| if i % WIDTH == 0 { 0.25 } else { 0.01 })
        .collect();
    let k_center: Vec<_> = (0..SEQ * WIDTH)
        .map(|i| {
            let row = ((i / WIDTH) % 13) as f32 - 6.0;
            row * if i % WIDTH == 0 { 0.02 } else { 0.001 }
        })
        .collect();
    let v_center: Vec<_> = (0..SEQ * WIDTH)
        .map(|i| ((i / WIDTH * 7 + i % WIDTH * 3) % 17) as f32 * 0.025 - 0.2)
        .collect();
    let q = bounded(&[SEQ, WIDTH], &q_center, 0.001);
    let k = bounded(&[SEQ, WIDTH], &k_center, 0.001);
    let v = bounded(&[SEQ, WIDTH], &v_center, 0.0005);
    let mut direction = Array2::zeros((1, SEQ * WIDTH));
    direction[[0, (SEQ - 1) * WIDTH]] = 1.0;
    direction[[0, (SEQ - 1) * WIDTH + 1]] = -0.25;
    let node = LinearBounds::symmetric(direction, Array1::zeros(1)).unwrap();
    let layer = SelfAttentionLayer::causal();
    assert!(matches!(
        propagate_dense_queries_for_test(&layer, &node, &q, &k, &v),
        Err(NyError::UnsupportedOp(message)) if message.contains("construction admission")
    ));
    let result = layer.propagate_crown_ternary(&node, &q, &k, &v).unwrap();
    let q_bound = result.0[0].as_ref().unwrap();
    let v_bound = result.0[2].as_ref().unwrap();
    assert!(q_bound
        .lower_a()
        .iter()
        .skip((SEQ - 1) * WIDTH)
        .any(|&a| a != 0.0));
    assert!(q_bound
        .lower_a()
        .iter()
        .take((SEQ - 1) * WIDTH)
        .all(|&a| a == 0.0));
    assert!(v_bound.lower_a().iter().any(|&a| a != 0.0));
    assert!(result.1.iter().chain(&result.2).all(|x| x.is_finite()));
    // An independent binary64 forward diagnostic on four joint-box controls.
    // Only the demanded last query is evaluated; no quadratic dense forward.
    for choice in 0..4 {
        let input = |bounds: &BoundedTensor| -> Vec<f32> {
            bounds
                .lower()
                .iter()
                .zip(bounds.upper())
                .enumerate()
                .map(
                    |(i, (&lo, &hi))| {
                        if (i * 13 + choice) % 4 < 2 {
                            lo
                        } else {
                            hi
                        }
                    },
                )
                .collect()
        };
        let (qp, kp, vp) = (input(&q), input(&k), input(&v));
        let scores: Vec<f64> = (0..SEQ)
            .map(|key| {
                (0..WIDTH)
                    .map(|d| qp[(SEQ - 1) * WIDTH + d] as f64 * kp[key * WIDTH + d] as f64)
                    .sum::<f64>()
                    * layer.resolve_scale(&q).unwrap() as f64
            })
            .collect();
        let shift = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<_> = scores.iter().map(|s| (s - shift).exp()).collect();
        let truth = weights
            .iter()
            .enumerate()
            .map(|(key, &p)| p * (vp[key * WIDTH] as f64 - 0.25 * vp[key * WIDTH + 1] as f64))
            .sum::<f64>()
            / weights.iter().sum::<f64>();
        let (mut lo, mut hi) = (result.1[0] as f64, result.2[0] as f64);
        for (bound, point) in result.0.iter().zip([&qp, &kp, &vp]) {
            let bound = bound.as_ref().unwrap();
            lo += bound
                .lower_a()
                .row(0)
                .iter()
                .zip(point)
                .map(|(&a, &x)| a as f64 * x as f64)
                .sum::<f64>();
            hi += bound
                .upper_a()
                .row(0)
                .iter()
                .zip(point)
                .map(|(&a, &x)| a as f64 * x as f64)
                .sum::<f64>();
        }
        assert!(
            lo <= truth && truth <= hi,
            "point {choice}: [{lo},{hi}] excludes {truth}"
        );
    }
}
