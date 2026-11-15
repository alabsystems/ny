// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::certify::work::WorkBudget;
use ndarray::{ArrayD, IxDyn};
use std::time::{Duration, Instant};

#[test]
fn checked_causal_route_reuses_the_common_budget_and_bounded_fallback() {
    let pre =
        BoundedTensor::new(ArrayD::zeros(IxDyn(&[1, 1])), ArrayD::zeros(IxDyn(&[1, 1]))).unwrap();
    let layer = CausalSoftmaxLayer::new(-1);
    let budget = WorkBudget::new(None, 1 << 20, 0, u64::MAX, 0, Duration::ZERO).unwrap();
    let _ = layer.propagate_ibp_with_budget(&pre, &budget).unwrap();
    let receipt = budget.receipt().unwrap();
    assert_eq!(receipt.completed, 1);
    assert_eq!(receipt.live_scratch_bytes, 0);
    assert!(receipt.work_reserved > 0 && receipt.peak_leased_bytes > 0);
    // The checked transcendental route cannot fit this same ledger. A cheap
    // one-key simplex identity can still complete under the remaining budget.
    let fallback = WorkBudget::new(None, 1 << 20, 0, 1000, 0, Duration::ZERO).unwrap();
    let result = layer.propagate_ibp_with_budget(&pre, &fallback).unwrap();
    assert_eq!(result.lower()[[0, 0]], 1.0);
    assert_eq!(result.upper()[[0, 0]], 1.0);
    let receipt = fallback.receipt().unwrap();
    assert_eq!(receipt.refused, 1);
    assert_eq!(receipt.admitted, 1);
    assert_eq!(receipt.completed, 1);
    assert!(receipt.work_reserved <= 1000);
    for budget in [
        WorkBudget::new(None, 1, 0, u64::MAX, 0, Duration::ZERO).unwrap(),
        WorkBudget::new(None, 1 << 20, 0, 0, 0, Duration::ZERO).unwrap(),
        WorkBudget::new(
            Some(
                Instant::now()
                    .checked_sub(Duration::from_millis(1))
                    .unwrap(),
            ),
            1 << 20,
            0,
            u64::MAX,
            0,
            Duration::ZERO,
        )
        .unwrap(),
    ] {
        assert!(layer.propagate_ibp_with_budget(&pre, &budget).is_err());
        assert_eq!(budget.receipt().unwrap().admitted, 0);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    }
}

#[test]
fn cubic_ratio_work_becomes_checked_linear_row_work_and_preserves_masked_zeros() {
    // Admission uses the actual causal triangle, not K ratios times K terms.
    let admitted = admission::Admission::new(&[1024, 1024], None, None).unwrap();
    assert!(admitted.checked);
    let short_window = admission::Admission::new(&[8, 1024, 1024], Some(7), None).unwrap();
    assert!(short_window.checked);
    // Exercise the real default route, with nonuniform finite score boxes.
    let shape = [2, 64, 64];
    let lower = ArrayD::from_shape_fn(IxDyn(&shape), |i| {
        ((i[1] * 13 + i[2] * 7) % 17) as f32 * 0.03 - 0.2
    });
    let upper = lower.mapv(|x| x + 0.01);
    let pre = BoundedTensor::new(lower, upper).unwrap();
    for window in [None, Some(0), Some(3)] {
        let mut layer = CausalSoftmaxLayer::new(-1);
        layer.window_size = window;
        let result = layer.propagate_ibp(&pre).unwrap();
        for b in 0..2 {
            for q in 0..64 {
                let active = layer.active_range(q, 64);
                let point: Vec<f64> = active
                    .clone()
                    .map(|k| pre.lower()[[b, q, k]] as f64)
                    .collect();
                let shift = point.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weight: Vec<_> = point.iter().map(|s| (s - shift).exp()).collect();
                let mass: f64 = weight.iter().sum();
                for (index, k) in active.clone().enumerate() {
                    let truth = weight[index] / mass;
                    assert!(
                        result.lower()[[b, q, k]] as f64 <= truth
                            && truth <= result.upper()[[b, q, k]] as f64
                    );
                }
                for k in 0..64 {
                    if !active.contains(&k) {
                        assert_eq!(result.lower()[[b, q, k]], 0.0);
                        assert_eq!(result.upper()[[b, q, k]], 0.0);
                    }
                }
            }
        }
    }
}
