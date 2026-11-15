// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use num_rational::BigRational;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

fn fixture(positions: usize, error_position: usize, mask: usize) -> (LinearLayer, LinearBounds) {
    let bias = 1048576.0;
    let layer =
        LinearLayer::new(Array2::zeros((1, 1)), Some(Array1::from_vec(vec![bias]))).unwrap();
    let mut bounds = LinearBounds::new(
        Array2::ones((1, positions)),
        Array1::from_vec(vec![-(positions as f32) * bias]),
        Array2::ones((1, positions)),
        Array1::from_vec(vec![-(positions as f32) * bias]),
    )
    .unwrap();
    let mut error = Array2::zeros((1, positions));
    error[[0, error_position]] = 2f32.powi(-23);
    bounds.lower_a_err = (mask & 1 != 0).then(|| error.clone());
    bounds.upper_a_err = (mask & 2 != 0).then_some(error);
    (layer, bounds)
}

fn contains_later_error(out: &LinearBounds, mask: usize) {
    assert!(out
        .lower_a()
        .iter()
        .chain(out.upper_a().iter())
        .all(|&v| v == 0.0));
    assert!(out
        .lower_a_err()
        .unwrap()
        .iter()
        .chain(out.upper_a_err().unwrap().iter())
        .all(|&v| v == 0.0));
    // A possible true coefficient at the late position is1 +/-2^-24,
    // within stored1 +/-2^-23. At the constant2^20 layer output and the
    // cancelling old bias, its exact real residual is +/-1/16.
    if mask & 1 != 0 {
        assert!(
            out.lower_b()[0] <= -0.0625,
            "later lower carrier excluded: bias={} exact=-0.0625",
            out.lower_b()[0]
        );
    }
    if mask & 2 != 0 {
        assert!(
            out.upper_b()[0] >= 0.0625,
            "later upper carrier excluded: bias={} exact=0.0625",
            out.upper_b()[0]
        );
    }
}

#[test]
fn sequence_bias_error_cpu_counterexample_contains_later_carrier_with_and_without_deadline_stack() {
    let (layer, bounds) = fixture(2, 1, 3);
    for deadline in [None, Some(Instant::now() + Duration::from_secs(60))] {
        for stack in [false, true] {
            let out = propagate_linear_cpu_with_deadline_and_stack(
                &layer, &bounds, deadline, false, stack,
            )
            .unwrap();
            contains_later_error(&out, 3);
        }
    }
}

struct ZeroWeightEngine(AtomicUsize);
impl GemmEngine for ZeroWeightEngine {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert!(b.iter().all(|&v| v == 0.0));
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(vec![0.0; m * n])
    }
}

#[test]
fn sequence_bias_error_engine_counterexample_executes_all_engine_calls_and_contains_carrier() {
    for mask in [2, 1, 3] {
        let (layer, bounds) = fixture(2, 1, mask);
        let engine = ZeroWeightEngine(AtomicUsize::new(0));
        let out = propagate_linear_via_gemm(&layer, &bounds, &engine, None).unwrap();
        assert_eq!(engine.0.load(Ordering::Relaxed), 4);
        contains_later_error(&out, mask);
    }
}

#[test]
fn sequence_bias_error_every_later_position_and_one_sided_plane_is_propagated() {
    for positions in [2, 3, 17] {
        for position in 1..positions {
            for mask in [1, 2, 3] {
                let (layer, bounds) = fixture(positions, position, mask);
                let out = propagate_linear_cpu_with_deadline_and_stack(
                    &layer, &bounds, None, false, false,
                )
                .unwrap();
                contains_later_error(&out, mask);
            }
        }
    }
}

#[test]
fn sequence_bias_error_single_position_keeps_frozen_historical_bias_bits() {
    let (layer, bounds) = fixture(1, 0, 3);
    let out =
        propagate_linear_cpu_with_deadline_and_stack(&layer, &bounds, None, false, false).unwrap();
    assert_eq!(out.lower_b()[0].to_bits(), 0xbe000001);
    assert_eq!(out.upper_b()[0].to_bits(), 0x3e000001);
}

#[test]
fn sequence_bias_error_general_shared_bias_encloses_exact_rational_all_position_penalty() {
    let positions = 3;
    let rows = 2;
    let k = 3;
    let bias = Array1::from_vec(vec![1.0e20, -1.0e20, 0.375]);
    let layer = LinearLayer::new(
        Array2::from_shape_vec((3, 2), vec![-1.0, 2.0, 3.0, 0.5, 0.0, -0.25]).unwrap(),
        Some(bias.clone()),
    )
    .unwrap();
    let lower = Array2::from_shape_fn((rows, k * positions), |(i, j)| ((i + j) % 4) as f32 - 1.0);
    let upper = -&lower;
    let errors = Array2::from_elem((rows, k * positions), 0.03125);
    let bounds = LinearBounds::new_or_conservative_with_err(
        lower.clone(),
        Array1::zeros(rows),
        upper.clone(),
        Array1::zeros(rows),
        errors.clone(),
        errors.clone(),
    )
    .unwrap();
    let out =
        propagate_linear_cpu_with_deadline_and_stack(&layer, &bounds, None, false, false).unwrap();
    let r = |v: f32| BigRational::from_float(f32_to_f64_exact(v)).unwrap();
    for i in 0..rows {
        let mut lo = BigRational::from_integer(0.into());
        let mut up = lo.clone();
        let mut penalty = lo.clone();
        for pos in 0..positions {
            for j in 0..k {
                lo += r(lower[[i, pos * k + j]]) * r(bias[j]);
                up += r(upper[[i, pos * k + j]]) * r(bias[j]);
                penalty += r(errors[[i, pos * k + j]]) * r(bias[j].abs());
            }
        }
        assert!(r(out.lower_b()[i]) <= lo - &penalty);
        assert!(r(out.upper_b()[i]) >= up + &penalty);
    }
}

#[test]
fn sequence_bias_error_late_poison_degrades_nonzero_bias_and_zero_bias_stays_exact() {
    for bias in [0.0, 1.0] {
        let layer =
            LinearLayer::new(Array2::zeros((1, 1)), Some(Array1::from_vec(vec![bias]))).unwrap();
        let mut bounds = LinearBounds::new(
            Array2::zeros((1, 2)),
            Array1::zeros(1),
            Array2::zeros((1, 2)),
            Array1::zeros(1),
        )
        .unwrap();
        let error = Array2::from_shape_vec((1, 2), vec![0.0, f32::NAN]).unwrap();
        bounds.lower_a_err = Some(error.clone());
        bounds.upper_a_err = Some(error);
        let out = propagate_linear_cpu_with_deadline_and_stack(&layer, &bounds, None, false, false)
            .unwrap();
        if bias == 0.0 {
            assert_eq!(out.lower_b()[0], 0.0);
            assert_eq!(out.upper_b()[0], 0.0);
        } else {
            assert_eq!(out.lower_b()[0], f32::NEG_INFINITY);
            assert_eq!(out.upper_b()[0], f32::INFINITY);
        }
    }
}
