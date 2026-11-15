// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Independent signed rational support controls and returned-plane census.

use super::tests::{q32, q64};
use super::*;
use ndarray::{array, IxDyn};
use num_rational::BigRational as Q;
use num_traits::{Signed, Zero};

fn carrier(rows: usize, columns: usize) -> BatchedLinearBounds {
    BatchedLinearBounds::new(
        ArrayD::zeros(IxDyn(&[rows, columns])),
        ArrayD::zeros(IxDyn(&[rows])),
        ArrayD::zeros(IxDyn(&[rows, columns])),
        ArrayD::zeros(IxDyn(&[rows])),
        vec![columns],
        vec![rows],
    )
    .unwrap()
}

fn planes(bounds: &BatchedLinearBounds) -> usize {
    usize::from(bounds.lower_a_err.is_some()) + usize::from(bounds.upper_a_err.is_some())
}

fn error32(error: Option<&ArrayD<f32>>, row: usize, col: usize) -> Q {
    error.map_or_else(Q::zero, |e| q32(e[[row, col]]))
}
fn error64(error: Option<&ArrayD<f64>>, row: usize, col: usize) -> Q {
    error.map_or_else(Q::zero, |e| q64(e[[row, col]]))
}

/// The source sidecar's exact signed support is computed entirely in rationals.
/// Coefficient centers, carried uncertainty and biases each retain their own
/// original bit values; no production cast or gap helper is used as the oracle.
fn signed_support_contains(
    source: &BatchedLinearBounds64,
    stored: &BatchedLinearBounds,
    point: &[f32],
) {
    stored.validate_internal_shapes().unwrap();
    stored.validate_no_nan().unwrap();
    assert_eq!(source.lower_a.shape(), stored.lower_a().shape());
    for row in 0..source.lower_b.len() {
        let mut old_lower = q64(source.lower_b[row]);
        let mut old_upper = q64(source.upper_b[row]);
        let mut new_lower = q32(stored.lower_b()[row]);
        let mut new_upper = q32(stored.upper_b()[row]);
        for (col, &x) in point.iter().enumerate() {
            let x = q32(x);
            old_lower += q64(source.lower_a[[row, col]]) * &x
                - error64(source.lower_a_err.as_ref(), row, col) * x.abs();
            old_upper += q64(source.upper_a[[row, col]]) * &x
                + error64(source.upper_a_err.as_ref(), row, col) * x.abs();
            new_lower += q32(stored.lower_a()[[row, col]]) * &x
                - error32(stored.lower_a_err.as_ref(), row, col) * x.abs();
            new_upper += q32(stored.upper_a()[[row, col]]) * &x
                + error32(stored.upper_a_err.as_ref(), row, col) * x.abs();
        }
        assert!(new_lower <= old_lower, "lower row {row}");
        assert!(new_upper >= old_upper, "upper row {row}");
    }
}

#[test]
fn batched_f64_nearest_exact_zero_and_explicit_zero_errors_allocate_no_planes() {
    let mut base = carrier(2, 3);
    base.lower_a[[0, 1]] = -0.0;
    base.upper_a[[1, 2]] = -0.0;
    base.lower_b[0] = -0.5;
    base.upper_b[0] = 0.125;
    for explicit_zero in [false, true] {
        let mut original = base.clone();
        if explicit_zero {
            original.lower_a_err = Some(ArrayD::zeros(original.lower_a.raw_dim()));
            original.upper_a_err = Some(ArrayD::zeros(original.upper_a.raw_dim()));
        }
        let source = BatchedLinearBounds64::from_f32(&original);
        let stored = source.clone().into_f32();
        assert_eq!(planes(&stored), 0);
        assert_eq!(stored.memory_bytes(), base.memory_bytes());
        for (a, b) in base.lower_a().iter().zip(stored.lower_a()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for point in [[-4.0, 0.0, 3.0], [0.25, -8.0, -0.125]] {
            signed_support_contains(&source, &stored, &point);
        }
        assert_eq!(stored.lower_b()[0].to_bits(), next_down_f32(-0.5).to_bits());
        assert_eq!(stored.upper_b()[0].to_bits(), next_up_f32(0.125).to_bits());
    }
}

#[test]
fn batched_f64_nearest_sparse_inexact_lower_allocates_only_its_required_plane() {
    let base = carrier(2, 4);
    let mut source = BatchedLinearBounds64::from_f32(&base);
    source.lower_a[[1, 3]] = 1.0 + 2f64.powi(-25);
    let stored = source.clone().into_f32();
    assert_eq!(planes(&stored), 1);
    assert!(stored.upper_a_err.is_none());
    let error = stored.lower_a_err.as_ref().unwrap();
    assert_eq!(error.iter().filter(|&&e| e != 0.0).count(), 1);
    assert_eq!(
        stored.memory_bytes(),
        base.memory_bytes() + 8 * size_of::<f32>()
    );
    assert!(q32(error[[1, 3]]) >= q64(source.lower_a[[1, 3]]) - q32(1.0));
    for point in [[1.0, -2.0, 4.0, -8.0], [-0.5, 3.0, -7.0, 16.0]] {
        signed_support_contains(&source, &stored, &point);
    }
}

#[test]
fn batched_f64_nearest_signed_inexact_ties_keep_full_original_and_carried_gap() {
    let base = carrier(2, 3);
    let mut source = BatchedLinearBounds64::from_f32(&base);
    source.lower_a = array![
        [1.0 + 2f64.powi(-24), -1.0 - 2f64.powi(-25), 0.1],
        [1.0 + 3.0 * 2f64.powi(-24), -0.3, 2f64.powi(-90)]
    ]
    .into_dyn();
    source.upper_a = -&source.lower_a;
    source.lower_a_err = Some(ArrayD::from_elem(IxDyn(&[2, 3]), 2f64.powi(-30)));
    source.upper_a_err = Some(ArrayD::from_elem(IxDyn(&[2, 3]), 2f64.powi(-31)));
    source.lower_b[0] = -0.1;
    source.upper_b[0] = 0.3;
    let stored = source.clone().into_f32();
    assert_eq!(planes(&stored), 2);
    assert_eq!(stored.lower_a()[[0, 0]].to_bits(), 1.0f32.to_bits());
    assert_eq!(
        stored.lower_a()[[1, 0]].to_bits(),
        next_up_f32(next_up_f32(1.0)).to_bits()
    );
    assert_eq!(
        stored.memory_bytes(),
        base.memory_bytes() + 12 * size_of::<f32>()
    );
    for row in 0..2 {
        for col in 0..3 {
            for (a, e, nominal, carried) in [
                (
                    &source.lower_a,
                    &stored.lower_a_err,
                    stored.lower_a(),
                    &source.lower_a_err,
                ),
                (
                    &source.upper_a,
                    &stored.upper_a_err,
                    stored.upper_a(),
                    &source.upper_a_err,
                ),
            ] {
                assert!(
                    error32(e.as_ref(), row, col)
                        >= (q64(a[[row, col]]) - q32(nominal[[row, col]])).abs()
                            + error64(carried.as_ref(), row, col)
                );
            }
        }
    }
    for point in [[-16.0, 3.0, -0.125], [4.0, -8.0, 0.25], [0.0; 3]] {
        signed_support_contains(&source, &stored, &point);
    }
}

#[test]
fn batched_f64_nearest_subnormal_centers_zero_and_errors_publish_normal_upward() {
    let base = carrier(1, 4);
    let mut source = BatchedLinearBounds64::from_f32(&base);
    let values = [
        f64::from_bits(1),
        f32_to_f64_exact(f32::from_bits(1)),
        -f32_to_f64_exact(f32::from_bits(0x007f_ffff)),
        f32_to_f64_exact(f32::MIN_POSITIVE),
    ];
    for (i, &value) in values.iter().enumerate() {
        source.lower_a[[0, i]] = value;
        source.upper_a[[0, i]] = -value;
    }
    let stored = source.clone().into_f32();
    assert_eq!(planes(&stored), 2);
    for col in 0..3 {
        assert_eq!(stored.lower_a()[[0, col]], 0.0);
        assert_eq!(stored.upper_a()[[0, col]], 0.0);
        assert!(q32(stored.lower_a_err.as_ref().unwrap()[[0, col]]) >= q64(values[col]).abs());
    }
    assert_eq!(
        stored.lower_a()[[0, 3]].to_bits(),
        f32::MIN_POSITIVE.to_bits()
    );
    for error in stored.lower_a_err.iter().chain(stored.upper_a_err.iter()) {
        assert!(error
            .iter()
            .all(|&e| e == 0.0 || e.to_bits() & 0x7f80_0000 != 0));
    }
    for point in [[-8.0, 3.0, -2.0, 0.25], [8.0, -3.0, 2.0, -0.25]] {
        signed_support_contains(&source, &stored, &point);
    }
}

#[test]
fn batched_f64_nearest_error_only_one_plane_survives_zero_nominal_and_poison() {
    let mut original = carrier(1, 2);
    original.lower_a_err = Some(array![[0.125, f32::from_bits(1)]].into_dyn());
    let source = BatchedLinearBounds64::from_f32(&original);
    let stored = source.clone().into_f32();
    assert_eq!(planes(&stored), 1);
    assert!(stored.upper_a_err.is_none());
    assert!(stored
        .lower_a()
        .iter()
        .chain(stored.upper_a())
        .all(|&a| a == 0.0));
    assert_eq!(stored.lower_a_err.as_ref().unwrap()[[0, 0]], 0.125);
    assert!(stored.lower_a_err.as_ref().unwrap()[[0, 1]] >= f32::MIN_POSITIVE);
    for point in [[-4.0, 8.0], [16.0, -2.0]] {
        signed_support_contains(&source, &stored, &point);
    }
    original.lower_a_err.as_mut().unwrap()[[0, 1]] = f32::from_bits(0x8000_0001);
    let poisoned = BatchedLinearBounds64::from_f32(&original).into_f32();
    assert_eq!(planes(&poisoned), 1);
    assert!(poisoned.upper_a_err.is_none());
    assert_eq!(
        poisoned.lower_a_err.as_ref().unwrap()[[0, 1]],
        f32::INFINITY
    );
    poisoned.validate_no_nan().unwrap();
}

#[test]
fn batched_f64_nearest_cancellation_keeps_fan_in_error_and_roundtrip_planes() {
    let mut positive = carrier(1, 1);
    positive.lower_a[[0, 0]] = 1e30;
    positive.upper_a[[0, 0]] = 1e30;
    let mut one = carrier(1, 1);
    one.lower_a[[0, 0]] = 1.0;
    one.upper_a[[0, 0]] = 1.0;
    let mut negative = carrier(1, 1);
    negative.lower_a[[0, 0]] = -1e30;
    negative.upper_a[[0, 0]] = -1e30;
    let mut source = BatchedLinearBounds64::from_f32(&positive);
    source.accumulate(&one);
    source.accumulate(&negative);
    let stored = source.clone().into_f32();
    assert_eq!(planes(&stored), 2);
    assert_eq!(stored.lower_a()[[0, 0]], 0.0);
    assert!(stored.lower_a_err.as_ref().unwrap()[[0, 0]] >= 1.0);
    for point in [[-3.0], [4.0]] {
        signed_support_contains(&source, &stored, &point);
        let exact_coefficient = q32(positive.lower_a()[[0, 0]])
            + q32(one.lower_a()[[0, 0]])
            + q32(negative.lower_a()[[0, 0]]);
        assert_eq!(exact_coefficient, q32(1.0));
        let exact_value = exact_coefficient * q32(point[0]);
        let lo = q32(stored.lower_b()[0])
            - q32(stored.lower_a_err.as_ref().unwrap()[[0, 0]]) * q32(point[0]).abs();
        let hi = q32(stored.upper_b()[0])
            + q32(stored.upper_a_err.as_ref().unwrap()[[0, 0]]) * q32(point[0]).abs();
        assert!(lo <= exact_value && hi >= exact_value);
    }
    let second = BatchedLinearBounds64::from_f32(&stored).into_f32();
    assert_eq!(planes(&second), 2);
    assert_eq!(stored.lower_a_err, second.lower_a_err);
    assert_eq!(stored.upper_a_err, second.upper_a_err);
}

#[test]
fn batched_f64_nearest_one_plane_is_consumed_by_sound_signed_concretization() {
    let base = carrier(1, 2);
    let mut source = BatchedLinearBounds64::from_f32(&base);
    source.lower_a[[0, 0]] = 1.0 + 2f64.powi(-25);
    source.lower_a[[0, 1]] = -0.3;
    source.upper_a[[0, 0]] = -1.0;
    source.upper_a[[0, 1]] = 0.5;
    source.upper_b[0] = 16.0;
    let stored = source.clone().into_f32();
    assert_eq!(planes(&stored), 1);
    let input =
        ny_tensor::BoundedTensor::new(array![-2.0, -3.0].into_dyn(), array![4.0, 5.0].into_dyn())
            .unwrap();
    let concrete = stored.concretize_sound(&input).unwrap();
    for point in [[-2.0, -3.0], [-2.0, 5.0], [4.0, -3.0], [4.0, 5.0]] {
        let mut lower = q64(source.lower_b[0]);
        let mut upper = q64(source.upper_b[0]);
        for (i, &x) in point.iter().enumerate() {
            lower += q64(source.lower_a[[0, i]]) * q32(x);
            upper += q64(source.upper_a[[0, i]]) * q32(x);
        }
        assert!(q32(concrete.lower()[0]) <= lower);
        assert!(q32(concrete.upper()[0]) >= upper);
    }
}
