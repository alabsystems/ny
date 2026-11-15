// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact rational controls for the scalar graph fan-in publication boundary.
//! Environment refusal uses a thread-local seam; ny-core's live probe tests
//! independently cover nearest rounding, FTZ, and DAZ observations.

use super::*;
use ndarray::{array, Array2};
use num_rational::BigRational;
use num_traits::Signed;
use std::cell::Cell;
use std::time::Duration;

thread_local! {
    static REFUSE_ENVIRONMENT: Cell<bool> = const { Cell::new(false) };
}

pub(super) fn environment_refused_for_test() -> bool {
    REFUSE_ENVIRONMENT.with(Cell::get)
}

struct RefusedEnvironment(bool);
impl RefusedEnvironment {
    fn new() -> Self {
        Self(REFUSE_ENVIRONMENT.with(|state| state.replace(true)))
    }
}
impl Drop for RefusedEnvironment {
    fn drop(&mut self) {
        REFUSE_ENVIRONMENT.with(|state| state.set(self.0));
    }
}

fn rational64(value: f64) -> BigRational {
    BigRational::from_float(value).expect("oracle operands are finite")
}

fn rational32(value: f32) -> BigRational {
    rational64(f32_to_f64_exact(value))
}

fn sidecar(lower: &[f64], upper: &[f64], lower_b: f64, upper_b: f64) -> LinearBounds64 {
    LinearBounds64 {
        lower_a: Array2::from_shape_vec((1, lower.len()), lower.to_vec()).unwrap(),
        lower_b: array![lower_b],
        upper_a: Array2::from_shape_vec((1, upper.len()), upper.to_vec()).unwrap(),
        upper_b: array![upper_b],
        lower_a_err: None,
        upper_a_err: None,
    }
}

fn accumulator_with_sidecar(indexed: bool, source: LinearBounds64) -> CrownMergeAccumulator {
    if indexed {
        let mut accumulator = CrownMergeAccumulator::new_indexed(&["residual".to_string()]);
        accumulator.indexed.as_mut().unwrap().merged_dense[0] = Some(source);
        accumulator
    } else {
        let mut accumulator = CrownMergeAccumulator::new();
        accumulator
            .merged_dense
            .insert("residual".to_string(), source);
        accumulator
    }
}

fn stored_source(accumulator: &CrownMergeAccumulator, indexed: bool) -> &LinearBounds64 {
    if indexed {
        accumulator.indexed.as_ref().unwrap().merged_dense[0]
            .as_ref()
            .unwrap()
    } else {
        &accumulator.merged_dense["residual"]
    }
}

fn assert_source_bits_equal(actual: &LinearBounds64, expected: &LinearBounds64) {
    for (actual, expected) in [
        (
            actual.lower_a().as_slice().unwrap(),
            expected.lower_a().as_slice().unwrap(),
        ),
        (
            actual.upper_a().as_slice().unwrap(),
            expected.upper_a().as_slice().unwrap(),
        ),
        (
            actual.lower_b().as_slice().unwrap(),
            expected.lower_b().as_slice().unwrap(),
        ),
        (
            actual.upper_b().as_slice().unwrap(),
            expected.upper_b().as_slice().unwrap(),
        ),
    ] {
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "a refused transaction must preserve the source bits"
        );
    }
    for (actual, expected) in [
        (actual.lower_a_err(), expected.lower_a_err()),
        (actual.upper_a_err(), expected.upper_a_err()),
    ] {
        assert_eq!(
            actual.map(|values| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()),
            expected.map(|values| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()),
        );
    }
}

/// Certify BOTH coefficient-error envelopes using rational arithmetic. This
/// oracle includes incoming uncertainty and does not repeat the conversion
/// implementation or depend on float tolerances.
fn assert_affine_containment(source: &LinearBounds64, published: &LinearBounds, point: &[f32]) {
    for row in 0..source.num_outputs() {
        for upper in [false, true] {
            let (source_a, source_b, source_e, a, b, e) = if upper {
                (
                    source.upper_a(),
                    source.upper_b(),
                    source.upper_a_err(),
                    published.upper_a(),
                    published.upper_b(),
                    published.upper_a_err(),
                )
            } else {
                (
                    source.lower_a(),
                    source.lower_b(),
                    source.lower_a_err(),
                    published.lower_a(),
                    published.lower_b(),
                    published.lower_a_err(),
                )
            };
            let mut original = rational64(source_b[row]);
            let mut result = rational32(b[row]);
            for (col, &value) in point.iter().enumerate() {
                let x = rational32(value);
                original += rational64(source_a[[row, col]]) * &x;
                result += rational32(a[[row, col]]) * &x;
                let original_error = rational64(source_e.map_or(0.0, |e| e[[row, col]])) * x.abs();
                let error = rational32(e.map_or(0.0, |e| e[[row, col]])) * x.abs();
                if upper {
                    original += original_error;
                    result += error;
                } else {
                    original -= original_error;
                    result -= error;
                }
            }
            assert!(
                if upper {
                    result >= original
                } else {
                    result <= original
                },
                "row {row}, upper={upper}, point={point:?}: {result} excludes {original}"
            );
        }
    }
}

#[test]
fn scalar_cast_exact_signed_cancellation_contains_original_zero_reproducer() {
    let _env = crate::tests::lock_env_shared();
    let source = sidecar(&[1.0, -1.0], &[0.0, 0.0], 0.0, 1.0);
    let published =
        CrownMergeAccumulator::downcast_dense_ref_with_deadline(&source, None, 0).unwrap();
    assert_eq!(published.lower_a(), &array![[1.0_f32, -1.0]]);
    assert!(
        !published.has_coeff_err(),
        "exact normal coefficients remain error-free"
    );
    assert_affine_containment(&source, &published, &[-1.0, -1.0]);
    let point = array![-1.0_f32, -1.0].into_dyn();
    let concrete =
        published.concretize_sound(&ny_tensor::BoundedTensor::new(point.clone(), point).unwrap());
    assert!(concrete.lower()[[0]] <= 0.0 && concrete.upper()[[0]] >= 0.0);
}

#[test]
fn scalar_cast_inexact_both_polarities_and_signed_vertices_are_rationally_contained() {
    let _env = crate::tests::lock_env_shared();
    let tiny = 2.0_f64.powi(-30);
    for source in [
        sidecar(&[1.0 + tiny, -1.0], &[1.0, -1.0 - tiny], -0.25, 0.25),
        sidecar(&[1.0, -1.0 - tiny], &[1.0 + tiny, -1.0], -0.25, 0.25),
    ] {
        let published =
            CrownMergeAccumulator::downcast_dense_ref_with_deadline(&source, None, 0).unwrap();
        assert!(published.lower_a_err().is_some() && published.upper_a_err().is_some());
        for x in [-1.0, 0.0, 1.0] {
            for y in [-1.0, 0.0, 1.0] {
                assert_affine_containment(&source, &published, &[x, y]);
            }
        }
    }
}

#[test]
fn scalar_cast_incoming_error_plus_cast_gap_never_shrinks_either_envelope() {
    let _env = crate::tests::lock_env_shared();
    let mut source = sidecar(
        &[1.0 + 2.0_f64.powi(-30), -0.5],
        &[1.0, -0.5 - 2.0_f64.powi(-30)],
        -0.125,
        0.125,
    );
    source.lower_a_err = Some(array![[0.1_f64, 2.0_f64.powi(-200)]]);
    source.upper_a_err = Some(array![[2.0_f64.powi(-200), 0.2]]);
    let published =
        CrownMergeAccumulator::downcast_dense_ref_with_deadline(&source, None, 0).unwrap();
    for point in [[-3.0, -2.0], [-3.0, 2.0], [3.0, -2.0], [3.0, 2.0]] {
        assert_affine_containment(&source, &published, &point);
    }
    for errors in [
        published.lower_a_err().unwrap(),
        published.upper_a_err().unwrap(),
    ] {
        assert!(errors.iter().all(|e| *e == 0.0 || e.is_normal()));
    }
}

#[test]
fn scalar_cast_binary32_subnormal_and_signed_zero_imports_preserve_all_source_bits() {
    let _env = crate::tests::lock_env_shared();
    let tiny = f32::from_bits(1);
    let mut source = LinearBounds::new(
        array![[tiny, -tiny, -0.0]],
        array![-tiny],
        array![[tiny, -tiny, 0.0]],
        array![tiny],
    )
    .unwrap();
    source.set_coeff_err(array![[tiny, 0.0, tiny]], array![[0.0, tiny, tiny]]);
    for widened in [
        CrownMergeAccumulator::widen_linear64_exact(&source),
        CrownMergeAccumulator::try_widen_linear64_with_deadline(
            &source,
            Some(Instant::now() + Duration::from_secs(5)),
            0,
        )
        .unwrap(),
    ] {
        assert_eq!(
            widened.lower_a()[[0, 0]].to_bits(),
            f32_to_f64_exact(tiny).to_bits()
        );
        assert_eq!(
            widened.lower_a()[[0, 1]].to_bits(),
            f32_to_f64_exact(-tiny).to_bits()
        );
        assert_eq!(widened.lower_a()[[0, 2]].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(
            widened.lower_b()[0].to_bits(),
            f32_to_f64_exact(-tiny).to_bits()
        );
        assert_eq!(
            widened.upper_b()[0].to_bits(),
            f32_to_f64_exact(tiny).to_bits()
        );
        assert_eq!(
            widened.lower_a_err().unwrap()[[0, 0]].to_bits(),
            f32_to_f64_exact(tiny).to_bits()
        );
    }
}

#[test]
fn scalar_cast_tiny_coefficients_and_biases_publish_flush_safe_containment() {
    let _env = crate::tests::lock_env_shared();
    for value in [
        f64::from_bits(1),
        2.0_f64.powi(-150),
        f32_to_f64_exact(f32::from_bits(1)),
        f32_to_f64_exact(f32::MIN_POSITIVE).next_down(),
    ] {
        let source = sidecar(&[value, -value], &[value, -value], -value, value);
        let published =
            CrownMergeAccumulator::downcast_dense_ref_with_deadline(&source, None, 0).unwrap();
        assert_eq!(published.lower_a()[[0, 0]].to_bits(), 0);
        assert_eq!(published.lower_a()[[0, 1]].to_bits(), (-0.0_f32).to_bits());
        for errors in [
            published.lower_a_err().unwrap(),
            published.upper_a_err().unwrap(),
        ] {
            assert!(errors.iter().all(|error| *error >= f32::MIN_POSITIVE));
        }
        assert_eq!(published.lower_b()[0], -f32::MIN_POSITIVE);
        assert_eq!(published.upper_b()[0], f32::MIN_POSITIVE);
        for point in [[-f32::MAX, -1.0], [-1.0, f32::MAX], [1.0, -f32::MAX]] {
            assert_affine_containment(&source, &published, &point);
        }
    }
}

#[test]
fn scalar_cast_full_difference_charges_subtraction_residual_before_tight_error_ceiling() {
    for (stored, source) in [
        (1.0_f32, -2.0_f64.powi(-100)),
        (-1.0, 2.0_f64.powi(-100)),
        (f32::MAX, -2.0_f64.powi(-100)),
    ] {
        let exact = (rational32(stored) - rational64(source)).abs();
        let charged = CrownMergeAccumulator::coefficient_cast_gap_up(stored, source);
        assert!(rational64(charged) >= exact);
        let error = CrownMergeAccumulator::err_to_f32(charged);
        if error.is_finite() {
            assert!(rational32(error) >= exact);
        }
    }
    assert_eq!(CrownMergeAccumulator::err_to_f32(0.0).to_bits(), 0);
    assert_eq!(
        CrownMergeAccumulator::err_to_f32(f64::from_bits(1)),
        f32::MIN_POSITIVE
    );
    for invalid in [-1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(CrownMergeAccumulator::err_to_f32(invalid), f32::INFINITY);
    }
}

#[test]
fn scalar_cast_generated_error_planes_are_paid_and_cap_refusal_is_atomic_in_both_stores() {
    crate::tests::with_env_edits(|env| {
        env.set("NY_DENSE_BUDGET_MB", "1");
        for indexed in [false, true] {
            for inexact in [false, true] {
                let coefficients = 40_000;
                let value = if inexact {
                    1.0 + 2.0_f64.powi(-30)
                } else {
                    1.0
                };
                let source = sidecar(
                    &vec![value; coefficients],
                    &vec![0.0; coefficients],
                    0.0,
                    1.0,
                );
                let mut accumulator = accumulator_with_sidecar(indexed, source.clone());
                let result = accumulator
                    .take_with_deadline("residual", Some(Instant::now() + Duration::from_secs(5)));
                if inexact {
                    let error = result.expect_err("four output planes exceed 1 MiB");
                    assert!(matches!(
                        error,
                        NyError::CpuMemoryExceeded {
                            required_bytes: 1_280_024,
                            budget_bytes: 1_048_576,
                            site: "CrownMergeAccumulator f64 take downcast staging"
                        }
                    ));
                    assert_source_bits_equal(stored_source(&accumulator, indexed), &source);
                    assert!(accumulator.contains_key("residual"));
                } else {
                    let published = result.unwrap().unwrap().into_dense().unwrap();
                    assert!(
                        !published.has_coeff_err(),
                        "the exact two-plane output fits 1 MiB"
                    );
                    assert!(!accumulator.contains_key("residual"));
                }
            }
        }
    });
}

#[test]
fn scalar_cast_expired_take_and_frontier_snapshot_leave_both_stores_unchanged() {
    let _env = crate::tests::lock_env_shared();
    let deadline = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
    for indexed in [false, true] {
        let source = sidecar(&[1.0 + 2.0_f64.powi(-30), -1.0], &[0.0, 0.0], 0.0, 1.0);
        let mut accumulator = accumulator_with_sidecar(indexed, source.clone());
        assert!(matches!(
            accumulator.take_with_deadline("residual", deadline),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert!(matches!(
            accumulator.snapshot_dense_with_deadline(deadline),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert_source_bits_equal(stored_source(&accumulator, indexed), &source);
        assert!(accumulator.contains_key("residual"));
    }
}

#[test]
fn scalar_cast_unqualified_binary64_environment_refuses_before_take_or_merge_publication() {
    let _env = crate::tests::lock_env_shared();
    require_f64_interval_proof_environment().expect("live normal host control");
    for indexed in [false, true] {
        for deadline in [None, Some(Instant::now() + Duration::from_secs(5))] {
            let source = sidecar(&[1.0], &[1.0], 0.0, 1.0);
            let mut accumulator = accumulator_with_sidecar(indexed, source.clone());
            let _refused = RefusedEnvironment::new();
            assert!(matches!(
                accumulator.take_with_deadline("residual", deadline),
                Err(NyError::SoundnessRefusal(_))
            ));
            assert!(matches!(
                accumulator.snapshot_dense_with_deadline(deadline),
                Err(NyError::SoundnessRefusal(_))
            ));
            assert!(matches!(
                accumulator.merge_dense_after_promotion_guard_with_deadline(
                    "residual",
                    LinearBounds::identity(1),
                    deadline
                ),
                Err(NyError::SoundnessRefusal(_))
            ));
            assert_source_bits_equal(stored_source(&accumulator, indexed), &source);
            assert!(accumulator.contains_key("residual"));

            let mut pending = if indexed {
                CrownMergeAccumulator::new_indexed(&["residual".to_string()])
            } else {
                CrownMergeAccumulator::new()
            };
            pending.insert(
                "residual".to_string(),
                CrownBounds::Dense(LinearBounds::identity(1)),
            );
            assert!(matches!(
                pending.merge_dense_after_promotion_guard_with_deadline(
                    "residual",
                    LinearBounds::identity(1),
                    deadline
                ),
                Err(NyError::SoundnessRefusal(_))
            ));
            let CrownBounds::Dense(unchanged) = pending.peek_pending("residual").unwrap() else {
                panic!("environment refusal must retain the pending Dense carrier");
            };
            assert_eq!(unchanged.lower_a(), &array![[1.0_f32]]);
            assert_eq!(unchanged.lower_b(), &array![0.0_f32]);
        }
    }
}

#[test]
fn scalar_cast_exact_signed_zero_needs_no_artificial_error_plane() {
    let _env = crate::tests::lock_env_shared();
    let source = sidecar(&[0.0, -0.0], &[-0.0, 0.0], -0.0, 0.0);
    let published =
        CrownMergeAccumulator::downcast_dense_ref_with_deadline(&source, None, 0).unwrap();
    assert!(!published.has_coeff_err());
    assert_eq!(published.lower_a()[[0, 1]].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(published.upper_a()[[0, 0]].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(published.lower_b()[0].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(published.upper_b()[0].to_bits(), 0);
    assert_affine_containment(&source, &published, &[-1.0, 1.0]);
}

#[test]
fn scalar_cast_bias_boundary_steps_never_publish_a_binary32_subnormal() {
    let min = f32_to_f64_exact(f32::MIN_POSITIVE);
    for value in [
        min.next_down(),
        min,
        min.next_up(),
        -min.next_down(),
        -min,
        -min.next_up(),
    ] {
        let lower = CrownMergeAccumulator::downcast_lower_bias(value).unwrap();
        let upper = CrownMergeAccumulator::downcast_upper_bias(value).unwrap();
        assert!(lower == 0.0 || lower.is_normal());
        assert!(upper == 0.0 || upper.is_normal());
        assert!(rational32(lower) <= rational64(value));
        assert!(rational32(upper) >= rational64(value));
    }
}

#[test]
fn scalar_cast_early_allocator_overage_cannot_consume_later_planes_allowance() {
    let mut overage = 0;
    let first = CrownMergeAccumulator::staging_elements_after_capacity(240, &mut overage, 60, 67);
    assert_eq!(
        first, 247,
        "seven extra elements do not erase seven unallocated nominal elements"
    );
    let second = CrownMergeAccumulator::staging_elements_after_capacity(240, &mut overage, 60, 60);
    assert_eq!(
        second, 247,
        "a later exact allocation cannot forget earlier overage"
    );
    let final_plane =
        CrownMergeAccumulator::staging_elements_after_capacity(240, &mut overage, 120, 121);
    assert_eq!(final_plane, 248);
    assert_eq!(
        CrownMergeAccumulator::staging_elements_after_capacity(usize::MAX, &mut overage, 1, 2),
        usize::MAX,
        "overflow remains a fail-closed receipt"
    );
}
