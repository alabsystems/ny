// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Refusal wiring uses a thread-local seam, never mutates hardware FP state.
//! ny-core separately checks actual nearest/gradual-underflow observations.

use super::*;
use crate::layers::{LayerNormLayer, RmsNormLayer};
use crate::LinearBounds;
use ndarray::{arr1, arr2};
use num_rational::BigRational;
use num_traits::Signed;
use std::cell::Cell;
use std::time::Duration;

thread_local! {
    static REFUSE_ENVIRONMENT: Cell<bool> = const { Cell::new(false) };
}

pub(super) fn refused_for_test() -> bool {
    REFUSE_ENVIRONMENT.with(Cell::get)
}

struct RefusalScope(bool);
impl RefusalScope {
    fn new(refuse: bool) -> Self {
        Self(REFUSE_ENVIRONMENT.with(|state| state.replace(refuse)))
    }
}
impl Drop for RefusalScope {
    fn drop(&mut self) {
        REFUSE_ENVIRONMENT.with(|state| state.set(self.0));
    }
}

fn incoming(active: bool) -> LinearBounds {
    let a = if active {
        arr2(&[[0.75, -1.25]])
    } else {
        arr2(&[[0.0, -0.0]])
    };
    let mut result = LinearBounds::new(a.clone(), arr1(&[0.125]), a, arr1(&[0.375])).unwrap();
    if active {
        result.set_coeff_err(arr2(&[[0.03125, 0.0625]]), arr2(&[[0.0625, 0.03125]]));
    }
    result
}

fn input(batch: bool) -> BoundedTensor {
    BoundedTensor::concrete(
        ArrayD::from_shape_vec(IxDyn(if batch { &[1, 2] } else { &[2] }), vec![-1.0, 1.0]).unwrap(),
    )
    .unwrap()
}

fn refused<T>(result: Result<T>) {
    assert!(
        matches!(result, Err(NyError::SoundnessRefusal(_))),
        "unsupported mathematical normalization arithmetic must fail closed"
    );
}

#[test]
fn radial_environment_default_scalar_and_batched_layernorm_rmsnorm_refuse_even_zero_demand() {
    let layer = LayerNormLayer::new(arr1(&[2.0, -0.5]), arr1(&[0.25, -0.125]), 3.0).unwrap();
    let rms = RmsNormLayer::new(arr1(&[2.0, -0.5]), 3.0).unwrap();
    let scalar_pre = input(false);
    let batched_pre = input(true);
    for active in [false, true] {
        let scalar = incoming(active);
        let batched = scalar_to_batched(&scalar).unwrap();
        geometry::reset_constructions();
        let _environment = RefusalScope::new(true);
        refused(layer.propagate_linear_with_bounds(&scalar, &scalar_pre));
        refused(rms.propagate_linear_with_bounds(&scalar, &scalar_pre));
        refused(layer.propagate_linear_batched_with_bounds(&batched, &batched_pre));
        refused(rms.propagate_linear_batched_with_bounds(&batched, &batched_pre));
        assert_eq!(geometry::constructions(), 0);
    }
}

#[test]
fn radial_environment_direct_candidate_geometry_norm_and_refinement_refuse_before_arithmetic() {
    let pre = input(true);
    let bounds = scalar_to_batched(&incoming(true)).unwrap();
    let gamma = arr1(&[2.0, -0.5]);
    let beta = arr1(&[0.25, -0.125]);
    let budget = WorkBudget::new(None, 1 << 20, 1 << 20, 1_000_000, 4, Duration::ZERO).unwrap();
    let before = budget.receipt().unwrap();
    geometry::reset_constructions();
    let _environment = RefusalScope::new(true);
    for centered in [false, true] {
        refused(candidate_batched(
            &bounds,
            &pre,
            &gamma,
            Some(&beta),
            3.0,
            centered,
            Some(&budget),
        ));
        refused(refine_batched(
            &bounds,
            bounds.clone(),
            &pre,
            &gamma,
            Some(&beta),
            3.0,
            centered,
        ));
        refused(Geometry::new(
            &[-1.0, 1.0],
            &[-1.0, 1.0],
            3.0,
            centered,
            Some(&budget),
        ));
    }
    refused(geometry::norm(
        &[I::point(f64::from_bits(1))],
        Some(&budget),
    ));
    assert_eq!(geometry::constructions(), 0);
    assert_eq!(
        budget.receipt().unwrap(),
        before,
        "environment refusal creates neither geometry nor a fresh allowance"
    );
}

fn rat(x: f32) -> BigRational {
    BigRational::from_float(ny_core::f32_to_f64_exact(x)).unwrap()
}

fn contains_original(result: &LinearBounds, original: &LinearBounds, beta: Option<&[f32]>) {
    // Original point mean0, variance1 and epsilon3 give denominator exactly2.
    let x = [-1.0, 1.0];
    let gamma = [2.0, -0.5];
    for upper in [false, true] {
        let (a, b, e) = if upper {
            (
                original.upper_a(),
                original.upper_b(),
                original.upper_a_err(),
            )
        } else {
            (
                original.lower_a(),
                original.lower_b(),
                original.lower_a_err(),
            )
        };
        let mut intended = rat(b[0]);
        for k in 0..2 {
            let y = rat(gamma[k]) * rat(x[k]) / num_bigint::BigInt::from(2)
                + rat(beta.map_or(0.0, |b| b[k]));
            intended += rat(a[[0, k]]) * &y;
            let error = rat(e.unwrap()[[0, k]]) * y.abs();
            if upper {
                intended += error;
            } else {
                intended -= error;
            }
        }
        let (a, b) = if upper {
            (result.upper_a(), result.upper_b())
        } else {
            (result.lower_a(), result.lower_b())
        };
        let value = (0..2).fold(rat(b[0]), |sum, k| sum + rat(a[[0, k]]) * rat(x[k]));
        if upper {
            assert!(value >= intended);
        } else {
            assert!(value <= intended);
        }
    }
}

#[test]
fn radial_environment_normal_default_and_direct_routes_keep_original_rational_controls() {
    let _environment = RefusalScope::new(false);
    ny_core::require_f64_interval_proof_environment().unwrap();
    let original = incoming(true);
    let pre = input(false);
    let batched = scalar_to_batched(&original).unwrap();
    let batched_pre = input(true);
    let beta = [0.25, -0.125];
    let layer = LayerNormLayer::new(arr1(&[2.0, -0.5]), arr1(&beta), 3.0).unwrap();
    let rms = RmsNormLayer::new(arr1(&[2.0, -0.5]), 3.0).unwrap();
    contains_original(
        &layer.propagate_linear_with_bounds(&original, &pre).unwrap(),
        &original,
        Some(&beta),
    );
    contains_original(
        &rms.propagate_linear_with_bounds(&original, &pre).unwrap(),
        &original,
        None,
    );
    contains_original(
        &batched_to_scalar(
            &layer
                .propagate_linear_batched_with_bounds(&batched, &batched_pre)
                .unwrap(),
        )
        .unwrap(),
        &original,
        Some(&beta),
    );
    contains_original(
        &batched_to_scalar(
            &rms.propagate_linear_batched_with_bounds(&batched, &batched_pre)
                .unwrap(),
        )
        .unwrap(),
        &original,
        None,
    );
    contains_original(
        &layer
            .propagate_radial_linear_with_bounds(&original, &pre, None)
            .unwrap(),
        &original,
        Some(&beta),
    );
    contains_original(
        &rms.propagate_radial_linear_with_bounds(&original, &pre, None)
            .unwrap(),
        &original,
        None,
    );
}
