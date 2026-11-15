// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//! Present strided coefficient errors must retain their mathematical meaning.
use super::*;
use crate::layers::{LayerNormLayer, RmsNormLayer};
use ndarray::{arr1, ShapeBuilder};
use num_rational::BigRational;
use std::time::{Duration, Instant};

const C: [[f32; 4]; 2] = [[-1.0, -0.5, -0.25, -0.125], [-0.5, -0.125, -0.25, -0.75]];
const E: [[f32; 4]; 2] = [[2.0, 1.0, 0.5, 0.25], [0.375, 0.5, 0.25, 0.125]];

fn incoming(groups: usize, fortran_errors: bool) -> BatchedLinearBounds {
    let a = ArrayD::from_shape_fn(IxDyn(&[groups, 2, 4]), |i| C[i[1]][i[2]]);
    let b = ArrayD::from_shape_fn(IxDyn(&[groups, 2]), |i| (i[0] + i[1]) as f32 * 0.125);
    let mut bounds =
        BatchedLinearBounds::new(a.clone(), b.clone(), a, b, vec![groups, 4], vec![groups, 2])
            .unwrap();
    let e = ArrayD::from_shape_fn(IxDyn(&[groups, 2, 4]).f(), |i| E[i[1]][i[2]]);
    let e = if fortran_errors {
        e
    } else {
        e.as_standard_layout().to_owned()
    };
    bounds.lower_a_err = Some(e.clone());
    bounds.upper_a_err = Some(e);
    bounds.validate_internal_shapes().unwrap();
    bounds.validate_no_nan().unwrap();
    bounds
}
fn boxed(groups: usize, lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[groups, 4]), lower.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[groups, 4]), upper.to_vec()).unwrap(),
    )
    .unwrap()
}
fn q(value: f32) -> BigRational {
    BigRational::from_float(value as f64).unwrap()
}
fn at(
    bounds: &BatchedLinearBounds,
    point: &[f32],
    group: usize,
    output: usize,
    lower: bool,
) -> BigRational {
    let (a, b) = if lower {
        (&bounds.lower_a, &bounds.lower_b)
    } else {
        (&bounds.upper_a, &bounds.upper_b)
    };
    (0..4).fold(q(b[[group, output]]), |sum, k| {
        sum + q(a[[group, output, k]]) * q(point[k])
    })
}
fn assert_same(a: &BatchedLinearBounds, b: &BatchedLinearBounds) {
    for (left, right) in [
        (&a.lower_a, &b.lower_a),
        (&a.upper_a, &b.upper_a),
        (&a.lower_b, &b.lower_b),
        (&a.upper_b, &b.upper_b),
    ] {
        assert_eq!(left.shape(), right.shape());
        assert_eq!(
            left.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            right.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    for (left, right) in [
        (&a.lower_a_err, &b.lower_a_err),
        (&a.upper_a_err, &b.upper_a_err),
    ] {
        assert_eq!(
            left.as_ref().map(|x| x.iter().copied().collect::<Vec<_>>()),
            right
                .as_ref()
                .map(|x| x.iter().copied().collect::<Vec<_>>())
        );
    }
}
/// The sampled points have x_k in {-1,1}; for centered normalization their
/// mean is zero. With authored epsilon3, variance1 gives the exact denominator2.
/// This oracle uses original coefficients, shifts and coefficient uncertainties.
fn intended(
    bounds: &BatchedLinearBounds,
    point: &[f32],
    gamma: &Array1<f32>,
    beta: Option<&Array1<f32>>,
    group: usize,
    output: usize,
    lower: bool,
) -> BigRational {
    let (a, b, e) = if lower {
        (&bounds.lower_a, &bounds.lower_b, &bounds.lower_a_err)
    } else {
        (&bounds.upper_a, &bounds.upper_b, &bounds.upper_a_err)
    };
    let mut result = q(b[[group, output]]);
    for k in 0..4 {
        let y = q(gamma[k]) * q(point[k]) / num_bigint::BigInt::from(2)
            + beta.map_or_else(|| q(0.0), |beta| q(beta[k]));
        result += q(a[[group, output, k]]) * &y;
        if let Some(e) = e {
            let abs_y = if y < q(0.0) { -y } else { y };
            let charge = q(e[[group, output, k]]) * abs_y;
            if lower {
                result -= charge;
            } else {
                result += charge;
            }
        }
    }
    result
}

#[test]
fn radial_layout_f_order_error_plane_replays_exact_rational_point_bounds() {
    let strided = incoming(1, true);
    assert!(strided.lower_a.is_standard_layout());
    let original_error = strided.lower_a_err.as_ref().unwrap();
    assert!(
        original_error
            .view()
            .into_shape_with_order((1, 2, 4))
            .is_err(),
        "the old baseline refused F-order carriers before reaching its as_slice omission"
    );
    assert!(
        original_error
            .slice(ndarray::s![0, 0, ..])
            .as_slice()
            .is_none(),
        "present logical error row is strided"
    );
    let contiguous = incoming(1, false);
    let point = [1.0, -1.0, 1.0, -1.0];
    let pre = boxed(1, &point, &point);
    let gamma = arr1(&[1.0, -2.0, 0.5, 1.5]);
    let beta = arr1(&[0.25, -0.5, 0.75, -1.0]);
    for centered in [false, true] {
        let checked =
            baseline::checked_baseline(&strided, &gamma, Some(&beta), 3.0, &pre, centered, None)
                .unwrap()
                .bounds;
        let reference =
            baseline::checked_baseline(&contiguous, &gamma, Some(&beta), 3.0, &pre, centered, None)
                .unwrap()
                .bounds;
        assert_same(&checked, &reference);
        let mut omitted = strided.clone();
        omitted.lower_a_err = None;
        omitted.upper_a_err = None;
        let omitted =
            baseline::checked_baseline(&omitted, &gamma, Some(&beta), 3.0, &pre, centered, None)
                .unwrap()
                .bounds;
        for output in 0..2 {
            let actual_lower = intended(&strided, &point, &gamma, Some(&beta), 0, output, true);
            let actual_upper = intended(&strided, &point, &gamma, Some(&beta), 0, output, false);
            assert!(at(&checked, &point, 0, output, true) <= actual_lower);
            assert!(at(&checked, &point, 0, output, false) >= actual_upper);
            assert!(
                at(&omitted, &point, 0, output, true) > actual_lower,
                "silently omitted errors exclude an admissible coefficient perturbation"
            );
            assert!(at(&omitted, &point, 0, output, false) < actual_upper);
        }
    }
}

#[test]
fn radial_layout_error_planes_match_on_multigroup_box_vertices() {
    let strided = incoming(2, true);
    let contiguous = incoming(2, false);
    let pre = boxed(2, &[-1.0; 8], &[1.0; 8]);
    let gamma = arr1(&[-1.25, 2.0, 0.5, -0.75]);
    let beta = arr1(&[0.25, -0.5, 0.75, -1.0]);
    for centered in [false, true] {
        let checked =
            baseline::checked_baseline(&strided, &gamma, Some(&beta), 3.0, &pre, centered, None)
                .unwrap()
                .bounds;
        let reference =
            baseline::checked_baseline(&contiguous, &gamma, Some(&beta), 3.0, &pre, centered, None)
                .unwrap()
                .bounds;
        assert_same(&checked, &reference);
        for bits in 0..16u32 {
            if centered && bits.count_ones() != 2 {
                continue;
            }
            let point =
                std::array::from_fn::<_, 4, _>(|k| if bits & (1 << k) == 0 { -1.0 } else { 1.0 });
            for group in 0..2 {
                for output in 0..2 {
                    assert!(
                        at(&checked, &point, group, output, true)
                            <= intended(&strided, &point, &gamma, Some(&beta), group, output, true)
                    );
                    assert!(
                        at(&checked, &point, group, output, false)
                            >= intended(
                                &strided,
                                &point,
                                &gamma,
                                Some(&beta),
                                group,
                                output,
                                false
                            )
                    );
                }
            }
        }
    }
}

#[test]
fn radial_layout_default_layernorm_and_rmsnorm_routes_keep_incoming_errors() {
    let strided = incoming(1, true);
    let contiguous = incoming(1, false);
    let point = [1.0, -1.0, 1.0, -1.0];
    let pre = boxed(1, &point, &point);
    let gamma = arr1(&[1.0, -2.0, 0.5, 1.5]);
    let beta = arr1(&[0.25, -0.5, 0.75, -1.0]);
    let ln = LayerNormLayer::new(gamma.clone(), beta.clone(), 3.0).unwrap();
    let rms = RmsNormLayer::new(gamma.clone(), 3.0).unwrap();
    for centered in [false, true] {
        let checked = if centered {
            ln.propagate_linear_batched_with_bounds(&strided, &pre)
        } else {
            rms.propagate_linear_batched_with_bounds(&strided, &pre)
        }
        .unwrap();
        let reference = if centered {
            ln.propagate_linear_batched_with_bounds(&contiguous, &pre)
        } else {
            rms.propagate_linear_batched_with_bounds(&contiguous, &pre)
        }
        .unwrap();
        assert_same(&checked, &reference);
        let beta = centered.then_some(&beta);
        for output in 0..2 {
            assert!(
                at(&checked, &point, 0, output, true)
                    <= intended(&strided, &point, &gamma, beta, 0, output, true)
            );
            assert!(
                at(&checked, &point, 0, output, false)
                    >= intended(&strided, &point, &gamma, beta, 0, output, false)
            );
        }
    }
}

#[test]
fn radial_layout_dominance_preserves_a_present_strided_error_envelope() {
    let mut incoming = incoming(1, true);
    incoming.lower_a_err = None;
    incoming.upper_a_err = None;
    let point = [1.0, -1.0, 1.0, -1.0];
    let pre = boxed(1, &point, &point);
    let gamma = arr1(&[1.0; 4]);
    let candidate = candidate_batched(&incoming, &pre, &gamma, None, 3.0, false, None).unwrap();
    let mut strided = candidate;
    strided.lower_b.mapv_inplace(|b| b - 0.25);
    strided.upper_b.mapv_inplace(|b| b + 0.25);
    let errors = ArrayD::from_shape_fn(IxDyn(&[1, 2, 4]).f(), |_| 0.5);
    strided.lower_a_err = Some(errors.clone());
    strided.upper_a_err = Some(errors);
    let mut contiguous = strided.clone();
    contiguous.lower_a_err = strided
        .lower_a_err
        .as_ref()
        .map(|a| a.as_standard_layout().to_owned());
    contiguous.upper_a_err = strided
        .upper_a_err
        .as_ref()
        .map(|a| a.as_standard_layout().to_owned());
    let checked = calculate(
        &incoming,
        Some(&strided),
        &pre,
        &gamma,
        None,
        3.0,
        false,
        None,
    )
    .unwrap();
    let reference = calculate(
        &incoming,
        Some(&contiguous),
        &pre,
        &gamma,
        None,
        3.0,
        false,
        None,
    )
    .unwrap();
    assert_same(&checked, &reference);
    assert_same(&checked, &strided);
    let mut omitted = strided.clone();
    omitted.lower_a_err = None;
    omitted.upper_a_err = None;
    let selected = calculate(
        &incoming,
        Some(&omitted),
        &pre,
        &gamma,
        None,
        3.0,
        false,
        None,
    )
    .unwrap();
    for output in 0..2 {
        assert!(at(&selected, &point, 0, output, true) > at(&checked, &point, 0, output, true));
        assert!(at(&selected, &point, 0, output, false) < at(&checked, &point, 0, output, false));
    }
}

#[test]
fn radial_layout_invalid_present_errors_refuse_and_optional_infinity_stays_conservative() {
    let pre = boxed(1, &[1.0, -1.0, 1.0, -1.0], &[1.0, -1.0, 1.0, -1.0]);
    let gamma = arr1(&[1.0; 4]);
    for invalid in [-0.25, f32::NAN, f32::INFINITY] {
        let mut bounds = incoming(1, true);
        bounds.lower_a_err.as_mut().unwrap()[[0, 0, 1]] = invalid;
        assert!(baseline::checked_baseline(&bounds, &gamma, None, 3.0, &pre, false, None).is_err());
    }
    let bounds = incoming(1, false);
    let mut baseline = bounds.clone();
    baseline.lower_a_err = Some(ArrayD::from_shape_fn(IxDyn(&[1, 2, 4]).f(), |_| {
        f32::INFINITY
    }));
    baseline.upper_a_err = baseline.lower_a_err.clone();
    baseline.validate_no_nan().unwrap();
    let result = refine_batched(&bounds, baseline.clone(), &pre, &gamma, None, 3.0, false).unwrap();
    assert_same(&result, &baseline);
    let expired = WorkBudget::new(
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
        1 << 20,
        1 << 20,
        1_000_000,
        10,
        Duration::ZERO,
    )
    .unwrap();
    let array = ArrayD::from_shape_fn(IxDyn(&[1, 2, 1024]).f(), |_| 0.5);
    assert!(matches!(
        logical_error_row(Some(array.slice(ndarray::s![0, 0, ..])), Some(&expired)),
        Err(NyError::DeadlineExceeded(_))
    ));
}
