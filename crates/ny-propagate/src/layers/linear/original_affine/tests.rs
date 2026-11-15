// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::LinearBounds;
use ndarray::{array, Array1, ArrayD, IxDyn};
use num_bigint::BigInt;
use num_rational::BigRational as Q;
use num_traits::{Signed, Zero};
use ny_tensor::BoundedTensor;
use std::{sync::Arc, time::Duration};

fn combine(requests: &[WorkRequest]) -> WorkRequest {
    requests
        .iter()
        .fold(WorkRequest::default(), |mut total, request| {
            total.scratch_bytes = total.scratch_bytes.max(request.scratch_bytes);
            total.persistent_bytes = total
                .persistent_bytes
                .checked_add(request.persistent_bytes)
                .unwrap();
            total.work = total.work.checked_add(request.work).unwrap();
            total.proposals = total.proposals.checked_add(request.proposals).unwrap();
            total
        })
}
fn admitted(request: WorkRequest) -> (WorkBudget, OriginalAffinePhase) {
    let budget = WorkBudget::new(
        None,
        request.scratch_bytes,
        request.persistent_bytes,
        request.work,
        request.proposals,
        Duration::ZERO,
    )
    .unwrap();
    let phase = OriginalAffinePhase::admit(&budget, LeaseClass::Optional, request, None).unwrap();
    (budget, phase)
}

/// Original bit interpretation without the kernel's widening helper or any
/// floating point casts/reductions. Includes binary32 subnormal low bits.
fn q(value: f32) -> Q {
    let bits = value.to_bits();
    let exponent = (bits >> 23) & 0xff;
    assert_ne!(exponent, 0xff);
    let fraction = bits & 0x7f_ffff;
    let (mantissa, power) = if exponent == 0 {
        (fraction, -149i32)
    } else {
        ((1 << 23) | fraction, exponent as i32 - 127 - 23)
    };
    let mut numerator = BigInt::from(mantissa);
    if bits >> 31 != 0 {
        numerator = -numerator;
    }
    if power >= 0 {
        Q::from_integer(numerator << power as usize)
    } else {
        Q::new(numerator, BigInt::from(1) << (-power) as usize)
    }
}

#[test]
fn original_affine_ibp_signed_positions_and_cancellation_rational() {
    let large = 2f32.powi(30);
    let small = 2f32.powi(-30);
    let weights = array![[large, 1.0, -large], [-large, -0.75, large]];
    let bias = array![0.0, -0.125];
    let lower = array![[1.0, small, 1.0], [-1.0, -0.5, -1.0]];
    let upper = array![[1.0, small, 1.0], [-0.75, 0.25, -0.5]];
    let input = BoundedTensor::new(lower.clone().into_dyn(), upper.clone().into_dyn()).unwrap();
    let request = combine(&[
        OriginalAffineView::validation_request(2, 3, true).unwrap(),
        ibp_request(2, 3, input.shape()).unwrap(),
    ]);
    let (_, mut phase) = admitted(request);
    let parameters =
        OriginalAffineView::validate(weights.view(), Some(bias.view()), &mut phase).unwrap();
    assert_eq!(parameters.weight.as_ptr(), weights.as_ptr());
    let bounds = ibp(parameters, &input, &mut phase).unwrap();
    for position in 0..2 {
        for row in 0..2 {
            let mut lo = q(bias[row]);
            let mut hi = lo.clone();
            for column in 0..3 {
                let a = q(weights[[row, column]]) * q(lower[[position, column]]);
                let b = q(weights[[row, column]]) * q(upper[[position, column]]);
                lo += a.clone().min(b.clone());
                hi += a.max(b);
            }
            assert!(q(bounds.lower()[[position, row]]) <= lo);
            assert!(q(bounds.upper()[[position, row]]) >= hi);
        }
    }
    // Liveness: even widening only the final cast of this naive cancellation
    // would miss the exact positive original term.
    let naive = (f64::from(large) + f64::from(small)) - f64::from(large);
    assert_eq!(naive, 0.0);
    assert!(q(ny_tensor::next_up_f32(naive as f32)) < q(small));
    assert!(q(bounds.upper()[[0, 0]]) >= q(small));
}

#[test]
fn original_affine_ibp_subnormal_original_bits_and_negative_constant() {
    let weights = array![[f32::from_bits(1)], [0.0]];
    let bias = array![0.0, -2.0];
    let input = BoundedTensor::concrete(array![1.0].into_dyn()).unwrap();
    let (_, mut phase) = admitted(combine(&[
        OriginalAffineView::validation_request(2, 1, true).unwrap(),
        ibp_request(2, 1, &[1]).unwrap(),
    ]));
    let parameters =
        OriginalAffineView::validate(weights.view(), Some(bias.view()), &mut phase).unwrap();
    let result = ibp(parameters, &input, &mut phase).unwrap();
    assert!(q(result.lower()[0]) <= q(f32::from_bits(1)));
    assert!(q(result.upper()[0]) >= q(f32::from_bits(1)));
    assert!(result.upper()[0] >= f32::MIN_POSITIVE);
    assert_eq!(result.lower()[1].to_bits(), (-2.0f32).to_bits());
    assert_eq!(result.upper()[1].to_bits(), (-2.0f32).to_bits());
}

fn incoming() -> LinearBounds {
    let large = 2f32.powi(30);
    LinearBounds::from_prevalidated_parts_with_optional_err(
        array![
            [large, 1.0, -large, -0.75, 0.125, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ],
        array![-0.5, 1.25],
        array![
            [-0.5, 0.75, 0.25, 1.25, -2.0, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ],
        array![0.75, 2.5],
        Some(array![
            [0.0, 0.125, 0.0, 0.25, 0.0, 0.0625],
            [0.25, 0.5, 0.125, 0.5, 0.25, 0.5]
        ]),
        Some(array![
            [0.0625, 0.0, 0.125, 0.0, 0.25, 0.0],
            [0.5, 0.125, 0.25, 0.125, 0.5, 0.25]
        ]),
    )
    .unwrap()
}

#[test]
fn original_affine_crown_signed_errors_biases_and_vertices_rational() {
    let weights = array![[1.0, -0.25], [2f32.powi(-30), 0.5], [1.0, 1.25]];
    let bias = array![-2.0, 0.125, -0.5];
    let incoming = incoming();
    let (_, mut phase) = admitted(combine(&[
        OriginalAffineView::validation_request(3, 2, true).unwrap(),
        crown_request(3, 2, 2, 6).unwrap(),
    ]));
    let parameters =
        OriginalAffineView::validate(weights.view(), Some(bias.view()), &mut phase).unwrap();
    let result = crown(parameters, &incoming, &mut phase).unwrap();
    for row in 0..2 {
        for (source, source_error, target, target_error) in [
            (
                incoming.lower_a(),
                incoming.lower_a_err().unwrap(),
                result.lower_a(),
                result.lower_a_err().unwrap(),
            ),
            (
                incoming.upper_a(),
                incoming.upper_a_err().unwrap(),
                result.upper_a(),
                result.upper_a_err().unwrap(),
            ),
        ] {
            for position in 0..2 {
                for j in 0..2 {
                    let mut central = Q::zero();
                    let mut propagated = Q::zero();
                    for k in 0..3 {
                        central += q(source[[row, position * 3 + k]]) * q(weights[[k, j]]);
                        propagated +=
                            q(source_error[[row, position * 3 + k]]) * q(weights[[k, j]]).abs();
                    }
                    let column = position * 2 + j;
                    assert!(
                        q(target_error[[row, column]])
                            >= (q(target[[row, column]]) - central).abs() + propagated
                    );
                }
            }
        }
        let mut want_lower = q(incoming.lower_b()[row]);
        let mut want_upper = q(incoming.upper_b()[row]);
        for position in 0..2 {
            for k in 0..3 {
                let column = position * 3 + k;
                want_lower += q(incoming.lower_a()[[row, column]]) * q(bias[k])
                    - q(incoming.lower_a_err().unwrap()[[row, column]]) * q(bias[k]).abs();
                want_upper += q(incoming.upper_a()[[row, column]]) * q(bias[k])
                    + q(incoming.upper_a_err().unwrap()[[row, column]]) * q(bias[k]).abs();
            }
        }
        assert!(q(result.lower_b()[row]) <= want_lower);
        assert!(q(result.upper_b()[row]) >= want_upper);
        // The second row is error-only. Dropping it or E*|bias| fails these
        // original inequalities even when every central input coefficient is zero.
        for vertex in 0..16 {
            let x: Vec<_> = (0..4)
                .map(|j| q(if vertex & (1 << j) == 0 { -0.75 } else { 0.5 }))
                .collect();
            let mut y = Vec::new();
            for position in 0..2 {
                for k in 0..3 {
                    let mut value = q(bias[k]);
                    for j in 0..2 {
                        value += q(weights[[k, j]]) * &x[position * 2 + j];
                    }
                    y.push(value);
                }
            }
            let mut old_lower = q(incoming.lower_b()[row]);
            let mut old_upper = q(incoming.upper_b()[row]);
            for (column, value) in y.iter().enumerate() {
                old_lower += q(incoming.lower_a()[[row, column]]) * value
                    - q(incoming.lower_a_err().unwrap()[[row, column]]) * value.abs();
                old_upper += q(incoming.upper_a()[[row, column]]) * value
                    + q(incoming.upper_a_err().unwrap()[[row, column]]) * value.abs();
            }
            let mut new_lower = q(result.lower_b()[row]);
            let mut new_upper = q(result.upper_b()[row]);
            for (column, value) in x.iter().enumerate() {
                new_lower += q(result.lower_a()[[row, column]]) * value
                    - q(result.lower_a_err().unwrap()[[row, column]]) * value.abs();
                new_upper += q(result.upper_a()[[row, column]]) * value
                    + q(result.upper_a_err().unwrap()[[row, column]]) * value.abs();
            }
            assert!(new_lower <= old_lower);
            assert!(new_upper >= old_upper);
        }
    }
    assert!(result
        .lower_a_err()
        .unwrap()
        .row(1)
        .iter()
        .any(|&error| error > 0.0));
    assert!(result.lower_b()[1] < incoming.lower_b()[1]);
    assert!(result.upper_b()[1] > incoming.upper_b()[1]);
}

#[test]
fn original_affine_crown_zero_carrier_keeps_nonzero_incoming_bias() {
    let weights = array![[-2.0, 3.0]];
    let bias = array![-4.0];
    let carrier =
        LinearBounds::new(array![[0.0]], array![-1.25], array![[0.0]], array![2.5]).unwrap();
    let (_, mut phase) = admitted(combine(&[
        OriginalAffineView::validation_request(1, 2, true).unwrap(),
        crown_request(1, 2, 1, 1).unwrap(),
    ]));
    let parameters =
        OriginalAffineView::validate(weights.view(), Some(bias.view()), &mut phase).unwrap();
    let result = crown(parameters, &carrier, &mut phase).unwrap();
    assert_eq!(result.lower_b()[0].to_bits(), (-1.25f32).to_bits());
    assert_eq!(result.upper_b()[0].to_bits(), 2.5f32.to_bits());
    assert!(result
        .lower_a()
        .iter()
        .chain(result.upper_a().iter())
        .chain(result.lower_a_err().unwrap().iter())
        .chain(result.upper_a_err().unwrap().iter())
        .all(|&value| value == 0.0));
}

#[test]
fn original_affine_sparse_low_bits_positions_and_original_margins_rational() {
    let weights = array![
        [
            f32::from_bits(1.0f32.to_bits() + 1),
            2f32.powi(30),
            f32::from_bits(1)
        ],
        [1.0, f32::from_bits(1), 0.0],
        [-1.0, -2f32.powi(30), -f32::from_bits(1)]
    ];
    let bias = array![-2.0, f32::from_bits((-2.0f32).to_bits() - 1), 0.125];
    let differences = [
        LogitDifference {
            target: 0,
            competitor: 1,
            position: 2,
        },
        LogitDifference {
            target: 1,
            competitor: 2,
            position: 0,
        },
    ];
    let (_, mut phase) = admitted(combine(&[
        OriginalAffineView::validation_request(3, 3, true).unwrap(),
        sparse_request(3, 3, 2, 3).unwrap(),
    ]));
    let parameters =
        OriginalAffineView::validate(weights.view(), Some(bias.view()), &mut phase).unwrap();
    let result = seed_logit_differences(parameters, 3, &differences, &mut phase).unwrap();
    assert_eq!(result.lower_a().dim(), (2, 9));
    for (row, difference) in differences.iter().enumerate() {
        for position in 0..3 {
            for j in 0..3 {
                let column = position * 3 + j;
                if position != difference.position {
                    assert_eq!(result.lower_a()[[row, column]].to_bits(), 0.0f32.to_bits());
                    assert_eq!(
                        result.lower_a_err().unwrap()[[row, column]].to_bits(),
                        0.0f32.to_bits()
                    );
                    continue;
                }
                let exact =
                    q(weights[[difference.target, j]]) - q(weights[[difference.competitor, j]]);
                assert!(
                    q(result.lower_a_err().unwrap()[[row, column]])
                        >= (q(result.lower_a()[[row, column]]) - &exact).abs()
                );
                assert!(
                    q(result.upper_a_err().unwrap()[[row, column]])
                        >= (q(result.upper_a()[[row, column]]) - exact).abs()
                );
            }
        }
        let exact_bias = q(bias[difference.target]) - q(bias[difference.competitor]);
        assert!(q(result.lower_b()[row]) <= exact_bias);
        assert!(q(result.upper_b()[row]) >= exact_bias);
        for vertex in 0..8 {
            let point: Vec<_> = (0..3)
                .map(|j| q(if vertex & (1 << j) == 0 { -1.0 } else { 0.75 }))
                .collect();
            let mut original = q(bias[difference.target]) - q(bias[difference.competitor]);
            let mut lower = q(result.lower_b()[row]);
            let mut upper = q(result.upper_b()[row]);
            for j in 0..3 {
                original += (q(weights[[difference.target, j]])
                    - q(weights[[difference.competitor, j]]))
                    * &point[j];
                let column = difference.position * 3 + j;
                lower += q(result.lower_a()[[row, column]]) * &point[j]
                    - q(result.lower_a_err().unwrap()[[row, column]]) * point[j].abs();
                upper += q(result.upper_a()[[row, column]]) * &point[j]
                    + q(result.upper_a_err().unwrap()[[row, column]]) * point[j].abs();
            }
            assert!(lower <= original);
            assert!(upper >= original);
        }
    }
    // The head's enormous exponent gap is genuine, and the lowest original
    // bits remain in the rational oracle rather than disappearing in f64.
    assert_ne!(q(weights[[0, 1]]) - q(weights[[1, 1]]), q(weights[[0, 1]]));
    assert!(result.lower_a_err().unwrap()[[0, 8]] >= f32::MIN_POSITIVE);
}

#[test]
fn original_affine_poisoned_metadata_originals_and_shapes_refuse() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let weights = array![[bad]];
        let (_, mut phase) = admitted(OriginalAffineView::validation_request(1, 1, false).unwrap());
        assert!(OriginalAffineView::validate(weights.view(), None, &mut phase).is_err());
    }
    let weights = array![[0.0]];
    for bad in [f32::NAN, f32::INFINITY] {
        let bias = array![bad];
        let (_, mut phase) = admitted(OriginalAffineView::validation_request(1, 1, true).unwrap());
        // Validate originals before algebraic zero shortcuts can hide poison.
        assert!(
            OriginalAffineView::validate(weights.view(), Some(bias.view()), &mut phase).is_err()
        );
    }
    for bad in [-0.25, f32::NAN, f32::INFINITY] {
        let mut carrier =
            LinearBounds::new(array![[0.0]], array![-1.0], array![[0.0]], array![2.0]).unwrap();
        carrier.lower_a_err = Some(array![[bad]]);
        let (_, mut phase) = admitted(combine(&[
            OriginalAffineView::validation_request(1, 1, false).unwrap(),
            crown_request(1, 1, 1, 1).unwrap(),
        ]));
        let parameters = OriginalAffineView::validate(weights.view(), None, &mut phase).unwrap();
        // Poison is checked even when original weights and central A are zero.
        assert!(crown(parameters, &carrier, &mut phase).is_err());
    }
    let (_, mut phase) = admitted(combine(&[
        OriginalAffineView::validation_request(1, 1, false).unwrap(),
        sparse_request(1, 1, 1, 1).unwrap(),
    ]));
    let parameters = OriginalAffineView::validate(weights.view(), None, &mut phase).unwrap();
    assert!(seed_logit_differences(
        parameters,
        1,
        &[LogitDifference {
            target: 1,
            competitor: 0,
            position: 0
        }],
        &mut phase
    )
    .is_err());
    assert!(crown_request(0, 1, 1, 1).is_err());
    assert!(crown_request(2, 1, 1, 3).is_err());
    assert!(ibp_request(1, 1, &[]).is_err());
    assert!(sparse_request(1, usize::MAX, usize::MAX, 2).is_err());

    let mut wrong_error_shape =
        LinearBounds::new(array![[0.0]], array![-1.0], array![[0.0]], array![2.0]).unwrap();
    wrong_error_shape.upper_a_err = Some(array![[0.0, 0.0]]);
    let (_, mut phase) = admitted(combine(&[
        OriginalAffineView::validation_request(1, 1, false).unwrap(),
        crown_request(1, 1, 1, 1).unwrap(),
    ]));
    let parameters = OriginalAffineView::validate(weights.view(), None, &mut phase).unwrap();
    assert!(crown(parameters, &wrong_error_shape, &mut phase).is_err());
}

#[test]
fn original_affine_overflow_and_nonfinite_input_profiles_are_controlled() {
    let weights = array![[f32::MAX]];
    let input = BoundedTensor::concrete(array![2.0].into_dyn()).unwrap();
    let (_, mut phase) = admitted(combine(&[
        OriginalAffineView::validation_request(1, 1, false).unwrap(),
        ibp_request(1, 1, &[1]).unwrap(),
    ]));
    let parameters = OriginalAffineView::validate(weights.view(), None, &mut phase).unwrap();
    let bounds = ibp(parameters, &input, &mut phase).unwrap();
    assert_eq!(bounds.lower()[0], f32::MAX);
    assert_eq!(bounds.upper()[0], f32::INFINITY);
    assert!(q(bounds.lower()[0]) <= q(f32::MAX) * q(2.0));

    let carrier =
        LinearBounds::new(array![[2.0]], array![0.0], array![[2.0]], array![0.0]).unwrap();
    let (_, mut phase) = admitted(combine(&[
        OriginalAffineView::validation_request(1, 1, false).unwrap(),
        crown_request(1, 1, 1, 1).unwrap(),
    ]));
    let parameters = OriginalAffineView::validate(weights.view(), None, &mut phase).unwrap();
    assert!(matches!(
        crown(parameters, &carrier, &mut phase),
        Err(NyError::NumericalInstability(_))
    ));

    let input = BoundedTensor::new_allow_infinite(
        array![f32::NEG_INFINITY].into_dyn(),
        array![f32::INFINITY].into_dyn(),
    )
    .unwrap();
    let (_, mut phase) = admitted(combine(&[
        OriginalAffineView::validation_request(1, 1, false).unwrap(),
        ibp_request(1, 1, &[1]).unwrap(),
    ]));
    let parameters = OriginalAffineView::validate(weights.view(), None, &mut phase).unwrap();
    assert!(matches!(
        ibp(parameters, &input, &mut phase),
        Err(NyError::NumericalInstability(_))
    ));
}

#[test]
fn original_affine_shared_admission_deadline_work_and_scratch_controls() {
    let request = OriginalAffineView::validation_request(1, 1, false).unwrap();
    let budget = WorkBudget::new(
        None,
        request.scratch_bytes,
        0,
        request.work - 1,
        0,
        Duration::ZERO,
    )
    .unwrap();
    assert!(OriginalAffinePhase::admit(&budget, LeaseClass::Optional, request, None).is_err());
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    let budget = WorkBudget::new(
        None,
        request.scratch_bytes,
        0,
        request.work,
        0,
        Duration::ZERO,
    )
    .unwrap();
    assert!(OriginalAffinePhase::admit(
        &budget,
        LeaseClass::Optional,
        request,
        Some(Instant::now())
    )
    .is_err());
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);

    let weights = Array2::<f32>::ones((1, 512));
    let input = BoundedTensor::concrete(ArrayD::ones(IxDyn(&[512]))).unwrap();
    let (_, mut phase) = admitted(combine(&[
        OriginalAffineView::validation_request(1, 512, false).unwrap(),
        ibp_request(1, 512, &[512]).unwrap(),
    ]));
    let parameters = OriginalAffineView::validate(weights.view(), None, &mut phase).unwrap();
    phase.fail_after = Some(phase.spent + 6000);
    assert!(matches!(
        ibp(parameters, &input, &mut phase),
        Err(NyError::DeadlineExceeded(_))
    ));
    let zero_scratch = WorkRequest {
        scratch_bytes: 0,
        ..request
    };
    let (_, mut phase) = admitted(zero_scratch);
    let weights = array![[1.0]];
    assert!(OriginalAffineView::validate(weights.view(), None, &mut phase).is_err());
}

#[test]
fn original_affine_phase_reservation_retains_actual_outputs_across_clones() {
    let weights = array![[1.0]];
    let input = BoundedTensor::concrete(array![-0.75].into_dyn()).unwrap();
    let request = combine(&[
        OriginalAffineView::validation_request(1, 1, false).unwrap(),
        ibp_request(1, 1, &[1]).unwrap(),
    ]);
    let (budget, mut phase) = admitted(request);
    let parameters = OriginalAffineView::validate(weights.view(), None, &mut phase).unwrap();
    let output = ibp(parameters, &input, &mut phase).unwrap();
    let owner = Arc::new((output, phase.commit().unwrap()));
    let cloned = owner.clone();
    drop(owner);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        request.persistent_bytes
    );
    assert!(cloned.0.lower()[0] <= -0.75 && cloned.0.upper()[0] >= -0.75);
    drop(cloned);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn original_affine_strided_original_parameters_preserve_logical_orientation() {
    let backing = array![[1.0, -2.0], [3.0, 0.5]];
    let weights = backing.slice(ndarray::s![.., ..;-1]);
    let input = BoundedTensor::concrete(array![-0.75, 1.25].into_dyn()).unwrap();
    let (_, mut phase) = admitted(combine(&[
        OriginalAffineView::validation_request(2, 2, false).unwrap(),
        ibp_request(2, 2, &[2]).unwrap(),
    ]));
    let parameters = OriginalAffineView::validate(weights, None, &mut phase).unwrap();
    let result = ibp(parameters, &input, &mut phase).unwrap();
    for row in 0..2 {
        let exact = q(weights[[row, 0]]) * q(-0.75) + q(weights[[row, 1]]) * q(1.25);
        assert!(q(result.lower()[row]) <= exact && q(result.upper()[row]) >= exact);
    }
}

#[test]
fn original_affine_owned_arrays_preserve_offset_strides_bits_and_rational_bounds() {
    let weights = array![
        [9.0, 8.0, 7.0, 6.0],
        [5.0, -0.0, f32::from_bits(1), -0.5],
        [4.0, 3.0, 2.0, 1.0],
        [0.0, f32::from_bits(0x8000_0001), 0.25, 2.0]
    ];
    let weight_origin = weights.as_ptr();
    let weights = weights.slice_move(ndarray::s![1..4;2, 1..4;-1]);
    let bias = array![9.0, -0.0, 8.0, f32::from_bits(1)];
    let bias_origin = bias.as_ptr();
    let bias = bias.slice_move(ndarray::s![1..;2]);
    assert_ne!(weights.as_ptr(), weight_origin);
    assert_ne!(bias.as_ptr(), bias_origin);
    assert!(!weights.is_standard_layout());
    assert!(!bias.is_standard_layout());
    assert!(weights.strides()[1] < 0);
    let input = BoundedTensor::concrete(array![-0.75, 1.25, -0.5].into_dyn()).unwrap();
    let request = combine(&[
        OriginalAffineView::validation_request(2, 3, true).unwrap(),
        ibp_request(2, 3, input.shape()).unwrap(),
    ]);
    let (_, mut phase) = admitted(request);
    let arrays = OriginalAffineArrays::validate(&weights, Some(&bias), &mut phase).unwrap();
    assert_eq!(arrays.in_features(), 3);
    assert_eq!(arrays.out_features(), 2);
    let spent = phase.spent;
    for _ in 0..8 {
        let view = arrays.view();
        assert_eq!(view.weight.as_ptr(), weights.as_ptr());
        assert_eq!(view.weight.dim(), weights.dim());
        assert_eq!(view.weight.strides(), weights.strides());
        assert_eq!(view.bias.unwrap().as_ptr(), bias.as_ptr());
        assert_eq!(view.bias.unwrap().strides(), bias.strides());
        for (&a, &b) in view.weight.iter().zip(weights.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (&a, &b) in view.bias.unwrap().iter().zip(bias.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
    assert_eq!(
        phase.spent, spent,
        "view reconstruction cannot repeat paid validation"
    );
    let bounds = ibp(arrays.view(), &input, &mut phase).unwrap();
    for row in 0..2 {
        let mut exact = q(bias[row]);
        for column in 0..3 {
            exact += q(weights[[row, column]]) * q(input.lower()[column]);
        }
        assert!(q(bounds.lower()[row]) <= exact);
        assert!(q(bounds.upper()[row]) >= exact);
    }
}

#[test]
fn original_affine_owned_arrays_no_bias_keep_original_low_bits() {
    let weights = array![[-0.0, f32::from_bits(1), f32::from_bits(0x8000_0001)]];
    let input = BoundedTensor::concrete(array![-1.0, 2.0, -3.0].into_dyn()).unwrap();
    let request = combine(&[
        OriginalAffineView::validation_request(1, 3, false).unwrap(),
        ibp_request(1, 3, input.shape()).unwrap(),
    ]);
    let (_, mut phase) = admitted(request);
    let arrays = OriginalAffineArrays::validate(&weights, None, &mut phase).unwrap();
    let view = arrays.view();
    assert!(view.bias.is_none());
    assert_eq!(view.weight.as_ptr(), weights.as_ptr());
    assert_eq!(view.weight[[0, 0]].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(view.weight[[0, 1]].to_bits(), 1);
    assert_eq!(view.weight[[0, 2]].to_bits(), 0x8000_0001);
    let result = ibp(view, &input, &mut phase).unwrap();
    let exact = q(f32::from_bits(1)) * q(2.0) + q(f32::from_bits(0x8000_0001)) * q(-3.0);
    assert!(q(result.lower()[0]) <= exact);
    assert!(q(result.upper()[0]) >= exact);
}

#[test]
fn original_affine_owned_arrays_keep_validation_poll_and_work_cadence() {
    let weights = Array2::ones((1, 2048));
    let bias = array![-0.0];
    let request = OriginalAffineView::validation_request(1, 2048, true).unwrap();
    for stop in [None, Some(0), Some(1), Some(4095), Some(4096), Some(5000)] {
        let (old_budget, mut old) = admitted(request);
        let (new_budget, mut new) = admitted(request);
        old.fail_after = stop;
        new.fail_after = stop;
        let original = OriginalAffineView::validate(weights.view(), Some(bias.view()), &mut old);
        let compact = OriginalAffineArrays::validate(&weights, Some(&bias), &mut new);
        assert_eq!(original.is_ok(), compact.is_ok());
        assert_eq!(compact.is_ok(), stop.is_none());
        assert_eq!(old.spent, new.spent);
        assert_eq!(old.work_left, new.work_left);
        assert_eq!(old.since_poll, new.since_poll);
        assert_eq!(old_budget.receipt().unwrap(), new_budget.receipt().unwrap());
        drop(old);
        drop(new);
        assert_eq!(old_budget.receipt().unwrap(), new_budget.receipt().unwrap());
        assert_eq!(new_budget.receipt().unwrap().live_scratch_bytes, 0);
        assert_eq!(new_budget.receipt().unwrap().live_persistent_bytes, 0);
    }
}

#[test]
fn original_affine_owned_arrays_refuse_shape_poison_and_real_cancellation() {
    let request = OriginalAffineView::validation_request(1, 1, true).unwrap();
    let clean = array![[0.0]];
    let short = Array1::<f32>::zeros(0);
    let long = array![0.0, 0.0];
    for bias in [&short, &long] {
        let (_, mut phase) = admitted(request);
        assert!(OriginalAffineArrays::validate(&clean, Some(bias), &mut phase).is_err());
        assert_eq!(phase.spent, 0);
    }
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let weight = array![[poison]];
        let bias = array![poison];
        let (_, mut phase) = admitted(request);
        assert!(OriginalAffineArrays::validate(&weight, None, &mut phase).is_err());
        let (_, mut phase) = admitted(request);
        assert!(OriginalAffineArrays::validate(&clean, Some(&bias), &mut phase).is_err());
    }
    let empty = Array2::<f32>::zeros((0, 1));
    let (_, mut phase) = admitted(request);
    assert!(OriginalAffineArrays::validate(&empty, None, &mut phase).is_err());

    use crate::certify::residency::{CancellationToken, MemoryAuthority};
    use crate::certify::work::{WorkLimits, WorkReserve};
    let cancellation = CancellationToken::default();
    let memory = MemoryAuthority::new(request.scratch_bytes + request.persistent_bytes);
    let budget = WorkBudget::new_in(
        &memory,
        &cancellation,
        WorkLimits {
            deadline: None,
            scratch_bytes: request.scratch_bytes,
            persistent_bytes: request.persistent_bytes,
            max_work: request.work,
            max_proposals: request.proposals,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    let mut phase =
        OriginalAffinePhase::admit(&budget, LeaseClass::Optional, request, None).unwrap();
    cancellation.cancel();
    assert!(OriginalAffineArrays::validate(&clean, None, &mut phase).is_err());
    assert_eq!(phase.spent, 0);
    drop(phase);
    assert_eq!(budget.receipt().unwrap().work_reserved, request.work);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
}
