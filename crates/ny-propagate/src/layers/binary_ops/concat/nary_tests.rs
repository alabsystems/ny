// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Original-coordinate controls for the scalar pullback used by intermediate
//! collectors. The independent label oracle uses actual tensor concatenation.
use super::*;
use ndarray::{Array1, Array2, ArrayD, Axis, IxDyn};
use num_rational::BigRational;
use num_traits::{Signed, Zero};

fn carrier(columns: usize) -> LinearBounds {
    let mut row = LinearBounds::new(
        Array2::from_shape_fn((3, columns), |(r, c)| {
            if r == 2 {
                -0.0
            } else {
                (r * columns + c) as f32 * 0.125 - 0.5
            }
        }),
        Array1::from_vec(vec![-0.125, 0.25, 0.0]),
        Array2::from_shape_fn((3, columns), |(r, c)| {
            if r == 2 {
                0.0
            } else {
                (r * columns + c) as f32 * -0.0625 + 0.375
            }
        }),
        Array1::from_vec(vec![0.25, 0.5, 0.0]),
    )
    .unwrap();
    row.lower_a_err = Some(Array2::from_shape_fn((3, columns), |(r, c)| {
        ((r * columns + c) % 5) as f32 * 0.015625
    }));
    row.upper_a_err = Some(Array2::from_shape_fn((3, columns), |(r, c)| {
        ((r * columns + c) % 7) as f32 * 0.0078125
    }));
    row
}

#[test]
fn scalar_concat_original_axis_mapping_preserves_all_four_plane_bits() {
    for (axis, shapes) in [
        (0, vec![vec![1, 2], vec![3, 2]]),
        (1, vec![vec![2, 1], vec![2, 2]]),
        (1, vec![vec![2, 1, 2], vec![2, 3, 2], vec![2, 2, 2]]),
        (-1, vec![vec![2, 2, 1], vec![2, 2, 3]]),
    ] {
        let columns = shapes.iter().map(|s| s.iter().product::<usize>()).sum();
        let incoming = carrier(columns);
        let parts = ConcatLayer::new(axis)
            .propagate_linear_nary(&incoming, &shapes)
            .unwrap();
        let mut next_label = 0usize;
        let labels = shapes
            .iter()
            .map(|shape| {
                let n = shape.iter().product::<usize>();
                let tensor =
                    ArrayD::from_shape_vec(IxDyn(shape), (next_label..next_label + n).collect())
                        .unwrap();
                next_label += n;
                tensor
            })
            .collect::<Vec<_>>();
        let resolved_axis = if axis < 0 {
            shapes[0].len() - 1
        } else {
            axis as usize
        };
        let output = ndarray::concatenate(
            Axis(resolved_axis),
            &labels
                .iter()
                .map(|labels| labels.view())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let flattened = output.iter().copied().collect::<Vec<_>>();
        let mut label_offset = 0;
        for part in parts {
            for r in 0..3 {
                for c in 0..part.num_inputs() {
                    let original = flattened
                        .iter()
                        .position(|&label| label == label_offset + c)
                        .unwrap();
                    for (got, expected) in [
                        (part.lower_a[[r, c]], incoming.lower_a[[r, original]]),
                        (part.upper_a[[r, c]], incoming.upper_a[[r, original]]),
                        (
                            part.lower_a_err.as_ref().unwrap()[[r, c]],
                            incoming.lower_a_err.as_ref().unwrap()[[r, original]],
                        ),
                        (
                            part.upper_a_err.as_ref().unwrap()[[r, c]],
                            incoming.upper_a_err.as_ref().unwrap()[[r, original]],
                        ),
                    ] {
                        assert_eq!(
                            got.to_bits(),
                            expected.to_bits(),
                            "axis={axis} row={r} col={c}"
                        );
                    }
                }
            }
            label_offset += part.num_inputs();
        }
    }
}

fn rat(value: f32) -> BigRational {
    BigRational::from_float(ny_core::f32_to_f64_exact(value)).unwrap()
}
fn evaluate(row: &LinearBounds, r: usize, x: &[BigRational], upper: bool) -> BigRational {
    let (a, b, e) = if upper {
        (&row.upper_a, &row.upper_b, row.upper_a_err.as_ref())
    } else {
        (&row.lower_a, &row.lower_b, row.lower_a_err.as_ref())
    };
    x.iter().enumerate().fold(rat(b[r]), |sum, (c, x)| {
        let error = e.map_or_else(BigRational::zero, |e| rat(e[[r, c]])) * x.abs();
        sum + rat(a[[r, c]]) * x + if upper { error } else { -error }
    })
}

#[test]
fn scalar_concat_nonlast_signed_error_rows_contain_all_original_rational_corners() {
    let incoming = carrier(6);
    let parts = ConcatLayer::new(1)
        .propagate_linear_nary(&incoming, &[vec![2, 1], vec![2, 2]])
        .unwrap();
    for mask in 0_usize..64 {
        let x = (0..6)
            .map(|i| {
                rat(if mask & (1_usize << i) == 0_usize {
                    -0.5
                } else {
                    0.75
                })
            })
            .collect::<Vec<_>>();
        // Original [2,1] and [2,2] slots concatenate row by row, not by
        // placing the whole first flattened tensor before the whole second.
        let y = vec![
            x[0].clone(),
            x[2].clone(),
            x[3].clone(),
            x[1].clone(),
            x[4].clone(),
            x[5].clone(),
        ];
        for r in 0..3 {
            for upper in [false, true] {
                let original = evaluate(&incoming, r, &y, upper);
                let actual =
                    evaluate(&parts[0], r, &x[..2], upper) + evaluate(&parts[1], r, &x[2..], upper);
                if upper {
                    assert!(actual >= original);
                } else {
                    assert!(actual <= original);
                }
            }
        }
    }
}

#[test]
fn scalar_concat_actual_singleton_restoration_matches_original_feature_axis() {
    let incoming = carrier(3);
    let parts = ConcatLayer::new(0)
        .with_restored_batch_axis_shift(true)
        .propagate_linear_nary(&incoming, &[vec![1, 1], vec![2]])
        .unwrap();
    assert_eq!(parts[0].num_inputs(), 1);
    assert_eq!(parts[1].num_inputs(), 2);
    for r in 0..3 {
        assert_eq!(
            parts[0].lower_a[[r, 0]].to_bits(),
            incoming.lower_a[[r, 0]].to_bits()
        );
        for c in 0..2 {
            assert_eq!(
                parts[1].upper_a[[r, c]].to_bits(),
                incoming.upper_a[[r, c + 1]].to_bits()
            );
            assert_eq!(
                parts[1].lower_a_err.as_ref().unwrap()[[r, c]].to_bits(),
                incoming.lower_a_err.as_ref().unwrap()[[r, c + 1]].to_bits()
            );
        }
    }
}

#[test]
fn scalar_concat_ambiguous_broadcast_incompatible_geometry_and_malformed_error_refuse() {
    assert!(ConcatLayer::new(1)
        .propagate_linear_nary(&carrier(5), &[vec![2, 1], vec![3, 1]])
        .is_err());
    assert!(ConcatLayer::new(1)
        .propagate_linear_nary(&carrier(6), &[vec![2, 2], vec![2]])
        .is_err());
    assert!(ConcatLayer::new(2)
        .propagate_linear_nary(&carrier(4), &[vec![2], vec![2]])
        .is_err());
    let mut malformed = carrier(6);
    malformed.lower_a_err = Some(Array2::zeros((3, 5)));
    assert!(ConcatLayer::new(1)
        .propagate_linear_nary(&malformed, &[vec![2, 1], vec![2, 2]])
        .is_err());
}

#[test]
fn scalar_concat_poisoned_coefficients_cannot_lose_conservative_bias_sentinels() {
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for upper in [false, true] {
            let mut source = carrier(6);
            if upper {
                source.upper_a[[1, 3]] = poison;
            } else {
                source.lower_a[[1, 3]] = poison;
            }
            assert!(ConcatLayer::new(1)
                .propagate_linear_nary(&source, &[vec![2, 1], vec![2, 2]])
                .is_err());
        }
    }
}
