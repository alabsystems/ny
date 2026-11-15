// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{s, Array2};
use num_rational::BigRational;

#[test]
fn sequence_bias_error_sum_encloses_every_position_against_exact_rationals() {
    for positions in [0, 1, 2, 17] {
        let bias = Array1::from_vec(vec![1.0, -1048576.0, f32::from_bits(1), 0.0, 0.375]);
        let errors = Array2::from_shape_fn((3, positions * bias.len()), |(i, j)| {
            [0.0, 0.03125, 0.0625, f32::from_bits(1)][(i + j) % 4]
        });
        for row in 0..3 {
            let got =
                sum_coeff_err_bias_up(errors.view(), row, bias.view(), positions, &mut |_| Ok(()))
                    .unwrap();
            let mut exact = BigRational::from_integer(0.into());
            for pos in 0..positions {
                for j in 0..bias.len() {
                    exact += BigRational::from_float(f32_to_f64_exact(
                        errors[[row, pos * bias.len() + j]],
                    ))
                    .unwrap()
                        * BigRational::from_float(f32_to_f64_exact(bias[j]).abs()).unwrap();
                }
            }
            assert!(BigRational::from_float(got).unwrap() >= exact);
        }
    }
}

#[test]
fn sequence_bias_error_later_poison_is_seen_and_zero_bias_still_dominates() {
    for poison in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -1.0,
        f32::from_bits(0x80000001),
    ] {
        for zero in [false, true] {
            let bias = Array1::from_vec(vec![if zero { 0.0 } else { -2.0 }, 0.0]);
            let mut error = Array2::zeros((1, 6));
            error[[0, 4]] = poison;
            let value =
                sum_coeff_err_bias_up(error.view(), 0, bias.view(), 3, &mut |_| Ok(())).unwrap();
            assert_eq!(value, if zero { 0.0 } else { f64::INFINITY });
        }
    }
}

#[test]
fn sequence_bias_error_strided_views_keep_logical_position_order_and_original_bits() {
    let errors = Array2::from_shape_fn((8, 2 * 3 * 257 + 2), |(i, j)| ((i + j) % 7) as f32 * 0.125);
    let bias = Array1::from_shape_fn(257 * 2 + 2, |j| j as f32 * 0.0625 - 3.0);
    let ev = errors.slice(s![1..7;2,1..1+2*3*257;2]);
    let bv = bias.slice(s![1..1+2*257;2]);
    for row in 0..3 {
        let mut total = 0;
        let mut batches = Vec::new();
        let got = sum_coeff_err_bias_up(ev, row, bv, 3, &mut |units| {
            assert!(units <= 4096);
            total += units;
            batches.push(units);
            Ok(())
        })
        .unwrap();
        let expected = sum_coeff_err_bias_up(
            ev.to_owned().view(),
            row,
            bv.to_owned().view(),
            3,
            &mut |_| Ok(()),
        )
        .unwrap();
        assert_eq!(got.to_bits(), expected.to_bits());
        assert_eq!(total, 64 + 32 * 3 * 257);
        assert_eq!(batches.first(), Some(&0));
        assert_eq!(batches.last(), Some(&0));
        assert_eq!(batches.iter().filter(|&&n| n == 4096).count(), 6);
    }
}

#[test]
fn sequence_bias_error_shape_and_overflow_errors_precede_term_work() {
    let error = Array2::zeros((1, 3));
    let bias = Array1::ones(2);
    for (row, positions) in [(0, 2), (1, 1), (0, usize::MAX)] {
        let mut calls = Vec::new();
        assert!(
            sum_coeff_err_bias_up(error.view(), row, bias.view(), positions, &mut |n| {
                calls.push(n);
                Ok(())
            })
            .is_err()
        );
        assert_eq!(calls, vec![0, 64]);
    }
}

#[test]
fn sequence_bias_error_every_prework_and_final_refusal_propagates_unchanged() {
    let error = Array2::ones((1, 3 * 257));
    let bias = Array1::ones(257);
    let total = 3 + 3 * 3; // entry, metadata, three batches per position, final
    for stop in 1..=total {
        let mut calls = 0;
        let result = sum_coeff_err_bias_up(error.view(), 0, bias.view(), 3, &mut |_| {
            calls += 1;
            if calls == stop {
                Err(NyError::DeadlineExceeded("original observer".into()))
            } else {
                Ok(())
            }
        });
        assert!(matches!(result,Err(NyError::DeadlineExceeded(s)) if s=="original observer"));
        assert_eq!(calls, stop);
    }
}

#[test]
fn sequence_bias_error_empty_bias_has_entry_and_final_checks_without_empty_position_spin() {
    let error = Array2::zeros((1, 0));
    let bias = Array1::zeros(0);
    let mut calls = Vec::new();
    let value = sum_coeff_err_bias_up(error.view(), 0, bias.view(), usize::MAX, &mut |n| {
        calls.push(n);
        Ok(())
    })
    .unwrap();
    assert_eq!(value, 0.0);
    assert_eq!(calls, vec![0, 64, 0]);
}
