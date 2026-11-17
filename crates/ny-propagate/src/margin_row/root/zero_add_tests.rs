// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use num_rational::BigRational as Q;
use num_traits::{Signed, Zero};

type Output = (Array2<f32>, Array2<f32>, Vec<f64>);

fn q(v: f64) -> Q {
    Q::from_float(v).expect("finite exact dyadic")
}

fn q32(v: f32) -> Q {
    q(ny_core::f32_to_f64_exact(v))
}

fn run(ma: &[f32], da: &[f32], mb: &[f32], db: &[f32], h: &[f64]) -> Output {
    let row = |values: &[f32]| Array2::from_shape_vec((1, values.len()), values.to_vec()).unwrap();
    add_op_signsafe(row(ma), &row(mb), row(da), &row(db), &[], 1, h)
}

// Supremum of each outward-wall violation over the entire signed input box.
// The oracle uses exact stored operand coefficients and the returned storage
// allowance. It does not treat D as unsigned or replace a box by sampled points.
fn excess(ma: &[f32], da: &[f32], mb: &[f32], db: &[f32], h: &[f64], out: &Output) -> (Q, Q) {
    let n = ma.len() - 1;
    let mut lower = -q(out.2[0]);
    let mut upper = lower.clone();
    for i in 0..=n {
        let center = q32(ma[i]) + q32(mb[i]);
        let radius = q32(da[i]) + q32(db[i]);
        let exact_lower = &center - &radius;
        let exact_upper = center + radius;
        let output_lower = q32(out.0[[0, i]]) - q32(out.1[[0, i]]);
        let output_upper = q32(out.0[[0, i]]) + q32(out.1[[0, i]]);
        if i == n {
            lower += output_lower - exact_lower;
            upper += exact_upper - output_upper;
        } else {
            lower += (output_lower - exact_lower).abs() * q(h[i]);
            upper += (exact_upper - output_upper).abs() * q(h[i]);
        }
    }
    (lower, upper)
}

#[test]
fn zero_add_coefficients_preserve_allocations_and_full_frame_bias_debt() {
    let n = 127;
    let h = vec![1.0; n + 1];
    let mb = Array2::<f32>::zeros((1, n + 1));
    let db = mb.clone();
    let mut expected_bias = None;
    for mask in [vec![], vec![u64::MAX], vec![1 << 63], vec![0]] {
        let ma = mb.clone();
        let da = db.clone();
        let (mp, dp) = (ma.as_ptr(), da.as_ptr());
        let out = add_op_signsafe(ma, &mb, da, &db, &mask, support_blk(n + 1), &h);
        assert_eq!(out.0.as_ptr(), mp);
        assert_eq!(out.1.as_ptr(), dp);
        for i in 0..n {
            assert_eq!(out.0[[0, i]].to_bits(), 0);
            assert_eq!(out.1[[0, i]].to_bits(), 0);
        }
        assert_eq!(out.0[[0, n]].to_bits(), 0);
        assert!(
            out.1[[0, n]] > 0.0,
            "the bias still pays the full-frame charge"
        );
        assert!(out.2[0] > 0.0, "bias storage allowance must remain");
        let bias = (out.1[[0, n]].to_bits(), out.2[0].to_bits());
        if let Some(expected) = expected_bias {
            assert_eq!(
                bias, expected,
                "support masks do not change the original frame"
            );
        }
        expected_bias = Some(bias);
        let zeros = vec![0.0; n + 1];
        let (lower, upper) = excess(&zeros, &zeros, &zeros, &zeros, &h, &out);
        assert!(lower <= Q::zero() && upper <= Q::zero());
    }
    // A huge magnitude at an excluded coefficient makes the full-frame floor
    // dominate f32 storage dust. Computing it from only visited or packed
    // coordinates would fail this exact lower bound, even for an all-zero row.
    let mut wide_frame = h;
    wide_frame[0] = 2.0f64.powi(900);
    let floor = q(4.0 * f64::MIN_POSITIVE) * q(wide_frame[0]);
    assert_eq!(floor, q(2.0f64.powi(-120)));
    for mask in [vec![0], vec![1u64 << 63]] {
        let out = add_op_signsafe(
            mb.clone(),
            &mb,
            db.clone(),
            &db,
            &mask,
            support_blk(n + 1),
            &wide_frame,
        );
        assert!(
            q32(out.1[[0, n]]) >= floor,
            "excluded coordinates still enter the full-frame floor"
        );
        assert_eq!(out.0[[0, 0]].to_bits(), 0);
        assert_eq!(out.1[[0, 0]].to_bits(), 0);
    }
}

#[test]
fn zero_add_coefficients_do_not_erase_signed_uncertainty_or_exceptional_values() {
    let tiny = f32::from_bits(1);
    for position in 0..4 {
        for value in [-0.0, 0.25, -0.25, tiny, -tiny, f32::INFINITY, f32::NAN] {
            let mut operands = [[0.0, 0.0]; 4];
            operands[position][0] = value;
            let out = run(
                &operands[0],
                &operands[1],
                &operands[2],
                &operands[3],
                &[1.0; 2],
            );
            assert!(
                out.0[[0, 0]].to_bits() != 0 || out.1[[0, 0]].to_bits() != 0,
                "nonqualifying operand erased: position={position}, value={value:?}"
            );
        }
    }
    for magnitude in [f64::INFINITY, f64::NAN, -1.0] {
        let out = run(
            &[0.0; 2],
            &[0.0; 2],
            &[0.0; 2],
            &[0.0; 2],
            &[magnitude, 1.0],
        );
        assert_ne!(
            out.1[[0, 0]].to_bits(),
            0,
            "invalid magnitude must use the ordinary path"
        );
    }
    let out = run(
        &[0.0, f32::INFINITY],
        &[0.0; 2],
        &[0.0; 2],
        &[0.0; 2],
        &[1.0; 2],
    );
    assert!(
        !out.0[[0, 1]].is_finite(),
        "bias failure must remain visible"
    );
}

#[test]
fn zero_add_coefficients_enclose_exact_signed_walls_and_reject_erasure_mutants() {
    let tiny = f32::from_bits(1);
    let h = [37.5, 2.0f64.powi(200), 2.0, 1.0];
    let fixtures = [
        (
            [0.0, tiny, -0.7, 0.125],
            [0.0, 0.0, -0.25, -0.0625],
            [0.0, -tiny, 0.3, 0.25],
            [0.0, tiny, 0.125, 0.03125],
        ),
        (
            [0.0, -tiny, 0.7, -0.25],
            [0.0, tiny, 0.25, 0.0625],
            [0.0, tiny, -0.7, 0.25],
            [0.0, 0.0, -0.125, -0.03125],
        ),
    ];
    for (ma, da, mb, db) in fixtures {
        let out = run(&ma, &da, &mb, &db, &h);
        assert_eq!(out.0[[0, 0]].to_bits(), 0);
        assert_eq!(out.1[[0, 0]].to_bits(), 0);
        let (lower, upper) = excess(&ma, &da, &mb, &db, &h, &out);
        assert!(
            lower <= Q::zero() && upper <= Q::zero(),
            "full signed-box wall violation"
        );
    }
    // A tempting zero-center-only or floating-zero predicate could erase
    // signed D or a source subnormal. Each mutant is checked against the
    // ORIGINAL exact input, while retaining the mutant's own output allowance.
    for (position, value, magnitude) in [
        (1, -0.25, 1.0),
        (0, tiny, 2.0f64.powi(200)),
        (3, tiny, 2.0f64.powi(200)),
    ] {
        let mut operands = [[0.0, 0.0]; 4];
        operands[position][0] = value;
        let frame = [magnitude, 1.0];
        let original = run(
            &operands[0],
            &operands[1],
            &operands[2],
            &operands[3],
            &frame,
        );
        let (lower, upper) = excess(
            &operands[0],
            &operands[1],
            &operands[2],
            &operands[3],
            &frame,
            &original,
        );
        assert!(lower <= Q::zero() && upper <= Q::zero());
        let mutant = run(&[0.0; 2], &[0.0; 2], &[0.0; 2], &[0.0; 2], &frame);
        let (lower, upper) = excess(
            &operands[0],
            &operands[1],
            &operands[2],
            &operands[3],
            &frame,
            &mutant,
        );
        assert!(
            lower > Q::zero() || upper > Q::zero(),
            "erasure mutant escaped exact oracle"
        );
    }
}
