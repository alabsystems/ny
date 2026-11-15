// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use num_rational::BigRational;

fn fixture(co: usize, p: usize) -> (ConvOp, Vec<f64>) {
    let k = 8;
    let mag: Vec<f64> = (0..24).map(|i| (i + 1) as f64 / 32.0).collect();
    let cv = ConvOp {
        input: 0,
        kernel: (co, 2, 2, 2),
        ishape: (2, 3, 4),
        oshape: (co, 1, p),
        stride: (2, 1),
        pads: (1, 0, 2, 0),
        transposed: false,
        wmat: (0..co * k).map(|i| (i as f64 - 11.0) / 16.0).collect(),
        wt: Vec::new(),
        bias: vec![0.0; co],
        bias_err: vec![0.0; co],
        weight_rel_err: 0.0,
        // Deliberately irregular compiled support: the proof must follow the
        // gather, rather than infer support from stride / padding metadata.
        gather: (0..k * p)
            .map(|i| {
                if i % 7 == 0 {
                    usize::MAX
                } else {
                    (i * 5 + 3) % 24
                }
            })
            .collect(),
        back_taps: Vec::new(),
        k_fwd: k,
        k_bwd: 0,
    };
    (cv, mag)
}

fn q(x: f64) -> BigRational {
    BigRational::from_float(x).unwrap()
}

fn exact_products(cv: &ConvOp, mag: &[f64], o: usize, p: usize) -> BigRational {
    let width = cv.oshape.1 * cv.oshape.2;
    let mut value = q(0.0);
    for t in 0..cv.k_fwd {
        let i = cv.gather[t * width + p];
        if i != usize::MAX {
            value += q(cv.wmat[o * cv.k_fwd + t].abs()) * q(mag[i]);
        }
    }
    value
}

#[test]
fn tap_majorant_contains_exact_products_for_compiled_support_and_transpose() {
    for transpose in [false, true] {
        let (mut cv, mag) = fixture(5, 13);
        cv.transposed = transpose;
        let mut work = MagnitudeWork::default();
        let bound = conv_magnitude(&cv, &mag, None, &mut work).unwrap();
        for o in 0..5 {
            for p in 0..13 {
                assert!(q(bound[o * 13 + p]) >= exact_products(&cv, &mag, o, p));
            }
        }
        assert_eq!(work.envelope_convs, 1);
        assert_eq!(work.products, 5 * 8);
        assert_eq!(work.gathered, 13 * 8);
        assert_eq!(work.direct_products, 5 * 13 * 8);
        assert!(work.products + work.gathered < work.direct_products);
    }
}

#[test]
fn tap_majorant_charges_cancellation_and_parameter_ball_separately() {
    let (mut cv, mut mag) = fixture(4, 8);
    mag.fill(2.0f64.powi(400));
    cv.gather.fill(0);
    for (i, w) in cv.wmat.iter_mut().enumerate() {
        *w = if i % 2 == 0 {
            2.0f64.powi(400)
        } else {
            -2.0f64.powi(400)
        };
    }
    cv.weight_rel_err = 1e-12;
    cv.bias_err.fill(1e-9);
    let bound = conv_magnitude(&cv, &mag, None, &mut MagnitudeWork::default()).unwrap();
    let signed: BigRational = cv.wmat[..8].iter().map(|w| q(*w) * q(mag[0])).sum();
    assert_eq!(signed, q(0.0));
    assert!(bound[0] > 1e240);
    assert!(q(bound[0]) >= exact_products(&cv, &mag, 0, 0));
    // A larger fold ball does not change |W'| mag. The frame applies relative
    // parameter error to this enclosure when forming output magnitudes, and
    // the reverse builder also charges that error independently.
    cv.weight_rel_err = 0.25;
    cv.bias_err.fill(1.0);
    assert_eq!(
        bound,
        conv_magnitude(&cv, &mag, None, &mut MagnitudeWork::default()).unwrap()
    );
}

#[test]
fn tap_majorant_covers_underflow_padding_and_signed_zero() {
    let (mut cv, mut mag) = fixture(4, 9);
    mag.fill(2.0f64.powi(-500));
    cv.wmat.fill(2.0f64.powi(-600));
    let bound = conv_magnitude(&cv, &mag, None, &mut MagnitudeWork::default()).unwrap();
    assert!(exact_products(&cv, &mag, 0, 1) > q(0.0));
    assert!(q(bound[1]) >= exact_products(&cv, &mag, 0, 1));
    cv.gather.fill(usize::MAX);
    mag.fill(-0.0);
    let padding = conv_magnitude(&cv, &mag, None, &mut MagnitudeWork::default()).unwrap();
    assert!(padding.iter().all(|v| v.is_finite() && *v >= 0.0));
}

#[test]
fn tap_majorant_reference_branch_preserves_spatial_rows() {
    for (co, p) in [(1, 7), (5, 1), (0, 0)] {
        let (cv, mag) = fixture(co, p);
        let mut work = MagnitudeWork::default();
        let bound = conv_magnitude(&cv, &mag, None, &mut work).unwrap();
        assert_eq!(work.envelope_convs, 0);
        for o in 0..co {
            for sp in 0..p {
                assert!(q(bound[o * p + sp]) >= exact_products(&cv, &mag, o, sp));
            }
        }
    }
}

#[test]
fn tap_majorant_refuses_invalid_live_content_and_overflow() {
    let (cv, mag) = fixture(4, 8);
    for mutant in 0..7 {
        let mut bad = cv.clone();
        let mut values = mag.clone();
        match mutant {
            0 => bad.gather[1] = 24,
            1 => bad.gather.pop().map(|_| ()).unwrap(),
            2 => bad.wmat[0] = f64::NAN,
            3 => values[0] = f64::INFINITY,
            4 => values[0] = -1.0,
            5 => bad.k_fwd = 7,
            _ => {
                bad.wmat.fill(f64::MAX);
                values.fill(2.0);
            }
        }
        let mut work = MagnitudeWork::default();
        assert!(
            conv_magnitude(&bad, &values, None, &mut work).is_err(),
            "mutant {mutant}"
        );
        assert_eq!(work.products, 0);
        assert_eq!(work.envelope_convs, 0);
    }
}

#[test]
fn tap_majorant_refuses_deadline_without_publishing_work() {
    let (cv, mag) = fixture(4, 8);
    let mut work = MagnitudeWork::default();
    let expired = Instant::now()
        .checked_sub(std::time::Duration::from_millis(1))
        .unwrap();
    assert!(matches!(
        conv_magnitude(&cv, &mag, Some(expired), &mut work),
        Err(ZonoRefusal::Deadline)
    ));
    assert_eq!(work.products, 0);
    assert!(conv_magnitude(&cv, &mag, None, &mut work).is_ok());
    assert!(work.products > 0);
}
