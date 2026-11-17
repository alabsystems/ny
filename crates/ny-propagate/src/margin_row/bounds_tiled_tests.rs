// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::super::{compose_viay, compose_viay_impl, head_gates, row_dots, MarginBatch, YBox};
use super::*;
use crate::margin_row::engine::BackwardEngine;
use crate::margin_row::net::TwinNet;
use crate::margin_row::root::RootGates;
use crate::margin_row::rounding::RoundMode;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use num_rational::BigRational;
use num_traits::Zero;

fn scalar(a: &[f64], b: &[f64], ca: &[f64], cb: &[f64]) -> Vec<f64> {
    a.chunks(ca.len())
        .zip(b.chunks(cb.len()))
        .map(|(a, b)| {
            let mut acc = 0.0;
            for j in 0..ca.len() {
                acc += ca[j] * a[j] + cb[j] * b[j];
            }
            acc
        })
        .collect()
}

#[test]
fn head_tiled_attained_minimum_matches_every_original_dust_predicate() {
    ny_core::require_f64_interval_proof_environment().unwrap();
    let values = [
        0.0,
        -0.0,
        f64::from_bits(1),
        -f64::from_bits(3),
        f64::from_bits(f64::MIN_POSITIVE.to_bits() - 1),
        f64::MIN_POSITIVE,
        f64::from_bits(f64::MIN_POSITIVE.to_bits() + 1),
        -1.0,
        0.5,
        2.0,
        2.0f64.powi(-500),
        2.0f64.powi(500),
        f64::MAX,
    ];
    for n in [1, 7, 8, 9, 17, 65] {
        let ny = values.len() + 1;
        let a: Vec<_> = (0..n * ny)
            .map(|i| {
                if i % ny == ny - 1 {
                    0.0
                } else {
                    values[(i * 7 + i / ny) % values.len()]
                }
            })
            .collect();
        let b: Vec<_> = a.iter().rev().copied().collect();
        let packed = TiledHeadRows::prepare(&a, &b, n, ny, MIN_WORK).unwrap();
        for shift in 0..values.len() {
            let ca: Vec<_> = (0..ny)
                .map(|j| values[(j + shift) % values.len()])
                .collect();
            let cb: Vec<_> = (0..ny)
                .map(|j| values[(j * 3 + shift) % values.len()])
                .collect();
            let expected = (0..n).any(|k| {
                (0..ny).any(|j| {
                    head_underflow(ca[j], a[k * ny + j]) || head_underflow(cb[j], b[k * ny + j])
                })
            });
            assert_eq!(
                packed.product_underflow(&ca, &cb),
                Some(expected),
                "n={n} shift={shift}"
            );
        }
    }
    // A small value in an inactive column must not implicate another column.
    let p = TiledHeadRows::prepare(&[f64::from_bits(1), 2.0], &[0.0, 0.0], 1, 2, MIN_WORK).unwrap();
    assert_eq!(p.product_underflow(&[0.0, 0.5], &[0.0, 0.0]), Some(false));

    // Isolate each normal/subnormal threshold so another column cannot mask
    // a false positive or false negative in the attained-minimum predicate.
    let min = f64::MIN_POSITIVE;
    let sub = f64::from_bits(1);
    for (a, c, expected, product_bits) in [
        (min, 1.0, false, min.to_bits()),
        (
            min,
            f64::from_bits(1.0f64.to_bits() - 1),
            false,
            min.to_bits(),
        ),
        (min, 1.0 - 2.0f64.powi(-52), true, min.to_bits() - 1),
        (sub, 2.0f64.powi(1023), false, 2.0f64.powi(-51).to_bits()),
        (sub, 0.5, true, 0),
        (sub, 0.0, false, 0),
    ] {
        assert_eq!(
            (std::hint::black_box(a) * std::hint::black_box(c)).to_bits(),
            product_bits
        );
        for n in [1, 7, 8, 9, 17, 65] {
            for sign in [1.0, -1.0] {
                let p = TiledHeadRows::prepare(&vec![sign * a; n], &vec![0.0; n], n, 1, MIN_WORK)
                    .unwrap();
                assert_eq!(p.product_underflow(&[c], &[0.0]), Some(expected));
                assert_eq!(head_underflow(sign * a, c), expected);
            }
        }
    }
}

#[test]
fn head_tiled_coordinate_tails_keep_order_signed_zero_and_nonfused_products() {
    let u = 2.0f64.powi(-27);
    let ca = [1.0 + u, 1.0, 1.0, -0.0, 0.5, -1.0, 2.0];
    let cb = [1.0, 1.0, -1.0, 0.0, 0.5, 1.0, -2.0];
    assert_ne!(
        (1.0 + u) * (1.0 - u) - 1.0,
        (1.0 + u).mul_add(1.0 - u, -1.0)
    );
    for n in [1, 7, 8, 9, 17, 65] {
        let mut a = Vec::new();
        let mut b = Vec::new();
        for k in 0..n {
            match k % 3 {
                0 => {
                    a.extend([1.0 - u, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0]);
                    b.extend([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
                }
                1 => {
                    a.extend([0.0, 1e16, -1e16, 0.0, 2.0, 0.0, 0.0]);
                    b.extend([0.0; 7]);
                }
                _ => {
                    a.extend([
                        0.0,
                        -0.0,
                        0.0,
                        -0.0,
                        f64::from_bits(3),
                        f64::MIN_POSITIVE,
                        -0.125,
                    ]);
                    b.extend([
                        0.0,
                        0.0,
                        -0.0,
                        0.0,
                        f64::from_bits(1),
                        f64::MIN_POSITIVE,
                        0.25,
                    ]);
                }
            }
        }
        let p = TiledHeadRows::prepare(&a, &b, n, 7, MIN_WORK).unwrap();
        let mut actual = vec![f64::NAN; n];
        p.compose(&ca, &cb, &mut actual);
        let expected = scalar(&a, &b, &ca, &cb);
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn head_tiled_independent_dyadic_rational_coefficients() {
    let (n, ny) = (17, 11);
    let a: Vec<_> = (0..n * ny)
        .map(|i| ((i * 3 % 13) as f64 - 6.0) / 8.0)
        .collect();
    let b: Vec<_> = (0..n * ny)
        .map(|i| ((i * 7 % 17) as f64 - 8.0) / 16.0)
        .collect();
    let ca: Vec<_> = (0..ny).map(|j| ((j % 5) as f64 - 2.0) / 4.0).collect();
    let cb: Vec<_> = (0..ny).map(|j| ((j % 7) as f64 - 3.0) / 8.0).collect();
    let p = TiledHeadRows::prepare(&a, &b, n, ny, MIN_WORK).unwrap();
    let mut actual = vec![0.0; n];
    p.compose(&ca, &cb, &mut actual);
    let q = |x| BigRational::from_float(x).unwrap();
    for k in 0..n {
        let expected = (0..ny).fold(BigRational::zero(), |sum, j| {
            sum + q(ca[j]) * q(a[k * ny + j]) + q(cb[j]) * q(b[k * ny + j])
        });
        assert_eq!(q(actual[k]), expected);
    }
}

#[test]
fn head_tiled_refusals_preserve_scalar_route() {
    // Reachable one-input, 512-head, 513-class geometry meets MIN_WORK but
    // would spend eight times the scalar arithmetic on padding lanes.
    for n in [1, 7] {
        let rows = vec![1.0; n * 512];
        assert!(TiledHeadRows::prepare(&rows, &rows, n, 512, 512).is_some());
        assert!(TiledHeadRows::for_composition(&rows, &rows, n, 512, 512).is_none());
    }
    let rows = vec![1.0; 8 * 64];
    assert!(TiledHeadRows::for_composition(&rows, &rows, 8, 64, 512).is_some());
    assert!(TiledHeadRows::prepare(&[], &[], usize::MAX, 2, MIN_WORK).is_none());
    assert!(TiledHeadRows::prepare(&[], &[], 1, usize::MAX, MIN_WORK).is_none());
    assert!(TiledHeadRows::prepare(&[1.0], &[1.0], 1, 1, 1).is_none());
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(TiledHeadRows::prepare(&[bad], &[1.0], 1, 1, MIN_WORK).is_none());
        assert!(TiledHeadRows::prepare(&[1.0], &[bad], 1, 1, MIN_WORK).is_none());
        let p = TiledHeadRows::prepare(&[1.0], &[1.0], 1, 1, MIN_WORK).unwrap();
        assert_eq!(p.product_underflow(&[bad], &[1.0]), None);
        assert_eq!(p.product_underflow(&[1.0], &[bad]), None);
    }
}

fn network() -> TwinNet {
    let (n, ny, no) = (257, 17, 66);
    TwinNet::compile(&TwinSpec {
        n_in: n,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0],
                bias: vec![0.0],
                bias_err: vec![0.0],
                weight_rel_err: 0.0,
                kernel: (1, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, n),
                oshape: (1, 1, n),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: (0..n * ny)
                    .map(|i| ((i * 7 % 17) as f64 - 8.0) / 256.0)
                    .collect(),
                bias: vec![0.125; ny],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (ny, n),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: (0..ny * no)
                    .map(|i| ((i * 11 % 31) as f64 - 15.0) / 16.0)
                    .collect(),
                bias: vec![0.25; no],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (no, ny),
            },
        ],
    })
    .unwrap()
}

#[test]
fn head_tiled_complete_margin_bits_match_scalar_in_both_modes() {
    let net = network();
    for mode in [RoundMode::Parity, RoundMode::Outward] {
        let root = RootGates::build(
            &net,
            &vec![-0.25; net.n_in],
            &vec![0.5; net.n_in],
            mode,
            None,
        )
        .unwrap();
        let eng = BackwardEngine::new(&net, &root);
        let (mut al, mut au) = eng.y_rows(None).unwrap();
        for exceptional in 0..3 {
            // Arithmetic-carrier mutation exercises the original directed
            // confirmation; it does not mint a network proof owner.
            if exceptional == 1 {
                al.a[[0, 0]] = f64::from_bits(1);
                au.a[[1, 1]] = -f64::from_bits(3);
            }
            let mut gates = head_gates(&YBox::from_rows(&eng, &al, &au), mode);
            let mut mb = MarginBatch::new(&net, 0, &(1..66).collect::<Vec<_>>()).unwrap();
            if exceptional == 2 {
                gates.alpha[0] = f64::from_bits(1);
                mb.wp[0] = 0.5;
                mb.wn[0] = 0.0;
                assert_eq!(mb.wp[0] * gates.alpha[0], 0.0);
                assert!(head_underflow(mb.wp[0], gates.alpha[0]));
            }
            let ad = row_dots(&root, &al);
            let ud = row_dots(&root, &au);
            assert!(TiledHeadRows::prepare(
                al.a.as_slice().unwrap(),
                au.a.as_slice().unwrap(),
                net.n_in,
                net.n_y,
                mb.nf()
            )
            .is_some());
            let (actual, counts) = super::super::confirm::with_counts(|| {
                compose_viay(&eng, &mb, &gates, &al, &au, &ad, &ud, mode)
            });
            if !mode.outward() {
                assert_eq!(counts, [0; 3]);
            } else if exceptional == 2 {
                // This fixture forces real underflow. A permanently refusing
                // or unused cache cannot pass the full-composition control.
                assert_eq!((counts[0], counts[1]), (1, 1));
                assert!(counts[2] > 0);
            }
            let scalar =
                compose_viay_impl::<false, true>(&eng, &mb, &gates, &al, &au, &ad, &ud, mode);
            assert!(actual.iter().all(|x| x.is_finite()));
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                scalar.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "mode={mode:?} exceptional={exceptional}"
            );
        }
    }
}
