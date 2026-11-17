// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Arithmetic carriers exercise numerical premises, not proof ownership.
use super::*;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use num_rational::BigRational;
use num_traits::{Signed, Zero};

fn root(n: usize) -> RootGates {
    let net = TwinNet::compile(&TwinSpec {
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
                weight: vec![0.0; n],
                bias: vec![0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (1, n),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![0.0; 2],
                bias: vec![0.0; 2],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
        ],
    })
    .unwrap();
    RootGates::build(
        &net,
        &vec![-0.5; n],
        &vec![1.0; n],
        RoundMode::Outward,
        None,
    )
    .unwrap()
}
fn pack(n: usize, ny: usize) -> PassOut {
    PassOut {
        a: Array2::zeros((n, ny)),
        e: Some(Array2::zeros((n, ny))),
        b: vec![0.0; ny],
        eb: vec![0.0; ny],
        coll: None,
        coll_rows: None,
        unst_rows: None,
    }
}
fn batch(ny: usize, nf: usize) -> (MarginBatch, HeadGates) {
    (
        MarginBatch {
            t: 0,
            adv: (1..=nf).collect(),
            wp: vec![0.5; ny * nf],
            wn: vec![-0.25; ny * nf],
            cst: vec![0.125; nf],
            n_y: ny,
        },
        HeadGates {
            alpha: vec![0.5; ny],
            s: vec![0.75; ny],
            c: vec![0.125; ny],
        },
    )
}
fn reference(
    root: &RootGates,
    mb: &MarginBatch,
    gates: &HeadGates,
    al: &PassOut,
    au: &PassOut,
    r: usize,
) -> Result<f64> {
    let check = || Ok(());
    head_confirm_viay_row(root, mb, gates, al, au, r, &mut HeadPoll::new(&check)?)
}
fn same_result(a: Result<f64>, b: Result<f64>) {
    match (a, b) {
        (Ok(a), Ok(b)) => assert_eq!(a.to_bits(), b.to_bits()),
        (Err(_), Err(_)) => (),
        (a, b) => panic!("confirmation differs: {a:?} vs {b:?}"),
    }
}

#[test]
fn head_confirm_cache_keeps_bias_first_directed_error_fold() {
    let root = root(3);
    let (mb, gates) = batch(1, 2);
    let mut al = pack(3, 1);
    let au = pack(3, 1);
    al.eb[0] = 1.0;
    al.e.as_mut().unwrap().fill(2.0f64.powi(-55));
    let mut prepared = Prepared::new(&root, &mb, &gates, &al, &au).unwrap();
    assert_eq!(prepared.eta[0][0].to_bits(), 1.0f64.to_bits() + 3);
    assert_ne!(
        prepared.eta[0][0].to_bits(),
        head_sum_hi(3.0 * 2.0f64.powi(-55), 1.0).to_bits()
    );
    for r in 0..mb.nf() {
        same_result(prepared.row(r), reference(&root, &mb, &gates, &al, &au, r));
    }
}

#[test]
fn head_confirm_cache_preserves_subnormal_endpoints_and_cancellation_bits() {
    let values = [
        0.0,
        -0.0,
        f64::from_bits(1),
        -f64::from_bits(3),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        0.5,
        -0.5,
        1.0,
        -1.0,
        2.0f64.powi(500),
        -2.0f64.powi(500),
    ];
    for n in [1, 3, 7, 8, 9, 17] {
        let root = root(n);
        let (ny, nf) = (11, 13);
        let (mut mb, mut gates) = batch(ny, nf);
        let (mut al, mut au) = (pack(n, ny), pack(n, ny));
        for (i, a) in al.a.iter_mut().enumerate() {
            *a = values[i % values.len()];
        }
        for (i, a) in au.a.iter_mut().enumerate() {
            *a = values[(i * 7 + 1) % values.len()];
        }
        for (i, e) in al.e.as_mut().unwrap().iter_mut().enumerate() {
            *e = values[i % 7].abs();
        }
        for (i, e) in au.e.as_mut().unwrap().iter_mut().enumerate() {
            *e = values[(i * 3) % 7].abs();
        }
        for j in 0..ny {
            gates.alpha[j] = [0.0, -0.0, 0.25, 0.5, 1.0][j % 5];
            gates.s[j] = [0.5, 0.25, 1.0, 0.0][j % 4];
            al.b[j] = values[j % values.len()];
            au.b[j] = -values[j % values.len()];
            al.eb[j] = values[j % 7].abs();
            au.eb[j] = values[(j + 1) % 7].abs();
        }
        for i in 0..ny * nf {
            mb.wp[i] = values[(i * 3) % values.len()].abs();
            mb.wn[i] = -values[(i * 5) % values.len()].abs();
        }
        mb.wp[2] = f64::from_bits(1);
        gates.alpha[2] = 0.25;
        let mut prepared = Prepared::new(&root, &mb, &gates, &al, &au).unwrap();
        for r in 0..nf {
            same_result(prepared.row(r), reference(&root, &mb, &gates, &al, &au, r));
        }
        prepared.row(0).unwrap();
        assert_eq!(prepared.coefficients[2][0], -f64::from_bits(1));
        assert_eq!(prepared.coefficients[2][1], f64::from_bits(1));
    }
}

#[test]
fn head_confirm_cache_independent_rational_full_bound() {
    let root = root(9);
    let (ny, nf) = (7, 5);
    let (mut mb, mut gates) = batch(ny, nf);
    let (mut al, mut au) = (pack(9, ny), pack(9, ny));
    for (i, x) in al.a.iter_mut().enumerate() {
        *x = ((i * 3 % 17) as f64 - 8.0) / 7.0;
    }
    for (i, x) in au.a.iter_mut().enumerate() {
        *x = ((i * 7 % 19) as f64 - 9.0) / 11.0;
    }
    for (i, x) in al.e.as_mut().unwrap().iter_mut().enumerate() {
        *x = (i % 5) as f64 / 103.0;
    }
    au.e.as_mut().unwrap().fill(f64::from_bits(1));
    for j in 0..ny {
        al.b[j] = j as f64 / 13.0;
        au.b[j] = -(j as f64) / 17.0;
        al.eb[j] = 1.0 / 31.0;
        au.eb[j] = f64::from_bits(3);
        gates.alpha[j] = j as f64 / 7.0;
        gates.s[j] = j as f64 / 9.0;
    }
    for i in 0..ny * nf {
        mb.wp[i] = (i % 5) as f64 / 3.0;
        mb.wn[i] = -((i % 7) as f64) / 11.0;
    }
    mb.wp[1] = f64::from_bits(1);
    let q = |x| BigRational::from_float(x).unwrap();
    let mut prepared = Prepared::new(&root, &mb, &gates, &al, &au).unwrap();
    for r in 0..nf {
        let ca: Vec<_> = (0..ny)
            .map(|j| q(mb.wp[r * ny + j]) * q(gates.alpha[j]))
            .collect();
        let cb: Vec<_> = (0..ny)
            .map(|j| q(mb.wn[r * ny + j]) * q(gates.s[j]))
            .collect();
        let mut exact = q(mb.cst[r]);
        for k in 0..9 {
            let c = (0..ny).fold(BigRational::zero(), |s, j| {
                s + &ca[j] * q(al.a[[k, j]]) + &cb[j] * q(au.a[[k, j]])
            });
            exact += (&c * q(root.lo[k])).min(&c * q(root.hi[k]));
        }
        for j in 0..ny {
            let mut el = q(al.eb[j]);
            let mut eu = q(au.eb[j]);
            for k in 0..9 {
                let mag = q(root.lo[k]).abs().max(q(root.hi[k]).abs());
                el += q(al.e.as_ref().unwrap()[[k, j]]) * &mag;
                eu += q(au.e.as_ref().unwrap()[[k, j]]) * mag;
            }
            exact +=
                &ca[j] * q(al.b[j]) + &cb[j] * q(au.b[j]) + q(mb.wn[r * ny + j]) * q(gates.c[j]);
            exact -= ca[j].abs() * el + cb[j].abs() * eu;
        }
        let actual = prepared.row(r).unwrap();
        assert!(q(actual) <= exact);
        same_result(Ok(actual), reference(&root, &mb, &gates, &al, &au, r));
    }
}

#[test]
fn head_confirm_cache_refusal_never_hides_nonfinite_zero_times_source() {
    let mut root = root(3);
    let (mut mb, mut gates) = batch(2, 2);
    mb.wp.fill(0.0);
    mb.wn.fill(0.0);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
        for site in 0..8 {
            let (mut al, mut au) = (pack(3, 2), pack(3, 2));
            root.hi[0] = 2.0;
            gates.alpha[0] = 0.5;
            match site {
                0 => al.a[[0, 0]] = bad,
                1 => au.b[0] = bad,
                2 => al.e.as_mut().unwrap()[[0, 0]] = bad,
                3 => au.eb[0] = bad,
                4 => gates.alpha[0] = bad,
                5 => {
                    al.a[[0, 0]] = bad;
                    mb.wp[0] = 4.0;
                }
                6 => root.hi[0] = bad,
                _ => {
                    al.eb[0] = bad;
                    al.e.as_mut().unwrap().fill(f64::MAX);
                }
            }
            let expected = reference(&root, &mb, &gates, &al, &au, 0);
            let actual = match Prepared::new(&root, &mb, &gates, &al, &au) {
                Ok(mut prepared) => prepared.row(0),
                Err(_) => reference(&root, &mb, &gates, &al, &au, 0),
            };
            same_result(actual, expected);
            mb.wp.fill(0.0);
        }
    }
}

#[test]
fn head_confirm_cache_capacity_shape_and_column_refusals() {
    let root = root(3);
    let (mb, gates) = batch(2, 2);
    let (mut al, au) = (pack(3, 2), pack(3, 2));
    assert!(Prepared::with_limit(&root, &mb, &gates, &al, &au, 95).is_err());
    let mut prepared = Prepared::with_limit(&root, &mb, &gates, &al, &au, 96).unwrap();
    assert!(prepared.row(2).is_err());
    same_result(prepared.row(1), reference(&root, &mb, &gates, &al, &au, 1));
    al.e = None;
    assert!(Prepared::new(&root, &mb, &gates, &al, &au).is_err());
    assert!(reference(&root, &mb, &gates, &al, &au, 0).is_err());
    al.e = Some(Array2::zeros((2, 3)));
    assert!(Prepared::new(&root, &mb, &gates, &al, &au).is_err());
    al.e = Some(Array2::zeros((3, 2)));
    al.a = Array2::zeros((2, 3)).reversed_axes();
    assert!(!al.a.is_standard_layout());
    assert!(Prepared::new(&root, &mb, &gates, &al, &au).is_err());
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn head_confirm_cache_requalifies_environment_on_each_use() {
    let _env = crate::tests::lock_env_shared();
    let root = root(3);
    let (mb, gates) = batch(2, 2);
    let (al, au) = (pack(3, 2), pack(3, 2));
    let mut prepared = Prepared::new(&root, &mb, &gates, &al, &au).unwrap();
    let before = reference(&root, &mb, &gates, &al, &au, 0).unwrap();
    let rounding = cooperative_tests::HeadDirectedRounding::enter();
    assert!(matches!(prepared.row(0), Err(NyError::SoundnessRefusal(_))));
    assert!(matches!(
        reference(&root, &mb, &gates, &al, &au, 0),
        Err(NyError::SoundnessRefusal(_))
    ));
    drop(rounding);
    assert_eq!(prepared.row(0).unwrap().to_bits(), before.to_bits());
}

#[test]
fn head_confirm_cache_rejects_overflowing_errors_even_with_zero_coefficients() {
    let mut root = root(3);
    root.hi[0] = 2.0;
    let (mut mb, gates) = batch(2, 2);
    mb.wp.fill(0.0);
    mb.wn.fill(0.0);
    let au = pack(3, 2);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, f64::MAX] {
        let mut al = pack(3, 2);
        al.e.as_mut().unwrap()[[0, 0]] = bad;
        assert!(Prepared::new(&root, &mb, &gates, &al, &au).is_err());
        assert!(reference(&root, &mb, &gates, &al, &au, 0).is_err());
    }
    let mut al = pack(3, 2);
    al.eb[0] = f64::MAX;
    al.e.as_mut().unwrap()[[1, 0]] = f64::MAX;
    assert!(Prepared::new(&root, &mb, &gates, &al, &au).is_err());
    assert!(reference(&root, &mb, &gates, &al, &au, 0).is_err());
    al = pack(3, 2);
    al.a[[0, 0]] = f64::INFINITY;
    let mut prepared = Prepared::new(&root, &mb, &gates, &al, &au).unwrap();
    assert!(prepared.row(0).is_err());
    assert!(reference(&root, &mb, &gates, &al, &au, 0).is_err());
}
