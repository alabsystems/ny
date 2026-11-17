// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use num_rational::BigRational as Q;
use num_traits::{Signed, Zero};

type Output = (Array2<f32>, Array2<f32>, Vec<f64>);

fn run(m: &[f32], d: &[f32], a: f64, s: f64, c: f64, h: &[f64], owned: bool) -> Output {
    let m = Array2::from_shape_vec((1, m.len()), m.to_vec()).unwrap();
    let d = Array2::from_shape_vec((1, d.len()), d.to_vec()).unwrap();
    let n = h.len() - 1;
    if owned {
        apply_gates_owned(m, d, &[a], &[s], &[c], n, RoundMode::Outward, None, 1, h)
    } else {
        apply_gates(&m, &d, &[a], &[s], &[c], n, RoundMode::Outward, None, 1, h)
    }
}

fn q(v: f64) -> Q {
    Q::from_float(v).expect("finite exact dyadic")
}

fn q32(v: f32) -> Q {
    q(ny_core::f32_to_f64_exact(v))
}

// Exact-rational wall excess over the full signed box after gstore allowance.
// That allowance cannot justify dropping source coefficients or inherited debt.
fn excess(m: &[f32], d: &[f32], a: f64, s: f64, c: f64, h: &[f64], out: &Output) -> (Q, Q) {
    let n = m.len() - 1;
    let mut lower = -q(out.2[0]);
    let mut upper = lower.clone();
    for i in 0..=n {
        let l = (q32(m[i]) - q32(d[i])) * q(a);
        let mut u = (q32(m[i]) + q32(d[i])) * q(s);
        if i == n {
            u += q(c);
        }
        let ml = q32(out.0[[0, i]]) - q32(out.1[[0, i]]);
        let mu = q32(out.0[[0, i]]) + q32(out.1[[0, i]]);
        if i == n {
            lower += ml - l;
            upper += u - mu;
        } else {
            lower += (ml - l).abs() * q(h[i]);
            upper += (u - mu).abs() * q(h[i]);
        }
    }
    (lower, upper)
}

#[test]
fn zero_gate_coefficients_keep_bias_and_owned_allocations() {
    // The last coefficient shares live block 63 with the bias.
    let n = 127;
    let m = Array2::<f32>::zeros((1, n + 1));
    let d = m.clone();
    let h = vec![1.0; n + 1];
    let mask = [1u64 << 63];
    let borrowed = apply_gates(
        &m,
        &d,
        &[0.37],
        &[0.5],
        &[0.75],
        n,
        RoundMode::Outward,
        Some(&mask),
        support_blk(n + 1),
        &h,
    );
    let (mp, dp) = (m.as_ptr(), d.as_ptr());
    let owned = apply_gates_owned(
        m,
        d,
        &[0.37],
        &[0.5],
        &[0.75],
        n,
        RoundMode::Outward,
        Some(&mask),
        support_blk(n + 1),
        &h,
    );
    assert_eq!(mp, owned.0.as_ptr());
    assert_eq!(dp, owned.1.as_ptr());
    for result in [&borrowed, &owned] {
        for i in 0..n {
            assert_eq!(result.0[[0, i]].to_bits(), 0);
            assert_eq!(result.1[[0, i]].to_bits(), 0);
        }
        assert_eq!(result.0[[0, n]], 0.375);
        assert!(result.1[[0, n]] >= 0.375);
        assert!(result.2[0] > 0.0, "bias storage still contributes");
        let (lo, hi) = excess(
            &vec![0.0; n + 1],
            &vec![0.0; n + 1],
            0.37,
            0.5,
            0.75,
            &h,
            result,
        );
        assert!(lo <= Q::zero() && hi <= Q::zero());
    }
    for (a, b) in [(&borrowed.0, &owned.0), (&borrowed.1, &owned.1)] {
        assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
    }
    assert_eq!(borrowed.2[0].to_bits(), owned.2[0].to_bits());
}

#[test]
fn zero_gate_coefficients_reject_signed_zero_uncertainty_and_nonfinite_operands() {
    let tiny = f32::from_bits(1);
    for owned in [false, true] {
        for (m, d) in [
            (-0.0, 0.0),
            (0.0, -0.0),
            (0.0, 0.25),
            (0.0, -0.25),
            (tiny, 0.0),
            (-tiny, 0.0),
            (0.0, tiny),
            (0.0, -tiny),
            (f32::INFINITY, 0.0),
            (f32::NAN, 0.0),
        ] {
            let out = run(&[m, 0.0], &[d, 0.0], 1.0, 1.0, 0.0, &[1.0, 1.0], owned);
            assert!(
                out.0[[0, 0]].to_bits() != 0 || out.1[[0, 0]].to_bits() != 0,
                "nonqualifying input erased: m={m:?}, d={d:?}"
            );
        }
        for (a, s, h) in [
            (f64::INFINITY, 1.0, 1.0),
            (1.0, f64::NAN, 1.0),
            (1.0, 1.0, f64::INFINITY),
            (1.0, 1.0, f64::NAN),
            (1.0, 1.0, -1.0),
        ] {
            let out = run(&[0.0; 2], &[0.0; 2], a, s, 0.0, &[h, 1.0], owned);
            assert!(out.0[[0, 0]].to_bits() != 0 || out.1[[0, 0]].to_bits() != 0);
        }
        let out = run(
            &[0.0; 2],
            &[0.0; 2],
            1.0,
            1.0,
            f64::INFINITY,
            &[1.0; 2],
            owned,
        );
        assert!(
            !out.0[[0, 1]].is_finite(),
            "bias failure must remain visible"
        );
    }
}

#[test]
fn zero_gate_coefficients_enclose_exact_signed_walls_and_detect_erasure_mutants() {
    let tiny = f32::from_bits(1);
    let h = [37.5, 2.0f64.powi(200), 2.0, 1.0];
    for owned in [false, true] {
        for (m, d) in [
            ([0.0, tiny, -0.7, 0.125], [0.0, 0.0, -0.25, -0.0625]),
            ([0.0, -tiny, 0.7, 0.0], [0.0, tiny, 0.25, 0.0]),
        ] {
            for a in [0.0, 1.0, 0.37, 0.999_999] {
                for s in [0.0, 1.0, 1e-6, 0.5] {
                    for c in [0.0, 0.4] {
                        let out = run(&m, &d, a, s, c, &h, owned);
                        assert_eq!(out.0[[0, 0]].to_bits(), 0);
                        assert_eq!(out.1[[0, 0]].to_bits(), 0);
                        let (lo, hi) = excess(&m, &d, a, s, c, &h, &out);
                        assert!(
                            lo <= Q::zero() && hi <= Q::zero(),
                            "exact signed-box failure: a={a}, s={s}, c={c}, owned={owned}"
                        );
                    }
                }
            }
        }
        // Erasing D coefficients or DAZ-sensitive subnormals must fail
        // against the stored source, even with the mutant's own storage dust.
        for (m, d, magnitude) in [
            (0.0, -0.25, 1.0),
            (tiny, 0.0, 2.0f64.powi(200)),
            (-tiny, 0.0, 2.0f64.powi(200)),
        ] {
            let hh = [magnitude, 1.0];
            let bad = run(&[0.0; 2], &[0.0; 2], 1.0, 1.0, 0.0, &hh, owned);
            let (lo, hi) = excess(&[m, 0.0], &[d, 0.0], 1.0, 1.0, 0.0, &hh, &bad);
            assert!(
                lo > Q::zero() || hi > Q::zero(),
                "erasure mutant escaped the oracle"
            );
        }
    }
}
