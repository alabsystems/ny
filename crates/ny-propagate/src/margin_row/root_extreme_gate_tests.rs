// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Independent real endpoint controls for exceptional ReLU gate arithmetic.
use super::*;
use crate::margin_row::head_group_source::HeadGroupSourceGuard;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use num_rational::BigRational as Q;
use num_traits::Zero;
use std::time::Duration;

fn q(value: f64) -> Q {
    Q::from_float(value).unwrap()
}

fn assert_exact_upper_line(l: f64, u: f64, s: f64, c: f64) {
    assert!(s.is_finite() && c.is_finite());
    assert!((0.0..=1.0).contains(&s) && c >= 0.0);
    let lo = q(l);
    let hi = q(u);
    // Endpoints and zero prove every segment; extra rational interior
    // vertices avoid any dependency on production endpoint arithmetic.
    let mut vertices = vec![lo.clone(), Q::zero(), hi.clone()];
    for k in 1..8 {
        vertices.push((&lo * q(f64::from(8 - k)) + &hi * q(f64::from(k))) / q(8.0));
    }
    for y in vertices {
        let line = q(s) * &y + q(c);
        assert!(
            line >= y.max(Q::zero()),
            "invalid line on [{l},{u}] with s={s},c={c}"
        );
    }
}

fn legacy_gates(l: f64, u: f64) -> (f64, f64, f64) {
    if l >= 0.0 {
        (1.0, 1.0, 0.0)
    } else if u > 0.0 {
        (
            if u >= -l { 1.0 } else { 0.0 },
            u / (u - l),
            -u * l / (u - l),
        )
    } else {
        (0.0, 0.0, 0.0)
    }
}

fn legacy_repaired(l: f64, u: f64, s: f64, mut c: f64) -> f64 {
    if (s == 0.0 && c == 0.0 && u <= 0.0) || (s == 1.0 && c == 0.0 && l >= 0.0) {
        return c;
    }
    if c < 0.0 {
        c = 0.0;
    }
    let sl = s * l;
    let su = s * u;
    let head = 4.0 * UNIT * (sl.abs() + su.abs() + c.abs() + u.abs());
    let d1 = -(sl + c) + head;
    let d2 = u - (su + c) + head;
    let bump = d1.max(d2).max(0.0);
    if bump > 0.0 {
        c = next_up(next_up(c + bump));
    }
    c
}

#[test]
fn root_extreme_gates_preserve_every_finite_legacy_slope_intercept_and_repair_bit() {
    let _env = crate::tests::lock_env_shared();
    ny_core::require_f64_interval_proof_environment().unwrap();
    let tiny = f64::from_bits(1);
    for (l, u) in [
        (-5.0, 3.0),
        (-0.125, 0.625),
        (-1e150, 1e150),
        (-1e-100, 3e-100),
        (-tiny, tiny),
        (-4.0 * tiny, 3.0 * tiny),
        (-f64::MAX, -0.0),
        (-0.0, f64::MAX),
    ] {
        let expected = legacy_gates(l, u);
        assert!(expected.1.is_finite() && expected.2.is_finite());
        let (alpha, mut slopes, mut intercepts) = gates_from_box(&[l], &[u]);
        assert_eq!(alpha[0].to_bits(), expected.0.to_bits());
        assert_eq!(slopes[0].to_bits(), expected.1.to_bits());
        assert_eq!(intercepts[0].to_bits(), expected.2.to_bits());
        let repaired = legacy_repaired(l, u, expected.1, expected.2);
        assert!(repaired.is_finite());
        repair_upper_lines(&[l], &[u], &mut slopes, &mut intercepts);
        assert_eq!(slopes[0].to_bits(), expected.1.to_bits());
        assert_eq!(intercepts[0].to_bits(), repaired.to_bits());
    }
}

#[test]
fn root_extreme_gates_numerator_overflow_has_finite_rational_upper_line() {
    let _env = crate::tests::lock_env_shared();
    ny_core::require_f64_interval_proof_environment().unwrap();
    for (l, u) in [(-1e300, 1e300), (-1e300, 1e299), (-1e299, 1e300)] {
        let old = legacy_gates(l, u);
        assert!(old.1.is_finite() && !old.2.is_finite());
        let (_, mut s, mut c) = gates_from_box(&[l], &[u]);
        assert_exact_upper_line(l, u, s[0], c[0]);
        repair_upper_lines(&[l], &[u], &mut s, &mut c);
        assert_exact_upper_line(l, u, s[0], c[0]);
    }
}

#[test]
fn root_extreme_gates_denominator_and_repair_head_overflow_stay_finite_and_certified() {
    let _env = crate::tests::lock_env_shared();
    ny_core::require_f64_interval_proof_environment().unwrap();
    for (l, u) in [
        (-f64::MAX, f64::MAX),
        (-1e308, f64::MAX),
        (-f64::MAX, 1e308),
    ] {
        assert!(!(u - l).is_finite());
        let (_, mut s, mut c) = gates_from_box(&[l], &[u]);
        assert_exact_upper_line(l, u, s[0], c[0]);
        assert!(!legacy_repaired(l, u, s[0], c[0]).is_finite());
        let slope_bits = s[0].to_bits();
        repair_upper_lines(&[l], &[u], &mut s, &mut c);
        assert_eq!(s[0].to_bits(), slope_bits);
        assert_exact_upper_line(l, u, s[0], c[0]);
    }
}

#[test]
fn root_extreme_gates_finite_legacy_chord_survives_repair_head_only_overflow() {
    let _env = crate::tests::lock_env_shared();
    ny_core::require_f64_interval_proof_environment().unwrap();
    for (l, u) in [(-1.0, 1e308), (-f64::from_bits(1), f64::MAX)] {
        let old = legacy_gates(l, u);
        assert!(old.1.is_finite() && old.2.is_finite());
        let (alpha, mut s, mut c) = gates_from_box(&[l], &[u]);
        assert_eq!(alpha[0].to_bits(), old.0.to_bits());
        assert_eq!(s[0].to_bits(), old.1.to_bits());
        assert_eq!(c[0].to_bits(), old.2.to_bits());
        assert!(!legacy_repaired(l, u, s[0], c[0]).is_finite());
        repair_upper_lines(&[l], &[u], &mut s, &mut c);
        assert_eq!(s[0].to_bits(), old.1.to_bits());
        assert_exact_upper_line(l, u, s[0], c[0]);
    }
}

#[test]
fn root_extreme_gates_exceptional_endpoint_repair_includes_signed_underflow_and_tiny_ratios() {
    let _env = crate::tests::lock_env_shared();
    ny_core::require_f64_interval_proof_environment().unwrap();
    let tiny = f64::from_bits(1);
    for (l, u, slope) in [
        (-tiny, tiny, 0.5),
        (-3.0 * tiny, 2.0 * tiny, 0.25),
        (-tiny, f64::MAX, 1.0),
        (-f64::MAX, tiny, 0.0),
    ] {
        let mut s = [slope];
        let mut c = [f64::INFINITY];
        repair_upper_lines(&[l], &[u], &mut s, &mut c);
        assert_eq!(s[0].to_bits(), slope.to_bits());
        assert_exact_upper_line(l, u, s[0], c[0]);
        let (stable_s, stable_c) = finite_upper_line_from_endpoints(l, u, None).unwrap();
        assert_exact_upper_line(l, u, stable_s, stable_c);
    }
    assert!(finite_upper_line_from_endpoints(-f64::INFINITY, 1.0, None).is_none());
    assert!(finite_upper_line_from_endpoints(-1.0, f64::NAN, None).is_none());
}

#[test]
fn root_extreme_gates_original_large_conv_subnormal_conv_root_reaches_real_completion() {
    let _env = crate::tests::lock_env_shared();
    let _authority = HeadGroupSourceGuard::enter(true);
    let tiny = f64::from_bits(1);
    let conv = |input, weight| TwinOpSpec::Conv {
        input,
        weight: vec![weight],
        bias: vec![0.0],
        bias_err: vec![0.0],
        weight_rel_err: 0.0,
        kernel: (1, 1, 1, 1),
        stride: (1, 1),
        pads: (0, 0, 0, 0),
        ishape: (1, 1, 1),
        oshape: (1, 1, 1),
    };
    let net = TwinNet::compile(&TwinSpec {
        n_in: 1,
        ops: vec![
            conv(0, 1e300),
            TwinOpSpec::Relu { input: 1 },
            conv(2, tiny),
            TwinOpSpec::ChannelAffine {
                input: 3,
                scale: vec![0.5],
                shift: vec![0.0],
                scale_rel_err: 0.0,
                shift_err: vec![0.0],
                shape: (1, 1, 1),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Flatten { input: 5 },
            TwinOpSpec::Gemm {
                input: 6,
                weight: vec![1.0, -1.0],
                bias: vec![0.0, 0.0],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
            TwinOpSpec::Relu { input: 7 },
            TwinOpSpec::Gemm {
                input: 8,
                weight: vec![1.0, -1.0, -0.5, 0.75],
                bias: vec![0.0, 0.0],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, 2),
            },
        ],
    })
    .unwrap();
    let at = Instant::now() + Duration::from_secs(30);
    let root = RootGates::build_prec_store(
        &net,
        &[-1.0],
        &[1.0],
        RoundMode::Outward,
        Some(at),
        false,
        false,
    )
    .unwrap();
    assert_eq!(root.layers.len(), 2);
    assert!(root.head_group_source.is_some());
    for gate in &root.layers {
        assert!(gate
            .l
            .iter()
            .chain(&gate.u)
            .chain(&gate.s)
            .chain(&gate.c)
            .all(|v| v.is_finite()));
        if gate.l[0] < 0.0 && gate.u[0] > 0.0 {
            assert_exact_upper_line(gate.l[0], gate.u[0], gate.s[0], gate.c[0]);
        }
    }
    for x in [-1.0, 0.0, 0.5, 1.0] {
        let first = q(1e300) * q(x);
        let second = first.clone().max(Q::zero()) * q(tiny) * q(0.5);
        for (gate, original) in root.layers.iter().zip([first, second]) {
            assert!(q(gate.l[0]) <= original && original <= q(gate.u[0]));
        }
    }
}
