// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use num_rational::BigRational as Q;
use num_traits::{Signed, Zero};
use std::time::Duration;

fn q(v: f64) -> Q {
    Q::from_float(v).expect("finite exact dyadic")
}
fn q32(v: f32) -> Q {
    q(ny_core::f32_to_f64_exact(v))
}

fn packed(n_in: usize, p: usize, coordinates: Vec<usize>, channels: usize) -> PackedRows<f32> {
    let slots = coordinates.len() / p;
    let coordinates = Array2::from_shape_vec((p, slots), coordinates).unwrap();
    let m = Array2::zeros((p * channels, slots + 1));
    let d = m.clone();
    PackedRows {
        n_in,
        p,
        coordinates,
        m,
        d,
    }
}

fn bits(a: &Array2<f32>, b: &Array2<f32>) {
    assert_eq!(a.dim(), b.dim());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "coefficient {i}");
    }
}

fn parity(a: &PackedRows<f32>, b: &PackedRows<f32>, h: &[f64]) -> (PackedRows<f32>, Vec<f64>) {
    let (ma, da) = a.materialize();
    let (mb, db) = b.materialize();
    let dense = add_op_signsafe(ma, &mb, da, &db, &[], support_blk(a.n_in + 1), h);
    let out = a
        .add_packed(b, h, None)
        .unwrap()
        .expect("compact Add admitted");
    let (m, d) = out.0.materialize();
    bits(&m, &dense.0);
    bits(&d, &dense.1);
    assert_eq!(
        out.1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        dense.2.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    out
}

// Exact supremum of lower/upper wall violations over the entire signed box.
// D remains signed. Bias is not multiplied by an input coordinate. The oracle
// includes the returned storage allowance and uses original operand bits.
fn excess(
    a: &PackedRows<f32>,
    b: &PackedRows<f32>,
    h: &[f64],
    out: &(PackedRows<f32>, Vec<f64>),
) -> (Q, Q) {
    let (ma, da) = a.materialize();
    let (mb, db) = b.materialize();
    let (m, d) = out.0.materialize();
    let (mut worst_l, mut worst_u) = (None::<Q>, None::<Q>);
    for j in 0..m.nrows() {
        let mut lower = -q(out.1[j]);
        let mut upper = lower.clone();
        for i in 0..=a.n_in {
            let center = q32(ma[[j, i]]) + q32(mb[[j, i]]);
            let radius = q32(da[[j, i]]) + q32(db[[j, i]]);
            let exact_l = &center - &radius;
            let exact_u = center + radius;
            let output_l = q32(m[[j, i]]) - q32(d[[j, i]]);
            let output_u = q32(m[[j, i]]) + q32(d[[j, i]]);
            if i == a.n_in {
                lower += output_l - exact_l;
                upper += exact_u - output_u;
            } else {
                lower += (output_l - exact_l).abs() * q(h[i]);
                upper += (exact_u - output_u).abs() * q(h[i]);
            }
        }
        worst_l = Some(worst_l.map_or(lower.clone(), |old| old.max(lower)));
        worst_u = Some(worst_u.map_or(upper.clone(), |old| old.max(upper)));
    }
    (worst_l.unwrap(), worst_u.unwrap())
}

#[test]
fn compact_add_unions_preserve_dense_arithmetic_and_shared_operands() {
    let mut a = packed(127, 2, vec![1, 9, 63, 3, usize::MAX, usize::MAX], 2);
    let mut b = packed(127, 2, vec![9, 17, 126, usize::MAX, 2, 3, 45, 80], 2);
    for (side, source) in [&mut a, &mut b].into_iter().enumerate() {
        for j in 0..source.m.nrows() {
            for slot in 0..source.m.ncols() {
                if slot != source.bias_column() && source.coordinates[[j % 2, slot]] == usize::MAX {
                    continue;
                }
                source.m[[j, slot]] = (j as f32 - slot as f32 + side as f32) * 0.3;
                source.d[[j, slot]] = (slot as f32 - j as f32 - 1.0) * 0.125;
            }
        }
    }
    a.m[[0, 0]] = -0.0;
    a.d[[0, 0]] = 0.0;
    a.d[[0, 2]] = -0.25;
    a.m[[2, 1]] = f32::from_bits(1);
    a.d[[2, 1]] = -f32::from_bits(1);
    b.m[[2, 0]] = -f32::from_bits(1);
    b.d[[2, 0]] = f32::from_bits(1);
    let before = [a.m.clone(), a.d.clone(), b.m.clone(), b.d.clone()];
    let plan = a.plan_add(&b, None).unwrap().unwrap();
    assert_eq!(plan.coordinates.row(0).to_vec(), vec![1, 9, 17, 63, 126]);
    assert_eq!(
        plan.coordinates.row(1).to_vec(),
        vec![2, 3, 45, 80, usize::MAX]
    );
    assert_eq!(
        plan.left.row(0).to_vec(),
        vec![0, 1, usize::MAX, 2, usize::MAX]
    );
    assert_eq!(
        plan.right.row(0).to_vec(),
        vec![usize::MAX, 0, 1, usize::MAX, 2]
    );
    let h = vec![1.0; 128];
    for (left, right) in [(&a, &b), (&b, &a), (&a, &a)] {
        let out = parity(left, right, &h);
        let (lower, upper) = excess(left, right, &h, &out);
        assert!(lower <= Q::zero() && upper <= Q::zero());
        assert_ne!(out.0.m.as_ptr(), left.m.as_ptr());
        assert_ne!(out.0.d.as_ptr(), left.d.as_ptr());
        assert_eq!(out.0.n_in, 127);
        assert!(out.0.allocated_bytes() < out.0.m.nrows() * 128 * 8);
    }
    for (after, before) in [&a.m, &a.d, &b.m, &b.d].into_iter().zip(&before) {
        bits(after, before);
    }
    let out = a.add_packed(&b, &h, None).unwrap().unwrap();
    assert!(
        out.0.d[[0, 0]] > 0.0,
        "one-sided signed zero must pay its store charge"
    );
    assert!(
        out.0.d[[0, 3]] < 0.0,
        "signed D must survive a one-sided union site"
    );
}

#[test]
fn compact_add_exact_walls_detect_signed_and_subnormal_erasure() {
    for (role, value, magnitude) in [
        (1, -0.25, 1.0),
        (0, f32::from_bits(1), 2.0f64.powi(200)),
        (1, f32::from_bits(1), 2.0f64.powi(200)),
    ] {
        let mut a = packed(127, 1, vec![63], 1);
        let b = packed(127, 1, vec![5], 1);
        if role == 0 {
            a.m[[0, 0]] = value;
        } else {
            a.d[[0, 0]] = value;
        }
        let mut h = vec![1.0; 128];
        h[63] = magnitude;
        let out = parity(&a, &b, &h);
        let (lower, upper) = excess(&a, &b, &h, &out);
        assert!(lower <= Q::zero() && upper <= Q::zero());
        let erased = packed(127, 1, vec![63], 1);
        let mutant = erased.add_packed(&b, &h, None).unwrap().unwrap();
        let (lower, upper) = excess(&a, &b, &h, &mutant);
        assert!(
            lower > Q::zero() || upper > Q::zero(),
            "erasure mutant escaped whole-box oracle"
        );
    }
}

#[test]
fn compact_add_empty_union_keeps_original_width_and_excluded_frame_floor() {
    let a = packed(127, 1, vec![], 1);
    let b = packed(127, 1, vec![], 1);
    let mut h = vec![1.0; 128];
    h[0] = 2.0f64.powi(900);
    let out = parity(&a, &b, &h);
    assert_eq!(out.0.coordinates.dim(), (1, 0));
    assert_eq!(out.0.m.dim(), (1, 1));
    assert_eq!(out.0.n_in, 127);
    assert!(
        q32(out.0.d[[0, 0]]) >= q(2.0f64.powi(-120)),
        "absent coordinate must contribute to floor"
    );
    assert!(out.1[0] > 0.0);
    let (lower, upper) = excess(&a, &b, &h, &out);
    assert!(lower <= Q::zero() && upper <= Q::zero());
    let narrow_frame = vec![1.0; 128];
    let mutant = a.add_packed(&b, &narrow_frame, None).unwrap().unwrap();
    assert!(
        q32(mutant.0.d[[0, 0]]) < q(2.0f64.powi(-120)),
        "floor witness must distinguish omission"
    );
}

#[test]
fn compact_add_refuses_invalid_geometry_frames_and_nonfinite_publication() {
    let make = || packed(127, 1, vec![3, usize::MAX], 1);
    let a = make();
    let h = vec![1.0; 128];
    let mut b = make();
    b.n_in = 126;
    assert!(a.add_packed(&b, &h, None).unwrap().is_none());
    let mut b = make();
    b.p = 0;
    assert!(a.add_packed(&b, &h, None).unwrap().is_none());
    let mut b = make();
    b.coordinates[[0, 1]] = 3;
    assert!(a.add_packed(&b, &h, None).unwrap().is_none());
    let mut b = make();
    b.coordinates[[0, 0]] = usize::MAX;
    b.coordinates[[0, 1]] = 7;
    assert!(a.add_packed(&b, &h, None).unwrap().is_none());
    let mut b = make();
    b.coordinates[[0, 0]] = 127;
    assert!(a.add_packed(&b, &h, None).unwrap().is_none());
    let mut b = make();
    b.m[[0, 1]] = -0.0;
    assert!(
        a.add_packed(&b, &h, None).unwrap().is_none(),
        "padding is raw +0"
    );
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut b = make();
        b.d[[0, 0]] = bad;
        assert!(a.add_packed(&b, &h, None).is_err());
    }
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        let mut invalid = h.clone();
        invalid[99] = bad;
        assert!(
            a.add_packed(&a, &invalid, None).is_err(),
            "absent frame still validated"
        );
    }
    assert!(a.add_packed(&a, &h[..127], None).is_err());
    assert!(a
        .add_packed(&a, &h, Some(Instant::now() - Duration::from_secs(1)))
        .is_err());
    let mut b = make();
    b.m[[0, 0]] = f32::MAX;
    assert!(
        b.add_packed(&b, &h, None).is_err(),
        "finite operands can overflow storage"
    );
    let full = packed(8, 1, (0..8).collect(), 1);
    assert!(full.add_packed(&full, &[1.0; 9], None).unwrap().is_none());
    let wide = PackedRows {
        n_in: 127,
        p: 1,
        coordinates: a.coordinates.clone(),
        m: a.m.mapv(f64::from),
        d: a.d.mapv(f64::from),
    };
    assert!(wide.add_packed(&wide, &h, None).unwrap().is_none());
    with_legacy_rounding_charge_for_test(|| assert!(a.add_packed(&a, &h, None).unwrap().is_none()));
}
