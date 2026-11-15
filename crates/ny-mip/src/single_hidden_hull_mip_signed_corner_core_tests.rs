// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Numerical/theorem controls, not authentic constructor or replay integration.
use super::*;

#[derive(Clone)]
struct Affine {
    q: Vec<I>,
    constant: I,
}
fn point(q: &[f64], constant: f64) -> Affine {
    Affine {
        q: q.iter().map(|&v| I::point(v).unwrap()).collect(),
        constant: I::point(constant).unwrap(),
    }
}
fn raw(a: &Affine, lo: &[f64], hi: &[f64]) -> f64 {
    a.q.iter()
        .zip(lo.iter().zip(hi))
        .fold(a.constant.lo, |v, (&q, (&l, &u))| {
            finite(add_down(v, q.minimum(l, u).unwrap())).unwrap()
        })
}
fn checked_constant(parts: &[Vec<Affine>], lo: &[f64], hi: &[f64], residual: f64) -> f64 {
    let sum = parts.iter().fold(0., |sum, masks| {
        let m = masks
            .iter()
            .map(|a| raw(a, lo, hi))
            .fold(f64::INFINITY, f64::min);
        finite(add_down(sum, m)).unwrap()
    });
    finite(add_down(sum, residual)).unwrap()
}
fn build(
    lo: &[f64],
    hi: &[f64],
    bits: &[u8],
    k: usize,
    parts: &[Vec<Affine>],
    g: &[f64],
    d: f64,
    residual: f64,
) -> Data {
    let constant = checked_constant(parts, lo, hi, residual);
    let mut rank = Ranker::new(lo, hi, bits, k).unwrap();
    for (c, masks) in parts.iter().enumerate() {
        assert_eq!(masks.len(), 1 << bits[c]);
        for (m, a) in masks.iter().enumerate() {
            rank.push(c, m, &a.q, raw(a, lo, hi), &mut || true).unwrap();
        }
    }
    let axes = rank.finish(constant, d, g, &mut || true).unwrap();
    let mut collect = Collector::new(axes).unwrap();
    for (c, masks) in parts.iter().enumerate() {
        for (m, a) in masks.iter().enumerate() {
            collect
                .push(c, m, &a.q, raw(a, lo, hi), &mut || true)
                .unwrap();
        }
    }
    collect.finish(constant, d, &mut || true).unwrap()
}
fn price(data: &Data, g: &[f64], on: &[usize], off: &[usize]) -> f64 {
    let q: Vec<_> = g.iter().map(|&v| I::point(v).unwrap()).collect();
    data.lower(&q, I::zero(), on, off, &mut || true).unwrap()
}
fn near_lower(actual: f64, exact: f64) {
    assert!(actual <= exact, "lower {actual} exceeds exact {exact}");
    assert!(
        exact - actual < 1e-10,
        "unexpected loss {actual} versus {exact}"
    );
}

fn cross_parts() -> Vec<Vec<Affine>> {
    // F=9/4-|x+y|-|x-y|. Each component is a pair of opposite
    // negative ReLUs; these four affine formulas are exact on CLOSED phases.
    [1., -1.]
        .into_iter()
        .map(|sign| {
            (0..4)
                .map(|m| {
                    let t = f64::from((m & 2 != 0) as u8) - f64::from((m & 1 != 0) as u8);
                    point(&[t, sign * t], 0.)
                })
                .collect()
        })
        .collect()
}

#[test]
fn corner_two_axes_escape_exact_one_axis_stall_and_certify_closed_graph() {
    let parts = cross_parts();
    let one = build(&[-1.; 2], &[1.; 2], &[2, 2], 1, &parts, &[0.; 2], 2.25, 0.);
    let two = build(&[-1.; 2], &[1.; 2], &[2, 2], 2, &parts, &[0.; 2], 2.25, 0.);
    assert_eq!(one.selected(), &[0]); // Exact rank tie, deterministic index.
    near_lower(price(&one, &[0.; 2], &[0, 0], &[0, 0]), -1.75);
    let bound = price(&two, &[0.; 2], &[0, 0], &[0, 0]);
    near_lower(bound, 0.25);
    assert!(bound > 0.);
    for x in [-1f64, -0.5, 0., 0.5, 1.] {
        for y in [-1f64, -0.5, 0., 0.5, 1.] {
            let graph = 2.25 - (x + y).abs() - (x - y).abs();
            assert!(bound <= graph);
            for (c, sign) in [1., -1.].into_iter().enumerate() {
                let a = x + sign * y;
                for m in 0..4 {
                    let active0 = m & 1 != 0;
                    let active1 = m & 2 != 0;
                    let closed0 = if active0 { a >= 0. } else { a <= 0. };
                    let closed1 = if active1 { -a >= 0. } else { -a <= 0. };
                    if closed0 && closed1 {
                        let q = &parts[c][m].q;
                        assert_eq!(q[0].lo * x + q[1].lo * y, -a.abs());
                    }
                }
            }
        }
    }
}

#[test]
fn corner_conditioned_child_closes_when_root_bound_is_negative() {
    // F=1/4-ReLU(x). The inactive child is safe although the root is not.
    let parts = vec![vec![point(&[0.], 0.), point(&[-1.], 0.)]];
    let data = build(&[-1.], &[1.], &[1], 1, &parts, &[0.], 0.25, 0.);
    near_lower(price(&data, &[0.], &[0], &[0]), -0.75);
    let child = price(&data, &[0.], &[0], &[1]);
    near_lower(child, 0.25);
    assert!(child > 0.);
    for x in [-1f64, -0.5, 0.] {
        assert!(child <= 0.25 - x.max(0.));
    }
    // Both closed phase choices at zero remain valid, including the weaker one.
    assert!(price(&data, &[0.], &[1], &[0]) <= 0.25);
}

#[test]
fn corner_same_component_conditioned_gains_are_maximized_never_summed() {
    // Valid scalar minorants of the constant graph F=1 on a degenerate box;
    // every zero preactivation admits both closed phase choices. This is a
    // pure envelope control, not a claim about constructor-generated alpha.
    let parts = vec![vec![
        point(&[0.], 1.),
        point(&[0.], 0.),
        point(&[0.], 0.),
        point(&[0.], -10.),
    ]];
    let data = build(&[0.], &[0.], &[2], 1, &parts, &[0.], 0., 0.);
    let root = price(&data, &[0.], &[0], &[0]);
    let first = price(&data, &[0.], &[0], &[1]);
    let second = price(&data, &[0.], &[0], &[2]);
    let both = price(&data, &[0.], &[0], &[3]);
    near_lower(root, -10.);
    near_lower(first, 0.);
    near_lower(second, 0.);
    near_lower(both, 0.);
    assert!(root + (first - root) + (second - root) > 1.);
    assert!(both <= 1.);
}

#[test]
fn corner_offset_keeps_old_residual_exported_g_and_adds_source_d_once() {
    let parts = vec![vec![point(&[0.], 7.), point(&[0.], 7.)]];
    let data = build(&[-1.], &[1.], &[1], 1, &parts, &[2.], 1., -0.25);
    near_lower(price(&data, &[2.], &[0], &[0]), 5.75);
    // A legal active sign term -beta*x is supplied once in global coefficients;
    // this method does not also add the ordinary child intercept/component gain.
    let q = [I::point(1.).unwrap()];
    near_lower(
        data.lower(&q, I::zero(), &[1], &[0], &mut || true).unwrap(),
        6.75,
    );
}

#[test]
fn corner_interval_gray_updates_enclose_nonsymmetric_endpoint_products() {
    let q = vec![I { lo: -1.5, hi: -0.5 }, I { lo: 0.25, hi: 0.75 }];
    let a = Affine {
        q,
        constant: I::point(0.5).unwrap(),
    };
    let parts = vec![vec![a.clone(), a]];
    let data = build(
        &[-0.75, 0.25],
        &[0.5, 1.25],
        &[1],
        2,
        &parts,
        &[0., 0.],
        0.,
        0.,
    );
    let lower = price(&data, &[0., 0.], &[0], &[0]);
    for a in [-1.5, -1., -0.5] {
        for b in [0.25, 0.5, 0.75] {
            for x in [-0.75, -0.125, 0.5] {
                for y in [0.25, 0.75, 1.25] {
                    assert!(lower <= 0.5 + a * x + b * y);
                }
            }
        }
    }
}

#[test]
fn corner_refuses_bad_shapes_partial_duplicate_and_nonfinite_coverage() {
    assert!(Ranker::new(&[], &[], &[1], 1).is_none());
    assert!(Ranker::new(&[0.], &[1.], &[0], 1).is_none());
    assert!(Ranker::new(&[1.], &[0.], &[1], 1).is_none());
    assert!(Ranker::new(&[0.], &[1.], &[1], 2).is_none());
    let q = [I::zero()];
    let mut rank = Ranker::new(&[-1.], &[1.], &[1], 1).unwrap();
    rank.push(0, 0, &q, 0., &mut || true).unwrap();
    assert!(rank.push(0, 0, &q, 0., &mut || true).is_none());
    assert!(rank.push(0, 1, &q, 0., &mut || true).is_none());
    assert!(rank.finish(0., 0., &[0.], &mut || true).is_none());
    let mut rank = Ranker::new(&[-1.], &[1.], &[1], 1).unwrap();
    rank.push(0, 0, &q, 0., &mut || true).unwrap();
    assert!(rank.finish(0., 0., &[0.], &mut || true).is_none());
    let mut rank = Ranker::new(&[-1.], &[1.], &[1], 1).unwrap();
    assert!(rank.push(0, 0, &q, f64::NAN, &mut || true).is_none());
    assert!(rank.push(0, 0, &q, 0., &mut || true).is_none());
}

#[test]
fn corner_cancellation_is_sticky_and_immutable_lower_refuses_conflicts() {
    let q = [I::zero()];
    let mut rank = Ranker::new(&[-1.], &[1.], &[1], 1).unwrap();
    assert!(rank.push(0, 0, &q, 0., &mut || false).is_none());
    assert!(rank.push(0, 0, &q, 0., &mut || true).is_none());
    let mut rank = Ranker::new(&[-1.], &[1.], &[1], 1).unwrap();
    for m in 0..2 {
        rank.push(0, m, &q, 0., &mut || true).unwrap();
    }
    let axes = rank.finish(0., 0., &[0.], &mut || true).unwrap();
    let mut collect = Collector::new(axes).unwrap();
    assert!(collect.push(0, 0, &q, 0., &mut || false).is_none());
    assert!(collect.push(0, 0, &q, 0., &mut || true).is_none());
    assert!(collect.finish(0., 0., &mut || true).is_none());
    let parts = vec![vec![point(&[0.], 0.), point(&[0.], 0.)]];
    let data = build(&[-1.], &[1.], &[1], 1, &parts, &[0.], 0., 0.);
    assert!(data
        .lower(&q, I::zero(), &[1], &[1], &mut || true)
        .is_none());
    assert!(data
        .lower(&q, I::zero(), &[2], &[0], &mut || true)
        .is_none());
    assert!(data
        .lower(&q, I::zero(), &[0], &[0], &mut || false)
        .is_none());
    // Data is immutable; extension credits are burned by the outer owner even
    // when a lower call refuses. Retry policy is deliberately not in this core.
}

#[test]
fn corner_complete_256_masks_128_corners_and_seal_word_census() {
    let parts = vec![(0usize..256)
        .map(|m| point(&[0.; 7], m.count_ones() as f64))
        .collect()];
    let mut data = build(&[-1.; 7], &[1.; 7], &[8], 7, &parts, &[0.; 7], 0., 0.);
    near_lower(price(&data, &[0.; 7], &[0], &[0]), 0.);
    // Unary summaries intentionally yield1, while the exact all-on mask is8.
    near_lower(price(&data, &[0.; 7], &[255], &[0]), 1.);
    let seal: Vec<_> = data.words().collect();
    assert_eq!(seal.len(), 6 + 14 + 1 + 7 + 17 * 128);
    data.values[128] = 42.;
    assert!(!data.words().eq(seal.iter().copied()));
}
