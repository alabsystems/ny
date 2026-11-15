// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Independent authored controls, not an original-benchmark performance claim.
use super::*;
use num_rational::BigRational as Q;
use num_traits::Zero;

pub(super) fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
pub(super) fn q(x: f64) -> Q {
    Q::from_float(x).unwrap()
}

pub(super) fn cycle(k: usize, bias: f64, permuted: bool) -> HullMarginProblem {
    let mut w = vec![0.; k * k];
    for h in 0..k {
        let edge = if permuted { k - 1 - h } else { h };
        let a = if permuted { k - 1 - edge } else { edge };
        let b = if permuted {
            k - 1 - (edge + 1) % k
        } else {
            (edge + 1) % k
        };
        w[h * k + a] = 1.;
        w[h * k + b] = -1.;
    }
    HullMarginProblem::new(
        w,
        vec![0.; k],
        k,
        &vec![-1.; k],
        &[bias],
        1,
        &[(0, 1.)],
        0.,
        vec![-1.; k],
        vec![1.; k],
    )
    .unwrap()
}

pub(super) fn manual(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    let mut m = HullMultipliers::zeros(p);
    for j in 0..p.hidden_dim {
        if p.c[j] > 0. {
            m.alpha[j] = 1.;
        }
        if p.c[j] < 0. {
            for i in 0..p.input_dim {
                m.g[j * p.input_dim + i] = 0.5 * p.w_row(j)[i];
            }
        }
    }
    let lower = certify_hull_lower_bound(p, &m).unwrap();
    prepare(p, m, lower, end(), &mut || true).unwrap()
}

/// Literal exact arithmetic over every branch and every original box corner.
/// No Source coefficient cache, selector, interval helper or LP is consulted.
pub(super) fn exact_support(p: &HullMarginProblem, heads: &[usize], g: &[f64]) -> Q {
    let mut best: Option<Q> = None;
    for mask in 0..1usize << heads.len() {
        let mut b = Q::zero();
        let mut v: Vec<_> = g.iter().map(|x| -q(*x)).collect();
        for (bit, &j) in heads.iter().enumerate() {
            if mask & (1 << bit) == 0 {
                continue;
            }
            let a = q(-p.c[j]);
            b += &a * q(p.b1[j]);
            for i in 0..p.input_dim {
                v[i] += &a * q(p.w_row(j)[i]);
            }
        }
        let value = b
            + (0..p.input_dim)
                .map(|i| (&v[i] * q(p.lower[i])).max(&v[i] * q(p.upper[i])))
                .sum::<Q>();
        if best.as_ref().is_none_or(|old| value > *old) {
            best = Some(value);
        }
    }
    best.unwrap()
}

/// Odd3 plus an independent pair, one genuine positive head and a final
/// stable negative singleton. Original graph min is bias-1.5.
pub(super) fn mixed(bias: f64) -> HullMarginProblem {
    let w = vec![
        1., -1., 0., 0., 0., 0., 1., -1., 0., 0., -1., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0.,
        0., 1., -1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
    ];
    HullMarginProblem::new(
        w,
        vec![0., 0., 0., 0., 0., 2., 1.],
        5,
        &[-1., -1., -1., -1., -1., 1., -0.5],
        &[bias],
        1,
        &[(0, 1.)],
        0.,
        vec![-1.; 5],
        vec![1.; 5],
    )
    .unwrap()
}

pub(super) fn pair_baseline(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    let mut cuts = manual(p);
    joint::discover_compact(&mut cuts, end(), &mut || true);
    assert!(
        cuts.compact.is_some(),
        "authored independent pair must improve the baseline"
    );
    cuts
}

pub(super) fn budget() -> Budget {
    Budget::new(end(), Duration::ZERO).unwrap()
}

#[test]
fn group_support_odd3_beats_every_pair_with_same_individual_geometry() {
    let p = cycle(3, 2.25, false);
    let cuts = manual(&p);
    let b = budget();
    for heads in [[0, 1], [0, 2], [1, 2]] {
        let row = support(&cuts, &heads, &b, &mut || true).unwrap();
        assert!(row.upper() >= 2. && row.upper() < 2.001);
        assert_eq!(row.mask_count(), 4);
    }
    let row = support(&cuts, &[0, 1, 2], &b, &mut || true).unwrap();
    assert_eq!(row.heads(), &[0, 1, 2]);
    assert_eq!(row.mask_count(), 8);
    assert!(row.upper() >= 2. && row.upper() < 2.001);
    let aggregate = compact::build_group_aggregate(&cuts, vec![row], &b, &mut || true).unwrap();
    let floor = compact::root_lower(&cuts, &aggregate, end(), &mut || true).unwrap();
    assert!(cuts.checked_lower < -0.749 && floor > 0.249 && floor <= 0.25);
}

#[test]
fn group_support_odd7_neutral_proper_subsets_do_not_hide_complete_cycle_gain() {
    for permuted in [false, true] {
        let p = cycle(7, 6.5, permuted);
        let cuts = manual(&p);
        for mask in 1usize..127 {
            let heads: Vec<_> = (0..7).filter(|j| mask & (1 << j) != 0).collect();
            if heads.len() < 2 {
                continue;
            }
            let row = support(&cuts, &heads, &budget(), &mut || true).unwrap();
            let exact = exact_support(&p, &heads, row.g());
            // Every proper subset is a forest and has exact support |G|*a.
            assert_eq!(exact, q(heads.len() as f64) * q(-p.c[0]));
            assert!(q(row.upper()) >= exact);
        }
        let row = support(&cuts, &(0..7).collect::<Vec<_>>(), &budget(), &mut || true).unwrap();
        assert_eq!(row.mask_count(), 128);
        assert!(row.upper() >= 6. && row.upper() < 6.001);
    }
}

#[test]
fn group_support_odd9_arity_limit_is_not_mistaken_for_a_complete_method() {
    let p = cycle(9, 8.5, false);
    let cuts = manual(&p);
    for omitted in 0..9 {
        let heads: Vec<_> = (0..9).filter(|j| *j != omitted).collect();
        let row = support(&cuts, &heads, &budget(), &mut || true).unwrap();
        assert!(row.upper() >= 8. && row.upper() < 8.001);
    }
    assert!(support(&cuts, &(0..9).collect::<Vec<_>>(), &budget(), &mut || true).is_none());
}

#[test]
fn group_support_literal_rational_all_masks_include_bias_and_last_input() {
    let p = HullMarginProblem::new(
        vec![1., -0.5, 0.75, -0.25, 1.5, -1., 0.5, 0.25, 1.25],
        vec![0.25, -0.5, 0.125],
        3,
        &[-0.5, -1.25, -0.75],
        &[4.],
        1,
        &[(0, 1.)],
        0.,
        vec![-1.5, 0.25, 0.5],
        vec![0.75, 1.25, 2.],
    )
    .unwrap();
    let mut m = HullMultipliers::zeros(&p);
    m.g = vec![0.125, -0.25, 0.375, -0.5, 0.75, 0.125, 0.25, -0.125, 0.5];
    let lower = certify_hull_lower_bound(&p, &m).unwrap();
    let cuts = prepare(&p, m, lower, end(), &mut || true).unwrap();
    // The kernel must derive support from this sealed finite point vector,
    // rather than assuming that an individual g is always .5W.
    let row = support(&cuts, &[0, 1, 2], &budget(), &mut || true).unwrap();
    let exact = exact_support(&p, row.heads(), row.g());
    assert!(q(row.upper()) >= exact);
    assert!(q(row.upper()) - exact < q(1e-10));
    assert_ne!(row.g()[2], 0.);
}

#[test]
fn group_support_empty_branch_can_have_a_negative_upper_on_last_coordinate() {
    let d = 80;
    let mut w = vec![0.; 3 * d];
    for j in 0..3 {
        w[j * d + d - 1] = 1.;
    }
    let mut lower = vec![0.; d];
    let mut upper = vec![0.; d];
    lower[d - 1] = 1.;
    upper[d - 1] = 2.;
    let p = HullMarginProblem::new(
        w,
        vec![-4.; 3],
        d,
        &[-1.; 3],
        &[1.],
        1,
        &[(0, 1.)],
        0.,
        lower,
        upper,
    )
    .unwrap();
    let cuts = manual(&p);
    let row = support(&cuts, &[0, 1, 2], &budget(), &mut || true).unwrap();
    let exact = exact_support(&p, row.heads(), row.g());
    assert!(exact < q(-1.49));
    assert!(row.upper() < -1.49);
    assert!(q(row.upper()) >= exact);
    assert_eq!(row.mask_count(), 8);
    assert!(row.g()[..d - 1].iter().all(|v| *v == 0.));
    assert!(row.g()[d - 1] > 1.49);
}

#[test]
fn group_support_rejects_duplicate_foreign_positive_and_nonfinite_proposals() {
    let p = mixed(2.);
    let mut cuts = manual(&p);
    for heads in [vec![], vec![0, 0], vec![0, 7], vec![0, 5]] {
        assert!(
            support(&cuts, &heads, &budget(), &mut || true).is_none(),
            "{heads:?}"
        );
    }
    for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        cuts.multipliers.g[0] = v;
        cuts.planes[0].g[0] = v;
        assert!(support(&cuts, &[0, 1, 2], &budget(), &mut || true).is_none());
    }
}
