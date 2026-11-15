// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Authored original-graph controls; no whole-verifier or timing separation.
use super::super::tests::{cuts, end, graph, rat, twins};
use super::*;
use num_rational::BigRational as Q;
use num_traits::Zero;

pub(super) fn problem(
    w: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: f64,
    lower: Vec<f64>,
    upper: Vec<f64>,
) -> HullMarginProblem {
    let dim = lower.len();
    let mut p =
        HullMarginProblem::new(w, b, dim, &c, &[d], 1, &[(0, 1.)], 0., lower, upper).unwrap();
    // Independently authored exact dyadic coefficient contract for these
    // kernel controls; actual CLI controls use the ordinary constructor path.
    p.c = c;
    p.d = d;
    p
}
pub(super) fn one(d: f64) -> HullMarginProblem {
    problem(vec![1.], vec![0.], vec![-1.], d, vec![-1.], vec![1.])
}
pub(super) fn plan(p: &HullMarginProblem) -> Plan {
    Plan {
        components: vec![Component {
            negative: p
                .c
                .iter()
                .enumerate()
                .filter_map(|(j, c)| (*c < 0.).then_some(j))
                .collect(),
            positive: vec![],
            positive_units: vec![],
            g: vec![0.; p.input_dim],
            alpha: vec![],
        }],
        leftover: p
            .c
            .iter()
            .enumerate()
            .filter_map(|(j, c)| (*c > 0.).then_some(j))
            .collect(),
    }
}
pub(super) fn checked_row(c: &HullRootMipCuts<'_>, plan: &Plan) -> SignedMajorant {
    let b = Budget::new(c, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(c.problem).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(c, plan, &b, b.search_end, &mut || true).unwrap();
    let signature = Arc::new(signature(c).collect());
    row(c, plan, checked, cost, &b, &signature)
}
pub(super) fn cached(c: &HullRootMipCuts<'_>, row: &SignedMajorant) -> Arc<Cache> {
    assert!(row
        .replay_inner(c, end(), &mut || true, false)
        .unwrap()
        .cache
        .is_none());
    let v = row.replay_with_cache(c, end(), &mut || true).unwrap();
    assert_eq!(v.root_lower.to_bits(), row.lower.to_bits());
    assert_eq!(row.stats().source_replays, 2);
    v.cache
        .expect("only a completed second publication pass mints the cache")
}
pub(super) fn split(j: usize, active: bool) -> HullSplit {
    HullSplit {
        neuron: j as u32,
        active,
    }
}
pub(super) fn cycle(d: f64, extra_t: bool) -> HullMarginProblem {
    let dim = if extra_t { 4 } else { 3 };
    let h = if extra_t { 4 } else { 3 };
    let mut w = vec![0.; dim * h];
    for j in 0..3 {
        w[j * dim + j] = 1.;
        w[j * dim + (j + 1) % 3] = -1.;
    }
    if extra_t {
        w[3 * dim + 3] = 1.;
    }
    problem(
        w,
        vec![0.; h],
        vec![-1.; h],
        d,
        vec![-1.; dim],
        vec![1.; dim],
    )
}

#[test]
fn signed_child_cache_requires_actual_second_complete_replay_and_keeps_root_bits() {
    let p = one(0.25);
    let c = cuts(&p);
    let row = checked_row(&c, &plan(&p));
    // Collection on the first pass is deliberately refused, not backfilled.
    let first = row.replay_with_cache(&c, end(), &mut || true).unwrap();
    assert!(first.cache.is_none());
    assert_eq!(row.stats().source_replays, 1);
    let second = row.replay_with_cache(&c, end(), &mut || true).unwrap();
    let cache = second.cache.unwrap();
    assert_eq!(cache.tables.len(), 1);
    assert_eq!(cache.tables[0].values.len(), 2);
    assert_eq!(cache.root_lower.to_bits(), row.lower.to_bits());
    let bound = cache.bind(&c, end(), &mut || true).unwrap();
    let root = bound.lower(&[], &[], end(), &mut || true).unwrap();
    assert!(root <= 0.25 - 1. && root > -0.75000001);
    assert!(row.replay_with_cache(&c, end(), &mut || true).is_none());
    assert_eq!(row.stats().source_replays, 2);
    let stats = cache.stats();
    assert_eq!(stats.collection_attempts, 1);
    assert_eq!(stats.cached_masks, 2);
    assert_eq!(stats.attempted_calls, 1);
    assert_eq!(stats.completed_calls, 1);
    assert_eq!(c.proposal_lp_calls, 0);
}

#[test]
fn signed_child_compatible_constant_strictly_closes_inactive_and_keeps_unsafe_active_open() {
    let p = one(0.25);
    let c = cuts(&p);
    let row = checked_row(&c, &plan(&p));
    let cache = cached(&c, &row);
    let bound = cache.bind(&c, end(), &mut || true).unwrap();
    let root = bound.lower(&[], &[], end(), &mut || true).unwrap();
    let inactive = bound
        .lower(&[split(0, false)], &[0.], end(), &mut || true)
        .unwrap();
    let active = bound
        .lower(&[split(0, true)], &[0.], end(), &mut || true)
        .unwrap();
    assert!(root < -0.749 && inactive > 0.249 && inactive <= 0.25 && active <= -0.75);
    // A fixed unconditional G=0/K=-1 cannot improve with inactive beta>=0.
    for beta in [0., 0.5, 2.] {
        assert!(0.25 - 1. - beta <= root + 1e-12);
    }
    assert_eq!(graph(&p, &[1.]), rat(-0.75));
    assert!(rat(active) <= graph(&p, &[1.]));
    // Both closed phases contain the exact zero face; neither is infeasible.
    assert_eq!(graph(&p, &[0.]), rat(0.25));
}

#[test]
fn signed_child_all_literal_cycle_masks_include_empty_full_and_closed_zero_faces() {
    let p = cycle(0.25, false);
    let c = cuts(&p);
    let row = checked_row(&c, &plan(&p));
    let cache = cached(&c, &row);
    let table = &cache.tables[0];
    assert_eq!(table.values.len(), 8);
    for mask in 0..8 {
        let mut coeff = vec![Q::zero(); 3];
        for j in 0..3 {
            if mask & (1 << j) != 0 {
                for i in 0..3 {
                    coeff[i] += rat(p.c[j]) * rat(p.w_row(j)[i]);
                }
            }
        }
        let literal = coeff
            .iter()
            .map(|q| (q * rat(-1.)).min(q * rat(1.)))
            .sum::<Q>();
        assert!(rat(table.values[mask]) <= literal);
        assert!(rat(table.values[mask]) > literal.clone() - rat(1e-10));
        assert_eq!(literal, rat(if mask == 0 || mask == 7 { 0. } else { -2. }));
    }
    let bound = cache.bind(&c, end(), &mut || true).unwrap();
    for active in [false, true] {
        let phases: Vec<_> = (0..3).map(|j| split(j, active)).collect();
        let value = bound.lower(&phases, &[0.; 3], end(), &mut || true).unwrap();
        assert!(value > 0.249 && value <= 0.25);
        assert_eq!(graph(&p, &[0.; 3]), rat(0.25));
    }
    let mixed = bound
        .lower(
            &[split(0, true), split(1, false), split(2, false)],
            &[0.; 3],
            end(),
            &mut || true,
        )
        .unwrap();
    assert!(rat(mixed) <= graph(&p, &[1., -1., 0.]));
    assert_eq!(graph(&p, &[1., -1., 0.]), rat(-1.75));
}

#[test]
fn signed_child_full_q_reprices_bias_beta_asymmetric_box_last_input_and_leftover_once() {
    let mut w = vec![0.; 60];
    w[0] = 1.;
    w[59] = 1.;
    let mut l = vec![0.; 30];
    let mut u = l.clone();
    l[0] = -0.5;
    u[0] = 0.75;
    l[29] = -1.5;
    u[29] = 0.75;
    let p = problem(w, vec![2., 0.25], vec![0.5, -1.], 0.125, l, u);
    let c = cuts(&p);
    let row = checked_row(&c, &plan(&p));
    let cache = cached(&c, &row);
    let bound = cache.bind(&c, end(), &mut || true).unwrap();
    for (active, beta, expected) in [
        (false, 0., 0.5),
        (false, 0.5, -0.125),
        (true, 0., -0.5),
        (true, 0.5, -1.),
    ] {
        let value = bound
            .lower(&[split(1, active)], &[beta], end(), &mut || true)
            .unwrap();
        assert!(value <= expected && value > expected - 1e-10);
        for a in [-0.5, 0.75] {
            for z in [-1.5, -0.25, 0.75] {
                if (active && z + 0.25 < 0.) || (!active && z + 0.25 > 0.) {
                    continue;
                }
                let mut x = vec![0.; 30];
                x[0] = a;
                x[29] = z;
                assert!(rat(value) <= graph(&p, &x));
            }
        }
    }
    assert!(cache.g[0] > 0. && cache.g[29] == 0.);
}

#[test]
fn signed_child_intercept_is_not_a_whole_floor_on_positive_minimum_box() {
    let p = problem(
        vec![1., 0., 0., 1.],
        vec![2., 0.25],
        vec![0.5, -1.],
        0.125,
        vec![0.25, -1.5],
        vec![1.25, 0.75],
    );
    let mut m = HullMultipliers::zeros(&p);
    m.alpha[0] = 1.;
    m.g[2] = 0.;
    m.g[3] = 0.5;
    let floor = certify_hull_lower_bound(&p, &m).unwrap();
    let c = prepare(&p, m, floor, end(), &mut || true).unwrap();
    let row = checked_row(&c, &plan(&p));
    let cache = cached(&c, &row);
    let bound = cache.bind(&c, end(), &mut || true).unwrap();
    let value = bound
        .lower(&[split(1, false)], &[0.], end(), &mut || true)
        .unwrap();
    assert!(value > 1.24999999 && value <= 1.25);
    assert_eq!(graph(&p, &[0.25, -0.25]), rat(1.25));
    assert!(rat(value) <= graph(&p, &[0.25, -0.25]));
    assert!(cache.root_lower > cache.constant);
}

#[test]
fn signed_child_original_head_maps_filter_disjoint_components_and_ignore_positive_phase_bits() {
    let p = problem(
        vec![1., 0., 1., 0., 0., 1., 0., 1.],
        vec![0.; 4],
        vec![1., -1., 1., -1.],
        0.25,
        vec![-1.; 2],
        vec![1.; 2],
    );
    let c = cuts(&p);
    let v = Plan {
        components: vec![
            Component {
                negative: vec![1],
                positive: vec![0],
                positive_units: vec![2],
                g: vec![0.; 2],
                alpha: vec![0.; 2],
            },
            Component {
                negative: vec![3],
                positive: vec![2],
                positive_units: vec![2],
                g: vec![0.; 2],
                alpha: vec![0.; 2],
            },
        ],
        leftover: vec![],
    };
    let row = checked_row(&c, &v);
    let cache = cached(&c, &row);
    let bound = cache.bind(&c, end(), &mut || true).unwrap();
    let root = bound.lower(&[], &[], end(), &mut || true).unwrap();
    let one = bound
        .lower(&[split(1, false)], &[0.], end(), &mut || true)
        .unwrap();
    let both = bound
        .lower(
            &[split(3, false), split(1, false)],
            &[0.; 2],
            end(),
            &mut || true,
        )
        .unwrap();
    assert!(root < -1.749 && one > -0.75000001 && one <= -0.75 && both > 0.249);
    let extra = bound
        .lower(
            &[split(3, false), split(0, true), split(1, false)],
            &[0.; 3],
            end(),
            &mut || true,
        )
        .unwrap();
    assert_eq!(extra.to_bits(), both.to_bits());
    assert_eq!(graph(&p, &[0., 0.]), rat(0.25));
}

#[test]
fn signed_child_malformed_splits_lengths_and_nonfinite_beta_cannot_create_a_bound() {
    let p = one(0.25);
    let c = cuts(&p);
    let row = checked_row(&c, &plan(&p));
    let cache = cached(&c, &row);
    let bound = cache.bind(&c, end(), &mut || true).unwrap();
    for list in [
        vec![split(0, true), split(0, false)],
        vec![split(0, false), split(0, false)],
        vec![split(1, true)],
    ] {
        assert!(bound
            .lower(&list, &vec![0.; list.len()], end(), &mut || true)
            .is_none());
    }
    assert!(bound
        .lower(&[split(0, false)], &[], end(), &mut || true)
        .is_none());
    for beta in [f64::INFINITY, f64::NEG_INFINITY] {
        assert!(bound
            .lower(&[split(0, false)], &[beta], end(), &mut || true)
            .is_none());
    }
    let zero = bound
        .lower(&[split(0, false)], &[0.], end(), &mut || true)
        .unwrap();
    // Match the existing native proposal sanitation; negative/NaN buys no dual.
    for beta in [f64::NAN, -1.] {
        assert_eq!(
            bound
                .lower(&[split(0, false)], &[beta], end(), &mut || true)
                .unwrap()
                .to_bits(),
            zero.to_bits()
        );
    }
}

#[test]
fn signed_child_real_generic_worker_nonpositive_cache_keeps_original_q_and_tuple() {
    let p = twins(-0.25, true);
    let c = cuts(&p);
    let before = c.multipliers.clone();
    let (row, r) = propose(&c, end(), Duration::ZERO);
    let row = row.unwrap();
    assert!(r.admitted && row.lower <= -0.25);
    let cache = cached(&c, &row);
    let bound = cache.bind(&c, end(), &mut || true).unwrap();
    let negative: Vec<_> =
        p.c.iter()
            .enumerate()
            .filter_map(|(j, c)| (*c < 0.).then_some(j))
            .collect();
    for mask in 0..1usize << negative.len() {
        let list: Vec<_> = negative
            .iter()
            .enumerate()
            .map(|(k, &j)| split(j, mask & (1 << k) != 0))
            .collect();
        let value = bound
            .lower(&list, &vec![0.; list.len()], end(), &mut || true)
            .unwrap();
        for x in [
            [p.lower[0], p.lower[1]],
            [p.lower[0], p.upper[1]],
            [p.upper[0], p.lower[1]],
            [p.upper[0], p.upper[1]],
        ] {
            let compatible = list.iter().all(|s| {
                let j = s.neuron as usize;
                let z = rat(p.b1[j]) + (0..2).map(|i| rat(p.w_row(j)[i]) * rat(x[i])).sum::<Q>();
                if s.active {
                    z >= Q::zero()
                } else {
                    z <= Q::zero()
                }
            });
            if compatible {
                assert!(rat(value) <= graph(&p, &x));
            }
        }
    }
    assert_eq!(before, c.multipliers);
    assert_eq!(c.proposal_lp_calls, 0);
    assert_eq!(row.stats().source_replays, 2);
}

#[test]
fn signed_child_real_pre_lp_worker_skips_cold_lp_where_scalar_cycle_geometry_stays_open() {
    let p = cycle(2.25, true);
    let c = cuts(&p);
    let row = checked_row(&c, &plan(&p));
    let cache = cached(&c, &row);
    let bound = cache.bind(&c, end(), &mut || true).unwrap();
    let phases = [split(3, false)];
    let (native, cached, status, calls) = bab::signed_child_test_node(&p, &phases, &bound, end());
    assert!(native.is_some_and(|v| v <= -0.749));
    assert!(cached.is_some_and(|v| v > 0.249 && v <= 0.25));
    assert_eq!(status, HullLpStatus::Skipped);
    assert_eq!(calls, 0);
    assert_eq!(cache.stats().lp_skips, 1);
    assert!(
        bound
            .lower(&[split(3, true)], &[0.], end(), &mut || true)
            .unwrap()
            <= -0.75
    );
    assert_eq!(graph(&p, &[1., -1., 0., 1.]), rat(-0.75));
    assert_eq!(row.stats().source_replays, 2);
    assert_eq!(c.proposal_lp_calls, 0);
}
