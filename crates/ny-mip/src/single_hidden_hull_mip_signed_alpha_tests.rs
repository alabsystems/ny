// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Exact small proposal oracles plus real full-mask/certificate/reserve controls.
use super::super::super as signed;
use super::*;
use num_rational::BigRational as Q;
use signed::tests::{cuts, end, graph, rat};

fn box_problem(lower: Vec<f64>, upper: Vec<f64>) -> HullMarginProblem {
    let d = lower.len();
    HullMarginProblem::new(
        vec![0.; d],
        vec![0.],
        d,
        &[1.],
        &[0.],
        1,
        &[(0, 1.)],
        0.,
        lower,
        upper,
    )
    .unwrap()
}
fn candidate(
    p: &HullMarginProblem,
    r: &[f64],
    v: &[f64],
    bias: f64,
    previous: f64,
) -> (f64, usize) {
    let q: Vec<_> = r.iter().zip(v).map(|(r, v)| r + previous * v).collect();
    let mut proposals = 0;
    let a = choose(
        p,
        r,
        v,
        0.1875,
        bias,
        previous,
        &q,
        0.1875 + previous * bias,
        &mut proposals,
        end(),
        &mut || true,
    )
    .unwrap();
    (a, proposals)
}
fn exact(p: &HullMarginProblem, r: &[f64], v: &[f64], bias: f64, a: f64) -> Q {
    rat(0.1875)
        + rat(a) * rat(bias)
        + (0..p.input_dim)
            .map(|i| {
                let q = rat(r[i]) + rat(a) * rat(v[i]);
                (q.clone() * rat(p.lower[i])).min(q * rat(p.upper[i]))
            })
            .sum::<Q>()
}
fn cover_fixture(bias: f64) -> (HullMarginProblem, Component) {
    let mut w = vec![0.; 24 * 30];
    for j in 0..24 {
        w[j * 30 + j % 8] = 1.;
    }
    let c: Vec<_> = (0..24).map(|j| if j < 16 { 0.5 } else { -1. }).collect();
    let p = HullMarginProblem::new(
        w,
        vec![0.; 24],
        30,
        &c,
        &[bias],
        1,
        &[(0, 1.)],
        0.,
        vec![-1.; 30],
        vec![1.; 30],
    )
    .unwrap();
    (
        p,
        Component {
            negative: (16..24).collect(),
            positive: (0..16).collect(),
            positive_units: vec![2; 16],
            g: vec![0.; 30],
            alpha: vec![0.5; 256 * 16],
        },
    )
}

#[test]
fn signed_alpha_sorted_unique_kink_prices_last_input_and_five_score_cap() {
    let p = box_problem(vec![-1.; 30], vec![1.; 30]);
    let mut r = vec![0.; 30];
    let mut v = vec![0.; 30];
    r[29] = -0.375;
    v[29] = 1.;
    let (a, n) = candidate(&p, &r, &v, 0., 0.);
    assert_eq!(a, 0.375);
    assert!(n > 0 && n <= 4); // plus one existing-cache reference score
    assert!(exact(&p, &r, &v, 0., a) > exact(&p, &r, &v, 0., 0.));
    assert_eq!(work(30).unwrap(), 4632);
    assert_eq!(coordinate_work(30).unwrap(), 4632);
    assert_eq!(scratch_bytes(30).unwrap(), 992);
    assert_eq!(coordinate_work(2).unwrap(), 144);
    assert_eq!(scratch_bytes(2).unwrap(), 0);
}

#[test]
fn signed_alpha_sorted_plateau_duplicate_events_and_signed_zero_keep_exact_bits() {
    let p = box_problem(vec![-1.; 30], vec![1.; 30]);
    let mut r = vec![0.; 30];
    let mut v = vec![0.; 30];
    r[0] = -0.25;
    r[29] = 0.75;
    v[0] = 1.;
    v[29] = -1.;
    assert_eq!(candidate(&p, &r, &v, 0., 0.5).0.to_bits(), 0.5f64.to_bits());
    r[0] = -0.375;
    r[29] = 0.75;
    v[29] = -2.;
    assert_eq!(candidate(&p, &r, &v, 0., 0.125).0, 0.375);
    r[0] = -0.5;
    r[29] = -0.5000000000000001;
    v[29] = 1.;
    let close = candidate(&p, &r, &v, 0., 0.).0;
    assert!((0.5..=0.5000000000000001).contains(&close));
    let large = 1_099_511_627_776.;
    r[0] = -large * 0.5;
    r[29] = large * 0.5;
    v[0] = large;
    v[29] = -large;
    let bias = 2. * large - 1.;
    let a = candidate(&p, &r, &v, bias, 0.25).0;
    assert_eq!(exact(&p, &r, &v, bias, a), exact(&p, &r, &v, bias, 0.5));
    r.fill(0.);
    v.fill(0.);
    assert_eq!(
        candidate(&p, &r, &v, 0., -0.).0.to_bits(),
        (-0f64).to_bits()
    );
    let fixed = box_problem(vec![0.25; 30], vec![0.25; 30]);
    v[29] = 1.;
    assert_eq!(candidate(&fixed, &r, &v, 0., 0.).0, 1.);
}

#[test]
fn signed_alpha_sorted_exhaustive_dyadic_oracle_matches_nominal_concave_maximum() {
    let p = box_problem(vec![-1.5, 0.25], vec![0.75, 1.25]);
    for r0 in -2..=2 {
        for r1 in -2..=2 {
            for v0 in -2..=2 {
                for v1 in -2..=2 {
                    for bias in -1..=1 {
                        let r = [f64::from(r0) / 4., f64::from(r1) / 4.];
                        let v = [f64::from(v0), f64::from(v1)];
                        let bias = f64::from(bias) / 4.;
                        let (a, n) = candidate(&p, &r, &v, bias, 0.375);
                        let mut points = vec![0., 1.];
                        for i in 0..2 {
                            if v[i] != 0. {
                                let point = -r[i] / v[i];
                                if (0. ..=1.).contains(&point) {
                                    points.push(point);
                                }
                            }
                        }
                        let optimum = points
                            .into_iter()
                            .map(|x| exact(&p, &r, &v, bias, x))
                            .max()
                            .unwrap();
                        assert_eq!(exact(&p, &r, &v, bias, a), optimum);
                        assert!(n <= 4);
                    }
                }
            }
        }
    }
}

#[test]
fn signed_alpha_sorted_degenerate_shapes_overflow_and_nonfinite_advice_refuse() {
    let p = box_problem(vec![-1.; 30], vec![1.; 30]);
    let mut r = vec![0.; 30];
    let mut v = vec![0.; 30];
    let q = vec![0.; 30];
    let mut n = 0;
    assert!(choose(
        &p,
        &r[..29],
        &v,
        0.,
        0.,
        0.,
        &q,
        0.,
        &mut n,
        end(),
        &mut || true
    )
    .is_none());
    v[29] = f64::NAN;
    assert!(choose(&p, &r, &v, 0., 0., 0., &q, 0., &mut n, end(), &mut || true).is_none());
    r[29] = f64::MAX;
    v[29] = f64::MIN_POSITIVE;
    assert!(choose(&p, &r, &v, 0., 0., 0., &q, 0., &mut n, end(), &mut || true).is_none());
    assert!(work(usize::MAX).is_none());
    assert!(coordinate_work(usize::MAX).is_none());
    assert!(scratch_bytes(usize::MAX).is_none());
}

#[test]
fn signed_alpha_sorted_poll_after_direct_score_retains_completed_proposal_count() {
    let p = box_problem(vec![-1.; 30], vec![1.; 30]);
    let mut r = vec![0.; 30];
    let mut v = vec![0.; 30];
    r[29] = -0.375;
    v[29] = 1.;
    let q: Vec<_> = r.iter().zip(&v).map(|(r, v)| r + 0.5 * v).collect();
    let mut polls = 0;
    let mut n = 0;
    assert!(
        choose(&p, &r, &v, 0., 0., 0.5, &q, 0., &mut n, end(), &mut || {
            polls += 1;
            polls <= 31
        })
        .is_none()
    );
    assert_eq!(n, 1);
}

#[test]
fn signed_alpha_sorted_full16_d30_cover_keeps_next_full_check_and_two_publication_replays() {
    let (p, component) = cover_fixture(0.25);
    let c = cuts(&p);
    let plan = Plan {
        components: vec![component.clone()],
        leftover: vec![],
    };
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(&p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let old = certificate::check(&c, &plan, &b, b.search_end, &mut || true).unwrap();
    let all = 256usize * 2 * 16 * coordinate_work(30).unwrap();
    assert_eq!(all, 37_945_344);
    assert!(all + cost + 256 * 25 * 31 * 4 < b.remaining().unwrap());
    let refined = optimize(&c, &component, vec![0.; 30], &b, cost, &mut || true).unwrap();
    for j in 0..16 {
        assert_eq!(refined.alpha[j], 0.);
        assert_eq!(refined.alpha[255 * 16 + j], 1.);
    }
    assert_eq!(b.state.lock().unwrap().completed_masks, 256);
    let plan = Plan {
        components: vec![refined],
        leftover: vec![],
    };
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&c, &plan, &b, b.search_end, &mut || true).unwrap();
    assert!(checked.root_lower > old.root_lower + 0.1);
    assert!(checked.root_lower > 0.249 && checked.root_lower <= 0.25);
    assert_eq!(b.state.lock().unwrap().reserved, 2 * cost);
    let seal = Arc::new(signature(&c).collect());
    let completed = row(&c, &plan, checked, cost, &b, &seal);
    for _ in 0..2 {
        assert!(completed
            .replay_inner(&c, end(), &mut || true, false)
            .is_some());
    }
    assert_eq!(completed.stats().source_replays, 2);
    assert!(completed.stats().work_units <= MAX_WORK);
}

#[test]
fn signed_alpha_sorted_unsafe_tie_and_biased_nonsymmetric_roles_stay_bounded() {
    for bias in [0.25, 0., -0.25] {
        let mut w = vec![0.; 4 * 30];
        for j in 0..4 {
            w[j * 30 + 28 + j % 2] = 1.;
        }
        let mut l = vec![-1.; 30];
        let mut u = vec![1.; 30];
        l[28] = -1.5;
        u[28] = 0.75;
        l[29] = 0.25;
        u[29] = 1.25;
        let p = HullMarginProblem::new(
            w,
            vec![0.25, -0.125, 0.25, -0.125],
            30,
            &[1., 1., -1., -1.],
            &[bias],
            1,
            &[(0, 1.)],
            0.,
            l,
            u,
        )
        .unwrap();
        let c = cuts(&p);
        let component = Component {
            negative: vec![2, 3],
            positive: vec![0, 1],
            positive_units: vec![2; 2],
            g: vec![0.; 30],
            alpha: vec![0.5; 8],
        };
        let plan = Plan {
            components: vec![component.clone()],
            leftover: vec![],
        };
        let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
        let cost = plan.cost(&p).unwrap();
        b.reserve(cost).unwrap();
        let component = optimize(&c, &component, vec![0.; 30], &b, cost, &mut || true).unwrap();
        b.proof(cost, false).unwrap();
        let checked = certificate::check(
            &c,
            &Plan {
                components: vec![component],
                leftover: vec![],
            },
            &b,
            b.search_end,
            &mut || true,
        )
        .unwrap();
        let mut point = vec![0.; 30];
        point[29] = 0.5;
        assert!(rat(checked.root_lower) <= graph(&p, &point));
        assert!(checked.root_lower <= bias);
        if bias > 0. {
            assert!(checked.root_lower > 0.249);
        } else {
            assert!(checked.root_lower <= 0.);
        }
        assert_eq!(b.state.lock().unwrap().completed_masks, 4);
    }
}

#[test]
fn signed_alpha_sorted_partial_cover_preserves_baseline_and_two_reserved_replays() {
    let (p, component) = cover_fixture(0.25);
    let c = cuts(&p);
    let plan = Plan {
        components: vec![component.clone()],
        leftover: vec![],
    };
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(&p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&c, &plan, &b, b.search_end, &mut || true).unwrap();
    let before = checked.root_lower.to_bits();
    let seal = Arc::new(signature(&c).collect());
    let baseline = row(&c, &plan, checked, cost, &b, &seal);
    assert!(optimize(&c, &component, vec![0.; 30], &b, cost, &mut || {
        b.state.lock().unwrap().completed_masks < 255
    })
    .is_none());
    assert_eq!(b.state.lock().unwrap().completed_masks, 255);
    assert_eq!(baseline.lower.to_bits(), before);
    b.charge(b.remaining().unwrap()).unwrap();
    for _ in 0..2 {
        assert_eq!(
            baseline
                .replay_inner(&c, end(), &mut || true, false)
                .unwrap()
                .root_lower
                .to_bits(),
            before
        );
    }
    assert_eq!(baseline.stats().source_replays, 2);
    assert!(baseline.stats().work_units <= MAX_WORK);
}
