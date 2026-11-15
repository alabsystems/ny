// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Independent exact authored graph controls; no benchmark claim.
use super::*;
use num_rational::BigRational as Q;
use num_traits::{Signed, Zero};

// Keep the no-cache test adapter with test support; numeric replay is always compiled.
impl SignedMajorant {
    pub(super) fn replay(
        &self,
        cuts: &HullRootMipCuts<'_>,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Replay> {
        self.replay_inner(cuts, end, poll, false)
    }
}
pub(super) fn end() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
pub(super) fn rat(v: f64) -> Q {
    Q::from_float(v).unwrap()
}
pub(super) fn twins(bias: f64, permuted: bool) -> HullMarginProblem {
    let w = [[1., 0.], [0., 1.], [1., 0.], [0., 1.]];
    let c = [1., 1., -1., -1.];
    let b = [0.25, -0.125, 0.25, -0.125];
    let mut weights = vec![0.; 8];
    let mut constants = vec![0.; 4];
    let mut objective = vec![0.; 4];
    for j in 0..4 {
        let h = if permuted { (3 * j + 1) % 4 } else { j };
        constants[h] = b[j];
        objective[h] = c[j];
        for i in 0..2 {
            let x = if permuted { 1 - i } else { i };
            weights[h * 2 + x] = w[j][i];
        }
    }
    let (l, u) = if permuted {
        (vec![0.25, -1.5], vec![1.25, 0.75])
    } else {
        (vec![-1.5, 0.25], vec![0.75, 1.25])
    };
    HullMarginProblem::new(
        weights,
        constants,
        2,
        &objective,
        &[bias],
        1,
        &[(0, 1.)],
        0.,
        l,
        u,
    )
    .unwrap()
}
pub(super) fn cuts(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    let mut m = HullMultipliers::zeros(p);
    for j in 0..p.hidden_dim {
        if p.c[j] > 0. {
            m.alpha[j] = 0.5;
        } else {
            for i in 0..p.input_dim {
                m.g[j * p.input_dim + i] = 0.5 * p.w_row(j)[i];
            }
        }
    }
    let floor = certify_hull_lower_bound(p, &m).unwrap();
    prepare(p, m, floor, end(), &mut || true).unwrap()
}
pub(super) fn graph(p: &HullMarginProblem, x: &[f64]) -> Q {
    rat(p.d)
        + (0..p.hidden_dim)
            .map(|j| {
                let z = rat(p.b1[j])
                    + (0..p.input_dim)
                        .map(|i| rat(p.w_row(j)[i]) * rat(x[i]))
                        .sum::<Q>();
                rat(p.c[j]) * z.max(Q::zero())
            })
            .sum::<Q>()
}
pub(super) fn exact_plan() -> Plan {
    Plan {
        components: vec![Component {
            negative: vec![2, 3],
            positive: vec![0, 1],
            positive_units: vec![2; 2],
            g: vec![0., 0.],
            alpha: vec![0., 0., 1., 0., 0., 1., 1., 1.],
        }],
        leftover: vec![],
    }
}
pub(super) fn checked(c: &HullRootMipCuts<'_>, plan: &Plan) -> certificate::Checked {
    let b = Budget::new(c, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(c.problem).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    certificate::check(c, plan, &b, b.search_end, &mut || true).unwrap()
}
#[test]
fn signed_root_generic_coordinate_search_closes_biased_independent_roles_and_permutations() {
    for permuted in [false, true] {
        let p = twins(0.25, permuted);
        let c = cuts(&p);
        let (row, stats) = propose(&c, end(), Duration::ZERO);
        let row = row.expect("generic affinity and finite coordinate search must close");
        assert!(row.lower > 0.249 && row.lower <= 0.25);
        assert!(
            stats.coordinate_proposals > 0 && stats.completed_masks > 0 && stats.checked_masks > 0
        );
        assert!(
            stats.refined_components > 0
                && stats.work_units <= MAX_WORK
                && stats.admitted_bytes <= MAX_BYTES
        );
        assert_eq!(stats.source_replays, 0);
        assert_eq!(c.proposal_lp_calls, 0);
        assert_eq!(
            row.replay_inner(&c, end(), &mut || true, false)
                .unwrap()
                .root_lower
                .to_bits(),
            row.lower.to_bits()
        );
    }
}
#[test]
fn signed_root_original_unsafe_and_tie_points_cannot_become_strictly_positive() {
    for bias in [-0.25, 0.] {
        let p = twins(bias, false);
        let c = cuts(&p);
        let (row, report) = propose(&c, end(), Duration::ZERO);
        let row = row.expect("a tighter nonpositive geometry row remains useful");
        // The authored original twins cancel exactly; stored c/d are
        // independently rounded down, so their graph can be slightly lower.
        let actual = graph(&p, &[0., 0.5]);
        assert!(actual <= rat(bias));
        assert!(rat(row.lower) <= actual && row.lower <= 0.);
        assert!(report.admitted && report.coordinate_proposals > 0);
    }
}
#[test]
fn signed_root_independent_mask_slopes_strictly_separate_common_theta_on_one_component() {
    let mut p = twins(0.25, false);
    p.b1.fill(0.);
    p.lower = vec![-1.; 2];
    p.upper = vec![1.; 2];
    let c = cuts(&p);
    let independent = checked(&c, &exact_plan());
    assert!(independent.root_lower > 0.249);
    for theta in [0., 0.125, 0.5, 0.875, 1.] {
        let mut plan = exact_plan();
        plan.components[0].alpha.fill(theta);
        let common = checked(&c, &plan);
        // On either singleton mask |theta-1|+|theta|=1 exactly.
        assert_eq!((rat(theta) - rat(1.)).abs() + rat(theta).abs(), rat(1.));
        assert!(common.root_lower <= -0.749);
    }
}
#[test]
fn signed_root_full_q_prices_nonzero_bias_asymmetric_box_last_input_and_d_once() {
    let p = twins(0.25, false);
    let c = cuts(&p);
    let mut plan = exact_plan();
    plan.components[0].g = vec![0., 0.25];
    let row = checked(&c, &plan);
    // Exact signed expression is zero: K=-.25*1.25 and root minG=.25*.25.
    assert!(row.constant > -0.31250001 && row.constant <= -0.3125);
    assert!(row.root_lower > -0.00000001 && row.root_lower <= 0.);
    assert_eq!(row.g, vec![0., 0.25]);
    for x in [[-1.5, 0.25], [-1.5, 1.25], [0.75, 0.25], [0.75, 1.25]] {
        let actual = graph(&p, &x);
        let affine = rat(row.constant) + rat(row.g[1]) * rat(x[1]);
        assert!(affine <= actual - rat(p.d));
        assert!(rat(row.root_lower) <= graph(&p, &x));
    }
}
#[test]
fn signed_root_unassigned_positive_mass_uses_its_original_alpha_and_all_coordinates() {
    let mut p = twins(0.25, false);
    p.w1.extend([0., 1.]);
    p.b1.push(2.);
    p.c.push(0.75);
    p.hidden_dim += 1;
    let c = cuts(&p);
    let mut plan = exact_plan();
    plan.leftover.push(4);
    let row = checked(&c, &plan);
    assert!(row.root_lower > 1.0937499 && row.root_lower <= 1.09375);
    for y in [p.lower[1], p.upper[1]] {
        assert!(rat(row.root_lower) <= graph(&p, &[0., y]));
    }
}
#[test]
fn signed_root_empty_full_masks_and_positive_lower_constant_are_all_completed() {
    let p = twins(0.25, false);
    let c = cuts(&p);
    let plan = exact_plan();
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(&p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let v = certificate::check(&c, &plan, &b, b.search_end, &mut || true).unwrap();
    assert_eq!(b.state.lock().unwrap().checked_masks, 4);
    assert!(v.constant > -1e-12 && v.constant <= 0. && v.root_lower > 0.249);
}
#[test]
fn signed_root_partition_and_alpha_validation_refuse_missing_duplicate_or_overdrawn_roles() {
    let p = twins(0.25, false);
    let c = cuts(&p);
    for case in 0..6 {
        let mut v = exact_plan();
        match case {
            0 => {
                v.components[0].negative.pop();
            }
            1 => {
                v.components[0].negative = vec![2, 2];
            }
            2 => {
                v.leftover.push(0);
            }
            3 => {
                v.components[0].alpha[0] = f64::NAN;
            }
            4 => {
                v.components[0].alpha[0] = 1.0001;
            }
            _ => {
                v.components[0].negative = vec![2; usize::BITS as usize];
            }
        }
        assert!(v.valid(&c).is_none());
        if case == 5 {
            assert!(v.masks().is_none() && v.cost(&p).is_none());
        }
    }
}
#[test]
fn signed_root_optional_search_never_mutates_original_attaining_tuple_scalar_or_rows() {
    let p = twins(0.25, false);
    let c = cuts(&p);
    let tuple = c.multipliers.clone();
    let scalar = c.checked_lower.to_bits();
    let planes = c
        .planes
        .iter()
        .map(|r| (r.head, r.upper.to_bits()))
        .collect::<Vec<_>>();
    let (row, _) = propose(&c, end(), Duration::ZERO);
    assert!(row.is_some());
    assert_eq!(c.multipliers, tuple);
    assert_eq!(c.checked_lower.to_bits(), scalar);
    assert_eq!(
        c.planes
            .iter()
            .map(|r| (r.head, r.upper.to_bits()))
            .collect::<Vec<_>>(),
        planes
    );
    assert_eq!(c.proposal_lp_calls, 0);
}
