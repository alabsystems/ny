// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;

fn duplicate_problem(bias: f64) -> HullMarginProblem {
    HullMarginProblem::new(
        vec![1.0, 1.0],
        vec![0.0, 0.0],
        1,
        &[1.0, -1.0],
        &[bias],
        1,
        &[(0, 1.0)],
        0.0,
        vec![-1.0],
        vec![1.0],
    )
    .unwrap()
}

fn seed() -> WarmCandidate {
    WarmCandidate {
        multipliers: HullMultipliers {
            alpha: vec![0.5, 0.0],
            g: vec![0.0, 0.5],
        },
        beta: vec![],
    }
}

fn live() -> Instant {
    Instant::now() + Duration::from_secs(2)
}

#[test]
fn hull_reuse_positive_alpha_coordinates_close_both_duplicate_phases() {
    // The graph margin is exactly 1/8. The inherited root minorant is negative;
    // alpha must become one/zero on the active/inactive negative-head child.
    let p = duplicate_problem(0.125);
    let bounds = preactivation_bounds(&p);
    let roles = classify(&p, &bounds);
    assert!(certify_hull_node_lower_bound(&p, &[], &seed().multipliers, &[]).unwrap() < 0.0);
    for active in [false, true] {
        let splits = [HullSplit { neuron: 1, active }];
        let (lower, proposed) = warm_child(&p, &roles, &splits, &seed(), live()).unwrap();
        assert!(lower > 0.124 && lower <= 0.125, "{active}: {lower}");
        assert_eq!(
            Some(lower),
            certify_hull_node_lower_bound(&p, &splits, &proposed.multipliers, &proposed.beta)
        );
    }
}

#[test]
fn hull_reuse_warm_positive_child_skips_cold_lp_proposal() {
    let p = duplicate_problem(0.125);
    let bounds = preactivation_bounds(&p);
    let roles = classify(&p, &bounds);
    let out = bound_node_counted(
        &p,
        &roles,
        &bounds,
        &[HullSplit {
            neuron: 1,
            active: false,
        }],
        live(),
        Some(&seed()),
        None,
    );
    assert!(out.certified.is_some_and(|v| v > 0.124 && v <= 0.125));
    assert_eq!(out.route, Some(HullBoundRoute::Subgradient));
    assert_eq!(out.lp_status, HullLpStatus::Skipped);
    assert!(!lp_failed(out.lp_status));
    assert_eq!(out.lp_value, None);
    assert_eq!(out.branch, None);
}

#[test]
fn hull_reuse_nonnegative_halfspace_beta_supplies_missing_child_geometry() {
    // m=x-1/2-(1/8)ReLU(x-3/4). On the selected active child x>=3/4,
    // the exact minimum is 1/4; an unpriced whole-box minorant is negative.
    let p = HullMarginProblem::new(
        vec![1.0, 1.0],
        vec![2.0, -0.75],
        1,
        &[1.0, -0.125],
        &[-2.5],
        1,
        &[(0, 1.0)],
        0.0,
        vec![-1.0],
        vec![1.0],
    )
    .unwrap();
    let bounds = preactivation_bounds(&p);
    let roles = classify(&p, &bounds);
    let hint = WarmCandidate {
        multipliers: triangle_multipliers(&p, &roles, &bounds),
        beta: vec![],
    };
    let splits = [HullSplit {
        neuron: 1,
        active: true,
    }];
    assert!(certify_hull_node_lower_bound(&p, &splits, &hint.multipliers, &[0.0]).unwrap() < 0.0);
    let (lower, proposal) = warm_child(&p, &roles, &splits, &hint, live()).unwrap();
    assert!(proposal.beta[0] > 0.0);
    assert!(lower > 0.249 && lower <= 0.25, "{lower}");
}

#[test]
fn hull_reuse_foreign_hint_cannot_close_unsafe_graph_or_renew_deadline() {
    let p = duplicate_problem(-0.125);
    let bounds = preactivation_bounds(&p);
    let roles = classify(&p, &bounds);
    for active in [false, true] {
        let splits = [HullSplit { neuron: 1, active }];
        let (lower, _) = warm_child(&p, &roles, &splits, &seed(), live()).unwrap();
        assert!(lower <= -0.125, "{lower}");
        assert!(warm_child(
            &p,
            &roles,
            &splits,
            &seed(),
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("overflow when subtracting duration from instant")
        )
        .is_none());
    }
    let mut malformed = seed();
    malformed
        .multipliers
        .g
        .resize(WARM_MAX_CANDIDATE_CELLS + 1, 0.0);
    assert!(warm_child(
        &p,
        &roles,
        &[HullSplit {
            neuron: 1,
            active: true
        }],
        &malformed,
        live()
    )
    .is_none());
}

#[test]
fn hull_reuse_coordinate_breakpoints_price_the_last_input() {
    let p = HullMarginProblem::new(
        vec![0.0, 1.0],
        vec![0.0],
        2,
        &[1.0],
        &[0.0],
        1,
        &[(0, 1.0)],
        0.0,
        vec![-1.0; 2],
        vec![1.0; 2],
    )
    .unwrap();
    assert_eq!(
        coordinate_proposal(&p, &[0.0, 1.0], &[0.0, -1.0], -0.25, 0.0, Some(1.0), live()),
        Some(1.0)
    );
    assert!(coordinate_proposal(
        &p,
        &[0.0, f64::NAN],
        &[0.0, -1.0],
        -0.25,
        0.0,
        Some(1.0),
        live()
    )
    .is_none());
}
