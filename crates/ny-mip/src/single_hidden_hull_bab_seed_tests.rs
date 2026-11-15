// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::single_hidden_hull::mip_cuts::propose_root_mip_cuts;

fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn abs_problem(c: f64) -> HullMarginProblem {
    // c-ReLU(x)+ReLU(2x)-ReLU(-x)+ReLU(-2x)=c+|x|.
    HullMarginProblem::new(
        vec![1.0, 2.0, -1.0, -2.0],
        vec![0.0; 4],
        1,
        &[-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        &[c, 0.0],
        2,
        &[(0, 1.0), (1, -1.0)],
        0.0,
        vec![-1.0],
        vec![1.0],
    )
    .unwrap()
}
fn root(p: &HullMarginProblem) -> (HullMultipliers, f64) {
    let bounds = preactivation_bounds(p);
    let roles = classify(p, &bounds);
    let m = triangle_multipliers(p, &roles, &bounds);
    let value = certify_hull_node_lower_bound(p, &[], &m, &[]).unwrap();
    (m, value)
}
fn positive_problem() -> (HullMarginProblem, HullMultipliers, f64) {
    // Constant inactive graph, so the whole original margin is +1/4.
    let p = HullMarginProblem::new(
        vec![0.0],
        vec![-1.0],
        1,
        &[-1.0],
        &[0.25],
        1,
        &[(0, 1.0)],
        0.0,
        vec![-1.0],
        vec![1.0],
    )
    .unwrap();
    let (m, value) = root(&p);
    (p, m, value)
}

#[test]
fn hull_seed_actual_completed_root_is_reused_without_another_root_lp() {
    let p = abs_problem(0.25);
    let cuts = propose_root_mip_cuts(&p, end()).unwrap();
    assert_eq!(cuts.proposal_lp_calls(), 1);
    assert!(cuts.checked_lower() <= 0.0);
    for workers in [1, 3] {
        let proof = cuts
            .continue_hull_bab(
                end(),
                HullBabOptions {
                    workers,
                    ..Default::default()
                },
            )
            .unwrap();
        let r = proof.report();
        assert!(proof.verified(), "{r:?}");
        assert_eq!(r.root_lp_calls, 0);
        assert_eq!(r.root_lp_status, HullLpStatus::Skipped);
        assert!(r.nodes >= 3 && r.closed >= 2 && r.open == 0);
        assert!(r.proposal_lp_calls < r.nodes);
    }
    assert_eq!(
        cuts.proposal_lp_calls(),
        1,
        "native reuse altered prior call census"
    );
}

#[test]
fn hull_seed_root_positive_needs_no_child_or_proposer_call() {
    let (p, m, value) = positive_problem();
    assert!(value > 0.249 && value <= 0.25);
    let r = from_completed(&p, &m, value, end(), Default::default()).unwrap();
    assert!(r.verified());
    assert_eq!((r.nodes, r.closed, r.open), (1, 1, 0));
    assert_eq!((r.root_lp_calls, r.proposal_lp_calls), (0, 0));
}

#[test]
fn hull_seed_complete_cover_proves_safe_and_never_unsafe_or_zero_tie() {
    for c in [0.25, 0.0, -0.25] {
        let p = abs_problem(c);
        let (m, value) = root(&p);
        assert!(value <= 0.0);
        let r = from_completed(&p, &m, value, end(), Default::default()).unwrap();
        assert_eq!(r.root_lp_calls, 0);
        assert_eq!(r.verified(), c > 0.0, "{c}: {r:?}");
        if c > 0.0 {
            assert!(r.closed >= 2 && r.open == 0)
        } else {
            // The original graph value at x=0 is exactly c<=0.
            assert!(p.b1.iter().all(|b| *b == 0.0));
            assert!(r.open > 0);
        }
    }
}

#[test]
fn hull_seed_attained_bits_shape_nonfinite_and_expired_proposals_refuse() {
    let p = abs_problem(0.25);
    let (m, value) = root(&p);
    assert!(root_from_completed(&p, &m, value.next_up(), end()).is_none());
    for mutation in 0..4 {
        let mut bad = m.clone();
        match mutation {
            0 => {
                bad.alpha.pop();
            }
            1 => {
                bad.g.pop();
            }
            2 => bad.g[0] = f64::INFINITY,
            _ => bad.g[0] = f64::NAN,
        }
        assert!(root_from_completed(&p, &bad, value, end()).is_none());
    }
    assert!(root_from_completed(
        &p,
        &m,
        value,
        Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("overflow when subtracting duration from instant")
    )
    .is_none());
}

#[test]
fn hull_seed_root_has_no_inherited_phase_or_beta_and_bad_phase_shapes_refuse() {
    let p = abs_problem(0.25);
    let (m, value) = root(&p);
    let (_, _, out) = root_from_completed(&p, &m, value, end()).unwrap();
    assert_eq!(out.warm.as_ref().unwrap().beta, [] as [f64; 0]);
    assert!(certify_hull_node_lower_bound(&p, &[], &m, &[1.0]).is_none());
    let split = HullSplit {
        neuron: 0,
        active: true,
    };
    assert!(certify_hull_node_lower_bound(&p, &[split], &m, &[f64::INFINITY]).is_none());
    assert!(certify_hull_node_lower_bound(&p, &[split, split], &m, &[0.0; 2]).is_none());
    assert!(certify_hull_node_lower_bound(
        &p,
        &[HullSplit {
            neuron: 99,
            active: false
        }],
        &m,
        &[0.0]
    )
    .is_none());
}

#[test]
fn hull_seed_late_already_closed_root_does_not_publish_proved() {
    let (p, m, value) = positive_problem();
    let (bounds, roles, root) = root_from_completed(&p, &m, value, end()).unwrap();
    let calls = AtomicUsize::new(0);
    let r = run_tree(
        &p,
        Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("overflow when subtracting duration from instant"),
        Default::default(),
        &bounds,
        &roles,
        root,
        &calls,
        0,
    );
    assert!(!r.verified());
    assert_eq!(r.stop, HullBabStop::Deadline);
    assert_eq!((r.closed, r.open), (1, 0));
    assert_eq!(r.proposal_lp_calls, 0);
}

#[test]
fn hull_seed_failed_branch_expansion_preserves_original_region() {
    let p = abs_problem(0.25);
    let (m, value) = root(&p);
    let (bounds, roles, mut out) = root_from_completed(&p, &m, value, end()).unwrap();
    out.branch = Some(usize::MAX);
    let calls = AtomicUsize::new(0);
    let r = run_tree(
        &p,
        end(),
        Default::default(),
        &bounds,
        &roles,
        out,
        &calls,
        0,
    );
    assert_eq!(r.stop, HullBabStop::WorkerFailure);
    assert_eq!(r.open, 1);
    assert!(!r.verified());
    assert_eq!(r.proposal_lp_calls, 0);
}

#[test]
fn hull_seed_node_cap_retains_the_original_open_cover_without_hidden_lp() {
    let p = abs_problem(0.25);
    let (m, value) = root(&p);
    let r = from_completed(
        &p,
        &m,
        value,
        end(),
        HullBabOptions {
            max_nodes: 1,
            workers: 3,
        },
    )
    .unwrap();
    assert!(!r.verified());
    assert_eq!(r.stop, HullBabStop::NodeCap);
    assert_eq!((r.nodes, r.closed, r.open), (1, 0, 1));
    assert_eq!(r.proposal_lp_calls, 0);
}

#[test]
fn hull_seed_existing_candidate_cell_cap_refuses_oversized_warm_storage() {
    let (d, h) = (128, 512);
    let p = HullMarginProblem::new(
        vec![0.125; d * h],
        vec![0.0; h],
        d,
        &vec![-1.0; h],
        &[0.0],
        1,
        &[(0, 1.0)],
        0.0,
        vec![-1.0; d],
        vec![1.0; d],
    )
    .unwrap();
    let (m, value) = root(&p);
    let (_, _, out) = root_from_completed(&p, &m, value, end()).unwrap();
    assert!(out.warm.is_none());
    assert_eq!(out.lp_status, HullLpStatus::Skipped);
}
