// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::tests::{cuts, end, exact_plan, twins};
use super::*;
#[test]
fn signed_root_last_mask_and_final_whole_pass_refusal_never_become_a_certificate() {
    let p = twins(0.25, false);
    let c = cuts(&p);
    let plan = exact_plan();
    for limit in [3, 4] {
        let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
        let cost = plan.cost(&p).unwrap();
        b.reserve(cost).unwrap();
        b.proof(cost, false).unwrap();
        let mut poll = || b.state.lock().unwrap().checked_masks < limit;
        assert!(certificate::check(&c, &plan, &b, b.search_end, &mut poll).is_none());
        assert_eq!(b.state.lock().unwrap().checked_masks, limit);
        assert_eq!(b.state.lock().unwrap().source_replays, 0);
    }
}
#[test]
fn signed_root_foreign_changed_source_tuple_and_selected_witness_bits_refuse() {
    let mut p = twins(0.25, false);
    let row = {
        let c = cuts(&p);
        propose(&c, end(), Duration::ZERO).0.unwrap()
    };
    let other = p.clone();
    assert!(row
        .replay_inner(&cuts(&other), end(), &mut || true, false)
        .is_none());
    p.b1[0] += 0.125;
    assert!(row
        .replay_inner(&cuts(&p), end(), &mut || true, false)
        .is_none());
    let p = twins(0.25, false);
    let mut c = cuts(&p);
    let (row, _) = propose(&c, end(), Duration::ZERO);
    let row = row.unwrap();
    c.multipliers.alpha[0] = 0.75;
    assert!(row.replay_inner(&c, end(), &mut || true, false).is_none());
    assert_eq!(row.stats().source_replays, 0);
    let c = cuts(&p);
    let (row, _) = propose(&c, end(), Duration::ZERO);
    let mut row = row.unwrap();
    row.plan.components[0].alpha[0] = f64::NAN;
    assert!(row.replay_inner(&c, end(), &mut || true, false).is_none());
    assert_eq!(row.stats().source_replays, 0);
}
#[test]
fn signed_root_expired_original_authority_cannot_be_promoted_by_a_later_end() {
    let p = twins(0.25, false);
    let c = cuts(&p);
    let (row, _) = propose(&c, end(), Duration::ZERO);
    let mut row = row.unwrap();
    row.budget.authority = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("overflow when subtracting duration from instant");
    assert!(row.replay_inner(&c, end(), &mut || true, false).is_none());
    assert!(
        !propose(
            &c,
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("overflow when subtracting duration from instant"),
            Duration::ZERO
        )
        .1
        .admitted
    );
}
#[test]
fn signed_root_two_reserved_full_source_replays_survive_exhausted_optional_credit() {
    let p = twins(0.25, false);
    let c = cuts(&p);
    let (row, _) = propose(&c, end(), Duration::ZERO);
    let row = row.unwrap();
    let left = row.budget.remaining().unwrap();
    row.budget.charge(left).unwrap();
    assert!(row.budget.charge(1).is_none());
    for _ in 0..2 {
        assert!(
            row.replay_inner(&c, end(), &mut || true, false)
                .unwrap()
                .root_lower
                > 0.249
        );
    }
    assert!(row.replay_inner(&c, end(), &mut || true, false).is_none());
    assert_eq!(row.stats().source_replays, 2);
    let state = row.budget.state.lock().unwrap();
    assert_eq!(state.reserved, 0);
    assert_eq!(state.work, MAX_WORK);
}
#[test]
fn signed_root_real_worker_keeps_earlier_whole_q_floor_when_later_search_credit_refuses() {
    let p = twins(0.25, false);
    let c = cuts(&p);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let mut exhausted = false;
    let mut poll = || {
        let improved = b.state.lock().unwrap().refined_components > 0;
        if improved && !exhausted {
            exhausted = true;
            let left = b.remaining().unwrap();
            b.charge(left).unwrap();
        }
        true
    };
    let (row, stats) = propose_with_budget(&c, &b, &mut poll);
    let row = row.unwrap();
    assert!(exhausted && stats.admitted && stats.refined_components == 1);
    assert!(row.lower > c.checked_lower + 0.1 && row.lower < 0.);
    let before = row.lower.to_bits();
    for _ in 0..2 {
        assert_eq!(
            row.replay_inner(&c, end(), &mut || true, false)
                .unwrap()
                .root_lower
                .to_bits(),
            before
        );
    }
    assert_eq!(c.proposal_lp_calls, 0);
}
#[test]
fn signed_root_malformed_shapes_overflows_masks_and_nominal_payload_refuse() {
    let p = twins(0.25, false);
    let mut c = cuts(&p);
    c.multipliers.g.pop();
    assert!(propose(&c, end(), Duration::ZERO).0.is_none());
    let c = cuts(&p);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    assert!(b.reserve(usize::MAX).is_none());
    assert!(b.charge(usize::MAX).is_none());
    assert_eq!(b.state.lock().unwrap().work, 0);
    let foreign = p.clone();
    assert!(propose_with_budget(&cuts(&foreign), &b, &mut || true)
        .0
        .is_none());
    let mut plan = exact_plan();
    plan.components[0].negative = vec![2; usize::BITS as usize];
    assert!(plan.masks().is_none() && plan.cost(&p).is_none() && plan.valid(&c).is_none());
    let large = HullMarginProblem::new(
        vec![1.; 512 * 128],
        vec![0.; 512],
        128,
        &(0..512)
            .map(|j| if j == 511 { -1. } else { 1. })
            .collect::<Vec<_>>(),
        &[0.25],
        1,
        &[(0, 1.)],
        0.,
        vec![-1.; 128],
        vec![1.; 128],
    )
    .unwrap();
    let large_c = cuts(&large);
    assert!(Budget::new(&large_c, end(), Duration::ZERO).is_none());
}
#[test]
fn signed_root_no_row_report_retains_work_reason_and_original_tuple_without_fabricated_replay() {
    let p = twins(0.25, false);
    let c = cuts(&p);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    b.charge(MAX_WORK - 1).unwrap();
    let old = c.multipliers.clone();
    let (row, r) = propose_with_budget(&c, &b, &mut || true);
    assert!(row.is_none() && r.attempted && !r.admitted && r.work_units > 0);
    assert_eq!(r.source_replays, 0);
    assert!(r.lower.is_none());
    assert_eq!(r.stop_reason, "work");
    assert_eq!(old, c.multipliers);
    assert_eq!(c.proposal_lp_calls, 0);
}
#[test]
fn signed_root_budget_aware_coordinate_subset_completes_masks_and_keeps_other_alpha() {
    let p = twins(0.25, false);
    let c = cuts(&p);
    let plan = exact_plan();
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(&p).unwrap();
    b.reserve(cost).unwrap();
    let masks = 4usize;
    let setup = masks * 5 * 3 * 4;
    let per_coordinate = masks * 2 * search::coordinate_work(2).unwrap();
    let keep = cost + setup + per_coordinate;
    b.charge(b.remaining().unwrap() - keep).unwrap();
    let old = vec![0.5; 8];
    let mut component = plan.components[0].clone();
    component.alpha = old.clone();
    let result = search::optimize(&c, &component, vec![0.; 2], &b, cost, &mut || true).unwrap();
    assert_eq!(b.state.lock().unwrap().completed_masks, 4);
    assert!(b.state.lock().unwrap().coordinate_proposals > 0);
    for mask in 0..4 {
        assert_eq!(
            result.alpha[2 * mask + 1].to_bits(),
            old[2 * mask + 1].to_bits()
        );
    }
    b.proof(cost, false).unwrap();
    let checked = certificate::check(
        &c,
        &Plan {
            components: vec![result],
            leftover: vec![],
        },
        &b,
        b.search_end,
        &mut || true,
    )
    .unwrap();
    assert!(checked.root_lower <= 0.25);
    d30_eight_negative_subset_control();
}

fn d30_eight_negative_subset_control() {
    let mut w = vec![0.; 24 * 30];
    for j in 0..24 {
        w[j * 30 + j % 8] = 1.;
    }
    let objective: Vec<_> = (0..24).map(|j| if j < 16 { 0.5 } else { -1. }).collect();
    let p = HullMarginProblem::new(
        w,
        vec![0.; 24],
        30,
        &objective,
        &[0.25],
        1,
        &[(0, 1.)],
        0.,
        vec![-1.; 30],
        vec![1.; 30],
    )
    .unwrap();
    let c = cuts(&p);
    let component = Component {
        negative: (16..24).collect(),
        positive: (0..16).collect(),
        positive_units: vec![2; 16],
        g: vec![0.; 30],
        alpha: vec![0.5; 256 * 16],
    };
    let plan = Plan {
        components: vec![component.clone()],
        leftover: vec![],
    };
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(&p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let baseline = certificate::check(&c, &plan, &b, b.search_end, &mut || true).unwrap();
    let coordinate = 256usize * 2 * search::coordinate_work(30).unwrap();
    let setup = 256usize * 25 * 31 * 4;
    // Keep the ORIGINAL strict-subset obligation, now against the actual
    // cheaper steering charge. A separate new control admits all16 at64M.
    let retained = cost + setup + 13 * coordinate;
    assert!(b.remaining().unwrap() > retained);
    b.charge(b.remaining().unwrap() - retained).unwrap();
    let refined = search::optimize(&c, &component, vec![0.; 30], &b, cost, &mut || true).unwrap();
    let active = refined.alpha[..16]
        .iter()
        .filter(|v| v.to_bits() != 0.5f64.to_bits())
        .count();
    assert_eq!(active, 13);
    for mask in 0..256 {
        for j in active..16 {
            assert_eq!(refined.alpha[mask * 16 + j], 0.5);
        }
    }
    assert_eq!(b.state.lock().unwrap().completed_masks, 256);
    let result = Plan {
        components: vec![refined],
        leftover: vec![],
    };
    b.proof(cost, false).unwrap();
    let proof = certificate::check(&c, &result, &b, b.search_end, &mut || true).unwrap();
    assert!(proof.root_lower > baseline.root_lower + 0.1 && proof.root_lower <= 0.25);
    assert_eq!(b.state.lock().unwrap().reserved, 2 * cost);
    let seal = Arc::new(signature(&c).collect());
    let completed = row(&c, &result, proof, cost, &b, &seal);
    for _ in 0..2 {
        assert!(completed
            .replay_inner(&c, end(), &mut || true, false)
            .is_some());
    }
    assert_eq!(completed.stats().source_replays, 2);
    assert!(completed.stats().work_units <= MAX_WORK);
}
