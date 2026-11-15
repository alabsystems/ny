// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Complete-row retention and unchanged shared authority/ledger controls.
use super::super::tests::{cuts, end, graph, rat};
use super::tests::{fixture, shared, whole};
use super::*;

fn bank(c: &HullRootMipCuts<'_>, p: &Plan, b: &Budget) -> SignedMajorant {
    let cost = p.cost(c.problem).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(c, p, b, b.search_end, &mut || true).unwrap();
    let seal = Arc::new(signature(c).collect());
    row(c, p, checked, cost, b, &seal)
}
#[test]
fn signed_mass_weaker_initial_seed_is_refined_before_comparison_with_banked_whole() {
    let p = fixture(0.25, false);
    let c = cuts(&p);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let mut selected = whole();
    let old = bank(&c, &selected, &b);
    let mut current = old.lower;
    let mut best = Some(old);
    let seal = Arc::new(signature(&c).collect());
    refine(
        &c,
        &mut selected,
        &mut current,
        &mut best,
        c.checked_lower,
        &b,
        &seal,
        &mut || true,
    );
    let row = best.unwrap();
    assert!(row.lower > 0.249 && row.lower <= 0.25);
    assert!(selected
        .components
        .iter()
        .all(|c| c.positive_units == vec![1]));
    assert_eq!(b.state.lock().unwrap().completed_masks, 512);
    assert!(b.state.lock().unwrap().checks <= MAX_CHECKS);
    b.trim_unspent_reservation(row.cost).unwrap();
    b.charge(b.remaining().unwrap()).unwrap();
    for _ in 0..2 {
        assert!(
            row.replay_inner(&c, end(), &mut || true, false)
                .unwrap()
                .root_lower
                > 0.249
        );
    }
    assert!(row.replay_inner(&c, end(), &mut || true, false).is_none());
    let counts = b.state.lock().unwrap();
    assert_eq!(counts.source_replays, 2);
    assert_eq!(counts.work, MAX_WORK);
}
#[test]
fn signed_mass_partial_second_component_keeps_the_completed_whole_row_and_replay_reserve() {
    let p = fixture(0.25, false);
    let c = cuts(&p);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let mut selected = whole();
    let old = bank(&c, &selected, &b);
    let before = old.lower.to_bits();
    let old_cost = old.cost;
    let mut current = old.lower;
    let mut best = Some(old);
    let seal = Arc::new(signature(&c).collect());
    let mut poll = || b.state.lock().unwrap().completed_masks < 256;
    refine(
        &c,
        &mut selected,
        &mut current,
        &mut best,
        c.checked_lower,
        &b,
        &seal,
        &mut poll,
    );
    assert_eq!(b.state.lock().unwrap().completed_masks, 256);
    let row = best.unwrap();
    assert_eq!(row.lower.to_bits(), before);
    assert_eq!(row.cost, old_cost);
    assert_eq!(selected.components[0].positive_units, vec![2]);
    b.trim_unspent_reservation(old_cost).unwrap();
    assert_eq!(b.state.lock().unwrap().reserved, 2 * old_cost);
    for _ in 0..2 {
        assert_eq!(
            row.replay_inner(&c, end(), &mut || true, false)
                .unwrap()
                .root_lower
                .to_bits(),
            before
        );
    }
}
#[test]
fn signed_mass_last_mask_and_final_poll_refuse_partial_512_mask_certificates() {
    let p = fixture(0.25, false);
    let c = cuts(&p);
    let plan = shared();
    for limit in [511, 512] {
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
fn signed_mass_optional_payload_and_work_refusal_preserve_the_original_whole_selection() {
    for payload in [false, true] {
        let p = fixture(0.25, false);
        let c = cuts(&p);
        let mut b = Budget::new(&c, end(), Duration::ZERO).unwrap();
        let mut selected = whole();
        let old = bank(&c, &selected, &b);
        let before = old.lower.to_bits();
        let mut current = old.lower;
        let mut best = Some(old);
        let seal = Arc::new(signature(&c).collect());
        if payload {
            b.admitted_bytes = MAX_BYTES;
        } else {
            b.charge(b.remaining().unwrap()).unwrap();
        }
        refine(
            &c,
            &mut selected,
            &mut current,
            &mut best,
            c.checked_lower,
            &b,
            &seal,
            &mut || true,
        );
        assert_eq!(best.unwrap().lower.to_bits(), before);
        assert_eq!(b.state.lock().unwrap().source_replays, 0);
        assert!(b.state.lock().unwrap().work <= MAX_WORK);
    }
}
#[test]
fn signed_mass_foreign_source_or_unit_witness_mutation_cannot_reuse_a_completed_row() {
    let mut p = fixture(0.25, false);
    let mut row = {
        let c = cuts(&p);
        let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
        bank(&c, &shared(), &b)
    };
    let foreign = p.clone();
    assert!(row
        .replay_inner(&cuts(&foreign), end(), &mut || true, false)
        .is_none());
    row.plan.components[0].positive_units[0] = 2;
    assert!(row
        .replay_inner(&cuts(&p), end(), &mut || true, false)
        .is_none());
    assert_eq!(row.stats().source_replays, 0);
    let row = {
        let c = cuts(&p);
        let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
        bank(&c, &shared(), &b)
    };
    p.w1[0] = 0.5;
    assert!(row
        .replay_inner(&cuts(&p), end(), &mut || true, false)
        .is_none());
}
#[test]
fn signed_mass_reservation_raise_is_transactional_and_cannot_reset_used_checks_or_replays() {
    let p = fixture(0.25, false);
    let c = cuts(&p);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let cost = whole().cost(&p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let old = {
        let c = b.state.lock().unwrap();
        (c.work, c.reserved)
    };
    assert!(b.reserve_at_least(usize::MAX).is_none());
    assert_eq!(
        {
            let c = b.state.lock().unwrap();
            (c.work, c.reserved)
        },
        old
    );
    b.reserve_at_least(cost + 1).unwrap();
    assert_eq!(b.state.lock().unwrap().checks, 1);
    b.trim_unspent_reservation(cost).unwrap();
    b.proof(cost, true).unwrap();
    assert!(b.reserve_at_least(cost).is_none());
    assert!(b.trim_unspent_reservation(cost).is_none());
    assert_eq!(b.state.lock().unwrap().replay_attempts, 1);
}
#[test]
fn signed_mass_exhausted_check_slots_or_original_clock_cannot_be_renewed_by_sharing() {
    for expired in [false, true] {
        let p = fixture(0.25, false);
        let c = cuts(&p);
        let mut b = Budget::new(&c, end(), Duration::ZERO).unwrap();
        let mut selected = whole();
        let old = bank(&c, &selected, &b);
        let before = old.lower.to_bits();
        let mut current = old.lower;
        let mut best = Some(old);
        let seal = Arc::new(signature(&c).collect());
        if expired {
            b.search_end = Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("overflow when subtracting duration from instant");
        } else {
            b.state.lock().unwrap().checks = MAX_CHECKS;
        }
        refine(
            &c,
            &mut selected,
            &mut current,
            &mut best,
            c.checked_lower,
            &b,
            &seal,
            &mut || true,
        );
        assert_eq!(best.unwrap().lower.to_bits(), before);
        assert_eq!(b.state.lock().unwrap().source_replays, 0);
    }
}
#[test]
fn signed_mass_source_dimension_mismatch_refuses_the_shared_worker_before_any_indexing() {
    let p = fixture(0.25, false);
    let c = cuts(&p);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let other = fixture(0.25, true);
    assert!(plan(&cuts(&other), &whole(), &b).is_none());
    assert_eq!(b.state.lock().unwrap().work, 0);
}
#[test]
fn signed_mass_d30_h128_default_worker_funds_real_shared_coverage_from_a_retained_partition() {
    let d = 30;
    let h = 128;
    let objective: Vec<_> = (0..h).map(|j| if j < 74 { 1. } else { -2. }).collect();
    let mut p = HullMarginProblem::new(
        vec![1. / 64.; d * h],
        vec![0.; h],
        d,
        &objective,
        &[-0.25],
        1,
        &[(0, 1.)],
        0.,
        vec![-1.; d],
        vec![1.; d],
    )
    .unwrap();
    p.c = objective;
    p.d = -0.25;
    let mut c = cuts(&p);
    let gb = group::Budget::new(end(), Duration::ZERO).unwrap();
    let sizes = [8, 8, 8, 8, 8, 2, 2, 2];
    let mut first = 74;
    let mut groups = Vec::new();
    for n in sizes {
        let heads: Vec<_> = (first..first + n).collect();
        first += n;
        groups.push(group::support(&c, &heads, &gb, &mut || true).unwrap());
    }
    // The retained partition is independently certified; its discovery is not
    // asserted. Eight remaining negatives become ordinary singleton components.
    c.compact = Some(compact::build_group_aggregate(&c, groups, &gb, &mut || true).unwrap());
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let mut saw_before_legacy = false;
    let mut poll = || {
        let counts = b.state.lock().unwrap();
        if counts.sharing_attempted
            && counts.direction.attempts == 0
            && counts.completed_masks > 256
            && counts.checked_masks >= 3 * 1308
        {
            saw_before_legacy = true;
        }
        true
    };
    let (row, report) = propose_with_budget(&c, &b, &mut poll);
    assert!(
        saw_before_legacy,
        "complete shared advice/check must precede expensive legacy work"
    );
    assert_eq!(report.components, 16);
    assert_eq!(report.negative_heads, 54);
    assert!(b.state.lock().unwrap().sharing_attempted);
    assert!(report.checked_masks >= 3 * 1308 && report.completed_masks > 256);
    assert!(report.work_units <= MAX_WORK && report.admitted_bytes <= MAX_BYTES);
    assert!(b.state.lock().unwrap().checks <= MAX_CHECKS);
    assert_eq!(c.proposal_lp_calls, 0);
    if let Some(row) = row {
        assert!(rat(row.lower) <= graph(&p, &vec![1.; d]));
    }
}

#[test]
fn signed_mass_completed_old_positive_proof_spends_no_optional_sharing_credit() {
    let p = fixture(0.25, false);
    let c = cuts(&p);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let mut selected = shared();
    let old = bank(&c, &selected, &b);
    let mut current = old.lower;
    assert!(current > 0.249);
    let before = {
        let c = b.state.lock().unwrap();
        (c.work, c.checks, c.reserved)
    };
    let mut best = Some(old);
    let seal = Arc::new(signature(&c).collect());
    refine(
        &c,
        &mut selected,
        &mut current,
        &mut best,
        c.checked_lower,
        &b,
        &seal,
        &mut || true,
    );
    let counts = b.state.lock().unwrap();
    assert!(!counts.sharing_attempted);
    assert_eq!((counts.work, counts.checks, counts.reserved), before);
    assert_eq!(counts.coordinate_proposals, 0);
}

fn independent_slopes() -> (HullMarginProblem, Plan) {
    let mut w = vec![0.; 18 * 2];
    w[0] = 1.;
    w[3] = 1.;
    for j in 2..18 {
        w[j * 2 + usize::from((j - 2) % 8 >= 4)] = 1.;
    }
    let objective: Vec<_> = (0..18).map(|j| if j < 2 { 8. } else { -1. }).collect();
    let mut p = HullMarginProblem::new(
        w,
        vec![0.; 18],
        2,
        &objective,
        &[0.25],
        1,
        &[(0, 1.)],
        0.,
        vec![-1.; 2],
        vec![1.; 2],
    )
    .unwrap();
    p.c = objective;
    p.d = 0.25;
    let components = (0..2)
        .map(|group| Component {
            negative: (2 + 8 * group..10 + 8 * group).collect(),
            positive: vec![group],
            positive_units: vec![2],
            g: if group == 0 {
                vec![2., -2.]
            } else {
                vec![-2., 2.]
            },
            alpha: (0u32..256)
                .map(|mask| {
                    let count = if group == 0 {
                        (mask & 15).count_ones()
                    } else {
                        (mask >> 4).count_ones()
                    };
                    (count as f64 + 2.) / 8.
                })
                .collect(),
        })
        .collect();
    (
        p,
        Plan {
            components,
            leftover: vec![],
        },
    )
}
fn independent_cuts(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    let mut m = HullMultipliers::zeros(p);
    m.alpha[0] = 0.5;
    m.alpha[1] = 0.25;
    for j in 2..18 {
        for i in 0..2 {
            m.g[j * 2 + i] = p.w_row(j)[i] * if i == 0 { 0.5 } else { 0.25 };
        }
    }
    let floor = certify_hull_lower_bound(p, &m).unwrap();
    prepare(p, m, floor, end(), &mut || true).unwrap()
}
#[test]
fn signed_mass_shared_spare_work_reaches_both_independent_positive_slopes_for_strict_gain() {
    let (p, mut selected) = independent_slopes();
    let c = independent_cuts(&p);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let old = bank(&c, &selected, &b);
    assert!(old.lower <= -3.75 && old.lower > -3.750001);
    let mut current = old.lower;
    let mut best = Some(old);
    let seal = Arc::new(signature(&c).collect());
    refine(
        &c,
        &mut selected,
        &mut current,
        &mut best,
        c.checked_lower,
        &b,
        &seal,
        &mut || true,
    );
    assert!(best.as_ref().unwrap().lower > 0.249 && best.as_ref().unwrap().lower <= 0.25);
    let counts = b.state.lock().unwrap();
    assert_eq!(counts.mass.funded_visits, 2);
    assert_eq!(counts.mass.completed_visits, 2);
    assert_eq!(counts.mass.final_checks, 1);
    assert!(counts.mass.selected);
    drop(counts);
    // Initial y alpha=.25 and original negative y g=.25 give shared G=0.
    // Improving x alone leaves -3 per component, BELOW the banked -3.75.
    assert!(selected.components.iter().all(|c| c
        .alpha
        .as_chunks::<2>()
        .0
        .iter()
        .any(|a| a[1] != 0.25)));
    for x in [[-1., -1.], [-1., 1.], [1., -1.], [1., 1.], [0., 0.]] {
        assert_eq!(graph(&p, &x), rat(0.25));
    }
}
#[test]
fn signed_mass_minimum_budget_protects_every_visit_and_final_checker_but_keeps_old_no_gain_row() {
    let (p, mut selected) = independent_slopes();
    let c = independent_cuts(&p);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let old = bank(&c, &selected, &b);
    let before = old.lower.to_bits();
    assert_eq!(old.cost, 77_248);
    // Independent arithmetic from D2/H18/P2/N16/K2 and two 256-mask groups:
    // geometry1608 +metadata672 +hold raise2*(87488-77248)
    // +two complete shared checks2*87488 +two one-coordinate visits2*107520.
    let keep = 1_608 + 672 + 2 * (87_488 - 77_248) + 2 * 87_488 + 2 * 107_520;
    b.charge(b.remaining().unwrap() - keep).unwrap();
    let mut current = old.lower;
    let mut best = Some(old);
    let seal = Arc::new(signature(&c).collect());
    refine(
        &c,
        &mut selected,
        &mut current,
        &mut best,
        c.checked_lower,
        &b,
        &seal,
        &mut || true,
    );
    assert_eq!(best.as_ref().unwrap().lower.to_bits(), before);
    assert_eq!(b.remaining().unwrap(), 0);
    let counts = b.state.lock().unwrap();
    assert_eq!(counts.mass.funded_visits, 2);
    assert_eq!(counts.mass.completed_visits, 2);
    assert_eq!(counts.mass.baseline_checks, 1);
    assert_eq!(counts.mass.final_checks, 1);
    assert!(!counts.mass.selected);
    assert_eq!(counts.completed_masks, 512);
    assert_eq!(counts.checked_masks, 1536);
    drop(counts);
    b.trim_unspent_reservation(77_248).unwrap();
    for _ in 0..2 {
        assert_eq!(
            best.as_ref()
                .unwrap()
                .replay_inner(&c, end(), &mut || true, false)
                .unwrap()
                .root_lower
                .to_bits(),
            before
        );
    }
}
