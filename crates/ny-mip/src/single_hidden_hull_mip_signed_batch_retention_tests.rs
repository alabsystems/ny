// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::super::tests::{cuts, end};
use super::tests::{baseline, family, leave, run, sum_min};
use super::*;

#[test]
fn signed_batch_last_search_mask_refusal_discards_every_staged_component() {
    let (p, mut plan) = family(&[2, 1], -0.25);
    let c = cuts(&p);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let (cost, mut current, previous) = baseline(&c, &plan, &b);
    let before = witness(&plan);
    let lower = current.to_bits();
    leave(&b, overhead(&p, &plan).unwrap() + cost + sum_min(&p, &plan));
    let mut best = Some(previous);
    let masks = plan.masks().unwrap();
    let mut poll = || b.state.lock().unwrap().completed_masks < masks - 1;
    assert!(run(&c, &mut plan, &b, cost, &mut current, &mut best, &mut poll).is_none());
    assert_eq!(b.state.lock().unwrap().completed_masks, masks - 1);
    assert_eq!(b.state.lock().unwrap().checked_masks, masks);
    assert_eq!(witness(&plan), before);
    assert_eq!(current.to_bits(), lower);
    assert_eq!(b.state.lock().unwrap().reserved, 2 * cost);
    for _ in 0..2 {
        assert_eq!(
            best.as_ref()
                .unwrap()
                .replay_inner(&c, end(), &mut || true, false)
                .unwrap()
                .root_lower
                .to_bits(),
            lower
        );
    }
}
#[test]
fn signed_batch_last_checker_mask_refusal_retains_original_checked_plan_and_floor() {
    let (p, mut plan) = family(&[2, 1], -0.25);
    let c = cuts(&p);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let (cost, mut current, previous) = baseline(&c, &plan, &b);
    let before = witness(&plan);
    let lower = current.to_bits();
    let masks = plan.masks().unwrap();
    leave(&b, overhead(&p, &plan).unwrap() + cost + sum_min(&p, &plan));
    let mut best = Some(previous);
    let mut poll = || b.state.lock().unwrap().checked_masks < 2 * masks - 1;
    assert!(run(&c, &mut plan, &b, cost, &mut current, &mut best, &mut poll).is_none());
    assert_eq!(b.state.lock().unwrap().completed_masks, masks);
    assert_eq!(b.state.lock().unwrap().checked_masks, 2 * masks - 1);
    assert_eq!(witness(&plan), before);
    assert_eq!(current.to_bits(), lower);
    assert_eq!(best.as_ref().unwrap().lower.to_bits(), lower);
    assert_eq!(b.state.lock().unwrap().reserved, 2 * cost);
}
#[test]
fn signed_batch_check_slot_pressure_can_complete_after_prepaying_the_last_slot() {
    let (p, mut plan) = family(&[1, 1], -0.25);
    let c = cuts(&p);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let (cost, mut current, previous) = baseline(&c, &plan, &b);
    b.state.lock().unwrap().checks = MAX_CHECKS - 1;
    let mut best = Some(previous);
    assert!(run(
        &c,
        &mut plan,
        &b,
        cost,
        &mut current,
        &mut best,
        &mut || true
    )
    .is_some());
    assert_eq!(b.state.lock().unwrap().checks, MAX_CHECKS);
    assert_eq!(b.state.lock().unwrap().completed_masks, 4);
    assert_eq!(b.state.lock().unwrap().checked_masks, 8);
    assert_eq!(b.state.lock().unwrap().reserved, 2 * cost);
}
#[test]
fn signed_batch_non_improving_complete_trial_keeps_checked_witness_and_best_bits() {
    let (p, mut plan) = family(&[2, 1], -0.25);
    for c in &mut plan.components {
        for mask in 0..1 << c.negative.len() {
            for j in 0..c.positive.len() {
                c.alpha[mask * c.positive.len() + j] = if mask & (1 << (j % c.negative.len())) != 0
                {
                    1.
                } else {
                    0.
                };
            }
        }
    }
    let c = cuts(&p);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let (cost, mut current, previous) = baseline(&c, &plan, &b);
    b.state.lock().unwrap().checks = MAX_CHECKS - 1;
    let lower = current.to_bits();
    let before = witness(&plan);
    let mut best = Some(previous);
    assert!(run(
        &c,
        &mut plan,
        &b,
        cost,
        &mut current,
        &mut best,
        &mut || true
    )
    .is_none());
    assert_eq!(witness(&plan), before);
    assert_eq!(current.to_bits(), lower);
    assert_eq!(best.as_ref().unwrap().lower.to_bits(), lower);
    assert_eq!(b.state.lock().unwrap().checked_masks, 12);
    assert_eq!(b.state.lock().unwrap().refined_components, 0);
}
#[test]
fn signed_batch_foreign_source_expired_clock_and_full_check_cap_refuse_without_mutation() {
    for case in 0..3 {
        let (p, mut plan) = family(&[2, 1], -0.25);
        let c = cuts(&p);
        let mut b = Budget::new(&c, end(), Duration::ZERO).unwrap();
        let (cost, mut current, previous) = baseline(&c, &plan, &b);
        let before = witness(&plan);
        let lower = current.to_bits();
        match case {
            0 => b.source ^= 1,
            1 => {
                b.search_end = Instant::now()
                    .checked_sub(Duration::from_secs(1))
                    .expect("overflow when subtracting duration from instant")
            }
            _ => b.state.lock().unwrap().checks = MAX_CHECKS,
        }
        let work = b.state.lock().unwrap().work;
        let mut best = Some(previous);
        assert!(run(
            &c,
            &mut plan,
            &b,
            cost,
            &mut current,
            &mut best,
            &mut || true
        )
        .is_none());
        assert_eq!(witness(&plan), before);
        assert_eq!(current.to_bits(), lower);
        assert_eq!(b.state.lock().unwrap().work, work);
    }
}
#[test]
fn signed_batch_bad_alpha_and_incomplete_role_partition_refuse_before_staging() {
    for case in 0..2 {
        let (p, mut plan) = family(&[2, 1], -0.25);
        let c = cuts(&p);
        let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
        let (cost, mut current, previous) = baseline(&c, &plan, &b);
        if case == 0 {
            plan.components[0].alpha[0] = f64::NAN;
        } else {
            plan.components[1].negative[0] = plan.components[0].negative[0];
        }
        let before = witness(&plan);
        let mut best = Some(previous);
        assert!(run(
            &c,
            &mut plan,
            &b,
            cost,
            &mut current,
            &mut best,
            &mut || true
        )
        .is_none());
        assert_eq!(witness(&plan), before);
        assert_eq!(b.state.lock().unwrap().completed_masks, 0);
        assert_eq!(b.state.lock().unwrap().checks, 1);
    }
}
#[test]
fn signed_batch_no_complete_minimum_or_wrong_check_quote_preserves_old_certificate() {
    for wrong_cost in [false, true] {
        let (p, mut plan) = family(&[2, 1], -0.25);
        let c = cuts(&p);
        let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
        let (cost, mut current, previous) = baseline(&c, &plan, &b);
        let before = witness(&plan);
        let lower = current.to_bits();
        if !wrong_cost {
            let minimum = plan
                .components
                .iter()
                .map(|c| search::work(&p, c).unwrap().cost(1).unwrap())
                .min()
                .unwrap();
            leave(&b, overhead(&p, &plan).unwrap() + cost + minimum - 1);
        }
        let mut best = Some(previous);
        let quote = if wrong_cost { cost - 1 } else { cost };
        assert!(run(
            &c,
            &mut plan,
            &b,
            quote,
            &mut current,
            &mut best,
            &mut || true
        )
        .is_none());
        assert_eq!(witness(&plan), before);
        assert_eq!(current.to_bits(), lower);
        assert_eq!(best.as_ref().unwrap().lower.to_bits(), lower);
        assert_eq!(b.state.lock().unwrap().completed_masks, 0);
        assert_eq!(b.state.lock().unwrap().reserved, 2 * cost);
    }
}
