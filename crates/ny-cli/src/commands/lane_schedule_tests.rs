// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for #lane-value-scheduler.
//!
//! Every assertion here is on the BUDGET LEDGER — grants, charges, yields,
//! absorbed seconds — and never on a log line. The budget source is
//! [`SimulatedBudget`], so no test sleeps, no test reads a clock, and the
//! numbers below are the measured ones from the survey rather than round
//! placeholders.

use super::*;
use std::time::Duration;

const S: fn(u64) -> Duration = Duration::from_secs;

fn ms(v: u64) -> Duration {
    Duration::from_millis(v)
}

/// The two production plans, as pure functions of live remaining, so the tests
/// exercise the SAME rules the lanes use rather than re-implementations.
fn lp_plan(remaining: Option<Duration>) -> Option<Duration> {
    crate::commands::beta_crown::sign_space_falsify::lp_lane_plan(remaining)
}
fn ste_plan(remaining: Option<Duration>) -> Option<Duration> {
    crate::commands::beta_crown::sign_space_falsify::ste_lane_plan(remaining)
}

/// A lane that reports a fixed cost and a fixed value.
fn scripted(
    name: &'static str,
    plan: impl Fn(Option<Duration>) -> Option<Duration> + 'static,
    prior_value: f64,
    block: impl Fn(Duration) -> LaneBlock + 'static,
) -> LaneSpec {
    LaneSpec {
        name,
        plan: Box::new(plan),
        prior_value,
        run: Box::new(block),
    }
}

/// The publication margin the production pool is sized against.
const MARGIN: Duration = crate::commands::beta_crown::sign_space_falsify::LANE_PUBLICATION_MARGIN;
const FLOOR: Duration = Duration::from_secs(20);

// ---------------------------------------------------------------------------
// (a) a stalled lane yields and its seconds are REALLOCATED, not lost
// ---------------------------------------------------------------------------

/// THE measured case, on the ledger.
///
/// Row A (`model_48_idx_1703_eps_1`): the LP lane held 217.52 s of a 217.5 s
/// grant, so the STE lane's cap computed to 117.51 s.
/// Row B (`model_64_idx_1703_eps_1`): the LP lane's stall rule fired at
/// 53.56 s, so STE's cap computed to its 240.10 s ceiling.
///
/// Under the scheduler the second shape must be a property of the ledger: the
/// yielded seconds have to show up as a LARGER GRANT downstream, not as
/// unspent pool.
#[test]
fn a_stalled_lane_yields_and_the_next_lane_absorbs_the_seconds() {
    let remaining = S(480);
    let pool = lane_schedule_pool(Some(remaining), MARGIN).expect("480s leaves a pool");

    let lanes = vec![
        // The LP lane stalls at 53.56 s of whatever it is granted.
        scripted("lp_sign_space", lp_plan, 4.0, |_w| {
            measured_block(0.0, 32, ms(53_560))
        }),
        // The STE lane spends whatever cap it is handed.
        scripted("ste_pgd", ste_plan, 3.0, |w| measured_block(3.0, 1725, w)),
    ];
    let (witness, ledger) = run_lane_schedule(
        lanes,
        Box::new(SimulatedBudget::new(remaining)),
        pool,
        FLOOR,
    );
    assert!(witness.is_none());

    let lp = ledger.row("lp_sign_space").expect("lp ran");
    let ste = ledger.row("ste_pgd").expect("ste ran");

    // The LP lane's grant is its own unchanged rule: (480 - 45) * 0.5.
    assert_eq!(lp.granted, ms(217_500), "{}", ledger.describe());
    assert_eq!(lp.spent, ms(53_560));
    assert_eq!(
        lp.yielded,
        ms(163_940),
        "the stall must return 163.94 s to the pool: {}",
        ledger.describe()
    );

    // THE POINT. Had the LP lane spent its whole grant, STE's own rule would
    // have given it 480 - 217.5 - 45 - 100 = 117.5 s -- the measured row-A
    // number. It yielded, so STE plans against the live 426.44 s instead and
    // reaches its 240 s ceiling -- the measured row-B number.
    assert_eq!(
        ste.counterfactual_grant,
        ms(117_500),
        "counterfactual must reproduce the measured row-A cap"
    );
    assert_eq!(
        ste.granted,
        S(240),
        "the trailing lane must reach its ceiling: {}",
        ledger.describe()
    );
    assert_eq!(
        ste.absorbed(),
        ms(122_500),
        "122.5 s of the 163.94 s yield is absorbed downstream (the rest is the \
         lane's own 4-minute ceiling refusing to grow further)"
    );
    assert!(
        ledger.total_reallocated() > Duration::ZERO,
        "reallocation must be visible in the ledger, not only in a log line"
    );
}

/// The negative control for the same ledger: when the first lane does NOT
/// stall, nothing is reallocated and the downstream grant is exactly the
/// counterfactual. Without this, the test above could pass on arithmetic that
/// always inflates the second grant.
#[test]
fn a_lane_that_spends_its_whole_grant_reallocates_nothing() {
    let remaining = S(480);
    let pool = lane_schedule_pool(Some(remaining), MARGIN).expect("pool");
    let lanes = vec![
        scripted("lp_sign_space", lp_plan, 4.0, |w| {
            measured_block(0.0, 370, w)
        }),
        scripted("ste_pgd", ste_plan, 3.0, |w| measured_block(1.0, 1764, w)),
    ];
    let (_, ledger) = run_lane_schedule(
        lanes,
        Box::new(SimulatedBudget::new(remaining)),
        pool,
        FLOOR,
    );
    let lp = ledger.row("lp_sign_space").expect("lp");
    let ste = ledger.row("ste_pgd").expect("ste");
    assert_eq!(lp.yielded, Duration::ZERO);
    assert_eq!(ste.granted, ms(117_500), "{}", ledger.describe());
    assert_eq!(ste.granted, ste.counterfactual_grant);
    assert_eq!(ledger.total_reallocated(), Duration::ZERO);
}

/// A yield must not evaporate into unclaimed pool either. The survey measured
/// 57.32 s claimed by no lane at all across three rows; a scheduler that
/// merely stopped a lane early would just widen that hole.
#[test]
fn a_yield_does_not_become_unclaimed_pool() {
    let remaining = S(480);
    let pool = lane_schedule_pool(Some(remaining), MARGIN).expect("pool");
    let lanes = vec![
        scripted("lp_sign_space", lp_plan, 4.0, |_w| {
            measured_block(0.0, 32, ms(53_560))
        }),
        scripted("ste_pgd", ste_plan, 3.0, |w| measured_block(2.0, 1725, w)),
    ];
    let (_, ledger) = run_lane_schedule(
        lanes,
        Box::new(SimulatedBudget::new(remaining)),
        pool,
        FLOOR,
    );
    let unclaimed_without_realloc = pool.saturating_sub(ms(53_560)).saturating_sub(ms(117_500));
    assert!(
        ledger.unspent() < unclaimed_without_realloc,
        "the freed seconds must be handed onward, not left unclaimed \
         (unspent {:?} vs {:?}): {}",
        ledger.unspent(),
        unclaimed_without_realloc,
        ledger.describe()
    );
}

// ---------------------------------------------------------------------------
// (b) the instance budget is never exceeded
// ---------------------------------------------------------------------------

/// The publication margin is the hard wall: measured max wall 455.97 s against
/// a 480 s budget, and ONE `error` row makes a family unbankable. The pool is
/// sized so the scheduled lanes can never collectively cross it, and the
/// trailing lane's own plan still subtracts the 100 s downstream reserve.
#[test]
fn the_publication_margin_and_downstream_reserve_survive_every_shape() {
    // Greedy lanes that always burn their whole cap, over the whole plausible
    // budget range including the ones where a lane cannot start at all.
    for secs in [60u64, 100, 120, 200, 300, 480, 600, 1200] {
        let remaining = S(secs);
        let Some(pool) = lane_schedule_pool(Some(remaining), MARGIN) else {
            continue;
        };
        let lanes = vec![
            scripted("lp_sign_space", lp_plan, 4.0, |w| {
                measured_block(0.0, 100, w)
            }),
            scripted("ste_pgd", ste_plan, 3.0, |w| measured_block(0.0, 100, w)),
        ];
        let (_, ledger) = run_lane_schedule(
            lanes,
            Box::new(SimulatedBudget::new(remaining)),
            pool,
            FLOOR,
        );
        assert!(
            ledger.spent <= pool,
            "at {secs}s the lanes spent {:?} of a {:?} pool: {}",
            ledger.spent,
            pool,
            ledger.describe()
        );
        assert!(
            ledger.spent + MARGIN <= remaining,
            "at {secs}s the 45s publication margin was crossed: {}",
            ledger.describe()
        );
        // And the trailing lane's own plan never spends the downstream reserve.
        if let Some(ste) = ledger.row("ste_pgd").filter(|r| r.granted > Duration::ZERO) {
            let reserve = crate::commands::beta_crown::sign_space_falsify::LANE_DOWNSTREAM_RESERVE;
            let lp_spent = ledger
                .row("lp_sign_space")
                .map_or(Duration::ZERO, |r| r.spent);
            assert!(
                lp_spent + ste.granted + MARGIN + reserve <= remaining,
                "at {secs}s the trailing lane's grant ate the {reserve:?} downstream \
                 reserve: {}",
                ledger.describe()
            );
        }
    }
}

/// A lane that overruns its window cannot pull the aggregate across the
/// margin: it is charged what it took and the NEXT lane is refused.
#[test]
fn an_overrun_is_absorbed_by_refusing_the_next_lane_not_by_crossing_the_margin() {
    let remaining = S(480);
    let pool = lane_schedule_pool(Some(remaining), MARGIN).expect("pool");
    let lanes = vec![
        // 217.5 s granted, 430 s taken: worse than any overrun ever measured.
        scripted("lp_sign_space", lp_plan, 4.0, |_w| {
            measured_block(0.0, 370, S(430))
        }),
        scripted("ste_pgd", ste_plan, 3.0, |w| measured_block(1.0, 1725, w)),
    ];
    let (_, ledger) = run_lane_schedule(
        lanes,
        Box::new(SimulatedBudget::new(remaining)),
        pool,
        FLOOR,
    );
    assert_eq!(
        ledger.row("lp_sign_space").expect("lp").spent,
        S(430),
        "the overrun must be charged truthfully"
    );
    let ste = ledger.row("ste_pgd").expect("ste has a row");
    assert_eq!(
        ste.granted,
        Duration::ZERO,
        "the lane behind an overrun must be refused, not squeezed: {}",
        ledger.describe()
    );
    assert!(ste.declined.is_some());
    assert_eq!(ste.spent, Duration::ZERO);
    // And the wall still holds: the overrun ate the pool, not the margin.
    assert!(
        ledger.spent + MARGIN <= remaining,
        "an overrun must not cross the publication margin: {}",
        ledger.describe()
    );
}

// ---------------------------------------------------------------------------
// (c) a declined lane is not charged; an overrunning lane IS charged
// ---------------------------------------------------------------------------

/// Two different refusals, and neither may cost seconds.
///
/// `plan -> None` is the lane's OWN rule refusing (too little budget to be
/// worth starting); `PhaseYield::Declined` is the lane refusing after it was
/// admitted. The first must not even be granted a window.
#[test]
fn a_declining_lane_is_charged_nothing_and_does_not_block_the_pipeline() {
    let remaining = S(480);
    let pool = lane_schedule_pool(Some(remaining), MARGIN).expect("pool");
    let lanes = vec![
        // A lane whose own rule says "not on this budget", whatever is left.
        scripted(
            "structural_probe",
            |_r| None,
            100.0,
            |w| measured_block(0.0, 0, w),
        ),
        // A lane that is admitted and then declines on shape.
        scripted("lp_sign_space", lp_plan, 4.0, |_w| {
            declined_block(
                DeclineReason::Unsupported("not a binarized net"),
                Duration::ZERO,
            )
        }),
        scripted("ste_pgd", ste_plan, 3.0, |w| measured_block(1.0, 1725, w)),
    ];
    let (_, ledger) = run_lane_schedule(
        lanes,
        Box::new(SimulatedBudget::new(remaining)),
        pool,
        FLOOR,
    );

    let probe = ledger.row("structural_probe").expect("probe has a row");
    assert_eq!(probe.spent, Duration::ZERO, "{}", ledger.describe());
    assert_eq!(probe.granted, Duration::ZERO, "never even admitted");
    assert!(probe.declined.is_some());

    let lp = ledger.row("lp_sign_space").expect("lp ran");
    assert_eq!(lp.spent, Duration::ZERO, "a Declined block costs nothing");
    assert!(lp.declined.is_some());

    // AND the pipeline continued: a decline is not an instance event (I2).
    let ste = ledger.row("ste_pgd").expect("ste ran");
    assert_eq!(
        ste.granted,
        S(240),
        "with both lanes ahead declining, the trailing lane must reach its \
         ceiling: {}",
        ledger.describe()
    );
    assert_eq!(ledger.spent, S(240));
}

/// The overrun is charged, not the granted window. The margin-row lane was
/// measured overrunning a 45 s slice by 113.1 s; believing the grant is how a
/// budget goes silently negative.
#[test]
fn an_overrunning_lane_is_charged_what_it_actually_took() {
    let remaining = S(480);
    let pool = lane_schedule_pool(Some(remaining), MARGIN).expect("pool");
    let lanes = vec![scripted("lp_sign_space", lp_plan, 4.0, |w| {
        measured_block(0.0, 370, w + ms(113_100))
    })];
    let (_, ledger) = run_lane_schedule(
        lanes,
        Box::new(SimulatedBudget::new(remaining)),
        pool,
        FLOOR,
    );
    let lp = ledger.row("lp_sign_space").expect("lp");
    assert_eq!(lp.granted, ms(217_500));
    assert_eq!(lp.spent, ms(330_600), "217.5 granted + 113.1 overrun");
    assert_eq!(ledger.spent, ms(330_600));
    assert_eq!(lp.yielded, Duration::ZERO, "an overrun yields nothing");
}

/// A candidate ends the schedule: the pool is not spent answering a question
/// already answered. The witness is CLAIMED and still has to pass the caller's
/// unchanged trusted-oracle gate.
#[test]
fn a_candidate_stops_the_schedule_and_leaves_the_rest_of_the_pool_unspent() {
    let remaining = S(480);
    let pool = lane_schedule_pool(Some(remaining), MARGIN).expect("pool");
    let lanes = vec![
        // The measured row-C shape: 131.04 s of a 217.5 s cap, then a candidate.
        scripted("lp_sign_space", lp_plan, 4.0, |_w| {
            candidate_block("(( X_0 0.5 ))".to_string(), ms(131_040))
        }),
        scripted("ste_pgd", ste_plan, 3.0, |w| measured_block(1.0, 1725, w)),
    ];
    let (witness, ledger) = run_lane_schedule(
        lanes,
        Box::new(SimulatedBudget::new(remaining)),
        pool,
        FLOOR,
    );
    assert_eq!(witness.as_deref(), Some("(( X_0 0.5 ))"));
    assert!(
        ledger.row("ste_pgd").is_none(),
        "no lane runs after a candidate: {}",
        ledger.describe()
    );
    assert_eq!(ledger.spent, ms(131_040));
}

/// A lane with no honest value signal is recorded as such rather than being
/// given a fabricated number. BaB reported `domains_explored = 0` on all three
/// timeout rows and `output_bound_width = null` on every one of them.
#[test]
fn a_lane_with_no_value_signal_says_so_in_the_ledger() {
    let remaining = S(480);
    let pool = lane_schedule_pool(Some(remaining), MARGIN).expect("pool");
    let lanes = vec![scripted("no_signal_lane", lp_plan, 1.0, |w| {
        unmeasurable_block(w)
    })];
    let (_, ledger) = run_lane_schedule(
        lanes,
        Box::new(SimulatedBudget::new(remaining)),
        pool,
        FLOOR,
    );
    assert_eq!(
        ledger.row("no_signal_lane").expect("row").value,
        Some(LaneValue::NoSignal),
        "a work counter must not be dressed up as progress"
    );
}

/// The value a lane reports is in ITS OWN units, and the ledger keeps the
/// denominator. Seconds are the wrong denominator for the LP lane: 0.42 /
/// 0.59 / 1.67 s per LP solve on three rows of ONE family.
#[test]
fn the_value_report_carries_the_lanes_own_work_unit() {
    let remaining = S(480);
    let pool = lane_schedule_pool(Some(remaining), MARGIN).expect("pool");
    let lanes = vec![scripted("lp_sign_space", lp_plan, 4.0, |_w| {
        measured_block(302.0, 311, ms(131_040))
    })];
    let (_, ledger) = run_lane_schedule(
        lanes,
        Box::new(SimulatedBudget::new(remaining)),
        pool,
        FLOOR,
    );
    let value = ledger
        .row("lp_sign_space")
        .expect("row")
        .value
        .clone()
        .expect("the lane priced itself");
    assert_eq!(
        value,
        LaneValue::Measured {
            gain: 302.0,
            work_units: 311,
        }
    );
    assert_eq!(
        value.work_units(),
        311,
        "the denominator is LP solves, not seconds"
    );
    // A lane that reports no signal reports no denominator either, rather than
    // a zero that reads like a measurement.
    assert_eq!(LaneValue::NoSignal.work_units(), 0);
}

// ---------------------------------------------------------------------------
// (d) defaults unchanged
// ---------------------------------------------------------------------------

/// DARK BY DEFAULT, asserted on the arming predicate itself rather than on a
/// log line. The predicate is the production one with only the lookup
/// injected, so this is a test of the shipped rule.
#[test]
fn the_scheduler_is_dark_unless_the_lever_says_exactly_one() {
    assert!(!lane_value_scheduler_armed_from(None), "absent is dark");
    assert!(!lane_value_scheduler_armed_from(Some("0")));
    assert!(!lane_value_scheduler_armed_from(Some("")));
    assert!(
        !lane_value_scheduler_armed_from(Some("true")),
        "fails closed"
    );
    assert!(!lane_value_scheduler_armed_from(Some("yes")));
    assert!(!lane_value_scheduler_armed_from(Some(" 1")));
    assert!(lane_value_scheduler_armed_from(Some("1")));
}

/// The lever is DECLARED, so it appears in the receipt and cannot be a raw
/// environment read.
#[test]
fn the_lever_is_declared_and_defaults_off() {
    let decl = &ny_levers::decls::dark_probes::LANE_VALUE_SCHEDULER;
    assert_eq!(decl.name, "NY_LANE_VALUE_SCHEDULER");
    assert!(ny_levers::all().get("NY_LANE_VALUE_SCHEDULER").is_some());
    assert!(!lane_value_scheduler_armed_from(None));
}

/// (d) THE DEFAULT PATHS ARE UNCHANGED, on the two numbers that decide it: the
/// window each lane derives when nothing is armed. These are the SAME
/// functions the unscheduled lanes call, so if a refactor changed either rule,
/// this fails.
#[test]
fn the_unscheduled_lane_windows_are_the_measured_ones() {
    // 480 s traffic_signs row, at the top of the attack slice.
    assert_eq!(lp_plan(Some(S(480))), Some(ms(217_500)));
    // ... and the trailing lane, if the LP lane had spent all of it.
    assert_eq!(ste_plan(Some(ms(262_500))), Some(ms(117_500)));
    // ... and after a stall at 53.56 s, the trailing lane's 4-minute ceiling.
    assert_eq!(ste_plan(Some(ms(426_440))), Some(S(240)));
    // The floor: too little budget to be worth starting is `None`, not a
    // truncated window.
    assert_eq!(lp_plan(Some(S(60))), None);
    assert_eq!(ste_plan(Some(S(150))), None);
}

/// Absence of a published deadline is information absent, never budget spent.
/// A scheduler that read it as "nothing left" would be permissive at exactly
/// the wrong moment; one that read it as "everything left" would size a pool
/// off nothing. It refuses to build a pool at all.
#[test]
fn an_unpublished_deadline_produces_no_pool_rather_than_an_infinite_one() {
    assert_eq!(lane_schedule_pool(None, MARGIN), None);
    assert_eq!(
        lane_schedule_pool(Some(S(30)), MARGIN),
        None,
        "30s < margin"
    );
    assert_eq!(lane_schedule_pool(Some(S(480)), MARGIN), Some(S(435)));
}
