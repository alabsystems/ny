// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for the Layer-A allocator wiring.
//!
//! The five obligations, and where each is discharged:
//!
//! (a) lever unset ⇒ byte-identical behaviour, asserted on a CODE PATH —
//!     `disarmed_every_lane_keeps_its_private_fraction` and
//!     `the_lever_is_dark_by_default`.
//! (b) a lane granted 0 is SKIPPED and its seconds land in another grant —
//!     `a_flat_objective_skips_the_upfront_lane_and_its_seconds_move`.
//! (c) grants + reserve never exceed the budget, randomized —
//!     `grants_plus_reserve_never_exceed_the_budget`.
//! (d) fail open ⇒ exactly today's plan — `a_forced_allocator_failure_falls_open`
//!     and `an_impossible_solve_cap_falls_open`.
//! (e) a FLAT tier zeroes the gradient-guided lane —
//!     `a_flat_objective_tier_zeroes_the_gradient_guided_lane`.

use super::*;

// ---------------------------------------------------------------------------
// (a) the disarmed arm
// ---------------------------------------------------------------------------

#[test]
fn the_lever_is_dark_by_default() {
    assert!(!lane_budget_allocator_armed_from(None));
    assert!(!lane_budget_allocator_armed_from(Some("0")));
    // Anything that is not exact "1" is a recorded rejection resolving to the
    // declaration's `false`, so a typo can never arm a scored run.
    assert!(!lane_budget_allocator_armed_from(Some("true")));
    assert!(!lane_budget_allocator_armed_from(Some(" 1")));
    assert!(lane_budget_allocator_armed_from(Some("1")));
}

/// (a) THE byte-identical assertion, on the code path rather than on a log
/// line: with no allocation in force every lane's window is `Private`, which
/// is the value each call site matches to take its own unchanged helper.
#[test]
fn disarmed_every_lane_keeps_its_private_fraction() {
    for lane in [
        AllocatedLane::SignSpace,
        AllocatedLane::StePgd,
        AllocatedLane::UpfrontAttack,
    ] {
        assert_eq!(lane_window(None, lane), LaneWindow::Private);
    }
}

/// (a) The admission composition, so "disarmed ⇒ nothing happens" is a fact
/// about one predicate rather than about three scattered conditions.
#[test]
fn the_allocator_is_admitted_only_when_it_alone_owns_the_caps() {
    // Disarmed dominates everything: nothing else is even consulted.
    for scheduler in [false, true] {
        for peel in [false, true] {
            assert!(!allocator_admitted(false, scheduler, peel));
        }
    }
    // Armed, but the in-flight scheduler already owns the two BNN lanes.
    assert!(!allocator_admitted(true, true, false));
    // Armed, but the traffic peel rewrites the objective the probe measures.
    assert!(!allocator_admitted(true, false, true));
    assert!(allocator_admitted(true, false, false));
}

#[test]
fn a_committed_cap_is_clamped_to_the_live_remaining_budget() {
    let margin = Duration::from_secs(45);
    // The common case: the cap fits.
    assert_eq!(
        clamp_to_live_remaining(
            Duration::from_mins(4),
            Some(Duration::from_secs(400)),
            margin
        ),
        Some(Duration::from_mins(4))
    );
    // An upstream overrun must SHRINK the cap, never push past the budget.
    assert_eq!(
        clamp_to_live_remaining(
            Duration::from_mins(4),
            Some(Duration::from_secs(100)),
            margin
        ),
        Some(Duration::from_secs(55))
    );
    // Nothing usable left: do not consult the lane at all.
    assert_eq!(
        clamp_to_live_remaining(
            Duration::from_mins(4),
            Some(Duration::from_secs(45)),
            margin
        ),
        None
    );
    assert_eq!(
        clamp_to_live_remaining(
            Duration::from_mins(4),
            Some(Duration::from_secs(10)),
            margin
        ),
        None
    );
    // No published deadline is information absent, never budget spent.
    assert_eq!(
        clamp_to_live_remaining(Duration::from_mins(4), None, margin),
        Some(Duration::from_mins(4))
    );
}

// ---------------------------------------------------------------------------
// The probe
// ---------------------------------------------------------------------------

#[test]
fn probe_points_are_inside_the_box_and_deterministic() {
    let lo = vec![-1.0f32, 0.0, 3.0];
    let hi = vec![1.0f32, 0.0, 4.0];
    let a = probe_points(&lo, &hi, OBJECTIVE_PROBE_POINTS);
    let b = probe_points(&lo, &hi, OBJECTIVE_PROBE_POINTS);
    assert_eq!(a, b, "the probe must be reproducible from the box alone");
    assert_eq!(a.len(), OBJECTIVE_PROBE_POINTS);
    for point in &a {
        for (i, &v) in point.iter().enumerate() {
            assert!(
                v >= lo[i] && v <= hi[i],
                "probe point escaped the box at coordinate {i}: {v}"
            );
        }
    }
    // A pinned coordinate stays pinned at every point.
    assert!(a.iter().all(|p| p[1] == 0.0));
}

#[test]
fn a_constant_objective_is_flat_and_a_varying_one_is_not() {
    let lo = vec![0.0f32; 4];
    let hi = vec![1.0f32; 4];
    let points = probe_points(&lo, &hi, OBJECTIVE_PROBE_POINTS);

    let flat = probe_objective(&points, OBJECTIVE_PROBE_WALL, Instant::now(), |_| {
        Some(-1.0)
    })
    .expect("a constant objective is evaluable everywhere");
    assert_eq!(flat.distinct_values, 1);
    assert_eq!(flat.points, OBJECTIVE_PROBE_POINTS);
    assert!(flat.is_flat());

    let varying = probe_objective(&points, OBJECTIVE_PROBE_WALL, Instant::now(), |p| {
        Some(f64::from(p[0]))
    })
    .expect("a varying objective is evaluable everywhere");
    assert!(!varying.is_flat());
    // Early exit: it stops at the SECOND distinct value, so a non-flat
    // objective costs ~2 forwards and never the full ladder.
    assert_eq!(varying.distinct_values, 2);
    assert!(
        varying.points <= 3,
        "a non-flat objective must exit early, took {} points",
        varying.points
    );
}

#[test]
fn a_non_finite_or_unevaluable_probe_is_inconclusive_not_flat() {
    let points = probe_points(&[0.0f32], &[1.0f32], OBJECTIVE_PROBE_POINTS);
    // `-inf` is what the margin surrogate returns for a constraint it cannot
    // express. Constant `-inf` is NOT evidence that the objective is constant.
    assert_eq!(
        probe_objective(&points, OBJECTIVE_PROBE_WALL, Instant::now(), |_| Some(
            f64::NEG_INFINITY
        )),
        None
    );
    assert_eq!(
        probe_objective(&points, OBJECTIVE_PROBE_WALL, Instant::now(), |_| None),
        None
    );
}

#[test]
fn a_probe_that_runs_out_of_wall_cannot_support_a_flat_verdict() {
    let points = probe_points(&[0.0f32], &[1.0f32], OBJECTIVE_PROBE_POINTS);
    let mut taken = 0usize;
    let probe = probe_objective(&points, Duration::from_millis(30), Instant::now(), |_| {
        taken += 1;
        if taken >= 3 {
            std::thread::sleep(Duration::from_millis(40));
        }
        Some(-1.0)
    })
    .expect("some points were evaluated");
    assert!(probe.points < OBJECTIVE_PROBE_MIN_POINTS);
    assert!(
        !probe.is_flat(),
        "a Flat verdict on {} points would zero a lane on nothing",
        probe.points
    );
}

// ---------------------------------------------------------------------------
// The plan
// ---------------------------------------------------------------------------

#[cfg(feature = "mip")]
mod plan {
    use super::*;

    /// The measured traffic timeout row, in whole seconds: the LP lane's
    /// `(480 - 45) * 0.5`, the STE lane's `remaining - 145`, and the upfront
    /// lane's `(remaining - 3) * 0.5` on a `Sign` net.
    fn measured_traffic_row() -> TodaySlices {
        TodaySlices {
            sign_space: Some(Duration::from_secs(217) + Duration::from_millis(520)),
            ste_pgd: Some(Duration::from_secs(117) + Duration::from_millis(510)),
            upfront: Some(Duration::from_secs(71)),
        }
    }

    fn cap_secs(window: LaneWindow) -> u64 {
        match window {
            LaneWindow::Cap(c) => c.as_secs(),
            LaneWindow::Skip => 0,
            LaneWindow::Private => panic!("an allocated lane must not report Private"),
        }
    }

    /// (e) A FLAT objective tier zeroes the gradient-guided lane.
    #[test]
    fn a_flat_objective_tier_zeroes_the_gradient_guided_lane() {
        let plan = plan_attack_slice(measured_traffic_row(), true, Duration::from_millis(500))
            .expect("the measured row allocates");
        assert_eq!(
            plan.window(AllocatedLane::UpfrontAttack),
            LaneWindow::Skip,
            "an exact-gradient lane on a single-valued objective is BLIND, not slow"
        );
    }

    /// (b) A lane granted 0 is SKIPPED and its seconds appear in another grant.
    #[test]
    fn a_flat_objective_skips_the_upfront_lane_and_its_seconds_move() {
        let today = measured_traffic_row();
        let informative =
            plan_attack_slice(today, false, Duration::from_millis(500)).expect("allocates");
        let flat = plan_attack_slice(today, true, Duration::from_millis(500)).expect("allocates");

        // The upfront lane keeps its cap when the objective carries signal and
        // is SKIPPED when it does not.
        assert_eq!(
            informative.window(AllocatedLane::UpfrontAttack),
            LaneWindow::Cap(Duration::from_secs(71))
        );
        assert_eq!(flat.window(AllocatedLane::UpfrontAttack), LaneWindow::Skip);

        // Its 71 s did not evaporate: STE-PGD climbs from today's 117 s cap to
        // its source-declared 240 s `max_wall_time` rung — the cap measured to
        // win three rows that 217.5 s did not — and the LP lane comes down to
        // the measured floor that its own control row licenses.
        assert_eq!(
            cap_secs(flat.window(AllocatedLane::StePgd)),
            240,
            "the freed seconds must land on a rung the receiving lane can plan against"
        );
        assert_eq!(
            cap_secs(flat.window(AllocatedLane::SignSpace)),
            LP_LANE_MEASURED_FLOOR.as_secs()
        );
        assert!(
            cap_secs(flat.window(AllocatedLane::StePgd))
                > cap_secs(informative.window(AllocatedLane::StePgd)),
            "the skipped lane's seconds must appear in another lane's grant"
        );
        // And nothing was invented: the pool is unchanged between the two arms.
        assert_eq!(flat.pool(), informative.pool());
        assert!(flat.granted() <= flat.pool());
    }

    /// I-FLOOR: with no structural zero, (C5) pins the plan at or above today.
    #[test]
    fn without_a_structural_zero_no_lane_drops_below_todays_cap() {
        let today = measured_traffic_row();
        let plan = plan_attack_slice(today, false, Duration::from_millis(500)).expect("allocates");
        assert!(cap_secs(plan.window(AllocatedLane::StePgd)) >= today.ste_pgd.unwrap().as_secs());
        assert!(
            cap_secs(plan.window(AllocatedLane::UpfrontAttack)) >= today.upfront.unwrap().as_secs()
        );
        // The LP lane carries the ONE named relaxation, to a MEASURED floor.
        assert!(
            cap_secs(plan.window(AllocatedLane::SignSpace)) >= LP_LANE_MEASURED_FLOOR.as_secs(),
            "the LP lane may never be floored below the cap its control row proves is enough"
        );
    }

    /// (c) The pool never exceeds today's attack-slice claim, and the grants
    /// never exceed the pool, on randomized inputs.
    #[test]
    fn grants_plus_reserve_never_exceed_the_budget() {
        // A deterministic LCG: a randomized-input test must be reproducible.
        let mut state: u64 = 0x2026_0819_0000_0001;
        let mut next = |hi: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % (hi + 1)
        };
        for case in 0..400u32 {
            let budget = Duration::from_secs(30 + next(1770));
            // Today's slices, drawn so their SUM is always inside the budget —
            // which is what today's chain of fixed fractions guarantees.
            let lp = next(budget.as_secs() / 2);
            let ste = next(budget.as_secs().saturating_sub(lp) / 2);
            let up = next(budget.as_secs().saturating_sub(lp + ste) / 2);
            let today = TodaySlices {
                sign_space: (lp > 0).then(|| Duration::from_secs(lp)),
                ste_pgd: (ste > 0).then(|| Duration::from_secs(ste)),
                upfront: (up > 0).then(|| Duration::from_secs(up)),
            };
            let flat = case % 3 == 0;
            let pool = today.pool();
            assert!(
                pool <= budget,
                "case {case}: the pool must never exceed the instance budget"
            );
            let Ok(plan) = plan_attack_slice(today, flat, Duration::from_millis(500)) else {
                // Fail-open is always allowed; it costs today's plan and no more.
                continue;
            };
            assert_eq!(plan.pool(), pool);
            assert!(
                plan.granted() <= plan.pool(),
                "case {case}: grants {:?} exceeded pool {:?}",
                plan.granted(),
                plan.pool()
            );
            // The reserve is everything the pool does not cover. Grants plus
            // reserve is therefore at most the budget, which is the invariant
            // that keeps a family bankable.
            let reserve = budget.saturating_sub(pool);
            assert!(
                plan.granted() + reserve <= budget,
                "case {case}: grants + reserve exceeded the instance budget"
            );
            assert_eq!(
                plan.granted() + plan.residual_to_bab(),
                plan.pool(),
                "case {case}: the residual must account for every ungranted second"
            );
        }
    }

    /// A lane the knapsack cannot represent keeps its private fraction rather
    /// than being rounded to zero and skipped.
    #[test]
    fn a_sub_second_lane_is_left_on_its_private_fraction_not_skipped() {
        let today = TodaySlices {
            sign_space: None,
            ste_pgd: Some(Duration::from_secs(30)),
            // 800 ms is `UPFRONT_ATTACK_MIN_BUDGET`: the smallest window that
            // lane will still take. It floors to zero whole seconds, so the
            // model must decline to decide for it.
            upfront: Some(Duration::from_millis(800)),
        };
        let plan = plan_attack_slice(today, true, Duration::from_millis(500)).expect("allocates");
        assert_eq!(
            plan.window(AllocatedLane::UpfrontAttack),
            LaneWindow::Private,
            "a lane the whole-second knapsack cannot represent must not be skipped by rounding"
        );
        assert_eq!(cap_secs(plan.window(AllocatedLane::StePgd)), 30);
    }

    #[test]
    fn no_lane_means_no_allocation() {
        assert_eq!(
            plan_attack_slice(TodaySlices::default(), true, Duration::from_millis(500)),
            Err(NoAllocation::NoLanes)
        );
    }

    /// (d) Fail open. A solve cap no solve can meet returns a typed reason and
    /// the caller runs exactly today's plan — which is what `LaneWindow::Private`
    /// at every call site means.
    #[test]
    fn an_impossible_solve_cap_falls_open() {
        let outcome = plan_attack_slice(measured_traffic_row(), true, Duration::from_nanos(1));
        assert!(
            matches!(outcome, Err(NoAllocation::FellOpen(_))),
            "expected a typed fall-open, got {outcome:?}"
        );
        // And the caller's behaviour under it is today's, on the code path.
        let allocation = outcome.ok();
        for lane in [
            AllocatedLane::SignSpace,
            AllocatedLane::StePgd,
            AllocatedLane::UpfrontAttack,
        ] {
            assert_eq!(lane_window(allocation.as_ref(), lane), LaneWindow::Private);
        }
    }

    /// (d) Fail open when there is nothing the whole-second knapsack can
    /// decide. Every sub-second cap floors away, the pool is empty, and the
    /// wiring returns a typed reason rather than inventing a plan.
    #[test]
    fn a_forced_allocator_failure_falls_open() {
        let today = TodaySlices {
            sign_space: Some(Duration::from_millis(400)),
            ste_pgd: None,
            upfront: None,
        };
        let outcome = plan_attack_slice(today, false, Duration::from_millis(500));
        assert_eq!(outcome, Err(NoAllocation::NoLanes));
        // And under it every call site is back on today's helper.
        for lane in [
            AllocatedLane::SignSpace,
            AllocatedLane::StePgd,
            AllocatedLane::UpfrontAttack,
        ] {
            assert_eq!(
                lane_window(outcome.as_ref().ok(), lane),
                LaneWindow::Private
            );
        }
    }

    /// The allocator is only ever allowed to move seconds INSIDE the attack
    /// slice: BaB's residual claim can never shrink.
    #[test]
    fn the_pool_never_exceeds_todays_attack_slice() {
        let today = measured_traffic_row();
        let expected = Duration::from_secs(217 + 117 + 71);
        assert_eq!(today.pool(), expected);
        for flat in [false, true] {
            let plan =
                plan_attack_slice(today, flat, Duration::from_millis(500)).expect("allocates");
            assert_eq!(plan.pool(), expected);
            assert!(plan.granted() <= expected);
        }
    }

    #[test]
    fn the_ledger_names_every_lane_and_the_residual() {
        let plan = plan_attack_slice(measured_traffic_row(), true, Duration::from_millis(500))
            .expect("allocates");
        let ledger = plan.ledger();
        for needle in [
            "pool",
            "granted",
            "residual to BaB",
            "sign_space",
            "ste_pgd",
            "upfront",
        ] {
            assert!(
                ledger.contains(needle),
                "ledger is missing {needle}: {ledger}"
            );
        }
        assert!(ledger.contains("SKIPPED"));
    }
}
