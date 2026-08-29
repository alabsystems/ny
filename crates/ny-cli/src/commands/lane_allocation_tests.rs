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
//! (f) and it zeroes NOTHING ELSE: a FLAT verdict may only reach the upfront
//!     lane when that lane actually climbs the probed objective —
//!     `a_flat_verdict_zeroes_the_upfront_lane_only_when_it_climbs_the_probed_objective`.

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
// (f) the narrowing: a FLAT verdict is not automatically evidence about the
//     upfront lane
// ---------------------------------------------------------------------------

/// The whole rule, as a truth table. Only ONE of the four shapes may zero.
#[test]
fn a_flat_verdict_zeroes_the_upfront_lane_only_when_it_climbs_the_probed_objective() {
    let climbs_the_probed_objective = UpfrontLaneShape {
        steers_on_a_surrogate_objective: false,
        enumerates_points: false,
    };
    assert!(flat_verdict_zeroes_upfront(
        true,
        climbs_the_probed_objective
    ));

    // A surrogate direction. MEASURED: on `traffic_signs_recognition_2023
    // model_30_idx_6371_eps_{3,5,10,15}` and `model_30_idx_178_eps_10` the probe
    // reports one distinct f32 value over 33 in-box points and the lane then
    // reaches an ORT-confirmed violation in 1-2 gradient steps, because on a
    // `Sign` net it differentiates `tanh(beta*z/rms)` and not the probed margin.
    // Zeroing it there removed the lane that WINS those five banked rows.
    assert!(!flat_verdict_zeroes_upfront(
        true,
        UpfrontLaneShape {
            steers_on_a_surrogate_objective: true,
            enumerates_points: false,
        }
    ));
    // An enumerating sub-lane. The probe takes 2 of the 32 corners of a
    // 5-dimensional box; `low_dim_ort_corner_falsify` visits all 32. A sample of
    // an enumeration's domain is not evidence against the enumeration.
    assert!(!flat_verdict_zeroes_upfront(
        true,
        UpfrontLaneShape {
            steers_on_a_surrogate_objective: false,
            enumerates_points: true,
        }
    ));
    assert!(!flat_verdict_zeroes_upfront(
        true,
        UpfrontLaneShape {
            steers_on_a_surrogate_objective: true,
            enumerates_points: true,
        }
    ));
}

/// No probe verdict, no zero — whatever the shape says.
#[test]
fn an_informative_or_unestablished_probe_never_zeroes_whatever_the_shape() {
    for surrogate in [false, true] {
        for enumerates in [false, true] {
            assert!(!flat_verdict_zeroes_upfront(
                false,
                UpfrontLaneShape {
                    steers_on_a_surrogate_objective: surrogate,
                    enumerates_points: enumerates,
                }
            ));
        }
    }
}

/// Exactly ONE of the four shapes may zero, so the type must not be
/// default-constructible: a `Default` derive would produce `false, false`,
/// which is precisely the zeroing arm, and an unread shape would silently
/// remove a lane. The caller therefore builds a shape only inside the branch
/// that has a verdict to narrow. This test pins that count.
#[test]
fn exactly_one_shape_of_four_may_zero_the_lane() {
    let zeroing: Vec<UpfrontLaneShape> = [false, true]
        .into_iter()
        .flat_map(|surrogate| {
            [false, true]
                .into_iter()
                .map(move |enumerates| UpfrontLaneShape {
                    steers_on_a_surrogate_objective: surrogate,
                    enumerates_points: enumerates,
                })
        })
        .filter(|&shape| flat_verdict_zeroes_upfront(true, shape))
        .collect();
    assert_eq!(
        zeroing,
        vec![UpfrontLaneShape {
            steers_on_a_surrogate_objective: false,
            enumerates_points: false,
        }],
        "only a lane that climbs the probed objective itself may be zeroed by a probe of it"
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
        assert_eq!(plan.authority(), AllocationAuthority::Knapsack);
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

// ---------------------------------------------------------------------------
// #lane-budget-frame — freed seconds must reach the INTENDED claimant
// ---------------------------------------------------------------------------
//
// (g) today's sizing rules are transcribed EXACTLY, machine-checked against the
//     production helpers — `todays_rules_are_transcribed_exactly`.
// (h) the mixture, named as arithmetic — `the_mixture_each_lane_absorbs_a_
//     different_share_of_a_freed_second`.
// (i) a lane that declines does NOT grow a downstream dynamic lane —
//     `a_declining_lane_no_longer_grows_the_downstream_dynamic_lane`.
// (j) and the intended recipient gets those seconds —
//     `the_freed_seconds_reach_the_residual_claimant_in_full`.
// (k) grants + reserves never exceed the instance budget, swept —
//     `framed_grants_plus_the_reserve_never_exceed_the_budget`.
// (l) defaults unchanged with the lever unset — `the_frame_lever_is_dark_by_
//     default` and `the_frame_is_the_weakest_cap_authority`.

/// The upfront lane's rule on a `Sign` net, at its shipped defaults.
///
/// `0.5`/`240 s` and `0.08`/`8 s` are the frozen declaration defaults in
/// `UpfrontAttackPolicy`. `the_upfront_sign_net_rule_is_transcribed_exactly` in
/// `vnncomp`'s own test module checks this against that lane's production branch,
/// including non-default governed values and an unbounded wall.
fn upfront_rule() -> SizingRule {
    upfront_sign_net_rule(0.5, Duration::from_mins(4), 0.08, Duration::from_secs(8))
}

/// The chain the whole attack slice is planned by, as pure arithmetic.
///
/// The same subtraction-of-caps `todays_attack_slice` performs
/// (`vnncomp.rs`), with `upfront_today_cap`'s `Sign`-net branch modelled by
/// [`upfront_rule`] so the chain needs no ONNX file. Structural admission
/// (route enabled, property admitted) is out of scope here and is what the
/// production mirror adds on top.
fn todays_slices(budget: Duration) -> TodaySlices {
    let sign_space = LP_LANE_RULE.window(Some(budget));
    let after_lp = budget.saturating_sub(sign_space.unwrap_or_default());
    let ste_pgd = STE_LANE_RULE.window(Some(after_lp));
    let after_ste = after_lp.saturating_sub(ste_pgd.unwrap_or_default());
    let upfront = upfront_rule().window(Some(after_ste));
    TodaySlices {
        sign_space,
        ste_pgd,
        upfront,
    }
}

/// What one pass over the attack slice produced.
#[derive(Debug, Clone, Copy, PartialEq)]
struct SliceRun {
    /// The WINDOW each lane was given, in slice order.
    windows: [Duration; 3],
    /// What reached the branch-and-bound residual claimant.
    to_residual: Duration,
}

/// TODAY: each lane re-derives its own window from the LIVE clock at the moment
/// it starts. `declines` marks a lane that is consulted and spends nothing.
fn live_clock_run(budget: Duration, declines: [bool; 3]) -> SliceRun {
    let rules = [LP_LANE_RULE, STE_LANE_RULE, upfront_rule()];
    let mut live = budget;
    let mut windows = [Duration::ZERO; 3];
    for lane in 0..3 {
        let window = rules[lane].window(Some(live)).unwrap_or_default();
        windows[lane] = window;
        if !declines[lane] {
            live = live.saturating_sub(window);
        }
    }
    SliceRun {
        windows,
        to_residual: live,
    }
}

/// #lane-budget-frame: every window is committed up front, from the SAME rules,
/// against ONE reference clock reading.
fn framed_run(budget: Duration, declines: [bool; 3]) -> SliceRun {
    let frame = AttackSliceAllocation::frame(todays_slices(budget))
        .expect("a 30 s+ budget consults at least one lane");
    let mut live = budget;
    let mut windows = [Duration::ZERO; 3];
    for (lane, id) in [
        AllocatedLane::SignSpace,
        AllocatedLane::StePgd,
        AllocatedLane::UpfrontAttack,
    ]
    .into_iter()
    .enumerate()
    {
        // Exactly what the call sites do: a committed cap, clamped DOWN to the
        // live remaining budget, and a `Skip` is not consulted at all.
        let window = match frame.window(id) {
            LaneWindow::Skip => Duration::ZERO,
            LaneWindow::Private => unreachable!("a frame commits every lane"),
            LaneWindow::Cap(cap) => cap.min(live),
        };
        windows[lane] = window;
        if !declines[lane] {
            live = live.saturating_sub(window);
        }
    }
    SliceRun {
        windows,
        to_residual: live,
    }
}

/// (g) The rules this module states as data ARE the rules the lanes run.
///
/// Checked against the production helpers themselves — `lp_lane_plan` and
/// `ste_lane_plan` are the very functions `run_sign_space_lane` and
/// `run_ste_pgd_lane` call — so the table in [`SizingRule`]'s doc comment
/// cannot drift from the code it describes.
#[cfg(feature = "mip")]
#[test]
fn todays_rules_are_transcribed_exactly() {
    use crate::commands::beta_crown::sign_space_falsify::{lp_lane_plan, ste_lane_plan};

    assert_eq!(LP_LANE_RULE.window(None), lp_lane_plan(None));
    assert_eq!(STE_LANE_RULE.window(None), ste_lane_plan(None));
    for secs in [
        0u64, 1, 20, 44, 45, 46, 65, 85, 100, 145, 146, 200, 480, 525, 1800, 3600,
    ] {
        for millis in [0u64, 1, 500, 999] {
            let remaining = Some(Duration::from_secs(secs) + Duration::from_millis(millis));
            assert_eq!(
                LP_LANE_RULE.window(remaining),
                lp_lane_plan(remaining),
                "LP lane at {remaining:?}"
            );
            assert_eq!(
                STE_LANE_RULE.window(remaining),
                ste_lane_plan(remaining),
                "STE lane at {remaining:?}"
            );
        }
    }
}

/// (h) THE BUG, NAMED. Every lane reads the live clock, but each absorbs a
/// DIFFERENT share of a second freed ahead of it — a half, all of it, or none.
/// That mixture is why "free budget by removing a lane" does not deliver it.
#[test]
fn the_mixture_each_lane_absorbs_a_different_share_of_a_freed_second() {
    let base = Duration::from_secs(150);
    let freed = Duration::from_secs(100);

    // Fraction-of-live: half of every freed second, by construction.
    assert_eq!(LP_LANE_RULE.absorbed(base, freed), Duration::from_secs(50));
    assert_eq!(
        upfront_rule().absorbed(base, freed),
        Duration::from_secs(50)
    );
    // Subtraction: ALL of every freed second, until the 4-minute wall binds.
    assert_eq!(
        STE_LANE_RULE.absorbed(Duration::from_secs(200), freed),
        freed
    );
    assert_eq!(
        STE_LANE_RULE.absorbed(Duration::from_secs(400), freed),
        // 400-145 = 255 -> capped 240; 500-145 = 355 -> capped 240. Wall binds.
        Duration::ZERO
    );
    // And WORSE than one-for-one across its 20 s floor: a lane that today's
    // chain declines outright takes a 105 s window off 100 freed seconds,
    // because the floor is a step and the rule reads the live clock.
    assert_eq!(STE_LANE_RULE.window(Some(base)), None);
    assert_eq!(
        STE_LANE_RULE.absorbed(base, freed),
        Duration::from_secs(105)
    );

    // THE FOUNDING OBSERVATION, reproduced as arithmetic. A 480 s traffic row
    // with ~478 s left when the slice starts: disabling the two BNN lanes frees
    // 333.0 s, and the upfront lane grows ITSELF by half of it.
    let budget = Duration::from_secs(478);
    let baseline = live_clock_run(budget, [false, false, false]);
    let bnn_disabled = live_clock_run(budget, [true, true, false]);
    let freed_upstream = baseline.windows[0] + baseline.windows[1];
    let absorbed = bnn_disabled.windows[2].saturating_sub(baseline.windows[2]);
    assert_eq!(absorbed, freed_upstream.mul_f64(0.5));
    assert!(
        bnn_disabled
            .to_residual
            .saturating_sub(baseline.to_residual)
            < freed_upstream,
        "today, the residual claimant does NOT receive the freed seconds in full"
    );
}

/// (i) THE FIX. With the frame in force, a lane that declines does not grow the
/// downstream dynamic lane: its window is identical whether or not the two
/// lanes ahead of it spent anything.
#[test]
fn a_declining_lane_no_longer_grows_the_downstream_dynamic_lane() {
    for secs in [100u64, 240, 300, 478, 480, 600, 1800] {
        let budget = Duration::from_secs(secs);
        let baseline = framed_run(budget, [false, false, false]);
        let bnn_disabled = framed_run(budget, [true, true, false]);
        assert_eq!(
            baseline.windows, bnn_disabled.windows,
            "framed windows moved at a {secs} s budget when the BNN lanes declined"
        );
        // And the control: today, at this same budget, it DOES grow.
        let today_baseline = live_clock_run(budget, [false, false, false]);
        let today_disabled = live_clock_run(budget, [true, true, false]);
        // The control reproduces the defect wherever the upfront lane's own
        // 4-minute wall is not already binding — at a 1800 s budget every lane
        // is capped out and there is nothing left to absorb.
        let upfront_wall = Duration::from_mins(4);
        if today_baseline.windows[0] + today_baseline.windows[1] > Duration::ZERO
            && today_baseline.windows[2] < upfront_wall
        {
            assert!(
                today_disabled.windows[2] > today_baseline.windows[2],
                "the control must reproduce the defect at a {secs} s budget"
            );
        }
    }
}

/// (j) And the seconds reach the lane the plan INTENDED them for: the
/// branch-and-bound residual claimant, in full, not a fraction of it.
#[test]
fn the_freed_seconds_reach_the_residual_claimant_in_full() {
    let budget = Duration::from_secs(478);
    let baseline = framed_run(budget, [false, false, false]);
    let bnn_disabled = framed_run(budget, [true, true, false]);
    let freed = baseline.windows[0] + baseline.windows[1];
    assert!(freed > Duration::from_mins(5), "freed {freed:?}");
    assert_eq!(
        bnn_disabled
            .to_residual
            .saturating_sub(baseline.to_residual),
        freed,
        "every freed second must reach the residual claimant"
    );

    // Against today, on the same row: the frame delivers strictly more.
    let today_disabled = live_clock_run(budget, [true, true, false]);
    assert!(bnn_disabled.to_residual > today_disabled.to_residual);
}

/// (k) Grants plus the reserve can never exceed the instance budget.
///
/// 3 s is the tightest reserve any committed lane holds — the upfront lane's
/// `UPFRONT_ATTACK_SAFETY_MARGIN`; the two BNN lanes hold 45 s. The bound is
/// exact rather than incidental: every rule takes at most
/// `live_at_its_start - its own margin`, so the running sum can never reach the
/// last margin.
#[test]
fn framed_grants_plus_the_reserve_never_exceed_the_budget() {
    let tightest_reserve = Duration::from_secs(3);
    for secs in (10u64..=1_800).step_by(7) {
        let budget = Duration::from_secs(secs);
        let Some(frame) = AttackSliceAllocation::frame(todays_slices(budget)) else {
            continue;
        };
        assert!(
            frame.granted() + tightest_reserve <= budget,
            "granted {:?} + reserve exceeds the {secs} s budget",
            frame.granted()
        );
        // The frame REDISTRIBUTES NOTHING, so the pool is the grants and there
        // is no planned residual: the residual is whatever the lanes decline at
        // run time, which is the whole point.
        assert_eq!(frame.granted(), frame.pool());
        assert_eq!(frame.residual_to_bab(), Duration::ZERO);
        // And it never claims a second today's own chain does not claim.
        assert!(frame.granted() <= todays_slices(budget).pool() + Duration::from_secs(3));
    }
}

/// (i, continued) THE HOLE A `Private` WINDOW WOULD LEAVE. A lane today's chain
/// declines must be SKIPPED, not sent back to its own live-clock helper — that
/// helper would hand it a full window the moment an upstream lane freed the
/// seconds, which is the absorption itself.
#[test]
fn a_lane_todays_chain_declines_is_skipped_never_left_private() {
    // A 100 s instance: the LP lane fits, the STE lane's subtraction underflows.
    let today = todays_slices(Duration::from_secs(100));
    assert!(today.sign_space.is_some());
    assert!(today.ste_pgd.is_none(), "the STE lane declines at 100 s");
    let frame = AttackSliceAllocation::frame(today).expect("the LP lane is consulted");
    assert_eq!(frame.window(AllocatedLane::StePgd), LaneWindow::Skip);
    assert!(matches!(
        frame.window(AllocatedLane::SignSpace),
        LaneWindow::Cap(_)
    ));
    // And the counterfactual the `Skip` closes: left to its own helper, that
    // lane would take a 4-minute window off a freed clock.
    assert_eq!(
        STE_LANE_RULE.window(Some(Duration::from_secs(72) + Duration::from_millis(500))),
        None,
        "declines against the CHAINED clock"
    );
    assert_eq!(
        STE_LANE_RULE.window(Some(Duration::from_secs(400))),
        Some(Duration::from_mins(4)),
        "but takes its full wall off a clock an upstream decline left ahead"
    );
}

/// A frame is only built when some lane is consulted; otherwise the caller
/// falls open to today's unframed run.
#[test]
fn no_lane_no_frame() {
    assert_eq!(AttackSliceAllocation::frame(TodaySlices::default()), None);
}

/// (l) Dark by default, and only exact `"1"` arms it.
#[test]
fn the_frame_lever_is_dark_by_default() {
    assert!(!lane_budget_frame_armed_from(None));
    assert!(!lane_budget_frame_armed_from(Some("0")));
    assert!(!lane_budget_frame_armed_from(Some("true")));
    assert!(!lane_budget_frame_armed_from(Some(" 1")));
    assert!(lane_budget_frame_armed_from(Some("1")));
}

/// (l) THE byte-identical assertion for this lever, on the decision function:
/// unset, no frame is admitted under ANY combination of the other authorities,
/// so every lane's window is `Private` and each takes its own helper.
#[test]
fn the_frame_is_the_weakest_cap_authority() {
    for allocator in [false, true] {
        for scheduler in [false, true] {
            for peel in [false, true] {
                assert!(
                    !frame_admitted(false, allocator, scheduler, peel),
                    "disarmed must dominate"
                );
            }
        }
    }
    // Armed, it stands down under either of the stronger authorities and under
    // the traffic peel; exactly one mechanism holds a lane's cap at a time.
    assert!(!frame_admitted(true, true, false, false));
    assert!(!frame_admitted(true, false, true, false));
    assert!(!frame_admitted(true, false, false, true));
    assert!(frame_admitted(true, false, false, false));
}

/// A skipped lane's receipt must name the authority that actually zeroed it.
#[test]
fn a_framed_zero_is_not_reported_as_the_knapsacks_structural_zero() {
    let frame = AttackSliceAllocation::frame(todays_slices(Duration::from_secs(100)))
        .expect("the LP lane is consulted");
    assert_eq!(frame.authority(), AllocationAuthority::Frame);
    assert_eq!(frame.zero_reason(), FRAME_ZERO_REASON);
    assert_ne!(frame.zero_reason(), KNAPSACK_ZERO_REASON);
    assert!(frame.ledger().contains("lane budget frame"));
}
