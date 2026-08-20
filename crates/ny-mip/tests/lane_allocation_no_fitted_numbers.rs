// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Two literals in the module are judgement calls rather than source citations
//! or sizing bounds. This file settles both BY EXECUTION.
use ny_mip::{
    allocate, AllocationOutcome, AllocationRequest, CapLadder, LaneRequest, ObjectiveRequirement,
    ObjectiveTier, Rung, RungOrigin, StructuralZero, DEFAULT_LANE_REACH_PRIOR,
};
use std::time::Duration;

fn ladder(points: &[(u64, f64)]) -> CapLadder {
    let mut rungs = vec![Rung {
        cap: Duration::ZERO,
        reach: 0.0,
        origin: RungOrigin::DoNotRun,
    }];
    for &(s, r) in points {
        rungs.push(Rung {
            cap: Duration::from_secs(s),
            reach: r,
            origin: RungOrigin::CallerSupplied,
        });
    }
    CapLadder::caller_supplied(rungs).unwrap()
}

fn caps(o: &AllocationOutcome) -> Vec<u64> {
    match o {
        AllocationOutcome::Allocated(p) => p.grants().iter().map(|g| g.cap.as_secs()).collect(),
        other => panic!("{other:?}"),
    }
}

/// The `2..=31` boundary in `ObjectiveTier` is the one threshold-looking literal
/// in the module. It is INERT: only `Flat` carries a structural zero, so no
/// allocation can depend on where `Staircase` and `Smooth` meet.
#[test]
fn the_staircase_smooth_boundary_cannot_change_any_allocation() {
    // no n_distinct >= 2 produces a zero, for any requirement
    for n in 2..200usize {
        for req in [
            ObjectiveRequirement::Exact,
            ObjectiveRequirement::EstimatedGradient,
            ObjectiveRequirement::ValueOnly,
        ] {
            assert!(
                StructuralZero::flat_objective(n, req).is_none(),
                "n_distinct={n} {req:?} produced a structural zero"
            );
        }
        // and the classification either side of 31 is never Flat
        assert_ne!(
            ObjectiveTier::from_distinct_objective_values(n),
            ObjectiveTier::Flat
        );
    }
    // the ONLY boundary that carries a zero is a count, not a cut
    assert_eq!(
        ObjectiveTier::from_distinct_objective_values(0),
        ObjectiveTier::Flat
    );
    assert_eq!(
        ObjectiveTier::from_distinct_objective_values(1),
        ObjectiveTier::Flat
    );
    assert!(StructuralZero::flat_objective(1, ObjectiveRequirement::ValueOnly).is_none());
    assert!(StructuralZero::flat_objective(1, ObjectiveRequirement::EstimatedGradient).is_some());
    // 31 vs 32 is a NAME change and nothing else: same allocation either side
    let mut plans = Vec::new();
    for n in [2usize, 31, 32, 4096] {
        let z = StructuralZero::flat_objective(n, ObjectiveRequirement::EstimatedGradient);
        assert!(z.is_none(), "n={n}");
        let req = AllocationRequest::new(
            vec![
                LaneRequest::new(ladder(&[(100, 0.9)])),
                LaneRequest::new(ladder(&[(100, 0.4)])),
            ],
            Duration::from_secs(150),
            Duration::ZERO,
        )
        .with_solve_cap(Duration::from_secs(2));
        plans.push(caps(&allocate(&req)));
    }
    assert!(
        plans.windows(2).all(|w| w[0] == w[1]),
        "tier boundary moved a plan: {plans:?}"
    );
}

/// The uniform prior is NOT inert. It cannot express a preference BETWEEN
/// lanes (it is the same scalar for all of them), but it does set how strongly
/// the objective saturates, and therefore how much the knapsack prefers
/// spreading budget over concentrating it. Here is a concrete flip.
#[test]
fn the_uniform_prior_is_a_real_knob_even_though_it_cannot_rank_lanes() {
    let build = |a: f64| {
        AllocationRequest::new(
            vec![
                LaneRequest::new(ladder(&[(10, 0.90)])).with_reach_prior(a),
                LaneRequest::new(ladder(&[(10, 0.50), (20, 0.95)])).with_reach_prior(a),
            ],
            Duration::from_secs(20),
            Duration::ZERO,
        )
        .with_solve_cap(Duration::from_secs(2))
    };
    let concentrated = caps(&allocate(&build(1.0)));
    let spread = caps(&allocate(&build(0.5)));
    eprintln!("  prior a=1.0 -> caps {concentrated:?}   (concentrate in one lane)");
    eprintln!("  prior a=0.5 -> caps {spread:?}   (spread across two lanes)");
    assert_ne!(
        concentrated, spread,
        "expected the prior to move the plan; if this ever stops holding the \
         module's claim that the prior is inert becomes true and this test can go"
    );
    assert_eq!(concentrated, vec![0, 20]);
    assert_eq!(spread, vec![10, 10]);
    // The default is the Beta(1,1) mean, and nothing in the crate overrides it.
    assert_eq!(DEFAULT_LANE_REACH_PRIOR, 0.5);
    assert_eq!(caps(&allocate(&build(DEFAULT_LANE_REACH_PRIOR))), spread);
}

/// Whatever the prior is, it is the SAME for every lane by default, so it
/// cannot rank them: swapping two lanes' ladders swaps their grants exactly.
#[test]
fn the_default_prior_is_symmetric_across_lanes() {
    let go = |first: &[(u64, f64)], second: &[(u64, f64)]| {
        caps(&allocate(
            &AllocationRequest::new(
                vec![
                    LaneRequest::new(ladder(first)),
                    LaneRequest::new(ladder(second)),
                ],
                Duration::from_mins(2),
                Duration::ZERO,
            )
            .with_solve_cap(Duration::from_secs(2)),
        ))
    };
    let a = &[(30u64, 0.4), (90, 0.8)][..];
    let b = &[(40u64, 0.7)][..];
    let mut swapped = go(b, a);
    swapped.reverse();
    assert_eq!(
        go(a, b),
        swapped,
        "the default prior ranked the lanes by position"
    );
}

/// `declared_ladder` builds its two source-read ladders through a VALIDATING
/// constructor and `expect`s success. Prove those two `expect`s unreachable
/// across the whole budget range a board can present (the 2026 board spans
/// 30..1800 s) plus the saturating extremes.
// Board budgets are quoted in SECONDS because that is how the 2026 board and
// every `instances.csv` state them; rewriting 480 s as 8 min would obscure the
// thing being tested.
#[allow(clippy::duration_suboptimal_units)]
#[test]
fn the_source_declared_ladders_never_hit_their_expect() {
    use ny_mip::{declared_ladder, LadderProvenance, Lane};
    let mut budgets: Vec<Duration> = (0..2000u64).map(Duration::from_secs).collect();
    budgets.extend([
        Duration::ZERO,
        Duration::from_millis(31_999),
        Duration::from_millis(32_001),
        Duration::from_millis(299_999),
        Duration::from_secs(1800),
        Duration::from_secs(u64::from(u32::MAX)),
        Duration::MAX,
    ]);
    for b in budgets {
        for lane in [Lane::BnnSignSpace, Lane::BnnStePgd] {
            let l = declared_ladder(lane, b).expect("these two lanes have source ladders");
            assert!(matches!(
                l.provenance(),
                LadderProvenance::ReadFromSource { .. }
            ));
            let rungs = l.rungs();
            assert_eq!(rungs[0].cap, Duration::ZERO);
            assert_eq!(rungs[0].reach, 0.0);
            for w in rungs.windows(2) {
                assert!(w[1].cap > w[0].cap, "caps not increasing at {b:?}");
                assert!(w[1].reach >= w[0].reach, "reach not monotone at {b:?}");
            }
            for r in rungs {
                assert_eq!(r.cap.subsec_nanos(), 0, "fractional rung at {b:?}");
                assert!(
                    (0.0..=1.0).contains(&r.reach),
                    "reach out of range at {b:?}"
                );
            }
            // no rung may exceed the budget it was built for
            assert!(
                rungs.last().unwrap().cap <= b.max(Duration::from_secs(b.as_secs())),
                "ladder for {lane:?} exceeds its budget at {b:?}"
            );
        }
    }
    // the five lanes with no source ladder must SAY SO, not guess one
    for lane in [
        Lane::UpfrontAttack,
        Lane::FalsifyPortfolio,
        Lane::MarginRowConcurrent,
        Lane::ForwardLinearAdmission,
        Lane::PostBabFrontier,
    ] {
        assert!(
            declared_ladder(lane, Duration::from_secs(480)).is_err(),
            "{lane:?} invented a ladder"
        );
    }
}
