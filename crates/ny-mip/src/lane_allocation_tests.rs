// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for the Layer A lane budget allocator.
//!
//! The load-bearing one is [`milp_matches_brute_force_on_random_instances`]:
//! it proves the log-linearisation and every constraint by comparing the MILP
//! against an independent enumeration of every feasible assignment.

use super::*;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Deterministic SplitMix64, local so the tests depend on nothing external.
struct Rng(u64);

impl Rng {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }
    fn unit(&mut self) -> f64 {
        (self.below(1_000_001) as f64) / 1_000_000.0
    }
}

/// Solve cap for the CORRECTNESS tests.
///
/// `ALLOC_SOLVE_CAP` (10 ms) is a scheduling parameter, not part of any
/// property being proved here: optimality, the structural zeros, precedence and
/// the pool bound are all statements about the MODEL, and a solve that is
/// abandoned mid-search states nothing about any of them. `cargo test` builds
/// unoptimised, the backend is exact-rational, and this host throttles, so the
/// correctness tests give the solve room and the 10 ms production default is
/// exercised where it belongs: in the fail-open test, and measured against a
/// realistic instance by the microbenchmark.
const TEST_SOLVE_CAP: Duration = Duration::from_secs(2);

/// A ladder from explicit `(seconds, reach)` pairs, rung 0 implied.
fn ladder(points: &[(u64, f64)]) -> CapLadder {
    let mut rungs = vec![Rung {
        cap: Duration::ZERO,
        reach: 0.0,
        origin: RungOrigin::DoNotRun,
    }];
    for &(secs, reach) in points {
        rungs.push(Rung {
            cap: Duration::from_secs(secs),
            reach,
            origin: RungOrigin::CallerSupplied,
        });
    }
    CapLadder::caller_supplied(rungs).expect("test ladder is well formed")
}

/// A lane whose ladder names `p_k` at each rung DIRECTLY, by setting the prior
/// scalar to 1 so `p = a * s = s`. Synthetic: it exists so a test can state the
/// step-like curve it wants to exercise. No shipped path sets a prior here.
fn lane_with_probabilities(points: &[(u64, f64)]) -> LaneRequest {
    LaneRequest::new(ladder(points)).with_reach_prior(1.0)
}

fn plan_of(outcome: AllocationOutcome) -> LanePlan {
    match outcome {
        AllocationOutcome::Allocated(plan) => plan,
        AllocationOutcome::UseExistingPlan(reason) => {
            panic!("expected a plan, fell open with {reason:?}")
        }
    }
}

/// Independent brute force: enumerate EVERY assignment, keep the feasible ones,
/// return the best objective. Shares no code with the MILP construction.
fn brute_force(request: &AllocationRequest) -> Option<(f64, Vec<usize>)> {
    let lanes = request.lanes();
    let pool = u128::from(request.pool().as_secs());
    let sizes: Vec<usize> = lanes.iter().map(|l| l.ladder().rungs().len()).collect();
    let total: usize = sizes.iter().product();
    let mut best: Option<(f64, Vec<usize>)> = None;
    for code in 0..total {
        let mut rest = code;
        let mut pick = Vec::with_capacity(sizes.len());
        for &size in &sizes {
            pick.push(rest % size);
            rest /= size;
        }
        // (A) structural zeros
        if lanes
            .iter()
            .zip(&pick)
            .any(|(lane, &j)| lane.structural_zero().is_some() && j != 0)
        {
            continue;
        }
        // (B) the pool
        let spent: u128 = lanes
            .iter()
            .zip(&pick)
            .map(|(lane, &j)| u128::from(lane.ladder().rungs()[j].cap.as_secs()))
            .sum();
        if spent > pool {
            continue;
        }
        // (P) precedence
        if lanes.iter().enumerate().any(|(k, lane)| {
            lane.requires
                .is_some_and(|pre| pick[k] > 0 && pick[pre] == 0)
        }) {
            continue;
        }
        // (C5) no-regression
        if lanes.iter().enumerate().any(|(k, lane)| {
            lane.structural_zero().is_none()
                && lane
                    .no_regression_floor
                    .is_some_and(|floor| lane.ladder().rungs()[pick[k]].cap < floor)
        }) {
            continue;
        }
        let value: f64 = lanes
            .iter()
            .zip(&pick)
            .map(|(lane, &j)| lane.log_miss_cost_at(j))
            .sum();
        if best.as_ref().is_none_or(|(b, _)| value < *b - 1e-12) {
            best = Some((value, pick));
        }
    }
    best
}

/// The greedy marginal-value rule the formulation says is wrong: start every
/// lane at rung 0, then repeatedly take the affordable single-lane upgrade with
/// the best value gained per second spent, until nothing improves.
fn greedy_marginal_value(request: &AllocationRequest) -> Vec<usize> {
    let lanes = request.lanes();
    let pool = u128::from(request.pool().as_secs());
    let mut pick = vec![0usize; lanes.len()];
    loop {
        let spent: u128 = lanes
            .iter()
            .zip(&pick)
            .map(|(lane, &j)| u128::from(lane.ladder().rungs()[j].cap.as_secs()))
            .sum();
        let mut best: Option<(f64, usize, usize)> = None;
        for (k, lane) in lanes.iter().enumerate() {
            if lane.structural_zero().is_some() {
                continue;
            }
            let here = &lane.ladder().rungs()[pick[k]];
            for (j, rung) in lane.ladder().rungs().iter().enumerate().skip(pick[k] + 1) {
                let extra = u128::from(rung.cap.as_secs() - here.cap.as_secs());
                if spent + extra > pool {
                    continue;
                }
                let gain = lane.log_miss_cost_at(pick[k]) - lane.log_miss_cost_at(j);
                if gain <= 0.0 || extra == 0 {
                    continue;
                }
                let ratio = gain / (extra as f64);
                if best.as_ref().is_none_or(|(b, _, _)| ratio > *b) {
                    best = Some((ratio, k, j));
                }
            }
        }
        match best {
            Some((_, k, j)) => pick[k] = j,
            None => return pick,
        }
    }
}

fn objective_of(request: &AllocationRequest, pick: &[usize]) -> f64 {
    request
        .lanes()
        .iter()
        .zip(pick)
        .map(|(lane, &j)| lane.log_miss_cost_at(j))
        .sum()
}

// ---------------------------------------------------------------------------
// (a) OPTIMALITY -- the test that matters
// ---------------------------------------------------------------------------

#[test]
fn milp_matches_brute_force_on_random_instances() {
    let mut rng = Rng::new(0x1AE0_A11C_0DE0_0001);
    let mut solved = 0usize;
    let mut fell_open = 0usize;
    for case in 0..120u64 {
        let k = 1 + rng.below(5) as usize;
        let mut lanes = Vec::with_capacity(k);
        for lane_index in 0..k {
            let rungs = 1 + rng.below(5) as usize;
            let mut points = Vec::with_capacity(rungs);
            let mut secs = 0u64;
            let mut reach = 0.0f64;
            for _ in 0..rungs {
                secs += 1 + rng.below(40);
                reach = (reach + rng.unit() * 0.4).min(1.0);
                points.push((secs, reach));
            }
            let mut lane = LaneRequest::new(ladder(&points));
            // A structural zero on roughly one lane in six.
            if rng.below(6) == 0 {
                lane = lane.zeroed(StructuralZero::StructurallyDeclined);
            }
            // A precedence edge on roughly one lane in five.
            if lane_index > 0 && rng.below(5) == 0 {
                lane = lane.requiring(rng.below(lane_index as u64) as usize);
            }
            // A (C5) no-regression floor on roughly one lane in four, drawn
            // from the lane's own rungs. Tight enough that some instances come
            // out genuinely infeasible, which exercises the fail-open arm.
            if rng.below(4) == 0 {
                let pick = rng.below(points.len() as u64) as usize;
                lane = lane.no_worse_than(Duration::from_secs(points[pick].0));
            }
            lanes.push(lane);
        }
        let budget = Duration::from_secs(20 + rng.below(120));
        let reserve = Duration::from_secs(rng.below(15));
        let request = AllocationRequest::new(lanes, budget, reserve).with_solve_cap(TEST_SOLVE_CAP);

        let expected = brute_force(&request);
        let outcome = allocate(&request);
        match (&outcome, expected) {
            (AllocationOutcome::Allocated(plan), Some((best, _))) => {
                solved += 1;
                assert!(
                    (plan.log_miss_total() - best).abs() < 1e-9,
                    "case {case}: MILP objective {} != brute force {best}",
                    plan.log_miss_total()
                );
                // and the plan the MILP returned is itself feasible+optimal
                let pick: Vec<usize> = plan.grants().iter().map(|g| g.rung).collect();
                assert!(
                    (objective_of(&request, &pick) - best).abs() < 1e-9,
                    "case {case}: the returned assignment does not achieve the optimum"
                );
            }
            (AllocationOutcome::UseExistingPlan(FallOpen::Infeasible), None) => {
                fell_open += 1;
            }
            (AllocationOutcome::UseExistingPlan(reason), expected) => {
                panic!("case {case}: fell open with {reason:?} while brute force said {expected:?}")
            }
            (AllocationOutcome::Allocated(_), None) => {
                panic!("case {case}: MILP produced a plan where brute force found none")
            }
        }
    }
    assert!(solved > 60, "only {solved} of 120 cases produced a plan");
    assert!(
        fell_open > 0,
        "no case was infeasible: the fail-open arm of this test never ran"
    );
    println!("optimality: {solved} solved, {fell_open} provably infeasible");
}

// ---------------------------------------------------------------------------
// (b) STEP-LIKE CURVES
// ---------------------------------------------------------------------------

#[test]
fn greedy_marginal_value_misses_a_late_step() {
    // Lane 0 is the step-like one: nearly worthless at 20 s, jumps at 100 s,
    // FLAT above. Lane 1 is a cheap high-ratio nibble that greedy grabs first
    // and which then leaves too little in the pool for lane 0's step.
    let stepped = lane_with_probabilities(&[(20, 0.02), (100, 0.35), (120, 0.35)]);
    let nibble = lane_with_probabilities(&[(25, 0.12)]);
    let request = AllocationRequest::new(
        vec![stepped, nibble],
        Duration::from_mins(2),
        Duration::ZERO,
    )
    .with_solve_cap(TEST_SOLVE_CAP);

    let greedy = greedy_marginal_value(&request);
    let outcome = allocate(&request);
    let plan = plan_of(outcome);
    let optimal: Vec<usize> = plan.grants().iter().map(|g| g.rung).collect();

    let greedy_value = objective_of(&request, &greedy);
    let optimal_value = objective_of(&request, &optimal);

    // The difference is demonstrated, not asserted: both rules ran.
    println!(
        "greedy picked rungs {greedy:?} (caps {:?}) for {greedy_value:.6}; \
         the allocator picked {optimal:?} (caps {:?}) for {optimal_value:.6}",
        greedy
            .iter()
            .enumerate()
            .map(|(k, &j)| request.lanes()[k].ladder().rungs()[j].cap.as_secs())
            .collect::<Vec<_>>(),
        optimal
            .iter()
            .enumerate()
            .map(|(k, &j)| request.lanes()[k].ladder().rungs()[j].cap.as_secs())
            .collect::<Vec<_>>(),
    );

    assert_eq!(greedy[0], 1, "greedy climbs only the first, cheap rung");
    assert!(
        optimal[0] >= 2,
        "the allocator must buy the rung ABOVE the step, got rung {}",
        optimal[0]
    );
    assert!(
        plan.grants()[0].cap >= Duration::from_secs(100),
        "the allocator's cap for the stepped lane is {:?}",
        plan.grants()[0].cap
    );
    assert!(
        optimal_value < greedy_value - 1e-9,
        "the allocator must strictly beat greedy: {optimal_value} vs {greedy_value}"
    );
    let (best, _) = brute_force(&request).expect("feasible");
    assert!((optimal_value - best).abs() < 1e-9);
}

#[test]
fn a_flat_top_of_the_ladder_buys_the_cheapest_rung_that_reaches_it() {
    // A lane measured flat in b: every nonzero rung has the same reach, so the
    // knapsack must buy the cheapest one and RETURN the rest of the pool.
    let flat = LaneRequest::new(ladder(&[(50, 0.6), (100, 0.6), (200, 0.6)]));
    let other = LaneRequest::new(ladder(&[(60, 0.9)]));
    let request =
        AllocationRequest::new(vec![flat, other], Duration::from_secs(200), Duration::ZERO)
            .with_solve_cap(TEST_SOLVE_CAP);
    let outcome = allocate(&request);
    let plan = plan_of(outcome);
    assert_eq!(
        plan.grants()[0].cap,
        Duration::from_secs(50),
        "the flat lane must take its cheapest rung"
    );
    assert_eq!(plan.grants()[1].cap, Duration::from_mins(1));
    assert_eq!(plan.committed(), Duration::from_secs(110));
}

// ---------------------------------------------------------------------------
// (c) STRUCTURAL ZERO
// ---------------------------------------------------------------------------

#[test]
fn a_structurally_zeroed_lane_gets_exactly_zero_and_its_seconds_move() {
    // The measured case: a flat objective tier (ONE distinct objective value
    // over the probe) on a gradient-guided lane.
    let zero = StructuralZero::flat_objective(1, ObjectiveRequirement::EstimatedGradient)
        .expect("one distinct value on a gradient-guided lane is a structural zero");

    let strong = || lane_with_probabilities(&[(50, 0.3), (100, 0.6)]);
    let weak = || lane_with_probabilities(&[(50, 0.2), (100, 0.5)]);

    // Without the zero, the pool goes to the STRONG lane.
    let baseline = AllocationRequest::new(
        vec![strong(), weak()],
        Duration::from_secs(100),
        Duration::ZERO,
    )
    .with_solve_cap(TEST_SOLVE_CAP);
    let baseline_plan = plan_of(allocate(&baseline));
    assert_eq!(baseline_plan.grants()[0].cap, Duration::from_secs(100));
    assert_eq!(baseline_plan.grants()[1].cap, Duration::ZERO);

    // Zero the strong lane on structural evidence: it gets EXACTLY zero and
    // every one of its seconds lands on the other lane.
    let zeroed = AllocationRequest::new(
        vec![strong().zeroed(zero), weak()],
        Duration::from_secs(100),
        Duration::ZERO,
    )
    .with_solve_cap(TEST_SOLVE_CAP);
    let plan = plan_of(allocate(&zeroed));
    assert_eq!(plan.grants()[0].rung, 0);
    assert_eq!(
        plan.grants()[0].cap,
        Duration::ZERO,
        "a structural zero must be allocated exactly zero seconds"
    );
    assert_eq!(
        plan.grants()[1].cap,
        Duration::from_secs(100),
        "the freed seconds must land on another lane"
    );
    assert_eq!(plan.committed(), Duration::from_secs(100));
}

#[test]
fn structural_zeros_refuse_evidence_that_does_not_imply_them() {
    // A flat objective does NOT zero a value-only lane: it has no direction to
    // lose. This is the guard that stops a zero being asserted on a hunch.
    assert!(StructuralZero::flat_objective(1, ObjectiveRequirement::ValueOnly).is_none());
    // A staircase objective does not zero anything.
    assert!(StructuralZero::flat_objective(6, ObjectiveRequirement::Exact).is_none());
    assert!(StructuralZero::flat_objective(33, ObjectiveRequirement::EstimatedGradient).is_none());
    // Only a STRUCTURAL decline receipt is a zero.
    assert!(StructuralZero::structurally_declined(false).is_none());
    assert_eq!(
        StructuralZero::structurally_declined(true),
        Some(StructuralZero::StructurallyDeclined)
    );
    // A ceiling zero needs the instance to be strictly above the ceiling.
    assert!(StructuralZero::above_admission_ceiling(32_768, 32_768).is_none());
    assert_eq!(
        StructuralZero::above_admission_ceiling(32_769, 32_768),
        Some(StructuralZero::AboveAdmissionCeiling {
            free_dims: 32_769,
            ceiling: 32_768
        })
    );
    // The tier boundary itself is the probe's distinct-value count and needs no
    // threshold at the only boundary that carries a zero.
    assert_eq!(
        ObjectiveTier::from_distinct_objective_values(1),
        ObjectiveTier::Flat
    );
    assert_eq!(
        ObjectiveTier::from_distinct_objective_values(6),
        ObjectiveTier::Staircase
    );
    assert_eq!(
        ObjectiveTier::from_distinct_objective_values(33),
        ObjectiveTier::Smooth
    );
}

#[test]
fn a_zeroed_lane_drags_everything_that_depends_on_it_to_zero() {
    let zero = StructuralZero::above_admission_ceiling(65_536, 32_768).expect("above the ceiling");
    let upstream = LaneRequest::new(ladder(&[(40, 0.8)])).zeroed(zero);
    let downstream = LaneRequest::new(ladder(&[(40, 0.9)])).requiring(0);
    let free = LaneRequest::new(ladder(&[(40, 0.5)]));
    let request = AllocationRequest::new(
        vec![upstream, downstream, free],
        Duration::from_secs(200),
        Duration::ZERO,
    )
    .with_solve_cap(TEST_SOLVE_CAP);
    let plan = plan_of(allocate(&request));
    assert_eq!(plan.grants()[0].cap, Duration::ZERO);
    assert_eq!(
        plan.grants()[1].cap,
        Duration::ZERO,
        "(P): a lane cannot run without the lane it needs"
    );
    assert_eq!(plan.grants()[2].cap, Duration::from_secs(40));
}

// ---------------------------------------------------------------------------
// (d) THE BUDGET IS NEVER EXCEEDED
// ---------------------------------------------------------------------------

#[test]
fn the_pool_including_the_reserve_is_never_exceeded() {
    let mut rng = Rng::new(0xB0D6_E700_0BEE_0002);
    let mut plans = 0usize;
    for case in 0..200u64 {
        let k = 1 + rng.below(6) as usize;
        let mut lanes = Vec::with_capacity(k);
        for _ in 0..k {
            let rungs = 1 + rng.below(6) as usize;
            let mut points = Vec::new();
            let mut secs = 0u64;
            let mut reach = 0.0f64;
            for _ in 0..rungs {
                secs += 1 + rng.below(90);
                reach = (reach + rng.unit() * 0.5).min(1.0);
                points.push((secs, reach));
            }
            let mut lane = LaneRequest::new(ladder(&points));
            if rng.below(8) == 0 {
                lane = lane.zeroed(StructuralZero::StructurallyDeclined);
            }
            lanes.push(lane);
        }
        // Reserves that sometimes eat the whole budget.
        let budget = Duration::from_secs(10 + rng.below(300));
        let reserve = Duration::from_secs(rng.below(120));
        let request = AllocationRequest::new(lanes, budget, reserve).with_solve_cap(TEST_SOLVE_CAP);
        match allocate(&request) {
            AllocationOutcome::Allocated(plan) => {
                plans += 1;
                let spent = plan.committed();
                assert!(
                    spent <= request.pool(),
                    "case {case}: committed {spent:?} exceeds pool {:?} (budget {budget:?}, \
                     reserve {reserve:?})",
                    request.pool()
                );
                if reserve <= budget {
                    assert!(
                        spent + reserve <= budget,
                        "case {case}: the reserve was eaten"
                    );
                }
                for grant in plan.grants() {
                    let rung = &request.lanes()[grant.lane].ladder().rungs()[grant.rung];
                    assert_eq!(grant.cap, rung.cap);
                    // A nonzero grant is never below the lane's own floor.
                    if grant.rung > 0 {
                        assert!(
                            grant.cap
                                >= request.lanes()[grant.lane]
                                    .ladder()
                                    .floor()
                                    .expect("a nonzero rung implies a floor")
                        );
                    }
                }
            }
            AllocationOutcome::UseExistingPlan(reason) => {
                assert!(
                    matches!(reason, FallOpen::Infeasible),
                    "case {case}: unexpected fall-open {reason:?}"
                );
            }
        }
    }
    assert!(plans > 150, "only {plans} of 200 cases produced a plan");
    println!("budget safety: {plans} plans checked against the pool");
}

#[test]
fn the_no_regression_floor_keeps_todays_plan_feasible() {
    // (C5): today's fixed slice is a lower bound, so today's plan is always a
    // feasible point and the optimum can only be better.
    let lane_a =
        LaneRequest::new(ladder(&[(30, 0.2), (90, 0.9)])).no_worse_than(Duration::from_secs(30));
    let lane_b =
        LaneRequest::new(ladder(&[(30, 0.9), (90, 0.95)])).no_worse_than(Duration::from_secs(30));
    let request =
        AllocationRequest::new(vec![lane_a, lane_b], Duration::from_mins(2), Duration::ZERO)
            .with_solve_cap(TEST_SOLVE_CAP);
    let plan = plan_of(allocate(&request));
    for grant in plan.grants() {
        assert!(
            grant.cap >= Duration::from_secs(30),
            "(C5) violated: lane {} got {:?}",
            grant.lane,
            grant.cap
        );
    }
    let (best, _) = brute_force(&request).expect("today's plan is feasible");
    assert!((plan.log_miss_total() - best).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// (e) FAIL OPEN
// ---------------------------------------------------------------------------

#[test]
fn an_expired_solve_cap_falls_open_without_panicking() {
    let lanes = vec![
        LaneRequest::new(ladder(&[(10, 0.3), (20, 0.6)])),
        LaneRequest::new(ladder(&[(10, 0.4), (20, 0.7)])),
    ];
    let request = AllocationRequest::new(lanes, Duration::from_mins(1), Duration::ZERO)
        .with_solve_cap(Duration::ZERO);
    assert_eq!(
        allocate(&request),
        AllocationOutcome::UseExistingPlan(FallOpen::SolveDeadline)
    );
}

#[test]
fn an_infeasible_model_falls_open() {
    // The reserve eats the whole budget, so even the all-zero assignment
    // cannot satisfy the knapsack row.
    let lanes = vec![LaneRequest::new(ladder(&[(10, 0.5)]))];
    let request = AllocationRequest::new(lanes, Duration::from_secs(10), Duration::from_secs(30));
    assert_eq!(request.pool(), Duration::ZERO);

    // and the sharper case: (C5) demands more than the pool holds.
    let lanes = vec![
        LaneRequest::new(ladder(&[(100, 0.5)])).no_worse_than(Duration::from_secs(100)),
        LaneRequest::new(ladder(&[(100, 0.5)])).no_worse_than(Duration::from_secs(100)),
    ];
    let request = AllocationRequest::new(lanes, Duration::from_secs(150), Duration::from_secs(10))
        .with_solve_cap(TEST_SOLVE_CAP);
    assert_eq!(
        allocate(&request),
        AllocationOutcome::UseExistingPlan(FallOpen::Infeasible)
    );
}

#[test]
fn malformed_input_falls_open_and_never_panics() {
    // A non-finite prior.
    let request = AllocationRequest::new(
        vec![LaneRequest::new(ladder(&[(10, 0.5)])).with_reach_prior(f64::NAN)],
        Duration::from_mins(1),
        Duration::ZERO,
    );
    assert_eq!(
        allocate(&request),
        AllocationOutcome::UseExistingPlan(FallOpen::NonFiniteInput { lane: 0 })
    );

    // A precedence edge naming a lane that does not exist.
    let request = AllocationRequest::new(
        vec![LaneRequest::new(ladder(&[(10, 0.5)])).requiring(7)],
        Duration::from_mins(1),
        Duration::ZERO,
    );
    assert_eq!(
        allocate(&request),
        AllocationOutcome::UseExistingPlan(FallOpen::UnknownPrecedence {
            lane: 0,
            requires: 7
        })
    );

    // More lanes than the allocator sizes for.
    let many: Vec<LaneRequest> = (0..=MAX_LANES)
        .map(|_| LaneRequest::new(ladder(&[(10, 0.5)])))
        .collect();
    let request = AllocationRequest::new(many, Duration::from_mins(10), Duration::ZERO);
    assert_eq!(
        allocate(&request),
        AllocationOutcome::UseExistingPlan(FallOpen::TooManyLanes {
            lanes: MAX_LANES + 1
        })
    );

    // No lanes at all is an empty plan, not a fall-open and not a panic.
    let request = AllocationRequest::new(Vec::new(), Duration::from_mins(1), Duration::ZERO);
    let plan = plan_of(allocate(&request));
    assert!(plan.grants().is_empty());
    assert_eq!(plan.committed(), Duration::ZERO);
}

#[test]
fn malformed_ladders_are_rejected_at_construction() {
    assert_eq!(
        CapLadder::caller_supplied(Vec::new()),
        Err(LadderError::Empty)
    );
    let nonzero_first = vec![Rung {
        cap: Duration::from_secs(1),
        reach: 0.0,
        origin: RungOrigin::CallerSupplied,
    }];
    assert_eq!(
        CapLadder::caller_supplied(nonzero_first),
        Err(LadderError::FirstRungIsNotZero)
    );
    let decreasing = vec![
        Rung {
            cap: Duration::ZERO,
            reach: 0.0,
            origin: RungOrigin::DoNotRun,
        },
        Rung {
            cap: Duration::from_secs(10),
            reach: 0.5,
            origin: RungOrigin::CallerSupplied,
        },
        Rung {
            cap: Duration::from_secs(5),
            reach: 0.6,
            origin: RungOrigin::CallerSupplied,
        },
    ];
    assert_eq!(
        CapLadder::caller_supplied(decreasing),
        Err(LadderError::CapNotIncreasing { rung: 2 })
    );
    let fractional = vec![
        Rung {
            cap: Duration::ZERO,
            reach: 0.0,
            origin: RungOrigin::DoNotRun,
        },
        Rung {
            cap: Duration::from_millis(1_500),
            reach: 0.5,
            origin: RungOrigin::CallerSupplied,
        },
    ];
    assert_eq!(
        CapLadder::caller_supplied(fractional),
        Err(LadderError::CapNotWholeSeconds { rung: 1 })
    );
}

// ---------------------------------------------------------------------------
// (f) NO ANSWER-BEARING TYPE IS REACHABLE
// ---------------------------------------------------------------------------

#[test]
fn no_answer_bearing_type_is_reachable() {
    // The structural version of "allocation is neutral": the module's source is
    // re-read and the build fails if a type that could carry an instance answer
    // appears in it at all -- not merely on the public surface.
    //
    // Comments are stripped first: the module doc necessarily uses these words
    // to say that they are forbidden.
    let source = include_str!("lane_allocation.rs");
    let code: String = source
        .lines()
        .filter(|line| {
            let t = line.trim_start();
            !t.starts_with("//") && !t.starts_with("/*") && !t.starts_with('*')
        })
        .collect::<Vec<_>>()
        .join("\n")
        .to_lowercase();

    for forbidden in [
        "unsat",
        "verdict",
        "counterexample",
        "witness",
        "mipresult",
        "signspaceoutcome",
        "phaseyield",
        "candidate",
        "refut",
        "certificate",
        "is_sat",
        "sat_",
        "_sat",
        "::sat",
        // and the anti-lookup-table rule, inherited from admission
        "filename",
        "file_name",
        "category",
        "preset",
        "benchmark",
        "instance_name",
        "directory",
        "onnx",
        "vnnlib",
    ] {
        assert!(
            !code.contains(forbidden),
            "`{forbidden}` appears in lane_allocation.rs outside a comment: the allocator \
             chooses caps and must not be able to name, carry or influence an instance answer"
        );
    }

    // It must also not reach for one: the only crate-internal seams it uses are
    // the IR, the lowering and the deadline wrapper.
    for line in source
        .lines()
        .filter(|l| l.trim_start().starts_with("use "))
    {
        assert!(
            line.contains("std::time") || line.contains("num_traits") || line.contains("crate::ir"),
            "unexpected import in the allocator: {line}"
        );
    }
}

// ---------------------------------------------------------------------------
// The declared ladders, verified against the source they cite
// ---------------------------------------------------------------------------

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("crates/ny-mip has a workspace root")
        .to_path_buf()
}

#[test]
fn every_cited_constant_still_reads_that_way_in_its_source() {
    for lane in [Lane::BnnSignSpace, Lane::BnnStePgd] {
        let ladder = declared_ladder(lane, Duration::from_mins(30)).expect("declared");
        let LadderProvenance::ReadFromSource { citations, .. } = ladder.provenance() else {
            panic!("{lane:?} must carry its citations");
        };
        assert!(!citations.is_empty());
        for citation in *citations {
            let path = repo_root().join(citation.file);
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cited file {} is unreadable: {e}", citation.file));
            let hits: Vec<usize> = text
                .lines()
                .enumerate()
                .filter(|(_, l)| l.contains(citation.item) && l.contains(citation.value))
                .map(|(i, _)| i + 1)
                .collect();
            assert!(
                !hits.is_empty(),
                "{}: `{} ... {}` is no longer in the source -- the ladder must be re-read, \
                 not patched",
                citation.file,
                citation.item,
                citation.value
            );
            if !hits.contains(&(citation.line as usize)) {
                println!(
                    "citation line drifted (advisory): {}:{} -> now at {hits:?} for `{}`",
                    citation.file, citation.line, citation.item
                );
            }
        }
    }
}

#[test]
fn the_declared_ladders_are_the_source_constants_and_nothing_else() {
    // sign-space: floor = stall_lp_solves * per_lp_time = 32 * 1 s; top = the
    // declared max_wall_time of 5 min.
    let ladder = declared_ladder(Lane::BnnSignSpace, Duration::from_mins(8)).expect("declared");
    let caps: Vec<u64> = ladder.rungs().iter().map(|r| r.cap.as_secs()).collect();
    assert_eq!(caps, vec![0, 32, 300]);
    assert_eq!(ladder.floor(), Some(Duration::from_secs(32)));
    assert_eq!(ladder.rungs()[1].origin, RungOrigin::DeclaredFloor);
    assert_eq!(ladder.rungs()[2].origin, RungOrigin::DeclaredSchedule);
    // reach is the declared LP-solve budget the rung affords, normalised.
    assert!((ladder.rungs()[1].reach - 32.0 / 300.0).abs() < 1e-12);
    assert!((ladder.rungs()[2].reach - 1.0).abs() < 1e-12);

    // Under a pool smaller than the declared wall the top rung is the pool.
    let ladder = declared_ladder(Lane::BnnSignSpace, Duration::from_secs(100)).expect("declared");
    let caps: Vec<u64> = ladder.rungs().iter().map(|r| r.cap.as_secs()).collect();
    assert_eq!(caps, vec![0, 32, 100]);
    assert_eq!(ladder.rungs()[2].origin, RungOrigin::BudgetTruncated);

    // Below its own floor the lane has only the do-not-run rung.
    let ladder = declared_ladder(Lane::BnnSignSpace, Duration::from_secs(20)).expect("declared");
    assert_eq!(ladder.rungs().len(), 1);
    assert_eq!(ladder.floor(), None);

    // STE-PGD: the ONE cap its source declares is `Duration::from_mins(4)`.
    let ladder = declared_ladder(Lane::BnnStePgd, Duration::from_mins(8)).expect("declared");
    let caps: Vec<u64> = ladder.rungs().iter().map(|r| r.cap.as_secs()).collect();
    assert_eq!(caps, vec![0, 240]);
    assert_eq!(ladder.rungs()[1].origin, RungOrigin::DeclaredSchedule);

    let ladder = declared_ladder(Lane::BnnStePgd, Duration::from_secs(150)).expect("declared");
    assert_eq!(
        ladder.rungs().len(),
        1,
        "240 s does not fit in a 150 s pool"
    );
}

#[test]
fn lanes_without_a_source_declared_ladder_say_so() {
    for lane in [
        Lane::UpfrontAttack,
        Lane::FalsifyPortfolio,
        Lane::MarginRowConcurrent,
        Lane::ForwardLinearAdmission,
        Lane::PostBabFrontier,
    ] {
        let unknown = declared_ladder(lane, Duration::from_mins(8))
            .expect_err("Layer A must not invent a ladder");
        assert_eq!(unknown.lane, lane);
        assert!(!unknown.why.is_empty());
    }
}

#[test]
fn the_traffic_shaped_instance_puts_the_flat_lane_at_zero_and_ste_at_its_declared_rung() {
    // The cold-start prediction, run on the DECLARED ladders: 480 s of budget,
    // a 145 s reserve, an objective probe that found ONE distinct value, and a
    // sign-space lane that steers by that objective. It must be zeroed, and
    // STE-PGD must land on the 240 s rung its own source declares.
    let budget = Duration::from_mins(8);
    let reserve = Duration::from_secs(145);
    let pool = budget.checked_sub(reserve).unwrap();
    let blind = StructuralZero::flat_objective(1, ObjectiveRequirement::EstimatedGradient)
        .expect("a constant objective blinds a gradient-guided lane");
    let lanes = vec![
        LaneRequest::new(declared_ladder(Lane::BnnSignSpace, pool).expect("declared"))
            .zeroed(blind),
        LaneRequest::new(declared_ladder(Lane::BnnStePgd, pool).expect("declared")),
    ];
    let request = AllocationRequest::new(lanes, budget, reserve).with_solve_cap(TEST_SOLVE_CAP);
    let plan = plan_of(allocate(&request));
    assert_eq!(plan.grants()[0].cap, Duration::ZERO);
    assert_eq!(
        plan.grants()[1].cap,
        Duration::from_mins(4),
        "STE-PGD must get the cap its own schedule declares"
    );
}

// ---------------------------------------------------------------------------
// Microbenchmark
// ---------------------------------------------------------------------------

#[test]
fn microbenchmark_a_realistic_five_lane_ten_rung_instance() {
    // K = 5 lanes x 10 rungs = 50 binaries, 5 (E) rows + 1 (B) row + 1 (P) row.
    // One lane carries a structural zero (pinned columns), one a precedence
    // edge. Budget and reserve are the traffic shape: 480 s and 145 s.
    let request = || {
        let mut rng = Rng::new(0x5EED_0BEE_F00D_0003);
        let mut lanes = Vec::with_capacity(5);
        for k in 0..5usize {
            let mut points = Vec::with_capacity(10);
            let mut secs = 0u64;
            let mut reach = 0.0f64;
            for j in 0..10 {
                secs += 12 + rng.below(25);
                // a step-like profile: most of the reach arrives at one rung
                reach = if j == 6 {
                    (reach + 0.5).min(1.0)
                } else {
                    (reach + rng.unit() * 0.05).min(1.0)
                };
                points.push((secs, reach));
            }
            let mut lane = LaneRequest::new(ladder(&points));
            if k == 3 {
                lane = lane.zeroed(StructuralZero::StructurallyDeclined);
            }
            if k == 4 {
                lane = lane.requiring(0);
            }
            lanes.push(lane);
        }
        AllocationRequest::new(lanes, Duration::from_mins(8), Duration::from_secs(145))
    };

    // 1. The TRUE solve cost: an uncapped-in-practice run, so the number
    //    reported is the solve and not a truncation.
    let generous = request().with_solve_cap(Duration::from_secs(30));
    let plan = plan_of(allocate(&generous)); // warm-up
    let runs = 20u32;
    let mut worst = Duration::ZERO;
    let started = Instant::now();
    for _ in 0..runs {
        let one = Instant::now();
        let outcome = allocate(&generous);
        worst = worst.max(one.elapsed());
        assert!(matches!(outcome, AllocationOutcome::Allocated(_)));
    }
    let mean = started.elapsed() / runs;
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    println!(
        "MICROBENCHMARK ({profile}) K=5 grid=10 -> 50 binaries, 7 rows: mean {mean:?}, \
         worst {worst:?} over {runs} solves"
    );
    println!(
        "  chosen caps {:?} s, committed {:?} of a {:?} pool, prior union reach {:.4}",
        plan.grants()
            .iter()
            .map(|g| g.cap.as_secs())
            .collect::<Vec<_>>(),
        plan.committed(),
        generous.pool(),
        plan.prior_union_reach()
    );
    let (best, _) = brute_force(&generous).expect("feasible");
    assert!(
        (plan.log_miss_total() - best).abs() < 1e-9,
        "the benchmarked instance must also be optimal"
    );

    // 2. The SHIPPED cap. Reported, not asserted in a debug build: the point of
    //    the fail-open path is that missing the cap costs the status quo, so a
    //    slow build must not turn a scheduling parameter into a red test.
    let production = request();
    assert_eq!(production.solve_cap, ALLOC_SOLVE_CAP);
    let landed = (0..runs)
        .filter(|_| matches!(allocate(&production), AllocationOutcome::Allocated(_)))
        .count();
    println!(
        "  under the shipped ALLOC_SOLVE_CAP of {ALLOC_SOLVE_CAP:?}: {landed}/{runs} solves \
         produced a plan, the rest fell open to the existing plan"
    );
    // Deliberately NOT asserted. A wall-clock bound in a test is a thermal
    // measurement of the host, and this one throttles; the property that must
    // hold is that missing the cap costs the status quo and nothing else, and
    // that is what `an_expired_solve_cap_falls_open_without_panicking` proves.
    assert!(landed <= runs as usize);
}
