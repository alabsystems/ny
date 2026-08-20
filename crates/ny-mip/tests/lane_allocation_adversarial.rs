// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! ADVERSARIAL verification of the Layer A lane allocator, written against the
//! PUBLIC API only. It shares no code and no private field access with
//! `lane_allocation_tests`, and it re-derives the objective from the documented
//! formula rather than calling `log_miss_cost_at`, so a formula that drifted
//! from its own doc comment would show up here as a disagreement.

use std::time::{Duration, Instant};

use ny_mip::{
    allocate, AllocationOutcome, AllocationRequest, CapLadder, FallOpen, LaneRequest,
    ObjectiveRequirement, Rung, RungOrigin, StructuralZero, MAX_LANES, MAX_RUNGS_PER_LANE,
    OBJECTIVE_SNAP_BITS, REACH_PROBABILITY_CLAMP,
};

const CAP: Duration = Duration::from_secs(5);

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0xDEAD_BEEF_CAFE_F00D)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next_u64() % n
        }
    }
    fn unit(&mut self) -> f64 {
        (self.below(10_001) as f64) / 10_000.0
    }
}

/// My own description of an instance. The `AllocationRequest` and the brute
/// force are each built from this, separately.
#[derive(Clone, Debug)]
struct Spec {
    lanes: Vec<LaneSpec>,
    budget: u64,
    reserve: u64,
}

#[derive(Clone, Debug)]
struct LaneSpec {
    /// (cap seconds, reach) for rungs 1..; rung 0 is implied.
    points: Vec<(u64, f64)>,
    prior: f64,
    zeroed: bool,
    requires: Option<usize>,
    floor: Option<u64>,
}

/// The documented objective, re-derived here: p = clamp(a*s), c = snap(ln(1-p)).
fn my_cost(spec: &LaneSpec, j: usize) -> f64 {
    let p = if spec.zeroed || j == 0 {
        0.0
    } else {
        (spec.prior * spec.points[j - 1].1).clamp(0.0, REACH_PROBABILITY_CLAMP)
    };
    let raw = (1.0 - p).ln();
    let scale = (2.0f64).powi(OBJECTIVE_SNAP_BITS);
    (raw * scale).round() / scale
}

fn my_cap(spec: &LaneSpec, j: usize) -> u64 {
    if j == 0 {
        0
    } else {
        spec.points[j - 1].0
    }
}

fn n_rungs(spec: &LaneSpec) -> usize {
    spec.points.len() + 1
}

fn to_request(spec: &Spec) -> AllocationRequest {
    let lanes = spec
        .lanes
        .iter()
        .map(|l| {
            let mut rungs = vec![Rung {
                cap: Duration::ZERO,
                reach: 0.0,
                origin: RungOrigin::DoNotRun,
            }];
            for &(secs, reach) in &l.points {
                rungs.push(Rung {
                    cap: Duration::from_secs(secs),
                    reach,
                    origin: RungOrigin::CallerSupplied,
                });
            }
            let ladder = CapLadder::caller_supplied(rungs).expect("well-formed ladder");
            let mut req = LaneRequest::new(ladder).with_reach_prior(l.prior);
            if l.zeroed {
                // Real structural evidence: a flat objective probe against a
                // gradient-guided lane.
                req = req.zeroed(
                    StructuralZero::flat_objective(1, ObjectiveRequirement::EstimatedGradient)
                        .expect("1 distinct value + gradient-guided IS a structural zero"),
                );
            }
            if let Some(p) = l.requires {
                req = req.requiring(p);
            }
            if let Some(f) = l.floor {
                req = req.no_worse_than(Duration::from_secs(f));
            }
            req
        })
        .collect();
    AllocationRequest::new(
        lanes,
        Duration::from_secs(spec.budget),
        Duration::from_secs(spec.reserve),
    )
    .with_solve_cap(CAP)
}

/// Independent enumeration of every assignment, filtered on (A),(B),(P),(C5).
/// Returns (best objective, a witness) or None when nothing is feasible.
fn brute(spec: &Spec) -> Option<(f64, Vec<usize>)> {
    let pool = spec.budget.saturating_sub(spec.reserve);
    let sizes: Vec<usize> = spec.lanes.iter().map(n_rungs).collect();
    let total: u128 = sizes.iter().map(|&s| s as u128).product();
    assert!(total < 20_000_000, "enumeration too large: {total}");
    let mut best: Option<(f64, Vec<usize>)> = None;
    for code in 0..total {
        let mut rest = code;
        let mut pick = Vec::with_capacity(sizes.len());
        for &s in &sizes {
            pick.push((rest % s as u128) as usize);
            rest /= s as u128;
        }
        let mut ok = true;
        let mut spent: u128 = 0;
        for (k, l) in spec.lanes.iter().enumerate() {
            if l.zeroed && pick[k] != 0 {
                ok = false;
                break;
            }
            if let Some(p) = l.requires {
                if pick[k] > 0 && pick[p] == 0 {
                    ok = false;
                    break;
                }
            }
            if !l.zeroed {
                if let Some(f) = l.floor {
                    if my_cap(l, pick[k]) < f {
                        ok = false;
                        break;
                    }
                }
            }
            spent += u128::from(my_cap(l, pick[k]));
        }
        if !ok || spent > u128::from(pool) {
            continue;
        }
        let value: f64 = spec
            .lanes
            .iter()
            .enumerate()
            .map(|(k, l)| my_cost(l, pick[k]))
            .sum();
        if best.as_ref().is_none_or(|(b, _)| value < *b - 1e-12) {
            best = Some((value, pick));
        }
    }
    best
}

/// Solve, cross-check against brute force, and return (wall, planned?).
fn check(spec: &Spec, label: &str) -> (Duration, bool) {
    let request = to_request(spec);
    let started = Instant::now();
    let outcome = allocate(&request);
    let elapsed = started.elapsed();
    let expect = brute(spec);
    let pool = spec.budget.saturating_sub(spec.reserve);
    let planned = matches!(outcome, AllocationOutcome::Allocated(_));
    match (outcome, expect) {
        (AllocationOutcome::Allocated(plan), Some((best, witness))) => {
            let got: f64 = plan
                .grants()
                .iter()
                .enumerate()
                .map(|(k, g)| my_cost(&spec.lanes[k], g.rung))
                .sum();
            assert!(
                (got - best).abs() <= 1e-9,
                "{label}: MILP objective {got} != brute-force optimum {best} \
                 (milp picked {:?}, brute {witness:?}) on {spec:?}",
                plan.grants().iter().map(|g| g.rung).collect::<Vec<_>>()
            );
            assert!(
                (plan.log_miss_total() - best).abs() <= 1e-9,
                "{label}: reported log_miss_total {} != optimum {best}",
                plan.log_miss_total()
            );
            // the plan itself must be feasible, independently re-checked
            assert_eq!(plan.grants().len(), spec.lanes.len(), "{label}: grant count");
            let mut spent = 0u64;
            for (k, g) in plan.grants().iter().enumerate() {
                let l = &spec.lanes[k];
                assert_eq!(g.lane, k, "{label}: grant order");
                assert!(g.rung < n_rungs(l), "{label}: rung out of range");
                assert_eq!(
                    g.cap,
                    Duration::from_secs(my_cap(l, g.rung)),
                    "{label}: cap does not match its rung"
                );
                assert!(!(l.zeroed && g.rung != 0), "{label}: zeroed lane ran");
                if let Some(p) = l.requires {
                    assert!(
                        !(g.rung > 0 && plan.grants()[p].rung == 0),
                        "{label}: precedence violated"
                    );
                }
                if !l.zeroed {
                    if let Some(f) = l.floor {
                        assert!(my_cap(l, g.rung) >= f, "{label}: (C5) violated");
                    }
                }
                assert!(
                    g.rung == 0 || g.cap >= Duration::from_secs(my_cap(l, 1)),
                    "{label}: a nonzero grant below the lane floor"
                );
                spent += my_cap(l, g.rung);
            }
            assert!(spent <= pool, "{label}: {spent}s committed of a {pool}s pool");
            assert_eq!(plan.committed(), Duration::from_secs(spent));
            assert!(
                Duration::from_secs(spent + spec.reserve) <= Duration::from_secs(spec.budget)
                    || spec.reserve > spec.budget,
                "{label}: committed + reserve exceeds budget"
            );
        }
        (AllocationOutcome::UseExistingPlan(FallOpen::Infeasible), None) => {}
        (AllocationOutcome::UseExistingPlan(FallOpen::Infeasible), Some((best, w))) => {
            panic!("{label}: STOP-THE-LINE -- declared infeasible but {w:?} scores {best} on {spec:?}")
        }
        (AllocationOutcome::Allocated(plan), None) => panic!(
            "{label}: STOP-THE-LINE -- produced a plan {:?} for a model brute force says is infeasible on {spec:?}",
            plan.grants()
        ),
        (AllocationOutcome::UseExistingPlan(other), _) => {
            panic!("{label}: unexpected fall-open {other:?} on {spec:?}")
        }
    }
    (elapsed, planned)
}

fn random_spec(rng: &mut Rng, max_lanes: usize, max_rungs: usize) -> Spec {
    let k = 1 + rng.below(max_lanes as u64) as usize;
    let mut lanes = Vec::with_capacity(k);
    for i in 0..k {
        let m = rng.below(max_rungs as u64 + 1) as usize;
        let mut caps: Vec<u64> = Vec::new();
        let mut cur = 0u64;
        for _ in 0..m {
            cur += 1 + rng.below(120);
            caps.push(cur);
        }
        // reaches: non-decreasing in [0,1], with deliberate TIES and PLATEAUS
        let mut reaches: Vec<f64> = Vec::with_capacity(m);
        let mut r = 0.0f64;
        for _ in 0..m {
            match rng.below(4) {
                0 => {}                                  // tie with the rung below
                1 => r = (r + 0.5 * (1.0 - r)).min(1.0), // a step
                _ => r = (r + rng.unit() * (1.0 - r)).min(1.0),
            }
            reaches.push(r);
        }
        let points: Vec<(u64, f64)> = caps.into_iter().zip(reaches).collect();
        lanes.push(LaneSpec {
            prior: match rng.below(6) {
                0 => 0.0,
                1 => 1.0,
                _ => rng.unit(),
            },
            zeroed: rng.below(6) == 0,
            requires: if i > 0 && rng.below(5) == 0 {
                Some(rng.below(i as u64) as usize)
            } else {
                None
            },
            floor: if !points.is_empty() && rng.below(4) == 0 {
                Some(points[rng.below(points.len() as u64) as usize].0)
            } else {
                None
            },
            points,
        });
    }
    let budget = rng.below(900);
    let reserve = match rng.below(8) {
        0 => budget + rng.below(200), // reserve exceeds the budget: pool 0
        _ => rng.below(budget.max(1)),
    };
    Spec {
        lanes,
        budget,
        reserve,
    }
}

#[test]
fn adversarial_optimality_against_brute_force() {
    let mut rng = Rng::new(0x0A11_C0DE_5EED);
    let mut solved = 0usize;
    let mut infeasible = 0usize;
    let mut walls: Vec<Duration> = Vec::new();
    let mut worst: Option<(Duration, u64, Spec)> = None;
    for i in 0..1500u64 {
        let spec = random_spec(&mut rng, 5, 5);
        let (w, planned) = check(&spec, &format!("random#{i}"));
        walls.push(w);
        if worst.as_ref().is_none_or(|(b, _, _)| w > *b) {
            worst = Some((w, i, spec.clone()));
        }
        if planned {
            solved += 1;
        } else {
            infeasible += 1;
        }
    }
    if let Some((w, i, spec)) = worst {
        let binaries: usize = spec.lanes.iter().map(|l| l.points.len() + 1).sum();
        eprintln!(
            "  WORST: random#{i} at {w:?}, {} lanes / {binaries} binaries, pool {}s",
            spec.lanes.len(),
            spec.budget.saturating_sub(spec.reserve)
        );
        for (k, l) in spec.lanes.iter().enumerate() {
            eprintln!(
                "    lane {k}: zeroed={} requires={:?} floor={:?} prior={:.4} caps={:?}",
                l.zeroed,
                l.requires,
                l.floor,
                l.prior,
                l.points.iter().map(|p| p.0).collect::<Vec<_>>()
            );
        }
        // re-time it five times in isolation to separate a real cost from host noise
        let req = to_request(&spec);
        let mut again = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            let _ = allocate(&req);
            again.push(t.elapsed());
        }
        eprintln!("  WORST re-timed in isolation: {again:?}");
    }
    walls.sort_unstable();
    eprintln!(
        "ADVERSARIAL OPTIMALITY: 1500 instances, {solved} solved, {infeasible} provably infeasible; \
         wall median {:?} p99 {:?} max {:?}",
        walls[walls.len() / 2],
        walls[walls.len() * 99 / 100],
        walls[walls.len() - 1]
    );
    assert!(solved > 0 && infeasible > 0, "both arms must be exercised");
}

#[test]
fn adversarial_optimality_at_full_width() {
    // MAX_LANES x a wide ladder: the largest shape the module accepts, still
    // enumerable. 8 lanes x 6 rungs = 7^8 = 5.7M assignments, so keep it small.
    let mut rng = Rng::new(0x0FF1_CE05_EEDA);
    let mut walls = Vec::new();
    for i in 0..120u64 {
        let spec = random_spec(&mut rng, MAX_LANES, 5);
        walls.push(check(&spec, &format!("wide#{i}")).0);
    }
    walls.sort_unstable();
    eprintln!(
        "ADVERSARIAL WIDE (K<=8): 120 instances, wall median {:?} max {:?}",
        walls[walls.len() / 2],
        walls[walls.len() - 1]
    );
}

#[test]
fn adversarial_degenerate_cases() {
    // 1. ALL LANES DECLINED
    let all_declined = Spec {
        lanes: (0..4)
            .map(|_| LaneSpec {
                points: vec![(10, 0.3), (50, 0.9)],
                prior: 0.5,
                zeroed: true,
                requires: None,
                floor: None,
            })
            .collect(),
        budget: 480,
        reserve: 45,
    };
    check(&all_declined, "all-declined");
    let plan = match allocate(&to_request(&all_declined)) {
        AllocationOutcome::Allocated(p) => p,
        other => panic!("all-declined should still plan: {other:?}"),
    };
    assert_eq!(plan.committed(), Duration::ZERO);
    assert!(plan.grants().iter().all(|g| g.rung == 0));
    assert_eq!(plan.log_miss_total(), 0.0);
    assert_eq!(plan.prior_union_reach(), 0.0);

    // 2. BUDGET BELOW EVERY FLOOR
    let starved = Spec {
        lanes: vec![
            LaneSpec {
                points: vec![(100, 0.5)],
                prior: 0.5,
                zeroed: false,
                requires: None,
                floor: None,
            },
            LaneSpec {
                points: vec![(240, 1.0)],
                prior: 0.5,
                zeroed: false,
                requires: None,
                floor: None,
            },
        ],
        budget: 60,
        reserve: 20,
    };
    check(&starved, "budget-below-every-floor");
    assert_eq!(
        match allocate(&to_request(&starved)) {
            AllocationOutcome::Allocated(p) => p.committed(),
            other => panic!("{other:?}"),
        },
        Duration::ZERO,
        "no lane may be handed a dribble"
    );

    // 3. SINGLE LANE, both arms
    for (budget, want) in [(500u64, 240u64), (100, 0)] {
        let single = Spec {
            lanes: vec![LaneSpec {
                points: vec![(240, 1.0)],
                prior: 0.5,
                zeroed: false,
                requires: None,
                floor: None,
            }],
            budget,
            reserve: 0,
        };
        check(&single, "single-lane");
        let p = match allocate(&to_request(&single)) {
            AllocationOutcome::Allocated(p) => p,
            other => panic!("{other:?}"),
        };
        assert_eq!(p.committed(), Duration::from_secs(want));
    }

    // 4. TIES IN COST -- identical reach at every rung: the cheapest cap that
    //    reaches must win, and the objective must equal the tied value.
    let ties = Spec {
        lanes: vec![LaneSpec {
            points: vec![(10, 0.7), (50, 0.7), (90, 0.7), (200, 0.7)],
            prior: 1.0,
            zeroed: false,
            requires: None,
            floor: None,
        }],
        budget: 400,
        reserve: 0,
    };
    check(&ties, "ties");
    let p = match allocate(&to_request(&ties)) {
        AllocationOutcome::Allocated(p) => p,
        other => panic!("{other:?}"),
    };
    assert_eq!(
        p.committed(),
        Duration::from_secs(10),
        "a tie must buy the cheapest rung"
    );

    // 5. ZERO-LENGTH GRID -- refused at construction, and a do-not-run-only
    //    ladder is legal and forces rung 0.
    assert!(
        CapLadder::caller_supplied(Vec::new()).is_err(),
        "empty grid must be refused"
    );
    let only_zero = Spec {
        lanes: vec![
            LaneSpec {
                points: Vec::new(),
                prior: 0.5,
                zeroed: false,
                requires: None,
                floor: None,
            },
            LaneSpec {
                points: vec![(30, 1.0)],
                prior: 0.5,
                zeroed: false,
                requires: None,
                floor: None,
            },
        ],
        budget: 200,
        reserve: 0,
    };
    check(&only_zero, "do-not-run-only-ladder");

    // 6. A LANE WHOSE ONLY NONZERO CAP EXCEEDS THE BUDGET
    let unaffordable = Spec {
        lanes: vec![
            LaneSpec {
                points: vec![(1000, 1.0)],
                prior: 1.0,
                zeroed: false,
                requires: None,
                floor: None,
            },
            LaneSpec {
                points: vec![(50, 0.2)],
                prior: 1.0,
                zeroed: false,
                requires: None,
                floor: None,
            },
        ],
        budget: 200,
        reserve: 0,
    };
    check(&unaffordable, "unaffordable-lane");
    let p = match allocate(&to_request(&unaffordable)) {
        AllocationOutcome::Allocated(p) => p,
        other => panic!("{other:?}"),
    };
    assert_eq!(p.grants()[0].rung, 0, "an unaffordable lane must not run");
    assert_eq!(p.grants()[1].cap, Duration::from_secs(50));

    // 7. RESERVE EXCEEDS THE BUDGET -- pool 0, nothing runs, no panic
    let over_reserved = Spec {
        lanes: vec![LaneSpec {
            points: vec![(30, 1.0)],
            prior: 1.0,
            zeroed: false,
            requires: None,
            floor: None,
        }],
        budget: 100,
        reserve: 500,
    };
    check(&over_reserved, "reserve-exceeds-budget");

    // 8. ZERO BUDGET, ZERO LANES
    let empty = Spec {
        lanes: Vec::new(),
        budget: 0,
        reserve: 0,
    };
    match allocate(&to_request(&empty)) {
        AllocationOutcome::Allocated(p) => assert!(p.grants().is_empty()),
        other => panic!("zero lanes must plan trivially: {other:?}"),
    }

    // 9. A PRECEDENCE CYCLE -- legal as a constraint (both or neither run)
    let cyc = Spec {
        lanes: vec![
            LaneSpec {
                points: vec![(50, 0.9)],
                prior: 1.0,
                zeroed: false,
                requires: Some(1),
                floor: None,
            },
            LaneSpec {
                points: vec![(50, 0.9)],
                prior: 1.0,
                zeroed: false,
                requires: Some(0),
                floor: None,
            },
        ],
        budget: 60,
        reserve: 0,
    };
    check(&cyc, "precedence-cycle");
    let p = match allocate(&to_request(&cyc)) {
        AllocationOutcome::Allocated(p) => p,
        other => panic!("{other:?}"),
    };
    assert!(
        p.grants().iter().all(|g| g.rung == 0),
        "a cycle that does not fit must run neither"
    );

    // 10. PRECEDENCE ON A ZEROED LANE drags the dependent to zero
    let dragged = Spec {
        lanes: vec![
            LaneSpec {
                points: vec![(50, 0.9)],
                prior: 1.0,
                zeroed: true,
                requires: None,
                floor: None,
            },
            LaneSpec {
                points: vec![(50, 0.9)],
                prior: 1.0,
                zeroed: false,
                requires: Some(0),
                floor: None,
            },
        ],
        budget: 500,
        reserve: 0,
    };
    check(&dragged, "zero-drags-dependent");

    // 11. (C5) THAT CANNOT FIT -> Infeasible, not a plan
    let cannot_fit = Spec {
        lanes: vec![LaneSpec {
            points: vec![(300, 1.0)],
            prior: 1.0,
            zeroed: false,
            requires: None,
            floor: Some(300),
        }],
        budget: 200,
        reserve: 0,
    };
    check(&cannot_fit, "c5-cannot-fit");
    assert!(matches!(
        allocate(&to_request(&cannot_fit)),
        AllocationOutcome::UseExistingPlan(FallOpen::Infeasible)
    ));

    // 12. MAX WIDTH ACCEPTED, ONE MORE REFUSED
    let at_max = Spec {
        lanes: (0..MAX_LANES)
            .map(|_| LaneSpec {
                points: vec![(10, 0.5)],
                prior: 0.5,
                zeroed: false,
                requires: None,
                floor: None,
            })
            .collect(),
        budget: 1000,
        reserve: 0,
    };
    check(&at_max, "at-max-lanes");
    let too_many = Spec {
        lanes: (0..=MAX_LANES)
            .map(|_| LaneSpec {
                points: vec![(10, 0.5)],
                prior: 0.5,
                zeroed: false,
                requires: None,
                floor: None,
            })
            .collect(),
        budget: 1000,
        reserve: 0,
    };
    assert!(matches!(
        allocate(&to_request(&too_many)),
        AllocationOutcome::UseExistingPlan(FallOpen::TooManyLanes { .. })
    ));

    // 13. FULL-DEPTH LADDER at MAX_RUNGS_PER_LANE
    let deep = Spec {
        lanes: vec![LaneSpec {
            points: (1..MAX_RUNGS_PER_LANE as u64)
                .map(|i| (i * 20, (i as f64) / (MAX_RUNGS_PER_LANE as f64)))
                .collect(),
            prior: 0.9,
            zeroed: false,
            requires: None,
            floor: None,
        }],
        budget: 400,
        reserve: 0,
    };
    check(&deep, "max-depth-ladder");
    assert!(
        CapLadder::caller_supplied(
            std::iter::once(Rung {
                cap: Duration::ZERO,
                reach: 0.0,
                origin: RungOrigin::DoNotRun
            })
            .chain((1..=MAX_RUNGS_PER_LANE as u64).map(|i| Rung {
                cap: Duration::from_secs(i),
                reach: 0.5,
                origin: RungOrigin::CallerSupplied
            }))
            .collect()
        )
        .is_err(),
        "one rung past the cap must be refused"
    );
}

/// The shipped 10 ms cap, measured over MANY instances rather than one.
#[test]
fn adversarial_solve_time_distribution_at_the_shipped_cap() {
    let mut rng = Rng::new(0x7157_7157_0001);
    let mut walls: Vec<Duration> = Vec::new();
    let mut fell_open = 0usize;
    let mut planned = 0usize;
    let mut infeasible = 0usize;
    // Realistic shapes: the source-declared ladders are 2-3 rungs, but size up
    // to the design note's 8-12 grid to find where the cap actually bites.
    for shape in [(3usize, 3usize), (5, 3), (5, 5), (8, 5), (5, 10), (8, 11)] {
        let (k, m) = shape;
        let mut shape_walls: Vec<Duration> = Vec::new();
        let mut shape_open = 0usize;
        for _ in 0..80 {
            let mut lanes = Vec::with_capacity(k);
            for i in 0..k {
                let mut cur = 0u64;
                let mut r = 0.0f64;
                let mut points = Vec::with_capacity(m);
                for _ in 0..m {
                    cur += 1 + rng.below(60);
                    r = (r + rng.unit() * (1.0 - r)).min(1.0);
                    points.push((cur, r));
                }
                lanes.push(LaneSpec {
                    points,
                    prior: rng.unit(),
                    zeroed: i == 0 && rng.below(3) == 0,
                    requires: if i > 0 && rng.below(4) == 0 {
                        Some(i - 1)
                    } else {
                        None
                    },
                    floor: None,
                });
            }
            let spec = Spec {
                lanes,
                budget: 480,
                reserve: 145,
            };
            // the SHIPPED 10 ms cap, not the generous test cap
            let shipped = shipped_request(&spec);
            let t = Instant::now();
            let outcome = allocate(&shipped);
            let w = t.elapsed();
            shape_walls.push(w);
            walls.push(w);
            match outcome {
                AllocationOutcome::Allocated(_) => planned += 1,
                AllocationOutcome::UseExistingPlan(FallOpen::Infeasible) => infeasible += 1,
                AllocationOutcome::UseExistingPlan(FallOpen::SolveDeadline)
                | AllocationOutcome::UseExistingPlan(FallOpen::NotOptimal) => {
                    fell_open += 1;
                    shape_open += 1;
                }
                other => panic!("unexpected outcome {other:?}"),
            }
        }
        shape_walls.sort_unstable();
        eprintln!(
            "  K={k:>2} rungs={m:>2} ({:>3} binaries): median {:>10?} p95 {:>10?} max {:>10?}  cap-misses {shape_open}/80",
            k * (m + 1),
            shape_walls[shape_walls.len() / 2],
            shape_walls[shape_walls.len() * 95 / 100],
            shape_walls[shape_walls.len() - 1]
        );
    }
    walls.sort_unstable();
    eprintln!(
        "SOLVE TIME at the shipped 10ms cap over {} instances: median {:?} p95 {:?} p99 {:?} MAX {:?}",
        walls.len(),
        walls[walls.len() / 2],
        walls[walls.len() * 95 / 100],
        walls[walls.len() * 99 / 100],
        walls[walls.len() - 1]
    );
    eprintln!(
        "  planned {planned}, provably infeasible {infeasible}, fell open on the cap {fell_open}"
    );
}

fn shipped_request(spec: &Spec) -> AllocationRequest {
    let lanes = spec
        .lanes
        .iter()
        .map(|l| {
            let mut rungs = vec![Rung {
                cap: Duration::ZERO,
                reach: 0.0,
                origin: RungOrigin::DoNotRun,
            }];
            for &(secs, reach) in &l.points {
                rungs.push(Rung {
                    cap: Duration::from_secs(secs),
                    reach,
                    origin: RungOrigin::CallerSupplied,
                });
            }
            let mut req = LaneRequest::new(CapLadder::caller_supplied(rungs).expect("ladder"))
                .with_reach_prior(l.prior);
            if l.zeroed {
                req = req.zeroed(
                    StructuralZero::flat_objective(1, ObjectiveRequirement::EstimatedGradient)
                        .unwrap(),
                );
            }
            if let Some(p) = l.requires {
                req = req.requiring(p);
            }
            req
        })
        .collect();
    // no .with_solve_cap: the SHIPPED default
    AllocationRequest::new(
        lanes,
        Duration::from_secs(spec.budget),
        Duration::from_secs(spec.reserve),
    )
}

// ---------------------------------------------------------------------------
// The worst case the sweep above found, characterised.
// ---------------------------------------------------------------------------

fn ol_ladder(points: &[(u64, f64)]) -> CapLadder {
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

/// random#461: 3 lanes / 14 binaries, pool 174 s, all priors 0, (C5) floors
/// that cannot both be met -> provably infeasible.
fn worst(prior: f64) -> Vec<LaneRequest> {
    vec![
        LaneRequest::new(ol_ladder(&[
            (111, 0.1),
            (133, 0.2),
            (191, 0.3),
            (310, 0.4),
            (407, 0.5),
        ]))
        .with_reach_prior(prior)
        .no_worse_than(Duration::from_secs(133)),
        LaneRequest::new(ol_ladder(&[(36, 0.1), (111, 0.2), (179, 0.3), (206, 0.4)]))
            .with_reach_prior(prior)
            .requiring(0),
        LaneRequest::new(ol_ladder(&[(84, 0.1), (97, 0.2)]))
            .with_reach_prior(prior)
            .no_worse_than(Duration::from_secs(97)),
    ]
}

#[test]
fn the_worst_case_solve_is_a_timer_not_a_search() {
    for cap_ms in [10u64, 50, 200, 1000, 2000, 5000] {
        let req = AllocationRequest::new(worst(0.0), Duration::from_secs(174), Duration::ZERO)
            .with_solve_cap(Duration::from_millis(cap_ms));
        let t = Instant::now();
        let outcome = allocate(&req);
        let w = t.elapsed();
        eprintln!("  solve_cap {cap_ms:>5} ms -> wall {w:>14?}  outcome {outcome:?}");
    }
    eprintln!("--- same shape, NONZERO priors (a real objective) ---");
    for cap_ms in [10u64, 1000, 5000] {
        let req = AllocationRequest::new(worst(0.7), Duration::from_secs(174), Duration::ZERO)
            .with_solve_cap(Duration::from_millis(cap_ms));
        let t = Instant::now();
        let outcome = allocate(&req);
        eprintln!(
            "  solve_cap {cap_ms:>5} ms -> wall {:>14?}  outcome {outcome:?}",
            t.elapsed()
        );
    }
    eprintln!("--- same shape, FEASIBLE (pool raised to 400 s) ---");
    for prior in [0.0f64, 0.7] {
        let req = AllocationRequest::new(worst(prior), Duration::from_secs(400), Duration::ZERO)
            .with_solve_cap(Duration::from_secs(5));
        let t = Instant::now();
        let outcome = allocate(&req);
        let ok = matches!(outcome, AllocationOutcome::Allocated(_));
        eprintln!("  prior {prior} -> wall {:>14?}  planned={ok}", t.elapsed());
    }
}
