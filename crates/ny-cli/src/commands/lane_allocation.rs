// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! LAYER A of the per-instance lane budget allocator, WIRED into the attack
//! slice of `run_and_translate`.
//!
//! The allocator itself is `ny_mip::lane_allocation` (the multiple-choice
//! knapsack, its cap ladders and its fail-open contract). This module is the
//! seam: it turns the three lanes that actually hold the attack-slice budget
//! into an [`ny_mip::AllocationRequest`], measures the ONE structural zero
//! Layer A can establish per instance, and hands each lane the cap the
//! allocator committed to instead of the private fraction the lane would
//! otherwise take.
//!
//! Design: `docs/LANE_BUDGET_ALLOCATION_MCKP_2026-08-19.md` and
//! `docs/LANE_BUDGET_OPTIMIZER_DESIGN_2026-08-19.md` §4.
//!
//! # Darkness
//!
//! Everything below is behind [`lane_budget_allocator_armed`]
//! (`NY_LANE_BUDGET_ALLOCATOR`, declared in `ny-levers`). Disarmed, the caller
//! never takes the objective probe, never builds a request and never enters the
//! solver; every lane's window comes back [`LaneWindow::Private`] and each lane
//! derives exactly the window it derives today, from the same private helper,
//! in the same order. That is asserted directly on the decision function by
//! `disarmed_every_lane_keeps_its_private_fraction`, not on a log line.
//!
//! # How this COMPOSES with `#lane-value-scheduler`
//!
//! It does not replace it. `crates/ny-cli/src/commands/lane_schedule.rs` is a
//! strict pipeline walk that handles yields IN FLIGHT — a stalled lane's
//! unspent seconds re-enter the pool as the next lane's cap. This module
//! chooses the caps JOINTLY and UP FRONT, which a greedy marginal-value walk
//! provably cannot do when a lane's value is step-like in its cap: every block
//! before the step has marginal value zero, so greedy never climbs it.
//!
//! The two are mutually exclusive per instance, and the caller enforces it: if
//! the scheduler is armed it owns the two BNN lanes and this allocator stands
//! down. There is exactly one authority over a lane's cap at any time.
//!
//! # Soundness
//!
//! Budget only. Every lane reached from here returns a CLAIMED counterexample
//! INPUT and never a verdict; the claim becomes a `sat` only by passing the
//! unchanged `gate_sat_with_trusted_oracle`, which this module does not touch
//! and cannot name. Nothing here can produce an `unsat`. A bad allocation
//! therefore costs a MISSED row and never a WRONG one.
//!
//! # The two invariants the wiring adds on top of the allocator's own
//!
//! **I-POOL — the pool is the sum of the caps TODAY'S fixed fractions would
//! hand the SAME lanes.** Not `remaining - 45 s`, and not the whole budget.
//! The knapsack may therefore only REDISTRIBUTE seconds inside the attack
//! slice; it can never take a second from the branch-and-bound residual
//! claimant downstream of it, which is what makes "no row may exceed its
//! official budget" true by arithmetic rather than by review. Every second the
//! knapsack does not grant flows to that residual claimant exactly as it does
//! today. `the_pool_never_exceeds_todays_attack_slice` and
//! `grants_plus_reserve_never_exceed_the_budget` assert it on randomized input.
//!
//! **I-FLOOR — (C5), with exactly one measured relaxation, named.** Every lane
//! without a structural zero carries a no-regression floor equal to today's
//! cap, so today's plan is a feasible point and the optimum is no worse than
//! today by construction. The single exception is the LP sign-space lane, which
//! may be floored as low as [`LP_LANE_MEASURED_FLOOR`] — see that constant for
//! the measured control row that licenses it, and for why relaxing it cannot
//! cost the rows that lane wins.

//! # WHERE THE "UNCLAIMED" SECONDS ACTUALLY GO — read before trying to spend them
//!
//! The per-lane traffic ledger shows 24.03 s and 27.19 s of a 480 s budget
//! reaching no lane at all. Both numbers decompose exactly, and NEITHER is a
//! budget-allocation defect:
//!
//! **24 s of it is the publication flush grace, and it is a RESERVE.**
//! `vnncomp::internal_timeout_secs` hands the internal verifier
//! `budget - max(budget / 20, 5)`; at 480 s that grace is exactly 24 s, which is
//! the 24.03 s on `model_64_idx_1703_eps_1` to within its rounding. The interval
//! exists so the JSON verdict is translated and `RESULTS_FILE` is published
//! before the scored budget elapses. Spending it on a lane is how a row lands
//! ABOVE its budget and becomes an un-measurable `error` row, and one `error`
//! row makes a whole family unbankable — traffic's longest row is already
//! 454.8 s of 480 s with 0 errors. So it is not handed to a lane. What this
//! module changes is that it is now NAMED: the pool is today's attack-slice
//! claim, the reserve is everything outside it, and the ledger prints both, so
//! the seconds are accounted rather than merely absent.
//!
//! **The rest is branch-and-bound returning early into a post-BaB lane that
//! was never entered — and the gate is the HOST, not the budget.**
//! `model_48_idx_1703_eps_1` gives BaB a 50 s grant, BaB spends 46.78 s, and
//! the log carries ZERO `Post-BaB` lines. The entry condition is
//! `deferred_pgd_consumer_available && postbab_escalation_allowed(..)`. The
//! verdict is `timeout` with no terminal ingress, so the second conjunct holds;
//! the first is
//! `route.attack_enabled() && margin_row_memory_allowed && !safenlp_direct_mip_first`,
//! and `margin_row_memory_allowed` is `optional_heavy_memory_allowed_now()`.
//! On a NON-LINUX host `ny_propagate::network::crown_memory::process_memory_envelope`
//! returns `ProcessMemoryEnvelope::Unavailable` by construction, and
//! `Unavailable` maps to `false`. So on the development host the post-BaB
//! attack lane, the concurrent margin-row lane and BOTH their reserves are
//! suppressed together, and the tail cannot be claimed by anything.
//!
//! That is a live-memory admission boundary for graph-heavy optional tails,
//! reached identically at every cap this module could choose, so no allocation
//! opens it. On a Linux competition box with the documented 24/32/80 GiB
//! envelope the block IS entered and `try_postbab_falsify` claims exactly those
//! seconds. Reproduced here on a real 120 s `cifar100_2024` row: the internal
//! verifier returned at 102.27 s leaving 13.68 s, verdict `timeout`, and no
//! `Post-BaB` line was emitted.
//!
//! The remaining honest recovery is the design note's (C3) — replace the fixed
//! `budget / 20` grace with a TIMED publication cost — and the seam for it is
//! already the `reserve` argument of `ny_mip::AllocationRequest::new`. It needs
//! a measured publication cost, not a cap change, so it is not done here.

// Without `mip` the allocator crate is not linked, so the planning surface has
// no consumer. The arming predicate is still read (the call site is
// feature-independent). Say so rather than accumulate warnings.
#![cfg_attr(not(feature = "mip"), allow(dead_code))]

use std::time::{Duration, Instant};

/// Whether Layer A of the lane budget allocator is admitted.
///
/// Exact `"1"` arms it, exact `"0"` disarms it, every other byte sequence is a
/// recorded rejection falling back to the declaration's `false`. Fails CLOSED.
pub(crate) fn lane_budget_allocator_armed() -> bool {
    ny_levers::read(&ny_levers::decls::dark_probes::LANE_BUDGET_ALLOCATOR)
        .value
        .as_bool()
}

/// The arming rule as a pure predicate over one raw environment string.
///
/// Same declaration, same parser, same chokepoint — only the lookup is
/// injected — so a test of this is a test of the production rule.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn lane_budget_allocator_armed_from(raw: Option<&str>) -> bool {
    let owned = raw.map(str::to_owned);
    ny_levers::read_with(
        &ny_levers::decls::dark_probes::LANE_BUDGET_ALLOCATOR,
        move |_| owned,
    )
    .value
    .as_bool()
}

/// Whether Layer A may plan this instance at all.
///
/// Three conditions, all of which must hold, expressed as one pure predicate so
/// the composition rule is testable rather than spread across an `if`:
///
/// * `armed` — the lever. Dark by default, so this is `false` on every scored
///   run and the other two are never even consulted.
/// * `!scheduler_armed` — `#lane-value-scheduler` owns the two BNN lanes in
///   flight when it is armed. Exactly one mechanism may hold a lane's cap.
/// * `!traffic_terminal_softmax_peel` — that route REWRITES the objective the
///   upfront lane steers on, so the in-box probe would be measuring a function
///   that lane is not using, and a structural zero drawn from it would be a
///   category error rather than a fact.
pub(crate) const fn allocator_admitted(
    armed: bool,
    scheduler_armed: bool,
    traffic_terminal_softmax_peel: bool,
) -> bool {
    armed && !scheduler_armed && !traffic_terminal_softmax_peel
}

// ---------------------------------------------------------------------------
// What a lane is told
// ---------------------------------------------------------------------------

/// The three lanes that hold the attack-slice budget.
///
/// A LANE name is the identity of a piece of ny's own code. Nothing in this
/// module keys on a benchmark, a category, a directory, a filename or a preset
/// key; the only per-instance input is the structural objective probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AllocatedLane {
    /// LP-guided sign-space search over binarized `Sign` conv suffixes.
    SignSpace,
    /// Straight-through-estimator PGD.
    StePgd,
    /// The upfront exact-gradient DLR-APGD attack.
    UpfrontAttack,
}

/// What the caller does with one lane's window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LaneWindow {
    /// No allocation is in force: the lane derives exactly the window it
    /// derives today, from its own private helper. This is the ONLY value
    /// reachable with the lever disarmed.
    Private,
    /// The allocator committed this lane to this cap. The lane is handed it
    /// through its own `*_granted` seam and plans its schedule against it.
    Cap(Duration),
    /// The allocator granted this lane ZERO seconds. It is SKIPPED — not run
    /// under a small cap — and its seconds appear in another lane's grant or
    /// in the residual that flows to branch-and-bound.
    Skip,
}

/// The caps today's fixed fractions would hand each lane, in today's order,
/// against today's deadline.
///
/// `None` means the lane is not consulted on this instance at all (disarmed,
/// or its own helper already declined the remaining budget as unusable), in
/// which case it is not a decision variable and contributes nothing to the
/// pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct TodaySlices {
    /// `lp_lane_plan(remaining)` — `(remaining - 45 s) * 0.5`, capped, floored.
    pub(crate) sign_space: Option<Duration>,
    /// `ste_lane_plan(remaining)` — `remaining - 45 s - 100 s`, capped.
    pub(crate) ste_pgd: Option<Duration>,
    /// The upfront lane's own window rule for this instance.
    pub(crate) upfront: Option<Duration>,
}

impl TodaySlices {
    /// The pool: the sum of the caps today's fixed fractions would hand these
    /// lanes, in WHOLE SECONDS (the unit the knapsack row is solved in).
    ///
    /// Whole seconds are taken per lane and then summed, never the other way
    /// round: `floor(a) + floor(b) + floor(c) <= floor(a + b + c)`, so the pool
    /// can only ever be at or below what today's plan claims. That direction is
    /// I-POOL.
    pub(crate) fn pool(self) -> Duration {
        [self.sign_space, self.ste_pgd, self.upfront]
            .into_iter()
            .flatten()
            .fold(Duration::ZERO, |sum, cap| {
                sum + Duration::from_secs(cap.as_secs())
            })
    }
}

/// A committed attack-slice plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttackSliceAllocation {
    sign_space: LaneWindow,
    ste_pgd: LaneWindow,
    upfront: LaneWindow,
    pool: Duration,
    granted: Duration,
    ledger: String,
}

impl AttackSliceAllocation {
    /// The window committed for `lane`.
    pub(crate) fn window(&self, lane: AllocatedLane) -> LaneWindow {
        match lane {
            AllocatedLane::SignSpace => self.sign_space,
            AllocatedLane::StePgd => self.ste_pgd,
            AllocatedLane::UpfrontAttack => self.upfront,
        }
    }

    /// The pool the knapsack was allowed to spend.
    pub(crate) fn pool(&self) -> Duration {
        self.pool
    }

    /// Seconds committed to lanes.
    pub(crate) fn granted(&self) -> Duration {
        self.granted
    }

    /// Seconds inside the pool that no lane claimed. They are not lost and they
    /// are not unowned: the attack slice simply returns early, and every one of
    /// them reaches the branch-and-bound residual claimant through the same
    /// `attack_start` subtraction that charges the lanes today.
    pub(crate) fn residual_to_bab(&self) -> Duration {
        self.pool.saturating_sub(self.granted)
    }

    /// One line, for the log and the flight receipt.
    pub(crate) fn ledger(&self) -> &str {
        &self.ledger
    }
}

/// The window for `lane` under an optional allocation.
///
/// This is the ONE function every call site consults, so "the lever is unset
/// ⇒ byte-identical behaviour" is a property of a single pure function rather
/// than of three scattered `if` statements.
pub(crate) fn lane_window(
    allocation: Option<&AttackSliceAllocation>,
    lane: AllocatedLane,
) -> LaneWindow {
    match allocation {
        None => LaneWindow::Private,
        Some(plan) => plan.window(lane),
    }
}

// ---------------------------------------------------------------------------
// The objective probe — the one structural zero Layer A can measure per row
// ---------------------------------------------------------------------------

/// Probe points evaluated before a FLAT verdict may be returned.
///
/// The in-tree precedent evaluates a net "at 33 points in the input box (both
/// corners, center, 30 random)" (`docs/YOLO_CONJUNCTION_AND_RELAXATION_GAP_2026-07-28.md`).
/// Same shape here, with the random points made DETERMINISTIC (see
/// [`probe_points`]) so a probe is reproducible from the box alone.
pub(crate) const OBJECTIVE_PROBE_POINTS: usize = 33;

/// Fewest points that may support a FLAT verdict.
///
/// A `Flat` tier zeroes a lane, so it must never rest on one or two forwards
/// that happened to fit inside the wall. If the probe runs out of wall before
/// this many points, the verdict is INCONCLUSIVE and no lane is zeroed.
pub(crate) const OBJECTIVE_PROBE_MIN_POINTS: usize = 8;

/// Wall for the whole probe.
///
/// 0.05 % of a 480 s instance and 0.25 % of a 100 s one. A NON-flat objective
/// exits at the second distinct value, so this wall is only ever paid in full
/// by an objective that is genuinely constant over the box — which is the case
/// the probe exists to establish.
pub(crate) const OBJECTIVE_PROBE_WALL: Duration = Duration::from_millis(250);

/// What the in-box probe found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ObjectiveProbe {
    /// Distinct f32 objective values observed.
    pub(crate) distinct_values: usize,
    /// Points actually evaluated.
    pub(crate) points: usize,
}

impl ObjectiveProbe {
    /// Whether this probe establishes a FLAT objective tier.
    ///
    /// One distinct f32 value, over at least [`OBJECTIVE_PROBE_MIN_POINTS`]
    /// points. Anything else is not flat, and "not established" and "not flat"
    /// are deliberately the same answer here: both leave every lane alone.
    pub(crate) fn is_flat(self) -> bool {
        self.distinct_values <= 1 && self.points >= OBJECTIVE_PROBE_MIN_POINTS
    }
}

/// Deterministic in-box probe points: both corners, the centre, then interior
/// points from a fixed integer hash of `(coordinate, point index)`.
///
/// Deterministic on purpose. A probe that decides whether a lane runs must be
/// reproducible from the box alone: two runs of the same instance must take the
/// same points, so a `Flat` verdict can be re-derived rather than re-rolled.
pub(crate) fn probe_points(box_lo: &[f32], box_hi: &[f32], points: usize) -> Vec<Vec<f32>> {
    let dims = box_lo.len().min(box_hi.len());
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(points);
    for t in 0..points {
        let point: Vec<f32> = (0..dims)
            .map(|i| {
                let (lo, hi) = (box_lo[i], box_hi[i]);
                match t {
                    0 => lo,
                    1 => hi,
                    2 => lo + (hi - lo) * 0.5,
                    _ => {
                        let frac = interior_fraction(i, t);
                        lo + (hi - lo) * frac
                    }
                }
            })
            .collect();
        out.push(point);
    }
    out
}

/// SplitMix64-style integer hash, used only to place interior probe points.
fn interior_fraction(coordinate: usize, point: usize) -> f32 {
    let mut z = (coordinate as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((point as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // 24 bits, so the fraction is exact in f32.
    ((z >> 40) as f32) / 16_777_216.0
}

/// Count the distinct f32 objective values over the probe points, exiting at
/// the FIRST second distinct value.
///
/// `margin_at` returns the objective at a point, or `None` if it could not be
/// evaluated. Any non-finite or unevaluable point makes the whole probe
/// INCONCLUSIVE: a `-inf` surrogate means the margin function could not express
/// the constraint, which is not evidence that the objective is constant.
pub(crate) fn probe_objective<F>(
    points: &[Vec<f32>],
    wall: Duration,
    started: Instant,
    mut margin_at: F,
) -> Option<ObjectiveProbe>
where
    F: FnMut(&[f32]) -> Option<f64>,
{
    let mut seen: Vec<u32> = Vec::with_capacity(4);
    let mut evaluated = 0usize;
    for point in points {
        if started.elapsed() >= wall {
            break;
        }
        let margin = margin_at(point)?;
        if !margin.is_finite() {
            return None;
        }
        let bits = (margin as f32).to_bits();
        evaluated += 1;
        if !seen.contains(&bits) {
            seen.push(bits);
        }
        if seen.len() > 1 {
            // Not flat. Nothing downstream distinguishes 2 from 33, so stop.
            break;
        }
    }
    (evaluated > 0).then_some(ObjectiveProbe {
        distinct_values: seen.len(),
        points: evaluated,
    })
}

/// Clamp a COMMITTED cap to the LIVE remaining budget before handing it over.
///
/// The allocator planned against a prediction of where each lane starts; the
/// published deadline is the fact. An upstream lane that overran its own cap
/// must not be able to push a downstream lane past the scored budget, because
/// a row that exceeds its budget is an un-measurable `error` row and an `error`
/// row makes a whole family unbankable. Clamping can only ever SHRINK a cap, so
/// it cannot itself cause an overrun.
///
/// `remaining == None` means no deadline is published — information absent,
/// never budget spent — and the committed cap is kept unchanged, which is what
/// every lane's own helper does in that case.
///
/// `margin` is the lane's own publication/confirmation reserve, passed in
/// rather than duplicated here: the BNN lanes reserve
/// `sign_space_falsify::LANE_PUBLICATION_MARGIN` and the upfront lane reserves
/// `UPFRONT_ATTACK_SAFETY_MARGIN`, and those are different numbers for good
/// reasons that belong to those lanes.
///
/// Returns `None` when nothing usable is left, which the call sites treat as
/// "do not consult this lane" — the same answer their own helpers give.
pub(crate) fn clamp_to_live_remaining(
    cap: Duration,
    remaining: Option<Duration>,
    margin: Duration,
) -> Option<Duration> {
    let Some(remaining) = remaining else {
        return Some(cap);
    };
    let usable = remaining.checked_sub(margin)?;
    let clamped = cap.min(usable);
    (!clamped.is_zero()).then_some(clamped)
}

// ---------------------------------------------------------------------------
// The plan
// ---------------------------------------------------------------------------

/// The lowest cap the LP sign-space lane may be floored to, relaxing (C5).
///
/// # The measured control row that licenses this, and it is the whole safety
/// argument
///
/// (C5) says a lane without a structural zero may not be granted less than
/// today's fixed slice, so that today's plan stays a feasible point. This is
/// the ONE lane where that floor is relaxed, and it is relaxed to a MEASURED
/// number rather than to zero.
///
/// `traffic_signs_recognition_2023 model_30_idx_1703_eps_1` is a row ny WINS.
/// Measured per-lane, sequential, one process, 480 s budget: the LP sign-space
/// lane used **131.04 s of its 217.5 s cap** and produced the candidate; every
/// later lane was never reached. THE CAP IS NOT BINDING WHERE THE LANE WORKS.
/// On the rows where it does not work it is pure waste: `model_48_idx_1703_eps_1`
/// held the full 217.52 s for 370 LP solves, 34 accepted flips, best pattern
/// margin -82 and NO candidate.
///
/// So a cap at or above ~131 s cannot cost the row the lane wins, and 140 s is
/// that number with a margin. Below it this module will not go: the floor is
/// `min(today's cap, 140 s)`, so on an instance whose today-cap is already
/// under 140 s nothing is relaxed at all.
///
/// This is the only relaxation of (C5) in the wiring. Widening it, or applying
/// it to another lane, needs its own control row.
pub(crate) const LP_LANE_MEASURED_FLOOR: Duration = Duration::from_secs(140);

/// Why the allocator was not consulted, or did not produce a plan. Every
/// variant means "run exactly today's plan".
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NoAllocation {
    /// No lane in the attack slice is consulted on this instance.
    NoLanes,
    /// A ladder the wiring built did not validate.
    MalformedLadder(String),
    /// The allocator declined; its typed reason, rendered.
    FellOpen(String),
    /// The returned plan did not survive the wiring's own re-check.
    RejectedInReadback(String),
}

#[cfg(feature = "mip")]
pub(crate) use mip_plan::plan_attack_slice;

#[cfg(feature = "mip")]
mod mip_plan {
    use super::{
        AllocatedLane, AttackSliceAllocation, LaneWindow, NoAllocation, TodaySlices,
        LP_LANE_MEASURED_FLOOR,
    };
    use std::time::Duration;

    use ny_mip::lane_allocation::{
        allocate, declared_ladder, AllocationOutcome, AllocationRequest, CapLadder, Lane,
        LaneRequest, ObjectiveRequirement, Rung, RungOrigin, StructuralZero,
    };

    /// A budget large enough that `declared_ladder` is never truncated by it,
    /// so the ladder it returns is the lane's UNTRUNCATED source-declared one.
    /// Used only to read the lane's floor and its declared schedule wall.
    const UNTRUNCATED: Duration = Duration::from_hours(24);

    /// The lane's source-declared `(floor, schedule wall)`, read through
    /// `ny_mip::declared_ladder` so the citations stay in one place.
    fn declared_bounds(lane: Lane) -> Option<(Duration, Duration)> {
        let ladder = declared_ladder(lane, UNTRUNCATED).ok()?;
        let floor = ladder.floor()?;
        let wall = ladder.rungs().last()?.cap;
        Some((floor, wall))
    }

    /// Build one lane's ladder from a candidate cap set.
    ///
    /// `reach` is `cap / declared_wall`: the fraction of the lane's own
    /// declared schedule this cap buys. That is not an invented profile, it is
    /// each lane's own rule restated — the LP walk's work is
    /// `min(cap / per_lp_time, max_lp_solves)` and `per_lp_time` is a declared
    /// constant, so its work is linear in the wall below the LP-count cap; the
    /// STE lane is an anytime search whose only declared budget IS its wall.
    /// What is NOT claimed, and what the ladder deliberately does not encode,
    /// is where the STE restart boundaries fall — the source declares no wall
    /// per restart, so those caps are not derivable and are Layer B.
    fn ladder_from(
        caps: &[Duration],
        declared_wall: Duration,
        pool: Duration,
    ) -> Option<CapLadder> {
        let wall_secs = declared_wall.as_secs().max(1) as f64;
        let mut secs: Vec<u64> = caps
            .iter()
            .map(|c| c.as_secs())
            .filter(|&s| s > 0 && s <= pool.as_secs())
            .collect();
        secs.sort_unstable();
        secs.dedup();
        let mut rungs = vec![Rung {
            cap: Duration::ZERO,
            reach: 0.0,
            origin: RungOrigin::DoNotRun,
        }];
        for s in secs {
            rungs.push(Rung {
                cap: Duration::from_secs(s),
                reach: ((s as f64) / wall_secs).clamp(0.0, 1.0),
                origin: RungOrigin::CallerSupplied,
            });
        }
        CapLadder::caller_supplied(rungs).ok()
    }

    /// Choose the attack slice's caps jointly, by solving the Layer-A knapsack.
    ///
    /// `objective_is_flat` is the structural probe's verdict on the objective
    /// the UPFRONT exact-gradient lane steers on (see the caller). The two BNN
    /// lanes are NOT zeroed by it: their extraction strips a terminal Softmax
    /// and they steer on the pre-Softmax integer logit margin, which is a
    /// different function from the one the probe measured. Zeroing a lane on
    /// evidence about someone else's objective is exactly the category error
    /// this module must not make.
    pub(crate) fn plan_attack_slice(
        today: TodaySlices,
        objective_is_flat: bool,
        solve_cap: Duration,
    ) -> Result<AttackSliceAllocation, NoAllocation> {
        let pool = today.pool();
        if pool.is_zero() {
            return Err(NoAllocation::NoLanes);
        }

        // Column order is fixed and is the lane's identity in the request.
        let mut lanes: Vec<LaneRequest> = Vec::with_capacity(3);
        let mut identity: Vec<AllocatedLane> = Vec::with_capacity(3);

        // A lane whose today-cap does not reach one WHOLE SECOND is not a
        // decision variable. The knapsack row is solved in whole seconds — a
        // cap is a schedule and sub-second granularity is meaningless against
        // budgets of 30-1800 s — so such a lane cannot be REPRESENTED, and the
        // only honest thing to do with a lane the model cannot represent is to
        // leave it exactly where it is. Making it a column instead would floor
        // its cap to zero and SKIP a lane today runs, which is a regression the
        // model would be inventing out of its own rounding.
        let decidable = |cap: Option<Duration>| cap.filter(|c| c.as_secs() >= 1);

        if let Some(today_cap) = decidable(today.sign_space) {
            let (floor, wall) = declared_bounds(Lane::BnnSignSpace)
                .ok_or_else(|| NoAllocation::MalformedLadder("bnn_sign_space".to_string()))?;
            // Candidate rungs: the lane's own floor, the measured (C5)
            // relaxation point, today's cap, and its declared schedule wall.
            let caps = [floor, LP_LANE_MEASURED_FLOOR, today_cap, wall];
            let ladder = ladder_from(&caps, wall, pool)
                .ok_or_else(|| NoAllocation::MalformedLadder("bnn_sign_space".to_string()))?;
            // I-FLOOR, with its one named relaxation.
            let c5 = today_cap.min(LP_LANE_MEASURED_FLOOR);
            lanes.push(LaneRequest::new(ladder).no_worse_than(c5));
            identity.push(AllocatedLane::SignSpace);
        }

        if let Some(today_cap) = decidable(today.ste_pgd) {
            let (_, wall) = declared_bounds(Lane::BnnStePgd)
                .ok_or_else(|| NoAllocation::MalformedLadder("bnn_ste_pgd".to_string()))?;
            let caps = [today_cap, wall];
            let ladder = ladder_from(&caps, wall, pool)
                .ok_or_else(|| NoAllocation::MalformedLadder("bnn_ste_pgd".to_string()))?;
            lanes.push(LaneRequest::new(ladder).no_worse_than(today_cap));
            identity.push(AllocatedLane::StePgd);
        }

        if let Some(today_cap) = decidable(today.upfront) {
            // Layer A cannot read this lane's ladder off source — its window is
            // a policy over predicted cost, and `declared_ladder` says so — so
            // the caller supplies the one rung it actually has, today's cap,
            // and makes no other claim. Its only decision here is run/skip.
            let ladder = ladder_from(&[today_cap], today_cap, pool)
                .ok_or_else(|| NoAllocation::MalformedLadder("upfront_attack".to_string()))?;
            let mut request = LaneRequest::new(ladder);
            // The ONE structural zero Layer A measures per instance. It is a
            // hard p = 0: an exact-gradient search on an objective that takes a
            // single f32 value over the whole probe is not slow, it is BLIND.
            match StructuralZero::flat_objective(
                if objective_is_flat { 1 } else { 2 },
                ObjectiveRequirement::Exact,
            ) {
                Some(zero) => {
                    request = request.zeroed(zero);
                }
                None => {
                    request = request.no_worse_than(today_cap);
                }
            }
            lanes.push(request);
            identity.push(AllocatedLane::UpfrontAttack);
        }

        if lanes.is_empty() {
            return Err(NoAllocation::NoLanes);
        }

        let request = AllocationRequest::new(lanes, pool, Duration::ZERO).with_solve_cap(solve_cap);
        let plan = match allocate(&request) {
            AllocationOutcome::Allocated(plan) => plan,
            AllocationOutcome::UseExistingPlan(reason) => {
                return Err(NoAllocation::FellOpen(format!("{reason:?}")))
            }
        };

        let mut windows = [LaneWindow::Private; 3];
        let mut granted = Duration::ZERO;
        for grant in plan.grants() {
            let Some(&lane) = identity.get(grant.lane) else {
                return Err(NoAllocation::RejectedInReadback(
                    "grant names a lane that is not in the request".to_string(),
                ));
            };
            let window = if grant.cap.is_zero() {
                LaneWindow::Skip
            } else {
                LaneWindow::Cap(grant.cap)
            };
            granted += grant.cap;
            windows[lane_index(lane)] = window;
        }
        // I-POOL, re-checked here and not merely trusted: the allocator already
        // proves it, and a defect there must cost a missed row, never an
        // over-spent budget.
        if granted > pool {
            return Err(NoAllocation::RejectedInReadback(format!(
                "grants {granted:?} exceed the pool {pool:?}"
            )));
        }

        let ledger = format!(
            "lane allocation: pool {:.2}s, granted {:.2}s, residual to BaB {:.2}s \
             [sign_space {}, ste_pgd {}, upfront {}] (objective tier {})",
            pool.as_secs_f64(),
            granted.as_secs_f64(),
            pool.saturating_sub(granted).as_secs_f64(),
            render(windows[0]),
            render(windows[1]),
            render(windows[2]),
            if objective_is_flat {
                "FLAT"
            } else {
                "not flat"
            },
        );

        Ok(AttackSliceAllocation {
            sign_space: windows[0],
            ste_pgd: windows[1],
            upfront: windows[2],
            pool,
            granted,
            ledger,
        })
    }

    const fn lane_index(lane: AllocatedLane) -> usize {
        match lane {
            AllocatedLane::SignSpace => 0,
            AllocatedLane::StePgd => 1,
            AllocatedLane::UpfrontAttack => 2,
        }
    }

    fn render(window: LaneWindow) -> String {
        match window {
            LaneWindow::Private => "not requested".to_string(),
            LaneWindow::Skip => "SKIPPED (0s)".to_string(),
            LaneWindow::Cap(cap) => format!("{:.2}s", cap.as_secs_f64()),
        }
    }
}

#[cfg(test)]
#[path = "lane_allocation_tests.rs"]
mod lane_allocation_tests;
