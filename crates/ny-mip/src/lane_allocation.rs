// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! LAYER A of the per-instance lane budget allocator: a multiple-choice
//! knapsack over per-lane cap ladders, solved exactly on the ay backend.
//!
//! Design: `docs/LANE_BUDGET_ALLOCATION_MCKP_2026-08-19.md` (the formulation)
//! and `docs/LANE_BUDGET_OPTIMIZER_DESIGN_2026-08-19.md` §4 (the MILP).
//!
//! # UNWIRED
//!
//! Nothing calls this module. It is a standalone allocator with its own tests;
//! it is deliberately NOT plumbed into the scored path, which is why it needs
//! no lever, no environment variable and no default change. It becomes wiring
//! only when the concurrent `#lane-value-scheduler` mechanism lands and a
//! caller replaces `Registry::schedule`'s proportional split with a call here.
//!
//! # What "Layer A" means
//!
//! **There is not one fitted coefficient in this file.** Everything that ends
//! up as an objective weight comes from exactly three places:
//!
//! 1. **Structural zeros** ([`StructuralZero`]) — hard `p_k = 0`, derived from
//!    a property of the instance, never from a measurement. Constructing one
//!    requires presenting the structural evidence, so a caller cannot assert a
//!    zero it cannot justify.
//! 2. **A declared step ladder** read off each lane's own schedule source
//!    ([`declared_ladder`]). Every rung carries a [`SourceCitation`] naming the
//!    file, the constant and its literal value. A lane whose ladder is not
//!    determinable from source returns [`UnknownLadder`] rather than an
//!    invented grid.
//! 3. **One uniform prior scalar**, [`DEFAULT_LANE_REACH_PRIOR`], identical for
//!    every lane. Being identical is the point: a scalar that is the same for
//!    all lanes cannot encode a preference between them, so it cannot smuggle
//!    in a benchmark-tuned ranking. It is the mean of a Beta(1,1) prior over an
//!    unobserved binary outcome, and it is not tuned against anything.
//!
//! Fitting the per-lane, per-bucket scalar `a_k(phi)` is Layer B and lives
//! nowhere in this crate.
//!
//! # The formulation
//!
//! Lanes `k = 1..K`, each with a cap grid `G_k = {0 = g[k][0] < ... < g[k][m]}`.
//! `g[k][0] = 0` means DO NOT RUN; every nonzero rung is at or above the lane's
//! own floor, so a lane can never be handed a dribble of leftover seconds.
//!
//! ```text
//! x[k][j] in {0,1}   = 1 iff lane k is committed to cap g[k][j]
//!
//! min  SUM_k SUM_j c[k][j] x[k][j]     c[k][j] = ln(1 - p_k(g[k][j])) <= 0
//!                                      c[k][0] = 0
//! (E)  SUM_j x[k][j] = 1                              one rung per lane
//! (B)  SUM_k SUM_j g[k][j] x[k][j] <= B - R           the pool
//! (P)  SUM_{j>=1} x[k][j] <= SUM_{j>=1} x[p][j]       k needs p's output
//! (A)  x[k][0] = 1 for every structurally zeroed lane (a BOUND FIX, not a row)
//! (C5) SUM_j g[k][j] x[k][j] >= a_k_today             optional no-regression
//! ```
//!
//! Minimising `SUM ln(1 - p)` is exactly maximising `1 - PROD (1 - p_k)`.
//!
//! **The log is the point.** Because the grid is discrete, `c[k][j]` is just a
//! number per rung: no concavity, no monotonicity and no differentiability in
//! `b` is assumed or needed. A curve that is 0.02 at one rung, 0.35 at the next
//! and flat above is represented EXACTLY and optimised EXACTLY. A greedy
//! marginal-value rule cannot climb such a curve — every rung before the step
//! has marginal value ~0 — and a continuous relaxation rounds through the step.
//! `lane_allocation_tests::greedy_marginal_value_misses_a_late_step` builds
//! that case and runs both rules side by side.
//!
//! # SOUNDNESS: the API cannot express an answer
//!
//! Allocation is budget-neutral in the strongest possible sense: it selects
//! which lane runs and under what cap and it can never change what a lane may
//! publish. That is made STRUCTURAL here rather than promised in a comment:
//!
//! * `ny-mip` does not depend on `ny-falsify`, so no lane outcome type is even
//!   in scope. [`ObjectiveRequirement`] and [`ObjectiveTier`] are deliberately
//!   local re-declarations of structural facts, not imports of a lane's result.
//! * Every type on this module's public surface is built from `Duration`,
//!   `f64`, `usize`, `&'static str` and this module's own plain enums. None of
//!   them has a variant, field or constructor that could carry an instance
//!   answer, and `lane_allocation_tests::no_answer_bearing_type_is_reachable`
//!   re-reads this source and fails the build if one appears.
//!
//! A bad allocation therefore costs a missed row and can never cost a wrong
//! one, which is what licenses the aggressive parts: zeroing a lane on a
//! structural property, and failing open on a solver timeout.
//!
//! # Fail open
//!
//! Anything but a proven optimum inside [`ALLOC_SOLVE_CAP`] returns
//! [`AllocationOutcome::UseExistingPlan`] with a typed reason and the caller
//! runs whatever it runs today. The 10 ms cap is enforced from OUTSIDE the ay
//! session by the existing detached-worker seam (`ay_lib::run_with_hard_deadline_at`),
//! because `SolveOpts` time limits are advisory to the engine's own checks.
//! The allocator must never consume the budget it exists to save.

use std::time::{Duration, Instant};

use num_traits::{One, Zero};

use crate::ir::{Col, MilpProblem};

// ---------------------------------------------------------------------------
// Declared constants. Every one of these is a NAMED DEFAULT, not a fit.
// ---------------------------------------------------------------------------

/// Hard wall for one allocator solve, enforced outside the ay session.
///
/// `ay_lib::hard_timeout_slice_secs` clamps to `[0.001, 86_400]` s, so 10 ms is
/// representable.
///
/// # Measured, because the design note's "expect microseconds" is not right
///
/// Optimised build, this host, through [`allocate`], on the realistic shape
/// (5 lanes x 10 rungs = 50 binaries, 7 rows, one lane structurally zeroed, one
/// precedence edge, 480 s budget and a 145 s reserve):
///
/// **mean 1.85 ms, worst 2.01 ms over 20 solves; 20 of 20 landed inside this
/// cap.** Milliseconds, not microseconds, and comfortably inside 10 ms.
///
/// An UNOPTIMISED build is ~10x slower (18.0 ms on the same instance), so
/// `cargo test` at this cap falls open — which is exactly why the correctness
/// tests set their own cap and only the microbenchmark reports against this one.
///
/// Relative scaling, unoptimised, random increasing ladders:
///
/// | lanes x rungs | binaries | per solve (debug) |
/// |---|---|---|
/// | 1 x 2  |   3 |  0.58 ms |
/// | 3 x 2  |   9 |  1.24 ms |
/// | 5 x 2  |  15 |  1.92 ms |
/// | 5 x 3  |  20 |  7.4 ms |
/// | 5 x 5  |  30 | 14.1 ms |
/// | 4 x 8  |  36 | 13.9 ms |
/// | 8 x 3  |  32 | 16.3 ms |
/// | 5 x 10 |  55 | 19.5 ms |
/// | 8 x 12 | 104 | 50.1 ms |
///
/// The cost is not smooth in the size: it is the branch-and-bound tree, and one
/// 32-binary instance can cost more than one 55-binary instance. Three things
/// took the 5 x 10 case from 216 ms to 19.5 ms and all three are in this module:
/// refusing the structure-recognition routes, the whole-second knapsack row,
/// and the integer objective (see [`OBJECTIVE_SNAP_BITS`]).
///
/// Missing the cap is not a correctness problem — it costs the existing plan
/// and nothing else — but it is a sizing rule worth stating: the ladders this
/// module can read off source ([`declared_ladder`]) are 2 and 3 rungs, so a
/// realistic 3-5 lane instance is 8-20 binaries and sits far inside the cap.
pub const ALLOC_SOLVE_CAP: Duration = Duration::from_millis(10);

/// `p_k` is clamped to `1 - EPS` so `ln(1 - p)` stays finite.
pub const REACH_PROBABILITY_CLAMP: f64 = 1.0 - 1e-6;

/// The one prior scalar in Layer A: `a_k` for EVERY lane.
///
/// It is the mean of a uniform Beta(1,1) prior over an unobserved binary
/// outcome. It is not tuned against any benchmark and must not be; making it
/// lane-specific or instance-specific is Layer B.
///
/// # What being uniform does and does NOT buy
///
/// It is IDENTICAL across lanes, so it cannot express a preference between
/// them: swapping two lanes' ladders swaps their grants exactly
/// (`the_default_prior_is_symmetric_across_lanes`). That is what rules out a
/// benchmark-tuned ranking hiding in this constant.
///
/// It is NOT inert, and it would be an overclaim to say so. `p = a * s` sits
/// INSIDE `ln(1 - p)`, so unlike the instance-level factor `q(phi)` of the
/// design note §2.4 — which multiplies the whole union and genuinely drops out
/// of the argmax — `a` changes how sharply the objective saturates, and
/// therefore how strongly the knapsack prefers spreading budget over
/// concentrating it. Measured flip, two lanes and a 20 s pool
/// (`the_uniform_prior_is_a_real_knob_even_though_it_cannot_rank_lanes`):
/// `a = 1.0` buys one lane's 20 s rung, `a = 0.5` buys both lanes' 10 s rungs.
///
/// So this is a declared modelling choice with a named default, not a free
/// parameter anyone may turn: moving it is a Layer B decision that needs
/// measured `a_k(phi)` behind it, not a knob to twist until a family scores.
pub const DEFAULT_LANE_REACH_PRIOR: f64 = 0.5;

/// Objective weights are snapped to a multiple of `2^-OBJECTIVE_SNAP_BITS`
/// before they reach the exact-rational backend, and the model's objective is
/// that snapped weight multiplied by `2^OBJECTIVE_SNAP_BITS`, i.e. an INTEGER.
/// See [`LaneRequest::log_miss_cost_at`] for why.
pub const OBJECTIVE_SNAP_BITS: i32 = 20;

/// Largest ladder this module will build or accept, per lane.
pub const MAX_RUNGS_PER_LANE: usize = 16;

/// Largest number of lanes this module will accept in one request.
pub const MAX_LANES: usize = 8;

// ---------------------------------------------------------------------------
// Citations
// ---------------------------------------------------------------------------

/// A pointer at the SOURCE CONSTANT a ladder rung was read from.
///
/// `line` is the line as of the read (2026-08-19) and is advisory: the cited
/// files are edited by other work, so `lane_allocation_tests` verifies the
/// citation by CONTENT — it asserts `item` and `value` still occur together on
/// one line of `file` — and reports the current line if it has drifted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceCitation {
    /// Repository-relative path of the file the constant was read from.
    pub file: &'static str,
    /// Line the constant sat on when it was read. Advisory; see the type doc.
    pub line: u32,
    /// The identifier, exactly as spelled in the source.
    pub item: &'static str,
    /// The literal, exactly as spelled in the source.
    pub value: &'static str,
}

// ---------------------------------------------------------------------------
// Structural facts. Local re-declarations, never imports of a lane result.
// ---------------------------------------------------------------------------

/// How much signal the objective carries, from the in-box probe.
///
/// Classified by `n_distinct`, the number of distinct f32 objective values over
/// the probe points: `1` is FLAT, a small count is a staircase, a count near
/// the probe size is smooth. The measured case that motivates the whole
/// mechanism: on `traffic_signs_recognition_2023` the post-Softmax objective
/// was the constant `-1.0` — ONE distinct f32 across 33 points on 45 of 45
/// rows — so a gradient-guided lane there is not slow, it is BLIND.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveTier {
    /// One distinct objective value over the probe. No direction exists.
    Flat,
    /// Few distinct values: an integer-logit staircase.
    Staircase,
    /// Distinct values at nearly every probe point.
    Smooth,
}

impl ObjectiveTier {
    /// Classify from the probe's distinct-value count.
    ///
    /// The only boundary that carries a structural zero is `Flat`, and it is
    /// the boundary that needs no threshold at all: exactly one distinct value.
    #[must_use]
    pub const fn from_distinct_objective_values(n_distinct: usize) -> Self {
        match n_distinct {
            0 | 1 => Self::Flat,
            2..=31 => Self::Staircase,
            _ => Self::Smooth,
        }
    }
}

/// What a lane needs from the objective in order to make progress at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveRequirement {
    /// Needs an exact gradient (inside the differentiable fragment).
    Exact,
    /// Needs an estimated gradient (finite differences, SPSA, NES, STE).
    EstimatedGradient,
    /// Needs only objective VALUES; can search without any direction.
    ValueOnly,
}

impl ObjectiveRequirement {
    /// Whether this lane steers by a direction derived from the objective.
    #[must_use]
    pub const fn is_gradient_guided(self) -> bool {
        matches!(self, Self::Exact | Self::EstimatedGradient)
    }
}

/// A HARD `p_k = 0`, carrying the structural evidence that produced it.
///
/// Every variant is a property of the instance or of the lane's own admission
/// contract. None of them is a measurement, a rate or a threshold fitted to a
/// benchmark; each one is a fact that makes the lane's success set EMPTY rather
/// than merely small. Use the checked constructors: they refuse to build a zero
/// whose stated evidence does not actually imply it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuralZero {
    /// The objective probe found ONE distinct value and the lane steers by a
    /// direction derived from that objective. There is no direction to find.
    FlatObjectiveTier {
        /// Distinct objective values observed over the probe points.
        distinct_objective_values: usize,
        /// What the lane needed from the objective.
        requirement: ObjectiveRequirement,
    },
    /// The lane returned a typed STRUCTURAL decline receipt for this instance
    /// (a shape it cannot encode, not a budget it did not get).
    StructurallyDeclined,
    /// The lane's own admission ceiling excludes this instance: its parameter
    /// space is declared over at most `ceiling` free dimensions and the
    /// instance has more.
    AboveAdmissionCeiling {
        /// Free dimensions the instance actually has.
        free_dims: u64,
        /// The lane's declared ceiling.
        ceiling: u64,
    },
}

impl StructuralZero {
    /// A flat objective zeroes a lane only if that lane steers by the objective.
    ///
    /// Returns `None` for a `ValueOnly` lane: a value-only search has no
    /// direction to lose, so a flat PROBE does not prove its success set empty.
    #[must_use]
    pub const fn flat_objective(
        distinct_objective_values: usize,
        requirement: ObjectiveRequirement,
    ) -> Option<Self> {
        if requirement.is_gradient_guided()
            && matches!(
                ObjectiveTier::from_distinct_objective_values(distinct_objective_values),
                ObjectiveTier::Flat
            )
        {
            Some(Self::FlatObjectiveTier {
                distinct_objective_values,
                requirement,
            })
        } else {
            None
        }
    }

    /// A decline is a zero only when the receipt says it is STRUCTURAL. A
    /// budget-below-floor decline is not: it is what the allocator is for.
    #[must_use]
    pub const fn structurally_declined(receipt_is_structural: bool) -> Option<Self> {
        if receipt_is_structural {
            Some(Self::StructurallyDeclined)
        } else {
            None
        }
    }

    /// A ceiling zero requires the instance to be strictly above the ceiling.
    #[must_use]
    pub const fn above_admission_ceiling(free_dims: u64, ceiling: u64) -> Option<Self> {
        if free_dims > ceiling {
            Some(Self::AboveAdmissionCeiling { free_dims, ceiling })
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Cap ladders
// ---------------------------------------------------------------------------

/// Why a rung exists. Every nonzero variant names a SOURCE-DERIVED reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RungOrigin {
    /// `g[k][0] = 0`: do not run this lane.
    DoNotRun,
    /// The smallest cap at which the lane can execute its own declared
    /// schedule to a typed conclusion rather than being cut off mid-work.
    DeclaredFloor,
    /// The lane's own declared default schedule wall.
    DeclaredSchedule,
    /// The declared schedule does not fit; this rung is the pool itself, which
    /// the lane can still plan against because its wall is a settable field.
    BudgetTruncated,
    /// Supplied by the caller (synthetic ladders, tests, and any lane whose
    /// ladder Layer A cannot read off source).
    CallerSupplied,
}

/// One plannable cap, with the RELATIVE work it buys inside its own lane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rung {
    /// The committed cap. `Duration::ZERO` on rung 0 only.
    pub cap: Duration,
    /// `s_k(j)` in `[0, 1]`: the fraction of the TOP rung's declared work
    /// budget this rung buys, non-decreasing along the ladder.
    ///
    /// This is a DECLARED relative work profile read off the lane's own
    /// schedule constants — not an estimated success rate. It says how much of
    /// its search the lane can afford here, and nothing about how likely that
    /// search is to succeed; the latter is the uniform prior `a_k`.
    pub reach: f64,
    /// Why this rung is here.
    pub origin: RungOrigin,
}

/// Where a ladder came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LadderProvenance {
    /// Read off the lane's own schedule source, with citations.
    ReadFromSource {
        /// The lane whose source was read.
        lane: Lane,
        /// The constants the rungs were derived from.
        citations: &'static [SourceCitation],
    },
    /// Handed in by the caller. Layer A makes no claim about it.
    CallerSupplied,
}

/// A lane's cap grid: `{0 = g[k][0] < g[k][1] < ... < g[k][m]}`.
#[derive(Debug, Clone, PartialEq)]
pub struct CapLadder {
    rungs: Vec<Rung>,
    provenance: LadderProvenance,
}

/// A ladder that failed validation. The allocator refuses malformed grids
/// rather than repairing them.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum LadderError {
    /// A ladder must have at least the do-not-run rung.
    #[error("ladder is empty")]
    Empty,
    /// More rungs than the allocator will size for.
    #[error("ladder has {rungs} rungs, at most {MAX_RUNGS_PER_LANE} are accepted")]
    TooManyRungs {
        /// The rung count offered.
        rungs: usize,
    },
    /// Rung 0 must be the do-not-run rung.
    #[error("rung 0 must be the zero cap with zero reach")]
    FirstRungIsNotZero,
    /// Caps must be strictly increasing.
    #[error("rung {rung} cap does not exceed the rung below it")]
    CapNotIncreasing {
        /// Index of the offending rung.
        rung: usize,
    },
    /// Reach must be non-decreasing and inside `[0, 1]`.
    #[error("rung {rung} reach {reach} is not a non-decreasing value in [0,1]")]
    ReachNotMonotone {
        /// Index of the offending rung.
        rung: usize,
        /// The offending value.
        reach: f64,
    },
    /// Caps enter the knapsack row in whole seconds.
    #[error("rung {rung} cap is not a whole number of seconds")]
    CapNotWholeSeconds {
        /// Index of the offending rung.
        rung: usize,
    },
}

impl CapLadder {
    /// Build and VALIDATE a caller-supplied ladder.
    pub fn caller_supplied(rungs: Vec<Rung>) -> Result<Self, LadderError> {
        Self::validated(rungs, LadderProvenance::CallerSupplied)
    }

    fn validated(rungs: Vec<Rung>, provenance: LadderProvenance) -> Result<Self, LadderError> {
        if rungs.is_empty() {
            return Err(LadderError::Empty);
        }
        if rungs.len() > MAX_RUNGS_PER_LANE {
            return Err(LadderError::TooManyRungs { rungs: rungs.len() });
        }
        if rungs[0].cap != Duration::ZERO || rungs[0].reach != 0.0 {
            return Err(LadderError::FirstRungIsNotZero);
        }
        for (j, rung) in rungs.iter().enumerate() {
            if !rung.reach.is_finite() || !(0.0..=1.0).contains(&rung.reach) {
                return Err(LadderError::ReachNotMonotone {
                    rung: j,
                    reach: rung.reach,
                });
            }
            // WHY WHOLE SECONDS. The knapsack row is solved in exact rational
            // arithmetic, where coefficient MAGNITUDE drives bit growth
            // directly: measured on this module's own K=5 x 10-rung instance,
            // the identical model took 139.9 ms per solve with the row in
            // milliseconds and 18.9 ms with it in seconds (debug, same host,
            // same answer). A cap is a schedule, and sub-second granularity is
            // meaningless against budgets of 30-1800 s, so the row is in whole
            // seconds and a fractional rung is a malformed ladder.
            if rung.cap.subsec_nanos() != 0 {
                return Err(LadderError::CapNotWholeSeconds { rung: j });
            }
            if j > 0 {
                if rung.cap <= rungs[j - 1].cap {
                    return Err(LadderError::CapNotIncreasing { rung: j });
                }
                if rung.reach < rungs[j - 1].reach {
                    return Err(LadderError::ReachNotMonotone {
                        rung: j,
                        reach: rung.reach,
                    });
                }
            }
        }
        Ok(Self { rungs, provenance })
    }

    /// The rungs, rung 0 first.
    #[must_use]
    pub fn rungs(&self) -> &[Rung] {
        &self.rungs
    }

    /// Where this ladder came from.
    #[must_use]
    pub const fn provenance(&self) -> &LadderProvenance {
        &self.provenance
    }

    /// The lane's own floor: the smallest nonzero cap on the ladder, i.e. the
    /// smallest cap it can be handed at all. `None` for a do-not-run-only grid.
    #[must_use]
    pub fn floor(&self) -> Option<Duration> {
        self.rungs.get(1).map(|rung| rung.cap)
    }
}

// ---------------------------------------------------------------------------
// The lanes, and the ladders Layer A can and cannot read off source
// ---------------------------------------------------------------------------

/// A lane identity, as the flight recorder already names them.
///
/// A LANE name is not an instance name: this is the identity of a piece of ny's
/// own code, never a benchmark, a category, a directory or a file. Nothing in
/// this module keys on anything about the instance except structural facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lane {
    /// LP-guided sign-space search over binarized `Sign` conv suffixes.
    BnnSignSpace,
    /// Straight-through-estimator PGD with iterated local search.
    BnnStePgd,
    /// The pinned upfront attack window.
    UpfrontAttack,
    /// The wired falsification strategy portfolio.
    FalsifyPortfolio,
    /// The margin row running beside branch-and-bound.
    MarginRowConcurrent,
    /// Forward-linear root tightening.
    ForwardLinearAdmission,
    /// The post-branch-and-bound frontier lane.
    PostBabFrontier,
}

/// The ladder for this lane is not determinable from source at Layer A.
///
/// This is returned rather than a guessed grid. Saying "unknown" is the whole
/// discipline: an invented rung is a fitted number wearing a citation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("no source-declared cap ladder for {lane:?}: {why}")]
pub struct UnknownLadder {
    /// The lane asked about.
    pub lane: Lane,
    /// Exactly what is missing from the source.
    pub why: &'static str,
}

/// `bnn_sign_space` schedule constants, read 2026-08-19.
///
/// The line numbers are as of the read; `bnn_sign_space.rs` is under concurrent
/// edit, so the tests verify these by content rather than by line.
const SIGN_SPACE_CITATIONS: &[SourceCitation] = &[
    SourceCitation {
        file: "crates/ny-mip/src/bnn_sign_space.rs",
        line: 758,
        item: "stall_lp_solves",
        value: "32",
    },
    SourceCitation {
        file: "crates/ny-mip/src/bnn_sign_space.rs",
        line: 770,
        item: "per_lp_time",
        value: "Duration::from_secs(1)",
    },
    SourceCitation {
        file: "crates/ny-mip/src/bnn_sign_space.rs",
        line: 749,
        item: "max_lp_solves",
        value: "20_000",
    },
    SourceCitation {
        file: "crates/ny-mip/src/bnn_sign_space.rs",
        line: 765,
        item: "max_wall_time",
        value: "Duration::from_mins(5)",
    },
];

/// `bnn_ste_pgd` schedule constants, read 2026-08-19.
const STE_PGD_CITATIONS: &[SourceCitation] = &[
    SourceCitation {
        file: "crates/ny-mip/src/bnn_ste_pgd.rs",
        line: 89,
        item: "max_wall_time",
        value: "Duration::from_mins(4)",
    },
    SourceCitation {
        file: "crates/ny-mip/src/bnn_ste_pgd.rs",
        line: 90,
        item: "climb_fraction",
        value: "0.25",
    },
    SourceCitation {
        file: "crates/ny-mip/src/bnn_ste_pgd.rs",
        line: 91,
        item: "iters",
        value: "120",
    },
    SourceCitation {
        file: "crates/ny-mip/src/bnn_ste_pgd.rs",
        line: 92,
        item: "max_restarts",
        value: "4096",
    },
];

/// `stall_lp_solves` — the never-started test's threshold in LP solves.
const SIGN_SPACE_STALL_LP_SOLVES: u32 = 32;
/// `per_lp_time` — the wall cap on ONE realizability LP.
const SIGN_SPACE_PER_LP_TIME: Duration = Duration::from_secs(1);
/// `max_lp_solves` — the whole-lane LP budget.
const SIGN_SPACE_MAX_LP_SOLVES: u64 = 20_000;
/// `max_wall_time` — the lane's declared default schedule wall.
const SIGN_SPACE_DECLARED_WALL: Duration = Duration::from_mins(5);
/// `max_wall_time` — STE-PGD's declared default schedule wall.
const STE_PGD_DECLARED_WALL: Duration = Duration::from_mins(4);

/// The cap ladder Layer A can read off `lane`'s own schedule source, for an
/// instance with `budget_available` seconds left in the pool.
///
/// # What is readable, and what is not
///
/// **`BnnSignSpace` — readable.** Its schedule is expressed in a work unit with
/// a DECLARED per-unit wall, so caps convert to work exactly:
///
/// * floor `= stall_lp_solves * per_lp_time = 32 * 1 s = 32 s`. This is the
///   smallest cap at which the lane can finish its own never-started test and
///   yield a typed conclusion instead of being cut off inside an LP.
/// * declared schedule `= max_wall_time = 300 s`, which also bounds
///   `max_lp_solves * per_lp_time = 20 000 s` far below the LP-count cap, so
///   the wall is what binds and the top rung is the wall.
/// * `reach` at a rung is the LP solves it affords over the LP solves the top
///   rung affords: `min(cap / per_lp_time, max_lp_solves)` normalised.
///
/// **`BnnStePgd` — partially readable, and the missing part is stated.** The
/// source declares the whole-call wall (240 s), the Stage A / Stage B split
/// (`climb_fraction = 0.25`, so Stage A gets `0.75 * cap`), the per-restart
/// step count (`iters = 120`) and the restart cap (`max_restarts = 4096`). It
/// does NOT declare a wall per restart or per gradient step, so **the caps at
/// which an integer number of restarts completes are NOT derivable from
/// source** — they need a measured per-restart cost, which is Layer B. The
/// ladder therefore has exactly the rungs the source does declare: do-not-run,
/// and the declared 240 s schedule. That is not a shortcut, it is the honest
/// grid, and it happens to be the one the measured evidence points at: 217.5 s
/// won nothing on the traffic rows and 240.10 s won three, and 240 s is
/// literally `Duration::from_mins(4)` in `bnn_ste_pgd.rs`.
///
/// **Everything else — unknown.** `UpfrontAttack`, `FalsifyPortfolio`,
/// `MarginRowConcurrent`, `ForwardLinearAdmission` and `PostBabFrontier` have
/// no self-declared cap ladder in this crate's source, so they return
/// [`UnknownLadder`]. A caller with a ladder of its own passes it through
/// [`CapLadder::caller_supplied`]; Layer A will not invent one.
pub fn declared_ladder(lane: Lane, budget_available: Duration) -> Result<CapLadder, UnknownLadder> {
    match lane {
        Lane::BnnSignSpace => Ok(sign_space_ladder(budget_available)),
        Lane::BnnStePgd => Ok(ste_pgd_ladder(budget_available)),
        Lane::UpfrontAttack => Err(UnknownLadder {
            lane,
            why: "its window is a policy over predicted cost, and no cap grid is declared in \
                  this crate's source",
        }),
        Lane::FalsifyPortfolio => Err(UnknownLadder {
            lane,
            why: "the portfolio's per-strategy point budgets are declared, but no wall per \
                  point is, so caps do not convert to work",
        }),
        Lane::MarginRowConcurrent => Err(UnknownLadder {
            lane,
            why: "it shares wall time with branch-and-bound, so its cap is a makespan over \
                  cores rather than a scalar this row can carry",
        }),
        Lane::ForwardLinearAdmission => Err(UnknownLadder {
            lane,
            why: "its window is predicted_cost times an admission margin, which is a per-instance \
                  prediction rather than a declared constant",
        }),
        Lane::PostBabFrontier => Err(UnknownLadder {
            lane,
            why: "no schedule constants for it exist in this crate's source",
        }),
    }
}

fn lp_solves_afforded(cap: Duration) -> u64 {
    let per = SIGN_SPACE_PER_LP_TIME.as_millis().max(1);
    u64::try_from(cap.as_millis() / per)
        .unwrap_or(u64::MAX)
        .min(SIGN_SPACE_MAX_LP_SOLVES)
}

fn sign_space_ladder(budget_available: Duration) -> CapLadder {
    // Rung caps are whole seconds (see `CapLadder::validated`), so a pool with
    // a fractional tail contributes only its whole-second part.
    let budget_available = Duration::from_secs(budget_available.as_secs());
    let floor = SIGN_SPACE_PER_LP_TIME * SIGN_SPACE_STALL_LP_SOLVES;
    let mut rungs = vec![Rung {
        cap: Duration::ZERO,
        reach: 0.0,
        origin: RungOrigin::DoNotRun,
    }];
    if budget_available >= floor {
        let top = SIGN_SPACE_DECLARED_WALL.min(budget_available);
        let top_work = lp_solves_afforded(top).max(1) as f64;
        rungs.push(Rung {
            cap: floor,
            reach: (lp_solves_afforded(floor) as f64 / top_work).min(1.0),
            origin: RungOrigin::DeclaredFloor,
        });
        if top > floor {
            rungs.push(Rung {
                cap: top,
                reach: 1.0,
                origin: if top < SIGN_SPACE_DECLARED_WALL {
                    RungOrigin::BudgetTruncated
                } else {
                    RungOrigin::DeclaredSchedule
                },
            });
        }
    }
    CapLadder::validated(
        rungs,
        LadderProvenance::ReadFromSource {
            lane: Lane::BnnSignSpace,
            citations: SIGN_SPACE_CITATIONS,
        },
    )
    .expect("the source-declared sign-space ladder is well formed by construction")
}

fn ste_pgd_ladder(budget_available: Duration) -> CapLadder {
    let mut rungs = vec![Rung {
        cap: Duration::ZERO,
        reach: 0.0,
        origin: RungOrigin::DoNotRun,
    }];
    if budget_available >= STE_PGD_DECLARED_WALL {
        rungs.push(Rung {
            cap: STE_PGD_DECLARED_WALL,
            reach: 1.0,
            origin: RungOrigin::DeclaredSchedule,
        });
    }
    CapLadder::validated(
        rungs,
        LadderProvenance::ReadFromSource {
            lane: Lane::BnnStePgd,
            citations: STE_PGD_CITATIONS,
        },
    )
    .expect("the source-declared STE-PGD ladder is well formed by construction")
}

// ---------------------------------------------------------------------------
// The request
// ---------------------------------------------------------------------------

/// One lane's entry in an allocation request.
#[derive(Debug, Clone, PartialEq)]
pub struct LaneRequest {
    ladder: CapLadder,
    reach_prior: f64,
    structural_zero: Option<StructuralZero>,
    requires: Option<usize>,
    no_regression_floor: Option<Duration>,
}

impl LaneRequest {
    /// A lane with the uniform prior and no structural zero.
    #[must_use]
    pub const fn new(ladder: CapLadder) -> Self {
        Self {
            ladder,
            reach_prior: DEFAULT_LANE_REACH_PRIOR,
            structural_zero: None,
            requires: None,
            no_regression_floor: None,
        }
    }

    /// Pin this lane to zero seconds on structural evidence, constraint (A).
    #[must_use]
    pub const fn zeroed(mut self, zero: StructuralZero) -> Self {
        self.structural_zero = Some(zero);
        self
    }

    /// Declare that this lane needs the output of lane `index`, constraint (P).
    #[must_use]
    pub const fn requiring(mut self, index: usize) -> Self {
        self.requires = Some(index);
        self
    }

    /// Constraint (C5): this lane may not be granted LESS than today's fixed
    /// slice would have given it, so today's plan stays a feasible point and
    /// the optimum is no worse than today by construction. Ignored on a lane
    /// with a structural zero, whose seconds admission already reclaimed.
    #[must_use]
    pub const fn no_worse_than(mut self, cap_today: Duration) -> Self {
        self.no_regression_floor = Some(cap_today);
        self
    }

    /// Override the uniform prior.
    ///
    /// Layer A callers should not need this; it exists so a Layer B estimator
    /// can supply `a_k(phi)` later WITHOUT this module changing. Nothing in
    /// this crate calls it, and no test tunes it.
    #[must_use]
    pub const fn with_reach_prior(mut self, prior: f64) -> Self {
        self.reach_prior = prior;
        self
    }

    /// This lane's ladder.
    #[must_use]
    pub const fn ladder(&self) -> &CapLadder {
        &self.ladder
    }

    /// The structural zero pinning this lane, if any.
    #[must_use]
    pub const fn structural_zero(&self) -> Option<StructuralZero> {
        self.structural_zero
    }

    /// `p_k(g[k][j]) = a_k * s_k(j)`, clamped, and hard 0 under a structural
    /// zero. This is the ONLY place a probability is formed.
    #[must_use]
    pub fn success_prior_at(&self, rung: usize) -> f64 {
        if self.structural_zero.is_some() {
            return 0.0;
        }
        let Some(step) = self.ladder.rungs.get(rung) else {
            return 0.0;
        };
        (self.reach_prior * step.reach).clamp(0.0, REACH_PROBABILITY_CLAMP)
    }

    /// `c[k][j] = ln(1 - p_k(g[k][j])) <= 0`, the objective weight.
    ///
    /// SNAPPED to a multiple of `2^-OBJECTIVE_SNAP_BITS`. The backend is
    /// exact-rational, so an arbitrary `f64` enters the model as a dyadic with
    /// a `2^52` denominator and every pivot carries those bits; snapping keeps
    /// the rationals short, and the model then carries this weight multiplied
    /// by `2^OBJECTIVE_SNAP_BITS`, which is an exact INTEGER. Scaling an
    /// objective by a positive constant does not move its argmax, so the plan
    /// is the same plan — it is only cheaper to prove optimal (measured on the
    /// K=5 x 10-rung instance: 108.3 ms with dyadic weights, 18.9 ms with
    /// integer weights, debug, same host, same answer).
    ///
    /// The snap is done HERE, in the one place a weight is formed, so the
    /// weights the solver optimises over are exactly the weights a caller reads
    /// back from this method: the optimum is exact with respect to the stated
    /// objective rather than approximate with respect to a hidden one. The snap
    /// itself is below `1e-6` in log-probability.
    #[must_use]
    pub fn log_miss_cost_at(&self, rung: usize) -> f64 {
        let raw = (1.0 - self.success_prior_at(rung)).ln();
        let scale = (2.0f64).powi(OBJECTIVE_SNAP_BITS);
        (raw * scale).round() / scale
    }
}

/// One instance's allocation problem.
#[derive(Debug, Clone, PartialEq)]
pub struct AllocationRequest {
    lanes: Vec<LaneRequest>,
    budget: Duration,
    reserve: Duration,
    solve_cap: Duration,
}

impl AllocationRequest {
    /// `budget` is the official instance budget `B`; `reserve` is `R`, the
    /// publication and downstream margin that the knapsack row subtracts.
    #[must_use]
    pub const fn new(lanes: Vec<LaneRequest>, budget: Duration, reserve: Duration) -> Self {
        Self {
            lanes,
            budget,
            reserve,
            solve_cap: ALLOC_SOLVE_CAP,
        }
    }

    /// Override the solve cap. The default is [`ALLOC_SOLVE_CAP`].
    #[must_use]
    pub const fn with_solve_cap(mut self, cap: Duration) -> Self {
        self.solve_cap = cap;
        self
    }

    /// The pool the knapsack row may spend: `B - R`, floored at zero.
    #[must_use]
    pub fn pool(&self) -> Duration {
        self.budget.saturating_sub(self.reserve)
    }

    /// The lanes, in request order. A lane's index IS its identity here.
    #[must_use]
    pub fn lanes(&self) -> &[LaneRequest] {
        &self.lanes
    }
}

// ---------------------------------------------------------------------------
// The outcome
// ---------------------------------------------------------------------------

/// What one lane was committed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneGrant {
    /// Index into [`AllocationRequest::lanes`].
    pub lane: usize,
    /// Index into that lane's ladder. `0` means do not run.
    pub rung: usize,
    /// The committed cap. `Duration::ZERO` iff `rung == 0`.
    pub cap: Duration,
}

/// A committed plan: exactly one rung per lane, inside the pool.
#[derive(Debug, Clone, PartialEq)]
pub struct LanePlan {
    grants: Vec<LaneGrant>,
    log_miss_total: f64,
}

impl LanePlan {
    /// One grant per lane, in request order.
    #[must_use]
    pub fn grants(&self) -> &[LaneGrant] {
        &self.grants
    }

    /// The optimised objective `SUM_k ln(1 - p_k)`; `<= 0`, lower is better.
    #[must_use]
    pub const fn log_miss_total(&self) -> f64 {
        self.log_miss_total
    }

    /// `1 - PROD_k (1 - p_k)` under the declared ladders and the uniform prior.
    ///
    /// A PRIOR, not a prediction: it is the objective the knapsack maximised,
    /// reported so a caller can log what it chose and why. Nothing consumes it.
    #[must_use]
    pub fn prior_union_reach(&self) -> f64 {
        1.0 - self.log_miss_total.exp()
    }

    /// Total committed seconds.
    #[must_use]
    pub fn committed(&self) -> Duration {
        self.grants
            .iter()
            .fold(Duration::ZERO, |sum, grant| sum + grant.cap)
    }
}

/// Why the allocator declined to produce a plan. Every variant means "run
/// whatever you run today"; none of them is an error the caller must handle.
#[derive(Debug, Clone, PartialEq)]
pub enum FallOpen {
    /// More lanes than the allocator sizes for.
    TooManyLanes {
        /// How many were offered.
        lanes: usize,
    },
    /// A ladder failed validation.
    MalformedLadder {
        /// Which lane.
        lane: usize,
        /// What was wrong.
        error: LadderError,
    },
    /// A precedence edge names a lane that is not in the request.
    UnknownPrecedence {
        /// Which lane declared it.
        lane: usize,
        /// The index it named.
        requires: usize,
    },
    /// A prior or a reach was not finite.
    NonFiniteInput {
        /// Which lane.
        lane: usize,
    },
    /// The IR did not lower.
    Lowering(String),
    /// The backend returned an error.
    SolverError(String),
    /// The 10 ms boundary expired; the worker was abandoned.
    SolveDeadline,
    /// The solve finished without a proven optimum.
    NotOptimal,
    /// No assignment satisfies the pool, the precedence edges and (C5) — which
    /// under (C5) means today's plan does not fit either, so today's plan is
    /// what already runs and is not made worse.
    Infeasible,
    /// The exact-rational primal was not a clean one-hot per lane.
    ReadbackNotOneHot,
    /// The backend's own point gate rejected the returned point.
    PointRejected,
    /// The returned assignment violated a constraint on re-check. Treated as a
    /// failure, never as advice.
    ConstraintViolatedInReadback,
}

/// The allocator's answer. There is no third case and no error to propagate.
#[derive(Debug, Clone, PartialEq)]
pub enum AllocationOutcome {
    /// A proven-optimal set of caps.
    Allocated(LanePlan),
    /// Fail open: keep the existing plan, for this typed reason.
    UseExistingPlan(FallOpen),
}

// ---------------------------------------------------------------------------
// The solve
// ---------------------------------------------------------------------------

/// What the detached worker hands back. `Chosen` carries one rung index per
/// lane; nothing else crosses the thread boundary.
enum WorkerResult {
    Chosen(Vec<usize>),
    FellOpen(FallOpen),
}

/// Whole seconds, rounding DOWN. Exact on a validated rung cap (which is a
/// whole number of seconds); on the pool it is conservative (the row may spend
/// less than the pool holds, never more) and on a (C5) floor it is lax by under
/// one second, which is not a regression anyone can schedule against.
fn whole_secs_f64(d: Duration) -> f64 {
    d.as_secs() as f64
}

/// Choose one cap per lane by solving the multiple-choice knapsack exactly.
///
/// Never panics, never blocks past `solve_cap`, and never returns anything but
/// a plan or a typed [`FallOpen`].
#[must_use]
pub fn allocate(request: &AllocationRequest) -> AllocationOutcome {
    match build(request) {
        Ok(built) => run(request, built),
        Err(reason) => AllocationOutcome::UseExistingPlan(reason),
    }
}

struct Built {
    problem: MilpProblem,
    objective: Vec<(usize, f64)>,
    cols: Vec<Vec<usize>>,
}

fn build(request: &AllocationRequest) -> Result<Built, FallOpen> {
    let lanes = request.lanes();
    if lanes.len() > MAX_LANES {
        return Err(FallOpen::TooManyLanes { lanes: lanes.len() });
    }
    for (k, lane) in lanes.iter().enumerate() {
        // Re-validate: a ladder can only reach here through a validating
        // constructor, but a caller that mutates one must fail OPEN, not panic.
        CapLadder::validated(lane.ladder.rungs.clone(), LadderProvenance::CallerSupplied)
            .map_err(|error| FallOpen::MalformedLadder { lane: k, error })?;
        if !lane.reach_prior.is_finite() || !(0.0..=1.0).contains(&lane.reach_prior) {
            return Err(FallOpen::NonFiniteInput { lane: k });
        }
        if let Some(requires) = lane.requires {
            if requires >= lanes.len() || requires == k {
                return Err(FallOpen::UnknownPrecedence { lane: k, requires });
            }
        }
    }

    let mut problem = MilpProblem::new();
    let mut cols: Vec<Vec<usize>> = Vec::with_capacity(lanes.len());
    let mut objective: Vec<(usize, f64)> = Vec::new();

    for (k, lane) in lanes.iter().enumerate() {
        let zeroed = lane.structural_zero.is_some();
        let mut row: Vec<Col> = Vec::with_capacity(lane.ladder.rungs.len());
        for j in 0..lane.ladder.rungs.len() {
            // (A) A structurally zeroed lane is a BOUND FIX, not a row: rung 0
            // is pinned to 1 and every nonzero rung to 0. `to_ay_model` lowers
            // a pinned binary cleanly, so this costs no rows and no search.
            let (lb, ub) = match (zeroed, j) {
                (true, 0) => (1.0, 1.0),
                (true, _) => (0.0, 0.0),
                (false, _) => (0.0, 1.0),
            };
            // `ColSpec.obj` is DROPPED by `to_ay_model`; the objective is set
            // as a full linear form on the lowered model instead.
            let col = problem.add_integer_col(0.0, lb, ub);
            let cost = lane.log_miss_cost_at(j);
            if !cost.is_finite() {
                return Err(FallOpen::NonFiniteInput { lane: k });
            }
            if cost != 0.0 {
                // The model carries the snapped weight times 2^SNAP_BITS: an
                // exact integer, and a positive rescale of the same objective.
                objective.push((col.0, cost * (2.0f64).powi(OBJECTIVE_SNAP_BITS)));
            }
            row.push(col);
        }
        // (E) exactly one rung per lane.
        problem.add_row(1.0, 1.0, row.iter().map(|&c| (c, 1.0)));
        cols.push(row.iter().map(|c| c.0).collect());
    }

    // (B) the pool, in whole milliseconds so the row arithmetic is exact.
    let pool = whole_secs_f64(request.pool());
    let mut budget_terms: Vec<(Col, f64)> = Vec::new();
    for (k, lane) in lanes.iter().enumerate() {
        for (j, rung) in lane.ladder.rungs.iter().enumerate() {
            budget_terms.push((Col(cols[k][j]), whole_secs_f64(rung.cap)));
        }
    }
    problem.add_row(f64::NEG_INFINITY, pool, budget_terms);

    // (P) precedence: k may run only if the lane it needs runs.
    for (k, lane) in lanes.iter().enumerate() {
        let Some(pre) = lane.requires else { continue };
        let mut terms: Vec<(Col, f64)> = cols[k][1..].iter().map(|&c| (Col(c), 1.0)).collect();
        terms.extend(cols[pre][1..].iter().map(|&c| (Col(c), -1.0)));
        problem.add_row(f64::NEG_INFINITY, 0.0, terms);
    }

    // (C5) no-regression, skipped for structurally zeroed lanes.
    for (k, lane) in lanes.iter().enumerate() {
        if lane.structural_zero.is_some() {
            continue;
        }
        let Some(floor) = lane.no_regression_floor else {
            continue;
        };
        let terms: Vec<(Col, f64)> = lane
            .ladder
            .rungs
            .iter()
            .enumerate()
            .map(|(j, rung)| (Col(cols[k][j]), whole_secs_f64(rung.cap)))
            .collect();
        problem.add_row(whole_secs_f64(floor), f64::INFINITY, terms);
    }

    Ok(Built {
        problem,
        objective,
        cols,
    })
}

fn run(request: &AllocationRequest, built: Built) -> AllocationOutcome {
    let lanes = request.lanes();
    if lanes.is_empty() {
        return AllocationOutcome::Allocated(LanePlan {
            grants: Vec::new(),
            log_miss_total: 0.0,
        });
    }

    let Some(deadline) = Instant::now().checked_add(request.solve_cap) else {
        return AllocationOutcome::UseExistingPlan(FallOpen::SolveDeadline);
    };
    let Built {
        problem,
        objective,
        cols,
    } = built;
    let cap = request.solve_cap;

    let solve = move || -> crate::Result<WorkerResult> {
        let mut model = crate::ay_lib::to_ay_model(&problem)?;
        let mut terms: Vec<(ay_milp::Col, f64)> = Vec::with_capacity(objective.len());
        for &(col, weight) in &objective {
            let Some(mapped) = model.col_at(col) else {
                return Ok(WorkerResult::FellOpen(FallOpen::Lowering(format!(
                    "objective column {col} disappeared during lowering"
                ))));
            };
            terms.push((mapped, weight));
        }
        // `c[k][j] = ln(1 - p) <= 0`, so MINIMISING the sum maximises
        // `1 - PROD (1 - p_k)`. Same argmax, one fewer negation.
        model.set_objective(&terms, ay_milp::Sense::Minimize);
        // The point gate needs the model; `BabSession::new` consumes it.
        let gate = model.clone();
        let opts = ay_milp::SolveOpts::new()
            .with_time_limit(cap)
            .with_deadline(deadline)
            // Nothing here consumes an infeasibility proof object: an
            // infeasible model IS the fail-open signal and the returned
            // assignment is re-checked against every row before it becomes a
            // plan. Capturing a tree witness would be work spent on evidence
            // no caller reads.
            .with_tree_cert_leaves(0)
            // The exact structure-recognition routes are pattern matches for
            // scheduling / network-design / PB shapes. An MCKP is none of them,
            // so every route is recognition work that always fails: measured on
            // this module's own K=5 x 10-rung instance it was the single
            // largest cost, 88.1 ms -> 29.6 ms per solve (debug) when refused.
            // Refusing them pins the solve on native branch-and-bound, which
            // still returns a proven optimum.
            .with_structure_routing(false);
        let mut session = ay_milp::BabSession::new(model, &opts)
            .map_err(|e| crate::MipError::Solver(e.to_string()))?;
        let outcome = session
            .check()
            .map_err(|e| crate::MipError::Solver(e.to_string()))?;
        // STRICT: only a proven optimum is a plan. `Feasible` is an incumbent
        // without an optimality claim and is treated as a miss, not as advice.
        let values = match outcome {
            ay_milp::Outcome::Optimal { model_values, .. } => model_values,
            ay_milp::Outcome::Infeasible { .. } => {
                return Ok(WorkerResult::FellOpen(FallOpen::Infeasible))
            }
            _ => return Ok(WorkerResult::FellOpen(FallOpen::NotOptimal)),
        };
        if values.len() != gate.num_cols() {
            return Ok(WorkerResult::FellOpen(FallOpen::ReadbackNotOneHot));
        }
        // Repeat the backend's own point gate locally: a forged, truncated or
        // stale point can only be REJECTED here, never promoted to a plan.
        if gate.check_point(&values).is_err() {
            return Ok(WorkerResult::FellOpen(FallOpen::PointRejected));
        }
        // Exact-rational one-hot readback: a binary that is neither exactly 0
        // nor exactly 1, or a lane without exactly one 1, is a FAILURE.
        let mut chosen = Vec::with_capacity(cols.len());
        for lane_cols in &cols {
            let mut hit: Option<usize> = None;
            for (j, &col) in lane_cols.iter().enumerate() {
                let value = &values[col];
                if value.is_zero() {
                    continue;
                }
                if !value.is_one() || hit.is_some() {
                    return Ok(WorkerResult::FellOpen(FallOpen::ReadbackNotOneHot));
                }
                hit = Some(j);
            }
            match hit {
                Some(j) => chosen.push(j),
                None => return Ok(WorkerResult::FellOpen(FallOpen::ReadbackNotOneHot)),
            }
        }
        Ok(WorkerResult::Chosen(chosen))
    };

    // The 10 ms cap is enforced from OUTSIDE the session: `SolveOpts` limits
    // are advisory to the engine's own checks, so the only sound enforcement
    // is to abandon the worker at the boundary.
    match crate::ay_lib::run_with_hard_deadline_at(deadline, "lane-allocation", solve) {
        Ok(Some(WorkerResult::Chosen(chosen))) => finish(request, &chosen),
        Ok(Some(WorkerResult::FellOpen(reason))) => AllocationOutcome::UseExistingPlan(reason),
        Ok(None) => AllocationOutcome::UseExistingPlan(FallOpen::SolveDeadline),
        Err(error) => AllocationOutcome::UseExistingPlan(FallOpen::SolverError(error.to_string())),
    }
}

/// Re-check every constraint against the returned assignment before promoting
/// it to a plan. Defence in depth: a solver defect can cost a missed row here,
/// never an over-spent budget.
fn finish(request: &AllocationRequest, chosen: &[usize]) -> AllocationOutcome {
    let lanes = request.lanes();
    if chosen.len() != lanes.len() {
        return AllocationOutcome::UseExistingPlan(FallOpen::ConstraintViolatedInReadback);
    }
    let mut grants = Vec::with_capacity(lanes.len());
    let mut spent_secs: u64 = 0;
    let mut log_miss_total = 0.0;
    for (k, lane) in lanes.iter().enumerate() {
        let j = chosen[k];
        let Some(rung) = lane.ladder.rungs.get(j) else {
            return AllocationOutcome::UseExistingPlan(FallOpen::ConstraintViolatedInReadback);
        };
        if lane.structural_zero.is_some() && j != 0 {
            return AllocationOutcome::UseExistingPlan(FallOpen::ConstraintViolatedInReadback);
        }
        if lane.structural_zero.is_none() {
            if let Some(floor) = lane.no_regression_floor {
                if rung.cap.as_secs() < floor.as_secs() {
                    return AllocationOutcome::UseExistingPlan(
                        FallOpen::ConstraintViolatedInReadback,
                    );
                }
            }
        }
        if let Some(pre) = lane.requires {
            if j > 0 && chosen.get(pre).copied().unwrap_or(0) == 0 {
                return AllocationOutcome::UseExistingPlan(FallOpen::ConstraintViolatedInReadback);
            }
        }
        spent_secs += rung.cap.as_secs();
        log_miss_total += lane.log_miss_cost_at(j);
        grants.push(LaneGrant {
            lane: k,
            rung: j,
            cap: rung.cap,
        });
    }
    if spent_secs > request.pool().as_secs() {
        return AllocationOutcome::UseExistingPlan(FallOpen::ConstraintViolatedInReadback);
    }
    AllocationOutcome::Allocated(LanePlan {
        grants,
        log_miss_total,
    })
}

#[cfg(test)]
#[path = "lane_allocation_tests.rs"]
mod lane_allocation_tests;
