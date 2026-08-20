// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! S9 `square` — random block sign-flip hill climbing.
//!
//! # The evidence
//!
//! Two calibration wins, and they are the two that matter:
//!
//! | family | free inputs | points to the win | strategies that ran first and lost |
//! |---|---:|---:|---|
//! | `soundnessbench` `model_36` | 128 | 2304 (9 batches of 256) | `special` 8, `corners_random` 512, `spsa` 5568 |
//! | `traffic_signs_recognition_2023` `model_48_idx_10495_eps_10` | 6912 | 768 (3 batches of 256) | `special` 8, `corners_random` 768, `spsa` 7168 |
//!
//! `square` is the ONLY strategy in the whole table that took either family.
//! Its Python docstring says why, and it names ny's `#deadlane` refusal set
//! exactly: it "keeps working when the objective is flat and every gradient
//! estimate (SPSA, NES) collapses to zero".
//!
//! That is not a hypothetical on `traffic_signs`. Measured this session on the
//! 2026 corpus: the BNN outputs are a one-hot INTEGER vector, exactly two
//! distinct output values over random in-box points, `all integral? True`, and
//! max-other-minus-true is exactly `-1.0` at every point. The objective is a
//! two-level step function, so every finite-difference estimator is identically
//! zero and every gradient-shaped strategy degenerates to blind uniform
//! sampling in a 2700-d box. The calibration row above shows the same signature
//! from the other side: `"best_output_margin": 0.0` with `spsa` having burned
//! 7168 points for nothing.
//!
//! # Why it reaches what nothing else in ny reaches
//!
//! Its point distribution is the thing, not its hill climbing. A row is the
//! incumbent with `max(1, n * fraction)` randomly chosen coordinates pushed to
//! a randomly chosen bound, and `fraction` anneals `0.5 * 0.85^k` down to
//! `1/n`. So it sweeps *partial* vertices — points with some coordinates at a
//! bound and the rest at the incumbent — annealing from half the box's
//! coordinates to one.
//!
//! No lane ny ships generates that. `corners_full`/`corners_random` produce
//! full vertices only (every coordinate at a bound). Uniform and Halton
//! sampling produce interior points and hit a bound with probability zero.
//! `special` produces eight fixed patterns. The partial-vertex faces in
//! between are unvisited, and on an integer-gated net they are where the
//! violating cells are.
//!
//! # Port fidelity
//!
//! `Searcher._strategy_square`, with the constants preserved
//! ([`SQUARE_INITIAL_FRACTION`], [`SQUARE_ANNEAL`],
//! [`SQUARE_ITERATIONS_PER_RESTART`]) and three chassis additions: a
//! per-strategy ceiling, a stall rule in its own work unit, and alternating
//! restart seeds (below). The RNG is this crate's `xoshiro256**`, not numpy's
//! PCG64, so the point *stream* differs from the Python's; the point
//! *distribution* and the annealing schedule do not.
//!
//! # The one behavioural change, and why the port needs it
//!
//! The Python seeds every restart from the shared incumbent
//! (`current = self.best_free.copy()`). That is safe in the Python because the
//! plan runs eleven strategies and `spsa`/`uniform`/`halton` keep depositing
//! INTERIOR incumbents — on `soundnessbench model_36` and
//! `traffic_signs model_48_idx_10495_eps_10` alike, `spsa` ran immediately
//! before `square` and burned 5568 and 7168 points respectively leaving an
//! interior point behind.
//!
//! With only `special` and `square` ported, that is no longer true, and the
//! failure is silent. An incumbent is adopted on a STRICT improvement over
//! `-inf`, so on a flat objective the first batch of any strategy sets it —
//! and `special`'s argmax under an exact tie is its first pattern, `all_low`,
//! a VERTEX. Every coordinate of a vertex is already at a bound, so every
//! block flip from it is a no-op and `square` degenerates precisely into
//! `corners_random`, which ny already ships. The strategy would still run, still
//! report points, and have lost the entire reach it was ported for.
//!
//! So: even restarts seed from the shared incumbent (restart 0 is
//! port-identical, and both calibration wins landed in restart 0, at batch 9
//! and batch 3), odd restarts seed from the box centre. When the incumbent is
//! informative nothing is given up; when it is degenerate the partial-vertex
//! reach survives. `tests/calibration_square.rs` exercises both arms.

use crate::admission::{
    Admission, AdmissionContext, AdmissionProfile, AdmissionStage, Arming, CostClass, Decline,
    ObjectiveRequirement, ParamSpace, SpecShape,
};
use crate::domain::SearchBox;
use crate::oracle::Oracle;
use crate::proposal::{Candidate, Effort, Proposal, StrategyName};
use crate::registry::{Budget, SearchState};
use crate::rng::Rng;
use crate::stall::{StallRule, WorkUnit};

/// `square`'s own free-dimension ceiling — **not** [`super::SPECIAL_MAX_FREE_DIMS`].
///
/// A batch costs `batch * n * fraction` coordinate writes, so unlike `special`
/// this strategy's cost is linear in the dimension and the ceiling is a real
/// budget guard rather than a memory guard. 262144 keeps the two measured
/// regimes (128 and 6912 free inputs) far inside, and covers a 224x224x3
/// image at 150528, while refusing the dimensions where a single batch would
/// eat the whole slice.
pub const SQUARE_MAX_FREE_DIMS: usize = 262_144;

/// Below two free dimensions a "block" of coordinates is a single coordinate
/// and this strategy is strictly worse than corner enumeration, which ny
/// already ships. Declining is the honest answer.
pub const SQUARE_MIN_FREE_DIMS: usize = 2;

/// Fraction of coordinates flipped in the first iteration of a restart.
pub const SQUARE_INITIAL_FRACTION: f64 = 0.5;

/// Geometric annealing applied to the block fraction each iteration.
pub const SQUARE_ANNEAL: f64 = 0.85;

/// Iterations before the search restarts from the shared incumbent.
pub const SQUARE_ITERATIONS_PER_RESTART: usize = 96;

/// Block-flip batches tolerated without a new best before abandoning.
///
/// Set the way `#bnn-lp-stall` was set: high enough that it provably changes
/// neither measured win. `soundnessbench model_36` was taken on batch 9 and
/// `traffic_signs model_48_idx_10495_eps_10` on batch 3, both far inside 32,
/// so this rule would have altered neither trajectory. What it does buy is the
/// flat-objective case, where after the first batch no margin ever improves and
/// the strategy would otherwise spend its entire slice: it now yields after
/// 32 batches instead of at the deadline, handing the remainder back.
pub const SQUARE_STALL_BATCHES: u64 = 32;

/// S9.
#[derive(Debug, Clone)]
pub struct Square {
    seed: u64,
}

impl Default for Square {
    fn default() -> Self {
        // The calibration's own seed, so a re-run of this crate against that
        // report starts from the same place the report's numbers came from.
        Self { seed: 20_260_808 }
    }
}

impl Square {
    /// Seed the strategy explicitly.
    pub const fn with_seed(seed: u64) -> Self {
        Self { seed }
    }

    /// Block size for iteration `k` of a restart, given `n` free dimensions.
    /// `max(1, int(n * fraction))` with `fraction = max(1/n, 0.5 * 0.85^k)`.
    pub fn block_size(n: usize, iteration: usize) -> usize {
        let mut fraction = SQUARE_INITIAL_FRACTION;
        let floor = 1.0 / (n.max(1) as f64);
        for _ in 0..iteration {
            fraction = (fraction * SQUARE_ANNEAL).max(floor);
        }
        ((n as f64 * fraction) as usize).max(1)
    }
}

impl crate::registry::Strategy for Square {
    fn name(&self) -> StrategyName {
        StrategyName::Square
    }

    fn deepest_stage(&self) -> AdmissionStage {
        AdmissionStage::SpecShape
    }

    /// `DEFAULT_PLAN`'s `("square", 0.16)`.
    fn plan_share(&self) -> f64 {
        0.16
    }

    fn stall_rule(&self) -> StallRule {
        StallRule::new(
            WorkUnit::BlockBatchesWithoutImprovement,
            SQUARE_STALL_BATCHES,
        )
    }

    fn admit(&self, ctx: &AdmissionContext<'_>) -> Admission {
        if ctx.arming() == Arming::Dark {
            return Admission::Declined(Decline::Disarmed);
        }
        let spec = ctx.spec();
        if let SpecShape::NonBoxInputAssertions { .. } = spec.shape {
            return Admission::Declined(Decline::SpecShapeUnsupported {
                want: SpecShape::BoxInputs,
                got: spec.shape,
            });
        }
        if spec.free_dims < SQUARE_MIN_FREE_DIMS {
            return Admission::Declined(Decline::FreeDimsBelowFloor {
                free: spec.free_dims,
                floor: SQUARE_MIN_FREE_DIMS,
            });
        }
        if spec.free_dims > SQUARE_MAX_FREE_DIMS {
            return Admission::Declined(Decline::FreeDimsAboveCeiling {
                free: spec.free_dims,
                ceiling: SQUARE_MAX_FREE_DIMS,
            });
        }
        Admission::Admitted(AdmissionProfile {
            // NOTE the asymmetry with an estimated-gradient strategy: `square`
            // does NOT decline on `ObjectiveQuality::Flat`. A flat objective is
            // the case it exists for, and the scheduler is expected to promote
            // it there rather than merely tolerate it.
            objective: ObjectiveRequirement::ValueOnly,
            needs_incumbent: false,
            needs_verifier_state: false,
            free_dims: spec.free_dims,
            declared_cost: CostClass::Openended,
            params: ParamSpace {
                free_dims_ceiling: SQUARE_MAX_FREE_DIMS,
                max_points: 1 << 22,
                max_restarts: usize::MAX,
            },
        })
    }

    fn search(
        &mut self,
        domain: &SearchBox,
        oracle: &mut dyn Oracle,
        budget: &Budget,
        state: &mut SearchState,
    ) -> Proposal {
        let started = std::time::Instant::now();
        let mut effort = Effort::default();
        let n = domain.free_dims();
        if n < SQUARE_MIN_FREE_DIMS {
            effort.wall = started.elapsed();
            return Proposal::Exhausted(effort);
        }

        let lo = domain.free_lo().to_vec();
        let hi = domain.free_hi().to_vec();
        let batch = budget.batch.max(1);
        let mut rng = Rng::new(self.seed);
        let mut scratch: Vec<usize> = Vec::new();
        let mut last_stall = 0u64;

        for restart in 0..budget.params.max_restarts {
            if budget.expired() || effort.points >= budget.params.max_points {
                break;
            }
            // Even restarts inherit; odd restarts start from the centre. See
            // the module docs: an inherited vertex incumbent turns every block
            // flip into a no-op.
            let mut current = if restart % 2 == 0 {
                state.best_free().to_vec()
            } else {
                domain.centre_free().to_vec()
            };
            let mut incumbent = f64::NEG_INFINITY;
            let mut fraction = SQUARE_INITIAL_FRACTION;
            // The stall tracker is PER WALK, exactly as `#bnn-lp-stall` is: it
            // abandons a walk that has paid its work without accepting
            // progress, and the next restart gets a fresh one.
            let mut stall = budget.stall_rule.tracker();

            for _ in 0..SQUARE_ITERATIONS_PER_RESTART {
                if budget.expired() || effort.points >= budget.params.max_points {
                    break;
                }
                let size = ((n as f64 * fraction) as usize).max(1);

                let mut rows: Vec<Vec<f64>> = Vec::with_capacity(batch);
                for _ in 0..batch {
                    let mut row = current.clone();
                    let picks = rng.choose_without_replacement(n, size, &mut scratch);
                    for pick in picks {
                        row[pick] = if rng.next_f64() < 0.5 {
                            hi[pick]
                        } else {
                            lo[pick]
                        };
                    }
                    rows.push(row);
                }

                // The candidate check happens here, BEFORE the stall counter is
                // consulted. That is what makes the stall rule a budget rule
                // and never a correctness one.
                match super::evaluate(
                    domain,
                    oracle,
                    state,
                    &mut effort,
                    StrategyName::Square,
                    &rows,
                ) {
                    super::Batch::Hit(candidate) => {
                        let mut hit_effort = effort.clone();
                        hit_effort.wall = started.elapsed();
                        hit_effort.stalled_units = stall.since_progress();
                        return Proposal::Candidate(Candidate::new(
                            candidate.inputs().to_vec(),
                            StrategyName::Square,
                            hit_effort,
                        ));
                    }
                    super::Batch::OracleFailed => {
                        effort.wall = started.elapsed();
                        effort.stalled_units = stall.since_progress();
                        return Proposal::Exhausted(effort);
                    }
                    super::Batch::Margins(margins) => {
                        let mut best_index = 0usize;
                        let mut best = f64::NEG_INFINITY;
                        for (index, &margin) in margins.iter().enumerate() {
                            if margin > best {
                                best = margin;
                                best_index = index;
                            }
                        }
                        let improved = best > incumbent;
                        if improved {
                            incumbent = best;
                            current.clone_from(&rows[best_index]);
                        }
                        stall.observe(improved);
                        if stall.stalled() {
                            break;
                        }
                    }
                }

                fraction = (fraction * SQUARE_ANNEAL).max(1.0 / n as f64);
            }
            last_stall = stall.since_progress();
        }

        effort.wall = started.elapsed();
        effort.stalled_units = last_stall;
        Proposal::Exhausted(effort)
    }
}
