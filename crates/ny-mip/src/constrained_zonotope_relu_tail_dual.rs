// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Replay-certified one-ReLU-tail bounds for a constrained zonotope.
//!
//! This deliberately unwired primitive bounds the declared margin
//!
//! ```text
//! bias + sum_i q_i ReLU(x_i),
//! x = c + G alpha + e,
//! alpha in [-1, 1]^m, C alpha <= d, |e_i| <= r_i.
//! ```
//!
//! The original entry point derives exact-dyadic coordinate enclosures from
//! the domain itself.  A separate entry point can intersect those enclosures
//! with [`CertifiedAuxiliaryBounds64`], whose concrete-witness proof obligation
//! remains caller-owned.  For each coordinate it then constructs an exact
//! rational affine minorant:
//!
//! - `u <= 0`: zero;
//! - `l >= 0`: the exact affine term `q_i x_i`;
//! - `l < 0 < u, q_i < 0`: the ReLU triangle upper chord multiplied by the
//!   negative coefficient; and
//! - `l < 0 < u, q_i > 0`: any directly chosen dyadic slope
//!   `0 <= k_i <= q_i`.
//!
//! A fixed exact line `A x + B` is represented to the outward evaluator by a
//! finite dyadic direction `k` plus the exact correction
//!
//! ```text
//! min_{x in [l,u]} (A-k)x + B.
//! ```
//!
//! Thus rounded `q`, chord slopes, products, and constants never acquire proof
//! authority.  Every candidate direction is evaluated first with zero
//! constraint multipliers through [`ConstrainedZonotope64::evaluate_dual`].
//! A supplied nonnegative multiplier is considered only afterward and retained
//! only when independent outward replay strictly improves the certified lower
//! bound.  Projected Adam is candidate search only.
//!
//! The module is not called by a CLI, verifier verdict, preset, or scored path.

use std::time::{Duration, Instant};

use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};

use crate::{CertifiedAuxiliaryBounds64, ConstrainedZonotope64, ConstrainedZonotope64Error};

/// Hard ceiling on the pre-ReLU value dimension accepted by the authority path.
pub const RELU_TAIL_DUAL_HARD_MAX_VALUE_DIM: usize = 131_072;
/// Hard ceiling on constrained-zonotope alpha symbols.
pub const RELU_TAIL_DUAL_HARD_MAX_ALPHA_DIM: usize = 8_192;
/// Hard ceiling on predicate rows.
pub const RELU_TAIL_DUAL_HARD_MAX_CONSTRAINTS: usize = 16_384;
/// Hard ceiling on dense predicate coefficients.
pub const RELU_TAIL_DUAL_HARD_MAX_CONSTRAINT_ELEMENTS: usize = 134_217_728;
/// Hard ceiling on sparse generator coefficients.
pub const RELU_TAIL_DUAL_HARD_MAX_GENERATOR_NONZEROS: usize = 134_217_728;
/// Hard ceiling on mandatory outward-evaluator scalar terms.
pub const RELU_TAIL_DUAL_HARD_MAX_BASELINE_TERMS: u64 = 8_589_934_592;
/// Maximum numerator or denominator bit length of one declared objective term.
pub const RELU_TAIL_DUAL_HARD_MAX_INPUT_RATIONAL_BITS: u64 = 8_192;
/// Maximum numerator or denominator bit length retained during exact arithmetic.
pub const RELU_TAIL_DUAL_HARD_MAX_INTERMEDIATE_RATIONAL_BITS: u64 = 32_768;
/// Hard ceiling on the aggregate bit lengths of a declared objective.
pub const RELU_TAIL_DUAL_HARD_MAX_TOTAL_RATIONAL_BITS: u64 = 16_777_216;
/// Hard ceiling on projected-Adam iterations.
pub const RELU_TAIL_DUAL_HARD_MAX_ITERATIONS: usize = 512;
/// Hard ceiling on positive unstable slopes optimized together.
pub const RELU_TAIL_DUAL_HARD_MAX_OPTIMIZABLE_SLOPES: usize = 131_072;
/// Hard ceiling on candidate-search scalar visits.
pub const RELU_TAIL_DUAL_HARD_MAX_SEARCH_WORK: u64 = 8_589_934_592;
/// Hard ceiling on caller-selected candidate-search wall time.
pub const RELU_TAIL_DUAL_HARD_MAX_WALL_TIME: Duration = Duration::from_mins(1);
/// Maximum scalar visits between candidate-only deadline checks.
const RELU_TAIL_DUAL_DEADLINE_CHECK_STRIDE: usize = 1_024;
/// Hard ceiling on independently optimized auxiliary-Box multipliers.
pub const RELU_TAIL_BOX_CUT_HARD_MAX_VARIABLES: usize = 262_144;
/// Hard ceiling on zero-start auxiliary-Box Adam schedules.
pub const RELU_TAIL_BOX_CUT_HARD_MAX_RESTARTS: usize = 2;
/// Hard ceiling on aggregate auxiliary-Box Adam updates.
pub const RELU_TAIL_BOX_CUT_HARD_MAX_ITERATIONS: usize = 512;
/// Hard ceiling on exact auxiliary-Box candidate replays.
pub const RELU_TAIL_BOX_CUT_HARD_MAX_EXACT_REPLAYS: usize = 2;
/// Hard ceiling on candidate-only auxiliary-Box scalar visits.
pub const RELU_TAIL_BOX_CUT_HARD_MAX_SEARCH_WORK: u64 = 8_589_934_592;
/// Hard ceiling on the auxiliary-Box candidate-search wall clock.
pub const RELU_TAIL_BOX_CUT_HARD_MAX_WALL_TIME: Duration = Duration::from_mins(1);
/// Largest accepted projected multiplier.
pub const RELU_TAIL_BOX_CUT_HARD_MAX_MULTIPLIER: f64 = 1_048_576.0;
const RELU_TAIL_BOX_CUT_ADAM_BETA1: f64 = 0.9;
const RELU_TAIL_BOX_CUT_ADAM_BETA2: f64 = 0.999;
const RELU_TAIL_BOX_CUT_ADAM_EPSILON: f64 = 1e-8;

/// Exact declared output margin.
///
/// Fields are private so construction always applies the rational size caps.
/// The values define the objective being bounded; they are not auxiliary proof
/// facts about the domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactReluTailMargin {
    coefficients: Vec<BigRational>,
    bias: BigRational,
}

impl ExactReluTailMargin {
    /// Construct an exact rational margin after enforcing authority-path caps.
    pub fn try_new(
        coefficients: Vec<BigRational>,
        bias: BigRational,
    ) -> Result<Self, ReluTailDualError> {
        check_declared_rationals(&coefficients, &bias)?;
        Ok(Self { coefficients, bias })
    }

    /// Exact margin coefficients in value-axis order.
    #[must_use]
    pub fn coefficients(&self) -> &[BigRational] {
        &self.coefficients
    }

    /// Exact affine bias.
    #[must_use]
    pub fn bias(&self) -> &BigRational {
        &self.bias
    }
}

/// Convert two finite binary64 output rows to an exact margin.
///
/// Each source coefficient is promoted to its exact IEEE-754 dyadic value
/// before subtraction.  In particular, this never uses the rounded operation
/// `target[i] - challenger[i]` or `target_bias - challenger_bias`.
pub fn exact_relu_tail_margin_from_f64_rows(
    target: &[f64],
    challenger: &[f64],
    target_bias: f64,
    challenger_bias: f64,
) -> Result<ExactReluTailMargin, ReluTailDualError> {
    if challenger.len() != target.len() {
        return Err(ReluTailDualError::Shape {
            field: "challenger row",
            expected: target.len(),
            got: challenger.len(),
        });
    }
    require_resource_limit(
        "output-row width",
        u64::try_from(target.len()).unwrap_or(u64::MAX),
        RELU_TAIL_DUAL_HARD_MAX_VALUE_DIM as u64,
    )?;
    let mut coefficients = Vec::new();
    try_reserve(&mut coefficients, target.len(), "exact margin coefficients")?;
    for (index, (&target_value, &challenger_value)) in target.iter().zip(challenger).enumerate() {
        let target_exact = exact_objective_f64(target_value, "target row", index)?;
        let challenger_exact = exact_objective_f64(challenger_value, "challenger row", index)?;
        coefficients.push(checked_rational(
            target_exact - challenger_exact,
            "output-row coefficient subtraction",
            index,
        )?);
    }
    let target_bias = exact_objective_f64(target_bias, "target bias", 0)?;
    let challenger_bias = exact_objective_f64(challenger_bias, "challenger bias", 0)?;
    let bias = checked_rational(
        target_bias - challenger_bias,
        "output-row bias subtraction",
        0,
    )?;
    ExactReluTailMargin::try_new(coefficients, bias)
}

/// Caller-tightenable limits for untrusted slope search.
///
/// Values above the hard ceilings are malformed and cause a certified-baseline
/// fallback.  These limits never authorize mandatory proof work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReluTailDualLimits {
    /// Maximum value dimension searched.
    pub max_value_dim: usize,
    /// Maximum alpha symbols searched.
    pub max_alpha_dim: usize,
    /// Maximum predicate rows searched.
    pub max_constraints: usize,
    /// Maximum generator nonzeros searched per iteration.
    pub max_generator_nonzeros: usize,
    /// Maximum positive unstable slopes optimized together.
    pub max_optimizable_slopes: usize,
    /// Maximum projected-Adam iterations.
    pub max_iterations: usize,
    /// Maximum conservatively counted scalar visits.
    pub max_search_work: u64,
    /// Maximum candidate-search wall time.
    pub max_wall_time: Duration,
}

/// Candidate-only projected-Adam configuration.
///
/// There is intentionally no `Default`: an experimental caller must choose an
/// explicit budget.  The primitive itself remains unwired and default-off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReluTailDualConfig {
    /// Number of projected-Adam updates.  Zero disables all non-baseline slopes.
    pub iterations: usize,
    /// Adam ascent learning rate.
    pub learning_rate: f64,
    /// First-moment decay in `[0,1)`.
    pub beta1: f64,
    /// Second-moment decay in `[0,1)`.
    pub beta2: f64,
    /// Positive denominator stabilizer.
    pub epsilon: f64,
    /// Candidate-search wall time, excluding exact setup and outward replay.
    pub wall_time: Duration,
    /// Caller-tightenable heuristic limits.
    pub limits: ReluTailDualLimits,
}

/// Checked resource plan for candidate-only slope search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReluTailDualPlan {
    /// Pre-ReLU value dimension.
    pub value_dim: usize,
    /// Constrained-zonotope alpha dimension.
    pub alpha_dim: usize,
    /// Predicate rows.
    pub constraints: usize,
    /// Sparse generator nonzeros visited per projected step.
    pub generator_nonzeros: usize,
    /// Positive unstable direct slopes.
    pub optimizable_slopes: usize,
    /// Planned Adam updates.
    pub iterations: usize,
    /// Conservative candidate-search scalar visits.
    pub search_work: u64,
}

/// Why the untrusted slope lane did or did not finish.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReluTailDualStatus {
    /// Endpoint, canonical, and projected-Adam candidates were replayed.
    Completed,
    /// No positive unstable coefficient exists.
    NoOptimizableSlopes,
    /// The caller explicitly selected zero iterations.
    SearchDisabled,
    /// Configuration or a caller-selected resource cap rejected search.
    ResourceFallback,
    /// Candidate search reached its wall-time budget.
    Deadline,
    /// Candidate-only arithmetic produced NaN or infinity.
    NonFiniteCandidate,
    /// A bounded candidate-only allocation failed.
    AllocationFallback,
}

/// Per-direction CPU outward replays with zero predicate multipliers.
///
/// These are slope-candidate measurements, not predicate-multiplier
/// candidates.  Every present value has passed the same exact-constant
/// combination and outward rounding as the accepted certificate.  An optional
/// field is `None` when that direction was not attempted or its replay was not
/// accepted, including replay-storage allocation fallback.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReluTailDualZeroPredicateCandidateReplays {
    /// Mandatory replay with every positive unstable ReLU slope set to zero.
    ///
    /// This is present in every successful result.  It is distinct from
    /// [`ReluTailDualResult::zero_multiplier_lower_bound`], which replays the
    /// final accepted direction without predicate multipliers.
    pub zero_positive_slope_lower_bound: f64,
    /// Replay with every positive unstable slope at its exact-safe upper end.
    pub upper_endpoint_lower_bound: Option<f64>,
    /// Replay with every positive unstable slope at its canonical triangle
    /// value.
    pub canonical_lower_bound: Option<f64>,
    /// Replay of the best direction proposed by the bounded Adam search.
    pub optimized_lower_bound: Option<f64>,
}

/// Certified result for the declared one-ReLU-tail margin.
#[derive(Clone, Debug, PartialEq)]
pub struct ReluTailDualResult {
    /// Best finite lower certificate, including exact bias/corrections.
    pub lower_bound: f64,
    /// Same accepted direction replayed with zero predicate multipliers.
    pub zero_multiplier_lower_bound: f64,
    /// Candidate-attributed CPU outward replays with zero predicate
    /// multipliers.
    pub zero_predicate_candidate_replays: ReluTailDualZeroPredicateCandidateReplays,
    /// Accepted finite dyadic direction over pre-ReLU coordinates.
    pub direction: Vec<f64>,
    /// Accepted finite nonnegative predicate multipliers.
    pub multipliers: Vec<f64>,
    /// Exact bias plus every fixed-line coefficient correction.
    pub exact_constant: BigRational,
    /// Number of positive unstable direct slopes.
    pub optimizable_slopes: usize,
    /// Successfully constructed and independently replayed directions,
    /// including the baseline.
    pub candidates_replayed: usize,
    /// Fully completed projected-Adam updates.
    pub iterations_completed: usize,
    /// Candidate-search outcome.
    pub status: ReluTailDualStatus,
    /// Checked search plan, absent when configuration/caps reject search.
    pub plan: Option<ReluTailDualPlan>,
    /// Whether the accepted certificate uses the supplied multipliers.
    pub supplied_multipliers_used: bool,
}

/// Exact coordinate geometry prepared once for many one-ReLU-tail margins.
///
/// The private borrow is the provenance link: a prepared value can only replay
/// certificates against the exact [`ConstrainedZonotope64`] from which its
/// coordinate hull was derived.  It cannot be detached from that domain or
/// supplied alongside a lookalike domain.  The domain is borrowed immutably,
/// so its center, generators, constraints, and box remainder cannot change
/// while this value is live.
///
/// Preparation performs the expensive exact-dyadic coordinate-radius pass
/// once.  [`Self::bound_margin_unwired`] still constructs each margin's exact
/// ReLU minorant and independently performs the mandatory CPU outward replay,
/// candidate portfolio, and optional supplied-multiplier replay.  Cached
/// geometry therefore removes repeated setup work without becoming proof
/// authority independent of the original domain.
///
/// This experimental type is not wired to a CLI, verifier verdict, preset, or
/// scored path.
///
/// The domain borrow cannot escape its source:
///
/// ```compile_fail
/// use ny_mip::{
///     prepare_relu_tail_triangle_dual_unwired, ConstrainedZonotope64,
///     PreparedReluTailGeometry64,
/// };
///
/// fn detached<'a>() -> PreparedReluTailGeometry64<'a> {
///     let domain =
///         ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
///     prepare_relu_tail_triangle_dual_unwired(&domain).unwrap()
/// }
/// ```
pub struct PreparedReluTailGeometry64<'domain> {
    domain: &'domain ConstrainedZonotope64,
    exact_coordinate_bounds: Vec<(BigRational, BigRational)>,
    generator_nonzeros: usize,
}

impl PreparedReluTailGeometry64<'_> {
    /// Domain value dimension covered by the prepared exact hull.
    #[must_use]
    pub fn value_dim(&self) -> usize {
        self.exact_coordinate_bounds.len()
    }

    /// Sparse generator additions paid once by exact hull preparation.
    ///
    /// Calling the legacy entry point for `n` margins performs this many
    /// additions `n` times.  A prepared value performs them once, regardless
    /// of how many margins are subsequently replayed.
    #[must_use]
    pub fn coordinate_hull_generator_additions(&self) -> usize {
        self.generator_nonzeros
    }

    /// Bound one exact margin using this domain-tied prepared geometry.
    ///
    /// Every call retains the original mandatory portfolio semantics: the
    /// zero-positive-slope direction is replayed first with zero predicate
    /// multipliers, search candidates cannot replace that authority path, and
    /// supplied multipliers are accepted only after an independent replay.
    ///
    /// # Errors
    ///
    /// Returns [`ReluTailDualError`] when the declared margin is malformed,
    /// exact line construction exceeds an immutable cap, or mandatory replay
    /// fails.  A margin with a dimension different from [`Self::value_dim`] is
    /// rejected before any replay.
    pub fn bound_margin_unwired(
        &self,
        margin: &ExactReluTailMargin,
        supplied_multipliers: Option<&[f64]>,
        config: ReluTailDualConfig,
    ) -> Result<ReluTailDualResult, ReluTailDualError> {
        check_mandatory_margin_resources(self.domain.value_dim(), margin)?;
        let line_plan =
            build_line_plan_from_bounds(self.domain, margin, &self.exact_coordinate_bounds)?;
        bound_relu_tail_triangle_dual_from_line_plan(
            self.domain,
            line_plan,
            self.generator_nonzeros,
            supplied_multipliers,
            config,
        )
    }

    /// Bound one exact margin using caller-certified auxiliary bounds.
    ///
    /// This is the prepared counterpart of
    /// [`bound_relu_tail_triangle_dual_with_auxiliary_bounds_unwired`].  It
    /// intersects `auxiliary` with the cached exact coordinate hull, so it
    /// does not revisit any sparse generator coefficient.  Directional replay
    /// still minimizes over the original domain borrowed by this value.
    ///
    /// The caller owns the same semantic proof obligation as the direct API:
    /// every concrete preactivation witness must be inside `auxiliary`.
    /// Auxiliary line changes are sound but not monotone when replayed over
    /// the larger zonotope.  Callers must therefore portfolio this result with
    /// [`Self::bound_margin_unwired`] and retain the better independently
    /// replayed lower bound.
    ///
    /// # Errors
    ///
    /// Returns [`ReluTailDualError::AuxiliaryDimensionMismatch`] when the
    /// auxiliary value dimension differs, or
    /// [`ReluTailDualError::EmptyAuxiliaryIntersection`] when an auxiliary
    /// interval is disjoint from the cached exact coordinate hull.  Margin,
    /// exact-line, and mandatory-replay failures match
    /// [`Self::bound_margin_unwired`].
    pub fn bound_margin_with_auxiliary_bounds_unwired(
        &self,
        auxiliary: &CertifiedAuxiliaryBounds64,
        margin: &ExactReluTailMargin,
        supplied_multipliers: Option<&[f64]>,
        config: ReluTailDualConfig,
    ) -> Result<ReluTailDualResult, ReluTailDualError> {
        if auxiliary.value_dim() != self.domain.value_dim() {
            return Err(ReluTailDualError::AuxiliaryDimensionMismatch {
                expected: self.domain.value_dim(),
                got: auxiliary.value_dim(),
            });
        }
        check_mandatory_margin_resources(self.domain.value_dim(), margin)?;
        let bounds = exact_coordinate_bounds_with_auxiliary_from_bounds(
            &self.exact_coordinate_bounds,
            auxiliary,
        )?;
        let line_plan = build_line_plan_from_bounds(self.domain, margin, &bounds)?;
        bound_relu_tail_triangle_dual_from_line_plan(
            self.domain,
            line_plan,
            self.generator_nonzeros,
            supplied_multipliers,
            config,
        )
    }

    /// Search auxiliary-Box multipliers and replay at most two exact cuts.
    ///
    /// This prepared-only M24 experiment evaluates the mandatory original
    /// M17 certificate first and the ordinary auxiliary M20 certificate
    /// second.  Only then may projected Adam propose nonnegative multipliers
    /// for strictly tighter auxiliary endpoints.  Approximate scores have no
    /// proof authority: every selectable Box cut is reconstructed with exact
    /// dyadic multiplier arithmetic, repaired over this prepared value's
    /// original CZ hull, and evaluated by the CPU outward dual path.  Supplied
    /// predicate multipliers are replayed independently after that zero-
    /// predicate replay.
    ///
    /// Selection is a strict ordered maximum, so ties retain M17 before M20
    /// before the first exactly replayed M24 candidate.  Invalid candidate
    /// data, limits, deadlines, allocations, and replay failures cannot
    /// remove either already-certified result.  The caller continues to own
    /// the semantic proof that every concrete witness lies in `auxiliary`.
    ///
    /// This method is deliberately unwired from commands, presets, verifier
    /// verdicts, and scored paths.
    ///
    /// # Errors
    ///
    /// Returns [`ReluTailDualError`] only when mandatory M17 setup or replay
    /// fails.  M20 and all optimizer failures are represented in the returned
    /// telemetry while preserving M17.
    pub fn bound_margin_with_optimized_auxiliary_box_cut_unwired(
        &self,
        auxiliary: &CertifiedAuxiliaryBounds64,
        margin: &ExactReluTailMargin,
        supplied_predicate_multipliers: Option<&[f64]>,
        relu_config: ReluTailDualConfig,
        box_config: ReluTailBoxCutOptimizerConfig,
    ) -> Result<ReluTailBoxCutOptimizedResult, ReluTailDualError> {
        // Mandatory authority is deliberately first and is never hidden by an
        // auxiliary shape/configuration defect.
        let original =
            self.bound_margin_unwired(margin, supplied_predicate_multipliers, relu_config)?;

        if auxiliary.value_dim() != self.domain.value_dim() {
            return Ok(finish_optimized_box_cut_result(
                original,
                None,
                None,
                ReluTailBoxCutOptimizerStatus::AuxiliaryFallback,
                None,
                BoxSearchTelemetry::default(),
            ));
        }

        let auxiliary_bounds = match exact_coordinate_bounds_with_auxiliary_from_bounds(
            &self.exact_coordinate_bounds,
            auxiliary,
        ) {
            Ok(bounds) => bounds,
            Err(_) => {
                return Ok(finish_optimized_box_cut_result(
                    original,
                    None,
                    None,
                    ReluTailBoxCutOptimizerStatus::AuxiliaryFallback,
                    None,
                    BoxSearchTelemetry::default(),
                ));
            }
        };
        let auxiliary_line =
            match build_line_plan_from_bounds(self.domain, margin, &auxiliary_bounds) {
                Ok(plan) => plan,
                Err(_) => {
                    return Ok(finish_optimized_box_cut_result(
                        original,
                        None,
                        None,
                        ReluTailBoxCutOptimizerStatus::AuxiliaryFallback,
                        None,
                        BoxSearchTelemetry::default(),
                    ));
                }
            };
        let auxiliary_result = match bound_relu_tail_triangle_dual_from_line_plan(
            self.domain,
            auxiliary_line,
            self.generator_nonzeros,
            supplied_predicate_multipliers,
            relu_config,
        ) {
            Ok(result) => result,
            Err(_) => {
                return Ok(finish_optimized_box_cut_result(
                    original,
                    None,
                    None,
                    ReluTailBoxCutOptimizerStatus::AuxiliaryFallback,
                    None,
                    BoxSearchTelemetry::default(),
                ));
            }
        };

        let box_variables =
            match count_tighter_auxiliary_box_endpoints(&self.exact_coordinate_bounds, auxiliary) {
                Ok(count) => count,
                Err(_) => {
                    return Ok(finish_optimized_box_cut_result(
                        original,
                        Some(auxiliary_result),
                        None,
                        ReluTailBoxCutOptimizerStatus::NonFiniteCandidate,
                        None,
                        BoxSearchTelemetry::default(),
                    ));
                }
            };
        if box_variables == 0 {
            return Ok(finish_optimized_box_cut_result(
                original,
                Some(auxiliary_result),
                None,
                ReluTailBoxCutOptimizerStatus::NoTighterAuxiliaryBox,
                None,
                BoxSearchTelemetry::default(),
            ));
        }

        let plan = match ReluTailBoxCutOptimizerPlan::checked(
            self.domain,
            self.generator_nonzeros,
            box_variables,
            box_config,
        ) {
            Ok(plan) => plan,
            Err(status) => {
                return Ok(finish_optimized_box_cut_result(
                    original,
                    Some(auxiliary_result),
                    None,
                    status,
                    None,
                    BoxSearchTelemetry::default(),
                ));
            }
        };
        if plan.total_iterations == 0 {
            return Ok(finish_optimized_box_cut_result(
                original,
                Some(auxiliary_result),
                None,
                ReluTailBoxCutOptimizerStatus::SearchDisabled,
                Some(plan),
                BoxSearchTelemetry::default(),
            ));
        }

        let search = optimize_auxiliary_box_multipliers(
            self.domain,
            &self.exact_coordinate_bounds,
            auxiliary,
            &auxiliary_result.direction,
            box_config,
            plan,
        );
        let mut best_box_cut = None;
        let mut exact_replays = 0_usize;
        let mut exact_replay_failed = false;
        for candidate in search.candidates.iter().take(plan.exact_replays) {
            exact_replays += 1;
            #[cfg(test)]
            let force_replay_failure =
                BOX_OPTIMIZER_FAIL_NEXT_EXACT_REPLAY.with(|fail| fail.replace(false));
            #[cfg(not(test))]
            let force_replay_failure = false;
            let replay = if force_replay_failure {
                Err(ReluTailDualError::NonFiniteArithmetic {
                    coordinate: 0,
                    operation: "injected optimized Box exact replay failure",
                })
            } else {
                expand_box_candidate(&search.variables, candidate, self.domain.value_dim())
                    .and_then(|(upper, lower)| {
                        build_auxiliary_box_cut_certificate_with_original_hull(
                            self.domain,
                            auxiliary,
                            &auxiliary_result,
                            &upper,
                            &lower,
                            supplied_predicate_multipliers,
                            &self.exact_coordinate_bounds,
                        )
                    })
            };
            match replay {
                Ok(certificate) => {
                    if best_box_cut
                        .as_ref()
                        .is_none_or(|best: &ReluTailBoxCutCertificate| {
                            certificate.lower_bound > best.lower_bound
                        })
                    {
                        best_box_cut = Some(certificate);
                    }
                }
                Err(_) => exact_replay_failed = true,
            }
        }

        let search_status = if exact_replay_failed {
            ReluTailBoxCutOptimizerStatus::ExactReplayFallback
        } else {
            search.status
        };
        Ok(finish_optimized_box_cut_result(
            original,
            Some(auxiliary_result),
            best_box_cut,
            search_status,
            Some(plan),
            BoxSearchTelemetry {
                iterations_completed: search.iterations_completed,
                restarts_completed: search.restarts_completed,
                candidates_scored: search.candidates_scored,
                exact_replays,
            },
        ))
    }
}

/// Portfolio member selected by the auxiliary-Box cut experiment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReluTailBoxCutSelection {
    /// The mandatory original M17 certificate.
    Original,
    /// The ordinary M20 auxiliary-line certificate without Box multipliers.
    Auxiliary,
    /// The independently replayed auxiliary-Box cut certificate.
    BoxCut,
}

/// Outcome of the optional auxiliary-Box cut lane.
///
/// Every fallback retains the mandatory original M17 result, and an available
/// M20 result, in the returned portfolio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReluTailBoxCutStatus {
    /// The cut direction and exact correction were independently replayed.
    Completed,
    /// M20 rejected the auxiliary bounds or its mandatory setup failed.
    AuxiliaryFallback,
    /// The two Box multiplier vectors did not both cover the value axis.
    InvalidBoxMultiplierShape {
        /// Required length of both vectors.
        expected: usize,
        /// Supplied upper-cut multiplier count.
        upper_got: usize,
        /// Supplied lower-cut multiplier count.
        lower_got: usize,
    },
    /// One Box multiplier was negative, NaN, or infinite.
    InvalidBoxMultiplierValue {
        /// Whether the malformed value belongs to the upper-cut vector.
        upper: bool,
        /// Coordinate of the malformed value.
        coordinate: usize,
    },
    /// Optional exact setup, finite conversion, allocation, or replay failed.
    CandidateFallback,
}

/// One replay-certified Box-cut candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct ReluTailBoxCutCertificate {
    /// Best finite lower certificate for this cut direction.
    pub lower_bound: f64,
    /// The same cut direction replayed with zero predicate multipliers.
    pub zero_predicate_lower_bound: f64,
    /// Finite dyadic direction actually consumed by the CZ evaluator.
    pub replay_direction: Vec<f64>,
    /// Nonnegative multipliers on `x_i - upper_i <= 0`.
    pub upper_box_multipliers: Vec<f64>,
    /// Nonnegative multipliers on `lower_i - x_i <= 0`.
    pub lower_box_multipliers: Vec<f64>,
    /// Predicate multipliers used by the accepted replay.
    pub predicate_multipliers: Vec<f64>,
    /// M20 line constant, Box constants, and exact direction-rounding repair.
    pub exact_constant: BigRational,
    /// Whether nonzero supplied predicate multipliers improved this cut.
    pub supplied_predicate_multipliers_used: bool,
}

/// Mandatory M17/M20/Box-cut portfolio result.
#[derive(Clone, Debug, PartialEq)]
pub struct ReluTailBoxCutDualResult {
    /// Largest independently certified lower bound in the portfolio.
    pub lower_bound: f64,
    /// Portfolio member that attained [`Self::lower_bound`].
    pub selected: ReluTailBoxCutSelection,
    /// Mandatory original M17 result, always evaluated first.
    pub original: ReluTailDualResult,
    /// Ordinary M20 result when auxiliary setup succeeded.
    pub auxiliary: Option<ReluTailDualResult>,
    /// Independently replayed cut candidate when optional setup succeeded.
    pub box_cut: Option<ReluTailBoxCutCertificate>,
    /// Outcome of the optional cut lane.
    pub status: ReluTailBoxCutStatus,
}

/// One zero-start projected-Adam schedule for auxiliary-Box multipliers.
///
/// The two intended winner-inspired schedules use initial learning rates
/// `0.005` and `0.1`.  A caller may split the bounded iteration budget and
/// choose the geometric decay explicitly.  Adam's moment parameters remain
/// fixed at `(0.9, 0.999, 1e-8)` so they cannot become proof inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReluTailBoxCutAdamSchedule {
    /// Fully completed projected updates attempted by this zero-start run.
    pub iterations: usize,
    /// Initial ascent learning rate.
    pub learning_rate: f64,
    /// Per-update geometric learning-rate decay in `(0, 1]`.
    pub decay: f64,
}

/// Caller-tightenable limits for optional auxiliary-Box candidate search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReluTailBoxCutOptimizerLimits {
    /// Maximum value dimension searched.
    pub max_value_dim: usize,
    /// Maximum independently optimized upper/lower Box multipliers.
    pub max_box_variables: usize,
    /// Maximum aggregate updates across both schedules.
    pub max_total_iterations: usize,
    /// Maximum nonempty zero-start schedules.
    pub max_restarts: usize,
    /// Maximum exact candidate replays after approximate search.
    pub max_exact_replays: usize,
    /// Maximum sparse generator coefficients visited per score.
    pub max_generator_nonzeros: usize,
    /// Maximum conservatively counted candidate-only scalar visits.
    pub max_search_work: u64,
    /// Maximum candidate-search wall time.
    pub max_wall_time: Duration,
}

/// Explicit, default-off configuration for auxiliary-Box multiplier search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReluTailBoxCutOptimizerConfig {
    /// Two bounded zero-start schedules, conventionally `(0.005, 0.1)`.
    pub schedules: [ReluTailBoxCutAdamSchedule; 2],
    /// Projection interval upper endpoint for every multiplier.
    pub multiplier_cap: f64,
    /// Candidate-only wall clock, excluding M17/M20 and exact replay.
    pub wall_time: Duration,
    /// Caller-tightenable heuristic limits.
    pub limits: ReluTailBoxCutOptimizerLimits,
}

/// Checked resource plan for auxiliary-Box multiplier candidate search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReluTailBoxCutOptimizerPlan {
    /// Pre-ReLU value dimension.
    pub value_dim: usize,
    /// Constrained-zonotope alpha dimension.
    pub alpha_dim: usize,
    /// Sparse generator coefficients visited per approximate score.
    pub generator_nonzeros: usize,
    /// Strictly useful upper/lower auxiliary endpoints represented by a
    /// projected multiplier.
    pub box_variables: usize,
    /// Nonempty zero-start schedules.
    pub restarts: usize,
    /// Aggregate planned Adam updates.
    pub total_iterations: usize,
    /// Maximum exact replays planned after approximate search.
    pub exact_replays: usize,
    /// Conservative candidate-only scalar visits.
    pub search_work: u64,
}

/// Outcome of the untrusted auxiliary-Box multiplier search lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReluTailBoxCutOptimizerStatus {
    /// Every requested schedule completed and its best candidate was replayed.
    Completed,
    /// Both schedules explicitly requested zero iterations.
    SearchDisabled,
    /// No auxiliary endpoint is strictly tighter than the prepared CZ hull.
    NoTighterAuxiliaryBox,
    /// M20 auxiliary setup failed; M17 remains authoritative.
    AuxiliaryFallback,
    /// A malformed numeric setting or hard-ceiling violation disabled search.
    InvalidConfig,
    /// A caller-selected limit rejected the checked work plan.
    ResourceFallback,
    /// Candidate search reached its bounded wall clock.
    Deadline,
    /// Approximate candidate arithmetic produced NaN or infinity.
    NonFiniteCandidate,
    /// A fallible candidate-only allocation failed.
    AllocationFallback,
    /// At least one proposed candidate failed exact outward replay.
    ExactReplayFallback,
}

/// Prepared M17/M20 plus optional replay-certified optimized Box cut.
#[derive(Clone, Debug, PartialEq)]
pub struct ReluTailBoxCutOptimizedResult {
    /// Largest independently certified lower bound in [`Self::portfolio`].
    pub lower_bound: f64,
    /// Strict, stable M17 then M20 then Box-cut selection.
    pub selected: ReluTailBoxCutSelection,
    /// Existing M22 portfolio container; exact replay remains its authority.
    pub portfolio: ReluTailBoxCutDualResult,
    /// Candidate-search outcome, separate from the M22 replay status.
    pub search_status: ReluTailBoxCutOptimizerStatus,
    /// Checked candidate work plan, absent when validation rejected it.
    pub search_plan: Option<ReluTailBoxCutOptimizerPlan>,
    /// Fully completed Adam updates across schedules.
    pub iterations_completed: usize,
    /// Fully completed zero-start schedules.
    pub restarts_completed: usize,
    /// Finite approximate objectives completed, including each zero start.
    pub candidates_scored: usize,
    /// Exact CPU outward replays attempted, bounded by the checked plan.
    pub exact_replays: usize,
}

/// Failure of exact setup or the mandatory zero-multiplier authority path.
///
/// Heuristic failures are represented by [`ReluTailDualStatus`] and retain the
/// already-certified baseline.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ReluTailDualError {
    /// A one-dimensional input has the wrong length.
    #[error("shape mismatch for {field}: expected {expected}, got {got}")]
    Shape {
        /// Input whose length is wrong.
        field: &'static str,
        /// Required length.
        expected: usize,
        /// Supplied length.
        got: usize,
    },
    /// Auxiliary bounds do not cover the domain value axis.
    #[error("auxiliary bounds have value dimension {got}; expected {expected}")]
    AuxiliaryDimensionMismatch {
        /// Constrained-zonotope value dimension.
        expected: usize,
        /// Auxiliary-bound dimension.
        got: usize,
    },
    /// An auxiliary interval is disjoint from the exact unconstrained CZ hull.
    #[error(
        "auxiliary bounds have empty intersection with the CZ hull at coordinate {coordinate}"
    )]
    EmptyAuxiliaryIntersection {
        /// Coordinate with a structurally inconsistent intersection.
        coordinate: usize,
    },
    /// An output-row source value is NaN or infinite.
    #[error("{field}[{index}] must be finite")]
    NonFiniteObjective {
        /// Source row or bias.
        field: &'static str,
        /// Flattened index.
        index: usize,
    },
    /// A mandatory resource exceeded its immutable ceiling.
    #[error("mandatory resource {resource} is {actual}, above hard limit {limit}")]
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Checked required amount.
        actual: u64,
        /// Immutable hard ceiling.
        limit: u64,
    },
    /// A resource calculation overflowed.
    #[error("resource size overflow while computing {resource}")]
    ResourceOverflow {
        /// Failed calculation.
        resource: &'static str,
    },
    /// A declared rational is too large for the authority path.
    #[error("{field}[{index}] has {bits} bits, above hard limit {limit}")]
    RationalInputLimit {
        /// Objective field.
        field: &'static str,
        /// Logical index.
        index: usize,
        /// Larger of numerator/denominator bit lengths.
        bits: u64,
        /// Immutable bit ceiling.
        limit: u64,
    },
    /// Exact arithmetic grew beyond its immutable intermediate cap.
    #[error("exact rational growth at coordinate {coordinate} while computing {operation}: {bits} bits above {limit}")]
    RationalGrowthLimit {
        /// Logical coordinate.
        coordinate: usize,
        /// Exact operation.
        operation: &'static str,
        /// Larger bit length.
        bits: u64,
        /// Immutable bit ceiling.
        limit: u64,
    },
    /// Bounded authority-path storage could not be reserved.
    #[error("unable to reserve mandatory storage for {resource}")]
    AllocationFailure {
        /// Requested storage.
        resource: &'static str,
    },
    /// No finite binary64 representation can support a useful certificate.
    #[error("non-finite arithmetic at coordinate {coordinate} while computing {operation}")]
    NonFiniteArithmetic {
        /// Logical coordinate.
        coordinate: usize,
        /// Failed conversion or combination.
        operation: &'static str,
    },
    /// The mandatory zero-multiplier replay failed closed.
    #[error("mandatory zero-multiplier replay failed: {0}")]
    Baseline(ConstrainedZonotope64Error),
}

/// Prepare exact coordinate geometry for repeated one-ReLU-tail margins.
///
/// Exact coordinate bounds deliberately ignore predicate constraints, matching
/// [`bound_relu_tail_triangle_dual_unwired`].  Constraints remain part of every
/// outward directional replay.  The returned value borrows `domain`, making
/// the geometry's provenance structural rather than a caller-supplied token.
///
/// # Errors
///
/// Returns [`ReluTailDualError`] if the domain exceeds immutable authority-path
/// limits, exact dyadic coordinate accumulation exceeds its rational cap, or
/// mandatory geometry storage cannot be reserved.
pub fn prepare_relu_tail_triangle_dual_unwired(
    domain: &ConstrainedZonotope64,
) -> Result<PreparedReluTailGeometry64<'_>, ReluTailDualError> {
    let generator_nonzeros = check_mandatory_domain_resources(domain)?;
    let exact_coordinate_bounds = exact_coordinate_bounds(domain)?;
    Ok(PreparedReluTailGeometry64 {
        domain,
        exact_coordinate_bounds,
        generator_nonzeros,
    })
}

/// Bound one exact margin after a single coordinatewise ReLU.
///
/// Coordinate enclosures are recomputed internally from `domain` using exact
/// dyadic arithmetic.  Predicate constraints are deliberately ignored for
/// those enclosures, which can weaken classification but cannot make it
/// unsound.  They are used again by outward directional replay.
///
/// `supplied_multipliers` is optional candidate data.  A wrong length,
/// negative value, NaN, infinity, or replay failure simply makes it unusable;
/// it never hides the mandatory zero-multiplier baseline.
///
/// # Errors
///
/// Returns [`ReluTailDualError`] only when exact setup or the mandatory
/// baseline cannot be completed within immutable authority-path limits.
pub fn bound_relu_tail_triangle_dual_unwired(
    domain: &ConstrainedZonotope64,
    margin: &ExactReluTailMargin,
    supplied_multipliers: Option<&[f64]>,
    config: ReluTailDualConfig,
) -> Result<ReluTailDualResult, ReluTailDualError> {
    let generator_nonzeros = check_mandatory_resources(domain, margin)?;
    let line_plan = build_line_plan(domain, margin)?;
    bound_relu_tail_triangle_dual_from_line_plan(
        domain,
        line_plan,
        generator_nonzeros,
        supplied_multipliers,
        config,
    )
}

fn bound_relu_tail_triangle_dual_from_line_plan(
    domain: &ConstrainedZonotope64,
    line_plan: LinePlan,
    generator_nonzeros: usize,
    supplied_multipliers: Option<&[f64]>,
    config: ReluTailDualConfig,
) -> Result<ReluTailDualResult, ReluTailDualError> {
    let mut zero = zero_f64(domain.constraint_count(), "zero multipliers")?;
    // Canonicalize every structural zero, including platforms which preserve
    // a negative-zero payload through allocation helpers.
    zero.fill(0.0);

    let supplied = valid_supplied_multipliers(supplied_multipliers, domain.constraint_count());
    let baseline_direction = clone_f64(&line_plan.fixed_direction, "baseline direction")?;

    // This call is deliberately first.  Search configuration and supplied
    // multiplier defects cannot hide a failure of the authority path.
    let baseline_zero_raw = domain
        .evaluate_dual(&baseline_direction, &zero)
        .map_err(ReluTailDualError::Baseline)?;
    let baseline_zero = combine_exact_lower(baseline_zero_raw.lower, &line_plan.exact_constant)?;
    let mut best = AcceptedCandidate {
        lower: baseline_zero,
        zero_lower: baseline_zero,
        direction: baseline_direction,
        multipliers: clone_f64(&zero, "accepted zero multipliers")?,
        supplied_used: false,
    };
    let mut candidate_replays = ReluTailDualZeroPredicateCandidateReplays {
        zero_positive_slope_lower_bound: baseline_zero,
        upper_endpoint_lower_bound: None,
        canonical_lower_bound: None,
        optimized_lower_bound: None,
    };
    if maybe_replay_supplied(domain, &line_plan.exact_constant, supplied, &mut best)
        == SuppliedReplayOutcome::AllocationFallback
    {
        return Ok(finish_result(
            best,
            line_plan,
            candidate_replays,
            1,
            0,
            ReluTailDualStatus::AllocationFallback,
            None,
        ));
    }
    let mut candidates_replayed = 1_usize;

    if line_plan.variables.is_empty() {
        return Ok(finish_result(
            best,
            line_plan,
            candidate_replays,
            candidates_replayed,
            0,
            ReluTailDualStatus::NoOptimizableSlopes,
            None,
        ));
    }
    if config.iterations == 0 {
        return Ok(finish_result(
            best,
            line_plan,
            candidate_replays,
            candidates_replayed,
            0,
            ReluTailDualStatus::SearchDisabled,
            None,
        ));
    }

    let Some(plan) = ReluTailDualPlan::checked(
        domain,
        generator_nonzeros,
        line_plan.variables.len(),
        config,
    ) else {
        return Ok(finish_result(
            best,
            line_plan,
            candidate_replays,
            candidates_replayed,
            0,
            ReluTailDualStatus::ResourceFallback,
            None,
        ));
    };

    let Some(mut endpoint) = candidate_clone_f64(&line_plan.fixed_direction) else {
        return Ok(finish_result(
            best,
            line_plan,
            candidate_replays,
            candidates_replayed,
            0,
            ReluTailDualStatus::AllocationFallback,
            Some(plan),
        ));
    };
    for variable in &line_plan.variables {
        endpoint[variable.coordinate] = variable.upper;
    }
    match replay_direction(
        domain,
        &line_plan.exact_constant,
        &line_plan.variables,
        endpoint,
        &zero,
        supplied,
        &mut best,
    ) {
        CandidateReplayOutcome::Replayed {
            zero_predicate_lower_bound,
        } => {
            candidate_replays.upper_endpoint_lower_bound = Some(zero_predicate_lower_bound);
            candidates_replayed += 1;
        }
        CandidateReplayOutcome::Rejected => {}
        CandidateReplayOutcome::AllocationFallback => {
            return Ok(finish_result(
                best,
                line_plan,
                candidate_replays,
                candidates_replayed,
                0,
                ReluTailDualStatus::AllocationFallback,
                Some(plan),
            ));
        }
    }

    let Some(mut canonical) = candidate_clone_f64(&line_plan.fixed_direction) else {
        return Ok(finish_result(
            best,
            line_plan,
            candidate_replays,
            candidates_replayed,
            0,
            ReluTailDualStatus::AllocationFallback,
            Some(plan),
        ));
    };
    for variable in &line_plan.variables {
        canonical[variable.coordinate] = variable.canonical;
    }
    let Some(canonical_replay) = candidate_clone_f64(&canonical) else {
        return Ok(finish_result(
            best,
            line_plan,
            candidate_replays,
            candidates_replayed,
            0,
            ReluTailDualStatus::AllocationFallback,
            Some(plan),
        ));
    };
    match replay_direction(
        domain,
        &line_plan.exact_constant,
        &line_plan.variables,
        canonical_replay,
        &zero,
        supplied,
        &mut best,
    ) {
        CandidateReplayOutcome::Replayed {
            zero_predicate_lower_bound,
        } => {
            candidate_replays.canonical_lower_bound = Some(zero_predicate_lower_bound);
            candidates_replayed += 1;
        }
        CandidateReplayOutcome::Rejected => {}
        CandidateReplayOutcome::AllocationFallback => {
            return Ok(finish_result(
                best,
                line_plan,
                candidate_replays,
                candidates_replayed,
                0,
                ReluTailDualStatus::AllocationFallback,
                Some(plan),
            ));
        }
    }

    let search = projected_adam_candidate(domain, &line_plan.variables, canonical, config);
    let (status, iterations_completed) = match search {
        Ok((candidate, iterations_completed)) => {
            let replay = replay_direction(
                domain,
                &line_plan.exact_constant,
                &line_plan.variables,
                candidate,
                &zero,
                supplied,
                &mut best,
            );
            match replay {
                CandidateReplayOutcome::Replayed {
                    zero_predicate_lower_bound,
                } => {
                    candidate_replays.optimized_lower_bound = Some(zero_predicate_lower_bound);
                    candidates_replayed += 1;
                    (ReluTailDualStatus::Completed, iterations_completed)
                }
                CandidateReplayOutcome::Rejected => {
                    (ReluTailDualStatus::NonFiniteCandidate, iterations_completed)
                }
                CandidateReplayOutcome::AllocationFallback => {
                    (ReluTailDualStatus::AllocationFallback, iterations_completed)
                }
            }
        }
        Err(CandidateFailure::Deadline(iterations_completed)) => {
            (ReluTailDualStatus::Deadline, iterations_completed)
        }
        Err(CandidateFailure::NonFinite(iterations_completed)) => {
            (ReluTailDualStatus::NonFiniteCandidate, iterations_completed)
        }
        Err(CandidateFailure::Allocation(iterations_completed)) => {
            (ReluTailDualStatus::AllocationFallback, iterations_completed)
        }
    };

    Ok(finish_result(
        best,
        line_plan,
        candidate_replays,
        candidates_replayed,
        iterations_completed,
        status,
        Some(plan),
    ))
}

/// Bound one exact margin using caller-certified auxiliary preactivation bounds.
///
/// For each coordinate, this intersects the exact unconstrained coordinate
/// hull recomputed from `domain` with the exact binary64 dyadics in
/// `auxiliary`.  The intersection is used only to classify the ReLU phase and
/// construct its exact affine minorant.  Directional certificate replay still
/// minimizes over the original constrained zonotope.
///
/// The semantic proof obligation is explicit: the caller must have established
/// that **every concrete preactivation witness** is inside `auxiliary`.  The
/// resulting line therefore need only be valid on the concrete set
/// `S ⊆ Z ∩ auxiliary`; it need not be valid at spurious points of `Z`
/// outside the auxiliary box.  Exact chord arithmetic and dyadic correction
/// are unchanged, and exact CPU outward replay over `Z` remains the authority
/// path.
///
/// Tightening is sound but not monotone in certificate quality: changing a
/// line that is valid on `S` can produce a weaker minimum when replayed over
/// the larger `Z`.  Callers must portfolio this result with
/// [`bound_relu_tail_triangle_dual_unwired`] and retain the better independently
/// certified lower bound.
///
/// This API does not add box-dual multipliers and remains unwired from CLI
/// commands, verifier verdicts, presets, and scored paths.
///
/// # Errors
///
/// In addition to [`bound_relu_tail_triangle_dual_unwired`] failures, returns
/// [`ReluTailDualError::AuxiliaryDimensionMismatch`] when the auxiliary value
/// dimension differs and [`ReluTailDualError::EmptyAuxiliaryIntersection`]
/// when an auxiliary interval is disjoint from the exact CZ coordinate hull.
pub fn bound_relu_tail_triangle_dual_with_auxiliary_bounds_unwired(
    domain: &ConstrainedZonotope64,
    auxiliary: &CertifiedAuxiliaryBounds64,
    margin: &ExactReluTailMargin,
    supplied_multipliers: Option<&[f64]>,
    config: ReluTailDualConfig,
) -> Result<ReluTailDualResult, ReluTailDualError> {
    if auxiliary.value_dim() != domain.value_dim() {
        return Err(ReluTailDualError::AuxiliaryDimensionMismatch {
            expected: domain.value_dim(),
            got: auxiliary.value_dim(),
        });
    }
    let generator_nonzeros = check_mandatory_resources(domain, margin)?;
    let line_plan = build_line_plan_with_auxiliary_bounds(domain, auxiliary, margin)?;
    bound_relu_tail_triangle_dual_from_line_plan(
        domain,
        line_plan,
        generator_nonzeros,
        supplied_multipliers,
        config,
    )
}

/// Portfolio M17 and M20 with one supplied auxiliary-Box cut candidate.
///
/// This first obtains the mandatory original M17 result.  It then obtains the
/// ordinary M20 result and treats its accepted line as `d dot x + K` on the
/// caller-certified concrete set `S subset Z intersection [lower, upper]`.
/// For nonnegative `mu_upper` and `mu_lower`, it independently replays
///
/// ```text
/// p = d + mu_upper - mu_lower
/// D_Z(p, lambda) + K - mu_upper dot upper + mu_lower dot lower.
/// ```
///
/// `p` is formed in exact rational arithmetic.  Its conversion to binary64 is
/// repaired coordinatewise over the exact original CZ hull, not over the
/// auxiliary Box.  All constants are combined exactly and only the final
/// result is rounded toward negative infinity.  Zero Box multipliers recover
/// the existing M17-shaped replay for the selected line; a nonrestrictive
/// auxiliary Box therefore recovers M17 bit-for-bit.
///
/// Box multipliers are optional candidate data.  Wrong shapes, negative or
/// non-finite values, exact-growth failure, allocation failure, or replay
/// failure retain the already-certified M17/M20 portfolio.  Selection changes
/// only for a strict lower-bound improvement, so this function cannot regress
/// below the original result.
///
/// The semantic proof obligation remains caller-owned: every concrete
/// preactivation witness must lie in `auxiliary`.  This function remains
/// unwired from commands, presets, verdicts, and scored paths.
///
/// # Errors
///
/// Returns [`ReluTailDualError`] only when the mandatory original M17 setup or
/// replay fails.  Every auxiliary or Box-cut failure is represented by
/// [`ReluTailBoxCutStatus`] and preserves that original certificate.
#[allow(clippy::too_many_arguments)]
pub fn bound_relu_tail_triangle_dual_with_auxiliary_box_cut_unwired(
    domain: &ConstrainedZonotope64,
    auxiliary: &CertifiedAuxiliaryBounds64,
    margin: &ExactReluTailMargin,
    upper_box_multipliers: &[f64],
    lower_box_multipliers: &[f64],
    supplied_predicate_multipliers: Option<&[f64]>,
    config: ReluTailDualConfig,
) -> Result<ReluTailBoxCutDualResult, ReluTailDualError> {
    // This authority path is deliberately first and cannot be suppressed by
    // any auxiliary fact or malformed optional multiplier.
    let original = bound_relu_tail_triangle_dual_unwired(
        domain,
        margin,
        supplied_predicate_multipliers,
        config,
    )?;

    let auxiliary_result = match bound_relu_tail_triangle_dual_with_auxiliary_bounds_unwired(
        domain,
        auxiliary,
        margin,
        supplied_predicate_multipliers,
        config,
    ) {
        Ok(result) => result,
        Err(_) => {
            return Ok(finish_box_cut_portfolio(
                original,
                None,
                None,
                ReluTailBoxCutStatus::AuxiliaryFallback,
            ));
        }
    };

    if upper_box_multipliers.len() != domain.value_dim()
        || lower_box_multipliers.len() != domain.value_dim()
    {
        return Ok(finish_box_cut_portfolio(
            original,
            Some(auxiliary_result),
            None,
            ReluTailBoxCutStatus::InvalidBoxMultiplierShape {
                expected: domain.value_dim(),
                upper_got: upper_box_multipliers.len(),
                lower_got: lower_box_multipliers.len(),
            },
        ));
    }
    if let Some(coordinate) = upper_box_multipliers
        .iter()
        .position(|value| !value.is_finite() || *value < 0.0)
    {
        return Ok(finish_box_cut_portfolio(
            original,
            Some(auxiliary_result),
            None,
            ReluTailBoxCutStatus::InvalidBoxMultiplierValue {
                upper: true,
                coordinate,
            },
        ));
    }
    if let Some(coordinate) = lower_box_multipliers
        .iter()
        .position(|value| !value.is_finite() || *value < 0.0)
    {
        return Ok(finish_box_cut_portfolio(
            original,
            Some(auxiliary_result),
            None,
            ReluTailBoxCutStatus::InvalidBoxMultiplierValue {
                upper: false,
                coordinate,
            },
        ));
    }

    let box_cut = match build_auxiliary_box_cut_certificate(
        domain,
        auxiliary,
        &auxiliary_result,
        upper_box_multipliers,
        lower_box_multipliers,
        supplied_predicate_multipliers,
    ) {
        Ok(certificate) => certificate,
        Err(_) => {
            return Ok(finish_box_cut_portfolio(
                original,
                Some(auxiliary_result),
                None,
                ReluTailBoxCutStatus::CandidateFallback,
            ));
        }
    };

    Ok(finish_box_cut_portfolio(
        original,
        Some(auxiliary_result),
        Some(box_cut),
        ReluTailBoxCutStatus::Completed,
    ))
}

fn finish_box_cut_portfolio(
    original: ReluTailDualResult,
    auxiliary: Option<ReluTailDualResult>,
    box_cut: Option<ReluTailBoxCutCertificate>,
    status: ReluTailBoxCutStatus,
) -> ReluTailBoxCutDualResult {
    let mut lower_bound = original.lower_bound;
    let mut selected = ReluTailBoxCutSelection::Original;
    if let Some(candidate) = auxiliary.as_ref() {
        if candidate.lower_bound > lower_bound {
            lower_bound = candidate.lower_bound;
            selected = ReluTailBoxCutSelection::Auxiliary;
        }
    }
    if let Some(candidate) = box_cut.as_ref() {
        if candidate.lower_bound > lower_bound {
            lower_bound = candidate.lower_bound;
            selected = ReluTailBoxCutSelection::BoxCut;
        }
    }
    ReluTailBoxCutDualResult {
        lower_bound,
        selected,
        original,
        auxiliary,
        box_cut,
        status,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BoxSearchTelemetry {
    iterations_completed: usize,
    restarts_completed: usize,
    candidates_scored: usize,
    exact_replays: usize,
}

fn finish_optimized_box_cut_result(
    original: ReluTailDualResult,
    auxiliary: Option<ReluTailDualResult>,
    box_cut: Option<ReluTailBoxCutCertificate>,
    search_status: ReluTailBoxCutOptimizerStatus,
    search_plan: Option<ReluTailBoxCutOptimizerPlan>,
    telemetry: BoxSearchTelemetry,
) -> ReluTailBoxCutOptimizedResult {
    let portfolio_status = if auxiliary.is_none() {
        ReluTailBoxCutStatus::AuxiliaryFallback
    } else if box_cut.is_some() {
        ReluTailBoxCutStatus::Completed
    } else {
        ReluTailBoxCutStatus::CandidateFallback
    };
    let portfolio = finish_box_cut_portfolio(original, auxiliary, box_cut, portfolio_status);
    ReluTailBoxCutOptimizedResult {
        lower_bound: portfolio.lower_bound,
        selected: portfolio.selected,
        portfolio,
        search_status,
        search_plan,
        iterations_completed: telemetry.iterations_completed,
        restarts_completed: telemetry.restarts_completed,
        candidates_scored: telemetry.candidates_scored,
        exact_replays: telemetry.exact_replays,
    }
}

impl ReluTailDualPlan {
    fn checked(
        domain: &ConstrainedZonotope64,
        generator_nonzeros: usize,
        optimizable_slopes: usize,
        config: ReluTailDualConfig,
    ) -> Option<Self> {
        if !valid_search_config(config) {
            return None;
        }
        let limits = config.limits;
        if domain.value_dim() > limits.max_value_dim
            || domain.alpha_dim() > limits.max_alpha_dim
            || domain.constraint_count() > limits.max_constraints
            || generator_nonzeros > limits.max_generator_nonzeros
            || optimizable_slopes > limits.max_optimizable_slopes
            || config.iterations > limits.max_iterations
        {
            return None;
        }
        // Let S be optimized slopes, V value axes, G sparse generator
        // entries, and A generator columns (including empty columns).
        // Startup initializes three S buffers and one V map, populates the S
        // map entries, clones a V direction, then scores V + G + A visits.
        // Each update clears and initializes S gradients, makes two sparse
        // subgradient passes, updates S slopes, scores V + G + A visits, and
        // conservatively prices a V best-direction copy on every iteration.
        let slopes = optimizable_slopes as u128;
        let values = domain.value_dim() as u128;
        let sparse = generator_nonzeros as u128;
        let alpha = domain.alpha_dim() as u128;
        let startup = slopes
            .checked_mul(4)?
            .checked_add(values.checked_mul(3)?)?
            .checked_add(sparse)?
            .checked_add(alpha)?;
        let per_iteration = slopes
            .checked_mul(3)?
            .checked_add(values.checked_mul(2)?)?
            .checked_add(sparse.checked_mul(3)?)?
            .checked_add(alpha.checked_mul(2)?)?;
        let work = per_iteration
            .checked_mul(config.iterations as u128)?
            .checked_add(startup)?;
        let search_work = u64::try_from(work).ok()?;
        if search_work > limits.max_search_work {
            return None;
        }
        Some(Self {
            value_dim: domain.value_dim(),
            alpha_dim: domain.alpha_dim(),
            constraints: domain.constraint_count(),
            generator_nonzeros,
            optimizable_slopes,
            iterations: config.iterations,
            search_work,
        })
    }
}

impl ReluTailBoxCutOptimizerPlan {
    fn checked(
        domain: &ConstrainedZonotope64,
        generator_nonzeros: usize,
        box_variables: usize,
        config: ReluTailBoxCutOptimizerConfig,
    ) -> Result<Self, ReluTailBoxCutOptimizerStatus> {
        if !valid_box_cut_optimizer_config(config) {
            return Err(ReluTailBoxCutOptimizerStatus::InvalidConfig);
        }
        let total_iterations = config.schedules.iter().try_fold(0_usize, |sum, schedule| {
            sum.checked_add(schedule.iterations)
        });
        let Some(total_iterations) = total_iterations else {
            return Err(ReluTailBoxCutOptimizerStatus::InvalidConfig);
        };
        let restarts = config
            .schedules
            .iter()
            .filter(|schedule| schedule.iterations > 0)
            .count();
        if total_iterations > RELU_TAIL_BOX_CUT_HARD_MAX_ITERATIONS
            || restarts > RELU_TAIL_BOX_CUT_HARD_MAX_RESTARTS
        {
            return Err(ReluTailBoxCutOptimizerStatus::InvalidConfig);
        }
        let limits = config.limits;
        let exact_replays = restarts;
        if domain.value_dim() > limits.max_value_dim
            || box_variables > limits.max_box_variables
            || total_iterations > limits.max_total_iterations
            || restarts > limits.max_restarts
            || exact_replays > limits.max_exact_replays
            || generator_nonzeros > limits.max_generator_nonzeros
            || config.wall_time > limits.max_wall_time
        {
            return Err(ReluTailBoxCutOptimizerStatus::ResourceFallback);
        }

        // Let V be value axes, B selected endpoint multipliers, A generator
        // columns, and G sparse generator entries.  Each restart allocates
        // five B buffers plus p/h, scores its zero start, and every update
        // performs one Adam pass, one full objective/supergradient score, and
        // a conservatively charged B-sized best-candidate copy.  The count is
        // intentionally pessimistic and uses checked u128 arithmetic.
        let values = domain.value_dim() as u128;
        let variables = box_variables as u128;
        let sparse = generator_nonzeros as u128;
        let alpha = domain.alpha_dim() as u128;
        let score = values
            .checked_mul(3)
            .and_then(|work| work.checked_add(variables.checked_mul(2)?))
            .and_then(|work| work.checked_add(sparse.checked_mul(2)?))
            .and_then(|work| work.checked_add(alpha))
            .ok_or(ReluTailBoxCutOptimizerStatus::ResourceFallback)?;
        let restart_startup = variables
            .checked_mul(6)
            .and_then(|work| work.checked_add(values.checked_mul(2)?))
            .and_then(|work| work.checked_add(score))
            .ok_or(ReluTailBoxCutOptimizerStatus::ResourceFallback)?;
        let per_iteration = variables
            .checked_mul(6)
            .and_then(|work| work.checked_add(score))
            .ok_or(ReluTailBoxCutOptimizerStatus::ResourceFallback)?;
        let work = values
            .checked_add(
                restart_startup
                    .checked_mul(restarts as u128)
                    .ok_or(ReluTailBoxCutOptimizerStatus::ResourceFallback)?,
            )
            .and_then(|work| work.checked_add(per_iteration.checked_mul(total_iterations as u128)?))
            .ok_or(ReluTailBoxCutOptimizerStatus::ResourceFallback)?;
        let search_work =
            u64::try_from(work).map_err(|_| ReluTailBoxCutOptimizerStatus::ResourceFallback)?;
        if search_work > limits.max_search_work {
            return Err(ReluTailBoxCutOptimizerStatus::ResourceFallback);
        }

        Ok(Self {
            value_dim: domain.value_dim(),
            alpha_dim: domain.alpha_dim(),
            generator_nonzeros,
            box_variables,
            restarts,
            total_iterations,
            exact_replays,
            search_work,
        })
    }
}

fn valid_box_cut_optimizer_config(config: ReluTailBoxCutOptimizerConfig) -> bool {
    let limits = config.limits;
    config.multiplier_cap.is_finite()
        && config.multiplier_cap > 0.0
        && config.multiplier_cap <= RELU_TAIL_BOX_CUT_HARD_MAX_MULTIPLIER
        && !config.wall_time.is_zero()
        && config.wall_time <= RELU_TAIL_BOX_CUT_HARD_MAX_WALL_TIME
        && config.schedules.iter().all(|schedule| {
            schedule.learning_rate.is_finite()
                && schedule.learning_rate > 0.0
                && schedule.decay.is_finite()
                && schedule.decay > 0.0
                && schedule.decay <= 1.0
                && schedule.iterations <= RELU_TAIL_BOX_CUT_HARD_MAX_ITERATIONS
        })
        && limits.max_value_dim <= RELU_TAIL_DUAL_HARD_MAX_VALUE_DIM
        && limits.max_box_variables <= RELU_TAIL_BOX_CUT_HARD_MAX_VARIABLES
        && limits.max_total_iterations <= RELU_TAIL_BOX_CUT_HARD_MAX_ITERATIONS
        && limits.max_restarts <= RELU_TAIL_BOX_CUT_HARD_MAX_RESTARTS
        && limits.max_exact_replays <= RELU_TAIL_BOX_CUT_HARD_MAX_EXACT_REPLAYS
        && limits.max_generator_nonzeros <= RELU_TAIL_DUAL_HARD_MAX_GENERATOR_NONZEROS
        && limits.max_search_work <= RELU_TAIL_BOX_CUT_HARD_MAX_SEARCH_WORK
        && !limits.max_wall_time.is_zero()
        && limits.max_wall_time <= RELU_TAIL_BOX_CUT_HARD_MAX_WALL_TIME
}

fn valid_search_config(config: ReluTailDualConfig) -> bool {
    let limits = config.limits;
    config.iterations <= RELU_TAIL_DUAL_HARD_MAX_ITERATIONS
        && config.learning_rate.is_finite()
        && config.learning_rate > 0.0
        && config.beta1.is_finite()
        && (0.0..1.0).contains(&config.beta1)
        && config.beta2.is_finite()
        && (0.0..1.0).contains(&config.beta2)
        && config.epsilon.is_finite()
        && config.epsilon > 0.0
        && !config.wall_time.is_zero()
        && config.wall_time <= limits.max_wall_time
        && limits.max_value_dim > 0
        && limits.max_value_dim <= RELU_TAIL_DUAL_HARD_MAX_VALUE_DIM
        && limits.max_alpha_dim <= RELU_TAIL_DUAL_HARD_MAX_ALPHA_DIM
        && limits.max_constraints <= RELU_TAIL_DUAL_HARD_MAX_CONSTRAINTS
        && limits.max_generator_nonzeros <= RELU_TAIL_DUAL_HARD_MAX_GENERATOR_NONZEROS
        && limits.max_optimizable_slopes > 0
        && limits.max_optimizable_slopes <= RELU_TAIL_DUAL_HARD_MAX_OPTIMIZABLE_SLOPES
        && limits.max_iterations > 0
        && limits.max_iterations <= RELU_TAIL_DUAL_HARD_MAX_ITERATIONS
        && limits.max_search_work > 0
        && limits.max_search_work <= RELU_TAIL_DUAL_HARD_MAX_SEARCH_WORK
        && !limits.max_wall_time.is_zero()
        && limits.max_wall_time <= RELU_TAIL_DUAL_HARD_MAX_WALL_TIME
}

#[derive(Clone, Debug)]
struct LinePlan {
    fixed_direction: Vec<f64>,
    exact_constant: BigRational,
    variables: Vec<SlopeVariable>,
}

#[derive(Clone, Debug)]
struct SlopeVariable {
    coordinate: usize,
    exact_upper: BigRational,
    upper: f64,
    canonical: f64,
}

fn build_line_plan(
    domain: &ConstrainedZonotope64,
    margin: &ExactReluTailMargin,
) -> Result<LinePlan, ReluTailDualError> {
    let bounds = exact_coordinate_bounds(domain)?;
    build_line_plan_from_bounds(domain, margin, &bounds)
}

fn build_line_plan_with_auxiliary_bounds(
    domain: &ConstrainedZonotope64,
    auxiliary: &CertifiedAuxiliaryBounds64,
    margin: &ExactReluTailMargin,
) -> Result<LinePlan, ReluTailDualError> {
    let bounds = exact_coordinate_bounds_with_auxiliary(domain, auxiliary)?;
    build_line_plan_from_bounds(domain, margin, &bounds)
}

fn build_line_plan_from_bounds(
    domain: &ConstrainedZonotope64,
    margin: &ExactReluTailMargin,
    bounds: &[(BigRational, BigRational)],
) -> Result<LinePlan, ReluTailDualError> {
    debug_assert_eq!(bounds.len(), domain.value_dim());
    let mut fixed_direction = zero_f64(domain.value_dim(), "fixed direction")?;
    let mut variables = Vec::new();
    try_reserve(
        &mut variables,
        domain.value_dim(),
        "positive unstable slopes",
    )?;
    let mut exact_constant = checked_rational(margin.bias.clone(), "declared margin bias", 0)?;

    for (coordinate, ((lower, upper), coefficient)) in
        bounds.iter().zip(margin.coefficients.iter()).enumerate()
    {
        let zero = BigRational::zero();
        if upper <= &zero || coefficient.is_zero() {
            continue;
        }
        if lower >= &zero {
            let direction = nearest_finite(coefficient, coordinate, "active exact coefficient")?;
            let correction = exact_line_correction(
                coefficient,
                &BigRational::zero(),
                direction,
                lower,
                upper,
                coordinate,
            )?;
            fixed_direction[coordinate] = canonical_zero(direction);
            exact_constant = checked_rational(
                exact_constant + correction,
                "active correction accumulation",
                coordinate,
            )?;
            continue;
        }

        debug_assert!(lower.is_negative() && upper.is_positive());
        if coefficient.is_negative() {
            let denominator =
                checked_rational(upper - lower, "unstable chord denominator", coordinate)?;
            let scale = checked_rational(upper / denominator, "unstable chord scale", coordinate)?;
            let exact_a = checked_rational(
                coefficient * &scale,
                "negative coefficient times chord scale",
                coordinate,
            )?;
            let exact_b = checked_rational(
                -checked_rational(
                    &exact_a * lower,
                    "negative chord slope times lower endpoint",
                    coordinate,
                )?,
                "negative chord intercept",
                coordinate,
            )?;
            let direction = nearest_finite(&exact_a, coordinate, "negative chord coefficient")?;
            let correction =
                exact_line_correction(&exact_a, &exact_b, direction, lower, upper, coordinate)?;
            fixed_direction[coordinate] = canonical_zero(direction);
            exact_constant = checked_rational(
                exact_constant + correction,
                "negative chord correction accumulation",
                coordinate,
            )?;
            continue;
        }

        let exact_upper = coefficient.clone();
        let upper_f64 = floor_finite(&exact_upper, coordinate, "positive slope upper endpoint")?;
        let denominator =
            checked_rational(upper - lower, "positive canonical denominator", coordinate)?;
        let canonical_exact = checked_rational(
            checked_rational(
                coefficient * upper,
                "positive coefficient times upper endpoint",
                coordinate,
            )? / denominator,
            "positive canonical slope",
            coordinate,
        )?;
        let canonical = floor_finite(
            &canonical_exact,
            coordinate,
            "positive canonical slope conversion",
        )?;
        if !valid_direct_slope(upper_f64, &exact_upper)
            || !valid_direct_slope(canonical, &exact_upper)
        {
            return Err(ReluTailDualError::NonFiniteArithmetic {
                coordinate,
                operation: "exact positive slope interval validation",
            });
        }
        variables.push(SlopeVariable {
            coordinate,
            exact_upper,
            upper: canonical_zero(upper_f64),
            canonical: canonical_zero(canonical),
        });
    }

    Ok(LinePlan {
        fixed_direction,
        exact_constant,
        variables,
    })
}

#[cfg(test)]
std::thread_local! {
    static EXACT_COORDINATE_HULL_PASSES: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
    static BOX_OPTIMIZER_FAIL_NEXT_ALLOCATION: std::cell::Cell<bool> = const {
        std::cell::Cell::new(false)
    };
    static BOX_OPTIMIZER_FAIL_NEXT_EXACT_REPLAY: std::cell::Cell<bool> = const {
        std::cell::Cell::new(false)
    };
}

fn exact_coordinate_bounds(
    domain: &ConstrainedZonotope64,
) -> Result<Vec<(BigRational, BigRational)>, ReluTailDualError> {
    #[cfg(test)]
    EXACT_COORDINATE_HULL_PASSES.with(|passes| passes.set(passes.get() + 1));

    let mut radii = Vec::new();
    try_reserve(&mut radii, domain.value_dim(), "exact coordinate radii")?;
    for (coordinate, &remainder) in domain.box_remainder().iter().enumerate() {
        radii.push(checked_rational(
            exact_domain_f64(remainder, coordinate, "box remainder")?,
            "box remainder",
            coordinate,
        )?);
    }
    for generator in domain.generators() {
        for (coordinate, coefficient) in generator.entries() {
            let coefficient = exact_domain_f64(coefficient, coordinate, "generator coefficient")?;
            radii[coordinate] = checked_rational(
                radii[coordinate].clone() + coefficient.abs(),
                "coordinate radius accumulation",
                coordinate,
            )?;
        }
    }

    let mut bounds = Vec::new();
    try_reserve(&mut bounds, domain.value_dim(), "exact coordinate bounds")?;
    for (coordinate, (&center, radius)) in domain.center().iter().zip(radii).enumerate() {
        let center = exact_domain_f64(center, coordinate, "center")?;
        let lower = checked_rational(&center - &radius, "coordinate lower bound", coordinate)?;
        let upper = checked_rational(center + radius, "coordinate upper bound", coordinate)?;
        bounds.push((lower, upper));
    }
    Ok(bounds)
}

fn exact_coordinate_bounds_with_auxiliary(
    domain: &ConstrainedZonotope64,
    auxiliary: &CertifiedAuxiliaryBounds64,
) -> Result<Vec<(BigRational, BigRational)>, ReluTailDualError> {
    debug_assert_eq!(domain.value_dim(), auxiliary.value_dim());
    let mut bounds = exact_coordinate_bounds(domain)?;
    for (coordinate, ((lower, upper), (&auxiliary_lower, &auxiliary_upper))) in bounds
        .iter_mut()
        .zip(auxiliary.lower().iter().zip(auxiliary.upper()))
        .enumerate()
    {
        let auxiliary_lower = checked_rational(
            exact_domain_f64(auxiliary_lower, coordinate, "auxiliary lower bound")?,
            "auxiliary lower bound",
            coordinate,
        )?;
        let auxiliary_upper = checked_rational(
            exact_domain_f64(auxiliary_upper, coordinate, "auxiliary upper bound")?,
            "auxiliary upper bound",
            coordinate,
        )?;
        if auxiliary_lower > *lower {
            *lower = auxiliary_lower;
        }
        if auxiliary_upper < *upper {
            *upper = auxiliary_upper;
        }
        if *lower > *upper {
            return Err(ReluTailDualError::EmptyAuxiliaryIntersection { coordinate });
        }
    }
    Ok(bounds)
}

fn exact_coordinate_bounds_with_auxiliary_from_bounds(
    domain_bounds: &[(BigRational, BigRational)],
    auxiliary: &CertifiedAuxiliaryBounds64,
) -> Result<Vec<(BigRational, BigRational)>, ReluTailDualError> {
    debug_assert_eq!(domain_bounds.len(), auxiliary.value_dim());
    let mut bounds = Vec::new();
    try_reserve(
        &mut bounds,
        domain_bounds.len(),
        "auxiliary exact coordinate bounds",
    )?;
    bounds.extend(domain_bounds.iter().cloned());
    for (coordinate, ((lower, upper), (&auxiliary_lower, &auxiliary_upper))) in bounds
        .iter_mut()
        .zip(auxiliary.lower().iter().zip(auxiliary.upper()))
        .enumerate()
    {
        let auxiliary_lower = checked_rational(
            exact_domain_f64(auxiliary_lower, coordinate, "auxiliary lower bound")?,
            "auxiliary lower bound",
            coordinate,
        )?;
        let auxiliary_upper = checked_rational(
            exact_domain_f64(auxiliary_upper, coordinate, "auxiliary upper bound")?,
            "auxiliary upper bound",
            coordinate,
        )?;
        if auxiliary_lower > *lower {
            *lower = auxiliary_lower;
        }
        if auxiliary_upper < *upper {
            *upper = auxiliary_upper;
        }
        if *lower > *upper {
            return Err(ReluTailDualError::EmptyAuxiliaryIntersection { coordinate });
        }
    }
    Ok(bounds)
}

fn count_tighter_auxiliary_box_endpoints(
    original_hull: &[(BigRational, BigRational)],
    auxiliary: &CertifiedAuxiliaryBounds64,
) -> Result<usize, ReluTailDualError> {
    debug_assert_eq!(original_hull.len(), auxiliary.value_dim());
    let mut count = 0_usize;
    for (coordinate, ((hull_lower, hull_upper), (&lower, &upper))) in original_hull
        .iter()
        .zip(auxiliary.lower().iter().zip(auxiliary.upper()))
        .enumerate()
    {
        let lower = exact_domain_f64(lower, coordinate, "Box optimizer lower endpoint")?;
        let upper = exact_domain_f64(upper, coordinate, "Box optimizer upper endpoint")?;
        if lower > *hull_lower {
            count = count
                .checked_add(1)
                .ok_or(ReluTailDualError::ResourceOverflow {
                    resource: "Box optimizer variables",
                })?;
        }
        if upper < *hull_upper {
            count = count
                .checked_add(1)
                .ok_or(ReluTailDualError::ResourceOverflow {
                    resource: "Box optimizer variables",
                })?;
        }
    }
    Ok(count)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoxVariableKind {
    Upper,
    Lower,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct BoxSearchVariable {
    coordinate: usize,
    kind: BoxVariableKind,
    endpoint: f64,
}

#[derive(Debug)]
struct BoxApproximateSearch {
    variables: Vec<BoxSearchVariable>,
    candidates: Vec<Vec<f64>>,
    status: ReluTailBoxCutOptimizerStatus,
    iterations_completed: usize,
    restarts_completed: usize,
    candidates_scored: usize,
}

fn optimize_auxiliary_box_multipliers(
    domain: &ConstrainedZonotope64,
    original_hull: &[(BigRational, BigRational)],
    auxiliary: &CertifiedAuxiliaryBounds64,
    line_direction: &[f64],
    config: ReluTailBoxCutOptimizerConfig,
    plan: ReluTailBoxCutOptimizerPlan,
) -> BoxApproximateSearch {
    let start = Instant::now();
    optimize_auxiliary_box_multipliers_with_clock(
        domain,
        original_hull,
        auxiliary,
        line_direction,
        config,
        plan,
        || start.elapsed(),
    )
}

fn optimize_auxiliary_box_multipliers_with_clock<C>(
    domain: &ConstrainedZonotope64,
    original_hull: &[(BigRational, BigRational)],
    auxiliary: &CertifiedAuxiliaryBounds64,
    line_direction: &[f64],
    config: ReluTailBoxCutOptimizerConfig,
    plan: ReluTailBoxCutOptimizerPlan,
    elapsed: C,
) -> BoxApproximateSearch
where
    C: FnMut() -> Duration,
{
    let mut deadline = CandidateDeadline::new(config.wall_time, elapsed);
    let mut result = BoxApproximateSearch {
        variables: Vec::new(),
        candidates: Vec::new(),
        status: ReluTailBoxCutOptimizerStatus::Completed,
        iterations_completed: 0,
        restarts_completed: 0,
        candidates_scored: 0,
    };
    // The candidate clock precedes every fallible candidate allocation and
    // every approximate score.  Mandatory M17/M20 work has already finished.
    if let Err(failure) = deadline.checkpoint(0) {
        result.status = box_status_from_candidate_failure(failure);
        return result;
    }
    result.variables = match collect_box_search_variables(
        original_hull,
        auxiliary,
        plan.box_variables,
        &mut deadline,
    ) {
        Ok(variables) => variables,
        Err(failure) => {
            result.status = box_status_from_candidate_failure(failure);
            return result;
        }
    };
    if result.variables.len() != plan.box_variables {
        result.status = ReluTailBoxCutOptimizerStatus::NonFiniteCandidate;
        return result;
    }
    if result.candidates.try_reserve_exact(plan.restarts).is_err() {
        result.status = ReluTailBoxCutOptimizerStatus::AllocationFallback;
        return result;
    }
    if let Err(failure) = deadline.visit(0) {
        result.status = box_status_from_candidate_failure(failure);
        return result;
    }

    for schedule in config
        .schedules
        .iter()
        .copied()
        .filter(|schedule| schedule.iterations > 0)
    {
        let restart = run_box_optimizer_restart(
            domain,
            line_direction,
            &result.variables,
            schedule,
            config.multiplier_cap,
            result.iterations_completed,
            &mut deadline,
        );
        result.iterations_completed += restart.iterations_completed;
        result.candidates_scored += restart.candidates_scored;
        if let Some(candidate) = restart.best_candidate {
            result.candidates.push(candidate);
        }
        if let Some(failure) = restart.failure {
            result.status = box_status_from_candidate_failure(failure);
            return result;
        }
        result.restarts_completed += 1;
    }
    result
}

fn collect_box_search_variables<C>(
    original_hull: &[(BigRational, BigRational)],
    auxiliary: &CertifiedAuxiliaryBounds64,
    expected: usize,
    deadline: &mut CandidateDeadline<C>,
) -> Result<Vec<BoxSearchVariable>, CandidateFailure>
where
    C: FnMut() -> Duration,
{
    let mut variables = Vec::new();
    variables
        .try_reserve_exact(expected)
        .map_err(|_| CandidateFailure::Allocation(0))?;
    for (coordinate, ((hull_lower, hull_upper), (&lower, &upper))) in original_hull
        .iter()
        .zip(auxiliary.lower().iter().zip(auxiliary.upper()))
        .enumerate()
    {
        deadline.visit(0)?;
        let Some(exact_lower) = BigRational::from_float(lower) else {
            return Err(CandidateFailure::NonFinite(0));
        };
        let Some(exact_upper) = BigRational::from_float(upper) else {
            return Err(CandidateFailure::NonFinite(0));
        };
        if exact_lower > *hull_lower {
            variables.push(BoxSearchVariable {
                coordinate,
                kind: BoxVariableKind::Lower,
                endpoint: lower,
            });
        }
        if exact_upper < *hull_upper {
            variables.push(BoxSearchVariable {
                coordinate,
                kind: BoxVariableKind::Upper,
                endpoint: upper,
            });
        }
    }
    Ok(variables)
}

#[derive(Debug)]
struct BoxRestartOutcome {
    best_candidate: Option<Vec<f64>>,
    failure: Option<CandidateFailure>,
    iterations_completed: usize,
    candidates_scored: usize,
}

fn run_box_optimizer_restart<C>(
    domain: &ConstrainedZonotope64,
    line_direction: &[f64],
    variables: &[BoxSearchVariable],
    schedule: ReluTailBoxCutAdamSchedule,
    multiplier_cap: f64,
    completed_before: usize,
    deadline: &mut CandidateDeadline<C>,
) -> BoxRestartOutcome
where
    C: FnMut() -> Duration,
{
    let failed_without_candidate = |failure| BoxRestartOutcome {
        best_candidate: None,
        failure: Some(failure),
        iterations_completed: 0,
        candidates_scored: 0,
    };
    if let Err(failure) = deadline.checkpoint(completed_before) {
        return failed_without_candidate(failure);
    }
    let mut multipliers = match box_candidate_zeros(variables.len(), completed_before, deadline) {
        Ok(values) => values,
        Err(failure) => return failed_without_candidate(failure),
    };
    let mut first = match box_candidate_zeros(variables.len(), completed_before, deadline) {
        Ok(values) => values,
        Err(failure) => return failed_without_candidate(failure),
    };
    let mut second = match box_candidate_zeros(variables.len(), completed_before, deadline) {
        Ok(values) => values,
        Err(failure) => return failed_without_candidate(failure),
    };
    let mut best = match box_candidate_zeros(variables.len(), completed_before, deadline) {
        Ok(values) => values,
        Err(failure) => return failed_without_candidate(failure),
    };
    let mut best_scratch = match box_candidate_zeros(variables.len(), completed_before, deadline) {
        Ok(values) => values,
        Err(failure) => return failed_without_candidate(failure),
    };
    let mut direction = match box_candidate_zeros(domain.value_dim(), completed_before, deadline) {
        Ok(values) => values,
        Err(failure) => return failed_without_candidate(failure),
    };
    let mut witness = match box_candidate_zeros(domain.value_dim(), completed_before, deadline) {
        Ok(values) => values,
        Err(failure) => return failed_without_candidate(failure),
    };

    let mut best_objective = match approximate_box_objective_and_witness(
        domain,
        line_direction,
        variables,
        &multipliers,
        &mut direction,
        &mut witness,
        completed_before,
        deadline,
    ) {
        Ok(objective) => objective,
        Err(failure) => return failed_without_candidate(failure),
    };
    let mut iterations_completed = 0_usize;
    let mut candidates_scored = 1_usize;

    for iteration in 0..schedule.iterations {
        let global_completed = completed_before + iterations_completed;
        if let Err(failure) = deadline.checkpoint(global_completed) {
            return BoxRestartOutcome {
                best_candidate: Some(best),
                failure: Some(failure),
                iterations_completed,
                candidates_scored,
            };
        }
        let step = match i32::try_from(iteration + 1) {
            Ok(step) => step,
            Err(_) => {
                return BoxRestartOutcome {
                    best_candidate: Some(best),
                    failure: Some(CandidateFailure::NonFinite(global_completed)),
                    iterations_completed,
                    candidates_scored,
                };
            }
        };
        let first_correction = 1.0 - RELU_TAIL_BOX_CUT_ADAM_BETA1.powi(step);
        let second_correction = 1.0 - RELU_TAIL_BOX_CUT_ADAM_BETA2.powi(step);
        let learning_rate = schedule.learning_rate * schedule.decay.powi(step - 1);
        if !first_correction.is_finite()
            || !second_correction.is_finite()
            || first_correction <= 0.0
            || second_correction <= 0.0
            || !learning_rate.is_finite()
        {
            return BoxRestartOutcome {
                best_candidate: Some(best),
                failure: Some(CandidateFailure::NonFinite(global_completed)),
                iterations_completed,
                candidates_scored,
            };
        }
        for (slot, variable) in variables.iter().enumerate() {
            if let Err(failure) = deadline.visit(global_completed) {
                return BoxRestartOutcome {
                    best_candidate: Some(best),
                    failure: Some(failure),
                    iterations_completed,
                    candidates_scored,
                };
            }
            let gradient = match variable.kind {
                BoxVariableKind::Upper => witness[variable.coordinate] - variable.endpoint,
                BoxVariableKind::Lower => variable.endpoint - witness[variable.coordinate],
            };
            first[slot] = RELU_TAIL_BOX_CUT_ADAM_BETA1 * first[slot]
                + (1.0 - RELU_TAIL_BOX_CUT_ADAM_BETA1) * gradient;
            second[slot] = RELU_TAIL_BOX_CUT_ADAM_BETA2 * second[slot]
                + (1.0 - RELU_TAIL_BOX_CUT_ADAM_BETA2) * gradient * gradient;
            let first_hat = first[slot] / first_correction;
            let second_hat = second[slot] / second_correction;
            let update =
                learning_rate * first_hat / (second_hat.sqrt() + RELU_TAIL_BOX_CUT_ADAM_EPSILON);
            let raw_candidate = multipliers[slot] + update;
            if !gradient.is_finite()
                || !first[slot].is_finite()
                || !second[slot].is_finite()
                || !first_hat.is_finite()
                || !second_hat.is_finite()
                || !update.is_finite()
                || !raw_candidate.is_finite()
            {
                return BoxRestartOutcome {
                    best_candidate: Some(best),
                    failure: Some(CandidateFailure::NonFinite(global_completed)),
                    iterations_completed,
                    candidates_scored,
                };
            }
            let candidate = raw_candidate.clamp(0.0, multiplier_cap);
            multipliers[slot] = canonical_zero(candidate);
        }

        let completed = global_completed + 1;
        let objective = match approximate_box_objective_and_witness(
            domain,
            line_direction,
            variables,
            &multipliers,
            &mut direction,
            &mut witness,
            completed,
            deadline,
        ) {
            Ok(objective) => objective,
            Err(failure) => {
                return BoxRestartOutcome {
                    best_candidate: Some(best),
                    failure: Some(failure),
                    iterations_completed,
                    candidates_scored,
                };
            }
        };
        iterations_completed += 1;
        candidates_scored += 1;
        if objective > best_objective {
            if let Err(failure) = copy_box_candidate_atomically(
                &mut best,
                &mut best_scratch,
                &multipliers,
                completed,
                deadline,
            ) {
                return BoxRestartOutcome {
                    best_candidate: Some(best),
                    failure: Some(failure),
                    iterations_completed,
                    candidates_scored,
                };
            }
            best_objective = objective;
        }
    }

    BoxRestartOutcome {
        best_candidate: Some(best),
        failure: None,
        iterations_completed,
        candidates_scored,
    }
}

fn box_candidate_zeros<C>(
    count: usize,
    iterations: usize,
    deadline: &mut CandidateDeadline<C>,
) -> Result<Vec<f64>, CandidateFailure>
where
    C: FnMut() -> Duration,
{
    #[cfg(test)]
    if BOX_OPTIMIZER_FAIL_NEXT_ALLOCATION.with(|fail| fail.replace(false)) {
        return Err(CandidateFailure::Allocation(iterations));
    }
    candidate_zeros(count, iterations, deadline)
}

#[allow(clippy::too_many_arguments)]
fn approximate_box_objective_and_witness<C>(
    domain: &ConstrainedZonotope64,
    line_direction: &[f64],
    variables: &[BoxSearchVariable],
    multipliers: &[f64],
    direction: &mut [f64],
    witness: &mut [f64],
    iterations: usize,
    deadline: &mut CandidateDeadline<C>,
) -> Result<f64, CandidateFailure>
where
    C: FnMut() -> Duration,
{
    debug_assert_eq!(line_direction.len(), domain.value_dim());
    debug_assert_eq!(direction.len(), domain.value_dim());
    debug_assert_eq!(witness.len(), domain.value_dim());
    debug_assert_eq!(variables.len(), multipliers.len());
    for (coordinate, ((target, witness_value), &source)) in direction
        .iter_mut()
        .zip(witness.iter_mut())
        .zip(line_direction)
        .enumerate()
    {
        deadline.visit(iterations)?;
        if !source.is_finite() {
            return Err(CandidateFailure::NonFinite(iterations));
        }
        *target = source;
        *witness_value = domain.center()[coordinate];
    }
    let mut objective = 0.0_f64;
    for (variable, &multiplier) in variables.iter().zip(multipliers) {
        deadline.visit(iterations)?;
        if !multiplier.is_finite() || multiplier < 0.0 {
            return Err(CandidateFailure::NonFinite(iterations));
        }
        match variable.kind {
            BoxVariableKind::Upper => {
                direction[variable.coordinate] += multiplier;
                objective -= multiplier * variable.endpoint;
            }
            BoxVariableKind::Lower => {
                direction[variable.coordinate] -= multiplier;
                objective += multiplier * variable.endpoint;
            }
        }
        if !direction[variable.coordinate].is_finite() || !objective.is_finite() {
            return Err(CandidateFailure::NonFinite(iterations));
        }
    }
    for coordinate in 0..domain.value_dim() {
        deadline.visit(iterations)?;
        let value = direction[coordinate];
        let sign = if value > 0.0 {
            1.0
        } else if value < 0.0 {
            -1.0
        } else {
            0.0
        };
        objective += value * domain.center()[coordinate];
        objective -= value.abs() * domain.box_remainder()[coordinate];
        witness[coordinate] -= sign * domain.box_remainder()[coordinate];
        if !objective.is_finite() || !witness[coordinate].is_finite() {
            return Err(CandidateFailure::NonFinite(iterations));
        }
    }
    for generator in domain.generators() {
        deadline.visit(iterations)?;
        let mut projection = 0.0_f64;
        for (coordinate, coefficient) in generator.entries() {
            deadline.visit(iterations)?;
            projection += direction[coordinate] * coefficient;
            if !projection.is_finite() {
                return Err(CandidateFailure::NonFinite(iterations));
            }
        }
        objective -= projection.abs();
        if !objective.is_finite() {
            return Err(CandidateFailure::NonFinite(iterations));
        }
        let sign = if projection > 0.0 {
            1.0
        } else if projection < 0.0 {
            -1.0
        } else {
            0.0
        };
        if sign != 0.0 {
            for (coordinate, coefficient) in generator.entries() {
                deadline.visit(iterations)?;
                witness[coordinate] -= sign * coefficient;
                if !witness[coordinate].is_finite() {
                    return Err(CandidateFailure::NonFinite(iterations));
                }
            }
        }
    }
    Ok(objective)
}

fn copy_box_candidate_atomically<C>(
    best: &mut Vec<f64>,
    scratch: &mut Vec<f64>,
    source: &[f64],
    iterations: usize,
    deadline: &mut CandidateDeadline<C>,
) -> Result<(), CandidateFailure>
where
    C: FnMut() -> Duration,
{
    debug_assert_eq!(best.len(), source.len());
    debug_assert_eq!(scratch.len(), source.len());
    for (target, &source) in scratch.iter_mut().zip(source) {
        deadline.visit(iterations)?;
        *target = source;
    }
    std::mem::swap(best, scratch);
    Ok(())
}

fn box_status_from_candidate_failure(failure: CandidateFailure) -> ReluTailBoxCutOptimizerStatus {
    match failure {
        CandidateFailure::Deadline(_) => ReluTailBoxCutOptimizerStatus::Deadline,
        CandidateFailure::NonFinite(_) => ReluTailBoxCutOptimizerStatus::NonFiniteCandidate,
        CandidateFailure::Allocation(_) => ReluTailBoxCutOptimizerStatus::AllocationFallback,
    }
}

fn expand_box_candidate(
    variables: &[BoxSearchVariable],
    candidate: &[f64],
    value_dim: usize,
) -> Result<(Vec<f64>, Vec<f64>), ReluTailDualError> {
    if candidate.len() != variables.len() {
        return Err(ReluTailDualError::Shape {
            field: "optimized Box multipliers",
            expected: variables.len(),
            got: candidate.len(),
        });
    }
    let mut upper = zero_f64(value_dim, "optimized upper Box multipliers")?;
    let mut lower = zero_f64(value_dim, "optimized lower Box multipliers")?;
    for (slot, (variable, &value)) in variables.iter().zip(candidate).enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(ReluTailDualError::NonFiniteArithmetic {
                coordinate: slot,
                operation: "optimized Box multiplier expansion",
            });
        }
        match variable.kind {
            BoxVariableKind::Upper => upper[variable.coordinate] = canonical_zero(value),
            BoxVariableKind::Lower => lower[variable.coordinate] = canonical_zero(value),
        }
    }
    Ok((upper, lower))
}

fn build_auxiliary_box_cut_certificate(
    domain: &ConstrainedZonotope64,
    auxiliary: &CertifiedAuxiliaryBounds64,
    auxiliary_result: &ReluTailDualResult,
    upper_box_multipliers: &[f64],
    lower_box_multipliers: &[f64],
    supplied_predicate_multipliers: Option<&[f64]>,
) -> Result<ReluTailBoxCutCertificate, ReluTailDualError> {
    debug_assert_eq!(auxiliary.value_dim(), domain.value_dim());
    debug_assert_eq!(auxiliary_result.direction.len(), domain.value_dim());
    debug_assert_eq!(upper_box_multipliers.len(), domain.value_dim());
    debug_assert_eq!(lower_box_multipliers.len(), domain.value_dim());

    let original_hull = exact_coordinate_bounds(domain)?;
    build_auxiliary_box_cut_certificate_with_original_hull(
        domain,
        auxiliary,
        auxiliary_result,
        upper_box_multipliers,
        lower_box_multipliers,
        supplied_predicate_multipliers,
        &original_hull,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_auxiliary_box_cut_certificate_with_original_hull(
    domain: &ConstrainedZonotope64,
    auxiliary: &CertifiedAuxiliaryBounds64,
    auxiliary_result: &ReluTailDualResult,
    upper_box_multipliers: &[f64],
    lower_box_multipliers: &[f64],
    supplied_predicate_multipliers: Option<&[f64]>,
    original_hull: &[(BigRational, BigRational)],
) -> Result<ReluTailBoxCutCertificate, ReluTailDualError> {
    debug_assert_eq!(auxiliary.value_dim(), domain.value_dim());
    debug_assert_eq!(auxiliary_result.direction.len(), domain.value_dim());
    debug_assert_eq!(upper_box_multipliers.len(), domain.value_dim());
    debug_assert_eq!(lower_box_multipliers.len(), domain.value_dim());
    debug_assert_eq!(original_hull.len(), domain.value_dim());

    // This hull deliberately precedes and excludes the auxiliary
    // intersection.  The residual between p* and the replayed dyadic p must be
    // valid at every point considered by D_Z, including spurious Z points
    // outside the certified Box.  The direct M22 wrapper computes it here;
    // M24 passes the domain-tied prepared copy without changing the replay.
    let mut replay_direction = zero_f64(domain.value_dim(), "Box-cut replay direction")?;
    let mut exact_constant = checked_rational(
        auxiliary_result.exact_constant.clone(),
        "Box-cut line constant",
        0,
    )?;

    for coordinate in 0..domain.value_dim() {
        let line_direction = checked_rational(
            exact_domain_f64(
                auxiliary_result.direction[coordinate],
                coordinate,
                "Box-cut source direction",
            )?,
            "Box-cut source direction",
            coordinate,
        )?;
        let upper_multiplier = checked_rational(
            exact_domain_f64(
                upper_box_multipliers[coordinate],
                coordinate,
                "upper Box multiplier",
            )?,
            "upper Box multiplier",
            coordinate,
        )?;
        let lower_multiplier = checked_rational(
            exact_domain_f64(
                lower_box_multipliers[coordinate],
                coordinate,
                "lower Box multiplier",
            )?,
            "lower Box multiplier",
            coordinate,
        )?;
        let exact_cut_direction = checked_rational(
            checked_rational(
                line_direction + &upper_multiplier,
                "line direction plus upper Box multiplier",
                coordinate,
            )? - &lower_multiplier,
            "cut direction minus lower Box multiplier",
            coordinate,
        )?;

        // Preserve the exact source bit pattern for zero multipliers.  Besides
        // making the recovery oracle explicit, this avoids depending on the
        // tie-breaking behavior of a redundant rational-to-f64 conversion.
        let replayed = if upper_box_multipliers[coordinate] == 0.0
            && lower_box_multipliers[coordinate] == 0.0
        {
            auxiliary_result.direction[coordinate]
        } else {
            nearest_finite(
                &exact_cut_direction,
                coordinate,
                "Box-cut direction conversion",
            )?
        };
        replay_direction[coordinate] = canonical_zero(replayed);
        let replayed_exact = checked_rational(
            exact_domain_f64(replayed, coordinate, "Box-cut replay direction")?,
            "Box-cut replay direction",
            coordinate,
        )?;
        let residual = checked_rational(
            &exact_cut_direction - replayed_exact,
            "exact Box-cut direction minus replay direction",
            coordinate,
        )?;
        let (hull_lower, hull_upper) = &original_hull[coordinate];
        let residual_endpoint = if residual.is_negative() {
            hull_upper
        } else {
            hull_lower
        };
        let rounding_repair = checked_rational(
            residual * residual_endpoint,
            "Box-cut direction rounding repair",
            coordinate,
        )?;

        let auxiliary_upper = checked_rational(
            exact_domain_f64(
                auxiliary.upper()[coordinate],
                coordinate,
                "Box-cut auxiliary upper endpoint",
            )?,
            "Box-cut auxiliary upper endpoint",
            coordinate,
        )?;
        let auxiliary_lower = checked_rational(
            exact_domain_f64(
                auxiliary.lower()[coordinate],
                coordinate,
                "Box-cut auxiliary lower endpoint",
            )?,
            "Box-cut auxiliary lower endpoint",
            coordinate,
        )?;
        let upper_constant = checked_rational(
            upper_multiplier * auxiliary_upper,
            "upper Box multiplier times endpoint",
            coordinate,
        )?;
        let lower_constant = checked_rational(
            lower_multiplier * auxiliary_lower,
            "lower Box multiplier times endpoint",
            coordinate,
        )?;
        exact_constant = checked_rational(
            exact_constant - upper_constant,
            "subtract upper Box-cut constant",
            coordinate,
        )?;
        exact_constant = checked_rational(
            exact_constant + lower_constant,
            "add lower Box-cut constant",
            coordinate,
        )?;
        exact_constant = checked_rational(
            exact_constant + rounding_repair,
            "add Box-cut direction rounding repair",
            coordinate,
        )?;
    }

    let mut zero = zero_f64(
        domain.constraint_count(),
        "Box-cut zero predicate multipliers",
    )?;
    zero.fill(0.0);
    let replay = domain
        .evaluate_dual(&replay_direction, &zero)
        .map_err(ReluTailDualError::Baseline)?;
    let zero_predicate_lower_bound = combine_exact_lower(replay.lower, &exact_constant)?;
    let mut lower_bound = zero_predicate_lower_bound;
    let mut predicate_multipliers = zero;
    let mut supplied_predicate_multipliers_used = false;

    if let Some(supplied) =
        valid_supplied_multipliers(supplied_predicate_multipliers, domain.constraint_count())
    {
        if let Ok(replay) = domain.evaluate_dual(&replay_direction, supplied) {
            if let Ok(candidate) = combine_exact_lower(replay.lower, &exact_constant) {
                if candidate > lower_bound {
                    predicate_multipliers =
                        clone_f64(supplied, "accepted Box-cut predicate multipliers")?;
                    lower_bound = candidate;
                    supplied_predicate_multipliers_used = true;
                }
            }
        }
    }

    Ok(ReluTailBoxCutCertificate {
        lower_bound,
        zero_predicate_lower_bound,
        replay_direction,
        upper_box_multipliers: clone_f64(upper_box_multipliers, "accepted upper Box multipliers")?,
        lower_box_multipliers: clone_f64(lower_box_multipliers, "accepted lower Box multipliers")?,
        predicate_multipliers,
        exact_constant,
        supplied_predicate_multipliers_used,
    })
}

fn exact_line_correction(
    exact_a: &BigRational,
    exact_b: &BigRational,
    direction: f64,
    lower: &BigRational,
    upper: &BigRational,
    coordinate: usize,
) -> Result<BigRational, ReluTailDualError> {
    let direction = exact_domain_f64(direction, coordinate, "direction correction")?;
    let difference = checked_rational(
        exact_a - direction,
        "exact coefficient minus dyadic direction",
        coordinate,
    )?;
    let endpoint = if difference.is_negative() {
        upper
    } else {
        lower
    };
    let product = checked_rational(
        difference * endpoint,
        "coefficient correction times endpoint",
        coordinate,
    )?;
    checked_rational(
        product + exact_b,
        "line intercept plus endpoint correction",
        coordinate,
    )
}

#[derive(Clone, Debug)]
struct AcceptedCandidate {
    lower: f64,
    zero_lower: f64,
    direction: Vec<f64>,
    multipliers: Vec<f64>,
    supplied_used: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum CandidateReplayOutcome {
    Replayed { zero_predicate_lower_bound: f64 },
    Rejected,
    AllocationFallback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SuppliedReplayOutcome {
    ReplayedOrUnused,
    AllocationFallback,
}

fn replay_direction(
    domain: &ConstrainedZonotope64,
    exact_constant: &BigRational,
    variables: &[SlopeVariable],
    direction: Vec<f64>,
    zero: &[f64],
    supplied: Option<&[f64]>,
    best: &mut AcceptedCandidate,
) -> CandidateReplayOutcome {
    replay_direction_with_cloner(
        domain,
        exact_constant,
        variables,
        direction,
        zero,
        supplied,
        best,
        &mut candidate_clone_f64,
    )
}

#[allow(clippy::too_many_arguments)]
fn replay_direction_with_cloner<F>(
    domain: &ConstrainedZonotope64,
    exact_constant: &BigRational,
    variables: &[SlopeVariable],
    direction: Vec<f64>,
    zero: &[f64],
    supplied: Option<&[f64]>,
    best: &mut AcceptedCandidate,
    clone_candidate: &mut F,
) -> CandidateReplayOutcome
where
    F: FnMut(&[f64]) -> Option<Vec<f64>>,
{
    if !valid_direction_slopes(&direction, variables) {
        return CandidateReplayOutcome::Rejected;
    }
    let Ok(bounds) = domain.evaluate_dual(&direction, zero) else {
        return CandidateReplayOutcome::Rejected;
    };
    let Ok(zero_lower) = combine_exact_lower(bounds.lower, exact_constant) else {
        return CandidateReplayOutcome::Rejected;
    };
    let Some(candidate_multipliers) = clone_candidate(zero) else {
        return CandidateReplayOutcome::AllocationFallback;
    };
    let mut candidate = AcceptedCandidate {
        lower: zero_lower,
        zero_lower,
        direction,
        multipliers: candidate_multipliers,
        supplied_used: false,
    };
    let supplied_outcome = maybe_replay_supplied_with_cloner(
        domain,
        exact_constant,
        supplied,
        &mut candidate,
        clone_candidate,
    );
    if candidate.lower > best.lower {
        *best = candidate;
    }
    match supplied_outcome {
        SuppliedReplayOutcome::ReplayedOrUnused => CandidateReplayOutcome::Replayed {
            zero_predicate_lower_bound: zero_lower,
        },
        SuppliedReplayOutcome::AllocationFallback => CandidateReplayOutcome::AllocationFallback,
    }
}

fn maybe_replay_supplied(
    domain: &ConstrainedZonotope64,
    exact_constant: &BigRational,
    supplied: Option<&[f64]>,
    candidate: &mut AcceptedCandidate,
) -> SuppliedReplayOutcome {
    maybe_replay_supplied_with_cloner(
        domain,
        exact_constant,
        supplied,
        candidate,
        &mut candidate_clone_f64,
    )
}

fn maybe_replay_supplied_with_cloner<F>(
    domain: &ConstrainedZonotope64,
    exact_constant: &BigRational,
    supplied: Option<&[f64]>,
    candidate: &mut AcceptedCandidate,
    clone_candidate: &mut F,
) -> SuppliedReplayOutcome
where
    F: FnMut(&[f64]) -> Option<Vec<f64>>,
{
    let Some(supplied) = supplied else {
        return SuppliedReplayOutcome::ReplayedOrUnused;
    };
    let Ok(bounds) = domain.evaluate_dual(&candidate.direction, supplied) else {
        return SuppliedReplayOutcome::ReplayedOrUnused;
    };
    let Ok(lower) = combine_exact_lower(bounds.lower, exact_constant) else {
        return SuppliedReplayOutcome::ReplayedOrUnused;
    };
    if lower > candidate.lower {
        let Some(accepted_multipliers) = clone_candidate(supplied) else {
            return SuppliedReplayOutcome::AllocationFallback;
        };
        candidate.lower = lower;
        candidate.multipliers = accepted_multipliers;
        candidate.supplied_used = true;
    }
    SuppliedReplayOutcome::ReplayedOrUnused
}

fn valid_supplied_multipliers(supplied: Option<&[f64]>, expected: usize) -> Option<&[f64]> {
    let supplied = supplied?;
    (supplied.len() == expected
        && supplied
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0))
    .then_some(supplied)
}

fn valid_direction_slopes(direction: &[f64], variables: &[SlopeVariable]) -> bool {
    variables.iter().all(|variable| {
        direction.get(variable.coordinate).is_some_and(|&slope| {
            slope.is_finite() && valid_direct_slope(slope, &variable.exact_upper)
        })
    })
}

fn valid_direct_slope(slope: f64, exact_upper: &BigRational) -> bool {
    if !slope.is_finite() || slope < 0.0 {
        return false;
    }
    BigRational::from_float(slope).is_some_and(|exact| exact <= *exact_upper)
}

fn finish_result(
    best: AcceptedCandidate,
    line_plan: LinePlan,
    zero_predicate_candidate_replays: ReluTailDualZeroPredicateCandidateReplays,
    candidates_replayed: usize,
    iterations_completed: usize,
    status: ReluTailDualStatus,
    plan: Option<ReluTailDualPlan>,
) -> ReluTailDualResult {
    ReluTailDualResult {
        lower_bound: best.lower,
        zero_multiplier_lower_bound: best.zero_lower,
        zero_predicate_candidate_replays,
        direction: best.direction,
        multipliers: best.multipliers,
        exact_constant: line_plan.exact_constant,
        optimizable_slopes: line_plan.variables.len(),
        candidates_replayed,
        iterations_completed,
        status,
        plan,
        supplied_multipliers_used: best.supplied_used,
    }
}

#[derive(Clone, Copy, Debug)]
enum CandidateFailure {
    Deadline(usize),
    NonFinite(usize),
    Allocation(usize),
}

fn projected_adam_candidate(
    domain: &ConstrainedZonotope64,
    variables: &[SlopeVariable],
    direction: Vec<f64>,
    config: ReluTailDualConfig,
) -> Result<(Vec<f64>, usize), CandidateFailure> {
    let start = Instant::now();
    projected_adam_candidate_with_clock(domain, variables, direction, config, || start.elapsed())
}

fn projected_adam_candidate_with_clock<C>(
    domain: &ConstrainedZonotope64,
    variables: &[SlopeVariable],
    mut direction: Vec<f64>,
    config: ReluTailDualConfig,
    elapsed: C,
) -> Result<(Vec<f64>, usize), CandidateFailure>
where
    C: FnMut() -> Duration,
{
    let mut deadline = CandidateDeadline::new(config.wall_time, elapsed);
    // The candidate-only clock starts inside the wrapper above.  Check it
    // before any allocation, initialization, or approximate scoring work.
    deadline.checkpoint(0)?;
    let mut first = candidate_zeros(variables.len(), 0, &mut deadline)?;
    let mut second = candidate_zeros(variables.len(), 0, &mut deadline)?;
    let mut gradient = candidate_zeros(variables.len(), 0, &mut deadline)?;
    let mut variable_slot = candidate_usizes(domain.value_dim(), 0, &mut deadline)?;
    for (slot, variable) in variables.iter().enumerate() {
        deadline.visit(0)?;
        variable_slot[variable.coordinate] = slot;
    }
    let mut best = candidate_clone_with_deadline(&direction, 0, &mut deadline)?;
    let mut best_objective =
        approximate_zero_multiplier_objective(domain, &direction, 0, &mut deadline)?;

    for iteration in 0..config.iterations {
        deadline.checkpoint(iteration)?;
        for value in &mut gradient {
            deadline.visit(iteration)?;
            *value = 0.0;
        }
        for (slot, variable) in variables.iter().enumerate() {
            deadline.visit(iteration)?;
            gradient[slot] =
                domain.center()[variable.coordinate] - domain.box_remainder()[variable.coordinate];
        }
        for generator in domain.generators() {
            // Count and check every generator column, including empty ones.
            deadline.visit(iteration)?;
            let mut projection = 0.0_f64;
            for (coordinate, coefficient) in generator.entries() {
                deadline.visit(iteration)?;
                projection += direction[coordinate] * coefficient;
                if !projection.is_finite() {
                    return Err(CandidateFailure::NonFinite(iteration));
                }
            }
            let sign = if projection > 0.0 {
                1.0
            } else if projection < 0.0 {
                -1.0
            } else {
                0.0
            };
            if sign != 0.0 {
                for (coordinate, coefficient) in generator.entries() {
                    deadline.visit(iteration)?;
                    let slot = variable_slot[coordinate];
                    if slot != usize::MAX {
                        gradient[slot] -= sign * coefficient;
                    }
                }
            }
        }

        let step =
            i32::try_from(iteration + 1).map_err(|_| CandidateFailure::NonFinite(iteration))?;
        let first_correction = 1.0 - config.beta1.powi(step);
        let second_correction = 1.0 - config.beta2.powi(step);
        if !first_correction.is_finite()
            || !second_correction.is_finite()
            || first_correction <= 0.0
            || second_correction <= 0.0
        {
            return Err(CandidateFailure::NonFinite(iteration));
        }
        for (slot, variable) in variables.iter().enumerate() {
            deadline.visit(iteration)?;
            first[slot] = config.beta1 * first[slot] + (1.0 - config.beta1) * gradient[slot];
            second[slot] = config.beta2 * second[slot]
                + (1.0 - config.beta2) * gradient[slot] * gradient[slot];
            let first_hat = first[slot] / first_correction;
            let second_hat = second[slot] / second_correction;
            let update = config.learning_rate * first_hat / (second_hat.sqrt() + config.epsilon);
            let candidate = (direction[variable.coordinate] + update).clamp(0.0, variable.upper);
            if !gradient[slot].is_finite()
                || !first[slot].is_finite()
                || !second[slot].is_finite()
                || !candidate.is_finite()
                || !valid_direct_slope(candidate, &variable.exact_upper)
            {
                return Err(CandidateFailure::NonFinite(iteration));
            }
            direction[variable.coordinate] = canonical_zero(candidate);
        }
        let completed = iteration + 1;
        let objective =
            approximate_zero_multiplier_objective(domain, &direction, completed, &mut deadline)?;
        if objective > best_objective {
            best_objective = objective;
            copy_candidate_direction(&mut best, &direction, completed, &mut deadline)?;
        }
    }
    deadline.checkpoint(config.iterations)?;
    Ok((best, config.iterations))
}

fn approximate_zero_multiplier_objective<C>(
    domain: &ConstrainedZonotope64,
    direction: &[f64],
    iterations: usize,
    deadline: &mut CandidateDeadline<C>,
) -> Result<f64, CandidateFailure>
where
    C: FnMut() -> Duration,
{
    let mut value = 0.0_f64;
    for coordinate in 0..domain.value_dim() {
        deadline.visit(iterations)?;
        value += direction[coordinate] * domain.center()[coordinate];
        value -= direction[coordinate].abs() * domain.box_remainder()[coordinate];
        if !value.is_finite() {
            return Err(CandidateFailure::NonFinite(iterations));
        }
    }
    for generator in domain.generators() {
        // Empty generator columns still consume bounded candidate time.
        deadline.visit(iterations)?;
        let mut projection = 0.0_f64;
        for (coordinate, coefficient) in generator.entries() {
            deadline.visit(iterations)?;
            projection += direction[coordinate] * coefficient;
            if !projection.is_finite() {
                return Err(CandidateFailure::NonFinite(iterations));
            }
        }
        value -= projection.abs();
        if !value.is_finite() {
            return Err(CandidateFailure::NonFinite(iterations));
        }
    }
    Ok(value)
}

fn candidate_zeros<C>(
    count: usize,
    iterations: usize,
    deadline: &mut CandidateDeadline<C>,
) -> Result<Vec<f64>, CandidateFailure>
where
    C: FnMut() -> Duration,
{
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| CandidateFailure::Allocation(iterations))?;
    for _ in 0..count {
        deadline.visit(iterations)?;
        values.push(0.0);
    }
    Ok(values)
}

fn candidate_usizes<C>(
    count: usize,
    iterations: usize,
    deadline: &mut CandidateDeadline<C>,
) -> Result<Vec<usize>, CandidateFailure>
where
    C: FnMut() -> Duration,
{
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| CandidateFailure::Allocation(iterations))?;
    for _ in 0..count {
        deadline.visit(iterations)?;
        values.push(usize::MAX);
    }
    Ok(values)
}

fn candidate_clone_with_deadline<C>(
    source: &[f64],
    iterations: usize,
    deadline: &mut CandidateDeadline<C>,
) -> Result<Vec<f64>, CandidateFailure>
where
    C: FnMut() -> Duration,
{
    let mut values = Vec::new();
    values
        .try_reserve_exact(source.len())
        .map_err(|_| CandidateFailure::Allocation(iterations))?;
    for &value in source {
        deadline.visit(iterations)?;
        values.push(value);
    }
    Ok(values)
}

fn copy_candidate_direction<C>(
    target: &mut [f64],
    source: &[f64],
    iterations: usize,
    deadline: &mut CandidateDeadline<C>,
) -> Result<(), CandidateFailure>
where
    C: FnMut() -> Duration,
{
    debug_assert_eq!(target.len(), source.len());
    for (target, &source) in target.iter_mut().zip(source) {
        deadline.visit(iterations)?;
        *target = source;
    }
    Ok(())
}

struct CandidateDeadline<C> {
    wall_time: Duration,
    elapsed: C,
    visits_until_check: usize,
}

impl<C> CandidateDeadline<C>
where
    C: FnMut() -> Duration,
{
    fn new(wall_time: Duration, elapsed: C) -> Self {
        Self {
            wall_time,
            elapsed,
            visits_until_check: 0,
        }
    }

    fn checkpoint(&mut self, iterations: usize) -> Result<(), CandidateFailure> {
        if (self.elapsed)() >= self.wall_time {
            return Err(CandidateFailure::Deadline(iterations));
        }
        self.visits_until_check = RELU_TAIL_DUAL_DEADLINE_CHECK_STRIDE;
        Ok(())
    }

    fn visit(&mut self, iterations: usize) -> Result<(), CandidateFailure> {
        if self.visits_until_check == 0 {
            self.checkpoint(iterations)?;
        }
        self.visits_until_check -= 1;
        Ok(())
    }
}

fn combine_exact_lower(
    replay_lower: f64,
    exact_constant: &BigRational,
) -> Result<f64, ReluTailDualError> {
    let replay = exact_domain_f64(replay_lower, 0, "outward replay lower bound")?;
    let combined = checked_rational(
        replay + exact_constant,
        "replay lower plus exact correction",
        0,
    )?;
    floor_finite(&combined, 0, "final lower-bound conversion")
}

fn floor_finite(
    value: &BigRational,
    coordinate: usize,
    operation: &'static str,
) -> Result<f64, ReluTailDualError> {
    if value.is_zero() {
        return Ok(0.0);
    }
    let max = BigRational::from_float(f64::MAX).expect("finite binary64 maximum");
    if value >= &max {
        return Ok(f64::MAX);
    }
    if value < &-max {
        return Err(ReluTailDualError::NonFiniteArithmetic {
            coordinate,
            operation,
        });
    }
    let mut candidate = value
        .to_f64()
        .ok_or(ReluTailDualError::NonFiniteArithmetic {
            coordinate,
            operation,
        })?;
    if !candidate.is_finite() {
        return Err(ReluTailDualError::NonFiniteArithmetic {
            coordinate,
            operation,
        });
    }
    let exact_candidate =
        BigRational::from_float(candidate).ok_or(ReluTailDualError::NonFiniteArithmetic {
            coordinate,
            operation,
        })?;
    if exact_candidate > *value {
        candidate = candidate.next_down();
        if !candidate.is_finite()
            || BigRational::from_float(candidate).is_none_or(|rounded| rounded > *value)
        {
            return Err(ReluTailDualError::NonFiniteArithmetic {
                coordinate,
                operation,
            });
        }
    }
    Ok(canonical_zero(candidate))
}

fn nearest_finite(
    value: &BigRational,
    coordinate: usize,
    operation: &'static str,
) -> Result<f64, ReluTailDualError> {
    let max = BigRational::from_float(f64::MAX).expect("finite binary64 maximum");
    if value >= &max {
        return Ok(f64::MAX);
    }
    if value <= &-max {
        return Ok(-f64::MAX);
    }
    let candidate = value
        .to_f64()
        .ok_or(ReluTailDualError::NonFiniteArithmetic {
            coordinate,
            operation,
        })?;
    if !candidate.is_finite() {
        return Err(ReluTailDualError::NonFiniteArithmetic {
            coordinate,
            operation,
        });
    }
    Ok(canonical_zero(candidate))
}

fn exact_objective_f64(
    value: f64,
    field: &'static str,
    index: usize,
) -> Result<BigRational, ReluTailDualError> {
    BigRational::from_float(value).ok_or(ReluTailDualError::NonFiniteObjective { field, index })
}

fn exact_domain_f64(
    value: f64,
    coordinate: usize,
    operation: &'static str,
) -> Result<BigRational, ReluTailDualError> {
    BigRational::from_float(value).ok_or(ReluTailDualError::NonFiniteArithmetic {
        coordinate,
        operation,
    })
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn rational_bits(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits())
}

fn checked_rational(
    value: BigRational,
    operation: &'static str,
    coordinate: usize,
) -> Result<BigRational, ReluTailDualError> {
    let bits = rational_bits(&value);
    if bits > RELU_TAIL_DUAL_HARD_MAX_INTERMEDIATE_RATIONAL_BITS {
        Err(ReluTailDualError::RationalGrowthLimit {
            coordinate,
            operation,
            bits,
            limit: RELU_TAIL_DUAL_HARD_MAX_INTERMEDIATE_RATIONAL_BITS,
        })
    } else {
        Ok(value)
    }
}

fn check_declared_rationals(
    coefficients: &[BigRational],
    bias: &BigRational,
) -> Result<(), ReluTailDualError> {
    require_resource_limit(
        "objective coefficients",
        u64::try_from(coefficients.len()).unwrap_or(u64::MAX),
        RELU_TAIL_DUAL_HARD_MAX_VALUE_DIM as u64,
    )?;
    let mut total = 0_u64;
    for (index, coefficient) in coefficients.iter().enumerate() {
        let bits = rational_bits(coefficient);
        if bits > RELU_TAIL_DUAL_HARD_MAX_INPUT_RATIONAL_BITS {
            return Err(ReluTailDualError::RationalInputLimit {
                field: "coefficients",
                index,
                bits,
                limit: RELU_TAIL_DUAL_HARD_MAX_INPUT_RATIONAL_BITS,
            });
        }
        total = total
            .checked_add(bits)
            .ok_or(ReluTailDualError::ResourceOverflow {
                resource: "total objective rational bits",
            })?;
    }
    let bias_bits = rational_bits(bias);
    if bias_bits > RELU_TAIL_DUAL_HARD_MAX_INPUT_RATIONAL_BITS {
        return Err(ReluTailDualError::RationalInputLimit {
            field: "bias",
            index: 0,
            bits: bias_bits,
            limit: RELU_TAIL_DUAL_HARD_MAX_INPUT_RATIONAL_BITS,
        });
    }
    total = total
        .checked_add(bias_bits)
        .ok_or(ReluTailDualError::ResourceOverflow {
            resource: "total objective rational bits",
        })?;
    require_resource_limit(
        "total objective rational bits",
        total,
        RELU_TAIL_DUAL_HARD_MAX_TOTAL_RATIONAL_BITS,
    )
}

fn check_mandatory_resources(
    domain: &ConstrainedZonotope64,
    margin: &ExactReluTailMargin,
) -> Result<usize, ReluTailDualError> {
    check_mandatory_margin_resources(domain.value_dim(), margin)?;
    check_mandatory_domain_resources(domain)
}

fn check_mandatory_margin_resources(
    value_dim: usize,
    margin: &ExactReluTailMargin,
) -> Result<(), ReluTailDualError> {
    if margin.coefficients.len() != value_dim {
        return Err(ReluTailDualError::Shape {
            field: "margin coefficients",
            expected: value_dim,
            got: margin.coefficients.len(),
        });
    }
    check_declared_rationals(&margin.coefficients, &margin.bias)
}

fn check_mandatory_domain_resources(
    domain: &ConstrainedZonotope64,
) -> Result<usize, ReluTailDualError> {
    require_resource_limit(
        "value dimension",
        u64::try_from(domain.value_dim()).unwrap_or(u64::MAX),
        RELU_TAIL_DUAL_HARD_MAX_VALUE_DIM as u64,
    )?;
    require_resource_limit(
        "alpha dimension",
        u64::try_from(domain.alpha_dim()).unwrap_or(u64::MAX),
        RELU_TAIL_DUAL_HARD_MAX_ALPHA_DIM as u64,
    )?;
    require_resource_limit(
        "constraints",
        u64::try_from(domain.constraint_count()).unwrap_or(u64::MAX),
        RELU_TAIL_DUAL_HARD_MAX_CONSTRAINTS as u64,
    )?;
    let constraint_elements = domain
        .constraint_count()
        .checked_mul(domain.alpha_dim())
        .ok_or(ReluTailDualError::ResourceOverflow {
            resource: "constraint elements",
        })?;
    require_resource_limit(
        "constraint elements",
        u64::try_from(constraint_elements).unwrap_or(u64::MAX),
        RELU_TAIL_DUAL_HARD_MAX_CONSTRAINT_ELEMENTS as u64,
    )?;
    let generator_nonzeros = generator_nonzeros(domain)?;
    require_resource_limit(
        "generator nonzeros",
        u64::try_from(generator_nonzeros).unwrap_or(u64::MAX),
        RELU_TAIL_DUAL_HARD_MAX_GENERATOR_NONZEROS as u64,
    )?;
    let terms = (constraint_elements as u128)
        .checked_add(generator_nonzeros as u128)
        .and_then(|value| value.checked_add(domain.value_dim() as u128))
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(ReluTailDualError::ResourceOverflow {
            resource: "mandatory baseline terms",
        })?;
    require_resource_limit(
        "mandatory baseline terms",
        terms,
        RELU_TAIL_DUAL_HARD_MAX_BASELINE_TERMS,
    )?;
    Ok(generator_nonzeros)
}

fn generator_nonzeros(domain: &ConstrainedZonotope64) -> Result<usize, ReluTailDualError> {
    domain.generators().iter().try_fold(0_usize, |sum, column| {
        sum.checked_add(column.nnz())
            .ok_or(ReluTailDualError::ResourceOverflow {
                resource: "generator nonzeros",
            })
    })
}

fn require_resource_limit(
    resource: &'static str,
    actual: u64,
    limit: u64,
) -> Result<(), ReluTailDualError> {
    if actual > limit {
        Err(ReluTailDualError::ResourceLimit {
            resource,
            actual,
            limit,
        })
    } else {
        Ok(())
    }
}

fn zero_f64(count: usize, resource: &'static str) -> Result<Vec<f64>, ReluTailDualError> {
    let mut values = Vec::new();
    try_reserve(&mut values, count, resource)?;
    values.resize(count, 0.0);
    Ok(values)
}

fn clone_f64(values: &[f64], resource: &'static str) -> Result<Vec<f64>, ReluTailDualError> {
    let mut result = Vec::new();
    try_reserve(&mut result, values.len(), resource)?;
    result.extend_from_slice(values);
    Ok(result)
}

fn candidate_clone_f64(values: &[f64]) -> Option<Vec<f64>> {
    let mut result = Vec::new();
    result.try_reserve_exact(values.len()).ok()?;
    result.extend_from_slice(values);
    Some(result)
}

fn try_reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    resource: &'static str,
) -> Result<(), ReluTailDualError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| ReluTailDualError::AllocationFailure { resource })
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};
    use num_bigint::BigInt;
    use proptest::prelude::*;

    use super::*;

    fn exact(value: f64) -> BigRational {
        BigRational::from_float(value).expect("finite test dyadic")
    }

    fn ratio(numerator: i64, denominator: i64) -> BigRational {
        BigRational::new(numerator.into(), denominator.into())
    }

    fn limits() -> ReluTailDualLimits {
        ReluTailDualLimits {
            max_value_dim: 4_096,
            max_alpha_dim: 4_096,
            max_constraints: 4_096,
            max_generator_nonzeros: 1_000_000,
            max_optimizable_slopes: 4_096,
            max_iterations: 64,
            max_search_work: 100_000_000,
            max_wall_time: Duration::from_secs(5),
        }
    }

    fn config(iterations: usize) -> ReluTailDualConfig {
        ReluTailDualConfig {
            iterations,
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            wall_time: Duration::from_secs(1),
            limits: limits(),
        }
    }

    fn box_config(low_iterations: usize, high_iterations: usize) -> ReluTailBoxCutOptimizerConfig {
        ReluTailBoxCutOptimizerConfig {
            schedules: [
                ReluTailBoxCutAdamSchedule {
                    iterations: low_iterations,
                    learning_rate: 0.005,
                    decay: 0.98,
                },
                ReluTailBoxCutAdamSchedule {
                    iterations: high_iterations,
                    learning_rate: 0.1,
                    decay: 0.98,
                },
            ],
            multiplier_cap: 16.0,
            wall_time: Duration::from_secs(1),
            limits: ReluTailBoxCutOptimizerLimits {
                max_value_dim: 4_096,
                max_box_variables: 8_192,
                max_total_iterations: 128,
                max_restarts: 2,
                max_exact_replays: 2,
                max_generator_nonzeros: 1_000_000,
                max_search_work: 100_000_000,
                max_wall_time: Duration::from_secs(5),
            },
        }
    }

    fn margin(coefficients: Vec<BigRational>, bias: BigRational) -> ExactReluTailMargin {
        ExactReluTailMargin::try_new(coefficients, bias).expect("small exact objective")
    }

    fn auxiliary(lower: Vec<f64>, upper: Vec<f64>) -> CertifiedAuxiliaryBounds64 {
        CertifiedAuxiliaryBounds64::try_new(lower, upper).expect("valid test auxiliary bounds")
    }

    fn assert_result_bit_identical(left: &ReluTailDualResult, right: &ReluTailDualResult) {
        assert_eq!(left.lower_bound.to_bits(), right.lower_bound.to_bits());
        assert_eq!(
            left.zero_multiplier_lower_bound.to_bits(),
            right.zero_multiplier_lower_bound.to_bits()
        );
        let replay_bits = |replay: ReluTailDualZeroPredicateCandidateReplays| {
            (
                replay.zero_positive_slope_lower_bound.to_bits(),
                replay.upper_endpoint_lower_bound.map(f64::to_bits),
                replay.canonical_lower_bound.map(f64::to_bits),
                replay.optimized_lower_bound.map(f64::to_bits),
            )
        };
        assert_eq!(
            replay_bits(left.zero_predicate_candidate_replays),
            replay_bits(right.zero_predicate_candidate_replays)
        );
        assert_eq!(
            left.direction
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            right
                .direction
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            left.multipliers
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            right
                .multipliers
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(left.exact_constant, right.exact_constant);
        assert_eq!(left.optimizable_slopes, right.optimizable_slopes);
        assert_eq!(left.candidates_replayed, right.candidates_replayed);
        assert_eq!(left.iterations_completed, right.iterations_completed);
        assert_eq!(left.status, right.status);
        assert_eq!(left.plan, right.plan);
        assert_eq!(
            left.supplied_multipliers_used,
            right.supplied_multipliers_used
        );
    }

    fn exact_relu(value: &BigRational) -> BigRational {
        value.clone().max(BigRational::zero())
    }

    #[test]
    fn exact_row_helper_subtracts_before_any_rounding() {
        let result = exact_relu_tail_margin_from_f64_rows(
            &[f64::MAX, -0.0],
            &[-f64::MAX, 0.0],
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
        )
        .unwrap();
        assert_eq!(result.coefficients()[0], exact(f64::MAX) - exact(-f64::MAX));
        assert!(result.coefficients()[0] > exact(f64::MAX));
        assert!(result.coefficients()[1].is_zero());
        assert_eq!(
            result.bias(),
            &(exact(f64::MIN_POSITIVE) - exact(-f64::MIN_POSITIVE))
        );
    }

    #[test]
    fn exact_row_helper_rejects_shape_and_nonfinite_inputs() {
        assert!(matches!(
            exact_relu_tail_margin_from_f64_rows(&[1.0], &[], 0.0, 0.0),
            Err(ReluTailDualError::Shape {
                field: "challenger row",
                ..
            })
        ));
        assert!(matches!(
            exact_relu_tail_margin_from_f64_rows(&[f64::NAN], &[0.0], 0.0, 0.0),
            Err(ReluTailDualError::NonFiniteObjective {
                field: "target row",
                ..
            })
        ));
        assert!(matches!(
            exact_relu_tail_margin_from_f64_rows(&[0.0], &[0.0], f64::INFINITY, 0.0),
            Err(ReluTailDualError::NonFiniteObjective {
                field: "target bias",
                ..
            })
        ));
    }

    #[test]
    fn exact_coefficient_above_f64_max_uses_finite_direction_and_correction() {
        let domain = ConstrainedZonotope64::from_certified_bounds(&[1.0], &[1.0], &[true]).unwrap();
        let declared =
            exact_relu_tail_margin_from_f64_rows(&[f64::MAX], &[-f64::MAX], 0.0, 0.0).unwrap();
        let result =
            bound_relu_tail_triangle_dual_unwired(&domain, &declared, None, config(0)).unwrap();
        assert_eq!(result.direction, vec![f64::MAX]);
        assert_eq!(result.lower_bound, f64::MAX);
        assert!(exact(result.lower_bound) <= declared.coefficients()[0]);
    }

    #[test]
    fn internally_derived_bounds_classify_active_inactive_and_unstable() {
        let domain = ConstrainedZonotope64::from_certified_bounds(
            &[-3.0, 1.0, -2.0],
            &[-1.0, 4.0, 3.0],
            &[false, false, false],
        )
        .unwrap();
        let declared = margin(vec![ratio(7, 5), ratio(1, 3), ratio(-5, 3)], ratio(2, 7));
        let plan = build_line_plan(&domain, &declared).unwrap();
        assert_eq!(plan.fixed_direction[0], 0.0);
        assert_ne!(plan.fixed_direction[1], 0.0);
        assert_ne!(plan.fixed_direction[2], 0.0);
        assert!(plan.variables.is_empty());
    }

    #[test]
    fn auxiliary_path_rejects_wrong_dimension_and_empty_intersection() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let wrong_dimension = auxiliary(Vec::new(), Vec::new());
        assert!(matches!(
            bound_relu_tail_triangle_dual_with_auxiliary_bounds_unwired(
                &domain,
                &wrong_dimension,
                &declared,
                None,
                config(0),
            ),
            Err(ReluTailDualError::AuxiliaryDimensionMismatch {
                expected: 1,
                got: 0
            })
        ));

        let disjoint = auxiliary(vec![2.0], vec![3.0]);
        assert!(matches!(
            bound_relu_tail_triangle_dual_with_auxiliary_bounds_unwired(
                &domain,
                &disjoint,
                &declared,
                None,
                config(0),
            ),
            Err(ReluTailDualError::EmptyAuxiliaryIntersection { coordinate: 0 })
        ));
    }

    #[test]
    fn nonrestrictive_auxiliary_path_is_bit_identical_to_original() {
        let domain = ConstrainedZonotope64::try_new(
            vec![0.0, 2.0],
            vec![vec![(0, 1.0), (1, 1.0)]],
            Array2::zeros((0, 1)),
            Vec::new(),
            vec![0.0, 0.0],
        )
        .unwrap();
        let declared = margin(vec![ratio(1, 1), ratio(-1, 2)], ratio(1, 7));
        let nonrestrictive = auxiliary(vec![-f64::MAX, -f64::MAX], vec![f64::MAX, f64::MAX]);
        let original =
            bound_relu_tail_triangle_dual_unwired(&domain, &declared, None, config(8)).unwrap();
        let with_auxiliary = bound_relu_tail_triangle_dual_with_auxiliary_bounds_unwired(
            &domain,
            &nonrestrictive,
            &declared,
            None,
            config(8),
        )
        .unwrap();
        assert_result_bit_identical(&original, &with_auxiliary);
    }

    #[test]
    fn auxiliary_phase_can_exclude_spurious_cz_points_but_is_not_monotone() {
        let domain = ConstrainedZonotope64::from_certified_bounds(
            &[-1.0, -1.0],
            &[1.0, 1.0],
            &[false, false],
        )
        .unwrap();
        let declared = margin(vec![ratio(1, 1), ratio(1, 1)], ratio(0, 1));
        let certified = auxiliary(vec![0.25, -1.0], vec![1.0, -0.25]);
        let original =
            bound_relu_tail_triangle_dual_unwired(&domain, &declared, None, config(0)).unwrap();
        let result = bound_relu_tail_triangle_dual_with_auxiliary_bounds_unwired(
            &domain,
            &certified,
            &declared,
            None,
            config(0),
        )
        .unwrap();

        assert_eq!(result.status, ReluTailDualStatus::NoOptimizableSlopes);
        assert_eq!(result.optimizable_slopes, 0);
        assert_eq!(result.direction, vec![1.0, 0.0]);
        // Replay still sees the spurious CZ point x0=-1, so the newly active
        // exact line is sound but weaker than the original zero-slope line.
        assert_eq!(result.lower_bound, -1.0);
        assert!(original.lower_bound > result.lower_bound);
        for x0 in [ratio(1, 4), ratio(1, 2), ratio(1, 1)] {
            for x1 in [ratio(-1, 1), ratio(-1, 4)] {
                let concrete = exact_relu(&x0) + exact_relu(&x1);
                assert!(exact(result.lower_bound) <= concrete);
            }
        }
    }

    #[test]
    fn auxiliary_negative_chord_is_exact_on_intersection_not_spurious_hull() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-2.0], &[3.0], &[false]).unwrap();
        let certified = auxiliary(vec![-0.5], vec![1.5]);
        let coefficient = ratio(-5, 3);
        let declared = margin(vec![coefficient.clone()], ratio(0, 1));
        let plan = build_line_plan_with_auxiliary_bounds(&domain, &certified, &declared).unwrap();
        let direction = exact(plan.fixed_direction[0]);
        for x in [
            ratio(-1, 2),
            ratio(-1, 4),
            ratio(0, 1),
            ratio(3, 4),
            ratio(3, 2),
        ] {
            let represented = &direction * &x + &plan.exact_constant;
            let true_term = &coefficient * exact_relu(&x);
            assert!(represented <= true_term, "x={x}");
        }
        // The tightened chord is deliberately not claimed on this spurious Z
        // point outside the certified concrete interval.
        let spurious = ratio(-2, 1);
        assert!(
            &direction * &spurious + &plan.exact_constant > &coefficient * exact_relu(&spurious)
        );

        let result = bound_relu_tail_triangle_dual_with_auxiliary_bounds_unwired(
            &domain,
            &certified,
            &declared,
            None,
            config(0),
        )
        .unwrap();
        for x in [ratio(-1, 2), ratio(0, 1), ratio(3, 2)] {
            assert!(exact(result.lower_bound) <= &coefficient * exact_relu(&x));
        }
    }

    #[test]
    fn prepared_geometry_is_bit_identical_and_computes_one_hull_for_many_margins() {
        let domain = ConstrainedZonotope64::try_new(
            vec![0.0, 2.0, -2.0],
            vec![vec![(0, 1.0), (1, 1.0)], vec![(0, 0.5), (2, 0.25)]],
            array![[1.0, 0.0], [-1.0, 0.0]],
            vec![0.75, 1.0],
            vec![0.0, 0.125, 0.5],
        )
        .unwrap();
        let margins = [
            margin(vec![ratio(1, 1), ratio(-1, 2), ratio(2, 1)], ratio(1, 7)),
            margin(vec![ratio(-5, 3), ratio(1, 3), ratio(-2, 1)], ratio(-2, 9)),
            margin(vec![ratio(0, 1), ratio(1, 7), ratio(3, 2)], ratio(5, 11)),
        ];
        let configs = [config(8), config(8), config(0)];
        let supplied = [0.25, 0.0];
        let expected: Vec<_> = margins
            .iter()
            .zip(configs)
            .map(|(declared, selected)| {
                bound_relu_tail_triangle_dual_unwired(&domain, declared, Some(&supplied), selected)
                    .unwrap()
            })
            .collect();

        EXACT_COORDINATE_HULL_PASSES.with(|passes| passes.set(0));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        assert_eq!(prepared.value_dim(), 3);
        assert_eq!(prepared.coordinate_hull_generator_additions(), 4);
        let actual: Vec<_> = margins
            .iter()
            .zip(configs)
            .map(|(declared, selected)| {
                prepared
                    .bound_margin_unwired(declared, Some(&supplied), selected)
                    .unwrap()
            })
            .collect();
        EXACT_COORDINATE_HULL_PASSES.with(|passes| assert_eq!(passes.get(), 1));

        for (legacy, cached) in expected.iter().zip(&actual) {
            assert_result_bit_identical(legacy, cached);
        }
        assert_eq!(actual[0].plan.unwrap().generator_nonzeros, 4);
        assert_eq!(actual[0].candidates_replayed, 4);
        assert_eq!(
            actual[0]
                .zero_predicate_candidate_replays
                .zero_positive_slope_lower_bound
                .to_bits(),
            expected[0]
                .zero_predicate_candidate_replays
                .zero_positive_slope_lower_bound
                .to_bits()
        );

        let wrong_dimension = margin(vec![ratio(1, 1)], ratio(0, 1));
        assert!(matches!(
            prepared.bound_margin_unwired(&wrong_dimension, None, config(0)),
            Err(ReluTailDualError::Shape {
                field: "margin coefficients",
                expected: 3,
                got: 1,
            })
        ));
        EXACT_COORDINATE_HULL_PASSES.with(|passes| assert_eq!(passes.get(), 1));
    }

    #[test]
    fn prepared_auxiliary_is_bit_identical_and_shares_the_single_hull() {
        let domain = ConstrainedZonotope64::try_new(
            vec![0.0, 2.0, -2.0],
            vec![vec![(0, 1.0), (1, 1.0)], vec![(0, 0.5), (2, 0.25)]],
            array![[1.0, 0.0], [-1.0, 0.0]],
            vec![0.75, 1.0],
            vec![0.0, 0.125, 0.5],
        )
        .unwrap();
        let margins = [
            margin(vec![ratio(1, 1), ratio(-1, 2), ratio(2, 1)], ratio(1, 7)),
            margin(vec![ratio(-5, 3), ratio(1, 3), ratio(-2, 1)], ratio(-2, 9)),
        ];
        let certified = auxiliary(vec![-0.75, 1.5, -2.5], vec![1.25, 2.5, -1.5]);
        let nonrestrictive = auxiliary(
            vec![-f64::MAX, -f64::MAX, -f64::MAX],
            vec![f64::MAX, f64::MAX, f64::MAX],
        );
        let supplied = [0.25, 0.0];
        let expected_original: Vec<_> = margins
            .iter()
            .map(|declared| {
                bound_relu_tail_triangle_dual_unwired(&domain, declared, Some(&supplied), config(8))
                    .unwrap()
            })
            .collect();
        let expected_auxiliary: Vec<_> = margins
            .iter()
            .map(|declared| {
                bound_relu_tail_triangle_dual_with_auxiliary_bounds_unwired(
                    &domain,
                    &certified,
                    declared,
                    Some(&supplied),
                    config(8),
                )
                .unwrap()
            })
            .collect();

        EXACT_COORDINATE_HULL_PASSES.with(|passes| passes.set(0));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        for ((declared, expected_original), expected_auxiliary) in margins
            .iter()
            .zip(&expected_original)
            .zip(&expected_auxiliary)
        {
            let original = prepared
                .bound_margin_unwired(declared, Some(&supplied), config(8))
                .unwrap();
            let with_auxiliary = prepared
                .bound_margin_with_auxiliary_bounds_unwired(
                    &certified,
                    declared,
                    Some(&supplied),
                    config(8),
                )
                .unwrap();
            assert_result_bit_identical(expected_original, &original);
            assert_result_bit_identical(expected_auxiliary, &with_auxiliary);
        }

        let recovered = prepared
            .bound_margin_with_auxiliary_bounds_unwired(
                &nonrestrictive,
                &margins[0],
                Some(&supplied),
                config(8),
            )
            .unwrap();
        assert_result_bit_identical(&expected_original[0], &recovered);
        assert_eq!(prepared.coordinate_hull_generator_additions(), 4);
        EXACT_COORDINATE_HULL_PASSES.with(|passes| assert_eq!(passes.get(), 1));
    }

    #[test]
    fn prepared_auxiliary_rejects_dimension_shape_and_empty_intersection() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));

        let wrong_dimension = auxiliary(Vec::new(), Vec::new());
        assert!(matches!(
            prepared.bound_margin_with_auxiliary_bounds_unwired(
                &wrong_dimension,
                &declared,
                None,
                config(0),
            ),
            Err(ReluTailDualError::AuxiliaryDimensionMismatch {
                expected: 1,
                got: 0,
            })
        ));

        let wrong_margin = margin(Vec::new(), ratio(0, 1));
        let matching = auxiliary(vec![-1.0], vec![1.0]);
        assert!(matches!(
            prepared.bound_margin_with_auxiliary_bounds_unwired(
                &matching,
                &wrong_margin,
                None,
                config(0),
            ),
            Err(ReluTailDualError::Shape {
                field: "margin coefficients",
                expected: 1,
                got: 0,
            })
        ));

        let disjoint = auxiliary(vec![2.0], vec![3.0]);
        assert!(matches!(
            prepared.bound_margin_with_auxiliary_bounds_unwired(
                &disjoint,
                &declared,
                None,
                config(0),
            ),
            Err(ReluTailDualError::EmptyAuxiliaryIntersection { coordinate: 0 })
        ));
    }

    #[test]
    fn prepared_auxiliary_preserves_resource_and_deadline_fallbacks() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
        let certified = auxiliary(vec![-0.75], vec![0.75]);
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();

        let mut resource_limited = config(8);
        resource_limited.limits.max_value_dim = 0;
        let direct_resource = bound_relu_tail_triangle_dual_with_auxiliary_bounds_unwired(
            &domain,
            &certified,
            &declared,
            None,
            resource_limited,
        )
        .unwrap();
        let prepared_resource = prepared
            .bound_margin_with_auxiliary_bounds_unwired(
                &certified,
                &declared,
                None,
                resource_limited,
            )
            .unwrap();
        assert_eq!(
            prepared_resource.status,
            ReluTailDualStatus::ResourceFallback
        );
        assert_eq!(prepared_resource.candidates_replayed, 1);
        assert_result_bit_identical(&direct_resource, &prepared_resource);

        let mut deadline = config(64);
        deadline.wall_time = Duration::from_nanos(1);
        let direct_deadline = bound_relu_tail_triangle_dual_with_auxiliary_bounds_unwired(
            &domain, &certified, &declared, None, deadline,
        )
        .unwrap();
        let prepared_deadline = prepared
            .bound_margin_with_auxiliary_bounds_unwired(&certified, &declared, None, deadline)
            .unwrap();
        assert_eq!(prepared_deadline.status, ReluTailDualStatus::Deadline);
        assert_eq!(prepared_deadline.candidates_replayed, 3);
        assert_result_bit_identical(&direct_deadline, &prepared_deadline);
    }

    #[test]
    fn prepared_geometry_preserves_resource_and_deadline_fallbacks() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();

        let mut resource_limited = config(8);
        resource_limited.limits.max_value_dim = 0;
        let legacy_resource =
            bound_relu_tail_triangle_dual_unwired(&domain, &declared, None, resource_limited)
                .unwrap();
        let prepared_resource = prepared
            .bound_margin_unwired(&declared, None, resource_limited)
            .unwrap();
        assert_eq!(
            prepared_resource.status,
            ReluTailDualStatus::ResourceFallback
        );
        assert_eq!(prepared_resource.candidates_replayed, 1);
        assert_result_bit_identical(&legacy_resource, &prepared_resource);

        let mut deadline = config(64);
        deadline.wall_time = Duration::from_nanos(1);
        let legacy_deadline =
            bound_relu_tail_triangle_dual_unwired(&domain, &declared, None, deadline).unwrap();
        let prepared_deadline = prepared
            .bound_margin_unwired(&declared, None, deadline)
            .unwrap();
        assert_eq!(prepared_deadline.status, ReluTailDualStatus::Deadline);
        assert_eq!(prepared_deadline.candidates_replayed, 3);
        assert_result_bit_identical(&legacy_deadline, &prepared_deadline);
    }

    #[test]
    fn zero_box_multipliers_recover_nonrestrictive_m17_bit_for_bit() {
        let domain = ConstrainedZonotope64::try_new(
            vec![0.0, 2.0],
            vec![vec![(0, 1.0), (1, 1.0)]],
            Array2::zeros((0, 1)),
            Vec::new(),
            vec![0.0, 0.0],
        )
        .unwrap();
        let declared = margin(vec![ratio(1, 1), ratio(-1, 2)], ratio(1, 7));
        let nonrestrictive = auxiliary(vec![-f64::MAX, -f64::MAX], vec![f64::MAX, f64::MAX]);
        let result = bound_relu_tail_triangle_dual_with_auxiliary_box_cut_unwired(
            &domain,
            &nonrestrictive,
            &declared,
            &[0.0, 0.0],
            &[0.0, 0.0],
            None,
            config(8),
        )
        .unwrap();

        assert_eq!(result.status, ReluTailBoxCutStatus::Completed);
        assert_eq!(result.selected, ReluTailBoxCutSelection::Original);
        let auxiliary = result.auxiliary.as_ref().unwrap();
        assert_result_bit_identical(&result.original, auxiliary);
        let cut = result.box_cut.as_ref().unwrap();
        assert_eq!(
            cut.lower_bound.to_bits(),
            result.original.lower_bound.to_bits()
        );
        assert_eq!(
            cut.zero_predicate_lower_bound.to_bits(),
            result.original.zero_multiplier_lower_bound.to_bits()
        );
        assert_eq!(
            cut.replay_direction
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            result
                .original
                .direction
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(cut.exact_constant, result.original.exact_constant);
    }

    #[test]
    fn lower_box_cut_removes_spurious_negative_active_points() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-2.0], &[2.0], &[false]).unwrap();
        let certified = auxiliary(vec![1.0], vec![2.0]);
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let result = bound_relu_tail_triangle_dual_with_auxiliary_box_cut_unwired(
            &domain,
            &certified,
            &declared,
            &[0.0],
            &[1.0],
            None,
            config(0),
        )
        .unwrap();

        assert_eq!(result.status, ReluTailBoxCutStatus::Completed);
        assert_eq!(result.selected, ReluTailBoxCutSelection::BoxCut);
        assert_eq!(result.original.lower_bound, 0.0);
        assert_eq!(result.auxiliary.as_ref().unwrap().lower_bound, -2.0);
        let cut = result.box_cut.as_ref().unwrap();
        assert_eq!(cut.replay_direction, vec![0.0]);
        assert!(cut.lower_bound > 0.99);
        assert!(exact(cut.lower_bound) <= ratio(1, 1));
    }

    #[test]
    fn upper_box_cut_has_the_opposite_polarity_and_closes_upper_gap() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-2.0], &[100.0], &[false]).unwrap();
        let certified = auxiliary(vec![1.0], vec![2.0]);
        let declared = margin(vec![ratio(-1, 1)], ratio(0, 1));
        let result = bound_relu_tail_triangle_dual_with_auxiliary_box_cut_unwired(
            &domain,
            &certified,
            &declared,
            &[1.0],
            &[0.0],
            None,
            config(0),
        )
        .unwrap();

        assert_eq!(result.selected, ReluTailBoxCutSelection::BoxCut);
        assert!(result.original.lower_bound < -90.0);
        let auxiliary_lower = result.auxiliary.as_ref().unwrap().lower_bound;
        assert!(auxiliary_lower <= -100.0);
        assert!(auxiliary_lower > -100.000_000_000_000_1);
        let cut = result.box_cut.as_ref().unwrap();
        assert_eq!(cut.replay_direction, vec![0.0]);
        assert!(cut.lower_bound >= -2.000_000_000_000_001);
        assert!(exact(cut.lower_bound) <= ratio(-2, 1));
    }

    #[test]
    fn rounded_cut_direction_is_repaired_on_original_cz_hull() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-2.0], &[3.0], &[false]).unwrap();
        let certified = auxiliary(vec![1.0], vec![2.0]);
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let tiny = 2.0_f64.powi(-54);
        let result = bound_relu_tail_triangle_dual_with_auxiliary_box_cut_unwired(
            &domain,
            &certified,
            &declared,
            &[tiny],
            &[0.0],
            None,
            config(0),
        )
        .unwrap();
        let cut = result.box_cut.as_ref().unwrap();

        // p* = 1 + 2^-54 rounds back to 1.  The Box constant is -2*mu,
        // while the positive residual is minimized at the original CZ lower
        // hull -2, adding another -2*mu.  Charging only the auxiliary lower
        // endpoint +1 would produce a different and unjustified constant.
        assert_eq!(cut.replay_direction, vec![1.0]);
        assert_eq!(cut.exact_constant, -exact(tiny) * ratio(4, 1));
        for x in [ratio(1, 1), ratio(3, 2), ratio(2, 1)] {
            assert!(exact(cut.lower_bound) <= exact_relu(&x));
        }
    }

    #[test]
    fn malformed_box_multipliers_cannot_suppress_mandatory_portfolio() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-2.0], &[2.0], &[false]).unwrap();
        let certified = auxiliary(vec![1.0], vec![2.0]);
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));

        let wrong_shape = bound_relu_tail_triangle_dual_with_auxiliary_box_cut_unwired(
            &domain,
            &certified,
            &declared,
            &[],
            &[1.0],
            None,
            config(0),
        )
        .unwrap();
        assert_eq!(
            wrong_shape.status,
            ReluTailBoxCutStatus::InvalidBoxMultiplierShape {
                expected: 1,
                upper_got: 0,
                lower_got: 1,
            }
        );
        assert!(wrong_shape.box_cut.is_none());
        assert!(wrong_shape.lower_bound >= wrong_shape.original.lower_bound);

        let malformed = bound_relu_tail_triangle_dual_with_auxiliary_box_cut_unwired(
            &domain,
            &certified,
            &declared,
            &[0.0],
            &[f64::NAN],
            None,
            config(0),
        )
        .unwrap();
        assert_eq!(
            malformed.status,
            ReluTailBoxCutStatus::InvalidBoxMultiplierValue {
                upper: false,
                coordinate: 0,
            }
        );
        assert!(malformed.box_cut.is_none());
        assert!(malformed.lower_bound >= malformed.original.lower_bound);
    }

    #[test]
    fn optimized_box_multipliers_strictly_close_lower_and_upper_one_dimensional_gaps() {
        let lower_domain =
            ConstrainedZonotope64::from_certified_bounds(&[-2.0], &[2.0], &[false]).unwrap();
        let lower_auxiliary = auxiliary(vec![1.0], vec![2.0]);
        let positive = margin(vec![ratio(1, 1)], ratio(0, 1));
        let lower_prepared = prepare_relu_tail_triangle_dual_unwired(&lower_domain).unwrap();
        let lower = lower_prepared
            .bound_margin_with_optimized_auxiliary_box_cut_unwired(
                &lower_auxiliary,
                &positive,
                None,
                config(0),
                box_config(20, 40),
            )
            .unwrap();
        assert_eq!(
            lower.search_status,
            ReluTailBoxCutOptimizerStatus::Completed
        );
        assert_eq!(lower.selected, ReluTailBoxCutSelection::BoxCut);
        assert!(lower.lower_bound > 0.98, "{lower:#?}");
        assert!(exact(lower.lower_bound) <= ratio(1, 1));
        assert_eq!(lower.exact_replays, 2);
        assert_eq!(lower.restarts_completed, 2);

        let upper_domain =
            ConstrainedZonotope64::from_certified_bounds(&[-2.0], &[100.0], &[false]).unwrap();
        let upper_auxiliary = auxiliary(vec![1.0], vec![2.0]);
        let negative = margin(vec![ratio(-1, 1)], ratio(0, 1));
        let upper_prepared = prepare_relu_tail_triangle_dual_unwired(&upper_domain).unwrap();
        let upper = upper_prepared
            .bound_margin_with_optimized_auxiliary_box_cut_unwired(
                &upper_auxiliary,
                &negative,
                None,
                config(0),
                box_config(20, 40),
            )
            .unwrap();
        assert_eq!(upper.selected, ReluTailBoxCutSelection::BoxCut);
        assert!(upper.lower_bound > -3.0, "{upper:#?}");
        assert!(exact(upper.lower_bound) <= ratio(-2, 1));
    }

    #[test]
    fn box_objective_supergradient_matches_finite_difference_and_zero_sign_convention() {
        let domain = ConstrainedZonotope64::try_new(
            vec![0.5, -0.25],
            vec![vec![(0, 1.0), (1, 2.0)]],
            Array2::zeros((0, 1)),
            Vec::new(),
            vec![0.1, 0.2],
        )
        .unwrap();
        let variables = vec![
            BoxSearchVariable {
                coordinate: 0,
                kind: BoxVariableKind::Upper,
                endpoint: 0.2,
            },
            BoxSearchVariable {
                coordinate: 1,
                kind: BoxVariableKind::Lower,
                endpoint: -0.1,
            },
        ];
        let line = [0.7, -0.4];
        let point = [0.3, 0.2];
        let mut direction = vec![0.0; 2];
        let mut witness = vec![0.0; 2];
        let mut deadline = CandidateDeadline::new(Duration::from_secs(1), || Duration::ZERO);
        let objective = approximate_box_objective_and_witness(
            &domain,
            &line,
            &variables,
            &point,
            &mut direction,
            &mut witness,
            0,
            &mut deadline,
        )
        .unwrap();
        assert!(objective.is_finite());
        let analytical = [witness[0] - 0.2, -0.1 - witness[1]];
        let epsilon = 1e-6;
        for slot in 0..2 {
            let mut plus = point;
            let mut minus = point;
            plus[slot] += epsilon;
            minus[slot] -= epsilon;
            let score = |candidate: &[f64]| {
                let mut direction = vec![0.0; 2];
                let mut witness = vec![0.0; 2];
                let mut deadline =
                    CandidateDeadline::new(Duration::from_secs(1), || Duration::ZERO);
                approximate_box_objective_and_witness(
                    &domain,
                    &line,
                    &variables,
                    candidate,
                    &mut direction,
                    &mut witness,
                    0,
                    &mut deadline,
                )
                .unwrap()
            };
            let numerical = (score(&plus) - score(&minus)) / (2.0 * epsilon);
            assert!((numerical - analytical[slot]).abs() < 1e-7);
        }

        let kink_domain = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![vec![(0, 1.0)]],
            Array2::zeros((0, 1)),
            Vec::new(),
            vec![0.25],
        )
        .unwrap();
        let kink_variables = [BoxSearchVariable {
            coordinate: 0,
            kind: BoxVariableKind::Upper,
            endpoint: 0.5,
        }];
        let mut direction = vec![9.0];
        let mut witness = vec![9.0];
        let mut deadline = CandidateDeadline::new(Duration::from_secs(1), || Duration::ZERO);
        let kink = approximate_box_objective_and_witness(
            &kink_domain,
            &[0.0],
            &kink_variables,
            &[0.0],
            &mut direction,
            &mut witness,
            0,
            &mut deadline,
        )
        .unwrap();
        assert_eq!(kink, 0.0);
        assert_eq!(direction, vec![0.0]);
        // sign(p)=0 and sign(G^T p)=0 leave h=center exactly.
        assert_eq!(witness, vec![0.0]);
        assert_eq!(witness[0] - kink_variables[0].endpoint, -0.5);
    }

    #[test]
    fn optimized_box_cut_is_sound_on_random_coupled_generator_segments() {
        let mut state = 0x4d32_5eed_9e37_79b9_u64;
        for case in 0..48 {
            let dimension = 1 + (case % 4);
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let mut centers = Vec::with_capacity(dimension);
            let mut entries = Vec::with_capacity(dimension);
            let mut coefficients = Vec::with_capacity(dimension);
            for coordinate in 0..dimension {
                let center_numerator = i64::try_from(next() % 7).unwrap() - 3;
                let mut generator_numerator = i64::try_from(next() % 7).unwrap() - 3;
                if generator_numerator == 0 {
                    generator_numerator = 1;
                }
                let mut coefficient_numerator = i64::try_from(next() % 7).unwrap() - 3;
                if coefficient_numerator == 0 {
                    coefficient_numerator = -1;
                }
                centers.push(center_numerator as f64 / 4.0);
                entries.push((coordinate, generator_numerator as f64 / 4.0));
                coefficients.push(ratio(coefficient_numerator, 2));
            }
            let domain = ConstrainedZonotope64::try_new(
                centers.clone(),
                vec![entries.clone()],
                Array2::zeros((0, 1)),
                Vec::new(),
                vec![0.0; dimension],
            )
            .unwrap();
            let alpha_lower = -0.5_f64;
            let alpha_upper = 0.75_f64;
            let mut lower = Vec::with_capacity(dimension);
            let mut upper = Vec::with_capacity(dimension);
            for (coordinate, &(_, generator)) in entries.iter().enumerate() {
                let left = centers[coordinate] + generator * alpha_lower;
                let right = centers[coordinate] + generator * alpha_upper;
                lower.push(left.min(right));
                upper.push(left.max(right));
            }
            let certified = auxiliary(lower, upper);
            let bias = ratio(i64::try_from(next() % 5).unwrap() - 2, 4);
            let declared = margin(coefficients.clone(), bias.clone());
            let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
            let result = prepared
                .bound_margin_with_optimized_auxiliary_box_cut_unwired(
                    &certified,
                    &declared,
                    None,
                    config(0),
                    box_config(4, 12),
                )
                .unwrap();

            let mut alphas = vec![ratio(-1, 2), ratio(3, 4)];
            for (coordinate, &(_, generator)) in entries.iter().enumerate() {
                let center = exact(centers[coordinate]);
                let generator = exact(generator);
                let crossing = -center / generator;
                if crossing >= ratio(-1, 2) && crossing <= ratio(3, 4) {
                    alphas.push(crossing);
                }
            }
            let exact_minimum = alphas
                .iter()
                .map(|alpha| {
                    centers.iter().zip(&entries).zip(&coefficients).fold(
                        bias.clone(),
                        |sum, ((&center, &(_, generator)), coefficient)| {
                            let value = exact(center) + exact(generator) * alpha;
                            sum + coefficient * exact_relu(&value)
                        },
                    )
                })
                .min()
                .unwrap();
            assert!(
                exact(result.lower_bound) <= exact_minimum,
                "case {case}: {} > {exact_minimum}",
                result.lower_bound
            );
        }
    }

    #[test]
    fn prepared_exact_box_replay_matches_direct_m22_including_rounding_collision() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-2.0], &[3.0], &[false]).unwrap();
        let certified = auxiliary(vec![1.0], vec![2.0]);
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        let auxiliary_result = prepared
            .bound_margin_with_auxiliary_bounds_unwired(&certified, &declared, None, config(0))
            .unwrap();
        let variables = vec![
            BoxSearchVariable {
                coordinate: 0,
                kind: BoxVariableKind::Lower,
                endpoint: 1.0,
            },
            BoxSearchVariable {
                coordinate: 0,
                kind: BoxVariableKind::Upper,
                endpoint: 2.0,
            },
        ];
        let tiny = 2.0_f64.powi(-54);
        let (upper, lower) = expand_box_candidate(&variables, &[0.0, tiny], 1).unwrap();
        let cached = build_auxiliary_box_cut_certificate_with_original_hull(
            &domain,
            &certified,
            &auxiliary_result,
            &upper,
            &lower,
            None,
            &prepared.exact_coordinate_bounds,
        )
        .unwrap();
        let direct = bound_relu_tail_triangle_dual_with_auxiliary_box_cut_unwired(
            &domain,
            &certified,
            &declared,
            &upper,
            &lower,
            None,
            config(0),
        )
        .unwrap();
        assert_eq!(&cached, direct.box_cut.as_ref().unwrap());
        assert_eq!(cached.replay_direction, vec![1.0]);
        assert_eq!(cached.exact_constant, -exact(tiny) * ratio(4, 1));

        let optimized = prepared
            .bound_margin_with_optimized_auxiliary_box_cut_unwired(
                &certified,
                &declared,
                None,
                config(0),
                box_config(4, 12),
            )
            .unwrap();
        let optimized_cut = optimized.portfolio.box_cut.as_ref().unwrap();
        let replayed = bound_relu_tail_triangle_dual_with_auxiliary_box_cut_unwired(
            &domain,
            &certified,
            &declared,
            &optimized_cut.upper_box_multipliers,
            &optimized_cut.lower_box_multipliers,
            None,
            config(0),
        )
        .unwrap();
        assert_eq!(optimized_cut, replayed.box_cut.as_ref().unwrap());
    }

    #[test]
    fn optimizer_fallbacks_retain_m17_m20_and_deadlines_keep_bounded_best() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-2.0], &[2.0], &[false]).unwrap();
        let certified = auxiliary(vec![1.0], vec![2.0]);
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();

        let zero = prepared
            .bound_margin_with_optimized_auxiliary_box_cut_unwired(
                &certified,
                &declared,
                None,
                config(0),
                box_config(0, 0),
            )
            .unwrap();
        assert_eq!(
            zero.search_status,
            ReluTailBoxCutOptimizerStatus::SearchDisabled
        );
        assert!(zero.portfolio.auxiliary.is_some());
        assert!(zero.portfolio.box_cut.is_none());
        assert_eq!(zero.exact_replays, 0);

        let mut invalid_config = box_config(4, 4);
        invalid_config.multiplier_cap = f64::NAN;
        let invalid = prepared
            .bound_margin_with_optimized_auxiliary_box_cut_unwired(
                &certified,
                &declared,
                None,
                config(0),
                invalid_config,
            )
            .unwrap();
        assert_eq!(
            invalid.search_status,
            ReluTailBoxCutOptimizerStatus::InvalidConfig
        );
        assert!(invalid.portfolio.auxiliary.is_some());

        let mut resource_config = box_config(4, 4);
        resource_config.limits.max_box_variables = 0;
        let resource = prepared
            .bound_margin_with_optimized_auxiliary_box_cut_unwired(
                &certified,
                &declared,
                None,
                config(0),
                resource_config,
            )
            .unwrap();
        assert_eq!(
            resource.search_status,
            ReluTailBoxCutOptimizerStatus::ResourceFallback
        );
        assert!(resource.portfolio.auxiliary.is_some());

        let mut overflowing_update_config = box_config(1, 0);
        overflowing_update_config.schedules[0].learning_rate = f64::MAX;
        let overflowing_update = prepared
            .bound_margin_with_optimized_auxiliary_box_cut_unwired(
                &certified,
                &declared,
                None,
                config(0),
                overflowing_update_config,
            )
            .unwrap();
        assert_eq!(
            overflowing_update.search_status,
            ReluTailBoxCutOptimizerStatus::NonFiniteCandidate
        );
        assert_eq!(
            overflowing_update.lower_bound.to_bits(),
            overflowing_update.portfolio.original.lower_bound.to_bits()
        );
        assert_eq!(
            overflowing_update.selected,
            ReluTailBoxCutSelection::Original
        );
        assert_eq!(overflowing_update.iterations_completed, 0);
        assert_eq!(overflowing_update.candidates_scored, 1);
        assert!(overflowing_update.exact_replays <= 1);

        BOX_OPTIMIZER_FAIL_NEXT_ALLOCATION.with(|fail| fail.set(true));
        let allocation = prepared
            .bound_margin_with_optimized_auxiliary_box_cut_unwired(
                &certified,
                &declared,
                None,
                config(0),
                box_config(0, 4),
            )
            .unwrap();
        assert_eq!(
            allocation.search_status,
            ReluTailBoxCutOptimizerStatus::AllocationFallback
        );
        assert!(allocation.portfolio.auxiliary.is_some());
        assert!(allocation.portfolio.box_cut.is_none());
        assert_eq!(allocation.selected, ReluTailBoxCutSelection::Original);
        assert_eq!(allocation.exact_replays, 0);

        BOX_OPTIMIZER_FAIL_NEXT_EXACT_REPLAY.with(|fail| fail.set(true));
        let exact_replay = prepared
            .bound_margin_with_optimized_auxiliary_box_cut_unwired(
                &certified,
                &declared,
                None,
                config(0),
                box_config(0, 4),
            )
            .unwrap();
        assert_eq!(
            exact_replay.search_status,
            ReluTailBoxCutOptimizerStatus::ExactReplayFallback
        );
        assert!(exact_replay.portfolio.auxiliary.is_some());
        assert!(exact_replay.portfolio.box_cut.is_none());
        assert_eq!(exact_replay.selected, ReluTailBoxCutSelection::Original);
        assert_eq!(exact_replay.exact_replays, 1);

        let clock_config = box_config(0, 4);
        let plan = ReluTailBoxCutOptimizerPlan::checked(
            &domain,
            prepared.generator_nonzeros,
            1,
            clock_config,
        )
        .unwrap();
        let preallocation = optimize_auxiliary_box_multipliers_with_clock(
            &domain,
            &prepared.exact_coordinate_bounds,
            &certified,
            &[1.0],
            clock_config,
            plan,
            || Duration::from_secs(1),
        );
        assert_eq!(
            preallocation.status,
            ReluTailBoxCutOptimizerStatus::Deadline
        );
        assert!(preallocation.candidates.is_empty());
        assert_eq!(preallocation.candidates_scored, 0);

        let clock_calls = std::cell::Cell::new(0_usize);
        let miditeration = optimize_auxiliary_box_multipliers_with_clock(
            &domain,
            &prepared.exact_coordinate_bounds,
            &certified,
            &[1.0],
            clock_config,
            plan,
            || {
                let call = clock_calls.get() + 1;
                clock_calls.set(call);
                if call >= 4 {
                    Duration::from_secs(1)
                } else {
                    Duration::ZERO
                }
            },
        );
        assert_eq!(miditeration.status, ReluTailBoxCutOptimizerStatus::Deadline);
        assert_eq!(miditeration.candidates.len(), 1);
        assert_eq!(miditeration.iterations_completed, 1);
        assert_eq!(miditeration.candidates_scored, 2);
        assert_eq!(miditeration.restarts_completed, 0);

        let variables = [BoxSearchVariable {
            coordinate: 0,
            kind: BoxVariableKind::Lower,
            endpoint: 1.0,
        }];
        let explosive_domain = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![vec![(0, 2.0)]],
            Array2::zeros((0, 1)),
            Vec::new(),
            vec![0.0],
        )
        .unwrap();
        let mut direction = vec![0.0];
        let mut witness = vec![0.0];
        let mut deadline = CandidateDeadline::new(Duration::from_secs(1), || Duration::ZERO);
        assert!(matches!(
            approximate_box_objective_and_witness(
                &explosive_domain,
                &[f64::MAX],
                &variables,
                &[0.0],
                &mut direction,
                &mut witness,
                0,
                &mut deadline,
            ),
            Err(CandidateFailure::NonFinite(0))
        ));
    }

    #[test]
    fn optimized_portfolio_ties_and_errors_keep_earlier_certificates() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
        let nonrestrictive = auxiliary(vec![-1.0], vec![1.0]);
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        let tied = prepared
            .bound_margin_with_optimized_auxiliary_box_cut_unwired(
                &nonrestrictive,
                &declared,
                None,
                config(0),
                box_config(4, 4),
            )
            .unwrap();
        assert_eq!(
            tied.search_status,
            ReluTailBoxCutOptimizerStatus::NoTighterAuxiliaryBox
        );
        assert_eq!(tied.selected, ReluTailBoxCutSelection::Original);
        assert_eq!(
            tied.lower_bound.to_bits(),
            tied.portfolio.original.lower_bound.to_bits()
        );

        let wrong_shape = auxiliary(Vec::new(), Vec::new());
        let retained = prepared
            .bound_margin_with_optimized_auxiliary_box_cut_unwired(
                &wrong_shape,
                &declared,
                None,
                config(0),
                box_config(4, 4),
            )
            .unwrap();
        assert_eq!(
            retained.search_status,
            ReluTailBoxCutOptimizerStatus::AuxiliaryFallback
        );
        assert_eq!(retained.selected, ReluTailBoxCutSelection::Original);
        assert!(retained.portfolio.auxiliary.is_none());
    }

    #[test]
    fn repeated_prepared_m17_m20_m24_uses_one_exact_hull_scan() {
        let domain = ConstrainedZonotope64::try_new(
            vec![0.0, 0.5],
            vec![vec![(0, 1.0), (1, -0.5)]],
            Array2::zeros((0, 1)),
            Vec::new(),
            vec![0.0, 0.25],
        )
        .unwrap();
        let certified = auxiliary(vec![-0.5, 0.0], vec![0.75, 0.75]);
        let margins = [
            margin(vec![ratio(1, 1), ratio(-1, 2)], ratio(0, 1)),
            margin(vec![ratio(-1, 1), ratio(3, 2)], ratio(1, 7)),
        ];
        EXACT_COORDINATE_HULL_PASSES.with(|passes| passes.set(0));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        for declared in &margins {
            let result = prepared
                .bound_margin_with_optimized_auxiliary_box_cut_unwired(
                    &certified,
                    declared,
                    None,
                    config(0),
                    box_config(2, 4),
                )
                .unwrap();
            assert!(result.lower_bound >= result.portfolio.original.lower_bound);
            assert!(result.portfolio.auxiliary.is_some());
        }
        EXACT_COORDINATE_HULL_PASSES.with(|passes| assert_eq!(passes.get(), 1));
    }

    #[test]
    fn optimized_box_cut_replays_supplied_predicates_only_after_zero_replay() {
        let domain = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![vec![(0, 1.0)]],
            array![[1.0]],
            vec![0.5],
            vec![0.0],
        )
        .unwrap();
        let certified = auxiliary(vec![-1.0], vec![0.5]);
        let declared = margin(vec![ratio(-1, 1)], ratio(0, 1));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        let search_config = box_config(1, 0);
        let zero_predicate = prepared
            .bound_margin_with_optimized_auxiliary_box_cut_unwired(
                &certified,
                &declared,
                None,
                config(0),
                search_config,
            )
            .unwrap();
        let cut = zero_predicate.portfolio.box_cut.as_ref().unwrap();
        assert!(!cut.supplied_predicate_multipliers_used);
        let cancelling = -cut.replay_direction[0];
        assert!(cancelling > 0.0);

        let supplied = prepared
            .bound_margin_with_optimized_auxiliary_box_cut_unwired(
                &certified,
                &declared,
                Some(&[cancelling]),
                config(0),
                search_config,
            )
            .unwrap();
        let supplied_cut = supplied.portfolio.box_cut.as_ref().unwrap();
        assert_eq!(
            supplied_cut.zero_predicate_lower_bound.to_bits(),
            cut.zero_predicate_lower_bound.to_bits()
        );
        assert!(supplied_cut.supplied_predicate_multipliers_used);
        assert!(supplied_cut.lower_bound > supplied_cut.zero_predicate_lower_bound);
    }

    #[test]
    fn negative_unstable_chord_and_rounding_correction_are_exact_minorants() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-2.0], &[3.0], &[false]).unwrap();
        let coefficient = ratio(-5, 3);
        let declared = margin(vec![coefficient.clone()], ratio(0, 1));
        let plan = build_line_plan(&domain, &declared).unwrap();
        let direction = exact(plan.fixed_direction[0]);
        for x in [
            ratio(-2, 1),
            ratio(-1, 2),
            ratio(0, 1),
            ratio(1, 2),
            ratio(3, 1),
        ] {
            let represented = &direction * &x + &plan.exact_constant;
            let true_term = &coefficient * exact_relu(&x);
            assert!(represented <= true_term, "x={x}");
        }
    }

    #[test]
    fn direct_positive_slopes_obey_exact_interval_at_triangle_vertices() {
        let q = ratio(1, 3);
        let upper = floor_finite(&q, 0, "test upper").unwrap();
        for slope in [0.0, upper, upper / 2.0] {
            assert!(valid_direct_slope(slope, &q));
            let slope = exact(slope);
            for x in [ratio(-2, 1), ratio(0, 1), ratio(3, 1)] {
                assert!(&slope * &x <= &q * exact_relu(&x));
            }
        }
        assert!(!valid_direct_slope(upper.next_up(), &q));
        assert!(!valid_direct_slope(-f64::from_bits(1), &q));
    }

    #[test]
    fn non_dyadic_active_coefficient_is_corrected_before_replay() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[1.0], &[2.0], &[false]).unwrap();
        let declared = margin(vec![ratio(1, 3)], ratio(0, 1));
        let result =
            bound_relu_tail_triangle_dual_unwired(&domain, &declared, None, config(0)).unwrap();
        assert_eq!(result.status, ReluTailDualStatus::NoOptimizableSlopes);
        assert!(exact(result.lower_bound) <= ratio(1, 3));
        assert!(result.lower_bound > 0.3);
    }

    #[test]
    fn canonical_slope_closes_a_correlated_tail_gap() {
        // x0 = alpha is unstable; x1 = 2 + alpha is active.  For
        // ReLU(x0) - x1/2, k=1/2 cancels the shared generator exactly.
        let domain = ConstrainedZonotope64::try_new(
            vec![0.0, 2.0],
            vec![vec![(0, 1.0), (1, 1.0)]],
            Array2::zeros((0, 1)),
            Vec::new(),
            vec![0.0, 0.0],
        )
        .unwrap();
        let declared = margin(vec![ratio(1, 1), ratio(-1, 2)], ratio(0, 1));
        let disabled =
            bound_relu_tail_triangle_dual_unwired(&domain, &declared, None, config(0)).unwrap();
        let optimized =
            bound_relu_tail_triangle_dual_unwired(&domain, &declared, None, config(8)).unwrap();
        assert_eq!(optimized.status, ReluTailDualStatus::Completed);
        assert!(optimized.lower_bound > disabled.lower_bound + 0.4);
        assert!(exact(optimized.lower_bound) <= ratio(-1, 1));
        assert!(optimized.lower_bound > -1.000_001);
        assert_eq!(optimized.candidates_replayed, 4);
        // S=1, V=2, G=2, A=1, I=8:
        // startup=4S+3V+G+A=13; iteration=3S+2V+3G+2A=15.
        assert_eq!(optimized.plan.unwrap().search_work, 133);
    }

    #[test]
    fn telemetry_attributes_ordered_zero_predicate_cpu_replays() {
        // The zero-slope and upper-endpoint candidates both leave a generator
        // residual of magnitude 1/2.  The canonical candidate cancels it, and
        // Adam retains that canonical optimum.  This makes every candidate's
        // independently replayed certificate attributable by value.
        let domain = ConstrainedZonotope64::try_new(
            vec![0.0, 2.0],
            vec![vec![(0, 1.0), (1, 1.0)]],
            Array2::zeros((0, 1)),
            Vec::new(),
            vec![0.0, 0.0],
        )
        .unwrap();
        let declared = margin(vec![ratio(1, 1), ratio(-1, 2)], ratio(0, 1));
        let line_plan = build_line_plan(&domain, &declared).unwrap();
        let mut upper_endpoint_direction = line_plan.fixed_direction.clone();
        let mut canonical_direction = line_plan.fixed_direction.clone();
        for variable in &line_plan.variables {
            upper_endpoint_direction[variable.coordinate] = variable.upper;
            canonical_direction[variable.coordinate] = variable.canonical;
        }
        let (optimized_direction, completed) = projected_adam_candidate(
            &domain,
            &line_plan.variables,
            canonical_direction.clone(),
            config(8),
        )
        .unwrap();
        assert_eq!(completed, 8);
        let zero = Vec::<f64>::new();
        let replay_lower_bound = |direction: &[f64]| {
            let outward = domain.evaluate_dual(direction, &zero).unwrap();
            combine_exact_lower(outward.lower, &line_plan.exact_constant).unwrap()
        };
        let expected_zero_slope = replay_lower_bound(&line_plan.fixed_direction);
        let expected_upper_endpoint = replay_lower_bound(&upper_endpoint_direction);
        let expected_canonical = replay_lower_bound(&canonical_direction);
        let expected_optimized = replay_lower_bound(&optimized_direction);
        let result =
            bound_relu_tail_triangle_dual_unwired(&domain, &declared, None, config(8)).unwrap();
        let replay = result.zero_predicate_candidate_replays;

        assert_eq!(
            replay.zero_positive_slope_lower_bound.to_bits(),
            expected_zero_slope.to_bits()
        );
        assert_eq!(
            replay.upper_endpoint_lower_bound.map(f64::to_bits),
            Some(expected_upper_endpoint.to_bits())
        );
        assert_eq!(
            replay.canonical_lower_bound.map(f64::to_bits),
            Some(expected_canonical.to_bits())
        );
        assert_eq!(
            replay.optimized_lower_bound.map(f64::to_bits),
            Some(expected_optimized.to_bits())
        );
        assert!(replay.zero_positive_slope_lower_bound < replay.canonical_lower_bound.unwrap());
        assert!(replay.upper_endpoint_lower_bound < replay.canonical_lower_bound);
        assert!(replay.canonical_lower_bound <= replay.optimized_lower_bound);
        assert_eq!(
            result.zero_multiplier_lower_bound.to_bits(),
            expected_optimized.to_bits()
        );
        assert_eq!(result.lower_bound.to_bits(), expected_optimized.to_bits());
        assert_eq!(result.candidates_replayed, 4);
    }

    #[test]
    fn zero_iterations_reports_only_zero_positive_slope_replay() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let result =
            bound_relu_tail_triangle_dual_unwired(&domain, &declared, None, config(0)).unwrap();
        let replay = result.zero_predicate_candidate_replays;

        assert_eq!(result.status, ReluTailDualStatus::SearchDisabled);
        assert_eq!(result.candidates_replayed, 1);
        assert_eq!(replay.zero_positive_slope_lower_bound, -0.0);
        assert_eq!(replay.upper_endpoint_lower_bound, None);
        assert_eq!(replay.canonical_lower_bound, None);
        assert_eq!(replay.optimized_lower_bound, None);
    }

    #[test]
    fn deterministic_clock_rejects_before_candidate_startup() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let line_plan = build_line_plan(&domain, &declared).unwrap();
        let mut canonical = line_plan.fixed_direction.clone();
        for variable in &line_plan.variables {
            canonical[variable.coordinate] = variable.canonical;
        }
        let deadline = config(8).wall_time;
        let result = projected_adam_candidate_with_clock(
            &domain,
            &line_plan.variables,
            canonical,
            config(8),
            || deadline,
        );
        assert!(matches!(result, Err(CandidateFailure::Deadline(0))));
    }

    #[test]
    fn deadline_guard_checks_after_each_bounded_chunk() {
        use std::cell::Cell;

        let clock_calls = Cell::new(0_usize);
        let wall_time = Duration::from_secs(1);
        let mut deadline = CandidateDeadline::new(wall_time, || {
            let call = clock_calls.get() + 1;
            clock_calls.set(call);
            if call >= 2 {
                wall_time
            } else {
                Duration::ZERO
            }
        });
        deadline.checkpoint(0).unwrap();
        for _ in 0..RELU_TAIL_DUAL_DEADLINE_CHECK_STRIDE {
            deadline.visit(0).unwrap();
        }
        assert!(matches!(
            deadline.visit(0),
            Err(CandidateFailure::Deadline(0))
        ));
        assert_eq!(clock_calls.get(), 2);
    }

    #[test]
    fn replay_storage_fallback_is_explicit_unattributed_and_retains_baseline() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let line_plan = build_line_plan(&domain, &declared).unwrap();
        let baseline =
            bound_relu_tail_triangle_dual_unwired(&domain, &declared, None, config(0)).unwrap();
        let mut accepted = AcceptedCandidate {
            lower: baseline.lower_bound,
            zero_lower: baseline.zero_multiplier_lower_bound,
            direction: baseline.direction,
            multipliers: baseline.multipliers,
            supplied_used: baseline.supplied_multipliers_used,
        };
        let retained = accepted.clone();
        let mut endpoint = line_plan.fixed_direction.clone();
        for variable in &line_plan.variables {
            endpoint[variable.coordinate] = variable.upper;
        }
        let mut fail_clone = |_: &[f64]| None::<Vec<f64>>;
        let outcome = replay_direction_with_cloner(
            &domain,
            &line_plan.exact_constant,
            &line_plan.variables,
            endpoint,
            &[],
            None,
            &mut accepted,
            &mut fail_clone,
        );
        assert_eq!(outcome, CandidateReplayOutcome::AllocationFallback);
        assert!(!matches!(outcome, CandidateReplayOutcome::Replayed { .. }));
        assert_eq!(accepted.lower, retained.lower);
        assert_eq!(accepted.direction, retained.direction);
        assert_eq!(accepted.multipliers, retained.multipliers);
    }

    #[test]
    fn supplied_multiplier_is_used_only_after_zero_replay_improves() {
        // x=alpha with alpha<=0.  The internal unconstrained box remains
        // [-1,1], so the negative-coefficient chord is fixed and lambda=1/2
        // cancels its projected generator.
        let domain = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![vec![(0, 1.0)]],
            array![[1.0]],
            vec![0.0],
            vec![0.0],
        )
        .unwrap();
        let declared = margin(vec![ratio(-1, 1)], ratio(0, 1));
        let result =
            bound_relu_tail_triangle_dual_unwired(&domain, &declared, Some(&[0.5]), config(0))
                .unwrap();
        assert!(result.supplied_multipliers_used);
        assert!(result.lower_bound > result.zero_multiplier_lower_bound);
        assert!(exact(result.lower_bound) <= ratio(-1, 2));

        let malformed =
            bound_relu_tail_triangle_dual_unwired(&domain, &declared, Some(&[-0.5]), config(0))
                .unwrap();
        assert!(!malformed.supplied_multipliers_used);
        assert_eq!(malformed.lower_bound, malformed.zero_multiplier_lower_bound);
    }

    #[test]
    fn final_floor_handles_signed_zero_and_subnormal_values() {
        let denominator = BigInt::from(1_u8) << 1_075_usize;
        let positive_half_subnormal = BigRational::new(BigInt::from(1), denominator.clone());
        let negative_half_subnormal = BigRational::new(BigInt::from(-1), denominator);
        let positive = floor_finite(&positive_half_subnormal, 0, "test").unwrap();
        let negative = floor_finite(&negative_half_subnormal, 0, "test").unwrap();
        let zero = floor_finite(&BigRational::zero(), 0, "test").unwrap();
        assert_eq!(positive.to_bits(), 0);
        assert_eq!(negative.to_bits(), (1_u64 << 63) | 1);
        assert_eq!(zero.to_bits(), 0);
    }

    #[test]
    fn empty_domain_preserves_exact_tiny_bias_outward() {
        let domain = ConstrainedZonotope64::try_new(
            Vec::new(),
            Vec::new(),
            Array2::zeros((0, 0)),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        let denominator = BigInt::from(1_u8) << 1_075_usize;
        let declared = margin(Vec::new(), BigRational::new(BigInt::from(-1), denominator));
        let result =
            bound_relu_tail_triangle_dual_unwired(&domain, &declared, None, config(0)).unwrap();
        assert_eq!(result.lower_bound.to_bits(), (1_u64 << 63) | 1);
    }

    #[test]
    fn rational_and_search_resource_limits_fail_closed() {
        let oversized = BigRational::from_integer(
            BigInt::from(1_u8)
                << usize::try_from(RELU_TAIL_DUAL_HARD_MAX_INPUT_RATIONAL_BITS).unwrap(),
        );
        assert!(matches!(
            ExactReluTailMargin::try_new(vec![oversized], BigRational::zero()),
            Err(ReluTailDualError::RationalInputLimit { .. })
        ));

        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let mut rejected = config(4);
        rejected.limits.max_value_dim = 0;
        let result =
            bound_relu_tail_triangle_dual_unwired(&domain, &declared, None, rejected).unwrap();
        assert_eq!(result.status, ReluTailDualStatus::ResourceFallback);
        assert_eq!(result.candidates_replayed, 1);
        assert_eq!(
            result
                .zero_predicate_candidate_replays
                .upper_endpoint_lower_bound,
            None
        );
        assert_eq!(
            result
                .zero_predicate_candidate_replays
                .canonical_lower_bound,
            None
        );
        assert_eq!(
            result
                .zero_predicate_candidate_replays
                .optimized_lower_bound,
            None
        );
    }

    #[test]
    fn auxiliary_search_cap_and_deadline_retain_certified_baseline() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
        let certified = auxiliary(vec![-0.75], vec![0.75]);
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));

        let mut rejected = config(4);
        rejected.limits.max_value_dim = 0;
        let fallback = bound_relu_tail_triangle_dual_with_auxiliary_bounds_unwired(
            &domain, &certified, &declared, None, rejected,
        )
        .unwrap();
        assert_eq!(fallback.status, ReluTailDualStatus::ResourceFallback);
        assert_eq!(fallback.candidates_replayed, 1);
        assert!(exact(fallback.lower_bound) <= BigRational::zero());

        let mut deadline = config(64);
        deadline.wall_time = Duration::from_nanos(1);
        let timed_out = bound_relu_tail_triangle_dual_with_auxiliary_bounds_unwired(
            &domain, &certified, &declared, None, deadline,
        )
        .unwrap();
        assert_eq!(timed_out.status, ReluTailDualStatus::Deadline);
        assert_eq!(timed_out.candidates_replayed, 3);
        assert!(timed_out
            .zero_predicate_candidate_replays
            .upper_endpoint_lower_bound
            .is_some());
        assert!(timed_out
            .zero_predicate_candidate_replays
            .canonical_lower_bound
            .is_some());
        assert!(timed_out
            .zero_predicate_candidate_replays
            .optimized_lower_bound
            .is_none());
        assert!(exact(timed_out.lower_bound) <= BigRational::zero());
    }

    #[test]
    fn candidate_deadline_retains_certified_replays() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let mut deadline = config(64);
        deadline.wall_time = Duration::from_nanos(1);
        let result =
            bound_relu_tail_triangle_dual_unwired(&domain, &declared, None, deadline).unwrap();
        assert_eq!(result.status, ReluTailDualStatus::Deadline);
        assert_eq!(result.candidates_replayed, 3);
        assert!(result
            .zero_predicate_candidate_replays
            .upper_endpoint_lower_bound
            .is_some());
        assert!(result
            .zero_predicate_candidate_replays
            .canonical_lower_bound
            .is_some());
        assert_eq!(
            result
                .zero_predicate_candidate_replays
                .optimized_lower_bound,
            None
        );
        assert!(exact(result.lower_bound) <= BigRational::zero());
    }

    #[test]
    fn exact_positive_slope_smaller_than_subnormal_degrades_to_zero() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
        let tiny = BigRational::new(BigInt::from(1), BigInt::from(1_u8) << 2_000_usize);
        let declared = margin(vec![tiny], BigRational::zero());
        let result =
            bound_relu_tail_triangle_dual_unwired(&domain, &declared, None, config(2)).unwrap();
        assert_eq!(result.direction[0].to_bits(), 0);
        assert_eq!(result.lower_bound.to_bits(), 0);
    }

    proptest! {
        #[test]
        fn certified_bound_never_exceeds_exact_independent_box_corners(
            raw in prop::collection::vec((-3_i8..=3, -3_i8..=3, -4_i8..=4), 1..=4),
            bias in -3_i8..=3,
        ) {
            let lower: Vec<f64> = raw
                .iter()
                .map(|(left, right, _)| f64::from((*left).min(*right)))
                .collect();
            let upper: Vec<f64> = raw
                .iter()
                .map(|(left, right, _)| f64::from((*left).max(*right)))
                .collect();
            let coefficients: Vec<BigRational> = raw
                .iter()
                .map(|(_, _, coefficient)| BigRational::from_integer((*coefficient).into()))
                .collect();
            let domain = ConstrainedZonotope64::from_certified_bounds(
                &lower,
                &upper,
                &vec![false; raw.len()],
            )
            .unwrap();
            let declared = margin(
                coefficients.clone(),
                BigRational::from_integer(bias.into()),
            );
            let result = bound_relu_tail_triangle_dual_unwired(
                &domain,
                &declared,
                None,
                config(4),
            )
            .unwrap();

            let mut exact_minimum: Option<BigRational> = None;
            for mask in 0..(1_usize << raw.len()) {
                let mut value = BigRational::from_integer(bias.into());
                for coordinate in 0..raw.len() {
                    let x = if mask & (1 << coordinate) == 0 {
                        exact(lower[coordinate])
                    } else {
                        exact(upper[coordinate])
                    };
                    value += &coefficients[coordinate] * exact_relu(&x);
                }
                if exact_minimum.as_ref().is_none_or(|best| value < *best) {
                    exact_minimum = Some(value);
                }
            }
            prop_assert!(exact(result.lower_bound) <= exact_minimum.unwrap());
        }

        #[test]
        fn auxiliary_bound_never_exceeds_exact_intersection_box_corners(
            raw in prop::collection::vec(
                (-3_i8..=3, -3_i8..=3, -3_i8..=3, -3_i8..=3, -4_i8..=4),
                1..=4,
            ),
            bias in -3_i8..=3,
        ) {
            let domain_lower: Vec<f64> = raw
                .iter()
                .map(|(left, right, auxiliary_left, auxiliary_right, _)| {
                    f64::from(
                        (*left)
                            .min(*right)
                            .min(*auxiliary_left)
                            .min(*auxiliary_right),
                    )
                })
                .collect();
            let domain_upper: Vec<f64> = raw
                .iter()
                .map(|(left, right, auxiliary_left, auxiliary_right, _)| {
                    f64::from(
                        (*left)
                            .max(*right)
                            .max(*auxiliary_left)
                            .max(*auxiliary_right),
                    )
                })
                .collect();
            let auxiliary_lower: Vec<f64> = raw
                .iter()
                .map(|(_, _, left, right, _)| f64::from((*left).min(*right)))
                .collect();
            let auxiliary_upper: Vec<f64> = raw
                .iter()
                .map(|(_, _, left, right, _)| f64::from((*left).max(*right)))
                .collect();
            let coefficients: Vec<BigRational> = raw
                .iter()
                .map(|(_, _, _, _, coefficient)| {
                    BigRational::from_integer((*coefficient).into())
                })
                .collect();
            let domain = ConstrainedZonotope64::from_certified_bounds(
                &domain_lower,
                &domain_upper,
                &vec![false; raw.len()],
            )
            .unwrap();
            let certified = auxiliary(auxiliary_lower.clone(), auxiliary_upper.clone());
            let declared = margin(
                coefficients.clone(),
                BigRational::from_integer(bias.into()),
            );
            let result = bound_relu_tail_triangle_dual_with_auxiliary_bounds_unwired(
                &domain,
                &certified,
                &declared,
                None,
                config(4),
            )
            .unwrap();

            // The independently certified concrete set for this oracle is the
            // full intersection box.  Its separable piecewise-linear minimum
            // occurs at one of these exactly enumerated dyadic corners.
            let mut exact_minimum: Option<BigRational> = None;
            for mask in 0..(1_usize << raw.len()) {
                let mut value = BigRational::from_integer(bias.into());
                for coordinate in 0..raw.len() {
                    let x = if mask & (1 << coordinate) == 0 {
                        exact(auxiliary_lower[coordinate])
                    } else {
                        exact(auxiliary_upper[coordinate])
                    };
                    value += &coefficients[coordinate] * exact_relu(&x);
                }
                if exact_minimum.as_ref().is_none_or(|best| value < *best) {
                    exact_minimum = Some(value);
                }
            }
            prop_assert!(exact(result.lower_bound) <= exact_minimum.unwrap());
        }

        #[test]
        fn m22_box_cut_and_portfolio_never_exceed_strict_auxiliary_box_corners(
            raw in prop::collection::vec(
                (
                    1_i8..=8,
                    1_i8..=8,
                    1_i8..=8,
                    1_i8..=8,
                    1_i8..=8,
                    1_i8..=8,
                    1_i8..=8,
                    1_i8..=8,
                ),
                2..=4,
            ),
            bias_numerator in -8_i8..=8,
            bias_denominator in 1_i8..=8,
        ) {
            let mut domain_lower = Vec::with_capacity(raw.len());
            let mut domain_upper = Vec::with_capacity(raw.len());
            let mut auxiliary_lower = Vec::with_capacity(raw.len());
            let mut auxiliary_upper = Vec::with_capacity(raw.len());
            let mut coefficients = Vec::with_capacity(raw.len());
            let mut upper_multipliers = Vec::with_capacity(raw.len());
            let mut lower_multipliers = Vec::with_capacity(raw.len());

            for (
                coordinate,
                (
                    lower_units,
                    width_units,
                    negative_padding_units,
                    positive_padding_units,
                    coefficient_numerator,
                    coefficient_denominator,
                    upper_multiplier_units,
                    lower_multiplier_units,
                ),
            ) in raw.iter().enumerate()
            {
                let lower = f64::from(*lower_units) * 0.25;
                let upper = lower + f64::from(*width_units) * 0.25;
                auxiliary_lower.push(lower);
                auxiliary_upper.push(upper);
                domain_lower.push(-f64::from(*negative_padding_units) * 0.25);
                domain_upper.push(upper + f64::from(*positive_padding_units) * 0.25);

                // Alternation guarantees genuinely mixed signed rational
                // objectives in every generated example.
                let sign = if coordinate % 2 == 0 { 1_i64 } else { -1_i64 };
                coefficients.push(BigRational::new(
                    (sign * i64::from(*coefficient_numerator)).into(),
                    i64::from(*coefficient_denominator).into(),
                ));

                if coordinate == 0 {
                    // Both cuts are nonzero, but their exact difference is
                    // 2^-61.  Every generated positive active coefficient is
                    // at least 1/8, so adding that difference to its binary64
                    // line direction requires a nonzero rounding repair.
                    upper_multipliers.push(2.0_f64.powi(-60));
                    lower_multipliers.push(2.0_f64.powi(-61));
                } else {
                    upper_multipliers.push(f64::from(*upper_multiplier_units) * 0.25);
                    lower_multipliers.push(f64::from(*lower_multiplier_units) * 0.25);
                }
            }

            for coordinate in 0..raw.len() {
                prop_assert!(domain_lower[coordinate] < auxiliary_lower[coordinate]);
                prop_assert!(auxiliary_lower[coordinate] < auxiliary_upper[coordinate]);
                prop_assert!(auxiliary_upper[coordinate] < domain_upper[coordinate]);
                prop_assert!(upper_multipliers[coordinate] > 0.0);
                prop_assert!(lower_multipliers[coordinate] > 0.0);
            }
            prop_assert!(coefficients.iter().any(Signed::is_positive));
            prop_assert!(coefficients.iter().any(Signed::is_negative));

            let domain = ConstrainedZonotope64::from_certified_bounds(
                &domain_lower,
                &domain_upper,
                &vec![false; raw.len()],
            )
            .unwrap();
            let certified = auxiliary(auxiliary_lower.clone(), auxiliary_upper.clone());
            let bias = BigRational::new(
                i64::from(bias_numerator).into(),
                i64::from(bias_denominator).into(),
            );
            let declared = margin(coefficients.clone(), bias.clone());
            let result = bound_relu_tail_triangle_dual_with_auxiliary_box_cut_unwired(
                &domain,
                &certified,
                &declared,
                &upper_multipliers,
                &lower_multipliers,
                None,
                config(0),
            )
            .unwrap();
            prop_assert_eq!(result.status, ReluTailBoxCutStatus::Completed);
            let auxiliary_result = result.auxiliary.as_ref().unwrap();
            let cut = result.box_cut.as_ref().unwrap();

            let exact_cut_direction = exact(auxiliary_result.direction[0])
                + exact(upper_multipliers[0])
                - exact(lower_multipliers[0]);
            prop_assert_ne!(exact_cut_direction, exact(cut.replay_direction[0]));

            // The concrete oracle set is the entire strict auxiliary Box.
            // A separable piecewise-linear ReLU margin reaches its minimum at
            // one of these exactly enumerated dyadic corners.
            let mut exact_minimum: Option<BigRational> = None;
            for mask in 0..(1_usize << raw.len()) {
                let mut value = bias.clone();
                for coordinate in 0..raw.len() {
                    let x = if mask & (1 << coordinate) == 0 {
                        exact(auxiliary_lower[coordinate])
                    } else {
                        exact(auxiliary_upper[coordinate])
                    };
                    value += &coefficients[coordinate] * exact_relu(&x);
                }
                if exact_minimum.as_ref().is_none_or(|best| value < *best) {
                    exact_minimum = Some(value);
                }
            }
            let exact_minimum = exact_minimum.unwrap();
            prop_assert!(exact(cut.lower_bound) <= exact_minimum);
            prop_assert!(exact(result.lower_bound) <= exact_minimum);
            prop_assert!(result.lower_bound >= result.original.lower_bound);
        }

        #[test]
        fn m22_zero_box_cut_randomly_recovers_nonrestrictive_m17_bits(
            raw in prop::collection::vec(
                (-8_i8..=8, 1_i8..=8, 1_i8..=8, -8_i8..=8, 1_i8..=8),
                1..=4,
            ),
            bias_numerator in -8_i8..=8,
            bias_denominator in 1_i8..=8,
        ) {
            let domain_lower: Vec<f64> = raw
                .iter()
                .map(|(center, left_width, _, _, _)| {
                    f64::from(*center - *left_width) * 0.25
                })
                .collect();
            let domain_upper: Vec<f64> = raw
                .iter()
                .map(|(center, _, right_width, _, _)| {
                    f64::from(*center + *right_width) * 0.25
                })
                .collect();
            let coefficients: Vec<BigRational> = raw
                .iter()
                .map(|(_, _, _, numerator, denominator)| {
                    BigRational::new(
                        i64::from(*numerator).into(),
                        i64::from(*denominator).into(),
                    )
                })
                .collect();
            let domain = ConstrainedZonotope64::from_certified_bounds(
                &domain_lower,
                &domain_upper,
                &vec![false; raw.len()],
            )
            .unwrap();
            let nonrestrictive = auxiliary(
                vec![-f64::MAX; raw.len()],
                vec![f64::MAX; raw.len()],
            );
            let declared = margin(
                coefficients,
                BigRational::new(
                    i64::from(bias_numerator).into(),
                    i64::from(bias_denominator).into(),
                ),
            );
            let zero = vec![0.0; raw.len()];
            let result = bound_relu_tail_triangle_dual_with_auxiliary_box_cut_unwired(
                &domain,
                &nonrestrictive,
                &declared,
                &zero,
                &zero,
                None,
                config(0),
            )
            .unwrap();

            prop_assert_eq!(result.status, ReluTailBoxCutStatus::Completed);
            prop_assert_eq!(result.selected, ReluTailBoxCutSelection::Original);
            prop_assert_eq!(result.lower_bound.to_bits(), result.original.lower_bound.to_bits());
            prop_assert_eq!(result.auxiliary.as_ref().unwrap(), &result.original);
            let cut = result.box_cut.as_ref().unwrap();
            prop_assert_eq!(cut.lower_bound.to_bits(), result.original.lower_bound.to_bits());
            prop_assert_eq!(
                cut.zero_predicate_lower_bound.to_bits(),
                result.original.zero_multiplier_lower_bound.to_bits()
            );
            prop_assert_eq!(
                cut.replay_direction.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                result.original.direction.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
            );
            prop_assert_eq!(&cut.exact_constant, &result.original.exact_constant);
        }
    }
}
