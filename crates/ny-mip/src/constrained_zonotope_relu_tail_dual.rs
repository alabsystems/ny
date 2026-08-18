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
//! The prepared M17/M20 portfolio is consumed by the CLI's explicitly enabled
//! cGAN input-leaf route. No default preset enables that route; the remaining
//! experimental entry points document their wiring status individually.

use std::time::{Duration, Instant};

use ndarray::ArrayView4;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};

use crate::constrained_zonotope_batch_norm::{
    batch_norm_affine_certificate_peak_live_bytes, certify_batch_norm_affine_surrogate_impl,
    ConstrainedZonotopeBatchNormAffineCertificateLimits, ConstrainedZonotopeBatchNormBudgetError,
    ConstrainedZonotopeBatchNormError, ConstrainedZonotopeBatchNormSpec,
};
use crate::constrained_zonotope_call_budget::{
    ConstrainedZonotopeCallAttempt, ConstrainedZonotopeCallBudget,
    ConstrainedZonotopeCallBudgetError, ConstrainedZonotopeCallGate,
    ConstrainedZonotopeCallOutcome, ConstrainedZonotopeCallTracker,
    ConstrainedZonotopePeakLiveBytes, InertConstrainedZonotopeCallGate,
};
use crate::constrained_zonotope_conv2d::{
    input_coordinate as conv2d_input_coordinate, output_dimension as conv2d_output_dimension,
    ConstrainedZonotopeConv2dError, ConstrainedZonotopeConv2dSpec,
};
use crate::constrained_zonotope_dual::{
    evaluate_constrained_zonotope64_dual_with_call_gate, ConstrainedZonotopeDualBudgetError,
    DUAL_SHAPE_ERROR_LIVE_BYTES,
};
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

// One retained exact rational can reach the 32,768-bit accepted ceiling in
// both numerator and denominator, while the checked operation constructing it
// can transiently carry roughly twice that width before rejection/reduction.
// This charge covers both integer payloads, their containers, and wide
// allocator/carry slack.
const RELU_TAIL_RATIONAL_LIVE_BYTES: usize = 64 * 1_024;
// A small fixed pool covers transient exact scalars held while one coordinate
// is converted, corrected, combined with a replay, or rounded for publication.
const RELU_TAIL_TRANSIENT_RATIONAL_SLOTS: usize = 16;
// M24 simultaneously retains its source/cut directions, two multiplier
// values, endpoint values, rounding repair, and accumulated exact constant.
// Keep this separate from the M17 line-builder pool so its peak model can be
// tightened independently without weakening either authority path.
const RELU_TAIL_BOX_CUT_TRANSIENT_RATIONAL_SLOTS: usize = 24;

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

/// Private provenance seal for an exact margin constructed by the
/// transactional Conv2d/BatchNorm pullback.
///
/// Caller-declared margins cannot acquire this wrapper: their only public
/// constructor remains [`ExactReluTailMargin::try_new`], with the stricter
/// input-rational ceiling.  The wrapper lets the downstream transaction retain
/// exact terms that grew past that input ceiling, but not past the immutable
/// intermediate or aggregate ceilings.
struct InternallyPulledReluTailMargin {
    margin: ExactReluTailMargin,
}

impl InternallyPulledReluTailMargin {
    fn as_exact_margin(&self) -> &ExactReluTailMargin {
        &self.margin
    }
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

/// Caller-tightenable geometry and post-certificate construction limits.
///
/// There is intentionally no `Default`: an experimental caller must price the
/// exact Conv2d transpose and channel-error margin construction explicitly.
/// Raw BatchNorm certification and both M17 calls retain their own checked
/// internal resource accounting and are not counted by the product field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReluTailConvBatchNormPullbackLimits {
    /// Maximum flat value count on the upstream side of the convolution.
    pub max_input_value_count: usize,
    /// Maximum flat value count on the downstream side of the convolution.
    pub max_output_value_count: usize,
    /// Maximum authored convolution weight elements inspected for finiteness.
    pub max_weight_elements: usize,
    /// Maximum output/kernel visits, including padding and structural zeros.
    pub max_kernel_visits: usize,
    /// Maximum exact products in post-certificate pulled-margin construction.
    pub max_pulled_margin_construction_exact_products: usize,
}

/// Checked geometry and work accounting for one transactional pullback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReluTailConvBatchNormPullbackPlan {
    /// Upstream pre-ReLU shape, also used by BatchNorm after that ReLU.
    pub input_shape: [usize; 3],
    /// Downstream convolution output shape `[channels, height, width]`.
    pub output_shape: [usize; 3],
    /// Weight shape `[output_channels, input_channels_per_group, height, width]`.
    pub weight_shape: [usize; 4],
    /// Authored convolution weight elements validated before exact work.
    pub weight_elements: usize,
    /// Output/kernel visits, including padding and structural zeros.
    pub kernel_visits: usize,
    /// Conservative exact-product bound for post-certificate margin construction.
    ///
    /// This excludes raw BatchNorm certification and both M17 calls, which
    /// perform separate checked preflights and immutable-cap validation.
    pub pulled_margin_construction_exact_product_bound: usize,
}

/// Two M17 certificates linked by one internally constructed exact pullback.
///
/// The transaction accepts no externally supplied downstream line and never
/// publishes the temporary pulled-back [`ExactReluTailMargin`].  The ordinary
/// downstream result fields remain available for certificate inspection.
#[derive(Clone, Debug, PartialEq)]
pub struct ReluTailConvBatchNormPullbackResult {
    /// M17 certificate for the caller-declared final margin.
    pub downstream: ReluTailDualResult,
    /// M17 certificate after exact Conv2d transpose and certified BatchNorm
    /// surrogate-error pullback.
    pub upstream: ReluTailDualResult,
    /// Checked tensor geometry and post-certificate construction work count.
    pub plan: ReluTailConvBatchNormPullbackPlan,
}

/// Transactional downstream M17 plus retained upstream M17/M20 portfolio.
///
/// The exact Conv2d/BatchNorm pullback is constructed once.  Its ordinary
/// upstream M17 replay is mandatory; the auxiliary-bound M20 replay is
/// optional and can never suppress that certificate.  No Box-cut lane is
/// present in this transaction.
#[derive(Clone, Debug, PartialEq)]
pub struct ReluTailConvBatchNormPullbackM17M20Result {
    /// M17 certificate for the caller-declared final margin.
    pub downstream: ReluTailDualResult,
    /// Strict ordered maximum of mandatory upstream M17 and optional M20.
    pub upstream: ReluTailBoxCutDualResult,
    /// Shared-firewall refusal that caused optional M20 to fall back.
    ///
    /// Invalid or disjoint auxiliary geometry is intentionally represented
    /// only by [`ReluTailBoxCutStatus::AuxiliaryFallback`]; this field retains
    /// allocation-accounting overflow, peak refusal, or deadline exhaustion.
    pub optional_budget_error: Option<ConstrainedZonotopeCallBudgetError>,
    /// Checked tensor geometry and post-certificate construction work count.
    pub plan: ReluTailConvBatchNormPullbackPlan,
}

/// Invalid geometry, BatchNorm certificate, or mandatory M17 proof work.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ReluTailConvBatchNormPullbackError {
    /// Convolution geometry, values, or caller-selected work limits failed.
    #[error(transparent)]
    Conv2d(#[from] ConstrainedZonotopeConv2dError),
    /// Raw BatchNorm data or its declared affine surrogate failed certification.
    #[error(transparent)]
    BatchNorm(#[from] ConstrainedZonotopeBatchNormError),
    /// Exact pullback arithmetic or either mandatory M17 replay failed.
    #[error(transparent)]
    ReluTail(#[from] ReluTailDualError),
}

/// Transactional pullback proof failure or shared-firewall refusal.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ReluTailConvBatchNormPullbackBudgetError {
    /// Geometry, exact arithmetic, or mandatory proof work was invalid.
    #[error(transparent)]
    Transform(#[from] ReluTailConvBatchNormPullbackError),
    /// The shared absolute deadline or peak-live-byte ceiling refused work.
    #[error(transparent)]
    Budget(#[from] ConstrainedZonotopeCallBudgetError),
}

impl From<ConstrainedZonotopeConv2dError> for ReluTailConvBatchNormPullbackBudgetError {
    fn from(error: ConstrainedZonotopeConv2dError) -> Self {
        ReluTailConvBatchNormPullbackError::Conv2d(error).into()
    }
}

impl From<ConstrainedZonotopeBatchNormError> for ReluTailConvBatchNormPullbackBudgetError {
    fn from(error: ConstrainedZonotopeBatchNormError) -> Self {
        ReluTailConvBatchNormPullbackError::BatchNorm(error).into()
    }
}

impl From<ReluTailDualError> for ReluTailConvBatchNormPullbackBudgetError {
    fn from(error: ReluTailDualError) -> Self {
        ReluTailConvBatchNormPullbackError::ReluTail(error).into()
    }
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
/// This experimental type is consumed by the explicitly enabled cGAN
/// input-leaf route. No default preset enables that route.
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
    conservative_live_bytes: usize,
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

    /// Conservative logical bytes retained by this prepared geometry.
    ///
    /// The count covers the prepared-value header, exact coordinate-pair
    /// storage, and a deliberately wide payload allowance for every retained
    /// rational.  It excludes the borrowed domain.  A caller retaining this
    /// value across a budgeted margin call must include both this count and its
    /// own accounting for the domain in that call's baseline.
    #[must_use]
    pub const fn conservative_live_bytes(&self) -> usize {
        self.conservative_live_bytes
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

    /// Bound one exact M17 margin behind the shared synchronous call firewall.
    ///
    /// The prepared hull remains caller-owned throughout this call.  Therefore
    /// `budget.baseline_live_bytes()` must include
    /// [`Self::conservative_live_bytes`], the borrowed domain, `margin`, any
    /// supplied multipliers, and every other buffer sharing the same ceiling.
    /// Exact line construction and every CPU outward replay use the same
    /// budget tracker as admission.
    ///
    /// # Errors
    ///
    /// Returns [`ReluTailDualBudgetError::Bound`] for the same mandatory proof
    /// failures as [`Self::bound_margin_unwired`] and
    /// [`ReluTailDualBudgetError::Budget`] when the caller's deadline or peak
    /// ceiling refuses work.
    pub fn bound_margin_unwired_with_budget(
        &self,
        margin: &ExactReluTailMargin,
        supplied_multipliers: Option<&[f64]>,
        config: ReluTailDualConfig,
        budget: ConstrainedZonotopeCallBudget,
    ) -> Result<ConstrainedZonotopeCallOutcome<ReluTailDualResult>, ReluTailDualBudgetError> {
        let mut gate = ConstrainedZonotopeCallTracker::from_system_clock(budget)?;
        let result = bound_prepared_relu_tail_margin_impl(
            self,
            margin,
            supplied_multipliers,
            config,
            &mut gate,
        )?;
        Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()))
    }

    /// Certify a final margin and immediately pull its accepted line backward
    /// through exact Conv2d weights and a certified BatchNorm surrogate.
    ///
    /// Both M17 calls, raw BatchNorm certification, exact pullback arithmetic,
    /// and publication run under one call gate.  No caller-authored direction
    /// or constant is accepted, and the temporary upstream exact margin never
    /// leaves this transaction.  The exact Conv2d transpose uses the same
    /// grouped CHW geometry as [`crate::constrained_zonotope_conv2d_unwired`].
    /// BatchNorm scale and bias errors are shared per channel; consequently the
    /// sound bias penalties are `scale_error * H_c` and
    /// `bias_error * |sum_i r_i|`, not a coordinate-wise error sum.
    ///
    /// The authored forward order assumed here is
    /// `Conv2d(BatchNorm(ReLU(x)))`: `x` is represented by `upstream`, its ReLU
    /// output is consumed by BatchNorm, and Conv2d produces the preactivation
    /// represented by `self`.  The caller must prove that this complete map
    /// sends every concrete `upstream` witness into the exact coordinate hull
    /// represented by `self`.  This semantic wiring obligation cannot be
    /// established from two abstract domains alone.
    /// `budget.baseline_live_bytes()` must include both prepared geometries and
    /// borrowed domains, the final margin, Conv2d/BatchNorm arrays, multiplier
    /// slices, and all other caller-retained storage sharing the ceiling.
    ///
    /// # Errors
    ///
    /// Fails closed on mismatched geometry, non-finite parameters, malformed
    /// BatchNorm data, exact-rational growth, either mandatory replay failure,
    /// or a shared deadline/peak-memory refusal.
    #[allow(clippy::too_many_arguments)]
    pub fn bound_conv2d_batch_norm_pullback_unwired_with_budget(
        &self,
        final_margin: &ExactReluTailMargin,
        downstream_supplied_multipliers: Option<&[f64]>,
        downstream_config: ReluTailDualConfig,
        upstream: &PreparedReluTailGeometry64<'_>,
        conv_input_shape: [usize; 3],
        conv_weights: ArrayView4<'_, f64>,
        conv_bias: &[f64],
        conv_spec: ConstrainedZonotopeConv2dSpec,
        batch_norm_spec: ConstrainedZonotopeBatchNormSpec<'_>,
        nominal_batch_norm_scale: &[f64],
        nominal_batch_norm_bias: &[f64],
        batch_norm_limits: ConstrainedZonotopeBatchNormAffineCertificateLimits,
        pullback_limits: ReluTailConvBatchNormPullbackLimits,
        upstream_supplied_multipliers: Option<&[f64]>,
        upstream_config: ReluTailDualConfig,
        budget: ConstrainedZonotopeCallBudget,
    ) -> Result<
        ConstrainedZonotopeCallOutcome<ReluTailConvBatchNormPullbackResult>,
        ReluTailConvBatchNormPullbackBudgetError,
    > {
        let (result, report) = self
            .bound_conv2d_batch_norm_pullback_unwired_attempt_with_budget(
                final_margin,
                downstream_supplied_multipliers,
                downstream_config,
                upstream,
                conv_input_shape,
                conv_weights,
                conv_bias,
                conv_spec,
                batch_norm_spec,
                nominal_batch_norm_scale,
                nominal_batch_norm_bias,
                batch_norm_limits,
                pullback_limits,
                upstream_supplied_multipliers,
                upstream_config,
                budget,
            )
            .into_parts();
        result.map(|value| ConstrainedZonotopeCallOutcome::new(value, report))
    }

    /// Attempt the transactional Conv2d/BatchNorm pullback and always return
    /// its call-local accounting receipt.
    ///
    /// Admission, validation, proof, deadline, and peak-memory failures are
    /// carried inside the returned [`ConstrainedZonotopeCallAttempt`].  This
    /// allows an enclosing optional portfolio lane to account for failed work
    /// before continuing with its mandatory fallback.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn bound_conv2d_batch_norm_pullback_unwired_attempt_with_budget(
        &self,
        final_margin: &ExactReluTailMargin,
        downstream_supplied_multipliers: Option<&[f64]>,
        downstream_config: ReluTailDualConfig,
        upstream: &PreparedReluTailGeometry64<'_>,
        conv_input_shape: [usize; 3],
        conv_weights: ArrayView4<'_, f64>,
        conv_bias: &[f64],
        conv_spec: ConstrainedZonotopeConv2dSpec,
        batch_norm_spec: ConstrainedZonotopeBatchNormSpec<'_>,
        nominal_batch_norm_scale: &[f64],
        nominal_batch_norm_bias: &[f64],
        batch_norm_limits: ConstrainedZonotopeBatchNormAffineCertificateLimits,
        pullback_limits: ReluTailConvBatchNormPullbackLimits,
        upstream_supplied_multipliers: Option<&[f64]>,
        upstream_config: ReluTailDualConfig,
        budget: ConstrainedZonotopeCallBudget,
    ) -> ConstrainedZonotopeCallAttempt<
        ReluTailConvBatchNormPullbackResult,
        ReluTailConvBatchNormPullbackBudgetError,
    > {
        let (mut gate, admission) =
            ConstrainedZonotopeCallTracker::from_system_clock_attempt(budget);
        let result = admission
            .map_err(ReluTailConvBatchNormPullbackBudgetError::from)
            .and_then(|()| {
                bound_conv2d_batch_norm_pullback_impl(
                    self,
                    final_margin,
                    downstream_supplied_multipliers,
                    downstream_config,
                    upstream,
                    conv_input_shape,
                    conv_weights,
                    conv_bias,
                    conv_spec,
                    batch_norm_spec,
                    nominal_batch_norm_scale,
                    nominal_batch_norm_bias,
                    batch_norm_limits,
                    pullback_limits,
                    upstream_supplied_multipliers,
                    upstream_config,
                    &mut gate,
                )
            });
        ConstrainedZonotopeCallAttempt::new(result, gate.report())
    }

    /// Run the transactional pullback with mandatory downstream/upstream M17
    /// certificates and an optional retained upstream M20 certificate.
    ///
    /// Invalid, disjoint, over-budget, or late auxiliary work is reported as
    /// [`ReluTailBoxCutStatus::AuxiliaryFallback`]; the mandatory upstream M17
    /// certificate remains authoritative. Ties retain M17 exactly. A budget
    /// refusal is additionally retained in
    /// [`ReluTailConvBatchNormPullbackM17M20Result::optional_budget_error`].
    ///
    /// The authored forward order is exactly
    /// `Conv2d(BatchNorm(ReLU(upstream)))`: `upstream` is the pre-ReLU domain,
    /// BatchNorm consumes that ReLU output, and `self` encloses the resulting
    /// Conv2d preactivation. The caller must prove that this complete map sends
    /// every concrete upstream witness into `self`. `upstream_auxiliary` must
    /// independently enclose those same concrete witnesses at the same
    /// pre-ReLU program location as `upstream`; a post-ReLU or other-layer
    /// enclosure is not a valid M20 premise.
    ///
    /// `budget.baseline_live_bytes()` must include both prepared geometries and
    /// borrowed domains, `upstream_auxiliary` and its endpoint arrays, the final
    /// margin, Conv2d weights and bias, every BatchNorm parameter and nominal
    /// surrogate array, both optional multiplier slices, configs/specs/limits,
    /// and all other caller-retained storage sharing the ceiling.
    ///
    /// # Errors
    ///
    /// Returns an error for tracker admission, invalid Conv2d/BatchNorm
    /// geometry, exact pullback failure, or either mandatory M17 failure.
    /// Optional M20 failures never make this method return `Err`.
    #[allow(clippy::too_many_arguments)]
    pub fn bound_conv2d_batch_norm_pullback_m17_m20_unwired_with_budget(
        &self,
        final_margin: &ExactReluTailMargin,
        downstream_supplied_multipliers: Option<&[f64]>,
        downstream_config: ReluTailDualConfig,
        upstream: &PreparedReluTailGeometry64<'_>,
        upstream_auxiliary: &CertifiedAuxiliaryBounds64,
        conv_input_shape: [usize; 3],
        conv_weights: ArrayView4<'_, f64>,
        conv_bias: &[f64],
        conv_spec: ConstrainedZonotopeConv2dSpec,
        batch_norm_spec: ConstrainedZonotopeBatchNormSpec<'_>,
        nominal_batch_norm_scale: &[f64],
        nominal_batch_norm_bias: &[f64],
        batch_norm_limits: ConstrainedZonotopeBatchNormAffineCertificateLimits,
        pullback_limits: ReluTailConvBatchNormPullbackLimits,
        upstream_supplied_multipliers: Option<&[f64]>,
        upstream_config: ReluTailDualConfig,
        budget: ConstrainedZonotopeCallBudget,
    ) -> Result<
        ConstrainedZonotopeCallOutcome<ReluTailConvBatchNormPullbackM17M20Result>,
        ReluTailConvBatchNormPullbackBudgetError,
    > {
        let (result, report) = self
            .bound_conv2d_batch_norm_pullback_m17_m20_unwired_attempt_with_budget(
                final_margin,
                downstream_supplied_multipliers,
                downstream_config,
                upstream,
                upstream_auxiliary,
                conv_input_shape,
                conv_weights,
                conv_bias,
                conv_spec,
                batch_norm_spec,
                nominal_batch_norm_scale,
                nominal_batch_norm_bias,
                batch_norm_limits,
                pullback_limits,
                upstream_supplied_multipliers,
                upstream_config,
                budget,
            )
            .into_parts();
        result.map(|value| ConstrainedZonotopeCallOutcome::new(value, report))
    }

    /// Attempt the retained M17/M20 transaction and always return accounting.
    ///
    /// This has the same forward-map, same-location auxiliary-premise, and
    /// complete caller-baseline obligations as
    /// [`Self::bound_conv2d_batch_norm_pullback_m17_m20_unwired_with_budget`].
    /// Admission and mandatory failures are carried in the attempt; optional
    /// M20 budget failures are retained inside a successful result.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn bound_conv2d_batch_norm_pullback_m17_m20_unwired_attempt_with_budget(
        &self,
        final_margin: &ExactReluTailMargin,
        downstream_supplied_multipliers: Option<&[f64]>,
        downstream_config: ReluTailDualConfig,
        upstream: &PreparedReluTailGeometry64<'_>,
        upstream_auxiliary: &CertifiedAuxiliaryBounds64,
        conv_input_shape: [usize; 3],
        conv_weights: ArrayView4<'_, f64>,
        conv_bias: &[f64],
        conv_spec: ConstrainedZonotopeConv2dSpec,
        batch_norm_spec: ConstrainedZonotopeBatchNormSpec<'_>,
        nominal_batch_norm_scale: &[f64],
        nominal_batch_norm_bias: &[f64],
        batch_norm_limits: ConstrainedZonotopeBatchNormAffineCertificateLimits,
        pullback_limits: ReluTailConvBatchNormPullbackLimits,
        upstream_supplied_multipliers: Option<&[f64]>,
        upstream_config: ReluTailDualConfig,
        budget: ConstrainedZonotopeCallBudget,
    ) -> ConstrainedZonotopeCallAttempt<
        ReluTailConvBatchNormPullbackM17M20Result,
        ReluTailConvBatchNormPullbackBudgetError,
    > {
        let (mut gate, admission) =
            ConstrainedZonotopeCallTracker::from_system_clock_attempt(budget);
        let result = admission
            .map_err(ReluTailConvBatchNormPullbackBudgetError::from)
            .and_then(|()| {
                bound_conv2d_batch_norm_pullback_m17_m20_impl(
                    self,
                    final_margin,
                    downstream_supplied_multipliers,
                    downstream_config,
                    upstream,
                    upstream_auxiliary,
                    conv_input_shape,
                    conv_weights,
                    conv_bias,
                    conv_spec,
                    batch_norm_spec,
                    nominal_batch_norm_scale,
                    nominal_batch_norm_bias,
                    batch_norm_limits,
                    pullback_limits,
                    upstream_supplied_multipliers,
                    upstream_config,
                    &mut gate,
                )
            });
        ConstrainedZonotopeCallAttempt::new(result, gate.report())
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn bound_conv2d_batch_norm_pullback_m17_m20_unwired_attempt_with_clock<N>(
        &self,
        final_margin: &ExactReluTailMargin,
        downstream_supplied_multipliers: Option<&[f64]>,
        downstream_config: ReluTailDualConfig,
        upstream: &PreparedReluTailGeometry64<'_>,
        upstream_auxiliary: &CertifiedAuxiliaryBounds64,
        conv_input_shape: [usize; 3],
        conv_weights: ArrayView4<'_, f64>,
        conv_bias: &[f64],
        conv_spec: ConstrainedZonotopeConv2dSpec,
        batch_norm_spec: ConstrainedZonotopeBatchNormSpec<'_>,
        nominal_batch_norm_scale: &[f64],
        nominal_batch_norm_bias: &[f64],
        batch_norm_limits: ConstrainedZonotopeBatchNormAffineCertificateLimits,
        pullback_limits: ReluTailConvBatchNormPullbackLimits,
        upstream_supplied_multipliers: Option<&[f64]>,
        upstream_config: ReluTailDualConfig,
        budget: ConstrainedZonotopeCallBudget,
        now: N,
    ) -> ConstrainedZonotopeCallAttempt<
        ReluTailConvBatchNormPullbackM17M20Result,
        ReluTailConvBatchNormPullbackBudgetError,
    >
    where
        N: FnMut(&'static str) -> Instant,
    {
        let (mut gate, admission) = ConstrainedZonotopeCallTracker::with_clock_attempt(budget, now);
        let result = admission
            .map_err(ReluTailConvBatchNormPullbackBudgetError::from)
            .and_then(|()| {
                bound_conv2d_batch_norm_pullback_m17_m20_impl(
                    self,
                    final_margin,
                    downstream_supplied_multipliers,
                    downstream_config,
                    upstream,
                    upstream_auxiliary,
                    conv_input_shape,
                    conv_weights,
                    conv_bias,
                    conv_spec,
                    batch_norm_spec,
                    nominal_batch_norm_scale,
                    nominal_batch_norm_bias,
                    batch_norm_limits,
                    pullback_limits,
                    upstream_supplied_multipliers,
                    upstream_config,
                    &mut gate,
                )
            });
        ConstrainedZonotopeCallAttempt::new(result, gate.report())
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn bound_conv2d_batch_norm_pullback_unwired_attempt_with_clock<N>(
        &self,
        final_margin: &ExactReluTailMargin,
        downstream_supplied_multipliers: Option<&[f64]>,
        downstream_config: ReluTailDualConfig,
        upstream: &PreparedReluTailGeometry64<'_>,
        conv_input_shape: [usize; 3],
        conv_weights: ArrayView4<'_, f64>,
        conv_bias: &[f64],
        conv_spec: ConstrainedZonotopeConv2dSpec,
        batch_norm_spec: ConstrainedZonotopeBatchNormSpec<'_>,
        nominal_batch_norm_scale: &[f64],
        nominal_batch_norm_bias: &[f64],
        batch_norm_limits: ConstrainedZonotopeBatchNormAffineCertificateLimits,
        pullback_limits: ReluTailConvBatchNormPullbackLimits,
        upstream_supplied_multipliers: Option<&[f64]>,
        upstream_config: ReluTailDualConfig,
        budget: ConstrainedZonotopeCallBudget,
        now: N,
    ) -> ConstrainedZonotopeCallAttempt<
        ReluTailConvBatchNormPullbackResult,
        ReluTailConvBatchNormPullbackBudgetError,
    >
    where
        N: FnMut(&'static str) -> Instant,
    {
        let (mut gate, admission) = ConstrainedZonotopeCallTracker::with_clock_attempt(budget, now);
        let result = admission
            .map_err(ReluTailConvBatchNormPullbackBudgetError::from)
            .and_then(|()| {
                bound_conv2d_batch_norm_pullback_impl(
                    self,
                    final_margin,
                    downstream_supplied_multipliers,
                    downstream_config,
                    upstream,
                    conv_input_shape,
                    conv_weights,
                    conv_bias,
                    conv_spec,
                    batch_norm_spec,
                    nominal_batch_norm_scale,
                    nominal_batch_norm_bias,
                    batch_norm_limits,
                    pullback_limits,
                    upstream_supplied_multipliers,
                    upstream_config,
                    &mut gate,
                )
            });
        ConstrainedZonotopeCallAttempt::new(result, gate.report())
    }

    #[cfg(test)]
    fn bound_margin_unwired_with_clock<N>(
        &self,
        margin: &ExactReluTailMargin,
        supplied_multipliers: Option<&[f64]>,
        config: ReluTailDualConfig,
        budget: ConstrainedZonotopeCallBudget,
        now: N,
    ) -> Result<ConstrainedZonotopeCallOutcome<ReluTailDualResult>, ReluTailDualBudgetError>
    where
        N: FnMut(&'static str) -> Instant,
    {
        let mut gate = ConstrainedZonotopeCallTracker::with_clock(budget, now)?;
        let result = bound_prepared_relu_tail_margin_impl(
            self,
            margin,
            supplied_multipliers,
            config,
            &mut gate,
        )?;
        Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()))
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

    /// Bound one exact M20 margin behind the shared synchronous call firewall.
    ///
    /// This clones and intersects the cached hull without revisiting generator
    /// entries, then gates exact line construction and every outward replay.
    /// It does not silently replace or suppress M17: a portfolio caller must
    /// independently invoke [`Self::bound_margin_unwired_with_budget`] and
    /// retain the better replay-certified result.  The caller continues to own
    /// the proof that every concrete preactivation witness lies in `auxiliary`.
    ///
    /// `budget.baseline_live_bytes()` must include
    /// [`Self::conservative_live_bytes`], the borrowed domain, all arguments,
    /// and other caller-owned buffers sharing the same ceiling.
    ///
    /// # Errors
    ///
    /// Returns [`ReluTailDualBudgetError::Bound`] for the same auxiliary,
    /// margin, exact-line, and mandatory-replay failures as
    /// [`Self::bound_margin_with_auxiliary_bounds_unwired`], and
    /// [`ReluTailDualBudgetError::Budget`] when the call firewall refuses work.
    pub fn bound_margin_with_auxiliary_bounds_unwired_with_budget(
        &self,
        auxiliary: &CertifiedAuxiliaryBounds64,
        margin: &ExactReluTailMargin,
        supplied_multipliers: Option<&[f64]>,
        config: ReluTailDualConfig,
        budget: ConstrainedZonotopeCallBudget,
    ) -> Result<ConstrainedZonotopeCallOutcome<ReluTailDualResult>, ReluTailDualBudgetError> {
        let mut gate = ConstrainedZonotopeCallTracker::from_system_clock(budget)?;
        let result = bound_prepared_relu_tail_margin_with_auxiliary_impl(
            self,
            auxiliary,
            margin,
            supplied_multipliers,
            config,
            0,
            &mut gate,
        )?;
        Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()))
    }

    /// Bound one margin with a mandatory M17 certificate and an optional M20
    /// auxiliary certificate behind one shared execution firewall.
    ///
    /// M17 is evaluated first and remains authoritative on every optional M20
    /// failure, including malformed/disjoint auxiliary bounds, allocation
    /// refusal, peak-budget refusal, or deadline exhaustion. The returned call
    /// report covers both the mandatory work and every attempted optional
    /// operation. Selection is a strict ordered maximum, so ties retain M17
    /// bit-for-bit.
    ///
    /// The caller owns the semantic proof that every concrete preactivation
    /// witness represented by this prepared domain lies in `auxiliary`.
    /// `budget.baseline_live_bytes()` must include the prepared geometry, the
    /// borrowed domain, both arguments, and all other caller-owned live state.
    /// The call itself accounts for the mandatory M17 result retained while
    /// M20 is validated, constructed, and replayed.
    ///
    /// # Errors
    ///
    /// Returns an error only when construction of the shared tracker or the
    /// mandatory M17 setup/replay fails. Optional M20 failures are represented
    /// by [`ReluTailBoxCutStatus::AuxiliaryFallback`] in the returned portfolio.
    pub fn bound_margin_m17_m20_unwired_with_budget(
        &self,
        auxiliary: &CertifiedAuxiliaryBounds64,
        margin: &ExactReluTailMargin,
        supplied_multipliers: Option<&[f64]>,
        config: ReluTailDualConfig,
        budget: ConstrainedZonotopeCallBudget,
    ) -> Result<ConstrainedZonotopeCallOutcome<ReluTailBoxCutDualResult>, ReluTailDualBudgetError>
    {
        let mut gate = ConstrainedZonotopeCallTracker::from_system_clock(budget)?;
        let original = bound_prepared_relu_tail_margin_impl(
            self,
            margin,
            supplied_multipliers,
            config,
            &mut gate,
        )?;
        let retained_original_bytes = relu_tail_dual_result_live_bytes(
            self.domain.value_dim(),
            self.domain.constraint_count(),
        )?;
        let auxiliary_result = bound_prepared_relu_tail_margin_with_auxiliary_impl(
            self,
            auxiliary,
            margin,
            supplied_multipliers,
            config,
            retained_original_bytes,
            &mut gate,
        )
        .ok();
        let status = if auxiliary_result.is_some() {
            ReluTailBoxCutStatus::Completed
        } else {
            ReluTailBoxCutStatus::AuxiliaryFallback
        };
        let result = finish_box_cut_portfolio(original, auxiliary_result, None, status);
        Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()))
    }

    #[cfg(test)]
    fn bound_margin_with_auxiliary_bounds_unwired_with_clock<N>(
        &self,
        auxiliary: &CertifiedAuxiliaryBounds64,
        margin: &ExactReluTailMargin,
        supplied_multipliers: Option<&[f64]>,
        config: ReluTailDualConfig,
        budget: ConstrainedZonotopeCallBudget,
        now: N,
    ) -> Result<ConstrainedZonotopeCallOutcome<ReluTailDualResult>, ReluTailDualBudgetError>
    where
        N: FnMut(&'static str) -> Instant,
    {
        let mut gate = ConstrainedZonotopeCallTracker::with_clock(budget, now)?;
        let result = bound_prepared_relu_tail_margin_with_auxiliary_impl(
            self,
            auxiliary,
            margin,
            supplied_multipliers,
            config,
            0,
            &mut gate,
        )?;
        Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()))
    }

    /// Bound one margin with mandatory M17, optional M20, and optional M24
    /// behind one shared synchronous execution firewall.
    ///
    /// The exact authority order is fixed: M17, then M20, then the first
    /// replay-certified M24 candidate. When the authenticated Box has no
    /// endpoint strictly inside the prepared hull, M17's accepted pointwise
    /// replay is reused as the explicit, bit-identical M20 certificate; its
    /// result fields describe that reused certificate, while the call report
    /// charges only work actually performed. Selection changes only on a
    /// strict improvement, so ties retain the earlier member bit-for-bit.
    /// M20 and every M24 phase are optional; their validation, resource,
    /// allocation, deadline, search, or exact-replay failures retain the best
    /// earlier certificate and publish complete attempted-work accounting.
    ///
    /// `budget.baseline_live_bytes()` has the same caller-owned obligations as
    /// [`Self::bound_margin_m17_m20_unwired_with_budget`]. This call adds the
    /// retained M17/M20 results, optimizer state, exact-replay scratch, and a
    /// prior retained M24 certificate to each applicable peak preflight.
    ///
    /// # Errors
    ///
    /// Returns an error only when shared admission or mandatory M17
    /// setup/replay fails. Optional firewall refusal is retained in
    /// [`ReluTailBoxCutBudgetedResult::optional_budget_error`].
    #[allow(clippy::too_many_arguments)]
    pub fn bound_margin_m17_m20_m24_unwired_with_budget(
        &self,
        auxiliary: &CertifiedAuxiliaryBounds64,
        margin: &ExactReluTailMargin,
        supplied_predicate_multipliers: Option<&[f64]>,
        relu_config: ReluTailDualConfig,
        box_config: ReluTailBoxCutOptimizerConfig,
        budget: ConstrainedZonotopeCallBudget,
    ) -> Result<ConstrainedZonotopeCallOutcome<ReluTailBoxCutBudgetedResult>, ReluTailDualBudgetError>
    {
        let mut gate = ConstrainedZonotopeCallTracker::from_system_clock(budget)?;
        self.bound_margin_m17_m20_m24_unwired_with_call_gate(
            auxiliary,
            margin,
            supplied_predicate_multipliers,
            relu_config,
            box_config,
            &mut gate,
        )
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn bound_margin_m17_m20_m24_unwired_with_clock<N>(
        &self,
        auxiliary: &CertifiedAuxiliaryBounds64,
        margin: &ExactReluTailMargin,
        supplied_predicate_multipliers: Option<&[f64]>,
        relu_config: ReluTailDualConfig,
        box_config: ReluTailBoxCutOptimizerConfig,
        budget: ConstrainedZonotopeCallBudget,
        now: N,
    ) -> Result<ConstrainedZonotopeCallOutcome<ReluTailBoxCutBudgetedResult>, ReluTailDualBudgetError>
    where
        N: FnMut(&'static str) -> Instant,
    {
        let mut gate = ConstrainedZonotopeCallTracker::with_clock(budget, now)?;
        self.bound_margin_m17_m20_m24_unwired_with_call_gate(
            auxiliary,
            margin,
            supplied_predicate_multipliers,
            relu_config,
            box_config,
            &mut gate,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn bound_margin_m17_m20_m24_unwired_with_call_gate<G>(
        &self,
        auxiliary: &CertifiedAuxiliaryBounds64,
        margin: &ExactReluTailMargin,
        supplied_predicate_multipliers: Option<&[f64]>,
        relu_config: ReluTailDualConfig,
        box_config: ReluTailBoxCutOptimizerConfig,
        gate: &mut G,
    ) -> Result<ConstrainedZonotopeCallOutcome<ReluTailBoxCutBudgetedResult>, ReluTailDualBudgetError>
    where
        G: ConstrainedZonotopeCallGate,
    {
        let original = bound_prepared_relu_tail_margin_impl(
            self,
            margin,
            supplied_predicate_multipliers,
            relu_config,
            gate,
        )?;
        let result_live_bytes = match relu_tail_dual_result_live_bytes(
            self.domain.value_dim(),
            self.domain.constraint_count(),
        ) {
            Ok(bytes) => bytes,
            Err(error) => {
                let status = box_optimizer_status_from_budget_error(&error);
                let result = finish_budgeted_optimized_box_cut_result(
                    original,
                    None,
                    None,
                    status,
                    None,
                    BoxSearchTelemetry::default(),
                    Some(error),
                );
                return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
            }
        };

        // Count useful Box endpoints before replaying M20.  If none is
        // strict, intersecting the prepared hull with `auxiliary` leaves the
        // exact line plan unchanged.  The already accepted M17 affine replay
        // is therefore also a valid M20 certificate, without assuming that a
        // fresh heuristic run under a different clock would follow the same
        // candidate schedule.  M24 also has no multiplier variables.
        // Validate the dimension explicitly before the gated counter: that
        // helper's zip is deliberately unchecked because all of its other
        // callers have already crossed an equivalent shape boundary.
        if let Err(error) = gate.checkpoint("prepared M20/M24 auxiliary validation") {
            let result = finish_budgeted_optimized_box_cut_result(
                original,
                None,
                None,
                ReluTailBoxCutOptimizerStatus::AuxiliaryFallback,
                None,
                BoxSearchTelemetry::default(),
                Some(error),
            );
            return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
        }
        if auxiliary.value_dim() != self.domain.value_dim() {
            let result = finish_budgeted_optimized_box_cut_result(
                original,
                None,
                None,
                ReluTailBoxCutOptimizerStatus::AuxiliaryFallback,
                None,
                BoxSearchTelemetry::default(),
                None,
            );
            return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
        }
        if let Err(error) = gate.checkpoint("prepared M20/M24 auxiliary validation complete") {
            let result = finish_budgeted_optimized_box_cut_result(
                original,
                None,
                None,
                ReluTailBoxCutOptimizerStatus::AuxiliaryFallback,
                None,
                BoxSearchTelemetry::default(),
                Some(error),
            );
            return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
        }

        let endpoint_count_peak = match box_cut_endpoint_count_peak_live_bytes().and_then(|bytes| {
            result_live_bytes.checked_add(bytes).ok_or(
                ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                    operation: "M24 endpoint count plus retained M17 result bytes",
                },
            )
        }) {
            Ok(bytes) => bytes,
            Err(error) => {
                let result = finish_budgeted_optimized_box_cut_result(
                    original,
                    None,
                    None,
                    ReluTailBoxCutOptimizerStatus::AuxiliaryFallback,
                    None,
                    BoxSearchTelemetry::default(),
                    Some(error),
                );
                return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
            }
        };
        if let Err(error) = gate.preflight_peak_live_bytes(endpoint_count_peak) {
            let result = finish_budgeted_optimized_box_cut_result(
                original,
                None,
                None,
                ReluTailBoxCutOptimizerStatus::AuxiliaryFallback,
                None,
                BoxSearchTelemetry::default(),
                Some(error),
            );
            return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
        }

        let box_variables = match count_tighter_auxiliary_box_endpoints_with_gate(
            &self.exact_coordinate_bounds,
            auxiliary,
            gate,
        ) {
            Ok(count) => count,
            Err(error) => {
                let optional_budget_error = match error {
                    ReluTailDualBudgetError::Budget(error) => Some(error),
                    ReluTailDualBudgetError::Bound(_) => None,
                };
                let result = finish_budgeted_optimized_box_cut_result(
                    original,
                    None,
                    None,
                    ReluTailBoxCutOptimizerStatus::AuxiliaryFallback,
                    None,
                    BoxSearchTelemetry::default(),
                    optional_budget_error,
                );
                return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
            }
        };
        if box_variables == 0 {
            let retained_result_bytes = match result_live_bytes.checked_mul(2) {
                Some(bytes) => bytes,
                None => {
                    let error = ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                        operation: "equivalent M17/M20 retained result bytes",
                    };
                    let result = finish_budgeted_optimized_box_cut_result(
                        original,
                        None,
                        None,
                        ReluTailBoxCutOptimizerStatus::AuxiliaryFallback,
                        None,
                        BoxSearchTelemetry::default(),
                        Some(error),
                    );
                    return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
                }
            };
            if let Err(error) = gate.preflight_peak_live_bytes(retained_result_bytes) {
                let result = finish_budgeted_optimized_box_cut_result(
                    original,
                    None,
                    None,
                    ReluTailBoxCutOptimizerStatus::AuxiliaryFallback,
                    None,
                    BoxSearchTelemetry::default(),
                    Some(error),
                );
                return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
            }
            let equivalent_auxiliary =
                match clone_equivalent_relu_tail_result_with_gate(&original, gate) {
                    Ok(result) => result,
                    Err(error) => {
                        let optional_budget_error = match error {
                            ReluTailDualBudgetError::Budget(error) => Some(error),
                            ReluTailDualBudgetError::Bound(_) => None,
                        };
                        let result = finish_budgeted_optimized_box_cut_result(
                            original,
                            None,
                            None,
                            ReluTailBoxCutOptimizerStatus::AuxiliaryFallback,
                            None,
                            BoxSearchTelemetry::default(),
                            optional_budget_error,
                        );
                        return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
                    }
                };
            let result = finish_budgeted_optimized_box_cut_result(
                original,
                Some(equivalent_auxiliary),
                None,
                ReluTailBoxCutOptimizerStatus::NoTighterAuxiliaryBox,
                None,
                BoxSearchTelemetry::default(),
                None,
            );
            return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
        }

        let auxiliary_result = match bound_prepared_relu_tail_margin_with_auxiliary_impl(
            self,
            auxiliary,
            margin,
            supplied_predicate_multipliers,
            relu_config,
            result_live_bytes,
            gate,
        ) {
            Ok(result) => result,
            Err(error) => {
                let optional_budget_error = match error {
                    ReluTailDualBudgetError::Budget(error) => Some(error),
                    ReluTailDualBudgetError::Bound(_) => None,
                };
                let result = finish_budgeted_optimized_box_cut_result(
                    original,
                    None,
                    None,
                    ReluTailBoxCutOptimizerStatus::AuxiliaryFallback,
                    None,
                    BoxSearchTelemetry::default(),
                    optional_budget_error,
                );
                return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
            }
        };

        let retained_result_bytes = match result_live_bytes.checked_mul(2) {
            Some(bytes) => bytes,
            None => {
                let error = ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                    operation: "M24 retained M17/M20 result bytes",
                };
                let result = finish_budgeted_optimized_box_cut_result(
                    original,
                    Some(auxiliary_result),
                    None,
                    ReluTailBoxCutOptimizerStatus::ResourceFallback,
                    None,
                    BoxSearchTelemetry::default(),
                    Some(error),
                );
                return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
            }
        };

        if let Err(error) = gate.checkpoint("M24 optimizer plan admission") {
            let status = box_optimizer_status_from_budget_error(&error);
            let result = finish_budgeted_optimized_box_cut_result(
                original,
                Some(auxiliary_result),
                None,
                status,
                None,
                BoxSearchTelemetry::default(),
                Some(error),
            );
            return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
        }
        let plan = match ReluTailBoxCutOptimizerPlan::checked(
            self.domain,
            self.generator_nonzeros,
            box_variables,
            box_config,
        ) {
            Ok(plan) => plan,
            Err(status) => {
                let result = finish_budgeted_optimized_box_cut_result(
                    original,
                    Some(auxiliary_result),
                    None,
                    status,
                    None,
                    BoxSearchTelemetry::default(),
                    None,
                );
                return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
            }
        };
        if plan.total_iterations == 0 {
            let result = finish_budgeted_optimized_box_cut_result(
                original,
                Some(auxiliary_result),
                None,
                ReluTailBoxCutOptimizerStatus::SearchDisabled,
                Some(plan),
                BoxSearchTelemetry::default(),
                None,
            );
            return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
        }

        let search_peak = box_cut_search_peak_live_bytes(plan);
        let search_peak = match search_peak.and_then(|bytes| {
            retained_result_bytes.checked_add(bytes).ok_or(
                ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                    operation: "M24 search plus retained result bytes",
                },
            )
        }) {
            Ok(bytes) => bytes,
            Err(error) => {
                let status = box_optimizer_status_from_budget_error(&error);
                let result = finish_budgeted_optimized_box_cut_result(
                    original,
                    Some(auxiliary_result),
                    None,
                    status,
                    Some(plan),
                    BoxSearchTelemetry::default(),
                    Some(error),
                );
                return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
            }
        };
        if let Err(error) = gate.preflight_peak_live_bytes(search_peak) {
            let status = box_optimizer_status_from_budget_error(&error);
            let result = finish_budgeted_optimized_box_cut_result(
                original,
                Some(auxiliary_result),
                None,
                status,
                Some(plan),
                BoxSearchTelemetry::default(),
                Some(error),
            );
            return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
        }

        let budgeted_search = optimize_auxiliary_box_multipliers_with_call_gate(
            self.domain,
            &self.exact_coordinate_bounds,
            auxiliary,
            &auxiliary_result.direction,
            box_config,
            plan,
            gate,
        );
        let search = budgeted_search.search;
        if let Some(error) = budgeted_search.budget_error {
            let status = box_optimizer_status_from_budget_error(&error);
            let result = finish_budgeted_optimized_box_cut_result(
                original,
                Some(auxiliary_result),
                None,
                status,
                Some(plan),
                BoxSearchTelemetry {
                    iterations_completed: search.iterations_completed,
                    restarts_completed: search.restarts_completed,
                    candidates_scored: search.candidates_scored,
                    exact_replays: 0,
                },
                Some(error),
            );
            return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
        }

        let search_retained_bytes = match box_cut_search_retained_live_bytes(plan) {
            Ok(bytes) => bytes,
            Err(error) => {
                let status = box_optimizer_status_from_budget_error(&error);
                let result = finish_budgeted_optimized_box_cut_result(
                    original,
                    Some(auxiliary_result),
                    None,
                    status,
                    Some(plan),
                    BoxSearchTelemetry {
                        iterations_completed: search.iterations_completed,
                        restarts_completed: search.restarts_completed,
                        candidates_scored: search.candidates_scored,
                        exact_replays: 0,
                    },
                    Some(error),
                );
                return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
            }
        };
        let exact_replay_owned_bytes = match box_cut_exact_replay_peak_live_bytes(
            self.domain.value_dim(),
            self.domain.constraint_count(),
        ) {
            Ok(bytes) => bytes,
            Err(error) => {
                let status = box_optimizer_status_from_budget_error(&error);
                let result = finish_budgeted_optimized_box_cut_result(
                    original,
                    Some(auxiliary_result),
                    None,
                    status,
                    Some(plan),
                    BoxSearchTelemetry {
                        iterations_completed: search.iterations_completed,
                        restarts_completed: search.restarts_completed,
                        candidates_scored: search.candidates_scored,
                        exact_replays: 0,
                    },
                    Some(error),
                );
                return Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()));
            }
        };

        let mut best_box_cut = None;
        let mut exact_replays = 0_usize;
        let mut exact_replay_failed = false;
        let mut optional_budget_error = None;
        for candidate in search.candidates.iter().take(plan.exact_replays) {
            let retained_certificate_bytes = if best_box_cut.is_some() {
                match relu_tail_box_cut_certificate_live_bytes(
                    self.domain.value_dim(),
                    self.domain.constraint_count(),
                ) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        optional_budget_error = Some(error);
                        break;
                    }
                }
            } else {
                0
            };
            let replay_peak = retained_result_bytes
                .checked_add(search_retained_bytes)
                .and_then(|bytes| bytes.checked_add(retained_certificate_bytes))
                .and_then(|bytes| bytes.checked_add(exact_replay_owned_bytes));
            let Some(replay_peak) = replay_peak else {
                optional_budget_error =
                    Some(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                        operation: "M24 exact replay aggregate peak",
                    });
                break;
            };
            if let Err(error) = gate.preflight_peak_live_bytes(replay_peak) {
                optional_budget_error = Some(error);
                break;
            }
            if let Err(error) = gate.checkpoint("M24 exact replay admission") {
                optional_budget_error = Some(error);
                break;
            }
            exact_replays += 1;
            #[cfg(test)]
            let force_replay_failure =
                BOX_OPTIMIZER_FAIL_NEXT_EXACT_REPLAY.with(|fail| fail.replace(false));
            #[cfg(not(test))]
            let force_replay_failure = false;
            let replay: Result<ReluTailBoxCutCertificate, ReluTailDualBudgetError> =
                if force_replay_failure {
                    Err(ReluTailDualError::NonFiniteArithmetic {
                        coordinate: 0,
                        operation: "injected budgeted optimized Box exact replay failure",
                    }
                    .into())
                } else {
                    expand_box_candidate_with_gate(
                        &search.variables,
                        candidate,
                        self.domain.value_dim(),
                        gate,
                    )
                    .and_then(|(upper, lower)| {
                        build_auxiliary_box_cut_certificate_with_original_hull_and_gate(
                            self.domain,
                            auxiliary,
                            &auxiliary_result,
                            &upper,
                            &lower,
                            supplied_predicate_multipliers,
                            &self.exact_coordinate_bounds,
                            gate,
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
                Err(ReluTailDualBudgetError::Bound(_)) => exact_replay_failed = true,
                Err(ReluTailDualBudgetError::Budget(error)) => {
                    optional_budget_error = Some(error);
                    break;
                }
            }
        }

        let search_status = if let Some(error) = optional_budget_error.as_ref() {
            box_optimizer_status_from_budget_error(error)
        } else if exact_replay_failed {
            ReluTailBoxCutOptimizerStatus::ExactReplayFallback
        } else {
            search.status
        };
        let result = finish_budgeted_optimized_box_cut_result(
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
            optional_budget_error,
        );
        Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()))
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

/// Outcome of an optional auxiliary-certificate lane.
///
/// Every fallback retains the mandatory original M17 result, and an available
/// M20 result, in the returned portfolio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReluTailBoxCutStatus {
    /// The requested optional certificate completed. For Box-cut entry points,
    /// the cut direction and exact correction were independently replayed.
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
    /// Outcome of the optional auxiliary-certificate lane.
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

/// Budgeted M17/M20/M24 result plus an optional-lane firewall receipt.
///
/// Keeping the refusal separate preserves the source-compatible shape of
/// [`ReluTailBoxCutOptimizedResult`] for existing unbudgeted callers. A
/// successful call always contains a sound `optimized` portfolio; refusal of
/// optional M20 or M24 work is reported here instead of replacing that value.
#[derive(Clone, Debug, PartialEq)]
pub struct ReluTailBoxCutBudgetedResult {
    /// Best sound portfolio completed before the shared firewall closed.
    pub optimized: ReluTailBoxCutOptimizedResult,
    /// Shared-firewall refusal from optional M20/M24 work, when present.
    pub optional_budget_error: Option<ConstrainedZonotopeCallBudgetError>,
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
    #[error(
        "exact rational growth at coordinate {coordinate} while computing {operation}: {bits} bits above {limit}"
    )]
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

/// ReLU-tail proof failure or call-local execution-firewall refusal.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ReluTailDualBudgetError {
    /// Exact geometry, declared objective, or mandatory replay was invalid.
    #[error(transparent)]
    Bound(#[from] ReluTailDualError),

    /// The caller's absolute deadline or aggregate peak-memory ceiling refused
    /// work.
    #[error(transparent)]
    Budget(#[from] ConstrainedZonotopeCallBudgetError),
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
    let conservative_live_bytes =
        prepared_relu_tail_geometry_live_bytes(domain.value_dim()).unwrap_or(usize::MAX);
    Ok(PreparedReluTailGeometry64 {
        domain,
        exact_coordinate_bounds,
        generator_nonzeros,
        conservative_live_bytes,
    })
}

/// Prepare domain-tied exact coordinate geometry behind the shared call
/// firewall.
///
/// The preflight covers the retained prepared geometry, the overlapping exact
/// coordinate-radius storage, and exact-arithmetic scratch.  The caller's
/// baseline must separately include the borrowed `domain` and all other live
/// storage sharing the same ceiling.  On success, callers retaining the result
/// across another budgeted call must add
/// [`PreparedReluTailGeometry64::conservative_live_bytes`] to that later
/// call's baseline.
///
/// # Errors
///
/// Returns [`ReluTailDualBudgetError::Bound`] for the same immutable geometry
/// failures as [`prepare_relu_tail_triangle_dual_unwired`] and
/// [`ReluTailDualBudgetError::Budget`] when the caller's deadline or peak-live
/// ceiling refuses work.
pub fn prepare_relu_tail_triangle_dual_unwired_with_budget(
    domain: &ConstrainedZonotope64,
    budget: ConstrainedZonotopeCallBudget,
) -> Result<ConstrainedZonotopeCallOutcome<PreparedReluTailGeometry64<'_>>, ReluTailDualBudgetError>
{
    let (result, report) =
        prepare_relu_tail_triangle_dual_unwired_attempt_with_budget(domain, budget).into_parts();
    result.map(|prepared| ConstrainedZonotopeCallOutcome::new(prepared, report))
}

/// Attempt prepared ReLU-tail geometry construction and always return its
/// call-local accounting receipt.
///
/// Unlike [`prepare_relu_tail_triangle_dual_unwired_with_budget`], admission,
/// geometry, deadline, and peak-memory failures are carried inside the returned
/// [`ConstrainedZonotopeCallAttempt`].  This is useful when an enclosing
/// transaction must account for a failed optional preparation.
#[must_use]
pub fn prepare_relu_tail_triangle_dual_unwired_attempt_with_budget(
    domain: &ConstrainedZonotope64,
    budget: ConstrainedZonotopeCallBudget,
) -> ConstrainedZonotopeCallAttempt<PreparedReluTailGeometry64<'_>, ReluTailDualBudgetError> {
    let (mut gate, admission) = ConstrainedZonotopeCallTracker::from_system_clock_attempt(budget);
    let result = admission
        .map_err(ReluTailDualBudgetError::from)
        .and_then(|()| prepare_relu_tail_triangle_dual_impl(domain, &mut gate));
    ConstrainedZonotopeCallAttempt::new(result, gate.report())
}

#[cfg(test)]
fn prepare_relu_tail_triangle_dual_unwired_with_clock<N>(
    domain: &ConstrainedZonotope64,
    budget: ConstrainedZonotopeCallBudget,
    now: N,
) -> Result<ConstrainedZonotopeCallOutcome<PreparedReluTailGeometry64<'_>>, ReluTailDualBudgetError>
where
    N: FnMut(&'static str) -> Instant,
{
    let (result, report) =
        prepare_relu_tail_triangle_dual_unwired_attempt_with_clock(domain, budget, now)
            .into_parts();
    result.map(|prepared| ConstrainedZonotopeCallOutcome::new(prepared, report))
}

#[cfg(test)]
fn prepare_relu_tail_triangle_dual_unwired_attempt_with_clock<N>(
    domain: &ConstrainedZonotope64,
    budget: ConstrainedZonotopeCallBudget,
    now: N,
) -> ConstrainedZonotopeCallAttempt<PreparedReluTailGeometry64<'_>, ReluTailDualBudgetError>
where
    N: FnMut(&'static str) -> Instant,
{
    let (mut gate, admission) = ConstrainedZonotopeCallTracker::with_clock_attempt(budget, now);
    let result = admission
        .map_err(ReluTailDualBudgetError::from)
        .and_then(|()| prepare_relu_tail_triangle_dual_impl(domain, &mut gate));
    ConstrainedZonotopeCallAttempt::new(result, gate.report())
}

fn prepare_relu_tail_triangle_dual_impl<'domain, G>(
    domain: &'domain ConstrainedZonotope64,
    gate: &mut G,
) -> Result<PreparedReluTailGeometry64<'domain>, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    gate.checkpoint("prepared ReLU-tail geometry validation")?;
    let generator_nonzeros = check_mandatory_domain_resources_with_gate(domain, gate)?;
    gate.checkpoint("prepared ReLU-tail geometry validation complete")?;
    let conservative_live_bytes = prepared_relu_tail_geometry_live_bytes(domain.value_dim())?;
    if gate.is_enforcing() {
        gate.preflight_peak_live_bytes(prepared_relu_tail_geometry_peak_live_bytes(
            domain.value_dim(),
        )?)?;
    }
    gate.checkpoint("prepared ReLU-tail geometry peak-memory preflight complete")?;
    let exact_coordinate_bounds = exact_coordinate_bounds_with_gate(domain, gate)?;
    gate.checkpoint("prepared ReLU-tail exact coordinate hull complete")?;
    let prepared = PreparedReluTailGeometry64 {
        domain,
        exact_coordinate_bounds,
        generator_nonzeros,
        conservative_live_bytes,
    };
    gate.checkpoint("prepared ReLU-tail geometry publication")?;
    Ok(prepared)
}

#[allow(clippy::too_many_arguments)]
fn bound_conv2d_batch_norm_pullback_impl<G>(
    downstream_prepared: &PreparedReluTailGeometry64<'_>,
    final_margin: &ExactReluTailMargin,
    downstream_supplied_multipliers: Option<&[f64]>,
    downstream_config: ReluTailDualConfig,
    upstream_prepared: &PreparedReluTailGeometry64<'_>,
    conv_input_shape: [usize; 3],
    conv_weights: ArrayView4<'_, f64>,
    conv_bias: &[f64],
    conv_spec: ConstrainedZonotopeConv2dSpec,
    batch_norm_spec: ConstrainedZonotopeBatchNormSpec<'_>,
    nominal_batch_norm_scale: &[f64],
    nominal_batch_norm_bias: &[f64],
    batch_norm_limits: ConstrainedZonotopeBatchNormAffineCertificateLimits,
    pullback_limits: ReluTailConvBatchNormPullbackLimits,
    upstream_supplied_multipliers: Option<&[f64]>,
    upstream_config: ReluTailDualConfig,
    gate: &mut G,
) -> Result<ReluTailConvBatchNormPullbackResult, ReluTailConvBatchNormPullbackBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    gate.checkpoint("transactional ReLU-tail pullback admission")?;
    let plan = validate_conv2d_batch_norm_pullback_with_gate(
        downstream_prepared,
        upstream_prepared,
        conv_input_shape,
        conv_weights,
        conv_bias,
        conv_spec,
        batch_norm_spec,
        pullback_limits,
        gate,
    )?;
    gate.checkpoint("transactional ReLU-tail pullback validation complete")?;

    if gate.is_enforcing() {
        let mut peak = ConstrainedZonotopePeakLiveBytes::new();
        peak.add_bytes(
            relu_tail_peak_live_bytes(
                downstream_prepared.domain,
                downstream_prepared.generator_nonzeros,
            )?,
            "transactional downstream M17 peak bytes",
        )?;
        peak.add_bytes(
            size_of::<ReluTailConvBatchNormPullbackResult>(),
            "transactional pullback result header bytes",
        )?;
        gate.preflight_peak_live_bytes(peak.finish())?;
    }
    gate.checkpoint("transactional downstream M17 peak-memory preflight complete")?;
    let downstream = bound_prepared_relu_tail_margin_impl(
        downstream_prepared,
        final_margin,
        downstream_supplied_multipliers,
        downstream_config,
        gate,
    )
    .map_err(map_relu_tail_pullback_budget_error)?;
    gate.checkpoint("transactional downstream M17 accepted line complete")?;

    let downstream_live_bytes = relu_tail_dual_result_live_bytes(
        downstream_prepared.value_dim(),
        downstream_prepared.domain.constraint_count(),
    )?;
    let pulled_margin = build_conv2d_batch_norm_pulled_margin_with_gate(
        upstream_prepared,
        &downstream,
        conv_input_shape,
        conv_weights,
        conv_bias,
        conv_spec,
        batch_norm_spec,
        nominal_batch_norm_scale,
        nominal_batch_norm_bias,
        batch_norm_limits,
        plan,
        downstream_live_bytes,
        gate,
    )?;
    gate.checkpoint("transactional exact pullback complete")?;

    if gate.is_enforcing() {
        let mut peak = ConstrainedZonotopePeakLiveBytes::new();
        peak.add_bytes(
            downstream_live_bytes,
            "retained transactional downstream result bytes",
        )?;
        peak.add_bytes(
            exact_relu_tail_margin_live_bytes(upstream_prepared.value_dim())?,
            "retained transactional pulled-margin bytes",
        )?;
        peak.add_bytes(
            relu_tail_peak_live_bytes(
                upstream_prepared.domain,
                upstream_prepared.generator_nonzeros,
            )?,
            "transactional upstream M17 peak bytes",
        )?;
        peak.add_bytes(
            size_of::<ReluTailConvBatchNormPullbackResult>(),
            "transactional pullback result header bytes",
        )?;
        gate.preflight_peak_live_bytes(peak.finish())?;
    }
    gate.checkpoint("transactional upstream M17 peak-memory preflight complete")?;
    let upstream = bound_prepared_internally_pulled_relu_tail_margin_impl(
        upstream_prepared,
        &pulled_margin,
        upstream_supplied_multipliers,
        upstream_config,
        gate,
    )
    .map_err(map_relu_tail_pullback_budget_error)?;
    drop(pulled_margin);
    gate.checkpoint("transactional upstream M17 complete")?;

    let result = ReluTailConvBatchNormPullbackResult {
        downstream,
        upstream,
        plan,
    };
    gate.checkpoint("transactional ReLU-tail pullback publication")?;
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn bound_conv2d_batch_norm_pullback_m17_m20_impl<G>(
    downstream_prepared: &PreparedReluTailGeometry64<'_>,
    final_margin: &ExactReluTailMargin,
    downstream_supplied_multipliers: Option<&[f64]>,
    downstream_config: ReluTailDualConfig,
    upstream_prepared: &PreparedReluTailGeometry64<'_>,
    upstream_auxiliary: &CertifiedAuxiliaryBounds64,
    conv_input_shape: [usize; 3],
    conv_weights: ArrayView4<'_, f64>,
    conv_bias: &[f64],
    conv_spec: ConstrainedZonotopeConv2dSpec,
    batch_norm_spec: ConstrainedZonotopeBatchNormSpec<'_>,
    nominal_batch_norm_scale: &[f64],
    nominal_batch_norm_bias: &[f64],
    batch_norm_limits: ConstrainedZonotopeBatchNormAffineCertificateLimits,
    pullback_limits: ReluTailConvBatchNormPullbackLimits,
    upstream_supplied_multipliers: Option<&[f64]>,
    upstream_config: ReluTailDualConfig,
    gate: &mut G,
) -> Result<ReluTailConvBatchNormPullbackM17M20Result, ReluTailConvBatchNormPullbackBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    gate.checkpoint("transactional retained M17/M20 ReLU-tail pullback admission")?;
    let plan = validate_conv2d_batch_norm_pullback_with_gate(
        downstream_prepared,
        upstream_prepared,
        conv_input_shape,
        conv_weights,
        conv_bias,
        conv_spec,
        batch_norm_spec,
        pullback_limits,
        gate,
    )?;
    gate.checkpoint("transactional retained M17/M20 ReLU-tail pullback validation complete")?;

    if gate.is_enforcing() {
        let mut peak = ConstrainedZonotopePeakLiveBytes::new();
        peak.add_bytes(
            relu_tail_peak_live_bytes(
                downstream_prepared.domain,
                downstream_prepared.generator_nonzeros,
            )?,
            "transactional retained downstream M17 peak bytes",
        )?;
        peak.add_bytes(
            size_of::<ReluTailConvBatchNormPullbackM17M20Result>(),
            "transactional retained pullback result header bytes",
        )?;
        gate.preflight_peak_live_bytes(peak.finish())?;
    }
    gate.checkpoint("transactional retained downstream M17 peak-memory preflight complete")?;
    let downstream = bound_prepared_relu_tail_margin_impl(
        downstream_prepared,
        final_margin,
        downstream_supplied_multipliers,
        downstream_config,
        gate,
    )
    .map_err(map_relu_tail_pullback_budget_error)?;
    gate.checkpoint("transactional retained downstream M17 accepted line complete")?;

    let downstream_live_bytes = relu_tail_dual_result_live_bytes(
        downstream_prepared.value_dim(),
        downstream_prepared.domain.constraint_count(),
    )?;
    let pulled_margin = build_conv2d_batch_norm_pulled_margin_with_gate(
        upstream_prepared,
        &downstream,
        conv_input_shape,
        conv_weights,
        conv_bias,
        conv_spec,
        batch_norm_spec,
        nominal_batch_norm_scale,
        nominal_batch_norm_bias,
        batch_norm_limits,
        plan,
        downstream_live_bytes,
        gate,
    )?;
    gate.checkpoint("transactional retained exact pullback complete")?;

    if gate.is_enforcing() {
        let mut peak = ConstrainedZonotopePeakLiveBytes::new();
        peak.add_bytes(
            downstream_live_bytes,
            "retained transactional downstream result bytes",
        )?;
        peak.add_bytes(
            exact_relu_tail_margin_live_bytes(upstream_prepared.value_dim())?,
            "retained transactional pulled-margin bytes",
        )?;
        peak.add_bytes(
            relu_tail_peak_live_bytes(
                upstream_prepared.domain,
                upstream_prepared.generator_nonzeros,
            )?,
            "transactional retained upstream M17 peak bytes",
        )?;
        peak.add_bytes(
            size_of::<ReluTailConvBatchNormPullbackM17M20Result>(),
            "transactional retained pullback result header bytes",
        )?;
        gate.preflight_peak_live_bytes(peak.finish())?;
    }
    gate.checkpoint("transactional retained upstream M17 peak-memory preflight complete")?;
    let upstream_original = bound_prepared_internally_pulled_relu_tail_margin_impl(
        upstream_prepared,
        &pulled_margin,
        upstream_supplied_multipliers,
        upstream_config,
        gate,
    )
    .map_err(map_relu_tail_pullback_budget_error)?;
    gate.checkpoint("transactional retained upstream M17 complete")?;

    // All work unique to M20, including checked accounting arithmetic, lives
    // inside one optional boundary. No overflow or firewall refusal here may
    // suppress the already completed mandatory upstream M17 certificate.
    let optional_attempt = (|| -> Result<ReluTailDualResult, ReluTailDualBudgetError> {
        let mut overlap = ConstrainedZonotopePeakLiveBytes::new();
        overlap
            .add_bytes(
                downstream_live_bytes,
                "retained transactional downstream result bytes during M20",
            )
            .map_err(ReluTailDualBudgetError::from)?;
        overlap
            .add_bytes(
                exact_relu_tail_margin_live_bytes(upstream_prepared.value_dim())?,
                "retained transactional pulled-margin bytes during M20",
            )
            .map_err(ReluTailDualBudgetError::from)?;
        overlap
            .add_bytes(
                relu_tail_dual_result_live_bytes(
                    upstream_prepared.value_dim(),
                    upstream_prepared.domain.constraint_count(),
                )?,
                "retained transactional upstream M17 result bytes during M20",
            )
            .map_err(ReluTailDualBudgetError::from)?;
        overlap
            .add_bytes(
                size_of::<ReluTailConvBatchNormPullbackM17M20Result>(),
                "transactional retained pullback result header bytes during M20",
            )
            .map_err(ReluTailDualBudgetError::from)?;
        bound_prepared_internally_pulled_relu_tail_margin_with_auxiliary_impl(
            upstream_prepared,
            upstream_auxiliary,
            &pulled_margin,
            upstream_supplied_multipliers,
            upstream_config,
            overlap.finish(),
            gate,
        )
    })();
    let (upstream_auxiliary_result, optional_budget_error) = match optional_attempt {
        Ok(result) => (Some(result), None),
        Err(ReluTailDualBudgetError::Budget(error)) => (None, Some(error)),
        Err(ReluTailDualBudgetError::Bound(_)) => (None, None),
    };
    let status = if upstream_auxiliary_result.is_some() {
        ReluTailBoxCutStatus::Completed
    } else {
        ReluTailBoxCutStatus::AuxiliaryFallback
    };
    drop(pulled_margin);
    let upstream =
        finish_box_cut_portfolio(upstream_original, upstream_auxiliary_result, None, status);

    // Deliberately no deadline checkpoint after optional M20: an exhausted
    // optional lane must publish the already completed mandatory M17 result.
    Ok(ReluTailConvBatchNormPullbackM17M20Result {
        downstream,
        upstream,
        optional_budget_error,
        plan,
    })
}

fn map_relu_tail_pullback_budget_error(
    error: ReluTailDualBudgetError,
) -> ReluTailConvBatchNormPullbackBudgetError {
    match error {
        ReluTailDualBudgetError::Bound(error) => {
            ReluTailConvBatchNormPullbackError::ReluTail(error).into()
        }
        ReluTailDualBudgetError::Budget(error) => error.into(),
    }
}

fn map_batch_norm_pullback_budget_error(
    error: ConstrainedZonotopeBatchNormBudgetError,
) -> ReluTailConvBatchNormPullbackBudgetError {
    match error {
        ConstrainedZonotopeBatchNormBudgetError::Transform(error) => {
            ReluTailConvBatchNormPullbackError::BatchNorm(error).into()
        }
        ConstrainedZonotopeBatchNormBudgetError::Budget(error) => error.into(),
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_conv2d_batch_norm_pullback_with_gate<G>(
    downstream_prepared: &PreparedReluTailGeometry64<'_>,
    upstream_prepared: &PreparedReluTailGeometry64<'_>,
    input_shape: [usize; 3],
    weights: ArrayView4<'_, f64>,
    bias: &[f64],
    spec: ConstrainedZonotopeConv2dSpec,
    batch_norm_spec: ConstrainedZonotopeBatchNormSpec<'_>,
    limits: ReluTailConvBatchNormPullbackLimits,
    gate: &mut G,
) -> Result<ReluTailConvBatchNormPullbackPlan, ReluTailConvBatchNormPullbackBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let input_value_count = pullback_checked_product(&input_shape, "pullback input value count")?;
    if upstream_prepared.value_dim() != input_value_count {
        return Err(ConstrainedZonotopeConv2dError::Shape {
            field: "upstream prepared geometry",
            expected: vec![input_value_count],
            got: vec![upstream_prepared.value_dim()],
        }
        .into());
    }
    pullback_check_limit(
        "input value count",
        input_value_count,
        limits.max_input_value_count,
    )?;
    pullback_check_hard_limit("input value count", input_value_count)?;

    let [input_channels, input_height, input_width] = input_shape;
    if input_channels == 0 || input_height == 0 || input_width == 0 {
        return Err(ConstrainedZonotopeConv2dError::InvalidSpec {
            message: format!("input shape must be non-empty, got {input_shape:?}"),
        }
        .into());
    }
    if batch_norm_spec.input_shape != input_shape {
        return Err(ConstrainedZonotopeBatchNormError::Shape {
            field: "BatchNorm input shape for Conv2d pullback",
            expected: input_shape.to_vec(),
            got: batch_norm_spec.input_shape.to_vec(),
        }
        .into());
    }
    if batch_norm_spec.channel_axis != 0 {
        return Err(ConstrainedZonotopeBatchNormError::InvalidSpec {
            message: format!(
                "Conv2d pullback requires channel-major axis 0, got {}",
                batch_norm_spec.channel_axis
            ),
        }
        .into());
    }
    if spec.groups == 0 {
        return Err(ConstrainedZonotopeConv2dError::InvalidSpec {
            message: "groups must be non-zero".to_string(),
        }
        .into());
    }
    if spec.stride.contains(&0) || spec.dilation.contains(&0) {
        return Err(ConstrainedZonotopeConv2dError::InvalidSpec {
            message: format!(
                "stride and dilation must be non-zero, got {:?} and {:?}",
                spec.stride, spec.dilation
            ),
        }
        .into());
    }

    let weight_shape_slice = weights.shape();
    let weight_shape = [
        weight_shape_slice[0],
        weight_shape_slice[1],
        weight_shape_slice[2],
        weight_shape_slice[3],
    ];
    let [output_channels, kernel_input_channels, kernel_height, kernel_width] = weight_shape;
    if output_channels == 0 || kernel_input_channels == 0 || kernel_height == 0 || kernel_width == 0
    {
        return Err(ConstrainedZonotopeConv2dError::InvalidSpec {
            message: format!("weight shape must be non-empty, got {weight_shape:?}"),
        }
        .into());
    }
    if input_channels % spec.groups != 0 || output_channels % spec.groups != 0 {
        return Err(ConstrainedZonotopeConv2dError::InvalidSpec {
            message: format!(
                "input/output channels {input_channels}/{output_channels} must be divisible by groups {}",
                spec.groups
            ),
        }
        .into());
    }
    let expected_kernel_input_channels = input_channels / spec.groups;
    if kernel_input_channels != expected_kernel_input_channels {
        return Err(ConstrainedZonotopeConv2dError::Shape {
            field: "weight input channels per group",
            expected: vec![expected_kernel_input_channels],
            got: vec![kernel_input_channels],
        }
        .into());
    }
    if bias.len() != output_channels {
        return Err(ConstrainedZonotopeConv2dError::Shape {
            field: "bias",
            expected: vec![output_channels],
            got: vec![bias.len()],
        }
        .into());
    }

    let weight_elements = pullback_checked_product(&weight_shape, "pullback weight elements")?;
    pullback_check_limit(
        "weight elements",
        weight_elements,
        limits.max_weight_elements,
    )?;
    pullback_check_hard_work_limit("weight elements", weight_elements)?;
    for (index, value) in weights.iter().copied().enumerate() {
        gate.charge_items(1, "transactional pullback weight validation")?;
        if !value.is_finite() {
            return Err(ConstrainedZonotopeConv2dError::NonFinite {
                field: "weights",
                index,
            }
            .into());
        }
    }
    for (index, &value) in bias.iter().enumerate() {
        gate.charge_items(1, "transactional pullback bias validation")?;
        if !value.is_finite() {
            return Err(ConstrainedZonotopeConv2dError::NonFinite {
                field: "bias",
                index,
            }
            .into());
        }
    }

    let output_height = conv2d_output_dimension(
        input_height,
        spec.padding[0],
        spec.padding[2],
        kernel_height,
        spec.dilation[0],
        spec.stride[0],
        "pullback output height",
    )?;
    let output_width = conv2d_output_dimension(
        input_width,
        spec.padding[1],
        spec.padding[3],
        kernel_width,
        spec.dilation[1],
        spec.stride[1],
        "pullback output width",
    )?;
    let output_shape = [output_channels, output_height, output_width];
    let output_value_count =
        pullback_checked_product(&output_shape, "pullback output value count")?;
    if downstream_prepared.value_dim() != output_value_count {
        return Err(ConstrainedZonotopeConv2dError::Shape {
            field: "downstream prepared geometry",
            expected: vec![output_value_count],
            got: vec![downstream_prepared.value_dim()],
        }
        .into());
    }
    pullback_check_limit(
        "output value count",
        output_value_count,
        limits.max_output_value_count,
    )?;
    pullback_check_hard_limit("output value count", output_value_count)?;
    let kernel_visits = pullback_checked_product(
        &[
            output_value_count,
            kernel_input_channels,
            kernel_height,
            kernel_width,
        ],
        "pullback kernel visits",
    )?;
    pullback_check_limit("kernel visits", kernel_visits, limits.max_kernel_visits)?;
    pullback_check_hard_work_limit("kernel visits", kernel_visits)?;

    let pulled_margin_construction_exact_product_bound = kernel_visits
        .checked_add(output_value_count)
        .and_then(|count| {
            input_value_count
                .checked_mul(3)
                .and_then(|n| count.checked_add(n))
        })
        .and_then(|count| {
            input_channels
                .checked_mul(3)
                .and_then(|n| count.checked_add(n))
        })
        .ok_or(ConstrainedZonotopeConv2dError::ResourceOverflow {
            operation: "post-certificate pulled-margin exact product bound",
        })?;
    pullback_check_limit(
        "post-certificate pulled-margin exact rational products",
        pulled_margin_construction_exact_product_bound,
        limits.max_pulled_margin_construction_exact_products,
    )?;
    pullback_check_hard_work_limit(
        "post-certificate pulled-margin exact rational products",
        pulled_margin_construction_exact_product_bound,
    )?;

    Ok(ReluTailConvBatchNormPullbackPlan {
        input_shape,
        output_shape,
        weight_shape,
        weight_elements,
        kernel_visits,
        pulled_margin_construction_exact_product_bound,
    })
}

fn pullback_checked_product(
    factors: &[usize],
    operation: &'static str,
) -> Result<usize, ConstrainedZonotopeConv2dError> {
    factors.iter().try_fold(1_usize, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or(ConstrainedZonotopeConv2dError::ResourceOverflow { operation })
    })
}

fn pullback_check_limit(
    resource: &'static str,
    required: usize,
    limit: usize,
) -> Result<(), ConstrainedZonotopeConv2dError> {
    if required > limit {
        Err(ConstrainedZonotopeConv2dError::ResourceLimit {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}

fn pullback_check_hard_limit(
    resource: &'static str,
    required: usize,
) -> Result<(), ConstrainedZonotopeConv2dError> {
    pullback_check_limit(resource, required, RELU_TAIL_DUAL_HARD_MAX_VALUE_DIM)
}

fn pullback_check_hard_work_limit(
    resource: &'static str,
    required: usize,
) -> Result<(), ConstrainedZonotopeConv2dError> {
    let hard = usize::try_from(RELU_TAIL_DUAL_HARD_MAX_BASELINE_TERMS).unwrap_or(usize::MAX);
    pullback_check_limit(resource, required, hard)
}

#[allow(clippy::too_many_arguments)]
fn build_conv2d_batch_norm_pulled_margin_with_gate<G>(
    upstream_prepared: &PreparedReluTailGeometry64<'_>,
    downstream: &ReluTailDualResult,
    input_shape: [usize; 3],
    weights: ArrayView4<'_, f64>,
    conv_bias: &[f64],
    conv_spec: ConstrainedZonotopeConv2dSpec,
    batch_norm_spec: ConstrainedZonotopeBatchNormSpec<'_>,
    nominal_batch_norm_scale: &[f64],
    nominal_batch_norm_bias: &[f64],
    batch_norm_limits: ConstrainedZonotopeBatchNormAffineCertificateLimits,
    plan: ReluTailConvBatchNormPullbackPlan,
    downstream_live_bytes: usize,
    gate: &mut G,
) -> Result<InternallyPulledReluTailMargin, ReluTailConvBatchNormPullbackBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let output_value_count = plan.output_shape[0]
        .checked_mul(plan.output_shape[1])
        .and_then(|count| count.checked_mul(plan.output_shape[2]))
        .ok_or(ConstrainedZonotopeConv2dError::ResourceOverflow {
            operation: "pullback downstream direction length",
        })?;
    if downstream.direction.len() != output_value_count {
        return Err(ReluTailDualError::Shape {
            field: "accepted downstream direction",
            expected: output_value_count,
            got: downstream.direction.len(),
        }
        .into());
    }

    let channel_count = input_shape[0];
    if gate.is_enforcing() {
        let mut peak = ConstrainedZonotopePeakLiveBytes::new();
        peak.add_bytes(
            downstream_live_bytes,
            "retained downstream result during BatchNorm certification",
        )?;
        peak.add_bytes(
            batch_norm_affine_certificate_peak_live_bytes(channel_count)?,
            "transactional BatchNorm certificate peak bytes",
        )?;
        peak.add_bytes(
            size_of::<ReluTailConvBatchNormPullbackResult>(),
            "transactional pullback result header bytes",
        )?;
        gate.preflight_peak_live_bytes(peak.finish())?;
    }
    gate.checkpoint("transactional BatchNorm certificate preflight complete")?;
    let certificate = certify_batch_norm_affine_surrogate_impl(
        batch_norm_spec,
        nominal_batch_norm_scale,
        nominal_batch_norm_bias,
        batch_norm_limits,
        gate,
    )
    .map_err(map_batch_norm_pullback_budget_error)?;
    gate.checkpoint("transactional BatchNorm certificate complete")?;

    if gate.is_enforcing() {
        let mut peak = ConstrainedZonotopePeakLiveBytes::new();
        peak.add_bytes(
            downstream_live_bytes,
            "retained downstream result during exact pullback",
        )?;
        peak.add_bytes(
            certificate.conservative_live_bytes(),
            "retained BatchNorm certificate during exact pullback",
        )?;
        peak.add_bytes(
            exact_relu_tail_margin_peak_live_bytes(upstream_prepared.value_dim())?,
            "transactional exact pulled-margin construction bytes",
        )?;
        peak.add_bytes(
            size_of::<ReluTailConvBatchNormPullbackResult>(),
            "transactional pullback result header bytes",
        )?;
        gate.preflight_peak_live_bytes(peak.finish())?;
    }
    gate.checkpoint("transactional exact pullback peak-memory preflight complete")?;

    let input_value_count = upstream_prepared.value_dim();
    let mut coefficients = Vec::new();
    try_reserve(
        &mut coefficients,
        input_value_count,
        "transactional pulled-margin coefficients",
    )?;
    for _ in 0..input_value_count {
        gate.charge_items(1, "transactional pulled-margin initialization")?;
        coefficients.push(BigRational::zero());
    }
    let mut exact_bias = checked_rational(
        downstream.exact_constant.clone(),
        "downstream exact-constant pullback",
        0,
    )?;

    let [input_channels, input_height, input_width] = input_shape;
    let [output_channels, kernel_input_channels, kernel_height, kernel_width] = plan.weight_shape;
    let [_output_channels, output_height, output_width] = plan.output_shape;
    let input_channels_per_group = input_channels / conv_spec.groups;
    let output_channels_per_group = output_channels / conv_spec.groups;

    for output_channel in 0..output_channels {
        let group = output_channel / output_channels_per_group;
        let input_channel_base = group * input_channels_per_group;
        let exact_conv_bias = exact_objective_f64(
            conv_bias[output_channel],
            "convolution bias",
            output_channel,
        )?;
        for output_y in 0..output_height {
            for output_x in 0..output_width {
                gate.charge_items(1, "transactional Conv2d transpose output")?;
                let output_index =
                    (output_channel * output_height + output_y) * output_width + output_x;
                let exact_direction = exact_objective_f64(
                    downstream.direction[output_index],
                    "accepted downstream direction",
                    output_index,
                )?;
                let bias_contribution = checked_rational(
                    &exact_direction * &exact_conv_bias,
                    "Conv2d bias pullback product",
                    output_index,
                )?;
                exact_bias = checked_rational(
                    exact_bias + bias_contribution,
                    "Conv2d bias pullback accumulation",
                    output_index,
                )?;

                for kernel_input_channel in 0..kernel_input_channels {
                    let input_channel = input_channel_base + kernel_input_channel;
                    for kernel_y in 0..kernel_height {
                        let input_y = conv2d_input_coordinate(
                            output_y,
                            conv_spec.stride[0],
                            kernel_y,
                            conv_spec.dilation[0],
                            conv_spec.padding[0],
                            input_height,
                        )?;
                        for kernel_x in 0..kernel_width {
                            gate.charge_items(1, "transactional Conv2d exact transpose")?;
                            let Some(input_y) = input_y else {
                                continue;
                            };
                            let Some(input_x) = conv2d_input_coordinate(
                                output_x,
                                conv_spec.stride[1],
                                kernel_x,
                                conv_spec.dilation[1],
                                conv_spec.padding[1],
                                input_width,
                            )?
                            else {
                                continue;
                            };
                            let weight =
                                weights[[output_channel, kernel_input_channel, kernel_y, kernel_x]];
                            if weight == 0.0 || exact_direction.is_zero() {
                                continue;
                            }
                            let input_index =
                                (input_channel * input_height + input_y) * input_width + input_x;
                            let exact_weight = exact_objective_f64(
                                weight,
                                "convolution weights",
                                (((output_channel * kernel_input_channels + kernel_input_channel)
                                    * kernel_height
                                    + kernel_y)
                                    * kernel_width)
                                    + kernel_x,
                            )?;
                            let contribution = checked_rational(
                                &exact_direction * exact_weight,
                                "Conv2d transpose product",
                                input_index,
                            )?;
                            coefficients[input_index] = checked_rational(
                                coefficients[input_index].clone() + contribution,
                                "Conv2d transpose accumulation",
                                input_index,
                            )?;
                        }
                    }
                }
            }
        }
    }
    gate.checkpoint("transactional exact Conv2d transpose complete")?;

    let channel_stride = input_height.checked_mul(input_width).ok_or(
        ConstrainedZonotopeConv2dError::ResourceOverflow {
            operation: "pullback channel stride",
        },
    )?;
    for channel in 0..input_channels {
        gate.charge_items(1, "transactional BatchNorm channel pullback")?;
        let channel_start = channel * channel_stride;
        let channel_end = channel_start + channel_stride;
        let mut coefficient_sum = BigRational::zero();
        let mut activation_min = BigRational::zero();
        let mut activation_max = BigRational::zero();
        for coordinate in channel_start..channel_end {
            gate.charge_items(1, "transactional BatchNorm error envelope")?;
            let coefficient = &coefficients[coordinate];
            coefficient_sum = checked_rational(
                coefficient_sum + coefficient,
                "BatchNorm shared coefficient sum",
                coordinate,
            )?;
            let (lower, upper) = &upstream_prepared.exact_coordinate_bounds[coordinate];
            let relu_lower = if lower.is_negative() {
                BigRational::zero()
            } else {
                lower.clone()
            };
            let relu_upper = if upper.is_negative() {
                BigRational::zero()
            } else {
                upper.clone()
            };
            let (minimum_product, maximum_product) = if coefficient.is_negative() {
                (
                    checked_rational(
                        coefficient * &relu_upper,
                        "BatchNorm scale-error lower envelope",
                        coordinate,
                    )?,
                    checked_rational(
                        coefficient * &relu_lower,
                        "BatchNorm scale-error upper envelope",
                        coordinate,
                    )?,
                )
            } else {
                (
                    checked_rational(
                        coefficient * &relu_lower,
                        "BatchNorm scale-error lower envelope",
                        coordinate,
                    )?,
                    checked_rational(
                        coefficient * &relu_upper,
                        "BatchNorm scale-error upper envelope",
                        coordinate,
                    )?,
                )
            };
            activation_min = checked_rational(
                activation_min + minimum_product,
                "BatchNorm scale-error lower accumulation",
                coordinate,
            )?;
            activation_max = checked_rational(
                activation_max + maximum_product,
                "BatchNorm scale-error upper accumulation",
                coordinate,
            )?;
        }

        let channel_certificate = &certificate.channels()[channel];
        if channel_certificate.scale_error().is_negative()
            || channel_certificate.bias_error().is_negative()
        {
            return Err(ConstrainedZonotopeBatchNormError::InvalidSpec {
                message: "internal affine-surrogate certificate error is negative".to_string(),
            }
            .into());
        }
        let nominal_scale = exact_objective_f64(
            nominal_batch_norm_scale[channel],
            "nominal BatchNorm scale",
            channel,
        )?;
        let nominal_bias = exact_objective_f64(
            nominal_batch_norm_bias[channel],
            "nominal BatchNorm bias",
            channel,
        )?;
        let activation_abs = activation_min.abs().max(activation_max.abs());
        let scale_penalty = checked_rational(
            channel_certificate.scale_error() * &activation_abs,
            "BatchNorm shared scale-error penalty",
            channel,
        )?;
        let bias_penalty = checked_rational(
            channel_certificate.bias_error() * coefficient_sum.abs(),
            "BatchNorm shared bias-error penalty",
            channel,
        )?;
        let nominal_bias_contribution = checked_rational(
            &coefficient_sum * nominal_bias,
            "BatchNorm nominal bias pullback",
            channel,
        )?;
        exact_bias = checked_rational(
            exact_bias + nominal_bias_contribution,
            "BatchNorm nominal bias accumulation",
            channel,
        )?;
        exact_bias = checked_rational(
            exact_bias - scale_penalty,
            "BatchNorm scale-error subtraction",
            channel,
        )?;
        exact_bias = checked_rational(
            exact_bias - bias_penalty,
            "BatchNorm bias-error subtraction",
            channel,
        )?;

        for coordinate in channel_start..channel_end {
            gate.charge_items(1, "transactional BatchNorm nominal scale pullback")?;
            coefficients[coordinate] = checked_rational(
                coefficients[coordinate].clone() * &nominal_scale,
                "BatchNorm nominal scale product",
                coordinate,
            )?;
        }
    }
    gate.checkpoint("transactional certified BatchNorm pullback complete")?;
    drop(certificate);
    gate.checkpoint("transactional pulled-margin declared-rational validation")?;
    check_internally_pulled_rationals_with_gate(&coefficients, &exact_bias, gate)
        .map_err(map_relu_tail_pullback_budget_error)?;
    gate.checkpoint("transactional pulled-margin declared-rational validation complete")?;
    Ok(InternallyPulledReluTailMargin {
        margin: ExactReluTailMargin {
            coefficients,
            bias: exact_bias,
        },
    })
}

fn bound_prepared_relu_tail_margin_impl<G>(
    prepared: &PreparedReluTailGeometry64<'_>,
    margin: &ExactReluTailMargin,
    supplied_multipliers: Option<&[f64]>,
    config: ReluTailDualConfig,
    gate: &mut G,
) -> Result<ReluTailDualResult, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    gate.checkpoint("prepared ReLU-tail margin validation")?;
    check_mandatory_margin_resources_with_gate(prepared.domain.value_dim(), margin, gate)?;
    gate.checkpoint("prepared ReLU-tail margin validation complete")?;
    bound_prepared_validated_relu_tail_margin_impl(
        prepared,
        margin,
        supplied_multipliers,
        config,
        gate,
    )
}

fn bound_prepared_internally_pulled_relu_tail_margin_impl<G>(
    prepared: &PreparedReluTailGeometry64<'_>,
    pulled_margin: &InternallyPulledReluTailMargin,
    supplied_multipliers: Option<&[f64]>,
    config: ReluTailDualConfig,
    gate: &mut G,
) -> Result<ReluTailDualResult, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let margin = pulled_margin.as_exact_margin();
    gate.checkpoint("prepared ReLU-tail margin validation")?;
    if margin.coefficients.len() != prepared.domain.value_dim() {
        return Err(ReluTailDualError::Shape {
            field: "margin coefficients",
            expected: prepared.domain.value_dim(),
            got: margin.coefficients.len(),
        }
        .into());
    }
    check_internally_pulled_rationals_with_gate(&margin.coefficients, &margin.bias, gate)?;
    gate.checkpoint("prepared ReLU-tail margin validation complete")?;
    bound_prepared_validated_relu_tail_margin_impl(
        prepared,
        margin,
        supplied_multipliers,
        config,
        gate,
    )
}

fn bound_prepared_internally_pulled_relu_tail_margin_with_auxiliary_impl<G>(
    prepared: &PreparedReluTailGeometry64<'_>,
    auxiliary: &CertifiedAuxiliaryBounds64,
    pulled_margin: &InternallyPulledReluTailMargin,
    supplied_multipliers: Option<&[f64]>,
    config: ReluTailDualConfig,
    overlapping_retained_bytes: usize,
    gate: &mut G,
) -> Result<ReluTailDualResult, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    gate.checkpoint("prepared internally pulled auxiliary ReLU-tail validation")?;
    if auxiliary.value_dim() != prepared.domain.value_dim() {
        return Err(ReluTailDualError::AuxiliaryDimensionMismatch {
            expected: prepared.domain.value_dim(),
            got: auxiliary.value_dim(),
        }
        .into());
    }
    let margin = pulled_margin.as_exact_margin();
    if margin.coefficients.len() != prepared.domain.value_dim() {
        return Err(ReluTailDualError::Shape {
            field: "margin coefficients",
            expected: prepared.domain.value_dim(),
            got: margin.coefficients.len(),
        }
        .into());
    }
    check_internally_pulled_rationals_with_gate(&margin.coefficients, &margin.bias, gate)?;
    gate.checkpoint("prepared internally pulled auxiliary ReLU-tail validation complete")?;
    if gate.is_enforcing() {
        let transform_peak =
            relu_tail_peak_live_bytes(prepared.domain, prepared.generator_nonzeros)?
                .checked_add(overlapping_retained_bytes)
                .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                    operation:
                        "prepared internally pulled auxiliary ReLU-tail overlapping retained bytes",
                })?;
        gate.preflight_peak_live_bytes(transform_peak)?;
    }
    gate.checkpoint(
        "prepared internally pulled auxiliary ReLU-tail peak-memory preflight complete",
    )?;
    let bounds = exact_coordinate_bounds_with_auxiliary_from_bounds_with_gate(
        &prepared.exact_coordinate_bounds,
        auxiliary,
        gate,
    )?;
    gate.checkpoint("prepared internally pulled auxiliary ReLU-tail intersection complete")?;
    let line_plan = build_line_plan_from_bounds_with_gate(prepared.domain, margin, &bounds, gate)?;
    gate.checkpoint(
        "prepared internally pulled auxiliary ReLU-tail exact line construction complete",
    )?;
    let result = bound_relu_tail_triangle_dual_from_line_plan_with_gate(
        prepared.domain,
        line_plan,
        prepared.generator_nonzeros,
        supplied_multipliers,
        config,
        gate,
    )?;
    gate.checkpoint("prepared internally pulled auxiliary ReLU-tail mandatory replay complete")?;
    Ok(result)
}

fn bound_prepared_validated_relu_tail_margin_impl<G>(
    prepared: &PreparedReluTailGeometry64<'_>,
    margin: &ExactReluTailMargin,
    supplied_multipliers: Option<&[f64]>,
    config: ReluTailDualConfig,
    gate: &mut G,
) -> Result<ReluTailDualResult, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    if gate.is_enforcing() {
        gate.preflight_peak_live_bytes(relu_tail_peak_live_bytes(
            prepared.domain,
            prepared.generator_nonzeros,
        )?)?;
    }
    gate.checkpoint("prepared ReLU-tail margin peak-memory preflight complete")?;
    let line_plan = build_line_plan_from_bounds_with_gate(
        prepared.domain,
        margin,
        &prepared.exact_coordinate_bounds,
        gate,
    )?;
    gate.checkpoint("prepared ReLU-tail exact line construction complete")?;
    let result = bound_relu_tail_triangle_dual_from_line_plan_with_gate(
        prepared.domain,
        line_plan,
        prepared.generator_nonzeros,
        supplied_multipliers,
        config,
        gate,
    )?;
    gate.checkpoint("prepared ReLU-tail mandatory replay complete")?;
    Ok(result)
}

fn bound_prepared_relu_tail_margin_with_auxiliary_impl<G>(
    prepared: &PreparedReluTailGeometry64<'_>,
    auxiliary: &CertifiedAuxiliaryBounds64,
    margin: &ExactReluTailMargin,
    supplied_multipliers: Option<&[f64]>,
    config: ReluTailDualConfig,
    overlapping_retained_bytes: usize,
    gate: &mut G,
) -> Result<ReluTailDualResult, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    gate.checkpoint("prepared auxiliary ReLU-tail validation")?;
    if auxiliary.value_dim() != prepared.domain.value_dim() {
        return Err(ReluTailDualError::AuxiliaryDimensionMismatch {
            expected: prepared.domain.value_dim(),
            got: auxiliary.value_dim(),
        }
        .into());
    }
    check_mandatory_margin_resources_with_gate(prepared.domain.value_dim(), margin, gate)?;
    gate.checkpoint("prepared auxiliary ReLU-tail validation complete")?;
    if gate.is_enforcing() {
        let transform_peak =
            relu_tail_peak_live_bytes(prepared.domain, prepared.generator_nonzeros)?
                .checked_add(overlapping_retained_bytes)
                .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                    operation: "prepared auxiliary ReLU-tail overlapping retained bytes",
                })?;
        gate.preflight_peak_live_bytes(transform_peak)?;
    }
    gate.checkpoint("prepared auxiliary ReLU-tail peak-memory preflight complete")?;
    let bounds = exact_coordinate_bounds_with_auxiliary_from_bounds_with_gate(
        &prepared.exact_coordinate_bounds,
        auxiliary,
        gate,
    )?;
    gate.checkpoint("prepared auxiliary ReLU-tail intersection complete")?;
    let line_plan = build_line_plan_from_bounds_with_gate(prepared.domain, margin, &bounds, gate)?;
    gate.checkpoint("prepared auxiliary ReLU-tail exact line construction complete")?;
    let result = bound_relu_tail_triangle_dual_from_line_plan_with_gate(
        prepared.domain,
        line_plan,
        prepared.generator_nonzeros,
        supplied_multipliers,
        config,
        gate,
    )?;
    gate.checkpoint("prepared auxiliary ReLU-tail mandatory replay complete")?;
    Ok(result)
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
    let mut gate = InertConstrainedZonotopeCallGate;
    match bound_relu_tail_triangle_dual_impl(
        domain,
        margin,
        supplied_multipliers,
        config,
        &mut gate,
    ) {
        Ok(result) => Ok(result),
        Err(ReluTailDualBudgetError::Bound(error)) => Err(error),
        Err(ReluTailDualBudgetError::Budget(_)) => {
            unreachable!("the inert ReLU-tail call gate cannot refuse work")
        }
    }
}

/// Bound one exact ReLU-tail margin behind the shared synchronous call
/// firewall.
///
/// The preflight covers exact coordinate geometry, exact line construction,
/// all candidate/search buffers, result storage, and the nested sparse-dual
/// error allowance. `budget.baseline_live_bytes()` must include `domain`,
/// `margin`, optional supplied multipliers, and every other caller-owned buffer
/// sharing the same ceiling.
pub fn bound_relu_tail_triangle_dual_unwired_with_budget(
    domain: &ConstrainedZonotope64,
    margin: &ExactReluTailMargin,
    supplied_multipliers: Option<&[f64]>,
    config: ReluTailDualConfig,
    budget: ConstrainedZonotopeCallBudget,
) -> Result<ConstrainedZonotopeCallOutcome<ReluTailDualResult>, ReluTailDualBudgetError> {
    let mut gate = ConstrainedZonotopeCallTracker::from_system_clock(budget)?;
    let result = bound_relu_tail_triangle_dual_impl(
        domain,
        margin,
        supplied_multipliers,
        config,
        &mut gate,
    )?;
    Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()))
}

#[cfg(test)]
fn bound_relu_tail_triangle_dual_unwired_with_clock<N>(
    domain: &ConstrainedZonotope64,
    margin: &ExactReluTailMargin,
    supplied_multipliers: Option<&[f64]>,
    config: ReluTailDualConfig,
    budget: ConstrainedZonotopeCallBudget,
    now: N,
) -> Result<ConstrainedZonotopeCallOutcome<ReluTailDualResult>, ReluTailDualBudgetError>
where
    N: FnMut(&'static str) -> Instant,
{
    let mut gate = ConstrainedZonotopeCallTracker::with_clock(budget, now)?;
    let result = bound_relu_tail_triangle_dual_impl(
        domain,
        margin,
        supplied_multipliers,
        config,
        &mut gate,
    )?;
    Ok(ConstrainedZonotopeCallOutcome::new(result, gate.report()))
}

fn bound_relu_tail_triangle_dual_impl<G>(
    domain: &ConstrainedZonotope64,
    margin: &ExactReluTailMargin,
    supplied_multipliers: Option<&[f64]>,
    config: ReluTailDualConfig,
    gate: &mut G,
) -> Result<ReluTailDualResult, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    gate.checkpoint("ReLU-tail input validation")?;
    let generator_nonzeros = check_mandatory_resources_with_gate(domain, margin, gate)?;
    gate.checkpoint("ReLU-tail input validation complete")?;
    if gate.is_enforcing() {
        gate.preflight_peak_live_bytes(relu_tail_peak_live_bytes(domain, generator_nonzeros)?)?;
    }
    gate.checkpoint("ReLU-tail peak-memory preflight complete")?;
    let line_plan = build_line_plan_with_gate(domain, margin, gate)?;
    gate.checkpoint("ReLU-tail exact line construction complete")?;
    let result = bound_relu_tail_triangle_dual_from_line_plan_with_gate(
        domain,
        line_plan,
        generator_nonzeros,
        supplied_multipliers,
        config,
        gate,
    )?;
    gate.checkpoint("ReLU-tail certificate publication")?;
    Ok(result)
}

fn bound_relu_tail_triangle_dual_from_line_plan(
    domain: &ConstrainedZonotope64,
    line_plan: LinePlan,
    generator_nonzeros: usize,
    supplied_multipliers: Option<&[f64]>,
    config: ReluTailDualConfig,
) -> Result<ReluTailDualResult, ReluTailDualError> {
    let mut gate = InertConstrainedZonotopeCallGate;
    match bound_relu_tail_triangle_dual_from_line_plan_with_gate(
        domain,
        line_plan,
        generator_nonzeros,
        supplied_multipliers,
        config,
        &mut gate,
    ) {
        Ok(result) => Ok(result),
        Err(ReluTailDualBudgetError::Bound(error)) => Err(error),
        Err(ReluTailDualBudgetError::Budget(_)) => {
            unreachable!("the inert ReLU-tail call gate cannot refuse work")
        }
    }
}

fn bound_relu_tail_triangle_dual_from_line_plan_with_gate<G>(
    domain: &ConstrainedZonotope64,
    line_plan: LinePlan,
    generator_nonzeros: usize,
    supplied_multipliers: Option<&[f64]>,
    config: ReluTailDualConfig,
    gate: &mut G,
) -> Result<ReluTailDualResult, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let mut zero = zero_f64_with_gate(
        domain.constraint_count(),
        "zero multipliers",
        "ReLU-tail zero-multiplier initialization",
        gate,
    )?;
    // Canonicalize every structural zero, including platforms which preserve
    // a negative-zero payload through allocation helpers.
    for value in &mut zero {
        gate.charge_items(1, "ReLU-tail zero canonicalization")?;
        *value = 0.0;
    }

    let supplied = valid_supplied_multipliers_with_gate(
        supplied_multipliers,
        domain.constraint_count(),
        gate,
    )?;
    let baseline_direction = clone_f64_with_gate(
        &line_plan.fixed_direction,
        "baseline direction",
        "ReLU-tail baseline-direction clone",
        gate,
    )?;

    // This call is deliberately first.  Search configuration and supplied
    // multiplier defects cannot hide a failure of the authority path.
    let baseline_zero_raw = match evaluate_constrained_zonotope64_dual_with_call_gate(
        domain,
        &baseline_direction,
        &zero,
        gate,
    ) {
        Ok(bounds) => bounds,
        Err(ConstrainedZonotopeDualBudgetError::Evaluation(error)) => {
            return Err(ReluTailDualError::Baseline(error.into()).into());
        }
        Err(ConstrainedZonotopeDualBudgetError::Budget(error)) => return Err(error.into()),
    };
    let baseline_zero = combine_exact_lower(baseline_zero_raw.lower, &line_plan.exact_constant)?;
    let mut best = AcceptedCandidate {
        lower: baseline_zero,
        zero_lower: baseline_zero,
        direction: baseline_direction,
        multipliers: clone_f64_with_gate(
            &zero,
            "accepted zero multipliers",
            "ReLU-tail accepted-multiplier clone",
            gate,
        )?,
        supplied_used: false,
    };
    let mut candidate_replays = ReluTailDualZeroPredicateCandidateReplays {
        zero_positive_slope_lower_bound: baseline_zero,
        upper_endpoint_lower_bound: None,
        canonical_lower_bound: None,
        optimized_lower_bound: None,
    };
    if maybe_replay_supplied_with_gate(
        domain,
        &line_plan.exact_constant,
        supplied,
        &mut best,
        gate,
    )? == SuppliedReplayOutcome::AllocationFallback
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

    let Some(mut endpoint) = candidate_clone_f64_with_gate(
        &line_plan.fixed_direction,
        "ReLU-tail endpoint-direction clone",
        gate,
    )?
    else {
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
        gate.charge_items(1, "ReLU-tail endpoint-direction construction")?;
        endpoint[variable.coordinate] = variable.upper;
    }
    match replay_direction_with_gate(
        domain,
        &line_plan.exact_constant,
        &line_plan.variables,
        endpoint,
        &zero,
        supplied,
        &mut best,
        gate,
    )? {
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

    let Some(mut canonical) = candidate_clone_f64_with_gate(
        &line_plan.fixed_direction,
        "ReLU-tail canonical-direction clone",
        gate,
    )?
    else {
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
        gate.charge_items(1, "ReLU-tail canonical-direction construction")?;
        canonical[variable.coordinate] = variable.canonical;
    }
    let Some(canonical_replay) =
        candidate_clone_f64_with_gate(&canonical, "ReLU-tail canonical replay clone", gate)?
    else {
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
    match replay_direction_with_gate(
        domain,
        &line_plan.exact_constant,
        &line_plan.variables,
        canonical_replay,
        &zero,
        supplied,
        &mut best,
        gate,
    )? {
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

    let search = projected_adam_candidate_with_call_gate(
        domain,
        &line_plan.variables,
        canonical,
        config,
        gate,
    );
    let (status, iterations_completed) = match search {
        Ok((candidate, iterations_completed)) => {
            let replay = replay_direction_with_gate(
                domain,
                &line_plan.exact_constant,
                &line_plan.variables,
                candidate,
                &zero,
                supplied,
                &mut best,
                gate,
            )?;
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
        Err(GatedCandidateFailure::Deadline(iterations_completed)) => {
            (ReluTailDualStatus::Deadline, iterations_completed)
        }
        Err(GatedCandidateFailure::NonFinite(iterations_completed)) => {
            (ReluTailDualStatus::NonFiniteCandidate, iterations_completed)
        }
        Err(GatedCandidateFailure::Allocation(iterations_completed)) => {
            (ReluTailDualStatus::AllocationFallback, iterations_completed)
        }
        Err(GatedCandidateFailure::Budget(error)) => return Err(error.into()),
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

#[allow(clippy::too_many_arguments)]
fn finish_budgeted_optimized_box_cut_result(
    original: ReluTailDualResult,
    auxiliary: Option<ReluTailDualResult>,
    box_cut: Option<ReluTailBoxCutCertificate>,
    search_status: ReluTailBoxCutOptimizerStatus,
    search_plan: Option<ReluTailBoxCutOptimizerPlan>,
    telemetry: BoxSearchTelemetry,
    optional_budget_error: Option<ConstrainedZonotopeCallBudgetError>,
) -> ReluTailBoxCutBudgetedResult {
    ReluTailBoxCutBudgetedResult {
        optimized: finish_optimized_box_cut_result(
            original,
            auxiliary,
            box_cut,
            search_status,
            search_plan,
            telemetry,
        ),
        optional_budget_error,
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

#[cfg(test)]
fn build_line_plan(
    domain: &ConstrainedZonotope64,
    margin: &ExactReluTailMargin,
) -> Result<LinePlan, ReluTailDualError> {
    let mut gate = InertConstrainedZonotopeCallGate;
    match build_line_plan_with_gate(domain, margin, &mut gate) {
        Ok(plan) => Ok(plan),
        Err(ReluTailDualBudgetError::Bound(error)) => Err(error),
        Err(ReluTailDualBudgetError::Budget(_)) => {
            unreachable!("the inert ReLU-tail call gate cannot refuse work")
        }
    }
}

fn build_line_plan_with_gate<G>(
    domain: &ConstrainedZonotope64,
    margin: &ExactReluTailMargin,
    gate: &mut G,
) -> Result<LinePlan, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let bounds = exact_coordinate_bounds_with_gate(domain, gate)?;
    build_line_plan_from_bounds_with_gate(domain, margin, &bounds, gate)
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
    let mut gate = InertConstrainedZonotopeCallGate;
    match build_line_plan_from_bounds_with_gate(domain, margin, bounds, &mut gate) {
        Ok(plan) => Ok(plan),
        Err(ReluTailDualBudgetError::Bound(error)) => Err(error),
        Err(ReluTailDualBudgetError::Budget(_)) => {
            unreachable!("the inert ReLU-tail call gate cannot refuse work")
        }
    }
}

fn build_line_plan_from_bounds_with_gate<G>(
    domain: &ConstrainedZonotope64,
    margin: &ExactReluTailMargin,
    bounds: &[(BigRational, BigRational)],
    gate: &mut G,
) -> Result<LinePlan, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    debug_assert_eq!(bounds.len(), domain.value_dim());
    let mut fixed_direction = zero_f64_with_gate(
        domain.value_dim(),
        "fixed direction",
        "ReLU-tail fixed-direction initialization",
        gate,
    )?;
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
        gate.charge_items(1, "ReLU-tail exact line construction")?;
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
            }
            .into());
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
    let mut gate = InertConstrainedZonotopeCallGate;
    match exact_coordinate_bounds_with_gate(domain, &mut gate) {
        Ok(bounds) => Ok(bounds),
        Err(ReluTailDualBudgetError::Bound(error)) => Err(error),
        Err(ReluTailDualBudgetError::Budget(_)) => {
            unreachable!("the inert ReLU-tail call gate cannot refuse work")
        }
    }
}

fn exact_coordinate_bounds_with_gate<G>(
    domain: &ConstrainedZonotope64,
    gate: &mut G,
) -> Result<Vec<(BigRational, BigRational)>, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    #[cfg(test)]
    EXACT_COORDINATE_HULL_PASSES.with(|passes| passes.set(passes.get() + 1));

    let mut radii = Vec::new();
    try_reserve(&mut radii, domain.value_dim(), "exact coordinate radii")?;
    for (coordinate, &remainder) in domain.box_remainder().iter().enumerate() {
        gate.charge_items(1, "ReLU-tail exact radius initialization")?;
        radii.push(checked_rational(
            exact_domain_f64(remainder, coordinate, "box remainder")?,
            "box remainder",
            coordinate,
        )?);
    }
    for generator in domain.generators() {
        gate.charge_items(1, "ReLU-tail exact generator-column scan")?;
        for (coordinate, coefficient) in generator.entries() {
            gate.charge_items(1, "ReLU-tail exact generator-entry accumulation")?;
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
        gate.charge_items(1, "ReLU-tail exact coordinate-bound materialization")?;
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

fn exact_coordinate_bounds_with_auxiliary_from_bounds_with_gate<G>(
    domain_bounds: &[(BigRational, BigRational)],
    auxiliary: &CertifiedAuxiliaryBounds64,
    gate: &mut G,
) -> Result<Vec<(BigRational, BigRational)>, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    debug_assert_eq!(domain_bounds.len(), auxiliary.value_dim());
    let mut bounds = Vec::new();
    try_reserve(
        &mut bounds,
        domain_bounds.len(),
        "auxiliary exact coordinate bounds",
    )?;
    for (coordinate, ((domain_lower, domain_upper), (&auxiliary_lower, &auxiliary_upper))) in
        domain_bounds
            .iter()
            .zip(auxiliary.lower().iter().zip(auxiliary.upper()))
            .enumerate()
    {
        gate.charge_items(
            1,
            "prepared auxiliary ReLU-tail lower-endpoint intersection",
        )?;
        let mut lower = domain_lower.clone();
        let auxiliary_lower = checked_rational(
            exact_domain_f64(auxiliary_lower, coordinate, "auxiliary lower bound")?,
            "auxiliary lower bound",
            coordinate,
        )?;
        if auxiliary_lower > lower {
            lower = auxiliary_lower;
        }

        gate.charge_items(
            1,
            "prepared auxiliary ReLU-tail upper-endpoint intersection",
        )?;
        let mut upper = domain_upper.clone();
        let auxiliary_upper = checked_rational(
            exact_domain_f64(auxiliary_upper, coordinate, "auxiliary upper bound")?,
            "auxiliary upper bound",
            coordinate,
        )?;
        if auxiliary_upper < upper {
            upper = auxiliary_upper;
        }
        if lower > upper {
            return Err(ReluTailDualError::EmptyAuxiliaryIntersection { coordinate }.into());
        }
        bounds.push((lower, upper));
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

fn count_tighter_auxiliary_box_endpoints_with_gate<G>(
    original_hull: &[(BigRational, BigRational)],
    auxiliary: &CertifiedAuxiliaryBounds64,
    gate: &mut G,
) -> Result<usize, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    debug_assert_eq!(original_hull.len(), auxiliary.value_dim());
    let mut count = 0_usize;
    for (coordinate, ((hull_lower, hull_upper), (&lower, &upper))) in original_hull
        .iter()
        .zip(auxiliary.lower().iter().zip(auxiliary.upper()))
        .enumerate()
    {
        gate.charge_items(2, "M24 tighter auxiliary endpoint count")?;
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
    gate.checkpoint("M24 tighter auxiliary endpoint count complete")?;
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

struct BudgetedBoxApproximateSearch {
    search: BoxApproximateSearch,
    budget_error: Option<ConstrainedZonotopeCallBudgetError>,
}

fn optimize_auxiliary_box_multipliers_with_call_gate<G>(
    domain: &ConstrainedZonotope64,
    original_hull: &[(BigRational, BigRational)],
    auxiliary: &CertifiedAuxiliaryBounds64,
    line_direction: &[f64],
    config: ReluTailBoxCutOptimizerConfig,
    plan: ReluTailBoxCutOptimizerPlan,
    gate: &mut G,
) -> BudgetedBoxApproximateSearch
where
    G: ConstrainedZonotopeCallGate,
{
    let start = Instant::now();
    optimize_auxiliary_box_multipliers_with_clock_and_call_gate(
        domain,
        original_hull,
        auxiliary,
        line_direction,
        config,
        plan,
        || start.elapsed(),
        gate,
    )
}

#[allow(clippy::too_many_arguments)]
fn optimize_auxiliary_box_multipliers_with_clock_and_call_gate<C, G>(
    domain: &ConstrainedZonotope64,
    original_hull: &[(BigRational, BigRational)],
    auxiliary: &CertifiedAuxiliaryBounds64,
    line_direction: &[f64],
    config: ReluTailBoxCutOptimizerConfig,
    plan: ReluTailBoxCutOptimizerPlan,
    elapsed: C,
    gate: &mut G,
) -> BudgetedBoxApproximateSearch
where
    C: FnMut() -> Duration,
    G: ConstrainedZonotopeCallGate,
{
    let mut deadline = CandidateDeadline::new(config.wall_time, elapsed);
    let mut search = BoxApproximateSearch {
        variables: Vec::new(),
        candidates: Vec::new(),
        status: ReluTailBoxCutOptimizerStatus::Completed,
        iterations_completed: 0,
        restarts_completed: 0,
        candidates_scored: 0,
    };
    if let Err(failure) =
        gated_candidate_checkpoint(&mut deadline, 0, gate, "M24 candidate search startup")
    {
        let budget_error = apply_gated_box_search_failure(&mut search, failure);
        return BudgetedBoxApproximateSearch {
            search,
            budget_error,
        };
    }
    search.variables = match collect_box_search_variables_with_gate(
        original_hull,
        auxiliary,
        plan.box_variables,
        &mut deadline,
        gate,
    ) {
        Ok(variables) => variables,
        Err(failure) => {
            let budget_error = apply_gated_box_search_failure(&mut search, failure);
            return BudgetedBoxApproximateSearch {
                search,
                budget_error,
            };
        }
    };
    if search.variables.len() != plan.box_variables {
        search.status = ReluTailBoxCutOptimizerStatus::NonFiniteCandidate;
        return BudgetedBoxApproximateSearch {
            search,
            budget_error: None,
        };
    }
    if search.candidates.try_reserve_exact(plan.restarts).is_err() {
        search.status = ReluTailBoxCutOptimizerStatus::AllocationFallback;
        return BudgetedBoxApproximateSearch {
            search,
            budget_error: None,
        };
    }
    if let Err(failure) =
        gated_candidate_checkpoint(&mut deadline, 0, gate, "M24 candidate storage publication")
    {
        let budget_error = apply_gated_box_search_failure(&mut search, failure);
        return BudgetedBoxApproximateSearch {
            search,
            budget_error,
        };
    }

    for schedule in config
        .schedules
        .iter()
        .copied()
        .filter(|schedule| schedule.iterations > 0)
    {
        let restart = run_box_optimizer_restart_with_gate(
            domain,
            line_direction,
            &search.variables,
            schedule,
            config.multiplier_cap,
            search.iterations_completed,
            &mut deadline,
            gate,
        );
        search.iterations_completed += restart.iterations_completed;
        search.candidates_scored += restart.candidates_scored;
        if let Some(candidate) = restart.best_candidate {
            search.candidates.push(candidate);
        }
        if let Some(failure) = restart.failure {
            let budget_error = apply_gated_box_search_failure(&mut search, failure);
            return BudgetedBoxApproximateSearch {
                search,
                budget_error,
            };
        }
        search.restarts_completed += 1;
    }
    if let Err(failure) = gated_candidate_checkpoint(
        &mut deadline,
        search.iterations_completed,
        gate,
        "M24 candidate search publication",
    ) {
        let budget_error = apply_gated_box_search_failure(&mut search, failure);
        return BudgetedBoxApproximateSearch {
            search,
            budget_error,
        };
    }
    BudgetedBoxApproximateSearch {
        search,
        budget_error: None,
    }
}

fn apply_gated_box_search_failure(
    search: &mut BoxApproximateSearch,
    failure: GatedCandidateFailure,
) -> Option<ConstrainedZonotopeCallBudgetError> {
    match failure {
        GatedCandidateFailure::Deadline(_) => {
            search.status = ReluTailBoxCutOptimizerStatus::Deadline;
            None
        }
        GatedCandidateFailure::NonFinite(_) => {
            search.status = ReluTailBoxCutOptimizerStatus::NonFiniteCandidate;
            None
        }
        GatedCandidateFailure::Allocation(_) => {
            search.status = ReluTailBoxCutOptimizerStatus::AllocationFallback;
            None
        }
        GatedCandidateFailure::Budget(error) => {
            search.status = box_optimizer_status_from_budget_error(&error);
            Some(error)
        }
    }
}

fn collect_box_search_variables_with_gate<C, G>(
    original_hull: &[(BigRational, BigRational)],
    auxiliary: &CertifiedAuxiliaryBounds64,
    expected: usize,
    deadline: &mut CandidateDeadline<C>,
    gate: &mut G,
) -> Result<Vec<BoxSearchVariable>, GatedCandidateFailure>
where
    C: FnMut() -> Duration,
    G: ConstrainedZonotopeCallGate,
{
    let mut variables = Vec::new();
    variables
        .try_reserve_exact(expected)
        .map_err(|_| GatedCandidateFailure::Allocation(0))?;
    for (coordinate, ((hull_lower, hull_upper), (&lower, &upper))) in original_hull
        .iter()
        .zip(auxiliary.lower().iter().zip(auxiliary.upper()))
        .enumerate()
    {
        gated_candidate_visit(
            deadline,
            0,
            gate,
            "M24 candidate endpoint-variable construction",
        )?;
        let Some(exact_lower) = BigRational::from_float(lower) else {
            return Err(GatedCandidateFailure::NonFinite(0));
        };
        let Some(exact_upper) = BigRational::from_float(upper) else {
            return Err(GatedCandidateFailure::NonFinite(0));
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

struct GatedBoxRestartOutcome {
    best_candidate: Option<Vec<f64>>,
    failure: Option<GatedCandidateFailure>,
    iterations_completed: usize,
    candidates_scored: usize,
}

#[allow(clippy::too_many_arguments)]
fn run_box_optimizer_restart_with_gate<C, G>(
    domain: &ConstrainedZonotope64,
    line_direction: &[f64],
    variables: &[BoxSearchVariable],
    schedule: ReluTailBoxCutAdamSchedule,
    multiplier_cap: f64,
    completed_before: usize,
    deadline: &mut CandidateDeadline<C>,
    gate: &mut G,
) -> GatedBoxRestartOutcome
where
    C: FnMut() -> Duration,
    G: ConstrainedZonotopeCallGate,
{
    let failed_without_candidate = |failure| GatedBoxRestartOutcome {
        best_candidate: None,
        failure: Some(failure),
        iterations_completed: 0,
        candidates_scored: 0,
    };
    if let Err(failure) = gated_candidate_checkpoint(
        deadline,
        completed_before,
        gate,
        "M24 candidate restart admission",
    ) {
        return failed_without_candidate(failure);
    }
    let mut multipliers = match gated_box_candidate_zeros(
        variables.len(),
        completed_before,
        deadline,
        gate,
        "M24 multiplier initialization",
    ) {
        Ok(values) => values,
        Err(failure) => return failed_without_candidate(failure),
    };
    let mut first = match gated_box_candidate_zeros(
        variables.len(),
        completed_before,
        deadline,
        gate,
        "M24 first-moment initialization",
    ) {
        Ok(values) => values,
        Err(failure) => return failed_without_candidate(failure),
    };
    let mut second = match gated_box_candidate_zeros(
        variables.len(),
        completed_before,
        deadline,
        gate,
        "M24 second-moment initialization",
    ) {
        Ok(values) => values,
        Err(failure) => return failed_without_candidate(failure),
    };
    let mut best = match gated_box_candidate_zeros(
        variables.len(),
        completed_before,
        deadline,
        gate,
        "M24 best-candidate initialization",
    ) {
        Ok(values) => values,
        Err(failure) => return failed_without_candidate(failure),
    };
    let mut best_scratch = match gated_box_candidate_zeros(
        variables.len(),
        completed_before,
        deadline,
        gate,
        "M24 best-candidate scratch initialization",
    ) {
        Ok(values) => values,
        Err(failure) => return failed_without_candidate(failure),
    };
    let mut direction = match gated_box_candidate_zeros(
        domain.value_dim(),
        completed_before,
        deadline,
        gate,
        "M24 candidate direction initialization",
    ) {
        Ok(values) => values,
        Err(failure) => return failed_without_candidate(failure),
    };
    let mut witness = match gated_box_candidate_zeros(
        domain.value_dim(),
        completed_before,
        deadline,
        gate,
        "M24 candidate witness initialization",
    ) {
        Ok(values) => values,
        Err(failure) => return failed_without_candidate(failure),
    };

    let mut best_objective = match approximate_box_objective_and_witness_with_gate(
        domain,
        line_direction,
        variables,
        &multipliers,
        &mut direction,
        &mut witness,
        completed_before,
        deadline,
        gate,
    ) {
        Ok(objective) => objective,
        Err(failure) => return failed_without_candidate(failure),
    };
    let mut iterations_completed = 0_usize;
    let mut candidates_scored = 1_usize;

    for iteration in 0..schedule.iterations {
        let global_completed = completed_before + iterations_completed;
        if let Err(failure) =
            gated_candidate_checkpoint(deadline, global_completed, gate, "M24 candidate iteration")
        {
            return GatedBoxRestartOutcome {
                best_candidate: Some(best),
                failure: Some(failure),
                iterations_completed,
                candidates_scored,
            };
        }
        let step = match i32::try_from(iteration + 1) {
            Ok(step) => step,
            Err(_) => {
                return GatedBoxRestartOutcome {
                    best_candidate: Some(best),
                    failure: Some(GatedCandidateFailure::NonFinite(global_completed)),
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
            return GatedBoxRestartOutcome {
                best_candidate: Some(best),
                failure: Some(GatedCandidateFailure::NonFinite(global_completed)),
                iterations_completed,
                candidates_scored,
            };
        }
        for (slot, variable) in variables.iter().enumerate() {
            if let Err(failure) = gated_candidate_visit(
                deadline,
                global_completed,
                gate,
                "M24 candidate Adam update",
            ) {
                return GatedBoxRestartOutcome {
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
                return GatedBoxRestartOutcome {
                    best_candidate: Some(best),
                    failure: Some(GatedCandidateFailure::NonFinite(global_completed)),
                    iterations_completed,
                    candidates_scored,
                };
            }
            multipliers[slot] = canonical_zero(raw_candidate.clamp(0.0, multiplier_cap));
        }

        let completed = global_completed + 1;
        let objective = match approximate_box_objective_and_witness_with_gate(
            domain,
            line_direction,
            variables,
            &multipliers,
            &mut direction,
            &mut witness,
            completed,
            deadline,
            gate,
        ) {
            Ok(objective) => objective,
            Err(failure) => {
                return GatedBoxRestartOutcome {
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
            if let Err(failure) = gated_copy_box_candidate_atomically(
                &mut best,
                &mut best_scratch,
                &multipliers,
                completed,
                deadline,
                gate,
            ) {
                return GatedBoxRestartOutcome {
                    best_candidate: Some(best),
                    failure: Some(failure),
                    iterations_completed,
                    candidates_scored,
                };
            }
            best_objective = objective;
        }
    }

    GatedBoxRestartOutcome {
        best_candidate: Some(best),
        failure: None,
        iterations_completed,
        candidates_scored,
    }
}

fn gated_box_candidate_zeros<C, G>(
    count: usize,
    iterations: usize,
    deadline: &mut CandidateDeadline<C>,
    gate: &mut G,
    checkpoint: &'static str,
) -> Result<Vec<f64>, GatedCandidateFailure>
where
    C: FnMut() -> Duration,
    G: ConstrainedZonotopeCallGate,
{
    #[cfg(test)]
    if BOX_OPTIMIZER_FAIL_NEXT_ALLOCATION.with(|fail| fail.replace(false)) {
        return Err(GatedCandidateFailure::Allocation(iterations));
    }
    gated_candidate_zeros(count, iterations, deadline, gate, checkpoint)
}

#[allow(clippy::too_many_arguments)]
fn approximate_box_objective_and_witness_with_gate<C, G>(
    domain: &ConstrainedZonotope64,
    line_direction: &[f64],
    variables: &[BoxSearchVariable],
    multipliers: &[f64],
    direction: &mut [f64],
    witness: &mut [f64],
    iterations: usize,
    deadline: &mut CandidateDeadline<C>,
    gate: &mut G,
) -> Result<f64, GatedCandidateFailure>
where
    C: FnMut() -> Duration,
    G: ConstrainedZonotopeCallGate,
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
        gated_candidate_visit(
            deadline,
            iterations,
            gate,
            "M24 candidate direction and witness initialization",
        )?;
        if !source.is_finite() {
            return Err(GatedCandidateFailure::NonFinite(iterations));
        }
        *target = source;
        *witness_value = domain.center()[coordinate];
    }
    let mut objective = 0.0_f64;
    for (variable, &multiplier) in variables.iter().zip(multipliers) {
        gated_candidate_visit(
            deadline,
            iterations,
            gate,
            "M24 candidate Box-variable score",
        )?;
        if !multiplier.is_finite() || multiplier < 0.0 {
            return Err(GatedCandidateFailure::NonFinite(iterations));
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
            return Err(GatedCandidateFailure::NonFinite(iterations));
        }
    }
    for coordinate in 0..domain.value_dim() {
        gated_candidate_visit(
            deadline,
            iterations,
            gate,
            "M24 candidate center/remainder score",
        )?;
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
            return Err(GatedCandidateFailure::NonFinite(iterations));
        }
    }
    for generator in domain.generators() {
        gated_candidate_visit(
            deadline,
            iterations,
            gate,
            "M24 candidate generator-column score",
        )?;
        let mut projection = 0.0_f64;
        for (coordinate, coefficient) in generator.entries() {
            gated_candidate_visit(
                deadline,
                iterations,
                gate,
                "M24 candidate generator-entry projection",
            )?;
            projection += direction[coordinate] * coefficient;
            if !projection.is_finite() {
                return Err(GatedCandidateFailure::NonFinite(iterations));
            }
        }
        objective -= projection.abs();
        if !objective.is_finite() {
            return Err(GatedCandidateFailure::NonFinite(iterations));
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
                gated_candidate_visit(
                    deadline,
                    iterations,
                    gate,
                    "M24 candidate sparse-witness replay",
                )?;
                witness[coordinate] -= sign * coefficient;
                if !witness[coordinate].is_finite() {
                    return Err(GatedCandidateFailure::NonFinite(iterations));
                }
            }
        }
    }
    Ok(objective)
}

fn gated_copy_box_candidate_atomically<C, G>(
    best: &mut Vec<f64>,
    scratch: &mut Vec<f64>,
    source: &[f64],
    iterations: usize,
    deadline: &mut CandidateDeadline<C>,
    gate: &mut G,
) -> Result<(), GatedCandidateFailure>
where
    C: FnMut() -> Duration,
    G: ConstrainedZonotopeCallGate,
{
    debug_assert_eq!(best.len(), source.len());
    debug_assert_eq!(scratch.len(), source.len());
    for (target, &source) in scratch.iter_mut().zip(source) {
        gated_candidate_visit(deadline, iterations, gate, "M24 best-candidate atomic copy")?;
        *target = source;
    }
    std::mem::swap(best, scratch);
    Ok(())
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

fn box_optimizer_status_from_budget_error(
    error: &ConstrainedZonotopeCallBudgetError,
) -> ReluTailBoxCutOptimizerStatus {
    match error {
        ConstrainedZonotopeCallBudgetError::DeadlineExpired { .. } => {
            ReluTailBoxCutOptimizerStatus::Deadline
        }
        ConstrainedZonotopeCallBudgetError::ResourceOverflow { .. }
        | ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { .. } => {
            ReluTailBoxCutOptimizerStatus::ResourceFallback
        }
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

fn expand_box_candidate_with_gate<G>(
    variables: &[BoxSearchVariable],
    candidate: &[f64],
    value_dim: usize,
    gate: &mut G,
) -> Result<(Vec<f64>, Vec<f64>), ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    if candidate.len() != variables.len() {
        return Err(ReluTailDualError::Shape {
            field: "optimized Box multipliers",
            expected: variables.len(),
            got: candidate.len(),
        }
        .into());
    }
    let mut upper = zero_f64_with_gate(
        value_dim,
        "optimized upper Box multipliers",
        "M24 upper Box-multiplier expansion",
        gate,
    )?;
    let mut lower = zero_f64_with_gate(
        value_dim,
        "optimized lower Box multipliers",
        "M24 lower Box-multiplier expansion",
        gate,
    )?;
    for (slot, (variable, &value)) in variables.iter().zip(candidate).enumerate() {
        gate.charge_items(1, "M24 Box-multiplier expansion")?;
        if !value.is_finite() || value < 0.0 {
            return Err(ReluTailDualError::NonFiniteArithmetic {
                coordinate: slot,
                operation: "optimized Box multiplier expansion",
            }
            .into());
        }
        match variable.kind {
            BoxVariableKind::Upper => upper[variable.coordinate] = canonical_zero(value),
            BoxVariableKind::Lower => lower[variable.coordinate] = canonical_zero(value),
        }
    }
    gate.checkpoint("M24 Box-multiplier expansion complete")?;
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
    let mut gate = InertConstrainedZonotopeCallGate;
    match build_auxiliary_box_cut_certificate_with_original_hull_and_gate(
        domain,
        auxiliary,
        auxiliary_result,
        upper_box_multipliers,
        lower_box_multipliers,
        supplied_predicate_multipliers,
        original_hull,
        &mut gate,
    ) {
        Ok(certificate) => Ok(certificate),
        Err(ReluTailDualBudgetError::Bound(error)) => Err(error),
        Err(ReluTailDualBudgetError::Budget(_)) => {
            unreachable!("the inert M24 replay gate cannot refuse work")
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_auxiliary_box_cut_certificate_with_original_hull_and_gate<G>(
    domain: &ConstrainedZonotope64,
    auxiliary: &CertifiedAuxiliaryBounds64,
    auxiliary_result: &ReluTailDualResult,
    upper_box_multipliers: &[f64],
    lower_box_multipliers: &[f64],
    supplied_predicate_multipliers: Option<&[f64]>,
    original_hull: &[(BigRational, BigRational)],
    gate: &mut G,
) -> Result<ReluTailBoxCutCertificate, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    debug_assert_eq!(auxiliary.value_dim(), domain.value_dim());
    debug_assert_eq!(auxiliary_result.direction.len(), domain.value_dim());
    debug_assert_eq!(upper_box_multipliers.len(), domain.value_dim());
    debug_assert_eq!(lower_box_multipliers.len(), domain.value_dim());
    debug_assert_eq!(original_hull.len(), domain.value_dim());

    if auxiliary.value_dim() != domain.value_dim() {
        return Err(ReluTailDualError::AuxiliaryDimensionMismatch {
            expected: domain.value_dim(),
            got: auxiliary.value_dim(),
        }
        .into());
    }
    for (field, got) in [
        ("Box-cut source direction", auxiliary_result.direction.len()),
        ("upper Box multipliers", upper_box_multipliers.len()),
        ("lower Box multipliers", lower_box_multipliers.len()),
        ("original Box-cut hull", original_hull.len()),
    ] {
        if got != domain.value_dim() {
            return Err(ReluTailDualError::Shape {
                field,
                expected: domain.value_dim(),
                got,
            }
            .into());
        }
    }
    gate.checkpoint("M24 exact Box-cut validation")?;

    // This hull deliberately precedes and excludes the auxiliary
    // intersection.  The residual between p* and the replayed dyadic p must be
    // valid at every point considered by D_Z, including spurious Z points
    // outside the certified Box.  The direct M22 wrapper computes it here;
    // M24 passes the domain-tied prepared copy without changing the replay.
    let mut replay_direction = zero_f64_with_gate(
        domain.value_dim(),
        "Box-cut replay direction",
        "M24 exact Box-cut direction initialization",
        gate,
    )?;
    let mut exact_constant = checked_rational(
        auxiliary_result.exact_constant.clone(),
        "Box-cut line constant",
        0,
    )?;

    for coordinate in 0..domain.value_dim() {
        gate.checkpoint("M24 exact Box-cut coordinate admission")?;
        gate.charge_items(16, "M24 exact Box-cut coordinate arithmetic")?;
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

    let mut zero = zero_f64_with_gate(
        domain.constraint_count(),
        "Box-cut zero predicate multipliers",
        "M24 zero predicate-multiplier initialization",
        gate,
    )?;
    for value in &mut zero {
        gate.charge_items(1, "M24 zero predicate-multiplier canonicalization")?;
        *value = 0.0;
    }
    let replay = match evaluate_constrained_zonotope64_dual_with_call_gate(
        domain,
        &replay_direction,
        &zero,
        gate,
    ) {
        Ok(bounds) => bounds,
        Err(ConstrainedZonotopeDualBudgetError::Evaluation(error)) => {
            return Err(ReluTailDualError::Baseline(error.into()).into());
        }
        Err(ConstrainedZonotopeDualBudgetError::Budget(error)) => return Err(error.into()),
    };
    gate.charge_items(1, "M24 exact zero-replay combination")?;
    let zero_predicate_lower_bound = combine_exact_lower(replay.lower, &exact_constant)?;
    let mut lower_bound = zero_predicate_lower_bound;
    let mut predicate_multipliers = zero;
    let mut supplied_predicate_multipliers_used = false;

    if let Some(supplied) = valid_supplied_multipliers_with_gate(
        supplied_predicate_multipliers,
        domain.constraint_count(),
        gate,
    )? {
        let supplied_replay = match evaluate_constrained_zonotope64_dual_with_call_gate(
            domain,
            &replay_direction,
            supplied,
            gate,
        ) {
            Ok(bounds) => Some(bounds),
            Err(ConstrainedZonotopeDualBudgetError::Evaluation(_)) => None,
            Err(ConstrainedZonotopeDualBudgetError::Budget(error)) => return Err(error.into()),
        };
        if let Some(replay) = supplied_replay {
            gate.charge_items(1, "M24 exact supplied-replay combination")?;
            if let Ok(candidate) = combine_exact_lower(replay.lower, &exact_constant) {
                if candidate > lower_bound {
                    predicate_multipliers = clone_f64_with_gate(
                        supplied,
                        "accepted Box-cut predicate multipliers",
                        "M24 accepted predicate-multiplier clone",
                        gate,
                    )?;
                    lower_bound = candidate;
                    supplied_predicate_multipliers_used = true;
                }
            }
        }
    }

    let accepted_upper = clone_f64_with_gate(
        upper_box_multipliers,
        "accepted upper Box multipliers",
        "M24 accepted upper Box-multiplier clone",
        gate,
    )?;
    let accepted_lower = clone_f64_with_gate(
        lower_box_multipliers,
        "accepted lower Box multipliers",
        "M24 accepted lower Box-multiplier clone",
        gate,
    )?;
    gate.checkpoint("M24 exact Box-cut publication")?;
    Ok(ReluTailBoxCutCertificate {
        lower_bound,
        zero_predicate_lower_bound,
        replay_direction,
        upper_box_multipliers: accepted_upper,
        lower_box_multipliers: accepted_lower,
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

#[allow(clippy::too_many_arguments)]
fn replay_direction_with_gate<G>(
    domain: &ConstrainedZonotope64,
    exact_constant: &BigRational,
    variables: &[SlopeVariable],
    direction: Vec<f64>,
    zero: &[f64],
    supplied: Option<&[f64]>,
    best: &mut AcceptedCandidate,
    gate: &mut G,
) -> Result<CandidateReplayOutcome, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    if !valid_direction_slopes_with_gate(&direction, variables, gate)? {
        return Ok(CandidateReplayOutcome::Rejected);
    }
    let bounds =
        match evaluate_constrained_zonotope64_dual_with_call_gate(domain, &direction, zero, gate) {
            Ok(bounds) => bounds,
            Err(ConstrainedZonotopeDualBudgetError::Evaluation(_)) => {
                return Ok(CandidateReplayOutcome::Rejected);
            }
            Err(ConstrainedZonotopeDualBudgetError::Budget(error)) => return Err(error.into()),
        };
    gate.charge_items(1, "ReLU-tail exact replay combination")?;
    let Ok(zero_lower) = combine_exact_lower(bounds.lower, exact_constant) else {
        return Ok(CandidateReplayOutcome::Rejected);
    };
    let Some(candidate_multipliers) =
        candidate_clone_f64_with_gate(zero, "ReLU-tail replay multiplier clone", gate)?
    else {
        return Ok(CandidateReplayOutcome::AllocationFallback);
    };
    let mut candidate = AcceptedCandidate {
        lower: zero_lower,
        zero_lower,
        direction,
        multipliers: candidate_multipliers,
        supplied_used: false,
    };
    let supplied_outcome =
        maybe_replay_supplied_with_gate(domain, exact_constant, supplied, &mut candidate, gate)?;
    if candidate.lower > best.lower {
        *best = candidate;
    }
    Ok(match supplied_outcome {
        SuppliedReplayOutcome::ReplayedOrUnused => CandidateReplayOutcome::Replayed {
            zero_predicate_lower_bound: zero_lower,
        },
        SuppliedReplayOutcome::AllocationFallback => CandidateReplayOutcome::AllocationFallback,
    })
}

#[cfg(test)]
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

fn maybe_replay_supplied_with_gate<G>(
    domain: &ConstrainedZonotope64,
    exact_constant: &BigRational,
    supplied: Option<&[f64]>,
    candidate: &mut AcceptedCandidate,
    gate: &mut G,
) -> Result<SuppliedReplayOutcome, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let Some(supplied) = supplied else {
        return Ok(SuppliedReplayOutcome::ReplayedOrUnused);
    };
    let bounds = match evaluate_constrained_zonotope64_dual_with_call_gate(
        domain,
        &candidate.direction,
        supplied,
        gate,
    ) {
        Ok(bounds) => bounds,
        Err(ConstrainedZonotopeDualBudgetError::Evaluation(_)) => {
            return Ok(SuppliedReplayOutcome::ReplayedOrUnused);
        }
        Err(ConstrainedZonotopeDualBudgetError::Budget(error)) => return Err(error.into()),
    };
    gate.charge_items(1, "ReLU-tail supplied replay combination")?;
    let Ok(lower) = combine_exact_lower(bounds.lower, exact_constant) else {
        return Ok(SuppliedReplayOutcome::ReplayedOrUnused);
    };
    if lower > candidate.lower {
        let Some(accepted_multipliers) = candidate_clone_f64_with_gate(
            supplied,
            "ReLU-tail accepted supplied-multiplier clone",
            gate,
        )?
        else {
            return Ok(SuppliedReplayOutcome::AllocationFallback);
        };
        candidate.lower = lower;
        candidate.multipliers = accepted_multipliers;
        candidate.supplied_used = true;
    }
    Ok(SuppliedReplayOutcome::ReplayedOrUnused)
}

#[cfg(test)]
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

fn valid_supplied_multipliers_with_gate<'a, G>(
    supplied: Option<&'a [f64]>,
    expected: usize,
    gate: &mut G,
) -> Result<Option<&'a [f64]>, ConstrainedZonotopeCallBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let Some(supplied) = supplied else {
        return Ok(None);
    };
    if supplied.len() != expected {
        return Ok(None);
    }
    for &value in supplied {
        gate.charge_items(1, "ReLU-tail supplied-multiplier validation")?;
        if !value.is_finite() || value < 0.0 {
            return Ok(None);
        }
    }
    Ok(Some(supplied))
}

#[cfg(test)]
fn valid_direction_slopes(direction: &[f64], variables: &[SlopeVariable]) -> bool {
    variables.iter().all(|variable| {
        direction.get(variable.coordinate).is_some_and(|&slope| {
            slope.is_finite() && valid_direct_slope(slope, &variable.exact_upper)
        })
    })
}

fn valid_direction_slopes_with_gate<G>(
    direction: &[f64],
    variables: &[SlopeVariable],
    gate: &mut G,
) -> Result<bool, ConstrainedZonotopeCallBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    for variable in variables {
        gate.charge_items(1, "ReLU-tail replay-slope validation")?;
        if direction.get(variable.coordinate).is_none_or(|&slope| {
            !slope.is_finite() || !valid_direct_slope(slope, &variable.exact_upper)
        }) {
            return Ok(false);
        }
    }
    Ok(true)
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

#[derive(Clone, Debug)]
enum GatedCandidateFailure {
    Deadline(usize),
    NonFinite(usize),
    Allocation(usize),
    Budget(ConstrainedZonotopeCallBudgetError),
}

impl From<CandidateFailure> for GatedCandidateFailure {
    fn from(failure: CandidateFailure) -> Self {
        match failure {
            CandidateFailure::Deadline(iterations) => Self::Deadline(iterations),
            CandidateFailure::NonFinite(iterations) => Self::NonFinite(iterations),
            CandidateFailure::Allocation(iterations) => Self::Allocation(iterations),
        }
    }
}

#[cfg(test)]
fn projected_adam_candidate(
    domain: &ConstrainedZonotope64,
    variables: &[SlopeVariable],
    direction: Vec<f64>,
    config: ReluTailDualConfig,
) -> Result<(Vec<f64>, usize), CandidateFailure> {
    let start = Instant::now();
    projected_adam_candidate_with_clock(domain, variables, direction, config, || start.elapsed())
}

#[cfg(test)]
fn projected_adam_candidate_with_clock<C>(
    domain: &ConstrainedZonotope64,
    variables: &[SlopeVariable],
    direction: Vec<f64>,
    config: ReluTailDualConfig,
    elapsed: C,
) -> Result<(Vec<f64>, usize), CandidateFailure>
where
    C: FnMut() -> Duration,
{
    let mut gate = InertConstrainedZonotopeCallGate;
    match projected_adam_candidate_with_clock_and_gate(
        domain, variables, direction, config, elapsed, &mut gate,
    ) {
        Ok(result) => Ok(result),
        Err(GatedCandidateFailure::Deadline(iterations)) => {
            Err(CandidateFailure::Deadline(iterations))
        }
        Err(GatedCandidateFailure::NonFinite(iterations)) => {
            Err(CandidateFailure::NonFinite(iterations))
        }
        Err(GatedCandidateFailure::Allocation(iterations)) => {
            Err(CandidateFailure::Allocation(iterations))
        }
        Err(GatedCandidateFailure::Budget(_)) => {
            unreachable!("the inert candidate call gate cannot refuse work")
        }
    }
}

fn projected_adam_candidate_with_call_gate<G>(
    domain: &ConstrainedZonotope64,
    variables: &[SlopeVariable],
    direction: Vec<f64>,
    config: ReluTailDualConfig,
    gate: &mut G,
) -> Result<(Vec<f64>, usize), GatedCandidateFailure>
where
    G: ConstrainedZonotopeCallGate,
{
    let start = Instant::now();
    projected_adam_candidate_with_clock_and_gate(
        domain,
        variables,
        direction,
        config,
        || start.elapsed(),
        gate,
    )
}

fn projected_adam_candidate_with_clock_and_gate<C, G>(
    domain: &ConstrainedZonotope64,
    variables: &[SlopeVariable],
    mut direction: Vec<f64>,
    config: ReluTailDualConfig,
    elapsed: C,
    gate: &mut G,
) -> Result<(Vec<f64>, usize), GatedCandidateFailure>
where
    C: FnMut() -> Duration,
    G: ConstrainedZonotopeCallGate,
{
    let mut deadline = CandidateDeadline::new(config.wall_time, elapsed);
    // The candidate-only clock starts inside the wrapper above.  Check it
    // before any allocation, initialization, or approximate scoring work.
    gated_candidate_checkpoint(&mut deadline, 0, gate, "ReLU-tail candidate startup")?;
    let mut first = gated_candidate_zeros(
        variables.len(),
        0,
        &mut deadline,
        gate,
        "ReLU-tail first-moment initialization",
    )?;
    let mut second = gated_candidate_zeros(
        variables.len(),
        0,
        &mut deadline,
        gate,
        "ReLU-tail second-moment initialization",
    )?;
    let mut gradient = gated_candidate_zeros(
        variables.len(),
        0,
        &mut deadline,
        gate,
        "ReLU-tail gradient initialization",
    )?;
    let mut variable_slot = gated_candidate_usizes(
        domain.value_dim(),
        0,
        &mut deadline,
        gate,
        "ReLU-tail candidate coordinate-map initialization",
    )?;
    for (slot, variable) in variables.iter().enumerate() {
        gated_candidate_visit(
            &mut deadline,
            0,
            gate,
            "ReLU-tail candidate coordinate-map construction",
        )?;
        variable_slot[variable.coordinate] = slot;
    }
    let mut best = gated_candidate_clone_with_deadline(
        &direction,
        0,
        &mut deadline,
        gate,
        "ReLU-tail candidate best-direction clone",
    )?;
    let mut best_objective = approximate_zero_multiplier_objective_with_gate(
        domain,
        &direction,
        0,
        &mut deadline,
        gate,
    )?;

    for iteration in 0..config.iterations {
        gated_candidate_checkpoint(
            &mut deadline,
            iteration,
            gate,
            "ReLU-tail candidate iteration",
        )?;
        for value in &mut gradient {
            gated_candidate_visit(
                &mut deadline,
                iteration,
                gate,
                "ReLU-tail candidate gradient clearing",
            )?;
            *value = 0.0;
        }
        for (slot, variable) in variables.iter().enumerate() {
            gated_candidate_visit(
                &mut deadline,
                iteration,
                gate,
                "ReLU-tail candidate center gradient",
            )?;
            gradient[slot] =
                domain.center()[variable.coordinate] - domain.box_remainder()[variable.coordinate];
        }
        for generator in domain.generators() {
            // Count and check every generator column, including empty ones.
            gated_candidate_visit(
                &mut deadline,
                iteration,
                gate,
                "ReLU-tail candidate generator-column gradient",
            )?;
            let mut projection = 0.0_f64;
            for (coordinate, coefficient) in generator.entries() {
                gated_candidate_visit(
                    &mut deadline,
                    iteration,
                    gate,
                    "ReLU-tail candidate generator-entry projection",
                )?;
                projection += direction[coordinate] * coefficient;
                if !projection.is_finite() {
                    return Err(GatedCandidateFailure::NonFinite(iteration));
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
                    gated_candidate_visit(
                        &mut deadline,
                        iteration,
                        gate,
                        "ReLU-tail candidate sparse-gradient replay",
                    )?;
                    let slot = variable_slot[coordinate];
                    if slot != usize::MAX {
                        gradient[slot] -= sign * coefficient;
                    }
                }
            }
        }

        let step = i32::try_from(iteration + 1)
            .map_err(|_| GatedCandidateFailure::NonFinite(iteration))?;
        let first_correction = 1.0 - config.beta1.powi(step);
        let second_correction = 1.0 - config.beta2.powi(step);
        if !first_correction.is_finite()
            || !second_correction.is_finite()
            || first_correction <= 0.0
            || second_correction <= 0.0
        {
            return Err(GatedCandidateFailure::NonFinite(iteration));
        }
        for (slot, variable) in variables.iter().enumerate() {
            gated_candidate_visit(
                &mut deadline,
                iteration,
                gate,
                "ReLU-tail candidate Adam update",
            )?;
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
                return Err(GatedCandidateFailure::NonFinite(iteration));
            }
            direction[variable.coordinate] = canonical_zero(candidate);
        }
        let completed = iteration + 1;
        let objective = approximate_zero_multiplier_objective_with_gate(
            domain,
            &direction,
            completed,
            &mut deadline,
            gate,
        )?;
        if objective > best_objective {
            best_objective = objective;
            gated_copy_candidate_direction(&mut best, &direction, completed, &mut deadline, gate)?;
        }
    }
    gated_candidate_checkpoint(
        &mut deadline,
        config.iterations,
        gate,
        "ReLU-tail candidate completion",
    )?;
    Ok((best, config.iterations))
}

fn gated_candidate_checkpoint<C, G>(
    deadline: &mut CandidateDeadline<C>,
    iterations: usize,
    gate: &mut G,
    checkpoint: &'static str,
) -> Result<(), GatedCandidateFailure>
where
    C: FnMut() -> Duration,
    G: ConstrainedZonotopeCallGate,
{
    gate.checkpoint(checkpoint)
        .map_err(GatedCandidateFailure::Budget)?;
    deadline
        .checkpoint(iterations)
        .map_err(GatedCandidateFailure::from)
}

fn gated_candidate_visit<C, G>(
    deadline: &mut CandidateDeadline<C>,
    iterations: usize,
    gate: &mut G,
    checkpoint: &'static str,
) -> Result<(), GatedCandidateFailure>
where
    C: FnMut() -> Duration,
    G: ConstrainedZonotopeCallGate,
{
    gate.charge_items(1, checkpoint)
        .map_err(GatedCandidateFailure::Budget)?;
    deadline
        .visit(iterations)
        .map_err(GatedCandidateFailure::from)
}

fn approximate_zero_multiplier_objective_with_gate<C, G>(
    domain: &ConstrainedZonotope64,
    direction: &[f64],
    iterations: usize,
    deadline: &mut CandidateDeadline<C>,
    gate: &mut G,
) -> Result<f64, GatedCandidateFailure>
where
    C: FnMut() -> Duration,
    G: ConstrainedZonotopeCallGate,
{
    let mut value = 0.0_f64;
    for coordinate in 0..domain.value_dim() {
        gated_candidate_visit(
            deadline,
            iterations,
            gate,
            "ReLU-tail candidate coordinate score",
        )?;
        value += direction[coordinate] * domain.center()[coordinate];
        value -= direction[coordinate].abs() * domain.box_remainder()[coordinate];
        if !value.is_finite() {
            return Err(GatedCandidateFailure::NonFinite(iterations));
        }
    }
    for generator in domain.generators() {
        gated_candidate_visit(
            deadline,
            iterations,
            gate,
            "ReLU-tail candidate generator-column score",
        )?;
        let mut projection = 0.0_f64;
        for (coordinate, coefficient) in generator.entries() {
            gated_candidate_visit(
                deadline,
                iterations,
                gate,
                "ReLU-tail candidate generator-entry score",
            )?;
            projection += direction[coordinate] * coefficient;
            if !projection.is_finite() {
                return Err(GatedCandidateFailure::NonFinite(iterations));
            }
        }
        value -= projection.abs();
        if !value.is_finite() {
            return Err(GatedCandidateFailure::NonFinite(iterations));
        }
    }
    Ok(value)
}

fn gated_candidate_zeros<C, G>(
    count: usize,
    iterations: usize,
    deadline: &mut CandidateDeadline<C>,
    gate: &mut G,
    checkpoint: &'static str,
) -> Result<Vec<f64>, GatedCandidateFailure>
where
    C: FnMut() -> Duration,
    G: ConstrainedZonotopeCallGate,
{
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| GatedCandidateFailure::Allocation(iterations))?;
    for _ in 0..count {
        gated_candidate_visit(deadline, iterations, gate, checkpoint)?;
        values.push(0.0);
    }
    Ok(values)
}

fn gated_candidate_usizes<C, G>(
    count: usize,
    iterations: usize,
    deadline: &mut CandidateDeadline<C>,
    gate: &mut G,
    checkpoint: &'static str,
) -> Result<Vec<usize>, GatedCandidateFailure>
where
    C: FnMut() -> Duration,
    G: ConstrainedZonotopeCallGate,
{
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| GatedCandidateFailure::Allocation(iterations))?;
    for _ in 0..count {
        gated_candidate_visit(deadline, iterations, gate, checkpoint)?;
        values.push(usize::MAX);
    }
    Ok(values)
}

fn gated_candidate_clone_with_deadline<C, G>(
    source: &[f64],
    iterations: usize,
    deadline: &mut CandidateDeadline<C>,
    gate: &mut G,
    checkpoint: &'static str,
) -> Result<Vec<f64>, GatedCandidateFailure>
where
    C: FnMut() -> Duration,
    G: ConstrainedZonotopeCallGate,
{
    let mut values = Vec::new();
    values
        .try_reserve_exact(source.len())
        .map_err(|_| GatedCandidateFailure::Allocation(iterations))?;
    for &value in source {
        gated_candidate_visit(deadline, iterations, gate, checkpoint)?;
        values.push(value);
    }
    Ok(values)
}

fn gated_copy_candidate_direction<C, G>(
    target: &mut [f64],
    source: &[f64],
    iterations: usize,
    deadline: &mut CandidateDeadline<C>,
    gate: &mut G,
) -> Result<(), GatedCandidateFailure>
where
    C: FnMut() -> Duration,
    G: ConstrainedZonotopeCallGate,
{
    debug_assert_eq!(target.len(), source.len());
    for (target, &source) in target.iter_mut().zip(source) {
        gated_candidate_visit(
            deadline,
            iterations,
            gate,
            "ReLU-tail candidate best-direction copy",
        )?;
        *target = source;
    }
    Ok(())
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

fn check_declared_rationals_with_gate<G>(
    coefficients: &[BigRational],
    bias: &BigRational,
    gate: &mut G,
) -> Result<(), ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    check_rationals_with_gate(
        coefficients,
        bias,
        RELU_TAIL_DUAL_HARD_MAX_INPUT_RATIONAL_BITS,
        gate,
    )
}

fn check_internally_pulled_rationals_with_gate<G>(
    coefficients: &[BigRational],
    bias: &BigRational,
    gate: &mut G,
) -> Result<(), ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    check_rationals_with_gate(
        coefficients,
        bias,
        RELU_TAIL_DUAL_HARD_MAX_INTERMEDIATE_RATIONAL_BITS,
        gate,
    )
}

fn check_rationals_with_gate<G>(
    coefficients: &[BigRational],
    bias: &BigRational,
    per_term_limit: u64,
    gate: &mut G,
) -> Result<(), ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    require_resource_limit(
        "objective coefficients",
        u64::try_from(coefficients.len()).unwrap_or(u64::MAX),
        RELU_TAIL_DUAL_HARD_MAX_VALUE_DIM as u64,
    )?;
    let mut total = 0_u64;
    for (index, coefficient) in coefficients.iter().enumerate() {
        gate.charge_items(1, "ReLU-tail declared-rational validation")?;
        let bits = rational_bits(coefficient);
        if bits > per_term_limit {
            return Err(ReluTailDualError::RationalInputLimit {
                field: "coefficients",
                index,
                bits,
                limit: per_term_limit,
            }
            .into());
        }
        total = total
            .checked_add(bits)
            .ok_or(ReluTailDualError::ResourceOverflow {
                resource: "total objective rational bits",
            })?;
    }
    let bias_bits = rational_bits(bias);
    if bias_bits > per_term_limit {
        return Err(ReluTailDualError::RationalInputLimit {
            field: "bias",
            index: 0,
            bits: bias_bits,
            limit: per_term_limit,
        }
        .into());
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
    )?;
    Ok(())
}

fn check_mandatory_resources(
    domain: &ConstrainedZonotope64,
    margin: &ExactReluTailMargin,
) -> Result<usize, ReluTailDualError> {
    check_mandatory_margin_resources(domain.value_dim(), margin)?;
    check_mandatory_domain_resources(domain)
}

fn check_mandatory_resources_with_gate<G>(
    domain: &ConstrainedZonotope64,
    margin: &ExactReluTailMargin,
    gate: &mut G,
) -> Result<usize, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    if margin.coefficients.len() != domain.value_dim() {
        return Err(ReluTailDualError::Shape {
            field: "margin coefficients",
            expected: domain.value_dim(),
            got: margin.coefficients.len(),
        }
        .into());
    }
    check_declared_rationals_with_gate(&margin.coefficients, &margin.bias, gate)?;
    check_mandatory_domain_resources_with_gate(domain, gate)
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

fn check_mandatory_margin_resources_with_gate<G>(
    value_dim: usize,
    margin: &ExactReluTailMargin,
    gate: &mut G,
) -> Result<(), ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    if margin.coefficients.len() != value_dim {
        return Err(ReluTailDualError::Shape {
            field: "margin coefficients",
            expected: value_dim,
            got: margin.coefficients.len(),
        }
        .into());
    }
    check_declared_rationals_with_gate(&margin.coefficients, &margin.bias, gate)
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

fn check_mandatory_domain_resources_with_gate<G>(
    domain: &ConstrainedZonotope64,
    gate: &mut G,
) -> Result<usize, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
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
    let generator_nonzeros = generator_nonzeros_with_gate(domain, gate)?;
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

fn generator_nonzeros_with_gate<G>(
    domain: &ConstrainedZonotope64,
    gate: &mut G,
) -> Result<usize, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let mut nonzeros = 0_usize;
    for column in domain.generators() {
        gate.charge_items(1, "ReLU-tail generator geometry validation")?;
        nonzeros =
            nonzeros
                .checked_add(column.nnz())
                .ok_or(ReluTailDualError::ResourceOverflow {
                    resource: "generator nonzeros",
                })?;
    }
    Ok(nonzeros)
}

fn prepared_relu_tail_geometry_live_bytes(
    value_dim: usize,
) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    let retained_rationals =
        value_dim
            .checked_mul(2)
            .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "prepared ReLU-tail retained rational count",
            })?;
    let mut live = ConstrainedZonotopePeakLiveBytes::new();
    live.add_bytes(
        size_of::<PreparedReluTailGeometry64<'static>>(),
        "prepared ReLU-tail geometry header bytes",
    )?;
    live.add_elements::<(BigRational, BigRational)>(
        value_dim,
        "prepared ReLU-tail coordinate-pair storage",
    )?;
    live.add_elements::<[u8; RELU_TAIL_RATIONAL_LIVE_BYTES]>(
        retained_rationals,
        "prepared ReLU-tail retained rational payloads",
    )?;
    Ok(live.finish())
}

fn exact_relu_tail_margin_live_bytes(
    value_dim: usize,
) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    let rational_count =
        value_dim
            .checked_add(1)
            .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "exact ReLU-tail margin rational count",
            })?;
    let mut live = ConstrainedZonotopePeakLiveBytes::new();
    live.add_bytes(
        size_of::<ExactReluTailMargin>(),
        "exact ReLU-tail margin header bytes",
    )?;
    live.add_elements::<BigRational>(value_dim, "exact ReLU-tail margin coefficient storage")?;
    live.add_elements::<[u8; RELU_TAIL_RATIONAL_LIVE_BYTES]>(
        rational_count,
        "exact ReLU-tail margin rational payloads",
    )?;
    Ok(live.finish())
}

fn exact_relu_tail_margin_peak_live_bytes(
    value_dim: usize,
) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    let mut peak = ConstrainedZonotopePeakLiveBytes::new();
    peak.add_bytes(
        exact_relu_tail_margin_live_bytes(value_dim)?,
        "retained exact pulled-margin bytes",
    )?;
    peak.add_elements::<[u8; RELU_TAIL_RATIONAL_LIVE_BYTES]>(
        RELU_TAIL_TRANSIENT_RATIONAL_SLOTS,
        "exact pulled-margin transient rational payloads",
    )?;
    Ok(peak.finish())
}

fn prepared_relu_tail_geometry_peak_live_bytes(
    value_dim: usize,
) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    let retained = prepared_relu_tail_geometry_live_bytes(value_dim)?;
    let scratch_rationals = value_dim
        .checked_add(RELU_TAIL_TRANSIENT_RATIONAL_SLOTS)
        .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
            operation: "prepared ReLU-tail scratch rational count",
        })?;
    let mut peak = ConstrainedZonotopePeakLiveBytes::new();
    peak.add_bytes(retained, "prepared ReLU-tail retained geometry bytes")?;
    peak.add_elements::<BigRational>(value_dim, "prepared ReLU-tail coordinate-radius storage")?;
    peak.add_elements::<[u8; RELU_TAIL_RATIONAL_LIVE_BYTES]>(
        scratch_rationals,
        "prepared ReLU-tail exact scratch rational payloads",
    )?;
    Ok(peak.finish())
}

fn relu_tail_dual_result_live_bytes(
    value_dim: usize,
    constraints: usize,
) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    let mut live = ConstrainedZonotopePeakLiveBytes::new();
    live.add_bytes(
        size_of::<ReluTailDualResult>(),
        "retained ReLU-tail result header bytes",
    )?;
    live.add_elements::<f64>(value_dim, "retained ReLU-tail direction storage")?;
    live.add_elements::<f64>(constraints, "retained ReLU-tail multiplier storage")?;
    live.add_elements::<[u8; RELU_TAIL_RATIONAL_LIVE_BYTES]>(
        1,
        "retained ReLU-tail exact-constant payload",
    )?;
    Ok(live.finish())
}

fn relu_tail_box_cut_certificate_live_bytes(
    value_dim: usize,
    constraints: usize,
) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    let value_vectors =
        value_dim
            .checked_mul(3)
            .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "retained M24 certificate value-vector elements",
            })?;
    let mut live = ConstrainedZonotopePeakLiveBytes::new();
    live.add_bytes(
        size_of::<ReluTailBoxCutCertificate>(),
        "retained M24 certificate header bytes",
    )?;
    live.add_elements::<f64>(value_vectors, "retained M24 certificate value vectors")?;
    live.add_elements::<f64>(
        constraints,
        "retained M24 certificate predicate multipliers",
    )?;
    live.add_elements::<[u8; RELU_TAIL_RATIONAL_LIVE_BYTES]>(
        1,
        "retained M24 certificate exact-constant payload",
    )?;
    Ok(live.finish())
}

fn box_cut_endpoint_count_peak_live_bytes() -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    let mut peak = ConstrainedZonotopePeakLiveBytes::new();
    peak.add_elements::<[u8; RELU_TAIL_RATIONAL_LIVE_BYTES]>(
        2,
        "M24 endpoint-count exact scratch rationals",
    )?;
    Ok(peak.finish())
}

fn box_cut_search_peak_live_bytes(
    plan: ReluTailBoxCutOptimizerPlan,
) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    // The active restart owns five Box-length buffers. At restart R it can
    // overlap only R-1 previously published candidates, hence R+4 rather than
    // R+5. After publication the five restart-local buffers have been dropped.
    let retained_and_current_box_buffers = plan
        .restarts
        .checked_add(4)
        .and_then(|buffers| buffers.checked_mul(plan.box_variables))
        .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
            operation: "M24 search Box-buffer elements",
        })?;
    let float_elements = plan
        .value_dim
        .checked_mul(2)
        .and_then(|elements| elements.checked_add(retained_and_current_box_buffers))
        .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
            operation: "M24 aggregate search f64 elements",
        })?;
    let mut peak = ConstrainedZonotopePeakLiveBytes::new();
    peak.add_bytes(
        size_of::<BoxApproximateSearch>(),
        "M24 search result header bytes",
    )?;
    peak.add_elements::<BoxSearchVariable>(plan.box_variables, "M24 search variables")?;
    peak.add_elements::<Vec<f64>>(plan.restarts, "M24 search candidate headers")?;
    peak.add_elements::<f64>(float_elements, "M24 search scratch and retained candidates")?;
    peak.add_elements::<[u8; RELU_TAIL_RATIONAL_LIVE_BYTES]>(
        2,
        "M24 search exact endpoint scratch rationals",
    )?;
    Ok(peak.finish())
}

fn box_cut_search_retained_live_bytes(
    plan: ReluTailBoxCutOptimizerPlan,
) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    let candidate_elements = plan.box_variables.checked_mul(plan.restarts).ok_or(
        ConstrainedZonotopeCallBudgetError::ResourceOverflow {
            operation: "retained M24 candidate elements",
        },
    )?;
    let mut live = ConstrainedZonotopePeakLiveBytes::new();
    live.add_bytes(
        size_of::<BoxApproximateSearch>(),
        "retained M24 search header bytes",
    )?;
    live.add_elements::<BoxSearchVariable>(plan.box_variables, "retained M24 search variables")?;
    live.add_elements::<Vec<f64>>(plan.restarts, "retained M24 candidate headers")?;
    live.add_elements::<f64>(candidate_elements, "retained M24 candidate values")?;
    Ok(live.finish())
}

fn box_cut_exact_replay_peak_live_bytes(
    value_dim: usize,
    constraints: usize,
) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    let value_elements =
        value_dim
            .checked_mul(5)
            .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "M24 exact replay value-vector elements",
            })?;
    let constraint_elements =
        constraints
            .checked_mul(2)
            .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "M24 exact replay predicate-vector elements",
            })?;
    let rational_slots = RELU_TAIL_BOX_CUT_TRANSIENT_RATIONAL_SLOTS
        .checked_add(1)
        .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
            operation: "M24 exact replay rational slots",
        })?;
    let mut peak = ConstrainedZonotopePeakLiveBytes::new();
    peak.add_bytes(
        size_of::<ReluTailBoxCutCertificate>() + 2 * size_of::<Vec<f64>>(),
        "M24 exact replay result and expanded-vector headers",
    )?;
    peak.add_elements::<f64>(value_elements, "M24 exact replay value vectors")?;
    peak.add_elements::<f64>(constraint_elements, "M24 exact replay predicate vectors")?;
    peak.add_elements::<[u8; RELU_TAIL_RATIONAL_LIVE_BYTES]>(
        rational_slots,
        "M24 exact replay rational payloads",
    )?;
    peak.add_bytes(
        DUAL_SHAPE_ERROR_LIVE_BYTES,
        "M24 nested dual error allowance",
    )?;
    Ok(peak.finish())
}

fn relu_tail_peak_live_bytes(
    domain: &ConstrainedZonotope64,
    _generator_nonzeros: usize,
) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    let value_dim = domain.value_dim();
    let constraints = domain.constraint_count();
    let rational_slots = value_dim
        .checked_mul(4)
        .and_then(|slots| slots.checked_add(RELU_TAIL_TRANSIENT_RATIONAL_SLOTS))
        .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
            operation: "ReLU-tail exact rational slots",
        })?;

    let mut peak = ConstrainedZonotopePeakLiveBytes::new();
    peak.add_elements::<[u8; RELU_TAIL_RATIONAL_LIVE_BYTES]>(
        rational_slots,
        "ReLU-tail exact rational live bytes",
    )?;
    // Fixed/best/candidate directions, endpoint/canonical replay storage, and
    // the projected-Adam direction/moment buffers can overlap.
    peak.add_elements::<f64>(
        value_dim
            .checked_mul(8)
            .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "ReLU-tail direction buffer elements",
            })?,
        "ReLU-tail direction and search buffers",
    )?;
    peak.add_elements::<usize>(value_dim, "ReLU-tail candidate coordinate map")?;
    // `SlopeVariable`'s exact rational payload is charged above. Account for
    // its coordinate and two binary64 proposal fields separately.
    peak.add_elements::<(usize, f64, f64)>(value_dim, "ReLU-tail slope-variable metadata")?;
    peak.add_elements::<f64>(
        constraints
            .checked_mul(3)
            .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "ReLU-tail multiplier buffer elements",
            })?,
        "ReLU-tail multiplier buffers",
    )?;
    peak.add_bytes(
        DUAL_SHAPE_ERROR_LIVE_BYTES,
        "ReLU-tail nested dual error allowance",
    )?;
    Ok(peak.finish())
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

fn zero_f64_with_gate<G>(
    count: usize,
    resource: &'static str,
    checkpoint: &'static str,
    gate: &mut G,
) -> Result<Vec<f64>, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let mut values = Vec::new();
    try_reserve(&mut values, count, resource)?;
    for _ in 0..count {
        gate.charge_items(1, checkpoint)?;
        values.push(0.0);
    }
    Ok(values)
}

fn clone_f64_with_gate<G>(
    values: &[f64],
    resource: &'static str,
    checkpoint: &'static str,
    gate: &mut G,
) -> Result<Vec<f64>, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let mut result = Vec::new();
    try_reserve(&mut result, values.len(), resource)?;
    for &value in values {
        gate.charge_items(1, checkpoint)?;
        result.push(value);
    }
    Ok(result)
}

/// Materialize M20's typed portfolio member when its authenticated auxiliary
/// Box is a coordinatewise superset of the prepared exact hull.
///
/// In that case M20 has exactly the same line geometry, and M17's accepted
/// pointwise affine replay remains valid on the identical intersection.
/// Keeping that certificate as a duplicate result preserves the public
/// M17/M20 attribution contract without claiming that a fresh heuristic run
/// under another clock would visit the same candidates. Vector storage
/// remains fallible and the final checkpoint makes publication transactional.
/// The exact constant is already covered by the caller's two-result peak
/// preflight and by the immutable rational-size checks that constructed
/// `source`.
fn clone_equivalent_relu_tail_result_with_gate<G>(
    source: &ReluTailDualResult,
    gate: &mut G,
) -> Result<ReluTailDualResult, ReluTailDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    gate.checkpoint("equivalent M20 result copy admission")?;
    let direction = clone_f64_with_gate(
        &source.direction,
        "equivalent M20 direction",
        "equivalent M20 direction copy",
        gate,
    )?;
    let multipliers = clone_f64_with_gate(
        &source.multipliers,
        "equivalent M20 predicate multipliers",
        "equivalent M20 predicate-multiplier copy",
        gate,
    )?;
    gate.charge_items(1, "equivalent M20 exact-constant copy")?;
    let result = ReluTailDualResult {
        lower_bound: source.lower_bound,
        zero_multiplier_lower_bound: source.zero_multiplier_lower_bound,
        zero_predicate_candidate_replays: source.zero_predicate_candidate_replays,
        direction,
        multipliers,
        exact_constant: source.exact_constant.clone(),
        optimizable_slopes: source.optimizable_slopes,
        candidates_replayed: source.candidates_replayed,
        iterations_completed: source.iterations_completed,
        status: source.status,
        plan: source.plan,
        supplied_multipliers_used: source.supplied_multipliers_used,
    };
    gate.checkpoint("equivalent M20 result copy publication")?;
    Ok(result)
}

fn candidate_clone_f64_with_gate<G>(
    values: &[f64],
    checkpoint: &'static str,
    gate: &mut G,
) -> Result<Option<Vec<f64>>, ConstrainedZonotopeCallBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let mut result = Vec::new();
    if result.try_reserve_exact(values.len()).is_err() {
        return Ok(None);
    }
    for &value in values {
        gate.charge_items(1, checkpoint)?;
        result.push(value);
    }
    Ok(Some(result))
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
    use std::cell::Cell;
    use std::mem::{size_of, size_of_val};

    use ndarray::{array, Array2, Array4};
    use num_bigint::BigInt;
    use proptest::prelude::*;

    use super::*;
    use crate::constrained_zonotope_batch_norm::ConstrainedZonotopeBatchNormMode;

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

    fn pullback_limits() -> ReluTailConvBatchNormPullbackLimits {
        ReluTailConvBatchNormPullbackLimits {
            max_input_value_count: 64,
            max_output_value_count: 64,
            max_weight_elements: 64,
            max_kernel_visits: 1_024,
            max_pulled_margin_construction_exact_products: 4_096,
        }
    }

    fn batch_norm_certificate_limits(
        channels: usize,
    ) -> ConstrainedZonotopeBatchNormAffineCertificateLimits {
        ConstrainedZonotopeBatchNormAffineCertificateLimits {
            max_rank: 3,
            max_channel_count: channels,
            max_parameter_elements: 6 * channels,
        }
    }

    fn caller_retained_domain_live_bytes(domain: &ConstrainedZonotope64) -> usize {
        let generator_entry_bytes = domain
            .generators()
            .iter()
            .map(|generator| generator.nnz() * size_of::<(usize, f64)>())
            .sum::<usize>();
        size_of::<ConstrainedZonotope64>()
            + size_of_val(domain.center())
            + size_of_val(domain.generators())
            + generator_entry_bytes
            + domain.constraints().len() * size_of::<f64>()
            + size_of_val(domain.rhs())
            + size_of_val(domain.box_remainder())
    }

    fn synthetic_accepted_line(
        direction: Vec<f64>,
        exact_constant: BigRational,
    ) -> ReluTailDualResult {
        ReluTailDualResult {
            lower_bound: 0.0,
            zero_multiplier_lower_bound: 0.0,
            zero_predicate_candidate_replays: ReluTailDualZeroPredicateCandidateReplays {
                zero_positive_slope_lower_bound: 0.0,
                upper_endpoint_lower_bound: None,
                canonical_lower_bound: None,
                optimized_lower_bound: None,
            },
            direction,
            multipliers: Vec::new(),
            exact_constant,
            optimizable_slopes: 0,
            candidates_replayed: 1,
            iterations_completed: 0,
            status: ReluTailDualStatus::SearchDisabled,
            plan: None,
            supplied_multipliers_used: false,
        }
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

    fn assert_f64_slice_bit_identical(left: &[f64], right: &[f64]) {
        assert_eq!(
            left.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            right
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    fn assert_box_cut_bit_identical(
        left: &ReluTailBoxCutCertificate,
        right: &ReluTailBoxCutCertificate,
    ) {
        assert_eq!(left.lower_bound.to_bits(), right.lower_bound.to_bits());
        assert_eq!(
            left.zero_predicate_lower_bound.to_bits(),
            right.zero_predicate_lower_bound.to_bits()
        );
        assert_f64_slice_bit_identical(&left.replay_direction, &right.replay_direction);
        assert_f64_slice_bit_identical(&left.upper_box_multipliers, &right.upper_box_multipliers);
        assert_f64_slice_bit_identical(&left.lower_box_multipliers, &right.lower_box_multipliers);
        assert_f64_slice_bit_identical(&left.predicate_multipliers, &right.predicate_multipliers);
        assert_eq!(left.exact_constant, right.exact_constant);
        assert_eq!(
            left.supplied_predicate_multipliers_used,
            right.supplied_predicate_multipliers_used
        );
    }

    fn assert_optimized_box_cut_bit_identical(
        left: &ReluTailBoxCutOptimizedResult,
        right: &ReluTailBoxCutOptimizedResult,
    ) {
        assert_eq!(left.lower_bound.to_bits(), right.lower_bound.to_bits());
        assert_eq!(left.selected, right.selected);
        assert_eq!(
            left.portfolio.lower_bound.to_bits(),
            right.portfolio.lower_bound.to_bits()
        );
        assert_eq!(left.portfolio.selected, right.portfolio.selected);
        assert_eq!(left.portfolio.status, right.portfolio.status);
        assert_result_bit_identical(&left.portfolio.original, &right.portfolio.original);
        match (&left.portfolio.auxiliary, &right.portfolio.auxiliary) {
            (Some(left), Some(right)) => assert_result_bit_identical(left, right),
            (None, None) => {}
            mismatch => panic!("auxiliary portfolio mismatch: {mismatch:?}"),
        }
        match (&left.portfolio.box_cut, &right.portfolio.box_cut) {
            (Some(left), Some(right)) => assert_box_cut_bit_identical(left, right),
            (None, None) => {}
            mismatch => panic!("Box-cut portfolio mismatch: {mismatch:?}"),
        }
        assert_eq!(left.search_status, right.search_status);
        assert_eq!(left.search_plan, right.search_plan);
        assert_eq!(left.iterations_completed, right.iterations_completed);
        assert_eq!(left.restarts_completed, right.restarts_completed);
        assert_eq!(left.candidates_scored, right.candidates_scored);
        assert_eq!(left.exact_replays, right.exact_replays);
    }

    #[derive(Clone, Copy, Debug)]
    struct M24TestPeaks {
        m17: usize,
        m20: usize,
        endpoint_count: usize,
        search: usize,
        first_replay: usize,
        second_replay: usize,
        retained_certificate: usize,
    }

    fn oracle_sum(parts: &[usize]) -> usize {
        parts
            .iter()
            .try_fold(0_usize, |sum, &part| sum.checked_add(part))
            .expect("test byte-oracle sum must fit usize")
    }

    fn oracle_product(left: usize, right: usize) -> usize {
        left.checked_mul(right)
            .expect("test byte-oracle product must fit usize")
    }

    fn oracle_elements<T>(count: usize) -> usize {
        oracle_product(count, size_of::<T>())
    }

    fn independent_relu_tail_result_bytes(value_dim: usize, constraints: usize) -> usize {
        oracle_sum(&[
            size_of::<ReluTailDualResult>(),
            oracle_elements::<f64>(value_dim),
            oracle_elements::<f64>(constraints),
            oracle_elements::<[u8; RELU_TAIL_RATIONAL_LIVE_BYTES]>(1),
        ])
    }

    fn independent_m24_test_peaks(
        prepared: &PreparedReluTailGeometry64<'_>,
        plan: ReluTailBoxCutOptimizerPlan,
    ) -> M24TestPeaks {
        let value_dim = prepared.domain.value_dim();
        let constraints = prepared.domain.constraint_count();

        let result = independent_relu_tail_result_bytes(value_dim, constraints);
        let retained_results = oracle_product(result, 2);

        let relu_rational_slots = oracle_sum(&[
            oracle_product(value_dim, 4),
            RELU_TAIL_TRANSIENT_RATIONAL_SLOTS,
        ]);
        let m17 = oracle_sum(&[
            oracle_elements::<[u8; RELU_TAIL_RATIONAL_LIVE_BYTES]>(relu_rational_slots),
            oracle_elements::<f64>(oracle_product(value_dim, 8)),
            oracle_elements::<usize>(value_dim),
            oracle_elements::<(usize, f64, f64)>(value_dim),
            oracle_elements::<f64>(oracle_product(constraints, 3)),
            DUAL_SHAPE_ERROR_LIVE_BYTES,
        ]);
        let m20 = oracle_sum(&[m17, result]);

        let endpoint_count_owned = oracle_elements::<[u8; RELU_TAIL_RATIONAL_LIVE_BYTES]>(2);
        // The budgeted core now proves endpoint usefulness before retaining
        // or replaying M20, so only the mandatory M17 result overlaps this
        // scan.
        let endpoint_count = oracle_sum(&[result, endpoint_count_owned]);

        // Five current restart buffers overlap at most R-1 earlier candidates.
        // Direction and witness are the separate 2*value_dim term below.
        let search_box_buffers =
            oracle_product(oracle_sum(&[plan.restarts, 4]), plan.box_variables);
        let search_float_elements =
            oracle_sum(&[oracle_product(plan.value_dim, 2), search_box_buffers]);
        let search_owned = oracle_sum(&[
            size_of::<BoxApproximateSearch>(),
            oracle_elements::<BoxSearchVariable>(plan.box_variables),
            oracle_elements::<Vec<f64>>(plan.restarts),
            oracle_elements::<f64>(search_float_elements),
            oracle_elements::<[u8; RELU_TAIL_RATIONAL_LIVE_BYTES]>(2),
        ]);
        let search = oracle_sum(&[retained_results, search_owned]);

        let retained_candidate_elements = oracle_product(plan.box_variables, plan.restarts);
        let search_retained = oracle_sum(&[
            size_of::<BoxApproximateSearch>(),
            oracle_elements::<BoxSearchVariable>(plan.box_variables),
            oracle_elements::<Vec<f64>>(plan.restarts),
            oracle_elements::<f64>(retained_candidate_elements),
        ]);

        let exact_replay_header = oracle_sum(&[
            size_of::<ReluTailBoxCutCertificate>(),
            oracle_elements::<Vec<f64>>(2),
        ]);
        let exact_replay_rational_slots =
            oracle_sum(&[RELU_TAIL_BOX_CUT_TRANSIENT_RATIONAL_SLOTS, 1]);
        let exact_replay_owned = oracle_sum(&[
            exact_replay_header,
            oracle_elements::<f64>(oracle_product(value_dim, 5)),
            oracle_elements::<f64>(oracle_product(constraints, 2)),
            oracle_elements::<[u8; RELU_TAIL_RATIONAL_LIVE_BYTES]>(exact_replay_rational_slots),
            DUAL_SHAPE_ERROR_LIVE_BYTES,
        ]);
        let first_replay = oracle_sum(&[retained_results, search_retained, exact_replay_owned]);

        let retained_certificate = oracle_sum(&[
            size_of::<ReluTailBoxCutCertificate>(),
            oracle_elements::<f64>(oracle_product(value_dim, 3)),
            oracle_elements::<f64>(constraints),
            oracle_elements::<[u8; RELU_TAIL_RATIONAL_LIVE_BYTES]>(1),
        ]);
        let second_replay = oracle_sum(&[first_replay, retained_certificate]);

        assert_eq!(
            relu_tail_dual_result_live_bytes(value_dim, constraints).unwrap(),
            result,
            "production M17/M20 result accounting must match the independent oracle"
        );
        assert_eq!(
            relu_tail_peak_live_bytes(prepared.domain, prepared.generator_nonzeros).unwrap(),
            m17,
            "production M17 peak accounting must match the independent oracle"
        );
        assert_eq!(
            box_cut_endpoint_count_peak_live_bytes().unwrap(),
            endpoint_count_owned,
            "production M24 endpoint-count accounting must match the independent oracle"
        );
        assert_eq!(
            box_cut_search_peak_live_bytes(plan).unwrap(),
            search_owned,
            "production M24 search accounting must match the independent oracle"
        );
        assert_eq!(
            box_cut_search_retained_live_bytes(plan).unwrap(),
            search_retained,
            "production retained-search accounting must match the independent oracle"
        );
        assert_eq!(
            box_cut_exact_replay_peak_live_bytes(value_dim, constraints).unwrap(),
            exact_replay_owned,
            "production M24 exact-replay accounting must match the independent oracle"
        );
        assert_eq!(
            relu_tail_box_cut_certificate_live_bytes(value_dim, constraints).unwrap(),
            retained_certificate,
            "production retained-certificate accounting must match the independent oracle"
        );

        M24TestPeaks {
            m17,
            m20,
            endpoint_count,
            search,
            first_replay,
            second_replay,
            retained_certificate,
        }
    }

    fn exact_relu(value: &BigRational) -> BigRational {
        value.clone().max(BigRational::zero())
    }

    #[test]
    fn transactional_pullback_uses_shared_channel_scale_and_bias_errors() {
        let upstream_domain = ConstrainedZonotope64::from_certified_bounds(
            &[-1.0, 1.0],
            &[1.0, 4.0],
            &[false, false],
        )
        .unwrap();
        let downstream_domain = ConstrainedZonotope64::from_certified_bounds(
            &[-32.0, -32.0],
            &[32.0, 32.0],
            &[false, false],
        )
        .unwrap();
        let upstream = prepare_relu_tail_triangle_dual_unwired(&upstream_domain).unwrap();
        let downstream_prepared =
            prepare_relu_tail_triangle_dual_unwired(&downstream_domain).unwrap();
        let weights = Array4::from_shape_vec((1, 1, 1, 1), vec![1.0]).unwrap();
        let conv_bias = [5.0];
        let input_shape = [1, 1, 2];
        let gamma = [3.0];
        let beta = [7.0];
        let mean = [0.0];
        let variance = [0.0];
        let nominal_scale = [2.0];
        let nominal_bias = [6.0];
        let batch_norm_spec = ConstrainedZonotopeBatchNormSpec {
            input_shape: &input_shape,
            channel_axis: 0,
            gamma: &gamma,
            beta: &beta,
            mean: &mean,
            variance: &variance,
            epsilon: 1.0,
            mode: ConstrainedZonotopeBatchNormMode::Inference,
        };
        let conv_spec = ConstrainedZonotopeConv2dSpec {
            stride: [1, 1],
            padding: [0, 0, 0, 0],
            dilation: [1, 1],
            groups: 1,
        };
        let mut gate = InertConstrainedZonotopeCallGate;
        let plan = validate_conv2d_batch_norm_pullback_with_gate(
            &downstream_prepared,
            &upstream,
            input_shape,
            weights.view(),
            &conv_bias,
            conv_spec,
            batch_norm_spec,
            pullback_limits(),
            &mut gate,
        )
        .unwrap();
        let accepted = synthetic_accepted_line(vec![2.0, -3.0], ratio(1, 1));
        let pulled = build_conv2d_batch_norm_pulled_margin_with_gate(
            &upstream,
            &accepted,
            input_shape,
            weights.view(),
            &conv_bias,
            conv_spec,
            batch_norm_spec,
            &nominal_scale,
            &nominal_bias,
            batch_norm_certificate_limits(1),
            plan,
            relu_tail_dual_result_live_bytes(2, downstream_domain.constraint_count()).unwrap(),
            &mut gate,
        )
        .unwrap();

        // r = [2,-3], R = -1, and ReLU hulls are [0,1] and [1,4].
        // Hence [Smin,Smax]=[-12,-1], H=12, and
        // B = 1 + (2-3)*5 + R*6 - 1*H - 1*|R| = -23.
        assert_eq!(
            pulled.as_exact_margin().coefficients(),
            &[ratio(4, 1), ratio(-6, 1)]
        );
        assert_eq!(pulled.as_exact_margin().bias(), &ratio(-23, 1));
    }

    #[test]
    fn internally_pulled_margin_accepts_input_limit_plus_one_but_not_intermediate_plus_one() {
        let upstream_domain =
            ConstrainedZonotope64::from_certified_bounds(&[0.0], &[0.0], &[true]).unwrap();
        let downstream_domain =
            ConstrainedZonotope64::from_certified_bounds(&[0.0], &[0.0], &[true]).unwrap();
        let upstream = prepare_relu_tail_triangle_dual_unwired(&upstream_domain).unwrap();
        let downstream = prepare_relu_tail_triangle_dual_unwired(&downstream_domain).unwrap();
        let weights = Array4::from_shape_vec((1, 1, 1, 1), vec![1.0]).unwrap();
        let zeros = [0.0];
        let ones = [1.0];
        let input_shape = [1, 1, 1];
        let batch_norm_spec = ConstrainedZonotopeBatchNormSpec {
            input_shape: &input_shape,
            channel_axis: 0,
            gamma: &ones,
            beta: &zeros,
            mean: &zeros,
            variance: &zeros,
            epsilon: 1.0,
            mode: ConstrainedZonotopeBatchNormMode::Inference,
        };
        let conv_spec = ConstrainedZonotopeConv2dSpec {
            stride: [1, 1],
            padding: [0; 4],
            dilation: [1, 1],
            groups: 1,
        };
        let mut gate = InertConstrainedZonotopeCallGate;
        let plan = validate_conv2d_batch_norm_pullback_with_gate(
            &downstream,
            &upstream,
            input_shape,
            weights.view(),
            &zeros,
            conv_spec,
            batch_norm_spec,
            pullback_limits(),
            &mut gate,
        )
        .unwrap();
        let input_limit_plus_one = BigRational::from_integer(
            BigInt::from(1_u8)
                << usize::try_from(RELU_TAIL_DUAL_HARD_MAX_INPUT_RATIONAL_BITS).unwrap(),
        );
        assert_eq!(
            rational_bits(&input_limit_plus_one),
            RELU_TAIL_DUAL_HARD_MAX_INPUT_RATIONAL_BITS + 1
        );
        assert!(matches!(
            ExactReluTailMargin::try_new(vec![BigRational::zero()], input_limit_plus_one.clone()),
            Err(ReluTailDualError::RationalInputLimit {
                field: "bias",
                bits,
                limit: RELU_TAIL_DUAL_HARD_MAX_INPUT_RATIONAL_BITS,
                ..
            }) if bits == RELU_TAIL_DUAL_HARD_MAX_INPUT_RATIONAL_BITS + 1
        ));

        let intermediate_limit = BigRational::from_integer(
            BigInt::from(1_u8)
                << usize::try_from(RELU_TAIL_DUAL_HARD_MAX_INTERMEDIATE_RATIONAL_BITS - 1).unwrap(),
        );
        assert_eq!(
            rational_bits(&intermediate_limit),
            RELU_TAIL_DUAL_HARD_MAX_INTERMEDIATE_RATIONAL_BITS
        );
        check_internally_pulled_rationals_with_gate(
            std::slice::from_ref(&intermediate_limit),
            &intermediate_limit,
            &mut gate,
        )
        .unwrap();

        let pulled = build_conv2d_batch_norm_pulled_margin_with_gate(
            &upstream,
            &synthetic_accepted_line(vec![0.0], input_limit_plus_one.clone()),
            input_shape,
            weights.view(),
            &zeros,
            conv_spec,
            batch_norm_spec,
            &ones,
            &zeros,
            batch_norm_certificate_limits(1),
            plan,
            relu_tail_dual_result_live_bytes(1, downstream_domain.constraint_count()).unwrap(),
            &mut gate,
        )
        .unwrap();
        assert_eq!(pulled.as_exact_margin().bias(), &input_limit_plus_one);
        let replay = bound_prepared_internally_pulled_relu_tail_margin_impl(
            &upstream,
            &pulled,
            None,
            config(0),
            &mut gate,
        )
        .unwrap();
        assert_eq!(replay.exact_constant, input_limit_plus_one);
        let auxiliary_replay =
            bound_prepared_internally_pulled_relu_tail_margin_with_auxiliary_impl(
                &upstream,
                &auxiliary(vec![0.0], vec![0.0]),
                &pulled,
                None,
                config(0),
                0,
                &mut gate,
            )
            .unwrap();
        assert_eq!(auxiliary_replay.exact_constant, input_limit_plus_one);

        let intermediate_limit_plus_one = BigRational::from_integer(
            BigInt::from(1_u8)
                << usize::try_from(RELU_TAIL_DUAL_HARD_MAX_INTERMEDIATE_RATIONAL_BITS).unwrap(),
        );
        let coefficient_rejected = check_internally_pulled_rationals_with_gate(
            std::slice::from_ref(&intermediate_limit_plus_one),
            &BigRational::zero(),
            &mut gate,
        );
        assert!(matches!(
            coefficient_rejected,
            Err(ReluTailDualBudgetError::Bound(
                ReluTailDualError::RationalInputLimit {
                    field: "coefficients",
                    index: 0,
                    bits,
                    limit: RELU_TAIL_DUAL_HARD_MAX_INTERMEDIATE_RATIONAL_BITS,
                }
            )) if bits == RELU_TAIL_DUAL_HARD_MAX_INTERMEDIATE_RATIONAL_BITS + 1
        ));
        let rejected = build_conv2d_batch_norm_pulled_margin_with_gate(
            &upstream,
            &synthetic_accepted_line(vec![0.0], intermediate_limit_plus_one.clone()),
            input_shape,
            weights.view(),
            &zeros,
            conv_spec,
            batch_norm_spec,
            &ones,
            &zeros,
            batch_norm_certificate_limits(1),
            plan,
            relu_tail_dual_result_live_bytes(1, downstream_domain.constraint_count()).unwrap(),
            &mut gate,
        );
        assert!(matches!(
            rejected,
            Err(ReluTailConvBatchNormPullbackBudgetError::Transform(
                ReluTailConvBatchNormPullbackError::ReluTail(
                    ReluTailDualError::RationalGrowthLimit {
                        operation: "downstream exact-constant pullback",
                        bits,
                        limit: RELU_TAIL_DUAL_HARD_MAX_INTERMEDIATE_RATIONAL_BITS,
                        ..
                    }
                )
            )) if bits == RELU_TAIL_DUAL_HARD_MAX_INTERMEDIATE_RATIONAL_BITS + 1
        ));

        // Defense in depth: even an impossible, directly forged private seal
        // cannot bypass the upstream replay's independent validation scan.
        let forged = InternallyPulledReluTailMargin {
            margin: ExactReluTailMargin {
                coefficients: vec![BigRational::zero()],
                bias: intermediate_limit_plus_one,
            },
        };
        let replay_rejected = bound_prepared_internally_pulled_relu_tail_margin_impl(
            &upstream,
            &forged,
            None,
            config(0),
            &mut gate,
        );
        assert!(matches!(
            replay_rejected,
            Err(ReluTailDualBudgetError::Bound(
                ReluTailDualError::RationalInputLimit {
                    field: "bias",
                    bits,
                    limit: RELU_TAIL_DUAL_HARD_MAX_INTERMEDIATE_RATIONAL_BITS,
                    ..
                }
            )) if bits == RELU_TAIL_DUAL_HARD_MAX_INTERMEDIATE_RATIONAL_BITS + 1
        ));
        let auxiliary_replay_rejected =
            bound_prepared_internally_pulled_relu_tail_margin_with_auxiliary_impl(
                &upstream,
                &auxiliary(vec![0.0], vec![0.0]),
                &forged,
                None,
                config(0),
                0,
                &mut gate,
            );
        assert!(matches!(
            auxiliary_replay_rejected,
            Err(ReluTailDualBudgetError::Bound(
                ReluTailDualError::RationalInputLimit {
                    field: "bias",
                    bits,
                    limit: RELU_TAIL_DUAL_HARD_MAX_INTERMEDIATE_RATIONAL_BITS,
                    ..
                }
            )) if bits == RELU_TAIL_DUAL_HARD_MAX_INTERMEDIATE_RATIONAL_BITS + 1
        ));
    }

    #[test]
    fn internally_pulled_margin_still_enforces_total_rational_bit_cap() {
        let maximum_width = BigRational::from_integer(
            BigInt::from(1_u8)
                << usize::try_from(RELU_TAIL_DUAL_HARD_MAX_INTERMEDIATE_RATIONAL_BITS - 1).unwrap(),
        );
        assert_eq!(
            rational_bits(&maximum_width),
            RELU_TAIL_DUAL_HARD_MAX_INTERMEDIATE_RATIONAL_BITS
        );
        let coefficients = vec![maximum_width; 512];
        let mut gate = InertConstrainedZonotopeCallGate;
        let result = check_internally_pulled_rationals_with_gate(
            &coefficients,
            &BigRational::zero(),
            &mut gate,
        );
        assert!(matches!(
            result,
            Err(ReluTailDualBudgetError::Bound(ReluTailDualError::ResourceLimit {
                resource: "total objective rational bits",
                actual,
                limit: RELU_TAIL_DUAL_HARD_MAX_TOTAL_RATIONAL_BITS,
            })) if actual == RELU_TAIL_DUAL_HARD_MAX_TOTAL_RATIONAL_BITS + 1
        ));
    }

    #[test]
    fn transactional_pullback_shared_bias_error_cancels_when_channel_sum_is_zero() {
        let upstream_domain =
            ConstrainedZonotope64::from_certified_bounds(&[0.0, 0.0], &[1.0, 2.0], &[true, true])
                .unwrap();
        let downstream_domain = ConstrainedZonotope64::from_certified_bounds(
            &[-8.0, -8.0],
            &[8.0, 8.0],
            &[false, false],
        )
        .unwrap();
        let upstream = prepare_relu_tail_triangle_dual_unwired(&upstream_domain).unwrap();
        let downstream_prepared =
            prepare_relu_tail_triangle_dual_unwired(&downstream_domain).unwrap();
        let weights = Array4::from_shape_vec((1, 1, 1, 1), vec![1.0]).unwrap();
        let input_shape = [1, 1, 2];
        let gamma = [2.0];
        let beta = [7.0];
        let zero = [0.0];
        let nominal_scale = [2.0];
        let nominal_bias = [6.0];
        let batch_norm_spec = ConstrainedZonotopeBatchNormSpec {
            input_shape: &input_shape,
            channel_axis: 0,
            gamma: &gamma,
            beta: &beta,
            mean: &zero,
            variance: &zero,
            epsilon: 1.0,
            mode: ConstrainedZonotopeBatchNormMode::Inference,
        };
        let conv_spec = ConstrainedZonotopeConv2dSpec {
            stride: [1, 1],
            padding: [0; 4],
            dilation: [1, 1],
            groups: 1,
        };
        let mut gate = InertConstrainedZonotopeCallGate;
        let plan = validate_conv2d_batch_norm_pullback_with_gate(
            &downstream_prepared,
            &upstream,
            input_shape,
            weights.view(),
            &zero,
            conv_spec,
            batch_norm_spec,
            pullback_limits(),
            &mut gate,
        )
        .unwrap();
        let pulled = build_conv2d_batch_norm_pulled_margin_with_gate(
            &upstream,
            &synthetic_accepted_line(vec![1.0, -1.0], ratio(0, 1)),
            input_shape,
            weights.view(),
            &zero,
            conv_spec,
            batch_norm_spec,
            &nominal_scale,
            &nominal_bias,
            batch_norm_certificate_limits(1),
            plan,
            relu_tail_dual_result_live_bytes(2, downstream_domain.constraint_count()).unwrap(),
            &mut gate,
        )
        .unwrap();

        assert_eq!(
            pulled.as_exact_margin().coefficients(),
            &[ratio(2, 1), ratio(-2, 1)]
        );
        assert_eq!(pulled.as_exact_margin().bias(), &ratio(0, 1));
    }

    #[test]
    fn transactional_pullback_matches_grouped_conv2d_chw_transpose() {
        let upstream_domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0; 4], &[1.0; 4], &[false; 4])
                .unwrap();
        let downstream_domain =
            ConstrainedZonotope64::from_certified_bounds(&[-16.0; 4], &[16.0; 4], &[false; 4])
                .unwrap();
        let upstream = prepare_relu_tail_triangle_dual_unwired(&upstream_domain).unwrap();
        let downstream_prepared =
            prepare_relu_tail_triangle_dual_unwired(&downstream_domain).unwrap();
        let weights = Array4::from_shape_vec((2, 1, 1, 1), vec![2.0, 3.0]).unwrap();
        let conv_bias = [1.0, -2.0];
        let input_shape = [2, 1, 2];
        let ones = [1.0, 1.0];
        let zeros = [0.0, 0.0];
        let batch_norm_spec = ConstrainedZonotopeBatchNormSpec {
            input_shape: &input_shape,
            channel_axis: 0,
            gamma: &ones,
            beta: &zeros,
            mean: &zeros,
            variance: &zeros,
            epsilon: 1.0,
            mode: ConstrainedZonotopeBatchNormMode::Inference,
        };
        let conv_spec = ConstrainedZonotopeConv2dSpec {
            stride: [1, 1],
            padding: [0; 4],
            dilation: [1, 1],
            groups: 2,
        };
        let mut gate = InertConstrainedZonotopeCallGate;
        let plan = validate_conv2d_batch_norm_pullback_with_gate(
            &downstream_prepared,
            &upstream,
            input_shape,
            weights.view(),
            &conv_bias,
            conv_spec,
            batch_norm_spec,
            pullback_limits(),
            &mut gate,
        )
        .unwrap();
        let pulled = build_conv2d_batch_norm_pulled_margin_with_gate(
            &upstream,
            &synthetic_accepted_line(vec![1.0, 2.0, -1.0, 4.0], ratio(5, 1)),
            input_shape,
            weights.view(),
            &conv_bias,
            conv_spec,
            batch_norm_spec,
            &ones,
            &zeros,
            batch_norm_certificate_limits(2),
            plan,
            relu_tail_dual_result_live_bytes(4, downstream_domain.constraint_count()).unwrap(),
            &mut gate,
        )
        .unwrap();

        assert_eq!(
            pulled.as_exact_margin().coefficients(),
            &[ratio(2, 1), ratio(4, 1), ratio(-3, 1), ratio(12, 1)]
        );
        assert_eq!(pulled.as_exact_margin().bias(), &ratio(2, 1));
        assert_eq!(plan.output_shape, [2, 1, 2]);
    }

    #[test]
    fn transactional_pullback_matches_spatial_padding_stride_and_dilation_oracle() {
        let upstream_domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0; 20], &[1.0; 20], &[false; 20])
                .unwrap();
        let downstream_domain =
            ConstrainedZonotope64::from_certified_bounds(&[-128.0; 6], &[128.0; 6], &[false; 6])
                .unwrap();
        let upstream = prepare_relu_tail_triangle_dual_unwired(&upstream_domain).unwrap();
        let downstream_prepared =
            prepare_relu_tail_triangle_dual_unwired(&downstream_domain).unwrap();
        let weights = Array4::from_shape_vec((1, 1, 2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let conv_bias = [5.0];
        let input_shape = [1, 4, 5];
        let ones = [1.0];
        let zeros = [0.0];
        let batch_norm_spec = ConstrainedZonotopeBatchNormSpec {
            input_shape: &input_shape,
            channel_axis: 0,
            gamma: &ones,
            beta: &zeros,
            mean: &zeros,
            variance: &zeros,
            epsilon: 1.0,
            mode: ConstrainedZonotopeBatchNormMode::Inference,
        };
        let conv_spec = ConstrainedZonotopeConv2dSpec {
            stride: [2, 2],
            padding: [1, 1, 0, 0],
            dilation: [2, 1],
            groups: 1,
        };
        let mut gate = InertConstrainedZonotopeCallGate;
        let plan = validate_conv2d_batch_norm_pullback_with_gate(
            &downstream_prepared,
            &upstream,
            input_shape,
            weights.view(),
            &conv_bias,
            conv_spec,
            batch_norm_spec,
            pullback_limits(),
            &mut gate,
        )
        .unwrap();
        let accepted = synthetic_accepted_line(vec![1.0, -2.0, 3.0, -1.0, 2.0, -2.0], ratio(1, 1));
        let pulled = build_conv2d_batch_norm_pulled_margin_with_gate(
            &upstream,
            &accepted,
            input_shape,
            weights.view(),
            &conv_bias,
            conv_spec,
            batch_norm_spec,
            &ones,
            &zeros,
            batch_norm_certificate_limits(1),
            plan,
            relu_tail_dual_result_live_bytes(6, downstream_domain.constraint_count()).unwrap(),
            &mut gate,
        )
        .unwrap();

        // Independent CHW oracle. The two output rows start at padded input
        // y=-1 and y=1; dilation maps kernel rows to offsets 0 and 2. The
        // three output columns start at x=-1,1,3. Applying W^T to directions
        // [[1,-2,3],[-1,2,-2]] gives these four input rows.
        let expected = [
            0, 0, 0, 0, 0, 2, -4, -4, 7, 8, 0, 0, 0, 0, 0, -4, 6, 8, -6, -8,
        ]
        .into_iter()
        .map(|value| ratio(value, 1))
        .collect::<Vec<_>>();
        assert_eq!(plan.output_shape, [1, 2, 3]);
        assert_eq!(pulled.as_exact_margin().coefficients(), expected.as_slice());
        // The downstream direction sums to one, so broadcast Conv bias adds
        // five to the accepted exact constant one.
        assert_eq!(pulled.as_exact_margin().bias(), &ratio(6, 1));
    }

    #[test]
    fn public_transaction_wires_both_m17_lines_and_enforces_complete_caller_peak() {
        let upstream_domain = ConstrainedZonotope64::from_certified_bounds(
            &[-1.0, -1.0],
            &[1.0, 1.0],
            &[false, false],
        )
        .unwrap();
        let downstream_domain =
            ConstrainedZonotope64::from_certified_bounds(&[0.0, 0.0], &[1.0, 1.0], &[true, true])
                .unwrap();
        let upstream = prepare_relu_tail_triangle_dual_unwired(&upstream_domain).unwrap();
        let downstream = prepare_relu_tail_triangle_dual_unwired(&downstream_domain).unwrap();
        let final_margin = margin(vec![ratio(1, 1), ratio(-1, 1)], ratio(0, 1));
        let weights = Array4::from_shape_vec((1, 1, 1, 1), vec![1.0]).unwrap();
        let zeros = [0.0];
        let ones = [1.0];
        let input_shape = [1, 1, 2];
        let batch_norm_spec = ConstrainedZonotopeBatchNormSpec {
            input_shape: &input_shape,
            channel_axis: 0,
            gamma: &ones,
            beta: &zeros,
            mean: &zeros,
            variance: &zeros,
            epsilon: 1.0,
            mode: ConstrainedZonotopeBatchNormMode::Inference,
        };
        let conv_spec = ConstrainedZonotopeConv2dSpec {
            stride: [1, 1],
            padding: [0; 4],
            dilation: [1, 1],
            groups: 1,
        };
        let downstream_config = config(0);
        let upstream_config = config(0);
        let certificate_limits = batch_norm_certificate_limits(1);
        let construction_limits = pullback_limits();
        let caller_retained_objects = [
            upstream.conservative_live_bytes(),
            downstream.conservative_live_bytes(),
            caller_retained_domain_live_bytes(&upstream_domain),
            caller_retained_domain_live_bytes(&downstream_domain),
            exact_relu_tail_margin_live_bytes(final_margin.coefficients().len()).unwrap(),
            size_of_val(&weights) + weights.len() * size_of::<f64>(),
            size_of_val(&zeros),
            size_of_val(&ones),
            size_of_val(&input_shape),
            size_of_val(&batch_norm_spec),
            size_of_val(&conv_spec),
            size_of_val(&downstream_config),
            size_of_val(&upstream_config),
            size_of_val(&certificate_limits),
            size_of_val(&construction_limits),
        ];
        let baseline = caller_retained_objects
            .into_iter()
            .try_fold(0_usize, usize::checked_add)
            .unwrap();
        let deadline = Instant::now() + Duration::from_mins(1);
        let run = |max_peak_live_bytes| {
            downstream.bound_conv2d_batch_norm_pullback_unwired_with_budget(
                &final_margin,
                None,
                downstream_config,
                &upstream,
                input_shape,
                weights.view(),
                &zeros,
                conv_spec,
                batch_norm_spec,
                &ones,
                &zeros,
                certificate_limits,
                construction_limits,
                None,
                upstream_config,
                ConstrainedZonotopeCallBudget::new(deadline, baseline, max_peak_live_bytes),
            )
        };
        let completed = run(usize::MAX).unwrap();
        assert_eq!(completed.value().plan.output_shape, [1, 1, 2]);
        // The downstream hull is [0,1]^2, so M17 retains the exact final line
        // x_0 - x_1 with zero correction.  The mandatory outward replay places
        // the represented analytic lower bound -1 two binary64 ULPs downward.
        assert_eq!(completed.value().downstream.direction, vec![1.0, -1.0]);
        assert_eq!(completed.value().downstream.exact_constant, ratio(0, 1));
        assert_eq!(
            completed.value().downstream.lower_bound.to_bits(),
            (-1.0_f64).next_down().next_down().to_bits()
        );
        // Identity BN/Conv pulls that line back to ReLU(x_0)-ReLU(x_1).
        // On [-1,1]^2 the zero-positive-slope M17 line is
        // 0*x_0 - x_1/2 - 1/2.  Both halves are exactly representable, so this
        // replay attains the analytic lower bound -1 without an extra ULP.
        assert_eq!(completed.value().upstream.direction, vec![0.0, -0.5]);
        assert_eq!(completed.value().upstream.exact_constant, ratio(-1, 2));
        assert_eq!(
            completed.value().upstream.lower_bound.to_bits(),
            (-1.0_f64).to_bits()
        );
        let exact_peak = completed.report().peak_live_bytes();
        assert!(matches!(
            run(exact_peak - 1),
            Err(ReluTailConvBatchNormPullbackBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { .. }
            ))
        ));

        let start = Instant::now();
        let attempt = |bias: &[f64], budget| {
            downstream.bound_conv2d_batch_norm_pullback_unwired_attempt_with_clock(
                &final_margin,
                None,
                downstream_config,
                &upstream,
                input_shape,
                weights.view(),
                bias,
                conv_spec,
                batch_norm_spec,
                &ones,
                &zeros,
                certificate_limits,
                construction_limits,
                None,
                upstream_config,
                budget,
                |_| start,
            )
        };
        let (admission, admission_report) = attempt(
            &zeros,
            ConstrainedZonotopeCallBudget::new(start, baseline, usize::MAX),
        )
        .into_parts();
        assert!(matches!(
            admission,
            Err(ReluTailConvBatchNormPullbackBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "admission"
                }
            ))
        ));
        assert_eq!(admission_report.peak_live_bytes(), baseline);
        assert_eq!(admission_report.charged_items(), 0);
        assert_eq!(admission_report.deadline_polls(), 1);

        let (transform, transform_report) = attempt(
            &[],
            ConstrainedZonotopeCallBudget::new(
                start + Duration::from_secs(1),
                baseline,
                usize::MAX,
            ),
        )
        .into_parts();
        assert!(matches!(
            transform,
            Err(ReluTailConvBatchNormPullbackBudgetError::Transform(
                ReluTailConvBatchNormPullbackError::Conv2d(ConstrainedZonotopeConv2dError::Shape {
                    field: "bias",
                    ..
                })
            ))
        ));
        assert_eq!(transform_report.peak_live_bytes(), baseline);
        assert_eq!(transform_report.charged_items(), 0);
        assert_eq!(transform_report.deadline_polls(), 2);
    }

    #[test]
    fn retained_pullback_m20_strictly_wins_ties_and_falls_back_transactionally() {
        // x_0=t and x_1=2t.  Original M17 for
        // ReLU(x_0)-ReLU(x_1) is -x_1/2-1, whose replay gives -2.
        // The certified fact x_0>=0 makes M20's line x_0-x_1/2-1,
        // cancelling the shared generator and improving the bound to -1.
        let upstream_domain = ConstrainedZonotope64::try_new(
            vec![0.0, 0.0],
            vec![vec![(0, 1.0), (1, 2.0)]],
            Array2::zeros((0, 1)),
            Vec::new(),
            vec![0.0, 0.0],
        )
        .unwrap();
        let downstream_domain =
            ConstrainedZonotope64::from_certified_bounds(&[0.0, 0.0], &[1.0, 2.0], &[true, true])
                .unwrap();
        let upstream = prepare_relu_tail_triangle_dual_unwired(&upstream_domain).unwrap();
        let downstream = prepare_relu_tail_triangle_dual_unwired(&downstream_domain).unwrap();
        let final_margin = margin(vec![ratio(1, 1), ratio(-1, 1)], ratio(0, 1));
        let weights = Array4::from_shape_vec((1, 1, 1, 1), vec![1.0]).unwrap();
        let zeros = [0.0];
        let ones = [1.0];
        let input_shape = [1, 1, 2];
        let batch_norm_spec = ConstrainedZonotopeBatchNormSpec {
            input_shape: &input_shape,
            channel_axis: 0,
            gamma: &ones,
            beta: &zeros,
            mean: &zeros,
            variance: &zeros,
            epsilon: 1.0,
            mode: ConstrainedZonotopeBatchNormMode::Inference,
        };
        let conv_spec = ConstrainedZonotopeConv2dSpec {
            stride: [1, 1],
            padding: [0; 4],
            dilation: [1, 1],
            groups: 1,
        };
        let downstream_config = config(0);
        let upstream_config = config(0);
        let certificate_limits = batch_norm_certificate_limits(1);
        let construction_limits = pullback_limits();
        let strengthening = auxiliary(vec![0.0, -2.0], vec![1.0, 2.0]);
        let caller_retained_objects = [
            upstream.conservative_live_bytes(),
            downstream.conservative_live_bytes(),
            caller_retained_domain_live_bytes(&upstream_domain),
            caller_retained_domain_live_bytes(&downstream_domain),
            exact_relu_tail_margin_live_bytes(final_margin.coefficients().len()).unwrap(),
            size_of_val(&strengthening)
                + size_of_val(strengthening.lower())
                + size_of_val(strengthening.upper()),
            size_of_val(&weights) + weights.len() * size_of::<f64>(),
            size_of_val(&zeros),
            size_of_val(&ones),
            size_of_val(&input_shape),
            size_of_val(&batch_norm_spec),
            size_of_val(&conv_spec),
            size_of_val(&downstream_config),
            size_of_val(&upstream_config),
            size_of_val(&certificate_limits),
            size_of_val(&construction_limits),
        ];
        let baseline = caller_retained_objects
            .into_iter()
            .try_fold(0_usize, usize::checked_add)
            .unwrap();
        let deadline = Instant::now() + Duration::from_mins(1);
        let run = |auxiliary: &CertifiedAuxiliaryBounds64, max_peak_live_bytes| {
            downstream.bound_conv2d_batch_norm_pullback_m17_m20_unwired_with_budget(
                &final_margin,
                None,
                downstream_config,
                &upstream,
                auxiliary,
                input_shape,
                weights.view(),
                &zeros,
                conv_spec,
                batch_norm_spec,
                &ones,
                &zeros,
                certificate_limits,
                construction_limits,
                None,
                upstream_config,
                ConstrainedZonotopeCallBudget::new(deadline, baseline, max_peak_live_bytes),
            )
        };

        let completed = run(&strengthening, usize::MAX).unwrap();
        let legacy = downstream
            .bound_conv2d_batch_norm_pullback_unwired_with_budget(
                &final_margin,
                None,
                downstream_config,
                &upstream,
                input_shape,
                weights.view(),
                &zeros,
                conv_spec,
                batch_norm_spec,
                &ones,
                &zeros,
                certificate_limits,
                construction_limits,
                None,
                upstream_config,
                ConstrainedZonotopeCallBudget::new(deadline, baseline, usize::MAX),
            )
            .unwrap();
        assert_result_bit_identical(&legacy.value().downstream, &completed.value().downstream);
        assert_result_bit_identical(
            &legacy.value().upstream,
            &completed.value().upstream.original,
        );
        assert_eq!(
            completed.value().upstream.status,
            ReluTailBoxCutStatus::Completed
        );
        assert!(completed.value().optional_budget_error.is_none());
        assert_eq!(
            completed.value().upstream.selected,
            ReluTailBoxCutSelection::Auxiliary
        );
        assert!(completed.value().upstream.box_cut.is_none());
        let strengthened = completed.value().upstream.auxiliary.as_ref().unwrap();
        assert!(
            strengthened.lower_bound > completed.value().upstream.original.lower_bound,
            "M20 must strictly strengthen this correlated fixture"
        );
        assert_eq!(strengthened.direction, vec![1.0, -0.5]);
        assert_eq!(strengthened.exact_constant, ratio(-1, 1));
        // The exact analytic minimum is -1.  As in the downstream replay
        // above, mandatory outward accumulation places that represented
        // endpoint two binary64 ULPs downward.
        assert_eq!(
            strengthened.lower_bound.to_bits(),
            (-1.0_f64).next_down().next_down().to_bits()
        );
        // The same-location auxiliary premise restricts the concrete shared
        // generator to t in [0,1]. Its exact objective is -t, with minimum -1.
        for t in [0.0_f64, 0.25, 0.5, 1.0] {
            let concrete = t.max(0.0) - (2.0 * t).max(0.0);
            assert!(concrete >= -1.0);
        }

        // The optional peak is strictly above the mandatory transaction peak.
        // Refusing its last byte must return M17 rather than fail the call.
        let exact_peak = completed.report().peak_live_bytes();
        let expected_transform_owned_peak = [
            relu_tail_dual_result_live_bytes(
                downstream.value_dim(),
                downstream_domain.constraint_count(),
            )
            .unwrap(),
            exact_relu_tail_margin_live_bytes(upstream.value_dim()).unwrap(),
            relu_tail_dual_result_live_bytes(
                upstream.value_dim(),
                upstream_domain.constraint_count(),
            )
            .unwrap(),
            relu_tail_peak_live_bytes(&upstream_domain, upstream.generator_nonzeros).unwrap(),
            size_of::<ReluTailConvBatchNormPullbackM17M20Result>(),
        ]
        .into_iter()
        .try_fold(0_usize, usize::checked_add)
        .unwrap();
        assert_eq!(
            exact_peak,
            baseline.checked_add(expected_transform_owned_peak).unwrap()
        );
        let peak_fallback = run(&strengthening, exact_peak - 1).unwrap();
        assert_eq!(
            peak_fallback.value().upstream.status,
            ReluTailBoxCutStatus::AuxiliaryFallback
        );
        assert_eq!(
            peak_fallback.value().upstream.selected,
            ReluTailBoxCutSelection::Original
        );
        assert!(peak_fallback.value().upstream.auxiliary.is_none());
        assert!(matches!(
            peak_fallback.value().optional_budget_error.as_ref(),
            Some(ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { .. })
        ));
        assert_result_bit_identical(
            &completed.value().upstream.original,
            &peak_fallback.value().upstream.original,
        );

        let nonrestrictive = auxiliary(vec![-1.0, -2.0], vec![1.0, 2.0]);
        let tied = run(&nonrestrictive, usize::MAX).unwrap();
        assert_eq!(
            tied.value().upstream.status,
            ReluTailBoxCutStatus::Completed
        );
        assert!(tied.value().optional_budget_error.is_none());
        assert_eq!(
            tied.value().upstream.selected,
            ReluTailBoxCutSelection::Original
        );
        assert_result_bit_identical(
            &tied.value().upstream.original,
            tied.value().upstream.auxiliary.as_ref().unwrap(),
        );

        for rejected in [
            auxiliary(vec![-1.0], vec![1.0]),
            auxiliary(vec![2.0, -2.0], vec![3.0, 2.0]),
        ] {
            let fallback = run(&rejected, usize::MAX).unwrap();
            assert_eq!(
                fallback.value().upstream.status,
                ReluTailBoxCutStatus::AuxiliaryFallback
            );
            assert_eq!(
                fallback.value().upstream.selected,
                ReluTailBoxCutSelection::Original
            );
            assert!(fallback.value().upstream.auxiliary.is_none());
            assert!(fallback.value().optional_budget_error.is_none());
        }

        let start = Instant::now();
        let expired = start + Duration::from_secs(2);
        let target = "prepared internally pulled auxiliary ReLU-tail validation";
        let target_polled = Cell::new(false);
        let attempt = downstream
            .bound_conv2d_batch_norm_pullback_m17_m20_unwired_attempt_with_clock(
                &final_margin,
                None,
                downstream_config,
                &upstream,
                &strengthening,
                input_shape,
                weights.view(),
                &zeros,
                conv_spec,
                batch_norm_spec,
                &ones,
                &zeros,
                certificate_limits,
                construction_limits,
                None,
                upstream_config,
                ConstrainedZonotopeCallBudget::new(
                    start + Duration::from_secs(1),
                    baseline,
                    usize::MAX,
                ),
                |checkpoint| {
                    if checkpoint == target {
                        target_polled.set(true);
                        expired
                    } else {
                        start
                    }
                },
            );
        assert!(target_polled.get());
        let (deadline_result, deadline_report) = attempt.into_parts();
        let deadline_result = deadline_result.unwrap();
        assert_eq!(
            deadline_result.upstream.status,
            ReluTailBoxCutStatus::AuxiliaryFallback
        );
        assert_eq!(
            deadline_result.upstream.selected,
            ReluTailBoxCutSelection::Original
        );
        assert!(deadline_result.upstream.auxiliary.is_none());
        assert!(matches!(
            deadline_result.optional_budget_error,
            Some(ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint })
                if checkpoint == target
        ));
        assert!(deadline_report.deadline_polls() > 1);
    }

    #[test]
    fn transactional_pulled_margin_final_scan_polls_inside_the_rational_walk() {
        let dimension = crate::CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL;
        // Keep the domain alpha-free so this test isolates the pulled-margin
        // rational walk rather than hitting the independent symbol ceiling.
        // The dense Conv transpose still constructs `dimension` coefficients.
        let lower = vec![0.0; dimension];
        let upper = vec![0.0; dimension];
        let declared_point = vec![true; dimension];
        let upstream_domain =
            ConstrainedZonotope64::from_certified_bounds(&lower, &upper, &declared_point).unwrap();
        let downstream_domain =
            ConstrainedZonotope64::from_certified_bounds(&[0.0], &[1.0], &[true]).unwrap();
        let upstream = prepare_relu_tail_triangle_dual_unwired(&upstream_domain).unwrap();
        let downstream = prepare_relu_tail_triangle_dual_unwired(&downstream_domain).unwrap();
        let final_margin = margin(vec![ratio(1, 1)], ratio(0, 1));
        let weights = Array4::from_elem((1, 1, 1, dimension), 1.0);
        let zeros = [0.0];
        let ones = [1.0];
        let input_shape = [1, 1, dimension];
        let batch_norm_spec = ConstrainedZonotopeBatchNormSpec {
            input_shape: &input_shape,
            channel_axis: 0,
            gamma: &ones,
            beta: &zeros,
            mean: &zeros,
            variance: &zeros,
            epsilon: 1.0,
            mode: ConstrainedZonotopeBatchNormMode::Inference,
        };
        let conv_spec = ConstrainedZonotopeConv2dSpec {
            stride: [1, 1],
            padding: [0; 4],
            dilation: [1, 1],
            groups: 1,
        };
        let start = Instant::now();
        let expired = start + Duration::from_secs(2);
        let target = "ReLU-tail declared-rational validation";
        let target_polled = Cell::new(false);
        let construction_limits = ReluTailConvBatchNormPullbackLimits {
            max_input_value_count: dimension,
            max_output_value_count: 1,
            max_weight_elements: dimension,
            max_kernel_visits: dimension,
            max_pulled_margin_construction_exact_products: dimension * 4 + 4,
        };
        let attempt = downstream.bound_conv2d_batch_norm_pullback_unwired_attempt_with_clock(
            &final_margin,
            None,
            config(0),
            &upstream,
            input_shape,
            weights.view(),
            &zeros,
            conv_spec,
            batch_norm_spec,
            &ones,
            &zeros,
            batch_norm_certificate_limits(1),
            construction_limits,
            None,
            config(0),
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
            |checkpoint| {
                if checkpoint == target {
                    target_polled.set(true);
                    expired
                } else {
                    start
                }
            },
        );
        assert!(target_polled.get());
        let (result, report) = attempt.into_parts();
        assert!(matches!(
            result,
            Err(ReluTailConvBatchNormPullbackBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
            )) if checkpoint == target
        ));
        assert!(report.peak_live_bytes() > 0);
        assert!(report.charged_items() >= dimension);
        assert!(report.deadline_polls() > 1);
    }

    #[test]
    fn budgeted_m17_is_bit_identical_and_reports_exact_peak() {
        let domain = ConstrainedZonotope64::try_new(
            vec![0.0, 0.25],
            vec![vec![(0, 1.0), (1, -0.5)]],
            array![[1.0]],
            vec![0.25],
            vec![0.0, 0.0],
        )
        .unwrap();
        let declared = margin(vec![ratio(1, 1), ratio(-1, 2)], ratio(1, 7));
        let supplied = [0.5];
        let legacy =
            bound_relu_tail_triangle_dual_unwired(&domain, &declared, Some(&supplied), config(3))
                .unwrap();

        let baseline = 37_usize;
        let deadline = Instant::now() + Duration::from_mins(1);
        let budgeted = bound_relu_tail_triangle_dual_unwired_with_budget(
            &domain,
            &declared,
            Some(&supplied),
            config(3),
            ConstrainedZonotopeCallBudget::new(deadline, baseline, usize::MAX),
        )
        .unwrap();
        assert_result_bit_identical(&legacy, budgeted.value());

        let value_dim = domain.value_dim();
        let constraints = domain.constraint_count();
        let rational_slots = 4 * value_dim + RELU_TAIL_TRANSIENT_RATIONAL_SLOTS;
        let transform_peak = rational_slots * RELU_TAIL_RATIONAL_LIVE_BYTES
            + 8 * value_dim * size_of::<f64>()
            + value_dim * size_of::<usize>()
            + value_dim * size_of::<(usize, f64, f64)>()
            + 3 * constraints * size_of::<f64>()
            + DUAL_SHAPE_ERROR_LIVE_BYTES;
        assert_eq!(
            budgeted.report().peak_live_bytes(),
            baseline + transform_peak
        );
        assert!(budgeted.report().charged_items() > 0);
        assert!(budgeted.report().deadline_polls() > 0);

        let at_boundary = bound_relu_tail_triangle_dual_unwired_with_budget(
            &domain,
            &declared,
            Some(&supplied),
            config(3),
            ConstrainedZonotopeCallBudget::new(deadline, baseline, baseline + transform_peak),
        )
        .unwrap();
        assert_result_bit_identical(&legacy, at_boundary.value());
        assert_eq!(
            at_boundary.report().peak_live_bytes(),
            baseline + transform_peak
        );
        assert!(matches!(
            bound_relu_tail_triangle_dual_unwired_with_budget(
                &domain,
                &declared,
                Some(&supplied),
                config(3),
                ConstrainedZonotopeCallBudget::new(
                    deadline,
                    baseline,
                    baseline + transform_peak - 1,
                ),
            ),
            Err(ReluTailDualBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                    required,
                    limit
                }
            )) if required == baseline + transform_peak
                && limit == baseline + transform_peak - 1
        ));
    }

    #[test]
    fn budgeted_m17_preserves_errors_and_refuses_admission_overflow_and_publication() {
        let domain = ConstrainedZonotope64::from_certified_bounds(
            &[-1.0, -1.0],
            &[1.0, 1.0],
            &[false, false],
        )
        .unwrap();
        let malformed = margin(vec![ratio(1, 1)], ratio(0, 1));
        let legacy = bound_relu_tail_triangle_dual_unwired(&domain, &malformed, None, config(0))
            .unwrap_err();
        let deadline = Instant::now() + Duration::from_mins(1);
        assert_eq!(
            bound_relu_tail_triangle_dual_unwired_with_budget(
                &domain,
                &malformed,
                None,
                config(0),
                ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
            )
            .unwrap_err(),
            ReluTailDualBudgetError::Bound(legacy)
        );

        let start = Instant::now();
        let reads = Cell::new(0_usize);
        let expired = bound_relu_tail_triangle_dual_unwired_with_clock(
            &domain,
            &malformed,
            None,
            config(0),
            ConstrainedZonotopeCallBudget::new(start, usize::MAX, 0),
            |_| {
                reads.set(reads.get() + 1);
                start
            },
        );
        assert!(matches!(
            expired,
            Err(ReluTailDualBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "admission"
                }
            ))
        ));
        assert_eq!(reads.get(), 1);

        let valid = margin(vec![ratio(1, 1), ratio(1, 1)], ratio(0, 1));
        let overflow = bound_relu_tail_triangle_dual_unwired_with_clock(
            &domain,
            &valid,
            None,
            config(0),
            ConstrainedZonotopeCallBudget::new(
                start + Duration::from_secs(1),
                usize::MAX,
                usize::MAX,
            ),
            |_| start,
        );
        assert!(matches!(
            overflow,
            Err(ReluTailDualBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                    operation: "aggregate peak-live bytes"
                }
            ))
        ));

        let active = ConstrainedZonotope64::from_certified_bounds(&[1.0], &[1.0], &[true]).unwrap();
        let active_margin = margin(vec![ratio(1, 1)], ratio(0, 1));
        let publication = bound_relu_tail_triangle_dual_unwired_with_clock(
            &active,
            &active_margin,
            None,
            config(0),
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
            |checkpoint| {
                if checkpoint == "ReLU-tail certificate publication" {
                    start + Duration::from_secs(2)
                } else {
                    start
                }
            },
        );
        assert!(matches!(
            publication,
            Err(ReluTailDualBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "ReLU-tail certificate publication"
                }
            ))
        ));
    }

    #[test]
    fn budgeted_m17_polls_exact_setup_generator_entries_and_candidate_search() {
        let dimension = crate::CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL;
        let fixed = vec![true; dimension];
        let lower = vec![0.0; dimension];
        let upper = vec![0.0; dimension];
        let domain = ConstrainedZonotope64::from_certified_bounds(&lower, &upper, &fixed).unwrap();
        let declared = margin(vec![BigRational::zero(); dimension], BigRational::zero());
        let start = Instant::now();
        let expired = start + Duration::from_secs(2);
        for phase in [
            "ReLU-tail declared-rational validation",
            "ReLU-tail exact radius initialization",
            "ReLU-tail exact coordinate-bound materialization",
            "ReLU-tail exact line construction",
        ] {
            let result = bound_relu_tail_triangle_dual_unwired_with_clock(
                &domain,
                &declared,
                None,
                config(0),
                ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
                |checkpoint| {
                    if checkpoint == phase {
                        expired
                    } else {
                        start
                    }
                },
            );
            assert!(
                matches!(
                    result,
                    Err(ReluTailDualBudgetError::Budget(
                        ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
                    )) if checkpoint == phase
                ),
                "deadline must be polled during {phase}"
            );
        }

        let entries = (0..dimension).map(|index| (index, 1.0)).collect();
        let sparse = ConstrainedZonotope64::try_new(
            vec![0.0; dimension],
            vec![entries],
            Array2::zeros((0, 1)),
            Vec::new(),
            vec![0.0; dimension],
        )
        .unwrap();
        let generator_entry = bound_relu_tail_triangle_dual_unwired_with_clock(
            &sparse,
            &declared,
            None,
            config(0),
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
            |checkpoint| {
                if checkpoint == "ReLU-tail exact generator-entry accumulation" {
                    expired
                } else {
                    start
                }
            },
        );
        assert!(matches!(
            generator_entry,
            Err(ReluTailDualBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "ReLU-tail exact generator-entry accumulation"
                }
            ))
        ));

        let unstable =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
        let unstable_margin = margin(vec![ratio(1, 1)], ratio(0, 1));
        let candidate = bound_relu_tail_triangle_dual_unwired_with_clock(
            &unstable,
            &unstable_margin,
            None,
            config(1),
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
            |checkpoint| {
                if checkpoint == "ReLU-tail candidate startup" {
                    expired
                } else {
                    start
                }
            },
        );
        assert!(matches!(
            candidate,
            Err(ReluTailDualBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "ReLU-tail candidate startup"
                }
            ))
        ));
    }

    #[test]
    fn budgeted_prepared_m17_and_m20_match_direct_results_with_one_hull_pass() {
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
        let supplied = [0.25, 0.0];
        let expected_m17: Vec<_> = margins
            .iter()
            .map(|declared| {
                bound_relu_tail_triangle_dual_unwired(&domain, declared, Some(&supplied), config(3))
                    .unwrap()
            })
            .collect();
        let expected_m20: Vec<_> = margins
            .iter()
            .map(|declared| {
                bound_relu_tail_triangle_dual_with_auxiliary_bounds_unwired(
                    &domain,
                    &certified,
                    declared,
                    Some(&supplied),
                    config(3),
                )
                .unwrap()
            })
            .collect();

        EXACT_COORDINATE_HULL_PASSES.with(|passes| passes.set(0));
        let deadline = Instant::now() + Duration::from_mins(1);
        let prepared_outcome = prepare_relu_tail_triangle_dual_unwired_with_budget(
            &domain,
            ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
        )
        .unwrap();
        let prepared = prepared_outcome.into_value();
        let baseline = prepared.conservative_live_bytes();
        for ((declared, expected_m17), expected_m20) in
            margins.iter().zip(&expected_m17).zip(&expected_m20)
        {
            let m17 = prepared
                .bound_margin_unwired_with_budget(
                    declared,
                    Some(&supplied),
                    config(3),
                    ConstrainedZonotopeCallBudget::new(deadline, baseline, usize::MAX),
                )
                .unwrap();
            assert_result_bit_identical(expected_m17, m17.value());
            let m17_report = m17.report();
            drop(m17);
            let m20 = prepared
                .bound_margin_with_auxiliary_bounds_unwired_with_budget(
                    &certified,
                    declared,
                    Some(&supplied),
                    config(3),
                    ConstrainedZonotopeCallBudget::new(deadline, baseline, usize::MAX),
                )
                .unwrap();
            assert_result_bit_identical(expected_m20, m20.value());
            let m20_report = m20.report();
            drop(m20);

            let portfolio = prepared
                .bound_margin_m17_m20_unwired_with_budget(
                    &certified,
                    declared,
                    Some(&supplied),
                    config(3),
                    ConstrainedZonotopeCallBudget::new(deadline, baseline, usize::MAX),
                )
                .unwrap();
            assert_result_bit_identical(expected_m17, &portfolio.value().original);
            assert_result_bit_identical(
                expected_m20,
                portfolio.value().auxiliary.as_ref().unwrap(),
            );
            assert_eq!(portfolio.value().box_cut, None);
            assert_eq!(portfolio.value().status, ReluTailBoxCutStatus::Completed);
            assert_eq!(
                portfolio.report().charged_items(),
                m17_report
                    .charged_items()
                    .checked_add(m20_report.charged_items())
                    .unwrap()
            );
            assert_eq!(
                portfolio.report().deadline_polls(),
                m17_report
                    .deadline_polls()
                    .checked_add(m20_report.deadline_polls())
                    .and_then(|polls| polls.checked_sub(1))
                    .unwrap()
            );
            assert!(portfolio.report().peak_live_bytes() > m17_report.peak_live_bytes());
            assert!(portfolio.report().peak_live_bytes() > m20_report.peak_live_bytes());
            let (expected_lower_bound, expected_selection) =
                if expected_m20.lower_bound > expected_m17.lower_bound {
                    (expected_m20.lower_bound, ReluTailBoxCutSelection::Auxiliary)
                } else {
                    (expected_m17.lower_bound, ReluTailBoxCutSelection::Original)
                };
            assert_eq!(portfolio.value().selected, expected_selection);
            assert_eq!(
                portfolio.value().lower_bound.to_bits(),
                expected_lower_bound.to_bits()
            );
        }
        EXACT_COORDINATE_HULL_PASSES.with(|passes| assert_eq!(passes.get(), 1));
    }

    #[test]
    fn prepared_m17_m20_portfolio_accounts_auxiliary_fallback() {
        let domain = ConstrainedZonotope64::from_certified_bounds(
            &[-1.0, -1.0],
            &[1.0, 1.0],
            &[false, false],
        )
        .unwrap();
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        let declared = margin(vec![ratio(1, 1), ratio(-1, 1)], ratio(0, 1));
        let wrong_dimension = auxiliary(vec![-1.0], vec![1.0]);
        let deadline = Instant::now() + Duration::from_mins(1);
        let m17 = prepared
            .bound_margin_unwired_with_budget(
                &declared,
                None,
                config(0),
                ConstrainedZonotopeCallBudget::new(
                    deadline,
                    prepared.conservative_live_bytes(),
                    usize::MAX,
                ),
            )
            .unwrap();
        let outcome = prepared
            .bound_margin_m17_m20_unwired_with_budget(
                &wrong_dimension,
                &declared,
                None,
                config(0),
                ConstrainedZonotopeCallBudget::new(
                    deadline,
                    prepared.conservative_live_bytes(),
                    usize::MAX,
                ),
            )
            .expect("optional M20 shape failure must retain accounted M17 authority");
        let portfolio = outcome.value();
        assert_eq!(portfolio.status, ReluTailBoxCutStatus::AuxiliaryFallback);
        assert_eq!(portfolio.selected, ReluTailBoxCutSelection::Original);
        assert_eq!(portfolio.auxiliary, None);
        assert_eq!(portfolio.box_cut, None);
        assert_eq!(
            portfolio.lower_bound.to_bits(),
            portfolio.original.lower_bound.to_bits()
        );
        assert_eq!(
            outcome.report().charged_items(),
            m17.report().charged_items()
        );
        assert_eq!(
            outcome.report().deadline_polls(),
            m17.report().deadline_polls() + 1
        );
    }

    #[test]
    fn prepared_m17_m20_portfolio_uses_strict_stable_selection() {
        let tied_domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
        let tied_auxiliary = auxiliary(vec![-1.0], vec![1.0]);
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let tied_prepared = prepare_relu_tail_triangle_dual_unwired(&tied_domain).unwrap();
        let deadline = Instant::now() + Duration::from_mins(1);
        let tied = tied_prepared
            .bound_margin_m17_m20_unwired_with_budget(
                &tied_auxiliary,
                &declared,
                None,
                config(0),
                ConstrainedZonotopeCallBudget::new(
                    deadline,
                    tied_prepared.conservative_live_bytes(),
                    usize::MAX,
                ),
            )
            .unwrap();
        assert_eq!(tied.value().status, ReluTailBoxCutStatus::Completed);
        assert_eq!(tied.value().selected, ReluTailBoxCutSelection::Original);
        assert_result_bit_identical(
            &tied.value().original,
            tied.value().auxiliary.as_ref().unwrap(),
        );
        assert_eq!(
            tied.value().lower_bound.to_bits(),
            tied.value().original.lower_bound.to_bits()
        );

        let improving_domain =
            ConstrainedZonotope64::from_certified_bounds(&[-2.0], &[2.0], &[false]).unwrap();
        let improving_auxiliary = auxiliary(vec![-2.0], vec![1.0]);
        let negative_margin = margin(vec![ratio(-1, 1)], ratio(0, 1));
        let improving_prepared =
            prepare_relu_tail_triangle_dual_unwired(&improving_domain).unwrap();
        let improved = improving_prepared
            .bound_margin_m17_m20_unwired_with_budget(
                &improving_auxiliary,
                &negative_margin,
                None,
                config(0),
                ConstrainedZonotopeCallBudget::new(
                    deadline,
                    improving_prepared.conservative_live_bytes(),
                    usize::MAX,
                ),
            )
            .unwrap();
        assert_eq!(improved.value().status, ReluTailBoxCutStatus::Completed);
        assert_eq!(
            improved.value().selected,
            ReluTailBoxCutSelection::Auxiliary
        );
        assert!(
            improved.value().auxiliary.as_ref().unwrap().lower_bound
                > improved.value().original.lower_bound
        );
        assert_eq!(
            improved.value().lower_bound.to_bits(),
            improved
                .value()
                .auxiliary
                .as_ref()
                .unwrap()
                .lower_bound
                .to_bits()
        );
    }

    #[test]
    fn prepared_attempt_receipts_survive_admission_and_peak_failures() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
        let start = Instant::now();
        let baseline = 37_usize;

        let (admission, admission_report) =
            prepare_relu_tail_triangle_dual_unwired_attempt_with_clock(
                &domain,
                ConstrainedZonotopeCallBudget::new(start, baseline, usize::MAX),
                |_| start,
            )
            .into_parts();
        assert!(matches!(
            admission,
            Err(ReluTailDualBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "admission"
                }
            ))
        ));
        assert_eq!(admission_report.peak_live_bytes(), baseline);
        assert_eq!(admission_report.charged_items(), 0);
        assert_eq!(admission_report.deadline_polls(), 1);

        let (peak, peak_report) = prepare_relu_tail_triangle_dual_unwired_attempt_with_clock(
            &domain,
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), baseline, baseline),
            |_| start,
        )
        .into_parts();
        assert!(matches!(
            peak,
            Err(ReluTailDualBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                    required,
                    limit
                }
            )) if required > baseline && limit == baseline
        ));
        assert_eq!(peak_report.peak_live_bytes(), baseline);
        assert!(peak_report.deadline_polls() >= 3);
    }

    #[test]
    fn budgeted_prepared_geometry_accounts_retained_bytes_and_exact_peaks() {
        let domain = ConstrainedZonotope64::try_new(
            vec![0.0, 1.0],
            vec![vec![(0, 1.0), (1, -0.5)]],
            array![[1.0]],
            vec![0.25],
            vec![0.0, 0.0],
        )
        .unwrap();
        let declared = margin(vec![ratio(1, 1), ratio(-1, 2)], ratio(1, 7));
        let certified = auxiliary(vec![-0.75, 0.75], vec![0.75, 1.25]);
        let start = Instant::now();
        let deadline = start + Duration::from_mins(1);
        let baseline = 37_usize;

        let first = prepare_relu_tail_triangle_dual_unwired_with_clock(
            &domain,
            ConstrainedZonotopeCallBudget::new(deadline, baseline, usize::MAX),
            |_| start,
        )
        .unwrap();
        let preparation_peak = first.report().peak_live_bytes();
        let conservative_live_bytes = first.value().conservative_live_bytes();
        let expected_live_bytes = size_of::<PreparedReluTailGeometry64<'static>>()
            + domain.value_dim() * size_of::<(BigRational, BigRational)>()
            + 2 * domain.value_dim() * RELU_TAIL_RATIONAL_LIVE_BYTES;
        assert_eq!(conservative_live_bytes, expected_live_bytes);
        assert!(preparation_peak > baseline + conservative_live_bytes);
        drop(first);

        let at_boundary = prepare_relu_tail_triangle_dual_unwired_with_clock(
            &domain,
            ConstrainedZonotopeCallBudget::new(deadline, baseline, preparation_peak),
            |_| start,
        )
        .unwrap();
        assert_eq!(at_boundary.report().peak_live_bytes(), preparation_peak);
        drop(at_boundary);
        assert!(matches!(
            prepare_relu_tail_triangle_dual_unwired_with_clock(
                &domain,
                ConstrainedZonotopeCallBudget::new(deadline, baseline, preparation_peak - 1),
                |_| start,
            ),
            Err(ReluTailDualBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { required, limit }
            )) if required == preparation_peak && limit == preparation_peak - 1
        ));

        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        assert_eq!(prepared.conservative_live_bytes(), conservative_live_bytes);
        let call_baseline = 43_usize + prepared.conservative_live_bytes();
        let first = prepared
            .bound_margin_with_auxiliary_bounds_unwired_with_clock(
                &certified,
                &declared,
                None,
                config(3),
                ConstrainedZonotopeCallBudget::new(deadline, call_baseline, usize::MAX),
                |_| start,
            )
            .unwrap();
        let call_peak = first.report().peak_live_bytes();
        assert!(call_peak > call_baseline);
        drop(first);
        let at_boundary = prepared
            .bound_margin_with_auxiliary_bounds_unwired_with_clock(
                &certified,
                &declared,
                None,
                config(3),
                ConstrainedZonotopeCallBudget::new(deadline, call_baseline, call_peak),
                |_| start,
            )
            .unwrap();
        assert_eq!(at_boundary.report().peak_live_bytes(), call_peak);
        drop(at_boundary);
        assert!(matches!(
            prepared.bound_margin_with_auxiliary_bounds_unwired_with_clock(
                &certified,
                &declared,
                None,
                config(3),
                ConstrainedZonotopeCallBudget::new(deadline, call_baseline, call_peak - 1),
                |_| start,
            ),
            Err(ReluTailDualBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { required, limit }
            )) if required == call_peak && limit == call_peak - 1
        ));

        let first_m17 = prepared
            .bound_margin_unwired_with_clock(
                &declared,
                None,
                config(3),
                ConstrainedZonotopeCallBudget::new(deadline, call_baseline, usize::MAX),
                |_| start,
            )
            .unwrap();
        let m17_peak = first_m17.report().peak_live_bytes();
        drop(first_m17);
        let m17_at_boundary = prepared
            .bound_margin_unwired_with_clock(
                &declared,
                None,
                config(3),
                ConstrainedZonotopeCallBudget::new(deadline, call_baseline, m17_peak),
                |_| start,
            )
            .unwrap();
        assert_eq!(m17_at_boundary.report().peak_live_bytes(), m17_peak);
        drop(m17_at_boundary);
        assert!(matches!(
            prepared.bound_margin_unwired_with_clock(
                &declared,
                None,
                config(3),
                ConstrainedZonotopeCallBudget::new(deadline, call_baseline, m17_peak - 1),
                |_| start,
            ),
            Err(ReluTailDualBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { required, limit }
            )) if required == m17_peak && limit == m17_peak - 1
        ));

        assert_eq!(m17_peak, call_peak);
        let retained_m17_bytes =
            independent_relu_tail_result_bytes(domain.value_dim(), domain.constraint_count());
        assert_eq!(
            relu_tail_dual_result_live_bytes(domain.value_dim(), domain.constraint_count())
                .unwrap(),
            retained_m17_bytes
        );
        let portfolio_peak = call_peak.checked_add(retained_m17_bytes).unwrap();
        let portfolio = prepared
            .bound_margin_m17_m20_unwired_with_budget(
                &certified,
                &declared,
                None,
                config(3),
                ConstrainedZonotopeCallBudget::new(deadline, call_baseline, usize::MAX),
            )
            .unwrap();
        assert_eq!(portfolio.value().status, ReluTailBoxCutStatus::Completed);
        assert_eq!(portfolio.report().peak_live_bytes(), portfolio_peak);
        drop(portfolio);

        let at_boundary = prepared
            .bound_margin_m17_m20_unwired_with_budget(
                &certified,
                &declared,
                None,
                config(3),
                ConstrainedZonotopeCallBudget::new(deadline, call_baseline, portfolio_peak),
            )
            .unwrap();
        assert_eq!(at_boundary.value().status, ReluTailBoxCutStatus::Completed);
        assert_eq!(at_boundary.report().peak_live_bytes(), portfolio_peak);
        drop(at_boundary);

        let below_boundary = prepared
            .bound_margin_m17_m20_unwired_with_budget(
                &certified,
                &declared,
                None,
                config(3),
                ConstrainedZonotopeCallBudget::new(deadline, call_baseline, portfolio_peak - 1),
            )
            .expect("optional M20 peak refusal must retain M17");
        assert_eq!(
            below_boundary.value().status,
            ReluTailBoxCutStatus::AuxiliaryFallback
        );
        assert_eq!(
            below_boundary.value().selected,
            ReluTailBoxCutSelection::Original
        );
        assert!(below_boundary.value().auxiliary.is_none());
        assert_eq!(below_boundary.report().peak_live_bytes(), m17_peak);
    }

    #[test]
    fn budgeted_prepared_deadlines_cover_preparation_intersection_line_and_replay() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let certified = auxiliary(vec![-0.75], vec![0.75]);
        let start = Instant::now();
        let deadline = start + Duration::from_secs(1);
        let expired = start + Duration::from_secs(2);
        let preparation_seam = "prepared ReLU-tail exact coordinate hull complete";
        assert!(matches!(
            prepare_relu_tail_triangle_dual_unwired_with_clock(
                &domain,
                ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
                move |checkpoint| {
                    if checkpoint == preparation_seam {
                        expired
                    } else {
                        start
                    }
                },
            ),
            Err(ReluTailDualBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
            )) if checkpoint == preparation_seam
        ));

        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        let baseline = prepared.conservative_live_bytes();
        for seam in [
            "prepared auxiliary ReLU-tail intersection complete",
            "prepared auxiliary ReLU-tail exact line construction complete",
            "prepared auxiliary ReLU-tail mandatory replay complete",
        ] {
            let result = prepared.bound_margin_with_auxiliary_bounds_unwired_with_clock(
                &certified,
                &declared,
                None,
                config(0),
                ConstrainedZonotopeCallBudget::new(deadline, baseline, usize::MAX),
                move |checkpoint| {
                    if checkpoint == seam {
                        expired
                    } else {
                        start
                    }
                },
            );
            assert!(
                matches!(
                    result,
                    Err(ReluTailDualBudgetError::Budget(
                        ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
                    )) if checkpoint == seam
                ),
                "deadline must close at {seam}"
            );
        }
        for seam in [
            "prepared ReLU-tail exact line construction complete",
            "prepared ReLU-tail mandatory replay complete",
        ] {
            let result = prepared.bound_margin_unwired_with_clock(
                &declared,
                None,
                config(0),
                ConstrainedZonotopeCallBudget::new(deadline, baseline, usize::MAX),
                move |checkpoint| {
                    if checkpoint == seam {
                        expired
                    } else {
                        start
                    }
                },
            );
            assert!(
                matches!(
                    result,
                    Err(ReluTailDualBudgetError::Budget(
                        ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
                    )) if checkpoint == seam
                ),
                "M17 deadline must close at {seam}"
            );
        }

        let dimension = crate::CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL / 2;
        let lower = vec![-1.0; dimension];
        let upper = vec![1.0; dimension];
        let fixed = vec![false; dimension];
        let wide_domain =
            ConstrainedZonotope64::from_certified_bounds(&lower, &upper, &fixed).unwrap();
        let wide_margin = margin(vec![BigRational::zero(); dimension], BigRational::zero());
        let wide_auxiliary = auxiliary(vec![-0.5; dimension], vec![0.5; dimension]);
        let wide_prepared = prepare_relu_tail_triangle_dual_unwired(&wide_domain).unwrap();
        let endpoint_seam = "prepared auxiliary ReLU-tail upper-endpoint intersection";
        assert!(matches!(
            wide_prepared.bound_margin_with_auxiliary_bounds_unwired_with_clock(
                &wide_auxiliary,
                &wide_margin,
                None,
                config(0),
                ConstrainedZonotopeCallBudget::new(
                    deadline,
                    wide_prepared.conservative_live_bytes(),
                    usize::MAX,
                ),
                move |checkpoint| {
                    if checkpoint == endpoint_seam {
                        expired
                    } else {
                        start
                    }
                },
            ),
            Err(ReluTailDualBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
            )) if checkpoint == endpoint_seam
        ));
    }

    #[test]
    fn budgeted_prepared_m20_preserves_auxiliary_dimension_and_disjoint_errors() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let wrong_dimension = auxiliary(Vec::new(), Vec::new());
        let disjoint = auxiliary(vec![2.0], vec![3.0]);
        let start = Instant::now();
        let budget = ConstrainedZonotopeCallBudget::new(
            start + Duration::from_secs(1),
            prepared.conservative_live_bytes(),
            usize::MAX,
        );

        for auxiliary in [&wrong_dimension, &disjoint] {
            let legacy = prepared
                .bound_margin_with_auxiliary_bounds_unwired(auxiliary, &declared, None, config(0))
                .unwrap_err();
            let budgeted = prepared
                .bound_margin_with_auxiliary_bounds_unwired_with_clock(
                    auxiliary,
                    &declared,
                    None,
                    config(0),
                    budget,
                    |_| start,
                )
                .unwrap_err();
            assert_eq!(budgeted, ReluTailDualBudgetError::Bound(legacy));
        }
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
    fn budgeted_m17_m20_m24_matches_legacy_with_one_prepared_hull_pass() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-2.0], &[2.0], &[false]).unwrap();
        let certified = auxiliary(vec![1.0], vec![2.0]);
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        EXACT_COORDINATE_HULL_PASSES.with(|passes| passes.set(0));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        let legacy = prepared
            .bound_margin_with_optimized_auxiliary_box_cut_unwired(
                &certified,
                &declared,
                None,
                config(0),
                box_config(1, 1),
            )
            .unwrap();

        let start = Instant::now();
        let budgeted = prepared
            .bound_margin_m17_m20_m24_unwired_with_clock(
                &certified,
                &declared,
                None,
                config(0),
                box_config(1, 1),
                ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 37, usize::MAX),
                |_| start,
            )
            .unwrap();
        assert_optimized_box_cut_bit_identical(&legacy, &budgeted.value().optimized);
        assert_eq!(budgeted.value().optional_budget_error, None);
        assert!(budgeted.report().charged_items() > 0);
        assert!(budgeted.report().deadline_polls() > 0);
        EXACT_COORDINATE_HULL_PASSES.with(|passes| assert_eq!(passes.get(), 1));
    }

    #[test]
    fn budgeted_no_tighter_auxiliary_elides_m20_replay_but_retains_its_certificate() {
        // The predicate multiplier is genuinely useful, so this also proves
        // that equivalent-certificate publication copies nonempty multiplier
        // storage and a nonzero exact constant rather than only scalar bits.
        let domain = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![vec![(0, 1.0)]],
            array![[1.0]],
            vec![0.0],
            vec![0.0],
        )
        .unwrap();
        // The lower endpoint is exact and the upper endpoint is wider.
        let certified = auxiliary(vec![-1.0], vec![2.0]);
        let declared = margin(vec![ratio(-1, 1)], ratio(1, 7));
        let supplied = [0.5];
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        assert_eq!(
            count_tighter_auxiliary_box_endpoints(&prepared.exact_coordinate_bounds, &certified)
                .unwrap(),
            0
        );
        let independently_replayed_m20 = prepared
            .bound_margin_with_auxiliary_bounds_unwired(
                &certified,
                &declared,
                Some(&supplied),
                config(0),
            )
            .unwrap();
        assert!(independently_replayed_m20.supplied_multipliers_used);
        assert!(!independently_replayed_m20.multipliers.is_empty());
        assert!(!independently_replayed_m20.exact_constant.is_zero());

        let m20_replay_admissions = Cell::new(0_usize);
        let start = Instant::now();
        let outcome = prepared
            .bound_margin_m17_m20_m24_unwired_with_clock(
                &certified,
                &declared,
                Some(&supplied),
                config(0),
                box_config(1, 1),
                ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
                |checkpoint| {
                    if checkpoint == "prepared auxiliary ReLU-tail validation" {
                        m20_replay_admissions.set(m20_replay_admissions.get() + 1);
                    }
                    start
                },
            )
            .unwrap();
        let receipt = outcome.value();
        assert_eq!(m20_replay_admissions.get(), 0);
        assert_eq!(receipt.optional_budget_error, None);
        assert_eq!(
            receipt.optimized.search_status,
            ReluTailBoxCutOptimizerStatus::NoTighterAuxiliaryBox
        );
        assert_eq!(receipt.optimized.search_plan, None);
        assert_eq!(receipt.optimized.iterations_completed, 0);
        assert_eq!(receipt.optimized.restarts_completed, 0);
        assert_eq!(receipt.optimized.candidates_scored, 0);
        assert_eq!(receipt.optimized.exact_replays, 0);
        assert_eq!(
            receipt.optimized.portfolio.status,
            ReluTailBoxCutStatus::CandidateFallback
        );
        assert_eq!(
            receipt.optimized.selected,
            ReluTailBoxCutSelection::Original
        );
        let equivalent_m20 = receipt.optimized.portfolio.auxiliary.as_ref().unwrap();
        assert_result_bit_identical(&receipt.optimized.portfolio.original, equivalent_m20);
        assert_result_bit_identical(&independently_replayed_m20, equivalent_m20);
    }

    #[test]
    fn budgeted_one_ulp_strict_endpoint_still_runs_real_m20() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
        let certified = auxiliary(vec![(-1.0_f64).next_up()], vec![1.0]);
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        assert_eq!(
            count_tighter_auxiliary_box_endpoints(&prepared.exact_coordinate_bounds, &certified)
                .unwrap(),
            1
        );

        let m20_replay_admissions = Cell::new(0_usize);
        let start = Instant::now();
        let outcome = prepared
            .bound_margin_m17_m20_m24_unwired_with_clock(
                &certified,
                &declared,
                None,
                config(0),
                box_config(0, 0),
                ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
                |checkpoint| {
                    if checkpoint == "prepared auxiliary ReLU-tail validation" {
                        m20_replay_admissions.set(m20_replay_admissions.get() + 1);
                    }
                    start
                },
            )
            .unwrap();
        let receipt = outcome.value();
        assert_eq!(m20_replay_admissions.get(), 1);
        assert_eq!(receipt.optional_budget_error, None);
        assert_eq!(
            receipt.optimized.search_status,
            ReluTailBoxCutOptimizerStatus::SearchDisabled
        );
        let plan = receipt.optimized.search_plan.unwrap();
        assert_eq!(plan.box_variables, 1);
        assert!(receipt.optimized.portfolio.auxiliary.is_some());
        assert_eq!(
            receipt.optimized.portfolio.status,
            ReluTailBoxCutStatus::CandidateFallback
        );
    }

    #[test]
    fn budgeted_endpoint_precheck_rejects_wrong_auxiliary_shape_fail_closed() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
        let wrong_shape = auxiliary(Vec::new(), Vec::new());
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        let start = Instant::now();
        let outcome = prepared
            .bound_margin_m17_m20_m24_unwired_with_clock(
                &wrong_shape,
                &declared,
                None,
                config(0),
                box_config(1, 1),
                ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
                |_| start,
            )
            .unwrap();
        let receipt = outcome.value();
        assert_eq!(receipt.optional_budget_error, None);
        assert_eq!(
            receipt.optimized.search_status,
            ReluTailBoxCutOptimizerStatus::AuxiliaryFallback
        );
        assert_eq!(
            receipt.optimized.portfolio.status,
            ReluTailBoxCutStatus::AuxiliaryFallback
        );
        assert!(receipt.optimized.portfolio.auxiliary.is_none());
        assert!(receipt.optimized.portfolio.box_cut.is_none());
        assert_eq!(receipt.optimized.search_plan, None);
        assert_eq!(
            receipt.optimized.selected,
            ReluTailBoxCutSelection::Original
        );
    }

    #[test]
    fn no_tighter_endpoint_and_copy_deadlines_retain_original_only() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[false]).unwrap();
        let certified = auxiliary(vec![-1.0], vec![1.0]);
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        let start = Instant::now();
        let deadline = start + Duration::from_secs(1);
        let expired = start + Duration::from_secs(2);
        for target in [
            "M24 tighter auxiliary endpoint count complete",
            "equivalent M20 result copy publication",
        ] {
            let outcome = prepared
                .bound_margin_m17_m20_m24_unwired_with_clock(
                    &certified,
                    &declared,
                    None,
                    config(0),
                    box_config(1, 1),
                    ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
                    |checkpoint| if checkpoint == target { expired } else { start },
                )
                .unwrap();
            let receipt = outcome.value();
            assert_eq!(
                receipt.optimized.search_status,
                ReluTailBoxCutOptimizerStatus::AuxiliaryFallback
            );
            assert_eq!(
                receipt.optimized.portfolio.status,
                ReluTailBoxCutStatus::AuxiliaryFallback
            );
            assert!(receipt.optimized.portfolio.auxiliary.is_none());
            assert!(receipt.optimized.portfolio.box_cut.is_none());
            assert_eq!(
                receipt.optional_budget_error,
                Some(ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint: target })
            );
            assert!(receipt.optimized.portfolio.original.lower_bound.is_finite());
        }
    }

    #[test]
    fn budgeted_m20_exact_peak_and_cap_minus_one_are_deterministic() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-2.0], &[2.0], &[false]).unwrap();
        let certified = auxiliary(vec![1.0], vec![2.0]);
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        let search_config = box_config(0, 0);
        let box_variables =
            count_tighter_auxiliary_box_endpoints(&prepared.exact_coordinate_bounds, &certified)
                .unwrap();
        let plan = ReluTailBoxCutOptimizerPlan::checked(
            &domain,
            prepared.generator_nonzeros,
            box_variables,
            search_config,
        )
        .unwrap();
        let peaks = independent_m24_test_peaks(&prepared, plan);
        assert!(peaks.m20 >= peaks.endpoint_count);

        let baseline = 41_usize;
        let start = Instant::now();
        let exact = prepared
            .bound_margin_m17_m20_m24_unwired_with_clock(
                &certified,
                &declared,
                None,
                config(0),
                search_config,
                ConstrainedZonotopeCallBudget::new(
                    start + Duration::from_secs(1),
                    baseline,
                    baseline + peaks.m20,
                ),
                |_| start,
            )
            .unwrap();
        assert_eq!(exact.value().optional_budget_error, None);
        assert!(exact.value().optimized.portfolio.auxiliary.is_some());
        assert_eq!(
            exact.value().optimized.search_status,
            ReluTailBoxCutOptimizerStatus::SearchDisabled
        );
        assert_eq!(exact.report().peak_live_bytes(), baseline + peaks.m20);

        let below = prepared
            .bound_margin_m17_m20_m24_unwired_with_clock(
                &certified,
                &declared,
                None,
                config(0),
                search_config,
                ConstrainedZonotopeCallBudget::new(
                    start + Duration::from_secs(1),
                    baseline,
                    baseline + peaks.m20 - 1,
                ),
                |_| start,
            )
            .unwrap();
        assert_eq!(
            below.value().optimized.search_status,
            ReluTailBoxCutOptimizerStatus::AuxiliaryFallback
        );
        assert_eq!(
            below.value().optimized.portfolio.status,
            ReluTailBoxCutStatus::AuxiliaryFallback
        );
        assert!(below.value().optimized.portfolio.auxiliary.is_none());
        assert!(below.value().optimized.portfolio.box_cut.is_none());
        assert_eq!(
            below.value().optimized.selected,
            ReluTailBoxCutSelection::Original
        );
        assert_result_bit_identical(
            &exact.value().optimized.portfolio.original,
            &below.value().optimized.portfolio.original,
        );
        assert_eq!(
            below.value().optional_budget_error,
            Some(ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                required: baseline + peaks.m20,
                limit: baseline + peaks.m20 - 1,
            })
        );
        assert_eq!(below.report().peak_live_bytes(), baseline + peaks.m17);
    }

    #[test]
    fn budgeted_first_m24_replay_exact_peak_and_cap_minus_one_are_deterministic() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-2.0], &[2.0], &[false]).unwrap();
        let certified = auxiliary(vec![1.0], vec![2.0]);
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        let search_config = box_config(1, 0);
        let box_variables =
            count_tighter_auxiliary_box_endpoints(&prepared.exact_coordinate_bounds, &certified)
                .unwrap();
        let plan = ReluTailBoxCutOptimizerPlan::checked(
            &domain,
            prepared.generator_nonzeros,
            box_variables,
            search_config,
        )
        .unwrap();
        let peaks = independent_m24_test_peaks(&prepared, plan);
        assert!(peaks.first_replay >= peaks.m20);
        assert!(peaks.first_replay >= peaks.endpoint_count);
        assert!(peaks.first_replay >= peaks.search);

        let baseline = 53_usize;
        let start = Instant::now();
        let exact = prepared
            .bound_margin_m17_m20_m24_unwired_with_clock(
                &certified,
                &declared,
                None,
                config(0),
                search_config,
                ConstrainedZonotopeCallBudget::new(
                    start + Duration::from_secs(1),
                    baseline,
                    baseline + peaks.first_replay,
                ),
                |_| start,
            )
            .unwrap();
        assert_eq!(exact.value().optional_budget_error, None);
        assert_eq!(exact.value().optimized.exact_replays, 1);
        assert!(exact.value().optimized.portfolio.box_cut.is_some());
        assert_eq!(
            exact.report().peak_live_bytes(),
            baseline + peaks.first_replay
        );

        let below = prepared
            .bound_margin_m17_m20_m24_unwired_with_clock(
                &certified,
                &declared,
                None,
                config(0),
                search_config,
                ConstrainedZonotopeCallBudget::new(
                    start + Duration::from_secs(1),
                    baseline,
                    baseline + peaks.first_replay - 1,
                ),
                |_| start,
            )
            .unwrap();
        assert_eq!(
            below.value().optimized.search_status,
            ReluTailBoxCutOptimizerStatus::ResourceFallback
        );
        assert_eq!(below.value().optimized.exact_replays, 0);
        assert!(below.value().optimized.portfolio.auxiliary.is_some());
        assert!(below.value().optimized.portfolio.box_cut.is_none());
        assert_result_bit_identical(
            &exact.value().optimized.portfolio.original,
            &below.value().optimized.portfolio.original,
        );
        assert_result_bit_identical(
            exact
                .value()
                .optimized
                .portfolio
                .auxiliary
                .as_ref()
                .unwrap(),
            below
                .value()
                .optimized
                .portfolio
                .auxiliary
                .as_ref()
                .unwrap(),
        );
        assert_eq!(
            below.value().optional_budget_error,
            Some(ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                required: baseline + peaks.first_replay,
                limit: baseline + peaks.first_replay - 1,
            })
        );
    }

    #[test]
    fn budgeted_second_m24_replay_accounts_for_and_retains_first_certificate() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-2.0], &[2.0], &[false]).unwrap();
        let certified = auxiliary(vec![1.0], vec![2.0]);
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        let search_config = box_config(1, 1);
        let box_variables =
            count_tighter_auxiliary_box_endpoints(&prepared.exact_coordinate_bounds, &certified)
                .unwrap();
        let plan = ReluTailBoxCutOptimizerPlan::checked(
            &domain,
            prepared.generator_nonzeros,
            box_variables,
            search_config,
        )
        .unwrap();
        let peaks = independent_m24_test_peaks(&prepared, plan);
        assert_eq!(
            peaks.second_replay - peaks.first_replay,
            peaks.retained_certificate
        );

        let baseline = 67_usize;
        let start = Instant::now();
        let full = prepared
            .bound_margin_m17_m20_m24_unwired_with_clock(
                &certified,
                &declared,
                None,
                config(0),
                search_config,
                ConstrainedZonotopeCallBudget::new(
                    start + Duration::from_secs(1),
                    baseline,
                    baseline + peaks.second_replay,
                ),
                |_| start,
            )
            .unwrap();
        assert_eq!(full.value().optional_budget_error, None);
        assert_eq!(full.value().optimized.exact_replays, 2);
        assert_eq!(
            full.value().optimized.search_status,
            ReluTailBoxCutOptimizerStatus::Completed
        );
        assert!(full.value().optimized.portfolio.box_cut.is_some());
        assert_eq!(
            full.report().peak_live_bytes(),
            baseline + peaks.second_replay
        );

        let at_first = prepared
            .bound_margin_m17_m20_m24_unwired_with_clock(
                &certified,
                &declared,
                None,
                config(0),
                search_config,
                ConstrainedZonotopeCallBudget::new(
                    start + Duration::from_secs(1),
                    baseline,
                    baseline + peaks.first_replay,
                ),
                |_| start,
            )
            .unwrap();
        assert_eq!(at_first.value().optimized.exact_replays, 1);
        assert_eq!(
            at_first.value().optimized.search_status,
            ReluTailBoxCutOptimizerStatus::ResourceFallback
        );
        assert!(at_first.value().optimized.portfolio.box_cut.is_some());
        assert_eq!(
            at_first.value().optional_budget_error,
            Some(ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                required: baseline + peaks.second_replay,
                limit: baseline + peaks.first_replay,
            })
        );

        let below_first = prepared
            .bound_margin_m17_m20_m24_unwired_with_clock(
                &certified,
                &declared,
                None,
                config(0),
                search_config,
                ConstrainedZonotopeCallBudget::new(
                    start + Duration::from_secs(1),
                    baseline,
                    baseline + peaks.first_replay - 1,
                ),
                |_| start,
            )
            .unwrap();
        assert_eq!(below_first.value().optimized.exact_replays, 0);
        assert!(below_first.value().optimized.portfolio.box_cut.is_none());
        assert_eq!(
            below_first.value().optional_budget_error,
            Some(ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                required: baseline + peaks.first_replay,
                limit: baseline + peaks.first_replay - 1,
            })
        );

        let below_second = prepared
            .bound_margin_m17_m20_m24_unwired_with_clock(
                &certified,
                &declared,
                None,
                config(0),
                search_config,
                ConstrainedZonotopeCallBudget::new(
                    start + Duration::from_secs(1),
                    baseline,
                    baseline + peaks.second_replay - 1,
                ),
                |_| start,
            )
            .unwrap();
        assert_eq!(
            below_second.value().optimized.search_status,
            ReluTailBoxCutOptimizerStatus::ResourceFallback
        );
        assert_eq!(below_second.value().optimized.exact_replays, 1);
        assert_box_cut_bit_identical(
            at_first
                .value()
                .optimized
                .portfolio
                .box_cut
                .as_ref()
                .unwrap(),
            below_second
                .value()
                .optimized
                .portfolio
                .box_cut
                .as_ref()
                .unwrap(),
        );
        assert_eq!(
            below_second.value().optional_budget_error,
            Some(ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                required: baseline + peaks.second_replay,
                limit: baseline + peaks.second_replay - 1,
            })
        );
        assert_eq!(
            below_second.report().peak_live_bytes(),
            baseline + peaks.first_replay
        );
    }

    #[test]
    fn budgeted_m17_m20_m24_deadlines_close_optional_phase_seams() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-2.0], &[2.0], &[false]).unwrap();
        let certified = auxiliary(vec![1.0], vec![2.0]);
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        let start = Instant::now();
        let deadline = start + Duration::from_secs(1);
        let expired = start + Duration::from_secs(2);

        for (target, expected_status, expect_auxiliary) in [
            (
                "prepared auxiliary ReLU-tail validation",
                ReluTailBoxCutOptimizerStatus::AuxiliaryFallback,
                false,
            ),
            (
                "M24 candidate search startup",
                ReluTailBoxCutOptimizerStatus::Deadline,
                true,
            ),
            (
                "M24 exact replay admission",
                ReluTailBoxCutOptimizerStatus::Deadline,
                true,
            ),
        ] {
            let outcome = prepared
                .bound_margin_m17_m20_m24_unwired_with_clock(
                    &certified,
                    &declared,
                    None,
                    config(0),
                    box_config(1, 0),
                    ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
                    |checkpoint| if checkpoint == target { expired } else { start },
                )
                .unwrap();
            let receipt = outcome.value();
            assert_eq!(receipt.optimized.search_status, expected_status);
            assert_eq!(
                receipt.optimized.portfolio.auxiliary.is_some(),
                expect_auxiliary
            );
            assert!(receipt.optimized.portfolio.box_cut.is_none());
            assert_eq!(receipt.optimized.exact_replays, 0);
            assert!(
                receipt.optimized.lower_bound >= receipt.optimized.portfolio.original.lower_bound
            );
            assert_eq!(
                receipt.optional_budget_error,
                Some(ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint: target })
            );
        }
    }

    #[test]
    fn gated_box_search_keeps_candidate_deadline_distinct_from_outer_budget() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-2.0], &[2.0], &[false]).unwrap();
        let certified = auxiliary(vec![1.0], vec![2.0]);
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        let auxiliary_result = prepared
            .bound_margin_with_auxiliary_bounds_unwired(&certified, &declared, None, config(0))
            .unwrap();
        let search_config = box_config(1, 0);
        let box_variables =
            count_tighter_auxiliary_box_endpoints(&prepared.exact_coordinate_bounds, &certified)
                .unwrap();
        let plan = ReluTailBoxCutOptimizerPlan::checked(
            &domain,
            prepared.generator_nonzeros,
            box_variables,
            search_config,
        )
        .unwrap();

        let start = Instant::now();
        let mut outer_gate = ConstrainedZonotopeCallTracker::with_clock(
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
            |_| start,
        )
        .unwrap();
        let search = optimize_auxiliary_box_multipliers_with_clock_and_call_gate(
            &domain,
            &prepared.exact_coordinate_bounds,
            &certified,
            &auxiliary_result.direction,
            search_config,
            plan,
            || search_config.wall_time,
            &mut outer_gate,
        );
        assert_eq!(
            search.search.status,
            ReluTailBoxCutOptimizerStatus::Deadline
        );
        assert_eq!(search.budget_error, None);
        assert!(search.search.variables.is_empty());
        assert!(search.search.candidates.is_empty());
        assert!(outer_gate.report().deadline_polls() >= 2);
    }

    #[test]
    fn budgeted_internal_box_search_deadline_retains_m17_m20() {
        let dimension = crate::CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL;
        let lower = vec![-2.0; dimension];
        let upper = vec![2.0; dimension];
        let remainder_only = vec![true; dimension];
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&lower, &upper, &remainder_only).unwrap();
        let certified = auxiliary(vec![1.0; dimension], vec![2.0; dimension]);
        let declared = margin(vec![ratio(1, 1); dimension], ratio(0, 1));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        let mut search_config = box_config(1, 0);
        search_config.wall_time = Duration::from_secs(5);
        search_config.limits.max_value_dim = dimension;
        search_config.limits.max_box_variables = dimension;

        let start = Instant::now();
        let deadline = start + Duration::from_secs(1);
        let expired = start + Duration::from_secs(2);
        let interior_polls = Cell::new(0_usize);
        let outcome = prepared
            .bound_margin_m17_m20_m24_unwired_with_clock(
                &certified,
                &declared,
                None,
                config(0),
                search_config,
                ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
                |checkpoint| {
                    if checkpoint == "M24 candidate direction and witness initialization" {
                        interior_polls.set(interior_polls.get() + 1);
                        expired
                    } else {
                        start
                    }
                },
            )
            .unwrap();
        assert_eq!(interior_polls.get(), 1);
        assert_eq!(
            outcome.value().optimized.search_status,
            ReluTailBoxCutOptimizerStatus::Deadline
        );
        assert!(outcome.value().optimized.portfolio.auxiliary.is_some());
        assert!(outcome.value().optimized.portfolio.box_cut.is_none());
        assert_eq!(outcome.value().optimized.exact_replays, 0);
        assert_eq!(
            outcome.value().optional_budget_error,
            Some(ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                checkpoint: "M24 candidate direction and witness initialization",
            })
        );
    }

    #[test]
    fn budgeted_exact_box_cut_publication_deadline_rolls_back_partial_certificate() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-2.0], &[2.0], &[false]).unwrap();
        let certified = auxiliary(vec![1.0], vec![2.0]);
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        let start = Instant::now();
        let deadline = start + Duration::from_secs(1);
        let expired = start + Duration::from_secs(2);
        let publication_polls = Cell::new(0_usize);
        let outcome = prepared
            .bound_margin_m17_m20_m24_unwired_with_clock(
                &certified,
                &declared,
                None,
                config(0),
                box_config(1, 0),
                ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
                |checkpoint| {
                    if checkpoint == "M24 exact Box-cut publication" {
                        publication_polls.set(publication_polls.get() + 1);
                        expired
                    } else {
                        start
                    }
                },
            )
            .unwrap();
        assert_eq!(publication_polls.get(), 1);
        assert_eq!(
            outcome.value().optimized.search_status,
            ReluTailBoxCutOptimizerStatus::Deadline
        );
        assert!(outcome.value().optimized.portfolio.auxiliary.is_some());
        assert!(outcome.value().optimized.portfolio.box_cut.is_none());
        assert_eq!(
            outcome.value().optimized.portfolio.status,
            ReluTailBoxCutStatus::CandidateFallback
        );
        assert_eq!(outcome.value().optimized.exact_replays, 1);
        assert_eq!(
            outcome.value().optional_budget_error,
            Some(ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                checkpoint: "M24 exact Box-cut publication",
            })
        );
    }

    #[test]
    fn budgeted_second_replay_deadline_retains_first_certificate_bitwise() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-2.0], &[2.0], &[false]).unwrap();
        let certified = auxiliary(vec![1.0], vec![2.0]);
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        let start = Instant::now();
        let deadline = start + Duration::from_secs(1);
        let expired = start + Duration::from_secs(2);

        let first_only = prepared
            .bound_margin_m17_m20_m24_unwired_with_clock(
                &certified,
                &declared,
                None,
                config(0),
                box_config(1, 0),
                ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
                |_| start,
            )
            .unwrap();
        let first_certificate = first_only
            .value()
            .optimized
            .portfolio
            .box_cut
            .as_ref()
            .unwrap();

        let replay_admissions = Cell::new(0_usize);
        let interrupted = prepared
            .bound_margin_m17_m20_m24_unwired_with_clock(
                &certified,
                &declared,
                None,
                config(0),
                box_config(1, 1),
                ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
                |checkpoint| {
                    if checkpoint == "M24 exact replay admission" {
                        let admission = replay_admissions.get() + 1;
                        replay_admissions.set(admission);
                        if admission == 2 {
                            return expired;
                        }
                    }
                    start
                },
            )
            .unwrap();
        assert_eq!(replay_admissions.get(), 2);
        assert_eq!(
            interrupted.value().optimized.search_status,
            ReluTailBoxCutOptimizerStatus::Deadline
        );
        assert_eq!(interrupted.value().optimized.exact_replays, 1);
        assert_box_cut_bit_identical(
            first_certificate,
            interrupted
                .value()
                .optimized
                .portfolio
                .box_cut
                .as_ref()
                .unwrap(),
        );
        assert_eq!(
            interrupted.value().optional_budget_error,
            Some(ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                checkpoint: "M24 exact replay admission",
            })
        );
    }

    #[test]
    fn budgeted_bound_failure_on_first_m24_candidate_continues_to_second() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-2.0], &[2.0], &[false]).unwrap();
        let certified = auxiliary(vec![1.0], vec![2.0]);
        let declared = margin(vec![ratio(1, 1)], ratio(0, 1));
        let prepared = prepare_relu_tail_triangle_dual_unwired(&domain).unwrap();
        let start = Instant::now();
        let budget =
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX);

        let second_only = prepared
            .bound_margin_m17_m20_m24_unwired_with_clock(
                &certified,
                &declared,
                None,
                config(0),
                box_config(0, 1),
                budget,
                |_| start,
            )
            .unwrap();
        let second_certificate = second_only
            .value()
            .optimized
            .portfolio
            .box_cut
            .as_ref()
            .unwrap();

        BOX_OPTIMIZER_FAIL_NEXT_EXACT_REPLAY.with(|fail| fail.set(true));
        let continued = prepared
            .bound_margin_m17_m20_m24_unwired_with_clock(
                &certified,
                &declared,
                None,
                config(0),
                box_config(1, 1),
                budget,
                |_| start,
            )
            .unwrap();
        assert_eq!(continued.value().optional_budget_error, None);
        assert_eq!(continued.value().optimized.exact_replays, 2);
        assert_eq!(
            continued.value().optimized.search_status,
            ReluTailBoxCutOptimizerStatus::ExactReplayFallback
        );
        assert_box_cut_bit_identical(
            second_certificate,
            continued
                .value()
                .optimized
                .portfolio
                .box_cut
                .as_ref()
                .unwrap(),
        );
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

        let clock_calls = Cell::new(0_usize);
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
