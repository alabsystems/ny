// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Property-bound attachment for the certified imgSz32 cGAN leaf-row bounder.
//!
//! The bounder itself is deliberately verdict-neutral.  This module is the
//! narrow authority layer: an exact `cgan_2023` route, authored ONNX/profile
//! seal, exact-decimal scalar-moat parse, complete two-singleton planner, graph
//! scope, five-dimensional leaf, one caller deadline, and one logical
//! peak-live ceiling must all agree before both strict safe inequalities can
//! discharge a leaf.  Every mismatch is `Undecided`; this lane never returns a
//! SAT verdict.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ny_mip::{
    ConstrainedZonotopeCallBudget, ConstrainedZonotopeCallBudgetError, ReluTailBoxCutOptimizerPlan,
    ReluTailBoxCutOptimizerStatus, ReluTailBoxCutSelection, ReluTailBoxCutStatus,
};
use ny_onnx::vnnlib::{
    load_vnnlib_with_certified_scalar_moat, CertifiedInputBox, CertifiedScalarMoat,
    OutputConstraint, VnnLibSpec,
};
use ny_onnx::{load_onnx_with_config, BatchNormFoldingPolicy, OnnxLoadConfig, OnnxModel};
use ny_propagate::beta_crown::graph_mip_leaf::{
    GraphInputLeafRequest, GraphMipLeafOracle, GraphMipLeafRequest, GraphMipLeafVerdict,
};
use ny_propagate::beta_crown::CutFoldScope;
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;
use tracing::{debug, info, warn};

use super::super::cz_cgan_sequential_unwired::{
    bound_cgan_imgsz32_leaf_rows_unwired, cgan_nch1_independent_interval_qualification_limits,
    cgan_nch3_independent_interval_qualification_limits, select_m17_m20_lower_bound,
    CganCzDepthTwoCompletedMeasurement, CganCzDepthTwoMeasurement, CganCzDepthTwoTransformFailure,
    CganCzImgSz32Profile, CganCzLeafRowBounds, CganCzLeafRowReport, CganCzLeafRowStatus,
    CganCzM17CandidateTelemetry, CganCzM20Status, CganCzM24Measurement, CganCzSequentialLimits,
    CganCzVerdictAuthority, CGAN_CZ_VERDICT_AUTHORITY,
};
use super::{CganInputLeafRoute, CGAN_DEPTH_TWO_PRODUCTION_MODE};

const LATENT_DIM: usize = 5;
const AUTHENTICATED_TOPOLOGY_WORK_ITEMS: usize = 837;
const NCH1_PARAMETER_ELEMENTS: usize = 529_538;
const NCH3_PARAMETER_ELEMENTS: usize = 530_404;
// Exact byte lengths of the two authored VNN-COMP cGAN model sources. These
// are deliberately equality seals, not loose allocation caps: a same-name
// replacement must decline before the second ONNX parse can allocate.
const NCH1_MODEL_SOURCE_BYTES: u64 = 2_187_864;
const NCH3_MODEL_SOURCE_BYTES: u64 = 2_191_328;
// The published cGAN properties observed across the 2025/2026 suites are
// 606--681 bytes. Leave deterministic serialization slack while keeping this
// setup lane far below the certified parser's independent expanded-size cap.
const MAX_PROPERTY_SOURCE_BYTES: u64 = 1 << 10;

/// Conservative accounting for the retained raw model, normalized graph,
/// property/root metadata, allocator slack, and borrowed leaf endpoints.
const RETAINED_LIVE_BYTES: usize = 64 << 20;
/// Default-dark hard ceiling for retained state, the sequential independent
/// and correlated prefixes, prepared tail geometry, and exact M17/M20 calls.
/// The route fails closed at this inherited ceiling; real non-point canaries
/// are still required before any default preset may promote it.
const MAX_PEAK_LIVE_BYTES: usize = 768 << 20;

/// Independently sealed fixed M24 measurement policy. These values repeat the
/// producer contract intentionally: the property attachment does not trust a
/// self-described search plan when deciding whether telemetry is admissible.
const M24_VALUE_DIM: usize = 512;
const M24_MAX_ALPHA_DIM: usize = 512;
const M24_MAX_BOX_VARIABLES: usize = 1_024;
const M24_TOTAL_ITERATIONS: usize = 8;
const M24_RESTARTS: usize = 2;
const M24_EXACT_REPLAYS: usize = 2;
const M24_MAX_GENERATOR_NONZEROS: usize = 150_000;
const M24_MAX_SEARCH_WORK: u64 = 3_104_960;

/// Independently sealed Relu_20 -> BN_21 -> Conv_22 depth-two geometry.
///
/// This is a dormant receipt-validation contract, not an enabled production
/// treatment. The production bounder supplies no depth-two context and emits
/// `NotRequested` for both sides. These values remain so malformed synthetic or
/// future canary receipts are excluded without gaining authority or causing a
/// historical receipt to decline.
const DEPTH_TWO_INPUT_SHAPE: [usize; 3] = [64, 4, 4];
const DEPTH_TWO_OUTPUT_SHAPE: [usize; 3] = [128, 2, 2];
const DEPTH_TWO_WEIGHT_SHAPE: [usize; 4] = [128, 64, 3, 3];
const DEPTH_TWO_WEIGHT_ELEMENTS: usize = 73_728;
const DEPTH_TWO_KERNEL_VISITS: usize = 294_912;
const DEPTH_TWO_EXACT_PRODUCT_BOUND: usize = 298_688;

/// Scored-canary admission policy.  The exact leaf replay is materially more
/// expensive than one ordinary input-split rebound, so attempts 1--16 retain
/// the original root/deep/compact frontier and attempts 17--32 are reserved
/// for substantially deeper or smaller, near-closed domains. These values
/// affect scheduling only: every admitted proof still replays both rows from
/// the authenticated input box and must satisfy the independent receipt
/// checks.
const LEAF_MIN_DEPTH: usize = 5;
const LEAF_MAX_NORMALIZED_VOLUME: f64 = 1.0 / 32.0;
const LEAF_MAX_WORST_SHORTFALL: f64 = 0.10;
const LEAF_PRIMARY_MAX_ATTEMPTS: u64 = 16;
const LEAF_RESERVED_MIN_DEPTH: usize = 12;
const LEAF_RESERVED_MAX_NORMALIZED_VOLUME: f64 = 1.0 / 4096.0;
const LEAF_RESERVED_MAX_WORST_SHORTFALL: f64 = 0.001;
const LEAF_MAX_ATTEMPTS: u64 = 32;
const LEAF_CALL_BUDGET: Duration = Duration::from_secs(15);
const LEAF_TOTAL_WALL_BUDGET: Duration = Duration::from_secs(90);
const LEAF_MIN_GLOBAL_REMAINING: Duration = Duration::from_mins(2);

#[derive(Clone, Copy, Debug, PartialEq)]
struct LeafAdmissionPolicy {
    min_depth: usize,
    max_normalized_volume: f64,
    max_worst_shortfall: f64,
    primary_max_attempts: u64,
    reserved_min_depth: usize,
    reserved_max_normalized_volume: f64,
    reserved_max_worst_shortfall: f64,
    max_attempts: u64,
    call_budget: Duration,
    total_wall_budget: Duration,
    min_global_remaining: Duration,
}

impl LeafAdmissionPolicy {
    const PRODUCTION: Self = Self {
        min_depth: LEAF_MIN_DEPTH,
        max_normalized_volume: LEAF_MAX_NORMALIZED_VOLUME,
        max_worst_shortfall: LEAF_MAX_WORST_SHORTFALL,
        primary_max_attempts: LEAF_PRIMARY_MAX_ATTEMPTS,
        reserved_min_depth: LEAF_RESERVED_MIN_DEPTH,
        reserved_max_normalized_volume: LEAF_RESERVED_MAX_NORMALIZED_VOLUME,
        reserved_max_worst_shortfall: LEAF_RESERVED_MAX_WORST_SHORTFALL,
        max_attempts: LEAF_MAX_ATTEMPTS,
        call_budget: LEAF_CALL_BUDGET,
        total_wall_budget: LEAF_TOTAL_WALL_BUDGET,
        min_global_remaining: LEAF_MIN_GLOBAL_REMAINING,
    };

    #[cfg(test)]
    const UNRESTRICTED_TEST: Self = Self {
        min_depth: 0,
        max_normalized_volume: 1.0,
        max_worst_shortfall: f64::MAX,
        primary_max_attempts: u64::MAX,
        reserved_min_depth: 0,
        reserved_max_normalized_volume: 1.0,
        reserved_max_worst_shortfall: f64::MAX,
        max_attempts: u64::MAX,
        call_budget: Duration::from_mins(1),
        total_wall_budget: Duration::from_mins(2),
        min_global_remaining: Duration::ZERO,
    };

    fn valid(self) -> bool {
        self.max_normalized_volume.is_finite()
            && (0.0..=1.0).contains(&self.max_normalized_volume)
            && self.max_worst_shortfall.is_finite()
            && self.max_worst_shortfall >= 0.0
            && self.primary_max_attempts > 0
            && self.primary_max_attempts <= self.max_attempts
            && self.reserved_min_depth >= self.min_depth
            && self.reserved_max_normalized_volume.is_finite()
            && (0.0..=self.max_normalized_volume).contains(&self.reserved_max_normalized_volume)
            && self.reserved_max_worst_shortfall.is_finite()
            && (0.0..=self.max_worst_shortfall).contains(&self.reserved_max_worst_shortfall)
            && !self.call_budget.is_zero()
            && self.call_budget <= self.total_wall_budget
            && (self.min_global_remaining.is_zero()
                || self.call_budget <= self.min_global_remaining)
    }

    fn latest_start_wall_nanos(self) -> u64 {
        duration_nanos_saturating(self.total_wall_budget.saturating_sub(self.call_budget))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LeafAttemptTranche {
    Primary,
    Reserved,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LeafFrontierRejection {
    DepthOrVolume,
    ObjectiveShortfall,
}

#[derive(Clone, Copy, Debug)]
enum LeafAdmissionSkip {
    DepthOrVolume,
    ObjectiveShortfall,
    GlobalRemaining,
    TotalWall,
    AttemptCap,
    Concurrent,
}

/// Per-property observation state.  Every hot-path update is a relaxed atomic;
/// telemetry never feeds a bound, selector, or verdict.  The final snapshot is
/// emitted through a separately retained attachment after synchronous BaB
/// returns, while first/power-of-two progress lines survive a watchdog kill.
struct CganInputLeafTelemetry {
    consultations: AtomicU64,
    attempts: AtomicU64,
    /// Subset of `attempts` attributable to ordinals 17--32.
    reserved_attempts: AtomicU64,
    completions: AtomicU64,
    /// Subset of `completions` attributable to reserved attempts.
    reserved_completions: AtomicU64,
    verified_leaves: AtomicU64,
    /// Subset of `verified_leaves` attributable to reserved attempts.
    reserved_verified_leaves: AtomicU64,
    late_results: AtomicU64,
    depth_or_volume_skips: AtomicU64,
    objective_shortfall_skips: AtomicU64,
    /// Reserved-frontier subsets of the two aggregate scheduling skips.
    reserved_depth_or_volume_skips: AtomicU64,
    reserved_objective_shortfall_skips: AtomicU64,
    global_remaining_skips: AtomicU64,
    total_wall_skips: AtomicU64,
    attempt_cap_skips: AtomicU64,
    concurrent_skips: AtomicU64,
    lower_m20_fallbacks: AtomicU64,
    negated_upper_m20_fallbacks: AtomicU64,
    lower_m20_strict_wins: AtomicU64,
    negated_upper_m20_strict_wins: AtomicU64,
    lower_m24_exact_present: AtomicU64,
    negated_upper_m24_exact_present: AtomicU64,
    lower_m24_missing: AtomicU64,
    negated_upper_m24_missing: AtomicU64,
    lower_m24_strict_wins: AtomicU64,
    negated_upper_m24_strict_wins: AtomicU64,
    max_lower_signed_gain_bits: AtomicU64,
    max_negated_upper_signed_gain_bits: AtomicU64,
    max_lower_m24_signed_gain_bits: AtomicU64,
    max_negated_upper_m24_signed_gain_bits: AtomicU64,
    max_lower_m17_optimized_improvement_bits: AtomicU64,
    max_negated_upper_m17_optimized_improvement_bits: AtomicU64,
    best_lower_threshold_residual_bits: AtomicU64,
    best_negated_upper_threshold_residual_bits: AtomicU64,
    best_joint_threshold_residual_bits: AtomicU64,
    best_counterfactual_m24_joint_threshold_residual_bits: AtomicU64,
    m24_only_would_verify: AtomicU64,
    /// Depth-two status counters count row sides, not leaf attempts.
    depth_two_completed_sides: AtomicU64,
    depth_two_invalid_sides: AtomicU64,
    depth_two_not_requested_sides: AtomicU64,
    depth_two_no_time_sides: AtomicU64,
    depth_two_budget_fallback_sides: AtomicU64,
    depth_two_transform_fallback_sides: AtomicU64,
    depth_two_incoherent_pairs: AtomicU64,
    lower_depth_two_strict_wins: AtomicU64,
    negated_upper_depth_two_strict_wins: AtomicU64,
    max_lower_depth_two_signed_gain_bits: AtomicU64,
    max_negated_upper_depth_two_signed_gain_bits: AtomicU64,
    best_counterfactual_depth_two_joint_threshold_residual_bits: AtomicU64,
    depth_two_only_would_verify: AtomicU64,
    total_wall_nanos: AtomicU64,
    max_attempt_wall_nanos: AtomicU64,
    /// Reserved-attempt subsets of aggregate wall time and its maximum.
    reserved_total_wall_nanos: AtomicU64,
    reserved_max_attempt_wall_nanos: AtomicU64,
    max_peak_live_bytes: AtomicU64,
    in_flight: AtomicBool,
    final_emitted: AtomicBool,
}

#[derive(Clone, Copy, Debug)]
struct CganInputLeafTelemetrySnapshot {
    consultations: u64,
    attempts: u64,
    reserved_attempts: u64,
    completions: u64,
    reserved_completions: u64,
    verified_leaves: u64,
    reserved_verified_leaves: u64,
    late_results: u64,
    depth_or_volume_skips: u64,
    objective_shortfall_skips: u64,
    reserved_depth_or_volume_skips: u64,
    reserved_objective_shortfall_skips: u64,
    global_remaining_skips: u64,
    total_wall_skips: u64,
    attempt_cap_skips: u64,
    concurrent_skips: u64,
    lower_m20_fallbacks: u64,
    negated_upper_m20_fallbacks: u64,
    lower_m20_strict_wins: u64,
    negated_upper_m20_strict_wins: u64,
    lower_m24_exact_present: u64,
    negated_upper_m24_exact_present: u64,
    lower_m24_missing: u64,
    negated_upper_m24_missing: u64,
    lower_m24_strict_wins: u64,
    negated_upper_m24_strict_wins: u64,
    max_lower_signed_gain: f64,
    max_negated_upper_signed_gain: f64,
    max_lower_m24_signed_gain: f64,
    max_negated_upper_m24_signed_gain: f64,
    max_lower_m17_optimized_improvement: f64,
    max_negated_upper_m17_optimized_improvement: f64,
    best_lower_threshold_residual: f64,
    best_negated_upper_threshold_residual: f64,
    best_joint_threshold_residual: f64,
    best_counterfactual_m24_joint_threshold_residual: f64,
    m24_only_would_verify: u64,
    depth_two_completed_sides: u64,
    depth_two_invalid_sides: u64,
    depth_two_not_requested_sides: u64,
    depth_two_no_time_sides: u64,
    depth_two_budget_fallback_sides: u64,
    depth_two_transform_fallback_sides: u64,
    depth_two_incoherent_pairs: u64,
    lower_depth_two_strict_wins: u64,
    negated_upper_depth_two_strict_wins: u64,
    max_lower_depth_two_signed_gain: f64,
    max_negated_upper_depth_two_signed_gain: f64,
    best_counterfactual_depth_two_joint_threshold_residual: f64,
    depth_two_only_would_verify: u64,
    total_wall_nanos: u64,
    max_attempt_wall_nanos: u64,
    reserved_total_wall_nanos: u64,
    reserved_max_attempt_wall_nanos: u64,
    max_peak_live_bytes: u64,
    in_flight: bool,
}

fn duration_nanos_saturating(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn atomic_max_finite(target: &AtomicU64, candidate: f64) {
    if !candidate.is_finite() {
        return;
    }
    let mut observed = target.load(Ordering::Relaxed);
    loop {
        if candidate <= f64::from_bits(observed) {
            return;
        }
        match target.compare_exchange_weak(
            observed,
            candidate.to_bits(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return,
            Err(actual) => observed = actual,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ValidatedDepthTwoMeasurement {
    counterfactual_lower_bound: f64,
    signed_gain: f64,
}

fn reconstruct_strict_m17_m20_attribution(
    m17_lower_bound: f64,
    m20_lower_bound: Option<f64>,
    m20_status: CganCzM20Status,
) -> Option<(f64, ReluTailBoxCutSelection)> {
    if !m17_lower_bound.is_finite() {
        return None;
    }
    match (m20_status, m20_lower_bound) {
        (CganCzM20Status::Completed, Some(m20)) if m20.is_finite() => {
            if m20 > m17_lower_bound {
                Some((m20, ReluTailBoxCutSelection::Auxiliary))
            } else {
                Some((m17_lower_bound, ReluTailBoxCutSelection::Original))
            }
        }
        (CganCzM20Status::Fallback, None) => {
            Some((m17_lower_bound, ReluTailBoxCutSelection::Original))
        }
        (CganCzM20Status::NotRequested, _) => None,
        _ => None,
    }
}

fn valid_depth_two_m17_candidates(candidate: &CganCzM17CandidateTelemetry) -> bool {
    if !candidate.selected_lower_bound.is_finite() {
        return false;
    }
    if !candidate.zero_positive_slope_lower_bound.is_finite()
        || !candidate
            .upper_endpoint_lower_bound
            .is_none_or(f64::is_finite)
        || !candidate.canonical_lower_bound.is_none_or(f64::is_finite)
        || !candidate.optimized_lower_bound.is_none_or(f64::is_finite)
        || !candidate.best_nonoptimized_lower_bound.is_finite()
        || !candidate.optimized_improvement.is_finite()
        || candidate.optimized_improvement < 0.0
        || candidate.candidates_replayed == 0
    {
        return false;
    }

    let mut best_nonoptimized = candidate.zero_positive_slope_lower_bound;
    for replay in [
        candidate.upper_endpoint_lower_bound,
        candidate.canonical_lower_bound,
    ]
    .into_iter()
    .flatten()
    {
        best_nonoptimized = best_nonoptimized.max(replay);
    }
    let optimized_improvement = candidate
        .optimized_lower_bound
        .map_or(0.0, |optimized| (optimized - best_nonoptimized).max(0.0));
    let best_zero_predicate = candidate
        .optimized_lower_bound
        .map_or(best_nonoptimized, |optimized| {
            best_nonoptimized.max(optimized)
        });
    candidate.best_nonoptimized_lower_bound.to_bits() == best_nonoptimized.to_bits()
        && candidate.optimized_improvement.to_bits() == optimized_improvement.to_bits()
        // The selected certificate may improve on every zero-predicate replay
        // through certified predicate multipliers, but it may not regress them.
        && candidate.selected_lower_bound >= best_zero_predicate
}

/// Validate one completed depth-two shadow observation for aggregation only.
///
/// This helper is deliberately never consulted by `completed_rows_from_receipt`
/// or the strict property comparison. A malformed observation is excluded from
/// diagnostics while the independently authenticated M17/M20 row remains
/// untouched.
fn valid_depth_two_completed_measurement(
    measurement: &CganCzDepthTwoCompletedMeasurement,
    historical_lower_bound: f64,
    outer_peak_live_bytes: usize,
    outer_max_peak_live_bytes: usize,
    outer_charged_items: usize,
    outer_deadline_polls: usize,
) -> Option<ValidatedDepthTwoMeasurement> {
    if !historical_lower_bound.is_finite()
        || measurement.historical_lower_bound.to_bits() != historical_lower_bound.to_bits()
        || !valid_depth_two_m17_candidates(&measurement.downstream_m17_candidates)
        || !valid_depth_two_m17_candidates(&measurement.upstream_m17_candidates)
        || !coherent_depth_two_m20_optional_budget_error(
            measurement.upstream_m20_status,
            measurement.upstream_m20_optional_budget_error.as_ref(),
            outer_max_peak_live_bytes,
        )
        || !measurement.counterfactual_lower_bound.is_finite()
        || !measurement.signed_gain.is_finite()
        || measurement.plan.input_shape != DEPTH_TWO_INPUT_SHAPE
        || measurement.plan.output_shape != DEPTH_TWO_OUTPUT_SHAPE
        || measurement.plan.weight_shape != DEPTH_TWO_WEIGHT_SHAPE
        || measurement.plan.weight_elements != DEPTH_TWO_WEIGHT_ELEMENTS
        || measurement.plan.kernel_visits != DEPTH_TWO_KERNEL_VISITS
        || measurement
            .plan
            .pulled_margin_construction_exact_product_bound
            != DEPTH_TWO_EXACT_PRODUCT_BOUND
        || measurement.peak_live_bytes == 0
        || measurement.peak_live_bytes > outer_peak_live_bytes
        || outer_peak_live_bytes > outer_max_peak_live_bytes
        || measurement.charged_items == 0
        || measurement.charged_items > outer_charged_items
        || measurement.deadline_polls == 0
        || measurement.deadline_polls > outer_deadline_polls
    {
        return None;
    }

    let (upstream_lower_bound, upstream_selection) = reconstruct_strict_m17_m20_attribution(
        measurement.upstream_m17_candidates.selected_lower_bound,
        measurement.upstream_m20_lower_bound,
        measurement.upstream_m20_status,
    )?;
    if measurement.upstream_m17_m20_selection != upstream_selection {
        return None;
    }
    let counterfactual_lower_bound = if upstream_lower_bound > historical_lower_bound {
        upstream_lower_bound
    } else {
        historical_lower_bound
    };
    let signed_gain = upstream_lower_bound - historical_lower_bound;
    if !signed_gain.is_finite()
        || measurement.counterfactual_lower_bound.to_bits() != counterfactual_lower_bound.to_bits()
        || measurement.signed_gain.to_bits() != signed_gain.to_bits()
    {
        return None;
    }
    Some(ValidatedDepthTwoMeasurement {
        counterfactual_lower_bound,
        signed_gain,
    })
}

fn coherent_depth_two_budget_fallback(
    error: &ConstrainedZonotopeCallBudgetError,
    outer_max_peak_live_bytes: usize,
) -> bool {
    match error {
        ConstrainedZonotopeCallBudgetError::DeadlineExpired { .. }
        | ConstrainedZonotopeCallBudgetError::ResourceOverflow { .. } => true,
        ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { required, limit } => {
            // The outer report cap was equality-authenticated before this
            // telemetry-only validator runs. Do not admit a structurally
            // plausible peak refusal minted against a different budget.
            required > limit && *limit == outer_max_peak_live_bytes
        }
    }
}

fn coherent_depth_two_m20_optional_budget_error(
    status: CganCzM20Status,
    error: Option<&ConstrainedZonotopeCallBudgetError>,
    outer_max_peak_live_bytes: usize,
) -> bool {
    match status {
        CganCzM20Status::Completed => error.is_none(),
        CganCzM20Status::Fallback => error.is_none_or(|error| {
            coherent_depth_two_budget_fallback(error, outer_max_peak_live_bytes)
        }),
        CganCzM20Status::NotRequested => false,
    }
}

impl CganInputLeafTelemetry {
    fn new() -> Self {
        let negative_infinity = f64::NEG_INFINITY.to_bits();
        Self {
            consultations: AtomicU64::new(0),
            attempts: AtomicU64::new(0),
            reserved_attempts: AtomicU64::new(0),
            completions: AtomicU64::new(0),
            reserved_completions: AtomicU64::new(0),
            verified_leaves: AtomicU64::new(0),
            reserved_verified_leaves: AtomicU64::new(0),
            late_results: AtomicU64::new(0),
            depth_or_volume_skips: AtomicU64::new(0),
            objective_shortfall_skips: AtomicU64::new(0),
            reserved_depth_or_volume_skips: AtomicU64::new(0),
            reserved_objective_shortfall_skips: AtomicU64::new(0),
            global_remaining_skips: AtomicU64::new(0),
            total_wall_skips: AtomicU64::new(0),
            attempt_cap_skips: AtomicU64::new(0),
            concurrent_skips: AtomicU64::new(0),
            lower_m20_fallbacks: AtomicU64::new(0),
            negated_upper_m20_fallbacks: AtomicU64::new(0),
            lower_m20_strict_wins: AtomicU64::new(0),
            negated_upper_m20_strict_wins: AtomicU64::new(0),
            lower_m24_exact_present: AtomicU64::new(0),
            negated_upper_m24_exact_present: AtomicU64::new(0),
            lower_m24_missing: AtomicU64::new(0),
            negated_upper_m24_missing: AtomicU64::new(0),
            lower_m24_strict_wins: AtomicU64::new(0),
            negated_upper_m24_strict_wins: AtomicU64::new(0),
            max_lower_signed_gain_bits: AtomicU64::new(negative_infinity),
            max_negated_upper_signed_gain_bits: AtomicU64::new(negative_infinity),
            max_lower_m24_signed_gain_bits: AtomicU64::new(negative_infinity),
            max_negated_upper_m24_signed_gain_bits: AtomicU64::new(negative_infinity),
            max_lower_m17_optimized_improvement_bits: AtomicU64::new(negative_infinity),
            max_negated_upper_m17_optimized_improvement_bits: AtomicU64::new(negative_infinity),
            best_lower_threshold_residual_bits: AtomicU64::new(negative_infinity),
            best_negated_upper_threshold_residual_bits: AtomicU64::new(negative_infinity),
            best_joint_threshold_residual_bits: AtomicU64::new(negative_infinity),
            best_counterfactual_m24_joint_threshold_residual_bits: AtomicU64::new(
                negative_infinity,
            ),
            m24_only_would_verify: AtomicU64::new(0),
            depth_two_completed_sides: AtomicU64::new(0),
            depth_two_invalid_sides: AtomicU64::new(0),
            depth_two_not_requested_sides: AtomicU64::new(0),
            depth_two_no_time_sides: AtomicU64::new(0),
            depth_two_budget_fallback_sides: AtomicU64::new(0),
            depth_two_transform_fallback_sides: AtomicU64::new(0),
            depth_two_incoherent_pairs: AtomicU64::new(0),
            lower_depth_two_strict_wins: AtomicU64::new(0),
            negated_upper_depth_two_strict_wins: AtomicU64::new(0),
            max_lower_depth_two_signed_gain_bits: AtomicU64::new(negative_infinity),
            max_negated_upper_depth_two_signed_gain_bits: AtomicU64::new(negative_infinity),
            best_counterfactual_depth_two_joint_threshold_residual_bits: AtomicU64::new(
                negative_infinity,
            ),
            depth_two_only_would_verify: AtomicU64::new(0),
            total_wall_nanos: AtomicU64::new(0),
            max_attempt_wall_nanos: AtomicU64::new(0),
            reserved_total_wall_nanos: AtomicU64::new(0),
            reserved_max_attempt_wall_nanos: AtomicU64::new(0),
            max_peak_live_bytes: AtomicU64::new(0),
            in_flight: AtomicBool::new(false),
            final_emitted: AtomicBool::new(false),
        }
    }

    fn record_consultation(&self) {
        self.consultations.fetch_add(1, Ordering::Relaxed);
    }

    fn record_skip(&self, reason: LeafAdmissionSkip) {
        let counter = match reason {
            LeafAdmissionSkip::DepthOrVolume => &self.depth_or_volume_skips,
            LeafAdmissionSkip::ObjectiveShortfall => &self.objective_shortfall_skips,
            LeafAdmissionSkip::GlobalRemaining => &self.global_remaining_skips,
            LeafAdmissionSkip::TotalWall => &self.total_wall_skips,
            LeafAdmissionSkip::AttemptCap => &self.attempt_cap_skips,
            LeafAdmissionSkip::Concurrent => &self.concurrent_skips,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    fn record_frontier_skip(&self, tranche: LeafAttemptTranche, rejection: LeafFrontierRejection) {
        match rejection {
            LeafFrontierRejection::DepthOrVolume => {
                self.record_skip(LeafAdmissionSkip::DepthOrVolume);
                if tranche == LeafAttemptTranche::Reserved {
                    self.reserved_depth_or_volume_skips
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
            LeafFrontierRejection::ObjectiveShortfall => {
                self.record_skip(LeafAdmissionSkip::ObjectiveShortfall);
                if tranche == LeafAttemptTranche::Reserved {
                    self.reserved_objective_shortfall_skips
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    /// Claim exactly the ordinal inspected while holding `in_flight`.
    ///
    /// The compare-exchange makes a stale tranche selection fail closed even
    /// if a future caller accidentally bypasses the single-flight protocol.
    fn reserve_attempt(&self, observed_attempts: u64, tranche: LeafAttemptTranche) -> Option<u64> {
        let attempt = observed_attempts.checked_add(1)?;
        self.attempts
            .compare_exchange(
                observed_attempts,
                attempt,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .ok()?;
        if tranche == LeafAttemptTranche::Reserved {
            self.reserved_attempts.fetch_add(1, Ordering::Relaxed);
        }
        Some(attempt)
    }

    fn record_depth_two_side(
        &self,
        measurement: &CganCzDepthTwoMeasurement,
        historical_lower_bound: f64,
        outer_peak_live_bytes: usize,
        outer_max_peak_live_bytes: usize,
        outer_charged_items: usize,
        outer_deadline_polls: usize,
    ) -> Option<ValidatedDepthTwoMeasurement> {
        match measurement {
            CganCzDepthTwoMeasurement::NotRequested => {
                self.depth_two_not_requested_sides
                    .fetch_add(1, Ordering::Relaxed);
                None
            }
            CganCzDepthTwoMeasurement::NoTime => {
                self.depth_two_no_time_sides.fetch_add(1, Ordering::Relaxed);
                None
            }
            CganCzDepthTwoMeasurement::BudgetFallback(error) => {
                if coherent_depth_two_budget_fallback(error, outer_max_peak_live_bytes) {
                    self.depth_two_budget_fallback_sides
                        .fetch_add(1, Ordering::Relaxed);
                } else {
                    self.depth_two_invalid_sides.fetch_add(1, Ordering::Relaxed);
                }
                None
            }
            CganCzDepthTwoMeasurement::TransformFallback(failure) => {
                match failure {
                    CganCzDepthTwoTransformFailure::Setup
                    | CganCzDepthTwoTransformFailure::Conv2d
                    | CganCzDepthTwoTransformFailure::BatchNorm
                    | CganCzDepthTwoTransformFailure::ReluTail => {
                        self.depth_two_transform_fallback_sides
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
                None
            }
            CganCzDepthTwoMeasurement::Completed(completed) => {
                let Some(validated) = valid_depth_two_completed_measurement(
                    completed,
                    historical_lower_bound,
                    outer_peak_live_bytes,
                    outer_max_peak_live_bytes,
                    outer_charged_items,
                    outer_deadline_polls,
                ) else {
                    self.depth_two_invalid_sides.fetch_add(1, Ordering::Relaxed);
                    return None;
                };
                self.depth_two_completed_sides
                    .fetch_add(1, Ordering::Relaxed);
                Some(validated)
            }
        }
    }

    fn record_completion(
        &self,
        rows: &CganCzLeafRowBounds,
        thresholds: AuthenticatedRows,
        peak_live_bytes: usize,
        max_peak_live_bytes: usize,
        charged_items: usize,
        deadline_polls: usize,
        tranche: LeafAttemptTranche,
    ) {
        match rows.lower_m20_status {
            CganCzM20Status::Completed => {
                if let Some(m20) = rows.lower_m20_lower_bound {
                    let m17 = rows.lower_m17_candidates.selected_lower_bound;
                    atomic_max_finite(&self.max_lower_signed_gain_bits, m20 - m17);
                    if m20 > m17 {
                        self.lower_m20_strict_wins.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            CganCzM20Status::Fallback => {
                self.lower_m20_fallbacks.fetch_add(1, Ordering::Relaxed);
            }
            CganCzM20Status::NotRequested => {}
        }
        match rows.negated_upper_m20_status {
            CganCzM20Status::Completed => {
                if let Some(m20) = rows.negated_upper_m20_lower_bound {
                    let m17 = rows.negated_upper_m17_candidates.selected_lower_bound;
                    atomic_max_finite(&self.max_negated_upper_signed_gain_bits, m20 - m17);
                    if m20 > m17 {
                        self.negated_upper_m20_strict_wins
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            CganCzM20Status::Fallback => {
                self.negated_upper_m20_fallbacks
                    .fetch_add(1, Ordering::Relaxed);
            }
            CganCzM20Status::NotRequested => {}
        }
        let mut counterfactual_lower = None;
        if let Some(measurement) = rows.lower_m24_measurement.as_ref() {
            counterfactual_lower = Some(measurement.counterfactual_lower_bound);
            if let Some(exact) = measurement.exact_box_cut_lower_bound {
                self.lower_m24_exact_present.fetch_add(1, Ordering::Relaxed);
                atomic_max_finite(&self.max_lower_m24_signed_gain_bits, exact - rows.lower_y);
                if exact > rows.lower_y {
                    self.lower_m24_strict_wins.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                self.lower_m24_missing.fetch_add(1, Ordering::Relaxed);
            }
        }
        let mut counterfactual_negated_upper = None;
        if let Some(measurement) = rows.negated_upper_m24_measurement.as_ref() {
            counterfactual_negated_upper = Some(measurement.counterfactual_lower_bound);
            if let Some(exact) = measurement.exact_box_cut_lower_bound {
                self.negated_upper_m24_exact_present
                    .fetch_add(1, Ordering::Relaxed);
                atomic_max_finite(
                    &self.max_negated_upper_m24_signed_gain_bits,
                    exact - rows.lower_neg_y,
                );
                if exact > rows.lower_neg_y {
                    self.negated_upper_m24_strict_wins
                        .fetch_add(1, Ordering::Relaxed);
                }
            } else {
                self.negated_upper_m24_missing
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        let depth_two_lower = self.record_depth_two_side(
            &rows.lower_depth_two_measurement,
            rows.lower_y,
            peak_live_bytes,
            max_peak_live_bytes,
            charged_items,
            deadline_polls,
        );
        let depth_two_negated_upper = self.record_depth_two_side(
            &rows.negated_upper_depth_two_measurement,
            rows.lower_neg_y,
            peak_live_bytes,
            max_peak_live_bytes,
            charged_items,
            deadline_polls,
        );
        atomic_max_finite(
            &self.max_lower_m17_optimized_improvement_bits,
            rows.lower_m17_candidates.optimized_improvement,
        );
        atomic_max_finite(
            &self.max_negated_upper_m17_optimized_improvement_bits,
            rows.negated_upper_m17_candidates.optimized_improvement,
        );
        let lower_residual = rows.lower_y - thresholds.y_threshold;
        let negated_upper_residual = rows.lower_neg_y - thresholds.neg_y_threshold;
        atomic_max_finite(&self.best_lower_threshold_residual_bits, lower_residual);
        atomic_max_finite(
            &self.best_negated_upper_threshold_residual_bits,
            negated_upper_residual,
        );
        atomic_max_finite(
            &self.best_joint_threshold_residual_bits,
            lower_residual.min(negated_upper_residual),
        );
        if let (Some(counterfactual_lower), Some(counterfactual_negated_upper)) =
            (counterfactual_lower, counterfactual_negated_upper)
        {
            let counterfactual_lower_residual = counterfactual_lower - thresholds.y_threshold;
            let counterfactual_negated_upper_residual =
                counterfactual_negated_upper - thresholds.neg_y_threshold;
            atomic_max_finite(
                &self.best_counterfactual_m24_joint_threshold_residual_bits,
                counterfactual_lower_residual.min(counterfactual_negated_upper_residual),
            );
            if counterfactual_lower_residual > 0.0
                && counterfactual_negated_upper_residual > 0.0
                && !(lower_residual > 0.0 && negated_upper_residual > 0.0)
            {
                self.m24_only_would_verify.fetch_add(1, Ordering::Relaxed);
            }
        }
        if depth_two_lower.is_some() || depth_two_negated_upper.is_some() {
            let counterfactual_lower = depth_two_lower.map_or(rows.lower_y, |measurement| {
                measurement.counterfactual_lower_bound
            });
            let counterfactual_negated_upper = depth_two_negated_upper
                .map_or(rows.lower_neg_y, |measurement| {
                    measurement.counterfactual_lower_bound
                });
            // Both values are independently claimed lower certificates for Y
            // and -Y over the same nonempty leaf. An inconsistent pair is
            // excluded from all gain telemetry but cannot reject the receipt.
            if counterfactual_lower > -counterfactual_negated_upper {
                self.depth_two_incoherent_pairs
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                if let Some(measurement) = depth_two_lower {
                    atomic_max_finite(
                        &self.max_lower_depth_two_signed_gain_bits,
                        measurement.signed_gain,
                    );
                    if measurement.counterfactual_lower_bound > rows.lower_y {
                        self.lower_depth_two_strict_wins
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
                if let Some(measurement) = depth_two_negated_upper {
                    atomic_max_finite(
                        &self.max_negated_upper_depth_two_signed_gain_bits,
                        measurement.signed_gain,
                    );
                    if measurement.counterfactual_lower_bound > rows.lower_neg_y {
                        self.negated_upper_depth_two_strict_wins
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
                let counterfactual_lower_residual = counterfactual_lower - thresholds.y_threshold;
                let counterfactual_negated_upper_residual =
                    counterfactual_negated_upper - thresholds.neg_y_threshold;
                atomic_max_finite(
                    &self.best_counterfactual_depth_two_joint_threshold_residual_bits,
                    counterfactual_lower_residual.min(counterfactual_negated_upper_residual),
                );
                if counterfactual_lower_residual > 0.0
                    && counterfactual_negated_upper_residual > 0.0
                    && !(lower_residual > 0.0 && negated_upper_residual > 0.0)
                {
                    self.depth_two_only_would_verify
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        self.max_peak_live_bytes.fetch_max(
            u64::try_from(peak_live_bytes).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.completions.fetch_add(1, Ordering::Relaxed);
        if tranche == LeafAttemptTranche::Reserved {
            self.reserved_completions.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn record_verified(&self, tranche: LeafAttemptTranche) {
        self.verified_leaves.fetch_add(1, Ordering::Relaxed);
        if tranche == LeafAttemptTranche::Reserved {
            self.reserved_verified_leaves
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    fn snapshot(&self) -> CganInputLeafTelemetrySnapshot {
        CganInputLeafTelemetrySnapshot {
            consultations: self.consultations.load(Ordering::Relaxed),
            attempts: self.attempts.load(Ordering::Relaxed),
            reserved_attempts: self.reserved_attempts.load(Ordering::Relaxed),
            completions: self.completions.load(Ordering::Relaxed),
            reserved_completions: self.reserved_completions.load(Ordering::Relaxed),
            verified_leaves: self.verified_leaves.load(Ordering::Relaxed),
            reserved_verified_leaves: self.reserved_verified_leaves.load(Ordering::Relaxed),
            late_results: self.late_results.load(Ordering::Relaxed),
            depth_or_volume_skips: self.depth_or_volume_skips.load(Ordering::Relaxed),
            objective_shortfall_skips: self.objective_shortfall_skips.load(Ordering::Relaxed),
            reserved_depth_or_volume_skips: self
                .reserved_depth_or_volume_skips
                .load(Ordering::Relaxed),
            reserved_objective_shortfall_skips: self
                .reserved_objective_shortfall_skips
                .load(Ordering::Relaxed),
            global_remaining_skips: self.global_remaining_skips.load(Ordering::Relaxed),
            total_wall_skips: self.total_wall_skips.load(Ordering::Relaxed),
            attempt_cap_skips: self.attempt_cap_skips.load(Ordering::Relaxed),
            concurrent_skips: self.concurrent_skips.load(Ordering::Relaxed),
            lower_m20_fallbacks: self.lower_m20_fallbacks.load(Ordering::Relaxed),
            negated_upper_m20_fallbacks: self.negated_upper_m20_fallbacks.load(Ordering::Relaxed),
            lower_m20_strict_wins: self.lower_m20_strict_wins.load(Ordering::Relaxed),
            negated_upper_m20_strict_wins: self
                .negated_upper_m20_strict_wins
                .load(Ordering::Relaxed),
            lower_m24_exact_present: self.lower_m24_exact_present.load(Ordering::Relaxed),
            negated_upper_m24_exact_present: self
                .negated_upper_m24_exact_present
                .load(Ordering::Relaxed),
            lower_m24_missing: self.lower_m24_missing.load(Ordering::Relaxed),
            negated_upper_m24_missing: self.negated_upper_m24_missing.load(Ordering::Relaxed),
            lower_m24_strict_wins: self.lower_m24_strict_wins.load(Ordering::Relaxed),
            negated_upper_m24_strict_wins: self
                .negated_upper_m24_strict_wins
                .load(Ordering::Relaxed),
            max_lower_signed_gain: f64::from_bits(
                self.max_lower_signed_gain_bits.load(Ordering::Relaxed),
            ),
            max_negated_upper_signed_gain: f64::from_bits(
                self.max_negated_upper_signed_gain_bits
                    .load(Ordering::Relaxed),
            ),
            max_lower_m24_signed_gain: f64::from_bits(
                self.max_lower_m24_signed_gain_bits.load(Ordering::Relaxed),
            ),
            max_negated_upper_m24_signed_gain: f64::from_bits(
                self.max_negated_upper_m24_signed_gain_bits
                    .load(Ordering::Relaxed),
            ),
            max_lower_m17_optimized_improvement: f64::from_bits(
                self.max_lower_m17_optimized_improvement_bits
                    .load(Ordering::Relaxed),
            ),
            max_negated_upper_m17_optimized_improvement: f64::from_bits(
                self.max_negated_upper_m17_optimized_improvement_bits
                    .load(Ordering::Relaxed),
            ),
            best_lower_threshold_residual: f64::from_bits(
                self.best_lower_threshold_residual_bits
                    .load(Ordering::Relaxed),
            ),
            best_negated_upper_threshold_residual: f64::from_bits(
                self.best_negated_upper_threshold_residual_bits
                    .load(Ordering::Relaxed),
            ),
            best_joint_threshold_residual: f64::from_bits(
                self.best_joint_threshold_residual_bits
                    .load(Ordering::Relaxed),
            ),
            best_counterfactual_m24_joint_threshold_residual: f64::from_bits(
                self.best_counterfactual_m24_joint_threshold_residual_bits
                    .load(Ordering::Relaxed),
            ),
            m24_only_would_verify: self.m24_only_would_verify.load(Ordering::Relaxed),
            depth_two_completed_sides: self.depth_two_completed_sides.load(Ordering::Relaxed),
            depth_two_invalid_sides: self.depth_two_invalid_sides.load(Ordering::Relaxed),
            depth_two_not_requested_sides: self
                .depth_two_not_requested_sides
                .load(Ordering::Relaxed),
            depth_two_no_time_sides: self.depth_two_no_time_sides.load(Ordering::Relaxed),
            depth_two_budget_fallback_sides: self
                .depth_two_budget_fallback_sides
                .load(Ordering::Relaxed),
            depth_two_transform_fallback_sides: self
                .depth_two_transform_fallback_sides
                .load(Ordering::Relaxed),
            depth_two_incoherent_pairs: self.depth_two_incoherent_pairs.load(Ordering::Relaxed),
            lower_depth_two_strict_wins: self.lower_depth_two_strict_wins.load(Ordering::Relaxed),
            negated_upper_depth_two_strict_wins: self
                .negated_upper_depth_two_strict_wins
                .load(Ordering::Relaxed),
            max_lower_depth_two_signed_gain: f64::from_bits(
                self.max_lower_depth_two_signed_gain_bits
                    .load(Ordering::Relaxed),
            ),
            max_negated_upper_depth_two_signed_gain: f64::from_bits(
                self.max_negated_upper_depth_two_signed_gain_bits
                    .load(Ordering::Relaxed),
            ),
            best_counterfactual_depth_two_joint_threshold_residual: f64::from_bits(
                self.best_counterfactual_depth_two_joint_threshold_residual_bits
                    .load(Ordering::Relaxed),
            ),
            depth_two_only_would_verify: self.depth_two_only_would_verify.load(Ordering::Relaxed),
            total_wall_nanos: self.total_wall_nanos.load(Ordering::Relaxed),
            max_attempt_wall_nanos: self.max_attempt_wall_nanos.load(Ordering::Relaxed),
            reserved_total_wall_nanos: self.reserved_total_wall_nanos.load(Ordering::Relaxed),
            reserved_max_attempt_wall_nanos: self
                .reserved_max_attempt_wall_nanos
                .load(Ordering::Relaxed),
            max_peak_live_bytes: self.max_peak_live_bytes.load(Ordering::Relaxed),
            in_flight: self.in_flight.load(Ordering::Acquire),
        }
    }
}

struct LeafAttemptGuard<'a> {
    telemetry: &'a CganInputLeafTelemetry,
    started: Instant,
    tranche: LeafAttemptTranche,
}

impl Drop for LeafAttemptGuard<'_> {
    fn drop(&mut self) {
        let elapsed = duration_nanos_saturating(self.started.elapsed());
        self.telemetry
            .total_wall_nanos
            .fetch_add(elapsed, Ordering::Relaxed);
        self.telemetry
            .max_attempt_wall_nanos
            .fetch_max(elapsed, Ordering::Relaxed);
        if self.tranche == LeafAttemptTranche::Reserved {
            self.telemetry
                .reserved_total_wall_nanos
                .fetch_add(elapsed, Ordering::Relaxed);
            self.telemetry
                .reserved_max_attempt_wall_nanos
                .fetch_max(elapsed, Ordering::Relaxed);
        }
        self.telemetry.in_flight.store(false, Ordering::Release);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LeafResourceLimits {
    retained_live_bytes: usize,
    max_peak_live_bytes: usize,
}

impl LeafResourceLimits {
    const PRODUCTION: Self = Self {
        retained_live_bytes: RETAINED_LIVE_BYTES,
        max_peak_live_bytes: MAX_PEAK_LIVE_BYTES,
    };

    fn valid(self) -> bool {
        self.retained_live_bytes <= self.max_peak_live_bytes
    }
}

/// The exact sign-normalized safe rows derived from the authenticated moat.
///
/// `Y > y_threshold` refutes the low unsafe clause; `-Y > neg_y_threshold`
/// refutes the high unsafe clause using source-authenticated directed bounds.
/// The `request_*` values separately authenticate the ordinary planner rows;
/// an exact decimal can lie just beyond its ordinary binary64 parse, so those
/// two roles must not share one rounded scalar. Request row order is
/// immaterial because both singleton clauses must be certified before
/// authority is available.
#[derive(Clone, Copy, Debug, PartialEq)]
struct AuthenticatedRows {
    request_y_threshold: f32,
    request_neg_y_threshold: f32,
    y_threshold: f64,
    neg_y_threshold: f64,
}

trait LeafRowBounder: Send + Sync {
    fn bound(
        &self,
        graph: &GraphNetwork,
        leaf_lower: &[f32],
        leaf_upper: &[f32],
        budget: ConstrainedZonotopeCallBudget,
    ) -> CganCzLeafRowReport;
}

struct AuthoredLeafRowBounder {
    profile: CganCzImgSz32Profile,
    model: OnnxModel,
    moat: CertifiedScalarMoat,
    limits: CganCzSequentialLimits,
}

impl LeafRowBounder for AuthoredLeafRowBounder {
    fn bound(
        &self,
        graph: &GraphNetwork,
        leaf_lower: &[f32],
        leaf_upper: &[f32],
        budget: ConstrainedZonotopeCallBudget,
    ) -> CganCzLeafRowReport {
        bound_cgan_imgsz32_leaf_rows_unwired(
            self.profile,
            &self.model,
            graph,
            leaf_lower,
            leaf_upper,
            self.moat,
            self.limits,
            budget,
        )
    }
}

struct CganInputLeafOracle {
    profile: CganCzImgSz32Profile,
    /// Process-unique graph identity without retaining a deep graph clone.
    /// The bounder separately replays full topology/parameter authentication
    /// against the request graph before publishing rows.
    expected_graph_scope: CutFoldScope,
    /// The BaB deadline that bounded construction. Leaf requests may carry
    /// this instant or an earlier one; no post-setup duration may extend it or
    /// mint new authority.
    authority_deadline: Instant,
    root_lower: [f32; LATENT_DIM],
    root_upper: [f32; LATENT_DIM],
    rows: AuthenticatedRows,
    resources: LeafResourceLimits,
    admission: LeafAdmissionPolicy,
    telemetry: CganInputLeafTelemetry,
    bounder: Arc<dyn LeafRowBounder>,
}

/// Keeps typed access to the per-property telemetry after the oracle is
/// coerced into the shared trait object (and possibly a composite).  Dispatch
/// explicitly emits after BaB returns; Drop covers every earlier return.
pub(super) struct CganInputLeafAttachment(Arc<CganInputLeafOracle>);

impl CganInputLeafAttachment {
    pub(super) fn oracle(&self) -> Arc<dyn GraphMipLeafOracle> {
        self.0.clone()
    }

    pub(super) fn emit_final_once(&self, status: &'static str) {
        self.0.emit_final_once(status);
    }
}

impl Drop for CganInputLeafAttachment {
    fn drop(&mut self) {
        self.emit_final_once("drop");
    }
}

fn write_telemetry_line<W: std::io::Write>(
    writer: &mut W,
    status: &'static str,
    profile: CganCzImgSz32Profile,
    policy: LeafAdmissionPolicy,
    snapshot: &CganInputLeafTelemetrySnapshot,
) -> std::io::Result<()> {
    let declines = snapshot
        .attempts
        .saturating_sub(snapshot.completions)
        .saturating_sub(u64::from(snapshot.in_flight));
    writeln!(
        writer,
        "NY_CGAN_INPUT_LEAF status={status} profile={profile:?} \
         depth_two_production_mode={} \
         consultations={} attempts={} reserved_attempts={} completions={} \
         reserved_completions={} declines={} in_flight={} verified={} reserved_verified={} late={} \
         skip_depth_volume={} skip_shortfall={} skip_remaining={} skip_total_wall={} \
         skip_attempt_cap={} skip_concurrent={} reserved_skip_depth_volume={} \
         reserved_skip_shortfall={} lower_fallbacks={} upper_fallbacks={} \
         lower_wins={} upper_wins={} max_lower_m20_gain={:.17e} \
         max_upper_m20_gain={:.17e} max_lower_m17_opt_gain={:.17e} \
         max_upper_m17_opt_gain={:.17e} best_lower_residual={:.17e} \
         best_upper_residual={:.17e} best_joint_residual={:.17e} \
         lower_m24_exact={} upper_m24_exact={} lower_m24_missing={} upper_m24_missing={} \
         lower_m24_wins={} upper_m24_wins={} max_lower_m24_gain={:.17e} \
         max_upper_m24_gain={:.17e} best_m24_joint_residual={:.17e} \
         m24_only_would_verify={} d2_completed_sides={} d2_invalid_sides={} \
         d2_not_requested_sides={} d2_no_time_sides={} d2_budget_fallback_sides={} \
         d2_transform_fallback_sides={} d2_incoherent_pairs={} lower_d2_wins={} \
         upper_d2_wins={} max_lower_d2_gain={:.17e} max_upper_d2_gain={:.17e} \
         best_d2_joint_residual={:.17e} d2_only_would_verify={} \
         total_wall_ms={:.3} max_attempt_ms={:.3} \
         reserved_total_wall_ms={:.3} reserved_max_attempt_ms={:.3} \
         max_peak_live_bytes={} policy_min_depth={} policy_max_volume={:.8e} \
         policy_max_shortfall={:.8e} policy_primary_max_attempts={} \
         policy_reserved_min_depth={} policy_reserved_max_volume={:.8e} \
         policy_reserved_max_shortfall={:.8e} policy_max_attempts={} \
         policy_call_secs={:.3} policy_total_wall_secs={:.3} policy_min_remaining_secs={:.3}",
        CGAN_DEPTH_TWO_PRODUCTION_MODE,
        snapshot.consultations,
        snapshot.attempts,
        snapshot.reserved_attempts,
        snapshot.completions,
        snapshot.reserved_completions,
        declines,
        snapshot.in_flight,
        snapshot.verified_leaves,
        snapshot.reserved_verified_leaves,
        snapshot.late_results,
        snapshot.depth_or_volume_skips,
        snapshot.objective_shortfall_skips,
        snapshot.global_remaining_skips,
        snapshot.total_wall_skips,
        snapshot.attempt_cap_skips,
        snapshot.concurrent_skips,
        snapshot.reserved_depth_or_volume_skips,
        snapshot.reserved_objective_shortfall_skips,
        snapshot.lower_m20_fallbacks,
        snapshot.negated_upper_m20_fallbacks,
        snapshot.lower_m20_strict_wins,
        snapshot.negated_upper_m20_strict_wins,
        snapshot.max_lower_signed_gain,
        snapshot.max_negated_upper_signed_gain,
        snapshot.max_lower_m17_optimized_improvement,
        snapshot.max_negated_upper_m17_optimized_improvement,
        snapshot.best_lower_threshold_residual,
        snapshot.best_negated_upper_threshold_residual,
        snapshot.best_joint_threshold_residual,
        snapshot.lower_m24_exact_present,
        snapshot.negated_upper_m24_exact_present,
        snapshot.lower_m24_missing,
        snapshot.negated_upper_m24_missing,
        snapshot.lower_m24_strict_wins,
        snapshot.negated_upper_m24_strict_wins,
        snapshot.max_lower_m24_signed_gain,
        snapshot.max_negated_upper_m24_signed_gain,
        snapshot.best_counterfactual_m24_joint_threshold_residual,
        snapshot.m24_only_would_verify,
        snapshot.depth_two_completed_sides,
        snapshot.depth_two_invalid_sides,
        snapshot.depth_two_not_requested_sides,
        snapshot.depth_two_no_time_sides,
        snapshot.depth_two_budget_fallback_sides,
        snapshot.depth_two_transform_fallback_sides,
        snapshot.depth_two_incoherent_pairs,
        snapshot.lower_depth_two_strict_wins,
        snapshot.negated_upper_depth_two_strict_wins,
        snapshot.max_lower_depth_two_signed_gain,
        snapshot.max_negated_upper_depth_two_signed_gain,
        snapshot.best_counterfactual_depth_two_joint_threshold_residual,
        snapshot.depth_two_only_would_verify,
        snapshot.total_wall_nanos as f64 / 1.0e6,
        snapshot.max_attempt_wall_nanos as f64 / 1.0e6,
        snapshot.reserved_total_wall_nanos as f64 / 1.0e6,
        snapshot.reserved_max_attempt_wall_nanos as f64 / 1.0e6,
        snapshot.max_peak_live_bytes,
        policy.min_depth,
        policy.max_normalized_volume,
        policy.max_worst_shortfall,
        policy.primary_max_attempts,
        policy.reserved_min_depth,
        policy.reserved_max_normalized_volume,
        policy.reserved_max_worst_shortfall,
        policy.max_attempts,
        policy.call_budget.as_secs_f64(),
        policy.total_wall_budget.as_secs_f64(),
        policy.min_global_remaining.as_secs_f64(),
    )
}

fn emit_telemetry_line(
    status: &'static str,
    profile: CganCzImgSz32Profile,
    policy: LeafAdmissionPolicy,
    snapshot: &CganInputLeafTelemetrySnapshot,
) {
    let stderr = std::io::stderr();
    let mut stderr = stderr.lock();
    let _ = write_telemetry_line(&mut stderr, status, profile, policy, snapshot);
}

fn write_m24_missing_certificate_lines<W: std::io::Write>(
    writer: &mut W,
    profile: CganCzImgSz32Profile,
    depth: usize,
    rows: &CganCzLeafRowBounds,
) -> usize {
    let mut emitted = 0;
    for (side, measurement) in [
        ("lower", rows.lower_m24_measurement.as_ref()),
        ("negated_upper", rows.negated_upper_m24_measurement.as_ref()),
    ] {
        let Some(measurement) = measurement else {
            continue;
        };
        if measurement.exact_box_cut_lower_bound.is_some() {
            continue;
        }
        if writeln!(
            writer,
            "NY_CGAN_INPUT_LEAF_M24 status=missing_certificate profile={profile:?} depth={depth} \
             side={side} m24_replay_status={:?} m24_search_status={:?} \
             m24_iterations={} m24_restarts={} m24_candidates={} m24_exact_replays={} \
             m24_optional_budget_error={:?} m24_plan={:?}",
            measurement.replay_status,
            measurement.search_status,
            measurement.iterations_completed,
            measurement.restarts_completed,
            measurement.candidates_scored,
            measurement.exact_replays,
            measurement.optional_budget_error.as_ref(),
            measurement.search_plan.as_ref(),
        )
        .is_ok()
        {
            emitted += 1;
        }
    }
    emitted
}

fn emit_m24_missing_certificate_lines(
    profile: CganCzImgSz32Profile,
    depth: usize,
    rows: &CganCzLeafRowBounds,
) {
    let stderr = std::io::stderr();
    let mut stderr = stderr.lock();
    let _ = write_m24_missing_certificate_lines(&mut stderr, profile, depth, rows);
}

fn expected_parameter_elements(profile: CganCzImgSz32Profile) -> usize {
    match profile {
        CganCzImgSz32Profile::Nch1 => NCH1_PARAMETER_ELEMENTS,
        CganCzImgSz32Profile::Nch3 => NCH3_PARAMETER_ELEMENTS,
    }
}

fn expected_model_source_bytes(profile: CganCzImgSz32Profile) -> u64 {
    match profile {
        CganCzImgSz32Profile::Nch1 => NCH1_MODEL_SOURCE_BYTES,
        CganCzImgSz32Profile::Nch3 => NCH3_MODEL_SOURCE_BYTES,
    }
}

fn authored_source_sizes_admitted(
    profile: CganCzImgSz32Profile,
    model_source_bytes: u64,
    property_source_bytes: u64,
) -> bool {
    model_source_bytes == expected_model_source_bytes(profile)
        && (1..=MAX_PROPERTY_SOURCE_BYTES).contains(&property_source_bytes)
}

fn authored_source_sizes(
    profile: CganCzImgSz32Profile,
    model_path: &Path,
    property_path: &Path,
) -> Option<(u64, u64)> {
    let model = std::fs::metadata(model_path).ok()?;
    let property = std::fs::metadata(property_path).ok()?;
    if !model.is_file()
        || !property.is_file()
        || !authored_source_sizes_admitted(profile, model.len(), property.len())
    {
        return None;
    }
    Some((model.len(), property.len()))
}

fn construction_deadline_live(deadline: Instant, stage: &'static str) -> bool {
    if Instant::now() < deadline {
        true
    } else {
        warn!(
            stage,
            "cGAN input-leaf oracle construction deadline expired"
        );
        false
    }
}

fn profile_from_model_path(path: &Path) -> Option<CganCzImgSz32Profile> {
    match path.file_name()?.to_str()? {
        "cGAN_imgSz32_nCh_1.onnx" => Some(CganCzImgSz32Profile::Nch1),
        "cGAN_imgSz32_nCh_3.onnx" => Some(CganCzImgSz32Profile::Nch3),
        _ => None,
    }
}

fn property_name_matches_profile(path: &Path, profile: CganCzImgSz32Profile) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let prefix = match profile {
        CganCzImgSz32Profile::Nch1 => "cGAN_imgSz32_nCh_1_prop_",
        CganCzImgSz32Profile::Nch3 => "cGAN_imgSz32_nCh_3_prop_",
    };
    // The authenticated competition sources are uncompressed. Keep gzip out
    // of this exact setup lane so the on-disk preflight is also the complete
    // property byte bound and no decompression happens before attachment.
    name.starts_with(prefix) && name.ends_with(".vnnlib")
}

fn same_parsed_property(left: &VnnLibSpec, right: &VnnLibSpec) -> bool {
    left.num_inputs == right.num_inputs
        && left.num_outputs == right.num_outputs
        && left.input_bounds == right.input_bounds
        && left.output_constraints == right.output_constraints
        && left.output_constraint_clauses == right.output_constraint_clauses
        && left.is_disjunction == right.is_disjunction
        && left.version == right.version
        && left.per_clause_input_bounds == right.per_clause_input_bounds
        && left.declared_input_bounds == right.declared_input_bounds
        && left.dual_network.is_none()
        && right.dual_network.is_none()
}

fn authenticated_rows(spec: &VnnLibSpec, moat: CertifiedScalarMoat) -> Option<AuthenticatedRows> {
    if spec.num_inputs != LATENT_DIM
        || spec.num_outputs != 1
        || !spec.is_disjunction
        || spec.dual_network.is_some()
        || spec.output_constraint_clauses.len() != 2
        || spec
            .output_constraint_clauses
            .iter()
            .any(|clause| clause.len() != 1)
        || spec
            .per_clause_input_bounds
            .iter()
            .any(|bounds| !bounds.is_empty())
    {
        return None;
    }

    let flattened: Vec<OutputConstraint> = spec
        .output_constraint_clauses
        .iter()
        .flat_map(|clause| clause.iter().cloned())
        .collect();
    if flattened != spec.output_constraints {
        return None;
    }

    let mut high = None;
    let mut low = None;
    for clause in &spec.output_constraint_clauses {
        match clause.first()? {
            OutputConstraint::GreaterEqConst(0, value) if value.is_finite() => {
                if high.replace(*value).is_some() {
                    return None;
                }
            }
            OutputConstraint::LessEqConst(0, value) if value.is_finite() => {
                if low.replace(*value).is_some() {
                    return None;
                }
            }
            _ => return None,
        }
    }
    let (high, low) = (high?, low?);
    if !(low < high
        && moat.low_upper().is_finite()
        && moat.high_lower().is_finite()
        && moat.low_upper() < moat.high_lower())
    {
        return None;
    }

    // Authenticate the ordinary grouped planner exactly, but prove the
    // source property against the independently extracted exact-decimal moat.
    let request_y_threshold = ny_core::f64_to_f32_up(low);
    let request_neg_y_threshold = -ny_core::f64_to_f32_down(high);
    let y_threshold = moat.low_upper().max(f64::from(request_y_threshold));
    let neg_y_threshold = (-moat.high_lower()).max(f64::from(request_neg_y_threshold));
    if !request_y_threshold.is_finite()
        || !request_neg_y_threshold.is_finite()
        || !y_threshold.is_finite()
        || !neg_y_threshold.is_finite()
        || y_threshold < moat.low_upper()
        || neg_y_threshold < -moat.high_lower()
        || y_threshold < f64::from(request_y_threshold)
        || neg_y_threshold < f64::from(request_neg_y_threshold)
    {
        return None;
    }
    Some(AuthenticatedRows {
        request_y_threshold,
        request_neg_y_threshold,
        y_threshold,
        neg_y_threshold,
    })
}

fn root_box(
    input: &BoundedTensor,
    certified: &CertifiedInputBox,
    spec: &VnnLibSpec,
) -> Option<([f32; LATENT_DIM], [f32; LATENT_DIM])> {
    if input.len() != LATENT_DIM || certified.len() != LATENT_DIM {
        return None;
    }
    let (expected_lower, expected_upper) = spec.split_input_bounds_f32();
    if expected_lower.len() != LATENT_DIM || expected_upper.len() != LATENT_DIM {
        return None;
    }
    let flat = input.flatten();
    let lower = flat.lower().as_slice()?;
    let upper = flat.upper().as_slice()?;
    if lower
        .iter()
        .zip(upper)
        .zip(0..LATENT_DIM)
        .any(|((&lo, &hi), index)| {
            !lo.is_finite()
                || !hi.is_finite()
                || lo > hi
                || lo.to_bits() != expected_lower[index].to_bits()
                || hi.to_bits() != expected_upper[index].to_bits()
                || f64::from(lo) > certified.lower()[index]
                || f64::from(hi) < certified.upper()[index]
        })
    {
        return None;
    }
    Some((lower.try_into().ok()?, upper.try_into().ok()?))
}

fn request_rows_match(req: &GraphInputLeafRequest<'_>, expected: AuthenticatedRows) -> bool {
    if req.objectives.nrows() != 2
        || req.objectives.ncols() != 1
        || req.thresholds.len() != 2
        || req.clause_sizes != [1, 1]
    {
        return false;
    }
    let mut saw_y = false;
    let mut saw_neg_y = false;
    for row in 0..2 {
        let coefficient = req.objectives[(row, 0)];
        let threshold = req.thresholds[row];
        if coefficient.to_bits() == 1.0_f32.to_bits()
            && threshold.to_bits() == expected.request_y_threshold.to_bits()
            && !saw_y
        {
            saw_y = true;
        } else if coefficient.to_bits() == (-1.0_f32).to_bits()
            && threshold.to_bits() == expected.request_neg_y_threshold.to_bits()
            && !saw_neg_y
        {
            saw_neg_y = true;
        } else {
            return false;
        }
    }
    saw_y && saw_neg_y
}

fn request_leaf_box(
    req: &GraphInputLeafRequest<'_>,
    root_lower: &[f32; LATENT_DIM],
    root_upper: &[f32; LATENT_DIM],
) -> Option<([f32; LATENT_DIM], [f32; LATENT_DIM])> {
    if req.input_bounds.len() != LATENT_DIM {
        return None;
    }
    let flat = req.input_bounds.flatten();
    let lower = flat.lower().as_slice()?;
    let upper = flat.upper().as_slice()?;
    if lower
        .iter()
        .zip(upper)
        .zip(root_lower.iter().zip(root_upper))
        .any(|((&lo, &hi), (&root_lo, &root_hi))| {
            !lo.is_finite() || !hi.is_finite() || lo > hi || lo < root_lo || hi > root_hi
        })
    {
        return None;
    }
    Some((lower.try_into().ok()?, upper.try_into().ok()?))
}

fn normalized_leaf_volume(
    lower: &[f32; LATENT_DIM],
    upper: &[f32; LATENT_DIM],
    root_lower: &[f32; LATENT_DIM],
    root_upper: &[f32; LATENT_DIM],
) -> Option<f64> {
    let mut volume = 1.0_f64;
    for index in 0..LATENT_DIM {
        let root_width = f64::from(root_upper[index]) - f64::from(root_lower[index]);
        let leaf_width = f64::from(upper[index]) - f64::from(lower[index]);
        if !root_width.is_finite()
            || !leaf_width.is_finite()
            || root_width < 0.0
            || leaf_width < 0.0
        {
            return None;
        }
        let ratio = if root_width == 0.0 {
            if leaf_width == 0.0 {
                1.0
            } else {
                return None;
            }
        } else {
            leaf_width / root_width
        };
        if !ratio.is_finite() || !(0.0..=1.0).contains(&ratio) {
            return None;
        }
        volume *= ratio;
    }
    volume.is_finite().then_some(volume)
}

fn advisory_worst_shortfall(req: &GraphInputLeafRequest<'_>) -> Option<f64> {
    if req.advisory_objective_bounds.len() != req.thresholds.len() {
        return None;
    }
    let mut worst = f64::NEG_INFINITY;
    for (&threshold, &(lower, upper)) in req.thresholds.iter().zip(req.advisory_objective_bounds) {
        if !threshold.is_finite() || !lower.is_finite() || !upper.is_finite() || lower > upper {
            return None;
        }
        worst = worst.max(f64::from(threshold) - f64::from(lower));
    }
    worst.is_finite().then_some(worst)
}

#[allow(clippy::too_many_arguments)]
fn leaf_frontier_rejection(
    depth: usize,
    normalized_volume: Option<f64>,
    worst_shortfall: Option<f64>,
    min_depth: usize,
    max_normalized_volume: f64,
    max_worst_shortfall: f64,
    admit_root_without_frontier: bool,
) -> Option<LeafFrontierRejection> {
    if admit_root_without_frontier && depth == 0 {
        return None;
    }
    if depth < min_depth && normalized_volume.is_none_or(|volume| volume > max_normalized_volume) {
        return Some(LeafFrontierRejection::DepthOrVolume);
    }
    if worst_shortfall.is_none_or(|shortfall| shortfall > max_worst_shortfall) {
        return Some(LeafFrontierRejection::ObjectiveShortfall);
    }
    None
}

/// Validate the counterfactual M24 observation without granting it authority.
///
/// The historical M17/M20 selector is reconstructed independently below. The
/// M24 receipt must describe a non-regressing strict extension of that exact
/// portfolio, and all work counters must remain within its checked plan.
fn valid_m24_measurement(
    measurement: &CganCzM24Measurement,
    m17_lower_bound: f64,
    m20_lower_bound: Option<f64>,
    m20_status: CganCzM20Status,
    authoritative_lower_bound: f64,
) -> bool {
    if !measurement.counterfactual_lower_bound.is_finite() || !authoritative_lower_bound.is_finite()
    {
        return false;
    }
    let Some((historical_lower_bound, historical_selection)) =
        reconstruct_strict_m17_m20_attribution(m17_lower_bound, m20_lower_bound, m20_status)
    else {
        return false;
    };
    if historical_lower_bound.to_bits() != authoritative_lower_bound.to_bits() {
        return false;
    }

    let exact_is_coherent = match measurement.exact_box_cut_lower_bound {
        Some(exact) => {
            exact.is_finite()
                && measurement.replay_status == ReluTailBoxCutStatus::Completed
                && measurement.search_plan.is_some()
                && measurement.exact_replays > 0
                && exact <= measurement.counterfactual_lower_bound
        }
        None => matches!(
            measurement.replay_status,
            ReluTailBoxCutStatus::AuxiliaryFallback | ReluTailBoxCutStatus::CandidateFallback
        ),
    };
    if !exact_is_coherent {
        return false;
    }
    let selection_is_coherent = match measurement.counterfactual_selection {
        ReluTailBoxCutSelection::BoxCut => {
            measurement.exact_box_cut_lower_bound.is_some_and(|exact| {
                exact.to_bits() == measurement.counterfactual_lower_bound.to_bits()
            }) && measurement.counterfactual_lower_bound > authoritative_lower_bound
        }
        selected => {
            selected == historical_selection
                && measurement.counterfactual_lower_bound.to_bits()
                    == authoritative_lower_bound.to_bits()
                && measurement
                    .exact_box_cut_lower_bound
                    .is_none_or(|exact| exact <= authoritative_lower_bound)
        }
    };
    if !selection_is_coherent {
        return false;
    }
    let replay_is_coherent = match m20_status {
        CganCzM20Status::Completed => matches!(
            measurement.replay_status,
            ReluTailBoxCutStatus::Completed | ReluTailBoxCutStatus::CandidateFallback
        ),
        CganCzM20Status::Fallback => {
            measurement.replay_status == ReluTailBoxCutStatus::AuxiliaryFallback
        }
        CganCzM20Status::NotRequested => false,
    };
    if !replay_is_coherent {
        return false;
    }
    let search_status_matches_m20 =
        if measurement.search_status == ReluTailBoxCutOptimizerStatus::AuxiliaryFallback {
            m20_status == CganCzM20Status::Fallback
        } else {
            m20_status == CganCzM20Status::Completed
        };
    if !search_status_matches_m20 {
        return false;
    }

    let counters_within_plan = if let Some(plan) = measurement.search_plan {
        let Some(max_candidates_scored) = plan.total_iterations.checked_add(plan.restarts) else {
            return false;
        };
        plan.value_dim == M24_VALUE_DIM
            && plan.alpha_dim <= M24_MAX_ALPHA_DIM
            && plan.box_variables > 0
            && plan.box_variables <= M24_MAX_BOX_VARIABLES
            && plan.total_iterations == M24_TOTAL_ITERATIONS
            && plan.restarts == M24_RESTARTS
            && plan.exact_replays == M24_EXACT_REPLAYS
            && plan.generator_nonzeros <= M24_MAX_GENERATOR_NONZEROS
            && m24_plan_search_work(plan) == Some(plan.search_work)
            && plan.search_work <= M24_MAX_SEARCH_WORK
            && measurement.iterations_completed <= plan.total_iterations
            && measurement.restarts_completed <= plan.restarts
            && measurement.candidates_scored <= max_candidates_scored
            && measurement.exact_replays <= plan.exact_replays
            && measurement.exact_replays <= measurement.candidates_scored
    } else {
        measurement.iterations_completed == 0
            && measurement.restarts_completed == 0
            && measurement.candidates_scored == 0
            && measurement.exact_replays == 0
    };
    if !counters_within_plan {
        return false;
    }
    let budget_error_is_coherent = match measurement.optional_budget_error.as_ref() {
        Some(ConstrainedZonotopeCallBudgetError::DeadlineExpired { .. }) => matches!(
            measurement.search_status,
            ReluTailBoxCutOptimizerStatus::Deadline
                | ReluTailBoxCutOptimizerStatus::AuxiliaryFallback
        ),
        Some(ConstrainedZonotopeCallBudgetError::ResourceOverflow { .. }) => matches!(
            measurement.search_status,
            ReluTailBoxCutOptimizerStatus::ResourceFallback
                | ReluTailBoxCutOptimizerStatus::AuxiliaryFallback
        ),
        Some(ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { required, limit }) => {
            required > limit
                && matches!(
                    measurement.search_status,
                    ReluTailBoxCutOptimizerStatus::ResourceFallback
                        | ReluTailBoxCutOptimizerStatus::AuxiliaryFallback
                )
        }
        None => true,
    };
    if !budget_error_is_coherent {
        return false;
    }

    match (
        measurement.search_plan,
        measurement.search_status,
        measurement.optional_budget_error.as_ref(),
    ) {
        // M20 can fail before M24 planning, with or without a firewall error.
        (None, ReluTailBoxCutOptimizerStatus::AuxiliaryFallback, _) => {
            m20_status == CganCzM20Status::Fallback
        }
        // Finite authenticated auxiliary bounds can only skip planning when
        // no endpoint is tighter.
        (None, ReluTailBoxCutOptimizerStatus::NoTighterAuxiliaryBox, None) => true,
        // The sealed cGAN dimensions fit the fixed plan ceilings, so a
        // pre-plan resource refusal must carry a firewall error.
        (None, ReluTailBoxCutOptimizerStatus::ResourceFallback, Some(_)) => true,
        // Before a plan exists, Deadline can only come from the shared gate.
        (
            None,
            ReluTailBoxCutOptimizerStatus::Deadline,
            Some(ConstrainedZonotopeCallBudgetError::DeadlineExpired { .. }),
        ) => true,
        // With this fixed 4+4 schedule, Completed means both zero-start
        // restarts, every update, and both exact replays finished.
        (Some(plan), ReluTailBoxCutOptimizerStatus::Completed, None) => {
            measurement.exact_box_cut_lower_bound.is_some()
                && measurement.iterations_completed == plan.total_iterations
                && measurement.restarts_completed == plan.restarts
                && measurement.candidates_scored == plan.total_iterations + plan.restarts
                && measurement.exact_replays == plan.exact_replays
        }
        // Post-plan resource refusal always originates in the shared
        // firewall and therefore carries its typed error.
        (
            Some(_),
            ReluTailBoxCutOptimizerStatus::ResourceFallback,
            Some(
                ConstrainedZonotopeCallBudgetError::ResourceOverflow { .. }
                | ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { .. },
            ),
        ) => true,
        // Deadline may be the candidate-only clock (None) or shared gate.
        (Some(_), ReluTailBoxCutOptimizerStatus::Deadline, None)
        | (
            Some(_),
            ReluTailBoxCutOptimizerStatus::Deadline,
            Some(ConstrainedZonotopeCallBudgetError::DeadlineExpired { .. }),
        ) => true,
        (Some(_), ReluTailBoxCutOptimizerStatus::NonFiniteCandidate, None)
        | (Some(_), ReluTailBoxCutOptimizerStatus::AllocationFallback, None) => true,
        (Some(_), ReluTailBoxCutOptimizerStatus::ExactReplayFallback, None) => {
            measurement.exact_replays > 0
        }
        // InvalidConfig and SearchDisabled are impossible under the sealed,
        // valid, nonzero policy. Every other combination is forged.
        _ => false,
    }
}

/// Recompute the core's pessimistic scalar-visit oracle independently from
/// the self-described receipt. A mere ceiling check would allow a malformed
/// producer to underreport an otherwise over-budget plan.
fn m24_plan_search_work(plan: ReluTailBoxCutOptimizerPlan) -> Option<u64> {
    let values = plan.value_dim as u128;
    let variables = plan.box_variables as u128;
    let sparse = plan.generator_nonzeros as u128;
    let alpha = plan.alpha_dim as u128;
    let score = values
        .checked_mul(3)?
        .checked_add(variables.checked_mul(2)?)?
        .checked_add(sparse.checked_mul(2)?)?
        .checked_add(alpha)?;
    let restart_startup = variables
        .checked_mul(6)?
        .checked_add(values.checked_mul(2)?)?
        .checked_add(score)?;
    let per_iteration = variables.checked_mul(6)?.checked_add(score)?;
    let work = values
        .checked_add(restart_startup.checked_mul(plan.restarts as u128)?)?
        .checked_add(per_iteration.checked_mul(plan.total_iterations as u128)?)?;
    u64::try_from(work).ok()
}

fn completed_rows_from_receipt(
    report: &CganCzLeafRowReport,
    profile: CganCzImgSz32Profile,
    deadline: Instant,
    resources: LeafResourceLimits,
) -> Option<&CganCzLeafRowBounds> {
    if report.authority != CGAN_CZ_VERDICT_AUTHORITY
        || report.authority != CganCzVerdictAuthority::DisabledPendingExactMoatReplay
        || report.profile != profile
        || report.deadline != deadline
        || report.baseline_live_bytes != resources.retained_live_bytes
        || report.max_peak_live_bytes != resources.max_peak_live_bytes
        || report.topology_work_items != AUTHENTICATED_TOPOLOGY_WORK_ITEMS
        || report.parameter_elements != expected_parameter_elements(profile)
        || report.peak_live_bytes < resources.retained_live_bytes
        || report.peak_live_bytes > resources.max_peak_live_bytes
        || report.charged_items == 0
        || report.deadline_polls == 0
    {
        return None;
    }
    let CganCzLeafRowStatus::Completed(rows) = &report.status else {
        return None;
    };
    // This authority seam is narrower than the verdict-neutral diagnostic
    // API: exact input-leaf receipts must prove that both optional M20 members
    // were attempted. `Fallback` is valid because its mandatory M17 sibling
    // remains certified and the shared call receipt accounts for the attempt.
    if rows.lower_m20_status == CganCzM20Status::NotRequested
        || rows.negated_upper_m20_status == CganCzM20Status::NotRequested
    {
        return None;
    }
    let selected_lower = select_m17_m20_lower_bound(
        rows.lower_m17_candidates.selected_lower_bound,
        rows.lower_m20_lower_bound,
        rows.lower_m20_status,
    )?;
    let selected_negated_upper = select_m17_m20_lower_bound(
        rows.negated_upper_m17_candidates.selected_lower_bound,
        rows.negated_upper_m20_lower_bound,
        rows.negated_upper_m20_status,
    )?;
    // Both authenticated rows must carry a typed M24 observation. These
    // receipts are checked for structural consistency and bounded work only;
    // their counterfactual values never participate in the authoritative
    // scalar equality checks or the strict property comparison below.
    let lower_m24 = rows.lower_m24_measurement.as_ref()?;
    let negated_upper_m24 = rows.negated_upper_m24_measurement.as_ref()?;
    if !valid_m24_measurement(
        lower_m24,
        rows.lower_m17_candidates.selected_lower_bound,
        rows.lower_m20_lower_bound,
        rows.lower_m20_status,
        selected_lower,
    ) || !valid_m24_measurement(
        negated_upper_m24,
        rows.negated_upper_m17_candidates.selected_lower_bound,
        rows.negated_upper_m20_lower_bound,
        rows.negated_upper_m20_status,
        selected_negated_upper,
    ) {
        return None;
    }
    if lower_m24.counterfactual_lower_bound > -negated_upper_m24.counterfactual_lower_bound
        || lower_m24
            .exact_box_cut_lower_bound
            .is_some_and(|lower| lower > -negated_upper_m24.counterfactual_lower_bound)
        || negated_upper_m24
            .exact_box_cut_lower_bound
            .is_some_and(|negated_upper| lower_m24.counterfactual_lower_bound > -negated_upper)
    {
        return None;
    }
    if !rows.lower_y.is_finite()
        || !rows.lower_neg_y.is_finite()
        || !rows.bn_tail_correction_upper.is_finite()
        || rows.bn_tail_correction_upper < 0.0
        || rows.lower_y > -rows.lower_neg_y
        || rows.lower_y.to_bits() != selected_lower.to_bits()
        || rows.lower_neg_y.to_bits() != selected_negated_upper.to_bits()
        || rows.lower_m17_status != rows.lower_m17_candidates.status
        || rows.negated_upper_m17_status != rows.negated_upper_m17_candidates.status
    {
        return None;
    }
    Some(rows)
}

impl CganInputLeafOracle {
    fn begin_attempt<'a>(
        &'a self,
        req: &GraphInputLeafRequest<'_>,
        lower: &[f32; LATENT_DIM],
        upper: &[f32; LATENT_DIM],
        now: Instant,
        deadline: Instant,
    ) -> Option<(Instant, LeafAttemptTranche, LeafAttemptGuard<'a>)> {
        self.telemetry.record_consultation();
        if !self.admission.valid() {
            return None;
        }

        // Compute both scheduling-only observations before taking the
        // single-flight token. The original frontier is a safe common
        // prefilter because policy validation guarantees the reserved
        // frontier is no looser. Tranche selection itself remains tied to the
        // exact ordinal inspected only after acquisition.
        let volume = normalized_leaf_volume(lower, upper, &self.root_lower, &self.root_upper);
        let shortfall = advisory_worst_shortfall(req);
        if let Some(rejection) = leaf_frontier_rejection(
            req.depth,
            volume,
            shortfall,
            self.admission.min_depth,
            self.admission.max_normalized_volume,
            self.admission.max_worst_shortfall,
            true,
        ) {
            self.telemetry
                .record_frontier_skip(LeafAttemptTranche::Primary, rejection);
            return None;
        }

        if deadline.saturating_duration_since(now) < self.admission.min_global_remaining {
            self.telemetry
                .record_skip(LeafAdmissionSkip::GlobalRemaining);
            return None;
        }
        if self.telemetry.total_wall_nanos.load(Ordering::Relaxed)
            > self.admission.latest_start_wall_nanos()
        {
            self.telemetry.record_skip(LeafAdmissionSkip::TotalWall);
            return None;
        }
        if self
            .telemetry
            .in_flight
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            self.telemetry.record_skip(LeafAdmissionSkip::Concurrent);
            return None;
        }
        // A preceding synchronous call publishes its elapsed wall time before
        // releasing `in_flight`. Re-read under that handoff so a waiter cannot
        // inherit the stale pre-lock value and over-admit the aggregate cap.
        if self.telemetry.total_wall_nanos.load(Ordering::Acquire)
            > self.admission.latest_start_wall_nanos()
        {
            self.telemetry.in_flight.store(false, Ordering::Release);
            self.telemetry.record_skip(LeafAdmissionSkip::TotalWall);
            return None;
        }

        let observed_attempts = self.telemetry.attempts.load(Ordering::Acquire);
        let tranche = if observed_attempts < self.admission.primary_max_attempts {
            LeafAttemptTranche::Primary
        } else if observed_attempts < self.admission.max_attempts {
            LeafAttemptTranche::Reserved
        } else {
            self.telemetry.in_flight.store(false, Ordering::Release);
            self.telemetry.record_skip(LeafAdmissionSkip::AttemptCap);
            return None;
        };
        if tranche == LeafAttemptTranche::Reserved {
            let rejection = if req.depth == 0 {
                Some(LeafFrontierRejection::DepthOrVolume)
            } else {
                leaf_frontier_rejection(
                    req.depth,
                    volume,
                    shortfall,
                    self.admission.reserved_min_depth,
                    self.admission.reserved_max_normalized_volume,
                    self.admission.reserved_max_worst_shortfall,
                    false,
                )
            };
            if let Some(rejection) = rejection {
                self.telemetry.in_flight.store(false, Ordering::Release);
                self.telemetry.record_frontier_skip(tranche, rejection);
                return None;
            }
        }
        let Some(attempt) = self.telemetry.reserve_attempt(observed_attempts, tranche) else {
            self.telemetry.in_flight.store(false, Ordering::Release);
            self.telemetry.record_skip(LeafAdmissionSkip::Concurrent);
            return None;
        };
        let call_deadline = now
            .checked_add(self.admission.call_budget)
            .unwrap_or(deadline)
            .min(deadline);
        let attempt_guard = LeafAttemptGuard {
            telemetry: &self.telemetry,
            started: now,
            tranche,
        };
        if attempt == 1
            || attempt.is_power_of_two()
            || attempt == self.admission.primary_max_attempts.saturating_add(1)
        {
            emit_telemetry_line(
                "progress",
                self.profile,
                self.admission,
                &self.telemetry.snapshot(),
            );
        }
        Some((call_deadline, tranche, attempt_guard))
    }

    fn emit_final_once(&self, status: &'static str) {
        if self.telemetry.final_emitted.swap(true, Ordering::AcqRel) {
            return;
        }
        let snapshot = self.telemetry.snapshot();
        emit_telemetry_line(status, self.profile, self.admission, &snapshot);
        crate::flight::note(
            "cgan_input_leaf_summary",
            if snapshot.attempts == 0 {
                crate::flight::FlightStatus::Skipped
            } else {
                crate::flight::FlightStatus::Ran
            },
            Some(format!(
                "profile={:?} status={} consultations={} attempts={} reserved_attempts={} \
                 completions={} reserved_completions={} declines={} verified={} \
                 reserved_verified={} lower_m20_wins={} upper_m20_wins={} \
                 lower_m20_fallbacks={} upper_m20_fallbacks={} best_joint_residual={:.17e} \
                 lower_m24_exact={} upper_m24_exact={} lower_m24_missing={} \
                 upper_m24_missing={} lower_m24_wins={} upper_m24_wins={} \
                 max_lower_m24_gain={:.17e} max_upper_m24_gain={:.17e} \
                 best_m24_joint_residual={:.17e} m24_only_would_verify={} \
                 depth_two_production_mode={} d2_completed_sides={} d2_invalid_sides={} \
                 d2_not_requested_sides={} \
                 d2_no_time_sides={} d2_budget_fallback_sides={} \
                 d2_transform_fallback_sides={} d2_incoherent_pairs={} \
                 lower_d2_wins={} upper_d2_wins={} max_lower_d2_gain={:.17e} \
                 max_upper_d2_gain={:.17e} best_d2_joint_residual={:.17e} \
                 d2_only_would_verify={} \
                 total_wall_ms={:.3} max_attempt_ms={:.3} reserved_total_wall_ms={:.3} \
                 reserved_max_attempt_ms={:.3} max_peak_live_bytes={}",
                self.profile,
                status,
                snapshot.consultations,
                snapshot.attempts,
                snapshot.reserved_attempts,
                snapshot.completions,
                snapshot.reserved_completions,
                snapshot
                    .attempts
                    .saturating_sub(snapshot.completions)
                    .saturating_sub(u64::from(snapshot.in_flight)),
                snapshot.verified_leaves,
                snapshot.reserved_verified_leaves,
                snapshot.lower_m20_strict_wins,
                snapshot.negated_upper_m20_strict_wins,
                snapshot.lower_m20_fallbacks,
                snapshot.negated_upper_m20_fallbacks,
                snapshot.best_joint_threshold_residual,
                snapshot.lower_m24_exact_present,
                snapshot.negated_upper_m24_exact_present,
                snapshot.lower_m24_missing,
                snapshot.negated_upper_m24_missing,
                snapshot.lower_m24_strict_wins,
                snapshot.negated_upper_m24_strict_wins,
                snapshot.max_lower_m24_signed_gain,
                snapshot.max_negated_upper_m24_signed_gain,
                snapshot.best_counterfactual_m24_joint_threshold_residual,
                snapshot.m24_only_would_verify,
                CGAN_DEPTH_TWO_PRODUCTION_MODE,
                snapshot.depth_two_completed_sides,
                snapshot.depth_two_invalid_sides,
                snapshot.depth_two_not_requested_sides,
                snapshot.depth_two_no_time_sides,
                snapshot.depth_two_budget_fallback_sides,
                snapshot.depth_two_transform_fallback_sides,
                snapshot.depth_two_incoherent_pairs,
                snapshot.lower_depth_two_strict_wins,
                snapshot.negated_upper_depth_two_strict_wins,
                snapshot.max_lower_depth_two_signed_gain,
                snapshot.max_negated_upper_depth_two_signed_gain,
                snapshot.best_counterfactual_depth_two_joint_threshold_residual,
                snapshot.depth_two_only_would_verify,
                snapshot.total_wall_nanos as f64 / 1.0e6,
                snapshot.max_attempt_wall_nanos as f64 / 1.0e6,
                snapshot.reserved_total_wall_nanos as f64 / 1.0e6,
                snapshot.reserved_max_attempt_wall_nanos as f64 / 1.0e6,
                snapshot.max_peak_live_bytes,
            )),
        );
    }

    fn solve_input_leaf_gated(&self, req: &GraphInputLeafRequest<'_>) -> GraphMipLeafVerdict {
        let Some(deadline) = req.deadline else {
            return GraphMipLeafVerdict::Undecided;
        };
        if !self.resources.valid()
            || deadline > self.authority_deadline
            || req.graph.cut_fold_scope() != self.expected_graph_scope
            || Instant::now() >= deadline
            || !request_rows_match(req, self.rows)
        {
            return GraphMipLeafVerdict::Undecided;
        }
        let Some((lower, upper)) = request_leaf_box(req, &self.root_lower, &self.root_upper) else {
            return GraphMipLeafVerdict::Undecided;
        };

        let now = Instant::now();
        let Some((call_deadline, tranche, _attempt_guard)) =
            self.begin_attempt(req, &lower, &upper, now, deadline)
        else {
            return GraphMipLeafVerdict::Undecided;
        };

        // The request's original immutable deadline anchors authority.  The
        // derived absolute call deadline can only shorten it and is sealed in
        // the receipt; it never lengthens or renews the caller's grant.
        let budget = ConstrainedZonotopeCallBudget::new(
            call_deadline,
            self.resources.retained_live_bytes,
            self.resources.max_peak_live_bytes,
        );
        let report = self.bounder.bound(req.graph, &lower, &upper, budget);
        if Instant::now() >= call_deadline {
            self.telemetry.late_results.fetch_add(1, Ordering::Relaxed);
            debug!(
                depth = req.depth,
                "cGAN input-leaf proof completed after its per-call deadline"
            );
            return GraphMipLeafVerdict::Undecided;
        }
        let Some(rows) =
            completed_rows_from_receipt(&report, self.profile, call_deadline, self.resources)
        else {
            debug!(
                depth = req.depth,
                profile = ?self.profile,
                status = ?report.status,
                "cGAN input-leaf proof declined or published an invalid receipt"
            );
            return GraphMipLeafVerdict::Undecided;
        };
        emit_m24_missing_certificate_lines(self.profile, req.depth, rows);
        if Instant::now() >= call_deadline {
            self.telemetry.late_results.fetch_add(1, Ordering::Relaxed);
            debug!(
                depth = req.depth,
                "cGAN input-leaf receipt diagnostics crossed its per-call deadline"
            );
            return GraphMipLeafVerdict::Undecided;
        }
        self.telemetry.record_completion(
            rows,
            self.rows,
            report.peak_live_bytes,
            report.max_peak_live_bytes,
            report.charged_items,
            report.deadline_polls,
            tranche,
        );

        // `lower_y` and `lower_neg_y` are independently outward-certified
        // lower bounds. Strict comparison against the authenticated f64 proof
        // thresholds proves both authored closed unsafe singleton clauses
        // impossible. The separately authenticated f32 request rows are no
        // stronger; one successful inequality is deliberately powerless.
        if !(rows.lower_y > self.rows.y_threshold && rows.lower_neg_y > self.rows.neg_y_threshold) {
            return GraphMipLeafVerdict::Undecided;
        }
        self.telemetry.record_verified(tranche);

        info!(
            depth = req.depth,
            profile = ?self.profile,
            lower_y = rows.lower_y,
            lower_neg_y = rows.lower_neg_y,
            y_threshold = self.rows.y_threshold,
            neg_y_threshold = self.rows.neg_y_threshold,
            topology_work_items = report.topology_work_items,
            parameter_elements = report.parameter_elements,
            peak_live_bytes = report.peak_live_bytes,
            charged_items = report.charged_items,
            deadline_polls = report.deadline_polls,
            "cGAN input-leaf certified both unsafe singleton clauses"
        );
        crate::flight::note(
            "cgan_input_leaf_proof",
            crate::flight::FlightStatus::Ran,
            Some(format!(
                "profile={:?} depth={} lower_y={:.17e} lower_neg_y={:.17e} \
                 peak_live_bytes={} charged_items={} deadline_polls={}",
                self.profile,
                req.depth,
                rows.lower_y,
                rows.lower_neg_y,
                report.peak_live_bytes,
                report.charged_items,
                report.deadline_polls,
            )),
        );
        GraphMipLeafVerdict::VerifiedAllRows
    }
}

impl GraphMipLeafOracle for CganInputLeafOracle {
    fn solve_input_leaf(&self, req: &GraphInputLeafRequest<'_>) -> GraphMipLeafVerdict {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.solve_input_leaf_gated(req)
        })) {
            Ok(verdict) => verdict,
            Err(_) => {
                warn!(
                    depth = req.depth,
                    "cGAN input-leaf proof panicked internally; declining fail-closed"
                );
                GraphMipLeafVerdict::Undecided
            }
        }
    }

    fn solve_leaf(&self, _req: &GraphMipLeafRequest<'_>) -> GraphMipLeafVerdict {
        GraphMipLeafVerdict::Undecided
    }
}

/// Build the exact-category, default-dark imgSz32 cGAN input-leaf oracle.
///
/// All constructor failures are ordinary declines.  The underlying authored
/// seal is replayed again on every leaf report before row bounds are consumed.
#[allow(clippy::too_many_arguments)]
pub(super) fn maybe_cgan_input_leaf_oracle(
    route: CganInputLeafRoute,
    authority_deadline: Instant,
    model_path: &Path,
    property_path: &Path,
    load_config: &OnnxLoadConfig,
    graph: &GraphNetwork,
    root_input: &BoundedTensor,
    current_spec: &VnnLibSpec,
) -> Option<CganInputLeafAttachment> {
    if !construction_deadline_live(authority_deadline, "entry") {
        return None;
    }
    if route != CganInputLeafRoute::Cgan2023
        || load_config.batch_norm_folding_policy() != BatchNormFoldingPolicy::PreserveRaw
        || !load_config.raw_float32_initializer_provenance_enabled()
        || !load_config.require_authored_float32_initializers()
    {
        warn!("cGAN input-leaf oracle declined: typed route or authored-load policy mismatch");
        return None;
    }
    let profile = match profile_from_model_path(model_path) {
        Some(profile) => profile,
        None => {
            warn!(
                model = %model_path.display(),
                "cGAN input-leaf oracle declined: model filename is not an authenticated profile"
            );
            return None;
        }
    };
    if !property_name_matches_profile(property_path, profile) {
        warn!("cGAN input-leaf oracle declined: property filename/profile mismatch");
        return None;
    }
    let source_sizes = match authored_source_sizes(profile, model_path, property_path) {
        Some(source_sizes) => source_sizes,
        None => {
            warn!(
                profile = ?profile,
                model = %model_path.display(),
                property = %property_path.display(),
                "cGAN input-leaf oracle declined: authored source metadata mismatch"
            );
            return None;
        }
    };
    if !construction_deadline_live(authority_deadline, "source-size preflight") {
        return None;
    }
    let (certified_spec, certified_input, moat) =
        match load_vnnlib_with_certified_scalar_moat(property_path) {
            Ok(certified) => certified,
            Err(error) => {
                warn!(
                    %error,
                    property = %property_path.display(),
                    "cGAN input-leaf oracle declined: certified property reload failed"
                );
                return None;
            }
        };
    if !construction_deadline_live(authority_deadline, "certified property reload") {
        return None;
    }
    if !same_parsed_property(current_spec, &certified_spec) {
        warn!("cGAN input-leaf oracle declined: live property differs from certified source parse");
        return None;
    }
    let rows = match authenticated_rows(&certified_spec, moat) {
        Some(rows) => rows,
        None => {
            warn!(
                profile = ?profile,
                num_inputs = certified_spec.num_inputs,
                num_outputs = certified_spec.num_outputs,
                is_disjunction = certified_spec.is_disjunction,
                dual_network = certified_spec.dual_network.is_some(),
                flat_constraints = ?certified_spec.output_constraints,
                clause_constraints = ?certified_spec.output_constraint_clauses,
                per_clause_input_bounds = ?certified_spec.per_clause_input_bounds,
                moat_low_upper = moat.low_upper(),
                moat_high_lower = moat.high_lower(),
                "cGAN input-leaf oracle declined: scalar moat row authentication failed"
            );
            return None;
        }
    };
    let (root_lower, root_upper) = match root_box(root_input, &certified_input, &certified_spec) {
        Some(root) => root,
        None => {
            warn!(
                profile = ?profile,
                root_values = root_input.len(),
                certified_values = certified_input.len(),
                "cGAN input-leaf oracle declined: live root box authentication failed"
            );
            return None;
        }
    };
    if !construction_deadline_live(authority_deadline, "property/root authentication") {
        return None;
    }
    let model = match load_onnx_with_config(model_path, load_config) {
        Ok(model) => model,
        Err(error) => {
            warn!(%error, "cGAN input-leaf oracle declined: authored model reload failed");
            return None;
        }
    };
    if !construction_deadline_live(authority_deadline, "authored model reload") {
        return None;
    }
    // Re-stat after both reloads. A source changed during construction cannot
    // inherit the preflight's bounded/authored-size evidence.
    if authored_source_sizes(profile, model_path, property_path) != Some(source_sizes) {
        warn!("cGAN input-leaf oracle declined: authored source sizes changed during reload");
        return None;
    }
    if !construction_deadline_live(authority_deadline, "post-reload source authentication") {
        return None;
    }
    let limits = match profile {
        CganCzImgSz32Profile::Nch1 => cgan_nch1_independent_interval_qualification_limits(),
        CganCzImgSz32Profile::Nch3 => cgan_nch3_independent_interval_qualification_limits(),
    };
    let bounder = AuthoredLeafRowBounder {
        profile,
        model,
        moat,
        limits,
    };
    info!(
        profile = ?profile,
        model_source_bytes = source_sizes.0,
        property_source_bytes = source_sizes.1,
        retained_live_bytes = RETAINED_LIVE_BYTES,
        max_peak_live_bytes = MAX_PEAK_LIVE_BYTES,
        admission_min_depth = LeafAdmissionPolicy::PRODUCTION.min_depth,
        admission_max_volume = LeafAdmissionPolicy::PRODUCTION.max_normalized_volume,
        admission_max_shortfall = LeafAdmissionPolicy::PRODUCTION.max_worst_shortfall,
        admission_primary_max_attempts = LeafAdmissionPolicy::PRODUCTION.primary_max_attempts,
        admission_reserved_min_depth = LeafAdmissionPolicy::PRODUCTION.reserved_min_depth,
        admission_reserved_max_volume = LeafAdmissionPolicy::PRODUCTION
            .reserved_max_normalized_volume,
        admission_reserved_max_shortfall = LeafAdmissionPolicy::PRODUCTION
            .reserved_max_worst_shortfall,
        admission_max_attempts = LeafAdmissionPolicy::PRODUCTION.max_attempts,
        admission_call_secs = LeafAdmissionPolicy::PRODUCTION.call_budget.as_secs_f64(),
        admission_total_wall_secs = LeafAdmissionPolicy::PRODUCTION
            .total_wall_budget
            .as_secs_f64(),
        depth_two_production_mode = CGAN_DEPTH_TWO_PRODUCTION_MODE,
        "cGAN input-leaf oracle armed (default-dark exact cgan_2023 route; depth-two replay disabled)"
    );
    Some(CganInputLeafAttachment(Arc::new(CganInputLeafOracle {
        profile,
        expected_graph_scope: graph.cut_fold_scope(),
        authority_deadline,
        root_lower,
        root_upper,
        rows,
        resources: LeafResourceLimits::PRODUCTION,
        admission: LeafAdmissionPolicy::PRODUCTION,
        telemetry: CganInputLeafTelemetry::new(),
        bounder: Arc::new(bounder),
    })))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::time::Duration;

    use ndarray::{Array1, Array2, ArrayD, IxDyn};
    use ny_mip::{
        ReluTailBoxCutOptimizerPlan, ReluTailConvBatchNormPullbackPlan, ReluTailDualStatus,
    };
    use ny_propagate::layers::LinearLayer;
    use ny_propagate::{GraphNode, Layer, NETWORK_INPUT};

    use super::super::super::cz_cgan_sequential_unwired::{
        CganCzM17CandidateTelemetry, CganCzProbeDecline,
    };
    use super::*;

    #[derive(Clone, Copy)]
    struct FakeRows {
        lower_y: f64,
        lower_neg_y: f64,
        lower_m24_exact: Option<f64>,
        negated_upper_m24_exact: Option<f64>,
        lower_depth_two: Option<f64>,
        negated_upper_depth_two: Option<f64>,
        required_peak: usize,
        receipt_peak_delta: isize,
    }

    struct FakeBounder {
        profile: CganCzImgSz32Profile,
        rows: FakeRows,
        budgets: Mutex<Vec<ConstrainedZonotopeCallBudget>>,
    }

    struct PanicBounder;

    impl LeafRowBounder for PanicBounder {
        fn bound(
            &self,
            _graph: &GraphNetwork,
            _leaf_lower: &[f32],
            _leaf_upper: &[f32],
            _budget: ConstrainedZonotopeCallBudget,
        ) -> CganCzLeafRowReport {
            panic!("simulated cGAN row-bound panic")
        }
    }

    fn telemetry(bound: f64) -> CganCzM17CandidateTelemetry {
        CganCzM17CandidateTelemetry {
            selected_lower_bound: bound,
            zero_positive_slope_lower_bound: bound,
            upper_endpoint_lower_bound: None,
            canonical_lower_bound: None,
            optimized_lower_bound: None,
            best_nonoptimized_lower_bound: bound,
            optimized_improvement: 0.0,
            optimizable_slopes: 0,
            candidates_replayed: 1,
            iterations_completed: 0,
            status: ReluTailDualStatus::SearchDisabled,
        }
    }

    #[test]
    fn user_facing_telemetry_marks_depth_two_as_production_disabled() {
        let snapshot = CganInputLeafTelemetry::new().snapshot();
        let mut output = Vec::new();
        write_telemetry_line(
            &mut output,
            "test",
            CganCzImgSz32Profile::Nch1,
            LeafAdmissionPolicy::PRODUCTION,
            &snapshot,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("depth_two_production_mode=disabled_not_requested"));
        assert!(output.contains("d2_completed_sides=0"));
        assert!(output.contains("d2_not_requested_sides=0"));
    }

    fn m24_plan() -> ReluTailBoxCutOptimizerPlan {
        let mut plan = ReluTailBoxCutOptimizerPlan {
            value_dim: M24_VALUE_DIM,
            alpha_dim: 1,
            generator_nonzeros: 1,
            box_variables: 1,
            restarts: M24_RESTARTS,
            total_iterations: M24_TOTAL_ITERATIONS,
            exact_replays: M24_EXACT_REPLAYS,
            search_work: 0,
        };
        plan.search_work = m24_plan_search_work(plan).unwrap();
        plan
    }

    fn m24_measurement(
        m17_lower_bound: f64,
        m20_lower_bound: Option<f64>,
        m20_status: CganCzM20Status,
        exact_box_cut_lower_bound: Option<f64>,
    ) -> CganCzM24Measurement {
        let authoritative_lower_bound =
            select_m17_m20_lower_bound(m17_lower_bound, m20_lower_bound, m20_status).unwrap();
        let historical_selection = if m20_lower_bound
            .is_some_and(|m20| m20_status == CganCzM20Status::Completed && m20 > m17_lower_bound)
        {
            ReluTailBoxCutSelection::Auxiliary
        } else {
            ReluTailBoxCutSelection::Original
        };
        let box_cut_wins =
            exact_box_cut_lower_bound.is_some_and(|exact| exact > authoritative_lower_bound);
        CganCzM24Measurement {
            exact_box_cut_lower_bound,
            counterfactual_lower_bound: if box_cut_wins {
                exact_box_cut_lower_bound.unwrap()
            } else {
                authoritative_lower_bound
            },
            counterfactual_selection: if box_cut_wins {
                ReluTailBoxCutSelection::BoxCut
            } else {
                historical_selection
            },
            replay_status: if exact_box_cut_lower_bound.is_some() {
                ReluTailBoxCutStatus::Completed
            } else if m20_status == CganCzM20Status::Fallback {
                ReluTailBoxCutStatus::AuxiliaryFallback
            } else {
                ReluTailBoxCutStatus::CandidateFallback
            },
            search_status: if exact_box_cut_lower_bound.is_some() {
                ReluTailBoxCutOptimizerStatus::Completed
            } else if m20_status == CganCzM20Status::Fallback {
                ReluTailBoxCutOptimizerStatus::AuxiliaryFallback
            } else {
                ReluTailBoxCutOptimizerStatus::NoTighterAuxiliaryBox
            },
            search_plan: exact_box_cut_lower_bound.map(|_| m24_plan()),
            iterations_completed: if exact_box_cut_lower_bound.is_some() {
                M24_TOTAL_ITERATIONS
            } else {
                0
            },
            restarts_completed: if exact_box_cut_lower_bound.is_some() {
                M24_RESTARTS
            } else {
                0
            },
            candidates_scored: if exact_box_cut_lower_bound.is_some() {
                M24_TOTAL_ITERATIONS + M24_RESTARTS
            } else {
                0
            },
            exact_replays: if exact_box_cut_lower_bound.is_some() {
                M24_EXACT_REPLAYS
            } else {
                0
            },
            optional_budget_error: None,
        }
    }

    fn depth_two_plan() -> ReluTailConvBatchNormPullbackPlan {
        ReluTailConvBatchNormPullbackPlan {
            input_shape: DEPTH_TWO_INPUT_SHAPE,
            output_shape: DEPTH_TWO_OUTPUT_SHAPE,
            weight_shape: DEPTH_TWO_WEIGHT_SHAPE,
            weight_elements: DEPTH_TWO_WEIGHT_ELEMENTS,
            kernel_visits: DEPTH_TWO_KERNEL_VISITS,
            pulled_margin_construction_exact_product_bound: DEPTH_TWO_EXACT_PRODUCT_BOUND,
        }
    }

    fn depth_two_measurement(
        historical_lower_bound: f64,
        upstream_m17_lower_bound: Option<f64>,
        peak_live_bytes: usize,
    ) -> CganCzDepthTwoMeasurement {
        let Some(upstream_m17_lower_bound) = upstream_m17_lower_bound else {
            return CganCzDepthTwoMeasurement::NotRequested;
        };
        depth_two_measurement_with_portfolio(
            historical_lower_bound,
            upstream_m17_lower_bound,
            Some(upstream_m17_lower_bound - 1.0),
            CganCzM20Status::Completed,
            None,
            ReluTailBoxCutSelection::Original,
            peak_live_bytes,
        )
    }

    fn depth_two_measurement_with_portfolio(
        historical_lower_bound: f64,
        upstream_m17_lower_bound: f64,
        upstream_m20_lower_bound: Option<f64>,
        upstream_m20_status: CganCzM20Status,
        upstream_m20_optional_budget_error: Option<ConstrainedZonotopeCallBudgetError>,
        upstream_m17_m20_selection: ReluTailBoxCutSelection,
        peak_live_bytes: usize,
    ) -> CganCzDepthTwoMeasurement {
        let (upstream_lower_bound, _) = reconstruct_strict_m17_m20_attribution(
            upstream_m17_lower_bound,
            upstream_m20_lower_bound,
            upstream_m20_status,
        )
        .expect("synthetic depth-two M17/M20 portfolio must be coherent");
        let counterfactual_lower_bound = if upstream_lower_bound > historical_lower_bound {
            upstream_lower_bound
        } else {
            historical_lower_bound
        };
        CganCzDepthTwoMeasurement::Completed(CganCzDepthTwoCompletedMeasurement {
            historical_lower_bound,
            downstream_m17_candidates: telemetry(historical_lower_bound),
            upstream_m17_candidates: telemetry(upstream_m17_lower_bound),
            upstream_m20_lower_bound,
            upstream_m20_status,
            upstream_m20_optional_budget_error,
            upstream_m17_m20_selection,
            counterfactual_lower_bound,
            signed_gain: upstream_lower_bound - historical_lower_bound,
            plan: depth_two_plan(),
            peak_live_bytes,
            charged_items: 1,
            deadline_polls: 1,
        })
    }

    impl LeafRowBounder for FakeBounder {
        fn bound(
            &self,
            _graph: &GraphNetwork,
            _leaf_lower: &[f32],
            _leaf_upper: &[f32],
            budget: ConstrainedZonotopeCallBudget,
        ) -> CganCzLeafRowReport {
            self.budgets.lock().unwrap().push(budget);
            let status = if self.rows.required_peak > budget.max_peak_live_bytes() {
                CganCzLeafRowStatus::Declined {
                    node: "fake",
                    reason: CganCzProbeDecline::Budget(
                        ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                            required: self.rows.required_peak,
                            limit: budget.max_peak_live_bytes(),
                        },
                    ),
                }
            } else {
                CganCzLeafRowStatus::Completed(CganCzLeafRowBounds {
                    lower_y: self.rows.lower_y,
                    lower_neg_y: self.rows.lower_neg_y,
                    bn_tail_correction_upper: 0.0,
                    lower_m17_status: ReluTailDualStatus::SearchDisabled,
                    negated_upper_m17_status: ReluTailDualStatus::SearchDisabled,
                    lower_m17_candidates: telemetry(self.rows.lower_y),
                    negated_upper_m17_candidates: telemetry(self.rows.lower_neg_y),
                    // Exercise the completed-M20 receipt path while leaving
                    // M17 as the strict portfolio winner in these oracle
                    // tests.
                    lower_m20_lower_bound: Some(self.rows.lower_y - 1.0),
                    negated_upper_m20_lower_bound: Some(self.rows.lower_neg_y - 1.0),
                    lower_m20_status: CganCzM20Status::Completed,
                    negated_upper_m20_status: CganCzM20Status::Completed,
                    lower_m24_measurement: Some(m24_measurement(
                        self.rows.lower_y,
                        Some(self.rows.lower_y - 1.0),
                        CganCzM20Status::Completed,
                        self.rows.lower_m24_exact,
                    )),
                    negated_upper_m24_measurement: Some(m24_measurement(
                        self.rows.lower_neg_y,
                        Some(self.rows.lower_neg_y - 1.0),
                        CganCzM20Status::Completed,
                        self.rows.negated_upper_m24_exact,
                    )),
                    lower_depth_two_measurement: depth_two_measurement(
                        self.rows.lower_y,
                        self.rows.lower_depth_two,
                        self.rows.required_peak,
                    ),
                    negated_upper_depth_two_measurement: depth_two_measurement(
                        self.rows.lower_neg_y,
                        self.rows.negated_upper_depth_two,
                        self.rows.required_peak,
                    ),
                })
            };
            let peak = if self.rows.receipt_peak_delta.is_negative() {
                self.rows
                    .required_peak
                    .saturating_sub(self.rows.receipt_peak_delta.unsigned_abs())
            } else {
                self.rows
                    .required_peak
                    .saturating_add(self.rows.receipt_peak_delta as usize)
            };
            CganCzLeafRowReport {
                authority: CGAN_CZ_VERDICT_AUTHORITY,
                profile: self.profile,
                deadline: budget.deadline(),
                baseline_live_bytes: budget.baseline_live_bytes(),
                max_peak_live_bytes: budget.max_peak_live_bytes(),
                status,
                topology_work_items: AUTHENTICATED_TOPOLOGY_WORK_ITEMS,
                parameter_elements: expected_parameter_elements(self.profile),
                peak_live_bytes: peak,
                charged_items: 1,
                deadline_polls: 1,
            }
        }
    }

    fn graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::new(
            "linear",
            Layer::Linear(
                LinearLayer::new(Array2::zeros((1, LATENT_DIM)), Some(Array1::zeros(1))).unwrap(),
            ),
            vec![NETWORK_INPUT.to_string()],
        ));
        graph.set_output("linear");
        graph
    }

    fn input(lower: [f32; LATENT_DIM], upper: [f32; LATENT_DIM]) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[LATENT_DIM]), lower.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[LATENT_DIM]), upper.to_vec()).unwrap(),
        )
        .unwrap()
    }

    fn oracle_at_deadline(
        graph: &GraphNetwork,
        profile: CganCzImgSz32Profile,
        rows: FakeRows,
        resources: LeafResourceLimits,
        authority_deadline: Instant,
    ) -> CganInputLeafOracle {
        CganInputLeafOracle {
            profile,
            expected_graph_scope: graph.cut_fold_scope(),
            authority_deadline,
            root_lower: [-1.0; LATENT_DIM],
            root_upper: [1.0; LATENT_DIM],
            rows: AuthenticatedRows {
                request_y_threshold: 0.25,
                request_neg_y_threshold: -0.75,
                y_threshold: 0.25,
                neg_y_threshold: -0.75,
            },
            resources,
            admission: LeafAdmissionPolicy::UNRESTRICTED_TEST,
            telemetry: CganInputLeafTelemetry::new(),
            bounder: Arc::new(FakeBounder {
                profile,
                rows,
                budgets: Mutex::new(Vec::new()),
            }),
        }
    }

    fn oracle(
        graph: &GraphNetwork,
        profile: CganCzImgSz32Profile,
        rows: FakeRows,
        resources: LeafResourceLimits,
    ) -> CganInputLeafOracle {
        oracle_at_deadline(
            graph,
            profile,
            rows,
            resources,
            Instant::now() + Duration::from_secs(10),
        )
    }

    fn request<'a>(
        graph: &'a GraphNetwork,
        input: &'a BoundedTensor,
        objectives: &'a Array2<f32>,
        thresholds: &'a [f32],
        clause_sizes: &'a [usize],
        deadline: Option<Instant>,
    ) -> GraphInputLeafRequest<'a> {
        GraphInputLeafRequest {
            graph,
            input_bounds: input,
            objectives,
            thresholds,
            advisory_objective_bounds: &[(-1.0, 1.0), (-1.0, 1.0)],
            clause_sizes,
            depth: 7,
            deadline,
        }
    }

    fn verified(verdict: GraphMipLeafVerdict) -> bool {
        matches!(verdict, GraphMipLeafVerdict::VerifiedAllRows)
    }

    fn certified_moat_source() -> String {
        let mut source = String::new();
        for index in 0..LATENT_DIM {
            source.push_str(&format!("(declare-const X_{index} Real)\n"));
        }
        source.push_str("(declare-const Y_0 Real)\n");
        for index in 0..LATENT_DIM {
            source.push_str(&format!("(assert (>= X_{index} -1.0))\n"));
            source.push_str(&format!("(assert (<= X_{index} 1.0))\n"));
        }
        // Deliberately reverse the common HIGH-then-LOW source order.
        source.push_str("(assert (or (and (<= Y_0 0.25)) (and (>= Y_0 0.75))))\n");
        source
    }

    #[test]
    fn authored_source_size_preflight_is_exact_and_one_byte_fail_closed() {
        for profile in [CganCzImgSz32Profile::Nch1, CganCzImgSz32Profile::Nch3] {
            let expected = expected_model_source_bytes(profile);
            assert!(authored_source_sizes_admitted(profile, expected, 681));
            assert!(!authored_source_sizes_admitted(profile, expected + 1, 681));
            assert!(!authored_source_sizes_admitted(
                profile,
                expected,
                MAX_PROPERTY_SOURCE_BYTES + 1
            ));
            assert!(!authored_source_sizes_admitted(profile, expected, 0));
        }
        assert!(!authored_source_sizes_admitted(
            CganCzImgSz32Profile::Nch1,
            NCH3_MODEL_SOURCE_BYTES,
            681
        ));

        // Exercise the real metadata seam with a sparse file: the rejection
        // must happen before any attempt to parse the model payload.
        let directory = tempfile::tempdir().unwrap();
        let model = directory.path().join("cGAN_imgSz32_nCh_1.onnx");
        let property = directory
            .path()
            .join("cGAN_imgSz32_nCh_1_prop_0_input_eps_0.010_output_eps_0.015.vnnlib");
        std::fs::File::create(&model)
            .unwrap()
            .set_len(NCH1_MODEL_SOURCE_BYTES + 1)
            .unwrap();
        std::fs::write(&property, b"x").unwrap();
        assert!(authored_source_sizes(CganCzImgSz32Profile::Nch1, &model, &property).is_none());
    }

    #[test]
    fn constructor_refuses_an_expired_authoritative_deadline_before_io() {
        let (spec, _, _) =
            ny_onnx::vnnlib::parse_vnnlib_with_certified_scalar_moat(&certified_moat_source())
                .unwrap();
        let (lower, upper) = spec.split_input_bounds_f32();
        let lower: [f32; LATENT_DIM] = lower.try_into().unwrap();
        let upper: [f32; LATENT_DIM] = upper.try_into().unwrap();
        let root = input(lower, upper);
        let graph = graph();
        let config = OnnxLoadConfig::default();
        assert!(maybe_cgan_input_leaf_oracle(
            CganInputLeafRoute::Cgan2023,
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .unwrap(),
            Path::new("/does/not/exist/cGAN_imgSz32_nCh_1.onnx"),
            Path::new(
                "/does/not/exist/cGAN_imgSz32_nCh_1_prop_0_input_eps_0.010_output_eps_0.015.vnnlib"
            ),
            &config,
            &graph,
            &root,
            &spec,
        )
        .is_none());
    }

    #[test]
    fn exact_decimal_moat_authenticates_both_directed_planner_rows() {
        let (spec, certified, moat) =
            ny_onnx::vnnlib::parse_vnnlib_with_certified_scalar_moat(&certified_moat_source())
                .unwrap();
        let rows = authenticated_rows(&spec, moat).expect("canonical scalar moat");
        assert_eq!(rows.request_y_threshold.to_bits(), 0.25_f32.to_bits());
        assert_eq!(
            rows.request_neg_y_threshold.to_bits(),
            (-0.75_f32).to_bits()
        );
        assert_eq!(rows.y_threshold.to_bits(), 0.25_f64.to_bits());
        assert_eq!(rows.neg_y_threshold.to_bits(), (-0.75_f64).to_bits());

        let (lower, upper) = spec.split_input_bounds_f32();
        let lower: [f32; LATENT_DIM] = lower.try_into().unwrap();
        let upper: [f32; LATENT_DIM] = upper.try_into().unwrap();
        let root = input(lower, upper);
        assert!(root_box(&root, &certified, &spec).is_some());
        let widened = input([ny_tensor::next_down_f32(lower[0]); LATENT_DIM], upper);
        assert!(root_box(&widened, &certified, &spec).is_none());

        let mut partial = spec;
        partial.output_constraint_clauses.pop();
        partial.output_constraints.pop();
        assert!(authenticated_rows(&partial, moat).is_none());
    }

    #[test]
    fn exact_decimal_moat_can_require_a_stronger_row_than_the_ordinary_planner() {
        const OFFICIAL_LOW: f64 = 0.5403130054473877;
        const OFFICIAL_HIGH: f64 = 0.570313036441803;
        let source = certified_moat_source()
            .replace("0.25", "0.5403130054473877")
            .replace("0.75", "0.570313036441803");
        let (spec, _, moat) =
            ny_onnx::vnnlib::parse_vnnlib_with_certified_scalar_moat(&source).unwrap();
        let rows = authenticated_rows(&spec, moat).expect("directed exact-decimal moat");

        assert_eq!(
            rows.request_y_threshold.to_bits(),
            ny_core::f64_to_f32_up(OFFICIAL_LOW).to_bits()
        );
        assert_eq!(
            rows.request_neg_y_threshold.to_bits(),
            (-ny_core::f64_to_f32_down(OFFICIAL_HIGH)).to_bits()
        );
        // LOW lies just above its ordinary binary64 parse, so its exact
        // outward upper endpoint is strictly stronger than the f32 request.
        assert!(rows.y_threshold > f64::from(rows.request_y_threshold));
        assert_eq!(rows.y_threshold.to_bits(), moat.low_upper().to_bits());
        // HIGH has the same decimal direction, but negating its certified
        // downward endpoint reverses the direction. The ordinary -Y request
        // is already exactly the stronger proof threshold for this property.
        assert_eq!(moat.high_lower().to_bits(), OFFICIAL_HIGH.to_bits());
        assert_eq!(
            rows.neg_y_threshold.to_bits(),
            (-moat.high_lower()).to_bits()
        );
        assert_eq!(
            rows.neg_y_threshold.to_bits(),
            f64::from(rows.request_neg_y_threshold).to_bits()
        );
    }

    #[test]
    fn exact_decimal_proof_threshold_denies_equality_after_f32_row_authentication() {
        let source = certified_moat_source()
            .replace("0.25", "0.5403130054473877")
            .replace("0.75", "0.570313036441803");
        let (spec, _, moat) =
            ny_onnx::vnnlib::parse_vnnlib_with_certified_scalar_moat(&source).unwrap();
        let rows = authenticated_rows(&spec, moat).expect("directed exact-decimal moat");
        let rounded_residual = rows.y_threshold - f64::from(rows.request_y_threshold);
        assert!(rounded_residual > 0.0);

        let graph = graph();
        let input = input([-0.5; LATENT_DIM], [0.5; LATENT_DIM]);
        let objectives = Array2::from_shape_vec((2, 1), vec![1.0, -1.0]).unwrap();
        let thresholds = [rows.request_y_threshold, rows.request_neg_y_threshold];
        let clauses = [1, 1];
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut oracle = oracle_at_deadline(
            &graph,
            CganCzImgSz32Profile::Nch1,
            FakeRows {
                // Equality is insufficient for the authored closed unsafe
                // clause, even though it is strictly above the f32 request.
                lower_y: rows.y_threshold,
                lower_neg_y: -0.56,
                lower_m24_exact: None,
                negated_upper_m24_exact: None,
                lower_depth_two: None,
                negated_upper_depth_two: None,
                required_peak: 101,
                receipt_peak_delta: 0,
            },
            LeafResourceLimits {
                retained_live_bytes: 64,
                max_peak_live_bytes: 101,
            },
            deadline,
        );
        oracle.rows = rows;
        let request = request(
            &graph,
            &input,
            &objectives,
            &thresholds,
            &clauses,
            Some(deadline),
        );

        assert!(request_rows_match(&request, rows));
        assert!(matches!(
            oracle.solve_input_leaf(&request),
            GraphMipLeafVerdict::Undecided
        ));
        let snapshot = oracle.telemetry.snapshot();
        assert_eq!(snapshot.completions, 1);
        assert_eq!(snapshot.verified_leaves, 0);
        assert_eq!(
            snapshot.best_lower_threshold_residual.to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            snapshot.best_joint_threshold_residual.to_bits(),
            0.0_f64.to_bits()
        );
    }

    #[test]
    fn exact_imgsz32_filenames_select_nch1_and_nch3_only() {
        assert_eq!(
            profile_from_model_path(Path::new("/x/cGAN_imgSz32_nCh_1.onnx")),
            Some(CganCzImgSz32Profile::Nch1)
        );
        assert_eq!(
            profile_from_model_path(Path::new("/x/cGAN_imgSz32_nCh_3.onnx")),
            Some(CganCzImgSz32Profile::Nch3)
        );
        for mismatch in [
            "cGAN_imgSz64_nCh_1.onnx",
            "cGAN_imgSz32_nCh_1_transposedConvPadding_1.onnx",
            "cGAN_imgSz32_nCh_3_nonlinear_activations.onnx",
            "cGAN_imgSz32_nCh_1.onnx.gz",
            "cgan_imgSz32_nCh_1.onnx",
        ] {
            assert_eq!(profile_from_model_path(Path::new(mismatch)), None);
        }
        assert!(property_name_matches_profile(
            Path::new("cGAN_imgSz32_nCh_1_prop_3_input_eps_0.010_output_eps_0.015.vnnlib"),
            CganCzImgSz32Profile::Nch1
        ));
        assert!(!property_name_matches_profile(
            Path::new("cGAN_imgSz32_nCh_3_prop_3_input_eps_0.015_output_eps_0.020.vnnlib"),
            CganCzImgSz32Profile::Nch1
        ));
        assert!(!property_name_matches_profile(
            Path::new("cGAN_imgSz32_nCh_1_prop_3_input_eps_0.010_output_eps_0.015.vnnlib.gz"),
            CganCzImgSz32Profile::Nch1
        ));
    }

    #[test]
    fn m20_receipt_requires_attempt_and_exact_strict_portfolio_selection() {
        let profile = CganCzImgSz32Profile::Nch1;
        let deadline = Instant::now() + Duration::from_secs(10);
        let resources = LeafResourceLimits {
            retained_live_bytes: 64,
            max_peak_live_bytes: 101,
        };
        let valid_rows = CganCzLeafRowBounds {
            lower_y: 0.6,
            lower_neg_y: -0.6,
            bn_tail_correction_upper: 0.0,
            lower_m17_status: ReluTailDualStatus::SearchDisabled,
            negated_upper_m17_status: ReluTailDualStatus::SearchDisabled,
            lower_m17_candidates: telemetry(0.4),
            negated_upper_m17_candidates: telemetry(-0.8),
            lower_m20_lower_bound: Some(0.6),
            negated_upper_m20_lower_bound: Some(-0.6),
            lower_m20_status: CganCzM20Status::Completed,
            negated_upper_m20_status: CganCzM20Status::Completed,
            lower_m24_measurement: Some(m24_measurement(
                0.4,
                Some(0.6),
                CganCzM20Status::Completed,
                None,
            )),
            negated_upper_m24_measurement: Some(m24_measurement(
                -0.8,
                Some(-0.6),
                CganCzM20Status::Completed,
                None,
            )),
            lower_depth_two_measurement: CganCzDepthTwoMeasurement::NotRequested,
            negated_upper_depth_two_measurement: CganCzDepthTwoMeasurement::NotRequested,
        };
        let valid_report = CganCzLeafRowReport {
            authority: CGAN_CZ_VERDICT_AUTHORITY,
            profile,
            deadline,
            baseline_live_bytes: resources.retained_live_bytes,
            max_peak_live_bytes: resources.max_peak_live_bytes,
            status: CganCzLeafRowStatus::Completed(valid_rows),
            topology_work_items: AUTHENTICATED_TOPOLOGY_WORK_ITEMS,
            parameter_elements: expected_parameter_elements(profile),
            peak_live_bytes: resources.max_peak_live_bytes,
            charged_items: 1,
            deadline_polls: 1,
        };
        assert!(completed_rows_from_receipt(&valid_report, profile, deadline, resources).is_some());

        // Depth-two is a shadow observation, not an authority prerequisite.
        // Even a forged huge value under a malformed plan cannot reject or
        // replace the independently reconstructed historical rows here.
        let mut malformed_depth_two_report = valid_report.clone();
        let CganCzLeafRowStatus::Completed(malformed_depth_two_rows) =
            &mut malformed_depth_two_report.status
        else {
            unreachable!();
        };
        malformed_depth_two_rows.lower_depth_two_measurement =
            depth_two_measurement(0.6, Some(1.0e100), resources.max_peak_live_bytes);
        let CganCzDepthTwoMeasurement::Completed(measurement) =
            &mut malformed_depth_two_rows.lower_depth_two_measurement
        else {
            unreachable!();
        };
        measurement.plan.weight_shape = [128, 64, 4, 4];
        assert!(completed_rows_from_receipt(
            &malformed_depth_two_report,
            profile,
            deadline,
            resources
        )
        .is_some());

        // A coherent fallback proves M20 was attempted and preserves M17.
        let mut fallback_report = valid_report.clone();
        let CganCzLeafRowStatus::Completed(fallback_rows) = &mut fallback_report.status else {
            unreachable!();
        };
        fallback_rows.lower_y = fallback_rows.lower_m17_candidates.selected_lower_bound;
        fallback_rows.lower_m20_lower_bound = None;
        fallback_rows.lower_m20_status = CganCzM20Status::Fallback;
        fallback_rows.lower_m24_measurement = Some(m24_measurement(
            fallback_rows.lower_m17_candidates.selected_lower_bound,
            None,
            CganCzM20Status::Fallback,
            None,
        ));
        assert!(
            completed_rows_from_receipt(&fallback_report, profile, deadline, resources).is_some()
        );

        // `NotRequested` is coherent for diagnostics but is never sufficient
        // at this exact-leaf authority seam.
        let mut not_attempted_report = fallback_report.clone();
        let CganCzLeafRowStatus::Completed(not_attempted_rows) = &mut not_attempted_report.status
        else {
            unreachable!();
        };
        not_attempted_rows.lower_m20_status = CganCzM20Status::NotRequested;
        assert!(
            completed_rows_from_receipt(&not_attempted_report, profile, deadline, resources)
                .is_none()
        );

        // A completed member must carry a finite certificate, and the scalar
        // row must be the exact strict-max selector output.
        let mut missing_certificate_report = valid_report.clone();
        let CganCzLeafRowStatus::Completed(missing_certificate_rows) =
            &mut missing_certificate_report.status
        else {
            unreachable!();
        };
        missing_certificate_rows.lower_m20_lower_bound = None;
        assert!(completed_rows_from_receipt(
            &missing_certificate_report,
            profile,
            deadline,
            resources
        )
        .is_none());

        let mut wrong_selection_report = valid_report.clone();
        let CganCzLeafRowStatus::Completed(wrong_selection_rows) =
            &mut wrong_selection_report.status
        else {
            unreachable!();
        };
        wrong_selection_rows.lower_m20_lower_bound = Some(0.7);
        assert!(
            completed_rows_from_receipt(&wrong_selection_report, profile, deadline, resources)
                .is_none()
        );

        let mut missing_m24_report = valid_report.clone();
        let CganCzLeafRowStatus::Completed(missing_m24_rows) = &mut missing_m24_report.status
        else {
            unreachable!();
        };
        missing_m24_rows.lower_m24_measurement = None;
        assert!(
            completed_rows_from_receipt(&missing_m24_report, profile, deadline, resources)
                .is_none()
        );

        let mut malformed_m24_plan_report = valid_report.clone();
        let CganCzLeafRowStatus::Completed(malformed_m24_plan_rows) =
            &mut malformed_m24_plan_report.status
        else {
            unreachable!();
        };
        malformed_m24_plan_rows.lower_m24_measurement = Some(m24_measurement(
            0.4,
            Some(0.6),
            CganCzM20Status::Completed,
            Some(0.5),
        ));
        assert!(completed_rows_from_receipt(
            &malformed_m24_plan_report,
            profile,
            deadline,
            resources
        )
        .is_some());
        let CganCzLeafRowStatus::Completed(malformed_m24_plan_rows) =
            &mut malformed_m24_plan_report.status
        else {
            unreachable!();
        };
        malformed_m24_plan_rows
            .lower_m24_measurement
            .as_mut()
            .unwrap()
            .search_plan
            .as_mut()
            .unwrap()
            .value_dim = M24_VALUE_DIM - 1;
        assert!(completed_rows_from_receipt(
            &malformed_m24_plan_report,
            profile,
            deadline,
            resources
        )
        .is_none());

        let mut underreported_m24_work = valid_report.clone();
        let CganCzLeafRowStatus::Completed(rows) = &mut underreported_m24_work.status else {
            unreachable!();
        };
        rows.lower_m24_measurement = Some(m24_measurement(
            0.4,
            Some(0.6),
            CganCzM20Status::Completed,
            Some(0.5),
        ));
        let plan = rows
            .lower_m24_measurement
            .as_mut()
            .unwrap()
            .search_plan
            .as_mut()
            .unwrap();
        plan.search_work -= 1;
        assert!(
            completed_rows_from_receipt(&underreported_m24_work, profile, deadline, resources)
                .is_none()
        );

        for impossible_status in [
            ReluTailBoxCutOptimizerStatus::SearchDisabled,
            ReluTailBoxCutOptimizerStatus::AuxiliaryFallback,
            ReluTailBoxCutOptimizerStatus::InvalidConfig,
            ReluTailBoxCutOptimizerStatus::Deadline,
            ReluTailBoxCutOptimizerStatus::ResourceFallback,
            ReluTailBoxCutOptimizerStatus::NonFiniteCandidate,
        ] {
            let mut forged = valid_report.clone();
            let CganCzLeafRowStatus::Completed(rows) = &mut forged.status else {
                unreachable!();
            };
            rows.lower_m24_measurement.as_mut().unwrap().search_status = impossible_status;
            assert!(
                completed_rows_from_receipt(&forged, profile, deadline, resources).is_none(),
                "planless {impossible_status:?} without a typed refusal must decline"
            );
        }

        let mut completed_without_replay = valid_report.clone();
        let CganCzLeafRowStatus::Completed(rows) = &mut completed_without_replay.status else {
            unreachable!();
        };
        rows.lower_m24_measurement = Some(m24_measurement(
            0.4,
            Some(0.6),
            CganCzM20Status::Completed,
            Some(0.5),
        ));
        let measurement = rows.lower_m24_measurement.as_mut().unwrap();
        measurement.exact_box_cut_lower_bound = None;
        measurement.replay_status = ReluTailBoxCutStatus::CandidateFallback;
        measurement.iterations_completed = 0;
        measurement.restarts_completed = 0;
        measurement.candidates_scored = 0;
        measurement.exact_replays = 0;
        assert!(completed_rows_from_receipt(
            &completed_without_replay,
            profile,
            deadline,
            resources
        )
        .is_none());

        let mut replay_failure_without_attempt = completed_without_replay.clone();
        let CganCzLeafRowStatus::Completed(rows) = &mut replay_failure_without_attempt.status
        else {
            unreachable!();
        };
        rows.lower_m24_measurement.as_mut().unwrap().search_status =
            ReluTailBoxCutOptimizerStatus::ExactReplayFallback;
        assert!(completed_rows_from_receipt(
            &replay_failure_without_attempt,
            profile,
            deadline,
            resources
        )
        .is_none());

        let mut resource_without_error = valid_report;
        let CganCzLeafRowStatus::Completed(rows) = &mut resource_without_error.status else {
            unreachable!();
        };
        rows.lower_m24_measurement = Some(m24_measurement(
            0.4,
            Some(0.6),
            CganCzM20Status::Completed,
            Some(0.5),
        ));
        rows.lower_m24_measurement.as_mut().unwrap().search_status =
            ReluTailBoxCutOptimizerStatus::ResourceFallback;
        assert!(
            completed_rows_from_receipt(&resource_without_error, profile, deadline, resources)
                .is_none()
        );

        let mut mismatched_m20_status = fallback_report.clone();
        let CganCzLeafRowStatus::Completed(rows) = &mut mismatched_m20_status.status else {
            unreachable!();
        };
        rows.lower_m24_measurement.as_mut().unwrap().search_status =
            ReluTailBoxCutOptimizerStatus::NoTighterAuxiliaryBox;
        assert!(
            completed_rows_from_receipt(&mismatched_m20_status, profile, deadline, resources)
                .is_none()
        );
    }

    #[test]
    fn m24_missing_certificate_lines_preserve_validated_per_side_causes() {
        let profile = CganCzImgSz32Profile::Nch1;
        let deadline = Instant::now() + Duration::from_secs(10);
        let resources = LeafResourceLimits {
            retained_live_bytes: 64,
            max_peak_live_bytes: 101,
        };
        let mut rows = CganCzLeafRowBounds {
            lower_y: 0.6,
            lower_neg_y: -0.6,
            bn_tail_correction_upper: 0.0,
            lower_m17_status: ReluTailDualStatus::SearchDisabled,
            negated_upper_m17_status: ReluTailDualStatus::SearchDisabled,
            lower_m17_candidates: telemetry(0.4),
            negated_upper_m17_candidates: telemetry(-0.8),
            lower_m20_lower_bound: Some(0.6),
            negated_upper_m20_lower_bound: Some(-0.6),
            lower_m20_status: CganCzM20Status::Completed,
            negated_upper_m20_status: CganCzM20Status::Completed,
            lower_m24_measurement: Some(m24_measurement(
                0.4,
                Some(0.6),
                CganCzM20Status::Completed,
                None,
            )),
            negated_upper_m24_measurement: Some(m24_measurement(
                -0.8,
                Some(-0.6),
                CganCzM20Status::Completed,
                None,
            )),
            lower_depth_two_measurement: CganCzDepthTwoMeasurement::NotRequested,
            negated_upper_depth_two_measurement: CganCzDepthTwoMeasurement::NotRequested,
        };

        let mut planless = Vec::new();
        assert_eq!(
            write_m24_missing_certificate_lines(&mut planless, profile, 11, &rows),
            2
        );
        let planless = String::from_utf8(planless).unwrap();
        assert!(planless.contains("side=lower"));
        assert!(planless.contains("side=negated_upper"));
        assert!(planless.contains("m24_search_status=NoTighterAuxiliaryBox"));
        assert!(planless.contains("m24_plan=None"));

        let lower = rows.lower_m24_measurement.as_mut().unwrap();
        lower.search_status = ReluTailBoxCutOptimizerStatus::ExactReplayFallback;
        lower.search_plan = Some(m24_plan());
        lower.iterations_completed = M24_TOTAL_ITERATIONS;
        lower.restarts_completed = M24_RESTARTS;
        lower.candidates_scored = M24_TOTAL_ITERATIONS + M24_RESTARTS;
        lower.exact_replays = 1;

        let negated_upper = rows.negated_upper_m24_measurement.as_mut().unwrap();
        negated_upper.search_status = ReluTailBoxCutOptimizerStatus::Deadline;
        negated_upper.search_plan = Some(m24_plan());
        negated_upper.iterations_completed = M24_TOTAL_ITERATIONS / 2;
        negated_upper.restarts_completed = 1;
        negated_upper.candidates_scored = M24_TOTAL_ITERATIONS / 2 + 1;
        negated_upper.optional_budget_error =
            Some(ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                checkpoint: "test M24 candidate clock",
            });

        let report = CganCzLeafRowReport {
            authority: CGAN_CZ_VERDICT_AUTHORITY,
            profile,
            deadline,
            baseline_live_bytes: resources.retained_live_bytes,
            max_peak_live_bytes: resources.max_peak_live_bytes,
            status: CganCzLeafRowStatus::Completed(rows.clone()),
            topology_work_items: AUTHENTICATED_TOPOLOGY_WORK_ITEMS,
            parameter_elements: expected_parameter_elements(profile),
            peak_live_bytes: resources.max_peak_live_bytes,
            charged_items: 1,
            deadline_polls: 1,
        };
        assert!(completed_rows_from_receipt(&report, profile, deadline, resources).is_some());

        let mut distinct = Vec::new();
        assert_eq!(
            write_m24_missing_certificate_lines(&mut distinct, profile, 12, &rows),
            2
        );
        let distinct = String::from_utf8(distinct).unwrap();
        let mut lines = distinct.lines();
        let lower = lines.next().unwrap();
        let negated_upper = lines.next().unwrap();
        assert!(lines.next().is_none());
        assert!(lower.contains("depth=12 side=lower"));
        assert!(lower.contains("m24_search_status=ExactReplayFallback"));
        assert!(lower.contains("m24_exact_replays=1"));
        assert!(lower.contains("m24_optional_budget_error=None"));
        assert!(negated_upper.contains("depth=12 side=negated_upper"));
        assert!(negated_upper.contains("m24_search_status=Deadline"));
        assert!(negated_upper.contains("DeadlineExpired"));
        assert!(negated_upper.contains("test M24 candidate clock"));

        rows.lower_m24_measurement = Some(m24_measurement(
            0.4,
            Some(0.6),
            CganCzM20Status::Completed,
            Some(0.6),
        ));
        rows.negated_upper_m24_measurement = Some(m24_measurement(
            -0.8,
            Some(-0.6),
            CganCzM20Status::Completed,
            Some(-0.6),
        ));
        let mut complete = Vec::new();
        assert_eq!(
            write_m24_missing_certificate_lines(&mut complete, profile, 13, &rows),
            0
        );
        assert!(complete.is_empty());
    }

    #[test]
    fn aggregate_telemetry_attributes_signed_m20_gains_and_joint_residual() {
        let aggregate = CganInputLeafTelemetry::new();
        let mut lower_m17 = telemetry(0.4);
        lower_m17.optimized_improvement = 0.03;
        let mut negated_upper_m17 = telemetry(-0.8);
        negated_upper_m17.optimized_improvement = 0.04;
        let rows = CganCzLeafRowBounds {
            lower_y: 0.6,
            lower_neg_y: -0.6,
            bn_tail_correction_upper: 0.0,
            lower_m17_status: ReluTailDualStatus::SearchDisabled,
            negated_upper_m17_status: ReluTailDualStatus::SearchDisabled,
            lower_m17_candidates: lower_m17,
            negated_upper_m17_candidates: negated_upper_m17,
            lower_m20_lower_bound: Some(0.6),
            negated_upper_m20_lower_bound: Some(-0.6),
            lower_m20_status: CganCzM20Status::Completed,
            negated_upper_m20_status: CganCzM20Status::Completed,
            lower_m24_measurement: Some(m24_measurement(
                0.4,
                Some(0.6),
                CganCzM20Status::Completed,
                None,
            )),
            negated_upper_m24_measurement: Some(m24_measurement(
                -0.8,
                Some(-0.6),
                CganCzM20Status::Completed,
                None,
            )),
            lower_depth_two_measurement: CganCzDepthTwoMeasurement::NotRequested,
            negated_upper_depth_two_measurement: CganCzDepthTwoMeasurement::NotRequested,
        };
        aggregate.record_completion(
            &rows,
            AuthenticatedRows {
                request_y_threshold: 0.25,
                request_neg_y_threshold: -0.75,
                y_threshold: 0.25,
                neg_y_threshold: -0.75,
            },
            123,
            123,
            1,
            1,
            LeafAttemptTranche::Reserved,
        );
        aggregate.record_verified(LeafAttemptTranche::Reserved);

        let snapshot = aggregate.snapshot();
        assert_eq!(snapshot.completions, 1);
        assert_eq!(snapshot.reserved_completions, 1);
        assert_eq!(snapshot.verified_leaves, 1);
        assert_eq!(snapshot.reserved_verified_leaves, 1);
        assert_eq!(snapshot.lower_m20_strict_wins, 1);
        assert_eq!(snapshot.negated_upper_m20_strict_wins, 1);
        assert_eq!(snapshot.lower_m20_fallbacks, 0);
        assert_eq!(snapshot.negated_upper_m20_fallbacks, 0);
        assert!((snapshot.max_lower_signed_gain - 0.2).abs() < 1.0e-15);
        assert!((snapshot.max_negated_upper_signed_gain - 0.2).abs() < 1.0e-15);
        assert!((snapshot.max_lower_m17_optimized_improvement - 0.03).abs() < 1.0e-15);
        assert!((snapshot.max_negated_upper_m17_optimized_improvement - 0.04).abs() < 1.0e-15);
        assert!((snapshot.best_lower_threshold_residual - 0.35).abs() < 1.0e-15);
        assert!((snapshot.best_negated_upper_threshold_residual - 0.15).abs() < 1.0e-15);
        assert!((snapshot.best_joint_threshold_residual - 0.15).abs() < 1.0e-15);
        assert_eq!(snapshot.lower_m24_exact_present, 0);
        assert_eq!(snapshot.negated_upper_m24_exact_present, 0);
        assert_eq!(snapshot.lower_m24_missing, 1);
        assert_eq!(snapshot.negated_upper_m24_missing, 1);
        assert_eq!(snapshot.lower_m24_strict_wins, 0);
        assert_eq!(snapshot.negated_upper_m24_strict_wins, 0);
        assert!((snapshot.best_counterfactual_m24_joint_threshold_residual - 0.15).abs() < 1.0e-15);
        assert_eq!(snapshot.m24_only_would_verify, 0);
        assert_eq!(snapshot.depth_two_not_requested_sides, 2);
        assert_eq!(snapshot.depth_two_completed_sides, 0);
        assert_eq!(snapshot.max_peak_live_bytes, 123);
    }

    #[test]
    fn aggregate_telemetry_does_not_fabricate_m20_gain_on_fallback_or_tie() {
        let aggregate = CganInputLeafTelemetry::new();
        let rows = CganCzLeafRowBounds {
            lower_y: -0.0,
            lower_neg_y: 0.5,
            bn_tail_correction_upper: 0.0,
            lower_m17_status: ReluTailDualStatus::SearchDisabled,
            negated_upper_m17_status: ReluTailDualStatus::SearchDisabled,
            lower_m17_candidates: telemetry(-0.0),
            negated_upper_m17_candidates: telemetry(0.5),
            lower_m20_lower_bound: Some(0.0),
            negated_upper_m20_lower_bound: None,
            lower_m20_status: CganCzM20Status::Completed,
            negated_upper_m20_status: CganCzM20Status::Fallback,
            lower_m24_measurement: Some(m24_measurement(
                -0.0,
                Some(0.0),
                CganCzM20Status::Completed,
                None,
            )),
            negated_upper_m24_measurement: Some(m24_measurement(
                0.5,
                None,
                CganCzM20Status::Fallback,
                None,
            )),
            lower_depth_two_measurement: CganCzDepthTwoMeasurement::NoTime,
            negated_upper_depth_two_measurement: CganCzDepthTwoMeasurement::BudgetFallback(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "synthetic depth-two deadline",
                },
            ),
        };
        aggregate.record_completion(
            &rows,
            AuthenticatedRows {
                request_y_threshold: -1.0,
                request_neg_y_threshold: -1.0,
                y_threshold: -1.0,
                neg_y_threshold: -1.0,
            },
            7,
            7,
            1,
            1,
            LeafAttemptTranche::Primary,
        );

        let snapshot = aggregate.snapshot();
        assert_eq!(snapshot.reserved_completions, 0);
        assert_eq!(snapshot.lower_m20_strict_wins, 0);
        assert_eq!(snapshot.negated_upper_m20_strict_wins, 0);
        assert_eq!(snapshot.negated_upper_m20_fallbacks, 1);
        assert_eq!(snapshot.max_lower_signed_gain.to_bits(), 0.0_f64.to_bits());
        assert_eq!(snapshot.max_negated_upper_signed_gain, f64::NEG_INFINITY);
        assert_eq!(snapshot.depth_two_no_time_sides, 1);
        assert_eq!(snapshot.depth_two_budget_fallback_sides, 1);
        assert_eq!(snapshot.depth_two_completed_sides, 0);
    }

    #[test]
    fn depth_two_candidate_summary_validation_reconstructs_zero_predicate_attribution() {
        let mut candidate = telemetry(0.4);
        candidate.zero_positive_slope_lower_bound = 0.2;
        candidate.upper_endpoint_lower_bound = Some(0.3);
        candidate.canonical_lower_bound = Some(0.25);
        candidate.optimized_lower_bound = Some(0.35);
        candidate.best_nonoptimized_lower_bound = 0.3;
        candidate.optimized_improvement = (0.35_f64 - 0.3).max(0.0);
        candidate.candidates_replayed = 4;
        assert!(valid_depth_two_m17_candidates(&candidate));

        let mut forged_best = candidate;
        forged_best.best_nonoptimized_lower_bound = 0.25;
        assert!(!valid_depth_two_m17_candidates(&forged_best));

        let mut forged_improvement = candidate;
        forged_improvement.optimized_improvement -= 0.01;
        assert!(!valid_depth_two_m17_candidates(&forged_improvement));

        let mut regressing_selected = candidate;
        regressing_selected.selected_lower_bound = 0.34;
        assert!(!valid_depth_two_m17_candidates(&regressing_selected));
    }

    #[test]
    fn depth_two_upstream_m20_selection_is_reconstructed_and_forgery_is_rejected() {
        let selected_m20 = depth_two_measurement_with_portfolio(
            0.2,
            0.25,
            Some(0.3),
            CganCzM20Status::Completed,
            None,
            ReluTailBoxCutSelection::Auxiliary,
            101,
        );
        let CganCzDepthTwoMeasurement::Completed(selected_m20) = selected_m20 else {
            unreachable!();
        };
        assert_eq!(
            valid_depth_two_completed_measurement(&selected_m20, 0.2, 101, 101, 1, 1),
            Some(ValidatedDepthTwoMeasurement {
                counterfactual_lower_bound: 0.3,
                signed_gain: 0.3 - 0.2,
            })
        );

        let mut forged_selection = selected_m20.clone();
        forged_selection.upstream_m17_m20_selection = ReluTailBoxCutSelection::Original;
        assert_eq!(
            valid_depth_two_completed_measurement(&forged_selection, 0.2, 101, 101, 1, 1),
            None
        );

        let mut completed_with_error = selected_m20.clone();
        completed_with_error.upstream_m20_optional_budget_error =
            Some(ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                checkpoint: "forged completed M20 refusal",
            });
        assert_eq!(
            valid_depth_two_completed_measurement(&completed_with_error, 0.2, 101, 101, 1, 1),
            None
        );

        let mut completed_without_m20 = selected_m20;
        completed_without_m20.upstream_m20_lower_bound = None;
        assert_eq!(
            valid_depth_two_completed_measurement(&completed_without_m20, 0.2, 101, 101, 1, 1),
            None
        );
        forged_selection.upstream_m17_m20_selection = ReluTailBoxCutSelection::BoxCut;
        assert_eq!(
            valid_depth_two_completed_measurement(&forged_selection, 0.2, 101, 101, 1, 1),
            None
        );

        let tied = depth_two_measurement_with_portfolio(
            0.2,
            0.3,
            Some(0.3),
            CganCzM20Status::Completed,
            None,
            ReluTailBoxCutSelection::Original,
            101,
        );
        let CganCzDepthTwoMeasurement::Completed(mut tied) = tied else {
            unreachable!();
        };
        assert!(valid_depth_two_completed_measurement(&tied, 0.2, 101, 101, 1, 1).is_some());
        tied.upstream_m17_m20_selection = ReluTailBoxCutSelection::Auxiliary;
        assert_eq!(
            valid_depth_two_completed_measurement(&tied, 0.2, 101, 101, 1, 1),
            None
        );

        let fallback = depth_two_measurement_with_portfolio(
            0.2,
            0.3,
            None,
            CganCzM20Status::Fallback,
            None,
            ReluTailBoxCutSelection::Original,
            101,
        );
        let CganCzDepthTwoMeasurement::Completed(fallback) = fallback else {
            unreachable!();
        };
        assert!(valid_depth_two_completed_measurement(&fallback, 0.2, 101, 101, 1, 1).is_some());

        let mut coherent_deadline_fallback = fallback.clone();
        coherent_deadline_fallback.upstream_m20_optional_budget_error =
            Some(ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                checkpoint: "synthetic optional M20 deadline",
            });
        assert!(valid_depth_two_completed_measurement(
            &coherent_deadline_fallback,
            0.2,
            101,
            101,
            1,
            1,
        )
        .is_some());

        let mut fallback_with_bound = fallback.clone();
        fallback_with_bound.upstream_m20_lower_bound = Some(0.31);
        assert_eq!(
            valid_depth_two_completed_measurement(&fallback_with_bound, 0.2, 101, 101, 1, 1),
            None
        );

        let mut not_requested = fallback.clone();
        not_requested.upstream_m20_status = CganCzM20Status::NotRequested;
        assert_eq!(
            valid_depth_two_completed_measurement(&not_requested, 0.2, 101, 101, 1, 1),
            None
        );

        let mut coherent_overflow_fallback = fallback.clone();
        coherent_overflow_fallback.upstream_m20_optional_budget_error =
            Some(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "synthetic optional M20 accounting",
            });
        assert!(valid_depth_two_completed_measurement(
            &coherent_overflow_fallback,
            0.2,
            101,
            101,
            1,
            1,
        )
        .is_some());

        let mut coherent_peak_fallback = fallback.clone();
        coherent_peak_fallback.upstream_m20_optional_budget_error =
            Some(ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                required: 102,
                limit: 101,
            });
        assert!(valid_depth_two_completed_measurement(
            &coherent_peak_fallback,
            0.2,
            101,
            101,
            1,
            1,
        )
        .is_some());

        let mut mismatched_peak_cap = coherent_peak_fallback;
        mismatched_peak_cap.upstream_m20_optional_budget_error =
            Some(ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                required: 102,
                limit: 100,
            });
        assert_eq!(
            valid_depth_two_completed_measurement(&mismatched_peak_cap, 0.2, 101, 101, 1, 1),
            None
        );

        let mut forged_peak_fallback = fallback;
        forged_peak_fallback.upstream_m20_optional_budget_error =
            Some(ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                required: 101,
                limit: 101,
            });
        assert_eq!(
            valid_depth_two_completed_measurement(&forged_peak_fallback, 0.2, 101, 101, 1, 1),
            None
        );
    }

    #[test]
    fn depth_two_outer_budget_fallback_binds_peak_limit_to_authenticated_cap() {
        let aggregate = CganInputLeafTelemetry::new();
        let deadline = CganCzDepthTwoMeasurement::BudgetFallback(
            ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                checkpoint: "synthetic depth-two deadline",
            },
        );
        assert_eq!(
            aggregate.record_depth_two_side(&deadline, 0.2, 90, 101, 1, 1),
            None
        );

        let coherent_peak = CganCzDepthTwoMeasurement::BudgetFallback(
            ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                required: 102,
                limit: 101,
            },
        );
        assert_eq!(
            aggregate.record_depth_two_side(&coherent_peak, 0.2, 90, 101, 1, 1),
            None
        );

        let mismatched_peak_cap = CganCzDepthTwoMeasurement::BudgetFallback(
            ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                required: 102,
                limit: 100,
            },
        );
        assert_eq!(
            aggregate.record_depth_two_side(&mismatched_peak_cap, 0.2, 90, 101, 1, 1),
            None
        );

        let snapshot = aggregate.snapshot();
        assert_eq!(snapshot.depth_two_budget_fallback_sides, 2);
        assert_eq!(snapshot.depth_two_invalid_sides, 1);
    }

    #[test]
    fn depth_two_telemetry_validates_receipts_and_tracks_only_coherent_shadow_gains() {
        let rows = CganCzLeafRowBounds {
            lower_y: 0.2,
            lower_neg_y: -0.8,
            bn_tail_correction_upper: 0.0,
            lower_m17_status: ReluTailDualStatus::SearchDisabled,
            negated_upper_m17_status: ReluTailDualStatus::SearchDisabled,
            lower_m17_candidates: telemetry(0.2),
            negated_upper_m17_candidates: telemetry(-0.8),
            lower_m20_lower_bound: Some(0.1),
            negated_upper_m20_lower_bound: Some(-0.9),
            lower_m20_status: CganCzM20Status::Completed,
            negated_upper_m20_status: CganCzM20Status::Completed,
            lower_m24_measurement: Some(m24_measurement(
                0.2,
                Some(0.1),
                CganCzM20Status::Completed,
                None,
            )),
            negated_upper_m24_measurement: Some(m24_measurement(
                -0.8,
                Some(-0.9),
                CganCzM20Status::Completed,
                None,
            )),
            lower_depth_two_measurement: depth_two_measurement(0.2, Some(0.3), 101),
            negated_upper_depth_two_measurement: depth_two_measurement(-0.8, Some(-0.7), 101),
        };
        let thresholds = AuthenticatedRows {
            request_y_threshold: 0.25,
            request_neg_y_threshold: -0.75,
            y_threshold: 0.25,
            neg_y_threshold: -0.75,
        };
        let aggregate = CganInputLeafTelemetry::new();
        aggregate.record_completion(
            &rows,
            thresholds,
            101,
            101,
            1,
            1,
            LeafAttemptTranche::Primary,
        );
        let snapshot = aggregate.snapshot();
        assert_eq!(snapshot.depth_two_completed_sides, 2);
        assert_eq!(snapshot.depth_two_invalid_sides, 0);
        assert_eq!(snapshot.depth_two_incoherent_pairs, 0);
        assert_eq!(snapshot.lower_depth_two_strict_wins, 1);
        assert_eq!(snapshot.negated_upper_depth_two_strict_wins, 1);
        assert!((snapshot.max_lower_depth_two_signed_gain - 0.1).abs() < 1.0e-15);
        assert!((snapshot.max_negated_upper_depth_two_signed_gain - 0.1).abs() < 1.0e-15);
        assert!(
            (snapshot.best_counterfactual_depth_two_joint_threshold_residual - 0.05).abs()
                < 1.0e-15
        );
        assert_eq!(snapshot.depth_two_only_would_verify, 1);

        let mut malformed_plan = rows.clone();
        let CganCzDepthTwoMeasurement::Completed(measurement) =
            &mut malformed_plan.lower_depth_two_measurement
        else {
            unreachable!();
        };
        measurement.plan.weight_shape = [128, 64, 4, 4];
        malformed_plan.negated_upper_depth_two_measurement =
            CganCzDepthTwoMeasurement::NotRequested;

        let mut malformed_candidates = rows.clone();
        let CganCzDepthTwoMeasurement::Completed(measurement) =
            &mut malformed_candidates.lower_depth_two_measurement
        else {
            unreachable!();
        };
        measurement
            .upstream_m17_candidates
            .best_nonoptimized_lower_bound = 0.25;
        malformed_candidates.negated_upper_depth_two_measurement =
            CganCzDepthTwoMeasurement::NotRequested;

        let mut transform_fallback = rows;
        transform_fallback.lower_depth_two_measurement =
            CganCzDepthTwoMeasurement::TransformFallback(CganCzDepthTwoTransformFailure::BatchNorm);
        transform_fallback.negated_upper_depth_two_measurement =
            CganCzDepthTwoMeasurement::NotRequested;

        let malformed_aggregate = CganInputLeafTelemetry::new();
        malformed_aggregate.record_completion(
            &malformed_plan,
            thresholds,
            101,
            101,
            1,
            1,
            LeafAttemptTranche::Primary,
        );
        malformed_aggregate.record_completion(
            &malformed_candidates,
            thresholds,
            101,
            101,
            1,
            1,
            LeafAttemptTranche::Primary,
        );
        malformed_aggregate.record_completion(
            &transform_fallback,
            thresholds,
            101,
            101,
            1,
            1,
            LeafAttemptTranche::Primary,
        );
        let malformed_snapshot = malformed_aggregate.snapshot();
        assert_eq!(malformed_snapshot.depth_two_completed_sides, 0);
        assert_eq!(malformed_snapshot.depth_two_invalid_sides, 2);
        assert_eq!(malformed_snapshot.depth_two_not_requested_sides, 3);
        assert_eq!(malformed_snapshot.depth_two_transform_fallback_sides, 1);
        assert_eq!(malformed_snapshot.lower_depth_two_strict_wins, 0);
        assert_eq!(
            malformed_snapshot.max_lower_depth_two_signed_gain,
            f64::NEG_INFINITY
        );
        assert_eq!(malformed_snapshot.depth_two_only_would_verify, 0);
    }

    #[test]
    fn production_admission_geometry_and_advisory_gap_are_scheduling_only() {
        assert!(LeafAdmissionPolicy::PRODUCTION.valid());
        assert_eq!(
            LeafAdmissionPolicy::PRODUCTION.primary_max_attempts,
            LEAF_PRIMARY_MAX_ATTEMPTS
        );
        assert_eq!(LEAF_PRIMARY_MAX_ATTEMPTS, 16);
        assert_eq!(LeafAdmissionPolicy::PRODUCTION.max_attempts, 32);
        assert_eq!(
            LeafAdmissionPolicy::PRODUCTION.reserved_min_depth,
            LEAF_RESERVED_MIN_DEPTH
        );
        assert_eq!(LEAF_RESERVED_MIN_DEPTH, 12);
        assert_eq!(
            LeafAdmissionPolicy::PRODUCTION
                .reserved_max_normalized_volume
                .to_bits(),
            (1.0_f64 / 4096.0).to_bits()
        );
        assert_eq!(
            LeafAdmissionPolicy::PRODUCTION
                .reserved_max_worst_shortfall
                .to_bits(),
            0.001_f64.to_bits()
        );
        let root_lower = [-1.0; LATENT_DIM];
        let root_upper = [1.0; LATENT_DIM];
        assert_eq!(
            normalized_leaf_volume(
                &[-0.5; LATENT_DIM],
                &[0.5; LATENT_DIM],
                &root_lower,
                &root_upper,
            ),
            Some(LEAF_MAX_NORMALIZED_VOLUME)
        );
        let reserved_lower = [-0.125, -0.125, -0.125, -0.125, -1.0];
        let reserved_upper = [0.125, 0.125, 0.125, 0.125, 1.0];
        assert_eq!(
            normalized_leaf_volume(&reserved_lower, &reserved_upper, &root_lower, &root_upper,),
            Some(LEAF_RESERVED_MAX_NORMALIZED_VOLUME)
        );
        assert_eq!(
            leaf_frontier_rejection(
                LEAF_RESERVED_MIN_DEPTH - 1,
                Some(LEAF_RESERVED_MAX_NORMALIZED_VOLUME),
                Some(LEAF_RESERVED_MAX_WORST_SHORTFALL),
                LEAF_RESERVED_MIN_DEPTH,
                LEAF_RESERVED_MAX_NORMALIZED_VOLUME,
                LEAF_RESERVED_MAX_WORST_SHORTFALL,
                false,
            ),
            None,
            "the exact compactness and advisory boundaries are admitted"
        );
        assert_eq!(
            leaf_frontier_rejection(
                LEAF_RESERVED_MIN_DEPTH - 1,
                Some(LEAF_RESERVED_MAX_NORMALIZED_VOLUME + f64::EPSILON),
                Some(LEAF_RESERVED_MAX_WORST_SHORTFALL),
                LEAF_RESERVED_MIN_DEPTH,
                LEAF_RESERVED_MAX_NORMALIZED_VOLUME,
                LEAF_RESERVED_MAX_WORST_SHORTFALL,
                false,
            ),
            Some(LeafFrontierRejection::DepthOrVolume)
        );
        assert_eq!(
            leaf_frontier_rejection(
                LEAF_RESERVED_MIN_DEPTH,
                Some(1.0),
                Some(LEAF_RESERVED_MAX_WORST_SHORTFALL),
                LEAF_RESERVED_MIN_DEPTH,
                LEAF_RESERVED_MAX_NORMALIZED_VOLUME,
                LEAF_RESERVED_MAX_WORST_SHORTFALL,
                false,
            ),
            None,
            "depth alone satisfies the reserved geometry disjunction"
        );
        assert_eq!(
            leaf_frontier_rejection(
                LEAF_RESERVED_MIN_DEPTH,
                Some(1.0),
                Some(LEAF_RESERVED_MAX_WORST_SHORTFALL + f64::EPSILON),
                LEAF_RESERVED_MIN_DEPTH,
                LEAF_RESERVED_MAX_NORMALIZED_VOLUME,
                LEAF_RESERVED_MAX_WORST_SHORTFALL,
                false,
            ),
            Some(LeafFrontierRejection::ObjectiveShortfall)
        );

        let graph = graph();
        let input = input([-0.5; LATENT_DIM], [0.5; LATENT_DIM]);
        let objectives = Array2::from_shape_vec((2, 1), vec![-1.0, 1.0]).unwrap();
        let thresholds = [-0.75, 0.25];
        let advisory = [(-0.80, 1.0), (0.20, 1.0)];
        let request = GraphInputLeafRequest {
            graph: &graph,
            input_bounds: &input,
            objectives: &objectives,
            advisory_objective_bounds: &advisory,
            thresholds: &thresholds,
            clause_sizes: &[1, 1],
            depth: LEAF_MIN_DEPTH,
            deadline: None,
        };
        let shortfall = advisory_worst_shortfall(&request).unwrap();
        assert!((shortfall - 0.05).abs() < 1.0e-6);

        // These carried bounds are deliberately optimistic and are never
        // consumed by `completed_rows_from_receipt` or the strict proof test.
        let optimistic = [(1000.0, 1001.0), (1000.0, 1001.0)];
        let optimistic_request = GraphInputLeafRequest {
            advisory_objective_bounds: &optimistic,
            ..request
        };
        assert!(advisory_worst_shortfall(&optimistic_request).unwrap() < 0.0);

        let invalid = [(0.0, f32::INFINITY), (1.0, 0.0)];
        let invalid_request = GraphInputLeafRequest {
            advisory_objective_bounds: &invalid,
            ..optimistic_request
        };
        assert!(advisory_worst_shortfall(&invalid_request).is_none());
    }

    #[test]
    fn reserved_tranche_preserves_ordinals_and_denies_even_a_tiny_root() {
        let graph = graph();
        let wide_input = input([-0.75; LATENT_DIM], [0.75; LATENT_DIM]);
        let wide_lower = [-0.75; LATENT_DIM];
        let wide_upper = [0.75; LATENT_DIM];
        let compact_lower = [-0.125, -0.125, -0.125, -0.125, -1.0];
        let compact_upper = [0.125, 0.125, 0.125, 0.125, 1.0];
        let compact_input = input(compact_lower, compact_upper);
        let objectives = Array2::from_shape_vec((2, 1), vec![-1.0, 1.0]).unwrap();
        let thresholds = [-0.75, 0.25];
        let primary_only_advisory = [(-0.80, 1.0), (0.20, 1.0)];
        let reserved_advisory = [(-0.7505, 1.0), (0.2495, 1.0)];
        let clauses = [1, 1];
        let authority_deadline = Instant::now() + Duration::from_mins(5);
        let make_oracle = || {
            let mut oracle = oracle_at_deadline(
                &graph,
                CganCzImgSz32Profile::Nch1,
                FakeRows {
                    lower_y: 0.5,
                    lower_neg_y: -0.5,
                    lower_m24_exact: None,
                    negated_upper_m24_exact: None,
                    lower_depth_two: None,
                    negated_upper_depth_two: None,
                    required_peak: 101,
                    receipt_peak_delta: 0,
                },
                LeafResourceLimits {
                    retained_live_bytes: 64,
                    max_peak_live_bytes: 101,
                },
                authority_deadline,
            );
            oracle.admission = LeafAdmissionPolicy::PRODUCTION;
            oracle
        };
        let mut request = GraphInputLeafRequest {
            graph: &graph,
            input_bounds: &wide_input,
            objectives: &objectives,
            advisory_objective_bounds: &primary_only_advisory,
            thresholds: &thresholds,
            clause_sizes: &clauses,
            depth: LEAF_RESERVED_MIN_DEPTH,
            deadline: Some(authority_deadline),
        };
        let now = Instant::now();

        let oracle = make_oracle();
        oracle.telemetry.attempts.store(15, Ordering::Relaxed);
        let (call_deadline, tranche, guard) = oracle
            .begin_attempt(&request, &wide_lower, &wide_upper, now, authority_deadline)
            .expect("ordinal 16 retains the primary frontier");
        assert_eq!(call_deadline, now + LEAF_CALL_BUDGET);
        assert_eq!(tranche, LeafAttemptTranche::Primary);
        drop(guard);
        let snapshot = oracle.telemetry.snapshot();
        assert_eq!(snapshot.attempts, 16);
        assert_eq!(snapshot.reserved_attempts, 0);

        assert!(oracle
            .begin_attempt(&request, &wide_lower, &wide_upper, now, authority_deadline,)
            .is_none());
        let snapshot = oracle.telemetry.snapshot();
        assert_eq!(snapshot.attempts, 16);
        assert_eq!(snapshot.objective_shortfall_skips, 1);
        assert_eq!(snapshot.reserved_objective_shortfall_skips, 1);

        request.advisory_objective_bounds = &reserved_advisory;
        let (_, tranche, guard) = oracle
            .begin_attempt(&request, &wide_lower, &wide_upper, now, authority_deadline)
            .expect("ordinal 17 admits a depth-qualified near-closed leaf");
        assert_eq!(tranche, LeafAttemptTranche::Reserved);
        drop(guard);
        let snapshot = oracle.telemetry.snapshot();
        assert_eq!(snapshot.attempts, 17);
        assert_eq!(snapshot.reserved_attempts, 1);
        assert!(snapshot.reserved_attempts <= snapshot.attempts);
        assert!(snapshot.reserved_completions <= snapshot.completions);
        assert!(snapshot.reserved_verified_leaves <= snapshot.verified_leaves);
        assert!(snapshot.reserved_total_wall_nanos <= snapshot.total_wall_nanos);
        assert!(snapshot.reserved_max_attempt_wall_nanos <= snapshot.max_attempt_wall_nanos);

        let root_oracle = make_oracle();
        root_oracle
            .telemetry
            .attempts
            .store(LEAF_PRIMARY_MAX_ATTEMPTS, Ordering::Relaxed);
        let root_request = GraphInputLeafRequest {
            graph: &graph,
            input_bounds: &compact_input,
            objectives: &objectives,
            advisory_objective_bounds: &reserved_advisory,
            thresholds: &thresholds,
            clause_sizes: &clauses,
            depth: 0,
            deadline: Some(authority_deadline),
        };
        assert!(root_oracle
            .begin_attempt(
                &root_request,
                &compact_lower,
                &compact_upper,
                now,
                authority_deadline,
            )
            .is_none());
        let snapshot = root_oracle.telemetry.snapshot();
        assert_eq!(snapshot.attempts, LEAF_PRIMARY_MAX_ATTEMPTS);
        assert_eq!(snapshot.reserved_attempts, 0);
        assert_eq!(snapshot.depth_or_volume_skips, 1);
        assert_eq!(snapshot.reserved_depth_or_volume_skips, 1);
        assert!(!snapshot.in_flight);

        let cap_oracle = make_oracle();
        cap_oracle.telemetry.attempts.store(31, Ordering::Relaxed);
        cap_oracle
            .telemetry
            .reserved_attempts
            .store(15, Ordering::Relaxed);
        let compact_request = GraphInputLeafRequest {
            depth: LEAF_RESERVED_MIN_DEPTH - 1,
            ..root_request
        };
        let (_, tranche, guard) = cap_oracle
            .begin_attempt(
                &compact_request,
                &compact_lower,
                &compact_upper,
                now,
                authority_deadline,
            )
            .expect("ordinal 32 admits at the exact compactness boundary");
        assert_eq!(tranche, LeafAttemptTranche::Reserved);
        drop(guard);
        assert!(cap_oracle
            .begin_attempt(
                &compact_request,
                &compact_lower,
                &compact_upper,
                now,
                authority_deadline,
            )
            .is_none());
        let snapshot = cap_oracle.telemetry.snapshot();
        assert_eq!(snapshot.attempts, LEAF_MAX_ATTEMPTS);
        assert_eq!(snapshot.reserved_attempts, 16);
        assert_eq!(snapshot.attempt_cap_skips, 1);
    }

    #[test]
    fn production_admission_enforces_frontier_deadline_attempt_and_wall_caps() {
        let graph = graph();
        let wide_input = input([-0.75; LATENT_DIM], [0.75; LATENT_DIM]);
        let lower = [-0.75; LATENT_DIM];
        let upper = [0.75; LATENT_DIM];
        let objectives = Array2::from_shape_vec((2, 1), vec![-1.0, 1.0]).unwrap();
        let thresholds = [-0.75, 0.25];
        let near = [(-0.80, 1.0), (0.20, 1.0)];
        let far = [(-1.0, 1.0), (-1.0, 1.0)];
        let clauses = [1, 1];
        let authority_deadline = Instant::now() + Duration::from_mins(5);
        let mut oracle = oracle_at_deadline(
            &graph,
            CganCzImgSz32Profile::Nch1,
            FakeRows {
                lower_y: 0.5,
                lower_neg_y: -0.5,
                lower_m24_exact: None,
                negated_upper_m24_exact: None,
                lower_depth_two: None,
                negated_upper_depth_two: None,
                required_peak: 101,
                receipt_peak_delta: 0,
            },
            LeafResourceLimits {
                retained_live_bytes: 64,
                max_peak_live_bytes: 101,
            },
            authority_deadline,
        );
        oracle.admission = LeafAdmissionPolicy {
            primary_max_attempts: 1,
            max_attempts: 1,
            ..LeafAdmissionPolicy::PRODUCTION
        };
        let mut request = GraphInputLeafRequest {
            graph: &graph,
            input_bounds: &wide_input,
            objectives: &objectives,
            advisory_objective_bounds: &near,
            thresholds: &thresholds,
            clause_sizes: &clauses,
            depth: LEAF_MIN_DEPTH - 1,
            deadline: Some(authority_deadline),
        };

        let now = Instant::now();
        assert!(oracle
            .begin_attempt(&request, &lower, &upper, now, authority_deadline)
            .is_none());
        request.depth = LEAF_MIN_DEPTH;
        request.advisory_objective_bounds = &far;
        assert!(oracle
            .begin_attempt(&request, &lower, &upper, now, authority_deadline)
            .is_none());
        request.advisory_objective_bounds = &near;
        assert!(oracle
            .begin_attempt(
                &request,
                &lower,
                &upper,
                now,
                (now + LEAF_MIN_GLOBAL_REMAINING)
                    .checked_sub(Duration::from_nanos(1))
                    .unwrap(),
            )
            .is_none());

        let (call_deadline, tranche, guard) = oracle
            .begin_attempt(&request, &lower, &upper, now, authority_deadline)
            .expect("near deep leaf with ample global time");
        assert_eq!(call_deadline, now + LEAF_CALL_BUDGET);
        assert_eq!(tranche, LeafAttemptTranche::Primary);
        drop(guard);

        oracle.telemetry.total_wall_nanos.store(
            oracle.admission.latest_start_wall_nanos() + 1,
            Ordering::Relaxed,
        );
        assert!(oracle
            .begin_attempt(&request, &lower, &upper, now, authority_deadline)
            .is_none());
        oracle
            .telemetry
            .total_wall_nanos
            .store(0, Ordering::Relaxed);
        assert!(oracle
            .begin_attempt(&request, &lower, &upper, now, authority_deadline)
            .is_none());

        let snapshot = oracle.telemetry.snapshot();
        assert_eq!(snapshot.attempts, 1);
        assert_eq!(snapshot.depth_or_volume_skips, 1);
        assert_eq!(snapshot.objective_shortfall_skips, 1);
        assert_eq!(snapshot.global_remaining_skips, 1);
        assert_eq!(snapshot.total_wall_skips, 1);
        assert_eq!(snapshot.attempt_cap_skips, 1);

        // Root admission ignores advisory tightness by design, but it remains
        // subject to the same deadline/resource firewall.
        let mut root_oracle = oracle_at_deadline(
            &graph,
            CganCzImgSz32Profile::Nch1,
            FakeRows {
                lower_y: 0.5,
                lower_neg_y: -0.5,
                lower_m24_exact: None,
                negated_upper_m24_exact: None,
                lower_depth_two: None,
                negated_upper_depth_two: None,
                required_peak: 101,
                receipt_peak_delta: 0,
            },
            LeafResourceLimits {
                retained_live_bytes: 64,
                max_peak_live_bytes: 101,
            },
            authority_deadline,
        );
        root_oracle.admission = LeafAdmissionPolicy::PRODUCTION;
        request.depth = 0;
        request.advisory_objective_bounds = &far;
        let root_guard = root_oracle
            .begin_attempt(&request, &lower, &upper, now, authority_deadline)
            .expect("one root probe")
            .2;
        drop(root_guard);

        let shallow_small_input = input([-0.5; LATENT_DIM], [0.5; LATENT_DIM]);
        let shallow_small_request = GraphInputLeafRequest {
            graph: &graph,
            input_bounds: &shallow_small_input,
            objectives: &objectives,
            advisory_objective_bounds: &near,
            thresholds: &thresholds,
            clause_sizes: &clauses,
            depth: LEAF_MIN_DEPTH - 1,
            deadline: Some(authority_deadline),
        };
        let shallow_small_guard = root_oracle
            .begin_attempt(
                &shallow_small_request,
                &[-0.5; LATENT_DIM],
                &[0.5; LATENT_DIM],
                now,
                authority_deadline,
            )
            .expect("normalized volume 1/32 admits before depth five")
            .2;
        drop(shallow_small_guard);
        assert_eq!(root_oracle.telemetry.snapshot().attempts, 2);
    }

    #[test]
    fn both_profiles_map_reversed_signed_rows_and_accept_no_later_caller_deadlines() {
        for profile in [CganCzImgSz32Profile::Nch1, CganCzImgSz32Profile::Nch3] {
            let graph = graph();
            let input = input([-0.5; LATENT_DIM], [0.5; LATENT_DIM]);
            // Reverse the source order: +Y first, -Y second. Authentication is
            // by exact sign+threshold, never by an assumed clause index.
            let objectives = Array2::from_shape_vec((2, 1), vec![1.0, -1.0]).unwrap();
            let thresholds = [0.25, -0.75];
            let clauses = [1, 1];
            let deadline = Instant::now() + Duration::from_secs(10);
            let resources = LeafResourceLimits {
                retained_live_bytes: 64,
                max_peak_live_bytes: 101,
            };
            let oracle = oracle_at_deadline(
                &graph,
                profile,
                FakeRows {
                    lower_y: 0.5,
                    lower_neg_y: -0.5,
                    lower_m24_exact: None,
                    negated_upper_m24_exact: None,
                    lower_depth_two: None,
                    negated_upper_depth_two: None,
                    required_peak: 101,
                    receipt_peak_delta: 0,
                },
                resources,
                deadline,
            );
            assert!(verified(oracle.solve_input_leaf(&request(
                &graph,
                &input,
                &objectives,
                &thresholds,
                &clauses,
                Some(deadline),
            ))));
            assert!(verified(oracle.solve_input_leaf(&request(
                &graph,
                &input,
                &objectives,
                &thresholds,
                &clauses,
                Some(deadline.checked_sub(Duration::from_nanos(1)).unwrap()),
            ))));
            assert!(!verified(oracle.solve_input_leaf(&request(
                &graph,
                &input,
                &objectives,
                &thresholds,
                &clauses,
                Some(deadline + Duration::from_nanos(1)),
            ))));
        }
    }

    #[test]
    fn one_certified_row_or_a_duplicate_sign_has_no_authority() {
        let graph = graph();
        let input = input([-0.5; LATENT_DIM], [0.5; LATENT_DIM]);
        let deadline = Instant::now() + Duration::from_secs(10);
        let oracle = oracle_at_deadline(
            &graph,
            CganCzImgSz32Profile::Nch1,
            FakeRows {
                lower_y: 0.5,
                lower_neg_y: -0.5,
                lower_m24_exact: None,
                negated_upper_m24_exact: None,
                lower_depth_two: None,
                negated_upper_depth_two: None,
                required_peak: 101,
                receipt_peak_delta: 0,
            },
            LeafResourceLimits {
                retained_live_bytes: 64,
                max_peak_live_bytes: 101,
            },
            deadline,
        );

        let one_objective = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        assert!(!verified(oracle.solve_input_leaf(&request(
            &graph,
            &input,
            &one_objective,
            &[0.25],
            &[1],
            Some(deadline),
        ))));

        let duplicates = Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap();
        assert!(!verified(oracle.solve_input_leaf(&request(
            &graph,
            &input,
            &duplicates,
            &[0.25, 0.25],
            &[1, 1],
            Some(deadline),
        ))));
    }

    #[test]
    fn both_strict_safe_inequalities_are_required() {
        let graph = graph();
        let input = input([-0.5; LATENT_DIM], [0.5; LATENT_DIM]);
        let objectives = Array2::from_shape_vec((2, 1), vec![-1.0, 1.0]).unwrap();
        let thresholds = [-0.75, 0.25];
        let clauses = [1, 1];
        for (lower_y, lower_neg_y) in [(0.25, -0.5), (0.5, -0.75)] {
            let oracle = oracle(
                &graph,
                CganCzImgSz32Profile::Nch1,
                FakeRows {
                    lower_y,
                    lower_neg_y,
                    lower_m24_exact: None,
                    negated_upper_m24_exact: None,
                    lower_depth_two: None,
                    negated_upper_depth_two: None,
                    required_peak: 101,
                    receipt_peak_delta: 0,
                },
                LeafResourceLimits {
                    retained_live_bytes: 64,
                    max_peak_live_bytes: 101,
                },
            );
            assert!(!verified(oracle.solve_input_leaf(&request(
                &graph,
                &input,
                &objectives,
                &thresholds,
                &clauses,
                Some(oracle.authority_deadline),
            ))));
        }
    }

    #[test]
    fn m24_only_crossing_is_telemetry_not_verdict_authority() {
        let graph = graph();
        let input = input([-0.5; LATENT_DIM], [0.5; LATENT_DIM]);
        let objectives = Array2::from_shape_vec((2, 1), vec![-1.0, 1.0]).unwrap();
        let thresholds = [-0.75, 0.25];
        let clauses = [1, 1];
        let oracle = oracle(
            &graph,
            CganCzImgSz32Profile::Nch1,
            FakeRows {
                // Both authoritative M17/M20 rows remain on the unsafe side.
                lower_y: 0.2,
                lower_neg_y: -0.8,
                // Both exact M24 observations would cross strictly.
                lower_m24_exact: Some(0.3),
                negated_upper_m24_exact: Some(-0.7),
                lower_depth_two: None,
                negated_upper_depth_two: None,
                required_peak: 101,
                receipt_peak_delta: 0,
            },
            LeafResourceLimits {
                retained_live_bytes: 64,
                max_peak_live_bytes: 101,
            },
        );
        let verdict = oracle.solve_input_leaf(&request(
            &graph,
            &input,
            &objectives,
            &thresholds,
            &clauses,
            Some(oracle.authority_deadline),
        ));
        assert!(matches!(verdict, GraphMipLeafVerdict::Undecided));

        let snapshot = oracle.telemetry.snapshot();
        assert_eq!(snapshot.completions, 1);
        assert_eq!(snapshot.verified_leaves, 0);
        assert_eq!(snapshot.lower_m24_exact_present, 1);
        assert_eq!(snapshot.negated_upper_m24_exact_present, 1);
        assert_eq!(snapshot.lower_m24_missing, 0);
        assert_eq!(snapshot.negated_upper_m24_missing, 0);
        assert_eq!(snapshot.lower_m24_strict_wins, 1);
        assert_eq!(snapshot.negated_upper_m24_strict_wins, 1);
        assert!((snapshot.max_lower_m24_signed_gain - 0.1).abs() < 1.0e-15);
        assert!((snapshot.max_negated_upper_m24_signed_gain - 0.1).abs() < 1.0e-15);
        assert!((snapshot.best_counterfactual_m24_joint_threshold_residual - 0.05).abs() < 1.0e-15);
        assert_eq!(snapshot.m24_only_would_verify, 1);
        assert!(snapshot.best_joint_threshold_residual < 0.0);
    }

    #[test]
    fn depth_two_crossings_and_huge_values_never_gain_verdict_authority() {
        let graph = graph();
        let input = input([-0.5; LATENT_DIM], [0.5; LATENT_DIM]);
        let objectives = Array2::from_shape_vec((2, 1), vec![-1.0, 1.0]).unwrap();
        let thresholds = [-0.75, 0.25];
        let clauses = [1, 1];
        let resources = LeafResourceLimits {
            retained_live_bytes: 64,
            max_peak_live_bytes: 101,
        };

        let coherent = oracle(
            &graph,
            CganCzImgSz32Profile::Nch1,
            FakeRows {
                lower_y: 0.2,
                lower_neg_y: -0.8,
                lower_m24_exact: None,
                negated_upper_m24_exact: None,
                lower_depth_two: Some(0.3),
                negated_upper_depth_two: Some(-0.7),
                required_peak: 101,
                receipt_peak_delta: 0,
            },
            resources,
        );
        let verdict = coherent.solve_input_leaf(&request(
            &graph,
            &input,
            &objectives,
            &thresholds,
            &clauses,
            Some(coherent.authority_deadline),
        ));
        assert!(matches!(verdict, GraphMipLeafVerdict::Undecided));
        let snapshot = coherent.telemetry.snapshot();
        assert_eq!(snapshot.verified_leaves, 0);
        assert_eq!(snapshot.depth_two_completed_sides, 2);
        assert_eq!(snapshot.depth_two_incoherent_pairs, 0);
        assert_eq!(snapshot.lower_depth_two_strict_wins, 1);
        assert_eq!(snapshot.negated_upper_depth_two_strict_wins, 1);
        assert_eq!(snapshot.depth_two_only_would_verify, 1);
        assert!(
            (snapshot.best_counterfactual_depth_two_joint_threshold_residual - 0.05).abs()
                < 1.0e-15
        );

        let huge = oracle(
            &graph,
            CganCzImgSz32Profile::Nch1,
            FakeRows {
                lower_y: 0.2,
                lower_neg_y: -0.8,
                lower_m24_exact: None,
                negated_upper_m24_exact: None,
                lower_depth_two: Some(1.0e100),
                negated_upper_depth_two: Some(1.0e100),
                required_peak: 101,
                receipt_peak_delta: 0,
            },
            resources,
        );
        let verdict = huge.solve_input_leaf(&request(
            &graph,
            &input,
            &objectives,
            &thresholds,
            &clauses,
            Some(huge.authority_deadline),
        ));
        assert!(matches!(verdict, GraphMipLeafVerdict::Undecided));
        let snapshot = huge.telemetry.snapshot();
        assert_eq!(snapshot.verified_leaves, 0);
        assert_eq!(snapshot.depth_two_completed_sides, 2);
        assert_eq!(snapshot.depth_two_incoherent_pairs, 1);
        assert_eq!(snapshot.lower_depth_two_strict_wins, 0);
        assert_eq!(snapshot.negated_upper_depth_two_strict_wins, 0);
        assert_eq!(snapshot.max_lower_depth_two_signed_gain, f64::NEG_INFINITY);
        assert_eq!(
            snapshot.max_negated_upper_depth_two_signed_gain,
            f64::NEG_INFINITY
        );
        assert_eq!(snapshot.depth_two_only_would_verify, 0);
    }

    #[test]
    fn deadline_one_byte_cap_and_receipt_mismatches_fail_closed() {
        let graph = graph();
        let input = input([-0.5; LATENT_DIM], [0.5; LATENT_DIM]);
        let objectives = Array2::from_shape_vec((2, 1), vec![-1.0, 1.0]).unwrap();
        let thresholds = [-0.75, 0.25];
        let clauses = [1, 1];
        let base_rows = FakeRows {
            lower_y: 0.5,
            lower_neg_y: -0.5,
            lower_m24_exact: None,
            negated_upper_m24_exact: None,
            lower_depth_two: None,
            negated_upper_depth_two: None,
            required_peak: 101,
            receipt_peak_delta: 0,
        };

        let expired_deadline = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap();
        let expired = oracle_at_deadline(
            &graph,
            CganCzImgSz32Profile::Nch1,
            base_rows,
            LeafResourceLimits {
                retained_live_bytes: 64,
                max_peak_live_bytes: 101,
            },
            expired_deadline,
        );
        assert!(!verified(expired.solve_input_leaf(&request(
            &graph,
            &input,
            &objectives,
            &thresholds,
            &clauses,
            Some(expired_deadline),
        ))));
        assert!(!verified(expired.solve_input_leaf(&request(
            &graph,
            &input,
            &objectives,
            &thresholds,
            &clauses,
            None,
        ))));

        let one_byte_low = oracle(
            &graph,
            CganCzImgSz32Profile::Nch1,
            base_rows,
            LeafResourceLimits {
                retained_live_bytes: 64,
                max_peak_live_bytes: 100,
            },
        );
        assert!(!verified(one_byte_low.solve_input_leaf(&request(
            &graph,
            &input,
            &objectives,
            &thresholds,
            &clauses,
            Some(one_byte_low.authority_deadline),
        ))));

        let bad_receipt = oracle(
            &graph,
            CganCzImgSz32Profile::Nch1,
            FakeRows {
                receipt_peak_delta: 1,
                ..base_rows
            },
            LeafResourceLimits {
                retained_live_bytes: 64,
                max_peak_live_bytes: 101,
            },
        );
        assert!(!verified(bad_receipt.solve_input_leaf(&request(
            &graph,
            &input,
            &objectives,
            &thresholds,
            &clauses,
            Some(bad_receipt.authority_deadline),
        ))));
    }

    #[test]
    fn bounder_panics_are_contained_as_undecided() {
        let graph = graph();
        let input = input([-0.5; LATENT_DIM], [0.5; LATENT_DIM]);
        let objectives = Array2::from_shape_vec((2, 1), vec![-1.0, 1.0]).unwrap();
        let authority_deadline = Instant::now() + Duration::from_secs(10);
        let oracle = CganInputLeafOracle {
            profile: CganCzImgSz32Profile::Nch1,
            expected_graph_scope: graph.cut_fold_scope(),
            authority_deadline,
            root_lower: [-1.0; LATENT_DIM],
            root_upper: [1.0; LATENT_DIM],
            rows: AuthenticatedRows {
                request_y_threshold: 0.25,
                request_neg_y_threshold: -0.75,
                y_threshold: 0.25,
                neg_y_threshold: -0.75,
            },
            resources: LeafResourceLimits {
                retained_live_bytes: 64,
                max_peak_live_bytes: 101,
            },
            admission: LeafAdmissionPolicy::UNRESTRICTED_TEST,
            telemetry: CganInputLeafTelemetry::new(),
            bounder: Arc::new(PanicBounder),
        };
        assert!(!verified(oracle.solve_input_leaf(&request(
            &graph,
            &input,
            &objectives,
            &[-0.75, 0.25],
            &[1, 1],
            Some(authority_deadline),
        ))));
        assert!(!oracle.telemetry.in_flight.load(Ordering::Acquire));
        assert_eq!(oracle.telemetry.snapshot().attempts, 1);
    }

    #[test]
    fn graph_scope_leaf_shape_bounds_and_threshold_drift_fail_closed() {
        let other_graph = graph();
        let graph = graph();
        let good_input = input([-0.5; LATENT_DIM], [0.5; LATENT_DIM]);
        let outside_input = input([-1.5; LATENT_DIM], [0.5; LATENT_DIM]);
        let short_input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![-0.5; 4]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.5; 4]).unwrap(),
        )
        .unwrap();
        let mut infeasible_input = good_input.clone();
        infeasible_input.mark_infeasible_all();
        let objectives = Array2::from_shape_vec((2, 1), vec![-1.0, 1.0]).unwrap();
        let clauses = [1, 1];
        let oracle = oracle(
            &graph,
            CganCzImgSz32Profile::Nch1,
            FakeRows {
                lower_y: 0.5,
                lower_neg_y: -0.5,
                lower_m24_exact: None,
                negated_upper_m24_exact: None,
                lower_depth_two: None,
                negated_upper_depth_two: None,
                required_peak: 101,
                receipt_peak_delta: 0,
            },
            LeafResourceLimits {
                retained_live_bytes: 64,
                max_peak_live_bytes: 101,
            },
        );
        let cases: [(&GraphNetwork, &BoundedTensor, [f32; 2]); 5] = [
            (&other_graph, &good_input, [-0.75, 0.25]),
            (&graph, &outside_input, [-0.75, 0.25]),
            (&graph, &short_input, [-0.75, 0.25]),
            (&graph, &infeasible_input, [-0.75, 0.25]),
            (
                &graph,
                &good_input,
                [-0.75, f32::from_bits(0.25_f32.to_bits() + 1)],
            ),
        ];
        for (request_graph, request_input, thresholds) in cases {
            assert!(!verified(oracle.solve_input_leaf(&request(
                request_graph,
                request_input,
                &objectives,
                &thresholds,
                &clauses,
                Some(oracle.authority_deadline),
            ))));
        }
    }
}
