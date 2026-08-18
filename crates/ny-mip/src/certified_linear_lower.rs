// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proof-carrying lower bounds for a fixed linear form over a small MILP.
//!
//! This is the reusable authority seam needed by post-prefix neural-network
//! oracles. The proposal route first asks AY to optimize only to *propose* a
//! binary32 lower bound, rounds it strictly downward, and checks it with a
//! separate decision solve. The decision-only route skips that
//! non-authoritative optimization and checks an explicit caller-selected
//! binary32 threshold directly:
//!
//! ```text
//! model ∧ linear_form <= proposed_lower
//! ```
//!
//! Before branch-and-bound, the decision route relaxes integrality and asks AY
//! for an exact certified lower row on the requested form. If that relaxation
//! lower is strictly above the selected threshold, its exact entailment is
//! enough: the relaxed feasible set contains the MILP feasible set. Otherwise
//! the original decision MILP remains the fail-closed fallback.
//!
//! A bound is returned only when one of those routes establishes the strict
//! separation and:
//!
//! 1. AY's relaxation entailment, root Farkas, or whole branch-tree
//!    certificate verifies exactly against the caller's lowered model; and
//! 2. every linear obligation is independently reconstructed from the
//!    original [`crate::ir::MilpProblem`] and accepted by
//!    [`ny_cert::check_entailment`] or [`ny_cert::check_farkas`].
//!
//! The optimization answer therefore has no authority.  It may be arbitrarily
//! wrong without producing a bound; only an independently replayed linear
//! proof can cross this API.

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ay_milp::{
    AdaptiveFiveLeafCombTargetFsbReport, AdaptiveFourLeafCombTargetFsbReport,
    AdaptiveThreeLeafTargetFsbReport, BabSession, BoundSide, CertifiedAdaptiveFiveLeafComb,
    CertifiedAdaptiveFourLeafComb, CertifiedAdaptiveThreeLeafHarvest,
    CertifiedAdaptiveThreeLeafTree, CertifiedBinaryAssignmentTree, CertifiedBinaryTreeHarvest,
    CertifiedRow as AyCertifiedRow, CertifiedSplitHarvest, FactRef,
    FarkasCertificate as AyFarkasCertificate, FixedAssignmentTreeWarmStart, LpSession,
    MilpInfeasibilityCertificate, Outcome, Sense as AySense, SolveOpts, TargetFsbOpts, TreeNode,
    MAX_TARGET_FSB_CANDIDATES,
};
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};
use ny_cert::{
    check_entailment, check_farkas, ConstraintKind, EntailmentCertificate,
    FarkasCertificate as NyFarkasCertificate, LinearConstraint, Rat,
};

use crate::ay_lib::{
    run_with_hard_deadline, run_with_hard_deadline_at, solve_opts, to_ay_model,
    SOLVE_THREAD_STACK_BYTES,
};
use crate::error::MipError;
use crate::ir::{Col, MilpProblem, RowSpec};

/// Hard ceiling on the branch-tree certificate admitted by this API.
///
/// The tail oracle this was built for has roughly 18 unstable ReLUs.  A proof
/// larger than this is no longer a bounded tail proof and is declined before
/// ny-cert replay can turn it into an unpriced exact-arithmetic workload.
pub const CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES: usize = 4_096;

/// The continuous-relaxation entailment is a fast path, not the whole proof
/// strategy. Reserve most of the shared deadline for the integral decision
/// fallback when the relaxation has a genuine integrality gap.
const RELAXATION_PROOF_SHARE: f64 = 0.25;
const RELAXATION_PROOF_CAP: Duration = Duration::from_secs(12);

/// A target-ranked one-split probe needs enough time for one cold root solve
/// plus two warm child repairs. This schedule is reachable only through the
/// explicit branch-advice API; the ordinary production path above is
/// deliberately unchanged.
const ADVISED_RELAXATION_PROOF_SHARE: f64 = 0.5;
const ADVISED_RELAXATION_PROOF_CAP: Duration = Duration::from_secs(24);

/// Two selected splits require four warm assignment solves after the cold
/// root. This larger slice remains opt-in and bounded: two-candidate and
/// oversized advice preserve the first-two control, while three through eight
/// candidates use bounded target-objective FSB to select a pair. Both paths
/// admit exactly four leaves.
const ADVISED_ASSIGNMENT_TREE_PROOF_SHARE: f64 = 0.8;
const ADVISED_ASSIGNMENT_TREE_PROOF_CAP: Duration = Duration::from_secs(36);

/// Bounded target-objective strong-branching policy for three through eight
/// canonical candidates. Its probes are scheduling advice only; the selected
/// four-leaf harvest still crosses AY verification and ny-cert replay.
const TARGET_FSB_MAX_PROBE_PIVOTS_PER_CALL: u64 = 25;
const TARGET_FSB_MAX_PROBE_CALLS: usize = 44;
const TARGET_FSB_PROBE_TIME_LIMIT: Duration = Duration::from_millis(1_500);
const TARGET_FSB_MAX_PROBE_SCRATCH_BYTES: usize = 128 << 20;

/// AY's fixed complete-assignment harvester admits at most sixteen leaves.
const FIXED_ASSIGNMENT_TREE_MAX_DEPTH: usize = 4;

/// The parallel canary is deliberately specific to the four selector bits
/// carried by the graph-MIP regional proof seam.
const PARALLEL_SELECTOR_TREE_DEPTH: usize = 4;
const PARALLEL_SELECTOR_TREE_LEAVES: usize = 1 << PARALLEL_SELECTOR_TREE_DEPTH;
const PARALLEL_SELECTOR_MAX_WORKERS: usize = PARALLEL_SELECTOR_TREE_LEAVES;
const PARALLEL_SELECTOR_MIN_OUTER_WAIT: Duration = Duration::from_millis(1);

/// Explicit AY chain-distress probe budget for the selector solve-profile
/// canary. The ordinary and range-only fixed-tree routes retain AY's
/// historical default.
pub const CERTIFIED_LINEAR_LOWER_SELECTOR_CHAIN_DISTRESS_PROBE_ITERS: u64 = 1_000;

/// Advice-only root probe used by the compact K16 tail's complete assignment
/// tree. The outer proof deadline remains authoritative.
pub const CERTIFIED_LINEAR_LOWER_COMPACT_TREE_ROOT_PROBE: Duration = Duration::from_millis(50);

/// Advice-only per-prefix bridge used after the compact tree root probe.
pub const CERTIFIED_LINEAR_LOWER_COMPACT_TREE_PREFIX_PROBE: Duration = Duration::from_millis(25);

fn parallel_selector_remaining_outer_wait(deadline: Instant, now: Instant) -> Option<Duration> {
    deadline
        .checked_duration_since(now)
        .filter(|remaining| *remaining >= PARALLEL_SELECTOR_MIN_OUTER_WAIT)
}

/// Preserve a small coordinator margin before starting an absolute-deadline
/// fixed-tree worker.
///
/// The AY session receives the exact caller-owned deadline, not this
/// duration.  The duration exists only to decline an already/nearly expired
/// request before spawning a detached worker whose result could no longer be
/// observed.
fn fixed_assignment_tree_remaining_outer_wait(deadline: Instant, now: Instant) -> Option<Duration> {
    deadline
        .checked_duration_since(now)
        .filter(|remaining| *remaining >= PARALLEL_SELECTOR_MIN_OUTER_WAIT)
}

fn fixed_assignment_tree_deadline_open(deadline: Instant) -> bool {
    Instant::now() < deadline
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LpSolvePolicy {
    Default,
    RangeLogical,
    SelectorSolveProfile,
    CompactTailProgressive,
}

impl LpSolvePolicy {
    fn apply(self, opts: SolveOpts) -> SolveOpts {
        match self {
            Self::Default => opts,
            Self::RangeLogical => opts.with_range_logical_triangular_crash(),
            Self::SelectorSolveProfile => opts
                .with_range_logical_triangular_crash()
                .with_chain_distress_probe_iters(Some(
                    CERTIFIED_LINEAR_LOWER_SELECTOR_CHAIN_DISTRESS_PROBE_ITERS,
                )),
            Self::CompactTailProgressive => opts.with_fixed_assignment_tree_warm_start(Some(
                FixedAssignmentTreeWarmStart::RootProbeThenProgressivePrefix {
                    root_time_limit: CERTIFIED_LINEAR_LOWER_COMPACT_TREE_ROOT_PROBE,
                    prefix_time_limit: CERTIFIED_LINEAR_LOWER_COMPACT_TREE_PREFIX_PROBE,
                    start_assignment: 0,
                },
            )),
        }
    }
}

/// Diagnostic-only target-FSB probe limits.
///
/// These two knobs affect scheduling advice only. They cannot alter the fixed
/// 44-call or 128 MiB workspace ceilings, and every selected tree must still
/// cross AY's exact whole-tree verification plus independent ny-cert replay.
/// Ordinary production entry points always use [`Self::production`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CertifiedLinearLowerTargetFsbProbeLimits {
    max_probe_pivots_per_call: u64,
    probe_time_limit: Duration,
}

impl CertifiedLinearLowerTargetFsbProbeLimits {
    /// The production target-FSB policy: 25 pivots per advice call and a
    /// 1,500 ms wall-clock cap shared by the complete probe scan.
    #[must_use]
    pub const fn production() -> Self {
        Self {
            max_probe_pivots_per_call: TARGET_FSB_MAX_PROBE_PIVOTS_PER_CALL,
            probe_time_limit: TARGET_FSB_PROBE_TIME_LIMIT,
        }
    }

    /// Construct explicit diagnostic probe limits.
    ///
    /// The shared probe limit remains subordinate to the proof worker's outer
    /// deadline. Zero resources are rejected instead of silently changing the
    /// advice path into an ordinary decline.
    pub fn new(
        max_probe_pivots_per_call: u64,
        probe_time_limit: Duration,
    ) -> Result<Self, MipError> {
        if max_probe_pivots_per_call == 0 {
            return Err(MipError::Encoding(
                "target-FSB max_probe_pivots_per_call must be nonzero".to_owned(),
            ));
        }
        if probe_time_limit.is_zero() {
            return Err(MipError::Encoding(
                "target-FSB probe_time_limit must be nonzero".to_owned(),
            ));
        }
        Ok(Self {
            max_probe_pivots_per_call,
            probe_time_limit,
        })
    }

    /// Per-advice-call dual-pivot cap.
    #[must_use]
    pub const fn max_probe_pivots_per_call(self) -> u64 {
        self.max_probe_pivots_per_call
    }

    /// Wall-clock cap shared by the complete target-FSB advice scan.
    #[must_use]
    pub const fn probe_time_limit(self) -> Duration {
        self.probe_time_limit
    }
}

/// Explicit budgets for one proposal-and-proof attempt.
///
/// There is intentionally no `Default`: a caller must price both the
/// non-authoritative optimization and the authoritative decision proof.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CertifiedLinearLowerBoundConfig {
    /// Wall-clock slice for AY's proposal optimization.
    pub proposal_timeout_secs: f64,
    /// Wall-clock slice for the separately solved exact relaxation entailment
    /// or MILP infeasibility proof.
    pub proof_timeout_secs: f64,
    /// Maximum AY branch-tree leaves accepted and replayed.
    pub max_tree_leaves: usize,
}

/// Explicit budget for certifying one caller-selected lower threshold.
///
/// Unlike [`CertifiedLinearLowerBoundConfig`], this decision-only route does
/// not spend time optimizing a non-authoritative proposal. The caller chooses
/// `q`; AY must prove either that the continuous relaxation entails a stronger
/// lower row or that `model ∧ linear_form <= q` is infeasible. Every proof
/// obligation is replayed exactly before `q` is returned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CertifiedLinearLowerDecisionConfig {
    /// Wall-clock slice for the exact relaxation entailment or MILP
    /// infeasibility proof.
    pub proof_timeout_secs: f64,
    /// Maximum AY branch-tree leaves accepted and replayed.
    pub max_tree_leaves: usize,
}

/// A finite binary32 lower bound carrying two independent exact replays.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CertifiedLinearLowerBound {
    /// Certified binary32 lower bound. The optimization route rounds its
    /// proposal strictly downward; the decision-only route returns the exact
    /// caller-selected binary32 threshold proved by the certificate.
    pub lower: f32,
    /// Exact proof route that authorized this bound.
    pub proof_route: CertifiedLinearLowerProofRoute,
    /// Number of leaves in AY's case-split certificate (zero for relaxation
    /// entailment and root Farkas).
    pub ay_tree_leaves: usize,
    /// Number of linear proof obligations independently accepted by ny-cert.
    ///
    /// The legacy field name includes the root/tree Farkas obligations this
    /// API originally admitted. A relaxation-fast-path result instead counts
    /// its one exact entailment replay here; both are non-negative linear
    /// combinations checked independently from AY.
    pub ny_cert_farkas_replays: usize,
}

/// Exact proof route that authorized a certified linear lower bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CertifiedLinearLowerProofRoute {
    /// The continuous relaxation entailed a strictly stronger lower row.
    RelaxationEntailment,
    /// The original decision MILP had an exact root Farkas certificate.
    RootFarkas,
    /// The original decision MILP had an exact branch-tree certificate.
    TreeFarkas,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReplayStats {
    proof_route: CertifiedLinearLowerProofRoute,
    tree_leaves: usize,
    linear_replays: usize,
}

/// Independently checked evidence that a continuous root problem is infeasible.
///
/// This marker is intentionally proof-data-free: AY's certificate is consumed
/// inside the hard-deadline worker, verified against AY's exact model, and then
/// reconstructed from the original [`MilpProblem`] for a separate
/// [`ny_cert::check_farkas`] replay before this value can be returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CertifiedContinuousRootInfeasibility {
    /// Number of positive exact multipliers in AY's verified root certificate.
    pub ay_farkas_multipliers: usize,
    /// Number of independent ny-cert Farkas replays (always one for this lane).
    pub ny_cert_farkas_replays: usize,
}

/// Exact-certificate resource ceilings for the small continuous-root lane.
///
/// These match the established AY LP-dual rational ceilings while additionally
/// bounding the facts expanded by both exact verifiers. The accounting is
/// deliberately explicit:
///
/// * numerator and denominator magnitude bits are each subject to the per-value
///   cap and are BOTH charged to the aggregate bit budget;
/// * every `FactRef` must be unique, preventing repeated expansion of one dense
///   row under split multipliers;
/// * stored row coefficient entries, canonical referenced-row nonzeros, and
///   the final expanded linear terms (row nonzeros plus column-bound facts) are
///   separately bounded;
/// * exact conversion bits for the original model's referenced coefficients
///   and bounds are charged, and each fact's expansion is weighted by the full
///   numerator-plus-denominator size of its multiplier. Independent count and
///   bit caps therefore cannot combine into multi-gigabyte bigint work.
///
/// The complete check runs after AY has generated/lifted the certificate but
/// before this lane's added explicit `AyFarkasCertificate::verify` call and its
/// independent ny-cert reconstruction/replay. It therefore prices those two
/// replay stages; it does not claim to bound certificate work already performed
/// internally by `LpSession::optimize_model_objective`.
const CONTINUOUS_ROOT_MAX_FARKAS_MULTIPLIERS: usize = 32_768;
const CONTINUOUS_ROOT_MAX_FARKAS_RATIONAL_BITS: u64 = 32_768;
const CONTINUOUS_ROOT_MAX_FARKAS_TOTAL_BITS: u64 = 16_777_216;
const CONTINUOUS_ROOT_MAX_FARKAS_REFERENCED_ROW_ENTRIES: usize = 1_000_000;
const CONTINUOUS_ROOT_MAX_FARKAS_REFERENCED_ROW_NONZEROS: usize = 1_000_000;
const CONTINUOUS_ROOT_MAX_FARKAS_EXPANDED_FACT_TERMS: usize = 1_000_000;
const CONTINUOUS_ROOT_MAX_FARKAS_MODEL_RATIONAL_BITS: u64 = 67_108_864;
const CONTINUOUS_ROOT_MAX_FARKAS_WEIGHTED_REPLAY_WORK: u64 = 134_217_728;
const CONTINUOUS_ROOT_MAX_COLUMNS: usize = 4_096;
const CONTINUOUS_ROOT_MAX_ROWS: usize = 8_192;

/// Pre-solve ceilings for converting the complete continuous-root IR into AY's
/// exact rational model. Certificate-dependent accounting below cannot protect
/// this earlier conversion, so every row entry and every finite model scalar
/// is priced independently before cloning or lowering the problem.
const CONTINUOUS_ROOT_PROBLEM_RESOURCE_LIMITS: ContinuousRootProblemResourceLimits =
    ContinuousRootProblemResourceLimits {
        max_columns: CONTINUOUS_ROOT_MAX_COLUMNS,
        max_rows: CONTINUOUS_ROOT_MAX_ROWS,
        max_row_entries: CONTINUOUS_ROOT_MAX_FARKAS_REFERENCED_ROW_ENTRIES,
        max_rational_bits: CONTINUOUS_ROOT_MAX_FARKAS_MODEL_RATIONAL_BITS,
    };

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ContinuousRootProblemResourceLimits {
    max_columns: usize,
    max_rows: usize,
    max_row_entries: usize,
    max_rational_bits: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ContinuousRootProblemResourceUsage {
    columns: usize,
    rows: usize,
    row_entries: usize,
    rational_bits: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContinuousRootProblemResourceRejection {
    MalformedModel,
    Columns,
    Rows,
    RowEntries,
    RationalBits,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ContinuousRootFarkasResourceLimits {
    max_multipliers: usize,
    max_rational_component_bits: u64,
    max_total_rational_bits: u64,
    max_referenced_row_entries: usize,
    max_referenced_row_nonzeros: usize,
    max_expanded_fact_terms: usize,
    max_model_rational_bits: u64,
    max_weighted_replay_work: u64,
}

const CONTINUOUS_ROOT_FARKAS_RESOURCE_LIMITS: ContinuousRootFarkasResourceLimits =
    ContinuousRootFarkasResourceLimits {
        max_multipliers: CONTINUOUS_ROOT_MAX_FARKAS_MULTIPLIERS,
        max_rational_component_bits: CONTINUOUS_ROOT_MAX_FARKAS_RATIONAL_BITS,
        max_total_rational_bits: CONTINUOUS_ROOT_MAX_FARKAS_TOTAL_BITS,
        max_referenced_row_entries: CONTINUOUS_ROOT_MAX_FARKAS_REFERENCED_ROW_ENTRIES,
        max_referenced_row_nonzeros: CONTINUOUS_ROOT_MAX_FARKAS_REFERENCED_ROW_NONZEROS,
        max_expanded_fact_terms: CONTINUOUS_ROOT_MAX_FARKAS_EXPANDED_FACT_TERMS,
        max_model_rational_bits: CONTINUOUS_ROOT_MAX_FARKAS_MODEL_RATIONAL_BITS,
        max_weighted_replay_work: CONTINUOUS_ROOT_MAX_FARKAS_WEIGHTED_REPLAY_WORK,
    };

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ContinuousRootFarkasResourceUsage {
    multipliers: usize,
    rational_bits: u64,
    referenced_row_entries: usize,
    referenced_row_nonzeros: usize,
    column_bound_facts: usize,
    expanded_fact_terms: usize,
    model_rational_bits: u64,
    weighted_replay_work: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContinuousRootFarkasResourceRejection {
    MultiplierCount,
    RationalComponentBits,
    RationalTotalBits,
    DuplicateFact { index: usize },
    MissingFact { index: usize },
    InfiniteFact { index: usize },
    UnsupportedFact { index: usize },
    MalformedRow { index: usize },
    ReferencedRowEntries,
    ReferencedRowNonzeros,
    ExpandedFactTerms,
    ModelRationalBits,
    WeightedReplayWork,
}

/// Process-wide admission for the exact lower-bound worker.
///
/// [`run_with_hard_deadline`] deliberately detaches an AY worker that exceeds
/// its wall-clock slice.  Without a lease retained by that detached worker,
/// repeated region/objective calls could accumulate several large exact MILPs
/// after timeouts and turn a fail-closed performance lane into an RSS hazard.
/// A new attempt therefore declines while the preceding worker is still alive.
static CERTIFIED_LINEAR_LOWER_WORKER_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Opaque process-wide admission for one exact lower-bound worker.
///
/// Callers that must perform substantial model construction can acquire this
/// *before* encoding and pass it to
/// [`certify_linear_lower_bound_with_ay_admission`] or
/// [`certify_linear_lower_bound_at_with_ay_admission`]. If a worker detached
/// at a prior hard deadline is still alive, acquisition fails immediately and
/// the caller can shed the entire encoding/RSS workload.
///
/// The private field prevents callers from forging or dropping an unacquired
/// admission and reopening the lane underneath a live worker:
///
/// ```compile_fail
/// let _forged = ny_mip::CertifiedLinearLowerWorkerAdmission { _private: () };
/// ```
#[must_use = "dropping the admission reopens the exact-worker lane"]
pub struct CertifiedLinearLowerWorkerAdmission {
    _private: (),
}

impl CertifiedLinearLowerWorkerAdmission {
    /// Try to reserve the single process-wide exact AY worker slot.
    pub fn try_acquire() -> Option<Self> {
        CERTIFIED_LINEAR_LOWER_WORKER_ACTIVE
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| Self { _private: () })
    }
}

impl Drop for CertifiedLinearLowerWorkerAdmission {
    fn drop(&mut self) {
        CERTIFIED_LINEAR_LOWER_WORKER_ACTIVE.store(false, Ordering::Release);
    }
}

fn next_down_f32(value: f32) -> f32 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    if magnitude > f32::INFINITY.to_bits() || bits == f32::NEG_INFINITY.to_bits() {
        return value;
    }
    if magnitude == 0 {
        return -f32::from_bits(1);
    }
    if bits & 0x8000_0000 == 0 {
        f32::from_bits(bits - 1)
    } else {
        f32::from_bits(bits + 1)
    }
}

/// Convert an exact rational proposal to a *strictly* smaller finite f32.
///
/// `ToPrimitive::to_f32` may round either way.  Moving one representable value
/// below a small numerical separation and checking the result again in exact
/// arithmetic removes any dependency on that rounding direction.  The
/// separation is authority-neutral (the exact proof remains mandatory), but
/// keeps AY's floating search from accepting an attained optimum through its
/// feasibility tolerance before it can construct the exact linear proof.
fn strict_outward_f32_lower(value: &BigRational) -> Option<f32> {
    let nearest = value.to_f32()?;
    if !nearest.is_finite() {
        return None;
    }
    let nearest64 = f64::from(nearest);
    let separation = 1.0e-6 * (1.0 + nearest64.abs());
    let separated = nearest64 - separation;
    if !separated.is_finite() {
        return None;
    }
    let mut lower = next_down_f32(separated as f32);
    for _ in 0..2 {
        if !lower.is_finite() {
            return None;
        }
        // Decode binary32 directly so host DAZ state cannot collapse a
        // subnormal during an intermediate floating conversion.
        let exact = BigRational::from_float(lower)?;
        if exact < *value {
            return Some(lower);
        }
        lower = next_down_f32(lower);
    }
    None
}

fn validate_config(config: CertifiedLinearLowerBoundConfig) -> Result<(), MipError> {
    for (field, value) in [
        ("proposal_timeout_secs", config.proposal_timeout_secs),
        ("proof_timeout_secs", config.proof_timeout_secs),
    ] {
        if !value.is_finite() || value <= 0.0 || value > 300.0 {
            return Err(MipError::Encoding(format!(
                "{field} must be finite and in (0, 300], got {value}"
            )));
        }
    }
    if config.max_tree_leaves == 0
        || config.max_tree_leaves > CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES
    {
        return Err(MipError::Encoding(format!(
            "max_tree_leaves must be in 1..={CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES}, got {}",
            config.max_tree_leaves
        )));
    }
    Ok(())
}

fn validate_decision_config(config: CertifiedLinearLowerDecisionConfig) -> Result<(), MipError> {
    if !config.proof_timeout_secs.is_finite()
        || config.proof_timeout_secs <= 0.0
        || config.proof_timeout_secs > 300.0
    {
        return Err(MipError::Encoding(format!(
            "proof_timeout_secs must be finite and in (0, 300], got {}",
            config.proof_timeout_secs
        )));
    }
    if config.max_tree_leaves == 0
        || config.max_tree_leaves > CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES
    {
        return Err(MipError::Encoding(format!(
            "max_tree_leaves must be in 1..={CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES}, got {}",
            config.max_tree_leaves
        )));
    }
    Ok(())
}

fn canonical_objective(
    problem: &MilpProblem,
    terms: &[(Col, f64)],
) -> Result<Vec<(Col, f64)>, MipError> {
    if terms.is_empty() {
        return Err(MipError::Encoding(
            "certified linear lower bound requires a nonempty objective".to_owned(),
        ));
    }
    let mut canonical = terms.to_vec();
    canonical.sort_unstable_by_key(|(col, _)| col.0);
    for (index, &(col, coeff)) in canonical.iter().enumerate() {
        if col.0 >= problem.num_cols() {
            return Err(MipError::Encoding(format!(
                "objective term {index} references column {}, but the model has {} columns",
                col.0,
                problem.num_cols()
            )));
        }
        if !coeff.is_finite() {
            return Err(MipError::Encoding(format!(
                "objective term {index} has non-finite coefficient"
            )));
        }
        if coeff == 0.0 {
            return Err(MipError::Encoding(format!(
                "objective term {index} has a zero coefficient"
            )));
        }
    }
    if canonical.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(MipError::Encoding(
            "certified linear objective contains duplicate columns".to_owned(),
        ));
    }
    Ok(canonical)
}

/// Normalize advice without granting it any authority over the model.
///
/// The one-split proof lane requires a complete binary disjunction, so only
/// unfixed integral `[0, 1]` columns survive. Stale and duplicate handles are
/// ignored in caller order. An empty result routes through the historical
/// no-advice proof path.
fn canonical_binary_branch_advice(problem: &MilpProblem, advice: &[Col]) -> Vec<Col> {
    let mut seen = vec![false; problem.num_cols()];
    let mut canonical = Vec::with_capacity(advice.len().min(problem.num_cols()));
    for &col in advice {
        let Some(spec) = problem.cols().get(col.0) else {
            continue;
        };
        if !spec.integer || spec.lb != 0.0 || spec.ub != 1.0 {
            continue;
        }
        if std::mem::replace(&mut seen[col.0], true) {
            continue;
        }
        canonical.push(col);
    }
    canonical
}

/// Validate a diagnostic fixed-assignment split list without reordering it.
///
/// Every position is proof-critical: AY interprets the first column as the
/// most-significant assignment bit. Silently filtering malformed handles would
/// therefore change both the requested topology and its leaf association.
fn fixed_assignment_tree_splits(
    problem: &MilpProblem,
    splits: &[Col],
) -> Result<Vec<Col>, MipError> {
    if !(1..=FIXED_ASSIGNMENT_TREE_MAX_DEPTH).contains(&splits.len()) {
        return Err(MipError::Encoding(format!(
            "fixed assignment-tree replay requires 1..={FIXED_ASSIGNMENT_TREE_MAX_DEPTH} split \
             columns, got {}",
            splits.len()
        )));
    }

    let mut seen = vec![false; problem.num_cols()];
    for (split_index, &col) in splits.iter().enumerate() {
        let Some(spec) = problem.cols().get(col.0) else {
            return Err(MipError::Encoding(format!(
                "fixed assignment-tree split {split_index} references column {}, but the model \
                 has {} columns",
                col.0,
                problem.num_cols()
            )));
        };
        if !spec.integer || spec.lb != 0.0 || spec.ub != 1.0 {
            return Err(MipError::Encoding(format!(
                "fixed assignment-tree split {split_index} column {} is not an unfixed integer \
                 [0, 1] column",
                col.0
            )));
        }
        if std::mem::replace(&mut seen[col.0], true) {
            return Err(MipError::Encoding(format!(
                "fixed assignment-tree split list repeats column {}",
                col.0
            )));
        }
    }
    Ok(splits.to_vec())
}

/// Validate the proof-critical selector order and parallelism of the unwired
/// four-selector canary.
///
/// Requiring exactly four selectors keeps the public experiment from silently
/// changing topology when it is eventually measured from graph-MIP. The
/// ordinary fixed-assignment API remains the general one-through-four lane.
fn parallel_selector_tree_request(
    problem: &MilpProblem,
    selectors: &[Col],
    max_workers: usize,
) -> Result<Vec<Col>, MipError> {
    if selectors.len() != PARALLEL_SELECTOR_TREE_DEPTH {
        return Err(MipError::Encoding(format!(
            "parallel selector-tree replay requires exactly \
             {PARALLEL_SELECTOR_TREE_DEPTH} selector columns, got {}",
            selectors.len()
        )));
    }
    if !(1..=PARALLEL_SELECTOR_MAX_WORKERS).contains(&max_workers) {
        return Err(MipError::Encoding(format!(
            "parallel selector-tree max_workers must be in \
             1..={PARALLEL_SELECTOR_MAX_WORKERS}, got {max_workers}"
        )));
    }
    fixed_assignment_tree_splits(problem, selectors)
}

/// Validate the explicit adaptive-three-leaf shortlist without reordering it.
///
/// Unlike ordinary branch advice, this diagnostic carries a caller-selected
/// root *index*. Silently filtering a stale, fixed, continuous, or duplicate
/// handle would retarget that index, so malformed candidates are rejected
/// instead. The returned vector is bit-for-bit the caller's order.
fn adaptive_three_leaf_candidates(
    problem: &MilpProblem,
    candidates: &[Col],
    root_candidate_index: usize,
) -> Result<Vec<Col>, MipError> {
    if !(2..=MAX_TARGET_FSB_CANDIDATES).contains(&candidates.len()) {
        return Err(MipError::Encoding(format!(
            "adaptive three-leaf target-FSB requires 2..={MAX_TARGET_FSB_CANDIDATES} candidates, \
             got {}",
            candidates.len()
        )));
    }
    if root_candidate_index >= candidates.len() {
        return Err(MipError::Encoding(format!(
            "adaptive three-leaf root candidate index {root_candidate_index} is outside {} \
             candidates",
            candidates.len()
        )));
    }

    let mut seen = vec![false; problem.num_cols()];
    for (candidate_index, &col) in candidates.iter().enumerate() {
        let Some(spec) = problem.cols().get(col.0) else {
            return Err(MipError::Encoding(format!(
                "adaptive three-leaf candidate {candidate_index} references column {}, but the \
                 model has {} columns",
                col.0,
                problem.num_cols()
            )));
        };
        if !spec.integer || spec.lb != 0.0 || spec.ub != 1.0 {
            return Err(MipError::Encoding(format!(
                "adaptive three-leaf candidate {candidate_index} column {} is not an unfixed \
                 integer [0, 1] column",
                col.0
            )));
        }
        if std::mem::replace(&mut seen[col.0], true) {
            return Err(MipError::Encoding(format!(
                "adaptive three-leaf candidate list repeats column {}",
                col.0
            )));
        }
    }
    Ok(candidates.to_vec())
}

/// Validate the explicit adaptive-four-leaf-comb shortlist without reordering.
///
/// The comb's caller-selected root is an index into this exact slice. Filtering
/// malformed handles would silently retarget that root, so unlike ordinary
/// branch advice every candidate must already be a unique, live, unfixed binary
/// column.
fn adaptive_four_leaf_comb_candidates(
    problem: &MilpProblem,
    candidates: &[Col],
    root_candidate_index: usize,
) -> Result<Vec<Col>, MipError> {
    if !(3..=MAX_TARGET_FSB_CANDIDATES).contains(&candidates.len()) {
        return Err(MipError::Encoding(format!(
            "adaptive four-leaf comb target-FSB requires 3..={MAX_TARGET_FSB_CANDIDATES} \
             candidates, got {}",
            candidates.len()
        )));
    }
    if root_candidate_index >= candidates.len() {
        return Err(MipError::Encoding(format!(
            "adaptive four-leaf comb root candidate index {root_candidate_index} is outside {} \
             candidates",
            candidates.len()
        )));
    }

    let mut seen = vec![false; problem.num_cols()];
    for (candidate_index, &col) in candidates.iter().enumerate() {
        let Some(spec) = problem.cols().get(col.0) else {
            return Err(MipError::Encoding(format!(
                "adaptive four-leaf comb candidate {candidate_index} references column {}, but \
                 the model has {} columns",
                col.0,
                problem.num_cols()
            )));
        };
        if !spec.integer || spec.lb != 0.0 || spec.ub != 1.0 {
            return Err(MipError::Encoding(format!(
                "adaptive four-leaf comb candidate {candidate_index} column {} is not an \
                 unfixed integer [0, 1] column",
                col.0
            )));
        }
        if std::mem::replace(&mut seen[col.0], true) {
            return Err(MipError::Encoding(format!(
                "adaptive four-leaf comb candidate list repeats column {}",
                col.0
            )));
        }
    }
    Ok(candidates.to_vec())
}

/// Validate the explicit adaptive-five-leaf-comb shortlist without reordering.
///
/// Every selected split is an index into this exact caller slice. Filtering a
/// malformed handle would silently retarget a proof-critical branch, so all
/// four through eight candidates must already be distinct live unfixed binary
/// columns.
fn adaptive_five_leaf_comb_candidates(
    problem: &MilpProblem,
    candidates: &[Col],
    root_candidate_index: usize,
) -> Result<Vec<Col>, MipError> {
    if !(4..=MAX_TARGET_FSB_CANDIDATES).contains(&candidates.len()) {
        return Err(MipError::Encoding(format!(
            "adaptive five-leaf comb target-FSB requires 4..={MAX_TARGET_FSB_CANDIDATES} \
             candidates, got {}",
            candidates.len()
        )));
    }
    if root_candidate_index >= candidates.len() {
        return Err(MipError::Encoding(format!(
            "adaptive five-leaf comb root candidate index {root_candidate_index} is outside {} \
             candidates",
            candidates.len()
        )));
    }

    let mut seen = vec![false; problem.num_cols()];
    for (candidate_index, &col) in candidates.iter().enumerate() {
        let Some(spec) = problem.cols().get(col.0) else {
            return Err(MipError::Encoding(format!(
                "adaptive five-leaf comb candidate {candidate_index} references column {}, but \
                 the model has {} columns",
                col.0,
                problem.num_cols()
            )));
        };
        if !spec.integer || spec.lb != 0.0 || spec.ub != 1.0 {
            return Err(MipError::Encoding(format!(
                "adaptive five-leaf comb candidate {candidate_index} column {} is not an \
                 unfixed integer [0, 1] column",
                col.0
            )));
        }
        if std::mem::replace(&mut seen[col.0], true) {
            return Err(MipError::Encoding(format!(
                "adaptive five-leaf comb candidate list repeats column {}",
                col.0
            )));
        }
    }
    Ok(candidates.to_vec())
}

fn exact_f64(value: f64, what: &str) -> Result<BigRational, MipError> {
    BigRational::from_float(value)
        .ok_or_else(|| MipError::Encoding(format!("{what} must be finite")))
}

/// Numerator-plus-denominator magnitude bits in the reduced exact rational
/// represented by one finite binary64 value, without allocating a bigint.
fn exact_f64_rational_bits(value: f64) -> Option<u64> {
    if !value.is_finite() {
        return None;
    }
    let bits = value.to_bits();
    let exponent_field = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (significand, exponent) = if exponent_field == 0 {
        (fraction, -1074_i32)
    } else {
        ((1_u64 << 52) | fraction, exponent_field - 1023 - 52)
    };
    if significand == 0 {
        // BigRational normalizes either signed zero to 0/1.
        return Some(1);
    }

    let significand_bits = u64::from(u64::BITS - significand.leading_zeros());
    if exponent >= 0 {
        return significand_bits
            .checked_add(u64::try_from(exponent).ok()?)
            .and_then(|numerator_bits| numerator_bits.checked_add(1));
    }

    let denominator_power = u32::try_from(-exponent).ok()?;
    let cancelled = significand.trailing_zeros().min(denominator_power);
    let reduced = significand >> cancelled;
    let numerator_bits = u64::from(u64::BITS - reduced.leading_zeros());
    let denominator_bits = u64::from(denominator_power - cancelled).checked_add(1)?;
    numerator_bits.checked_add(denominator_bits)
}

fn continuous_root_problem_resource_usage(
    problem: &MilpProblem,
) -> Result<ContinuousRootProblemResourceUsage, ContinuousRootProblemResourceRejection> {
    continuous_root_problem_resource_usage_with_limits(
        problem,
        CONTINUOUS_ROOT_PROBLEM_RESOURCE_LIMITS,
    )
}

fn continuous_root_problem_resource_usage_with_limits(
    problem: &MilpProblem,
    limits: ContinuousRootProblemResourceLimits,
) -> Result<ContinuousRootProblemResourceUsage, ContinuousRootProblemResourceRejection> {
    fn charge(
        value: f64,
        rational_bits: &mut u64,
        limit: u64,
    ) -> Result<(), ContinuousRootProblemResourceRejection> {
        let bits = exact_f64_rational_bits(value)
            .ok_or(ContinuousRootProblemResourceRejection::MalformedModel)?;
        *rational_bits = rational_bits
            .checked_add(bits)
            .ok_or(ContinuousRootProblemResourceRejection::RationalBits)?;
        if *rational_bits > limit {
            return Err(ContinuousRootProblemResourceRejection::RationalBits);
        }
        Ok(())
    }

    fn charge_bound(
        value: f64,
        rational_bits: &mut u64,
        limit: u64,
    ) -> Result<(), ContinuousRootProblemResourceRejection> {
        if value.is_nan() {
            return Err(ContinuousRootProblemResourceRejection::MalformedModel);
        }
        if value.is_finite() {
            charge(value, rational_bits, limit)?;
        }
        Ok(())
    }

    let columns = problem.num_cols();
    let rows = problem.num_rows();
    // These O(1) gates must precede every vector scan.  Otherwise arbitrarily
    // many empty rows or all-continuous columns could evade the entry/bit
    // accounting and consume unbounded synchronous work before the deadline
    // worker is even launched.
    if columns > limits.max_columns {
        return Err(ContinuousRootProblemResourceRejection::Columns);
    }
    if rows > limits.max_rows {
        return Err(ContinuousRootProblemResourceRejection::Rows);
    }

    let mut rational_bits = 0_u64;
    for column in problem.cols() {
        if !column.obj.is_finite()
            || column.lb > column.ub
            || column.lb == f64::INFINITY
            || column.ub == f64::NEG_INFINITY
        {
            return Err(ContinuousRootProblemResourceRejection::MalformedModel);
        }
        charge(column.obj, &mut rational_bits, limits.max_rational_bits)?;
        charge_bound(column.lb, &mut rational_bits, limits.max_rational_bits)?;
        charge_bound(column.ub, &mut rational_bits, limits.max_rational_bits)?;
    }

    let mut row_entries = 0_usize;
    for row in problem.rows() {
        if row.lb > row.ub || row.lb == f64::INFINITY || row.ub == f64::NEG_INFINITY {
            return Err(ContinuousRootProblemResourceRejection::MalformedModel);
        }
        charge_bound(row.lb, &mut rational_bits, limits.max_rational_bits)?;
        charge_bound(row.ub, &mut rational_bits, limits.max_rational_bits)?;
        row_entries = row_entries
            .checked_add(row.coeffs.len())
            .ok_or(ContinuousRootProblemResourceRejection::RowEntries)?;
        if row_entries > limits.max_row_entries {
            return Err(ContinuousRootProblemResourceRejection::RowEntries);
        }
        for &(column, coefficient) in &row.coeffs {
            if column >= problem.num_cols() {
                return Err(ContinuousRootProblemResourceRejection::MalformedModel);
            }
            charge(coefficient, &mut rational_bits, limits.max_rational_bits)?;
        }
    }

    Ok(ContinuousRootProblemResourceUsage {
        columns,
        rows,
        row_entries,
        rational_bits,
    })
}

fn rat(value: &BigRational) -> Result<Rat, MipError> {
    Rat::from_bigints(value.numer().clone(), value.denom().clone())
        .map_err(|error| MipError::Solver(format!("ny-cert rational conversion failed: {error}")))
}

fn variable_name(index: usize) -> String {
    format!("x{index}")
}

/// Match `ay_milp::Model::add_row`'s duplicate merge exactly.
fn canonical_row_coefficients(row: &RowSpec) -> Result<Vec<(usize, f64)>, MipError> {
    let mut coeffs = row.coeffs.clone();
    coeffs.sort_unstable_by_key(|&(col, _)| col);
    coeffs.dedup_by(|later, first| {
        if later.0 == first.0 {
            first.1 += later.1;
            true
        } else {
            false
        }
    });
    if coeffs.iter().any(|&(_, coeff)| !coeff.is_finite()) {
        return Err(MipError::Encoding(
            "row duplicate merge produced a non-finite coefficient".to_owned(),
        ));
    }
    coeffs.retain(|&(_, coeff)| coeff != 0.0);
    Ok(coeffs)
}

fn linear_constraint(
    kind: ConstraintKind,
    coeffs: impl IntoIterator<Item = (usize, BigRational)>,
    constant: &BigRational,
) -> Result<LinearConstraint, MipError> {
    let mut coefficients = BTreeMap::new();
    for (index, coeff) in coeffs {
        coefficients.insert(variable_name(index), rat(&coeff)?);
    }
    Ok(LinearConstraint {
        kind,
        coefficients,
        constant: rat(constant)?,
    })
}

fn fact_constraint(
    problem: &MilpProblem,
    fact: FactRef,
    effective_lb: &[Option<BigRational>],
    effective_ub: &[Option<BigRational>],
) -> Result<LinearConstraint, MipError> {
    match fact {
        FactRef::RowBound { row, side } => {
            let row_index = row.index();
            let row = problem.rows().get(row_index).ok_or_else(|| {
                MipError::Solver(format!("AY certificate references missing row {row_index}"))
            })?;
            let coeffs = canonical_row_coefficients(row)?
                .into_iter()
                .map(|(index, coeff)| Ok((index, exact_f64(coeff, "row coefficient")?)))
                .collect::<Result<Vec<_>, MipError>>()?;
            match side {
                BoundSide::Lower => linear_constraint(
                    ConstraintKind::Ge,
                    coeffs,
                    &exact_f64(row.lb, "row lower bound")?,
                ),
                BoundSide::Upper => linear_constraint(
                    ConstraintKind::Le,
                    coeffs,
                    &exact_f64(row.ub, "row upper bound")?,
                ),
            }
        }
        FactRef::ColBound { col, side } => {
            let index = col.index();
            if index >= problem.num_cols() {
                return Err(MipError::Solver(format!(
                    "AY certificate references missing column {index}"
                )));
            }
            let bound = match side {
                BoundSide::Lower => effective_lb.get(index).and_then(Clone::clone),
                BoundSide::Upper => effective_ub.get(index).and_then(Clone::clone),
            }
            .ok_or_else(|| {
                MipError::Solver(format!(
                    "AY certificate references an infinite effective column bound at {index}"
                ))
            })?;
            linear_constraint(
                match side {
                    BoundSide::Lower => ConstraintKind::Ge,
                    BoundSide::Upper => ConstraintKind::Le,
                },
                [(index, BigRational::from_integer(1.into()))],
                &bound,
            )
        }
        _ => Err(MipError::Solver(
            "AY certificate references an unsupported fact kind".to_owned(),
        )),
    }
}

fn replay_farkas_with_ny_cert(
    problem: &MilpProblem,
    cert: &AyFarkasCertificate,
    effective_lb: &[Option<BigRational>],
    effective_ub: &[Option<BigRational>],
) -> Result<(), MipError> {
    if ny_cert::rational::poisoned() {
        return Err(MipError::Solver(
            "ny-cert rational arena was already poisoned".to_owned(),
        ));
    }
    let mut constraints = Vec::with_capacity(cert.multipliers.len());
    let mut multipliers = Vec::with_capacity(cert.multipliers.len());
    for multiplier in &cert.multipliers {
        if multiplier.coeff <= BigRational::zero() {
            return Err(MipError::Solver(
                "AY certificate contains a nonpositive multiplier".to_owned(),
            ));
        }
        constraints.push(fact_constraint(
            problem,
            multiplier.fact,
            effective_lb,
            effective_ub,
        )?);
        multipliers.push(rat(&multiplier.coeff)?);
    }
    let cert = NyFarkasCertificate {
        constraints,
        multipliers,
    };
    check_farkas(&cert)
        .map_err(|error| MipError::Solver(format!("ny-cert Farkas replay failed: {error}")))?;
    if ny_cert::rational::poisoned() {
        return Err(MipError::Solver(
            "ny-cert rational arena became poisoned during Farkas replay".to_owned(),
        ));
    }
    Ok(())
}

fn base_column_bounds(
    problem: &MilpProblem,
) -> Result<(Vec<Option<BigRational>>, Vec<Option<BigRational>>), MipError> {
    let mut lower = Vec::with_capacity(problem.num_cols());
    let mut upper = Vec::with_capacity(problem.num_cols());
    for (index, col) in problem.cols().iter().enumerate() {
        lower.push(if col.lb.is_finite() {
            Some(exact_f64(col.lb, &format!("column {index} lower bound"))?)
        } else {
            None
        });
        upper.push(if col.ub.is_finite() {
            Some(exact_f64(col.ub, &format!("column {index} upper bound"))?)
        } else {
            None
        });
    }
    Ok((lower, upper))
}

fn canonical_exact_objective(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
) -> Result<BTreeMap<usize, BigRational>, MipError> {
    let mut exact = BTreeMap::new();
    for &(col, coefficient) in objective {
        if col.0 >= problem.num_cols() {
            return Err(MipError::Encoding(format!(
                "relaxation objective references missing column {}",
                col.0
            )));
        }
        let coefficient = exact_f64(coefficient, "relaxation objective coefficient")?;
        if exact.insert(col.0, coefficient).is_some() {
            return Err(MipError::Encoding(
                "relaxation objective contains duplicate columns".to_owned(),
            ));
        }
    }
    Ok(exact)
}

fn canonical_certified_row(
    problem: &MilpProblem,
    row: &AyCertifiedRow,
) -> Result<BTreeMap<usize, BigRational>, MipError> {
    let mut exact = BTreeMap::<usize, BigRational>::new();
    for &(col, ref coefficient) in &row.coeffs {
        let col = usize::try_from(col).map_err(|_| {
            MipError::Solver("AY certified-row column does not fit usize".to_owned())
        })?;
        if col >= problem.num_cols() {
            return Err(MipError::Solver(format!(
                "AY certified row references missing column {col}"
            )));
        }
        *exact.entry(col).or_default() += coefficient;
    }
    exact.retain(|_, coefficient| !coefficient.is_zero());
    Ok(exact)
}

fn replay_entailment_with_ny_cert(
    problem: &MilpProblem,
    row: &AyCertifiedRow,
) -> Result<(), MipError> {
    if ny_cert::rational::poisoned() {
        return Err(MipError::Solver(
            "ny-cert rational arena was already poisoned".to_owned(),
        ));
    }
    let (lower, upper) = base_column_bounds(problem)?;
    let mut premises = Vec::with_capacity(row.multipliers.len());
    let mut multipliers = Vec::with_capacity(row.multipliers.len());
    for multiplier in &row.multipliers {
        if multiplier.coeff < BigRational::zero() {
            return Err(MipError::Solver(
                "AY certified row contains a negative multiplier".to_owned(),
            ));
        }
        if multiplier.coeff.is_zero() {
            continue;
        }
        premises.push(fact_constraint(problem, multiplier.fact, &lower, &upper)?);
        multipliers.push(rat(&multiplier.coeff)?);
    }
    let conclusion = linear_constraint(
        ConstraintKind::Ge,
        canonical_certified_row(problem, row)?,
        &row.lb,
    )?;
    let certificate = EntailmentCertificate {
        premises,
        multipliers,
        conclusion,
    };
    check_entailment(&certificate)
        .map_err(|error| MipError::Solver(format!("ny-cert entailment replay failed: {error}")))?;
    if ny_cert::rational::poisoned() {
        return Err(MipError::Solver(
            "ny-cert rational arena became poisoned during entailment replay".to_owned(),
        ));
    }
    Ok(())
}

fn replay_root_farkas(
    problem: &MilpProblem,
    cert: &AyFarkasCertificate,
) -> Result<ReplayStats, MipError> {
    let (lower, upper) = base_column_bounds(problem)?;
    replay_farkas_with_ny_cert(problem, cert, &lower, &upper)?;
    Ok(ReplayStats {
        proof_route: CertifiedLinearLowerProofRoute::RootFarkas,
        tree_leaves: 0,
        linear_replays: 1,
    })
}

fn continuous_root_farkas_resource_usage(
    problem: &MilpProblem,
    cert: &AyFarkasCertificate,
) -> Result<ContinuousRootFarkasResourceUsage, ContinuousRootFarkasResourceRejection> {
    continuous_root_farkas_resource_usage_with_limits(
        problem,
        cert,
        CONTINUOUS_ROOT_FARKAS_RESOURCE_LIMITS,
    )
}

fn continuous_root_farkas_resource_usage_with_limits(
    problem: &MilpProblem,
    cert: &AyFarkasCertificate,
    limits: ContinuousRootFarkasResourceLimits,
) -> Result<ContinuousRootFarkasResourceUsage, ContinuousRootFarkasResourceRejection> {
    if cert.multipliers.len() > limits.max_multipliers {
        return Err(ContinuousRootFarkasResourceRejection::MultiplierCount);
    }

    // ny-cert's independent replay first converts every finite base column
    // bound, even when the final certificate does not reference that bound.
    // Price that exact binary64-to-rational work before constructing proof IR.
    let mut model_rational_bits = 0_u64;
    for column in problem.cols() {
        for bound in [column.lb, column.ub] {
            if !bound.is_finite() {
                continue;
            }
            let bits = exact_f64_rational_bits(bound)
                .ok_or(ContinuousRootFarkasResourceRejection::ModelRationalBits)?;
            model_rational_bits = model_rational_bits
                .checked_add(bits)
                .ok_or(ContinuousRootFarkasResourceRejection::ModelRationalBits)?;
            if model_rational_bits > limits.max_model_rational_bits {
                return Err(ContinuousRootFarkasResourceRejection::ModelRationalBits);
            }
        }
    }
    let mut weighted_replay_work = model_rational_bits;
    if weighted_replay_work > limits.max_weighted_replay_work {
        return Err(ContinuousRootFarkasResourceRejection::WeightedReplayWork);
    }

    // First pass: reject duplicate or malformed fact identities and cap the
    // raw data that the canonical-row scan below could touch. This pass does no
    // exact linear combination and allocates only one entry per admitted fact.
    let mut seen_facts = HashSet::with_capacity(cert.multipliers.len());
    let mut rational_bits = 0_u64;
    let mut referenced_row_entries = 0_usize;
    let mut column_bound_facts = 0_usize;
    let mut referenced_rows = Vec::with_capacity(cert.multipliers.len());
    for (index, multiplier) in cert.multipliers.iter().enumerate() {
        let numerator_bits = multiplier.coeff.numer().bits();
        let denominator_bits = multiplier.coeff.denom().bits();
        if numerator_bits > limits.max_rational_component_bits
            || denominator_bits > limits.max_rational_component_bits
        {
            return Err(ContinuousRootFarkasResourceRejection::RationalComponentBits);
        }
        let value_bits = numerator_bits
            .checked_add(denominator_bits)
            .ok_or(ContinuousRootFarkasResourceRejection::RationalTotalBits)?;
        rational_bits = rational_bits
            .checked_add(value_bits)
            .ok_or(ContinuousRootFarkasResourceRejection::RationalTotalBits)?;
        if rational_bits > limits.max_total_rational_bits {
            return Err(ContinuousRootFarkasResourceRejection::RationalTotalBits);
        }
        if !seen_facts.insert(multiplier.fact) {
            return Err(ContinuousRootFarkasResourceRejection::DuplicateFact { index });
        }

        match multiplier.fact {
            FactRef::RowBound { row, side } => {
                let Some(row_spec) = problem.rows().get(row.index()) else {
                    return Err(ContinuousRootFarkasResourceRejection::MissingFact { index });
                };
                let bound = match side {
                    BoundSide::Lower => row_spec.lb,
                    BoundSide::Upper => row_spec.ub,
                };
                if !bound.is_finite() {
                    return Err(ContinuousRootFarkasResourceRejection::InfiniteFact { index });
                }
                if row_spec.coeffs.iter().any(|&(column, coefficient)| {
                    column >= problem.num_cols() || !coefficient.is_finite()
                }) {
                    return Err(ContinuousRootFarkasResourceRejection::MalformedRow { index });
                }
                referenced_row_entries = referenced_row_entries
                    .checked_add(row_spec.coeffs.len())
                    .ok_or(ContinuousRootFarkasResourceRejection::ReferencedRowEntries)?;
                if referenced_row_entries > limits.max_referenced_row_entries {
                    return Err(ContinuousRootFarkasResourceRejection::ReferencedRowEntries);
                }
                referenced_rows.push((index, row.index(), value_bits, bound));
            }
            FactRef::ColBound { col, side } => {
                let Some(column) = problem.cols().get(col.index()) else {
                    return Err(ContinuousRootFarkasResourceRejection::MissingFact { index });
                };
                let bound = match side {
                    BoundSide::Lower => column.lb,
                    BoundSide::Upper => column.ub,
                };
                if !bound.is_finite() {
                    return Err(ContinuousRootFarkasResourceRejection::InfiniteFact { index });
                }
                column_bound_facts = column_bound_facts
                    .checked_add(1)
                    .ok_or(ContinuousRootFarkasResourceRejection::ExpandedFactTerms)?;
                weighted_replay_work = weighted_replay_work
                    .checked_add(value_bits)
                    .ok_or(ContinuousRootFarkasResourceRejection::WeightedReplayWork)?;
                if weighted_replay_work > limits.max_weighted_replay_work {
                    return Err(ContinuousRootFarkasResourceRejection::WeightedReplayWork);
                }
            }
            _ => {
                return Err(ContinuousRootFarkasResourceRejection::UnsupportedFact { index });
            }
        }
    }

    // Second pass: count the exact canonical nonzeros both verifiers expand.
    // Cache by row identity because a finite range row may legitimately appear
    // once per side; its work is charged twice below, but canonicalizing it for
    // this admission check need not allocate twice.
    let mut row_nonzero_cache = BTreeMap::<usize, (usize, u64)>::new();
    let mut referenced_row_nonzeros = 0_usize;
    for (multiplier_index, row_index, multiplier_bits, bound) in referenced_rows {
        let (nonzeros, coefficient_bits) = if let Some(&cached) = row_nonzero_cache.get(&row_index)
        {
            cached
        } else {
            let row = &problem.rows()[row_index];
            let canonical = canonical_row_coefficients(row).map_err(|_| {
                ContinuousRootFarkasResourceRejection::MalformedRow {
                    index: multiplier_index,
                }
            })?;
            let mut coefficient_bits = 0_u64;
            for &(_, coefficient) in &canonical {
                let bits = exact_f64_rational_bits(coefficient).ok_or(
                    ContinuousRootFarkasResourceRejection::MalformedRow {
                        index: multiplier_index,
                    },
                )?;
                coefficient_bits = coefficient_bits
                    .checked_add(bits)
                    .ok_or(ContinuousRootFarkasResourceRejection::ModelRationalBits)?;
            }
            let cached = (canonical.len(), coefficient_bits);
            row_nonzero_cache.insert(row_index, cached);
            cached
        };
        referenced_row_nonzeros = referenced_row_nonzeros
            .checked_add(nonzeros)
            .ok_or(ContinuousRootFarkasResourceRejection::ReferencedRowNonzeros)?;
        if referenced_row_nonzeros > limits.max_referenced_row_nonzeros {
            return Err(ContinuousRootFarkasResourceRejection::ReferencedRowNonzeros);
        }

        let bound_bits = exact_f64_rational_bits(bound)
            .ok_or(ContinuousRootFarkasResourceRejection::ModelRationalBits)?;
        let fact_model_bits = coefficient_bits
            .checked_add(bound_bits)
            .ok_or(ContinuousRootFarkasResourceRejection::ModelRationalBits)?;
        model_rational_bits = model_rational_bits
            .checked_add(fact_model_bits)
            .ok_or(ContinuousRootFarkasResourceRejection::ModelRationalBits)?;
        if model_rational_bits > limits.max_model_rational_bits {
            return Err(ContinuousRootFarkasResourceRejection::ModelRationalBits);
        }
        let multiplier_work = u64::try_from(nonzeros.max(1))
            .ok()
            .and_then(|terms| terms.checked_mul(multiplier_bits))
            .ok_or(ContinuousRootFarkasResourceRejection::WeightedReplayWork)?;
        weighted_replay_work = weighted_replay_work
            .checked_add(fact_model_bits)
            .and_then(|work| work.checked_add(multiplier_work))
            .ok_or(ContinuousRootFarkasResourceRejection::WeightedReplayWork)?;
        if weighted_replay_work > limits.max_weighted_replay_work {
            return Err(ContinuousRootFarkasResourceRejection::WeightedReplayWork);
        }
    }

    let expanded_fact_terms = referenced_row_nonzeros
        .checked_add(column_bound_facts)
        .ok_or(ContinuousRootFarkasResourceRejection::ExpandedFactTerms)?;
    if expanded_fact_terms > limits.max_expanded_fact_terms {
        return Err(ContinuousRootFarkasResourceRejection::ExpandedFactTerms);
    }

    Ok(ContinuousRootFarkasResourceUsage {
        multipliers: cert.multipliers.len(),
        rational_bits,
        referenced_row_entries,
        referenced_row_nonzeros,
        column_bound_facts,
        expanded_fact_terms,
        model_rational_bits,
        weighted_replay_work,
    })
}

fn replay_tree_farkas(
    problem: &MilpProblem,
    root: &TreeNode,
    max_tree_leaves: usize,
) -> Result<ReplayStats, MipError> {
    let (mut lower, mut upper) = base_column_bounds(problem)?;
    enum Step<'a> {
        Visit(&'a TreeNode),
        Tighten {
            col: usize,
            upper: bool,
            to: Box<BigRational>,
            child: &'a TreeNode,
        },
        Restore {
            col: usize,
            upper: bool,
        },
    }
    let mut undo: Vec<Option<BigRational>> = Vec::new();
    let mut stack = vec![Step::Visit(root)];
    let mut leaves = 0usize;
    while let Some(step) = stack.pop() {
        match step {
            Step::Visit(TreeNode::Leaf { farkas }) => {
                leaves = leaves
                    .checked_add(1)
                    .ok_or_else(|| MipError::Solver("AY tree leaf count overflow".to_owned()))?;
                if leaves > max_tree_leaves {
                    return Err(MipError::Solver(format!(
                        "AY tree has more than the admitted {max_tree_leaves} leaves"
                    )));
                }
                replay_farkas_with_ny_cert(problem, farkas, &lower, &upper)?;
            }
            Step::Visit(TreeNode::Split { col, cut, lo, hi }) => {
                let index = col.index();
                let Some(spec) = problem.cols().get(index) else {
                    return Err(MipError::Solver(format!(
                        "AY tree splits on missing column {index}"
                    )));
                };
                if !spec.integer || !cut.is_integer() {
                    return Err(MipError::Solver(format!(
                        "AY tree has an invalid integer split on column {index}"
                    )));
                }
                stack.push(Step::Restore {
                    col: index,
                    upper: false,
                });
                stack.push(Step::Tighten {
                    col: index,
                    upper: false,
                    to: Box::new(cut.clone() + BigRational::from_integer(1.into())),
                    child: hi,
                });
                stack.push(Step::Restore {
                    col: index,
                    upper: true,
                });
                stack.push(Step::Tighten {
                    col: index,
                    upper: true,
                    to: Box::new(cut.clone()),
                    child: lo,
                });
            }
            Step::Tighten {
                col,
                upper: is_upper,
                to,
                child,
            } => {
                let to = *to;
                let slot = if is_upper {
                    &mut upper[col]
                } else {
                    &mut lower[col]
                };
                undo.push(slot.clone());
                *slot = Some(match slot.take() {
                    Some(previous) => {
                        if is_upper {
                            previous.min(to)
                        } else {
                            previous.max(to)
                        }
                    }
                    None => to,
                });
                stack.push(Step::Visit(child));
            }
            Step::Restore {
                col,
                upper: is_upper,
            } => {
                let previous = undo.pop().ok_or_else(|| {
                    MipError::Solver("unbalanced AY tree replay stack".to_owned())
                })?;
                if is_upper {
                    upper[col] = previous;
                } else {
                    lower[col] = previous;
                }
            }
        }
    }
    if leaves == 0 || !undo.is_empty() {
        return Err(MipError::Solver(
            "AY tree replay was empty or structurally unbalanced".to_owned(),
        ));
    }
    Ok(ReplayStats {
        proof_route: CertifiedLinearLowerProofRoute::TreeFarkas,
        tree_leaves: leaves,
        linear_replays: leaves,
    })
}

/// Replay a whole AY tree while charging every exact ny-cert leaf to one
/// caller-owned absolute deadline.
///
/// A leaf replay is indivisible, so an expiry observed immediately after a
/// checker call discards that late result and prevents traversal from reaching
/// any later leaf.  The detached outer worker independently prevents a late
/// result from reaching the caller while an individual exact-arithmetic call
/// is still running.
fn replay_tree_farkas_until(
    problem: &MilpProblem,
    root: &TreeNode,
    max_tree_leaves: usize,
    deadline: Instant,
) -> Result<Option<ReplayStats>, MipError> {
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }
    let (mut lower, mut upper) = base_column_bounds(problem)?;
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }
    enum Step<'a> {
        Visit(&'a TreeNode),
        Tighten {
            col: usize,
            upper: bool,
            to: Box<BigRational>,
            child: &'a TreeNode,
        },
        Restore {
            col: usize,
            upper: bool,
        },
    }
    let mut undo: Vec<Option<BigRational>> = Vec::new();
    let mut stack = vec![Step::Visit(root)];
    let mut leaves = 0usize;
    while let Some(step) = stack.pop() {
        if !fixed_assignment_tree_deadline_open(deadline) {
            return Ok(None);
        }
        match step {
            Step::Visit(TreeNode::Leaf { farkas }) => {
                leaves = leaves
                    .checked_add(1)
                    .ok_or_else(|| MipError::Solver("AY tree leaf count overflow".to_owned()))?;
                if leaves > max_tree_leaves {
                    return Err(MipError::Solver(format!(
                        "AY tree has more than the admitted {max_tree_leaves} leaves"
                    )));
                }
                if !fixed_assignment_tree_deadline_open(deadline) {
                    return Ok(None);
                }
                replay_farkas_with_ny_cert(problem, farkas, &lower, &upper)?;
                if !fixed_assignment_tree_deadline_open(deadline) {
                    return Ok(None);
                }
            }
            Step::Visit(TreeNode::Split { col, cut, lo, hi }) => {
                let index = col.index();
                let Some(spec) = problem.cols().get(index) else {
                    return Err(MipError::Solver(format!(
                        "AY tree splits on missing column {index}"
                    )));
                };
                if !spec.integer || !cut.is_integer() {
                    return Err(MipError::Solver(format!(
                        "AY tree has an invalid integer split on column {index}"
                    )));
                }
                stack.push(Step::Restore {
                    col: index,
                    upper: false,
                });
                stack.push(Step::Tighten {
                    col: index,
                    upper: false,
                    to: Box::new(cut.clone() + BigRational::from_integer(1.into())),
                    child: hi,
                });
                stack.push(Step::Restore {
                    col: index,
                    upper: true,
                });
                stack.push(Step::Tighten {
                    col: index,
                    upper: true,
                    to: Box::new(cut.clone()),
                    child: lo,
                });
            }
            Step::Tighten {
                col,
                upper: is_upper,
                to,
                child,
            } => {
                let to = *to;
                let slot = if is_upper {
                    &mut upper[col]
                } else {
                    &mut lower[col]
                };
                undo.push(slot.clone());
                *slot = Some(match slot.take() {
                    Some(previous) => {
                        if is_upper {
                            previous.min(to)
                        } else {
                            previous.max(to)
                        }
                    }
                    None => to,
                });
                stack.push(Step::Visit(child));
            }
            Step::Restore {
                col,
                upper: is_upper,
            } => {
                let previous = undo.pop().ok_or_else(|| {
                    MipError::Solver("unbalanced AY tree replay stack".to_owned())
                })?;
                if is_upper {
                    upper[col] = previous;
                } else {
                    lower[col] = previous;
                }
            }
        }
    }
    if leaves == 0 || !undo.is_empty() {
        return Err(MipError::Solver(
            "AY tree replay was empty or structurally unbalanced".to_owned(),
        ));
    }
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }
    Ok(Some(ReplayStats {
        proof_route: CertifiedLinearLowerProofRoute::TreeFarkas,
        tree_leaves: leaves,
        linear_replays: leaves,
    }))
}

fn solve_proposal(
    problem: MilpProblem,
    objective: Vec<(Col, f64)>,
    timeout_secs: f64,
    worker_lease: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<BigRational>, MipError> {
    run_with_hard_deadline(timeout_secs, "linear-lower-proposal", move || {
        // If the hard deadline detaches this worker, retain process-wide
        // admission until the AY session actually exits.
        let _worker_lease = worker_lease;
        let mut model = to_ay_model(&problem)?;
        let mut ay_objective = Vec::with_capacity(objective.len());
        for (col, coeff) in objective {
            let ay_col = model.col_at(col.0).ok_or_else(|| {
                MipError::Encoding(format!("objective column {} disappeared", col.0))
            })?;
            ay_objective.push((ay_col, coeff));
        }
        model.set_objective(&ay_objective, AySense::Minimize);
        let opts = solve_opts(timeout_secs).with_tree_cert_leaves(0);
        let mut session =
            BabSession::new(model, &opts).map_err(|error| MipError::Solver(error.to_string()))?;
        match session
            .check()
            .map_err(|error| MipError::Solver(error.to_string()))?
        {
            Outcome::Optimal { value, .. } => Ok(Some(value)),
            // AY d512e58b5 (pinned rev 8c623810): a branch-and-bound run that
            // is interrupted with NO incumbent but a rigorous frontier bound
            // now reports `Bound { rigorous: true }` where it previously
            // reported `Unknown { Timeout }`. This lane sets a real objective
            // (`set_objective`, above), so it is one of the few NY solves that
            // can observe it at all — AY withholds the outcome on
            // zero-objective feasibility models (`report_tree_bound =
            // !zero_objective`, ay bab.rs:27253).
            //
            // A dual bound `L <= v*` is a PERFECTLY VALID PROPOSAL here, just a
            // weaker one than the optimum: `strict_outward_f32_lower` returns a
            // threshold strictly below its input, so the proposed `lower`
            // satisfies `lower < L <= v*`, and the caller then re-establishes
            // it from scratch through `solve_and_replay_decision_proof`. Per
            // this module's contract (above): "The optimization answer
            // therefore has no authority. It may be arbitrarily wrong without
            // producing a bound." An unsound proposal cannot mint a bound — it
            // can only fail the replay and return `Ok(None)`, exactly as a
            // missing proposal does today.
            //
            // Gate on `rigorous` anyway. AY forbids using a non-rigorous bound
            // to exclude feasible points (ay outcome.rs:106-109), and spending
            // this lane's expensive `proof_timeout_secs` replay budget on a
            // heuristic number is a waste even though it is not a hazard.
            Outcome::Bound {
                dual_bound,
                rigorous: true,
            } => {
                tracing::debug!(
                    "linear-lower proposal from a rigorous interrupted-tree bound \
                     (no incumbent); weaker than the optimum, still independently replayed"
                );
                Ok(Some(dual_bound))
            }
            // Same shape, but the tree DID find an incumbent before it was
            // interrupted. `Feasible.dual_bound` is rigorous BY CONTRACT when
            // present (`None` whenever any part of the tree was discarded
            // without proof — ay contract property 3), so `Some` is safe to
            // propose on. The incumbent's own objective is an UPPER bound on
            // the optimum and is deliberately NOT used here.
            Outcome::Feasible {
                dual_bound: Some(dual_bound),
                ..
            } => {
                tracing::debug!(
                    "linear-lower proposal from an interrupted tree's rigorous dual bound \
                     (incumbent present)"
                );
                Ok(Some(dual_bound))
            }
            other => {
                tracing::debug!("linear-lower proposal declined: no rigorous bound ({other:?})");
                Ok(None)
            }
        }
    })
    .map(Option::flatten)
}

/// `NY_MIP_TRACE` — dark, print-only harvest/report tracing (lever-debt batch
/// B1 preparation; declared as `ny_levers::decls::telemetry::MIP_TRACE`).
///
/// PRESENCE gate: ANY present value arms it — including `"0"`, the empty
/// string, and non-UTF-8 — exactly like the historical
/// `mip_trace_armed()` sites this helper collapses.
/// The lookup therefore goes through `var_os` plus a lossy conversion rather
/// than the chokepoint's plain `env::var`, which would fold a present
/// non-UTF-8 value into "absent" and disarm a trace the operator set.
/// Deliberately resolved PER CALL, not latched: the historical sites re-read
/// the environment on every call, and these report paths run per MIP
/// solve/report, where one env lookup is noise. This remains live process state
/// until Phase 2 injects a per-run `LeverSet`.
fn mip_trace_armed() -> bool {
    !matches!(
        ny_levers::read_with(&ny_levers::decls::telemetry::MIP_TRACE, |name| {
            std::env::var_os(name).map(|value| value.to_string_lossy().into_owned())
        })
        .value,
        ny_levers::LeverValue::Unset
    )
}

fn replay_relaxation_root_entailment(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    relaxed_model: &ay_milp::Model,
    requested: &BigRational,
    row: AyCertifiedRow,
    trace_context: &str,
) -> Result<ReplayStats, MipError> {
    if mip_trace_armed() {
        eprintln!(
            "NY_MIP_TRACE {trace_context}: lb={}/{} requested={}/{} multipliers={} sufficient={}",
            row.lb.numer(),
            row.lb.denom(),
            requested.numer(),
            requested.denom(),
            row.multipliers.len(),
            row.lb > *requested,
        );
    }
    if row.lb <= *requested {
        return Err(MipError::Solver(format!(
            "AY {trace_context} did not strictly clear the requested threshold"
        )));
    }
    row.verify(relaxed_model).map_err(|error| {
        MipError::Solver(format!(
            "AY {trace_context} failed independent verification: {error}"
        ))
    })?;
    let expected = canonical_exact_objective(problem, objective)?;
    let actual = canonical_certified_row(problem, &row)?;
    if actual != expected {
        return Err(MipError::Solver(format!(
            "AY {trace_context} does not match the requested objective"
        )));
    }
    replay_entailment_with_ny_cert(problem, &row)?;
    Ok(ReplayStats {
        proof_route: CertifiedLinearLowerProofRoute::RelaxationEntailment,
        tree_leaves: 0,
        linear_replays: 1,
    })
}

fn replay_relaxation_root_entailment_until(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    relaxed_model: &ay_milp::Model,
    requested: &BigRational,
    row: AyCertifiedRow,
    trace_context: &str,
    deadline: Instant,
) -> Result<Option<ReplayStats>, MipError> {
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }
    if mip_trace_armed() {
        eprintln!(
            "NY_MIP_TRACE {trace_context}: lb={}/{} requested={}/{} multipliers={} sufficient={}",
            row.lb.numer(),
            row.lb.denom(),
            requested.numer(),
            requested.denom(),
            row.multipliers.len(),
            row.lb > *requested,
        );
    }
    if row.lb <= *requested {
        return Err(MipError::Solver(format!(
            "AY {trace_context} did not strictly clear the requested threshold"
        )));
    }
    row.verify(relaxed_model).map_err(|error| {
        MipError::Solver(format!(
            "AY {trace_context} failed independent verification: {error}"
        ))
    })?;
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }
    let expected = canonical_exact_objective(problem, objective)?;
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }
    let actual = canonical_certified_row(problem, &row)?;
    if actual != expected {
        return Err(MipError::Solver(format!(
            "AY {trace_context} does not match the requested objective"
        )));
    }
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }
    replay_entailment_with_ny_cert(problem, &row)?;
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }
    Ok(Some(ReplayStats {
        proof_route: CertifiedLinearLowerProofRoute::RelaxationEntailment,
        tree_leaves: 0,
        linear_replays: 1,
    }))
}

#[derive(Debug)]
enum ParallelSelectorLeafEvidence {
    ConditionalRow(Box<AyCertifiedRow>),
    Infeasible(AyFarkasCertificate),
}

#[derive(Debug)]
struct IndexedParallelSelectorLeaf {
    assignment: usize,
    evidence: ParallelSelectorLeafEvidence,
}

/// Run canonical assignment jobs through a bounded scoped pool.
///
/// The pool stores successful values by assignment index, never completion
/// order. A single absolute deadline is checked before dispatch and after
/// every worker call. Panics, weak/missing results, duplicate writes, poisoned
/// coordination state, or a late completion all decline the entire batch.
/// Scoped workers are joined before return, which is what keeps the outer
/// exact-worker admission alive across every inner worker lifetime. Every
/// worker receives the same explicit 64 MiB stack as ny-mip's detached AY
/// worker. Among concurrently completed errors, the lowest canonical
/// assignment wins deterministically and is never masked by a decline.
fn run_bounded_canonical_assignment_workers<T, F>(
    assignment_count: usize,
    max_workers: usize,
    deadline: Instant,
    worker: F,
) -> Result<Option<Vec<T>>, MipError>
where
    T: Send,
    F: Fn(usize) -> Result<Option<T>, MipError> + Sync,
{
    if assignment_count == 0 || max_workers == 0 || Instant::now() >= deadline {
        return Ok(None);
    }
    let worker_count = assignment_count.min(max_workers);
    let next_assignment = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let declined = AtomicBool::new(false);
    let results = Mutex::new(
        (0..assignment_count)
            .map(|_| None)
            .collect::<Vec<Option<T>>>(),
    );
    let errors = Mutex::new(
        (0..assignment_count)
            .map(|_| None)
            .collect::<Vec<Option<MipError>>>(),
    );
    let mut spawn_error = None;

    std::thread::scope(|scope| {
        for worker_index in 0..worker_count {
            let worker = &worker;
            let next_assignment = &next_assignment;
            let stop = &stop;
            let declined = &declined;
            let results = &results;
            let errors = &errors;
            let spawned = std::thread::Builder::new()
                .name(format!("ny-mip-ay-selector-leaf-{worker_index}"))
                .stack_size(SOLVE_THREAD_STACK_BYTES)
                .spawn_scoped(scope, move || loop {
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    let assignment = next_assignment.fetch_add(1, Ordering::Relaxed);
                    if assignment >= assignment_count {
                        break;
                    }
                    if Instant::now() >= deadline {
                        declined.store(true, Ordering::Release);
                        stop.store(true, Ordering::Release);
                        break;
                    }

                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        worker(assignment)
                    }));
                    let completed_late = Instant::now() >= deadline;
                    match outcome {
                        Ok(Ok(Some(value))) if !completed_late => match results.lock() {
                            Ok(mut slots) if slots[assignment].is_none() => {
                                slots[assignment] = Some(value);
                            }
                            _ => {
                                declined.store(true, Ordering::Release);
                                stop.store(true, Ordering::Release);
                                break;
                            }
                        },
                        Ok(Ok(Some(_))) | Ok(Ok(None)) | Err(_) => {
                            declined.store(true, Ordering::Release);
                            stop.store(true, Ordering::Release);
                            break;
                        }
                        Ok(Err(error)) => {
                            match errors.lock() {
                                Ok(mut slots) if slots[assignment].is_none() => {
                                    slots[assignment] = Some(error);
                                }
                                _ => declined.store(true, Ordering::Release),
                            }
                            stop.store(true, Ordering::Release);
                            break;
                        }
                    }
                });
            if let Err(error) = spawned {
                stop.store(true, Ordering::Release);
                spawn_error = Some(MipError::Solver(format!(
                    "spawning parallel AY selector worker {worker_index}: {error}"
                )));
                break;
            }
        }
    });

    if let Some(error) = spawn_error {
        return Err(error);
    }
    let mut errors = match errors.into_inner() {
        Ok(errors) => errors,
        Err(_) => return Ok(None),
    };
    for error in &mut errors {
        if let Some(error) = error.take() {
            return Err(error);
        }
    }
    if Instant::now() >= deadline || declined.load(Ordering::Acquire) {
        return Ok(None);
    }
    let mut slots = match results.into_inner() {
        Ok(slots) => slots,
        Err(_) => return Ok(None),
    };
    let mut canonical = Vec::with_capacity(assignment_count);
    for slot in &mut slots {
        let Some(value) = slot.take() else {
            return Ok(None);
        };
        canonical.push(value);
    }
    Ok(Some(canonical))
}

fn selector_assignment_value(assignment: usize, selector_index: usize) -> bool {
    let shift = PARALLEL_SELECTOR_TREE_DEPTH - 1 - selector_index;
    assignment & (1usize << shift) != 0
}

fn canonical_assignment_tree(
    split_cols: &[ay_milp::Col],
    leaves: &mut impl Iterator<Item = AyFarkasCertificate>,
) -> Result<TreeNode, MipError> {
    let Some((&split, remaining)) = split_cols.split_first() else {
        return leaves
            .next()
            .map(|farkas| TreeNode::Leaf { farkas })
            .ok_or_else(|| {
                MipError::Solver(
                    "parallel selector-tree composition omitted a canonical leaf".to_owned(),
                )
            });
    };
    let lo = canonical_assignment_tree(remaining, leaves)?;
    let hi = canonical_assignment_tree(remaining, leaves)?;
    Ok(TreeNode::Split {
        col: split,
        cut: BigRational::zero(),
        lo: Box::new(lo),
        hi: Box::new(hi),
    })
}

fn compose_and_replay_parallel_selector_tree(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    selectors: &[Col],
    ay_selectors: &[ay_milp::Col],
    leaves: Vec<IndexedParallelSelectorLeaf>,
    max_tree_leaves: usize,
) -> Result<ReplayStats, MipError> {
    if selectors.len() != PARALLEL_SELECTOR_TREE_DEPTH
        || ay_selectors.len() != PARALLEL_SELECTOR_TREE_DEPTH
        || leaves.len() != PARALLEL_SELECTOR_TREE_LEAVES
        || max_tree_leaves < PARALLEL_SELECTOR_TREE_LEAVES
    {
        return Err(MipError::Solver(
            "parallel selector-tree composition received the wrong topology".to_owned(),
        ));
    }

    let requested = exact_f64(
        f64::from(requested_lower),
        "requested certified lower threshold",
    )?;
    let expected_objective = canonical_exact_objective(problem, objective)?;
    let decision = linear_lower_decision_problem(problem, objective, requested_lower);
    let decision_model = to_ay_model(&decision)?;
    let decision_row = decision_model.row_at(problem.num_rows()).ok_or_else(|| {
        MipError::Encoding(
            "appended parallel selector-tree decision row disappeared during AY lowering"
                .to_owned(),
        )
    })?;

    let mut farkas_leaves = Vec::with_capacity(PARALLEL_SELECTOR_TREE_LEAVES);
    for (canonical_assignment, leaf) in leaves.into_iter().enumerate() {
        if leaf.assignment != canonical_assignment {
            return Err(MipError::Solver(format!(
                "parallel selector-tree leaf {} claimed assignment {}",
                canonical_assignment, leaf.assignment
            )));
        }
        let branch_bounds = ay_selectors
            .iter()
            .enumerate()
            .map(|(selector_index, &selector)| {
                if selector_assignment_value(canonical_assignment, selector_index) {
                    (
                        selector,
                        BoundSide::Lower,
                        BigRational::from_integer(1.into()),
                    )
                } else {
                    (selector, BoundSide::Upper, BigRational::zero())
                }
            })
            .collect::<Vec<_>>();
        let farkas = match leaf.evidence {
            ParallelSelectorLeafEvidence::ConditionalRow(row) => {
                if row.lb <= requested {
                    return Err(MipError::Solver(format!(
                        "parallel selector-tree assignment {canonical_assignment:04b} returned a \
                         non-strict conditional row"
                    )));
                }
                if canonical_certified_row(problem, &row)? != expected_objective {
                    return Err(MipError::Solver(format!(
                        "parallel selector-tree assignment {canonical_assignment:04b} changed the \
                         requested objective"
                    )));
                }
                (*row)
                    .into_farkas_against_row_upper(&decision_model, decision_row, &branch_bounds)
                    .ok_or_else(|| {
                        MipError::Solver(format!(
                            "parallel selector-tree assignment {canonical_assignment:04b} failed \
                         exact decision-row composition"
                        ))
                    })?
            }
            ParallelSelectorLeafEvidence::Infeasible(farkas) => farkas,
        };
        farkas_leaves.push(farkas);
    }

    let mut farkas_iter = farkas_leaves.into_iter();
    let root = canonical_assignment_tree(ay_selectors, &mut farkas_iter)?;
    if farkas_iter.next().is_some() {
        return Err(MipError::Solver(
            "parallel selector-tree composition retained extra leaves".to_owned(),
        ));
    }
    let certificate = MilpInfeasibilityCertificate { root };
    certificate.verify(&decision_model).map_err(|error| {
        MipError::Solver(format!(
            "AY parallel selector-tree certificate failed independent verification: {error}"
        ))
    })?;
    replay_tree_farkas(&decision, &certificate.root, max_tree_leaves)
}

fn compose_and_replay_assignment_tree(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    selected: &[Col],
    ay_splits: &[ay_milp::Col],
    tree: CertifiedBinaryAssignmentTree,
    max_tree_leaves: usize,
    trace_context: &str,
) -> Result<ReplayStats, MipError> {
    if !(1..=FIXED_ASSIGNMENT_TREE_MAX_DEPTH).contains(&selected.len())
        || selected.len() != ay_splits.len()
        || tree.split_cols() != ay_splits
    {
        return Err(MipError::Solver(format!(
            "AY {trace_context} changed the selected split columns"
        )));
    }
    let expected_leaves = 1usize << selected.len();
    if tree.num_leaves() != expected_leaves || tree.num_leaves() > max_tree_leaves {
        return Err(MipError::Solver(format!(
            "AY {trace_context} returned {} leaves, expected {expected_leaves} and admitted \
             maximum is {max_tree_leaves}",
            tree.num_leaves(),
        )));
    }
    if mip_trace_armed() {
        let cols = selected
            .iter()
            .map(|col| col.0.to_string())
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "NY_MIP_TRACE {trace_context}: cols=[{cols}] leaves={}",
            tree.num_leaves()
        );
    }

    // The harvest names only the base relaxation's rows and columns.
    // Restoring integrality and appending the exact decision row preserves
    // those identities. AY owns the assignment-to-leaf association and
    // verifies the composed complete-assignment certificate before exposing it
    // to NY.
    let decision = linear_lower_decision_problem(problem, objective, requested_lower);
    let decision_model = to_ay_model(&decision)?;
    let decision_row = decision_model.row_at(problem.num_rows()).ok_or_else(|| {
        MipError::Encoding(
            "appended linear-lower decision row disappeared during AY lowering".to_owned(),
        )
    })?;
    let certificate = tree
        .into_farkas_against_row_upper(&decision_model, decision_row)
        .ok_or_else(|| {
            MipError::Solver(format!(
                "AY {trace_context} failed exact decision-row composition"
            ))
        })?;
    certificate.verify(&decision_model).map_err(|error| {
        MipError::Solver(format!(
            "AY {trace_context} certificate failed independent verification: {error}"
        ))
    })?;
    replay_tree_farkas(&decision, &certificate.root, max_tree_leaves)
}

#[allow(clippy::too_many_arguments)]
fn compose_and_replay_assignment_tree_until(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    selected: &[Col],
    ay_splits: &[ay_milp::Col],
    tree: CertifiedBinaryAssignmentTree,
    max_tree_leaves: usize,
    trace_context: &str,
    deadline: Instant,
) -> Result<Option<ReplayStats>, MipError> {
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }
    if !(1..=FIXED_ASSIGNMENT_TREE_MAX_DEPTH).contains(&selected.len())
        || selected.len() != ay_splits.len()
        || tree.split_cols() != ay_splits
    {
        return Err(MipError::Solver(format!(
            "AY {trace_context} changed the selected split columns"
        )));
    }
    let expected_leaves = 1usize << selected.len();
    if tree.num_leaves() != expected_leaves || tree.num_leaves() > max_tree_leaves {
        return Err(MipError::Solver(format!(
            "AY {trace_context} returned {} leaves, expected {expected_leaves} and admitted \
             maximum is {max_tree_leaves}",
            tree.num_leaves(),
        )));
    }
    if mip_trace_armed() {
        let cols = selected
            .iter()
            .map(|col| col.0.to_string())
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "NY_MIP_TRACE {trace_context}: cols=[{cols}] leaves={}",
            tree.num_leaves()
        );
    }
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }

    // The original absolute clock covers every allocation and exact
    // validation below: construction of the decision clone, AY lowering,
    // exact row composition, whole-tree verification, and leaf-by-leaf
    // ny-cert replay.
    let decision = linear_lower_decision_problem(problem, objective, requested_lower);
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }
    let decision_model = to_ay_model(&decision)?;
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }
    let decision_row = decision_model.row_at(problem.num_rows()).ok_or_else(|| {
        MipError::Encoding(
            "appended linear-lower decision row disappeared during AY lowering".to_owned(),
        )
    })?;
    let certificate = tree
        .into_farkas_against_row_upper(&decision_model, decision_row)
        .ok_or_else(|| {
            MipError::Solver(format!(
                "AY {trace_context} failed exact decision-row composition"
            ))
        })?;
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }
    certificate.verify(&decision_model).map_err(|error| {
        MipError::Solver(format!(
            "AY {trace_context} certificate failed independent verification: {error}"
        ))
    })?;
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }
    replay_tree_farkas_until(&decision, &certificate.root, max_tree_leaves, deadline)
}

fn adaptive_target_fsb_opts() -> TargetFsbOpts {
    let limits = CertifiedLinearLowerTargetFsbProbeLimits::production();
    TargetFsbOpts::new()
        .with_max_probe_pivots_per_call(limits.max_probe_pivots_per_call())
        .with_max_probe_calls(TARGET_FSB_MAX_PROBE_CALLS)
        .with_probe_time_limit(limits.probe_time_limit())
        .with_max_probe_scratch_bytes(TARGET_FSB_MAX_PROBE_SCRATCH_BYTES)
}

fn validate_adaptive_three_leaf_report_and_tree(
    candidates: &[ay_milp::Col],
    root_candidate_index: usize,
    hard_value: bool,
    report: &AdaptiveThreeLeafTargetFsbReport,
    tree: &CertifiedAdaptiveThreeLeafTree,
) -> Result<ay_milp::Col, MipError> {
    let root_split = candidates[root_candidate_index];
    let expected_probe_calls = candidates
        .len()
        .checked_sub(1)
        .and_then(|remaining| remaining.checked_mul(2))
        .ok_or_else(|| MipError::Solver("adaptive three-leaf probe-count overflow".to_owned()))?;
    let Some(second_candidate_index) = report.second_candidate_index() else {
        return Err(MipError::Solver(
            "AY adaptive three-leaf tree report omitted the second candidate index".to_owned(),
        ));
    };
    let Some(second_split) = report.second_split() else {
        return Err(MipError::Solver(
            "AY adaptive three-leaf tree report omitted the second split".to_owned(),
        ));
    };
    if report.candidate_count() != candidates.len()
        || report.probe_calls() != expected_probe_calls
        || report.root_candidate_index() != root_candidate_index
        || report.root_split() != root_split
        || report.hard_value() != hard_value
        || report.hard_grandchild_lower_bounds().is_none()
    {
        return Err(MipError::Solver(format!(
            "AY adaptive three-leaf tree report is inconsistent: candidates={}/{} probes={}/{} \
             root={}/{} hard={}/{} grandchild_bounds={}",
            report.candidate_count(),
            candidates.len(),
            report.probe_calls(),
            expected_probe_calls,
            report.root_split().index(),
            root_split.index(),
            report.hard_value(),
            hard_value,
            report.hard_grandchild_lower_bounds().is_some(),
        )));
    }
    if second_candidate_index == root_candidate_index
        || candidates.get(second_candidate_index) != Some(&second_split)
        || second_split == root_split
    {
        return Err(MipError::Solver(format!(
            "AY adaptive three-leaf report selected invalid second candidate \
             {second_candidate_index} at column {}",
            second_split.index(),
        )));
    }
    if tree.num_leaves() != 3
        || tree.root_split() != root_split
        || tree.hard_value() != hard_value
        || tree.second_split() != second_split
    {
        return Err(MipError::Solver(format!(
            "AY adaptive three-leaf carrier is inconsistent: leaves={} root={}/{} hard={}/{} \
             second={}/{}",
            tree.num_leaves(),
            tree.root_split().index(),
            root_split.index(),
            tree.hard_value(),
            hard_value,
            tree.second_split().index(),
            second_split.index(),
        )));
    }
    Ok(second_split)
}

fn validate_adaptive_three_leaf_certificate_shape(
    root: &TreeNode,
    root_split: ay_milp::Col,
    hard_value: bool,
    second_split: ay_milp::Col,
) -> Result<(), MipError> {
    let TreeNode::Split { col, cut, lo, hi } = root else {
        return Err(MipError::Solver(
            "AY adaptive three-leaf certificate has no root split".to_owned(),
        ));
    };
    if *col != root_split || !cut.is_zero() {
        return Err(MipError::Solver(format!(
            "AY adaptive three-leaf certificate changed root split {}/{} or its binary cut",
            col.index(),
            root_split.index()
        )));
    }

    let (sibling, hard_child) = if hard_value {
        (&**lo, &**hi)
    } else {
        (&**hi, &**lo)
    };
    if !matches!(sibling, TreeNode::Leaf { .. }) {
        return Err(MipError::Solver(
            "AY adaptive three-leaf certificate split the easy root sibling".to_owned(),
        ));
    }
    let TreeNode::Split { col, cut, lo, hi } = hard_child else {
        return Err(MipError::Solver(
            "AY adaptive three-leaf certificate did not split the hard root child".to_owned(),
        ));
    };
    if *col != second_split
        || !cut.is_zero()
        || !matches!(&**lo, TreeNode::Leaf { .. })
        || !matches!(&**hi, TreeNode::Leaf { .. })
    {
        return Err(MipError::Solver(format!(
            "AY adaptive three-leaf certificate changed second split {}/{} or its two-leaf shape",
            col.index(),
            second_split.index()
        )));
    }
    Ok(())
}

fn compose_and_replay_adaptive_three_leaf_tree(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    root_split: ay_milp::Col,
    hard_value: bool,
    second_split: ay_milp::Col,
    tree: CertifiedAdaptiveThreeLeafTree,
    max_tree_leaves: usize,
) -> Result<ReplayStats, MipError> {
    if tree.num_leaves() != 3 || max_tree_leaves < 3 {
        return Err(MipError::Solver(format!(
            "AY adaptive three-leaf harvest returned {} leaves, admitted maximum is \
             {max_tree_leaves}",
            tree.num_leaves()
        )));
    }

    // The carrier may mix conditional rows with direct exact-Farkas leaves.
    // Its consuming composer owns that association and produces one ordinary
    // whole-tree certificate; NY deliberately validates only its public shape.
    let decision = linear_lower_decision_problem(problem, objective, requested_lower);
    let decision_model = to_ay_model(&decision)?;
    let decision_row = decision_model.row_at(problem.num_rows()).ok_or_else(|| {
        MipError::Encoding(
            "appended adaptive three-leaf decision row disappeared during AY lowering".to_owned(),
        )
    })?;
    let certificate = tree
        .into_farkas_against_row_upper(&decision_model, decision_row)
        .ok_or_else(|| {
            MipError::Solver(
                "AY adaptive three-leaf carrier failed exact decision-row composition".to_owned(),
            )
        })?;
    validate_adaptive_three_leaf_certificate_shape(
        &certificate.root,
        root_split,
        hard_value,
        second_split,
    )?;
    certificate.verify(&decision_model).map_err(|error| {
        MipError::Solver(format!(
            "AY adaptive three-leaf certificate failed independent whole-tree verification: \
             {error}"
        ))
    })?;
    let replay = replay_tree_farkas(&decision, &certificate.root, max_tree_leaves)?;
    if replay.tree_leaves != 3 || replay.linear_replays != 3 {
        return Err(MipError::Solver(format!(
            "NY adaptive three-leaf replay admitted {} leaves and {} linear obligations",
            replay.tree_leaves, replay.linear_replays
        )));
    }
    Ok(replay)
}

fn try_relaxed_linear_lower_adaptive_three_leaf_target_fsb(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    candidates: &[Col],
    root_candidate_index: usize,
    hard_value: bool,
    max_tree_leaves: usize,
    opts: &SolveOpts,
) -> Result<Option<ReplayStats>, MipError> {
    if max_tree_leaves < 3 {
        return Ok(None);
    }

    let mut relaxed_problem = problem.clone();
    relaxed_problem.relax_integrality();
    let relaxed_model = to_ay_model(&relaxed_problem)?;
    let ay_objective = objective
        .iter()
        .map(|&(col, coefficient)| {
            relaxed_model
                .col_at(col.0)
                .map(|ay_col| (ay_col, coefficient))
                .ok_or_else(|| {
                    MipError::Encoding(format!(
                        "adaptive three-leaf objective column {} disappeared",
                        col.0
                    ))
                })
        })
        .collect::<Result<Vec<_>, MipError>>()?;
    let ay_candidates = candidates
        .iter()
        .map(|candidate| {
            relaxed_model.col_at(candidate.0).ok_or_else(|| {
                MipError::Encoding(format!(
                    "adaptive three-leaf candidate column {} disappeared",
                    candidate.0
                ))
            })
        })
        .collect::<Result<Vec<_>, MipError>>()?;
    let root_split = ay_candidates[root_candidate_index];
    let requested = exact_f64(
        f64::from(requested_lower),
        "requested certified lower threshold",
    )?;
    let mut session = LpSession::new(&relaxed_model, opts)
        .map_err(|error| MipError::Solver(error.to_string()))?;
    let fsb_opts = adaptive_target_fsb_opts();
    let Some((harvest, report)) = session
        .harvest_cut_or_adaptive_three_leaf_target_fsb_stronger_than(
            &ay_objective,
            AySense::Minimize,
            &ay_candidates,
            root_candidate_index,
            hard_value,
            &requested,
            &fsb_opts,
        )
    else {
        return Ok(None);
    };

    if report.candidate_count() != candidates.len()
        || report.root_candidate_index() != root_candidate_index
        || report.root_split() != root_split
        || report.hard_value() != hard_value
    {
        return Err(MipError::Solver(format!(
            "AY adaptive three-leaf report changed the request: candidates={}/{} root={}/{} \
             root_index={}/{} hard={}/{}",
            report.candidate_count(),
            candidates.len(),
            report.root_split().index(),
            root_split.index(),
            report.root_candidate_index(),
            root_candidate_index,
            report.hard_value(),
            hard_value,
        )));
    }

    match harvest {
        CertifiedAdaptiveThreeLeafHarvest::Root(row) => {
            if report.probe_calls() != 0
                || report.second_candidate_index().is_some()
                || report.second_split().is_some()
                || report.hard_grandchild_lower_bounds().is_some()
            {
                return Err(MipError::Solver(
                    "AY adaptive three-leaf root report carried probe or second-level data"
                        .to_owned(),
                ));
            }
            replay_relaxation_root_entailment(
                problem,
                objective,
                &relaxed_model,
                &requested,
                row,
                "adaptive three-leaf root entailment",
            )
            .map(Some)
        }
        CertifiedAdaptiveThreeLeafHarvest::Tree(tree) => {
            let second_split = validate_adaptive_three_leaf_report_and_tree(
                &ay_candidates,
                root_candidate_index,
                hard_value,
                &report,
                &tree,
            )?;
            if mip_trace_armed() {
                eprintln!(
                    "NY_MIP_TRACE adaptive three-leaf target-FSB report: candidates={} \
                     probes={} root={} hard={} second={} hard_grandchildren={:?}",
                    report.candidate_count(),
                    report.probe_calls(),
                    root_split.index(),
                    hard_value,
                    second_split.index(),
                    report.hard_grandchild_lower_bounds(),
                );
            }
            compose_and_replay_adaptive_three_leaf_tree(
                problem,
                objective,
                requested_lower,
                root_split,
                hard_value,
                second_split,
                *tree,
                max_tree_leaves,
            )
            .map(Some)
        }
    }
}

fn validate_adaptive_four_leaf_comb_report_and_carrier(
    candidates: &[ay_milp::Col],
    root_candidate_index: usize,
    root_hard_value: bool,
    report: &AdaptiveFourLeafCombTargetFsbReport,
    comb: &CertifiedAdaptiveFourLeafComb,
) -> Result<(ay_milp::Col, bool, ay_milp::Col), MipError> {
    let root_split = *candidates.get(root_candidate_index).ok_or_else(|| {
        MipError::Solver(format!(
            "adaptive four-leaf comb root index {root_candidate_index} disappeared from {} \
             candidates",
            candidates.len()
        ))
    })?;
    let second_stage_probe_calls = candidates
        .len()
        .checked_sub(1)
        .and_then(|remaining| remaining.checked_mul(2))
        .ok_or_else(|| {
            MipError::Solver("adaptive four-leaf second-stage probe-count overflow".to_owned())
        })?;
    let third_stage_probe_calls = candidates
        .len()
        .checked_sub(2)
        .and_then(|remaining| remaining.checked_mul(2))
        .ok_or_else(|| {
            MipError::Solver("adaptive four-leaf third-stage probe-count overflow".to_owned())
        })?;
    let expected_probe_calls = second_stage_probe_calls
        .checked_add(third_stage_probe_calls)
        .ok_or_else(|| {
            MipError::Solver("adaptive four-leaf total probe-count overflow".to_owned())
        })?;
    let formula_probe_calls = candidates
        .len()
        .checked_mul(4)
        .and_then(|calls| calls.checked_sub(6))
        .ok_or_else(|| {
            MipError::Solver("adaptive four-leaf formula probe-count overflow".to_owned())
        })?;
    if expected_probe_calls != formula_probe_calls {
        return Err(MipError::Solver(
            "adaptive four-leaf probe-count formulas disagree".to_owned(),
        ));
    }

    let second_bounds = report.second_child_lower_bounds();
    let third_bounds = report.third_child_lower_bounds();
    let valid_probe_bound = |bound: f64| bound.is_finite() || bound == f64::NEG_INFINITY;
    let probe_bounds_valid = second_bounds
        .into_iter()
        .chain(third_bounds)
        .all(valid_probe_bound);
    let expected_second_hard_value = second_bounds[1] < second_bounds[0];
    if report.candidate_count() != candidates.len()
        || report.probe_calls() != expected_probe_calls
        || report.second_stage_probe_calls() != second_stage_probe_calls
        || report.third_stage_probe_calls() != third_stage_probe_calls
        || report.root_candidate_index() != root_candidate_index
        || report.root_split() != root_split
        || report.root_hard_value() != root_hard_value
        || report.second_hard_value() != expected_second_hard_value
        || !probe_bounds_valid
    {
        return Err(MipError::Solver(format!(
            "AY adaptive four-leaf comb report is inconsistent: candidates={}/{} \
             probes={}/{expected_probe_calls} stages={}/{},{} /{} root={}/{} \
             root_index={}/{} root_hard={}/{} second_hard={}/{} bounds_valid={probe_bounds_valid}",
            report.candidate_count(),
            candidates.len(),
            report.probe_calls(),
            report.second_stage_probe_calls(),
            second_stage_probe_calls,
            report.third_stage_probe_calls(),
            third_stage_probe_calls,
            report.root_split().index(),
            root_split.index(),
            report.root_candidate_index(),
            root_candidate_index,
            report.root_hard_value(),
            root_hard_value,
            report.second_hard_value(),
            expected_second_hard_value,
        )));
    }

    let second_candidate_index = report.second_candidate_index();
    let second_split = report.second_split();
    if second_candidate_index == root_candidate_index
        || candidates.get(second_candidate_index) != Some(&second_split)
        || second_split == root_split
    {
        return Err(MipError::Solver(format!(
            "AY adaptive four-leaf comb selected invalid second candidate \
             {second_candidate_index} at column {}",
            second_split.index(),
        )));
    }

    let third_candidate_index = report.third_candidate_index();
    let third_split = report.third_split();
    if third_candidate_index == root_candidate_index
        || third_candidate_index == second_candidate_index
        || candidates.get(third_candidate_index) != Some(&third_split)
        || third_split == root_split
        || third_split == second_split
    {
        return Err(MipError::Solver(format!(
            "AY adaptive four-leaf comb selected invalid third candidate \
             {third_candidate_index} at column {}",
            third_split.index(),
        )));
    }

    if comb.num_leaves() != 4
        || comb.root_split() != root_split
        || comb.root_hard_value() != root_hard_value
        || comb.second_split() != second_split
        || comb.second_hard_value() != expected_second_hard_value
        || comb.third_split() != third_split
    {
        return Err(MipError::Solver(format!(
            "AY adaptive four-leaf comb carrier is inconsistent: leaves={} root={}/{} \
             root_hard={}/{} second={}/{} second_hard={}/{} third={}/{}",
            comb.num_leaves(),
            comb.root_split().index(),
            root_split.index(),
            comb.root_hard_value(),
            root_hard_value,
            comb.second_split().index(),
            second_split.index(),
            comb.second_hard_value(),
            expected_second_hard_value,
            comb.third_split().index(),
            third_split.index(),
        )));
    }

    Ok((second_split, expected_second_hard_value, third_split))
}

fn validate_adaptive_four_leaf_comb_certificate_shape(
    root: &TreeNode,
    root_split: ay_milp::Col,
    root_hard_value: bool,
    second_split: ay_milp::Col,
    second_hard_value: bool,
    third_split: ay_milp::Col,
) -> Result<(), MipError> {
    let TreeNode::Split { col, cut, lo, hi } = root else {
        return Err(MipError::Solver(
            "AY adaptive four-leaf comb certificate has no root split".to_owned(),
        ));
    };
    if *col != root_split || !cut.is_zero() {
        return Err(MipError::Solver(format!(
            "AY adaptive four-leaf comb certificate changed root split {}/{} or its binary cut",
            col.index(),
            root_split.index()
        )));
    }

    let (root_easy, second_node) = if root_hard_value {
        (&**lo, &**hi)
    } else {
        (&**hi, &**lo)
    };
    if !matches!(root_easy, TreeNode::Leaf { .. }) {
        return Err(MipError::Solver(
            "AY adaptive four-leaf comb split the easy root sibling".to_owned(),
        ));
    }
    let TreeNode::Split { col, cut, lo, hi } = second_node else {
        return Err(MipError::Solver(
            "AY adaptive four-leaf comb did not split the hard root child".to_owned(),
        ));
    };
    if *col != second_split || !cut.is_zero() {
        return Err(MipError::Solver(format!(
            "AY adaptive four-leaf comb certificate changed second split {}/{} or its binary cut",
            col.index(),
            second_split.index()
        )));
    }

    let (second_easy, third_node) = if second_hard_value {
        (&**lo, &**hi)
    } else {
        (&**hi, &**lo)
    };
    if !matches!(second_easy, TreeNode::Leaf { .. }) {
        return Err(MipError::Solver(
            "AY adaptive four-leaf comb split the easy second sibling".to_owned(),
        ));
    }
    let TreeNode::Split { col, cut, lo, hi } = third_node else {
        return Err(MipError::Solver(
            "AY adaptive four-leaf comb did not split both hard assignments".to_owned(),
        ));
    };
    if *col != third_split
        || !cut.is_zero()
        || !matches!(&**lo, TreeNode::Leaf { .. })
        || !matches!(&**hi, TreeNode::Leaf { .. })
    {
        return Err(MipError::Solver(format!(
            "AY adaptive four-leaf comb certificate changed third split {}/{} or its two-leaf \
             terminal shape",
            col.index(),
            third_split.index()
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compose_and_replay_adaptive_four_leaf_comb(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    root_split: ay_milp::Col,
    root_hard_value: bool,
    second_split: ay_milp::Col,
    second_hard_value: bool,
    third_split: ay_milp::Col,
    comb: CertifiedAdaptiveFourLeafComb,
    max_tree_leaves: usize,
) -> Result<ReplayStats, MipError> {
    if comb.num_leaves() != 4 || max_tree_leaves < 4 {
        return Err(MipError::Solver(format!(
            "AY adaptive four-leaf comb returned {} leaves, admitted maximum is \
             {max_tree_leaves}",
            comb.num_leaves()
        )));
    }

    // The opaque carrier owns the proof-critical leaf-to-path association.
    // Once it closes every conditional row against the appended decision row,
    // NY admits only the explicit public comb topology and independently
    // replays all four resulting Farkas leaves.
    let decision = linear_lower_decision_problem(problem, objective, requested_lower);
    let decision_model = to_ay_model(&decision)?;
    let decision_row = decision_model.row_at(problem.num_rows()).ok_or_else(|| {
        MipError::Encoding(
            "appended adaptive four-leaf comb decision row disappeared during AY lowering"
                .to_owned(),
        )
    })?;
    let certificate = comb
        .into_farkas_against_row_upper(&decision_model, decision_row)
        .ok_or_else(|| {
            MipError::Solver(
                "AY adaptive four-leaf comb carrier failed exact decision-row composition"
                    .to_owned(),
            )
        })?;
    validate_adaptive_four_leaf_comb_certificate_shape(
        &certificate.root,
        root_split,
        root_hard_value,
        second_split,
        second_hard_value,
        third_split,
    )?;
    certificate.verify(&decision_model).map_err(|error| {
        MipError::Solver(format!(
            "AY adaptive four-leaf comb certificate failed independent whole-tree verification: \
             {error}"
        ))
    })?;
    let replay = replay_tree_farkas(&decision, &certificate.root, max_tree_leaves)?;
    if replay.proof_route != CertifiedLinearLowerProofRoute::TreeFarkas
        || replay.tree_leaves != 4
        || replay.linear_replays != 4
    {
        return Err(MipError::Solver(format!(
            "NY adaptive four-leaf comb replay used route {:?}, admitted {} leaves, and checked {} \
             linear obligations",
            replay.proof_route, replay.tree_leaves, replay.linear_replays
        )));
    }
    Ok(replay)
}

fn try_relaxed_linear_lower_adaptive_four_leaf_comb_target_fsb(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    candidates: &[Col],
    root_candidate_index: usize,
    root_hard_value: bool,
    max_tree_leaves: usize,
    opts: &SolveOpts,
) -> Result<Option<ReplayStats>, MipError> {
    if max_tree_leaves < 4 {
        return Ok(None);
    }

    let mut relaxed_problem = problem.clone();
    relaxed_problem.relax_integrality();
    let relaxed_model = to_ay_model(&relaxed_problem)?;
    let ay_objective = objective
        .iter()
        .map(|&(col, coefficient)| {
            relaxed_model
                .col_at(col.0)
                .map(|ay_col| (ay_col, coefficient))
                .ok_or_else(|| {
                    MipError::Encoding(format!(
                        "adaptive four-leaf comb objective column {} disappeared",
                        col.0
                    ))
                })
        })
        .collect::<Result<Vec<_>, MipError>>()?;
    let ay_candidates = candidates
        .iter()
        .map(|candidate| {
            relaxed_model.col_at(candidate.0).ok_or_else(|| {
                MipError::Encoding(format!(
                    "adaptive four-leaf comb candidate column {} disappeared",
                    candidate.0
                ))
            })
        })
        .collect::<Result<Vec<_>, MipError>>()?;
    let root_split = ay_candidates[root_candidate_index];
    let requested = exact_f64(
        f64::from(requested_lower),
        "requested certified lower threshold",
    )?;
    let mut session = LpSession::new(&relaxed_model, opts)
        .map_err(|error| MipError::Solver(error.to_string()))?;
    let fsb_opts = adaptive_target_fsb_opts();
    let Some((comb, report)) = session.harvest_adaptive_four_leaf_comb_target_fsb_stronger_than(
        &ay_objective,
        AySense::Minimize,
        &ay_candidates,
        root_candidate_index,
        root_hard_value,
        &requested,
        &fsb_opts,
    ) else {
        return Ok(None);
    };
    let (second_split, second_hard_value, third_split) =
        validate_adaptive_four_leaf_comb_report_and_carrier(
            &ay_candidates,
            root_candidate_index,
            root_hard_value,
            &report,
            &comb,
        )?;

    if mip_trace_armed() {
        eprintln!(
            "NY_MIP_TRACE adaptive four-leaf comb target-FSB report: candidates={} \
             probes={}/{}/{} root={} root_hard={} second={} second_hard={} \
             second_children={:?} third={} third_children={:?}",
            report.candidate_count(),
            report.probe_calls(),
            report.second_stage_probe_calls(),
            report.third_stage_probe_calls(),
            root_split.index(),
            u8::from(root_hard_value),
            second_split.index(),
            u8::from(second_hard_value),
            report.second_child_lower_bounds(),
            third_split.index(),
            report.third_child_lower_bounds(),
        );
    }
    compose_and_replay_adaptive_four_leaf_comb(
        problem,
        objective,
        requested_lower,
        root_split,
        root_hard_value,
        second_split,
        second_hard_value,
        third_split,
        comb,
        max_tree_leaves,
    )
    .map(Some)
}

fn validate_adaptive_five_leaf_comb_report_and_carrier(
    candidates: &[ay_milp::Col],
    root_candidate_index: usize,
    root_hard_value: bool,
    report: &AdaptiveFiveLeafCombTargetFsbReport,
    comb: &CertifiedAdaptiveFiveLeafComb,
) -> Result<(ay_milp::Col, bool, ay_milp::Col, bool, ay_milp::Col), MipError> {
    let root_split = *candidates.get(root_candidate_index).ok_or_else(|| {
        MipError::Solver(format!(
            "adaptive five-leaf comb root index {root_candidate_index} disappeared from {} \
             candidates",
            candidates.len()
        ))
    })?;
    let stage_calls = |removed: usize, label: &str| {
        candidates
            .len()
            .checked_sub(removed)
            .and_then(|remaining| remaining.checked_mul(2))
            .ok_or_else(|| {
                MipError::Solver(format!(
                    "adaptive five-leaf {label}-stage probe-count overflow"
                ))
            })
    };
    let second_stage_probe_calls = stage_calls(1, "second")?;
    let third_stage_probe_calls = stage_calls(2, "third")?;
    let fourth_stage_probe_calls = stage_calls(3, "fourth")?;
    let expected_probe_calls = second_stage_probe_calls
        .checked_add(third_stage_probe_calls)
        .and_then(|calls| calls.checked_add(fourth_stage_probe_calls))
        .ok_or_else(|| {
            MipError::Solver("adaptive five-leaf total probe-count overflow".to_owned())
        })?;
    let formula_probe_calls = candidates
        .len()
        .checked_mul(6)
        .and_then(|calls| calls.checked_sub(12))
        .ok_or_else(|| {
            MipError::Solver("adaptive five-leaf formula probe-count overflow".to_owned())
        })?;
    if expected_probe_calls != formula_probe_calls {
        return Err(MipError::Solver(
            "adaptive five-leaf probe-count formulas disagree".to_owned(),
        ));
    }

    let second_bounds = report.second_child_lower_bounds();
    let third_bounds = report.third_child_lower_bounds();
    let fourth_bounds = report.fourth_child_lower_bounds();
    let valid_probe_bound = |bound: f64| bound.is_finite() || bound == f64::NEG_INFINITY;
    let probe_bounds_valid = second_bounds
        .into_iter()
        .chain(third_bounds)
        .chain(fourth_bounds)
        .all(valid_probe_bound);
    let expected_second_hard_value = second_bounds[1] < second_bounds[0];
    let expected_third_hard_value = third_bounds[1] < third_bounds[0];
    if report.candidate_count() != candidates.len()
        || report.probe_calls() != expected_probe_calls
        || report.second_stage_probe_calls() != second_stage_probe_calls
        || report.third_stage_probe_calls() != third_stage_probe_calls
        || report.fourth_stage_probe_calls() != fourth_stage_probe_calls
        || report.root_candidate_index() != root_candidate_index
        || report.root_split() != root_split
        || report.root_hard_value() != root_hard_value
        || report.second_hard_value() != expected_second_hard_value
        || report.third_hard_value() != expected_third_hard_value
        || !probe_bounds_valid
    {
        return Err(MipError::Solver(format!(
            "AY adaptive five-leaf comb report is inconsistent: candidates={}/{} \
             probes={}/{expected_probe_calls} stages={}/{},{}/{},{}/{} root={}/{} \
             root_index={}/{} root_hard={}/{} second_hard={}/{} third_hard={}/{} \
             bounds_valid={probe_bounds_valid}",
            report.candidate_count(),
            candidates.len(),
            report.probe_calls(),
            report.second_stage_probe_calls(),
            second_stage_probe_calls,
            report.third_stage_probe_calls(),
            third_stage_probe_calls,
            report.fourth_stage_probe_calls(),
            fourth_stage_probe_calls,
            report.root_split().index(),
            root_split.index(),
            report.root_candidate_index(),
            root_candidate_index,
            report.root_hard_value(),
            root_hard_value,
            report.second_hard_value(),
            expected_second_hard_value,
            report.third_hard_value(),
            expected_third_hard_value,
        )));
    }

    let second_candidate_index = report.second_candidate_index();
    let second_split = report.second_split();
    if second_candidate_index == root_candidate_index
        || candidates.get(second_candidate_index) != Some(&second_split)
        || second_split == root_split
    {
        return Err(MipError::Solver(format!(
            "AY adaptive five-leaf comb selected invalid second candidate \
             {second_candidate_index} at column {}",
            second_split.index(),
        )));
    }

    let third_candidate_index = report.third_candidate_index();
    let third_split = report.third_split();
    if [root_candidate_index, second_candidate_index].contains(&third_candidate_index)
        || candidates.get(third_candidate_index) != Some(&third_split)
        || [root_split, second_split].contains(&third_split)
    {
        return Err(MipError::Solver(format!(
            "AY adaptive five-leaf comb selected invalid third candidate \
             {third_candidate_index} at column {}",
            third_split.index(),
        )));
    }

    let fourth_candidate_index = report.fourth_candidate_index();
    let fourth_split = report.fourth_split();
    if [
        root_candidate_index,
        second_candidate_index,
        third_candidate_index,
    ]
    .contains(&fourth_candidate_index)
        || candidates.get(fourth_candidate_index) != Some(&fourth_split)
        || [root_split, second_split, third_split].contains(&fourth_split)
    {
        return Err(MipError::Solver(format!(
            "AY adaptive five-leaf comb selected invalid fourth candidate \
             {fourth_candidate_index} at column {}",
            fourth_split.index(),
        )));
    }

    if comb.num_leaves() != 5
        || comb.root_split() != root_split
        || comb.root_hard_value() != root_hard_value
        || comb.second_split() != second_split
        || comb.second_hard_value() != expected_second_hard_value
        || comb.third_split() != third_split
        || comb.third_hard_value() != expected_third_hard_value
        || comb.fourth_split() != fourth_split
    {
        return Err(MipError::Solver(format!(
            "AY adaptive five-leaf comb carrier is inconsistent: leaves={} root={}/{} \
             root_hard={}/{} second={}/{} second_hard={}/{} third={}/{} \
             third_hard={}/{} fourth={}/{}",
            comb.num_leaves(),
            comb.root_split().index(),
            root_split.index(),
            comb.root_hard_value(),
            root_hard_value,
            comb.second_split().index(),
            second_split.index(),
            comb.second_hard_value(),
            expected_second_hard_value,
            comb.third_split().index(),
            third_split.index(),
            comb.third_hard_value(),
            expected_third_hard_value,
            comb.fourth_split().index(),
            fourth_split.index(),
        )));
    }

    Ok((
        second_split,
        expected_second_hard_value,
        third_split,
        expected_third_hard_value,
        fourth_split,
    ))
}

#[allow(clippy::too_many_arguments)]
fn validate_adaptive_five_leaf_comb_certificate_shape(
    root: &TreeNode,
    root_split: ay_milp::Col,
    root_hard_value: bool,
    second_split: ay_milp::Col,
    second_hard_value: bool,
    third_split: ay_milp::Col,
    third_hard_value: bool,
    fourth_split: ay_milp::Col,
) -> Result<(), MipError> {
    fn split<'a>(
        node: &'a TreeNode,
        expected: ay_milp::Col,
        label: &str,
    ) -> Result<(&'a TreeNode, &'a TreeNode), MipError> {
        let TreeNode::Split { col, cut, lo, hi } = node else {
            return Err(MipError::Solver(format!(
                "AY adaptive five-leaf comb certificate has no {label} split"
            )));
        };
        if *col != expected || !cut.is_zero() {
            return Err(MipError::Solver(format!(
                "AY adaptive five-leaf comb certificate changed {label} split {}/{} or its \
                 binary cut",
                col.index(),
                expected.index()
            )));
        }
        Ok((&**lo, &**hi))
    }
    fn easy_and_hard<'a>(
        lo: &'a TreeNode,
        hi: &'a TreeNode,
        hard_value: bool,
    ) -> (&'a TreeNode, &'a TreeNode) {
        if hard_value {
            (lo, hi)
        } else {
            (hi, lo)
        }
    }
    fn require_leaf(node: &TreeNode, label: &str) -> Result<(), MipError> {
        if !matches!(node, TreeNode::Leaf { .. }) {
            return Err(MipError::Solver(format!(
                "AY adaptive five-leaf comb split the easy {label} sibling"
            )));
        }
        Ok(())
    }

    let (root_lo, root_hi) = split(root, root_split, "root")?;
    let (root_easy, second_node) = easy_and_hard(root_lo, root_hi, root_hard_value);
    require_leaf(root_easy, "root")?;

    let (second_lo, second_hi) = split(second_node, second_split, "second")?;
    let (second_easy, third_node) = easy_and_hard(second_lo, second_hi, second_hard_value);
    require_leaf(second_easy, "second")?;

    let (third_lo, third_hi) = split(third_node, third_split, "third")?;
    let (third_easy, fourth_node) = easy_and_hard(third_lo, third_hi, third_hard_value);
    require_leaf(third_easy, "third")?;

    let (fourth_lo, fourth_hi) = split(fourth_node, fourth_split, "fourth")?;
    if !matches!(fourth_lo, TreeNode::Leaf { .. }) || !matches!(fourth_hi, TreeNode::Leaf { .. }) {
        return Err(MipError::Solver(
            "AY adaptive five-leaf comb terminal split does not have exactly two leaves".to_owned(),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compose_and_replay_adaptive_five_leaf_comb(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    root_split: ay_milp::Col,
    root_hard_value: bool,
    second_split: ay_milp::Col,
    second_hard_value: bool,
    third_split: ay_milp::Col,
    third_hard_value: bool,
    fourth_split: ay_milp::Col,
    comb: CertifiedAdaptiveFiveLeafComb,
    max_tree_leaves: usize,
) -> Result<ReplayStats, MipError> {
    if comb.num_leaves() != 5 || max_tree_leaves < 5 {
        return Err(MipError::Solver(format!(
            "AY adaptive five-leaf comb returned {} leaves, admitted maximum is \
             {max_tree_leaves}",
            comb.num_leaves()
        )));
    }

    let decision = linear_lower_decision_problem(problem, objective, requested_lower);
    let decision_model = to_ay_model(&decision)?;
    let decision_row = decision_model.row_at(problem.num_rows()).ok_or_else(|| {
        MipError::Encoding(
            "appended adaptive five-leaf comb decision row disappeared during AY lowering"
                .to_owned(),
        )
    })?;
    let certificate = comb
        .into_farkas_against_row_upper(&decision_model, decision_row)
        .ok_or_else(|| {
            MipError::Solver(
                "AY adaptive five-leaf comb carrier failed exact decision-row composition"
                    .to_owned(),
            )
        })?;
    validate_adaptive_five_leaf_comb_certificate_shape(
        &certificate.root,
        root_split,
        root_hard_value,
        second_split,
        second_hard_value,
        third_split,
        third_hard_value,
        fourth_split,
    )?;
    certificate.verify(&decision_model).map_err(|error| {
        MipError::Solver(format!(
            "AY adaptive five-leaf comb certificate failed independent whole-tree verification: \
             {error}"
        ))
    })?;
    let replay = replay_tree_farkas(&decision, &certificate.root, max_tree_leaves)?;
    if replay.proof_route != CertifiedLinearLowerProofRoute::TreeFarkas
        || replay.tree_leaves != 5
        || replay.linear_replays != 5
    {
        return Err(MipError::Solver(format!(
            "NY adaptive five-leaf comb replay used route {:?}, admitted {} leaves, and checked \
             {} linear obligations",
            replay.proof_route, replay.tree_leaves, replay.linear_replays
        )));
    }
    Ok(replay)
}

#[allow(clippy::too_many_arguments)]
fn try_relaxed_linear_lower_adaptive_five_leaf_comb_target_fsb(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    candidates: &[Col],
    root_candidate_index: usize,
    root_hard_value: bool,
    max_tree_leaves: usize,
    opts: &SolveOpts,
) -> Result<Option<ReplayStats>, MipError> {
    if max_tree_leaves < 5 {
        return Ok(None);
    }

    let mut relaxed_problem = problem.clone();
    relaxed_problem.relax_integrality();
    let relaxed_model = to_ay_model(&relaxed_problem)?;
    let ay_objective = objective
        .iter()
        .map(|&(col, coefficient)| {
            relaxed_model
                .col_at(col.0)
                .map(|ay_col| (ay_col, coefficient))
                .ok_or_else(|| {
                    MipError::Encoding(format!(
                        "adaptive five-leaf comb objective column {} disappeared",
                        col.0
                    ))
                })
        })
        .collect::<Result<Vec<_>, MipError>>()?;
    let ay_candidates = candidates
        .iter()
        .map(|candidate| {
            relaxed_model.col_at(candidate.0).ok_or_else(|| {
                MipError::Encoding(format!(
                    "adaptive five-leaf comb candidate column {} disappeared",
                    candidate.0
                ))
            })
        })
        .collect::<Result<Vec<_>, MipError>>()?;
    let root_split = ay_candidates[root_candidate_index];
    let requested = exact_f64(
        f64::from(requested_lower),
        "requested certified lower threshold",
    )?;
    let mut session = LpSession::new(&relaxed_model, opts)
        .map_err(|error| MipError::Solver(error.to_string()))?;
    let fsb_opts = adaptive_target_fsb_opts();
    let Some((comb, report)) = session.harvest_adaptive_five_leaf_comb_target_fsb_stronger_than(
        &ay_objective,
        AySense::Minimize,
        &ay_candidates,
        root_candidate_index,
        root_hard_value,
        &requested,
        &fsb_opts,
    ) else {
        return Ok(None);
    };
    let (second_split, second_hard_value, third_split, third_hard_value, fourth_split) =
        validate_adaptive_five_leaf_comb_report_and_carrier(
            &ay_candidates,
            root_candidate_index,
            root_hard_value,
            &report,
            &comb,
        )?;

    if mip_trace_armed() {
        eprintln!(
            "NY_MIP_TRACE adaptive five-leaf comb target-FSB report: candidates={} \
             probes={}/{}/{}/{} root={} root_hard={} second={} second_hard={} \
             second_children={:?} third={} third_hard={} third_children={:?} \
             fourth={} fourth_children={:?}",
            report.candidate_count(),
            report.probe_calls(),
            report.second_stage_probe_calls(),
            report.third_stage_probe_calls(),
            report.fourth_stage_probe_calls(),
            root_split.index(),
            u8::from(root_hard_value),
            second_split.index(),
            u8::from(second_hard_value),
            report.second_child_lower_bounds(),
            third_split.index(),
            u8::from(third_hard_value),
            report.third_child_lower_bounds(),
            fourth_split.index(),
            report.fourth_child_lower_bounds(),
        );
    }
    compose_and_replay_adaptive_five_leaf_comb(
        problem,
        objective,
        requested_lower,
        root_split,
        root_hard_value,
        second_split,
        second_hard_value,
        third_split,
        third_hard_value,
        fourth_split,
        comb,
        max_tree_leaves,
    )
    .map(Some)
}

fn solve_parallel_selector_leaf(
    base_model: &ay_milp::Model,
    ay_objective: &[(ay_milp::Col, f64)],
    ay_selectors: &[ay_milp::Col],
    assignment: usize,
    requested: &BigRational,
    deadline: Instant,
    opts: &SolveOpts,
) -> Result<Option<IndexedParallelSelectorLeaf>, MipError> {
    if assignment >= PARALLEL_SELECTOR_TREE_LEAVES || Instant::now() >= deadline {
        return Ok(None);
    }

    let mut leaf_model = base_model.clone();
    for (selector_index, &selector) in ay_selectors.iter().enumerate() {
        let value = f64::from(u8::from(selector_assignment_value(
            assignment,
            selector_index,
        )));
        leaf_model.fix_col(selector, value);
    }
    // Set before constructing the session so the fallback optimization shares
    // the same leaf box and absolute deadline as the threshold harvester.
    leaf_model.set_objective(ay_objective, AySense::Minimize);
    let mut session =
        LpSession::new(&leaf_model, opts).map_err(|error| MipError::Solver(error.to_string()))?;

    if let Some(row) = session.harvest_cut_stronger_than(ay_objective, AySense::Minimize, requested)
    {
        if Instant::now() >= deadline {
            return Ok(None);
        }
        row.verify(&leaf_model).map_err(|error| {
            MipError::Solver(format!(
                "AY parallel selector leaf {assignment:04b} row failed verification: {error}"
            ))
        })?;
        if row.lb <= *requested {
            return Ok(None);
        }
        return Ok(Some(IndexedParallelSelectorLeaf {
            assignment,
            evidence: ParallelSelectorLeafEvidence::ConditionalRow(Box::new(row)),
        }));
    }

    // The harvester intentionally returns no row for an infeasible fixed box.
    // Reusing the same session for its model objective recovers either the
    // exact optimality row or the direct Farkas witness without granting a new
    // clock.
    if Instant::now() >= deadline {
        return Ok(None);
    }
    let outcome = session
        .optimize_model_objective()
        .map_err(|error| MipError::Solver(error.to_string()))?;
    if Instant::now() >= deadline {
        return Ok(None);
    }
    let evidence = match outcome {
        Outcome::Optimal {
            cert: Some(cert), ..
        } => {
            cert.verify(&leaf_model).map_err(|error| {
                MipError::Solver(format!(
                    "AY parallel selector leaf {assignment:04b} optimality certificate failed \
                     verification: {error}"
                ))
            })?;
            let row = cert.into_certified_row();
            if row.lb <= *requested {
                return Ok(None);
            }
            row.verify(&leaf_model).map_err(|error| {
                MipError::Solver(format!(
                    "AY parallel selector leaf {assignment:04b} fallback row failed verification: \
                     {error}"
                ))
            })?;
            ParallelSelectorLeafEvidence::ConditionalRow(Box::new(row))
        }
        Outcome::Infeasible {
            cert: Some(farkas), ..
        } => {
            farkas.verify(&leaf_model).map_err(|error| {
                MipError::Solver(format!(
                    "AY parallel selector leaf {assignment:04b} Farkas certificate failed \
                     verification: {error}"
                ))
            })?;
            ParallelSelectorLeafEvidence::Infeasible(farkas)
        }
        _ => return Ok(None),
    };
    Ok(Some(IndexedParallelSelectorLeaf {
        assignment,
        evidence,
    }))
}

#[allow(clippy::too_many_arguments)]
fn try_relaxed_linear_lower_parallel_selector_tree(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    selectors: &[Col],
    max_workers: usize,
    max_tree_leaves: usize,
    deadline: Instant,
    opts: &SolveOpts,
) -> Result<Option<ReplayStats>, MipError> {
    if max_tree_leaves < PARALLEL_SELECTOR_TREE_LEAVES || Instant::now() >= deadline {
        return Ok(None);
    }

    // This is the only integrality relaxation and AY lowering in the lane.
    // Every assignment worker clones this immutable base model, then changes
    // only its own four selector bounds.
    let mut relaxed_problem = problem.clone();
    relaxed_problem.relax_integrality();
    let relaxed_model = to_ay_model(&relaxed_problem)?;
    let ay_objective = objective
        .iter()
        .map(|&(col, coefficient)| {
            relaxed_model
                .col_at(col.0)
                .map(|ay_col| (ay_col, coefficient))
                .ok_or_else(|| {
                    MipError::Encoding(format!(
                        "parallel selector-tree objective column {} disappeared",
                        col.0
                    ))
                })
        })
        .collect::<Result<Vec<_>, MipError>>()?;
    let ay_selectors = selectors
        .iter()
        .map(|selector| {
            relaxed_model.col_at(selector.0).ok_or_else(|| {
                MipError::Encoding(format!(
                    "parallel selector-tree column {} disappeared",
                    selector.0
                ))
            })
        })
        .collect::<Result<Vec<_>, MipError>>()?;
    let requested = exact_f64(
        f64::from(requested_lower),
        "requested certified lower threshold",
    )?;

    let Some(leaves) = run_bounded_canonical_assignment_workers(
        PARALLEL_SELECTOR_TREE_LEAVES,
        max_workers,
        deadline,
        |assignment| {
            solve_parallel_selector_leaf(
                &relaxed_model,
                &ay_objective,
                &ay_selectors,
                assignment,
                &requested,
                deadline,
                opts,
            )
        },
    )?
    else {
        return Ok(None);
    };
    if Instant::now() >= deadline {
        return Ok(None);
    }
    let replay = compose_and_replay_parallel_selector_tree(
        problem,
        objective,
        requested_lower,
        selectors,
        &ay_selectors,
        leaves,
        max_tree_leaves,
    )?;
    if Instant::now() >= deadline {
        return Ok(None);
    }
    Ok(Some(replay))
}

fn try_relaxed_linear_lower_fixed_assignment_tree(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    splits: &[Col],
    max_tree_leaves: usize,
    opts: &SolveOpts,
) -> Result<Option<ReplayStats>, MipError> {
    let mut relaxed_problem = problem.clone();
    relaxed_problem.relax_integrality();
    let relaxed_model = to_ay_model(&relaxed_problem)?;
    let ay_objective = objective
        .iter()
        .map(|&(col, coefficient)| {
            relaxed_model
                .col_at(col.0)
                .map(|ay_col| (ay_col, coefficient))
                .ok_or_else(|| {
                    MipError::Encoding(format!(
                        "fixed assignment-tree objective column {} disappeared",
                        col.0
                    ))
                })
        })
        .collect::<Result<Vec<_>, MipError>>()?;
    let ay_splits = splits
        .iter()
        .map(|split| {
            relaxed_model.col_at(split.0).ok_or_else(|| {
                MipError::Encoding(format!(
                    "fixed assignment-tree split column {} disappeared",
                    split.0
                ))
            })
        })
        .collect::<Result<Vec<_>, MipError>>()?;
    let requested = exact_f64(
        f64::from(requested_lower),
        "requested certified lower threshold",
    )?;
    let mut session = LpSession::new(&relaxed_model, opts)
        .map_err(|error| MipError::Solver(error.to_string()))?;
    let Some(harvest) = session.harvest_cut_or_binary_assignment_tree_stronger_than(
        &ay_objective,
        AySense::Minimize,
        &ay_splits,
        &requested,
    ) else {
        return Ok(None);
    };

    match harvest {
        CertifiedBinaryTreeHarvest::Root(row) => replay_relaxation_root_entailment(
            problem,
            objective,
            &relaxed_model,
            &requested,
            row,
            "fixed assignment-tree root entailment",
        )
        .map(Some),
        CertifiedBinaryTreeHarvest::Tree(tree) => {
            let expected_leaves = 1usize << splits.len();
            if expected_leaves > max_tree_leaves {
                return Ok(None);
            }
            compose_and_replay_assignment_tree(
                problem,
                objective,
                requested_lower,
                splits,
                &ay_splits,
                tree,
                max_tree_leaves,
                "fixed assignment-tree harvest",
            )
            .map(Some)
        }
    }
}

fn try_relaxed_linear_lower_fixed_assignment_tree_until(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    splits: &[Col],
    max_tree_leaves: usize,
    deadline: Instant,
    opts: &SolveOpts,
) -> Result<Option<ReplayStats>, MipError> {
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }
    let mut relaxed_problem = problem.clone();
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }
    relaxed_problem.relax_integrality();
    let relaxed_model = to_ay_model(&relaxed_problem)?;
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }
    let ay_objective = objective
        .iter()
        .map(|&(col, coefficient)| {
            relaxed_model
                .col_at(col.0)
                .map(|ay_col| (ay_col, coefficient))
                .ok_or_else(|| {
                    MipError::Encoding(format!(
                        "fixed assignment-tree objective column {} disappeared",
                        col.0
                    ))
                })
        })
        .collect::<Result<Vec<_>, MipError>>()?;
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }
    let ay_splits = splits
        .iter()
        .map(|split| {
            relaxed_model.col_at(split.0).ok_or_else(|| {
                MipError::Encoding(format!(
                    "fixed assignment-tree split column {} disappeared",
                    split.0
                ))
            })
        })
        .collect::<Result<Vec<_>, MipError>>()?;
    let requested = exact_f64(
        f64::from(requested_lower),
        "requested certified lower threshold",
    )?;
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }
    let mut session = LpSession::new(&relaxed_model, opts)
        .map_err(|error| MipError::Solver(error.to_string()))?;
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }
    let Some(harvest) = session.harvest_cut_or_binary_assignment_tree_stronger_than(
        &ay_objective,
        AySense::Minimize,
        &ay_splits,
        &requested,
    ) else {
        return Ok(None);
    };
    if !fixed_assignment_tree_deadline_open(deadline) {
        return Ok(None);
    }

    match harvest {
        CertifiedBinaryTreeHarvest::Root(row) => replay_relaxation_root_entailment_until(
            problem,
            objective,
            &relaxed_model,
            &requested,
            row,
            "fixed assignment-tree root entailment",
            deadline,
        ),
        CertifiedBinaryTreeHarvest::Tree(tree) => {
            let expected_leaves = 1usize << splits.len();
            if expected_leaves > max_tree_leaves {
                return Ok(None);
            }
            compose_and_replay_assignment_tree_until(
                problem,
                objective,
                requested_lower,
                splits,
                &ay_splits,
                tree,
                max_tree_leaves,
                "fixed assignment-tree harvest",
                deadline,
            )
        }
    }
}

#[cfg(test)]
fn try_relaxed_linear_lower_proof(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    branch_advice: &[Col],
    max_tree_leaves: usize,
    opts: &SolveOpts,
) -> Result<Option<ReplayStats>, MipError> {
    try_relaxed_linear_lower_proof_with_target_fsb_probe_limits(
        problem,
        objective,
        requested_lower,
        branch_advice,
        max_tree_leaves,
        opts,
        CertifiedLinearLowerTargetFsbProbeLimits::production(),
    )
}

fn try_relaxed_linear_lower_proof_with_target_fsb_probe_limits(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    branch_advice: &[Col],
    max_tree_leaves: usize,
    opts: &SolveOpts,
    target_fsb_probe_limits: CertifiedLinearLowerTargetFsbProbeLimits,
) -> Result<Option<ReplayStats>, MipError> {
    let mut relaxed_problem = problem.clone();
    relaxed_problem.relax_integrality();
    let relaxed_model = to_ay_model(&relaxed_problem)?;
    let mut ay_objective = Vec::with_capacity(objective.len());
    for &(col, coefficient) in objective {
        let ay_col = relaxed_model.col_at(col.0).ok_or_else(|| {
            MipError::Encoding(format!("relaxation objective column {} disappeared", col.0))
        })?;
        ay_objective.push((ay_col, coefficient));
    }
    let requested = exact_f64(
        f64::from(requested_lower),
        "requested certified lower threshold",
    )?;
    let mut session = LpSession::new(&relaxed_model, opts)
        .map_err(|error| MipError::Solver(error.to_string()))?;

    // Preserve the ordinary route exactly: the combined warm-split API is a
    // weak-dual-only probe and deliberately has different fallback behavior.
    // It is entered only with explicit, live advice and room to admit both
    // leaves.
    if branch_advice.is_empty() || max_tree_leaves < 2 {
        let Some(row) =
            session.harvest_cut_stronger_than(&ay_objective, AySense::Minimize, &requested)
        else {
            return Ok(None);
        };
        if mip_trace_armed() {
            eprintln!(
                "NY_MIP_TRACE relaxation harvest: lb={}/{} requested={}/{} multipliers={} sufficient={}",
                row.lb.numer(),
                row.lb.denom(),
                requested.numer(),
                requested.denom(),
                row.multipliers.len(),
                row.lb > requested,
            );
        }
        // Bound strength is a cheap fail-closed gate. Check it before repeating
        // AY's exact matrix verification so an insufficient weak row leaves the
        // remaining deadline to the decision-MILP fallback.
        if row.lb <= requested {
            return Ok(None);
        }
        row.verify(&relaxed_model).map_err(|error| {
            MipError::Solver(format!(
                "AY relaxation entailment failed independent verification: {error}"
            ))
        })?;
        // The relaxed copy differs only in integrality metadata. Match and replay
        // the certified row against the caller's original rows and bounds so that
        // integrality never becomes an implicit linear premise.
        let expected = canonical_exact_objective(problem, objective)?;
        let actual = canonical_certified_row(problem, &row)?;
        if actual != expected {
            return Err(MipError::Solver(
                "AY relaxation entailment does not match the requested objective".to_owned(),
            ));
        }
        replay_entailment_with_ny_cert(problem, &row)?;
        return Ok(Some(ReplayStats {
            proof_route: CertifiedLinearLowerProofRoute::RelaxationEntailment,
            tree_leaves: 0,
            linear_replays: 1,
        }));
    }

    if branch_advice.len() >= 2 && max_tree_leaves >= 4 {
        if (3..=8).contains(&branch_advice.len()) {
            let ay_candidates = branch_advice
                .iter()
                .map(|candidate| {
                    relaxed_model.col_at(candidate.0).ok_or_else(|| {
                        MipError::Encoding(format!(
                            "target-FSB relaxation candidate column {} disappeared",
                            candidate.0
                        ))
                    })
                })
                .collect::<Result<Vec<_>, MipError>>()?;
            let target_fsb_opts = TargetFsbOpts::new()
                .with_max_probe_pivots_per_call(target_fsb_probe_limits.max_probe_pivots_per_call())
                .with_max_probe_calls(TARGET_FSB_MAX_PROBE_CALLS)
                .with_probe_time_limit(target_fsb_probe_limits.probe_time_limit())
                .with_max_probe_scratch_bytes(TARGET_FSB_MAX_PROBE_SCRATCH_BYTES);
            let Some((harvest, report)) = session
                .harvest_cut_or_target_fsb_assignment_tree_stronger_than(
                    &ay_objective,
                    AySense::Minimize,
                    &ay_candidates,
                    &requested,
                    &target_fsb_opts,
                )
            else {
                return Ok(None);
            };
            if report.candidate_count() != branch_advice.len() {
                return Err(MipError::Solver(format!(
                    "AY target-FSB report changed the candidate count from {} to {}",
                    branch_advice.len(),
                    report.candidate_count()
                )));
            }

            return match harvest {
                CertifiedBinaryTreeHarvest::Root(row) => {
                    if !report.selected_splits().is_empty() || report.probe_calls() != 0 {
                        return Err(MipError::Solver(
                            "AY target-FSB root report carried split selection or probe calls"
                                .to_owned(),
                        ));
                    }
                    replay_relaxation_root_entailment(
                        problem,
                        objective,
                        &relaxed_model,
                        &requested,
                        row,
                        "target-FSB root entailment",
                    )
                    .map(Some)
                }
                CertifiedBinaryTreeHarvest::Tree(tree) => {
                    let report_splits = report.selected_splits();
                    let expected_probe_calls = branch_advice.len() * 6 - 4;
                    if report_splits.len() != 2
                        || report_splits[0] == report_splits[1]
                        || report.probe_calls() != expected_probe_calls
                        || tree.num_leaves() != 4
                        || tree.split_cols() != report_splits
                    {
                        return Err(MipError::Solver(format!(
                            "AY target-FSB tree report is inconsistent: selected={} probes={}/{} leaves={}",
                            report_splits.len(),
                            report.probe_calls(),
                            expected_probe_calls,
                            tree.num_leaves()
                        )));
                    }
                    let selected = report_splits
                        .iter()
                        .map(|split| {
                            ay_candidates
                                .iter()
                                .position(|candidate| candidate == split)
                                .map(|index| branch_advice[index])
                                .ok_or_else(|| {
                                    MipError::Solver(format!(
                                        "AY target-FSB selected non-candidate column {}",
                                        split.index()
                                    ))
                                })
                        })
                        .collect::<Result<Vec<_>, MipError>>()?;
                    if mip_trace_armed() {
                        eprintln!(
                            "NY_MIP_TRACE target-FSB report: candidates={} probes={} first_worst={:?} joint_worst={:?}",
                            report.candidate_count(),
                            report.probe_calls(),
                            report.first_worst_lower_bound(),
                            report.joint_worst_lower_bound(),
                        );
                    }
                    compose_and_replay_assignment_tree(
                        problem,
                        objective,
                        requested_lower,
                        &selected,
                        report_splits,
                        tree,
                        max_tree_leaves,
                        "target-FSB assignment-tree harvest",
                    )
                    .map(Some)
                }
            };
        }

        // Exactly two candidates preserve the caller-ranked control. Oversized
        // advice is deliberately not scanned: retain the historical first-two
        // bounded behavior rather than silently repricing the request.
        let selected = &branch_advice[..2];
        let ay_splits = selected
            .iter()
            .map(|split| {
                relaxed_model.col_at(split.0).ok_or_else(|| {
                    MipError::Encoding(format!(
                        "advised relaxation assignment-tree column {} disappeared",
                        split.0
                    ))
                })
            })
            .collect::<Result<Vec<_>, MipError>>()?;
        let Some(harvest) = session.harvest_cut_or_binary_assignment_tree_stronger_than(
            &ay_objective,
            AySense::Minimize,
            &ay_splits,
            &requested,
        ) else {
            return Ok(None);
        };
        return match harvest {
            CertifiedBinaryTreeHarvest::Root(row) => replay_relaxation_root_entailment(
                problem,
                objective,
                &relaxed_model,
                &requested,
                row,
                "advised assignment-tree root entailment",
            )
            .map(Some),
            CertifiedBinaryTreeHarvest::Tree(tree) => compose_and_replay_assignment_tree(
                problem,
                objective,
                requested_lower,
                selected,
                &ay_splits,
                tree,
                max_tree_leaves,
                "advised assignment-tree harvest",
            )
            .map(Some),
        };
    }

    let split = branch_advice[0];
    let ay_split = relaxed_model.col_at(split.0).ok_or_else(|| {
        MipError::Encoding(format!(
            "advised relaxation split column {} disappeared",
            split.0
        ))
    })?;
    let Some(harvest) = session.harvest_cut_or_binary_split_stronger_than(
        &ay_objective,
        AySense::Minimize,
        ay_split,
        &requested,
    ) else {
        return Ok(None);
    };
    match harvest {
        CertifiedSplitHarvest::Root(row) => {
            if mip_trace_armed() {
                eprintln!(
                    "NY_MIP_TRACE advised root harvest: lb={}/{} requested={}/{} multipliers={} sufficient={}",
                    row.lb.numer(),
                    row.lb.denom(),
                    requested.numer(),
                    requested.denom(),
                    row.multipliers.len(),
                    row.lb > requested,
                );
            }
            if row.lb <= requested {
                return Err(MipError::Solver(
                    "AY advised root harvest did not strictly clear the requested threshold"
                        .to_owned(),
                ));
            }
            row.verify(&relaxed_model).map_err(|error| {
                MipError::Solver(format!(
                    "AY advised root entailment failed independent verification: {error}"
                ))
            })?;
            let expected = canonical_exact_objective(problem, objective)?;
            let actual = canonical_certified_row(problem, &row)?;
            if actual != expected {
                return Err(MipError::Solver(
                    "AY advised root entailment does not match the requested objective".to_owned(),
                ));
            }
            replay_entailment_with_ny_cert(problem, &row)?;
            Ok(Some(ReplayStats {
                proof_route: CertifiedLinearLowerProofRoute::RelaxationEntailment,
                tree_leaves: 0,
                linear_replays: 1,
            }))
        }
        CertifiedSplitHarvest::Split { zero, one } => {
            if zero.lb <= requested || one.lb <= requested {
                return Err(MipError::Solver(
                    "AY advised child harvest did not strictly clear the requested threshold"
                        .to_owned(),
                ));
            }
            let expected = canonical_exact_objective(problem, objective)?;
            for (child, row) in [("zero", &zero), ("one", &one)] {
                let actual = canonical_certified_row(problem, row)?;
                if actual != expected {
                    return Err(MipError::Solver(format!(
                        "AY advised {child}-child row does not match the requested objective"
                    )));
                }
            }
            if mip_trace_armed() {
                eprintln!(
                    "NY_MIP_TRACE advised split harvest: col={} zero_lb={}/{} one_lb={}/{} requested={}/{}",
                    split.0,
                    zero.lb.numer(),
                    zero.lb.denom(),
                    one.lb.numer(),
                    one.lb.denom(),
                    requested.numer(),
                    requested.denom(),
                );
            }

            // Child rows name the base relaxation's rows and columns. Appending
            // the decision row preserves those identities, while restoring
            // integrality makes x<=0 / x>=1 a complete checked disjunction.
            let decision = linear_lower_decision_problem(problem, objective, requested_lower);
            let decision_model = to_ay_model(&decision)?;
            let decision_row = decision_model.row_at(problem.num_rows()).ok_or_else(|| {
                MipError::Encoding(
                    "appended linear-lower decision row disappeared during AY lowering".to_owned(),
                )
            })?;
            let decision_split = decision_model.col_at(split.0).ok_or_else(|| {
                MipError::Encoding(format!(
                    "advised decision split column {} disappeared",
                    split.0
                ))
            })?;
            let zero_cut = BigRational::zero();
            let one_cut = BigRational::from_integer(1.into());
            let lo = zero
                .into_farkas_against_row_upper(
                    &decision_model,
                    decision_row,
                    &[(decision_split, BoundSide::Upper, zero_cut.clone())],
                )
                .ok_or_else(|| {
                    MipError::Solver(
                        "AY zero-child row failed exact decision-row composition".to_owned(),
                    )
                })?;
            let hi = one
                .into_farkas_against_row_upper(
                    &decision_model,
                    decision_row,
                    &[(decision_split, BoundSide::Lower, one_cut)],
                )
                .ok_or_else(|| {
                    MipError::Solver(
                        "AY one-child row failed exact decision-row composition".to_owned(),
                    )
                })?;
            let certificate = MilpInfeasibilityCertificate {
                root: TreeNode::Split {
                    col: decision_split,
                    cut: zero_cut,
                    lo: Box::new(TreeNode::Leaf { farkas: lo }),
                    hi: Box::new(TreeNode::Leaf { farkas: hi }),
                },
            };
            certificate.verify(&decision_model).map_err(|error| {
                MipError::Solver(format!(
                    "AY advised two-leaf certificate failed independent verification: {error}"
                ))
            })?;
            replay_tree_farkas(&decision, &certificate.root, max_tree_leaves).map(Some)
        }
    }
}

fn solve_and_replay_milp_proof(
    problem: &MilpProblem,
    opts: &SolveOpts,
    max_tree_leaves: usize,
    branch_advice: &[Col],
) -> Result<Option<ReplayStats>, MipError> {
    let model = to_ay_model(problem)?;
    let mut session =
        BabSession::new(model, opts).map_err(|error| MipError::Solver(error.to_string()))?;
    if !branch_advice.is_empty() {
        let ay_advice = branch_advice
            .iter()
            .map(|col| {
                session.model().col_at(col.0).ok_or_else(|| {
                    MipError::Encoding(format!(
                        "advised fallback branch column {} disappeared",
                        col.0
                    ))
                })
            })
            .collect::<Result<Vec<_>, MipError>>()?;
        session.hint_branch_order(&ay_advice);
        session.shortlist_root_strong_branch_candidates(&ay_advice);
    }
    let outcome = session
        .check()
        .map_err(|error| MipError::Solver(error.to_string()))?;
    match outcome {
        Outcome::Infeasible {
            cert: Some(cert), ..
        } => {
            cert.verify(session.model()).map_err(|error| {
                MipError::Solver(format!(
                    "AY root Farkas certificate failed independent verification: {error}"
                ))
            })?;
            replay_root_farkas(problem, &cert).map(Some)
        }
        Outcome::Infeasible {
            cert: None,
            tree_cert: Some(tree),
        } => {
            if tree.num_leaves() > max_tree_leaves {
                return Ok(None);
            }
            tree.verify(session.model()).map_err(|error| {
                MipError::Solver(format!(
                    "AY tree certificate failed independent verification: {error}"
                ))
            })?;
            replay_tree_farkas(problem, &tree.root, max_tree_leaves).map(Some)
        }
        _ => Ok(None),
    }
}

fn solve_and_replay_continuous_root_infeasibility(
    problem: &MilpProblem,
    opts: &SolveOpts,
) -> Result<Option<CertifiedContinuousRootInfeasibility>, MipError> {
    let model = to_ay_model(problem)?;
    // This authority lane has already rejected every integer column. Go
    // directly through AY's warm LP session instead of paying the generic
    // branch-and-bound ingress merely to rediscover a continuous root. The LP
    // outcome carries the same exact Farkas type and is explicitly verified
    // below before ny-cert reconstructs it from the original IR.
    let mut session =
        LpSession::new(&model, opts).map_err(|error| MipError::Solver(error.to_string()))?;
    let outcome = session
        .optimize_model_objective()
        .map_err(|error| MipError::Solver(error.to_string()))?;
    let Outcome::Infeasible {
        cert: Some(cert), ..
    } = outcome
    else {
        // Feasible, Unknown, or tree-only evidence has no authority in this
        // deliberately root-LP-only lane.
        return Ok(None);
    };
    let resources = match continuous_root_farkas_resource_usage(problem, &cert) {
        Ok(resources) => resources,
        Err(reason) => {
            tracing::warn!(
                multipliers = cert.multipliers.len(),
                ?reason,
                "continuous root Farkas certificate exceeded exact replay limits; declining"
            );
            return Ok(None);
        }
    };
    tracing::debug!(
        multipliers = resources.multipliers,
        rational_bits = resources.rational_bits,
        referenced_row_entries = resources.referenced_row_entries,
        referenced_row_nonzeros = resources.referenced_row_nonzeros,
        column_bound_facts = resources.column_bound_facts,
        expanded_fact_terms = resources.expanded_fact_terms,
        model_rational_bits = resources.model_rational_bits,
        weighted_replay_work = resources.weighted_replay_work,
        "continuous root Farkas certificate admitted for explicit exact replay"
    );
    cert.verify(&model).map_err(|error| {
        MipError::Solver(format!(
            "AY continuous root Farkas certificate failed exact verification: {error}"
        ))
    })?;
    let replay = replay_root_farkas(problem, &cert)?;
    if replay.proof_route != CertifiedLinearLowerProofRoute::RootFarkas
        || replay.tree_leaves != 0
        || replay.linear_replays != 1
    {
        return Ok(None);
    }
    Ok(Some(CertifiedContinuousRootInfeasibility {
        ay_farkas_multipliers: resources.multipliers,
        ny_cert_farkas_replays: replay.linear_replays,
    }))
}

/// Prove that a continuous [`MilpProblem`] is infeasible at its root LP before
/// one caller-owned absolute deadline.
///
/// `Some` has verdict authority only after the linked AY engine has exported
/// and exactly verified a root Farkas certificate and ny-cert has independently
/// reconstructed and replayed every referenced row or column-bound fact from
/// the original IR. Integer columns, a marked decision row, an expired
/// deadline, feasible/unknown outcomes, missing or tree-only evidence, and
/// resource-limit refusals return `Ok(None)`. Before AY is launched, the
/// complete IR also crosses overflow-checked row-entry and exact-rational-bit
/// ceilings; certificate-dependent replay accounting alone would be too late
/// to bound exact model construction.
///
/// The worker admission is retained inside a detached deadline worker until
/// that worker really exits. Thus a timed-out exact operation cannot accumulate
/// additional AY workers through repeated optional calls.
pub fn certify_continuous_root_infeasibility_with_ay_until(
    problem: &MilpProblem,
    deadline: Instant,
) -> Result<Option<CertifiedContinuousRootInfeasibility>, MipError> {
    if Instant::now() >= deadline
        || problem.margin_row().is_some()
        || continuous_root_problem_resource_usage(problem).is_err()
        || problem.cols().iter().any(|column| column.integer)
    {
        return Ok(None);
    }
    let Some(admission) = CertifiedLinearLowerWorkerAdmission::try_acquire() else {
        return Ok(None);
    };
    certify_continuous_root_infeasibility_with_ay_until_admission(problem, deadline, admission)
}

/// Admission-owning counterpart to
/// [`certify_continuous_root_infeasibility_with_ay_until`].
///
/// Callers may acquire the opaque worker slot before bounded model construction
/// and pass it here, shedding that work when a previous detached AY worker is
/// still alive. Model cloning and worker setup consume the same absolute
/// `deadline`; neither operation creates a fresh relative slice.
pub fn certify_continuous_root_infeasibility_with_ay_until_admission(
    problem: &MilpProblem,
    deadline: Instant,
    admission: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<CertifiedContinuousRootInfeasibility>, MipError> {
    if Instant::now() >= deadline || problem.margin_row().is_some() {
        return Ok(None);
    }

    let resources = match continuous_root_problem_resource_usage(problem) {
        Ok(resources) => resources,
        Err(reason) => {
            tracing::warn!(
                ?reason,
                "continuous root model exceeded exact pre-solve limits; declining"
            );
            return Ok(None);
        }
    };
    if problem.cols().iter().any(|column| column.integer) {
        return Ok(None);
    }
    tracing::debug!(
        columns = resources.columns,
        rows = resources.rows,
        row_entries = resources.row_entries,
        rational_bits = resources.rational_bits,
        "continuous root model admitted for exact lowering"
    );

    let problem = problem.clone();
    if Instant::now() >= deadline {
        return Ok(None);
    }
    run_with_hard_deadline_at(deadline, "continuous-root-farkas", move || {
        // If the outer wait expires, the detached worker retains admission
        // through AY verification and independent ny-cert replay.
        let _admission = admission;
        let Some(remaining) = deadline
            .checked_duration_since(Instant::now())
            .filter(|remaining| !remaining.is_zero())
        else {
            return Ok(None);
        };
        let opts = solve_opts(remaining.as_secs_f64())
            .with_deadline(deadline)
            .with_tree_cert_leaves(0)
            .with_require_certificates(true)
            .with_structure_routing(false);
        solve_and_replay_continuous_root_infeasibility(&problem, &opts)
    })
    .map(Option::flatten)
}

fn linear_lower_decision_problem(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
) -> MilpProblem {
    let mut decision = problem.clone();
    decision.add_row(
        f64::NEG_INFINITY,
        f64::from(requested_lower),
        objective.iter().copied(),
    );
    decision
}

/// Capture the exact fallback decision model before the relaxation fast path.
///
/// Even though the fast path has a bounded share of the proof slice, a hard
/// deadline can detach its worker. Capturing first preserves the model needed
/// for an offline AY comparison.
/// Keep the ordinary path allocation-free: cloning occurs only when the
/// existing `NY_MIP_DUMP` diagnostic is explicitly armed.
fn maybe_dump_linear_lower_decision(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
) -> Option<std::path::PathBuf> {
    if !crate::dump::enabled() {
        return None;
    }
    let decision = linear_lower_decision_problem(problem, objective, requested_lower);
    crate::dump::maybe_dump(&decision)
}

fn solve_and_replay_fixed_assignment_tree_proof(
    problem: MilpProblem,
    objective: Vec<(Col, f64)>,
    requested_lower: f32,
    splits: Vec<Col>,
    timeout_secs: f64,
    max_tree_leaves: usize,
    lp_solve_policy: LpSolvePolicy,
    worker_lease: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<ReplayStats>, MipError> {
    let _ = maybe_dump_linear_lower_decision(&problem, &objective, requested_lower);
    let proof_deadline = Instant::now()
        .checked_add(Duration::from_secs_f64(timeout_secs))
        .ok_or_else(|| MipError::Encoding("linear-lower proof deadline overflow".to_owned()))?;
    run_with_hard_deadline(
        timeout_secs,
        "linear-lower-fixed-assignment-tree-proof",
        move || {
            // Retain process-wide admission if the hard deadline detaches this
            // exact AY/replay worker.
            let _worker_lease = worker_lease;
            let opts = lp_solve_policy.apply(
                solve_opts(timeout_secs)
                    .with_deadline(proof_deadline)
                    .with_tree_cert_leaves(0)
                    .with_require_certificates(true),
            );
            try_relaxed_linear_lower_fixed_assignment_tree(
                &problem,
                &objective,
                requested_lower,
                &splits,
                max_tree_leaves,
                &opts,
            )
        },
    )
    .map(Option::flatten)
}

#[allow(clippy::too_many_arguments)]
fn solve_and_replay_fixed_assignment_tree_proof_until_with_setup_control<N, H, S>(
    problem: &MilpProblem,
    objective: Vec<(Col, f64)>,
    requested_lower: f32,
    splits: Vec<Col>,
    proof_deadline: Instant,
    max_tree_leaves: usize,
    lp_solve_policy: LpSolvePolicy,
    worker_lease: CertifiedLinearLowerWorkerAdmission,
    mut now_during_setup: N,
    after_problem_clone: H,
    before_worker_spawn: S,
) -> Result<Option<ReplayStats>, MipError>
where
    N: FnMut() -> Instant,
    H: FnOnce(),
    S: FnOnce(),
{
    if fixed_assignment_tree_remaining_outer_wait(proof_deadline, now_during_setup()).is_none() {
        return Ok(None);
    }

    // The absolute clock was established by the caller before entry. Optional
    // diagnostic construction and emission therefore consume the same budget
    // as the full proof.
    let _ = maybe_dump_linear_lower_decision(problem, &objective, requested_lower);
    if fixed_assignment_tree_remaining_outer_wait(proof_deadline, now_during_setup()).is_none() {
        return Ok(None);
    }

    // Keep the complete original MILP alive inside a detached proof worker.
    // The post-clone sample is critical: cloning a large encoded network must
    // never buy AY a fresh relative slice.
    let problem = problem.clone();
    after_problem_clone();
    if fixed_assignment_tree_remaining_outer_wait(proof_deadline, now_during_setup()).is_none() {
        return Ok(None);
    }

    before_worker_spawn();
    run_with_hard_deadline_at(
        proof_deadline,
        "linear-lower-fixed-assignment-tree-proof-until",
        move || {
            // If one exact operation is non-interruptible at the boundary,
            // retain process-wide admission until the detached worker exits.
            let _worker_lease = worker_lease;
            let Some(remaining) =
                fixed_assignment_tree_remaining_outer_wait(proof_deadline, Instant::now())
            else {
                return Ok(None);
            };
            let opts = lp_solve_policy.apply(
                solve_opts(remaining.as_secs_f64())
                    .with_deadline(proof_deadline)
                    .with_tree_cert_leaves(0)
                    .with_require_certificates(true),
            );
            if !fixed_assignment_tree_deadline_open(proof_deadline) {
                return Ok(None);
            }
            try_relaxed_linear_lower_fixed_assignment_tree_until(
                &problem,
                &objective,
                requested_lower,
                &splits,
                max_tree_leaves,
                proof_deadline,
                &opts,
            )
        },
    )
    .map(Option::flatten)
}

#[allow(clippy::too_many_arguments)]
fn solve_and_replay_fixed_assignment_tree_proof_until(
    problem: &MilpProblem,
    objective: Vec<(Col, f64)>,
    requested_lower: f32,
    splits: Vec<Col>,
    proof_deadline: Instant,
    max_tree_leaves: usize,
    lp_solve_policy: LpSolvePolicy,
    worker_lease: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<ReplayStats>, MipError> {
    solve_and_replay_fixed_assignment_tree_proof_until_with_setup_control(
        problem,
        objective,
        requested_lower,
        splits,
        proof_deadline,
        max_tree_leaves,
        lp_solve_policy,
        worker_lease,
        Instant::now,
        || {},
        || {},
    )
}

#[allow(clippy::too_many_arguments)]
fn solve_and_replay_parallel_selector_tree_proof(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    selectors: &[Col],
    max_workers: usize,
    timeout_secs: f64,
    max_tree_leaves: usize,
    worker_lease: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<ReplayStats>, MipError> {
    // Start the one authoritative clock before diagnostic dumping or cloning.
    // All leaf sessions receive this exact Instant; no worker can manufacture
    // a fresh per-leaf budget.
    let wall = Duration::from_secs_f64(timeout_secs);
    let proof_deadline = Instant::now()
        .checked_add(wall)
        .ok_or_else(|| MipError::Encoding("linear-lower proof deadline overflow".to_owned()))?;
    let _ = maybe_dump_linear_lower_decision(problem, objective, requested_lower);
    let problem = problem.clone();
    let objective = objective.to_vec();
    let selectors = selectors.to_vec();

    run_with_hard_deadline_at(
        proof_deadline,
        "linear-lower-parallel-selector-tree-proof",
        move || {
            // A detached coordinator owns this lease until its scoped pool has
            // joined every AY leaf worker and exact replay has returned.
            let _worker_lease = worker_lease;
            let Some(remaining) =
                parallel_selector_remaining_outer_wait(proof_deadline, Instant::now())
            else {
                return Ok(None);
            };
            let remaining_secs = remaining.as_secs_f64();
            let opts = solve_opts(remaining_secs)
                .with_deadline(proof_deadline)
                .with_tree_cert_leaves(0)
                .with_require_certificates(true);
            try_relaxed_linear_lower_parallel_selector_tree(
                &problem,
                &objective,
                requested_lower,
                &selectors,
                max_workers,
                max_tree_leaves,
                proof_deadline,
                &opts,
            )
        },
    )
    .map(Option::flatten)
}

fn solve_and_replay_adaptive_three_leaf_target_fsb_proof(
    problem: MilpProblem,
    objective: Vec<(Col, f64)>,
    requested_lower: f32,
    candidates: Vec<Col>,
    root_candidate_index: usize,
    hard_value: bool,
    timeout_secs: f64,
    max_tree_leaves: usize,
    worker_lease: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<ReplayStats>, MipError> {
    let _ = maybe_dump_linear_lower_decision(&problem, &objective, requested_lower);
    let proof_deadline = Instant::now()
        .checked_add(Duration::from_secs_f64(timeout_secs))
        .ok_or_else(|| MipError::Encoding("linear-lower proof deadline overflow".to_owned()))?;
    run_with_hard_deadline(
        timeout_secs,
        "linear-lower-adaptive-three-leaf-target-fsb-proof",
        move || {
            // Retain process-wide admission if the hard deadline detaches this
            // exact AY/replay worker.
            let _worker_lease = worker_lease;
            let opts = solve_opts(timeout_secs)
                .with_deadline(proof_deadline)
                .with_tree_cert_leaves(0)
                .with_require_certificates(true);
            try_relaxed_linear_lower_adaptive_three_leaf_target_fsb(
                &problem,
                &objective,
                requested_lower,
                &candidates,
                root_candidate_index,
                hard_value,
                max_tree_leaves,
                &opts,
            )
        },
    )
    .map(Option::flatten)
}

fn solve_and_replay_adaptive_four_leaf_comb_target_fsb_proof(
    problem: MilpProblem,
    objective: Vec<(Col, f64)>,
    requested_lower: f32,
    candidates: Vec<Col>,
    root_candidate_index: usize,
    root_hard_value: bool,
    timeout_secs: f64,
    max_tree_leaves: usize,
    worker_lease: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<ReplayStats>, MipError> {
    let _ = maybe_dump_linear_lower_decision(&problem, &objective, requested_lower);
    let proof_deadline = Instant::now()
        .checked_add(Duration::from_secs_f64(timeout_secs))
        .ok_or_else(|| MipError::Encoding("linear-lower proof deadline overflow".to_owned()))?;
    run_with_hard_deadline(
        timeout_secs,
        "linear-lower-adaptive-four-leaf-comb-target-fsb-proof",
        move || {
            // Retain process-wide admission if the hard deadline detaches this
            // exact AY/replay worker.
            let _worker_lease = worker_lease;
            let opts = solve_opts(timeout_secs)
                .with_deadline(proof_deadline)
                .with_tree_cert_leaves(0)
                .with_require_certificates(true);
            try_relaxed_linear_lower_adaptive_four_leaf_comb_target_fsb(
                &problem,
                &objective,
                requested_lower,
                &candidates,
                root_candidate_index,
                root_hard_value,
                max_tree_leaves,
                &opts,
            )
        },
    )
    .map(Option::flatten)
}

fn solve_and_replay_adaptive_five_leaf_comb_target_fsb_proof(
    problem: MilpProblem,
    objective: Vec<(Col, f64)>,
    requested_lower: f32,
    candidates: Vec<Col>,
    root_candidate_index: usize,
    root_hard_value: bool,
    timeout_secs: f64,
    max_tree_leaves: usize,
    lp_solve_policy: LpSolvePolicy,
    worker_lease: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<ReplayStats>, MipError> {
    let _ = maybe_dump_linear_lower_decision(&problem, &objective, requested_lower);
    let proof_deadline = Instant::now()
        .checked_add(Duration::from_secs_f64(timeout_secs))
        .ok_or_else(|| MipError::Encoding("linear-lower proof deadline overflow".to_owned()))?;
    run_with_hard_deadline(
        timeout_secs,
        "linear-lower-adaptive-five-leaf-comb-target-fsb-proof",
        move || {
            // Retain process-wide admission if the hard deadline detaches this
            // exact AY/replay worker.
            let _worker_lease = worker_lease;
            let opts = lp_solve_policy.apply(
                solve_opts(timeout_secs)
                    .with_deadline(proof_deadline)
                    .with_tree_cert_leaves(0)
                    .with_require_certificates(true),
            );
            try_relaxed_linear_lower_adaptive_five_leaf_comb_target_fsb(
                &problem,
                &objective,
                requested_lower,
                &candidates,
                root_candidate_index,
                root_hard_value,
                max_tree_leaves,
                &opts,
            )
        },
    )
    .map(Option::flatten)
}

fn solve_and_replay_decision_proof(
    problem: MilpProblem,
    objective: Vec<(Col, f64)>,
    requested_lower: f32,
    branch_advice: Vec<Col>,
    timeout_secs: f64,
    max_tree_leaves: usize,
    worker_lease: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<ReplayStats>, MipError> {
    solve_and_replay_decision_proof_with_target_fsb_probe_limits(
        problem,
        objective,
        requested_lower,
        branch_advice,
        timeout_secs,
        max_tree_leaves,
        worker_lease,
        CertifiedLinearLowerTargetFsbProbeLimits::production(),
    )
}

fn solve_and_replay_decision_proof_with_target_fsb_probe_limits(
    problem: MilpProblem,
    objective: Vec<(Col, f64)>,
    requested_lower: f32,
    branch_advice: Vec<Col>,
    timeout_secs: f64,
    max_tree_leaves: usize,
    worker_lease: CertifiedLinearLowerWorkerAdmission,
    target_fsb_probe_limits: CertifiedLinearLowerTargetFsbProbeLimits,
) -> Result<Option<ReplayStats>, MipError> {
    let _ = maybe_dump_linear_lower_decision(&problem, &objective, requested_lower);
    let wall = Duration::from_secs_f64(timeout_secs);
    let proof_deadline = Instant::now()
        .checked_add(wall)
        .ok_or_else(|| MipError::Encoding("linear-lower proof deadline overflow".to_owned()))?;
    run_with_hard_deadline(timeout_secs, "linear-lower-proof", move || {
        // Exact replay can outlive the caller's hard deadline too.  Keep new
        // proof attempts shed until this detached worker has really stopped.
        let _worker_lease = worker_lease;
        let relaxation_started = Instant::now();
        let advised_assignment_tree_enabled = branch_advice.len() >= 2 && max_tree_leaves >= 4;
        let advised_split_enabled = !branch_advice.is_empty() && max_tree_leaves >= 2;
        let (relaxation_share, relaxation_cap) = if advised_assignment_tree_enabled {
            (
                ADVISED_ASSIGNMENT_TREE_PROOF_SHARE,
                ADVISED_ASSIGNMENT_TREE_PROOF_CAP,
            )
        } else if advised_split_enabled {
            (ADVISED_RELAXATION_PROOF_SHARE, ADVISED_RELAXATION_PROOF_CAP)
        } else {
            (RELAXATION_PROOF_SHARE, RELAXATION_PROOF_CAP)
        };
        let relaxation_wall = proof_deadline
            .checked_duration_since(relaxation_started)
            .unwrap_or(Duration::ZERO)
            .mul_f64(relaxation_share)
            .min(relaxation_cap);
        let relaxation_deadline = relaxation_started
            .checked_add(relaxation_wall)
            .unwrap_or(proof_deadline)
            .min(proof_deadline);
        let relaxation_opts = solve_opts(relaxation_wall.as_secs_f64())
            .with_deadline(relaxation_deadline)
            .with_tree_cert_leaves(0)
            .with_require_certificates(true);
        let relaxation_t0 = Instant::now();
        if mip_trace_armed() {
            eprintln!(
                "NY_MIP_TRACE relaxation proof budget: {:.3}s",
                relaxation_wall.as_secs_f64()
            );
        }
        let relaxation_replay = try_relaxed_linear_lower_proof_with_target_fsb_probe_limits(
            &problem,
            &objective,
            requested_lower,
            &branch_advice,
            max_tree_leaves,
            &relaxation_opts,
            target_fsb_probe_limits,
        )?;
        if mip_trace_armed() {
            eprintln!(
                "NY_MIP_TRACE relaxation proof finished: {:.3}s certified={}",
                relaxation_t0.elapsed().as_secs_f64(),
                relaxation_replay.is_some()
            );
        }
        if let Some(replay) = relaxation_replay {
            return Ok(Some(replay));
        }

        let Some(remaining) = proof_deadline.checked_duration_since(Instant::now()) else {
            return Ok(None);
        };
        if remaining.is_zero() {
            return Ok(None);
        }
        let decision = linear_lower_decision_problem(&problem, &objective, requested_lower);
        if mip_trace_armed() {
            eprintln!(
                "NY_MIP_TRACE decision MILP fallback budget: {:.3}s",
                remaining.as_secs_f64()
            );
        }
        // PIN THIS SOLVE ON NATIVE BRANCH-AND-BOUND. AY `e431ad018` added exact
        // structure-recognition routes that claim an ordinary native check
        // first, and they do not export the artifact this lane replays: native
        // B&B is "the only lane exporting a root Farkas or a whole-tree
        // case-split certificate" (ay `session.rs:5254-5258`). Without this the
        // tree proof simply stops arriving and the bound is refused —
        // `integrality_gap_lower_bound_replays_every_tree_leaf` fails against
        // AY >= e431ad018 and passes with the routes off.
        //
        // Typed and per-session, so it is in-policy: the process-wide spelling
        // is `AY_MILP_NO_STRUCTURE_ROUTE`, which `docs/SOLVER_POLICY.md` forbids
        // NY from writing, and it is a `OnceLock` read that would latch the
        // whole process off the first solve to touch it.
        let fallback_opts = solve_opts(remaining.as_secs_f64())
            .with_deadline(proof_deadline)
            .with_tree_cert_leaves(max_tree_leaves)
            .with_require_certificates(true)
            .with_structure_routing(false);
        solve_and_replay_milp_proof(&decision, &fallback_opts, max_tree_leaves, &branch_advice)
    })
    .map(Option::flatten)
}

/// Propose and independently certify a lower bound on a fixed linear form.
///
/// `problem` may contain continuous and binary columns, but must not carry a
/// marked margin row: this function appends its own typed decision row and
/// refuses any competing solver-routing identity.  `objective` is interpreted
/// over the existing columns and may not contain duplicates or zeros.
///
/// `Ok(None)` is the ordinary fail-closed outcome: AY did not complete the
/// proposal, could not export a strict relaxation entailment or bounded proof
/// tree, or the separate decision problem was feasible/inconclusive. A
/// malformed call or a disagreement between exact checkers is an error.
pub fn certify_linear_lower_bound_with_ay(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    config: CertifiedLinearLowerBoundConfig,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    validate_config(config)?;
    if problem.margin_row().is_some() {
        return Err(MipError::Encoding(
            "certified linear lower bound refuses a pre-marked decision row".to_owned(),
        ));
    }
    let objective = canonical_objective(problem, objective)?;
    let Some(proposal_admission) = CertifiedLinearLowerWorkerAdmission::try_acquire() else {
        tracing::warn!(
            "certified linear lower bound declined: a prior exact AY worker is still active"
        );
        return Ok(None);
    };
    certify_linear_lower_bound_with_ay_prepared(problem, objective, config, proposal_admission)
}

/// Certify a fixed linear lower bound using an admission acquired before model
/// construction.
///
/// This is equivalent to [`certify_linear_lower_bound_with_ay`], except the
/// caller supplies the opaque exact-worker slot. It exists so large encoders can
/// decline before allocating their model when a prior hard-deadline worker is
/// still alive.
pub fn certify_linear_lower_bound_with_ay_admission(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    config: CertifiedLinearLowerBoundConfig,
    proposal_admission: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    validate_config(config)?;
    if problem.margin_row().is_some() {
        return Err(MipError::Encoding(
            "certified linear lower bound refuses a pre-marked decision row".to_owned(),
        ));
    }
    let objective = canonical_objective(problem, objective)?;
    certify_linear_lower_bound_with_ay_prepared(problem, objective, config, proposal_admission)
}

/// Prove a caller-selected lower threshold without first optimizing a proposal.
///
/// The returned `lower` is exactly `requested_lower`. It has authority only
/// because AY either derived a strictly stronger objective row from the
/// continuous relaxation or proved
/// `problem ∧ objective <= requested_lower` infeasible. The resulting exact
/// linear obligation is independently replayed by ny-cert. A feasible
/// equality, timeout, missing bounded certificate, or replay disagreement
/// returns no bound (or an error for malformed/checker-divergent input).
pub fn certify_linear_lower_bound_at_with_ay(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    config: CertifiedLinearLowerDecisionConfig,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    validate_decision_config(config)?;
    if problem.margin_row().is_some() {
        return Err(MipError::Encoding(
            "certified linear lower bound refuses a pre-marked decision row".to_owned(),
        ));
    }
    if !requested_lower.is_finite() {
        return Err(MipError::Encoding(
            "requested certified lower threshold must be finite".to_owned(),
        ));
    }
    let objective = canonical_objective(problem, objective)?;
    let Some(proof_admission) = CertifiedLinearLowerWorkerAdmission::try_acquire() else {
        tracing::warn!(
            "certified linear lower threshold declined: a prior exact AY worker is still active"
        );
        return Ok(None);
    };
    certify_linear_lower_bound_at_with_ay_prepared(
        problem,
        objective,
        requested_lower,
        Vec::new(),
        config,
        proof_admission,
    )
}

/// Prove a caller-selected lower threshold with ordered binary branch advice.
///
/// Advice can change only proof scheduling. Unique, unfixed integer `[0, 1]`
/// columns are retained in caller order; out-of-range, continuous, fixed, and
/// duplicate handles are ignored. One retained column is tried as one complete
/// warm LP split. With exactly two retained columns and room for four leaves,
/// they form a complete depth-two assignment tree. Three through eight use a
/// bounded target-objective strong-branching scan to select the two tree
/// columns; more than eight preserve the fixed first-two control. Probe
/// rankings have no proof authority. The whole retained list guides fallback
/// search, and every successful root or split proof still crosses both exact
/// verification layers.
pub fn certify_linear_lower_bound_at_with_ay_branch_advice(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    branch_advice: &[Col],
    config: CertifiedLinearLowerDecisionConfig,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    certify_linear_lower_bound_at_with_ay_branch_advice_impl(
        problem,
        objective,
        requested_lower,
        branch_advice,
        config,
        CertifiedLinearLowerTargetFsbProbeLimits::production(),
    )
}

/// Diagnostic-only sibling of
/// [`certify_linear_lower_bound_at_with_ay_branch_advice`].
///
/// This unwired API permits measurement harnesses to vary only target-FSB's
/// per-call pivot cap and shared probe duration. The call-count and scratch
/// ceilings remain fixed, the outer proof deadline remains authoritative, and
/// the selected four-leaf proof crosses the same exact verification layers.
/// Production callers should use the ordinary entry point, which always
/// applies [`CertifiedLinearLowerTargetFsbProbeLimits::production`].
pub fn certify_linear_lower_bound_at_with_ay_branch_advice_with_target_fsb_probe_limits_unwired(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    branch_advice: &[Col],
    config: CertifiedLinearLowerDecisionConfig,
    target_fsb_probe_limits: CertifiedLinearLowerTargetFsbProbeLimits,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    certify_linear_lower_bound_at_with_ay_branch_advice_impl(
        problem,
        objective,
        requested_lower,
        branch_advice,
        config,
        target_fsb_probe_limits,
    )
}

fn prepare_fixed_assignment_tree_request(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    splits: &[Col],
    config: CertifiedLinearLowerDecisionConfig,
) -> Result<(Vec<(Col, f64)>, Vec<Col>), MipError> {
    validate_decision_config(config)?;
    if problem.margin_row().is_some() {
        return Err(MipError::Encoding(
            "certified linear lower bound refuses a pre-marked decision row".to_owned(),
        ));
    }
    if !requested_lower.is_finite() {
        return Err(MipError::Encoding(
            "requested certified lower threshold must be finite".to_owned(),
        ));
    }
    let objective = canonical_objective(problem, objective)?;
    let splits = fixed_assignment_tree_splits(problem, splits)?;
    Ok((objective, splits))
}

fn prepare_fixed_assignment_tree_request_until(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    splits: &[Col],
    proof_deadline: Instant,
    max_tree_leaves: usize,
) -> Result<Option<(Vec<(Col, f64)>, Vec<Col>)>, MipError> {
    if fixed_assignment_tree_remaining_outer_wait(proof_deadline, Instant::now()).is_none() {
        return Ok(None);
    }
    if max_tree_leaves == 0 || max_tree_leaves > CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES {
        return Err(MipError::Encoding(format!(
            "max_tree_leaves must be in 1..={CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES}, got \
             {max_tree_leaves}"
        )));
    }
    if problem.margin_row().is_some() {
        return Err(MipError::Encoding(
            "certified linear lower bound refuses a pre-marked decision row".to_owned(),
        ));
    }
    if !requested_lower.is_finite() {
        return Err(MipError::Encoding(
            "requested certified lower threshold must be finite".to_owned(),
        ));
    }
    let objective = canonical_objective(problem, objective);
    if fixed_assignment_tree_remaining_outer_wait(proof_deadline, Instant::now()).is_none() {
        return Ok(None);
    }
    let objective = objective?;
    let splits = fixed_assignment_tree_splits(problem, splits);
    if fixed_assignment_tree_remaining_outer_wait(proof_deadline, Instant::now()).is_none() {
        return Ok(None);
    }
    let splits = splits?;
    Ok(Some((objective, splits)))
}

fn certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_prepared(
    problem: &MilpProblem,
    objective: Vec<(Col, f64)>,
    requested_lower: f32,
    splits: Vec<Col>,
    config: CertifiedLinearLowerDecisionConfig,
    lp_solve_policy: LpSolvePolicy,
    proof_admission: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    let Some(replay) = solve_and_replay_fixed_assignment_tree_proof(
        problem.clone(),
        objective,
        requested_lower,
        splits,
        config.proof_timeout_secs,
        config.max_tree_leaves,
        lp_solve_policy,
        proof_admission,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(CertifiedLinearLowerBound {
        lower: requested_lower,
        proof_route: replay.proof_route,
        ay_tree_leaves: replay.tree_leaves,
        ny_cert_farkas_replays: replay.linear_replays,
    }))
}

fn certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_prepared_until(
    problem: &MilpProblem,
    objective: Vec<(Col, f64)>,
    requested_lower: f32,
    splits: Vec<Col>,
    proof_deadline: Instant,
    max_tree_leaves: usize,
    proof_admission: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    let Some(replay) = solve_and_replay_fixed_assignment_tree_proof_until(
        problem,
        objective,
        requested_lower,
        splits,
        proof_deadline,
        max_tree_leaves,
        LpSolvePolicy::Default,
        proof_admission,
    )?
    else {
        return Ok(None);
    };
    if !fixed_assignment_tree_deadline_open(proof_deadline) {
        return Ok(None);
    }
    Ok(Some(CertifiedLinearLowerBound {
        lower: requested_lower,
        proof_route: replay.proof_route,
        ay_tree_leaves: replay.tree_leaves,
        ny_cert_farkas_replays: replay.linear_replays,
    }))
}

/// Admission-owning fixed complete-assignment-tree proof.
///
/// This is the production-gated counterpart to
/// [`certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_unwired`].
/// The caller must reserve the process-wide exact-worker admission before
/// constructing a potentially large model. The lease is consumed by this
/// call and remains owned by a detached proof worker until that worker really
/// exits.
///
/// `splits` must contain one through four distinct, unfixed integer `[0, 1]`
/// columns. Their caller order is proof-critical and is preserved exactly:
/// AY treats `splits[0]` as the most-significant assignment bit and harvests
/// all `2^splits.len()` leaves. A malformed handle is rejected rather than
/// filtered or retargeted.
///
/// This lane has no selector, adaptive tree, or decision-MILP fallback. It
/// returns a bound only when AY exports either a strictly stronger root row or
/// a complete fixed-assignment tree, AY verifies the composed certificate, and
/// NY independently replays every linear obligation.
pub fn certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_admission(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    splits: &[Col],
    config: CertifiedLinearLowerDecisionConfig,
    proof_admission: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    let (objective, splits) =
        prepare_fixed_assignment_tree_request(problem, objective, requested_lower, splits, config)?;
    certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_prepared(
        problem,
        objective,
        requested_lower,
        splits,
        config,
        LpSolvePolicy::Default,
        proof_admission,
    )
}

/// Admission-owning fixed complete-assignment-tree proof with AY's scoped
/// range-logical triangular-crash initialization.
///
/// This has the same validation, exact AY certificate verification, complete
/// tree requirement, and independent ny-cert replay contract as
/// [`certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_admission`].
/// Only the float-LP initialization advice differs, and the typed request is
/// scoped to the session constructed by this call.
pub fn certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_range_logical_admission(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    splits: &[Col],
    config: CertifiedLinearLowerDecisionConfig,
    proof_admission: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    let (objective, splits) =
        prepare_fixed_assignment_tree_request(problem, objective, requested_lower, splits, config)?;
    certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_prepared(
        problem,
        objective,
        requested_lower,
        splits,
        config,
        LpSolvePolicy::RangeLogical,
        proof_admission,
    )
}

/// Admission-owning fixed complete-assignment-tree proof with the scoped AY
/// selector solve profile.
///
/// This has the same validation, exact AY certificate verification, complete
/// tree requirement, and independent ny-cert replay contract as
/// [`certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_admission`].
/// Only session-local float-LP advice differs: range-logical triangular crash
/// is combined with an explicit
/// [`CERTIFIED_LINEAR_LOWER_SELECTOR_CHAIN_DISTRESS_PROBE_ITERS`] budget.
pub fn certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_selector_solve_profile_admission(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    splits: &[Col],
    config: CertifiedLinearLowerDecisionConfig,
    proof_admission: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    let (objective, splits) =
        prepare_fixed_assignment_tree_request(problem, objective, requested_lower, splits, config)?;
    certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_prepared(
        problem,
        objective,
        requested_lower,
        splits,
        config,
        LpSolvePolicy::SelectorSolveProfile,
        proof_admission,
    )
}

/// Admission-owning fixed complete-assignment-tree proof with the compact-tail
/// root-probe and progressive-prefix warm start.
///
/// This changes float advice only. AY's root probe is capped at 50 ms, each
/// progressive prefix at 25 ms, and the Gray walk starts at assignment zero.
/// Neither a stopped probe nor a prefix status is evidence. A result still
/// requires a strictly sufficient exact root row or every fixed-assignment
/// leaf, AY's exact composed-certificate verification, and independent
/// `ny-cert` replay of every returned linear obligation.
///
/// The caller-provided proof timeout is the single outer deadline and is never
/// extended by either local advice cap.
pub fn certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_compact_progressive_admission(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    splits: &[Col],
    config: CertifiedLinearLowerDecisionConfig,
    proof_admission: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    let (objective, splits) =
        prepare_fixed_assignment_tree_request(problem, objective, requested_lower, splits, config)?;
    certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_prepared(
        problem,
        objective,
        requested_lower,
        splits,
        config,
        LpSolvePolicy::CompactTailProgressive,
        proof_admission,
    )
}

/// Diagnostic-only fixed complete-assignment-tree proof.
///
/// `splits` must contain one through four distinct, unfixed integer `[0, 1]`
/// columns. Their caller order is proof-critical and is preserved exactly:
/// AY treats `splits[0]` as the most-significant assignment bit and harvests
/// all `2^splits.len()` leaves. A malformed handle is rejected rather than
/// filtered or retargeted.
///
/// This unwired lane has no selector, adaptive tree, or decision-MILP fallback.
/// It returns a bound only when AY exports either a strictly stronger root row
/// or a complete fixed-assignment tree, AY verifies the composed certificate,
/// and NY independently replays every linear obligation.
pub fn certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_unwired(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    splits: &[Col],
    config: CertifiedLinearLowerDecisionConfig,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    let (objective, splits) =
        prepare_fixed_assignment_tree_request(problem, objective, requested_lower, splits, config)?;
    let Some(proof_admission) = CertifiedLinearLowerWorkerAdmission::try_acquire() else {
        tracing::warn!(
            "fixed assignment-tree replay declined: a prior exact AY worker is still active"
        );
        return Ok(None);
    };
    certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_prepared(
        problem,
        objective,
        requested_lower,
        splits,
        config,
        LpSolvePolicy::Default,
        proof_admission,
    )
}

/// Diagnostic-only fixed complete-assignment-tree proof against one absolute
/// deadline.
///
/// Unlike
/// [`certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_unwired`],
/// this API never constructs a relative proof slice. `proof_deadline` must be
/// established by the caller before entry and is passed unchanged through
/// optional dumping, the full problem clone, AY setup/solve, exact whole-tree
/// validation, and every ny-cert replay. An expired or nearly expired request
/// is an ordinary fail-closed decline and does not spawn a worker.
///
/// `splits` has the same proof-critical contract as the relative sibling: one
/// through four distinct, unfixed integer `[0, 1]` columns in exact caller
/// order. `max_tree_leaves` is bounded by
/// [`CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES`].
pub fn certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_until_unwired(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    splits: &[Col],
    proof_deadline: Instant,
    max_tree_leaves: usize,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    let Some((objective, splits)) = prepare_fixed_assignment_tree_request_until(
        problem,
        objective,
        requested_lower,
        splits,
        proof_deadline,
        max_tree_leaves,
    )?
    else {
        return Ok(None);
    };
    let Some(proof_admission) = CertifiedLinearLowerWorkerAdmission::try_acquire() else {
        tracing::warn!(
            "absolute-deadline fixed assignment-tree replay declined: a prior exact AY worker is \
             still active"
        );
        return Ok(None);
    };
    certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_prepared_until(
        problem,
        objective,
        requested_lower,
        splits,
        proof_deadline,
        max_tree_leaves,
        proof_admission,
    )
}

fn prepare_parallel_selector_tree_request(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    selectors: &[Col],
    max_workers: usize,
    config: CertifiedLinearLowerDecisionConfig,
) -> Result<(Vec<(Col, f64)>, Vec<Col>), MipError> {
    validate_decision_config(config)?;
    if problem.margin_row().is_some() {
        return Err(MipError::Encoding(
            "certified linear lower bound refuses a pre-marked decision row".to_owned(),
        ));
    }
    if !requested_lower.is_finite() {
        return Err(MipError::Encoding(
            "requested certified lower threshold must be finite".to_owned(),
        ));
    }
    let objective = canonical_objective(problem, objective)?;
    let selectors = parallel_selector_tree_request(problem, selectors, max_workers)?;
    Ok((objective, selectors))
}

#[allow(clippy::too_many_arguments)]
fn certify_linear_lower_bound_at_with_ay_parallel_selector_tree_prepared(
    problem: &MilpProblem,
    objective: Vec<(Col, f64)>,
    requested_lower: f32,
    selectors: Vec<Col>,
    max_workers: usize,
    config: CertifiedLinearLowerDecisionConfig,
    proof_admission: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    let Some(replay) = solve_and_replay_parallel_selector_tree_proof(
        problem,
        &objective,
        requested_lower,
        &selectors,
        max_workers,
        config.proof_timeout_secs,
        config.max_tree_leaves,
        proof_admission,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(CertifiedLinearLowerBound {
        lower: requested_lower,
        proof_route: replay.proof_route,
        ay_tree_leaves: replay.tree_leaves,
        ny_cert_farkas_replays: replay.linear_replays,
    }))
}

/// Admission-owning bounded-parallel four-selector proof canary.
///
/// This API is deliberately unwired from production graph-MIP routing. The
/// four `selectors` are preserved in caller/MSB order and all sixteen fixed
/// assignments are solved under one shared absolute deadline by at most
/// `max_workers` scoped workers. A result is returned only after every leaf is
/// present, strictly sufficient or directly infeasible, associated with its
/// canonical assignment, composed into a complete AY tree, verified by AY,
/// and independently replayed by ny-cert.
///
/// `max_workers` must be in `1..=16`. The caller-supplied admission remains
/// owned until all inner workers exit, including when the outer hard deadline
/// detaches the coordinator.
#[allow(clippy::too_many_arguments)]
pub fn certify_linear_lower_bound_at_with_ay_parallel_selector_tree_admission(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    selectors: &[Col],
    max_workers: usize,
    config: CertifiedLinearLowerDecisionConfig,
    proof_admission: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    let (objective, selectors) = prepare_parallel_selector_tree_request(
        problem,
        objective,
        requested_lower,
        selectors,
        max_workers,
        config,
    )?;
    certify_linear_lower_bound_at_with_ay_parallel_selector_tree_prepared(
        problem,
        objective,
        requested_lower,
        selectors,
        max_workers,
        config,
        proof_admission,
    )
}

/// Explicit opt-in bounded-parallel four-selector proof canary.
///
/// This diagnostic entry point acquires the process-wide exact-worker
/// admission itself and otherwise has the same proof and failure contract as
/// [`certify_linear_lower_bound_at_with_ay_parallel_selector_tree_admission`].
/// It is not called by any production verifier.
#[allow(clippy::too_many_arguments)]
pub fn certify_linear_lower_bound_at_with_ay_parallel_selector_tree_unwired(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    selectors: &[Col],
    max_workers: usize,
    config: CertifiedLinearLowerDecisionConfig,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    let (objective, selectors) = prepare_parallel_selector_tree_request(
        problem,
        objective,
        requested_lower,
        selectors,
        max_workers,
        config,
    )?;
    let Some(proof_admission) = CertifiedLinearLowerWorkerAdmission::try_acquire() else {
        tracing::warn!(
            "parallel selector-tree replay declined: a prior exact AY worker is still active"
        );
        return Ok(None);
    };
    certify_linear_lower_bound_at_with_ay_parallel_selector_tree_prepared(
        problem,
        objective,
        requested_lower,
        selectors,
        max_workers,
        config,
        proof_admission,
    )
}

/// Diagnostic-only adaptive three-leaf target-FSB proof.
///
/// `candidates` is a strict caller-ordered shortlist of two through eight
/// distinct, unfixed binary columns. `root_candidate_index` selects its root
/// split without reordering, and `hard_value` identifies the root child that
/// needs one additional split. AY probes both values of every non-root
/// candidate under that child, then exactly harvests the untouched sibling
/// plus the selected hard child's two grandchildren.
///
/// This unwired lane has fixed advice ceilings: 25 dual pivots per call,
/// 44 calls, 1,500 ms shared probe time, and 128 MiB selector scratch. A tree
/// is authoritative only after report/carrier/three-leaf shape validation,
/// AY's exact whole-tree verification, and three independent ny-cert Farkas
/// replays. The carrier may mix conditional-row and already-infeasible leaves;
/// NY validates their completed Farkas obligations uniformly. There is no
/// complete-tree or decision-MILP fallback, so this entry point cannot alter
/// either existing target-FSB policy.
pub fn certify_linear_lower_bound_at_with_ay_adaptive_three_leaf_target_fsb_unwired(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    candidates: &[Col],
    root_candidate_index: usize,
    hard_value: bool,
    config: CertifiedLinearLowerDecisionConfig,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    validate_decision_config(config)?;
    if problem.margin_row().is_some() {
        return Err(MipError::Encoding(
            "certified linear lower bound refuses a pre-marked decision row".to_owned(),
        ));
    }
    if !requested_lower.is_finite() {
        return Err(MipError::Encoding(
            "requested certified lower threshold must be finite".to_owned(),
        ));
    }
    let objective = canonical_objective(problem, objective)?;
    let candidates = adaptive_three_leaf_candidates(problem, candidates, root_candidate_index)?;
    if config.max_tree_leaves < 3 {
        return Ok(None);
    }
    let Some(proof_admission) = CertifiedLinearLowerWorkerAdmission::try_acquire() else {
        tracing::warn!(
            "adaptive three-leaf target-FSB declined: a prior exact AY worker is still active"
        );
        return Ok(None);
    };
    let Some(replay) = solve_and_replay_adaptive_three_leaf_target_fsb_proof(
        problem.clone(),
        objective,
        requested_lower,
        candidates,
        root_candidate_index,
        hard_value,
        config.proof_timeout_secs,
        config.max_tree_leaves,
        proof_admission,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(CertifiedLinearLowerBound {
        lower: requested_lower,
        proof_route: replay.proof_route,
        ay_tree_leaves: replay.tree_leaves,
        ny_cert_farkas_replays: replay.linear_replays,
    }))
}

/// Diagnostic-only adaptive four-leaf comb target-FSB proof.
///
/// `candidates` is a strict caller-ordered shortlist of three through eight
/// distinct, unfixed binary columns. `root_candidate_index` selects the root
/// split without reordering, and `root_hard_value` identifies the root child
/// refined by the rest of the comb. AY exactifies the opposite root child,
/// target-probes one second split under the hard root value, refines the
/// strictly weaker second child (`false` wins a tie), target-probes a terminal
/// third split below both hard assignments, and exactifies the remaining three
/// leaves.
///
/// This unwired, tree-only lane has fixed advice ceilings: 25 dual pivots per
/// call, 44 calls, 1,500 ms shared probe time, and 128 MiB selector scratch. A
/// result is authoritative only after report/carrier/four-leaf topology
/// validation, AY's exact whole-tree verification, and four independent
/// ny-cert Farkas replays. The carrier may mix conditional-row and already
/// infeasible leaves. There is no root fast path, complete-tree path, or
/// decision-MILP fallback, so this entry point cannot alter production policy.
pub fn certify_linear_lower_bound_at_with_ay_adaptive_four_leaf_comb_target_fsb_unwired(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    candidates: &[Col],
    root_candidate_index: usize,
    root_hard_value: bool,
    config: CertifiedLinearLowerDecisionConfig,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    validate_decision_config(config)?;
    if problem.margin_row().is_some() {
        return Err(MipError::Encoding(
            "certified linear lower bound refuses a pre-marked decision row".to_owned(),
        ));
    }
    if !requested_lower.is_finite() {
        return Err(MipError::Encoding(
            "requested certified lower threshold must be finite".to_owned(),
        ));
    }
    let objective = canonical_objective(problem, objective)?;
    let candidates = adaptive_four_leaf_comb_candidates(problem, candidates, root_candidate_index)?;
    if config.max_tree_leaves < 4 {
        return Ok(None);
    }
    let Some(proof_admission) = CertifiedLinearLowerWorkerAdmission::try_acquire() else {
        tracing::warn!(
            "adaptive four-leaf comb target-FSB declined: a prior exact AY worker is still active"
        );
        return Ok(None);
    };
    let Some(replay) = solve_and_replay_adaptive_four_leaf_comb_target_fsb_proof(
        problem.clone(),
        objective,
        requested_lower,
        candidates,
        root_candidate_index,
        root_hard_value,
        config.proof_timeout_secs,
        config.max_tree_leaves,
        proof_admission,
    )?
    else {
        return Ok(None);
    };
    if replay.proof_route != CertifiedLinearLowerProofRoute::TreeFarkas
        || replay.tree_leaves != 4
        || replay.linear_replays != 4
    {
        return Err(MipError::Solver(format!(
            "adaptive four-leaf comb authority returned route {:?}, {} leaves, and {} replays",
            replay.proof_route, replay.tree_leaves, replay.linear_replays
        )));
    }
    Ok(Some(CertifiedLinearLowerBound {
        lower: requested_lower,
        proof_route: replay.proof_route,
        ay_tree_leaves: replay.tree_leaves,
        ny_cert_farkas_replays: replay.linear_replays,
    }))
}

fn prepare_adaptive_five_leaf_comb_target_fsb_request(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    candidates: &[Col],
    root_candidate_index: usize,
    config: CertifiedLinearLowerDecisionConfig,
) -> Result<Option<(Vec<(Col, f64)>, Vec<Col>)>, MipError> {
    validate_decision_config(config)?;
    if problem.margin_row().is_some() {
        return Err(MipError::Encoding(
            "certified linear lower bound refuses a pre-marked decision row".to_owned(),
        ));
    }
    if !requested_lower.is_finite() {
        return Err(MipError::Encoding(
            "requested certified lower threshold must be finite".to_owned(),
        ));
    }
    let objective = canonical_objective(problem, objective)?;
    let candidates = adaptive_five_leaf_comb_candidates(problem, candidates, root_candidate_index)?;
    if config.max_tree_leaves < 5 {
        return Ok(None);
    }
    Ok(Some((objective, candidates)))
}

#[allow(clippy::too_many_arguments)]
fn certify_linear_lower_bound_at_with_ay_adaptive_five_leaf_comb_target_fsb_prepared(
    problem: &MilpProblem,
    objective: Vec<(Col, f64)>,
    requested_lower: f32,
    candidates: Vec<Col>,
    root_candidate_index: usize,
    root_hard_value: bool,
    config: CertifiedLinearLowerDecisionConfig,
    lp_solve_policy: LpSolvePolicy,
    proof_admission: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    let Some(replay) = solve_and_replay_adaptive_five_leaf_comb_target_fsb_proof(
        problem.clone(),
        objective,
        requested_lower,
        candidates,
        root_candidate_index,
        root_hard_value,
        config.proof_timeout_secs,
        config.max_tree_leaves,
        lp_solve_policy,
        proof_admission,
    )?
    else {
        return Ok(None);
    };
    if replay.proof_route != CertifiedLinearLowerProofRoute::TreeFarkas
        || replay.tree_leaves != 5
        || replay.linear_replays != 5
    {
        return Err(MipError::Solver(format!(
            "adaptive five-leaf comb authority returned route {:?}, {} leaves, and {} replays",
            replay.proof_route, replay.tree_leaves, replay.linear_replays
        )));
    }
    Ok(Some(CertifiedLinearLowerBound {
        lower: requested_lower,
        proof_route: replay.proof_route,
        ay_tree_leaves: replay.tree_leaves,
        ny_cert_farkas_replays: replay.linear_replays,
    }))
}

/// Admission-owning adaptive depth-four five-leaf comb target-FSB proof.
///
/// This is the production-gated counterpart to
/// [`certify_linear_lower_bound_at_with_ay_adaptive_five_leaf_comb_target_fsb_unwired`].
/// The caller must reserve the process-wide exact-worker admission before
/// constructing a potentially large model. The lease is consumed by this
/// call and remains owned by a detached proof worker until that worker really
/// exits.
///
/// Candidate, report, carrier, topology, AY whole-tree, and independent
/// five-leaf ny-cert replay validation is identical to the unwired seam. This
/// narrowly scoped production route additionally requests AY's range-logical
/// triangular-crash LP initialization for its own session.
#[allow(clippy::too_many_arguments)]
pub fn certify_linear_lower_bound_at_with_ay_adaptive_five_leaf_comb_target_fsb_admission(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    candidates: &[Col],
    root_candidate_index: usize,
    root_hard_value: bool,
    config: CertifiedLinearLowerDecisionConfig,
    proof_admission: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    let Some((objective, candidates)) = prepare_adaptive_five_leaf_comb_target_fsb_request(
        problem,
        objective,
        requested_lower,
        candidates,
        root_candidate_index,
        config,
    )?
    else {
        return Ok(None);
    };
    certify_linear_lower_bound_at_with_ay_adaptive_five_leaf_comb_target_fsb_prepared(
        problem,
        objective,
        requested_lower,
        candidates,
        root_candidate_index,
        root_hard_value,
        config,
        LpSolvePolicy::RangeLogical,
        proof_admission,
    )
}

/// Diagnostic-only adaptive depth-four five-leaf comb target-FSB proof.
///
/// `candidates` is a strict caller-ordered shortlist of four through eight
/// distinct, unfixed binary columns. `root_candidate_index` selects the root
/// split without reordering, and `root_hard_value` identifies the root child
/// refined by the rest of the comb. Below that child, AY runs three complete
/// target-FSB scans. The strictly weaker selected value continues at the
/// second and third levels (`false` wins an exact tie), yielding the exact
/// topology root-easy, second-easy, third-easy, and two terminal fourth
/// children.
///
/// This unwired, tree-only lane has fixed advice ceilings: 25 dual pivots per
/// quick call, 44 calls, 1,500 ms shared probe time, and 128 MiB selector
/// scratch. A result is authoritative only after report/carrier/five-leaf
/// topology validation, AY's exact whole-tree verification, and five
/// independent ny-cert Farkas replays. There is no root fast path,
/// complete-tree path, or decision-MILP fallback, so this entry point cannot
/// alter production policy.
pub fn certify_linear_lower_bound_at_with_ay_adaptive_five_leaf_comb_target_fsb_unwired(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    candidates: &[Col],
    root_candidate_index: usize,
    root_hard_value: bool,
    config: CertifiedLinearLowerDecisionConfig,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    let Some((objective, candidates)) = prepare_adaptive_five_leaf_comb_target_fsb_request(
        problem,
        objective,
        requested_lower,
        candidates,
        root_candidate_index,
        config,
    )?
    else {
        return Ok(None);
    };
    let Some(proof_admission) = CertifiedLinearLowerWorkerAdmission::try_acquire() else {
        tracing::warn!(
            "adaptive five-leaf comb target-FSB declined: a prior exact AY worker is still active"
        );
        return Ok(None);
    };
    certify_linear_lower_bound_at_with_ay_adaptive_five_leaf_comb_target_fsb_prepared(
        problem,
        objective,
        requested_lower,
        candidates,
        root_candidate_index,
        root_hard_value,
        config,
        LpSolvePolicy::Default,
        proof_admission,
    )
}

fn certify_linear_lower_bound_at_with_ay_branch_advice_impl(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    branch_advice: &[Col],
    config: CertifiedLinearLowerDecisionConfig,
    target_fsb_probe_limits: CertifiedLinearLowerTargetFsbProbeLimits,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    validate_decision_config(config)?;
    if problem.margin_row().is_some() {
        return Err(MipError::Encoding(
            "certified linear lower bound refuses a pre-marked decision row".to_owned(),
        ));
    }
    if !requested_lower.is_finite() {
        return Err(MipError::Encoding(
            "requested certified lower threshold must be finite".to_owned(),
        ));
    }
    let objective = canonical_objective(problem, objective)?;
    let branch_advice = canonical_binary_branch_advice(problem, branch_advice);
    let Some(proof_admission) = CertifiedLinearLowerWorkerAdmission::try_acquire() else {
        tracing::warn!(
            "certified linear lower threshold declined: a prior exact AY worker is still active"
        );
        return Ok(None);
    };
    certify_linear_lower_bound_at_with_ay_prepared_with_target_fsb_probe_limits(
        problem,
        objective,
        requested_lower,
        branch_advice,
        config,
        proof_admission,
        target_fsb_probe_limits,
    )
}

/// Decision-only counterpart to
/// [`certify_linear_lower_bound_with_ay_admission`].
///
/// The caller supplies an admission acquired before potentially large model
/// construction. No separate proposal worker is launched: the lease is
/// retained by the exact proof worker until that worker really exits.
pub fn certify_linear_lower_bound_at_with_ay_admission(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    config: CertifiedLinearLowerDecisionConfig,
    proof_admission: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    validate_decision_config(config)?;
    if problem.margin_row().is_some() {
        return Err(MipError::Encoding(
            "certified linear lower bound refuses a pre-marked decision row".to_owned(),
        ));
    }
    if !requested_lower.is_finite() {
        return Err(MipError::Encoding(
            "requested certified lower threshold must be finite".to_owned(),
        ));
    }
    let objective = canonical_objective(problem, objective)?;
    certify_linear_lower_bound_at_with_ay_prepared(
        problem,
        objective,
        requested_lower,
        Vec::new(),
        config,
        proof_admission,
    )
}

/// Admission-owning counterpart to
/// [`certify_linear_lower_bound_at_with_ay_branch_advice`].
///
/// This preserves early workload shedding for large encoders while supplying
/// the same proof-only branch guidance.
pub fn certify_linear_lower_bound_at_with_ay_branch_advice_admission(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    branch_advice: &[Col],
    config: CertifiedLinearLowerDecisionConfig,
    proof_admission: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    validate_decision_config(config)?;
    if problem.margin_row().is_some() {
        return Err(MipError::Encoding(
            "certified linear lower bound refuses a pre-marked decision row".to_owned(),
        ));
    }
    if !requested_lower.is_finite() {
        return Err(MipError::Encoding(
            "requested certified lower threshold must be finite".to_owned(),
        ));
    }
    let objective = canonical_objective(problem, objective)?;
    let branch_advice = canonical_binary_branch_advice(problem, branch_advice);
    certify_linear_lower_bound_at_with_ay_prepared(
        problem,
        objective,
        requested_lower,
        branch_advice,
        config,
        proof_admission,
    )
}

fn certify_linear_lower_bound_at_with_ay_prepared(
    problem: &MilpProblem,
    objective: Vec<(Col, f64)>,
    requested_lower: f32,
    branch_advice: Vec<Col>,
    config: CertifiedLinearLowerDecisionConfig,
    proof_admission: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    certify_linear_lower_bound_at_with_ay_prepared_with_target_fsb_probe_limits(
        problem,
        objective,
        requested_lower,
        branch_advice,
        config,
        proof_admission,
        CertifiedLinearLowerTargetFsbProbeLimits::production(),
    )
}

fn certify_linear_lower_bound_at_with_ay_prepared_with_target_fsb_probe_limits(
    problem: &MilpProblem,
    objective: Vec<(Col, f64)>,
    requested_lower: f32,
    branch_advice: Vec<Col>,
    config: CertifiedLinearLowerDecisionConfig,
    proof_admission: CertifiedLinearLowerWorkerAdmission,
    target_fsb_probe_limits: CertifiedLinearLowerTargetFsbProbeLimits,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    let Some(replay) = solve_and_replay_decision_proof_with_target_fsb_probe_limits(
        problem.clone(),
        objective,
        requested_lower,
        branch_advice,
        config.proof_timeout_secs,
        config.max_tree_leaves,
        proof_admission,
        target_fsb_probe_limits,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(CertifiedLinearLowerBound {
        lower: requested_lower,
        proof_route: replay.proof_route,
        ay_tree_leaves: replay.tree_leaves,
        ny_cert_farkas_replays: replay.linear_replays,
    }))
}

fn certify_linear_lower_bound_with_ay_prepared(
    problem: &MilpProblem,
    objective: Vec<(Col, f64)>,
    config: CertifiedLinearLowerBoundConfig,
    proposal_admission: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    let Some(optimum) = solve_proposal(
        problem.clone(),
        objective.clone(),
        config.proposal_timeout_secs,
        proposal_admission,
    )?
    else {
        return Ok(None);
    };
    let Some(lower) = strict_outward_f32_lower(&optimum) else {
        return Ok(None);
    };

    let Some(proof_lease) = CertifiedLinearLowerWorkerAdmission::try_acquire() else {
        tracing::warn!(
            "certified linear lower bound declined before replay: another exact AY worker is active"
        );
        return Ok(None);
    };
    let Some(replay) = solve_and_replay_decision_proof(
        problem.clone(),
        objective,
        lower,
        Vec::new(),
        config.proof_timeout_secs,
        config.max_tree_leaves,
        proof_lease,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(CertifiedLinearLowerBound {
        lower,
        proof_route: replay.proof_route,
        ay_tree_leaves: replay.tree_leaves,
        ny_cert_farkas_replays: replay.linear_replays,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_test_utils::env::{lock_env, ScopedEnvVar};

    static CERTIFIED_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn next_down_crosses_positive_infinity_directionally() {
        assert_eq!(next_down_f32(f32::INFINITY), f32::MAX);
        assert_eq!(next_down_f32(f32::NEG_INFINITY), f32::NEG_INFINITY);
    }

    fn config() -> CertifiedLinearLowerBoundConfig {
        CertifiedLinearLowerBoundConfig {
            proposal_timeout_secs: 10.0,
            proof_timeout_secs: 10.0,
            max_tree_leaves: 64,
        }
    }

    fn decision_config() -> CertifiedLinearLowerDecisionConfig {
        CertifiedLinearLowerDecisionConfig {
            proof_timeout_secs: 10.0,
            max_tree_leaves: 64,
        }
    }

    #[test]
    fn lp_solve_policy_is_typed_and_scoped_to_the_selected_options() {
        let default = SolveOpts::new();
        let unchanged = LpSolvePolicy::Default.apply(default.clone());
        let range_only = LpSolvePolicy::RangeLogical.apply(default.clone());
        let selector_profile = LpSolvePolicy::SelectorSolveProfile.apply(default.clone());

        assert!(!default.range_logical_triangular_crash());
        assert!(!unchanged.range_logical_triangular_crash());
        assert!(range_only.chain_distress_probe_iters().is_none());
        assert!(range_only.range_logical_triangular_crash());
        assert!(selector_profile.range_logical_triangular_crash());
        assert_eq!(
            selector_profile.chain_distress_probe_iters(),
            Some(CERTIFIED_LINEAR_LOWER_SELECTOR_CHAIN_DISTRESS_PROBE_ITERS)
        );
        assert_eq!(default.chain_distress_probe_iters(), None);
        assert_eq!(unchanged.chain_distress_probe_iters(), None);
    }

    fn one_split_integrality_gap_problem() -> (MilpProblem, Col, Col) {
        // The relaxation attains z=1/2 at x=1/2, while either binary child
        // forces z>=1. Thus z<=3/4 needs exactly the x=0 / x=1 split.
        let mut problem = MilpProblem::new();
        let x = problem.add_integer_col(0.0, 0.0, 1.0);
        let z = problem.add_col(0.0, 0.0, 2.0);
        problem.add_row(0.0, f64::INFINITY, [(z, 1.0), (x, -1.0)]);
        problem.add_row(1.0, f64::INFINITY, [(z, 1.0), (x, 1.0)]);
        (problem, x, z)
    }

    fn two_split_integrality_gap_problem() -> (MilpProblem, Col, Col, Col) {
        // The relaxed epigraph is
        //
        //   z >= x + y - 1/2
        //   z >= 1/2 - x - y.
        //
        // Its root minimum is zero. After splitting only x, the x=0 child can
        // still take y=1/2 and z=0. Every complete binary assignment to x,y,
        // however, forces z>=1/2, so z<=1/4 needs exactly four leaves.
        let mut problem = MilpProblem::new();
        let x = problem.add_integer_col(0.0, 0.0, 1.0);
        let y = problem.add_integer_col(0.0, 0.0, 1.0);
        let z = problem.add_col(0.0, 0.0, 2.0);
        problem.add_row(-0.5, f64::INFINITY, [(z, 1.0), (x, -1.0), (y, -1.0)]);
        problem.add_row(0.5, f64::INFINITY, [(z, 1.0), (x, 1.0), (y, 1.0)]);
        (problem, x, y, z)
    }

    fn four_split_complete_assignment_problem() -> (MilpProblem, [Col; 4], Col) {
        // The relaxed root takes every binary at 1/2 and z=1/2. At every
        // complete binary assignment, at least one of z>=x_i and z>=1-x_i
        // forces z>=1. The fixed harvester deliberately retains all sixteen
        // caller-ordered leaves even though an adaptive tree could stop early.
        let mut problem = MilpProblem::new();
        let splits = std::array::from_fn(|_| problem.add_integer_col(0.0, 0.0, 1.0));
        let z = problem.add_col(0.0, 0.0, 2.0);
        for split in splits {
            problem.add_row(0.0, f64::INFINITY, [(z, 1.0), (split, -1.0)]);
            problem.add_row(1.0, f64::INFINITY, [(z, 1.0), (split, 1.0)]);
        }
        (problem, splits, z)
    }

    fn four_split_weak_leaf_problem() -> (MilpProblem, [Col; 4], Col) {
        // Every nonzero assignment forces z>=1, but the all-zero leaf permits
        // z=0. One weak leaf must reject the complete harvest.
        let mut problem = MilpProblem::new();
        let splits = std::array::from_fn(|_| problem.add_integer_col(0.0, 0.0, 1.0));
        let z = problem.add_col(0.0, 0.0, 2.0);
        for split in splits {
            problem.add_row(0.0, f64::INFINITY, [(z, 1.0), (split, -1.0)]);
        }
        (problem, splits, z)
    }

    fn four_selector_mixed_evidence_problem() -> (MilpProblem, [Col; 4], Col) {
        // selector[0]=0 contradicts the first row and must be represented by
        // a direct fixed-box Farkas leaf. selector[0]=1 implies z>=1 and
        // produces a strict conditional objective row. The other selectors
        // remain unconstrained so the complete tree still has sixteen leaves.
        let mut problem = MilpProblem::new();
        let selectors = std::array::from_fn(|_| problem.add_integer_col(0.0, 0.0, 1.0));
        let z = problem.add_col(0.0, 0.0, 2.0);
        problem.add_row(1.0, f64::INFINITY, [(selectors[0], 1.0)]);
        problem.add_row(0.0, f64::INFINITY, [(z, 1.0), (selectors[0], -1.0)]);
        (problem, selectors, z)
    }

    fn target_fsb_nonprefix_pair_problem() -> (MilpProblem, Col, Col, Col, Col, Col) {
        // p=max(x, 1-x) and q=max(z, 1-z) in the lower epigraph. At the
        // relaxed root p+q=1. Splitting either useful binary raises the lower
        // bound to 3/2, while splitting both raises it to 2. The unconnected
        // dummy makes caller-prefix [x,dummy] insufficient at 7/4, so a
        // successful four-leaf proof from [x,dummy,z] demonstrates that
        // target-FSB selected the non-prefix pair [x,z].
        let mut problem = MilpProblem::new();
        let x = problem.add_integer_col(0.0, 0.0, 1.0);
        let dummy = problem.add_integer_col(0.0, 0.0, 1.0);
        let z = problem.add_integer_col(0.0, 0.0, 1.0);
        let p = problem.add_col(0.0, 0.0, 2.0);
        let q = problem.add_col(0.0, 0.0, 2.0);
        problem.add_row(0.0, f64::INFINITY, [(p, 1.0), (x, -1.0)]);
        problem.add_row(1.0, f64::INFINITY, [(p, 1.0), (x, 1.0)]);
        problem.add_row(0.0, f64::INFINITY, [(q, 1.0), (z, -1.0)]);
        problem.add_row(1.0, f64::INFINITY, [(q, 1.0), (z, 1.0)]);
        (problem, x, dummy, z, p, q)
    }

    fn adaptive_three_leaf_mixed_problem(hard_value: bool) -> (MilpProblem, Col, Col, Col, Col) {
        // The easy root sibling contradicts the first row and therefore
        // contributes a direct exact-Farkas leaf. On the hard root value,
        // y=1/2 permits z=1/2, while either y endpoint forces z>=1. A dummy
        // candidate stays weak, so the adaptive scan must retain caller root
        // index 1 and select the non-prefix second split y.
        let mut problem = MilpProblem::new();
        let root = problem.add_integer_col(0.0, 0.0, 1.0);
        let dummy = problem.add_integer_col(0.0, 0.0, 1.0);
        let y = problem.add_integer_col(0.0, 0.0, 1.0);
        let z = problem.add_col(0.0, 0.0, 2.0);
        if hard_value {
            problem.add_row(1.0, f64::INFINITY, [(root, 1.0)]);
        } else {
            problem.add_row(f64::NEG_INFINITY, 0.0, [(root, 1.0)]);
        }
        problem.add_row(0.0, f64::INFINITY, [(z, 1.0), (y, -1.0)]);
        problem.add_row(1.0, f64::INFINITY, [(z, 1.0), (y, 1.0)]);
        (problem, root, dummy, y, z)
    }

    #[allow(clippy::type_complexity)]
    fn adaptive_four_leaf_comb_problem(
        root_hard_value: bool,
        second_hard_value: bool,
        infeasible_root_easy: bool,
    ) -> (MilpProblem, Col, Col, Col, Col, Col, Col) {
        // Candidate order at the public seam will be
        // [dummy1, root, dummy2, second, third]. The two dummies retain the
        // current relaxation bound, so both FSB stages must choose non-prefix
        // partners. Every exact comb leaf proves p>7/8. Optionally the root-easy
        // assignment is contradictory, producing a direct Farkas leaf while
        // the other three remain conditional rows.
        let mut problem = MilpProblem::new();
        let root = problem.add_integer_col(0.0, 0.0, 1.0);
        let dummy1 = problem.add_integer_col(0.0, 0.0, 1.0);
        let second = problem.add_integer_col(0.0, 0.0, 1.0);
        let dummy2 = problem.add_integer_col(0.0, 0.0, 1.0);
        let third = problem.add_integer_col(0.0, 0.0, 1.0);
        let p = problem.add_col(0.0, 0.0, 2.0);

        if root_hard_value {
            // Easy root=0: p >= 1-root.
            problem.add_row(1.0, f64::INFINITY, [(p, 1.0), (root, 1.0)]);
        } else {
            // Easy root=1: p >= root.
            problem.add_row(0.0, f64::INFINITY, [(p, 1.0), (root, -1.0)]);
        }

        if second_hard_value {
            // Easy second=0, hard second=1. Fixing third below the hard
            // assignment raises both terminal bounds from 3/4 to 1.
            problem.add_row(1.0, f64::INFINITY, [(p, 1.0), (second, 1.0)]);
            problem.add_row(0.0, f64::INFINITY, [(p, 1.0), (second, -0.75)]);
            problem.add_row(
                -1.0,
                f64::INFINITY,
                [(p, 1.0), (third, -1.0), (second, -1.0)],
            );
            problem.add_row(0.0, f64::INFINITY, [(p, 1.0), (third, 1.0), (second, -1.0)]);
        } else {
            // Easy second=1, hard second=0.
            problem.add_row(0.0, f64::INFINITY, [(p, 1.0), (second, -1.0)]);
            problem.add_row(0.75, f64::INFINITY, [(p, 1.0), (second, 0.75)]);
            problem.add_row(0.0, f64::INFINITY, [(p, 1.0), (third, -1.0), (second, 1.0)]);
            problem.add_row(1.0, f64::INFINITY, [(p, 1.0), (third, 1.0), (second, 1.0)]);
        }

        if infeasible_root_easy {
            if root_hard_value {
                problem.add_row(1.0, f64::INFINITY, [(root, 1.0)]);
            } else {
                problem.add_row(f64::NEG_INFINITY, 0.0, [(root, 1.0)]);
            }
        }
        (problem, root, dummy1, second, dummy2, third, p)
    }

    fn empty_farkas_leaf() -> TreeNode {
        TreeNode::Leaf {
            farkas: AyFarkasCertificate {
                multipliers: Vec::new(),
            },
        }
    }

    fn binary_tree_node(col: ay_milp::Col, lo: TreeNode, hi: TreeNode) -> TreeNode {
        TreeNode::Split {
            col,
            cut: BigRational::zero(),
            lo: Box::new(lo),
            hi: Box::new(hi),
        }
    }

    fn four_leaf_comb_tree(
        root: ay_milp::Col,
        root_hard_value: bool,
        second: ay_milp::Col,
        second_hard_value: bool,
        third: ay_milp::Col,
    ) -> TreeNode {
        let third_node = binary_tree_node(third, empty_farkas_leaf(), empty_farkas_leaf());
        let second_node = if second_hard_value {
            binary_tree_node(second, empty_farkas_leaf(), third_node)
        } else {
            binary_tree_node(second, third_node, empty_farkas_leaf())
        };
        if root_hard_value {
            binary_tree_node(root, empty_farkas_leaf(), second_node)
        } else {
            binary_tree_node(root, second_node, empty_farkas_leaf())
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn five_leaf_comb_tree(
        root: ay_milp::Col,
        root_hard_value: bool,
        second: ay_milp::Col,
        second_hard_value: bool,
        third: ay_milp::Col,
        third_hard_value: bool,
        fourth: ay_milp::Col,
    ) -> TreeNode {
        let fourth_node = binary_tree_node(fourth, empty_farkas_leaf(), empty_farkas_leaf());
        let third_node = if third_hard_value {
            binary_tree_node(third, empty_farkas_leaf(), fourth_node)
        } else {
            binary_tree_node(third, fourth_node, empty_farkas_leaf())
        };
        let second_node = if second_hard_value {
            binary_tree_node(second, empty_farkas_leaf(), third_node)
        } else {
            binary_tree_node(second, third_node, empty_farkas_leaf())
        };
        if root_hard_value {
            binary_tree_node(root, empty_farkas_leaf(), second_node)
        } else {
            binary_tree_node(root, second_node, empty_farkas_leaf())
        }
    }

    #[test]
    fn target_fsb_probe_limits_pin_production_and_reject_zero_resources() {
        let production = CertifiedLinearLowerTargetFsbProbeLimits::production();
        assert_eq!(production.max_probe_pivots_per_call(), 25);
        assert_eq!(production.probe_time_limit(), Duration::from_millis(1_500));

        let diagnostic =
            CertifiedLinearLowerTargetFsbProbeLimits::new(7, Duration::from_millis(250))
                .expect("positive diagnostic limits");
        assert_eq!(diagnostic.max_probe_pivots_per_call(), 7);
        assert_eq!(diagnostic.probe_time_limit(), Duration::from_millis(250));
        assert!(matches!(
            CertifiedLinearLowerTargetFsbProbeLimits::new(0, Duration::from_millis(250)),
            Err(MipError::Encoding(_))
        ));
        assert!(matches!(
            CertifiedLinearLowerTargetFsbProbeLimits::new(7, Duration::ZERO),
            Err(MipError::Encoding(_))
        ));
    }

    #[test]
    fn branch_advice_keeps_unique_live_binaries_in_caller_order() {
        let mut problem = MilpProblem::new();
        let continuous = problem.add_col(0.0, 0.0, 1.0);
        let first = problem.add_integer_col(0.0, 0.0, 1.0);
        let fixed = problem.add_integer_col(0.0, 1.0, 1.0);
        let second = problem.add_integer_col(0.0, 0.0, 1.0);

        assert_eq!(
            canonical_binary_branch_advice(
                &problem,
                &[
                    continuous,
                    second,
                    Col(problem.num_cols() + 10),
                    fixed,
                    first,
                    second,
                ],
            ),
            vec![second, first]
        );
    }

    #[test]
    fn fixed_assignment_tree_splits_preserve_order_and_reject_malformed_lists() {
        let mut problem = MilpProblem::new();
        let first = problem.add_integer_col(0.0, 0.0, 1.0);
        let continuous = problem.add_col(0.0, 0.0, 1.0);
        let second = problem.add_integer_col(0.0, 0.0, 1.0);
        let fixed = problem.add_integer_col(0.0, 1.0, 1.0);
        let third = problem.add_integer_col(0.0, 0.0, 1.0);
        let fourth = problem.add_integer_col(0.0, 0.0, 1.0);

        assert_eq!(
            fixed_assignment_tree_splits(&problem, &[fourth, second, first, third])
                .expect("ordered fixed split list"),
            vec![fourth, second, first, third]
        );
        for malformed in [
            fixed_assignment_tree_splits(&problem, &[]),
            fixed_assignment_tree_splits(&problem, &[first, second, third, fourth, first]),
            fixed_assignment_tree_splits(&problem, &[first, continuous]),
            fixed_assignment_tree_splits(&problem, &[first, fixed]),
            fixed_assignment_tree_splits(&problem, &[first, first]),
            fixed_assignment_tree_splits(&problem, &[first, Col(problem.num_cols() + 1)]),
        ] {
            assert!(
                matches!(malformed, Err(MipError::Encoding(_))),
                "malformed fixed splits must fail instead of changing topology"
            );
        }
    }

    #[test]
    fn parallel_selector_tree_requires_exact_topology_and_bounded_workers() {
        let (problem, selectors, _) = four_split_complete_assignment_problem();
        assert_eq!(
            parallel_selector_tree_request(
                &problem,
                &[selectors[3], selectors[1], selectors[0], selectors[2]],
                16,
            )
            .expect("exact four-selector request"),
            vec![selectors[3], selectors[1], selectors[0], selectors[2]]
        );
        for malformed in [
            parallel_selector_tree_request(&problem, &selectors[..3], 4),
            parallel_selector_tree_request(&problem, &selectors, 0),
            parallel_selector_tree_request(&problem, &selectors, 17),
            parallel_selector_tree_request(
                &problem,
                &[selectors[0], selectors[1], selectors[2], selectors[0]],
                4,
            ),
        ] {
            assert!(
                matches!(malformed, Err(MipError::Encoding(_))),
                "parallel canary must not alter malformed proof topology"
            );
        }
    }

    #[test]
    fn bounded_assignment_pool_is_canonical_and_never_exceeds_its_cap() {
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let thread_names = Mutex::new(Vec::new());
        let deadline = Instant::now() + Duration::from_secs(2);
        let values = run_bounded_canonical_assignment_workers(16, 4, deadline, |assignment| {
            let now = active.fetch_add(1, Ordering::AcqRel) + 1;
            peak.fetch_max(now, Ordering::AcqRel);
            thread_names.lock().expect("thread-name capture").push(
                std::thread::current()
                    .name()
                    .expect("explicit selector worker name")
                    .to_owned(),
            );
            // Deliberately scramble completion order.
            std::thread::sleep(Duration::from_millis(
                u64::try_from(4 - assignment % 4).expect("small sleep"),
            ));
            active.fetch_sub(1, Ordering::AcqRel);
            Ok(Some(assignment))
        })
        .expect("worker callback cannot error")
        .expect("all canonical jobs complete");
        assert_eq!(values, (0..16).collect::<Vec<_>>());
        assert!(peak.load(Ordering::Acquire) > 1);
        assert!(peak.load(Ordering::Acquire) <= 4);
        assert_eq!(SOLVE_THREAD_STACK_BYTES, 64 * 1024 * 1024);
        assert!(thread_names
            .into_inner()
            .expect("thread-name capture is not poisoned")
            .iter()
            .all(|name| name.starts_with("ny-mip-ay-selector-leaf-")));
    }

    #[test]
    fn bounded_assignment_pool_fails_closed_on_missing_panic_error_or_lateness() {
        let deadline = Instant::now() + Duration::from_secs(1);
        assert!(
            run_bounded_canonical_assignment_workers(16, 4, deadline, |assignment| {
                Ok((assignment != 7).then_some(assignment))
            })
            .expect("a missing leaf is not a checker disagreement")
            .is_none()
        );

        let deadline = Instant::now() + Duration::from_secs(1);
        assert!(
            run_bounded_canonical_assignment_workers(16, 4, deadline, |assignment| {
                assert_ne!(assignment, 3, "injected worker panic");
                Ok(Some(assignment))
            })
            .expect("a worker panic is a fail-closed decline")
            .is_none()
        );

        let deadline = Instant::now() + Duration::from_millis(5);
        assert!(
            run_bounded_canonical_assignment_workers(1, 1, deadline, |assignment| {
                std::thread::sleep(Duration::from_millis(10));
                Ok(Some(assignment))
            })
            .expect("a late worker is a fail-closed decline")
            .is_none()
        );

        let deadline = Instant::now() + Duration::from_secs(1);
        assert!(matches!(
            run_bounded_canonical_assignment_workers::<usize, _>(
                1,
                1,
                deadline,
                |_| Err(MipError::Solver("injected leaf error".to_owned())),
            ),
            Err(MipError::Solver(message)) if message == "injected leaf error"
        ));
    }

    #[test]
    fn concurrent_decline_never_masks_canonical_leaf_error() {
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
        let deadline = Instant::now() + Duration::from_secs(1);
        let result =
            run_bounded_canonical_assignment_workers::<usize, _>(3, 3, deadline, |assignment| {
                barrier.wait();
                match assignment {
                    0 => Err(MipError::Solver("canonical zero error".to_owned())),
                    1 => Err(MipError::Solver("later one error".to_owned())),
                    _ => Ok(None),
                }
            });
        assert!(matches!(
            result,
            Err(MipError::Solver(message)) if message == "canonical zero error"
        ));
    }

    #[test]
    fn parallel_selector_outer_wait_charges_all_preparation_time() {
        let started = Instant::now();
        let deadline = started + Duration::from_millis(100);
        assert_eq!(
            parallel_selector_remaining_outer_wait(deadline, started + Duration::from_millis(40)),
            Some(Duration::from_millis(60))
        );
        assert_eq!(
            parallel_selector_remaining_outer_wait(
                deadline,
                deadline
                    .checked_sub(Duration::from_micros(999))
                    .expect("deadline has a sub-millisecond predecessor"),
            ),
            None,
            "sub-millisecond remainder must not be rounded up by the hard-deadline clamp"
        );
        assert_eq!(
            parallel_selector_remaining_outer_wait(deadline, deadline + Duration::from_millis(1)),
            None
        );
    }

    #[test]
    fn adaptive_three_leaf_candidates_preserve_strict_caller_indexing() {
        let mut problem = MilpProblem::new();
        let first = problem.add_integer_col(0.0, 0.0, 1.0);
        let continuous = problem.add_col(0.0, 0.0, 1.0);
        let second = problem.add_integer_col(0.0, 0.0, 1.0);

        assert_eq!(
            adaptive_three_leaf_candidates(&problem, &[second, first], 1)
                .expect("ordered binary shortlist"),
            vec![second, first]
        );
        for malformed in [
            adaptive_three_leaf_candidates(&problem, &[second, first], 2),
            adaptive_three_leaf_candidates(&problem, &[first, continuous], 0),
            adaptive_three_leaf_candidates(&problem, &[first, first], 0),
            adaptive_three_leaf_candidates(&problem, &[first], 0),
        ] {
            assert!(
                matches!(malformed, Err(MipError::Encoding(_))),
                "malformed indexed advice must fail instead of retargeting the root"
            );
        }
    }

    #[test]
    fn adaptive_four_leaf_comb_candidates_preserve_strict_caller_indexing() {
        let mut problem = MilpProblem::new();
        let first = problem.add_integer_col(0.0, 0.0, 1.0);
        let continuous = problem.add_col(0.0, 0.0, 1.0);
        let second = problem.add_integer_col(0.0, 0.0, 1.0);
        let fixed = problem.add_integer_col(0.0, 1.0, 1.0);
        let third = problem.add_integer_col(0.0, 0.0, 1.0);
        let mut oversized = vec![first, second, third];
        for _ in 0..6 {
            oversized.push(problem.add_integer_col(0.0, 0.0, 1.0));
        }

        assert_eq!(
            adaptive_four_leaf_comb_candidates(&problem, &[second, first, third], 1)
                .expect("ordered three-candidate comb shortlist"),
            vec![second, first, third]
        );
        assert_eq!(
            adaptive_four_leaf_comb_candidates(&problem, &oversized[..8], 7)
                .expect("the fixed eight-candidate ceiling remains admissible"),
            oversized[..8]
        );
        assert!(
            matches!(
                adaptive_four_leaf_comb_candidates(&problem, &oversized, 0),
                Err(MipError::Encoding(_))
            ),
            "a nine-candidate comb shortlist must fail instead of exceeding fixed probe ceilings"
        );
        for malformed in [
            adaptive_four_leaf_comb_candidates(&problem, &[second, first, third], 3),
            adaptive_four_leaf_comb_candidates(&problem, &[first, continuous, third], 0),
            adaptive_four_leaf_comb_candidates(&problem, &[first, fixed, third], 0),
            adaptive_four_leaf_comb_candidates(&problem, &[first, first, third], 0),
            adaptive_four_leaf_comb_candidates(&problem, &[first, second], 0),
            adaptive_four_leaf_comb_candidates(
                &problem,
                &[first, second, Col(problem.num_cols() + 1)],
                0,
            ),
        ] {
            assert!(
                matches!(malformed, Err(MipError::Encoding(_))),
                "malformed comb candidates must fail instead of retargeting the root"
            );
        }
    }

    #[test]
    fn adaptive_five_leaf_comb_candidates_preserve_strict_caller_indexing() {
        let mut problem = MilpProblem::new();
        let first = problem.add_integer_col(0.0, 0.0, 1.0);
        let continuous = problem.add_col(0.0, 0.0, 1.0);
        let second = problem.add_integer_col(0.0, 0.0, 1.0);
        let fixed = problem.add_integer_col(0.0, 1.0, 1.0);
        let third = problem.add_integer_col(0.0, 0.0, 1.0);
        let fourth = problem.add_integer_col(0.0, 0.0, 1.0);
        let mut oversized = vec![first, second, third, fourth];
        for _ in 0..5 {
            oversized.push(problem.add_integer_col(0.0, 0.0, 1.0));
        }

        assert_eq!(
            adaptive_five_leaf_comb_candidates(&problem, &[second, fourth, first, third], 2,)
                .expect("ordered four-candidate five-leaf shortlist"),
            vec![second, fourth, first, third]
        );
        assert_eq!(
            adaptive_five_leaf_comb_candidates(&problem, &oversized[..8], 7)
                .expect("the fixed eight-candidate ceiling remains admissible"),
            oversized[..8]
        );
        assert!(matches!(
            adaptive_five_leaf_comb_candidates(&problem, &oversized, 0),
            Err(MipError::Encoding(_))
        ));
        for malformed in [
            adaptive_five_leaf_comb_candidates(&problem, &[first, second, third, fourth], 4),
            adaptive_five_leaf_comb_candidates(&problem, &[first, continuous, third, fourth], 0),
            adaptive_five_leaf_comb_candidates(&problem, &[first, fixed, third, fourth], 0),
            adaptive_five_leaf_comb_candidates(&problem, &[first, second, first, fourth], 0),
            adaptive_five_leaf_comb_candidates(&problem, &[first, second, third], 0),
            adaptive_five_leaf_comb_candidates(
                &problem,
                &[first, second, third, Col(problem.num_cols() + 1)],
                0,
            ),
        ] {
            assert!(
                matches!(malformed, Err(MipError::Encoding(_))),
                "malformed five-leaf candidates must fail instead of retargeting the root"
            );
        }
    }

    #[test]
    fn adaptive_four_leaf_comb_topology_validator_rejects_mutations() {
        let mut problem = MilpProblem::new();
        let root = problem.add_integer_col(0.0, 0.0, 1.0);
        let second = problem.add_integer_col(0.0, 0.0, 1.0);
        let third = problem.add_integer_col(0.0, 0.0, 1.0);
        let wrong = problem.add_integer_col(0.0, 0.0, 1.0);
        let model = to_ay_model(&problem).expect("binary topology model lowers to AY");
        let [root, second, third, wrong] =
            [root, second, third, wrong].map(|col| model.col_at(col.0).expect("lowered AY column"));

        for root_hard_value in [false, true] {
            for second_hard_value in [false, true] {
                let valid =
                    four_leaf_comb_tree(root, root_hard_value, second, second_hard_value, third);
                validate_adaptive_four_leaf_comb_certificate_shape(
                    &valid,
                    root,
                    root_hard_value,
                    second,
                    second_hard_value,
                    third,
                )
                .expect("all four comb orientations have the required topology");
            }
        }

        let wrong_root = four_leaf_comb_tree(wrong, true, second, false, third);
        let wrong_second = four_leaf_comb_tree(root, true, wrong, true, third);
        let mut nonzero_root_cut = four_leaf_comb_tree(root, true, second, true, third);
        let TreeNode::Split { cut, .. } = &mut nonzero_root_cut else {
            unreachable!("comb helper always returns a split")
        };
        *cut = BigRational::from_integer(1.into());
        let split_easy_root = binary_tree_node(
            root,
            binary_tree_node(wrong, empty_farkas_leaf(), empty_farkas_leaf()),
            four_leaf_comb_tree(second, false, wrong, false, third),
        );
        let missing_third = binary_tree_node(
            root,
            empty_farkas_leaf(),
            binary_tree_node(second, empty_farkas_leaf(), empty_farkas_leaf()),
        );
        let wrong_third = four_leaf_comb_tree(root, true, second, true, wrong);
        let split_terminal = binary_tree_node(
            root,
            empty_farkas_leaf(),
            binary_tree_node(
                second,
                empty_farkas_leaf(),
                binary_tree_node(
                    third,
                    binary_tree_node(wrong, empty_farkas_leaf(), empty_farkas_leaf()),
                    empty_farkas_leaf(),
                ),
            ),
        );
        for malformed in [
            wrong_root,
            wrong_second,
            nonzero_root_cut,
            split_easy_root,
            missing_third,
            wrong_third,
            split_terminal,
        ] {
            assert!(
                matches!(
                    validate_adaptive_four_leaf_comb_certificate_shape(
                        &malformed, root, true, second, true, third,
                    ),
                    Err(MipError::Solver(_))
                ),
                "mutated comb topology must fail before whole-tree authority"
            );
        }
    }

    #[test]
    fn adaptive_five_leaf_comb_topology_validator_rejects_mutations() {
        let mut problem = MilpProblem::new();
        let root = problem.add_integer_col(0.0, 0.0, 1.0);
        let second = problem.add_integer_col(0.0, 0.0, 1.0);
        let third = problem.add_integer_col(0.0, 0.0, 1.0);
        let fourth = problem.add_integer_col(0.0, 0.0, 1.0);
        let wrong = problem.add_integer_col(0.0, 0.0, 1.0);
        let model = to_ay_model(&problem).expect("binary topology model lowers to AY");
        let [root, second, third, fourth, wrong] = [root, second, third, fourth, wrong]
            .map(|col| model.col_at(col.0).expect("lowered AY column"));

        for root_hard_value in [false, true] {
            for second_hard_value in [false, true] {
                for third_hard_value in [false, true] {
                    let valid = five_leaf_comb_tree(
                        root,
                        root_hard_value,
                        second,
                        second_hard_value,
                        third,
                        third_hard_value,
                        fourth,
                    );
                    validate_adaptive_five_leaf_comb_certificate_shape(
                        &valid,
                        root,
                        root_hard_value,
                        second,
                        second_hard_value,
                        third,
                        third_hard_value,
                        fourth,
                    )
                    .expect("all eight five-leaf comb orientations have the required topology");
                }
            }
        }

        let wrong_root = five_leaf_comb_tree(wrong, true, second, true, third, true, fourth);
        let wrong_second = five_leaf_comb_tree(root, true, wrong, true, third, true, fourth);
        let wrong_third = five_leaf_comb_tree(root, true, second, true, wrong, true, fourth);
        let wrong_fourth = five_leaf_comb_tree(root, true, second, true, third, true, wrong);
        let mut nonzero_root_cut =
            five_leaf_comb_tree(root, true, second, true, third, true, fourth);
        let TreeNode::Split { cut, .. } = &mut nonzero_root_cut else {
            unreachable!("five-leaf helper always returns a split")
        };
        *cut = BigRational::from_integer(1.into());
        let split_easy_root = binary_tree_node(
            root,
            binary_tree_node(wrong, empty_farkas_leaf(), empty_farkas_leaf()),
            binary_tree_node(
                second,
                empty_farkas_leaf(),
                binary_tree_node(
                    third,
                    empty_farkas_leaf(),
                    binary_tree_node(fourth, empty_farkas_leaf(), empty_farkas_leaf()),
                ),
            ),
        );
        let missing_fourth = binary_tree_node(
            root,
            empty_farkas_leaf(),
            binary_tree_node(
                second,
                empty_farkas_leaf(),
                binary_tree_node(third, empty_farkas_leaf(), empty_farkas_leaf()),
            ),
        );
        let split_terminal = binary_tree_node(
            root,
            empty_farkas_leaf(),
            binary_tree_node(
                second,
                empty_farkas_leaf(),
                binary_tree_node(
                    third,
                    empty_farkas_leaf(),
                    binary_tree_node(
                        fourth,
                        binary_tree_node(wrong, empty_farkas_leaf(), empty_farkas_leaf()),
                        empty_farkas_leaf(),
                    ),
                ),
            ),
        );
        for malformed in [
            wrong_root,
            wrong_second,
            wrong_third,
            wrong_fourth,
            nonzero_root_cut,
            split_easy_root,
            missing_fourth,
            split_terminal,
        ] {
            assert!(
                matches!(
                    validate_adaptive_five_leaf_comb_certificate_shape(
                        &malformed, root, true, second, true, third, true, fourth,
                    ),
                    Err(MipError::Solver(_))
                ),
                "mutated five-leaf topology must fail before whole-tree authority"
            );
        }
    }

    #[test]
    fn decision_dump_is_default_off() {
        let _env_lock = lock_env();
        let _dump = ScopedEnvVar::unset("NY_MIP_DUMP");
        let mut problem = MilpProblem::new();
        let x = problem.add_col(0.0, -1.0, 1.0);

        assert!(
            maybe_dump_linear_lower_decision(&problem, &[(x, 1.0)], 0.25).is_none(),
            "an unset diagnostic must not clone or dump a decision model"
        );
        assert_eq!(
            problem.num_rows(),
            0,
            "default-off capture must not mutate the caller's model"
        );
    }

    #[test]
    fn decision_dump_captures_exact_threshold_row_before_relaxation() {
        struct TestDumpDir(std::path::PathBuf);

        impl Drop for TestDumpDir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        let _env_lock = lock_env();
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock must follow the Unix epoch")
            .as_nanos();
        let dump_dir = TestDumpDir(std::env::temp_dir().join(format!(
            "ny-mip-linear-lower-dump-{}-{unique}",
            std::process::id()
        )));
        std::fs::create_dir(&dump_dir.0).expect("temporary dump directory");
        let dump_path = dump_dir.0.to_str().expect("temporary path must be UTF-8");
        let _dump = ScopedEnvVar::set("NY_MIP_DUMP", dump_path);
        let mut problem = MilpProblem::new();
        let x = problem.add_col(0.0, -1.0, 1.0);
        let y = problem.add_integer_col(0.0, 0.0, 1.0);
        problem.add_row(-0.5, f64::INFINITY, [(x, 1.0)]);
        let requested_lower = -0.679_319_44_f32;
        let objective = [(x, 1.25), (y, -0.5)];

        let artifact = maybe_dump_linear_lower_decision(&problem, &objective, requested_lower)
            .expect("armed capture must emit the decision model");
        assert_eq!(
            artifact.extension().and_then(|ext| ext.to_str()),
            Some("milp")
        );
        assert_eq!(artifact.parent(), Some(dump_dir.0.as_path()));
        let text = std::fs::read_to_string(&artifact).expect("dump must be readable");
        let captured = crate::dump::from_milp_text(&text).expect("dump must round-trip");
        let row = captured
            .rows()
            .last()
            .expect("decision row must be appended");
        assert_eq!(row.lb.to_bits(), f64::NEG_INFINITY.to_bits());
        assert_eq!(
            row.ub.to_bits(),
            f64::from(requested_lower).to_bits(),
            "the binary32 threshold must widen to the exact binary64 value"
        );
        assert_eq!(
            row.coeffs
                .iter()
                .map(|&(col, coeff)| (col, coeff.to_bits()))
                .collect::<Vec<_>>(),
            objective
                .iter()
                .map(|&(col, coeff)| (col.0, coeff.to_bits()))
                .collect::<Vec<_>>(),
            "the arbitrary linear objective must survive bit-exactly as the decision row"
        );
        assert!(
            captured.cols()[y.0].integer,
            "capture occurs before the relaxation mutates integrality"
        );
    }

    #[test]
    fn exact_worker_lease_sheds_overlap_and_reopens_after_exit() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let lease =
            CertifiedLinearLowerWorkerAdmission::try_acquire().expect("first worker admitted");
        assert!(
            CertifiedLinearLowerWorkerAdmission::try_acquire().is_none(),
            "an overlapping exact worker must be shed"
        );
        drop(lease);
        let reopened = CertifiedLinearLowerWorkerAdmission::try_acquire()
            .expect("admission reopens after exit");
        drop(reopened);
    }

    #[test]
    fn detached_worker_retains_admission_until_actual_exit() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let admission =
            CertifiedLinearLowerWorkerAdmission::try_acquire().expect("worker admitted");
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel::<()>(0);
        let result: Result<Option<()>, MipError> =
            run_with_hard_deadline(0.01, "linear-lower-lease-test", move || {
                let _admission = admission;
                let _ = release_rx.recv();
                Ok(())
            });
        assert!(
            matches!(result, Ok(None)),
            "the caller must detach at its hard deadline"
        );
        assert!(
            CertifiedLinearLowerWorkerAdmission::try_acquire().is_none(),
            "the detached worker must retain admission"
        );
        release_tx.send(()).expect("release detached worker");

        let wait_until = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(reopened) = CertifiedLinearLowerWorkerAdmission::try_acquire() {
                drop(reopened);
                break;
            }
            assert!(
                Instant::now() < wait_until,
                "admission did not reopen after the detached worker exited"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn detached_parallel_coordinator_retains_admission_until_inner_pool_exits() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let admission =
            CertifiedLinearLowerWorkerAdmission::try_acquire().expect("worker admitted");
        let release = std::sync::Arc::new(AtomicBool::new(false));
        let worker_release = std::sync::Arc::clone(&release);
        let result: Result<Option<Vec<usize>>, MipError> =
            run_with_hard_deadline(0.01, "parallel-selector-lease-test", move || {
                let _admission = admission;
                run_bounded_canonical_assignment_workers(
                    16,
                    4,
                    Instant::now() + Duration::from_secs(2),
                    |_| {
                        while !worker_release.load(Ordering::Acquire) {
                            std::thread::yield_now();
                        }
                        Ok(Some(1usize))
                    },
                )
            })
            .map(Option::flatten);
        assert!(
            matches!(result, Ok(None)),
            "the outer caller must detach while inner workers remain live"
        );
        assert!(
            CertifiedLinearLowerWorkerAdmission::try_acquire().is_none(),
            "inner worker lifetimes must keep the process-wide admission closed"
        );
        release.store(true, Ordering::Release);

        let wait_until = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(reopened) = CertifiedLinearLowerWorkerAdmission::try_acquire() {
                drop(reopened);
                break;
            }
            assert!(
                Instant::now() < wait_until,
                "admission did not reopen after every inner worker exited"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn continuous_lower_bound_requires_exact_linear_replay() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let mut problem = MilpProblem::new();
        let x = problem.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
        problem.add_row(1.0, f64::INFINITY, [(x, 1.0)]);

        let certified = certify_linear_lower_bound_with_ay(&problem, &[(x, 1.0)], config())
            .expect("solver/checkers agree")
            .expect("root proof");
        assert!(certified.lower < 1.0);
        assert!(certified.lower > 0.99);
        assert_eq!(
            certified.proof_route,
            CertifiedLinearLowerProofRoute::RelaxationEntailment
        );
        assert_eq!(certified.ay_tree_leaves, 0);
        assert_eq!(certified.ny_cert_farkas_replays, 1);
    }

    #[test]
    fn relaxation_entailment_requires_strict_exact_separation() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let mut problem = MilpProblem::new();
        let x = problem.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
        problem.add_row(1.0, f64::INFINITY, [(x, 1.0)]);
        let objective = canonical_objective(&problem, &[(x, 1.0)]).unwrap();
        let opts = solve_opts(10.0).with_require_certificates(true);

        let replay = try_relaxed_linear_lower_proof(&problem, &objective, 0.99, &[], 64, &opts)
            .expect("AY and ny-cert agree")
            .expect("the relaxed optimum strictly exceeds 0.99");
        assert_eq!(
            replay.proof_route,
            CertifiedLinearLowerProofRoute::RelaxationEntailment
        );
        assert_eq!(replay.tree_leaves, 0);
        assert_eq!(replay.linear_replays, 1);

        assert!(
            try_relaxed_linear_lower_proof(&problem, &objective, 1.0, &[], 64, &opts)
                .expect("equality is an ordinary non-proof")
                .is_none(),
            "an attained optimum must not prove the non-strict decision row infeasible"
        );
    }

    #[test]
    fn advised_relaxation_builds_and_replays_exact_two_leaf_tree() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let (problem, x, z) = one_split_integrality_gap_problem();
        let objective = canonical_objective(&problem, &[(z, 1.0)]).unwrap();
        let opts = solve_opts(10.0).with_require_certificates(true);

        let replay = try_relaxed_linear_lower_proof(&problem, &objective, 0.75, &[x], 2, &opts)
            .expect("AY and ny-cert agree")
            .expect("both binary children strictly refute z<=3/4");
        assert_eq!(
            replay.proof_route,
            CertifiedLinearLowerProofRoute::TreeFarkas
        );
        assert_eq!(replay.tree_leaves, 2);
        assert_eq!(replay.linear_replays, 2);

        assert!(
            try_relaxed_linear_lower_proof(&problem, &objective, 0.75, &[x], 1, &opts)
                .expect("one-leaf admission declines without checker disagreement")
                .is_none(),
            "a two-leaf proof must not cross a one-leaf admission"
        );
    }

    #[test]
    fn advised_relaxation_builds_only_the_admitted_exact_four_leaf_tree() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let (problem, x, y, z) = two_split_integrality_gap_problem();
        let objective = canonical_objective(&problem, &[(z, 1.0)]).unwrap();
        let opts = solve_opts(10.0).with_require_certificates(true);

        assert!(
            try_relaxed_linear_lower_proof(&problem, &objective, 0.25, &[], 4, &opts)
                .expect("ordinary relaxation solve/checkers agree")
                .is_none(),
            "the historical no-advice lane must not introduce case splits"
        );
        assert!(
            try_relaxed_linear_lower_proof(&problem, &objective, 0.25, &[x], 4, &opts)
                .expect("one-split solve/checkers agree")
                .is_none(),
            "one split leaves x=0, y=1/2 feasible at z=0"
        );
        assert!(
            try_relaxed_linear_lower_proof(&problem, &objective, 0.25, &[x, y], 3, &opts)
                .expect("an underpriced tree is an ordinary decline")
                .is_none(),
            "a four-leaf proof must not cross a three-leaf admission"
        );

        let replay = try_relaxed_linear_lower_proof(&problem, &objective, 0.25, &[x, y], 4, &opts)
            .expect("AY and ny-cert agree")
            .expect("all four binary assignments strictly refute z<=1/4");
        assert_eq!(
            replay.proof_route,
            CertifiedLinearLowerProofRoute::TreeFarkas
        );
        assert_eq!(replay.tree_leaves, 4);
        assert_eq!(replay.linear_replays, 4);
    }

    #[test]
    fn fixed_assignment_tree_admission_modes_replay_identical_sixteen_leaf_proofs() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let (problem, splits, z) = four_split_complete_assignment_problem();
        let config = CertifiedLinearLowerDecisionConfig {
            proof_timeout_secs: 10.0,
            max_tree_leaves: 16,
        };

        let certified = certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_admission(
            &problem,
            &[(z, 1.0)],
            0.75,
            &[splits[3], splits[1], splits[0], splits[2]],
            config,
            CertifiedLinearLowerWorkerAdmission::try_acquire()
                .expect("caller owns the exact-worker admission"),
        )
        .expect("AY and ny-cert agree")
        .expect("all sixteen assignments strictly refute z<=3/4");
        assert_eq!(
            certified.proof_route,
            CertifiedLinearLowerProofRoute::TreeFarkas
        );
        assert_eq!(certified.ay_tree_leaves, 16);
        assert_eq!(certified.ny_cert_farkas_replays, 16);

        let range_logical =
            certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_range_logical_admission(
                &problem,
                &[(z, 1.0)],
                0.75,
                &[splits[3], splits[1], splits[0], splits[2]],
                config,
                CertifiedLinearLowerWorkerAdmission::try_acquire()
                    .expect("caller owns the range-logical exact-worker admission"),
            )
            .expect("range-logical AY and ny-cert agree")
            .expect("range-logical advice preserves the complete exact proof");
        assert_eq!(range_logical, certified);

        let selector_profile =
            certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_selector_solve_profile_admission(
                &problem,
                &[(z, 1.0)],
                0.75,
                &[splits[3], splits[1], splits[0], splits[2]],
                config,
                CertifiedLinearLowerWorkerAdmission::try_acquire()
                    .expect("caller owns the selector-profile exact-worker admission"),
            )
            .expect("selector-profile AY and ny-cert agree")
            .expect("selector-profile advice preserves the complete exact proof");
        assert_eq!(selector_profile, certified);

        let compact_progressive =
            certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_compact_progressive_admission(
                &problem,
                &[(z, 1.0)],
                0.75,
                &[splits[3], splits[1], splits[0], splits[2]],
                config,
                CertifiedLinearLowerWorkerAdmission::try_acquire()
                    .expect("caller owns the compact-progressive exact-worker admission"),
            )
            .expect("compact-progressive AY and ny-cert agree")
            .expect("advice-only root/prefix probes preserve the complete exact proof");
        assert_eq!(compact_progressive, certified);

        let reopened = CertifiedLinearLowerWorkerAdmission::try_acquire()
            .expect("completed fixed-tree proof releases caller admission");
        drop(reopened);
    }

    #[test]
    fn absolute_fixed_tree_charges_full_clone_and_skips_worker_after_setup_expiry() {
        let _env_lock = lock_env();
        let _dump = ScopedEnvVar::unset("NY_MIP_DUMP");
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let (problem, splits, z) = four_split_complete_assignment_problem();
        let objective = canonical_objective(&problem, &[(z, 1.0)]).expect("canonical objective");
        let started = Instant::now();
        let deadline = started + Duration::from_secs(30);
        let clone_observed = std::sync::Arc::new(AtomicBool::new(false));
        let worker_spawn_attempted = std::sync::Arc::new(AtomicBool::new(false));
        let clock_clone_observed = std::sync::Arc::clone(&clone_observed);
        let hook_clone_observed = std::sync::Arc::clone(&clone_observed);
        let hook_worker_spawn_attempted = std::sync::Arc::clone(&worker_spawn_attempted);

        let result = solve_and_replay_fixed_assignment_tree_proof_until_with_setup_control(
            &problem,
            objective,
            0.75,
            splits.to_vec(),
            deadline,
            16,
            LpSolvePolicy::Default,
            CertifiedLinearLowerWorkerAdmission::try_acquire()
                .expect("absolute fixed-tree worker admitted"),
            move || {
                if clock_clone_observed.load(Ordering::Acquire) {
                    deadline
                } else {
                    started
                }
            },
            move || hook_clone_observed.store(true, Ordering::Release),
            move || hook_worker_spawn_attempted.store(true, Ordering::Release),
        )
        .expect("setup expiry is an ordinary decline");

        assert!(result.is_none(), "expired setup must not produce authority");
        assert!(
            clone_observed.load(Ordering::Acquire),
            "the injected deadline advanced only after the full problem clone"
        );
        assert!(
            !worker_spawn_attempted.load(Ordering::Acquire),
            "post-clone expiry must be observed before worker construction"
        );
        let reopened = CertifiedLinearLowerWorkerAdmission::try_acquire()
            .expect("setup-only expiry leaves no surviving exact worker");
        drop(reopened);
    }

    #[test]
    fn absolute_fixed_tree_declines_expired_or_nearly_expired_without_worker() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let (problem, splits, z) = four_split_complete_assignment_problem();
        for proof_deadline in [Instant::now(), Instant::now() + Duration::from_micros(500)] {
            let result = certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_until_unwired(
                &problem,
                &[(z, 1.0)],
                0.75,
                &splits,
                proof_deadline,
                16,
            )
            .expect("absolute expiry is an ordinary decline");
            assert!(
                result.is_none(),
                "expired/sub-millisecond request must not produce authority"
            );
            let reopened = CertifiedLinearLowerWorkerAdmission::try_acquire()
                .expect("early absolute decline must not leave a worker alive");
            drop(reopened);
        }
    }

    #[test]
    fn fixed_assignment_tree_admission_rejects_malformed_splits_and_releases_lease() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let (problem, splits, z) = four_split_complete_assignment_problem();
        let config = CertifiedLinearLowerDecisionConfig {
            proof_timeout_secs: 10.0,
            max_tree_leaves: 16,
        };

        let result = certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_admission(
            &problem,
            &[(z, 1.0)],
            0.75,
            &[splits[0], splits[0]],
            config,
            CertifiedLinearLowerWorkerAdmission::try_acquire()
                .expect("caller owns the exact-worker admission"),
        );
        assert!(
            matches!(result, Err(MipError::Encoding(_))),
            "duplicate proof-critical splits must fail closed"
        );

        let reopened = CertifiedLinearLowerWorkerAdmission::try_acquire()
            .expect("validation failure releases caller admission");
        drop(reopened);
    }

    #[test]
    fn fixed_assignment_tree_admission_fails_closed_when_one_leaf_is_weak() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let (problem, splits, z) = four_split_weak_leaf_problem();
        let config = CertifiedLinearLowerDecisionConfig {
            proof_timeout_secs: 10.0,
            max_tree_leaves: 16,
        };

        assert!(
            certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_admission(
                &problem,
                &[(z, 1.0)],
                0.75,
                &splits,
                config,
                CertifiedLinearLowerWorkerAdmission::try_acquire()
                    .expect("caller owns the exact-worker admission"),
            )
            .expect("a weak leaf is an ordinary non-proof")
            .is_none(),
            "one weak assignment must reject the entire tree"
        );

        let reopened = CertifiedLinearLowerWorkerAdmission::try_acquire()
            .expect("ordinary fixed-tree decline releases caller admission");
        drop(reopened);
    }

    #[test]
    fn compact_progressive_tree_fails_closed_on_sat_leaf_and_timeout() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let (problem, splits, z) = four_split_weak_leaf_problem();
        let result =
            certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_compact_progressive_admission(
                &problem,
                &[(z, 1.0)],
                0.75,
                &splits,
                CertifiedLinearLowerDecisionConfig {
                    proof_timeout_secs: 10.0,
                    max_tree_leaves: 16,
                },
                CertifiedLinearLowerWorkerAdmission::try_acquire()
                    .expect("caller owns the compact-progressive admission"),
            )
            .expect("a feasible weak leaf is an ordinary non-proof");
        assert!(
            result.is_none(),
            "one SAT/weak assignment must reject the complete-tree authority"
        );

        let (problem, splits, z) = four_split_complete_assignment_problem();
        let timed_out =
            certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_compact_progressive_admission(
                &problem,
                &[(z, 1.0)],
                0.75,
                &splits,
                CertifiedLinearLowerDecisionConfig {
                    proof_timeout_secs: 1.0e-9,
                    max_tree_leaves: 16,
                },
                CertifiedLinearLowerWorkerAdmission::try_acquire()
                    .expect("completed weak-leaf attempt releases admission"),
            )
            .expect("deadline expiry is an ordinary non-proof");
        assert!(
            timed_out.is_none(),
            "timeout/Unknown must never become fixed-tree authority"
        );
    }

    #[test]
    fn parallel_selector_tree_worker_counts_produce_identical_verified_metadata() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let (problem, selectors, z) = four_split_complete_assignment_problem();
        let config = CertifiedLinearLowerDecisionConfig {
            proof_timeout_secs: 10.0,
            max_tree_leaves: 16,
        };
        let mut baseline = None;
        for max_workers in [1, 2, 4, 8, 16] {
            let certified = certify_linear_lower_bound_at_with_ay_parallel_selector_tree_unwired(
                &problem,
                &[(z, 1.0)],
                0.75,
                &[selectors[3], selectors[1], selectors[0], selectors[2]],
                max_workers,
                config,
            )
            .expect("AY and ny-cert agree")
            .expect("all sixteen canonical assignments certify");
            assert_eq!(
                certified.proof_route,
                CertifiedLinearLowerProofRoute::TreeFarkas
            );
            assert_eq!(certified.ay_tree_leaves, 16);
            assert_eq!(certified.ny_cert_farkas_replays, 16);
            match baseline {
                Some(expected) => assert_eq!(certified, expected),
                None => baseline = Some(certified),
            }
        }
    }

    #[test]
    fn parallel_selector_tree_composes_mixed_rows_and_farkas_but_rejects_misassociation() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let (problem, selectors, z) = four_selector_mixed_evidence_problem();
        let config = CertifiedLinearLowerDecisionConfig {
            proof_timeout_secs: 10.0,
            max_tree_leaves: 16,
        };
        let certified = certify_linear_lower_bound_at_with_ay_parallel_selector_tree_unwired(
            &problem,
            &[(z, 1.0)],
            0.75,
            &selectors,
            8,
            config,
        )
        .expect("mixed AY evidence and ny-cert agree")
        .expect("all mixed leaves compose");
        assert_eq!(
            certified,
            CertifiedLinearLowerBound {
                lower: 0.75,
                proof_route: CertifiedLinearLowerProofRoute::TreeFarkas,
                ay_tree_leaves: 16,
                ny_cert_farkas_replays: 16,
            }
        );

        let objective = canonical_objective(&problem, &[(z, 1.0)]).unwrap();
        let mut relaxed_problem = problem.clone();
        relaxed_problem.relax_integrality();
        let relaxed_model = to_ay_model(&relaxed_problem).unwrap();
        let ay_objective = vec![(relaxed_model.col_at(z.0).unwrap(), 1.0)];
        let ay_selectors = selectors
            .iter()
            .map(|selector| relaxed_model.col_at(selector.0).unwrap())
            .collect::<Vec<_>>();
        let requested = exact_f64(0.75, "test requested threshold").unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let opts = solve_opts(10.0)
            .with_deadline(deadline)
            .with_tree_cert_leaves(0)
            .with_require_certificates(true);
        let mut leaves = run_bounded_canonical_assignment_workers(
            PARALLEL_SELECTOR_TREE_LEAVES,
            8,
            deadline,
            |assignment| {
                solve_parallel_selector_leaf(
                    &relaxed_model,
                    &ay_objective,
                    &ay_selectors,
                    assignment,
                    &requested,
                    deadline,
                    &opts,
                )
            },
        )
        .expect("leaf workers do not error")
        .expect("all mixed leaves exist");
        assert_eq!(
            leaves
                .iter()
                .filter(|leaf| matches!(
                    leaf.evidence,
                    ParallelSelectorLeafEvidence::ConditionalRow(_)
                ))
                .count(),
            8
        );
        assert_eq!(
            leaves
                .iter()
                .filter(|leaf| matches!(leaf.evidence, ParallelSelectorLeafEvidence::Infeasible(_)))
                .count(),
            8
        );

        leaves[0].assignment = 1;
        assert!(matches!(
            compose_and_replay_parallel_selector_tree(
                &problem,
                &objective,
                0.75,
                &selectors,
                &ay_selectors,
                leaves,
                16,
            ),
            Err(MipError::Solver(message)) if message.contains("claimed assignment")
        ));
    }

    #[test]
    fn parallel_selector_tree_fails_closed_on_one_weak_leaf_and_releases_admission() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let (problem, selectors, z) = four_split_weak_leaf_problem();
        let config = CertifiedLinearLowerDecisionConfig {
            proof_timeout_secs: 10.0,
            max_tree_leaves: 16,
        };
        assert!(
            certify_linear_lower_bound_at_with_ay_parallel_selector_tree_admission(
                &problem,
                &[(z, 1.0)],
                0.75,
                &selectors,
                4,
                config,
                CertifiedLinearLowerWorkerAdmission::try_acquire()
                    .expect("caller owns the exact-worker admission"),
            )
            .expect("a weak leaf is an ordinary non-proof")
            .is_none(),
            "one weak assignment must reject the whole parallel tree"
        );
        let reopened = CertifiedLinearLowerWorkerAdmission::try_acquire()
            .expect("scoped workers release admission only after joining");
        drop(reopened);
    }

    #[test]
    fn target_fsb_selects_verified_nonprefix_four_leaf_tree() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let (problem, x, dummy, z, p, q) = target_fsb_nonprefix_pair_problem();
        let objective = canonical_objective(&problem, &[(p, 1.0), (q, 1.0)]).unwrap();
        let opts = solve_opts(10.0).with_require_certificates(true);
        let requested = 7.0_f32 / 4.0;

        assert!(
            try_relaxed_linear_lower_proof(&problem, &objective, requested, &[x, dummy], 4, &opts,)
                .expect("fixed-prefix solve/checkers agree")
                .is_none(),
            "the fixed [x,dummy] tree leaves z relaxed and cannot prove 7/4"
        );

        let replay = try_relaxed_linear_lower_proof(
            &problem,
            &objective,
            requested,
            &[x, dummy, z],
            4,
            &opts,
        )
        .expect("target-FSB solve and both exact checkers agree")
        .expect("target-FSB must select the useful non-prefix [x,z] pair");
        assert_eq!(
            replay.proof_route,
            CertifiedLinearLowerProofRoute::TreeFarkas
        );
        assert_eq!(replay.tree_leaves, 4);
        assert_eq!(replay.linear_replays, 4);
    }

    #[test]
    fn diagnostic_target_fsb_probe_limits_keep_exact_authority_layers() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let (problem, x, dummy, z, p, q) = target_fsb_nonprefix_pair_problem();
        let limits = CertifiedLinearLowerTargetFsbProbeLimits::new(7, Duration::from_millis(250))
            .expect("bounded diagnostic limits");
        let proof =
            certify_linear_lower_bound_at_with_ay_branch_advice_with_target_fsb_probe_limits_unwired(
                &problem,
                &[(p, 1.0), (q, 1.0)],
                7.0_f32 / 4.0,
                &[x, dummy, z],
                CertifiedLinearLowerDecisionConfig {
                    proof_timeout_secs: 10.0,
                    max_tree_leaves: 4,
                },
                limits,
            )
            .expect("AY and both exact authority layers agree")
            .expect("diagnostic target-FSB limits retain the non-prefix tree proof");
        assert_eq!(
            proof.proof_route,
            CertifiedLinearLowerProofRoute::TreeFarkas
        );
        assert_eq!(proof.ay_tree_leaves, 4);
        assert_eq!(proof.ny_cert_farkas_replays, 4);
    }

    #[test]
    fn adaptive_three_leaf_target_fsb_replays_both_orientations_and_mixed_leaf_kinds() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        for hard_value in [false, true] {
            let (problem, root, dummy, second, z) = adaptive_three_leaf_mixed_problem(hard_value);
            let proof =
                certify_linear_lower_bound_at_with_ay_adaptive_three_leaf_target_fsb_unwired(
                    &problem,
                    &[(z, 1.0)],
                    0.75,
                    &[dummy, root, second],
                    1,
                    hard_value,
                    CertifiedLinearLowerDecisionConfig {
                        proof_timeout_secs: 10.0,
                        max_tree_leaves: 3,
                    },
                )
                .expect("AY whole-tree verification and NY leaf replay agree")
                .expect("one Farkas sibling and two hard grandchildren close z<=3/4");
            assert_eq!(proof.lower.to_bits(), 0.75_f32.to_bits());
            assert_eq!(
                proof.proof_route,
                CertifiedLinearLowerProofRoute::TreeFarkas
            );
            assert_eq!(proof.ay_tree_leaves, 3);
            assert_eq!(proof.ny_cert_farkas_replays, 3);
        }
    }

    #[test]
    fn adaptive_three_leaf_target_fsb_respects_three_leaf_admission() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let (problem, root, dummy, second, z) = adaptive_three_leaf_mixed_problem(true);
        let proof = certify_linear_lower_bound_at_with_ay_adaptive_three_leaf_target_fsb_unwired(
            &problem,
            &[(z, 1.0)],
            0.75,
            &[root, dummy, second],
            0,
            true,
            CertifiedLinearLowerDecisionConfig {
                proof_timeout_secs: 10.0,
                max_tree_leaves: 2,
            },
        )
        .expect("underpriced diagnostic is an ordinary decline");
        assert!(
            proof.is_none(),
            "an adaptive three-leaf proof must not cross a two-leaf admission"
        );
    }

    #[test]
    fn adaptive_three_leaf_target_fsb_root_fast_path_keeps_zero_probe_authority() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let mut problem = MilpProblem::new();
        let root = problem.add_integer_col(0.0, 0.0, 1.0);
        let other = problem.add_integer_col(0.0, 0.0, 1.0);
        let z = problem.add_col(0.0, 0.0, 2.0);
        problem.add_row(1.0, f64::INFINITY, [(z, 1.0)]);

        let proof = certify_linear_lower_bound_at_with_ay_adaptive_three_leaf_target_fsb_unwired(
            &problem,
            &[(z, 1.0)],
            0.75,
            &[root, other],
            0,
            true,
            CertifiedLinearLowerDecisionConfig {
                proof_timeout_secs: 10.0,
                max_tree_leaves: 3,
            },
        )
        .expect("AY root verification and NY entailment replay agree")
        .expect("root relaxation already closes z<=3/4");
        assert_eq!(
            proof.proof_route,
            CertifiedLinearLowerProofRoute::RelaxationEntailment
        );
        assert_eq!(proof.ay_tree_leaves, 0);
        assert_eq!(proof.ny_cert_farkas_replays, 1);
    }

    #[test]
    fn adaptive_four_leaf_comb_replays_all_orientations_nonprefix_and_mixed_leaf_kinds() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        for root_hard_value in [false, true] {
            for second_hard_value in [false, true] {
                let infeasible_root_easy = root_hard_value == second_hard_value;
                let (problem, root, dummy1, second, dummy2, third, p) =
                    adaptive_four_leaf_comb_problem(
                        root_hard_value,
                        second_hard_value,
                        infeasible_root_easy,
                    );
                let proof =
                    certify_linear_lower_bound_at_with_ay_adaptive_four_leaf_comb_target_fsb_unwired(
                        &problem,
                        &[(p, 1.0)],
                        0.875,
                        &[dummy1, root, dummy2, second, third],
                        1,
                        root_hard_value,
                        CertifiedLinearLowerDecisionConfig {
                            proof_timeout_secs: 10.0,
                            max_tree_leaves: 4,
                        },
                    )
                    .expect("AY whole-comb verification and NY leaf replay agree")
                    .expect("one root-easy leaf and the three nested leaves close p<=7/8");
                assert_eq!(proof.lower.to_bits(), 0.875_f32.to_bits());
                assert_eq!(
                    proof.proof_route,
                    CertifiedLinearLowerProofRoute::TreeFarkas
                );
                assert_eq!(proof.ay_tree_leaves, 4);
                assert_eq!(proof.ny_cert_farkas_replays, 4);
            }
        }
    }

    #[test]
    fn adaptive_five_leaf_comb_replays_exactly_five_tied_tree_only_leaves() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let mut problem = MilpProblem::new();
        let root = problem.add_integer_col(0.0, 0.0, 1.0);
        let second = problem.add_integer_col(0.0, 0.0, 1.0);
        let third = problem.add_integer_col(0.0, 0.0, 1.0);
        let fourth = problem.add_integer_col(0.0, 0.0, 1.0);
        let p = problem.add_col(0.0, 0.0, 2.0);
        problem.add_row(1.0, f64::INFINITY, [(p, 1.0)]);
        let candidates = [root, second, third, fourth];

        let proof =
            certify_linear_lower_bound_at_with_ay_adaptive_five_leaf_comb_target_fsb_admission(
                &problem,
                &[(p, 1.0)],
                0.75,
                &candidates,
                0,
                false,
                CertifiedLinearLowerDecisionConfig {
                    proof_timeout_secs: 10.0,
                    max_tree_leaves: 5,
                },
                CertifiedLinearLowerWorkerAdmission::try_acquire()
                    .expect("test owns the exact-worker admission"),
            )
            .expect("AY whole-comb verification and NY leaf replay agree")
            .expect("every exact leaf inherits the global p>=1 row");
        assert_eq!(proof.lower.to_bits(), 0.75_f32.to_bits());
        assert_eq!(
            proof.proof_route,
            CertifiedLinearLowerProofRoute::TreeFarkas
        );
        assert_eq!(proof.ay_tree_leaves, 5);
        assert_eq!(proof.ny_cert_farkas_replays, 5);

        assert!(
            certify_linear_lower_bound_at_with_ay_adaptive_five_leaf_comb_target_fsb_unwired(
                &problem,
                &[(p, 1.0)],
                0.75,
                &candidates,
                0,
                false,
                CertifiedLinearLowerDecisionConfig {
                    proof_timeout_secs: 10.0,
                    max_tree_leaves: 4,
                },
            )
            .expect("underpriced five-leaf diagnostic is an ordinary decline")
            .is_none(),
            "a five-leaf comb must not cross four-leaf admission"
        );
        assert!(matches!(
            certify_linear_lower_bound_at_with_ay_adaptive_five_leaf_comb_target_fsb_unwired(
                &problem,
                &[(p, 1.0)],
                0.75,
                &candidates[..3],
                0,
                false,
                CertifiedLinearLowerDecisionConfig {
                    proof_timeout_secs: 10.0,
                    max_tree_leaves: 5,
                },
            ),
            Err(MipError::Encoding(_))
        ));
    }

    #[test]
    fn adaptive_four_leaf_comb_respects_admission_and_rejects_malformed_candidates() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let (problem, root, dummy1, second, dummy2, third, p) =
            adaptive_four_leaf_comb_problem(true, false, false);
        let candidates = [dummy1, root, dummy2, second, third];
        let proof =
            certify_linear_lower_bound_at_with_ay_adaptive_four_leaf_comb_target_fsb_unwired(
                &problem,
                &[(p, 1.0)],
                0.875,
                &candidates,
                1,
                true,
                CertifiedLinearLowerDecisionConfig {
                    proof_timeout_secs: 10.0,
                    max_tree_leaves: 3,
                },
            )
            .expect("underpriced comb diagnostic is an ordinary decline");
        assert!(
            proof.is_none(),
            "a four-leaf comb must not cross a three-leaf admission"
        );

        for malformed in [
            certify_linear_lower_bound_at_with_ay_adaptive_four_leaf_comb_target_fsb_unwired(
                &problem,
                &[(p, 1.0)],
                0.875,
                &[root, second],
                0,
                true,
                CertifiedLinearLowerDecisionConfig {
                    proof_timeout_secs: 10.0,
                    max_tree_leaves: 4,
                },
            ),
            certify_linear_lower_bound_at_with_ay_adaptive_four_leaf_comb_target_fsb_unwired(
                &problem,
                &[(p, 1.0)],
                0.875,
                &[root, second, second, third],
                0,
                true,
                CertifiedLinearLowerDecisionConfig {
                    proof_timeout_secs: 10.0,
                    max_tree_leaves: 4,
                },
            ),
            certify_linear_lower_bound_at_with_ay_adaptive_four_leaf_comb_target_fsb_unwired(
                &problem,
                &[(p, 1.0)],
                0.875,
                &candidates,
                candidates.len(),
                true,
                CertifiedLinearLowerDecisionConfig {
                    proof_timeout_secs: 10.0,
                    max_tree_leaves: 4,
                },
            ),
        ] {
            assert!(
                matches!(malformed, Err(MipError::Encoding(_))),
                "malformed public comb requests must fail before AY"
            );
        }
    }

    #[test]
    fn adaptive_four_leaf_comb_remains_tree_only_and_checks_false_hard_ties() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let (mut problem, root, _, second, _, third, p) =
            adaptive_four_leaf_comb_problem(false, false, false);
        // The unfixed root now proves the target. AY's comb API deliberately
        // skips that fast path, scans tied candidates deterministically, and
        // still returns an exact four-leaf tree.
        problem.add_row(1.0, f64::INFINITY, [(p, 1.0)]);
        let proof =
            certify_linear_lower_bound_at_with_ay_adaptive_four_leaf_comb_target_fsb_unwired(
                &problem,
                &[(p, 1.0)],
                0.875,
                &[root, second, third],
                0,
                false,
                CertifiedLinearLowerDecisionConfig {
                    proof_timeout_secs: 10.0,
                    max_tree_leaves: 4,
                },
            )
            .expect("tree-only tie-oriented comb verifies at both authority layers")
            .expect("all four exact leaves inherit the global p>=1 row");
        assert_eq!(
            proof.proof_route,
            CertifiedLinearLowerProofRoute::TreeFarkas
        );
        assert_eq!(proof.ay_tree_leaves, 4);
        assert_eq!(proof.ny_cert_farkas_replays, 4);
    }

    #[test]
    fn selected_threshold_branch_advice_crosses_public_authority_seam() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let (problem, x, z) = one_split_integrality_gap_problem();

        let certified = certify_linear_lower_bound_at_with_ay_branch_advice(
            &problem,
            &[(z, 1.0)],
            0.75,
            &[x],
            decision_config(),
        )
        .expect("solver/checkers agree")
        .expect("the advised two-child proof closes the integrality gap");
        assert_eq!(certified.lower.to_bits(), 0.75_f32.to_bits());
        assert_eq!(
            certified.proof_route,
            CertifiedLinearLowerProofRoute::TreeFarkas
        );
        assert_eq!(certified.ay_tree_leaves, 2);
        assert_eq!(certified.ny_cert_farkas_replays, 2);
    }

    #[test]
    fn selected_continuous_threshold_skips_proposal_and_replays_linear_proof() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let mut problem = MilpProblem::new();
        let x = problem.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
        problem.add_row(1.0, f64::INFINITY, [(x, 1.0)]);

        let certified =
            certify_linear_lower_bound_at_with_ay(&problem, &[(x, 1.0)], 0.99, decision_config())
                .expect("solver/checkers agree")
                .expect("strictly separated root proof");
        assert_eq!(certified.lower.to_bits(), 0.99_f32.to_bits());
        assert_eq!(
            certified.proof_route,
            CertifiedLinearLowerProofRoute::RelaxationEntailment
        );
        assert_eq!(certified.ay_tree_leaves, 0);
        assert_eq!(certified.ny_cert_farkas_replays, 1);

        assert!(
            certify_linear_lower_bound_at_with_ay(&problem, &[(x, 1.0)], 1.0, decision_config(),)
                .expect("feasible equality is an ordinary non-certificate")
                .is_none(),
            "x=1 satisfies the non-strict decision row, so q=1 is not certified"
        );
    }

    #[test]
    fn continuous_root_infeasibility_requires_verified_root_farkas_and_replay() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let mut infeasible = MilpProblem::new();
        let x = infeasible.add_col(0.0, 0.0, 1.0);
        infeasible.add_row(2.0, f64::INFINITY, [(x, 1.0)]);

        let certified = certify_continuous_root_infeasibility_with_ay_until(
            &infeasible,
            Instant::now() + Duration::from_secs(10),
        )
        .expect("AY and ny-cert agree")
        .expect("x in [0,1] and x>=2 has a root Farkas proof");
        assert!(certified.ay_farkas_multipliers >= 2);
        assert_eq!(certified.ny_cert_farkas_replays, 1);

        let mut feasible = MilpProblem::new();
        let x = feasible.add_col(0.0, 0.0, 1.0);
        feasible.add_row(0.5, f64::INFINITY, [(x, 1.0)]);
        assert!(certify_continuous_root_infeasibility_with_ay_until(
            &feasible,
            Instant::now() + Duration::from_secs(10),
        )
        .expect("a feasible LP is an ordinary non-proof")
        .is_none());

        let mut integral = MilpProblem::new();
        integral.add_integer_col(0.0, 0.0, 1.0);
        assert!(certify_continuous_root_infeasibility_with_ay_until(
            &integral,
            Instant::now() + Duration::from_secs(10),
        )
        .expect("integer models decline before solving")
        .is_none());
        assert!(certify_continuous_root_infeasibility_with_ay_until(
            &infeasible,
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .expect("one millisecond fits before now"),
        )
        .expect("an expired deadline is an ordinary non-proof")
        .is_none());
    }

    #[test]
    fn continuous_root_ny_cert_replay_rejects_corruption_and_stale_ir() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let mut problem = MilpProblem::new();
        let x = problem.add_col(0.0, 0.0, 1.0);
        let row = problem.add_row(2.0, f64::INFINITY, [(x, 1.0)]);
        let model = to_ay_model(&problem).expect("valid continuous IR");
        let ay_row = model.row_at(row.0).expect("row identity is preserved");
        let ay_x = model.col_at(x.0).expect("column identity is preserved");
        let one = BigRational::from_integer(1.into());
        let certificate = AyFarkasCertificate {
            multipliers: vec![
                ay_milp::Multiplier {
                    fact: FactRef::RowBound {
                        row: ay_row,
                        side: BoundSide::Lower,
                    },
                    coeff: one.clone(),
                },
                ay_milp::Multiplier {
                    fact: FactRef::ColBound {
                        col: ay_x,
                        side: BoundSide::Upper,
                    },
                    coeff: one,
                },
            ],
        };
        certificate
            .verify(&model)
            .expect("hand-built AY certificate is valid");
        replay_root_farkas(&problem, &certificate)
            .expect("independent ny-cert replay accepts the original evidence");

        let mut wrong_multiplier = certificate.clone();
        wrong_multiplier.multipliers[0].coeff = BigRational::from_integer(2.into());
        assert!(replay_root_farkas(&problem, &wrong_multiplier).is_err());

        let mut wrong_side = certificate.clone();
        wrong_side.multipliers[0].fact = FactRef::RowBound {
            row: ay_row,
            side: BoundSide::Upper,
        };
        assert!(replay_root_farkas(&problem, &wrong_side).is_err());

        let mut nonpositive = certificate.clone();
        nonpositive.multipliers[0].coeff = BigRational::zero();
        assert!(replay_root_farkas(&problem, &nonpositive).is_err());

        let mut stale_problem = MilpProblem::new();
        let stale_x = stale_problem.add_col(0.0, 0.0, 1.0);
        stale_problem.add_row(0.5, f64::INFINITY, [(stale_x, 1.0)]);
        assert!(replay_root_farkas(&stale_problem, &certificate).is_err());
    }

    #[test]
    fn continuous_root_resource_check_rejects_duplicate_facts_before_explicit_replay() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let mut problem = MilpProblem::new();
        let x = problem.add_col(0.0, 0.0, 1.0);
        let row = problem.add_row(2.0, f64::INFINITY, [(x, 1.0)]);
        let model = to_ay_model(&problem).expect("valid continuous IR");
        let ay_row = model.row_at(row.0).expect("row identity is preserved");
        let ay_x = model.col_at(x.0).expect("column identity is preserved");
        let half = BigRational::new(1.into(), 2.into());
        let certificate = AyFarkasCertificate {
            multipliers: vec![
                ay_milp::Multiplier {
                    fact: FactRef::RowBound {
                        row: ay_row,
                        side: BoundSide::Lower,
                    },
                    coeff: half.clone(),
                },
                ay_milp::Multiplier {
                    fact: FactRef::RowBound {
                        row: ay_row,
                        side: BoundSide::Lower,
                    },
                    coeff: half,
                },
                ay_milp::Multiplier {
                    fact: FactRef::ColBound {
                        col: ay_x,
                        side: BoundSide::Upper,
                    },
                    coeff: BigRational::from_integer(1.into()),
                },
            ],
        };

        certificate
            .verify(&model)
            .expect("splitting one multiplier preserves the exact proof");
        assert_eq!(
            continuous_root_farkas_resource_usage(&problem, &certificate),
            Err(ContinuousRootFarkasResourceRejection::DuplicateFact { index: 1 })
        );
    }

    #[test]
    fn continuous_root_resource_check_accounts_for_dense_fact_expansion() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let mut problem = MilpProblem::new();
        let columns = std::array::from_fn::<_, 4, _>(|_| problem.add_col(0.0, 0.0, 0.0));
        let row = problem.add_row(1.0, f64::INFINITY, columns.map(|column| (column, 1.0)));
        let model = to_ay_model(&problem).expect("valid continuous IR");
        let ay_row = model.row_at(row.0).expect("row identity is preserved");
        let one = BigRational::from_integer(1.into());
        let mut multipliers = vec![ay_milp::Multiplier {
            fact: FactRef::RowBound {
                row: ay_row,
                side: BoundSide::Lower,
            },
            coeff: one.clone(),
        }];
        for column in columns {
            multipliers.push(ay_milp::Multiplier {
                fact: FactRef::ColBound {
                    col: model
                        .col_at(column.0)
                        .expect("column identity is preserved"),
                    side: BoundSide::Upper,
                },
                coeff: one.clone(),
            });
        }
        let certificate = AyFarkasCertificate { multipliers };

        certificate
            .verify(&model)
            .expect("dense hand-built AY certificate is valid");
        assert_eq!(
            continuous_root_farkas_resource_usage(&problem, &certificate),
            Ok(ContinuousRootFarkasResourceUsage {
                multipliers: 5,
                rational_bits: 10,
                referenced_row_entries: 4,
                referenced_row_nonzeros: 4,
                column_bound_facts: 4,
                expanded_fact_terms: 8,
                model_rational_bits: 18,
                weighted_replay_work: 34,
            })
        );

        let mut limits = CONTINUOUS_ROOT_FARKAS_RESOURCE_LIMITS;
        limits.max_total_rational_bits = 9;
        assert_eq!(
            continuous_root_farkas_resource_usage_with_limits(&problem, &certificate, limits),
            Err(ContinuousRootFarkasResourceRejection::RationalTotalBits)
        );

        let mut limits = CONTINUOUS_ROOT_FARKAS_RESOURCE_LIMITS;
        limits.max_referenced_row_entries = 3;
        assert_eq!(
            continuous_root_farkas_resource_usage_with_limits(&problem, &certificate, limits),
            Err(ContinuousRootFarkasResourceRejection::ReferencedRowEntries)
        );

        let mut limits = CONTINUOUS_ROOT_FARKAS_RESOURCE_LIMITS;
        limits.max_referenced_row_nonzeros = 3;
        assert_eq!(
            continuous_root_farkas_resource_usage_with_limits(&problem, &certificate, limits),
            Err(ContinuousRootFarkasResourceRejection::ReferencedRowNonzeros)
        );

        let mut limits = CONTINUOUS_ROOT_FARKAS_RESOURCE_LIMITS;
        limits.max_expanded_fact_terms = 7;
        assert_eq!(
            continuous_root_farkas_resource_usage_with_limits(&problem, &certificate, limits),
            Err(ContinuousRootFarkasResourceRejection::ExpandedFactTerms)
        );

        let mut limits = CONTINUOUS_ROOT_FARKAS_RESOURCE_LIMITS;
        limits.max_model_rational_bits = 17;
        assert_eq!(
            continuous_root_farkas_resource_usage_with_limits(&problem, &certificate, limits),
            Err(ContinuousRootFarkasResourceRejection::ModelRationalBits)
        );

        let mut limits = CONTINUOUS_ROOT_FARKAS_RESOURCE_LIMITS;
        limits.max_weighted_replay_work = 33;
        assert_eq!(
            continuous_root_farkas_resource_usage_with_limits(&problem, &certificate, limits),
            Err(ContinuousRootFarkasResourceRejection::WeightedReplayWork)
        );
    }

    #[test]
    fn exact_f64_resource_bit_count_matches_big_rational_reduction() {
        for value in [
            0.0,
            -0.0,
            1.0,
            -0.5,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            f64::MAX,
        ] {
            let exact = BigRational::from_float(value).expect("finite binary64");
            assert_eq!(
                exact_f64_rational_bits(value),
                Some(exact.numer().bits() + exact.denom().bits()),
                "mismatched exact-rational bit charge for {value:e}"
            );
        }
        assert_eq!(exact_f64_rational_bits(f64::INFINITY), None);
        assert_eq!(exact_f64_rational_bits(f64::NAN), None);
    }

    #[test]
    fn continuous_root_problem_preprices_complete_exact_model() {
        let mut problem = MilpProblem::new();
        let x = problem.add_col(0.0, 0.0, 1.0);
        problem.add_row(2.0, f64::INFINITY, [(x, 1.0)]);

        assert_eq!(
            continuous_root_problem_resource_usage(&problem),
            Ok(ContinuousRootProblemResourceUsage {
                columns: 1,
                rows: 1,
                row_entries: 1,
                rational_bits: 9,
            })
        );

        let mut limits = CONTINUOUS_ROOT_PROBLEM_RESOURCE_LIMITS;
        limits.max_row_entries = 0;
        assert_eq!(
            continuous_root_problem_resource_usage_with_limits(&problem, limits),
            Err(ContinuousRootProblemResourceRejection::RowEntries)
        );

        let mut limits = CONTINUOUS_ROOT_PROBLEM_RESOURCE_LIMITS;
        limits.max_rational_bits = 8;
        assert_eq!(
            continuous_root_problem_resource_usage_with_limits(&problem, limits),
            Err(ContinuousRootProblemResourceRejection::RationalBits)
        );

        let mut limits = CONTINUOUS_ROOT_PROBLEM_RESOURCE_LIMITS;
        limits.max_columns = 0;
        assert_eq!(
            continuous_root_problem_resource_usage_with_limits(&problem, limits),
            Err(ContinuousRootProblemResourceRejection::Columns)
        );

        let mut empty_row_problem = MilpProblem::new();
        empty_row_problem.add_row(f64::NEG_INFINITY, f64::INFINITY, []);
        let mut limits = CONTINUOUS_ROOT_PROBLEM_RESOURCE_LIMITS;
        limits.max_rows = 0;
        assert_eq!(
            continuous_root_problem_resource_usage_with_limits(&empty_row_problem, limits),
            Err(ContinuousRootProblemResourceRejection::Rows),
            "empty unbounded rows still consume model and clone resources"
        );

        let mut continuous_columns = MilpProblem::new();
        continuous_columns.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
        continuous_columns.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
        let mut limits = CONTINUOUS_ROOT_PROBLEM_RESOURCE_LIMITS;
        limits.max_columns = 1;
        assert_eq!(
            continuous_root_problem_resource_usage_with_limits(&continuous_columns, limits),
            Err(ContinuousRootProblemResourceRejection::Columns),
            "all-continuous columns must be capped before the integer scan"
        );

        let mut malformed = MilpProblem::new();
        let x = malformed.add_col(0.0, 0.0, 1.0);
        malformed.add_row(0.0, 1.0, [(x, f64::NAN)]);
        assert_eq!(
            continuous_root_problem_resource_usage(&malformed),
            Err(ContinuousRootProblemResourceRejection::MalformedModel)
        );
    }

    #[test]
    fn continuous_root_resource_check_rejects_dense_large_multiplier_cross_product() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        const DENSE_TERMS: usize = 4_096;

        let mut problem = MilpProblem::new();
        let mut coefficients = Vec::with_capacity(DENSE_TERMS);
        for _ in 0..DENSE_TERMS {
            let column = problem.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
            coefficients.push((column, 1.0));
        }
        let row = problem.add_row(1.0, f64::INFINITY, coefficients);
        let model = to_ay_model(&problem).expect("valid dense continuous IR");
        let ay_row = model.row_at(row.0).expect("row identity is preserved");
        // 32,767 numerator bits plus the denominator's one bit reaches the
        // ordinary per-component/aggregate ceilings without exceeding them.
        let numerator = num_bigint::BigInt::from(1_u8) << 32_766_usize;
        let certificate = AyFarkasCertificate {
            multipliers: vec![ay_milp::Multiplier {
                fact: FactRef::RowBound {
                    row: ay_row,
                    side: BoundSide::Lower,
                },
                coeff: BigRational::from_integer(numerator),
            }],
        };

        assert_eq!(
            continuous_root_farkas_resource_usage(&problem, &certificate),
            Err(ContinuousRootFarkasResourceRejection::WeightedReplayWork),
            "individually admitted nnz and multiplier-bit caps must not multiply into unbounded replay work"
        );
    }

    #[test]
    fn infeasible_relaxation_falls_back_to_root_farkas() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let mut problem = MilpProblem::new();
        let x = problem.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
        problem.add_row(1.0, f64::INFINITY, [(x, 1.0)]);
        problem.add_row(f64::NEG_INFINITY, 0.0, [(x, 1.0)]);

        let certified =
            certify_linear_lower_bound_at_with_ay(&problem, &[(x, 1.0)], 0.0, decision_config())
                .expect("solver/checkers agree")
                .expect("infeasible model has a root proof");
        assert_eq!(
            certified.proof_route,
            CertifiedLinearLowerProofRoute::RootFarkas
        );
        assert_eq!(certified.ay_tree_leaves, 0);
        assert_eq!(certified.ny_cert_farkas_replays, 1);
    }

    #[test]
    fn selected_integral_threshold_replays_every_tree_leaf() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let mut problem = MilpProblem::new();
        let x = problem.add_integer_col(0.0, 0.0, 1.0);
        let y = problem.add_integer_col(0.0, 0.0, 1.0);
        let z = problem.add_integer_col(0.0, 0.0, 1.0);
        problem.add_row(1.5, f64::INFINITY, [(x, 1.0), (y, 1.0), (z, 1.0)]);
        let objective = canonical_objective(&problem, &[(x, 1.0), (y, 1.0), (z, 1.0)]).unwrap();
        let opts = solve_opts(10.0).with_require_certificates(true);
        assert!(
            try_relaxed_linear_lower_proof(&problem, &objective, 1.75, &[], 64, &opts)
                .expect("relaxation solve/checkers agree")
                .is_none(),
            "LP optimum 1.5 cannot certify an integer-only lower threshold of 1.75"
        );

        let certified = certify_linear_lower_bound_at_with_ay(
            &problem,
            &[(x, 1.0), (y, 1.0), (z, 1.0)],
            1.75,
            decision_config(),
        )
        .expect("solver/checkers agree")
        .expect("integer-separated tree proof");
        assert_eq!(certified.lower, 1.75);
        assert_eq!(
            certified.proof_route,
            CertifiedLinearLowerProofRoute::TreeFarkas
        );
        assert!(certified.ay_tree_leaves >= 2);
        assert_eq!(certified.ny_cert_farkas_replays, certified.ay_tree_leaves);
    }

    #[test]
    fn integrality_gap_lower_bound_replays_every_tree_leaf() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        // LP relaxation: x=y=z=0.5 gives x+y+z=1.5.
        // Binary model: x+y+z>=1.5 forces at least two variables on, so
        // min(x+y+z)=2.
        // The downward q is between the two and therefore needs an integer
        // case split; a root-LP Farkas proof cannot exist.
        let mut problem = MilpProblem::new();
        let x = problem.add_integer_col(0.0, 0.0, 1.0);
        let y = problem.add_integer_col(0.0, 0.0, 1.0);
        let z = problem.add_integer_col(0.0, 0.0, 1.0);
        problem.add_row(1.5, f64::INFINITY, [(x, 1.0), (y, 1.0), (z, 1.0)]);

        let certified =
            certify_linear_lower_bound_with_ay(&problem, &[(x, 1.0), (y, 1.0), (z, 1.0)], config())
                .expect("solver/checkers agree")
                .expect("tree proof");
        assert!(certified.lower < 2.0);
        assert!(certified.lower > 1.5);
        assert_eq!(
            certified.proof_route,
            CertifiedLinearLowerProofRoute::TreeFarkas
        );
        assert!(certified.ay_tree_leaves >= 2);
        assert_eq!(certified.ny_cert_farkas_replays, certified.ay_tree_leaves);
    }

    #[test]
    fn malformed_objective_and_unpriced_tree_are_rejected() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let mut problem = MilpProblem::new();
        let x = problem.add_col(0.0, 0.0, 1.0);
        assert!(certify_linear_lower_bound_with_ay(&problem, &[], config()).is_err());
        assert!(
            certify_linear_lower_bound_with_ay(&problem, &[(x, 1.0), (x, 2.0)], config()).is_err()
        );
        let mut invalid = config();
        invalid.max_tree_leaves = CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES + 1;
        assert!(certify_linear_lower_bound_with_ay(&problem, &[(x, 1.0)], invalid).is_err());

        for nonfinite in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
            assert!(certify_linear_lower_bound_at_with_ay(
                &problem,
                &[(x, 1.0)],
                nonfinite,
                decision_config(),
            )
            .is_err());
        }
        let mut invalid_decision = decision_config();
        invalid_decision.proof_timeout_secs = 0.0;
        assert!(certify_linear_lower_bound_at_with_ay(
            &problem,
            &[(x, 1.0)],
            0.5,
            invalid_decision,
        )
        .is_err());
    }
}
