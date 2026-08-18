// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Deadline-bounded GPU-only margin-alpha optimization over one complete
//! selected root-`C` row set.
//!
//! This is a subordinate, exact-dark consumer of [`super::atomic_cuda_rows`].
//! It evaluates the bootstrap alpha (`alpha0`) as one atomic transaction over
//! every supplied row, uses CUDA's call-local deadline-bounded joint adjoint
//! only to propose an Adam step, evaluates the proposal (`alpha1`) over that
//! same complete row set, and selects one whole evaluated bounds/state pair.
//! A caller may supply only unresolved rows when every omitted source row has
//! an independent certified proof and all objectives, thresholds, reference
//! intervals, and `C` rows were compacted through one source map. This module
//! never samples or independently filters the selected rows.
//! An independently gated CIFAR experiment evaluates a fixed three-LR bracket
//! from the same alpha0/adjoint. The bracket is one fail-closed transaction:
//! every candidate must finish and validate before a unique strict winner can
//! replace alpha0.
//! A mutually exclusive top-K experiment selects the eight worst unresolved
//! rows, flattens them into one joint adjoint (whose ReLU gradient is their
//! exact sum at alpha0), applies one configured-base Adam step, and authorizes
//! it only through one complete selected-row reevaluation.
//! No gradient can become proof authority and no row-wise mixture can be paired
//! with an alpha state that did not produce it.
//! The typed multi-iteration route is independently default-dark and capped at
//! eight proposals. It dynamically rebinds the worst unresolved row after every
//! accepted whole-pair step, differentiates at that latest accepted state, and
//! advances the existing Adam time/LR schedule. A proposal replaces authority
//! only when its finite complete-`C` hinge is strictly larger; bounds and alpha
//! state always move together.
//! An exact-dark child replaces the per-round point mass with a persistent
//! multiplicative-weights row player.  Every row retains a strictly positive
//! scale, and requests wider than eight rows are summed across bounded resident
//! adjoint chunks. The first full-row plan is uniform; only later plans use a
//! post-update distribution, and telemetry distinguishes those cases.
//! Candidate authority remains the same strict complete-C hinge comparison.
//!
//! The row player observes slacks from the authoritative intersection of the
//! reference and CUDA bounds. The adjoint differentiates the CUDA relaxation
//! before that intersection, so it is an exact gradient of the weighted CUDA
//! surrogate, not generally a derivative of a reference-dominated intersected
//! endpoint. This mismatch affects proposal quality only: complete-C
//! reevaluation remains the sole acceptance and publication authority.

use std::collections::{BTreeSet, HashMap};
use std::time::{Duration, Instant};

use ndarray::{Array1, Array2};
use ny_tensor::BoundedTensor;

use crate::beta_crown::config::ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS;
use crate::bounds::{AdamParams, GraphAlphaState};
use crate::network::core::{GraphNetwork, NETWORK_INPUT};

use super::atomic_cuda_rows::{
    AtomicCudaRowsCommit, AtomicCudaRowsOutcome, AtomicCudaRowsRefusal, AtomicCudaRowsRequest,
};
use super::resnet_decompose::{
    joint_alpha_grads_fold_gpu_with_deadline, resnet_gpu_enabled, DeadlineJointAlphaFoldError,
};
use super::resnet_skeleton::{build_resnet_segment_skeleton, extract_skeleton_enabled};
use super::row_weights::RowWeights;

fn parse_root_alpha_cuda_margin_step(raw: Option<&str>) -> bool {
    raw == Some("1")
}

const ATOMIC_CUDA_MARGIN_LR_MULTIPLIERS: [f32; 3] = [0.3, 1.0, 2.0];
const ATOMIC_CUDA_MARGIN_TOPK: usize = 8;
/// Maximum cooperatively deadline-bounded resident adjoint calls in one MW
/// gradient proposal. The shared engine reference is acquired once around the
/// complete chunk loop; this is a call-count bound, not a no-contention claim.
const ATOMIC_CUDA_MARGIN_MW_MAX_GRADIENT_CALLS: usize = 32;
/// Hard host-side admission cap for the exact positive-weight row player.
///
/// CIFAR100 has 99 margin rows and TinyImageNet has up to 200.  The resident
/// adjoint consumes these in [`ATOMIC_CUDA_MARGIN_TOPK`]-sized chunks, so this
/// cap bounds both the seed allocation and the number of accelerator calls
/// without widening the backend's established eight-row call surface.
const ATOMIC_CUDA_MARGIN_MW_MAX_ROWS: usize =
    ATOMIC_CUDA_MARGIN_TOPK * ATOMIC_CUDA_MARGIN_MW_MAX_GRADIENT_CALLS;
/// At most 256 KiB per f32 seed (and another 256 KiB for the immutable oriented
/// rows). This bounds committed memory touches, not merely `Vec::try_reserve`
/// under an overcommitting allocator.
const ATOMIC_CUDA_MARGIN_MW_MAX_SEED_ELEMENTS: usize = 65_536;
// CIFAR's configured alpha-CROWN LR is at most 0.25. The 2x arm deliberately
// reaches 0.5, while all candidate alphas remain projected into [0, 1].
const ATOMIC_CUDA_MARGIN_MAX_BASE_LR: f32 = 0.25;
const ATOMIC_CUDA_MARGIN_MAX_CANDIDATE_LR: f32 = 0.5;
const ATOMIC_CUDA_MARGIN_PUBLICATION_RESERVE: Duration = Duration::from_millis(25);

/// Subordinate gate. Callers read it only after
/// `NY_ROOT_ALPHA_CUDA_ROWS=1` has armed the parent transaction.
pub(crate) fn root_alpha_cuda_margin_step_enabled() -> bool {
    parse_root_alpha_cuda_margin_step(
        std::env::var("NY_ROOT_ALPHA_CUDA_MARGIN_STEP")
            .ok()
            .as_deref(),
    )
}

fn parse_root_alpha_cuda_margin_lr_bracket(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Exact-dark child of the existing rows + margin-step gates. Keeping this
/// lookup inside [`AtomicCudaMarginStepRequest::run`] means setting the bracket
/// alone cannot construct an optimizer, request a factory, or execute CUDA.
fn root_alpha_cuda_margin_lr_bracket_enabled() -> bool {
    parse_root_alpha_cuda_margin_lr_bracket(
        std::env::var("NY_ROOT_ALPHA_CUDA_MARGIN_LR_BRACKET")
            .ok()
            .as_deref(),
    )
}

fn parse_root_alpha_cuda_margin_topk(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Exact-dark sibling of the LR bracket. One flattened top-K joint adjoint
/// proposes one state; only a subsequent complete-C evaluation may publish it.
fn root_alpha_cuda_margin_topk_enabled() -> bool {
    parse_root_alpha_cuda_margin_topk(
        std::env::var("NY_ROOT_ALPHA_CUDA_MARGIN_TOPK")
            .ok()
            .as_deref(),
    )
}

fn parse_root_alpha_cuda_margin_mw(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Exact-dark child of the typed multi-iteration route.  The short-circuit is
/// load-bearing: callers with `iterations == 0` must not even read the child
/// environment variable.
fn root_alpha_cuda_margin_mw_enabled_if<F>(iterations: usize, read: F) -> bool
where
    F: FnOnce() -> Option<String>,
{
    iterations > 0 && parse_root_alpha_cuda_margin_mw(read().as_deref())
}

fn root_alpha_cuda_margin_mw_enabled(iterations: usize) -> bool {
    root_alpha_cuda_margin_mw_enabled_if(iterations, || {
        std::env::var("NY_ROOT_ALPHA_CUDA_MARGIN_MW").ok()
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AtomicCudaMarginSearchPolicy {
    Legacy,
    LearningRateBracket,
    TopK,
}

fn multi_iteration_search_policy_is_exclusive(
    iterations: usize,
    topk_enabled: bool,
    lr_bracket_enabled: bool,
) -> Result<(), AtomicCudaMarginStepRefusal> {
    if iterations > 0 && (topk_enabled || lr_bracket_enabled) {
        Err(AtomicCudaMarginStepRefusal::ConflictingSearchPolicy)
    } else {
        Ok(())
    }
}

fn select_margin_search_policy(
    topk_enabled: bool,
    lr_bracket_enabled: bool,
) -> Result<AtomicCudaMarginSearchPolicy, AtomicCudaMarginStepRefusal> {
    match (topk_enabled, lr_bracket_enabled) {
        (false, false) => Ok(AtomicCudaMarginSearchPolicy::Legacy),
        (false, true) => Ok(AtomicCudaMarginSearchPolicy::LearningRateBracket),
        (true, false) => Ok(AtomicCudaMarginSearchPolicy::TopK),
        (true, true) => Err(AtomicCudaMarginStepRefusal::ConflictingSearchPolicy),
    }
}

/// Why the optional alpha1 proposal was not selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AtomicCudaMarginStepRefusal {
    Alpha0Rows(AtomicCudaRowsRefusal),
    Alpha1Rows(AtomicCudaRowsRefusal),
    ObjectiveMode,
    GradientPolicy,
    OptimizerPolicy,
    ThresholdShape,
    NonFiniteThreshold,
    BoundsShape,
    BoundsNonFiniteOrInverted,
    ScoreNonFinite,
    NoUnverifiedBinding,
    NonReluAlphaState,
    SpecAxisAlphaState,
    InvalidAdamParams,
    ResnetGpuDisabled,
    SkeletonDisabled,
    SkeletonRefused,
    FoldRefused,
    ReluAlignment,
    MissingPreactivation,
    PreLowerShape,
    InputNonFinite,
    DeadlineExceeded,
    FactoryUnavailable,
    FactoryAdmissionError,
    NoSoundGpuRoute,
    JointUnavailable,
    JointNonFinite,
    JointMapping,
    AlphaUpdateRefused,
    AlphaDidNotMove,
    ConflictingSearchPolicy,
    InvalidIterationPolicy,
    WorkDeadlineExceeded,
    InvalidTopKPolicy,
    InvalidTopKObjective,
    InvalidMultiplicativeWeights,
    InvalidLearningRateBracket,
    IncompleteLearningRateBracket,
    MalformedLearningRateCandidate,
}

impl AtomicCudaMarginStepRefusal {
    pub(crate) fn telemetry_reason(self) -> &'static str {
        match self {
            Self::Alpha0Rows(_) => "alpha0_rows",
            Self::Alpha1Rows(_) => "alpha1_rows",
            Self::ObjectiveMode => "objective_mode",
            Self::GradientPolicy => "gradient_policy",
            Self::OptimizerPolicy => "optimizer_policy",
            Self::ThresholdShape => "threshold_shape",
            Self::NonFiniteThreshold => "nonfinite_threshold",
            Self::BoundsShape => "bounds_shape",
            Self::BoundsNonFiniteOrInverted => "bounds_nonfinite_or_inverted",
            Self::ScoreNonFinite => "score_nonfinite",
            Self::NoUnverifiedBinding => "no_unverified_binding",
            Self::NonReluAlphaState => "non_relu_alpha_state",
            Self::SpecAxisAlphaState => "spec_axis_alpha_state",
            Self::InvalidAdamParams => "invalid_adam_params",
            Self::ResnetGpuDisabled => "resnet_gpu_disabled",
            Self::SkeletonDisabled => "skeleton_disabled",
            Self::SkeletonRefused => "skeleton_refused",
            Self::FoldRefused => "fold_refused",
            Self::ReluAlignment => "relu_alignment",
            Self::MissingPreactivation => "missing_preactivation",
            Self::PreLowerShape => "pre_lower_shape",
            Self::InputNonFinite => "input_nonfinite",
            Self::DeadlineExceeded => "deadline_exceeded",
            Self::FactoryUnavailable => "factory_unavailable",
            Self::FactoryAdmissionError => "factory_admission_error",
            Self::NoSoundGpuRoute => "no_sound_gpu_route",
            Self::JointUnavailable => "joint_unavailable",
            Self::JointNonFinite => "joint_nonfinite",
            Self::JointMapping => "joint_mapping",
            Self::AlphaUpdateRefused => "alpha_update_refused",
            Self::AlphaDidNotMove => "alpha_did_not_move",
            Self::ConflictingSearchPolicy => "conflicting_search_policy",
            Self::InvalidIterationPolicy => "invalid_iteration_policy",
            Self::WorkDeadlineExceeded => "work_deadline_exceeded",
            Self::InvalidTopKPolicy => "invalid_topk_policy",
            Self::InvalidTopKObjective => "invalid_topk_objective",
            Self::InvalidMultiplicativeWeights => "invalid_multiplicative_weights",
            Self::InvalidLearningRateBracket => "invalid_learning_rate_bracket",
            Self::IncompleteLearningRateBracket => "incomplete_learning_rate_bracket",
            Self::MalformedLearningRateCandidate => "malformed_learning_rate_candidate",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct AtomicCudaMarginBracketPolicy {
    candidate_lrs: [f32; 3],
    work_deadline: Instant,
    authority_deadline: Instant,
}

impl AtomicCudaMarginBracketPolicy {
    fn new(
        adam: AdamParams,
        configured_base_learning_rate: f32,
        authority_deadline: Instant,
    ) -> Result<Self, AtomicCudaMarginStepRefusal> {
        Self::new_at(
            adam,
            configured_base_learning_rate,
            authority_deadline,
            Instant::now(),
        )
    }

    fn new_at(
        adam: AdamParams,
        configured_base_learning_rate: f32,
        authority_deadline: Instant,
        now: Instant,
    ) -> Result<Self, AtomicCudaMarginStepRefusal> {
        validate_adam(adam)?;
        if !configured_base_learning_rate.is_finite()
            || configured_base_learning_rate <= 0.0
            || configured_base_learning_rate > ATOMIC_CUDA_MARGIN_MAX_BASE_LR
        {
            return Err(AtomicCudaMarginStepRefusal::InvalidLearningRateBracket);
        }
        let candidate_lrs =
            ATOMIC_CUDA_MARGIN_LR_MULTIPLIERS.map(|scale| configured_base_learning_rate * scale);
        if candidate_lrs
            .iter()
            .any(|&lr| !valid_bracket_candidate_lr(lr))
        {
            return Err(AtomicCudaMarginStepRefusal::InvalidLearningRateBracket);
        }
        let work_deadline = authority_deadline
            .checked_sub(ATOMIC_CUDA_MARGIN_PUBLICATION_RESERVE)
            .ok_or(AtomicCudaMarginStepRefusal::DeadlineExceeded)?;
        if now >= work_deadline {
            return Err(AtomicCudaMarginStepRefusal::DeadlineExceeded);
        }
        Ok(Self {
            candidate_lrs,
            work_deadline,
            authority_deadline,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct AtomicCudaMarginTopKPolicy {
    learning_rate: f32,
    work_deadline: Instant,
    authority_deadline: Instant,
}

impl AtomicCudaMarginTopKPolicy {
    fn new(
        adam: AdamParams,
        configured_base_learning_rate: f32,
        authority_deadline: Instant,
    ) -> Result<Self, AtomicCudaMarginStepRefusal> {
        Self::new_at(
            adam,
            configured_base_learning_rate,
            authority_deadline,
            Instant::now(),
        )
    }

    fn new_at(
        adam: AdamParams,
        configured_base_learning_rate: f32,
        authority_deadline: Instant,
        now: Instant,
    ) -> Result<Self, AtomicCudaMarginStepRefusal> {
        validate_adam(adam)?;
        if !configured_base_learning_rate.is_finite()
            || configured_base_learning_rate <= 0.0
            || configured_base_learning_rate > ATOMIC_CUDA_MARGIN_MAX_BASE_LR
        {
            return Err(AtomicCudaMarginStepRefusal::InvalidTopKPolicy);
        }
        let work_deadline = authority_deadline
            .checked_sub(ATOMIC_CUDA_MARGIN_PUBLICATION_RESERVE)
            .ok_or(AtomicCudaMarginStepRefusal::DeadlineExceeded)?;
        if now >= work_deadline {
            return Err(AtomicCudaMarginStepRefusal::DeadlineExceeded);
        }
        Ok(Self {
            learning_rate: configured_base_learning_rate,
            work_deadline,
            authority_deadline,
        })
    }
}

/// Bounded typed exact-`C` iteration policy. The work deadline deliberately
/// ends before the hard publication authority so candidate teardown and moving
/// the latest accepted whole pair happen under a separate reserve.
#[derive(Clone, Copy, Debug, PartialEq)]
struct AtomicCudaMarginIterationsPolicy {
    iterations: usize,
    learning_rate_decay: f32,
    work_deadline: Instant,
    authority_deadline: Instant,
}

impl AtomicCudaMarginIterationsPolicy {
    fn new(
        iterations: usize,
        adam: AdamParams,
        learning_rate_decay: f32,
        authority_deadline: Instant,
    ) -> Result<Self, AtomicCudaMarginStepRefusal> {
        Self::new_at(
            iterations,
            adam,
            learning_rate_decay,
            authority_deadline,
            Instant::now(),
        )
    }

    fn new_at(
        iterations: usize,
        adam: AdamParams,
        learning_rate_decay: f32,
        authority_deadline: Instant,
        now: Instant,
    ) -> Result<Self, AtomicCudaMarginStepRefusal> {
        validate_adam(adam)?;
        if !(1..=ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS).contains(&iterations)
            || !learning_rate_decay.is_finite()
            || learning_rate_decay <= 0.0
        {
            return Err(AtomicCudaMarginStepRefusal::InvalidIterationPolicy);
        }
        let work_deadline = authority_deadline
            .checked_sub(ATOMIC_CUDA_MARGIN_PUBLICATION_RESERVE)
            .ok_or(AtomicCudaMarginStepRefusal::WorkDeadlineExceeded)?;
        if now >= work_deadline {
            return Err(AtomicCudaMarginStepRefusal::WorkDeadlineExceeded);
        }
        Ok(Self {
            iterations,
            learning_rate_decay,
            work_deadline,
            authority_deadline,
        })
    }

    fn adam_for_offset(
        self,
        base: AdamParams,
        offset: usize,
    ) -> Result<AdamParams, AtomicCudaMarginStepRefusal> {
        if offset >= self.iterations {
            return Err(AtomicCudaMarginStepRefusal::InvalidIterationPolicy);
        }
        let t = base
            .t
            .checked_add(offset)
            .ok_or(AtomicCudaMarginStepRefusal::InvalidIterationPolicy)?;
        let exponent = i32::try_from(offset)
            .map_err(|_| AtomicCudaMarginStepRefusal::InvalidIterationPolicy)?;
        let learning_rate = base.learning_rate * self.learning_rate_decay.powi(exponent);
        let scheduled = AdamParams {
            learning_rate,
            t,
            ..base
        };
        validate_adam(scheduled)?;
        Ok(scheduled)
    }
}

#[inline]
fn valid_bracket_candidate_lr(learning_rate: f32) -> bool {
    learning_rate.is_finite()
        && learning_rate > 0.0
        && learning_rate <= ATOMIC_CUDA_MARGIN_MAX_CANDIDATE_LR
}

fn poll_bracket_authority_with_clock<F>(
    policy: AtomicCudaMarginBracketPolicy,
    now: &mut F,
) -> Result<(), AtomicCudaMarginStepRefusal>
where
    F: FnMut() -> Instant,
{
    if now() >= policy.work_deadline {
        return Err(AtomicCudaMarginStepRefusal::DeadlineExceeded);
    }
    if now() >= policy.authority_deadline {
        return Err(AtomicCudaMarginStepRefusal::DeadlineExceeded);
    }
    Ok(())
}

/// Destroy all candidate-owned state before the refusal's final authority
/// poll. Full-C bounds and alpha states can be large enough that their
/// destructors are material deadline work; committing alpha0 after a poll but
/// before that destruction would create an unaccounted publication interval.
fn bracket_refusal_after_drop_with_clock<T, F>(
    owned_candidate_state: T,
    policy: AtomicCudaMarginBracketPolicy,
    refusal: AtomicCudaMarginStepRefusal,
    now: &mut F,
) -> AtomicCudaMarginStepRefusal
where
    F: FnMut() -> Instant,
{
    drop(owned_candidate_state);
    poll_bracket_authority_with_clock(policy, now).map_or_else(|deadline| deadline, |()| refusal)
}

#[derive(Clone, Debug, PartialEq)]
struct BindingRow {
    index: usize,
    slack: f32,
    lower_objective: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
struct MarginScore {
    hinge: f32,
    binding: Option<BindingRow>,
}

/// Gradient-only objective for one typed iteration.  `binding_row` remains the
/// worst complete-C row for telemetry; `lower_objectives` may be either that
/// one row (the legacy route) or every positively weighted row (MW).
#[derive(Clone, Debug, PartialEq)]
struct AtomicCudaMarginIterationPlan {
    binding_row: usize,
    lower_objectives: Vec<f32>,
    num_specs: usize,
    /// True only when this seed uses a distribution after at least one prior
    /// row-player update. The first MW seed is deliberately uniform, and a
    /// one-row player has no adaptive choice.
    adaptive_weights: bool,
}

/// Persistent exact Hedge row player driving a weighted CUDA-relaxation
/// gradient surrogate. Authoritative intersected slacks update its weights;
/// complete-C reevaluation alone decides whether its proposal is useful.
///
/// The oriented objective rows are immutable and allocated once.  Only the
/// simplex weights and the flattened scaled seed change between iterations.
struct MultiplicativeWeightsRowPlayer {
    weights: RowWeights,
    lower_objectives: Vec<Vec<f32>>,
    verify_upper_bound: bool,
}

impl MultiplicativeWeightsRowPlayer {
    fn new(
        spec_matrix: &Array2<f32>,
        verify_upper_bound: bool,
        horizon: usize,
        deadline: Instant,
    ) -> Result<Self, AtomicCudaMarginStepRefusal> {
        deadline_check(deadline)?;
        let rows = spec_matrix.nrows();
        let columns = spec_matrix.ncols();
        let seed_elements = rows
            .checked_mul(columns)
            .ok_or(AtomicCudaMarginStepRefusal::InvalidMultiplicativeWeights)?;
        if !(1..=ATOMIC_CUDA_MARGIN_MW_MAX_ROWS).contains(&rows)
            || columns == 0
            || seed_elements > ATOMIC_CUDA_MARGIN_MW_MAX_SEED_ELEMENTS
        {
            return Err(AtomicCudaMarginStepRefusal::InvalidMultiplicativeWeights);
        }

        let mut lower_objectives = Vec::new();
        lower_objectives
            .try_reserve_exact(rows)
            .map_err(|_| AtomicCudaMarginStepRefusal::InvalidMultiplicativeWeights)?;
        let mut host_work = 0usize;
        for source in spec_matrix.rows() {
            let mut objective = Vec::new();
            objective
                .try_reserve_exact(columns)
                .map_err(|_| AtomicCudaMarginStepRefusal::InvalidMultiplicativeWeights)?;
            for &value in source {
                deadline_host_work(deadline, &mut host_work)?;
                if !value.is_finite() {
                    return Err(AtomicCudaMarginStepRefusal::InvalidMultiplicativeWeights);
                }
                objective.push(if verify_upper_bound { -value } else { value });
            }
            lower_objectives.push(objective);
        }
        let weights = RowWeights::with_horizon(rows, horizon)
            .map_err(|_| AtomicCudaMarginStepRefusal::InvalidMultiplicativeWeights)?;
        deadline_check(deadline)?;
        Ok(Self {
            weights,
            lower_objectives,
            verify_upper_bound,
        })
    }

    fn objective_for(
        &mut self,
        bounds: &BoundedTensor,
        thresholds: &[f32],
        binding_row: usize,
        deadline: Instant,
    ) -> Result<AtomicCudaMarginIterationPlan, AtomicCudaMarginStepRefusal> {
        deadline_check(deadline)?;
        let rows = self.weights.rows();
        if bounds.shape() != [rows]
            || thresholds.len() != rows
            || binding_row >= rows
            || self.lower_objectives.len() != rows
        {
            return Err(AtomicCudaMarginStepRefusal::InvalidMultiplicativeWeights);
        }

        let mut slacks = Vec::new();
        slacks
            .try_reserve_exact(rows)
            .map_err(|_| AtomicCudaMarginStepRefusal::InvalidMultiplicativeWeights)?;
        let mut host_work = 0usize;
        for ((&lower, &upper), &threshold) in bounds
            .lower()
            .iter()
            .zip(bounds.upper().iter())
            .zip(thresholds)
        {
            deadline_host_work(deadline, &mut host_work)?;
            if !lower.is_finite() || !upper.is_finite() || lower > upper || !threshold.is_finite() {
                return Err(AtomicCudaMarginStepRefusal::InvalidMultiplicativeWeights);
            }
            let slack = if self.verify_upper_bound {
                threshold - upper
            } else {
                lower - threshold
            };
            if !slack.is_finite() {
                return Err(AtomicCudaMarginStepRefusal::InvalidMultiplicativeWeights);
            }
            slacks.push(slack);
        }

        if !self.weights.min_weight().is_finite() || self.weights.min_weight() <= 0.0 {
            return Err(AtomicCudaMarginStepRefusal::InvalidMultiplicativeWeights);
        }
        let adaptive_weights = self.weights.rows() > 1 && self.weights.rounds() > 0;
        // Standard Hedge chooses p_t from losses observed before round t. Build
        // this round's alpha-player seed from that pre-update distribution,
        // then observe the current complete-C slacks to prepare p_{t+1}.
        let lower_objectives = self
            .weights
            .scaled_seed(&self.lower_objectives)
            .map_err(|_| AtomicCudaMarginStepRefusal::InvalidMultiplicativeWeights)?;
        validate_joint_objective_shape(&lower_objectives, rows, self.lower_objectives[0].len())?;
        self.weights
            .update(&slacks)
            .map_err(|_| AtomicCudaMarginStepRefusal::InvalidMultiplicativeWeights)?;
        deadline_check(deadline)?;
        Ok(AtomicCudaMarginIterationPlan {
            binding_row,
            lower_objectives,
            num_specs: rows,
            adaptive_weights,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
struct AtomicCudaMarginTopKPlan {
    row_indices: Vec<usize>,
    lower_objectives: Vec<f32>,
}

fn prepare_topk_objective(
    bounds: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    verify_upper_bound: bool,
    deadline: Instant,
) -> Result<AtomicCudaMarginTopKPlan, AtomicCudaMarginStepRefusal> {
    deadline_check(deadline)?;
    let rows = spec_matrix.nrows();
    let columns = spec_matrix.ncols();
    if rows == 0 || columns == 0 || thresholds.len() != rows || bounds.shape() != [rows] {
        return Err(AtomicCudaMarginStepRefusal::InvalidTopKObjective);
    }

    let mut ranked = Vec::with_capacity(rows);
    let mut host_work = 0usize;
    for (row, ((&lower, &upper), &threshold)) in bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .zip(thresholds)
        .enumerate()
    {
        deadline_host_work(deadline, &mut host_work)?;
        if !lower.is_finite() || !upper.is_finite() || lower > upper || !threshold.is_finite() {
            return Err(AtomicCudaMarginStepRefusal::InvalidTopKObjective);
        }
        let slack = if verify_upper_bound {
            threshold - upper
        } else {
            lower - threshold
        };
        if !slack.is_finite() {
            return Err(AtomicCudaMarginStepRefusal::InvalidTopKObjective);
        }
        if slack <= 0.0 {
            ranked.push((row, slack));
        }
    }
    ranked.sort_by(|&(left_row, left_slack), &(right_row, right_slack)| {
        left_slack
            .total_cmp(&right_slack)
            .then_with(|| left_row.cmp(&right_row))
    });
    ranked.truncate(ATOMIC_CUDA_MARGIN_TOPK);
    if ranked.is_empty() {
        return Err(AtomicCudaMarginStepRefusal::NoUnverifiedBinding);
    }

    let objective_len = ranked
        .len()
        .checked_mul(columns)
        .ok_or(AtomicCudaMarginStepRefusal::InvalidTopKObjective)?;
    let mut lower_objectives = Vec::new();
    lower_objectives
        .try_reserve_exact(objective_len)
        .map_err(|_| AtomicCudaMarginStepRefusal::InvalidTopKObjective)?;
    let mut row_indices = Vec::with_capacity(ranked.len());
    for (row, _) in ranked {
        deadline_host_work(deadline, &mut host_work)?;
        row_indices.push(row);
        for &value in spec_matrix.row(row) {
            deadline_host_work(deadline, &mut host_work)?;
            if !value.is_finite() {
                return Err(AtomicCudaMarginStepRefusal::InvalidTopKObjective);
            }
            lower_objectives.push(if verify_upper_bound { -value } else { value });
        }
    }
    if lower_objectives.len() != objective_len {
        return Err(AtomicCudaMarginStepRefusal::InvalidTopKObjective);
    }
    deadline_check(deadline)?;
    Ok(AtomicCudaMarginTopKPlan {
        row_indices,
        lower_objectives,
    })
}

/// One indivisible bracket item. `bounds` is constructed only by evaluating
/// `alpha_state` through a complete [`AtomicCudaRowsRequest`]; neither field is
/// exposed separately to the selector.
#[derive(Debug)]
struct AtomicCudaMarginCertifiedPair {
    ordinal: usize,
    learning_rate: f32,
    score: f32,
    bounds: Box<BoundedTensor>,
    alpha_state: Box<GraphAlphaState>,
}

#[derive(Clone, Debug, PartialEq)]
struct AtomicCudaMarginIterationsSummary {
    binding_rows: Box<[usize]>,
    attempted_iterations: usize,
    accepted_iterations: usize,
    /// Exact child-gate value read inside the typed multi-iteration route.
    multiplicative_weights_requested: bool,
    /// True only when an MW plan reached `evaluate_step`; this does not claim
    /// that the gradient/candidate transaction completed.
    multiplicative_weights_plan_dispatched: bool,
    /// True only when at least one complete candidate pair returned from an MW
    /// plan that used a post-update distribution. A completed first uniform
    /// proposal alone is intentionally not called adaptive MW.
    multiplicative_weights_effective: bool,
    /// Number of dispatched plans whose complete candidate pair returned.
    completed_proposals: usize,
    /// Number of full-row plans dispatched with a post-update row-player
    /// distribution. A tied update can remain numerically uniform, but it is
    /// still distinct from the unconditionally uniform first plan.
    adaptive_plan_dispatches: usize,
    /// Common planned `num_specs` handed to `evaluate_step`, or `None` when no
    /// gradient plan was dispatched. It is plan evidence, not a GPU-success
    /// claim; `completed_proposals` authenticates completed transactions.
    gradient_plan_num_specs: Option<usize>,
    /// Number of exact-C rows resident in the gradient player.
    gradient_row_count: usize,
    initial_score: f32,
    final_score: f32,
    stop_refusal: Option<AtomicCudaMarginStepRefusal>,
}

/// The multi-iteration selector can return only the original complete bounds
/// allocation or one complete evaluated bounds/state pair. There is no shape
/// capable of representing row-wise bounds from multiple alpha states.
#[derive(Debug)]
enum AtomicCudaMarginIterationsChoice {
    Alpha0 {
        bounds: Box<BoundedTensor>,
        summary: AtomicCudaMarginIterationsSummary,
    },
    Candidate {
        pair: AtomicCudaMarginCertifiedPair,
        summary: AtomicCudaMarginIterationsSummary,
    },
}

enum AtomicCudaMarginCurrentState<'a> {
    Alpha0(&'a GraphAlphaState),
    Accepted(Box<GraphAlphaState>),
}

impl AtomicCudaMarginCurrentState<'_> {
    fn as_ref(&self) -> &GraphAlphaState {
        match self {
            Self::Alpha0(state) => state,
            Self::Accepted(state) => state,
        }
    }
}

fn map_iteration_work_refusal(refusal: AtomicCudaMarginStepRefusal) -> AtomicCudaMarginStepRefusal {
    if margin_refusal_observed_deadline(refusal) {
        AtomicCudaMarginStepRefusal::WorkDeadlineExceeded
    } else {
        refusal
    }
}

/// CPU-injectable exact-`C` multi-iteration selector. `evaluate_step` is the
/// only backend seam: production computes a joint gradient at `current_state`,
/// applies the supplied scheduled Adam parameters, then evaluates the complete
/// `C` matrix. Tests inject deterministic pairs to mutation-guard rebinding,
/// strict acceptance, deadline separation, and whole-pair transport without a
/// CUDA device.
#[allow(clippy::too_many_arguments)]
fn run_complete_multi_iterations_with_clock<F, N>(
    alpha0: &GraphAlphaState,
    alpha0_bounds: Box<BoundedTensor>,
    alpha0_score: MarginScore,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    verify_upper_bound: bool,
    base_adam: AdamParams,
    policy: AtomicCudaMarginIterationsPolicy,
    multiplicative_weights: bool,
    mut evaluate_step: F,
    mut now: N,
) -> AtomicCudaMarginIterationsChoice
where
    F: FnMut(
        &GraphAlphaState,
        &AtomicCudaMarginIterationPlan,
        AdamParams,
        Instant,
    )
        -> Result<(Box<BoundedTensor>, Box<GraphAlphaState>), AtomicCudaMarginStepRefusal>,
    N: FnMut() -> Instant,
{
    let initial_score = alpha0_score.hinge;
    let mut current_score = alpha0_score;
    let mut current_bounds = alpha0_bounds;
    let mut current_state = AtomicCudaMarginCurrentState::Alpha0(alpha0);
    let mut binding_rows = Vec::with_capacity(policy.iterations);
    let mut attempted_iterations = 0usize;
    let mut accepted_iterations = 0usize;
    let mut completed_proposals = 0usize;
    let mut adaptive_plan_dispatches = 0usize;
    let mut adaptive_completed_proposals = 0usize;
    let mut last_accepted_ordinal = 0usize;
    let mut last_accepted_learning_rate = base_adam.learning_rate;
    let mut stop_refusal = None;
    let gradient_row_count = spec_matrix.nrows();
    let expected_gradient_num_specs = if multiplicative_weights {
        gradient_row_count
    } else {
        1
    };
    let mut gradient_plan_num_specs = None;
    // Lazily construct the full-C row player only after a binding row exists.
    // An already verified alpha0 therefore performs no seed allocation.
    let mut row_player = None;

    for offset in 0..policy.iterations {
        if now() >= policy.work_deadline {
            stop_refusal = Some(AtomicCudaMarginStepRefusal::WorkDeadlineExceeded);
            break;
        }
        let Some(binding_index) = current_score.binding.as_ref().map(|binding| binding.index)
        else {
            stop_refusal = Some(AtomicCudaMarginStepRefusal::NoUnverifiedBinding);
            break;
        };
        let adam = match policy.adam_for_offset(base_adam, offset) {
            Ok(adam) => adam,
            Err(refusal) => {
                stop_refusal = Some(refusal);
                break;
            }
        };
        let plan = if multiplicative_weights {
            if row_player.is_none() {
                row_player = match MultiplicativeWeightsRowPlayer::new(
                    spec_matrix,
                    verify_upper_bound,
                    policy.iterations,
                    policy.work_deadline,
                ) {
                    Ok(player) => Some(player),
                    Err(refusal) => {
                        stop_refusal = Some(map_iteration_work_refusal(refusal));
                        break;
                    }
                };
            }
            match row_player
                .as_mut()
                .expect("MW player initialized above")
                .objective_for(
                    &current_bounds,
                    thresholds,
                    binding_index,
                    policy.work_deadline,
                ) {
                Ok(plan) => plan,
                Err(refusal) => {
                    stop_refusal = Some(map_iteration_work_refusal(refusal));
                    break;
                }
            }
        } else {
            let binding = current_score
                .binding
                .clone()
                .expect("binding index was checked above");
            AtomicCudaMarginIterationPlan {
                binding_row: binding.index,
                lower_objectives: binding.lower_objective,
                num_specs: 1,
                adaptive_weights: false,
            }
        };
        if plan.num_specs == 0 || plan.num_specs != expected_gradient_num_specs {
            stop_refusal = Some(if multiplicative_weights {
                AtomicCudaMarginStepRefusal::InvalidMultiplicativeWeights
            } else {
                AtomicCudaMarginStepRefusal::JointMapping
            });
            break;
        }
        let expected_adaptive =
            multiplicative_weights && gradient_row_count > 1 && attempted_iterations > 0;
        if plan.adaptive_weights != expected_adaptive {
            stop_refusal = Some(if multiplicative_weights {
                AtomicCudaMarginStepRefusal::InvalidMultiplicativeWeights
            } else {
                AtomicCudaMarginStepRefusal::JointMapping
            });
            break;
        }
        match gradient_plan_num_specs {
            None => gradient_plan_num_specs = Some(plan.num_specs),
            Some(previous) if previous == plan.num_specs => {}
            Some(_) => {
                stop_refusal = Some(AtomicCudaMarginStepRefusal::JointMapping);
                break;
            }
        }
        binding_rows.push(plan.binding_row);
        attempted_iterations += 1;
        if plan.adaptive_weights {
            adaptive_plan_dispatches += 1;
        }

        let (candidate_bounds, candidate_state) =
            match evaluate_step(current_state.as_ref(), &plan, adam, policy.work_deadline) {
                Ok(pair) => pair,
                Err(refusal) => {
                    stop_refusal = Some(map_iteration_work_refusal(refusal));
                    break;
                }
            };
        completed_proposals += 1;
        if plan.adaptive_weights {
            adaptive_completed_proposals += 1;
        }
        if now() >= policy.work_deadline {
            drop((candidate_bounds, candidate_state));
            stop_refusal = Some(AtomicCudaMarginStepRefusal::WorkDeadlineExceeded);
            break;
        }
        let candidate_score = match score_complete_c(
            &candidate_bounds,
            spec_matrix,
            thresholds,
            verify_upper_bound,
            policy.work_deadline,
        ) {
            Ok(score) => score,
            Err(refusal) => {
                drop((candidate_bounds, candidate_state));
                stop_refusal = Some(map_iteration_work_refusal(refusal));
                break;
            }
        };
        if now() >= policy.work_deadline {
            drop((candidate_bounds, candidate_state));
            stop_refusal = Some(AtomicCudaMarginStepRefusal::WorkDeadlineExceeded);
            break;
        }

        // Strict finite full-C hinge selection. `score_complete_c` validated
        // every row and both endpoints; the whole pair moves together or both
        // candidate allocations are discarded.
        if candidate_score.hinge > current_score.hinge {
            current_bounds = candidate_bounds;
            current_state = AtomicCudaMarginCurrentState::Accepted(candidate_state);
            current_score = candidate_score;
            accepted_iterations += 1;
            last_accepted_ordinal = offset;
            last_accepted_learning_rate = adam.learning_rate;
        } else {
            drop((candidate_bounds, candidate_state));
        }
    }

    let multiplicative_weights_plan_dispatched = multiplicative_weights
        && attempted_iterations > 0
        && gradient_plan_num_specs == Some(gradient_row_count);
    let multiplicative_weights_effective =
        multiplicative_weights_plan_dispatched && adaptive_completed_proposals > 0;
    let summary = AtomicCudaMarginIterationsSummary {
        binding_rows: binding_rows.into_boxed_slice(),
        attempted_iterations,
        accepted_iterations,
        multiplicative_weights_requested: multiplicative_weights,
        multiplicative_weights_plan_dispatched,
        multiplicative_weights_effective,
        completed_proposals,
        adaptive_plan_dispatches,
        gradient_plan_num_specs,
        gradient_row_count,
        initial_score,
        final_score: current_score.hinge,
        stop_refusal,
    };
    match current_state {
        AtomicCudaMarginCurrentState::Alpha0(_) => AtomicCudaMarginIterationsChoice::Alpha0 {
            bounds: current_bounds,
            summary,
        },
        AtomicCudaMarginCurrentState::Accepted(alpha_state) => {
            AtomicCudaMarginIterationsChoice::Candidate {
                pair: AtomicCudaMarginCertifiedPair {
                    ordinal: last_accepted_ordinal,
                    learning_rate: last_accepted_learning_rate,
                    score: current_score.hinge,
                    bounds: current_bounds,
                    alpha_state,
                },
                summary,
            }
        }
    }
}

#[derive(Debug)]
struct AtomicCudaMarginPreparedCandidate {
    ordinal: usize,
    learning_rate: f32,
    alpha_state: Box<GraphAlphaState>,
}

#[derive(Debug)]
enum AtomicCudaMarginBracketChoice {
    Alpha0 { best_candidate_score: f32 },
    Candidate(AtomicCudaMarginCertifiedPair),
}

#[derive(Debug)]
enum AtomicCudaMarginTopKChoice {
    Alpha0 {
        candidate_score: f32,
        row_indices: Box<[usize]>,
    },
    Candidate {
        pair: AtomicCudaMarginCertifiedPair,
        row_indices: Box<[usize]>,
    },
}

fn admit_topk_choice_with_clock<F>(
    choice: AtomicCudaMarginTopKChoice,
    work_deadline: Instant,
    mut now: F,
) -> Result<AtomicCudaMarginTopKChoice, AtomicCudaMarginStepRefusal>
where
    F: FnMut() -> Instant,
{
    if now() >= work_deadline {
        drop(choice);
        Err(AtomicCudaMarginStepRefusal::DeadlineExceeded)
    } else {
        Ok(choice)
    }
}

fn score_complete_c(
    bounds: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    verify_upper_bound: bool,
    deadline: Instant,
) -> Result<MarginScore, AtomicCudaMarginStepRefusal> {
    deadline_check(deadline)?;
    let rows = spec_matrix.nrows();
    if thresholds.len() != rows {
        return Err(AtomicCudaMarginStepRefusal::ThresholdShape);
    }
    if bounds.shape() != [rows] {
        return Err(AtomicCudaMarginStepRefusal::BoundsShape);
    }

    let mut hinge = 0.0f64;
    let mut binding: Option<BindingRow> = None;
    let mut host_work = 0usize;
    for (row, ((&lower, &upper), &threshold)) in bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .zip(thresholds)
        .enumerate()
    {
        deadline_host_work(deadline, &mut host_work)?;
        if !threshold.is_finite() {
            return Err(AtomicCudaMarginStepRefusal::NonFiniteThreshold);
        }
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(AtomicCudaMarginStepRefusal::BoundsNonFiniteOrInverted);
        }
        let slack = if verify_upper_bound {
            threshold - upper
        } else {
            lower - threshold
        };
        if !slack.is_finite() {
            return Err(AtomicCudaMarginStepRefusal::ScoreNonFinite);
        }
        hinge += f64::from(slack).min(0.0);
        if slack <= 0.0
            && !binding
                .as_ref()
                .is_some_and(|current| slack >= current.slack)
        {
            let source = spec_matrix.row(row);
            let mut lower_objective = Vec::with_capacity(source.len());
            for &value in source {
                deadline_host_work(deadline, &mut host_work)?;
                if !value.is_finite() {
                    return Err(AtomicCudaMarginStepRefusal::ScoreNonFinite);
                }
                lower_objective.push(if verify_upper_bound { -value } else { value });
            }
            binding = Some(BindingRow {
                index: row,
                slack,
                lower_objective,
            });
        }
    }
    let hinge = hinge as f32;
    if !hinge.is_finite() {
        return Err(AtomicCudaMarginStepRefusal::ScoreNonFinite);
    }
    deadline_check(deadline)?;
    Ok(MarginScore { hinge, binding })
}

fn validate_complete_relu_candidate(
    alpha0: &GraphAlphaState,
    candidate: &GraphAlphaState,
    relu_names: &[String],
) -> Result<(), AtomicCudaMarginStepRefusal> {
    validate_relu_alignment(candidate, relu_names)
        .map_err(|_| AtomicCudaMarginStepRefusal::MalformedLearningRateCandidate)?;
    if !candidate.monotone_s_shaped_alphas.is_empty()
        || !candidate.sqrt_alphas.is_empty()
        || !candidate.reciprocal_alphas.is_empty()
        || candidate.spatial_shapes != alpha0.spatial_shapes
        || [
            candidate.alphas.len(),
            candidate.alphas_upper.len(),
            candidate.unstable_mask.len(),
            candidate.velocity.len(),
            candidate.adam_m.len(),
            candidate.adam_v.len(),
            candidate.velocity_upper.len(),
            candidate.adam_m_upper.len(),
            candidate.adam_v_upper.len(),
        ]
        .iter()
        .any(|&entries| entries != relu_names.len())
    {
        return Err(AtomicCudaMarginStepRefusal::MalformedLearningRateCandidate);
    }

    for name in relu_names {
        let expected = alpha0
            .relu_len(name)
            .ok_or(AtomicCudaMarginStepRefusal::MalformedLearningRateCandidate)?;
        if candidate.relu_len(name) != Some(expected)
            || candidate.unstable_mask.get(name) != alpha0.unstable_mask.get(name)
        {
            return Err(AtomicCudaMarginStepRefusal::MalformedLearningRateCandidate);
        }
        for values in [candidate.alphas.get(name), candidate.alphas_upper.get(name)] {
            let values =
                values.ok_or(AtomicCudaMarginStepRefusal::MalformedLearningRateCandidate)?;
            if values.len() != expected
                || values
                    .iter()
                    .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
            {
                return Err(AtomicCudaMarginStepRefusal::MalformedLearningRateCandidate);
            }
        }
        for values in [
            candidate.velocity.get(name),
            candidate.adam_m.get(name),
            candidate.adam_v.get(name),
            candidate.velocity_upper.get(name),
            candidate.adam_m_upper.get(name),
            candidate.adam_v_upper.get(name),
        ] {
            let values =
                values.ok_or(AtomicCudaMarginStepRefusal::MalformedLearningRateCandidate)?;
            if values.len() != expected || values.iter().any(|value| !value.is_finite()) {
                return Err(AtomicCudaMarginStepRefusal::MalformedLearningRateCandidate);
            }
        }
    }
    Ok(())
}

fn prepare_complete_lr_bracket(
    alpha0: &GraphAlphaState,
    relu_names: &[String],
    gradients: &[Vec<f32>],
    adam: AdamParams,
    policy: AtomicCudaMarginBracketPolicy,
) -> Result<Vec<AtomicCudaMarginPreparedCandidate>, AtomicCudaMarginStepRefusal> {
    let mut prepared = Vec::with_capacity(policy.candidate_lrs.len());
    for (ordinal, &learning_rate) in policy.candidate_lrs.iter().enumerate() {
        deadline_check(policy.work_deadline)?;
        let candidate_adam = AdamParams {
            learning_rate,
            ..adam
        };
        let alpha_state = apply_checked_adam_step(
            alpha0,
            relu_names,
            gradients,
            candidate_adam,
            policy.work_deadline,
        )?;
        validate_complete_relu_candidate(alpha0, &alpha_state, relu_names)?;
        prepared.push(AtomicCudaMarginPreparedCandidate {
            ordinal,
            learning_rate,
            alpha_state: Box::new(alpha_state),
        });
    }
    deadline_check(policy.work_deadline)?;
    Ok(prepared)
}

#[allow(clippy::too_many_arguments)]
fn select_complete_lr_bracket_with_clock<F>(
    alpha0: &GraphAlphaState,
    alpha0_score: f32,
    relu_names: &[String],
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    verify_upper_bound: bool,
    policy: AtomicCudaMarginBracketPolicy,
    candidates: Vec<AtomicCudaMarginCertifiedPair>,
    mut now: F,
) -> Result<AtomicCudaMarginBracketChoice, AtomicCudaMarginStepRefusal>
where
    F: FnMut() -> Instant,
{
    let mut candidates = candidates;
    let decision = (|| {
        if now() >= policy.work_deadline {
            return Err(AtomicCudaMarginStepRefusal::DeadlineExceeded);
        }
        if !alpha0_score.is_finite() {
            return Err(AtomicCudaMarginStepRefusal::ScoreNonFinite);
        }
        if candidates.len() != policy.candidate_lrs.len() {
            return Err(AtomicCudaMarginStepRefusal::IncompleteLearningRateBracket);
        }

        let mut best_ordinal = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        let mut best_tied = false;
        for (expected_ordinal, candidate) in candidates.iter().enumerate() {
            if now() >= policy.work_deadline {
                return Err(AtomicCudaMarginStepRefusal::DeadlineExceeded);
            }
            if candidate.ordinal != expected_ordinal
                || candidate.learning_rate.to_bits()
                    != policy.candidate_lrs[expected_ordinal].to_bits()
                || candidate.bounds.shape() != [spec_matrix.nrows()]
            {
                return Err(AtomicCudaMarginStepRefusal::MalformedLearningRateCandidate);
            }
            validate_complete_relu_candidate(alpha0, &candidate.alpha_state, relu_names)?;
            let rescored = score_complete_c(
                &candidate.bounds,
                spec_matrix,
                thresholds,
                verify_upper_bound,
                policy.work_deadline,
            )?
            .hinge;
            if !candidate.score.is_finite() || candidate.score.to_bits() != rescored.to_bits() {
                return Err(AtomicCudaMarginStepRefusal::MalformedLearningRateCandidate);
            }

            if rescored > best_score {
                best_ordinal = expected_ordinal;
                best_score = rescored;
                best_tied = false;
            } else if rescored == best_score {
                best_tied = true;
            }
        }

        Ok((best_ordinal, best_score, best_tied))
    })();
    let (best_ordinal, best_score, best_tied) = match decision {
        Ok(decision) => decision,
        Err(refusal) => {
            return Err(bracket_refusal_after_drop_with_clock(
                candidates, policy, refusal, &mut now,
            ));
        }
    };

    if best_tied || best_score <= alpha0_score {
        // Every candidate is rejected on this branch. Destroy all large
        // candidate state before the publication-authority poll.
        drop(candidates);
        poll_bracket_authority_with_clock(policy, &mut now)?;
        Ok(AtomicCudaMarginBracketChoice::Alpha0 {
            best_candidate_score: best_score,
        })
    } else if best_ordinal >= candidates.len() {
        Err(bracket_refusal_after_drop_with_clock(
            candidates,
            policy,
            AtomicCudaMarginStepRefusal::IncompleteLearningRateBracket,
            &mut now,
        ))
    } else {
        let selected = candidates.swap_remove(best_ordinal);
        // Destroy all unselected pairs before the publication-authority poll.
        // Moving a surviving selected pair afterward is allocation-free.
        drop(candidates);
        if let Err(deadline) = poll_bracket_authority_with_clock(policy, &mut now) {
            // The selected pair will not be published. Its destructor is now
            // deadline work too, so poll again only after it has been dropped.
            return Err(bracket_refusal_after_drop_with_clock(
                selected, policy, deadline, &mut now,
            ));
        }
        Ok(AtomicCudaMarginBracketChoice::Candidate(selected))
    }
}

/// A post-alpha0 refusal is committed: the complete alpha0 pair remains
/// authoritative and ordinary CROWN must not run.
pub(crate) enum AtomicCudaMarginStepCommit {
    /// The bracket completed enough work to commit the atomic Stage-A route,
    /// but scratch/candidate teardown reached the absolute publication
    /// deadline. No bounds or alpha state remain publishable.
    DeadlineExceeded,
    Alpha0Retained {
        bounds: Box<BoundedTensor>,
        refusal: AtomicCudaMarginStepRefusal,
    },
    Alpha0Selected {
        bounds: Box<BoundedTensor>,
        binding_row: usize,
        alpha0_score: f32,
        alpha1_score: f32,
    },
    Alpha1Selected {
        bounds: Box<BoundedTensor>,
        alpha_state: Box<GraphAlphaState>,
        binding_row: usize,
        alpha0_score: f32,
        alpha1_score: f32,
    },
    TopKAlpha0Selected {
        bounds: Box<BoundedTensor>,
        row_indices: Box<[usize]>,
        alpha0_score: f32,
        alpha1_score: f32,
    },
    TopKAlpha1Selected {
        bounds: Box<BoundedTensor>,
        alpha_state: Box<GraphAlphaState>,
        row_indices: Box<[usize]>,
        alpha0_score: f32,
        alpha1_score: f32,
    },
    MultiAlpha0Selected {
        bounds: Box<BoundedTensor>,
        binding_rows: Box<[usize]>,
        attempted_iterations: usize,
        multiplicative_weights_requested: bool,
        multiplicative_weights_plan_dispatched: bool,
        multiplicative_weights_effective: bool,
        completed_proposals: usize,
        adaptive_plan_dispatches: usize,
        gradient_plan_num_specs: Option<usize>,
        gradient_row_count: usize,
        initial_score: f32,
        final_score: f32,
        stop_refusal: Option<AtomicCudaMarginStepRefusal>,
    },
    MultiAlphaSelected {
        bounds: Box<BoundedTensor>,
        alpha_state: Box<GraphAlphaState>,
        binding_rows: Box<[usize]>,
        attempted_iterations: usize,
        accepted_iterations: usize,
        multiplicative_weights_requested: bool,
        multiplicative_weights_plan_dispatched: bool,
        multiplicative_weights_effective: bool,
        completed_proposals: usize,
        adaptive_plan_dispatches: usize,
        gradient_plan_num_specs: Option<usize>,
        gradient_row_count: usize,
        initial_score: f32,
        final_score: f32,
        stop_refusal: Option<AtomicCudaMarginStepRefusal>,
    },
}

fn deadline_commit_after_drop_with_clock<T, F>(
    owned_candidate_state: T,
    now: &mut F,
) -> AtomicCudaMarginStepCommit
where
    F: FnMut() -> Instant,
{
    drop(owned_candidate_state);
    // This observation is deliberately after teardown. Once an expiry has
    // already been observed, the outcome remains DeadlineExceeded even if an
    // injected test clock is non-monotonic.
    let _ = now();
    AtomicCudaMarginStepCommit::DeadlineExceeded
}

fn margin_refusal_observed_deadline(refusal: AtomicCudaMarginStepRefusal) -> bool {
    matches!(
        refusal,
        AtomicCudaMarginStepRefusal::DeadlineExceeded
            | AtomicCudaMarginStepRefusal::Alpha0Rows(AtomicCudaRowsRefusal::DeadlineExceeded)
            | AtomicCudaMarginStepRefusal::Alpha1Rows(AtomicCudaRowsRefusal::DeadlineExceeded)
    )
}

fn finalize_alpha0_retained_with_clock<T, F>(
    owned_scratch: T,
    alpha0_bounds: Box<BoundedTensor>,
    refusal: AtomicCudaMarginStepRefusal,
    authority_deadline: Instant,
    mut now: F,
) -> AtomicCudaMarginStepCommit
where
    F: FnMut() -> Instant,
{
    drop(owned_scratch);
    if margin_refusal_observed_deadline(refusal) || now() >= authority_deadline {
        deadline_commit_after_drop_with_clock(alpha0_bounds, &mut now)
    } else {
        AtomicCudaMarginStepCommit::Alpha0Retained {
            bounds: alpha0_bounds,
            refusal,
        }
    }
}

fn finalize_complete_lr_bracket_with_clock<T, F>(
    owned_scratch: T,
    alpha0_bounds: Box<BoundedTensor>,
    binding_row: usize,
    alpha0_score: f32,
    result: Result<AtomicCudaMarginBracketChoice, AtomicCudaMarginStepRefusal>,
    authority_deadline: Instant,
    mut now: F,
) -> AtomicCudaMarginStepCommit
where
    F: FnMut() -> Instant,
{
    // The joint gradient and binding-score workspace can be full-alpha-sized.
    // They are never part of a published result, so destroy them before the
    // outermost publication-authority decision.
    drop(owned_scratch);

    match result {
        Err(refusal) => finalize_alpha0_retained_with_clock(
            (),
            alpha0_bounds,
            refusal,
            authority_deadline,
            &mut now,
        ),
        Ok(AtomicCudaMarginBracketChoice::Alpha0 {
            best_candidate_score,
        }) => {
            if now() >= authority_deadline {
                deadline_commit_after_drop_with_clock(alpha0_bounds, &mut now)
            } else {
                AtomicCudaMarginStepCommit::Alpha0Selected {
                    bounds: alpha0_bounds,
                    binding_row,
                    alpha0_score,
                    alpha1_score: best_candidate_score,
                }
            }
        }
        Ok(AtomicCudaMarginBracketChoice::Candidate(candidate)) => {
            // Alpha0 is not part of a candidate publication. Destroy it inside
            // the 25 ms reserve, then make the outermost authority decision.
            drop(alpha0_bounds);
            if now() >= authority_deadline {
                deadline_commit_after_drop_with_clock(candidate, &mut now)
            } else {
                AtomicCudaMarginStepCommit::Alpha1Selected {
                    bounds: candidate.bounds,
                    alpha_state: candidate.alpha_state,
                    binding_row,
                    alpha0_score,
                    alpha1_score: candidate.score,
                }
            }
        }
    }
}

fn finalize_topk_with_clock<T, F>(
    owned_scratch: T,
    alpha0_bounds: Box<BoundedTensor>,
    alpha0_score: f32,
    result: Result<AtomicCudaMarginTopKChoice, AtomicCudaMarginStepRefusal>,
    authority_deadline: Instant,
    mut now: F,
) -> AtomicCudaMarginStepCommit
where
    F: FnMut() -> Instant,
{
    drop(owned_scratch);
    match result {
        Err(refusal) => finalize_alpha0_retained_with_clock(
            (),
            alpha0_bounds,
            refusal,
            authority_deadline,
            &mut now,
        ),
        Ok(AtomicCudaMarginTopKChoice::Alpha0 {
            candidate_score,
            row_indices,
        }) => {
            if now() >= authority_deadline {
                deadline_commit_after_drop_with_clock((alpha0_bounds, row_indices), &mut now)
            } else {
                AtomicCudaMarginStepCommit::TopKAlpha0Selected {
                    bounds: alpha0_bounds,
                    row_indices,
                    alpha0_score,
                    alpha1_score: candidate_score,
                }
            }
        }
        Ok(AtomicCudaMarginTopKChoice::Candidate { pair, row_indices }) => {
            drop(alpha0_bounds);
            if now() >= authority_deadline {
                deadline_commit_after_drop_with_clock((pair, row_indices), &mut now)
            } else {
                AtomicCudaMarginStepCommit::TopKAlpha1Selected {
                    bounds: pair.bounds,
                    alpha_state: pair.alpha_state,
                    row_indices,
                    alpha0_score,
                    alpha1_score: pair.score,
                }
            }
        }
    }
}

fn finalize_multi_iterations_with_clock<T, F>(
    owned_scratch: T,
    choice: AtomicCudaMarginIterationsChoice,
    authority_deadline: Instant,
    mut now: F,
) -> AtomicCudaMarginStepCommit
where
    F: FnMut() -> Instant,
{
    drop(owned_scratch);
    if now() >= authority_deadline {
        return deadline_commit_after_drop_with_clock(choice, &mut now);
    }
    match choice {
        AtomicCudaMarginIterationsChoice::Alpha0 { bounds, summary } => {
            debug_assert_eq!(summary.accepted_iterations, 0);
            AtomicCudaMarginStepCommit::MultiAlpha0Selected {
                bounds,
                binding_rows: summary.binding_rows,
                attempted_iterations: summary.attempted_iterations,
                multiplicative_weights_requested: summary.multiplicative_weights_requested,
                multiplicative_weights_plan_dispatched: summary
                    .multiplicative_weights_plan_dispatched,
                multiplicative_weights_effective: summary.multiplicative_weights_effective,
                completed_proposals: summary.completed_proposals,
                adaptive_plan_dispatches: summary.adaptive_plan_dispatches,
                gradient_plan_num_specs: summary.gradient_plan_num_specs,
                gradient_row_count: summary.gradient_row_count,
                initial_score: summary.initial_score,
                final_score: summary.final_score,
                stop_refusal: summary.stop_refusal,
            }
        }
        AtomicCudaMarginIterationsChoice::Candidate { pair, summary } => {
            AtomicCudaMarginStepCommit::MultiAlphaSelected {
                bounds: pair.bounds,
                alpha_state: pair.alpha_state,
                binding_rows: summary.binding_rows,
                attempted_iterations: summary.attempted_iterations,
                accepted_iterations: summary.accepted_iterations,
                multiplicative_weights_requested: summary.multiplicative_weights_requested,
                multiplicative_weights_plan_dispatched: summary
                    .multiplicative_weights_plan_dispatched,
                multiplicative_weights_effective: summary.multiplicative_weights_effective,
                completed_proposals: summary.completed_proposals,
                adaptive_plan_dispatches: summary.adaptive_plan_dispatches,
                gradient_plan_num_specs: summary.gradient_plan_num_specs,
                gradient_row_count: summary.gradient_row_count,
                initial_score: summary.initial_score,
                final_score: summary.final_score,
                stop_refusal: summary.stop_refusal,
            }
        }
    }
}

/// Only an alpha0 refusal before its backend commitment permits the parent
/// Stage-A caller to preserve its historical fallback policy.
// Boxing `Committed` (144B vs the 32B refusal) is not worth it here: the enum
// is returned once per Stage-A step, never in a tight loop, and the Stage-A
// consumers destructure it through nested patterns
// (`MarginStep(Committed(Alpha0Retained { .. }))` in
// `beta_crown/engine/graph/multi_objective/root.rs`) that a `Box` payload
// cannot be matched through on stable. The indirection would buy an
// allocation and a large pattern rewrite for no measured gain.
#[allow(clippy::large_enum_variant)]
pub(crate) enum AtomicCudaMarginStepOutcome {
    RefusedBeforeCommit { refusal: AtomicCudaRowsRefusal },
    Committed(AtomicCudaMarginStepCommit),
}

/// Complete selected-row request for one evaluated margin-alpha proposal.
pub(crate) struct AtomicCudaMarginStepRequest<'a> {
    graph: &'a GraphNetwork,
    input: &'a BoundedTensor,
    target_node: &'a str,
    node_bounds: &'a HashMap<String, BoundedTensor>,
    alpha0: &'a GraphAlphaState,
    spec_matrix: &'a Array2<f32>,
    reference: &'a BoundedTensor,
    thresholds: &'a [f32],
    verify_upper_bound: bool,
    all_rows_required: bool,
    analytic_gradient: bool,
    adam_optimizer: bool,
    adam: AdamParams,
    multi_iterations: usize,
    learning_rate_decay: f32,
    bracket_base_learning_rate: f32,
    deadline: Instant,
}

impl<'a> AtomicCudaMarginStepRequest<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        graph: &'a GraphNetwork,
        input: &'a BoundedTensor,
        target_node: &'a str,
        node_bounds: &'a HashMap<String, BoundedTensor>,
        alpha0: &'a GraphAlphaState,
        spec_matrix: &'a Array2<f32>,
        reference: &'a BoundedTensor,
        thresholds: &'a [f32],
        verify_upper_bound: bool,
        all_rows_required: bool,
        analytic_gradient: bool,
        adam_optimizer: bool,
        adam: AdamParams,
        multi_iterations: usize,
        learning_rate_decay: f32,
        bracket_base_learning_rate: f32,
        deadline: Instant,
    ) -> Self {
        Self {
            graph,
            input,
            target_node,
            node_bounds,
            alpha0,
            spec_matrix,
            reference,
            thresholds,
            verify_upper_bound,
            all_rows_required,
            analytic_gradient,
            adam_optimizer,
            adam,
            multi_iterations,
            learning_rate_decay,
            bracket_base_learning_rate,
            deadline,
        }
    }

    pub(crate) fn run(self) -> AtomicCudaMarginStepOutcome {
        let alpha0_bounds = match AtomicCudaRowsRequest::new(
            self.graph,
            self.input,
            self.target_node,
            self.node_bounds,
            self.alpha0,
            self.spec_matrix,
            self.reference,
            self.deadline,
        )
        .run()
        {
            AtomicCudaRowsOutcome::RefusedBeforeCommit { refusal } => {
                return AtomicCudaMarginStepOutcome::RefusedBeforeCommit { refusal };
            }
            AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::ReferenceRetained {
                bounds,
                refusal,
            }) => {
                return AtomicCudaMarginStepOutcome::Committed(
                    finalize_alpha0_retained_with_clock(
                        (),
                        bounds,
                        AtomicCudaMarginStepRefusal::Alpha0Rows(refusal),
                        self.deadline,
                        Instant::now,
                    ),
                );
            }
            AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::CudaIntersection(bounds)) => {
                bounds
            }
            AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::DeadlineExceeded) => {
                return AtomicCudaMarginStepOutcome::Committed(
                    AtomicCudaMarginStepCommit::DeadlineExceeded,
                );
            }
        };

        if !self.all_rows_required {
            return AtomicCudaMarginStepOutcome::Committed(finalize_alpha0_retained_with_clock(
                (),
                alpha0_bounds,
                AtomicCudaMarginStepRefusal::ObjectiveMode,
                self.deadline,
                Instant::now,
            ));
        }
        if !self.analytic_gradient {
            return AtomicCudaMarginStepOutcome::Committed(finalize_alpha0_retained_with_clock(
                (),
                alpha0_bounds,
                AtomicCudaMarginStepRefusal::GradientPolicy,
                self.deadline,
                Instant::now,
            ));
        }
        if !self.adam_optimizer {
            return AtomicCudaMarginStepOutcome::Committed(finalize_alpha0_retained_with_clock(
                (),
                alpha0_bounds,
                AtomicCudaMarginStepRefusal::OptimizerPolicy,
                self.deadline,
                Instant::now,
            ));
        }
        if Instant::now() >= self.deadline {
            return AtomicCudaMarginStepOutcome::Committed(finalize_alpha0_retained_with_clock(
                (),
                alpha0_bounds,
                AtomicCudaMarginStepRefusal::DeadlineExceeded,
                self.deadline,
                Instant::now,
            ));
        }
        let alpha0_score = match score_complete_c(
            &alpha0_bounds,
            self.spec_matrix,
            self.thresholds,
            self.verify_upper_bound,
            self.deadline,
        ) {
            Ok(score) => score,
            Err(refusal) => {
                return AtomicCudaMarginStepOutcome::Committed(
                    finalize_alpha0_retained_with_clock(
                        (),
                        alpha0_bounds,
                        refusal,
                        self.deadline,
                        Instant::now,
                    ),
                );
            }
        };
        let topk_enabled = root_alpha_cuda_margin_topk_enabled();
        let lr_bracket_enabled = root_alpha_cuda_margin_lr_bracket_enabled();
        if let Err(refusal) = multi_iteration_search_policy_is_exclusive(
            self.multi_iterations,
            topk_enabled,
            lr_bracket_enabled,
        ) {
            return AtomicCudaMarginStepOutcome::Committed(finalize_alpha0_retained_with_clock(
                alpha0_score,
                alpha0_bounds,
                refusal,
                self.deadline,
                Instant::now,
            ));
        }
        if self.multi_iterations > 0 {
            if alpha_state_has_spec_axis(self.alpha0) {
                return AtomicCudaMarginStepOutcome::Committed(
                    finalize_alpha0_retained_with_clock(
                        alpha0_score,
                        alpha0_bounds,
                        AtomicCudaMarginStepRefusal::SpecAxisAlphaState,
                        self.deadline,
                        Instant::now,
                    ),
                );
            }
            let policy = match AtomicCudaMarginIterationsPolicy::new(
                self.multi_iterations,
                self.adam,
                self.learning_rate_decay,
                self.deadline,
            ) {
                Ok(policy) => policy,
                Err(refusal) => {
                    return AtomicCudaMarginStepOutcome::Committed(
                        finalize_alpha0_retained_with_clock(
                            alpha0_score,
                            alpha0_bounds,
                            refusal,
                            self.deadline,
                            Instant::now,
                        ),
                    );
                }
            };
            // Read the MW child only inside the armed typed-iteration branch.
            // With typed iterations dark there is no environment lookup and no
            // row-player allocation.
            let multiplicative_weights = root_alpha_cuda_margin_mw_enabled(self.multi_iterations);
            let choice = self.run_complete_multi_iterations(
                alpha0_bounds,
                alpha0_score,
                policy,
                multiplicative_weights,
            );
            return AtomicCudaMarginStepOutcome::Committed(finalize_multi_iterations_with_clock(
                (),
                choice,
                policy.authority_deadline,
                Instant::now,
            ));
        }

        let Some(binding) = alpha0_score.binding.as_ref() else {
            return AtomicCudaMarginStepOutcome::Committed(finalize_alpha0_retained_with_clock(
                alpha0_score,
                alpha0_bounds,
                AtomicCudaMarginStepRefusal::NoUnverifiedBinding,
                self.deadline,
                Instant::now,
            ));
        };
        let binding_index = binding.index;
        let alpha0_hinge = alpha0_score.hinge;

        let search_policy = match select_margin_search_policy(topk_enabled, lr_bracket_enabled) {
            Ok(policy) => policy,
            Err(refusal) => {
                return AtomicCudaMarginStepOutcome::Committed(finalize_topk_with_clock(
                    alpha0_score,
                    alpha0_bounds,
                    alpha0_hinge,
                    Err(refusal),
                    self.deadline,
                    Instant::now,
                ));
            }
        };

        if matches!(search_policy, AtomicCudaMarginSearchPolicy::TopK) {
            let policy = match AtomicCudaMarginTopKPolicy::new(
                self.adam,
                self.bracket_base_learning_rate,
                self.deadline,
            ) {
                Ok(policy) => policy,
                Err(refusal) => {
                    return AtomicCudaMarginStepOutcome::Committed(finalize_topk_with_clock(
                        alpha0_score,
                        alpha0_bounds,
                        alpha0_hinge,
                        Err(refusal),
                        self.deadline,
                        Instant::now,
                    ));
                }
            };
            let plan = match prepare_topk_objective(
                &alpha0_bounds,
                self.spec_matrix,
                self.thresholds,
                self.verify_upper_bound,
                policy.work_deadline,
            ) {
                Ok(plan) => plan,
                Err(refusal) => {
                    return AtomicCudaMarginStepOutcome::Committed(finalize_topk_with_clock(
                        alpha0_score,
                        alpha0_bounds,
                        alpha0_hinge,
                        Err(refusal),
                        policy.authority_deadline,
                        Instant::now,
                    ));
                }
            };
            let gradients = match self.deadline_joint_gradients_for(
                self.alpha0,
                &plan.lower_objectives,
                plan.row_indices.len(),
                policy.work_deadline,
            ) {
                Ok(gradients) => gradients,
                Err(refusal) => {
                    return AtomicCudaMarginStepOutcome::Committed(finalize_topk_with_clock(
                        (plan, alpha0_score),
                        alpha0_bounds,
                        alpha0_hinge,
                        Err(refusal),
                        policy.authority_deadline,
                        Instant::now,
                    ));
                }
            };
            let result =
                self.run_complete_topk(&plan, alpha0_hinge, &gradients.0, &gradients.1, policy);
            return AtomicCudaMarginStepOutcome::Committed(finalize_topk_with_clock(
                (gradients, plan, alpha0_score),
                alpha0_bounds,
                alpha0_hinge,
                result,
                policy.authority_deadline,
                Instant::now,
            ));
        }

        if matches!(
            search_policy,
            AtomicCudaMarginSearchPolicy::LearningRateBracket
        ) {
            let policy = match AtomicCudaMarginBracketPolicy::new(
                self.adam,
                self.bracket_base_learning_rate,
                self.deadline,
            ) {
                Ok(policy) => policy,
                Err(refusal) => {
                    return AtomicCudaMarginStepOutcome::Committed(
                        finalize_complete_lr_bracket_with_clock(
                            alpha0_score,
                            alpha0_bounds,
                            binding_index,
                            alpha0_hinge,
                            Err(refusal),
                            self.deadline,
                            Instant::now,
                        ),
                    );
                }
            };
            let gradients = match self
                .deadline_joint_gradients(&binding.lower_objective, policy.work_deadline)
            {
                Ok(gradients) => gradients,
                Err(refusal) => {
                    return AtomicCudaMarginStepOutcome::Committed(
                        finalize_complete_lr_bracket_with_clock(
                            alpha0_score,
                            alpha0_bounds,
                            binding_index,
                            alpha0_hinge,
                            Err(refusal),
                            policy.authority_deadline,
                            Instant::now,
                        ),
                    );
                }
            };
            let result =
                self.run_complete_lr_bracket(&alpha0_score, &gradients.0, &gradients.1, policy);
            return AtomicCudaMarginStepOutcome::Committed(
                finalize_complete_lr_bracket_with_clock(
                    (gradients, alpha0_score),
                    alpha0_bounds,
                    binding_index,
                    alpha0_hinge,
                    result,
                    policy.authority_deadline,
                    Instant::now,
                ),
            );
        }

        let gradients = match self.deadline_joint_gradients(&binding.lower_objective, self.deadline)
        {
            Ok(gradients) => gradients,
            Err(refusal) => {
                return AtomicCudaMarginStepOutcome::Committed(
                    finalize_alpha0_retained_with_clock(
                        alpha0_score,
                        alpha0_bounds,
                        refusal,
                        self.deadline,
                        Instant::now,
                    ),
                );
            }
        };

        let alpha1 = match apply_checked_adam_step(
            self.alpha0,
            &gradients.0,
            &gradients.1,
            self.adam,
            self.deadline,
        ) {
            Ok(alpha) => alpha,
            Err(refusal) => {
                return AtomicCudaMarginStepOutcome::Committed(
                    finalize_alpha0_retained_with_clock(
                        (gradients, alpha0_score),
                        alpha0_bounds,
                        refusal,
                        self.deadline,
                        Instant::now,
                    ),
                );
            }
        };
        if Instant::now() >= self.deadline {
            return AtomicCudaMarginStepOutcome::Committed(finalize_alpha0_retained_with_clock(
                (alpha1, gradients, alpha0_score),
                alpha0_bounds,
                AtomicCudaMarginStepRefusal::DeadlineExceeded,
                self.deadline,
                Instant::now,
            ));
        }

        let alpha1_bounds = match AtomicCudaRowsRequest::new(
            self.graph,
            self.input,
            self.target_node,
            self.node_bounds,
            &alpha1,
            self.spec_matrix,
            self.reference,
            self.deadline,
        )
        .run()
        {
            AtomicCudaRowsOutcome::RefusedBeforeCommit { refusal } => {
                return AtomicCudaMarginStepOutcome::Committed(
                    finalize_alpha0_retained_with_clock(
                        (alpha1, gradients, alpha0_score),
                        alpha0_bounds,
                        AtomicCudaMarginStepRefusal::Alpha1Rows(refusal),
                        self.deadline,
                        Instant::now,
                    ),
                );
            }
            AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::ReferenceRetained {
                bounds,
                refusal,
            }) => {
                return AtomicCudaMarginStepOutcome::Committed(
                    finalize_alpha0_retained_with_clock(
                        (bounds, alpha1, gradients, alpha0_score),
                        alpha0_bounds,
                        AtomicCudaMarginStepRefusal::Alpha1Rows(refusal),
                        self.deadline,
                        Instant::now,
                    ),
                );
            }
            AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::CudaIntersection(bounds)) => {
                *bounds
            }
            AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::DeadlineExceeded) => {
                return AtomicCudaMarginStepOutcome::Committed(
                    finalize_alpha0_retained_with_clock(
                        (alpha1, gradients, alpha0_score),
                        alpha0_bounds,
                        AtomicCudaMarginStepRefusal::DeadlineExceeded,
                        self.deadline,
                        Instant::now,
                    ),
                );
            }
        };
        let alpha1_score = match score_complete_c(
            &alpha1_bounds,
            self.spec_matrix,
            self.thresholds,
            self.verify_upper_bound,
            self.deadline,
        ) {
            Ok(score) => score.hinge,
            Err(refusal) => {
                return AtomicCudaMarginStepOutcome::Committed(
                    finalize_alpha0_retained_with_clock(
                        (alpha1_bounds, alpha1, gradients, alpha0_score),
                        alpha0_bounds,
                        refusal,
                        self.deadline,
                        Instant::now,
                    ),
                );
            }
        };
        let result = if alpha1_score > alpha0_hinge {
            Ok(AtomicCudaMarginBracketChoice::Candidate(
                AtomicCudaMarginCertifiedPair {
                    ordinal: 0,
                    learning_rate: self.adam.learning_rate,
                    score: alpha1_score,
                    bounds: Box::new(alpha1_bounds),
                    alpha_state: Box::new(alpha1),
                },
            ))
        } else {
            drop((alpha1_bounds, alpha1));
            Ok(AtomicCudaMarginBracketChoice::Alpha0 {
                best_candidate_score: alpha1_score,
            })
        };
        AtomicCudaMarginStepOutcome::Committed(finalize_complete_lr_bracket_with_clock(
            (gradients, alpha0_score),
            alpha0_bounds,
            binding_index,
            alpha0_hinge,
            result,
            self.deadline,
            Instant::now,
        ))
    }

    fn run_complete_topk(
        &self,
        plan: &AtomicCudaMarginTopKPlan,
        alpha0_score: f32,
        relu_names: &[String],
        gradients: &[Vec<f32>],
        policy: AtomicCudaMarginTopKPolicy,
    ) -> Result<AtomicCudaMarginTopKChoice, AtomicCudaMarginStepRefusal> {
        deadline_check(policy.work_deadline)?;
        if plan.row_indices.is_empty()
            || plan.row_indices.len() > ATOMIC_CUDA_MARGIN_TOPK
            || plan.lower_objectives.len()
                != plan
                    .row_indices
                    .len()
                    .checked_mul(self.spec_matrix.ncols())
                    .ok_or(AtomicCudaMarginStepRefusal::InvalidTopKObjective)?
        {
            return Err(AtomicCudaMarginStepRefusal::InvalidTopKObjective);
        }
        let candidate_adam = AdamParams {
            learning_rate: policy.learning_rate,
            ..self.adam
        };
        let alpha1 = apply_checked_adam_step(
            self.alpha0,
            relu_names,
            gradients,
            candidate_adam,
            policy.work_deadline,
        )?;
        validate_complete_relu_candidate(self.alpha0, &alpha1, relu_names)?;
        deadline_check(policy.work_deadline)?;

        let alpha1_bounds = match AtomicCudaRowsRequest::new(
            self.graph,
            self.input,
            self.target_node,
            self.node_bounds,
            &alpha1,
            self.spec_matrix,
            self.reference,
            policy.work_deadline,
        )
        .run()
        {
            AtomicCudaRowsOutcome::RefusedBeforeCommit { refusal }
            | AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::ReferenceRetained {
                refusal,
                ..
            }) => return Err(AtomicCudaMarginStepRefusal::Alpha1Rows(refusal)),
            AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::CudaIntersection(bounds)) => {
                bounds
            }
            AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::DeadlineExceeded) => {
                return Err(AtomicCudaMarginStepRefusal::DeadlineExceeded);
            }
        };
        deadline_check(policy.work_deadline)?;
        let score = score_complete_c(
            &alpha1_bounds,
            self.spec_matrix,
            self.thresholds,
            self.verify_upper_bound,
            policy.work_deadline,
        )?
        .hinge;
        deadline_check(policy.work_deadline)?;

        let row_indices = plan.row_indices.clone().into_boxed_slice();
        if !alpha0_score.is_finite() {
            return Err(AtomicCudaMarginStepRefusal::ScoreNonFinite);
        }
        let choice = if score > alpha0_score {
            AtomicCudaMarginTopKChoice::Candidate {
                pair: AtomicCudaMarginCertifiedPair {
                    ordinal: 0,
                    learning_rate: policy.learning_rate,
                    score,
                    bounds: alpha1_bounds,
                    alpha_state: Box::new(alpha1),
                },
                row_indices,
            }
        } else {
            drop((alpha1_bounds, alpha1));
            AtomicCudaMarginTopKChoice::Alpha0 {
                candidate_score: score,
                row_indices,
            }
        };
        admit_topk_choice_with_clock(choice, policy.work_deadline, Instant::now)
    }

    fn run_complete_multi_iterations(
        &self,
        alpha0_bounds: Box<BoundedTensor>,
        alpha0_score: MarginScore,
        policy: AtomicCudaMarginIterationsPolicy,
        multiplicative_weights: bool,
    ) -> AtomicCudaMarginIterationsChoice {
        run_complete_multi_iterations_with_clock(
            self.alpha0,
            alpha0_bounds,
            alpha0_score,
            self.spec_matrix,
            self.thresholds,
            self.verify_upper_bound,
            self.adam,
            policy,
            multiplicative_weights,
            |current_state, plan, adam, work_deadline| {
                self.evaluate_complete_multi_step(current_state, plan, adam, work_deadline)
            },
            Instant::now,
        )
    }

    /// Produce one indivisible candidate pair at the latest accepted state.
    /// The gradient is proposal material only; complete-C CUDA evaluation is
    /// the sole source of candidate authority.
    fn evaluate_complete_multi_step(
        &self,
        current_state: &GraphAlphaState,
        plan: &AtomicCudaMarginIterationPlan,
        adam: AdamParams,
        work_deadline: Instant,
    ) -> Result<(Box<BoundedTensor>, Box<GraphAlphaState>), AtomicCudaMarginStepRefusal> {
        deadline_check(work_deadline)?;
        let (relu_names, gradients) = self.deadline_joint_gradients_for(
            current_state,
            &plan.lower_objectives,
            plan.num_specs,
            work_deadline,
        )?;
        let candidate =
            apply_checked_adam_step(current_state, &relu_names, &gradients, adam, work_deadline)?;
        validate_complete_relu_candidate(current_state, &candidate, &relu_names)?;
        deadline_check(work_deadline)?;

        let bounds = match AtomicCudaRowsRequest::new(
            self.graph,
            self.input,
            self.target_node,
            self.node_bounds,
            &candidate,
            self.spec_matrix,
            self.reference,
            work_deadline,
        )
        .run()
        {
            AtomicCudaRowsOutcome::RefusedBeforeCommit { refusal }
            | AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::ReferenceRetained {
                refusal,
                ..
            }) => return Err(AtomicCudaMarginStepRefusal::Alpha1Rows(refusal)),
            AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::CudaIntersection(bounds)) => {
                bounds
            }
            AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::DeadlineExceeded) => {
                return Err(AtomicCudaMarginStepRefusal::DeadlineExceeded);
            }
        };
        deadline_check(work_deadline)?;
        Ok((bounds, Box::new(candidate)))
    }

    fn run_complete_lr_bracket(
        &self,
        alpha0_score: &MarginScore,
        relu_names: &[String],
        gradients: &[Vec<f32>],
        policy: AtomicCudaMarginBracketPolicy,
    ) -> Result<AtomicCudaMarginBracketChoice, AtomicCudaMarginStepRefusal> {
        deadline_check(policy.work_deadline)?;
        let prepared = match prepare_complete_lr_bracket(
            self.alpha0,
            relu_names,
            gradients,
            self.adam,
            policy,
        ) {
            Ok(prepared) => prepared,
            Err(refusal) => {
                let mut now = Instant::now;
                return Err(bracket_refusal_after_drop_with_clock(
                    (),
                    policy,
                    refusal,
                    &mut now,
                ));
            }
        };
        let mut prepared = prepared.into_iter();
        let mut candidates = Vec::with_capacity(policy.candidate_lrs.len());
        let evaluated = (|| {
            for candidate in prepared.by_ref() {
                deadline_check(policy.work_deadline)?;
                let bounds = match AtomicCudaRowsRequest::new(
                    self.graph,
                    self.input,
                    self.target_node,
                    self.node_bounds,
                    &candidate.alpha_state,
                    self.spec_matrix,
                    self.reference,
                    policy.work_deadline,
                )
                .run()
                {
                    AtomicCudaRowsOutcome::RefusedBeforeCommit { refusal }
                    | AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::ReferenceRetained {
                        refusal,
                        ..
                    }) => {
                        return Err(AtomicCudaMarginStepRefusal::Alpha1Rows(refusal));
                    }
                    AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::CudaIntersection(
                        bounds,
                    )) => bounds,
                    AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::DeadlineExceeded) => {
                        return Err(AtomicCudaMarginStepRefusal::DeadlineExceeded);
                    }
                };
                deadline_check(policy.work_deadline)?;
                let score = score_complete_c(
                    &bounds,
                    self.spec_matrix,
                    self.thresholds,
                    self.verify_upper_bound,
                    policy.work_deadline,
                )?
                .hinge;
                deadline_check(policy.work_deadline)?;
                candidates.push(AtomicCudaMarginCertifiedPair {
                    ordinal: candidate.ordinal,
                    learning_rate: candidate.learning_rate,
                    score,
                    bounds,
                    alpha_state: candidate.alpha_state,
                });
            }
            Ok(())
        })();
        if let Err(refusal) = evaluated {
            let mut now = Instant::now;
            return Err(bracket_refusal_after_drop_with_clock(
                (prepared, candidates),
                policy,
                refusal,
                &mut now,
            ));
        }
        drop(prepared);
        select_complete_lr_bracket_with_clock(
            self.alpha0,
            alpha0_score.hinge,
            relu_names,
            self.spec_matrix,
            self.thresholds,
            self.verify_upper_bound,
            policy,
            candidates,
            Instant::now,
        )
    }

    fn deadline_joint_gradients(
        &self,
        lower_objective: &[f32],
        deadline: Instant,
    ) -> Result<(Vec<String>, Vec<Vec<f32>>), AtomicCudaMarginStepRefusal> {
        self.deadline_joint_gradients_for(self.alpha0, lower_objective, 1, deadline)
    }

    fn deadline_joint_gradients_for(
        &self,
        alpha_state: &GraphAlphaState,
        lower_objectives: &[f32],
        num_specs: usize,
        deadline: Instant,
    ) -> Result<(Vec<String>, Vec<Vec<f32>>), AtomicCudaMarginStepRefusal> {
        deadline_check(deadline)?;
        validate_joint_objective_shape(lower_objectives, num_specs, self.spec_matrix.ncols())?;
        let expected_gradient_calls = bounded_mw_gradient_call_count(num_specs)
            .ok_or(AtomicCudaMarginStepRefusal::JointMapping)?;
        if alpha_state.monotone_s_shaped_alphas.is_empty()
            && alpha_state.sqrt_alphas.is_empty()
            && alpha_state.reciprocal_alphas.is_empty()
        {
            // ReLU-only state: supported.
        } else {
            return Err(AtomicCudaMarginStepRefusal::NonReluAlphaState);
        }
        validate_adam(self.adam)?;
        if !resnet_gpu_enabled() {
            return Err(AtomicCudaMarginStepRefusal::ResnetGpuDisabled);
        }
        if !extract_skeleton_enabled() {
            return Err(AtomicCudaMarginStepRefusal::SkeletonDisabled);
        }
        let skeleton = build_resnet_segment_skeleton(
            self.graph,
            self.input,
            self.target_node,
            self.node_bounds,
            self.node_bounds,
            Some(alpha_state),
            /* allow_pure_chain = */ false,
        )
        .ok_or(AtomicCudaMarginStepRefusal::SkeletonRefused)?;
        deadline_check(deadline)?;
        let (segments, relu_names, _frontier_abs, _node_abs) = skeleton
            .fold_for_domain(
                self.graph,
                self.input,
                self.node_bounds,
                self.node_bounds,
                Some(alpha_state),
            )
            .ok_or(AtomicCudaMarginStepRefusal::FoldRefused)?;
        validate_relu_alignment(alpha_state, &relu_names)?;
        let pre_lowers = masked_pre_lowers(
            self.graph,
            self.input,
            self.node_bounds,
            alpha_state,
            &relu_names,
            deadline,
        )?;
        let in_lo: Vec<f32> = self.input.lower().iter().copied().collect();
        let in_hi: Vec<f32> = self.input.upper().iter().copied().collect();
        if in_lo.len() != in_hi.len()
            || in_lo
                .iter()
                .zip(&in_hi)
                .any(|(&lower, &upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
        {
            return Err(AtomicCudaMarginStepRefusal::InputNonFinite);
        }
        deadline_check(deadline)?;

        enum Admission {
            Refused(AtomicCudaMarginStepRefusal),
            Attempt(Result<Vec<Vec<f32>>, AtomicCudaMarginStepRefusal>),
        }
        // This accessor lends one shared `Sync` engine reference from a
        // `OnceLock`; it is not a global execution lock and does not promise
        // zero contention. Keep that same reference across at most 32 calls,
        // each of which is independently cooperatively deadline-bounded.
        let admitted = crate::sound_f64_gemm::with_engine_deadline(deadline, |engine| {
            let Some(gpu) = engine
                .as_gpu_crown_backward()
                .filter(|gpu| gpu.provides_sound_gpu_crown())
            else {
                return Admission::Refused(AtomicCudaMarginStepRefusal::NoSoundGpuRoute);
            };
            if !gpu.provides_deadline_bounded_joint_alpha_gradient_resident() {
                return Admission::Refused(AtomicCudaMarginStepRefusal::JointUnavailable);
            }
            let attempt = (|| {
                let output_dim = self.spec_matrix.ncols();
                let chunk_elements = ATOMIC_CUDA_MARGIN_TOPK
                    .checked_mul(output_dim)
                    .ok_or(AtomicCudaMarginStepRefusal::JointMapping)?;
                let mut chunks = lower_objectives.chunks(chunk_elements);
                let first = chunks
                    .next()
                    .ok_or(AtomicCudaMarginStepRefusal::JointMapping)?;
                let first_specs = first.len() / output_dim;
                let mut accumulated = joint_alpha_grads_fold_gpu_with_deadline(
                    gpu,
                    &segments,
                    first,
                    first_specs,
                    output_dim,
                    &in_lo,
                    &in_hi,
                    &pre_lowers,
                    relu_names.len(),
                    deadline,
                )
                .map_err(map_joint_error)?;
                let mut gradient_calls = 1usize;

                // The exact row-weighted gradient is additive across seed
                // rows.  Wider MW requests are therefore evaluated through
                // bounded eight-row resident calls, never one unbounded GPU
                // allocation.  The <=8 legacy and top-K paths execute only the
                // first call above, byte-for-byte as before.
                let mut host_work = 0usize;
                for chunk in chunks {
                    deadline_check(deadline)?;
                    if gradient_calls >= ATOMIC_CUDA_MARGIN_MW_MAX_GRADIENT_CALLS
                        || chunk.is_empty()
                        || !chunk.len().is_multiple_of(output_dim)
                    {
                        return Err(AtomicCudaMarginStepRefusal::JointMapping);
                    }
                    let chunk_specs = chunk.len() / output_dim;
                    let partial = joint_alpha_grads_fold_gpu_with_deadline(
                        gpu,
                        &segments,
                        chunk,
                        chunk_specs,
                        output_dim,
                        &in_lo,
                        &in_hi,
                        &pre_lowers,
                        relu_names.len(),
                        deadline,
                    )
                    .map_err(map_joint_error)?;
                    gradient_calls += 1;
                    accumulate_joint_gradient_chunk(
                        &mut accumulated,
                        partial,
                        deadline,
                        &mut host_work,
                    )?;
                }
                if gradient_calls != expected_gradient_calls {
                    return Err(AtomicCudaMarginStepRefusal::JointMapping);
                }
                deadline_check(deadline)?;
                Ok(accumulated)
            })();
            Admission::Attempt(attempt)
        });
        let gradients = match admitted {
            Ok(Some(Admission::Attempt(result))) => result?,
            Ok(Some(Admission::Refused(refusal))) => return Err(refusal),
            Ok(None) => return Err(AtomicCudaMarginStepRefusal::FactoryUnavailable),
            Err(_) if Instant::now() >= deadline => {
                return Err(AtomicCudaMarginStepRefusal::DeadlineExceeded);
            }
            Err(_) => return Err(AtomicCudaMarginStepRefusal::FactoryAdmissionError),
        };
        deadline_check(deadline)?;
        Ok((relu_names, gradients))
    }
}

fn map_joint_error(error: DeadlineJointAlphaFoldError) -> AtomicCudaMarginStepRefusal {
    match error {
        DeadlineJointAlphaFoldError::DeadlineExpired => {
            AtomicCudaMarginStepRefusal::DeadlineExceeded
        }
        DeadlineJointAlphaFoldError::JointUnavailable => {
            AtomicCudaMarginStepRefusal::JointUnavailable
        }
        DeadlineJointAlphaFoldError::NonFiniteGradient => {
            AtomicCudaMarginStepRefusal::JointNonFinite
        }
        DeadlineJointAlphaFoldError::MappingMismatch => AtomicCudaMarginStepRefusal::JointMapping,
    }
}

fn accumulate_joint_gradient_chunk(
    accumulated: &mut [Vec<f32>],
    partial: Vec<Vec<f32>>,
    deadline: Instant,
    host_work: &mut usize,
) -> Result<(), AtomicCudaMarginStepRefusal> {
    if accumulated.len() != partial.len() {
        return Err(AtomicCudaMarginStepRefusal::JointMapping);
    }
    for (total, addend) in accumulated.iter_mut().zip(partial) {
        if total.len() != addend.len() {
            return Err(AtomicCudaMarginStepRefusal::JointMapping);
        }
        for (total, addend) in total.iter_mut().zip(addend) {
            deadline_host_work(deadline, host_work)?;
            *total += addend;
            if !total.is_finite() {
                return Err(AtomicCudaMarginStepRefusal::JointNonFinite);
            }
        }
    }
    Ok(())
}

fn deadline_check(deadline: Instant) -> Result<(), AtomicCudaMarginStepRefusal> {
    if Instant::now() >= deadline {
        Err(AtomicCudaMarginStepRefusal::DeadlineExceeded)
    } else {
        Ok(())
    }
}

fn validate_joint_objective_shape(
    lower_objectives: &[f32],
    num_specs: usize,
    output_dim: usize,
) -> Result<(), AtomicCudaMarginStepRefusal> {
    let expected = num_specs
        .checked_mul(output_dim)
        .ok_or(AtomicCudaMarginStepRefusal::JointMapping)?;
    if num_specs == 0
        || num_specs > ATOMIC_CUDA_MARGIN_MW_MAX_ROWS
        || output_dim == 0
        || lower_objectives.len() != expected
        || lower_objectives.iter().any(|value| !value.is_finite())
    {
        Err(AtomicCudaMarginStepRefusal::JointMapping)
    } else {
        Ok(())
    }
}

fn bounded_mw_gradient_call_count(num_specs: usize) -> Option<usize> {
    let calls = num_specs.div_ceil(ATOMIC_CUDA_MARGIN_TOPK);
    (num_specs > 0 && calls <= ATOMIC_CUDA_MARGIN_MW_MAX_GRADIENT_CALLS).then_some(calls)
}

fn validate_adam(adam: AdamParams) -> Result<(), AtomicCudaMarginStepRefusal> {
    if adam.learning_rate.is_finite()
        && adam.learning_rate > 0.0
        && adam.beta1.is_finite()
        && (0.0..1.0).contains(&adam.beta1)
        && adam.beta2.is_finite()
        && (0.0..1.0).contains(&adam.beta2)
        && adam.epsilon.is_finite()
        && adam.epsilon > 0.0
        && adam.t > 0
    {
        Ok(())
    } else {
        Err(AtomicCudaMarginStepRefusal::InvalidAdamParams)
    }
}

/// This optimizer currently transports only the shared ReLU alpha axis. An
/// inconsistent partially initialized spec-axis state is refused too: silently
/// dropping either deltas or their optimizer moments would pair bounds with a
/// state different from the one whose backward pass they describe.
fn alpha_state_has_spec_axis(alpha: &GraphAlphaState) -> bool {
    !alpha.spec_deltas.is_empty()
        || !alpha.spec_slot_rows.is_empty()
        || !alpha.spec_adam_m.is_empty()
        || !alpha.spec_adam_v.is_empty()
}

fn validate_relu_alignment(
    alpha: &GraphAlphaState,
    relu_names: &[String],
) -> Result<(), AtomicCudaMarginStepRefusal> {
    let expected: BTreeSet<&str> = alpha.relu_nodes().collect();
    let actual: BTreeSet<&str> = relu_names.iter().map(String::as_str).collect();
    if expected.is_empty() || expected.len() != relu_names.len() || expected != actual {
        return Err(AtomicCudaMarginStepRefusal::ReluAlignment);
    }
    Ok(())
}

fn masked_pre_lowers(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    alpha: &GraphAlphaState,
    relu_names: &[String],
    deadline: Instant,
) -> Result<Vec<Vec<f32>>, AtomicCudaMarginStepRefusal> {
    let mut out = Vec::with_capacity(relu_names.len());
    let mut host_work = 0usize;
    for name in relu_names {
        deadline_host_work(deadline, &mut host_work)?;
        let node = graph
            .nodes
            .get(name)
            .ok_or(AtomicCudaMarginStepRefusal::MissingPreactivation)?;
        let input_name = node
            .inputs
            .first()
            .ok_or(AtomicCudaMarginStepRefusal::MissingPreactivation)?;
        let pre = if input_name == NETWORK_INPUT {
            input
        } else {
            node_bounds
                .get(input_name)
                .ok_or(AtomicCudaMarginStepRefusal::MissingPreactivation)?
        };
        let mask_raw = alpha
            .relu_unstable_mask(name)
            .ok_or(AtomicCudaMarginStepRefusal::ReluAlignment)?;
        let mask = if alpha.spatial_shape(name).is_some() {
            alpha.expand_mask(name, mask_raw)
        } else {
            mask_raw.clone()
        };
        if mask.len() != pre.lower().len() {
            return Err(AtomicCudaMarginStepRefusal::PreLowerShape);
        }
        let mut masked = Vec::new();
        masked
            .try_reserve_exact(mask.len())
            .map_err(|_| AtomicCudaMarginStepRefusal::PreLowerShape)?;
        for (&lower, &unstable) in pre.lower().iter().zip(mask.iter()) {
            deadline_host_work(deadline, &mut host_work)?;
            if !lower.is_finite() {
                return Err(AtomicCudaMarginStepRefusal::BoundsNonFiniteOrInverted);
            }
            masked.push(if unstable { lower } else { 0.0 });
        }
        out.push(masked);
    }
    deadline_check(deadline)?;
    Ok(out)
}

fn deadline_host_work(
    deadline: Instant,
    completed: &mut usize,
) -> Result<(), AtomicCudaMarginStepRefusal> {
    if completed.is_multiple_of(4096) {
        deadline_check(deadline)?;
    }
    *completed = completed
        .checked_add(1)
        .ok_or(AtomicCudaMarginStepRefusal::ReluAlignment)?;
    Ok(())
}

fn apply_checked_adam_step(
    alpha0: &GraphAlphaState,
    relu_names: &[String],
    gradients: &[Vec<f32>],
    adam: AdamParams,
    deadline: Instant,
) -> Result<GraphAlphaState, AtomicCudaMarginStepRefusal> {
    validate_adam(adam)?;
    validate_relu_alignment(alpha0, relu_names)?;
    if gradients.len() != relu_names.len() {
        return Err(AtomicCudaMarginStepRefusal::ReluAlignment);
    }

    let mut prepared = Vec::with_capacity(relu_names.len());
    let mut host_work = 0usize;
    for (name, gradient) in relu_names.iter().zip(gradients) {
        deadline_host_work(deadline, &mut host_work)?;
        if gradient.iter().any(|value| !value.is_finite()) {
            return Err(AtomicCudaMarginStepRefusal::JointNonFinite);
        }
        let raw = Array1::from_vec(gradient.clone());
        let reduced = alpha0.reduce_gradient(name, &raw);
        let expected = alpha0
            .relu_len(name)
            .ok_or(AtomicCudaMarginStepRefusal::ReluAlignment)?;
        if reduced.len() != expected || reduced.iter().any(|value| !value.is_finite()) {
            return Err(AtomicCudaMarginStepRefusal::ReluAlignment);
        }
        if alpha0.alphas.get(name).map(|values| values.len()) != Some(expected)
            || alpha0.alphas_upper.get(name).map(|values| values.len()) != Some(expected)
            || alpha0.unstable_mask.get(name).map(|values| values.len()) != Some(expected)
        {
            return Err(AtomicCudaMarginStepRefusal::AlphaUpdateRefused);
        }
        for map in [
            &alpha0.adam_m,
            &alpha0.adam_v,
            &alpha0.adam_m_upper,
            &alpha0.adam_v_upper,
        ] {
            if map.get(name).map(|values| values.len()) != Some(expected) {
                return Err(AtomicCudaMarginStepRefusal::AlphaUpdateRefused);
            }
        }
        for values in [
            &alpha0.alphas[name],
            &alpha0.alphas_upper[name],
            &alpha0.adam_m[name],
            &alpha0.adam_v[name],
            &alpha0.adam_m_upper[name],
            &alpha0.adam_v_upper[name],
        ] {
            if values.iter().any(|value| !value.is_finite()) {
                return Err(AtomicCudaMarginStepRefusal::AlphaUpdateRefused);
            }
        }
        if alpha0.alphas[name]
            .iter()
            .chain(alpha0.alphas_upper[name].iter())
            .any(|value| !(0.0..=1.0).contains(value))
        {
            return Err(AtomicCudaMarginStepRefusal::AlphaUpdateRefused);
        }
        prepared.push((name.clone(), reduced.mapv(|value| -value)));
    }
    deadline_check(deadline)?;

    // Candidate-only clone: no partial optimizer mutation can escape.
    let mut alpha1 = alpha0.clone();
    deadline_check(deadline)?;
    for (name, descent_gradient) in &prepared {
        deadline_host_work(deadline, &mut host_work)?;
        alpha1.update_adam(name, descent_gradient, &adam);
        alpha1.update_adam_upper(name, descent_gradient, &adam);
    }
    deadline_check(deadline)?;

    let mut moved = false;
    for name in relu_names {
        let (before_lower, before_upper) = alpha0
            .relu_alpha_pair(name)
            .ok_or(AtomicCudaMarginStepRefusal::AlphaUpdateRefused)?;
        let (after_lower, after_upper) = alpha1
            .relu_alpha_pair(name)
            .ok_or(AtomicCudaMarginStepRefusal::AlphaUpdateRefused)?;
        if before_lower.len() != after_lower.len() || before_upper.len() != after_upper.len() {
            return Err(AtomicCudaMarginStepRefusal::AlphaUpdateRefused);
        }
        for ((&before, &after), (&before_u, &after_u)) in before_lower
            .iter()
            .zip(after_lower)
            .zip(before_upper.iter().zip(after_upper))
        {
            deadline_host_work(deadline, &mut host_work)?;
            if !after.is_finite()
                || !after_u.is_finite()
                || !(0.0..=1.0).contains(&after)
                || !(0.0..=1.0).contains(&after_u)
            {
                return Err(AtomicCudaMarginStepRefusal::AlphaUpdateRefused);
            }
            moved |= before.to_bits() != after.to_bits() || before_u.to_bits() != after_u.to_bits();
        }
        for values in [
            &alpha1.adam_m[name],
            &alpha1.adam_v[name],
            &alpha1.adam_m_upper[name],
            &alpha1.adam_v_upper[name],
        ] {
            if values.iter().any(|value| !value.is_finite()) {
                return Err(AtomicCudaMarginStepRefusal::AlphaUpdateRefused);
            }
        }
    }
    deadline_check(deadline)?;
    if !moved {
        return Err(AtomicCudaMarginStepRefusal::AlphaDidNotMove);
    }
    Ok(alpha1)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::time::Duration;

    use ndarray::{arr1, arr2, Array3};

    use super::*;

    struct DropFlag<'a>(&'a Cell<bool>);

    impl Drop for DropFlag<'_> {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }

    fn bounds(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        BoundedTensor::new(
            Array1::from_vec(lower.to_vec()).into_dyn(),
            Array1::from_vec(upper.to_vec()).into_dyn(),
        )
        .expect("valid test bounds")
    }

    fn alpha_fixture() -> GraphAlphaState {
        let mut alpha = GraphAlphaState::new();
        alpha
            .add_relu_node("relu_a", &bounds(&[-2.0, -1.0], &[1.0, 2.0]), false)
            .expect("alpha fixture");
        alpha
            .add_relu_node("relu_b", &bounds(&[-1.0], &[3.0]), false)
            .expect("alpha fixture");
        alpha
    }

    fn alpha_with_marker(marker: f32) -> Box<GraphAlphaState> {
        let mut alpha = alpha_fixture();
        alpha.alphas.get_mut("relu_a").expect("fixture lower alpha")[0] = marker;
        alpha
            .alphas_upper
            .get_mut("relu_a")
            .expect("fixture upper alpha")[0] = marker;
        Box::new(alpha)
    }

    fn alpha_marker(alpha: &GraphAlphaState) -> f32 {
        alpha.alphas["relu_a"][0]
    }

    fn complete_score(
        candidate: &BoundedTensor,
        spec: &Array2<f32>,
        thresholds: &[f32],
    ) -> MarginScore {
        score_complete_c(
            candidate,
            spec,
            thresholds,
            false,
            Instant::now() + Duration::from_secs(10),
        )
        .expect("valid complete-C score")
    }

    fn bracket_fixture(
        now: Instant,
    ) -> (GraphAlphaState, Vec<String>, AtomicCudaMarginBracketPolicy) {
        let alpha0 = alpha_fixture();
        let relu_names = vec!["relu_b".to_string(), "relu_a".to_string()];
        let policy = AtomicCudaMarginBracketPolicy::new_at(
            AdamParams::new(0.25, 1),
            0.25,
            now + Duration::from_secs(10),
            now,
        )
        .expect("valid bracket policy");
        (alpha0, relu_names, policy)
    }

    fn complete_candidate_pairs(
        alpha0: &GraphAlphaState,
        relu_names: &[String],
        policy: AtomicCudaMarginBracketPolicy,
        scores: [f32; 3],
    ) -> Vec<AtomicCudaMarginCertifiedPair> {
        let gradients = vec![vec![2.0], vec![4.0, -4.0]];
        prepare_complete_lr_bracket(
            alpha0,
            relu_names,
            &gradients,
            AdamParams::new(0.25, 1),
            policy,
        )
        .expect("complete prepared bracket")
        .into_iter()
        .zip(scores)
        .map(|(candidate, score)| AtomicCudaMarginCertifiedPair {
            ordinal: candidate.ordinal,
            learning_rate: candidate.learning_rate,
            score,
            bounds: Box::new(bounds(&[score], &[score + 0.5])),
            alpha_state: candidate.alpha_state,
        })
        .collect()
    }

    fn select_test_bracket<F>(
        alpha0: &GraphAlphaState,
        relu_names: &[String],
        policy: AtomicCudaMarginBracketPolicy,
        alpha0_score: f32,
        candidates: Vec<AtomicCudaMarginCertifiedPair>,
        now: F,
    ) -> Result<AtomicCudaMarginBracketChoice, AtomicCudaMarginStepRefusal>
    where
        F: FnMut() -> Instant,
    {
        select_complete_lr_bracket_with_clock(
            alpha0,
            alpha0_score,
            relu_names,
            &arr2(&[[1.0]]),
            &[0.0],
            false,
            policy,
            candidates,
            now,
        )
    }

    #[test]
    fn gate_is_exact_and_default_dark() {
        assert!(!parse_root_alpha_cuda_margin_step(None));
        for raw in ["", "0", "true", " 1", "1 "] {
            assert!(!parse_root_alpha_cuda_margin_step(Some(raw)), "raw={raw:?}");
        }
        assert!(parse_root_alpha_cuda_margin_step(Some("1")));

        assert!(!parse_root_alpha_cuda_margin_lr_bracket(None));
        for raw in ["", "0", "true", " 1", "1 "] {
            assert!(
                !parse_root_alpha_cuda_margin_lr_bracket(Some(raw)),
                "raw={raw:?}"
            );
        }
        assert!(parse_root_alpha_cuda_margin_lr_bracket(Some("1")));

        assert!(!parse_root_alpha_cuda_margin_topk(None));
        for raw in ["", "0", "true", " 1", "1 "] {
            assert!(!parse_root_alpha_cuda_margin_topk(Some(raw)), "raw={raw:?}");
        }
        assert!(parse_root_alpha_cuda_margin_topk(Some("1")));

        assert!(!parse_root_alpha_cuda_margin_mw(None));
        for raw in ["", "0", "true", " 1", "1 "] {
            assert!(!parse_root_alpha_cuda_margin_mw(Some(raw)), "raw={raw:?}");
        }
        assert!(parse_root_alpha_cuda_margin_mw(Some("1")));

        let reads = Cell::new(0usize);
        assert!(!root_alpha_cuda_margin_mw_enabled_if(0, || {
            reads.set(reads.get() + 1);
            Some("1".to_string())
        }));
        assert_eq!(reads.get(), 0, "a dark typed parent must not read MW");
        assert!(root_alpha_cuda_margin_mw_enabled_if(3, || {
            reads.set(reads.get() + 1);
            Some("1".to_string())
        }));
        assert_eq!(reads.get(), 1);

        assert_eq!(
            select_margin_search_policy(false, false),
            Ok(AtomicCudaMarginSearchPolicy::Legacy)
        );
        assert_eq!(
            select_margin_search_policy(false, true),
            Ok(AtomicCudaMarginSearchPolicy::LearningRateBracket)
        );
        assert_eq!(
            select_margin_search_policy(true, false),
            Ok(AtomicCudaMarginSearchPolicy::TopK)
        );
        assert_eq!(
            select_margin_search_policy(true, true),
            Err(AtomicCudaMarginStepRefusal::ConflictingSearchPolicy)
        );

        assert_eq!(
            multi_iteration_search_policy_is_exclusive(0, true, true),
            Ok(())
        );
        assert_eq!(
            multi_iteration_search_policy_is_exclusive(3, false, false),
            Ok(())
        );
        for (topk, bracket) in [(true, false), (false, true), (true, true)] {
            assert_eq!(
                multi_iteration_search_policy_is_exclusive(1, topk, bracket),
                Err(AtomicCudaMarginStepRefusal::ConflictingSearchPolicy)
            );
        }
    }

    #[test]
    fn multi_iteration_policy_is_dark_capped_and_continues_adam_schedule() {
        let now = Instant::now();
        let authority_deadline = now + Duration::from_secs(2);
        let base = AdamParams::new(0.2, 21);
        let policy = AtomicCudaMarginIterationsPolicy::new_at(
            ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS,
            base,
            0.5,
            authority_deadline,
            now,
        )
        .expect("the hard cap is admitted");
        assert_eq!(
            policy.work_deadline,
            authority_deadline
                .checked_sub(ATOMIC_CUDA_MARGIN_PUBLICATION_RESERVE)
                .expect("the test deadline admits the reserve")
        );
        assert_eq!(policy.authority_deadline, authority_deadline);
        assert_eq!(policy.adam_for_offset(base, 0).expect("offset zero").t, 21);
        let third = policy.adam_for_offset(base, 2).expect("continued schedule");
        assert_eq!(third.t, 23);
        assert_eq!(third.learning_rate.to_bits(), 0.05_f32.to_bits());

        for iterations in [0, ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS + 1] {
            assert_eq!(
                AtomicCudaMarginIterationsPolicy::new_at(
                    iterations,
                    base,
                    0.5,
                    authority_deadline,
                    now,
                )
                .unwrap_err(),
                AtomicCudaMarginStepRefusal::InvalidIterationPolicy
            );
        }
        for decay in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(
                AtomicCudaMarginIterationsPolicy::new_at(1, base, decay, authority_deadline, now,)
                    .unwrap_err(),
                AtomicCudaMarginStepRefusal::InvalidIterationPolicy
            );
        }
        assert_eq!(
            AtomicCudaMarginIterationsPolicy::new_at(
                1,
                base,
                0.5,
                now + ATOMIC_CUDA_MARGIN_PUBLICATION_RESERVE,
                now,
            )
            .unwrap_err(),
            AtomicCudaMarginStepRefusal::WorkDeadlineExceeded
        );
    }

    #[test]
    fn multiplicative_weights_builds_positive_direction_aware_full_row_seed() {
        let deadline = Instant::now() + Duration::from_secs(2);
        let spec = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let thresholds = [0.0, 0.0];
        let candidate = bounds(&[-3.0, -1.0], &[-2.0, 1.0]);

        let mut lower = MultiplicativeWeightsRowPlayer::new(&spec, false, 2, deadline)
            .expect("bounded lower row player");
        let lower_plan = lower
            .objective_for(&candidate, &thresholds, 0, deadline)
            .expect("finite lower MW objective");
        assert_eq!(lower_plan.binding_row, 0);
        assert_eq!(lower_plan.num_specs, 2);
        assert!(!lower_plan.adaptive_weights);
        assert_eq!(lower_plan.lower_objectives.len(), 4);
        assert_eq!(
            lower_plan.lower_objectives[0].to_bits(),
            lower_plan.lower_objectives[3].to_bits(),
            "round one must use the uniform pre-update distribution"
        );
        assert!(lower_plan.lower_objectives[3] > 0.0);
        assert_eq!(lower_plan.lower_objectives[1].to_bits(), 0.0_f32.to_bits());
        assert_eq!(lower_plan.lower_objectives[2].to_bits(), 0.0_f32.to_bits());
        let lower_second = lower
            .objective_for(&candidate, &thresholds, 0, deadline)
            .expect("second lower MW objective");
        assert!(lower_second.adaptive_weights);
        assert!(
            lower_second.lower_objectives[0] > lower_second.lower_objectives[3],
            "round two must react to the previously observed worst row"
        );

        let mut upper = MultiplicativeWeightsRowPlayer::new(&spec, true, 2, deadline)
            .expect("bounded upper row player");
        let upper_candidate = bounds(&[0.0, 0.0], &[3.0, 1.0]);
        let upper_plan = upper
            .objective_for(&upper_candidate, &thresholds, 0, deadline)
            .expect("finite upper MW objective");
        assert_eq!(upper_plan.num_specs, 2);
        assert!(!upper_plan.adaptive_weights);
        assert_eq!(
            upper_plan.lower_objectives[0].to_bits(),
            upper_plan.lower_objectives[3].to_bits(),
            "upper-bound round one must also be uniform"
        );
        assert!(upper_plan.lower_objectives[3] < 0.0);
        let upper_second = upper
            .objective_for(&upper_candidate, &thresholds, 0, deadline)
            .expect("second upper MW objective");
        assert!(upper_second.adaptive_weights);
        assert!(
            upper_second.lower_objectives[0] < upper_second.lower_objectives[3],
            "upper-bound orientation must put more negative seed mass on the worst row"
        );
    }

    #[test]
    fn multiplicative_weights_route_keeps_complete_hinge_as_authority() {
        let now = Instant::now();
        let spec = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let thresholds = [0.0, 0.0];
        let base_adam = AdamParams::new(0.1, 1);
        let policy = AtomicCudaMarginIterationsPolicy::new_at(
            1,
            base_adam,
            1.0,
            now + Duration::from_secs(2),
            now,
        )
        .expect("valid MW policy");
        let alpha0 = alpha_fixture();
        let initial = Box::new(bounds(&[-3.0, -1.0], &[1.0, 1.0]));
        let initial_score = complete_score(&initial, &spec, &thresholds);
        let mut observed_plan = None;

        let choice = run_complete_multi_iterations_with_clock(
            &alpha0,
            initial,
            initial_score,
            &spec,
            &thresholds,
            false,
            base_adam,
            policy,
            true,
            |_state, plan, _adam, _deadline| {
                observed_plan = Some(plan.clone());
                Ok((
                    Box::new(bounds(&[-1.5, -0.5], &[1.0, 1.0])),
                    alpha_with_marker(0.8),
                ))
            },
            || now,
        );

        let plan = observed_plan.expect("MW must dispatch one proposal");
        assert_eq!(plan.binding_row, 0);
        assert_eq!(plan.num_specs, 2);
        assert!(!plan.adaptive_weights);
        assert_eq!(
            plan.lower_objectives[0].to_bits(),
            plan.lower_objectives[3].to_bits(),
            "the first dispatched MW plan is the uniform pre-update play"
        );
        let AtomicCudaMarginIterationsChoice::Candidate { pair, summary } = choice else {
            panic!("a strict complete-C hinge improvement must publish its whole pair");
        };
        assert_eq!(summary.initial_score.to_bits(), (-4.0_f32).to_bits());
        assert_eq!(summary.final_score.to_bits(), (-2.0_f32).to_bits());
        assert_eq!(summary.accepted_iterations, 1);
        assert!(summary.multiplicative_weights_requested);
        assert!(summary.multiplicative_weights_plan_dispatched);
        assert!(!summary.multiplicative_weights_effective);
        assert_eq!(summary.completed_proposals, 1);
        assert_eq!(summary.adaptive_plan_dispatches, 0);
        assert_eq!(summary.gradient_plan_num_specs, Some(2));
        assert_eq!(summary.gradient_row_count, 2);
        assert_eq!(pair.bounds.lower().as_slice(), Some(&[-1.5, -0.5][..]));
        assert_eq!(alpha_marker(&pair.alpha_state).to_bits(), 0.8_f32.to_bits());
    }

    #[test]
    fn multiplicative_weights_rejection_reuses_authoritative_bounds_and_state() {
        let now = Instant::now();
        let spec = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let thresholds = [0.0, 0.0];
        let base_adam = AdamParams::new(0.1, 1);
        let policy = AtomicCudaMarginIterationsPolicy::new_at(
            2,
            base_adam,
            1.0,
            now + Duration::from_secs(2),
            now,
        )
        .expect("valid MW rejection policy");
        let alpha0 = alpha_fixture();
        let alpha0_marker = alpha_marker(&alpha0);
        let initial = Box::new(bounds(&[-4.0, -1.0], &[1.0, 1.0]));
        let initial_score = complete_score(&initial, &spec, &thresholds);
        let mut observed_state = Vec::new();
        let mut observed_weights = Vec::new();
        let mut proposal = 0usize;

        let choice = run_complete_multi_iterations_with_clock(
            &alpha0,
            initial,
            initial_score,
            &spec,
            &thresholds,
            false,
            base_adam,
            policy,
            true,
            |state, plan, _adam, _deadline| {
                observed_state.push(alpha_marker(state));
                observed_weights.push((plan.lower_objectives[0], plan.lower_objectives[3]));
                proposal += 1;
                Ok(match proposal {
                    1 => (
                        Box::new(bounds(&[-6.0, -6.0], &[1.0, 1.0])),
                        alpha_with_marker(0.9),
                    ),
                    2 => (
                        Box::new(bounds(&[-1.0, -1.0], &[1.0, 1.0])),
                        alpha_with_marker(0.7),
                    ),
                    _ => unreachable!("policy admits exactly two proposals"),
                })
            },
            || now,
        );

        assert_eq!(proposal, 2);
        assert!(observed_state
            .iter()
            .all(|marker| marker.to_bits() == alpha0_marker.to_bits()));
        assert!(
            observed_weights[1].0 > observed_weights[0].0
                && observed_weights[1].1 < observed_weights[0].1,
            "the persistent row player must update again from the retained slacks"
        );
        let AtomicCudaMarginIterationsChoice::Candidate { pair, summary } = choice else {
            panic!("the second strict improvement must publish its whole pair");
        };
        assert_eq!(summary.attempted_iterations, 2);
        assert_eq!(summary.accepted_iterations, 1);
        assert!(summary.multiplicative_weights_requested);
        assert!(summary.multiplicative_weights_plan_dispatched);
        assert!(summary.multiplicative_weights_effective);
        assert_eq!(summary.completed_proposals, 2);
        assert_eq!(summary.adaptive_plan_dispatches, 1);
        assert_eq!(summary.gradient_plan_num_specs, Some(2));
        assert_eq!(summary.gradient_row_count, 2);
        assert_eq!(summary.initial_score.to_bits(), (-5.0_f32).to_bits());
        assert_eq!(summary.final_score.to_bits(), (-2.0_f32).to_bits());
        assert_eq!(pair.bounds.lower().as_slice(), Some(&[-1.0, -1.0][..]));
        assert_eq!(alpha_marker(&pair.alpha_state).to_bits(), 0.7_f32.to_bits());
    }

    #[test]
    fn multiplicative_weights_refuses_above_its_bounded_row_cap() {
        assert_eq!(ATOMIC_CUDA_MARGIN_MW_MAX_ROWS, 256);
        assert_eq!(bounded_mw_gradient_call_count(0), None);
        assert_eq!(bounded_mw_gradient_call_count(1), Some(1));
        assert_eq!(
            bounded_mw_gradient_call_count(ATOMIC_CUDA_MARGIN_MW_MAX_ROWS),
            Some(ATOMIC_CUDA_MARGIN_MW_MAX_GRADIENT_CALLS)
        );
        assert_eq!(
            bounded_mw_gradient_call_count(ATOMIC_CUDA_MARGIN_MW_MAX_ROWS + 1),
            None
        );
        let deadline = Instant::now() + Duration::from_secs(2);
        let oversized = Array2::zeros((ATOMIC_CUDA_MARGIN_MW_MAX_ROWS + 1, 1));
        assert!(matches!(
            MultiplicativeWeightsRowPlayer::new(&oversized, false, 1, deadline),
            Err(AtomicCudaMarginStepRefusal::InvalidMultiplicativeWeights)
        ));
    }

    #[test]
    fn multiplicative_weights_refuses_above_its_bounded_seed_element_cap() {
        let deadline = Instant::now() + Duration::from_secs(2);
        let columns = ATOMIC_CUDA_MARGIN_MW_MAX_SEED_ELEMENTS / ATOMIC_CUDA_MARGIN_MW_MAX_ROWS + 1;
        let oversized = Array2::zeros((ATOMIC_CUDA_MARGIN_MW_MAX_ROWS, columns));
        assert!(matches!(
            MultiplicativeWeightsRowPlayer::new(&oversized, false, 1, deadline),
            Err(AtomicCudaMarginStepRefusal::InvalidMultiplicativeWeights)
        ));
    }

    #[test]
    fn multiplicative_weights_off_preserves_exact_legacy_single_row_plan() {
        let now = Instant::now();
        let spec = arr2(&[[2.0, -1.0], [0.0, 3.0]]);
        let thresholds = [0.0, 0.0];
        let base_adam = AdamParams::new(0.1, 1);
        let policy = AtomicCudaMarginIterationsPolicy::new_at(
            1,
            base_adam,
            1.0,
            now + Duration::from_secs(2),
            now,
        )
        .expect("valid legacy policy");
        let alpha0 = alpha_fixture();
        let initial = Box::new(bounds(&[-4.0, -1.0], &[1.0, 1.0]));
        let initial_score = complete_score(&initial, &spec, &thresholds);
        let mut observed = None;

        let choice = run_complete_multi_iterations_with_clock(
            &alpha0,
            initial,
            initial_score,
            &spec,
            &thresholds,
            false,
            base_adam,
            policy,
            false,
            |_state, plan, _adam, _deadline| {
                observed = Some(plan.clone());
                Err(AtomicCudaMarginStepRefusal::WorkDeadlineExceeded)
            },
            || now,
        );

        let plan = observed.expect("legacy route must dispatch its binding row");
        assert_eq!(plan.binding_row, 0);
        assert_eq!(plan.num_specs, 1);
        assert_eq!(plan.lower_objectives, vec![2.0, -1.0]);
        let AtomicCudaMarginIterationsChoice::Alpha0 { summary, .. } = choice else {
            panic!("a failed legacy proposal must retain alpha0");
        };
        assert!(!summary.multiplicative_weights_requested);
        assert!(!summary.multiplicative_weights_plan_dispatched);
        assert!(!summary.multiplicative_weights_effective);
        assert_eq!(summary.completed_proposals, 0);
        assert_eq!(summary.adaptive_plan_dispatches, 0);
        assert_eq!(summary.gradient_plan_num_specs, Some(1));
        assert_eq!(summary.gradient_row_count, 2);
    }

    #[test]
    fn multiplicative_weights_refusal_authenticates_plan_without_claiming_effective() {
        let now = Instant::now();
        let spec = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let thresholds = [0.0, 0.0];
        let base_adam = AdamParams::new(0.1, 1);
        let policy = AtomicCudaMarginIterationsPolicy::new_at(
            1,
            base_adam,
            1.0,
            now + Duration::from_secs(2),
            now,
        )
        .expect("valid MW refusal policy");
        let alpha0 = alpha_fixture();
        let initial = Box::new(bounds(&[-2.0, -1.0], &[1.0, 1.0]));
        let initial_score = complete_score(&initial, &spec, &thresholds);

        let choice = run_complete_multi_iterations_with_clock(
            &alpha0,
            initial,
            initial_score,
            &spec,
            &thresholds,
            false,
            base_adam,
            policy,
            true,
            |_state, plan, _adam, _deadline| {
                assert_eq!(plan.num_specs, 2);
                Err(AtomicCudaMarginStepRefusal::JointUnavailable)
            },
            || now,
        );

        let AtomicCudaMarginIterationsChoice::Alpha0 { summary, .. } = choice else {
            panic!("a refused MW gradient must retain alpha0");
        };
        assert_eq!(summary.attempted_iterations, 1);
        assert_eq!(summary.completed_proposals, 0);
        assert_eq!(summary.adaptive_plan_dispatches, 0);
        assert!(summary.multiplicative_weights_requested);
        assert!(summary.multiplicative_weights_plan_dispatched);
        assert!(!summary.multiplicative_weights_effective);
        assert_eq!(summary.gradient_plan_num_specs, Some(2));
        assert_eq!(summary.gradient_row_count, 2);
        assert_eq!(
            summary.stop_refusal,
            Some(AtomicCudaMarginStepRefusal::JointUnavailable)
        );
    }

    #[test]
    fn multiplicative_weights_adaptive_dispatch_requires_completion_to_be_effective() {
        let now = Instant::now();
        let spec = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let thresholds = [0.0, 0.0];
        let base_adam = AdamParams::new(0.1, 1);
        let policy = AtomicCudaMarginIterationsPolicy::new_at(
            2,
            base_adam,
            1.0,
            now + Duration::from_secs(2),
            now,
        )
        .expect("valid adaptive refusal policy");
        let alpha0 = alpha_fixture();
        let initial = Box::new(bounds(&[-2.0, -1.0], &[1.0, 1.0]));
        let initial_score = complete_score(&initial, &spec, &thresholds);
        let mut calls = 0usize;

        let choice = run_complete_multi_iterations_with_clock(
            &alpha0,
            initial,
            initial_score,
            &spec,
            &thresholds,
            false,
            base_adam,
            policy,
            true,
            |_state, plan, _adam, _deadline| {
                calls += 1;
                match calls {
                    1 => {
                        assert!(!plan.adaptive_weights);
                        Ok((
                            Box::new(bounds(&[-3.0, -3.0], &[1.0, 1.0])),
                            alpha_with_marker(0.9),
                        ))
                    }
                    2 => {
                        assert!(plan.adaptive_weights);
                        Err(AtomicCudaMarginStepRefusal::JointUnavailable)
                    }
                    _ => unreachable!("the second dispatch refuses"),
                }
            },
            || now,
        );

        let AtomicCudaMarginIterationsChoice::Alpha0 { summary, .. } = choice else {
            panic!("a rejected uniform proposal and refused adaptive proposal retain alpha0");
        };
        assert_eq!(summary.attempted_iterations, 2);
        assert_eq!(summary.completed_proposals, 1);
        assert_eq!(summary.adaptive_plan_dispatches, 1);
        assert!(summary.multiplicative_weights_plan_dispatched);
        assert!(!summary.multiplicative_weights_effective);
        assert_eq!(
            summary.stop_refusal,
            Some(AtomicCudaMarginStepRefusal::JointUnavailable)
        );
    }

    #[test]
    fn multiplicative_weights_admission_failure_retains_alpha0_without_dispatch() {
        let now = Instant::now();
        let rows = ATOMIC_CUDA_MARGIN_MW_MAX_ROWS + 1;
        let spec = Array2::zeros((rows, 1));
        let thresholds = vec![0.0; rows];
        let lower = vec![-1.0; rows];
        let upper = vec![1.0; rows];
        let initial = Box::new(bounds(&lower, &upper));
        let initial_score = complete_score(&initial, &spec, &thresholds);
        let alpha0 = alpha_fixture();
        let base_adam = AdamParams::new(0.1, 1);
        let policy = AtomicCudaMarginIterationsPolicy::new_at(
            1,
            base_adam,
            1.0,
            now + Duration::from_secs(2),
            now,
        )
        .expect("valid bounded policy");
        let mut dispatched = false;

        let choice = run_complete_multi_iterations_with_clock(
            &alpha0,
            initial,
            initial_score,
            &spec,
            &thresholds,
            false,
            base_adam,
            policy,
            true,
            |_state, _plan, _adam, _deadline| {
                dispatched = true;
                unreachable!("an inadmissible MW seed must not dispatch")
            },
            || now,
        );

        assert!(!dispatched);
        let AtomicCudaMarginIterationsChoice::Alpha0 { bounds, summary } = choice else {
            panic!("MW admission failure must retain alpha0");
        };
        assert_eq!(bounds.lower().as_slice(), Some(lower.as_slice()));
        assert_eq!(summary.attempted_iterations, 0);
        assert_eq!(summary.accepted_iterations, 0);
        assert!(summary.multiplicative_weights_requested);
        assert!(!summary.multiplicative_weights_plan_dispatched);
        assert!(!summary.multiplicative_weights_effective);
        assert_eq!(summary.completed_proposals, 0);
        assert_eq!(summary.adaptive_plan_dispatches, 0);
        assert_eq!(summary.gradient_plan_num_specs, None);
        assert_eq!(summary.gradient_row_count, rows);
        assert_eq!(
            summary.stop_refusal,
            Some(AtomicCudaMarginStepRefusal::InvalidMultiplicativeWeights)
        );
    }

    #[test]
    fn multi_iteration_rebinds_and_uses_latest_accepted_state_and_schedule() {
        let now = Instant::now();
        let spec = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let thresholds = [0.0, 0.0];
        let alpha0 = alpha_fixture();
        let alpha0_bounds = Box::new(bounds(&[-3.0, -1.0], &[1.0, 1.0]));
        let alpha0_score = complete_score(&alpha0_bounds, &spec, &thresholds);
        let base_adam = AdamParams::new(0.2, 21);
        let policy = AtomicCudaMarginIterationsPolicy::new_at(
            2,
            base_adam,
            0.5,
            now + Duration::from_secs(2),
            now,
        )
        .expect("valid bounded iteration policy");
        let mut observed_rows = Vec::new();
        let mut observed_markers = Vec::new();
        let mut observed_schedule = Vec::new();
        let mut proposal = 0usize;

        let choice = run_complete_multi_iterations_with_clock(
            &alpha0,
            alpha0_bounds,
            alpha0_score,
            &spec,
            &thresholds,
            false,
            base_adam,
            policy,
            false,
            |state, plan, adam, _deadline| {
                observed_rows.push(plan.binding_row);
                observed_markers.push(alpha_marker(state));
                observed_schedule.push((adam.t, adam.learning_rate));
                proposal += 1;
                Ok(match proposal {
                    1 => (
                        Box::new(bounds(&[-0.5, -2.0], &[1.0, 1.0])),
                        alpha_with_marker(0.6),
                    ),
                    2 => (
                        Box::new(bounds(&[-0.1, -0.2], &[1.0, 1.0])),
                        alpha_with_marker(0.7),
                    ),
                    _ => unreachable!("policy admits exactly two proposals"),
                })
            },
            || now,
        );

        assert_eq!(observed_rows, vec![0, 1], "the worst row must rebind");
        assert_eq!(observed_markers[1].to_bits(), 0.6_f32.to_bits());
        assert_eq!(observed_schedule[0].0, 21);
        assert_eq!(observed_schedule[1].0, 22);
        assert_eq!(observed_schedule[0].1.to_bits(), 0.2_f32.to_bits());
        assert_eq!(observed_schedule[1].1.to_bits(), 0.1_f32.to_bits());

        let AtomicCudaMarginIterationsChoice::Candidate { pair, summary } = choice else {
            panic!("both strict improvements must select the latest whole pair");
        };
        assert_eq!(&*summary.binding_rows, &[0, 1]);
        assert_eq!(summary.attempted_iterations, 2);
        assert_eq!(summary.accepted_iterations, 2);
        assert_eq!(summary.initial_score.to_bits(), (-4.0_f32).to_bits());
        assert_eq!(summary.final_score.to_bits(), (-0.3_f32).to_bits());
        assert_eq!(pair.bounds.lower().as_slice(), Some(&[-0.1, -0.2][..]));
        assert_eq!(alpha_marker(&pair.alpha_state).to_bits(), 0.7_f32.to_bits());
    }

    #[test]
    fn multi_iteration_rejection_keeps_last_accepted_state_but_advances_schedule() {
        let now = Instant::now();
        let spec = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let thresholds = [0.0, 0.0];
        let alpha0 = alpha_fixture();
        let alpha0_bounds = Box::new(bounds(&[-4.0, -1.0], &[1.0, 1.0]));
        let alpha0_score = complete_score(&alpha0_bounds, &spec, &thresholds);
        let base_adam = AdamParams::new(0.2, 21);
        let policy = AtomicCudaMarginIterationsPolicy::new_at(
            3,
            base_adam,
            0.5,
            now + Duration::from_secs(2),
            now,
        )
        .expect("valid three-proposal policy");
        let mut observed_rows = Vec::new();
        let mut observed_markers = Vec::new();
        let mut observed_schedule = Vec::new();
        let mut proposal = 0usize;

        let choice = run_complete_multi_iterations_with_clock(
            &alpha0,
            alpha0_bounds,
            alpha0_score,
            &spec,
            &thresholds,
            false,
            base_adam,
            policy,
            false,
            |state, plan, adam, _deadline| {
                observed_rows.push(plan.binding_row);
                observed_markers.push(alpha_marker(state));
                observed_schedule.push((adam.t, adam.learning_rate));
                proposal += 1;
                Ok(match proposal {
                    1 => (
                        Box::new(bounds(&[-0.5, -2.0], &[1.0, 1.0])),
                        alpha_with_marker(0.6),
                    ),
                    2 => (
                        Box::new(bounds(&[-5.0, -5.0], &[1.0, 1.0])),
                        alpha_with_marker(0.9),
                    ),
                    3 => (
                        Box::new(bounds(&[-0.1, -0.2], &[1.0, 1.0])),
                        alpha_with_marker(0.7),
                    ),
                    _ => unreachable!("policy admits exactly three proposals"),
                })
            },
            || now,
        );

        assert_eq!(observed_rows, vec![0, 1, 1]);
        assert_eq!(
            observed_markers[0].to_bits(),
            alpha_marker(&alpha0).to_bits()
        );
        assert_eq!(observed_markers[1].to_bits(), 0.6_f32.to_bits());
        assert_eq!(
            observed_markers[2].to_bits(),
            0.6_f32.to_bits(),
            "a rejected proposal must not become the next gradient state"
        );
        assert_eq!(
            observed_schedule
                .iter()
                .map(|&(t, learning_rate)| (t, learning_rate.to_bits()))
                .collect::<Vec<_>>(),
            vec![
                (21, 0.2_f32.to_bits()),
                (22, 0.1_f32.to_bits()),
                (23, 0.05_f32.to_bits()),
            ],
            "the optimizer schedule advances across a rejected proposal"
        );

        let AtomicCudaMarginIterationsChoice::Candidate { pair, summary } = choice else {
            panic!("the first and third proposals must select the final whole pair");
        };
        assert_eq!(&*summary.binding_rows, &[0, 1, 1]);
        assert_eq!(summary.attempted_iterations, 3);
        assert_eq!(summary.accepted_iterations, 2);
        assert_eq!(summary.final_score.to_bits(), (-0.3_f32).to_bits());
        assert_eq!(pair.bounds.lower().as_slice(), Some(&[-0.1, -0.2][..]));
        assert_eq!(alpha_marker(&pair.alpha_state).to_bits(), 0.7_f32.to_bits());
    }

    #[test]
    fn multi_iteration_requires_strict_full_c_improvement_and_never_mixes_rows() {
        let now = Instant::now();
        let spec = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let thresholds = [0.0, 0.0];
        let base_adam = AdamParams::new(0.1, 1);
        let policy = AtomicCudaMarginIterationsPolicy::new_at(
            1,
            base_adam,
            1.0,
            now + Duration::from_secs(2),
            now,
        )
        .expect("valid policy");

        let alpha0 = alpha_fixture();
        let original = Box::new(bounds(&[-2.0, -1.0], &[1.0, 1.0]));
        let original_score = complete_score(&original, &spec, &thresholds);
        let equal_choice = run_complete_multi_iterations_with_clock(
            &alpha0,
            original,
            original_score,
            &spec,
            &thresholds,
            false,
            base_adam,
            policy,
            false,
            |_state, _binding, _adam, _deadline| {
                Ok((
                    Box::new(bounds(&[-1.0, -2.0], &[1.0, 1.0])),
                    alpha_with_marker(0.9),
                ))
            },
            || now,
        );
        let AtomicCudaMarginIterationsChoice::Alpha0 {
            bounds: retained_bounds,
            summary,
        } = equal_choice
        else {
            panic!("an equal complete-C hinge must retain alpha0");
        };
        assert_eq!(retained_bounds.lower().as_slice(), Some(&[-2.0, -1.0][..]));
        assert_eq!(summary.accepted_iterations, 0);

        let original = Box::new(bounds(&[-2.0, -1.0], &[1.0, 1.0]));
        let original_score = complete_score(&original, &spec, &thresholds);
        let improved_choice = run_complete_multi_iterations_with_clock(
            &alpha0,
            original,
            original_score,
            &spec,
            &thresholds,
            false,
            base_adam,
            policy,
            false,
            |_state, _binding, _adam, _deadline| {
                // Row 0 improves and row 1 regresses, while the complete hinge
                // improves from -3.0 to -1.6. Elementwise max mixing would
                // fabricate [-0.1, -1.0], which no alpha state certified.
                Ok((
                    Box::new(bounds(&[-0.1, -1.5], &[1.0, 1.0])),
                    alpha_with_marker(0.8),
                ))
            },
            || now,
        );
        let AtomicCudaMarginIterationsChoice::Candidate { pair, summary } = improved_choice else {
            panic!("the strict full-C improvement must transport its whole pair");
        };
        assert_eq!(pair.bounds.lower().as_slice(), Some(&[-0.1, -1.5][..]));
        assert_eq!(alpha_marker(&pair.alpha_state).to_bits(), 0.8_f32.to_bits());
        assert_eq!(summary.accepted_iterations, 1);
    }

    #[test]
    fn multi_iteration_hard_cap_executes_at_most_eight_proposals() {
        let now = Instant::now();
        let spec = arr2(&[[1.0]]);
        let thresholds = [0.0];
        let base_adam = AdamParams::new(0.1, 4);
        let policy = AtomicCudaMarginIterationsPolicy::new_at(
            ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS,
            base_adam,
            1.0,
            now + Duration::from_secs(2),
            now,
        )
        .expect("the cap is valid");
        let alpha0 = alpha_fixture();
        let initial = Box::new(bounds(&[-9.0], &[1.0]));
        let initial_score = complete_score(&initial, &spec, &thresholds);
        let mut calls = 0usize;
        let choice = run_complete_multi_iterations_with_clock(
            &alpha0,
            initial,
            initial_score,
            &spec,
            &thresholds,
            false,
            base_adam,
            policy,
            false,
            |_state, _binding, _adam, _deadline| {
                calls += 1;
                Ok((
                    Box::new(bounds(&[-9.0 + calls as f32], &[1.0])),
                    alpha_with_marker(calls as f32 / 10.0),
                ))
            },
            || now,
        );
        let AtomicCudaMarginIterationsChoice::Candidate { summary, .. } = choice else {
            panic!("all capped proposals strictly improve");
        };
        assert_eq!(calls, ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS);
        assert_eq!(
            summary.attempted_iterations,
            ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS
        );
        assert_eq!(
            summary.accepted_iterations,
            ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS
        );
        assert_eq!(
            summary.binding_rows.len(),
            ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS
        );
    }

    #[test]
    fn work_deadline_can_publish_alpha0_but_hard_authority_drops_any_pair() {
        let now = Instant::now();
        let spec = arr2(&[[1.0]]);
        let thresholds = [0.0];
        let base_adam = AdamParams::new(0.1, 1);
        let policy = AtomicCudaMarginIterationsPolicy::new_at(
            1,
            base_adam,
            1.0,
            now + Duration::from_secs(2),
            now,
        )
        .expect("valid reserved policy");
        let alpha0 = alpha_fixture();
        let initial = Box::new(bounds(&[-2.0], &[1.0]));
        let initial_score = complete_score(&initial, &spec, &thresholds);
        let mut evaluated = false;
        let choice = run_complete_multi_iterations_with_clock(
            &alpha0,
            initial,
            initial_score,
            &spec,
            &thresholds,
            false,
            base_adam,
            policy,
            false,
            |_state, _binding, _adam, _deadline| {
                evaluated = true;
                unreachable!("work cutoff must precede proposal evaluation")
            },
            || policy.work_deadline,
        );
        assert!(!evaluated);
        match finalize_multi_iterations_with_clock((), choice, policy.authority_deadline, || {
            policy.work_deadline
        }) {
            AtomicCudaMarginStepCommit::MultiAlpha0Selected {
                stop_refusal,
                attempted_iterations,
                multiplicative_weights_requested,
                multiplicative_weights_plan_dispatched,
                multiplicative_weights_effective,
                completed_proposals,
                adaptive_plan_dispatches,
                gradient_plan_num_specs,
                gradient_row_count,
                ..
            } => {
                assert_eq!(attempted_iterations, 0);
                assert!(!multiplicative_weights_requested);
                assert!(!multiplicative_weights_plan_dispatched);
                assert!(!multiplicative_weights_effective);
                assert_eq!(completed_proposals, 0);
                assert_eq!(adaptive_plan_dispatches, 0);
                assert_eq!(gradient_plan_num_specs, None);
                assert_eq!(gradient_row_count, 1);
                assert_eq!(
                    stop_refusal,
                    Some(AtomicCudaMarginStepRefusal::WorkDeadlineExceeded)
                );
            }
            _ => panic!("the reserve must permit publishing the existing alpha0 pair"),
        }

        let initial = Box::new(bounds(&[-2.0], &[1.0]));
        let initial_score = complete_score(&initial, &spec, &thresholds);
        let candidate = run_complete_multi_iterations_with_clock(
            &alpha0,
            initial,
            initial_score,
            &spec,
            &thresholds,
            false,
            base_adam,
            policy,
            false,
            |_state, _binding, _adam, _deadline| {
                Ok((Box::new(bounds(&[-1.0], &[1.0])), alpha_with_marker(0.8)))
            },
            || now,
        );
        assert!(matches!(
            finalize_multi_iterations_with_clock((), candidate, policy.authority_deadline, || {
                policy.authority_deadline
            },),
            AtomicCudaMarginStepCommit::DeadlineExceeded
        ));
    }

    #[test]
    fn any_partial_or_complete_spec_axis_state_is_refused() {
        let mut alpha = alpha_fixture();
        assert!(!alpha_state_has_spec_axis(&alpha));
        alpha.spec_slot_rows.push(0);
        assert!(alpha_state_has_spec_axis(&alpha));

        let mut optimizer_only = alpha_fixture();
        optimizer_only.spec_adam_m.insert(
            "relu_a".to_string(),
            Array2::zeros((1, optimizer_only.relu_len("relu_a").expect("fixture width"))),
        );
        assert!(alpha_state_has_spec_axis(&optimizer_only));
    }

    #[test]
    fn bracket_policy_is_fixed_bounded_and_reserves_publication_time() {
        let now = Instant::now();
        let hard_deadline = now + Duration::from_secs(2);
        let next_decayed_learning_rate = 0.25 * 0.98_f32.powf(20.0);
        let policy = AtomicCudaMarginBracketPolicy::new_at(
            AdamParams::new(next_decayed_learning_rate, 21),
            0.25,
            hard_deadline,
            now,
        )
        .expect("CIFAR configured base LR is admissible");
        assert_eq!(
            policy.candidate_lrs.map(f32::to_bits),
            [0.25_f32 * 0.3, 0.25, 0.5].map(f32::to_bits)
        );
        assert_ne!(
            policy.candidate_lrs[1].to_bits(),
            next_decayed_learning_rate.to_bits(),
            "the bracket must use the configured base, not the dark single-step decay"
        );
        assert_eq!(
            policy.work_deadline,
            hard_deadline
                .checked_sub(ATOMIC_CUDA_MARGIN_PUBLICATION_RESERVE)
                .expect("the test deadline admits the reserve")
        );
        assert_eq!(policy.authority_deadline, hard_deadline);

        for learning_rate in [0.0, -0.1, f32::NAN, f32::INFINITY] {
            assert_eq!(
                AtomicCudaMarginBracketPolicy::new_at(
                    AdamParams::new(learning_rate, 1),
                    0.25,
                    hard_deadline,
                    now
                )
                .unwrap_err(),
                AtomicCudaMarginStepRefusal::InvalidAdamParams,
                "base={learning_rate:?}"
            );
        }
        for configured_base in [0.0, -0.1, f32::NAN, f32::INFINITY] {
            assert_eq!(
                AtomicCudaMarginBracketPolicy::new_at(
                    AdamParams::new(0.1, 1),
                    configured_base,
                    hard_deadline,
                    now,
                )
                .unwrap_err(),
                AtomicCudaMarginStepRefusal::InvalidLearningRateBracket,
                "configured base={configured_base:?}"
            );
        }
        assert_eq!(
            AtomicCudaMarginBracketPolicy::new_at(
                AdamParams::new(0.1, 1),
                f32::from_bits(0.25_f32.to_bits() + 1),
                hard_deadline,
                now,
            )
            .unwrap_err(),
            AtomicCudaMarginStepRefusal::InvalidLearningRateBracket
        );
        assert_eq!(
            AtomicCudaMarginBracketPolicy::new_at(
                AdamParams::new(0.25, 1),
                0.25,
                now + ATOMIC_CUDA_MARGIN_PUBLICATION_RESERVE,
                now,
            )
            .unwrap_err(),
            AtomicCudaMarginStepRefusal::DeadlineExceeded
        );
    }

    #[test]
    fn topk_policy_is_fixed_bounded_and_reserves_publication_time() {
        let now = Instant::now();
        let hard_deadline = now + Duration::from_secs(2);
        let policy =
            AtomicCudaMarginTopKPolicy::new_at(AdamParams::new(0.1, 21), 0.25, hard_deadline, now)
                .expect("valid fixed top-K policy");
        assert_eq!(policy.learning_rate.to_bits(), 0.25_f32.to_bits());
        assert_eq!(
            policy.work_deadline,
            hard_deadline
                .checked_sub(ATOMIC_CUDA_MARGIN_PUBLICATION_RESERVE)
                .expect("publication reserve")
        );
        assert_eq!(policy.authority_deadline, hard_deadline);

        for base in [0.0, -0.1, f32::NAN, f32::INFINITY] {
            assert_eq!(
                AtomicCudaMarginTopKPolicy::new_at(
                    AdamParams::new(0.1, 21),
                    base,
                    hard_deadline,
                    now,
                )
                .unwrap_err(),
                AtomicCudaMarginStepRefusal::InvalidTopKPolicy
            );
        }
        assert_eq!(
            AtomicCudaMarginTopKPolicy::new_at(
                AdamParams::new(0.1, 21),
                f32::from_bits(0.25_f32.to_bits() + 1),
                hard_deadline,
                now,
            )
            .unwrap_err(),
            AtomicCudaMarginStepRefusal::InvalidTopKPolicy
        );
        assert_eq!(
            AtomicCudaMarginTopKPolicy::new_at(
                AdamParams::new(0.1, 21),
                0.25,
                now + ATOMIC_CUDA_MARGIN_PUBLICATION_RESERVE,
                now,
            )
            .unwrap_err(),
            AtomicCudaMarginStepRefusal::DeadlineExceeded
        );
    }

    #[test]
    fn topk_objective_is_deterministic_flattened_and_direction_aware() {
        let deadline = Instant::now() + Duration::from_secs(10);
        let spec = Array2::from_shape_fn((10, 3), |(row, column)| {
            row as f32 * 10.0 + column as f32 + 0.25
        });
        let expected_rows = vec![6, 7, 9, 1, 3, 8, 2, 5];

        let lower = [-1.0, -5.0, -3.0, -5.0, 1.0, -2.0, -8.0, -7.0, -4.0, -6.0];
        let upper: Vec<f32> = lower.iter().map(|value| value + 0.5).collect();
        let lower_plan =
            prepare_topk_objective(&bounds(&lower, &upper), &spec, &[0.0; 10], false, deadline)
                .expect("lower-margin top-K");
        assert_eq!(lower_plan.row_indices, expected_rows);
        let expected_lower: Vec<f32> = expected_rows
            .iter()
            .flat_map(|&row| spec.row(row).to_vec())
            .collect();
        assert_eq!(lower_plan.lower_objectives, expected_lower);

        let upper = [1.0, 5.0, 3.0, 5.0, -1.0, 2.0, 8.0, 7.0, 4.0, 6.0];
        let lower: Vec<f32> = upper.iter().map(|value| value - 0.5).collect();
        let upper_plan =
            prepare_topk_objective(&bounds(&lower, &upper), &spec, &[0.0; 10], true, deadline)
                .expect("upper-margin top-K");
        assert_eq!(upper_plan.row_indices, expected_rows);
        let expected_upper: Vec<f32> = expected_rows
            .iter()
            .flat_map(|&row| {
                spec.row(row)
                    .iter()
                    .map(|value| -*value)
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(upper_plan.lower_objectives, expected_upper);
        assert_eq!(
            upper_plan.lower_objectives.len(),
            ATOMIC_CUDA_MARGIN_TOPK * spec.ncols()
        );
    }

    #[test]
    fn joint_objective_shape_enforces_flattening_and_bounded_mw_cap() {
        assert_eq!(validate_joint_objective_shape(&[1.0; 24], 8, 3), Ok(()));
        assert_eq!(
            validate_joint_objective_shape(
                &vec![1.0; ATOMIC_CUDA_MARGIN_MW_MAX_ROWS * 3],
                ATOMIC_CUDA_MARGIN_MW_MAX_ROWS,
                3,
            ),
            Ok(())
        );
        for (values, specs, output_dim) in [
            (vec![1.0; 24], 0, 3),
            (
                vec![1.0; (ATOMIC_CUDA_MARGIN_MW_MAX_ROWS + 1) * 3],
                ATOMIC_CUDA_MARGIN_MW_MAX_ROWS + 1,
                3,
            ),
            (vec![1.0; 23], 8, 3),
            (vec![1.0; 24], 8, 0),
            (vec![f32::NAN; 24], 8, 3),
        ] {
            assert_eq!(
                validate_joint_objective_shape(&values, specs, output_dim),
                Err(AtomicCudaMarginStepRefusal::JointMapping)
            );
        }
    }

    #[test]
    fn bounded_joint_chunks_add_gradients_and_refuse_malformed_results() {
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut host_work = 0usize;
        let mut accumulated = vec![vec![1.0, 2.0], vec![3.0]];
        accumulate_joint_gradient_chunk(
            &mut accumulated,
            vec![vec![0.5, -0.5], vec![4.0]],
            deadline,
            &mut host_work,
        )
        .expect("matching finite chunks add exactly");
        assert_eq!(accumulated, vec![vec![1.5, 1.5], vec![7.0]]);
        assert_eq!(host_work, 3);

        assert_eq!(
            accumulate_joint_gradient_chunk(
                &mut accumulated,
                vec![vec![1.0]],
                deadline,
                &mut host_work,
            ),
            Err(AtomicCudaMarginStepRefusal::JointMapping)
        );
        assert_eq!(
            accumulate_joint_gradient_chunk(
                &mut [vec![f32::MAX]],
                vec![vec![f32::MAX]],
                deadline,
                &mut host_work,
            ),
            Err(AtomicCudaMarginStepRefusal::JointNonFinite)
        );
    }

    #[test]
    fn bracket_alpha_candidates_are_all_derived_from_the_same_alpha0() {
        let now = Instant::now();
        let (alpha0, relu_names, policy) = bracket_fixture(now);
        let gradients = vec![vec![2.0], vec![4.0, -4.0]];
        let before = alpha0.alpha("relu_a").cloned();
        let candidates = prepare_complete_lr_bracket(
            &alpha0,
            &relu_names,
            &gradients,
            AdamParams::new(0.25, 1),
            policy,
        )
        .expect("complete bracket");

        assert_eq!(candidates.len(), 3);
        for candidate in &candidates {
            let direct = apply_checked_adam_step(
                &alpha0,
                &relu_names,
                &gradients,
                AdamParams::new(candidate.learning_rate, 1),
                policy.work_deadline,
            )
            .expect("independent direct candidate");
            assert_eq!(
                candidate.alpha_state.alpha("relu_a"),
                direct.alpha("relu_a"),
                "ordinal {} must not inherit an earlier candidate",
                candidate.ordinal
            );
            assert_eq!(
                candidate.alpha_state.alpha_upper("relu_a"),
                direct.alpha_upper("relu_a")
            );
        }
        assert_eq!(alpha0.alpha("relu_a").cloned(), before);
        assert_ne!(
            candidates[0].alpha_state.alpha("relu_a"),
            candidates[2].alpha_state.alpha("relu_a")
        );
    }

    #[test]
    fn unique_strict_best_selects_its_whole_state_bounds_pair() {
        let now = Instant::now();
        let (alpha0, relu_names, policy) = bracket_fixture(now);
        let candidates = complete_candidate_pairs(&alpha0, &relu_names, policy, [-8.0, -3.0, -6.0]);
        let expected_alpha = candidates[1]
            .alpha_state
            .alpha("relu_a")
            .expect("candidate alpha")
            .clone();
        let choice = select_test_bracket(&alpha0, &relu_names, policy, -9.0, candidates, || now)
            .expect("complete bracket selects");

        let AtomicCudaMarginBracketChoice::Candidate(selected) = choice else {
            panic!("strict best candidate must be selected");
        };
        assert_eq!(selected.ordinal, 1);
        assert_eq!(selected.learning_rate.to_bits(), 0.25_f32.to_bits());
        assert_eq!(selected.score.to_bits(), (-3.0_f32).to_bits());
        assert_eq!(selected.bounds.lower()[0].to_bits(), (-3.0_f32).to_bits());
        assert_eq!(
            selected.alpha_state.alpha("relu_a"),
            Some(&expected_alpha),
            "the chosen bound must carry only its own exact alpha state"
        );
    }

    #[test]
    fn any_best_tie_or_non_improvement_retains_alpha0() {
        let now = Instant::now();
        let (alpha0, relu_names, policy) = bracket_fixture(now);
        for (alpha0_score, scores) in [
            (-9.0, [-8.0, -3.0, -3.0]),
            (-3.0, [-8.0, -3.0, -6.0]),
            (-2.0, [-8.0, -3.0, -6.0]),
        ] {
            let candidates = complete_candidate_pairs(&alpha0, &relu_names, policy, scores);
            let choice = select_test_bracket(
                &alpha0,
                &relu_names,
                policy,
                alpha0_score,
                candidates,
                || now,
            )
            .expect("tie/non-improvement is a valid alpha0 selection");
            let AtomicCudaMarginBracketChoice::Alpha0 {
                best_candidate_score,
            } = choice
            else {
                panic!("ties and regressions must retain alpha0");
            };
            assert_eq!(
                best_candidate_score.to_bits(),
                scores
                    .into_iter()
                    .fold(f32::NEG_INFINITY, f32::max)
                    .to_bits()
            );
        }
    }

    #[test]
    fn partial_or_malformed_bracket_refuses_before_candidate_selection() {
        let now = Instant::now();
        let (alpha0, relu_names, policy) = bracket_fixture(now);

        let mut partial =
            complete_candidate_pairs(&alpha0, &relu_names, policy, [-8.0, -3.0, -6.0]);
        partial.pop();
        assert_eq!(
            select_test_bracket(&alpha0, &relu_names, policy, -9.0, partial, || now).unwrap_err(),
            AtomicCudaMarginStepRefusal::IncompleteLearningRateBracket
        );

        let mut bad_score =
            complete_candidate_pairs(&alpha0, &relu_names, policy, [-8.0, -3.0, -6.0]);
        bad_score[2].score = f32::NAN;
        assert_eq!(
            select_test_bracket(&alpha0, &relu_names, policy, -9.0, bad_score, || now).unwrap_err(),
            AtomicCudaMarginStepRefusal::MalformedLearningRateCandidate
        );

        let mut wrong_pairing =
            complete_candidate_pairs(&alpha0, &relu_names, policy, [-8.0, -3.0, -6.0]);
        *wrong_pairing[0].bounds = bounds(&[-7.0], &[-6.5]);
        assert_eq!(
            select_test_bracket(&alpha0, &relu_names, policy, -9.0, wrong_pairing, || now)
                .unwrap_err(),
            AtomicCudaMarginStepRefusal::MalformedLearningRateCandidate
        );

        let mut bad_state =
            complete_candidate_pairs(&alpha0, &relu_names, policy, [-8.0, -3.0, -6.0]);
        bad_state[1].alpha_state.alphas.get_mut("relu_a").unwrap()[0] = f32::NAN;
        assert_eq!(
            select_test_bracket(&alpha0, &relu_names, policy, -9.0, bad_state, || now).unwrap_err(),
            AtomicCudaMarginStepRefusal::MalformedLearningRateCandidate
        );

        let mut wrong_schedule =
            complete_candidate_pairs(&alpha0, &relu_names, policy, [-8.0, -3.0, -6.0]);
        wrong_schedule[1].learning_rate =
            f32::from_bits(wrong_schedule[1].learning_rate.to_bits() + 1);
        assert_eq!(
            select_test_bracket(&alpha0, &relu_names, policy, -9.0, wrong_schedule, || now)
                .unwrap_err(),
            AtomicCudaMarginStepRefusal::MalformedLearningRateCandidate
        );
    }

    #[test]
    fn malformed_selector_performs_post_drop_work_and_authority_polls() {
        let now = Instant::now();
        let (alpha0, relu_names, policy) = bracket_fixture(now);
        let mut candidates =
            complete_candidate_pairs(&alpha0, &relu_names, policy, [-8.0, -3.0, -6.0]);
        candidates[0].score = f32::NAN;
        let calls = Cell::new(0usize);
        let result = select_test_bracket(&alpha0, &relu_names, policy, -9.0, candidates, || {
            calls.set(calls.get() + 1);
            now
        });
        assert_eq!(
            result.unwrap_err(),
            AtomicCudaMarginStepRefusal::MalformedLearningRateCandidate
        );
        assert_eq!(
            calls.get(),
            4,
            "initial/candidate polls must be followed by two post-drop authority polls"
        );
    }

    #[test]
    fn deadline_before_or_after_candidate_destruction_refuses_the_whole_bracket() {
        let now = Instant::now();
        let (alpha0, relu_names, policy) = bracket_fixture(now);
        let candidates = complete_candidate_pairs(&alpha0, &relu_names, policy, [-8.0, -3.0, -6.0]);
        assert_eq!(
            select_test_bracket(&alpha0, &relu_names, policy, -9.0, candidates, || policy
                .work_deadline)
            .unwrap_err(),
            AtomicCudaMarginStepRefusal::DeadlineExceeded
        );

        let candidates = complete_candidate_pairs(&alpha0, &relu_names, policy, [-8.0, -3.0, -6.0]);
        let calls = Cell::new(0usize);
        let result = select_test_bracket(&alpha0, &relu_names, policy, -9.0, candidates, || {
            let call = calls.get();
            calls.set(call + 1);
            if call < 4 {
                now
            } else {
                policy.work_deadline
            }
        });
        assert_eq!(
            result.unwrap_err(),
            AtomicCudaMarginStepRefusal::DeadlineExceeded
        );
        assert_eq!(
            calls.get(),
            6,
            "expiry drops the selected pair and polls again before alpha0 publication"
        );

        let candidates = complete_candidate_pairs(&alpha0, &relu_names, policy, [-8.0, -3.0, -6.0]);
        let calls = Cell::new(0usize);
        let result = select_test_bracket(&alpha0, &relu_names, policy, -9.0, candidates, || {
            let call = calls.get();
            calls.set(call + 1);
            if call < 5 {
                now
            } else {
                policy.authority_deadline
            }
        });
        assert_eq!(
            result.unwrap_err(),
            AtomicCudaMarginStepRefusal::DeadlineExceeded
        );
        assert_eq!(
            calls.get(),
            7,
            "authority expiry drops the selected pair and polls again before alpha0 publication"
        );
    }

    #[test]
    fn refusal_cleanup_drops_owned_state_before_its_final_authority_polls() {
        let now = Instant::now();
        let (_, _, policy) = bracket_fixture(now);
        let dropped = Cell::new(false);
        let calls = Cell::new(0usize);
        let mut clock = || {
            assert!(
                dropped.get(),
                "candidate-owned state must be destroyed before the cleanup poll"
            );
            calls.set(calls.get() + 1);
            now
        };
        let refusal = bracket_refusal_after_drop_with_clock(
            DropFlag(&dropped),
            policy,
            AtomicCudaMarginStepRefusal::MalformedLearningRateCandidate,
            &mut clock,
        );
        assert_eq!(
            refusal,
            AtomicCudaMarginStepRefusal::MalformedLearningRateCandidate
        );
        assert_eq!(calls.get(), 2, "both work and authority remain live");
    }

    #[test]
    fn bracket_error_commit_keeps_exact_alpha0_and_never_exports_state() {
        let now = Instant::now();
        let (_, _, policy) = bracket_fixture(now);
        let alpha0_bounds = Box::new(bounds(&[-9.0], &[2.0]));
        let alpha0_ptr = std::ptr::from_ref(alpha0_bounds.as_ref());
        let lower_bits: Vec<u32> = alpha0_bounds.lower().iter().map(|v| v.to_bits()).collect();
        let upper_bits: Vec<u32> = alpha0_bounds.upper().iter().map(|v| v.to_bits()).collect();
        let scratch_dropped = Cell::new(false);
        let commit = finalize_complete_lr_bracket_with_clock(
            DropFlag(&scratch_dropped),
            alpha0_bounds,
            7,
            -9.0,
            Err(AtomicCudaMarginStepRefusal::IncompleteLearningRateBracket),
            policy.authority_deadline,
            || {
                assert!(
                    scratch_dropped.get(),
                    "joint-gradient scratch must be dropped before publication authority"
                );
                now
            },
        );
        let AtomicCudaMarginStepCommit::Alpha0Retained { bounds, refusal } = commit else {
            panic!("a failed bracket must commit only alpha0");
        };
        assert!(
            std::ptr::eq(alpha0_ptr, bounds.as_ref()),
            "live refusal must move the existing certified alpha0 allocation"
        );
        assert_eq!(
            refusal,
            AtomicCudaMarginStepRefusal::IncompleteLearningRateBracket
        );
        assert_eq!(
            bounds
                .lower()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            lower_bits
        );
        assert_eq!(
            bounds
                .upper()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            upper_bits
        );
    }

    #[test]
    fn observed_rows_deadline_refusals_never_publish_committed_alpha0() {
        let now = Instant::now();
        let authority_deadline = now + Duration::from_secs(10);
        for refusal in [
            AtomicCudaMarginStepRefusal::DeadlineExceeded,
            AtomicCudaMarginStepRefusal::Alpha0Rows(AtomicCudaRowsRefusal::DeadlineExceeded),
            AtomicCudaMarginStepRefusal::Alpha1Rows(AtomicCudaRowsRefusal::DeadlineExceeded),
        ] {
            let scratch_dropped = Cell::new(false);
            let calls = Cell::new(0usize);
            let commit = finalize_alpha0_retained_with_clock(
                DropFlag(&scratch_dropped),
                Box::new(bounds(&[-9.0], &[2.0])),
                refusal,
                authority_deadline,
                || {
                    assert!(
                        scratch_dropped.get(),
                        "all scratch must be torn down before deadline routing"
                    );
                    calls.set(calls.get() + 1);
                    now
                },
            );
            assert!(matches!(
                commit,
                AtomicCudaMarginStepCommit::DeadlineExceeded
            ));
            assert_eq!(
                calls.get(),
                1,
                "an already-observed expiry stays typed after teardown"
            );
        }
    }

    #[test]
    fn bracket_alpha0_selection_moves_existing_box_after_live_authority_poll() {
        let now = Instant::now();
        let authority_deadline = now + Duration::from_secs(10);
        let alpha0_bounds = Box::new(bounds(&[-9.0], &[2.0]));
        let alpha0_ptr = std::ptr::from_ref(alpha0_bounds.as_ref());
        let commit = finalize_complete_lr_bracket_with_clock(
            (),
            alpha0_bounds,
            7,
            -9.0,
            Ok(AtomicCudaMarginBracketChoice::Alpha0 {
                best_candidate_score: -10.0,
            }),
            authority_deadline,
            || now,
        );
        let AtomicCudaMarginStepCommit::Alpha0Selected { bounds, .. } = commit else {
            panic!("a live non-improving bracket must select alpha0");
        };
        assert!(
            std::ptr::eq(alpha0_ptr, bounds.as_ref()),
            "legacy/bracket alpha0 publication must not allocate after its final poll"
        );
    }

    #[test]
    fn topk_commit_moves_one_whole_certified_state_bounds_pair() {
        let now = Instant::now();
        let policy = AtomicCudaMarginTopKPolicy::new_at(
            AdamParams::new(0.1, 21),
            0.25,
            now + Duration::from_secs(10),
            now,
        )
        .expect("top-K policy");
        let mut selected_state = alpha_fixture();
        selected_state.alphas.get_mut("relu_a").unwrap()[0] = 0.75;
        let expected_alpha = selected_state.alphas["relu_a"][0].to_bits();
        let scratch_dropped = Cell::new(false);
        let commit = finalize_topk_with_clock(
            DropFlag(&scratch_dropped),
            Box::new(bounds(&[-9.0], &[2.0])),
            -9.0,
            Ok(AtomicCudaMarginTopKChoice::Candidate {
                pair: AtomicCudaMarginCertifiedPair {
                    ordinal: 0,
                    learning_rate: 0.25,
                    score: -3.0,
                    bounds: Box::new(bounds(&[-3.0], &[-2.5])),
                    alpha_state: Box::new(selected_state),
                },
                row_indices: vec![6, 7, 9].into_boxed_slice(),
            }),
            policy.authority_deadline,
            || {
                assert!(scratch_dropped.get());
                now
            },
        );
        let AtomicCudaMarginStepCommit::TopKAlpha1Selected {
            bounds,
            alpha_state,
            row_indices,
            alpha0_score,
            alpha1_score,
        } = commit
        else {
            panic!("strict top-K candidate must move as one pair");
        };
        assert_eq!(bounds.lower()[0].to_bits(), (-3.0_f32).to_bits());
        assert_eq!(alpha_state.alphas["relu_a"][0].to_bits(), expected_alpha);
        assert_eq!(&*row_indices, &[6, 7, 9]);
        assert_eq!(alpha0_score.to_bits(), (-9.0_f32).to_bits());
        assert_eq!(alpha1_score.to_bits(), (-3.0_f32).to_bits());
    }

    #[test]
    fn topk_post_packaging_work_expiry_refuses_both_complete_choices() {
        let now = Instant::now();
        let work_deadline = now + Duration::from_secs(1);
        for choice in [
            AtomicCudaMarginTopKChoice::Candidate {
                pair: AtomicCudaMarginCertifiedPair {
                    ordinal: 0,
                    learning_rate: 0.25,
                    score: -3.0,
                    bounds: Box::new(bounds(&[-3.0], &[-2.5])),
                    alpha_state: Box::new(alpha_fixture()),
                },
                row_indices: vec![6, 7, 9].into_boxed_slice(),
            },
            AtomicCudaMarginTopKChoice::Alpha0 {
                candidate_score: -10.0,
                row_indices: vec![6, 7, 9].into_boxed_slice(),
            },
        ] {
            let calls = Cell::new(0usize);
            let refusal = admit_topk_choice_with_clock(choice, work_deadline, || {
                calls.set(calls.get() + 1);
                work_deadline
            })
            .expect_err("a choice packaged at the work cutoff must not be admitted");
            assert_eq!(refusal, AtomicCudaMarginStepRefusal::DeadlineExceeded);
            assert_eq!(calls.get(), 1, "the packaging boundary has one clock seam");
        }
    }

    #[test]
    fn topk_post_packaging_live_admission_moves_both_complete_choices() {
        let now = Instant::now();
        let work_deadline = now + Duration::from_secs(1);

        let candidate_bounds = Box::new(bounds(&[-3.0], &[-2.5]));
        let candidate_bounds_ptr = std::ptr::from_ref(candidate_bounds.as_ref());
        let candidate_alpha = Box::new(alpha_fixture());
        let candidate_alpha_ptr = std::ptr::from_ref(candidate_alpha.as_ref());
        let candidate_rows = vec![6, 7, 9].into_boxed_slice();
        let candidate_rows_ptr = candidate_rows.as_ptr();
        let admitted = admit_topk_choice_with_clock(
            AtomicCudaMarginTopKChoice::Candidate {
                pair: AtomicCudaMarginCertifiedPair {
                    ordinal: 0,
                    learning_rate: 0.25,
                    score: -3.0,
                    bounds: candidate_bounds,
                    alpha_state: candidate_alpha,
                },
                row_indices: candidate_rows,
            },
            work_deadline,
            || now,
        )
        .expect("a choice packaged before the work cutoff must be admitted");
        let AtomicCudaMarginTopKChoice::Candidate { pair, row_indices } = admitted else {
            panic!("the complete candidate choice must be preserved");
        };
        assert!(std::ptr::eq(candidate_bounds_ptr, pair.bounds.as_ref()));
        assert!(std::ptr::eq(candidate_alpha_ptr, pair.alpha_state.as_ref()));
        assert_eq!(candidate_rows_ptr, row_indices.as_ptr());

        let alpha0_rows = vec![11, 13].into_boxed_slice();
        let alpha0_rows_ptr = alpha0_rows.as_ptr();
        let admitted = admit_topk_choice_with_clock(
            AtomicCudaMarginTopKChoice::Alpha0 {
                candidate_score: -10.0,
                row_indices: alpha0_rows,
            },
            work_deadline,
            || now,
        )
        .expect("a choice packaged before the work cutoff must be admitted");
        let AtomicCudaMarginTopKChoice::Alpha0 { row_indices, .. } = admitted else {
            panic!("the complete alpha0 choice must be preserved");
        };
        assert_eq!(alpha0_rows_ptr, row_indices.as_ptr());
    }

    #[test]
    fn topk_non_improvement_publishes_exact_alpha0_and_rows() {
        let now = Instant::now();
        let alpha0_bounds = Box::new(bounds(&[-9.0], &[2.0]));
        let alpha0_ptr = std::ptr::from_ref(alpha0_bounds.as_ref());
        let lower_bits: Vec<u32> = alpha0_bounds.lower().iter().map(|v| v.to_bits()).collect();
        let upper_bits: Vec<u32> = alpha0_bounds.upper().iter().map(|v| v.to_bits()).collect();
        let commit = finalize_topk_with_clock(
            (),
            alpha0_bounds,
            -9.0,
            Ok(AtomicCudaMarginTopKChoice::Alpha0 {
                candidate_score: -10.0,
                row_indices: vec![6, 7, 9].into_boxed_slice(),
            }),
            now + Duration::from_secs(10),
            || now,
        );
        let AtomicCudaMarginStepCommit::TopKAlpha0Selected {
            bounds,
            row_indices,
            alpha0_score,
            alpha1_score,
        } = commit
        else {
            panic!("top-K non-improvement must retain alpha0");
        };
        assert!(
            std::ptr::eq(alpha0_ptr, bounds.as_ref()),
            "live alpha0 selection must not allocate after its authority poll"
        );
        assert_eq!(
            bounds
                .lower()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            lower_bits
        );
        assert_eq!(
            bounds
                .upper()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            upper_bits
        );
        assert_eq!(&*row_indices, &[6, 7, 9]);
        assert_eq!(alpha0_score.to_bits(), (-9.0_f32).to_bits());
        assert_eq!(alpha1_score.to_bits(), (-10.0_f32).to_bits());
    }

    #[test]
    fn topk_post_cleanup_expiry_commits_typed_deadline_without_state() {
        let now = Instant::now();
        let authority_deadline = now + Duration::from_secs(10);
        let scratch_dropped = Cell::new(false);
        let calls = Cell::new(0usize);
        let commit = finalize_topk_with_clock(
            DropFlag(&scratch_dropped),
            Box::new(bounds(&[-9.0], &[2.0])),
            -9.0,
            Ok(AtomicCudaMarginTopKChoice::Candidate {
                pair: AtomicCudaMarginCertifiedPair {
                    ordinal: 0,
                    learning_rate: 0.25,
                    score: -3.0,
                    bounds: Box::new(bounds(&[-3.0], &[-2.5])),
                    alpha_state: Box::new(alpha_fixture()),
                },
                row_indices: vec![6, 7, 9].into_boxed_slice(),
            }),
            authority_deadline,
            || {
                assert!(scratch_dropped.get());
                calls.set(calls.get() + 1);
                authority_deadline
            },
        );
        assert!(matches!(
            commit,
            AtomicCudaMarginStepCommit::DeadlineExceeded
        ));
        assert_eq!(
            calls.get(),
            2,
            "expiry must be observed again after the selected top-K pair is dropped"
        );
    }

    #[test]
    fn outer_candidate_cleanup_expiry_commits_typed_deadline_without_state() {
        let now = Instant::now();
        let (alpha0, relu_names, policy) = bracket_fixture(now);
        let candidate = complete_candidate_pairs(&alpha0, &relu_names, policy, [-8.0, -3.0, -6.0])
            .into_iter()
            .next()
            .expect("candidate");
        let scratch_dropped = Cell::new(false);
        let calls = Cell::new(0usize);
        let commit = finalize_complete_lr_bracket_with_clock(
            DropFlag(&scratch_dropped),
            Box::new(bounds(&[-9.0], &[2.0])),
            7,
            -9.0,
            Ok(AtomicCudaMarginBracketChoice::Candidate(candidate)),
            policy.authority_deadline,
            || {
                assert!(
                    scratch_dropped.get(),
                    "joint-gradient scratch must precede every authority observation"
                );
                calls.set(calls.get() + 1);
                policy.authority_deadline
            },
        );
        assert!(matches!(
            commit,
            AtomicCudaMarginStepCommit::DeadlineExceeded
        ));
        assert_eq!(
            calls.get(),
            2,
            "expiry must be observed again after the selected candidate is dropped"
        );
    }

    #[test]
    fn lower_mode_scores_complete_hinge_and_selects_first_worst_tie() {
        let spec = arr2(&[[1.0, -1.0], [0.5, 0.5], [-1.0, 1.0]]);
        let scored = score_complete_c(
            &bounds(&[-2.0, -2.0, 3.0], &[-1.0, -1.0, 4.0]),
            &spec,
            &[0.0, 0.0, 0.0],
            false,
            Instant::now() + Duration::from_secs(10),
        )
        .expect("score");
        assert_eq!(scored.hinge, -4.0);
        assert_eq!(
            scored.binding,
            Some(BindingRow {
                index: 0,
                slack: -2.0,
                lower_objective: vec![1.0, -1.0],
            })
        );
    }

    #[test]
    fn independently_certified_rows_can_be_omitted_without_changing_active_hinge() {
        let full_spec = arr2(&[[1.0_f32, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, -1.0]]);
        let deadline = Instant::now() + Duration::from_secs(10);
        let full = score_complete_c(
            &bounds(&[1.0, -2.0, 1.0, -1.0], &[2.0, -1.0, 2.0, 0.0]),
            &full_spec,
            &[0.0, 0.0, 0.0, 0.0],
            false,
            deadline,
        )
        .expect("full selected-row score");

        // Source rows 0 and 2 have strictly positive certified slack, so their
        // hinge terms are exactly zero. Compact source rows [1, 3] preserve the
        // full hinge while expressing their binding row in compact coordinates.
        let compact_spec = arr2(&[[0.0_f32, 1.0], [1.0, -1.0]]);
        let compact = score_complete_c(
            &bounds(&[-2.0, -1.0], &[-1.0, 0.0]),
            &compact_spec,
            &[0.0, 0.0],
            false,
            deadline,
        )
        .expect("compact selected-row score");

        assert_eq!(full.hinge.to_bits(), compact.hinge.to_bits());
        assert_eq!(full.hinge, -3.0);
        assert_eq!(full.binding.as_ref().map(|binding| binding.index), Some(1));
        assert_eq!(
            compact.binding.as_ref().map(|binding| binding.index),
            Some(0)
        );
        assert_eq!(
            full.binding
                .as_ref()
                .map(|binding| binding.lower_objective.as_slice()),
            compact
                .binding
                .as_ref()
                .map(|binding| binding.lower_objective.as_slice())
        );
    }

    #[test]
    fn upper_mode_sign_flips_the_binding_objective() {
        let spec = arr2(&[[1.0, -2.0], [3.0, 4.0]]);
        let scored = score_complete_c(
            &bounds(&[-3.0, -2.0], &[2.0, 1.0]),
            &spec,
            &[0.0, 2.0],
            true,
            Instant::now() + Duration::from_secs(10),
        )
        .expect("score");
        assert_eq!(scored.hinge, -2.0);
        assert_eq!(
            scored.binding,
            Some(BindingRow {
                index: 0,
                slack: -2.0,
                lower_objective: vec![-1.0, 2.0],
            })
        );
    }

    #[test]
    fn fully_verified_vector_has_no_binding_gradient_authority() {
        let spec = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let scored = score_complete_c(
            &bounds(&[0.5, 1.5], &[1.0, 2.0]),
            &spec,
            &[0.0, 1.0],
            false,
            Instant::now() + Duration::from_secs(10),
        )
        .expect("score");
        assert_eq!(scored.hinge, 0.0);
        assert!(scored.binding.is_none());
    }

    #[test]
    fn checked_adam_is_private_clamped_and_updates_exact_aligned_state() {
        let alpha0 = alpha_fixture();
        let names = vec!["relu_b".to_string(), "relu_a".to_string()];
        let gradients = vec![vec![2.0], vec![4.0, -4.0]];
        let alpha1 = apply_checked_adam_step(
            &alpha0,
            &names,
            &gradients,
            AdamParams::new(0.1, 1),
            Instant::now() + Duration::from_secs(10),
        )
        .expect("checked update");

        assert_eq!(alpha0.alpha("relu_b"), Some(&arr1(&[1.0])));
        assert_eq!(alpha1.alpha("relu_b"), Some(&arr1(&[1.0])));
        assert_eq!(alpha0.alpha("relu_a"), Some(&arr1(&[0.0, 1.0])));
        let updated = alpha1.alpha("relu_a").expect("updated alpha");
        assert!(updated[0] > 0.0 && updated[0] <= 1.0);
        assert!(updated[1] >= 0.0 && updated[1] < 1.0);
        assert_eq!(alpha1.alpha("relu_a"), alpha1.alpha_upper("relu_a"));
    }

    #[test]
    fn checked_adam_refuses_partial_relu_alignment_without_mutating_source() {
        let alpha0 = alpha_fixture();
        let before = alpha0.alpha("relu_a").cloned();
        let result = apply_checked_adam_step(
            &alpha0,
            &["relu_a".to_string()],
            &[vec![1.0, 1.0]],
            AdamParams::new(0.1, 1),
            Instant::now() + Duration::from_secs(10),
        );
        assert_eq!(
            result.err(),
            Some(AtomicCudaMarginStepRefusal::ReluAlignment)
        );
        assert_eq!(alpha0.alpha("relu_a").cloned(), before);
    }

    #[test]
    fn checked_adam_reduces_full_spatial_gradient_for_channel_only_alpha() {
        let pre = BoundedTensor::new(
            Array3::from_shape_vec((2, 1, 2), vec![-2.0, -1.0, -1.0, -3.0])
                .expect("shape")
                .into_dyn(),
            Array3::from_shape_vec((2, 1, 2), vec![1.0, 3.0, 4.0, 2.0])
                .expect("shape")
                .into_dyn(),
        )
        .expect("valid preactivation");
        let mut alpha0 = GraphAlphaState::new();
        alpha0
            .add_relu_node("relu", &pre, true)
            .expect("channel alpha");
        assert_eq!(alpha0.alpha("relu").map(Array1::len), Some(2));

        let alpha1 = apply_checked_adam_step(
            &alpha0,
            &["relu".to_string()],
            &[vec![2.0, 2.0, -3.0, -3.0]],
            AdamParams::new(0.1, 1),
            Instant::now() + Duration::from_secs(10),
        )
        .expect("spatial gradient reduces exactly");
        assert_ne!(alpha0.alpha("relu"), alpha1.alpha("relu"));
    }

    #[test]
    fn score_refuses_partial_vectors_or_nonfinite_thresholds() {
        let spec = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(
            score_complete_c(
                &bounds(&[0.0], &[1.0]),
                &spec,
                &[0.0, 0.0],
                false,
                Instant::now() + Duration::from_secs(10),
            )
            .err(),
            Some(AtomicCudaMarginStepRefusal::BoundsShape)
        );
        assert_eq!(
            score_complete_c(
                &bounds(&[0.0, 1.0], &[1.0, 2.0]),
                &spec,
                &[0.0, f32::INFINITY],
                false,
                Instant::now() + Duration::from_secs(10),
            )
            .err(),
            Some(AtomicCudaMarginStepRefusal::NonFiniteThreshold)
        );
    }

    #[test]
    fn score_obeys_the_authoritative_deadline() {
        let spec = arr2(&[[1.0]]);
        assert_eq!(
            score_complete_c(
                &bounds(&[0.0], &[1.0]),
                &spec,
                &[0.0],
                false,
                Instant::now(),
            )
            .err(),
            Some(AtomicCudaMarginStepRefusal::DeadlineExceeded)
        );
    }
}
