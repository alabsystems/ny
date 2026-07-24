// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Untrusted batched Adam proposals for constrained-zonotope dual bounds.
//!
//! GPU/GEMM arithmetic in this module has no proof authority.  It searches for
//! nonnegative multipliers for
//!
//! ```text
//! maximize -d^T lambda - ||g + C^T lambda||_1, lambda >= 0,
//! ```
//!
//! batching the lower problem (`g`) and upper problem (`-g`) for every supplied
//! direction.  Every candidate is independently replayed by
//! [`ConstrainedZonotope64::evaluate_dual`], whose outward `f64` result is the
//! sole acceptance gate.  Engine errors, panics, timeouts, malformed outputs,
//! non-finite candidate arithmetic, invalid configuration, and candidate-search
//! resource rejection all return the already-certified zero-multiplier baseline.
//!
//! This module is deliberately unwired and cannot emit a verifier verdict.

// `catch_unwind` is a real fallback only when the final binary uses unwinding.
// Fail at compile time instead of silently shipping dead panic recovery.
#[cfg(panic = "abort")]
compile_error!(
    "replay-gated batched Adam requires panic=unwind; isolate the engine in a subprocess before enabling panic=abort"
);

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::{Duration, Instant};

use ndarray::{ArrayView1, ArrayView2};
use ny_core::GemmEngine;

use crate::{ConstrainedZonotope64, ConstrainedZonotope64Error, CoordinateDualProposal};

/// Hard ceiling on simultaneously proposed output directions.
pub const BATCHED_ADAM_HARD_MAX_DIRECTIONS: usize = 64;
/// Hard ceiling on projected-Adam iterations.
pub const BATCHED_ADAM_HARD_MAX_ITERATIONS: usize = 512;
/// Hard ceiling on the flattened constrained-zonotope value dimension.
pub const BATCHED_ADAM_HARD_MAX_VALUE_DIM: usize = 131_072;
/// Hard ceiling on rows in `C alpha <= d`.
pub const BATCHED_ADAM_HARD_MAX_CONSTRAINTS: usize = 16_384;
/// Hard ceiling on constrained-zonotope alpha symbols.
pub const BATCHED_ADAM_HARD_MAX_ALPHA_DIM: usize = 8_192;
/// Hard ceiling on dense elements in `C`.
pub const BATCHED_ADAM_HARD_MAX_CONSTRAINT_ELEMENTS: usize = 134_217_728;
/// Hard ceiling on nonzeros visited while projecting sparse generators.
pub const BATCHED_ADAM_HARD_MAX_GENERATOR_NONZEROS: usize = 134_217_728;
/// Hard ceiling on direction elements visited by the proposer.
pub const BATCHED_ADAM_HARD_MAX_DIRECTION_ELEMENTS: usize = 8_388_608;
/// Hard ceiling on direction-by-generator-nonzero projection products.
pub const BATCHED_ADAM_HARD_MAX_PROJECTION_PRODUCTS: u64 = 8_589_934_592;
/// Hard ceiling on mandatory baseline dual terms across every direction.
///
/// This counts dense `C^T lambda` entries plus sparse generator entries.  The
/// value-domain center/remainder work is bounded separately by direction
/// elements.
pub const BATCHED_ADAM_HARD_MAX_BASELINE_DUAL_TERMS: u64 = 8_589_934_592;
/// Hard ceiling on the `2 * directions * constraints` multiplier batch.
pub const BATCHED_ADAM_HARD_MAX_MULTIPLIER_ELEMENTS: usize = 2_097_152;
/// Hard ceiling on explicitly retained candidate-search `f32` elements.
pub const BATCHED_ADAM_HARD_MAX_WORKING_F32_ELEMENTS: usize = 300_000_000;
/// Hard ceiling on scalar products submitted to GEMM across all iterations.
pub const BATCHED_ADAM_HARD_MAX_GEMM_PRODUCTS: u64 = 20_000_000_000_000;
/// Hard ceiling on caller-selected candidate-search wall time.
pub const BATCHED_ADAM_HARD_MAX_WALL_TIME: Duration = Duration::from_mins(1);

/// Maximum number of charged CPU loop work items between deadline polls.
///
/// Every converted or inspected output value, transposed element, sparse
/// entry, generator visit, sign element, and Adam element charges at least one
/// item; zero filling uses chunks no larger than this interval. Engine calls
/// remain externally guarded, but all CPU-owned candidate loops can and must
/// be cooperatively interruptible at a small, fixed granularity.
const CPU_DEADLINE_POLL_INTERVAL: usize = 16_384;

/// Caller-tightenable resource limits.  There is intentionally no `Default`:
/// an experimental caller must choose every cap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchedAdamLimits {
    /// Maximum supplied directions.
    pub max_directions: usize,
    /// Maximum Adam iterations.
    pub max_iterations: usize,
    /// Maximum flattened value dimension.
    pub max_value_dim: usize,
    /// Maximum constraint rows.
    pub max_constraints: usize,
    /// Maximum alpha symbols.
    pub max_alpha_dim: usize,
    /// Maximum dense constraint-matrix elements.
    pub max_constraint_elements: usize,
    /// Maximum sparse generator nonzeros.
    pub max_generator_nonzeros: usize,
    /// Maximum direction elements.
    pub max_direction_elements: usize,
    /// Maximum projection products, counted independently of machine `usize`.
    pub max_projection_products: u64,
    /// Maximum batched multiplier elements.
    pub max_multiplier_elements: usize,
    /// Maximum explicitly retained candidate-search `f32` elements.
    pub max_working_f32_elements: usize,
    /// Maximum scalar GEMM products over the whole search.
    pub max_gemm_products: u64,
    /// Maximum candidate-search wall time.
    pub max_wall_time: Duration,
}

/// Adam search configuration.  All values are candidate-search parameters;
/// changing them cannot bypass outward replay.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchedAdamConfig {
    /// Number of projected-Adam updates.
    pub iterations: usize,
    /// Adam ascent learning rate.
    pub learning_rate: f32,
    /// First-moment decay in `[0, 1)`.
    pub beta1: f32,
    /// Second-moment decay in `[0, 1)`.
    pub beta2: f32,
    /// Positive denominator stabilizer.
    pub epsilon: f32,
    /// Wall time for candidate preparation and GEMM search, excluding mandatory
    /// baseline construction and independent outward replay.
    pub wall_time: Duration,
    /// Caller-tightenable resource caps.
    pub limits: BatchedAdamLimits,
}

/// Checked dimensions and conservative work accounting for one batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchedAdamPlan {
    /// Number of supplied directions.
    pub directions: usize,
    /// Lower and upper search lanes (`2 * directions`).
    pub lanes: usize,
    /// Flattened value dimension.
    pub value_dim: usize,
    /// Number of constraint rows.
    pub constraints: usize,
    /// Number of alpha symbols.
    pub alpha_dim: usize,
    /// Dense elements in `C`.
    pub constraint_elements: usize,
    /// Sparse generator nonzeros visited per direction.
    pub generator_nonzeros: usize,
    /// Supplied direction elements.
    pub direction_elements: usize,
    /// Direction-by-generator-nonzero products.
    pub projection_products: u64,
    /// Batched multiplier elements.
    pub multiplier_elements: usize,
    /// Conservatively retained candidate-search `f32` elements.  This excludes
    /// the input domain, mandatory `f64` baselines, and engine-private packing.
    pub working_f32_elements: usize,
    /// Scalar products submitted to GEMM over every iteration.
    pub gemm_products: u64,
    /// Planned Adam updates.
    pub iterations: usize,
}

/// Why the untrusted search did or did not reach CPU replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchedAdamStatus {
    /// Candidate search completed and all retained candidates were CPU-replayed.
    Completed,
    /// Configuration or a caller-selected search cap rejected the heuristic.
    ResourceFallback,
    /// An empty direction, constraint, or alpha axis made search unnecessary.
    NoSearchNeeded,
    /// Candidate preparation or search reached its wall-time budget.
    Deadline,
    /// The untrusted GEMM engine returned an error.
    EngineError,
    /// The untrusted GEMM engine panicked.
    EnginePanic,
    /// The untrusted GEMM engine returned the wrong number of elements.
    MalformedEngineOutput,
    /// Candidate-only arithmetic produced NaN or infinity.
    NonFiniteCandidate,
    /// A bounded heuristic allocation could not be reserved.
    AllocationFallback,
}

/// Independently replayed results for every supplied direction.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchedAdamProposal {
    /// One independently replayed result per supplied direction.
    pub proposals: Vec<CoordinateDualProposal>,
    /// Checked search plan, or `None` when configuration/caps rejected search.
    pub plan: Option<BatchedAdamPlan>,
    /// Outcome of the untrusted search lane.
    pub status: BatchedAdamStatus,
    /// Fully completed Adam updates.
    pub iterations_completed: usize,
    /// Calls made to [`GemmEngine::gemm_f32_fast`].
    pub engine_calls: usize,
}

/// Failure of the mandatory zero-multiplier authority path.  Heuristic search
/// failures are represented by [`BatchedAdamStatus`] and are not errors.
#[derive(Debug, thiserror::Error)]
pub enum BatchedAdamProposerError {
    /// Mandatory direction shape is incompatible with the domain.
    #[error("direction width mismatch: expected {value_dim}, got {direction_width}")]
    BaselineShape {
        /// Domain value dimension.
        value_dim: usize,
        /// Supplied direction width.
        direction_width: usize,
    },
    /// A hard mandatory-path resource ceiling was exceeded before allocation.
    #[error("mandatory baseline resource {resource} is {actual}, above hard limit {limit}")]
    BaselineResourceLimit {
        /// Resource rejected before baseline allocation/evaluation.
        resource: &'static str,
        /// Checked requested amount.
        actual: usize,
        /// Non-configurable hard ceiling.
        limit: usize,
    },
    /// A mandatory baseline work product exceeded its fixed `u64` ceiling.
    #[error("mandatory baseline work {resource} is {actual}, above hard limit {limit}")]
    BaselineWorkLimit {
        /// Work product rejected before baseline evaluation.
        resource: &'static str,
        /// Checked requested work.
        actual: u64,
        /// Non-configurable hard ceiling.
        limit: u64,
    },
    /// A checked mandatory-path size calculation overflowed.
    #[error("mandatory baseline size overflow while computing {resource}")]
    BaselineResourceOverflow {
        /// Calculation that could not be represented.
        resource: &'static str,
    },
    /// Mandatory baseline storage could not be reserved.
    #[error("unable to reserve mandatory baseline storage for {resource}")]
    BaselineAllocation { resource: &'static str },
    /// The strict outward evaluator rejected a zero-multiplier baseline.
    #[error("zero-multiplier baseline failed for direction {direction}: {source}")]
    Baseline {
        direction: usize,
        #[source]
        source: ConstrainedZonotope64Error,
    },
}

/// Search all lower/upper directions with batched approximate GEMMs and accept
/// only independently outward-replayed improvements.
///
/// # Errors
///
/// Returns [`BatchedAdamProposerError`] only when the mandatory zero-multiplier
/// baseline is malformed, exceeds its non-configurable hard resource ceiling,
/// cannot be allocated, or cannot be evaluated by the outward CPU authority.
/// Every candidate-search failure instead returns that certified baseline.
pub fn propose_batched_adam_unwired(
    domain: &ConstrainedZonotope64,
    directions: ArrayView2<'_, f64>,
    config: BatchedAdamConfig,
    engine: &dyn GemmEngine,
) -> Result<BatchedAdamProposal, BatchedAdamProposerError> {
    // Mandatory proof-authority work has separate non-configurable hard caps.
    // A malformed heuristic configuration must not authorize an unbounded
    // baseline allocation, and it must not hide a baseline shape error.
    check_mandatory_baseline_resources(domain, directions)?;
    let mut baselines = mandatory_baselines(domain, directions)?;
    let Some(plan) = BatchedAdamPlan::checked(domain, directions, config) else {
        return Ok(BatchedAdamProposal {
            proposals: baselines,
            plan: None,
            status: BatchedAdamStatus::ResourceFallback,
            iterations_completed: 0,
            engine_calls: 0,
        });
    };

    if plan.constraints == 0 || plan.alpha_dim == 0 || plan.directions == 0 {
        return Ok(BatchedAdamProposal {
            proposals: baselines,
            plan: Some(plan),
            status: BatchedAdamStatus::NoSearchNeeded,
            iterations_completed: 0,
            engine_calls: 0,
        });
    }

    let start = Instant::now();
    let search = run_candidate_search(domain, directions, config, plan, engine, start);
    let (multipliers, iterations_completed, engine_calls) = match search {
        Ok(result) => result,
        Err(failure) => {
            return Ok(BatchedAdamProposal {
                proposals: baselines,
                plan: Some(plan),
                status: failure.status(),
                iterations_completed: failure.iterations_completed(),
                engine_calls: failure.engine_calls(),
            });
        }
    };

    replay_candidates(domain, directions, &multipliers, &mut baselines);
    Ok(BatchedAdamProposal {
        proposals: baselines,
        plan: Some(plan),
        status: BatchedAdamStatus::Completed,
        iterations_completed,
        engine_calls,
    })
}

impl BatchedAdamPlan {
    fn checked(
        domain: &ConstrainedZonotope64,
        directions: ArrayView2<'_, f64>,
        config: BatchedAdamConfig,
    ) -> Option<Self> {
        if !valid_config(config) || directions.ncols() != domain.value_dim() {
            return None;
        }
        let limits = config.limits;
        let direction_count = directions.nrows();
        let lanes = direction_count.checked_mul(2)?;
        let value_dim = domain.value_dim();
        let constraints = domain.constraint_count();
        let alpha_dim = domain.alpha_dim();

        if direction_count > limits.max_directions
            || value_dim > limits.max_value_dim
            || constraints > limits.max_constraints
            || alpha_dim > limits.max_alpha_dim
        {
            return None;
        }
        let constraint_elements = constraints.checked_mul(alpha_dim)?;
        if constraint_elements > limits.max_constraint_elements {
            return None;
        }
        let generator_nonzeros = domain
            .generators()
            .iter()
            .try_fold(0_usize, |sum, generator| sum.checked_add(generator.nnz()))?;
        if generator_nonzeros > limits.max_generator_nonzeros {
            return None;
        }
        let direction_elements = direction_count.checked_mul(value_dim)?;
        if direction_elements > limits.max_direction_elements {
            return None;
        }
        let projection_products =
            u64::try_from((direction_count as u128).checked_mul(generator_nonzeros as u128)?)
                .ok()?;
        if projection_products > limits.max_projection_products {
            return None;
        }
        let multiplier_elements = lanes.checked_mul(constraints)?;
        if multiplier_elements > limits.max_multiplier_elements {
            return None;
        }

        // Peak retained candidate buffers are C + C^T + rhs, signed g plus the
        // first GEMM's in-place signs, and lambda + first moment + second moment
        // plus the second GEMM's gradient.  Engine-private packing is excluded.
        let projected_elements = lanes.checked_mul(alpha_dim)?;
        let working_f32_elements = checked_working_f32_elements(
            constraint_elements,
            projected_elements,
            multiplier_elements,
            constraints,
        )?;
        if working_f32_elements > limits.max_working_f32_elements {
            return None;
        }

        let products_per_gemm = (lanes as u128)
            .checked_mul(constraints as u128)?
            .checked_mul(alpha_dim as u128)?;
        let gemm_products_u128 = products_per_gemm
            .checked_mul(2)?
            .checked_mul(config.iterations as u128)?;
        let gemm_products = u64::try_from(gemm_products_u128).ok()?;
        if gemm_products > limits.max_gemm_products {
            return None;
        }

        Some(Self {
            directions: direction_count,
            lanes,
            value_dim,
            constraints,
            alpha_dim,
            constraint_elements,
            generator_nonzeros,
            direction_elements,
            projection_products,
            multiplier_elements,
            working_f32_elements,
            gemm_products,
            iterations: config.iterations,
        })
    }
}

fn checked_working_f32_elements(
    constraint_elements: usize,
    projected_elements: usize,
    multiplier_elements: usize,
    constraints: usize,
) -> Option<usize> {
    constraint_elements
        .checked_mul(2)?
        .checked_add(projected_elements.checked_mul(2)?)?
        .checked_add(multiplier_elements.checked_mul(4)?)?
        .checked_add(constraints)
}

fn valid_config(config: BatchedAdamConfig) -> bool {
    let limits = config.limits;
    config.iterations > 0
        && config.iterations <= limits.max_iterations
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
        && limits.max_directions > 0
        && limits.max_directions <= BATCHED_ADAM_HARD_MAX_DIRECTIONS
        && limits.max_iterations > 0
        && limits.max_iterations <= BATCHED_ADAM_HARD_MAX_ITERATIONS
        && limits.max_value_dim > 0
        && limits.max_value_dim <= BATCHED_ADAM_HARD_MAX_VALUE_DIM
        && limits.max_constraints > 0
        && limits.max_constraints <= BATCHED_ADAM_HARD_MAX_CONSTRAINTS
        && limits.max_alpha_dim > 0
        && limits.max_alpha_dim <= BATCHED_ADAM_HARD_MAX_ALPHA_DIM
        && limits.max_constraint_elements > 0
        && limits.max_constraint_elements <= BATCHED_ADAM_HARD_MAX_CONSTRAINT_ELEMENTS
        && limits.max_generator_nonzeros > 0
        && limits.max_generator_nonzeros <= BATCHED_ADAM_HARD_MAX_GENERATOR_NONZEROS
        && limits.max_direction_elements > 0
        && limits.max_direction_elements <= BATCHED_ADAM_HARD_MAX_DIRECTION_ELEMENTS
        && limits.max_projection_products > 0
        && limits.max_projection_products <= BATCHED_ADAM_HARD_MAX_PROJECTION_PRODUCTS
        && limits.max_multiplier_elements > 0
        && limits.max_multiplier_elements <= BATCHED_ADAM_HARD_MAX_MULTIPLIER_ELEMENTS
        && limits.max_working_f32_elements > 0
        && limits.max_working_f32_elements <= BATCHED_ADAM_HARD_MAX_WORKING_F32_ELEMENTS
        && limits.max_gemm_products > 0
        && limits.max_gemm_products <= BATCHED_ADAM_HARD_MAX_GEMM_PRODUCTS
        && !limits.max_wall_time.is_zero()
        && limits.max_wall_time <= BATCHED_ADAM_HARD_MAX_WALL_TIME
}

fn mandatory_baselines(
    domain: &ConstrainedZonotope64,
    directions: ArrayView2<'_, f64>,
) -> Result<Vec<CoordinateDualProposal>, BatchedAdamProposerError> {
    let mut zero = Vec::new();
    zero.try_reserve_exact(domain.constraint_count())
        .map_err(|_| BatchedAdamProposerError::BaselineAllocation {
            resource: "zero multipliers",
        })?;
    zero.resize(domain.constraint_count(), 0.0);

    let mut proposals = Vec::new();
    proposals
        .try_reserve_exact(directions.nrows())
        .map_err(|_| BatchedAdamProposerError::BaselineAllocation {
            resource: "direction proposals",
        })?;
    for (direction_index, direction) in directions.rows().into_iter().enumerate() {
        let bounds = evaluate_baseline_direction(domain, direction, &zero).map_err(|source| {
            BatchedAdamProposerError::Baseline {
                direction: direction_index,
                source,
            }
        })?;
        proposals.push(CoordinateDualProposal {
            bounds,
            lower_multipliers: clone_f64(&zero, "lower zero multipliers")?,
            upper_multipliers: clone_f64(&zero, "upper zero multipliers")?,
            lower_improved: false,
            upper_improved: false,
        });
    }
    Ok(proposals)
}

fn check_mandatory_baseline_resources(
    domain: &ConstrainedZonotope64,
    directions: ArrayView2<'_, f64>,
) -> Result<(), BatchedAdamProposerError> {
    if directions.ncols() != domain.value_dim() {
        return Err(BatchedAdamProposerError::BaselineShape {
            value_dim: domain.value_dim(),
            direction_width: directions.ncols(),
        });
    }
    require_baseline_limit(
        "directions",
        directions.nrows(),
        BATCHED_ADAM_HARD_MAX_DIRECTIONS,
    )?;
    require_baseline_limit(
        "value dimension",
        domain.value_dim(),
        BATCHED_ADAM_HARD_MAX_VALUE_DIM,
    )?;
    require_baseline_limit(
        "constraints",
        domain.constraint_count(),
        BATCHED_ADAM_HARD_MAX_CONSTRAINTS,
    )?;
    require_baseline_limit(
        "alpha dimension",
        domain.alpha_dim(),
        BATCHED_ADAM_HARD_MAX_ALPHA_DIM,
    )?;
    let constraint_elements = domain
        .constraint_count()
        .checked_mul(domain.alpha_dim())
        .ok_or(BatchedAdamProposerError::BaselineResourceOverflow {
            resource: "constraint elements",
        })?;
    require_baseline_limit(
        "constraint elements",
        constraint_elements,
        BATCHED_ADAM_HARD_MAX_CONSTRAINT_ELEMENTS,
    )?;
    let generator_nonzeros = domain
        .generators()
        .iter()
        .try_fold(0_usize, |count, generator| {
            count.checked_add(generator.nnz()).ok_or(
                BatchedAdamProposerError::BaselineResourceOverflow {
                    resource: "generator nonzeros",
                },
            )
        })?;
    require_baseline_limit(
        "generator nonzeros",
        generator_nonzeros,
        BATCHED_ADAM_HARD_MAX_GENERATOR_NONZEROS,
    )?;
    let direction_elements = directions.nrows().checked_mul(directions.ncols()).ok_or(
        BatchedAdamProposerError::BaselineResourceOverflow {
            resource: "direction elements",
        },
    )?;
    require_baseline_limit(
        "direction elements",
        direction_elements,
        BATCHED_ADAM_HARD_MAX_DIRECTION_ELEMENTS,
    )?;
    let baseline_dual_terms = (directions.nrows() as u128)
        .checked_mul(
            (constraint_elements as u128)
                .checked_add(generator_nonzeros as u128)
                .ok_or(BatchedAdamProposerError::BaselineResourceOverflow {
                    resource: "baseline dual terms",
                })?,
        )
        .and_then(|count| u64::try_from(count).ok())
        .ok_or(BatchedAdamProposerError::BaselineResourceOverflow {
            resource: "baseline dual terms",
        })?;
    require_baseline_work_limit(
        "dense and sparse dual terms",
        baseline_dual_terms,
        BATCHED_ADAM_HARD_MAX_BASELINE_DUAL_TERMS,
    )?;
    let lanes = directions.nrows().checked_mul(2).ok_or(
        BatchedAdamProposerError::BaselineResourceOverflow {
            resource: "lower/upper lanes",
        },
    )?;
    // Baselines retain one shared zero vector plus independent lower and upper
    // clones for every direction.
    let retained_multiplier_vectors =
        lanes
            .checked_add(1)
            .ok_or(BatchedAdamProposerError::BaselineResourceOverflow {
                resource: "baseline multiplier vectors",
            })?;
    let multiplier_elements = retained_multiplier_vectors
        .checked_mul(domain.constraint_count())
        .ok_or(BatchedAdamProposerError::BaselineResourceOverflow {
            resource: "baseline multiplier elements",
        })?;
    require_baseline_limit(
        "baseline multiplier elements",
        multiplier_elements,
        BATCHED_ADAM_HARD_MAX_MULTIPLIER_ELEMENTS,
    )
}

fn require_baseline_limit(
    resource: &'static str,
    actual: usize,
    limit: usize,
) -> Result<(), BatchedAdamProposerError> {
    if actual > limit {
        Err(BatchedAdamProposerError::BaselineResourceLimit {
            resource,
            actual,
            limit,
        })
    } else {
        Ok(())
    }
}

fn require_baseline_work_limit(
    resource: &'static str,
    actual: u64,
    limit: u64,
) -> Result<(), BatchedAdamProposerError> {
    if actual > limit {
        Err(BatchedAdamProposerError::BaselineWorkLimit {
            resource,
            actual,
            limit,
        })
    } else {
        Ok(())
    }
}

fn evaluate_baseline_direction(
    domain: &ConstrainedZonotope64,
    direction: ArrayView1<'_, f64>,
    zero: &[f64],
) -> Result<crate::ConstrainedZonotopeDualBounds, ConstrainedZonotope64Error> {
    if let Some(direction) = direction.as_slice() {
        return domain.evaluate_dual(direction, zero);
    }
    // ndarray permits strided and reversed rows.  Copy one bounded row so the
    // mandatory CPU evaluator sees the same logical values in row order.
    let mut contiguous = Vec::new();
    contiguous.try_reserve_exact(direction.len()).map_err(|_| {
        ConstrainedZonotope64Error::AllocationFailure {
            resource: "non-contiguous baseline direction",
        }
    })?;
    contiguous.extend(direction.iter().copied());
    domain.evaluate_dual(&contiguous, zero)
}

fn clone_f64(values: &[f64], resource: &'static str) -> Result<Vec<f64>, BatchedAdamProposerError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(values.len())
        .map_err(|_| BatchedAdamProposerError::BaselineAllocation { resource })?;
    result.extend_from_slice(values);
    Ok(result)
}

#[derive(Clone, Copy, Debug)]
enum CandidateFailure {
    Deadline { iterations: usize, calls: usize },
    EngineError { iterations: usize, calls: usize },
    EnginePanic { iterations: usize, calls: usize },
    MalformedEngineOutput { iterations: usize, calls: usize },
    NonFinite { iterations: usize, calls: usize },
    Allocation { iterations: usize, calls: usize },
}

impl CandidateFailure {
    const fn status(self) -> BatchedAdamStatus {
        match self {
            Self::Deadline { .. } => BatchedAdamStatus::Deadline,
            Self::EngineError { .. } => BatchedAdamStatus::EngineError,
            Self::EnginePanic { .. } => BatchedAdamStatus::EnginePanic,
            Self::MalformedEngineOutput { .. } => BatchedAdamStatus::MalformedEngineOutput,
            Self::NonFinite { .. } => BatchedAdamStatus::NonFiniteCandidate,
            Self::Allocation { .. } => BatchedAdamStatus::AllocationFallback,
        }
    }

    const fn iterations_completed(self) -> usize {
        match self {
            Self::Deadline { iterations, .. }
            | Self::EngineError { iterations, .. }
            | Self::EnginePanic { iterations, .. }
            | Self::MalformedEngineOutput { iterations, .. }
            | Self::NonFinite { iterations, .. }
            | Self::Allocation { iterations, .. } => iterations,
        }
    }

    const fn engine_calls(self) -> usize {
        match self {
            Self::Deadline { calls, .. }
            | Self::EngineError { calls, .. }
            | Self::EnginePanic { calls, .. }
            | Self::MalformedEngineOutput { calls, .. }
            | Self::NonFinite { calls, .. }
            | Self::Allocation { calls, .. } => calls,
        }
    }
}

fn run_candidate_search(
    domain: &ConstrainedZonotope64,
    directions: ArrayView2<'_, f64>,
    config: BatchedAdamConfig,
    plan: BatchedAdamPlan,
    engine: &dyn GemmEngine,
    start: Instant,
) -> Result<(Vec<f32>, usize, usize), CandidateFailure> {
    let mut calls = 0_usize;
    let deadline = CandidateDeadline {
        start,
        limit: config.wall_time,
    };
    deadline.check(0, calls)?;
    let constraints = f32_constraints(domain, plan, deadline)?;
    deadline.check(0, calls)?;
    let transposed = transpose_constraints(&constraints, plan, deadline)?;
    deadline.check(0, calls)?;
    let rhs = f32_rhs(domain, plan, deadline)?;
    deadline.check(0, calls)?;
    let signed_g = projected_generators(domain, directions, plan, deadline)?;
    deadline.check(0, calls)?;
    let mut lambda = zero_f32(plan.multiplier_elements, "lambda", 0, calls, deadline)?;
    let mut first = zero_f32(plan.multiplier_elements, "first moment", 0, calls, deadline)?;
    let mut second = zero_f32(
        plan.multiplier_elements,
        "second moment",
        0,
        calls,
        deadline,
    )?;

    for iteration in 0..config.iterations {
        deadline.check(iteration, calls)?;
        let projected = gemm_candidate(
            engine,
            plan.lanes,
            plan.constraints,
            plan.alpha_dim,
            &lambda,
            &constraints,
            iteration,
            &mut calls,
            deadline,
        )?;
        deadline.check(iteration, calls)?;
        let signs = signed_projection(projected, &signed_g, iteration, calls, deadline)?;
        let gradient_product = gemm_candidate(
            engine,
            plan.lanes,
            plan.alpha_dim,
            plan.constraints,
            &signs,
            &transposed,
            iteration,
            &mut calls,
            deadline,
        )?;
        deadline.check(iteration, calls)?;
        adam_update(
            &mut lambda,
            &mut first,
            &mut second,
            gradient_product,
            &rhs,
            config,
            iteration,
            calls,
            deadline,
        )?;
    }
    deadline.check(config.iterations, calls)?;
    Ok((lambda, config.iterations, calls))
}

fn f32_constraints(
    domain: &ConstrainedZonotope64,
    plan: BatchedAdamPlan,
    deadline: CandidateDeadline,
) -> Result<Vec<f32>, CandidateFailure> {
    let source = domain.constraints();
    let source = source.as_slice().ok_or(CandidateFailure::NonFinite {
        iterations: 0,
        calls: 0,
    })?;
    convert_f32(source, plan.constraint_elements, deadline)
}

fn transpose_constraints(
    constraints: &[f32],
    plan: BatchedAdamPlan,
    deadline: CandidateDeadline,
) -> Result<Vec<f32>, CandidateFailure> {
    let mut transposed = zero_f32(plan.constraint_elements, "C transpose", 0, 0, deadline)?;
    let mut until_deadline_poll = 0;
    for row in 0..plan.constraints {
        for column in 0..plan.alpha_dim {
            deadline.poll(&mut until_deadline_poll, 0, 0)?;
            transposed[column * plan.constraints + row] =
                constraints[row * plan.alpha_dim + column];
        }
    }
    deadline.check(0, 0)?;
    Ok(transposed)
}

fn f32_rhs(
    domain: &ConstrainedZonotope64,
    plan: BatchedAdamPlan,
    deadline: CandidateDeadline,
) -> Result<Vec<f32>, CandidateFailure> {
    convert_f32(domain.rhs(), plan.constraints, deadline)
}

fn projected_generators(
    domain: &ConstrainedZonotope64,
    directions: ArrayView2<'_, f64>,
    plan: BatchedAdamPlan,
    deadline: CandidateDeadline,
) -> Result<Vec<f32>, CandidateFailure> {
    let count = plan
        .lanes
        .checked_mul(plan.alpha_dim)
        .ok_or(CandidateFailure::Allocation {
            iterations: 0,
            calls: 0,
        })?;
    let mut projected = zero_f32(count, "signed projected generators", 0, 0, deadline)?;
    let mut until_deadline_poll = 0;
    for (direction_index, direction) in directions.rows().into_iter().enumerate() {
        for (generator_index, generator) in domain.generators().iter().enumerate() {
            // Empty sparse columns still incur one lane-projection visit, so
            // charge the outer loop as well as every stored coefficient.
            deadline.poll(&mut until_deadline_poll, 0, 0)?;
            let mut sum = 0.0_f64;
            for (value_index, coefficient) in generator.entries() {
                deadline.poll(&mut until_deadline_poll, 0, 0)?;
                let product = direction[value_index] * coefficient;
                sum += product;
                if !product.is_finite() || !sum.is_finite() {
                    return Err(CandidateFailure::NonFinite {
                        iterations: 0,
                        calls: 0,
                    });
                }
            }
            let candidate = sum as f32;
            if !candidate.is_finite() {
                return Err(CandidateFailure::NonFinite {
                    iterations: 0,
                    calls: 0,
                });
            }
            let lower = (2 * direction_index) * plan.alpha_dim + generator_index;
            let upper = (2 * direction_index + 1) * plan.alpha_dim + generator_index;
            projected[lower] = candidate;
            projected[upper] = -candidate;
        }
    }
    deadline.check(0, 0)?;
    Ok(projected)
}

fn convert_f32(
    source: &[f64],
    expected: usize,
    deadline: CandidateDeadline,
) -> Result<Vec<f32>, CandidateFailure> {
    if source.len() != expected {
        return Err(CandidateFailure::MalformedEngineOutput {
            iterations: 0,
            calls: 0,
        });
    }
    let mut converted = Vec::new();
    converted
        .try_reserve_exact(expected)
        .map_err(|_| CandidateFailure::Allocation {
            iterations: 0,
            calls: 0,
        })?;
    let mut until_deadline_poll = 0;
    for &value in source {
        deadline.poll(&mut until_deadline_poll, 0, 0)?;
        let candidate = value as f32;
        if !candidate.is_finite() {
            return Err(CandidateFailure::NonFinite {
                iterations: 0,
                calls: 0,
            });
        }
        converted.push(candidate);
    }
    deadline.check(0, 0)?;
    Ok(converted)
}

fn zero_f32(
    count: usize,
    _resource: &'static str,
    iterations: usize,
    calls: usize,
    deadline: CandidateDeadline,
) -> Result<Vec<f32>, CandidateFailure> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| CandidateFailure::Allocation { iterations, calls })?;
    while values.len() < count {
        deadline.check(iterations, calls)?;
        let next_len = values
            .len()
            .saturating_add(CPU_DEADLINE_POLL_INTERVAL)
            .min(count);
        values.resize(next_len, 0.0);
    }
    deadline.check(iterations, calls)?;
    Ok(values)
}

fn gemm_candidate(
    engine: &dyn GemmEngine,
    m: usize,
    k: usize,
    n: usize,
    left: &[f32],
    right: &[f32],
    iteration: usize,
    calls: &mut usize,
    deadline: CandidateDeadline,
) -> Result<Vec<f32>, CandidateFailure> {
    *calls = calls.checked_add(1).ok_or(CandidateFailure::Allocation {
        iterations: iteration,
        calls: *calls,
    })?;
    let call = catch_unwind(AssertUnwindSafe(|| {
        engine.gemm_f32_fast(m, k, n, left, right)
    }));
    let result = match call {
        Ok(Ok(result)) => result,
        Ok(Err(_)) => {
            return Err(CandidateFailure::EngineError {
                iterations: iteration,
                calls: *calls,
            });
        }
        Err(_) => {
            return Err(CandidateFailure::EnginePanic {
                iterations: iteration,
                calls: *calls,
            });
        }
    };
    let expected = m
        .checked_mul(n)
        .ok_or(CandidateFailure::MalformedEngineOutput {
            iterations: iteration,
            calls: *calls,
        })?;
    if result.len() != expected {
        return Err(CandidateFailure::MalformedEngineOutput {
            iterations: iteration,
            calls: *calls,
        });
    }
    let mut until_deadline_poll = 0;
    for value in &result {
        deadline.poll(&mut until_deadline_poll, iteration, *calls)?;
        if !value.is_finite() {
            return Err(CandidateFailure::NonFinite {
                iterations: iteration,
                calls: *calls,
            });
        }
    }
    deadline.check(iteration, *calls)?;
    Ok(result)
}

fn signed_projection(
    mut projected: Vec<f32>,
    signed_g: &[f32],
    iteration: usize,
    calls: usize,
    deadline: CandidateDeadline,
) -> Result<Vec<f32>, CandidateFailure> {
    if projected.len() != signed_g.len() {
        return Err(CandidateFailure::MalformedEngineOutput {
            iterations: iteration,
            calls,
        });
    }
    let mut until_deadline_poll = 0;
    for (value, &offset) in projected.iter_mut().zip(signed_g) {
        deadline.poll(&mut until_deadline_poll, iteration, calls)?;
        *value += offset;
        if !value.is_finite() {
            return Err(CandidateFailure::NonFinite {
                iterations: iteration,
                calls,
            });
        }
        *value = if *value > 0.0 {
            1.0
        } else if *value < 0.0 {
            -1.0
        } else {
            0.0
        };
    }
    deadline.check(iteration, calls)?;
    Ok(projected)
}

#[allow(clippy::too_many_arguments)]
fn adam_update(
    lambda: &mut [f32],
    first: &mut [f32],
    second: &mut [f32],
    mut gradient: Vec<f32>,
    rhs: &[f32],
    config: BatchedAdamConfig,
    iteration: usize,
    calls: usize,
    deadline: CandidateDeadline,
) -> Result<(), CandidateFailure> {
    if gradient.len() != lambda.len() || first.len() != lambda.len() || second.len() != lambda.len()
    {
        return Err(CandidateFailure::MalformedEngineOutput {
            iterations: iteration,
            calls,
        });
    }
    if rhs.is_empty() || !lambda.len().is_multiple_of(rhs.len()) {
        return Err(CandidateFailure::MalformedEngineOutput {
            iterations: iteration,
            calls,
        });
    }
    let step_index = i32::try_from(iteration + 1).map_err(|_| CandidateFailure::NonFinite {
        iterations: iteration,
        calls,
    })?;
    let first_correction = 1.0 - config.beta1.powi(step_index);
    let second_correction = 1.0 - config.beta2.powi(step_index);
    if !first_correction.is_finite()
        || !second_correction.is_finite()
        || first_correction <= 0.0
        || second_correction <= 0.0
    {
        return Err(CandidateFailure::NonFinite {
            iterations: iteration,
            calls,
        });
    }

    let mut until_deadline_poll = 0;
    for index in 0..lambda.len() {
        deadline.poll(&mut until_deadline_poll, iteration, calls)?;
        let row = index % rhs.len();
        gradient[index] = -rhs[row] - gradient[index];
        first[index] = config.beta1 * first[index] + (1.0 - config.beta1) * gradient[index];
        second[index] =
            config.beta2 * second[index] + (1.0 - config.beta2) * gradient[index].powi(2);
        let first_hat = first[index] / first_correction;
        let second_hat = second[index] / second_correction;
        let update = config.learning_rate * first_hat / (second_hat.sqrt() + config.epsilon);
        let candidate = (lambda[index] + update).max(0.0);
        if !gradient[index].is_finite()
            || !first[index].is_finite()
            || !second[index].is_finite()
            || !candidate.is_finite()
        {
            return Err(CandidateFailure::NonFinite {
                iterations: iteration,
                calls,
            });
        }
        lambda[index] = candidate;
    }
    deadline.check(iteration + 1, calls)?;
    Ok(())
}

fn check_deadline(
    start: Instant,
    limit: Duration,
    iterations: usize,
    calls: usize,
) -> Result<(), CandidateFailure> {
    if start.elapsed() >= limit {
        Err(CandidateFailure::Deadline { iterations, calls })
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct CandidateDeadline {
    start: Instant,
    limit: Duration,
}

impl CandidateDeadline {
    fn check(self, iterations: usize, calls: usize) -> Result<(), CandidateFailure> {
        check_deadline(self.start, self.limit, iterations, calls)
    }

    fn poll(
        self,
        until_deadline_poll: &mut usize,
        iterations: usize,
        calls: usize,
    ) -> Result<(), CandidateFailure> {
        if *until_deadline_poll == 0 {
            self.check(iterations, calls)?;
            *until_deadline_poll = CPU_DEADLINE_POLL_INTERVAL;
        }
        *until_deadline_poll -= 1;
        Ok(())
    }
}

fn replay_candidates(
    domain: &ConstrainedZonotope64,
    directions: ArrayView2<'_, f64>,
    multipliers: &[f32],
    proposals: &mut [CoordinateDualProposal],
) {
    let rows = domain.constraint_count();
    for (direction_index, direction) in directions.rows().into_iter().enumerate() {
        let lower_start = (2 * direction_index) * rows;
        let upper_start = (2 * direction_index + 1) * rows;
        let Some(lower) = candidate_f64(multipliers.get(lower_start..lower_start + rows)) else {
            continue;
        };
        let Some(upper) = candidate_f64(multipliers.get(upper_start..upper_start + rows)) else {
            continue;
        };
        if let Some(direction) = direction.as_slice() {
            replay_one_direction(
                domain,
                direction,
                lower,
                upper,
                &mut proposals[direction_index],
            );
            continue;
        }
        let mut contiguous = Vec::new();
        if contiguous.try_reserve_exact(direction.len()).is_err() {
            continue;
        }
        contiguous.extend(direction.iter().copied());
        replay_one_direction(
            domain,
            &contiguous,
            lower,
            upper,
            &mut proposals[direction_index],
        );
    }
}

fn replay_one_direction(
    domain: &ConstrainedZonotope64,
    direction: &[f64],
    lower: Vec<f64>,
    upper: Vec<f64>,
    proposal: &mut CoordinateDualProposal,
) {
    let baseline = proposal.bounds;
    if let Ok(bounds) = domain.evaluate_dual(direction, &lower) {
        if bounds.lower > baseline.lower {
            proposal.bounds.lower = bounds.lower;
            proposal.lower_multipliers = lower;
            proposal.lower_improved = true;
        }
    }
    if let Ok(bounds) = domain.evaluate_dual(direction, &upper) {
        if bounds.upper < baseline.upper {
            proposal.bounds.upper = bounds.upper;
            proposal.upper_multipliers = upper;
            proposal.upper_improved = true;
        }
    }
}

fn candidate_f64(source: Option<&[f32]>) -> Option<Vec<f64>> {
    let source = source?;
    let mut result = Vec::new();
    result.try_reserve_exact(source.len()).ok()?;
    for &value in source {
        if !value.is_finite() || value < 0.0 {
            return None;
        }
        result.push(f64::from(value));
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};
    use ny_core::{GemmEngine, NyError, Result as NyResult};

    use super::*;

    struct CpuEngine;

    impl GemmEngine for CpuEngine {
        fn gemm_f32(
            &self,
            m: usize,
            k: usize,
            n: usize,
            a: &[f32],
            b: &[f32],
        ) -> NyResult<Vec<f32>> {
            if a.len() != m * k || b.len() != k * n {
                return Err(NyError::InvalidSpec("bad test GEMM shape".into()));
            }
            let mut output = vec![0.0; m * n];
            for row in 0..m {
                for column in 0..n {
                    for inner in 0..k {
                        output[row * n + column] += a[row * k + inner] * b[inner * n + column];
                    }
                }
            }
            Ok(output)
        }
    }

    struct FailingEngine;

    impl GemmEngine for FailingEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> NyResult<Vec<f32>> {
            Err(NyError::InternalError("injected candidate failure".into()))
        }
    }

    struct PanicEngine;

    impl GemmEngine for PanicEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> NyResult<Vec<f32>> {
            panic!("injected candidate panic")
        }
    }

    struct NanEngine;

    impl GemmEngine for NanEngine {
        fn gemm_f32(
            &self,
            m: usize,
            _k: usize,
            n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> NyResult<Vec<f32>> {
            Ok(vec![f32::NAN; m * n])
        }
    }

    struct WrongLengthEngine;

    impl GemmEngine for WrongLengthEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> NyResult<Vec<f32>> {
            Ok(Vec::new())
        }
    }

    struct FastOnlyEngine;

    impl GemmEngine for FastOnlyEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> NyResult<Vec<f32>> {
            panic!("proof-contract GEMM must not be used for heuristic search")
        }

        fn gemm_f32_fast(
            &self,
            m: usize,
            k: usize,
            n: usize,
            a: &[f32],
            b: &[f32],
        ) -> NyResult<Vec<f32>> {
            CpuEngine.gemm_f32(m, k, n, a, b)
        }
    }

    fn limits() -> BatchedAdamLimits {
        BatchedAdamLimits {
            max_directions: 16,
            max_iterations: 200,
            max_value_dim: 1_000,
            max_constraints: 1_000,
            max_alpha_dim: 1_000,
            max_constraint_elements: 1_000_000,
            max_generator_nonzeros: 1_000_000,
            max_direction_elements: 1_000_000,
            max_projection_products: 1_000_000,
            max_multiplier_elements: 1_000_000,
            max_working_f32_elements: 10_000_000,
            max_gemm_products: 10_000_000_000,
            max_wall_time: Duration::from_secs(10),
        }
    }

    fn config() -> BatchedAdamConfig {
        BatchedAdamConfig {
            iterations: 100,
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
            wall_time: Duration::from_secs(5),
            limits: limits(),
        }
    }

    fn point_domain() -> ConstrainedZonotope64 {
        // alpha <= 0 and -alpha <= 0 force alpha = 0 while the zero dual sees
        // the unconstrained [-1,1] generator box.
        ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![vec![(0, 1.0)]],
            array![[1.0], [-1.0]],
            vec![0.0, 0.0],
            vec![0.0],
        )
        .unwrap()
    }

    #[test]
    fn cpu_adam_candidates_only_improve_after_outward_replay() {
        let domain = point_domain();
        let directions = array![[1.0], [-2.0]];
        let result =
            propose_batched_adam_unwired(&domain, directions.view(), config(), &CpuEngine).unwrap();
        assert_eq!(result.status, BatchedAdamStatus::Completed);
        assert_eq!(result.iterations_completed, 100);
        assert_eq!(result.engine_calls, 200);
        for proposal in result.proposals {
            assert!(proposal.lower_improved);
            assert!(proposal.upper_improved);
            assert!(proposal.bounds.lower <= 0.0);
            assert!(proposal.bounds.upper >= 0.0);
            assert!(proposal.bounds.lower > -2.0);
            assert!(proposal.bounds.upper < 2.0);
        }
    }

    #[test]
    fn search_uses_only_the_explicit_soundness_free_gemm_seam() {
        let result = propose_batched_adam_unwired(
            &point_domain(),
            array![[1.0]].view(),
            config(),
            &FastOnlyEngine,
        )
        .unwrap();
        assert_eq!(result.status, BatchedAdamStatus::Completed);
        assert!(result.proposals[0].lower_improved);
        assert!(result.proposals[0].upper_improved);
    }

    #[test]
    fn every_engine_failure_keeps_byte_identical_zero_baseline() {
        let domain = point_domain();
        let directions = array![[1.0]];
        let zero = vec![0.0; domain.constraint_count()];
        let baseline = domain.evaluate_dual(&[1.0], &zero).unwrap();
        for (engine, expected) in [
            (
                &FailingEngine as &dyn GemmEngine,
                BatchedAdamStatus::EngineError,
            ),
            (
                &PanicEngine as &dyn GemmEngine,
                BatchedAdamStatus::EnginePanic,
            ),
            (
                &WrongLengthEngine as &dyn GemmEngine,
                BatchedAdamStatus::MalformedEngineOutput,
            ),
            (
                &NanEngine as &dyn GemmEngine,
                BatchedAdamStatus::NonFiniteCandidate,
            ),
        ] {
            let result =
                propose_batched_adam_unwired(&domain, directions.view(), config(), engine).unwrap();
            assert_eq!(result.status, expected);
            assert_eq!(result.proposals[0].bounds, baseline);
            assert_eq!(result.proposals[0].lower_multipliers, zero);
            assert_eq!(result.proposals[0].upper_multipliers, zero);
            assert!(!result.proposals[0].lower_improved);
            assert!(!result.proposals[0].upper_improved);
        }
    }

    #[test]
    fn invalid_config_and_each_dimension_cap_keep_baseline_without_engine() {
        let domain = point_domain();
        let directions = array![[1.0]];
        let mut configs = Vec::new();
        let mut invalid = config();
        invalid.learning_rate = f32::NAN;
        configs.push(invalid);
        let mut invalid = config();
        invalid.wall_time = Duration::ZERO;
        configs.push(invalid);
        let mut invalid = config();
        invalid.limits.max_constraints = 1;
        configs.push(invalid);
        let mut invalid = config();
        invalid.limits.max_alpha_dim = 0;
        configs.push(invalid);
        let mut invalid = config();
        invalid.limits.max_constraint_elements = 1;
        configs.push(invalid);
        let mut invalid = config();
        invalid.limits.max_multiplier_elements = 1;
        configs.push(invalid);
        let mut invalid = config();
        invalid.limits.max_gemm_products = 1;
        configs.push(invalid);

        for invalid in configs {
            let result =
                propose_batched_adam_unwired(&domain, directions.view(), invalid, &PanicEngine)
                    .unwrap();
            assert_eq!(result.status, BatchedAdamStatus::ResourceFallback);
            assert_eq!(result.engine_calls, 0);
        }
    }

    #[test]
    fn zero_constraints_or_symbols_never_call_engine() {
        let domain = ConstrainedZonotope64::try_new(
            vec![1.0],
            Vec::new(),
            Array2::zeros((2, 0)),
            vec![0.0, 0.0],
            vec![0.25],
        )
        .unwrap();
        let result =
            propose_batched_adam_unwired(&domain, array![[1.0]].view(), config(), &PanicEngine)
                .unwrap();
        assert_eq!(result.status, BatchedAdamStatus::NoSearchNeeded);
        assert_eq!(result.engine_calls, 0);
    }

    #[test]
    fn malformed_direction_still_fails_on_mandatory_baseline() {
        let error = propose_batched_adam_unwired(
            &point_domain(),
            Array2::zeros((1, 2)).view(),
            config(),
            &CpuEngine,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            BatchedAdamProposerError::BaselineShape {
                value_dim: 1,
                direction_width: 2
            }
        ));
    }

    #[test]
    fn mandatory_caps_reject_before_baseline_or_engine_allocation() {
        let directions = Array2::zeros((BATCHED_ADAM_HARD_MAX_DIRECTIONS + 1, 1));
        let error = propose_batched_adam_unwired(
            &point_domain(),
            directions.view(),
            config(),
            &PanicEngine,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            BatchedAdamProposerError::BaselineResourceLimit {
                resource: "directions",
                actual: 65,
                limit: 64
            }
        ));

        let over_cap_domain = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![vec![(0, 1.0)]],
            Array2::zeros((BATCHED_ADAM_HARD_MAX_CONSTRAINTS + 1, 1)),
            vec![0.0; BATCHED_ADAM_HARD_MAX_CONSTRAINTS + 1],
            vec![0.0],
        )
        .unwrap();
        let error = propose_batched_adam_unwired(
            &over_cap_domain,
            array![[1.0]].view(),
            config(),
            &PanicEngine,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            BatchedAdamProposerError::BaselineResourceLimit {
                resource: "constraints",
                actual: 16_385,
                limit: 16_384
            }
        ));

        let over_alpha_domain = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![Vec::new(); BATCHED_ADAM_HARD_MAX_ALPHA_DIM + 1],
            Array2::zeros((1, BATCHED_ADAM_HARD_MAX_ALPHA_DIM + 1)),
            vec![0.0],
            vec![0.0],
        )
        .unwrap();
        let error = propose_batched_adam_unwired(
            &over_alpha_domain,
            array![[1.0]].view(),
            config(),
            &PanicEngine,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            BatchedAdamProposerError::BaselineResourceLimit {
                resource: "alpha dimension",
                actual: 8_193,
                limit: 8_192
            }
        ));

        // The mandatory allocation retains one shared zero vector in addition
        // to both multiplier clones for every direction.  This shape passes
        // the old `2*D*C` check but exceeds the actual `(2*D+1)*C` ceiling.
        let multiplier_domain = ConstrainedZonotope64::try_new(
            vec![0.0],
            Vec::new(),
            Array2::zeros((BATCHED_ADAM_HARD_MAX_CONSTRAINTS, 0)),
            vec![0.0; BATCHED_ADAM_HARD_MAX_CONSTRAINTS],
            vec![0.0],
        )
        .unwrap();
        let directions = Array2::zeros((BATCHED_ADAM_HARD_MAX_DIRECTIONS, 1));
        let error = propose_batched_adam_unwired(
            &multiplier_domain,
            directions.view(),
            config(),
            &PanicEngine,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            BatchedAdamProposerError::BaselineResourceLimit {
                resource: "baseline multiplier elements",
                actual: 2_113_536,
                limit: 2_097_152
            }
        ));

        assert!(matches!(
            require_baseline_work_limit(
                "synthetic",
                BATCHED_ADAM_HARD_MAX_BASELINE_DUAL_TERMS + 1,
                BATCHED_ADAM_HARD_MAX_BASELINE_DUAL_TERMS,
            ),
            Err(BatchedAdamProposerError::BaselineWorkLimit {
                resource: "synthetic",
                ..
            })
        ));
    }

    #[test]
    fn strided_direction_rows_are_baselined_and_replayed_in_logical_order() {
        let domain = ConstrainedZonotope64::try_new(
            vec![0.0, 0.0],
            vec![vec![(0, 1.0)]],
            array![[1.0], [-1.0]],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        )
        .unwrap();
        let storage = array![[1.0, -2.0], [0.0, 0.0]];
        let directions = storage.t();
        assert!(directions.row(0).as_slice().is_none());
        let result =
            propose_batched_adam_unwired(&domain, directions, config(), &CpuEngine).unwrap();
        assert_eq!(result.status, BatchedAdamStatus::Completed);
        for proposal in result.proposals {
            assert!(proposal.lower_improved);
            assert!(proposal.upper_improved);
            assert!(proposal.bounds.lower <= 0.0);
            assert!(proposal.bounds.upper >= 0.0);
        }
    }

    #[test]
    fn deadline_after_slow_candidate_returns_baseline() {
        struct SlowEngine;
        impl GemmEngine for SlowEngine {
            fn gemm_f32(
                &self,
                m: usize,
                _k: usize,
                n: usize,
                _a: &[f32],
                _b: &[f32],
            ) -> NyResult<Vec<f32>> {
                std::thread::sleep(Duration::from_millis(3));
                Ok(vec![0.0; m * n])
            }
        }
        let mut short = config();
        short.wall_time = Duration::from_millis(1);
        let result =
            propose_batched_adam_unwired(&point_domain(), array![[1.0]].view(), short, &SlowEngine)
                .unwrap();
        assert_eq!(result.status, BatchedAdamStatus::Deadline);
        assert_eq!(result.proposals[0].bounds.lower, -1.0);
        assert_eq!(result.proposals[0].bounds.upper, 1.0);
    }

    fn expired_deadline() -> CandidateDeadline {
        CandidateDeadline {
            start: Instant::now()
                .checked_sub(Duration::from_mins(1))
                .expect("one minute fits in Instant history"),
            limit: Duration::from_millis(1),
        }
    }

    #[test]
    fn expired_deadline_stops_each_cpu_preparation_loop() {
        let domain = point_domain();
        let directions = array![[1.0]];
        let plan = BatchedAdamPlan::checked(&domain, directions.view(), config()).unwrap();

        assert!(matches!(
            f32_constraints(&domain, plan, expired_deadline()),
            Err(CandidateFailure::Deadline {
                iterations: 0,
                calls: 0
            })
        ));
        assert!(matches!(
            transpose_constraints(&[1.0, -1.0], plan, expired_deadline()),
            Err(CandidateFailure::Deadline {
                iterations: 0,
                calls: 0
            })
        ));
        assert!(matches!(
            f32_rhs(&domain, plan, expired_deadline()),
            Err(CandidateFailure::Deadline {
                iterations: 0,
                calls: 0
            })
        ));
        assert!(matches!(
            projected_generators(&domain, directions.view(), plan, expired_deadline()),
            Err(CandidateFailure::Deadline {
                iterations: 0,
                calls: 0
            })
        ));
        assert!(matches!(
            zero_f32(1, "test", 0, 0, expired_deadline()),
            Err(CandidateFailure::Deadline {
                iterations: 0,
                calls: 0
            })
        ));
    }

    #[test]
    fn expired_deadline_stops_each_per_iteration_cpu_loop() {
        let mut calls = 0;
        assert!(matches!(
            gemm_candidate(
                &CpuEngine,
                1,
                1,
                1,
                &[1.0],
                &[1.0],
                3,
                &mut calls,
                expired_deadline(),
            ),
            Err(CandidateFailure::Deadline {
                iterations: 3,
                calls: 1
            })
        ));

        assert!(matches!(
            signed_projection(vec![0.0], &[0.0], 3, 7, expired_deadline()),
            Err(CandidateFailure::Deadline {
                iterations: 3,
                calls: 7
            })
        ));

        let mut lambda = [0.0];
        let mut first = [0.0];
        let mut second = [0.0];
        assert!(matches!(
            adam_update(
                &mut lambda,
                &mut first,
                &mut second,
                vec![0.0],
                &[0.0],
                config(),
                3,
                7,
                expired_deadline(),
            ),
            Err(CandidateFailure::Deadline {
                iterations: 3,
                calls: 7
            })
        ));
    }

    #[test]
    fn expired_deadline_stops_search_before_the_engine() {
        let domain = point_domain();
        let directions = array![[1.0]];
        let search_config = config();
        let plan = BatchedAdamPlan::checked(&domain, directions.view(), search_config).unwrap();
        let failure = run_candidate_search(
            &domain,
            directions.view(),
            search_config,
            plan,
            &PanicEngine,
            expired_deadline().start,
        )
        .unwrap_err();
        assert!(matches!(
            failure,
            CandidateFailure::Deadline {
                iterations: 0,
                calls: 0
            }
        ));
    }

    #[test]
    fn metaroom_terminal_estimate_is_checked_without_large_allocations() {
        const DIRECTIONS: usize = 19;
        const LANES: usize = 2 * DIRECTIONS;
        const CONSTRAINTS: usize = 9_912;
        const ALPHA_DIM: usize = 5_117;
        const ITERATIONS: u64 = 150;

        let constraint_elements = CONSTRAINTS.checked_mul(ALPHA_DIM).unwrap();
        let multiplier_elements = LANES.checked_mul(CONSTRAINTS).unwrap();
        let projected_elements = LANES.checked_mul(ALPHA_DIM).unwrap();
        let working_f32_elements = checked_working_f32_elements(
            constraint_elements,
            projected_elements,
            multiplier_elements,
            CONSTRAINTS,
        )
        .unwrap();
        let gemm_products = u64::try_from(LANES)
            .unwrap()
            .checked_mul(u64::try_from(CONSTRAINTS).unwrap())
            .unwrap()
            .checked_mul(u64::try_from(ALPHA_DIM).unwrap())
            .unwrap()
            .checked_mul(2)
            .unwrap()
            .checked_mul(ITERATIONS)
            .unwrap();

        assert_eq!(constraint_elements, 50_719_704);
        assert_eq!(multiplier_elements, 376_656);
        assert_eq!(projected_elements, 194_446);
        assert_eq!(working_f32_elements, 103_344_836);
        assert_eq!(gemm_products, 578_204_625_600);
        assert!(constraint_elements <= BATCHED_ADAM_HARD_MAX_CONSTRAINT_ELEMENTS);
        assert!(multiplier_elements <= BATCHED_ADAM_HARD_MAX_MULTIPLIER_ELEMENTS);
        assert!(working_f32_elements <= BATCHED_ADAM_HARD_MAX_WORKING_F32_ELEMENTS);
        assert!(gemm_products <= BATCHED_ADAM_HARD_MAX_GEMM_PRODUCTS);
    }
}
