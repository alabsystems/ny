// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unwired exact-AY LP diagnostic for a constrained-zonotope ReLU/affine tail.
//!
//! This module deliberately has no verifier or verdict integration.  It builds
//! an outer LP for
//!
//! ```text
//! x = c + G alpha + e,  alpha in [-1, 1],  C alpha <= d,  |e_i| <= r_i
//! h = relu(x)
//! z = W h + b
//! ```
//!
//! and obtains a rigorous lower bound on every requested target margin
//! `z[target] - z[challenger]` from AY.  A caller may inspect the diagnostic,
//! but nothing in this module promotes it to a scored verification verdict.
//!
//! Every stored CZ and affine coefficient is passed to AY as the exact dyadic
//! denoted by its `f64`.  Unstable ReLUs use the convex-hull relaxation.  Its
//! rational slope generally is not representable as `f64`, so the encoded
//! slope is accompanied by an exact-rationally computed upward intercept
//! correction over the whole sound preactivation interval.  The resulting
//! line is an outer line; rounded numerics are never trusted as a proof.  The
//! public entry point runtime-probes gradual binary64 underflow before using
//! adjacent-float outward arithmetic and rejects FTZ/DAZ environments.

use std::time::{Duration, Instant};

use ay_milp::{Col, LpSession, Model, Outcome, Sense, SolveOpts};
use ndarray::ArrayView2;
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};

use crate::ConstrainedZonotope64;

/// Absolute implementation caps.  Caller-selected limits must be no larger.
pub const TAIL_LP_HARD_MAX_VALUE_DIM: usize = 512;
/// Absolute alpha-symbol cap.
pub const TAIL_LP_HARD_MAX_ALPHA_DIM: usize = 5_000;
/// Absolute predicate-row cap.
pub const TAIL_LP_HARD_MAX_CONSTRAINTS: usize = 12_000;
/// Absolute dense predicate-storage scan cap.
pub const TAIL_LP_HARD_MAX_CONSTRAINT_ELEMENTS: usize = 32_000_000;
/// Absolute sparse generator cap.
pub const TAIL_LP_HARD_MAX_GENERATOR_NONZEROS: usize = 2_000_000;
/// Absolute predicate nonzero cap.
pub const TAIL_LP_HARD_MAX_CONSTRAINT_NONZEROS: usize = 24_000_000;
/// Absolute tail-output cap.
pub const TAIL_LP_HARD_MAX_OUTPUT_DIM: usize = 64;
/// Absolute unstable-ReLU cap.
pub const TAIL_LP_HARD_MAX_UNSTABLE_RELUS: usize = 512;
/// Absolute AY model-column cap.
pub const TAIL_LP_HARD_MAX_MODEL_COLUMNS: usize = 12_000;
/// Absolute AY model-row cap.
pub const TAIL_LP_HARD_MAX_MODEL_ROWS: usize = 24_000;
/// Absolute AY matrix-nonzero cap.
pub const TAIL_LP_HARD_MAX_MODEL_NONZEROS: usize = 28_000_000;
/// Absolute solve-count cap.
pub const TAIL_LP_HARD_MAX_SOLVES: usize = 63;
/// Absolute wall-clock budget accepted by this experimental primitive.
pub const TAIL_LP_HARD_MAX_WALL_TIME: Duration = Duration::from_mins(5);
/// Absolute AY memory-budget setting accepted by this primitive.
pub const TAIL_LP_HARD_MAX_AY_MEMORY_BYTES: usize = 2_147_483_648;

/// Matches AY's own solver-thread stack headroom.  The worker is detached on
/// hard timeout because AY exposes no safe cancellation hook.
const TAIL_LP_SOLVE_THREAD_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Explicit resource firewall for the unwired tail LP.
///
/// There is intentionally no `Default`: an experimental caller must price
/// every potentially large representation and solve.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeTailLpLimits {
    pub max_value_dim: usize,
    pub max_alpha_dim: usize,
    pub max_constraints: usize,
    pub max_constraint_elements: usize,
    pub max_generator_nonzeros: usize,
    pub max_constraint_nonzeros: usize,
    pub max_output_dim: usize,
    pub max_unstable_relus: usize,
    pub max_model_columns: usize,
    pub max_model_rows: usize,
    pub max_model_nonzeros: usize,
    pub max_solves: usize,
}

/// Explicit execution policy for the unwired tail LP.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeTailLpConfig {
    /// Output whose margin against each selected challenger is minimized.  The
    /// original all-challenger entry point selects every other output.
    pub target_output: usize,
    /// One absolute deadline shared by planning, construction, and all solves.
    /// It starts when the diagnostic function is entered; construction of the
    /// supplied post-affine constrained zonotope is upstream and excluded.
    pub wall_time: Duration,
    /// AY's retained-memory budget setting.
    pub ay_memory_budget_bytes: usize,
    /// Research-only threshold for reporting an exact-ReLU MILP candidate.
    pub exact_milp_binary_cap: usize,
    /// Complete shape and work limits.
    pub limits: ConstrainedZonotopeTailLpLimits,
}

/// Checked resource accounting for the built LP.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeTailLpPlan {
    pub value_dim: usize,
    pub alpha_dim: usize,
    pub constraint_count: usize,
    pub constraint_elements: usize,
    pub generator_nonzeros: usize,
    pub constraint_nonzeros: usize,
    pub output_dim: usize,
    pub inactive_relus: usize,
    pub active_relus: usize,
    pub unstable_relus: usize,
    pub model_columns: usize,
    pub model_rows: usize,
    pub model_nonzeros: usize,
    /// Number of requested challenger objectives and margin rows.
    pub solve_count: usize,
}

/// Why one margin did not receive a rigorous AY lower bound.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TailLpInconclusiveReason {
    DeadlineBeforeSolve,
    HardDeadlineExceeded,
    AyUnknown(String),
    ModelInfeasible,
    ObjectiveUnbounded,
    NonRigorousBound,
    UnexpectedOutcome,
    SolverError(String),
    NotAttemptedAfterEarlierDecline,
}

/// Result for one `target - challenger` margin.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TailLpMarginOutcome {
    /// A mathematically rigorous lower bound on the LP optimum.
    RigorousLowerBound(BigRational),
    /// No proof-bearing lower bound was obtained.
    Inconclusive(TailLpInconclusiveReason),
}

/// One challenger diagnostic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TailLpMarginDiagnostic {
    pub challenger_output: usize,
    pub outcome: TailLpMarginOutcome,
}

/// Cap-based assessment of a future exact-ReLU MILP encoding.
///
/// `within_declared_caps` is not a runtime prediction or a proof.  It only
/// says the standard one-binary-per-unstable-ReLU model fits the caller's
/// declared binary cap and this module's structural hard caps.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TailLpExactMilpAssessment {
    pub relu_binary_count: usize,
    pub configured_binary_cap: usize,
    pub estimated_model_columns: usize,
    pub estimated_model_rows: usize,
    pub estimated_model_nonzeros: usize,
    pub within_declared_caps: bool,
}

/// Complete non-verdict diagnostic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeTailLpDiagnostic {
    pub plan: ConstrainedZonotopeTailLpPlan,
    pub target_output: usize,
    pub margins: Vec<TailLpMarginDiagnostic>,
    /// Smallest available rigorous bound.  Its signed distance from zero is
    /// the useful LP certification gap; unresolved margins are counted below.
    pub minimum_rigorous_lower_bound: Option<BigRational>,
    pub unresolved_margin_count: usize,
    /// True only when all requested challengers received rigorous bounds and
    /// every exact rational bound is strictly positive.  For the subset entry
    /// point this makes no claim about challengers the caller did not request.
    pub all_margins_strictly_positive: bool,
    pub exact_milp: TailLpExactMilpAssessment,
    pub elapsed: Duration,
}

/// Fail-closed construction errors.
#[derive(Debug, thiserror::Error)]
pub enum ConstrainedZonotopeTailLpError {
    #[error("invalid tail-LP configuration for {field}: {message}")]
    InvalidConfig {
        field: &'static str,
        message: &'static str,
    },
    #[error("shape mismatch for {field}: expected {expected:?}, got {got:?}")]
    Shape {
        field: &'static str,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("tail parameter {field}[{index}] must be finite")]
    NonFiniteParameter { field: &'static str, index: usize },
    #[error("configured {resource} limit {configured} exceeds hard cap {hard}")]
    ConfiguredLimitExceedsHardCap {
        resource: &'static str,
        configured: usize,
        hard: usize,
    },
    #[error("{resource} requires {required}, exceeding limit {limit}")]
    LimitExceeded {
        resource: &'static str,
        required: usize,
        limit: usize,
    },
    #[error("resource size overflow while computing {operation}")]
    ResourceOverflow { operation: &'static str },
    #[error("unable to reserve storage for {resource}")]
    AllocationFailure { resource: &'static str },
    #[error(
        "non-finite outward arithmetic at coordinate {coordinate} while computing {operation}"
    )]
    NonFiniteArithmetic {
        coordinate: usize,
        operation: &'static str,
    },
    #[error("unsupported floating-point environment: {requirement}")]
    UnsupportedFloatingPoint { requirement: &'static str },
    #[error("tail-LP deadline expired during {phase}")]
    DeadlineExpired { phase: &'static str },
    #[error("AY LP session construction failed: {message}")]
    SolverSetup { message: String },
}

#[derive(Clone, Debug)]
enum ReluEncoding {
    Inactive,
    Active,
    Unstable { slope: f64, intercept: f64 },
}

#[derive(Clone, Debug)]
struct BuildPlan {
    public: ConstrainedZonotopeTailLpPlan,
    challengers: Vec<usize>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    relu: Vec<ReluEncoding>,
    generator_row_counts: Vec<usize>,
    constraint_row_counts: Vec<usize>,
    output_row_counts: Vec<usize>,
}

/// Build and solve an exact-AY outer LP for a CZ -> ReLU -> affine tail,
/// selecting every affine output other than `config.target_output`.
///
/// This function is intentionally unwired.  `all_margins_strictly_positive`
/// describes only this diagnostic's outer LP and is never consumed by a NY
/// verifier path.
pub fn diagnose_constrained_zonotope_relu_affine_tail_with_ay_lp_unwired(
    domain: &ConstrainedZonotope64,
    output_weights: ArrayView2<'_, f64>,
    output_bias: &[f64],
    config: ConstrainedZonotopeTailLpConfig,
) -> Result<ConstrainedZonotopeTailLpDiagnostic, ConstrainedZonotopeTailLpError> {
    diagnose_constrained_zonotope_relu_affine_tail_with_ay_lp_for_challengers_unwired_impl(
        domain,
        output_weights,
        output_bias,
        None,
        config,
    )
}

/// Build and solve the same exact-AY outer LP for only the requested output
/// challengers.
///
/// `challenger_outputs` must be nonempty, unique, in range, and must not
/// contain `config.target_output`.  Its order is preserved in both the AY
/// incremental solve sequence and the returned margin diagnostics.  This is
/// intentionally unwired and does not promote a subset result to a verifier
/// verdict; callers remain responsible for proving every property clause.
pub fn diagnose_constrained_zonotope_relu_affine_tail_with_ay_lp_for_challengers_unwired(
    domain: &ConstrainedZonotope64,
    output_weights: ArrayView2<'_, f64>,
    output_bias: &[f64],
    challenger_outputs: &[usize],
    config: ConstrainedZonotopeTailLpConfig,
) -> Result<ConstrainedZonotopeTailLpDiagnostic, ConstrainedZonotopeTailLpError> {
    diagnose_constrained_zonotope_relu_affine_tail_with_ay_lp_for_challengers_unwired_impl(
        domain,
        output_weights,
        output_bias,
        Some(challenger_outputs),
        config,
    )
}

fn diagnose_constrained_zonotope_relu_affine_tail_with_ay_lp_for_challengers_unwired_impl(
    domain: &ConstrainedZonotope64,
    output_weights: ArrayView2<'_, f64>,
    output_bias: &[f64],
    challenger_outputs: Option<&[usize]>,
    config: ConstrainedZonotopeTailLpConfig,
) -> Result<ConstrainedZonotopeTailLpDiagnostic, ConstrainedZonotopeTailLpError> {
    // The public contract defines one wall budget from function entry. Keep
    // the floating-point qualification and config validation inside it too,
    // even though both are intentionally small and allocation-free.
    let started = Instant::now();
    require_gradual_underflow()?;
    validate_config(&config)?;
    let deadline = started.checked_add(config.wall_time).ok_or(
        ConstrainedZonotopeTailLpError::InvalidConfig {
            field: "wall_time",
            message: "cannot be represented as an Instant deadline",
        },
    )?;
    let build = plan(
        domain,
        output_weights,
        output_bias,
        challenger_outputs,
        &config,
        deadline,
    )?;
    check_deadline(deadline, "AY model construction")?;
    let (model, margin_columns) = build_model(
        domain,
        output_weights,
        output_bias,
        config.target_output,
        &build,
        deadline,
    )?;
    let margins = solve_margins_with_hard_deadline(
        model,
        margin_columns,
        deadline,
        config.ay_memory_budget_bytes,
    )?;

    let minimum_rigorous_lower_bound = margins
        .iter()
        .filter_map(|margin| match &margin.outcome {
            TailLpMarginOutcome::RigorousLowerBound(bound) => Some(bound),
            TailLpMarginOutcome::Inconclusive(_) => None,
        })
        .min()
        .cloned();
    let unresolved_margin_count = margins
        .iter()
        .filter(|margin| matches!(margin.outcome, TailLpMarginOutcome::Inconclusive(_)))
        .count();
    let all_margins_strictly_positive = margins.len() == build.public.solve_count
        && margins.iter().all(|margin| {
            matches!(
                &margin.outcome,
                TailLpMarginOutcome::RigorousLowerBound(bound) if bound > &BigRational::zero()
            )
        });
    let exact_milp = assess_exact_milp(&build.public, config.exact_milp_binary_cap)?;

    Ok(ConstrainedZonotopeTailLpDiagnostic {
        plan: build.public,
        target_output: config.target_output,
        margins,
        minimum_rigorous_lower_bound,
        unresolved_margin_count,
        all_margins_strictly_positive,
        exact_milp,
        elapsed: started.elapsed(),
    })
}

/// Construct and use the owned AY session on a detached worker, enforcing the
/// diagnostic's remaining wall budget from outside AY.
///
/// AY's cooperative deadline checks can be delayed by one large rational or
/// factorization step.  On timeout the receiver is abandoned and every margin
/// is reported inconclusive.  The worker keeps the same expired AY deadline
/// and normally exits at its next internal poll, but it may retain the model
/// and continue running until then (or until process teardown).  This bounded
/// detached-worker cost is accepted only for this default-off diagnostic; it
/// is what prevents a late AY poll from extending the caller's wall budget.
fn solve_margins_with_hard_deadline(
    model: Model,
    margin_columns: Vec<(usize, Col)>,
    deadline: Instant,
    ay_memory_budget_bytes: usize,
) -> Result<Vec<TailLpMarginDiagnostic>, ConstrainedZonotopeTailLpError> {
    let challengers: Vec<_> = margin_columns
        .iter()
        .map(|&(challenger, _)| challenger)
        .collect();
    match run_with_hard_deadline(deadline, "cz-tail-ay-lp", move || {
        solve_margins_owned(model, margin_columns, deadline, ay_memory_budget_bytes)
    })? {
        Some(margins) => Ok(margins),
        None => {
            tracing::warn!(
                "AY constrained-zonotope tail LP exceeded its hard wall deadline; abandoning \
                 the worker and returning every margin inconclusive"
            );
            Ok(inconclusive_margins(
                &challengers,
                TailLpInconclusiveReason::HardDeadlineExceeded,
            ))
        }
    }
}

fn run_with_hard_deadline<T, F>(
    deadline: Instant,
    label: &'static str,
    work: F,
) -> Result<Option<T>, ConstrainedZonotopeTailLpError>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name(format!("ny-mip-{label}"))
        .stack_size(TAIL_LP_SOLVE_THREAD_STACK_BYTES)
        .spawn(move || {
            // A receiver gone after the deadline makes this send fail; that is
            // the expected detached-worker case and is deliberately ignored.
            let _ = sender.send(work());
        })
        .map_err(|error| ConstrainedZonotopeTailLpError::SolverSetup {
            message: format!("spawning detached AY tail-LP worker ({label}): {error}"),
        })?;
    let remaining = deadline
        .checked_duration_since(Instant::now())
        .unwrap_or(Duration::ZERO);
    match receiver.recv_timeout(remaining) {
        Ok(result) => Ok(Some(result)),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Ok(None),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            Err(ConstrainedZonotopeTailLpError::SolverSetup {
                message: format!(
                    "detached AY tail-LP worker ({label}) exited without a result (panicked)"
                ),
            })
        }
    }
}

fn solve_margins_owned(
    model: Model,
    margin_columns: Vec<(usize, Col)>,
    deadline: Instant,
    ay_memory_budget_bytes: usize,
) -> Vec<TailLpMarginDiagnostic> {
    let challengers: Vec<_> = margin_columns
        .iter()
        .map(|&(challenger, _)| challenger)
        .collect();
    let opts = SolveOpts::new()
        .with_deadline(deadline)
        .with_threads(1)
        .with_determinism(true)
        .with_memory_budget(Some(ay_memory_budget_bytes))
        .with_tree_cert_leaves(0);
    let mut session = match LpSession::new(&model, &opts) {
        Ok(session) => session,
        Err(error) => {
            return inconclusive_margins(
                &challengers,
                TailLpInconclusiveReason::SolverError(error.to_string()),
            );
        }
    };

    let mut margins = Vec::with_capacity(margin_columns.len());
    let mut halted = false;
    for (challenger_output, margin_col) in margin_columns {
        if halted {
            margins.push(TailLpMarginDiagnostic {
                challenger_output,
                outcome: TailLpMarginOutcome::Inconclusive(
                    TailLpInconclusiveReason::NotAttemptedAfterEarlierDecline,
                ),
            });
            continue;
        }
        if Instant::now() >= deadline {
            halted = true;
            margins.push(TailLpMarginDiagnostic {
                challenger_output,
                outcome: TailLpMarginOutcome::Inconclusive(
                    TailLpInconclusiveReason::DeadlineBeforeSolve,
                ),
            });
            continue;
        }
        let outcome = match session.rigorous_bound(margin_col, Sense::Minimize) {
            Ok(Outcome::Bound {
                dual_bound,
                rigorous: true,
            }) => TailLpMarginOutcome::RigorousLowerBound(dual_bound),
            Ok(Outcome::Bound {
                rigorous: false, ..
            }) => {
                halted = true;
                TailLpMarginOutcome::Inconclusive(TailLpInconclusiveReason::NonRigorousBound)
            }
            Ok(Outcome::Unknown { reason }) => {
                halted = true;
                TailLpMarginOutcome::Inconclusive(TailLpInconclusiveReason::AyUnknown(format!(
                    "{reason:?}"
                )))
            }
            Ok(Outcome::Infeasible { .. }) => {
                halted = true;
                TailLpMarginOutcome::Inconclusive(TailLpInconclusiveReason::ModelInfeasible)
            }
            Ok(Outcome::Unbounded) => {
                halted = true;
                TailLpMarginOutcome::Inconclusive(TailLpInconclusiveReason::ObjectiveUnbounded)
            }
            Ok(_) => {
                halted = true;
                TailLpMarginOutcome::Inconclusive(TailLpInconclusiveReason::UnexpectedOutcome)
            }
            Err(error) => {
                halted = true;
                TailLpMarginOutcome::Inconclusive(TailLpInconclusiveReason::SolverError(
                    error.to_string(),
                ))
            }
        };
        margins.push(TailLpMarginDiagnostic {
            challenger_output,
            outcome,
        });
    }
    margins
}

fn inconclusive_margins(
    challengers: &[usize],
    reason: TailLpInconclusiveReason,
) -> Vec<TailLpMarginDiagnostic> {
    challengers
        .iter()
        .map(|&challenger_output| TailLpMarginDiagnostic {
            challenger_output,
            outcome: TailLpMarginOutcome::Inconclusive(reason.clone()),
        })
        .collect()
}

fn validate_config(
    config: &ConstrainedZonotopeTailLpConfig,
) -> Result<(), ConstrainedZonotopeTailLpError> {
    if config.wall_time.is_zero() {
        return Err(ConstrainedZonotopeTailLpError::InvalidConfig {
            field: "wall_time",
            message: "must be nonzero",
        });
    }
    if config.wall_time > TAIL_LP_HARD_MAX_WALL_TIME {
        return Err(ConstrainedZonotopeTailLpError::InvalidConfig {
            field: "wall_time",
            message: "exceeds the 300-second hard cap",
        });
    }
    if config.ay_memory_budget_bytes == 0
        || config.ay_memory_budget_bytes > TAIL_LP_HARD_MAX_AY_MEMORY_BYTES
    {
        return Err(ConstrainedZonotopeTailLpError::InvalidConfig {
            field: "ay_memory_budget_bytes",
            message: "must be in 1..=2 GiB",
        });
    }
    for (resource, configured, hard) in [
        (
            "value dimension",
            config.limits.max_value_dim,
            TAIL_LP_HARD_MAX_VALUE_DIM,
        ),
        (
            "alpha dimension",
            config.limits.max_alpha_dim,
            TAIL_LP_HARD_MAX_ALPHA_DIM,
        ),
        (
            "constraint rows",
            config.limits.max_constraints,
            TAIL_LP_HARD_MAX_CONSTRAINTS,
        ),
        (
            "constraint elements",
            config.limits.max_constraint_elements,
            TAIL_LP_HARD_MAX_CONSTRAINT_ELEMENTS,
        ),
        (
            "generator nonzeros",
            config.limits.max_generator_nonzeros,
            TAIL_LP_HARD_MAX_GENERATOR_NONZEROS,
        ),
        (
            "constraint nonzeros",
            config.limits.max_constraint_nonzeros,
            TAIL_LP_HARD_MAX_CONSTRAINT_NONZEROS,
        ),
        (
            "output dimension",
            config.limits.max_output_dim,
            TAIL_LP_HARD_MAX_OUTPUT_DIM,
        ),
        (
            "unstable ReLUs",
            config.limits.max_unstable_relus,
            TAIL_LP_HARD_MAX_UNSTABLE_RELUS,
        ),
        (
            "model columns",
            config.limits.max_model_columns,
            TAIL_LP_HARD_MAX_MODEL_COLUMNS,
        ),
        (
            "model rows",
            config.limits.max_model_rows,
            TAIL_LP_HARD_MAX_MODEL_ROWS,
        ),
        (
            "model nonzeros",
            config.limits.max_model_nonzeros,
            TAIL_LP_HARD_MAX_MODEL_NONZEROS,
        ),
        (
            "solve count",
            config.limits.max_solves,
            TAIL_LP_HARD_MAX_SOLVES,
        ),
    ] {
        if configured > hard {
            return Err(
                ConstrainedZonotopeTailLpError::ConfiguredLimitExceedsHardCap {
                    resource,
                    configured,
                    hard,
                },
            );
        }
    }
    Ok(())
}

fn plan(
    domain: &ConstrainedZonotope64,
    output_weights: ArrayView2<'_, f64>,
    output_bias: &[f64],
    challenger_outputs: Option<&[usize]>,
    config: &ConstrainedZonotopeTailLpConfig,
    deadline: Instant,
) -> Result<BuildPlan, ConstrainedZonotopeTailLpError> {
    let value_dim = domain.value_dim();
    let alpha_dim = domain.alpha_dim();
    let constraint_count = domain.constraint_count();
    let output_dim = output_weights.nrows();
    if output_weights.ncols() != value_dim {
        return Err(ConstrainedZonotopeTailLpError::Shape {
            field: "output_weights",
            expected: vec![output_dim, value_dim],
            got: output_weights.shape().to_vec(),
        });
    }
    if output_bias.len() != output_dim {
        return Err(ConstrainedZonotopeTailLpError::Shape {
            field: "output_bias",
            expected: vec![output_dim],
            got: vec![output_bias.len()],
        });
    }
    if output_dim < 2 {
        return Err(ConstrainedZonotopeTailLpError::InvalidConfig {
            field: "output dimension",
            message: "must contain a target and at least one challenger",
        });
    }
    if config.target_output >= output_dim {
        return Err(ConstrainedZonotopeTailLpError::InvalidConfig {
            field: "target_output",
            message: "is outside the affine output dimension",
        });
    }
    check_limit("value dimension", value_dim, config.limits.max_value_dim)?;
    check_limit("alpha dimension", alpha_dim, config.limits.max_alpha_dim)?;
    check_limit(
        "constraint rows",
        constraint_count,
        config.limits.max_constraints,
    )?;
    check_limit("output dimension", output_dim, config.limits.max_output_dim)?;
    let solve_count = challenger_outputs.map_or(output_dim - 1, <[usize]>::len);
    if solve_count == 0 {
        return Err(ConstrainedZonotopeTailLpError::InvalidConfig {
            field: "challenger_outputs",
            message: "must contain at least one output",
        });
    }
    check_limit("solve count", solve_count, config.limits.max_solves)?;
    let mut challengers = Vec::new();
    try_reserve(&mut challengers, solve_count, "challenger outputs")?;
    if let Some(requested) = challenger_outputs {
        let mut seen = Vec::new();
        try_reserve(&mut seen, output_dim, "challenger membership")?;
        seen.resize(output_dim, false);
        for &challenger in requested {
            if challenger >= output_dim {
                return Err(ConstrainedZonotopeTailLpError::InvalidConfig {
                    field: "challenger_outputs",
                    message: "contains an output outside the affine output dimension",
                });
            }
            if challenger == config.target_output {
                return Err(ConstrainedZonotopeTailLpError::InvalidConfig {
                    field: "challenger_outputs",
                    message: "must not contain the target output",
                });
            }
            if seen[challenger] {
                return Err(ConstrainedZonotopeTailLpError::InvalidConfig {
                    field: "challenger_outputs",
                    message: "must contain unique outputs",
                });
            }
            seen[challenger] = true;
            challengers.push(challenger);
        }
    } else {
        challengers.extend((0..output_dim).filter(|&output| output != config.target_output));
    }
    let constraint_elements =
        checked_product(constraint_count, alpha_dim, "constraint_count * alpha_dim")?;
    check_limit(
        "constraint elements",
        constraint_elements,
        config.limits.max_constraint_elements,
    )?;

    let mut generator_row_counts = try_zero_vec(value_dim, "generator row counts")?;
    let mut radii = Vec::new();
    try_reserve(&mut radii, value_dim, "preactivation radii")?;
    radii.extend_from_slice(domain.box_remainder());
    let mut generator_nonzeros = 0_usize;
    for (generator_index, generator) in domain.generators().iter().enumerate() {
        if generator_index & 0x3ff == 0 {
            check_deadline(deadline, "generator accounting")?;
        }
        generator_nonzeros =
            checked_sum(generator_nonzeros, generator.nnz(), "generator nonzeros")?;
        check_limit(
            "generator nonzeros",
            generator_nonzeros,
            config.limits.max_generator_nonzeros,
        )?;
        for (coordinate, coefficient) in generator.entries() {
            generator_row_counts[coordinate] =
                checked_sum(generator_row_counts[coordinate], 1, "generator row count")?;
            radii[coordinate] =
                add_nonnegative_upward(radii[coordinate], coefficient.abs(), coordinate)?;
        }
    }

    let mut lower = Vec::new();
    let mut upper = Vec::new();
    let mut relu = Vec::new();
    try_reserve(&mut lower, value_dim, "preactivation lower bounds")?;
    try_reserve(&mut upper, value_dim, "preactivation upper bounds")?;
    try_reserve(&mut relu, value_dim, "ReLU encodings")?;
    let mut inactive_relus = 0_usize;
    let mut active_relus = 0_usize;
    let mut unstable_relus = 0_usize;
    for coordinate in 0..value_dim {
        if coordinate & 0x3f == 0 {
            check_deadline(deadline, "preactivation bound construction")?;
        }
        let center = exact_finite(domain.center()[coordinate], coordinate, "center conversion")?;
        let radius = exact_finite(radii[coordinate], coordinate, "radius conversion")?;
        let lo = floor_finite(
            &(center.clone() - &radius),
            coordinate,
            "lower-bound conversion",
        )?;
        let hi = ceil_finite(&(center + radius), coordinate, "upper-bound conversion")?;
        lower.push(lo);
        upper.push(hi);
        if hi <= 0.0 {
            inactive_relus += 1;
            relu.push(ReluEncoding::Inactive);
        } else if lo >= 0.0 {
            active_relus += 1;
            relu.push(ReluEncoding::Active);
        } else {
            unstable_relus += 1;
            let (slope, intercept) = outward_relu_upper_line(lo, hi, coordinate)?;
            relu.push(ReluEncoding::Unstable { slope, intercept });
        }
    }
    check_limit(
        "unstable ReLUs",
        unstable_relus,
        config.limits.max_unstable_relus,
    )?;

    let mut constraint_row_counts = try_zero_vec(constraint_count, "constraint row counts")?;
    let mut constraint_nonzeros = 0_usize;
    for (index, coefficient) in domain.constraints().iter().copied().enumerate() {
        if index & 0x3fff == 0 {
            check_deadline(deadline, "constraint accounting")?;
        }
        if coefficient != 0.0 {
            let row = index / alpha_dim.max(1);
            constraint_row_counts[row] =
                checked_sum(constraint_row_counts[row], 1, "constraint row count")?;
            constraint_nonzeros = checked_sum(constraint_nonzeros, 1, "constraint nonzeros")?;
            check_limit(
                "constraint nonzeros",
                constraint_nonzeros,
                config.limits.max_constraint_nonzeros,
            )?;
        }
    }

    let mut output_row_counts = try_zero_vec(output_dim, "output row counts")?;
    for (index, coefficient) in output_weights.iter().copied().enumerate() {
        if index & 0xfff == 0 {
            check_deadline(deadline, "affine parameter accounting")?;
        }
        if !coefficient.is_finite() {
            return Err(ConstrainedZonotopeTailLpError::NonFiniteParameter {
                field: "output_weights",
                index,
            });
        }
        if coefficient != 0.0 {
            let row = index / value_dim.max(1);
            output_row_counts[row] = checked_sum(output_row_counts[row], 1, "output row count")?;
        }
    }
    for (index, &bias) in output_bias.iter().enumerate() {
        if !bias.is_finite() {
            return Err(ConstrainedZonotopeTailLpError::NonFiniteParameter {
                field: "output_bias",
                index,
            });
        }
    }

    let model_columns = checked_sum(
        checked_sum(
            checked_sum(
                checked_sum(
                    checked_sum(alpha_dim, value_dim, "alpha + x columns")?,
                    value_dim,
                    "alpha + x + error columns",
                )?,
                value_dim,
                "alpha + x + error + ReLU columns",
            )?,
            output_dim,
            "tail output columns",
        )?,
        solve_count,
        "margin columns",
    )?;
    check_limit(
        "model columns",
        model_columns,
        config.limits.max_model_columns,
    )?;
    let relu_rows = checked_sum(
        active_relus,
        checked_product(unstable_relus, 2, "two unstable-ReLU rows")?,
        "ReLU rows",
    )?;
    let model_rows = checked_sum(
        checked_sum(
            checked_sum(
                checked_sum(constraint_count, value_dim, "CZ rows")?,
                relu_rows,
                "CZ + ReLU rows",
            )?,
            output_dim,
            "tail affine rows",
        )?,
        solve_count,
        "margin rows",
    )?;
    check_limit("model rows", model_rows, config.limits.max_model_rows)?;

    let output_weight_nonzeros = output_row_counts.iter().try_fold(0_usize, |sum, &count| {
        checked_sum(sum, count, "output weight nonzeros")
    })?;
    let model_nonzeros = checked_sum(
        checked_sum(
            checked_sum(
                checked_sum(
                    checked_sum(
                        constraint_nonzeros,
                        checked_sum(
                            generator_nonzeros,
                            checked_product(value_dim, 2, "x/error equality terms")?,
                            "CZ equality nonzeros",
                        )?,
                        "predicate + CZ equality nonzeros",
                    )?,
                    checked_sum(
                        checked_product(active_relus, 2, "active-ReLU terms")?,
                        checked_product(unstable_relus, 4, "unstable-ReLU terms")?,
                        "ReLU nonzeros",
                    )?,
                    "preactivation + ReLU nonzeros",
                )?,
                checked_sum(output_weight_nonzeros, output_dim, "affine row nonzeros")?,
                "tail affine model nonzeros",
            )?,
            checked_product(solve_count, 3, "margin row nonzeros")?,
            "complete model nonzeros",
        )?,
        0,
        "complete model nonzeros",
    )?;
    check_limit(
        "model nonzeros",
        model_nonzeros,
        config.limits.max_model_nonzeros,
    )?;

    Ok(BuildPlan {
        public: ConstrainedZonotopeTailLpPlan {
            value_dim,
            alpha_dim,
            constraint_count,
            constraint_elements,
            generator_nonzeros,
            constraint_nonzeros,
            output_dim,
            inactive_relus,
            active_relus,
            unstable_relus,
            model_columns,
            model_rows,
            model_nonzeros,
            solve_count,
        },
        challengers,
        lower,
        upper,
        relu,
        generator_row_counts,
        constraint_row_counts,
        output_row_counts,
    })
}

fn build_model(
    domain: &ConstrainedZonotope64,
    output_weights: ArrayView2<'_, f64>,
    output_bias: &[f64],
    target_output: usize,
    build: &BuildPlan,
    deadline: Instant,
) -> Result<(Model, Vec<(usize, Col)>), ConstrainedZonotopeTailLpError> {
    let mut model = Model::new();
    let alpha_columns: Vec<_> = (0..build.public.alpha_dim)
        .map(|_| model.add_col(-1.0, 1.0))
        .collect();
    let x_columns: Vec<_> = build
        .lower
        .iter()
        .zip(&build.upper)
        .map(|(&lower, &upper)| model.add_col(lower, upper))
        .collect();
    let error_columns: Vec<_> = domain
        .box_remainder()
        .iter()
        .map(|&radius| model.add_col(-radius, radius))
        .collect();
    let relu_columns: Vec<_> = build
        .relu
        .iter()
        .enumerate()
        .map(|(coordinate, encoding)| match encoding {
            ReluEncoding::Inactive => model.add_col(0.0, 0.0),
            ReluEncoding::Active => {
                model.add_col(build.lower[coordinate].max(0.0), build.upper[coordinate])
            }
            ReluEncoding::Unstable { .. } => model.add_col(0.0, build.upper[coordinate]),
        })
        .collect();
    let output_columns: Vec<_> = (0..build.public.output_dim)
        .map(|_| model.add_col(f64::NEG_INFINITY, f64::INFINITY))
        .collect();
    let margin_columns: Vec<_> = build
        .challengers
        .iter()
        .map(|&challenger| (challenger, model.add_col(f64::NEG_INFINITY, f64::INFINITY)))
        .collect();
    debug_assert_eq!(model.num_cols(), build.public.model_columns);

    for row in 0..build.public.constraint_count {
        if row & 0x3f == 0 {
            check_deadline(deadline, "predicate-row lowering")?;
        }
        let mut terms = Vec::new();
        try_reserve(
            &mut terms,
            build.constraint_row_counts[row],
            "predicate row terms",
        )?;
        for alpha in 0..build.public.alpha_dim {
            let coefficient = domain.constraints()[[row, alpha]];
            if coefficient != 0.0 {
                terms.push((alpha_columns[alpha], coefficient));
            }
        }
        model.add_row(f64::NEG_INFINITY, domain.rhs()[row], &terms);
    }

    let mut generator_rows = Vec::new();
    try_reserve(
        &mut generator_rows,
        build.public.value_dim,
        "generator row buffers",
    )?;
    for (coordinate, &count) in build.generator_row_counts.iter().enumerate() {
        let mut row = Vec::new();
        try_reserve(
            &mut row,
            checked_sum(count, 2, "generator equality row capacity")?,
            "generator equality row terms",
        )?;
        row.push((x_columns[coordinate], 1.0));
        row.push((error_columns[coordinate], -1.0));
        generator_rows.push(row);
    }
    for (alpha, generator) in domain.generators().iter().enumerate() {
        if alpha & 0x3ff == 0 {
            check_deadline(deadline, "generator-row lowering")?;
        }
        for (coordinate, coefficient) in generator.entries() {
            generator_rows[coordinate].push((alpha_columns[alpha], -coefficient));
        }
    }
    for (coordinate, terms) in generator_rows.iter().enumerate() {
        model.add_row(
            domain.center()[coordinate],
            domain.center()[coordinate],
            terms,
        );
    }

    for coordinate in 0..build.public.value_dim {
        match build.relu[coordinate] {
            ReluEncoding::Inactive => {}
            ReluEncoding::Active => {
                model.add_row(
                    0.0,
                    0.0,
                    &[
                        (relu_columns[coordinate], 1.0),
                        (x_columns[coordinate], -1.0),
                    ],
                );
            }
            ReluEncoding::Unstable { slope, intercept } => {
                model.add_row(
                    0.0,
                    f64::INFINITY,
                    &[
                        (relu_columns[coordinate], 1.0),
                        (x_columns[coordinate], -1.0),
                    ],
                );
                model.add_row(
                    f64::NEG_INFINITY,
                    intercept,
                    &[
                        (relu_columns[coordinate], 1.0),
                        (x_columns[coordinate], -slope),
                    ],
                );
            }
        }
    }

    for output in 0..build.public.output_dim {
        if output & 0xf == 0 {
            check_deadline(deadline, "tail-affine lowering")?;
        }
        let mut terms = Vec::new();
        try_reserve(
            &mut terms,
            checked_sum(build.output_row_counts[output], 1, "output row capacity")?,
            "tail affine row terms",
        )?;
        terms.push((output_columns[output], 1.0));
        for coordinate in 0..build.public.value_dim {
            let weight = output_weights[[output, coordinate]];
            if weight != 0.0 {
                terms.push((relu_columns[coordinate], -weight));
            }
        }
        model.add_row(output_bias[output], output_bias[output], &terms);
    }
    for &(challenger, margin) in &margin_columns {
        model.add_row(
            0.0,
            0.0,
            &[
                (margin, 1.0),
                (output_columns[target_output], -1.0),
                (output_columns[challenger], 1.0),
            ],
        );
    }
    debug_assert_eq!(model.num_rows(), build.public.model_rows);
    Ok((model, margin_columns))
}

fn assess_exact_milp(
    plan: &ConstrainedZonotopeTailLpPlan,
    configured_binary_cap: usize,
) -> Result<TailLpExactMilpAssessment, ConstrainedZonotopeTailLpError> {
    let estimated_model_columns = checked_sum(
        plan.model_columns,
        plan.unstable_relus,
        "exact-MILP columns",
    )?;
    let estimated_model_rows =
        checked_sum(plan.model_rows, plan.unstable_relus, "exact-MILP rows")?;
    let estimated_model_nonzeros = checked_sum(
        plan.model_nonzeros,
        checked_product(
            plan.unstable_relus,
            3,
            "exact-MILP additional ReLU nonzeros",
        )?,
        "exact-MILP nonzeros",
    )?;
    let within_declared_caps = plan.unstable_relus <= configured_binary_cap
        && estimated_model_columns <= TAIL_LP_HARD_MAX_MODEL_COLUMNS
        && estimated_model_rows <= TAIL_LP_HARD_MAX_MODEL_ROWS
        && estimated_model_nonzeros <= TAIL_LP_HARD_MAX_MODEL_NONZEROS;
    Ok(TailLpExactMilpAssessment {
        relu_binary_count: plan.unstable_relus,
        configured_binary_cap,
        estimated_model_columns,
        estimated_model_rows,
        estimated_model_nonzeros,
        within_declared_caps,
    })
}

/// Sum nonnegative exact-dyadic terms with one-ULP upward protection whenever
/// the hardware addition may round.  The returned dyadic encloses the exact
/// sum and is later used only to widen coordinate bounds.
fn add_nonnegative_upward(
    left: f64,
    right: f64,
    coordinate: usize,
) -> Result<f64, ConstrainedZonotopeTailLpError> {
    debug_assert!(left >= 0.0 && right >= 0.0);
    if right == 0.0 {
        return Ok(left);
    }
    if left == 0.0 {
        return Ok(right);
    }
    let rounded = left + right;
    if !rounded.is_finite() {
        return Err(ConstrainedZonotopeTailLpError::NonFiniteArithmetic {
            coordinate,
            operation: "generator-radius addition",
        });
    }
    let outward = rounded.next_up();
    if !outward.is_finite() {
        return Err(ConstrainedZonotopeTailLpError::NonFiniteArithmetic {
            coordinate,
            operation: "generator-radius upward rounding",
        });
    }
    Ok(outward)
}

/// Reject FTZ/DAZ before adjacent-float intervals are used as proof objects.
///
/// `black_box` keeps these operations in the active scalar environment rather
/// than allowing constant folding under the compiler's abstract IEEE model.
fn require_gradual_underflow() -> Result<(), ConstrainedZonotopeTailLpError> {
    let half = std::hint::black_box(0.5_f64);
    let min_normal = std::hint::black_box(f64::MIN_POSITIVE);
    let min_subnormal = std::hint::black_box(f64::from_bits(1));
    let two_subnormals = std::hint::black_box(f64::from_bits(2));

    let half_min_normal = std::hint::black_box(min_normal * half);
    let recovered_min_subnormal = std::hint::black_box(two_subnormals * half);
    let added_subnormals = std::hint::black_box(min_subnormal + min_subnormal);
    if half_min_normal.to_bits() != 0x0008_0000_0000_0000
        || recovered_min_subnormal.to_bits() != 1
        || added_subnormals.to_bits() != 2
    {
        return Err(ConstrainedZonotopeTailLpError::UnsupportedFloatingPoint {
            requirement: "IEEE-754 binary64 gradual underflow (FTZ/DAZ disabled)",
        });
    }
    Ok(())
}

/// Return `s_hat, b_hat` such that
/// `relu(x) <= s_hat*x + b_hat` throughout exact-dyadic `[lower, upper]`.
fn outward_relu_upper_line(
    lower: f64,
    upper: f64,
    coordinate: usize,
) -> Result<(f64, f64), ConstrainedZonotopeTailLpError> {
    debug_assert!(lower < 0.0 && upper > 0.0);
    let lower_r = exact_finite(lower, coordinate, "ReLU lower conversion")?;
    let upper_r = exact_finite(upper, coordinate, "ReLU upper conversion")?;
    let denominator = &upper_r - &lower_r;
    let exact_slope = &upper_r / &denominator;
    let exact_intercept = -(&lower_r * &upper_r) / denominator;
    let slope = nearest_finite(&exact_slope, coordinate, "ReLU hull slope conversion")?;
    let slope_r = exact_finite(slope, coordinate, "rounded ReLU slope conversion")?;
    let slope_deficit = &exact_slope - slope_r;
    let at_lower = &slope_deficit * &lower_r;
    let at_upper = &slope_deficit * &upper_r;
    let correction = if at_lower >= at_upper {
        at_lower
    } else {
        at_upper
    };
    let required_intercept = exact_intercept + correction;
    let intercept = ceil_finite(
        &required_intercept,
        coordinate,
        "ReLU hull intercept correction",
    )?;
    Ok((slope, intercept))
}

fn exact_finite(
    value: f64,
    coordinate: usize,
    operation: &'static str,
) -> Result<BigRational, ConstrainedZonotopeTailLpError> {
    BigRational::from_float(value).ok_or(ConstrainedZonotopeTailLpError::NonFiniteArithmetic {
        coordinate,
        operation,
    })
}

fn nearest_finite(
    value: &BigRational,
    coordinate: usize,
    operation: &'static str,
) -> Result<f64, ConstrainedZonotopeTailLpError> {
    let candidate = value
        .to_f64()
        .ok_or(ConstrainedZonotopeTailLpError::NonFiniteArithmetic {
            coordinate,
            operation,
        })?;
    if !candidate.is_finite() {
        return Err(ConstrainedZonotopeTailLpError::NonFiniteArithmetic {
            coordinate,
            operation,
        });
    }
    Ok(candidate)
}

fn floor_finite(
    value: &BigRational,
    coordinate: usize,
    operation: &'static str,
) -> Result<f64, ConstrainedZonotopeTailLpError> {
    let mut candidate = nearest_finite(value, coordinate, operation)?;
    if exact_finite(candidate, coordinate, operation)? > *value {
        candidate = candidate.next_down();
        if !candidate.is_finite() || exact_finite(candidate, coordinate, operation)? > *value {
            return Err(ConstrainedZonotopeTailLpError::NonFiniteArithmetic {
                coordinate,
                operation,
            });
        }
    }
    Ok(candidate)
}

fn ceil_finite(
    value: &BigRational,
    coordinate: usize,
    operation: &'static str,
) -> Result<f64, ConstrainedZonotopeTailLpError> {
    let mut candidate = nearest_finite(value, coordinate, operation)?;
    if exact_finite(candidate, coordinate, operation)? < *value {
        candidate = candidate.next_up();
        if !candidate.is_finite() || exact_finite(candidate, coordinate, operation)? < *value {
            return Err(ConstrainedZonotopeTailLpError::NonFiniteArithmetic {
                coordinate,
                operation,
            });
        }
    }
    Ok(candidate)
}

fn check_deadline(
    deadline: Instant,
    phase: &'static str,
) -> Result<(), ConstrainedZonotopeTailLpError> {
    if Instant::now() >= deadline {
        Err(ConstrainedZonotopeTailLpError::DeadlineExpired { phase })
    } else {
        Ok(())
    }
}

fn check_limit(
    resource: &'static str,
    required: usize,
    limit: usize,
) -> Result<(), ConstrainedZonotopeTailLpError> {
    if required > limit {
        Err(ConstrainedZonotopeTailLpError::LimitExceeded {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}

fn checked_sum(
    left: usize,
    right: usize,
    operation: &'static str,
) -> Result<usize, ConstrainedZonotopeTailLpError> {
    left.checked_add(right)
        .ok_or(ConstrainedZonotopeTailLpError::ResourceOverflow { operation })
}

fn checked_product(
    left: usize,
    right: usize,
    operation: &'static str,
) -> Result<usize, ConstrainedZonotopeTailLpError> {
    left.checked_mul(right)
        .ok_or(ConstrainedZonotopeTailLpError::ResourceOverflow { operation })
}

fn try_reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    resource: &'static str,
) -> Result<(), ConstrainedZonotopeTailLpError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| ConstrainedZonotopeTailLpError::AllocationFailure { resource })
}

fn try_zero_vec(
    len: usize,
    resource: &'static str,
) -> Result<Vec<usize>, ConstrainedZonotopeTailLpError> {
    let mut values = Vec::new();
    try_reserve(&mut values, len, resource)?;
    values.resize(len, 0);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};
    use num_traits::Signed;
    use proptest::prelude::*;

    use super::*;

    fn test_limits() -> ConstrainedZonotopeTailLpLimits {
        ConstrainedZonotopeTailLpLimits {
            max_value_dim: 8,
            max_alpha_dim: 8,
            max_constraints: 8,
            max_constraint_elements: 64,
            max_generator_nonzeros: 64,
            max_constraint_nonzeros: 64,
            max_output_dim: 8,
            max_unstable_relus: 8,
            max_model_columns: 64,
            max_model_rows: 64,
            max_model_nonzeros: 256,
            max_solves: 7,
        }
    }

    fn test_config() -> ConstrainedZonotopeTailLpConfig {
        ConstrainedZonotopeTailLpConfig {
            target_output: 0,
            wall_time: Duration::from_secs(10),
            ay_memory_budget_bytes: 64 * 1024 * 1024,
            exact_milp_binary_cap: 4,
            limits: test_limits(),
        }
    }

    fn one_alpha_domain(with_constraint: bool) -> ConstrainedZonotope64 {
        let (constraints, rhs) = if with_constraint {
            // -alpha <= -1/2, hence alpha >= 1/2.
            (array![[-1.0]], vec![-0.5])
        } else {
            (Array2::zeros((0, 1)), Vec::new())
        };
        ConstrainedZonotope64::try_new(vec![0.0], vec![vec![(0, 1.0)]], constraints, rhs, vec![0.0])
            .expect("valid test CZ")
    }

    #[test]
    fn exact_predicate_and_relu_hull_prove_positive_margin() {
        let domain = one_alpha_domain(true);
        let weights = array![[1.0], [0.0]];
        let diagnostic = diagnose_constrained_zonotope_relu_affine_tail_with_ay_lp_unwired(
            &domain,
            weights.view(),
            &[0.0, 0.25],
            test_config(),
        )
        .expect("tail LP");

        assert_eq!(diagnostic.plan.unstable_relus, 1);
        assert_eq!(diagnostic.plan.constraint_nonzeros, 1);
        assert_eq!(diagnostic.plan.model_columns, 7);
        assert_eq!(diagnostic.plan.model_rows, 7);
        assert_eq!(diagnostic.plan.model_nonzeros, 14);
        assert_eq!(diagnostic.unresolved_margin_count, 0);
        assert!(diagnostic.all_margins_strictly_positive);
        assert_eq!(
            diagnostic.minimum_rigorous_lower_bound,
            Some(BigRational::new(1.into(), 4.into()))
        );
        assert_eq!(diagnostic.exact_milp.relu_binary_count, 1);
        assert!(diagnostic.exact_milp.within_declared_caps);
    }

    #[test]
    fn challenger_subset_preserves_order_and_prices_only_requested_margins() {
        let domain = one_alpha_domain(true);
        let weights = array![[1.0], [0.0], [0.0], [0.0]];
        let diagnostic =
            diagnose_constrained_zonotope_relu_affine_tail_with_ay_lp_for_challengers_unwired(
                &domain,
                weights.view(),
                &[0.0, 0.25, 0.125, 0.375],
                &[3, 1],
                test_config(),
            )
            .expect("subset tail LP");

        assert_eq!(diagnostic.plan.output_dim, 4);
        assert_eq!(diagnostic.plan.solve_count, 2);
        assert_eq!(diagnostic.plan.model_columns, 10);
        assert_eq!(diagnostic.plan.model_rows, 10);
        assert_eq!(diagnostic.plan.model_nonzeros, 19);
        assert_eq!(
            diagnostic
                .margins
                .iter()
                .map(|margin| margin.challenger_output)
                .collect::<Vec<_>>(),
            vec![3, 1]
        );
        assert_eq!(
            diagnostic.minimum_rigorous_lower_bound,
            Some(BigRational::new(1.into(), 8.into()))
        );
        assert!(diagnostic.all_margins_strictly_positive);
    }

    #[test]
    fn original_entry_point_retains_all_challengers_in_output_order() {
        let domain = one_alpha_domain(true);
        let weights = array![[1.0], [0.0], [0.0], [0.0]];
        let diagnostic = diagnose_constrained_zonotope_relu_affine_tail_with_ay_lp_unwired(
            &domain,
            weights.view(),
            &[0.0, 0.25, 0.125, 0.375],
            test_config(),
        )
        .expect("all-challenger tail LP");

        assert_eq!(diagnostic.plan.solve_count, 3);
        assert_eq!(diagnostic.plan.model_columns, 11);
        assert_eq!(diagnostic.plan.model_rows, 11);
        assert_eq!(diagnostic.plan.model_nonzeros, 22);
        assert_eq!(
            diagnostic
                .margins
                .iter()
                .map(|margin| margin.challenger_output)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert!(diagnostic.all_margins_strictly_positive);
    }

    #[test]
    fn challenger_subset_rejects_empty_duplicate_target_and_out_of_range() {
        let domain = one_alpha_domain(false);
        let weights = array![[1.0], [0.0], [0.0], [0.0]];
        let bias = [0.0; 4];

        for (challengers, expected_message) in [
            (&[][..], "must contain at least one output"),
            (&[1, 1][..], "must contain unique outputs"),
            (&[0][..], "must not contain the target output"),
            (
                &[4][..],
                "contains an output outside the affine output dimension",
            ),
        ] {
            let error =
                diagnose_constrained_zonotope_relu_affine_tail_with_ay_lp_for_challengers_unwired(
                    &domain,
                    weights.view(),
                    &bias,
                    challengers,
                    test_config(),
                )
                .expect_err("invalid challenger subset must reject");
            assert!(matches!(
                error,
                ConstrainedZonotopeTailLpError::InvalidConfig {
                    field: "challenger_outputs",
                    message,
                } if message == expected_message
            ));
        }
    }

    #[test]
    fn challenger_subset_solve_cap_is_applied_to_requested_count() {
        let domain = one_alpha_domain(false);
        let weights = array![[1.0], [0.0], [0.0], [0.0]];
        let mut config = test_config();
        config.limits.max_solves = 1;
        let error =
            diagnose_constrained_zonotope_relu_affine_tail_with_ay_lp_for_challengers_unwired(
                &domain,
                weights.view(),
                &[0.0; 4],
                &[3, 1],
                config,
            )
            .expect_err("requested solve count must respect the caller cap");
        assert!(matches!(
            error,
            ConstrainedZonotopeTailLpError::LimitExceeded {
                resource: "solve count",
                required: 2,
                limit: 1,
            }
        ));
    }

    #[test]
    fn absent_predicate_exposes_nonpositive_gap_without_claiming_proof() {
        let domain = one_alpha_domain(false);
        let weights = array![[1.0], [0.0]];
        let diagnostic = diagnose_constrained_zonotope_relu_affine_tail_with_ay_lp_unwired(
            &domain,
            weights.view(),
            &[0.0, 0.25],
            test_config(),
        )
        .expect("tail LP");

        assert!(!diagnostic.all_margins_strictly_positive);
        assert_eq!(diagnostic.unresolved_margin_count, 0);
        assert_eq!(
            diagnostic.minimum_rigorous_lower_bound,
            Some(BigRational::new((-1).into(), 4.into()))
        );
    }

    #[test]
    fn independent_remainder_variable_reaches_the_margin_bound() {
        let domain = ConstrainedZonotope64::try_new(
            vec![1.0],
            Vec::new(),
            Array2::zeros((0, 0)),
            Vec::new(),
            vec![0.25],
        )
        .expect("valid remainder CZ");
        let weights = array![[1.0], [0.0]];
        let diagnostic = diagnose_constrained_zonotope_relu_affine_tail_with_ay_lp_unwired(
            &domain,
            weights.view(),
            &[0.0, 0.5],
            test_config(),
        )
        .expect("tail LP");

        assert_eq!(diagnostic.plan.active_relus, 1);
        assert_eq!(diagnostic.plan.unstable_relus, 0);
        assert_eq!(
            diagnostic.minimum_rigorous_lower_bound,
            Some(BigRational::new(1.into(), 4.into()))
        );
        assert!(diagnostic.all_margins_strictly_positive);
    }

    #[test]
    fn resource_cap_rejects_before_model_construction() {
        let domain = one_alpha_domain(false);
        let weights = array![[1.0], [0.0]];
        let mut config = test_config();
        config.limits.max_generator_nonzeros = 0;
        let error = diagnose_constrained_zonotope_relu_affine_tail_with_ay_lp_unwired(
            &domain,
            weights.view(),
            &[0.0, 0.0],
            config,
        )
        .expect_err("generator cap must reject");
        assert!(matches!(
            error,
            ConstrainedZonotopeTailLpError::LimitExceeded {
                resource: "generator nonzeros",
                required: 1,
                limit: 0,
            }
        ));
    }

    #[test]
    fn zero_deadline_is_rejected_fail_closed() {
        let domain = one_alpha_domain(false);
        let weights = array![[1.0], [0.0]];
        let mut config = test_config();
        config.wall_time = Duration::ZERO;
        let error = diagnose_constrained_zonotope_relu_affine_tail_with_ay_lp_unwired(
            &domain,
            weights.view(),
            &[0.0, 0.0],
            config,
        )
        .expect_err("zero deadline must reject");
        assert!(matches!(
            error,
            ConstrainedZonotopeTailLpError::InvalidConfig {
                field: "wall_time",
                ..
            }
        ));
    }

    #[test]
    fn gradual_underflow_probe_qualifies_adjacent_float_arithmetic() {
        require_gradual_underflow().expect("test host must preserve binary64 subnormals");
        let min_subnormal = f64::from_bits(1);
        assert_eq!(
            add_nonnegative_upward(min_subnormal, min_subnormal, 0)
                .expect("outward subnormal sum")
                .to_bits(),
            3
        );
    }

    #[test]
    fn detached_worker_enforces_outer_deadline() {
        let started = Instant::now();
        let result = run_with_hard_deadline(
            started + Duration::from_millis(20),
            "test-tail-timeout",
            || {
                std::thread::sleep(Duration::from_millis(500));
                7_usize
            },
        )
        .expect("worker launch");
        assert_eq!(result, None);
        assert!(
            started.elapsed() < Duration::from_millis(400),
            "outer deadline waited for the detached worker"
        );
    }

    proptest! {
        #[test]
        fn outward_radius_sum_dominates_exact_dyadic_sum(
            terms in prop::collection::vec(0_u16..=10_000, 1..64)
        ) {
            let mut outward = 0.0_f64;
            let mut exact = BigRational::zero();
            for term in terms {
                let value = f64::from(term) / 16.0;
                outward = add_nonnegative_upward(outward, value, 0).expect("finite sum");
                exact += BigRational::from_float(value).expect("finite dyadic");
            }
            prop_assert!(BigRational::from_float(outward).expect("finite outward") >= exact);
        }

        #[test]
        fn corrected_relu_hull_line_is_outward_over_the_whole_interval(
            lower_magnitude in 1_u16..=4_000,
            upper_magnitude in 1_u16..=4_000,
            numerator in 0_u16..=1_000,
        ) {
            let lower = -(f64::from(lower_magnitude) / 32.0);
            let upper = f64::from(upper_magnitude) / 32.0;
            let (slope, intercept) = outward_relu_upper_line(lower, upper, 0)
                .expect("finite hull line");
            let lower_r = BigRational::from_float(lower).expect("lower dyadic");
            let upper_r = BigRational::from_float(upper).expect("upper dyadic");
            let slope_r = BigRational::from_float(slope).expect("slope dyadic");
            let intercept_r = BigRational::from_float(intercept).expect("intercept dyadic");
            let t = BigRational::new(i64::from(numerator).into(), 1_000_i64.into());
            let x = &lower_r + (&upper_r - &lower_r) * t;
            let relu_x = if x.is_negative() { BigRational::zero() } else { x.clone() };
            let encoded_upper = slope_r * x + intercept_r;
            prop_assert!(encoded_upper >= relu_x);
        }
    }
}
