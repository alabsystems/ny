// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Replay-certified coordinate stabilization for constrained zonotopes.
//!
//! A constrained-zonotope ReLU transformer needs bounds for individual
//! preactivation coordinates.  The ordinary axis hull ignores
//! `C alpha <= d`, so it cannot discover phases implied by accumulated ReLU
//! predicates.  This module builds a bounded batch of unit directions, asks
//! the untrusted batched-Adam lane for multiplier candidates, and exposes only
//! the bounds independently replayed by the outward CPU dual evaluator.
//!
//! Coordinate selection and the GEMM engine have no proof authority.  They may
//! affect which phases are discovered, never whether a reported phase is
//! valid.  A downstream ReLU transformer must additionally establish that the
//! supplied constrained zonotope contains every concrete preactivation
//! witness.  This module is deliberately unwired and cannot emit a verifier
//! verdict.

use ndarray::Array2;
use ny_core::GemmEngine;

use crate::{
    propose_batched_adam_unwired, BatchedAdamConfig, BatchedAdamPlan, BatchedAdamProposerError,
    BatchedAdamStatus, ConstrainedZonotope64, CoordinateDualProposal,
    BATCHED_ADAM_HARD_MAX_DIRECTIONS, BATCHED_ADAM_HARD_MAX_DIRECTION_ELEMENTS,
    BATCHED_ADAM_HARD_MAX_VALUE_DIM,
};

/// Absolute ceiling on selected coordinate axes.
pub const AXIS_STABILIZER_HARD_MAX_AXES: usize = BATCHED_ADAM_HARD_MAX_DIRECTIONS;
/// Absolute ceiling on the constrained-zonotope value dimension.
pub const AXIS_STABILIZER_HARD_MAX_VALUE_DIM: usize = BATCHED_ADAM_HARD_MAX_VALUE_DIM;
/// Absolute ceiling on the dense unit-direction matrix.
pub const AXIS_STABILIZER_HARD_MAX_DIRECTION_ELEMENTS: usize =
    BATCHED_ADAM_HARD_MAX_DIRECTION_ELEMENTS;

/// Caller-tightenable limits for allocations owned by the unit-axis adapter.
///
/// Candidate-search limits remain independently controlled by
/// [`BatchedAdamConfig`].  There is intentionally no `Default`: an experiment
/// must choose both adapter and search budgets explicitly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AxisStabilizerLimits {
    /// Maximum selected coordinates.
    pub max_axes: usize,
    /// Maximum flattened domain value dimension.
    pub max_value_dim: usize,
    /// Maximum `axes * value_dim` binary64 direction elements.
    pub max_direction_elements: usize,
}

/// Checked allocation shape for one axis-stabilization batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AxisStabilizerPlan {
    /// Number of selected coordinate axes.
    pub axes: usize,
    /// Flattened constrained-zonotope value dimension.
    pub value_dim: usize,
    /// Elements in the dense one-hot direction matrix.
    pub direction_elements: usize,
}

/// Certified phase implied by one independently replayed coordinate interval.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CertifiedAxisPhase {
    /// The certified upper endpoint is nonpositive.
    Inactive,
    /// The certified lower endpoint is nonnegative.
    Active,
    /// The certified interval straddles zero.
    Unstable,
    /// Independently certified endpoints cross, proving the abstract
    /// constraint set has no witness in this direction.
    Infeasible,
}

/// One selected axis and its replay-certified dual proposal.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedAxisProposal {
    /// Coordinate in the flattened constrained-zonotope value vector.
    pub coordinate: usize,
    /// Accepted lower/upper bounds and their replayed multipliers.
    pub dual: CoordinateDualProposal,
    /// Phase derived only from `dual.bounds`.
    pub phase: CertifiedAxisPhase,
}

/// Replay-certified result for the selected coordinate batch.
#[derive(Clone, Debug, PartialEq)]
pub struct AxisStabilizerProposal {
    /// Checked unit-direction allocation plan.
    pub axis_plan: AxisStabilizerPlan,
    /// Candidate-search plan, absent when its explicit caps rejected search.
    pub adam_plan: Option<BatchedAdamPlan>,
    /// Why candidate search did or did not complete.
    pub status: BatchedAdamStatus,
    /// Fully completed projected-Adam updates.
    pub iterations_completed: usize,
    /// Calls made through the soundness-free GEMM seam.
    pub engine_calls: usize,
    /// Results in exactly the caller-supplied coordinate order.
    pub axes: Vec<CertifiedAxisProposal>,
}

/// Malformed selection, exhausted adapter resources, or mandatory replay
/// failure.
#[derive(Debug, thiserror::Error)]
pub enum AxisStabilizerError {
    /// A caller limit attempted to exceed an absolute implementation ceiling.
    #[error("invalid {resource} limit {supplied}; hard maximum is {hard_max}")]
    InvalidLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Caller-supplied limit.
        supplied: usize,
        /// Absolute hard maximum.
        hard_max: usize,
    },
    /// A valid caller-selected limit was exhausted.
    #[error("{resource} requires {required}, exceeding limit {limit}")]
    LimitExceeded {
        /// Bounded resource.
        resource: &'static str,
        /// Checked requirement.
        required: usize,
        /// Caller-selected limit.
        limit: usize,
    },
    /// A requested flattened coordinate is outside the domain.
    #[error("coordinate {coordinate} is outside value dimension {value_dim}")]
    CoordinateOutOfRange {
        /// Invalid coordinate.
        coordinate: usize,
        /// Domain value dimension.
        value_dim: usize,
    },
    /// Repeating an axis would spend bounded work without adding evidence.
    #[error("coordinate {coordinate} was selected more than once")]
    DuplicateCoordinate {
        /// Repeated coordinate.
        coordinate: usize,
    },
    /// A checked shape calculation overflowed `usize`.
    #[error("resource size overflow while computing {resource}")]
    ResourceOverflow {
        /// Calculation that overflowed.
        resource: &'static str,
    },
    /// The bounded unit-direction allocation failed.
    #[error("unable to allocate the bounded unit-direction matrix")]
    AllocationFailure,
    /// The checked shape could not be represented by `ndarray`.
    #[error("unable to construct the checked unit-direction matrix shape")]
    DirectionShape,
    /// The underlying proposer returned a malformed result count.
    #[error("axis proposer returned {actual} results for {expected} coordinates")]
    ProposalLength {
        /// Number of selected coordinates.
        expected: usize,
        /// Number of returned proposals.
        actual: usize,
    },
    /// Mandatory zero-baseline construction or outward replay failed.
    #[error(transparent)]
    Proposer(#[from] BatchedAdamProposerError),
}

/// Optimize and independently replay dual bounds for selected coordinate axes.
///
/// The coordinate order is preserved.  Duplicate or out-of-range coordinates
/// fail before this adapter allocates.  The adapter-owned dense direction
/// matrix and returned axis vector are bounded by both caller limits and fixed
/// hard ceilings before the untrusted engine can run.
///
/// [`propose_batched_adam_unwired`] separately hard-bounds its mandatory
/// zero-multiplier baseline and caller-bounds its retained candidate-search
/// buffers; engine-private packing is outside those memory caps.  Its candidate
/// wall-time clock starts after baseline construction and stops before final
/// outward CPU replay, while deadline checks cannot preempt an in-flight engine
/// call.  A production caller must therefore enforce an outer hard deadline
/// around this entire unwired experiment (and isolate an engine that cannot
/// itself guarantee return).
///
/// # Errors
///
/// Returns [`AxisStabilizerError`] for malformed selection/limits, checked
/// resource exhaustion, allocation failure, or failure of the mandatory
/// outward zero-multiplier authority path.  Candidate-only search failures are
/// returned as a successful proposal with baseline bounds and a non-completed
/// [`BatchedAdamStatus`].
pub fn propose_axis_stabilization_unwired(
    domain: &ConstrainedZonotope64,
    coordinates: &[usize],
    limits: AxisStabilizerLimits,
    adam: BatchedAdamConfig,
    engine: &dyn GemmEngine,
) -> Result<AxisStabilizerProposal, AxisStabilizerError> {
    validate_limits(limits)?;
    let value_dim = domain.value_dim();
    check_limit(
        "selected coordinate axes",
        coordinates.len(),
        limits.max_axes,
    )?;
    check_limit("value dimension", value_dim, limits.max_value_dim)?;

    for (index, &coordinate) in coordinates.iter().enumerate() {
        if coordinate >= value_dim {
            return Err(AxisStabilizerError::CoordinateOutOfRange {
                coordinate,
                value_dim,
            });
        }
        if coordinates[..index].contains(&coordinate) {
            return Err(AxisStabilizerError::DuplicateCoordinate { coordinate });
        }
    }

    let direction_elements =
        coordinates
            .len()
            .checked_mul(value_dim)
            .ok_or(AxisStabilizerError::ResourceOverflow {
                resource: "unit-direction elements",
            })?;
    check_limit(
        "unit-direction elements",
        direction_elements,
        limits.max_direction_elements,
    )?;
    let axis_plan = AxisStabilizerPlan {
        axes: coordinates.len(),
        value_dim,
        direction_elements,
    };

    let mut direction_values = Vec::new();
    direction_values
        .try_reserve_exact(direction_elements)
        .map_err(|_| AxisStabilizerError::AllocationFailure)?;
    direction_values.resize(direction_elements, 0.0_f64);
    for (row, &coordinate) in coordinates.iter().enumerate() {
        let row_offset =
            row.checked_mul(value_dim)
                .ok_or(AxisStabilizerError::ResourceOverflow {
                    resource: "unit-direction row offset",
                })?;
        let index =
            row_offset
                .checked_add(coordinate)
                .ok_or(AxisStabilizerError::ResourceOverflow {
                    resource: "unit-direction coordinate offset",
                })?;
        direction_values[index] = 1.0;
    }
    let directions = Array2::from_shape_vec((coordinates.len(), value_dim), direction_values)
        .map_err(|_| AxisStabilizerError::DirectionShape)?;

    let proposal = propose_batched_adam_unwired(domain, directions.view(), adam, engine)?;
    if proposal.proposals.len() != coordinates.len() {
        return Err(AxisStabilizerError::ProposalLength {
            expected: coordinates.len(),
            actual: proposal.proposals.len(),
        });
    }
    let mut axes = Vec::new();
    axes.try_reserve_exact(coordinates.len())
        .map_err(|_| AxisStabilizerError::AllocationFailure)?;
    for (&coordinate, dual) in coordinates.iter().zip(proposal.proposals) {
        let phase = classify_bounds(dual.bounds.lower, dual.bounds.upper);
        axes.push(CertifiedAxisProposal {
            coordinate,
            dual,
            phase,
        });
    }

    Ok(AxisStabilizerProposal {
        axis_plan,
        adam_plan: proposal.plan,
        status: proposal.status,
        iterations_completed: proposal.iterations_completed,
        engine_calls: proposal.engine_calls,
        axes,
    })
}

fn classify_bounds(lower: f64, upper: f64) -> CertifiedAxisPhase {
    if lower > upper {
        CertifiedAxisPhase::Infeasible
    } else if upper <= 0.0 {
        // Keep the base ReLU transformer's upper-first convention at the
        // singleton zero interval.
        CertifiedAxisPhase::Inactive
    } else if lower >= 0.0 {
        CertifiedAxisPhase::Active
    } else {
        CertifiedAxisPhase::Unstable
    }
}

fn validate_limits(limits: AxisStabilizerLimits) -> Result<(), AxisStabilizerError> {
    for (resource, supplied, hard_max) in [
        (
            "selected coordinate axes",
            limits.max_axes,
            AXIS_STABILIZER_HARD_MAX_AXES,
        ),
        (
            "value dimension",
            limits.max_value_dim,
            AXIS_STABILIZER_HARD_MAX_VALUE_DIM,
        ),
        (
            "unit-direction elements",
            limits.max_direction_elements,
            AXIS_STABILIZER_HARD_MAX_DIRECTION_ELEMENTS,
        ),
    ] {
        if supplied > hard_max {
            return Err(AxisStabilizerError::InvalidLimit {
                resource,
                supplied,
                hard_max,
            });
        }
    }
    Ok(())
}

fn check_limit(
    resource: &'static str,
    required: usize,
    limit: usize,
) -> Result<(), AxisStabilizerError> {
    if required > limit {
        Err(AxisStabilizerError::LimitExceeded {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use ndarray::array;
    use ny_core::{GemmEngine, NyError, Result as NyResult};

    use super::*;
    use crate::{BatchedAdamLimits, BatchedAdamStatus};

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
            Err(NyError::InternalError(
                "injected axis search failure".into(),
            ))
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
            panic!("axis adapter should reject before search")
        }
    }

    fn limits() -> AxisStabilizerLimits {
        AxisStabilizerLimits {
            max_axes: 8,
            max_value_dim: 32,
            max_direction_elements: 256,
        }
    }

    fn adam() -> BatchedAdamConfig {
        BatchedAdamConfig {
            iterations: 150,
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
            wall_time: Duration::from_secs(5),
            limits: BatchedAdamLimits {
                max_directions: 8,
                max_iterations: 200,
                max_value_dim: 32,
                max_constraints: 32,
                max_alpha_dim: 32,
                max_constraint_elements: 1_024,
                max_generator_nonzeros: 1_024,
                max_direction_elements: 256,
                max_projection_products: 16_384,
                max_multiplier_elements: 1_024,
                max_working_f32_elements: 100_000,
                max_gemm_products: 100_000_000,
                max_wall_time: Duration::from_secs(10),
            },
        }
    }

    fn two_phase_domain() -> ConstrainedZonotope64 {
        // alpha_0 <= -0.5 makes x_0 inactive; -alpha_1 <= -0.5 makes
        // x_1 active.  The zero-multiplier hull is [-1, 1] for both.
        ConstrainedZonotope64::try_new(
            vec![0.0, 0.0],
            vec![vec![(0, 1.0)], vec![(1, 1.0)]],
            array![[1.0, 0.0], [0.0, -1.0]],
            vec![-0.5, -0.5],
            vec![0.0, 0.0],
        )
        .unwrap()
    }

    #[test]
    fn replayed_axis_duals_discover_both_relu_phases() {
        let result = propose_axis_stabilization_unwired(
            &two_phase_domain(),
            &[1, 0],
            limits(),
            adam(),
            &CpuEngine,
        )
        .unwrap();
        assert_eq!(result.status, BatchedAdamStatus::Completed);
        assert_eq!(result.axis_plan.direction_elements, 4);
        assert_eq!(result.axes[0].coordinate, 1);
        assert_eq!(result.axes[0].phase, CertifiedAxisPhase::Active);
        assert!(result.axes[0].dual.bounds.lower > 0.0);
        assert_eq!(result.axes[1].coordinate, 0);
        assert_eq!(result.axes[1].phase, CertifiedAxisPhase::Inactive);
        assert!(result.axes[1].dual.bounds.upper < 0.0);
        assert!(result
            .axes
            .iter()
            .all(|axis| { axis.dual.lower_improved || axis.dual.upper_improved }));
    }

    #[test]
    fn engine_failure_retains_unstable_zero_multiplier_axes() {
        let result = propose_axis_stabilization_unwired(
            &two_phase_domain(),
            &[0, 1],
            limits(),
            adam(),
            &FailingEngine,
        )
        .unwrap();
        assert_eq!(result.status, BatchedAdamStatus::EngineError);
        assert_eq!(result.engine_calls, 1);
        for axis in result.axes {
            assert_eq!(axis.phase, CertifiedAxisPhase::Unstable);
            assert_eq!(axis.dual.bounds.lower, -1.0);
            assert_eq!(axis.dual.bounds.upper, 1.0);
            assert!(!axis.dual.lower_improved);
            assert!(!axis.dual.upper_improved);
            assert!(axis
                .dual
                .lower_multipliers
                .iter()
                .all(|&value| value == 0.0));
            assert!(axis
                .dual
                .upper_multipliers
                .iter()
                .all(|&value| value == 0.0));
        }
    }

    #[test]
    fn malformed_axes_fail_before_engine() {
        let domain = two_phase_domain();
        assert!(matches!(
            propose_axis_stabilization_unwired(&domain, &[0, 0], limits(), adam(), &PanicEngine,),
            Err(AxisStabilizerError::DuplicateCoordinate { coordinate: 0 })
        ));
        assert!(matches!(
            propose_axis_stabilization_unwired(&domain, &[2], limits(), adam(), &PanicEngine,),
            Err(AxisStabilizerError::CoordinateOutOfRange {
                coordinate: 2,
                value_dim: 2
            })
        ));
    }

    #[test]
    fn every_caller_and_hard_adapter_limit_fails_before_engine() {
        let domain = two_phase_domain();

        let mut caller_axes = limits();
        caller_axes.max_axes = 1;
        assert!(matches!(
            propose_axis_stabilization_unwired(&domain, &[0, 1], caller_axes, adam(), &PanicEngine,),
            Err(AxisStabilizerError::LimitExceeded {
                resource: "selected coordinate axes",
                required: 2,
                limit: 1
            })
        ));

        let mut caller_value_dim = limits();
        caller_value_dim.max_value_dim = 1;
        assert!(matches!(
            propose_axis_stabilization_unwired(
                &domain,
                &[0],
                caller_value_dim,
                adam(),
                &PanicEngine,
            ),
            Err(AxisStabilizerError::LimitExceeded {
                resource: "value dimension",
                required: 2,
                limit: 1
            })
        ));

        let mut caller_direction_elements = limits();
        caller_direction_elements.max_direction_elements = 3;
        assert!(matches!(
            propose_axis_stabilization_unwired(
                &domain,
                &[0, 1],
                caller_direction_elements,
                adam(),
                &PanicEngine,
            ),
            Err(AxisStabilizerError::LimitExceeded {
                resource: "unit-direction elements",
                required: 4,
                limit: 3
            })
        ));

        let mut hard_axes = limits();
        hard_axes.max_axes = AXIS_STABILIZER_HARD_MAX_AXES + 1;
        assert!(matches!(
            propose_axis_stabilization_unwired(&domain, &[0], hard_axes, adam(), &PanicEngine,),
            Err(AxisStabilizerError::InvalidLimit {
                resource: "selected coordinate axes",
                supplied,
                hard_max: AXIS_STABILIZER_HARD_MAX_AXES,
            }) if supplied == AXIS_STABILIZER_HARD_MAX_AXES + 1
        ));

        let mut hard_value_dim = limits();
        hard_value_dim.max_value_dim = AXIS_STABILIZER_HARD_MAX_VALUE_DIM + 1;
        assert!(matches!(
            propose_axis_stabilization_unwired(&domain, &[0], hard_value_dim, adam(), &PanicEngine,),
            Err(AxisStabilizerError::InvalidLimit {
                resource: "value dimension",
                supplied,
                hard_max: AXIS_STABILIZER_HARD_MAX_VALUE_DIM,
            }) if supplied == AXIS_STABILIZER_HARD_MAX_VALUE_DIM + 1
        ));

        let mut hard_direction_elements = limits();
        hard_direction_elements.max_direction_elements =
            AXIS_STABILIZER_HARD_MAX_DIRECTION_ELEMENTS + 1;
        assert!(matches!(
            propose_axis_stabilization_unwired(
                &domain,
                &[0],
                hard_direction_elements,
                adam(),
                &PanicEngine,
            ),
            Err(AxisStabilizerError::InvalidLimit {
                resource: "unit-direction elements",
                supplied,
                hard_max: AXIS_STABILIZER_HARD_MAX_DIRECTION_ELEMENTS,
            }) if supplied == AXIS_STABILIZER_HARD_MAX_DIRECTION_ELEMENTS + 1
        ));
    }

    #[test]
    fn empty_axis_batch_is_a_bounded_noop() {
        let result = propose_axis_stabilization_unwired(
            &two_phase_domain(),
            &[],
            limits(),
            adam(),
            &PanicEngine,
        )
        .unwrap();
        assert_eq!(result.status, BatchedAdamStatus::NoSearchNeeded);
        assert_eq!(result.axis_plan.direction_elements, 0);
        assert!(result.axes.is_empty());
        assert_eq!(result.engine_calls, 0);
    }

    #[test]
    fn phase_classifier_covers_zero_and_crossed_endpoint_branches() {
        assert_eq!(classify_bounds(-0.0, 0.0), CertifiedAxisPhase::Inactive);
        assert_eq!(classify_bounds(0.0, 1.0), CertifiedAxisPhase::Active);
        assert_eq!(classify_bounds(-1.0, 1.0), CertifiedAxisPhase::Unstable);
        assert_eq!(classify_bounds(1.0, -1.0), CertifiedAxisPhase::Infeasible);
    }
}
