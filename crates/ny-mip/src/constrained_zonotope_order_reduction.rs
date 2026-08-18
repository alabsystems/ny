// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proof-safe, call-budgeted order reduction for constrained zonotopes.
//!
//! Let the input be
//!
//! ```text
//! x = c + G_K alpha_K + G_D alpha_D + e,
//! alpha in [-1, 1]^m, C_K alpha_K + C_D alpha_D <= d, |e_i| <= r_i.
//! ```
//!
//! Dropping the columns in `D` is sound when their value contribution is
//! absorbed into the independent box remainder and their predicate
//! contribution is projected outward:
//!
//! ```text
//! r'_i = r_i + sum(j in D) |G_ij|
//! d'_q = d_q + sum(j in D) |C_qj|.
//! ```
//!
//! Every original witness maps to `alpha_K` plus
//! `e' = e + G_D alpha_D`, and satisfies `C_K alpha_K <= d'` by the triangle
//! inequality. All additions are rounded toward positive infinity, so the
//! stored binary64 domain encloses the exact-dyadic projection.
//!
//! A caller-protected prefix is always retained. Remaining columns are ranked
//! deterministically by their largest absolute coefficient in either `G` or
//! `C`; ties retain the lower original index. This preserves the five latent
//! CGAN symbols while preferentially keeping later correlations that still
//! have material geometric or predicate leverage.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::mem::size_of;

use ndarray::Array2;

use crate::constrained_zonotope64::{
    ConstrainedZonotope64, ConstrainedZonotope64CallGateError, ConstrainedZonotope64Error,
};
use crate::constrained_zonotope_call_budget::{
    ConstrainedZonotopeCallBudget, ConstrainedZonotopeCallBudgetError, ConstrainedZonotopeCallGate,
    ConstrainedZonotopeCallOutcome, ConstrainedZonotopeCallTracker,
    ConstrainedZonotopePeakLiveBytes,
};

/// Absolute ceiling on the value dimension accepted by order reduction.
pub const ORDER_REDUCTION_HARD_MAX_VALUE_DIM: usize = 1_048_576;
/// Absolute ceiling on input alpha symbols.
pub const ORDER_REDUCTION_HARD_MAX_INPUT_ALPHA_DIM: usize = 65_536;
/// Absolute ceiling on retained alpha symbols.
pub const ORDER_REDUCTION_HARD_MAX_OUTPUT_ALPHA_DIM: usize = 65_536;
/// Absolute ceiling on predicate rows.
pub const ORDER_REDUCTION_HARD_MAX_CONSTRAINTS: usize = 65_536;
/// Absolute ceiling on the dense input predicate matrix.
pub const ORDER_REDUCTION_HARD_MAX_CONSTRAINT_ELEMENTS: usize = 33_554_432;
/// Absolute ceiling on sparse input generator coefficients.
pub const ORDER_REDUCTION_HARD_MAX_GENERATOR_NNZ: usize = 16_777_216;

/// Caller-tightenable structural limits for one order-reduction call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeOrderReductionLimits {
    /// Maximum value dimension.
    pub max_value_dim: usize,
    /// Maximum input alpha dimension.
    pub max_input_alpha_dim: usize,
    /// Maximum retained alpha dimension.
    pub max_output_alpha_dim: usize,
    /// Maximum predicate row count.
    pub max_constraints: usize,
    /// Maximum elements in the dense input predicate matrix.
    pub max_constraint_elements: usize,
    /// Maximum sparse input generator coefficients.
    pub max_generator_nnz: usize,
}

impl Default for ConstrainedZonotopeOrderReductionLimits {
    fn default() -> Self {
        Self {
            max_value_dim: ORDER_REDUCTION_HARD_MAX_VALUE_DIM,
            max_input_alpha_dim: ORDER_REDUCTION_HARD_MAX_INPUT_ALPHA_DIM,
            max_output_alpha_dim: ORDER_REDUCTION_HARD_MAX_OUTPUT_ALPHA_DIM,
            max_constraints: ORDER_REDUCTION_HARD_MAX_CONSTRAINTS,
            max_constraint_elements: ORDER_REDUCTION_HARD_MAX_CONSTRAINT_ELEMENTS,
            max_generator_nnz: ORDER_REDUCTION_HARD_MAX_GENERATOR_NNZ,
        }
    }
}

/// Deterministic accounting for a completed order reduction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeOrderReductionPlan {
    input_alpha_dim: usize,
    output_alpha_dim: usize,
    protected_prefix: usize,
    input_generator_nnz: usize,
    output_generator_nnz: usize,
    discarded_generator_nnz: usize,
    input_constraint_elements: usize,
    output_constraint_elements: usize,
}

impl ConstrainedZonotopeOrderReductionPlan {
    /// Alpha symbols before reduction.
    #[must_use]
    pub const fn input_alpha_dim(self) -> usize {
        self.input_alpha_dim
    }

    /// Alpha symbols retained after reduction.
    #[must_use]
    pub const fn output_alpha_dim(self) -> usize {
        self.output_alpha_dim
    }

    /// Leading symbols retained independently of magnitude.
    #[must_use]
    pub const fn protected_prefix(self) -> usize {
        self.protected_prefix
    }

    /// Sparse generator coefficients before reduction.
    #[must_use]
    pub const fn input_generator_nnz(self) -> usize {
        self.input_generator_nnz
    }

    /// Sparse generator coefficients retained after reduction.
    #[must_use]
    pub const fn output_generator_nnz(self) -> usize {
        self.output_generator_nnz
    }

    /// Sparse generator coefficients absorbed into the box remainder.
    #[must_use]
    pub const fn discarded_generator_nnz(self) -> usize {
        self.discarded_generator_nnz
    }

    /// Dense predicate elements before reduction.
    #[must_use]
    pub const fn input_constraint_elements(self) -> usize {
        self.input_constraint_elements
    }

    /// Dense predicate elements after projection.
    #[must_use]
    pub const fn output_constraint_elements(self) -> usize {
        self.output_constraint_elements
    }
}

/// Structural or arithmetic refusal from order reduction.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ConstrainedZonotopeOrderReductionError {
    /// A caller limit exceeds the implementation ceiling.
    #[error("invalid {resource} limit {supplied}; hard maximum is {hard_max}")]
    InvalidLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Caller-supplied limit.
        supplied: usize,
        /// Absolute implementation ceiling.
        hard_max: usize,
    },

    /// The requested protected prefix cannot fit in the retained dimension.
    #[error(
        "protected alpha prefix {protected_prefix} exceeds retained alpha dimension {output_alpha_dim}"
    )]
    InvalidProtectedPrefix {
        /// Requested protected prefix.
        protected_prefix: usize,
        /// Effective retained dimension.
        output_alpha_dim: usize,
    },

    /// The input or planned output exceeds a valid caller limit.
    #[error("{resource} requires {required}, exceeding limit {limit}")]
    LimitExceeded {
        /// Bounded resource.
        resource: &'static str,
        /// Conservatively required amount.
        required: usize,
        /// Effective caller limit.
        limit: usize,
    },

    /// A resource calculation overflowed `usize`.
    #[error("resource size overflow while computing {operation}")]
    ResourceOverflow {
        /// Failed calculation.
        operation: &'static str,
    },

    /// A bounded allocation request failed.
    #[error("unable to reserve storage for {resource}")]
    AllocationFailure {
        /// Requested container.
        resource: &'static str,
    },

    /// A projected outward sum has no finite binary64 enclosure.
    #[error("non-finite arithmetic while computing {operation}")]
    NonFiniteArithmetic {
        /// Failed projection operation.
        operation: &'static str,
    },

    /// The host flushes binary64 subnormals and cannot support this proof path.
    #[error("unsupported floating-point environment: {requirement}")]
    UnsupportedFloatingPoint {
        /// Required IEEE behavior.
        requirement: &'static str,
    },

    /// Final domain validation failed closed.
    #[error(transparent)]
    Domain(#[from] ConstrainedZonotope64Error),
}

/// A transform refusal or call-local execution-firewall refusal.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ConstrainedZonotopeOrderReductionBudgetError {
    /// Structural or numeric order-reduction failure.
    #[error(transparent)]
    Transform(#[from] ConstrainedZonotopeOrderReductionError),
    /// Deadline or peak-live-byte failure.
    #[error(transparent)]
    Budget(#[from] ConstrainedZonotopeCallBudgetError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RankedGenerator {
    magnitude_bits: u64,
    index: usize,
}

impl Ord for RankedGenerator {
    fn cmp(&self, other: &Self) -> Ordering {
        self.magnitude_bits
            .cmp(&other.magnitude_bits)
            // For equal magnitudes, the lower original index is stronger.
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for RankedGenerator {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug)]
struct Geometry {
    value_dim: usize,
    input_alpha_dim: usize,
    output_alpha_dim: usize,
    constraint_count: usize,
    input_constraint_elements: usize,
    output_constraint_elements: usize,
    input_generator_nnz: usize,
}

/// Retain a bounded number of constrained-zonotope generator columns.
///
/// `protected_prefix` leading columns are kept unconditionally. The remaining
/// capacity is filled by deterministic coefficient magnitude. The supplied
/// budget is absolute and call-local; callers include retained input storage
/// in `baseline_live_bytes`.
pub fn constrained_zonotope_order_reduce_unwired_with_budget(
    input: &ConstrainedZonotope64,
    output_alpha_dim: usize,
    protected_prefix: usize,
    limits: ConstrainedZonotopeOrderReductionLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> Result<
    ConstrainedZonotopeCallOutcome<(ConstrainedZonotope64, ConstrainedZonotopeOrderReductionPlan)>,
    ConstrainedZonotopeOrderReductionBudgetError,
> {
    let mut tracker = ConstrainedZonotopeCallTracker::from_system_clock(budget)?;
    reduce_with_gate(
        input,
        output_alpha_dim,
        protected_prefix,
        limits,
        &mut tracker,
    )
}

fn reduce_with_gate<G>(
    input: &ConstrainedZonotope64,
    requested_output_alpha_dim: usize,
    protected_prefix: usize,
    limits: ConstrainedZonotopeOrderReductionLimits,
    gate: &mut G,
) -> Result<
    ConstrainedZonotopeCallOutcome<(ConstrainedZonotope64, ConstrainedZonotopeOrderReductionPlan)>,
    ConstrainedZonotopeOrderReductionBudgetError,
>
where
    G: ConstrainedZonotopeCallGate,
{
    validate_limits(limits)?;
    require_gradual_underflow()?;
    gate.checkpoint("order reduction floating-point preflight")?;
    gate.checkpoint("order reduction geometry")?;
    let geometry = plan_geometry(input, requested_output_alpha_dim, limits, gate)?;
    if protected_prefix > geometry.output_alpha_dim {
        return Err(
            ConstrainedZonotopeOrderReductionError::InvalidProtectedPrefix {
                protected_prefix,
                output_alpha_dim: geometry.output_alpha_dim,
            }
            .into(),
        );
    }

    let transform_owned_bytes = peak_live_bytes(geometry)?;
    gate.preflight_peak_live_bytes(transform_owned_bytes)?;
    gate.checkpoint("order reduction peak-memory preflight complete")?;

    let keep = select_generators(input, geometry, protected_prefix, gate)?;
    gate.checkpoint("order reduction selection complete")?;

    let mut center = reserved_vec_with_gate(
        geometry.value_dim,
        "order-reduction center",
        "order reduction center allocation",
        gate,
    )?;
    for chunk in input
        .center()
        .chunks(crate::CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL)
    {
        gate.charge_items(chunk.len(), "order reduction center clone")?;
        center.extend_from_slice(chunk);
    }

    let mut box_remainder = reserved_vec_with_gate(
        geometry.value_dim,
        "order-reduction box remainder",
        "order reduction box-remainder allocation",
        gate,
    )?;
    for chunk in input
        .box_remainder()
        .chunks(crate::CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL)
    {
        gate.charge_items(chunk.len(), "order reduction box-remainder clone")?;
        box_remainder.extend_from_slice(chunk);
    }

    let mut sparse_generators = reserved_vec_with_gate(
        geometry.output_alpha_dim,
        "order-reduction generator columns",
        "order reduction generator-column allocation",
        gate,
    )?;
    let mut output_generator_nnz = 0_usize;
    let mut discarded_generator_nnz = 0_usize;
    for (generator_index, generator) in input.generators().iter().enumerate() {
        gate.charge_items(1, "order reduction generator projection")?;
        if keep[generator_index] {
            let mut entries = reserved_vec_with_gate(
                generator.nnz(),
                "order-reduction generator entries",
                "order reduction generator-entry allocation",
                gate,
            )?;
            for (value_index, coefficient) in generator.entries() {
                gate.charge_items(1, "order reduction generator clone")?;
                entries.push((value_index, coefficient));
            }
            output_generator_nnz = output_generator_nnz.checked_add(entries.len()).ok_or(
                ConstrainedZonotopeOrderReductionError::ResourceOverflow {
                    operation: "retained generator nonzeros",
                },
            )?;
            sparse_generators.push(entries);
        } else {
            discarded_generator_nnz = discarded_generator_nnz.checked_add(generator.nnz()).ok_or(
                ConstrainedZonotopeOrderReductionError::ResourceOverflow {
                    operation: "discarded generator nonzeros",
                },
            )?;
            for (value_index, coefficient) in generator.entries() {
                gate.charge_items(1, "order reduction remainder projection")?;
                box_remainder[value_index] = add_upper(
                    box_remainder[value_index],
                    coefficient.abs(),
                    "order-reduction box remainder",
                )?;
            }
        }
    }
    gate.checkpoint("order reduction generator projection complete")?;

    let mut output_constraint_values = reserved_vec_with_gate(
        geometry.output_constraint_elements,
        "order-reduction constraint matrix",
        "order reduction constraint-matrix allocation",
        gate,
    )?;
    let mut rhs = reserved_vec_with_gate(
        geometry.constraint_count,
        "order-reduction right-hand side",
        "order reduction right-hand-side allocation",
        gate,
    )?;
    let constraints = input.constraints();
    for row in 0..geometry.constraint_count {
        gate.charge_items(1, "order reduction constraint projection rows")?;
        let mut projected_rhs = input.rhs()[row];
        for column in 0..geometry.input_alpha_dim {
            gate.charge_items(1, "order reduction constraint projection")?;
            let coefficient = constraints[[row, column]];
            if keep[column] {
                output_constraint_values.push(coefficient);
            } else {
                projected_rhs = add_upper(
                    projected_rhs,
                    coefficient.abs(),
                    "order-reduction projected right-hand side",
                )?;
            }
        }
        rhs.push(projected_rhs);
    }
    let output_constraints = Array2::from_shape_vec(
        (geometry.constraint_count, geometry.output_alpha_dim),
        output_constraint_values,
    )
    .map_err(
        |_| ConstrainedZonotopeOrderReductionError::ResourceOverflow {
            operation: "order-reduction constraint matrix shape",
        },
    )?;

    gate.checkpoint("order reduction domain materialization")?;
    let output = ConstrainedZonotope64::try_new_with_call_gate(
        center,
        sparse_generators,
        output_constraints,
        rhs,
        box_remainder,
        gate,
    )
    .map_err(|error| match error {
        ConstrainedZonotope64CallGateError::Domain(error) => {
            ConstrainedZonotopeOrderReductionBudgetError::Transform(
                ConstrainedZonotopeOrderReductionError::Domain(error),
            )
        }
        ConstrainedZonotope64CallGateError::Budget(error) => {
            ConstrainedZonotopeOrderReductionBudgetError::Budget(error)
        }
    })?;
    gate.checkpoint("order reduction domain materialization complete")?;

    let plan = ConstrainedZonotopeOrderReductionPlan {
        input_alpha_dim: geometry.input_alpha_dim,
        output_alpha_dim: geometry.output_alpha_dim,
        protected_prefix,
        input_generator_nnz: geometry.input_generator_nnz,
        output_generator_nnz,
        discarded_generator_nnz,
        input_constraint_elements: geometry.input_constraint_elements,
        output_constraint_elements: geometry.output_constraint_elements,
    };
    gate.checkpoint("order reduction publication")?;
    Ok(ConstrainedZonotopeCallOutcome::new(
        (output, plan),
        gate.report(),
    ))
}

fn validate_limits(
    limits: ConstrainedZonotopeOrderReductionLimits,
) -> Result<(), ConstrainedZonotopeOrderReductionError> {
    check_hard_limit(
        "value dimension",
        limits.max_value_dim,
        ORDER_REDUCTION_HARD_MAX_VALUE_DIM,
    )?;
    check_hard_limit(
        "input alpha dimension",
        limits.max_input_alpha_dim,
        ORDER_REDUCTION_HARD_MAX_INPUT_ALPHA_DIM,
    )?;
    check_hard_limit(
        "output alpha dimension",
        limits.max_output_alpha_dim,
        ORDER_REDUCTION_HARD_MAX_OUTPUT_ALPHA_DIM,
    )?;
    check_hard_limit(
        "constraint count",
        limits.max_constraints,
        ORDER_REDUCTION_HARD_MAX_CONSTRAINTS,
    )?;
    check_hard_limit(
        "constraint elements",
        limits.max_constraint_elements,
        ORDER_REDUCTION_HARD_MAX_CONSTRAINT_ELEMENTS,
    )?;
    check_hard_limit(
        "generator nonzeros",
        limits.max_generator_nnz,
        ORDER_REDUCTION_HARD_MAX_GENERATOR_NNZ,
    )
}

fn plan_geometry<G>(
    input: &ConstrainedZonotope64,
    requested_output_alpha_dim: usize,
    limits: ConstrainedZonotopeOrderReductionLimits,
    gate: &mut G,
) -> Result<Geometry, ConstrainedZonotopeOrderReductionBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let value_dim = input.value_dim();
    let input_alpha_dim = input.alpha_dim();
    let output_alpha_dim = requested_output_alpha_dim.min(input_alpha_dim);
    let constraint_count = input.constraint_count();
    check_limit("value dimension", value_dim, limits.max_value_dim)?;
    check_limit(
        "input alpha dimension",
        input_alpha_dim,
        limits.max_input_alpha_dim,
    )?;
    check_limit(
        "output alpha dimension",
        output_alpha_dim,
        limits.max_output_alpha_dim,
    )?;
    check_limit("constraint count", constraint_count, limits.max_constraints)?;
    let input_constraint_elements = constraint_count.checked_mul(input_alpha_dim).ok_or(
        ConstrainedZonotopeOrderReductionError::ResourceOverflow {
            operation: "input constraint elements",
        },
    )?;
    check_limit(
        "constraint elements",
        input_constraint_elements,
        limits.max_constraint_elements,
    )?;
    let output_constraint_elements = constraint_count.checked_mul(output_alpha_dim).ok_or(
        ConstrainedZonotopeOrderReductionError::ResourceOverflow {
            operation: "output constraint elements",
        },
    )?;

    let mut input_generator_nnz = 0_usize;
    for generator in input.generators() {
        gate.charge_items(1, "order reduction generator geometry")?;
        input_generator_nnz = input_generator_nnz.checked_add(generator.nnz()).ok_or(
            ConstrainedZonotopeOrderReductionError::ResourceOverflow {
                operation: "input generator nonzeros",
            },
        )?;
    }
    check_limit(
        "generator nonzeros",
        input_generator_nnz,
        limits.max_generator_nnz,
    )?;

    Ok(Geometry {
        value_dim,
        input_alpha_dim,
        output_alpha_dim,
        constraint_count,
        input_constraint_elements,
        output_constraint_elements,
        input_generator_nnz,
    })
}

fn peak_live_bytes(geometry: Geometry) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    let mut peak = ConstrainedZonotopePeakLiveBytes::new();
    peak.add_elements::<bool>(geometry.input_alpha_dim, "order-reduction keep-mask bytes")?;
    peak.add_elements::<Reverse<RankedGenerator>>(
        geometry.output_alpha_dim,
        "order-reduction ranking-heap bytes",
    )?;
    peak.add_elements::<f64>(
        geometry.value_dim.checked_mul(2).ok_or(
            ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "order-reduction value-vector bytes",
            },
        )?,
        "order-reduction value-vector bytes",
    )?;
    peak.add_elements::<f64>(
        geometry.output_constraint_elements,
        "order-reduction constraint bytes",
    )?;
    peak.add_elements::<f64>(
        geometry.constraint_count,
        "order-reduction right-hand-side bytes",
    )?;

    // `try_new` validates into a second sparse-column representation while
    // the staged `(usize, f64)` buffers are still partly live. Count both
    // complete representations, including each Vec header.
    peak.add_elements::<Vec<(usize, f64)>>(
        geometry.output_alpha_dim.checked_mul(2).ok_or(
            ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "order-reduction generator-header bytes",
            },
        )?,
        "order-reduction generator-header bytes",
    )?;
    peak.add_elements::<(usize, f64)>(
        geometry.input_generator_nnz.checked_mul(2).ok_or(
            ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "order-reduction generator-entry bytes",
            },
        )?,
        "order-reduction generator-entry bytes",
    )?;

    // Account for container headers whose backing storage is charged above.
    peak.add_bytes(
        6_usize.checked_mul(size_of::<Vec<usize>>()).ok_or(
            ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "order-reduction container-header bytes",
            },
        )?,
        "order-reduction container-header bytes",
    )?;
    Ok(peak.finish())
}

fn select_generators<G>(
    input: &ConstrainedZonotope64,
    geometry: Geometry,
    protected_prefix: usize,
    gate: &mut G,
) -> Result<Vec<bool>, ConstrainedZonotopeOrderReductionBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let mut keep = reserved_vec_with_gate(
        geometry.input_alpha_dim,
        "order-reduction keep mask",
        "order reduction keep-mask allocation",
        gate,
    )?;
    for _ in 0..geometry.input_alpha_dim {
        gate.charge_items(1, "order reduction keep-mask initialization")?;
        keep.push(false);
    }
    for retained in &mut keep[..protected_prefix] {
        gate.charge_items(1, "order reduction protected-prefix selection")?;
        *retained = true;
    }

    let candidate_capacity = geometry.output_alpha_dim - protected_prefix;
    if candidate_capacity == geometry.input_alpha_dim - protected_prefix {
        for retained in &mut keep[protected_prefix..] {
            gate.charge_items(1, "order reduction identity selection")?;
            *retained = true;
        }
        return Ok(keep);
    }
    if candidate_capacity == 0 {
        return Ok(keep);
    }

    let mut strongest = BinaryHeap::<Reverse<RankedGenerator>>::new();
    gate.checkpoint("order reduction ranking-heap allocation")?;
    strongest
        .try_reserve_exact(candidate_capacity)
        .map_err(
            |_| ConstrainedZonotopeOrderReductionError::AllocationFailure {
                resource: "order-reduction ranking heap",
            },
        )?;
    let constraints = input.constraints();
    let heap_height = usize::try_from(usize::BITS - candidate_capacity.max(1).leading_zeros())
        .expect("usize bit width fits usize");
    // A replacement can perform both one pop and one push. Charge a strict
    // upper bound before entering the uninterruptible std heap operation.
    let heap_work_per_candidate = heap_height * 2 + 3;
    for generator_index in protected_prefix..geometry.input_alpha_dim {
        let mut magnitude_bits = 0_u64;
        for (_, coefficient) in input.generators()[generator_index].entries() {
            gate.charge_items(1, "order reduction generator ranking")?;
            magnitude_bits = magnitude_bits.max(coefficient.abs().to_bits());
        }
        for row in 0..geometry.constraint_count {
            gate.charge_items(1, "order reduction constraint ranking")?;
            magnitude_bits =
                magnitude_bits.max(constraints[[row, generator_index]].abs().to_bits());
        }

        let candidate = RankedGenerator {
            magnitude_bits,
            index: generator_index,
        };
        gate.charge_items(
            heap_work_per_candidate,
            "order reduction ranking-heap update",
        )?;
        if strongest.len() < candidate_capacity {
            strongest.push(Reverse(candidate));
        } else if strongest
            .peek()
            .is_some_and(|weakest| candidate > weakest.0)
        {
            let _ = strongest.pop();
            strongest.push(Reverse(candidate));
        }
    }
    for Reverse(generator) in strongest {
        gate.charge_items(
            heap_work_per_candidate,
            "order reduction ranking-heap drain",
        )?;
        keep[generator.index] = true;
    }
    Ok(keep)
}

fn add_upper(
    left: f64,
    nonnegative_right: f64,
    operation: &'static str,
) -> Result<f64, ConstrainedZonotopeOrderReductionError> {
    debug_assert!(left.is_finite());
    debug_assert!(nonnegative_right.is_finite() && nonnegative_right >= 0.0);
    if nonnegative_right == 0.0 {
        return Ok(left);
    }
    if left == 0.0 {
        return Ok(nonnegative_right);
    }
    let nearest = left + nonnegative_right;
    if !nearest.is_finite() {
        return Err(ConstrainedZonotopeOrderReductionError::NonFiniteArithmetic { operation });
    }
    let outward = nearest.next_up();
    if !outward.is_finite() {
        // `nearest == f64::MAX` can still be an exact or already-upward
        // rounding of a finite exact sum. Knuth's error-free TwoSum residual
        // distinguishes that permissive case from an exact sum above MAX.
        let virtual_right = nearest - left;
        let residual = (left - (nearest - virtual_right)) + (nonnegative_right - virtual_right);
        if residual.is_finite() && residual <= 0.0 {
            return Ok(nearest);
        }
        return Err(ConstrainedZonotopeOrderReductionError::NonFiniteArithmetic { operation });
    }
    Ok(outward)
}

/// Reject FTZ/DAZ before adjacent-float additions are used as proof objects.
fn require_gradual_underflow() -> Result<(), ConstrainedZonotopeOrderReductionError> {
    let half = std::hint::black_box(0.5_f64);
    let min_normal = std::hint::black_box(f64::MIN_POSITIVE);
    let min_subnormal = std::hint::black_box(f64::from_bits(1));
    let two_subnormals = std::hint::black_box(f64::from_bits(2));
    if std::hint::black_box(min_normal * half).to_bits() != 0x0008_0000_0000_0000
        || std::hint::black_box(two_subnormals * half).to_bits() != 1
        || std::hint::black_box(min_subnormal + min_subnormal).to_bits() != 2
    {
        return Err(
            ConstrainedZonotopeOrderReductionError::UnsupportedFloatingPoint {
                requirement: "IEEE-754 binary64 gradual underflow (FTZ/DAZ disabled)",
            },
        );
    }
    Ok(())
}

fn check_hard_limit(
    resource: &'static str,
    supplied: usize,
    hard_max: usize,
) -> Result<(), ConstrainedZonotopeOrderReductionError> {
    if supplied > hard_max {
        return Err(ConstrainedZonotopeOrderReductionError::InvalidLimit {
            resource,
            supplied,
            hard_max,
        });
    }
    Ok(())
}

fn check_limit(
    resource: &'static str,
    required: usize,
    limit: usize,
) -> Result<(), ConstrainedZonotopeOrderReductionError> {
    if required > limit {
        return Err(ConstrainedZonotopeOrderReductionError::LimitExceeded {
            resource,
            required,
            limit,
        });
    }
    Ok(())
}

fn reserved_vec<T>(
    capacity: usize,
    resource: &'static str,
) -> Result<Vec<T>, ConstrainedZonotopeOrderReductionError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| ConstrainedZonotopeOrderReductionError::AllocationFailure { resource })?;
    Ok(values)
}

fn reserved_vec_with_gate<T, G>(
    capacity: usize,
    resource: &'static str,
    allocation_checkpoint: &'static str,
    gate: &mut G,
) -> Result<Vec<T>, ConstrainedZonotopeOrderReductionBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    gate.checkpoint(allocation_checkpoint)?;
    Ok(reserved_vec(capacity, resource)?)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::time::{Duration, Instant};

    use ndarray::array;
    use num_rational::BigRational;
    use proptest::prelude::*;

    use super::*;

    fn limits() -> ConstrainedZonotopeOrderReductionLimits {
        ConstrainedZonotopeOrderReductionLimits {
            max_value_dim: 16,
            max_input_alpha_dim: 16,
            max_output_alpha_dim: 16,
            max_constraints: 16,
            max_constraint_elements: 256,
            max_generator_nnz: 256,
        }
    }

    fn budget() -> ConstrainedZonotopeCallBudget {
        ConstrainedZonotopeCallBudget::new(Instant::now() + Duration::from_mins(1), 0, 1 << 20)
    }

    fn sample_domain() -> ConstrainedZonotope64 {
        ConstrainedZonotope64::try_new(
            vec![0.5, -1.0],
            vec![
                vec![(0, 2.0), (1, 1.0)],
                vec![(0, -3.0), (1, 0.25)],
                vec![(0, 0.5), (1, -4.0)],
            ],
            array![[1.0, -2.0, 0.5], [-0.5, 0.25, -1.0]],
            vec![0.75, 0.1],
            vec![0.125, 0.5],
        )
        .unwrap()
    }

    fn rat(value: f64) -> BigRational {
        BigRational::from_float(value).expect("finite binary64 test value")
    }

    fn finite_from_bits(bits: u64) -> f64 {
        const EXPONENT: u64 = 0x7ff0_0000_0000_0000;
        let finite_bits = if bits & EXPONENT == EXPONENT {
            bits ^ 0x0010_0000_0000_0000
        } else {
            bits
        };
        f64::from_bits(finite_bits)
    }

    fn expect_deadline_at(
        input: &ConstrainedZonotope64,
        output_alpha_dim: usize,
        protected_prefix: usize,
        limits: ConstrainedZonotopeOrderReductionLimits,
        phase: &'static str,
    ) {
        let start = Instant::now();
        let deadline = start + Duration::from_secs(1);
        let mut tracker = ConstrainedZonotopeCallTracker::with_clock(
            ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
            move |checkpoint| {
                if checkpoint == phase {
                    deadline
                } else {
                    start
                }
            },
        )
        .unwrap();
        let result = reduce_with_gate(
            input,
            output_alpha_dim,
            protected_prefix,
            limits,
            &mut tracker,
        );
        assert!(matches!(
            result,
            Err(ConstrainedZonotopeOrderReductionBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
            )) if checkpoint == phase
        ));
    }

    #[test]
    fn protected_prefix_and_strongest_column_are_retained() {
        let outcome = constrained_zonotope_order_reduce_unwired_with_budget(
            &sample_domain(),
            2,
            1,
            limits(),
            budget(),
        )
        .unwrap();
        let (output, plan) = outcome.into_value();
        assert_eq!(plan.input_alpha_dim(), 3);
        assert_eq!(plan.output_alpha_dim(), 2);
        assert_eq!(plan.protected_prefix(), 1);
        assert_eq!(plan.input_generator_nnz(), 6);
        assert_eq!(plan.output_generator_nnz(), 4);
        assert_eq!(plan.discarded_generator_nnz(), 2);
        assert_eq!(
            output.generators()[0].entries().collect::<Vec<_>>(),
            vec![(0, 2.0), (1, 1.0)]
        );
        assert_eq!(
            output.generators()[1].entries().collect::<Vec<_>>(),
            vec![(0, 0.5), (1, -4.0)]
        );
        assert_eq!(output.box_remainder()[0], (0.125_f64 + 3.0).next_up());
        assert_eq!(output.box_remainder()[1], (0.5_f64 + 0.25).next_up());
        assert_eq!(output.constraints(), array![[1.0, 0.5], [-0.5, -1.0]]);
        assert_eq!(output.rhs()[0], (0.75_f64 + 2.0).next_up());
        assert_eq!(output.rhs()[1], (0.1_f64 + 0.25).next_up());
    }

    #[test]
    fn every_original_witness_maps_into_the_projected_domain() {
        let input = sample_domain();
        let output =
            constrained_zonotope_order_reduce_unwired_with_budget(&input, 2, 1, limits(), budget())
                .unwrap()
                .into_value()
                .0;

        for alpha0 in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            for alpha1 in [-1.0, -0.5, 0.0, 0.5, 1.0] {
                for alpha2 in [-1.0, -0.5, 0.0, 0.5, 1.0] {
                    let alpha = [alpha0, alpha1, alpha2];
                    if (0..input.constraint_count()).any(|row| {
                        let lhs = (0..input.alpha_dim())
                            .map(|column| input.constraints()[[row, column]] * alpha[column])
                            .sum::<f64>();
                        lhs > input.rhs()[row]
                    }) {
                        continue;
                    }

                    let beta = [alpha0, alpha2];
                    for row in 0..output.constraint_count() {
                        let lhs = (0..output.alpha_dim())
                            .map(|column| output.constraints()[[row, column]] * beta[column])
                            .sum::<f64>();
                        assert!(lhs <= output.rhs()[row]);
                    }
                    for value_index in 0..input.value_dim() {
                        let discarded = input.generators()[1]
                            .entries()
                            .find_map(|(index, coefficient)| {
                                (index == value_index).then_some(coefficient * alpha1)
                            })
                            .unwrap_or(0.0);
                        for old_error in [
                            -input.box_remainder()[value_index],
                            0.0,
                            input.box_remainder()[value_index],
                        ] {
                            assert!(
                                (discarded + old_error).abs()
                                    <= output.box_remainder()[value_index]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn zero_retained_columns_becomes_a_sound_box() {
        let input = ConstrainedZonotope64::try_new(
            vec![1.0],
            vec![vec![(0, -2.0)], vec![(0, 0.5)]],
            Array2::zeros((0, 2)),
            Vec::new(),
            vec![0.25],
        )
        .unwrap();
        let output =
            constrained_zonotope_order_reduce_unwired_with_budget(&input, 0, 0, limits(), budget())
                .unwrap()
                .into_value()
                .0;
        assert_eq!(output.alpha_dim(), 0);
        assert_eq!(output.constraints().shape(), &[0, 0]);
        let expected = add_upper(add_upper(0.25, 2.0, "test").unwrap(), 0.5, "test").unwrap();
        assert_eq!(output.box_remainder(), &[expected]);
    }

    #[test]
    fn equal_magnitudes_keep_lower_original_index() {
        let input = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![vec![(0, 1.0)], vec![(0, -1.0)], vec![(0, 1.0)]],
            Array2::zeros((0, 3)),
            Vec::new(),
            vec![0.0],
        )
        .unwrap();
        let output =
            constrained_zonotope_order_reduce_unwired_with_budget(&input, 1, 0, limits(), budget())
                .unwrap()
                .into_value()
                .0;
        assert_eq!(
            output.generators()[0].entries().collect::<Vec<_>>(),
            vec![(0, 1.0)]
        );
    }

    #[test]
    fn predicate_magnitude_can_preserve_a_geometrically_zero_column() {
        let input = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![Vec::new(), vec![(0, 2.0)]],
            array![[10.0, 0.0]],
            vec![0.5],
            vec![0.0],
        )
        .unwrap();
        let output =
            constrained_zonotope_order_reduce_unwired_with_budget(&input, 1, 0, limits(), budget())
                .unwrap()
                .into_value()
                .0;
        assert_eq!(output.alpha_dim(), 1);
        assert_eq!(output.generators()[0].nnz(), 0);
        assert_eq!(output.constraints(), array![[10.0]]);
        assert_eq!(output.box_remainder(), &[2.0]);
    }

    #[test]
    fn identity_capacity_is_bit_exact() {
        let input = sample_domain();
        let output = constrained_zonotope_order_reduce_unwired_with_budget(
            &input,
            input.alpha_dim(),
            input.alpha_dim(),
            limits(),
            budget(),
        )
        .unwrap()
        .into_value()
        .0;
        assert_eq!(output, input);
    }

    #[test]
    fn oversized_request_is_permissively_clamped_to_identity() {
        let input = sample_domain();
        let exact_limits = ConstrainedZonotopeOrderReductionLimits {
            max_value_dim: input.value_dim(),
            max_input_alpha_dim: input.alpha_dim(),
            max_output_alpha_dim: input.alpha_dim(),
            max_constraints: input.constraint_count(),
            max_constraint_elements: input.constraints().len(),
            max_generator_nnz: input
                .generators()
                .iter()
                .map(|generator| generator.nnz())
                .sum(),
        };
        let output = constrained_zonotope_order_reduce_unwired_with_budget(
            &input,
            usize::MAX,
            input.alpha_dim(),
            exact_limits,
            budget(),
        )
        .unwrap()
        .into_value()
        .0;
        assert_eq!(output, input);
    }

    #[test]
    fn peak_live_formula_composes_exactly_with_the_caller_baseline() {
        let geometry = Geometry {
            value_dim: 2,
            input_alpha_dim: 3,
            output_alpha_dim: 2,
            constraint_count: 2,
            input_constraint_elements: 6,
            output_constraint_elements: 4,
            input_generator_nnz: 6,
        };
        let transform_owned = peak_live_bytes(geometry).unwrap();
        let expected = 3 * size_of::<bool>()
            + 2 * size_of::<Reverse<RankedGenerator>>()
            + 4 * size_of::<f64>()
            + 4 * size_of::<f64>()
            + 2 * size_of::<f64>()
            + 4 * size_of::<Vec<(usize, f64)>>()
            + 12 * size_of::<(usize, f64)>()
            + 6 * size_of::<Vec<usize>>();
        assert_eq!(transform_owned, expected);

        let start = Instant::now();
        let baseline = 7;
        let accepted = constrained_zonotope_order_reduce_unwired_with_budget(
            &sample_domain(),
            2,
            1,
            limits(),
            ConstrainedZonotopeCallBudget::new(
                start + Duration::from_mins(1),
                baseline,
                baseline + transform_owned,
            ),
        )
        .unwrap();
        assert_eq!(
            accepted.report().peak_live_bytes(),
            baseline + transform_owned
        );

        let refused = constrained_zonotope_order_reduce_unwired_with_budget(
            &sample_domain(),
            2,
            1,
            limits(),
            ConstrainedZonotopeCallBudget::new(
                start + Duration::from_mins(1),
                baseline,
                baseline + transform_owned - 1,
            ),
        );
        assert!(matches!(
            refused,
            Err(ConstrainedZonotopeOrderReductionBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { .. }
            ))
        ));
    }

    #[test]
    fn invalid_protected_prefix_and_limits_fail_closed() {
        let input = sample_domain();
        assert!(matches!(
            constrained_zonotope_order_reduce_unwired_with_budget(&input, 1, 2, limits(), budget()),
            Err(ConstrainedZonotopeOrderReductionBudgetError::Transform(
                ConstrainedZonotopeOrderReductionError::InvalidProtectedPrefix { .. }
            ))
        ));
        let mut invalid = limits();
        invalid.max_input_alpha_dim = ORDER_REDUCTION_HARD_MAX_INPUT_ALPHA_DIM + 1;
        assert!(matches!(
            constrained_zonotope_order_reduce_unwired_with_budget(&input, 1, 0, invalid, budget()),
            Err(ConstrainedZonotopeOrderReductionBudgetError::Transform(
                ConstrainedZonotopeOrderReductionError::InvalidLimit { .. }
            ))
        ));
    }

    #[test]
    fn allocation_failure_and_preallocation_deadline_are_typed() {
        assert!(matches!(
            reserved_vec::<u8>(usize::MAX, "impossible allocation"),
            Err(ConstrainedZonotopeOrderReductionError::AllocationFailure {
                resource: "impossible allocation"
            })
        ));
        expect_deadline_at(
            &sample_domain(),
            2,
            1,
            limits(),
            "order reduction keep-mask allocation",
        );
    }

    #[test]
    fn memory_limit_refuses_before_selection_allocation() {
        let now = Instant::now();
        let result = constrained_zonotope_order_reduce_unwired_with_budget(
            &sample_domain(),
            2,
            1,
            limits(),
            ConstrainedZonotopeCallBudget::new(now + Duration::from_mins(1), 7, 7),
        );
        assert!(matches!(
            result,
            Err(ConstrainedZonotopeOrderReductionBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { .. }
            ))
        ));
    }

    #[test]
    fn injected_deadline_expires_before_selected_state_is_consumed() {
        let start = Instant::now();
        let expired = start + Duration::from_secs(2);
        let reads = Cell::new(0_usize);
        let mut tracker = ConstrainedZonotopeCallTracker::with_clock(
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, 1 << 20),
            |checkpoint| {
                reads.set(reads.get() + 1);
                if checkpoint == "order reduction selection complete" {
                    expired
                } else {
                    start
                }
            },
        )
        .unwrap();
        let result = reduce_with_gate(&sample_domain(), 2, 1, limits(), &mut tracker);
        assert!(matches!(
            result,
            Err(ConstrainedZonotopeOrderReductionBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "order reduction selection complete"
                }
            ))
        ));
        assert!(reads.get() >= 2);
    }

    #[test]
    fn deadline_polling_continues_through_final_domain_validation() {
        let dimension = crate::CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL;
        let input = ConstrainedZonotope64::try_new(
            vec![0.0; dimension],
            Vec::new(),
            Array2::zeros((0, 0)),
            Vec::new(),
            vec![0.0; dimension],
        )
        .unwrap();
        let mut large_limits = limits();
        large_limits.max_value_dim = dimension;

        let start = Instant::now();
        let expired = start + Duration::from_secs(2);
        let mut tracker = ConstrainedZonotopeCallTracker::with_clock(
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, 1 << 20),
            |checkpoint| {
                if checkpoint == "constrained-zonotope finite-value validation" {
                    expired
                } else {
                    start
                }
            },
        )
        .unwrap();
        let result = reduce_with_gate(&input, 0, 0, large_limits, &mut tracker);
        assert!(matches!(
            result,
            Err(ConstrainedZonotopeOrderReductionBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "constrained-zonotope finite-value validation"
                }
            ))
        ));
    }

    #[test]
    fn hard_polling_covers_bulk_initialization_empty_columns_rows_and_heap() {
        let items = crate::CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL;
        let many_empty_generators = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![Vec::new(); items],
            Array2::zeros((0, items)),
            Vec::new(),
            vec![0.0],
        )
        .unwrap();
        let generator_limits = ConstrainedZonotopeOrderReductionLimits {
            max_value_dim: 1,
            max_input_alpha_dim: items,
            max_output_alpha_dim: items,
            max_constraints: 0,
            max_constraint_elements: 0,
            max_generator_nnz: 0,
        };
        for (phase, output_alpha_dim) in [
            ("order reduction keep-mask initialization", 0),
            ("order reduction generator projection", 0),
            ("order reduction ranking-heap update", 1),
            ("order reduction ranking-heap drain", items / 2),
        ] {
            expect_deadline_at(
                &many_empty_generators,
                output_alpha_dim,
                0,
                generator_limits,
                phase,
            );
        }

        let many_zero_width_rows = ConstrainedZonotope64::try_new(
            Vec::new(),
            Vec::new(),
            Array2::zeros((items, 0)),
            vec![0.0; items],
            Vec::new(),
        )
        .unwrap();
        let row_limits = ConstrainedZonotopeOrderReductionLimits {
            max_value_dim: 0,
            max_input_alpha_dim: 0,
            max_output_alpha_dim: 0,
            max_constraints: items,
            max_constraint_elements: 0,
            max_generator_nnz: 0,
        };
        expect_deadline_at(
            &many_zero_width_rows,
            0,
            0,
            row_limits,
            "order reduction constraint projection rows",
        );
    }

    #[test]
    fn outward_projection_refuses_nonfinite_sum() {
        assert!(matches!(
            add_upper(f64::MAX, f64::MAX, "overflow"),
            Err(
                ConstrainedZonotopeOrderReductionError::NonFiniteArithmetic {
                    operation: "overflow"
                }
            )
        ));
    }

    #[test]
    fn outward_projection_handles_cancellation_subnormals_and_max_boundary() {
        let min_subnormal = f64::from_bits(1);
        for (left, right) in [
            (-1.0, 1.0),
            (-f64::MIN_POSITIVE, f64::MIN_POSITIVE),
            (-f64::from_bits(2), min_subnormal),
            (min_subnormal, min_subnormal),
        ] {
            let exact = rat(left) + rat(right);
            let outward = add_upper(left, right, "adversarial sum").unwrap();
            assert!(rat(outward) >= exact);
        }

        let below_max = f64::MAX.next_down();
        let final_ulp = f64::MAX - below_max;
        assert_eq!(
            add_upper(below_max, final_ulp, "exact maximum").unwrap(),
            f64::MAX
        );
        assert_eq!(
            add_upper(-min_subnormal, f64::MAX, "rounded-up maximum").unwrap(),
            f64::MAX
        );
        assert!(matches!(
            add_upper(f64::MAX, min_subnormal, "above maximum"),
            Err(
                ConstrainedZonotopeOrderReductionError::NonFiniteArithmetic {
                    operation: "above maximum"
                }
            )
        ));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn outward_addition_encloses_the_exact_dyadic_sum(
            left_bits in any::<u64>(),
            right_bits in any::<u64>(),
        ) {
            let left = finite_from_bits(left_bits);
            let right = finite_from_bits(right_bits & 0x7fff_ffff_ffff_ffff);
            let exact = rat(left) + rat(right);
            let maximum = rat(f64::MAX);
            match add_upper(left, right, "property sum") {
                Ok(outward) => {
                    prop_assert!(exact <= maximum);
                    prop_assert!(rat(outward) >= exact);
                }
                Err(ConstrainedZonotopeOrderReductionError::NonFiniteArithmetic {
                    operation: "property sum",
                }) => prop_assert!(exact > maximum),
                other => prop_assert!(false, "unexpected result: {other:?}"),
            }
        }

        #[test]
        fn selection_and_projection_match_the_reference_formula(
            columns in prop::collection::vec(
                (1_u8..=8, any::<bool>(), -8_i8..=8, -8_i8..=8),
                1..=8,
            ),
            rhs0 in -16_i8..=16,
            rhs1 in -16_i8..=16,
            remainder in 0_u8..=8,
            output_seed in any::<u8>(),
            prefix_seed in any::<u8>(),
        ) {
            let alpha_dim = columns.len();
            let output_alpha_dim = usize::from(output_seed) % (alpha_dim + 1);
            let protected_prefix = usize::from(prefix_seed) % (output_alpha_dim + 1);
            let mut generators = Vec::with_capacity(alpha_dim);
            let mut constraint_values = Vec::with_capacity(2 * alpha_dim);
            for (index, &(magnitude, positive, constraint0, _)) in columns.iter().enumerate() {
                let coefficient = if positive {
                    f64::from(magnitude)
                } else {
                    -f64::from(magnitude)
                };
                generators.push(vec![(index, coefficient)]);
                constraint_values.push(f64::from(constraint0));
            }
            constraint_values.extend(
                columns
                    .iter()
                    .map(|&(_, _, _, constraint1)| f64::from(constraint1)),
            );
            let input = ConstrainedZonotope64::try_new(
                vec![0.0; alpha_dim],
                generators,
                Array2::from_shape_vec((2, alpha_dim), constraint_values).unwrap(),
                vec![f64::from(rhs0), f64::from(rhs1)],
                vec![f64::from(remainder); alpha_dim],
            )
            .unwrap();
            let output = constrained_zonotope_order_reduce_unwired_with_budget(
                &input,
                output_alpha_dim,
                protected_prefix,
                limits(),
                budget(),
            )
            .unwrap()
            .into_value()
            .0;

            let score = |index: usize| {
                let (generator, _, constraint0, constraint1) = columns[index];
                generator
                    .max(constraint0.unsigned_abs())
                    .max(constraint1.unsigned_abs())
            };
            let mut ranked = (protected_prefix..alpha_dim).collect::<Vec<_>>();
            ranked.sort_unstable_by(|&left, &right| {
                score(right)
                    .cmp(&score(left))
                    .then_with(|| left.cmp(&right))
            });
            let mut kept = (0..protected_prefix)
                .chain(ranked.into_iter().take(output_alpha_dim - protected_prefix))
                .collect::<Vec<_>>();
            kept.sort_unstable();

            let observed = output
                .generators()
                .iter()
                .map(|generator| generator.entries().next().unwrap().0)
                .collect::<Vec<_>>();
            prop_assert_eq!(&observed, &kept);
            for (output_column, &input_column) in kept.iter().enumerate() {
                prop_assert_eq!(
                    output.generators()[output_column].entries().collect::<Vec<_>>(),
                    input.generators()[input_column].entries().collect::<Vec<_>>()
                );
                prop_assert_eq!(
                    output.constraints()[[0, output_column]],
                    input.constraints()[[0, input_column]]
                );
                prop_assert_eq!(
                    output.constraints()[[1, output_column]],
                    input.constraints()[[1, input_column]]
                );
            }

            for value_index in 0..alpha_dim {
                let mut exact_remainder = rat(f64::from(remainder));
                if kept.binary_search(&value_index).is_err() {
                    exact_remainder += rat(f64::from(columns[value_index].0));
                }
                prop_assert!(rat(output.box_remainder()[value_index]) >= exact_remainder);
            }
            for row in 0..2 {
                let mut exact_rhs = rat(input.rhs()[row]);
                for input_column in 0..alpha_dim {
                    if kept.binary_search(&input_column).is_err() {
                        exact_rhs += rat(input.constraints()[[row, input_column]].abs());
                    }
                }
                prop_assert!(rat(output.rhs()[row]) >= exact_rhs);
            }
        }
    }
}
