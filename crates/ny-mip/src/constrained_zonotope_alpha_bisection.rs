// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact-dyadic zero bisection of one constrained-zonotope alpha symbol.
//!
//! For
//!
//! ```text
//! x = c + G alpha + e,
//! alpha in [-1, 1]^m, C alpha <= d, |e_i| <= r_i,
//! ```
//!
//! this primitive protects the selected alpha column and constructs the two
//! half-domain children by substituting
//!
//! ```text
//! alpha_k = (beta_k + sigma) / 2, sigma in {-1, +1}.
//! ```
//!
//! Alpha order and dimension are unchanged. The negative child uses
//! `sigma = -1` and represents `alpha_k in [-1, 0]`; the positive child uses
//! `sigma = +1` and represents `alpha_k in [0, 1]`. Every binary64 rounding
//! loss in `c + sigma * g_k / 2` and `g_k / 2` is charged to the independent
//! box remainder. Predicate rows use `C_k / 2` and
//! `d - sigma * C_k / 2`; coefficient and subtraction loss are widened only
//! into the child right-hand side.
//!
//! This module is deliberately **unwired**. It supplies a generic partition
//! primitive for future complete search, but no runner, verifier verdict, or
//! scored path selects or consumes these children.

use ndarray::Array2;

use crate::constrained_zonotope64::ConstrainedZonotope64CallGateError;
use crate::constrained_zonotope_call_budget::{
    ConstrainedZonotopeCallBudget, ConstrainedZonotopeCallBudgetError, ConstrainedZonotopeCallGate,
    ConstrainedZonotopeCallOutcome, ConstrainedZonotopeCallTracker,
    ConstrainedZonotopePeakLiveBytes, InertConstrainedZonotopeCallGate,
};
use crate::{ConstrainedZonotope64, ConstrainedZonotope64Error};

/// Explicit resource limits for one protected-alpha bisection.
///
/// There is intentionally no `Default`: an experimental caller must select
/// every cap before duplicating a proof domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeAlphaBisectionLimits {
    /// Maximum flat value dimension.
    pub max_value_dim: usize,
    /// Maximum alpha dimension.
    pub max_alpha_dim: usize,
    /// Maximum sparse generator nonzeros in the input or either child.
    pub max_generator_nonzeros: usize,
    /// Maximum retained predicate rows.
    pub max_constraint_count: usize,
    /// Maximum retained predicate matrix elements.
    pub max_constraint_elements: usize,
}

/// Checked geometry and work accounting for a completed bisection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeAlphaBisectionPlan {
    /// Selected alpha column, preserved at this index in both children.
    pub alpha_axis: usize,
    /// Flat value dimension.
    pub value_dim: usize,
    /// Alpha dimension, unchanged in both children.
    pub alpha_dim: usize,
    /// Predicate-row count, unchanged in both children.
    pub constraint_count: usize,
    /// Predicate matrix elements copied into each child.
    pub constraint_elements_per_child: usize,
    /// Sparse input generator nonzeros.
    pub input_generator_nonzeros: usize,
    /// Nonzeros in the selected generator column before scaling.
    pub split_generator_nonzeros: usize,
    /// Sparse generator nonzeros stored in each child after scaling.
    pub output_generator_nonzeros_per_child: usize,
    /// Value and predicate coefficients halved for each child.
    pub halved_terms_per_child: usize,
}

/// The two outward children of an exact protected-alpha zero partition.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstrainedZonotopeAlphaBisection {
    negative: ConstrainedZonotope64,
    positive: ConstrainedZonotope64,
}

impl ConstrainedZonotopeAlphaBisection {
    /// Child representing the original selected alpha in `[-1, 0]`.
    #[must_use]
    pub const fn negative(&self) -> &ConstrainedZonotope64 {
        &self.negative
    }

    /// Child representing the original selected alpha in `[0, 1]`.
    #[must_use]
    pub const fn positive(&self) -> &ConstrainedZonotope64 {
        &self.positive
    }

    /// Consume the partition in `(negative, positive)` order.
    #[must_use]
    pub fn into_children(self) -> (ConstrainedZonotope64, ConstrainedZonotope64) {
        (self.negative, self.positive)
    }
}

/// Invalid split geometry, exhausted resources, or non-enclosable arithmetic.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConstrainedZonotopeAlphaBisectionError {
    /// The requested alpha column does not exist.
    #[error("alpha axis {axis} is out of range for alpha dimension {alpha_dim}")]
    AlphaAxisOutOfRange {
        /// Invalid requested column.
        axis: usize,
        /// Available alpha dimension.
        alpha_dim: usize,
    },

    /// A checked size or work calculation overflowed `usize`.
    #[error("resource size overflow while computing {operation}")]
    ResourceOverflow {
        /// Calculation that overflowed.
        operation: &'static str,
    },

    /// An explicit caller-selected cap was exceeded.
    #[error("resource limit exceeded for {resource}: required {required}, limit {limit}")]
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Required count at rejection.
        required: usize,
        /// Configured maximum.
        limit: usize,
    },

    /// A bounded allocation request was rejected by the allocator.
    #[error("unable to reserve storage for {resource}")]
    AllocationFailure {
        /// Requested container.
        resource: &'static str,
    },

    /// A finite upper enclosure could not be represented.
    #[error("non-finite outward arithmetic while computing {operation}")]
    NonFiniteArithmetic {
        /// Failed arithmetic step.
        operation: &'static str,
    },

    /// The host cannot support error-free binary64 splitting.
    #[error("unsupported floating-point environment: {requirement}")]
    UnsupportedFloatingPoint {
        /// Required IEEE behavior.
        requirement: &'static str,
    },

    /// The validated child could not be materialized as a domain.
    #[error(transparent)]
    Domain(#[from] ConstrainedZonotope64Error),
}

/// Primitive or call-firewall refusal from a budgeted bisection.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConstrainedZonotopeAlphaBisectionBudgetError {
    /// Split geometry, resources, or outward arithmetic were invalid.
    #[error(transparent)]
    Transform(#[from] ConstrainedZonotopeAlphaBisectionError),

    /// The caller's deadline or aggregate peak-memory ceiling refused work.
    #[error(transparent)]
    Budget(#[from] ConstrainedZonotopeCallBudgetError),
}

/// Bisect one protected alpha symbol at zero without wiring the result into a
/// verifier or verdict path.
pub fn bisect_constrained_zonotope_protected_alpha_unwired(
    input: &ConstrainedZonotope64,
    alpha_axis: usize,
    limits: ConstrainedZonotopeAlphaBisectionLimits,
) -> Result<
    (
        ConstrainedZonotopeAlphaBisection,
        ConstrainedZonotopeAlphaBisectionPlan,
    ),
    ConstrainedZonotopeAlphaBisectionError,
> {
    let mut gate = InertConstrainedZonotopeCallGate;
    match bisection_impl(input, alpha_axis, limits, &mut gate) {
        Ok(value) => Ok(value),
        Err(ConstrainedZonotopeAlphaBisectionBudgetError::Transform(error)) => Err(error),
        Err(ConstrainedZonotopeAlphaBisectionBudgetError::Budget(_)) => {
            unreachable!("the inert alpha-bisection call gate cannot refuse work")
        }
    }
}

/// Bisect one protected alpha symbol behind the shared synchronous call
/// firewall.
///
/// `budget.baseline_live_bytes()` must include the borrowed input and other
/// caller-retained storage. The complete logical peak for both unpublished
/// children is preflighted before cloning either child.
pub fn bisect_constrained_zonotope_protected_alpha_unwired_with_budget(
    input: &ConstrainedZonotope64,
    alpha_axis: usize,
    limits: ConstrainedZonotopeAlphaBisectionLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> Result<
    ConstrainedZonotopeCallOutcome<(
        ConstrainedZonotopeAlphaBisection,
        ConstrainedZonotopeAlphaBisectionPlan,
    )>,
    ConstrainedZonotopeAlphaBisectionBudgetError,
> {
    let mut gate = ConstrainedZonotopeCallTracker::from_system_clock(budget)?;
    let value = bisection_impl(input, alpha_axis, limits, &mut gate)?;
    Ok(ConstrainedZonotopeCallOutcome::new(value, gate.report()))
}

#[cfg(test)]
fn bisect_constrained_zonotope_protected_alpha_unwired_with_clock<N>(
    input: &ConstrainedZonotope64,
    alpha_axis: usize,
    limits: ConstrainedZonotopeAlphaBisectionLimits,
    budget: ConstrainedZonotopeCallBudget,
    now: N,
) -> Result<
    ConstrainedZonotopeCallOutcome<(
        ConstrainedZonotopeAlphaBisection,
        ConstrainedZonotopeAlphaBisectionPlan,
    )>,
    ConstrainedZonotopeAlphaBisectionBudgetError,
>
where
    N: FnMut(&'static str) -> std::time::Instant,
{
    let mut gate = ConstrainedZonotopeCallTracker::with_clock(budget, now)?;
    let value = bisection_impl(input, alpha_axis, limits, &mut gate)?;
    Ok(ConstrainedZonotopeCallOutcome::new(value, gate.report()))
}

fn bisection_impl<G>(
    input: &ConstrainedZonotope64,
    alpha_axis: usize,
    limits: ConstrainedZonotopeAlphaBisectionLimits,
    gate: &mut G,
) -> Result<
    (
        ConstrainedZonotopeAlphaBisection,
        ConstrainedZonotopeAlphaBisectionPlan,
    ),
    ConstrainedZonotopeAlphaBisectionBudgetError,
>
where
    G: ConstrainedZonotopeCallGate,
{
    require_binary64_proof_environment()?;
    gate.checkpoint("alpha bisection floating-point preflight")?;
    let geometry = validate_geometry(input, alpha_axis, limits, gate)?;
    gate.checkpoint("alpha bisection geometry validation complete")?;

    if gate.is_enforcing() {
        gate.preflight_peak_live_bytes(bisection_peak_live_bytes(geometry)?)?;
    }
    gate.checkpoint("alpha bisection peak-memory preflight complete")?;

    let negative = build_child(input, alpha_axis, -1.0, geometry, gate)?;
    gate.checkpoint("alpha bisection negative child complete")?;
    let positive = build_child(input, alpha_axis, 1.0, geometry, gate)?;
    gate.checkpoint("alpha bisection positive child complete")?;

    debug_assert_eq!(negative.alpha_dim(), input.alpha_dim());
    debug_assert_eq!(positive.alpha_dim(), input.alpha_dim());
    debug_assert_eq!(negative.constraint_count(), input.constraint_count());
    debug_assert_eq!(positive.constraint_count(), input.constraint_count());
    debug_assert_eq!(
        total_generator_nonzeros(&negative),
        geometry.output_generator_nonzeros
    );
    debug_assert_eq!(
        total_generator_nonzeros(&positive),
        geometry.output_generator_nonzeros
    );

    let plan = ConstrainedZonotopeAlphaBisectionPlan {
        alpha_axis,
        value_dim: input.value_dim(),
        alpha_dim: input.alpha_dim(),
        constraint_count: input.constraint_count(),
        constraint_elements_per_child: geometry.constraint_elements,
        input_generator_nonzeros: geometry.input_generator_nonzeros,
        split_generator_nonzeros: geometry.split_generator_nonzeros,
        output_generator_nonzeros_per_child: geometry.output_generator_nonzeros,
        halved_terms_per_child: geometry.halved_terms,
    };
    let children = ConstrainedZonotopeAlphaBisection { negative, positive };
    gate.checkpoint("alpha bisection publication")?;
    Ok((children, plan))
}

#[derive(Clone, Copy, Debug)]
struct Geometry {
    value_dim: usize,
    alpha_dim: usize,
    constraint_count: usize,
    constraint_elements: usize,
    input_generator_nonzeros: usize,
    split_generator_nonzeros: usize,
    output_generator_nonzeros: usize,
    halved_terms: usize,
}

fn validate_geometry<G>(
    input: &ConstrainedZonotope64,
    alpha_axis: usize,
    limits: ConstrainedZonotopeAlphaBisectionLimits,
    gate: &mut G,
) -> Result<Geometry, ConstrainedZonotopeAlphaBisectionBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    if alpha_axis >= input.alpha_dim() {
        return Err(
            ConstrainedZonotopeAlphaBisectionError::AlphaAxisOutOfRange {
                axis: alpha_axis,
                alpha_dim: input.alpha_dim(),
            }
            .into(),
        );
    }
    check_limit("value dimension", input.value_dim(), limits.max_value_dim)?;
    check_limit("alpha dimension", input.alpha_dim(), limits.max_alpha_dim)?;
    check_limit(
        "constraint count",
        input.constraint_count(),
        limits.max_constraint_count,
    )?;
    let constraint_elements = input
        .constraint_count()
        .checked_mul(input.alpha_dim())
        .ok_or(ConstrainedZonotopeAlphaBisectionError::ResourceOverflow {
            operation: "constraint matrix elements",
        })?;
    check_limit(
        "constraint matrix elements",
        constraint_elements,
        limits.max_constraint_elements,
    )?;

    let mut input_generator_nonzeros = 0_usize;
    for generator in input.generators() {
        gate.charge_items(1, "alpha bisection generator geometry")?;
        input_generator_nonzeros = input_generator_nonzeros
            .checked_add(generator.nnz())
            .ok_or(ConstrainedZonotopeAlphaBisectionError::ResourceOverflow {
                operation: "input generator nonzeros",
            })?;
    }
    check_limit(
        "generator nonzeros",
        input_generator_nonzeros,
        limits.max_generator_nonzeros,
    )?;
    let split_generator_nonzeros = input.generators()[alpha_axis].nnz();
    let mut scaled_nonzeros = 0_usize;
    for (_, coefficient) in input.generators()[alpha_axis].entries() {
        gate.charge_items(1, "alpha bisection selected-generator preflight")?;
        if coefficient * 0.5 != 0.0 {
            scaled_nonzeros = scaled_nonzeros.checked_add(1).ok_or(
                ConstrainedZonotopeAlphaBisectionError::ResourceOverflow {
                    operation: "scaled generator nonzeros",
                },
            )?;
        }
    }
    let output_generator_nonzeros = input_generator_nonzeros
        .checked_sub(split_generator_nonzeros)
        .and_then(|count| count.checked_add(scaled_nonzeros))
        .ok_or(ConstrainedZonotopeAlphaBisectionError::ResourceOverflow {
            operation: "output generator nonzeros",
        })?;
    check_limit(
        "generator nonzeros",
        output_generator_nonzeros,
        limits.max_generator_nonzeros,
    )?;
    let halved_terms = split_generator_nonzeros
        .checked_add(input.constraint_count())
        .ok_or(ConstrainedZonotopeAlphaBisectionError::ResourceOverflow {
            operation: "halved split terms",
        })?;

    Ok(Geometry {
        value_dim: input.value_dim(),
        alpha_dim: input.alpha_dim(),
        constraint_count: input.constraint_count(),
        constraint_elements,
        input_generator_nonzeros,
        split_generator_nonzeros,
        output_generator_nonzeros,
        halved_terms,
    })
}

fn bisection_peak_live_bytes(
    geometry: Geometry,
) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    // At the peak, the first child is retained while the second child has both
    // candidate and validated representations live during materialization.
    // Three complete logical copies are therefore charged. The input belongs
    // in the caller's baseline.
    let mut peak = ConstrainedZonotopePeakLiveBytes::new();
    peak.add_elements::<f64>(
        checked_peak_product(
            geometry.value_dim,
            6,
            "alpha-bisection center/remainder copies",
        )?,
        "alpha-bisection center/remainder bytes",
    )?;
    peak.add_elements::<Vec<(usize, f64)>>(
        checked_peak_product(
            geometry.alpha_dim,
            3,
            "alpha-bisection generator header copies",
        )?,
        "alpha-bisection generator-header bytes",
    )?;
    peak.add_elements::<(usize, f64)>(
        checked_peak_product(
            geometry.input_generator_nonzeros,
            3,
            "alpha-bisection generator entry copies",
        )?,
        "alpha-bisection generator-entry bytes",
    )?;
    peak.add_elements::<f64>(
        checked_peak_product(
            geometry.constraint_elements,
            3,
            "alpha-bisection predicate matrix copies",
        )?,
        "alpha-bisection predicate-matrix bytes",
    )?;
    peak.add_elements::<f64>(
        checked_peak_product(
            geometry.constraint_count,
            3,
            "alpha-bisection predicate rhs copies",
        )?,
        "alpha-bisection predicate-rhs bytes",
    )?;
    Ok(peak.finish())
}

fn checked_peak_product(
    left: usize,
    right: usize,
    operation: &'static str,
) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    left.checked_mul(right)
        .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow { operation })
}

fn build_child<G>(
    input: &ConstrainedZonotope64,
    alpha_axis: usize,
    sigma: f64,
    geometry: Geometry,
    gate: &mut G,
) -> Result<ConstrainedZonotope64, ConstrainedZonotopeAlphaBisectionBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    debug_assert!(sigma == -1.0 || sigma == 1.0);
    let branch = if sigma < 0.0 { "negative" } else { "positive" };

    let mut center = clone_slice(
        input.center(),
        "alpha-bisection child center",
        "alpha bisection center clone",
        gate,
    )?;
    let mut box_remainder = clone_slice(
        input.box_remainder(),
        "alpha-bisection child box remainder",
        "alpha bisection remainder clone",
        gate,
    )?;

    let mut generators = Vec::new();
    gate.checkpoint("alpha bisection generator-column allocation")?;
    reserve(
        &mut generators,
        input.alpha_dim(),
        "alpha-bisection generator columns",
    )?;
    for (generator_index, generator) in input.generators().iter().enumerate() {
        gate.charge_items(1, "alpha bisection generator-column clone")?;
        let mut entries = Vec::new();
        gate.checkpoint("alpha bisection generator-entry allocation")?;
        reserve(
            &mut entries,
            generator.nnz(),
            "alpha-bisection generator entries",
        )?;
        for (value_index, coefficient) in generator.entries() {
            gate.charge_items(1, "alpha bisection generator-entry transform")?;
            if generator_index == alpha_axis {
                let (scaled, scaling_error) = half_with_error(coefficient)?;
                let (shifted, addition_error) = two_sum(center[value_index], sigma * scaled)?;
                center[value_index] = shifted;
                let mut representation_error = addition_error.abs();
                representation_error = add_nonnegative_upper(
                    representation_error,
                    scaling_error,
                    "alpha-bisection center scaling error",
                )?;
                representation_error = add_nonnegative_upper(
                    representation_error,
                    scaling_error,
                    "alpha-bisection generator scaling error",
                )?;
                box_remainder[value_index] = add_nonnegative_upper(
                    box_remainder[value_index],
                    representation_error,
                    "alpha-bisection box remainder",
                )?;
                if scaled != 0.0 {
                    entries.push((value_index, scaled));
                }
            } else {
                entries.push((value_index, coefficient));
            }
        }
        generators.push(entries);
    }
    debug_assert_eq!(
        generators.iter().map(Vec::len).sum::<usize>(),
        geometry.output_generator_nonzeros
    );
    gate.checkpoint(if branch == "negative" {
        "alpha bisection negative value transform complete"
    } else {
        "alpha bisection positive value transform complete"
    })?;

    let mut constraint_values = Vec::new();
    gate.checkpoint("alpha bisection constraint-matrix allocation")?;
    reserve(
        &mut constraint_values,
        geometry.constraint_elements,
        "alpha-bisection constraint matrix",
    )?;
    let mut rhs = Vec::new();
    gate.checkpoint("alpha bisection predicate-rhs allocation")?;
    reserve(
        &mut rhs,
        input.constraint_count(),
        "alpha-bisection right-hand side",
    )?;
    let constraints = input.constraints();
    for row in 0..input.constraint_count() {
        gate.charge_items(1, "alpha bisection predicate-row transform")?;
        let selected = constraints[[row, alpha_axis]];
        let (scaled, scaling_error) = half_with_error(selected)?;
        for column in 0..input.alpha_dim() {
            gate.charge_items(1, "alpha bisection predicate-element clone")?;
            constraint_values.push(if column == alpha_axis {
                scaled
            } else {
                constraints[[row, column]]
            });
        }

        let (shifted_rhs, subtraction_error) = two_sum(input.rhs()[row], -sigma * scaled)?;
        // Let exact C_k/2 = scaled + delta, |delta| <= scaling_error.
        // Relative to the exact substituted row, storing `scaled` changes the
        // left side by `-delta * beta` and changes the shifted right side by
        // `-sigma * delta`. Their combined positive displacement is at most
        // `2 * scaling_error` for beta in [-1, 1]. A negative TwoSum residual
        // already makes the stored right side looser and needs no charge.
        let mut rhs_widening = subtraction_error.max(0.0);
        rhs_widening = add_nonnegative_upper(
            rhs_widening,
            scaling_error,
            "alpha-bisection predicate coefficient error",
        )?;
        rhs_widening = add_nonnegative_upper(
            rhs_widening,
            scaling_error,
            "alpha-bisection predicate shift error",
        )?;
        rhs.push(add_upper(
            shifted_rhs,
            rhs_widening,
            "alpha-bisection predicate rhs",
        )?);
    }
    let constraints = Array2::from_shape_vec(
        (input.constraint_count(), input.alpha_dim()),
        constraint_values,
    )
    .map_err(
        |_| ConstrainedZonotopeAlphaBisectionError::ResourceOverflow {
            operation: "constraint matrix shape",
        },
    )?;
    gate.checkpoint(if branch == "negative" {
        "alpha bisection negative predicate transform complete"
    } else {
        "alpha bisection positive predicate transform complete"
    })?;

    gate.checkpoint(if branch == "negative" {
        "alpha bisection negative domain materialization"
    } else {
        "alpha bisection positive domain materialization"
    })?;
    ConstrainedZonotope64::try_new_with_call_gate(
        center,
        generators,
        constraints,
        rhs,
        box_remainder,
        gate,
    )
    .map_err(|error| match error {
        ConstrainedZonotope64CallGateError::Domain(error) => {
            ConstrainedZonotopeAlphaBisectionBudgetError::Transform(
                ConstrainedZonotopeAlphaBisectionError::Domain(error),
            )
        }
        ConstrainedZonotope64CallGateError::Budget(error) => {
            ConstrainedZonotopeAlphaBisectionBudgetError::Budget(error)
        }
    })
}

fn total_generator_nonzeros(domain: &ConstrainedZonotope64) -> usize {
    domain.generators().iter().map(|column| column.nnz()).sum()
}

fn clone_slice<T: Copy, G>(
    source: &[T],
    resource: &'static str,
    checkpoint: &'static str,
    gate: &mut G,
) -> Result<Vec<T>, ConstrainedZonotopeAlphaBisectionBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let mut output = Vec::new();
    gate.checkpoint(checkpoint)?;
    reserve(&mut output, source.len(), resource)?;
    for &value in source {
        gate.charge_items(1, checkpoint)?;
        output.push(value);
    }
    Ok(output)
}

fn reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    resource: &'static str,
) -> Result<(), ConstrainedZonotopeAlphaBisectionError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| ConstrainedZonotopeAlphaBisectionError::AllocationFailure { resource })
}

fn check_limit(
    resource: &'static str,
    required: usize,
    limit: usize,
) -> Result<(), ConstrainedZonotopeAlphaBisectionError> {
    if required > limit {
        return Err(ConstrainedZonotopeAlphaBisectionError::ResourceLimit {
            resource,
            required,
            limit,
        });
    }
    Ok(())
}

/// Return `value / 2` and a finite radius enclosing any underflow-rounding
/// loss. Every non-underflowing binary64 division by two is exact. Under
/// gradual underflow, an inexact subnormal half differs by exactly half of the
/// least subnormal, so one least subnormal is a finite outward enclosure.
fn half_with_error(value: f64) -> Result<(f64, f64), ConstrainedZonotopeAlphaBisectionError> {
    let half = value * 0.5;
    if !half.is_finite() {
        return Err(
            ConstrainedZonotopeAlphaBisectionError::NonFiniteArithmetic {
                operation: "alpha-bisection coefficient scaling",
            },
        );
    }
    let error = if half * 2.0 == value {
        0.0
    } else {
        f64::from_bits(1)
    };
    Ok((half, error))
}

/// Error-free TwoSum: returns a rounded sum and an exactly represented
/// residual such that `sum + residual` is the exact sum of the two finite
/// binary64 inputs.
fn two_sum(left: f64, right: f64) -> Result<(f64, f64), ConstrainedZonotopeAlphaBisectionError> {
    let sum = left + right;
    if !sum.is_finite() {
        return Err(
            ConstrainedZonotopeAlphaBisectionError::NonFiniteArithmetic {
                operation: "alpha-bisection exact sum",
            },
        );
    }
    let right_virtual = sum - left;
    let left_virtual = sum - right_virtual;
    let right_roundoff = right - right_virtual;
    let left_roundoff = left - left_virtual;
    let residual = left_roundoff + right_roundoff;
    if !residual.is_finite() {
        return Err(
            ConstrainedZonotopeAlphaBisectionError::NonFiniteArithmetic {
                operation: "alpha-bisection sum residual",
            },
        );
    }
    Ok((sum, residual))
}

fn add_nonnegative_upper(
    left: f64,
    right: f64,
    operation: &'static str,
) -> Result<f64, ConstrainedZonotopeAlphaBisectionError> {
    debug_assert!(left >= 0.0 && right >= 0.0);
    if left == 0.0 {
        return Ok(right);
    }
    if right == 0.0 {
        return Ok(left);
    }
    add_upper(left, right, operation)
}

/// Least adjacent upper enclosure of an exact sum, using TwoSum's residual to
/// avoid widening when the rounded result is already on the upper side.
fn add_upper(
    left: f64,
    right: f64,
    operation: &'static str,
) -> Result<f64, ConstrainedZonotopeAlphaBisectionError> {
    if right == 0.0 {
        return Ok(left);
    }
    let (sum, residual) = two_sum(left, right)?;
    if residual <= 0.0 {
        return Ok(sum);
    }
    let upper = sum.next_up();
    if !upper.is_finite() {
        return Err(ConstrainedZonotopeAlphaBisectionError::NonFiniteArithmetic { operation });
    }
    Ok(upper)
}

fn require_binary64_proof_environment() -> Result<(), ConstrainedZonotopeAlphaBisectionError> {
    // Reuse the process-wide proof-environment probe rather than maintaining a
    // weaker local copy.  In particular, the shared probe exercises halfway
    // additions on *both* sides of one, which distinguishes every directed
    // rounding mode from round-to-nearest/ties-to-even.
    if !ny_core::has_f64_interval_proof_environment() {
        return Err(
            ConstrainedZonotopeAlphaBisectionError::UnsupportedFloatingPoint {
                requirement: "IEEE-754 binary64 round-to-nearest-ties-even with gradual underflow",
            },
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::mem::size_of;
    use std::time::{Duration, Instant};

    use ndarray::{array, Array2};
    use num_rational::BigRational;
    use num_traits::{Signed, Zero};
    use proptest::prelude::*;

    use super::*;

    fn limits() -> ConstrainedZonotopeAlphaBisectionLimits {
        ConstrainedZonotopeAlphaBisectionLimits {
            max_value_dim: 10_000,
            max_alpha_dim: 10_000,
            max_generator_nonzeros: 100_000,
            max_constraint_count: 100_000,
            max_constraint_elements: 1_000_000,
        }
    }

    fn sample_input() -> ConstrainedZonotope64 {
        ConstrainedZonotope64::try_new(
            vec![0.25, -1.5],
            vec![vec![(0, 0.5), (1, -0.25)], vec![(0, 0.75)]],
            array![[0.5, -0.25], [-0.75, 0.5]],
            vec![0.75, 1.0],
            vec![0.125, 0.25],
        )
        .unwrap()
    }

    fn rat(value: f64) -> BigRational {
        BigRational::from_float(value).expect("finite test value")
    }

    fn coefficient(domain: &ConstrainedZonotope64, alpha: usize, coordinate: usize) -> f64 {
        domain.generators()[alpha]
            .entries()
            .find_map(|(index, value)| (index == coordinate).then_some(value))
            .unwrap_or(0.0)
    }

    fn exact_nominal(
        domain: &ConstrainedZonotope64,
        alphas: &[BigRational],
        coordinate: usize,
    ) -> BigRational {
        let mut value = rat(domain.center()[coordinate]);
        for (alpha, generator) in alphas.iter().zip(domain.generators()) {
            for (value_index, generator_coefficient) in generator.entries() {
                if value_index == coordinate {
                    value += alpha * rat(generator_coefficient);
                }
            }
        }
        value
    }

    fn row_slack(
        domain: &ConstrainedZonotope64,
        alphas: &[BigRational],
        row: usize,
    ) -> BigRational {
        let mut lhs = BigRational::zero();
        for column in 0..domain.alpha_dim() {
            lhs += rat(domain.constraints()[[row, column]]) * &alphas[column];
        }
        lhs - rat(domain.rhs()[row])
    }

    fn mapped_beta(original: &[BigRational], alpha_axis: usize, sigma: i32) -> Vec<BigRational> {
        let mut beta = original.to_vec();
        beta[alpha_axis] = &original[alpha_axis] * BigRational::from_integer(2.into())
            - BigRational::from_integer(sigma.into());
        beta
    }

    fn assert_mapped_witness_contained(
        input: &ConstrainedZonotope64,
        child: &ConstrainedZonotope64,
        original_alpha: &[BigRational],
        child_alpha: &[BigRational],
    ) {
        for coordinate in 0..input.value_dim() {
            let nominal_difference = (exact_nominal(input, original_alpha, coordinate)
                - exact_nominal(child, child_alpha, coordinate))
            .abs();
            let available =
                rat(child.box_remainder()[coordinate]) - rat(input.box_remainder()[coordinate]);
            assert!(
                nominal_difference <= available,
                "coordinate {coordinate}: difference={nominal_difference}, available={available}"
            );
        }
        for row in 0..input.constraint_count() {
            if row_slack(input, original_alpha, row) <= BigRational::zero() {
                assert!(
                    row_slack(child, child_alpha, row) <= BigRational::zero(),
                    "mapped witness violates child row {row}"
                );
            }
        }
    }

    #[test]
    fn dyadic_split_preserves_alpha_order_and_exact_sampled_partition() {
        let input = sample_input();
        let (children, plan) =
            bisect_constrained_zonotope_protected_alpha_unwired(&input, 0, limits()).unwrap();

        assert_eq!(children.negative().alpha_dim(), input.alpha_dim());
        assert_eq!(children.positive().alpha_dim(), input.alpha_dim());
        assert_eq!(plan.alpha_axis, 0);
        assert_eq!(plan.value_dim, 2);
        assert_eq!(plan.alpha_dim, 2);
        assert_eq!(plan.constraint_count, 2);
        assert_eq!(plan.constraint_elements_per_child, 4);
        assert_eq!(plan.input_generator_nonzeros, 3);
        assert_eq!(plan.split_generator_nonzeros, 2);
        assert_eq!(plan.output_generator_nonzeros_per_child, 3);
        assert_eq!(plan.halved_terms_per_child, 4);

        for child in [children.negative(), children.positive()] {
            assert_eq!(coefficient(child, 0, 0), 0.25);
            assert_eq!(coefficient(child, 0, 1), -0.125);
            assert_eq!(coefficient(child, 1, 0), 0.75);
            assert_eq!(child.constraints(), array![[0.25, -0.25], [-0.375, 0.5]]);
            assert_eq!(child.box_remainder(), input.box_remainder());
        }
        assert_eq!(children.negative().center(), &[0.0, -1.375]);
        assert_eq!(children.positive().center(), &[0.5, -1.625]);
        assert_eq!(children.negative().rhs(), &[1.0, 0.625]);
        assert_eq!(children.positive().rhs(), &[0.5, 1.375]);

        let halves = [
            (-1_i32, children.negative(), [-4_i32, -2, 0]),
            (1_i32, children.positive(), [0_i32, 2, 4]),
        ];
        for (sigma, child, alpha0_quarters) in halves {
            for alpha0_quarters in alpha0_quarters {
                for alpha1_quarters in [-4_i32, 0, 4] {
                    let original = vec![
                        BigRational::new(alpha0_quarters.into(), 4.into()),
                        BigRational::new(alpha1_quarters.into(), 4.into()),
                    ];
                    let beta = mapped_beta(&original, 0, sigma);
                    for coordinate in 0..input.value_dim() {
                        assert_eq!(
                            exact_nominal(&input, &original, coordinate),
                            exact_nominal(child, &beta, coordinate)
                        );
                    }
                    for row in 0..input.constraint_count() {
                        assert_eq!(
                            row_slack(&input, &original, row),
                            row_slack(child, &beta, row)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn nondyadic_and_subnormal_rounding_is_charged_outward() {
        let least = f64::from_bits(1);
        let three_least = f64::from_bits(3);
        let half_ulp_at_one = f64::from_bits(0x3ca0_0000_0000_0000);
        let input = ConstrainedZonotope64::try_new(
            vec![1.0, f64::from_bits(2)],
            vec![vec![(0, half_ulp_at_one * 2.0), (1, three_least)]],
            array![[half_ulp_at_one * 2.0], [three_least]],
            vec![1.0, f64::from_bits(2)],
            vec![0.0, 0.0],
        )
        .unwrap();
        let (children, _) =
            bisect_constrained_zonotope_protected_alpha_unwired(&input, 0, limits()).unwrap();

        assert_eq!(children.positive().center()[0], 1.0);
        assert!(
            rat(children.positive().box_remainder()[0]) >= rat(half_ulp_at_one),
            "the tie-to-even center loss must enter the remainder"
        );
        for child in [children.negative(), children.positive()] {
            assert_eq!(coefficient(child, 0, 1), f64::from_bits(2));
            assert!(
                rat(child.box_remainder()[1]) >= rat(least) * BigRational::from_integer(2.into()),
                "center and generator underflow losses must both be charged"
            );
        }

        for (sigma, child, original_values) in [
            (-1_i32, children.negative(), [-1_i32, -1, 0]),
            (1_i32, children.positive(), [0_i32, 1, 1]),
        ] {
            for (numerator, denominator) in original_values.into_iter().zip([1_i32, 2, 1]) {
                let original = vec![BigRational::new(numerator.into(), denominator.into())];
                let beta = mapped_beta(&original, 0, sigma);
                assert_mapped_witness_contained(&input, child, &original, &beta);
            }
        }
    }

    #[test]
    fn empty_scaled_column_still_preserves_the_alpha_axis() {
        let least = f64::from_bits(1);
        let input = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![vec![(0, least)], vec![(0, 1.0)]],
            array![[least, 1.0]],
            vec![1.0],
            vec![0.0],
        )
        .unwrap();
        let (children, plan) =
            bisect_constrained_zonotope_protected_alpha_unwired(&input, 0, limits()).unwrap();
        assert_eq!(plan.output_generator_nonzeros_per_child, 1);
        for child in [children.negative(), children.positive()] {
            assert_eq!(child.alpha_dim(), 2);
            assert_eq!(child.generators()[0].nnz(), 0);
            assert_eq!(
                child.generators()[1].entries().collect::<Vec<_>>(),
                vec![(0, 1.0)]
            );
        }
    }

    #[test]
    fn malformed_axis_and_explicit_limits_fail_before_duplication() {
        let input = sample_input();
        assert_eq!(
            bisect_constrained_zonotope_protected_alpha_unwired(&input, 2, limits()),
            Err(
                ConstrainedZonotopeAlphaBisectionError::AlphaAxisOutOfRange {
                    axis: 2,
                    alpha_dim: 2
                }
            )
        );
        let no_alpha = ConstrainedZonotope64::try_new(
            vec![0.0],
            Vec::new(),
            Array2::zeros((0, 0)),
            Vec::new(),
            vec![0.0],
        )
        .unwrap();
        assert!(matches!(
            bisect_constrained_zonotope_protected_alpha_unwired(&no_alpha, 0, limits()),
            Err(
                ConstrainedZonotopeAlphaBisectionError::AlphaAxisOutOfRange {
                    axis: 0,
                    alpha_dim: 0
                }
            )
        ));

        let mut bounded = limits();
        bounded.max_constraint_elements = 3;
        assert!(matches!(
            bisect_constrained_zonotope_protected_alpha_unwired(&input, 0, bounded),
            Err(ConstrainedZonotopeAlphaBisectionError::ResourceLimit {
                resource: "constraint matrix elements",
                required: 4,
                limit: 3
            })
        ));
    }

    #[test]
    fn budgeted_split_is_identical_and_accounts_both_private_children() {
        let input = sample_input();
        let legacy =
            bisect_constrained_zonotope_protected_alpha_unwired(&input, 0, limits()).unwrap();
        let deadline = Instant::now() + Duration::from_mins(1);
        let baseline = 7_usize;
        let outcome = bisect_constrained_zonotope_protected_alpha_unwired_with_budget(
            &input,
            0,
            limits(),
            ConstrainedZonotopeCallBudget::new(deadline, baseline, usize::MAX),
        )
        .unwrap();
        assert_eq!(outcome.value(), &legacy);

        let transform_peak = 6 * input.value_dim() * size_of::<f64>()
            + 3 * input.alpha_dim() * size_of::<Vec<(usize, f64)>>()
            + 3 * 3 * size_of::<(usize, f64)>()
            + 3 * 4 * size_of::<f64>()
            + 3 * input.constraint_count() * size_of::<f64>();
        assert_eq!(
            outcome.report().peak_live_bytes(),
            baseline + transform_peak
        );
        assert!(outcome.report().charged_items() > 0);
        assert!(outcome.report().deadline_polls() > 0);

        let exact_peak = outcome.report().peak_live_bytes();
        bisect_constrained_zonotope_protected_alpha_unwired_with_budget(
            &input,
            0,
            limits(),
            ConstrainedZonotopeCallBudget::new(deadline, baseline, exact_peak),
        )
        .unwrap();
        assert!(matches!(
            bisect_constrained_zonotope_protected_alpha_unwired_with_budget(
                &input,
                0,
                limits(),
                ConstrainedZonotopeCallBudget::new(deadline, baseline, exact_peak - 1),
            ),
            Err(ConstrainedZonotopeAlphaBisectionBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { .. }
            ))
        ));
    }

    #[test]
    fn deadline_refuses_every_publication_phase_without_returning_children() {
        let input = sample_input();
        let start = Instant::now();
        let expired = start + Duration::from_secs(2);
        for seam in [
            "alpha bisection floating-point preflight",
            "alpha bisection geometry validation complete",
            "alpha bisection peak-memory preflight complete",
            "alpha bisection center clone",
            "alpha bisection remainder clone",
            "alpha bisection generator-column allocation",
            "alpha bisection generator-entry allocation",
            "alpha bisection negative value transform complete",
            "alpha bisection constraint-matrix allocation",
            "alpha bisection predicate-rhs allocation",
            "alpha bisection negative predicate transform complete",
            "alpha bisection negative domain materialization",
            "alpha bisection negative child complete",
            "alpha bisection positive value transform complete",
            "alpha bisection positive predicate transform complete",
            "alpha bisection positive domain materialization",
            "alpha bisection positive child complete",
            "alpha bisection publication",
        ] {
            let reads = Cell::new(0_usize);
            let result = bisect_constrained_zonotope_protected_alpha_unwired_with_clock(
                &input,
                0,
                limits(),
                ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
                |checkpoint| {
                    reads.set(reads.get() + 1);
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
                    Err(ConstrainedZonotopeAlphaBisectionBudgetError::Budget(
                        ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
                    )) if checkpoint == seam
                ),
                "deadline seam {seam} must refuse; got {result:?}"
            );
            assert!(reads.get() > 0);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn randomized_finite_dyadics_preserve_every_sampled_half_witness(
            center_seed in -10_000_i32..=10_000,
            generator_seed in (-10_000_i32..=10_000).prop_filter(
                "the canonical sparse generator coefficient must be nonzero",
                |value| *value != 0,
            ),
            constraint_seed in -10_000_i32..=10_000,
            remainder_seed in 0_u16..=1_000,
        ) {
            let center = f64::from(center_seed) / 10.0;
            let generator = f64::from(generator_seed) / 10.0;
            let constraint = f64::from(constraint_seed) / 10.0;
            let rhs = constraint.abs() + 1.0;
            let remainder = f64::from(remainder_seed) / 100.0;
            let input = ConstrainedZonotope64::try_new(
                vec![center],
                vec![vec![(0, generator)]],
                array![[constraint]],
                vec![rhs],
                vec![remainder],
            )
            .unwrap();
            let (children, _) =
                bisect_constrained_zonotope_protected_alpha_unwired(&input, 0, limits())
                    .unwrap();

            for (sigma, child, numerators) in [
                (-1_i32, children.negative(), [-2_i32, -1, 0]),
                (1_i32, children.positive(), [0_i32, 1, 2]),
            ] {
                for numerator in numerators {
                    let original = vec![BigRational::new(numerator.into(), 2.into())];
                    let beta = mapped_beta(&original, 0, sigma);
                    assert_mapped_witness_contained(&input, child, &original, &beta);
                }
            }
        }
    }
}
