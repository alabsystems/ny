// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proof-safe, outward binary64 interval propagation for unwired diagnostics.
//!
//! This deliberately small Box domain exists to measure an independent source
//! of intermediate bounds alongside the constrained-zonotope experiments. It
//! is not reachable from a verifier command or verdict. Every multiplication
//! and reduction is enclosed on the CPU with adjacent finite `f64` endpoints;
//! CUDA is neither used nor trusted. Callers must still establish the
//! provenance of input endpoints and network parameters.
//!
//! NY's existing graph-level f64 cell evaluator targets near-point leaf
//! escalation and exposes only the terminal tensor. This primitive is separate
//! because the experiment needs caller-priced per-layer work, intermediate
//! pre-ReLU boxes, and raw-parameter provenance checks at the CLI bridge.

use ndarray::{Array2, ArrayView2, ArrayView4};

use crate::constrained_zonotope64::ConstrainedZonotope64CallGateError;
use crate::constrained_zonotope_call_budget::{
    ConstrainedZonotopeCallBudget, ConstrainedZonotopeCallBudgetError, ConstrainedZonotopeCallGate,
    ConstrainedZonotopeCallOutcome, ConstrainedZonotopeCallTracker,
    ConstrainedZonotopePeakLiveBytes,
};
use crate::{ConstrainedZonotope64, ConstrainedZonotopeConv2dSpec};

/// Absolute ceilings for this diagnostic-only domain.
pub const BOX64_HARD_MAX_VALUES: usize = 2_000_000;
/// Maximum persistent lower/upper endpoints.
pub const BOX64_HARD_MAX_STORED_F64: usize = 4_000_000;
/// Maximum parameter values inspected by one transform.
pub const BOX64_HARD_MAX_WEIGHT_ELEMENTS: usize = 100_000_000;
/// Maximum logical contraction visits in one transform.
pub const BOX64_HARD_MAX_WORK_ITEMS: usize = 2_000_000_000;
/// Maximum scalar products enclosed in one transform.
pub const BOX64_HARD_MAX_SCALAR_PRODUCTS: usize = 2_000_000_000;

/// Explicit caller-selected firewall for every Box operation.
///
/// There is intentionally no `Default`. The operation rejects both work that
/// exceeds a selected cap and selected caps above the absolute hard ceilings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CertifiedBox64Limits {
    /// Maximum input or output value dimension.
    pub max_values: usize,
    /// Maximum lower plus upper endpoints retained by an output.
    pub max_stored_f64: usize,
    /// Maximum logical weight elements inspected for finiteness.
    pub max_weight_elements: usize,
    /// Maximum matrix/kernel/generator visits, including structural zeros.
    pub max_work_items: usize,
    /// Maximum endpoint multiplications actually enclosed.
    pub max_scalar_products: usize,
}

/// One finite, certified outer axis-aligned box.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedBox64 {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

/// Checked accounting for a Box affine transform.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CertifiedBoxAffinePlan {
    /// Input scalar count.
    pub input_values: usize,
    /// Output scalar count.
    pub output_values: usize,
    /// Logical matrix elements visited.
    pub matrix_visits: usize,
    /// Nonzero-weight endpoint products enclosed.
    pub scalar_products: usize,
}

/// Checked accounting for a Box convolution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CertifiedBoxConv2dPlan {
    /// Input shape in unbatched NCHW order.
    pub input_shape: [usize; 3],
    /// Output shape in unbatched NCHW order.
    pub output_shape: [usize; 3],
    /// Kernel shape `[out, in_per_group, kh, kw]`.
    pub weight_shape: [usize; 4],
    /// Logical kernel visits, including padding and zeros.
    pub kernel_visits: usize,
    /// Nonzero in-bounds endpoint products enclosed.
    pub scalar_products: usize,
}

/// Checked accounting for an unconstrained CZ axis hull.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CertifiedBoxHullPlan {
    /// Value dimension of the source and output.
    pub value_dim: usize,
    /// Sparse generator columns visited, including empty columns.
    pub generator_columns: usize,
    /// Sparse generator coefficients accumulated into coordinate radii.
    pub generator_nonzeros: usize,
    /// Combined generator-column plus coefficient visits.
    pub generator_work_items: usize,
    /// Existing predicates deliberately ignored by this hull.
    pub ignored_constraints: usize,
    /// Directed additions used for generator radii.
    pub radius_additions: usize,
}

/// Invalid data, exhausted resources, or arithmetic with no finite enclosure.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CertifiedBox64Error {
    /// Parallel inputs or parameters disagree in shape.
    #[error("shape mismatch for {field}: expected {expected:?}, got {got:?}")]
    Shape {
        /// Rejected input.
        field: &'static str,
        /// Required shape.
        expected: Vec<usize>,
        /// Supplied shape.
        got: Vec<usize>,
    },
    /// A convolution or limit is structurally invalid.
    #[error("invalid certified Box specification: {message}")]
    InvalidSpec {
        /// Concrete rejected precondition.
        message: String,
    },
    /// An endpoint or parameter is not finite.
    #[error("{field}[{index}] must be finite")]
    NonFinite {
        /// Rejected array.
        field: &'static str,
        /// Flattened index.
        index: usize,
    },
    /// A lower endpoint exceeds its upper endpoint.
    #[error("lower[{index}] exceeds upper[{index}]")]
    ReversedBounds {
        /// Rejected coordinate.
        index: usize,
    },
    /// Checked resource arithmetic overflowed.
    #[error("resource size overflow while computing {operation}")]
    ResourceOverflow {
        /// Failed calculation.
        operation: &'static str,
    },
    /// A caller-selected or absolute cap was exceeded.
    #[error("resource limit exceeded for {resource}: required {required}, limit {limit}")]
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Required count.
        required: usize,
        /// Effective cap.
        limit: usize,
    },
    /// A bounded allocation failed.
    #[error("unable to reserve storage for {resource}")]
    AllocationFailure {
        /// Requested buffer.
        resource: &'static str,
    },
    /// Outward widening reached infinity or NaN.
    #[error("non-finite outward arithmetic while computing {operation}")]
    NonFiniteArithmetic {
        /// Failed operation.
        operation: &'static str,
    },
    /// The host does not preserve binary64 subnormals.
    #[error("unsupported floating-point environment: {requirement}")]
    UnsupportedFloatingPoint {
        /// Required IEEE behavior.
        requirement: &'static str,
    },
}

/// Typed refusal from a budgeted remainder-only CZ/Box bridge.
///
/// These bridges are deliberately narrower than
/// [`unconstrained_zonotope_box_unwired`]. They preserve an exact structural
/// boundary: the source or result CZ has no alpha symbols and no predicate
/// rows, so all uncertainty is carried by its independent box remainder.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CertifiedBox64BridgeError {
    /// A remainder-only CZ cannot carry alpha symbols, including empty sparse
    /// generator columns.
    #[error("remainder-only CZ bridge requires alpha_dim == 0, got {alpha_dim}")]
    NonzeroAlphaDimension {
        /// Rejected source alpha dimension.
        alpha_dim: usize,
    },

    /// Even a zero-width predicate row can make a zero-symbol domain empty,
    /// so predicate rows are never ignored by this bridge.
    #[error("remainder-only CZ bridge requires constraint_count == 0, got {constraint_count}")]
    NonzeroPredicateCount {
        /// Rejected source predicate-row count.
        constraint_count: usize,
    },

    /// Box invariants, caller limits, or outward arithmetic failed closed.
    #[error(transparent)]
    Box(#[from] CertifiedBox64Error),

    /// The validated zero-symbol result could not be materialized.
    #[error(transparent)]
    Domain(#[from] crate::ConstrainedZonotope64Error),

    /// The caller's deadline or aggregate peak-memory ceiling refused work.
    #[error(transparent)]
    Budget(#[from] ConstrainedZonotopeCallBudgetError),
}

impl CertifiedBox64 {
    /// Copy caller-certified outer endpoints into a bounded Box domain.
    pub fn from_certified_bounds(
        lower: &[f64],
        upper: &[f64],
        limits: CertifiedBox64Limits,
    ) -> Result<Self, CertifiedBox64Error> {
        validate_limits(limits)?;
        if upper.len() != lower.len() {
            return Err(CertifiedBox64Error::Shape {
                field: "upper bounds",
                expected: vec![lower.len()],
                got: vec![upper.len()],
            });
        }
        check_output_storage(lower.len(), limits)?;
        validate_finite("lower", lower.iter().copied())?;
        validate_finite("upper", upper.iter().copied())?;
        for (index, (&lo, &hi)) in lower.iter().zip(upper).enumerate() {
            if lo > hi {
                return Err(CertifiedBox64Error::ReversedBounds { index });
            }
        }
        Ok(Self {
            lower: clone_bounded(lower, "Box lower endpoints")?,
            upper: clone_bounded(upper, "Box upper endpoints")?,
        })
    }

    /// Number of scalar coordinates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lower.len()
    }

    /// Whether the box is empty-dimensional.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lower.is_empty()
    }

    /// Certified lower endpoints.
    #[must_use]
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }

    /// Certified upper endpoints.
    #[must_use]
    pub fn upper(&self) -> &[f64] {
        &self.upper
    }

    /// Apply elementwise ReLU exactly to the interval endpoints.
    pub fn relu_unwired(&self, limits: CertifiedBox64Limits) -> Result<Self, CertifiedBox64Error> {
        validate_limits(limits)?;
        check_output_storage(self.len(), limits)?;
        check_limit("ReLU coordinate visits", self.len(), limits.max_work_items)?;
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        try_reserve(&mut lower, self.len(), "ReLU lower endpoints")?;
        try_reserve(&mut upper, self.len(), "ReLU upper endpoints")?;
        for (&lo, &hi) in self.lower.iter().zip(&self.upper) {
            // ReLU is monotone and max with exactly representable zero needs no
            // floating-point arithmetic or extra widening.
            lower.push(lo.max(0.0));
            upper.push(hi.max(0.0));
        }
        Ok(Self { lower, upper })
    }
}

/// Convert a remainder-only constrained zonotope into its certified axis box
/// behind the shared synchronous call firewall.
///
/// The source must have exactly zero alpha symbols and zero predicate rows.
/// Consequently its represented set is the Cartesian product
/// `center[i] +/- box_remainder[i]`; each endpoint is rounded outward here.
/// Empty value dimensions remain valid empty-dimensional boxes. The caller's
/// budget baseline must include the borrowed source and other retained data.
pub fn certified_box_from_remainder_only_zonotope_unwired_with_budget(
    input: &ConstrainedZonotope64,
    limits: CertifiedBox64Limits,
    budget: ConstrainedZonotopeCallBudget,
) -> Result<ConstrainedZonotopeCallOutcome<CertifiedBox64>, CertifiedBox64BridgeError> {
    let mut gate = ConstrainedZonotopeCallTracker::from_system_clock(budget)?;
    let value = certified_box_from_remainder_only_zonotope_impl(input, limits, &mut gate)?;
    Ok(ConstrainedZonotopeCallOutcome::new(value, gate.report()))
}

/// Recenter a certified axis box into a zero-symbol, zero-predicate
/// constrained zonotope behind the shared synchronous call firewall.
///
/// Midpoints are nominal binary64 values. Each symmetric radius is widened
/// upward enough to contain both exact-dyadic source endpoints. The conversion
/// therefore preserves the complete source box but may contain additional
/// points when the source interval is asymmetric around its stored midpoint.
pub fn remainder_only_zonotope_from_certified_box_unwired_with_budget(
    input: &CertifiedBox64,
    limits: CertifiedBox64Limits,
    budget: ConstrainedZonotopeCallBudget,
) -> Result<ConstrainedZonotopeCallOutcome<ConstrainedZonotope64>, CertifiedBox64BridgeError> {
    let mut gate = ConstrainedZonotopeCallTracker::from_system_clock(budget)?;
    let value = remainder_only_zonotope_from_certified_box_impl(
        input,
        limits,
        BoxRecenterOperation::Identity,
        &mut gate,
    )?;
    Ok(ConstrainedZonotopeCallOutcome::new(value, gate.report()))
}

/// Apply exact interval ReLU endpoints and recenter the result as a
/// zero-symbol, zero-predicate constrained zonotope.
///
/// ReLU is monotone, so `[lower.max(0), upper.max(0)]` is the exact interval
/// image. Only the subsequent symmetric binary64 recentering is outward. This
/// fused helper keeps validation, allocation, numeric work, result
/// materialization, and publication under one caller-owned budget.
pub fn certified_box_relu_recenter_unwired_with_budget(
    input: &CertifiedBox64,
    limits: CertifiedBox64Limits,
    budget: ConstrainedZonotopeCallBudget,
) -> Result<ConstrainedZonotopeCallOutcome<ConstrainedZonotope64>, CertifiedBox64BridgeError> {
    let mut gate = ConstrainedZonotopeCallTracker::from_system_clock(budget)?;
    let value = remainder_only_zonotope_from_certified_box_impl(
        input,
        limits,
        BoxRecenterOperation::Relu,
        &mut gate,
    )?;
    Ok(ConstrainedZonotopeCallOutcome::new(value, gate.report()))
}

#[cfg(test)]
fn certified_box_from_remainder_only_zonotope_unwired_with_clock<N>(
    input: &ConstrainedZonotope64,
    limits: CertifiedBox64Limits,
    budget: ConstrainedZonotopeCallBudget,
    now: N,
) -> Result<ConstrainedZonotopeCallOutcome<CertifiedBox64>, CertifiedBox64BridgeError>
where
    N: FnMut(&'static str) -> std::time::Instant,
{
    let mut gate = ConstrainedZonotopeCallTracker::with_clock(budget, now)?;
    let value = certified_box_from_remainder_only_zonotope_impl(input, limits, &mut gate)?;
    Ok(ConstrainedZonotopeCallOutcome::new(value, gate.report()))
}

#[cfg(test)]
fn certified_box_relu_recenter_unwired_with_clock<N>(
    input: &CertifiedBox64,
    limits: CertifiedBox64Limits,
    budget: ConstrainedZonotopeCallBudget,
    now: N,
) -> Result<ConstrainedZonotopeCallOutcome<ConstrainedZonotope64>, CertifiedBox64BridgeError>
where
    N: FnMut(&'static str) -> std::time::Instant,
{
    let mut gate = ConstrainedZonotopeCallTracker::with_clock(budget, now)?;
    let value = remainder_only_zonotope_from_certified_box_impl(
        input,
        limits,
        BoxRecenterOperation::Relu,
        &mut gate,
    )?;
    Ok(ConstrainedZonotopeCallOutcome::new(value, gate.report()))
}

fn certified_box_from_remainder_only_zonotope_impl<G>(
    input: &ConstrainedZonotope64,
    limits: CertifiedBox64Limits,
    gate: &mut G,
) -> Result<CertifiedBox64, CertifiedBox64BridgeError>
where
    G: ConstrainedZonotopeCallGate,
{
    require_gradual_underflow()?;
    gate.checkpoint("remainder-only CZ-to-Box floating-point preflight")?;
    validate_limits(limits)?;
    require_remainder_only_zonotope(input)?;

    let value_dim = input.value_dim();
    check_output_storage(value_dim, limits)?;
    check_limit(
        "remainder-only CZ-to-Box coordinate visits",
        value_dim,
        limits.max_work_items,
    )?;
    gate.checkpoint("remainder-only CZ-to-Box geometry validation complete")?;
    gate.preflight_peak_live_bytes(box_bridge_output_bytes(value_dim)?)?;
    gate.checkpoint("remainder-only CZ-to-Box peak-memory preflight complete")?;

    let mut lower = Vec::new();
    let mut upper = Vec::new();
    gate.checkpoint("remainder-only CZ-to-Box lower allocation")?;
    try_reserve(&mut lower, value_dim, "remainder-only CZ lower endpoints")?;
    gate.checkpoint("remainder-only CZ-to-Box upper allocation")?;
    try_reserve(&mut upper, value_dim, "remainder-only CZ upper endpoints")?;

    for coordinate in 0..value_dim {
        gate.charge_items(1, "remainder-only CZ-to-Box coordinate transform")?;
        let center = input.center()[coordinate];
        let radius = input.box_remainder()[coordinate];
        if !center.is_finite() {
            return Err(CertifiedBox64Error::NonFinite {
                field: "CZ center",
                index: coordinate,
            }
            .into());
        }
        if !radius.is_finite() {
            return Err(CertifiedBox64Error::NonFinite {
                field: "CZ remainder",
                index: coordinate,
            }
            .into());
        }
        if radius < 0.0 {
            return Err(CertifiedBox64Error::InvalidSpec {
                message: format!("CZ remainder[{coordinate}] must be nonnegative"),
            }
            .into());
        }
        if radius == 0.0 {
            lower.push(center);
            upper.push(center);
        } else {
            lower.push(sub_down(
                center,
                radius,
                "remainder-only CZ lower endpoint",
            )?);
            upper.push(add_up(center, radius, "remainder-only CZ upper endpoint")?);
        }
    }
    gate.checkpoint("remainder-only CZ-to-Box numeric transform complete")?;

    let output = CertifiedBox64 { lower, upper };
    debug_assert_eq!(output.len(), value_dim);
    debug_assert!(output
        .lower()
        .iter()
        .zip(output.upper())
        .all(|(&lower, &upper)| lower <= upper));
    gate.checkpoint("remainder-only CZ-to-Box publication")?;
    Ok(output)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BoxRecenterOperation {
    Identity,
    Relu,
}

fn remainder_only_zonotope_from_certified_box_impl<G>(
    input: &CertifiedBox64,
    limits: CertifiedBox64Limits,
    operation: BoxRecenterOperation,
    gate: &mut G,
) -> Result<ConstrainedZonotope64, CertifiedBox64BridgeError>
where
    G: ConstrainedZonotopeCallGate,
{
    require_gradual_underflow()?;
    gate.checkpoint("certified Box-to-CZ floating-point preflight")?;
    validate_limits(limits)?;
    let value_dim = input.len();
    if input.upper().len() != value_dim {
        return Err(CertifiedBox64Error::Shape {
            field: "Box upper endpoints",
            expected: vec![value_dim],
            got: vec![input.upper().len()],
        }
        .into());
    }
    check_output_storage(value_dim, limits)?;
    // The fused transform charges one numeric recenter visit, then the shared
    // CZ constructor charges center finiteness, remainder finiteness, and
    // remainder-sign validation for every coordinate. Price that complete
    // deterministic plan before allocating either output vector.
    let charged_work = value_dim
        .checked_mul(4)
        .ok_or(CertifiedBox64Error::ResourceOverflow {
            operation: "certified Box-to-CZ charged work items",
        })?;
    check_limit(
        "certified Box-to-CZ charged work items",
        charged_work,
        limits.max_work_items,
    )?;
    gate.checkpoint("certified Box-to-CZ geometry validation complete")?;
    gate.preflight_peak_live_bytes(box_bridge_output_bytes(value_dim)?)?;
    gate.checkpoint("certified Box-to-CZ peak-memory preflight complete")?;

    let mut center = Vec::new();
    let mut box_remainder = Vec::new();
    gate.checkpoint("certified Box-to-CZ center allocation")?;
    try_reserve(&mut center, value_dim, "recentered CZ centers")?;
    gate.checkpoint("certified Box-to-CZ remainder allocation")?;
    try_reserve(
        &mut box_remainder,
        value_dim,
        "recentered CZ box remainders",
    )?;

    for coordinate in 0..value_dim {
        gate.charge_items(1, "certified Box-to-CZ coordinate transform")?;
        let mut lower = input.lower()[coordinate];
        let mut upper = input.upper()[coordinate];
        validate_box_coordinate(lower, upper, coordinate)?;
        if operation == BoxRecenterOperation::Relu {
            // ReLU is monotone and max with exactly representable zero needs
            // no floating-point arithmetic or endpoint widening.
            lower = lower.max(0.0);
            upper = upper.max(0.0);
        }
        let (coordinate_center, coordinate_remainder) = recenter_outward(lower, upper)?;
        center.push(coordinate_center);
        box_remainder.push(coordinate_remainder);
    }
    gate.checkpoint("certified Box-to-CZ numeric transform complete")?;

    gate.checkpoint("certified Box-to-CZ domain materialization")?;
    let constraints = Array2::from_shape_vec((0, 0), Vec::new()).map_err(|_| {
        CertifiedBox64Error::ResourceOverflow {
            operation: "empty remainder-only CZ constraint matrix shape",
        }
    })?;
    let output = ConstrainedZonotope64::try_new_with_call_gate(
        center,
        Vec::new(),
        constraints,
        Vec::new(),
        box_remainder,
        gate,
    )
    .map_err(map_remainder_only_domain_error)?;
    gate.checkpoint("certified Box-to-CZ domain materialization complete")?;
    debug_assert_eq!(output.alpha_dim(), 0);
    debug_assert_eq!(output.constraint_count(), 0);
    gate.checkpoint("certified Box-to-CZ publication")?;
    Ok(output)
}

fn require_remainder_only_zonotope(
    input: &ConstrainedZonotope64,
) -> Result<(), CertifiedBox64BridgeError> {
    if input.box_remainder().len() != input.value_dim() {
        return Err(CertifiedBox64Error::Shape {
            field: "CZ box remainder",
            expected: vec![input.value_dim()],
            got: vec![input.box_remainder().len()],
        }
        .into());
    }
    if input.alpha_dim() != 0 {
        return Err(CertifiedBox64BridgeError::NonzeroAlphaDimension {
            alpha_dim: input.alpha_dim(),
        });
    }
    if input.constraint_count() != 0 {
        return Err(CertifiedBox64BridgeError::NonzeroPredicateCount {
            constraint_count: input.constraint_count(),
        });
    }
    Ok(())
}

fn validate_box_coordinate(
    lower: f64,
    upper: f64,
    coordinate: usize,
) -> Result<(), CertifiedBox64Error> {
    if !lower.is_finite() {
        return Err(CertifiedBox64Error::NonFinite {
            field: "Box lower",
            index: coordinate,
        });
    }
    if !upper.is_finite() {
        return Err(CertifiedBox64Error::NonFinite {
            field: "Box upper",
            index: coordinate,
        });
    }
    if lower > upper {
        return Err(CertifiedBox64Error::ReversedBounds { index: coordinate });
    }
    Ok(())
}

fn recenter_outward(lower: f64, upper: f64) -> Result<(f64, f64), CertifiedBox64Error> {
    debug_assert!(lower.is_finite() && upper.is_finite() && lower <= upper);
    let center = f64::midpoint(lower, upper);
    if !center.is_finite() || center < lower || center > upper {
        return Err(CertifiedBox64Error::NonFiniteArithmetic {
            operation: "certified Box midpoint",
        });
    }
    let lower_radius =
        nonnegative_difference_up(center, lower, "certified Box recenter lower half-width")?;
    let upper_radius =
        nonnegative_difference_up(upper, center, "certified Box recenter upper half-width")?;
    let radius = lower_radius.max(upper_radius);
    if !radius.is_finite() || radius < 0.0 {
        return Err(CertifiedBox64Error::NonFiniteArithmetic {
            operation: "certified Box recenter radius",
        });
    }
    Ok((center, radius))
}

/// Subtract two ordered finite binary64 values, widening only when exactness
/// is not established. Subtraction by zero is exact. Sterbenz's lemma makes
/// same-sign operands within a factor of two exact as well; retaining that
/// case is essential for `[0, u]` ReLU boxes to remain nonnegative after
/// symmetric recentering.
fn nonnegative_difference_up(
    minuend: f64,
    subtrahend: f64,
    operation: &'static str,
) -> Result<f64, CertifiedBox64Error> {
    debug_assert!(minuend.is_finite() && subtrahend.is_finite());
    debug_assert!(minuend >= subtrahend);
    let difference = minuend - subtrahend;
    if !difference.is_finite() || difference < 0.0 {
        return Err(CertifiedBox64Error::NonFiniteArithmetic { operation });
    }
    if difference == 0.0 || minuend == 0.0 || subtrahend == 0.0 {
        return Ok(difference);
    }
    let same_sign = minuend.is_sign_positive() == subtrahend.is_sign_positive();
    let smaller_magnitude = minuend.abs().min(subtrahend.abs());
    let larger_magnitude = minuend.abs().max(subtrahend.abs());
    if same_sign && larger_magnitude / 2.0 <= smaller_magnitude {
        return Ok(difference);
    }
    round_up(difference, operation)
}

fn box_bridge_output_bytes(value_dim: usize) -> Result<usize, CertifiedBox64Error> {
    let stored = value_dim
        .checked_mul(2)
        .ok_or(CertifiedBox64Error::ResourceOverflow {
            operation: "remainder-only bridge stored f64 count",
        })?;
    let mut peak = ConstrainedZonotopePeakLiveBytes::new();
    peak.add_elements::<f64>(stored, "remainder-only bridge output storage")
        .map_err(|error| match error {
            ConstrainedZonotopeCallBudgetError::ResourceOverflow { operation } => {
                CertifiedBox64Error::ResourceOverflow { operation }
            }
            ConstrainedZonotopeCallBudgetError::DeadlineExpired { .. }
            | ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { .. } => {
                unreachable!("peak-size construction performs no deadline or cap checks")
            }
        })?;
    Ok(peak.finish())
}

fn map_remainder_only_domain_error(
    error: ConstrainedZonotope64CallGateError,
) -> CertifiedBox64BridgeError {
    match error {
        ConstrainedZonotope64CallGateError::Domain(error) => error.into(),
        ConstrainedZonotope64CallGateError::Budget(error) => error.into(),
    }
}

/// Propagate one certified Box through a dense affine map.
pub fn certified_box_affine_unwired(
    input: &CertifiedBox64,
    weights: ArrayView2<'_, f64>,
    bias: &[f64],
    limits: CertifiedBox64Limits,
) -> Result<(CertifiedBox64, CertifiedBoxAffinePlan), CertifiedBox64Error> {
    require_gradual_underflow()?;
    validate_limits(limits)?;
    let input_values = input.len();
    let output_values = weights.nrows();
    if input_values == 0 || output_values == 0 {
        return Err(CertifiedBox64Error::InvalidSpec {
            message: format!(
                "affine input/output dimensions must be nonzero, got {input_values}/{output_values}"
            ),
        });
    }
    if weights.ncols() != input_values {
        return Err(CertifiedBox64Error::Shape {
            field: "affine weights",
            expected: vec![output_values, input_values],
            got: weights.shape().to_vec(),
        });
    }
    if bias.len() != output_values {
        return Err(CertifiedBox64Error::Shape {
            field: "affine bias",
            expected: vec![output_values],
            got: vec![bias.len()],
        });
    }
    check_limit("input value count", input_values, limits.max_values)?;
    check_output_storage(output_values, limits)?;
    let weight_elements =
        output_values
            .checked_mul(input_values)
            .ok_or(CertifiedBox64Error::ResourceOverflow {
                operation: "affine weight elements",
            })?;
    check_limit(
        "weight elements",
        weight_elements,
        limits.max_weight_elements,
    )?;
    check_limit("matrix visits", weight_elements, limits.max_work_items)?;
    validate_finite("affine weights", weights.iter().copied())?;
    validate_finite("affine bias", bias.iter().copied())?;

    let mut lower = Vec::new();
    let mut upper = Vec::new();
    try_reserve(&mut lower, output_values, "affine lower endpoints")?;
    try_reserve(&mut upper, output_values, "affine upper endpoints")?;
    let mut scalar_products = 0_usize;
    for output in 0..output_values {
        let mut sum = OutwardInterval::exact(bias[output]);
        for input_index in 0..input_values {
            let weight = weights[[output, input_index]];
            if weight == 0.0 {
                continue;
            }
            consume_products(&mut scalar_products, 2, limits.max_scalar_products)?;
            sum = sum.add(OutwardInterval::scale(
                weight,
                input.lower[input_index],
                input.upper[input_index],
            )?)?;
        }
        lower.push(sum.lo);
        upper.push(sum.hi);
    }
    Ok((
        CertifiedBox64 { lower, upper },
        CertifiedBoxAffinePlan {
            input_values,
            output_values,
            matrix_visits: weight_elements,
            scalar_products,
        },
    ))
}

/// Propagate one certified Box through grouped NCHW Conv2d.
pub fn certified_box_conv2d_unwired(
    input: &CertifiedBox64,
    input_shape: [usize; 3],
    weights: ArrayView4<'_, f64>,
    bias: &[f64],
    spec: ConstrainedZonotopeConv2dSpec,
    limits: CertifiedBox64Limits,
) -> Result<(CertifiedBox64, CertifiedBoxConv2dPlan), CertifiedBox64Error> {
    require_gradual_underflow()?;
    validate_limits(limits)?;
    if input_shape.contains(&0) {
        return Err(CertifiedBox64Error::InvalidSpec {
            message: format!("convolution input shape must be positive, got {input_shape:?}"),
        });
    }
    let input_values = checked_product(&input_shape, "convolution input values")?;
    if input.len() != input_values {
        return Err(CertifiedBox64Error::Shape {
            field: "convolution input",
            expected: vec![input_values],
            got: vec![input.len()],
        });
    }
    check_limit("input value count", input_values, limits.max_values)?;
    if spec.groups == 0 || spec.stride.contains(&0) || spec.dilation.contains(&0) {
        return Err(CertifiedBox64Error::InvalidSpec {
            message: "groups, strides, and dilations must be positive".to_string(),
        });
    }
    let weight_shape: [usize; 4] =
        weights
            .shape()
            .try_into()
            .map_err(|_| CertifiedBox64Error::Shape {
                field: "convolution weights",
                expected: vec![4],
                got: vec![weights.ndim()],
            })?;
    let [input_channels, input_height, input_width] = input_shape;
    let [output_channels, kernel_input_channels, kernel_height, kernel_width] = weight_shape;
    if output_channels == 0 {
        return Err(CertifiedBox64Error::InvalidSpec {
            message: "convolution output channel count must be positive".to_string(),
        });
    }
    if input_channels % spec.groups != 0 || output_channels % spec.groups != 0 {
        return Err(CertifiedBox64Error::InvalidSpec {
            message: format!(
                "channels {input_channels}/{output_channels} must be divisible by groups {}",
                spec.groups
            ),
        });
    }
    let expected_kernel_inputs = input_channels / spec.groups;
    if kernel_input_channels != expected_kernel_inputs {
        return Err(CertifiedBox64Error::Shape {
            field: "convolution kernel input channels",
            expected: vec![expected_kernel_inputs],
            got: vec![kernel_input_channels],
        });
    }
    if bias.len() != output_channels {
        return Err(CertifiedBox64Error::Shape {
            field: "convolution bias",
            expected: vec![output_channels],
            got: vec![bias.len()],
        });
    }
    let output_height = output_extent(
        input_height,
        kernel_height,
        spec.dilation[0],
        spec.padding[0],
        spec.padding[2],
        spec.stride[0],
    )?;
    let output_width = output_extent(
        input_width,
        kernel_width,
        spec.dilation[1],
        spec.padding[1],
        spec.padding[3],
        spec.stride[1],
    )?;
    let output_shape = [output_channels, output_height, output_width];
    let output_values = checked_product(&output_shape, "convolution output values")?;
    check_output_storage(output_values, limits)?;
    let weight_elements = checked_product(&weight_shape, "convolution weight elements")?;
    check_limit(
        "weight elements",
        weight_elements,
        limits.max_weight_elements,
    )?;
    let kernel_visits = output_values
        .checked_mul(kernel_input_channels)
        .and_then(|count| count.checked_mul(kernel_height))
        .and_then(|count| count.checked_mul(kernel_width))
        .ok_or(CertifiedBox64Error::ResourceOverflow {
            operation: "convolution kernel visits",
        })?;
    check_limit("kernel visits", kernel_visits, limits.max_work_items)?;
    validate_finite("convolution weights", weights.iter().copied())?;
    validate_finite("convolution bias", bias.iter().copied())?;

    let mut lower = Vec::new();
    let mut upper = Vec::new();
    try_reserve(&mut lower, output_values, "convolution lower endpoints")?;
    try_reserve(&mut upper, output_values, "convolution upper endpoints")?;
    let input_channels_per_group = input_channels / spec.groups;
    let output_channels_per_group = output_channels / spec.groups;
    let mut scalar_products = 0_usize;
    for output_channel in 0..output_channels {
        let input_channel_base =
            (output_channel / output_channels_per_group) * input_channels_per_group;
        for output_y in 0..output_height {
            for output_x in 0..output_width {
                let mut sum = OutwardInterval::exact(bias[output_channel]);
                for kernel_input_channel in 0..kernel_input_channels {
                    let input_channel = input_channel_base + kernel_input_channel;
                    for kernel_y in 0..kernel_height {
                        let Some(input_y) = input_coordinate(
                            output_y,
                            spec.stride[0],
                            kernel_y,
                            spec.dilation[0],
                            spec.padding[0],
                            input_height,
                        )?
                        else {
                            continue;
                        };
                        for kernel_x in 0..kernel_width {
                            let Some(input_x) = input_coordinate(
                                output_x,
                                spec.stride[1],
                                kernel_x,
                                spec.dilation[1],
                                spec.padding[1],
                                input_width,
                            )?
                            else {
                                continue;
                            };
                            let weight =
                                weights[[output_channel, kernel_input_channel, kernel_y, kernel_x]];
                            if weight == 0.0 {
                                continue;
                            }
                            consume_products(&mut scalar_products, 2, limits.max_scalar_products)?;
                            let input_index =
                                (input_channel * input_height + input_y) * input_width + input_x;
                            sum = sum.add(OutwardInterval::scale(
                                weight,
                                input.lower[input_index],
                                input.upper[input_index],
                            )?)?;
                        }
                    }
                }
                lower.push(sum.lo);
                upper.push(sum.hi);
            }
        }
    }
    Ok((
        CertifiedBox64 { lower, upper },
        CertifiedBoxConv2dPlan {
            input_shape,
            output_shape,
            weight_shape,
            kernel_visits,
            scalar_products,
        },
    ))
}

/// Outward axis hull of a constrained zonotope while deliberately ignoring
/// `C alpha <= d`.
///
/// This mirrors the radius used by the current ReLU classifier: each stored
/// generator contributes its absolute coefficient and the independent Box
/// remainder is charged once. Predicate rows are counted but not consumed.
pub fn unconstrained_zonotope_box_unwired(
    input: &ConstrainedZonotope64,
    limits: CertifiedBox64Limits,
) -> Result<(CertifiedBox64, CertifiedBoxHullPlan), CertifiedBox64Error> {
    require_gradual_underflow()?;
    validate_limits(limits)?;
    let value_dim = input.value_dim();
    check_output_storage(value_dim, limits)?;
    validate_finite("CZ center", input.center().iter().copied())?;
    validate_finite("CZ remainder", input.box_remainder().iter().copied())?;

    // Price every sparse-column visit before walking the column list. Empty
    // columns are real work and must not bypass `max_work_items`. Once that
    // outer scan is bounded, inspect only O(1) `nnz` metadata and reject the
    // complete `alpha_dim + nnz` plan before traversing coefficients.
    let generator_columns = input.alpha_dim();
    check_limit(
        "CZ generator column visits",
        generator_columns,
        limits.max_work_items,
    )?;
    let generator_nonzeros = input.generators().iter().try_fold(
        0_usize,
        |count, generator| -> Result<_, CertifiedBox64Error> {
            count
                .checked_add(generator.nnz())
                .ok_or(CertifiedBox64Error::ResourceOverflow {
                    operation: "CZ generator nonzeros",
                })
        },
    )?;
    let generator_work_items = generator_columns.checked_add(generator_nonzeros).ok_or(
        CertifiedBox64Error::ResourceOverflow {
            operation: "CZ generator column plus coefficient visits",
        },
    )?;
    check_limit(
        "CZ generator visits",
        generator_work_items,
        limits.max_work_items,
    )?;

    let mut radius = clone_bounded(input.box_remainder(), "CZ radius scratch")?;
    let mut visited_nonzeros = 0_usize;
    for generator in input.generators() {
        for (coordinate, coefficient) in generator.entries() {
            if !coefficient.is_finite() {
                return Err(CertifiedBox64Error::NonFinite {
                    field: "CZ generators",
                    index: visited_nonzeros,
                });
            }
            radius[coordinate] = add_up(radius[coordinate], coefficient.abs(), "CZ radius sum")?;
            visited_nonzeros += 1;
        }
    }
    debug_assert_eq!(visited_nonzeros, generator_nonzeros);
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    try_reserve(&mut lower, value_dim, "CZ hull lower endpoints")?;
    try_reserve(&mut upper, value_dim, "CZ hull upper endpoints")?;
    for (coordinate, (&center, &radius)) in input.center().iter().zip(&radius).enumerate() {
        if radius < 0.0 || !radius.is_finite() {
            return Err(CertifiedBox64Error::NonFiniteArithmetic {
                operation: "CZ coordinate radius",
            });
        }
        if radius == 0.0 {
            lower.push(center);
            upper.push(center);
        } else {
            lower.push(sub_down(center, radius, "CZ hull lower")?);
            upper.push(add_up(center, radius, "CZ hull upper")?);
        }
        debug_assert!(lower[coordinate] <= upper[coordinate]);
    }
    Ok((
        CertifiedBox64 { lower, upper },
        CertifiedBoxHullPlan {
            value_dim,
            generator_columns,
            generator_nonzeros,
            generator_work_items,
            ignored_constraints: input.constraint_count(),
            radius_additions: generator_nonzeros,
        },
    ))
}

#[derive(Clone, Copy, Debug)]
struct OutwardInterval {
    lo: f64,
    hi: f64,
}

impl OutwardInterval {
    fn exact(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }

    fn scale(weight: f64, lo: f64, hi: f64) -> Result<Self, CertifiedBox64Error> {
        debug_assert!(weight != 0.0);
        debug_assert!(lo <= hi);
        if lo == 0.0 && hi == 0.0 {
            return Ok(Self::exact(0.0));
        }
        if weight == 1.0 {
            return Ok(Self { lo, hi });
        }
        if weight == -1.0 {
            return Ok(Self { lo: -hi, hi: -lo });
        }
        let (lower_operand, upper_operand) = if weight > 0.0 { (lo, hi) } else { (hi, lo) };
        Ok(Self {
            lo: round_down(
                weight * lower_operand,
                "certified Box interval product lower",
            )?,
            hi: round_up(
                weight * upper_operand,
                "certified Box interval product upper",
            )?,
        })
    }

    fn add(self, rhs: Self) -> Result<Self, CertifiedBox64Error> {
        Ok(Self {
            lo: add_down(self.lo, rhs.lo, "certified Box interval sum lower")?,
            hi: add_up(self.hi, rhs.hi, "certified Box interval sum upper")?,
        })
    }
}

fn validate_limits(limits: CertifiedBox64Limits) -> Result<(), CertifiedBox64Error> {
    for (resource, selected, hard) in [
        (
            "selected maximum values",
            limits.max_values,
            BOX64_HARD_MAX_VALUES,
        ),
        (
            "selected maximum stored f64",
            limits.max_stored_f64,
            BOX64_HARD_MAX_STORED_F64,
        ),
        (
            "selected maximum weight elements",
            limits.max_weight_elements,
            BOX64_HARD_MAX_WEIGHT_ELEMENTS,
        ),
        (
            "selected maximum work items",
            limits.max_work_items,
            BOX64_HARD_MAX_WORK_ITEMS,
        ),
        (
            "selected maximum scalar products",
            limits.max_scalar_products,
            BOX64_HARD_MAX_SCALAR_PRODUCTS,
        ),
    ] {
        check_limit(resource, selected, hard)?;
    }
    Ok(())
}

fn check_output_storage(
    values: usize,
    limits: CertifiedBox64Limits,
) -> Result<(), CertifiedBox64Error> {
    check_limit("value count", values, limits.max_values)?;
    let stored = values
        .checked_mul(2)
        .ok_or(CertifiedBox64Error::ResourceOverflow {
            operation: "Box lower plus upper endpoints",
        })?;
    check_limit("stored f64 endpoints", stored, limits.max_stored_f64)
}

fn output_extent(
    input: usize,
    kernel: usize,
    dilation: usize,
    padding_before: usize,
    padding_after: usize,
    stride: usize,
) -> Result<usize, CertifiedBox64Error> {
    if input == 0 || kernel == 0 {
        return Err(CertifiedBox64Error::InvalidSpec {
            message: "convolution input and kernel extents must be positive".to_string(),
        });
    }
    let effective_kernel = kernel
        .checked_sub(1)
        .and_then(|extent| extent.checked_mul(dilation))
        .and_then(|extent| extent.checked_add(1))
        .ok_or(CertifiedBox64Error::ResourceOverflow {
            operation: "effective kernel extent",
        })?;
    let padded = input
        .checked_add(padding_before)
        .and_then(|extent| extent.checked_add(padding_after))
        .ok_or(CertifiedBox64Error::ResourceOverflow {
            operation: "padded input extent",
        })?;
    if padded < effective_kernel {
        return Err(CertifiedBox64Error::InvalidSpec {
            message: format!(
                "effective kernel extent {effective_kernel} exceeds padded input {padded}"
            ),
        });
    }
    Ok((padded - effective_kernel) / stride + 1)
}

fn input_coordinate(
    output: usize,
    stride: usize,
    kernel: usize,
    dilation: usize,
    padding_before: usize,
    input_extent: usize,
) -> Result<Option<usize>, CertifiedBox64Error> {
    let unpadded = output
        .checked_mul(stride)
        .and_then(|value| {
            kernel
                .checked_mul(dilation)
                .and_then(|offset| value.checked_add(offset))
        })
        .ok_or(CertifiedBox64Error::ResourceOverflow {
            operation: "convolution input coordinate",
        })?;
    let Some(coordinate) = unpadded.checked_sub(padding_before) else {
        return Ok(None);
    };
    Ok((coordinate < input_extent).then_some(coordinate))
}

fn consume_products(
    current: &mut usize,
    additional: usize,
    limit: usize,
) -> Result<(), CertifiedBox64Error> {
    *current = current
        .checked_add(additional)
        .ok_or(CertifiedBox64Error::ResourceOverflow {
            operation: "Box scalar products",
        })?;
    check_limit("scalar products", *current, limit)
}

fn checked_product(
    dimensions: &[usize],
    operation: &'static str,
) -> Result<usize, CertifiedBox64Error> {
    dimensions.iter().try_fold(1_usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .ok_or(CertifiedBox64Error::ResourceOverflow { operation })
    })
}

fn validate_finite(
    field: &'static str,
    values: impl IntoIterator<Item = f64>,
) -> Result<(), CertifiedBox64Error> {
    for (index, value) in values.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(CertifiedBox64Error::NonFinite { field, index });
        }
    }
    Ok(())
}

fn check_limit(
    resource: &'static str,
    required: usize,
    limit: usize,
) -> Result<(), CertifiedBox64Error> {
    if required > limit {
        return Err(CertifiedBox64Error::ResourceLimit {
            resource,
            required,
            limit,
        });
    }
    Ok(())
}

fn clone_bounded<T: Clone>(
    values: &[T],
    resource: &'static str,
) -> Result<Vec<T>, CertifiedBox64Error> {
    let mut output = Vec::new();
    try_reserve(&mut output, values.len(), resource)?;
    output.extend_from_slice(values);
    Ok(output)
}

fn try_reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    resource: &'static str,
) -> Result<(), CertifiedBox64Error> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| CertifiedBox64Error::AllocationFailure { resource })
}

fn round_down(value: f64, operation: &'static str) -> Result<f64, CertifiedBox64Error> {
    if !value.is_finite() {
        return Err(CertifiedBox64Error::NonFiniteArithmetic { operation });
    }
    let outward = value.next_down();
    if !outward.is_finite() {
        return Err(CertifiedBox64Error::NonFiniteArithmetic { operation });
    }
    Ok(outward)
}

fn round_up(value: f64, operation: &'static str) -> Result<f64, CertifiedBox64Error> {
    if !value.is_finite() {
        return Err(CertifiedBox64Error::NonFiniteArithmetic { operation });
    }
    let outward = value.next_up();
    if !outward.is_finite() {
        return Err(CertifiedBox64Error::NonFiniteArithmetic { operation });
    }
    Ok(outward)
}

fn add_down(left: f64, right: f64, operation: &'static str) -> Result<f64, CertifiedBox64Error> {
    if right == 0.0 {
        return Ok(left);
    }
    if left == 0.0 {
        return Ok(right);
    }
    round_down(left + right, operation)
}

fn add_up(left: f64, right: f64, operation: &'static str) -> Result<f64, CertifiedBox64Error> {
    if right == 0.0 {
        return Ok(left);
    }
    if left == 0.0 {
        return Ok(right);
    }
    round_up(left + right, operation)
}

fn sub_down(left: f64, right: f64, operation: &'static str) -> Result<f64, CertifiedBox64Error> {
    if right == 0.0 {
        return Ok(left);
    }
    round_down(left - right, operation)
}

fn require_gradual_underflow() -> Result<(), CertifiedBox64Error> {
    // Prevent constant folding into rustc's abstract IEEE model. These three
    // operations must observe the active scalar environment used below: a
    // normal-to-subnormal product, a subnormal product, and subnormal addition.
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
        return Err(CertifiedBox64Error::UnsupportedFloatingPoint {
            requirement: "IEEE-754 binary64 gradual underflow (FTZ/DAZ disabled)",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::mem::size_of;
    use std::time::{Duration, Instant};

    use ndarray::{array, Array4};
    use num_rational::BigRational;
    use proptest::prelude::*;

    use super::*;

    fn limits() -> CertifiedBox64Limits {
        CertifiedBox64Limits {
            max_values: 128,
            max_stored_f64: 256,
            max_weight_elements: 256,
            max_work_items: 4_096,
            max_scalar_products: 8_192,
        }
    }

    fn budget(
        start: Instant,
        baseline_live_bytes: usize,
        max_peak_live_bytes: usize,
    ) -> ConstrainedZonotopeCallBudget {
        ConstrainedZonotopeCallBudget::new(
            start + Duration::from_mins(1),
            baseline_live_bytes,
            max_peak_live_bytes,
        )
    }

    fn exact(value: f64) -> BigRational {
        BigRational::from_float(value).unwrap()
    }

    fn assert_remainder_coordinate_contains(
        domain: &ConstrainedZonotope64,
        coordinate: usize,
        lower: f64,
        upper: f64,
    ) {
        let center = exact(domain.center()[coordinate]);
        let radius = exact(domain.box_remainder()[coordinate]);
        assert!(center.clone() - radius.clone() <= exact(lower));
        assert!(center + radius >= exact(upper));
    }

    #[test]
    fn runtime_environment_preserves_required_gradual_underflow() {
        require_gradual_underflow().unwrap();
    }

    #[test]
    fn affine_relu_and_grouped_conv_enclose_expected_ranges() {
        let input =
            CertifiedBox64::from_certified_bounds(&[-1.0, 2.0], &[3.0, 4.0], limits()).unwrap();
        let (affine, plan) = certified_box_affine_unwired(
            &input,
            array![[2.0, -3.0], [-1.0, 1.0]].view(),
            &[0.5, -2.0],
            limits(),
        )
        .unwrap();
        assert!(affine.lower()[0] <= -13.5 && affine.upper()[0] >= 0.5);
        assert!(affine.lower()[1] <= -3.0 && affine.upper()[1] >= 3.0);
        assert_eq!(plan.matrix_visits, 4);
        assert_eq!(plan.scalar_products, 8);
        let relu = affine.relu_unwired(limits()).unwrap();
        assert_eq!(relu.lower(), &[0.0, 0.0]);
        assert!(relu.upper()[0] >= 0.5 && relu.upper()[1] >= 3.0);

        let image = CertifiedBox64::from_certified_bounds(
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 2.0, 3.0, 4.0],
            limits(),
        )
        .unwrap();
        let weights = Array4::from_shape_vec((2, 1, 1, 1), vec![2.0, -1.0]).unwrap();
        let (conv, conv_plan) = certified_box_conv2d_unwired(
            &image,
            [2, 1, 2],
            weights.view(),
            &[0.5, 1.0],
            ConstrainedZonotopeConv2dSpec {
                stride: [1, 1],
                padding: [0, 0, 0, 0],
                dilation: [1, 1],
                groups: 2,
            },
            limits(),
        )
        .unwrap();
        for (got_lo, got_hi, exact) in [
            (conv.lower()[0], conv.upper()[0], 2.5),
            (conv.lower()[1], conv.upper()[1], 4.5),
            (conv.lower()[2], conv.upper()[2], -2.0),
            (conv.lower()[3], conv.upper()[3], -3.0),
        ] {
            assert!(got_lo <= exact && got_hi >= exact);
        }
        assert_eq!(conv_plan.output_shape, [2, 1, 2]);
    }

    #[test]
    fn unconstrained_cz_hull_charges_generators_remainder_once_and_ignores_rows() {
        let cz = ConstrainedZonotope64::try_new(
            vec![1.0, -2.0],
            vec![vec![(0, 2.0), (1, -1.0)], vec![(0, -0.5)]],
            Array2::from_shape_vec((1, 2), vec![1.0, -1.0]).unwrap(),
            vec![0.25],
            vec![0.25, 0.5],
        )
        .unwrap();
        let (hull, plan) = unconstrained_zonotope_box_unwired(&cz, limits()).unwrap();
        assert!(hull.lower()[0] <= -1.75 && hull.upper()[0] >= 3.75);
        assert!(hull.lower()[1] <= -3.5 && hull.upper()[1] >= -0.5);
        assert_eq!(plan.value_dim, 2);
        assert_eq!(plan.generator_columns, 2);
        assert_eq!(plan.generator_nonzeros, 3);
        assert_eq!(plan.generator_work_items, 5);
        assert_eq!(plan.radius_additions, 3);
        assert_eq!(plan.ignored_constraints, 1);

        let mut capped = limits();
        capped.max_work_items = 4;
        assert!(matches!(
            unconstrained_zonotope_box_unwired(&cz, capped),
            Err(CertifiedBox64Error::ResourceLimit {
                resource: "CZ generator visits",
                required: 5,
                limit: 4,
            })
        ));
    }

    #[test]
    fn remainder_only_bridges_preserve_exact_dyadic_enclosure_and_accounting() {
        let source = ConstrainedZonotope64::try_new(
            vec![0.1, -2.0, f64::from_bits(2)],
            Vec::new(),
            Array2::zeros((0, 0)),
            Vec::new(),
            vec![0.2, 0.0, f64::from_bits(1)],
        )
        .unwrap();
        let start = Instant::now();
        let baseline = 11;
        let expected_peak = baseline + 2 * source.value_dim() * size_of::<f64>();
        let boxed = certified_box_from_remainder_only_zonotope_unwired_with_clock(
            &source,
            limits(),
            budget(start, baseline, expected_peak),
            |_| start,
        )
        .unwrap();
        assert_eq!(boxed.report().peak_live_bytes(), expected_peak);
        assert_eq!(boxed.report().charged_items(), source.value_dim());
        for coordinate in 0..source.value_dim() {
            let exact_center = exact(source.center()[coordinate]);
            let exact_radius = exact(source.box_remainder()[coordinate]);
            assert!(
                exact(boxed.value().lower()[coordinate]) <= exact_center.clone() - &exact_radius
            );
            assert!(exact(boxed.value().upper()[coordinate]) >= exact_center + exact_radius);
        }

        let recentered = remainder_only_zonotope_from_certified_box_unwired_with_budget(
            boxed.value(),
            limits(),
            budget(Instant::now(), baseline, expected_peak),
        )
        .unwrap();
        assert_eq!(recentered.value().alpha_dim(), 0);
        assert_eq!(recentered.value().constraint_count(), 0);
        assert_eq!(recentered.report().peak_live_bytes(), expected_peak);
        // One numeric visit plus gated center finiteness, remainder finiteness,
        // and remainder-sign validation per coordinate. No generator or
        // predicate storage exists to validate.
        assert_eq!(recentered.report().charged_items(), 4 * source.value_dim());
        for coordinate in 0..source.value_dim() {
            assert_remainder_coordinate_contains(
                recentered.value(),
                coordinate,
                boxed.value().lower()[coordinate],
                boxed.value().upper()[coordinate],
            );
        }
    }

    #[test]
    fn remainder_only_source_rejects_symbols_and_predicates_separately() {
        let symbol = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![Vec::new()],
            Array2::zeros((0, 1)),
            Vec::new(),
            vec![1.0],
        )
        .unwrap();
        let predicate = ConstrainedZonotope64::try_new(
            vec![0.0],
            Vec::new(),
            Array2::zeros((1, 0)),
            vec![-1.0],
            vec![1.0],
        )
        .unwrap();
        let start = Instant::now();
        assert_eq!(
            certified_box_from_remainder_only_zonotope_unwired_with_clock(
                &symbol,
                limits(),
                budget(start, 0, usize::MAX),
                |_| start,
            ),
            Err(CertifiedBox64BridgeError::NonzeroAlphaDimension { alpha_dim: 1 })
        );
        assert_eq!(
            certified_box_from_remainder_only_zonotope_unwired_with_clock(
                &predicate,
                limits(),
                budget(start, 0, usize::MAX),
                |_| start,
            ),
            Err(CertifiedBox64BridgeError::NonzeroPredicateCount {
                constraint_count: 1
            })
        );
    }

    #[test]
    fn exact_interval_relu_phases_recenter_into_remainder_only_domain() {
        let input = CertifiedBox64::from_certified_bounds(
            &[-4.0, 2.0, -3.0, -0.0],
            &[-1.0, 6.0, 5.0, 0.0],
            limits(),
        )
        .unwrap();
        let start = Instant::now();
        let output = certified_box_relu_recenter_unwired_with_clock(
            &input,
            limits(),
            budget(start, 0, usize::MAX),
            |_| start,
        )
        .unwrap();
        let output = output.value();
        assert_eq!(output.alpha_dim(), 0);
        assert_eq!(output.constraint_count(), 0);

        // Negative and zero phases are exact points. Sterbenz-exact half
        // widths keep both the positive and unstable phases exact here; in
        // particular the unstable ReLU box must not leak below zero.
        assert_eq!(output.center()[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(output.box_remainder()[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(output.center()[1].to_bits(), 4.0_f64.to_bits());
        assert_eq!(output.box_remainder()[1].to_bits(), 2.0_f64.to_bits());
        assert_remainder_coordinate_contains(output, 1, 2.0, 6.0);
        assert_eq!(output.center()[2].to_bits(), 2.5_f64.to_bits());
        assert_eq!(output.box_remainder()[2].to_bits(), 2.5_f64.to_bits());
        assert_eq!(output.center()[2] - output.box_remainder()[2], 0.0);
        assert_eq!(output.center()[2] + output.box_remainder()[2], 5.0);
        assert_remainder_coordinate_contains(output, 2, 0.0, 5.0);
        assert_eq!(output.center()[3].to_bits(), 0.0_f64.to_bits());
        assert_eq!(output.box_remainder()[3].to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn exact_full_finite_range_recenters_without_spurious_overflow() {
        let input =
            CertifiedBox64::from_certified_bounds(&[-f64::MAX], &[f64::MAX], limits()).unwrap();
        let output = remainder_only_zonotope_from_certified_box_unwired_with_budget(
            &input,
            limits(),
            budget(Instant::now(), 0, usize::MAX),
        )
        .unwrap();
        assert_eq!(output.value().center()[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(output.value().box_remainder()[0], f64::MAX);
    }

    #[test]
    fn remainder_only_bridges_preserve_empty_dimension() {
        let empty_cz = ConstrainedZonotope64::try_new(
            Vec::new(),
            Vec::new(),
            Array2::zeros((0, 0)),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        let start = Instant::now();
        let boxed = certified_box_from_remainder_only_zonotope_unwired_with_clock(
            &empty_cz,
            limits(),
            budget(start, 7, 7),
            |_| start,
        )
        .unwrap();
        assert!(boxed.value().is_empty());
        assert_eq!(boxed.report().peak_live_bytes(), 7);
        assert_eq!(boxed.report().charged_items(), 0);

        let relu = certified_box_relu_recenter_unwired_with_clock(
            boxed.value(),
            limits(),
            budget(start, 7, 7),
            |_| start,
        )
        .unwrap();
        assert_eq!(relu.value().value_dim(), 0);
        assert_eq!(relu.value().alpha_dim(), 0);
        assert_eq!(relu.value().constraint_count(), 0);
        assert_eq!(relu.report().charged_items(), 0);
    }

    #[test]
    fn relu_recenter_revalidates_and_rejects_an_empty_coordinate_interval() {
        // Public construction cannot create this state; exercise the private
        // invariant check so a future internal constructor cannot bypass it.
        let malformed = CertifiedBox64 {
            lower: vec![1.0],
            upper: vec![-1.0],
        };
        let start = Instant::now();
        assert_eq!(
            certified_box_relu_recenter_unwired_with_clock(
                &malformed,
                limits(),
                budget(start, 0, usize::MAX),
                |_| start,
            ),
            Err(CertifiedBox64BridgeError::Box(
                CertifiedBox64Error::ReversedBounds { index: 0 }
            ))
        );
    }

    #[test]
    fn bridge_limits_and_peak_preflight_fail_before_allocation() {
        let input =
            CertifiedBox64::from_certified_bounds(&[-1.0, -2.0, -3.0], &[1.0, 2.0, 3.0], limits())
                .unwrap();
        let start = Instant::now();
        let baseline = 5;
        let exact_peak = baseline + 2 * input.len() * size_of::<f64>();
        certified_box_relu_recenter_unwired_with_clock(
            &input,
            limits(),
            budget(start, baseline, exact_peak),
            |_| start,
        )
        .unwrap();
        assert!(matches!(
            certified_box_relu_recenter_unwired_with_clock(
                &input,
                limits(),
                budget(start, baseline, exact_peak - 1),
                |_| start,
            ),
            Err(CertifiedBox64BridgeError::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                    required,
                    limit,
                }
            )) if required == exact_peak && limit == exact_peak - 1
        ));

        let exact_work = 4 * input.len();
        let mut capped = limits();
        capped.max_work_items = exact_work - 1;
        assert!(matches!(
            certified_box_relu_recenter_unwired_with_clock(
                &input,
                capped,
                budget(start, 0, usize::MAX),
                |_| start,
            ),
            Err(CertifiedBox64BridgeError::Box(
                CertifiedBox64Error::ResourceLimit {
                    resource: "certified Box-to-CZ charged work items",
                    required: 12,
                    limit: 11,
                }
            ))
        ));
    }

    #[test]
    fn bridge_deadlines_cover_numeric_materialization_and_publication_phases() {
        let input = CertifiedBox64::from_certified_bounds(&[-1.0], &[2.0], limits()).unwrap();
        let start = Instant::now();
        let expired = start + Duration::from_secs(2);
        for seam in [
            "certified Box-to-CZ numeric transform complete",
            "certified Box-to-CZ domain materialization complete",
            "certified Box-to-CZ publication",
        ] {
            let result = certified_box_relu_recenter_unwired_with_clock(
                &input,
                limits(),
                ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
                |checkpoint| if checkpoint == seam { expired } else { start },
            );
            assert!(
                matches!(
                    result,
                    Err(CertifiedBox64BridgeError::Budget(
                        ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
                    )) if checkpoint == seam
                ),
                "deadline seam {seam} must refuse publication"
            );
        }
    }

    #[test]
    fn bridge_charging_polls_inside_coordinate_and_domain_validation() {
        let dimension = crate::CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL;
        let input = CertifiedBox64::from_certified_bounds(
            &vec![-1.0; dimension],
            &vec![1.0; dimension],
            CertifiedBox64Limits {
                max_values: dimension,
                max_stored_f64: 2 * dimension,
                max_weight_elements: 0,
                max_work_items: 4 * dimension,
                max_scalar_products: 0,
            },
        )
        .unwrap();
        let large_limits = CertifiedBox64Limits {
            max_values: dimension,
            max_stored_f64: 2 * dimension,
            max_weight_elements: 0,
            max_work_items: 4 * dimension,
            max_scalar_products: 0,
        };
        let start = Instant::now();
        let expired = start + Duration::from_secs(2);
        for phase in [
            "certified Box-to-CZ coordinate transform",
            "constrained-zonotope finite-value validation",
        ] {
            let reads = Cell::new(0_usize);
            let result = certified_box_relu_recenter_unwired_with_clock(
                &input,
                large_limits,
                ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
                |checkpoint| {
                    reads.set(reads.get() + 1);
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
                    Err(CertifiedBox64BridgeError::Budget(
                        ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
                    )) if checkpoint == phase
                ),
                "deadline must be polled during {phase}"
            );
            assert!(reads.get() > 1);
        }
    }

    #[test]
    fn empty_generator_columns_are_preflighted_and_charged() {
        const EMPTY_COLUMNS: usize = 64;
        let cz = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![vec![]; EMPTY_COLUMNS],
            Array2::zeros((0, EMPTY_COLUMNS)),
            vec![],
            vec![0.0],
        )
        .unwrap();

        let mut rejected = limits();
        rejected.max_work_items = 0;
        assert!(matches!(
            unconstrained_zonotope_box_unwired(&cz, rejected),
            Err(CertifiedBox64Error::ResourceLimit {
                resource: "CZ generator column visits",
                required: EMPTY_COLUMNS,
                limit: 0,
            })
        ));

        let mut exact = limits();
        exact.max_work_items = EMPTY_COLUMNS;
        let (_, plan) = unconstrained_zonotope_box_unwired(&cz, exact).unwrap();
        assert_eq!(plan.generator_columns, EMPTY_COLUMNS);
        assert_eq!(plan.generator_nonzeros, 0);
        assert_eq!(plan.generator_work_items, EMPTY_COLUMNS);
        assert_eq!(plan.radius_additions, 0);
    }

    proptest! {
        #[test]
        fn affine_outward_endpoints_enclose_exact_dyadic_extrema(
            lo0 in -100_i16..=100,
            width0 in 0_u8..=20,
            lo1 in -100_i16..=100,
            width1 in 0_u8..=20,
            w0 in -20_i16..=20,
            w1 in -20_i16..=20,
            bias in -20_i16..=20,
        ) {
            let lower = [f64::from(lo0) / 8.0, f64::from(lo1) / 8.0];
            let upper = [
                lower[0] + f64::from(width0) / 16.0,
                lower[1] + f64::from(width1) / 16.0,
            ];
            let weights = [f64::from(w0) / 8.0, f64::from(w1) / 8.0];
            let bias = f64::from(bias) / 8.0;
            let input = CertifiedBox64::from_certified_bounds(&lower, &upper, limits()).unwrap();
            let matrix = Array2::from_shape_vec((1, 2), weights.to_vec()).unwrap();
            let (output, _) = certified_box_affine_unwired(
                &input,
                matrix.view(),
                &[bias],
                limits(),
            ).unwrap();

            let exact = |value: f64| BigRational::from_float(value).unwrap();
            let mut exact_lo = exact(bias);
            let mut exact_hi = exact(bias);
            for coordinate in 0..2 {
                let weight = exact(weights[coordinate]);
                let left = &weight * exact(lower[coordinate]);
                let right = weight * exact(upper[coordinate]);
                exact_lo += left.clone().min(right.clone());
                exact_hi += left.max(right);
            }
            prop_assert!(BigRational::from_float(output.lower()[0]).unwrap() <= exact_lo);
            prop_assert!(BigRational::from_float(output.upper()[0]).unwrap() >= exact_hi);
        }
    }

    #[test]
    fn every_cap_rejects_before_unbounded_work() {
        let input =
            CertifiedBox64::from_certified_bounds(&[-1.0, -1.0], &[1.0, 1.0], limits()).unwrap();
        let weights = array![[1.0, 1.0]];
        for capped in [
            CertifiedBox64Limits {
                max_values: 1,
                ..limits()
            },
            CertifiedBox64Limits {
                max_stored_f64: 1,
                ..limits()
            },
            CertifiedBox64Limits {
                max_weight_elements: 1,
                ..limits()
            },
            CertifiedBox64Limits {
                max_work_items: 1,
                ..limits()
            },
            CertifiedBox64Limits {
                max_scalar_products: 1,
                ..limits()
            },
        ] {
            assert!(certified_box_affine_unwired(&input, weights.view(), &[0.0], capped).is_err());
        }
    }
}
