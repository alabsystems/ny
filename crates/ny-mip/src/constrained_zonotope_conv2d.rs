// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Outward sparse `f64` convolution for an unwired constrained zonotope.
//!
//! The transform preserves local generator support instead of materializing a
//! dense `(value_dim, alpha_dim)` matrix.  Every scalar operation is enclosed
//! by adjacent `f64` endpoints.  Rounding width from the center and generator
//! contractions, together with the image of the input box remainder, is
//! accumulated into the output box remainder.  Therefore the returned domain
//! encloses the real convolution of the exact dyadic values supplied here.
//!
//! This module is deliberately **unwired**.  It does not read ONNX tensors,
//! choose ReLU relaxations, reduce generators, run on CUDA, or affect a scored
//! verdict.  In particular, callers must obtain weights and input domains from
//! proof-qualified sources; converting an already-rounded `f32` abstract state
//! cannot recover arithmetic error that was previously discarded.

use ndarray::{Array2, ArrayView4};

use crate::constrained_zonotope64::ConstrainedZonotope64CallGateError;
use crate::constrained_zonotope_call_budget::{
    ConstrainedZonotopeCallBudget, ConstrainedZonotopeCallBudgetError, ConstrainedZonotopeCallGate,
    ConstrainedZonotopeCallOutcome, ConstrainedZonotopeCallTracker,
    ConstrainedZonotopePeakLiveBytes, InertConstrainedZonotopeCallGate,
};
use crate::{ConstrainedZonotope64, ConstrainedZonotope64Error};

/// NCHW-without-batch convolution parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeConv2dSpec {
    /// Vertical and horizontal stride.
    pub stride: [usize; 2],
    /// Padding in ONNX order: top, left, bottom, right.
    pub padding: [usize; 4],
    /// Vertical and horizontal kernel dilation.
    pub dilation: [usize; 2],
    /// ONNX convolution group count.
    pub groups: usize,
}

/// Explicit resource limits for [`constrained_zonotope_conv2d_unwired`].
///
/// There is intentionally no `Default`: an experimental caller must choose
/// every cap.  Counts cover memory owned by the returned domain and the
/// transform's explicit sparse scratch, but not allocator bookkeeping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeConv2dLimits {
    /// Maximum input or output scalar value dimension.
    pub max_value_count: usize,
    /// Maximum alpha dimension and per-output dense scratch length.
    pub max_alpha_dim: usize,
    /// Maximum sparse generator nonzeros in either the input or output.
    pub max_generator_nonzeros: usize,
    /// Maximum logical kernel/weight elements inspected for finiteness.
    pub max_weight_elements: usize,
    /// Maximum nested output/kernel visits, including padding and zero weights.
    pub max_kernel_visits: usize,
    /// Maximum interval multiplications performed by the transform.
    pub max_interval_products: usize,
    /// Maximum retained predicate rows, including rows with zero alpha width.
    pub max_constraint_count: usize,
    /// Maximum number of retained `C` matrix elements.
    pub max_constraint_elements: usize,
}

/// Checked shape and work accounting for one completed convolution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeConv2dPlan {
    /// Input shape `[channels, height, width]`.
    pub input_shape: [usize; 3],
    /// Output shape `[channels, height, width]`.
    pub output_shape: [usize; 3],
    /// Kernel shape `[output_channels, input_channels_per_group, height, width]`.
    pub weight_shape: [usize; 4],
    /// Alpha dimension, unchanged by a linear transform.
    pub alpha_dim: usize,
    /// Constraint count, unchanged by a linear transform.
    pub constraint_count: usize,
    /// Logical weight elements validated before the transform.
    pub weight_elements: usize,
    /// Nested output/kernel visits, including padding and structural zeros.
    pub kernel_visits: usize,
    /// Sparse nonzeros in the input generators.
    pub input_generator_nonzeros: usize,
    /// Sparse nonzeros retained in the output generators.
    pub output_generator_nonzeros: usize,
    /// Interval products actually evaluated, including remainder products.
    pub interval_products: usize,
}

/// Invalid convolution data, exhausted resources, or arithmetic that could
/// not be enclosed by finite `f64` endpoints.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConstrainedZonotopeConv2dError {
    /// A tensor or vector has the wrong shape.
    #[error("shape mismatch for {field}: expected {expected:?}, got {got:?}")]
    Shape {
        /// Input whose shape is wrong.
        field: &'static str,
        /// Required shape.
        expected: Vec<usize>,
        /// Supplied shape.
        got: Vec<usize>,
    },

    /// A convolution parameter is structurally invalid.
    #[error("invalid convolution specification: {message}")]
    InvalidSpec {
        /// Concrete validation failure.
        message: String,
    },

    /// A supplied weight or bias is NaN or infinite.
    #[error("{field}[{index}] must be finite")]
    NonFinite {
        /// Input containing the value.
        field: &'static str,
        /// Row-major flattened index.
        index: usize,
    },

    /// A checked dimension or work calculation overflowed `usize`.
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

    /// An internal sparse-accumulation invariant was violated.
    #[error("internal convolution invariant violated: {message}")]
    InvariantViolation {
        /// Invariant that did not hold.
        message: &'static str,
    },

    /// An adjacent-float interval endpoint became non-finite.
    #[error("non-finite outward arithmetic while computing {operation}")]
    NonFiniteArithmetic {
        /// Failed contraction step.
        operation: &'static str,
    },

    /// The host flushes binary64 subnormals and cannot support this proof path.
    #[error("unsupported floating-point environment: {requirement}")]
    UnsupportedFloatingPoint {
        /// Required IEEE behavior.
        requirement: &'static str,
    },

    /// The validated result could not be materialized as a domain.
    #[error(transparent)]
    Domain(#[from] ConstrainedZonotope64Error),
}

/// Primitive or call-firewall refusal from budgeted convolution.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConstrainedZonotopeConv2dBudgetError {
    /// Geometry, resources, or outward arithmetic were invalid.
    #[error(transparent)]
    Transform(#[from] ConstrainedZonotopeConv2dError),

    /// The caller's deadline or aggregate peak-memory ceiling refused work.
    #[error(transparent)]
    Budget(#[from] ConstrainedZonotopeCallBudgetError),
}

/// Apply an exact-dyadic grouped 2-D convolution while preserving sparse local
/// generator support and charging all floating-point width to the box
/// remainder.
///
/// `input_shape` and the output use flattened `[channel, height, width]` order.
/// `weights` use ONNX `[output_channel, input_channel_per_group, kh, kw]`
/// order.  There is no batch axis.  The predicate `C alpha <= d` is copied
/// unchanged because convolution is affine in the represented values.
pub fn constrained_zonotope_conv2d_unwired(
    input: &ConstrainedZonotope64,
    input_shape: [usize; 3],
    weights: ArrayView4<'_, f64>,
    bias: &[f64],
    spec: ConstrainedZonotopeConv2dSpec,
    limits: ConstrainedZonotopeConv2dLimits,
) -> Result<(ConstrainedZonotope64, ConstrainedZonotopeConv2dPlan), ConstrainedZonotopeConv2dError>
{
    let mut gate = InertConstrainedZonotopeCallGate;
    match constrained_zonotope_conv2d_impl(
        input,
        input_shape,
        weights,
        bias,
        spec,
        limits,
        &mut gate,
    ) {
        Ok(value) => Ok(value),
        Err(ConstrainedZonotopeConv2dBudgetError::Transform(error)) => Err(error),
        Err(ConstrainedZonotopeConv2dBudgetError::Budget(_)) => {
            unreachable!("the inert Conv2d call gate cannot refuse work")
        }
    }
}

/// Apply convolution behind a synchronous call-local execution firewall.
///
/// The complete transform-owned logical peak is preflighted before adjacency,
/// scratch, or output allocation. `budget.baseline_live_bytes()` must include
/// the input, weights, bias, and any other caller-retained storage sharing the
/// ceiling. A completed domain remains private until the final checkpoint.
pub fn constrained_zonotope_conv2d_unwired_with_budget(
    input: &ConstrainedZonotope64,
    input_shape: [usize; 3],
    weights: ArrayView4<'_, f64>,
    bias: &[f64],
    spec: ConstrainedZonotopeConv2dSpec,
    limits: ConstrainedZonotopeConv2dLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> Result<
    ConstrainedZonotopeCallOutcome<(ConstrainedZonotope64, ConstrainedZonotopeConv2dPlan)>,
    ConstrainedZonotopeConv2dBudgetError,
> {
    let mut gate = ConstrainedZonotopeCallTracker::from_system_clock(budget)?;
    let value = constrained_zonotope_conv2d_impl(
        input,
        input_shape,
        weights,
        bias,
        spec,
        limits,
        &mut gate,
    )?;
    Ok(ConstrainedZonotopeCallOutcome::new(value, gate.report()))
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn constrained_zonotope_conv2d_unwired_with_clock<N>(
    input: &ConstrainedZonotope64,
    input_shape: [usize; 3],
    weights: ArrayView4<'_, f64>,
    bias: &[f64],
    spec: ConstrainedZonotopeConv2dSpec,
    limits: ConstrainedZonotopeConv2dLimits,
    budget: ConstrainedZonotopeCallBudget,
    now: N,
) -> Result<
    ConstrainedZonotopeCallOutcome<(ConstrainedZonotope64, ConstrainedZonotopeConv2dPlan)>,
    ConstrainedZonotopeConv2dBudgetError,
>
where
    N: FnMut(&'static str) -> std::time::Instant,
{
    let mut gate = ConstrainedZonotopeCallTracker::with_clock(budget, now)?;
    let value = constrained_zonotope_conv2d_impl(
        input,
        input_shape,
        weights,
        bias,
        spec,
        limits,
        &mut gate,
    )?;
    Ok(ConstrainedZonotopeCallOutcome::new(value, gate.report()))
}

#[allow(clippy::too_many_arguments)]
fn constrained_zonotope_conv2d_impl<G>(
    input: &ConstrainedZonotope64,
    input_shape: [usize; 3],
    weights: ArrayView4<'_, f64>,
    bias: &[f64],
    spec: ConstrainedZonotopeConv2dSpec,
    limits: ConstrainedZonotopeConv2dLimits,
    gate: &mut G,
) -> Result<
    (ConstrainedZonotope64, ConstrainedZonotopeConv2dPlan),
    ConstrainedZonotopeConv2dBudgetError,
>
where
    G: ConstrainedZonotopeCallGate,
{
    require_gradual_underflow()?;
    gate.checkpoint("Conv2d floating-point preflight")?;
    let geometry =
        validate_geometry_with_gate(input, input_shape, weights, bias, spec, limits, gate)?;
    gate.checkpoint("Conv2d geometry validation complete")?;

    let mut input_generator_nonzeros = 0_usize;
    for generator in input.generators() {
        gate.charge_items(1, "Conv2d generator geometry validation")?;
        input_generator_nonzeros = input_generator_nonzeros
            .checked_add(generator.nnz())
            .ok_or(ConstrainedZonotopeConv2dError::ResourceOverflow {
                operation: "input generator nonzeros",
            })?;
        check_limit(
            "input generator nonzeros",
            input_generator_nonzeros,
            limits.max_generator_nonzeros,
        )?;
    }
    if gate.is_enforcing() {
        gate.preflight_peak_live_bytes(conv2d_peak_live_bytes(
            input,
            geometry,
            input_generator_nonzeros,
            limits,
        )?)?;
    }
    gate.checkpoint("Conv2d peak-memory preflight complete")?;

    let adjacency = build_input_adjacency(input, input_generator_nonzeros, gate)?;
    gate.checkpoint("Conv2d adjacency construction complete")?;
    let expected_interval_products = if gate.is_enforcing() {
        Some(preflight_interval_products_with_gate(
            input,
            input_shape,
            weights,
            spec,
            geometry,
            &adjacency,
            limits.max_interval_products,
            gate,
        )?)
    } else {
        None
    };
    gate.checkpoint("Conv2d interval-product preflight complete")?;
    let alpha_dim = input.alpha_dim();
    let mut generator_scratch: Vec<Option<OutwardInterval>> = Vec::new();
    gate.checkpoint("Conv2d generator-scratch allocation")?;
    try_reserve(
        &mut generator_scratch,
        alpha_dim,
        "generator interval scratch",
    )?;
    for _ in 0..alpha_dim {
        gate.charge_items(1, "Conv2d generator-scratch initialization")?;
        generator_scratch.push(None);
    }
    let mut touched_generators = Vec::new();
    gate.checkpoint("Conv2d touched-generator allocation")?;
    try_reserve(
        &mut touched_generators,
        alpha_dim,
        "touched-generator scratch",
    )?;

    let output_value_count = geometry.output_value_count;
    let mut output_center = Vec::new();
    let mut output_remainder = Vec::new();
    gate.checkpoint("Conv2d output-center allocation")?;
    try_reserve(&mut output_center, output_value_count, "output center")?;
    gate.checkpoint("Conv2d output-remainder allocation")?;
    try_reserve(
        &mut output_remainder,
        output_value_count,
        "output box remainder",
    )?;

    let mut output_generators: Vec<Vec<(usize, f64)>> = Vec::new();
    gate.checkpoint("Conv2d generator-column allocation")?;
    try_reserve(
        &mut output_generators,
        alpha_dim,
        "output generator columns",
    )?;
    for _ in 0..alpha_dim {
        gate.charge_items(1, "Conv2d generator-column initialization")?;
        output_generators.push(Vec::new());
    }

    let [input_channels, input_height, input_width] = input_shape;
    let [output_channels, kernel_input_channels, kernel_height, kernel_width] =
        geometry.weight_shape;
    let [_output_channels, output_height, output_width] = geometry.output_shape;
    let input_channels_per_group = input_channels / spec.groups;
    let output_channels_per_group = output_channels / spec.groups;
    debug_assert_eq!(kernel_input_channels, input_channels_per_group);

    let mut interval_products = 0_usize;
    let mut output_generator_nonzeros = 0_usize;

    for output_channel in 0..output_channels {
        let group = output_channel / output_channels_per_group;
        let input_channel_base = group * input_channels_per_group;
        for output_y in 0..output_height {
            for output_x in 0..output_width {
                gate.charge_items(1, "Conv2d output transform")?;
                let output_index = output_center.len();
                let mut center_sum = OutwardInterval::exact(bias[output_channel]);
                let mut remainder_sum = OutwardInterval::zero();

                for kernel_input_channel in 0..kernel_input_channels {
                    let input_channel = input_channel_base + kernel_input_channel;
                    for kernel_y in 0..kernel_height {
                        gate.charge_items(1, "Conv2d kernel transform")?;
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
                            gate.charge_items(1, "Conv2d kernel transform")?;
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
                            let input_index =
                                (input_channel * input_height + input_y) * input_width + input_x;

                            consume_product(&mut interval_products, limits.max_interval_products)?;
                            center_sum = center_sum
                                .add(outward_product(weight, input.center()[input_index])?)?;

                            let input_radius = input.box_remainder()[input_index];
                            if input_radius != 0.0 {
                                consume_product(
                                    &mut interval_products,
                                    limits.max_interval_products,
                                )?;
                                remainder_sum = remainder_sum
                                    .add(outward_product(weight.abs(), input_radius)?.abs())?;
                            }

                            for &(generator_index, coefficient) in &adjacency[input_index] {
                                gate.charge_items(1, "Conv2d generator accumulation")?;
                                consume_product(
                                    &mut interval_products,
                                    limits.max_interval_products,
                                )?;
                                let contribution = outward_product(weight, coefficient)?;
                                let slot = &mut generator_scratch[generator_index];
                                if let Some(current) = *slot {
                                    *slot = Some(current.add(contribution)?);
                                } else {
                                    *slot = Some(contribution);
                                    touched_generators.push(generator_index);
                                }
                            }
                        }
                    }
                }

                let (nominal_center, center_error) = center_sum.nominal_and_error()?;
                let mut total_remainder = remainder_sum.hi;
                total_remainder = add_nonnegative_upper(total_remainder, center_error)?;

                for &generator_index in &touched_generators {
                    gate.charge_items(1, "Conv2d generator publication staging")?;
                    let interval = generator_scratch[generator_index].take().ok_or(
                        ConstrainedZonotopeConv2dError::InvariantViolation {
                            message: "a touched generator has no accumulated interval",
                        },
                    )?;
                    let (coefficient, coefficient_error) = interval.nominal_and_error()?;
                    total_remainder = add_nonnegative_upper(total_remainder, coefficient_error)?;
                    if coefficient != 0.0 {
                        output_generator_nonzeros = output_generator_nonzeros
                            .checked_add(1)
                            .ok_or(ConstrainedZonotopeConv2dError::ResourceOverflow {
                                operation: "output generator nonzeros",
                            })?;
                        check_limit(
                            "output generator nonzeros",
                            output_generator_nonzeros,
                            limits.max_generator_nonzeros,
                        )?;
                        gate.checkpoint("Conv2d generator-entry allocation")?;
                        let generator = &mut output_generators[generator_index];
                        // The peak preflight charges one candidate slot plus
                        // one validated-constructor slot per retained
                        // coefficient. Amortized growth may request additional
                        // speculative candidate capacity, so the enforcing path
                        // must request exactly the one charged slot. Preserve
                        // the legacy API's allocation policy verbatim.
                        let reservation = if gate.is_enforcing() {
                            generator.try_reserve_exact(1)
                        } else {
                            generator.try_reserve(1)
                        };
                        reservation.map_err(|_| {
                            ConstrainedZonotopeConv2dError::AllocationFailure {
                                resource: "output generator coefficients",
                            }
                        })?;
                        generator.push((output_index, coefficient));
                    }
                }
                touched_generators.clear();

                output_center.push(nominal_center);
                output_remainder.push(total_remainder);
            }
        }
    }
    gate.checkpoint("Conv2d numeric transform complete")?;

    debug_assert_eq!(output_center.len(), output_value_count);
    if let Some(expected) = expected_interval_products {
        debug_assert_eq!(interval_products, expected);
    }
    let constraints = clone_constraints(input, gate)?;
    gate.checkpoint("Conv2d constraint clone complete")?;
    gate.checkpoint("Conv2d right-hand-side allocation")?;
    let rhs = clone_slice_with_gate(input.rhs(), "constraint right-hand side", gate)?;
    gate.checkpoint("Conv2d right-hand-side clone complete")?;
    gate.checkpoint("Conv2d domain materialization")?;
    let output = ConstrainedZonotope64::try_new_with_call_gate(
        output_center,
        output_generators,
        constraints,
        rhs,
        output_remainder,
        gate,
    )
    .map_err(|error| match error {
        ConstrainedZonotope64CallGateError::Domain(error) => {
            ConstrainedZonotopeConv2dBudgetError::Transform(ConstrainedZonotopeConv2dError::Domain(
                error,
            ))
        }
        ConstrainedZonotope64CallGateError::Budget(error) => {
            ConstrainedZonotopeConv2dBudgetError::Budget(error)
        }
    })?;
    gate.checkpoint("Conv2d domain materialization complete")?;
    let plan = ConstrainedZonotopeConv2dPlan {
        input_shape,
        output_shape: geometry.output_shape,
        weight_shape: geometry.weight_shape,
        alpha_dim,
        constraint_count: input.constraint_count(),
        weight_elements: geometry.weight_elements,
        kernel_visits: geometry.kernel_visits,
        input_generator_nonzeros,
        output_generator_nonzeros,
        interval_products,
    };
    gate.checkpoint("Conv2d publication")?;
    Ok((output, plan))
}

#[derive(Clone, Copy, Debug)]
struct Geometry {
    output_shape: [usize; 3],
    weight_shape: [usize; 4],
    output_value_count: usize,
    constraint_elements: usize,
    weight_elements: usize,
    kernel_visits: usize,
}

/// Conservative transform-owned peak. Scratch from disjoint phases is summed
/// deliberately. Two complete generator representations cover both exact
/// candidate-buffer relocation during growth and candidate/private overlap
/// during final materialization; those phases do not overlap each other.
/// Retained input, weights, and bias belong in the caller's baseline.
fn conv2d_peak_live_bytes(
    input: &ConstrainedZonotope64,
    geometry: Geometry,
    input_generator_nonzeros: usize,
    limits: ConstrainedZonotopeConv2dLimits,
) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    let output_generator_slots = input
        .alpha_dim()
        .checked_mul(geometry.output_value_count)
        .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
            operation: "Conv2d output generator slots",
        })?;
    let output_generator_nonzeros = output_generator_slots
        .min(limits.max_generator_nonzeros)
        .min(limits.max_interval_products);

    let mut peak = ConstrainedZonotopePeakLiveBytes::new();
    peak.add_elements::<usize>(input.value_dim(), "Conv2d adjacency-count bytes")?;
    peak.add_elements::<Vec<(usize, f64)>>(input.value_dim(), "Conv2d adjacency-column bytes")?;
    peak.add_elements::<(usize, f64)>(input_generator_nonzeros, "Conv2d adjacency-entry bytes")?;
    peak.add_elements::<Option<OutwardInterval>>(
        input.alpha_dim(),
        "Conv2d generator-scratch bytes",
    )?;
    peak.add_elements::<usize>(input.alpha_dim(), "Conv2d touched-generator bytes")?;
    peak.add_elements::<f64>(geometry.output_value_count, "Conv2d output-center bytes")?;
    peak.add_elements::<f64>(geometry.output_value_count, "Conv2d output-remainder bytes")?;
    let doubled_alpha_headers = input.alpha_dim().checked_mul(2).ok_or(
        ConstrainedZonotopeCallBudgetError::ResourceOverflow {
            operation: "Conv2d doubled generator-column headers",
        },
    )?;
    peak.add_elements::<Vec<(usize, f64)>>(
        doubled_alpha_headers,
        "Conv2d candidate and validated generator-column bytes",
    )?;
    let doubled_output_nonzeros = output_generator_nonzeros.checked_mul(2).ok_or(
        ConstrainedZonotopeCallBudgetError::ResourceOverflow {
            operation: "Conv2d doubled output generator nonzeros",
        },
    )?;
    peak.add_elements::<(usize, f64)>(
        doubled_output_nonzeros,
        "Conv2d candidate and validated generator-entry bytes",
    )?;
    peak.add_elements::<f64>(
        geometry.constraint_elements,
        "Conv2d constraint-matrix bytes",
    )?;
    peak.add_elements::<f64>(
        input.constraint_count(),
        "Conv2d constraint right-hand-side bytes",
    )?;
    Ok(peak.finish())
}

#[allow(clippy::too_many_arguments)]
fn validate_geometry_with_gate<G>(
    input: &ConstrainedZonotope64,
    input_shape: [usize; 3],
    weights: ArrayView4<'_, f64>,
    bias: &[f64],
    spec: ConstrainedZonotopeConv2dSpec,
    limits: ConstrainedZonotopeConv2dLimits,
    gate: &mut G,
) -> Result<Geometry, ConstrainedZonotopeConv2dBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let input_value_count = checked_product(&input_shape, "convolution input value count")?;
    if input.value_dim() != input_value_count {
        return Err(ConstrainedZonotopeConv2dError::Shape {
            field: "input domain",
            expected: vec![input_value_count],
            got: vec![input.value_dim()],
        }
        .into());
    }
    check_limit(
        "input value count",
        input_value_count,
        limits.max_value_count,
    )?;
    check_limit("alpha dimension", input.alpha_dim(), limits.max_alpha_dim)?;
    check_limit(
        "constraint count",
        input.constraint_count(),
        limits.max_constraint_count,
    )?;

    let constraint_elements = input
        .constraint_count()
        .checked_mul(input.alpha_dim())
        .ok_or(ConstrainedZonotopeConv2dError::ResourceOverflow {
            operation: "constraint matrix elements",
        })?;
    check_limit(
        "constraint matrix elements",
        constraint_elements,
        limits.max_constraint_elements,
    )?;

    let [input_channels, input_height, input_width] = input_shape;
    if input_channels == 0 || input_height == 0 || input_width == 0 {
        return Err(ConstrainedZonotopeConv2dError::InvalidSpec {
            message: format!("input shape must be non-empty, got {input_shape:?}"),
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
    let weight_elements = checked_product(&weight_shape, "convolution weight elements")?;
    check_limit(
        "weight elements",
        weight_elements,
        limits.max_weight_elements,
    )?;
    validate_finite_with_gate("weights", weights.iter().copied(), gate)?;
    validate_finite_with_gate("bias", bias.iter().copied(), gate)?;

    let output_height = output_dimension(
        input_height,
        spec.padding[0],
        spec.padding[2],
        kernel_height,
        spec.dilation[0],
        spec.stride[0],
        "output height",
    )?;
    let output_width = output_dimension(
        input_width,
        spec.padding[1],
        spec.padding[3],
        kernel_width,
        spec.dilation[1],
        spec.stride[1],
        "output width",
    )?;
    let output_shape = [output_channels, output_height, output_width];
    let output_value_count = checked_product(&output_shape, "convolution output value count")?;
    check_limit(
        "output value count",
        output_value_count,
        limits.max_value_count,
    )?;
    let kernel_visits = checked_product(
        &[
            output_value_count,
            kernel_input_channels,
            kernel_height,
            kernel_width,
        ],
        "convolution kernel visits",
    )?;
    check_limit("kernel visits", kernel_visits, limits.max_kernel_visits)?;

    Ok(Geometry {
        output_shape,
        weight_shape,
        output_value_count,
        constraint_elements,
        weight_elements,
        kernel_visits,
    })
}

pub(crate) fn output_dimension(
    input: usize,
    padding_before: usize,
    padding_after: usize,
    kernel: usize,
    dilation: usize,
    stride: usize,
    operation: &'static str,
) -> Result<usize, ConstrainedZonotopeConv2dError> {
    let effective_kernel = kernel
        .checked_sub(1)
        .and_then(|value| value.checked_mul(dilation))
        .and_then(|value| value.checked_add(1))
        .ok_or(ConstrainedZonotopeConv2dError::ResourceOverflow { operation })?;
    let padded = input
        .checked_add(padding_before)
        .and_then(|value| value.checked_add(padding_after))
        .ok_or(ConstrainedZonotopeConv2dError::ResourceOverflow { operation })?;
    if padded < effective_kernel {
        return Err(ConstrainedZonotopeConv2dError::InvalidSpec {
            message: format!(
                "{operation} has padded input {padded} smaller than effective kernel {effective_kernel}"
            ),
        });
    }
    Ok((padded - effective_kernel) / stride + 1)
}

pub(crate) fn input_coordinate(
    output: usize,
    stride: usize,
    kernel: usize,
    dilation: usize,
    padding_before: usize,
    input_size: usize,
) -> Result<Option<usize>, ConstrainedZonotopeConv2dError> {
    let padded_coordinate = output
        .checked_mul(stride)
        .and_then(|value| {
            kernel
                .checked_mul(dilation)
                .and_then(|term| value.checked_add(term))
        })
        .ok_or(ConstrainedZonotopeConv2dError::ResourceOverflow {
            operation: "convolution input coordinate",
        })?;
    if padded_coordinate < padding_before {
        return Ok(None);
    }
    let coordinate = padded_coordinate - padding_before;
    Ok((coordinate < input_size).then_some(coordinate))
}

/// Count every interval product before output allocation or floating-point
/// contraction. This makes the caller's work cap a true preflight boundary.
#[allow(clippy::too_many_arguments)]
fn preflight_interval_products_with_gate<G>(
    input: &ConstrainedZonotope64,
    input_shape: [usize; 3],
    weights: ArrayView4<'_, f64>,
    spec: ConstrainedZonotopeConv2dSpec,
    geometry: Geometry,
    adjacency: &[Vec<(usize, f64)>],
    limit: usize,
    gate: &mut G,
) -> Result<usize, ConstrainedZonotopeConv2dBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let [input_channels, input_height, input_width] = input_shape;
    let [output_channels, kernel_input_channels, kernel_height, kernel_width] =
        geometry.weight_shape;
    let [_output_channels, output_height, output_width] = geometry.output_shape;
    let input_channels_per_group = input_channels / spec.groups;
    let output_channels_per_group = output_channels / spec.groups;
    let mut products = 0_usize;

    for output_channel in 0..output_channels {
        let group = output_channel / output_channels_per_group;
        let input_channel_base = group * input_channels_per_group;
        for output_y in 0..output_height {
            for output_x in 0..output_width {
                gate.charge_items(1, "Conv2d interval-product output preflight")?;
                for kernel_input_channel in 0..kernel_input_channels {
                    let input_channel = input_channel_base + kernel_input_channel;
                    for kernel_y in 0..kernel_height {
                        gate.charge_items(1, "Conv2d interval-product kernel preflight")?;
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
                            gate.charge_items(1, "Conv2d interval-product kernel preflight")?;
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
                            let input_index =
                                (input_channel * input_height + input_y) * input_width + input_x;
                            let products_here = 1_usize
                                .checked_add(usize::from(input.box_remainder()[input_index] != 0.0))
                                .and_then(|count| count.checked_add(adjacency[input_index].len()))
                                .ok_or(ConstrainedZonotopeConv2dError::ResourceOverflow {
                                    operation: "interval product count",
                                })?;
                            consume_product_batch_equivalent(&mut products, products_here, limit)?;
                        }
                    }
                }
            }
        }
    }
    Ok(products)
}

fn build_input_adjacency<G>(
    input: &ConstrainedZonotope64,
    total_nonzeros: usize,
    gate: &mut G,
) -> Result<Vec<Vec<(usize, f64)>>, ConstrainedZonotopeConv2dBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let mut counts = Vec::new();
    gate.checkpoint("Conv2d adjacency-count allocation")?;
    try_reserve(&mut counts, input.value_dim(), "input adjacency counts")?;
    for _ in 0..input.value_dim() {
        gate.charge_items(1, "Conv2d adjacency-count initialization")?;
        counts.push(0_usize);
    }
    for generator in input.generators() {
        gate.charge_items(1, "Conv2d adjacency generator counting")?;
        for (value_index, _) in generator.entries() {
            gate.charge_items(1, "Conv2d adjacency entry counting")?;
            counts[value_index] = counts[value_index].checked_add(1).ok_or(
                ConstrainedZonotopeConv2dError::ResourceOverflow {
                    operation: "per-value generator adjacency",
                },
            )?;
        }
    }

    let mut adjacency = Vec::new();
    gate.checkpoint("Conv2d adjacency-column allocation")?;
    try_reserve(
        &mut adjacency,
        input.value_dim(),
        "input generator adjacency",
    )?;
    for &count in &counts {
        gate.charge_items(1, "Conv2d adjacency-column construction")?;
        let mut entries = Vec::new();
        gate.checkpoint("Conv2d adjacency-entry allocation")?;
        try_reserve(&mut entries, count, "input generator adjacency entries")?;
        adjacency.push(entries);
    }
    let mut filled_nonzeros = 0_usize;
    for (generator_index, generator) in input.generators().iter().enumerate() {
        gate.charge_items(1, "Conv2d adjacency generator fill")?;
        for (value_index, coefficient) in generator.entries() {
            gate.charge_items(1, "Conv2d adjacency entry fill")?;
            adjacency[value_index].push((generator_index, coefficient));
            filled_nonzeros = filled_nonzeros.checked_add(1).ok_or(
                ConstrainedZonotopeConv2dError::ResourceOverflow {
                    operation: "filled input generator adjacency",
                },
            )?;
        }
    }
    debug_assert_eq!(filled_nonzeros, total_nonzeros);
    Ok(adjacency)
}

fn clone_constraints<G>(
    input: &ConstrainedZonotope64,
    gate: &mut G,
) -> Result<Array2<f64>, ConstrainedZonotopeConv2dBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let shape = (input.constraint_count(), input.alpha_dim());
    let element_count =
        shape
            .0
            .checked_mul(shape.1)
            .ok_or(ConstrainedZonotopeConv2dError::ResourceOverflow {
                operation: "constraint matrix elements",
            })?;
    let constraints = input.constraints();
    let mut values = Vec::new();
    gate.checkpoint("Conv2d constraint-matrix allocation")?;
    try_reserve(&mut values, element_count, "constraint matrix")?;
    for row in 0..shape.0 {
        gate.charge_items(1, "Conv2d constraint-row clone")?;
        for column in 0..shape.1 {
            gate.charge_items(1, "Conv2d constraint-element clone")?;
            values.push(constraints[[row, column]]);
        }
    }
    Array2::from_shape_vec(shape, values).map_err(|_| {
        ConstrainedZonotopeConv2dBudgetError::Transform(
            ConstrainedZonotopeConv2dError::ResourceOverflow {
                operation: "constraint matrix shape",
            },
        )
    })
}

fn clone_slice_with_gate<T: Copy, G>(
    source: &[T],
    resource: &'static str,
    gate: &mut G,
) -> Result<Vec<T>, ConstrainedZonotopeConv2dBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let mut output = Vec::new();
    try_reserve(&mut output, source.len(), resource)?;
    for &value in source {
        gate.charge_items(1, "Conv2d right-hand-side clone")?;
        output.push(value);
    }
    Ok(output)
}

fn validate_finite_with_gate<G>(
    field: &'static str,
    values: impl IntoIterator<Item = f64>,
    gate: &mut G,
) -> Result<(), ConstrainedZonotopeConv2dBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    for (index, value) in values.into_iter().enumerate() {
        gate.charge_items(1, "Conv2d finite-parameter validation")?;
        if !value.is_finite() {
            return Err(ConstrainedZonotopeConv2dError::NonFinite { field, index }.into());
        }
    }
    Ok(())
}

fn checked_product(
    dimensions: &[usize],
    operation: &'static str,
) -> Result<usize, ConstrainedZonotopeConv2dError> {
    dimensions.iter().try_fold(1_usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or(ConstrainedZonotopeConv2dError::ResourceOverflow { operation })
    })
}

fn check_limit(
    resource: &'static str,
    required: usize,
    limit: usize,
) -> Result<(), ConstrainedZonotopeConv2dError> {
    if required > limit {
        return Err(ConstrainedZonotopeConv2dError::ResourceLimit {
            resource,
            required,
            limit,
        });
    }
    Ok(())
}

fn consume_product(count: &mut usize, limit: usize) -> Result<(), ConstrainedZonotopeConv2dError> {
    *count = count
        .checked_add(1)
        .ok_or(ConstrainedZonotopeConv2dError::ResourceOverflow {
            operation: "interval product count",
        })?;
    check_limit("interval products", *count, limit)
}

/// Advance a preflight count in constant time while preserving the exact
/// failure that repeated [`consume_product`] calls would expose.
///
/// In particular, a batch that crosses a finite limit reports the first
/// refused item (`limit + 1`), not the end of the batch. If the limit is
/// `usize::MAX`, arithmetic overflow remains the first possible refusal.
fn consume_product_batch_equivalent(
    count: &mut usize,
    additional: usize,
    limit: usize,
) -> Result<(), ConstrainedZonotopeConv2dError> {
    let Some(required) = count.checked_add(additional) else {
        if limit < usize::MAX {
            return Err(ConstrainedZonotopeConv2dError::ResourceLimit {
                resource: "interval products",
                required: limit + 1,
                limit,
            });
        }
        return Err(ConstrainedZonotopeConv2dError::ResourceOverflow {
            operation: "interval product count",
        });
    };
    if required > limit {
        debug_assert!(limit < usize::MAX);
        return Err(ConstrainedZonotopeConv2dError::ResourceLimit {
            resource: "interval products",
            required: limit + 1,
            limit,
        });
    }
    *count = required;
    Ok(())
}

fn try_reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    resource: &'static str,
) -> Result<(), ConstrainedZonotopeConv2dError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| ConstrainedZonotopeConv2dError::AllocationFailure { resource })
}

/// Reject FTZ/DAZ before adjacent-float intervals are used as proof objects.
fn require_gradual_underflow() -> Result<(), ConstrainedZonotopeConv2dError> {
    let half = std::hint::black_box(0.5_f64);
    let min_normal = std::hint::black_box(f64::MIN_POSITIVE);
    let min_subnormal = std::hint::black_box(f64::from_bits(1));
    let two_subnormals = std::hint::black_box(f64::from_bits(2));
    if std::hint::black_box(min_normal * half).to_bits() != 0x0008_0000_0000_0000
        || std::hint::black_box(two_subnormals * half).to_bits() != 1
        || std::hint::black_box(min_subnormal + min_subnormal).to_bits() != 2
    {
        return Err(ConstrainedZonotopeConv2dError::UnsupportedFloatingPoint {
            requirement: "IEEE-754 binary64 gradual underflow (FTZ/DAZ disabled)",
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct OutwardInterval {
    lo: f64,
    hi: f64,
}

impl OutwardInterval {
    const fn zero() -> Self {
        Self { lo: 0.0, hi: 0.0 }
    }

    const fn exact(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }

    fn is_exact_zero(self) -> bool {
        self.lo == 0.0 && self.hi == 0.0
    }

    fn add(self, rhs: Self) -> Result<Self, ConstrainedZonotopeConv2dError> {
        if self.is_exact_zero() {
            return Ok(rhs);
        }
        if rhs.is_exact_zero() {
            return Ok(self);
        }
        Ok(Self {
            lo: round_down(self.lo + rhs.lo, "convolution interval sum")?,
            hi: round_up(self.hi + rhs.hi, "convolution interval sum")?,
        })
    }

    const fn abs(self) -> Self {
        if self.lo >= 0.0 {
            self
        } else if self.hi <= 0.0 {
            Self {
                lo: -self.hi,
                hi: -self.lo,
            }
        } else {
            Self {
                lo: 0.0,
                hi: if -self.lo > self.hi {
                    -self.lo
                } else {
                    self.hi
                },
            }
        }
    }

    fn nominal_and_error(self) -> Result<(f64, f64), ConstrainedZonotopeConv2dError> {
        if self.lo == self.hi {
            return Ok((self.lo, 0.0));
        }
        let midpoint = self.lo * 0.5 + self.hi * 0.5;
        let nominal = if midpoint.is_finite() && midpoint >= self.lo && midpoint <= self.hi {
            midpoint
        } else {
            self.lo
        };
        let lower_error = upper_difference(nominal, self.lo)?;
        let upper_error = upper_difference(self.hi, nominal)?;
        Ok((nominal, lower_error.max(upper_error)))
    }
}

fn outward_product(
    left: f64,
    right: f64,
) -> Result<OutwardInterval, ConstrainedZonotopeConv2dError> {
    if left == 0.0 || right == 0.0 {
        return Ok(OutwardInterval::zero());
    }
    if left == 1.0 {
        return Ok(OutwardInterval::exact(right));
    }
    if right == 1.0 {
        return Ok(OutwardInterval::exact(left));
    }
    if left == -1.0 {
        return Ok(OutwardInterval::exact(-right));
    }
    if right == -1.0 {
        return Ok(OutwardInterval::exact(-left));
    }
    let product = left * right;
    Ok(OutwardInterval {
        lo: round_down(product, "convolution interval product")?,
        hi: round_up(product, "convolution interval product")?,
    })
}

fn upper_difference(upper: f64, lower: f64) -> Result<f64, ConstrainedZonotopeConv2dError> {
    if upper == lower {
        return Ok(0.0);
    }
    round_up(upper - lower, "convolution representation error")
}

fn add_nonnegative_upper(left: f64, right: f64) -> Result<f64, ConstrainedZonotopeConv2dError> {
    if left == 0.0 {
        return Ok(right);
    }
    if right == 0.0 {
        return Ok(left);
    }
    round_up(left + right, "convolution box remainder sum")
}

fn round_down(value: f64, operation: &'static str) -> Result<f64, ConstrainedZonotopeConv2dError> {
    if !value.is_finite() {
        return Err(ConstrainedZonotopeConv2dError::NonFiniteArithmetic { operation });
    }
    let outward = value.next_down();
    if !outward.is_finite() {
        return Err(ConstrainedZonotopeConv2dError::NonFiniteArithmetic { operation });
    }
    Ok(outward)
}

fn round_up(value: f64, operation: &'static str) -> Result<f64, ConstrainedZonotopeConv2dError> {
    if !value.is_finite() {
        return Err(ConstrainedZonotopeConv2dError::NonFiniteArithmetic { operation });
    }
    let outward = value.next_up();
    if !outward.is_finite() {
        return Err(ConstrainedZonotopeConv2dError::NonFiniteArithmetic { operation });
    }
    Ok(outward)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::mem::size_of;
    use std::time::{Duration, Instant};

    use ndarray::{array, Array2, Array4};
    use num_rational::BigRational;
    use num_traits::{Signed, Zero};
    use proptest::prelude::*;

    use super::*;

    fn limits() -> ConstrainedZonotopeConv2dLimits {
        ConstrainedZonotopeConv2dLimits {
            max_value_count: 10_000,
            max_alpha_dim: 1_000,
            max_generator_nonzeros: 100_000,
            max_weight_elements: 100_000,
            max_kernel_visits: 1_000_000,
            max_interval_products: 1_000_000,
            max_constraint_count: 100_000,
            max_constraint_elements: 100_000,
        }
    }

    fn spec() -> ConstrainedZonotopeConv2dSpec {
        ConstrainedZonotopeConv2dSpec {
            stride: [1, 1],
            padding: [0, 0, 0, 0],
            dilation: [1, 1],
            groups: 1,
        }
    }

    fn simple_input() -> ConstrainedZonotope64 {
        ConstrainedZonotope64::try_new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![vec![(0, 0.25), (3, -0.5)], vec![(1, 0.75)]],
            array![[1.0, -0.25]],
            vec![0.5],
            vec![0.1, 0.2, 0.3, 0.4],
        )
        .unwrap()
    }

    fn rat(value: f64) -> BigRational {
        BigRational::from_float(value).expect("finite test value")
    }

    fn bounded_normal(raw: u64, exponent: i16) -> f64 {
        debug_assert!((-1_022..=1_023).contains(&exponent));
        let sign = raw & (1_u64 << 63);
        let fraction = raw & ((1_u64 << 52) - 1);
        let biased_exponent = u64::try_from(i32::from(exponent) + 1_023).unwrap();
        f64::from_bits(sign | (biased_exponent << 52) | fraction)
    }

    fn coefficient(domain: &ConstrainedZonotope64, generator: usize, value: usize) -> f64 {
        domain.generators()[generator]
            .entries()
            .find_map(|(index, coefficient)| (index == value).then_some(coefficient))
            .unwrap_or(0.0)
    }

    #[test]
    fn exact_dyadic_convolution_preserves_constraints_and_contains_transform() {
        let input = simple_input();
        let weights = Array4::from_shape_vec((1, 1, 2, 2), vec![1.0, -2.0, 0.5, 3.0]).unwrap();
        let (output, plan) = constrained_zonotope_conv2d_unwired(
            &input,
            [1, 2, 2],
            weights.view(),
            &[0.125],
            spec(),
            limits(),
        )
        .unwrap();

        assert_eq!(plan.output_shape, [1, 1, 1]);
        assert_eq!(output.constraints(), input.constraints());
        assert_eq!(output.rhs(), input.rhs());

        let exact_center = rat(0.125)
            + rat(1.0) * rat(1.0)
            + rat(-2.0) * rat(2.0)
            + rat(0.5) * rat(3.0)
            + rat(3.0) * rat(4.0);
        let exact_g0 = rat(1.0) * rat(0.25) + rat(3.0) * rat(-0.5);
        let exact_g1 = rat(-2.0) * rat(0.75);
        let propagated_remainder =
            rat(1.0) * rat(0.1) + rat(2.0) * rat(0.2) + rat(0.5) * rat(0.3) + rat(3.0) * rat(0.4);
        let required = (exact_center - rat(output.center()[0])).abs()
            + (exact_g0 - rat(coefficient(&output, 0, 0))).abs()
            + (exact_g1 - rat(coefficient(&output, 1, 0))).abs()
            + propagated_remainder;
        assert!(rat(output.box_remainder()[0]) >= required);
    }

    #[test]
    fn budgeted_conv2d_is_bit_identical_and_reports_complete_peak() {
        let input = simple_input();
        let weights = Array4::from_shape_vec((1, 1, 2, 2), vec![1.0, -2.0, 0.5, 3.0]).unwrap();
        let legacy = constrained_zonotope_conv2d_unwired(
            &input,
            [1, 2, 2],
            weights.view(),
            &[0.125],
            spec(),
            limits(),
        )
        .unwrap();
        let deadline = Instant::now() + Duration::from_mins(1);
        let outcome = constrained_zonotope_conv2d_unwired_with_budget(
            &input,
            [1, 2, 2],
            weights.view(),
            &[0.125],
            spec(),
            limits(),
            ConstrainedZonotopeCallBudget::new(deadline, 13, usize::MAX),
        )
        .unwrap();
        assert_eq!(outcome.value(), &legacy);
        // Independent inventory of the fixed case: adjacency count/touched
        // indices, adjacency/candidate/validated column headers, adjacency plus
        // candidate/validated entries, interval scratch, and the
        // center/remainder/constraint/rhs scalars.
        let transform_owned_peak = 6 * size_of::<usize>()
            + 8 * size_of::<Vec<(usize, f64)>>()
            + 7 * size_of::<(usize, f64)>()
            + 2 * size_of::<Option<OutwardInterval>>()
            + 5 * size_of::<f64>();
        assert_eq!(
            outcome.report().peak_live_bytes(),
            13 + transform_owned_peak
        );
        assert!(outcome.report().charged_items() > 0);
        assert!(outcome.report().deadline_polls() > 0);

        let exact_peak = outcome.report().peak_live_bytes();
        constrained_zonotope_conv2d_unwired_with_budget(
            &input,
            [1, 2, 2],
            weights.view(),
            &[0.125],
            spec(),
            limits(),
            ConstrainedZonotopeCallBudget::new(deadline, 13, exact_peak),
        )
        .unwrap();
        assert!(matches!(
            constrained_zonotope_conv2d_unwired_with_budget(
                &input,
                [1, 2, 2],
                weights.view(),
                &[0.125],
                spec(),
                limits(),
                ConstrainedZonotopeCallBudget::new(deadline, 13, exact_peak - 1),
            ),
            Err(ConstrainedZonotopeConv2dBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { .. }
            ))
        ));
        assert!(matches!(
            constrained_zonotope_conv2d_unwired_with_budget(
                &input,
                [1, 2, 2],
                weights.view(),
                &[0.125],
                spec(),
                limits(),
                ConstrainedZonotopeCallBudget::new(deadline, usize::MAX, usize::MAX),
            ),
            Err(ConstrainedZonotopeConv2dBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                    operation: "aggregate peak-live bytes"
                }
            ))
        ));
    }

    #[test]
    fn budget_refuses_baseline_at_admission_and_each_publication_seam() {
        let input = simple_input();
        let weights = Array4::ones((1, 1, 2, 2));
        let start = Instant::now();
        let reads = Cell::new(0_usize);
        let baseline = constrained_zonotope_conv2d_unwired_with_clock(
            &input,
            [1, 2, 2],
            weights.view(),
            &[0.0],
            spec(),
            limits(),
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 5, 4),
            |_| {
                reads.set(reads.get() + 1);
                start
            },
        );
        assert!(matches!(
            baseline,
            Err(ConstrainedZonotopeConv2dBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                    required: 5,
                    limit: 4
                }
            ))
        ));
        assert_eq!(reads.get(), 1);

        for seam in [
            "Conv2d geometry validation complete",
            "Conv2d peak-memory preflight complete",
            "Conv2d adjacency construction complete",
            "Conv2d interval-product preflight complete",
            "Conv2d numeric transform complete",
            "Conv2d constraint clone complete",
            "Conv2d right-hand-side clone complete",
            "constrained-zonotope generator-column allocation",
            "Conv2d domain materialization complete",
            "Conv2d publication",
        ] {
            let expired = start + Duration::from_secs(2);
            let result = constrained_zonotope_conv2d_unwired_with_clock(
                &input,
                [1, 2, 2],
                weights.view(),
                &[0.0],
                spec(),
                limits(),
                ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, 1 << 20),
                |checkpoint| if checkpoint == seam { expired } else { start },
            );
            assert!(
                matches!(
                    result,
                    Err(ConstrainedZonotopeConv2dBudgetError::Budget(
                        ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
                    )) if checkpoint == seam
                ),
                "deadline seam {seam} must refuse"
            );
        }
    }

    #[test]
    fn deadline_polls_inside_product_preflight_and_dense_kernel_loop() {
        let dimension = crate::CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL;
        let input = ConstrainedZonotope64::try_new(
            vec![1.0; dimension],
            Vec::new(),
            Array2::zeros((0, 0)),
            Vec::new(),
            vec![0.0; dimension],
        )
        .unwrap();
        let weights = Array4::ones((1, 1, 1, dimension));
        let mut large = limits();
        large.max_value_count = dimension;
        large.max_weight_elements = dimension;
        large.max_kernel_visits = dimension;
        large.max_interval_products = dimension;
        let start = Instant::now();
        let expired = start + Duration::from_secs(2);
        for phase in [
            "Conv2d interval-product kernel preflight",
            "Conv2d kernel transform",
        ] {
            let result = constrained_zonotope_conv2d_unwired_with_clock(
                &input,
                [1, 1, dimension],
                weights.view(),
                &[0.0],
                spec(),
                large,
                ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, 1 << 20),
                |checkpoint| if checkpoint == phase { expired } else { start },
            );
            assert!(
                matches!(
                    result,
                    Err(ConstrainedZonotopeConv2dBudgetError::Budget(
                        ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
                    )) if checkpoint == phase
                ),
                "deadline must be polled during {phase}"
            );
        }
    }

    #[test]
    fn deadline_polling_continues_through_final_domain_validation() {
        let dimension = crate::CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL;
        let input = ConstrainedZonotope64::try_new(
            vec![1.0],
            Vec::new(),
            Array2::zeros((0, 0)),
            Vec::new(),
            vec![0.0],
        )
        .unwrap();
        let weights = Array4::ones((dimension, 1, 1, 1));
        let bias = vec![0.0; dimension];
        let mut large = limits();
        large.max_value_count = dimension;
        large.max_weight_elements = dimension;
        large.max_kernel_visits = dimension;
        large.max_interval_products = dimension;
        let start = Instant::now();
        let expired = start + Duration::from_secs(2);
        let result = constrained_zonotope_conv2d_unwired_with_clock(
            &input,
            [1, 1, 1],
            weights.view(),
            &bias,
            spec(),
            large,
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
            |checkpoint| {
                if checkpoint == "constrained-zonotope finite-value validation" {
                    expired
                } else {
                    start
                }
            },
        );
        assert!(matches!(
            result,
            Err(ConstrainedZonotopeConv2dBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "constrained-zonotope finite-value validation"
                }
            ))
        ));
    }

    #[test]
    fn deadline_polls_while_cloning_many_zero_width_constraint_rows() {
        let row_count = crate::CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL;
        let input = ConstrainedZonotope64::try_new(
            vec![1.0],
            Vec::new(),
            Array2::zeros((row_count, 0)),
            vec![0.0; row_count],
            vec![0.0],
        )
        .unwrap();
        let weights = Array4::ones((1, 1, 1, 1));
        let mut large = limits();
        large.max_constraint_count = row_count;
        let start = Instant::now();
        let expired = start + Duration::from_secs(2);
        let result = constrained_zonotope_conv2d_unwired_with_clock(
            &input,
            [1, 1, 1],
            weights.view(),
            &[0.0],
            spec(),
            large,
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
            |checkpoint| {
                if checkpoint == "Conv2d constraint-row clone" {
                    expired
                } else {
                    start
                }
            },
        );
        assert!(matches!(
            result,
            Err(ConstrainedZonotopeConv2dBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "Conv2d constraint-row clone"
                }
            ))
        ));
    }

    #[test]
    fn grouped_dilated_padding_uses_onnx_channel_geometry() {
        let input = ConstrainedZonotope64::from_certified_bounds(
            &(0..18).map(f64::from).collect::<Vec<_>>(),
            &(0..18).map(f64::from).collect::<Vec<_>>(),
            &[true; 18],
        )
        .unwrap();
        let weights = Array4::from_shape_vec((2, 1, 2, 2), vec![1.0; 8]).unwrap();
        let grouped = ConstrainedZonotopeConv2dSpec {
            stride: [1, 1],
            padding: [1, 1, 0, 0],
            dilation: [2, 2],
            groups: 2,
        };
        let (output, plan) = constrained_zonotope_conv2d_unwired(
            &input,
            [2, 3, 3],
            weights.view(),
            &[0.0, 0.0],
            grouped,
            limits(),
        )
        .unwrap();
        assert_eq!(plan.output_shape, [2, 2, 2]);
        assert_eq!(output.value_dim(), 8);
        // Bottom-right output in group 0 reads indices 0,2,6,8.
        assert_eq!(output.center()[3], 16.0);
        // The same site in group 1 reads the second input channel.
        assert_eq!(output.center()[7], 52.0);
    }

    #[test]
    fn grouped_strided_asymmetric_padding_and_dilation_match_dense_oracle() {
        let input_shape = [4, 4, 5];
        let input_values = (1..=80).map(f64::from).collect::<Vec<_>>();
        let input =
            ConstrainedZonotope64::from_certified_bounds(&input_values, &input_values, &[true; 80])
                .unwrap();
        let weights = Array4::ones((4, 2, 2, 2));
        let bias = [0.0, 100.0, 200.0, 300.0];
        let grouped = ConstrainedZonotopeConv2dSpec {
            stride: [2, 1],
            padding: [1, 2, 0, 1],
            dilation: [2, 2],
            groups: 2,
        };
        let legacy = constrained_zonotope_conv2d_unwired(
            &input,
            input_shape,
            weights.view(),
            &bias,
            grouped,
            limits(),
        )
        .unwrap();
        assert_eq!(legacy.1.output_shape, [4, 2, 6]);

        let mut valid_products = 0_usize;
        for output_channel in 0..4 {
            let input_channel_base = (output_channel / 2) * 2;
            for output_y in 0..2 {
                for output_x in 0..6 {
                    let output_index = (output_channel * 2 + output_y) * 6 + output_x;
                    let mut exact = rat(bias[output_channel]);
                    for kernel_input_channel in 0..2 {
                        for kernel_y in 0..2 {
                            let input_y = isize::try_from(output_y * 2 + kernel_y * 2).unwrap() - 1;
                            if !(0..4).contains(&input_y) {
                                continue;
                            }
                            for kernel_x in 0..2 {
                                let input_x = isize::try_from(output_x + kernel_x * 2).unwrap() - 2;
                                if !(0..5).contains(&input_x) {
                                    continue;
                                }
                                let input_channel = input_channel_base + kernel_input_channel;
                                let input_index =
                                    (input_channel * 4 + usize::try_from(input_y).unwrap()) * 5
                                        + usize::try_from(input_x).unwrap();
                                exact += rat(input_values[input_index]);
                                valid_products += 1;
                            }
                        }
                    }
                    let represented_error = (exact - rat(legacy.0.center()[output_index])).abs();
                    assert!(
                        rat(legacy.0.box_remainder()[output_index]) >= represented_error,
                        "output {output_index} does not contain the independent dense oracle"
                    );
                }
            }
        }
        assert_eq!(legacy.1.interval_products, valid_products);

        let deadline = Instant::now() + Duration::from_mins(1);
        let budgeted = constrained_zonotope_conv2d_unwired_with_budget(
            &input,
            input_shape,
            weights.view(),
            &bias,
            grouped,
            limits(),
            ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
        )
        .unwrap();
        assert_eq!(budgeted.value(), &legacy);
    }

    #[test]
    fn malformed_shapes_and_resource_caps_fail_closed() {
        let input = ConstrainedZonotope64::from_certified_bounds(&[0.0], &[1.0], &[false]).unwrap();
        let weights = Array4::ones((1, 1, 1, 1));
        let mut tiny = limits();
        tiny.max_interval_products = 0;
        assert!(matches!(
            constrained_zonotope_conv2d_unwired(
                &input,
                [1, 1, 1],
                weights.view(),
                &[0.0],
                spec(),
                tiny,
            ),
            Err(ConstrainedZonotopeConv2dError::ResourceLimit {
                resource: "interval products",
                ..
            })
        ));
        assert!(constrained_zonotope_conv2d_unwired(
            &input,
            [1, 1, 2],
            weights.view(),
            &[0.0],
            spec(),
            limits(),
        )
        .is_err());
        let mut bad_spec = spec();
        bad_spec.groups = 0;
        assert!(constrained_zonotope_conv2d_unwired(
            &input,
            [1, 1, 1],
            weights.view(),
            &[0.0],
            bad_spec,
            limits(),
        )
        .is_err());

        let point_with_zero_width_rows = ConstrainedZonotope64::try_new(
            vec![0.0],
            Vec::new(),
            Array2::zeros((2, 0)),
            vec![0.0, 0.0],
            vec![0.0],
        )
        .unwrap();
        let mut no_constraint_rows = limits();
        no_constraint_rows.max_constraint_count = 0;
        assert!(matches!(
            constrained_zonotope_conv2d_unwired(
                &point_with_zero_width_rows,
                [1, 1, 1],
                weights.view(),
                &[0.0],
                spec(),
                no_constraint_rows,
            ),
            Err(ConstrainedZonotopeConv2dError::ResourceLimit {
                resource: "constraint count",
                ..
            })
        ));

        let wide_kernel = Array4::zeros((1, 1, 1, 4));
        let padded = ConstrainedZonotopeConv2dSpec {
            stride: [1, 1],
            padding: [0, 4, 0, 4],
            dilation: [1, 1],
            groups: 1,
        };
        let mut no_kernel_visits = limits();
        no_kernel_visits.max_kernel_visits = 0;
        assert!(matches!(
            constrained_zonotope_conv2d_unwired(
                &input,
                [1, 1, 1],
                wide_kernel.view(),
                &[0.0],
                padded,
                no_kernel_visits,
            ),
            Err(ConstrainedZonotopeConv2dError::ResourceLimit {
                resource: "kernel visits",
                ..
            })
        ));

        let mut no_weight_validation = limits();
        no_weight_validation.max_weight_elements = 0;
        assert!(matches!(
            constrained_zonotope_conv2d_unwired(
                &input,
                [1, 1, 1],
                weights.view(),
                &[0.0],
                spec(),
                no_weight_validation,
            ),
            Err(ConstrainedZonotopeConv2dError::ResourceLimit {
                resource: "weight elements",
                ..
            })
        ));

        let sparse_input = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![vec![(0, 1.0)], vec![(0, -1.0)]],
            Array2::zeros((0, 2)),
            Vec::new(),
            vec![0.0],
        )
        .unwrap();
        let mut one_generator_nonzero = limits();
        one_generator_nonzero.max_generator_nonzeros = 1;
        assert!(matches!(
            constrained_zonotope_conv2d_unwired(
                &sparse_input,
                [1, 1, 1],
                weights.view(),
                &[0.0],
                spec(),
                one_generator_nonzero,
            ),
            Err(ConstrainedZonotopeConv2dError::ResourceLimit {
                resource: "input generator nonzeros",
                required: 2,
                limit: 1,
            })
        ));
    }

    #[test]
    fn interval_product_preflight_preserves_first_excess_payload() {
        let input = ConstrainedZonotope64::try_new(
            vec![1.0],
            vec![vec![(0, 0.25)], vec![(0, -0.5)]],
            Array2::zeros((0, 2)),
            Vec::new(),
            vec![0.0],
        )
        .unwrap();
        let weights = Array4::ones((1, 1, 1, 1));
        let start = Instant::now();

        for limit in 0..3 {
            let mut capped = limits();
            capped.max_interval_products = limit;
            let legacy = constrained_zonotope_conv2d_unwired(
                &input,
                [1, 1, 1],
                weights.view(),
                &[0.0],
                spec(),
                capped,
            )
            .unwrap_err();
            assert_eq!(
                legacy,
                ConstrainedZonotopeConv2dError::ResourceLimit {
                    resource: "interval products",
                    required: limit + 1,
                    limit,
                }
            );
            let budgeted = constrained_zonotope_conv2d_unwired_with_clock(
                &input,
                [1, 1, 1],
                weights.view(),
                &[0.0],
                spec(),
                capped,
                ConstrainedZonotopeCallBudget::new(start + Duration::from_mins(1), 0, usize::MAX),
                |_| start,
            )
            .unwrap_err();
            assert_eq!(
                budgeted,
                ConstrainedZonotopeConv2dBudgetError::Transform(legacy)
            );
        }

        let mut unlimited_count = usize::MAX - 1;
        assert_eq!(
            consume_product_batch_equivalent(&mut unlimited_count, 2, usize::MAX),
            Err(ConstrainedZonotopeConv2dError::ResourceOverflow {
                operation: "interval product count"
            })
        );
        let mut finitely_capped_count = usize::MAX - 1;
        assert_eq!(
            consume_product_batch_equivalent(&mut finitely_capped_count, 2, usize::MAX - 1,),
            Err(ConstrainedZonotopeConv2dError::ResourceLimit {
                resource: "interval products",
                required: usize::MAX,
                limit: usize::MAX - 1,
            })
        );
    }

    #[test]
    fn legacy_validation_order_and_error_payloads_remain_exact() {
        let input = ConstrainedZonotope64::try_new(
            vec![0.0],
            Vec::new(),
            Array2::zeros((0, 0)),
            Vec::new(),
            vec![0.0],
        )
        .unwrap();
        let weights = Array4::ones((1, 1, 1, 1));
        let mut invalid_groups = spec();
        invalid_groups.groups = 0;
        assert_eq!(
            constrained_zonotope_conv2d_unwired(
                &input,
                [1, 1, 2],
                weights.view(),
                &[0.0],
                invalid_groups,
                limits(),
            ),
            Err(ConstrainedZonotopeConv2dError::Shape {
                field: "input domain",
                expected: vec![2],
                got: vec![1],
            })
        );

        let mut no_input_values = limits();
        no_input_values.max_value_count = 0;
        assert_eq!(
            constrained_zonotope_conv2d_unwired(
                &input,
                [1, 1, 1],
                weights.view(),
                &[0.0],
                invalid_groups,
                no_input_values,
            ),
            Err(ConstrainedZonotopeConv2dError::ResourceLimit {
                resource: "input value count",
                required: 1,
                limit: 0,
            })
        );
        assert_eq!(
            constrained_zonotope_conv2d_unwired(
                &input,
                [1, 1, 1],
                weights.view(),
                &[0.0],
                invalid_groups,
                limits(),
            ),
            Err(ConstrainedZonotopeConv2dError::InvalidSpec {
                message: "groups must be non-zero".to_owned(),
            })
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn adjacent_float_contractions_enclose_exact_rational_formula(
            center in prop::collection::vec(-1000_i16..=1000, 9),
            generator in prop::collection::vec(-1000_i16..=1000, 9),
            remainder in prop::collection::vec(0_u8..=20, 9),
            weights_raw in prop::collection::vec(-1000_i16..=1000, 4),
            bias_raw in -1000_i16..=1000,
        ) {
            let scale = |value: i16| f64::from(value) / 64.0;
            let centers: Vec<f64> = center.into_iter().map(scale).collect();
            let generator_values: Vec<(usize, f64)> = generator
                .into_iter()
                .enumerate()
                .filter_map(|(index, value)| {
                    let value = scale(value);
                    (value != 0.0).then_some((index, value))
                })
                .collect();
            let remainders: Vec<f64> = remainder
                .into_iter()
                .map(|value| f64::from(value) / 4096.0)
                .collect();
            let input = ConstrainedZonotope64::try_new(
                centers,
                vec![generator_values],
                Array2::zeros((0, 1)),
                Vec::new(),
                remainders,
            ).unwrap();
            let weight_values: Vec<f64> = weights_raw.into_iter().map(scale).collect();
            let weights = Array4::from_shape_vec((1, 1, 2, 2), weight_values).unwrap();
            let bias = scale(bias_raw);
            let (output, _) = constrained_zonotope_conv2d_unwired(
                &input,
                [1, 3, 3],
                weights.view(),
                &[bias],
                spec(),
                limits(),
            ).unwrap();

            for output_y in 0..2 {
                for output_x in 0..2 {
                    let output_index = output_y * 2 + output_x;
                    let mut exact_center = rat(bias);
                    let mut exact_generator = BigRational::zero();
                    let mut exact_remainder = BigRational::zero();
                    for kernel_y in 0..2 {
                        for kernel_x in 0..2 {
                            let input_index = (output_y + kernel_y) * 3 + output_x + kernel_x;
                            let weight = weights[[0, 0, kernel_y, kernel_x]];
                            exact_center += rat(weight) * rat(input.center()[input_index]);
                            exact_generator += rat(weight)
                                * rat(coefficient(&input, 0, input_index));
                            exact_remainder += rat(weight).abs()
                                * rat(input.box_remainder()[input_index]);
                        }
                    }
                    let required = (exact_center - rat(output.center()[output_index])).abs()
                        + (exact_generator
                            - rat(coefficient(&output, 0, output_index)))
                        .abs()
                        + exact_remainder;
                    prop_assert!(rat(output.box_remainder()[output_index]) >= required);
                }
            }
        }

        #[test]
        fn mixed_scale_contractions_enclose_exact_rational_formula(
            center_raw in prop::collection::vec((any::<u64>(), -200_i16..=200), 4),
            generator_raw in prop::collection::vec((any::<u64>(), -200_i16..=200), 4),
            remainder_raw in prop::collection::vec((any::<u64>(), -500_i16..=-100), 4),
            weights_raw in prop::collection::vec((any::<u64>(), -200_i16..=200), 4),
            bias_raw in (any::<u64>(), -200_i16..=200),
        ) {
            let centers: Vec<f64> = center_raw
                .into_iter()
                .map(|(raw, exponent)| bounded_normal(raw, exponent))
                .collect();
            let generator_values: Vec<(usize, f64)> = generator_raw
                .into_iter()
                .enumerate()
                .map(|(index, (raw, exponent))| (index, bounded_normal(raw, exponent)))
                .collect();
            let remainders: Vec<f64> = remainder_raw
                .into_iter()
                .map(|(raw, exponent)| bounded_normal(raw & !(1_u64 << 63), exponent))
                .collect();
            let input = ConstrainedZonotope64::try_new(
                centers,
                vec![generator_values],
                Array2::zeros((0, 1)),
                Vec::new(),
                remainders,
            ).unwrap();
            let weight_values: Vec<f64> = weights_raw
                .into_iter()
                .map(|(raw, exponent)| bounded_normal(raw, exponent))
                .collect();
            let weights = Array4::from_shape_vec((1, 1, 2, 2), weight_values).unwrap();
            let bias = bounded_normal(bias_raw.0, bias_raw.1);
            let (output, _) = constrained_zonotope_conv2d_unwired(
                &input,
                [1, 2, 2],
                weights.view(),
                &[bias],
                spec(),
                limits(),
            ).unwrap();

            let mut exact_center = rat(bias);
            let mut exact_generator = BigRational::zero();
            let mut exact_remainder = BigRational::zero();
            for index in 0..4 {
                let kernel_y = index / 2;
                let kernel_x = index % 2;
                let weight = weights[[0, 0, kernel_y, kernel_x]];
                exact_center += rat(weight) * rat(input.center()[index]);
                exact_generator += rat(weight) * rat(coefficient(&input, 0, index));
                exact_remainder += rat(weight).abs() * rat(input.box_remainder()[index]);
            }
            let required = (exact_center - rat(output.center()[0])).abs()
                + (exact_generator - rat(coefficient(&output, 0, 0))).abs()
                + exact_remainder;
            prop_assert!(rat(output.box_remainder()[0]) >= required);
        }
    }
}
