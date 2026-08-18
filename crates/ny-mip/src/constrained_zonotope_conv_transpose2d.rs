// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Outward sparse `f64` transposed convolution for an unwired constrained zonotope.
//!
//! The transform implements ONNX `ConvTranspose` weight layout
//! `[input_channel, output_channel_per_group, kernel_height, kernel_width]`.
//! It preserves sparse generator support, copies the predicate
//! `C alpha <= d`, and charges the image of the input box remainder plus every
//! center/generator rounding interval to the output box remainder.  The result
//! therefore encloses the real transposed convolution of the exact dyadic
//! `f64` values supplied by the caller.
//!
//! This module is deliberately **unwired** and CPU-only.  It does not read ONNX
//! tensors, convert an `f32` abstract state, choose proof authority, run on
//! CUDA, or affect a scored verdict.  A future caller must provide
//! proof-qualified inputs and preserve this fail-closed boundary.

use ndarray::{Array2, ArrayView4};

use crate::constrained_zonotope64::ConstrainedZonotope64CallGateError;
use crate::constrained_zonotope_call_budget::{
    ConstrainedZonotopeCallBudget, ConstrainedZonotopeCallBudgetError, ConstrainedZonotopeCallGate,
    ConstrainedZonotopeCallOutcome, ConstrainedZonotopeCallTracker,
    ConstrainedZonotopePeakLiveBytes, InertConstrainedZonotopeCallGate,
};
use crate::{ConstrainedZonotope64, ConstrainedZonotope64Error};

/// NCHW-without-batch ONNX `ConvTranspose` parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeConvTranspose2dSpec {
    /// Vertical and horizontal stride.
    pub stride: [usize; 2],
    /// Padding in ONNX order: top, left, bottom, right.
    pub padding: [usize; 4],
    /// Vertical and horizontal kernel dilation.
    pub dilation: [usize; 2],
    /// Extra high-side output extent.  NY requires each value to be less than
    /// the corresponding stride.
    pub output_padding: [usize; 2],
    /// ONNX transposed-convolution group count.
    pub groups: usize,
}

/// Explicit resource limits for
/// [`constrained_zonotope_conv_transpose2d_unwired`].
///
/// There is intentionally no `Default`: an experimental caller must choose
/// every cap.  Counts cover memory owned by the returned domain and explicit
/// sparse scratch, but not allocator bookkeeping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeConvTranspose2dLimits {
    /// Maximum input or output scalar value dimension.
    pub max_value_count: usize,
    /// Maximum alpha dimension and per-output dense scratch length.
    pub max_alpha_dim: usize,
    /// Maximum sparse generator nonzeros in either the input or output.
    pub max_generator_nonzeros: usize,
    /// Maximum logical kernel/weight elements inspected for finiteness.
    pub max_weight_elements: usize,
    /// Maximum nested output/kernel visits, including unreachable taps and zeros.
    pub max_kernel_visits: usize,
    /// Maximum interval multiplications performed by the transform.
    pub max_interval_products: usize,
    /// Maximum retained predicate rows, including rows with zero alpha width.
    pub max_constraint_count: usize,
    /// Maximum number of retained `C` matrix elements.
    pub max_constraint_elements: usize,
}

/// Checked shape and work accounting for one completed transposed convolution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeConvTranspose2dPlan {
    /// Input shape `[channels, height, width]`.
    pub input_shape: [usize; 3],
    /// Output shape `[channels, height, width]`.
    pub output_shape: [usize; 3],
    /// ONNX weight shape
    /// `[input_channels, output_channels_per_group, height, width]`.
    pub weight_shape: [usize; 4],
    /// Alpha dimension, unchanged by an affine transform.
    pub alpha_dim: usize,
    /// Constraint count, unchanged by an affine transform.
    pub constraint_count: usize,
    /// Logical weight elements validated before the transform.
    pub weight_elements: usize,
    /// Nested output/kernel visits, including unreachable taps and zeros.
    pub kernel_visits: usize,
    /// Sparse nonzeros in the input generators.
    pub input_generator_nonzeros: usize,
    /// Sparse nonzeros retained in the output generators.
    pub output_generator_nonzeros: usize,
    /// Interval products actually evaluated, including remainder products.
    pub interval_products: usize,
}

/// Invalid transposed-convolution data, exhausted resources, or arithmetic
/// that could not be enclosed by finite `f64` endpoints.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConstrainedZonotopeConvTranspose2dError {
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

    /// A transposed-convolution parameter is structurally invalid.
    #[error("invalid transposed-convolution specification: {message}")]
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
    #[error("internal transposed-convolution invariant violated: {message}")]
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

/// Primitive or call-firewall refusal from budgeted transposed convolution.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConstrainedZonotopeConvTranspose2dBudgetError {
    /// Geometry, resources, or outward arithmetic were invalid.
    #[error(transparent)]
    Transform(#[from] ConstrainedZonotopeConvTranspose2dError),

    /// The caller's deadline or aggregate peak-memory ceiling refused work.
    #[error(transparent)]
    Budget(#[from] ConstrainedZonotopeCallBudgetError),
}

/// Apply an exact-dyadic grouped 2-D transposed convolution while preserving
/// sparse local generator support and charging every floating-point interval
/// width to the box remainder.
///
/// `input_shape` and the output use flattened `[channel, height, width]` order.
/// `weights` use ONNX
/// `[input_channel, output_channel_per_group, kh, kw]` order.  There is no
/// batch axis.  The predicate `C alpha <= d` is copied unchanged because
/// transposed convolution is affine in the represented values.
pub fn constrained_zonotope_conv_transpose2d_unwired(
    input: &ConstrainedZonotope64,
    input_shape: [usize; 3],
    weights: ArrayView4<'_, f64>,
    bias: &[f64],
    spec: ConstrainedZonotopeConvTranspose2dSpec,
    limits: ConstrainedZonotopeConvTranspose2dLimits,
) -> Result<
    (
        ConstrainedZonotope64,
        ConstrainedZonotopeConvTranspose2dPlan,
    ),
    ConstrainedZonotopeConvTranspose2dError,
> {
    let mut gate = InertConstrainedZonotopeCallGate;
    match constrained_zonotope_conv_transpose2d_impl(
        input,
        input_shape,
        weights,
        bias,
        spec,
        limits,
        &mut gate,
    ) {
        Ok(value) => Ok(value),
        Err(ConstrainedZonotopeConvTranspose2dBudgetError::Transform(error)) => Err(error),
        Err(ConstrainedZonotopeConvTranspose2dBudgetError::Budget(_)) => {
            unreachable!("the inert ConvTranspose2d call gate cannot refuse work")
        }
    }
}

/// Apply transposed convolution behind a synchronous call-local execution
/// firewall.
///
/// The complete transform-owned logical peak is preflighted before adjacency,
/// scratch, or output allocation.  `budget.baseline_live_bytes()` must include
/// the input, weights, bias, and any other caller-retained storage sharing the
/// ceiling.  A completed domain remains private until the final deadline
/// checkpoint.
pub fn constrained_zonotope_conv_transpose2d_unwired_with_budget(
    input: &ConstrainedZonotope64,
    input_shape: [usize; 3],
    weights: ArrayView4<'_, f64>,
    bias: &[f64],
    spec: ConstrainedZonotopeConvTranspose2dSpec,
    limits: ConstrainedZonotopeConvTranspose2dLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> Result<
    ConstrainedZonotopeCallOutcome<(
        ConstrainedZonotope64,
        ConstrainedZonotopeConvTranspose2dPlan,
    )>,
    ConstrainedZonotopeConvTranspose2dBudgetError,
> {
    let mut gate = ConstrainedZonotopeCallTracker::from_system_clock(budget)?;
    let value = constrained_zonotope_conv_transpose2d_impl(
        input,
        input_shape,
        weights,
        bias,
        spec,
        limits,
        &mut gate,
    )?;
    let report = gate.report();
    Ok(ConstrainedZonotopeCallOutcome::new(value, report))
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn constrained_zonotope_conv_transpose2d_unwired_with_clock<N>(
    input: &ConstrainedZonotope64,
    input_shape: [usize; 3],
    weights: ArrayView4<'_, f64>,
    bias: &[f64],
    spec: ConstrainedZonotopeConvTranspose2dSpec,
    limits: ConstrainedZonotopeConvTranspose2dLimits,
    budget: ConstrainedZonotopeCallBudget,
    now: N,
) -> Result<
    ConstrainedZonotopeCallOutcome<(
        ConstrainedZonotope64,
        ConstrainedZonotopeConvTranspose2dPlan,
    )>,
    ConstrainedZonotopeConvTranspose2dBudgetError,
>
where
    N: FnMut(&'static str) -> std::time::Instant,
{
    let mut gate = ConstrainedZonotopeCallTracker::with_clock(budget, now)?;
    let value = constrained_zonotope_conv_transpose2d_impl(
        input,
        input_shape,
        weights,
        bias,
        spec,
        limits,
        &mut gate,
    )?;
    let report = gate.report();
    Ok(ConstrainedZonotopeCallOutcome::new(value, report))
}

#[allow(clippy::too_many_arguments)]
fn constrained_zonotope_conv_transpose2d_impl<G>(
    input: &ConstrainedZonotope64,
    input_shape: [usize; 3],
    weights: ArrayView4<'_, f64>,
    bias: &[f64],
    spec: ConstrainedZonotopeConvTranspose2dSpec,
    limits: ConstrainedZonotopeConvTranspose2dLimits,
    gate: &mut G,
) -> Result<
    (
        ConstrainedZonotope64,
        ConstrainedZonotopeConvTranspose2dPlan,
    ),
    ConstrainedZonotopeConvTranspose2dBudgetError,
>
where
    G: ConstrainedZonotopeCallGate,
{
    require_gradual_underflow()?;
    gate.checkpoint("ConvTranspose2d floating-point preflight")?;
    let geometry =
        validate_geometry_with_gate(input, input_shape, weights, bias, spec, limits, gate)?;
    gate.checkpoint("ConvTranspose2d geometry validation complete")?;

    let mut input_generator_nonzeros = 0_usize;
    for generator in input.generators() {
        gate.charge_items(1, "ConvTranspose2d generator geometry validation")?;
        input_generator_nonzeros = input_generator_nonzeros
            .checked_add(generator.nnz())
            .ok_or(ConstrainedZonotopeConvTranspose2dError::ResourceOverflow {
                operation: "input generator nonzeros",
            })?;
        check_limit(
            "input generator nonzeros",
            input_generator_nonzeros,
            limits.max_generator_nonzeros,
        )?;
    }
    if gate.is_enforcing() {
        gate.preflight_peak_live_bytes(conv_transpose2d_peak_live_bytes(
            input,
            geometry,
            input_generator_nonzeros,
            limits,
        )?)?;
    }
    gate.checkpoint("ConvTranspose2d peak-memory preflight complete")?;

    let adjacency = build_input_adjacency(input, input_generator_nonzeros, gate)?;
    gate.checkpoint("ConvTranspose2d adjacency construction complete")?;
    let expected_interval_products = preflight_interval_products_with_gate(
        input,
        input_shape,
        weights,
        spec,
        geometry,
        &adjacency,
        limits.max_interval_products,
        gate,
    )?;
    gate.checkpoint("ConvTranspose2d interval-product preflight complete")?;
    let alpha_dim = input.alpha_dim();
    let mut generator_scratch: Vec<Option<OutwardInterval>> = Vec::new();
    gate.checkpoint("ConvTranspose2d generator-scratch allocation")?;
    try_reserve(
        &mut generator_scratch,
        alpha_dim,
        "generator interval scratch",
    )?;
    for _ in 0..alpha_dim {
        gate.charge_items(1, "ConvTranspose2d generator-scratch initialization")?;
        generator_scratch.push(None);
    }
    let mut touched_generators = Vec::new();
    gate.checkpoint("ConvTranspose2d touched-generator allocation")?;
    try_reserve(
        &mut touched_generators,
        alpha_dim,
        "touched-generator scratch",
    )?;

    let output_value_count = checked_product(
        &geometry.output_shape,
        "transposed-convolution output value count",
    )?;
    let mut output_center = Vec::new();
    let mut output_remainder = Vec::new();
    gate.checkpoint("ConvTranspose2d output-center allocation")?;
    try_reserve(&mut output_center, output_value_count, "output center")?;
    gate.checkpoint("ConvTranspose2d output-remainder allocation")?;
    try_reserve(
        &mut output_remainder,
        output_value_count,
        "output box remainder",
    )?;

    let mut output_generators: Vec<Vec<(usize, f64)>> = Vec::new();
    gate.checkpoint("ConvTranspose2d generator-column allocation")?;
    try_reserve(
        &mut output_generators,
        alpha_dim,
        "output generator columns",
    )?;
    for _ in 0..alpha_dim {
        gate.charge_items(1, "ConvTranspose2d generator-column initialization")?;
        output_generators.push(Vec::new());
    }

    let [input_channels, input_height, input_width] = input_shape;
    let [_weight_input_channels, output_channels_per_group, kernel_height, kernel_width] =
        geometry.weight_shape;
    let [output_channels, output_height, output_width] = geometry.output_shape;
    let input_channels_per_group = input_channels / spec.groups;

    let mut interval_products = 0_usize;
    let mut output_generator_nonzeros = 0_usize;

    for output_channel in 0..output_channels {
        let group = output_channel / output_channels_per_group;
        let kernel_output_channel = output_channel % output_channels_per_group;
        let input_channel_base = group * input_channels_per_group;
        for output_y in 0..output_height {
            for output_x in 0..output_width {
                gate.charge_items(1, "ConvTranspose2d output transform")?;
                let output_index = output_center.len();
                let mut center_sum = OutwardInterval::exact(bias[output_channel]);
                let mut remainder_sum = OutwardInterval::zero();

                for local_input_channel in 0..input_channels_per_group {
                    let input_channel = input_channel_base + local_input_channel;
                    for kernel_y in 0..kernel_height {
                        gate.charge_items(1, "ConvTranspose2d kernel transform")?;
                        let Some(input_y) = input_coordinate(
                            output_y,
                            spec.padding[0],
                            kernel_y,
                            spec.dilation[0],
                            spec.stride[0],
                            input_height,
                        )?
                        else {
                            continue;
                        };
                        for kernel_x in 0..kernel_width {
                            gate.charge_items(1, "ConvTranspose2d kernel transform")?;
                            let Some(input_x) = input_coordinate(
                                output_x,
                                spec.padding[1],
                                kernel_x,
                                spec.dilation[1],
                                spec.stride[1],
                                input_width,
                            )?
                            else {
                                continue;
                            };
                            let weight =
                                weights[[input_channel, kernel_output_channel, kernel_y, kernel_x]];
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
                                gate.charge_items(1, "ConvTranspose2d generator accumulation")?;
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
                    gate.charge_items(1, "ConvTranspose2d generator publication staging")?;
                    let interval = generator_scratch[generator_index].take().ok_or(
                        ConstrainedZonotopeConvTranspose2dError::InvariantViolation {
                            message: "a touched generator has no accumulated interval",
                        },
                    )?;
                    let (coefficient, coefficient_error) = interval.nominal_and_error()?;
                    total_remainder = add_nonnegative_upper(total_remainder, coefficient_error)?;
                    if coefficient != 0.0 {
                        output_generator_nonzeros = output_generator_nonzeros
                            .checked_add(1)
                            .ok_or(ConstrainedZonotopeConvTranspose2dError::ResourceOverflow {
                                operation: "output generator nonzeros",
                            })?;
                        check_limit(
                            "output generator nonzeros",
                            output_generator_nonzeros,
                            limits.max_generator_nonzeros,
                        )?;
                        gate.checkpoint("ConvTranspose2d generator-entry allocation")?;
                        output_generators[generator_index]
                            .try_reserve(1)
                            .map_err(|_| {
                                ConstrainedZonotopeConvTranspose2dError::AllocationFailure {
                                    resource: "output generator coefficients",
                                }
                            })?;
                        output_generators[generator_index].push((output_index, coefficient));
                    }
                }
                touched_generators.clear();

                output_center.push(nominal_center);
                output_remainder.push(total_remainder);
            }
        }
    }
    gate.checkpoint("ConvTranspose2d numeric transform complete")?;

    debug_assert_eq!(output_center.len(), output_value_count);
    debug_assert_eq!(interval_products, expected_interval_products);
    let constraints = clone_constraints(input, gate)?;
    gate.checkpoint("ConvTranspose2d constraint clone complete")?;
    gate.checkpoint("ConvTranspose2d right-hand-side allocation")?;
    let rhs = clone_slice_with_gate(input.rhs(), "constraint right-hand side", gate)?;
    gate.checkpoint("ConvTranspose2d right-hand-side clone complete")?;
    gate.checkpoint("ConvTranspose2d domain materialization")?;
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
            ConstrainedZonotopeConvTranspose2dBudgetError::Transform(
                ConstrainedZonotopeConvTranspose2dError::Domain(error),
            )
        }
        ConstrainedZonotope64CallGateError::Budget(error) => {
            ConstrainedZonotopeConvTranspose2dBudgetError::Budget(error)
        }
    })?;
    gate.checkpoint("ConvTranspose2d domain materialization complete")?;
    let plan = ConstrainedZonotopeConvTranspose2dPlan {
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
    gate.checkpoint("ConvTranspose2d publication")?;
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

/// Conservative transform-owned peak.  Scratch from disjoint phases is summed
/// deliberately.  Candidate generator buffers and `try_new`'s validated
/// generator buffers are both counted because their materialization can
/// overlap.  The retained input, weights, and bias belong in the caller's
/// baseline.
fn conv_transpose2d_peak_live_bytes(
    input: &ConstrainedZonotope64,
    geometry: Geometry,
    input_generator_nonzeros: usize,
    limits: ConstrainedZonotopeConvTranspose2dLimits,
) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    let output_generator_slots = input
        .alpha_dim()
        .checked_mul(geometry.output_value_count)
        .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
            operation: "ConvTranspose2d output generator slots",
        })?;
    let output_generator_nonzeros = output_generator_slots
        .min(limits.max_generator_nonzeros)
        .min(limits.max_interval_products);

    let mut peak = ConstrainedZonotopePeakLiveBytes::new();
    peak.add_elements::<usize>(input.value_dim(), "ConvTranspose2d adjacency-count bytes")?;
    peak.add_elements::<Vec<(usize, f64)>>(
        input.value_dim(),
        "ConvTranspose2d adjacency-column bytes",
    )?;
    peak.add_elements::<(usize, f64)>(
        input_generator_nonzeros,
        "ConvTranspose2d adjacency-entry bytes",
    )?;
    peak.add_elements::<Option<OutwardInterval>>(
        input.alpha_dim(),
        "ConvTranspose2d generator-scratch bytes",
    )?;
    peak.add_elements::<usize>(input.alpha_dim(), "ConvTranspose2d touched-generator bytes")?;
    peak.add_elements::<f64>(
        geometry.output_value_count,
        "ConvTranspose2d output-center bytes",
    )?;
    peak.add_elements::<f64>(
        geometry.output_value_count,
        "ConvTranspose2d output-remainder bytes",
    )?;
    peak.add_elements::<Vec<(usize, f64)>>(
        input.alpha_dim(),
        "ConvTranspose2d candidate generator-column bytes",
    )?;
    peak.add_elements::<(usize, f64)>(
        output_generator_nonzeros,
        "ConvTranspose2d candidate generator-entry bytes",
    )?;
    peak.add_elements::<f64>(
        geometry.constraint_elements,
        "ConvTranspose2d constraint-matrix bytes",
    )?;
    peak.add_elements::<f64>(
        input.constraint_count(),
        "ConvTranspose2d constraint right-hand-side bytes",
    )?;
    peak.add_elements::<Vec<(usize, f64)>>(
        input.alpha_dim(),
        "ConvTranspose2d validated generator-column bytes",
    )?;
    peak.add_elements::<(usize, f64)>(
        output_generator_nonzeros,
        "ConvTranspose2d validated generator-entry bytes",
    )?;
    Ok(peak.finish())
}

#[allow(clippy::too_many_arguments)]
fn validate_geometry_with_gate<G>(
    input: &ConstrainedZonotope64,
    input_shape: [usize; 3],
    weights: ArrayView4<'_, f64>,
    bias: &[f64],
    spec: ConstrainedZonotopeConvTranspose2dSpec,
    limits: ConstrainedZonotopeConvTranspose2dLimits,
    gate: &mut G,
) -> Result<Geometry, ConstrainedZonotopeConvTranspose2dBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let input_value_count =
        checked_product(&input_shape, "transposed-convolution input value count")?;
    if input.value_dim() != input_value_count {
        return Err(ConstrainedZonotopeConvTranspose2dError::Shape {
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
        .ok_or(ConstrainedZonotopeConvTranspose2dError::ResourceOverflow {
            operation: "constraint matrix elements",
        })?;
    check_limit(
        "constraint matrix elements",
        constraint_elements,
        limits.max_constraint_elements,
    )?;

    let [input_channels, input_height, input_width] = input_shape;
    if input_channels == 0 || input_height == 0 || input_width == 0 {
        return Err(ConstrainedZonotopeConvTranspose2dError::InvalidSpec {
            message: format!("input shape must be non-empty, got {input_shape:?}"),
        }
        .into());
    }
    if spec.groups == 0 {
        return Err(ConstrainedZonotopeConvTranspose2dError::InvalidSpec {
            message: "groups must be non-zero".to_string(),
        }
        .into());
    }
    if spec.stride.contains(&0) || spec.dilation.contains(&0) {
        return Err(ConstrainedZonotopeConvTranspose2dError::InvalidSpec {
            message: format!(
                "stride and dilation must be non-zero, got {:?} and {:?}",
                spec.stride, spec.dilation
            ),
        }
        .into());
    }
    if spec.output_padding[0] >= spec.stride[0] || spec.output_padding[1] >= spec.stride[1] {
        return Err(ConstrainedZonotopeConvTranspose2dError::InvalidSpec {
            message: format!(
                "output_padding {:?} must be less than stride {:?} per dimension",
                spec.output_padding, spec.stride
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
    let [weight_input_channels, output_channels_per_group, kernel_height, kernel_width] =
        weight_shape;
    if weight_input_channels == 0
        || output_channels_per_group == 0
        || kernel_height == 0
        || kernel_width == 0
    {
        return Err(ConstrainedZonotopeConvTranspose2dError::InvalidSpec {
            message: format!("weight shape must be non-empty, got {weight_shape:?}"),
        }
        .into());
    }
    if weight_input_channels != input_channels {
        return Err(ConstrainedZonotopeConvTranspose2dError::Shape {
            field: "weight input channels",
            expected: vec![input_channels],
            got: vec![weight_input_channels],
        }
        .into());
    }
    if input_channels % spec.groups != 0 {
        return Err(ConstrainedZonotopeConvTranspose2dError::InvalidSpec {
            message: format!(
                "input channels {input_channels} must be divisible by groups {}",
                spec.groups
            ),
        }
        .into());
    }
    let output_channels = output_channels_per_group.checked_mul(spec.groups).ok_or(
        ConstrainedZonotopeConvTranspose2dError::ResourceOverflow {
            operation: "transposed-convolution output channels",
        },
    )?;
    if bias.len() != output_channels {
        return Err(ConstrainedZonotopeConvTranspose2dError::Shape {
            field: "bias",
            expected: vec![output_channels],
            got: vec![bias.len()],
        }
        .into());
    }

    let weight_elements = checked_product(&weight_shape, "transposed-convolution weight elements")?;
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
        spec.output_padding[0],
        "transposed-convolution output height",
    )?;
    let output_width = output_dimension(
        input_width,
        spec.padding[1],
        spec.padding[3],
        kernel_width,
        spec.dilation[1],
        spec.stride[1],
        spec.output_padding[1],
        "transposed-convolution output width",
    )?;
    let output_shape = [output_channels, output_height, output_width];
    let output_value_count =
        checked_product(&output_shape, "transposed-convolution output value count")?;
    check_limit(
        "output value count",
        output_value_count,
        limits.max_value_count,
    )?;

    let input_channels_per_group = input_channels / spec.groups;
    let kernel_visits = checked_product(
        &[
            output_value_count,
            input_channels_per_group,
            kernel_height,
            kernel_width,
        ],
        "transposed-convolution kernel visits",
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

#[allow(clippy::too_many_arguments)]
fn output_dimension(
    input: usize,
    padding_before: usize,
    padding_after: usize,
    kernel: usize,
    dilation: usize,
    stride: usize,
    output_padding: usize,
    operation: &'static str,
) -> Result<usize, ConstrainedZonotopeConvTranspose2dError> {
    let effective_kernel = kernel
        .checked_sub(1)
        .and_then(|value| value.checked_mul(dilation))
        .and_then(|value| value.checked_add(1))
        .ok_or(ConstrainedZonotopeConvTranspose2dError::ResourceOverflow { operation })?;
    let expanded = input
        .checked_sub(1)
        .and_then(|value| value.checked_mul(stride))
        .and_then(|value| value.checked_add(effective_kernel))
        .and_then(|value| value.checked_add(output_padding))
        .ok_or(ConstrainedZonotopeConvTranspose2dError::ResourceOverflow { operation })?;
    let total_padding = padding_before
        .checked_add(padding_after)
        .ok_or(ConstrainedZonotopeConvTranspose2dError::ResourceOverflow { operation })?;
    if expanded <= total_padding {
        return Err(ConstrainedZonotopeConvTranspose2dError::InvalidSpec {
            message: format!(
                "{operation} has expanded extent {expanded} no larger than total padding {total_padding}"
            ),
        });
    }
    Ok(expanded - total_padding)
}

fn input_coordinate(
    output: usize,
    padding_before: usize,
    kernel: usize,
    dilation: usize,
    stride: usize,
    input_size: usize,
) -> Result<Option<usize>, ConstrainedZonotopeConvTranspose2dError> {
    let padded_output = output.checked_add(padding_before).ok_or(
        ConstrainedZonotopeConvTranspose2dError::ResourceOverflow {
            operation: "transposed-convolution padded output coordinate",
        },
    )?;
    let kernel_offset = kernel.checked_mul(dilation).ok_or(
        ConstrainedZonotopeConvTranspose2dError::ResourceOverflow {
            operation: "transposed-convolution kernel coordinate",
        },
    )?;
    if padded_output < kernel_offset {
        return Ok(None);
    }
    let expanded_input = padded_output - kernel_offset;
    if !expanded_input.is_multiple_of(stride) {
        return Ok(None);
    }
    let coordinate = expanded_input / stride;
    Ok((coordinate < input_size).then_some(coordinate))
}

/// Count every interval product before output allocation or floating-point
/// contraction.  This makes the caller's work cap a true preflight boundary,
/// rather than discovering an exhausted budget after a partial transform.
fn preflight_interval_products_with_gate<G>(
    input: &ConstrainedZonotope64,
    input_shape: [usize; 3],
    weights: ArrayView4<'_, f64>,
    spec: ConstrainedZonotopeConvTranspose2dSpec,
    geometry: Geometry,
    adjacency: &[Vec<(usize, f64)>],
    limit: usize,
    gate: &mut G,
) -> Result<usize, ConstrainedZonotopeConvTranspose2dBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let [input_channels, input_height, input_width] = input_shape;
    let [_weight_input_channels, output_channels_per_group, kernel_height, kernel_width] =
        geometry.weight_shape;
    let [output_channels, output_height, output_width] = geometry.output_shape;
    let input_channels_per_group = input_channels / spec.groups;
    let mut products = 0_usize;

    for output_channel in 0..output_channels {
        let group = output_channel / output_channels_per_group;
        let kernel_output_channel = output_channel % output_channels_per_group;
        let input_channel_base = group * input_channels_per_group;
        for output_y in 0..output_height {
            for output_x in 0..output_width {
                gate.charge_items(1, "ConvTranspose2d interval-product preflight")?;
                for local_input_channel in 0..input_channels_per_group {
                    let input_channel = input_channel_base + local_input_channel;
                    for kernel_y in 0..kernel_height {
                        gate.charge_items(1, "ConvTranspose2d interval-product preflight")?;
                        let Some(input_y) = input_coordinate(
                            output_y,
                            spec.padding[0],
                            kernel_y,
                            spec.dilation[0],
                            spec.stride[0],
                            input_height,
                        )?
                        else {
                            continue;
                        };
                        for kernel_x in 0..kernel_width {
                            gate.charge_items(1, "ConvTranspose2d interval-product preflight")?;
                            let Some(input_x) = input_coordinate(
                                output_x,
                                spec.padding[1],
                                kernel_x,
                                spec.dilation[1],
                                spec.stride[1],
                                input_width,
                            )?
                            else {
                                continue;
                            };
                            let weight =
                                weights[[input_channel, kernel_output_channel, kernel_y, kernel_x]];
                            if weight == 0.0 {
                                continue;
                            }
                            let input_index =
                                (input_channel * input_height + input_y) * input_width + input_x;
                            let products_here = 1_usize
                                .checked_add(usize::from(input.box_remainder()[input_index] != 0.0))
                                .and_then(|count| count.checked_add(adjacency[input_index].len()))
                                .ok_or(
                                    ConstrainedZonotopeConvTranspose2dError::ResourceOverflow {
                                        operation: "interval product count",
                                    },
                                )?;
                            products = products.checked_add(products_here).ok_or(
                                ConstrainedZonotopeConvTranspose2dError::ResourceOverflow {
                                    operation: "interval product count",
                                },
                            )?;
                            check_limit("interval products", products, limit)?;
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
) -> Result<Vec<Vec<(usize, f64)>>, ConstrainedZonotopeConvTranspose2dBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let mut counts = Vec::new();
    gate.checkpoint("ConvTranspose2d adjacency-count allocation")?;
    try_reserve(&mut counts, input.value_dim(), "input adjacency counts")?;
    for _ in 0..input.value_dim() {
        gate.charge_items(1, "ConvTranspose2d adjacency-count initialization")?;
        counts.push(0_usize);
    }
    for generator in input.generators() {
        gate.charge_items(1, "ConvTranspose2d adjacency counting")?;
        for (value_index, _) in generator.entries() {
            gate.charge_items(1, "ConvTranspose2d adjacency counting")?;
            counts[value_index] = counts[value_index].checked_add(1).ok_or(
                ConstrainedZonotopeConvTranspose2dError::ResourceOverflow {
                    operation: "per-value generator adjacency",
                },
            )?;
        }
    }

    let mut adjacency = Vec::new();
    gate.checkpoint("ConvTranspose2d adjacency-column allocation")?;
    try_reserve(
        &mut adjacency,
        input.value_dim(),
        "input generator adjacency",
    )?;
    for &count in &counts {
        gate.charge_items(1, "ConvTranspose2d adjacency-column construction")?;
        let mut entries = Vec::new();
        gate.checkpoint("ConvTranspose2d adjacency-entry allocation")?;
        try_reserve(&mut entries, count, "input generator adjacency entries")?;
        adjacency.push(entries);
    }
    let mut filled_nonzeros = 0_usize;
    for (generator_index, generator) in input.generators().iter().enumerate() {
        gate.charge_items(1, "ConvTranspose2d adjacency fill")?;
        for (value_index, coefficient) in generator.entries() {
            gate.charge_items(1, "ConvTranspose2d adjacency fill")?;
            adjacency[value_index].push((generator_index, coefficient));
            filled_nonzeros = filled_nonzeros.checked_add(1).ok_or(
                ConstrainedZonotopeConvTranspose2dError::ResourceOverflow {
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
) -> Result<Array2<f64>, ConstrainedZonotopeConvTranspose2dBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let shape = (input.constraint_count(), input.alpha_dim());
    let element_count = shape.0.checked_mul(shape.1).ok_or(
        ConstrainedZonotopeConvTranspose2dError::ResourceOverflow {
            operation: "constraint matrix elements",
        },
    )?;
    let constraints = input.constraints();
    let mut values = Vec::new();
    gate.checkpoint("ConvTranspose2d constraint-matrix allocation")?;
    try_reserve(&mut values, element_count, "constraint matrix")?;
    for row in 0..shape.0 {
        gate.charge_items(1, "ConvTranspose2d constraint-matrix clone")?;
        for column in 0..shape.1 {
            gate.charge_items(1, "ConvTranspose2d constraint-matrix clone")?;
            values.push(constraints[[row, column]]);
        }
    }
    Array2::from_shape_vec(shape, values).map_err(|_| {
        ConstrainedZonotopeConvTranspose2dError::ResourceOverflow {
            operation: "constraint matrix shape",
        }
        .into()
    })
}

fn clone_slice_with_gate<T: Copy, G>(
    source: &[T],
    resource: &'static str,
    gate: &mut G,
) -> Result<Vec<T>, ConstrainedZonotopeConvTranspose2dBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let mut output = Vec::new();
    try_reserve(&mut output, source.len(), resource)?;
    for &value in source {
        gate.charge_items(1, "ConvTranspose2d right-hand-side clone")?;
        output.push(value);
    }
    Ok(output)
}

fn validate_finite_with_gate<G>(
    field: &'static str,
    values: impl IntoIterator<Item = f64>,
    gate: &mut G,
) -> Result<(), ConstrainedZonotopeConvTranspose2dBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    for (index, value) in values.into_iter().enumerate() {
        gate.charge_items(1, "ConvTranspose2d finite-parameter validation")?;
        if !value.is_finite() {
            return Err(ConstrainedZonotopeConvTranspose2dError::NonFinite { field, index }.into());
        }
    }
    Ok(())
}

fn checked_product(
    dimensions: &[usize],
    operation: &'static str,
) -> Result<usize, ConstrainedZonotopeConvTranspose2dError> {
    dimensions.iter().try_fold(1_usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or(ConstrainedZonotopeConvTranspose2dError::ResourceOverflow { operation })
    })
}

fn check_limit(
    resource: &'static str,
    required: usize,
    limit: usize,
) -> Result<(), ConstrainedZonotopeConvTranspose2dError> {
    if required > limit {
        return Err(ConstrainedZonotopeConvTranspose2dError::ResourceLimit {
            resource,
            required,
            limit,
        });
    }
    Ok(())
}

fn consume_product(
    count: &mut usize,
    limit: usize,
) -> Result<(), ConstrainedZonotopeConvTranspose2dError> {
    *count =
        count
            .checked_add(1)
            .ok_or(ConstrainedZonotopeConvTranspose2dError::ResourceOverflow {
                operation: "interval product count",
            })?;
    check_limit("interval products", *count, limit)
}

fn try_reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    resource: &'static str,
) -> Result<(), ConstrainedZonotopeConvTranspose2dError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| ConstrainedZonotopeConvTranspose2dError::AllocationFailure { resource })
}

/// Reject FTZ/DAZ before adjacent-float intervals are used as proof objects.
fn require_gradual_underflow() -> Result<(), ConstrainedZonotopeConvTranspose2dError> {
    let half = std::hint::black_box(0.5_f64);
    let min_normal = std::hint::black_box(f64::MIN_POSITIVE);
    let min_subnormal = std::hint::black_box(f64::from_bits(1));
    let two_subnormals = std::hint::black_box(f64::from_bits(2));
    if std::hint::black_box(min_normal * half).to_bits() != 0x0008_0000_0000_0000
        || std::hint::black_box(two_subnormals * half).to_bits() != 1
        || std::hint::black_box(min_subnormal + min_subnormal).to_bits() != 2
    {
        return Err(
            ConstrainedZonotopeConvTranspose2dError::UnsupportedFloatingPoint {
                requirement: "IEEE-754 binary64 gradual underflow (FTZ/DAZ disabled)",
            },
        );
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

    fn add(self, rhs: Self) -> Result<Self, ConstrainedZonotopeConvTranspose2dError> {
        if self.is_exact_zero() {
            return Ok(rhs);
        }
        if rhs.is_exact_zero() {
            return Ok(self);
        }
        Ok(Self {
            lo: round_down(self.lo + rhs.lo, "transposed-convolution interval sum")?,
            hi: round_up(self.hi + rhs.hi, "transposed-convolution interval sum")?,
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

    fn nominal_and_error(self) -> Result<(f64, f64), ConstrainedZonotopeConvTranspose2dError> {
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
) -> Result<OutwardInterval, ConstrainedZonotopeConvTranspose2dError> {
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
        lo: round_down(product, "transposed-convolution interval product")?,
        hi: round_up(product, "transposed-convolution interval product")?,
    })
}

fn upper_difference(
    upper: f64,
    lower: f64,
) -> Result<f64, ConstrainedZonotopeConvTranspose2dError> {
    if upper == lower {
        return Ok(0.0);
    }
    round_up(upper - lower, "transposed-convolution representation error")
}

fn add_nonnegative_upper(
    left: f64,
    right: f64,
) -> Result<f64, ConstrainedZonotopeConvTranspose2dError> {
    if left == 0.0 {
        return Ok(right);
    }
    if right == 0.0 {
        return Ok(left);
    }
    round_up(left + right, "transposed-convolution box remainder sum")
}

fn round_down(
    value: f64,
    operation: &'static str,
) -> Result<f64, ConstrainedZonotopeConvTranspose2dError> {
    if !value.is_finite() {
        return Err(ConstrainedZonotopeConvTranspose2dError::NonFiniteArithmetic { operation });
    }
    let outward = value.next_down();
    if !outward.is_finite() {
        return Err(ConstrainedZonotopeConvTranspose2dError::NonFiniteArithmetic { operation });
    }
    Ok(outward)
}

fn round_up(
    value: f64,
    operation: &'static str,
) -> Result<f64, ConstrainedZonotopeConvTranspose2dError> {
    if !value.is_finite() {
        return Err(ConstrainedZonotopeConvTranspose2dError::NonFiniteArithmetic { operation });
    }
    let outward = value.next_up();
    if !outward.is_finite() {
        return Err(ConstrainedZonotopeConvTranspose2dError::NonFiniteArithmetic { operation });
    }
    Ok(outward)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use ndarray::{array, Array2, Array4};
    use num_rational::BigRational;
    use num_traits::{Signed, Zero};

    use super::*;

    fn limits() -> ConstrainedZonotopeConvTranspose2dLimits {
        ConstrainedZonotopeConvTranspose2dLimits {
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

    fn spec() -> ConstrainedZonotopeConvTranspose2dSpec {
        ConstrainedZonotopeConvTranspose2dSpec {
            stride: [1, 1],
            padding: [0, 0, 0, 0],
            dilation: [1, 1],
            output_padding: [0, 0],
            groups: 1,
        }
    }

    fn rat(value: f64) -> BigRational {
        BigRational::from_float(value).expect("finite test value")
    }

    fn coefficient(domain: &ConstrainedZonotope64, generator: usize, value: usize) -> f64 {
        domain.generators()[generator]
            .entries()
            .find_map(|(index, coefficient)| (index == value).then_some(coefficient))
            .unwrap_or(0.0)
    }

    struct ExactImage {
        center: Vec<BigRational>,
        generators: Vec<Vec<BigRational>>,
        remainder: Vec<BigRational>,
    }

    /// Independent exact-rational scatter oracle for ONNX ConvTranspose.
    fn exact_scatter(
        input: &ConstrainedZonotope64,
        input_shape: [usize; 3],
        weights: ArrayView4<'_, f64>,
        bias: &[f64],
        spec: ConstrainedZonotopeConvTranspose2dSpec,
        output_shape: [usize; 3],
    ) -> ExactImage {
        let [input_channels, input_height, input_width] = input_shape;
        let [output_channels, output_height, output_width] = output_shape;
        let output_spatial = output_height * output_width;
        let output_value_count = output_channels * output_spatial;
        let output_channels_per_group = weights.shape()[1];
        let input_channels_per_group = input_channels / spec.groups;
        let kernel_height = weights.shape()[2];
        let kernel_width = weights.shape()[3];

        let mut center = Vec::with_capacity(output_value_count);
        for &bias_value in bias {
            for _ in 0..output_spatial {
                center.push(rat(bias_value));
            }
        }
        let mut generators = vec![vec![BigRational::zero(); output_value_count]; input.alpha_dim()];
        let mut remainder = vec![BigRational::zero(); output_value_count];

        for input_channel in 0..input_channels {
            let group = input_channel / input_channels_per_group;
            for input_y in 0..input_height {
                for input_x in 0..input_width {
                    let input_index =
                        (input_channel * input_height + input_y) * input_width + input_x;
                    for local_output_channel in 0..output_channels_per_group {
                        let output_channel =
                            group * output_channels_per_group + local_output_channel;
                        for kernel_y in 0..kernel_height {
                            let padded_y = input_y * spec.stride[0] + kernel_y * spec.dilation[0];
                            let Some(output_y) = padded_y.checked_sub(spec.padding[0]) else {
                                continue;
                            };
                            if output_y >= output_height {
                                continue;
                            }
                            for kernel_x in 0..kernel_width {
                                let padded_x =
                                    input_x * spec.stride[1] + kernel_x * spec.dilation[1];
                                let Some(output_x) = padded_x.checked_sub(spec.padding[1]) else {
                                    continue;
                                };
                                if output_x >= output_width {
                                    continue;
                                }
                                let output_index = (output_channel * output_height + output_y)
                                    * output_width
                                    + output_x;
                                let weight = weights
                                    [[input_channel, local_output_channel, kernel_y, kernel_x]];
                                center[output_index] +=
                                    rat(weight) * rat(input.center()[input_index]);
                                remainder[output_index] +=
                                    rat(weight).abs() * rat(input.box_remainder()[input_index]);
                                for (generator, exact_column) in generators.iter_mut().enumerate() {
                                    exact_column[output_index] += rat(weight)
                                        * rat(coefficient(input, generator, input_index));
                                }
                            }
                        }
                    }
                }
            }
        }
        ExactImage {
            center,
            generators,
            remainder,
        }
    }

    fn assert_encloses_exact(output: &ConstrainedZonotope64, exact: &ExactImage) {
        assert_eq!(output.value_dim(), exact.center.len());
        for value in 0..output.value_dim() {
            let mut required = (&exact.center[value] - rat(output.center()[value])).abs()
                + &exact.remainder[value];
            for generator in 0..output.alpha_dim() {
                required += (&exact.generators[generator][value]
                    - rat(coefficient(output, generator, value)))
                .abs();
            }
            assert!(
                rat(output.box_remainder()[value]) >= required,
                "value {value}: stored radius {} does not enclose exact requirement {required}",
                output.box_remainder()[value]
            );
        }
    }

    #[test]
    fn ordinary_exact_dyadic_transform_preserves_constraints_and_encloses() {
        let input = ConstrainedZonotope64::try_new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![vec![(0, 0.25), (3, -0.5)], vec![(1, 0.75)]],
            array![[1.0, -0.25]],
            vec![0.5],
            vec![0.125, 0.25, 0.375, 0.5],
        )
        .unwrap();
        let weights = Array4::from_shape_vec((1, 1, 2, 2), vec![1.0, -2.0, 0.5, 3.0]).unwrap();
        let bias = [0.125];
        let (output, plan) = constrained_zonotope_conv_transpose2d_unwired(
            &input,
            [1, 2, 2],
            weights.view(),
            &bias,
            spec(),
            limits(),
        )
        .unwrap();

        assert_eq!(plan.output_shape, [1, 3, 3]);
        assert_eq!(plan.weight_shape, [1, 1, 2, 2]);
        assert_eq!(output.constraints(), input.constraints());
        assert_eq!(output.rhs(), input.rhs());
        let exact = exact_scatter(
            &input,
            [1, 2, 2],
            weights.view(),
            &bias,
            spec(),
            plan.output_shape,
        );
        assert_encloses_exact(&output, &exact);

        // The center pixel receives all four input/tap pairs.  This catches a
        // mistaken ordinary-Conv gather or a flipped kernel.
        assert_eq!(exact.center[4], rat(2.125));
    }

    #[test]
    fn grouped_strided_asymmetric_padding_dilation_and_output_padding_match_onnx() {
        let center: Vec<f64> = (1..=16).map(f64::from).collect();
        let input = ConstrainedZonotope64::try_new(
            center,
            vec![
                vec![(0, 0.5), (5, -0.25), (10, 0.75), (15, -1.0)],
                vec![(3, -0.125), (12, 0.375)],
            ],
            Array2::zeros((0, 2)),
            Vec::new(),
            (0..16)
                .map(|index| f64::from((index % 4) as u8) / 64.0)
                .collect(),
        )
        .unwrap();
        let weights = Array4::from_shape_vec(
            (4, 2, 2, 2),
            (1..=32).map(|value| f64::from(value) / 8.0).collect(),
        )
        .unwrap();
        let grouped = ConstrainedZonotopeConvTranspose2dSpec {
            stride: [2, 3],
            padding: [1, 0, 0, 1],
            dilation: [2, 1],
            output_padding: [1, 2],
            groups: 2,
        };
        let bias = [0.25, -0.5, 1.0, -1.25];
        let (output, plan) = constrained_zonotope_conv_transpose2d_unwired(
            &input,
            [4, 2, 2],
            weights.view(),
            &bias,
            grouped,
            limits(),
        )
        .unwrap();

        assert_eq!(plan.output_shape, [4, 5, 6]);
        assert_eq!(plan.kernel_visits, 4 * 5 * 6 * 2 * 2 * 2);
        let exact = exact_scatter(
            &input,
            [4, 2, 2],
            weights.view(),
            &bias,
            grouped,
            plan.output_shape,
        );
        assert_encloses_exact(&output, &exact);

        // High-side output-padding cells are represented, while group 0 output
        // channels never receive the much larger group 1 input channels.
        assert_eq!(exact.center.len(), 4 * 5 * 6);
        assert!(exact.center[..2 * 5 * 6]
            .iter()
            .all(|value| value < &rat(100.0)));
        assert!(exact.center[2 * 5 * 6..]
            .iter()
            .any(|value| value > &rat(100.0)));
    }

    #[test]
    fn output_padding_extends_the_high_side_with_onnx_scatter_values() {
        let input = ConstrainedZonotope64::from_certified_bounds(&[3.0], &[3.0], &[true]).unwrap();
        let weights = Array4::from_shape_vec((1, 1, 1, 3), vec![1.0, 2.0, 4.0]).unwrap();
        let padded = ConstrainedZonotopeConvTranspose2dSpec {
            stride: [1, 2],
            padding: [0, 1, 0, 1],
            dilation: [1, 1],
            output_padding: [0, 1],
            groups: 1,
        };
        let (output, plan) = constrained_zonotope_conv_transpose2d_unwired(
            &input,
            [1, 1, 1],
            weights.view(),
            &[0.25],
            padded,
            limits(),
        )
        .unwrap();

        assert_eq!(plan.output_shape, [1, 1, 2]);
        // kx=1 lands at x=0 and kx=2 lands in the high-side extent restored
        // by output_padding.  The latter is a real contribution, not a zero
        // cell, under ONNX/PyTorch ConvTranspose geometry.
        let exact = exact_scatter(
            &input,
            [1, 1, 1],
            weights.view(),
            &[0.25],
            padded,
            plan.output_shape,
        );
        assert_eq!(exact.center, vec![rat(6.25), rat(12.25)]);
        assert_encloses_exact(&output, &exact);
    }

    #[test]
    fn malformed_nonfinite_and_resource_requests_fail_closed() {
        let input = ConstrainedZonotope64::from_certified_bounds(&[0.0], &[1.0], &[false]).unwrap();
        let weights = Array4::ones((1, 1, 1, 1));

        assert!(matches!(
            constrained_zonotope_conv_transpose2d_unwired(
                &input,
                [1, 1, 2],
                weights.view(),
                &[0.0],
                spec(),
                limits(),
            ),
            Err(ConstrainedZonotopeConvTranspose2dError::Shape {
                field: "input domain",
                ..
            })
        ));

        let bad_weight = Array4::from_shape_vec((1, 1, 1, 1), vec![f64::INFINITY]).unwrap();
        assert!(matches!(
            constrained_zonotope_conv_transpose2d_unwired(
                &input,
                [1, 1, 1],
                bad_weight.view(),
                &[0.0],
                spec(),
                limits(),
            ),
            Err(ConstrainedZonotopeConvTranspose2dError::NonFinite {
                field: "weights",
                index: 0
            })
        ));
        assert!(matches!(
            constrained_zonotope_conv_transpose2d_unwired(
                &input,
                [1, 1, 1],
                weights.view(),
                &[f64::NAN],
                spec(),
                limits(),
            ),
            Err(ConstrainedZonotopeConvTranspose2dError::NonFinite {
                field: "bias",
                index: 0
            })
        ));

        let mut bad_spec = spec();
        bad_spec.groups = 0;
        assert!(matches!(
            constrained_zonotope_conv_transpose2d_unwired(
                &input,
                [1, 1, 1],
                weights.view(),
                &[0.0],
                bad_spec,
                limits(),
            ),
            Err(ConstrainedZonotopeConvTranspose2dError::InvalidSpec { .. })
        ));
        bad_spec = spec();
        bad_spec.stride = [2, 1];
        bad_spec.output_padding = [2, 0];
        assert!(matches!(
            constrained_zonotope_conv_transpose2d_unwired(
                &input,
                [1, 1, 1],
                weights.view(),
                &[0.0],
                bad_spec,
                limits(),
            ),
            Err(ConstrainedZonotopeConvTranspose2dError::InvalidSpec { .. })
        ));
        bad_spec = spec();
        bad_spec.padding = [1, 0, 1, 0];
        assert!(matches!(
            constrained_zonotope_conv_transpose2d_unwired(
                &input,
                [1, 1, 1],
                weights.view(),
                &[0.0],
                bad_spec,
                limits(),
            ),
            Err(ConstrainedZonotopeConvTranspose2dError::InvalidSpec { .. })
        ));

        let mut tiny = limits();
        tiny.max_weight_elements = 0;
        assert!(matches!(
            constrained_zonotope_conv_transpose2d_unwired(
                &input,
                [1, 1, 1],
                weights.view(),
                &[0.0],
                spec(),
                tiny,
            ),
            Err(ConstrainedZonotopeConvTranspose2dError::ResourceLimit {
                resource: "weight elements",
                required: 1,
                limit: 0,
            })
        ));
        tiny = limits();
        tiny.max_kernel_visits = 0;
        assert!(matches!(
            constrained_zonotope_conv_transpose2d_unwired(
                &input,
                [1, 1, 1],
                weights.view(),
                &[0.0],
                spec(),
                tiny,
            ),
            Err(ConstrainedZonotopeConvTranspose2dError::ResourceLimit {
                resource: "kernel visits",
                required: 1,
                limit: 0,
            })
        ));
        tiny = limits();
        tiny.max_interval_products = 0;
        assert!(matches!(
            constrained_zonotope_conv_transpose2d_unwired(
                &input,
                [1, 1, 1],
                weights.view(),
                &[0.0],
                spec(),
                tiny,
            ),
            Err(ConstrainedZonotopeConvTranspose2dError::ResourceLimit {
                resource: "interval products",
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
        tiny = limits();
        tiny.max_generator_nonzeros = 1;
        assert!(matches!(
            constrained_zonotope_conv_transpose2d_unwired(
                &sparse_input,
                [1, 1, 1],
                weights.view(),
                &[0.0],
                spec(),
                tiny,
            ),
            Err(ConstrainedZonotopeConvTranspose2dError::ResourceLimit {
                resource: "input generator nonzeros",
                required: 2,
                limit: 1,
            })
        ));

        bad_spec = spec();
        bad_spec.stride = [usize::MAX, 1];
        let two_values =
            ConstrainedZonotope64::from_certified_bounds(&[0.0, 0.0], &[1.0, 1.0], &[false; 2])
                .unwrap();
        assert!(matches!(
            constrained_zonotope_conv_transpose2d_unwired(
                &two_values,
                [1, 2, 1],
                weights.view(),
                &[0.0],
                bad_spec,
                limits(),
            ),
            Err(ConstrainedZonotopeConvTranspose2dError::ResourceOverflow { .. })
        ));
    }

    #[test]
    fn mixed_scale_contractions_charge_exact_rational_rounding_width() {
        let pow2 = |exponent: i32| {
            let biased = u64::try_from(exponent + 1_023).unwrap();
            f64::from_bits(biased << 52)
        };
        let input = ConstrainedZonotope64::try_new(
            vec![
                pow2(300) * f64::from_bits(0x3ff0_0000_0000_0001),
                -pow2(300),
                pow2(-300),
                -pow2(-300) * f64::from_bits(0x3ff0_0000_0000_0001),
            ],
            vec![
                vec![(0, pow2(200)), (1, -pow2(200)), (3, pow2(-200))],
                vec![(0, pow2(-250)), (2, -pow2(250))],
            ],
            Array2::zeros((0, 2)),
            Vec::new(),
            vec![pow2(-400), pow2(-350), pow2(-450), pow2(-425)],
        )
        .unwrap();
        let weights = Array4::from_shape_vec(
            (1, 1, 2, 2),
            vec![
                f64::from_bits(0x3ff0_0000_0000_0001),
                -f64::from_bits(0x3fef_ffff_ffff_ffff),
                1.5,
                -1.25,
            ],
        )
        .unwrap();
        let bias = [pow2(-500)];
        let (output, plan) = constrained_zonotope_conv_transpose2d_unwired(
            &input,
            [1, 2, 2],
            weights.view(),
            &bias,
            spec(),
            limits(),
        )
        .unwrap();
        let exact = exact_scatter(
            &input,
            [1, 2, 2],
            weights.view(),
            &bias,
            spec(),
            plan.output_shape,
        );
        assert_encloses_exact(&output, &exact);

        let center_index = 4;
        assert!(
            rat(output.box_remainder()[center_index]) > exact.remainder[center_index],
            "mixed-scale center must charge arithmetic width beyond the input remainder image"
        );
    }

    fn budget_input() -> ConstrainedZonotope64 {
        ConstrainedZonotope64::try_new(
            vec![1.0],
            vec![vec![(0, 0.5)]],
            array![[1.0]],
            vec![1.0],
            vec![0.25],
        )
        .unwrap()
    }

    #[test]
    fn budgeted_conv_transpose_checks_peak_boundary_overflow_and_admission() {
        let input = budget_input();
        let weights = Array4::from_shape_vec((1, 1, 1, 1), vec![1.0]).unwrap();
        let start = Instant::now();
        let deadline = start + Duration::from_mins(1);
        let baseline = 53;

        let first = constrained_zonotope_conv_transpose2d_unwired_with_clock(
            &input,
            [1, 1, 1],
            weights.view(),
            &[0.0],
            spec(),
            limits(),
            ConstrainedZonotopeCallBudget::new(deadline, baseline, usize::MAX),
            |_| start,
        )
        .unwrap();
        let legacy = constrained_zonotope_conv_transpose2d_unwired(
            &input,
            [1, 1, 1],
            weights.view(),
            &[0.0],
            spec(),
            limits(),
        )
        .unwrap();
        assert_eq!(first.value(), &legacy);
        let required = first.report().peak_live_bytes();
        assert!(required > baseline);

        let at_boundary = constrained_zonotope_conv_transpose2d_unwired_with_clock(
            &input,
            [1, 1, 1],
            weights.view(),
            &[0.0],
            spec(),
            limits(),
            ConstrainedZonotopeCallBudget::new(deadline, baseline, required),
            |_| start,
        )
        .unwrap();
        assert_eq!(at_boundary.report().peak_live_bytes(), required);

        assert_eq!(
            constrained_zonotope_conv_transpose2d_unwired_with_clock(
                &input,
                [1, 1, 1],
                weights.view(),
                &[0.0],
                spec(),
                limits(),
                ConstrainedZonotopeCallBudget::new(deadline, baseline, required - 1),
                |_| start,
            ),
            Err(ConstrainedZonotopeConvTranspose2dBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                    required,
                    limit: required - 1,
                }
            ))
        );

        assert!(matches!(
            constrained_zonotope_conv_transpose2d_unwired_with_clock(
                &input,
                [1, 1, 1],
                weights.view(),
                &[0.0],
                spec(),
                limits(),
                ConstrainedZonotopeCallBudget::new(deadline, usize::MAX, usize::MAX),
                |_| start,
            ),
            Err(ConstrainedZonotopeConvTranspose2dBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                    operation: "aggregate peak-live bytes"
                }
            ))
        ));

        assert!(matches!(
            constrained_zonotope_conv_transpose2d_unwired_with_clock(
                &input,
                [1, 1, 1],
                weights.view(),
                &[0.0],
                spec(),
                limits(),
                ConstrainedZonotopeCallBudget::new(start, 0, usize::MAX),
                |_| start,
            ),
            Err(ConstrainedZonotopeConvTranspose2dBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "admission"
                }
            ))
        ));
    }

    #[test]
    fn conv_transpose_deadline_refuses_every_major_seam_and_no_partial_output_escapes() {
        let input = budget_input();
        let original = input.clone();
        let weights = Array4::from_shape_vec((1, 1, 1, 1), vec![1.0]).unwrap();
        let seams = [
            "ConvTranspose2d geometry validation complete",
            "ConvTranspose2d adjacency construction complete",
            "ConvTranspose2d interval-product preflight complete",
            "ConvTranspose2d numeric transform complete",
            "ConvTranspose2d constraint clone complete",
            "ConvTranspose2d domain materialization complete",
            "ConvTranspose2d publication",
        ];

        for seam in seams {
            let start = Instant::now();
            let deadline = start + Duration::from_mins(1);
            let result = constrained_zonotope_conv_transpose2d_unwired_with_clock(
                &input,
                [1, 1, 1],
                weights.view(),
                &[0.0],
                spec(),
                limits(),
                ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
                move |checkpoint| {
                    if checkpoint == seam {
                        deadline
                    } else {
                        start
                    }
                },
            );
            assert!(matches!(
                result,
                Err(
                    ConstrainedZonotopeConvTranspose2dBudgetError::Budget(
                        ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
                    )
                ) if checkpoint == seam
            ));
            assert_eq!(
                input, original,
                "a declined call at {seam} mutated its input"
            );
        }
    }

    #[test]
    fn conv_transpose_deadline_polls_within_validation_adjacency_numeric_and_clone_phases() {
        const ITEMS: usize = crate::CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL;
        let start = Instant::now();
        let deadline = start + Duration::from_mins(1);

        let wide_weights = Array4::from_shape_vec((1, 1, 1, ITEMS), vec![1.0; ITEMS]).unwrap();
        let point = ConstrainedZonotope64::try_new(
            vec![0.0],
            Vec::new(),
            Array2::zeros((0, 0)),
            Vec::new(),
            vec![0.0],
        )
        .unwrap();
        let mut wide_limits = limits();
        wide_limits.max_value_count = ITEMS;
        wide_limits.max_weight_elements = ITEMS;
        wide_limits.max_kernel_visits = usize::MAX;
        wide_limits.max_interval_products = usize::MAX;
        let validation = constrained_zonotope_conv_transpose2d_unwired_with_clock(
            &point,
            [1, 1, 1],
            wide_weights.view(),
            &[0.0],
            spec(),
            wide_limits,
            ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
            move |checkpoint| {
                if checkpoint == "ConvTranspose2d finite-parameter validation" {
                    deadline
                } else {
                    start
                }
            },
        );
        assert!(matches!(
            validation,
            Err(ConstrainedZonotopeConvTranspose2dBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "ConvTranspose2d finite-parameter validation"
                }
            ))
        ));

        let dense = ConstrainedZonotope64::try_new(
            vec![0.0; ITEMS],
            vec![(0..ITEMS).map(|index| (index, 1.0)).collect()],
            Array2::zeros((0, 1)),
            Vec::new(),
            vec![0.0; ITEMS],
        )
        .unwrap();
        let unit_weights = Array4::from_shape_vec((1, 1, 1, 1), vec![1.0]).unwrap();
        let dense_limits = ConstrainedZonotopeConvTranspose2dLimits {
            max_value_count: ITEMS,
            max_alpha_dim: 1,
            max_generator_nonzeros: ITEMS,
            max_weight_elements: 1,
            max_kernel_visits: ITEMS,
            max_interval_products: ITEMS * 2,
            max_constraint_count: 0,
            max_constraint_elements: 0,
        };
        for phase in [
            "ConvTranspose2d adjacency counting",
            "ConvTranspose2d interval-product preflight",
        ] {
            let result = constrained_zonotope_conv_transpose2d_unwired_with_clock(
                &dense,
                [1, 1, ITEMS],
                unit_weights.view(),
                &[0.0],
                spec(),
                dense_limits,
                ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
                move |checkpoint| {
                    if checkpoint == phase {
                        deadline
                    } else {
                        start
                    }
                },
            );
            assert!(matches!(
                result,
                Err(
                    ConstrainedZonotopeConvTranspose2dBudgetError::Budget(
                        ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
                    )
                ) if checkpoint == phase
            ));
        }

        let numeric_point = ConstrainedZonotope64::try_new(
            vec![0.0; ITEMS],
            Vec::new(),
            Array2::zeros((0, 0)),
            Vec::new(),
            vec![0.0; ITEMS],
        )
        .unwrap();
        let numeric = constrained_zonotope_conv_transpose2d_unwired_with_clock(
            &numeric_point,
            [1, 1, ITEMS],
            unit_weights.view(),
            &[0.0],
            spec(),
            dense_limits,
            ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
            move |checkpoint| {
                if checkpoint == "ConvTranspose2d output transform" {
                    deadline
                } else {
                    start
                }
            },
        );
        assert!(matches!(
            numeric,
            Err(ConstrainedZonotopeConvTranspose2dBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "ConvTranspose2d output transform"
                }
            ))
        ));

        let adjacency_initialization = constrained_zonotope_conv_transpose2d_unwired_with_clock(
            &numeric_point,
            [1, 1, ITEMS],
            unit_weights.view(),
            &[0.0],
            spec(),
            dense_limits,
            ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
            move |checkpoint| {
                if checkpoint == "ConvTranspose2d adjacency-count initialization" {
                    deadline
                } else {
                    start
                }
            },
        );
        assert!(matches!(
            adjacency_initialization,
            Err(ConstrainedZonotopeConvTranspose2dBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "ConvTranspose2d adjacency-count initialization"
                }
            ))
        ));

        let many_empty_generators = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![Vec::new(); ITEMS],
            Array2::zeros((0, ITEMS)),
            Vec::new(),
            vec![0.0],
        )
        .unwrap();
        let empty_generator_limits = ConstrainedZonotopeConvTranspose2dLimits {
            max_value_count: 1,
            max_alpha_dim: ITEMS,
            max_generator_nonzeros: 0,
            max_weight_elements: 1,
            max_kernel_visits: 1,
            max_interval_products: 1,
            max_constraint_count: 0,
            max_constraint_elements: 0,
        };
        for phase in [
            "ConvTranspose2d generator-scratch initialization",
            "ConvTranspose2d generator-column initialization",
        ] {
            let result = constrained_zonotope_conv_transpose2d_unwired_with_clock(
                &many_empty_generators,
                [1, 1, 1],
                unit_weights.view(),
                &[0.0],
                spec(),
                empty_generator_limits,
                ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
                move |checkpoint| {
                    if checkpoint == phase {
                        deadline
                    } else {
                        start
                    }
                },
            );
            assert!(matches!(
                result,
                Err(
                    ConstrainedZonotopeConvTranspose2dBudgetError::Budget(
                        ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
                    )
                ) if checkpoint == phase
            ));
        }

        let constrained = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![Vec::new(); 128],
            Array2::zeros((128, 128)),
            vec![0.0; 128],
            vec![0.0],
        )
        .unwrap();
        let clone_limits = ConstrainedZonotopeConvTranspose2dLimits {
            max_value_count: 1,
            max_alpha_dim: 128,
            max_generator_nonzeros: 0,
            max_weight_elements: 1,
            max_kernel_visits: 1,
            max_interval_products: 1,
            max_constraint_count: 128,
            max_constraint_elements: ITEMS,
        };
        let clone = constrained_zonotope_conv_transpose2d_unwired_with_clock(
            &constrained,
            [1, 1, 1],
            unit_weights.view(),
            &[0.0],
            spec(),
            clone_limits,
            ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
            move |checkpoint| {
                if checkpoint == "ConvTranspose2d constraint-matrix clone" {
                    deadline
                } else {
                    start
                }
            },
        );
        assert!(matches!(
            clone,
            Err(ConstrainedZonotopeConvTranspose2dBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "ConvTranspose2d constraint-matrix clone"
                }
            ))
        ));

        let rhs_input = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![Vec::new()],
            Array2::zeros((ITEMS, 1)),
            vec![0.0; ITEMS],
            vec![0.0],
        )
        .unwrap();
        let rhs_limits = ConstrainedZonotopeConvTranspose2dLimits {
            max_value_count: 1,
            max_alpha_dim: 1,
            max_generator_nonzeros: 0,
            max_weight_elements: 1,
            max_kernel_visits: 1,
            max_interval_products: 1,
            max_constraint_count: ITEMS,
            max_constraint_elements: ITEMS,
        };
        let rhs_clone = constrained_zonotope_conv_transpose2d_unwired_with_clock(
            &rhs_input,
            [1, 1, 1],
            unit_weights.view(),
            &[0.0],
            spec(),
            rhs_limits,
            ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
            move |checkpoint| {
                if checkpoint == "ConvTranspose2d right-hand-side clone" {
                    deadline
                } else {
                    start
                }
            },
        );
        assert!(matches!(
            rhs_clone,
            Err(ConstrainedZonotopeConvTranspose2dBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "ConvTranspose2d right-hand-side clone"
                }
            ))
        ));
    }
}
