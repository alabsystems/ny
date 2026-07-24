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
    require_gradual_underflow()?;
    let geometry = validate_geometry(input, input_shape, weights, bias, spec, limits)?;

    let input_generator_nonzeros = input.generators().iter().try_fold(
        0_usize,
        |sum, generator| -> Result<_, ConstrainedZonotopeConv2dError> {
            let required = sum.checked_add(generator.nnz()).ok_or(
                ConstrainedZonotopeConv2dError::ResourceOverflow {
                    operation: "input generator nonzeros",
                },
            )?;
            check_limit(
                "input generator nonzeros",
                required,
                limits.max_generator_nonzeros,
            )?;
            Ok(required)
        },
    )?;

    let adjacency = build_input_adjacency(input, input_generator_nonzeros)?;
    let alpha_dim = input.alpha_dim();
    let mut generator_scratch: Vec<Option<OutwardInterval>> = Vec::new();
    try_reserve(
        &mut generator_scratch,
        alpha_dim,
        "generator interval scratch",
    )?;
    generator_scratch.resize(alpha_dim, None);
    let mut touched_generators = Vec::new();
    try_reserve(
        &mut touched_generators,
        alpha_dim,
        "touched-generator scratch",
    )?;

    let output_value_count =
        checked_product(&geometry.output_shape, "convolution output value count")?;
    let mut output_center = Vec::new();
    let mut output_remainder = Vec::new();
    try_reserve(&mut output_center, output_value_count, "output center")?;
    try_reserve(
        &mut output_remainder,
        output_value_count,
        "output box remainder",
    )?;

    let mut output_generators: Vec<Vec<(usize, f64)>> = Vec::new();
    try_reserve(
        &mut output_generators,
        alpha_dim,
        "output generator columns",
    )?;
    output_generators.resize_with(alpha_dim, Vec::new);

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
                let output_index = output_center.len();
                let mut center_sum = OutwardInterval::exact(bias[output_channel]);
                let mut remainder_sum = OutwardInterval::zero();

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
                        output_generators[generator_index]
                            .try_reserve(1)
                            .map_err(|_| ConstrainedZonotopeConv2dError::AllocationFailure {
                                resource: "output generator coefficients",
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

    debug_assert_eq!(output_center.len(), output_value_count);
    let constraints = clone_constraints(input)?;
    let rhs = clone_slice(input.rhs(), "constraint right-hand side")?;
    let output = ConstrainedZonotope64::try_new(
        output_center,
        output_generators,
        constraints,
        rhs,
        output_remainder,
    )?;
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
    Ok((output, plan))
}

#[derive(Clone, Copy, Debug)]
struct Geometry {
    output_shape: [usize; 3],
    weight_shape: [usize; 4],
    weight_elements: usize,
    kernel_visits: usize,
}

fn validate_geometry(
    input: &ConstrainedZonotope64,
    input_shape: [usize; 3],
    weights: ArrayView4<'_, f64>,
    bias: &[f64],
    spec: ConstrainedZonotopeConv2dSpec,
    limits: ConstrainedZonotopeConv2dLimits,
) -> Result<Geometry, ConstrainedZonotopeConv2dError> {
    let input_value_count = checked_product(&input_shape, "convolution input value count")?;
    if input.value_dim() != input_value_count {
        return Err(ConstrainedZonotopeConv2dError::Shape {
            field: "input domain",
            expected: vec![input_value_count],
            got: vec![input.value_dim()],
        });
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
        });
    }
    if spec.groups == 0 {
        return Err(ConstrainedZonotopeConv2dError::InvalidSpec {
            message: "groups must be non-zero".to_string(),
        });
    }
    if spec.stride.contains(&0) || spec.dilation.contains(&0) {
        return Err(ConstrainedZonotopeConv2dError::InvalidSpec {
            message: format!(
                "stride and dilation must be non-zero, got {:?} and {:?}",
                spec.stride, spec.dilation
            ),
        });
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
        });
    }
    if input_channels % spec.groups != 0 || output_channels % spec.groups != 0 {
        return Err(ConstrainedZonotopeConv2dError::InvalidSpec {
            message: format!(
                "input/output channels {input_channels}/{output_channels} must be divisible by groups {}",
                spec.groups
            ),
        });
    }
    let expected_kernel_input_channels = input_channels / spec.groups;
    if kernel_input_channels != expected_kernel_input_channels {
        return Err(ConstrainedZonotopeConv2dError::Shape {
            field: "weight input channels per group",
            expected: vec![expected_kernel_input_channels],
            got: vec![kernel_input_channels],
        });
    }
    if bias.len() != output_channels {
        return Err(ConstrainedZonotopeConv2dError::Shape {
            field: "bias",
            expected: vec![output_channels],
            got: vec![bias.len()],
        });
    }
    let weight_elements = checked_product(&weight_shape, "convolution weight elements")?;
    check_limit(
        "weight elements",
        weight_elements,
        limits.max_weight_elements,
    )?;
    validate_finite("weights", weights.iter().copied())?;
    validate_finite("bias", bias.iter().copied())?;

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
        weight_elements,
        kernel_visits,
    })
}

fn output_dimension(
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

fn input_coordinate(
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

fn build_input_adjacency(
    input: &ConstrainedZonotope64,
    total_nonzeros: usize,
) -> Result<Vec<Vec<(usize, f64)>>, ConstrainedZonotopeConv2dError> {
    let mut counts = Vec::new();
    try_reserve(&mut counts, input.value_dim(), "input adjacency counts")?;
    counts.resize(input.value_dim(), 0_usize);
    for generator in input.generators() {
        for (value_index, _) in generator.entries() {
            counts[value_index] = counts[value_index].checked_add(1).ok_or(
                ConstrainedZonotopeConv2dError::ResourceOverflow {
                    operation: "per-value generator adjacency",
                },
            )?;
        }
    }

    let mut adjacency = Vec::new();
    try_reserve(
        &mut adjacency,
        input.value_dim(),
        "input generator adjacency",
    )?;
    for &count in &counts {
        let mut entries = Vec::new();
        try_reserve(&mut entries, count, "input generator adjacency entries")?;
        adjacency.push(entries);
    }
    for (generator_index, generator) in input.generators().iter().enumerate() {
        for (value_index, coefficient) in generator.entries() {
            adjacency[value_index].push((generator_index, coefficient));
        }
    }
    debug_assert_eq!(
        adjacency.iter().map(Vec::len).sum::<usize>(),
        total_nonzeros
    );
    Ok(adjacency)
}

fn clone_constraints(
    input: &ConstrainedZonotope64,
) -> Result<Array2<f64>, ConstrainedZonotopeConv2dError> {
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
    try_reserve(&mut values, element_count, "constraint matrix")?;
    for row in 0..shape.0 {
        for column in 0..shape.1 {
            values.push(constraints[[row, column]]);
        }
    }
    Array2::from_shape_vec(shape, values).map_err(|_| {
        ConstrainedZonotopeConv2dError::ResourceOverflow {
            operation: "constraint matrix shape",
        }
    })
}

fn clone_slice<T: Copy>(
    source: &[T],
    resource: &'static str,
) -> Result<Vec<T>, ConstrainedZonotopeConv2dError> {
    let mut output = Vec::new();
    try_reserve(&mut output, source.len(), resource)?;
    output.extend_from_slice(source);
    Ok(output)
}

fn validate_finite(
    field: &'static str,
    values: impl IntoIterator<Item = f64>,
) -> Result<(), ConstrainedZonotopeConv2dError> {
    for (index, value) in values.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(ConstrainedZonotopeConv2dError::NonFinite { field, index });
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
        let input = ConstrainedZonotope64::try_new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![vec![(0, 0.25), (3, -0.5)], vec![(1, 0.75)]],
            array![[1.0, -0.25]],
            vec![0.5],
            vec![0.1, 0.2, 0.3, 0.4],
        )
        .unwrap();
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
