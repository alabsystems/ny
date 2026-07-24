// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Outward sparse `f64` affine propagation for an unwired constrained zonotope.
//!
//! The transform consumes normalized `[output, input]` weights and preserves
//! sparse generator columns. Every evaluated product and reduction is enclosed
//! by adjacent `f64` endpoints. The image of the input box remainder and the
//! representation error of each output center/generator coefficient are
//! accumulated into the output box remainder, enclosing the real affine image
//! of the exact dyadic values supplied here.
//!
//! This module is deliberately **unwired**. It does not read ONNX tensors,
//! normalize Gemm attributes, choose activation relaxations, run on CUDA, or
//! affect a scored verdict. Callers must obtain weights and input domains from
//! proof-qualified sources; converting an already-rounded `f32` abstract state
//! cannot recover arithmetic error discarded earlier.

use ndarray::{Array2, ArrayView2};

use crate::{ConstrainedZonotope64, ConstrainedZonotope64Error};

/// Explicit resource limits for [`constrained_zonotope_affine_unwired`].
///
/// There is intentionally no `Default`: an experimental caller must choose
/// every cap. Counts cover memory owned by the returned domain and the
/// transform's explicit sparse scratch, but not allocator bookkeeping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeAffineLimits {
    /// Maximum number of input scalar values.
    pub max_input_value_count: usize,
    /// Maximum number of output scalar values.
    pub max_output_value_count: usize,
    /// Maximum alpha dimension and per-output dense scratch length.
    pub max_alpha_dim: usize,
    /// Maximum sparse generator nonzeros in either the input or output.
    pub max_generator_nonzeros: usize,
    /// Maximum logical weight elements inspected for finiteness.
    pub max_weight_elements: usize,
    /// Maximum output/input matrix visits, including structural-zero weights.
    pub max_matrix_visits: usize,
    /// Maximum interval multiplications actually performed by the transform.
    pub max_interval_products: usize,
    /// Maximum retained predicate rows, including rows with zero alpha width.
    pub max_constraint_count: usize,
    /// Maximum number of retained `C` matrix elements.
    pub max_constraint_elements: usize,
}

/// Checked shape and work accounting for one completed affine transform.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeAffinePlan {
    /// Number of input scalar values.
    pub input_value_count: usize,
    /// Number of output scalar values.
    pub output_value_count: usize,
    /// Alpha dimension, unchanged by an affine transform.
    pub alpha_dim: usize,
    /// Constraint count, unchanged by an affine transform.
    pub constraint_count: usize,
    /// Retained `C` matrix elements.
    pub constraint_elements: usize,
    /// Logical weight elements validated before the transform.
    pub weight_elements: usize,
    /// Output/input visits, including structural zeros.
    pub matrix_visits: usize,
    /// Sparse nonzeros in the input generators.
    pub input_generator_nonzeros: usize,
    /// Sparse nonzeros retained in the output generators.
    pub output_generator_nonzeros: usize,
    /// Interval products actually evaluated, including remainder products.
    pub interval_products: usize,
}

/// Invalid affine data, exhausted resources, or arithmetic that could not be
/// enclosed by finite `f64` endpoints.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConstrainedZonotopeAffineError {
    /// A matrix or vector has the wrong shape.
    #[error("shape mismatch for {field}: expected {expected:?}, got {got:?}")]
    Shape {
        /// Input whose shape is wrong.
        field: &'static str,
        /// Required shape.
        expected: Vec<usize>,
        /// Supplied shape.
        got: Vec<usize>,
    },

    /// A dimension is structurally invalid.
    #[error("invalid affine specification: {message}")]
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
    #[error("internal affine invariant violated: {message}")]
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

/// Apply an exact-dyadic affine map while preserving sparse generator support
/// and charging all floating-point width to the output box remainder.
///
/// `weights` must be normalized to `[output, input]`; `bias` has one value per
/// output. The predicate `C alpha <= d` is copied unchanged because the map is
/// affine in represented values.
pub fn constrained_zonotope_affine_unwired(
    input: &ConstrainedZonotope64,
    weights: ArrayView2<'_, f64>,
    bias: &[f64],
    limits: ConstrainedZonotopeAffineLimits,
) -> Result<(ConstrainedZonotope64, ConstrainedZonotopeAffinePlan), ConstrainedZonotopeAffineError>
{
    require_gradual_underflow()?;
    let geometry = validate_geometry(input, weights, bias, limits)?;

    let input_generator_nonzeros =
        input
            .generators()
            .iter()
            .try_fold(0_usize, |sum, generator| {
                sum.checked_add(generator.nnz()).ok_or(
                    ConstrainedZonotopeAffineError::ResourceOverflow {
                        operation: "input generator nonzeros",
                    },
                )
            })?;
    check_limit(
        "input generator nonzeros",
        input_generator_nonzeros,
        limits.max_generator_nonzeros,
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

    let mut output_center = Vec::new();
    let mut output_remainder = Vec::new();
    try_reserve(
        &mut output_center,
        geometry.output_value_count,
        "output center",
    )?;
    try_reserve(
        &mut output_remainder,
        geometry.output_value_count,
        "output box remainder",
    )?;

    let mut output_generators: Vec<Vec<(usize, f64)>> = Vec::new();
    try_reserve(
        &mut output_generators,
        alpha_dim,
        "output generator columns",
    )?;
    output_generators.resize_with(alpha_dim, Vec::new);

    let mut interval_products = 0_usize;
    let mut output_generator_nonzeros = 0_usize;

    for output_index in 0..geometry.output_value_count {
        let mut center_sum = OutwardInterval::exact(bias[output_index]);
        let mut remainder_sum = OutwardInterval::zero();

        for input_index in 0..geometry.input_value_count {
            let weight = weights[[output_index, input_index]];
            if weight == 0.0 {
                continue;
            }

            consume_product(&mut interval_products, limits.max_interval_products)?;
            center_sum = center_sum.add(outward_product(weight, input.center()[input_index])?)?;

            let input_radius = input.box_remainder()[input_index];
            if input_radius != 0.0 {
                consume_product(&mut interval_products, limits.max_interval_products)?;
                remainder_sum =
                    remainder_sum.add(outward_product(weight.abs(), input_radius)?.abs())?;
            }

            for &(generator_index, coefficient) in &adjacency[input_index] {
                consume_product(&mut interval_products, limits.max_interval_products)?;
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

        let (nominal_center, center_error) = center_sum.nominal_and_error()?;
        let mut total_remainder = remainder_sum.hi;
        total_remainder = add_nonnegative_upper(total_remainder, center_error)?;

        for &generator_index in &touched_generators {
            let interval = generator_scratch[generator_index].take().ok_or(
                ConstrainedZonotopeAffineError::InvariantViolation {
                    message: "a touched generator has no accumulated interval",
                },
            )?;
            let (coefficient, coefficient_error) = interval.nominal_and_error()?;
            total_remainder = add_nonnegative_upper(total_remainder, coefficient_error)?;
            if coefficient != 0.0 {
                output_generator_nonzeros = output_generator_nonzeros.checked_add(1).ok_or(
                    ConstrainedZonotopeAffineError::ResourceOverflow {
                        operation: "output generator nonzeros",
                    },
                )?;
                check_limit(
                    "output generator nonzeros",
                    output_generator_nonzeros,
                    limits.max_generator_nonzeros,
                )?;
                output_generators[generator_index]
                    .try_reserve(1)
                    .map_err(|_| ConstrainedZonotopeAffineError::AllocationFailure {
                        resource: "output generator coefficients",
                    })?;
                output_generators[generator_index].push((output_index, coefficient));
            }
        }
        touched_generators.clear();

        output_center.push(nominal_center);
        output_remainder.push(total_remainder);
    }

    let constraints = clone_constraints(input)?;
    let rhs = clone_slice(input.rhs(), "constraint right-hand side")?;
    let output = ConstrainedZonotope64::try_new(
        output_center,
        output_generators,
        constraints,
        rhs,
        output_remainder,
    )?;
    let plan = ConstrainedZonotopeAffinePlan {
        input_value_count: geometry.input_value_count,
        output_value_count: geometry.output_value_count,
        alpha_dim,
        constraint_count: input.constraint_count(),
        constraint_elements: geometry.constraint_elements,
        weight_elements: geometry.weight_elements,
        matrix_visits: geometry.matrix_visits,
        input_generator_nonzeros,
        output_generator_nonzeros,
        interval_products,
    };
    Ok((output, plan))
}

#[derive(Clone, Copy, Debug)]
struct Geometry {
    input_value_count: usize,
    output_value_count: usize,
    constraint_elements: usize,
    weight_elements: usize,
    matrix_visits: usize,
}

fn validate_geometry(
    input: &ConstrainedZonotope64,
    weights: ArrayView2<'_, f64>,
    bias: &[f64],
    limits: ConstrainedZonotopeAffineLimits,
) -> Result<Geometry, ConstrainedZonotopeAffineError> {
    let input_value_count = input.value_dim();
    let output_value_count = weights.nrows();
    if input_value_count == 0 || output_value_count == 0 {
        return Err(ConstrainedZonotopeAffineError::InvalidSpec {
            message: format!(
                "input and output dimensions must be non-zero, got {input_value_count} and {output_value_count}"
            ),
        });
    }
    if weights.ncols() != input_value_count {
        return Err(ConstrainedZonotopeAffineError::Shape {
            field: "weight input dimension",
            expected: vec![output_value_count, input_value_count],
            got: weights.shape().to_vec(),
        });
    }
    if bias.len() != output_value_count {
        return Err(ConstrainedZonotopeAffineError::Shape {
            field: "bias",
            expected: vec![output_value_count],
            got: vec![bias.len()],
        });
    }
    check_limit(
        "input value count",
        input_value_count,
        limits.max_input_value_count,
    )?;
    check_limit(
        "output value count",
        output_value_count,
        limits.max_output_value_count,
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
        .ok_or(ConstrainedZonotopeAffineError::ResourceOverflow {
            operation: "constraint matrix elements",
        })?;
    check_limit(
        "constraint matrix elements",
        constraint_elements,
        limits.max_constraint_elements,
    )?;

    let weight_elements = output_value_count.checked_mul(input_value_count).ok_or(
        ConstrainedZonotopeAffineError::ResourceOverflow {
            operation: "affine weight elements",
        },
    )?;
    check_limit(
        "weight elements",
        weight_elements,
        limits.max_weight_elements,
    )?;
    let matrix_visits = weight_elements;
    check_limit("matrix visits", matrix_visits, limits.max_matrix_visits)?;
    validate_finite("weights", weights.iter().copied())?;
    validate_finite("bias", bias.iter().copied())?;

    Ok(Geometry {
        input_value_count,
        output_value_count,
        constraint_elements,
        weight_elements,
        matrix_visits,
    })
}

fn build_input_adjacency(
    input: &ConstrainedZonotope64,
    total_nonzeros: usize,
) -> Result<Vec<Vec<(usize, f64)>>, ConstrainedZonotopeAffineError> {
    let mut counts = Vec::new();
    try_reserve(&mut counts, input.value_dim(), "input adjacency counts")?;
    counts.resize(input.value_dim(), 0_usize);
    for generator in input.generators() {
        for (value_index, _) in generator.entries() {
            counts[value_index] = counts[value_index].checked_add(1).ok_or(
                ConstrainedZonotopeAffineError::ResourceOverflow {
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
) -> Result<Array2<f64>, ConstrainedZonotopeAffineError> {
    let shape = (input.constraint_count(), input.alpha_dim());
    let element_count =
        shape
            .0
            .checked_mul(shape.1)
            .ok_or(ConstrainedZonotopeAffineError::ResourceOverflow {
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
        ConstrainedZonotopeAffineError::ResourceOverflow {
            operation: "constraint matrix shape",
        }
    })
}

fn clone_slice<T: Copy>(
    source: &[T],
    resource: &'static str,
) -> Result<Vec<T>, ConstrainedZonotopeAffineError> {
    let mut output = Vec::new();
    try_reserve(&mut output, source.len(), resource)?;
    output.extend_from_slice(source);
    Ok(output)
}

fn validate_finite(
    field: &'static str,
    values: impl IntoIterator<Item = f64>,
) -> Result<(), ConstrainedZonotopeAffineError> {
    for (index, value) in values.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(ConstrainedZonotopeAffineError::NonFinite { field, index });
        }
    }
    Ok(())
}

fn check_limit(
    resource: &'static str,
    required: usize,
    limit: usize,
) -> Result<(), ConstrainedZonotopeAffineError> {
    if required > limit {
        return Err(ConstrainedZonotopeAffineError::ResourceLimit {
            resource,
            required,
            limit,
        });
    }
    Ok(())
}

fn consume_product(count: &mut usize, limit: usize) -> Result<(), ConstrainedZonotopeAffineError> {
    *count = count
        .checked_add(1)
        .ok_or(ConstrainedZonotopeAffineError::ResourceOverflow {
            operation: "interval product count",
        })?;
    check_limit("interval products", *count, limit)
}

fn try_reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    resource: &'static str,
) -> Result<(), ConstrainedZonotopeAffineError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| ConstrainedZonotopeAffineError::AllocationFailure { resource })
}

/// Reject FTZ/DAZ before adjacent-float intervals are used as proof objects.
fn require_gradual_underflow() -> Result<(), ConstrainedZonotopeAffineError> {
    let half = std::hint::black_box(0.5_f64);
    let min_normal = std::hint::black_box(f64::MIN_POSITIVE);
    let min_subnormal = std::hint::black_box(f64::from_bits(1));
    let two_subnormals = std::hint::black_box(f64::from_bits(2));
    if std::hint::black_box(min_normal * half).to_bits() != 0x0008_0000_0000_0000
        || std::hint::black_box(two_subnormals * half).to_bits() != 1
        || std::hint::black_box(min_subnormal + min_subnormal).to_bits() != 2
    {
        return Err(ConstrainedZonotopeAffineError::UnsupportedFloatingPoint {
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

    fn add(self, rhs: Self) -> Result<Self, ConstrainedZonotopeAffineError> {
        if self.is_exact_zero() {
            return Ok(rhs);
        }
        if rhs.is_exact_zero() {
            return Ok(self);
        }
        Ok(Self {
            lo: round_down(self.lo + rhs.lo, "affine interval sum")?,
            hi: round_up(self.hi + rhs.hi, "affine interval sum")?,
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

    fn nominal_and_error(self) -> Result<(f64, f64), ConstrainedZonotopeAffineError> {
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
) -> Result<OutwardInterval, ConstrainedZonotopeAffineError> {
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
        lo: round_down(product, "affine interval product")?,
        hi: round_up(product, "affine interval product")?,
    })
}

fn upper_difference(upper: f64, lower: f64) -> Result<f64, ConstrainedZonotopeAffineError> {
    if upper == lower {
        return Ok(0.0);
    }
    round_up(upper - lower, "affine representation error")
}

fn add_nonnegative_upper(left: f64, right: f64) -> Result<f64, ConstrainedZonotopeAffineError> {
    if left == 0.0 {
        return Ok(right);
    }
    if right == 0.0 {
        return Ok(left);
    }
    round_up(left + right, "affine box remainder sum")
}

fn round_down(value: f64, operation: &'static str) -> Result<f64, ConstrainedZonotopeAffineError> {
    if !value.is_finite() {
        return Err(ConstrainedZonotopeAffineError::NonFiniteArithmetic { operation });
    }
    let outward = value.next_down();
    if !outward.is_finite() {
        return Err(ConstrainedZonotopeAffineError::NonFiniteArithmetic { operation });
    }
    Ok(outward)
}

fn round_up(value: f64, operation: &'static str) -> Result<f64, ConstrainedZonotopeAffineError> {
    if !value.is_finite() {
        return Err(ConstrainedZonotopeAffineError::NonFiniteArithmetic { operation });
    }
    let outward = value.next_up();
    if !outward.is_finite() {
        return Err(ConstrainedZonotopeAffineError::NonFiniteArithmetic { operation });
    }
    Ok(outward)
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};
    use num_rational::BigRational;
    use num_traits::{Signed, Zero};
    use proptest::prelude::*;

    use super::*;

    fn limits() -> ConstrainedZonotopeAffineLimits {
        ConstrainedZonotopeAffineLimits {
            max_input_value_count: 10_000,
            max_output_value_count: 10_000,
            max_alpha_dim: 1_000,
            max_generator_nonzeros: 100_000,
            max_weight_elements: 1_000_000,
            max_matrix_visits: 1_000_000,
            max_interval_products: 2_000_000,
            max_constraint_count: 100_000,
            max_constraint_elements: 100_000,
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

    fn exact_required_remainder(
        input: &ConstrainedZonotope64,
        output: &ConstrainedZonotope64,
        weights: ArrayView2<'_, f64>,
        bias: &[f64],
        output_index: usize,
    ) -> BigRational {
        let mut exact_center = rat(bias[output_index]);
        let mut exact_remainder = BigRational::zero();
        for input_index in 0..input.value_dim() {
            let weight = weights[[output_index, input_index]];
            exact_center += rat(weight) * rat(input.center()[input_index]);
            exact_remainder += rat(weight).abs() * rat(input.box_remainder()[input_index]);
        }

        let mut required = (exact_center - rat(output.center()[output_index])).abs();
        required += exact_remainder;
        for generator_index in 0..input.alpha_dim() {
            let mut exact_generator = BigRational::zero();
            for input_index in 0..input.value_dim() {
                exact_generator += rat(weights[[output_index, input_index]])
                    * rat(coefficient(input, generator_index, input_index));
            }
            required +=
                (exact_generator - rat(coefficient(output, generator_index, output_index))).abs();
        }
        required
    }

    fn assert_exact_enclosure(
        input: &ConstrainedZonotope64,
        output: &ConstrainedZonotope64,
        weights: ArrayView2<'_, f64>,
        bias: &[f64],
    ) {
        for output_index in 0..output.value_dim() {
            assert!(
                rat(output.box_remainder()[output_index])
                    >= exact_required_remainder(input, output, weights, bias, output_index),
                "output coordinate {output_index} under-enclosed the exact affine image"
            );
        }
    }

    #[test]
    fn exact_rational_affine_image_preserves_constraints_and_sparse_columns() {
        let input = ConstrainedZonotope64::try_new(
            vec![1.25, -2.0, 3.5],
            vec![vec![(0, 0.25), (2, -0.5)], vec![(1, 0.75)]],
            array![[1.0, -0.25]],
            vec![0.5],
            vec![0.125, 0.25, 0.5],
        )
        .unwrap();
        let weights = array![[1.0, -2.0, 0.5], [-0.25, 3.0, 2.0]];
        let bias = [0.125, -0.5];
        let (output, plan) =
            constrained_zonotope_affine_unwired(&input, weights.view(), &bias, limits()).unwrap();

        assert_eq!(output.constraints(), input.constraints());
        assert_eq!(output.rhs(), input.rhs());
        assert_eq!(plan.input_value_count, 3);
        assert_eq!(plan.output_value_count, 2);
        assert_eq!(plan.alpha_dim, 2);
        assert_eq!(plan.constraint_count, 1);
        assert_eq!(plan.constraint_elements, 2);
        assert_eq!(plan.weight_elements, 6);
        assert_eq!(plan.matrix_visits, 6);
        assert_eq!(plan.input_generator_nonzeros, 3);
        assert_eq!(plan.output_generator_nonzeros, 4);
        assert_eq!(plan.interval_products, 18);
        assert_exact_enclosure(&input, &output, weights.view(), &bias);
    }

    #[test]
    fn structural_zero_matrix_is_bounded_by_visits_not_products() {
        let input = ConstrainedZonotope64::try_new(
            vec![1.0, 2.0],
            vec![vec![(0, 0.5)]],
            Array2::zeros((0, 1)),
            Vec::new(),
            vec![0.25, 0.5],
        )
        .unwrap();
        let weights = Array2::zeros((2, 2));
        let mut zero_products = limits();
        zero_products.max_interval_products = 0;
        let (output, plan) = constrained_zonotope_affine_unwired(
            &input,
            weights.view(),
            &[1.0, -1.0],
            zero_products,
        )
        .unwrap();
        assert_eq!(output.center(), &[1.0, -1.0]);
        assert_eq!(output.box_remainder(), &[0.0, 0.0]);
        assert_eq!(plan.matrix_visits, 4);
        assert_eq!(plan.interval_products, 0);
        assert_eq!(plan.output_generator_nonzeros, 0);
    }

    fn simple_input() -> ConstrainedZonotope64 {
        ConstrainedZonotope64::try_new(
            vec![1.0, 2.0],
            vec![vec![(0, 0.25), (1, 0.5)]],
            array![[1.0], [-1.0]],
            vec![1.0, 1.0],
            vec![0.125, 0.25],
        )
        .unwrap()
    }

    fn assert_limit(
        result: Result<
            (ConstrainedZonotope64, ConstrainedZonotopeAffinePlan),
            ConstrainedZonotopeAffineError,
        >,
        resource: &'static str,
    ) {
        assert!(matches!(
            result,
            Err(ConstrainedZonotopeAffineError::ResourceLimit {
                resource: actual,
                ..
            }) if actual == resource
        ));
    }

    #[test]
    fn every_declared_resource_cap_fails_closed() {
        let input = simple_input();
        let weights = Array2::ones((2, 2));

        let mut capped = limits();
        capped.max_input_value_count = 1;
        assert_limit(
            constrained_zonotope_affine_unwired(&input, weights.view(), &[0.0, 0.0], capped),
            "input value count",
        );

        let mut capped = limits();
        capped.max_output_value_count = 1;
        assert_limit(
            constrained_zonotope_affine_unwired(&input, weights.view(), &[0.0, 0.0], capped),
            "output value count",
        );

        let mut capped = limits();
        capped.max_alpha_dim = 0;
        assert_limit(
            constrained_zonotope_affine_unwired(&input, weights.view(), &[0.0, 0.0], capped),
            "alpha dimension",
        );

        let mut capped = limits();
        capped.max_generator_nonzeros = 1;
        assert_limit(
            constrained_zonotope_affine_unwired(&input, weights.view(), &[0.0, 0.0], capped),
            "input generator nonzeros",
        );

        let mut capped = limits();
        capped.max_weight_elements = 3;
        assert_limit(
            constrained_zonotope_affine_unwired(&input, weights.view(), &[0.0, 0.0], capped),
            "weight elements",
        );

        let mut capped = limits();
        capped.max_matrix_visits = 3;
        assert_limit(
            constrained_zonotope_affine_unwired(&input, weights.view(), &[0.0, 0.0], capped),
            "matrix visits",
        );
        let mut nonfinite_weights = weights.clone();
        nonfinite_weights[[0, 0]] = f64::NAN;
        assert_limit(
            constrained_zonotope_affine_unwired(
                &input,
                nonfinite_weights.view(),
                &[0.0, 0.0],
                capped,
            ),
            "matrix visits",
        );

        let mut capped = limits();
        capped.max_interval_products = 0;
        assert_limit(
            constrained_zonotope_affine_unwired(&input, weights.view(), &[0.0, 0.0], capped),
            "interval products",
        );

        let mut capped = limits();
        capped.max_constraint_count = 1;
        assert_limit(
            constrained_zonotope_affine_unwired(&input, weights.view(), &[0.0, 0.0], capped),
            "constraint count",
        );

        let mut capped = limits();
        capped.max_constraint_elements = 1;
        assert_limit(
            constrained_zonotope_affine_unwired(&input, weights.view(), &[0.0, 0.0], capped),
            "constraint matrix elements",
        );
    }

    #[test]
    fn alpha_zero_constraint_rows_are_capped_independently() {
        let point_with_zero_width_rows = ConstrainedZonotope64::try_new(
            vec![2.0],
            Vec::new(),
            Array2::zeros((2, 0)),
            vec![0.0, 0.0],
            vec![0.0],
        )
        .unwrap();
        let weights = array![[1.0]];
        let mut no_constraint_rows = limits();
        no_constraint_rows.max_constraint_count = 0;
        no_constraint_rows.max_constraint_elements = 0;
        assert_limit(
            constrained_zonotope_affine_unwired(
                &point_with_zero_width_rows,
                weights.view(),
                &[0.0],
                no_constraint_rows,
            ),
            "constraint count",
        );
    }

    #[test]
    fn nonstandard_constraint_layout_is_preserved_logically() {
        let constraints = array![[1.0, 2.0], [-3.0, 4.0]].reversed_axes();
        assert!(constraints.as_slice().is_none());
        let input = ConstrainedZonotope64::try_new(
            vec![1.0, -2.0],
            vec![vec![(0, 0.5)], vec![(1, -0.25)]],
            constraints,
            vec![5.0, 6.0],
            vec![0.0, 0.0],
        )
        .unwrap();
        let weights = array![[1.0, 0.0], [0.0, 1.0]];
        let (output, _) =
            constrained_zonotope_affine_unwired(&input, weights.view(), &[0.0, 0.0], limits())
                .unwrap();
        assert_eq!(output.constraints(), input.constraints());
        assert_eq!(output.rhs(), input.rhs());
    }

    #[test]
    fn output_generator_growth_is_capped_after_sparse_merging() {
        let input = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![vec![(0, 1.0)]],
            Array2::zeros((0, 1)),
            Vec::new(),
            vec![0.0],
        )
        .unwrap();
        let weights = array![[1.0], [1.0]];
        let mut one_nonzero = limits();
        one_nonzero.max_generator_nonzeros = 1;
        assert_limit(
            constrained_zonotope_affine_unwired(&input, weights.view(), &[0.0, 0.0], one_nonzero),
            "output generator nonzeros",
        );
    }

    #[test]
    fn malformed_and_nonfinite_inputs_return_typed_errors() {
        let input = simple_input();
        let wrong_width = Array2::ones((1, 1));
        assert!(matches!(
            constrained_zonotope_affine_unwired(&input, wrong_width.view(), &[0.0], limits(),),
            Err(ConstrainedZonotopeAffineError::Shape {
                field: "weight input dimension",
                ..
            })
        ));

        let weights = Array2::ones((1, 2));
        assert!(matches!(
            constrained_zonotope_affine_unwired(&input, weights.view(), &[], limits()),
            Err(ConstrainedZonotopeAffineError::Shape { field: "bias", .. })
        ));

        let nonfinite_weight = array![[1.0, f64::NAN]];
        assert!(matches!(
            constrained_zonotope_affine_unwired(&input, nonfinite_weight.view(), &[0.0], limits(),),
            Err(ConstrainedZonotopeAffineError::NonFinite {
                field: "weights",
                index: 1
            })
        ));
        assert!(matches!(
            constrained_zonotope_affine_unwired(&input, weights.view(), &[f64::INFINITY], limits(),),
            Err(ConstrainedZonotopeAffineError::NonFinite {
                field: "bias",
                index: 0
            })
        ));

        let huge = ConstrainedZonotope64::try_new(
            vec![f64::MAX],
            Vec::new(),
            Array2::zeros((0, 0)),
            Vec::new(),
            vec![0.0],
        )
        .unwrap();
        assert!(matches!(
            constrained_zonotope_affine_unwired(&huge, array![[2.0]].view(), &[0.0], limits(),),
            Err(ConstrainedZonotopeAffineError::NonFiniteArithmetic { .. })
        ));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn randomized_ordinary_scale_contains_exact_rational_affine_image(
            center_raw in prop::collection::vec(-1000_i16..=1000, 5),
            generator_raw in prop::collection::vec(-1000_i16..=1000, 10),
            remainder_raw in prop::collection::vec(0_u8..=20, 5),
            weights_raw in prop::collection::vec(-1000_i16..=1000, 15),
            bias_raw in prop::collection::vec(-1000_i16..=1000, 3),
        ) {
            let scale = |value: i16| f64::from(value) / 64.0;
            let centers: Vec<f64> = center_raw.into_iter().map(scale).collect();
            let sparse_generators: Vec<Vec<(usize, f64)>> = generator_raw
                .as_chunks::<5>()
                .0
                .iter()
                .map(|values| {
                    values
                        .iter()
                        .copied()
                        .enumerate()
                        .filter_map(|(index, value)| {
                            let coefficient = scale(value);
                            (coefficient != 0.0).then_some((index, coefficient))
                        })
                        .collect()
                })
                .collect();
            let remainders: Vec<f64> = remainder_raw
                .into_iter()
                .map(|value| f64::from(value) / 4096.0)
                .collect();
            let input = ConstrainedZonotope64::try_new(
                centers,
                sparse_generators,
                Array2::zeros((0, 2)),
                Vec::new(),
                remainders,
            ).unwrap();
            let weights = Array2::from_shape_vec(
                (3, 5),
                weights_raw.into_iter().map(scale).collect(),
            ).unwrap();
            let bias: Vec<f64> = bias_raw.into_iter().map(scale).collect();
            let (output, _) = constrained_zonotope_affine_unwired(
                &input,
                weights.view(),
                &bias,
                limits(),
            ).unwrap();
            for output_index in 0..output.value_dim() {
                prop_assert!(
                    rat(output.box_remainder()[output_index])
                        >= exact_required_remainder(
                            &input,
                            &output,
                            weights.view(),
                            &bias,
                            output_index,
                        )
                );
            }
        }

        #[test]
        fn randomized_mixed_scale_contains_exact_rational_affine_image(
            center_raw in prop::collection::vec((any::<u64>(), -200_i16..=200), 4),
            generator_raw in prop::collection::vec((any::<u64>(), -200_i16..=200), 8),
            remainder_raw in prop::collection::vec((any::<u64>(), -500_i16..=-100), 4),
            weights_raw in prop::collection::vec((any::<u64>(), -200_i16..=200), 8),
            bias_raw in prop::collection::vec((any::<u64>(), -200_i16..=200), 2),
        ) {
            let centers: Vec<f64> = center_raw
                .into_iter()
                .map(|(raw, exponent)| bounded_normal(raw, exponent))
                .collect();
            let sparse_generators: Vec<Vec<(usize, f64)>> = generator_raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|values| {
                    values
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(index, (raw, exponent))| {
                            (index, bounded_normal(raw, exponent))
                        })
                        .collect()
                })
                .collect();
            let remainders: Vec<f64> = remainder_raw
                .into_iter()
                .map(|(raw, exponent)| {
                    bounded_normal(raw & !(1_u64 << 63), exponent)
                })
                .collect();
            let input = ConstrainedZonotope64::try_new(
                centers,
                sparse_generators,
                Array2::zeros((0, 2)),
                Vec::new(),
                remainders,
            ).unwrap();
            let weights = Array2::from_shape_vec(
                (2, 4),
                weights_raw
                    .into_iter()
                    .map(|(raw, exponent)| bounded_normal(raw, exponent))
                    .collect(),
            ).unwrap();
            let bias: Vec<f64> = bias_raw
                .into_iter()
                .map(|(raw, exponent)| bounded_normal(raw, exponent))
                .collect();
            let (output, _) = constrained_zonotope_affine_unwired(
                &input,
                weights.view(),
                &bias,
                limits(),
            ).unwrap();
            for output_index in 0..output.value_dim() {
                prop_assert!(
                    rat(output.box_remainder()[output_index])
                        >= exact_required_remainder(
                            &input,
                            &output,
                            weights.view(),
                            &bias,
                            output_index,
                        )
                );
            }
        }
    }
}
