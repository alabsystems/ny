// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Fail-closed outward evaluator for constrained-zonotope dual bounds.
//!
//! This module is the first, deliberately **unwired** constrained-zonotope
//! milestone.  It evaluates a caller-supplied nonnegative dual multiplier for
//!
//! ```text
//! X = { c + G alpha + e : alpha in [-1, 1]^m, C alpha <= d,
//!                           |e_i| <= r_i }.
//! ```
//!
//! For a direction `q`, let `g = G^T q`.  Any `lambda >= 0` gives the real-
//! arithmetic certificates
//!
//! ```text
//! lower = q c - lambda d - ||g + C^T lambda||_1,
//! upper = q c + lambda d + ||g - C^T lambda||_1,
//! box   = sum_i |q_i| r_i.
//! ```
//!
//! When a box remainder is supplied, `box` is subtracted from `lower` and
//! added to `upper` exactly once.  The legacy entry point has an implicit zero
//! remainder and remains bit-identical to the same evaluation with an explicit
//! all-zero remainder.
//!
//! Every input `f64` is interpreted as its **exact IEEE-754 dyadic value**.
//! Products and sums are enclosed immediately after each floating-point
//! operation, and interval operations preserve that enclosure.  This argument
//! requires IEEE-754 scalar `f64` operations with gradual underflow (no FTZ/DAZ
//! mode).  The public entry point probes gradual underflow with runtime-black-
//! boxed operands and fails closed when the requirement is not met.  If an
//! input, intermediate result, or outward endpoint is non-finite, evaluation
//! likewise fails closed instead of returning a certificate.
//!
//! # Soundness boundary
//!
//! This function does not construct a [`ny_tensor::zonotope::Star`], optimize
//! `lambda`, establish predicate feasibility, or admit a verifier verdict.  In
//! particular, converting the existing `f32` Star state to `f64` cannot recover
//! rounding error already introduced by its construction and transformers.
//! These bounds therefore do **not** make the current `f32` Star path
//! proof-safe and are not wired to any scored or verdict-producing path.
//!
//! Target triage for Metaroom119 found only 161 declared non-point inputs among
//! 5,376 input coordinates.  A guarded NY root probe found 4,956 unstable ReLUs
//! (2.88%), suggesting a sparse final axis near 5,117.  There is a precision
//! trap: NY's current `f32` VNNLIB conversion outward-widens exact decimal point
//! constraints, so reconstructing radii from a [`ny_tensor::BoundedTensor`] can
//! make all 5,376 inputs appear non-point.  The unwired
//! [`ConstrainedZonotope64::from_certified_bounds`] constructor accepts
//! separately certified `f64` enclosures and declared-point metadata, prunes
//! marked symbols without pruning their enclosure width, and charges rounding
//! deficits to a box remainder.  Parser qualification and batched/grouped
//! generator convolution remain future gates; the current
//! one-convolution-per-generator Star path must not become a dense scalar
//! convolution loop.

use ndarray::{ArrayView1, ArrayView2};

use crate::constrained_zonotope64::ConstrainedZonotope64;

/// A pair of finite, outward directional certificates.
///
/// Subject to the documented exact-IEEE input interpretation, every feasible
/// `alpha` and every independent box error satisfy
/// `lower <= q (c + G alpha + e) <= upper`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstrainedZonotopeDualBounds {
    /// Outward-rounded lower certificate.
    pub lower: f64,
    /// Outward-rounded upper certificate.
    pub upper: f64,
}

/// A malformed candidate or an arithmetic condition that cannot produce a
/// finite outward certificate.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConstrainedZonotopeDualError {
    /// One of the vector/matrix dimensions is inconsistent with the domain.
    #[error("shape mismatch for {field}: expected {expected:?}, got {got:?}")]
    Shape {
        /// Input whose shape is wrong.
        field: &'static str,
        /// Required shape.
        expected: Vec<usize>,
        /// Supplied shape.
        got: Vec<usize>,
    },

    /// An input coefficient is NaN or infinite.
    #[error("{field}[{index}] must be finite")]
    NonFiniteInput {
        /// Flattened input name.
        field: &'static str,
        /// Logical row-major index within the input.
        index: usize,
    },

    /// A Lagrange multiplier is negative.
    #[error("multipliers[{index}] must be nonnegative")]
    NegativeMultiplier {
        /// Index of the negative multiplier.
        index: usize,
    },

    /// An independent box-remainder radius is negative.
    #[error("box_remainder[{index}] must be nonnegative")]
    NegativeBoxRemainder {
        /// Index of the negative radius.
        index: usize,
    },

    /// A floating-point operation overflowed, or outward widening reached an
    /// infinity.  Returning an infinite endpoint would not be a useful finite
    /// certificate, so the evaluator fails closed.
    #[error("non-finite outward arithmetic while computing {operation}")]
    NonFiniteArithmetic {
        /// Stage at which finite containment became impossible.
        operation: &'static str,
    },

    /// The host floating-point environment does not preserve `f64`
    /// subnormals, invalidating one-successor/predecessor containment.
    #[error("unsupported floating-point environment: {requirement}")]
    UnsupportedFloatingPoint {
        /// Arithmetic property required by the evaluator.
        requirement: &'static str,
    },
}

/// Evaluate one constrained-zonotope direction with rigorous outward `f64`
/// arithmetic.
///
/// Shapes are `center: (n)`, `generators: (n, m)`, `constraints: (k, m)`,
/// `rhs: (k)`, `direction: (n)`, and `multipliers: (k)`.  Empty value,
/// generator, and constraint dimensions are supported when their shapes are
/// mutually consistent.
///
/// `multipliers` is a supplied dual candidate; this function does no LP solve.
/// A zero multiplier is valid and recovers the ordinary box-zonotope bound,
/// bit-for-bit through the same projection arithmetic.
///
/// # Errors
///
/// Returns [`ConstrainedZonotopeDualError`] on any shape mismatch, non-finite
/// input, negative multiplier, overflow, or non-finite outward endpoint.
pub fn evaluate_constrained_zonotope_dual(
    center: ArrayView1<'_, f64>,
    generators: ArrayView2<'_, f64>,
    constraints: ArrayView2<'_, f64>,
    rhs: ArrayView1<'_, f64>,
    direction: ArrayView1<'_, f64>,
    multipliers: ArrayView1<'_, f64>,
) -> Result<ConstrainedZonotopeDualBounds, ConstrainedZonotopeDualError> {
    evaluate_dense_constrained_zonotope_dual(
        center,
        generators,
        constraints,
        rhs,
        None,
        direction,
        multipliers,
    )
}

/// Evaluate a dense constrained zonotope plus an independent box remainder.
///
/// This has the same exact-IEEE input interpretation and fail-closed behavior
/// as [`evaluate_constrained_zonotope_dual`].  `box_remainder` has shape `(n)`,
/// must be finite and nonnegative, and contributes `sum_i |q_i| r_i` exactly
/// once to each directional endpoint.
pub fn evaluate_constrained_zonotope_dual_with_box_remainder(
    center: ArrayView1<'_, f64>,
    generators: ArrayView2<'_, f64>,
    constraints: ArrayView2<'_, f64>,
    rhs: ArrayView1<'_, f64>,
    box_remainder: ArrayView1<'_, f64>,
    direction: ArrayView1<'_, f64>,
    multipliers: ArrayView1<'_, f64>,
) -> Result<ConstrainedZonotopeDualBounds, ConstrainedZonotopeDualError> {
    evaluate_dense_constrained_zonotope_dual(
        center,
        generators,
        constraints,
        rhs,
        Some(box_remainder),
        direction,
        multipliers,
    )
}

fn evaluate_dense_constrained_zonotope_dual(
    center: ArrayView1<'_, f64>,
    generators: ArrayView2<'_, f64>,
    constraints: ArrayView2<'_, f64>,
    rhs: ArrayView1<'_, f64>,
    box_remainder: Option<ArrayView1<'_, f64>>,
    direction: ArrayView1<'_, f64>,
    multipliers: ArrayView1<'_, f64>,
) -> Result<ConstrainedZonotopeDualBounds, ConstrainedZonotopeDualError> {
    validate_shapes(center, generators, constraints, rhs, direction, multipliers)?;
    if let Some(remainder) = box_remainder {
        if remainder.len() != center.len() {
            return Err(ConstrainedZonotopeDualError::Shape {
                field: "box_remainder",
                expected: vec![center.len()],
                got: vec![remainder.len()],
            });
        }
    }
    validate_finite("center", center.iter().copied())?;
    validate_finite("generators", generators.iter().copied())?;
    validate_finite("constraints", constraints.iter().copied())?;
    validate_finite("rhs", rhs.iter().copied())?;
    validate_finite("direction", direction.iter().copied())?;
    validate_finite("multipliers", multipliers.iter().copied())?;
    if let Some(remainder) = box_remainder {
        validate_finite("box_remainder", remainder.iter().copied())?;
        for (index, &radius) in remainder.iter().enumerate() {
            if radius < 0.0 {
                return Err(ConstrainedZonotopeDualError::NegativeBoxRemainder { index });
            }
        }
    }
    validate_nonnegative_multipliers(multipliers.iter().copied())?;
    require_gradual_underflow()?;

    let value_dim = center.len();
    let alpha_dim = generators.ncols();
    let constraint_count = constraints.nrows();

    let projected_center = outward_dot(
        (0..value_dim).map(|index| (direction[index], center[index])),
        "q dot center",
    )?;
    let multiplier_rhs = outward_dot(
        (0..constraint_count).map(|row| (multipliers[row], rhs[row])),
        "lambda dot rhs",
    )?;

    let box_projection = if let Some(remainder) = box_remainder {
        outward_dot(
            (0..value_dim).map(|index| (direction[index].abs(), remainder[index])),
            "direction times box remainder",
        )?
    } else {
        OutwardInterval::zero()
    };

    evaluate_projected_dual(
        projected_center,
        multiplier_rhs,
        alpha_dim,
        constraints,
        multipliers,
        |column| {
            outward_dot(
                (0..value_dim).map(|row| (direction[row], generators[[row, column]])),
                "G transpose times q",
            )
        },
        box_projection,
    )
}

/// Evaluate a sparse [`ConstrainedZonotope64`] direction with a supplied
/// nonnegative dual candidate.
///
/// This performs no optimization.  It consumes every stored constraint row,
/// making it compatible with a proposer that supplies multipliers for a full
/// `C alpha <= d` system while keeping certification independent of that
/// proposer.  The domain's independent box remainder is charged exactly once.
pub fn evaluate_constrained_zonotope64_dual(
    domain: &ConstrainedZonotope64,
    direction: &[f64],
    multipliers: &[f64],
) -> Result<ConstrainedZonotopeDualBounds, ConstrainedZonotopeDualError> {
    let value_dim = domain.value_dim();
    let alpha_dim = domain.alpha_dim();
    let constraint_count = domain.constraint_count();
    if direction.len() != value_dim {
        return Err(ConstrainedZonotopeDualError::Shape {
            field: "direction",
            expected: vec![value_dim],
            got: vec![direction.len()],
        });
    }
    if multipliers.len() != constraint_count {
        return Err(ConstrainedZonotopeDualError::Shape {
            field: "multipliers",
            expected: vec![constraint_count],
            got: vec![multipliers.len()],
        });
    }
    validate_finite("direction", direction.iter().copied())?;
    validate_finite("multipliers", multipliers.iter().copied())?;
    validate_nonnegative_multipliers(multipliers.iter().copied())?;
    require_gradual_underflow()?;

    let projected_center = outward_dot(
        (0..value_dim).map(|index| (direction[index], domain.center()[index])),
        "q dot center",
    )?;
    let multiplier_rhs = outward_dot(
        (0..constraint_count).map(|row| (multipliers[row], domain.rhs()[row])),
        "lambda dot rhs",
    )?;
    let box_projection = outward_dot(
        (0..value_dim).map(|index| (direction[index].abs(), domain.box_remainder()[index])),
        "direction times box remainder",
    )?;

    evaluate_projected_dual(
        projected_center,
        multiplier_rhs,
        alpha_dim,
        domain.constraints_ref().view(),
        ArrayView1::from(multipliers),
        |column| {
            outward_dot(
                domain.generators()[column]
                    .raw_entries()
                    .map(|(row, coefficient)| (direction[row], coefficient)),
                "sparse G transpose times q",
            )
        },
        box_projection,
    )
}

fn evaluate_projected_dual(
    projected_center: OutwardInterval,
    multiplier_rhs: OutwardInterval,
    alpha_dim: usize,
    constraints: ArrayView2<'_, f64>,
    multipliers: ArrayView1<'_, f64>,
    mut projected_generator: impl FnMut(usize) -> Result<OutwardInterval, ConstrainedZonotopeDualError>,
    box_projection: OutwardInterval,
) -> Result<ConstrainedZonotopeDualBounds, ConstrainedZonotopeDualError> {
    let constraint_count = constraints.nrows();

    let mut norm_plus = OutwardInterval::zero();
    let mut norm_minus = OutwardInterval::zero();
    for column in 0..alpha_dim {
        let projected_generator = projected_generator(column)?;
        let constraint_projection = outward_dot(
            (0..constraint_count).map(|row| (constraints[[row, column]], multipliers[row])),
            "C transpose times lambda",
        )?;

        let plus = projected_generator
            .add(constraint_projection, "g plus C transpose lambda")?
            .abs();
        let minus = projected_generator
            .sub(constraint_projection, "g minus C transpose lambda")?
            .abs();
        norm_plus = norm_plus.add(plus, "lower one-norm reduction")?;
        norm_minus = norm_minus.add(minus, "upper one-norm reduction")?;
    }

    let lower = projected_center
        .sub(multiplier_rhs, "lower center minus lambda rhs")?
        .sub(norm_plus, "lower certificate")?
        .sub(box_projection, "lower box remainder")?
        .lo;
    let upper = projected_center
        .add(multiplier_rhs, "upper center plus lambda rhs")?
        .add(norm_minus, "upper certificate")?
        .add(box_projection, "upper box remainder")?
        .hi;

    // Every operation above rejects non-finite endpoints.  Keep this final
    // guard at the public boundary so later refactors cannot weaken fail-closed
    // behavior by accidentally constructing an interval directly.
    if !lower.is_finite() || !upper.is_finite() {
        return Err(ConstrainedZonotopeDualError::NonFiniteArithmetic {
            operation: "final certificates",
        });
    }
    Ok(ConstrainedZonotopeDualBounds { lower, upper })
}

fn validate_nonnegative_multipliers(
    multipliers: impl IntoIterator<Item = f64>,
) -> Result<(), ConstrainedZonotopeDualError> {
    for (index, multiplier) in multipliers.into_iter().enumerate() {
        if multiplier < 0.0 {
            return Err(ConstrainedZonotopeDualError::NegativeMultiplier { index });
        }
    }
    Ok(())
}

fn validate_shapes(
    center: ArrayView1<'_, f64>,
    generators: ArrayView2<'_, f64>,
    constraints: ArrayView2<'_, f64>,
    rhs: ArrayView1<'_, f64>,
    direction: ArrayView1<'_, f64>,
    multipliers: ArrayView1<'_, f64>,
) -> Result<(), ConstrainedZonotopeDualError> {
    let value_dim = center.len();
    if direction.len() != value_dim {
        return Err(ConstrainedZonotopeDualError::Shape {
            field: "direction",
            expected: vec![value_dim],
            got: vec![direction.len()],
        });
    }
    if generators.nrows() != value_dim {
        return Err(ConstrainedZonotopeDualError::Shape {
            field: "generators",
            expected: vec![value_dim, generators.ncols()],
            got: generators.shape().to_vec(),
        });
    }

    let alpha_dim = generators.ncols();
    if constraints.ncols() != alpha_dim {
        return Err(ConstrainedZonotopeDualError::Shape {
            field: "constraints",
            expected: vec![constraints.nrows(), alpha_dim],
            got: constraints.shape().to_vec(),
        });
    }

    let constraint_count = constraints.nrows();
    if rhs.len() != constraint_count {
        return Err(ConstrainedZonotopeDualError::Shape {
            field: "rhs",
            expected: vec![constraint_count],
            got: vec![rhs.len()],
        });
    }
    if multipliers.len() != constraint_count {
        return Err(ConstrainedZonotopeDualError::Shape {
            field: "multipliers",
            expected: vec![constraint_count],
            got: vec![multipliers.len()],
        });
    }
    Ok(())
}

fn validate_finite(
    field: &'static str,
    values: impl IntoIterator<Item = f64>,
) -> Result<(), ConstrainedZonotopeDualError> {
    for (index, value) in values.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(ConstrainedZonotopeDualError::NonFiniteInput { field, index });
        }
    }
    Ok(())
}

/// Reject FTZ/DAZ modes before relying on adjacent-float containment.
///
/// `black_box` prevents these probes from being constant-folded into the
/// compiler's abstract IEEE model: they must observe the active scalar `f64`
/// environment used by the following arithmetic.
fn require_gradual_underflow() -> Result<(), ConstrainedZonotopeDualError> {
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
        return Err(ConstrainedZonotopeDualError::UnsupportedFloatingPoint {
            requirement: "IEEE-754 binary64 gradual underflow (FTZ/DAZ disabled)",
        });
    }
    Ok(())
}

/// Closed interval whose finite endpoints enclose an exact real expression.
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

    fn add(self, rhs: Self, operation: &'static str) -> Result<Self, ConstrainedZonotopeDualError> {
        // These identities are exact in real arithmetic and preserve the
        // lambda=0 path without accumulating gratuitous subnormal width.
        if self.is_exact_zero() {
            return Ok(rhs);
        }
        if rhs.is_exact_zero() {
            return Ok(self);
        }
        let lo = round_down(self.lo + rhs.lo, operation)?;
        let hi = round_up(self.hi + rhs.hi, operation)?;
        Ok(Self { lo, hi })
    }

    fn sub(self, rhs: Self, operation: &'static str) -> Result<Self, ConstrainedZonotopeDualError> {
        self.add(rhs.neg(), operation)
    }

    const fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }

    fn abs(self) -> Self {
        if self.lo >= 0.0 {
            self
        } else if self.hi <= 0.0 {
            self.neg()
        } else {
            Self {
                lo: 0.0,
                hi: (-self.lo).max(self.hi),
            }
        }
    }
}

fn outward_dot(
    pairs: impl IntoIterator<Item = (f64, f64)>,
    operation: &'static str,
) -> Result<OutwardInterval, ConstrainedZonotopeDualError> {
    let mut sum = OutwardInterval::zero();
    for (left, right) in pairs {
        let product = outward_product(left, right, operation)?;
        sum = sum.add(product, operation)?;
    }
    Ok(sum)
}

fn outward_product(
    left: f64,
    right: f64,
    operation: &'static str,
) -> Result<OutwardInterval, ConstrainedZonotopeDualError> {
    // Exact identities avoid widening structural zeros/ones.  Inputs were
    // already checked finite, so none of the IEEE 0*inf edge cases apply.
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
    let lo = round_down(product, operation)?;
    let hi = round_up(product, operation)?;
    Ok(OutwardInterval { lo, hi })
}

fn round_down(value: f64, operation: &'static str) -> Result<f64, ConstrainedZonotopeDualError> {
    if !value.is_finite() {
        return Err(ConstrainedZonotopeDualError::NonFiniteArithmetic { operation });
    }
    let outward = value.next_down();
    if !outward.is_finite() {
        return Err(ConstrainedZonotopeDualError::NonFiniteArithmetic { operation });
    }
    Ok(outward)
}

fn round_up(value: f64, operation: &'static str) -> Result<f64, ConstrainedZonotopeDualError> {
    if !value.is_finite() {
        return Err(ConstrainedZonotopeDualError::NonFiniteArithmetic { operation });
    }
    let outward = value.next_up();
    if !outward.is_finite() {
        return Err(ConstrainedZonotopeDualError::NonFiniteArithmetic { operation });
    }
    Ok(outward)
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array1, Array2};
    use num_rational::BigRational;
    use num_traits::Zero;
    use proptest::prelude::*;

    use super::*;

    fn rat(value: f64) -> BigRational {
        BigRational::from_float(value).expect("test inputs and outputs are finite")
    }

    fn exact_bounds(
        center: &[f64],
        generators: &Array2<f64>,
        constraints: &Array2<f64>,
        rhs: &[f64],
        direction: &[f64],
        multipliers: &[f64],
    ) -> (BigRational, BigRational) {
        let dot = |left: &[f64], right: &[f64]| {
            left.iter()
                .zip(right)
                .fold(BigRational::zero(), |sum, (&a, &b)| sum + rat(a) * rat(b))
        };
        let projected_center = dot(direction, center);
        let multiplier_rhs = dot(multipliers, rhs);

        let mut norm_plus = BigRational::zero();
        let mut norm_minus = BigRational::zero();
        for column in 0..generators.ncols() {
            let mut projected_generator = BigRational::zero();
            for row in 0..generators.nrows() {
                projected_generator += rat(direction[row]) * rat(generators[[row, column]]);
            }
            let mut constraint_projection = BigRational::zero();
            for row in 0..constraints.nrows() {
                constraint_projection += rat(constraints[[row, column]]) * rat(multipliers[row]);
            }
            let plus = projected_generator.clone() + constraint_projection.clone();
            let minus = projected_generator - constraint_projection;
            norm_plus += if plus < BigRational::zero() {
                -plus
            } else {
                plus
            };
            norm_minus += if minus < BigRational::zero() {
                -minus
            } else {
                minus
            };
        }

        (
            projected_center.clone() - multiplier_rhs.clone() - norm_plus,
            projected_center + multiplier_rhs + norm_minus,
        )
    }

    fn evaluate_owned(
        center: &Array1<f64>,
        generators: &Array2<f64>,
        constraints: &Array2<f64>,
        rhs: &Array1<f64>,
        direction: &Array1<f64>,
        multipliers: &Array1<f64>,
    ) -> Result<ConstrainedZonotopeDualBounds, ConstrainedZonotopeDualError> {
        evaluate_constrained_zonotope_dual(
            center.view(),
            generators.view(),
            constraints.view(),
            rhs.view(),
            direction.view(),
            multipliers.view(),
        )
    }

    /// Materialize a finite normal with an arbitrary 52-bit mantissa at a
    /// bounded binary exponent.  This produces difficult exact-IEEE inputs
    /// without risking the overflow/underflow cases tested separately.
    fn bounded_normal(raw: u64, exponent: i32) -> f64 {
        assert!((-1_022..=1_023).contains(&exponent));
        let sign = raw & (1_u64 << 63);
        let fraction = raw & ((1_u64 << 52) - 1);
        let biased_exponent = u64::try_from(exponent + 1_023).unwrap();
        f64::from_bits(sign | (biased_exponent << 52) | fraction)
    }

    #[test]
    fn dyadic_toy_matches_exact_python_oracle_and_feasible_grid() {
        // This is also pinned by scripts/constrained_zonotope_dual_oracle.py:
        // exact leq_rhs results are lower=3/8 and upper=41/8.
        let center = array![1.5];
        let generators = array![[2.0, -0.25]];
        let constraints = array![[-1.0, 0.0], [0.0, 1.0]];
        let rhs = array![0.0, 0.5];
        let direction = array![1.0];
        let multipliers = array![1.0, 0.25];
        let bounds = evaluate_owned(
            &center,
            &generators,
            &constraints,
            &rhs,
            &direction,
            &multipliers,
        )
        .unwrap();

        assert!(rat(bounds.lower) <= BigRational::new(3.into(), 8.into()));
        assert!(rat(bounds.upper) >= BigRational::new(41.into(), 8.into()));

        // Exhaust the 1/8-spaced toy grid.  Every point satisfying C alpha <= d
        // must lie inside the certified directional enclosure.
        for a0_integer in -8..=8 {
            for a1_integer in -8..=8 {
                let a0 = f64::from(a0_integer) / 8.0;
                let a1 = f64::from(a1_integer) / 8.0;
                if -a0 > 0.0 || a1 > 0.5 {
                    continue;
                }
                let concrete = 1.5 + 2.0 * a0 - 0.25 * a1;
                assert!(bounds.lower <= concrete, "alpha=({a0}, {a1})");
                assert!(concrete <= bounds.upper, "alpha=({a0}, {a1})");
            }
        }
    }

    #[test]
    fn lambda_zero_is_bit_identical_to_empty_predicate() {
        let center = array![0.25, -1.5];
        let generators = array![[0.5, -0.75, 0.125], [1.0, 0.25, -0.5]];
        let direction = array![0.75, -0.25];

        let constraints = array![[3.0, -4.0, 0.5], [-1.0, 2.0, 7.0]];
        let rhs = array![9.0, -3.0];
        let multipliers = array![0.0, -0.0];
        let constrained = evaluate_owned(
            &center,
            &generators,
            &constraints,
            &rhs,
            &direction,
            &multipliers,
        )
        .unwrap();

        let no_constraints = Array2::zeros((0, generators.ncols()));
        let empty = Array1::zeros(0);
        let unconstrained = evaluate_owned(
            &center,
            &generators,
            &no_constraints,
            &empty,
            &direction,
            &empty,
        )
        .unwrap();
        assert_eq!(constrained, unconstrained);
    }

    #[test]
    fn explicit_zero_remainder_is_bit_identical_to_legacy_evaluator() {
        let center = array![0.25, -1.5];
        let generators = array![[0.5, -0.75, 0.125], [1.0, 0.25, -0.5]];
        let constraints = array![[3.0, -4.0, 0.5], [-1.0, 2.0, 7.0]];
        let rhs = array![9.0, -3.0];
        let direction = array![0.75, -0.25];
        let multipliers = array![0.5, 0.125];
        let legacy = evaluate_owned(
            &center,
            &generators,
            &constraints,
            &rhs,
            &direction,
            &multipliers,
        )
        .unwrap();
        let explicit = evaluate_constrained_zonotope_dual_with_box_remainder(
            center.view(),
            generators.view(),
            constraints.view(),
            rhs.view(),
            array![0.0, -0.0].view(),
            direction.view(),
            multipliers.view(),
        )
        .unwrap();
        assert_eq!(legacy, explicit);
        assert_eq!(legacy.lower.to_bits(), explicit.lower.to_bits());
        assert_eq!(legacy.upper.to_bits(), explicit.upper.to_bits());
    }

    #[test]
    fn box_remainder_is_charged_once_in_lambda_formula() {
        let center = array![1.5];
        let generators = array![[2.0, -0.25]];
        let constraints = array![[-1.0, 0.0], [0.0, 1.0]];
        let rhs = array![0.0, 0.5];
        let remainder = array![0.125];
        let direction = array![1.0];
        let multipliers = array![1.0, 0.25];
        let bounds = evaluate_constrained_zonotope_dual_with_box_remainder(
            center.view(),
            generators.view(),
            constraints.view(),
            rhs.view(),
            remainder.view(),
            direction.view(),
            multipliers.view(),
        )
        .unwrap();

        let exact_lower = BigRational::new(1.into(), 4.into());
        let exact_upper = BigRational::new(21.into(), 4.into());
        assert!(rat(bounds.lower) <= exact_lower);
        assert!(rat(bounds.upper) >= exact_upper);
        // A double charge would reach 1/8 and 43/8; outward noise from the
        // single charge is many orders of magnitude smaller here.
        assert!(bounds.lower > 0.125);
        assert!(bounds.upper < 5.375);
    }

    #[test]
    fn malformed_box_remainders_fail_closed() {
        let center = array![0.0];
        let generators = array![[1.0]];
        let constraints = Array2::zeros((0, 1));
        let empty = Array1::zeros(0);
        let direction = array![1.0];

        let cases = [
            evaluate_constrained_zonotope_dual_with_box_remainder(
                center.view(),
                generators.view(),
                constraints.view(),
                empty.view(),
                Array1::zeros(0).view(),
                direction.view(),
                empty.view(),
            ),
            evaluate_constrained_zonotope_dual_with_box_remainder(
                center.view(),
                generators.view(),
                constraints.view(),
                empty.view(),
                array![f64::NAN].view(),
                direction.view(),
                empty.view(),
            ),
            evaluate_constrained_zonotope_dual_with_box_remainder(
                center.view(),
                generators.view(),
                constraints.view(),
                empty.view(),
                array![-f64::MIN_POSITIVE].view(),
                direction.view(),
                empty.view(),
            ),
        ];
        assert!(matches!(
            &cases[0],
            Err(ConstrainedZonotopeDualError::Shape {
                field: "box_remainder",
                ..
            })
        ));
        assert!(matches!(
            &cases[1],
            Err(ConstrainedZonotopeDualError::NonFiniteInput {
                field: "box_remainder",
                ..
            })
        ));
        assert!(matches!(
            &cases[2],
            Err(ConstrainedZonotopeDualError::NegativeBoxRemainder { .. })
        ));
    }

    #[test]
    fn every_shape_mismatch_is_rejected_before_arithmetic() {
        let center = array![0.0, 0.0];
        let generators = Array2::zeros((2, 3));
        let constraints = Array2::zeros((2, 3));
        let rhs = array![0.0, 0.0];
        let direction = array![0.0, 0.0];
        let multipliers = array![0.0, 0.0];

        let cases = [
            evaluate_constrained_zonotope_dual(
                center.view(),
                Array2::zeros((1, 3)).view(),
                constraints.view(),
                rhs.view(),
                direction.view(),
                multipliers.view(),
            ),
            evaluate_constrained_zonotope_dual(
                center.view(),
                generators.view(),
                Array2::zeros((2, 4)).view(),
                rhs.view(),
                direction.view(),
                multipliers.view(),
            ),
            evaluate_constrained_zonotope_dual(
                center.view(),
                generators.view(),
                constraints.view(),
                array![0.0].view(),
                direction.view(),
                multipliers.view(),
            ),
            evaluate_constrained_zonotope_dual(
                center.view(),
                generators.view(),
                constraints.view(),
                rhs.view(),
                array![0.0].view(),
                multipliers.view(),
            ),
            evaluate_constrained_zonotope_dual(
                center.view(),
                generators.view(),
                constraints.view(),
                rhs.view(),
                direction.view(),
                array![0.0].view(),
            ),
        ];
        for result in cases {
            assert!(matches!(
                result,
                Err(ConstrainedZonotopeDualError::Shape { .. })
            ));
        }
    }

    #[test]
    fn every_input_family_rejects_nonfinite_values() {
        let center = array![0.0];
        let generators = array![[1.0]];
        let constraints = array![[1.0]];
        let rhs = array![1.0];
        let direction = array![1.0];
        let multipliers = array![1.0];

        let bad_center = array![f64::NAN];
        let bad_generators = array![[f64::INFINITY]];
        let bad_constraints = array![[f64::NEG_INFINITY]];
        let bad_rhs = array![f64::NAN];
        let bad_direction = array![f64::INFINITY];
        let bad_multipliers = array![f64::NEG_INFINITY];
        let cases = [
            evaluate_owned(
                &bad_center,
                &generators,
                &constraints,
                &rhs,
                &direction,
                &multipliers,
            ),
            evaluate_owned(
                &center,
                &bad_generators,
                &constraints,
                &rhs,
                &direction,
                &multipliers,
            ),
            evaluate_owned(
                &center,
                &generators,
                &bad_constraints,
                &rhs,
                &direction,
                &multipliers,
            ),
            evaluate_owned(
                &center,
                &generators,
                &constraints,
                &bad_rhs,
                &direction,
                &multipliers,
            ),
            evaluate_owned(
                &center,
                &generators,
                &constraints,
                &rhs,
                &bad_direction,
                &multipliers,
            ),
            evaluate_owned(
                &center,
                &generators,
                &constraints,
                &rhs,
                &direction,
                &bad_multipliers,
            ),
        ];
        for result in cases {
            assert!(matches!(
                result,
                Err(ConstrainedZonotopeDualError::NonFiniteInput { .. })
            ));
        }
    }

    #[test]
    fn negative_multiplier_and_overflow_fail_closed() {
        let center = array![0.0];
        let generators = array![[1.0]];
        let constraints = array![[1.0]];
        let rhs = array![0.0];
        let direction = array![1.0];
        let negative = array![-f64::MIN_POSITIVE];
        assert!(matches!(
            evaluate_owned(
                &center,
                &generators,
                &constraints,
                &rhs,
                &direction,
                &negative,
            ),
            Err(ConstrainedZonotopeDualError::NegativeMultiplier { index: 0 })
        ));

        let overflowing_center = array![f64::MAX];
        let no_generators = Array2::zeros((1, 0));
        let no_constraints = Array2::zeros((0, 0));
        let empty = Array1::zeros(0);
        let double = array![2.0];
        assert!(matches!(
            evaluate_owned(
                &overflowing_center,
                &no_generators,
                &no_constraints,
                &empty,
                &double,
                &empty,
            ),
            Err(ConstrainedZonotopeDualError::NonFiniteArithmetic { .. })
        ));

        // Even when the nearest operation is finite, an infinite outward
        // neighbor is rejected instead of being published as a certificate.
        assert!(matches!(
            round_up(f64::MAX, "test outward endpoint"),
            Err(ConstrainedZonotopeDualError::NonFiniteArithmetic { .. })
        ));
    }

    #[test]
    fn subnormal_underflow_and_near_zero_accumulation_are_enclosed() {
        require_gradual_underflow().unwrap();

        let half_normal = outward_product(f64::MIN_POSITIVE, 0.5, "half min normal").unwrap();
        let exact_half_normal = rat(f64::MIN_POSITIVE) * rat(0.5);
        assert!(rat(half_normal.lo) <= exact_half_normal);
        assert!(rat(half_normal.hi) >= exact_half_normal);

        // The exact product is 2^-1075, halfway below the minimum f64
        // subnormal.  Round-to-nearest produces zero, but the adjacent outward
        // endpoints must still contain the exact rational product.
        let min_subnormal = f64::from_bits(1);
        let below_subnormal = outward_product(min_subnormal, 0.5, "below subnormal").unwrap();
        let exact_below_subnormal = rat(min_subnormal) * rat(0.5);
        assert!(rat(below_subnormal.lo) <= exact_below_subnormal);
        assert!(rat(below_subnormal.hi) >= exact_below_subnormal);

        // Cancel two exact subnormal products, then retain one minimum
        // subnormal.  This exercises interval accumulation across zero.
        let cancellation = outward_dot(
            [
                (f64::MIN_POSITIVE, 0.5),
                (-f64::MIN_POSITIVE, 0.5),
                (min_subnormal, 1.0),
            ],
            "subnormal cancellation",
        )
        .unwrap();
        let exact_cancellation = rat(min_subnormal);
        assert!(rat(cancellation.lo) <= exact_cancellation);
        assert!(rat(cancellation.hi) >= exact_cancellation);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        #[test]
        fn random_outward_results_enclose_exact_big_rational_formula(
            center_raw in prop::collection::vec(-512i16..=512, 3),
            generators_raw in prop::collection::vec(-512i16..=512, 12),
            constraints_raw in prop::collection::vec(-128i16..=128, 8),
            direction_raw in prop::collection::vec(-128i16..=128, 3),
            multipliers_raw in prop::collection::vec(0u16..=128, 2),
        ) {
            // Dyadic scaling keeps generated magnitudes bounded while the exact
            // oracle still interprets each materialized f64 coefficient itself.
            let center = Array1::from_iter(center_raw.into_iter().map(|v| f64::from(v) / 16.0));
            let generators = Array2::from_shape_vec(
                (3, 4),
                generators_raw.into_iter().map(|v| f64::from(v) / 16.0).collect(),
            ).unwrap();
            let constraints = Array2::from_shape_vec(
                (2, 4),
                constraints_raw.into_iter().map(|v| f64::from(v) / 16.0).collect(),
            ).unwrap();
            let direction = Array1::from_iter(
                direction_raw.into_iter().map(|v| f64::from(v) / 16.0),
            );
            let multipliers = Array1::from_iter(
                multipliers_raw.into_iter().map(|v| f64::from(v) / 16.0),
            );
            // Make the entire alpha box feasible: d_i = ||C_i||_1 + 1.
            let rhs = Array1::from_iter((0..2).map(|row| {
                constraints.row(row).iter().map(|value| value.abs()).sum::<f64>() + 1.0
            }));

            let outward = evaluate_owned(
                &center,
                &generators,
                &constraints,
                &rhs,
                &direction,
                &multipliers,
            ).unwrap();
            let (exact_lower, exact_upper) = exact_bounds(
                center.as_slice().unwrap(),
                &generators,
                &constraints,
                rhs.as_slice().unwrap(),
                direction.as_slice().unwrap(),
                multipliers.as_slice().unwrap(),
            );

            prop_assert!(rat(outward.lower) <= exact_lower,
                "outward lower {} exceeded exact {}", outward.lower, exact_lower);
            prop_assert!(rat(outward.upper) >= exact_upper,
                "outward upper {} fell below exact {}", outward.upper, exact_upper);
            prop_assert!(exact_lower <= exact_upper);
        }

        #[test]
        fn mixed_scale_random_mantissas_enclose_exact_big_rational_formula(
            shared_raw in any::<u64>(),
            shared_direction_raw in any::<u64>(),
            center_tail_raw in prop::collection::vec(any::<u64>(), 2),
            direction_tail_raw in prop::collection::vec(any::<u64>(), 2),
            generators_raw in prop::collection::vec(any::<u64>(), 20),
            constraints_raw in prop::collection::vec(any::<u64>(), 15),
            multipliers_raw in prop::collection::vec(any::<u64>(), 3),
        ) {
            // The first two center/direction terms nearly cancel after products
            // with independently nontrivial mantissas.  The remaining values
            // span 400 binary exponents, forcing rounded products and lossy
            // accumulations while remaining far from overflow and underflow.
            let shared = bounded_normal(shared_raw, 200).abs();
            let shared_direction = bounded_normal(shared_direction_raw, -200).abs();
            let center = array![
                shared,
                shared.next_up(),
                bounded_normal(center_tail_raw[0], -200),
                bounded_normal(center_tail_raw[1], 20),
            ];
            let direction = array![
                shared_direction,
                -shared_direction,
                bounded_normal(direction_tail_raw[0], 200),
                bounded_normal(direction_tail_raw[1], -20),
            ];

            const GENERATOR_EXPONENTS: [i32; 5] = [-200, -20, 0, 20, 200];
            let generators = Array2::from_shape_vec(
                (4, 5),
                generators_raw
                    .into_iter()
                    .enumerate()
                    .map(|(index, raw)| bounded_normal(raw, GENERATOR_EXPONENTS[index % 5]))
                    .collect(),
            ).unwrap();
            const CONSTRAINT_EXPONENTS: [i32; 5] = [-100, -10, 0, 10, 100];
            let constraints = Array2::from_shape_vec(
                (3, 5),
                constraints_raw
                    .into_iter()
                    .enumerate()
                    .map(|(index, raw)| bounded_normal(raw, CONSTRAINT_EXPONENTS[index % 5]))
                    .collect(),
            ).unwrap();
            const MULTIPLIER_EXPONENTS: [i32; 3] = [-100, 0, 100];
            let multipliers = Array1::from_iter(
                multipliers_raw
                    .into_iter()
                    .enumerate()
                    .map(|(index, raw)| {
                        bounded_normal(raw, MULTIPLIER_EXPONENTS[index]).abs()
                    }),
            );

            // Every C row has box maximum below 2^104, so rhs=2^300 makes
            // the complete alpha box feasible without itself requiring a
            // rounded reduction.
            let rhs_value = bounded_normal(0, 300);
            let rhs = array![rhs_value, rhs_value, rhs_value];

            let outward = evaluate_owned(
                &center,
                &generators,
                &constraints,
                &rhs,
                &direction,
                &multipliers,
            ).unwrap();
            let (exact_lower, exact_upper) = exact_bounds(
                center.as_slice().unwrap(),
                &generators,
                &constraints,
                rhs.as_slice().unwrap(),
                direction.as_slice().unwrap(),
                multipliers.as_slice().unwrap(),
            );

            prop_assert!(rat(outward.lower) <= exact_lower,
                "mixed-scale lower {} exceeded exact {}", outward.lower, exact_lower);
            prop_assert!(rat(outward.upper) >= exact_upper,
                "mixed-scale upper {} fell below exact {}", outward.upper, exact_upper);
            prop_assert!(exact_lower <= exact_upper);
        }
    }
}
