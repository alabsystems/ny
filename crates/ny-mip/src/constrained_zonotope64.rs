// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Finite `f64` storage for an unwired constrained zonotope with box remainder.
//!
//! [`ConstrainedZonotope64`] represents
//!
//! ```text
//! { c + G alpha + e : alpha in [-1, 1]^m, C alpha <= d,
//!                       |e_i| <= r_i }
//! ```
//!
//! with sparse generator columns and an independent, nonnegative box
//! remainder `r`.  Every stored `f64` is interpreted as its exact IEEE-754
//! dyadic value.  The certified-bounds constructor deliberately treats the
//! caller's declared-point mask as a decomposition hint only: a marked
//! coordinate gets no `alpha` symbol, but its complete supplied enclosure is
//! retained in the box remainder.  It can therefore never turn a nonzero-width
//! enclosure into a point.
//!
//! This is a deliberately **unwired** construction milestone.  The gated
//! `ny-cli` Metaroom-stem experiment can now supply certified VNN-LIB
//! enclosures to this type, but no command or scored verifier calls that seam.
//! General affine transforms, generator reduction, a complete network
//! pipeline, and verdict/scored-path integration remain unqualified.
//! In particular, this module does not make an enclosure reconstructed from an
//! already-rounded `f32` tensor proof-safe.  The separate unwired
//! `constrained_zonotope_conv2d` and `constrained_zonotope_relu` modules now
//! supply exact-dyadic propagation for those operations; the remaining
//! network transforms and scored-path wiring stay future gates.

use ndarray::{Array2, ArrayView2};
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};

use crate::constrained_zonotope_dual::{
    evaluate_constrained_zonotope64_dual, ConstrainedZonotopeDualBounds,
    ConstrainedZonotopeDualError,
};

/// One nonzero coefficient in a sparse generator column.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SparseGeneratorEntry64 {
    value_index: usize,
    coefficient: f64,
}

/// A canonical sparse generator column.
///
/// Entries are finite, nonzero, in range for their owning domain, and strictly
/// ordered by value index.  The private representation can only be created by
/// the validating [`ConstrainedZonotope64`] constructors.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseGenerator64 {
    entries: Vec<SparseGeneratorEntry64>,
}

impl SparseGenerator64 {
    /// Number of explicitly stored nonzero coefficients.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Iterate through `(value_index, coefficient)` pairs in increasing index
    /// order.
    pub fn entries(&self) -> impl ExactSizeIterator<Item = (usize, f64)> + '_ {
        self.entries
            .iter()
            .map(|entry| (entry.value_index, entry.coefficient))
    }

    pub(crate) fn raw_entries(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.entries()
    }
}

/// A finite flat constrained zonotope plus an independent box remainder.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstrainedZonotope64 {
    center: Vec<f64>,
    generators: Vec<SparseGenerator64>,
    constraints: Array2<f64>,
    rhs: Vec<f64>,
    box_remainder: Vec<f64>,
}

/// Invalid domain data or arithmetic that could not be enclosed finitely.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConstrainedZonotope64Error {
    /// Parallel one-dimensional inputs disagree in length.
    #[error("shape mismatch for {field}: expected {expected}, got {got}")]
    Shape {
        /// Input whose length is wrong.
        field: &'static str,
        /// Required length.
        expected: usize,
        /// Supplied length.
        got: usize,
    },

    /// A bound or stored coefficient is NaN or infinite.
    #[error("{field}[{index}] must be finite")]
    NonFinite {
        /// Flattened input name.
        field: &'static str,
        /// Logical row-major index.
        index: usize,
    },

    /// A certified lower endpoint exceeds its upper endpoint.
    #[error("lower[{index}] exceeds upper[{index}]")]
    ReversedBounds {
        /// Coordinate with invalid ordering.
        index: usize,
    },

    /// Box remainders must be nonnegative.
    #[error("box_remainder[{index}] must be nonnegative")]
    NegativeRemainder {
        /// Coordinate with a negative radius.
        index: usize,
    },

    /// A sparse entry addresses a coordinate outside the flat value axis.
    #[error(
        "generator {generator} entry {entry} has value index {value_index}, but value dimension is {value_dim}"
    )]
    GeneratorIndexOutOfRange {
        /// Generator column.
        generator: usize,
        /// Position within the sparse column.
        entry: usize,
        /// Invalid value-axis index.
        value_index: usize,
        /// Domain value dimension.
        value_dim: usize,
    },

    /// Sparse entries must have strictly increasing, unique value indices.
    #[error("generator {generator} entry {entry} is not strictly index-ordered")]
    GeneratorOrder {
        /// Generator column.
        generator: usize,
        /// First offending entry.
        entry: usize,
    },

    /// Structural zeros are omitted from the canonical sparse form.
    #[error("generator {generator} entry {entry} has a zero coefficient")]
    ZeroSparseCoefficient {
        /// Generator column.
        generator: usize,
        /// Offending entry.
        entry: usize,
    },

    /// A dimension/resource calculation overflowed `usize`.
    #[error("resource size overflow while computing {operation}")]
    ResourceOverflow {
        /// Calculation that could not be represented.
        operation: &'static str,
    },

    /// A bounded allocation request was rejected by the allocator.
    #[error("unable to reserve storage for {resource}")]
    AllocationFailure {
        /// Requested container.
        resource: &'static str,
    },

    /// A finite enclosure could not be constructed.
    #[error(
        "non-finite or non-adjacent arithmetic at coordinate {index} while computing {operation}"
    )]
    NonFiniteArithmetic {
        /// Coordinate being decomposed.
        index: usize,
        /// Failed construction step.
        operation: &'static str,
    },

    /// Directional evaluation failed closed.
    #[error(transparent)]
    Dual(#[from] ConstrainedZonotopeDualError),
}

impl ConstrainedZonotope64 {
    /// Validate and construct a general sparse constrained zonotope.
    ///
    /// `sparse_generators` is column-major: each outer element is one alpha
    /// symbol and contains increasing `(value_index, coefficient)` entries.
    /// The constraint matrix has shape `(constraint_count, alpha_dim)`.
    pub fn try_new(
        center: Vec<f64>,
        sparse_generators: Vec<Vec<(usize, f64)>>,
        constraints: Array2<f64>,
        rhs: Vec<f64>,
        box_remainder: Vec<f64>,
    ) -> Result<Self, ConstrainedZonotope64Error> {
        let value_dim = center.len();
        let alpha_dim = sparse_generators.len();
        let constraint_count = constraints.nrows();

        if box_remainder.len() != value_dim {
            return Err(ConstrainedZonotope64Error::Shape {
                field: "box_remainder",
                expected: value_dim,
                got: box_remainder.len(),
            });
        }
        if constraints.ncols() != alpha_dim {
            return Err(ConstrainedZonotope64Error::Shape {
                field: "constraints columns",
                expected: alpha_dim,
                got: constraints.ncols(),
            });
        }
        if rhs.len() != constraint_count {
            return Err(ConstrainedZonotope64Error::Shape {
                field: "rhs",
                expected: constraint_count,
                got: rhs.len(),
            });
        }

        checked_resource_product(value_dim, alpha_dim, "value_dim * alpha_dim")?;
        let constraint_elements =
            checked_resource_product(constraint_count, alpha_dim, "constraint_count * alpha_dim")?;
        if constraint_elements != constraints.len() {
            return Err(ConstrainedZonotope64Error::ResourceOverflow {
                operation: "constraint matrix shape",
            });
        }

        validate_finite("center", center.iter().copied())?;
        validate_finite("constraints", constraints.iter().copied())?;
        validate_finite("rhs", rhs.iter().copied())?;
        validate_finite("box_remainder", box_remainder.iter().copied())?;
        for (index, &remainder) in box_remainder.iter().enumerate() {
            if remainder < 0.0 {
                return Err(ConstrainedZonotope64Error::NegativeRemainder { index });
            }
        }

        let mut generators = Vec::new();
        try_reserve(&mut generators, alpha_dim, "sparse generator columns")?;
        let mut total_nnz = 0_usize;
        for (generator_index, entries) in sparse_generators.into_iter().enumerate() {
            let column_start = total_nnz;
            total_nnz = total_nnz.checked_add(entries.len()).ok_or(
                ConstrainedZonotope64Error::ResourceOverflow {
                    operation: "total sparse generator nonzeros",
                },
            )?;
            let mut validated = Vec::new();
            try_reserve(
                &mut validated,
                entries.len(),
                "sparse generator coefficients",
            )?;
            let mut previous_index = None;
            for (entry_index, (value_index, coefficient)) in entries.into_iter().enumerate() {
                if value_index >= value_dim {
                    return Err(ConstrainedZonotope64Error::GeneratorIndexOutOfRange {
                        generator: generator_index,
                        entry: entry_index,
                        value_index,
                        value_dim,
                    });
                }
                if previous_index.is_some_and(|previous| value_index <= previous) {
                    return Err(ConstrainedZonotope64Error::GeneratorOrder {
                        generator: generator_index,
                        entry: entry_index,
                    });
                }
                if !coefficient.is_finite() {
                    return Err(ConstrainedZonotope64Error::NonFinite {
                        field: "generators",
                        index: column_start + entry_index,
                    });
                }
                if coefficient == 0.0 {
                    return Err(ConstrainedZonotope64Error::ZeroSparseCoefficient {
                        generator: generator_index,
                        entry: entry_index,
                    });
                }
                previous_index = Some(value_index);
                validated.push(SparseGeneratorEntry64 {
                    value_index,
                    coefficient,
                });
            }
            generators.push(SparseGenerator64 { entries: validated });
        }

        // Include all persistent scalar containers in one checked accounting
        // sum.  This is not an allocation cap; it prevents wrapped sizing from
        // entering later batching/planning code.
        let _stored_scalars = value_dim
            .checked_add(value_dim)
            .and_then(|count| count.checked_add(total_nnz))
            .and_then(|count| count.checked_add(constraint_elements))
            .and_then(|count| count.checked_add(rhs.len()))
            .ok_or(ConstrainedZonotope64Error::ResourceOverflow {
                operation: "total stored scalar count",
            })?;

        Ok(Self {
            center,
            generators,
            constraints,
            rhs,
            box_remainder,
        })
    }

    /// Construct an axis-aligned domain from caller-certified outer `f64`
    /// lower/upper enclosures and a declared-point decomposition mask.
    ///
    /// Each unmarked coordinate receives exactly one independent alpha symbol.
    /// A marked coordinate receives none; all supplied width is represented by
    /// its independent box remainder.  For either case, midpoint and half-width
    /// rounding deficits are computed over the exact dyadic endpoint values and
    /// rounded upward into the box remainder.
    ///
    /// The caller remains responsible for proving that each supplied `f64`
    /// interval encloses the source specification.  The mask is not trusted to
    /// prove equality and cannot reduce the supplied interval.
    pub fn from_certified_bounds(
        lower: &[f64],
        upper: &[f64],
        declared_point: &[bool],
    ) -> Result<Self, ConstrainedZonotope64Error> {
        let value_dim = lower.len();
        if upper.len() != value_dim {
            return Err(ConstrainedZonotope64Error::Shape {
                field: "upper",
                expected: value_dim,
                got: upper.len(),
            });
        }
        if declared_point.len() != value_dim {
            return Err(ConstrainedZonotope64Error::Shape {
                field: "declared_point",
                expected: value_dim,
                got: declared_point.len(),
            });
        }
        validate_finite("lower", lower.iter().copied())?;
        validate_finite("upper", upper.iter().copied())?;

        let alpha_dim = declared_point.iter().filter(|&&point| !point).count();
        checked_resource_product(value_dim, alpha_dim, "value_dim * alpha_dim")?;

        let mut center = Vec::new();
        let mut generators = Vec::new();
        let mut box_remainder = Vec::new();
        try_reserve(&mut center, value_dim, "centers")?;
        try_reserve(&mut generators, alpha_dim, "axis generator columns")?;
        try_reserve(&mut box_remainder, value_dim, "box remainders")?;

        for index in 0..value_dim {
            let lo = lower[index];
            let hi = upper[index];
            if lo > hi {
                return Err(ConstrainedZonotope64Error::ReversedBounds { index });
            }

            let midpoint = nominal_midpoint(lo, hi);
            let half_width = if declared_point[index] {
                0.0
            } else {
                nominal_half_width(lo, hi)
            };
            if !midpoint.is_finite() || !half_width.is_finite() || half_width < 0.0 {
                return Err(ConstrainedZonotope64Error::NonFiniteArithmetic {
                    index,
                    operation: "nominal midpoint/half-width",
                });
            }
            let remainder = exact_endpoint_deficit(lo, hi, midpoint, half_width, index)?;

            center.push(midpoint);
            box_remainder.push(remainder);
            if !declared_point[index] {
                let entries = if half_width == 0.0 {
                    Vec::new()
                } else {
                    vec![(index, half_width)]
                };
                generators.push(entries);
            }
        }

        let constraints = Array2::from_shape_vec((0, alpha_dim), Vec::new()).map_err(|_| {
            ConstrainedZonotope64Error::ResourceOverflow {
                operation: "empty constraint matrix shape",
            }
        })?;
        Self::try_new(center, generators, constraints, Vec::new(), box_remainder)
    }

    /// Flat value dimension.
    #[must_use]
    pub fn value_dim(&self) -> usize {
        self.center.len()
    }

    /// Number of independent alpha symbols.
    #[must_use]
    pub fn alpha_dim(&self) -> usize {
        self.generators.len()
    }

    /// Number of rows in `C alpha <= d`.
    #[must_use]
    pub fn constraint_count(&self) -> usize {
        self.constraints.nrows()
    }

    /// Exact-dyadic nominal center.
    #[must_use]
    pub fn center(&self) -> &[f64] {
        &self.center
    }

    /// Sparse generator columns, one per alpha symbol.
    #[must_use]
    pub fn generators(&self) -> &[SparseGenerator64] {
        &self.generators
    }

    /// Constraint matrix `C`, shaped `(constraint_count, alpha_dim)`.
    #[must_use]
    pub fn constraints(&self) -> ArrayView2<'_, f64> {
        self.constraints.view()
    }

    /// Constraint right-hand side `d`.
    #[must_use]
    pub fn rhs(&self) -> &[f64] {
        &self.rhs
    }

    /// Independent nonnegative box remainder radii.
    #[must_use]
    pub fn box_remainder(&self) -> &[f64] {
        &self.box_remainder
    }

    /// Evaluate one supplied nonnegative dual candidate with rigorous outward
    /// arithmetic, including the independent box remainder exactly once.
    pub fn evaluate_dual(
        &self,
        direction: &[f64],
        multipliers: &[f64],
    ) -> Result<ConstrainedZonotopeDualBounds, ConstrainedZonotope64Error> {
        Ok(evaluate_constrained_zonotope64_dual(
            self,
            direction,
            multipliers,
        )?)
    }

    pub(crate) fn constraints_ref(&self) -> &Array2<f64> {
        &self.constraints
    }
}

fn validate_finite(
    field: &'static str,
    values: impl IntoIterator<Item = f64>,
) -> Result<(), ConstrainedZonotope64Error> {
    for (index, value) in values.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(ConstrainedZonotope64Error::NonFinite { field, index });
        }
    }
    Ok(())
}

fn checked_resource_product(
    left: usize,
    right: usize,
    operation: &'static str,
) -> Result<usize, ConstrainedZonotope64Error> {
    left.checked_mul(right)
        .ok_or(ConstrainedZonotope64Error::ResourceOverflow { operation })
}

fn try_reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    resource: &'static str,
) -> Result<(), ConstrainedZonotope64Error> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| ConstrainedZonotope64Error::AllocationFailure { resource })
}

/// A finite-overflow-safe nominal midpoint.  Soundness does not depend on the
/// rounding of this operation: exact endpoint deficits are charged below.
fn nominal_midpoint(lower: f64, upper: f64) -> f64 {
    f64::midpoint(lower, upper)
}

/// A finite-overflow-safe nominal half-width.  Same-sign subtraction cannot
/// overflow; opposite-sign endpoints are halved before subtraction.
fn nominal_half_width(lower: f64, upper: f64) -> f64 {
    if lower < 0.0 && upper > 0.0 {
        upper * 0.5 - lower * 0.5
    } else {
        (upper - lower) * 0.5
    }
}

/// Compute `max(0, c - g - lower, upper - c - g)` exactly over the supplied
/// dyadic `f64`s, then return its least adjacent finite upward enclosure.
fn exact_endpoint_deficit(
    lower: f64,
    upper: f64,
    center: f64,
    generator_radius: f64,
    index: usize,
) -> Result<f64, ConstrainedZonotope64Error> {
    let lower = exact_rational(lower, index, "lower endpoint conversion")?;
    let upper = exact_rational(upper, index, "upper endpoint conversion")?;
    let center = exact_rational(center, index, "midpoint conversion")?;
    let radius = exact_rational(generator_radius, index, "half-width conversion")?;

    let lower_deficit = &center - &radius - &lower;
    let upper_deficit = &upper - &center - &radius;
    let mut deficit = BigRational::zero();
    if lower_deficit > deficit {
        deficit = lower_deficit;
    }
    if upper_deficit > deficit {
        deficit = upper_deficit;
    }
    ceil_nonnegative_rational_to_f64(&deficit, index)
}

fn exact_rational(
    value: f64,
    index: usize,
    operation: &'static str,
) -> Result<BigRational, ConstrainedZonotope64Error> {
    BigRational::from_float(value)
        .ok_or(ConstrainedZonotope64Error::NonFiniteArithmetic { index, operation })
}

fn ceil_nonnegative_rational_to_f64(
    value: &BigRational,
    index: usize,
) -> Result<f64, ConstrainedZonotope64Error> {
    if value.is_zero() {
        return Ok(0.0);
    }
    let mut candidate = value
        .to_f64()
        .ok_or(ConstrainedZonotope64Error::NonFiniteArithmetic {
            index,
            operation: "box-remainder rational conversion",
        })?;
    if !candidate.is_finite() || candidate < 0.0 {
        return Err(ConstrainedZonotope64Error::NonFiniteArithmetic {
            index,
            operation: "box-remainder rational conversion",
        });
    }
    if exact_rational(candidate, index, "box-remainder comparison")? < *value {
        candidate = candidate.next_up();
        if !candidate.is_finite()
            || exact_rational(candidate, index, "upward box-remainder comparison")? < *value
        {
            // `ToPrimitive` is expected to return the nearest representable
            // float.  Never loop through an unbounded number of successors if
            // that contract changes: reject instead.
            return Err(ConstrainedZonotope64Error::NonFiniteArithmetic {
                index,
                operation: "adjacent upward box-remainder rounding",
            });
        }
    }
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};
    use num_traits::Signed;
    use proptest::prelude::*;

    use super::*;

    fn rat(value: f64) -> BigRational {
        BigRational::from_float(value).expect("finite test value")
    }

    fn exact_coordinate_extent(
        domain: &ConstrainedZonotope64,
        coordinate: usize,
    ) -> (BigRational, BigRational) {
        let center = rat(domain.center()[coordinate]);
        let mut radius = rat(domain.box_remainder()[coordinate]);
        for generator in domain.generators() {
            for (value_index, coefficient) in generator.entries() {
                if value_index == coordinate {
                    radius += rat(coefficient).abs();
                }
            }
        }
        (center.clone() - radius.clone(), center + radius)
    }

    fn assert_bounds_contained(domain: &ConstrainedZonotope64, lower: &[f64], upper: &[f64]) {
        for index in 0..lower.len() {
            let (domain_lower, domain_upper) = exact_coordinate_extent(domain, index);
            assert!(
                domain_lower <= rat(lower[index]),
                "lower coordinate {index}"
            );
            assert!(
                domain_upper >= rat(upper[index]),
                "upper coordinate {index}"
            );
        }
    }

    fn bounded_normal(raw: u64, exponent: i32) -> f64 {
        assert!((-1_022..=1_023).contains(&exponent));
        let sign = raw & (1_u64 << 63);
        let fraction = raw & ((1_u64 << 52) - 1);
        let biased_exponent = u64::try_from(exponent + 1_023).unwrap();
        f64::from_bits(sign | (biased_exponent << 52) | fraction)
    }

    #[test]
    fn declared_point_mask_only_changes_decomposition() {
        let lower = [1.0_f64.next_down(), -3.0, 7.0];
        let upper = [1.0_f64.next_up(), 5.0, 7.0];
        let all_symbols =
            ConstrainedZonotope64::from_certified_bounds(&lower, &upper, &[false, false, false])
                .unwrap();
        let all_remainders =
            ConstrainedZonotope64::from_certified_bounds(&lower, &upper, &[true, true, true])
                .unwrap();

        assert_eq!(all_symbols.alpha_dim(), 3);
        assert_eq!(all_remainders.alpha_dim(), 0);
        assert_bounds_contained(&all_symbols, &lower, &upper);
        assert_bounds_contained(&all_remainders, &lower, &upper);
        assert!(all_remainders.box_remainder()[0] > 0.0);
        assert!(all_remainders.box_remainder()[1] >= 4.0);
    }

    #[test]
    fn metaroom_shape_prunes_5215_point_symbols_without_losing_width() {
        const VALUE_DIM: usize = 5_376;
        const POINT_DIM: usize = 5_215;
        let mut lower = vec![0.0; VALUE_DIM];
        let mut upper = vec![0.0; VALUE_DIM];
        let mut point = vec![true; VALUE_DIM];
        for index in 0..VALUE_DIM {
            if index < POINT_DIM {
                lower[index] = 1.0_f64.next_down();
                upper[index] = 1.0_f64.next_up();
            } else {
                lower[index] = -0.25;
                upper[index] = 0.75;
                point[index] = false;
            }
        }

        let domain = ConstrainedZonotope64::from_certified_bounds(&lower, &upper, &point).unwrap();
        assert_eq!(domain.value_dim(), VALUE_DIM);
        assert_eq!(domain.alpha_dim(), VALUE_DIM - POINT_DIM);
        assert_eq!(domain.alpha_dim(), 161);
        assert_eq!(domain.constraint_count(), 0);
        assert_eq!(domain.constraints().shape(), &[0, 161]);
        assert_eq!(
            domain
                .generators()
                .iter()
                .map(SparseGenerator64::nnz)
                .sum::<usize>(),
            161
        );
        assert!(domain.box_remainder()[0] > 0.0);
        assert_bounds_contained(&domain, &lower, &upper);
    }

    #[test]
    fn mixed_scale_subnormal_and_extreme_intervals_are_exactly_contained() {
        let min_subnormal = f64::from_bits(1);
        let lower = [
            -f64::MAX,
            min_subnormal,
            -f64::MIN_POSITIVE,
            1.0,
            1.0_f64.next_down(),
        ];
        let upper = [
            f64::MAX,
            f64::from_bits(2),
            f64::MIN_POSITIVE,
            1.0_f64.next_up(),
            1.0_f64.next_up(),
        ];
        let domain = ConstrainedZonotope64::from_certified_bounds(
            &lower,
            &upper,
            &[false, false, true, false, true],
        )
        .unwrap();
        assert_bounds_contained(&domain, &lower, &upper);
        assert_eq!(domain.generators()[0].entries().next(), Some((0, f64::MAX)));
        assert_eq!(domain.box_remainder()[0], 0.0);
        // The exact half-width is 2^-1075, so the nominal generator rounds to
        // zero.  The alpha symbol remains present and the full missing width is
        // charged to the independent remainder.
        assert_eq!(domain.generators()[1].nnz(), 0);
        assert_eq!(domain.box_remainder()[1], min_subnormal);
        assert!(domain.box_remainder()[2] >= f64::MIN_POSITIVE);
        assert!(domain.box_remainder()[4] > 0.0);
    }

    #[test]
    fn malformed_storage_and_bounds_fail_closed() {
        assert!(matches!(
            ConstrainedZonotope64::from_certified_bounds(&[0.0], &[], &[false]),
            Err(ConstrainedZonotope64Error::Shape { field: "upper", .. })
        ));
        assert!(matches!(
            ConstrainedZonotope64::from_certified_bounds(&[0.0], &[0.0], &[]),
            Err(ConstrainedZonotope64Error::Shape {
                field: "declared_point",
                ..
            })
        ));
        for (lower, upper) in [
            (f64::NAN, 0.0),
            (0.0, f64::INFINITY),
            (1.0, 1.0_f64.next_down()),
        ] {
            assert!(
                ConstrainedZonotope64::from_certified_bounds(&[lower], &[upper], &[false]).is_err()
            );
        }

        let no_constraints = Array2::zeros((0, 1));
        assert!(matches!(
            ConstrainedZonotope64::try_new(
                vec![0.0],
                vec![vec![(1, 1.0)]],
                no_constraints.clone(),
                vec![],
                vec![0.0],
            ),
            Err(ConstrainedZonotope64Error::GeneratorIndexOutOfRange { .. })
        ));
        assert!(matches!(
            ConstrainedZonotope64::try_new(
                vec![0.0],
                vec![vec![(0, 0.0)]],
                no_constraints.clone(),
                vec![],
                vec![0.0],
            ),
            Err(ConstrainedZonotope64Error::ZeroSparseCoefficient { .. })
        ));
        assert!(matches!(
            ConstrainedZonotope64::try_new(
                vec![0.0],
                vec![vec![(0, f64::NAN)]],
                no_constraints,
                vec![],
                vec![0.0],
            ),
            Err(ConstrainedZonotope64Error::NonFinite {
                field: "generators",
                ..
            })
        ));
        assert!(matches!(
            ConstrainedZonotope64::try_new(
                vec![0.0, 0.0],
                vec![vec![(1, 1.0), (0, 2.0)]],
                Array2::zeros((0, 1)),
                vec![],
                vec![0.0, 0.0],
            ),
            Err(ConstrainedZonotope64Error::GeneratorOrder { .. })
        ));
        assert!(matches!(
            ConstrainedZonotope64::try_new(
                vec![0.0],
                vec![],
                Array2::zeros((0, 1)),
                vec![],
                vec![0.0],
            ),
            Err(ConstrainedZonotope64Error::Shape {
                field: "constraints columns",
                ..
            })
        ));
        assert!(matches!(
            ConstrainedZonotope64::try_new(
                vec![0.0],
                vec![],
                Array2::zeros((0, 0)),
                vec![],
                vec![-f64::MIN_POSITIVE],
            ),
            Err(ConstrainedZonotope64Error::NegativeRemainder { .. })
        ));
    }

    #[test]
    fn resource_product_overflow_is_rejected() {
        assert!(matches!(
            checked_resource_product(usize::MAX, 2, "test product"),
            Err(ConstrainedZonotope64Error::ResourceOverflow {
                operation: "test product"
            })
        ));
    }

    #[test]
    fn sparse_constraints_and_remainder_reach_outward_dual() {
        let domain = ConstrainedZonotope64::try_new(
            vec![1.5],
            vec![vec![(0, 2.0)], vec![(0, -0.25)]],
            array![[-1.0, 0.0], [0.0, 1.0]],
            vec![0.0, 0.5],
            vec![0.125],
        )
        .unwrap();
        let bounds = domain.evaluate_dual(&[1.0], &[1.0, 0.25]).unwrap();
        let exact_lower = BigRational::new(1.into(), 4.into());
        let exact_upper = BigRational::new(21.into(), 4.into());
        assert!(rat(bounds.lower) <= exact_lower);
        assert!(rat(bounds.upper) >= exact_upper);
        // Distinguish one box charge from an accidental double charge.
        assert!(bounds.lower > 0.125);
        assert!(bounds.upper < 5.375);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        #[test]
        fn arbitrary_mantissa_corners_are_contained_by_exact_rational_domain(
            raw_pairs in prop::collection::vec((any::<u64>(), any::<u64>()), 5),
            point_mask in prop::collection::vec(any::<bool>(), 5),
            exponent in -400i32..=400,
        ) {
            let mut lower = Vec::with_capacity(5);
            let mut upper = Vec::with_capacity(5);
            for (left_raw, right_raw) in raw_pairs {
                let left = bounded_normal(left_raw, exponent);
                let right = bounded_normal(right_raw, exponent);
                lower.push(left.min(right));
                upper.push(left.max(right));
            }
            let domain = ConstrainedZonotope64::from_certified_bounds(
                &lower,
                &upper,
                &point_mask,
            ).unwrap();

            for corner in 0_u32..(1_u32 << 5) {
                for coordinate in 0..5 {
                    let concrete = if corner & (1 << coordinate) == 0 {
                        lower[coordinate]
                    } else {
                        upper[coordinate]
                    };
                    let (domain_lower, domain_upper) =
                        exact_coordinate_extent(&domain, coordinate);
                    prop_assert!(domain_lower <= rat(concrete));
                    prop_assert!(domain_upper >= rat(concrete));
                }
            }
        }
    }
}
