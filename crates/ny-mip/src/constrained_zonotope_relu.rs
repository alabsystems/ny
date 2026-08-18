// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unwired, proof-safe ReLU propagation for [`ConstrainedZonotope64`].
//!
//! The input set is
//!
//! ```text
//! x = c + G alpha + e,  alpha in [-1, 1]^m,  C alpha <= d,
//!                         |e_i| <= r_i.
//! ```
//!
//! Coordinate bounds are computed over the exact dyadic values stored in the
//! domain:
//!
//! ```text
//! l_i = c_i - sum_j |G_ij| - r_i
//! u_i = c_i + sum_j |G_ij| + r_i.
//! ```
//!
//! Ignoring `C alpha <= d` can only widen these bounds.  A stable-active
//! coordinate is copied bit-for-bit; a stable-inactive coordinate becomes
//! exact zero.  For `l < 0 < u`, the exact-rational DeepZ enclosure is
//!
//! ```text
//! a = u / (u - l),  b = -u l / (2 (u - l)),
//! relu(x) in a x + b + b beta,  beta in [-1, 1].
//! ```
//!
//! The nominal center, old generator coefficients, and new one-coordinate
//! generator are rounded to finite `f64`.  Their complete exact-rational
//! rounding discrepancy, together with `a r_i`, is rounded upward once into
//! the output box remainder.  Consequently no nearest-rounded coefficient is
//! trusted as exact real arithmetic.
//!
//! Existing predicate constraints are copied exactly and extended by zero
//! columns for the new beta symbols.  The default transformer deliberately
//! preserves that historical behavior.  The separate opt-in projected-
//! constraint transformer additionally records `y_i >= 0` and `y_i >= x_i`
//! for every unstable coordinate.  Those rows eliminate both the input and
//! output box remainders into their right-hand sides and charge the exact
//! binary64 subtraction residual of every stored coefficient.  They therefore
//! retain the complete witness proof instead of silently dropping `e_i`.
//!
//! The opt-in auxiliary-bound variants intersect the exact unconstrained CZ
//! hull with caller-certified outward bounds before classifying and relaxing a
//! coordinate. Their contract is deliberately about concrete witnesses: the
//! caller proves that every concrete preactivation lies in the auxiliary box,
//! and the result preserves those witnesses. The implementation does **not**
//! assume that every spurious point of the input CZ lies in that box.
//!
//! This module is deliberately **unwired**.  It is not reachable from a
//! verifier verdict or a scored path.  Wiring additionally requires explicit
//! deadline/cancellation polling and a checked aggregate peak-byte budget; the
//! count ceilings below are not a substitute for either gate.

use std::cmp::Ordering;
use std::mem::size_of;

use ndarray::Array2;
use num_bigint::{BigInt, BigUint};
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};

use crate::constrained_zonotope_call_budget::{
    ConstrainedZonotopeCallBudget, ConstrainedZonotopeCallBudgetError, ConstrainedZonotopeCallGate,
    ConstrainedZonotopeCallOutcome, ConstrainedZonotopeCallTracker,
    ConstrainedZonotopePeakLiveBytes, InertConstrainedZonotopeCallGate,
};
use crate::{
    certified_auxiliary_bounds64::CertifiedAuxiliaryBounds64,
    constrained_zonotope64::{
        ConstrainedZonotope64, ConstrainedZonotope64CallGateError, ConstrainedZonotope64Error,
    },
};

/// Absolute implementation ceilings for the unwired exact-rational prototype.
pub const RELU_HARD_MAX_VALUE_DIM: usize = 1_048_576;
/// Absolute ceiling on old plus newly introduced alpha symbols.
pub const RELU_HARD_MAX_OUTPUT_ALPHA_DIM: usize = 65_536;
/// Absolute ceiling on output predicate rows.
pub const RELU_HARD_MAX_CONSTRAINTS: usize = 65_536;
/// Absolute ceiling on the dense output constraint matrix (256 MiB of `f64`).
pub const RELU_HARD_MAX_CONSTRAINT_ELEMENTS: usize = 33_554_432;
/// Absolute ceiling on the conservative output sparse-nonzero plan.
pub const RELU_HARD_MAX_GENERATOR_NNZ: usize = 16_777_216;
/// Absolute ceiling on unstable coordinates/new symbols.
pub const RELU_HARD_MAX_UNSTABLE: usize = 65_536;
/// Absolute ceiling on conservative exact-rational contributor work.
pub const RELU_HARD_MAX_EXACT_TERMS: usize = 40_000_000;

// The exact dyadic radius at one coordinate spans at most the finite binary64
// exponent range plus log2(RELU_HARD_MAX_GENERATOR_NNZ) carry bits. This wide
// logical charge covers its BigUint payload and container.
const RELU_DYADIC_RADIUS_LIVE_BYTES_PER_COORDINATE: usize = 1_024;
// Each retained unstable plan owns several <=~10,000-bit rational/dyadic
// payloads. This deliberately broad charge includes every numerator,
// denominator, container, and transient clone retained across later phases.
const RELU_UNSTABLE_PLAN_LIVE_BYTES: usize = 64 * 1_024;
// One exact coordinate/coefficient is processed at a time. Keep its
// BigInt/BigRational operands, cross-products, normalization scratch, and
// allocator-independent container storage separate from retained plans. This
// remains required when a caller proves `max_unstable == 0` but supplies
// auxiliary bounds whose exact intersection still needs transient arithmetic.
const RELU_EXACT_TRANSIENT_LIVE_BYTES: usize = 64 * 1_024;
// Projected subtraction errors are exact dyadics with the same finite exponent
// span as a coordinate radius.
const RELU_PROJECTED_ERROR_LIVE_BYTES: usize = 1_024;

// Every finite binary64 magnitude is below 2^1024 and is an integer multiple
// of 2^-1074. At most RELU_HARD_MAX_GENERATOR_NNZ coefficients plus one box
// remainder contribute to a coordinate, and that term count is below 2^25.
// Therefore the aligned exact nonnegative sum has fewer than 2^2123 units of
// 2^-1074 and needs at most 2,123 significant bits.
const RELU_MAX_DYADIC_ACCUMULATOR_BITS: usize = 2_123;
const RELU_MAX_LIVE_DYADIC_BIGUINTS: usize = 3;
const _: () = assert!(RELU_HARD_MAX_GENERATOR_NNZ + 1 < (1_usize << 25));
const _: () = assert!(RELU_MAX_DYADIC_ACCUMULATOR_BITS >= 1_024 + 1_074 + 25);

// On the same grid a hull endpoint has at most 2,124 bits and its endpoint
// difference at most 2,125. Every exact geometric denominator divides that
// difference times a power no larger than 2^1075. The exact values themselves
// stay below the hard term-count bound times the largest finite binary64
// magnitude. Consequently even an LCM-scaled Ratio numerator stays below
// 5,000 bits. Coefficient-error alignment spans at most the 2,045-bit
// binary64 exponent range and also stays below 5,000 bits. Use twice that
// derived ceiling. A source-level liveness inventory of the named operands,
// Ratio results, LCM operands, and GCD/reduction scratch has fewer than 48
// simultaneously live integer payloads; both constants deliberately round
// upward.
const RELU_MAX_UNSTABLE_INTEGER_BITS: usize = 10_000;
const RELU_MAX_RETAINED_UNSTABLE_BIGINTS: usize = 5;
const RELU_MAX_LIVE_EXACT_TRANSIENT_BIGINTS: usize = 48;

const fn bigint_payload_bytes(bits: usize) -> usize {
    let limb_bits = usize::BITS as usize;
    bits.div_ceil(limb_bits) * size_of::<usize>()
}

/// Caller-tightenable resource limits for ReLU propagation.
///
/// The public function always checks these limits before allocating output
/// storage.  Values above the absolute implementation ceilings are malformed,
/// not permission to consume more resources.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReluTransformLimits {
    /// Maximum flat value dimension.
    pub max_value_dim: usize,
    /// Maximum old plus newly introduced alpha symbols.
    pub max_output_alpha_dim: usize,
    /// Maximum output predicate rows.
    pub max_constraints: usize,
    /// Maximum elements in the dense output constraint matrix.
    pub max_constraint_elements: usize,
    /// Maximum conservative output generator nonzeros.
    pub max_generator_nnz: usize,
    /// Maximum unstable coordinates/new symbols.
    pub max_unstable: usize,
    /// Maximum conservative exact-rational contributor count.
    pub max_exact_terms: usize,
}

impl Default for ReluTransformLimits {
    fn default() -> Self {
        Self {
            max_value_dim: RELU_HARD_MAX_VALUE_DIM,
            max_output_alpha_dim: RELU_HARD_MAX_OUTPUT_ALPHA_DIM,
            max_constraints: RELU_HARD_MAX_CONSTRAINTS,
            max_constraint_elements: RELU_HARD_MAX_CONSTRAINT_ELEMENTS,
            max_generator_nnz: RELU_HARD_MAX_GENERATOR_NNZ,
            max_unstable: RELU_HARD_MAX_UNSTABLE,
            max_exact_terms: RELU_HARD_MAX_EXACT_TERMS,
        }
    }
}

/// Malformed limits, exhausted resources, or arithmetic that cannot be
/// enclosed in the finite output representation.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ReluTransformError {
    /// A caller limit attempted to exceed an absolute implementation ceiling.
    #[error("invalid {resource} limit {supplied}; hard maximum is {hard_max}")]
    InvalidLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Caller-supplied limit.
        supplied: usize,
        /// Absolute implementation ceiling.
        hard_max: usize,
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
        /// Calculation that could not be represented.
        operation: &'static str,
    },

    /// A bounded allocation request was rejected.
    #[error("unable to reserve storage for {resource}")]
    AllocationFailure {
        /// Requested container.
        resource: &'static str,
    },

    /// Auxiliary bounds do not cover the input value axis.
    #[error("auxiliary bounds have value dimension {got}; expected {expected}")]
    AuxiliaryDimensionMismatch {
        /// Input CZ value dimension.
        expected: usize,
        /// Auxiliary bound dimension.
        got: usize,
    },

    /// An auxiliary interval is disjoint from the exact unconstrained CZ hull.
    #[error(
        "auxiliary bounds have empty intersection with the CZ hull at coordinate {coordinate}"
    )]
    EmptyAuxiliaryIntersection {
        /// Coordinate with a structurally inconsistent intersection.
        coordinate: usize,
    },

    /// An exact value could not be represented by a finite nominal/remainder.
    #[error("non-finite arithmetic at coordinate {coordinate} while computing {operation}")]
    NonFiniteArithmetic {
        /// Output coordinate being transformed.
        coordinate: usize,
        /// Failed construction step.
        operation: &'static str,
    },

    /// The host flushes binary64 subnormals and cannot support this proof path.
    #[error("unsupported floating-point environment: {requirement}")]
    UnsupportedFloatingPoint {
        /// Required IEEE behavior.
        requirement: &'static str,
    },

    /// Final validated-domain construction failed closed.
    #[error(transparent)]
    Domain(#[from] ConstrainedZonotope64Error),
}

/// Primitive or call-firewall refusal from a budgeted ReLU transform.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ReluTransformBudgetError {
    /// Limits, exact arithmetic, or domain construction failed.
    #[error(transparent)]
    Transform(#[from] ReluTransformError),

    /// The caller's deadline or aggregate peak-memory ceiling refused work.
    #[error(transparent)]
    Budget(#[from] ConstrainedZonotopeCallBudgetError),
}

#[derive(Clone, Copy, Debug)]
enum CoordinatePlan {
    Inactive,
    Active,
    Unstable(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CoordinateClass {
    Inactive,
    Active,
    Unstable,
}

#[derive(Clone, Debug)]
struct UnstablePlan {
    scale_numerator: BigUint,
    scale_denominator: BigUint,
    nominal_center: f64,
    nominal_noise: f64,
    exact_error: BigRational,
    coefficient_error_numerator: ExactNonnegativeDyadic,
}

// Three full-size BigUints cover the retained accumulator, a shifted operand,
// and the result/reallocation payload of the largest dyadic update. Container
// headers are included; allocator metadata/capacity rounding remains in the
// caller's documented moat.
const _: () = assert!(
    RELU_DYADIC_RADIUS_LIVE_BYTES_PER_COORDINATE
        >= RELU_MAX_LIVE_DYADIC_BIGUINTS
            * (bigint_payload_bytes(RELU_MAX_DYADIC_ACCUMULATOR_BITS) + size_of::<BigUint>())
            + size_of::<ExactNonnegativeDyadic>()
);
const _: () = assert!(
    RELU_PROJECTED_ERROR_LIVE_BYTES
        >= RELU_MAX_LIVE_DYADIC_BIGUINTS
            * (bigint_payload_bytes(RELU_MAX_DYADIC_ACCUMULATOR_BITS) + size_of::<BigUint>())
            + size_of::<ExactNonnegativeDyadic>()
);

// A retained plan has two scale integers, two exact-error Ratio integers, and
// one coefficient-error dyadic integer. Count each independently.
const _: () = assert!(
    RELU_UNSTABLE_PLAN_LIVE_BYTES
        >= RELU_MAX_RETAINED_UNSTABLE_BIGINTS
            * (bigint_payload_bytes(RELU_MAX_UNSTABLE_INTEGER_BITS) + size_of::<BigInt>())
            + size_of::<UnstablePlan>()
);
// The independent transient charge covers every current exact operand plus
// the largest operator scratch inventory while all retained plans remain live.
const _: () = assert!(
    RELU_EXACT_TRANSIENT_LIVE_BYTES
        >= RELU_MAX_LIVE_EXACT_TRANSIENT_BIGINTS
            * (bigint_payload_bytes(RELU_MAX_UNSTABLE_INTEGER_BITS) + size_of::<BigInt>())
);

#[derive(Clone, Copy, Debug)]
struct ResourcePlan {
    output_alpha_dim: usize,
    output_constraint_count: usize,
    output_constraint_elements: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PredicateMode {
    Preserve,
    ProjectReluGeometry,
}

/// An exact nonnegative dyadic `significand * 2^binary_exponent`.
///
/// IEEE-754 values are dyadics already.  Keeping their nonnegative sum in this
/// form avoids the numerator/denominator GCD normalization that
/// `BigRational::add_assign` performs for every generator coefficient.  The
/// significand is kept odd when nonzero, making equality and comparison cheap
/// even when input exponents span the complete binary64 range.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ExactNonnegativeDyadic {
    significand: BigUint,
    binary_exponent: i32,
}

impl ExactNonnegativeDyadic {
    fn add_abs_finite(
        &mut self,
        value: f64,
        coordinate: usize,
        operation: &'static str,
    ) -> Result<(), ReluTransformError> {
        let (term_significand, term_exponent) =
            abs_finite_dyadic_parts(value, coordinate, operation)?;
        self.add_biguint_dyadic(BigUint::from(term_significand), term_exponent);
        Ok(())
    }

    /// Add the absolute difference between two exact dyadic terms without
    /// reducing an arbitrary rational.  The significands may already contain
    /// a shared non-dyadic scale numerator or denominator.
    fn add_abs_difference(
        &mut self,
        left_significand: BigUint,
        left_exponent: i32,
        right_significand: BigUint,
        right_exponent: i32,
    ) {
        let common_exponent = left_exponent.min(right_exponent);
        let left_shift = usize::try_from(left_exponent - common_exponent)
            .expect("finite dyadic exponent difference fits usize");
        let right_shift = usize::try_from(right_exponent - common_exponent)
            .expect("finite dyadic exponent difference fits usize");
        let left = left_significand << left_shift;
        let right = right_significand << right_shift;
        let difference = if left >= right {
            left - right
        } else {
            right - left
        };
        self.add_biguint_dyadic(difference, common_exponent);
    }

    fn add_biguint_dyadic(&mut self, mut significand: BigUint, mut binary_exponent: i32) {
        let Some(trailing_zeros) = significand.trailing_zeros() else {
            return;
        };
        if trailing_zeros != 0 {
            let shift = usize::try_from(trailing_zeros)
                .expect("bounded exact-term count keeps dyadic shift in usize");
            significand >>= shift;
            binary_exponent += i32::try_from(trailing_zeros)
                .expect("bounded exact-term count keeps exponent in i32");
        }

        if self.significand.is_zero() {
            self.significand = significand;
            self.binary_exponent = binary_exponent;
            return;
        }

        if binary_exponent < self.binary_exponent {
            let shift = usize::try_from(self.binary_exponent - binary_exponent)
                .expect("finite dyadic exponent difference fits usize");
            self.significand <<= shift;
            self.binary_exponent = binary_exponent;
        }
        let shift = usize::try_from(binary_exponent - self.binary_exponent)
            .expect("finite dyadic exponent difference fits usize");
        if shift == 0 {
            self.significand += significand;
        } else {
            self.significand += significand << shift;
        }
        self.normalize();
    }

    /// Compare this radius with the exact absolute value of a finite `f64`.
    fn cmp_abs_finite(
        &self,
        value: f64,
        coordinate: usize,
        operation: &'static str,
    ) -> Result<Ordering, ReluTransformError> {
        let (other_significand, other_exponent) =
            abs_finite_dyadic_parts(value, coordinate, operation)?;
        if self.significand.is_zero() {
            return Ok(0_u64.cmp(&other_significand));
        }
        if other_significand == 0 {
            return Ok(Ordering::Greater);
        }

        Ok(match self.binary_exponent.cmp(&other_exponent) {
            Ordering::Equal => self.significand.cmp(&BigUint::from(other_significand)),
            Ordering::Less => {
                let shift = usize::try_from(other_exponent - self.binary_exponent)
                    .expect("finite binary64 exponent difference fits usize");
                self.significand
                    .cmp(&(BigUint::from(other_significand) << shift))
            }
            Ordering::Greater => {
                let shift = usize::try_from(self.binary_exponent - other_exponent)
                    .expect("finite binary64 exponent difference fits usize");
                (&self.significand << shift).cmp(&BigUint::from(other_significand))
            }
        })
    }

    fn to_big_rational(&self) -> BigRational {
        if self.significand.is_zero() {
            return BigRational::zero();
        }
        let significand = BigInt::from(self.significand.clone());
        if self.binary_exponent >= 0 {
            BigRational::from_integer(
                significand
                    << usize::try_from(self.binary_exponent)
                        .expect("nonnegative i32 exponent fits usize"),
            )
        } else {
            BigRational::new(
                significand,
                BigInt::from(1_u8)
                    << usize::try_from(-self.binary_exponent)
                        .expect("negated finite binary64 exponent fits usize"),
            )
        }
    }

    fn normalize(&mut self) {
        let Some(trailing_zeros) = self.significand.trailing_zeros() else {
            self.binary_exponent = 0;
            return;
        };
        if trailing_zeros == 0 {
            return;
        }
        let shift = usize::try_from(trailing_zeros)
            .expect("bounded exact-term count keeps dyadic shift in usize");
        self.significand >>= shift;
        self.binary_exponent +=
            i32::try_from(trailing_zeros).expect("bounded exact-term count keeps exponent in i32");
    }
}

/// Decompose `abs(value)` exactly and remove all powers of two from its
/// significand.  Finite normal values are `mantissa * 2^(biased_exp - 1075)`;
/// finite subnormals are `fraction * 2^-1074`.
fn abs_finite_dyadic_parts(
    value: f64,
    coordinate: usize,
    operation: &'static str,
) -> Result<(u64, i32), ReluTransformError> {
    const FRACTION_MASK: u64 = (1_u64 << 52) - 1;
    const EXPONENT_MASK: u64 = 0x7ff;

    let magnitude_bits = value.to_bits() & (u64::MAX >> 1);
    let biased_exponent = (magnitude_bits >> 52) & EXPONENT_MASK;
    let fraction = magnitude_bits & FRACTION_MASK;
    if biased_exponent == EXPONENT_MASK {
        return Err(ReluTransformError::NonFiniteArithmetic {
            coordinate,
            operation,
        });
    }
    if biased_exponent == 0 && fraction == 0 {
        return Ok((0, 0));
    }

    let (mut significand, mut binary_exponent) = if biased_exponent == 0 {
        (fraction, -1074)
    } else {
        (
            (1_u64 << 52) | fraction,
            i32::try_from(biased_exponent).expect("binary64 exponent fits i32") - 1075,
        )
    };
    let trailing_zeros = significand.trailing_zeros();
    significand >>= trailing_zeros;
    binary_exponent += i32::try_from(trailing_zeros).expect("u64 trailing zeros fit i32");
    Ok((significand, binary_exponent))
}

/// Round `coefficient * scale_numerator / scale_denominator` to binary64 and
/// charge its complete exact rounding discrepancy to a shared numerator.
///
/// The scale is positive.  Both the input coefficient and the rounded nominal
/// are exact dyadics, so their discrepancy can be accumulated as
///
/// ```text
/// |n s 2^e - d p 2^f| / d
/// ```
///
/// without constructing and reducing a `BigRational` for the error of every
/// coefficient.  The caller divides the accumulated dyadic numerator by `d`
/// once per unstable coordinate.
fn nearest_scaled_dyadic(
    coefficient: f64,
    scale_numerator: &BigUint,
    scale_denominator: &BigUint,
    coefficient_error_numerator: &mut ExactNonnegativeDyadic,
    coordinate: usize,
) -> Result<f64, ReluTransformError> {
    let (coefficient_significand, coefficient_exponent) =
        abs_finite_dyadic_parts(coefficient, coordinate, "input generator coefficient")?;
    if coefficient_significand == 0 {
        // `BigRational::from_float(-0.0)` has no negative-zero representation,
        // so preserve the historical path's positive zero here.
        return Ok(0.0);
    }

    let ideal_significand = scale_numerator * coefficient_significand;
    let mut raw_numerator = BigInt::from(ideal_significand.clone());
    let mut raw_denominator = BigInt::from(scale_denominator.clone());
    if coefficient_exponent >= 0 {
        raw_numerator <<= usize::try_from(coefficient_exponent)
            .expect("nonnegative finite binary64 exponent fits usize");
    } else {
        raw_denominator <<= usize::try_from(-coefficient_exponent)
            .expect("negated finite binary64 exponent fits usize");
    }
    // `new_raw` deliberately skips GCD normalization.  `Ratio::to_f64`
    // performs exact quotient rounding directly, so reduction is unnecessary.
    let ideal_magnitude = BigRational::new_raw(raw_numerator, raw_denominator);
    let nominal_magnitude = nearest_finite(
        &ideal_magnitude,
        coordinate,
        "unstable generator coefficient",
    )?;
    let nominal = if coefficient.is_sign_negative() {
        -nominal_magnitude
    } else {
        nominal_magnitude
    };

    let (nominal_significand, nominal_exponent) =
        abs_finite_dyadic_parts(nominal, coordinate, "unstable generator coefficient")?;
    let nominal_scaled_significand = scale_denominator * nominal_significand;
    coefficient_error_numerator.add_abs_difference(
        ideal_significand,
        coefficient_exponent,
        nominal_scaled_significand,
        nominal_exponent,
    );
    Ok(nominal)
}

fn classify_coordinate_exact(
    center: f64,
    radius: &ExactNonnegativeDyadic,
    coordinate: usize,
) -> Result<CoordinateClass, ReluTransformError> {
    let radius_vs_abs_center = radius.cmp_abs_finite(center, coordinate, "input center")?;

    // At equality, a nonpositive center has upper == 0 and is inactive; a
    // positive center has lower == 0 and is active.  Exact zero follows the
    // old upper-first test and is therefore inactive.
    if center <= 0.0 && radius_vs_abs_center != Ordering::Greater {
        Ok(CoordinateClass::Inactive)
    } else if center > 0.0 && radius_vs_abs_center != Ordering::Greater {
        Ok(CoordinateClass::Active)
    } else {
        Ok(CoordinateClass::Unstable)
    }
}

/// Intersect one exact CZ hull coordinate with caller-certified binary64
/// endpoints. The returned center is retained so an unstable caller does not
/// repeat exact conversion work.
fn intersect_auxiliary_coordinate_exact(
    center_value: f64,
    radius: &ExactNonnegativeDyadic,
    auxiliary: &CertifiedAuxiliaryBounds64,
    coordinate: usize,
) -> Result<(CoordinateClass, BigRational, BigRational, BigRational), ReluTransformError> {
    let center = exact_finite(center_value, coordinate, "input center")?;
    let radius = radius.to_big_rational();
    let cz_lower = &center - &radius;
    let cz_upper = &center + &radius;
    let auxiliary_lower = exact_finite(
        auxiliary.lower()[coordinate],
        coordinate,
        "auxiliary lower bound",
    )?;
    let auxiliary_upper = exact_finite(
        auxiliary.upper()[coordinate],
        coordinate,
        "auxiliary upper bound",
    )?;
    let lower = if auxiliary_lower > cz_lower {
        auxiliary_lower
    } else {
        cz_lower
    };
    let upper = if auxiliary_upper < cz_upper {
        auxiliary_upper
    } else {
        cz_upper
    };
    if lower > upper {
        return Err(ReluTransformError::EmptyAuxiliaryIntersection { coordinate });
    }
    let class = if upper <= BigRational::zero() {
        CoordinateClass::Inactive
    } else if lower >= BigRational::zero() {
        CoordinateClass::Active
    } else {
        CoordinateClass::Unstable
    };
    Ok((class, center, lower, upper))
}

/// Propagate a constrained zonotope through elementwise ReLU.
///
/// This keeps all existing alpha symbols and predicate constraints.  One
/// unconstrained, one-coordinate alpha symbol is appended for each unstable
/// coordinate.  Every operation is subject to both caller-tightenable limits
/// and absolute hard ceilings.
///
/// # Errors
///
/// Fails closed on malformed/exhausted limits, checked-size overflow,
/// allocation failure, or when the exact-rational enclosure cannot be stored
/// with a finite `f64` remainder.
pub fn transform_relu_unwired(
    input: &ConstrainedZonotope64,
    limits: ReluTransformLimits,
) -> Result<ConstrainedZonotope64, ReluTransformError> {
    transform_relu_legacy(input, None, limits, PredicateMode::Preserve)
}

/// Propagate ReLU and append projected `y >= 0` and `y >= x` predicates.
///
/// For an unstable coordinate, the output domain stores
/// `y = c_y + G_y alpha + e_y` while the input stored
/// `x = c_x + G_x alpha + e_x`.  Eliminating
/// `|e_y| <= r_y` and `|e_x| <= r_x` gives the proof-safe rows
///
/// ```text
/// -G_y alpha          <= c_y + r_y,
/// (G_x - G_y) alpha  <= c_y - c_x + r_y + r_x.
/// ```
///
/// `G_y` includes the new DeepZ beta column.  When `G_x - G_y` is rounded to
/// binary64, the exact TwoSum residual is added outward to the second right-
/// hand side.  Thus every concrete ReLU witness retained by the base
/// transformer still satisfies the appended predicates.
///
/// These rows are necessary alpha-space projections of ReLU geometry.  When a
/// domain has nonzero independent box remainder, they do not claim that every
/// arbitrary realization of that abstract error satisfies the pointwise ReLU
/// equations.
///
/// This remains unwired and cannot affect verifier verdicts or scored paths.
///
/// # Errors
///
/// Fails closed under the same conditions as [`transform_relu_unwired`], and
/// additionally when a projected coefficient or right-hand side cannot be
/// represented finitely within the supplied resource limits.
pub fn transform_relu_projected_constraints_unwired(
    input: &ConstrainedZonotope64,
    limits: ReluTransformLimits,
) -> Result<ConstrainedZonotope64, ReluTransformError> {
    transform_relu_legacy(input, None, limits, PredicateMode::ProjectReluGeometry)
}

/// Propagate ReLU using independently certified auxiliary concrete bounds.
///
/// For each coordinate this intersects the exact unconstrained CZ hull with
/// the exact binary64 dyadics in `auxiliary`, then uses the intersection for
/// stable-phase classification and, when unstable, DeepZ parameters.
///
/// The semantic proof obligation is explicit: the caller must have established
/// that **every concrete preactivation witness** lies in `auxiliary`. The
/// returned domain preserves those concrete witnesses. It need not preserve
/// arbitrary spurious CZ points outside the auxiliary interval, and this API
/// does not assert `input CZ subset auxiliary box`.
///
/// This remains default-off and unwired from verifier verdicts.
///
/// # Errors
///
/// In addition to [`transform_relu_unwired`] failures, rejects mismatched
/// auxiliary dimensions and any auxiliary interval disjoint from the exact CZ
/// hull.
pub fn transform_relu_with_auxiliary_bounds_unwired(
    input: &ConstrainedZonotope64,
    auxiliary: &CertifiedAuxiliaryBounds64,
    limits: ReluTransformLimits,
) -> Result<ConstrainedZonotope64, ReluTransformError> {
    transform_relu_legacy(input, Some(auxiliary), limits, PredicateMode::Preserve)
}

/// Propagate ReLU with certified auxiliary bounds and projected ReLU rows.
///
/// This has the same concrete-witness-only contract as
/// [`transform_relu_with_auxiliary_bounds_unwired`]. The projected `y >= 0`
/// and `y >= x` rows are necessary for every retained concrete graph witness;
/// they make no claim about an arbitrary spurious CZ point excluded by the
/// auxiliary bounds.
///
/// This remains default-off and unwired from verifier verdicts.
///
/// # Errors
///
/// Fails closed under the same conditions as
/// [`transform_relu_with_auxiliary_bounds_unwired`] and
/// [`transform_relu_projected_constraints_unwired`].
pub fn transform_relu_projected_constraints_with_auxiliary_bounds_unwired(
    input: &ConstrainedZonotope64,
    auxiliary: &CertifiedAuxiliaryBounds64,
    limits: ReluTransformLimits,
) -> Result<ConstrainedZonotope64, ReluTransformError> {
    transform_relu_legacy(
        input,
        Some(auxiliary),
        limits,
        PredicateMode::ProjectReluGeometry,
    )
}

fn transform_relu_legacy(
    input: &ConstrainedZonotope64,
    auxiliary: Option<&CertifiedAuxiliaryBounds64>,
    limits: ReluTransformLimits,
    predicate_mode: PredicateMode,
) -> Result<ConstrainedZonotope64, ReluTransformError> {
    let mut gate = InertConstrainedZonotopeCallGate;
    match transform_relu_impl(input, auxiliary, limits, predicate_mode, &mut gate) {
        Ok(value) => Ok(value),
        Err(ReluTransformBudgetError::Transform(error)) => Err(error),
        Err(ReluTransformBudgetError::Budget(_)) => {
            unreachable!("the inert ReLU call gate cannot refuse work")
        }
    }
}

/// Propagate ReLU behind a synchronous call-local execution firewall.
///
/// The preflight includes conservative charges for every retained exact
/// dyadic/rational payload. `budget.baseline_live_bytes()` must include the
/// input and all other caller-owned storage sharing the ceiling.
pub fn transform_relu_unwired_with_budget(
    input: &ConstrainedZonotope64,
    limits: ReluTransformLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> Result<ConstrainedZonotopeCallOutcome<ConstrainedZonotope64>, ReluTransformBudgetError> {
    transform_relu_with_budget_impl(input, None, limits, PredicateMode::Preserve, budget)
}

/// Budgeted ReLU with projected `y >= 0` and `y >= x` predicates.
pub fn transform_relu_projected_constraints_unwired_with_budget(
    input: &ConstrainedZonotope64,
    limits: ReluTransformLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> Result<ConstrainedZonotopeCallOutcome<ConstrainedZonotope64>, ReluTransformBudgetError> {
    transform_relu_with_budget_impl(
        input,
        None,
        limits,
        PredicateMode::ProjectReluGeometry,
        budget,
    )
}

/// Budgeted ReLU using caller-certified auxiliary concrete bounds.
pub fn transform_relu_with_auxiliary_bounds_unwired_with_budget(
    input: &ConstrainedZonotope64,
    auxiliary: &CertifiedAuxiliaryBounds64,
    limits: ReluTransformLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> Result<ConstrainedZonotopeCallOutcome<ConstrainedZonotope64>, ReluTransformBudgetError> {
    transform_relu_with_budget_impl(
        input,
        Some(auxiliary),
        limits,
        PredicateMode::Preserve,
        budget,
    )
}

/// Budgeted auxiliary-bound ReLU with projected ReLU predicates.
pub fn transform_relu_projected_constraints_with_auxiliary_bounds_unwired_with_budget(
    input: &ConstrainedZonotope64,
    auxiliary: &CertifiedAuxiliaryBounds64,
    limits: ReluTransformLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> Result<ConstrainedZonotopeCallOutcome<ConstrainedZonotope64>, ReluTransformBudgetError> {
    transform_relu_with_budget_impl(
        input,
        Some(auxiliary),
        limits,
        PredicateMode::ProjectReluGeometry,
        budget,
    )
}

fn transform_relu_with_budget_impl(
    input: &ConstrainedZonotope64,
    auxiliary: Option<&CertifiedAuxiliaryBounds64>,
    limits: ReluTransformLimits,
    predicate_mode: PredicateMode,
    budget: ConstrainedZonotopeCallBudget,
) -> Result<ConstrainedZonotopeCallOutcome<ConstrainedZonotope64>, ReluTransformBudgetError> {
    let mut gate = ConstrainedZonotopeCallTracker::from_system_clock(budget)?;
    let value = transform_relu_impl(input, auxiliary, limits, predicate_mode, &mut gate)?;
    Ok(ConstrainedZonotopeCallOutcome::new(value, gate.report()))
}

#[cfg(test)]
fn transform_relu_with_clock<N>(
    input: &ConstrainedZonotope64,
    auxiliary: Option<&CertifiedAuxiliaryBounds64>,
    limits: ReluTransformLimits,
    predicate_mode: PredicateMode,
    budget: ConstrainedZonotopeCallBudget,
    now: N,
) -> Result<ConstrainedZonotopeCallOutcome<ConstrainedZonotope64>, ReluTransformBudgetError>
where
    N: FnMut(&'static str) -> std::time::Instant,
{
    let mut gate = ConstrainedZonotopeCallTracker::with_clock(budget, now)?;
    let value = transform_relu_impl(input, auxiliary, limits, predicate_mode, &mut gate)?;
    Ok(ConstrainedZonotopeCallOutcome::new(value, gate.report()))
}

fn transform_relu_impl<G>(
    input: &ConstrainedZonotope64,
    auxiliary: Option<&CertifiedAuxiliaryBounds64>,
    limits: ReluTransformLimits,
    predicate_mode: PredicateMode,
    gate: &mut G,
) -> Result<ConstrainedZonotope64, ReluTransformBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    validate_limits(limits)?;
    if gate.is_enforcing() {
        require_relu_gradual_underflow()?;
    }
    gate.checkpoint("ReLU floating-point and limit preflight")?;

    let value_dim = input.value_dim();
    let input_alpha_dim = input.alpha_dim();
    let constraint_count = input.constraint_count();
    if let Some(auxiliary) = auxiliary {
        if auxiliary.value_dim() != value_dim {
            return Err(ReluTransformError::AuxiliaryDimensionMismatch {
                expected: value_dim,
                got: auxiliary.value_dim(),
            }
            .into());
        }
    }
    check_limit("value dimension", value_dim, limits.max_value_dim)?;
    check_limit(
        "input alpha dimension",
        input_alpha_dim,
        limits.max_output_alpha_dim,
    )?;
    check_limit("constraint count", constraint_count, limits.max_constraints)?;

    let mut input_nnz = 0_usize;
    for column in input.generators() {
        gate.charge_items(1, "ReLU input generator geometry")?;
        input_nnz =
            input_nnz
                .checked_add(column.nnz())
                .ok_or(ReluTransformError::ResourceOverflow {
                    operation: "input generator nonzeros",
                })?;
    }
    let mut resources = ResourcePlan::checked(
        value_dim,
        input_alpha_dim,
        input_nnz,
        constraint_count,
        0,
        predicate_mode,
        auxiliary.is_some(),
        limits,
    )?;
    let unstable_capacity_bound = relu_unstable_bound(
        value_dim,
        input_alpha_dim,
        input_nnz,
        constraint_count,
        predicate_mode,
        limits,
    );
    if gate.is_enforcing() {
        gate.preflight_peak_live_bytes(relu_peak_live_bytes(
            input,
            value_dim,
            input_alpha_dim,
            input_nnz,
            constraint_count,
            predicate_mode,
            auxiliary.is_some(),
            limits,
        )?)?;
    }
    gate.checkpoint("ReLU peak-memory preflight complete")?;

    // `r_i + sum_j |G_ij|` is accumulated exactly over the stored dyadics.
    // Do not use BigRational here: normalizing a denominator after every
    // sparse coefficient dominates large convolutional ReLU stages.
    let mut radii = Vec::new();
    gate.checkpoint("ReLU exact-radius allocation")?;
    try_reserve(&mut radii, value_dim, "exact dyadic coordinate radii")?;
    for (coordinate, &remainder) in input.box_remainder().iter().enumerate() {
        gate.charge_items(1, "ReLU exact-radius initialization")?;
        let mut radius = ExactNonnegativeDyadic::default();
        radius.add_abs_finite(remainder, coordinate, "input box remainder")?;
        radii.push(radius);
    }
    for generator in input.generators() {
        gate.charge_items(1, "ReLU radius generator accumulation")?;
        for (coordinate, coefficient) in generator.entries() {
            gate.charge_items(1, "ReLU radius coefficient accumulation")?;
            radii[coordinate].add_abs_finite(
                coefficient,
                coordinate,
                "input generator coefficient",
            )?;
        }
    }
    gate.checkpoint("ReLU exact-radius construction complete")?;

    let mut plans = Vec::new();
    gate.checkpoint("ReLU coordinate-plan allocation")?;
    try_reserve(&mut plans, value_dim, "ReLU coordinate plans")?;
    let mut unstable_plans = Vec::new();
    if gate.is_enforcing() && unstable_capacity_bound != 0 {
        // Budgeted calls reserve their complete preflighted header capacity
        // while the vector is empty. A later amortized growth could otherwise
        // copy more than the hard per-poll item count in one uninterruptible
        // reallocation. Preserve the historical growth path for inert legacy
        // calls so their allocation behavior and error ordering stay intact.
        gate.checkpoint("ReLU unstable-plan capacity allocation")?;
        try_reserve(
            &mut unstable_plans,
            unstable_capacity_bound,
            "unstable ReLU coordinate plans",
        )?;
    }
    let mut unstable_count = 0_usize;
    for coordinate in 0..value_dim {
        gate.charge_items(1, "ReLU coordinate classification")?;
        let center_value = input.center()[coordinate];
        let auxiliary_intersection = match auxiliary {
            Some(auxiliary) => Some(intersect_auxiliary_coordinate_exact(
                center_value,
                &radii[coordinate],
                auxiliary,
                coordinate,
            )?),
            None => None,
        };
        let coordinate_class = match &auxiliary_intersection {
            Some((class, _, _, _)) => *class,
            None => classify_coordinate_exact(center_value, &radii[coordinate], coordinate)?,
        };
        match coordinate_class {
            CoordinateClass::Inactive => plans.push(CoordinatePlan::Inactive),
            CoordinateClass::Active => plans.push(CoordinatePlan::Active),
            CoordinateClass::Unstable => {
                unstable_count =
                    unstable_count
                        .checked_add(1)
                        .ok_or(ReluTransformError::ResourceOverflow {
                            operation: "unstable coordinate count",
                        })?;
                // Reject the fully planned output before allocating or performing
                // exact-rational parameter work for this unstable coordinate.
                resources = ResourcePlan::checked(
                    value_dim,
                    input_alpha_dim,
                    input_nnz,
                    constraint_count,
                    unstable_count,
                    predicate_mode,
                    auxiliary.is_some(),
                    limits,
                )?;
                debug_assert!(unstable_count <= unstable_capacity_bound);

                // DeepZ parameters are needed only for unstable coordinates.
                // Without auxiliary bounds, materialize the exact CZ hull only
                // here. The auxiliary path already materialized and intersected
                // its exact bounds during classification.
                let (center, lower, upper) = match auxiliary_intersection {
                    Some((CoordinateClass::Unstable, center, lower, upper)) => {
                        (center, lower, upper)
                    }
                    Some(_) => unreachable!("coordinate class came from the same intersection"),
                    None => {
                        let center = exact_finite(center_value, coordinate, "input center")?;
                        let radius = radii[coordinate].to_big_rational();
                        let lower = &center - &radius;
                        let upper = &center + &radius;
                        (center, lower, upper)
                    }
                };
                let denominator = &upper - &lower;
                // Here lower < 0 < upper, so denominator, slope, and intercept are
                // strictly positive in exact arithmetic.
                let slope = &upper / &denominator;
                let intercept =
                    -(&upper * &lower) / (&denominator * BigRational::from_integer(2.into()));
                let ideal_center = &slope * &center + &intercept;
                let nominal_center =
                    nearest_finite(&ideal_center, coordinate, "unstable nominal center")?;
                let nominal_noise =
                    nearest_finite(&intercept, coordinate, "unstable nominal noise")?;

                let mut exact_error = (&ideal_center
                    - exact_finite(nominal_center, coordinate, "unstable nominal center")?)
                .abs();
                exact_error += (&intercept
                    - exact_finite(nominal_noise, coordinate, "unstable nominal noise")?)
                .abs();
                exact_error += &slope
                    * exact_finite(
                        input.box_remainder()[coordinate],
                        coordinate,
                        "input box remainder",
                    )?;
                let scale_numerator = slope.numer().magnitude().clone();
                let scale_denominator = slope.denom().magnitude().clone();
                debug_assert!(!scale_numerator.is_zero());
                debug_assert!(!scale_denominator.is_zero());
                let unstable_index = unstable_plans.len();
                if !gate.is_enforcing() {
                    gate.checkpoint("ReLU unstable-plan allocation")?;
                    try_reserve_amortized(
                        &mut unstable_plans,
                        1,
                        "unstable ReLU coordinate plans",
                    )?;
                }
                debug_assert!(!gate.is_enforcing() || unstable_index < unstable_capacity_bound);
                // All exact operands have a hard bit-size bound, but still
                // poll after the nontrivial rational phase and before making
                // its retained plan visible to later phases.
                gate.checkpoint("ReLU unstable-plan publication")?;
                unstable_plans.push(UnstablePlan {
                    scale_numerator,
                    scale_denominator,
                    nominal_center,
                    nominal_noise,
                    exact_error,
                    coefficient_error_numerator: ExactNonnegativeDyadic::default(),
                });
                plans.push(CoordinatePlan::Unstable(unstable_index));
            }
        }
    }
    gate.checkpoint("ReLU coordinate classification complete")?;
    drop(radii);

    let output_alpha_dim = resources.output_alpha_dim;
    let output_constraint_count = resources.output_constraint_count;
    let output_constraint_elements = resources.output_constraint_elements;

    let mut output_generators = Vec::new();
    gate.checkpoint("ReLU output generator-column allocation")?;
    try_reserve(
        &mut output_generators,
        output_alpha_dim,
        "output generator columns",
    )?;
    for generator in input.generators() {
        gate.charge_items(1, "ReLU old generator transform")?;
        let mut entries = Vec::new();
        gate.checkpoint("ReLU output generator-entry allocation")?;
        try_reserve(&mut entries, generator.nnz(), "output generator entries")?;
        for (coordinate, coefficient) in generator.entries() {
            gate.charge_items(1, "ReLU old generator-entry transform")?;
            match plans[coordinate] {
                CoordinatePlan::Inactive => {}
                CoordinatePlan::Active => entries.push((coordinate, coefficient)),
                CoordinatePlan::Unstable(unstable_index) => {
                    let unstable = &mut unstable_plans[unstable_index];
                    let nominal = nearest_scaled_dyadic(
                        coefficient,
                        &unstable.scale_numerator,
                        &unstable.scale_denominator,
                        &mut unstable.coefficient_error_numerator,
                        coordinate,
                    )?;
                    if nominal != 0.0 {
                        entries.push((coordinate, nominal));
                    }
                }
            }
        }
        output_generators.push(entries);
    }
    gate.checkpoint("ReLU old generator transform complete")?;

    // All coefficient discrepancies at an unstable coordinate share the
    // slope denominator.  Convert and divide their exact dyadic numerator
    // once, rather than normalizing one arbitrary rational per coefficient.
    for unstable in &mut unstable_plans {
        gate.charge_items(1, "ReLU coefficient-error materialization")?;
        if !unstable.coefficient_error_numerator.significand.is_zero() {
            let numerator = unstable.coefficient_error_numerator.to_big_rational();
            let denominator =
                BigRational::from_integer(BigInt::from(unstable.scale_denominator.clone()));
            unstable.exact_error += numerator / denominator;
        }
    }

    // New beta columns are appended in increasing coordinate order.  An exact
    // positive intercept may round to zero; the empty alpha column is retained
    // and the complete missing coefficient is already charged above.
    for (coordinate, plan) in plans.iter().enumerate() {
        gate.charge_items(1, "ReLU beta generator construction")?;
        if let CoordinatePlan::Unstable(unstable_index) = plan {
            let nominal_noise = unstable_plans[*unstable_index].nominal_noise;
            let mut entries = Vec::new();
            if nominal_noise != 0.0 {
                gate.checkpoint("ReLU beta generator-entry allocation")?;
                try_reserve(&mut entries, 1, "new ReLU generator entry")?;
                entries.push((coordinate, nominal_noise));
            }
            output_generators.push(entries);
        }
    }
    gate.checkpoint("ReLU beta generator construction complete")?;

    let mut output_center = Vec::new();
    let mut output_remainder = Vec::new();
    gate.checkpoint("ReLU output-center allocation")?;
    try_reserve(&mut output_center, value_dim, "output centers")?;
    gate.checkpoint("ReLU output-remainder allocation")?;
    try_reserve(&mut output_remainder, value_dim, "output remainders")?;
    for (coordinate, plan) in plans.iter().enumerate() {
        gate.charge_items(1, "ReLU output-coordinate materialization")?;
        match plan {
            CoordinatePlan::Inactive => {
                output_center.push(0.0);
                output_remainder.push(0.0);
            }
            CoordinatePlan::Active => {
                output_center.push(input.center()[coordinate]);
                output_remainder.push(input.box_remainder()[coordinate]);
            }
            CoordinatePlan::Unstable(unstable_index) => {
                let unstable = &unstable_plans[*unstable_index];
                output_center.push(unstable.nominal_center);
                output_remainder.push(ceil_finite_nonnegative(
                    &unstable.exact_error,
                    coordinate,
                    "unstable box remainder",
                )?);
            }
        }
    }
    gate.checkpoint("ReLU output-coordinate materialization complete")?;

    let mut constraint_values = Vec::new();
    gate.checkpoint("ReLU constraint-matrix allocation")?;
    try_reserve(
        &mut constraint_values,
        output_constraint_elements,
        "output constraint matrix",
    )?;
    for row in 0..constraint_count {
        gate.charge_items(1, "ReLU retained constraint-row clone")?;
        for column in 0..input_alpha_dim {
            gate.charge_items(1, "ReLU retained constraint-element clone")?;
            constraint_values.push(input.constraints()[[row, column]]);
        }
        append_zeros_with_gate(
            &mut constraint_values,
            unstable_count,
            gate,
            "ReLU retained constraint zero-extension",
        )?;
    }
    let projected_constraint_elements = output_constraint_elements
        .checked_sub(constraint_values.len())
        .ok_or(ReluTransformError::ResourceOverflow {
            operation: "projected ReLU constraint initialization",
        })?;
    append_zeros_with_gate(
        &mut constraint_values,
        projected_constraint_elements,
        gate,
        "ReLU projected constraint initialization",
    )?;
    debug_assert_eq!(constraint_values.len(), output_constraint_elements);

    // `y >= 0` and `y >= x` are projected onto the alpha symbols by
    // eliminating the independent output and input box errors.  Negating a
    // stored binary64 coefficient is exact.  The only new coefficient
    // rounding occurs where an old-alpha input and output coefficient overlap
    // in `G_x - G_y`; Knuth TwoSum returns that exact residual as another
    // binary64 dyadic, which is accumulated without rounding for the RHS.
    let mut projected_subtraction_errors = Vec::new();
    if predicate_mode == PredicateMode::ProjectReluGeometry {
        gate.checkpoint("ReLU projected-error allocation")?;
        try_reserve(
            &mut projected_subtraction_errors,
            unstable_count,
            "projected ReLU subtraction errors",
        )?;
        for _ in 0..unstable_count {
            gate.charge_items(1, "ReLU projected-error initialization")?;
            projected_subtraction_errors.push(ExactNonnegativeDyadic::default());
        }

        for (column, generator) in input.generators().iter().enumerate() {
            gate.charge_items(1, "ReLU projected input generator walk")?;
            for (coordinate, coefficient) in generator.entries() {
                gate.charge_items(1, "ReLU projected input generator-entry walk")?;
                if let CoordinatePlan::Unstable(unstable_index) = plans[coordinate] {
                    let dominance_row = constraint_count + 2 * unstable_index + 1;
                    constraint_values[dominance_row * output_alpha_dim + column] = coefficient;
                }
            }
        }

        for (column, generator) in output_generators.iter().enumerate() {
            gate.charge_items(1, "ReLU projected output generator walk")?;
            for (coordinate, coefficient) in generator {
                gate.charge_items(1, "ReLU projected output generator-entry walk")?;
                let CoordinatePlan::Unstable(unstable_index) = plans[*coordinate] else {
                    continue;
                };
                let zero_row = constraint_count + 2 * unstable_index;
                let dominance_row = zero_row + 1;
                constraint_values[zero_row * output_alpha_dim + column] = -*coefficient;

                let dominance_index = dominance_row * output_alpha_dim + column;
                if column < input_alpha_dim {
                    let input_coefficient = constraint_values[dominance_index];
                    debug_assert_ne!(input_coefficient, 0.0);
                    let (difference, residual) =
                        exact_subtract_with_residual(input_coefficient, *coefficient, *coordinate)?;
                    constraint_values[dominance_index] = difference;
                    projected_subtraction_errors[unstable_index].add_abs_finite(
                        residual,
                        *coordinate,
                        "projected constraint subtraction residual",
                    )?;
                } else {
                    debug_assert_eq!(column, input_alpha_dim + unstable_index);
                    constraint_values[dominance_index] = -*coefficient;
                }
            }
        }
    }
    gate.checkpoint("ReLU projected constraint construction complete")?;

    gate.checkpoint("ReLU constraint-matrix shape materialization")?;
    let constraints = Array2::from_shape_vec(
        (output_constraint_count, output_alpha_dim),
        constraint_values,
    )
    .map_err(|_| ReluTransformError::ResourceOverflow {
        operation: "output constraint matrix shape",
    })?;

    let mut rhs = Vec::new();
    gate.checkpoint("ReLU constraint-right-hand-side allocation")?;
    try_reserve(&mut rhs, output_constraint_count, "output constraint rhs")?;
    for &value in input.rhs() {
        gate.charge_items(1, "ReLU retained constraint-right-hand-side clone")?;
        rhs.push(value);
    }
    if predicate_mode == PredicateMode::ProjectReluGeometry {
        for (coordinate, plan) in plans.iter().enumerate() {
            gate.charge_items(1, "ReLU projected constraint-right-hand-side construction")?;
            let CoordinatePlan::Unstable(unstable_index) = plan else {
                continue;
            };

            let output_center_exact = exact_finite(
                output_center[coordinate],
                coordinate,
                "projected constraint output center",
            )?;
            let output_remainder_exact = exact_finite(
                output_remainder[coordinate],
                coordinate,
                "projected constraint output remainder",
            )?;
            let zero_rhs = &output_center_exact + &output_remainder_exact;
            rhs.push(ceil_finite(
                &zero_rhs,
                coordinate,
                "projected nonnegative constraint rhs",
            )?);

            let input_center_exact = exact_finite(
                input.center()[coordinate],
                coordinate,
                "projected constraint input center",
            )?;
            let input_remainder_exact = exact_finite(
                input.box_remainder()[coordinate],
                coordinate,
                "projected constraint input remainder",
            )?;
            let dominance_rhs = output_center_exact - input_center_exact
                + output_remainder_exact
                + input_remainder_exact
                + projected_subtraction_errors[*unstable_index].to_big_rational();
            rhs.push(ceil_finite(
                &dominance_rhs,
                coordinate,
                "projected dominance constraint rhs",
            )?);
        }
    }
    gate.checkpoint("ReLU constraint-right-hand-side construction complete")?;

    gate.checkpoint("ReLU domain materialization")?;
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
            ReluTransformBudgetError::Transform(ReluTransformError::Domain(error))
        }
        ConstrainedZonotope64CallGateError::Budget(error) => {
            ReluTransformBudgetError::Budget(error)
        }
    })?;
    gate.checkpoint("ReLU domain materialization complete")?;
    gate.checkpoint("ReLU publication")?;
    Ok(output)
}

/// Conservative transform-owned logical peak for every output that can pass
/// the caller's structural limits. Retained exact-arithmetic payloads receive
/// fixed wide charges derived from the hard binary64/exact-term ceilings.
fn relu_peak_live_bytes(
    input: &ConstrainedZonotope64,
    value_dim: usize,
    input_alpha_dim: usize,
    input_nnz: usize,
    input_constraint_count: usize,
    predicate_mode: PredicateMode,
    has_auxiliary_bounds: bool,
    limits: ReluTransformLimits,
) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    let unstable_bound = relu_unstable_bound(
        value_dim,
        input_alpha_dim,
        input_nnz,
        input_constraint_count,
        predicate_mode,
        limits,
    );
    let output_alpha_bound = input_alpha_dim.checked_add(unstable_bound).ok_or(
        ConstrainedZonotopeCallBudgetError::ResourceOverflow {
            operation: "ReLU peak output alpha dimension",
        },
    )?;
    let projected_rows = match predicate_mode {
        PredicateMode::Preserve => 0,
        PredicateMode::ProjectReluGeometry => unstable_bound.checked_mul(2).ok_or(
            ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "ReLU peak projected row count",
            },
        )?,
    };
    let output_constraint_count = input_constraint_count
        .checked_add(projected_rows)
        .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
            operation: "ReLU peak output constraint count",
        })?
        .min(limits.max_constraints);
    let output_constraint_elements = output_constraint_count
        .checked_mul(output_alpha_bound)
        .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
            operation: "ReLU peak constraint elements",
        })?
        .min(limits.max_constraint_elements);
    // Every old candidate column reserves its complete input `nnz`, including
    // coefficients later removed by inactive/zero classification. Each beta
    // column reserves at most one more entry. The validated constructor can
    // retain no more than the same total while both representations overlap.
    let output_generator_nonzeros = input_nnz
        .checked_add(unstable_bound)
        .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
            operation: "ReLU peak output generator nonzeros",
        })?
        .min(limits.max_generator_nnz);

    let mut peak = ConstrainedZonotopePeakLiveBytes::new();
    peak.add_elements::<[u8; RELU_DYADIC_RADIUS_LIVE_BYTES_PER_COORDINATE]>(
        value_dim,
        "ReLU exact-radius live bytes",
    )?;
    peak.add_elements::<CoordinatePlan>(value_dim, "ReLU coordinate-plan bytes")?;
    // During classification all radii and prior retained plans are live
    // together. During later exact coefficient/RHS work all retained plans
    // overlap one complete operator scratch inventory, so these are additive
    // charges rather than alternative phase maxima.
    peak.add_elements::<[u8; RELU_UNSTABLE_PLAN_LIVE_BYTES]>(
        unstable_bound,
        "ReLU unstable exact-plan live bytes",
    )?;
    if value_dim != 0 && (has_auxiliary_bounds || unstable_bound != 0) {
        peak.add_bytes(
            RELU_EXACT_TRANSIENT_LIVE_BYTES,
            "ReLU exact-arithmetic transient bytes",
        )?;
    }
    peak.add_elements::<f64>(value_dim, "ReLU output-center bytes")?;
    peak.add_elements::<f64>(value_dim, "ReLU output-remainder bytes")?;

    let doubled_alpha_headers = output_alpha_bound.checked_mul(2).ok_or(
        ConstrainedZonotopeCallBudgetError::ResourceOverflow {
            operation: "ReLU doubled generator-column headers",
        },
    )?;
    peak.add_elements::<Vec<(usize, f64)>>(
        doubled_alpha_headers,
        "ReLU candidate and validated generator-column bytes",
    )?;
    let doubled_generator_nonzeros = output_generator_nonzeros.checked_mul(2).ok_or(
        ConstrainedZonotopeCallBudgetError::ResourceOverflow {
            operation: "ReLU doubled generator nonzeros",
        },
    )?;
    peak.add_elements::<(usize, f64)>(
        doubled_generator_nonzeros,
        "ReLU candidate and validated generator-entry bytes",
    )?;
    peak.add_elements::<f64>(
        output_constraint_elements,
        "ReLU output constraint-matrix bytes",
    )?;
    peak.add_elements::<f64>(
        output_constraint_count,
        "ReLU output constraint-right-hand-side bytes",
    )?;
    if predicate_mode == PredicateMode::ProjectReluGeometry {
        peak.add_elements::<[u8; RELU_PROJECTED_ERROR_LIVE_BYTES]>(
            unstable_bound,
            "ReLU projected exact-error live bytes",
        )?;
    }

    // `input` is borrowed and therefore belongs in the caller baseline. Keep
    // this assertion local to catch accidental accounting drift.
    debug_assert_eq!(input.value_dim(), value_dim);
    Ok(peak.finish())
}

fn relu_unstable_bound(
    value_dim: usize,
    input_alpha_dim: usize,
    input_nnz: usize,
    input_constraint_count: usize,
    predicate_mode: PredicateMode,
    limits: ReluTransformLimits,
) -> usize {
    let mut unstable_bound = value_dim
        .min(limits.max_unstable)
        .min(limits.max_output_alpha_dim.saturating_sub(input_alpha_dim))
        .min(limits.max_generator_nnz.saturating_sub(input_nnz));
    if predicate_mode == PredicateMode::ProjectReluGeometry {
        unstable_bound = unstable_bound.min(
            limits
                .max_constraints
                .saturating_sub(input_constraint_count)
                / 2,
        );
    }
    unstable_bound
}

/// Reject FTZ/DAZ before a budgeted proof path converts or rounds subnormals.
fn require_relu_gradual_underflow() -> Result<(), ReluTransformError> {
    let half = std::hint::black_box(0.5_f64);
    let min_normal = std::hint::black_box(f64::MIN_POSITIVE);
    let min_subnormal = std::hint::black_box(f64::from_bits(1));
    let two_subnormals = std::hint::black_box(f64::from_bits(2));
    if std::hint::black_box(min_normal * half).to_bits() != 0x0008_0000_0000_0000
        || std::hint::black_box(two_subnormals * half).to_bits() != 1
        || std::hint::black_box(min_subnormal + min_subnormal).to_bits() != 2
    {
        return Err(ReluTransformError::UnsupportedFloatingPoint {
            requirement: "IEEE-754 binary64 gradual underflow (FTZ/DAZ disabled)",
        });
    }
    Ok(())
}

fn validate_limits(limits: ReluTransformLimits) -> Result<(), ReluTransformError> {
    for (resource, supplied, hard_max) in [
        (
            "value dimension",
            limits.max_value_dim,
            RELU_HARD_MAX_VALUE_DIM,
        ),
        (
            "output alpha dimension",
            limits.max_output_alpha_dim,
            RELU_HARD_MAX_OUTPUT_ALPHA_DIM,
        ),
        (
            "constraint count",
            limits.max_constraints,
            RELU_HARD_MAX_CONSTRAINTS,
        ),
        (
            "constraint elements",
            limits.max_constraint_elements,
            RELU_HARD_MAX_CONSTRAINT_ELEMENTS,
        ),
        (
            "generator nonzeros",
            limits.max_generator_nnz,
            RELU_HARD_MAX_GENERATOR_NNZ,
        ),
        (
            "unstable coordinates",
            limits.max_unstable,
            RELU_HARD_MAX_UNSTABLE,
        ),
        (
            "exact-rational terms",
            limits.max_exact_terms,
            RELU_HARD_MAX_EXACT_TERMS,
        ),
    ] {
        if supplied > hard_max {
            return Err(ReluTransformError::InvalidLimit {
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
) -> Result<(), ReluTransformError> {
    if required > limit {
        Err(ReluTransformError::LimitExceeded {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}

impl ResourcePlan {
    fn checked(
        value_dim: usize,
        input_alpha_dim: usize,
        input_nnz: usize,
        input_constraint_count: usize,
        unstable_count: usize,
        predicate_mode: PredicateMode,
        has_auxiliary_bounds: bool,
        limits: ReluTransformLimits,
    ) -> Result<Self, ReluTransformError> {
        check_limit("unstable coordinates", unstable_count, limits.max_unstable)?;
        let output_alpha_dim = input_alpha_dim.checked_add(unstable_count).ok_or(
            ReluTransformError::ResourceOverflow {
                operation: "output alpha dimension",
            },
        )?;
        check_limit(
            "output alpha dimension",
            output_alpha_dim,
            limits.max_output_alpha_dim,
        )?;
        let projected_constraint_count = match predicate_mode {
            PredicateMode::Preserve => 0,
            PredicateMode::ProjectReluGeometry => {
                unstable_count
                    .checked_mul(2)
                    .ok_or(ReluTransformError::ResourceOverflow {
                        operation: "projected ReLU constraint count",
                    })?
            }
        };
        let output_constraint_count = input_constraint_count
            .checked_add(projected_constraint_count)
            .ok_or(ReluTransformError::ResourceOverflow {
                operation: "output constraint count",
            })?;
        check_limit(
            "constraint count",
            output_constraint_count,
            limits.max_constraints,
        )?;
        let output_constraint_elements = output_constraint_count
            .checked_mul(output_alpha_dim)
            .ok_or(ReluTransformError::ResourceOverflow {
                operation: "output constraint elements",
            })?;
        check_limit(
            "constraint elements",
            output_constraint_elements,
            limits.max_constraint_elements,
        )?;
        let planned_output_nnz =
            input_nnz
                .checked_add(unstable_count)
                .ok_or(ReluTransformError::ResourceOverflow {
                    operation: "planned output generator nonzeros",
                })?;
        check_limit(
            "generator nonzeros",
            planned_output_nnz,
            limits.max_generator_nnz,
        )?;
        let projected_exact_terms = match (predicate_mode, unstable_count) {
            (PredicateMode::Preserve, _) | (PredicateMode::ProjectReluGeometry, 0) => 0,
            // At most one exact subtraction residual per old sparse
            // coefficient plus six exact RHS contributors across the two
            // projected rows for each unstable coordinate.
            (PredicateMode::ProjectReluGeometry, _) => input_nnz
                .checked_add(unstable_count.checked_mul(6).ok_or(
                    ReluTransformError::ResourceOverflow {
                        operation: "projected constraint RHS terms",
                    },
                )?)
                .ok_or(ReluTransformError::ResourceOverflow {
                    operation: "projected constraint exact terms",
                })?,
        };
        let auxiliary_exact_terms = if has_auxiliary_bounds {
            // Exact center, radius, lower endpoint, and upper endpoint per
            // coordinate. Arithmetic/comparisons reuse those values.
            value_dim
                .checked_mul(4)
                .ok_or(ReluTransformError::ResourceOverflow {
                    operation: "auxiliary-bound exact terms",
                })?
        } else {
            0
        };
        let exact_terms =
            value_dim
                .checked_add(input_nnz.checked_mul(2).ok_or(
                    ReluTransformError::ResourceOverflow {
                        operation: "coefficient-rounding exact terms",
                    },
                )?)
                .and_then(|count| count.checked_add(unstable_count.checked_mul(4)?))
                .and_then(|count| count.checked_add(projected_exact_terms))
                .and_then(|count| count.checked_add(auxiliary_exact_terms))
                .ok_or(ReluTransformError::ResourceOverflow {
                    operation: "total ReLU exact terms",
                })?;
        check_limit("exact-rational terms", exact_terms, limits.max_exact_terms)?;

        Ok(Self {
            output_alpha_dim,
            output_constraint_count,
            output_constraint_elements,
        })
    }
}

fn try_reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    resource: &'static str,
) -> Result<(), ReluTransformError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| ReluTransformError::AllocationFailure { resource })
}

fn try_reserve_amortized<T>(
    values: &mut Vec<T>,
    additional: usize,
    resource: &'static str,
) -> Result<(), ReluTransformError> {
    values
        .try_reserve(additional)
        .map_err(|_| ReluTransformError::AllocationFailure { resource })
}

fn append_zeros_with_gate<G>(
    values: &mut Vec<f64>,
    mut additional: usize,
    gate: &mut G,
    checkpoint: &'static str,
) -> Result<(), ReluTransformBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    while additional != 0 {
        let chunk = additional.min(crate::CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL);
        let new_len =
            values
                .len()
                .checked_add(chunk)
                .ok_or(ReluTransformError::ResourceOverflow {
                    operation: "ReLU constraint zero initialization",
                })?;
        gate.charge_items(chunk, checkpoint)?;
        values.resize(new_len, 0.0);
        // `charge_items` may poll before the physical bulk initialization.
        // Poll after every hard-sized chunk as well, so a closed deadline is
        // observed before another chunk begins.
        gate.checkpoint(checkpoint)?;
        additional -= chunk;
    }
    Ok(())
}

fn exact_finite(
    value: f64,
    coordinate: usize,
    operation: &'static str,
) -> Result<BigRational, ReluTransformError> {
    BigRational::from_float(value).ok_or(ReluTransformError::NonFiniteArithmetic {
        coordinate,
        operation,
    })
}

fn nearest_finite(
    value: &BigRational,
    coordinate: usize,
    operation: &'static str,
) -> Result<f64, ReluTransformError> {
    let candidate = value
        .to_f64()
        .ok_or(ReluTransformError::NonFiniteArithmetic {
            coordinate,
            operation,
        })?;
    if !candidate.is_finite() {
        return Err(ReluTransformError::NonFiniteArithmetic {
            coordinate,
            operation,
        });
    }
    Ok(candidate)
}

/// Return the rounded difference and a binary64 residual whose exact sum is
/// `left - right`.  Knuth TwoSum is error-free for finite IEEE-754 addition
/// when its intermediates do not overflow.  A rare intermediate-overflow case
/// falls back to an exact-rational residual and accepts it only when that
/// residual is itself exactly representable as binary64.
fn exact_subtract_with_residual(
    left: f64,
    right: f64,
    coordinate: usize,
) -> Result<(f64, f64), ReluTransformError> {
    let negated_right = -right;
    let difference = left + negated_right;
    if !left.is_finite() || !right.is_finite() || !difference.is_finite() {
        return Err(ReluTransformError::NonFiniteArithmetic {
            coordinate,
            operation: "projected constraint coefficient subtraction",
        });
    }

    let right_virtual = difference - left;
    let left_virtual = difference - right_virtual;
    let right_residual = negated_right - right_virtual;
    let left_residual = left - left_virtual;
    let residual = left_residual + right_residual;
    if right_virtual.is_finite()
        && left_virtual.is_finite()
        && right_residual.is_finite()
        && left_residual.is_finite()
        && residual.is_finite()
    {
        return Ok((difference, residual));
    }

    let exact_residual = exact_finite(left, coordinate, "projected subtraction fallback left")?
        - exact_finite(right, coordinate, "projected subtraction fallback right")?
        - exact_finite(
            difference,
            coordinate,
            "projected subtraction fallback difference",
        )?;
    let fallback = nearest_finite(
        &exact_residual,
        coordinate,
        "projected subtraction fallback residual",
    )?;
    if exact_finite(
        fallback,
        coordinate,
        "projected subtraction fallback residual",
    )? != exact_residual
    {
        return Err(ReluTransformError::NonFiniteArithmetic {
            coordinate,
            operation: "inexact projected subtraction fallback residual",
        });
    }
    Ok((difference, fallback))
}

fn ceil_finite(
    value: &BigRational,
    coordinate: usize,
    operation: &'static str,
) -> Result<f64, ReluTransformError> {
    let mut candidate = nearest_finite(value, coordinate, operation)?;
    if exact_finite(candidate, coordinate, operation)? < *value {
        candidate = candidate.next_up();
        if !candidate.is_finite() || exact_finite(candidate, coordinate, operation)? < *value {
            return Err(ReluTransformError::NonFiniteArithmetic {
                coordinate,
                operation,
            });
        }
    }
    Ok(candidate)
}

fn ceil_finite_nonnegative(
    value: &BigRational,
    coordinate: usize,
    operation: &'static str,
) -> Result<f64, ReluTransformError> {
    if value.is_negative() {
        return Err(ReluTransformError::NonFiniteArithmetic {
            coordinate,
            operation,
        });
    }
    if value.is_zero() {
        return Ok(0.0);
    }
    let candidate = ceil_finite(value, coordinate, operation)?;
    if candidate < 0.0 {
        return Err(ReluTransformError::NonFiniteArithmetic {
            coordinate,
            operation,
        });
    }
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::mem::size_of;
    use std::time::{Duration, Instant};

    use ndarray::{array, Array2};
    use proptest::prelude::*;

    use super::*;

    fn exact(value: f64) -> BigRational {
        BigRational::from_float(value).expect("finite test value")
    }

    fn rational(value: i64, denominator: i64) -> BigRational {
        BigRational::new(value.into(), denominator.into())
    }

    fn finite_f64_bits() -> impl Strategy<Value = u64> {
        any::<u64>().prop_filter("finite binary64", |bits| ((bits >> 52) & 0x7ff) != 0x7ff)
    }

    fn dyadic_radius(values: &[f64]) -> ExactNonnegativeDyadic {
        let mut radius = ExactNonnegativeDyadic::default();
        for &value in values {
            radius
                .add_abs_finite(value, 0, "test dyadic accumulation")
                .unwrap();
        }
        radius
    }

    fn budget_input() -> ConstrainedZonotope64 {
        ConstrainedZonotope64::try_new(
            vec![-3.0, 4.0, 0.25],
            vec![
                vec![(0, 0.5), (1, 0.25), (2, 1.0)],
                vec![(1, -0.5), (2, 0.5)],
            ],
            array![[1.0, -2.0], [-0.5, 0.25]],
            vec![0.75, 1.0],
            vec![0.125, 0.25, 0.125],
        )
        .unwrap()
    }

    fn rational_radius(values: &[f64]) -> BigRational {
        values
            .iter()
            .fold(BigRational::zero(), |sum, &value| sum + exact(value).abs())
    }

    fn rational_classification(center: f64, values: &[f64]) -> CoordinateClass {
        let center = exact(center);
        let radius = rational_radius(values);
        let lower = &center - &radius;
        let upper = &center + &radius;
        if upper <= BigRational::zero() {
            CoordinateClass::Inactive
        } else if lower >= BigRational::zero() {
            CoordinateClass::Active
        } else {
            CoordinateClass::Unstable
        }
    }

    fn exact_value(
        domain: &ConstrainedZonotope64,
        alphas: &[BigRational],
        errors: &[BigRational],
        coordinate: usize,
    ) -> BigRational {
        let mut value = exact(domain.center()[coordinate]) + &errors[coordinate];
        for (alpha, generator) in alphas.iter().zip(domain.generators()) {
            for (value_index, coefficient) in generator.entries() {
                if value_index == coordinate {
                    value += alpha * exact(coefficient);
                }
            }
        }
        value
    }

    fn exact_bounds(
        domain: &ConstrainedZonotope64,
        coordinate: usize,
    ) -> (BigRational, BigRational) {
        let center = exact(domain.center()[coordinate]);
        let mut radius = exact(domain.box_remainder()[coordinate]);
        for generator in domain.generators() {
            for (value_index, coefficient) in generator.entries() {
                if value_index == coordinate {
                    radius += exact(coefficient).abs();
                }
            }
        }
        (center.clone() - radius.clone(), center + radius)
    }

    fn constraints_hold(domain: &ConstrainedZonotope64, alphas: &[BigRational]) -> bool {
        (0..domain.constraint_count()).all(|row| {
            let lhs = (0..domain.alpha_dim()).fold(BigRational::zero(), |sum, column| {
                sum + exact(domain.constraints()[[row, column]]) * &alphas[column]
            });
            lhs <= exact(domain.rhs()[row])
        })
    }

    fn assert_relu_witness_included(
        input: &ConstrainedZonotope64,
        output: &ConstrainedZonotope64,
        input_alphas: &[BigRational],
        input_errors: &[BigRational],
    ) {
        assert_eq!(input_alphas.len(), input.alpha_dim());
        assert_eq!(input_errors.len(), input.value_dim());
        assert!(constraints_hold(input, input_alphas));
        for (coordinate, error) in input_errors.iter().enumerate() {
            assert!(error.abs() <= exact(input.box_remainder()[coordinate]));
        }

        let unstable_coordinates: Vec<_> = (0..input.value_dim())
            .filter(|&coordinate| {
                let (lower, upper) = exact_bounds(input, coordinate);
                lower < BigRational::zero() && upper > BigRational::zero()
            })
            .collect();
        assert_eq!(
            output.alpha_dim(),
            input.alpha_dim() + unstable_coordinates.len()
        );

        let mut output_alphas = input_alphas.to_vec();
        for &coordinate in &unstable_coordinates {
            let (lower, upper) = exact_bounds(input, coordinate);
            let denominator = &upper - &lower;
            let slope = &upper / &denominator;
            let intercept =
                -(&upper * &lower) / (&denominator * BigRational::from_integer(2.into()));
            let x = exact_value(input, input_alphas, input_errors, coordinate);
            let y = x.clone().max(BigRational::zero());
            let beta = (y - &slope * x - &intercept) / &intercept;
            assert!(beta >= -BigRational::from_integer(1.into()));
            assert!(beta <= BigRational::from_integer(1.into()));
            output_alphas.push(beta);
        }
        assert!(constraints_hold(output, &output_alphas));

        for coordinate in 0..input.value_dim() {
            let x = exact_value(input, input_alphas, input_errors, coordinate);
            let y = x.max(BigRational::zero());
            let nominal = exact_value(
                output,
                &output_alphas,
                &vec![BigRational::zero(); output.value_dim()],
                coordinate,
            );
            let correction = y - nominal;
            assert!(
                correction.abs() <= exact(output.box_remainder()[coordinate]),
                "coordinate {coordinate}: |{correction}| > {}",
                exact(output.box_remainder()[coordinate])
            );
        }
    }

    fn intersected_test_bounds(
        input: &ConstrainedZonotope64,
        auxiliary: &CertifiedAuxiliaryBounds64,
        coordinate: usize,
    ) -> (BigRational, BigRational) {
        let (cz_lower, cz_upper) = exact_bounds(input, coordinate);
        let auxiliary_lower = exact(auxiliary.lower()[coordinate]);
        let auxiliary_upper = exact(auxiliary.upper()[coordinate]);
        let lower = if auxiliary_lower > cz_lower {
            auxiliary_lower
        } else {
            cz_lower
        };
        let upper = if auxiliary_upper < cz_upper {
            auxiliary_upper
        } else {
            cz_upper
        };
        assert!(lower <= upper);
        (lower, upper)
    }

    /// Replay the existential concrete-witness proof for the auxiliary path.
    /// This intentionally requires only the selected concrete witness to lie
    /// in the certified box; it never assumes the whole CZ lies there.
    fn assert_auxiliary_relu_witness_included(
        input: &ConstrainedZonotope64,
        auxiliary: &CertifiedAuxiliaryBounds64,
        output: &ConstrainedZonotope64,
        input_alphas: &[BigRational],
        input_errors: &[BigRational],
    ) {
        assert_eq!(auxiliary.value_dim(), input.value_dim());
        assert_eq!(input_alphas.len(), input.alpha_dim());
        assert_eq!(input_errors.len(), input.value_dim());
        assert!(constraints_hold(input, input_alphas));
        for (coordinate, error) in input_errors.iter().enumerate() {
            assert!(error.abs() <= exact(input.box_remainder()[coordinate]));
            let concrete = exact_value(input, input_alphas, input_errors, coordinate);
            assert!(concrete >= exact(auxiliary.lower()[coordinate]));
            assert!(concrete <= exact(auxiliary.upper()[coordinate]));
        }

        let unstable_coordinates: Vec<_> = (0..input.value_dim())
            .filter(|&coordinate| {
                let (lower, upper) = intersected_test_bounds(input, auxiliary, coordinate);
                lower < BigRational::zero() && upper > BigRational::zero()
            })
            .collect();
        assert_eq!(
            output.alpha_dim(),
            input.alpha_dim() + unstable_coordinates.len()
        );

        let mut output_alphas = input_alphas.to_vec();
        for &coordinate in &unstable_coordinates {
            let (lower, upper) = intersected_test_bounds(input, auxiliary, coordinate);
            let denominator = &upper - &lower;
            let slope = &upper / &denominator;
            let intercept =
                -(&upper * &lower) / (&denominator * BigRational::from_integer(2.into()));
            let x = exact_value(input, input_alphas, input_errors, coordinate);
            let y = x.clone().max(BigRational::zero());
            let beta = (y - &slope * x - &intercept) / &intercept;
            assert!(beta >= -BigRational::from_integer(1.into()));
            assert!(beta <= BigRational::from_integer(1.into()));
            output_alphas.push(beta);
        }
        assert!(constraints_hold(output, &output_alphas));

        let zero_errors = vec![BigRational::zero(); output.value_dim()];
        for coordinate in 0..input.value_dim() {
            let x = exact_value(input, input_alphas, input_errors, coordinate);
            let y = x.max(BigRational::zero());
            let nominal = exact_value(output, &output_alphas, &zero_errors, coordinate);
            let correction = y - nominal;
            assert!(
                correction.abs() <= exact(output.box_remainder()[coordinate]),
                "coordinate {coordinate}: |{correction}| > {}",
                exact(output.box_remainder()[coordinate])
            );
        }
    }

    #[test]
    fn stable_coordinates_are_exact_and_constraints_are_preserved() {
        let input = budget_input();
        let output = transform_relu_unwired(&input, ReluTransformLimits::default()).unwrap();

        assert_eq!(output.center()[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(output.box_remainder()[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(output.center()[1].to_bits(), input.center()[1].to_bits());
        assert_eq!(
            output.box_remainder()[1].to_bits(),
            input.box_remainder()[1].to_bits()
        );
        assert_eq!(output.alpha_dim(), input.alpha_dim() + 1);
        for row in 0..input.constraint_count() {
            for column in 0..input.alpha_dim() {
                assert_eq!(
                    output.constraints()[[row, column]].to_bits(),
                    input.constraints()[[row, column]].to_bits()
                );
            }
            for column in input.alpha_dim()..output.alpha_dim() {
                assert_eq!(
                    output.constraints()[[row, column]].to_bits(),
                    0.0_f64.to_bits()
                );
            }
        }
        assert_eq!(output.rhs(), input.rhs());

        let stable_positive_coefficients: Vec<_> = output
            .generators()
            .iter()
            .take(input.alpha_dim())
            .flat_map(|generator| generator.entries())
            .filter(|(coordinate, _)| *coordinate == 1)
            .map(|(_, coefficient)| coefficient.to_bits())
            .collect();
        assert_eq!(
            stable_positive_coefficients,
            vec![0.25_f64.to_bits(), (-0.5_f64).to_bits()]
        );
        assert!(output
            .generators()
            .iter()
            .all(|generator| generator.entries().all(|(coordinate, _)| coordinate != 0)));
    }

    #[test]
    fn budgeted_relu_is_bit_identical_and_peak_terms_are_independent() {
        let input = budget_input();
        let limits = ReluTransformLimits::default();
        let legacy = transform_relu_unwired(&input, limits).unwrap();
        let deadline = Instant::now() + Duration::from_mins(1);
        let baseline = 13_usize;
        let outcome = transform_relu_unwired_with_budget(
            &input,
            limits,
            ConstrainedZonotopeCallBudget::new(deadline, baseline, usize::MAX),
        )
        .unwrap();
        assert_eq!(outcome.value(), &legacy);
        assert!(outcome.report().charged_items() > 0);
        assert!(outcome.report().deadline_polls() > 0);

        // Independent expansion of `relu_peak_live_bytes` for this input.
        let value_dim = 3_usize;
        let input_alpha_dim = 2_usize;
        let input_nnz = 5_usize;
        let unstable_bound = 3_usize;
        let output_alpha_bound = input_alpha_dim + unstable_bound;
        let output_constraint_count = 2_usize;
        let output_constraint_elements = output_constraint_count * output_alpha_bound;
        let output_nnz_bound = input_nnz + unstable_bound;
        let transform_peak = value_dim * RELU_DYADIC_RADIUS_LIVE_BYTES_PER_COORDINATE
            + value_dim * size_of::<CoordinatePlan>()
            + unstable_bound * RELU_UNSTABLE_PLAN_LIVE_BYTES
            + RELU_EXACT_TRANSIENT_LIVE_BYTES
            + 2 * value_dim * size_of::<f64>()
            + 2 * output_alpha_bound * size_of::<Vec<(usize, f64)>>()
            + 2 * output_nnz_bound * size_of::<(usize, f64)>()
            + output_constraint_elements * size_of::<f64>()
            + output_constraint_count * size_of::<f64>();
        assert_eq!(
            outcome.report().peak_live_bytes(),
            baseline + transform_peak
        );

        transform_relu_unwired_with_budget(
            &input,
            limits,
            ConstrainedZonotopeCallBudget::new(deadline, baseline, baseline + transform_peak),
        )
        .unwrap();
        assert!(matches!(
            transform_relu_unwired_with_budget(
                &input,
                limits,
                ConstrainedZonotopeCallBudget::new(
                    deadline,
                    baseline,
                    baseline + transform_peak - 1,
                ),
            ),
            Err(ReluTransformBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { .. }
            ))
        ));

        let legacy_projected =
            transform_relu_projected_constraints_unwired(&input, limits).unwrap();
        let budgeted_projected = transform_relu_projected_constraints_unwired_with_budget(
            &input,
            limits,
            ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
        )
        .unwrap();
        assert_eq!(budgeted_projected.value(), &legacy_projected);
    }

    #[test]
    fn every_legacy_api_matches_its_budgeted_value_and_error_order() {
        let input = budget_input();
        let limits = ReluTransformLimits::default();
        let auxiliary =
            CertifiedAuxiliaryBounds64::try_new(vec![-4.0, 2.0, -2.0], vec![-2.0, 6.0, 2.0])
                .unwrap();
        let deadline = Instant::now() + Duration::from_mins(1);
        let budget = || ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX);

        assert_eq!(
            transform_relu_unwired_with_budget(&input, limits, budget())
                .unwrap()
                .into_value(),
            transform_relu_unwired(&input, limits).unwrap(),
        );
        assert_eq!(
            transform_relu_projected_constraints_unwired_with_budget(&input, limits, budget())
                .unwrap()
                .into_value(),
            transform_relu_projected_constraints_unwired(&input, limits).unwrap(),
        );
        assert_eq!(
            transform_relu_with_auxiliary_bounds_unwired_with_budget(
                &input,
                &auxiliary,
                limits,
                budget(),
            )
            .unwrap()
            .into_value(),
            transform_relu_with_auxiliary_bounds_unwired(&input, &auxiliary, limits).unwrap(),
        );
        assert_eq!(
            transform_relu_projected_constraints_with_auxiliary_bounds_unwired_with_budget(
                &input,
                &auxiliary,
                limits,
                budget(),
            )
            .unwrap()
            .into_value(),
            transform_relu_projected_constraints_with_auxiliary_bounds_unwired(
                &input, &auxiliary, limits,
            )
            .unwrap(),
        );

        // Structural exhaustion must retain the exact legacy error payload in
        // both non-auxiliary APIs.
        let exhausted = ReluTransformLimits {
            max_value_dim: input.value_dim() - 1,
            ..limits
        };
        let legacy = transform_relu_unwired(&input, exhausted).unwrap_err();
        assert_eq!(
            transform_relu_unwired_with_budget(&input, exhausted, budget()).unwrap_err(),
            ReluTransformBudgetError::Transform(legacy),
        );
        let legacy = transform_relu_projected_constraints_unwired(&input, exhausted).unwrap_err();
        assert_eq!(
            transform_relu_projected_constraints_unwired_with_budget(&input, exhausted, budget(),)
                .unwrap_err(),
            ReluTransformBudgetError::Transform(legacy),
        );

        // Limit validation historically precedes auxiliary-shape validation.
        // Exercise that ordering and then the shape payload itself for both
        // auxiliary APIs.
        let wrong_shape = CertifiedAuxiliaryBounds64::try_new(vec![], vec![]).unwrap();
        let malformed = ReluTransformLimits {
            max_unstable: RELU_HARD_MAX_UNSTABLE + 1,
            ..limits
        };
        let legacy = transform_relu_with_auxiliary_bounds_unwired(&input, &wrong_shape, malformed)
            .unwrap_err();
        assert!(matches!(legacy, ReluTransformError::InvalidLimit { .. }));
        assert_eq!(
            transform_relu_with_auxiliary_bounds_unwired_with_budget(
                &input,
                &wrong_shape,
                malformed,
                budget(),
            )
            .unwrap_err(),
            ReluTransformBudgetError::Transform(legacy),
        );
        let legacy = transform_relu_projected_constraints_with_auxiliary_bounds_unwired(
            &input,
            &wrong_shape,
            malformed,
        )
        .unwrap_err();
        assert!(matches!(legacy, ReluTransformError::InvalidLimit { .. }));
        assert_eq!(
            transform_relu_projected_constraints_with_auxiliary_bounds_unwired_with_budget(
                &input,
                &wrong_shape,
                malformed,
                budget(),
            )
            .unwrap_err(),
            ReluTransformBudgetError::Transform(legacy),
        );

        let legacy =
            transform_relu_with_auxiliary_bounds_unwired(&input, &wrong_shape, limits).unwrap_err();
        assert_eq!(
            transform_relu_with_auxiliary_bounds_unwired_with_budget(
                &input,
                &wrong_shape,
                limits,
                budget(),
            )
            .unwrap_err(),
            ReluTransformBudgetError::Transform(legacy),
        );
        let legacy = transform_relu_projected_constraints_with_auxiliary_bounds_unwired(
            &input,
            &wrong_shape,
            limits,
        )
        .unwrap_err();
        assert_eq!(
            transform_relu_projected_constraints_with_auxiliary_bounds_unwired_with_budget(
                &input,
                &wrong_shape,
                limits,
                budget(),
            )
            .unwrap_err(),
            ReluTransformBudgetError::Transform(legacy),
        );

        let disjoint =
            CertifiedAuxiliaryBounds64::try_new(vec![0.0, 2.0, -2.0], vec![1.0, 6.0, 2.0]).unwrap();
        let legacy =
            transform_relu_with_auxiliary_bounds_unwired(&input, &disjoint, limits).unwrap_err();
        assert_eq!(
            transform_relu_with_auxiliary_bounds_unwired_with_budget(
                &input,
                &disjoint,
                limits,
                budget(),
            )
            .unwrap_err(),
            ReluTransformBudgetError::Transform(legacy),
        );
        let legacy = transform_relu_projected_constraints_with_auxiliary_bounds_unwired(
            &input, &disjoint, limits,
        )
        .unwrap_err();
        assert_eq!(
            transform_relu_projected_constraints_with_auxiliary_bounds_unwired_with_budget(
                &input,
                &disjoint,
                limits,
                budget(),
            )
            .unwrap_err(),
            ReluTransformBudgetError::Transform(legacy),
        );
    }

    #[test]
    fn stable_auxiliary_exact_work_has_a_fixed_transient_charge() {
        let input = ConstrainedZonotope64::try_new(
            vec![1.0],
            Vec::new(),
            Array2::zeros((0, 0)),
            Vec::new(),
            vec![0.0],
        )
        .unwrap();
        let auxiliary = CertifiedAuxiliaryBounds64::try_new(vec![0.5], vec![1.0]).unwrap();
        let limits = ReluTransformLimits {
            max_unstable: 0,
            ..ReluTransformLimits::default()
        };
        let without_auxiliary =
            relu_peak_live_bytes(&input, 1, 0, 0, 0, PredicateMode::Preserve, false, limits)
                .unwrap();
        let with_auxiliary =
            relu_peak_live_bytes(&input, 1, 0, 0, 0, PredicateMode::Preserve, true, limits)
                .unwrap();
        assert_eq!(
            with_auxiliary,
            without_auxiliary + RELU_EXACT_TRANSIENT_LIVE_BYTES,
        );

        let baseline = 29;
        let deadline = Instant::now() + Duration::from_mins(1);
        let outcome = transform_relu_with_auxiliary_bounds_unwired_with_budget(
            &input,
            &auxiliary,
            limits,
            ConstrainedZonotopeCallBudget::new(deadline, baseline, baseline + with_auxiliary),
        )
        .unwrap();
        assert_eq!(
            outcome.value(),
            &transform_relu_with_auxiliary_bounds_unwired(&input, &auxiliary, limits).unwrap(),
        );
        assert_eq!(
            outcome.report().peak_live_bytes(),
            baseline + with_auxiliary,
        );
        assert!(matches!(
            transform_relu_with_auxiliary_bounds_unwired_with_budget(
                &input,
                &auxiliary,
                limits,
                ConstrainedZonotopeCallBudget::new(
                    deadline,
                    baseline,
                    baseline + with_auxiliary - 1,
                ),
            ),
            Err(ReluTransformBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { .. }
            ))
        ));
    }

    #[test]
    fn candidate_generator_capacity_is_charged_even_when_output_nnz_is_zero() {
        let input = ConstrainedZonotope64::try_new(
            vec![-10.0],
            vec![
                vec![(0, 1.0)],
                vec![(0, 1.0)],
                vec![(0, 1.0)],
                vec![(0, 1.0)],
            ],
            Array2::zeros((0, 4)),
            Vec::new(),
            vec![0.0],
        )
        .unwrap();
        let limits = ReluTransformLimits {
            max_unstable: 0,
            ..ReluTransformLimits::default()
        };
        let transform_peak =
            relu_peak_live_bytes(&input, 1, 4, 4, 0, PredicateMode::Preserve, false, limits)
                .unwrap();
        let independently_expanded = RELU_DYADIC_RADIUS_LIVE_BYTES_PER_COORDINATE
            + size_of::<CoordinatePlan>()
            + 2 * size_of::<f64>()
            + 2 * 4 * size_of::<Vec<(usize, f64)>>()
            + 2 * 4 * size_of::<(usize, f64)>();
        assert_eq!(transform_peak, independently_expanded);

        let deadline = Instant::now() + Duration::from_mins(1);
        let outcome = transform_relu_unwired_with_budget(
            &input,
            limits,
            ConstrainedZonotopeCallBudget::new(deadline, 0, transform_peak),
        )
        .unwrap();
        assert_eq!(
            outcome
                .value()
                .generators()
                .iter()
                .map(|column| column.nnz())
                .sum::<usize>(),
            0,
        );
        assert_eq!(outcome.report().peak_live_bytes(), transform_peak);
        assert!(matches!(
            transform_relu_unwired_with_budget(
                &input,
                limits,
                ConstrainedZonotopeCallBudget::new(deadline, 0, transform_peak - 1),
            ),
            Err(ReluTransformBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { .. }
            ))
        ));
    }

    #[test]
    fn budget_refuses_admission_overflow_and_publication_seams() {
        let input = budget_input();
        let limits = ReluTransformLimits::default();
        let start = Instant::now();
        let reads = Cell::new(0_usize);
        let baseline = transform_relu_with_clock(
            &input,
            None,
            limits,
            PredicateMode::Preserve,
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 5, 4),
            |_| {
                reads.set(reads.get() + 1);
                start
            },
        );
        assert!(matches!(
            baseline,
            Err(ReluTransformBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                    required: 5,
                    limit: 4
                }
            ))
        ));
        assert_eq!(reads.get(), 1);

        assert!(matches!(
            transform_relu_with_clock(
                &input,
                None,
                limits,
                PredicateMode::Preserve,
                ConstrainedZonotopeCallBudget::new(
                    start + Duration::from_secs(1),
                    usize::MAX,
                    usize::MAX,
                ),
                |_| start,
            ),
            Err(ReluTransformBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                    operation: "aggregate peak-live bytes"
                }
            ))
        ));

        for seam in [
            "ReLU unstable-plan publication",
            "ReLU coordinate classification complete",
            "constrained-zonotope generator-column allocation",
            "ReLU publication",
        ] {
            let expired = start + Duration::from_secs(2);
            let result = transform_relu_with_clock(
                &input,
                None,
                limits,
                PredicateMode::Preserve,
                ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
                |checkpoint| if checkpoint == seam { expired } else { start },
            );
            assert!(
                matches!(
                    result,
                    Err(ReluTransformBudgetError::Budget(
                        ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
                    )) if checkpoint == seam
                ),
                "deadline seam {seam} must refuse"
            );
        }
    }

    #[test]
    fn deadline_polls_inside_dense_relu_loops() {
        let dimension = crate::CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL;
        let input = ConstrainedZonotope64::try_new(
            vec![-1.0; dimension],
            Vec::new(),
            Array2::zeros((0, 0)),
            Vec::new(),
            vec![0.0; dimension],
        )
        .unwrap();
        let start = Instant::now();
        let expired = start + Duration::from_secs(2);
        let result = transform_relu_with_clock(
            &input,
            None,
            ReluTransformLimits::default(),
            PredicateMode::Preserve,
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
            |checkpoint| {
                if checkpoint == "ReLU exact-radius initialization" {
                    expired
                } else {
                    start
                }
            },
        );
        assert!(matches!(
            result,
            Err(ReluTransformBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "ReLU exact-radius initialization"
                }
            ))
        ));
    }

    #[test]
    fn physical_zero_initialization_is_interleaved_with_deadline_polls() {
        const ITEMS: usize = crate::CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL;
        for checkpoint in [
            "ReLU retained constraint zero-extension",
            "ReLU projected constraint initialization",
        ] {
            let start = Instant::now();
            let expired = start + Duration::from_secs(2);
            let checkpoint_reads = Cell::new(0_usize);
            let mut gate = ConstrainedZonotopeCallTracker::with_clock(
                ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
                |seen| {
                    if seen == checkpoint {
                        let reads = checkpoint_reads.get() + 1;
                        checkpoint_reads.set(reads);
                        if reads == 2 {
                            return expired;
                        }
                    }
                    start
                },
            )
            .unwrap();
            let mut values = Vec::new();
            values.try_reserve_exact(2 * ITEMS + 1).unwrap();
            let result = append_zeros_with_gate(&mut values, 2 * ITEMS + 1, &mut gate, checkpoint);
            assert!(matches!(
                result,
                Err(ReluTransformBudgetError::Budget(
                    ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                        checkpoint: refused
                    }
                )) if refused == checkpoint
            ));
            assert_eq!(values.len(), ITEMS);
            assert_eq!(checkpoint_reads.get(), 2);
        }
    }

    #[test]
    fn empty_generator_columns_and_zero_width_rows_are_charged() {
        const ITEMS: usize = crate::CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL;
        let start = Instant::now();
        let expired = start + Duration::from_secs(2);

        let empty_columns = ConstrainedZonotope64::try_new(
            vec![-1.0],
            vec![Vec::new(); ITEMS],
            Array2::zeros((0, ITEMS)),
            Vec::new(),
            vec![0.0],
        )
        .unwrap();
        let result = transform_relu_with_clock(
            &empty_columns,
            None,
            ReluTransformLimits::default(),
            PredicateMode::Preserve,
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
            |checkpoint| {
                if checkpoint == "ReLU input generator geometry" {
                    expired
                } else {
                    start
                }
            },
        );
        assert!(matches!(
            result,
            Err(ReluTransformBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "ReLU input generator geometry"
                }
            ))
        ));

        let zero_width_rows = ConstrainedZonotope64::try_new(
            vec![-1.0],
            Vec::new(),
            Array2::zeros((ITEMS, 0)),
            vec![0.0; ITEMS],
            vec![0.0],
        )
        .unwrap();
        let result = transform_relu_with_clock(
            &zero_width_rows,
            None,
            ReluTransformLimits::default(),
            PredicateMode::Preserve,
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
            |checkpoint| {
                if checkpoint == "ReLU retained constraint-row clone" {
                    expired
                } else {
                    start
                }
            },
        );
        assert!(matches!(
            result,
            Err(ReluTransformBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "ReLU retained constraint-row clone"
                }
            ))
        ));
    }

    #[test]
    fn peak_and_plan_arithmetic_and_allocation_fail_closed() {
        let input = budget_input();
        assert!(matches!(
            relu_peak_live_bytes(
                &input,
                input.value_dim(),
                usize::MAX,
                0,
                0,
                PredicateMode::Preserve,
                false,
                ReluTransformLimits::default(),
            ),
            Err(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "ReLU doubled generator-column headers"
            })
        ));

        let unbounded = ReluTransformLimits {
            max_value_dim: usize::MAX,
            max_output_alpha_dim: usize::MAX,
            max_constraints: usize::MAX,
            max_constraint_elements: usize::MAX,
            max_generator_nnz: usize::MAX,
            max_unstable: usize::MAX,
            max_exact_terms: usize::MAX,
        };
        assert!(matches!(
            ResourcePlan::checked(
                1,
                usize::MAX,
                0,
                0,
                1,
                PredicateMode::Preserve,
                false,
                unbounded,
            ),
            Err(ReluTransformError::ResourceOverflow {
                operation: "output alpha dimension"
            })
        ));

        let mut exact = Vec::<u8>::new();
        assert_eq!(
            try_reserve(&mut exact, usize::MAX, "test exact allocation").unwrap_err(),
            ReluTransformError::AllocationFailure {
                resource: "test exact allocation"
            },
        );
        let mut amortized = Vec::<u8>::new();
        assert_eq!(
            try_reserve_amortized(&mut amortized, usize::MAX, "test amortized allocation")
                .unwrap_err(),
            ReluTransformError::AllocationFailure {
                resource: "test amortized allocation"
            },
        );
    }

    #[test]
    fn host_gradual_underflow_probe_accepts_the_proof_environment() {
        require_relu_gradual_underflow().unwrap();
        assert_eq!(
            ReluTransformError::UnsupportedFloatingPoint {
                requirement: "IEEE-754 binary64 gradual underflow (FTZ/DAZ disabled)",
            }
            .to_string(),
            "unsupported floating-point environment: IEEE-754 binary64 gradual underflow (FTZ/DAZ disabled)",
        );
    }

    #[test]
    fn projected_constraints_remove_impossible_deepz_states() {
        // For x = alpha in [-1, 1], DeepZ stores
        // y = 1/4 + (1/2) alpha + (1/4) beta.  The unconstrained abstraction
        // admits states below zero and below x.  The two projected rows remove
        // exactly those states while retaining concrete ReLU witnesses.
        let input = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![vec![(0, 1.0)]],
            Array2::zeros((0, 1)),
            vec![],
            vec![0.0],
        )
        .unwrap();
        let preserved = transform_relu_unwired(&input, ReluTransformLimits::default()).unwrap();
        let projected =
            transform_relu_projected_constraints_unwired(&input, ReluTransformLimits::default())
                .unwrap();

        assert_eq!(preserved.constraint_count(), 0);
        assert_eq!(projected.constraint_count(), 2);
        assert_eq!(projected.alpha_dim(), 2);
        assert_eq!(
            projected.constraints(),
            &array![[-0.5, -0.25], [0.5, -0.25]]
        );
        assert_eq!(projected.rhs(), &[0.25, 0.25]);

        // alpha=1,beta=-1 gives y=1/2 < x=1; alpha=-1,beta=-1 gives
        // y=-1/2 < 0.  Both remain legal in the preserved domain but are
        // excluded by the corresponding projected inequality.
        for spurious in [
            vec![rational(1, 1), rational(-1, 1)],
            vec![rational(-1, 1), rational(-1, 1)],
        ] {
            assert!(constraints_hold(&preserved, &spurious));
            assert!(!constraints_hold(&projected, &spurious));
        }

        // Exact endpoint and kink witnesses remain feasible.
        for concrete in [
            vec![rational(1, 1), rational(1, 1)],
            vec![rational(-1, 1), rational(1, 1)],
            vec![rational(0, 1), rational(-1, 1)],
        ] {
            assert!(constraints_hold(&projected, &concrete));
        }
    }

    #[test]
    fn auxiliary_bounds_stabilize_coordinates_and_reduce_domain_resources() {
        // The predicate rows prove alpha0 in [-1, -1/4] and alpha1 in
        // [1/4, 1]. The unconstrained CZ hull deliberately ignores those rows
        // and sees both coordinates as unstable in [-1, 1].
        let input = ConstrainedZonotope64::try_new(
            vec![0.0, 0.0],
            vec![vec![(0, 1.0)], vec![(1, 1.0)]],
            array![[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            vec![-0.25, 1.0, 1.0, -0.25],
            vec![0.0, 0.0],
        )
        .unwrap();
        let auxiliary =
            CertifiedAuxiliaryBounds64::try_new(vec![-1.0, 0.25], vec![-0.25, 1.0]).unwrap();
        let baseline =
            transform_relu_projected_constraints_unwired(&input, ReluTransformLimits::default())
                .unwrap();
        let stabilized = transform_relu_projected_constraints_with_auxiliary_bounds_unwired(
            &input,
            &auxiliary,
            ReluTransformLimits::default(),
        )
        .unwrap();

        assert_eq!(baseline.alpha_dim(), 4);
        assert_eq!(baseline.constraint_count(), 8);
        assert_eq!(stabilized.alpha_dim(), 2);
        assert_eq!(stabilized.constraint_count(), 4);
        let baseline_nnz: usize = baseline
            .generators()
            .iter()
            .map(|column| column.nnz())
            .sum();
        let stabilized_nnz: usize = stabilized
            .generators()
            .iter()
            .map(|column| column.nnz())
            .sum();
        assert!(stabilized_nnz < baseline_nnz);
        assert_eq!(stabilized.center(), &[0.0, 0.0]);
        assert_eq!(stabilized.box_remainder(), &[0.0, 0.0]);
        assert!(stabilized.generators()[0].entries().next().is_none());
        assert_eq!(
            stabilized.generators()[1].entries().collect::<Vec<_>>(),
            vec![(1, 1.0)]
        );

        for alpha0 in [rational(-1, 1), rational(-1, 4)] {
            for alpha1 in [rational(1, 4), rational(1, 1)] {
                assert_auxiliary_relu_witness_included(
                    &input,
                    &auxiliary,
                    &stabilized,
                    &[alpha0.clone(), alpha1],
                    &[BigRational::zero(), BigRational::zero()],
                );
            }
        }
    }

    #[test]
    fn spurious_cz_points_outside_auxiliary_bounds_are_not_promised() {
        // Imagine an upstream proof establishes that the concrete subset of
        // this coarse CZ has x in [-1, -1/4]. The auxiliary transform may make
        // ReLU exactly zero without preserving the spurious CZ point x=1.
        let input = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![vec![(0, 1.0)]],
            Array2::zeros((0, 1)),
            vec![],
            vec![0.0],
        )
        .unwrap();
        let auxiliary = CertifiedAuxiliaryBounds64::try_new(vec![-1.0], vec![-0.25]).unwrap();
        let output = transform_relu_with_auxiliary_bounds_unwired(
            &input,
            &auxiliary,
            ReluTransformLimits::default(),
        )
        .unwrap();
        assert_eq!(output.alpha_dim(), 1);
        assert_eq!(output.center(), &[0.0]);
        assert_eq!(output.box_remainder(), &[0.0]);
        assert!(output.generators()[0].entries().next().is_none());

        assert_auxiliary_relu_witness_included(
            &input,
            &auxiliary,
            &output,
            &[rational(-1, 2)],
            &[BigRational::zero()],
        );

        let spurious_alpha = [rational(1, 1)];
        let spurious_relu = exact_value(&input, &spurious_alpha, &[BigRational::zero()], 0)
            .max(BigRational::zero());
        let output_at_same_alpha = exact_value(&output, &spurious_alpha, &[BigRational::zero()], 0);
        assert_eq!(spurious_relu, rational(1, 1));
        assert_eq!(output_at_same_alpha, BigRational::zero());
        assert!((spurious_relu - output_at_same_alpha).abs() > exact(output.box_remainder()[0]));
    }

    #[test]
    fn nonrestrictive_auxiliary_bounds_are_bit_identical_to_baseline() {
        let input = ConstrainedZonotope64::try_new(
            vec![0.0, 2.0, -2.0],
            vec![vec![(0, 1.0), (1, 0.25), (2, -0.25)]],
            array![[1.0]],
            vec![0.75],
            vec![0.125, 0.25, 0.25],
        )
        .unwrap();
        let auxiliary =
            CertifiedAuxiliaryBounds64::try_new(vec![-2.0, 1.0, -3.0], vec![2.0, 3.0, -1.0])
                .unwrap();
        assert_eq!(
            transform_relu_with_auxiliary_bounds_unwired(
                &input,
                &auxiliary,
                ReluTransformLimits::default(),
            )
            .unwrap(),
            transform_relu_unwired(&input, ReluTransformLimits::default()).unwrap()
        );
        assert_eq!(
            transform_relu_projected_constraints_with_auxiliary_bounds_unwired(
                &input,
                &auxiliary,
                ReluTransformLimits::default(),
            )
            .unwrap(),
            transform_relu_projected_constraints_unwired(&input, ReluTransformLimits::default(),)
                .unwrap()
        );
    }

    #[test]
    fn auxiliary_nondyadic_slope_uses_exact_rational_rounding_path() {
        let input = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![vec![(0, 0.75)]],
            Array2::zeros((0, 1)),
            vec![],
            vec![0.0625],
        )
        .unwrap();
        let auxiliary = CertifiedAuxiliaryBounds64::try_new(vec![-0.2], vec![0.4]).unwrap();
        let output = transform_relu_projected_constraints_with_auxiliary_bounds_unwired(
            &input,
            &auxiliary,
            ReluTransformLimits::default(),
        )
        .unwrap();

        let lower = exact(-0.2);
        let upper = exact(0.4);
        let denominator = &upper - &lower;
        let slope = &upper / &denominator;
        let intercept = -(&upper * &lower) / (&denominator * BigRational::from_integer(2.into()));
        let expected_center = nearest_finite(&intercept, 0, "test auxiliary center").unwrap();
        let expected_old_coefficient =
            nearest_finite(&(&slope * exact(0.75)), 0, "test auxiliary coefficient").unwrap();
        let expected_noise = nearest_finite(&intercept, 0, "test auxiliary noise").unwrap();
        assert_eq!(output.center()[0].to_bits(), expected_center.to_bits());
        assert_eq!(
            output.generators()[0].entries().next().unwrap().1.to_bits(),
            expected_old_coefficient.to_bits()
        );
        assert_eq!(
            output.generators()[1].entries().next().unwrap().1.to_bits(),
            expected_noise.to_bits()
        );

        for (alpha, error) in [
            (rational(-1, 4), BigRational::zero()),
            (rational(0, 1), rational(-1, 16)),
            (rational(0, 1), rational(1, 16)),
            (rational(1, 2), BigRational::zero()),
        ] {
            assert_auxiliary_relu_witness_included(&input, &auxiliary, &output, &[alpha], &[error]);
        }
    }

    #[test]
    fn auxiliary_shape_intersection_and_exact_work_fail_closed() {
        let input = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![vec![(0, 1.0)]],
            Array2::zeros((0, 1)),
            vec![],
            vec![0.0],
        )
        .unwrap();
        let wrong_shape = CertifiedAuxiliaryBounds64::try_new(vec![], vec![]).unwrap();
        assert!(matches!(
            transform_relu_with_auxiliary_bounds_unwired(
                &input,
                &wrong_shape,
                ReluTransformLimits::default(),
            ),
            Err(ReluTransformError::AuxiliaryDimensionMismatch {
                expected: 1,
                got: 0
            })
        ));

        let disjoint = CertifiedAuxiliaryBounds64::try_new(vec![2.0], vec![3.0]).unwrap();
        assert!(matches!(
            transform_relu_with_auxiliary_bounds_unwired(
                &input,
                &disjoint,
                ReluTransformLimits::default(),
            ),
            Err(ReluTransformError::EmptyAuxiliaryIntersection { coordinate: 0 })
        ));

        let auxiliary = CertifiedAuxiliaryBounds64::try_new(vec![-0.5], vec![0.5]).unwrap();
        let base = ReluTransformLimits::default();
        assert!(matches!(
            transform_relu_with_auxiliary_bounds_unwired(
                &input,
                &auxiliary,
                ReluTransformLimits {
                    max_exact_terms: 10,
                    ..base
                },
            ),
            Err(ReluTransformError::LimitExceeded {
                resource: "exact-rational terms",
                required: 11,
                limit: 10
            })
        ));
        transform_relu_with_auxiliary_bounds_unwired(
            &input,
            &auxiliary,
            ReluTransformLimits {
                max_exact_terms: 11,
                ..base
            },
        )
        .unwrap();
    }

    #[test]
    fn dyadic_accumulator_matches_rational_oracle_at_ieee_extremes() {
        let min_subnormal = f64::from_bits(1);
        let max_subnormal = f64::from_bits((1_u64 << 52) - 1);
        let values = [
            0.0,
            -0.0,
            min_subnormal,
            -min_subnormal,
            max_subnormal,
            -f64::MIN_POSITIVE,
            0.5,
            -1.0,
            f64::from_bits(1.0_f64.to_bits() + 1),
            f64::MAX,
            -f64::MAX,
        ];

        let mut dyadic = ExactNonnegativeDyadic::default();
        let mut oracle = BigRational::zero();
        for value in values {
            dyadic
                .add_abs_finite(value, 0, "extreme test value")
                .unwrap();
            oracle += exact(value).abs();
            assert_eq!(dyadic.to_big_rational(), oracle);
            if !dyadic.significand.is_zero() {
                assert_eq!(dyadic.significand.trailing_zeros(), Some(0));
            }
        }
    }

    fn assert_scaled_coefficients_match_rational_oracle(
        scale_numerator: u64,
        scale_denominator: u64,
        coefficients: &[f64],
    ) {
        assert!(scale_numerator > 0);
        assert!(scale_numerator <= scale_denominator);
        let numerator = BigUint::from(scale_numerator);
        let denominator = BigUint::from(scale_denominator);
        let scale = BigRational::new(
            BigInt::from(numerator.clone()),
            BigInt::from(denominator.clone()),
        );
        let mut accumulated = ExactNonnegativeDyadic::default();
        let mut oracle_error = BigRational::zero();

        for &coefficient in coefficients {
            let nominal =
                nearest_scaled_dyadic(coefficient, &numerator, &denominator, &mut accumulated, 0)
                    .unwrap();
            let ideal = &scale * exact(coefficient);
            let oracle_nominal =
                nearest_finite(&ideal, 0, "scaled coefficient rational oracle").unwrap();
            assert_eq!(
                nominal.to_bits(),
                oracle_nominal.to_bits(),
                "scale={scale_numerator}/{scale_denominator}, coefficient={coefficient:?}"
            );
            oracle_error += (&ideal - exact(oracle_nominal)).abs();
        }

        let accumulated_error =
            accumulated.to_big_rational() / BigRational::from_integer(BigInt::from(denominator));
        assert_eq!(accumulated_error, oracle_error);
    }

    #[test]
    fn scaled_dyadic_rounding_matches_rational_oracle_at_ieee_extremes() {
        let min_subnormal = f64::from_bits(1);
        let max_subnormal = f64::from_bits((1_u64 << 52) - 1);
        let coefficients = [
            0.0,
            -0.0,
            min_subnormal,
            -min_subnormal,
            max_subnormal,
            -max_subnormal,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            0.5,
            -1.0,
            f64::from_bits(1.0_f64.to_bits() + 1),
            f64::MAX.next_down(),
            f64::MAX,
            -f64::MAX,
        ];

        for (numerator, denominator) in [
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 3),
            (9_007_199_254_740_991, 9_007_199_254_740_992),
        ] {
            assert_scaled_coefficients_match_rational_oracle(numerator, denominator, &coefficients);
        }
    }

    #[test]
    fn two_sum_subtraction_handles_ieee_extremes() {
        let min_subnormal = f64::from_bits(1);
        let adjacent_max = f64::MAX.next_down();
        let fallback_left = f64::from_bits(0xffb0_0000_0000_000c);
        let fallback_right = -f64::MAX;
        let mirrored_fallback_left = -fallback_left;
        let mirrored_fallback_right = f64::MAX;
        let cases = [
            (0.0, -0.0),
            (min_subnormal, -min_subnormal),
            (f64::MIN_POSITIVE, min_subnormal),
            (-f64::MIN_POSITIVE, -min_subnormal),
            (f64::MAX, adjacent_max),
            (-f64::MAX, -adjacent_max),
            (f64::MAX, f64::MAX),
            (-f64::MAX, -f64::MAX),
            (fallback_left, fallback_right),
            (mirrored_fallback_left, mirrored_fallback_right),
            (f64::MAX, -f64::MAX),
            (-f64::MAX, f64::MAX),
        ];

        // These two finite differences make Knuth's first recovery
        // intermediate overflow.  They therefore force the exact-rational
        // fallback rather than merely checking the ordinary TwoSum path.
        for (left, right) in [
            (fallback_left, fallback_right),
            (mirrored_fallback_left, mirrored_fallback_right),
        ] {
            let difference = left - right;
            assert!(difference.is_finite());
            assert!(!(difference - left).is_finite());
        }

        for (left, right) in cases {
            let oracle = exact(left) - exact(right);
            if (left - right).is_finite() {
                let (difference, residual) = exact_subtract_with_residual(left, right, 0).unwrap();
                assert_eq!(exact(difference) + exact(residual), oracle);
            } else {
                assert!(matches!(
                    exact_subtract_with_residual(left, right, 0),
                    Err(ReluTransformError::NonFiniteArithmetic { .. })
                ));
            }
        }
    }

    #[test]
    fn exact_sign_classification_matches_rational_boundaries() {
        let min_subnormal = f64::from_bits(1);
        let cases: &[(f64, &[f64])] = &[
            (-0.0, &[]),
            (0.0, &[]),
            (-min_subnormal, &[min_subnormal]),
            (min_subnormal, &[min_subnormal]),
            (-1.0, &[0.5, 0.5]),
            (1.0, &[0.5, 0.5]),
            (-1.0, &[1.0, min_subnormal]),
            (1.0, &[1.0, min_subnormal]),
            (-0.75, &[0.5, 0.25]),
            (0.75, &[0.5, 0.25]),
            (-0.75, &[0.5, 0.25, min_subnormal]),
            (0.75, &[0.5, 0.25, min_subnormal]),
            (-f64::MAX, &[f64::MAX]),
            (f64::MAX, &[f64::MAX]),
            (-f64::MAX, &[f64::MAX, min_subnormal]),
            (f64::MAX, &[f64::MAX, min_subnormal]),
        ];

        for &(center, terms) in cases {
            let radius = dyadic_radius(terms);
            assert_eq!(
                classify_coordinate_exact(center, &radius, 0).unwrap(),
                rational_classification(center, terms),
                "center={center:?}, terms={terms:?}"
            );
        }
    }

    #[test]
    fn dyadic_accumulator_rejects_nonfinite_operands() {
        for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let mut radius = ExactNonnegativeDyadic::default();
            assert!(matches!(
                radius.add_abs_finite(value, 17, "nonfinite test"),
                Err(ReluTransformError::NonFiniteArithmetic {
                    coordinate: 17,
                    operation: "nonfinite test",
                })
            ));
            assert!(matches!(
                radius.cmp_abs_finite(value, 17, "nonfinite comparison"),
                Err(ReluTransformError::NonFiniteArithmetic {
                    coordinate: 17,
                    operation: "nonfinite comparison",
                })
            ));
        }
    }

    #[test]
    fn exhaustive_nonzero_remainder_witnesses_are_included() {
        let input = ConstrainedZonotope64::try_new(
            vec![-2.0, 3.0, 0.125],
            vec![
                vec![(0, 0.25), (1, 0.5), (2, 1.25)],
                vec![(1, -0.25), (2, -0.75)],
            ],
            array![[1.0, 1.0], [-1.0, 0.0]],
            vec![1.0, 1.0],
            vec![0.125, 0.25, 0.375],
        )
        .unwrap();
        let outputs = [
            transform_relu_unwired(&input, ReluTransformLimits::default()).unwrap(),
            transform_relu_projected_constraints_unwired(&input, ReluTransformLimits::default())
                .unwrap(),
        ];
        let alpha_values = [
            rational(-1, 1),
            rational(-1, 2),
            rational(0, 1),
            rational(1, 2),
            rational(1, 1),
        ];
        let error_scales = [rational(-1, 1), rational(0, 1), rational(1, 1)];

        let mut checked = 0_usize;
        for alpha0 in &alpha_values {
            for alpha1 in &alpha_values {
                let alphas = vec![alpha0.clone(), alpha1.clone()];
                if !constraints_hold(&input, &alphas) {
                    continue;
                }
                for e0 in &error_scales {
                    for e1 in &error_scales {
                        for e2 in &error_scales {
                            let errors = vec![
                                e0 * exact(input.box_remainder()[0]),
                                e1 * exact(input.box_remainder()[1]),
                                e2 * exact(input.box_remainder()[2]),
                            ];
                            for output in &outputs {
                                assert_relu_witness_included(&input, output, &alphas, &errors);
                            }
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(checked >= 500);
    }

    #[test]
    fn mixed_scale_and_subnormal_rounding_is_charged() {
        let min_subnormal = f64::from_bits(1);
        let input = ConstrainedZonotope64::try_new(
            vec![0.0, 1.0e200, -1.0e-200, min_subnormal],
            vec![
                vec![(0, 1.0), (1, 1.0e199), (2, 2.0e-200)],
                vec![(0, -0.25), (3, min_subnormal)],
            ],
            Array2::zeros((0, 2)),
            vec![],
            vec![0.125, 1.0e198, min_subnormal, min_subnormal],
        )
        .unwrap();
        let outputs = [
            transform_relu_unwired(&input, ReluTransformLimits::default()).unwrap(),
            transform_relu_projected_constraints_unwired(&input, ReluTransformLimits::default())
                .unwrap(),
        ];
        for alpha0 in [rational(-1, 1), rational(0, 1), rational(1, 1)] {
            for alpha1 in [rational(-1, 1), rational(1, 1)] {
                for error_sign in [rational(-1, 1), rational(1, 1)] {
                    let errors: Vec<_> = input
                        .box_remainder()
                        .iter()
                        .map(|&radius| &error_sign * exact(radius))
                        .collect();
                    for output in &outputs {
                        assert_relu_witness_included(
                            &input,
                            output,
                            &[alpha0.clone(), alpha1.clone()],
                            &errors,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn every_resource_cap_and_malformed_limit_fails_closed() {
        let input = ConstrainedZonotope64::try_new(
            vec![0.0, 2.0],
            vec![vec![(0, 1.0), (1, 0.5)]],
            array![[1.0]],
            vec![1.0],
            vec![0.25, 0.0],
        )
        .unwrap();
        let base = ReluTransformLimits::default();
        for limited in [
            ReluTransformLimits {
                max_value_dim: 1,
                ..base
            },
            ReluTransformLimits {
                max_output_alpha_dim: 1,
                ..base
            },
            ReluTransformLimits {
                max_constraints: 0,
                ..base
            },
            ReluTransformLimits {
                max_constraint_elements: 1,
                ..base
            },
            ReluTransformLimits {
                max_generator_nnz: 2,
                ..base
            },
            ReluTransformLimits {
                max_unstable: 0,
                ..base
            },
            ReluTransformLimits {
                max_exact_terms: 3,
                ..base
            },
        ] {
            assert!(matches!(
                transform_relu_unwired(&input, limited),
                Err(ReluTransformError::LimitExceeded { .. })
            ));
        }
        for malformed in [
            ReluTransformLimits {
                max_value_dim: RELU_HARD_MAX_VALUE_DIM + 1,
                ..base
            },
            ReluTransformLimits {
                max_output_alpha_dim: RELU_HARD_MAX_OUTPUT_ALPHA_DIM + 1,
                ..base
            },
            ReluTransformLimits {
                max_constraints: RELU_HARD_MAX_CONSTRAINTS + 1,
                ..base
            },
            ReluTransformLimits {
                max_constraint_elements: RELU_HARD_MAX_CONSTRAINT_ELEMENTS + 1,
                ..base
            },
            ReluTransformLimits {
                max_generator_nnz: RELU_HARD_MAX_GENERATOR_NNZ + 1,
                ..base
            },
            ReluTransformLimits {
                max_unstable: RELU_HARD_MAX_UNSTABLE + 1,
                ..base
            },
            ReluTransformLimits {
                max_exact_terms: RELU_HARD_MAX_EXACT_TERMS + 1,
                ..base
            },
        ] {
            assert!(matches!(
                transform_relu_unwired(&input, malformed),
                Err(ReluTransformError::InvalidLimit { .. })
            ));
        }
    }

    #[test]
    fn projected_constraint_resources_fail_closed_before_output_allocation() {
        let input = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![vec![(0, 1.0)]],
            array![[1.0]],
            vec![1.0],
            vec![0.0],
        )
        .unwrap();
        let base = ReluTransformLimits::default();

        // One unstable coordinate produces two new rows and one new alpha:
        // three rows by two columns, or six dense elements.
        for limited in [
            ReluTransformLimits {
                max_constraints: 2,
                ..base
            },
            ReluTransformLimits {
                max_constraint_elements: 5,
                ..base
            },
            ReluTransformLimits {
                // The preserved plan needs 7 terms.  Projecting adds one
                // subtraction term plus six RHS contributors.
                max_exact_terms: 7,
                ..base
            },
        ] {
            assert!(matches!(
                transform_relu_projected_constraints_unwired(&input, limited),
                Err(ReluTransformError::LimitExceeded { .. })
            ));
        }
        transform_relu_unwired(
            &input,
            ReluTransformLimits {
                max_exact_terms: 7,
                ..base
            },
        )
        .unwrap();
    }

    #[test]
    fn required_nonfinite_output_fails_closed() {
        let input = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![vec![(0, f64::MAX)]; 8],
            Array2::zeros((0, 8)),
            vec![],
            vec![0.0],
        )
        .unwrap();
        let result = transform_relu_unwired(&input, ReluTransformLimits::default());
        assert!(
            matches!(
                result,
                Err(ReluTransformError::NonFiniteArithmetic { coordinate: 0, .. })
            ),
            "{result:?}"
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn arbitrary_finite_dyadic_sums_and_signs_match_rational_oracle(
            value_bits in prop::collection::vec(finite_f64_bits(), 0..96),
            center_bits in finite_f64_bits(),
        ) {
            let values: Vec<_> = value_bits.into_iter().map(f64::from_bits).collect();
            let center = f64::from_bits(center_bits);
            let radius = dyadic_radius(&values);
            let oracle = rational_radius(&values);

            prop_assert_eq!(&radius.to_big_rational(), &oracle);
            prop_assert_eq!(
                radius.cmp_abs_finite(center, 0, "property center").unwrap(),
                oracle.cmp(&exact(center).abs()),
            );
            prop_assert_eq!(
                classify_coordinate_exact(center, &radius, 0).unwrap(),
                rational_classification(center, &values),
            );
        }

        #[test]
        fn arbitrary_scaled_dyadic_rounding_matches_rational_oracle(
            coefficient_bits in finite_f64_bits(),
            scale_numerator in 1_u32..=65_535,
            denominator_extra in 0_u32..=65_535,
        ) {
            let coefficient = f64::from_bits(coefficient_bits);
            let scale_denominator = u64::from(scale_numerator)
                + u64::from(denominator_extra);
            let numerator = BigUint::from(scale_numerator);
            let denominator = BigUint::from(scale_denominator);
            let scale = BigRational::new(
                BigInt::from(numerator.clone()),
                BigInt::from(denominator.clone()),
            );
            let mut accumulated = ExactNonnegativeDyadic::default();
            let nominal = nearest_scaled_dyadic(
                coefficient,
                &numerator,
                &denominator,
                &mut accumulated,
                0,
            ).unwrap();
            let ideal = scale * exact(coefficient);
            let oracle_nominal = nearest_finite(
                &ideal,
                0,
                "property scaled coefficient rational oracle",
            ).unwrap();
            let oracle_error = (&ideal - exact(oracle_nominal)).abs();
            let accumulated_error = accumulated.to_big_rational()
                / BigRational::from_integer(BigInt::from(denominator));

            prop_assert_eq!(nominal.to_bits(), oracle_nominal.to_bits());
            prop_assert_eq!(accumulated_error, oracle_error);
        }

        #[test]
        fn two_sum_subtraction_residual_matches_exact_rational_difference(
            left_bits in finite_f64_bits(),
            right_bits in finite_f64_bits(),
        ) {
            let left = f64::from_bits(left_bits);
            let right = f64::from_bits(right_bits);
            let oracle = exact(left) - exact(right);
            match exact_subtract_with_residual(left, right, 0) {
                Ok((difference, residual)) => {
                    prop_assert!(difference.is_finite());
                    prop_assert!(residual.is_finite());
                    prop_assert_eq!(exact(difference) + exact(residual), oracle);
                }
                Err(ReluTransformError::NonFiniteArithmetic { .. }) => {
                    prop_assert!(!(left - right).is_finite());
                }
                Err(other) => prop_assert!(false, "unexpected subtraction error: {other:?}"),
            }
        }

        #[test]
        fn certified_auxiliary_interval_preserves_random_constrained_witnesses(
            center_seed in -16i16..=16,
            generator_seed in 1u16..=16,
            remainder_seed in 0u16..=8,
            lower_alpha_seed in -8i16..=-1,
            upper_alpha_seed in 1i16..=8,
            alpha_mix in 0u16..=16,
            error_mix in -16i16..=16,
        ) {
            // Every quantity is a small dyadic. The predicate rows prove the
            // complete concrete alpha interval, and adding +/- remainder gives
            // an exact independently certified auxiliary enclosure.
            let center = f64::from(center_seed) / 8.0;
            let generator = f64::from(generator_seed) / 8.0;
            let remainder = f64::from(remainder_seed) / 16.0;
            let lower_alpha = f64::from(lower_alpha_seed) / 8.0;
            let upper_alpha = f64::from(upper_alpha_seed) / 8.0;
            let auxiliary_lower = center + generator * lower_alpha - remainder;
            let auxiliary_upper = center + generator * upper_alpha + remainder;
            let input = ConstrainedZonotope64::try_new(
                vec![center],
                vec![vec![(0, generator)]],
                array![[1.0], [-1.0]],
                vec![upper_alpha, -lower_alpha],
                vec![remainder],
            ).unwrap();
            let auxiliary = CertifiedAuxiliaryBounds64::try_new(
                vec![auxiliary_lower],
                vec![auxiliary_upper],
            ).unwrap();

            // Affinity means these four corners establish enclosure of the
            // complete constrained alpha/error rectangle used by the test.
            for alpha_endpoint in [lower_alpha, upper_alpha] {
                for error_endpoint in [-remainder, remainder] {
                    let corner = center + generator * alpha_endpoint + error_endpoint;
                    prop_assert!(corner >= auxiliary_lower);
                    prop_assert!(corner <= auxiliary_upper);
                }
            }

            let alpha = lower_alpha
                + (upper_alpha - lower_alpha) * (f64::from(alpha_mix) / 16.0);
            let error = remainder * (f64::from(error_mix) / 16.0);
            let outputs = [
                transform_relu_with_auxiliary_bounds_unwired(
                    &input,
                    &auxiliary,
                    ReluTransformLimits::default(),
                ).unwrap(),
                transform_relu_projected_constraints_with_auxiliary_bounds_unwired(
                    &input,
                    &auxiliary,
                    ReluTransformLimits::default(),
                ).unwrap(),
            ];
            for output in &outputs {
                assert_auxiliary_relu_witness_included(
                    &input,
                    &auxiliary,
                    output,
                    &[exact(alpha)],
                    &[exact(error)],
                );
            }
        }

        #[test]
        fn random_dyadic_witnesses_survive_exact_rational_replay(
            center_seed in prop::collection::vec(-32i16..=32, 4),
            coefficient_seed in prop::collection::vec(-16i16..=16, 12),
            remainder_seed in prop::collection::vec(1u16..=8, 4),
            alpha_seed in prop::collection::vec(-8i16..=8, 3),
            error_seed in prop::collection::vec(-8i16..=8, 4),
            exponents in prop::collection::vec(-500i16..=500, 4),
        ) {
            let scale = |seed: i16, exponent: i16| {
                f64::from(seed) * 2.0_f64.powi(i32::from(exponent))
            };
            let centers: Vec<_> = (0..4)
                .map(|coordinate| scale(center_seed[coordinate], exponents[coordinate]))
                .collect();
            let mut generators = vec![Vec::new(), Vec::new(), Vec::new()];
            for generator in 0..3 {
                for coordinate in 0..4 {
                    let coefficient = scale(
                        coefficient_seed[generator * 4 + coordinate],
                        exponents[coordinate],
                    );
                    if coefficient != 0.0 {
                        generators[generator].push((coordinate, coefficient));
                    }
                }
            }
            let remainders: Vec<_> = (0..4)
                .map(|coordinate| {
                    f64::from(remainder_seed[coordinate])
                        * 2.0_f64.powi(i32::from(exponents[coordinate]) - 4)
                })
                .collect();
            let input = ConstrainedZonotope64::try_new(
                centers,
                generators,
                Array2::zeros((0, 3)),
                vec![],
                remainders,
            ).unwrap();
            let outputs = [
                transform_relu_unwired(&input, ReluTransformLimits::default()).unwrap(),
                transform_relu_projected_constraints_unwired(
                    &input,
                    ReluTransformLimits::default(),
                ).unwrap(),
            ];
            let alphas: Vec<_> = alpha_seed
                .iter()
                .map(|&seed| rational(i64::from(seed), 8))
                .collect();
            let errors: Vec<_> = error_seed
                .iter()
                .zip(input.box_remainder())
                .map(|(&seed, &radius)| rational(i64::from(seed), 8) * exact(radius))
                .collect();
            for output in &outputs {
                assert_relu_witness_included(&input, output, &alphas, &errors);
            }
        }
    }
}
