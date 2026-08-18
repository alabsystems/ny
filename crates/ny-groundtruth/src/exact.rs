// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact constant handling for ground-truth graph construction.
//!
//! Graph constants are stored as f32 (the `ny-propagate` layer width). The
//! §2.3 contract of `docs/GEOMETRIC_GROUND_TRUTH_PLAN.md` requires that no
//! constant is ever *silently rounded* on its way into a graph:
//!
//! 1. every caller-supplied f64 parameter must be finite and round-trip
//!    exactly through f32 ([`require_exact_f32`]), and
//! 2. every constant *derived* at build time (products like `r^2`, projection
//!    entries `a_i * a_j`, biases `-(I - a a^T) p`) is computed in exact
//!    arbitrary-precision rational arithmetic ([`BigRational`]) and only
//!    accepted if the exact result is itself an f32 value
//!    ([`rational_to_exact_f32`]).
//!
//! Anything else is rejected with a typed error; the plan's interval-widening
//! alternative is a follow-up.

use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};

use crate::error::{GroundTruthError, Result};

/// Validate that `value` is finite and round-trips f64 -> f32 -> f64 exactly.
///
/// This is the entry check for every caller-supplied parameter: the returned
/// f32 denotes *exactly* the same real number as the f64 the caller passed.
pub(crate) fn require_exact_f32(name: &str, value: f64) -> Result<f32> {
    if !value.is_finite() {
        return Err(GroundTruthError::NonFiniteParameter {
            name: name.to_string(),
            value,
        });
    }
    let cast = value as f32;
    if f64::from(cast) == value {
        Ok(cast)
    } else {
        Err(GroundTruthError::InexactParameter {
            name: name.to_string(),
            value,
        })
    }
}

/// Validate a 3-vector parameter component-wise via [`require_exact_f32`].
pub(crate) fn require_exact_vec3(name: &str, v: [f64; 3]) -> Result<[f32; 3]> {
    Ok([
        require_exact_f32(&format!("{name}[0]"), v[0])?,
        require_exact_f32(&format!("{name}[1]"), v[1])?,
        require_exact_f32(&format!("{name}[2]"), v[2])?,
    ])
}

/// Exact rational value of a finite f32 (every finite float is rational).
pub(crate) fn rational(value: f32) -> BigRational {
    BigRational::from_float(value).expect("finite f32 has an exact rational image")
}

/// Convert an exactly computed rational constant to f32, rejecting any value
/// that f32 cannot represent exactly.
///
/// Soundness: the candidate is produced by a (correctly rounded) f64
/// conversion followed by an f32 cast, but acceptance is decided solely by
/// the exact rational round-trip comparison, so no rounding can slip through.
/// If `q` is exactly an f32 value, the correctly rounded conversions return
/// that value and the comparison succeeds; otherwise it fails for every
/// candidate.
pub(crate) fn rational_to_exact_f32(name: &str, q: &BigRational) -> Result<f32> {
    let inexact = || GroundTruthError::InexactDerivedConstant {
        name: name.to_string(),
    };
    let approx = q.to_f64().ok_or_else(inexact)?;
    if !approx.is_finite() {
        return Err(inexact());
    }
    let cast = approx as f32;
    if cast.is_finite() && rational(cast) == *q {
        Ok(cast)
    } else {
        Err(inexact())
    }
}

/// Validate that `axis` is finite, f32-exact, and *exactly* unit length under
/// exact rational arithmetic. Returns the validated f32 components.
///
/// The residual formulas for cylinder/cone/torus use the orthogonal projection
/// `I - a a^T`, which is only a projection for `||a|| = 1`; accepting a nearly
/// unit axis would silently change the ground-truth zero set.
pub(crate) fn require_unit_axis(name: &str, axis: [f64; 3]) -> Result<[f32; 3]> {
    let a = require_exact_vec3(name, axis)?;
    let norm_sq: BigRational = a
        .iter()
        .map(|&ai| {
            let q = rational(ai);
            &q * &q
        })
        .fold(BigRational::zero(), |acc, q| acc + q);
    if norm_sq == BigRational::one() {
        Ok(a)
    } else {
        Err(GroundTruthError::AxisNotUnit {
            name: name.to_string(),
            norm_sq: norm_sq.to_f64().unwrap_or(f64::NAN),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_f32_accepts_dyadic_and_rejects_decimal() {
        assert_eq!(require_exact_f32("x", 0.5).unwrap(), 0.5_f32);
        assert_eq!(require_exact_f32("x", -3.0).unwrap(), -3.0_f32);
        assert!(matches!(
            require_exact_f32("x", 0.1),
            Err(GroundTruthError::InexactParameter { .. })
        ));
        assert!(matches!(
            require_exact_f32("x", f64::NAN),
            Err(GroundTruthError::NonFiniteParameter { .. })
        ));
        assert!(matches!(
            require_exact_f32("x", f64::INFINITY),
            Err(GroundTruthError::NonFiniteParameter { .. })
        ));
    }

    #[test]
    fn rational_round_trip_detects_inexact_squares() {
        // 1.5^2 = 2.25 is exactly representable.
        let ok = rational(1.5) * rational(1.5);
        assert_eq!(rational_to_exact_f32("r^2", &ok).unwrap(), 2.25_f32);

        // 8191.5 is f32-exact (14-bit significand), but its square
        // (2^14 - 1)^2 / 4 has a 28-bit odd numerator: not f32-exact.
        let r = require_exact_f32("r", 8191.5).unwrap();
        let sq = rational(r) * rational(r);
        assert!(matches!(
            rational_to_exact_f32("r^2", &sq),
            Err(GroundTruthError::InexactDerivedConstant { .. })
        ));
    }

    #[test]
    fn unit_axis_is_signed_basis_only_in_f32() {
        assert_eq!(
            require_unit_axis("a", [0.0, 0.0, 1.0]).unwrap(),
            [0.0, 0.0, 1.0]
        );
        assert_eq!(
            require_unit_axis("a", [0.0, -1.0, 0.0]).unwrap(),
            [0.0, -1.0, 0.0]
        );
        // Exactly representable but not unit.
        assert!(matches!(
            require_unit_axis("a", [1.0, 1.0, 0.0]),
            Err(GroundTruthError::AxisNotUnit { .. })
        ));
        // (0.6, 0.8, 0) is unit over the reals, but 0.6/0.8 are not f32-exact:
        // rejected earlier as inexact parameters (never silently rounded).
        assert!(matches!(
            require_unit_axis("a", [0.6, 0.8, 0.0]),
            Err(GroundTruthError::InexactParameter { .. })
        ));
    }
}
