// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest soundness fuzzing for CROWN backward composition primitives.
//!
//! Verifies three properties of `compose_lower`/`compose_upper`:
//! 1. Directed rounding soundness: lower <= true product, upper >= true product
//! 2. NaN freedom: no NaN in outputs for any IEEE 754 input
//! 3. Overflow detection: nonfinite flag ↔ non-finite product
//!
//! Part of #3517 composition soundness hardening.

use proptest::prelude::*;

use crate::layers::activations::LinearRelaxation;
use crate::layers::common::compose::{compose_lower, compose_upper};

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// For any finite coefficient and finite relaxation slopes, directed
    /// rounding must be sound: compose_lower <= true product (for lower)
    /// and compose_upper >= true product (for upper).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_compose_directed_rounding(
        a in -1.0e6f32..1.0e6,
        ls in -10.0f32..10.0, li in -10.0f32..10.0,
        us in -10.0f32..10.0, ui in -10.0f32..10.0,
    ) {
        let relax = LinearRelaxation::new(ls, li, us, ui);
        let lo = compose_lower(a, &relax);
        let hi = compose_upper(a, &relax);
        if a > 0.0 {
            let true_lo = a * ls;
            let true_hi = a * us;
            if true_lo.is_finite() {
                prop_assert!(lo.new_coeff <= true_lo,
                    "lower({a}) = {} > true {true_lo}", lo.new_coeff);
            }
            if true_hi.is_finite() {
                prop_assert!(hi.new_coeff >= true_hi,
                    "upper({a}) = {} < true {true_hi}", hi.new_coeff);
            }
        } else if a < 0.0 {
            let true_lo = a * us;
            let true_hi = a * ls;
            if true_lo.is_finite() {
                prop_assert!(lo.new_coeff <= true_lo,
                    "lower({a}) = {} > true {true_lo}", lo.new_coeff);
            }
            if true_hi.is_finite() {
                prop_assert!(hi.new_coeff >= true_hi,
                    "upper({a}) = {} < true {true_hi}", hi.new_coeff);
            }
        }
    }

    /// No NaN in any output field for any f32 coefficient with finite relaxation.
    /// Contract: relaxation slopes/intercepts are always finite (computed from
    /// valid pre-activation bounds). Slopes may be Inf for zero-width domains.
    /// Coefficients may be any f32 (NaN/Inf are degenerate but must not produce
    /// NaN in coefficient output). Intercept NaN is acceptable when nonfinite=true
    /// because callers replace the entire row with +/-Inf bias.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_compose_nan_freedom(
        a in prop::num::f32::ANY,
        ls in prop_oneof![
            -100.0f32..100.0,
            Just(f32::INFINITY), Just(f32::NEG_INFINITY),
        ],
        li in -100.0f32..100.0,
        us in prop_oneof![
            -100.0f32..100.0,
            Just(f32::INFINITY), Just(f32::NEG_INFINITY),
        ],
        ui in -100.0f32..100.0,
    ) {
        let relax = LinearRelaxation::new(ls, li, us, ui);
        let lo = compose_lower(a, &relax);
        let hi = compose_upper(a, &relax);
        // Coefficient must never be NaN regardless of input
        prop_assert!(!lo.new_coeff.is_nan(), "lower coeff NaN for a={a}");
        prop_assert!(!hi.new_coeff.is_nan(), "upper coeff NaN for a={a}");
        // Intercept must be NaN-free on the finite product path
        if !lo.nonfinite {
            prop_assert!(!lo.intercept_contrib.is_nan(),
                "lower intercept NaN on finite path for a={a}");
        }
        if !hi.nonfinite {
            prop_assert!(!hi.intercept_contrib.is_nan(),
                "upper intercept NaN on finite path for a={a}");
        }
    }

    /// When product overflows, nonfinite flag is set and coeff is zeroed.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_compose_overflow_detection(
        a in prop_oneof![
            Just(f32::MAX), Just(-f32::MAX),
            Just(f32::MAX / 2.0), Just(-f32::MAX / 2.0),
            -1.0e6f32..1.0e6,
        ],
        s in prop_oneof![
            Just(f32::MAX), Just(-f32::MAX),
            Just(f32::MAX / 2.0), Just(-f32::MAX / 2.0),
            -10.0f32..10.0,
        ],
    ) {
        let relax = LinearRelaxation::new(s, 0.0, s, 0.0);
        let lo = compose_lower(a, &relax);
        let hi = compose_upper(a, &relax);
        // If product overflows, coeff must be zeroed and flag set
        for (lbl, r) in [("lower", &lo), ("upper", &hi)] {
            if r.nonfinite {
                prop_assert!(r.new_coeff == 0.0,
                    "{}: nonfinite but coeff not zeroed: {}", lbl, r.new_coeff);
            } else if a != 0.0 {
                prop_assert!(r.new_coeff.is_finite(),
                    "{}: finite flag but coeff={}", lbl, r.new_coeff);
            }
        }
    }
}
