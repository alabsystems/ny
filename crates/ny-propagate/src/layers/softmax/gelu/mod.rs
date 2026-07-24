// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GELU activation layer with IBP and CROWN bound propagation.
//!
//! Supports both Erf and Tanh approximations, with heuristic and sound
//! relaxation modes. Sound mode uses precomputed tangent tables ported
//! from auto_LiRPA's BoundGelu.

mod crown;
pub(crate) mod eval;
mod heuristic_relax;
#[cfg(test)]
mod sound_precision_tests;
mod sound_relax;
mod sound_tables;

use std::borrow::Cow;

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use super::super::common::{BoundPropagation, PARALLEL_ELEMENT_THRESHOLD};
use crate::LinearBounds;

pub use heuristic_relax::RelaxationMode;

// Re-exports for tests and Kani proofs. These functions have no production callers outside this
// module but are exercised by unit tests via `super::*` and `crate::layers::*` (#3240),
// and by external Kani proof harnesses via the `kani-proofs` feature (#2305).
pub use eval::{gelu_eval, gelu_tanh_inflection_point};
pub use heuristic_relax::{adaptive_gelu_linear_relaxation, gelu_linear_relaxation};
pub use sound_relax::{gelu_sound_linear_relaxation, gelu_tanh_sound_linear_relaxation};

pub(crate) use eval::gelu_bound_interval;

/// GELU approximation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeluApproximation {
    /// Exact: `0.5 * x * (1 + erf(x / sqrt(2)))`.
    Erf,
    /// Approximate: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    Tanh,
}

/// GELU activation layer.
#[derive(Debug, Clone)]
pub struct GELULayer {
    pub approximation: GeluApproximation,
    /// Relaxation mode for CROWN backward propagation.
    /// Default is Chord for backwards compatibility.
    pub relaxation_mode: RelaxationMode,
    /// Use sound (no sampling) relaxation.
    ///
    /// When true, uses precomputed tangent tables instead of sampling-based bounds.
    /// This provides provable soundness. Supported for both Erf and Tanh approximations.
    ///
    /// Default: `true` (sound by default). Set to `false` only if you need the heuristic
    /// Chord/Tangent/TwoSlope/Adaptive modes for experimentation.
    pub sound: bool,
}

impl GELULayer {
    /// Create a new GELU layer with the given approximation mode.
    pub fn new(approximation: GeluApproximation) -> Self {
        Self {
            approximation,
            relaxation_mode: RelaxationMode::default(),
            sound: true,
        }
    }

    /// Create a new GELU layer with specified relaxation mode.
    ///
    /// Note: `sound` defaults to `true`. The `relaxation_mode` is only used when
    /// `sound` is `false` (for heuristic experimentation).
    pub fn with_relaxation(approximation: GeluApproximation, mode: RelaxationMode) -> Self {
        Self {
            approximation,
            relaxation_mode: mode,
            sound: true,
        }
    }

    /// Create a new GELU layer with adaptive relaxation (best tightness).
    ///
    /// Sets `sound: false` to enable the heuristic adaptive mode.
    pub fn adaptive(approximation: GeluApproximation) -> Self {
        Self {
            approximation,
            relaxation_mode: RelaxationMode::Adaptive,
            sound: false,
        }
    }

    /// Create a sound GELU layer.
    ///
    /// Uses precomputed tangent tables for provably sound relaxation.
    /// This is now the default; this constructor is kept for explicitness.
    pub fn sound(approximation: GeluApproximation) -> Self {
        Self {
            approximation,
            relaxation_mode: RelaxationMode::default(),
            sound: true,
        }
    }

    /// Returns true if this layer uses sound (no sampling) relaxation.
    pub fn is_sound(&self) -> bool {
        self.sound
    }
}

impl Default for GELULayer {
    fn default() -> Self {
        Self {
            approximation: GeluApproximation::Erf,
            relaxation_mode: RelaxationMode::default(),
            sound: true,
        }
    }
}

impl BoundPropagation for GELULayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let mut out_lower = input.lower().clone();
        let mut out_upper = input.upper().clone();
        let approx = self.approximation;

        let zip = ndarray::Zip::from(&mut out_lower)
            .and(&mut out_upper)
            .and(input.lower())
            .and(input.upper());

        if input.len() >= PARALLEL_ELEMENT_THRESHOLD {
            zip.par_for_each(|ol, ou, &il, &iu| {
                let (l, u) = gelu_bound_interval(il, iu, approx);
                *ol = l;
                *ou = u;
            });
        } else {
            zip.for_each(|ol, ou, &il, &iu| {
                let (l, u) = gelu_bound_interval(il, iu, approx);
                *ol = l;
                *ou = u;
            });
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "GELU is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
                .to_string(),
        ))
    }

    fn requires_pre_activation_bounds(&self) -> bool {
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        GELULayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

#[cfg(test)]
mod tests {
    use super::eval::{gelu_erf, gelu_tanh};
    use super::*;
    use ndarray::arr1;
    use proptest::prelude::*;

    fn f32_any_with_specials() -> impl Strategy<Value = f32> {
        prop_oneof![
            Just(f32::NEG_INFINITY),
            Just(f32::INFINITY),
            Just(0.0_f32),
            Just(f32::NAN),
            prop::num::f32::ANY
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(256) })]

        /// #1836 acceptance: exercise GELU(erf) eval on ANY f32 plus explicit IEEE corner values.
        #[test]
        fn proptest_gelu_erf_eval_handles_special_values(x in f32_any_with_specials()) {
            let y = gelu_erf(x);
            if x.is_nan() {
                prop_assert!(y.is_nan(), "gelu_erf(NaN) should be NaN, got {}", y);
            } else if x == f32::NEG_INFINITY {
                prop_assert_eq!(y, 0.0, "gelu_erf(-inf) should be 0.0, got {}", y);
            } else if x == f32::INFINITY {
                prop_assert_eq!(y, f32::INFINITY, "gelu_erf(+inf) should be +inf, got {}", y);
            } else {
                prop_assert!(!y.is_nan(), "gelu_erf({}) should not be NaN, got {}", x, y);
            }
        }

        /// #1836 acceptance: exercise GELU(tanh) eval on ANY f32 plus explicit IEEE corner values.
        #[test]
        fn proptest_gelu_tanh_eval_handles_special_values(x in f32_any_with_specials()) {
            let y = gelu_tanh(x);
            if x.is_nan() {
                prop_assert!(y.is_nan(), "gelu_tanh(NaN) should be NaN, got {}", y);
            } else if x == f32::NEG_INFINITY {
                prop_assert_eq!(y, 0.0, "gelu_tanh(-inf) should be 0.0, got {}", y);
            } else if x == f32::INFINITY {
                prop_assert_eq!(y, f32::INFINITY, "gelu_tanh(+inf) should be +inf, got {}", y);
            } else {
                prop_assert!(!y.is_nan(), "gelu_tanh({}) should not be NaN, got {}", x, y);
            }
        }

        /// #1836 acceptance: finite-width IBP intervals must not produce NaN bounds.
        #[test]
        fn proptest_gelu_ibp_no_nan_for_finite_intervals(a in prop::num::f32::ANY, b in prop::num::f32::ANY) {
            prop_assume!(a.is_finite() && b.is_finite());
            prop_assume!(a.abs() <= 1.0e6 && b.abs() <= 1.0e6);
            let (l, u) = (a.min(b), a.max(b));
            let input = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();

            for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
                let layer = GELULayer::new(approx);
                let output = layer.propagate_ibp(&input).unwrap();
                let lower = output.lower()[[0]];
                let upper = output.upper()[[0]];
                prop_assert!(!lower.is_nan(), "GELU({approx:?}) IBP lower is NaN for [{l}, {u}]");
                prop_assert!(!upper.is_nan(), "GELU({approx:?}) IBP upper is NaN for [{l}, {u}]");
                prop_assert!(
                    lower <= upper,
                    "GELU({approx:?}) IBP bounds inverted for [{l}, {u}]: {lower} > {upper}"
                );
            }
        }
    }

    // ── f64 chord precision proptest (#2624) ─────────────────────────

    /// ULP distance between two f32 values, handling sign correctly.
    fn ulp_distance(a: f32, b: f32) -> u64 {
        fn to_ordered(x: f32) -> i64 {
            let bits = x.to_bits() as i32;
            if bits < 0 {
                (0x8000_0000_u32 as i32 - bits) as i64
            } else {
                bits as i64
            }
        }
        (to_ordered(a) - to_ordered(b)).unsigned_abs()
    }

    proptest! {
        #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

        /// #2624: Verify GELU heuristic chord slope has f64 precision for narrow intervals.
        /// For intervals with width in [1e-8, 1e-4], gelu_linear_relaxation uses a chord
        /// slope computed in f64. Verify the returned slope matches an independent f64 reference.
        #[test]
        fn proptest_gelu_heuristic_chord_f64_precision(l in -8.0f32..8.0, width_exp in -7.0f64..-4.0) {
            let delta = 10.0_f64.powf(width_exp) as f32;
            let u = l + delta;
            prop_assume!(u > l);
            // Stay above the 1e-8 degenerate guard in gelu_linear_relaxation
            prop_assume!((u - l) > 1e-8);

            for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
                let (ls, _li, us, _ui) = gelu_linear_relaxation(l, u, approx);

                // The heuristic chord mode uses same slope for lower and upper
                prop_assert_eq!(ls, us, "GELU heuristic chord: ls != us for [{}, {}]", l, u);

                // Independent f64 reference
                let l64 = l as f64;
                let u64 = u as f64;
                let gelu_f64 = |x: f64| match approx {
                    GeluApproximation::Erf => eval::gelu_erf_f64(x),
                    GeluApproximation::Tanh => eval::gelu_tanh_f64(x),
                };
                let fl64 = gelu_f64(l64);
                let fu64 = gelu_f64(u64);
                let ref_slope = ((fu64 - fl64) / (u64 - l64)) as f32;

                let slope_ulps = ulp_distance(ls, ref_slope);
                prop_assert!(
                    slope_ulps <= 1,
                    "GELU({approx:?}) heuristic chord slope not within 1 ULP: \
                     got {ls} vs ref {ref_slope} ({slope_ulps} ULPs) for [{l}, {u}]"
                );
            }
        }
    }

    /// Regression test for #1836: gelu_erf(-inf) must return 0, not NaN.
    /// GELU(x) = 0.5 * x * (1 + erf(x/√2)); at x = -inf this is 0.5*(-inf)*0 = NaN
    /// without the guard. Correct limit: GELU(-inf) = 0.
    #[test]
    fn test_gelu_erf_neg_infinity_returns_zero() {
        let result = gelu_erf(f32::NEG_INFINITY);
        assert_eq!(result, 0.0, "gelu_erf(-inf) should be 0.0, got {result}");
    }

    /// Regression test for #1836: gelu_erf(+inf) must return +inf.
    #[test]
    fn test_gelu_erf_pos_infinity_returns_pos_infinity() {
        let result = gelu_erf(f32::INFINITY);
        assert_eq!(
            result,
            f32::INFINITY,
            "gelu_erf(+inf) should be +inf, got {result}"
        );
    }

    /// Regression test for #1836: gelu_erf(NaN) must return NaN.
    #[test]
    fn test_gelu_erf_nan_returns_nan() {
        let result = gelu_erf(f32::NAN);
        assert!(result.is_nan(), "gelu_erf(NaN) should be NaN, got {result}");
    }

    /// Regression test for #1836: gelu_tanh(-inf) must return 0, not NaN.
    #[test]
    fn test_gelu_tanh_neg_infinity_returns_zero() {
        let result = gelu_tanh(f32::NEG_INFINITY);
        assert_eq!(result, 0.0, "gelu_tanh(-inf) should be 0.0, got {result}");
    }

    /// Regression test for #1836: gelu_tanh(+inf) must return +inf.
    #[test]
    fn test_gelu_tanh_pos_infinity_returns_pos_infinity() {
        let result = gelu_tanh(f32::INFINITY);
        assert_eq!(
            result,
            f32::INFINITY,
            "gelu_tanh(+inf) should be +inf, got {result}"
        );
    }

    /// Regression test for #1836: gelu_tanh(NaN) must return NaN.
    #[test]
    fn test_gelu_tanh_nan_returns_nan() {
        let result = gelu_tanh(f32::NAN);
        assert!(
            result.is_nan(),
            "gelu_tanh(NaN) should be NaN, got {result}"
        );
    }

    /// Regression test for #1836: gelu_eval dispatches correctly for both approximations.
    #[test]
    fn test_gelu_eval_infinite_bounds() {
        assert_eq!(gelu_eval(f32::NEG_INFINITY, GeluApproximation::Erf), 0.0);
        assert_eq!(
            gelu_eval(f32::INFINITY, GeluApproximation::Erf),
            f32::INFINITY
        );
        assert_eq!(gelu_eval(f32::NEG_INFINITY, GeluApproximation::Tanh), 0.0);
        assert_eq!(
            gelu_eval(f32::INFINITY, GeluApproximation::Tanh),
            f32::INFINITY
        );
    }

    /// Regression test for #1836: gelu_bound_interval with infinite inputs must not produce NaN.
    #[test]
    fn test_gelu_bound_interval_infinite_inputs() {
        let (min_v, max_v) =
            gelu_bound_interval(f32::NEG_INFINITY, f32::INFINITY, GeluApproximation::Erf);
        assert!(
            !min_v.is_nan(),
            "gelu_bound_interval min must not be NaN, got {min_v}"
        );
        assert!(
            !max_v.is_nan(),
            "gelu_bound_interval max must not be NaN, got {max_v}"
        );
        // GELU min is at critical point ≈ -0.17, so min_v should be around that
        assert!(min_v <= 0.0, "gelu_bound_interval min should be <= 0");

        let (min_v, max_v) = gelu_bound_interval(f32::NEG_INFINITY, 0.0, GeluApproximation::Erf);
        assert!(
            !min_v.is_nan(),
            "gelu_bound_interval(-inf, 0) min must not be NaN"
        );
        assert!(
            !max_v.is_nan(),
            "gelu_bound_interval(-inf, 0) max must not be NaN"
        );
    }
}
