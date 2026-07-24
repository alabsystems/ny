// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN and IBP soundness tests for piecewise activation layers.
//!
//! Negative-coefficient tests are in [`crown_piecewise_negcoeff`](super::crown_piecewise_negcoeff).
//! Asymmetric-bound tests are in [`crown_piecewise_asymmetric`](super::crown_piecewise_asymmetric).
//!
//! ## CROWN soundness status
//!
//! **PReLU**: CROWN is sound — proptest passes across negative and positive slopes.
//!
//! **Clip**: CROWN is sound — 6-case analytical relaxation. Fixed in #1714.
//!
//! **HardSigmoid**: CROWN is sound — 6-case analytical relaxation. Fixed in #1714.
//!
//! **Softsign**: CROWN is sound — analytical tangent-line relaxation. Fixed in #1715.
//!
//! **HardSwish**: CROWN is sound — analytical max-deviation relaxation. Fixed in #1715.
//!
//! ## IBP soundness status
//!
//! All five piecewise layers (Clip, HardSigmoid, Softsign, PReLU, HardSwish)
//! have sound IBP implementations. Most are monotonic (IBP = evaluate f at
//! bounds); HardSwish also checks the critical point at x=-1.5.
//! IBP proptests verify this.

use crate::layers::common::BoundPropagation;
use crate::layers::{
    ClipLayer, HardSigmoidLayer, HardSwishLayer, LeakyReLULayer, PReluLayer, ReLULayer,
    SoftsignLayer,
};
use crate::LinearBounds;
use ndarray::arr1;
use ny_core::NyError;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{
    assert_crown_backward_sound, hardswish_eval, sample_points, CROWN_TOLERANCE, FP_TOLERANCE,
};

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    // =========================================================================
    // PReLU CROWN backward soundness (SOUND — proptest passes)
    // =========================================================================

    /// PReLU(x, slope) = x if x >= 0, slope*x if x < 0.
    /// CROWN relaxation must remain sound across negative and positive slopes.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_prelu_crown(l in -10.0f32..10.0, delta in 0.0f32..20.0, slope in -50.0f32..50.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);
        prop_assume!(slope.abs() >= 0.01);

        let layer = PReluLayer::from_scalar(slope);
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            |x| if x >= 0.0 { x } else { slope * x },
            &result,
            &format!("PReLU({slope})"),
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Clip CROWN backward soundness (6-case analytical relaxation)
    // =========================================================================

    /// Clip(x, 0, 1) = clamp(x, 0, 1). Piecewise linear with two breakpoints.
    /// Fixed by 6-case analytical relaxation per BoundHardTanh pattern.
    /// Reference: designs/2026-02-08-piecewise-crown-relaxation-fixes.md Part 1
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_clip_crown(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let layer = ClipLayer::new(0.0, 1.0);
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            |x| x.clamp(0.0, 1.0),
            &result,
            "Clip(0,1)",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // HardSigmoid CROWN backward soundness (6-case analytical relaxation)
    // =========================================================================

    /// HardSigmoid(x) = clamp(0.2*x + 0.5, 0, 1). Piecewise linear.
    /// Fixed by 6-case analytical relaxation per BoundHardTanh pattern.
    /// Reference: designs/2026-02-08-piecewise-crown-relaxation-fixes.md Part 2
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_hardsigmoid_crown(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let layer = HardSigmoidLayer::default_params();
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            |x| (0.2 * x + 0.5).clamp(0.0, 1.0),
            &result,
            "HardSigmoid",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Softsign CROWN backward soundness (analytical tangent-line relaxation)
    // =========================================================================

    /// Softsign(x) = x / (1 + |x|). S-shaped: convex for x<0, concave for x>0.
    /// Analytical CROWN relaxation using tangent-line approach per BoundSShaped.
    /// Reference: designs/2026-02-08-piecewise-crown-relaxation-fixes.md Part 3
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_softsign_crown(l in -20.0f32..20.0, delta in 0.0f32..40.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let layer = SoftsignLayer::new();
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            |x| x / (1.0 + x.abs()),
            &result,
            "Softsign",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // HardSwish CROWN backward soundness (analytical max-deviation relaxation)
    // =========================================================================

    /// HardSwish(x) = x * clamp(x/6 + 0.5, 0, 1). Non-monotonic (min at x≈-1.5).
    /// Analytical max-deviation relaxation for boundary-crossing intervals.
    /// Reference: designs/2026-02-08-piecewise-crown-relaxation-fixes.md Part 4
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_hardswish_crown(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let layer = HardSwishLayer::new();
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            hardswish_eval,
            &result,
            "HardSwish",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // ReLU CROWN backward soundness
    // =========================================================================

    /// ReLU(x) = max(0, x). Piecewise linear: identity for x >= 0, zero for x < 0.
    /// CROWN relaxation:
    ///   Upper: lambda * (x - l) where lambda = u/(u-l) for crossing neurons
    ///   Lower: alpha * x where alpha ∈ {0, 1} heuristic
    ///
    /// Reference: alpha-beta-CROWN auto_LiRPA/operators/relu.py
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_relu_crown(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let layer = ReLULayer::new();
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            |x| x.max(0.0),
            &result,
            "ReLU",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // LeakyReLU CROWN backward soundness
    // =========================================================================

    /// LeakyReLU(x, alpha) = x if x >= 0, alpha*x if x < 0.
    /// CROWN relaxation follows the same pattern as ReLU with slope = alpha
    /// for the negative region instead of slope = 0.
    ///
    /// Tests with alpha in (-3.0, 3.0), excluding near-zero, to cover
    /// alpha < 0, alpha in (0,1), and alpha > 1.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_leaky_relu_crown(l in -10.0f32..10.0, delta in 0.0f32..20.0, alpha in -3.0f32..3.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);
        prop_assume!(alpha.abs() >= 0.01);

        let layer = LeakyReLULayer::new(alpha);
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            |x| if x >= 0.0 { x } else { alpha * x },
            &result,
            "LeakyReLU",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // IBP soundness proptests (all layers — all pass)
    // =========================================================================

    /// ReLU IBP: monotonically increasing, so IBP is trivially sound.
    /// ReLU is the most fundamental activation in NN verification; proptest IBP
    /// coverage was previously missing despite every other piecewise activation
    /// having one. Added to close this gap.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_relu_ibp(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let layer = ReLULayer::new();
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let output = layer.propagate_ibp(&input).unwrap();
        let out_l = output.lower()[[0]];
        let out_u = output.upper()[[0]];

        // Output bounds must be valid
        prop_assert!(out_l <= out_u + 1e-6, "IBP output: lower {} > upper {}", out_l, out_u);

        // Sample points must be within output bounds
        for x in sample_points(l, u, 50) {
            let fx = x.max(0.0);
            prop_assert!(
                out_l <= fx + FP_TOLERANCE,
                "ReLU IBP lower {} > relu({}) = {}", out_l, x, fx
            );
            prop_assert!(
                out_u >= fx - FP_TOLERANCE,
                "ReLU IBP upper {} < relu({}) = {}", out_u, x, fx
            );
        }
    }

    /// Clip IBP: clip is monotonically increasing, so IBP is sound.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_clip_ibp(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let layer = ClipLayer::new(0.0, 1.0);
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let output = layer.propagate_ibp(&input).unwrap();
        let out_l = output.lower()[[0]];
        let out_u = output.upper()[[0]];

        // Output bounds must be valid
        prop_assert!(out_l <= out_u + 1e-6, "IBP output: lower > upper");

        // Sample points must be within output bounds
        for x in sample_points(l, u, 50) {
            let fx = x.clamp(0.0, 1.0);
            prop_assert!(
                out_l <= fx + FP_TOLERANCE,
                "Clip IBP lower {} > clip({}) = {}", out_l, x, fx
            );
            prop_assert!(
                out_u >= fx - FP_TOLERANCE,
                "Clip IBP upper {} < clip({}) = {}", out_u, x, fx
            );
        }
    }

    /// HardSigmoid IBP: monotonically increasing, IBP is sound.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_hardsigmoid_ibp(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let layer = HardSigmoidLayer::default_params();
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let output = layer.propagate_ibp(&input).unwrap();
        let out_l = output.lower()[[0]];
        let out_u = output.upper()[[0]];

        prop_assert!(out_l <= out_u + 1e-6, "IBP output: lower > upper");

        for x in sample_points(l, u, 50) {
            let fx = (0.2 * x + 0.5).clamp(0.0, 1.0);
            prop_assert!(
                out_l <= fx + FP_TOLERANCE,
                "HardSigmoid IBP lower {} > f({}) = {}", out_l, x, fx
            );
            prop_assert!(
                out_u >= fx - FP_TOLERANCE,
                "HardSigmoid IBP upper {} < f({}) = {}", out_u, x, fx
            );
        }
    }

    /// Softsign IBP: monotonically increasing, IBP is sound.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_softsign_ibp(l in -20.0f32..20.0, delta in 0.0f32..40.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let layer = SoftsignLayer::new();
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let output = layer.propagate_ibp(&input).unwrap();
        let out_l = output.lower()[[0]];
        let out_u = output.upper()[[0]];

        prop_assert!(out_l <= out_u + 1e-6, "IBP output: lower > upper");

        for x in sample_points(l, u, 50) {
            let fx = x / (1.0 + x.abs());
            prop_assert!(
                out_l <= fx + FP_TOLERANCE,
                "Softsign IBP lower {} > f({}) = {}", out_l, x, fx
            );
            prop_assert!(
                out_u >= fx - FP_TOLERANCE,
                "Softsign IBP upper {} < f({}) = {}", out_u, x, fx
            );
        }
    }

    /// PReLU IBP: monotonically increasing for positive slopes, IBP is sound.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_prelu_ibp(l in -10.0f32..10.0, delta in 0.0f32..20.0, slope in 0.01f32..50.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);
        let layer = PReluLayer::from_scalar(slope);
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let output = layer.propagate_ibp(&input).unwrap();
        let out_l = output.lower()[[0]];
        let out_u = output.upper()[[0]];

        prop_assert!(out_l <= out_u + 1e-6, "IBP output: lower > upper");

        for x in sample_points(l, u, 50) {
            let fx = if x >= 0.0 { x } else { slope * x };
            prop_assert!(
                out_l <= fx + FP_TOLERANCE,
                "PReLU IBP lower {} > f({}) = {}", out_l, x, fx
            );
            prop_assert!(
                out_u >= fx - FP_TOLERANCE,
                "PReLU IBP upper {} < f({}) = {}", out_u, x, fx
            );
        }
    }

    /// PReLU IBP with negative slopes (alpha < 0): V-shaped function.
    /// PReLU(x) = x if x >= 0, slope*x if x < 0. With slope < 0, the negative
    /// branch reflects upward, creating a V-shape with minimum at x=0.
    /// Part of #1914: verifies the IBP crossing fix for negative slopes.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_prelu_ibp_negative_slope(
        l in -10.0f32..10.0,
        delta in 0.0f32..20.0,
        slope in -50.0f32..-0.01,
    ) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let layer = PReluLayer::from_scalar(slope);
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let output = layer.propagate_ibp(&input).unwrap();
        let out_l = output.lower()[[0]];
        let out_u = output.upper()[[0]];

        prop_assert!(out_l <= out_u + 1e-6, "IBP output: lower {} > upper {}", out_l, out_u);

        for x in sample_points(l, u, 50) {
            let fx = if x >= 0.0 { x } else { slope * x };
            prop_assert!(
                out_l <= fx + FP_TOLERANCE,
                "PReLU neg slope IBP lower {} > f({}) = {} (slope={})", out_l, x, fx, slope
            );
            prop_assert!(
                out_u >= fx - FP_TOLERANCE,
                "PReLU neg slope IBP upper {} < f({}) = {} (slope={})", out_u, x, fx, slope
            );
        }
    }

    /// PReLU IBP with alpha=0 boundary case: reduces to ReLU.
    /// PReLU(x, 0) = x if x >= 0, 0 if x < 0. This is exactly ReLU.
    /// Part of #1914.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_prelu_ibp_alpha_zero(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let layer = PReluLayer::from_scalar(0.0);
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let output = layer.propagate_ibp(&input).unwrap();
        let out_l = output.lower()[[0]];
        let out_u = output.upper()[[0]];

        prop_assert!(out_l <= out_u + 1e-6, "IBP output: lower {} > upper {}", out_l, out_u);

        for x in sample_points(l, u, 50) {
            let fx = x.max(0.0); // ReLU
            prop_assert!(
                out_l <= fx + FP_TOLERANCE,
                "PReLU(0) IBP lower {} > relu({}) = {}", out_l, x, fx
            );
            prop_assert!(
                out_u >= fx - FP_TOLERANCE,
                "PReLU(0) IBP upper {} < relu({}) = {}", out_u, x, fx
            );
        }
    }

}

// =============================================================================
// CROWN soundness regression tests for FORMERLY unsound relaxations.
//
// Each test pins concrete inputs where the old chord-based CROWN relaxation
// violated the envelope condition, and asserts the fixed analytical relaxation
// satisfies lower <= f(x) <= upper on them. A failure here is a soundness
// REGRESSION. See #1714, #1715 and
// designs/2026-02-08-piecewise-crown-relaxation-fixes.md.
// =============================================================================

/// Clip CROWN: regression test for boundary-crossing soundness.
/// Previously the chord was used for both upper and lower bounds, causing
/// upper < f(x) in the identity region. Fixed by 6-case analytical relaxation.
/// Reference: designs/2026-02-08-piecewise-crown-relaxation-fixes.md Part 1
#[ntest::timeout(10000)]
#[test]
fn regression_clip_crown_unsound_crossing() {
    let l = 0.0f32;
    let u = 7.364379f32;

    let layer = ClipLayer::new(0.0, 1.0);
    let identity = LinearBounds::identity(1);
    let pre_activation = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();

    let result = layer
        .propagate_linear_with_bounds(&identity, &pre_activation)
        .unwrap();

    // Verify soundness at multiple sample points across the interval.
    for i in 0..=100 {
        let t = i as f32 / 100.0;
        let x = (l + t * (u - l)).clamp(l, u);
        let fx = x.clamp(0.0, 1.0);
        let lower = result.lower_a[[0, 0]] * x + result.lower_b[0];
        let upper = result.upper_a[[0, 0]] * x + result.upper_b[0];
        assert!(
            lower <= fx + 1e-5,
            "Clip CROWN lower bound violated at x={x}: lower={lower} > clip({x})={fx}"
        );
        assert!(
            upper >= fx - 1e-5,
            "Clip CROWN upper bound violated at x={x}: upper={upper} < clip({x})={fx}"
        );
    }
}

/// HardSigmoid CROWN: regression test for boundary-crossing soundness.
/// Previously the chord was used for both upper and lower bounds, causing
/// upper < f(x) in the linear region. Fixed by 6-case analytical relaxation.
/// Reference: designs/2026-02-08-piecewise-crown-relaxation-fixes.md Part 2
#[ntest::timeout(10000)]
#[test]
fn regression_hardsigmoid_crown_unsound_crossing() {
    let l = 0.0f32;
    let u = 14.728758f32;

    let layer = HardSigmoidLayer::default_params();
    let identity = LinearBounds::identity(1);
    let pre_activation = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();

    let result = layer
        .propagate_linear_with_bounds(&identity, &pre_activation)
        .unwrap();

    // Verify soundness at multiple sample points across the interval.
    for i in 0..=100 {
        let t = i as f32 / 100.0;
        let x = (l + t * (u - l)).clamp(l, u);
        let fx = (0.2 * x + 0.5).clamp(0.0, 1.0);
        let lower = result.lower_a[[0, 0]] * x + result.lower_b[0];
        let upper = result.upper_a[[0, 0]] * x + result.upper_b[0];
        assert!(
            lower <= fx + 1e-5,
            "HardSigmoid CROWN lower violated at x={x}: lower={lower} > f({x})={fx}"
        );
        assert!(
            upper >= fx - 1e-5,
            "HardSigmoid CROWN upper violated at x={x}: upper={upper} < f({x})={fx}"
        );
    }
}

/// Softsign CROWN: regression test for boundary-crossing soundness.
/// Previously the chord sandwich with 50-point sampling missed the max
/// deviation for wide asymmetric intervals. Fixed by analytical tangent-line
/// relaxation per BoundSShaped.
/// Reference: designs/2026-02-08-piecewise-crown-relaxation-fixes.md Part 3
#[ntest::timeout(10000)]
#[test]
fn regression_softsign_crown_unsound_wide() {
    // Use multiple intervals that previously triggered unsoundness:
    // asymmetric wide intervals where sampling-based chord missed the peak.
    let test_intervals: &[(f32, f32)] = &[(-5.3238926, 29.337702), (0.0, 29.457516), (-2.0, 25.0)];

    let layer = SoftsignLayer::new();

    for &(l, u) in test_intervals {
        let identity = LinearBounds::identity(1);
        let pre_activation =
            BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        // Verify soundness at 100 sample points across the interval.
        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let x = (l + t * (u - l)).clamp(l, u);
            let fx = x / (1.0 + x.abs());
            let lower = result.lower_a[[0, 0]] * x + result.lower_b[0];
            let upper = result.upper_a[[0, 0]] * x + result.upper_b[0];
            assert!(
                lower <= fx + 1e-4,
                "Softsign CROWN lower violated on [{l}, {u}] at x={x}: \
                 lower={lower} > f(x)={fx}"
            );
            assert!(
                upper >= fx - 1e-4,
                "Softsign CROWN upper violated on [{l}, {u}] at x={x}: \
                 upper={upper} < f(x)={fx}"
            );
        }
    }
}

/// HardSwish CROWN: regression test for boundary-crossing soundness.
/// Previously the chord was used as lower bound, overestimating at the
/// boundary crossing. Fixed by analytical max-deviation relaxation.
/// Reference: designs/2026-02-08-piecewise-crown-relaxation-fixes.md Part 4
#[ntest::timeout(10000)]
#[test]
fn regression_hardswish_crown_unsound_crossing() {
    let l = -3.889787f32;
    let u = 10.838971f32;

    let layer = HardSwishLayer::new();
    let identity = LinearBounds::identity(1);
    let pre_activation = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();

    let result = layer
        .propagate_linear_with_bounds(&identity, &pre_activation)
        .unwrap();

    // Verify soundness across the full interval.
    for i in 0..=100 {
        let t = i as f32 / 100.0;
        let x = (l + t * (u - l)).clamp(l, u);
        let fx = hardswish_eval(x);
        let lower = result.lower_a[[0, 0]] * x + result.lower_b[0];
        let upper = result.upper_a[[0, 0]] * x + result.upper_b[0];
        assert!(
            lower <= fx + 1e-4,
            "HardSwish CROWN lower violated at x={x}: lower={lower} > f(x)={fx}"
        );
        assert!(
            upper >= fx - 1e-4,
            "HardSwish CROWN upper violated at x={x}: upper={upper} < f(x)={fx}"
        );
    }
}

// =============================================================================
// REGRESSION: LeakyReLU/PReLU alpha > 1 (#1803, P264 finding, W1 794 fix)
//
// For alpha > 1, the piecewise kink is concave (slope decreases: alpha → 1).
// Original code assumed convex kink (alpha < 1), causing:
// 1. Infinite-bound lower envelope unsound: min(alpha,1)*x = x > alpha*x for x<0
// 2. Crossing-case chord used as upper bound, but for concave kink chord is lower
//
// Fixed in W1 794:
// - Infinite: lower = (1, (alpha-1)*l) for [l,+inf), (-inf) for others
// - Crossing: swap chord/tangent roles vs alpha <= 1 case
// Proptests above now cover alpha in (0.01, 3.0).
// =============================================================================

/// Regression: PReLU alpha=2.0 on [-1, +inf). Verifies W1 794 fix.
/// Before fix: lower(x) = min(2,1)*(-1) + 0 = -1, but f(-1) = 2*(-1) = -2. Unsound.
/// After fix: lower(x) = 1*(-1) + (2-1)*(-1) = -2 = f(-1). Sound.
#[ntest::timeout(10000)]
#[test]
fn regression_prelu_alpha_gt_one_infinite_upper() {
    // Updated for #2977: domain_guard rejects non-finite pre-activation.
    let alpha = 2.0_f32;
    let layer = PReluLayer::from_scalar(alpha);
    let identity = LinearBounds::identity(1);
    let pre_activation =
        BoundedTensor::new_unchecked(arr1(&[-1.0]).into_dyn(), arr1(&[f32::INFINITY]).into_dyn())
            .unwrap();

    let result = layer.propagate_linear_with_bounds(&identity, &pre_activation);
    assert!(
        matches!(result, Err(NyError::NumericalInstability(_))),
        "PReLU with u=+inf should trigger domain_guard: got {:?}",
        result
    );
}

/// Regression: LeakyReLU alpha=1.5 crossing [-3, 5]. Verifies concave-kink swap.
/// Before fix: chord used as upper, but chord lies BELOW concave kink at origin.
/// After fix: chord is lower, tangent (y=alpha*x or y=x) is upper.
#[ntest::timeout(10000)]
#[test]
fn regression_leaky_relu_alpha_gt_one_crossing() {
    let alpha = 1.5_f32;
    let l = -3.0_f32;
    let u = 5.0_f32;
    let layer = LeakyReLULayer::new(alpha);
    let identity = LinearBounds::identity(1);
    let pre_activation = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();

    let result = layer
        .propagate_linear_with_bounds(&identity, &pre_activation)
        .unwrap();

    // Verify at dense sample including the kink at x=0
    for &x in &[
        -3.0_f32, -2.0, -1.0, -0.5, -0.01, 0.0, 0.01, 0.5, 1.0, 3.0, 5.0,
    ] {
        let fx = if x >= 0.0 { x } else { alpha * x };
        let lower = result.lower_a[[0, 0]] * x + result.lower_b[0];
        let upper = result.upper_a[[0, 0]] * x + result.upper_b[0];
        assert!(
            lower <= fx + 1e-4,
            "LeakyReLU(alpha=1.5) lower unsound at x={x}: lower={lower} > f(x)={fx}"
        );
        assert!(
            upper >= fx - 1e-4,
            "LeakyReLU(alpha=1.5) upper unsound at x={x}: upper={upper} < f(x)={fx}"
        );
    }
}
