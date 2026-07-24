// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Softplus activation layer for bound propagation.
//!
//! Softplus is a smooth approximation to ReLU. It's monotonically increasing
//! and convex, which allows for tight linear relaxations using chord (upper)
//! and tangent (lower) bounds.

use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use std::borrow::Cow;
use tracing::debug;

use super::super::common::{
    crown_elementwise_backward, crown_elementwise_backward_batched,
    crown_elementwise_backward_patches, ibp_bound_interval_parallel, non_finite_domain_guard,
    BoundPropagation,
};
use super::s_shaped::sigmoid_f64;
use crate::layers::activations::LinearRelaxation;
use crate::{BatchedLinearBounds, LinearBounds};

/// Softplus activation: y = ln(1 + exp(x))
///
/// Smooth approximation to ReLU. Monotonically increasing with range (0, +∞).
/// Properties:
/// - softplus(0) ≈ ln(2) ≈ 0.693
/// - For large x: softplus(x) ≈ x
/// - For large negative x: softplus(x) ≈ 0
/// - Derivative: sigmoid(x)
#[derive(Debug, Clone, Default)]
pub struct SoftplusLayer;

impl SoftplusLayer {
    /// Create a new Softplus layer.
    pub fn new() -> Self {
        Self
    }
}

#[cfg(test)]
fn softplus(x: f32) -> f32 {
    // Stable formulation for all x.
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Compute softplus bound interval for [l, u] with directed rounding.
/// Since softplus is monotonically increasing: softplus(l) <= softplus(x) <= softplus(u).
/// Computes in f64 and applies next_down/next_up to guarantee conservative f32 bounds.
/// Part of #3245.
fn softplus_bound_interval(l: f32, u: f32) -> (f32, f32) {
    // Range clamp: softplus(x) > 0 for all real x. Directed rounding can push
    // past zero for extreme negative inputs (e.g., softplus(-1000) → 0 → -1e-45). (#3316)
    (
        next_down_f32(softplus_f64(l as f64) as f32).max(0.0),
        next_up_f32(softplus_f64(u as f64) as f32),
    )
}

fn softplus_f64(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

const SOFTPLUS_RELAX_EPS: f32 = 1e-6;

fn softplus_finalize(
    l: f32,
    u: f32,
    lower_slope: f64,
    lower_intercept: f64,
    upper_slope: f64,
    upper_intercept: f64,
) -> (f32, f32, f32, f32) {
    let max_abs_x = l.abs().max(u.abs()) as f64;
    if !max_abs_x.is_finite() {
        // Drive bounds to [-inf, +inf]: sound and correctly handled by concretize_sound.
        // Previous f32::MAX can overflow to Inf during CROWN backward multiplication,
        // producing NaN in A-matrix accumulation.
        return (0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY);
    }
    let lower_slope_f = lower_slope as f32;
    let upper_slope_f = upper_slope as f32;
    // next_up_f32 on the f64→f32 cast ensures the error bound is >= the true
    // value, preventing intercept widening from underestimating rounding (#2636).
    let lower_err = next_up_f32(((lower_slope - lower_slope_f as f64).abs() * max_abs_x) as f32);
    let upper_err = next_up_f32(((upper_slope - upper_slope_f as f64).abs() * max_abs_x) as f32);

    let lower_intercept_f =
        next_down_f32((lower_intercept as f32) - SOFTPLUS_RELAX_EPS - lower_err);
    let upper_intercept_f = next_up_f32((upper_intercept as f32) + SOFTPLUS_RELAX_EPS + upper_err);

    (
        lower_slope_f,
        lower_intercept_f,
        upper_slope_f,
        upper_intercept_f,
    )
}

/// Linear relaxation for softplus on interval [l, u].
fn softplus_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    // Guard: NaN bounds → return (-inf, +inf) intercepts so CROWN drives bounds to ±inf.
    if l.is_nan() || u.is_nan() {
        return LinearRelaxation::new(0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY);
    }

    // Guard: infinite bounds. Softplus(x) -> 0 as x -> -inf, Softplus(x) -> x as x -> +inf.
    // softplus_finalize catches infinite max_abs_x, but tangent computations at ±inf
    // produce NaN (0 * inf) before reaching finalize. Handle explicitly.
    if l.is_infinite() || u.is_infinite() {
        // Same as finalize fallback: drive to [-inf, +inf] for sound CROWN handling.
        return LinearRelaxation::new(0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY);
    }

    // Handle degenerate cases
    if (u - l).abs() < 1e-8 {
        let l64 = l as f64;
        let slope = sigmoid_f64(l64);
        let intercept = softplus_f64(l64) - slope * l64;
        let (ls, li, us, ui) = softplus_finalize(l, u, slope, intercept, slope, intercept);
        return LinearRelaxation::new(ls, li, us, ui);
    }

    let l64 = l as f64;
    let u64 = u as f64;
    let spl = softplus_f64(l64);
    let spu = softplus_f64(u64);

    // Softplus is convex and increasing: chord is an upper bound, tangent is a lower bound.
    let chord_slope = (spu - spl) / (u64 - l64);
    let chord_intercept = spl - chord_slope * l64;

    // Lower bound: tangent to the convex softplus. The tightest single tangent
    // line is the one PARALLEL to the chord — i.e. tangent at the point d where
    // softplus'(d) = sigmoid(d) = chord_slope. This minimizes the worst-case gap
    // between the line and the function (typically 35–75% tighter than the looser
    // endpoint-tangent heuristic). Because softplus is convex, the tangent at ANY
    // point d is a global lower bound, so the choice of d only affects tightness,
    // never soundness. The derivative sigmoid is strictly increasing, so the
    // tangent point is found by a monotone binary search; the slope stays in (0,1)
    // (no magnitude blow-up). This matches the s-shaped (#tanh/sigmoid) tangent-
    // table approach and alpha-beta-CROWN's convex `bound_relax` (tangent at d).
    //
    // Endpoint tangents bracket the parallel point: sigmoid(l) < chord_slope <
    // sigmoid(u) for a strictly convex, non-degenerate interval. If that ordering
    // does not hold (degenerate/near-flat interval where the chord slope rounds to
    // an endpoint derivative), fall back to the better endpoint tangent — always
    // sound, identical to the prior behavior.
    let tangent_l_slope = sigmoid_f64(l64);
    let tangent_l_intercept = spl - tangent_l_slope * l64;
    let tangent_u_slope = sigmoid_f64(u64);
    let tangent_u_intercept = spu - tangent_u_slope * u64;

    let (lower_slope, lower_intercept) =
        if chord_slope > tangent_l_slope && chord_slope < tangent_u_slope {
            // Binary search for d in (l, u) with sigmoid(d) == chord_slope.
            let mut lo = l64;
            let mut hi = u64;
            for _ in 0..60 {
                // Bit-identical: the bracket stays inside the f32-cast [l64, u64].
                let m = f64::midpoint(lo, hi);
                if sigmoid_f64(m) < chord_slope {
                    lo = m;
                } else {
                    hi = m;
                }
            }
            let d = f64::midpoint(lo, hi);
            let d_slope = sigmoid_f64(d);
            (d_slope, softplus_f64(d) - d_slope * d)
        } else {
            // Degenerate ordering: pick the higher of the two endpoint tangents.
            let mid = l64 + 0.5 * (u64 - l64);
            let tangent_l_mid = tangent_l_slope * mid + tangent_l_intercept;
            let tangent_u_mid = tangent_u_slope * mid + tangent_u_intercept;
            if tangent_l_mid >= tangent_u_mid {
                (tangent_l_slope, tangent_l_intercept)
            } else {
                (tangent_u_slope, tangent_u_intercept)
            }
        };

    let (ls, li, us, ui) = softplus_finalize(
        l,
        u,
        lower_slope,
        lower_intercept,
        chord_slope,
        chord_intercept,
    );
    LinearRelaxation::new(ls, li, us, ui)
}

impl BoundPropagation for SoftplusLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        ibp_bound_interval_parallel(input, softplus_bound_interval)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "Softplus is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
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
        SoftplusLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

impl SoftplusLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        non_finite_domain_guard("Softplus", pre_activation)?;
        debug!("Softplus layer CROWN backward propagation with pre-activation bounds");
        crown_elementwise_backward(bounds, pre_activation, softplus_linear_relaxation)
    }

    /// Batched CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        non_finite_domain_guard("Softplus", pre_activation)?;
        debug!("Softplus layer batched CROWN backward propagation");
        crown_elementwise_backward_batched(bounds, pre_activation, softplus_linear_relaxation)
    }

    /// Patches CROWN backward propagation with pre-activation bounds.
    /// Part of #2613 Phase 2: generic activation Patches support.
    pub(crate) fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        non_finite_domain_guard("Softplus", pre_activation)?;
        crown_elementwise_backward_patches(bounds, pre_activation, softplus_linear_relaxation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::assert_relaxation_sound;
    use ndarray::arr1;
    use ny_tensor::BoundedTensor;
    use proptest::prelude::*;

    /// Independent f64 softplus reference for strict proptest. (#3292)
    /// softplus(x) = ln(1 + exp(x)), numerically stable formulation.
    fn softplus_f64_reference(x: f64) -> f64 {
        if x > 0.0 {
            x + (-x).exp().ln_1p()
        } else {
            x.exp().ln_1p()
        }
    }

    /// The parallel-to-chord lower tangent must be at least as tight as the
    /// previous endpoint-tangent heuristic at every interior point, and remain
    /// a valid (sound) lower bound. Worst-case-gap improvement validates the
    /// tightening landed.
    #[ntest::timeout(10000)]
    #[test]
    fn test_softplus_lower_tangent_tighter_than_endpoint() {
        // Reproduce the OLD endpoint-tangent lower line for comparison.
        fn old_endpoint_lower(l: f64, u: f64) -> (f64, f64) {
            let spl = softplus_f64(l);
            let spu = softplus_f64(u);
            let tl_s = sigmoid_f64(l);
            let tl_i = spl - tl_s * l;
            let tu_s = sigmoid_f64(u);
            let tu_i = spu - tu_s * u;
            let mid = l + 0.5 * (u - l);
            if tl_s * mid + tl_i >= tu_s * mid + tu_i {
                (tl_s, tl_i)
            } else {
                (tu_s, tu_i)
            }
        }
        for &(l, u) in &[
            (-3.0_f32, 3.0_f32),
            (-1.0, 1.0),
            (0.0, 4.0),
            (-5.0, 0.5),
            (-2.0, 2.0),
            (-8.0, -2.0),
        ] {
            let relax = softplus_linear_relaxation(l, u);
            let (os, oi) = old_endpoint_lower(l as f64, u as f64);
            let mut new_maxgap = 0.0f64;
            let mut old_maxgap = 0.0f64;
            for k in 0..=2000 {
                let x = l as f64 + (u as f64 - l as f64) * (k as f64 / 2000.0);
                let fx = softplus_f64_reference(x);
                // Soundness: new lower line must not exceed softplus anywhere.
                let new_val = relax.lower_slope as f64 * x + relax.lower_intercept as f64;
                assert!(
                    new_val <= fx + 1e-5,
                    "softplus new lower UNSOUND at x={x} in [{l},{u}]: {new_val} > {fx}"
                );
                new_maxgap = new_maxgap.max(fx - new_val);
                old_maxgap = old_maxgap.max(fx - (os * x + oi));
            }
            assert!(
                new_maxgap <= old_maxgap + 1e-4,
                "parallel tangent not tighter for [{l},{u}]: new {new_maxgap} > old {old_maxgap}"
            );
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_softplus_linear_relaxation_sound() {
        let intervals = [(-12.0, -3.0), (-1.0, 1.0), (0.2, 4.5)];
        for (l, u) in intervals {
            let relaxation = softplus_linear_relaxation(l, u);
            assert_relaxation_sound(l, u, relaxation, softplus, 1e-4, "softplus");
        }
    }

    #[test]
    fn test_softplus_non_finite_fallback_returns_infinite_intercepts() {
        // Regression for #2567: non-finite inputs must drive to [-inf, +inf]
        // instead of finite sentinels that can overflow in backward accumulation.
        let (_, li_finalize, _, ui_finalize) =
            softplus_finalize(f32::INFINITY, 1.0, 0.5, 0.0, 0.5, 0.0);
        assert_eq!(li_finalize, f32::NEG_INFINITY);
        assert_eq!(ui_finalize, f32::INFINITY);

        for (l, u) in [
            (f32::NEG_INFINITY, 1.0),
            (-1.0, f32::INFINITY),
            (f32::NAN, 1.0),
            (-1.0, f32::NAN),
        ] {
            let r = softplus_linear_relaxation(l, u);
            assert_eq!(r.lower_slope, 0.0);
            assert_eq!(r.upper_slope, 0.0);
            assert_eq!(r.lower_intercept, f32::NEG_INFINITY);
            assert_eq!(r.upper_intercept, f32::INFINITY);
        }
    }

    // ── CROWN backward tests ───────────────────────────────────────────

    #[test]
    fn test_crown_backward_crossing_soundness() {
        let layer = SoftplusLayer::new();
        let l = -3.0_f32;
        let u = 3.0_f32;
        let pre = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

        let la = result.lower_a[[0, 0]];
        let lb = result.lower_b[0];
        let ua = result.upper_a[[0, 0]];
        let ub = result.upper_b[0];

        for k in 0..=50 {
            let x = l + (u - l) * (k as f32 / 50.0);
            let y = softplus(x);
            assert!(
                la * x + lb <= y + 1e-3,
                "Softplus CROWN lb violated at x={x}: {} > {y}",
                la * x + lb
            );
            assert!(
                ua * x + ub >= y - 1e-3,
                "Softplus CROWN ub violated at x={x}: {} < {y}",
                ua * x + ub
            );
        }
    }

    #[test]
    fn test_crown_backward_positive_region() {
        // In positive region, softplus ≈ x, so slopes should be near 1
        let layer = SoftplusLayer::new();
        let pre =
            BoundedTensor::new(arr1(&[3.0_f32]).into_dyn(), arr1(&[8.0_f32]).into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

        for k in 0..=50 {
            let x = 3.0 + 5.0 * (k as f32 / 50.0);
            let y = softplus(x);
            assert!(
                result.lower_a[[0, 0]] * x + result.lower_b[0] <= y + 1e-3,
                "positive softplus lb violated at x={x}"
            );
            assert!(
                result.upper_a[[0, 0]] * x + result.upper_b[0] >= y - 1e-3,
                "positive softplus ub violated at x={x}"
            );
        }
    }

    #[test]
    fn test_crown_backward_negative_region() {
        // In negative region, softplus ≈ 0
        let layer = SoftplusLayer::new();
        let pre =
            BoundedTensor::new(arr1(&[-8.0_f32]).into_dyn(), arr1(&[-2.0_f32]).into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

        for k in 0..=50 {
            let x = -8.0 + 6.0 * (k as f32 / 50.0);
            let y = softplus(x);
            assert!(
                result.lower_a[[0, 0]] * x + result.lower_b[0] <= y + 1e-3,
                "negative softplus lb violated at x={x}"
            );
            assert!(
                result.upper_a[[0, 0]] * x + result.upper_b[0] >= y - 1e-3,
                "negative softplus ub violated at x={x}"
            );
        }
    }

    #[test]
    fn test_crown_backward_multi_neuron() {
        let layer = SoftplusLayer::new();
        let pre = BoundedTensor::new(
            arr1(&[-3.0_f32, 1.0]).into_dyn(),
            arr1(&[1.0_f32, 5.0]).into_dyn(),
        )
        .unwrap();
        let bounds = LinearBounds::identity(2);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

        for neuron in 0..2 {
            let la = result.lower_a[[neuron, neuron]];
            let lb = result.lower_b[neuron];
            let ua = result.upper_a[[neuron, neuron]];
            let ub = result.upper_b[neuron];
            let lo = pre.lower()[neuron];
            let hi = pre.upper()[neuron];

            for k in 0..=20 {
                let x = lo + (hi - lo) * (k as f32 / 20.0);
                let y = softplus(x);
                assert!(
                    la * x + lb <= y + 1e-3,
                    "neuron {neuron} lb violated at x={x}"
                );
                assert!(
                    ua * x + ub >= y - 1e-3,
                    "neuron {neuron} ub violated at x={x}"
                );
            }
        }
    }

    /// Regression test for #2567, updated for #3070: CROWN backward through
    /// Softplus with non-finite pre-activation bounds must be rejected by
    /// `non_finite_domain_guard` with NumericalInstability error.
    ///
    /// Original bug (#2567): softplus_finalize returned `f32::MAX` as upper
    /// intercept for non-finite inputs, causing NaN via `Inf + (-Inf)`.
    /// Fix (#3070): `non_finite_domain_guard` now rejects non-finite bounds
    /// at the CROWN backward entry point, preventing any relaxation computation
    /// on corrupted inputs.
    #[test]
    fn test_crown_backward_non_finite_preact_no_nan_regression_2567() {
        let layer = SoftplusLayer::new();

        // All non-finite pre-activation cases must be rejected by domain guard.
        let non_finite_cases: &[(f32, f32, &str)] = &[
            (f32::NEG_INFINITY, 1.0, "lower=-inf"),
            (-1.0, f32::INFINITY, "upper=+inf"),
            (f32::NEG_INFINITY, f32::INFINITY, "both inf"),
            (f32::NAN, 1.0, "lower=NaN"),
            (-1.0, f32::NAN, "upper=NaN"),
        ];

        for &(l, u, label) in non_finite_cases {
            let pre =
                BoundedTensor::new_unchecked(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();
            let bounds = LinearBounds::new(
                ndarray::Array2::from_elem((1, 1), 2.5_f32),
                ndarray::Array1::from_elem(1, -1.0_f32),
                ndarray::Array2::from_elem((1, 1), 3.0_f32),
                ndarray::Array1::from_elem(1, 1.0_f32),
            )
            .unwrap();

            let result = layer.propagate_linear_with_bounds(&bounds, &pre);
            assert!(
                result.is_err(),
                "{label}: non-finite pre-activation should be rejected by domain guard"
            );
            let err_msg = format!("{}", result.unwrap_err());
            assert!(
                err_msg.contains("non-finite") || err_msg.contains("NumericalInstability"),
                "{label}: error should be NumericalInstability, got: {err_msg}"
            );
        }
    }

    #[test]
    fn test_propagate_linear_requires_preact() {
        let layer = SoftplusLayer::new();
        let bounds = LinearBounds::identity(1);
        assert!(
            layer.propagate_linear(&bounds).is_err(),
            "Softplus CROWN without pre-activation bounds should fail"
        );
        assert!(layer.requires_pre_activation_bounds());
    }

    // ── Strict zero-tolerance CROWN relaxation proptest (#3292) ──────────
    //
    // Pattern from #3285: f64-evaluated reference with zero tolerance catches
    // f32 cancellation bugs invisible to magnitude-scaled tolerance tests.

    proptest! {
        #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

        /// Strict soundness proptest for softplus CROWN relaxation.
        /// Uses f64 reference (softplus_f64_reference) with zero tolerance on 200-point grid.
        /// Ref: alpha-beta-CROWN auto_LiRPA softplus relaxation, #3292.
        #[test]
        fn proptest_softplus_relaxation_strict_soundness(
            l in -10.0f32..10.0,
            width in 0.01f32..20.0,
        ) {
            let u = l + width;
            let relax = softplus_linear_relaxation(l, u);
            let ls = relax.lower_slope;
            let li = relax.lower_intercept;
            let us = relax.upper_slope;
            let ui = relax.upper_intercept;

            prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

            for k in 0..=200 {
                let t = k as f64 / 200.0;
                let x = l as f64 + t * (u as f64 - l as f64);
                let x = x.clamp(l as f64, u as f64);
                let fx = softplus_f64_reference(x);

                let lower_val = ls as f64 * x + li as f64;
                prop_assert!(
                    lower_val <= fx,
                    "softplus lower bound UNSOUND at x={x}: {lower_val} > softplus({x})={fx}, \
                     interval=[{l}, {u}], gap={}", lower_val - fx
                );

                let upper_val = us as f64 * x + ui as f64;
                prop_assert!(
                    upper_val >= fx,
                    "softplus upper bound UNSOUND at x={x}: {upper_val} < softplus({x})={fx}, \
                     interval=[{l}, {u}], gap={}", fx - upper_val
                );
            }
        }
    }
}
