// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Neuron selection heuristics for branching in branch-and-bound.

mod genbab;
mod graph;
mod graph_coefficients;
pub(in crate::beta_crown::engine) mod kfsb_shared;
mod score;
mod sequential;

use std::sync::Arc;

use ndarray::Array2;
use ny_core::Result;
use ny_tensor::BoundedTensor;
use tracing::{debug, trace};

use crate::beta_crown::bab_cuts::CutPool;
use crate::beta_crown::branching::{BranchingHeuristic, NeuronConstraint};
use crate::beta_crown::config::KfsbReduceOp;
use crate::beta_crown::domain::{
    BabDomain, GraphBabDomain, MultiObjectiveGraphBabDomain, NodeBoundsView, ObjectiveAggregation,
};
use crate::beta_crown::nonlinear_branching::{BranchingDecision, NonlinearBranching};
use crate::beta_crown::state::{BetaState, DomainAlphaState};
use crate::layers::activations::RELU_RELAX_MIN_WIDTH;
use crate::{GraphNetwork, Layer, Network, NETWORK_INPUT};

use self::score::{compute_babsr_intercept_only_score, compute_babsr_score_parts, BabsrScoreParts};
use super::tensor_ext::BoundedTensorExt;
use super::BetaCrownVerifier;

const RELU_INTERCEPT_MIN_WIDTH: f32 = 1e-6;

/// Returns true if the layer is a zero-threshold binary activation (ReLU or Sign).
///
/// Both ReLU and Sign split at x=0 with the same child-domain half-space semantics:
/// - active branch: constrain pre-activation to [0, u]
/// - inactive branch: constrain pre-activation to [l, 0]
///
/// Part of #3769: enable BaB branching on Sign neurons.
#[inline]
fn is_zero_threshold_binary_activation(layer: &Layer) -> bool {
    matches!(layer, Layer::ReLU(_) | Layer::Sign(_))
}

/// Compute a fixed CROWN proxy slope for Sign neurons, used by BaBSR/kFSB
/// coefficient scoring.
///
/// For boundary cases [0, u] and [l, 0], returns the Sign CROWN relaxation
/// slope from `sign_crown_relaxation`. For the fully unstable [l, u] (l<0<u)
/// case, returns 0.0 because the fixed-CROWN Sign relaxation is constant
/// [-1, 1] there (non-zero slopes belong to a future BoundSignMerge alpha packet).
///
/// Reference: piecewise_constant.rs `sign_crown_relaxation`.
/// Part of #3769.
fn sign_fixed_crown_proxy_slope(lower: f32, upper: f32) -> f32 {
    if !lower.is_finite() || !upper.is_finite() {
        return 0.0;
    }
    if lower == 0.0 && upper > 0.0 {
        1.0 / upper.max(1e-6)
    } else if lower < 0.0 && upper == 0.0 {
        -1.0 / lower.min(-1e-6)
    } else {
        0.0
    }
}

fn relu_intercept_score(lower: f32, upper: f32) -> f32 {
    if !lower.is_finite() || !upper.is_finite() {
        tracing::warn!(
            "NaN/Inf bounds in relu_intercept_score, returning 0.0: lower={lower}, upper={upper}"
        );
        return 0.0;
    }
    let width = upper - lower;
    if width.abs() <= RELU_INTERCEPT_MIN_WIDTH {
        return 0.0;
    }
    let intercept = (-lower * upper) / width;
    if intercept.is_finite() {
        intercept
    } else {
        0.0
    }
}

#[inline]
fn relu_upper_slope(lower: f32, upper: f32) -> f32 {
    if !lower.is_finite() || !upper.is_finite() {
        // NaN or Inf bounds: return conservative slope 1.0 (identity passthrough,
        // i.e. no tightening). Without this guard, NaN lower enters the crossing
        // branch because NaN >= 0.0 is false, and Rust's f32::max treats
        // NaN.max(1e-8) = 1e-8, producing upper/1e-8 which can be ~1e8.
        tracing::warn!(
            "NaN/Inf bounds in relu_upper_slope, returning 1.0 (conservative): lower={lower}, upper={upper}"
        );
        return 1.0;
    }
    if lower >= 0.0 {
        1.0
    } else if upper <= 0.0 {
        0.0
    } else {
        upper / (upper - lower).max(RELU_RELAX_MIN_WIDTH)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        relu_intercept_score, relu_upper_slope, RELU_INTERCEPT_MIN_WIDTH, RELU_RELAX_MIN_WIDTH,
    };

    // ---- relu_intercept_score tests ----

    #[test]
    fn test_relu_intercept_score_typical_unstable_interval() {
        // x in [-1, 2]: intercept = -(-1)*2 / (2 - (-1)) = 2/3
        let score = relu_intercept_score(-1.0, 2.0);
        let expected = 2.0 / 3.0;
        assert!(
            (score - expected).abs() < 1e-6,
            "expected {expected}, got {score}"
        );
    }

    #[test]
    fn test_relu_intercept_score_nan_lower_returns_zero() {
        assert_eq!(relu_intercept_score(f32::NAN, 1.0), 0.0);
    }

    #[test]
    fn test_relu_intercept_score_nan_upper_returns_zero() {
        assert_eq!(relu_intercept_score(-1.0, f32::NAN), 0.0);
    }

    #[test]
    fn test_relu_intercept_score_inf_lower_returns_zero() {
        assert_eq!(relu_intercept_score(f32::NEG_INFINITY, 1.0), 0.0);
    }

    #[test]
    fn test_relu_intercept_score_inf_upper_returns_zero() {
        assert_eq!(relu_intercept_score(-1.0, f32::INFINITY), 0.0);
    }

    #[test]
    fn test_relu_intercept_score_near_zero_width_returns_zero() {
        // Width below RELU_INTERCEPT_MIN_WIDTH (1e-6) should return 0.0
        let l = -5e-7_f32;
        let u = 4e-7_f32;
        assert!((u - l).abs() <= RELU_INTERCEPT_MIN_WIDTH);
        assert_eq!(relu_intercept_score(l, u), 0.0);
    }

    #[test]
    fn test_relu_intercept_score_exactly_at_min_width_returns_zero() {
        // Exactly at the boundary: width == RELU_INTERCEPT_MIN_WIDTH
        let l = -5e-7_f32;
        let u = 5e-7_f32;
        // width = 1e-6 == RELU_INTERCEPT_MIN_WIDTH, so abs() <= guard triggers
        assert_eq!(relu_intercept_score(l, u), 0.0);
    }

    #[test]
    fn test_relu_intercept_score_stable_positive_returns_expected_value() {
        // Both bounds positive: l=1.0, u=2.0. intercept = -l*u/width = -1*2/1 = -2.
        // The function returns the intercept regardless of stability; the caller
        // uses this to rank neurons for branching.
        let score = relu_intercept_score(1.0, 2.0);
        assert!(
            (score - (-2.0)).abs() < 1e-6,
            "expected -2.0 for stable positive [1,2], got {score}"
        );
    }

    #[test]
    fn test_relu_intercept_score_large_bounds_correct_value() {
        // Large but finite bounds: intercept = -(-1e18)*1e18 / (2e18) = 5e17
        let score = relu_intercept_score(-1e18, 1e18);
        let expected = 5e17_f32;
        assert!(
            score.is_finite(),
            "Large bounds produced non-finite score: {score}"
        );
        assert!(
            (score - expected).abs() / expected < 1e-5,
            "expected ~{expected} for [-1e18, 1e18], got {score}"
        );
    }

    // ---- relu_upper_slope tests ----

    #[test]
    fn test_relu_upper_slope_matches_legacy_formula_for_typical_widths() {
        let cases = [
            (-1.0_f32, 2.0_f32),
            (-0.5_f32, 0.75_f32),
            (-2.0_f32, 3.0_f32),
        ];
        for (l, u) in cases {
            assert!(l < 0.0 && u > 0.0);
            assert!(u - l >= RELU_RELAX_MIN_WIDTH);
            let legacy = u / (u - l);
            assert_eq!(relu_upper_slope(l, u), legacy);
        }
    }

    #[test]
    fn test_relu_upper_slope_uses_guard_for_tiny_unstable_interval() {
        let l = -5e-9_f32;
        let u = 4e-9_f32;
        assert!(l < 0.0 && u > 0.0);
        assert!(u - l < RELU_RELAX_MIN_WIDTH);

        let guarded = relu_upper_slope(l, u);
        let expected_guarded = u / RELU_RELAX_MIN_WIDTH;
        let legacy = u / (u - l);

        assert!(guarded.is_finite());
        assert_eq!(guarded, expected_guarded);
        assert!(legacy > guarded);
    }

    #[test]
    fn test_relu_upper_slope_stable_positive_returns_one() {
        // Fully positive intervals: ReLU is identity, slope must be 1.0.
        assert_eq!(relu_upper_slope(0.0, 1.0), 1.0);
        assert_eq!(relu_upper_slope(0.5, 2.0), 1.0);
        assert_eq!(relu_upper_slope(1e-38, 1e38), 1.0);
    }

    #[test]
    fn test_relu_upper_slope_stable_negative_returns_zero() {
        // Fully negative intervals: ReLU is zero, slope must be 0.0.
        assert_eq!(relu_upper_slope(-2.0, 0.0), 0.0);
        assert_eq!(relu_upper_slope(-3.0, -1.0), 0.0);
        assert_eq!(relu_upper_slope(-1e38, -1e-38), 0.0);
    }

    #[test]
    fn test_relu_upper_slope_nan_lower_returns_conservative() {
        // NaN lower should return 1.0 (conservative identity passthrough),
        // not the wildly inflated value from the crossing branch.
        assert_eq!(relu_upper_slope(f32::NAN, 1.0), 1.0);
    }

    #[test]
    fn test_relu_upper_slope_nan_upper_returns_conservative() {
        assert_eq!(relu_upper_slope(-1.0, f32::NAN), 1.0);
    }

    #[test]
    fn test_relu_upper_slope_inf_returns_conservative() {
        assert_eq!(relu_upper_slope(f32::NEG_INFINITY, 1.0), 1.0);
        assert_eq!(relu_upper_slope(-1.0, f32::INFINITY), 1.0);
    }

    #[test]
    fn test_relu_upper_slope_crossing_is_in_zero_one() {
        // For any crossing neuron (l < 0 < u), slope must be in (0, 1].
        let cases = [
            (-1.0_f32, 2.0_f32),
            (-0.001, 0.001),
            (-100.0, 0.001),
            (-0.001, 100.0),
            (-1e-20, 1e-20),
        ];
        for (l, u) in cases {
            let slope = relu_upper_slope(l, u);
            assert!(
                slope > 0.0 && slope <= 1.0,
                "relu_upper_slope({l}, {u}) = {slope}, expected in (0, 1]"
            );
        }
    }
}
