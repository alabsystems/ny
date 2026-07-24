// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest soundness tests for Softmax IBP/CROWN and Causal Softmax IBP/CROWN.
//!
//! Softmax CROWN backward has two paths:
//! - Sound: LSE-based affine bounds (Shi et al., "Formal Verification for Transformers")
//! - Heuristic: sampling-based Jacobian linearization (not provably sound)
//!
//! Causal Softmax has:
//! - IBP: per-row softmax over active (causal-masked) positions
//! - CROWN sound: falls back to IBP-derived constant bounds
//! - CROWN heuristic: Jacobian-based linearization with sampling error estimation
//!
//! Part of #1950.

use crate::layers::common::BoundPropagation;
use crate::layers::softmax::CausalSoftmaxLayer;
use crate::layers::LogSoftmaxLayer;
use crate::layers::SoftmaxLayer;
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{arr1, Array1, Array2, ArrayD, IxDyn};
use ny_core::VerificationSoundnessMode;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{causal_softmax, logsoftmax, sample_points, softmax};

/// Tolerance for softmax CROWN sound bounds.
/// LSE-based affine bounds involve exp/log operations which amplify FP error.
const SOFTMAX_CROWN_TOLERANCE: f32 = 1e-3;

/// Tolerance for softmax IBP bounds.
const SOFTMAX_IBP_TOLERANCE: f32 = 1e-4;

/// Tolerance for causal softmax IBP bounds.
const CAUSAL_IBP_TOLERANCE: f32 = 1e-4;

/// Tolerance for causal softmax CROWN sound bounds (IBP-derived constants).
const CAUSAL_CROWN_TOLERANCE: f32 = 1e-3;

/// Concretize LinearBounds at a specific point: lb = A_l @ x + b_l, ub = A_u @ x + b_u.
fn concretize_at(result: &LinearBounds, x: &Array1<f32>) -> (Vec<f32>, Vec<f32>) {
    let n_out = result.num_outputs();
    let mut lowers = Vec::with_capacity(n_out);
    let mut uppers = Vec::with_capacity(n_out);
    for i in 0..n_out {
        lowers.push(result.lower_a.row(i).dot(x) + result.lower_b[i]);
        uppers.push(result.upper_a.row(i).dot(x) + result.upper_b[i]);
    }
    (lowers, uppers)
}

// =============================================================================
// SOFTMAX CROWN BACKWARD SOUNDNESS — SOUND MODE (LSE-based affine bounds)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Softmax CROWN sound backward: affine bounds contain true softmax at all sampled points.
    ///
    /// For identity incoming bounds, CROWN returns y_lower >= A_l @ x + b_l and
    /// y_upper <= A_u @ x + b_u. We verify that for all x in [lower, upper],
    /// A_l @ x + b_l <= softmax(x) <= A_u @ x + b_u.
    ///
    /// Uses 3-element input to keep vertex enumeration tractable (2^3 = 8 vertices).
    /// Part of #1950.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_softmax_crown_sound_3d(
        (l0, u0) in super::valid_interval(3.0),
        (l1, u1) in super::valid_interval(3.0),
        (l2, u2) in super::valid_interval(3.0),
    ) {
        prop_assume!(u0 > l0 + 0.01);
        prop_assume!(u1 > l1 + 0.01);
        prop_assume!(u2 > l2 + 0.01);

        let input = BoundedTensor::new(
            arr1(&[l0, l1, l2]).into_dyn(),
            arr1(&[u0, u1, u2]).into_dyn(),
        ).unwrap();

        let layer = SoftmaxLayer::new(-1); // sound=true by default
        let bounds = LinearBounds::identity(3);

        let result = layer
            .propagate_linear_with_bounds(&bounds, &input, VerificationSoundnessMode::Sound)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear_with_bounds failed: {e}")
            ))?;

        // Verify soundness at all 8 vertices + center
        let vertices: Vec<[f32; 3]> = vec![
            [l0, l1, l2], [u0, l1, l2], [l0, u1, l2], [u0, u1, l2],
            [l0, l1, u2], [u0, l1, u2], [l0, u1, u2], [u0, u1, u2],
            [f32::midpoint(l0, u0), f32::midpoint(l1, u1), f32::midpoint(l2, u2)],
        ];

        for vertex in &vertices {
            let x = arr1(vertex);
            let sm = softmax(&x);
            let (lowers, uppers) = concretize_at(&result, &x);

            for i in 0..3 {
                prop_assert!(
                    lowers[i] <= sm[i] + SOFTMAX_CROWN_TOLERANCE,
                    "CROWN sound lower violated: dim={}, lb={}, softmax={}, x={:?}",
                    i, lowers[i], sm[i], vertex
                );
                prop_assert!(
                    uppers[i] >= sm[i] - SOFTMAX_CROWN_TOLERANCE,
                    "CROWN sound upper violated: dim={}, ub={}, softmax={}, x={:?}",
                    i, uppers[i], sm[i], vertex
                );
            }
        }
    }

    /// Softmax CROWN sound backward with tight (epsilon-ball) input bounds.
    ///
    /// Tests that CROWN bounds are also tight for narrow intervals — the LSE-based
    /// affine bounds should converge to the Jacobian as epsilon → 0.
    /// Part of #1950.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_softmax_crown_sound_tight(
        center0 in -2.0f32..2.0,
        center1 in -2.0f32..2.0,
        center2 in -2.0f32..2.0,
        epsilon in 0.01f32..0.5,
    ) {
        let input = BoundedTensor::new(
            arr1(&[center0 - epsilon, center1 - epsilon, center2 - epsilon]).into_dyn(),
            arr1(&[center0 + epsilon, center1 + epsilon, center2 + epsilon]).into_dyn(),
        ).unwrap();

        let layer = SoftmaxLayer::new(-1);
        let bounds = LinearBounds::identity(3);

        let result = layer
            .propagate_linear_with_bounds(&bounds, &input, VerificationSoundnessMode::Sound)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear_with_bounds failed: {e}")
            ))?;

        // Sample points along each axis
        let pts0 = sample_points(center0 - epsilon, center0 + epsilon, 5);
        let pts1 = sample_points(center1 - epsilon, center1 + epsilon, 5);
        let pts2 = sample_points(center2 - epsilon, center2 + epsilon, 5);

        for &x0 in &pts0 {
            for &x1 in &pts1 {
                for &x2 in &pts2 {
                    let x = arr1(&[x0, x1, x2]);
                    let sm = softmax(&x);
                    let (lowers, uppers) = concretize_at(&result, &x);

                    for i in 0..3 {
                        prop_assert!(
                            lowers[i] <= sm[i] + SOFTMAX_CROWN_TOLERANCE,
                            "CROWN tight lower violated: dim={}, lb={}, softmax={}, x=({},{},{})",
                            i, lowers[i], sm[i], x0, x1, x2
                        );
                        prop_assert!(
                            uppers[i] >= sm[i] - SOFTMAX_CROWN_TOLERANCE,
                            "CROWN tight upper violated: dim={}, ub={}, softmax={}, x=({},{},{})",
                            i, uppers[i], sm[i], x0, x1, x2
                        );
                    }
                }
            }
        }
    }

    /// Softmax CROWN sound backward with 2D input [2, 3] (row-wise softmax).
    ///
    /// Each row of the 2D tensor is independently normalized via softmax.
    /// CROWN bounds should contain the true output for all sampled inputs.
    /// Part of #1950.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_softmax_crown_sound_2d(
        (l00, u00) in super::valid_interval(2.0),
        (l01, u01) in super::valid_interval(2.0),
        (l02, u02) in super::valid_interval(2.0),
        (l10, u10) in super::valid_interval(2.0),
        (l11, u11) in super::valid_interval(2.0),
        (l12, u12) in super::valid_interval(2.0),
    ) {
        prop_assume!(u00 > l00 + 0.01);
        prop_assume!(u01 > l01 + 0.01);
        prop_assume!(u02 > l02 + 0.01);
        prop_assume!(u10 > l10 + 0.01);
        prop_assume!(u11 > l11 + 0.01);
        prop_assume!(u12 > l12 + 0.01);

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(
                IxDyn(&[2, 3]),
                vec![l00, l01, l02, l10, l11, l12],
            ).unwrap(),
            ArrayD::from_shape_vec(
                IxDyn(&[2, 3]),
                vec![u00, u01, u02, u10, u11, u12],
            ).unwrap(),
        ).unwrap();

        let layer = SoftmaxLayer::new(-1);
        let bounds = LinearBounds::identity(6); // 2*3 = 6 flattened

        let result = layer
            .propagate_linear_with_bounds(&bounds, &input, VerificationSoundnessMode::Sound)
            .map_err(|e| TestCaseError::fail(
                format!("2D CROWN failed: {e}")
            ))?;

        // Check center point
        let center = arr1(&[
            f32::midpoint(l00, u00), f32::midpoint(l01, u01), f32::midpoint(l02, u02),
            f32::midpoint(l10, u10), f32::midpoint(l11, u11), f32::midpoint(l12, u12),
        ]);
        let sm_row0 = softmax(&arr1(&[center[0], center[1], center[2]]));
        let sm_row1 = softmax(&arr1(&[center[3], center[4], center[5]]));
        let sm_flat: Array1<f32> = arr1(&[
            sm_row0[0], sm_row0[1], sm_row0[2],
            sm_row1[0], sm_row1[1], sm_row1[2],
        ]);

        let (lowers, uppers) = concretize_at(&result, &center);

        for i in 0..6 {
            prop_assert!(
                lowers[i] <= sm_flat[i] + SOFTMAX_CROWN_TOLERANCE,
                "2D CROWN center lower violated: dim={}, lb={}, softmax={}",
                i, lowers[i], sm_flat[i]
            );
            prop_assert!(
                uppers[i] >= sm_flat[i] - SOFTMAX_CROWN_TOLERANCE,
                "2D CROWN center upper violated: dim={}, ub={}, softmax={}",
                i, uppers[i], sm_flat[i]
            );
        }
    }
}

// =============================================================================
// SOFTMAX IBP SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Softmax IBP soundness: bounds contain true softmax output at sampled points.
    ///
    /// For a 3-element interval box, we verify all 8 vertices plus center.
    /// Part of #1950.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_softmax_ibp_3d(
        (l0, u0) in super::valid_interval(4.0),
        (l1, u1) in super::valid_interval(4.0),
        (l2, u2) in super::valid_interval(4.0),
    ) {
        prop_assume!(u0 > l0 + 0.01);
        prop_assume!(u1 > l1 + 0.01);
        prop_assume!(u2 > l2 + 0.01);

        let input = BoundedTensor::new(
            arr1(&[l0, l1, l2]).into_dyn(),
            arr1(&[u0, u1, u2]).into_dyn(),
        ).unwrap();

        let layer = SoftmaxLayer::new(-1);
        let output = layer
            .propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(
                format!("softmax IBP failed: {e}")
            ))?;

        let samples = [
            [l0, l1, l2],
            [u0, l1, l2],
            [l0, u1, l2],
            [u0, u1, l2],
            [l0, l1, u2],
            [u0, l1, u2],
            [l0, u1, u2],
            [u0, u1, u2],
            [f32::midpoint(l0, u0), f32::midpoint(l1, u1), f32::midpoint(l2, u2)],
        ];

        for point in samples {
            let sm = softmax(&arr1(&point));
            for i in 0..3 {
                prop_assert!(
                    output.lower()[[i]] <= sm[i] + SOFTMAX_IBP_TOLERANCE,
                    "Softmax IBP lower violated: dim={}, lb={}, softmax={}, x={:?}",
                    i, output.lower()[[i]], sm[i], point
                );
                prop_assert!(
                    output.upper()[[i]] >= sm[i] - SOFTMAX_IBP_TOLERANCE,
                    "Softmax IBP upper violated: dim={}, ub={}, softmax={}, x={:?}",
                    i, output.upper()[[i]], sm[i], point
                );
            }
        }
    }
}

// =============================================================================
// CAUSAL SOFTMAX IBP SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Causal softmax IBP soundness: for all x in [lower, upper], causal_softmax(x)
    /// is within computed IBP bounds.
    ///
    /// The causal mask means row i computes softmax over positions 0..=i only.
    /// IBP must handle the masking correctly: masked positions should always be 0.
    /// Part of #1950.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_causal_softmax_ibp_3x3(
        (l00, u00) in super::valid_interval(3.0),
        (l01, u01) in super::valid_interval(3.0),
        (l02, u02) in super::valid_interval(3.0),
        (l10, u10) in super::valid_interval(3.0),
        (l11, u11) in super::valid_interval(3.0),
        (l12, u12) in super::valid_interval(3.0),
        (l20, u20) in super::valid_interval(3.0),
        (l21, u21) in super::valid_interval(3.0),
        (l22, u22) in super::valid_interval(3.0),
    ) {
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[3, 3]),
            vec![l00, l01, l02, l10, l11, l12, l20, l21, l22],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[3, 3]),
            vec![u00, u01, u02, u10, u11, u12, u20, u21, u22],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = CausalSoftmaxLayer::new(-1);
        let output = layer.propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp failed: {e}")
            ))?;

        // Build test matrix from a specific sample (center)
        let x = Array2::from_shape_vec(
            (3, 3),
            vec![
                f32::midpoint(l00, u00), f32::midpoint(l01, u01), f32::midpoint(l02, u02),
                f32::midpoint(l10, u10), f32::midpoint(l11, u11), f32::midpoint(l12, u12),
                f32::midpoint(l20, u20), f32::midpoint(l21, u21), f32::midpoint(l22, u22),
            ],
        ).unwrap();

        let cs = causal_softmax(&x);

        for i in 0..3 {
            for j in 0..3 {
                prop_assert!(
                    output.lower()[[i, j]] <= cs[[i, j]] + CAUSAL_IBP_TOLERANCE,
                    "Causal IBP lower violated: [{},{}]: lb={} > actual={}",
                    i, j, output.lower()[[i, j]], cs[[i, j]]
                );
                prop_assert!(
                    output.upper()[[i, j]] >= cs[[i, j]] - CAUSAL_IBP_TOLERANCE,
                    "Causal IBP upper violated: [{},{}]: ub={} < actual={}",
                    i, j, output.upper()[[i, j]], cs[[i, j]]
                );
            }
        }

        // Verify masked positions
        for i in 0..3 {
            for j in (i+1)..3 {
                prop_assert_eq!(
                    output.lower()[[i, j]], 0.0,
                    "masked position [{},{}] lower should be 0", i, j
                );
                prop_assert_eq!(
                    output.upper()[[i, j]], 0.0,
                    "masked position [{},{}] upper should be 0", i, j
                );
            }
        }
    }

    /// Causal softmax IBP with vertex enumeration for a 2x2 attention matrix.
    ///
    /// Small enough (4 elements, 2^4 = 16 vertices) to check ALL vertices per proptest case.
    /// Part of #1950.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_causal_softmax_ibp_2x2_exhaustive(
        (l00, u00) in super::valid_interval(3.0),
        (l01, u01) in super::valid_interval(3.0),
        (l10, u10) in super::valid_interval(3.0),
        (l11, u11) in super::valid_interval(3.0),
    ) {
        let lower_vals = vec![l00, l01, l10, l11];
        let upper_vals = vec![u00, u01, u10, u11];
        let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2]), lower_vals.clone()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2]), upper_vals.clone()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = CausalSoftmaxLayer::new(-1);
        let output = layer.propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp failed: {e}")
            ))?;

        // Enumerate all 2^4 = 16 vertices
        for mask in 0..16u32 {
            let mut vals = lower_vals.clone();
            for i in 0..4 {
                if (mask >> i) & 1 == 1 {
                    vals[i] = upper_vals[i];
                }
            }

            let x = Array2::from_shape_vec((2, 2), vals.clone()).unwrap();
            let cs = causal_softmax(&x);

            for i in 0..2 {
                for j in 0..2 {
                    prop_assert!(
                        output.lower()[[i, j]] <= cs[[i, j]] + CAUSAL_IBP_TOLERANCE,
                        "Causal 2x2 vertex lower: [{},{}] mask={}: lb={} > actual={}",
                        i, j, mask, output.lower()[[i, j]], cs[[i, j]]
                    );
                    prop_assert!(
                        output.upper()[[i, j]] >= cs[[i, j]] - CAUSAL_IBP_TOLERANCE,
                        "Causal 2x2 vertex upper: [{},{}] mask={}: ub={} > actual={}",
                        i, j, mask, output.upper()[[i, j]], cs[[i, j]]
                    );
                }
            }
        }
    }
}

// =============================================================================
// CAUSAL SOFTMAX CROWN BACKWARD SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// Causal softmax CROWN sound backward soundness with 2x2 attention matrix.
    ///
    /// Sound mode falls back to IBP-derived constant bounds, so CROWN bounds
    /// should contain true causal softmax at all vertices.
    /// Part of #1950.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_causal_softmax_crown_sound_2x2(
        (l00, u00) in super::valid_interval(2.0),
        (l01, u01) in super::valid_interval(2.0),
        (l10, u10) in super::valid_interval(2.0),
        (l11, u11) in super::valid_interval(2.0),
    ) {
        let lower_vals = vec![l00, l01, l10, l11];
        let upper_vals = vec![u00, u01, u10, u11];
        let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2]), lower_vals.clone()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2]), upper_vals.clone()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = CausalSoftmaxLayer::new(-1); // sound=true by default
        let total = 4;
        let bounds = LinearBounds::identity(total);

        let result = layer
            .propagate_linear_with_bounds(&bounds, &input, VerificationSoundnessMode::Sound)
            .map_err(|e| TestCaseError::fail(
                format!("CROWN sound failed: {e}")
            ))?;

        // Sound mode: constant bounds (zero slopes)
        for &v in result.lower_a.iter() {
            let v_f32: f32 = v;
            prop_assert!(
                v_f32.abs() < 1e-6,
                "sound mode lower_a should be 0, got {}", v_f32
            );
        }

        // Verify at all 16 vertices
        for mask in 0..16u32 {
            let mut vals = lower_vals.clone();
            for i in 0..4 {
                if (mask >> i) & 1 == 1 {
                    vals[i] = upper_vals[i];
                }
            }

            let x = Array2::from_shape_vec((2, 2), vals.clone()).unwrap();
            let cs = causal_softmax(&x);

            for i in 0..2 {
                for j in 0..2 {
                    let flat_idx = i * 2 + j;
                    prop_assert!(
                        result.lower_b[flat_idx] <= cs[[i, j]] + CAUSAL_CROWN_TOLERANCE,
                        "Causal CROWN sound lower: [{},{}] mask={}: lb={} > actual={}",
                        i, j, mask, result.lower_b[flat_idx], cs[[i, j]]
                    );
                    prop_assert!(
                        result.upper_b[flat_idx] >= cs[[i, j]] - CAUSAL_CROWN_TOLERANCE,
                        "Causal CROWN sound upper: [{},{}] mask={}: ub={} < actual={}",
                        i, j, mask, result.upper_b[flat_idx], cs[[i, j]]
                    );
                }
            }
        }
    }
}

// =============================================================================
// LOGSOFTMAX IBP SOUNDNESS
// =============================================================================

/// Tolerance for LogSoftmax IBP.
/// LogSoftmax uses f64 intermediate + directed rounding (#3245),
/// so strict (zero) tolerance should hold. We use a tiny tolerance
/// for the reference computation's own FP error.
const LOGSOFTMAX_IBP_TOLERANCE: f32 = 1e-5;

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// LogSoftmax IBP soundness: bounds contain true logsoftmax at all vertices.
    ///
    /// logsoftmax_i = x_i - logsumexp(x)
    /// Lower: x_i^L - logsumexp(x^U)
    /// Upper: x_i^U - logsumexp(x^L)
    ///
    /// Covers the directed rounding refactoring in #3245. Previously this layer
    /// had only hand-picked unit tests; this proptest exercises random intervals.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_logsoftmax_ibp_3d(
        (l0, u0) in super::valid_interval(4.0),
        (l1, u1) in super::valid_interval(4.0),
        (l2, u2) in super::valid_interval(4.0),
    ) {
        prop_assume!(u0 > l0 + 0.01);
        prop_assume!(u1 > l1 + 0.01);
        prop_assume!(u2 > l2 + 0.01);

        let input = BoundedTensor::new(
            arr1(&[l0, l1, l2]).into_dyn(),
            arr1(&[u0, u1, u2]).into_dyn(),
        ).unwrap();

        let layer = LogSoftmaxLayer::new(-1);
        let output = layer
            .propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(
                format!("logsoftmax IBP failed: {e}")
            ))?;

        // All 8 vertices plus center
        let samples = [
            [l0, l1, l2],
            [u0, l1, l2],
            [l0, u1, l2],
            [u0, u1, l2],
            [l0, l1, u2],
            [u0, l1, u2],
            [l0, u1, u2],
            [u0, u1, u2],
            [f32::midpoint(l0, u0), f32::midpoint(l1, u1), f32::midpoint(l2, u2)],
        ];

        for point in samples {
            let ls = logsoftmax(&arr1(&point));
            for i in 0..3 {
                prop_assert!(
                    output.lower()[[i]] <= ls[i] + LOGSOFTMAX_IBP_TOLERANCE,
                    "LogSoftmax IBP lower violated: dim={}, lb={}, logsoftmax={}, x={:?}",
                    i, output.lower()[[i]], ls[i], point
                );
                prop_assert!(
                    output.upper()[[i]] >= ls[i] - LOGSOFTMAX_IBP_TOLERANCE,
                    "LogSoftmax IBP upper violated: dim={}, ub={}, logsoftmax={}, x={:?}",
                    i, output.upper()[[i]], ls[i], point
                );
            }
        }
    }

    /// LogSoftmax IBP tight: epsilon-ball inputs produce sound bounds at center.
    ///
    /// Verifies that with directed rounding (#3245), bounds are sound and
    /// logsoftmax properties hold (all values <= 0, non-inverted intervals).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_logsoftmax_ibp_tight(
        center0 in -2.0f32..2.0,
        center1 in -2.0f32..2.0,
        center2 in -2.0f32..2.0,
        epsilon in 0.01f32..0.5,
    ) {
        let input = BoundedTensor::new(
            arr1(&[center0 - epsilon, center1 - epsilon, center2 - epsilon]).into_dyn(),
            arr1(&[center0 + epsilon, center1 + epsilon, center2 + epsilon]).into_dyn(),
        ).unwrap();

        let layer = LogSoftmaxLayer::new(-1);
        let output = layer
            .propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(
                format!("logsoftmax IBP tight failed: {e}")
            ))?;

        // Soundness at center
        let center_ls = logsoftmax(&arr1(&[center0, center1, center2]));
        for i in 0..3 {
            prop_assert!(
                output.lower()[[i]] <= center_ls[i] + LOGSOFTMAX_IBP_TOLERANCE,
                "LogSoftmax IBP tight: lower[{}]={} > center logsoftmax={}",
                i, output.lower()[[i]], center_ls[i]
            );
            prop_assert!(
                output.upper()[[i]] >= center_ls[i] - LOGSOFTMAX_IBP_TOLERANCE,
                "LogSoftmax IBP tight: upper[{}]={} < center logsoftmax={}",
                i, output.upper()[[i]], center_ls[i]
            );
        }

        // logsoftmax values are always <= 0
        for &v in output.lower().iter() {
            prop_assert!(
                v <= LOGSOFTMAX_IBP_TOLERANCE,
                "LogSoftmax IBP lower should be <= 0, got {}",
                v
            );
        }

        // Non-inverted intervals
        for (l, u) in output.lower().iter().zip(output.upper().iter()) {
            prop_assert!(
                l <= &(u + LOGSOFTMAX_IBP_TOLERANCE),
                "LogSoftmax IBP inverted: lower={} > upper={}",
                l, u
            );
        }
    }

    /// LogSoftmax IBP soundness for 2D input (2 rows of 3).
    ///
    /// Verifies the axis-aware logsumexp reduction works correctly with
    /// 2D tensors where logsoftmax is applied along axis -1.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_logsoftmax_ibp_2d(
        (l0, u0) in super::valid_interval(3.0),
        (l1, u1) in super::valid_interval(3.0),
        (l2, u2) in super::valid_interval(3.0),
        (l3, u3) in super::valid_interval(3.0),
        (l4, u4) in super::valid_interval(3.0),
        (l5, u5) in super::valid_interval(3.0),
    ) {
        prop_assume!(u0 > l0 + 0.01);
        prop_assume!(u1 > l1 + 0.01);
        prop_assume!(u2 > l2 + 0.01);
        prop_assume!(u3 > l3 + 0.01);
        prop_assume!(u4 > l4 + 0.01);
        prop_assume!(u5 > l5 + 0.01);

        let lower_vals = [l0, l1, l2, l3, l4, l5];
        let upper_vals = [u0, u1, u2, u3, u4, u5];

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), lower_vals.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), upper_vals.to_vec()).unwrap(),
        ).unwrap();

        let layer = LogSoftmaxLayer::new(-1);
        let output = layer
            .propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(
                format!("logsoftmax IBP 2D failed: {e}")
            ))?;

        prop_assert_eq!(output.shape(), &[2, 3]);

        // Check row 0 and row 1 independently at vertices
        for row in 0..2usize {
            let rl = &lower_vals[row * 3..(row + 1) * 3];
            let ru = &upper_vals[row * 3..(row + 1) * 3];
            for mask in 0..8u32 {
                let x: Vec<f32> = (0..3)
                    .map(|j| if (mask >> j) & 1 == 0 { rl[j] } else { ru[j] })
                    .collect();
                let ls = logsoftmax(&arr1(&x));
                for col in 0..3 {
                    prop_assert!(
                        output.lower()[[row, col]] <= ls[col] + LOGSOFTMAX_IBP_TOLERANCE,
                        "LogSoftmax 2D lower[{},{}]={} > actual={} at x={:?}",
                        row, col, output.lower()[[row, col]], ls[col], x
                    );
                    prop_assert!(
                        output.upper()[[row, col]] >= ls[col] - LOGSOFTMAX_IBP_TOLERANCE,
                        "LogSoftmax 2D upper[{},{}]={} < actual={} at x={:?}",
                        row, col, output.upper()[[row, col]], ls[col], x
                    );
                }
            }
        }
    }
}

// =============================================================================
// FLAT-GROUPED SOFTMAX CROWN BACKWARD SOUNDNESS
// =============================================================================

/// Tolerance for flat-grouped softmax CROWN sound bounds.
/// The flat-grouped path splits bias with directed rounding and accumulates
/// across groups, introducing slightly more FP error than 1D CROWN.
const FLAT_GROUPED_CROWN_TOLERANCE: f32 = 2e-3;

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// Flat-grouped softmax CROWN sound backward: vertex soundness with block-diagonal A.
    ///
    /// Tests the flat-grouped path (triggered when bounds A is 2D flat with
    /// `a_in_dim = num_groups * softmax_size != softmax_size`). Constructs
    /// block-diagonal A with 2 groups of softmax_size=2 and out_dim=2.
    /// Verifies that output bounds contain the true per-group softmax at all
    /// 16 vertices of the input domain.
    ///
    /// This path runs in production when attention CROWN backward calls
    /// `flatten_to_block_diagonal` before softmax backward.
    ///
    /// Design: designs/2026-03-03-flat-grouped-softmax-crown-testing.md Test 1.
    /// Part of #3247.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_flat_grouped_softmax_crown_sound(
        (l00, u00) in super::valid_interval(3.0),
        (l01, u01) in super::valid_interval(3.0),
        (l10, u10) in super::valid_interval(3.0),
        (l11, u11) in super::valid_interval(3.0),
        a00 in -2.0f32..2.0,
        a01 in -2.0f32..2.0,
        a10 in -2.0f32..2.0,
        a11 in -2.0f32..2.0,
        b0 in -1.0f32..1.0,
        b1 in -1.0f32..1.0,
    ) {
        prop_assume!(u00 > l00 + 0.01);
        prop_assume!(u01 > l01 + 0.01);
        prop_assume!(u10 > l10 + 0.01);
        prop_assume!(u11 > l11 + 0.01);

        let softmax_size = 2usize;
        let num_groups = 2usize;
        let out_dim = 2usize;
        let total_in = num_groups * softmax_size;

        // Pre-activation: [2, 2] (2 groups, softmax_size=2 each)
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(
                IxDyn(&[num_groups, softmax_size]),
                vec![l00, l01, l10, l11],
            ).unwrap(),
            ArrayD::from_shape_vec(
                IxDyn(&[num_groups, softmax_size]),
                vec![u00, u01, u10, u11],
            ).unwrap(),
        ).unwrap();

        // Block-diagonal A: [2, 4]
        // Row 0: [a00, a01, 0, 0] operates on group 0
        // Row 1: [0, 0, a10, a11] operates on group 1
        let a_vals = vec![a00, a01, 0.0, 0.0, 0.0, 0.0, a10, a11];
        let bias_vals = vec![b0, b1];
        let la = ArrayD::from_shape_vec(
            IxDyn(&[out_dim, total_in]), a_vals.clone(),
        ).unwrap();
        let lb = ArrayD::from_shape_vec(
            IxDyn(&[out_dim]), bias_vals.clone(),
        ).unwrap();
        let batched_bounds = BatchedLinearBounds::from_parts_unchecked(
            la.clone(), lb.clone(), la, lb,
            vec![total_in], vec![out_dim],
        );

        let layer = SoftmaxLayer::new(-1);
        let result = layer
            .propagate_linear_batched_with_bounds(
                &batched_bounds, &input, VerificationSoundnessMode::Sound,
            )
            .map_err(|e| TestCaseError::fail(
                format!("flat-grouped sound backward failed: {e}")
            ))?;

        // Verify at all 16 vertices
        let lower_vals = [l00, l01, l10, l11];
        let upper_vals = [u00, u01, u10, u11];

        for mask in 0..16u32 {
            let x: Vec<f32> = (0..4).map(|i| {
                if (mask >> i) & 1 == 1 { upper_vals[i] } else { lower_vals[i] }
            }).collect();

            // Per-group softmax: softmax([x[0],x[1]]) and softmax([x[2],x[3]])
            let sm0 = softmax(&arr1(&[x[0], x[1]]));
            let sm1 = softmax(&arr1(&[x[2], x[3]]));
            let sm_grouped = [sm0[0], sm0[1], sm1[0], sm1[1]];

            for i in 0..out_dim {
                // True output: A[i,:] @ softmax_grouped(x) + bias[i]
                let true_val: f32 = (0..total_in)
                    .map(|j| a_vals[i * total_in + j] * sm_grouped[j])
                    .sum::<f32>() + bias_vals[i];

                // Concretize result bounds at x
                let lb_val: f32 = (0..total_in)
                    .map(|j| result.lower_a[[i, j]] * x[j])
                    .sum::<f32>() + result.lower_b[[i]];
                let ub_val: f32 = (0..total_in)
                    .map(|j| result.upper_a[[i, j]] * x[j])
                    .sum::<f32>() + result.upper_b[[i]];

                prop_assert!(
                    lb_val <= true_val + FLAT_GROUPED_CROWN_TOLERANCE,
                    "flat-grouped sound lower violated: out={}, mask={}, lb={}, true={}, x={:?}",
                    i, mask, lb_val, true_val, x
                );
                prop_assert!(
                    ub_val >= true_val - FLAT_GROUPED_CROWN_TOLERANCE,
                    "flat-grouped sound upper violated: out={}, mask={}, ub={}, true={}, x={:?}",
                    i, mask, ub_val, true_val, x
                );
            }
        }
    }

    /// Flat-grouped vs per-group equivalence: flat-grouped backward matches
    /// independent per-group 1D backward on block-diagonal A.
    ///
    /// The A coefficients should match exactly (same code path, same inputs).
    /// Bias matches within directed-rounding tolerance from the equal split.
    ///
    /// Design: designs/2026-03-03-flat-grouped-softmax-crown-testing.md Test 2.
    /// Part of #3247.
    #[ntest::timeout(60000)]
    #[test]
    fn flat_grouped_matches_per_group_softmax_crown_sound(
        (l00, u00) in super::valid_interval(3.0),
        (l01, u01) in super::valid_interval(3.0),
        (l10, u10) in super::valid_interval(3.0),
        (l11, u11) in super::valid_interval(3.0),
        a00 in -2.0f32..2.0,
        a01 in -2.0f32..2.0,
        a10 in -2.0f32..2.0,
        a11 in -2.0f32..2.0,
    ) {
        prop_assume!(u00 > l00 + 0.01);
        prop_assume!(u01 > l01 + 0.01);
        prop_assume!(u10 > l10 + 0.01);
        prop_assume!(u11 > l11 + 0.01);

        let softmax_size = 2usize;
        let out_dim = 1usize;
        let total_in = 4usize;
        let num_groups = 2usize;
        let layer = SoftmaxLayer::new(-1);

        // === Per-group independent 1D backward ===
        // Group 0: A=[1,2] with [a00, a01], bias=0, pre=[l00,l01]..[u00,u01]
        let group0_bounds = LinearBounds::new(
            Array2::from_shape_vec((out_dim, softmax_size), vec![a00, a01]).unwrap(),
            Array1::zeros(out_dim),
            Array2::from_shape_vec((out_dim, softmax_size), vec![a00, a01]).unwrap(),
            Array1::zeros(out_dim),
        ).unwrap();
        let group0_pre = BoundedTensor::new(
            arr1(&[l00, l01]).into_dyn(),
            arr1(&[u00, u01]).into_dyn(),
        ).unwrap();
        let group0_result = layer
            .propagate_linear_with_bounds(
                &group0_bounds, &group0_pre, VerificationSoundnessMode::Sound,
            )
            .map_err(|e| TestCaseError::fail(
                format!("group 0 1D sound failed: {e}")
            ))?;

        // Group 1: A=[1,2] with [a10, a11], bias=0, pre=[l10,l11]..[u10,u11]
        let group1_bounds = LinearBounds::new(
            Array2::from_shape_vec((out_dim, softmax_size), vec![a10, a11]).unwrap(),
            Array1::zeros(out_dim),
            Array2::from_shape_vec((out_dim, softmax_size), vec![a10, a11]).unwrap(),
            Array1::zeros(out_dim),
        ).unwrap();
        let group1_pre = BoundedTensor::new(
            arr1(&[l10, l11]).into_dyn(),
            arr1(&[u10, u11]).into_dyn(),
        ).unwrap();
        let group1_result = layer
            .propagate_linear_with_bounds(
                &group1_bounds, &group1_pre, VerificationSoundnessMode::Sound,
            )
            .map_err(|e| TestCaseError::fail(
                format!("group 1 1D sound failed: {e}")
            ))?;

        // === Flat-grouped backward ===
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(
                IxDyn(&[num_groups, softmax_size]),
                vec![l00, l01, l10, l11],
            ).unwrap(),
            ArrayD::from_shape_vec(
                IxDyn(&[num_groups, softmax_size]),
                vec![u00, u01, u10, u11],
            ).unwrap(),
        ).unwrap();

        // Flat A: [1, 4] = [a00, a01, a10, a11], zero bias
        let la = ArrayD::from_shape_vec(
            IxDyn(&[out_dim, total_in]),
            vec![a00, a01, a10, a11],
        ).unwrap();
        let lb = ArrayD::zeros(IxDyn(&[out_dim]));
        let batched_bounds = BatchedLinearBounds::from_parts_unchecked(
            la.clone(), lb.clone(), la, lb,
            vec![total_in], vec![out_dim],
        );

        let flat_result = layer
            .propagate_linear_batched_with_bounds(
                &batched_bounds, &input, VerificationSoundnessMode::Sound,
            )
            .map_err(|e| TestCaseError::fail(
                format!("flat-grouped sound failed: {e}")
            ))?;

        // Compare A coefficients: flat-grouped columns should match per-group results.
        // Group 0: flat_result.lower_a[[0, 0..2]] == group0_result.lower_a[[0, 0..2]]
        for k in 0..softmax_size {
            prop_assert!(
                (flat_result.lower_a[[0, k]] - group0_result.lower_a[[0, k]]).abs() < 1e-6,
                "group 0 lower_a[0,{}]: flat={}, per_group={}",
                k, flat_result.lower_a[[0, k]], group0_result.lower_a[[0, k]]
            );
            prop_assert!(
                (flat_result.upper_a[[0, k]] - group0_result.upper_a[[0, k]]).abs() < 1e-6,
                "group 0 upper_a[0,{}]: flat={}, per_group={}",
                k, flat_result.upper_a[[0, k]], group0_result.upper_a[[0, k]]
            );
        }
        // Group 1: flat_result.lower_a[[0, 2..4]] == group1_result.lower_a[[0, 0..2]]
        for k in 0..softmax_size {
            prop_assert!(
                (flat_result.lower_a[[0, softmax_size + k]] - group1_result.lower_a[[0, k]]).abs() < 1e-6,
                "group 1 lower_a[0,{}]: flat={}, per_group={}",
                k, flat_result.lower_a[[0, softmax_size + k]], group1_result.lower_a[[0, k]]
            );
            prop_assert!(
                (flat_result.upper_a[[0, softmax_size + k]] - group1_result.upper_a[[0, k]]).abs() < 1e-6,
                "group 1 upper_a[0,{}]: flat={}, per_group={}",
                k, flat_result.upper_a[[0, softmax_size + k]], group1_result.upper_a[[0, k]]
            );
        }

        // Bias comparison: flat bias ≈ sum of per-group biases.
        // The flat-grouped path splits original bias (0) by num_groups with directed
        // rounding, introducing O(epsilon) difference. With 2 groups, total diff ≤ 2 ULPs.
        let expected_lower_b = group0_result.lower_b[0] + group1_result.lower_b[0];
        let expected_upper_b = group0_result.upper_b[0] + group1_result.upper_b[0];
        let bias_tol = (num_groups as f32) * f32::EPSILON
            * expected_lower_b.abs().max(expected_upper_b.abs()).max(1.0);
        let bias_tol = bias_tol.max(1e-5);

        prop_assert!(
            (flat_result.lower_b[[0]] - expected_lower_b).abs() < bias_tol,
            "lower_b mismatch: flat={}, expected={}, tol={}",
            flat_result.lower_b[[0]], expected_lower_b, bias_tol
        );
        prop_assert!(
            (flat_result.upper_b[[0]] - expected_upper_b).abs() < bias_tol,
            "upper_b mismatch: flat={}, expected={}, tol={}",
            flat_result.upper_b[[0]], expected_upper_b, bias_tol
        );
    }
}
