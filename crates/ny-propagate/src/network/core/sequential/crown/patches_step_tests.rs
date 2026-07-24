// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for patches-aware CROWN backward-step dispatch.
//!
//! Tests the dispatch logic in `crown_backward_step_patches`:
//! - Dense mode delegates correctly to `crown_backward_step`
//! - Patches→Dense termination for structural layers (Linear, Flatten)
//! - Patches-native activation dispatch (ReLU stays in Patches mode)

use super::*;
use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
use crate::bounds::LinearBounds;
use crate::layers::{AddLayer, Layer, LinearLayer, MulBinaryLayer, ReLULayer, SkipMergeLayer};
use crate::BoundPropagation;
use ndarray::{arr1, arr2, Array1, ArrayD, IxDyn};
use ny_core::Result;
use ny_tensor::BoundedTensor;

/// Helper: create pre-activation BoundedTensor from 1D arrays.
fn bounded_1d(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        Array1::from_vec(lower.to_vec()).into_dyn(),
        Array1::from_vec(upper.to_vec()).into_dyn(),
    )
    .expect("valid bounds")
}

/// Helper: create pre-activation BoundedTensor from 3D shape (C, H, W).
fn bounded_3d(shape: (usize, usize, usize), lower_val: f32, upper_val: f32) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[shape.0, shape.1, shape.2]), lower_val),
        ArrayD::from_elem(IxDyn(&[shape.0, shape.1, shape.2]), upper_val),
    )
    .expect("valid bounds")
}

// ── Dense-mode pass-through tests ─────────────────────────────────────────
// When CrownBounds is already Dense, patches_step delegates to crown_backward_step.

/// Dense + ReLU fully active: identity pass-through (same as backward_step).
///
/// Verifies that crown_backward_step_patches correctly delegates Dense-mode
/// ReLU backward to crown_backward_step, which uses the trait path.
#[test]
fn test_patches_step_dense_relu_fully_active() -> Result<()> {
    let layer = Layer::ReLU(ReLULayer::new());
    let mut bounds = CrownBounds::Dense(LinearBounds::identity(2));
    let pre_act = bounded_1d(&[1.0, 2.0], &[3.0, 4.0]);

    let result = crown_backward_step_patches(&layer, &mut bounds, &pre_act, None, 0, "test", None)?;
    assert!(
        matches!(result, CrownStepResult::Continue),
        "Dense ReLU fully active should Continue"
    );

    // Verify bounds are still Dense.
    let lb = match &bounds {
        CrownBounds::Dense(lb) => lb,
        CrownBounds::Patches(_) => panic!("expected Dense, got Patches"),
    };
    // Fully active ReLU = identity: A-matrices unchanged.
    assert!((lb.lower_a()[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((lb.lower_a()[[1, 1]] - 1.0).abs() < 1e-6);
    assert!(lb.lower_a()[[0, 1]].abs() < 1e-6);
    assert!(lb.lower_a()[[1, 0]].abs() < 1e-6);
    Ok(())
}

/// Dense + Linear: weight matrix composition through patches dispatch.
///
/// Verifies that the patches step correctly delegates Dense-mode Linear
/// backward to crown_backward_step, producing A_new = I @ W = W.
#[test]
fn test_patches_step_dense_linear_composes_weight() -> Result<()> {
    let weight = arr2(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let bias = arr1(&[0.1f32, 0.2, 0.3]);
    let layer = Layer::Linear(LinearLayer::new(weight.clone(), Some(bias))?);
    let mut bounds = CrownBounds::Dense(LinearBounds::identity(3));
    let pre_act = bounded_1d(&[0.0, 0.0], &[1.0, 1.0]);

    let result = crown_backward_step_patches(&layer, &mut bounds, &pre_act, None, 0, "test", None)?;
    assert!(
        matches!(result, CrownStepResult::Continue),
        "Dense Linear should Continue"
    );

    let lb = match &bounds {
        CrownBounds::Dense(lb) => lb,
        CrownBounds::Patches(_) => panic!("expected Dense, got Patches"),
    };
    assert_eq!(lb.num_outputs(), 3);
    assert_eq!(lb.num_inputs(), 2);

    // A_new = I @ W = W.
    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (lb.lower_a()[[i, j]] - weight[[i, j]]).abs() < 1e-5,
                "lower_a[{i},{j}] = {} != weight = {}",
                lb.lower_a()[[i, j]],
                weight[[i, j]]
            );
        }
    }
    Ok(())
}

/// Dense + Add: multi-input ops return IbpFallback through patches dispatch.
///
/// Sequential networks can't resolve second inputs for binary ops.
/// Verifies that patches_step delegates correctly and the IbpFallback propagates.
#[test]
fn test_patches_step_dense_add_returns_ibp_fallback() -> Result<()> {
    let layer = Layer::Add(AddLayer);
    let mut bounds = CrownBounds::Dense(LinearBounds::identity(2));
    let pre_act = bounded_1d(&[0.0, 0.0], &[1.0, 1.0]);

    let result = crown_backward_step_patches(&layer, &mut bounds, &pre_act, None, 0, "test", None)?;
    assert!(
        matches!(result, CrownStepResult::IbpFallback(_)),
        "Dense Add must return IbpFallback in sequential CROWN"
    );
    Ok(())
}

/// Dense + MulBinary: binary ops return IbpFallback through patches dispatch.
#[test]
fn test_patches_step_dense_mul_binary_returns_ibp_fallback() -> Result<()> {
    let layer = Layer::MulBinary(MulBinaryLayer);
    let mut bounds = CrownBounds::Dense(LinearBounds::identity(2));
    let pre_act = bounded_1d(&[0.0, 0.0], &[1.0, 1.0]);

    let result = crown_backward_step_patches(&layer, &mut bounds, &pre_act, None, 0, "test", None)?;
    assert!(
        matches!(result, CrownStepResult::IbpFallback(_)),
        "Dense MulBinary must return IbpFallback in sequential CROWN"
    );
    Ok(())
}

/// Dense + SkipMerge: identity pass-through preserves bounds unchanged.
#[test]
fn test_patches_step_dense_skip_merge_passthrough() -> Result<()> {
    let layer = Layer::SkipMerge(SkipMergeLayer::new());
    let original = LinearBounds::new(
        arr2(&[[2.0, 3.0]]),
        arr1(&[0.5]),
        arr2(&[[4.0, 5.0]]),
        arr1(&[0.7]),
    )?;
    let mut bounds = CrownBounds::Dense(original.clone());
    let pre_act = bounded_1d(&[0.0, 0.0], &[1.0, 1.0]);

    let result = crown_backward_step_patches(&layer, &mut bounds, &pre_act, None, 0, "test", None)?;
    assert!(
        matches!(result, CrownStepResult::Continue),
        "Dense SkipMerge should Continue"
    );

    let lb = match &bounds {
        CrownBounds::Dense(lb) => lb,
        CrownBounds::Patches(_) => panic!("expected Dense, got Patches"),
    };
    assert_eq!(lb.lower_a(), original.lower_a(), "lower_a unchanged");
    assert_eq!(lb.upper_a(), original.upper_a(), "upper_a unchanged");
    assert_eq!(lb.lower_b(), original.lower_b(), "lower_b unchanged");
    assert_eq!(lb.upper_b(), original.upper_b(), "upper_b unchanged");
    Ok(())
}

// ── Patches→Dense termination tests ───────────────────────────────────────
// Structural layers (Linear, Flatten, Reshape) terminate Patches mode by
// converting to Dense before dispatch.

/// Patches + Linear → Patches→Dense termination, then standard Linear backward.
///
/// Verifies that crown_backward_step_patches converts Patches to Dense
/// at a Linear layer boundary, then dispatches the Dense backward.
#[test]
fn test_patches_step_patches_to_dense_termination_linear() -> Result<()> {
    // Identity patches for spatial tensor (C=1, H=1, W=2) → 2 elements.
    let spatial = (1, 1, 2);
    let pb = PatchesLinearBounds::identity(spatial, spatial);
    let mut bounds = CrownBounds::Patches(Box::new(pb));

    // Linear: 2→2 identity weight, zero bias.
    let weight = arr2(&[[1.0f32, 0.0], [0.0, 1.0]]);
    let layer = Layer::Linear(LinearLayer::new(weight, None)?);
    let pre_act = bounded_1d(&[0.0, 0.0], &[1.0, 1.0]);

    let result = crown_backward_step_patches(&layer, &mut bounds, &pre_act, None, 0, "test", None)?;
    assert!(
        matches!(result, CrownStepResult::Continue),
        "Patches→Dense Linear should Continue"
    );

    // After termination: bounds must be Dense.
    let lb = match &bounds {
        CrownBounds::Dense(lb) => lb,
        CrownBounds::Patches(_) => panic!("expected Dense after Linear termination"),
    };

    // Identity Linear backward through identity patches = identity.
    assert_eq!(lb.num_outputs(), 2);
    assert_eq!(lb.num_inputs(), 2);
    assert!(
        (lb.lower_a()[[0, 0]] - 1.0).abs() < 1e-5,
        "identity preserved: lower_a[0,0] = {}",
        lb.lower_a()[[0, 0]]
    );
    assert!(
        (lb.lower_a()[[1, 1]] - 1.0).abs() < 1e-5,
        "identity preserved: lower_a[1,1] = {}",
        lb.lower_a()[[1, 1]]
    );
    Ok(())
}

// ── Patches-native activation dispatch tests ──────────────────────────────
// Element-wise activations in Patches mode use patches-native backward,
// preserving the sparse structure.

/// Patches + ReLU fully active: stays in Patches mode (patches-native dispatch).
///
/// When pre-activation bounds are all positive, ReLU is identity and the
/// patches activation backward should preserve Patches structure.
/// Reference: crown_elementwise_backward_patches applies per-element slopes.
#[test]
fn test_patches_step_patches_relu_fully_active_stays_patches() -> Result<()> {
    let spatial = (1, 1, 2);
    let pb = PatchesLinearBounds::identity(spatial, spatial);
    let mut bounds = CrownBounds::Patches(Box::new(pb));

    let layer = Layer::ReLU(ReLULayer::new());
    // Fully active: all lower bounds positive.
    let pre_act = bounded_3d(spatial, 1.0, 3.0);

    let result = crown_backward_step_patches(&layer, &mut bounds, &pre_act, None, 0, "test", None)?;
    assert!(
        matches!(result, CrownStepResult::Continue),
        "Patches ReLU fully active should Continue"
    );

    // Fully active ReLU should stay in Patches mode (no Dense conversion needed).
    assert!(
        matches!(bounds, CrownBounds::Patches(_)),
        "ReLU fully active should preserve Patches mode"
    );
    Ok(())
}

/// Patches + ReLU fully inactive: stays in Patches mode with zero coefficients.
///
/// When all pre-activation upper bounds are non-positive, ReLU output is zero.
/// The patches backward should zero out the A-matrices while staying in Patches mode.
#[test]
fn test_patches_step_patches_relu_fully_inactive_stays_patches() -> Result<()> {
    let spatial = (1, 1, 2);
    let pb = PatchesLinearBounds::identity(spatial, spatial);
    let mut bounds = CrownBounds::Patches(Box::new(pb));

    let layer = Layer::ReLU(ReLULayer::new());
    // Fully inactive: all upper bounds non-positive.
    let pre_act = bounded_3d(spatial, -3.0, -1.0);

    let result = crown_backward_step_patches(&layer, &mut bounds, &pre_act, None, 0, "test", None)?;
    assert!(
        matches!(result, CrownStepResult::Continue),
        "Patches ReLU fully inactive should Continue"
    );

    // Should stay in Patches mode.
    assert!(
        matches!(bounds, CrownBounds::Patches(_)),
        "ReLU fully inactive should preserve Patches mode"
    );

    // Verify zero output: convert to Dense and check A-matrices are zero.
    let lb = bounds.ensure_dense()?;
    assert!(
        lb.lower_a().iter().all(|&v| v.abs() < 1e-6),
        "fully inactive ReLU: lower_a should be zero, got {:?}",
        lb.lower_a()
    );
    assert!(
        lb.upper_a().iter().all(|&v| v.abs() < 1e-6),
        "fully inactive ReLU: upper_a should be zero, got {:?}",
        lb.upper_a()
    );
    Ok(())
}

// ── Dense-mode end-to-end soundness ───────────────────────────────────────

/// Verify CROWN bounds contain true ReLU(Wx+b) output for 25 sampled inputs.
fn assert_linear_relu_soundness(
    crown_flat: &BoundedTensor,
    weight: &ndarray::Array2<f32>,
    bias: &Array1<f32>,
) {
    for xi in 0..5 {
        for xj in 0..5 {
            let x0 = -1.0 + 2.0 * (xi as f32) / 4.0;
            let x1 = -1.0 + 2.0 * (xj as f32) / 4.0;
            let z0 = weight[[0, 0]] * x0 + weight[[0, 1]] * x1 + bias[0];
            let z1 = weight[[1, 0]] * x0 + weight[[1, 1]] * x1 + bias[1];
            let y0 = z0.max(0.0);
            let y1 = z1.max(0.0);
            assert!(
                crown_flat.lower()[[0]] <= y0 + 1e-5,
                "lower[0]={} > true y0={y0} at ({x0},{x1})",
                crown_flat.lower()[[0]]
            );
            assert!(
                crown_flat.upper()[[0]] >= y0 - 1e-5,
                "upper[0]={} < true y0={y0} at ({x0},{x1})",
                crown_flat.upper()[[0]]
            );
            assert!(
                crown_flat.lower()[[1]] <= y1 + 1e-5,
                "lower[1]={} > true y1={y1} at ({x0},{x1})",
                crown_flat.lower()[[1]]
            );
            assert!(
                crown_flat.upper()[[1]] >= y1 - 1e-5,
                "upper[1]={} < true y1={y1} at ({x0},{x1})",
                crown_flat.upper()[[1]]
            );
        }
    }
}

/// Dense-mode Linear→ReLU backward through patches dispatch must produce
/// sound bounds (contain true output for all inputs in domain).
///
/// Same soundness check as backward_step_tests but going through the
/// patches dispatch layer to verify no information is lost in the wrapper.
#[test]
fn test_patches_step_dense_linear_relu_soundness() -> Result<()> {
    let weight = arr2(&[[1.0f32, -1.0], [-2.0, 1.0]]);
    let bias = arr1(&[0.5f32, -0.5]);
    let linear_layer = Layer::Linear(LinearLayer::new(weight.clone(), Some(bias.clone()))?);
    let relu_layer = Layer::ReLU(ReLULayer::new());
    let input = bounded_1d(&[-1.0, -1.0], &[1.0, 1.0]);

    // Forward IBP for pre-activation bounds.
    let post_linear = linear_layer.propagate_ibp(&input)?;

    // CROWN backward through patches dispatch (Dense mode).
    let mut bounds = CrownBounds::Dense(LinearBounds::identity(post_linear.flatten().len()));
    let r1 = crown_backward_step_patches(
        &relu_layer,
        &mut bounds,
        &post_linear,
        None,
        1,
        "test",
        None,
    )?;
    assert!(matches!(r1, CrownStepResult::Continue));
    let r2 =
        crown_backward_step_patches(&linear_layer, &mut bounds, &input, None, 0, "test", None)?;
    assert!(matches!(r2, CrownStepResult::Continue));

    // Concretize and verify soundness against 25 sampled inputs.
    let lb = match &bounds {
        CrownBounds::Dense(lb) => lb,
        CrownBounds::Patches(_) => panic!("expected Dense"),
    };
    let crown_flat = lb.concretize(&input).flatten();
    assert_linear_relu_soundness(&crown_flat, &weight, &bias);
    Ok(())
}
