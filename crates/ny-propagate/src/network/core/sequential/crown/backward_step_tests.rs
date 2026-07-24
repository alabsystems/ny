// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for dense CROWN backward-step dispatch.

use super::*;
use crate::layers::{AddLayer, Layer, LinearLayer, MulBinaryLayer, ReLULayer, SkipMergeLayer};
use ndarray::{arr1, arr2, Array1};
use ny_core::Result;
use ny_tensor::BoundedTensor;

/// Helper: create identity LinearBounds for `dim` outputs/inputs.
fn identity_bounds(dim: usize) -> LinearBounds {
    LinearBounds::identity(dim)
}

/// Helper: create pre-activation BoundedTensor from 1D arrays.
fn bounded_1d(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        Array1::from_vec(lower.to_vec()).into_dyn(),
        Array1::from_vec(upper.to_vec()).into_dyn(),
    )
    .expect("valid bounds")
}

// ── crown_backward_step: Linear layer ─────────────────────────────────────

/// Linear layer backward: y = Wx + b produces A_new = A @ W^T, b_new = A @ b + b_old.
/// With identity incoming bounds, A_new should equal W^T.
///
/// Reference: CROWN backward for linear layers composes the weight matrix.
/// alpha-beta-CROWN: auto_LiRPA/operators/linear.py
#[test]
fn test_crown_backward_step_linear_composes_weight_matrix() -> Result<()> {
    // 2-input, 3-output linear: W = [[1, 2], [3, 4], [5, 6]], b = [0.1, 0.2, 0.3]
    let weight = arr2(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let bias = arr1(&[0.1f32, 0.2, 0.3]);
    let layer = Layer::Linear(LinearLayer::new(weight.clone(), Some(bias.clone()))?);

    // Identity bounds: 3 outputs, 3 inputs (matching layer output dim).
    let mut lb = identity_bounds(3);
    let pre_act = bounded_1d(&[0.0, 0.0], &[1.0, 1.0]);

    let result = crown_backward_step(&layer, &mut lb, &pre_act, None, 0, "test")?;
    assert!(matches!(result, CrownStepResult::Continue));

    // After backward through y = Wx + b with identity incoming bounds:
    // A_new = I @ W = W, and the bounds should map to 2 inputs.
    assert_eq!(lb.num_outputs(), 3);
    assert_eq!(lb.num_inputs(), 2);

    // Lower and upper A matrices should match the weight (with possible
    // directed rounding tolerance for GPU/faer paths).
    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (lb.lower_a()[[i, j]] - weight[[i, j]]).abs() < 1e-5,
                "lower_a[{i},{j}] = {} != weight[{i},{j}] = {}",
                lb.lower_a()[[i, j]],
                weight[[i, j]]
            );
            assert!(
                (lb.upper_a()[[i, j]] - weight[[i, j]]).abs() < 1e-5,
                "upper_a[{i},{j}] = {} != weight[{i},{j}] = {}",
                lb.upper_a()[[i, j]],
                weight[[i, j]]
            );
        }
    }

    // Bias should incorporate the linear layer's bias: b_new = b_old + b_layer.
    // With identity incoming bounds (b_old = 0), b_new = b_layer.
    for i in 0..3 {
        assert!(
            (lb.lower_b()[i] - bias[i]).abs() < 1e-4,
            "lower_b[{i}] = {}, expected {}",
            lb.lower_b()[i],
            bias[i]
        );
        assert!(
            (lb.upper_b()[i] - bias[i]).abs() < 1e-4,
            "upper_b[{i}] = {}, expected {}",
            lb.upper_b()[i],
            bias[i]
        );
    }
    Ok(())
}

// ── crown_backward_step: ReLU ─────────────────────────────────────────────

/// ReLU backward with fully active pre-activation (all l >= 0): identity pass-through.
///
/// When all pre-activation lower bounds are non-negative, ReLU is the identity
/// function and CROWN backward should preserve bounds exactly.
/// Reference: relu_linear_relaxation when l >= 0 returns (slope=1, intercept=0).
#[test]
fn test_crown_backward_step_relu_fully_active() -> Result<()> {
    let layer = Layer::ReLU(ReLULayer::new());
    let mut lb = identity_bounds(2);
    // Pre-activation: all positive → ReLU is identity.
    let pre_act = bounded_1d(&[1.0, 2.0], &[3.0, 4.0]);

    let result = crown_backward_step(&layer, &mut lb, &pre_act, None, 0, "test")?;
    assert!(matches!(result, CrownStepResult::Continue));

    // Identity pass-through: A matrices unchanged, bias unchanged.
    assert_eq!(lb.num_outputs(), 2);
    assert_eq!(lb.num_inputs(), 2);
    assert!((lb.lower_a()[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((lb.lower_a()[[1, 1]] - 1.0).abs() < 1e-6);
    assert!(lb.lower_a()[[0, 1]].abs() < 1e-6);
    assert!(lb.lower_a()[[1, 0]].abs() < 1e-6);
    Ok(())
}

/// ReLU backward with fully inactive pre-activation (all u <= 0): zero output.
///
/// When all pre-activation upper bounds are non-positive, ReLU output is zero
/// and CROWN backward should produce zero A-matrices with zero biases.
/// Reference: relu_linear_relaxation when u <= 0 returns (slope=0, intercept=0).
#[test]
fn test_crown_backward_step_relu_fully_inactive() -> Result<()> {
    let layer = Layer::ReLU(ReLULayer::new());
    let mut lb = identity_bounds(2);
    // Pre-activation: all negative → ReLU kills everything.
    let pre_act = bounded_1d(&[-4.0, -3.0], &[-2.0, -1.0]);

    let result = crown_backward_step(&layer, &mut lb, &pre_act, None, 0, "test")?;
    assert!(matches!(result, CrownStepResult::Continue));

    // Zero output: A matrices should be zero, biases should be zero.
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                lb.lower_a()[[i, j]].abs() < 1e-6,
                "lower_a[{i},{j}] = {} should be 0",
                lb.lower_a()[[i, j]]
            );
            assert!(
                lb.upper_a()[[i, j]].abs() < 1e-6,
                "upper_a[{i},{j}] = {} should be 0",
                lb.upper_a()[[i, j]]
            );
        }
    }
    Ok(())
}

/// ReLU backward with unstable neurons (l < 0 < u): produces triangle relaxation.
///
/// For unstable neuron j with bounds [l, u]:
/// - Upper bound slope = u / (u - l), intercept = -l*u / (u - l)
/// - Lower bound slope = 0 (adaptive, but default alpha=0 in sequential CROWN)
///
/// Reference: CROWN linear relaxation of ReLU.
/// alpha-beta-CROWN: auto_LiRPA/operators/relu.py:bound_relax
#[test]
fn test_crown_backward_step_relu_unstable_soundness() -> Result<()> {
    let layer = Layer::ReLU(ReLULayer::new());
    let mut lb = identity_bounds(1);
    // Single unstable neuron: l = -2, u = 4.
    // Upper slope = 4 / (4 - (-2)) = 4/6 = 2/3
    // Upper intercept = -(-2)*4 / 6 = 8/6 = 4/3
    let pre_act = bounded_1d(&[-2.0], &[4.0]);

    let result = crown_backward_step(&layer, &mut lb, &pre_act, None, 0, "test")?;
    assert!(matches!(result, CrownStepResult::Continue));

    // Concretize and verify soundness: bounds must contain true ReLU output.
    // ReLU([-2, 4]) = [0, 4].
    let concrete = lb.concretize(&pre_act);
    let concrete_flat = concrete.flatten();
    assert!(
        concrete_flat.lower()[[0]] <= 0.0,
        "lower bound {} must be <= 0.0 (ReLU of -2 = 0)",
        concrete_flat.lower()[[0]]
    );
    assert!(
        concrete_flat.upper()[[0]] >= 4.0,
        "upper bound {} must be >= 4.0 (ReLU of 4 = 4)",
        concrete_flat.upper()[[0]]
    );
    Ok(())
}

// ── crown_backward_step: Multi-input ops → IbpFallback ────────────────────

/// Multi-input ops (Add, MulBinary, etc.) must return IbpFallback in sequential CROWN.
///
/// Sequential networks don't have named graph edges, so binary/n-ary ops
/// can't resolve their second input. The backward step pre-filters these.
#[test]
fn test_crown_backward_step_add_returns_ibp_fallback() -> Result<()> {
    let layer = Layer::Add(AddLayer);
    let mut lb = identity_bounds(2);
    let pre_act = bounded_1d(&[0.0, 0.0], &[1.0, 1.0]);

    let result = crown_backward_step(&layer, &mut lb, &pre_act, None, 0, "test")?;
    assert!(
        matches!(result, CrownStepResult::IbpFallback(_)),
        "Add must return IbpFallback in sequential CROWN"
    );
    Ok(())
}

#[test]
fn test_crown_backward_step_mul_binary_returns_ibp_fallback() -> Result<()> {
    let layer = Layer::MulBinary(MulBinaryLayer);
    let mut lb = identity_bounds(2);
    let pre_act = bounded_1d(&[0.0, 0.0], &[1.0, 1.0]);

    let result = crown_backward_step(&layer, &mut lb, &pre_act, None, 0, "test")?;
    assert!(
        matches!(result, CrownStepResult::IbpFallback(_)),
        "MulBinary must return IbpFallback in sequential CROWN"
    );
    Ok(())
}

// ── crown_backward_step: SkipMerge pass-through ───────────────────────────

/// SkipMerge must be identity in sequential CROWN — bounds pass unchanged.
#[test]
fn test_crown_backward_step_skip_merge_passthrough() -> Result<()> {
    let layer = Layer::SkipMerge(SkipMergeLayer::new());
    let mut lb = LinearBounds::new(
        arr2(&[[2.0, 3.0]]),
        arr1(&[0.5]),
        arr2(&[[4.0, 5.0]]),
        arr1(&[0.7]),
    )?;
    let pre_act = bounded_1d(&[0.0, 0.0], &[1.0, 1.0]);

    let original_lower_a = lb.lower_a().clone();
    let original_upper_a = lb.upper_a().clone();
    let original_lower_b = lb.lower_b().clone();
    let original_upper_b = lb.upper_b().clone();

    let result = crown_backward_step(&layer, &mut lb, &pre_act, None, 0, "test")?;
    assert!(matches!(result, CrownStepResult::Continue));

    // Bounds must be exactly unchanged.
    assert_eq!(lb.lower_a(), &original_lower_a);
    assert_eq!(lb.upper_a(), &original_upper_a);
    assert_eq!(lb.lower_b(), &original_lower_b);
    assert_eq!(lb.upper_b(), &original_upper_b);
    Ok(())
}

// ── crown_backward_ibp_concretize ─────────────────────────────────────────

/// IBP concretization fallback must produce constant bounds (A=0) that
/// contain the true layer output.
///
/// When a layer's CROWN backward fails, the fallback concretizes accumulated
/// CROWN bounds through IBP at that layer. The result has zero A-coefficients
/// (no input dependence) with biases set to the concretized interval.
#[test]
fn test_crown_backward_ibp_concretize_produces_sound_constant_bounds() -> Result<()> {
    // Use ReLU as a proxy to test concretization. Manually invoke
    // crown_backward_ibp_concretize directly.
    let layer = Layer::ReLU(ReLULayer::new());
    let mut lb = identity_bounds(2);
    // Pre-activation: [-3, 5] and [1, 4]. IBP of ReLU: [0, 5] and [1, 4].
    let pre_act = bounded_1d(&[-3.0, 1.0], &[5.0, 4.0]);

    let result =
        crown_backward_ibp_concretize(&layer, &mut lb, &pre_act, 0, "test", "test concretization")?;
    assert!(matches!(result, CrownStepResult::Continue));

    // A matrices must be all zeros (constant bounds).
    assert!(
        lb.lower_a().iter().all(|&v| v == 0.0),
        "lower_a must be zero after IBP concretization"
    );
    assert!(
        lb.upper_a().iter().all(|&v| v == 0.0),
        "upper_a must be zero after IBP concretization"
    );

    // Bias must contain the true ReLU output: [0, 5] and [1, 4].
    // Directed rounding may widen by 1 ULP.
    assert!(
        lb.lower_b()[0] <= 0.0 + 1e-5,
        "lower_b[0] = {} must be <= 0.0 (ReLU lower of -3)",
        lb.lower_b()[0]
    );
    assert!(
        lb.upper_b()[0] >= 5.0 - 1e-5,
        "upper_b[0] = {} must be >= 5.0 (ReLU upper of 5)",
        lb.upper_b()[0]
    );
    assert!(
        lb.lower_b()[1] <= 1.0 + 1e-5,
        "lower_b[1] = {} must be <= 1.0",
        lb.lower_b()[1]
    );
    assert!(
        lb.upper_b()[1] >= 4.0 - 1e-5,
        "upper_b[1] = {} must be >= 4.0",
        lb.upper_b()[1]
    );
    Ok(())
}

/// IBP concretization with non-identity incoming bounds: must concretize
/// the accumulated linear function through IBP at the failing layer.
#[test]
fn test_crown_backward_ibp_concretize_with_scaled_incoming_bounds() -> Result<()> {
    let layer = Layer::ReLU(ReLULayer::new());
    // Incoming bounds: 1 output = 2 * x0 + 3 * x1 + offset 0.
    let mut lb = LinearBounds::new(
        arr2(&[[2.0, 3.0]]),
        arr1(&[0.0]),
        arr2(&[[2.0, 3.0]]),
        arr1(&[0.0]),
    )?;
    // Pre-activation: x0 in [-1, 3], x1 in [0, 2]. IBP of ReLU: [0, 3] and [0, 2].
    let pre_act = bounded_1d(&[-1.0, 0.0], &[3.0, 2.0]);

    // Concretize through IBP at the ReLU layer:
    // ReLU([-1,3]) = [0,3], ReLU([0,2]) = [0,2]
    // Then concretize 2*[0,3] + 3*[0,2] = [0, 6] + [0, 6] = [0, 12]
    let result = crown_backward_ibp_concretize(
        &layer,
        &mut lb,
        &pre_act,
        0,
        "test",
        "test scaled concretization",
    )?;
    assert!(matches!(result, CrownStepResult::Continue));

    // A matrices must be all zeros.
    assert!(lb.lower_a().iter().all(|&v| v == 0.0));
    assert!(lb.upper_a().iter().all(|&v| v == 0.0));

    // Output bounds must contain [0, 12].
    assert!(
        lb.lower_b()[0] <= 0.0 + 1e-3,
        "lower_b[0] = {} must be <= 0",
        lb.lower_b()[0]
    );
    assert!(
        lb.upper_b()[0] >= 12.0 - 1e-3,
        "upper_b[0] = {} must be >= 12",
        lb.upper_b()[0]
    );
    Ok(())
}

// ── Soundness: concretized bounds contain true output ─────────────────────

/// End-to-end soundness check: Linear → ReLU backward produces bounds that
/// contain the true network output for sampled inputs within the input domain.
///
/// This is the fundamental CROWN soundness property: for all x in [l, u],
/// the concretized bounds must satisfy lb(x) <= f(x) <= ub(x).
#[test]
fn test_crown_backward_linear_relu_end_to_end_soundness() -> Result<()> {
    // Network: y = ReLU(Wx + b)
    // W = [[1, -1], [-2, 1]], b = [0.5, -0.5]
    let weight = arr2(&[[1.0f32, -1.0], [-2.0, 1.0]]);
    let bias = arr1(&[0.5f32, -0.5]);
    let linear_layer = Layer::Linear(LinearLayer::new(weight.clone(), Some(bias.clone()))?);
    let relu_layer = Layer::ReLU(ReLULayer::new());

    // Input domain: x0 in [-1, 1], x1 in [-1, 1].
    let input = bounded_1d(&[-1.0, -1.0], &[1.0, 1.0]);

    // Forward IBP to get pre-activation bounds for each layer.
    let pre_linear = input.clone();
    let post_linear = linear_layer.propagate_ibp(&pre_linear)?;
    let post_relu = relu_layer.propagate_ibp(&post_linear)?;

    // CROWN backward: start from identity at output, go through ReLU then Linear.
    let mut lb = identity_bounds(post_relu.len());

    let relu_result = crown_backward_step(&relu_layer, &mut lb, &post_linear, None, 1, "test")?;
    assert!(matches!(relu_result, CrownStepResult::Continue));

    let linear_result = crown_backward_step(&linear_layer, &mut lb, &pre_linear, None, 0, "test")?;
    assert!(matches!(linear_result, CrownStepResult::Continue));

    // Concretize CROWN bounds.
    let crown_concrete = lb.concretize(&input);
    let crown_flat = crown_concrete.flatten();

    // Verify soundness: sample 25 points in the input domain and check
    // that the true network output is contained within CROWN bounds.
    for xi in 0..5 {
        for xj in 0..5 {
            let x0 = -1.0 + 2.0 * (xi as f32) / 4.0;
            let x1 = -1.0 + 2.0 * (xj as f32) / 4.0;

            // True output: ReLU(W @ [x0, x1] + b)
            let z0 = weight[[0, 0]] * x0 + weight[[0, 1]] * x1 + bias[0];
            let z1 = weight[[1, 0]] * x0 + weight[[1, 1]] * x1 + bias[1];
            let y0 = z0.max(0.0);
            let y1 = z1.max(0.0);

            assert!(
                crown_flat.lower()[[0]] <= y0 + 1e-5,
                "CROWN lower[0]={} > true y0={} at ({x0},{x1})",
                crown_flat.lower()[[0]],
                y0
            );
            assert!(
                crown_flat.upper()[[0]] >= y0 - 1e-5,
                "CROWN upper[0]={} < true y0={} at ({x0},{x1})",
                crown_flat.upper()[[0]],
                y0
            );
            assert!(
                crown_flat.lower()[[1]] <= y1 + 1e-5,
                "CROWN lower[1]={} > true y1={} at ({x0},{x1})",
                crown_flat.lower()[[1]],
                y1
            );
            assert!(
                crown_flat.upper()[[1]] >= y1 - 1e-5,
                "CROWN upper[1]={} < true y1={} at ({x0},{x1})",
                crown_flat.upper()[[1]],
                y1
            );
        }
    }
    Ok(())
}
