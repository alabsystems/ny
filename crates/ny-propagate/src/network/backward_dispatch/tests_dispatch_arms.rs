// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for untested backward CROWN dispatch arms (#3448).
//!
//! Covers: Transpose, Tile, Slice, Gather, ConvTranspose1d,
//! ConvTranspose2d, MatMul success-path, BilinearCrown success-path.

use std::collections::HashMap;

use ndarray::{Array1, ArrayD, IxDyn};

use super::dispatch::dispatch_backward_layer;
use super::types::{BackwardDispatchResult, DispatchContext};
use crate::bounds::LinearBounds;
use crate::layers::{
    BilinearCrownLayer, Conv2dLayer, ConvTranspose1dLayer, ConvTranspose2dLayer, GatherLayer,
    Layer, MatMulLayer, SliceLayer, TileLayer, TransposeLayer,
};
use crate::MulBinaryRelaxationMode;
use ny_tensor::BoundedTensor;

/// Helper: create identity LinearBounds of given dimension.
fn identity_lb(dim: usize) -> LinearBounds {
    LinearBounds::identity(dim)
}

/// Helper: create a simple BoundedTensor of shape [dim].
fn simple_bounds(dim: usize) -> BoundedTensor {
    let lower = ArrayD::from_elem(IxDyn(&[dim]), -1.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[dim]), 1.0_f32);
    BoundedTensor::new(lower, upper).unwrap()
}

/// Helper: create a shaped BoundedTensor.
fn shaped_bounds(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), -1.0_f32),
        ArrayD::from_elem(IxDyn(shape), 1.0_f32),
    )
    .unwrap()
}

/// Helper: build a DispatchContext for a node.
fn make_ctx<'a>(
    layer: &'a Layer,
    pre_act: &'a BoundedTensor,
    net_input: &'a BoundedTensor,
    node_bounds: &'a HashMap<String, BoundedTensor>,
    inputs: &'a [String],
) -> DispatchContext<'a> {
    DispatchContext {
        node_name: "test_node",
        layer,
        inputs,
        pre_activation: pre_act,
        network_input: net_input,
        node_bounds: node_bounds.into(),
        engine: None,
        deadline: None,
        bilinear_alphas: None,
        mul_binary_relaxation: MulBinaryRelaxationMode::default(),
        mul_binary_alphas: None,
        norm_inv_rms_override: None,
    }
}

// ===================================================================
// Transpose dispatch (#3448)
// ===================================================================

/// Transpose dispatch: 2D transpose [3, 4] → [4, 3] returns Single.
#[test]
fn dispatch_transpose_returns_single_3448() {
    let layer = Layer::Transpose(TransposeLayer::new(vec![1, 0]));
    let pre_act = shaped_bounds(&[3, 4]);
    let flat_dim = 3 * 4;
    let net_input = simple_bounds(flat_dim);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs);
    // Output of transpose([3,4], axes=[1,0]) is [4,3] → flat dim = 12
    let lb = identity_lb(flat_dim);

    let result = dispatch_backward_layer(&ctx, &lb).expect("Transpose dispatch should succeed");
    assert!(
        matches!(result, BackwardDispatchResult::Single(_)),
        "Transpose should return Single, got {result:?}"
    );
}

/// Transpose dispatch: verify A-matrix is the correct permutation matrix.
///
/// For Transpose([1,0]) on shape [2, 3]:
///   input[i, j] → output[j, i]
///   flat input: i*3 + j, flat output: j*2 + i
/// The backward A-matrix should be [6, 6] with exactly one 1.0 per row/col.
#[test]
fn dispatch_transpose_a_matrix_nontrivial_3448() {
    let layer = Layer::Transpose(TransposeLayer::new(vec![1, 0]));
    let pre_act = shaped_bounds(&[2, 3]);
    let flat_dim = 6;
    let net_input = simple_bounds(flat_dim);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs);
    let lb = identity_lb(flat_dim);

    let result = dispatch_backward_layer(&ctx, &lb).unwrap();
    match result {
        BackwardDispatchResult::Single(bounds) => {
            assert_eq!(bounds.lower_a.shape(), &[6, 6], "Transpose A-matrix shape");
            // Build expected permutation: output[j*2+i] comes from input[i*3+j]
            // So A[output_flat, input_flat] = 1.0
            let mut expected = [[0.0_f32; 6]; 6];
            for i in 0..2 {
                for j in 0..3 {
                    let input_flat = i * 3 + j;
                    let output_flat = j * 2 + i;
                    expected[output_flat][input_flat] = 1.0;
                }
            }
            for (row, expected_row) in expected.iter().enumerate() {
                for (col, &exp) in expected_row.iter().enumerate() {
                    let val = bounds.lower_a[[row, col]];
                    assert!(
                        (val - exp).abs() < 1e-6,
                        "Transpose A[{row}, {col}] = {val}, expected {exp}"
                    );
                }
            }
        }
        other => panic!("Expected Single, got {other:?}"),
    }
}

// ===================================================================
// Tile dispatch (#3448)
// ===================================================================

/// Tile dispatch: repeat [4] along axis 0 twice → [8] returns Single
/// with correct aggregation structure.
///
/// The backward of Tile sums columns corresponding to each replica:
/// new_A[:, k] = sum(A[:, j] for j in replicas_of_k).
/// With identity incoming bounds, the result is [8, 4] where rows 0-3
/// and rows 4-7 each have a 1.0 at the column matching the source input.
#[test]
fn dispatch_tile_returns_single_3448() {
    let layer = Layer::Tile(TileLayer::new(0, 2));
    let pre_act = shaped_bounds(&[4]);
    let net_input = simple_bounds(4);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs);
    // Output flat dim = 8 (tiled)
    let lb = identity_lb(8);

    let result = dispatch_backward_layer(&ctx, &lb).expect("Tile dispatch should succeed");
    match result {
        BackwardDispatchResult::Single(bounds) => {
            // A-matrix maps 8 output dims back to 4 input dims
            assert_eq!(
                bounds.lower_a.shape(),
                &[8, 4],
                "Tile backward A-matrix shape: [output=8, input=4]"
            );
            // Each row should have exactly one nonzero entry (1.0) at the
            // source input index. Rows 0-3 map to inputs 0-3 (rep 0),
            // rows 4-7 map to inputs 0-3 (rep 1).
            for row in 0..8 {
                let expected_col = row % 4;
                for col in 0..4 {
                    let val = bounds.lower_a[[row, col]];
                    if col == expected_col {
                        assert!(
                            (val - 1.0).abs() < 1e-6,
                            "Tile A[{row}, {col}] should be 1.0, got {val}"
                        );
                    } else {
                        assert!(
                            val.abs() < 1e-6,
                            "Tile A[{row}, {col}] should be 0.0, got {val}"
                        );
                    }
                }
            }
        }
        other => panic!("Expected Single, got {other:?}"),
    }
}

// ===================================================================
// Slice dispatch (#3448)
// ===================================================================

/// Slice dispatch: slice [6] at axis 0 from [1..4] → [3] returns Single.
#[test]
fn dispatch_slice_returns_single_3448() {
    let layer = Layer::Slice(SliceLayer::new(0, 1, 4));
    let pre_act = shaped_bounds(&[6]);
    let net_input = simple_bounds(6);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs);
    // Slice output dim = 3 (indices 1..4)
    let lb = identity_lb(3);

    let result = dispatch_backward_layer(&ctx, &lb).expect("Slice dispatch should succeed");
    assert!(
        matches!(result, BackwardDispatchResult::Single(_)),
        "Slice should return Single, got {result:?}"
    );
}

/// Slice dispatch: verify the backward A-matrix is a selection matrix.
///
/// Slice(axis=0, start=1, end=3) on input [5] selects indices [1, 2].
/// The backward A-matrix should be [2, 5] with:
///   row 0: [0, 1, 0, 0, 0]  (output 0 ← input 1)
///   row 1: [0, 0, 1, 0, 0]  (output 1 ← input 2)
#[test]
fn dispatch_slice_a_matrix_structure_3448() {
    let layer = Layer::Slice(SliceLayer::new(0, 1, 3));
    let pre_act = shaped_bounds(&[5]);
    let net_input = simple_bounds(5);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs);
    // Slice output: indices [1, 2] → dim 2
    let lb = identity_lb(2);

    let result = dispatch_backward_layer(&ctx, &lb).unwrap();
    match result {
        BackwardDispatchResult::Single(bounds) => {
            assert_eq!(
                bounds.lower_a.shape(),
                &[2, 5],
                "Slice backward A-matrix shape mismatch"
            );
            // Verify selection pattern: row r selects input (start + r)
            for row in 0..2 {
                let expected_col = 1 + row; // start=1
                for col in 0..5 {
                    let val = bounds.lower_a[[row, col]];
                    if col == expected_col {
                        assert!(
                            (val - 1.0).abs() < 1e-6,
                            "Slice A[{row}, {col}] should be 1.0, got {val}"
                        );
                    } else {
                        assert!(
                            val.abs() < 1e-6,
                            "Slice A[{row}, {col}] should be 0.0, got {val}"
                        );
                    }
                }
            }
        }
        other => panic!("Expected Single, got {other:?}"),
    }
}

// ===================================================================
// Gather dispatch (#3448)
// ===================================================================

/// Gather dispatch: gather from [4] at indices [1, 3] → [2] returns Single.
#[test]
fn dispatch_gather_returns_single_3448() {
    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1_i64, 3]).unwrap();
    let layer = Layer::Gather(GatherLayer::new(0, Some(indices), vec![2]));
    let pre_act = shaped_bounds(&[4]);
    let net_input = simple_bounds(4);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs);
    // Gather output: 2 elements selected
    let lb = identity_lb(2);

    let result = dispatch_backward_layer(&ctx, &lb).expect("Gather dispatch should succeed");
    assert!(
        matches!(result, BackwardDispatchResult::Single(_)),
        "Gather should return Single, got {result:?}"
    );
}

/// Gather dispatch: verify A-matrix maps gathered indices correctly.
#[test]
fn dispatch_gather_a_matrix_structure_3448() {
    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0_i64, 2]).unwrap();
    let layer = Layer::Gather(GatherLayer::new(0, Some(indices), vec![2]));
    let pre_act = shaped_bounds(&[4]);
    let net_input = simple_bounds(4);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs);
    let lb = identity_lb(2);

    let result = dispatch_backward_layer(&ctx, &lb).unwrap();
    match result {
        BackwardDispatchResult::Single(bounds) => {
            // A-matrix: [2, 4] mapping 2 outputs back to 4 inputs
            assert_eq!(
                bounds.lower_a.shape(),
                &[2, 4],
                "Gather backward A-matrix shape mismatch"
            );
            // Row 0 should select input[0], row 1 should select input[2]
            assert!(
                bounds.lower_a[[0, 0]].abs() > 0.0,
                "Gather row 0 should have nonzero weight at col 0"
            );
            assert!(
                bounds.lower_a[[1, 2]].abs() > 0.0,
                "Gather row 1 should have nonzero weight at col 2"
            );
        }
        other => panic!("Expected Single, got {other:?}"),
    }
}

// ===================================================================
// ConvTranspose1d dispatch (#3448)
// ===================================================================

/// ConvTranspose1d dispatch: 1D transposed convolution returns Single.
#[test]
fn dispatch_conv_transpose1d_returns_single_3448() {
    // kernel: (in_channels=2, out_channels=1, kernel_size=3)
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![1.0_f32, 0.5, -0.25, 0.0, 1.0, -0.5])
            .unwrap();
    let bias = Array1::from_vec(vec![0.1_f32]);
    let conv =
        ConvTranspose1dLayer::new(kernel, Some(bias), 1, 0).expect("valid conv transpose 1d");
    let layer = Layer::ConvTranspose1d(conv);

    // Pre-activation: (in_channels=2, length=4)
    let pre_act = shaped_bounds(&[2, 4]);
    let flat_dim = 2 * 4;
    let net_input = simple_bounds(flat_dim);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs);
    // Output of ConvTranspose1d with stride=1, pad=0, kernel=3: length = 4 + 3 - 1 = 6
    // Output shape: (out_channels=1, length=6) → flat dim = 6
    let lb = identity_lb(6);

    let result =
        dispatch_backward_layer(&ctx, &lb).expect("ConvTranspose1d dispatch should succeed");
    assert!(
        matches!(result, BackwardDispatchResult::Single(_)),
        "ConvTranspose1d should return Single, got {result:?}"
    );
}

/// ConvTranspose1d dispatch: 1D input (below 2D requirement) returns error.
#[test]
fn dispatch_conv_transpose1d_1d_input_returns_error_3448() {
    let kernel = ArrayD::from_elem(IxDyn(&[2, 1, 3]), 0.0_f32);
    let conv = ConvTranspose1dLayer::new(kernel, None, 1, 0).expect("valid conv transpose 1d");
    let layer = Layer::ConvTranspose1d(conv);

    let pre_act = shaped_bounds(&[4]); // 1D — too few dimensions
    let net_input = simple_bounds(4);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs);
    let lb = identity_lb(4);

    let result = dispatch_backward_layer(&ctx, &lb);
    assert!(result.is_err(), "ConvTranspose1d with 1D input should fail");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("ConvTranspose1d") && err_msg.contains(">= 2D"),
        "Expected dimension error, got: {err_msg}"
    );
}

// ===================================================================
// ConvTranspose2d dispatch (#3448)
// ===================================================================

/// ConvTranspose2d dispatch: valid 3D input returns Single with shaped bounds.
#[test]
fn dispatch_conv_transpose2d_returns_single_3448() {
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0_f32, 0.5, -0.25, 0.75]).unwrap();
    let bias = Array1::from_vec(vec![0.2_f32]);
    let conv = ConvTranspose2dLayer::new(kernel, Some(bias), (1, 1), (0, 0))
        .expect("valid conv transpose 2d");
    let layer = Layer::ConvTranspose2d(conv);

    // Pre-activation: (channels=1, height=2, width=3).
    let pre_act = shaped_bounds(&[1, 2, 3]);
    let flat_input_dim = 2 * 3;
    let net_input = simple_bounds(flat_input_dim);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs);
    // Output shape: (1, 3, 4) => flat dim 12.
    let lb = identity_lb(12);

    let result =
        dispatch_backward_layer(&ctx, &lb).expect("ConvTranspose2d dispatch should succeed");
    match result {
        BackwardDispatchResult::Single(bounds) => {
            assert_eq!(bounds.lower_a.shape(), &[12, 6]);
            assert_eq!(bounds.upper_a.shape(), &[12, 6]);
            assert!(
                bounds.lower_a.iter().any(|v| v.abs() > 0.0),
                "ConvTranspose2d backward A-matrix should be non-trivial"
            );
        }
        other => panic!("Expected Single, got {other:?}"),
    }
}

/// ConvTranspose2d dispatch: 2D input (below 3D requirement) returns UnsupportedOp.
#[test]
fn dispatch_conv_transpose2d_2d_input_returns_unsupported_op_3448() {
    let kernel = ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), 0.0_f32);
    let conv =
        ConvTranspose2dLayer::new(kernel, None, (1, 1), (0, 0)).expect("valid conv transpose 2d");
    let layer = Layer::ConvTranspose2d(conv);

    let pre_act = shaped_bounds(&[4, 4]); // 2D — below 3D requirement
    let net_input = simple_bounds(16);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs);
    let lb = identity_lb(16);

    let err =
        dispatch_backward_layer(&ctx, &lb).expect_err("ConvTranspose2d with 2D input should fail");
    match err {
        ny_core::NyError::UnsupportedOp(msg) => {
            assert!(
                msg.contains("ConvTranspose2d") && msg.contains(">= 3D"),
                "Expected dimension error, got: {msg}"
            );
        }
        other => panic!("Expected UnsupportedOp, got: {other:?}"),
    }
}

// ===================================================================
// MatMul success-path dispatch (#3448)
// ===================================================================

/// MatMul dispatch: success path with valid 2-input bounds returns Binary.
#[test]
fn dispatch_matmul_success_returns_binary_3448() {
    let layer = Layer::MatMul(MatMulLayer::new(false, None));
    // MatMul: A [2, 3] @ B [3, 2] → [2, 2]
    let input_a = shaped_bounds(&[2, 3]);
    let input_b = shaped_bounds(&[3, 2]);
    let mut node_bounds = HashMap::new();
    node_bounds.insert("a".to_string(), input_a.clone());
    node_bounds.insert("b".to_string(), input_b);
    let inputs = vec!["a".to_string(), "b".to_string()];
    let ctx = make_ctx(&layer, &input_a, &input_a, &node_bounds, &inputs);
    // Output: [2, 2] → flat dim = 4
    let lb = identity_lb(4);

    let result = dispatch_backward_layer(&ctx, &lb).expect("MatMul dispatch should succeed");
    match result {
        BackwardDispatchResult::Binary {
            bounds_a,
            bounds_b,
            bias_lower,
            bias_upper,
        } => {
            // #2617: A-matrix bounds must have zero bias
            assert!(
                bounds_a.lower_b.iter().all(|&v| v == 0.0),
                "MatMul bounds_a lower_b should be zero"
            );
            assert!(
                bounds_a.upper_b.iter().all(|&v| v == 0.0),
                "MatMul bounds_a upper_b should be zero"
            );
            assert!(
                bounds_b.lower_b.iter().all(|&v| v == 0.0),
                "MatMul bounds_b lower_b should be zero"
            );
            assert!(
                bounds_b.upper_b.iter().all(|&v| v == 0.0),
                "MatMul bounds_b upper_b should be zero"
            );
            // Bias channel should have correct dimension
            assert_eq!(bias_lower.len(), 4, "MatMul bias_lower should have dim 4");
            assert_eq!(bias_upper.len(), 4, "MatMul bias_upper should have dim 4");
        }
        other => panic!("Expected Binary for MatMul success path, got {other:?}"),
    }
}

// ===================================================================
// BilinearCrown success-path dispatch (#3448)
// ===================================================================

/// BilinearCrown dispatch: success path with valid 2-input bounds returns Binary.
#[test]
fn dispatch_bilinear_success_returns_binary_3448() {
    let layer = Layer::BilinearCrown(BilinearCrownLayer::new(false, None));
    // BilinearCrown: A [2, 3] @ B [3, 2] → [2, 2]
    let input_a = shaped_bounds(&[2, 3]);
    let input_b = shaped_bounds(&[3, 2]);
    let mut node_bounds = HashMap::new();
    node_bounds.insert("a".to_string(), input_a.clone());
    node_bounds.insert("b".to_string(), input_b);
    let inputs = vec!["a".to_string(), "b".to_string()];
    let ctx = make_ctx(&layer, &input_a, &input_a, &node_bounds, &inputs);
    let lb = identity_lb(4);

    let result = dispatch_backward_layer(&ctx, &lb).expect("BilinearCrown dispatch should succeed");
    match result {
        BackwardDispatchResult::Binary {
            bounds_a,
            bounds_b,
            bias_lower,
            bias_upper,
        } => {
            // #2617: A-matrix bounds must have zero bias
            assert!(
                bounds_a.lower_b.iter().all(|&v| v == 0.0),
                "BilinearCrown bounds_a lower_b should be zero"
            );
            assert!(
                bounds_a.upper_b.iter().all(|&v| v == 0.0),
                "BilinearCrown bounds_a upper_b should be zero"
            );
            assert!(
                bounds_b.lower_b.iter().all(|&v| v == 0.0),
                "BilinearCrown bounds_b lower_b should be zero"
            );
            assert!(
                bounds_b.upper_b.iter().all(|&v| v == 0.0),
                "BilinearCrown bounds_b upper_b should be zero"
            );
            assert_eq!(
                bias_lower.len(),
                4,
                "BilinearCrown bias_lower should have dim 4"
            );
            assert_eq!(
                bias_upper.len(),
                4,
                "BilinearCrown bias_upper should have dim 4"
            );
        }
        other => panic!("Expected Binary for BilinearCrown, got {other:?}"),
    }
}

// ===================================================================
// Conv2d dispatch (#3720): verify dispatch_conv_engine_aware path
// ===================================================================

/// Conv2d dispatch: success path with valid 3D input returns Single.
/// #3720: Conv2d now uses dispatch_conv_engine_aware (same as Conv1d,
/// ConvTranspose1d, ConvTranspose2d) instead of inline map_err.
#[test]
fn dispatch_conv2d_returns_single_3720() {
    // kernel: (out_channels=1, in_channels=1, h=1, w=1) — simplest valid conv
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0_f32]).unwrap();
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();
    let layer = Layer::Conv2d(conv);

    // Pre-activation: (channels=1, height=3, width=3) — 3D
    let pre_act = shaped_bounds(&[1, 3, 3]);
    let flat_dim = 3 * 3;
    let net_input = simple_bounds(flat_dim);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs);
    // Conv2d with 1x1 kernel, stride=1, pad=0: output same shape [1, 3, 3]
    let lb = identity_lb(flat_dim);

    let result = dispatch_backward_layer(&ctx, &lb).expect("Conv2d dispatch should succeed");
    assert!(
        matches!(result, BackwardDispatchResult::Single(_)),
        "Conv2d should return Single, got {result:?}"
    );
}

/// Conv2d dispatch: 2D input (below 3D requirement) returns UnsupportedOp.
/// #3720: After refactor to dispatch_conv_engine_aware, the dimension check
/// returns UnsupportedOp (structured) instead of being inline.
#[test]
fn dispatch_conv2d_2d_input_returns_unsupported_op_3720() {
    let kernel = ArrayD::from_elem(IxDyn(&[1, 1, 3, 3]), 0.0_f32);
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();
    let layer = Layer::Conv2d(conv);

    let pre_act = shaped_bounds(&[4, 4]); // 2D — below 3D requirement
    let net_input = simple_bounds(16);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs);
    let lb = identity_lb(16);

    let err = dispatch_backward_layer(&ctx, &lb).expect_err("Conv2d with 2D input should fail");
    match err {
        ny_core::NyError::UnsupportedOp(msg) => {
            assert!(
                msg.contains("Conv2d") && msg.contains(">= 3D"),
                "Expected dimension error, got: {msg}"
            );
        }
        other => panic!("Expected UnsupportedOp (not InvalidSpec), got: {other:?}"),
    }
}
