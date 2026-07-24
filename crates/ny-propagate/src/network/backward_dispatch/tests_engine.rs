// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GemmEngine-aware backward dispatch tests (#3959).

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::GemmEngine;

use super::dispatch::dispatch_backward_layer;
use super::types::{BackwardDispatchResult, DispatchContext};
use crate::bounds::LinearBounds;
use crate::layers::{Conv1dLayer, Conv2dLayer, ConvTranspose2dLayer, Layer, LinearLayer};
use crate::tests::assert_linear_bounds_close;
use crate::MulBinaryRelaxationMode;
use ny_tensor::BoundedTensor;
use ny_test_utils::CountingGemmEngine;

fn identity_lb(dim: usize) -> LinearBounds {
    LinearBounds::identity(dim)
}

fn simple_bounds(dim: usize) -> BoundedTensor {
    let lower = ArrayD::from_elem(IxDyn(&[dim]), -1.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[dim]), 1.0_f32);
    BoundedTensor::new(lower, upper).unwrap()
}

fn sample_conv_transpose2d() -> ConvTranspose2dLayer {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0_f32, -0.5, 0.25, 2.0])
        .expect("valid kernel shape");
    let bias = Array1::from_vec(vec![0.75_f32]);
    ConvTranspose2dLayer::new(kernel, Some(bias), (1, 1), (0, 0)).expect("valid conv transpose")
}

fn sample_conv_transpose2d_pre_act() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![-1.0_f32, 0.0, 0.5, 1.0])
            .expect("valid lower input"),
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.5_f32, 2.0, 2.5, 3.0])
            .expect("valid upper input"),
    )
    .expect("valid bounded tensor")
}

fn sample_conv_transpose2d_node_lb() -> LinearBounds {
    LinearBounds::new(
        Array2::from_shape_vec(
            (2, 9),
            vec![
                1.0, -0.5, 0.25, 0.0, 1.5, -1.0, 0.75, 0.2, -0.3, -0.1, 0.4, 1.25, -0.75, 0.5, 0.6,
                -1.2, 0.8, 0.9,
            ],
        )
        .expect("valid lower_a"),
        Array1::from_vec(vec![0.1_f32, -0.3]),
        Array2::from_shape_vec(
            (2, 9),
            vec![
                0.9, 0.1, -0.25, 0.7, -1.5, 0.4, 0.0, 1.1, -0.8, 1.2, -0.6, 0.3, 0.5, 0.9, -1.4,
                0.2, 0.75, 0.6,
            ],
        )
        .expect("valid upper_a"),
        Array1::from_vec(vec![0.4_f32, 0.2]),
    )
    .expect("valid linear bounds")
}

fn make_ctx<'a>(
    layer: &'a Layer,
    pre_act: &'a BoundedTensor,
    net_input: &'a BoundedTensor,
    node_bounds: &'a HashMap<String, BoundedTensor>,
    inputs: &'a [String],
    engine: Option<&'a dyn GemmEngine>,
) -> DispatchContext<'a> {
    DispatchContext {
        node_name: "test_node",
        layer,
        inputs,
        pre_activation: pre_act,
        network_input: net_input,
        node_bounds: node_bounds.into(),
        engine,
        deadline: None,
        bilinear_alphas: None,
        mul_binary_relaxation: MulBinaryRelaxationMode::default(),
        mul_binary_alphas: None,
        norm_inv_rms_override: None,
    }
}

#[test]
fn dispatch_linear_with_engine_matches_none_baseline_3959() {
    let weight =
        Array2::from_shape_vec((3, 3), vec![0.5, -0.3, 0.1, -0.2, 0.8, 0.4, 0.3, -0.1, 0.6])
            .unwrap();
    let bias = Array1::from_vec(vec![0.1, -0.2, 0.05]);
    let layer = Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap());

    let bounds = simple_bounds(3);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let lb = identity_lb(3);

    let ctx_none = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs, None);
    let result_none = dispatch_backward_layer(&ctx_none, &lb).unwrap();
    let bounds_none = match result_none {
        BackwardDispatchResult::Single(b) => *b,
        other => panic!("Expected Single, got {other:?}"),
    };

    let engine = CountingGemmEngine::new();
    let ctx_eng = make_ctx(
        &layer,
        &bounds,
        &bounds,
        &node_bounds,
        &inputs,
        Some(&engine),
    );
    let result_eng = dispatch_backward_layer(&ctx_eng, &lb).unwrap();
    let bounds_eng = match result_eng {
        BackwardDispatchResult::Single(b) => *b,
        other => panic!("Expected Single, got {other:?}"),
    };

    assert!(
        engine.gemm_calls() > 0,
        "GemmEngine should be invoked for Linear dispatch, got 0 calls"
    );

    let tol = 1e-5;
    for row in 0..3 {
        for col in 0..3 {
            assert!(
                (bounds_eng.lower_a()[[row, col]] - bounds_none.lower_a()[[row, col]]).abs() < tol,
                "lower_a mismatch at [{row}, {col}]"
            );
            assert!(
                (bounds_eng.upper_a()[[row, col]] - bounds_none.upper_a()[[row, col]]).abs() < tol,
                "upper_a mismatch at [{row}, {col}]"
            );
        }
        assert!(
            (bounds_eng.lower_b()[row] - bounds_none.lower_b()[row]).abs() < tol,
            "lower_b mismatch at [{row}]"
        );
        assert!(
            (bounds_eng.upper_b()[row] - bounds_none.upper_b()[row]).abs() < tol,
            "upper_b mismatch at [{row}]"
        );
    }
}

#[test]
fn dispatch_conv2d_with_engine_matches_none_baseline_3959() {
    // Serialized on the shared env lock: the dead-work skip bypasses the
    // GemmEngine entirely, and since the 2026-07-20 default flip the skip is
    // ON when `NY_CONV_SKIP_DEAD_F32` is unset — so this engine-vs-none
    // baseline test must PIN THE KILL-SWITCH ("0") to exercise the pair path
    // its `gemm_calls() > 0` assertion requires (#wall-deadwork).
    crate::tests::with_serialized_env_vars(&[("NY_CONV_SKIP_DEAD_F32", "0")], || {
        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![2.0_f32]).unwrap();
        let bias = Array1::from_vec(vec![0.5_f32]);
        let layer = Layer::Conv2d(Conv2dLayer::new(kernel, Some(bias), (1, 1), (0, 0)).unwrap());

        let pre_act = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 2, 2]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[1, 2, 2]), 1.0_f32),
        )
        .unwrap();
        let net_input = simple_bounds(4);
        let node_bounds = HashMap::new();
        let inputs = vec!["_input".to_string()];
        let lb = identity_lb(4);

        let ctx_none = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs, None);
        let result_none = dispatch_backward_layer(&ctx_none, &lb).unwrap();
        let bounds_none = match result_none {
            BackwardDispatchResult::Single(b) => *b,
            other => panic!("Expected Single, got {other:?}"),
        };

        let engine = CountingGemmEngine::new();
        let ctx_eng = make_ctx(
            &layer,
            &pre_act,
            &net_input,
            &node_bounds,
            &inputs,
            Some(&engine),
        );
        let result_eng = dispatch_backward_layer(&ctx_eng, &lb).unwrap();
        let bounds_eng = match result_eng {
            BackwardDispatchResult::Single(b) => *b,
            other => panic!("Expected Single, got {other:?}"),
        };

        assert!(
            engine.gemm_calls() > 0,
            "GemmEngine should be invoked for Conv2d dispatch, got 0 calls"
        );

        assert_linear_bounds_close(&bounds_eng, &bounds_none, 1e-5, "Conv2d engine path");
    });
}

#[test]
fn dispatch_conv_transpose2d_with_engine_matches_none_baseline_3959() {
    // Serialized on the shared env lock: the ConvTranspose dead-work skip
    // (#wall-deadwork port) bypasses the GemmEngine f32 pair entirely and is
    // ON when `NY_CONV_SKIP_DEAD_F32` is unset — pin the kill-switch ("0") to
    // exercise the pair path this test's `gemm_calls() > 0` assertion needs.
    crate::tests::with_serialized_env_vars(&[("NY_CONV_SKIP_DEAD_F32", "0")], || {
        let layer = Layer::ConvTranspose2d(sample_conv_transpose2d());
        let pre_act = sample_conv_transpose2d_pre_act();
        let net_input = pre_act.clone();
        let node_bounds = HashMap::new();
        let inputs = vec!["_input".to_string()];
        let node_lb = sample_conv_transpose2d_node_lb();

        let ctx_none = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs, None);
        let result_none = dispatch_backward_layer(&ctx_none, &node_lb).unwrap();
        let bounds_none = match result_none {
            BackwardDispatchResult::Single(b) => *b,
            other => panic!("Expected Single, got {other:?}"),
        };

        let engine = CountingGemmEngine::new();
        let ctx_eng = make_ctx(
            &layer,
            &pre_act,
            &net_input,
            &node_bounds,
            &inputs,
            Some(&engine),
        );
        let result_eng = dispatch_backward_layer(&ctx_eng, &node_lb).unwrap();
        let bounds_eng = match result_eng {
            BackwardDispatchResult::Single(b) => *b,
            other => panic!("Expected Single, got {other:?}"),
        };

        assert!(
            engine.gemm_calls() > 0,
            "GemmEngine should be invoked for ConvTranspose2d dispatch, got 0 calls"
        );

        assert_linear_bounds_close(
            &bounds_eng,
            &bounds_none,
            1e-4,
            "ConvTranspose2d engine path",
        );
    });
}

#[test]
fn dispatch_conv1d_with_engine_matches_none_baseline_3959() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 1, 2]), vec![0.5_f32, -0.3, 0.2, 0.8]).unwrap();
    let bias = Array1::from_vec(vec![0.1_f32, -0.1]);
    let layer = Layer::Conv1d(Conv1dLayer::new(kernel, Some(bias), 1, 0).unwrap());

    let pre_act = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 4]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 4]), 1.0_f32),
    )
    .unwrap();
    let net_input = simple_bounds(4);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let lb = identity_lb(6);

    let ctx_none = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs, None);
    let result_none = dispatch_backward_layer(&ctx_none, &lb).unwrap();
    let bounds_none = match result_none {
        BackwardDispatchResult::Single(b) => *b,
        other => panic!("Expected Single, got {other:?}"),
    };

    let engine = CountingGemmEngine::new();
    let ctx_eng = make_ctx(
        &layer,
        &pre_act,
        &net_input,
        &node_bounds,
        &inputs,
        Some(&engine),
    );
    let result_eng = dispatch_backward_layer(&ctx_eng, &lb).unwrap();
    let bounds_eng = match result_eng {
        BackwardDispatchResult::Single(b) => *b,
        other => panic!("Expected Single, got {other:?}"),
    };

    assert!(
        engine.gemm_calls() > 0,
        "GemmEngine should be invoked for Conv1d dispatch, got 0 calls"
    );

    assert_linear_bounds_close(&bounds_eng, &bounds_none, 1e-5, "Conv1d engine path");
}
