// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::BetaCrownVerifier;
#[ntest::timeout(5000)]
#[test]
fn test_infer_conv2d_input_hw_from_last3_channels_first() {
    let hw = BetaCrownVerifier::infer_conv2d_input_hw(&[2, 3, 4], 2, "Conv2d").unwrap();
    assert_eq!((3, 4), hw);
}

#[ntest::timeout(5000)]
#[test]
fn test_infer_conv2d_input_hw_from_last3_channels_last() {
    let hw = BetaCrownVerifier::infer_conv2d_input_hw(&[3, 4, 2], 2, "Conv2d").unwrap();
    assert_eq!((3, 4), hw);
}

#[ntest::timeout(5000)]
#[test]
fn test_infer_conv2d_input_hw_from_flattened_features() {
    let hw = BetaCrownVerifier::infer_conv2d_input_hw(&[18, 2], 2, "Conv2d").unwrap();
    assert_eq!((3, 3), hw);
}

#[ntest::timeout(5000)]
#[test]
fn test_infer_conv2d_input_hw_from_flattened_batch_features() {
    let hw = BetaCrownVerifier::infer_conv2d_input_hw(&[2, 32], 2, "Conv2d").unwrap();
    assert_eq!((4, 4), hw);
}

#[ntest::timeout(5000)]
#[test]
fn test_infer_conv2d_input_hw_prefers_last_dim_when_valid() {
    let hw = BetaCrownVerifier::infer_conv2d_input_hw(&[72, 32], 2, "Conv2d").unwrap();
    assert_eq!((4, 4), hw);
}

#[ntest::timeout(5000)]
#[test]
fn test_infer_conv2d_input_hw_prefers_non_channel_dim_when_last_is_channels() {
    let hw = BetaCrownVerifier::infer_conv2d_input_hw(&[18, 2], 2, "Conv2d").unwrap();
    assert_eq!((3, 3), hw);
}

#[ntest::timeout(5000)]
#[test]
fn test_infer_conv2d_input_hw_from_single_flattened_dim() {
    let hw = BetaCrownVerifier::infer_conv2d_input_hw(&[18], 2, "Conv2d").unwrap();
    assert_eq!((3, 3), hw);
}

#[ntest::timeout(5000)]
#[test]
fn test_infer_conv2d_input_hw_ambiguous_channels() {
    let hw = BetaCrownVerifier::infer_conv2d_input_hw(&[2, 2, 4], 2, "Conv2d").unwrap();
    assert_eq!((2, 4), hw);
}

#[ntest::timeout(5000)]
#[test]
fn test_infer_conv1d_input_len_from_flattened_batch_features() {
    let len = BetaCrownVerifier::infer_conv1d_input_len(&[2, 48], 3, "Conv1d").unwrap();
    assert_eq!(16, len);
}

#[ntest::timeout(5000)]
#[test]
fn test_infer_conv1d_input_len_prefers_non_channel_dim() {
    let len = BetaCrownVerifier::infer_conv1d_input_len(&[32, 2], 2, "Conv1d").unwrap();
    assert_eq!(32, len);
}

#[ntest::timeout(5000)]
#[test]
fn test_infer_conv1d_input_len_from_single_flattened_dim() {
    let len = BetaCrownVerifier::infer_conv1d_input_len(&[24], 3, "Conv1d").unwrap();
    assert_eq!(8, len);
}

#[ntest::timeout(5000)]
#[test]
fn test_infer_conv2d_input_hw_errors_when_ambiguous() {
    let err = BetaCrownVerifier::infer_conv2d_input_hw(&[7, 5], 2, "Conv2d").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("inferrable input H/W"),
        "unexpected error message: {}",
        msg
    );
    assert!(
        msg.contains("Conv2d"),
        "missing layer label in error message: {}",
        msg
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_infer_conv1d_input_len_errors_when_invalid() {
    let err = BetaCrownVerifier::infer_conv1d_input_len(&[7], 2, "Conv1d").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("inferrable input length"),
        "unexpected error message: {}",
        msg
    );
    assert!(
        msg.contains("Conv1d"),
        "missing layer label in error message: {}",
        msg
    );
}

/// Regression test for #2827: zero in_channels must return NyError, not panic.
#[ntest::timeout(5000)]
#[test]
fn test_infer_conv2d_input_hw_zero_channels_returns_error() {
    let err = BetaCrownVerifier::infer_conv2d_input_hw(&[3, 28, 28], 0, "Conv2d").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("in_channels must be > 0"),
        "expected zero-channel guard, got: {msg}"
    );
}

/// Regression test for #2827: zero in_channels for conv1d must return error.
#[ntest::timeout(5000)]
#[test]
fn test_infer_conv1d_input_len_zero_channels_returns_error() {
    let err = BetaCrownVerifier::infer_conv1d_input_len(&[48], 0, "Conv1d").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("in_channels must be > 0"),
        "expected zero-channel guard, got: {msg}"
    );
}

/// Regression test for #2827: zero in_channels with 2D shape also returns error.
#[ntest::timeout(5000)]
#[test]
fn test_infer_conv2d_input_hw_zero_channels_2d_shape_returns_error() {
    let err =
        BetaCrownVerifier::infer_conv2d_input_hw(&[32, 18], 0, "ConvTranspose2d").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("in_channels must be > 0"),
        "expected zero-channel guard, got: {msg}"
    );
}

/// Regression test for #2176: ReLU backward must return an error on dimension
/// mismatch instead of silently returning unsound identity bounds.
#[ntest::timeout(5000)]
#[test]
fn test_relu_backward_returns_error_on_dimension_mismatch() {
    use crate::beta_crown::state::{BetaState, DomainAlphaState};
    use crate::LinearBounds;
    use ndarray::arr1;
    use ny_core::NyError;
    use ny_tensor::BoundedTensor;

    let verifier = BetaCrownVerifier::new(Default::default());

    // output_bounds with num_inputs=3, but pre_bounds with 2 elements → mismatch
    let output_bounds = LinearBounds::identity(3);
    let pre_bounds = BoundedTensor::new(
        arr1(&[-1.0f32, 1.0]).into_dyn(),
        arr1(&[0.0f32, 2.0]).into_dyn(),
    )
    .unwrap();
    let beta_state = BetaState {
        entries: vec![],
        slow_weights: None,
        ..BetaState::empty()
    };
    let alpha_state = DomainAlphaState::empty();

    let err = verifier
        .relu_backward_with_alpha_beta(
            &output_bounds,
            &pre_bounds,
            None,
            &beta_state,
            &alpha_state,
            0,
        )
        .unwrap_err();

    match err {
        NyError::InternalError(msg) => {
            assert!(
                msg.contains("dimension mismatch"),
                "expected dimension mismatch error, got: {msg}"
            );
            assert!(
                msg.contains("3 inputs"),
                "expected input count in error: {msg}"
            );
            assert!(
                msg.contains("2 neurons"),
                "expected neuron count in error: {msg}"
            );
        }
        other => panic!("expected InternalError, got {other}"),
    }
}

/// Regression test for #2184: UnsupportedOp CROWN backward fallback must
/// return constant (A=0) bounds derived from IBP, not identity(dim).
///
/// GatherLayer has IBP support but returns UnsupportedOp for CROWN backward.
/// When input_dim=3 and output_dim=2, the old code returned identity(2), which
/// was unsound: it said "output = post-activation input" (ignoring accumulated
/// linear bounds from later layers). The fix concretizes accumulated bounds
/// through the IBP result and returns A=0, b=concretized.
#[ntest::timeout(5000)]
#[test]
fn test_unsupported_layer_backward_returns_constant_not_identity_2184() {
    let (result, ()) = propagate_gather_backward_fixture(&[1.0, 1.0], 0.0, &[1.0, 1.0], 0.0);

    // Result: (1 output, 3 inputs) = (num_outputs, pre_bounds.len()).
    assert_eq!(result.num_outputs(), 1, "should preserve output count");
    assert_eq!(
        result.num_inputs(),
        3,
        "should map to pre-activation input space"
    );

    // A matrices must be all zeros (constant bounds, no input dependence).
    assert!(
        result.lower_a.iter().all(|&v| v == 0.0),
        "lower_a should be zero"
    );
    assert!(
        result.upper_a.iter().all(|&v| v == 0.0),
        "upper_a should be zero"
    );

    // Gather picks indices [0, 2] from [1..10, 2..20, 3..30] => output [1..10, 3..30].
    // Concretization: lower = 1*1 + 1*3 = 4, upper = 1*10 + 1*30 = 40.
    assert!(
        result.lower_b[0] <= 4.0 + 1e-5,
        "lower_b: got {}",
        result.lower_b[0]
    );
    assert!(
        result.upper_b[0] >= 40.0 - 1e-5,
        "upper_b: got {}",
        result.upper_b[0]
    );
}

/// Regression test for #2184: concretization in UnsupportedOp fallback must use
/// signed interval arithmetic (negative coefficients take upper for lower bound
/// and lower for upper bound).
#[ntest::timeout(5000)]
#[test]
fn test_unsupported_layer_backward_concretization_signed_coefficients_2184() {
    // Signed coefficients: Lower = 1 + 2*1 + (-3)*30 = -87, Upper = 2 + 4*10 + (-1)*3 = 39
    let (result, ()) = propagate_gather_backward_fixture(&[2.0, -3.0], 1.0, &[4.0, -1.0], 2.0);
    assert_concretized_zero_weight(&result, 3, -87.0, 39.0);
}

/// Shared fixture: Gather(axis=0, indices=[0,2]) on 3-element input [1..10, 2..20, 3..30].
/// Returns the propagated `LinearBounds` result.
fn propagate_gather_backward_fixture(
    lower_a: &[f32],
    lower_b: f32,
    upper_a: &[f32],
    upper_b: f32,
) -> (crate::LinearBounds, ()) {
    use crate::beta_crown::state::{BetaState, DomainAlphaState};
    use crate::layers::{GatherLayer, Layer};
    use crate::LinearBounds;
    use ndarray::{arr1, Array2, ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    let verifier = BetaCrownVerifier::new(Default::default());
    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0i64, 2]).unwrap();
    let gather = Layer::Gather(GatherLayer::new(0, Some(indices), vec![]));
    let pre_bounds = BoundedTensor::new(
        arr1(&[1.0_f32, 2.0, 3.0]).into_dyn(),
        arr1(&[10.0_f32, 20.0, 30.0]).into_dyn(),
    )
    .unwrap();
    let ncols = lower_a.len();
    let output_bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, ncols), lower_a.to_vec()).unwrap(),
        lower_b: arr1(&[lower_b]),
        upper_a: Array2::from_shape_vec((1, ncols), upper_a.to_vec()).unwrap(),
        upper_b: arr1(&[upper_b]),
        lower_a_err: None,
        upper_a_err: None,
    };
    let beta_state = BetaState::empty();
    let alpha_state = DomainAlphaState::empty();
    let result = verifier
        .propagate_layer_backward_with_alpha_beta(
            &gather,
            &output_bounds,
            &pre_bounds,
            None,
            &beta_state,
            &alpha_state,
            0,
            None,
        )
        .expect("UnsupportedOp fallback should concretize successfully");
    (result, ())
}

/// Assert that a CROWN backward result has zero-weight A matrices and
/// expected concretized scalar bounds in b vectors.
fn assert_concretized_zero_weight(
    result: &crate::LinearBounds,
    num_inputs: usize,
    expected_lower: f32,
    expected_upper: f32,
) {
    assert_eq!(result.num_outputs(), 1);
    assert_eq!(result.num_inputs(), num_inputs);
    assert!(
        result.lower_a.iter().all(|&v| v == 0.0),
        "lower_a all zeros"
    );
    assert!(
        result.upper_a.iter().all(|&v| v == 0.0),
        "upper_a all zeros"
    );
    assert!(
        (result.lower_b[0] - expected_lower).abs() <= 1e-5,
        "lower_b: expected {expected_lower}, got {}",
        result.lower_b[0]
    );
    assert!(
        (result.upper_b[0] - expected_upper).abs() <= 1e-5,
        "upper_b: expected {expected_upper}, got {}",
        result.upper_b[0]
    );
}
