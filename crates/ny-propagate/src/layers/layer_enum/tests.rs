// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::softmax::GeluApproximation;
use super::*;
use crate::BatchedLinearBounds;
use ndarray::{arr1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

// -- Helper: create simple test layers --

fn relu() -> Layer {
    Layer::ReLU(ReLULayer::new())
}
fn exp() -> Layer {
    Layer::Exp(ExpLayer::new())
}
fn add_binary() -> Layer {
    Layer::Add(AddLayer)
}
fn matmul() -> Layer {
    Layer::MatMul(MatMulLayer::new(false, None))
}
fn where_l() -> Layer {
    Layer::Where(WhereLayer::new())
}
fn self_attn() -> Layer {
    Layer::SelfAttention(SelfAttentionLayer::standard())
}
fn linear() -> Layer {
    Layer::Linear(LinearLayer::new(Array2::eye(2), Some(ndarray::Array1::zeros(2))).unwrap())
}
fn softmax() -> Layer {
    Layer::Softmax(SoftmaxLayer::new(-1))
}
fn flatten() -> Layer {
    Layer::Flatten(FlattenLayer::new(1))
}
fn transpose_l() -> Layer {
    Layer::Transpose(TransposeLayer::transpose_2d())
}
fn add_const() -> Layer {
    Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(
        IxDyn(&[1]),
        1.0_f32,
    )))
}
fn slice_l() -> Layer {
    Layer::Slice(SliceLayer::new(0, 0, 2))
}
fn tile_l() -> Layer {
    Layer::Tile(TileLayer::new(0, 2))
}
fn expand_like_l() -> Layer {
    Layer::ExpandLikeLastAxis(ExpandLikeLastAxisLayer::new())
}
fn reduce_sum() -> Layer {
    Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true))
}
fn reduce_mean() -> Layer {
    Layer::ReduceMean(ReduceMeanLayer::new(vec![-1], true))
}
fn reduce_max() -> Layer {
    Layer::ReduceMax(ReduceMaxLayer::new(vec![-1], true))
}
fn reduce_min() -> Layer {
    Layer::ReduceMin(ReduceMinLayer::new(vec![-1], true))
}
fn topk() -> Layer {
    Layer::Topk(TopkLayer::values(2, -1))
}
fn scatter_add_binary() -> Layer {
    Layer::ScatterAdd(ScatterAddLayer::new(
        -1,
        Some(ArrayD::zeros(IxDyn(&[4]))),
        None,
        None,
    ))
}
fn index_add_ternary() -> Layer {
    Layer::IndexAdd(IndexAddLayer::new(-1, None, None, None))
}

fn assert_layer_is_ternary(layer: Layer, expected: bool) {
    let layer_type = layer.layer_type().to_owned();
    assert_eq!(
        layer.is_ternary(),
        expected,
        "{layer_type} ternary classification mismatch"
    );
}

fn assert_layer_supports_batched_crown(layer: Layer, expected: bool) {
    let layer_type = layer.layer_type().to_owned();
    assert_eq!(
        layer.supports_batched_crown(),
        expected,
        "{layer_type} supports_batched_crown() mismatch"
    );
}

fn assert_layer_supports_batched_crown_with_conv2d(layer: Layer, expected: bool) {
    let layer_type = layer.layer_type().to_owned();
    assert_eq!(
        layer.supports_batched_crown_with_conv2d(),
        expected,
        "{layer_type} supports_batched_crown_with_conv2d() mismatch"
    );
}

// -- layer_type() --

#[test]
fn test_layer_type_activations() {
    assert_eq!(relu().layer_type(), "ReLU");
    assert_eq!(exp().layer_type(), "Exp");
    assert_eq!(Layer::SiLU(SiLULayer::new()).layer_type(), "SiLU");
    assert_eq!(
        Layer::GELU(GELULayer::new(GeluApproximation::Erf)).layer_type(),
        "GELU"
    );
    assert_eq!(Layer::Tanh(TanhLayer::new()).layer_type(), "Tanh");
    assert_eq!(Layer::Sigmoid(SigmoidLayer::new()).layer_type(), "Sigmoid");
}

#[test]
fn test_layer_type_binary_ops() {
    assert_eq!(add_binary().layer_type(), "Add");
    assert_eq!(matmul().layer_type(), "MatMul");
    assert_eq!(Layer::Sub(SubLayer).layer_type(), "Sub");
    assert_eq!(Layer::Div(DivLayer).layer_type(), "Div");
    assert_eq!(Layer::Atan2(Atan2Layer).layer_type(), "Atan2");
    assert_eq!(Layer::MulBinary(MulBinaryLayer).layer_type(), "MulBinary");
    assert_eq!(
        Layer::BilinearCrown(BilinearCrownLayer::new(false, None)).layer_type(),
        "BilinearCrown"
    );
    assert_eq!(Layer::MinBinary(MinBinaryLayer).layer_type(), "MinBinary");
    assert_eq!(Layer::MaxBinary(MaxBinaryLayer).layer_type(), "MaxBinary");
    assert_eq!(Layer::Concat(ConcatLayer::new(0)).layer_type(), "Concat");
    assert_eq!(expand_like_l().layer_type(), "ExpandLikeLastAxis");
}

#[test]
fn test_layer_type_transforms() {
    assert_eq!(flatten().layer_type(), "Flatten");
    assert_eq!(transpose_l().layer_type(), "Transpose");
    assert_eq!(
        Layer::Reshape(ReshapeLayer::new(vec![2, 3])).layer_type(),
        "Reshape"
    );
    assert_eq!(Layer::Squeeze(SqueezeLayer::new(1)).layer_type(), "Squeeze");
    assert_eq!(
        Layer::Unsqueeze(UnsqueezeLayer::new(0)).layer_type(),
        "Unsqueeze"
    );
    assert_eq!(scatter_add_binary().layer_type(), "ScatterAdd");
    assert_eq!(index_add_ternary().layer_type(), "IndexAdd");
    assert_eq!(topk().layer_type(), "Topk");
}

// -- is_binary() --

#[test]
fn test_is_binary_true_for_all_binary_ops() {
    let binary_layers = vec![
        Layer::MatMul(MatMulLayer::new(false, None)),
        Layer::MulBinary(MulBinaryLayer),
        Layer::Add(AddLayer),
        Layer::Concat(ConcatLayer::new(0)),
        Layer::Sub(SubLayer),
        Layer::Div(DivLayer),
        Layer::Atan2(Atan2Layer),
        Layer::BilinearCrown(BilinearCrownLayer::new(false, None)),
        Layer::MinBinary(MinBinaryLayer),
        Layer::MaxBinary(MaxBinaryLayer),
        expand_like_l(),
        scatter_add_binary(),
    ];
    for layer in &binary_layers {
        assert!(
            layer.is_binary(),
            "Expected is_binary()=true for {}",
            layer.layer_type()
        );
    }
}

#[test]
fn test_is_binary_false_for_unary_layers() {
    let unary_layers = vec![relu(), exp(), linear(), softmax(), flatten(), add_const()];
    for layer in &unary_layers {
        assert!(
            !layer.is_binary(),
            "Expected is_binary()=false for {}",
            layer.layer_type()
        );
    }
}

// -- is_ternary() --

#[test]
fn test_is_ternary_true_for_ternary_ops() {
    assert_layer_is_ternary(where_l(), true);
    assert_layer_is_ternary(self_attn(), true);
    assert_layer_is_ternary(index_add_ternary(), true);
}

#[test]
fn test_is_ternary_false_for_binary_and_unary() {
    assert_layer_is_ternary(relu(), false);
    assert_layer_is_ternary(add_binary(), false);
    assert_layer_is_ternary(matmul(), false);
    assert_layer_is_ternary(linear(), false);
    assert_layer_is_ternary(topk(), false);
}

#[test]
fn test_min_inputs_tracks_accumulate_activation_arity() {
    let unary_scatter = Layer::ScatterAdd(ScatterAddLayer::new(
        -1,
        Some(ArrayD::zeros(IxDyn(&[4]))),
        Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![0_i64, 1]).unwrap()),
        None,
    ));
    assert_eq!(unary_scatter.min_inputs(), 1);
    assert_eq!(scatter_add_binary().min_inputs(), 2);
    assert_eq!(index_add_ternary().min_inputs(), 3);
}

// -- is_elementwise_activation() --

#[test]
fn test_is_elementwise_activation_true() {
    let activations = vec![
        Layer::ReLU(ReLULayer::new()),
        Layer::GELU(GELULayer::new(GeluApproximation::Erf)),
        Layer::SiLU(SiLULayer::new()),
        Layer::Tanh(TanhLayer::new()),
        Layer::Sigmoid(SigmoidLayer::new()),
        Layer::Exp(ExpLayer::new()),
        Layer::Log(LogLayer::new()),
        Layer::Softplus(SoftplusLayer::new()),
        Layer::HardSwish(HardSwishLayer::new()),
        Layer::Mish(MishLayer::new()),
        Layer::Softsign(SoftsignLayer::new()),
        Layer::Snake(SnakeLayer::new(1.0).expect("test: valid Snake")),
        Layer::Abs(AbsLayer::new()),
        Layer::Floor(FloorLayer),
        Layer::Ceil(CeilLayer),
        Layer::Round(RoundLayer),
        Layer::Sign(SignLayer),
        Layer::Reciprocal(ReciprocalLayer::new()),
    ];
    for layer in &activations {
        assert!(
            layer.is_elementwise_activation(),
            "Expected is_elementwise_activation()=true for {}",
            layer.layer_type()
        );
    }
}

#[test]
fn test_is_elementwise_activation_false_for_non_elementwise() {
    let non_elementwise = vec![
        softmax(),
        linear(),
        flatten(),
        add_binary(),
        matmul(),
        add_const(),
    ];
    for layer in &non_elementwise {
        assert!(
            !layer.is_elementwise_activation(),
            "Expected is_elementwise_activation()=false for {}",
            layer.layer_type()
        );
    }
}

// -- supports_batched_crown() --

#[test]
fn test_supports_batched_crown_includes_expected_layers() {
    assert_layer_supports_batched_crown(relu(), true);
    assert_layer_supports_batched_crown(exp(), true);
    assert_layer_supports_batched_crown(linear(), true);
    assert_layer_supports_batched_crown(softmax(), true);
    assert_layer_supports_batched_crown(Layer::CausalSoftmax(CausalSoftmaxLayer::new(-1)), true);
    assert_layer_supports_batched_crown(Layer::LogSoftmax(LogSoftmaxLayer::new(-1)), true);
    assert_layer_supports_batched_crown(
        Layer::LogSumExp(LogSumExpLayer::new(vec![-1], true)),
        true,
    );
    assert_layer_supports_batched_crown(flatten(), true);
    assert_layer_supports_batched_crown(transpose_l(), true);
    assert_layer_supports_batched_crown(add_const(), true);
    // Tile, Reduction, and Slice ops (#287)
    assert_layer_supports_batched_crown(tile_l(), true);
    assert_layer_supports_batched_crown(slice_l(), true);
    assert_layer_supports_batched_crown(reduce_sum(), true);
    assert_layer_supports_batched_crown(reduce_mean(), true);
    assert_layer_supports_batched_crown(reduce_max(), true);
    assert_layer_supports_batched_crown(reduce_min(), true);
    // BatchNorm: has batched CROWN dispatch + implementation (#3281)
    let bn = BatchNormLayer::from_scale_bias(
        ArrayD::from_elem(IxDyn(&[2]), 1.0_f32),
        ArrayD::from_elem(IxDyn(&[2]), 0.0_f32),
    )
    .unwrap();
    assert_layer_supports_batched_crown(Layer::BatchNorm(bn), true);
}

#[test]
fn test_supports_batched_crown_excludes_binary_ops() {
    assert_layer_supports_batched_crown(add_binary(), false);
    assert_layer_supports_batched_crown(matmul(), false);
    assert_layer_supports_batched_crown(Layer::MulBinary(MulBinaryLayer), false);
    assert_layer_supports_batched_crown(expand_like_l(), false);
}

// -- supports_batched_crown_with_conv2d() --

#[test]
fn test_supports_batched_crown_with_conv2d_excludes_softmax_family() {
    assert_layer_supports_batched_crown_with_conv2d(softmax(), false);
    assert_layer_supports_batched_crown_with_conv2d(
        Layer::CausalSoftmax(CausalSoftmaxLayer::new(-1)),
        false,
    );
    assert_layer_supports_batched_crown_with_conv2d(
        Layer::LogSoftmax(LogSoftmaxLayer::new(-1)),
        false,
    );
    assert_layer_supports_batched_crown_with_conv2d(
        Layer::LogSumExp(LogSumExpLayer::new(vec![-1], true)),
        false,
    );
    assert_layer_supports_batched_crown_with_conv2d(relu(), true);
    assert_layer_supports_batched_crown_with_conv2d(linear(), true);
    assert_layer_supports_batched_crown_with_conv2d(flatten(), true);
    // Tile, Reductions, and Slice are compatible with Conv2d graphs
    assert_layer_supports_batched_crown_with_conv2d(tile_l(), true);
    assert_layer_supports_batched_crown_with_conv2d(slice_l(), true);
    assert_layer_supports_batched_crown_with_conv2d(reduce_sum(), true);
    assert_layer_supports_batched_crown_with_conv2d(reduce_mean(), true);
    assert_layer_supports_batched_crown_with_conv2d(reduce_max(), true);
    assert_layer_supports_batched_crown_with_conv2d(reduce_min(), true);
}

// -- Error dispatch: binary on unary --

#[test]
fn test_propagate_ibp_binary_errors_for_unary_layer() {
    let input =
        BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
    let result = relu().propagate_ibp_binary(&input, &input);
    assert!(
        result.is_err(),
        "ReLU binary IBP dispatch should reject unary layers"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("not a binary"),
        "Expected 'not a binary' error, got: {err_msg}"
    );
}

// -- Error dispatch: ternary on non-ternary --

#[test]
fn test_propagate_ibp_ternary_errors_for_non_ternary() {
    let input =
        BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
    let result = add_binary().propagate_ibp_ternary(&input, &input, &input);
    assert!(
        result.is_err(),
        "Add binary IBP ternary dispatch should reject non-ternary layers"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("not a ternary"),
        "Expected 'not a ternary' error, got: {err_msg}"
    );
}

// -- Batched CROWN backward unsupported --

#[test]
fn test_batched_crown_backward_errors_for_unsupported() {
    let bounds = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_elem(IxDyn(&[1, 1]), 1.0_f32),
        ArrayD::from_elem(IxDyn(&[1]), 0.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1]), 1.0_f32),
        ArrayD::from_elem(IxDyn(&[1]), 0.0_f32),
        vec![1],
        vec![1],
    );
    let result = add_binary().propagate_crown_backward_batched(&bounds, None, None);
    assert!(
        result.is_err(),
        "Add binary batched CROWN backward should reject unsupported layers"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("not implemented"),
        "Expected 'not implemented' error, got: {err_msg}"
    );
}

// -- Consistency: binary and ternary are mutually exclusive --

#[test]
fn test_binary_and_ternary_mutually_exclusive() {
    let all_layers = vec![
        relu(),
        exp(),
        linear(),
        add_binary(),
        expand_like_l(),
        matmul(),
        where_l(),
        self_attn(),
        softmax(),
        flatten(),
        add_const(),
    ];
    for layer in &all_layers {
        assert!(
            !(layer.is_binary() && layer.is_ternary()),
            "{} is both binary and ternary — violation",
            layer.layer_type()
        );
    }
}

#[test]
fn test_elementwise_activations_support_batched_crown() {
    let activations = vec![relu(), exp(), Layer::Tanh(TanhLayer::new())];
    for layer in &activations {
        assert!(
            layer.supports_batched_crown(),
            "Elementwise activation {} should support batched CROWN",
            layer.layer_type()
        );
    }
}
