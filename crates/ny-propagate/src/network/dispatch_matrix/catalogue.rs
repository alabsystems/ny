// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Canonical catalogue of every [`Layer`] variant: `(name, sample layer)`.
//!
//! The sample carries a representative payload so the classifiers in
//! [`super::classify`] — and the runtime cross-checks in [`super::tests`] —
//! can be run against it. Only the discriminant matters for classification;
//! the payload matters for the runtime probes, so every sample is built at the
//! shared width [`CATALOGUE_WIDTH`] with the simplest configuration that a
//! dispatcher will actually accept.
//!
//! The list length is pinned by [`super::LAYER_VARIANT_COUNT`], asserted in
//! tests, so it cannot silently drift from the enum. That assertion is the
//! *second* line of defence: the four exhaustive matches in
//! [`super::classify`] already refuse to compile when a variant is added.
//!
//! Order mirrors the source enum declaration in `layers::layer_enum` for easy
//! cross-referencing.

use ndarray::{array, Array1, Array2, ArrayD, IxDyn};

use crate::layers::{
    AbsLayer, AdaIN1dLayer, AddConstantLayer, AddLayer, ArctanLayer, ArgMaxLayer, ArgMinLayer,
    ArgSortLayer, Atan2Layer, AttentionMask, AveragePoolLayer, BatchNormLayer, BilinearCrownLayer,
    CausalSoftmaxLayer, CeilLayer, CeluLayer, ClipLayer, CompareLayer, CompareOp,
    CompareTensorLayer, ConcatLayer, Conv1dLayer, Conv2dLayer, Conv3dLayer, ConvTranspose1dLayer,
    ConvTranspose2dLayer, CosLayer, CumsumLayer, DivConstantLayer, DivLayer, EluLayer,
    EpsilonExpandLayer, ErfLayer, ExpLayer, ExpandLikeLastAxisLayer, FlattenLayer, FloorLayer,
    GELULayer, GatherLayer, GeluApproximation, GroupNormLayer, HardSigmoidLayer, HardSwishLayer,
    IndexAddLayer, InstanceNorm1dLayer, IntervalConstLayer, Layer, LayerNormLayer, LeakyReLULayer,
    LinearLayer, LogLayer, LogSoftmaxLayer, LogSumExpLayer, MatMulLayer, MaxBinaryLayer,
    MaxPool2dLayer, MinBinaryLayer, MishLayer, MulBinaryLayer, MulConstantLayer, NonZeroLayer,
    OpaqueSkipLayer, PReluLayer, PadLayer, PadMode, PowConstantLayer, QdqPerturbationLayer,
    ReLULayer, ReciprocalLayer, ReduceMaxLayer, ReduceMeanLayer, ReduceMinLayer, ReduceSumLayer,
    ReshapeLayer, ResizeLayer, RmsNormLayer, RopeLayer, RoundLayer, ScatterAddLayer,
    ScatterNdLayer, SelfAttentionLayer, SeluLayer, ShrinkLayer, SiLULayer, SigmoidLayer, SignLayer,
    SinLayer, SkipMergeLayer, SliceLayer, SnakeLayer, SoftmaxLayer, SoftplusLayer, SoftsignLayer,
    SqrtLayer, SqueezeLayer, SubConstantLayer, SubLayer, TanLayer, TanhLayer, ThresholdedReluLayer,
    TileLayer, TopkLayer, TopkOutputKind, TransposeLayer, TruncLayer, UnsqueezeLayer, WhereLayer,
};

/// Feature width every catalogue sample is built at.
///
/// Small enough to keep the runtime probes cheap, and a power of two so the
/// convolutional and pooling samples have a legal spatial interpretation.
pub(super) const CATALOGUE_WIDTH: usize = 4;

/// `Layer` variants for which no value can be constructed today, with the
/// reason — so [`canonical_layers`] can be complete-by-construction without
/// pretending a sample exists.
///
/// This is NOT a hole in the guard. The four classifiers in
/// [`super::classify`] match over the TYPE, so every variant here is still
/// classified at compile time and a new variant still breaks the build. What
/// these variants lack is a *runtime probe*, and that is a property of the
/// variant, not of the guard.
pub(super) const UNCONSTRUCTIBLE_VARIANTS: &[(&str, &str)] = &[];

fn dense(n: usize) -> LinearLayer {
    LinearLayer::new(Array2::eye(n), Some(Array1::zeros(n))).expect("identity dense sample")
}

fn point_only(n: usize) -> crate::layers::PointLinearLayer {
    let budget = crate::certify::work::WorkBudget::new(
        None,
        1 << 20,
        1 << 20,
        1 << 20,
        16,
        std::time::Duration::ZERO,
    )
    .expect("point-only catalogue budget");
    let weights = Array2::eye(n);
    crate::layers::PointLinearLayer::from_original_view(
        weights.view(),
        false,
        None,
        crate::layers::PointLinearBuildPolicy {
            budget: &budget,
            deadline: None,
            max_resident_bytes: 1 << 20,
            max_scratch_bytes: 1 << 20,
            max_work: 1 << 20,
        },
    )
    .expect("point-only catalogue sample")
}

fn ones(n: usize) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(&[n]), 1.0_f32)
}

fn zeros(n: usize) -> ArrayD<f32> {
    ArrayD::zeros(IxDyn(&[n]))
}

fn instance_norm(n: usize) -> InstanceNorm1dLayer {
    InstanceNorm1dLayer::new(Array1::ones(n), Array1::zeros(n), 1e-5).expect("instance-norm sample")
}

/// Every CONSTRUCTIBLE `Layer` variant, once, in enum-declaration order.
///
/// [`UNCONSTRUCTIBLE_VARIANTS`] holds the rest; together they are the complete
/// enum.
///
/// Built fresh on each call: `Layer` is not `Copy` and several dispatchers take
/// it by reference while mutating a clone, so handing out a shared static would
/// be a footgun for no measurable gain (the whole catalogue costs microseconds).
#[allow(clippy::too_many_lines)]
pub(super) fn canonical_layers() -> Vec<(&'static str, Layer)> {
    let n = CATALOGUE_WIDTH;
    vec![
        ("Linear", Layer::Linear(dense(n))),
        ("PointLinear", Layer::PointLinear(point_only(n))),
        (
            "IntervalConst",
            Layer::IntervalConst(
                IntervalConstLayer::try_new_linear(dense(n), Array2::from_elem((n, n), 0.125))
                    .expect("interval-const sample"),
            ),
        ),
        (
            "Conv1d",
            Layer::Conv1d(
                Conv1dLayer::new(
                    ArrayD::from_elem(IxDyn(&[1, 1, 1]), 1.0_f32),
                    Some(Array1::zeros(1)),
                    1,
                    0,
                )
                .expect("conv1d sample"),
            ),
        ),
        (
            "Conv2d",
            Layer::Conv2d(
                Conv2dLayer::new(
                    ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), 1.0_f32),
                    Some(Array1::zeros(1)),
                    (1, 1),
                    (0, 0),
                )
                .expect("conv2d sample"),
            ),
        ),
        (
            // Conv3d's sample is deliberately all-ones geometry, like the
            // Conv1d/Conv2d samples: only the discriminant matters to the
            // classifiers, and the runtime probes feed a rank-1 width-4 box
            // that no convolution accepts anyway.
            "Conv3d",
            Layer::Conv3d(
                Conv3dLayer::new(
                    ArrayD::from_elem(IxDyn(&[1, 1, 1, 1, 1]), 1.0_f32),
                    Some(Array1::zeros(1)),
                    [1, 1, 1],
                    [0, 0, 0],
                )
                .expect("conv3d sample"),
            ),
        ),
        (
            "ConvTranspose1d",
            Layer::ConvTranspose1d(
                ConvTranspose1dLayer::new(
                    ArrayD::from_elem(IxDyn(&[1, 1, 1]), 1.0_f32),
                    Some(Array1::zeros(1)),
                    1,
                    0,
                )
                .expect("conv-transpose1d sample"),
            ),
        ),
        (
            "ConvTranspose2d",
            Layer::ConvTranspose2d(
                ConvTranspose2dLayer::new(
                    ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), 1.0_f32),
                    Some(Array1::zeros(1)),
                    (1, 1),
                    (0, 0),
                )
                .expect("conv-transpose2d sample"),
            ),
        ),
        (
            "AveragePool",
            Layer::AveragePool(AveragePoolLayer::new((1, 1), (1, 1), (0, 0), false)),
        ),
        (
            "MaxPool2d",
            Layer::MaxPool2d(MaxPool2dLayer::new((1, 1), (1, 1), (0, 0))),
        ),
        ("ReLU", Layer::ReLU(ReLULayer)),
        ("LeakyReLU", Layer::LeakyReLU(LeakyReLULayer::new(0.01))),
        ("Clip", Layer::Clip(ClipLayer::new(-1.0, 1.0))),
        ("Elu", Layer::Elu(EluLayer::new(1.0))),
        ("Selu", Layer::Selu(SeluLayer::new())),
        (
            "PRelu",
            Layer::PRelu(PReluLayer::new(Array1::from_elem(n, 0.25)).expect("prelu sample")),
        ),
        (
            "HardSigmoid",
            Layer::HardSigmoid(HardSigmoidLayer::new(0.2, 0.5)),
        ),
        ("HardSwish", Layer::HardSwish(HardSwishLayer::new())),
        ("SiLU", Layer::SiLU(SiLULayer::new())),
        ("Exp", Layer::Exp(ExpLayer::new())),
        ("Log", Layer::Log(LogLayer::new())),
        ("Celu", Layer::Celu(CeluLayer::new(1.0))),
        ("Mish", Layer::Mish(MishLayer::new())),
        ("LogSoftmax", Layer::LogSoftmax(LogSoftmaxLayer::new(-1))),
        (
            "LogSumExp",
            Layer::LogSumExp(LogSumExpLayer::new(vec![-1], false)),
        ),
        (
            "ThresholdedRelu",
            Layer::ThresholdedRelu(ThresholdedReluLayer::new(1.0)),
        ),
        ("Shrink", Layer::Shrink(ShrinkLayer::new(0.0, 0.5))),
        ("Softsign", Layer::Softsign(SoftsignLayer::new())),
        (
            "Snake",
            Layer::Snake(SnakeLayer::new(1.0).expect("snake sample")),
        ),
        ("Softmax", Layer::Softmax(SoftmaxLayer::new(-1))),
        (
            "CausalSoftmax",
            Layer::CausalSoftmax(CausalSoftmaxLayer::new(-1)),
        ),
        (
            "SelfAttention",
            Layer::SelfAttention(SelfAttentionLayer::new(AttentionMask::Standard, None)),
        ),
        ("GELU", Layer::GELU(GELULayer::new(GeluApproximation::Tanh))),
        (
            "LayerNorm",
            Layer::LayerNorm(
                LayerNormLayer::new(Array1::ones(n), Array1::zeros(n), 1e-5)
                    .expect("layer-norm sample"),
            ),
        ),
        (
            "RmsNorm",
            Layer::RmsNorm(RmsNormLayer::new(Array1::ones(n), 1e-5).expect("rms-norm sample")),
        ),
        ("InstanceNorm1d", Layer::InstanceNorm1d(instance_norm(n))),
        (
            "GroupNorm",
            Layer::GroupNorm(
                GroupNormLayer::new(Array1::ones(n), Array1::zeros(n), 1, 1e-5)
                    .expect("group-norm sample"),
            ),
        ),
        (
            "AdaIN1d",
            Layer::AdaIN1d(
                AdaIN1dLayer::new_identity_style(instance_norm(n)).expect("fixed-style adain"),
            ),
        ),
        (
            "BatchNorm",
            Layer::BatchNorm(
                BatchNormLayer::new(&ones(n), &zeros(n), &zeros(n), &ones(n), 1e-5)
                    .expect("batch-norm sample"),
            ),
        ),
        ("MatMul", Layer::MatMul(MatMulLayer::new(false, None))),
        ("MulBinary", Layer::MulBinary(MulBinaryLayer)),
        ("Add", Layer::Add(AddLayer)),
        ("Concat", Layer::Concat(ConcatLayer::new(-1))),
        ("Sub", Layer::Sub(SubLayer)),
        ("Div", Layer::Div(DivLayer)),
        ("Atan2", Layer::Atan2(Atan2Layer)),
        (
            "BilinearCrown",
            Layer::BilinearCrown(BilinearCrownLayer::new(false, None)),
        ),
        ("MinBinary", Layer::MinBinary(MinBinaryLayer)),
        ("MaxBinary", Layer::MaxBinary(MaxBinaryLayer)),
        (
            "AddConstant",
            Layer::AddConstant(AddConstantLayer::new(zeros(n))),
        ),
        ("Transpose", Layer::Transpose(TransposeLayer::new(vec![0]))),
        ("Reshape", Layer::Reshape(ReshapeLayer::new(vec![n as i64]))),
        ("Flatten", Layer::Flatten(FlattenLayer::new(0))),
        (
            "Pad",
            Layer::Pad(PadLayer::new(vec![(0, 0)], PadMode::Constant(0.0))),
        ),
        (
            "MulConstant",
            Layer::MulConstant(MulConstantLayer::new(ones(n))),
        ),
        ("Abs", Layer::Abs(AbsLayer::new())),
        ("Sqrt", Layer::Sqrt(SqrtLayer::new())),
        (
            "DivConstant",
            Layer::DivConstant(DivConstantLayer::new(ones(n))),
        ),
        (
            "SubConstant",
            Layer::SubConstant(SubConstantLayer::new(zeros(n))),
        ),
        (
            "PowConstant",
            Layer::PowConstant(PowConstantLayer::new(2.0)),
        ),
        (
            "ReduceMean",
            Layer::ReduceMean(ReduceMeanLayer::new(vec![-1], false)),
        ),
        (
            "ReduceSum",
            Layer::ReduceSum(ReduceSumLayer::new(vec![-1], false)),
        ),
        (
            "ReduceMax",
            Layer::ReduceMax(ReduceMaxLayer::new(vec![-1], false)),
        ),
        (
            "ReduceMin",
            Layer::ReduceMin(ReduceMinLayer::new(vec![-1], false)),
        ),
        (
            "Topk",
            Layer::Topk(TopkLayer::new(1, -1, TopkOutputKind::Values)),
        ),
        ("ArgMax", Layer::ArgMax(ArgMaxLayer::new(-1, false))),
        ("ArgMin", Layer::ArgMin(ArgMinLayer::new(-1, false))),
        ("ArgSort", Layer::ArgSort(ArgSortLayer::new(-1, false))),
        ("CumSum", Layer::CumSum(CumsumLayer::new(-1, false, false))),
        ("Tanh", Layer::Tanh(TanhLayer::new())),
        ("Sigmoid", Layer::Sigmoid(SigmoidLayer::new())),
        ("Erf", Layer::Erf(ErfLayer::new())),
        ("Softplus", Layer::Softplus(SoftplusLayer::new())),
        ("Sin", Layer::Sin(SinLayer::new())),
        ("Cos", Layer::Cos(CosLayer::new())),
        ("Tan", Layer::Tan(TanLayer::new())),
        ("Arctan", Layer::Arctan(ArctanLayer::new())),
        (
            "RoPE",
            Layer::RoPE(
                RopeLayer::new(vec![1.0, 1.0], vec![0.0, 0.0]).expect("identity rope sample"),
            ),
        ),
        ("Tile", Layer::Tile(TileLayer::new(-1, 1))),
        (
            "ExpandLikeLastAxis",
            Layer::ExpandLikeLastAxis(ExpandLikeLastAxisLayer::new()),
        ),
        (
            "Gather",
            Layer::Gather(GatherLayer::new(
                0,
                Some(ArrayD::from_shape_vec(IxDyn(&[1]), vec![0_i64]).expect("gather indices")),
                vec![1],
            )),
        ),
        (
            "ScatterAdd",
            Layer::ScatterAdd(ScatterAddLayer::new(0, None, None, None)),
        ),
        (
            "IndexAdd",
            Layer::IndexAdd(IndexAddLayer::new(0, None, None, None)),
        ),
        (
            "ScatterNd",
            Layer::ScatterNd(ScatterNdLayer::new(None, None, None)),
        ),
        ("Slice", Layer::Slice(SliceLayer::new(-1, 0, n))),
        ("Squeeze", Layer::Squeeze(SqueezeLayer::new(0))),
        ("Unsqueeze", Layer::Unsqueeze(UnsqueezeLayer::new(0))),
        ("Resize", Layer::Resize(ResizeLayer::new(1, 1))),
        (
            "Compare",
            Layer::Compare(CompareLayer::new(0.0, CompareOp::Gt)),
        ),
        (
            "CompareTensor",
            Layer::CompareTensor(CompareTensorLayer::new(CompareOp::Gt)),
        ),
        ("Where", Layer::Where(WhereLayer::new())),
        ("NonZero", Layer::NonZero(NonZeroLayer)),
        ("Floor", Layer::Floor(FloorLayer::new())),
        ("Ceil", Layer::Ceil(CeilLayer::new())),
        ("Round", Layer::Round(RoundLayer::new())),
        ("Trunc", Layer::Trunc(TruncLayer::new())),
        ("Sign", Layer::Sign(SignLayer::new())),
        ("Reciprocal", Layer::Reciprocal(ReciprocalLayer::new())),
        ("SkipMerge", Layer::SkipMerge(SkipMergeLayer::new())),
        ("OpaqueSkip", Layer::OpaqueSkip(OpaqueSkipLayer::new())),
        (
            "QdqPerturbation",
            Layer::QdqPerturbation(
                QdqPerturbationLayer::try_certified(&[0.1], &[0], 0, 255)
                    .expect("certified QDQ envelope sample"),
            ),
        ),
        (
            "EpsilonExpand",
            Layer::EpsilonExpand(EpsilonExpandLayer::uniform(0.01).expect("epsilon-expand sample")),
        ),
    ]
}

/// The variable-style `AdaIN1d` sample.
///
/// Not part of [`canonical_layers`] — it is the SAME enum variant as the fixed
/// style sample, and the catalogue holds one entry per variant. It exists
/// because the fixed/variable split is a real classification boundary in the
/// batched dispatcher, so the cross-checks probe both configurations.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn variable_style_adain() -> Layer {
    Layer::AdaIN1d(
        AdaIN1dLayer::variable_style(instance_norm(CATALOGUE_WIDTH))
            .expect("variable-style adain sample"),
    )
}

/// A `Where` whose two branches are frozen constants.
///
/// Same reason as [`variable_style_adain`]: `Layer::Where` classifies
/// differently in the batched dispatcher depending on whether its branches are
/// embedded constants, and the cross-checks probe both.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn embedded_constant_where() -> Layer {
    Layer::Where(WhereLayer::with_constants(
        Some(array![1.0_f32, 1.0, 1.0, 1.0].into_dyn()),
        Some(array![0.0_f32, 0.0, 0.0, 0.0].into_dyn()),
    ))
}
