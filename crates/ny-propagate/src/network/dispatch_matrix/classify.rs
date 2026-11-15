// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The four exhaustive `match &Layer` classifiers — one per dispatcher.
//!
//! Every function here is an **exhaustive match with no wildcard arm**. Adding
//! a variant to [`crate::layers::Layer`] is a compile error in all four until
//! someone classifies it, which is the entire point of this module: the
//! three silent-degradation bugs this guard exists to prevent were each a
//! variant that the DENSE dispatcher handled and the BATCHED one quietly did
//! not, with no compile error and no test to notice.
//!
//! See the [module docs](super) for what each [`DispatchVerdict`] means and for
//! which runtime cross-check pins each classifier to its dispatcher.

use crate::layers::Layer;

// Diagnostic point-only representations have no proof fallback.
use super::DispatchVerdict::{self, FallsBack, Refused, Supported};

// ===================================================================
// 1. Dense CROWN backward — `network::backward_dispatch::dispatch_backward_layer`
// ===================================================================

/// Classify `layer` against the canonical DENSE CROWN backward dispatcher.
///
/// Ground truth: `dispatch_backward_layer_inner` in
/// `network/backward_dispatch/dispatch.rs`, which is exhaustive (no
/// catch-all). `Supported` here means that dispatcher has an arm
/// producing real linear bounds; `FallsBack` means the arm exists but returns
/// `BackwardDispatchResult::Unsupported`, which every caller turns into a sound
/// IBP concretization of the node.
#[allow(clippy::match_same_arms, clippy::too_many_lines)]
pub(super) fn dense_crown_backward(layer: &Layer) -> DispatchVerdict {
    match layer {
        Layer::PointLinear(_) => Refused("diagnostic point-only representation"),
        // --- Real dense arms -------------------------------------------------
        Layer::Linear(_)
        | Layer::IntervalConst(_)
        | Layer::Conv1d(_)
        | Layer::Conv2d(_)
        | Layer::ConvTranspose1d(_)
        | Layer::ConvTranspose2d(_)
        | Layer::Transpose(_)
        | Layer::Tile(_)
        | Layer::Slice(_)
        | Layer::Gather(_)
        | Layer::Add(_)
        | Layer::Sub(_)
        | Layer::ExpandLikeLastAxis(_)
        | Layer::Concat(_)
        | Layer::MatMul(_)
        | Layer::BilinearCrown(_)
        | Layer::SkipMerge(_)
        | Layer::OpaqueSkip(_)
        | Layer::MulBinary(_)
        | Layer::ScatterAdd(_)
        | Layer::IndexAdd(_)
        | Layer::ScatterNd(_)
        | Layer::SelfAttention(_)
        | Layer::RmsNorm(_)
        | Layer::MinBinary(_)
        | Layer::MaxBinary(_)
        | Layer::Atan2(_) => Supported,

        // --- Unified unary trait arm: layers with a real CROWN relaxation ----
        // Elementwise activations, softmax family, normalization, constant
        // arithmetic, linear reductions, shape transforms, pooling, RoPE.
        Layer::GELU(_)
        | Layer::SiLU(_)
        | Layer::Tanh(_)
        | Layer::Sigmoid(_)
        | Layer::Erf(_)
        | Layer::Exp(_)
        | Layer::Log(_)
        | Layer::Sqrt(_)
        | Layer::Reciprocal(_)
        | Layer::Softplus(_)
        | Layer::HardSwish(_)
        | Layer::Mish(_)
        | Layer::Selu(_)
        | Layer::Softsign(_)
        | Layer::Arctan(_)
        | Layer::Tan(_)
        | Layer::Sin(_)
        | Layer::Cos(_)
        | Layer::Elu(_)
        | Layer::Celu(_)
        | Layer::LeakyReLU(_)
        | Layer::HardSigmoid(_)
        | Layer::Clip(_)
        | Layer::ThresholdedRelu(_)
        | Layer::Abs(_)
        | Layer::PowConstant(_)
        | Layer::Floor(_)
        | Layer::Ceil(_)
        | Layer::Round(_)
        | Layer::Trunc(_)
        | Layer::Sign(_)
        | Layer::PRelu(_)
        | Layer::Shrink(_)
        | Layer::Snake(_)
        | Layer::Compare(_)
        | Layer::Softmax(_)
        | Layer::CausalSoftmax(_)
        | Layer::LogSoftmax(_)
        | Layer::LogSumExp(_)
        | Layer::LayerNorm(_)
        | Layer::InstanceNorm1d(_)
        | Layer::GroupNorm(_)
        | Layer::AdaIN1d(_)
        | Layer::BatchNorm(_)
        | Layer::AddConstant(_)
        | Layer::MulConstant(_)
        | Layer::DivConstant(_)
        | Layer::SubConstant(_)
        | Layer::ReduceMean(_)
        | Layer::ReduceSum(_)
        | Layer::CumSum(_)
        | Layer::ReduceMax(_)
        | Layer::ReduceMin(_)
        | Layer::Flatten(_)
        | Layer::Reshape(_)
        | Layer::Squeeze(_)
        | Layer::Unsqueeze(_)
        | Layer::Pad(_)
        | Layer::Resize(_)
        | Layer::QdqPerturbation(_)
        | Layer::EpsilonExpand(_)
        | Layer::AveragePool(_)
        | Layer::MaxPool2d(_)
        | Layer::RoPE(_) => Supported,

        // --- Deliberate fallbacks -------------------------------------------
        // ReLU: alpha/beta parameters live in the CALLER's state, so the shared
        // dispatcher refuses rather than silently emitting the fixed-slope
        // relaxation behind an optimizer's back. Every caller pre-filters it.
        Layer::ReLU(_) => FallsBack(
            "alpha/beta ReLU relaxation is caller state; every dispatch site pre-filters ReLU",
        ),
        // Where: the four-tier condition routing lives in the graph CROWN
        // coordinator, which has the node's condition bounds; the shared
        // dispatcher does not.
        Layer::Where(_) => {
            FallsBack("condition-tier routing is site-specific to the graph CROWN coordinator")
        }
        // Div: site-specific reciprocal-envelope handler in the DAG CROWN
        // coordinator (`backward_div_to_numerator_with_deadline`).
        Layer::Div(_) => {
            FallsBack("reciprocal-envelope handler is site-specific to the graph CROWN coordinator")
        }
        // Index-valued selection: the output is an INDEX, not a value, so no
        // linear relaxation in the value domain exists. IBP is exact wherever
        // the inputs separate and the sound index hull elsewhere.
        Layer::Topk(_) | Layer::ArgMax(_) | Layer::ArgMin(_) | Layer::ArgSort(_) => {
            FallsBack("index-valued output has no linear relaxation; IBP index hull is the bound")
        }
        // Step functions of two tensors: {0,1}-valued, no useful linear
        // relaxation (the unary `Compare` gets one only because its threshold
        // is a constant).
        Layer::CompareTensor(_) => {
            FallsBack("two-tensor step function has no useful linear relaxation; IBP-only")
        }

        // NonZero's output SHAPE depends on values, so there is no fixed
        // coefficient matrix to propagate. NOT a refusal: the trait arm reports
        // `Unsupported`, and the caller's IBP fallback publishes the sound
        // `[0, dim-1]` index hull, so the node still gets a (very loose) bound.
        Layer::NonZero(_) => FallsBack(
            "data-dependent output shape has no fixed coefficient matrix; the IBP fallback \
             publishes the sound [0, dim-1] index hull instead",
        ),
        // Conv3d is affine, so a CROWN backward EXISTS in principle — it is a
        // rank-3-spatial transposed convolution. What does not exist is a
        // CERTIFIED one: the f32 transpose-conv coefficient needs the same
        // `cast_err + γ_n·S + prop` machinery Conv1d/Conv2d carry
        // (#vnncomp-aw-soundness), and shipping the coefficient without it is
        // the false-proof case that machinery was written to close. Until that
        // lands, `propagate_linear` refuses and this dispatcher IBP-falls-back.
        // NOT `Refused`: the node still gets a sound (loose) bound.
        Layer::Conv3d(_) => FallsBack(
            "no CERTIFIED rank-3-spatial transposed convolution yet; the uncertified f32 \
             coefficient is the false-proof case, so the layer refuses and the caller \
             concretizes through its certified IBP forward",
        ),
    }
}

// ===================================================================
// 2. Batched CROWN backward — the batched graph path
// ===================================================================

/// Classify `layer` against the BATCHED CROWN backward path.
///
/// Ground truth is a pair of dispatchers that together decide what the batched
/// path does with a node:
///
/// * `Layer::propagate_crown_backward_batched` (`layers/layer_enum/dispatch.rs`)
///   for unary nodes — exhaustive, no catch-all;
/// * `GraphNetwork::dispatch_special_batched_operator`
///   (`network/core/graph/crown_batched/dispatch.rs`) for the multi-input and
///   graph-structured operators — exhaustive since this guard landed.
///
/// The network-level admission gate `Layer::supports_batched_crown` must agree
/// with this classification for every unary variant; that is asserted by
/// `tests::batched_admission_gate_matches_the_per_layer_arm`, and is
/// exactly the check that the CumSum regression would have failed.
///
/// `FallsBack` here means the batched pass keeps its established sound
/// fallback for the node — partial CROWN / IBP concretization — which is
/// correct but strictly looser than the dense arm the same variant enjoys.
#[allow(clippy::match_same_arms, clippy::too_many_lines)]
pub(super) fn batched_crown_backward(layer: &Layer) -> DispatchVerdict {
    match layer {
        Layer::PointLinear(_) => Refused("diagnostic point-only representation"),
        // --- Real batched arms (unary) --------------------------------------
        // Every elementwise activation routes through the macro-generated arms.
        Layer::ReLU(_)
        | Layer::GELU(_)
        | Layer::SiLU(_)
        | Layer::Tanh(_)
        | Layer::Sigmoid(_)
        | Layer::Erf(_)
        | Layer::Exp(_)
        | Layer::Log(_)
        | Layer::Sqrt(_)
        | Layer::Reciprocal(_)
        | Layer::Softplus(_)
        | Layer::HardSwish(_)
        | Layer::Mish(_)
        | Layer::Selu(_)
        | Layer::Softsign(_)
        | Layer::Arctan(_)
        | Layer::Tan(_)
        | Layer::Sin(_)
        | Layer::Cos(_)
        | Layer::Elu(_)
        | Layer::Celu(_)
        | Layer::LeakyReLU(_)
        | Layer::HardSigmoid(_)
        | Layer::Clip(_)
        | Layer::ThresholdedRelu(_)
        | Layer::Abs(_)
        | Layer::PowConstant(_)
        | Layer::Floor(_)
        | Layer::Ceil(_)
        | Layer::Round(_)
        | Layer::Trunc(_)
        | Layer::Sign(_)
        | Layer::PRelu(_)
        | Layer::Shrink(_)
        | Layer::Snake(_)
        | Layer::Compare(_)
        // Softmax family, normalization, linear ops, shape transforms.
        | Layer::Softmax(_)
        | Layer::CausalSoftmax(_)
        | Layer::LogSoftmax(_)
        | Layer::LogSumExp(_)
        | Layer::LayerNorm(_)
        | Layer::RmsNorm(_)
        | Layer::InstanceNorm1d(_)
        | Layer::GroupNorm(_)
        | Layer::BatchNorm(_)
        | Layer::Linear(_)
        | Layer::RoPE(_)
        | Layer::Conv1d(_)
        | Layer::ConvTranspose1d(_)
        | Layer::Conv2d(_)
        | Layer::ConvTranspose2d(_)
        | Layer::Flatten(_)
        | Layer::Reshape(_)
        | Layer::Pad(_)
        | Layer::Squeeze(_)
        | Layer::Unsqueeze(_)
        | Layer::Resize(_)
        | Layer::Transpose(_)
        | Layer::Slice(_)
        | Layer::Tile(_)
        | Layer::AddConstant(_)
        | Layer::SubConstant(_)
        | Layer::MulConstant(_)
        | Layer::DivConstant(_)
        | Layer::ReduceMean(_)
        | Layer::ReduceSum(_)
        | Layer::ReduceMax(_)
        | Layer::ReduceMin(_)
        | Layer::QdqPerturbation(_)
        | Layer::EpsilonExpand(_) => Supported,

        // CumSum: grouped last-axis prefix/suffix scan with its own certified
        // `γ_axislen·S` coefficient error. REGRESSION SITE — the batched
        // backward was implemented and dispatched, but the variant was missing
        // from `supports_batched_crown`, so a single CumSum node disqualified
        // an entire network from the batched path.
        Layer::CumSum(_) => Supported,

        // Fixed-style AdaIN has a batched arm; variable style does not reach a
        // usable one (`effective_instance_norm` needs the style operands, which
        // the batched carrier cannot supply), and the network gate excludes it.
        Layer::AdaIN1d(adain) if adain.requires_style_inputs() => {
            FallsBack("variable-style AdaIN needs live style operands the batched carrier lacks")
        }
        Layer::AdaIN1d(_) => Supported,

        // --- Real batched arms (multi-input / graph-structured) -------------
        // `dispatch_special_batched_operator`. Each of these can still decline
        // for a specific node's layout and keep the sound fallback; the verdict
        // records that the ARM exists, not that every node takes it.
        Layer::Add(_) | Layer::ExpandLikeLastAxis(_) | Layer::MatMul(_) | Layer::MulBinary(_)
        | Layer::BilinearCrown(_) => Supported,
        // REGRESSION SITE: the dense dispatcher has long had a `Sub` arm; the
        // batched one had none, so every `Sub` node — the join of
        // every fusion difference-graph — concretized through the
        // correlation-blind interval difference.
        Layer::Sub(_) => Supported,
        // REGRESSION SITE: `ConcatLayer::propagate_linear_batched_binary` was
        // implemented and tested with ZERO production callers.
        Layer::Concat(_) => Supported,
        // REGRESSION SITE: no batched `Where` arm existed at all, so a causal /
        // padding attention mask blurred into `[min(t,f), max(t,f)]` at every
        // position.
        Layer::Where(_) => Supported,

        // --- Deliberate fallbacks: dense relaxation exists, batched does not -
        // Each of these HAS a dense CROWN relaxation written against a flat
        // `[out, in]` `LinearBounds` whose columns index the whole tensor. The
        // batched carrier is `[batch.., out_dim, in_dim]` (per-position,
        // last-axis-only), so reaching the dense envelope from the batched
        // dispatcher would mean re-deriving it — the "false proof" case. Each
        // keeps the established sound IBP / partial-CROWN fallback until the
        // per-layer batched relaxation is ported next to its dense twin.
        Layer::Div(_) => FallsBack(
            "dense Div is a site-specific coordinator handler whose concretize branch needs a \
             bias-to-network-input channel the batched accumulator lacks",
        ),
        Layer::MinBinary(_) | Layer::MaxBinary(_) => FallsBack(
            "dense piecewise envelope is built over the joint (x, y) box across two FLAT operands",
        ),
        Layer::Atan2(_) => {
            FallsBack("dense C¹ envelope is built over a flat operand pair; no batched port yet")
        }
        Layer::Gather(_) => {
            FallsBack("backward is an index map over the FLAT tensor, not over a last axis")
        }
        Layer::AveragePool(_) | Layer::MaxPool2d(_) => {
            FallsBack("spatial backward is an index map over the FLAT tensor, not over a last axis")
        }
        Layer::ScatterAdd(_) | Layer::IndexAdd(_) | Layer::ScatterNd(_) => {
            FallsBack("scatter backward is an index map over the FLAT tensor")
        }
        Layer::SelfAttention(_) => {
            FallsBack("dense ternary attention composition has no batched port")
        }
        Layer::SkipMerge(_) | Layer::OpaqueSkip(_) => {
            FallsBack("pass-through / opaque carriers are resolved by the graph driver, not here")
        }
        Layer::CompareTensor(_) => FallsBack("IBP-only step function; no linear relaxation at all"),
        Layer::Topk(_) | Layer::ArgMax(_) | Layer::ArgMin(_) | Layer::ArgSort(_) => {
            FallsBack("index-valued output has no linear relaxation; IBP index hull is the bound")
        }
        // IntervalConst: batched support would have to carry `(|A| + err_in)·R`
        // through the BATCHED error channel. Refusing degrades to the scalar
        // backward, which DOES discharge the enclosure — never to a verdict.
        Layer::IntervalConst(_) => {
            FallsBack("certified-enclosure constant needs a batched coefficient-error channel")
        }

        Layer::NonZero(_) => FallsBack(
            "data-dependent output shape has no fixed coefficient matrix; the partial-CROWN \
             fallback publishes the sound [0, dim-1] index hull instead",
        ),
        // Conv3d: no CROWN backward at all, batched or dense (see the dense
        // classifier). The batched dispatcher's explicit unsupported arm fires
        // and the partial-CROWN fallback concretizes through the certified IBP.
        Layer::Conv3d(_) => {
            FallsBack("no certified rank-3-spatial transposed convolution exists, dense or batched")
        }
    }
}

/// Whether this variant's batched arm lives in the GRAPH-level dispatcher
/// (`GraphNetwork::dispatch_special_batched_operator`) rather than in
/// `Layer::propagate_crown_backward_batched`.
///
/// Mirrors that dispatcher's arm list. Deliberately a `matches!` rather than a
/// fifth exhaustive match: it is derived information, and a new variant that
/// gains a graph-level arm without being listed here is caught at RUN time by
/// `tests::batched_matrix_matches_the_real_dispatchers` — it would
/// declare `Supported` while the per-layer probe reports no arm. The
/// compile-time gate for that dispatcher lives where it belongs, in the
/// dispatcher's own no-catch-all match.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn batched_arm_is_graph_level(layer: &Layer) -> bool {
    matches!(
        layer,
        Layer::Add(_)
            | Layer::Sub(_)
            | Layer::Concat(_)
            | Layer::Where(_)
            | Layer::MatMul(_)
            | Layer::MulBinary(_)
            | Layer::BilinearCrown(_)
            | Layer::ExpandLikeLastAxis(_)
    )
}

// ===================================================================
// 3. Alpha entry — `graph_alpha::propagate_dag::init_dag_alpha_state`
// ===================================================================

/// Classify `layer` against the DAG α-CROWN router's alpha-bearing set.
///
/// Ground truth: `init_dag_alpha_state` in
/// `network/graph_alpha/propagate_dag/init.rs`. `Supported` means the router
/// registers optimizable alpha parameters for this variant AND its presence
/// holds the graph in the optimizer loop. `FallsBack` means the variant's
/// relaxation is taken at a FIXED slope — sound, but never optimized — so a
/// graph made only of such variants leaves through the early-return door.
///
/// The `MulBinary` arm is the one that moved most recently
/// (#mulbinary-alpha-router): a graph whose only optimizable relaxations were
/// multiplicative used to leave through that door before any alpha state was
/// built, so every SwiGLU-style `up * silu(gate)` FFN reached plain batched
/// CROWN with no alpha optimization at all.
#[allow(clippy::match_same_arms, clippy::too_many_lines)]
pub(super) fn alpha_entry(layer: &Layer) -> DispatchVerdict {
    match layer {
        Layer::PointLinear(_) => Refused("diagnostic point-only representation"),
        // --- Alpha-bearing --------------------------------------------------
        Layer::ReLU(_) => Supported,
        Layer::Sigmoid(_) | Layer::Tanh(_) => Supported,
        // Sqrt / Reciprocal register alphas only where the pre-activation box
        // is sign-definite; the router checks that per node. The variant is
        // alpha-bearing either way — an indefinite box means no alpha for THAT
        // node, not a missing entry point.
        Layer::Sqrt(_) | Layer::Reciprocal(_) => Supported,
        Layer::BilinearCrown(_) => Supported,
        Layer::MulBinary(_) => Supported,

        // --- Fixed-slope relaxation (no alpha entry point) ------------------
        // These have real CROWN relaxations with an alpha DEGREE OF FREEDOM in
        // the literature, but no entry point in this router yet. Listing them
        // separately from the exact/linear ops keeps the difference between
        // "nothing to optimize" and "something to optimize, not wired" visible.
        Layer::GELU(_) => FallsBack(
            "GELU alpha exists in the BLOCK-WISE optimizer only; the DAG router has no entry point",
        ),
        Layer::Elu(_)
        | Layer::Celu(_)
        | Layer::Selu(_)
        | Layer::SiLU(_)
        | Layer::Mish(_)
        | Layer::Softplus(_)
        | Layer::HardSwish(_)
        | Layer::HardSigmoid(_)
        | Layer::Softsign(_)
        | Layer::Erf(_)
        | Layer::Arctan(_)
        | Layer::Snake(_)
        | Layer::Exp(_)
        | Layer::Log(_) => {
            FallsBack("S-shaped/convex relaxation is taken at its fixed slope; no alpha entry")
        }
        Layer::LeakyReLU(_)
        | Layer::PRelu(_)
        | Layer::Clip(_)
        | Layer::ThresholdedRelu(_)
        | Layer::Shrink(_)
        | Layer::Abs(_)
        | Layer::MaxPool2d(_) => {
            FallsBack("piecewise-linear relaxation is taken at its fixed slope; no alpha entry")
        }
        Layer::Softmax(_)
        | Layer::CausalSoftmax(_)
        | Layer::LogSoftmax(_)
        | Layer::LogSumExp(_)
        | Layer::LayerNorm(_)
        | Layer::RmsNorm(_)
        | Layer::InstanceNorm1d(_)
        | Layer::GroupNorm(_)
        | Layer::AdaIN1d(_)
        | Layer::BatchNorm(_)
        | Layer::SelfAttention(_) => {
            FallsBack("normalization/attention relaxation is not alpha-parameterized in ny")
        }
        Layer::MatMul(_) => FallsBack(
            "alpha-bearing only through the BilinearCrown rewrite; raw MatMul has no alpha map",
        ),
        Layer::Div(_) | Layer::Atan2(_) | Layer::MinBinary(_) | Layer::MaxBinary(_) => {
            FallsBack("binary envelope is taken at its fixed construction; no alpha entry")
        }
        Layer::Sin(_) | Layer::Cos(_) | Layer::Tan(_) => {
            FallsBack("periodic relaxation is taken at its fixed slope; no alpha entry")
        }
        Layer::PowConstant(_) | Layer::Sign(_) | Layer::Compare(_) | Layer::CompareTensor(_) => {
            FallsBack("relaxation is fixed by construction; no alpha entry")
        }
        Layer::Floor(_) | Layer::Ceil(_) | Layer::Round(_) | Layer::Trunc(_) => {
            FallsBack("piecewise-constant relaxation is fixed by construction; no alpha entry")
        }
        Layer::Topk(_) | Layer::ArgMax(_) | Layer::ArgMin(_) | Layer::ArgSort(_) => {
            FallsBack("index-valued output has nothing to optimize")
        }

        // --- Exact / linear: nothing to optimize -----------------------------
        // Not a gap: an exact backward has no relaxation slack for an alpha to
        // move. Recorded as a fallback (rather than a refusal) because the
        // router really does walk past them without registering state.
        Layer::Linear(_)
        | Layer::IntervalConst(_)
        | Layer::Conv1d(_)
        | Layer::Conv2d(_)
        // Conv3d is affine too — no relaxation slack for an alpha to move,
        // whether or not its backward is ever written.
        | Layer::Conv3d(_)
        | Layer::ConvTranspose1d(_)
        | Layer::ConvTranspose2d(_)
        | Layer::AveragePool(_)
        | Layer::Add(_)
        | Layer::Sub(_)
        | Layer::Concat(_)
        | Layer::AddConstant(_)
        | Layer::SubConstant(_)
        | Layer::MulConstant(_)
        | Layer::DivConstant(_)
        | Layer::Transpose(_)
        | Layer::Reshape(_)
        | Layer::Flatten(_)
        | Layer::Squeeze(_)
        | Layer::Unsqueeze(_)
        | Layer::Pad(_)
        | Layer::Resize(_)
        | Layer::Slice(_)
        | Layer::Tile(_)
        | Layer::Gather(_)
        | Layer::ScatterAdd(_)
        | Layer::IndexAdd(_)
        | Layer::ScatterNd(_)
        | Layer::ExpandLikeLastAxis(_)
        | Layer::ReduceMean(_)
        | Layer::ReduceSum(_)
        | Layer::ReduceMax(_)
        | Layer::ReduceMin(_)
        | Layer::CumSum(_)
        | Layer::RoPE(_)
        | Layer::Where(_)
        | Layer::SkipMerge(_)
        | Layer::OpaqueSkip(_)
        | Layer::QdqPerturbation(_)
        | Layer::EpsilonExpand(_) => {
            FallsBack("exact or linear backward; no relaxation slack for an alpha to move")
        }

        Layer::NonZero(_) => {
            FallsBack("data-dependent output shape carries no relaxation for an alpha to optimize")
        }
    }
}

// ===================================================================
// 4. Block-wise CROWN — `core::graph::crown_block_wise::backward`
// ===================================================================

/// Classify `layer` against the per-block CROWN backward dispatcher.
///
/// Ground truth: `crown_backward_within_block_with_engine` in
/// `network/core/graph/crown_block_wise/backward.rs`. That dispatcher is a thin
/// shell over the BATCHED per-layer backward — it owns explicit arms for `Add`,
/// the decomposed normalizations, and alpha-`GELU`, routes a named set of
/// binary/normalization nodes straight to `apply_ibp_fallback_at_node`, and
/// sends everything else through `Layer::propagate_crown_backward_batched`.
///
/// So a variant's block-wise verdict is either an OVERRIDE the block driver
/// owns, or the verdict of the batched per-layer backward it forwards to. Both
/// cases are spelled out below, variant by variant, with **no catch-all** — the
/// delegation is expressed as an explicit [`BlockWiseArm::InheritsBatchedUnary`]
/// so a new `Layer` variant still fails to compile here.
///
/// `tests::block_wise_inherits_only_variants_without_a_graph_level_batched_arm` then
/// pins the delegated half against the batched matrix at runtime.
pub(super) fn block_wise(layer: &Layer) -> DispatchVerdict {
    match block_wise_arm(layer) {
        BlockWiseArm::Own(verdict) => verdict,
        // The block driver's unary catch-all forwards to
        // `Layer::propagate_crown_backward_batched`, so the verdict is exactly
        // what the batched per-layer dispatcher does. Every variant routed here
        // is unary, so the batched matrix's graph-dispatcher arms — which the
        // per-block driver cannot reach — never apply.
        BlockWiseArm::InheritsBatchedUnary => batched_crown_backward(layer),
    }
}

/// Which half of [`block_wise`] a variant lands in.
enum BlockWiseArm {
    /// The per-block driver owns an explicit arm for this variant.
    Own(DispatchVerdict),
    /// The per-block driver's unary catch-all forwards this variant to
    /// `Layer::propagate_crown_backward_batched`.
    InheritsBatchedUnary,
}

#[allow(clippy::match_same_arms, clippy::too_many_lines)]
fn block_wise_arm(layer: &Layer) -> BlockWiseArm {
    use BlockWiseArm::{InheritsBatchedUnary, Own};
    match layer {
        Layer::PointLinear(_) => Own(Refused("diagnostic point-only representation")),
        // --- Explicit block-wise arms ---------------------------------------
        // Add: identity backward to both inputs — the residual join, and the
        // reason the fallback arms below must NOT return early.
        Layer::Add(_) => Own(Supported),
        // Decomposed normalization CROWN backward through primitives
        // (x -> mean -> d -> d² -> var -> sqrt -> 1/std -> ...), with an
        // interval-derived bias-only fallback when the decomposition degrades.
        Layer::LayerNorm(_) | Layer::RmsNorm(_) | Layer::InstanceNorm1d(_) => Own(Supported),
        // AdaIN reuses the decomposed InstanceNorm helper through
        // `effective_instance_norm`, which only exists for the FIXED style.
        Layer::AdaIN1d(adain) if adain.requires_style_inputs() => Own(FallsBack(
            "variable-style AdaIN has no effective InstanceNorm to decompose",
        )),
        Layer::AdaIN1d(_) => Own(Supported),

        // --- Explicit block-wise fallbacks ----------------------------------
        // Named in the dispatcher's own partial-fallback arm. These are NOT
        // routed to the batched backward at all: the block driver concretizes
        // the sub-path to a constant interval and accumulates it as a
        // bias-only contribution at the block input.
        Layer::BatchNorm(_) | Layer::GroupNorm(_) => Own(FallsBack(
            "block driver concretizes these normalizations to a bias-only contribution",
        )),
        Layer::MatMul(_) | Layer::MulBinary(_) | Layer::ExpandLikeLastAxis(_) => Own(FallsBack(
            "multi-input node inside a block is concretized to a bias-only contribution",
        )),
        Layer::BilinearCrown(_) => Own(FallsBack(
            "bilinear alphas are block-state the per-block driver does not carry",
        )),
        // Sub / Concat / Where / SelfAttention reach the block driver's UNARY
        // catch-all, which calls the batched per-layer backward — and that has
        // no arm for a multi-input op, so they degrade. The batched GRAPH
        // dispatcher's arms for Sub/Concat/Where are NOT reachable from here:
        // the per-block driver never calls `dispatch_special_batched_operator`.
        // This is the one place in the matrix where a variant is `Supported`
        // for the batched graph path and `FallsBack` for the block path, so it
        // is stated as an override rather than inherited.
        Layer::Sub(_) | Layer::Concat(_) | Layer::Where(_) => Own(FallsBack(
            "per-block driver routes non-Add multi-input nodes through the UNARY batched backward, \
             which has no arm for them; the batched GRAPH dispatcher's arms are unreachable here",
        )),

        // --- Delegated to the batched per-layer backward ---------------------
        // Every variant below is unary as far as this driver is concerned, so
        // the batched matrix's verdict is the block-wise verdict verbatim.
        Layer::Linear(_)
        | Layer::IntervalConst(_)
        | Layer::Conv1d(_)
        | Layer::Conv2d(_)
        | Layer::Conv3d(_)
        | Layer::ConvTranspose1d(_)
        | Layer::ConvTranspose2d(_)
        | Layer::AveragePool(_)
        | Layer::MaxPool2d(_)
        | Layer::ReLU(_)
        | Layer::LeakyReLU(_)
        | Layer::Clip(_)
        | Layer::Elu(_)
        | Layer::Selu(_)
        | Layer::PRelu(_)
        | Layer::HardSigmoid(_)
        | Layer::HardSwish(_)
        | Layer::SiLU(_)
        | Layer::Exp(_)
        | Layer::Log(_)
        | Layer::Celu(_)
        | Layer::Mish(_)
        | Layer::LogSoftmax(_)
        | Layer::LogSumExp(_)
        | Layer::ThresholdedRelu(_)
        | Layer::Shrink(_)
        | Layer::Softsign(_)
        | Layer::Snake(_)
        | Layer::Softmax(_)
        | Layer::CausalSoftmax(_)
        | Layer::SelfAttention(_)
        | Layer::GELU(_)
        | Layer::Div(_)
        | Layer::Atan2(_)
        | Layer::MinBinary(_)
        | Layer::MaxBinary(_)
        | Layer::AddConstant(_)
        | Layer::Transpose(_)
        | Layer::Reshape(_)
        | Layer::Flatten(_)
        | Layer::Pad(_)
        | Layer::MulConstant(_)
        | Layer::Abs(_)
        | Layer::Sqrt(_)
        | Layer::DivConstant(_)
        | Layer::SubConstant(_)
        | Layer::PowConstant(_)
        | Layer::ReduceMean(_)
        | Layer::ReduceSum(_)
        | Layer::ReduceMax(_)
        | Layer::ReduceMin(_)
        | Layer::Topk(_)
        | Layer::ArgMax(_)
        | Layer::ArgMin(_)
        | Layer::ArgSort(_)
        | Layer::CumSum(_)
        | Layer::Tanh(_)
        | Layer::Sigmoid(_)
        | Layer::Erf(_)
        | Layer::Softplus(_)
        | Layer::Sin(_)
        | Layer::Cos(_)
        | Layer::Tan(_)
        | Layer::Arctan(_)
        | Layer::RoPE(_)
        | Layer::Tile(_)
        | Layer::Gather(_)
        | Layer::ScatterAdd(_)
        | Layer::IndexAdd(_)
        | Layer::ScatterNd(_)
        | Layer::Slice(_)
        | Layer::Squeeze(_)
        | Layer::Unsqueeze(_)
        | Layer::Resize(_)
        | Layer::Compare(_)
        | Layer::CompareTensor(_)
        | Layer::NonZero(_)
        | Layer::Floor(_)
        | Layer::Ceil(_)
        | Layer::Round(_)
        | Layer::Trunc(_)
        | Layer::Sign(_)
        | Layer::Reciprocal(_)
        | Layer::SkipMerge(_)
        | Layer::OpaqueSkip(_)
        | Layer::QdqPerturbation(_)
        | Layer::EpsilonExpand(_) => InheritsBatchedUnary,
    }
}

/// Whether the block-wise driver DELEGATES this variant to the batched
/// per-layer backward (rather than owning an explicit arm for it).
///
/// Used by `tests::block_wise_inherits_only_variants_without_a_graph_level_batched_arm`
/// to state the delegation as a checkable property rather than prose.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn block_wise_inherits_batched(layer: &Layer) -> bool {
    matches!(block_wise_arm(layer), BlockWiseArm::InheritsBatchedUnary)
}
