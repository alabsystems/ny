// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Layer-type backward dispatch for β-CROWN with α,β parameters.
//!
//! Dispatches backward propagation to the appropriate layer handler based on
//! layer type. For ReLU layers, uses the α,β (and optionally arelu_cut)
//! backward pass. For other layers, delegates to the layer's own CROWN
//! backward implementation, with IBP fallback for unsupported layers.

use std::collections::HashMap;

use ndarray::{Array1, Array2};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::warn;

use crate::beta_crown::state::{AreluState, BetaState, DomainAlphaState};
use crate::{BoundPropagation, Layer, LinearBounds};

use super::super::BetaCrownVerifier;

impl BetaCrownVerifier {
    /// Backward propagation through a single layer with both α and β parameters.
    ///
    /// When `arelu_state` is provided and contains cut data for this layer,
    /// uses the arelu_cut algorithm to tighten ReLU bounds during backward pass.
    // Justification: Layer backward propagation needs layer, bounds, constraints,
    // beta/alpha/arelu state, layer index, and engine — full BaB verification context.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine) fn propagate_layer_backward_with_alpha_beta(
        &self,
        layer: &Layer,
        output_bounds: &LinearBounds,
        pre_bounds: &BoundedTensor,
        constraints: Option<&HashMap<usize, bool>>,
        beta_state: &BetaState,
        alpha_state: &DomainAlphaState,
        arelu_state: Option<&AreluState>,
        layer_idx: usize,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<LinearBounds> {
        match layer {
            Layer::Linear(linear) => linear
                .propagate_linear_with_engine(output_bounds, engine)
                .map(|cow| cow.into_owned()),
            Layer::ReLU(_) => {
                // Use arelu_cut integration if cuts are active for this layer
                if let Some(arelu) = arelu_state {
                    if arelu.has_cut_mask.contains_key(&layer_idx) {
                        return self.relu_backward_with_alpha_beta_arelu(
                            output_bounds,
                            pre_bounds,
                            constraints,
                            beta_state,
                            alpha_state,
                            arelu,
                            layer_idx,
                        );
                    }
                }
                // Fall back to standard backward pass
                self.relu_backward_with_alpha_beta(
                    output_bounds,
                    pre_bounds,
                    constraints,
                    beta_state,
                    alpha_state,
                    layer_idx,
                )
            }
            Layer::Flatten(_) | Layer::Reshape(_) | Layer::ExpandLikeLastAxis(_) => {
                Ok(output_bounds.clone())
            }
            Layer::Resize(resize) => resize.propagate_linear_with_bounds(output_bounds, pre_bounds),
            Layer::Pad(pad) => pad.propagate_linear_with_bounds(output_bounds, pre_bounds),
            Layer::Transpose(t) => {
                // Transpose permutes elements — must permute CROWN coefficient columns.
                // Reference: graph_crown/propagation.rs and graph_alpha/backward.rs
                let mut transpose_with_shape = t.clone();
                transpose_with_shape.set_input_shape(pre_bounds.shape().to_vec());
                match transpose_with_shape.propagate_linear(output_bounds) {
                    Ok(std::borrow::Cow::Owned(new_bounds)) => Ok(new_bounds),
                    Ok(std::borrow::Cow::Borrowed(_)) => Ok(output_bounds.clone()),
                    Err(e) => Err(e),
                }
            }
            Layer::Tile(t) => {
                // Tile needs input_shape before dispatch (same pattern as Transpose).
                // Reference: ibp.rs Tile arm, dispatch.rs Tile arm (#3114).
                let mut tile_with_shape = t.clone();
                tile_with_shape.set_input_shape(pre_bounds.shape().to_vec());
                match tile_with_shape.propagate_linear(output_bounds) {
                    Ok(std::borrow::Cow::Owned(new_bounds)) => Ok(new_bounds),
                    Ok(std::borrow::Cow::Borrowed(_)) => Ok(output_bounds.clone()),
                    Err(e) => Err(e),
                }
            }
            Layer::AveragePool(pool) => {
                pool.propagate_linear_with_bounds(output_bounds, pre_bounds)
            }
            Layer::MaxPool2d(pool) => pool.propagate_linear_with_bounds(output_bounds, pre_bounds),
            Layer::Conv2d(conv) => {
                let input_shape = pre_bounds.shape();
                let in_c = conv.in_channels();
                let (in_h, in_w) = Self::infer_conv2d_input_hw(input_shape, in_c, "Conv2d")?;

                let mut conv_with_shape = conv.clone();
                conv_with_shape.set_input_shape(in_h, in_w);
                // #3399: Thread engine through Conv2d backward for GPU acceleration.
                // #3813: ShapeMismatch triggers IBP concretization fallback.
                match conv_with_shape.propagate_linear_with_engine(output_bounds, engine) {
                    Ok(std::borrow::Cow::Owned(new_bounds)) => Ok(new_bounds),
                    Ok(std::borrow::Cow::Borrowed(_)) => Ok(output_bounds.clone()),
                    Err(NyError::ShapeMismatch { .. }) => {
                        beta_crown_ibp_concretize(layer, output_bounds, pre_bounds, layer_idx)
                    }
                    Err(e) => Err(e),
                }
            }
            Layer::ConvTranspose2d(conv) => {
                let input_shape = pre_bounds.shape();
                let in_c = conv.in_channels();
                let (in_h, in_w) =
                    Self::infer_conv2d_input_hw(input_shape, in_c, "ConvTranspose2d")?;

                let mut conv_with_shape = conv.clone();
                conv_with_shape.set_input_shape(in_h, in_w);
                // #3813: ShapeMismatch triggers IBP concretization fallback.
                match conv_with_shape.propagate_linear_with_engine(output_bounds, engine) {
                    Ok(std::borrow::Cow::Owned(new_bounds)) => Ok(new_bounds),
                    Ok(std::borrow::Cow::Borrowed(_)) => Ok(output_bounds.clone()),
                    Err(NyError::ShapeMismatch { .. }) => {
                        beta_crown_ibp_concretize(layer, output_bounds, pre_bounds, layer_idx)
                    }
                    Err(e) => Err(e),
                }
            }
            Layer::Conv1d(conv) => {
                let input_shape = pre_bounds.shape();
                let in_c = conv.in_channels();
                let in_len = Self::infer_conv1d_input_len(input_shape, in_c, "Conv1d")?;

                let mut conv_with_shape = conv.clone();
                conv_with_shape.set_input_length(in_len);
                // #3598: Thread engine through Conv1d backward for GPU acceleration.
                // #3813: ShapeMismatch triggers IBP concretization fallback.
                match conv_with_shape.propagate_linear_with_engine(output_bounds, engine) {
                    Ok(std::borrow::Cow::Owned(new_bounds)) => Ok(new_bounds),
                    Ok(std::borrow::Cow::Borrowed(_)) => Ok(output_bounds.clone()),
                    Err(NyError::ShapeMismatch { .. }) => {
                        beta_crown_ibp_concretize(layer, output_bounds, pre_bounds, layer_idx)
                    }
                    Err(e) => Err(e),
                }
            }
            Layer::ConvTranspose1d(conv) => {
                let input_shape = pre_bounds.shape();
                let in_c = conv.in_channels();
                let in_len = Self::infer_conv1d_input_len(input_shape, in_c, "ConvTranspose1d")?;

                let mut conv_with_shape = conv.clone();
                conv_with_shape.set_input_length(in_len);
                // #3598: Thread engine through ConvTranspose1d backward for GPU acceleration.
                // #3813: ShapeMismatch triggers IBP concretization fallback.
                match conv_with_shape.propagate_linear_with_engine(output_bounds, engine) {
                    Ok(std::borrow::Cow::Owned(new_bounds)) => Ok(new_bounds),
                    Ok(std::borrow::Cow::Borrowed(_)) => Ok(output_bounds.clone()),
                    Err(NyError::ShapeMismatch { .. }) => {
                        beta_crown_ibp_concretize(layer, output_bounds, pre_bounds, layer_idx)
                    }
                    Err(e) => Err(e),
                }
            }
            // === All other layers: exhaustive trait dispatch (#3424) ===
            // Every variant listed — no catch-all. Compiler catches new variants.
            Layer::Slice(_)
            | Layer::Gather(_)
            | Layer::ScatterNd(_)
            | Layer::Add(_)
            | Layer::Sub(_)
            | Layer::Concat(_)
            | Layer::MatMul(_)
            | Layer::BilinearCrown(_)
            | Layer::SkipMerge(_)
            | Layer::OpaqueSkip(_)
            | Layer::QdqPerturbation(_)
            | Layer::MulBinary(_)
            | Layer::Where(_)
            | Layer::Div(_)
            | Layer::Atan2(_)
            | Layer::MinBinary(_)
            | Layer::MaxBinary(_)
            | Layer::GELU(_)
            | Layer::SiLU(_)
            | Layer::Tanh(_)
            | Layer::Sigmoid(_)
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
            | Layer::CompareTensor(_)
            | Layer::Softmax(_)
            | Layer::CausalSoftmax(_)
            | Layer::LogSoftmax(_)
            | Layer::LogSumExp(_)
            | Layer::LayerNorm(_)
            | Layer::RmsNorm(_)
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
            | Layer::Topk(_)
            | Layer::ArgMax(_)
            | Layer::ArgMin(_)
            | Layer::ArgSort(_)
            | Layer::Squeeze(_)
            | Layer::Unsqueeze(_)
            | Layer::ScatterAdd(_)
            | Layer::IndexAdd(_)
            | Layer::RoPE(_)
            | Layer::NonZero(_)
            | Layer::SelfAttention(_) => {
                beta_crown_ibp_fallback(layer, output_bounds, pre_bounds, layer_idx)
            }
        }
    }
}

/// IBP fallback for unsupported layers in β-CROWN backward pass.
///
/// Attempts CROWN backward via the trait method; on UnsupportedOp,
/// UnsupportedConfiguration, or NumericalInstability, concretizes to
/// constant linear bounds (A=0, b=concretized IBP). Sound but maximally loose.
pub(super) fn beta_crown_ibp_fallback(
    layer: &Layer,
    output_bounds: &LinearBounds,
    pre_bounds: &BoundedTensor,
    layer_idx: usize,
) -> Result<LinearBounds> {
    match layer.propagate_crown_backward(output_bounds, Some(pre_bounds)) {
        Ok(new_bounds) => Ok(new_bounds),
        // #3166: Catch UnsupportedOp and UnsupportedConfiguration.
        // #2888: NumericalInstability also triggers IBP fallback — non-finite
        // pre-activation bounds should degrade gracefully, not abort the BaB pass.
        Err(
            NyError::UnsupportedOp(ref msg)
            | NyError::UnsupportedConfiguration(ref msg)
            | NyError::NumericalInstability(ref msg),
        ) => {
            warn!(
                "β-CROWN backward: layer {} ({}) unsupported/unstable, \
                 concretizing to IBP constant bounds: {}",
                layer_idx,
                layer.layer_type(),
                msg
            );
            beta_crown_ibp_concretize(layer, output_bounds, pre_bounds, layer_idx)
        }
        // #3813: ShapeMismatch triggers IBP fallback (RSPLITTER dimension changes).
        // IBP concretization is always sound.
        Err(NyError::ShapeMismatch {
            ref expected,
            ref got,
        }) => {
            warn!(
                "β-CROWN backward: layer {} ({}) shape mismatch \
                 (expected {:?}, got {:?}), concretizing to IBP constant bounds",
                layer_idx,
                layer.layer_type(),
                expected,
                got
            );
            beta_crown_ibp_concretize(layer, output_bounds, pre_bounds, layer_idx)
        }
        Err(e) => Err(e),
    }
}

/// IBP concretization fallback: concretize accumulated CROWN bounds through
/// IBP at a single layer. Sound but maximally loose.
fn beta_crown_ibp_concretize(
    layer: &Layer,
    output_bounds: &LinearBounds,
    pre_bounds: &BoundedTensor,
    _layer_idx: usize,
) -> Result<LinearBounds> {
    let post_bounds = layer.propagate_ibp(pre_bounds)?;
    // #2239: directed rounding on f64→f32 for soundness.
    let concretized = output_bounds.concretize_sound(&post_bounds);
    let concretized_flat = concretized.flatten();
    let num_outputs = concretized_flat.len();
    // `len()` == `flatten().len()` (flatten preserves element count) with no allocation.
    let num_inputs = pre_bounds.len();
    LinearBounds::new_or_conservative(
        Array2::zeros((num_outputs, num_inputs)),
        Array1::from_vec(concretized_flat.lower().iter().copied().collect()),
        Array2::zeros((num_outputs, num_inputs)),
        Array1::from_vec(concretized_flat.upper().iter().copied().collect()),
    )
}
