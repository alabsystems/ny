// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Legacy β-only backward propagation (test-only).
//!
//! These backward pass implementations use only β parameters (no α optimization
//! or arelu cuts). They exist for gradient computation tests and as reference
//! implementations for verifying the production α,β paths.

use std::collections::HashMap;

use crate::beta_crown::state::BetaState;
use crate::layers::activations::RELU_RELAX_MIN_WIDTH;
use crate::{BoundPropagation, Layer, LinearBounds};
use ndarray::{Array1, Array2};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::super::BetaCrownVerifier;

/// Recorded lower-bound ReLU relaxation for gradient computation.
/// Legacy: only used by test code (compute_beta_gradients in tests/gradients.rs).
#[derive(Debug, Clone)]
pub(in crate::beta_crown::engine) struct ReluLowerRelaxation {
    pub slopes: Vec<f32>,
    pub intercepts: Vec<f32>,
}

impl BetaCrownVerifier {
    /// ReLU backward pass with β constraints for split neurons (legacy).
    ///
    /// This is the core of β-CROWN: when a neuron is constrained via split,
    /// we use exact slopes (0 or 1) instead of relaxations. Additionally,
    /// we add β contributions to the A matrix for Lagrangian optimization.
    pub(in crate::beta_crown::engine) fn relu_backward_with_beta(
        &self,
        output_bounds: &LinearBounds,
        pre_bounds: &BoundedTensor,
        constraints: Option<&HashMap<usize, bool>>,
        beta_state: &BetaState,
        layer_idx: usize,
    ) -> Result<LinearBounds> {
        let pre_flat = pre_bounds.flatten();
        let num_neurons = pre_flat.len();
        let num_outputs = output_bounds.num_outputs();

        if output_bounds.num_inputs() != num_neurons {
            return Err(NyError::InternalError(format!(
                "ReLU backward (β) dimension mismatch at layer {}: output_bounds has {} inputs but layer has {} neurons",
                layer_idx,
                output_bounds.num_inputs(),
                num_neurons,
            )));
        }

        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        // f64 bias accumulators to prevent catastrophic cancellation (#2336, #1745).
        // Pattern matches common/mod.rs and compute_joint_gradients.
        let mut new_lower_b_f64 = output_bounds.lower_b().mapv(|x| x as f64);
        let mut new_upper_b_f64 = output_bounds.upper_b().mapv(|x| x as f64);

        for j in 0..num_neurons {
            let l = pre_flat.lower()[[j]];
            let u = pre_flat.upper()[[j]];

            // Check if this neuron is constrained
            let constraint = constraints.and_then(|c| c.get(&j).copied());

            // Determine relaxation based on constraint
            let (lower_slope, lower_intercept, upper_slope, upper_intercept) =
                if let Some(is_active) = constraint {
                    if is_active {
                        // Neuron is constrained to be active (x >= 0)
                        // ReLU(x) = x, so slope = 1, intercept = 0
                        (1.0, 0.0, 1.0, 0.0)
                    } else {
                        // Neuron is constrained to be inactive (x <= 0)
                        // ReLU(x) = 0, so slope = 0, intercept = 0
                        (0.0, 0.0, 0.0, 0.0)
                    }
                } else if l.is_nan() || u.is_nan() {
                    // NaN bounds → fail closed to unbounded intercepts (sound).
                    (0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY)
                } else if l >= 0.0 {
                    // Always active
                    (1.0, 0.0, 1.0, 0.0)
                } else if u <= 0.0 {
                    // Always inactive
                    (0.0, 0.0, 0.0, 0.0)
                } else if l.is_infinite() && u.is_infinite() {
                    // Both -Inf and +Inf: no finite affine upper envelope.
                    // Match relu_linear_relaxation() at relu/mod.rs:37-39. #2805
                    (0.0, 0.0, 0.0, f32::INFINITY)
                } else if u.is_infinite() {
                    // Finite l < 0 < +Inf: chord limit → slope=1, intercept=-l.
                    // Match relu_linear_relaxation() at relu/mod.rs:41-43. #2805
                    (1.0, 0.0, 1.0, -l)
                } else if l.is_infinite() {
                    // -Inf < 0 < finite u: tight upper envelope is constant y <= u.
                    // Match relu_linear_relaxation() at relu/mod.rs:45-47. #2805
                    (0.0, 0.0, 0.0, u)
                } else {
                    // Unstable: use CROWN relaxation
                    // Clamp width to avoid division by zero when u ≈ l
                    let width = (u - l).max(RELU_RELAX_MIN_WIDTH);
                    let upper_slope = u / width;
                    let upper_intercept = -l * u / width;
                    let lower_slope = if u > -l { 1.0 } else { 0.0 };
                    let lower_intercept = 0.0;
                    (lower_slope, lower_intercept, upper_slope, upper_intercept)
                };

            // Apply relaxation to each output (f64 bias accumulation, #2336)
            for i in 0..num_outputs {
                let la_ij = output_bounds.lower_a()[[i, j]];
                let ua_ij = output_bounds.upper_a()[[i, j]];

                // Lower bound computation (for lA)
                if la_ij > 0.0 {
                    new_lower_a[[i, j]] = la_ij * lower_slope;
                    new_lower_b_f64[i] += la_ij as f64 * lower_intercept as f64;
                } else if la_ij < 0.0 {
                    new_lower_a[[i, j]] = la_ij * upper_slope;
                    new_lower_b_f64[i] += la_ij as f64 * upper_intercept as f64;
                } else {
                    // Keep exact zero to avoid 0 * (+/-inf) -> NaN when NaN fallback is active.
                    new_lower_a[[i, j]] = 0.0;
                }

                // Upper bound computation (for uA)
                if ua_ij > 0.0 {
                    new_upper_a[[i, j]] = ua_ij * upper_slope;
                    new_upper_b_f64[i] += ua_ij as f64 * upper_intercept as f64;
                } else if ua_ij < 0.0 {
                    new_upper_a[[i, j]] = ua_ij * lower_slope;
                    new_upper_b_f64[i] += ua_ij as f64 * lower_intercept as f64;
                } else {
                    // Keep exact zero to avoid 0 * (+/-inf) -> NaN when NaN fallback is active.
                    new_upper_a[[i, j]] = 0.0;
                }
            }

            // Add beta contribution for constrained neurons (Lagrangian term)
            if let Some(signed_beta) = beta_state.signed_beta(layer_idx, j) {
                // #2415: Skip non-finite beta to avoid poisoning the entire A-matrix.
                // Non-finite beta means the Lagrangian multiplier optimization produced
                // invalid output; skipping preserves valid pre-beta bounds (sound).
                if signed_beta.is_finite() {
                    for i in 0..num_outputs {
                        new_lower_a[[i, j]] -= signed_beta;
                        new_upper_a[[i, j]] += signed_beta;
                    }
                } else {
                    tracing::warn!(
                        layer_idx,
                        neuron_idx = j,
                        signed_beta,
                        "Skipping non-finite beta contribution in relu_backward_with_beta"
                    );
                }
            }
        }

        // Convert f64 bias accumulators back to f32 with conservative rounding (#2336).
        let new_lower_b = new_lower_b_f64.mapv(|x| next_down_f32(x as f32));
        let new_upper_b = new_upper_b_f64.mapv(|x| next_up_f32(x as f32));

        LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
    }

    /// Backward propagation through a single layer with β constraints (legacy).
    // Justification: Bound propagation requires distinct mathematical inputs (layer,
    // output bounds, pre-activation bounds, constraints, beta state, index) that cannot
    // be meaningfully grouped — each is an independent component of the backward pass.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine) fn propagate_layer_backward_with_beta(
        &self,
        layer: &Layer,
        output_bounds: &LinearBounds,
        pre_bounds: &BoundedTensor,
        constraints: Option<&HashMap<usize, bool>>,
        beta_state: &BetaState,
        layer_idx: usize,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<LinearBounds> {
        match layer {
            Layer::Linear(linear) => {
                let weight = &linear.weight;
                let new_lower_a = output_bounds.lower_a().dot(weight);
                let new_upper_a = output_bounds.upper_a().dot(weight);

                // #2423: Bias accumulation in f64 with directed rounding to prevent
                // catastrophic cancellation. Matches crown_single.rs pattern.
                // Reference: layers/linear/bias.rs accumulate_bias_f64.
                let (new_lower_b, new_upper_b) = if let Some(bias) = &linear.bias {
                    use crate::layers::linear::bias::{
                        accumulate_bias_f64, finalize_bias_directed, BiasBlockParams,
                    };
                    let num_outputs = output_bounds.num_outputs();
                    let out_features = bias.len();
                    let mut lower_accum = vec![0.0_f64; num_outputs];
                    let mut upper_accum = vec![0.0_f64; num_outputs];
                    let block = BiasBlockParams {
                        num_outputs,
                        out_features,
                        col_offset: 0,
                    };
                    accumulate_bias_f64(
                        &mut (&mut lower_accum[..], &mut upper_accum[..]),
                        |i, j| output_bounds.lower_a()[[i, j]],
                        |i, j| output_bounds.upper_a()[[i, j]],
                        bias,
                        &block,
                    );
                    finalize_bias_directed(
                        &Array1::from(lower_accum),
                        &Array1::from(upper_accum),
                        output_bounds.lower_b(),
                        output_bounds.upper_b(),
                    )
                } else {
                    (
                        output_bounds.lower_b().clone(),
                        output_bounds.upper_b().clone(),
                    )
                };

                LinearBounds::new_or_conservative(
                    new_lower_a,
                    new_lower_b,
                    new_upper_a,
                    new_upper_b,
                )
            }
            Layer::ReLU(_) => self.relu_backward_with_beta(
                output_bounds,
                pre_bounds,
                constraints,
                beta_state,
                layer_idx,
            ),
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

                match conv_with_shape.propagate_linear_with_engine(output_bounds, engine) {
                    Ok(std::borrow::Cow::Owned(new_bounds)) => Ok(new_bounds),
                    Ok(std::borrow::Cow::Borrowed(_)) => Ok(output_bounds.clone()),
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

                match conv_with_shape.propagate_linear_with_engine(output_bounds, engine) {
                    Ok(std::borrow::Cow::Owned(new_bounds)) => Ok(new_bounds),
                    Ok(std::borrow::Cow::Borrowed(_)) => Ok(output_bounds.clone()),
                    Err(e) => Err(e),
                }
            }
            Layer::Conv1d(conv) => {
                let input_shape = pre_bounds.shape();
                let in_c = conv.in_channels();
                let in_len = Self::infer_conv1d_input_len(input_shape, in_c, "Conv1d")?;

                let mut conv_with_shape = conv.clone();
                conv_with_shape.set_input_length(in_len);

                match conv_with_shape.propagate_linear_with_engine(output_bounds, engine) {
                    Ok(std::borrow::Cow::Owned(new_bounds)) => Ok(new_bounds),
                    Ok(std::borrow::Cow::Borrowed(_)) => Ok(output_bounds.clone()),
                    Err(e) => Err(e),
                }
            }
            Layer::ConvTranspose1d(conv) => {
                let input_shape = pre_bounds.shape();
                let in_c = conv.in_channels();
                let in_len = Self::infer_conv1d_input_len(input_shape, in_c, "ConvTranspose1d")?;

                let mut conv_with_shape = conv.clone();
                conv_with_shape.set_input_length(in_len);

                match conv_with_shape.propagate_linear_with_engine(output_bounds, engine) {
                    Ok(std::borrow::Cow::Owned(new_bounds)) => Ok(new_bounds),
                    Ok(std::borrow::Cow::Borrowed(_)) => Ok(output_bounds.clone()),
                    Err(e) => Err(e),
                }
            }
            // === All other layers: exhaustive trait dispatch (#3424) ===
            // Every variant listed — no catch-all. Compiler catches new variants.
            Layer::Slice(_)
            | Layer::Gather(_)
            | Layer::ScatterAdd(_)
            | Layer::IndexAdd(_)
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
            | Layer::RoPE(_)
            | Layer::NonZero(_)
            | Layer::SelfAttention(_)
            | Layer::Compare(_)
            | Layer::CompareTensor(_) => super::layer_dispatch::beta_crown_ibp_fallback(
                layer,
                output_bounds,
                pre_bounds,
                layer_idx,
            ),
        }
    }

    /// ReLU backward pass with β constraints, recording the relaxation used for
    /// the lower bound. This is used by gradient computation.
    pub(in crate::beta_crown::engine) fn relu_backward_with_beta_record_relaxation(
        &self,
        output_bounds: &LinearBounds,
        pre_bounds: &BoundedTensor,
        constraints: Option<&HashMap<usize, bool>>,
        beta_state: &BetaState,
        layer_idx: usize,
    ) -> Result<(LinearBounds, ReluLowerRelaxation)> {
        let pre_flat = pre_bounds.flatten();
        let num_neurons = pre_flat.len();
        let num_outputs = output_bounds.num_outputs();

        if output_bounds.num_inputs() != num_neurons {
            return Err(NyError::InternalError(format!(
                "ReLU backward (β record) dimension mismatch at layer {}: output_bounds has {} inputs but layer has {} neurons",
                layer_idx,
                output_bounds.num_inputs(),
                num_neurons,
            )));
        }

        if num_outputs != 1 {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Analytical β gradient recording expects a single objective output row (got {num_outputs})"
            )));
        }

        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        // f64 bias accumulators to prevent catastrophic cancellation (#2336, #1745).
        let mut new_lower_b_f64 = output_bounds.lower_b().mapv(|x| x as f64);
        let mut new_upper_b_f64 = output_bounds.upper_b().mapv(|x| x as f64);

        let mut slopes: Vec<f32> = vec![0.0; num_neurons];
        let mut intercepts: Vec<f32> = vec![0.0; num_neurons];

        for j in 0..num_neurons {
            let l = pre_flat.lower()[[j]];
            let u = pre_flat.upper()[[j]];

            let constraint = constraints.and_then(|c| c.get(&j).copied());

            let (lower_slope, lower_intercept, upper_slope, upper_intercept) =
                if let Some(is_active) = constraint {
                    if is_active {
                        (1.0, 0.0, 1.0, 0.0)
                    } else {
                        (0.0, 0.0, 0.0, 0.0)
                    }
                } else if l.is_nan() || u.is_nan() {
                    // NaN bounds → fail closed to unbounded intercepts (sound).
                    (0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY)
                } else if l >= 0.0 {
                    (1.0, 0.0, 1.0, 0.0)
                } else if u <= 0.0 {
                    (0.0, 0.0, 0.0, 0.0)
                } else if l.is_infinite() && u.is_infinite() {
                    // Both -Inf and +Inf: no finite affine upper envelope.
                    // Match relu_linear_relaxation() at relu/mod.rs:37-39. #2805
                    (0.0, 0.0, 0.0, f32::INFINITY)
                } else if u.is_infinite() {
                    // Finite l < 0 < +Inf: chord limit → slope=1, intercept=-l.
                    // Match relu_linear_relaxation() at relu/mod.rs:41-43. #2805
                    (1.0, 0.0, 1.0, -l)
                } else if l.is_infinite() {
                    // -Inf < 0 < finite u: tight upper envelope is constant y <= u.
                    // Match relu_linear_relaxation() at relu/mod.rs:45-47. #2805
                    (0.0, 0.0, 0.0, u)
                } else {
                    // Clamp width to avoid division by zero when u ≈ l
                    let width = (u - l).max(RELU_RELAX_MIN_WIDTH);
                    let upper_slope = u / width;
                    let upper_intercept = -l * u / width;
                    let lower_slope = if u > -l { 1.0 } else { 0.0 };
                    (lower_slope, 0.0, upper_slope, upper_intercept)
                };

            for i in 0..num_outputs {
                let la_ij = output_bounds.lower_a()[[i, j]];
                let ua_ij = output_bounds.upper_a()[[i, j]];

                if la_ij > 0.0 {
                    new_lower_a[[i, j]] = la_ij * lower_slope;
                    new_lower_b_f64[i] += la_ij as f64 * lower_intercept as f64;
                } else if la_ij < 0.0 {
                    new_lower_a[[i, j]] = la_ij * upper_slope;
                    new_lower_b_f64[i] += la_ij as f64 * upper_intercept as f64;
                } else {
                    // Keep exact zero to avoid 0 * (+/-inf) -> NaN when NaN fallback is active.
                    new_lower_a[[i, j]] = 0.0;
                }

                if ua_ij > 0.0 {
                    new_upper_a[[i, j]] = ua_ij * upper_slope;
                    new_upper_b_f64[i] += ua_ij as f64 * upper_intercept as f64;
                } else if ua_ij < 0.0 {
                    new_upper_a[[i, j]] = ua_ij * lower_slope;
                    new_upper_b_f64[i] += ua_ij as f64 * lower_intercept as f64;
                } else {
                    // Keep exact zero to avoid 0 * (+/-inf) -> NaN when NaN fallback is active.
                    new_upper_a[[i, j]] = 0.0;
                }
            }

            let la_j = output_bounds.lower_a()[[0, j]];
            if la_j > 0.0 {
                slopes[j] = lower_slope;
                intercepts[j] = lower_intercept;
            } else {
                slopes[j] = upper_slope;
                intercepts[j] = upper_intercept;
            }

            if let Some(signed_beta) = beta_state.signed_beta(layer_idx, j) {
                // #2415: Skip non-finite beta to avoid poisoning the entire A-matrix.
                // Non-finite beta means the Lagrangian multiplier optimization produced
                // invalid output; skipping preserves valid pre-beta bounds (sound).
                if signed_beta.is_finite() {
                    for i in 0..num_outputs {
                        new_lower_a[[i, j]] -= signed_beta;
                        new_upper_a[[i, j]] += signed_beta;
                    }
                } else {
                    tracing::warn!(
                        layer_idx,
                        neuron_idx = j,
                        signed_beta,
                        "Skipping non-finite beta contribution in relu_backward_with_beta_record_relaxation"
                    );
                }
            }
        }

        // Convert f64 bias accumulators back to f32 with conservative rounding (#2336).
        let new_lower_b = new_lower_b_f64.mapv(|x| next_down_f32(x as f32));
        let new_upper_b = new_upper_b_f64.mapv(|x| next_up_f32(x as f32));

        Ok((
            LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)?,
            ReluLowerRelaxation { slopes, intercepts },
        ))
    }
}
