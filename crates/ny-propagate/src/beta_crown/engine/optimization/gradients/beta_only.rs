// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Beta-only gradient computation (test-only path).

use super::super::super::backward::ReluLowerRelaxation;
use super::super::super::BetaCrownVerifier;
use ndarray::{Array1, Array2};
use ny_core::Result;
use ny_tensor::BoundedTensor;
use std::sync::Arc;

use crate::beta_crown::branching::SplitHistory;
use crate::beta_crown::state::BetaState;
use crate::{Layer, LinearBounds, Network};

impl BetaCrownVerifier {
    /// Compute (sub)gradients of the active scalar lower bound w.r.t. β parameters.
    ///
    /// For each constrained neuron (layer k, neuron j), β modifies the lower-bound
    /// backward coefficients as `lA_k[j] -= sign_j * β_j`, so:
    ///
    /// d(lb)/d(β_j) = -sign_j * d(lb)/d(lA_k[j])
    ///
    /// We compute `d(lb)/d(lA_k[j])` exactly for the current piecewise-linear relaxation
    /// choices by:
    /// - Running a backward pass to the input while recording, per ReLU layer, which
    ///   (slope, intercept) branch was selected for the lower bound.
    /// - Selecting the concretization point x* from the final input coefficients.
    /// - Running a forward sensitivity pass in homogeneous coordinates so that bias and
    ///   intercept contributions are included exactly.
    #[cfg(test)]
    pub(crate) fn compute_beta_gradients(
        &self,
        network: &Network,
        input: &BoundedTensor,
        history: &SplitHistory,
        layer_bounds: &[Arc<BoundedTensor>],
        beta_state: &mut BetaState,
        objective_lower_idx: usize,
    ) -> Result<()> {
        if beta_state.is_empty() {
            return Ok(());
        }

        self.validate_layer_bounds_len(network, layer_bounds)?;
        self.validate_split_history(network, input, layer_bounds, history)?;

        let output_dim =
            self.output_dim_from_layer_bounds(layer_bounds, "compute_beta_gradients")?;
        if objective_lower_idx >= output_dim {
            return Err(ny_core::NyError::NumericalInstability(format!(
                "Objective lower index {} out of range (output_dim={})",
                objective_lower_idx, output_dim
            )));
        }

        // Build constraint lookup: layer_idx -> neuron_idx -> is_active
        let mut constraints: std::collections::HashMap<
            usize,
            std::collections::HashMap<usize, bool>,
        > = std::collections::HashMap::new();
        for c in &history.constraints {
            constraints
                .entry(c.layer_idx)
                .or_default()
                .insert(c.neuron_idx, c.is_active);
        }

        // Map layer -> beta entry indices for efficient gradient fill during forward sensitivity pass.
        let mut beta_entries_by_layer: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for (entry_idx, entry) in beta_state.entries.iter().enumerate() {
            beta_entries_by_layer
                .entry(entry.layer_idx)
                .or_default()
                .push(entry_idx);
        }

        let mut relu_relaxations: Vec<Option<ReluLowerRelaxation>> =
            vec![None; network.layers.len()];

        // Backward pass for the active scalar lower-bound objective output element.
        // We record the lower-bound ReLU relaxation (slope/intercept choices) for each ReLU layer
        // under the current coefficients, since those choices determine the piecewise-linear
        // bound and its (sub)gradient.
        let mut lower_a = Array2::<f32>::zeros((1, output_dim));
        lower_a[[0, objective_lower_idx]] = 1.0;
        // Phase 4 audit: identity objective row + zero bias — trivially finite.
        let mut lin_bounds = LinearBounds::new(
            lower_a.clone(),
            Array1::<f32>::zeros(1),
            lower_a,
            Array1::<f32>::zeros(1),
        )?;

        for (layer_idx, layer) in network.layers.iter().enumerate().rev() {
            // Use references instead of clones (Arc derefs to inner BoundedTensor)
            let pre_bounds: &BoundedTensor = if layer_idx == 0 {
                input
            } else {
                layer_bounds[layer_idx - 1].as_ref()
            };

            let layer_constraints = constraints.get(&layer_idx);

            match layer {
                Layer::Linear(linear) => {
                    let weight = &linear.weight;
                    let mut new_lower_a = lin_bounds.lower_a().dot(weight);
                    let mut new_upper_a = lin_bounds.upper_a().dot(weight);

                    let mut new_lower_b = if let Some(bias) = &linear.bias {
                        lin_bounds.lower_b() + &lin_bounds.lower_a().dot(bias)
                    } else {
                        lin_bounds.lower_b().clone()
                    };

                    let mut new_upper_b = if let Some(bias) = &linear.bias {
                        lin_bounds.upper_b() + &lin_bounds.upper_a().dot(bias)
                    } else {
                        lin_bounds.upper_b().clone()
                    };

                    // #2977: Non-finite row fallback (matching crown_single.rs #2681).
                    // Dot products of finite weights can overflow to Inf in f32. For rows
                    // with non-finite coefficients, zero the entire row and set bias to
                    // ±Inf — sound but maximally loose for that output.
                    let num_outputs = new_lower_a.nrows();
                    let num_inputs = new_lower_a.ncols();
                    for i in 0..num_outputs {
                        if new_lower_a.row(i).iter().any(|v| !v.is_finite()) {
                            for j in 0..num_inputs {
                                new_lower_a[[i, j]] = 0.0;
                            }
                            new_lower_b[i] = f32::NEG_INFINITY;
                        }
                        if new_upper_a.row(i).iter().any(|v| !v.is_finite()) {
                            for j in 0..num_inputs {
                                new_upper_a[[i, j]] = 0.0;
                            }
                            new_upper_b[i] = f32::INFINITY;
                        }
                    }

                    lin_bounds = LinearBounds::new_or_conservative(
                        new_lower_a,
                        new_lower_b,
                        new_upper_a,
                        new_upper_b,
                    )?;
                }
                Layer::ReLU(_) => {
                    let (new_bounds, relaxation) = self.relu_backward_with_beta_record_relaxation(
                        &lin_bounds,
                        pre_bounds,
                        layer_constraints,
                        beta_state,
                        layer_idx,
                    )?;
                    relu_relaxations[layer_idx] = Some(relaxation);
                    lin_bounds = new_bounds;
                }
                // === All other layers: exhaustive error (#3424) ===
                // Analytical β gradients only support Linear/ReLU networks.
                Layer::Conv1d(_)
                | Layer::Conv2d(_)
                | Layer::ConvTranspose1d(_)
                | Layer::ConvTranspose2d(_)
                | Layer::Transpose(_)
                | Layer::Tile(_)
                | Layer::Slice(_)
                | Layer::Gather(_)
                | Layer::ScatterAdd(_)
                | Layer::IndexAdd(_)
                | Layer::ScatterNd(_)
                | Layer::Pad(_)
                | Layer::Resize(_)
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
                | Layer::Flatten(_)
                | Layer::Reshape(_)
                | Layer::ExpandLikeLastAxis(_)
                | Layer::Squeeze(_)
                | Layer::Unsqueeze(_)
                | Layer::AveragePool(_)
                | Layer::MaxPool2d(_)
                | Layer::RoPE(_)
                | Layer::NonZero(_)
                | Layer::SelfAttention(_) => {
                    return Err(ny_core::NyError::UnsupportedConfiguration(format!(
                        "Analytical β gradients only supported for Linear/ReLU networks (saw {:?})",
                        layer
                    )));
                }
            }
        }

        // Choose the concretization point x* based on the final input-layer coefficients.
        // lb = lA(x*) + lB, where x*_i is lower_i if lA_i >= 0 else upper_i.
        let input_flat = input.flatten();
        let input_dim = input_flat.len();
        if lin_bounds.lower_a().ncols() != input_dim {
            return Err(ny_core::NyError::shape_mismatch(
                vec![input_dim],
                vec![lin_bounds.lower_a().ncols()],
            ));
        }

        let mut x_star = Array1::<f32>::zeros(input_dim + 1);
        for i in 0..input_dim {
            let coeff = lin_bounds.lower_a()[[0, i]];
            x_star[i] = if coeff >= 0.0 {
                input_flat.lower()[[i]]
            } else {
                input_flat.upper()[[i]]
            };
        }
        x_star[input_dim] = 1.0;

        // Forward sensitivity pass in the augmented space (homogeneous coordinates).
        // This yields w_k = P_k * x_star for each layer input space, where P_k is the product
        // of the recorded piecewise-linear bound transforms up to that layer.
        //
        // For any coefficient a_k[j] at layer k input, d(lb)/d(a_k[j]) = w_k[j], including
        // both coefficient->input effects and coefficient->bias/intercept effects.
        let mut w = x_star;
        for (layer_idx, layer) in network.layers.iter().enumerate() {
            if let Some(entry_indices) = beta_entries_by_layer.get(&layer_idx) {
                let w_len = w.len();
                for &entry_idx in entry_indices {
                    let neuron_idx = beta_state.entries[entry_idx].neuron_idx;
                    if neuron_idx + 1 >= w_len {
                        return Err(ny_core::NyError::NumericalInstability(format!(
                            "β entry neuron index {} out of range for layer {} (w_len={})",
                            neuron_idx, layer_idx, w_len
                        )));
                    }
                    let sign = beta_state.entries[entry_idx].sign;
                    beta_state.entries[entry_idx].grad = -sign * w[neuron_idx];
                }
            }

            match layer {
                Layer::Linear(linear) => {
                    let in_dim = linear.weight.ncols();
                    let out_dim = linear.weight.nrows();
                    if w.len() != in_dim + 1 {
                        return Err(ny_core::NyError::shape_mismatch(
                            vec![in_dim + 1],
                            vec![w.len()],
                        ));
                    }

                    let w_in = w.slice(ndarray::s![..in_dim]).to_owned();
                    let w_const = w[in_dim];
                    let mut w_out = linear.weight.dot(&w_in);
                    if let Some(bias) = &linear.bias {
                        w_out = &w_out + &(bias * w_const);
                    }

                    let mut w_aug = Array1::<f32>::zeros(out_dim + 1);
                    w_aug.slice_mut(ndarray::s![..out_dim]).assign(&w_out);
                    w_aug[out_dim] = w_const;
                    w = w_aug;
                }
                Layer::ReLU(_) => {
                    let relaxation = relu_relaxations[layer_idx].as_ref().ok_or_else(|| {
                        ny_core::NyError::NumericalInstability(format!(
                            "Missing recorded ReLU relaxation for layer {}",
                            layer_idx
                        ))
                    })?;

                    let n = relaxation.slopes.len();
                    if w.len() != n + 1 {
                        return Err(ny_core::NyError::shape_mismatch(vec![n + 1], vec![w.len()]));
                    }

                    let w_const = w[n];
                    let mut w_aug = Array1::<f32>::zeros(n + 1);
                    for j in 0..n {
                        w_aug[j] = relaxation.slopes[j] * w[j] + relaxation.intercepts[j] * w_const;
                    }
                    w_aug[n] = w_const;
                    w = w_aug;
                }
                // === All other layers: exhaustive error (#3424) ===
                Layer::Conv1d(_)
                | Layer::Conv2d(_)
                | Layer::ConvTranspose1d(_)
                | Layer::ConvTranspose2d(_)
                | Layer::Transpose(_)
                | Layer::Tile(_)
                | Layer::Slice(_)
                | Layer::Gather(_)
                | Layer::ScatterAdd(_)
                | Layer::IndexAdd(_)
                | Layer::ScatterNd(_)
                | Layer::Pad(_)
                | Layer::Resize(_)
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
                | Layer::Flatten(_)
                | Layer::Reshape(_)
                | Layer::ExpandLikeLastAxis(_)
                | Layer::Squeeze(_)
                | Layer::Unsqueeze(_)
                | Layer::AveragePool(_)
                | Layer::MaxPool2d(_)
                | Layer::RoPE(_)
                | Layer::NonZero(_)
                | Layer::SelfAttention(_) => {
                    return Err(ny_core::NyError::UnsupportedConfiguration(format!(
                        "Analytical β gradients only supported for Linear/ReLU networks (saw {:?})",
                        layer
                    )));
                }
            }
        }

        Ok(())
    }
}
