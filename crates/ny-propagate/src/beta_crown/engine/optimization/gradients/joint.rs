// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Joint α/β/λ gradient computation.

use super::super::super::BetaCrownVerifier;
use super::{infer_spatial_1d, infer_spatial_2d, propagate_linear_or_err};
use ndarray::{Array1, Array2};
use ny_core::{nan_propagating_max, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use std::sync::Arc;

use crate::beta_crown::bab_cuts::CutPool;
use crate::beta_crown::branching::SplitHistory;
use crate::beta_crown::state::{BetaState, DomainAlphaState};
use crate::layers::activations::RELU_RELAX_MIN_WIDTH;
use crate::layers::common::BoundPropagation;
use crate::{Layer, LinearBounds, Network};

impl BetaCrownVerifier {
    /// Compute joint gradients for α, β, and λ (cut) parameters.
    ///
    /// Uses the augmented chain rule to compute exact gradients:
    /// - For β: d(lb)/d(β_j) = -sign_j * d(lb)/d(lA_k[j])
    /// - For α: d(lb)/d(α_j) = d(lb)/d(lower_slope[j]) where the α controls the lower bound slope
    /// - For λ: d(lb)/d(λ_c) = constraint_min_c - bias_c (Lagrangian gradient)
    // Justification: Gradient computation needs network, input, history, layer bounds,
    // mutable alpha/beta/cut state for gradient accumulation, plus objective index.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute_joint_gradients(
        &self,
        network: &Network,
        input: &BoundedTensor,
        history: &SplitHistory,
        layer_bounds: &[Arc<BoundedTensor>],
        beta_state: &mut BetaState,
        alpha_state: &mut DomainAlphaState,
        cut_pool: &mut CutPool,
        objective_lower_idx: usize,
    ) -> Result<()> {
        // If nothing to optimize, skip gradient computation
        let has_cuts = !cut_pool.is_empty() && self.config.enable_cuts;
        if beta_state.is_empty() && alpha_state.is_empty() && !has_cuts {
            return Ok(());
        }

        self.validate_layer_bounds_len(network, layer_bounds)?;
        self.validate_split_history(network, input, layer_bounds, history)?;

        let output_dim =
            self.output_dim_from_layer_bounds(layer_bounds, "compute_joint_gradients")?;
        if objective_lower_idx >= output_dim {
            return Err(NyError::NumericalInstability(format!(
                "Objective lower index {} out of range (output_dim={})",
                objective_lower_idx, output_dim
            )));
        }

        // Build constraint lookup
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

        // Map layer -> beta entry indices
        let mut beta_entries_by_layer: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for (entry_idx, entry) in beta_state.entries.iter().enumerate() {
            beta_entries_by_layer
                .entry(entry.layer_idx)
                .or_default()
                .push(entry_idx);
        }

        // Storage for ReLU relaxation info during backward pass
        let mut relu_lower_slopes: Vec<Option<Vec<f32>>> = vec![None; network.layers.len()];
        let mut relu_upper_slopes: Vec<Option<Vec<f32>>> = vec![None; network.layers.len()];
        let mut relu_upper_intercepts: Vec<Option<Vec<f32>>> = vec![None; network.layers.len()];
        // Backward coefficient (lower_a row 0) at each ReLU, BEFORE relaxation (for α gradient).
        let mut relu_backward_coeff: Vec<Option<Vec<f32>>> = vec![None; network.layers.len()];

        // Backward pass: compute linear bounds while recording relaxation choices
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

                    // #2977: Non-finite row fallback — zero row + ±Inf bias (crown_single.rs #2681).
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
                    let pre_flat = pre_bounds.flatten();
                    let num_neurons = pre_flat.len();
                    let num_outputs = lin_bounds.num_outputs();
                    let layer_constraints = constraints.get(&layer_idx);

                    // Save backward coefficient BEFORE relaxation for alpha gradient.
                    let back_coeff: Vec<f32> = (0..num_neurons)
                        .map(|j| lin_bounds.lower_a()[[0, j]])
                        .collect();
                    relu_backward_coeff[layer_idx] = Some(back_coeff);

                    let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
                    let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
                    // f64 bias accumulators to prevent catastrophic cancellation (#2336, #1745).
                    let mut new_lower_b_f64 = lin_bounds.lower_b().mapv(|x| x as f64);
                    let mut new_upper_b_f64 = lin_bounds.upper_b().mapv(|x| x as f64);

                    // Record relaxation slopes/intercepts for forward sensitivity
                    let mut lower_slopes = vec![0.0f32; num_neurons];
                    let mut upper_slopes = vec![0.0f32; num_neurons];
                    let mut upper_intercepts = vec![0.0f32; num_neurons];

                    for j in 0..num_neurons {
                        let l = pre_flat.lower()[[j]];
                        let u = pre_flat.upper()[[j]];
                        let constraint = layer_constraints.and_then(|c| c.get(&j).copied());

                        // Determine relaxation
                        let (lower_slope, lower_intercept, upper_slope, upper_intercept) =
                            if let Some(is_active) = constraint {
                                if is_active {
                                    (1.0, 0.0, 1.0, 0.0)
                                } else {
                                    (0.0, 0.0, 0.0, 0.0)
                                }
                            } else if l >= 0.0 {
                                (1.0, 0.0, 1.0, 0.0)
                            } else if u <= 0.0 {
                                (0.0, 0.0, 0.0, 0.0)
                            } else if !l.is_finite() || !u.is_finite() {
                                // NaN/Inf bounds from upstream instability — treat as
                                // inactive (slope=0) to avoid NaN propagation through
                                // the u/(u-l) division. Ref: #2902.
                                (0.0, 0.0, 0.0, 0.0)
                            } else {
                                // Unstable: use α if available
                                // Clamp width to avoid division by zero when u ≈ l (#2697).
                                // Matches the production backward pass guard in relu_backward.rs.
                                let width = (u - l).max(RELU_RELAX_MIN_WIDTH);
                                let upper_slope_val = u / width;
                                let upper_intercept_val = -l * u / width;
                                let lower_slope_val = alpha_state.alpha(layer_idx, j);
                                (lower_slope_val, 0.0, upper_slope_val, upper_intercept_val)
                            };

                        lower_slopes[j] = lower_slope;
                        upper_slopes[j] = upper_slope;
                        upper_intercepts[j] = upper_intercept;

                        // Apply relaxation (f64 bias accumulation, #2336)
                        for i in 0..num_outputs {
                            let la_ij = lin_bounds.lower_a()[[i, j]];
                            let ua_ij = lin_bounds.upper_a()[[i, j]];

                            // Guard: skip zero coefficients to avoid IEEE 754 NaN
                            // from 0.0 * ±inf (#1736, #3066).
                            if la_ij > 0.0 {
                                new_lower_a[[i, j]] = la_ij * lower_slope;
                                new_lower_b_f64[i] += la_ij as f64 * lower_intercept;
                            } else if la_ij < 0.0 {
                                new_lower_a[[i, j]] = la_ij * upper_slope;
                                new_lower_b_f64[i] += la_ij as f64 * upper_intercept as f64;
                            }
                            // la_ij == 0.0: new_lower_a stays 0.0, no bias contribution

                            if ua_ij > 0.0 {
                                new_upper_a[[i, j]] = ua_ij * upper_slope;
                                new_upper_b_f64[i] += ua_ij as f64 * upper_intercept as f64;
                            } else if ua_ij < 0.0 {
                                new_upper_a[[i, j]] = ua_ij * lower_slope;
                                new_upper_b_f64[i] += ua_ij as f64 * lower_intercept;
                            }
                            // ua_ij == 0.0: new_upper_a stays 0.0, no bias contribution
                        }

                        // Add beta contribution
                        if let Some(signed_beta) = beta_state.signed_beta(layer_idx, j) {
                            // #2415: Skip non-finite beta to avoid poisoning the entire A-matrix.
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
                                    "Skipping non-finite beta contribution in compute_beta_gradients"
                                );
                            }
                        }
                    }

                    relu_lower_slopes[layer_idx] = Some(lower_slopes);
                    relu_upper_slopes[layer_idx] = Some(upper_slopes);
                    relu_upper_intercepts[layer_idx] = Some(upper_intercepts);

                    // f64→f32 with directed rounding (#2336): lower toward -∞, upper toward +∞.
                    let new_lower_b = new_lower_b_f64.mapv(|x| next_down_f32(x as f32));
                    let new_upper_b = new_upper_b_f64.mapv(|x| next_up_f32(x as f32));

                    // KEEP unchecked: ReLU slopes stay in [0, 1], non-finite beta
                    // contributions are skipped above, and directed rounding
                    // only widens finite f64 intermediates by one ULP.
                    lin_bounds = LinearBounds::from_parts_unchecked(
                        new_lower_a,
                        new_lower_b,
                        new_upper_a,
                        new_upper_b,
                    );
                }
                Layer::Flatten(_) | Layer::Reshape(_) => {
                    // Shape-only layers: pass through unchanged to preserve dimensions
                    // (no-op in gradient computation backward pass)
                }
                Layer::Resize(resize) => {
                    lin_bounds = resize.propagate_linear_with_bounds(&lin_bounds, pre_bounds)?;
                }
                Layer::Pad(pad) => {
                    lin_bounds = pad.propagate_linear_with_bounds(&lin_bounds, pre_bounds)?;
                }
                Layer::Transpose(t) => {
                    let mut t = t.clone();
                    t.set_input_shape(pre_bounds.shape().to_vec());
                    propagate_linear_or_err(&t, &mut lin_bounds)?;
                }
                Layer::AveragePool(pool) => {
                    // Pooling changes dimensionality; propagate linear bounds so earlier layers
                    // see the correct coefficient shapes.
                    lin_bounds = pool.propagate_linear_with_bounds(&lin_bounds, pre_bounds)?;
                }
                Layer::MaxPool2d(pool) => {
                    // MaxPool2d changes dimensionality; use its CROWN relaxation to propagate bounds.
                    lin_bounds = pool.propagate_linear_with_bounds(&lin_bounds, pre_bounds)?;
                }
                Layer::Conv2d(conv) => {
                    let (in_h, in_w) = infer_spatial_2d(
                        pre_bounds.shape(),
                        conv.in_channels(),
                        "Conv2d",
                        layer_idx,
                    )?;
                    let mut c = conv.clone();
                    c.set_input_shape(in_h, in_w);
                    propagate_linear_or_err(&c, &mut lin_bounds)?;
                }
                Layer::ConvTranspose2d(conv) => {
                    let (in_h, in_w) = infer_spatial_2d(
                        pre_bounds.shape(),
                        conv.in_channels(),
                        "ConvTranspose2d",
                        layer_idx,
                    )?;
                    let mut c = conv.clone();
                    c.set_input_shape(in_h, in_w);
                    propagate_linear_or_err(&c, &mut lin_bounds)?;
                }
                Layer::Conv1d(conv) => {
                    let in_len = infer_spatial_1d(
                        pre_bounds.shape(),
                        conv.in_channels(),
                        "Conv1d",
                        layer_idx,
                    )?;
                    let mut c = conv.clone();
                    c.set_input_length(in_len);
                    propagate_linear_or_err(&c, &mut lin_bounds)?;
                }
                Layer::ConvTranspose1d(conv) => {
                    let in_len = infer_spatial_1d(
                        pre_bounds.shape(),
                        conv.in_channels(),
                        "ConvTranspose1d",
                        layer_idx,
                    )?;
                    let mut c = conv.clone();
                    c.set_input_length(in_len);
                    propagate_linear_or_err(&c, &mut lin_bounds)?;
                }
                // === All other layers: exhaustive IBP concretization (#3424) ===
                // Every variant listed — no catch-all. Compiler catches new variants.
                Layer::Tile(_)
                | Layer::Slice(_)
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
                | Layer::RoPE(_)
                | Layer::NonZero(_)
                | Layer::SelfAttention(_)
                | Layer::ExpandLikeLastAxis(_) => {
                    // Concretize accumulated linear bounds through IBP, then continue
                    // with constant bounds (A=0, b=concretized). Sound but loose.
                    let post_bounds = layer.propagate_ibp(pre_bounds)?;
                    let concretized = lin_bounds.concretize_sound(&post_bounds);
                    let concretized_flat = concretized.flatten();
                    let num_outputs = concretized_flat.len();
                    // `len()` == `flatten().len()` (flatten preserves element count) with no allocation.
                    let num_inputs = pre_bounds.len();
                    lin_bounds = LinearBounds::new_or_conservative(
                        Array2::zeros((num_outputs, num_inputs)),
                        Array1::from_vec(concretized_flat.lower().iter().copied().collect()),
                        Array2::zeros((num_outputs, num_inputs)),
                        Array1::from_vec(concretized_flat.upper().iter().copied().collect()),
                    )?;
                }
            }
        }

        // Compute concretization point x*
        let input_flat = input.flatten();
        let final_a = lin_bounds.lower_a().row(0);
        let x_star: Vec<f32> = (0..input_flat.len())
            .map(|i| {
                if final_a[i] >= 0.0 {
                    input_flat.lower()[[i]]
                } else {
                    input_flat.upper()[[i]]
                }
            })
            .collect();
        let x_star_arr = Array1::from_vec(x_star);

        // Forward sensitivity pass in homogeneous coordinates
        // w = [x*; 1] initially, then propagate through each layer
        let mut w: Vec<f32> = x_star_arr.to_vec();
        w.push(1.0); // Augment with constant 1

        for (layer_idx, layer) in network.layers.iter().enumerate() {
            // Use references instead of clones (Arc derefs to inner BoundedTensor)
            let pre_bounds: &BoundedTensor = if layer_idx == 0 {
                input
            } else {
                layer_bounds[layer_idx - 1].as_ref()
            };

            match layer {
                Layer::Linear(linear) => {
                    let weight = &linear.weight;
                    let dim_out = weight.nrows();
                    let dim_in = weight.ncols();

                    let w_const = w[dim_in]; // The homogeneous coordinate

                    // Apply Linear transform: w_out = W @ w_in + b * w_const
                    let w_in = Array1::from_vec(w[0..dim_in].to_vec());
                    let mut w_out = weight.dot(&w_in);

                    if let Some(bias) = &linear.bias {
                        for i in 0..dim_out {
                            w_out[i] += bias[i] * w_const;
                        }
                    }

                    w = w_out.to_vec();
                    w.push(w_const);
                }
                Layer::ReLU(_) => {
                    let pre_flat = pre_bounds.flatten();
                    let num_neurons = pre_flat.len();
                    let layer_constraints = constraints.get(&layer_idx);

                    // Compute β gradients before applying ReLU transform
                    if let Some(entry_indices) = beta_entries_by_layer.get(&layer_idx) {
                        for &entry_idx in entry_indices {
                            let entry = &beta_state.entries[entry_idx];
                            let neuron_idx = entry.neuron_idx;
                            if neuron_idx < num_neurons {
                                // d(lb)/d(β_j) = -sign_j * w[neuron_idx]
                                let grad = -entry.sign * w[neuron_idx];
                                beta_state.entries[entry_idx].grad = grad;
                            }
                        }
                    }

                    // α gradient: d(lb)/d(α_j) = max(A_j, 0) * w[j]
                    // A_j = backward coeff before relaxation; w[j] = forward concretization.
                    // Ref: auto_LiRPA/operators/clampmult.py:65-127 (ClampedMultiplication.backward)
                    if let Some(back_coeff) = &relu_backward_coeff[layer_idx] {
                        for neuron_idx in 0..num_neurons {
                            if alpha_state.is_unstable(layer_idx, neuron_idx) {
                                let l = pre_flat.lower()[[neuron_idx]];
                                let u = pre_flat.upper()[[neuron_idx]];
                                let constraint =
                                    layer_constraints.and_then(|c| c.get(&neuron_idx).copied());
                                if constraint.is_none() && l < 0.0 && u > 0.0 {
                                    // NaN-safe: propagate NaN instead of silently zeroing (#2643)
                                    let grad = nan_propagating_max(back_coeff[neuron_idx], 0.0)
                                        * w[neuron_idx];
                                    alpha_state.accumulate_grad(layer_idx, neuron_idx, grad);
                                }
                            }
                        }
                    }

                    // Apply ReLU transform to sensitivity (sign-dependent branch).
                    // Ref: auto_LiRPA ClampedMultiplication (clampmult.py:37-43)
                    if let Some(l_slopes) = &relu_lower_slopes[layer_idx] {
                        let u_slopes = relu_upper_slopes[layer_idx].as_ref().ok_or_else(|| {
                            NyError::InternalError(format!(
                                "ReLU upper slopes missing at layer {layer_idx} despite lower slopes present"
                            ))
                        })?;
                        let u_intercepts =
                            relu_upper_intercepts[layer_idx].as_ref().ok_or_else(|| {
                                NyError::InternalError(format!(
                                    "ReLU upper intercepts missing at layer {layer_idx} despite lower slopes present"
                                ))
                            })?;
                        let back_coeff =
                            relu_backward_coeff[layer_idx].as_ref().ok_or_else(|| {
                                NyError::InternalError(format!(
                                    "ReLU backward coeff missing at layer {layer_idx} despite lower slopes present"
                                ))
                            })?;
                        let w_const = w[num_neurons];
                        let mut w_out = vec![0.0f32; num_neurons];

                        for j in 0..num_neurons {
                            if back_coeff[j] >= 0.0 {
                                // Lower relaxation: slope * x (intercept = 0 for ReLU)
                                w_out[j] = l_slopes[j] * w[j];
                            } else {
                                // Upper relaxation: slope * x + intercept
                                w_out[j] = u_slopes[j] * w[j] + u_intercepts[j] * w_const;
                            }
                        }

                        w = w_out;
                        w.push(w_const);
                    }
                }
                // === All other layers: exhaustive zero-sensitivity (#3424) ===
                // A=0 → no upstream dependence; disables optimization for prior parameters.
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
                    let post_bounds = layer.propagate_ibp(pre_bounds)?;
                    let dim = post_bounds.len();
                    w = vec![0.0; dim];
                    w.push(1.0);
                }
            }
        }

        // GCP-CROWN: Compute cut gradients using ReLU indicators.
        if has_cuts {
            super::cut_gradients::compute_cut_gradients(cut_pool, history, layer_bounds);
        }

        Ok(())
    }
}
