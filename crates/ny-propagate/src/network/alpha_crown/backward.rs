// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared backward pass for alpha-CROWN.
//!
//! The three alpha-CROWN backward passes (main optimization loop, single-pass
//! for numerical gradients, and intermediates pass for chain-rule) share
//! identical layer-iteration logic. This module extracts that shared skeleton
//! into `backward_pass_core`, parameterized by `BackwardPassConfig` to capture
//! the small differences between call sites.
//!
//! Reference: designs/2026-02-14-alpha-crown-backward-dedup.md

use std::borrow::Cow;
use std::collections::HashMap;
use std::time::Instant;

use super::helpers::build_layer_to_relu_idx;
use ndarray::{Array1, Array2};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::warn;

use crate::bounds::{AlphaState, LinearBounds};
use crate::layers::{BoundPropagation, Layer};
use crate::network::graph_alpha::invprop_backward::augment_bounds_with_constraints;

/// Configuration for the shared backward pass.
pub(super) struct BackwardPassConfig<'a> {
    /// Whether to track gradients at ReLU layers (main optimization loop).
    pub track_gradients: bool,
    /// Whether to store A matrices and pre-ReLU bounds (intermediates pass).
    pub store_intermediates: bool,
    /// Whether to clone bounds before INVPROP augmentation for best-of comparison.
    pub best_of_oc: bool,
    /// Optional GEMM acceleration engine for linear layer propagation.
    pub engine: Option<&'a dyn GemmEngine>,
    /// Literal verifier authority for this backward pass.
    pub deadline: Option<Instant>,
    /// Mapping from layer index to ReLU index.
    pub layer_to_relu_idx: &'a HashMap<usize, usize>,
    /// Indices of ReLU layers in the network.
    pub relu_layer_indices: &'a [usize],
}

/// Output from a successful backward pass.
pub(super) struct BackwardPassOutput {
    /// Final linear bounds after backward propagation.
    pub linear_bounds: LinearBounds,
    /// Gradients at each ReLU layer for lower alpha path (populated when `track_gradients` is true).
    pub gradients: Vec<Array1<f32>>,
    /// Gradients at each ReLU layer for upper alpha path (populated when `track_gradients` is true).
    /// Used for independent optimization of alpha_upper (#3393).
    pub gradients_upper: Vec<Array1<f32>>,
    /// A matrices at each ReLU (populated when `store_intermediates` is true).
    /// In reverse layer order (caller should reverse for forward order).
    pub a_at_relu: Vec<Array2<f32>>,
    /// Pre-ReLU bounds at each ReLU (populated when `store_intermediates` is true).
    /// In reverse layer order (caller should reverse for forward order).
    pub pre_relu_bounds: Vec<(Array1<f32>, Array1<f32>)>,
    /// Linear bounds before INVPROP augmentation (populated when `best_of_oc` is true).
    pub bounds_without_oc: Option<LinearBounds>,
}

/// Result type for the backward pass: either success or unsupported-layer fallback.
pub(super) enum BackwardPassResult {
    /// Backward pass completed successfully.
    Success(Box<BackwardPassOutput>),
    /// An unsupported layer was encountered; caller should fall back to CROWN.
    Fallback,
}

#[inline]
fn check_backward_deadline(deadline: Option<Instant>, phase: &str) -> Result<()> {
    if deadline.is_some_and(|value| Instant::now() >= value) {
        Err(NyError::DeadlineExceeded(format!(
            "sequential alpha-CROWN backward: deadline exceeded {phase}"
        )))
    } else {
        Ok(())
    }
}

/// Propagate a generic (non-Linear, non-ReLU) layer backward via the
/// `propagate_crown_backward` trait method, with standard error handling.
///
/// Returns `Ok(None)` on success (bounds updated in-place via return value),
/// `Ok(Some(Fallback))` on unsupported/unstable layers, or `Err` on critical errors.
fn propagate_generic_layer_backward(
    layer: &Layer,
    linear_bounds: &LinearBounds,
    pre_activation: &BoundedTensor,
    layer_idx: usize,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Result<(LinearBounds, Option<BackwardPassResult>)> {
    check_backward_deadline(deadline, "before generic layer dispatch")?;

    // Conv layers: use engine-aware path for GPU acceleration (#3598).
    let backward_result = match layer {
        Layer::Conv1d(c) => {
            let input_shape = pre_activation.shape();
            let in_len = input_shape.last().copied().unwrap_or(0);
            let mut conv = c.clone();
            conv.set_input_length(in_len);
            conv.propagate_linear_with_engine_and_deadline(linear_bounds, engine, deadline)
                .map(|cow| cow.into_owned())
        }
        Layer::ConvTranspose1d(c) => {
            let input_shape = pre_activation.shape();
            let in_len = input_shape.last().copied().unwrap_or(0);
            let mut conv = c.clone();
            conv.set_input_length(in_len);
            conv.propagate_linear_with_engine_and_deadline(linear_bounds, engine, deadline)
                .map(|cow| cow.into_owned())
        }
        Layer::Conv2d(c) => {
            let input_shape = pre_activation.shape();
            let (in_h, in_w) = if input_shape.len() >= 3 {
                (
                    input_shape[input_shape.len() - 2],
                    input_shape[input_shape.len() - 1],
                )
            } else {
                (0, 0)
            };
            let mut conv = c.clone();
            conv.set_input_shape(in_h, in_w);
            conv.propagate_linear_with_engine_and_deadline(linear_bounds, engine, deadline)
                .map(|cow| cow.into_owned())
        }
        Layer::ConvTranspose2d(c) => {
            let input_shape = pre_activation.shape();
            let (in_h, in_w) = if input_shape.len() >= 3 {
                (
                    input_shape[input_shape.len() - 2],
                    input_shape[input_shape.len() - 1],
                )
            } else {
                (0, 0)
            };
            let mut conv = c.clone();
            conv.set_input_shape(in_h, in_w);
            conv.propagate_linear_with_engine_and_deadline(linear_bounds, engine, deadline)
                .map(|cow| cow.into_owned())
        }
        _ => layer.propagate_crown_backward(linear_bounds, Some(pre_activation)),
    };
    check_backward_deadline(deadline, "after generic layer dispatch")?;
    match backward_result {
        Ok(next) => Ok((next, None)),
        // #3166: Catch UnsupportedConfiguration alongside UnsupportedOp.
        // #2888: NumericalInstability also triggers fallback.
        // #3813: ShapeMismatch from Dense Conv2d backward when graph restructuring
        // (e.g., RSPLITTER) changes intermediate dimensions. IBP fallback is sound.
        Err(
            e @ NyError::UnsupportedOp(_)
            | e @ NyError::UnsupportedConfiguration(_)
            | e @ NyError::NumericalInstability(_)
            | e @ NyError::ShapeMismatch { .. },
        ) => {
            warn!(
                "α-CROWN backward: layer {} ({}) unsupported/unstable: {}",
                layer_idx,
                layer.layer_type(),
                e
            );
            Ok((linear_bounds.clone(), Some(BackwardPassResult::Fallback)))
        }
        // #3107: LayerError may wrap critical errors — inspect source before fallback.
        Err(NyError::LayerError { source, .. })
            if matches!(
                source.as_ref(),
                NyError::SoundnessRefusal(_)
                    | NyError::InternalError(_)
                    | NyError::DeadlineExceeded(_)
            ) =>
        {
            Err(*source)
        }
        Err(e @ NyError::LayerError { .. }) => {
            warn!(
                "α-CROWN backward: layer {} ({}) unsupported (wrapped): {}",
                layer_idx,
                layer.layer_type(),
                e
            );
            Ok((linear_bounds.clone(), Some(BackwardPassResult::Fallback)))
        }
        Err(err) => Err(err),
    }
}

/// Shared backward pass through a sequential network.
///
/// Iterates layers in reverse, dispatching Linear/ReLU/other layers. The
/// `config` struct controls which optional data is collected at each ReLU
/// and how INVPROP augmentation is applied.
///
/// Returns `BackwardPassResult::Fallback` when an unsupported layer is hit,
/// allowing the caller to handle the CROWN fallback appropriately (the main
/// and single passes return full CROWN bounds; the intermediates pass returns
/// constant `LinearBounds`).
pub(super) fn backward_pass_core(
    layers: &[Layer],
    input: &BoundedTensor,
    layer_bounds: &[BoundedTensor],
    alpha_state: &AlphaState,
    output_dim: usize,
    config: &BackwardPassConfig<'_>,
) -> Result<BackwardPassResult> {
    check_backward_deadline(config.deadline, "before initialization")?;
    let mut linear_bounds = LinearBounds::identity(output_dim);

    // INVPROP assume-violation dual, folded into the OUTPUT IDENTITY SEED (re-seed).
    // The merged coefficients `I +/- C^T gamma` then propagate through the ordinary
    // sign-aware backward loop (each nonlinearity selects its relaxation branch by
    // the augmented coefficient's sign), and the attached certified per-coefficient
    // error is carried outward (`Sigma_k err_in . |W|`) to concretization. With
    // gamma == 0 (un-optimized) this is the identity map => byte-identical baseline.
    if let Some(ref state) = alpha_state.invprop_state {
        if let Some(gammas) = state.layer_gammas(crate::invprop::INVPROP_OUTPUT_SEED) {
            if let Some((gammas_lower, gammas_upper)) = gammas
                .active
                .then(|| gammas.checked_bound_gammas())
                .flatten()
            {
                linear_bounds = augment_bounds_with_constraints(
                    &linear_bounds,
                    &state.constraints,
                    &gammas_lower.to_owned(),
                    &gammas_upper.to_owned(),
                );
            }
        }
    }

    // Initialize gradient storage if tracking (both lower and upper alpha paths, #3393)
    let mut gradients: Vec<Array1<f32>> = if config.track_gradients {
        config
            .relu_layer_indices
            .iter()
            .map(|&relu_idx| {
                let pre_act = if relu_idx == 0 {
                    input
                } else {
                    &layer_bounds[relu_idx - 1]
                };
                Array1::zeros(pre_act.len())
            })
            .collect()
    } else {
        Vec::new()
    };
    let mut gradients_upper: Vec<Array1<f32>> = if config.track_gradients {
        gradients.iter().map(|g| Array1::zeros(g.len())).collect()
    } else {
        Vec::new()
    };

    // Initialize intermediate storage if collecting
    let mut a_at_relu: Vec<Array2<f32>> = Vec::new();
    let mut pre_relu_bounds: Vec<(Array1<f32>, Array1<f32>)> = Vec::new();

    let mut bounds_without_oc: Option<LinearBounds> = None;

    // Backward pass through layers
    for (layer_idx, layer) in layers.iter().enumerate().rev() {
        check_backward_deadline(config.deadline, "between layers")?;
        let pre_activation = if layer_idx == 0 {
            input
        } else {
            &layer_bounds[layer_idx - 1]
        };

        match layer {
            Layer::Linear(l) => {
                let next = l.propagate_linear_with_engine_and_deadline(
                    &linear_bounds,
                    config.engine,
                    config.deadline,
                )?;
                if let Cow::Owned(next) = next {
                    linear_bounds = next;
                }
            }
            Layer::ReLU(r) => {
                if let Some(&relu_idx) = config.layer_to_relu_idx.get(&layer_idx) {
                    // Store intermediates before ReLU if requested
                    if config.store_intermediates {
                        a_at_relu.push(linear_bounds.lower_a().clone());
                        let (lower, upper) = pre_activation
                            .flatten_to_ix1(&format!("pre-ReLU layer {}", layer_idx))?;
                        pre_relu_bounds.push((lower, upper));
                    }

                    if let Some(alpha) = alpha_state.alpha(relu_idx) {
                        let alpha_upper = alpha_state.alpha_upper(relu_idx);
                        if config.track_gradients {
                            let (new_bounds, grad, grad_upper) = r.propagate_linear_with_alpha(
                                &linear_bounds,
                                pre_activation,
                                alpha,
                                alpha_upper,
                            )?;
                            gradients[relu_idx] = grad;
                            gradients_upper[relu_idx] = grad_upper;
                            linear_bounds = new_bounds;
                        } else {
                            linear_bounds = r.propagate_linear_with_alpha_bound_only(
                                &linear_bounds,
                                pre_activation,
                                alpha,
                                alpha_upper,
                            )?;
                        }
                    } else {
                        linear_bounds =
                            r.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                    }
                } else {
                    linear_bounds =
                        r.propagate_linear_with_bounds(&linear_bounds, pre_activation)?;
                }
            }
            // === All other layers: exhaustive trait dispatch (#3424) ===
            // Every variant listed — no catch-all. Compiler catches new variants.
            Layer::Conv1d(_)
            | Layer::Conv2d(_)
            | Layer::ConvTranspose1d(_)
            | Layer::ConvTranspose2d(_)
            | Layer::Transpose(_)
            | Layer::Tile(_)
            | Layer::Slice(_)
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
            | Layer::Squeeze(_)
            | Layer::Unsqueeze(_)
            | Layer::Pad(_)
            | Layer::Resize(_)
            | Layer::ScatterAdd(_)
            | Layer::IndexAdd(_)
            | Layer::AveragePool(_)
            | Layer::MaxPool2d(_)
            | Layer::RoPE(_)
            | Layer::NonZero(_)
            | Layer::SelfAttention(_)
            | Layer::ExpandLikeLastAxis(_) => {
                let (new_bounds, fallback) = propagate_generic_layer_backward(
                    layer,
                    &linear_bounds,
                    pre_activation,
                    layer_idx,
                    config.engine,
                    config.deadline,
                )?;
                if let Some(result) = fallback {
                    return Ok(result);
                }
                linear_bounds = new_bounds;
            }
        }

        // INVPROP augmentation
        if let Some(ref state) = alpha_state.invprop_state {
            let layer_name = format!("/layer.{}", layer_idx);
            if let Some(gammas) = state.layer_gammas(&layer_name) {
                if let Some((gammas_lower, gammas_upper)) = gammas
                    .active
                    .then(|| gammas.checked_bound_gammas())
                    .flatten()
                {
                    if config.best_of_oc {
                        bounds_without_oc = Some(linear_bounds.clone());
                    }

                    linear_bounds = augment_bounds_with_constraints(
                        &linear_bounds,
                        &state.constraints,
                        &gammas_lower.to_owned(),
                        &gammas_upper.to_owned(),
                    );
                }
            }
        }
        check_backward_deadline(config.deadline, "after layer dispatch")?;
    }

    check_backward_deadline(config.deadline, "before input augmentation")?;
    // Input-level INVPROP augmentation (#2928).
    // After the backward loop completes, apply constraint augmentation at the
    // network input level if configured. Mirrors graph path:
    // propagate_sequential.rs lines 478-504.
    // Note: no best_of_oc snapshot here — matches graph path, which only
    // snapshots bounds_without_oc inside the per-layer loop.
    if let Some(ref state) = alpha_state.invprop_state {
        if let Some(gammas) = state.layer_gammas(crate::NETWORK_INPUT) {
            if let Some((gammas_lower, gammas_upper)) = gammas
                .active
                .then(|| gammas.checked_bound_gammas())
                .flatten()
            {
                linear_bounds = augment_bounds_with_constraints(
                    &linear_bounds,
                    &state.constraints,
                    &gammas_lower.to_owned(),
                    &gammas_upper.to_owned(),
                );
            }
        }
    }

    check_backward_deadline(config.deadline, "before returning")?;
    Ok(BackwardPassResult::Success(Box::new(BackwardPassOutput {
        linear_bounds,
        gradients,
        gradients_upper,
        a_at_relu,
        pre_relu_bounds,
        bounds_without_oc,
    })))
}

/// Run a simple backward pass (no gradient tracking, no best-of-OC).
///
/// Handles the common setup: validate output_dim, build ReLU index maps,
/// construct config, and call `backward_pass_core`. Used by `single_pass_impl`
/// and `with_intermediates_impl`.
pub(super) fn run_simple_backward_pass(
    layers: &[Layer],
    input: &BoundedTensor,
    layer_bounds: &[BoundedTensor],
    alpha_state: &AlphaState,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    store_intermediates: bool,
    context: &str,
) -> Result<BackwardPassResult> {
    let output_dim = layer_bounds.last().map(|b| b.len()).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "Alpha-CROWN {} expected at least one intermediate bound",
            context,
        ))
    })?;

    let (relu_layer_indices, layer_to_relu_idx) = build_layer_to_relu_idx(layers);

    let bp_config = BackwardPassConfig {
        track_gradients: false,
        store_intermediates,
        best_of_oc: false,
        engine,
        deadline,
        layer_to_relu_idx: &layer_to_relu_idx,
        relu_layer_indices: &relu_layer_indices,
    };

    backward_pass_core(
        layers,
        input,
        layer_bounds,
        alpha_state,
        output_dim,
        &bp_config,
    )
}

#[cfg(test)]
mod deadline_tests {
    use super::*;
    use crate::layers::{FlattenLayer, LinearLayer};
    use ndarray::{arr1, arr2};
    use ny_test_utils::CountingGemmEngine;
    use std::time::Duration;

    fn scalar_input() -> BoundedTensor {
        BoundedTensor::new(arr1(&[-1.0f32]).into_dyn(), arr1(&[1.0f32]).into_dyn())
            .expect("valid scalar input")
    }

    #[test]
    fn expired_backward_refuses_linear_before_engine_launch() {
        let input = scalar_input();
        let layers = vec![Layer::Linear(
            LinearLayer::new(arr2(&[[1.0f32]]), None).expect("valid linear layer"),
        )];
        let layer_bounds = vec![input.clone()];
        let alpha_state =
            AlphaState::from_preactivation_bounds(&[], &[]).expect("empty alpha state");
        let engine = CountingGemmEngine::new();
        let relu_layer_indices = Vec::new();
        let layer_to_relu_idx = HashMap::new();
        let config = BackwardPassConfig {
            track_gradients: false,
            store_intermediates: false,
            best_of_oc: false,
            engine: Some(&engine),
            deadline: Some(
                Instant::now()
                    .checked_sub(Duration::from_millis(1))
                    .expect("one millisecond fits before the current instant"),
            ),
            layer_to_relu_idx: &layer_to_relu_idx,
            relu_layer_indices: &relu_layer_indices,
        };

        let error =
            match backward_pass_core(&layers, &input, &layer_bounds, &alpha_state, 1, &config) {
                Err(error) => error,
                Ok(_) => panic!("expired deadline must remain a structured error"),
            };

        assert!(error.is_deadline_exceeded(), "unexpected error: {error}");
        assert_eq!(
            engine.gemm_calls(),
            0,
            "expired backward must not launch the configured GEMM engine"
        );
    }

    #[test]
    fn expired_generic_dispatch_is_not_converted_to_fallback() {
        let input = scalar_input();
        let layer = Layer::Flatten(FlattenLayer::new(1));
        let error = match propagate_generic_layer_backward(
            &layer,
            &LinearBounds::identity(1),
            &input,
            0,
            None,
            Some(
                Instant::now()
                    .checked_sub(Duration::from_millis(1))
                    .expect("one millisecond fits before the current instant"),
            ),
        ) {
            Err(error) => error,
            Ok(_) => panic!("expired generic dispatch must not become Fallback"),
        };

        assert!(error.is_deadline_exceeded(), "unexpected error: {error}");
    }
}
