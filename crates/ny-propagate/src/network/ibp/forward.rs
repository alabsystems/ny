// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Pure IBP forward propagation logic for sequential networks.

use super::helpers::check_sequential_ibp_nan;
use crate::contiguous_flat_slice;
use crate::layers::{BoundPropagation, Layer};
use crate::network::core::Network;
use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, GemmEngine, GpuIbpLayer, NyError, Result};
use ny_tensor::BoundedTensor;
use std::time::Instant;
use tracing::{debug, instrument};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum LeadingAxisMode {
    Plain,
    PreserveLeadingAxis,
}

fn propagate_layer_ibp_with_engine(
    layer: &Layer,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    leading_axis_mode: LeadingAxisMode,
) -> Result<BoundedTensor> {
    match (layer, leading_axis_mode) {
        (Layer::Linear(linear), _) => linear.propagate_ibp_with_engine(input, engine),
        (Layer::Conv1d(conv), _) => conv.propagate_ibp_with_engine(input, engine),
        (Layer::Conv2d(conv), _) => conv.propagate_ibp_with_engine(input, engine),
        (Layer::ConvTranspose1d(conv), _) => conv.propagate_ibp_with_engine(input, engine),
        (Layer::ConvTranspose2d(conv), _) => conv.propagate_ibp_with_engine(input, engine),
        (Layer::Flatten(layer), LeadingAxisMode::PreserveLeadingAxis) => {
            layer.propagate_ibp_preserve_leading_axis(input)
        }
        (Layer::Reshape(layer), LeadingAxisMode::PreserveLeadingAxis) => {
            layer.propagate_ibp_preserve_leading_axis(input)
        }
        _ => layer.propagate_ibp(input),
    }
}

#[instrument(skip(network, input), fields(num_layers = network.layers.len(), input_shape = ?input.shape()))]
pub(super) fn propagate_ibp(network: &Network, input: &BoundedTensor) -> Result<BoundedTensor> {
    if network.has_self_attention() {
        return Err(NyError::UnsupportedConfiguration(
            "SelfAttention requires a graph network; use GraphNetwork IBP or CROWN".to_string(),
        ));
    }
    let mut current = input.clone();
    for (i, layer) in network.layers.iter().enumerate() {
        debug!("IBP propagating through layer {}", i);
        current = layer
            .propagate_ibp(&current)
            .map_err(|e| NyError::LayerError {
                layer_index: i,
                layer_type: layer.layer_type().to_string(),
                source: Box::new(e),
            })?;
        check_sequential_ibp_nan(&current, "Sequential IBP", i, layer.layer_type())?;
    }
    Ok(current)
}

/// Try to lower a sequential network to the dense-chain GPU IBP layer subset.
///
/// Returns `Some(layers)` only if every layer is `Linear`, `ReLU`, `Flatten`,
/// or `Reshape`. Returns `None` for any unsupported layer (Conv, Sign, MatMul,
/// etc.) or if shape computation fails, in which case the caller falls back
/// to the per-layer loop.
///
/// `input_shape` is the shape of the input tensor (from `BoundedTensor::shape()`).
///
/// Reference: designs/2026-03-18-issue-4081-gpu-ibp-forward-gap2-addendum.md §1
pub(crate) fn try_lower_dense_chain(
    network: &Network,
    input_shape: &[usize],
) -> Option<Vec<GpuIbpLayer>> {
    let mut gpu_layers = Vec::with_capacity(network.layers.len());
    let mut current_shape = input_shape.to_vec();
    for layer in &network.layers {
        match layer {
            Layer::Linear(linear) => {
                let weight_slice = linear.weight.as_slice()?;
                let (out_features, in_features) = linear.weight.dim();
                let bias = match linear.bias.as_ref() {
                    Some(bias) => Some(bias.as_slice()?.to_vec().into()),
                    None => None,
                };
                let last_dim = current_shape.last_mut()?;
                if *last_dim != in_features {
                    return None;
                }
                gpu_layers.push(GpuIbpLayer::Linear {
                    weight: weight_slice.to_vec().into(),
                    bias,
                    out_features,
                    in_features,
                });
                // Linear: [..., in_features] -> [..., out_features]
                *last_dim = out_features;
            }
            Layer::Conv2d(conv) => {
                // Only groups=1 Conv2d is supported on the resident GPU path.
                if conv.groups != 1 {
                    return None;
                }
                let kernel_shape = conv.kernel.shape();
                if kernel_shape.len() != 4 {
                    return None;
                }
                let out_channels = kernel_shape[0];
                let in_channels = kernel_shape[1];
                let kernel_h = kernel_shape[2];
                let kernel_w = kernel_shape[3];
                let (stride_h, stride_w) = conv.stride;
                let (pad_h, pad_w) = conv.padding;

                // current_shape must be [..., C, H, W]
                let ndim = current_shape.len();
                if ndim < 3 {
                    return None;
                }
                let input_h = current_shape[ndim - 2];
                let input_w = current_shape[ndim - 1];
                let input_c = current_shape[ndim - 3];
                if input_c != in_channels * conv.groups {
                    return None;
                }

                let weight_slice = conv.kernel.as_slice()?;
                let bias = match conv.bias.as_ref() {
                    Some(b) => Some(b.as_slice()?.to_vec().into()),
                    None => None,
                };

                gpu_layers.push(GpuIbpLayer::Conv2d {
                    weight: weight_slice.to_vec().into(),
                    bias,
                    out_channels,
                    in_channels,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    groups: conv.groups,
                    input_h,
                    input_w,
                });

                // Update shape: [..., out_channels, out_h, out_w]
                let out_h = (input_h + 2 * pad_h).checked_sub(kernel_h)? / stride_h + 1;
                let out_w = (input_w + 2 * pad_w).checked_sub(kernel_w)? / stride_w + 1;
                current_shape[ndim - 3] = out_channels;
                current_shape[ndim - 2] = out_h;
                current_shape[ndim - 1] = out_w;
            }
            Layer::ReLU(_) => {
                // ReLU is elementwise and shape-preserving; element count from
                // current shape.
                let num_elements = checked_shape_product(&current_shape)?;
                gpu_layers.push(GpuIbpLayer::ReLU { num_elements });
            }
            Layer::Flatten(f) => {
                let output_shape = f.compute_output_shape(&current_shape).ok()?;
                gpu_layers.push(GpuIbpLayer::View {
                    output_shape: output_shape.clone().into(),
                });
                current_shape = output_shape;
            }
            Layer::Reshape(r) => {
                let output_shape = r.compute_output_shape(&current_shape).ok()?;
                gpu_layers.push(GpuIbpLayer::View {
                    output_shape: output_shape.clone().into(),
                });
                current_shape = output_shape;
            }
            // Unsupported layer: fall back to per-layer loop.
            _ => return None,
        }
    }
    Some(gpu_layers)
}

pub(super) fn propagate_ibp_with_engine(
    network: &Network,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
) -> Result<BoundedTensor> {
    propagate_ibp_with_engine_mode(network, input, engine, LeadingAxisMode::Plain)
}

pub(super) fn propagate_ibp_with_engine_preserve_leading_axis(
    network: &Network,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
) -> Result<BoundedTensor> {
    propagate_ibp_with_engine_mode(network, input, engine, LeadingAxisMode::PreserveLeadingAxis)
}

/// True CONCRETE (point) forward evaluation at `input`.
///
/// `propagate_ibp_with_engine` propagates a *box* through the network; for a
/// degenerate (point) input the per-layer IBP relaxations still produce a NON-zero
/// width because several layers (notably BatchNorm) deliberately apply outward
/// widening + directed `next_down`/`next_up` rounding for SOUNDNESS, and the deep
/// conv stack then amplifies that seed interval EXPONENTIALLY (IBP wrapping). The
/// resulting box can be very wide even at a single point (observed: 0.04–4.1 wide
/// on cgan_2023 generators), so taking `output.lower()` as "the concrete forward
/// value" — as the PGD evaluator historically did — returns a value far below the
/// true network output, fabricating false counterexamples that the trusted ORT
/// oracle then rejects (cgan_2023 unknown-downgrade bug).
///
/// This routine instead RE-COLLAPSES the running tensor to its interval center
/// after every layer, so each layer sees a degenerate point and contributes only
/// its own ~ULP of rounding (which does NOT amplify across layers). The result is
/// a faithful f32 point forward that matches ONNX Runtime to ~1e-6 on the
/// cgan_2023 generators. NON-soundness-critical: used only for sat-finding /
/// witness evaluation, never to decide a Verified/unsat verdict.
pub(super) fn propagate_concrete_point(
    network: &Network,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    leading_axis_mode: LeadingAxisMode,
) -> Result<BoundedTensor> {
    if network.has_self_attention() {
        return Err(NyError::UnsupportedConfiguration(
            "SelfAttention requires a graph network; use GraphNetwork IBP or CROWN".to_string(),
        ));
    }

    // Dense-chain fast path: a pure Linear/ReLU/Flatten/Reshape chain has NO
    // soundness widening (no BatchNorm, no directed-rounding conv), so a point input
    // stays degenerate through the whole network and there is nothing to amplify.
    // Delegate to the regular box forward — which preserves the GPU-resident /
    // cached-plan fast paths (#4081/#4268) — and take the (degenerate) center.
    // `try_lower_dense_chain` returns Some ONLY for exactly that widening-free set.
    if try_lower_dense_chain(network, input.shape()).is_some() {
        let out = propagate_ibp_with_engine_mode(network, input, engine, leading_axis_mode)?;
        return BoundedTensor::concrete(out.center());
    }

    // General path: collapse the input to a point up front (callers pass a
    // degenerate box, but be defensive: a point forward is only defined at the box
    // center).
    let mut current = BoundedTensor::concrete(input.center())?;
    for (i, layer) in network.layers.iter().enumerate() {
        let next = propagate_layer_ibp_with_engine(layer, &current, engine, leading_axis_mode)
            .map_err(|e| NyError::LayerError {
                layer_index: i,
                layer_type: layer.layer_type().to_string(),
                source: Box::new(e),
            })?;
        check_sequential_ibp_nan(&next, "Concrete point forward", i, layer.layer_type())?;
        // Re-collapse to the interval center so the next layer sees a point and the
        // seed width cannot be amplified by the rest of the (deep) network.
        current = BoundedTensor::concrete(next.center())?;
    }
    Ok(current)
}

pub(super) fn propagate_concrete_point_plain(
    network: &Network,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
) -> Result<BoundedTensor> {
    propagate_concrete_point(network, input, engine, LeadingAxisMode::Plain)
}

pub(super) fn propagate_concrete_point_preserve_leading_axis(
    network: &Network,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
) -> Result<BoundedTensor> {
    propagate_concrete_point(network, input, engine, LeadingAxisMode::PreserveLeadingAxis)
}

fn propagate_ibp_with_engine_mode(
    network: &Network,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    leading_axis_mode: LeadingAxisMode,
) -> Result<BoundedTensor> {
    if network.has_self_attention() {
        return Err(NyError::UnsupportedConfiguration(
            "SelfAttention requires a graph network; use GraphNetwork IBP or CROWN".to_string(),
        ));
    }

    // Resident fast path: if the engine supports GPU IBP forward and the network
    // is a pure dense chain (Linear/ReLU/Flatten/Reshape), run the entire forward
    // pass on GPU with a single command submission (#4081 Gap 2).
    //
    // This path fires for both Plain and PreserveLeadingAxis modes.
    // `try_lower_dense_chain` computes shapes from the full (possibly batched)
    // input tensor. For networks with Flatten(0) and a prepended restart axis,
    // the lowerer produces a shape mismatch (batch dim folded into features)
    // and returns None, causing correct fallback to the per-layer loop where
    // PreserveLeadingAxis mode kicks in (#4345).
    if let Some(gpu) = engine.and_then(|e| e.as_gpu_ibp_forward()) {
        if let Some(gpu_layers) = try_lower_dense_chain(network, input.shape()) {
            let input_lower = contiguous_flat_slice(input.lower());
            let input_upper = contiguous_flat_slice(input.upper());

            // A wgpu error (e.g. a validation error on a degenerate dispatch) is
            // returned as Err by the GPU engine — it must NOT abort. Fall back to
            // the per-layer loop below, which produces the sound CPU result.
            // (#live wgpu validation panic / CPU fallback soundness.)
            match gpu.ibp_forward_gpu(&gpu_layers, &input_lower, &input_upper, input.shape()) {
                Ok(result) => {
                    let lower =
                        ArrayD::from_shape_vec(IxDyn(&result.output_shape), result.lower_bounds)
                            .map_err(|e| {
                                NyError::InternalError(format!(
                                    "GPU IBP forward: output shape mismatch: {e}"
                                ))
                            })?;
                    let upper =
                        ArrayD::from_shape_vec(IxDyn(&result.output_shape), result.upper_bounds)
                            .map_err(|e| {
                                NyError::InternalError(format!(
                                    "GPU IBP forward: output shape mismatch: {e}"
                                ))
                            })?;

                    debug!(
                        "GPU IBP resident forward completed for {} layers",
                        gpu_layers.len()
                    );
                    return BoundedTensor::new(lower, upper);
                }
                Err(e) => {
                    debug!(
                        "GPU IBP resident forward failed ({e}), falling back to per-layer CPU/engine loop"
                    );
                }
            }
        }
    }

    // Fallback: per-layer loop with optional per-layer engine acceleration.
    let mut current = input.clone();
    for (i, layer) in network.layers.iter().enumerate() {
        debug!(
            "IBP propagating through layer {} with engine={} mode={:?}",
            i,
            engine.is_some(),
            leading_axis_mode,
        );
        current = propagate_layer_ibp_with_engine(layer, &current, engine, leading_axis_mode)
            .map_err(|e| NyError::LayerError {
                layer_index: i,
                layer_type: layer.layer_type().to_string(),
                source: Box::new(e),
            })?;
        check_sequential_ibp_nan(&current, "Sequential IBP", i, layer.layer_type())?;
    }
    Ok(current)
}

pub(super) fn propagate_ibp_sound(
    network: &Network,
    input: &BoundedTensor,
) -> Result<BoundedTensor> {
    // No engine: default to the proven-sound CPU loop (behavior-preserving).
    propagate_ibp_sound_with_engine(network, input, None)
}

/// SOUND (verdict-legal) IBP forward with an optional GPU engine
/// (`docs/SOUND_GPU_IBP_PLAN.md` §6.3, T1.1).
///
/// When the soundness gate is engaged AND `engine` advertises a sound GPU IBP
/// forward (`provides_sound_gpu_ibp`), a SEQUENTIAL dense chain is dispatched onto
/// the certified GPU sound path (`ibp_forward_gpu_sound`), whose interval is a
/// SUPERSET of both the true range and the CPU `propagate_ibp_sound` bound. Any
/// failure — an unsupported layer in the chain, a shape it cannot lower, or a wgpu
/// error — falls through to the proven-sound per-layer CPU loop, so a verdict is
/// never decided by a failed GPU op.
///
/// The GPU route is restricted to `try_lower_dense_chain` (sequential) networks;
/// graph/DAG verdict paths keep the CPU loop until T1.0 forwards the sound flag
/// through the graph IBP accessor. With `engine == None` (the default entry) the
/// gate route yields `None`, so this is exactly the CPU loop.
pub(super) fn propagate_ibp_sound_with_engine(
    network: &Network,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
) -> Result<BoundedTensor> {
    if network.has_self_attention() {
        return Err(NyError::UnsupportedConfiguration(
            "SelfAttention requires a graph network; use GraphNetwork IBP or CROWN".to_string(),
        ));
    }

    // Sound GPU dense-chain fast path (gate-engaged verdicts only).
    if let Some((gpu, use_sound)) = crate::sound_gpu_gate::gpu_ibp_forward_route(engine) {
        if use_sound {
            if let Some(gpu_layers) = try_lower_dense_chain(network, input.shape()) {
                let input_lower = contiguous_flat_slice(input.lower());
                let input_upper = contiguous_flat_slice(input.upper());
                match gpu.ibp_forward_gpu_sound(
                    &gpu_layers,
                    &input_lower,
                    &input_upper,
                    input.shape(),
                ) {
                    Ok(result) => {
                        let lower = ArrayD::from_shape_vec(
                            IxDyn(&result.output_shape),
                            result.lower_bounds,
                        )
                        .map_err(|e| {
                            NyError::InternalError(format!(
                                "sound GPU IBP forward: output shape mismatch: {e}"
                            ))
                        })?;
                        let upper = ArrayD::from_shape_vec(
                            IxDyn(&result.output_shape),
                            result.upper_bounds,
                        )
                        .map_err(|e| {
                            NyError::InternalError(format!(
                                "sound GPU IBP forward: output shape mismatch: {e}"
                            ))
                        })?;
                        debug!(
                            "sound GPU IBP resident forward completed for {} layers",
                            gpu_layers.len()
                        );
                        return BoundedTensor::new(lower, upper);
                    }
                    Err(e) => {
                        debug!(
                            "sound GPU IBP forward failed ({e}); falling back to CPU sound loop"
                        );
                    }
                }
            }
        }
    }

    // Fallback: proven-sound per-layer CPU loop.
    let mut current = input.clone();
    for (i, layer) in network.layers.iter().enumerate() {
        debug!("IBP (sound) propagating through layer {}", i);
        current = match layer {
            // Linear layers: n-ULP rounding proportional to dot product size
            // (in_features + 2 ULPs), covering accumulated rounding from matmul.
            Layer::Linear(linear) => linear.propagate_ibp_sound(&current),
            // Conv family: the f32 window-sum needs the certified Higham error, not the
            // 1-ULP rounding below (unsound under cancellation). #vnncomp-aw-soundness.
            Layer::Conv1d(conv) => conv.propagate_ibp_sound_with_engine(&current, None),
            Layer::Conv2d(conv) => conv.propagate_ibp_sound_with_engine(&current, None),
            Layer::ConvTranspose1d(conv) => conv.propagate_ibp_sound_with_engine(&current, None),
            Layer::ConvTranspose2d(conv) => conv.propagate_ibp_sound_with_engine(&current, None),
            // AveragePool: certified γ⁶⁴_{k+1}·S/d window-sum residual — the plain
            // forward's outward 1-ULP store covers only the f64→f32 cast, not the
            // f64 accumulation residual (unsound under ≥2^29 cancellation). See
            // `AveragePoolLayer::propagate_ibp_sound` (#avgpool-1ulp-arm).
            Layer::AveragePool(pool) => pool.propagate_ibp_sound(&current),
            // All other layers: standard propagation + 1-ULP rounding. ASSUMPTION
            // arm, not a certificate: valid for f32-exact ops, self-outward-rounding
            // layers (BatchNorm) and faithfully-rounded pointwise libm calls;
            // accumulating ops still on this arm (Softmax denominator, LayerNorm/
            // RMSNorm statistics, reductions) are NOT certified
            // (#sound-ibp-generic-arm — see `Network::propagate_ibp_sound` docs).
            _ => {
                let mut result = layer.propagate_ibp(&current)?;
                result.round_for_soundness_inplace();
                Ok(result)
            }
        }
        .map_err(|e| NyError::LayerError {
            layer_index: i,
            layer_type: layer.layer_type().to_string(),
            source: Box::new(e),
        })?;
        check_sequential_ibp_nan(&current, "Sequential IBP (sound)", i, layer.layer_type())?;
    }
    Ok(current)
}

pub(super) fn collect_ibp_bounds(
    network: &Network,
    input: &BoundedTensor,
) -> Result<Vec<BoundedTensor>> {
    let n = network.layers.len();
    if n == 0 {
        return Ok(vec![]);
    }
    if network.has_self_attention() {
        return Err(NyError::UnsupportedConfiguration(
            "SelfAttention requires a graph network; use GraphNetwork IBP or CROWN".to_string(),
        ));
    }
    let mut bounds = Vec::with_capacity(n);
    let mut current = input.clone();

    // Process all but last layer with cloning
    for (i, layer) in network.layers[..n - 1].iter().enumerate() {
        current = layer
            .propagate_ibp(&current)
            .map_err(|e| NyError::LayerError {
                layer_index: i,
                layer_type: layer.layer_type().to_string(),
                source: Box::new(e),
            })?;
        check_sequential_ibp_nan(&current, "Sequential IBP collect", i, layer.layer_type())?;
        bounds.push(current.clone());
    }

    // Process last layer without cloning (move ownership)
    let last_layer = &network.layers[n - 1];
    current = last_layer
        .propagate_ibp(&current)
        .map_err(|e| NyError::LayerError {
            layer_index: n - 1,
            layer_type: last_layer.layer_type().to_string(),
            source: Box::new(e),
        })?;
    check_sequential_ibp_nan(
        &current,
        "Sequential IBP collect",
        n - 1,
        last_layer.layer_type(),
    )?;
    bounds.push(current); // Move, not clone

    Ok(bounds)
}

pub(super) fn collect_ibp_bounds_sound(
    network: &Network,
    input: &BoundedTensor,
) -> Result<Vec<BoundedTensor>> {
    let n = network.layers.len();
    if n == 0 {
        return Ok(vec![]);
    }
    if network.has_self_attention() {
        return Err(NyError::UnsupportedConfiguration(
            "SelfAttention requires a graph network; use GraphNetwork IBP or CROWN".to_string(),
        ));
    }
    let mut bounds = Vec::with_capacity(n);
    let mut current = input.clone();

    for (i, layer) in network.layers.iter().enumerate() {
        current = match layer {
            Layer::Linear(linear) => linear.propagate_ibp_sound(&current),
            // Conv family: certified Higham error (see propagate_ibp_sound (sequential)).
            Layer::Conv1d(conv) => conv.propagate_ibp_sound_with_engine(&current, None),
            Layer::Conv2d(conv) => conv.propagate_ibp_sound_with_engine(&current, None),
            Layer::ConvTranspose1d(conv) => conv.propagate_ibp_sound_with_engine(&current, None),
            Layer::ConvTranspose2d(conv) => conv.propagate_ibp_sound_with_engine(&current, None),
            // AveragePool: certified γ⁶⁴ window-sum residual (see
            // propagate_ibp_sound_with_engine above; #avgpool-1ulp-arm).
            Layer::AveragePool(pool) => pool.propagate_ibp_sound(&current),
            _ => {
                let mut result = layer.propagate_ibp(&current)?;
                result.round_for_soundness_inplace();
                Ok(result)
            }
        }
        .map_err(|e| NyError::LayerError {
            layer_index: i,
            layer_type: layer.layer_type().to_string(),
            source: Box::new(e),
        })?;
        check_sequential_ibp_nan(
            &current,
            "Sequential IBP collect (sound)",
            i,
            layer.layer_type(),
        )?;
        bounds.push(current.clone());
    }

    Ok(bounds)
}

pub(super) fn collect_ibp_bounds_with_deadline(
    network: &Network,
    input: &BoundedTensor,
    _deadline: Option<Instant>,
) -> Result<Vec<BoundedTensor>> {
    // IBP must run to completion to produce bounds for all layers —
    // the CROWN backward pass needs pre-activation bounds at every layer.
    // Deadline checking happens AFTER IBP in the CROWN-IBP collection
    // function, which skips the expensive per-layer CROWN partial passes
    // when the deadline has been consumed by IBP (#3397).
    collect_ibp_bounds(network, input)
}
