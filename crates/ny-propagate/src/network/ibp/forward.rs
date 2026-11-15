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
pub(crate) enum LeadingAxisMode {
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
pub(crate) fn try_lower_dense_chain(
    network: &Network,
    input_shape: &[usize],
) -> Option<Vec<GpuIbpLayer>> {
    lower_dense_chain_until(network, input_shape, None)
}

/// The same diagnostic lowering, with bounded polls while copying constants.
/// A missing supported lowering is distinct from an expired caller authority.
pub(crate) fn try_lower_dense_chain_until(
    network: &Network,
    input_shape: &[usize],
    deadline: Instant,
) -> Result<Option<Vec<GpuIbpLayer>>> {
    lower_checkpoint(Some(deadline))?;
    let layers = lower_dense_chain_until(network, input_shape, Some(deadline));
    lower_checkpoint(Some(deadline))?;
    Ok(layers)
}

fn lower_checkpoint(deadline: Option<Instant>) -> Result<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded("PGD dense plan lowering".into()));
    }
    Ok(())
}

fn copy_dense_constant_until(
    values: &[f32],
    deadline: Option<Instant>,
) -> Option<std::sync::Arc<[f32]>> {
    lower_checkpoint(deadline).ok()?;
    if deadline.is_none() {
        return Some(values.to_vec().into());
    }
    let mut copied = Vec::with_capacity(values.len());
    for chunk in values.chunks(4_096) {
        lower_checkpoint(deadline).ok()?;
        copied.extend_from_slice(chunk);
    }
    lower_checkpoint(deadline).ok()?;
    Some(copied.into())
}

fn lower_dense_chain_until(
    network: &Network,
    input_shape: &[usize],
    deadline: Option<Instant>,
) -> Option<Vec<GpuIbpLayer>> {
    lower_checkpoint(deadline).ok()?;
    let mut gpu_layers = Vec::with_capacity(network.layers.len());
    let mut current_shape = input_shape.to_vec();
    for layer in &network.layers {
        lower_checkpoint(deadline).ok()?;
        match layer {
            Layer::Linear(linear) => {
                let weight_slice = linear.weight.as_slice()?;
                let (out_features, in_features) = linear.weight.dim();
                let bias = match linear.bias.as_ref() {
                    Some(bias) => Some(copy_dense_constant_until(bias.as_slice()?, deadline)?),
                    None => None,
                };
                let last_dim = current_shape.last_mut()?;
                if *last_dim != in_features {
                    return None;
                }
                gpu_layers.push(GpuIbpLayer::Linear {
                    weight: copy_dense_constant_until(weight_slice, deadline)?,
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
                    Some(b) => Some(copy_dense_constant_until(b.as_slice()?, deadline)?),
                    None => None,
                };

                gpu_layers.push(GpuIbpLayer::Conv2d {
                    weight: copy_dense_constant_until(weight_slice, deadline)?,
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
/// Linear and causal softmax use their explicit diagnostic point evaluators.
/// Other layers still re-collapse their interval output after every operation;
/// that legacy diagnostic approximation does not promise deployed-kernel parity.
/// NON-soundness-critical: used only for sat-finding / witness proposals, never
/// to decide a Verified/unsat verdict.
pub(super) fn propagate_concrete_point(
    network: &Network,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    leading_axis_mode: LeadingAxisMode,
) -> Result<BoundedTensor> {
    propagate_concrete_point_until(network, input, engine, leading_axis_mode, None)
}

/// Diagnostic point evaluation with an optional earlier caller authority.
/// The published instance deadline remains a cap; neither authority creates a
/// fresh duration. This point route never certifies an interval or UNSAT.
pub(crate) fn propagate_concrete_point_until(
    network: &Network,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    leading_axis_mode: LeadingAxisMode,
    caller_deadline: Option<Instant>,
) -> Result<BoundedTensor> {
    if network.has_self_attention() {
        return Err(NyError::UnsupportedConfiguration(
            "SelfAttention requires a graph network; use GraphNetwork IBP or CROWN".to_string(),
        ));
    }

    // General path: collapse the input to a point up front (callers pass a
    // degenerate box, but be defensive: a point forward is only defined at the box
    // center).
    let deadline = match (caller_deadline, ny_core::instance_budget::deadline()) {
        (Some(caller), Some(instance)) => Some(caller.min(instance)),
        (caller, instance) => caller.or(instance),
    };
    let poll = || {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            Err(NyError::DeadlineExceeded(
                "sequential concrete point".into(),
            ))
        } else {
            Ok(())
        }
    };
    let center = |bounds: &BoundedTensor| {
        let values = bounds.center_with_poll(poll)?;
        BoundedTensor::concrete_with_poll(values, poll)
    };
    let mut current = center(input)?;

    // Whole-network IBP plans return enclosing intervals, including execution
    // allowances; their centers are not a concrete execution contract. Only
    // each named point kernel below may evaluate this diagnostic route.
    for (i, layer) in network.layers.iter().enumerate() {
        poll()?;
        let next = match layer {
            Layer::Linear(linear) => {
                linear.propagate_concrete_point_with_engine_and_deadline(&current, engine, deadline)
            }
            Layer::PointLinear(linear) => linear
                .propagate_concrete_point_with_receipt(&current, engine, deadline)
                .map(|(point, _)| point),
            Layer::CausalSoftmax(softmax) => {
                softmax.propagate_concrete_point_with_deadline(&current, deadline)
            }
            _ => propagate_layer_ibp_with_engine(layer, &current, engine, leading_axis_mode),
        }
        .map_err(|e| NyError::LayerError {
            layer_index: i,
            layer_type: layer.layer_type().to_string(),
            source: Box::new(e),
        })?;
        check_sequential_ibp_nan(&next, "Concrete point forward", i, layer.layer_type())?;
        // Re-collapse to the interval center so the next layer sees a point and the
        // seed width cannot be amplified by the rest of the (deep) network.
        current = center(&next)?;
    }
    poll()?;
    Ok(current)
}

#[cfg(test)]
#[path = "forward/point_tests.rs"]
mod point_tests;

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
    // pass on GPU with a single command submission.
    //
    // This path fires for both Plain and PreserveLeadingAxis modes.
    // `try_lower_dense_chain` computes shapes from the full (possibly batched)
    // input tensor. For networks with Flatten(0) and a prepended restart axis,
    // the lowerer produces a shape mismatch (batch dim folded into features)
    // and returns None, causing correct fallback to the per-layer loop where
    // PreserveLeadingAxis mode kicks in.
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

/// Per-layer IBP collector shared by [`collect_ibp_bounds`] and
/// [`collect_ibp_bounds_with_deadline`].
///
/// # Conv-family / AveragePool arms are CERTIFIED, not plain (#seq-plain-collector-sound)
///
/// This collector is the base of every sequential verdict path that is not
/// `collect_ibp_bounds_sound`: the input-split per-domain layer bounds
/// (`beta_crown/engine/input_split.rs`), the sequential root IBP fast path and
/// root layer bounds (`beta_crown/engine/bounds.rs`, `verify_phases/initial.rs`),
/// the CROWN-IBP base (`network/ibp/crown_ibp.rs`) and the CROWN-fast / batched
/// intermediate bounds (`core/sequential/crown/{fast,batched}.rs`). Each of those
/// reads ReLU stability (`l >= 0` / `u <= 0`) and verdict boxes straight off the
/// collected layer bounds. The PLAIN conv arms (`Conv2dLayer::propagate_ibp`,
/// `Conv1dLayer::propagate_ibp`, the ConvTranspose variants, `AveragePool`)
/// accumulate a K-term window sum in round-to-nearest f32 with no directed
/// rounding and no Higham term, so under cancellation the box can sit INSIDE the
/// true output (documented at `conv2d/bound.rs` `propagate_ibp_sound_with_engine`:
/// `W=[2^24,1,-2^24,4]` loses ~1.0). An inward per-layer box classifies a truly
/// unstable ReLU as stable, or shrinks a domain's output box below the threshold,
/// which is a false VERIFIED domain and hence a false `unsat`.
///
/// The repair is at the funnel rather than at the call sites: conv-family layers
/// go through their certified Higham forward and AveragePool through its
/// certified window-sum residual — exactly the dispatch `collect_ibp_bounds_sound`
/// and the graph lane (`graph_alpha/bounds/ibp.rs`) already use. The generic arm is
/// UNCHANGED (no `round_for_soundness`), so Linear-only networks — whose plain
/// `LinearLayer::propagate_ibp` is already Higham-certified — remain byte-identical
/// and no MLP verdict moves. Looser-or-equal boxes only: this can turn a proof into
/// a timeout, never the reverse. The `deadline` is forwarded to the pollable conv
/// kernels so finite authority is preserved; `None` keeps the abssum-certificate arm.
fn collect_ibp_bounds_core(
    network: &Network,
    input: &BoundedTensor,
    deadline: Option<Instant>,
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
        // One certified router for both the plain and the deadline-bearing
        // lanes (#crown-partial-sound-forward x #seq-plain-collector-sound):
        // conv family + AveragePool get their certified enclosures, Linear and
        // the exact-in-f32 arms stay byte-identical, and every remaining layer
        // is rounded outward rather than published verbatim.
        current = propagate_ibp_certified_with_deadline(layer, &current, deadline)
            .map_err(|e| NyError::LayerError {
                layer_index: i,
                layer_type: layer.layer_type().to_string(),
                source: Box::new(e),
            })?
            .into_tensor();
        check_sequential_ibp_nan(&current, "Sequential IBP collect", i, layer.layer_type())?;
        if i + 1 < n {
            bounds.push(current.clone());
        }
    }
    bounds.push(current); // Move the last layer's bounds, not clone

    Ok(bounds)
}

pub(super) fn collect_ibp_bounds(
    network: &Network,
    input: &BoundedTensor,
) -> Result<Vec<BoundedTensor>> {
    collect_ibp_bounds_core(network, input, None)
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

/// A per-layer interval box produced by [`propagate_ibp_certified`]: every
/// element is an outward-rounded enclosure of the layer's true output range
/// over the (enclosing) input box it was computed from.
///
/// #crown-partial-sound-forward — type-level provenance for the CROWN-IBP
/// partial walk. `crown_partial.rs` used to seed, scatter and merge through the
/// PLAIN `Layer::propagate_ibp` box (`crown_partial.rs` "output_bounds"), and
/// that box is published VERBATIM into the returned CROWN-IBP bound on every
/// row the sparse seed does not track (`sparse_merge.rs`
/// `merge_sparse_crown_with_ibp` / `scatter_sparse_crown_into_ibp`) and wins
/// the per-element intersection on tracked rows whenever it is tighter. The
/// plain conv-family arm is round-to-nearest f32 with no directed rounding and
/// no Higham term, documented as able to sit INWARD of the true supremum
/// (`conv2d/bound.rs` `propagate_ibp_sound_with_engine` doc, the
/// `W=[2^24,1,-2^24,4]` cancellation). Inward + published = a false stability
/// classification = a possible wrong VERIFIED. Those sites now accept only this
/// type, so a plain box cannot be handed to them without a compile error.
///
/// The ONLY constructor is [`propagate_ibp_certified`]: there is deliberately no
/// `from_tensor`, so a reviewer can grep this file for every way a value of this
/// type comes into existence. (The boxes those sites INTERSECT with — the
/// collector's incumbent and the CROWN concretization — are certified by their
/// own construction, see [`collect_ibp_bounds_with_deadline`] and
/// `LinearBounds::concretize_sound_with_deadline`; an intersection of two
/// enclosures encloses, and a disjoint pair unions.)
#[derive(Debug, Clone)]
pub(super) struct CertifiedIbpBox(BoundedTensor);

impl CertifiedIbpBox {
    /// Borrow the enclosing box.
    #[inline]
    pub(super) fn as_tensor(&self) -> &BoundedTensor {
        &self.0
    }

    /// Unwrap the enclosing box.
    #[inline]
    pub(super) fn into_tensor(self) -> BoundedTensor {
        self.0
    }
}

/// CERTIFIED per-layer interval forward for the sequential collector family.
///
/// The arm table is the one [`collect_ibp_bounds_sound`] uses, with two
/// deliberate deviations that keep Linear-only (MLP) networks BYTE-IDENTICAL
/// to the plain collector while the conv family gains its Higham enclosure:
///
/// * `Linear` takes the PLAIN arm: that arm is itself certified — the
///   abssum-Higham certificate `γ_{n+2}·S + 2u·|y|` is applied inside
///   `LinearLayer::propagate_ibp` (`linear/ibp.rs`, both the rank-1 f64 path
///   and the N-D/GEMM f32 path), and `propagate_ibp_sound` only stacks a
///   further `n+2` ULPs on top of it. Stacking would change every published
///   MLP bound by a few ULPs for no soundness gain.
/// * `ReLU`, `Flatten` and `Reshape` are EXACT in f32 (`max(0,x)` and data
///   movement round nothing), so no widening is applied; the sound collector's
///   blanket 1-ULP arm would again perturb MLP bounds for no gain.
///
/// Everything else follows the sound collector: the conv family (now including
/// `Conv3d`, which owns a certified arm) and `AveragePool` take their certified
/// Higham forwards; every remaining kind takes the plain arm plus the 1-ULP
/// `round_for_soundness_inplace` widening. The 1-ULP class is the residual the
/// second audit already names for the DAG collector's generic arm and is not
/// tightened here.
pub(super) fn propagate_ibp_certified(
    layer: &Layer,
    input: &BoundedTensor,
) -> Result<CertifiedIbpBox> {
    propagate_ibp_certified_with_deadline(layer, input, None)
}

/// The deadline-bearing form of [`propagate_ibp_certified`], and the SINGLE
/// certified layer router — [`propagate_ibp_certified`] and
/// [`collect_ibp_bounds_core`] both go through it, so the two cannot drift.
///
/// The conv family's certified kernels are pollable, so an expired deadline
/// inside a conv forward surfaces as an error (a timeout) rather than an opaque
/// kernel running past finite authority (#seq-plain-collector-sound). Every
/// other arm is not deadline-aware and behaves exactly as before.
///
/// Merge note (#crown-partial-sound-forward x #seq-plain-collector-sound): two
/// branches independently certified this walk, one with the deadline and a
/// narrower layer set, one with the full layer set and no deadline. Taking
/// either side alone would have dropped the other's coverage — specifically
/// `Conv3d` and the outward-rounded generic arm, which the deadline-bearing
/// version did not have. This is the union of the two, routed once.
pub(super) fn propagate_ibp_certified_with_deadline(
    layer: &Layer,
    input: &BoundedTensor,
    deadline: Option<Instant>,
) -> Result<CertifiedIbpBox> {
    let out = match layer {
        // Self-certified plain arm (see doc above): byte-identical to the
        // plain collector on MLPs.
        Layer::Linear(linear) => linear.propagate_ibp(input)?,
        // Conv family: certified Higham error term, outward-rounded.
        Layer::Conv1d(conv) => {
            conv.propagate_ibp_sound_with_engine_and_deadline(input, None, deadline)?
        }
        Layer::Conv2d(conv) => {
            conv.propagate_ibp_sound_with_engine_and_deadline(input, None, deadline)?
        }
        Layer::Conv3d(conv) => {
            conv.propagate_ibp_sound_with_engine_and_deadline(input, None, deadline)?
        }
        Layer::ConvTranspose1d(conv) => {
            conv.propagate_ibp_sound_with_engine_and_deadline(input, None, deadline)?
        }
        Layer::ConvTranspose2d(conv) => {
            conv.propagate_ibp_sound_with_engine_and_deadline(input, None, deadline)?
        }
        // AveragePool: certified γ⁶⁴ window-sum residual (#avgpool-1ulp-arm).
        Layer::AveragePool(pool) => pool.propagate_ibp_sound(input)?,
        // Exact in f32: nothing to widen.
        Layer::ReLU(_) | Layer::Flatten(_) | Layer::Reshape(_) => layer.propagate_ibp(input)?,
        _ => {
            let mut result = layer.propagate_ibp(input)?;
            result.round_for_soundness_inplace();
            result
        }
    };
    Ok(CertifiedIbpBox(out))
}

/// Collect the per-layer incumbent for the deadline-bearing (= verdict-bearing)
/// sequential lanes: CROWN-IBP collection (`crown_ibp.rs` `collect_core`, the
/// `precomputed_ibp: None` arm) and the sequential β-CROWN engine's root and
/// per-domain boxes (`beta_crown/engine/bounds.rs`, `input_split.rs`,
/// `verify_phases/initial.rs`).
///
/// #crown-partial-sound-forward: this used to be the PLAIN [`collect_ibp_bounds`].
/// The incumbent it produces is published VERBATIM by the collector on the
/// ReLU-stable skip and on every `ForwardFallback` arm, is the operand every
/// CROWN box is intersected with (tighter wins, so an inward plain incumbent
/// survives a certified CROWN box), and `bounds.rs` returns `Verified` straight
/// off its last entry with no CROWN at all. It is therefore built with
/// [`propagate_ibp_certified`] — Linear-only networks stay byte-identical (see
/// that function), conv-family networks gain the Higham enclosure. The plain
/// [`collect_ibp_bounds`] remains for the non-deadline public API.
///
/// IBP must run to completion to produce bounds for all layers — the CROWN
/// backward pass needs pre-activation bounds at every layer. Deadline checking
/// happens AFTER IBP in the CROWN-IBP collection function, which skips the
/// expensive per-layer CROWN partial passes when the deadline has been consumed
/// by IBP.
pub(super) fn collect_ibp_bounds_with_deadline(
    network: &Network,
    input: &BoundedTensor,
    deadline: Option<Instant>,
) -> Result<Vec<BoundedTensor>> {
    // IBP runs to completion to produce bounds for all layers — the CROWN
    // backward pass needs pre-activation bounds at every layer. The CROWN-IBP
    // collection function skips the expensive per-layer CROWN partial passes when
    // the deadline has been consumed by IBP. The deadline IS forwarded to
    // the certified conv-family kernels (#seq-plain-collector-sound): those are
    // pollable, so an expired deadline inside a conv forward surfaces as an
    // error (a timeout) instead of an opaque kernel running past finite authority.
    // Linear and the generic arm are not deadline-aware and behave exactly as before.
    collect_ibp_bounds_core(network, input, deadline)
}
