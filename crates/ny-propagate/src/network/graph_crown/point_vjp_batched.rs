// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched exact point-VJP plan for the PGD attack (#batched-vjp).
//!
//! # What this provides
//!
//! The sequential exact-gradient PGD step ([`GraphNetwork::attack_point_gradient`])
//! costs ~140 ms per restart per step on a deep conv chain, so a 121 s attack
//! budget covers ~1 deep restart while the alpha-beta-CROWN reference runs 250.
//! This module builds the two host-side ingredients that let the GPU compute the
//! exact gradients of K restarts in ONE wide resident pass
//! (`GpuCrownBackward::crown_point_vjp_batched`):
//!
//! 1. [`GraphNetwork::build_point_vjp_batch_plan`] — a backward-order (output→
//!    input) [`GpuCrownLayer`] template of the network with the positions of its
//!    per-restart ReLU MASK slots. Built once per attack; the `Linear`/`Conv2d`
//!    weights are `Arc`-shared across every restart and step.
//! 2. [`point_vjp_forward_masks`] — a batched CPU forward over the SAME template
//!    that captures each restart point's ReLU masks (`pre_act > 0`) in exactly the
//!    fold order the backward consumes them, plus the network outputs (for the
//!    joint-margin rows / counterexample screen).
//!
//! Running the forward on the extracted template (rather than the graph) makes the
//! mask ↔ backward-slot correspondence hold BY CONSTRUCTION — the same layer list,
//! the same flat buffers, the same fold order — which is the fragile part of any
//! batched VJP (a mask/layout mismatch silently yields wrong gradients).
//!
//! # Supported fragment
//!
//! A PURE UNARY CHAIN of `{Linear, Conv1d, Conv2d, ReLU, Flatten, Reshape,
//! AddConstant, SubConstant, MulConstant, DivConstant}` from the network output to
//! the network input (Flatten/Reshape fold away — flat-buffer no-ops). Anything
//! else (residual `Add`, pooling, S-shaped activations, …) returns `None` and the
//! caller keeps the sequential exact-gradient loop. ATTACK-ONLY: gradients steer
//! PGD; every counterexample is concretely re-validated elsewhere, so nothing here
//! can affect a verdict.

use ndarray::{Array2, ArrayView1, ArrayView2};
use ny_core::{GpuCrownLayer, NyError, Result};
use ny_tensor::BoundedTensor;
use rayon::prelude::*;

use crate::layers::Layer;
use crate::network::core::{try_extract_single_gpu_layer, GraphNetwork, NETWORK_INPUT};

/// One forward-order op over the flat buffer, referencing the backward template.
#[derive(Clone, Copy, Debug)]
enum VjpForwardOp {
    /// `layers_backward[idx]` is a `Linear`.
    Linear { idx: usize },
    /// `layers_backward[idx]` is a `Conv2d` (Conv1d maps here with `kernel_h == 1`).
    Conv2d { idx: usize },
    /// `layers_backward[idx]` is a STATIC affine `Activation` (constant arithmetic):
    /// `y = lower_slope ⊙ x + lower_intercept` (lower == upper by construction).
    Affine { idx: usize },
    /// A per-restart ReLU mask slot: capture `mask[slot] = (x > 0)`, apply
    /// `y = max(x, 0)`. `slot` indexes `mask_positions` (backward/fold order).
    ReluMask { slot: usize, num_neurons: usize },
}

/// Shared batched point-VJP plan: the backward-order GPU layer template, its
/// per-restart ReLU mask slots, and the forward op list for CPU mask capture.
pub struct PointVjpBatchPlan {
    /// Backward-order (output→input) layer template. `Linear`/`Conv2d` weights are
    /// `Arc`-shared; `Activation` entries are either per-restart ReLU MASK slots
    /// (listed in `mask_positions`) or static affine constant-arithmetic ops.
    pub layers_backward: Vec<GpuCrownLayer>,
    /// Indices into `layers_backward` of the ReLU mask slots, in backward (fold)
    /// order — the order `masks[k]` must use.
    pub mask_positions: Vec<usize>,
    /// The ReLU node names aligned with `mask_positions` (diagnostics/tests).
    pub relu_nodes_backward: Vec<String>,
    /// Flattened network input dimension.
    pub input_dim: usize,
    /// Flattened network output dimension.
    pub output_dim: usize,
    /// Forward-order ops (input→output) over the flat buffer for mask capture.
    ops_forward: Vec<VjpForwardOp>,
}

impl GraphNetwork {
    /// Build the batched point-VJP plan for this graph, or `None` when the graph
    /// is outside the supported pure-chain fragment (caller falls back to the
    /// sequential exact gradient). Built ONCE per attack; only the per-restart
    /// masks change between steps.
    ///
    /// `input` is the attack's input box — used only for SHAPES (one cheap
    /// concrete forward at the box center resolves every node's shape for the
    /// Conv2d/constant extraction; the ReLU slots' template values are never
    /// used, they are replaced per restart by the masks).
    pub fn build_point_vjp_batch_plan(&self, input: &BoundedTensor) -> Option<PointVjpBatchPlan> {
        if self.nodes.is_empty() {
            return None;
        }
        let center = BoundedTensor::concrete(input.center()).ok()?;
        let node_bounds = self.collect_node_bounds(&center).ok()?;
        let output_name = self.output_name().to_string();
        let output_dim = node_bounds.get(&output_name)?.len();
        let input_dim = input.len();
        if output_dim == 0 || input_dim == 0 {
            return None;
        }

        let mut layers: Vec<GpuCrownLayer> = Vec::new();
        let mut mask_positions: Vec<usize> = Vec::new();
        let mut relu_nodes: Vec<String> = Vec::new();
        let mut current = output_name;
        let max_steps = self.nodes.len() + 1;
        let mut steps = 0usize;
        while current != NETWORK_INPUT {
            steps += 1;
            if steps > max_steps {
                return None;
            }
            let node = self.nodes.get(&current)?;
            // Pure unary chain only: any fan-in (residual Add, …) refuses here.
            let input_name = node.require_unary_input().ok()?.to_string();
            let pre: &BoundedTensor = if input_name == NETWORK_INPUT {
                &center
            } else {
                node_bounds.get(&input_name)?
            };
            match &node.layer {
                Layer::ReLU(_) => {
                    // Record the slot BEFORE extraction (ReLU pushes exactly one
                    // Activation); the template relaxation values are placeholders.
                    mask_positions.push(layers.len());
                    relu_nodes.push(current.clone());
                    try_extract_single_gpu_layer(&node.layer, pre, &mut layers)?;
                }
                Layer::Linear(_)
                | Layer::Conv1d(_)
                | Layer::Conv2d(_)
                | Layer::Flatten(_)
                | Layer::Reshape(_)
                | Layer::AddConstant(_)
                | Layer::SubConstant(_)
                | Layer::MulConstant(_)
                | Layer::DivConstant(_) => {
                    try_extract_single_gpu_layer(&node.layer, pre, &mut layers)?;
                }
                // Outside the exact-mask chain fragment → sequential fallback.
                _ => return None,
            }
            current = input_name;
        }
        if layers.is_empty() {
            return None;
        }

        // Forward op list: template reversed (input→output), mask slots resolved.
        let slot_of = |idx: usize| mask_positions.iter().position(|&p| p == idx);
        let mut ops_forward = Vec::with_capacity(layers.len());
        for idx in (0..layers.len()).rev() {
            let op = match &layers[idx] {
                GpuCrownLayer::Linear { .. } => VjpForwardOp::Linear { idx },
                GpuCrownLayer::Conv2d { .. } => VjpForwardOp::Conv2d { idx },
                GpuCrownLayer::Activation { num_neurons, .. } => match slot_of(idx) {
                    Some(slot) => VjpForwardOp::ReluMask {
                        slot,
                        num_neurons: *num_neurons,
                    },
                    None => VjpForwardOp::Affine { idx },
                },
                _ => return None,
            };
            ops_forward.push(op);
        }

        Some(PointVjpBatchPlan {
            layers_backward: layers,
            mask_positions,
            relu_nodes_backward: relu_nodes,
            input_dim,
            output_dim,
            ops_forward,
        })
    }
}

/// Batched CPU forward over the plan's template for `points` (each a flat
/// `input_dim` vector): returns `(masks, outputs)` where `masks[k][r]` is restart
/// `k`'s 0/1 mask for mask slot `r` (aligned with `plan.mask_positions`, i.e. the
/// fold order the wide backward consumes) and `outputs[k]` is the flat network
/// output (`output_dim`). Restart points are independent → rayon-parallel.
///
/// The mask convention is `pre_act > 0 → 1.0, else 0.0` (a ReLU exactly at 0 has
/// zero slope), matching the exact-at-a-point CROWN ReLU collapse the sequential
/// [`GraphNetwork::attack_point_gradient`] path uses.
pub fn point_vjp_forward_masks(
    plan: &PointVjpBatchPlan,
    points: &[Vec<f32>],
) -> Result<(Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>)> {
    let results: Result<Vec<(Vec<Vec<f32>>, Vec<f32>)>> = points
        .par_iter()
        .map(|x| point_vjp_forward_one(plan, x))
        .collect();
    let mut masks = Vec::with_capacity(points.len());
    let mut outputs = Vec::with_capacity(points.len());
    for (m, o) in results? {
        masks.push(m);
        outputs.push(o);
    }
    Ok((masks, outputs))
}

/// One point's forward over the template: masks (per slot, fold order) + output.
fn point_vjp_forward_one(plan: &PointVjpBatchPlan, x: &[f32]) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
    if x.len() != plan.input_dim {
        return Err(NyError::shape_mismatch(vec![plan.input_dim], vec![x.len()]));
    }
    let mut buf: Vec<f32> = x.to_vec();
    let mut masks: Vec<Vec<f32>> = vec![Vec::new(); plan.mask_positions.len()];
    for op in &plan.ops_forward {
        buf = match *op {
            VjpForwardOp::Linear { idx } => {
                let GpuCrownLayer::Linear {
                    weight,
                    bias,
                    out_features,
                    in_features,
                    ..
                } = &plan.layers_backward[idx]
                else {
                    return Err(NyError::InvalidSpec("point-vjp: op/layer mismatch".into()));
                };
                if buf.len() != *in_features {
                    return Err(NyError::shape_mismatch(vec![*in_features], vec![buf.len()]));
                }
                let w = ArrayView2::from_shape((*out_features, *in_features), weight.as_ref())
                    .map_err(|e| NyError::InvalidSpec(format!("point-vjp linear shape: {e}")))?;
                let mut y = w
                    .dot(&ArrayView1::from(&buf[..]))
                    .into_raw_vec_and_offset()
                    .0;
                if let Some(b) = bias {
                    for (yi, bi) in y.iter_mut().zip(b.iter()) {
                        *yi += bi;
                    }
                }
                y
            }
            VjpForwardOp::Conv2d { idx } => conv2d_forward(&plan.layers_backward[idx], &buf)?,
            VjpForwardOp::Affine { idx } => {
                let GpuCrownLayer::Activation {
                    lower_slope,
                    lower_intercept,
                    num_neurons,
                    ..
                } = &plan.layers_backward[idx]
                else {
                    return Err(NyError::InvalidSpec("point-vjp: op/layer mismatch".into()));
                };
                if buf.len() != *num_neurons {
                    return Err(NyError::shape_mismatch(vec![*num_neurons], vec![buf.len()]));
                }
                buf.iter()
                    .zip(lower_slope.iter().zip(lower_intercept.iter()))
                    .map(|(&v, (&s, &c))| v * s + c)
                    .collect()
            }
            VjpForwardOp::ReluMask { slot, num_neurons } => {
                if buf.len() != num_neurons {
                    return Err(NyError::shape_mismatch(vec![num_neurons], vec![buf.len()]));
                }
                masks[slot] = buf
                    .iter()
                    .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
                    .collect();
                buf.iter().map(|&v| v.max(0.0)).collect()
            }
        };
    }
    if buf.len() != plan.output_dim {
        return Err(NyError::shape_mismatch(
            vec![plan.output_dim],
            vec![buf.len()],
        ));
    }
    Ok((masks, buf))
}

/// Concrete Conv2d forward from the extracted descriptor via im2col + GEMM
/// (`weight_col` is `(out_c, in_c*kh*kw)` row-major, exactly the CROWN backward
/// layout, so forward/backward index the SAME kernel buffer).
/// `pub(in ...)`: shared with the resnet-segment batched VJP's mask-capture
/// forward (`point_vjp_batched_resnet`), which applies the SAME descriptor.
pub(in crate::network::graph_crown) fn conv2d_forward(
    layer: &GpuCrownLayer,
    x: &[f32],
) -> Result<Vec<f32>> {
    let GpuCrownLayer::Conv2d {
        weight_col,
        bias_expanded,
        out_channels,
        in_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        out_h,
        out_w,
        in_h,
        in_w,
        ..
    } = layer
    else {
        return Err(NyError::InvalidSpec("point-vjp: op/layer mismatch".into()));
    };
    let (oc, ic, kh, kw) = (*out_channels, *in_channels, *kernel_h, *kernel_w);
    let (sh, sw, ph, pw) = (*stride_h, *stride_w, *pad_h, *pad_w);
    let (oh, ow, ih, iw) = (*out_h, *out_w, *in_h, *in_w);
    if x.len() != ic * ih * iw {
        return Err(NyError::shape_mismatch(vec![ic, ih, iw], vec![x.len()]));
    }
    let kcols = ic * kh * kw;
    let spatial = oh * ow;
    // im2col: col[(c*kh+ky)*kw+kx, oy*ow+ox] = x[c, oy*sh+ky-ph, ox*sw+kx-pw] (0 pad).
    let mut col = Array2::<f32>::zeros((kcols, spatial));
    for c in 0..ic {
        for ky in 0..kh {
            for kx in 0..kw {
                let row = (c * kh + ky) * kw + kx;
                for oy in 0..oh {
                    let iy = (oy * sh + ky) as isize - ph as isize;
                    if iy < 0 || iy >= ih as isize {
                        continue;
                    }
                    for ox in 0..ow {
                        let ix = (ox * sw + kx) as isize - pw as isize;
                        if ix < 0 || ix >= iw as isize {
                            continue;
                        }
                        col[[row, oy * ow + ox]] = x[(c * ih + iy as usize) * iw + ix as usize];
                    }
                }
            }
        }
    }
    let w = ArrayView2::from_shape((oc, kcols), weight_col.as_ref())
        .map_err(|e| NyError::InvalidSpec(format!("point-vjp conv shape: {e}")))?;
    // (oc, kcols) @ (kcols, oh*ow) → (oc, oh*ow) row-major == flat (oc, oh, ow).
    let mut y = w.dot(&col).into_raw_vec_and_offset().0;
    if let Some(b) = bias_expanded {
        if b.len() == y.len() {
            for (yi, bi) in y.iter_mut().zip(b.iter()) {
                *yi += bi;
            }
        }
    }
    Ok(y)
}

#[cfg(test)]
#[path = "point_vjp_batched_tests.rs"]
mod tests;
