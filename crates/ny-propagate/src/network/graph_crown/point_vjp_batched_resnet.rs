// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched exact point-VJP plan for RESNET DAGs (#batched-vjp-resnet).
//!
//! # Why
//!
//! The pure-chain batched VJP plan ([`super::point_vjp_batched`]) refuses any
//! fan-in, so on the cifar100/tinyimagenet conv RESNETS (residual `Add` blocks)
//! the attack falls back to the sequential [`GraphNetwork::attack_point_gradient`]
//! at ~93 ms/step — a 100 s budget covers ~1000 total steps while the
//! alpha-beta-CROWN reference runs 250 restarts × 1000 steps. This module builds
//! the SAME two host-side ingredients as the chain plan, but over a backward-order
//! [`GpuResnetSegment`] template (chains + identity/projection residual blocks),
//! which the GPU resident fold already knows how to fold: at a concrete point a
//! residual merge's reverse rule is the plain fan-in ADD, exactly what the fold's
//! `Residual`/`ResidualProj` handling computes (`A_in = backward_F(A) + A` /
//! `backward_F(A) + backward_P(A)`). With per-restart 0/1 ReLU mask slopes the
//! folded input-level coefficient rows ARE the exact per-restart point gradients.
//!
//! # Flattened traversal convention (mask slots)
//!
//! Mask slots are indexed by the FLATTENED layer traversal of
//! `segments_backward`: for each segment in (backward) order — `Chain` layers in
//! stored order; `Residual` F-branch layers; `ResidualProj` F-branch then
//! P-branch layers. The GPU stacker ([`ny-gpu`]'s
//! `crown_point_vjp_batched_resnet`) walks the same traversal, so the mask ↔
//! backward-slot correspondence holds BY CONSTRUCTION, and the CPU mask-capture
//! forward below interprets the SAME template (same flat buffers, same layers) so
//! the masks it captures are exactly the ones the backward consumes.
//!
//! # Supported fragment
//!
//! `{Linear, Conv1d, Conv2d, ReLU, Flatten, Reshape, AddConstant, SubConstant,
//! MulConstant, DivConstant}` plus binary residual `Add` merges whose two branches
//! are pure unary chains of the same ops (identity skip or projection skip; nested
//! residuals inside a branch refuse). Anything else returns `None` and the caller
//! keeps its fallbacks. ATTACK-ONLY: gradients steer PGD; every counterexample is
//! concretely re-validated elsewhere (ORT gate), so nothing here can affect a
//! verdict — a wrong gradient could only waste attack steps.

use ndarray::{ArrayView1, ArrayView2};
use ny_core::{GpuCrownBackward, GpuCrownLayer, GpuResnetSegment, NyError, Result};
use ny_tensor::BoundedTensor;
use rayon::prelude::*;

use crate::layers::Layer;
use crate::network::core::{try_extract_single_gpu_layer, GraphNetwork, NETWORK_INPUT};

use super::point_vjp_batched::{conv2d_forward, point_vjp_forward_masks, PointVjpBatchPlan};

/// Shared batched resnet point-VJP plan: the backward-order segment template plus
/// its per-restart ReLU mask slots (flat-traversal positions).
pub struct PointVjpResnetPlan {
    /// Backward-order (output→input) segment template. `Linear`/`Conv2d` weights
    /// are `Arc`-shared; `Activation` entries are either per-restart ReLU MASK
    /// slots (listed in `mask_flat_positions`) or static affine ops.
    pub segments_backward: Vec<GpuResnetSegment>,
    /// ReLU mask slot positions in the FLATTENED layer traversal (see module
    /// docs), in traversal order — the order `masks[k]` must use.
    pub mask_flat_positions: Vec<usize>,
    /// The ReLU node names aligned with `mask_flat_positions` (diagnostics/tests).
    pub relu_nodes: Vec<String>,
    /// Flattened network input dimension.
    pub input_dim: usize,
    /// Flattened network output dimension.
    pub output_dim: usize,
    /// Per-segment base offsets into the flattened traversal (forward interp).
    seg_flat_base: Vec<usize>,
}

/// One branch extracted during the backward walk: its GPU layers (backward
/// order) plus the LOCAL (branch-relative) positions + node names of its ReLU
/// mask slots.
struct BranchExtract {
    layers: Vec<GpuCrownLayer>,
    relu_local: Vec<(usize, String)>,
}

impl GraphNetwork {
    /// Build the batched resnet point-VJP plan for this graph, or `None` when the
    /// graph is outside the supported chain+residual fragment (caller falls back
    /// to the sequential exact gradient). Built ONCE per attack; only the
    /// per-restart masks change between steps.
    ///
    /// `input` is the attack's input box — used only for SHAPES (one cheap
    /// concrete forward at the box center resolves every node's shape; ReLU
    /// template slopes are placeholders, replaced per restart by the masks).
    ///
    /// Returns `Some` only when the graph contains at least one residual `Add`
    /// (pure chains keep the proven [`Self::build_point_vjp_batch_plan`] path).
    pub fn build_point_vjp_resnet_plan(&self, input: &BoundedTensor) -> Option<PointVjpResnetPlan> {
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
        let resolve = |name: &str| -> Option<&BoundedTensor> {
            if name == NETWORK_INPUT {
                Some(&center)
            } else {
                node_bounds.get(name)
            }
        };

        let mut segments: Vec<GpuResnetSegment> = Vec::new();
        let mut mask_flat: Vec<usize> = Vec::new();
        let mut relu_nodes: Vec<String> = Vec::new();
        let mut flat_total = 0usize; // layers committed to `segments` so far
        let mut chain = BranchExtract {
            layers: Vec::new(),
            relu_local: Vec::new(),
        };
        let mut current = output_name;
        let mut saw_residual = false;
        let max_steps = self.nodes.len() + 1;
        let mut steps = 0usize;

        // Commit a finished branch: rebase its local ReLU slots onto the global
        // flat traversal and advance the committed-layer counter.
        let commit = |b: BranchExtract,
                      flat_total: &mut usize,
                      mask_flat: &mut Vec<usize>,
                      relu_nodes: &mut Vec<String>|
         -> Vec<GpuCrownLayer> {
            for (local, name) in b.relu_local {
                mask_flat.push(*flat_total + local);
                relu_nodes.push(name);
            }
            *flat_total += b.layers.len();
            b.layers
        };

        loop {
            steps += 1;
            if steps > max_steps {
                return None;
            }
            if current == NETWORK_INPUT {
                break;
            }
            let node = self.nodes.get(&current)?;
            match node.inputs.len() {
                1 => {
                    let input_name = node.require_unary_input().ok()?.to_string();
                    let pre = resolve(&input_name)?;
                    extract_attack_layer(&current, &node.layer, pre, &mut chain)?;
                    current = input_name;
                }
                2 => {
                    // Residual merge — Add only (exact identity Jacobian).
                    if !matches!(node.layer, Layer::Add(_)) {
                        return None;
                    }
                    // Flush the plain chain accumulated downstream of this block.
                    if !chain.layers.is_empty() {
                        let layers = commit(
                            std::mem::replace(
                                &mut chain,
                                BranchExtract {
                                    layers: Vec::new(),
                                    relu_local: Vec::new(),
                                },
                            ),
                            &mut flat_total,
                            &mut mask_flat,
                            &mut relu_nodes,
                        );
                        segments.push(GpuResnetSegment::Chain(layers));
                    }
                    let in_a = node.inputs[0].clone();
                    let in_b = node.inputs[1].clone();
                    let z = self.vjp_common_ancestor(&in_a, &in_b)?;
                    let fa = extract_attack_branch(self, &in_a, &z, &resolve)?;
                    let fb = extract_attack_branch(self, &in_b, &z, &resolve)?;
                    let seg = match (fa.layers.is_empty(), fb.layers.is_empty()) {
                        // out = z + z: not a residual block we model.
                        (true, true) => return None,
                        (false, true) => {
                            let f = commit(fa, &mut flat_total, &mut mask_flat, &mut relu_nodes);
                            GpuResnetSegment::Residual(f)
                        }
                        (true, false) => {
                            let f = commit(fb, &mut flat_total, &mut mask_flat, &mut relu_nodes);
                            GpuResnetSegment::Residual(f)
                        }
                        (false, false) => {
                            // Projection skip: F then P in flat traversal order.
                            let f = commit(fa, &mut flat_total, &mut mask_flat, &mut relu_nodes);
                            let p = commit(fb, &mut flat_total, &mut mask_flat, &mut relu_nodes);
                            GpuResnetSegment::ResidualProj(f, p)
                        }
                    };
                    segments.push(seg);
                    saw_residual = true;
                    current = z;
                }
                _ => return None,
            }
        }
        if !chain.layers.is_empty() {
            let layers = commit(chain, &mut flat_total, &mut mask_flat, &mut relu_nodes);
            segments.push(GpuResnetSegment::Chain(layers));
        }
        if segments.is_empty() || !saw_residual {
            return None;
        }

        // Per-segment flat base offsets (forward interpretation).
        let mut seg_flat_base = Vec::with_capacity(segments.len());
        let mut base = 0usize;
        for seg in &segments {
            seg_flat_base.push(base);
            base += match seg {
                GpuResnetSegment::Chain(l) | GpuResnetSegment::Residual(l) => l.len(),
                GpuResnetSegment::ResidualProj(f, p) => f.len() + p.len(),
            };
        }
        debug_assert_eq!(base, flat_total);

        Some(PointVjpResnetPlan {
            segments_backward: segments,
            mask_flat_positions: mask_flat,
            relu_nodes,
            input_dim,
            output_dim,
            seg_flat_base,
        })
    }

    /// The topologically-latest common ancestor of `in_a` and `in_b` (the residual
    /// block input `z`), or `NETWORK_INPUT` when their only common origin is the
    /// network input. Mirrors the certified resnet decomposition's rule.
    fn vjp_common_ancestor(&self, in_a: &str, in_b: &str) -> Option<String> {
        if in_a == NETWORK_INPUT || in_b == NETWORK_INPUT {
            return Some(NETWORK_INPUT.to_string());
        }
        let anc = self.all_ancestors().ok()?;
        let anc_a = anc.get(in_a)?;
        let set_b: std::collections::HashSet<&str> =
            anc.get(in_b)?.iter().map(String::as_str).collect();
        if let Some(z) = anc_a.iter().rev().find(|n| set_b.contains(n.as_str())) {
            return Some(z.clone());
        }
        Some(NETWORK_INPUT.to_string())
    }
}

/// Extract one unary node's GPU layer(s) into the branch, restricted to the
/// EXACT-at-a-point attack fragment, recording ReLU mask slots. `None` → refuse
/// (caller falls back). NOTE: the whitelist matters — `try_extract_single_gpu_layer`
/// also accepts S-shaped activations whose baked relaxation would NOT be the exact
/// point derivative (and is never re-baked per restart), so only the listed ops pass.
fn extract_attack_layer(
    node_name: &str,
    layer: &Layer,
    pre: &BoundedTensor,
    branch: &mut BranchExtract,
) -> Option<()> {
    match layer {
        Layer::ReLU(_) => {
            let before = branch.layers.len();
            try_extract_single_gpu_layer(layer, pre, &mut branch.layers)?;
            // A ReLU extracts to exactly one Activation (placeholder slopes).
            if branch.layers.len() != before + 1
                || !matches!(branch.layers[before], GpuCrownLayer::Activation { .. })
            {
                return None;
            }
            branch.relu_local.push((before, node_name.to_string()));
            Some(())
        }
        Layer::Linear(_)
        | Layer::Conv1d(_)
        | Layer::Conv2d(_)
        | Layer::Flatten(_)
        | Layer::Reshape(_)
        | Layer::AddConstant(_)
        | Layer::SubConstant(_)
        | Layer::MulConstant(_)
        | Layer::DivConstant(_) => try_extract_single_gpu_layer(layer, pre, &mut branch.layers),
        _ => None,
    }
}

/// Walk a pure unary chain backward from `branch_start` until reaching `z`,
/// extracting each node's layer(s). `branch_start == z` yields an empty branch
/// (the identity-skip case). `None` → not a clean unary path to `z`.
fn extract_attack_branch<'a>(
    graph: &GraphNetwork,
    branch_start: &str,
    z: &str,
    resolve: &impl Fn(&str) -> Option<&'a BoundedTensor>,
) -> Option<BranchExtract> {
    let mut branch = BranchExtract {
        layers: Vec::new(),
        relu_local: Vec::new(),
    };
    let mut current = branch_start.to_string();
    let max_steps = graph.nodes.len() + 1;
    let mut steps = 0usize;
    while current != z {
        steps += 1;
        if steps > max_steps {
            return None;
        }
        // Overshooting to the network input means `z` is not on this chain.
        if current == NETWORK_INPUT {
            return None;
        }
        let node = graph.nodes.get(&current)?;
        if node.inputs.len() != 1 {
            // Nested residual inside a branch — refuse.
            return None;
        }
        let input_name = node.require_unary_input().ok()?.to_string();
        let pre = resolve(&input_name)?;
        extract_attack_layer(&current, &node.layer, pre, &mut branch)?;
        current = input_name;
    }
    Some(branch)
}

/// Batched CPU forward over the resnet plan's template for `points` (each a flat
/// `input_dim` vector): returns `(masks, outputs)` where `masks[k][r]` is restart
/// `k`'s 0/1 mask for mask slot `r` (aligned with `plan.mask_flat_positions`) and
/// `outputs[k]` is the flat network output. Restart points are independent →
/// rayon-parallel. Mask convention: `pre_act > 0 → 1.0, else 0.0` — identical to
/// the chain plan and the sequential exact gradient.
pub fn point_vjp_resnet_forward_masks(
    plan: &PointVjpResnetPlan,
    points: &[Vec<f32>],
) -> Result<(Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>)> {
    let results: Result<Vec<(Vec<Vec<f32>>, Vec<f32>)>> = points
        .par_iter()
        .map(|x| point_vjp_resnet_forward_one(plan, x))
        .collect();
    let mut masks = Vec::with_capacity(points.len());
    let mut outputs = Vec::with_capacity(points.len());
    for (m, o) in results? {
        masks.push(m);
        outputs.push(o);
    }
    Ok((masks, outputs))
}

/// One point's forward over the segment template (segments interpreted in
/// REVERSE = input→output order; each branch's layers applied in reverse of
/// their stored backward order).
fn point_vjp_resnet_forward_one(
    plan: &PointVjpResnetPlan,
    x: &[f32],
) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
    if x.len() != plan.input_dim {
        return Err(NyError::shape_mismatch(vec![plan.input_dim], vec![x.len()]));
    }
    let mut buf: Vec<f32> = x.to_vec();
    let mut masks: Vec<Vec<f32>> = vec![Vec::new(); plan.mask_flat_positions.len()];
    for (seg_idx, seg) in plan.segments_backward.iter().enumerate().rev() {
        let base = plan.seg_flat_base[seg_idx];
        buf = match seg {
            GpuResnetSegment::Chain(layers) => {
                apply_branch_forward(plan, layers, base, buf, &mut masks)?
            }
            GpuResnetSegment::Residual(f) => {
                let z = buf.clone();
                let mut y = apply_branch_forward(plan, f, base, buf, &mut masks)?;
                if y.len() != z.len() {
                    return Err(NyError::shape_mismatch(vec![z.len()], vec![y.len()]));
                }
                for (yi, zi) in y.iter_mut().zip(z.iter()) {
                    *yi += zi;
                }
                y
            }
            GpuResnetSegment::ResidualProj(f, p) => {
                let fo = apply_branch_forward(plan, f, base, buf.clone(), &mut masks)?;
                let po = apply_branch_forward(plan, p, base + f.len(), buf, &mut masks)?;
                if fo.len() != po.len() {
                    return Err(NyError::shape_mismatch(vec![fo.len()], vec![po.len()]));
                }
                fo.iter().zip(po.iter()).map(|(&a, &b)| a + b).collect()
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

/// Apply one branch's layers FORWARD (reverse of stored backward order) to the
/// flat buffer, capturing masks at the branch's ReLU slots (`base + stored_idx`
/// is a layer's flat-traversal position).
fn apply_branch_forward(
    plan: &PointVjpResnetPlan,
    branch: &[GpuCrownLayer],
    base: usize,
    mut buf: Vec<f32>,
    masks: &mut [Vec<f32>],
) -> Result<Vec<f32>> {
    for stored_idx in (0..branch.len()).rev() {
        let flat = base + stored_idx;
        let slot = plan.mask_flat_positions.iter().position(|&p| p == flat);
        buf = match &branch[stored_idx] {
            GpuCrownLayer::Linear {
                weight,
                bias,
                out_features,
                in_features,
            } => {
                if buf.len() != *in_features {
                    return Err(NyError::shape_mismatch(vec![*in_features], vec![buf.len()]));
                }
                let w = ArrayView2::from_shape((*out_features, *in_features), weight.as_ref())
                    .map_err(|e| NyError::InvalidSpec(format!("resnet-vjp linear shape: {e}")))?;
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
            layer @ GpuCrownLayer::Conv2d { .. } => conv2d_forward(layer, &buf)?,
            GpuCrownLayer::Activation {
                lower_slope,
                lower_intercept,
                num_neurons,
                ..
            } => {
                if buf.len() != *num_neurons {
                    return Err(NyError::shape_mismatch(vec![*num_neurons], vec![buf.len()]));
                }
                if let Some(r) = slot {
                    // Per-restart ReLU mask slot: capture `pre > 0`, apply ReLU.
                    masks[r] = buf
                        .iter()
                        .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
                        .collect();
                    buf.iter().map(|&v| v.max(0.0)).collect()
                } else {
                    // Static affine (constant arithmetic): lower == upper.
                    buf.iter()
                        .zip(lower_slope.iter().zip(lower_intercept.iter()))
                        .map(|(&v, (&s, &c))| v * s + c)
                        .collect()
                }
            }
            _ => {
                return Err(NyError::InvalidSpec(
                    "resnet-vjp: unsupported layer in forward".into(),
                ))
            }
        };
    }
    Ok(buf)
}

/// Unified batched point-VJP plan: the pure-chain plan when the graph is a pure
/// unary chain, else the resnet-segment plan when the graph is a clean
/// chain+residual DAG. One `build`/`forward_masks`/`gpu_vjp` surface so the
/// attack drivers stay template-agnostic.
pub enum PointVjpWavePlan {
    /// Pure unary chain (the original #batched-vjp plan).
    Chain(PointVjpBatchPlan),
    /// Chain + residual blocks (#batched-vjp-resnet).
    Resnet(PointVjpResnetPlan),
}

impl PointVjpWavePlan {
    /// Build the batched plan for this graph, preferring the proven chain plan;
    /// `None` when neither template fits (caller falls back to the sequential
    /// exact-gradient loop).
    pub fn build(graph: &GraphNetwork, input: &BoundedTensor) -> Option<Self> {
        if let Some(p) = graph.build_point_vjp_batch_plan(input) {
            return Some(Self::Chain(p));
        }
        graph.build_point_vjp_resnet_plan(input).map(Self::Resnet)
    }

    /// Flattened network output dimension.
    pub fn output_dim(&self) -> usize {
        match self {
            Self::Chain(p) => p.output_dim,
            Self::Resnet(p) => p.output_dim,
        }
    }

    /// Flattened network input dimension.
    pub fn input_dim(&self) -> usize {
        match self {
            Self::Chain(p) => p.input_dim,
            Self::Resnet(p) => p.input_dim,
        }
    }

    /// Batched CPU template forward: per-restart ReLU masks (slot order) +
    /// network outputs.
    pub fn forward_masks(
        &self,
        points: &[Vec<f32>],
    ) -> Result<(Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>)> {
        match self {
            Self::Chain(p) => point_vjp_forward_masks(p, points),
            Self::Resnet(p) => point_vjp_resnet_forward_masks(p, points),
        }
    }

    /// ONE wide GPU pass: all K exact per-restart gradients (`spec_rows` is
    /// `K × output_dim` row-major). Any `Err` → caller's sequential fallback.
    pub fn gpu_vjp(
        &self,
        gpu: &dyn GpuCrownBackward,
        masks: &[Vec<Vec<f32>>],
        spec_rows: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Chain(p) => gpu.crown_point_vjp_batched(
                &p.layers_backward,
                &p.mask_positions,
                masks,
                spec_rows,
                p.output_dim,
                p.input_dim,
            ),
            Self::Resnet(p) => gpu.crown_point_vjp_batched_resnet(
                &p.segments_backward,
                &p.mask_flat_positions,
                masks,
                spec_rows,
                p.output_dim,
                p.input_dim,
            ),
        }
    }
}

#[cfg(test)]
#[path = "point_vjp_batched_resnet_tests.rs"]
mod tests;
