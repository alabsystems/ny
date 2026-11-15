// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched point-VJP plan for a MAIN network plus additive input-fed SIDE
//! BRANCHES (#vjp-side-branches).
//!
//! # Why
//!
//! SoundnessBench's `model_residual.onnx` is a plain conv chain (which the chain
//! plan handles) with three small "carrier" branches ADDED to its output:
//!
//! ```text
//! out = main(x) + Σ_k W_k · ReLU(A_k x + c_k) + bias
//! ```
//!
//! where each carrier is an AFFINE chain from the input (Reshape → Conv → Conv →
//! Conv), one ReLU, then an affine chain (Flatten → MatMul) into the sum. The
//! chain plan refuses the fan-in and the resnet plan refuses the nested `Add`s,
//! so the batched multi-restart attack fell back to ~4 sequential restarts where
//! alpha-beta-CROWN runs 250 — and the planted counterexamples sit in thin
//! slivers only many restarts find.
//!
//! # What it does
//!
//! Decomposes the output into its `Add`/`AddConstant` addends; exactly ONE addend
//! is the main network (built on the proven chain/resnet plan over the truncated
//! subgraph that computes it), every other is a side branch
//! `post(ReLU(pre(x)))` whose `pre`/`post` are chains of affine layers, captured
//! EXACTLY as (sparse) matrices by evaluating the chains at 0 and at unit
//! vectors. The forward adds each branch's value; the VJP adds
//! `Aᵀ(mask ⊙ Wᵀ s)` per branch to the main GPU gradient.
//!
//! Before it is returned, the decomposed forward is checked against the graph's
//! own concrete forward at several points; any mismatch refuses the plan.
//!
//! ATTACK-ONLY: gradients steer PGD; every counterexample is re-validated on the
//! full graph and by the trusted ORT gate, so nothing here can affect a verdict.
//!
//! # The exact-f64 twin (#sbench-f64-attack)
//!
//! [`PointVjpF64Plan`] re-evaluates the SAME decomposition (chain main network
//! plus side branches, or a pure chain) in plain `f64` arithmetic, with an
//! exact-`f64` point VJP. SoundnessBench's planted counterexample regions are
//! thinner than the f32 output resolution (measured ORT margins of the field's
//! witnesses: 1e-7..1e-5, against an f32 ULP of ~1e-6 at the outputs'
//! magnitude), so a search that ranks points by an f32 forward chases rounding
//! noise. The twin supplies a noise-free margin and its exact gradient to the
//! wrapper's float64 target-margin repair. Same ATTACK-ONLY contract: nothing
//! here is a bound, and every candidate still passes the trusted ORT + true-f64
//! gate before a `sat` is published.

use std::collections::HashSet;
use std::time::Instant;

use ndarray::{Array2, ArrayD, ArrayView1, IxDyn};
use ny_core::{GpuCrownBackward, GpuCrownLayer, Result};
use ny_tensor::BoundedTensor;
use rayon::prelude::*;

use crate::layers::Layer;
use crate::network::core::{GraphNetwork, NETWORK_INPUT};
use crate::network::graph_ibp_f64_cell::Interval64;

use super::point_vjp_batched::PointVjpBatchPlan;
use super::point_vjp_batched_resnet::PointVjpWavePlan;

/// One input-fed side branch `y = W ReLU(A x + c) + b`, stored sparse.
#[derive(Debug, Clone)]
pub(crate) struct SideBranch {
    /// The branch's affine `pre` chain (input → ReLU input), forward order.
    /// Retained so the exact-f64 twin can re-capture `A`/`c` without the f32
    /// rounding of [`affine_matrix`] (#sbench-f64-attack).
    pre_nodes: Vec<String>,
    /// The branch's affine `post` chain (ReLU output → addend), forward order,
    /// and the ReLU shape `W`/`b` were captured at. Retained so the exact-f64
    /// twin can re-capture `W`/`b` when the f32 capture is not provably exact
    /// (a biased post map rounds `W + b`; #sbench-f64-attack review).
    post_nodes: Vec<String>,
    relu_shape: Vec<usize>,
    /// Rows of `A` (one per hidden unit): `(input index, coefficient)`.
    a_rows: Vec<Vec<(u32, f32)>>,
    c: Vec<f32>,
    /// Columns of `W` (one per hidden unit): `(output index, coefficient)`.
    w_cols: Vec<Vec<(u32, f32)>>,
    b: Vec<f32>,
}

impl SideBranch {
    fn hidden(&self) -> usize {
        self.c.len()
    }

    /// Pre-activations `A x + c`.
    fn pre(&self, x: &[f32]) -> Vec<f32> {
        self.a_rows
            .iter()
            .zip(&self.c)
            .map(|(row, &c)| c + row.iter().map(|&(k, a)| a * x[k as usize]).sum::<f32>())
            .collect()
    }

    /// Add `W ReLU(z) + b` into `out`; returns the 0/1 mask of `z > 0`.
    fn forward_into(&self, x: &[f32], out: &mut [f32]) -> Vec<f32> {
        for (o, &b) in out.iter_mut().zip(&self.b) {
            *o += b;
        }
        let z = self.pre(x);
        let mut mask = Vec::with_capacity(z.len());
        for (i, &zi) in z.iter().enumerate() {
            if zi > 0.0 {
                mask.push(1.0);
                for &(j, w) in &self.w_cols[i] {
                    out[j as usize] += w * zi;
                }
            } else {
                mask.push(0.0);
            }
        }
        mask
    }

    /// Add `Aᵀ (mask ⊙ Wᵀ s)` into `grad`.
    fn vjp_into(&self, mask: &[f32], s: &[f32], grad: &mut [f32]) {
        for i in 0..self.hidden() {
            if mask[i] == 0.0 {
                continue;
            }
            let t: f32 = self.w_cols[i].iter().map(|&(j, w)| w * s[j as usize]).sum();
            if t == 0.0 {
                continue;
            }
            for &(k, a) in &self.a_rows[i] {
                grad[k as usize] += t * a;
            }
        }
    }
}

/// Main-network plan plus exact side-branch terms.
pub struct PointVjpSideBranchPlan {
    pub(crate) main: Box<PointVjpWavePlan>,
    main_slots: usize,
    pub(crate) branches: Vec<SideBranch>,
    out_bias: Vec<f32>,
    input_dim: usize,
    output_dim: usize,
    /// The attack box's input shape the branch maps were captured at.
    input_shape: Vec<usize>,
}

impl PointVjpSideBranchPlan {
    /// Flattened output dimension.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Flattened input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Per-restart masks (main slots, then one slot per branch) and outputs.
    pub fn forward_masks(
        &self,
        points: &[Vec<f32>],
    ) -> Result<(Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>)> {
        let (mut masks, mut outputs) = self.main.forward_masks(points)?;
        for ((m, out), x) in masks.iter_mut().zip(outputs.iter_mut()).zip(points) {
            if m.len() != self.main_slots
                || out.len() != self.output_dim
                || x.len() != self.input_dim
            {
                return Err(ny_core::NyError::InvalidSpec(
                    "vjp-side-branches: main plan shape drifted".into(),
                ));
            }
            for (o, &b) in out.iter_mut().zip(&self.out_bias) {
                *o += b;
            }
            for br in &self.branches {
                m.push(br.forward_into(x, out));
            }
        }
        Ok((masks, outputs))
    }

    /// Main gradient on the GPU plus every branch's exact term on the CPU.
    pub fn gpu_vjp(
        &self,
        gpu: &dyn GpuCrownBackward,
        masks: &[Vec<Vec<f32>>],
        spec_rows: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        let expected = self.main_slots + self.branches.len();
        if masks.iter().any(|m| m.len() != expected)
            || spec_rows.len() != masks.len() * self.output_dim
        {
            return Err(ny_core::NyError::InvalidSpec(
                "vjp-side-branches: mask/spec layout mismatch".into(),
            ));
        }
        let main_masks: Vec<Vec<Vec<f32>>> = masks
            .iter()
            .map(|m| m[..self.main_slots].to_vec())
            .collect();
        let mut grads = self.main.gpu_vjp(gpu, &main_masks, spec_rows)?;
        self.add_side_vjp(masks, spec_rows, &mut grads);
        Ok(grads)
    }

    /// The branches' share of the per-restart gradients (split out so it is
    /// testable without a GPU).
    pub(crate) fn add_side_vjp(
        &self,
        masks: &[Vec<Vec<f32>>],
        spec_rows: &[f32],
        grads: &mut [Vec<f32>],
    ) {
        for (k, (m, g)) in masks.iter().zip(grads.iter_mut()).enumerate() {
            let s = &spec_rows[k * self.output_dim..(k + 1) * self.output_dim];
            for (bi, br) in self.branches.iter().enumerate() {
                br.vjp_into(&m[self.main_slots + bi], s, g);
            }
        }
    }
}

/// Layers that are affine in their single input.
fn is_affine(layer: &Layer) -> bool {
    matches!(
        layer,
        Layer::Linear(_)
            | Layer::Conv1d(_)
            | Layer::Conv2d(_)
            | Layer::Reshape(_)
            | Layer::Flatten(_)
            | Layer::Transpose(_)
            | Layer::AddConstant(_)
            | Layer::SubConstant(_)
            | Layer::MulConstant(_)
            | Layer::DivConstant(_)
    )
}

/// A side branch's node chains: `pre` (input → ReLU input) and `post` (ReLU →
/// addend), both in FORWARD order.
struct BranchChains {
    pre: Vec<String>,
    relu: String,
    post: Vec<String>,
}

impl GraphNetwork {
    /// Build the main-plus-side-branches plan, or `None` when the graph does not
    /// decompose that way (the caller keeps its fallbacks).
    pub fn build_point_vjp_side_branch_plan(
        &self,
        input: &BoundedTensor,
    ) -> Option<PointVjpSideBranchPlan> {
        let mut terms = Vec::new();
        let mut consts: Vec<ArrayD<f32>> = Vec::new();
        self.collect_addends(self.output_name(), &mut terms, &mut consts, 0)?;
        if terms.len() < 2 {
            return None;
        }
        // Exactly one addend that is NOT side-branch shaped is the main network.
        // When every addend is side-branch shaped, the first is served by the
        // chain/resnet plan instead (any choice is exact); two or more
        // non-side addends refuse.
        let classified: Vec<(String, Option<BranchChains>)> = terms
            .into_iter()
            .map(|t| {
                let c = self.side_branch_chains(&t);
                (t, c)
            })
            .collect();
        let mains = classified.iter().filter(|(_, c)| c.is_none()).count();
        let mut main: Option<String> = None;
        let mut chains = Vec::new();
        for (i, (t, c)) in classified.into_iter().enumerate() {
            match c {
                None => main = Some(t),
                Some(_) if mains == 0 && i == 0 => main = Some(t),
                Some(c) => chains.push(c),
            }
        }
        if mains > 1 {
            return None;
        }
        let main = main?;
        if chains.is_empty() {
            return None;
        }
        let sub = self.ancestor_subgraph(&main)?;
        let main_plan = match sub.build_point_vjp_batch_plan(input) {
            Some(p) => PointVjpWavePlan::Chain(p),
            None => PointVjpWavePlan::Resnet(sub.build_point_vjp_resnet_plan(input)?),
        };
        let input_dim = input.len();
        let output_dim = main_plan.output_dim();
        if main_plan.input_dim() != input_dim || output_dim == 0 {
            return None;
        }
        let mut out_bias = vec![0.0f32; output_dim];
        for c in &consts {
            match c.len() {
                1 => out_bias
                    .iter_mut()
                    .for_each(|o| *o += c.iter().next().copied().unwrap_or(0.0)),
                n if n == output_dim => out_bias
                    .iter_mut()
                    .zip(c.iter())
                    .for_each(|(o, &v)| *o += v),
                _ => return None,
            }
        }
        let center = BoundedTensor::concrete(input.center()).ok()?;
        let bounds = self.collect_node_bounds(&center).ok()?;
        let mut branches = Vec::with_capacity(chains.len());
        for ch in &chains {
            let relu_shape = bounds.get(&ch.relu)?.shape().to_vec();
            let pre_in_shape = input.shape().to_vec();
            let (a_dense, c) = if ch.pre.is_empty() {
                let h = input_dim;
                let mut a = vec![0.0f32; h * h];
                (0..h).for_each(|i| a[i * h + i] = 1.0);
                (a, vec![0.0; h])
            } else {
                let g = self.chain_graph(&ch.pre, false)?;
                affine_matrix(&g, &pre_in_shape)?
            };
            let h = c.len();
            if h == 0 || a_dense.len() != h * input_dim || relu_shape.iter().product::<usize>() != h
            {
                return None;
            }
            let (w_dense, b) = if ch.post.is_empty() {
                let mut w = vec![0.0f32; h * h];
                (0..h).for_each(|i| w[i * h + i] = 1.0);
                (w, vec![0.0; h])
            } else {
                let g = self.chain_graph(&ch.post, true)?;
                affine_matrix(&g, &relu_shape)?
            };
            if b.len() != output_dim || w_dense.len() != output_dim * h {
                return None;
            }
            // Row-major dense `A` is h x d; `W` is out x h.
            let a_rows = (0..h)
                .map(|i| {
                    (0..input_dim)
                        .filter_map(|k| {
                            let v = a_dense[i * input_dim + k];
                            (v != 0.0).then_some((k as u32, v))
                        })
                        .collect()
                })
                .collect();
            let w_cols = (0..h)
                .map(|i| {
                    (0..output_dim)
                        .filter_map(|j| {
                            let v = w_dense[j * h + i];
                            (v != 0.0).then_some((j as u32, v))
                        })
                        .collect()
                })
                .collect();
            branches.push(SideBranch {
                pre_nodes: ch.pre.clone(),
                post_nodes: ch.post.clone(),
                relu_shape,
                a_rows,
                c,
                w_cols,
                b,
            });
        }
        let probe = center.center().iter().copied().collect::<Vec<f32>>();
        let main_slots = main_plan
            .forward_masks(std::slice::from_ref(&probe))
            .ok()?
            .0
            .first()?
            .len();
        let plan = PointVjpSideBranchPlan {
            main: Box::new(main_plan),
            main_slots,
            branches,
            out_bias,
            input_dim,
            output_dim,
            input_shape: input.shape().to_vec(),
        };
        // Self-check: the decomposed forward must reproduce the graph's own
        // concrete forward (centre, both corners, and an alternating corner).
        let lo: Vec<f32> = input.lower().iter().copied().collect();
        let hi: Vec<f32> = input.upper().iter().copied().collect();
        let alt: Vec<f32> = lo
            .iter()
            .zip(&hi)
            .enumerate()
            .map(|(i, (&l, &h))| if i % 2 == 0 { l } else { h })
            .collect();
        let points = vec![probe, lo, hi, alt];
        let (_, outs) = plan.forward_masks(&points).ok()?;
        for (x, y) in points.iter().zip(&outs) {
            let arr = ArrayD::from_shape_vec(IxDyn(input.shape()), x.clone()).ok()?;
            let full = self
                .propagate_concrete_point(&BoundedTensor::concrete(arr).ok()?, None, None)
                .ok()?
                .center();
            if full.len() != y.len() {
                return None;
            }
            let scale = 1.0 + full.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            if full.iter().zip(y).any(|(a, b)| {
                !matches!(
                    (a - b).abs().partial_cmp(&(1e-4 * scale)),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                )
            }) {
                return None;
            }
        }
        eprintln!(
            "[vjp-side-branches] main={} branches={} hidden={:?}",
            match plan.main.as_ref() {
                PointVjpWavePlan::Chain(_) => "chain",
                PointVjpWavePlan::Resnet(_) => "resnet",
                PointVjpWavePlan::SideBranched(_) => "side",
            },
            plan.branches.len(),
            plan.branches
                .iter()
                .map(SideBranch::hidden)
                .collect::<Vec<_>>()
        );
        Some(plan)
    }

    /// Split `name` into `Add` addends and `AddConstant` constants.
    fn collect_addends(
        &self,
        name: &str,
        terms: &mut Vec<String>,
        consts: &mut Vec<ArrayD<f32>>,
        depth: usize,
    ) -> Option<()> {
        if depth > 64 || name == NETWORK_INPUT {
            return None;
        }
        let node = self.node(name)?;
        match (&node.layer, node.inputs.len()) {
            (Layer::Add(_), 2) => {
                self.collect_addends(&node.inputs[0], terms, consts, depth + 1)?;
                self.collect_addends(&node.inputs[1], terms, consts, depth + 1)
            }
            (Layer::AddConstant(c), 1) => {
                consts.push(c.constant.clone());
                self.collect_addends(&node.inputs[0], terms, consts, depth + 1)
            }
            _ => {
                terms.push(name.to_string());
                Some(())
            }
        }
    }

    /// `name` as `post(ReLU(pre(x)))` with affine single-input chains, or `None`.
    fn side_branch_chains(&self, name: &str) -> Option<BranchChains> {
        let mut post = Vec::new();
        let mut cur = name.to_string();
        let relu = loop {
            if cur == NETWORK_INPUT || post.len() > 64 {
                return None;
            }
            let node = self.node(&cur)?;
            if node.inputs.len() != 1 {
                return None;
            }
            if matches!(node.layer, Layer::ReLU(_)) {
                break cur.clone();
            }
            if !is_affine(&node.layer) {
                return None;
            }
            post.push(cur.clone());
            cur = node.inputs[0].clone();
        };
        let mut pre = Vec::new();
        cur = self.node(&relu)?.inputs[0].clone();
        while cur != NETWORK_INPUT {
            if pre.len() > 64 {
                return None;
            }
            let node = self.node(&cur)?;
            if node.inputs.len() != 1 || !is_affine(&node.layer) {
                return None;
            }
            pre.push(cur.clone());
            cur = node.inputs[0].clone();
        }
        pre.reverse();
        post.reverse();
        Some(BranchChains { pre, relu, post })
    }

    /// The subgraph computing `out` (its ancestor closure, in execution order).
    fn ancestor_subgraph(&self, out: &str) -> Option<GraphNetwork> {
        let mut need: HashSet<String> = HashSet::new();
        let mut stack = vec![out.to_string()];
        need.insert(out.to_string());
        while let Some(n) = stack.pop() {
            for inp in &self.node(&n)?.inputs {
                if inp != NETWORK_INPUT && need.insert(inp.clone()) {
                    stack.push(inp.clone());
                }
            }
        }
        let mut g = GraphNetwork::new();
        for n in self.exec_order().ok()? {
            if need.contains(n) {
                g.try_add_node(self.node(n)?.clone()).ok()?;
            }
        }
        g.set_output(out);
        Some(g)
    }

    /// A single-input chain as its own graph; `rebase` rewires the first node
    /// onto the graph input (for a chain that starts after a ReLU).
    fn chain_graph(&self, nodes: &[String], rebase: bool) -> Option<GraphNetwork> {
        let mut g = GraphNetwork::new();
        for (i, n) in nodes.iter().enumerate() {
            let mut node = self.node(n)?.clone();
            if i == 0 && rebase {
                node.inputs = vec![NETWORK_INPUT.to_string()];
            }
            g.try_add_node(node).ok()?;
        }
        g.set_output(nodes.last()?.clone());
        Some(g)
    }
}

/// `(M row-major, m)` with `g(x) = M x + m` for an affine graph `g` whose input
/// has shape `in_shape`, from evaluations at 0 and at every unit vector.
fn affine_matrix(g: &GraphNetwork, in_shape: &[usize]) -> Option<(Vec<f32>, Vec<f32>)> {
    let n: usize = in_shape.iter().product();
    let eval = |x: Vec<f32>| -> Option<Vec<f32>> {
        let arr = ArrayD::from_shape_vec(IxDyn(in_shape), x).ok()?;
        let out = g
            .propagate_concrete_point(&BoundedTensor::concrete(arr).ok()?, None, None)
            .ok()?
            .center();
        let v: Vec<f32> = out.iter().copied().collect();
        v.iter().all(|x| x.is_finite()).then_some(v)
    };
    let m = eval(vec![0.0; n])?;
    let rows = m.len();
    let mut dense = vec![0.0f32; rows * n];
    for k in 0..n {
        let mut e = vec![0.0f32; n];
        e[k] = 1.0;
        let col = eval(e)?;
        if col.len() != rows {
            return None;
        }
        for r in 0..rows {
            dense[r * n + k] = col[r] - m[r];
        }
    }
    Some((dense, m))
}

// ---------------------------------------------------------------------------
// #sbench-f64-attack: the exact-f64 twin of the chain / side-branch plans.
// ---------------------------------------------------------------------------

/// Conv2d geometry of one twin op (the `GpuCrownLayer::Conv2d` descriptor).
#[derive(Debug, Clone, Copy)]
struct F64ConvGeom {
    oc: usize,
    ic: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    oh: usize,
    ow: usize,
    ih: usize,
    iw: usize,
}

/// One forward-order op of the exact-f64 chain twin. Weights are the f32
/// template's own values widened exactly (`f64::from` is lossless), so the twin
/// evaluates the SAME real function the f32 template rounds.
#[derive(Debug, Clone)]
enum F64ChainOp {
    /// `y = W x + b`, `W` row-major `(out, in)`.
    Linear { w: Array2<f64>, b: Option<Vec<f64>> },
    /// im2col convolution over the same descriptor the f32 template uses.
    Conv2d {
        w_col: Array2<f64>,
        bias: Option<Vec<f64>>,
        /// Boxed: inline it made `Conv2d` 184 bytes against `Linear`'s 88
        /// (clippy::large_enum_variant, workspace threshold 64).
        geom: Box<F64ConvGeom>,
    },
    /// Static constant arithmetic `y = s * x + t` (a template entry whose lower
    /// and upper relaxations coincide). A constant the template already folded
    /// in f32 (e.g. `DivConstant`'s reciprocal) keeps that fold: guidance only.
    Affine {
        slope: Vec<f64>,
        intercept: Vec<f64>,
    },
    /// A per-point ReLU; the tape records `x > 0` (zero slope exactly at 0,
    /// the f32 template's convention).
    Relu { n: usize },
}

/// Forward-order exact-f64 ops of a pure-chain plan, or `None` when a template
/// entry has no twin. The f32 plan's own forward accepts exactly these three
/// template kinds, so `None` here means a template the f32 plan refuses too.
fn f64_chain_ops(plan: &PointVjpBatchPlan) -> Option<Vec<F64ChainOp>> {
    let widen = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| f64::from(x)).collect() };
    let mut ops = Vec::with_capacity(plan.layers_backward.len());
    for idx in (0..plan.layers_backward.len()).rev() {
        let op = match &plan.layers_backward[idx] {
            GpuCrownLayer::Linear {
                weight,
                bias,
                out_features,
                in_features,
                ..
            } => {
                let (out_f, in_f) = (*out_features, *in_features);
                if weight.len() != out_f.checked_mul(in_f)? {
                    return None;
                }
                let w = Array2::from_shape_vec((out_f, in_f), widen(&weight[..])).ok()?;
                let b = match bias {
                    Some(b) if b.len() == out_f => Some(widen(&b[..])),
                    Some(_) => return None,
                    None => None,
                };
                F64ChainOp::Linear { w, b }
            }
            GpuCrownLayer::Conv2d {
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
            } => {
                let geom = F64ConvGeom {
                    oc: *out_channels,
                    ic: *in_channels,
                    kh: *kernel_h,
                    kw: *kernel_w,
                    sh: *stride_h,
                    sw: *stride_w,
                    ph: *pad_h,
                    pw: *pad_w,
                    oh: *out_h,
                    ow: *out_w,
                    ih: *in_h,
                    iw: *in_w,
                };
                let kcols = geom.ic.checked_mul(geom.kh)?.checked_mul(geom.kw)?;
                if weight_col.len() != geom.oc.checked_mul(kcols)? {
                    return None;
                }
                let w_col =
                    Array2::from_shape_vec((geom.oc, kcols), widen(&weight_col[..])).ok()?;
                // Mirror the f32 template exactly: it applies the expanded bias
                // only when its length matches the output.
                let out_len = geom.oc.checked_mul(geom.oh)?.checked_mul(geom.ow)?;
                let bias = bias_expanded
                    .as_ref()
                    .filter(|b| b.len() == out_len)
                    .map(|b| widen(&b[..]));
                F64ChainOp::Conv2d {
                    w_col,
                    bias,
                    geom: Box::new(geom),
                }
            }
            GpuCrownLayer::Activation {
                lower_slope,
                upper_slope,
                lower_intercept,
                upper_intercept,
                num_neurons,
            } => {
                if plan.mask_positions.contains(&idx) {
                    F64ChainOp::Relu { n: *num_neurons }
                } else {
                    if lower_slope != upper_slope
                        || lower_intercept != upper_intercept
                        || lower_slope.len() != *num_neurons
                        || lower_intercept.len() != *num_neurons
                    {
                        return None;
                    }
                    F64ChainOp::Affine {
                        slope: widen(&lower_slope[..]),
                        intercept: widen(&lower_intercept[..]),
                    }
                }
            }
            _ => return None,
        };
        ops.push(op);
    }
    Some(ops)
}

/// Visit every in-bounds im2col tap `(row, spatial, source)` of `g` — the same
/// index map the f32 template's `conv2d_forward` builds, shared by the twin's
/// forward (gather) and VJP (scatter-add), so the two are exact adjoints.
fn f64_conv_for_each_tap(g: &F64ConvGeom, mut visit: impl FnMut(usize, usize, usize)) {
    for c in 0..g.ic {
        for ky in 0..g.kh {
            for kx in 0..g.kw {
                let row = (c * g.kh + ky) * g.kw + kx;
                for oy in 0..g.oh {
                    let iy = (oy * g.sh + ky) as isize - g.ph as isize;
                    if iy < 0 || iy >= g.ih as isize {
                        continue;
                    }
                    for ox in 0..g.ow {
                        let ix = (ox * g.sw + kx) as isize - g.pw as isize;
                        if ix < 0 || ix >= g.iw as isize {
                            continue;
                        }
                        visit(
                            row,
                            oy * g.ow + ox,
                            (c * g.ih + iy as usize) * g.iw + ix as usize,
                        );
                    }
                }
            }
        }
    }
}

fn f64_conv_forward(
    w_col: &Array2<f64>,
    bias: Option<&[f64]>,
    g: &F64ConvGeom,
    x: &[f64],
) -> Option<Vec<f64>> {
    if x.len() != g.ic * g.ih * g.iw {
        return None;
    }
    let kcols = g.ic * g.kh * g.kw;
    let spatial = g.oh * g.ow;
    let mut col = Array2::<f64>::zeros((kcols, spatial));
    f64_conv_for_each_tap(g, |row, sp, src| col[[row, sp]] = x[src]);
    // (oc, kcols) @ (kcols, oh*ow) -> (oc, oh*ow), row-major == flat (oc, oh, ow).
    let y = w_col.dot(&col);
    let mut out: Vec<f64> = y.iter().copied().collect();
    if let Some(b) = bias {
        for (yi, bi) in out.iter_mut().zip(b) {
            *yi += bi;
        }
    }
    Some(out)
}

fn f64_conv_vjp(w_col: &Array2<f64>, g: &F64ConvGeom, grad_out: &[f64]) -> Option<Vec<f64>> {
    let spatial = g.oh * g.ow;
    if grad_out.len() != g.oc * spatial {
        return None;
    }
    let gy = ndarray::ArrayView2::from_shape((g.oc, spatial), grad_out).ok()?;
    // (kcols, oc) @ (oc, oh*ow) -> (kcols, oh*ow), then col2im scatter-add.
    let gcol = w_col.t().dot(&gy);
    let mut gx = vec![0.0f64; g.ic * g.ih * g.iw];
    f64_conv_for_each_tap(g, |row, sp, src| gx[src] += gcol[[row, sp]]);
    Some(gx)
}

/// Exact-f64 forward over the chain twin: `(output, relu masks in forward order)`.
fn f64_chain_forward(ops: &[F64ChainOp], x: &[f64]) -> Option<(Vec<f64>, Vec<Vec<bool>>)> {
    let mut buf = x.to_vec();
    let mut masks = Vec::new();
    for op in ops {
        buf = match op {
            F64ChainOp::Linear { w, b } => {
                if buf.len() != w.ncols() {
                    return None;
                }
                let mut y = w.dot(&ArrayView1::from(&buf[..])).to_vec();
                if let Some(b) = b {
                    for (yi, bi) in y.iter_mut().zip(b) {
                        *yi += bi;
                    }
                }
                y
            }
            F64ChainOp::Conv2d { w_col, bias, geom } => {
                f64_conv_forward(w_col, bias.as_deref(), geom, &buf)?
            }
            F64ChainOp::Affine { slope, intercept } => {
                if buf.len() != slope.len() {
                    return None;
                }
                buf.iter()
                    .zip(slope.iter().zip(intercept))
                    .map(|(&v, (&s, &t))| v * s + t)
                    .collect()
            }
            F64ChainOp::Relu { n } => {
                if buf.len() != *n {
                    return None;
                }
                masks.push(buf.iter().map(|&v| v > 0.0).collect());
                buf.iter().map(|&v| v.max(0.0)).collect()
            }
        };
    }
    Some((buf, masks))
}

/// Exact-f64 VJP over the chain twin: `d(s . y)/dx` at the taped point.
fn f64_chain_vjp(ops: &[F64ChainOp], masks: &[Vec<bool>], s: &[f64]) -> Option<Vec<f64>> {
    let mut g = s.to_vec();
    let mut remaining = masks.len();
    for op in ops.iter().rev() {
        g = match op {
            F64ChainOp::Linear { w, .. } => {
                if g.len() != w.nrows() {
                    return None;
                }
                w.t().dot(&ArrayView1::from(&g[..])).to_vec()
            }
            F64ChainOp::Conv2d { w_col, geom, .. } => f64_conv_vjp(w_col, geom, &g)?,
            F64ChainOp::Affine { slope, .. } => {
                if g.len() != slope.len() {
                    return None;
                }
                g.iter().zip(slope).map(|(&gi, &s)| gi * s).collect()
            }
            F64ChainOp::Relu { n } => {
                remaining = remaining.checked_sub(1)?;
                let mask = masks.get(remaining)?;
                if mask.len() != *n || g.len() != *n {
                    return None;
                }
                g.iter()
                    .zip(mask)
                    .map(|(&gi, &on)| if on { gi } else { 0.0 })
                    .collect()
            }
        };
    }
    (remaining == 0).then_some(g)
}

/// One side branch in exact f64: `y = W ReLU(A x + c) + b`, stored sparse.
#[derive(Debug, Clone)]
struct SideBranchF64 {
    a_rows: Vec<Vec<(u32, f64)>>,
    c: Vec<f64>,
    w_cols: Vec<Vec<(u32, f64)>>,
    b: Vec<f64>,
}

impl SideBranchF64 {
    /// Add `W ReLU(A x + c) + b` into `out`; returns the mask of `A x + c > 0`.
    fn forward_into(&self, x: &[f64], out: &mut [f64]) -> Vec<bool> {
        for (o, &b) in out.iter_mut().zip(&self.b) {
            *o += b;
        }
        let mut mask = Vec::with_capacity(self.c.len());
        for (i, (row, &c)) in self.a_rows.iter().zip(&self.c).enumerate() {
            let z = c + row.iter().map(|&(k, a)| a * x[k as usize]).sum::<f64>();
            let on = z > 0.0;
            if on {
                for &(j, w) in &self.w_cols[i] {
                    out[j as usize] += w * z;
                }
            }
            mask.push(on);
        }
        mask
    }

    /// Add `A^T (mask * W^T s)` into `grad`.
    fn vjp_into(&self, mask: &[bool], s: &[f64], grad: &mut [f64]) {
        for (i, &on) in mask.iter().enumerate() {
            if !on {
                continue;
            }
            let t: f64 = self.w_cols[i].iter().map(|&(j, w)| w * s[j as usize]).sum();
            if t == 0.0 {
                continue;
            }
            for &(k, a) in &self.a_rows[i] {
                grad[k as usize] += t * a;
            }
        }
    }
}

/// `(M row-major, m)` with `g(x) = M x + m`, captured through the sound f64
/// cell walk at 0 and at every unit vector (the midpoint of a point-input
/// enclosure, which is ULP-tight). `None` when the chain is outside the cell's
/// op set, the walk fails, or `deadline` passes — the caller then keeps the
/// widened f32 capture.
fn affine_matrix_f64(
    g: &GraphNetwork,
    in_shape: &[usize],
    deadline: Option<Instant>,
) -> Option<(Vec<f64>, Vec<f64>)> {
    if !g.supports_ibp_f64_cell() {
        return None;
    }
    let n: usize = in_shape.iter().product();
    let eval = |x: Vec<f64>| -> Option<Vec<f64>> {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return None;
        }
        let arr = ArrayD::from_shape_vec(IxDyn(in_shape), x).ok()?;
        let out = g.propagate_ibp_f64_cell(&Interval64::point(arr)).ok()?;
        let v: Vec<f64> = out
            .lower
            .iter()
            .zip(out.upper.iter())
            .map(|(&lo, &hi)| 0.5 * lo + 0.5 * hi)
            .collect();
        v.iter().all(|x| x.is_finite()).then_some(v)
    };
    let m = eval(vec![0.0; n])?;
    let rows = m.len();
    let mut dense = vec![0.0f64; rows.checked_mul(n)?];
    for k in 0..n {
        let mut e = vec![0.0f64; n];
        e[k] = 1.0;
        let col = eval(e)?;
        if col.len() != rows {
            return None;
        }
        for r in 0..rows {
            dense[r * n + k] = col[r] - m[r];
        }
    }
    Some((dense, m))
}

/// Whether the f32 unit-vector capture of a branch's `post` map is EXACT, so
/// widening it loses nothing: the chain holds exactly one `Linear` plus pure
/// data movement (`Reshape`/`Flatten`/`Transpose`), and the captured bias
/// `b = g(0)` is exactly zero. Each captured entry is then `W e_k` — one exact
/// product plus exact zeros — with no `+ b` to round. A biased post `Linear`
/// is NOT exact: the capture stores `fl(fl(W + b) - b)`, off by up to half an
/// f32 ULP of `W + b` (#sbench-f64-attack review).
fn f32_post_capture_is_exact(graph: &GraphNetwork, br: &SideBranch) -> bool {
    if br.b.iter().any(|&v| v != 0.0) {
        return false;
    }
    let mut weight_ops = 0usize;
    for name in &br.post_nodes {
        match graph.node(name).map(|node| &node.layer) {
            Some(Layer::Linear(_)) => weight_ops += 1,
            Some(Layer::Reshape(_) | Layer::Flatten(_) | Layer::Transpose(_)) => {}
            _ => return false,
        }
    }
    weight_ops == 1
}

/// The f64 twin of one side branch.
///
/// * `A`/`c` are re-captured through the exact f64 cell when the branch has a
///   `pre` chain and `graph` is supplied (the f32 capture rounds each
///   unit-vector evaluation, ~1e-7 relative — the same order as the margins
///   this twin exists to resolve); otherwise the f32 capture is widened.
/// * `W`/`b` are widened when the f32 capture is provably exact (an identity
///   post map, or [`f32_post_capture_is_exact`] — e.g. a bias-free single
///   `Linear` carrier), and otherwise re-captured through the f64 cell the
///   same way; a failed re-capture keeps the widened (inexact) capture.
///
/// Returns `(twin, exact)`, `exact` meaning BOTH maps are exact f64.
fn side_branch_f64(
    graph: Option<&GraphNetwork>,
    br: &SideBranch,
    input_shape: &[usize],
    input_dim: usize,
    deadline: Option<Instant>,
) -> (SideBranchF64, bool) {
    let widen_sparse = |v: &[Vec<(u32, f32)>]| -> Vec<Vec<(u32, f64)>> {
        v.iter()
            .map(|r| r.iter().map(|&(k, a)| (k, f64::from(a))).collect())
            .collect()
    };
    let widen = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| f64::from(x)).collect() };
    let hidden = br.c.len();
    let output_dim = br.b.len();

    // Pre map `A x + c` (input -> ReLU input).
    let (a_rows, c, pre_exact) = if br.pre_nodes.is_empty() {
        // Identity pre map: the capture is exactly I and 0 already.
        (widen_sparse(&br.a_rows), widen(&br.c), true)
    } else {
        let recaptured = graph
            .and_then(|g| g.chain_graph(&br.pre_nodes, false))
            .and_then(|g| affine_matrix_f64(&g, input_shape, deadline))
            .filter(|(a, c)| c.len() == hidden && a.len() == hidden * input_dim);
        match recaptured {
            Some((a_dense, c)) => {
                let a_rows: Vec<Vec<(u32, f64)>> = (0..hidden)
                    .map(|i| {
                        (0..input_dim)
                            .filter_map(|k| {
                                let v = a_dense[i * input_dim + k];
                                (v != 0.0).then_some((k as u32, v))
                            })
                            .collect()
                    })
                    .collect();
                (a_rows, c, true)
            }
            None => (widen_sparse(&br.a_rows), widen(&br.c), false),
        }
    };

    // Post map `W r + b` (ReLU output -> addend).
    let (w_cols, b, post_exact) =
        if br.post_nodes.is_empty() || graph.is_some_and(|g| f32_post_capture_is_exact(g, br)) {
            (widen_sparse(&br.w_cols), widen(&br.b), true)
        } else {
            let recaptured = graph
                .and_then(|g| g.chain_graph(&br.post_nodes, true))
                .and_then(|g| affine_matrix_f64(&g, &br.relu_shape, deadline))
                .filter(|(w, b)| b.len() == output_dim && w.len() == output_dim * hidden);
            match recaptured {
                Some((w_dense, b)) => {
                    // Row-major dense `W` is out x hidden; store it by column.
                    let w_cols: Vec<Vec<(u32, f64)>> = (0..hidden)
                        .map(|i| {
                            (0..output_dim)
                                .filter_map(|j| {
                                    let v = w_dense[j * hidden + i];
                                    (v != 0.0).then_some((j as u32, v))
                                })
                                .collect()
                        })
                        .collect();
                    (w_cols, b, true)
                }
                None => (widen_sparse(&br.w_cols), widen(&br.b), false),
            }
        };

    (
        SideBranchF64 {
            a_rows,
            c,
            w_cols,
            b,
        },
        pre_exact && post_exact,
    )
}

/// Exact-f64 twin of the batched point-VJP plans (#sbench-f64-attack): a pure
/// chain, or a chain main network plus additive side branches. The resnet
/// template has no twin (`build` returns `None`).
///
/// ATTACK-ONLY. Its outputs rank candidate points and its VJP steers a
/// search; neither is a bound, and nothing it returns can reach a verdict
/// except through the wrapper's unchanged trusted ORT + true-f64 gate.
#[derive(Debug, Clone)]
pub struct PointVjpF64Plan {
    main_ops: Vec<F64ChainOp>,
    branches: Vec<SideBranchF64>,
    out_bias: Vec<f64>,
    input_dim: usize,
    output_dim: usize,
    kind: &'static str,
    exact_branch_maps: usize,
}

/// Per-point activation record consumed by [`PointVjpF64Plan::vjp`].
#[derive(Debug, Clone)]
pub struct PointVjpF64Tape {
    main_masks: Vec<Vec<bool>>,
    branch_masks: Vec<Vec<bool>>,
}

impl PointVjpF64Plan {
    /// Build the twin for `graph` over the attack box `input`: the side-branch
    /// decomposition when the output is a sum of addends (cheap structural
    /// refusal otherwise), else the pure chain. `deadline` bounds only the
    /// optional exact re-capture of branch maps (a lapse keeps the widened
    /// f32 capture), never the build's own validity.
    pub fn build(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Option<Self> {
        if let Some(side) = graph.build_point_vjp_side_branch_plan(input) {
            return Self::from_side_branch_plan(Some(graph), &side, deadline);
        }
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return None;
        }
        let chain = graph.build_point_vjp_batch_plan(input)?;
        Self::from_chain_plan(&chain)
    }

    /// The twin of a pure-chain plan.
    pub fn from_chain_plan(plan: &PointVjpBatchPlan) -> Option<Self> {
        let main_ops = f64_chain_ops(plan)?;
        Some(Self {
            main_ops,
            branches: Vec::new(),
            out_bias: vec![0.0; plan.output_dim],
            input_dim: plan.input_dim,
            output_dim: plan.output_dim,
            kind: "chain",
            exact_branch_maps: 0,
        })
    }

    /// The twin of a side-branch plan whose main network is a pure chain.
    pub(crate) fn from_side_branch_plan(
        graph: Option<&GraphNetwork>,
        plan: &PointVjpSideBranchPlan,
        deadline: Option<Instant>,
    ) -> Option<Self> {
        let PointVjpWavePlan::Chain(main) = &*plan.main else {
            return None;
        };
        if main.input_dim != plan.input_dim || main.output_dim != plan.output_dim {
            return None;
        }
        let main_ops = f64_chain_ops(main)?;
        let mut branches = Vec::with_capacity(plan.branches.len());
        let mut exact_branch_maps = 0usize;
        for br in &plan.branches {
            let (twin, exact) =
                side_branch_f64(graph, br, &plan.input_shape, plan.input_dim, deadline);
            exact_branch_maps += usize::from(exact);
            branches.push(twin);
        }
        Some(Self {
            main_ops,
            branches,
            out_bias: plan.out_bias.iter().map(|&v| f64::from(v)).collect(),
            input_dim: plan.input_dim,
            output_dim: plan.output_dim,
            kind: "side",
            exact_branch_maps,
        })
    }

    /// Flattened input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Flattened output dimension.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// `"chain"` or `"side"` (telemetry).
    pub fn kind(&self) -> &'static str {
        self.kind
    }

    /// Number of side branches.
    pub fn branch_count(&self) -> usize {
        self.branches.len()
    }

    /// Side branches whose `A`/`c` AND `W`/`b` are both exact f64 (identity
    /// maps and provably exact f32 post captures included).
    pub fn exact_branch_maps(&self) -> usize {
        self.exact_branch_maps
    }

    /// Exact-f64 forward at `x` (flat, `input_dim`): the output and the tape
    /// the VJP needs. `None` on a shape mismatch or a non-finite value.
    pub fn forward(&self, x: &[f64]) -> Option<(Vec<f64>, PointVjpF64Tape)> {
        if x.len() != self.input_dim || !x.iter().all(|v| v.is_finite()) {
            return None;
        }
        let (mut out, main_masks) = f64_chain_forward(&self.main_ops, x)?;
        if out.len() != self.output_dim {
            return None;
        }
        for (o, &b) in out.iter_mut().zip(&self.out_bias) {
            *o += b;
        }
        let branch_masks: Vec<Vec<bool>> = self
            .branches
            .iter()
            .map(|br| br.forward_into(x, &mut out))
            .collect();
        out.iter().all(|v| v.is_finite()).then_some((
            out,
            PointVjpF64Tape {
                main_masks,
                branch_masks,
            },
        ))
    }

    /// Exact-f64 `d(s . y)/dx` at the point `tape` was recorded at.
    pub fn vjp(&self, tape: &PointVjpF64Tape, s: &[f64]) -> Option<Vec<f64>> {
        if s.len() != self.output_dim || tape.branch_masks.len() != self.branches.len() {
            return None;
        }
        let mut grad = f64_chain_vjp(&self.main_ops, &tape.main_masks, s)?;
        if grad.len() != self.input_dim {
            return None;
        }
        for (br, mask) in self.branches.iter().zip(&tape.branch_masks) {
            if mask.len() != br.c.len() {
                return None;
            }
            br.vjp_into(mask, s, &mut grad);
        }
        grad.iter().all(|v| v.is_finite()).then_some(grad)
    }

    /// [`Self::vjp`] for several output rows at one point (rayon-parallel);
    /// `None` if any row fails.
    pub fn vjp_rows(&self, tape: &PointVjpF64Tape, rows: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
        rows.par_iter().map(|s| self.vjp(tape, s)).collect()
    }
}

#[cfg(test)]
#[path = "point_vjp_batched_side_tests.rs"]
mod tests;
