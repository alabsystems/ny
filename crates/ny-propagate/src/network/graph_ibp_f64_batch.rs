// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched MULTI-BOX sound f64 interval forward over a [`GraphNetwork`]
//! (#f64-batch-boxes).
//!
//! # Why
//!
//! The box-refinement screen (nn4sys mscn, #f64-leaf) bounds each clause box
//! with one [`GraphNetwork::propagate_ibp_f64_cell`] walk. At mscn's
//! per-box shapes every Linear is a THIN interval GEMV (left rows m in
//! {1, 2, 3, 6}) against a huge shared constant weight (k = n = 2048), so the
//! fast Rump midpoint-radius kernel (#f64-blas-gemm) never fires — it is
//! gated to m >= 16 where it wins 5-198x, and MEASURED to lose below that.
//! The screen processes waves of up to 256 boxes: stacking a wave's boxes as
//! extra leading rows turns each Linear into ONE FAT interval GEMM with
//! m = W * rows >= 16, exactly the regime the kernel was built for.
//!
//! # Design: per-box ops, stacked Linear
//!
//! [`GraphNetwork::propagate_ibp_f64_cells`] walks the DAG once for W boxes,
//! holding W per-box [`Interval64`] values per live node:
//! - every op EXCEPT Linear evaluates per box through the exact same
//!   [`eval_node`] the single-box walk uses (parallel across boxes) —
//!   BIT-IDENTICAL per box to W independent walks by construction;
//! - Linear stacks the W per-box inputs (each `[rows.., in]`, identical
//!   shapes — boxes share the network input shape) into one
//!   `[W * rows, in]` interval matrix, runs the SAME
//!   [`eval_linear_with_bias`] entry the single-box walk uses, and splits
//!   the result back per box.
//!
//! # Soundness / per-box isolation
//!
//! A row of a Linear output depends ONLY on the matching row of its input
//! (both the scalar Higham loop and the Rump kernel compute each output
//! entry from its own left row and the shared weight column), so stacking
//! rows of independent boxes mixes NOTHING across boxes: box b's result is a
//! function of box b's input alone. The only behavioral difference vs W
//! independent walks is KERNEL SELECTION inside `eval_linear_with_bias`:
//! the stacked left operand can cross the fast-kernel row gate where the
//! per-box shapes could not. Both kernels are sound enclosures, and the fast
//! interval additionally CONTAINS the scalar one per entry (gate a1 in
//! `graph_ibp_f64_gemm`), so per box the batched result equals the single-box
//! result whenever kernel selection agrees and is a (still sound, slightly
//! wider) superset otherwise. The unit gates below assert both: bit-equality
//! at same-kernel shapes, containment at fat shapes, and a cross-box
//! contamination test (mutating one box leaves the others bit-identical).
//!
//! Fail-closed: ANY per-box op error (unsupported op, Div straddling zero
//! for one box, shape surprise) fails the WHOLE batched walk; the caller
//! must fall back to per-box evaluation — which reproduces current behavior
//! byte-for-byte and lets the healthy boxes proceed.
//!
//! Kill-switch: callers gate on [`batch_boxes_enabled`]
//! (`NY_F64_BATCH_BOXES=0` disables the batched lane; batteries-included
//! default-ON).

use std::collections::HashMap;

use ndarray::{Array2, ArrayD, Ix2, IxDyn};
use ny_core::{NyError, Result};
use rayon::prelude::*;

use crate::layers::Layer;

use super::core::graph::{GraphNetwork, NETWORK_INPUT};
use super::graph_ibp_f64_cell::{eval_linear_with_bias, eval_node, Interval64};
use super::graph_ibp_f64_gemm::{
    abs_colmax, fast_gemm_enabled, rump_interval_matmul_point_b, FAST_GEMM_MIN_ROWS,
    FAST_GEMM_MIN_VOLUME,
};

/// Batteries-included default-ON; `NY_F64_BATCH_BOXES=0` is the kill-switch
/// that restores per-box evaluation everywhere.
pub fn batch_boxes_enabled() -> bool {
    !std::env::var("NY_F64_BATCH_BOXES").is_ok_and(|v| v == "0")
}

/// A Linear node's weight prepared ONCE for the Rump point-B kernel: the
/// exact f64 transpose `Wᵀ` (every f32 is exactly representable) and its
/// elementwise magnitude `|Wᵀ|`. Rebuilding these per call — plus the
/// kernel's own midpoint/radius split of the weight — was measured to
/// DOMINATE thin-m batched calls on the 2048-wide mscn Linears (the
/// kernel's documented ~39ms flat floor); with the prepared pair the
/// per-call cost is the GEMMs themselves.
pub(super) struct PreparedPointWeight {
    /// `Wᵀ` as exact f64, shape `[in, out]`.
    wt: Array2<f64>,
    /// `|Wᵀ|`, shape `[in, out]`.
    wt_abs: Array2<f64>,
    /// Exact per-column maxima of `|Wᵀ|` (length `out`) — the `cmax` factor
    /// of the kernel's rank-1 magnitude bound (#rank1-radius).
    wt_colmax: Vec<f64>,
}

/// Per-graph cache of prepared f64 Linear weights, keyed by node name.
/// Build once per screen run ([`GraphNetwork::build_f64_weight_cache`]) and
/// pass to the batched walks; the same node's weight then converts exactly
/// once. Purely a SPEED cache: the kernel results are bit-identical with
/// and without it (see `rump_interval_matmul_point_b`).
pub struct F64WeightCache {
    prepared: HashMap<String, PreparedPointWeight>,
}

impl F64WeightCache {
    /// The prepared weight of Linear node `name`, if cached.
    pub(super) fn get(&self, name: &str) -> Option<&PreparedPointWeight> {
        self.prepared.get(name)
    }
}

/// Minimum parameter count of the LARGEST output-ancestor Linear for the
/// batched multi-box lane to be worthwhile
/// ([`GraphNetwork::f64_batch_worthwhile`]). MEASURED on the nn4sys mscn
/// screens (M4 Max, release): at 2048-wide weights (4.2M-12.6M params per
/// Linear) batching the wave into fat Rump GEMMs cut the
/// cardinality_1_1_2048_dual screen 21.9s -> 13.4s; at 128-wide weights
/// (<= 16K params) the per-box scalar loops are so cheap that the batched
/// walk's serial per-node structure LOSES to the wave's coarse per-box
/// rayon parallelism (cardinality_1_240_128_dual regressed from unsat
/// inside 20s to unknown). 2^20 sits between the bands with a wide margin
/// on both sides.
const BATCH_MIN_LINEAR_PARAMS: usize = 1 << 20;

impl GraphNetwork {
    /// Whether the batched multi-box f64 lane is worth using for this graph:
    /// true iff some output-ancestor Linear is FAT enough
    /// ([`BATCH_MIN_LINEAR_PARAMS`]) that stacked interval GEMMs beat
    /// parallel per-box scalar loops. Cheap nets keep the per-box lane —
    /// bit-identical behavior to no batching at all.
    pub fn f64_batch_worthwhile(&self) -> bool {
        match self.output_ancestors() {
            Ok(needed) => needed.iter().any(|name| {
                self.node(name).is_some_and(|node| match node.layer() {
                    Layer::Linear(linear) => {
                        let (out_dim, in_dim) = linear.weight.dim();
                        out_dim.saturating_mul(in_dim) >= BATCH_MIN_LINEAR_PARAMS
                    }
                    _ => false,
                })
            }),
            Err(_) => false,
        }
    }

    /// Prepare every output-ancestor Linear node's weight for the batched
    /// f64 walks (parallel across nodes — the 37.8M-param mscn dual costs
    /// ~2 f64 copies + a transposed traversal per weight, real time on the
    /// screen's first wave otherwise). Memory: 2 f64 copies of each Linear
    /// weight (16 bytes per parameter) — bounded by the model size and
    /// freed with the cache.
    pub fn build_f64_weight_cache(&self) -> F64WeightCache {
        let names: Vec<&str> = match self.output_ancestors() {
            Ok(needed) => needed.into_iter().collect(),
            Err(_) => Vec::new(),
        };
        let prepared: HashMap<String, PreparedPointWeight> = names
            .par_iter()
            .filter_map(|name| {
                let node = self.node(name)?;
                match node.layer() {
                    Layer::Linear(linear) => {
                        let wt = linear.weight.t().map(|&w| f64::from(w));
                        let wt_abs = wt.mapv(f64::abs);
                        let wt_colmax = abs_colmax(&wt_abs.view());
                        Some((
                            node.name().to_string(),
                            PreparedPointWeight {
                                wt,
                                wt_abs,
                                wt_colmax,
                            },
                        ))
                    }
                    _ => None,
                }
            })
            .collect();
        F64WeightCache { prepared }
    }
}

impl GraphNetwork {
    /// Sound f64 interval forward for W INDEPENDENT input boxes in one DAG
    /// walk (module docs: per-box ops, stacked fat-GEMM Linear). Returns one
    /// output interval per input box, in order.
    ///
    /// Every box's result is a sound enclosure depending only on its own
    /// box; it is bit-identical to `propagate_ibp_f64_cell(&inputs[b])`
    /// whenever the stacked Linear takes the same kernel, and a containing
    /// superset when the stacked shape promotes the fast Rump kernel.
    ///
    /// Errors fail the WHOLE batch (fail-closed): callers must fall back to
    /// per-box walks, which also isolates any single poisoned box (e.g. a
    /// Div divisor straddling zero for that box only).
    pub fn propagate_ibp_f64_cells(&self, inputs: &[Interval64]) -> Result<Vec<Interval64>> {
        self.propagate_ibp_f64_cells_cached(inputs, None)
    }

    /// [`Self::propagate_ibp_f64_cells`] with an optional prepared-weight
    /// cache ([`Self::build_f64_weight_cache`]) — bit-identical results,
    /// skips the per-call f64 weight conversion/split.
    pub fn propagate_ibp_f64_cells_cached(
        &self,
        inputs: &[Interval64],
        weights: Option<&F64WeightCache>,
    ) -> Result<Vec<Interval64>> {
        let w = inputs.len();
        if w == 0 {
            return Ok(Vec::new());
        }
        let in_shape = inputs[0].lower.shape().to_vec();
        for x in inputs {
            if x.lower.shape() != in_shape.as_slice() || x.upper.shape() != in_shape.as_slice() {
                return Err(NyError::InvalidSpec(
                    "f64 batch eval: input boxes must share one shape".to_string(),
                ));
            }
        }

        let needed = self.output_ancestors()?;

        // Consumer refcounts (with multiplicity) for cache eviction: W
        // per-box tensors per live node is the whole memory cost of the
        // batched walk, so values are dropped as soon as their last consumer
        // has run. The output node gets one extra count so it survives.
        let mut remaining: HashMap<&str, usize> = HashMap::new();
        for name in &needed {
            let node = self.node(name).ok_or_else(|| {
                NyError::InvalidSpec(format!("f64 batch eval: missing node '{name}'"))
            })?;
            for input in node.inputs() {
                if input != NETWORK_INPUT {
                    *remaining.entry(input.as_str()).or_insert(0) += 1;
                }
            }
        }
        *remaining.entry(self.output_name()).or_insert(0) += 1;

        let mut cache: HashMap<&str, Vec<Interval64>> = HashMap::new();
        for node_name in self.exec_order()? {
            if !needed.contains(node_name.as_str()) {
                continue;
            }
            let node = self.node(node_name).ok_or_else(|| {
                NyError::InvalidSpec(format!("f64 batch eval: missing node '{node_name}'"))
            })?;

            let outs: Vec<Interval64> = match node.layer() {
                // Fat-GEMM lane: stack the W per-box inputs as leading rows
                // of ONE Linear evaluation (row-independent, module docs).
                Layer::Linear(linear) => {
                    let input_name = node.inputs().first().ok_or_else(|| {
                        NyError::InvalidSpec("f64 batch eval: Linear missing its input".to_string())
                    })?;
                    let xs: &[Interval64] = if input_name == NETWORK_INPUT {
                        inputs
                    } else {
                        cache
                            .get(input_name.as_str())
                            .map(Vec::as_slice)
                            .ok_or_else(|| {
                                NyError::InvalidSpec(format!(
                                    "f64 batch eval: '{input_name}' not computed"
                                ))
                            })?
                    };
                    let prepared = weights.and_then(|c| c.get(node_name.as_str()));
                    eval_linear_stacked_prepared(linear, xs, true, prepared)?
                }
                // Everything else: the EXACT single-box rules, per box
                // (parallel across boxes — read-only view of the cache).
                _ => {
                    let cache_ref = &cache;
                    (0..w)
                        .into_par_iter()
                        .map(|b| {
                            let resolve = |name: &str| -> Result<Interval64> {
                                resolve_box(inputs, cache_ref, name, b)
                            };
                            eval_node(node.layer(), node, &resolve)
                        })
                        .collect::<Result<Vec<Interval64>>>()?
                }
            };
            if outs.len() != w {
                return Err(NyError::InvalidSpec(format!(
                    "f64 batch eval: node '{node_name}' produced {} results for {w} boxes",
                    outs.len()
                )));
            }
            // Per-box independence invariant: all boxes share every node
            // shape (they share the input shape and the DAG is shape-static
            // per walk); a divergence would mean box-dependent shapes, which
            // the stacked Linear lane must never see.
            if outs
                .iter()
                .any(|o| o.lower.shape() != outs[0].lower.shape())
            {
                return Err(NyError::InvalidSpec(format!(
                    "f64 batch eval: node '{node_name}' shapes diverged across boxes"
                )));
            }

            // Evict inputs whose last consumer just ran.
            for input in node.inputs() {
                if input == NETWORK_INPUT {
                    continue;
                }
                if let Some(count) = remaining.get_mut(input.as_str()) {
                    *count -= 1;
                    if *count == 0 {
                        cache.remove(input.as_str());
                    }
                }
            }
            cache.insert(node.name(), outs);
        }

        cache.remove(self.output_name()).ok_or_else(|| {
            NyError::InvalidSpec("f64 batch eval: output node not computed".to_string())
        })
    }
}

/// Fast stacked-Linear evaluation against a PREPARED point weight: the same
/// row/volume gates as the standard fast path (`try_eval_linear_fast` in
/// the cell module), the Rump point-B kernel (bit-identical to the standard
/// fast path's kernel — see `rump_interval_matmul_point_b`), and the
/// identical post-GEMM bias fold. `None` = below the gates, kill-switched,
/// shape surprise, or kernel decline — caller falls back to the standard
/// entry.
fn try_eval_linear_prepared(
    linear: &crate::layers::LinearLayer,
    stacked: &Interval64,
    include_bias: bool,
    p: &PreparedPointWeight,
) -> Option<Interval64> {
    if !fast_gemm_enabled() || stacked.lower.shape().len() != 2 {
        return None;
    }
    let (out_dim, in_dim) = linear.weight.dim();
    let m = stacked.lower.shape()[0];
    if stacked.lower.shape()[1] != in_dim
        || p.wt.dim() != (in_dim, out_dim)
        || m < FAST_GEMM_MIN_ROWS
        || m.checked_mul(out_dim)
            .and_then(|v| v.checked_mul(in_dim))
            .is_none_or(|v| v <= FAST_GEMM_MIN_VOLUME)
    {
        return None;
    }
    let a_lo = stacked.lower.view().into_dimensionality::<Ix2>().ok()?;
    let a_hi = stacked.upper.view().into_dimensionality::<Ix2>().ok()?;
    let (mut lo, mut hi) =
        rump_interval_matmul_point_b(a_lo, a_hi, p.wt.view(), p.wt_abs.view(), &p.wt_colmax)?;
    if include_bias {
        if let Some(bias) = linear.bias.as_ref() {
            for mut row_lo in lo.rows_mut() {
                for (l, &b) in row_lo.iter_mut().zip(bias.iter()) {
                    *l = (*l + f64::from(b)).next_down();
                }
            }
            for mut row_hi in hi.rows_mut() {
                for (h, &b) in row_hi.iter_mut().zip(bias.iter()) {
                    *h = (*h + f64::from(b)).next_up();
                }
            }
        }
    }
    Some(Interval64 {
        lower: lo.into_dyn(),
        upper: hi.into_dyn(),
    })
}

/// Resolve one box's value of `name` (network input or cached node output).
fn resolve_box(
    inputs: &[Interval64],
    cache: &HashMap<&str, Vec<Interval64>>,
    name: &str,
    b: usize,
) -> Result<Interval64> {
    if name == NETWORK_INPUT {
        return Ok(inputs[b].clone());
    }
    cache
        .get(name)
        .and_then(|v| v.get(b))
        .cloned()
        .ok_or_else(|| NyError::InvalidSpec(format!("f64 batch eval: '{name}' not computed")))
}

/// Stack W same-shaped per-box (or per-channel, #f64-mvf) inputs
/// `[rows.., in]` into one `[W * rows, in]` interval matrix, evaluate the
/// Linear ONCE through the standard entry (which picks the fast Rump kernel
/// at fat stacked shapes), and split the `[W * rows, out]` result back per
/// input.
///
/// Row-independence of Linear makes this exact per input: output row r
/// depends only on input row r and the shared constant weight/bias, so
/// input b's block of rows is precisely what an unstacked evaluation of
/// input b would feed the same kernel. `include_bias = false` serves the
/// mean-value derivative channels (#f64-mvf), which map through `W·d`
/// without the constant bias.
///
/// With a prepared weight ([`F64WeightCache`]): when the stacked shape
/// crosses the fast-kernel gates, the prepared pair goes straight to the
/// Rump point-B kernel — bit-identical to the unprepared fast path minus
/// its per-call weight conversion/split; every other case (thin stacks,
/// `NY_F64_BLAS=0`, kernel decline) takes the standard entry unchanged.
pub(super) fn eval_linear_stacked_prepared(
    linear: &crate::layers::LinearLayer,
    xs: &[Interval64],
    include_bias: bool,
    prepared: Option<&PreparedPointWeight>,
) -> Result<Vec<Interval64>> {
    let w = xs.len();
    let shape = xs[0].lower.shape().to_vec();
    if shape.is_empty() {
        return Err(NyError::UnsupportedOp(
            "f64 batch eval: Linear on rank-0 input".to_string(),
        ));
    }
    for x in xs {
        if x.lower.shape() != shape.as_slice() || x.upper.shape() != shape.as_slice() {
            return Err(NyError::InvalidSpec(
                "f64 batch eval: Linear box shapes diverged".to_string(),
            ));
        }
    }
    let rows: usize = shape[..shape.len() - 1].iter().product();
    let in_dim = shape[shape.len() - 1];
    let per_box = rows * in_dim;

    let mut lo = Vec::with_capacity(w * per_box);
    let mut hi = Vec::with_capacity(w * per_box);
    for x in xs {
        let l_std = x.lower.as_standard_layout();
        let h_std = x.upper.as_standard_layout();
        let (l, h) = match (l_std.as_slice(), h_std.as_slice()) {
            (Some(l), Some(h)) => (l, h),
            _ => {
                return Err(NyError::InvalidSpec(
                    "f64 batch eval: Linear input not contiguous".to_string(),
                ))
            }
        };
        lo.extend_from_slice(l);
        hi.extend_from_slice(h);
    }
    let stacked = Interval64 {
        lower: ArrayD::from_shape_vec(IxDyn(&[w * rows, in_dim]), lo)
            .map_err(|e| NyError::InvalidSpec(format!("f64 batch eval: stack: {e}")))?,
        upper: ArrayD::from_shape_vec(IxDyn(&[w * rows, in_dim]), hi)
            .map_err(|e| NyError::InvalidSpec(format!("f64 batch eval: stack: {e}")))?,
    };

    // Prepared point-B kernel when the stacked shape crosses the same gates
    // the standard entry's fast path uses; a kernel decline (`None`) falls
    // through to the standard entry, which reproduces the identical result
    // (or its scalar path below the gates / under NY_F64_BLAS=0).
    let out =
        match prepared.and_then(|p| try_eval_linear_prepared(linear, &stacked, include_bias, p)) {
            Some(out) => out,
            None => eval_linear_with_bias(linear, &stacked, include_bias)?,
        };

    let (out_dim, _) = linear.weight.dim();
    let mut out_shape = shape[..shape.len() - 1].to_vec();
    out_shape.push(out_dim);
    let out_per_box = rows * out_dim;
    let out_lo_std = out.lower.as_standard_layout();
    let out_hi_std = out.upper.as_standard_layout();
    let (out_lo, out_hi) = match (out_lo_std.as_slice(), out_hi_std.as_slice()) {
        (Some(l), Some(h)) => (l, h),
        _ => {
            return Err(NyError::InvalidSpec(
                "f64 batch eval: Linear output not contiguous".to_string(),
            ))
        }
    };
    if out_lo.len() != w * out_per_box {
        return Err(NyError::InvalidSpec(
            "f64 batch eval: Linear output size mismatch".to_string(),
        ));
    }

    (0..w)
        .map(|b| {
            let base = b * out_per_box;
            Ok(Interval64 {
                lower: ArrayD::from_shape_vec(
                    IxDyn(&out_shape),
                    out_lo[base..base + out_per_box].to_vec(),
                )
                .map_err(|e| NyError::InvalidSpec(format!("f64 batch eval: unstack: {e}")))?,
                upper: ArrayD::from_shape_vec(
                    IxDyn(&out_shape),
                    out_hi[base..base + out_per_box].to_vec(),
                )
                .map_err(|e| NyError::InvalidSpec(format!("f64 batch eval: unstack: {e}")))?,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Batching gates: per-box bit-equality at same-kernel shapes, containment +
// sampled soundness at fat (Rump-promoted) shapes, cross-box contamination.
// If ANY of these fails, the batched lane must be gated OFF.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::super::core::graph::GraphNode;
    use super::*;
    use crate::layers::{
        AddConstantLayer, DivLayer, LinearLayer, MulBinaryLayer, ReLULayer, ReduceSumLayer,
        SigmoidLayer, TanhLayer,
    };
    use ndarray::Array2;

    /// Deterministic xorshift stream — no extra dev-dep, reproducible seeds.
    struct Rng(u64);
    impl Rng {
        fn next_unit(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    const ROWS: usize = 4;
    const IN: usize = 16;
    const HID: usize = 16;

    /// mscn-shaped DAG at test scale: input [ROWS, IN]
    /// --Linear(IN->HID)--> ReLU --Linear(HID->IN)--> Mul(input)
    /// --ReduceSum--> Div(2.5 + ReduceSum(relu)) --> Sigmoid.
    /// Exercises the stacked-Linear lane (twice), elementwise, reduction,
    /// runtime-conditioned Div, and Sigmoid.
    fn build_test_graph(rng: &mut Rng) -> GraphNetwork {
        let mut w1 = Array2::<f32>::zeros((HID, IN));
        for v in w1.iter_mut() {
            *v = (rng.next_unit() * 2.0 - 1.0) as f32;
        }
        let mut b1 = ndarray::Array1::<f32>::zeros(HID);
        for v in b1.iter_mut() {
            *v = (rng.next_unit() * 2.0 - 1.0) as f32;
        }
        let mut w2 = Array2::<f32>::zeros((IN, HID));
        for v in w2.iter_mut() {
            *v = (rng.next_unit() * 2.0 - 1.0) as f32;
        }
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "lin1",
            Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["lin1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "lin2",
            Layer::Linear(LinearLayer::new(w2, None).unwrap()),
            vec!["relu".to_string()],
        ));
        g.add_node(GraphNode::new(
            "mul",
            Layer::MulBinary(MulBinaryLayer),
            vec!["lin2".to_string(), NETWORK_INPUT.to_string()],
        ));
        g.add_node(GraphNode::new(
            "sum",
            Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
            vec!["mul".to_string()],
        ));
        g.add_node(GraphNode::new(
            "relu_sum",
            Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
            vec!["relu".to_string()],
        ));
        g.add_node(GraphNode::new(
            "denom",
            Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(
                IxDyn(&[1]),
                2.5f32,
            ))),
            vec!["relu_sum".to_string()],
        ));
        g.add_node(GraphNode::new(
            "div",
            Layer::Div(DivLayer),
            vec!["sum".to_string(), "denom".to_string()],
        ));
        g.add_node(GraphNode::new(
            "out",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["div".to_string()],
        ));
        g.set_output("out");
        g
    }

    fn random_box(rng: &mut Rng, width: f64) -> Interval64 {
        let n = ROWS * IN;
        let mut lo = Vec::with_capacity(n);
        let mut hi = Vec::with_capacity(n);
        for _ in 0..n {
            let c = rng.next_unit() * 2.0 - 1.0;
            let w = rng.next_unit() * width;
            lo.push(c - w);
            hi.push(c + w);
        }
        Interval64 {
            lower: ArrayD::from_shape_vec(IxDyn(&[ROWS, IN]), lo).unwrap(),
            upper: ArrayD::from_shape_vec(IxDyn(&[ROWS, IN]), hi).unwrap(),
        }
    }

    fn bits(arr: &ArrayD<f64>) -> Vec<u64> {
        arr.iter().map(|v| v.to_bits()).collect()
    }

    /// GATE 1 (task test, "same kernel selection forced"): W = 3 boxes keep
    /// the stacked Linear m = 12 BELOW the fast-kernel row gate (16), so the
    /// batched walk uses the EXACT same scalar kernels as three independent
    /// per-box walks — every box's result must be BIT-IDENTICAL (mutual
    /// containment as equality).
    #[test]
    fn batched_equals_per_box_bitwise_when_kernels_agree() {
        let mut rng = Rng(0x00C0_FFEE_5EED_0001);
        let g = build_test_graph(&mut rng);
        let boxes: Vec<Interval64> = (0..3).map(|_| random_box(&mut rng, 0.25)).collect();
        // Compile-time gate: 3 boxes must stay below the fast-kernel row gate.
        const _: () = assert!(3 * ROWS < FAST_GEMM_MIN_ROWS);

        let batched = g.propagate_ibp_f64_cells(&boxes).expect("batched walk");
        for (b, x) in boxes.iter().enumerate() {
            let single = g.propagate_ibp_f64_cell(x).expect("per-box walk");
            assert_eq!(
                bits(&batched[b].lower),
                bits(&single.lower),
                "box {b}: batched lower diverged from per-box (same-kernel regime)"
            );
            assert_eq!(
                bits(&batched[b].upper),
                bits(&single.upper),
                "box {b}: batched upper diverged from per-box (same-kernel regime)"
            );
        }
    }

    /// GATE 2: W = 16 boxes promote the stacked Linears to the fast Rump
    /// kernel (m = 64 >= 16, volume > threshold). Per box, the batched
    /// result must CONTAIN the per-box result (the fast kernel's interval
    /// contains the scalar one, and every downstream op is
    /// inclusion-monotone) AND every sampled concrete forward.
    #[test]
    fn batched_fat_path_contains_per_box_and_samples() {
        let mut rng = Rng(0x00C0_FFEE_5EED_0002);
        let g = build_test_graph(&mut rng);
        let w = 40usize;
        assert!(w * ROWS >= FAST_GEMM_MIN_ROWS);
        assert!(w * ROWS * IN * HID > FAST_GEMM_MIN_VOLUME);
        let boxes: Vec<Interval64> = (0..w).map(|_| random_box(&mut rng, 0.1)).collect();

        let batched = g.propagate_ibp_f64_cells(&boxes).expect("batched walk");
        for (b, x) in boxes.iter().enumerate() {
            let single = g.propagate_ibp_f64_cell(x).expect("per-box walk");
            for ((bl, bu), (sl, su)) in batched[b]
                .lower
                .iter()
                .zip(batched[b].upper.iter())
                .zip(single.lower.iter().zip(single.upper.iter()))
            {
                assert!(
                    bl <= sl && bu >= su,
                    "box {b}: batched [{bl}, {bu}] does not contain per-box [{sl}, {su}]"
                );
            }
        }
        // Sampled soundness: concrete f64 forwards stay inside the batched
        // enclosure of their OWN box.
        for (b, x) in boxes.iter().enumerate() {
            for _ in 0..20 {
                let mut sample = x.lower.clone();
                for (s, (&l, &h)) in sample.iter_mut().zip(x.lower.iter().zip(x.upper.iter())) {
                    *s = l + (h - l) * rng.next_unit();
                }
                let point = Interval64::point(sample);
                // Point forward through the sound per-box walk: its interval
                // contains the true forward, so it must sit inside the
                // batched box enclosure.
                let y = g.propagate_ibp_f64_cell(&point).expect("point walk");
                for ((&yl, &yu), (&bl, &bu)) in y
                    .lower
                    .iter()
                    .zip(y.upper.iter())
                    .zip(batched[b].lower.iter().zip(batched[b].upper.iter()))
                {
                    // Bit-identical containment probe: f64::midpoint rounds differently past f64::MAX/2.
                    #[allow(clippy::manual_midpoint)]
                    let mid = 0.5 * (yl + yu);
                    assert!(
                        bl <= mid && mid <= bu,
                        "box {b}: sampled forward {mid} escapes batched [{bl}, {bu}]"
                    );
                }
            }
        }
    }

    /// GATE 3 (task test, cross-box contamination): mutate ONE box of a
    /// fat-path wave — every other box's batched result must be
    /// BYTE-IDENTICAL to before (a box's result depends only on its own
    /// input box), and the mutated box's must change.
    #[test]
    fn batched_mutating_one_box_leaves_others_bit_identical() {
        let mut rng = Rng(0x00C0_FFEE_5EED_0003);
        let g = build_test_graph(&mut rng);
        let w = 40usize;
        let mut boxes: Vec<Interval64> = (0..w).map(|_| random_box(&mut rng, 0.1)).collect();

        let before = g.propagate_ibp_f64_cells(&boxes).expect("batched walk");
        // Mutate box 1: widen every axis by 0.05 both ways.
        boxes[1] = Interval64 {
            lower: boxes[1].lower.mapv(|v| v - 0.05),
            upper: boxes[1].upper.mapv(|v| v + 0.05),
        };
        let after = g.propagate_ibp_f64_cells(&boxes).expect("batched walk");

        for b in 0..w {
            if b == 1 {
                assert_ne!(
                    bits(&before[b].lower),
                    bits(&after[b].lower),
                    "box 1 was widened — its result must change"
                );
                continue;
            }
            assert_eq!(
                bits(&before[b].lower),
                bits(&after[b].lower),
                "box {b}: lower changed when only box 1 was mutated — cross-box contamination"
            );
            assert_eq!(
                bits(&before[b].upper),
                bits(&after[b].upper),
                "box {b}: upper changed when only box 1 was mutated — cross-box contamination"
            );
        }
    }

    /// The prepared-weight cache is a pure SPEED cache: cached and uncached
    /// batched results must be BIT-IDENTICAL on a fat wave (the point-B
    /// kernel is bit-equal to the standard fast path, and everything else
    /// is untouched).
    #[test]
    fn prepared_weight_cache_is_bit_identical() {
        let mut rng = Rng(0x00C0_FFEE_5EED_0005);
        let g = build_test_graph(&mut rng);
        let boxes: Vec<Interval64> = (0..40).map(|_| random_box(&mut rng, 0.1)).collect();
        let cache = g.build_f64_weight_cache();
        let plain = g.propagate_ibp_f64_cells(&boxes).expect("plain");
        let cached = g
            .propagate_ibp_f64_cells_cached(&boxes, Some(&cache))
            .expect("cached");
        for b in 0..boxes.len() {
            assert_eq!(
                bits(&plain[b].lower),
                bits(&cached[b].lower),
                "box {b}: cached lower diverged"
            );
            assert_eq!(
                bits(&plain[b].upper),
                bits(&cached[b].upper),
                "box {b}: cached upper diverged"
            );
        }
    }

    /// Fail-closed: one poisoned box (Div divisor straddling zero) fails the
    /// WHOLE batch — the caller falls back per box, where the healthy boxes
    /// still succeed and the poisoned one still fails.
    #[test]
    fn batched_fails_closed_on_one_poisoned_box() {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::new(
            "d",
            Layer::Div(DivLayer),
            vec![NETWORK_INPUT.to_string(), NETWORK_INPUT.to_string()],
        ));
        g.set_output("d");
        let healthy = Interval64 {
            lower: ArrayD::from_elem(IxDyn(&[2]), 1.0),
            upper: ArrayD::from_elem(IxDyn(&[2]), 2.0),
        };
        let poisoned = Interval64 {
            lower: ArrayD::from_elem(IxDyn(&[2]), -1.0),
            upper: ArrayD::from_elem(IxDyn(&[2]), 1.0),
        };
        let boxes = vec![healthy.clone(), poisoned, healthy.clone()];
        assert!(g.propagate_ibp_f64_cells(&boxes).is_err());
        // Per-box fallback: healthy boxes still evaluate.
        assert!(g.propagate_ibp_f64_cell(&healthy).is_ok());
    }

    /// Fail-closed: unsupported op fails the batch; shape-mismatched boxes
    /// are rejected up front; the empty batch is trivially fine.
    #[test]
    fn batched_rejects_unsupported_and_mismatched_inputs() {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input("t", Layer::Tanh(TanhLayer)));
        g.set_output("t");
        let x = Interval64 {
            lower: ArrayD::from_elem(IxDyn(&[2]), 0.0),
            upper: ArrayD::from_elem(IxDyn(&[2]), 1.0),
        };
        assert!(g.propagate_ibp_f64_cells(&[x.clone(), x]).is_err());

        let mut rng = Rng(0x00C0_FFEE_5EED_0004);
        let g = build_test_graph(&mut rng);
        let good = random_box(&mut rng, 0.1);
        let bad = Interval64 {
            lower: ArrayD::from_elem(IxDyn(&[ROWS * IN]), 0.0),
            upper: ArrayD::from_elem(IxDyn(&[ROWS * IN]), 1.0),
        };
        assert!(g.propagate_ibp_f64_cells(&[good, bad]).is_err());
        assert!(g.propagate_ibp_f64_cells(&[]).unwrap().is_empty());
    }

    /// Kill-switch: NY_F64_BATCH_BOXES=0 gates the batched lane off for
    /// callers. (Serialized + restored via the blessed env choke point.)
    #[test]
    fn batch_boxes_kill_switch() {
        ny_test_utils::env::with_env_edits(|env| {
            env.set("NY_F64_BATCH_BOXES", "0");
            assert!(!batch_boxes_enabled());
            env.remove("NY_F64_BATCH_BOXES");
            assert!(batch_boxes_enabled());
        });
    }

    /// Timing probe (ignored; run with --ignored --nocapture): batched wave
    /// vs per-box loop at the REAL mscn_2048d shapes (rows 6, k = n = 2048
    /// Linears), W in {16, 64, 256}.
    #[test]
    #[ignore = "manual timing probe"]
    fn bench_batched_vs_per_box_mscn_shapes() {
        use std::time::Instant;
        let mut rng = Rng(0xB7E1_5162_8AED_2A6A);
        let (rows, k) = (6usize, 2048usize);
        let mut w1 = Array2::<f32>::zeros((k, k));
        for v in w1.iter_mut() {
            *v = (rng.next_unit() * 2.0 - 1.0) as f32;
        }
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "lin1",
            Layer::Linear(LinearLayer::new(w1.clone(), None).unwrap()),
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["lin1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "lin2",
            Layer::Linear(LinearLayer::new(w1, None).unwrap()),
            vec!["relu".to_string()],
        ));
        g.add_node(GraphNode::new(
            "sum",
            Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
            vec!["lin2".to_string()],
        ));
        g.add_node(GraphNode::new(
            "out",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["sum".to_string()],
        ));
        g.set_output("out");

        for &w in &[16usize, 64, 256] {
            let boxes: Vec<Interval64> = (0..w)
                .map(|_| {
                    let n = rows * k;
                    let mut lo = Vec::with_capacity(n);
                    let mut hi = Vec::with_capacity(n);
                    for _ in 0..n {
                        let c = rng.next_unit() * 2.0 - 1.0;
                        let r = rng.next_unit() * 1e-6;
                        lo.push(c - r);
                        hi.push(c + r);
                    }
                    Interval64 {
                        lower: ArrayD::from_shape_vec(IxDyn(&[rows, k]), lo).unwrap(),
                        upper: ArrayD::from_shape_vec(IxDyn(&[rows, k]), hi).unwrap(),
                    }
                })
                .collect();

            let t0 = Instant::now();
            let batched = g.propagate_ibp_f64_cells(&boxes).unwrap();
            let batched_t = t0.elapsed();
            std::hint::black_box(batched);

            let t1 = Instant::now();
            let per_box: Vec<Interval64> = boxes
                .iter()
                .map(|x| g.propagate_ibp_f64_cell(x).unwrap())
                .collect();
            let per_box_t = t1.elapsed();
            std::hint::black_box(per_box);

            println!(
                "[W={w} rows={rows} k=n={k}] batched {batched_t:?} per-box(serial) {per_box_t:?} \
                 speedup {:.1}x",
                per_box_t.as_secs_f64() / batched_t.as_secs_f64()
            );
        }
    }
}
