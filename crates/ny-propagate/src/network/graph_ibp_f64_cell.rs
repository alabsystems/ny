// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound f64 interval forward over a [`GraphNetwork`] for (near-)point inputs.
//!
//! # Why f64 (#cctsdb Phase C)
//!
//! The cctsdb_yolo_2023 cell-enumeration driver fixes every input dimension to
//! a point and needs a decisive enclosure of `Y_0` against the 0.5 threshold.
//! Plain f32 interval propagation amplifies interval width by roughly x2-7 per
//! convolution through the 30-conv backbone (measured ~2e10 end-to-end), so
//! even 1-ulp f32 input widening blows up to +/-1e4 at the heads — undecidable.
//! In f64 the rounding seeds are ~1e-16 relative, so the amplified output width
//! stays ~1e-4: decisively below the property margin.
//!
//! # Second consumer: box-refinement f64 leaf escalation (#f64-leaf)
//!
//! The nn4sys mscn band clauses have margins (~1e-5) BELOW the sound f32
//! forward floor: the disjunctive box-refinement screen refines clause boxes
//! down to f32 resolution and the f32 outward rounding still straddles the
//! threshold. Its terminal-failure leaves are re-evaluated here (gated by
//! [`GraphNetwork::supports_ibp_f64_cell`], kill-switch `NY_F64_LEAF=0`),
//! which motivated the Sigmoid / ReduceSum / live-MatMul / N-D-Linear ops.
//!
//! # Soundness contract
//!
//! For every node the computed interval ENCLOSES the exact real-arithmetic
//! image of the input box:
//! - data-movement ops (slice/reshape/gather/concat/transpose/resize/maxpool/
//!   relu/clip/min/max/trunc) are EXACT in f64 — endpoints are copied or
//!   compared, never rounded;
//! - element-wise arithmetic (+, -, *, /) computes endpoint candidates in f64
//!   round-to-nearest and widens the result OUTWARD by 1 ulp per endpoint
//!   (a nearest-rounded f64 op is within 0.5 ulp of the real value);
//! - dot products (Conv2d / Linear) accumulate naively in f64 and widen by the
//!   standard Higham bound `gamma_n * sum(|terms|)` with `n = terms + 1`,
//!   which encloses every rounding error of the products and the summation
//!   (Higham, *Accuracy and Stability of Numerical Algorithms*, sec. 3.5);
//! - index ops (ArgMax) return the sound candidate-set hull; comparisons
//!   (Equal) return {0}, {1}, or [0,1] exactly as the f32 CompareTensor.
//!
//! Unsupported layers FAIL CLOSED with `UnsupportedOp`: the caller must treat
//! the cell as undecided. f32 weights convert to f64 exactly (no rounding).

use std::collections::{HashMap, HashSet};

use ndarray::{ArrayD, ArrayView2, Ix2, IxDyn};
use ny_core::{NyError, Result};

use crate::layers::misc::CompareOp;
use crate::layers::Layer;

use super::core::graph::{GraphNetwork, NETWORK_INPUT};
use super::graph_ibp_f64_gemm::{
    fast_gemm_enabled, rump_interval_matmul, FAST_GEMM_MIN_ROWS, FAST_GEMM_MIN_VOLUME,
};

/// f64 interval tensor: elementwise `lower <= upper`.
#[derive(Debug, Clone)]
pub struct Interval64 {
    pub lower: ArrayD<f64>,
    pub upper: ArrayD<f64>,
}

impl Interval64 {
    pub fn point(values: ArrayD<f64>) -> Self {
        Self {
            lower: values.clone(),
            upper: values,
        }
    }

    /// Exact f32 -> f64 widening of an interval (every f32 is representable
    /// in f64, so no rounding occurs).
    pub fn from_f32(lower: &ArrayD<f32>, upper: &ArrayD<f32>) -> Self {
        Self {
            lower: lower.mapv(|v| v as f64),
            upper: upper.mapv(|v| v as f64),
        }
    }

    fn shape(&self) -> &[usize] {
        self.lower.shape()
    }

    fn reshaped(&self, shape: &[usize]) -> Result<Self> {
        let reshape = |arr: &ArrayD<f64>| -> Result<ArrayD<f64>> {
            arr.clone()
                .into_shape_with_order(IxDyn(shape))
                .map_err(|e| NyError::InvalidSpec(format!("f64 cell reshape failed: {e}")))
        };
        Ok(Self {
            lower: reshape(&self.lower)?,
            upper: reshape(&self.upper)?,
        })
    }
}

/// Unit roundoff for f64.
const U64_EPS: f64 = 2.220446049250313e-16 / 2.0; // 2^-53

/// Higham gamma_n = n*u / (1 - n*u); infinite/invalid for n*u >= 1.
pub(super) fn gamma_n(n: usize) -> Result<f64> {
    let nu = n as f64 * U64_EPS;
    if nu >= 1.0 {
        return Err(NyError::InvalidSpec(
            "f64 cell eval: dot product too long for Higham bound".to_string(),
        ));
    }
    Ok(nu / (1.0 - nu))
}

/// Widen an interval endpoint pair outward by one ulp each.
#[inline]
pub(super) fn widen1(lo: f64, hi: f64) -> (f64, f64) {
    (lo.next_down(), hi.next_up())
}

/// Outward widening (in f64 ulps) applied to transcendental results to cover
/// libm's not-correctly-rounded `exp` (~1 ulp documented worst case on the
/// platforms NY targets) plus the few surrounding elementary-op roundings
/// (the stable sigmoid composes exp + add + div, <= ~4-5 ulps total). 8 ulps
/// costs ~1.8e-15 relative — irrelevant against the 1e-5-scale nn4sys mscn
/// band margins the f64 leaf escalation exists to decide (#f64-leaf).
pub(super) const TRANSCENDENTAL_ULPS: u32 = 8;

/// Step a value `n` ulps toward -inf.
#[inline]
pub(super) fn widen_down_n(x: f64, n: u32) -> f64 {
    let mut v = x;
    for _ in 0..n {
        v = v.next_down();
    }
    v
}

/// Step a value `n` ulps toward +inf.
#[inline]
pub(super) fn widen_up_n(x: f64, n: u32) -> f64 {
    let mut v = x;
    for _ in 0..n {
        v = v.next_up();
    }
    v
}

/// Numerically STABLE sigmoid: for x <= 0, σ(x) = e^x / (1 + e^x) (e^x <= 1,
/// no overflow); for x > 0, σ(x) = 1 / (1 + e^-x). The naive single-branch
/// form overflows `exp` to +inf for x < -709.78 and collapses to exactly 0.0,
/// whose ulp-widened upper (~4e-323) would sit BELOW the true value
/// (~1e-309 at x = -710) — a non-enclosing bound. The stable form keeps every
/// intermediate finite so the relative-ulp widening argument applies at all
/// inputs. (When e^x underflows to 0 for x < -745.2, the true σ(x) < 2.5e-324
/// is still covered by the 8-ulp outward step from 0.)
#[inline]
pub(super) fn stable_sigmoid_f64(x: f64) -> f64 {
    if x <= 0.0 {
        let t = x.exp();
        t / (1.0 + t)
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Elementwise binary interval op with NumPy broadcasting.
///
/// `combine(alo, ahi, blo, bhi)` must return a sound REAL-arithmetic interval
/// for the op; the result is widened 1 ulp outward per endpoint to absorb the
/// f64 rounding of the endpoint computation itself (skipped when `exact`).
///
/// The four broadcast views are materialized to standard layout (a free
/// borrow for already-contiguous inputs — the dominant same-shape case — and
/// a small copy for stride-0 broadcast axes) so the element loop runs on
/// FLAT slices: the previous `indexed_iter` + per-element `IxDyn` clones and
/// indexed lookups dominated the tiny-tensor mscn walks (#f64-flat-elemwise,
/// ~13% whole-run `Baseiter` self time). VALUE-IDENTICAL: flat element `e`
/// of a standard-layout copy of a broadcast view is exactly the value
/// `indexed_iter` yielded at logical (row-major) index `e`, and `combine` is
/// applied in the same row-major order with the same endpoint pairs
/// (bit-identity gate: `broadcast_binary_flat_matches_indexed_reference`).
pub(super) fn broadcast_binary(
    a: &Interval64,
    b: &Interval64,
    exact: bool,
    combine: impl Fn(f64, f64, f64, f64) -> Result<(f64, f64)>,
) -> Result<Interval64> {
    let out_shape = crate::shape::broadcast_shapes(a.shape(), b.shape()).ok_or_else(|| {
        NyError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        }
    })?;
    let bc_err = || NyError::InvalidSpec("f64 cell eval: broadcast failed".to_string());
    let alo = a.lower.broadcast(IxDyn(&out_shape)).ok_or_else(bc_err)?;
    let ahi = a.upper.broadcast(IxDyn(&out_shape)).ok_or_else(bc_err)?;
    let blo = b.lower.broadcast(IxDyn(&out_shape)).ok_or_else(bc_err)?;
    let bhi = b.upper.broadcast(IxDyn(&out_shape)).ok_or_else(bc_err)?;

    let alo_std = alo.as_standard_layout();
    let ahi_std = ahi.as_standard_layout();
    let blo_std = blo.as_standard_layout();
    let bhi_std = bhi.as_standard_layout();
    let (alo_s, ahi_s, blo_s, bhi_s) = match (
        alo_std.as_slice(),
        ahi_std.as_slice(),
        blo_std.as_slice(),
        bhi_std.as_slice(),
    ) {
        (Some(al), Some(ah), Some(bl), Some(bh)) => (al, ah, bl, bh),
        _ => return Err(bc_err()),
    };

    let n: usize = out_shape.iter().product();
    let mut out_lo = Vec::with_capacity(n);
    let mut out_hi = Vec::with_capacity(n);
    for i in 0..n {
        let (mut lo, mut hi) = combine(alo_s[i], ahi_s[i], blo_s[i], bhi_s[i])?;
        if !exact {
            (lo, hi) = widen1(lo, hi);
        }
        out_lo.push(lo);
        out_hi.push(hi);
    }
    Ok(Interval64 {
        lower: ArrayD::from_shape_vec(IxDyn(&out_shape), out_lo)
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: broadcast out: {e}")))?,
        upper: ArrayD::from_shape_vec(IxDyn(&out_shape), out_hi)
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: broadcast out: {e}")))?,
    })
}

/// Sound real interval product of `[al, ah] * [bl, bh]` (endpoint candidates).
pub(super) fn interval_mul(al: f64, ah: f64, bl: f64, bh: f64) -> (f64, f64) {
    let candidates = [al * bl, al * bh, ah * bl, ah * bh];
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for c in candidates {
        lo = lo.min(c);
        hi = hi.max(c);
    }
    (lo, hi)
}

impl GraphNetwork {
    /// Sound f64 interval forward, restricted to the ancestors of the output
    /// node. Fails closed (`UnsupportedOp`/`InvalidSpec`) on anything outside
    /// the supported op set — the caller must treat that as "cell undecided".
    pub fn propagate_ibp_f64_cell(&self, input: &Interval64) -> Result<Interval64> {
        // Restrict evaluation to ancestors of the output: dead branches (e.g.
        // cctsdb's OpaqueSkip'd Slice_34/38 whose consumers all const-folded
        // away) must not fail-closed a decidable cell.
        let needed = self.output_ancestors()?;

        let mut cache: HashMap<&str, Interval64> = HashMap::new();
        let exec_order = self.exec_order()?;
        for node_name in exec_order {
            if !needed.contains(node_name.as_str()) {
                continue;
            }
            let node = self.node(node_name).ok_or_else(|| {
                NyError::InvalidSpec(format!("f64 cell eval: missing node '{node_name}'"))
            })?;
            let resolve = |name: &str| -> Result<Interval64> {
                if name == NETWORK_INPUT {
                    return Ok(input.clone());
                }
                cache.get(name).cloned().ok_or_else(|| {
                    NyError::InvalidSpec(format!("f64 cell eval: '{name}' not computed"))
                })
            };
            let out = eval_node(node.layer(), node, &resolve)?;
            cache.insert(node.name(), out);
        }
        cache.get(self.output_name()).cloned().ok_or_else(|| {
            NyError::InvalidSpec("f64 cell eval: output node not computed".to_string())
        })
    }

    /// Whether every output-ancestor node of this graph is in the op set
    /// supported by [`Self::propagate_ibp_f64_cell`]. Callers that escalate
    /// many boxes (the box-refinement f64 leaf escalation, #f64-leaf) should
    /// gate on this once instead of paying a failed walk per box. Runtime
    /// conditions (e.g. a Div divisor interval containing zero for a specific
    /// box) can still fail an individual walk — always fail-closed on `Err`.
    pub fn supports_ibp_f64_cell(&self) -> bool {
        match self.output_ancestors() {
            Ok(needed) => needed.iter().all(|name| {
                self.node(name)
                    .is_some_and(|node| cell_supports_layer(node.layer()))
            }),
            Err(_) => false,
        }
    }

    /// Names of all ancestor nodes of the output node (inclusive).
    pub(super) fn output_ancestors(&self) -> Result<HashSet<&str>> {
        let mut needed: HashSet<&str> = HashSet::new();
        let mut stack: Vec<&str> = vec![self.output_name()];
        while let Some(name) = stack.pop() {
            if name == NETWORK_INPUT || !needed.insert(name) {
                continue;
            }
            let node = self.node(name).ok_or_else(|| {
                NyError::InvalidSpec(format!("f64 cell eval: missing ancestor '{name}'"))
            })?;
            for input in node.inputs() {
                stack.push(input);
            }
        }
        Ok(needed)
    }
}

/// Resolve this node's inputs in ONNX order (interleaving Concat constants).
fn node_inputs(
    node: &super::core::graph::GraphNode,
    resolve: &dyn Fn(&str) -> Result<Interval64>,
) -> Result<Vec<Interval64>> {
    if let Layer::Concat(concat) = node.layer() {
        if let Some(ref constant_inputs) = concat.constant_inputs {
            let mut graph_idx = 0usize;
            return constant_inputs
                .iter()
                .map(|slot| match slot {
                    Some(constant) => Ok(Interval64::from_f32(constant.lower(), constant.upper())),
                    None => {
                        let name = node.inputs().get(graph_idx).ok_or_else(|| {
                            NyError::InvalidSpec(
                                "f64 cell eval: Concat ran out of graph inputs".to_string(),
                            )
                        })?;
                        graph_idx += 1;
                        resolve(name)
                    }
                })
                .collect();
        }
    }
    node.inputs().iter().map(|name| resolve(name)).collect()
}

/// Static layer-support predicate for [`GraphNetwork::supports_ibp_f64_cell`].
///
/// KEEP IN SYNC with the `eval_node` dispatch below: every arm that can
/// succeed must be listed here (including its static preconditions — e.g.
/// Gather needs constant indices); anything else must return `false` so the
/// escalation gate fails closed. Shape/rank/runtime conditions (Div divisor
/// sign, MatMul rank) are still checked per walk.
fn cell_supports_layer(layer: &Layer) -> bool {
    match layer {
        Layer::Reshape(_)
        | Layer::Flatten(_)
        | Layer::Slice(_)
        | Layer::Squeeze(_)
        | Layer::Unsqueeze(_)
        | Layer::Transpose(_)
        | Layer::Concat(_)
        | Layer::Resize(_)
        | Layer::MaxPool2d(_)
        | Layer::ReLU(_)
        | Layer::Clip(_)
        | Layer::Trunc(_)
        | Layer::Sigmoid(_)
        | Layer::MinBinary(_)
        | Layer::MaxBinary(_)
        | Layer::ArgMax(_)
        | Layer::Add(_)
        | Layer::Sub(_)
        | Layer::MulBinary(_)
        | Layer::Div(_)
        | Layer::AddConstant(_)
        | Layer::MulConstant(_)
        | Layer::ReduceSum(_)
        | Layer::Linear(_)
        | Layer::MatMul(_)
        | Layer::Conv2d(_) => true,
        Layer::Gather(gather) => gather.constant_indices().is_some(),
        Layer::CompareTensor(compare) => compare.op == CompareOp::Eq,
        Layer::ScatterNd(scatter) => {
            scatter.data_constant().is_some()
                && scatter.updates_constant().is_some()
                && !scatter.has_static_indices()
        }
        _ => false,
    }
}

pub(super) fn eval_node(
    layer: &Layer,
    node: &super::core::graph::GraphNode,
    resolve: &dyn Fn(&str) -> Result<Interval64>,
) -> Result<Interval64> {
    let inputs = node_inputs(node, resolve)?;
    let unary = || -> Result<&Interval64> {
        inputs
            .first()
            .ok_or_else(|| NyError::InvalidSpec("f64 cell eval: missing unary input".to_string()))
    };
    let binary = || -> Result<(&Interval64, &Interval64)> {
        match (inputs.first(), inputs.get(1)) {
            (Some(a), Some(b)) => Ok((a, b)),
            _ => Err(NyError::InvalidSpec(
                "f64 cell eval: missing binary inputs".to_string(),
            )),
        }
    };

    match layer {
        // ---- exact data movement -------------------------------------------------
        Layer::Reshape(reshape) => {
            let x = unary()?;
            let out_shape = reshape.compute_output_shape(x.shape())?;
            x.reshaped(&out_shape)
        }
        // Exact data movement like Reshape: ONNX Flatten collapses the shape to
        // 2-D around `axis`; endpoints are copied verbatim (row-major), never
        // rounded. Unblocks the exact-f64 witness gate on soundnessbench
        // (Conv x6 -> Flatten -> Gemm x2), whose whole-net escalation previously
        // failed closed here (#sb-rebank lever 2).
        Layer::Flatten(flatten) => {
            let x = unary()?;
            let out_shape = flatten.compute_output_shape(x.shape())?;
            x.reshaped(&out_shape)
        }
        Layer::Slice(slice) => {
            let x = unary()?;
            let (axis, start, end) = slice.resolved_range(x.shape())?;
            let take = |arr: &ArrayD<f64>| {
                arr.slice_axis(ndarray::Axis(axis), ndarray::Slice::from(start..end))
                    .to_owned()
            };
            Ok(Interval64 {
                lower: take(&x.lower),
                upper: take(&x.upper),
            })
        }
        Layer::Squeeze(squeeze) => {
            let x = unary()?;
            let ndim = x.shape().len();
            let axis = resolve_axis_i64(squeeze.axis as i64, ndim, "Squeeze")?;
            if x.shape()[axis] != 1 {
                return Err(NyError::InvalidSpec(format!(
                    "f64 cell eval: Squeeze axis {axis} has size {}",
                    x.shape()[axis]
                )));
            }
            let mut shape = x.shape().to_vec();
            shape.remove(axis);
            x.reshaped(&shape)
        }
        Layer::Unsqueeze(unsqueeze) => {
            let x = unary()?;
            let ndim = x.shape().len();
            let axis = resolve_axis_i64(unsqueeze.axis as i64, ndim + 1, "Unsqueeze")?;
            let mut shape = x.shape().to_vec();
            shape.insert(axis, 1);
            x.reshaped(&shape)
        }
        Layer::Transpose(transpose) => {
            let x = unary()?;
            if transpose.axes.len() != x.shape().len() {
                return Err(NyError::InvalidSpec(
                    "f64 cell eval: Transpose perm rank mismatch".to_string(),
                ));
            }
            let perm = transpose.axes.clone();
            Ok(Interval64 {
                lower: x
                    .lower
                    .clone()
                    .permuted_axes(IxDyn(&perm))
                    .as_standard_layout()
                    .to_owned(),
                upper: x
                    .upper
                    .clone()
                    .permuted_axes(IxDyn(&perm))
                    .as_standard_layout()
                    .to_owned(),
            })
        }
        Layer::Gather(gather) => eval_gather(gather, unary()?),
        Layer::Concat(concat) => eval_concat(concat, &inputs),
        Layer::Resize(resize) => eval_resize(resize, unary()?),
        Layer::MaxPool2d(pool) => eval_maxpool(pool, unary()?),

        // ---- exact monotone / selection ops ---------------------------------------
        Layer::ReLU(_) => {
            let x = unary()?;
            Ok(Interval64 {
                lower: x.lower.mapv(|v| v.max(0.0)),
                upper: x.upper.mapv(|v| v.max(0.0)),
            })
        }
        Layer::Clip(clip) => {
            let x = unary()?;
            let (min, max) = (clip.min as f64, clip.max as f64);
            Ok(Interval64 {
                lower: x.lower.mapv(|v| v.clamp(min, max)),
                upper: x.upper.mapv(|v| v.clamp(min, max)),
            })
        }
        Layer::Trunc(_) => {
            // trunc (round-toward-zero) is monotone non-decreasing and exact in f64.
            let x = unary()?;
            Ok(Interval64 {
                lower: x.lower.mapv(f64::trunc),
                upper: x.upper.mapv(f64::trunc),
            })
        }
        Layer::Sigmoid(_) => {
            // Monotone increasing: [σ(l), σ(u)], widened TRANSCENDENTAL_ULPS
            // outward for libm exp error and clamped to sigmoid's true range
            // [0, 1] (clamping toward the true range is sound because
            // 0 <= σ(x) <= 1 for ALL real x). nn4sys mscn output head.
            let x = unary()?;
            Ok(Interval64 {
                lower: x
                    .lower
                    .mapv(|v| widen_down_n(stable_sigmoid_f64(v), TRANSCENDENTAL_ULPS).max(0.0)),
                upper: x
                    .upper
                    .mapv(|v| widen_up_n(stable_sigmoid_f64(v), TRANSCENDENTAL_ULPS).min(1.0)),
            })
        }
        Layer::MinBinary(_) => {
            let (a, b) = binary()?;
            broadcast_binary(a, b, true, |al, ah, bl, bh| Ok((al.min(bl), ah.min(bh))))
        }
        Layer::MaxBinary(_) => {
            let (a, b) = binary()?;
            broadcast_binary(a, b, true, |al, ah, bl, bh| Ok((al.max(bl), ah.max(bh))))
        }
        Layer::ArgMax(argmax) => eval_argmax(argmax, unary()?),
        Layer::CompareTensor(compare) => {
            let (a, b) = binary()?;
            if compare.op != CompareOp::Eq {
                return Err(NyError::UnsupportedOp(
                    "f64 cell eval: only Eq CompareTensor supported".to_string(),
                ));
            }
            broadcast_binary(a, b, true, |al, ah, bl, bh| {
                Ok(if al == ah && bl == bh && al == bl {
                    (1.0, 1.0)
                } else if al > bh || ah < bl {
                    (0.0, 0.0)
                } else {
                    (0.0, 1.0)
                })
            })
        }

        // ---- element-wise arithmetic (1-ulp outward) -------------------------------
        Layer::Add(_) => {
            let (a, b) = binary()?;
            broadcast_binary(a, b, false, |al, ah, bl, bh| Ok((al + bl, ah + bh)))
        }
        Layer::Sub(_) => {
            let (a, b) = binary()?;
            broadcast_binary(a, b, false, |al, ah, bl, bh| Ok((al - bh, ah - bl)))
        }
        Layer::MulBinary(_) => {
            let (a, b) = binary()?;
            broadcast_binary(a, b, false, |al, ah, bl, bh| {
                Ok(interval_mul(al, ah, bl, bh))
            })
        }
        Layer::Div(_) => {
            let (a, b) = binary()?;
            broadcast_binary(a, b, false, |al, ah, bl, bh| {
                if bl <= 0.0 && bh >= 0.0 {
                    return Err(NyError::InvalidSpec(
                        "f64 cell eval: Div divisor interval contains zero".to_string(),
                    ));
                }
                let candidates = [al / bl, al / bh, ah / bl, ah / bh];
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for c in candidates {
                    lo = lo.min(c);
                    hi = hi.max(c);
                }
                Ok((lo, hi))
            })
        }
        Layer::AddConstant(add) => {
            let x = unary()?;
            let c = Interval64::from_f32(add.constant(), add.constant());
            broadcast_binary(x, &c, false, |al, ah, bl, bh| Ok((al + bl, ah + bh)))
        }
        Layer::MulConstant(mul) => {
            let x = unary()?;
            let c = Interval64::from_f32(mul.constant(), mul.constant());
            broadcast_binary(x, &c, false, |al, ah, bl, bh| {
                Ok(interval_mul(al, ah, bl, bh))
            })
        }

        // ---- reductions (Higham widening) -------------------------------------------
        Layer::ReduceSum(reduce) => eval_reduce_sum(reduce, unary()?),

        // ---- dot products (Higham widening) ----------------------------------------
        Layer::Linear(linear) => eval_linear(linear, unary()?),
        Layer::MatMul(matmul) => {
            let (a, b) = binary()?;
            eval_matmul(matmul, a, b)
        }
        Layer::Conv2d(conv) => eval_conv2d(conv, unary()?),

        // ---- scatter ---------------------------------------------------------------
        Layer::ScatterNd(scatter) => eval_scatter_nd(scatter, &inputs),

        other => Err(NyError::UnsupportedOp(format!(
            "f64 cell eval: unsupported layer {}",
            other.layer_type()
        ))),
    }
}

fn resolve_axis_i64(axis: i64, ndim: usize, op: &str) -> Result<usize> {
    let resolved = if axis < 0 { axis + ndim as i64 } else { axis };
    if resolved < 0 || resolved >= ndim as i64 {
        return Err(NyError::InvalidSpec(format!(
            "f64 cell eval: {op} axis {axis} out of range for rank {ndim}"
        )));
    }
    Ok(resolved as usize)
}

fn eval_gather(gather: &crate::layers::GatherLayer, x: &Interval64) -> Result<Interval64> {
    let indices = gather.constant_indices().ok_or_else(|| {
        NyError::UnsupportedOp("f64 cell eval: Gather requires constant indices".to_string())
    })?;
    let axis = resolve_axis_i64(gather.axis_raw(), x.shape().len(), "Gather")?;
    let axis_len = x.shape()[axis] as i64;
    // Output shape: data[..axis] ++ indices.shape ++ data[axis+1..]
    let mut out_shape: Vec<usize> = x.shape()[..axis].to_vec();
    out_shape.extend_from_slice(indices.shape());
    out_shape.extend_from_slice(&x.shape()[axis + 1..]);

    let take = |arr: &ArrayD<f64>| -> Result<ArrayD<f64>> {
        // Select each index along `axis` in indices order, then reshape the
        // gathered axis to the indices shape.
        let mut slices: Vec<ArrayD<f64>> = Vec::with_capacity(indices.len().max(1));
        for &raw in indices.iter() {
            let idx = if raw < 0 { raw + axis_len } else { raw };
            if idx < 0 || idx >= axis_len {
                return Err(NyError::InvalidSpec(format!(
                    "f64 cell eval: Gather index {raw} out of range for axis len {axis_len}"
                )));
            }
            slices.push(
                arr.index_axis(ndarray::Axis(axis), idx as usize)
                    .insert_axis(ndarray::Axis(axis))
                    .to_owned(),
            );
        }
        let views: Vec<_> = slices.iter().map(|s| s.view()).collect();
        let stacked = if views.is_empty() {
            // Empty indices: empty gathered axis.
            arr.slice_axis(ndarray::Axis(axis), ndarray::Slice::from(0..0))
                .to_owned()
        } else {
            ndarray::concatenate(ndarray::Axis(axis), &views)
                .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: Gather concat: {e}")))?
        };
        stacked
            .into_shape_with_order(IxDyn(&out_shape))
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: Gather reshape: {e}")))
    };
    Ok(Interval64 {
        lower: take(&x.lower)?,
        upper: take(&x.upper)?,
    })
}

pub(super) fn eval_concat(
    concat: &crate::layers::ConcatLayer,
    inputs: &[Interval64],
) -> Result<Interval64> {
    let first = inputs.first().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: Concat needs at least one input".to_string())
    })?;
    let axis = concat.normalize_axis(first.shape().len())?;
    let lo_views: Vec<_> = inputs.iter().map(|i| i.lower.view()).collect();
    let hi_views: Vec<_> = inputs.iter().map(|i| i.upper.view()).collect();
    let lower = ndarray::concatenate(ndarray::Axis(axis), &lo_views)
        .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: Concat: {e}")))?;
    let upper = ndarray::concatenate(ndarray::Axis(axis), &hi_views)
        .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: Concat: {e}")))?;
    Ok(Interval64 { lower, upper })
}

fn eval_resize(resize: &crate::layers::ResizeLayer, x: &Interval64) -> Result<Interval64> {
    // Nearest-neighbor integer upsample of the last two axes (floor mapping):
    // out[..., i, j] = in[..., i / sh, j / sw] — a pure copy, exact.
    let ndim = x.shape().len();
    if ndim < 2 {
        return Err(NyError::InvalidSpec(
            "f64 cell eval: Resize needs rank >= 2".to_string(),
        ));
    }
    let (sh, sw) = (resize.scale_h, resize.scale_w);
    let mut out_shape = x.shape().to_vec();
    out_shape[ndim - 2] *= sh;
    out_shape[ndim - 1] *= sw;
    let take = |arr: &ArrayD<f64>| -> ArrayD<f64> {
        ArrayD::from_shape_fn(IxDyn(&out_shape), |idx| {
            let mut src: Vec<usize> = (0..ndim).map(|d| idx[d]).collect();
            src[ndim - 2] /= sh;
            src[ndim - 1] /= sw;
            arr[IxDyn(&src)]
        })
    };
    Ok(Interval64 {
        lower: take(&x.lower),
        upper: take(&x.upper),
    })
}

fn eval_maxpool(pool: &crate::layers::MaxPool2dLayer, x: &Interval64) -> Result<Interval64> {
    let shape = x.shape().to_vec();
    let ndim = shape.len();
    if !(3..=4).contains(&ndim) {
        return Err(NyError::InvalidSpec(
            "f64 cell eval: MaxPool2d needs rank 3 or 4".to_string(),
        ));
    }
    let (in_h, in_w) = (shape[ndim - 2], shape[ndim - 1]);
    let (kh, kw) = pool.kernel_size;
    let (sh, sw) = pool.stride;
    let (ph, pw) = pool.padding;
    let out_h = (in_h + 2 * ph).saturating_sub(kh) / sh + 1;
    let out_w = (in_w + 2 * pw).saturating_sub(kw) / sw + 1;
    let mut out_shape = shape;
    out_shape[ndim - 2] = out_h;
    out_shape[ndim - 1] = out_w;

    let take = |arr: &ArrayD<f64>| -> ArrayD<f64> {
        ArrayD::from_shape_fn(IxDyn(&out_shape), |idx| {
            let idx_vec: Vec<usize> = (0..ndim).map(|d| idx[d]).collect();
            let (oh, ow) = (idx_vec[ndim - 2], idx_vec[ndim - 1]);
            let mut best = f64::NEG_INFINITY;
            let mut any_valid = false;
            for dh in 0..kh {
                for dw in 0..kw {
                    let ih = (oh * sh + dh) as isize - ph as isize;
                    let iw = (ow * sw + dw) as isize - pw as isize;
                    if ih < 0 || iw < 0 || ih >= in_h as isize || iw >= in_w as isize {
                        continue; // padded region: valid-only pooling
                    }
                    let mut src = idx_vec.clone();
                    src[ndim - 2] = ih as usize;
                    src[ndim - 1] = iw as usize;
                    best = best.max(arr[IxDyn(&src)]);
                    any_valid = true;
                }
            }
            if any_valid {
                best
            } else {
                0.0
            }
        })
    };
    Ok(Interval64 {
        lower: take(&x.lower),
        upper: take(&x.upper),
    })
}

fn eval_argmax(argmax: &crate::layers::ArgMaxLayer, x: &Interval64) -> Result<Interval64> {
    let ndim = x.shape().len();
    let axis = resolve_axis_i64(argmax.axis, ndim, "ArgMax")?;
    let axis_len = x.shape()[axis];
    if axis_len == 0 {
        return Err(NyError::InvalidSpec(
            "f64 cell eval: ArgMax over empty axis".to_string(),
        ));
    }
    let mut out_shape = x.shape().to_vec();
    if argmax.keepdims {
        out_shape[axis] = 1;
    } else {
        out_shape.remove(axis);
    }

    let mut out_lo = ArrayD::zeros(IxDyn(&out_shape));
    let mut out_hi = ArrayD::zeros(IxDyn(&out_shape));
    // Iterate lanes along `axis`.
    let lanes_lo = x.lower.lanes(ndarray::Axis(axis));
    let lanes_hi = x.upper.lanes(ndarray::Axis(axis));
    for ((lane_lo, lane_hi), (slot_lo, slot_hi)) in lanes_lo
        .into_iter()
        .zip(lanes_hi)
        .zip(out_lo.iter_mut().zip(out_hi.iter_mut()))
    {
        // Candidate set: i is a possible argmax iff upper_i >= max_j lower_j.
        let best_lower = lane_lo.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut min_cand = usize::MAX;
        let mut max_cand = 0usize;
        for (i, &hi) in lane_hi.iter().enumerate() {
            if hi >= best_lower {
                min_cand = min_cand.min(i);
                max_cand = max_cand.max(i);
            }
        }
        if min_cand == usize::MAX {
            return Err(NyError::InvalidSpec(
                "f64 cell eval: ArgMax found no candidates".to_string(),
            ));
        }
        *slot_lo = min_cand as f64;
        *slot_hi = max_cand as f64;
    }
    Ok(Interval64 {
        lower: out_lo,
        upper: out_hi,
    })
}

fn eval_linear(linear: &crate::layers::LinearLayer, x: &Interval64) -> Result<Interval64> {
    eval_linear_with_bias(linear, x, true)
}

/// Linear with an explicit bias switch: the mean-value derivative channel
/// (#f64-mvf) maps derivatives through the SAME Higham-widened `W·d` but the
/// bias (a constant) contributes zero derivative, so it must be excluded.
pub(super) fn eval_linear_with_bias(
    linear: &crate::layers::LinearLayer,
    x: &Interval64,
    include_bias: bool,
) -> Result<Interval64> {
    eval_linear_with_bias_inner(linear, x, include_bias, true)
}

/// Scalar-only Linear — the reference path the fast kernel wiring is
/// soundness-tested against.
#[cfg(test)]
pub(super) fn eval_linear_with_bias_scalar(
    linear: &crate::layers::LinearLayer,
    x: &Interval64,
    include_bias: bool,
) -> Result<Interval64> {
    eval_linear_with_bias_inner(linear, x, include_bias, false)
}

fn eval_linear_with_bias_inner(
    linear: &crate::layers::LinearLayer,
    x: &Interval64,
    include_bias: bool,
    allow_fast: bool,
) -> Result<Interval64> {
    // N-D input, weight applied to the LAST axis (matches the f32 Linear IBP
    // convention `[...batch, in] -> [...batch, out]`). Rank-1 (the original
    // cctsdb cell case) is the batch=1 special case. nn4sys mscn applies its
    // set-MLPs row-wise over [rows, in] inputs (#f64-leaf).
    let shape = x.shape().to_vec();
    if shape.is_empty() {
        return Err(NyError::UnsupportedOp(
            "f64 cell eval: Linear on rank-0 input".to_string(),
        ));
    }
    let (out_dim, in_dim) = linear.weight.dim();
    if shape[shape.len() - 1] != in_dim {
        return Err(NyError::ShapeMismatch {
            expected: vec![in_dim],
            got: shape,
        });
    }
    let batch: usize = shape[..shape.len() - 1].iter().product();

    let x_lo_owned = x.lower.as_standard_layout();
    let x_hi_owned = x.upper.as_standard_layout();
    let x_lo = x_lo_owned.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: linear input not contiguous".to_string())
    })?;
    let x_hi = x_hi_owned.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: linear input not contiguous".to_string())
    })?;

    let mut out_shape = shape[..shape.len() - 1].to_vec();
    out_shape.push(out_dim);

    // Fast path (#f64-blas-gemm): Rump midpoint-radius interval GEMM on
    // BLAS for FAT batches; the constant weight is a point operand so it
    // costs 3 plain GEMMs. Thin batches (see FAST_GEMM_MIN_ROWS: measured
    // slower on BLAS) take the unrolled scalar loop below. `None`
    // (non-finite input, layout surprise, gamma overflow) also falls back.
    if allow_fast
        && fast_gemm_enabled()
        && batch >= FAST_GEMM_MIN_ROWS
        && batch
            .checked_mul(out_dim)
            .and_then(|v| v.checked_mul(in_dim))
            .is_some_and(|v| v > FAST_GEMM_MIN_VOLUME)
    {
        if let Some(out) =
            try_eval_linear_fast(linear, x_lo, x_hi, batch, in_dim, include_bias, &out_shape)
        {
            return Ok(out);
        }
    }

    let gamma = gamma_n(in_dim + 2)?;
    let weight_owned = linear.weight.as_standard_layout();
    let weight = weight_owned.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: linear weight not contiguous".to_string())
    })?;
    let mut out_lo = vec![0.0f64; batch * out_dim];
    let mut out_hi = vec![0.0f64; batch * out_dim];
    for row in 0..batch {
        let x_base = row * in_dim;
        let xl_row = &x_lo[x_base..x_base + in_dim];
        let xu_row = &x_hi[x_base..x_base + in_dim];
        for o in 0..out_dim {
            let w_row = &weight[o * in_dim..(o + 1) * in_dim];
            let (mut lo, mut hi, mut abs) = interval_dot_f32w(w_row, xl_row, xu_row);
            if include_bias {
                if let Some(bias) = linear.bias.as_ref() {
                    let b = bias[o] as f64;
                    lo += b;
                    hi += b;
                    abs += b.abs();
                }
            }
            let err = gamma * abs;
            out_lo[row * out_dim + o] = (lo - err).next_down();
            out_hi[row * out_dim + o] = (hi + err).next_up();
        }
    }
    Ok(Interval64 {
        lower: ArrayD::from_shape_vec(IxDyn(&out_shape), out_lo)
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: linear out: {e}")))?,
        upper: ArrayD::from_shape_vec(IxDyn(&out_shape), out_hi)
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: linear out: {e}")))?,
    })
}

/// Interval dot product of one f32 weight row against an f64 interval
/// vector: returns `(lo, hi, abs)` with `lo = Σ min(w·xl, w·xu)`,
/// `hi = Σ max(w·xl, w·xu)`, `abs = Σ max(|pl|, |pu|)` — exactly the
/// quantities the Higham-widened scalar Linear needs.
///
/// 4-way unrolled with independent accumulator lanes (breaks the FP add
/// dependency chain: measured ~3x on the mscn m<=6, k=n=2048 shapes, where
/// this loop IS the f64 cell's per-node cost). SOUNDNESS: re-associating
/// the sum changes only WHICH nearest-rounded partial sums occur; the
/// Higham bound `gamma_n * Σ|terms|` applied by the caller covers EVERY
/// summation order (each term passes through <= n roundings), so the
/// widened interval still encloses the exact value. The branchless
/// `min/max` selects the same two products as the old `w >= 0.0` branch
/// for every non-NaN input.
#[inline]
fn interval_dot_f32w(w_row: &[f32], xl: &[f64], xu: &[f64]) -> (f64, f64, f64) {
    const LANES: usize = 4;
    let mut lo = [0.0f64; LANES];
    let mut hi = [0.0f64; LANES];
    let mut abs = [0.0f64; LANES];
    let (wc, w_rem) = w_row.as_chunks::<LANES>();
    let (xlc, xl_rem) = xl.as_chunks::<LANES>();
    let (xuc, xu_rem) = xu.as_chunks::<LANES>();
    for ((w4, l4), u4) in wc.iter().zip(xlc).zip(xuc) {
        for lane in 0..LANES {
            let w = f64::from(w4[lane]);
            let a = w * l4[lane];
            let b = w * u4[lane];
            let (pl, pu) = if a < b { (a, b) } else { (b, a) };
            lo[lane] += pl;
            hi[lane] += pu;
            abs[lane] += pl.abs().max(pu.abs());
        }
    }
    for ((&w, &l), &u) in w_rem.iter().zip(xl_rem.iter()).zip(xu_rem.iter()) {
        let w = f64::from(w);
        let a = w * l;
        let b = w * u;
        let (pl, pu) = if a < b { (a, b) } else { (b, a) };
        lo[0] += pl;
        hi[0] += pu;
        abs[0] += pl.abs().max(pu.abs());
    }
    (
        (lo[0] + lo[1]) + (lo[2] + lo[3]),
        (hi[0] + hi[1]) + (hi[2] + hi[3]),
        (abs[0] + abs[1]) + (abs[2] + abs[3]),
    )
}

/// Fast Linear via the Rump midpoint-radius GEMM kernel
/// (`super::graph_ibp_f64_gemm`, soundness argument there): `[x] @ W^T` with
/// the exact f64 weight as a POINT operand, bias folded afterwards with 1-ulp
/// outward widening per endpoint (`fl(lo + b)` is within 0.5 ulp of the real
/// sum, and `b` converts f32 -> f64 exactly). Returns `None` to fall back to
/// the scalar Higham loop.
fn try_eval_linear_fast(
    linear: &crate::layers::LinearLayer,
    x_lo: &[f64],
    x_hi: &[f64],
    batch: usize,
    in_dim: usize,
    include_bias: bool,
    out_shape: &[usize],
) -> Option<Interval64> {
    if std::env::var("NY_F64_BLAS_LOG").is_ok_and(|v| v == "1") {
        let (out_dim, _) = linear.weight.dim();
        eprintln!("NY_F64_BLAS linear m={batch} k={in_dim} n={out_dim}");
    }
    let a_lo = ArrayView2::from_shape((batch, in_dim), x_lo).ok()?;
    let a_hi = ArrayView2::from_shape((batch, in_dim), x_hi).ok()?;
    // Exact f32 -> f64 transpose of the weight: a point interval [wt, wt].
    let wt = linear.weight.t().map(|&w| f64::from(w));
    let (mut lo, mut hi) = rump_interval_matmul(a_lo, a_hi, wt.view(), wt.view())?;
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
        lower: lo.into_dyn().into_shape_with_order(IxDyn(out_shape)).ok()?,
        upper: hi.into_dyn().into_shape_with_order(IxDyn(out_shape)).ok()?,
    })
}

/// ReduceSum over resolved axes: each axis sum accumulates in f64
/// round-to-nearest and widens by the Higham bound `gamma_n * sum(|terms|)`
/// plus 1 ulp per endpoint — enclosing every rounding of the summation.
pub(super) fn eval_reduce_sum(
    reduce: &crate::layers::ReduceSumLayer,
    x: &Interval64,
) -> Result<Interval64> {
    let ndim = x.shape().len();
    let mut axes: Vec<usize> = if reduce.axes.is_empty() {
        (0..ndim).collect()
    } else {
        reduce
            .axes
            .iter()
            .map(|&axis| resolve_axis_i64(axis, ndim, "ReduceSum"))
            .collect::<Result<Vec<_>>>()?
    };
    axes.sort_unstable();
    let n_before = axes.len();
    axes.dedup();
    if axes.len() != n_before {
        return Err(NyError::InvalidSpec(
            "f64 cell eval: duplicate ReduceSum axes".to_string(),
        ));
    }

    let mut lo = x.lower.clone();
    let mut hi = x.upper.clone();
    // Descending so earlier reductions don't shift later axis indices.
    for &axis in axes.iter().rev() {
        let n = lo.shape()[axis];
        let gamma = gamma_n(n + 1)?;
        let sum_widened = |arr: &ArrayD<f64>, is_lower: bool| -> ArrayD<f64> {
            arr.map_axis(ndarray::Axis(axis), |lane| {
                let mut s = 0.0f64;
                let mut abs = 0.0f64;
                for &v in lane.iter() {
                    s += v;
                    abs += v.abs();
                }
                let err = gamma * abs;
                if is_lower {
                    (s - err).next_down()
                } else {
                    (s + err).next_up()
                }
            })
        };
        let new_lo = sum_widened(&lo, true);
        let new_hi = sum_widened(&hi, false);
        if reduce.keepdims {
            let mut shape = new_lo.shape().to_vec();
            shape.insert(axis, 1);
            lo = new_lo.into_shape_with_order(IxDyn(&shape)).map_err(|e| {
                NyError::InvalidSpec(format!("f64 cell eval: ReduceSum keepdims: {e}"))
            })?;
            hi = new_hi.into_shape_with_order(IxDyn(&shape)).map_err(|e| {
                NyError::InvalidSpec(format!("f64 cell eval: ReduceSum keepdims: {e}"))
            })?;
        } else {
            lo = new_lo;
            hi = new_hi;
        }
    }
    Ok(Interval64 {
        lower: lo,
        upper: hi,
    })
}

/// Live-x-live rank-2 interval matmul `[m,k] @ [k,n]` (honoring the layer's
/// optional B-transpose and scale). Large shapes take the Rump
/// midpoint-radius BLAS kernel (#f64-blas-gemm, kill-switch `NY_F64_BLAS=0`);
/// everything else — and every case the fast kernel declines — takes the
/// scalar corner-product path. Both ENCLOSE the true interval product.
pub(super) fn eval_matmul(
    layer: &crate::layers::MatMulLayer,
    a: &Interval64,
    b: &Interval64,
) -> Result<Interval64> {
    if let Some(out) = try_eval_matmul_fast(layer, a, b) {
        return Ok(out);
    }
    eval_matmul_scalar(layer, a, b)
}

/// Fast rank-2 MatMul via the Rump midpoint-radius GEMM kernel; the optional
/// scale applies as one interval product widened 1 ulp (same tail as the
/// scalar path). `None` = use the scalar path (small shape, kill-switch,
/// shape problem — reported by the scalar path — or non-finite input).
fn try_eval_matmul_fast(
    layer: &crate::layers::MatMulLayer,
    a: &Interval64,
    b: &Interval64,
) -> Option<Interval64> {
    if !fast_gemm_enabled() || a.shape().len() != 2 || b.shape().len() != 2 {
        return None;
    }
    let transpose_b = layer.transpose_b();
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let (k2, n) = if transpose_b {
        (b.shape()[1], b.shape()[0])
    } else {
        (b.shape()[0], b.shape()[1])
    };
    if k != k2
        || m < FAST_GEMM_MIN_ROWS
        || m.checked_mul(n)
            .and_then(|v| v.checked_mul(k))
            .is_none_or(|v| v <= FAST_GEMM_MIN_VOLUME)
    {
        return None;
    }
    fn view2(arr: &ArrayD<f64>) -> Option<ArrayView2<'_, f64>> {
        arr.view().into_dimensionality::<Ix2>().ok()
    }
    let (a_lo, a_hi) = (view2(&a.lower)?, view2(&a.upper)?);
    let (mut b_lo, mut b_hi) = (view2(&b.lower)?, view2(&b.upper)?);
    if transpose_b {
        b_lo = b_lo.reversed_axes();
        b_hi = b_hi.reversed_axes();
    }
    let (mut lo, mut hi) = rump_interval_matmul(a_lo, a_hi, b_lo, b_hi)?;
    if let Some(s) = layer.scale().map(f64::from) {
        for (l, h) in lo.iter_mut().zip(hi.iter_mut()) {
            let (nl, nh) = interval_mul(*l, *h, s, s);
            let (nl, nh) = widen1(nl, nh);
            *l = nl;
            *h = nh;
        }
    }
    Some(Interval64 {
        lower: lo.into_dyn(),
        upper: hi.into_dyn(),
    })
}

/// Scalar rank-2 interval matmul: per output element, k interval corner
/// products accumulate in f64 and widen by the Higham bound like
/// `eval_linear`; the optional scale applies as one more interval product
/// widened 1 ulp. Fails closed on other ranks. This is the reference path
/// the fast kernel is soundness-tested against.
pub(super) fn eval_matmul_scalar(
    layer: &crate::layers::MatMulLayer,
    a: &Interval64,
    b: &Interval64,
) -> Result<Interval64> {
    if a.shape().len() != 2 || b.shape().len() != 2 {
        return Err(NyError::UnsupportedOp(format!(
            "f64 cell eval: MatMul supports rank-2 x rank-2 only (got {:?} x {:?})",
            a.shape(),
            b.shape()
        )));
    }
    let transpose_b = layer.transpose_b();
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let (k2, n) = if transpose_b {
        (b.shape()[1], b.shape()[0])
    } else {
        (b.shape()[0], b.shape()[1])
    };
    if k != k2 {
        return Err(NyError::ShapeMismatch {
            expected: vec![k],
            got: vec![k2],
        });
    }
    let scale = layer.scale().map(f64::from);

    let contiguous = |arr: &ArrayD<f64>| -> Result<Vec<f64>> { Ok(arr.iter().copied().collect()) };
    let a_lo = contiguous(&a.lower)?;
    let a_hi = contiguous(&a.upper)?;
    let b_lo = contiguous(&b.lower)?;
    let b_hi = contiguous(&b.upper)?;
    // Flat index of B[l, j] honoring the optional transpose.
    let bidx = |l: usize, j: usize| if transpose_b { j * k + l } else { l * n + j };

    let gamma = gamma_n(k + 2)?;
    let mut out_lo = vec![0.0f64; m * n];
    let mut out_hi = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut lo = 0.0f64;
            let mut hi = 0.0f64;
            let mut abs = 0.0f64;
            for l in 0..k {
                let (pl, pu) = interval_mul(
                    a_lo[i * k + l],
                    a_hi[i * k + l],
                    b_lo[bidx(l, j)],
                    b_hi[bidx(l, j)],
                );
                lo += pl;
                hi += pu;
                abs += pl.abs().max(pu.abs());
            }
            let err = gamma * abs;
            let (mut lo, mut hi) = ((lo - err).next_down(), (hi + err).next_up());
            if let Some(s) = scale {
                (lo, hi) = interval_mul(lo, hi, s, s);
                (lo, hi) = widen1(lo, hi);
            }
            out_lo[i * n + j] = lo;
            out_hi[i * n + j] = hi;
        }
    }
    Ok(Interval64 {
        lower: ArrayD::from_shape_vec(IxDyn(&[m, n]), out_lo)
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: matmul out: {e}")))?,
        upper: ArrayD::from_shape_vec(IxDyn(&[m, n]), out_hi)
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: matmul out: {e}")))?,
    })
}

fn eval_conv2d(conv: &crate::layers::Conv2dLayer, x: &Interval64) -> Result<Interval64> {
    let shape = x.shape().to_vec();
    let ndim = shape.len();
    if !(3..=4).contains(&ndim) {
        return Err(NyError::InvalidSpec(
            "f64 cell eval: Conv2d needs rank 3 or 4".to_string(),
        ));
    }
    let has_batch = ndim == 4;
    let batch = if has_batch { shape[0] } else { 1 };
    let (in_c, in_h, in_w) = if has_batch {
        (shape[1], shape[2], shape[3])
    } else {
        (shape[0], shape[1], shape[2])
    };
    let kshape = conv.kernel.shape();
    let (out_c, kc, kh, kw) = (kshape[0], kshape[1], kshape[2], kshape[3]);
    let groups = conv.groups;
    if in_c != kc * groups || out_c % groups != 0 {
        return Err(NyError::ShapeMismatch {
            expected: vec![kc * groups],
            got: vec![in_c],
        });
    }
    let (sh, sw) = conv.stride;
    let (ph, pw) = conv.padding;
    let (dh, dw) = conv.dilation;
    let eff_kh = (kh - 1) * dh + 1;
    let eff_kw = (kw - 1) * dw + 1;
    let out_h = (in_h + 2 * ph)
        .checked_sub(eff_kh)
        .ok_or_else(|| NyError::InvalidSpec("f64 cell eval: conv kernel too large".to_string()))?
        / sh
        + 1;
    let out_w = (in_w + 2 * pw)
        .checked_sub(eff_kw)
        .ok_or_else(|| NyError::InvalidSpec("f64 cell eval: conv kernel too large".to_string()))?
        / sw
        + 1;
    let out_c_per_group = out_c / groups;

    let n_terms = kc * kh * kw + 2;
    let gamma = gamma_n(n_terms)?;

    // Flat standard-layout copies for fast indexing.
    let x_lo_owned = x.lower.as_standard_layout();
    let x_hi_owned = x.upper.as_standard_layout();
    let x_lo = x_lo_owned.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: conv input not contiguous".to_string())
    })?;
    let x_hi = x_hi_owned.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: conv input not contiguous".to_string())
    })?;
    let kernel_owned = conv.kernel.as_standard_layout();
    let kernel = kernel_owned.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: conv kernel not contiguous".to_string())
    })?;

    let out_shape: Vec<usize> = if has_batch {
        vec![batch, out_c, out_h, out_w]
    } else {
        vec![out_c, out_h, out_w]
    };
    let out_len = batch * out_c * out_h * out_w;
    let mut out_lo = vec![0.0f64; out_len];
    let mut out_hi = vec![0.0f64; out_len];

    let in_hw = in_h * in_w;
    for b in 0..batch {
        let in_base_b = b * in_c * in_hw;
        for oc in 0..out_c {
            let group = oc / out_c_per_group;
            let ic_base = group * kc;
            let k_base_oc = oc * kc * kh * kw;
            let bias = conv
                .bias
                .as_ref()
                .map(|bias| bias[oc] as f64)
                .unwrap_or(0.0);
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut lo = bias;
                    let mut hi = bias;
                    let mut abs = bias.abs();
                    for ic_off in 0..kc {
                        let ic = ic_base + ic_off;
                        let in_base = in_base_b + ic * in_hw;
                        let k_base = k_base_oc + ic_off * kh * kw;
                        for r in 0..kh {
                            let ih = (oh * sh + r * dh) as isize - ph as isize;
                            if ih < 0 || ih >= in_h as isize {
                                continue;
                            }
                            let row_base = in_base + ih as usize * in_w;
                            let k_row = k_base + r * kw;
                            for c in 0..kw {
                                let iw = (ow * sw + c * dw) as isize - pw as isize;
                                if iw < 0 || iw >= in_w as isize {
                                    continue;
                                }
                                let w = kernel[k_row + c] as f64;
                                if w == 0.0 {
                                    continue;
                                }
                                let xi = row_base + iw as usize;
                                let (xl, xu) = (x_lo[xi], x_hi[xi]);
                                let (pl, pu) = if w >= 0.0 {
                                    (w * xl, w * xu)
                                } else {
                                    (w * xu, w * xl)
                                };
                                lo += pl;
                                hi += pu;
                                abs += pl.abs().max(pu.abs());
                            }
                        }
                    }
                    let err = gamma * abs;
                    let out_idx = ((b * out_c + oc) * out_h + oh) * out_w + ow;
                    out_lo[out_idx] = (lo - err).next_down();
                    out_hi[out_idx] = (hi + err).next_up();
                }
            }
        }
    }
    Ok(Interval64 {
        lower: ArrayD::from_shape_vec(IxDyn(&out_shape), out_lo)
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: conv out: {e}")))?,
        upper: ArrayD::from_shape_vec(IxDyn(&out_shape), out_hi)
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: conv out: {e}")))?,
    })
}

/// ScatterND with embedded constant data/updates and a dynamic indices input
/// whose per-cell interval is SINGLETON integers (the cctsdb cell case).
///
/// Rows with any coordinate out of range are SKIPPED — exactly matching the
/// clamped-window semantics of the source graph (design B4: the unclamped
/// static-max window emits sentinel rows the true graph never writes).
/// Non-singleton coordinates fail closed.
fn eval_scatter_nd(
    scatter: &crate::layers::ScatterNdLayer,
    inputs: &[Interval64],
) -> Result<Interval64> {
    let data = scatter.data_constant().ok_or_else(|| {
        NyError::UnsupportedOp("f64 cell eval: ScatterND requires constant data".to_string())
    })?;
    let updates = scatter.updates_constant().ok_or_else(|| {
        NyError::UnsupportedOp("f64 cell eval: ScatterND requires constant updates".to_string())
    })?;
    if scatter.has_static_indices() || inputs.len() != 1 {
        return Err(NyError::UnsupportedOp(
            "f64 cell eval: ScatterND expects exactly the dynamic indices input".to_string(),
        ));
    }
    let indices = &inputs[0];

    let data_shape = data.shape().to_vec();
    let idx_shape = indices.shape().to_vec();
    let index_depth = *idx_shape.last().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: ScatterND indices rank 0".to_string())
    })?;
    if index_depth == 0 || index_depth > data_shape.len() {
        return Err(NyError::InvalidSpec(
            "f64 cell eval: ScatterND index depth out of range".to_string(),
        ));
    }
    let prefix_elems: usize = idx_shape[..idx_shape.len() - 1].iter().product();
    let remainder_shape = &data_shape[index_depth..];
    let slice_len: usize = remainder_shape.iter().product();
    let expected_updates: Vec<usize> = idx_shape[..idx_shape.len() - 1]
        .iter()
        .copied()
        .chain(remainder_shape.iter().copied())
        .collect();
    if updates.shape() != expected_updates.as_slice() {
        return Err(NyError::ShapeMismatch {
            expected: expected_updates,
            got: updates.shape().to_vec(),
        });
    }

    // Strides of data.
    let mut strides = vec![1usize; data_shape.len()];
    for i in (1..data_shape.len()).rev() {
        strides[i - 1] = strides[i] * data_shape[i];
    }

    let idx_lo_owned = indices.lower.as_standard_layout();
    let idx_hi_owned = indices.upper.as_standard_layout();
    let idx_lo = idx_lo_owned
        .as_slice()
        .ok_or_else(|| NyError::InvalidSpec("f64 cell eval: indices not contiguous".to_string()))?;
    let idx_hi = idx_hi_owned
        .as_slice()
        .ok_or_else(|| NyError::InvalidSpec("f64 cell eval: indices not contiguous".to_string()))?;
    let updates_flat: Vec<f64> = updates.iter().map(|&v| v as f64).collect();

    let mut out: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    'rows: for row in 0..prefix_elems {
        let mut base = 0usize;
        for axis in 0..index_depth {
            let flat = row * index_depth + axis;
            let (lo, hi) = (idx_lo[flat], idx_hi[flat]);
            if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
                return Err(NyError::UnsupportedOp(
                    "f64 cell eval: ScatterND index interval not finite".to_string(),
                ));
            }
            // The TRUE coordinate is an integer (ONNX ScatterND indices are
            // int64; the source graph builds them from Cast-to-int outputs and
            // integer arithmetic) lying inside this sound enclosure. Per cell
            // the enclosure is only a few ulps wide (outward widening of the
            // affine coordinate chain), so its integer hull normally has
            // EXACTLY one member — that member IS the true coordinate.
            // Anything ambiguous fails closed.
            let ilo = lo.ceil();
            let ihi = hi.floor();
            if ilo != ihi {
                return Err(NyError::UnsupportedOp(
                    "f64 cell eval: ScatterND index interval not a unique integer".to_string(),
                ));
            }
            let axis_len = data_shape[axis] as i64;
            let mut coord = ilo as i64;
            if coord < 0 {
                coord += axis_len;
            }
            if coord < 0 || coord >= axis_len {
                // Out-of-range sentinel row (clamped window edge): no write.
                continue 'rows;
            }
            base += coord as usize * strides[axis];
        }
        for off in 0..slice_len {
            out[base + off] = updates_flat[row * slice_len + off];
        }
    }

    let arr = ArrayD::from_shape_vec(IxDyn(&data_shape), out)
        .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: scatter out: {e}")))?;
    Ok(Interval64 {
        lower: arr.clone(),
        upper: arr,
    })
}

#[cfg(test)]
mod tests {
    use super::super::core::graph::GraphNode;
    use super::*;
    use crate::layers::{
        AddConstantLayer, AddLayer, DivLayer, LinearLayer, MatMulLayer, MulBinaryLayer, ReLULayer,
        ReduceSumLayer, SigmoidLayer, SliceLayer, SubLayer, TanhLayer,
    };
    use ndarray::{arr1, arr2};

    #[test]
    fn interval_mul_covers_sign_cases() {
        let (lo, hi) = interval_mul(-2.0, 3.0, -5.0, 7.0);
        assert_eq!(lo, -15.0); // 3 * -5
        assert_eq!(hi, 21.0); // 3 * 7
    }

    #[test]
    fn gamma_grows_with_terms() {
        let g10 = gamma_n(10).unwrap();
        let g1000 = gamma_n(1000).unwrap();
        assert!(g10 > 0.0 && g1000 > g10);
        assert!(g1000 < 1e-12);
    }

    #[test]
    fn widen1_moves_outward() {
        let (lo, hi) = widen1(1.0, 1.0);
        assert!(lo < 1.0 && hi > 1.0);
    }

    // -----------------------------------------------------------------------
    // f64 leaf-escalation op set (#f64-leaf): Sigmoid/ReduceSum/MatMul/N-D
    // Linear over an mscn-shaped DAG.
    // -----------------------------------------------------------------------

    /// Deterministic pseudo-random stream (xorshift) — no extra dev-dep.
    struct Rng(u64);
    impl Rng {
        fn next_unit(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn box64(lo: [f32; 4], hi: [f32; 4]) -> Interval64 {
        Interval64::from_f32(&arr1(&lo).into_dyn(), &arr1(&hi).into_dyn())
    }

    /// mscn-shaped DAG: input [4] --Linear(4->3)--> ReLU --Mul(slice of
    /// input)--> Add(slice) --ReduceSum--> Div(by 2.5 + ReduceSum(relu)) -->
    /// Sigmoid. Exercises Linear (N-D convention on rank-1), ReLU, MulBinary,
    /// Add, ReduceSum, Div, Sigmoid, Slice.
    fn build_mscn_like_graph() -> GraphNetwork {
        let w = arr2(&[
            [0.5f32, -1.25, 2.0, 0.75],
            [-0.375, 1.5, -0.625, 1.0],
            [1.125, 0.25, -1.75, -0.5],
        ]);
        let b = arr1(&[0.125f32, -0.25, 0.5]);
        let linear = LinearLayer::new(w, Some(b)).unwrap();
        let slice = SliceLayer::new(0, 0, 3);

        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input("lin", Layer::Linear(linear)));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["lin".to_string()],
        ));
        g.add_node(GraphNode::from_input("head", Layer::Slice(slice)));
        g.add_node(GraphNode::binary(
            "mul",
            Layer::MulBinary(MulBinaryLayer),
            "relu",
            "head",
        ));
        g.add_node(GraphNode::binary(
            "add",
            Layer::Add(AddLayer),
            "mul",
            "head",
        ));
        g.add_node(GraphNode::new(
            "sum",
            Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
            vec!["add".to_string()],
        ));
        // Divisor: 2.5 + sum(relu) >= 2.5 > 0 (sign-definite).
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
        g.add_node(GraphNode::binary(
            "div",
            Layer::Div(DivLayer),
            "sum",
            "denom",
        ));
        g.add_node(GraphNode::new(
            "out",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["div".to_string()],
        ));
        g.set_output("out");
        g
    }

    /// Plain f64 (round-to-nearest) concrete forward of the same DAG — the
    /// sample oracle for the enclosure property test.
    fn mscn_like_concrete(x: &[f64; 4]) -> f64 {
        let w = [
            [0.5f64, -1.25, 2.0, 0.75],
            [-0.375, 1.5, -0.625, 1.0],
            [1.125, 0.25, -1.75, -0.5],
        ];
        let b = [0.125f64, -0.25, 0.5];
        let mut relu = [0.0f64; 3];
        for o in 0..3 {
            let mut s = b[o];
            for j in 0..4 {
                s += w[o][j] * x[j];
            }
            relu[o] = s.max(0.0);
        }
        let head = [x[0], x[1], x[2]];
        let mut sum = 0.0f64;
        let mut relu_sum = 0.0f64;
        for i in 0..3 {
            sum += relu[i] * head[i] + head[i];
            relu_sum += relu[i];
        }
        let div = sum / (2.5 + relu_sum);
        stable_sigmoid_f64(div)
    }

    /// ENCLOSURE: every sampled concrete forward (and every box corner) lies
    /// inside the f64 interval on the mscn-shaped DAG over a wide box.
    #[test]
    fn f64_cell_encloses_sampled_forwards_mscn_dag() {
        let g = build_mscn_like_graph();
        assert!(g.supports_ibp_f64_cell(), "test DAG must be supported");
        let lo = [-0.5f32, 0.25, -1.0, 0.125];
        let hi = [0.75f32, 1.5, -0.25, 0.875];
        let out = g.propagate_ibp_f64_cell(&box64(lo, hi)).unwrap();
        let (out_l, out_u) = (out.lower[[0]], out.upper[[0]]);
        assert!(out_l <= out_u);

        let mut rng = Rng(0x9E3779B97F4A7C15);
        for _ in 0..2000 {
            let mut x = [0.0f64; 4];
            for i in 0..4 {
                let (l, h) = (f64::from(lo[i]), f64::from(hi[i]));
                x[i] = l + (h - l) * rng.next_unit();
            }
            let y = mscn_like_concrete(&x);
            assert!(
                out_l <= y && y <= out_u,
                "sample {y} escapes f64 interval [{out_l}, {out_u}] at x={x:?}"
            );
        }
        for mask in 0..16u32 {
            let mut x = [0.0f64; 4];
            for i in 0..4 {
                x[i] = if mask & (1 << i) != 0 {
                    f64::from(hi[i])
                } else {
                    f64::from(lo[i])
                };
            }
            let y = mscn_like_concrete(&x);
            assert!(
                out_l <= y && y <= out_u,
                "corner {y} escapes f64 interval [{out_l}, {out_u}]"
            );
        }
    }

    /// ENCLOSURE on a MatMul/Mul/Add/ReLU DAG with a live rank-2 MatMul
    /// (both operands perturbed).
    #[test]
    fn f64_cell_encloses_matmul_dag() {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::new(
            "mm",
            Layer::MatMul(MatMulLayer::new(false, None)),
            vec![NETWORK_INPUT.to_string(), NETWORK_INPUT.to_string()],
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["mm".to_string()],
        ));
        g.add_node(GraphNode::new(
            "mul",
            Layer::MulBinary(MulBinaryLayer),
            vec!["relu".to_string(), NETWORK_INPUT.to_string()],
        ));
        g.add_node(GraphNode::new(
            "out",
            Layer::Add(AddLayer),
            vec!["mul".to_string(), NETWORK_INPUT.to_string()],
        ));
        g.set_output("out");
        assert!(g.supports_ibp_f64_cell());

        let lo = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-0.5f32, 0.25, -0.75, 0.5]).unwrap();
        let hi = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.5f32, 1.0, 0.25, 1.25]).unwrap();
        let input = Interval64::from_f32(&lo, &hi);
        let out = g.propagate_ibp_f64_cell(&input).unwrap();

        let mut rng = Rng(0xDEADBEEFCAFEF00D);
        for _ in 0..2000 {
            let mut x = [[0.0f64; 2]; 2];
            for i in 0..2 {
                for j in 0..2 {
                    let l = f64::from(lo[[i, j]]);
                    let h = f64::from(hi[[i, j]]);
                    x[i][j] = l + (h - l) * rng.next_unit();
                }
            }
            for i in 0..2 {
                for j in 0..2 {
                    let mm = x[i][0] * x[0][j] + x[i][1] * x[1][j];
                    let y = mm.max(0.0) * x[i][j] + x[i][j];
                    let (l, u) = (out.lower[[i, j]], out.upper[[i, j]]);
                    assert!(
                        l <= y && y <= u,
                        "sample {y} escapes [{l}, {u}] at [{i},{j}]"
                    );
                }
            }
        }
    }

    /// TIGHTNESS: on a degenerate (point) box the interval must be within a
    /// few ulps of the concrete forward — the whole point of the f64 leaf
    /// escalation is that this width sits far below f32's ~1e-7 floor.
    #[test]
    fn f64_cell_point_box_is_ulp_tight() {
        let g = build_mscn_like_graph();
        let p = [0.375f32, 0.625, -0.5, 0.25];
        let out = g.propagate_ibp_f64_cell(&box64(p, p)).unwrap();
        let (l, u) = (out.lower[[0]], out.upper[[0]]);
        let y = mscn_like_concrete(&[
            f64::from(p[0]),
            f64::from(p[1]),
            f64::from(p[2]),
            f64::from(p[3]),
        ]);
        assert!(l <= y && y <= u, "concrete {y} outside [{l}, {u}]");
        let width = u - l;
        assert!(
            width < 1e-12,
            "point-box width {width} not ulp-tight (f32 floor is ~1e-7)"
        );
        assert!(width < f64::from(f32::EPSILON) * y.abs().max(1.0));
    }

    /// Div with a divisor interval straddling zero fails closed.
    #[test]
    fn f64_cell_div_straddling_zero_fails_closed() {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::new(
            "d",
            Layer::Div(DivLayer),
            vec![NETWORK_INPUT.to_string(), NETWORK_INPUT.to_string()],
        ));
        g.set_output("d");
        assert!(g.supports_ibp_f64_cell()); // statically supported...
        let input = box64([-1.0; 4], [1.0; 4]);
        // ...but the runtime divisor-sign check fails closed for this box.
        assert!(g.propagate_ibp_f64_cell(&input).is_err());
    }

    /// Sub interval direction: [a]-[b] uses (a_l - b_u, a_u - b_l).
    #[test]
    fn f64_cell_sub_interval_direction() {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "head",
            Layer::Slice(SliceLayer::new(0, 0, 2)),
        ));
        g.add_node(GraphNode::from_input(
            "tail",
            Layer::Slice(SliceLayer::new(0, 2, 4)),
        ));
        g.add_node(GraphNode::binary(
            "sub",
            Layer::Sub(SubLayer),
            "head",
            "tail",
        ));
        g.set_output("sub");
        let input = box64([1.0, 2.0, 0.25, 0.5], [1.5, 3.0, 0.75, 1.0]);
        let out = g.propagate_ibp_f64_cell(&input).unwrap();
        // head[0]-tail[0]: [1-0.75, 1.5-0.25] = [0.25, 1.25] (±1 ulp outward)
        assert!(out.lower[[0]] <= 0.25 && out.lower[[0]] > 0.25 - 1e-12);
        assert!(out.upper[[0]] >= 1.25 && out.upper[[0]] < 1.25 + 1e-12);
    }

    /// Unsupported op (Tanh) is rejected by BOTH the static gate and the walk.
    #[test]
    fn f64_cell_fails_closed_on_unsupported_op() {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input("t", Layer::Tanh(TanhLayer)));
        g.set_output("t");
        assert!(!g.supports_ibp_f64_cell());
        let input = box64([0.0; 4], [1.0; 4]);
        assert!(g.propagate_ibp_f64_cell(&input).is_err());
    }

    /// Sigmoid saturation regression: at x = -710 the naive 1/(1+e^-x)
    /// overflows exp and collapses to 0.0 with a non-enclosing widened upper;
    /// the stable form must still enclose the true value ~4.47e-309.
    #[test]
    fn f64_cell_sigmoid_extreme_negative_still_encloses() {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "s",
            Layer::Sigmoid(SigmoidLayer::new()),
        ));
        g.set_output("s");
        let x = -710.0f32;
        let input = Interval64::from_f32(&arr1(&[x]).into_dyn(), &arr1(&[x]).into_dyn());
        let out = g.propagate_ibp_f64_cell(&input).unwrap();
        let truth = f64::from(x).exp(); // σ(x) ≈ e^x for very negative x
        assert!(truth > 0.0, "test needs a representable subnormal truth");
        assert!(
            out.lower[[0]] <= truth && truth <= out.upper[[0]],
            "σ(-710)≈{truth:e} escapes [{:e}, {:e}]",
            out.lower[[0]],
            out.upper[[0]]
        );
        assert!(out.lower[[0]] >= 0.0);
    }

    /// ReduceSum keepdims=false and multi-axis resolution stay sound/exact
    /// on a point box.
    #[test]
    fn f64_cell_reduce_sum_axes_and_keepdims() {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "sum",
            Layer::ReduceSum(ReduceSumLayer::new(vec![0], false)),
        ));
        g.set_output("sum");
        let vals = arr1(&[1.5f32, -2.25, 4.0, 0.125]).into_dyn();
        let input = Interval64::from_f32(&vals, &vals);
        let out = g.propagate_ibp_f64_cell(&input).unwrap();
        assert_eq!(out.lower.shape(), &[] as &[usize]);
        let truth = 1.5 - 2.25 + 4.0 + 0.125;
        let l = *out.lower.iter().next().unwrap();
        let u = *out.upper.iter().next().unwrap();
        assert!(l <= truth && truth <= u);
        assert!(u - l < 1e-13);
    }

    // -----------------------------------------------------------------------
    // Fast Rump-GEMM wiring (#f64-blas-gemm): above-threshold Linear/MatMul
    // take the BLAS midpoint-radius kernel. The layer results must ENCLOSE
    // the scalar-path results AND sampled true products. (The assertions
    // also hold if NY_F64_BLAS=0 forces the scalar path — enclosure is
    // path-independent; the kernel itself is unit-gated in
    // graph_ibp_f64_gemm.)
    // -----------------------------------------------------------------------

    fn random_interval64(rng: &mut Rng, shape: &[usize], width: f64) -> Interval64 {
        let n: usize = shape.iter().product();
        let mut lo = Vec::with_capacity(n);
        let mut hi = Vec::with_capacity(n);
        for _ in 0..n {
            let c = rng.next_unit() * 2.0 - 1.0;
            let w = rng.next_unit() * width;
            lo.push(c - w);
            hi.push(c + w);
        }
        Interval64 {
            lower: ArrayD::from_shape_vec(IxDyn(shape), lo).unwrap(),
            upper: ArrayD::from_shape_vec(IxDyn(shape), hi).unwrap(),
        }
    }

    fn sample_inside64(rng: &mut Rng, x: &Interval64) -> ArrayD<f64> {
        let mut out = x.lower.clone();
        for (o, (&l, &h)) in out.iter_mut().zip(x.lower.iter().zip(x.upper.iter())) {
            *o = l + (h - l) * rng.next_unit();
        }
        out
    }

    /// Rank-3 Linear above the fast-path threshold: fast wiring ⊇ scalar
    /// path ⊇ sampled x@W^T + b (both bias and no-bias, the MVF case).
    #[test]
    fn linear_fast_path_encloses_scalar_and_samples() {
        let (d0, d1, in_dim, out_dim) = (4usize, 12usize, 96usize, 64usize);
        assert!(
            d0 * d1 * in_dim * out_dim > 32_768,
            "test must cross the threshold"
        );
        let mut rng = Rng(0xC0FFEE0DDF00DBAD);
        // Heap-allocated: a [[f32; 96]; 64] literal is a 24 KiB stack array.
        let mut w = ndarray::Array2::<f32>::zeros((64, 96));
        for v in w.iter_mut() {
            *v = (rng.next_unit() * 2.0 - 1.0) as f32;
        }
        let mut b = arr1(&[0f32; 64]);
        for v in b.iter_mut() {
            *v = (rng.next_unit() * 2.0 - 1.0) as f32;
        }
        let linear = LinearLayer::new(w.clone(), Some(b.clone())).unwrap();
        let x = random_interval64(&mut rng, &[d0, d1, in_dim], 0.25);

        for include_bias in [true, false] {
            let fast = eval_linear_with_bias(&linear, &x, include_bias).unwrap();
            let scalar = eval_linear_with_bias_scalar(&linear, &x, include_bias).unwrap();
            assert_eq!(fast.lower.shape(), &[d0, d1, out_dim]);
            for (f, s) in fast.lower.iter().zip(scalar.lower.iter()) {
                assert!(f <= s, "fast lower {f} above scalar lower {s}");
            }
            for (f, s) in fast.upper.iter().zip(scalar.upper.iter()) {
                assert!(f >= s, "fast upper {f} below scalar upper {s}");
            }
            for _ in 0..50 {
                let xs = sample_inside64(&mut rng, &x);
                for i0 in 0..d0 {
                    for i1 in 0..d1 {
                        for o in 0..out_dim {
                            let mut y = if include_bias { f64::from(b[o]) } else { 0.0 };
                            for i in 0..in_dim {
                                y += f64::from(w[[o, i]]) * xs[[i0, i1, i]];
                            }
                            assert!(
                                fast.lower[[i0, i1, o]] <= y && y <= fast.upper[[i0, i1, o]],
                                "sample {y} escapes fast Linear interval"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Timing probe at the REAL nn4sys mscn_2048d shapes (ignored; run with
    /// --ignored --nocapture): scalar Linear vs the fast path INCLUDING its
    /// per-call weight conversion vs the kernel with pre-converted weights.
    #[test]
    #[ignore = "manual timing probe"]
    fn bench_linear_fast_vs_scalar_mscn_shapes() {
        use std::time::Instant;
        let mut rng = Rng(0xB7E151628AED2A6A);
        for &(m, k, n) in &[
            (6usize, 2048usize, 2048usize),
            (3, 2048, 2048),
            (2, 2048, 2048),
            (1, 6144, 2048),
            (6, 13, 2048),
            (22, 2048, 2048),
            (64, 2048, 2048),
        ] {
            let mut w = ndarray::Array2::<f32>::zeros((n, k));
            for v in w.iter_mut() {
                *v = (rng.next_unit() * 2.0 - 1.0) as f32;
            }
            let linear = LinearLayer::new(w.clone(), None).unwrap();
            let x = random_interval64(&mut rng, &[m, k], 1e-6);
            let reps = 20u32;

            let t0 = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(eval_linear_with_bias_scalar(&linear, &x, true).unwrap());
            }
            let scalar = t0.elapsed() / reps;

            let t1 = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(eval_linear_with_bias(&linear, &x, true).unwrap());
            }
            let fast = t1.elapsed() / reps;

            // Kernel-only: pre-converted weights (what a per-run cache buys).
            let wt = linear.weight.t().map(|&v| f64::from(v));
            let a_lo = x.lower.view().into_dimensionality::<Ix2>().unwrap();
            let a_hi = x.upper.view().into_dimensionality::<Ix2>().unwrap();
            let t2 = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(
                    rump_interval_matmul(a_lo.view(), a_hi.view(), wt.view(), wt.view()).unwrap(),
                );
            }
            let kernel = t2.elapsed() / reps;
            println!(
                "[{m}x{k}x{n}] scalar {scalar:?} fast(with-convert) {fast:?} \
                 kernel(preconverted) {kernel:?}"
            );
        }
    }

    /// Above-threshold MatMul (plain / transpose_b / scaled): fast wiring
    /// ⊇ scalar path ⊇ sampled true products.
    #[test]
    fn matmul_fast_path_encloses_scalar_and_samples() {
        let (m, k, n) = (40usize, 64usize, 40usize);
        assert!(m * k * n > 32_768, "test must cross the threshold");
        let mut rng = Rng(0x0DDBA11FADEDFACE);
        for (transpose_b, scale) in [(false, None), (true, None), (false, Some(0.125f32))] {
            let layer = MatMulLayer::new(transpose_b, scale);
            let a = random_interval64(&mut rng, &[m, k], 0.5);
            let b_shape = if transpose_b { [n, k] } else { [k, n] };
            let b = random_interval64(&mut rng, &b_shape, 0.5);

            let fast = eval_matmul(&layer, &a, &b).unwrap();
            let scalar = eval_matmul_scalar(&layer, &a, &b).unwrap();
            for (f, s) in fast.lower.iter().zip(scalar.lower.iter()) {
                assert!(f <= s, "fast lower {f} above scalar lower {s}");
            }
            for (f, s) in fast.upper.iter().zip(scalar.upper.iter()) {
                assert!(f >= s, "fast upper {f} below scalar upper {s}");
            }
            let s64 = scale.map_or(1.0, f64::from);
            for _ in 0..50 {
                let a_s = sample_inside64(&mut rng, &a);
                let b_s = sample_inside64(&mut rng, &b);
                for i in 0..m {
                    for j in 0..n {
                        let mut y = 0.0f64;
                        for l in 0..k {
                            let bv = if transpose_b {
                                b_s[[j, l]]
                            } else {
                                b_s[[l, j]]
                            };
                            y += a_s[[i, l]] * bv;
                        }
                        y *= s64;
                        assert!(
                            fast.lower[[i, j]] <= y && y <= fast.upper[[i, j]],
                            "sample {y} escapes fast MatMul interval at [{i},{j}]"
                        );
                    }
                }
            }
        }
    }
    /// Reference `broadcast_binary` (the pre-flat implementation, verbatim):
    /// `indexed_iter` over the broadcast views with per-element indexed
    /// lookups. The flat production path must be BIT-IDENTICAL to this on
    /// every input — same values, same error behavior.
    fn broadcast_binary_indexed_reference(
        a: &Interval64,
        b: &Interval64,
        exact: bool,
        combine: impl Fn(f64, f64, f64, f64) -> Result<(f64, f64)>,
    ) -> Result<Interval64> {
        let out_shape = crate::shape::broadcast_shapes(a.lower.shape(), b.lower.shape())
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: a.lower.shape().to_vec(),
                got: b.lower.shape().to_vec(),
            })?;
        let bc_err = || NyError::InvalidSpec("f64 cell eval: broadcast failed".to_string());
        let alo = a.lower.broadcast(IxDyn(&out_shape)).ok_or_else(bc_err)?;
        let ahi = a.upper.broadcast(IxDyn(&out_shape)).ok_or_else(bc_err)?;
        let blo = b.lower.broadcast(IxDyn(&out_shape)).ok_or_else(bc_err)?;
        let bhi = b.upper.broadcast(IxDyn(&out_shape)).ok_or_else(bc_err)?;

        let mut out_lo = ArrayD::zeros(IxDyn(&out_shape));
        let mut out_hi = ArrayD::zeros(IxDyn(&out_shape));
        for (idx, &al) in alo.indexed_iter() {
            let (mut lo, mut hi) =
                combine(al, ahi[idx.clone()], blo[idx.clone()], bhi[idx.clone()])?;
            if !exact {
                (lo, hi) = widen1(lo, hi);
            }
            out_lo[idx.clone()] = lo;
            out_hi[idx] = hi;
        }
        Ok(Interval64 {
            lower: out_lo,
            upper: out_hi,
        })
    }

    /// Bit-identity gate for the flat `broadcast_binary` (#f64-flat-elemwise):
    /// across same-shape, scalar-broadcast, axis-broadcast, non-contiguous
    /// (transposed view), and empty operands, the flat path must produce
    /// bit-identical outputs to the indexed reference for every elementwise
    /// rule used by `eval_node` (add / sub / interval mul / min-exact), and
    /// identical error behavior for a failing `combine`.
    #[test]
    fn broadcast_binary_flat_matches_indexed_reference() {
        let mut rng = Rng(0x0DDB_1773_57A7_E001);
        let mk = |rng: &mut Rng, shape: &[usize], width: f64| -> Interval64 {
            let n: usize = shape.iter().product();
            let mut lo = Vec::with_capacity(n);
            let mut hi = Vec::with_capacity(n);
            for _ in 0..n {
                let c = rng.next_unit() * 4.0 - 2.0;
                let w = rng.next_unit() * width;
                lo.push(c - w);
                hi.push(c + w);
            }
            Interval64 {
                lower: ArrayD::from_shape_vec(IxDyn(shape), lo).unwrap(),
                upper: ArrayD::from_shape_vec(IxDyn(shape), hi).unwrap(),
            }
        };
        // Transposed (non-standard-layout) variant of an interval.
        let transpose2 = |x: &Interval64| -> Interval64 {
            Interval64 {
                lower: x.lower.clone().permuted_axes(IxDyn(&[1, 0])),
                upper: x.upper.clone().permuted_axes(IxDyn(&[1, 0])),
            }
        };

        let shapes: &[(&[usize], &[usize])] = &[
            (&[3, 128], &[3, 128]), // same shape (dominant mscn case)
            (&[3, 128], &[128]),    // trailing-axis broadcast
            (&[3, 1], &[1, 4]),     // two-sided broadcast
            (&[1], &[2, 3]),        // scalar-vs-tensor
            (&[2, 3, 4], &[3, 1]),  // rank + axis broadcast
            (&[0, 4], &[4]),        // empty output
        ];
        type Rule = (bool, fn(f64, f64, f64, f64) -> Result<(f64, f64)>);
        let rules: &[Rule] = &[
            (false, |al, ah, bl, bh| Ok((al + bl, ah + bh))),
            (false, |al, ah, bl, bh| Ok((al - bh, ah - bl))),
            (false, |al, ah, bl, bh| Ok(interval_mul(al, ah, bl, bh))),
            (true, |al, ah, bl, bh| Ok((al.min(bl), ah.min(bh)))),
        ];
        for &(sa, sb) in shapes {
            let a = mk(&mut rng, sa, 0.5);
            let b = mk(&mut rng, sb, 0.5);
            for (r, &(exact, rule)) in rules.iter().enumerate() {
                let flat = broadcast_binary(&a, &b, exact, rule).unwrap();
                let reference = broadcast_binary_indexed_reference(&a, &b, exact, rule).unwrap();
                let bits = |x: &ArrayD<f64>| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert_eq!(
                    bits(&flat.lower),
                    bits(&reference.lower),
                    "rule {r} {sa:?}x{sb:?}: flat lower diverged"
                );
                assert_eq!(
                    bits(&flat.upper),
                    bits(&reference.upper),
                    "rule {r} {sa:?}x{sb:?}: flat upper diverged"
                );
            }
        }

        // Non-contiguous operands (transposed views) still agree bitwise.
        let a = mk(&mut rng, &[4, 6], 0.3);
        let b = mk(&mut rng, &[6, 4], 0.3);
        let bt = transpose2(&b);
        let flat = broadcast_binary(&a, &bt, false, |al, ah, bl, bh| {
            Ok(interval_mul(al, ah, bl, bh))
        })
        .unwrap();
        let reference = broadcast_binary_indexed_reference(&a, &bt, false, |al, ah, bl, bh| {
            Ok(interval_mul(al, ah, bl, bh))
        })
        .unwrap();
        assert_eq!(
            flat.lower.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            reference
                .lower
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            flat.upper.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            reference
                .upper
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );

        // Error parity: a combine that fails on a specific element (Div-style
        // divisor check) errors on both paths with the same message.
        let a = mk(&mut rng, &[2, 3], 0.5);
        let mut b = mk(&mut rng, &[2, 3], 0.0);
        b.lower[[1, 1]] = -1.0;
        b.upper[[1, 1]] = 1.0;
        let div_rule = |al: f64, ah: f64, bl: f64, bh: f64| -> Result<(f64, f64)> {
            if bl <= 0.0 && bh >= 0.0 {
                return Err(NyError::InvalidSpec("divisor straddles zero".to_string()));
            }
            Ok((al / bl.max(bh), ah / bl.min(bh)))
        };
        let flat_err = broadcast_binary(&a, &b, false, div_rule).unwrap_err();
        let ref_err = broadcast_binary_indexed_reference(&a, &b, false, div_rule).unwrap_err();
        assert_eq!(flat_err.to_string(), ref_err.to_string());

        // Shape-mismatch parity.
        let a = mk(&mut rng, &[2, 3], 0.1);
        let b = mk(&mut rng, &[2, 4], 0.1);
        assert!(broadcast_binary(&a, &b, true, |al, ah, _, _| Ok((al, ah))).is_err());
        assert!(
            broadcast_binary_indexed_reference(&a, &b, true, |al, ah, _, _| Ok((al, ah))).is_err()
        );
    }

    // -----------------------------------------------------------------------
    // Flatten (#sb-rebank lever 2): exact data movement, whitelist coverage.
    // -----------------------------------------------------------------------

    /// Flatten is EXACT data movement: bit-identical endpoints, row-major order
    /// preserved, ONNX 2-D collapse semantics for axis 0 / mid / negative /
    /// rank axes. This is the arm that unblocks `supports_ibp_f64_cell` on the
    /// soundnessbench net (Conv x6 -> Flatten -> Gemm x2).
    #[test]
    fn flatten_is_exact_data_movement_and_supported() {
        use crate::layers::FlattenLayer;

        // Whitelist coverage: the support predicate and the eval arm must stay
        // in sync (the escalation gate fails closed otherwise).
        assert!(cell_supports_layer(&Layer::Flatten(FlattenLayer::new(1))));

        let mut rng = Rng(0x5b_f1a7);
        let vals: Vec<f64> = (0..24).map(|_| rng.next_unit() * 2.0 - 1.0).collect();
        let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), vals).unwrap();
        let upper = lower.mapv(|v| v + 0.5);
        let x = Interval64 {
            lower: lower.clone(),
            upper: upper.clone(),
        };

        for (axis, want_shape) in [
            (0i32, vec![1usize, 24]),
            (1, vec![2, 12]),
            (2, vec![6, 4]),
            (3, vec![24, 1]),
            (-1, vec![6, 4]),
        ] {
            let mut g = GraphNetwork::new();
            g.add_node(GraphNode::from_input(
                "flat",
                Layer::Flatten(FlattenLayer::new(axis)),
            ));
            g.set_output("flat");
            assert!(g.supports_ibp_f64_cell(), "axis {axis}: gate must open");
            let out = g.propagate_ibp_f64_cell(&x).unwrap();
            assert_eq!(out.lower.shape(), &want_shape[..], "axis {axis}");
            // Bit-exact row-major copy: no widening, no reordering.
            let got_lo: Vec<u64> = out.lower.iter().map(|v| v.to_bits()).collect();
            let want_lo: Vec<u64> = lower.iter().map(|v| v.to_bits()).collect();
            assert_eq!(got_lo, want_lo, "axis {axis}: lower endpoints moved");
            let got_hi: Vec<u64> = out.upper.iter().map(|v| v.to_bits()).collect();
            let want_hi: Vec<u64> = upper.iter().map(|v| v.to_bits()).collect();
            assert_eq!(got_hi, want_hi, "axis {axis}: upper endpoints moved");
        }
    }

    /// Flatten feeding a Linear (the soundnessbench Conv->Flatten->Gemm shape):
    /// the cell walk through Flatten must agree with manually pre-flattened
    /// input on the SAME Linear — bit-identical bounds.
    #[test]
    fn flatten_then_linear_matches_preflattened() {
        use crate::layers::FlattenLayer;

        let w = arr2(&[
            [0.5f32, -1.25, 2.0, 0.75, -0.375, 1.5],
            [1.125, 0.25, -1.75, -0.5, 0.625, -1.0],
        ]);
        let b = arr1(&[0.125f32, -0.25]);

        // Graph A: rank-3 input [1,2,3] -> Flatten(1) -> Linear(6->2).
        let mut ga = GraphNetwork::new();
        ga.add_node(GraphNode::from_input(
            "flat",
            Layer::Flatten(FlattenLayer::new(1)),
        ));
        ga.add_node(GraphNode::new(
            "lin",
            Layer::Linear(LinearLayer::new(w.clone(), Some(b.clone())).unwrap()),
            vec!["flat".to_string()],
        ));
        ga.set_output("lin");

        // Graph B: the identical Linear on the already-flat [1,6] input.
        let mut gb = GraphNetwork::new();
        gb.add_node(GraphNode::from_input(
            "lin",
            Layer::Linear(LinearLayer::new(w, Some(b)).unwrap()),
        ));
        gb.set_output("lin");

        let mut rng = Rng(0xf1a77e4);
        for _ in 0..8 {
            let vals: Vec<f64> = (0..6).map(|_| rng.next_unit() * 4.0 - 2.0).collect();
            let point3 = ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), vals.clone()).unwrap();
            let point2 = ArrayD::from_shape_vec(IxDyn(&[1, 6]), vals).unwrap();
            let oa = ga
                .propagate_ibp_f64_cell(&Interval64::point(point3))
                .unwrap();
            let ob = gb
                .propagate_ibp_f64_cell(&Interval64::point(point2))
                .unwrap();
            let bits = |a: &ArrayD<f64>| a.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&oa.lower), bits(&ob.lower));
            assert_eq!(bits(&oa.upper), bits(&ob.upper));
        }
    }
}
