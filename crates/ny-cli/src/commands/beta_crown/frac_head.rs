// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Structural verifier for "normalized-power fractional head" DAGs
//! (nn4sys pensieve `*_parallel`).
//!
//! Recognized shape (per head, two heads subtracted at the output):
//!
//! ```text
//!   r = ReLU(...)                      (logits, r >= 0, dim n)
//!   p = r^k                            (PowConstant, integer k >= 1)
//!   D = ReduceSum(p)                   (scalar)
//!   w = p / D                          (Div: normalized weights, sum(w) = 1)
//!   s = Linear(w) = sum_i c_i w_i + b  (weight row c, scalar)
//!   Y = s_A - s_B                      (Sub = graph output, dim 1)
//! ```
//!
//! Generic interval/CROWN propagation through the `Div` is catastrophically
//! loose here (pensieve root gap ~120 units vs a true output range under one
//! unit wide) because it drops the numerator/denominator correlation.
//! (Historically the generic graph pipeline ALSO mis-evaluated this head
//! outright: the legacy batch-squeeze `axis-1` adjustment turned the ONNX
//! `ReduceSum(axes=[1])` into a size-1-axis no-op on the runtime `[1, n]`
//! tensors, so the graph forward computed `w = p / p = 1` and both heads
//! collapsed to the constant `Σ c_i` — graph `Y = 0` vs ORT `Y ≈ 6.5`. Fixed:
//! reduction axes are now stored trailing-relative, see
//! `ConvertContext::remap_axis_trailing` in ny-build; the graph forward now
//! matches ORT on this family.) This module still treats the head
//! ANALYTICALLY for tightness and validates its reading against the trusted
//! ONNX-Runtime forward before activating.
//!
//! Two complementary sound bounds are intersected per box:
//!
//! 1. **Threshold-spec** (correlation-aware): for `D = Σ p_i > 0`,
//!    `s >= t  ⇔  g_t := Σ (c_i - t) p_i >= 0` (multiply the mediant through
//!    by the positive denominator). Each `g_t` is one CROWN spec row over
//!    the pow graph (prefix + `p = r^k`), so the linear backward captures
//!    inter-logit correlation; a threshold grid in both directions plus an
//!    all-ones row (denominator positivity witness) is evaluated in a single
//!    backward, then refined a stage deeper. Row entries are rounded toward
//!    the sound side (`w_i <= c_i - t` for lower claims, `>=` for upper).
//! 2. **Vertex** (relaxation-free in `p`-space): with interval bounds
//!    `p ∈ [pl, pu]`, the EXACT range of `s = Σ c_i p_i / Σ p_i` over the box
//!    is attained at "threshold vertices" of the coefficient-sorted order
//!    (∂s/∂p_i has the sign of `c_i - s`); all `n+1` sorted-order cuts are
//!    enumerated with outward f64 rounding. Tighter than (1) on wide boxes
//!    where the `r^k` secant relaxation is loose.
//!
//! The default bound path FUSES both: the threshold-grid CROWN pass carries
//! `n` identity rows (per-logit `p_i` boxes) and the all-ones row (a
//! correlation-aware `Σ p_i` range), so no separate prefix pass is needed
//! per refinement step (~2x eval throughput), and the vertex range is
//! solved over the denominator-constrained box
//! `p ∈ [pl, pu] ∩ {Σ p_i ∈ [D_lo, D_hi]}` (exact 1-budget box LP via
//! Dinkelbach bisection — recovers part of the box-level correlation loss).
//! Child leaves refine on a GEOMETRIC threshold grid anchored at the
//! inherited claim (finest step `W/(P-1)^2` at the current bound), so
//! sub-uniform-grid improvements keep accumulating instead of freezing the
//! frontier. `NY_FRAC_HEAD_CLASSIC=1` reverts to the pre-fusion path.
//!
//! `Y = s_A - s_B` concretizes by interval subtraction — lossless for the
//! pensieve family because the two heads read disjoint input halves.
//!
//! Logit bounds come from the existing graph CROWN machinery run on the
//! per-head PREFIX subgraph (input → logit ReLU / pow). The prefixes contain
//! only Linear/Conv/ReLU/Slice/Gather/Concat/Flatten/Reshape (+PowConstant)
//! nodes, are well under the per-node CROWN-IBP threshold, and inherit the
//! engine's soundness contract. Boxes that no method can bound stay
//! unbounded and are refined first; they can never be claimed.
//!
//! The driver runs one small input-split refinement per head (the heads are
//! independent, so refining `min s_A` and `max s_B` separately converges
//! without the joint-box product blowup); children inherit their parent's
//! claims (subset boxes). It checks the vnnlib constraints against the
//! covering `Y` interval each round and fails open (`None`, normal pipeline
//! continues) on any structural mismatch, arithmetic guard, deadline, or
//! budget cap. SAT is only ever claimed from a concrete trusted-forward
//! violation — and the vnncomp harness re-validates any witness against
//! ONNX-Runtime downstream before scoring.
//!
//! Disable with `NY_NO_FRAC_HEAD=1` (batteries-included: default ON).

use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;

use ndarray::{Array2, ArrayD, IxDyn};
use ny_onnx::vnnlib::VnnLibSpec;
use ny_propagate::{BabVerificationStatus, BetaCrownResult, GraphNetwork, Layer, NETWORK_INPUT};
use rayon::prelude::*;
use tracing::{debug, info, warn};

use super::cell_enum::{box_definitely_safe, concrete_violates};
use super::BetaCrownModel;

/// Maximum logit width `n` accepted by the detector (pensieve uses 6).
const MAX_LOGITS: usize = 64;
/// Maximum integer exponent accepted for the PowConstant (pensieve uses 3).
const MAX_EXPONENT: u32 = 8;
/// Leaves refined per net per round (children bounded in parallel).
const REFINE_BATCH: usize = 16;
/// Hard cap on live leaves per net.
const MAX_LEAVES: usize = 200_000;
/// Hard cap on refinement rounds.
const MAX_ROUNDS: usize = 100_000;
/// Relative tolerance of the concrete structural self-check.
const SELF_CHECK_REL_TOL: f64 = 1e-2;
/// A dimension narrower than `root_width * 2^-24` is no longer split.
const MIN_REL_WIDTH: f64 = 1.0 / (1u64 << 24) as f64;

/// Entry point: decide a normalized-power fractional-head instance, or
/// `None` to fall through to the normal pipeline (always sound).
pub(super) fn try_frac_head_verification(
    model_net: &BetaCrownModel,
    input_shape: &[usize],
    vnnlib: &VnnLibSpec,
    deadline: Instant,
) -> Option<BetaCrownResult> {
    if std::env::var_os("NY_NO_FRAC_HEAD").is_some_and(|v| v == "1") {
        return None;
    }
    let BetaCrownModel::Graph(graph) = model_net else {
        return None;
    };
    let start = Instant::now();

    let plan = detect(graph, input_shape, vnnlib)?;
    info!(
        "Fractional-head verifier qualifies: n={} logits, k={}, {} free dim(s) \
         (netA {}, netB {})",
        plan.heads[0].coeffs.len(),
        plan.heads[0].exponent,
        plan.free_dims,
        plan.heads[0].dims.len(),
        plan.heads[1].dims.len(),
    );

    // Cheap SAT probe at the root center: a definitely-violating f32 interval
    // enclosure at a point is a real counterexample (re-confirmed concretely
    // here AND by the vnncomp ORT gate downstream).
    if let Some(result) = violated_at_center(graph, &plan, vnnlib, start) {
        return Some(result);
    }

    run_refinement(&plan, vnnlib, deadline, start)
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/// One recognized head: prefix graph (input → logit ReLU) plus the analytic
/// tail parameters.
struct HeadPlan {
    prefix: GraphNetwork,
    /// Prefix extended with the PowConstant node (output `p = r^k`); the
    /// threshold-spec bound runs CROWN on this graph so the linear backward
    /// captures inter-logit correlation that per-logit boxes lose.
    pow_graph: GraphNetwork,
    /// Linear row `c` applied to the normalized weights (f64-widened).
    coeffs: Vec<f64>,
    /// Linear bias (0.0 when absent).
    bias: f64,
    /// PowConstant exponent `k`.
    exponent: u32,
    /// Splittable input dims for this head with their influence scores
    /// (per-dim prefix IBP output width — exactly 0 for unreachable dims).
    dims: Vec<(usize, f64)>,
}

struct FracHeadPlan {
    input_shape: Vec<usize>,
    root_lo: Vec<f64>,
    root_hi: Vec<f64>,
    free_dims: usize,
    heads: [HeadPlan; 2],
}

/// Raw node names of one head's tail, before prefix extraction.
struct HeadNodes<'a> {
    relu: &'a str,
    pow: &'a str,
    coeffs: Vec<f64>,
    bias: f64,
    exponent: u32,
}

fn detect(
    graph: &GraphNetwork,
    input_shape: &[usize],
    vnnlib: &VnnLibSpec,
) -> Option<FracHeadPlan> {
    // Spec shape: one global input box, a single scalar output.
    if vnnlib.dual_network.is_some() {
        return None;
    }
    if vnnlib
        .per_clause_input_bounds
        .iter()
        .any(|bounds| !bounds.is_empty())
    {
        return None;
    }
    if vnnlib.output_constraints.is_empty() && vnnlib.output_constraint_clauses.is_empty() {
        return None;
    }
    if vnnlib.num_outputs != 1 {
        return None;
    }
    let n_inputs: usize = input_shape.iter().product();
    if vnnlib.input_bounds.len() != n_inputs || n_inputs == 0 {
        return None;
    }
    let mut root_lo = Vec::with_capacity(n_inputs);
    let mut root_hi = Vec::with_capacity(n_inputs);
    for &(lo, hi) in &vnnlib.input_bounds {
        if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
            return None;
        }
        root_lo.push(lo);
        root_hi.push(hi);
    }

    // Output = Sub(headA, headB).
    let sub = graph.node(graph.output_name())?;
    if !matches!(sub.layer(), Layer::Sub(_)) || sub.inputs().len() != 2 {
        return None;
    }
    let head_a = parse_head(graph, &sub.inputs()[0])?;
    let head_b = parse_head(graph, &sub.inputs()[1])?;

    let prefix_a = extract_prefix(graph, head_a.relu)?;
    let prefix_b = extract_prefix(graph, head_b.relu)?;
    // The pow graphs (output `p = r^k`) drive the threshold-spec bound; the
    // Pow node is cloned from the original graph, exponent and all.
    let pow_a = extract_prefix(graph, head_a.pow)?;
    let pow_b = extract_prefix(graph, head_b.pow)?;

    // Concrete structural self-check: the analytic head formula must
    // reproduce the full graph's forward at sample points. Guards against
    // any mis-reading of the ONNX→graph conversion (axes, scaling, order).
    if !self_check(
        graph,
        &prefix_a,
        &head_a,
        &prefix_b,
        &head_b,
        &root_lo,
        &root_hi,
        input_shape,
    ) {
        warn!("Fractional-head detection: structural self-check failed; falling through");
        return None;
    }

    let dims_a = influence_dims(&prefix_a, &root_lo, &root_hi, input_shape)?;
    let dims_b = influence_dims(&prefix_b, &root_lo, &root_hi, input_shape)?;
    let free_dims = root_lo
        .iter()
        .zip(&root_hi)
        .filter(|(lo, hi)| hi > lo)
        .count();

    Some(FracHeadPlan {
        input_shape: input_shape.to_vec(),
        root_lo,
        root_hi,
        free_dims,
        heads: [
            HeadPlan {
                prefix: prefix_a,
                pow_graph: pow_a,
                coeffs: head_a.coeffs,
                bias: head_a.bias,
                exponent: head_a.exponent,
                dims: dims_a,
            },
            HeadPlan {
                prefix: prefix_b,
                pow_graph: pow_b,
                coeffs: head_b.coeffs,
                bias: head_b.bias,
                exponent: head_b.exponent,
                dims: dims_b,
            },
        ],
    })
}

/// Parse one head tail: Linear(Div(Pow(relu), ReduceSum(Pow(relu)))).
fn parse_head<'a>(graph: &'a GraphNetwork, linear_name: &str) -> Option<HeadNodes<'a>> {
    let linear_node = graph.node(linear_name)?;
    let Layer::Linear(linear) = linear_node.layer() else {
        return None;
    };
    if linear_node.inputs().len() != 1 {
        return None;
    }
    let n = linear.weight.ncols();
    if linear.weight.nrows() != 1 || !(2..=MAX_LOGITS).contains(&n) {
        return None;
    }
    if linear.weight.iter().any(|w| !w.is_finite()) {
        return None;
    }
    let bias = match &linear.bias {
        None => 0.0,
        Some(b) if b.len() == 1 && b[0].is_finite() => f64::from(b[0]),
        Some(_) => return None,
    };
    let coeffs: Vec<f64> = linear.weight.row(0).iter().map(|&w| f64::from(w)).collect();

    let div_node = graph.node(&linear_node.inputs()[0])?;
    if !matches!(div_node.layer(), Layer::Div(_)) || div_node.inputs().len() != 2 {
        return None;
    }
    let pow_name = &div_node.inputs()[0];
    let rsum_node = graph.node(&div_node.inputs()[1])?;
    let Layer::ReduceSum(rsum) = rsum_node.layer() else {
        return None;
    };
    if rsum_node.inputs().len() != 1 || &rsum_node.inputs()[0] != pow_name {
        return None;
    }

    let pow_node = graph.node(pow_name)?;
    let Layer::PowConstant(pow) = pow_node.layer() else {
        return None;
    };
    if pow_node.inputs().len() != 1 {
        return None;
    }
    let exp = f64::from(pow.exponent());
    if !(exp.is_finite() && exp >= 1.0 && exp.fract() == 0.0 && exp <= f64::from(MAX_EXPONENT)) {
        return None;
    }
    let exponent = exp as u32;

    // The ReduceSum must total ALL `n` logit elements: the pow tensor's
    // declared shape must be `[1, ..., 1, n]` with the reduction on the last
    // axis. Fail-open when shape metadata is unavailable.
    let pow_shape = graph.declared_shape(pow_name)?;
    let (last, rest) = pow_shape.split_last()?;
    if *last != n || rest.iter().any(|&d| d != 1) {
        return None;
    }
    let rank = pow_shape.len() as i64;
    let axes_ok = rsum.axes.iter().all(|&ax| ax == -1 || ax == rank - 1) && !rsum.axes.is_empty();
    if !axes_ok {
        return None;
    }

    let relu_node = graph.node(&pow_node.inputs()[0])?;
    if !matches!(relu_node.layer(), Layer::ReLU(_)) {
        return None;
    }

    Some(HeadNodes {
        relu: relu_node.name(),
        pow: pow_node.name(),
        coeffs,
        bias,
        exponent,
    })
}

/// Clone the backward closure of `target` into a fresh graph whose output is
/// `target` (the logit ReLU).
fn extract_prefix(graph: &GraphNetwork, target: &str) -> Option<GraphNetwork> {
    let mut needed: HashSet<String> = HashSet::new();
    let mut stack = vec![target.to_string()];
    while let Some(name) = stack.pop() {
        if !needed.insert(name.clone()) {
            continue;
        }
        let node = graph.node(&name)?;
        for input in node.inputs() {
            if input != NETWORK_INPUT && !needed.contains(input) {
                stack.push(input.clone());
            }
        }
    }
    let mut prefix = GraphNetwork::new();
    // node_names() preserves insertion (topological) order.
    for name in graph.node_names() {
        if needed.contains(name) {
            prefix.try_add_node(graph.node(name)?.clone()).ok()?;
        }
    }
    prefix.set_output(target.to_string());
    Some(prefix)
}

/// Evaluate the analytic head formula at a concrete logit vector.
fn head_formula(r: &[f64], coeffs: &[f64], bias: f64, k: u32) -> Option<f64> {
    let mut num = 0.0;
    let mut den = 0.0;
    for (&ri, &ci) in r.iter().zip(coeffs) {
        let p = ri.max(0.0).powi(k as i32);
        num += ci * p;
        den += p;
    }
    if den <= 0.0 || !den.is_finite() || !num.is_finite() {
        return None;
    }
    Some(num / den + bias)
}

/// Exact concrete f32 forward of a graph at a point, flattened.
fn point_forward(graph: &GraphNetwork, point_f32: &[f32], shape: &[usize]) -> Option<Vec<f64>> {
    let arr = ArrayD::from_shape_vec(IxDyn(shape), point_f32.to_vec()).ok()?;
    let input = ny_tensor::BoundedTensor::concrete(arr).ok()?;
    let out = graph.propagate_concrete_point(&input, None, None).ok()?;
    Some(out.center().iter().map(|&x| f64::from(x)).collect())
}

/// Trusted full-model forward at a point: the lazily-built ONNX-Runtime
/// session registered for this instance (the same oracle family that gates
/// every `sat`), falling back to the internal graph forward when ORT is
/// unavailable (unit tests, NY_ORT_ATTACK=0).
///
/// The ORT-first order matters: ORT is the trusted oracle for the REAL model
/// semantics. (Historically the internal graph mis-evaluated the
/// pensieve-parallel head outright — the legacy batch-squeeze `axis-1`
/// adjustment turned the ONNX `ReduceSum(axes=[1])` into a size-1-axis no-op
/// on runtime `[1, n]` tensors, so `w = p/p = 1`; fixed via trailing-relative
/// reduction axes in ny-build — but the analytic formula should still be
/// validated against ORT, not ny's own conversion.)
fn trusted_full_forward(
    graph: &GraphNetwork,
    point_f32: &[f32],
    shape: &[usize],
) -> Option<Vec<f64>> {
    if let Some(out) = super::verify::ort_attack::ort_forward_flat(point_f32) {
        return Some(out.into_iter().map(f64::from).collect());
    }
    point_forward(graph, point_f32, shape)
}

/// Verify at sample points that `Y == head_A - head_B` under the analytic
/// formula (tolerant of f32 forward noise).
#[allow(clippy::too_many_arguments)]
fn self_check(
    graph: &GraphNetwork,
    prefix_a: &GraphNetwork,
    head_a: &HeadNodes<'_>,
    prefix_b: &GraphNetwork,
    head_b: &HeadNodes<'_>,
    root_lo: &[f64],
    root_hi: &[f64],
    shape: &[usize],
) -> bool {
    let points: [Vec<f32>; 3] = [
        root_lo
            .iter()
            .zip(root_hi)
            .map(|(&l, &h)| f64::midpoint(l, h) as f32)
            .collect(),
        root_lo.iter().map(|&l| l as f32).collect(),
        root_hi.iter().map(|&h| h as f32).collect(),
    ];
    for point in &points {
        let Some(y_graph) = trusted_full_forward(graph, point, shape) else {
            debug!("Fractional-head self-check: full-model point forward failed");
            return false;
        };
        if y_graph.len() != 1 {
            debug!(
                "Fractional-head self-check: output len {} != 1",
                y_graph.len()
            );
            return false;
        }
        let Some(ra) = point_forward(prefix_a, point, shape) else {
            debug!("Fractional-head self-check: prefix A point forward failed");
            return false;
        };
        let Some(rb) = point_forward(prefix_b, point, shape) else {
            debug!("Fractional-head self-check: prefix B point forward failed");
            return false;
        };
        if ra.len() != head_a.coeffs.len() || rb.len() != head_b.coeffs.len() {
            debug!(
                "Fractional-head self-check: logit lens {}/{} vs coeffs {}/{}",
                ra.len(),
                rb.len(),
                head_a.coeffs.len(),
                head_b.coeffs.len()
            );
            return false;
        }
        let Some(sa) = head_formula(&ra, &head_a.coeffs, head_a.bias, head_a.exponent) else {
            debug!("Fractional-head self-check: head A formula undefined");
            return false;
        };
        let Some(sb) = head_formula(&rb, &head_b.coeffs, head_b.bias, head_b.exponent) else {
            debug!("Fractional-head self-check: head B formula undefined");
            return false;
        };
        let y_formula = sa - sb;
        let tol = SELF_CHECK_REL_TOL * (1.0 + y_graph[0].abs());
        if (y_formula - y_graph[0]).abs() > tol {
            debug!(
                "Fractional-head self-check mismatch: graph Y={} formula Y={}",
                y_graph[0], y_formula
            );
            return false;
        }
    }
    true
}

/// Per-head splittable dims with influence scores. A dim's influence is the
/// total width of the prefix IBP output when ONLY that dim carries its full
/// root width (all other dims pinned to center). IBP is an enclosure, so an
/// influence of exactly 0 proves the head does not depend on the dim (the
/// pensieve heads read disjoint input halves — this recovers that structure
/// without any slice arithmetic).
fn influence_dims(
    prefix: &GraphNetwork,
    root_lo: &[f64],
    root_hi: &[f64],
    input_shape: &[usize],
) -> Option<Vec<(usize, f64)>> {
    let center: Vec<f32> = root_lo
        .iter()
        .zip(root_hi)
        .map(|(&l, &h)| f64::midpoint(l, h) as f32)
        .collect();
    let total_width = |lo: Vec<f32>, hi: Vec<f32>| -> Option<f64> {
        let lo_arr = ArrayD::from_shape_vec(IxDyn(input_shape), lo).ok()?;
        let hi_arr = ArrayD::from_shape_vec(IxDyn(input_shape), hi).ok()?;
        let input = ny_tensor::BoundedTensor::new(lo_arr, hi_arr).ok()?;
        let out = prefix.propagate_ibp(&input).ok()?;
        let (l, u) = out.lower_upper();
        Some(
            l.iter()
                .zip(u.iter())
                .map(|(&a, &b)| (f64::from(b) - f64::from(a)).max(0.0))
                .sum(),
        )
    };
    // f32 interval arithmetic gives every eval a few-ULP floor even at a pure
    // point; influence must clear that noise, not merely be nonzero (else the
    // disjoint-halves structure is lost and both nets split every dim).
    let noise = total_width(center.clone(), center.clone())?;
    let threshold = noise * 4.0 + 1e-9;
    let mut dims = Vec::new();
    for d in 0..root_lo.len() {
        if root_hi[d] <= root_lo[d] {
            continue;
        }
        let mut lo = center.clone();
        let mut hi = center.clone();
        lo[d] = f32_down(root_lo[d]);
        hi[d] = f32_up(root_hi[d]);
        let influence = total_width(lo, hi)?;
        if influence > threshold {
            dims.push((d, influence));
        }
    }
    Some(dims)
}

// ---------------------------------------------------------------------------
// Directed f64 arithmetic + the linear-fractional head interval
// ---------------------------------------------------------------------------

#[inline]
fn f32_down(x: f64) -> f32 {
    let f = x as f32;
    if f64::from(f) > x {
        f.next_down()
    } else {
        f
    }
}

#[inline]
fn f32_up(x: f64) -> f32 {
    let f = x as f32;
    if f64::from(f) < x {
        f.next_up()
    } else {
        f
    }
}

/// `x^k` rounded toward -inf, for `x >= 0`.
#[inline]
fn pow_down(x: f64, k: u32) -> f64 {
    let mut acc = 1.0_f64;
    for _ in 0..k {
        acc = (acc * x).next_down().max(0.0);
    }
    acc
}

/// `x^k` rounded toward +inf, for `x >= 0`.
#[inline]
fn pow_up(x: f64, k: u32) -> f64 {
    let mut acc = 1.0_f64;
    for _ in 0..k {
        acc = (acc * x).next_up();
    }
    acc
}

/// Sound interval for `s = (Σ c_i p_i) / (Σ p_i) + bias` over `p ∈ [pl, pu]`,
/// `p >= 0`. Requires the denominator to be strictly positive over the whole
/// box (`Σ pl > 0`), else `None` (the head can hit 0/0 — unclaimable).
///
/// The extrema of a linear-fractional function over a box lie at vertices
/// where each coordinate is at the bound selected by `sign(c_i - s*)`; with
/// `c` sorted ascending those are exactly the `n+1` prefix cuts (put the
/// smallest-`c` block at the upper bound to minimize, at the lower bound to
/// maximize; coordinates with `c_i == s*` do not affect the value). Every
/// vertex value is evaluated with outward rounding and the outer min/max is
/// taken, so the result encloses the true range regardless of which vertex is
/// optimal.
fn head_range(pl: &[f64], pu: &[f64], coeffs: &[f64], bias: f64) -> Option<(f64, f64)> {
    let n = pl.len();
    if n == 0 || pu.len() != n || coeffs.len() != n {
        return None;
    }
    for i in 0..n {
        if !(pl[i].is_finite() && pu[i].is_finite() && 0.0 <= pl[i] && pl[i] <= pu[i]) {
            return None;
        }
    }
    // Denominator strictly positive everywhere in the box.
    let mut den_min = 0.0_f64;
    for &p in pl {
        den_min = (den_min + p).next_down();
    }
    // NaN-preserving fail-closed gate: a NaN den_min must decline (not "proven positive").
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(den_min > 0.0) {
        return None;
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| coeffs[a].total_cmp(&coeffs[b]));

    let mut s_min = f64::INFINITY;
    let mut s_max = f64::NEG_INFINITY;
    let mut vertex = vec![0.0_f64; n];
    for cut in 0..=n {
        // Minimizing vertex: smallest-c block at upper bound.
        for (rank, &idx) in order.iter().enumerate() {
            vertex[idx] = if rank < cut { pu[idx] } else { pl[idx] };
        }
        let lo = vertex_value_down(&vertex, coeffs)?;
        if lo < s_min {
            s_min = lo;
        }
        // Maximizing vertex: smallest-c block at lower bound.
        for (rank, &idx) in order.iter().enumerate() {
            vertex[idx] = if rank < cut { pl[idx] } else { pu[idx] };
        }
        let hi = vertex_value_up(&vertex, coeffs)?;
        if hi > s_max {
            s_max = hi;
        }
    }
    if !(s_min.is_finite() && s_max.is_finite()) {
        return None;
    }
    Some(((s_min + bias).next_down(), (s_max + bias).next_up()))
}

/// Sound interval for `s = (Σ c_i p_i) / (Σ p_i) + bias` over the
/// denominator-constrained box `p ∈ [pl, pu] ∩ {Σ p_i ∈ [den_lo, den_hi]}`.
///
/// The plain vertex range (`head_range`) is exact over the box but its
/// extreme vertices put whole coordinate blocks at opposite ends — exactly
/// the correlation the CROWN ones-row rules out. Constraining `Σ p_i` to the
/// correlation-aware range shrinks the feasible set (sound: the true `p`
/// satisfies both enclosures), often substantially on wide boxes.
///
/// Method (Dinkelbach): for `D > 0`, `s <= t` over the set iff
/// `max Σ (c_i - t) p_i <= 0`, and `s >= t` iff `min Σ (c_i - t) p_i >= 0`.
/// The inner problems are 1-budget box LPs solved exactly by a greedy fill
/// (exchange argument); both objectives are monotone decreasing in `t`, so
/// each end is a 60-step bisection over `[cmin, cmax]` (which always
/// contains `s` when `D > 0`). All arithmetic is outward-rounded and the
/// budget window is only ever ENLARGED (a superset feasible region weakens
/// but never unsounds the claim). Returns `None` (fail open) unless the
/// denominator is proven strictly positive.
fn head_range_with_denominator(
    pl: &[f64],
    pu: &[f64],
    coeffs: &[f64],
    bias: f64,
    den_lo: f64,
    den_hi: f64,
) -> Option<(f64, f64)> {
    let n = pl.len();
    if n == 0 || pu.len() != n || coeffs.len() != n {
        return None;
    }
    for i in 0..n {
        if !(pl[i].is_finite() && pu[i].is_finite() && 0.0 <= pl[i] && pl[i] <= pu[i]) {
            return None;
        }
    }
    if !(den_lo.is_finite() && den_hi.is_finite() && den_lo <= den_hi) {
        return None;
    }
    // Enlarge-only clamp of the budget window against the box-implied sums
    // (`Σ↓pl`, `Σ↑pu`), with a few ULPs of slack for the greedy's plain-f64
    // mass accounting.
    let mut sum_lo = 0.0_f64;
    let mut sum_hi = 0.0_f64;
    for i in 0..n {
        sum_lo = (sum_lo + pl[i]).next_down();
        sum_hi = (sum_hi + pu[i]).next_up();
    }
    let budget_slack = sum_hi.abs().max(1.0) * 1e-12;
    let dl = (den_lo.max(sum_lo) - budget_slack).next_down();
    let du = (den_hi.min(sum_hi) + budget_slack).next_up();
    // NaN-preserving fail-closed gate: a NaN dl must decline (not "proven positive").
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(dl > 0.0) || du < dl {
        return None;
    }

    let cmin = coeffs.iter().copied().fold(f64::INFINITY, f64::min);
    let cmax = coeffs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !(cmin.is_finite() && cmax.is_finite() && cmax >= cmin) {
        return None;
    }
    let mut order: Vec<usize> = (0..n).collect();

    // Upper end: smallest t with (rounded-up) max Σ (c_i - t) p_i <= 0.
    let phi = |t: f64, order: &mut [usize]| -> Option<f64> {
        let w: Vec<f64> = coeffs.iter().map(|&c| (c - t).next_up()).collect();
        budget_lp_max(&w, pl, pu, dl, du, order)
    };
    let s_max = {
        if phi(cmax, &mut order)? > 0.0 {
            cmax
        } else if phi(cmin, &mut order)? <= 0.0 {
            cmin
        } else {
            let (mut lo_t, mut hi_t) = (cmin, cmax);
            for _ in 0..60 {
                // Bisection pivot on the sound budget-LP bound path: `(l+u)/2` kept
                // verbatim — `f64::midpoint` differs at overflow edges and the
                // produced threshold anchor must not move.
                #[allow(clippy::manual_midpoint)]
                let mid = 0.5 * (lo_t + hi_t);
                if !(mid > lo_t && mid < hi_t) {
                    break;
                }
                if phi(mid, &mut order)? <= 0.0 {
                    hi_t = mid;
                } else {
                    lo_t = mid;
                }
            }
            hi_t
        }
    };

    // Lower end: largest t with (rounded-down) min Σ (c_i - t) p_i >= 0,
    // via min Σ w p = -max Σ (-w) p with the same up-rounded greedy.
    let psi = |t: f64, order: &mut [usize]| -> Option<f64> {
        let w: Vec<f64> = coeffs.iter().map(|&c| (t - c).next_up()).collect();
        budget_lp_max(&w, pl, pu, dl, du, order).map(|v| -v)
    };
    let s_min = {
        if psi(cmin, &mut order)? < 0.0 {
            cmin
        } else if psi(cmax, &mut order)? >= 0.0 {
            cmax
        } else {
            let (mut lo_t, mut hi_t) = (cmin, cmax);
            for _ in 0..60 {
                // Bisection pivot on the sound budget-LP bound path: `(l+u)/2` kept
                // verbatim — `f64::midpoint` differs at overflow edges and the
                // produced threshold anchor must not move.
                #[allow(clippy::manual_midpoint)]
                let mid = 0.5 * (lo_t + hi_t);
                if !(mid > lo_t && mid < hi_t) {
                    break;
                }
                if psi(mid, &mut order)? >= 0.0 {
                    lo_t = mid;
                } else {
                    hi_t = mid;
                }
            }
            lo_t
        }
    };
    if s_max < s_min {
        return None;
    }
    Some(((s_min + bias).next_down(), (s_max + bias).next_up()))
}

/// Exact maximum of `Σ w_i p_i` over `p ∈ [pl, pu], Σ p_i ∈ [dl, du]`,
/// rounded UP (a sound over-estimate). Greedy: start at `pl`, raise
/// coordinates in descending-`w` order — positive-`w` first up to the `du`
/// budget, then (only if `Σ pl < dl`) keep raising the least-harmful
/// coordinates until the `dl` floor is met. Optimal by exchange argument for
/// a single budget constraint. `None` when the floor is unreachable.
fn budget_lp_max(
    w: &[f64],
    pl: &[f64],
    pu: &[f64],
    dl: f64,
    du: f64,
    order: &mut [usize],
) -> Option<f64> {
    let n = w.len();
    debug_assert_eq!(order.len(), n);
    for (k, o) in order.iter_mut().enumerate() {
        *o = k;
    }
    order.sort_by(|&a, &b| w[b].total_cmp(&w[a]));
    let mut mass: f64 = pl.iter().sum();
    // Raised amount per coordinate (on top of pl).
    let mut raise = vec![0.0_f64; n];
    for &i in order.iter() {
        if w[i] <= 0.0 {
            break;
        }
        if mass >= du {
            break;
        }
        let r = (pu[i] - pl[i]).min(du - mass).max(0.0);
        raise[i] = r;
        mass += r;
    }
    if mass < dl {
        for &i in order.iter() {
            if mass >= dl {
                break;
            }
            let r = (pu[i] - pl[i] - raise[i]).min(dl - mass).max(0.0);
            raise[i] += r;
            mass += r;
        }
        if mass < dl {
            return None;
        }
    }
    let mut acc = 0.0_f64;
    let mut err_scale = 0.0_f64;
    for i in 0..n {
        acc = (acc + (w[i] * (pl[i] + raise[i])).next_up()).next_up();
        err_scale = (err_scale + (w[i].abs() * pu[i]).next_up()).next_up();
    }
    // Cover the greedy's plain-f64 mass accounting (relative error <=
    // n*eps ~ 1e-14): inflate by 1e-12 of the max attainable magnitude.
    // The bisection consumer compares against 0, so this only costs
    // ~err_scale*1e-12/D in claimed threshold resolution (negligible).
    acc = (acc + err_scale.max(1.0) * 1e-12).next_up();
    acc.is_finite().then_some(acc)
}

/// `(Σ c_i p_i) / (Σ p_i)` rounded toward -inf at a concrete vertex.
fn vertex_value_down(p: &[f64], coeffs: &[f64]) -> Option<f64> {
    let mut num_dn = 0.0_f64;
    let mut den_dn = 0.0_f64;
    let mut den_up = 0.0_f64;
    for (&pi, &ci) in p.iter().zip(coeffs) {
        num_dn = (num_dn + (ci * pi).next_down()).next_down();
        den_dn = (den_dn + pi).next_down();
        den_up = (den_up + pi).next_up();
    }
    // NaN-preserving fail-closed gate: a NaN den_dn must decline (not "proven positive").
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(den_dn > 0.0) {
        return None;
    }
    let v = if num_dn >= 0.0 {
        (num_dn / den_up).next_down()
    } else {
        (num_dn / den_dn).next_down()
    };
    v.is_finite().then_some(v)
}

/// `(Σ c_i p_i) / (Σ p_i)` rounded toward +inf at a concrete vertex.
fn vertex_value_up(p: &[f64], coeffs: &[f64]) -> Option<f64> {
    let mut num_up = 0.0_f64;
    let mut den_dn = 0.0_f64;
    let mut den_up = 0.0_f64;
    for (&pi, &ci) in p.iter().zip(coeffs) {
        num_up = (num_up + (ci * pi).next_up()).next_up();
        den_dn = (den_dn + pi).next_down();
        den_up = (den_up + pi).next_up();
    }
    // NaN-preserving fail-closed gate: a NaN den_dn must decline (not "proven positive").
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(den_dn > 0.0) {
        return None;
    }
    let v = if num_up >= 0.0 {
        (num_up / den_dn).next_up()
    } else {
        (num_up / den_up).next_up()
    };
    v.is_finite().then_some(v)
}

// ---------------------------------------------------------------------------
// Per-domain head bound: prefix CROWN logits → p = r^k → head interval
// ---------------------------------------------------------------------------

/// Threshold-grid points per stage (two-stage: resolution
/// `c_range / (STAGE_POINTS-1)^2`).
const STAGE_POINTS: usize = 33;

/// Which end(s) of the head interval a caller actually consumes. The
/// refinement loop only ever drives ONE end per net (the other is tracked
/// conservatively), so the threshold grid can drop half its rows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BoundNeed {
    Lower,
    Upper,
    Both,
}

impl BoundNeed {
    fn lower(self) -> bool {
        matches!(self, BoundNeed::Lower | BoundNeed::Both)
    }
    fn upper(self) -> bool {
        matches!(self, BoundNeed::Upper | BoundNeed::Both)
    }
}

/// Sound intersection of optional enclosures; an empty intersection means a
/// bug on one side — fail open rather than claim.
fn intersect_bounds(a: Option<(f64, f64)>, b: Option<(f64, f64)>) -> Option<(f64, f64)> {
    match (a, b) {
        (Some((al, au)), Some((bl, bu))) => {
            let lo = al.max(bl);
            let hi = au.min(bu);
            (lo <= hi).then_some((lo, hi))
        }
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    }
}

/// `NY_FRAC_HEAD_CLASSIC=1` reverts to the pre-fusion bound path (plain
/// specs engine entry, no gradient scores, separate prefix-CROWN vertex
/// pass, no denominator-constrained LP) — kill-switch for the new bound.
fn classic_mode() -> bool {
    static CLASSIC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CLASSIC.get_or_init(|| std::env::var_os("NY_FRAC_HEAD_CLASSIC").is_some_and(|v| v == "1"))
}

/// Result of one per-leaf head bound evaluation: the sound `s` enclosure
/// plus optional gradient split scores (per input dim `|A_d|` of the first
/// UNPROVEN threshold row — the row the next split should flip).
struct HeadBoundEval {
    s: Option<(f64, f64)>,
    scores: Option<Vec<f64>>,
}

fn bound_head(
    head: &HeadPlan,
    lo: &[f64],
    hi: &[f64],
    input_shape: &[usize],
    need: BoundNeed,
    parent: Option<(f64, f64)>,
) -> Option<HeadBoundEval> {
    let lo_f32: Vec<f32> = lo.iter().map(|&x| f32_down(x)).collect();
    let hi_f32: Vec<f32> = hi.iter().map(|&x| f32_up(x)).collect();
    let lo_arr = ArrayD::from_shape_vec(IxDyn(input_shape), lo_f32).ok()?;
    let hi_arr = ArrayD::from_shape_vec(IxDyn(input_shape), hi_f32).ok()?;
    let input = ny_tensor::BoundedTensor::new(lo_arr, hi_arr).ok()?;

    if classic_mode() {
        let t0 = Instant::now();
        let threshold = match parent {
            Some(parent) => bound_head_threshold_adaptive(head, &input, need, parent),
            None => bound_head_threshold(head, &input, need),
        }
        .and_then(|o| o.s);
        let t1 = Instant::now();
        let vertex = bound_head_vertex(head, &input);
        probe::record_bound_timing(t1 - t0, t1.elapsed(), threshold, vertex, need);
        return Some(HeadBoundEval {
            s: intersect_bounds(intersect_bounds(threshold, vertex), parent),
            scores: None,
        });
    }

    // The parent's claim is inherited outright (the child box is a subset,
    // so any `s >= t` / `s <= t` claim over the parent holds here too). It
    // both floors the result and licenses the adaptive one-call threshold
    // grid (denominator positivity was already proven on the parent).
    let t0 = Instant::now();
    let outcome = match parent {
        Some(parent) => bound_head_threshold_adaptive(head, &input, need, parent),
        None => bound_head_threshold(head, &input, need),
    };
    let t1 = Instant::now();
    let (threshold, fused_vertex, scores) = match outcome {
        Some(o) => {
            // Vertex bound on the p-boxes harvested from the SAME CROWN
            // pass, tightened by the correlation-aware denominator range
            // (ones row): exact over `p ∈ box ∩ {Σp ∈ [dl, du]}`.
            let v = o.pbox.as_ref().and_then(|pb| {
                head_range_with_denominator(
                    &pb.p_lo,
                    &pb.p_hi,
                    &head.coeffs,
                    head.bias,
                    pb.den_lo,
                    pb.den_hi,
                )
            });
            (o.s, v, o.scores)
        }
        None => (None, None, None),
    };
    // Fail-open: when the fused pass produced nothing, fall back to the
    // r-space prefix-CROWN vertex pass. Also run it at the root (one-off)
    // where its exact `pow_down/pow_up` box cubes add tightness.
    let vertex = if (threshold.is_none() && fused_vertex.is_none()) || parent.is_none() {
        probe::count_fallback();
        bound_head_vertex(head, &input)
    } else {
        None
    };
    probe::record_bound_timing(
        t1 - t0,
        t1.elapsed(),
        threshold,
        intersect_bounds(fused_vertex, vertex),
        need,
    );
    let s = intersect_bounds(
        intersect_bounds(intersect_bounds(threshold, fused_vertex), vertex),
        parent,
    );
    Some(HeadBoundEval { s, scores })
}

/// p-boxes + denominator range harvested from a threshold-grid CROWN pass.
struct PBox {
    p_lo: Vec<f64>,
    p_hi: Vec<f64>,
    den_lo: f64,
    den_hi: f64,
}

/// Threshold bound result plus the fused per-logit/denominator rows and
/// gradient split scores.
struct ThresholdOutcome {
    s: Option<(f64, f64)>,
    pbox: Option<PBox>,
    scores: Option<Vec<f64>>,
}

/// Intersect two optional p-boxes (both sound enclosures of the same `p`).
fn intersect_pbox(a: Option<PBox>, b: Option<PBox>) -> Option<PBox> {
    match (a, b) {
        (Some(mut a), Some(b)) => {
            if a.p_lo.len() != b.p_lo.len() {
                return None;
            }
            for i in 0..a.p_lo.len() {
                a.p_lo[i] = a.p_lo[i].max(b.p_lo[i]);
                a.p_hi[i] = a.p_hi[i].min(b.p_hi[i]);
                if a.p_hi[i] < a.p_lo[i] {
                    return None;
                }
            }
            a.den_lo = a.den_lo.max(b.den_lo);
            a.den_hi = a.den_hi.min(b.den_hi);
            (a.den_hi >= a.den_lo).then_some(a)
        }
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    }
}

/// Quadratically-spaced offsets in `(0, w]`: finest step `w/(points-1)^2`
/// next to 0, full coverage up to `w`. The refinement frontier crawls in
/// sub-grid steps, so resolution must be finest where the bound currently
/// sits (a uniform grid freezes the frontier once per-split improvements
/// drop below one step — measured on pensieve w=0.5: the top 512 leaves all
/// pinned at the identical inherited bound).
fn geometric_offsets(w: f64, points: usize) -> Vec<f64> {
    // NaN-preserving fail-closed gate: a NaN width must yield no offsets.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(w > 0.0) || points < 2 {
        return Vec::new();
    }
    (1..points)
        .map(|j| {
            let f = j as f64 / (points - 1) as f64;
            w * f * f
        })
        .collect()
}

/// Adaptive threshold refinement for a child leaf: one spec call around the
/// inherited claim (extended a bounded number of times while an end keeps
/// walking to its grid edge). Denominator positivity is inherited from the
/// parent's claim, so no ones-row gate is needed. The grid is geometric —
/// finest next to the inherited bound — and its width is capped by the
/// parent's remaining interval (a child can never prove past the other end).
fn bound_head_threshold_adaptive(
    head: &HeadPlan,
    input: &ny_tensor::BoundedTensor,
    need: BoundNeed,
    parent: (f64, f64),
) -> Option<ThresholdOutcome> {
    let c = &head.coeffs;
    let cmin = c.iter().copied().fold(f64::INFINITY, f64::min);
    let cmax = c.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !(cmin.is_finite() && cmax.is_finite() && cmax >= cmin) {
        return None;
    }
    // cmin/cmax are checked finite above, so span_cap cannot be NaN and the
    // positive-gate rewrite is exact.
    let span_cap = (cmax - cmin) / 16.0;
    if span_cap <= 0.0 {
        return None;
    }

    let mut s_lo = (parent.0 - head.bias).clamp(cmin, cmax);
    let mut s_hi = (parent.1 - head.bias).clamp(cmin, cmax);
    let mut pbox: Option<PBox> = None;
    let mut scores: Option<Vec<f64>> = None;
    for _extension in 0..3 {
        let w_lo = (s_hi - s_lo).min(span_cap);
        let w_hi = (s_hi - s_lo).min(span_cap);
        let mut ts = Vec::new();
        let mut n_lo = 0usize;
        if need.lower() {
            let offsets = geometric_offsets(w_lo, STAGE_POINTS);
            n_lo = offsets.len();
            ts.extend(offsets.iter().map(|&o| (s_lo + o).min(cmax)));
        }
        if need.upper() {
            ts.extend(
                geometric_offsets(w_hi, STAGE_POINTS)
                    .iter()
                    .map(|&o| (s_hi - o).max(cmin)),
            );
        }
        if ts.is_empty() {
            // Parent interval already degenerate: nothing to refine, but the
            // clamped parent claim itself is still sound.
            break;
        }
        let eval = eval_threshold_grid(head, input, &ts, need)?;
        pbox = intersect_pbox(pbox, harvest_pbox(&eval));
        if eval.scores.is_some() {
            scores = eval.scores.clone();
        }
        let (prev_lo, prev_hi) = (s_lo, s_hi);
        for (j, &t) in ts.iter().enumerate() {
            let drives_lower = need.lower() && j < n_lo;
            if drives_lower && eval.lower_lb(j) >= 0.0 && t > s_lo {
                s_lo = t;
            }
            if !drives_lower && eval.upper_ub(j) <= 0.0 && t < s_hi {
                s_hi = t;
            }
        }
        // Extend only while an end walked to its grid edge (bracket-limited).
        let lo_at_edge = need.lower() && (s_lo - prev_lo) >= w_lo * 0.9 && s_lo < cmax;
        let hi_at_edge = need.upper() && (prev_hi - s_hi) >= w_hi * 0.9 && s_hi > cmin;
        if !lo_at_edge && !hi_at_edge {
            break;
        }
    }
    if s_hi < s_lo {
        return None;
    }
    Some(ThresholdOutcome {
        s: Some(((s_lo + head.bias).next_down(), (s_hi + head.bias).next_up())),
        pbox,
        scores,
    })
}

/// Threshold-spec bound: `s >= t` iff `g_t := Σ (c_i - t) p_i >= 0` given
/// `p >= 0` and `Σ p_i > 0` (multiply the mediant through by the positive
/// denominator). Each `g_t` is one CROWN spec row over the pow graph — the
/// linear backward through `p = r^k` and the shared prefix captures the
/// inter-logit correlation that per-logit boxes lose. A grid of thresholds
/// (both directions, plus an all-ones row proving `Σ p_i > 0`) is evaluated
/// in a single backward per stage; a second stage refines inside the
/// bracketing step.
///
/// Row rounding: for a LOWER claim (`s >= t`) entries are rounded DOWN
/// (`w_i <= c_i - t`, so `g_t >= Σ w_i p_i >= lb` for `p >= 0`); for an UPPER
/// claim entries are rounded UP.
fn bound_head_threshold(
    head: &HeadPlan,
    input: &ny_tensor::BoundedTensor,
    need: BoundNeed,
) -> Option<ThresholdOutcome> {
    let c = &head.coeffs;
    let cmin = c.iter().copied().fold(f64::INFINITY, f64::min);
    let cmax = c.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !(cmin.is_finite() && cmax.is_finite() && cmax >= cmin) {
        return None;
    }

    // Stage 1: full coefficient range.
    let ts1 = linspace(cmin, cmax, STAGE_POINTS);
    let eval1 = eval_threshold_grid(head, input, &ts1, need)?;
    let mut pbox = harvest_pbox(&eval1);
    let mut scores = eval1.scores.clone();
    // NaN-preserving fail-closed gate: a NaN ones_lb is "not proven positive".
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(eval1.ones_lb > 0.0) {
        // Denominator not proven positive over the box: the mediant can hit
        // 0/0 — unclaimable by this method (the p-box is still a sound
        // enclosure; its consumer re-checks positivity itself).
        return Some(ThresholdOutcome {
            s: None,
            pbox,
            scores,
        });
    }
    // `s ∈ [cmin, cmax]` holds unconditionally for a mediant with `D > 0`.
    let mut s_lo = cmin;
    let mut s_hi = cmax;
    for (j, &t) in ts1.iter().enumerate() {
        if eval1.lower_lb(j) >= 0.0 && t > s_lo {
            s_lo = t;
        }
        if eval1.upper_ub(j) <= 0.0 && t < s_hi {
            s_hi = t;
        }
    }

    // Stage 2: refine the needed end(s) inside their bracketing step.
    let step = (cmax - cmin) / ((STAGE_POINTS - 1) as f64);
    if step > 0.0 {
        let mut ts2 = Vec::new();
        if need.lower() {
            ts2.extend(linspace(s_lo, (s_lo + step).min(cmax), STAGE_POINTS));
        }
        if need.upper() {
            ts2.extend(linspace((s_hi - step).max(cmin), s_hi, STAGE_POINTS));
        }
        let eval2 = eval_threshold_grid(head, input, &ts2, need)?;
        pbox = intersect_pbox(pbox, harvest_pbox(&eval2));
        if eval2.scores.is_some() {
            scores = eval2.scores.clone();
        }
        for (j, &t) in ts2.iter().enumerate() {
            if eval2.lower_lb(j) >= 0.0 && t > s_lo {
                s_lo = t;
            }
            if eval2.upper_ub(j) <= 0.0 && t < s_hi {
                s_hi = t;
            }
        }
    }

    if s_hi < s_lo {
        // Relaxation inconsistency — fail open to the vertex method.
        return None;
    }
    Some(ThresholdOutcome {
        s: Some(((s_lo + head.bias).next_down(), (s_hi + head.bias).next_up())),
        pbox,
        scores,
    })
}

/// Extract the fused per-logit p-box + denominator range from a grid eval.
/// `p = r^k >= 0` in the reals, so clamping the lower end at 0 is sound.
fn harvest_pbox(eval: &ThresholdGridEval) -> Option<PBox> {
    if eval.p_lo.is_empty() || eval.p_lo.len() != eval.p_hi.len() {
        return None;
    }
    let mut p_lo = Vec::with_capacity(eval.p_lo.len());
    let mut p_hi = Vec::with_capacity(eval.p_hi.len());
    for (&l, &u) in eval.p_lo.iter().zip(&eval.p_hi) {
        if !u.is_finite() || u < 0.0 {
            return None;
        }
        let l = if l.is_finite() {
            l.max(0.0)
        } else {
            return None;
        };
        if u < l {
            return None;
        }
        p_lo.push(l);
        p_hi.push(u);
    }
    // The ones row bounds Σ p_i with prefix correlation preserved; it can
    // only tighten the box-implied denominator range.
    let den_lo = eval.ones_lb;
    let den_hi = eval.ones_ub;
    if !(den_lo.is_finite() && den_hi.is_finite() && den_hi >= 0.0) {
        return None;
    }
    Some(PBox {
        p_lo,
        p_hi,
        den_lo,
        den_hi,
    })
}

struct ThresholdGridEval {
    /// Sound lower bound of `Σ round_down(c_i - t_j) p_i` per grid point
    /// (empty when the lower end was not requested).
    lower_lbs: Vec<f64>,
    /// Sound upper bound of `Σ round_up(c_i - t_j) p_i` per grid point
    /// (empty when the upper end was not requested).
    upper_ubs: Vec<f64>,
    /// Sound lower bound of `Σ p_i` (denominator positivity witness).
    ones_lb: f64,
    /// Sound upper bound of `Σ p_i`.
    ones_ub: f64,
    /// Sound per-logit bounds of `p_i` (identity rows, same CROWN pass).
    p_lo: Vec<f64>,
    p_hi: Vec<f64>,
    /// Gradient split scores over input dims (see `eval_threshold_grid`).
    scores: Option<Vec<f64>>,
}

impl ThresholdGridEval {
    /// `-inf` (never claims) when the lower end was not evaluated.
    fn lower_lb(&self, j: usize) -> f64 {
        self.lower_lbs.get(j).copied().unwrap_or(f64::NEG_INFINITY)
    }
    /// `+inf` (never claims) when the upper end was not evaluated.
    fn upper_ub(&self, j: usize) -> f64 {
        self.upper_ubs.get(j).copied().unwrap_or(f64::INFINITY)
    }
}

/// One spec-guided CROWN backward over the pow graph evaluating every grid
/// threshold in the requested rounding direction(s) plus the all-ones
/// denominator row and one identity row per logit (fused p-box: the same
/// backward yields per-logit `p_i` bounds and the correlation-aware `Σ p_i`
/// range at marginal cost, replacing the separate prefix vertex pass).
fn eval_threshold_grid(
    head: &HeadPlan,
    input: &ny_tensor::BoundedTensor,
    ts: &[f64],
    need: BoundNeed,
) -> Option<ThresholdGridEval> {
    let n = head.coeffs.len();
    let dirs = usize::from(need.lower()) + usize::from(need.upper());
    let rows = ts.len() * dirs + n + 1;
    let mut specs = Array2::<f32>::zeros((rows, n));
    let mut row = 0usize;
    let mut lower_rows = Vec::new();
    let mut upper_rows = Vec::new();
    for &t in ts {
        if need.lower() {
            for i in 0..n {
                specs[[row, i]] = f32_down(head.coeffs[i] - t);
            }
            lower_rows.push(row);
            row += 1;
        }
        if need.upper() {
            for i in 0..n {
                specs[[row, i]] = f32_up(head.coeffs[i] - t);
            }
            upper_rows.push(row);
            row += 1;
        }
    }
    // Identity rows (exact 0/1 entries — both bound directions sound).
    let ident_base = row;
    for i in 0..n {
        specs[[ident_base + i, i]] = 1.0;
    }
    for i in 0..n {
        specs[[rows - 1, i]] = 1.0;
    }
    // The `_with_linear` request routes through the FULL spec-CROWN
    // backward (and additionally yields input-space coefficients for
    // gradient split scoring). The plain `propagate_crown_with_specs`
    // entry serves a forward-linear fast path that is catastrophically
    // loose on this pow graph (measured ~4 units of root slack per head
    // on pensieve w=0.5) — routing around it is most of this verifier's
    // tightness. `NY_FRAC_HEAD_CLASSIC=1` restores the old entry.
    let (out, linear) = if classic_mode() {
        (
            head.pow_graph
                .propagate_crown_with_specs_and_engine(input, &specs, None)
                .ok()?,
            None,
        )
    } else {
        head.pow_graph
            .propagate_crown_with_specs_and_engine_with_linear(input, &specs, None)
            .ok()?
    };
    let (l, u) = out.lower_upper();
    if l.len() != rows {
        return None;
    }
    let l: Vec<f32> = l.iter().copied().collect();
    let u: Vec<f32> = u.iter().copied().collect();
    let lower_lbs: Vec<f64> = lower_rows.iter().map(|&r| f64::from(l[r])).collect();
    let upper_ubs: Vec<f64> = upper_rows.iter().map(|&r| f64::from(u[r])).collect();
    if lower_lbs.iter().any(|v| v.is_nan()) || upper_ubs.iter().any(|v| v.is_nan()) {
        return None;
    }
    let p_lo: Vec<f64> = (0..n).map(|i| f64::from(l[ident_base + i])).collect();
    let p_hi: Vec<f64> = (0..n).map(|i| f64::from(u[ident_base + i])).collect();

    // Gradient split scores: |input coefficients| of the first UNPROVEN
    // threshold row per driven direction — the row a split should flip
    // next. First-order lb gain from halving dim d is |A_d|*width_d/2.
    let scores = linear.as_ref().and_then(|lin| {
        let la = lin.lower_a();
        let ua = lin.upper_a();
        if la.nrows() != rows || ua.nrows() != rows {
            return None;
        }
        let dims = la.ncols();
        let mut acc = vec![0.0_f64; dims];
        let mut any = false;
        if !lower_lbs.is_empty() {
            let j = lower_lbs
                .iter()
                .position(|&v| v < 0.0)
                .unwrap_or(lower_lbs.len() - 1);
            let row = lower_rows[j];
            for d in 0..dims {
                acc[d] += f64::from(la[[row, d]].abs());
            }
            any = true;
        }
        if !upper_ubs.is_empty() {
            let j = upper_ubs
                .iter()
                .position(|&v| v > 0.0)
                .unwrap_or(upper_ubs.len() - 1);
            let row = upper_rows[j];
            for d in 0..dims {
                acc[d] += f64::from(ua[[row, d]].abs());
            }
            any = true;
        }
        (any && acc.iter().all(|v| v.is_finite())).then_some(acc)
    });

    Some(ThresholdGridEval {
        lower_lbs,
        upper_ubs,
        ones_lb: f64::from(l[rows - 1]),
        ones_ub: f64::from(u[rows - 1]),
        p_lo,
        p_hi,
        scores,
    })
}

/// Inclusive evenly-spaced grid.
fn linspace(a: f64, b: f64, points: usize) -> Vec<f64> {
    if points < 2 || b <= a {
        return vec![a];
    }
    let step = (b - a) / ((points - 1) as f64);
    (0..points).map(|i| a + step * i as f64).collect()
}

/// Fallback bound: per-logit CROWN boxes + the exact linear-fractional
/// vertex range (loses inter-logit correlation but needs no spec support).
fn bound_head_vertex(head: &HeadPlan, input: &ny_tensor::BoundedTensor) -> Option<(f64, f64)> {
    let r = head.prefix.propagate_crown_fixed_slope(input).ok()?;
    let pb = pbox_from_r_bounds(head, &r)?;
    head_range(&pb.p_lo, &pb.p_hi, &head.coeffs, head.bias)
}

/// Cube (`k`-power) the per-logit r-box into a p-box with exact directed
/// rounding. `den_lo/den_hi` are the box-implied sums.
fn pbox_from_r_bounds(head: &HeadPlan, r: &ny_tensor::BoundedTensor) -> Option<PBox> {
    let (rl, ru) = r.lower_upper();
    let n = head.coeffs.len();
    if rl.len() != n {
        return None;
    }
    let mut p_lo = Vec::with_capacity(n);
    let mut p_hi = Vec::with_capacity(n);
    let mut den_lo = 0.0_f64;
    let mut den_hi = 0.0_f64;
    for (&l, &u) in rl.iter().zip(ru.iter()) {
        if !l.is_finite() || !u.is_finite() {
            return None;
        }
        // The prefix output is a ReLU: intersecting with [0, ∞) is sound.
        let l = f64::from(l).max(0.0);
        let u = f64::from(u).max(0.0);
        if u < l {
            return None;
        }
        let pl = pow_down(l, head.exponent);
        let pu = pow_up(u, head.exponent);
        den_lo = (den_lo + pl).next_down();
        den_hi = (den_hi + pu).next_up();
        p_lo.push(pl);
        p_hi.push(pu);
    }
    Some(PBox {
        p_lo,
        p_hi,
        den_lo,
        den_hi,
    })
}

// ---------------------------------------------------------------------------
// Refinement driver
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Leaf {
    lo: Vec<f64>,
    hi: Vec<f64>,
    /// Head bound over this leaf, `None` = unbounded (never claimable,
    /// refined first).
    s: Option<(f64, f64)>,
    /// Gradient split scores from this leaf's own bound eval (`|A_d|` of the
    /// first unproven threshold row); `None` falls back to root influence.
    scores: Option<Vec<f64>>,
    depth: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Objective {
    /// Refine toward raising the head's minimum (`min s` is the active end).
    Min,
    /// Refine toward lowering the head's maximum (`max s` is the active end).
    Max,
}

struct HeapLeaf {
    /// Higher key = refined earlier.
    key: f64,
    leaf: Leaf,
}

impl PartialEq for HeapLeaf {
    fn eq(&self, other: &Self) -> bool {
        self.key.total_cmp(&other.key) == CmpOrdering::Equal
    }
}
impl Eq for HeapLeaf {}
impl PartialOrd for HeapLeaf {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapLeaf {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.key.total_cmp(&other.key)
    }
}

struct NetState<'a> {
    head: &'a HeadPlan,
    objective: Objective,
    heap: BinaryHeap<HeapLeaf>,
    /// Active-end extreme of leaves that can no longer be split.
    frozen_active: f64,
    /// Conservative (never-tightening) passive-end cover.
    passive: f64,
    root_widths: Vec<f64>,
    live_leaves: usize,
    bound_evals: usize,
    max_depth: u32,
}

impl<'a> NetState<'a> {
    fn new(head: &'a HeadPlan, objective: Objective, root: Leaf, root_widths: Vec<f64>) -> Self {
        let passive = match objective {
            Objective::Min => root.s.map_or(f64::INFINITY, |(_, u)| u),
            Objective::Max => root.s.map_or(f64::NEG_INFINITY, |(l, _)| l),
        };
        let frozen_active = match objective {
            Objective::Min => f64::INFINITY,
            Objective::Max => f64::NEG_INFINITY,
        };
        let mut state = Self {
            head,
            objective,
            heap: BinaryHeap::new(),
            frozen_active,
            passive,
            root_widths,
            live_leaves: 0,
            bound_evals: 1,
            max_depth: 0,
        };
        state.push_leaf(root);
        state
    }

    fn key_of(&self, leaf: &Leaf) -> f64 {
        match (self.objective, leaf.s) {
            // Worst (lowest) lower bound first.
            (Objective::Min, Some((l, _))) => -l,
            // Worst (highest) upper bound first.
            (Objective::Max, Some((_, u))) => u,
            // Unbounded leaves block any claim: always first.
            _ => f64::INFINITY,
        }
    }

    fn push_leaf(&mut self, leaf: Leaf) {
        // Track the conservative passive end.
        match self.objective {
            Objective::Min => {
                let hi = leaf.s.map_or(f64::INFINITY, |(_, u)| u);
                if hi > self.passive {
                    self.passive = hi;
                }
            }
            Objective::Max => {
                let lo = leaf.s.map_or(f64::NEG_INFINITY, |(l, _)| l);
                if lo < self.passive {
                    self.passive = lo;
                }
            }
        }
        self.max_depth = self.max_depth.max(leaf.depth);
        if self.splittable_dim(&leaf).is_some() {
            let key = self.key_of(&leaf);
            self.heap.push(HeapLeaf { key, leaf });
            self.live_leaves += 1;
        } else {
            // Terminal: folds permanently into the active-end extreme.
            match (self.objective, leaf.s) {
                (Objective::Min, Some((l, _))) => self.frozen_active = self.frozen_active.min(l),
                (Objective::Max, Some((_, u))) => self.frozen_active = self.frozen_active.max(u),
                (Objective::Min, None) => self.frozen_active = f64::NEG_INFINITY,
                (Objective::Max, None) => self.frozen_active = f64::INFINITY,
            }
        }
    }

    /// Best splittable dim: the leaf's own gradient scores (`|A_d| × width`,
    /// first-order lb gain of halving dim `d`) when available, else root
    /// influence × relative remaining width.
    fn splittable_dim(&self, leaf: &Leaf) -> Option<usize> {
        // Gradient-guided pass (leaf-local, from the binding CROWN row).
        if let Some(scores) = &leaf.scores {
            let mut best: Option<(usize, f64)> = None;
            for &(d, _) in &self.head.dims {
                let width = leaf.hi[d] - leaf.lo[d];
                let root_width = self.root_widths[d];
                if root_width <= 0.0 || width <= root_width * MIN_REL_WIDTH {
                    continue;
                }
                let Some(&g) = scores.get(d) else { continue };
                let score = g * width;
                if score > 0.0 && best.is_none_or(|(_, s)| score > s) {
                    best = Some((d, score));
                }
            }
            if let Some((d, _)) = best {
                return Some(d);
            }
            // All-zero gradients: fall through to the influence heuristic.
        }
        let mut best: Option<(usize, f64)> = None;
        for &(d, influence) in &self.head.dims {
            let width = leaf.hi[d] - leaf.lo[d];
            let root_width = self.root_widths[d];
            if root_width <= 0.0 || width <= root_width * MIN_REL_WIDTH {
                continue;
            }
            let score = influence * (width / root_width);
            if best.is_none_or(|(_, s)| score > s) {
                best = Some((d, score));
            }
        }
        best.map(|(d, _)| d)
    }

    /// Active-end cover value over ALL current leaves.
    fn active_cover(&self) -> f64 {
        let heap_val = match (self.objective, self.heap.peek()) {
            (Objective::Min, Some(top)) => -top.key,
            (Objective::Max, Some(top)) => top.key,
            (Objective::Min, None) => f64::INFINITY,
            (Objective::Max, None) => f64::NEG_INFINITY,
        };
        match self.objective {
            Objective::Min => heap_val.min(self.frozen_active),
            Objective::Max => heap_val.max(self.frozen_active),
        }
    }

    /// Head cover interval [L, U] over all current leaves.
    fn cover(&self) -> (f64, f64) {
        match self.objective {
            Objective::Min => (self.active_cover(), self.passive),
            Objective::Max => (self.passive, self.active_cover()),
        }
    }

    /// True when further refinement cannot improve the active cover (blocked
    /// by an unsplittable leaf).
    fn stuck(&self) -> bool {
        if self.heap.is_empty() {
            return true;
        }
        let heap_val = match (self.objective, self.heap.peek()) {
            (Objective::Min, Some(top)) => -top.key,
            (Objective::Max, Some(top)) => top.key,
            _ => return true,
        };
        match self.objective {
            Objective::Min => {
                self.frozen_active < heap_val || self.frozen_active == f64::NEG_INFINITY
            }
            Objective::Max => self.frozen_active > heap_val || self.frozen_active == f64::INFINITY,
        }
    }

    /// Refine up to `batch` worst leaves; returns false when nothing was
    /// refined (heap exhausted). Children past `deadline` stay unbounded
    /// (sound: unbounded leaves can never be claimed).
    fn refine_round(&mut self, batch: usize, input_shape: &[usize], deadline: Instant) -> bool {
        let mut work = Vec::with_capacity(batch);
        for _ in 0..batch {
            let Some(top) = self.heap.pop() else { break };
            self.live_leaves -= 1;
            work.push(top.leaf);
        }
        if work.is_empty() {
            return false;
        }
        let mut splits = Vec::with_capacity(work.len());
        for leaf in work {
            // splittable_dim held when pushed; re-check (width may be tiny).
            match self.splittable_dim(&leaf) {
                Some(d) => splits.push((leaf, d)),
                None => self.push_leaf(leaf),
            }
        }
        let head = self.head;
        // Only the driven end needs threshold rows; the passive end is
        // tracked conservatively (and by the vertex intersection).
        let need = match self.objective {
            Objective::Min => BoundNeed::Lower,
            Objective::Max => BoundNeed::Upper,
        };
        let children: Vec<Leaf> = splits
            .par_iter()
            .flat_map_iter(|(leaf, d)| {
                let d = *d;
                // BaB split point on the bound path: `(l+u)/2` kept verbatim —
                // `f64::midpoint` differs at overflow edges and the produced
                // subdomain bounds must not move.
                #[allow(clippy::manual_midpoint)]
                let mid = 0.5 * (leaf.lo[d] + leaf.hi[d]);
                let halves = [(leaf.lo[d], mid), (mid, leaf.hi[d])];
                halves.into_iter().map(move |(a, b)| {
                    let mut lo = leaf.lo.clone();
                    let mut hi = leaf.hi.clone();
                    lo[d] = a;
                    hi[d] = b;
                    let (s, scores) = if Instant::now() < deadline {
                        match bound_head(head, &lo, &hi, input_shape, need, leaf.s) {
                            Some(eval) => (eval.s, eval.scores.or_else(|| leaf.scores.clone())),
                            None => (None, leaf.scores.clone()),
                        }
                    } else {
                        // Past-deadline children still inherit the parent's
                        // (sound) claim so the exit-time cover stays finite.
                        (leaf.s, leaf.scores.clone())
                    };
                    Leaf {
                        lo,
                        hi,
                        s,
                        scores,
                        depth: leaf.depth + 1,
                    }
                })
            })
            .collect();
        self.bound_evals += children.len();
        for child in children {
            self.push_leaf(child);
        }
        true
    }
}

fn run_refinement(
    plan: &FracHeadPlan,
    vnnlib: &VnnLibSpec,
    deadline: Instant,
    start: Instant,
) -> Option<BetaCrownResult> {
    // Which direction of the Y cover can prove safety?
    let raise_lb = box_definitely_safe(&[f64::INFINITY], &[f64::INFINITY], vnnlib);
    let lower_ub = box_definitely_safe(&[f64::NEG_INFINITY], &[f64::NEG_INFINITY], vnnlib);
    let (obj_a, obj_b) = match (raise_lb, lower_ub) {
        (true, _) => (Objective::Min, Objective::Max),
        (_, true) => (Objective::Max, Objective::Min),
        _ => {
            debug!("Fractional-head: constraint shape needs both cover ends; falling through");
            return None;
        }
    };

    let root_widths: Vec<f64> = plan
        .root_lo
        .iter()
        .zip(&plan.root_hi)
        .map(|(&l, &h)| h - l)
        .collect();

    let mut root_bounds: Vec<(Option<(f64, f64)>, Option<Vec<f64>>)> = plan
        .heads
        .par_iter()
        .map(|head| {
            match bound_head(
                head,
                &plan.root_lo,
                &plan.root_hi,
                &plan.input_shape,
                BoundNeed::Both,
                None,
            ) {
                Some(eval) => (eval.s, eval.scores),
                None => (None, None),
            }
        })
        .collect();
    info!(
        "Fractional-head root bounds: sA={:?} sB={:?}",
        root_bounds[0].0, root_bounds[1].0
    );

    let (root_s_b, root_scores_b) = root_bounds.pop()?;
    let (root_s_a, root_scores_a) = root_bounds.pop()?;
    let mut net_a = NetState::new(
        &plan.heads[0],
        obj_a,
        Leaf {
            lo: plan.root_lo.clone(),
            hi: plan.root_hi.clone(),
            s: root_s_a,
            scores: root_scores_a,
            depth: 0,
        },
        root_widths.clone(),
    );
    let mut net_b = NetState::new(
        &plan.heads[1],
        obj_b,
        Leaf {
            lo: plan.root_lo.clone(),
            hi: plan.root_hi.clone(),
            s: root_s_b,
            scores: root_scores_b,
            depth: 0,
        },
        root_widths,
    );

    let mut rounds = 0usize;
    loop {
        let (la, ua) = net_a.cover();
        let (lb, ub) = net_b.cover();
        let ylb = (la - ub).next_down();
        let yub = (ua - lb).next_up();
        if box_definitely_safe(&[ylb], &[yub], vnnlib) {
            let explored = net_a.bound_evals + net_b.bound_evals;
            let verified = net_a.live_leaves + net_b.live_leaves;
            info!(
                "Fractional-head VERIFIED: Y in [{ylb:.6}, {yub:.6}] after {rounds} round(s), \
                 {explored} head bound(s), {:.1}s",
                start.elapsed().as_secs_f64()
            );
            return Some(BetaCrownResult {
                result: BabVerificationStatus::Verified,
                domains_explored: explored,
                domains_verified: verified.max(1),
                cuts_generated: 0,
                max_depth_reached: net_a.max_depth.max(net_b.max_depth) as usize,
                time_elapsed: start.elapsed(),
                output_bounds: None,
            });
        }

        if rounds.is_multiple_of(25) && rounds > 0 {
            debug!(
                "Fractional-head round {rounds}: Y in [{ylb:.4}, {yub:.4}], \
                 leaves A={} B={}",
                net_a.live_leaves, net_b.live_leaves
            );
        }

        if rounds >= MAX_ROUNDS
            || Instant::now() >= deadline
            || net_a.live_leaves + net_b.live_leaves > MAX_LEAVES
        {
            info!(
                "Fractional-head inconclusive (Y in [{ylb:.6}, {yub:.6}], {} rounds, \
                 {} leaves); falling through",
                rounds,
                net_a.live_leaves + net_b.live_leaves
            );
            if probe::enabled() {
                probe::report_timing();
                probe::probe_net(&net_a, &plan.input_shape, "netA");
                probe::probe_net(&net_b, &plan.input_shape, "netB");
            }
            return None;
        }

        let progressed_a = net_a.refine_round(REFINE_BATCH, &plan.input_shape, deadline);
        let progressed_b = net_b.refine_round(REFINE_BATCH, &plan.input_shape, deadline);
        if (!progressed_a || net_a.stuck()) && (!progressed_b || net_b.stuck()) {
            debug!("Fractional-head: both nets stuck; falling through");
            return None;
        }
        rounds += 1;
    }
}

/// SAT probe: a concretely-violating trusted forward at the root center
/// point (cell_enum's `confirm_violation` semantics — the generic interval
/// evaluators are uselessly wide through the Div even at a point). The
/// emitted witness is re-checked by the vnncomp trusted-oracle ORT gate
/// downstream before any `sat` is scored. Only reachable after `detect`'s
/// self-check validated the trusted forward against the analytic formula.
fn violated_at_center(
    graph: &GraphNetwork,
    plan: &FracHeadPlan,
    vnnlib: &VnnLibSpec,
    start: Instant,
) -> Option<BetaCrownResult> {
    // Nearest-rounded f32 center: like cell_enum's witness, it lands within
    // the widened input box the rest of the pipeline (and the ORT gate) uses.
    let center: Vec<f32> = plan
        .root_lo
        .iter()
        .zip(&plan.root_hi)
        .map(|(&l, &h)| f64::midpoint(l, h) as f32)
        .collect();
    let y = trusted_full_forward(graph, &center, &plan.input_shape)?;
    let output: Vec<f32> = y.iter().map(|&v| v as f32).collect();
    let output_arr = ArrayD::from_shape_vec(IxDyn(&[output.len()]), output.clone()).ok()?;
    if output.len() != vnnlib.num_outputs || !concrete_violates(&output_arr, vnnlib) {
        return None;
    }
    info!("Fractional-head: root-center point concretely violates; reporting sat");
    Some(BetaCrownResult {
        result: BabVerificationStatus::Violated {
            counterexample: center,
            output,
        },
        domains_explored: 1,
        domains_verified: 0,
        cuts_generated: 0,
        max_depth_reached: 0,
        time_elapsed: start.elapsed(),
        output_bounds: None,
    })
}

// ---------------------------------------------------------------------------
// Diagnostics probe (NY_FRAC_HEAD_PROBE=1): frontier slack attribution.
// Never affects verification results — logging and counters only.
// ---------------------------------------------------------------------------

mod probe {
    use super::*;
    use std::time::Duration;

    pub(super) fn enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var_os("NY_FRAC_HEAD_PROBE").is_some_and(|v| v == "1"))
    }

    static THRESHOLD_NS: AtomicU64 = AtomicU64::new(0);
    static VERTEX_NS: AtomicU64 = AtomicU64::new(0);
    static CALLS: AtomicU64 = AtomicU64::new(0);
    /// Calls where the threshold bound was strictly inside the vertex bound
    /// on the driven end (threshold "won"), and vice versa.
    static THRESH_WINS_LO: AtomicU64 = AtomicU64::new(0);
    static VERTEX_WINS_LO: AtomicU64 = AtomicU64::new(0);
    static THRESH_WINS_HI: AtomicU64 = AtomicU64::new(0);
    static VERTEX_WINS_HI: AtomicU64 = AtomicU64::new(0);
    static FALLBACKS: AtomicU64 = AtomicU64::new(0);

    pub(super) fn count_fallback() {
        if enabled() {
            FALLBACKS.fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    pub(super) fn record_bound_timing(
        threshold_time: Duration,
        vertex_time: Duration,
        threshold: Option<(f64, f64)>,
        vertex: Option<(f64, f64)>,
        need: BoundNeed,
    ) {
        if !enabled() {
            return;
        }
        THRESHOLD_NS.fetch_add(threshold_time.as_nanos() as u64, AtomicOrdering::Relaxed);
        VERTEX_NS.fetch_add(vertex_time.as_nanos() as u64, AtomicOrdering::Relaxed);
        CALLS.fetch_add(1, AtomicOrdering::Relaxed);
        if let (Some((tl, tu)), Some((vl, vu))) = (threshold, vertex) {
            if need.lower() {
                if tl > vl {
                    THRESH_WINS_LO.fetch_add(1, AtomicOrdering::Relaxed);
                } else if vl > tl {
                    VERTEX_WINS_LO.fetch_add(1, AtomicOrdering::Relaxed);
                }
            }
            if need.upper() {
                if tu < vu {
                    THRESH_WINS_HI.fetch_add(1, AtomicOrdering::Relaxed);
                } else if vu < tu {
                    VERTEX_WINS_HI.fetch_add(1, AtomicOrdering::Relaxed);
                }
            }
        }
    }

    pub(super) fn report_timing() {
        info!(
            "[probe timing] bound calls={} threshold={:.2}s vertex={:.2}s fallbacks={}; \
             driven-end wins: thresh lo={} hi={} / vertex lo={} hi={}",
            CALLS.load(AtomicOrdering::Relaxed),
            THRESHOLD_NS.load(AtomicOrdering::Relaxed) as f64 / 1e9,
            VERTEX_NS.load(AtomicOrdering::Relaxed) as f64 / 1e9,
            FALLBACKS.load(AtomicOrdering::Relaxed),
            THRESH_WINS_LO.load(AtomicOrdering::Relaxed),
            THRESH_WINS_HI.load(AtomicOrdering::Relaxed),
            VERTEX_WINS_LO.load(AtomicOrdering::Relaxed),
            VERTEX_WINS_HI.load(AtomicOrdering::Relaxed),
        );
    }

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn rand_unit(state: &mut u64) -> f64 {
        (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
    }

    pub(super) fn probe_net(state: &NetState<'_>, input_shape: &[usize], label: &str) {
        let mut leaves: Vec<&HeapLeaf> = state.heap.iter().collect();
        leaves.sort_by(|a, b| b.key.total_cmp(&a.key));
        let k = |i: usize| leaves.get(i).map(|h| h.key);
        info!(
            "[probe {label}] objective={} live={} frozen_active={} passive={:.4} \
             keys[0]={:?} [7]={:?} [63]={:?} [511]={:?}",
            match state.objective {
                Objective::Min => "min",
                Objective::Max => "max",
            },
            state.live_leaves,
            state.frozen_active,
            state.passive,
            k(0),
            k(7),
            k(63),
            k(511),
        );
        let depths: Vec<u32> = leaves.iter().take(8).map(|h| h.leaf.depth).collect();
        info!(
            "[probe {label}] top-8 depths={depths:?} max_depth={}",
            state.max_depth
        );
        for (rank, hl) in leaves.iter().take(2).enumerate() {
            probe_leaf(state.head, &hl.leaf, input_shape, label, rank);
        }
    }

    #[allow(clippy::too_many_lines)]
    fn probe_leaf(head: &HeadPlan, leaf: &Leaf, input_shape: &[usize], label: &str, rank: usize) {
        let n = head.coeffs.len();
        let lo_f32: Vec<f32> = leaf.lo.iter().map(|&x| f32_down(x)).collect();
        let hi_f32: Vec<f32> = leaf.hi.iter().map(|&x| f32_up(x)).collect();
        let Ok(lo_arr) = ArrayD::from_shape_vec(IxDyn(input_shape), lo_f32) else {
            return;
        };
        let Ok(hi_arr) = ArrayD::from_shape_vec(IxDyn(input_shape), hi_f32) else {
            return;
        };
        let Ok(input) = ny_tensor::BoundedTensor::new(lo_arr, hi_arr) else {
            return;
        };

        let outcome = bound_head_threshold(head, &input, BoundNeed::Both);
        let (threshold, den_vertex, den_range) = match &outcome {
            Some(o) => {
                let dv = o.pbox.as_ref().and_then(|pb| {
                    head_range_with_denominator(
                        &pb.p_lo,
                        &pb.p_hi,
                        &head.coeffs,
                        head.bias,
                        pb.den_lo,
                        pb.den_hi,
                    )
                });
                let dr = o.pbox.as_ref().map(|pb| (pb.den_lo, pb.den_hi));
                (o.s, dv, dr)
            }
            None => (None, None, None),
        };
        let vertex = bound_head_vertex(head, &input);

        // Prefix CROWN logit boxes.
        let crown_boxes: Option<(Vec<f64>, Vec<f64>)> = head
            .prefix
            .propagate_crown_fixed_slope(&input)
            .ok()
            .and_then(|r| {
                let (rl, ru) = r.lower_upper();
                (rl.len() == n).then(|| {
                    (
                        rl.iter().map(|&x| f64::from(x).max(0.0)).collect(),
                        ru.iter().map(|&x| f64::from(x).max(0.0)).collect(),
                    )
                })
            });

        // Monte-Carlo truth estimates over the leaf box.
        let mut rng = 0xA5A5_5A5A_DEAD_BEEFu64 ^ (leaf.depth as u64) ^ ((rank as u64) << 32);
        let samples = 2048usize;
        let mut emp_lo = vec![f64::INFINITY; n];
        let mut emp_hi = vec![f64::NEG_INFINITY; n];
        let mut s_min = f64::INFINITY;
        let mut s_max = f64::NEG_INFINITY;
        let mut ok = 0usize;
        for _ in 0..samples {
            let point: Vec<f32> = leaf
                .lo
                .iter()
                .zip(&leaf.hi)
                .map(|(&l, &h)| (l + (h - l) * rand_unit(&mut rng)) as f32)
                .collect();
            let Some(r) = point_forward(&head.prefix, &point, input_shape) else {
                continue;
            };
            if r.len() != n {
                continue;
            }
            for i in 0..n {
                emp_lo[i] = emp_lo[i].min(r[i]);
                emp_hi[i] = emp_hi[i].max(r[i]);
            }
            if let Some(s) = head_formula(&r, &head.coeffs, head.bias, head.exponent) {
                s_min = s_min.min(s);
                s_max = s_max.max(s);
                ok += 1;
            }
        }
        // Vertex bound on the EMPIRICAL logit boxes (separates prefix-box
        // slack from box-level correlation loss).
        let emp_vertex = if ok > 0 {
            let pl: Vec<f64> = emp_lo
                .iter()
                .map(|&l| pow_down(l.max(0.0), head.exponent))
                .collect();
            let pu: Vec<f64> = emp_hi
                .iter()
                .map(|&u| pow_up(u.max(0.0), head.exponent))
                .collect();
            head_range(&pl, &pu, &head.coeffs, head.bias)
        } else {
            None
        };

        let wide_dims: Vec<(usize, f64)> = head
            .dims
            .iter()
            .filter_map(|&(d, _)| {
                let w = leaf.hi[d] - leaf.lo[d];
                (w > 0.0).then_some((d, w))
            })
            .collect();
        info!(
            "[probe {label} leaf#{rank}] depth={} inherited={:?} threshold={:?} vertex={:?} \
             den_vertex={:?} den_range={:?}",
            leaf.depth, leaf.s, threshold, vertex, den_vertex, den_range
        );
        info!(
            "[probe {label} leaf#{rank}] mc({ok} pts): s_true~[{s_min:.4}, {s_max:.4}] \
             emp_vertex={emp_vertex:?}",
        );
        if let Some((cl, cu)) = &crown_boxes {
            let crown_w: Vec<f64> = cl.iter().zip(cu).map(|(&l, &u)| u - l).collect();
            let emp_w: Vec<f64> = emp_lo
                .iter()
                .zip(&emp_hi)
                .map(|(&l, &u)| (u - l).max(0.0))
                .collect();
            info!(
                "[probe {label} leaf#{rank}] logit crown lo={:?}",
                cl.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>()
            );
            info!(
                "[probe {label} leaf#{rank}] logit crown hi={:?}",
                cu.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>()
            );
            info!(
                "[probe {label} leaf#{rank}] logit widths crown={:?} emp={:?}",
                crown_w
                    .iter()
                    .map(|x| format!("{x:.3}"))
                    .collect::<Vec<_>>(),
                emp_w.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>()
            );
        }
        info!(
            "[probe {label} leaf#{rank}] wide dims (idx,width)={:?} coeffs={:?} bias={:.4} k={}",
            wide_dims
                .iter()
                .map(|(d, w)| format!("({d},{w:.4})"))
                .collect::<Vec<_>>(),
            head.coeffs
                .iter()
                .map(|x| format!("{x:.3}"))
                .collect::<Vec<_>>(),
            head.bias,
            head.exponent,
        );
    }
}

#[cfg(test)]
mod tests;
