// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #pwl-lookup: the exact 1-D range of a ReLU LOOKUP TABLE (DARK,
//! `bab.pwl_lookup_exact_range`).
//!
//! ## The motif
//!
//! The "linear" ml4acopf_2024 nets implement their output Sigmoid and their
//! Cos/Sin corrections as piecewise-linear lookup tables:
//!
//! ```text
//! s = bias + Σ_k w_k · ReLU(x − b_k)          (k = 1..K, K = 18 / 54 / 55)
//! ONNX:  Unsqueeze(x, -1) → Sub(·, b[K]) → Relu → MatMul(·, w[K]) [→ Add(bias)]
//! ny:    <x: last axis 1> → SubConstant(b) → ReLU → Linear(W ∈ R^{1×K}) → Squeeze(-1)
//! abc:   ReduceSum(Mul(Relu(Sub(Unsqueeze(x, -1), b)), w), -1)
//! ```
//!
//! Hinge-by-hinge IBP bounds each `w_k · ReLU(x − b_k)` independently and sums
//! the K intervals, which forgets that every hinge reads the SAME scalar `x`.
//! On a 54-hinge saturating sigmoid that costs ~1e-6..1e-2 of width, and the
//! generation-bound properties (prop3/prop4) have margins of 1e-6..6.5e-6: all
//! 12 prop3/prop4 rows on the six linear nets sit `unknown`/`timeout` while the
//! SAME properties on the native-Sigmoid nets bank `unsat` in ~5 s.
//!
//! ## What this module computes
//!
//! `f(x) = bias + Σ_k w_k · max(0, x − b_k)` is continuous and affine between
//! consecutive breakpoints, so over `[l, u]` its minimum and maximum are
//! attained on the finite candidate set `{l, u} ∪ {b_k : l < b_k < u}`. Each
//! candidate is evaluated in f64 with EVERY operation rounded outward
//! (`next_down`/`next_up` on each difference, product and partial sum — an
//! interval enclosure of the exact real value, at least as strong as a Higham
//! `γ_n·Σ|w_k||t − b_k|` charge), and the resulting f64 range is rounded
//! outward to f32 with [`cast_f64_to_f32_down`] / [`cast_f64_to_f32_up`].
//!
//! The per-element result is INTERSECTED with the plain node box
//! (shrink-only). Both are enclosures of the same value over the same input
//! box, so an empty intersection is impossible unless one of them is wrong: in
//! that case the WHOLE plain box is kept (fail closed — an empty box is never
//! published) and a `contradiction` is counted and printed. A contradiction on
//! a scored row is a stop-and-report event.
//!
//! ## Where it reads its inputs
//!
//! The scalar `x` enclosure is read from the bound map entry of the
//! SubConstant's INPUT node (whatever produced it — an Unsqueeze on the
//! authored nets). No Unsqueeze semantics are assumed: the runtime shape checks
//! require that input to end in an axis of size 1, the SubConstant output to be
//! that shape with the last axis widened to K, and the tightened node's output
//! to carry exactly one element per scalar `x` in the same row-major order.
//! Anything else DECLINES and the plain box is published untouched.
//!
//! ## Delivery (dark by default)
//!
//! Graph-local flag [`GraphNetwork::set_pwl_lookup_exact_range`], stamped by the
//! CLI from `BetaCrownConfig::pwl_lookup_exact_range` (preset key
//! `bab.pwl_lookup_exact_range`, default `false`). Unarmed, every entry point
//! returns the plain box it was given, byte-for-byte; no shipped yaml arms it.
//!
//! ## Observable
//!
//! Unconditional (armed only), logarithmically rate-limited:
//! `[pwl-lookup] armed=1 calls=<n> tables=<n> hinges=<k,…> elems=<n>
//! tightened=<n> contradiction=<n> declined=<n>`, plus a census line the CLI
//! prints once per armed run (`[pwl-lookup] armed=1 census tables=<n> …`), so a
//! null result is distinguishable from "never engaged".

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use ny_tensor::{cast_f64_to_f32_down, cast_f64_to_f32_up, BoundedTensor};

use crate::layers::Layer;
use crate::network::{GraphNetwork, GraphNode};

/// Elements processed between wall-clock polls of the (optional) deadline.
const PWL_POLL_ELEMENTS: usize = 64;

/// Upper bound on `elements × candidates × hinges` scalar work per node
/// before the tightening declines (the plain box is always a valid answer).
/// The ml4acopf 300-bus sigmoid table is `438 × 56 × 54 ≈ 1.3e6`.
const PWL_MAX_WORK: u128 = 1 << 30;

/// Which authored spelling of the lookup table matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PwlLookupForm {
    /// `ReLU → Linear(W ∈ R^{1×K}, optional bias of length 1)`: ONNX
    /// `MatMul(relu, w[K])` after ny's 1-D weight promotion (the builder's
    /// trailing `Squeeze(-1)` follows and is exact).
    LinearRow,
    /// `ReLU → MulConstant(w[K]) → ReduceSum(last axis)`: the spelling
    /// alpha-beta-CROWN's `merge_relu_lookup_table` optimizer matches.
    ReduceSumMul {
        /// Whether the reduced axis is kept (size 1) in the output.
        keepdims: bool,
    },
}

/// A structurally matched lookup table, with its constants copied out.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PwlLookupTable {
    /// The node whose output is the scalar `x` (last axis of size 1).
    pub(crate) scalar_input: String,
    /// The `SubConstant(b)` node (its bounds give the runtime K-axis check).
    pub(crate) sub_node: String,
    /// `MulConstant(w)` node of the ReduceSum spelling (shape check only).
    pub(crate) mul_node: Option<String>,
    /// Breakpoints `b_k` (row-major order of the constant's last axis).
    pub(crate) thresholds: Vec<f32>,
    /// Hinge weights `w_k`, same order and length as `thresholds`.
    pub(crate) weights: Vec<f32>,
    /// Constant added after the weighted sum (Linear bias; 0 otherwise).
    pub(crate) bias: f32,
    pub(crate) form: PwlLookupForm,
}

/// The values of a constant that varies ONLY along its last axis (every other
/// axis has size 1). `None` for a rank-0 constant or any other layout.
fn last_axis_vector(constant: &ndarray::ArrayD<f32>) -> Option<Vec<f32>> {
    let shape = constant.shape();
    let (&last, leading) = shape.split_last()?;
    if last == 0 || leading.iter().any(|&d| d != 1) {
        return None;
    }
    Some(constant.iter().copied().collect())
}

/// Directed-rounding f64 enclosure `[lo, hi]` of the EXACT real value
/// `f(t) = bias + Σ_k w_k · max(0, t − b_k)` at the exact point `t`.
///
/// Every difference, product and partial sum is rounded to nearest and then
/// stepped one f64 ULP outward, so each intermediate interval contains the
/// exact real intermediate (round-to-nearest error is at most half an ULP; the
/// one-ULP step covers it, including at binade boundaries). A difference that
/// rounds to exactly `0.0` is exact (gradual underflow: `a − b == 0` iff
/// `a == b`), so an inactive hinge at its own breakpoint contributes exactly
/// zero. f32-derived operands cannot overflow or underflow in f64 here:
/// `|w|·|t − b| <= 3.4e38 · 6.8e38 ≪ f64::MAX`, and a nonzero difference or
/// product of f32 values is far above the f64 normal range floor.
pub(crate) fn pwl_point_enclosure(
    t: f64,
    thresholds: &[f32],
    weights: &[f32],
    bias: f32,
) -> (f64, f64) {
    let mut lo = f64::from(bias);
    let mut hi = lo;
    for (&b, &w) in thresholds.iter().zip(weights) {
        let d = t - f64::from(b);
        if d == 0.0 {
            // Exact: the hinge sits exactly at its breakpoint, ReLU(0) = 0.
            continue;
        }
        let (d_lo, d_hi) = (d.next_down(), d.next_up());
        let (r_lo, r_hi) = (d_lo.max(0.0), d_hi.max(0.0));
        if r_hi == 0.0 {
            // Inactive over the whole difference enclosure: contributes 0.
            continue;
        }
        let w = f64::from(w);
        let (p_lo, p_hi) = if w >= 0.0 {
            ((w * r_lo).next_down(), (w * r_hi).next_up())
        } else {
            ((w * r_hi).next_down(), (w * r_lo).next_up())
        };
        lo = (lo + p_lo).next_down();
        hi = (hi + p_hi).next_up();
    }
    (lo, hi)
}

/// Certified f64 enclosure of the RANGE of `f` over `[l, u]`: the hull of the
/// point enclosures at `{l, u} ∪ {b_k : l < b_k < u}`.
///
/// `None` on a non-finite or inverted interval, mismatched constant lengths,
/// or a non-finite result.
pub(crate) fn pwl_range_f64(
    l: f32,
    u: f32,
    thresholds: &[f32],
    weights: &[f32],
    bias: f32,
) -> Option<(f64, f64)> {
    if !(l.is_finite() && u.is_finite() && bias.is_finite())
        || l > u
        || thresholds.is_empty()
        || thresholds.len() != weights.len()
    {
        return None;
    }
    let (lf, uf) = (f64::from(l), f64::from(u));
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut visit = |t: f64| {
        let (a, b) = pwl_point_enclosure(t, thresholds, weights, bias);
        lo = lo.min(a);
        hi = hi.max(b);
    };
    visit(lf);
    if uf > lf {
        visit(uf);
    }
    for &b in thresholds {
        let bf = f64::from(b);
        if bf > lf && bf < uf {
            visit(bf);
        }
    }
    (lo.is_finite() && hi.is_finite() && lo <= hi).then_some((lo, hi))
}

/// [`pwl_range_f64`] rounded OUTWARD to f32 (tightest directed conversion).
pub(crate) fn pwl_range_f32(
    l: f32,
    u: f32,
    thresholds: &[f32],
    weights: &[f32],
    bias: f32,
) -> Option<(f32, f32)> {
    let (lo, hi) = pwl_range_f64(l, u, thresholds, weights, bias)?;
    let (lo, hi) = (cast_f64_to_f32_down(lo), cast_f64_to_f32_up(hi));
    (lo.is_finite() && hi.is_finite() && lo <= hi).then_some((lo, hi))
}

/// Runtime layout check for a matched table. `Err(reason)` declines.
///
/// * `scalar_shape` — bounds of the SubConstant's input: rank >= 1, last axis 1;
/// * `sub_shape`    — SubConstant output: `scalar_shape` with the last axis K;
/// * `mul_shape`    — ReduceSum spelling only: the MulConstant output must not
///   broadcast beyond `sub_shape`;
/// * `out_shape`    — the tightened node: one element per scalar `x`, same
///   row-major order (`scalar_shape`, or its prefix for a non-keepdims sum).
pub(crate) fn check_lookup_shapes(
    table: &PwlLookupTable,
    scalar_shape: &[usize],
    sub_shape: &[usize],
    mul_shape: Option<&[usize]>,
    out_shape: &[usize],
) -> Result<(), &'static str> {
    let k = table.thresholds.len();
    let Some((&scalar_last, prefix)) = scalar_shape.split_last() else {
        return Err("scalar input is rank 0");
    };
    if scalar_last != 1 {
        return Err("scalar input's last axis is not 1");
    }
    match sub_shape.split_last() {
        Some((&sub_last, sub_prefix)) if sub_last == k && sub_prefix == prefix => {}
        _ => return Err("sub output is not the scalar shape widened to K on the last axis"),
    }
    match table.form {
        PwlLookupForm::LinearRow => {
            if out_shape != scalar_shape {
                return Err("linear output shape differs from the scalar input shape");
            }
        }
        PwlLookupForm::ReduceSumMul { keepdims } => {
            if mul_shape != Some(sub_shape) {
                return Err("mul output shape differs from the sub output shape");
            }
            let expected: &[usize] = if keepdims { scalar_shape } else { prefix };
            if out_shape != expected {
                return Err("reduce output shape differs from the scalar input layout");
            }
        }
    }
    Ok(())
}

/// Result of one tightening attempt on one node.
#[derive(Debug)]
pub(crate) enum PwlTightenOutcome {
    /// A (possibly identical) shrink-only box. `tightened` counts elements
    /// whose interval strictly narrowed; `elems` counts elements evaluated.
    Applied {
        /// Boxed: a `BoundedTensor` made `Applied` 280 bytes against 24
        /// (clippy::large_enum_variant, workspace threshold 64).
        bounds: Box<BoundedTensor>,
        elems: usize,
        tightened: usize,
    },
    /// The exact range and the plain box were DISJOINT at `element`: one of
    /// the two enclosures is wrong. The caller keeps the plain box.
    Contradiction {
        element: usize,
        plain: (f32, f32),
        range: (f32, f32),
    },
    /// Not applied (deadline, cost cap, element-count mismatch, NaN input).
    Declined(&'static str),
}

/// Pure kernel: intersect `plain` elementwise with the exact lookup range of
/// each scalar `x` enclosure in `scalar`. Shapes must already have passed
/// [`check_lookup_shapes`]; the element counts are re-checked here.
pub(crate) fn tighten_lookup_output(
    table: &PwlLookupTable,
    scalar: &BoundedTensor,
    plain: &BoundedTensor,
    deadline: Option<Instant>,
) -> PwlTightenOutcome {
    let n = plain.lower().len();
    if scalar.lower().len() != n || plain.upper().len() != n || scalar.upper().len() != n {
        return PwlTightenOutcome::Declined("element count mismatch");
    }
    let k = table.thresholds.len() as u128;
    if (n as u128).saturating_mul(k + 2).saturating_mul(k) > PWL_MAX_WORK {
        return PwlTightenOutcome::Declined("cost cap");
    }
    let mut lower = plain.lower().to_owned();
    let mut upper = plain.upper().to_owned();
    let mut tightened = 0usize;
    let mut elems = 0usize;
    for (index, (((&xl, &xu), lo), hi)) in scalar
        .lower()
        .iter()
        .zip(scalar.upper().iter())
        .zip(lower.iter_mut())
        .zip(upper.iter_mut())
        .enumerate()
    {
        if index % PWL_POLL_ELEMENTS == 0 && deadline.is_some_and(|d| Instant::now() >= d) {
            return PwlTightenOutcome::Declined("deadline");
        }
        let (pl, pu) = (*lo, *hi);
        if pl.is_nan() || pu.is_nan() {
            // Never mask a NaN: the caller's NaN firewall must still see it.
            return PwlTightenOutcome::Declined("nan in plain bounds");
        }
        let Some((rl, ru)) = pwl_range_f32(xl, xu, &table.thresholds, &table.weights, table.bias)
        else {
            // Non-finite / inverted scalar enclosure: keep this element plain.
            continue;
        };
        elems += 1;
        let (nl, nu) = (pl.max(rl), pu.min(ru));
        if nl > nu {
            return PwlTightenOutcome::Contradiction {
                element: index,
                plain: (pl, pu),
                range: (rl, ru),
            };
        }
        if nl > pl || nu < pu {
            tightened += 1;
        }
        *lo = nl;
        *hi = nu;
    }
    match BoundedTensor::new_allow_infinite(lower, upper) {
        Ok(bounds) => PwlTightenOutcome::Applied {
            bounds: Box::new(bounds),
            elems,
            tightened,
        },
        Err(_) => PwlTightenOutcome::Declined("rebuild refused"),
    }
}

// ---------------------------------------------------------------------------
// Engagement telemetry (process-global; graph clones are per-domain).
// ---------------------------------------------------------------------------

static PWL_CALLS: AtomicU64 = AtomicU64::new(0);
static PWL_ELEMS: AtomicU64 = AtomicU64::new(0);
static PWL_TIGHTENED: AtomicU64 = AtomicU64::new(0);
static PWL_CONTRADICTIONS: AtomicU64 = AtomicU64::new(0);
static PWL_DECLINED: AtomicU64 = AtomicU64::new(0);
/// Distinct engaged tables `(node name, K)` in first-seen order.
static PWL_TABLES: Mutex<Vec<(String, usize)>> = Mutex::new(Vec::new());

/// Record one armed attempt and print the rate-limited engagement line
/// (every power-of-two call count, and every contradiction).
fn record_pwl_attempt(node_name: &str, k: usize, outcome: &PwlTightenOutcome) {
    let calls = PWL_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    let mut contradiction_detail = None;
    match outcome {
        PwlTightenOutcome::Applied {
            elems, tightened, ..
        } => {
            PWL_ELEMS.fetch_add(*elems as u64, Ordering::Relaxed);
            PWL_TIGHTENED.fetch_add(*tightened as u64, Ordering::Relaxed);
        }
        PwlTightenOutcome::Contradiction {
            element,
            plain,
            range,
        } => {
            PWL_CONTRADICTIONS.fetch_add(1, Ordering::Relaxed);
            contradiction_detail = Some(format!(
                "node={node_name} element={element} plain=[{:e},{:e}] range=[{:e},{:e}]",
                plain.0, plain.1, range.0, range.1
            ));
        }
        PwlTightenOutcome::Declined(reason) => {
            PWL_DECLINED.fetch_add(1, Ordering::Relaxed);
            tracing::trace!(node = node_name, reason, "pwl lookup tightening declined");
        }
    }
    let (tables, hinges) = {
        let mut seen = PWL_TABLES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !seen.iter().any(|(name, _)| name == node_name) {
            seen.push((node_name.to_string(), k));
        }
        let hinges = seen
            .iter()
            .map(|(_, k)| k.to_string())
            .collect::<Vec<_>>()
            .join(",");
        (seen.len(), hinges)
    };
    if let Some(detail) = contradiction_detail.as_deref() {
        eprintln!("[pwl-lookup] CONTRADICTION (plain box kept; stop and report) {detail}");
    }
    if calls.is_power_of_two() || contradiction_detail.is_some() {
        eprintln!(
            "[pwl-lookup] armed=1 calls={calls} tables={tables} hinges={hinges} elems={} \
             tightened={} contradiction={} declined={}",
            PWL_ELEMS.load(Ordering::Relaxed),
            PWL_TIGHTENED.load(Ordering::Relaxed),
            PWL_CONTRADICTIONS.load(Ordering::Relaxed),
            PWL_DECLINED.load(Ordering::Relaxed),
        );
    }
}

impl GraphNetwork {
    /// Arm or disarm the exact lookup-table range on THIS graph (and every
    /// later clone). Stamped by the verifier from
    /// `BetaCrownConfig::pwl_lookup_exact_range`; default `false`.
    ///
    /// No bound cache is invalidated: a map computed in either state is a
    /// certified enclosure of the same function over the same box, so serving
    /// one to the other is sound (only its tightness differs).
    pub fn set_pwl_lookup_exact_range(&mut self, enabled: bool) {
        self.pwl_lookup_armed = enabled;
    }

    /// Whether the exact lookup-table range is armed on this graph.
    pub fn pwl_lookup_exact_range(&self) -> bool {
        self.pwl_lookup_armed
    }

    /// Structural match of the lookup-table motif ending at `node` (see the
    /// module docs). `None` on ANY deviation: non-constant or non-last-axis
    /// constants, a reversed Sub, a weight that is not a single row, a bias
    /// that is not a scalar, mismatched K, or a non-finite constant.
    pub(crate) fn match_pwl_lookup(&self, node: &GraphNode) -> Option<PwlLookupTable> {
        let (relu_name, weights, bias, form, mul_node) = match node.layer() {
            Layer::Linear(linear) => {
                let [relu_name] = node.inputs() else {
                    return None;
                };
                let weight = linear.weight();
                if weight.nrows() != 1 || weight.ncols() == 0 {
                    return None;
                }
                let bias = match linear.bias() {
                    None => 0.0,
                    Some(bias) if bias.len() == 1 => bias[0],
                    Some(_) => return None,
                };
                (
                    relu_name.as_str(),
                    weight.row(0).to_vec(),
                    bias,
                    PwlLookupForm::LinearRow,
                    None,
                )
            }
            Layer::ReduceSum(reduce) => {
                let [mul_name] = node.inputs() else {
                    return None;
                };
                // ONLY the negative spelling: `-1` is the last axis under
                // both the ONNX (batched) and ny's batch-stripped internal
                // convention, whereas a non-negative index is ambiguous
                // between the two — and a sum over the WRONG axis could
                // still pass the shape checks when the element count
                // happens to equal K. Declining costs engagement only.
                if reduce.axes.as_slice() != [-1] {
                    return None;
                }
                let mul = self.node(mul_name)?;
                let Layer::MulConstant(scale) = mul.layer() else {
                    return None;
                };
                let [relu_name] = mul.inputs() else {
                    return None;
                };
                (
                    relu_name.as_str(),
                    last_axis_vector(scale.constant())?,
                    0.0,
                    PwlLookupForm::ReduceSumMul {
                        keepdims: reduce.keepdims,
                    },
                    Some(mul_name.clone()),
                )
            }
            _ => return None,
        };
        let relu = self.node(relu_name)?;
        if !matches!(relu.layer(), Layer::ReLU(_)) {
            return None;
        }
        let [sub_name] = relu.inputs() else {
            return None;
        };
        let sub = self.node(sub_name)?;
        let Layer::SubConstant(shift) = sub.layer() else {
            return None;
        };
        if shift.reverse {
            return None;
        }
        let [scalar_input] = sub.inputs() else {
            return None;
        };
        let thresholds = last_axis_vector(shift.constant())?;
        if thresholds.len() != weights.len()
            || !bias.is_finite()
            || !thresholds
                .iter()
                .chain(weights.iter())
                .all(|v| v.is_finite())
        {
            return None;
        }
        Some(PwlLookupTable {
            scalar_input: scalar_input.clone(),
            sub_node: sub_name.clone(),
            mul_node,
            thresholds,
            weights,
            bias,
            form,
        })
    }

    /// Structural census of every lookup table in this graph, in execution
    /// order: `(tightened node name, K)`. Independent of the armed flag, so the
    /// CLI can print what an armed run WOULD engage on.
    pub fn pwl_lookup_census(&self) -> Vec<(String, usize)> {
        let Ok(order) = self.exec_order() else {
            return Vec::new();
        };
        order
            .iter()
            .filter_map(|name| {
                let node = self.node(name)?;
                self.match_pwl_lookup(node)
                    .map(|table| (name.clone(), table.thresholds.len()))
            })
            .collect()
    }

    /// The IBP hook: when ARMED and `node` ends a lookup table whose runtime
    /// layout checks pass, return `plain` intersected with the exact range;
    /// otherwise return `plain` unchanged (byte-for-byte). Never errors: every
    /// failure mode keeps the plain box, which is always a valid answer.
    pub(crate) fn pwl_lookup_tighten_node(
        &self,
        node_name: &str,
        node: &GraphNode,
        plain: BoundedTensor,
        input: &BoundedTensor,
        cache: &HashMap<String, BoundedTensor>,
        deadline: Option<Instant>,
    ) -> BoundedTensor {
        if !self.pwl_lookup_armed || !matches!(node.layer(), Layer::Linear(_) | Layer::ReduceSum(_))
        {
            return plain;
        }
        let Some(table) = self.match_pwl_lookup(node) else {
            return plain;
        };
        let k = table.thresholds.len();
        let outcome = match (
            self.bounds_ref(&table.scalar_input, input, cache),
            self.bounds_ref(&table.sub_node, input, cache),
        ) {
            (Ok(scalar), Ok(sub)) => {
                let mul_shape = table
                    .mul_node
                    .as_deref()
                    .and_then(|name| self.bounds_ref(name, input, cache).ok())
                    .map(BoundedTensor::shape);
                match check_lookup_shapes(
                    &table,
                    scalar.shape(),
                    sub.shape(),
                    mul_shape,
                    plain.shape(),
                ) {
                    Ok(()) => tighten_lookup_output(&table, scalar, &plain, deadline),
                    Err(reason) => PwlTightenOutcome::Declined(reason),
                }
            }
            _ => PwlTightenOutcome::Declined("input bounds unavailable"),
        };
        record_pwl_attempt(node_name, k, &outcome);
        match outcome {
            PwlTightenOutcome::Applied { bounds, .. } => *bounds,
            PwlTightenOutcome::Contradiction { .. } | PwlTightenOutcome::Declined(_) => plain,
        }
    }
}

#[cfg(test)]
#[path = "pwl_lookup_tests.rs"]
mod tests;
