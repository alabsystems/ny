// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Joint (shared-variable) relaxation for `ReduceSum(A * B)`.
//!
//! ## What this is for
//!
//! A `MulBinary` node whose two operands are *different functions of the same
//! input* is relaxed by every factor-wise method — McCormick, interval product,
//! `mul.middle` — as if the two factors were independent. On a Lyapunov
//! decrease condition stated as `(f(x) - x)^T Q (f(x) + x)` that discards
//! exactly the correlation that makes the property true: the two factors share
//! `x`, so at the maximizing `x` of one factor the other is not free.
//!
//! MEASURED on `lsnc_relu` quadrotor2d row 2 (the `S2` term of the attribution
//! in `reports/measured-2026/lsnc_relu_artifacts`): boxes centred on the
//! certified maximizer, factor-wise CROWN needs ~48 input splits before its
//! upper bound on `DV` drops below the `1e-6` threshold, and the joint bound
//! below needs ~24 — a factor of `2^24` in domain count.
//!
//! ## What it computes
//!
//! With `A_i(x)` and `B_i(x)` each enclosed by a CROWN affine pair on the
//! current box, write each as a midpoint affine plus a bounded residual:
//!
//! ```text
//!   A_i(x) = a_i(x) + ea_i,  |ea_i| <= ra_i
//!   B_i(x) = b_i(x) + eb_i,  |eb_i| <= rb_i
//! ```
//!
//! Then
//!
//! ```text
//!   sum_i A_i B_i = sum_i a_i(x) b_i(x)          <- ONE quadratic in x
//!                 + sum_i (a_i eb_i + b_i ea_i + ea_i eb_i)
//! ```
//!
//! The first term is a single quadratic form over the input box, bounded
//! rigorously below; the second is bounded by interval arithmetic. The
//! correlation survives because the quadratic is maximized *jointly* over one
//! `x`, not factor by factor.
//!
//! ## Why the quadratic bound is rigorous
//!
//! For `q(x) = x'Hx + g'x + c0` with `x = m + d`, `|d| <= w`:
//!
//! ```text
//!   q = q(m) + (2 H_s m + g)'d + d' H_s d
//! ```
//!
//! The linear term is bounded exactly by `sum_j |grad_j| w_j`. For the
//! quadratic term take ANY real `U` (columns `u_k`) and ANY real `lambda_k`
//! and split
//!
//! ```text
//!   d' H_s d = sum_k lambda_k (u_k'd)^2 + d' E d,   E = H_s - U diag(lambda) U'
//! ```
//!
//! which is an identity for every `U`, `lambda` — no orthogonality and no
//! accuracy assumption is needed. Then `(u_k'd)^2 <= (|u_k|'w)^2` and
//! `d'Ed <= (|E| w)'w`. `U`/`lambda` come from a plain cyclic Jacobi sweep, so
//! a poor decomposition only loosens the bound through a larger `E`; it can
//! never make it unsound.
//!
//! ## Where it is applied
//!
//! The tightened scalar enclosure replaces the `ReduceSum` node's box, and a
//! downstream IBP resweep — through `GraphNetwork::node_ibp_step`, the exact
//! per-node dispatch the foundational forward pass uses — carries it to the
//! graph output. Every step is an intersection of two valid enclosures, so the
//! pass is shrink-only and fail-open: any refusal keeps the caller's bounds.
//!
//! DARK BY DEFAULT behind [`ny_levers::decls::dark_probes::JOINT_BILINEAR_QUAD`].
//! This is a bound path on a family where ny has published false `unsat`
//! verdicts before (retracted in `807bf511`). A typed preset route exists so a
//! deliberate measured treatment can reach the competition binary, but no
//! shipped preset arms it; every proof it enables still requires adjudication
//! against an independent reference.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use ndarray::Array2;
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;

use crate::bounds::LinearBounds;
use crate::layers::Layer;
use crate::GraphNetwork;

/// Largest flattened input dimension the joint bound will run at. The quadratic
/// assembly is `O(n*d^2)` and the Jacobi sweep `O(d^3)`; both are trivial for the
/// low-dimensional control benchmarks this targets and neither is affordable on
/// an image input, so the pass declines rather than paying for a refusal later.
const MAX_INPUT_DIM: usize = 64;
/// Largest number of summed products in one bilinear reduction.
const MAX_TERMS: usize = 256;
/// Largest graph this pass will touch. Deliberately small: the input-linear
/// certificate below runs UNINTERRUPTIBLY (see the `None` deadline at its call
/// site), so the node count is the only thing bounding its cost.
const MAX_NODES: usize = 512;
/// Jacobi sweeps. Convergence is a tightness question, never a soundness one.
const JACOBI_SWEEPS: usize = 32;

// Process-global CLI -> propagation bridge. `0` keeps the typed layer absent,
// `1` is explicit OFF, and `2` is explicit ON. Presence must be retained so a
// scored preset can deliberately disarm an inherited choice as well as arm it.
static JOINT_BILINEAR_PRESET: AtomicU8 = AtomicU8::new(0);

fn encode_bool_preset(value: Option<bool>) -> u8 {
    match value {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    }
}

fn decode_bool_preset(value: u8) -> Option<bool> {
    match value {
        1 => Some(false),
        2 => Some(true),
        _ => None,
    }
}

/// Deliver `solver.joint_bilinear_quad` from the CLI's typed preset.
///
/// `None` restores the historical default layer. A present legacy environment
/// value retains precedence in both directions through [`ny_levers::read_over_config`].
pub fn set_joint_bilinear_preset(value: Option<bool>) {
    JOINT_BILINEAR_PRESET.store(encode_bool_preset(value), Ordering::Release);
}

fn joint_bilinear_preset() -> Option<bool> {
    decode_bool_preset(JOINT_BILINEAR_PRESET.load(Ordering::Acquire))
}

/// Whether the joint bilinear relaxation is armed.
///
/// Read through the levers registry (`NY_JOINT_BILINEAR_QUAD`,
/// `decls::dark_probes::JOINT_BILINEAR_QUAD`) over the optional typed preset —
/// this adds no raw environment read. The same central gate is queried both at
/// dense-spec route admission and before the scalar per-domain joint pass.
/// Dark by default: absent preset + absent environment resolves to `false`;
/// either exact environment token wins over the typed value.
pub(crate) fn joint_bilinear_armed() -> bool {
    ny_levers::read_over_config(
        &ny_levers::decls::dark_probes::JOINT_BILINEAR_QUAD,
        joint_bilinear_preset().map(ny_levers::LeverValue::Bool),
    )
    .expect("a typed bool must be admissible for a bool lever")
    .value
    .as_bool()
}

/// Cheap necessary condition for the joint pass to engage on `input`.
///
/// This deliberately checks only conditions that can be decided without the
/// per-domain node-bound map.  `false` guarantees [`joint_bilinear_spec_bounds`]
/// would decline before publishing a bound; `true` only means the graph merits
/// the scalar evaluation needed to inspect shapes and certificates.  Keeping
/// this predicate in the joint module pins its caps and recognized topology to
/// the implementation rather than duplicating them at route call sites.
pub(crate) fn joint_bilinear_may_apply(graph: &GraphNetwork, input: &BoundedTensor) -> bool {
    let input_dim = input.len();
    if input_dim == 0 || input_dim > MAX_INPUT_DIM || graph.nodes.len() > MAX_NODES {
        return false;
    }

    graph.nodes.values().any(|node| {
        if !matches!(&node.layer, Layer::ReduceSum(_)) {
            return false;
        }
        let inputs = node.inputs();
        if inputs.len() != 1 {
            return false;
        }
        graph.nodes.get(inputs[0].as_str()).is_some_and(|source| {
            matches!(&source.layer, Layer::MulBinary(_)) && source.inputs().len() == 2
        })
    })
}

/// One detected `ReduceSum(A * B)` site.
#[derive(Debug, Clone)]
struct BilinearSite {
    /// Name of the `ReduceSum` node whose box the joint bound replaces.
    reduce: String,
    /// The two `MulBinary` operand node names.
    left: String,
    right: String,
    /// Number of summed products.
    terms: usize,
}

/// A CROWN affine pair reduced to midpoint affine + residual radius, in f64.
struct MidpointAffine {
    /// `n x d` midpoint coefficients.
    coeff: Vec<f64>,
    /// `n` midpoint offsets.
    offset: Vec<f64>,
    /// `n` residual radii: `|value - midpoint affine| <= radius` on the box.
    radius: Vec<f64>,
    /// `n` bounds on `|midpoint affine|` over the box.
    magnitude: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl MidpointAffine {
    fn from_linear(linear: &LinearBounds, mid: &[f64], half: &[f64]) -> Option<Self> {
        let lower_a = linear.lower_a();
        let upper_a = linear.upper_a();
        let lower_b = linear.lower_b();
        let upper_b = linear.upper_b();
        let rows = lower_a.nrows();
        let cols = lower_a.ncols();
        if cols != mid.len() || rows == 0 || upper_a.nrows() != rows || upper_a.ncols() != cols {
            return None;
        }
        if lower_b.len() != rows || upper_b.len() != rows {
            return None;
        }
        let mut coeff = vec![0.0f64; rows * cols];
        let mut offset = vec![0.0f64; rows];
        let mut radius = vec![0.0f64; rows];
        let mut magnitude = vec![0.0f64; rows];
        for i in 0..rows {
            let bl = f64::from(lower_b[i]);
            let bu = f64::from(upper_b[i]);
            if !bl.is_finite() || !bu.is_finite() {
                return None;
            }
            // Midpoint affine, and the half-gap affine whose maximum over the
            // box is the residual radius.
            let mut mid_value = f64::midpoint(bl, bu);
            let mut mid_dev = 0.0f64;
            let mut gap_value = 0.5 * (bu - bl);
            let mut gap_dev = 0.0f64;
            for j in 0..cols {
                let al = f64::from(lower_a[[i, j]]);
                let au = f64::from(upper_a[[i, j]]);
                if !al.is_finite() || !au.is_finite() {
                    return None;
                }
                let m = f64::midpoint(al, au);
                let g = 0.5 * (au - al);
                coeff[i * cols + j] = m;
                mid_value += m * mid[j];
                mid_dev += m.abs() * half[j];
                gap_value += g * mid[j];
                gap_dev += g.abs() * half[j];
            }
            offset[i] = f64::midpoint(bl, bu);
            magnitude[i] = mid_value.abs() + mid_dev;
            radius[i] = (gap_value + gap_dev).max(0.0);
            if !magnitude[i].is_finite() || !radius[i].is_finite() {
                return None;
            }
        }
        Some(Self {
            coeff,
            offset,
            radius,
            magnitude,
            rows,
            cols,
        })
    }
}

/// Cyclic Jacobi eigen-sweep of a symmetric matrix.
///
/// Returns `(lambda, u)` with `u` stored row-major as columns
/// `u[k * n + j] = U[j][k]`. No accuracy or orthogonality guarantee is claimed
/// or needed: the caller re-derives the exact residual `H - U diag(lambda) U'`
/// and charges it to the bound.
fn jacobi_eigh(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = matrix.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..JACOBI_SWEEPS {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off <= 0.0 || !off.is_finite() {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                if !t.is_finite() {
                    continue;
                }
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut lambda = vec![0.0f64; n];
    for i in 0..n {
        lambda[i] = a[i * n + i];
    }
    // Repack V's columns as contiguous rows for the caller.
    let mut cols = vec![0.0f64; n * n];
    for k in 0..n {
        for j in 0..n {
            cols[k * n + j] = v[j * n + k];
        }
    }
    (lambda, cols)
}

/// Rigorous upper bound on `max_{|d| <= half} d' H d` for symmetric `H`.
fn quadratic_deviation_max(h: &[f64], n: usize, half: &[f64]) -> f64 {
    let (lambda, cols) = jacobi_eigh(h, n);
    let mut total = 0.0f64;
    for k in 0..n {
        if lambda[k] <= 0.0 {
            continue;
        }
        let mut s = 0.0f64;
        for j in 0..n {
            s += cols[k * n + j].abs() * half[j];
        }
        total += lambda[k] * s * s;
    }
    // Exact residual of the (approximate) decomposition, charged in full.
    for j in 0..n {
        for l in 0..n {
            let mut recon = 0.0f64;
            for k in 0..n {
                recon += lambda[k] * cols[k * n + j] * cols[k * n + l];
            }
            let e = h[j * n + l] - recon;
            total += e.abs() * half[j] * half[l];
        }
    }
    total
}

/// Sound enclosure of `sum_i A_i(x) B_i(x)` over the box, or `None` on refusal.
fn joint_sum_bound(
    a: &MidpointAffine,
    b: &MidpointAffine,
    mid: &[f64],
    half: &[f64],
) -> Option<(f64, f64)> {
    let n = a.rows;
    let d = a.cols;
    if b.rows != n || b.cols != d || d != mid.len() || d != half.len() {
        return None;
    }
    // Symmetric quadratic form of sum_i a_i(x) b_i(x).
    let mut h = vec![0.0f64; d * d];
    let mut g = vec![0.0f64; d];
    let mut c0 = 0.0f64;
    for i in 0..n {
        c0 += a.offset[i] * b.offset[i];
        for j in 0..d {
            let aij = a.coeff[i * d + j];
            let bij = b.coeff[i * d + j];
            g[j] += a.offset[i] * bij + b.offset[i] * aij;
            for l in 0..d {
                h[j * d + l] += f64::midpoint(aij * b.coeff[i * d + l], bij * a.coeff[i * d + l]);
            }
        }
    }
    // Centre value and gradient at the box centre.
    let mut base = c0;
    for j in 0..d {
        base += g[j] * mid[j];
        for l in 0..d {
            base += h[j * d + l] * mid[j] * mid[l];
        }
    }
    let mut linear = 0.0f64;
    for j in 0..d {
        let mut grad = g[j];
        for l in 0..d {
            grad += 2.0 * h[j * d + l] * mid[l];
        }
        linear += grad.abs() * half[j];
    }
    let quad_up = quadratic_deviation_max(&h, d, half);
    let negated: Vec<f64> = h.iter().map(|value| -value).collect();
    let quad_down = quadratic_deviation_max(&negated, d, half);
    // Residual (relaxation-gap) cross terms.
    let mut residual = 0.0f64;
    for i in 0..n {
        residual +=
            a.magnitude[i] * b.radius[i] + b.magnitude[i] * a.radius[i] + a.radius[i] * b.radius[i];
    }
    let scale = base.abs() + linear + quad_up.abs() + quad_down.abs() + residual + 1.0;
    let ops = (n * d * d + d * d * d + d * d) as f64;
    let margin = scale * (16.0 * ops * f64::EPSILON + 1e-12);
    let upper = base + linear + quad_up + residual + margin;
    let lower = base - linear - quad_down - residual - margin;
    if !upper.is_finite() || !lower.is_finite() || lower > upper {
        return None;
    }
    Some((lower, upper))
}

/// Find every `ReduceSum(MulBinary(A, B))` site that fully reduces to a scalar.
fn detect_sites(
    graph: &GraphNetwork,
    bounds: &HashMap<String, BoundedTensor>,
) -> Vec<BilinearSite> {
    let mut sites = Vec::new();
    for (name, node) in &graph.nodes {
        if !matches!(&node.layer, Layer::ReduceSum(_)) {
            continue;
        }
        let inputs = node.inputs();
        if inputs.len() != 1 {
            continue;
        }
        let Some(reduce_box) = bounds.get(name.as_str()) else {
            continue;
        };
        if reduce_box.len() != 1 {
            continue;
        }
        let Some(mul_node) = graph.nodes.get(inputs[0].as_str()) else {
            continue;
        };
        if !matches!(&mul_node.layer, Layer::MulBinary(_)) {
            continue;
        }
        let mul_inputs = mul_node.inputs();
        if mul_inputs.len() != 2 {
            continue;
        }
        let (Some(left_box), Some(right_box)) = (
            bounds.get(mul_inputs[0].as_str()),
            bounds.get(mul_inputs[1].as_str()),
        ) else {
            continue;
        };
        let terms = left_box.len();
        // Elementwise only: a broadcasting Mul does not have one product per
        // operand element and the quadratic assembly below would be wrong.
        if !(2..=MAX_TERMS).contains(&terms) || right_box.len() != terms {
            continue;
        }
        if bounds
            .get(inputs[0].as_str())
            .is_none_or(|product| product.len() != terms)
        {
            continue;
        }
        sites.push(BilinearSite {
            reduce: name.clone(),
            left: mul_inputs[0].clone(),
            right: mul_inputs[1].clone(),
            terms,
        });
    }
    sites.sort_by(|a, b| a.reduce.cmp(&b.reduce));
    sites
}

/// Joint-bilinear spec bounds for one box, or `None` when the pass declines.
///
/// The returned tensor is a valid enclosure of the same spec rows the caller's
/// CROWN pass bounds; the caller intersects. Every refusal path returns
/// `Ok(None)` so a decline is byte-identical to the disarmed arm.
#[allow(clippy::too_many_arguments)]
pub(crate) fn joint_bilinear_spec_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    caller_node_bounds: Option<&HashMap<String, BoundedTensor>>,
) -> Result<Option<BoundedTensor>> {
    let flat_input = input.flatten();
    let input_dim = flat_input.len();
    if input_dim == 0 || input_dim > MAX_INPUT_DIM || graph.nodes.len() > MAX_NODES {
        return Ok(None);
    }
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Ok(None);
    }
    // Intermediate bounds. The caller has ALREADY paid for a map on this exact
    // box (the IBP-enhanced input-split path merges inherited alpha/clip bounds
    // with a fresh per-domain IBP pass); re-collecting here was measured at 19x
    // the per-domain cost on lsnc_relu row 2 — 2,410 domains against 45,822 in
    // the same 22.5 s. Reuse it when it covers the graph, and only collect when
    // there is nothing to reuse.
    let collected;
    let base = match caller_node_bounds {
        Some(map) if map.len() >= graph.nodes.len() => map,
        _ => {
            let Ok(map) =
                graph.collect_node_bounds_with_engine_and_deadline(input, engine, deadline)
            else {
                return Ok(None);
            };
            collected = map;
            &collected
        }
    };
    let sites = detect_sites(graph, base);
    tracing::debug!(
        "joint-bilinear: {} ReduceSum(A*B) site(s) detected in a {}-node graph \
         (node-bound map {}, reused {})",
        sites.len(),
        graph.nodes.len(),
        base.len(),
        caller_node_bounds.is_some_and(|map| map.len() >= graph.nodes.len())
    );
    if sites.is_empty() {
        return Ok(None);
    }
    let mut mid = vec![0.0f64; input_dim];
    let mut half = vec![0.0f64; input_dim];
    for (index, (low, high)) in flat_input
        .lower()
        .iter()
        .zip(flat_input.upper().iter())
        .enumerate()
    {
        let low = f64::from(*low);
        let high = f64::from(*high);
        if !low.is_finite() || !high.is_finite() || low > high {
            return Ok(None);
        }
        mid[index] = f64::midpoint(low, high);
        half[index] = 0.5 * (high - low);
    }

    let mut narrowed: HashSet<String> = HashSet::new();
    let mut bounds = base.clone();
    for site in &sites {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            break;
        }
        // The deadline is deliberately `None`. `propagate_crown_input_linear_to_node`
        // refuses outright when a finite authority is supplied — its whole-array
        // assembly phases are not yet cooperative — so a deadline here does not
        // shorten the work, it removes the certificate entirely. Cost is bounded
        // instead by the MAX_NODES / MAX_INPUT_DIM / MAX_TERMS gates above, and
        // the caller's deadline is polled on either side of the call.
        let left = match graph
            .propagate_crown_input_linear_to_node(input, &site.left, base, base, engine, None)
        {
            Ok(linear) => linear,
            Err(err) => {
                tracing::debug!(
                    "joint-bilinear: '{}' declined, no input-linear certificate for '{}': {err}",
                    site.reduce,
                    site.left
                );
                continue;
            }
        };
        let right = match graph.propagate_crown_input_linear_to_node(
            input,
            &site.right,
            base,
            base,
            engine,
            None,
        ) {
            Ok(linear) => linear,
            Err(err) => {
                tracing::debug!(
                    "joint-bilinear: '{}' declined, no input-linear certificate for '{}': {err}",
                    site.reduce,
                    site.right
                );
                continue;
            }
        };
        if left.num_outputs() != site.terms
            || right.num_outputs() != site.terms
            || left.num_inputs() != input_dim
            || right.num_inputs() != input_dim
        {
            tracing::debug!(
                "joint-bilinear: '{}' declined on certificate shape: left {}x{}, right {}x{}, \
                 expected {}x{}",
                site.reduce,
                left.num_outputs(),
                left.num_inputs(),
                right.num_outputs(),
                right.num_inputs(),
                site.terms,
                input_dim
            );
            continue;
        }
        let (Some(left_affine), Some(right_affine)) = (
            MidpointAffine::from_linear(&left, &mid, &half),
            MidpointAffine::from_linear(&right, &mid, &half),
        ) else {
            tracing::debug!(
                "joint-bilinear: '{}' declined, non-finite certificate coefficients",
                site.reduce
            );
            continue;
        };
        let Some((low, high)) = joint_sum_bound(&left_affine, &right_affine, &mid, &half) else {
            tracing::debug!(
                "joint-bilinear: '{}' declined, the quadratic bound is not finite",
                site.reduce
            );
            continue;
        };
        let Some(existing) = bounds.get(&site.reduce) else {
            continue;
        };
        // f32 publication is outward: the narrowing cast rounds to nearest, so
        // step one ulp out on each side before the box is published.
        let candidate = BoundedTensor::new(
            existing.lower().mapv(|_| next_down_f32(low as f32)),
            existing.upper().mapv(|_| next_up_f32(high as f32)),
        );
        let candidate = match candidate {
            Ok(candidate) => candidate,
            Err(err) => {
                tracing::debug!(
                    "joint-bilinear: '{}' declined, [{low}, {high}] is not a publishable box: {err}",
                    site.reduce
                );
                continue;
            }
        };
        let Some((tightened, disjoint)) = existing.intersection_per_element(&candidate) else {
            tracing::debug!(
                "joint-bilinear: '{}' declined, the intersection helper refused",
                site.reduce
            );
            continue;
        };
        if disjoint != 0 {
            tracing::debug!(
                "joint-bilinear: '{}' declined, {disjoint} disjoint element(s) against the \
                 existing box",
                site.reduce
            );
            continue;
        }
        tracing::debug!(
            "joint-bilinear: '{}' ({} terms) IBP [{}, {}] joint [{}, {}]",
            site.reduce,
            site.terms,
            existing
                .lower()
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min),
            existing
                .upper()
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max),
            low,
            high
        );
        if tightened.lower() != existing.lower() || tightened.upper() != existing.upper() {
            narrowed.insert(site.reduce.clone());
        }
        bounds.insert(site.reduce.clone(), tightened);
    }
    if narrowed.is_empty() {
        return Ok(None);
    }

    // Downstream resweep, through the same per-node IBP dispatch the
    // foundational forward pass uses. Shrink-only and fail-open.
    let exec_order: Vec<String> = graph.exec_order()?.to_vec();
    for node_name in exec_order {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Ok(None);
        }
        let Some(node) = graph.nodes.get(node_name.as_str()) else {
            continue;
        };
        if !node.inputs().iter().any(|name| narrowed.contains(name)) {
            continue;
        }
        let Ok(step) =
            graph.node_ibp_step(&node_name, node, input, &bounds, false, engine, deadline)
        else {
            continue;
        };
        let Some(existing) = bounds.get(&node_name) else {
            continue;
        };
        let Some((tightened, disjoint)) = existing.intersection_per_element(&step) else {
            continue;
        };
        if disjoint != 0 {
            continue;
        }
        if tightened.lower() != existing.lower() || tightened.upper() != existing.upper() {
            narrowed.insert(node_name.clone());
        }
        bounds.insert(node_name, tightened);
    }

    let output_node = if graph.output_node.is_empty() {
        match graph.exec_order()?.last() {
            Some(name) => name.clone(),
            None => return Ok(None),
        }
    } else {
        graph.output_node.clone()
    };
    if !narrowed.contains(&output_node) {
        tracing::debug!("joint-bilinear: tightening did not reach the output node '{output_node}'");
        return Ok(None);
    }
    let spec = graph.propagate_crown_with_specs_fallback_ibp(
        input,
        spec_matrix,
        &bounds,
        output_node.as_str(),
    )?;
    Ok(Some(spec))
}

fn next_down_f32(value: f32) -> f32 {
    if value.is_nan() || value == f32::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f32::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f32::from_bits(bits - 1)
    } else {
        f32::from_bits(bits + 1)
    }
}

fn next_up_f32(value: f32) -> f32 {
    if value.is_nan() || value == f32::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f32::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f32::from_bits(bits + 1)
    } else {
        f32::from_bits(bits - 1)
    }
}

#[cfg(test)]
#[path = "joint_bilinear_tests.rs"]
mod tests;
