// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Forward-map alpha optimizer (#w4-root-alpha-opt): choose per-neuron ReLU
//! lower slopes that tighten the forward-linear C-margin ROOT bound.
//!
//! # Why this exists
//!
//! W4-7 built the alpha-FED forward map (any per-neuron `α ∈ [0, 1]` composes
//! soundly with intercept 0 on crossing neurons) but proved the alpha WARMUP's
//! slopes are optimized for the GPU-backward relaxation and are ~8-10x LOOSER
//! for the forward map. The forward map needs alphas optimized against its OWN
//! objective: the C-margin lower bound of the unverified (straggler) spec rows.
//!
//! # The surrogate (why this is cheap)
//!
//! A full certified map rebuild is ~20s at cifar100 release scale — the conv
//! coefficient composition (im2col + f64 GEMM over every network-input column)
//! dominates and depends on every upstream ReLU diagonal, so nothing useful is
//! cacheable across alpha changes. Instead of rebuilding per candidate, this
//! module optimizes on a POINT-EVALUATION surrogate:
//!
//! * For the margin row `r`, the fixed-slope composed map gives the
//!   concretization vertex `x*_r` (per input coordinate: the box end selected
//!   by the composed lower-coefficient sign). At `x*_r` the composed lower
//!   bound EQUALS the composed affine lower function evaluated at `x*_r`.
//! * Evaluating the composed lower/upper functions at a point never needs the
//!   O(input_dim) coefficient matrices: carry two VECTORS per node (the lower
//!   and upper affine field values `L_v(x), U_v(x)`) through the same
//!   recurrences the certified pass uses (center-radius conv/dense, diagonal
//!   ReLU, Add). One pass costs a handful of direct convolutions — milliseconds
//!   instead of seconds.
//! * With relaxation slopes and vertices held fixed, the margin value is
//!   MULTILINEAR in the alphas (linear in each coordinate separately), so the
//!   per-coordinate optimum is a vertex of `[0, 1]` and the exact gradient is
//!   one adjoint (backward sensitivity) pass: `∂g/∂α_i = λ_i · L_pred,i(x*)`
//!   — the margin-row sensitivity at the ReLU output times the pre-activation
//!   lower field. Coordinate moves toward the preferred vertex with a halving
//!   step (interior points reachable) plus surrogate re-evaluation give a
//!   guarded ascent that never returns worse-than-adaptive alphas.
//!
//! # Soundness
//!
//! The optimizer is a HEURISTIC that only CHOOSES alphas. Every claimed bound
//! comes from the certified alpha-fed rebuild
//! (`collect_forward_linear_state_cached_with_alphas`) which is sound for ANY
//! `α ∈ [0, 1]`, and the caller intersects element-wise with the fixed-slope
//! candidates. Nothing computed here reaches the verdict directly.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use ndarray::{Array1, Array2};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use rayon::prelude::*;

use crate::bounds::LinearBounds;
use crate::layers::activations::relu::relu_linear_relaxation;
use crate::layers::Layer;

use super::image::{resolve_conv_geometry, ConvGeometry};
use super::{GraphNetwork, NETWORK_INPUT};

/// Cap on the number of straggler rows carried in the surrogate objective
/// (each row adds one forward-field row to every batched pass).
const MAX_ROWS: usize = 12;
/// Maximum accepted coordinate sweeps.
const MAX_SWEEPS: usize = 6;
/// Smallest step toward the preferred vertex before a sweep gives up.
const MIN_STEP: f64 = 0.12;

/// Summary of one optimizer run (kept `Copy` so the caller can memoize it).
#[derive(Debug, Clone, Copy)]
pub(crate) struct AlphaOptStats {
    /// Surrogate min-row objective at the adaptive starting alphas.
    pub(crate) baseline_min: f64,
    /// Surrogate min-row objective at the returned alphas (>= baseline_min).
    pub(crate) predicted_min: f64,
    /// Accepted sweeps.
    pub(crate) sweeps: usize,
    /// Crossing coordinates moved off their adaptive value.
    pub(crate) moved: usize,
    /// Crossing coordinates strictly inside (0, 1) in the result.
    pub(crate) interior: usize,
    /// Straggler rows in the surrogate objective.
    pub(crate) rows: usize,
}

enum SurrOp {
    Conv {
        /// Kernel, (oc, ic, kh, kw) C-order flat, f64.
        w: Vec<f64>,
        /// |Kernel|, same layout.
        wabs: Vec<f64>,
        /// Per-output-channel bias (empty = no bias).
        bias: Vec<f64>,
        /// Boxed: keeps the variant near the others' size (clippy `large_enum_variant`).
        geo: Box<ConvGeometry>,
    },
    Dense {
        /// Weight, (m, k) row-major flat, f64.
        w: Vec<f64>,
        wabs: Vec<f64>,
        /// Per-row bias (empty = no bias).
        bias: Vec<f64>,
        m: usize,
        k: usize,
    },
    Relu {
        relu_idx: usize,
    },
    Add,
    Pass,
}

struct SurrNode {
    /// Predecessors as indices into the exec-ordered node list
    /// (`None` = the network input).
    inputs: Vec<Option<usize>>,
    dim: usize,
    op: SurrOp,
}

/// Fixed relaxation snapshot of one ReLU node (taken from the cached
/// fixed-slope pass's running pre-activation bounds — the same source the
/// certified pass consumed).
struct SurrRelu {
    name: String,
    /// Adaptive lower slope (the α ∈ {0,1} the fixed pass used on crossing
    /// neurons; exact 0/1 slope on stable neurons).
    dl_adaptive: Vec<f64>,
    /// Lower intercept (0 for crossing/stable; kept general).
    cl: Vec<f64>,
    /// Chord upper slope / intercept (never touched by alpha).
    du: Vec<f64>,
    cu: Vec<f64>,
    crossing: Vec<bool>,
}

struct Surrogate {
    nodes: Vec<SurrNode>,
    relus: Vec<SurrRelu>,
    output_idx: usize,
    input_dim: usize,
}

/// Build the fixed-relaxation surrogate net from the cached fixed-slope pass
/// state. Returns `Ok(None)` (fail open — the fixed candidates stand) when the
/// graph leaves the certified image op surface or any relaxation datum is
/// non-finite.
fn build_surrogate(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
) -> Result<Option<Surrogate>> {
    let exec = graph.topological_sort()?;
    let mut idx_of: HashMap<&str, usize> = HashMap::with_capacity(exec.len());
    let mut nodes: Vec<SurrNode> = Vec::with_capacity(exec.len());
    let mut relus: Vec<SurrRelu> = Vec::new();

    for (i, name) in exec.iter().enumerate() {
        let node = graph.nodes.get(name).ok_or_else(|| {
            NyError::InvalidSpec(format!("alpha-opt surrogate: unknown node '{name}'"))
        })?;
        let dim = match node_bounds.get(name) {
            Some(b) => b.len(),
            None => return Ok(None),
        };
        let mut inputs = Vec::with_capacity(node.inputs.len());
        for pred in &node.inputs {
            if pred == NETWORK_INPUT {
                inputs.push(None);
            } else {
                match idx_of.get(pred.as_str()) {
                    Some(&p) => inputs.push(Some(p)),
                    None => return Ok(None),
                }
            }
        }
        let pred_bounds = |slot: usize| -> Option<&BoundedTensor> {
            match inputs.get(slot) {
                Some(None) => Some(input),
                Some(Some(p)) => node_bounds.get(&exec[*p]),
                None => None,
            }
        };
        let pred_dim = |slot: usize| -> usize {
            match inputs.get(slot) {
                Some(None) => input.len(),
                Some(Some(p)) => nodes[*p].dim,
                None => 0,
            }
        };

        let op = match &node.layer {
            Layer::Conv2d(conv) => {
                if inputs.len() != 1 {
                    return Ok(None);
                }
                let pred_shape = match &inputs[0] {
                    None => input.shape().to_vec(),
                    Some(p) => match node_bounds.get(&exec[*p]) {
                        Some(b) => b.shape().to_vec(),
                        None => return Ok(None),
                    },
                };
                let geo = match resolve_conv_geometry(name, conv, &pred_shape, pred_dim(0), dim) {
                    Ok(geo) => geo,
                    Err(NyError::UnsupportedConfiguration(_) | NyError::ShapeMismatch { .. }) => {
                        return Ok(None)
                    }
                    Err(e) => return Err(e),
                };
                let kernel_spatial = geo.kh * geo.kw;
                let mut w = vec![0.0f64; geo.out_c * geo.in_c * kernel_spatial];
                for oc in 0..geo.out_c {
                    for ic in 0..geo.in_c {
                        for ki in 0..geo.kh {
                            for kj in 0..geo.kw {
                                w[((oc * geo.in_c + ic) * geo.kh + ki) * geo.kw + kj] =
                                    f64::from(conv.kernel[[oc, ic, ki, kj]]);
                            }
                        }
                    }
                }
                let wabs: Vec<f64> = w.iter().map(|v| v.abs()).collect();
                let bias = conv
                    .bias
                    .as_ref()
                    .map(|b| b.iter().map(|&v| f64::from(v)).collect())
                    .unwrap_or_default();
                SurrOp::Conv {
                    w,
                    wabs,
                    bias,
                    geo: Box::new(geo),
                }
            }
            Layer::Linear(linear) => {
                if inputs.len() != 1 {
                    return Ok(None);
                }
                let m = linear.weight.nrows();
                let k = linear.weight.ncols();
                if m != dim || k != pred_dim(0) {
                    return Ok(None);
                }
                let mut w = vec![0.0f64; m * k];
                for r in 0..m {
                    for c in 0..k {
                        w[r * k + c] = f64::from(linear.weight[[r, c]]);
                    }
                }
                let wabs: Vec<f64> = w.iter().map(|v| v.abs()).collect();
                let bias = linear
                    .bias
                    .as_ref()
                    .map(|b| b.iter().map(|&v| f64::from(v)).collect())
                    .unwrap_or_default();
                SurrOp::Dense {
                    w,
                    wabs,
                    bias,
                    m,
                    k,
                }
            }
            Layer::ReLU(_) => {
                if inputs.len() != 1 || pred_dim(0) != dim {
                    return Ok(None);
                }
                let pre = match pred_bounds(0) {
                    Some(b) => b.flatten(),
                    None => return Ok(None),
                };
                let mut dl = vec![0.0f64; dim];
                let mut cl = vec![0.0f64; dim];
                let mut du = vec![0.0f64; dim];
                let mut cu = vec![0.0f64; dim];
                let mut crossing = vec![false; dim];
                for j in 0..dim {
                    let (l, u) = (pre.lower()[j], pre.upper()[j]);
                    let relax = relu_linear_relaxation(l, u);
                    let vals = [
                        relax.lower_slope,
                        relax.lower_intercept,
                        relax.upper_slope,
                        relax.upper_intercept,
                    ];
                    if vals.iter().any(|v| !v.is_finite()) {
                        return Ok(None);
                    }
                    dl[j] = f64::from(relax.lower_slope);
                    cl[j] = f64::from(relax.lower_intercept);
                    du[j] = f64::from(relax.upper_slope);
                    cu[j] = f64::from(relax.upper_intercept);
                    crossing[j] = l < 0.0 && u > 0.0 && l.is_finite() && u.is_finite();
                }
                relus.push(SurrRelu {
                    name: name.clone(),
                    dl_adaptive: dl,
                    cl,
                    du,
                    cu,
                    crossing,
                });
                SurrOp::Relu {
                    relu_idx: relus.len() - 1,
                }
            }
            Layer::Add(_) => {
                if inputs.len() != 2 || pred_dim(0) != dim || pred_dim(1) != dim {
                    return Ok(None);
                }
                SurrOp::Add
            }
            Layer::Flatten(_) | Layer::Reshape(_) | Layer::Squeeze(_) | Layer::Unsqueeze(_) => {
                if inputs.len() != 1 || pred_dim(0) != dim {
                    return Ok(None);
                }
                SurrOp::Pass
            }
            _ => return Ok(None),
        };

        idx_of.insert(exec[i].as_str(), i);
        nodes.push(SurrNode { inputs, dim, op });
    }

    let output_name = if graph.output_node.is_empty() {
        match exec.last() {
            Some(last) => last.clone(),
            None => return Ok(None),
        }
    } else {
        graph.output_node.clone()
    };
    let Some(&output_idx) = idx_of.get(output_name.as_str()) else {
        return Ok(None);
    };

    Ok(Some(Surrogate {
        nodes,
        relus,
        output_idx,
        input_dim: input.len(),
    }))
}

/// Direct batched conv: `out[r] = conv(x[r])` for each row of `xs`
/// (`(rows, conv_in)` → `(rows, conv_out)`), plain f64, no bias.
fn conv_apply_batch(w: &[f64], geo: &ConvGeometry, xs: &Array2<f64>) -> Array2<f64> {
    let rows = xs.nrows();
    let conv_out = geo.conv_out_size();
    let in_spatial = geo.in_h * geo.in_w;
    let out_spatial = geo.spatial();
    let (sh, sw) = geo.stride;
    let (ph, pw) = geo.padding;
    let (dh, dw) = geo.dilation;
    let xs_flat = xs.as_slice().expect("row-major xs");
    let conv_in = geo.conv_in_size();
    let mut out = Array2::<f64>::zeros((rows, conv_out));
    out.as_slice_mut()
        .expect("row-major out")
        .par_chunks_mut(conv_out)
        .enumerate()
        .for_each(|(r, orow)| {
            let x = &xs_flat[r * conv_in..(r + 1) * conv_in];
            for oc in 0..geo.out_c {
                let w_oc = &w[oc * geo.contraction..(oc + 1) * geo.contraction];
                for oh in 0..geo.out_h {
                    for ow in 0..geo.out_w {
                        let mut acc = 0.0f64;
                        for ic in 0..geo.in_c {
                            let x_ic = &x[ic * in_spatial..(ic + 1) * in_spatial];
                            let w_ic = &w_oc[ic * geo.kh * geo.kw..(ic + 1) * geo.kh * geo.kw];
                            for ki in 0..geo.kh {
                                let ih = (oh * sh + ki * dh) as isize - ph as isize;
                                if ih < 0 || ih >= geo.in_h as isize {
                                    continue;
                                }
                                let x_row = &x_ic[ih as usize * geo.in_w..];
                                let w_row = &w_ic[ki * geo.kw..(ki + 1) * geo.kw];
                                for kj in 0..geo.kw {
                                    let iw = (ow * sw + kj * dw) as isize - pw as isize;
                                    if iw < 0 || iw >= geo.in_w as isize {
                                        continue;
                                    }
                                    acc += w_row[kj] * x_row[iw as usize];
                                }
                            }
                        }
                        orow[oc * out_spatial + oh * geo.out_w + ow] = acc;
                    }
                }
            }
        });
    out
}

/// Transposed batched conv (adjoint of [`conv_apply_batch`]):
/// `(rows, conv_out)` sensitivities → `(rows, conv_in)`.
fn conv_apply_batch_t(w: &[f64], geo: &ConvGeometry, lams: &Array2<f64>) -> Array2<f64> {
    let rows = lams.nrows();
    let conv_in = geo.conv_in_size();
    let conv_out = geo.conv_out_size();
    let in_spatial = geo.in_h * geo.in_w;
    let out_spatial = geo.spatial();
    let (sh, sw) = geo.stride;
    let (ph, pw) = geo.padding;
    let (dh, dw) = geo.dilation;
    let lam_flat = lams.as_slice().expect("row-major lams");
    let kernel_spatial = geo.kh * geo.kw;
    let mut out = Array2::<f64>::zeros((rows, conv_in));
    out.as_slice_mut()
        .expect("row-major out")
        .par_chunks_mut(conv_in)
        .enumerate()
        .for_each(|(r, orow)| {
            let lam = &lam_flat[r * conv_out..(r + 1) * conv_out];
            for ic in 0..geo.in_c {
                for ih in 0..geo.in_h {
                    for iw in 0..geo.in_w {
                        let mut acc = 0.0f64;
                        for ki in 0..geo.kh {
                            let oh_num = ih as isize + ph as isize - (ki * dh) as isize;
                            if oh_num < 0 {
                                continue;
                            }
                            let oh_num = oh_num as usize;
                            if !oh_num.is_multiple_of(sh) {
                                continue;
                            }
                            let oh = oh_num / sh;
                            if oh >= geo.out_h {
                                continue;
                            }
                            for kj in 0..geo.kw {
                                let ow_num = iw as isize + pw as isize - (kj * dw) as isize;
                                if ow_num < 0 {
                                    continue;
                                }
                                let ow_num = ow_num as usize;
                                if !ow_num.is_multiple_of(sw) {
                                    continue;
                                }
                                let ow = ow_num / sw;
                                if ow >= geo.out_w {
                                    continue;
                                }
                                let k_idx = (ic * geo.kh + ki) * geo.kw + kj;
                                let mut w_idx = k_idx;
                                let mut l_idx = oh * geo.out_w + ow;
                                for _oc in 0..geo.out_c {
                                    acc += w[w_idx] * lam[l_idx];
                                    w_idx += geo.in_c * kernel_spatial;
                                    l_idx += out_spatial;
                                }
                            }
                        }
                        orow[ic * in_spatial + ih * geo.in_w + iw] = acc;
                    }
                }
            }
        });
    out
}

fn dense_apply_batch(w: &[f64], m: usize, k: usize, xs: &Array2<f64>) -> Array2<f64> {
    let rows = xs.nrows();
    let xs_flat = xs.as_slice().expect("row-major xs");
    let mut out = Array2::<f64>::zeros((rows, m));
    out.as_slice_mut()
        .expect("row-major out")
        .par_chunks_mut(m)
        .enumerate()
        .for_each(|(r, orow)| {
            let x = &xs_flat[r * k..(r + 1) * k];
            for (i, o) in orow.iter_mut().enumerate() {
                let wr = &w[i * k..(i + 1) * k];
                *o = wr.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            }
        });
    out
}

fn dense_apply_batch_t(w: &[f64], m: usize, k: usize, lams: &Array2<f64>) -> Array2<f64> {
    let rows = lams.nrows();
    let lam_flat = lams.as_slice().expect("row-major lams");
    let mut out = Array2::<f64>::zeros((rows, k));
    out.as_slice_mut()
        .expect("row-major out")
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(r, orow)| {
            let lam = &lam_flat[r * m..(r + 1) * m];
            for (i, &l) in lam.iter().enumerate() {
                if l == 0.0 {
                    continue;
                }
                let wr = &w[i * k..(i + 1) * k];
                for (o, &wv) in orow.iter_mut().zip(wr.iter()) {
                    *o += l * wv;
                }
            }
        });
    out
}

/// Center-radius affine field step shared by Conv/Dense:
/// `L' = A·c − |A|·r + b`, `U' = A·c + |A|·r + b` with `c = (L+U)/2`,
/// `r = (U−L)/2` — the same algebra the certified compositions use.
fn affine_fields(
    apply: impl Fn(&Array2<f64>, bool) -> Array2<f64>,
    l: &Array2<f64>,
    u: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let c = (l + u) * 0.5;
    let r = (u - l) * 0.5;
    let gc = apply(&c, false);
    let radius_zero = r.iter().all(|v| *v == 0.0);
    if radius_zero {
        (gc.clone(), gc)
    } else {
        let gr = apply(&r, true);
        (&gc - &gr, &gc + &gr)
    }
}

fn add_channel_bias(fields: &mut Array2<f64>, geo: &ConvGeometry, bias: &[f64]) {
    if bias.is_empty() {
        return;
    }
    let spatial = geo.spatial();
    for mut row in fields.rows_mut() {
        for oc in 0..geo.out_c {
            let b = bias[oc];
            for s in 0..spatial {
                row[oc * spatial + s] += b;
            }
        }
    }
}

fn add_dense_bias(fields: &mut Array2<f64>, bias: &[f64]) {
    if bias.is_empty() {
        return;
    }
    for mut row in fields.rows_mut() {
        for (v, b) in row.iter_mut().zip(bias.iter()) {
            *v += b;
        }
    }
}

struct ForwardFields {
    out_l: Array2<f64>,
    out_u: Array2<f64>,
    /// Per-ReLU pre-activation LOWER fields (only when requested).
    relu_pre_l: Option<Vec<Array2<f64>>>,
}

/// Evaluate the lower/upper affine fields at the batch of points `xs`
/// (one row per straggler-row vertex), under the given alphas.
fn forward_fields(
    s: &Surrogate,
    alphas: &[Vec<f64>],
    xs: &Array2<f64>,
    want_pre: bool,
    deadline: Option<Instant>,
) -> Result<ForwardFields> {
    let rows = xs.nrows();
    let mut fields: Vec<Option<(Array2<f64>, Array2<f64>)>> = vec![None; s.nodes.len()];
    let mut relu_pre_l = want_pre.then(|| Vec::with_capacity(s.relus.len()));

    let fetch = |fields: &Vec<Option<(Array2<f64>, Array2<f64>)>>,
                 slot: &Option<usize>|
     -> Result<(Array2<f64>, Array2<f64>)> {
        match slot {
            None => Ok((xs.clone(), xs.clone())),
            Some(p) => fields[*p]
                .clone()
                .ok_or_else(|| NyError::InternalError("alpha-opt: missing upstream field".into())),
        }
    };

    for (i, node) in s.nodes.iter().enumerate() {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(NyError::DeadlineExceeded(
                "alpha-opt: deadline exceeded in forward field pass".into(),
            ));
        }
        let out = match &node.op {
            SurrOp::Conv { w, wabs, bias, geo } => {
                let (l, u) = fetch(&fields, &node.inputs[0])?;
                let (mut nl, mut nu) = affine_fields(
                    |m, abs| conv_apply_batch(if abs { wabs } else { w }, geo, m),
                    &l,
                    &u,
                );
                add_channel_bias(&mut nl, geo, bias);
                add_channel_bias(&mut nu, geo, bias);
                (nl, nu)
            }
            SurrOp::Dense {
                w,
                wabs,
                bias,
                m,
                k,
            } => {
                let (l, u) = fetch(&fields, &node.inputs[0])?;
                let (mut nl, mut nu) = affine_fields(
                    |x, abs| dense_apply_batch(if abs { wabs } else { w }, *m, *k, x),
                    &l,
                    &u,
                );
                add_dense_bias(&mut nl, bias);
                add_dense_bias(&mut nu, bias);
                (nl, nu)
            }
            SurrOp::Relu { relu_idx } => {
                let (l, u) = fetch(&fields, &node.inputs[0])?;
                let relu = &s.relus[*relu_idx];
                let alpha = &alphas[*relu_idx];
                if let Some(pre) = relu_pre_l.as_mut() {
                    pre.push(l.clone());
                }
                let mut nl = l;
                let mut nu = u;
                for r in 0..rows {
                    for j in 0..node.dim {
                        let dl = if relu.crossing[j] {
                            alpha[j]
                        } else {
                            relu.dl_adaptive[j]
                        };
                        nl[[r, j]] = dl * nl[[r, j]] + relu.cl[j];
                        nu[[r, j]] = relu.du[j] * nu[[r, j]] + relu.cu[j];
                    }
                }
                (nl, nu)
            }
            SurrOp::Add => {
                let (la, ua) = fetch(&fields, &node.inputs[0])?;
                let (lb, ub) = fetch(&fields, &node.inputs[1])?;
                (&la + &lb, &ua + &ub)
            }
            SurrOp::Pass => fetch(&fields, &node.inputs[0])?,
        };
        fields[i] = Some(out);
    }

    let (out_l, out_u) = fields[s.output_idx]
        .take()
        .ok_or_else(|| NyError::InternalError("alpha-opt: output field missing".into()))?;
    Ok(ForwardFields {
        out_l,
        out_u,
        relu_pre_l,
    })
}

/// Per-row surrogate margin values: `g_r = Σ_k C⁺_rk·L_out[r,k] + C⁻_rk·U_out[r,k]`.
fn margin_values(
    out_l: &Array2<f64>,
    out_u: &Array2<f64>,
    cpos: &Array2<f64>,
    cneg: &Array2<f64>,
) -> Vec<f64> {
    (0..out_l.nrows())
        .map(|r| {
            let mut acc = 0.0f64;
            for k in 0..out_l.ncols() {
                acc += cpos[[r, k]] * out_l[[r, k]] + cneg[[r, k]] * out_u[[r, k]];
            }
            acc
        })
        .collect()
}

/// One adjoint pass: sensitivities of the weighted objective w.r.t. each
/// crossing alpha. Seeds carry the row weights (`λ_L = w_r·C_r⁺`,
/// `λ_U = w_r·C_r⁻`). Returns per-ReLU gradient vectors.
fn alpha_gradients(
    s: &Surrogate,
    alphas: &[Vec<f64>],
    relu_pre_l: &[Array2<f64>],
    seed_l: Array2<f64>,
    seed_u: Array2<f64>,
    deadline: Option<Instant>,
) -> Result<Vec<Vec<f64>>> {
    let rows = seed_l.nrows();
    let mut lams: Vec<Option<(Array2<f64>, Array2<f64>)>> = vec![None; s.nodes.len()];
    lams[s.output_idx] = Some((seed_l, seed_u));
    let mut grads: Vec<Vec<f64>> = s
        .relus
        .iter()
        .map(|r| vec![0.0f64; r.dl_adaptive.len()])
        .collect();

    let accumulate = |lams: &mut Vec<Option<(Array2<f64>, Array2<f64>)>>,
                      slot: &Option<usize>,
                      contrib: (Array2<f64>, Array2<f64>)| {
        let Some(p) = slot else { return };
        match &mut lams[*p] {
            Some((l, u)) => {
                *l += &contrib.0;
                *u += &contrib.1;
            }
            entry @ None => *entry = Some(contrib),
        }
    };

    for i in (0..s.nodes.len()).rev() {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(NyError::DeadlineExceeded(
                "alpha-opt: deadline exceeded in adjoint pass".into(),
            ));
        }
        let Some((lam_l, lam_u)) = lams[i].take() else {
            continue;
        };
        let node = &s.nodes[i];
        match &node.op {
            SurrOp::Conv { w, wabs, geo, .. } => {
                // Adjoint of the center-radius step: with s = λL+λU, d = λU−λL,
                // λL_up = (Aᵀs − |A|ᵀd)/2, λU_up = (Aᵀs + |A|ᵀd)/2.
                let ssum = &lam_l + &lam_u;
                let diff = &lam_u - &lam_l;
                let tc = conv_apply_batch_t(w, geo, &ssum);
                let tr = conv_apply_batch_t(wabs, geo, &diff);
                accumulate(
                    &mut lams,
                    &node.inputs[0],
                    ((&tc - &tr) * 0.5, (&tc + &tr) * 0.5),
                );
            }
            SurrOp::Dense { w, wabs, m, k, .. } => {
                let ssum = &lam_l + &lam_u;
                let diff = &lam_u - &lam_l;
                let tc = dense_apply_batch_t(w, *m, *k, &ssum);
                let tr = dense_apply_batch_t(wabs, *m, *k, &diff);
                accumulate(
                    &mut lams,
                    &node.inputs[0],
                    ((&tc - &tr) * 0.5, (&tc + &tr) * 0.5),
                );
            }
            SurrOp::Relu { relu_idx } => {
                let relu = &s.relus[*relu_idx];
                let alpha = &alphas[*relu_idx];
                let pre_l = &relu_pre_l[*relu_idx];
                let grad = &mut grads[*relu_idx];
                let mut nl = lam_l;
                let mut nu = lam_u;
                for r in 0..rows {
                    for j in 0..node.dim {
                        if relu.crossing[j] {
                            grad[j] += nl[[r, j]] * pre_l[[r, j]];
                        }
                        let dl = if relu.crossing[j] {
                            alpha[j]
                        } else {
                            relu.dl_adaptive[j]
                        };
                        nl[[r, j]] *= dl;
                        nu[[r, j]] *= relu.du[j];
                    }
                }
                accumulate(&mut lams, &node.inputs[0], (nl, nu));
            }
            SurrOp::Add => {
                accumulate(&mut lams, &node.inputs[0], (lam_l.clone(), lam_u.clone()));
                accumulate(&mut lams, &node.inputs[1], (lam_l, lam_u));
            }
            SurrOp::Pass => {
                accumulate(&mut lams, &node.inputs[0], (lam_l, lam_u));
            }
        }
    }
    Ok(grads)
}

/// Softmin weights over the row values (concentrating on the worst rows).
fn softmin_weights(vals: &[f64]) -> Vec<f64> {
    let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
    let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let temp = ((max - min) / 3.0).max(1e-3);
    let mut w: Vec<f64> = vals.iter().map(|v| (-(v - min) / temp).exp()).collect();
    let sum: f64 = w.iter().sum();
    if sum > 0.0 && sum.is_finite() {
        for v in &mut w {
            *v /= sum;
        }
    } else {
        let uniform = 1.0 / w.len().max(1) as f64;
        w.fill(uniform);
    }
    w
}

/// Optimize per-neuron forward-map lower slopes against the C-margin
/// objective of the unverified (straggler) spec rows. Pure heuristic — see
/// module docs. Returns `Ok(None)` when the surrogate refuses, no row is
/// unverified, or no improvement over the adaptive start is found.
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_margin_alphas(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    current_lower: Option<&BoundedTensor>,
    node_bounds: &HashMap<String, BoundedTensor>,
    output_lb: &LinearBounds,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Result<Option<(BTreeMap<String, Array1<f32>>, AlphaOptStats)>> {
    let Some(surrogate) = build_surrogate(graph, input, node_bounds)? else {
        return Ok(None);
    };
    if !surrogate
        .relus
        .iter()
        .any(|r| r.crossing.iter().any(|&c| c))
    {
        return Ok(None);
    }

    // Compose C with the OUTPUT map once (tiny certified GEMM): the composed
    // lower rows give the per-row concretization vertices and the baseline.
    let input_flat = input.flatten();
    let input_mag: Vec<f64> = input_flat
        .lower()
        .iter()
        .zip(input_flat.upper().iter())
        .map(|(&l, &u)| f64::from(l).abs().max(f64::from(u).abs()))
        .collect();
    if spec_matrix.ncols() != output_lb.num_outputs() {
        return Ok(None);
    }
    let composed = super::image::compose_dense_affine_forward(
        "alpha-opt margin",
        spec_matrix,
        None,
        output_lb,
        &input_mag,
        engine,
        None,
    )?;
    let concretized = composed.concretize_sound(input);

    // Row selection: current best lower bounds (the intersected root
    // candidate when available, else the fixed-map composition), threshold-0
    // margin convention. Worst rows first, capped.
    let n_rows = spec_matrix.nrows();
    let row_lower: Vec<f64> = match current_lower {
        Some(bounds) if bounds.len() == n_rows => {
            let flat = bounds.flatten();
            (0..n_rows)
                .map(|r| f64::from(flat.lower()[r]).max(f64::from(concretized.lower()[r])))
                .collect()
        }
        _ => (0..n_rows)
            .map(|r| f64::from(concretized.lower()[r]))
            .collect(),
    };
    let mut straggler_rows: Vec<usize> = (0..n_rows).filter(|&r| row_lower[r] < 0.0).collect();
    straggler_rows.sort_by(|&a, &b| row_lower[a].total_cmp(&row_lower[b]));
    straggler_rows.truncate(MAX_ROWS);
    if straggler_rows.is_empty() {
        return Ok(None);
    }
    let k_rows = straggler_rows.len();

    // Per-row concretization vertex from the composed lower-coefficient signs,
    // and the C⁺/C⁻ seeds.
    let in_lo: Vec<f64> = input_flat.lower().iter().map(|&v| f64::from(v)).collect();
    let in_hi: Vec<f64> = input_flat.upper().iter().map(|&v| f64::from(v)).collect();
    let lower_a = composed.lower_a();
    let out_dim = spec_matrix.ncols();
    let mut xs = Array2::<f64>::zeros((k_rows, surrogate.input_dim));
    let mut cpos = Array2::<f64>::zeros((k_rows, out_dim));
    let mut cneg = Array2::<f64>::zeros((k_rows, out_dim));
    for (ri, &row) in straggler_rows.iter().enumerate() {
        for j in 0..surrogate.input_dim {
            xs[[ri, j]] = if lower_a[[row, j]] >= 0.0 {
                in_lo[j]
            } else {
                in_hi[j]
            };
        }
        for k in 0..out_dim {
            let c = f64::from(spec_matrix[[row, k]]);
            cpos[[ri, k]] = c.max(0.0);
            cneg[[ri, k]] = c.min(0.0);
        }
    }

    // Adaptive start (exactly the slopes the fixed pass used).
    let alpha0: Vec<Vec<f64>> = surrogate
        .relus
        .iter()
        .map(|r| r.dl_adaptive.clone())
        .collect();

    let eval = |alphas: &[Vec<f64>], want_pre: bool| -> Result<(Vec<f64>, ForwardFields)> {
        let fields = forward_fields(&surrogate, alphas, &xs, want_pre, deadline)?;
        let vals = margin_values(&fields.out_l, &fields.out_u, &cpos, &cneg);
        Ok((vals, fields))
    };

    let (mut best_vals, _) = eval(&alpha0, false)?;
    let baseline_min = best_vals.iter().copied().fold(f64::INFINITY, f64::min);
    if !baseline_min.is_finite() {
        return Ok(None);
    }
    let mut best_alpha = alpha0.clone();
    let mut best_min = baseline_min;
    let mut eta = 1.0f64;
    let mut sweeps = 0usize;

    'sweeps: for _ in 0..MAX_SWEEPS {
        // Gradient of the softmin-weighted objective at the current best.
        let weights = softmin_weights(&best_vals);
        let mut best_g: f64 = best_vals
            .iter()
            .zip(weights.iter())
            .map(|(v, w)| v * w)
            .sum();
        let mut seed_l = cpos.clone();
        let mut seed_u = cneg.clone();
        for (ri, &w) in weights.iter().enumerate() {
            for k in 0..out_dim {
                seed_l[[ri, k]] *= w;
                seed_u[[ri, k]] *= w;
            }
        }
        let (grads, pre_fields) = {
            let (_, fields) = match eval(&best_alpha, true) {
                Ok(res) => res,
                Err(NyError::DeadlineExceeded(_)) => break 'sweeps,
                Err(e) => return Err(e),
            };
            let pre = fields
                .relu_pre_l
                .ok_or_else(|| NyError::InternalError("alpha-opt: pre fields missing".into()))?;
            let grads =
                match alpha_gradients(&surrogate, &best_alpha, &pre, seed_l, seed_u, deadline) {
                    Ok(g) => g,
                    Err(NyError::DeadlineExceeded(_)) => break 'sweeps,
                    Err(e) => return Err(e),
                };
            (grads, pre)
        };
        drop(pre_fields);
        if grads.iter().zip(surrogate.relus.iter()).all(|(g, r)| {
            g.iter()
                .zip(r.crossing.iter())
                .all(|(v, &c)| !c || *v == 0.0)
        }) {
            break;
        }

        // Guarded coordinate move toward the per-coordinate preferred vertex.
        let mut accepted = false;
        let mut step = eta;
        // Geometric line search: `step` halves each rejection, so the loop
        // terminates (0.12 = MIN_STEP > 0; a NaN step exits immediately).
        #[allow(clippy::while_float)]
        while step >= MIN_STEP {
            let mut cand = best_alpha.clone();
            let mut moved_any = false;
            for (ri, relu) in surrogate.relus.iter().enumerate() {
                for j in 0..relu.crossing.len() {
                    if !relu.crossing[j] || grads[ri][j] == 0.0 {
                        continue;
                    }
                    let next = (cand[ri][j] + step * grads[ri][j].signum()).clamp(0.0, 1.0);
                    if next != cand[ri][j] {
                        cand[ri][j] = next;
                        moved_any = true;
                    }
                }
            }
            if !moved_any {
                break;
            }
            let (vals, _) = match eval(&cand, false) {
                Ok(res) => res,
                Err(NyError::DeadlineExceeded(_)) => break 'sweeps,
                Err(e) => return Err(e),
            };
            let g: f64 = vals.iter().zip(weights.iter()).map(|(v, w)| v * w).sum();
            let vmin = vals.iter().copied().fold(f64::INFINITY, f64::min);
            if g.is_finite() && g > best_g + 1e-9 * (1.0 + best_g.abs()) {
                best_alpha = cand;
                best_vals = vals;
                best_g = g;
                best_min = vmin;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }
        sweeps += 1;
        eta = (step * 1.5).min(1.0);
        if deadline.is_some_and(|d| Instant::now() >= d) {
            break;
        }
    }

    // Only worth the ~20s certified rebuild when the surrogate predicts a
    // real improvement of the worst straggler row...
    let improve_tol = 1e-6 * (1.0 + baseline_min.abs());
    // Negated form is deliberate: a NaN best_min/baseline_min makes the `>` false,
    // so we skip the rebuild (fail-closed); `<=` would proceed on NaN.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if sweeps == 0 || !(best_min > baseline_min + improve_tol) {
        return Ok(None);
    }
    // ...AND the optimized map can actually move the INTERSECTION: the
    // caller element-wise intersects with the current root candidate, so the
    // rebuild only helps if some selected row's alpha-map bound exceeds the
    // candidate's. The surrogate value is an OPTIMISTIC (fixed-vertex,
    // fixed-slope) estimate of the true rebuilt bound, so this is a
    // necessary condition — skipping on its failure never discards a win.
    // MEASURED (release, cifar100 @95s, 4 instances): the post-W4-4 GPU
    // per-entry backward dominates the root candidate at −4.3..−9.4 while
    // the optimized forward margin converges at −17..−27; without this gate
    // the rebuild burned ~25s of BaB budget per instance for an intersection
    // no-op.
    let beats_candidate = straggler_rows
        .iter()
        .zip(best_vals.iter())
        .any(|(&row, &val)| val > row_lower[row] + 1e-6 * (1.0 + row_lower[row].abs()));
    if !beats_candidate {
        tracing::info!(
            rows = k_rows,
            sweeps,
            surrogate_baseline_min = baseline_min,
            surrogate_predicted_min = best_min,
            candidate_min = straggler_rows
                .iter()
                .map(|&r| row_lower[r])
                .fold(f64::INFINITY, f64::min),
            "forward-map alpha optimizer: optimized surrogate cannot beat the current root candidate on any straggler row — skipping the certified rebuild (#w4-root-alpha-opt)"
        );
        return Ok(None);
    }

    let mut moved = 0usize;
    let mut interior = 0usize;
    let mut alphas_out: BTreeMap<String, Array1<f32>> = BTreeMap::new();
    for (ri, relu) in surrogate.relus.iter().enumerate() {
        if !relu.crossing.iter().any(|&c| c) {
            continue;
        }
        let mut arr = Array1::<f32>::from_elem(relu.crossing.len(), f32::NAN);
        for j in 0..relu.crossing.len() {
            if relu.crossing[j] {
                let v = best_alpha[ri][j];
                arr[j] = v as f32;
                if (v - alpha0[ri][j]).abs() > 1e-6 {
                    moved += 1;
                }
                if v > 0.0 && v < 1.0 {
                    interior += 1;
                }
            }
        }
        alphas_out.insert(relu.name.clone(), arr);
    }
    if moved == 0 {
        return Ok(None);
    }

    Ok(Some((
        alphas_out,
        AlphaOptStats {
            baseline_min,
            predicted_min: best_min,
            sweeps,
            moved,
            interior,
            rows: k_rows,
        },
    )))
}

#[cfg(test)]
mod grad_tests {
    use super::*;

    /// The surrogate objective is exactly LINEAR in each alpha coordinate
    /// (multilinear overall), so the adjoint gradient must match a central
    /// finite difference to floating-point precision. This pins the adjoint
    /// recurrences (center-radius conv/dense transpose, ReLU diagonal, Add
    /// fan-in) against the forward field evaluation.
    #[test]
    fn test_alpha_gradient_matches_finite_difference() {
        let (graph, input) = super::super::tests_image::build_residual_dag_for_grad_test();
        let map = graph
            .collect_forward_linear_bounds_dag_with_engine(&input, None)
            .expect("fixed pass");
        let surrogate = build_surrogate(&graph, &input, &map)
            .expect("build must not error")
            .expect("residual DAG is on the surrogate surface");
        assert!(
            surrogate
                .relus
                .iter()
                .any(|r| r.crossing.iter().any(|&c| c)),
            "fixture must have crossing neurons"
        );

        // Two evaluation points inside the box (deterministic).
        let input_flat = input.flatten();
        let n = input_flat.len();
        let mut xs = Array2::<f64>::zeros((2, n));
        for j in 0..n {
            let l = f64::from(input_flat.lower()[j]);
            let u = f64::from(input_flat.upper()[j]);
            xs[[0, j]] = l + (u - l) * 0.25;
            xs[[1, j]] = l + (u - l) * 0.75;
        }
        // Margin seeds for 2 rows over the 3 outputs, with row weights baked in.
        let out_dim = surrogate.nodes[surrogate.output_idx].dim;
        assert_eq!(out_dim, 3);
        let spec = [[1.0f64, -1.0, 0.0], [-0.5, 0.0, 1.5]];
        let weights = [0.7f64, 0.3];
        let mut cpos = Array2::<f64>::zeros((2, 3));
        let mut cneg = Array2::<f64>::zeros((2, 3));
        for r in 0..2 {
            for k in 0..3 {
                cpos[[r, k]] = spec[r][k].max(0.0);
                cneg[[r, k]] = spec[r][k].min(0.0);
            }
        }

        // Mixed starting alphas (interior, so both vertices are reachable).
        let alphas: Vec<Vec<f64>> = surrogate
            .relus
            .iter()
            .map(|r| {
                r.crossing
                    .iter()
                    .enumerate()
                    .map(|(j, _)| 0.3 + 0.4 * ((j % 3) as f64) / 2.0)
                    .collect()
            })
            .collect();

        let objective = |a: &[Vec<f64>]| -> f64 {
            let fields = forward_fields(&surrogate, a, &xs, false, None).expect("fields");
            let vals = margin_values(&fields.out_l, &fields.out_u, &cpos, &cneg);
            vals.iter().zip(weights.iter()).map(|(v, w)| v * w).sum()
        };

        let fields = forward_fields(&surrogate, &alphas, &xs, true, None).expect("fields");
        let pre = fields.relu_pre_l.expect("pre fields");
        let mut seed_l = cpos.clone();
        let mut seed_u = cneg.clone();
        for (r, &w) in weights.iter().enumerate() {
            for k in 0..3 {
                seed_l[[r, k]] *= w;
                seed_u[[r, k]] *= w;
            }
        }
        let grads =
            alpha_gradients(&surrogate, &alphas, &pre, seed_l, seed_u, None).expect("adjoint");

        let h = 0.05f64;
        let mut checked = 0usize;
        for (ri, relu) in surrogate.relus.iter().enumerate() {
            for j in 0..relu.crossing.len() {
                if !relu.crossing[j] {
                    continue;
                }
                let mut plus = alphas.clone();
                plus[ri][j] += h;
                let mut minus = alphas.clone();
                minus[ri][j] -= h;
                let fd = (objective(&plus) - objective(&minus)) / (2.0 * h);
                let ad = grads[ri][j];
                let tol = 1e-7 * (1.0 + fd.abs().max(ad.abs()));
                assert!(
                    (fd - ad).abs() <= tol,
                    "grad mismatch at relu {ri} ('{}') coord {j}: fd={fd}, adjoint={ad}",
                    relu.name
                );
                checked += 1;
            }
        }
        assert!(checked >= 3, "need >=3 crossing coordinates, got {checked}");

        // Degenerate-box sanity: evaluating the fields at a point equals the
        // concretized composed bound at that point for a zero-radius box is
        // implicitly covered by the MC containment suite; here assert the
        // objective is finite.
        assert!(objective(&alphas).is_finite());
    }
}
