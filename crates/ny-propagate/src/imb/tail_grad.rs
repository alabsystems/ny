// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bespoke tail CROWN backward + reverse-mode alpha gradient — the exact-gradient
//! port of numpy `disc_affine_alpha` + `grad_alpha` (item A), FLATTENED to dense
//! per-segment matrices for speed (the speed pass).
//!
//! The tail (seam → Y) is a chain of linear layers separated by the 2 tail ReLUs.
//! We compose each maximal linear RUN into ONE dense affine map `(W, b)` ONCE via
//! ny's own `propagate_crown_backward` starting from an identity `LinearBounds`
//! (so the dense extraction reuses the exact primitive that matched ny to 1e-6 —
//! consistency preserved by construction). The per-iteration alpha ascent is then
//! pure dense matmul (`a·W` backward / `W·abar` forward) instead of a per-layer
//! CROWN dispatch — ~100× faster per iter.
//!
//! Used ONLY to SELECT the alpha; the FINAL sound `(p,q)` is still built by ny's
//! `tail_lower_functional(best_alpha)` (unchanged). Any `alpha ∈ [0,1]` yields a
//! sound `(p,q)` via the caller's `fold_coeff_err_over_box_eager`, so a bug here can
//! only mis-select alpha, never produce an unsound floor.

use std::collections::HashMap;
use std::time::Instant;

use ndarray::{Array1, Array2};
use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;

use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::layers::{BoundPropagation, Layer};
use crate::{GraphNetwork, GraphNode, NETWORK_INPUT};

use super::{env_f64, env_usize};

// ===========================================================================
// Dense tail representation
// ===========================================================================

/// One tail op in Y→seam order: a dense affine map or a ReLU relaxation.
enum TailOp {
    /// A composed linear run: forward `out = W·in + b` (W is [out × in]).
    Dense { w: Array2<f32>, b: Array1<f32> },
    /// A tail ReLU: `relu_node` is the alpha key; `[l,u]` its pre-activation anchor.
    Relu {
        relu_node: String,
        l: Vec<f32>,
        u: Vec<f32>,
    },
}

/// Per-ReLU relaxation state recorded by the forward pass for the gradient.
struct ReluState {
    a_out: Vec<f32>,
    unst: Vec<bool>,
    iu: Vec<f32>,
    slope: Vec<f32>,
    neg: Vec<bool>,
}

/// The conv's input spatial shape (its CROWN backward needs it; ny sets it
/// per-backward): from the input node's bounds, or the seam shape at the seam.
fn conv_input_shape(
    node: &GraphNode,
    tail_ibp: &HashMap<String, BoundedTensor>,
    seam_shape: &[usize],
) -> Option<Vec<usize>> {
    let inp = node.inputs().first()?;
    if inp == NETWORK_INPUT {
        Some(seam_shape.to_vec())
    } else {
        tail_ibp.get(inp).map(|b| b.lower().shape().to_vec())
    }
}

/// Clone a conv layer with its `input_shape` set from the `[.., H, W]` input shape
/// (no-op for non-conv layers).
fn shaped_conv_layer(layer: &Layer, input_shape: &[usize]) -> Layer {
    let mut l = layer.clone();
    if input_shape.len() >= 2 {
        let h = input_shape[input_shape.len() - 2];
        let w = input_shape[input_shape.len() - 1];
        match &mut l {
            Layer::Conv2d(c) => c.set_input_shape(h, w),
            Layer::ConvTranspose2d(c) => c.set_input_shape(h, w),
            _ => {}
        }
    }
    l
}

/// Per-neuron CROWN ReLU relaxation from anchors `[l,u]` and the free lower slope
/// `alpha` (unstable only): `(su, sl, iu, unstable)`.
fn relu_relax_params(l: f32, u: f32, alpha_i: f32) -> (f32, f32, f32, bool) {
    if u <= 0.0 {
        (0.0, 0.0, 0.0, false) // dead
    } else if l >= 0.0 {
        (1.0, 1.0, 0.0, false) // linear (y = x)
    } else {
        let su = u / (u - l);
        (su, alpha_i, -l * su, true) // upper chord; lower slope = alpha
    }
}

/// Compose a maximal linear RUN (in seam→Y exec order) into one dense `(W, b)` with
/// `out = W·in + b`, by backward-propagating an identity `LinearBounds` from the run
/// OUTPUT through ny's per-layer CROWN adjoint. `W = lower_a` (`[out × in]`),
/// `b = lower_b`. This is the SAME relaxation-free affine transform ny's tail CROWN
/// uses (BN is affine → no slack), so the dense map is exact.
fn build_segment_dense(
    tail: &GraphNetwork,
    run: &[String],
    tail_ibp: &HashMap<String, BoundedTensor>,
    seam_shape: &[usize],
) -> Result<(Array2<f32>, Array1<f32>), String> {
    let last = run
        .last()
        .ok_or_else(|| "empty linear segment (tail begins/ends with a ReLU?)".to_string())?;
    let out_dim = tail_ibp
        .get(last)
        .ok_or_else(|| format!("segment out node '{last}' not in tail_ibp"))?
        .flatten()
        .len();
    let mut lb = LinearBounds::identity(out_dim);
    for name in run.iter().rev() {
        let node = tail
            .nodes
            .get(name)
            .ok_or_else(|| format!("segment node '{name}' missing"))?;
        let is_conv = matches!(node.layer, Layer::Conv2d(_) | Layer::ConvTranspose2d(_));
        let pre_act = tail_ibp.get(name);
        let ltype = node.layer.layer_type();
        let mk_err = |e: ny_core::NyError| {
            format!("segment propagate_crown_backward '{name}' ({ltype}): {e}")
        };
        lb = if is_conv {
            let ishape = conv_input_shape(node, tail_ibp, seam_shape)
                .ok_or_else(|| format!("conv '{name}': input shape unresolved"))?;
            shaped_conv_layer(&node.layer, &ishape)
                .propagate_crown_backward(&lb, pre_act)
                .map_err(mk_err)?
        } else {
            node.layer
                .propagate_crown_backward(&lb, pre_act)
                .map_err(mk_err)?
        };
    }
    Ok((lb.lower_a().to_owned(), lb.lower_b().to_owned()))
}

/// Build the dense tail op sequence (Y→seam order) ONCE: split the tail exec into
/// linear runs at ReLU boundaries, dense-compose each run, and interleave the ReLU
/// anchors (from the CROWN-tight `tail_ibp`).
fn build_tail_ops(
    tail: &GraphNetwork,
    tail_ibp: &HashMap<String, BoundedTensor>,
    seam_shape: &[usize],
) -> Result<Vec<TailOp>, String> {
    let exec = tail
        .exec_order()
        .map_err(|e| format!("tail.exec_order: {e}"))?;
    // Linear runs (seam→Y) + ReLU anchors between them.
    let mut runs: Vec<Vec<String>> = vec![Vec::new()];
    let mut relus: Vec<(String, Vec<f32>, Vec<f32>)> = Vec::new();
    for name in exec {
        let node = tail
            .nodes
            .get(name)
            .ok_or_else(|| format!("tail node '{name}' missing"))?;
        if matches!(node.layer, Layer::ReLU(_)) {
            let src = node
                .inputs
                .first()
                .ok_or_else(|| format!("relu '{name}': no input"))?;
            let anchor = tail_ibp
                .get(src)
                .ok_or_else(|| format!("relu '{name}': anchor '{src}' not in tail_ibp"))?
                .flatten();
            let l = anchor
                .lower()
                .as_slice()
                .ok_or_else(|| format!("relu '{name}': anchor lower non-contiguous"))?
                .to_vec();
            let u = anchor
                .upper()
                .as_slice()
                .ok_or_else(|| format!("relu '{name}': anchor upper non-contiguous"))?
                .to_vec();
            relus.push((name.clone(), l, u));
            runs.push(Vec::new());
        } else {
            runs.last_mut().unwrap().push(name.clone());
        }
    }
    // Dense-compose each run.
    let mut denses: Vec<(Array2<f32>, Array1<f32>)> = Vec::with_capacity(runs.len());
    for run in &runs {
        denses.push(build_segment_dense(tail, run, tail_ibp, seam_shape)?);
    }
    // Assemble Y→seam: Dense(last run), then Relu(i), Dense(i) for i = n-1 .. 0.
    let n = relus.len();
    let mut ops: Vec<TailOp> = Vec::with_capacity(2 * n + 1);
    let (w, b) = denses[n].clone();
    ops.push(TailOp::Dense { w, b });
    for i in (0..n).rev() {
        let (relu_node, l, u) = relus[i].clone();
        ops.push(TailOp::Relu { relu_node, l, u });
        let (w, b) = denses[i].clone();
        ops.push(TailOp::Dense { w, b });
    }
    Ok(ops)
}

// ===========================================================================
// Dense forward (disc_affine_alpha) + reverse gradient (grad_alpha)
// ===========================================================================

/// Dense tail CROWN backward → `(p, q, relu_states)` for a given `alpha`. `a` starts
/// as `obj_row` over the tail output; each `Dense` does `q += a·b; a = a·W`; each
/// `Relu` relaxes `a` with the free alpha lower slope (records its state).
fn disc_affine_alpha_dense(
    ops: &[TailOp],
    alpha: &GraphAlphaState,
    obj_row: &[f32],
) -> Result<(Vec<f32>, f32, HashMap<String, ReluState>), String> {
    let mut a: Array1<f32> = Array1::from(obj_row.to_vec());
    let mut q = 0.0f64;
    let mut states: HashMap<String, ReluState> = HashMap::new();

    for op in ops {
        match op {
            TailOp::Dense { w, b } => {
                if a.len() != w.nrows() {
                    return Err(format!(
                        "dense: coeff dim {} != W rows {} (out)",
                        a.len(),
                        w.nrows()
                    ));
                }
                q += a.dot(b) as f64; // a·bias into q
                a = a.dot(w); // [1×out]·[out×in] = [1×in]
            }
            TailOp::Relu { relu_node, l, u } => {
                let dim = a.len();
                if l.len() != dim || u.len() != dim {
                    return Err(format!(
                        "relu '{relu_node}': anchor dim {}/{} != coeff dim {dim}",
                        l.len(),
                        u.len()
                    ));
                }
                let alpha_arr = alpha.alpha(relu_node);
                let a_out: Vec<f32> = a.to_vec();
                let mut new_a = vec![0.0f32; dim];
                let mut unst = vec![false; dim];
                let mut iu = vec![0.0f32; dim];
                let mut slope = vec![0.0f32; dim];
                let mut neg = vec![false; dim];
                for i in 0..dim {
                    let alpha_i = alpha_arr
                        .and_then(|arr| arr.get(i).copied())
                        .unwrap_or(if u[i] >= -l[i] { 1.0 } else { 0.0 });
                    let (su_i, sl_i, iu_i, is_unst) = relu_relax_params(l[i], u[i], alpha_i);
                    unst[i] = is_unst;
                    iu[i] = iu_i;
                    let a_neg = a_out[i] < 0.0;
                    neg[i] = a_neg;
                    if a_neg {
                        q += a_out[i] as f64 * iu_i as f64;
                    }
                    let sl = if a_out[i] >= 0.0 { sl_i } else { su_i };
                    slope[i] = sl;
                    new_a[i] = a_out[i] * sl;
                }
                states.insert(
                    relu_node.clone(),
                    ReluState {
                        a_out,
                        unst,
                        iu,
                        slope,
                        neg,
                    },
                );
                a = Array1::from(new_a);
            }
        }
    }
    Ok((a.to_vec(), q as f32, states))
}

/// Dense reverse-mode `d(p·ystar + q)/d alpha`. `abar` starts as `ystar` over the
/// seam; each `Dense` (reverse) applies the forward map `abar = W·abar + b`; each
/// `Relu` accumulates `grad[node][i] += abar[i]·a_out[i]` for kept unstable neurons.
fn grad_alpha_dense(
    ops: &[TailOp],
    states: &HashMap<String, ReluState>,
    ystar: &[f32],
) -> Result<HashMap<String, Vec<f32>>, String> {
    let mut abar: Array1<f32> = Array1::from(ystar.to_vec());
    let mut grads: HashMap<String, Vec<f32>> = HashMap::new();
    for op in ops.iter().rev() {
        match op {
            TailOp::Dense { w, b } => {
                if abar.len() != w.ncols() {
                    return Err(format!(
                        "grad dense: abar dim {} != W cols {} (in)",
                        abar.len(),
                        w.ncols()
                    ));
                }
                abar = w.dot(&abar) + b; // [out×in]·[in] + [out]
            }
            TailOp::Relu { relu_node, .. } => {
                let st = states
                    .get(relu_node)
                    .ok_or_else(|| format!("grad: no relu state for '{relu_node}'"))?;
                let dim = abar.len();
                if st.a_out.len() != dim {
                    return Err(format!(
                        "grad relu '{relu_node}': abar dim {dim} != a_out {}",
                        st.a_out.len()
                    ));
                }
                let g = grads
                    .entry(relu_node.clone())
                    .or_insert_with(|| vec![0.0f32; dim]);
                for i in 0..dim {
                    if st.unst[i] && st.a_out[i] >= 0.0 {
                        g[i] += abar[i] * st.a_out[i];
                    }
                }
                for i in 0..dim {
                    abar[i] = abar[i] * st.slope[i] + if st.neg[i] { st.iu[i] } else { 0.0 };
                }
            }
        }
    }
    Ok(grads)
}

// ===========================================================================
// Ascent driver
// ===========================================================================

/// Tail ReLU nodes (in exec order) paired with their pre-activation source node.
fn tail_relu_srcs(tail: &GraphNetwork, exec: &[String]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for name in exec {
        if let Some(node) = tail.nodes.get(name) {
            if matches!(node.layer, Layer::ReLU(_)) {
                if let Some(src) = node.inputs.first() {
                    out.push((name.clone(), src.clone()));
                }
            }
        }
    }
    out
}

/// Set the unstable lower slopes of every tail ReLU to a constant policy
/// (`adaptive` = `u ≥ −l ? 1 : 0`, else `half` = 0.5).
fn init_alpha_policy(
    base: &GraphAlphaState,
    tail: &GraphNetwork,
    tail_ibp: &HashMap<String, BoundedTensor>,
    exec: &[String],
    adaptive: bool,
) -> GraphAlphaState {
    let mut s = base.clone();
    for (relu, src) in tail_relu_srcs(tail, exec) {
        let Some(anchor) = tail_ibp.get(&src) else {
            continue;
        };
        let af = anchor.flatten();
        let (Some(l), Some(u)) = (af.lower().as_slice(), af.upper().as_slice()) else {
            continue;
        };
        let (l, u) = (l.to_vec(), u.to_vec());
        if let Some((lo, _up)) = s.relu_alpha_pair_mut(&relu) {
            for i in 0..lo.len().min(l.len()) {
                if l[i] < 0.0 && u[i] > 0.0 {
                    lo[i] = if adaptive {
                        if u[i] >= -l[i] {
                            1.0
                        } else {
                            0.0
                        }
                    } else {
                        0.5
                    };
                }
            }
        }
    }
    s
}

/// `min_i (p·yseam_i + q)` + the argmin index (f64 accumulation). Parallelized over
/// the (several-thousand) samples — this runs once per gradient iteration, so the
/// scan is a real inner-loop cost. The reduction is a total-order min (score, then
/// SMALLEST index on a tie), which is associative + commutative, so the result is
/// DETERMINISTIC regardless of rayon's split order (reproducible alpha trajectory).
fn sample_min_arg(yseam: &[Vec<f32>], p: &[f32], q: f32) -> (f32, usize) {
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
    let eval = |i: usize, y: &Vec<f32>| -> (f32, usize) {
        let mut s = q as f64;
        for (yi, pi) in y.iter().zip(p.iter()) {
            s += *yi as f64 * *pi as f64;
        }
        (s as f32, i)
    };
    let pick = |a: (f32, usize), b: (f32, usize)| -> (f32, usize) {
        if b.0 < a.0 || (b.0 == a.0 && b.1 < a.1) {
            b
        } else {
            a
        }
    };
    // Region-parallel path: run the reduce SEQUENTIALLY so the N region workers don't
    // each fan this out (`crate::imb::region_seq_inner`). Deterministic either way (the
    // total-order min is associative+commutative), so the alpha trajectory is identical.
    if crate::imb::region_seq_inner() {
        yseam
            .iter()
            .enumerate()
            .fold((f32::INFINITY, usize::MAX), |acc, (i, y)| {
                pick(acc, eval(i, y))
            })
    } else {
        yseam
            .par_iter()
            .enumerate()
            .map(|(i, y)| eval(i, y))
            .reduce(|| (f32::INFINITY, usize::MAX), pick)
    }
}

/// Exact-gradient ascent (numpy `build_functional`), FLATTENED: dense tail ops built
/// once, then per-iter `disc_affine_alpha_dense` / `grad_alpha_dense` (pure matmul).
/// From 2 inits {adaptive, half}, `lr=4.0 ×0.998/iter`, maximize `min_i(p·yseam_i+q)`.
/// Returns the best alpha and its sample-min. Logs bespoke `(p,q)` for the init once
/// (caller compares to ny `tail_lower_functional` — the consistency gate).
#[allow(clippy::too_many_arguments)]
pub(super) fn optimize_tail_alpha_bespoke(
    tail: &GraphNetwork,
    seam_box: &BoundedTensor,
    tail_ibp: &HashMap<String, BoundedTensor>,
    yseam: &[Vec<f32>],
    obj_row: &[f32],
    init_alpha: &GraphAlphaState,
    _engine: Option<&dyn GemmEngine>,
    deadline: Instant,
) -> Option<(GraphAlphaState, f32)> {
    let exec = match tail.exec_order() {
        Ok(e) => e.to_vec(),
        Err(e) => {
            eprintln!("[imb] tail-grad SKIP: tail.exec_order failed: {e}");
            return None;
        }
    };
    let seam_shape = seam_box.lower().shape().to_vec();
    let iters = env_usize("NY_IMB_TAIL_OPT_ITERS", 1000);
    let lr0 = env_f64("NY_IMB_TAIL_OPT_LR", 4.0) as f32;
    let relu_srcs = tail_relu_srcs(tail, &exec);

    // One-time SETUP dump.
    let mut ibp_keys: Vec<&String> = tail_ibp.keys().collect();
    ibp_keys.sort();
    eprintln!(
        "[imb] tail-grad SETUP: seam_dim={} tail_exec={exec:?}",
        seam_shape.iter().product::<usize>()
    );
    eprintln!("[imb] tail-grad SETUP: tail_ibp_keys={ibp_keys:?}");
    for (relu, src) in &relu_srcs {
        let alpha_len = init_alpha.alpha(relu).map(|a| a.len());
        let anchor_len = tail_ibp.get(src).map(|b| b.flatten().len());
        eprintln!(
            "[imb] tail-grad SETUP: relu={relu} src={src} alpha_len={alpha_len:?} anchor_len={anchor_len:?}"
        );
    }

    // Build the dense tail segments ONCE (the amortized cost).
    let t0 = Instant::now();
    let ops = match build_tail_ops(tail, tail_ibp, &seam_shape) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[imb] tail-grad SKIP: build_tail_ops failed: {e}");
            return None;
        }
    };
    let n_dense = ops
        .iter()
        .filter(|o| matches!(o, TailOp::Dense { .. }))
        .count();
    eprintln!(
        "[imb] tail-grad: dense tail built in {:.2}s ({} ops, {n_dense} dense segments)",
        t0.elapsed().as_secs_f64(),
        ops.len()
    );

    // Consistency check (logged once): dense bespoke (p,q) for the init alpha.
    match disc_affine_alpha_dense(&ops, init_alpha, obj_row) {
        Ok((p, q, _)) => {
            let l1: f64 = p.iter().map(|c| c.abs() as f64).sum();
            eprintln!(
                "[imb] tail-grad CONSISTENCY bespoke(init): q={q:+.6} |p|1={l1:.4} (compare to ny tail_lower_functional)"
            );
        }
        Err(e) => {
            eprintln!("[imb] tail-grad SKIP: disc_affine_alpha_dense(init) failed: {e}");
            return None;
        }
    }

    let inits = [
        init_alpha_policy(init_alpha, tail, tail_ibp, &exec, true),
        init_alpha_policy(init_alpha, tail, tail_ibp, &exec, false),
    ];

    // One gradient-ascent trajectory from a single init. Pure/read-only over the
    // shared `ops`/`yseam`/`relu_srcs`, so the two inits run in PARALLEL (rayon::join)
    // — each dense per-iteration backprop is the dominant cost, and the two are
    // independent, so this ~halves the wall. Returns `(best_alpha, best_score,
    // start_score)`; `None` on a dense-op error (mirrors the prior early return).
    let run_init = |init: GraphAlphaState| -> Option<(GraphAlphaState, f32, f32)> {
        let mut alpha = init;
        let mut lr = lr0;
        let mut best_alpha = alpha.clone();
        let mut best_score = f32::NEG_INFINITY;
        let mut start_score = f32::NAN;
        for it in 0..iters {
            if Instant::now() >= deadline {
                break;
            }
            let (p, q, states) = disc_affine_alpha_dense(&ops, &alpha, obj_row)
                .map_err(|e| eprintln!("[imb] tail-grad SKIP: disc(iter {it}): {e}"))
                .ok()?;
            let (score, xi) = sample_min_arg(yseam, &p, q);
            if it == 0 {
                start_score = score;
            }
            if score > best_score {
                best_score = score;
                best_alpha = alpha.clone();
            }
            let grads = grad_alpha_dense(&ops, &states, &yseam[xi])
                .map_err(|e| eprintln!("[imb] tail-grad SKIP: grad(iter {it}): {e}"))
                .ok()?;
            for (relu, _src) in &relu_srcs {
                if let Some(g) = grads.get(relu) {
                    if let Some((lo, _up)) = alpha.relu_alpha_pair_mut(relu) {
                        for (a, &gi) in lo.iter_mut().zip(g.iter()) {
                            if gi.is_finite() {
                                *a = (*a + lr * gi).clamp(0.0, 1.0);
                            }
                        }
                    }
                }
            }
            lr *= 0.998;
        }
        Some((best_alpha, best_score, start_score))
    };

    let [init_a, init_b] = inits;
    // Region-parallel path: run the 2 inits SEQUENTIALLY (each region is already one of
    // the N concurrent workers; `rayon::join` here would fan out and contend). Otherwise
    // run them in parallel. Result is identical (both inits are always evaluated).
    let (ra, rb) = if crate::imb::region_seq_inner() {
        (run_init(init_a), run_init(init_b))
    } else {
        rayon::join(|| run_init(init_a), || run_init(init_b))
    };
    // Keep whichever init reached the higher sample-min (both are sound; the choice is
    // heuristic). Fall back to the other on a per-init dense error.
    let start_score = ra.as_ref().map(|r| r.2).unwrap_or(f32::NAN);
    let (best_alpha, best_score) = match (ra, rb) {
        (Some((aa, sa, _)), Some((ab, sb, _))) => {
            if sa >= sb {
                (aa, sa)
            } else {
                (ab, sb)
            }
        }
        (Some((aa, sa, _)), None) => (aa, sa),
        (None, Some((ab, sb, _))) => (ab, sb),
        (None, None) => return None,
    };

    eprintln!("[imb] tail-grad: iters={iters}x2(par) sample_min {start_score:.6}->{best_score:.6}");
    Some((best_alpha, best_score))
}
