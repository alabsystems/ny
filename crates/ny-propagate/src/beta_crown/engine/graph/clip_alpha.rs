// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph-local adapter for `clip_in_alpha_crown`.
//!
//! Computes **forward** linear bounds (each node as a function of the network
//! input) by accumulating through the graph in topological order, then feeds
//! these into `clip_interm_domain_full` to tighten intermediate bounds using
//! split-derived constraints.
//!
//! The key distinction from backward CROWN capture: backward CROWN produces
//! output-relative bounds (`output = A * node + b`), but `clip_interm_domain_full`
//! needs input-relative bounds (`node = A * input + b`). This module computes
//! the latter via forward accumulation.
//!
//! Reference: alpha-beta-CROWN `clip_domains.py` / `BoundedModule._get_interm_bounds`
//! which computes forward linear relaxations through the computational graph.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};

use crate::batched_domain::CachedLinearBounds;
use crate::beta_crown::branching::GraphSplitHistory;
use crate::clip_interm_domain::{
    build_split_constraints, merge_bounds, sort_out_constraints, tighten_with_constraints,
};
use crate::{GraphNetwork, Layer, LinearBounds, NETWORK_INPUT};

/// Batch linear bounds: (lower_A, lower_b, upper_A, upper_b).
type BatchLinearParts = (Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>);

/// Compute forward linear bounds for each node: `node(x) ∈ [lA*x + lb, uA*x + ub]`.
///
/// Traverses the graph in topological order, accumulating linear relaxations:
/// - **Linear(W, b)**: standard interval arithmetic composition
/// - **ReLU (constrained active)**: identity pass-through
/// - **ReLU (constrained inactive)**: zero
/// - **ReLU (unconstrained)**: triangle relaxation using pre-activation bounds
/// - **Other layers**: conservative (unbounded) fallback
// Generic over the map value (`BoundedTensor` or `Arc<BoundedTensor>` via
// `Borrow`): the constrained-propagation caches are `Arc`-shared (#cone-delta
// increment 2) while `clip_complete` builds a plain owned map; both read the
// same tensor values.
pub(in crate::beta_crown::engine::graph) fn compute_forward_linear_bounds<
    V: Borrow<BoundedTensor>,
>(
    graph: &GraphNetwork,
    split_history: &GraphSplitHistory,
    exec_order: &[String],
    bounds_cache: &HashMap<String, V>,
    constrained_input: &BoundedTensor,
) -> Result<CachedLinearBounds> {
    // `len()` == `flatten().len()` (flatten preserves element count) with no allocation.
    let input_dim = constrained_input.len();
    let mut forward_bounds: HashMap<String, LinearBounds> = HashMap::new();

    // Per-input-dimension magnitude bound `max(|x_l|, |x_u|)`, used to fold the
    // f32 coefficient-storage roundoff of binary compositions (Add) *outward*
    // into the bias so the stored affine relaxation stays sound over the box.
    let (x_l, x_u) = constrained_input.flatten_to_ix1("forward_linear_bounds_abs_x")?;
    let abs_x: Array1<f32> = x_l
        .iter()
        .zip(x_u.iter())
        .map(|(&l, &u)| l.abs().max(u.abs()))
        .collect();

    for node_name in exec_order {
        let node = graph.nodes.get(node_name).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "forward linear bounds: unknown node '{}'",
                node_name
            ))
        })?;

        let pred_name = match node.inputs.first() {
            Some(name) => name.as_str(),
            None => continue,
        };

        let pred_bounds = if pred_name == NETWORK_INPUT {
            LinearBounds::identity(input_dim)
        } else if let Some(pb) = forward_bounds.get(pred_name) {
            pb.clone()
        } else {
            // Predecessor not yet computed (e.g., skip connection source) — conservative
            let num_outputs = bounds_cache
                .get(node_name)
                .map(|b| b.borrow().len())
                .unwrap_or(input_dim);
            LinearBounds::conservative(num_outputs, input_dim)
        };

        let node_bounds = match &node.layer {
            Layer::Linear(linear) => {
                compose_linear_forward(&pred_bounds, &linear.weight, linear.bias.as_ref())
            }
            Layer::ReLU(_) => compose_relu_forward(
                &pred_bounds,
                node_name,
                split_history,
                bounds_cache.get(pred_name).map(|b| b.borrow()),
            ),
            Layer::Add(_) => {
                // Residual/skip Add `C = A + B`: compose BOTH predecessors'
                // forward relaxations. The single-predecessor `pred_bounds`
                // above only covers input 0; without composing input 1 the Add
                // (and therefore every downstream node) fell through to the
                // conservative ±∞ arm below, which is what left residual nets
                // (cersyve finetune, nn4sys) un-tightenable by clip_interm.
                let fetch_fwd = |name: &str| -> LinearBounds {
                    if name == NETWORK_INPUT {
                        LinearBounds::identity(input_dim)
                    } else if let Some(pb) = forward_bounds.get(name) {
                        pb.clone()
                    } else {
                        let n = bounds_cache
                            .get(name)
                            .map(|b| b.borrow().len())
                            .unwrap_or(input_dim);
                        LinearBounds::conservative(n, input_dim)
                    }
                };
                match node.inputs.get(1) {
                    Some(second) => compose_add_forward(&pred_bounds, &fetch_fwd(second), &abs_x),
                    // Unary/broadcast Add we can't model as a binary sum — stay
                    // conservative (sound) rather than guess.
                    None => Ok(LinearBounds::conservative(
                        pred_bounds.num_outputs(),
                        input_dim,
                    )),
                }
            }
            _ => {
                let num_outputs = bounds_cache
                    .get(node_name)
                    .map(|b| b.borrow().len())
                    .unwrap_or(pred_bounds.num_outputs());
                Ok(LinearBounds::conservative(num_outputs, input_dim))
            }
        }?;

        forward_bounds.insert(node_name.clone(), node_bounds);
    }

    Ok(CachedLinearBounds::from_linear_bounds_map(forward_bounds))
}

/// Compose forward linear bounds through a Linear layer: `y = W * x_prev + bias`.
///
/// Uses interval arithmetic to handle the sign-dependent composition:
/// `lower_new = W_pos * lower_prev + W_neg * upper_prev + bias`
/// `upper_new = W_pos * upper_prev + W_neg * lower_prev + bias`
fn compose_linear_forward(
    pred: &LinearBounds,
    weight: &Array2<f32>,
    bias: Option<&Array1<f32>>,
) -> Result<LinearBounds> {
    let w_pos = weight.mapv(|v| v.max(0.0));
    let w_neg = weight.mapv(|v| v.min(0.0));

    let lower_a = w_pos.dot(pred.lower_a()) + w_neg.dot(pred.upper_a());
    let upper_a = w_pos.dot(pred.upper_a()) + w_neg.dot(pred.lower_a());

    // The predecessor's bias may be ±∞ when an unsupported node (e.g. a residual
    // Add skip-connection — this forward pass only models Linear/ReLU) was
    // assigned conservative bounds upstream. A plain `w.dot(±∞)` then evaluates
    // `0·(±∞) = NaN` for every zero weight, poisoning the whole bias vector with
    // NaN and tripping the downstream firewall → conservative for the WHOLE
    // node. The mathematically correct interval value of a zero-weight term is
    // exactly 0 (a zero coefficient means that predecessor neuron cannot affect
    // this output, regardless of how wide its bound is), so a saturating dot
    // that defines `0·∞ = 0` yields finite-where-possible / ±∞-where-genuine
    // bias values without the spurious NaN. This is sound: each output term is
    // either a real finite/±∞ contribution or a genuine 0, never a NaN. (#3438)
    let mut lower_b = saturating_matvec(&w_pos, pred.lower_b(), &w_neg, pred.upper_b());
    let mut upper_b = saturating_matvec(&w_pos, pred.upper_b(), &w_neg, pred.lower_b());

    if let Some(b) = bias {
        lower_b += b;
        upper_b += b;
    }

    LinearBounds::new_or_conservative(lower_a, lower_b, upper_a, upper_b)
}

/// Sound bias matvec `out[i] = Σ_k w_pos[i,k]·p[k] + w_neg[i,k]·q[k]` with the
/// convention `0·(±∞) = 0` (IEEE `0·∞ = NaN` would poison the whole row).
///
/// `w_pos >= 0` and `w_neg <= 0`, and `p`/`q` may contain ±∞ from upstream
/// conservative bounds. A zero weight contributes exactly 0 (the predecessor
/// neuron does not affect this output); a nonzero weight against a ±∞ bound
/// contributes the genuine ±∞. Accumulation can still legitimately reach ±∞,
/// which the firewall accepts in biases (only NaN is rejected there), and which
/// `concretize` degrades to `[-∞, +∞]` for that output — sound and the same
/// conservative outcome the firewall would have produced, but ONLY for the rows
/// that genuinely need it rather than the whole node.
fn saturating_matvec(
    w_pos: &Array2<f32>,
    p: &Array1<f32>,
    w_neg: &Array2<f32>,
    q: &Array1<f32>,
) -> Array1<f32> {
    let rows = w_pos.nrows();
    let cols = w_pos.ncols();
    let mut out = Array1::<f32>::zeros(rows);
    for i in 0..rows {
        let mut acc = 0.0f32;
        for k in 0..cols {
            let wp = w_pos[[i, k]];
            if wp != 0.0 {
                acc += wp * p[k];
            }
            let wn = w_neg[[i, k]];
            if wn != 0.0 {
                acc += wn * q[k];
            }
        }
        out[i] = acc;
    }
    out
}

/// Compose forward linear bounds through a ReLU layer (elementwise).
///
/// For each neuron i:
/// - Constrained active (split history): identity pass-through
/// - Constrained inactive (split history): zero
/// - Stable active (l >= 0): identity
/// - Stable inactive (u <= 0): zero
/// - Unstable (l < 0 < u): triangle relaxation
///   - Upper: `y <= λ*(z - l)` where `λ = u/(u-l)`
///   - Lower: `y >= α*z` where `α = 1 if u > -l, else 0` (area heuristic)
fn compose_relu_forward(
    pred: &LinearBounds,
    relu_node_name: &str,
    split_history: &GraphSplitHistory,
    pre_activation_bounds: Option<&BoundedTensor>,
) -> Result<LinearBounds> {
    let num_neurons = pred.num_outputs();

    let mut lower_a = pred.lower_a().clone();
    let mut lower_b = pred.lower_b().clone();
    let mut upper_a = pred.upper_a().clone();
    let mut upper_b = pred.upper_b().clone();

    let pre_flat = pre_activation_bounds.map(|b| b.flatten());

    for i in 0..num_neurons {
        if let Some(is_active) = split_history.is_constrained(relu_node_name, i) {
            if !is_active {
                // Inactive: zero all bounds for this neuron
                lower_a.row_mut(i).fill(0.0);
                lower_b[i] = 0.0;
                upper_a.row_mut(i).fill(0.0);
                upper_b[i] = 0.0;
            }
            // Active: identity, no change needed
        } else {
            // Unconstrained: use pre-activation bounds for relaxation
            let (l, u) = pre_flat
                .as_ref()
                .map(|flat| {
                    let lo = flat
                        .lower()
                        .iter()
                        .copied()
                        .nth(i)
                        .unwrap_or(f32::NEG_INFINITY);
                    let up = flat.upper().iter().copied().nth(i).unwrap_or(f32::INFINITY);
                    (lo, up)
                })
                .unwrap_or((f32::NEG_INFINITY, f32::INFINITY));

            if l >= 0.0 {
                // Stable active: identity
            } else if u <= 0.0 {
                // Stable inactive: zero
                lower_a.row_mut(i).fill(0.0);
                lower_b[i] = 0.0;
                upper_a.row_mut(i).fill(0.0);
                upper_b[i] = 0.0;
            } else {
                // Unstable: triangle relaxation
                let lambda = u / (u - l);
                // Upper: y <= λ*(z - l), z_upper = upper_a*x + upper_b
                // → upper_a_new = λ * upper_a, upper_b_new = λ * (upper_b - l)
                upper_a.row_mut(i).mapv_inplace(|v| v * lambda);
                upper_b[i] = lambda * (upper_b[i] - l);
                // Lower: y >= α*z, z_lower = lower_a*x + lower_b
                // Heuristic: α = 1 if u > -l (area-based), else 0
                let alpha = if u > -l { 1.0 } else { 0.0 };
                lower_a.row_mut(i).mapv_inplace(|v| v * alpha);
                // `α == 0` zeroes this neuron's lower relaxation; when the
                // upstream bias is ±∞ (an unsupported residual-Add predecessor
                // got conservative bounds), IEEE `0·(±∞) = NaN` would poison the
                // bias and trip the firewall for the whole node. α==0 means the
                // lower bound is identically 0, so set it directly (#3438).
                if alpha == 0.0 {
                    lower_b[i] = 0.0;
                } else {
                    lower_b[i] *= alpha;
                }
            }
        }
    }

    // Migrated from from_parts_unchecked: ReLU forward composition can
    // propagate NaN/Inf from upstream compose_linear_forward accumulation.
    // NaN firewall falls back to conservative bounds. See #3438.
    LinearBounds::new_or_conservative(lower_a, lower_b, upper_a, upper_b)
}

/// Compose forward linear bounds through an elementwise Add node `C = A + B`.
///
/// Both predecessors are already bounded as affine functions of the network
/// input over the box:
/// - `A(x) ∈ [lA_A·x + lb_A, uA_A·x + ub_A]`
/// - `B(x) ∈ [lA_B·x + lb_B, uA_B·x + ub_B]`
///
/// so their sum is bounded *componentwise* by the sum of the relaxations:
/// - `C(x) ≥ (lA_A + lA_B)·x + (lb_A + lb_B)`
/// - `C(x) ≤ (uA_A + uA_B)·x + (ub_A + ub_B)`
///
/// This is the forward analogue of [`AddLayer::propagate_linear_binary`], which
/// in the *backward* pass routes incoming bounds to both inputs; here we combine
/// the two already-computed input-relative relaxations. Modelling Add as a real
/// composition — instead of the conservative ±∞ fallback the generic arm
/// produces — is what lets `clip_interm_domain` tighten hidden bounds that live
/// downstream of a residual/skip connection.
///
/// # Soundness under directed rounding
///
/// Each coefficient sum is accumulated in `f64` (the sum of two `f32`s is exact
/// in `f64`) and stored back as `f32`; only that final store perturbs the
/// coefficient, by at most `u·|coeff|` with `u = 2^-24`. The worst-case effect
/// of those per-coefficient perturbations over the box `|x_j| ≤ abs_x[j]` is
/// `Σ_j u·|coeff[i,j]|·abs_x[j]`, which [`widen_bias_outward`] folds *outward*
/// into the bias (lower bias rounded down, upper bias rounded up, with a `2u`
/// safety factor and a directed final round). The stored affine form is
/// therefore a valid under/over-approximation of `A(x)+B(x)` for every `x` in
/// the box — never tighter than the true sum. Broadcasting Adds (mismatched
/// arity / input dimension) fall back to conservative bounds rather than guess.
fn compose_add_forward(
    pred_a: &LinearBounds,
    pred_b: &LinearBounds,
    abs_x: &Array1<f32>,
) -> Result<LinearBounds> {
    // Elementwise Add requires matching output arity and a shared input dim.
    if pred_a.num_outputs() != pred_b.num_outputs()
        || pred_a.lower_a().ncols() != pred_b.lower_a().ncols()
    {
        let n = pred_a.num_outputs().max(pred_b.num_outputs());
        return Ok(LinearBounds::conservative(n, abs_x.len()));
    }

    let lower_a = sum_coeffs_f64(pred_a.lower_a(), pred_b.lower_a());
    let upper_a = sum_coeffs_f64(pred_a.upper_a(), pred_b.upper_a());
    let lower_b_raw = sum_bias_f64(pred_a.lower_b(), pred_b.lower_b());
    let upper_b_raw = sum_bias_f64(pred_a.upper_b(), pred_b.upper_b());

    let lower_b = widen_bias_outward(&lower_a, &lower_b_raw, abs_x, false);
    let upper_b = widen_bias_outward(&upper_a, &upper_b_raw, abs_x, true);

    LinearBounds::new_or_conservative(lower_a, lower_b, upper_a, upper_b)
}

/// Elementwise `a + b` of two coefficient matrices, accumulated in `f64`.
///
/// The sum of two `f32`s is exact in `f64`, so the only rounding is the final
/// `f32` store (≤ `u·|result|`), which the caller compensates for outward via
/// [`widen_bias_outward`]. Same-signed ±∞ saturate; the only NaN-producing case
/// (`+∞ + −∞`) is caught by the downstream coefficient firewall, which degrades
/// the whole node to sound conservative bounds.
fn sum_coeffs_f64(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros(a.raw_dim());
    for ((o, &av), &bv) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = ((av as f64) + (bv as f64)) as f32;
    }
    out
}

/// Elementwise `a + b` of two bias vectors, accumulated in `f64`. ±∞ biases are
/// permitted (they encode conservative bounds); only NaN is rejected downstream.
fn sum_bias_f64(a: &Array1<f32>, b: &Array1<f32>) -> Array1<f32> {
    let mut out = Array1::<f32>::zeros(a.raw_dim());
    for ((o, &av), &bv) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = ((av as f64) + (bv as f64)) as f32;
    }
    out
}

/// Fold the f32 coefficient-storage roundoff outward into the bias so the affine
/// form `coeff·x + bias` stays a sound lower (`upper = false`) / upper bound over
/// a box with `|x_j| ≤ abs_x[j]`.
///
/// Each stored coefficient can deviate from the exact real value by at most
/// `u·|coeff|` (`u = 2^-24`, round-to-nearest); doubled to `2u` for headroom and
/// summed against the box magnitude gives a per-row slack
/// `Σ_j 2u·|coeff[i,j]|·abs_x[j] + 2u·|bias[i]|` that dominates the worst-case
/// excursion. Subtracting it from the lower bias / adding it to the upper bias,
/// then a directed final round, guarantees the affine form never excludes a true
/// value. A zero-width input dimension (`abs_x[j] == 0`) contributes nothing and
/// is skipped (avoids `0·∞ = NaN` against a genuinely unbounded coefficient).
fn widen_bias_outward(
    coeff: &Array2<f32>,
    bias: &Array1<f32>,
    abs_x: &Array1<f32>,
    upper: bool,
) -> Array1<f32> {
    // 2u with u = 2^-24 (one f32 round-to-nearest ulp), giving a 2× margin that
    // also absorbs the negligible f64-accumulation error.
    const REL: f64 = 2.0 / ((1u64 << 24) as f64);
    let rows = coeff.nrows();
    let cols = coeff.ncols();
    let mut out = bias.clone();
    for i in 0..rows {
        let mut slack = 0.0f64;
        for j in 0..cols {
            let ax = abs_x[j];
            if ax == 0.0 {
                continue;
            }
            let c = coeff[[i, j]].abs();
            if c != 0.0 {
                slack += REL * (c as f64) * (ax as f64);
            }
        }
        slack += REL * (bias[i].abs() as f64);
        let slack = slack as f32;
        out[i] = if upper {
            next_up_f32(out[i] + slack)
        } else {
            next_down_f32(out[i] - slack)
        };
    }
    out
}

/// Tighten intermediate bounds using forward linear bounds and split constraints.
///
/// Unlike `clip_interm_domain_full` which only selects unstable neurons (`l < 0 < u`),
/// this function tightens ALL neurons in each node. This is correct for graph
/// clip_in_alpha_crown because split constraints can tighten bounds that are already
/// stable (e.g., linear1 with bounds [0, 1] can be tightened to [0, 0.2] if a
/// downstream inactive constraint implies x <= 0.2).
///
/// Uses the lower-level APIs: `build_split_constraints` + `sort_out_constraints` +
/// `tighten_with_constraints` + `merge_bounds`, bypassing `select_objective_neurons`.
pub(in crate::beta_crown::engine::graph) fn apply_graph_clip_in_alpha_crown(
    graph: &GraphNetwork,
    split_history: &GraphSplitHistory,
    exec_order: &[String],
    bounds_cache: &mut HashMap<String, Arc<BoundedTensor>>,
    constrained_input: &BoundedTensor,
    forward_linear_bounds: &CachedLinearBounds,
    _topk: usize,
) -> Result<()> {
    let node_names: Vec<&str> = exec_order
        .iter()
        .filter_map(|node_name| {
            bounds_cache
                .contains_key(node_name)
                .then_some(node_name.as_str())
        })
        .collect();

    if node_names.is_empty() {
        return Ok(());
    }

    let (input_lower, input_upper) =
        constrained_input.flatten_to_ix1("clip_in_alpha_crown_input")?;
    let x_dim = input_lower.len();

    // Step 1: Build split constraints from forward linear bounds
    let linear_bounds_for_split =
        |relu_node_name: &str, neuron_idx: usize| -> Option<(Array1<f32>, f32, Array1<f32>, f32)> {
            let relu_node = graph.nodes.get(relu_node_name)?;
            if !matches!(relu_node.layer, Layer::ReLU(_)) {
                return None;
            }
            let pre_node_name = relu_node.inputs.first()?.as_str();
            let lb = forward_linear_bounds.linear_bounds(pre_node_name)?;
            single_neuron_linear_bounds(&lb, neuron_idx)
        };

    let constraints = build_split_constraints(split_history, linear_bounds_for_split, x_dim)?;
    if constraints.is_empty() {
        return Ok(());
    }

    let preprocessed = sort_out_constraints(&constraints, &input_lower, &input_upper)?;
    if preprocessed.a_active.nrows() == 0 {
        return Ok(());
    }

    // Step 2: Tighten ALL neurons in each node (not just unstable)
    for node_name in &node_names {
        let Some(fwd_lb) = forward_linear_bounds.linear_bounds(node_name) else {
            continue;
        };
        let Some(old_bounds) = bounds_cache.get(*node_name) else {
            continue;
        };

        let (old_lower, old_upper) = old_bounds.flatten_to_ix1("clip_in_alpha_crown_node")?;
        let n_neurons = old_lower.len();
        let all_indices: Vec<usize> = (0..n_neurons).collect();

        let Some((obj_lower_a, obj_lower_b, obj_upper_a, obj_upper_b)) =
            selected_neuron_linear_bounds(&fwd_lb, &all_indices)
        else {
            continue;
        };

        let (tightened_lower, tightened_upper) = tighten_with_constraints(
            &preprocessed,
            &obj_lower_a,
            &obj_lower_b,
            &obj_upper_a,
            &obj_upper_b,
            &input_lower,
            &input_upper,
        )?;

        let (merged_lower, merged_upper) = merge_bounds(
            &old_lower,
            &old_upper,
            &tightened_lower,
            &tightened_upper,
            &all_indices,
        );

        if bounds_changed(
            old_lower.iter().copied(),
            old_upper.iter().copied(),
            merged_lower.iter().copied(),
            merged_upper.iter().copied(),
        ) {
            let shape = old_bounds.shape().to_vec();
            let lower_arr: ArrayD<f32> =
                merged_lower
                    .into_shape_clone(IxDyn(&shape))
                    .map_err(|err| {
                        NyError::InternalError(format!(
                            "clip_in_alpha_crown: reshape lower failed for '{}': {}",
                            node_name, err
                        ))
                    })?;
            let upper_arr: ArrayD<f32> =
                merged_upper
                    .into_shape_clone(IxDyn(&shape))
                    .map_err(|err| {
                        NyError::InternalError(format!(
                            "clip_in_alpha_crown: reshape upper failed for '{}': {}",
                            node_name, err
                        ))
                    })?;
            // Widen repair instead of strict `new`: tightened intermediate bounds
            // may be non-finite (±Inf from a degenerate BatchNorm channel with
            // var+eps ~= 0, or NaN escaping an upstream firewall). Widen maps NaN
            // to the conservative direction (-inf lower / +inf upper) and keeps
            // ±Inf as-is — a sound unbounded intermediate — so the graph
            // beta-CROWN attempt can split/time out rather than aborting here.
            // Fresh Arc: clip replaces entries wholesale, never mutates a
            // shared tensor in place (#cone-delta increment 2 aliasing rule).
            bounds_cache.insert(
                (*node_name).to_string(),
                Arc::new(BoundedTensor::new_repaired(
                    lower_arr,
                    upper_arr,
                    RepairStrategy::Widen,
                )?),
            );
        }
    }

    Ok(())
}

fn single_neuron_linear_bounds(
    linear_bounds: &LinearBounds,
    neuron_idx: usize,
) -> Option<(Array1<f32>, f32, Array1<f32>, f32)> {
    if neuron_idx >= linear_bounds.num_outputs() {
        return None;
    }

    Some((
        linear_bounds.lower_a().row(neuron_idx).to_owned(),
        linear_bounds.lower_b()[neuron_idx],
        linear_bounds.upper_a().row(neuron_idx).to_owned(),
        linear_bounds.upper_b()[neuron_idx],
    ))
}

fn selected_neuron_linear_bounds(
    linear_bounds: &LinearBounds,
    neuron_indices: &[usize],
) -> Option<BatchLinearParts> {
    let n_selected = neuron_indices.len();
    let n_inputs = linear_bounds.num_inputs();

    let mut lower_a = Array2::zeros((n_selected, n_inputs));
    let mut lower_b = Array1::zeros(n_selected);
    let mut upper_a = Array2::zeros((n_selected, n_inputs));
    let mut upper_b = Array1::zeros(n_selected);

    for (row_idx, &neuron_idx) in neuron_indices.iter().enumerate() {
        if neuron_idx >= linear_bounds.num_outputs() {
            return None;
        }

        lower_a
            .row_mut(row_idx)
            .assign(&linear_bounds.lower_a().row(neuron_idx));
        lower_b[row_idx] = linear_bounds.lower_b()[neuron_idx];
        upper_a
            .row_mut(row_idx)
            .assign(&linear_bounds.upper_a().row(neuron_idx));
        upper_b[row_idx] = linear_bounds.upper_b()[neuron_idx];
    }

    Some((lower_a, lower_b, upper_a, upper_b))
}

fn bounds_changed(
    old_lower: impl Iterator<Item = f32>,
    old_upper: impl Iterator<Item = f32>,
    new_lower: impl Iterator<Item = f32>,
    new_upper: impl Iterator<Item = f32>,
) -> bool {
    old_lower.zip(old_upper).zip(new_lower.zip(new_upper)).any(
        |((old_l, old_u), (new_l, new_u))| {
            new_l > old_l
                || new_u < old_u
                || new_l.is_nan()
                || new_u.is_nan()
                || old_l.is_nan()
                || old_u.is_nan()
        },
    )
}

#[cfg(test)]
mod tests {
    //! NaN-safety regression tests for the forward linear-bounds composition
    //! through residual nets (cersyve / nn4sys). The forward pass models only
    //! Linear/ReLU; residual `Add` skip-connections fall through to conservative
    //! `±∞` biases, and a naive `w.dot(±∞)` evaluates `0·∞ = NaN` for every zero
    //! weight, poisoning the whole bias row and tripping the downstream firewall
    //! (which then degrades the WHOLE node to `[-∞,+∞]`). The fix (#3438) is a
    //! `saturating_matvec` defining `0·∞ = 0` plus an `α==0` special-case in the
    //! ReLU lower relaxation. These tests pin the exact regression.

    use super::*;
    use ny_core::Result;

    /// `0·(±∞)` must yield `0`, never NaN, in the bias matvec. With `w_pos >= 0`
    /// against a `-∞` lower bias and `w_neg <= 0` against a `+∞` upper bias, every
    /// nonzero-weight infinite term is `-∞`; there is never a `+∞` to cancel it,
    /// so the accumulation is finite-where-zero-weighted and `-∞`-where-genuine —
    /// always sound, never NaN.
    #[test]
    fn saturating_matvec_treats_zero_times_inf_as_zero() {
        // Two outputs. Predecessor is conservative: lower_b = -inf, upper_b = +inf.
        let p = Array1::from(vec![f32::NEG_INFINITY, f32::NEG_INFINITY]); // pred.lower_b
        let q = Array1::from(vec![f32::INFINITY, f32::INFINITY]); // pred.upper_b

        // Row 0: zero weight on the inf predecessor -> contributes exactly 0.
        // Row 1: nonzero weight on the inf predecessor -> genuine -inf.
        let w_pos = ndarray::array![[0.0_f32, 0.0], [1.0, 0.0]];
        let w_neg = ndarray::array![[-1.0_f32, 0.0], [0.0, 0.0]];

        let lower_b = saturating_matvec(&w_pos, &p, &w_neg, &q);

        // Row 0: 0·(-inf) [w_pos] + (-1)·(+inf) [w_neg] -> the w_neg term is a
        // genuine nonzero weight against +inf, so the row is -inf (NOT NaN).
        assert!(
            !lower_b[0].is_nan(),
            "row0 must not be NaN, got {}",
            lower_b[0]
        );
        assert_eq!(lower_b[0], f32::NEG_INFINITY);
        // Row 1: 1·(-inf) -> -inf; never NaN.
        assert!(
            !lower_b[1].is_nan(),
            "row1 must not be NaN, got {}",
            lower_b[1]
        );
        assert_eq!(lower_b[1], f32::NEG_INFINITY);
    }

    /// A fully zero weight against ±∞ predecessor biases must yield a finite (0)
    /// bias for that output — the predecessor cannot affect it at all.
    #[test]
    fn saturating_matvec_zero_row_is_finite_zero() {
        let p = Array1::from(vec![f32::NEG_INFINITY]);
        let q = Array1::from(vec![f32::INFINITY]);
        let w_pos = ndarray::array![[0.0_f32]];
        let w_neg = ndarray::array![[0.0_f32]];
        let out = saturating_matvec(&w_pos, &p, &w_neg, &q);
        assert_eq!(
            out[0], 0.0,
            "zero-weight output must be exactly 0, got {}",
            out[0]
        );
        assert!(out[0].is_finite());
    }

    /// End-to-end: composing a Linear layer on top of a conservative (±∞-bias)
    /// residual-Add predecessor must NOT trip the firewall (no NaN in any bias),
    /// and the resulting bounds must stay sound (finite-or-genuine-±∞, never NaN).
    #[test]
    fn compose_linear_forward_no_nan_on_conservative_pred() -> Result<()> {
        // Predecessor: conservative bounds (this is exactly what a residual Add
        // skip-connection produces in the forward pass). 2 outputs, 2 inputs.
        let pred = LinearBounds::conservative(2, 2);

        // Linear layer with a mix of zero and nonzero, +/- weights — the zero
        // entries are the ones that used to produce 0·∞ = NaN.
        let weight = ndarray::array![[0.0_f32, -1.0], [1.0, 0.0]];
        let bias = Array1::from(vec![0.5_f32, -0.25]);

        let out = compose_linear_forward(&pred, &weight, Some(&bias))?;

        // No NaN anywhere (firewall would NOT fire on this).
        assert!(
            out.lower_b().iter().all(|v| !v.is_nan()),
            "lower_b has NaN: {:?}",
            out.lower_b()
        );
        assert!(
            out.upper_b().iter().all(|v| !v.is_nan()),
            "upper_b has NaN: {:?}",
            out.upper_b()
        );
        assert!(out.lower_a().iter().all(|v| !v.is_nan()));
        assert!(out.upper_a().iter().all(|v| !v.is_nan()));

        // Soundness: lower_b <= upper_b elementwise (allowing ±∞).
        for (lo, up) in out.lower_b().iter().zip(out.upper_b().iter()) {
            assert!(lo <= up, "unsound: lower_b {} > upper_b {}", lo, up);
        }
        Ok(())
    }

    /// End-to-end: the ReLU unstable-triangle lower relaxation with `α == 0`
    /// against a conservative (±∞-bias) predecessor must set the lower bias to 0
    /// (the relaxation `y >= 0·z` is identically 0), NOT `0·(±∞) = NaN`.
    #[test]
    fn compose_relu_forward_no_nan_on_conservative_pred() -> Result<()> {
        use ny_tensor::BoundedTensor;

        // Conservative predecessor (residual-Add fallthrough): lower_b = -inf.
        let pred = LinearBounds::conservative(2, 2);

        // Unconstrained ReLU with unstable pre-activations that force the α==0
        // branch: u <= -l (i.e. |lower| >= |upper|) -> alpha = 0.
        // Neuron 0: l = -2, u = 1  -> u <= -l (1 <= 2) -> alpha = 0 (the bug path).
        // Neuron 1: l = -1, u = 2  -> u >  -l (2 >  1) -> alpha = 1.
        let lower = ndarray::arr1(&[-2.0_f32, -1.0]).into_dyn();
        let upper = ndarray::arr1(&[1.0_f32, 2.0]).into_dyn();
        let pre = BoundedTensor::new(lower, upper)?;

        let history = GraphSplitHistory::new();
        let out = compose_relu_forward(&pred, "relu_test", &history, Some(&pre))?;

        // No NaN anywhere -> firewall does not fire.
        assert!(
            out.lower_b().iter().all(|v| !v.is_nan()),
            "lower_b has NaN: {:?}",
            out.lower_b()
        );
        assert!(
            out.upper_b().iter().all(|v| !v.is_nan()),
            "upper_b has NaN: {:?}",
            out.upper_b()
        );

        // Neuron 0 (alpha==0): lower relaxation y >= 0 -> lower_b must be exactly 0.
        assert_eq!(
            out.lower_b()[0],
            0.0,
            "alpha==0 neuron lower_b must be 0, got {}",
            out.lower_b()[0]
        );

        // Soundness: lower_b <= upper_b.
        for (lo, up) in out.lower_b().iter().zip(out.upper_b().iter()) {
            assert!(lo <= up, "unsound: lower_b {} > upper_b {}", lo, up);
        }
        Ok(())
    }

    /// Guard against silent regression to the OLD behavior: the naive
    /// `w.dot(±∞)` would have produced NaN here. Confirm the saturating path
    /// does not, and that the result is NOT all-conservative when only some rows
    /// genuinely need it (the whole point of the fix: degrade only the rows that
    /// touch a nonzero-weighted ±∞, not the entire node).
    #[test]
    fn compose_linear_forward_degrades_only_affected_rows() -> Result<()> {
        let pred = LinearBounds::conservative(2, 2);
        // Row 0: only zero weights on the conservative pred -> finite bias.
        // Row 1: a nonzero weight on the conservative pred -> genuine ±∞ bias.
        let weight = ndarray::array![[0.0_f32, 0.0], [0.0, 1.0]];
        let bias = Array1::from(vec![7.0_f32, 0.0]);
        let out = compose_linear_forward(&pred, &weight, Some(&bias))?;

        // Row 0: finite (= bias 7.0), proving we did NOT degrade the whole node.
        assert!(
            out.lower_b()[0].is_finite(),
            "row0 lower_b should be finite, got {}",
            out.lower_b()[0]
        );
        assert!(
            out.upper_b()[0].is_finite(),
            "row0 upper_b should be finite, got {}",
            out.upper_b()[0]
        );
        assert_eq!(out.lower_b()[0], 7.0);
        assert_eq!(out.upper_b()[0], 7.0);
        // Row 1: genuine ±∞ (sound, the pred is truly unbounded through this weight).
        assert_eq!(out.lower_b()[1], f32::NEG_INFINITY);
        assert_eq!(out.upper_b()[1], f32::INFINITY);
        Ok(())
    }

    /// Residual `Add` forward composition sums BOTH predecessors' relaxations
    /// (instead of the old conservative ±∞ fallthrough) and stays sound over the
    /// box: composed coefficients are the exact elementwise sum, and the bias is
    /// folded *outward* (lower bias ≤ exact sum, upper bias ≥ exact sum) to absorb
    /// the f32 coefficient-storage roundoff.
    #[test]
    fn compose_add_forward_sums_and_stays_sound() -> Result<()> {
        // Two predecessors, each 1 output over a 2-D input box [-1, 1]^2.
        let pred_a = LinearBounds::new(
            ndarray::array![[1.0_f32, 0.5]],
            Array1::from(vec![0.1_f32]),
            ndarray::array![[1.0_f32, 0.5]],
            Array1::from(vec![0.3_f32]),
        )?;
        let pred_b = LinearBounds::new(
            ndarray::array![[-0.5_f32, 2.0]],
            Array1::from(vec![-0.2_f32]),
            ndarray::array![[-0.5_f32, 2.0]],
            Array1::from(vec![0.4_f32]),
        )?;
        let abs_x = Array1::from(vec![1.0_f32, 1.0]);

        let out = compose_add_forward(&pred_a, &pred_b, &abs_x)?;

        // Coefficients are the exact elementwise sum.
        assert_eq!(out.lower_a(), &ndarray::array![[0.5_f32, 2.5]]);
        assert_eq!(out.upper_a(), &ndarray::array![[0.5_f32, 2.5]]);

        // Bias folded outward: lower ≤ exact (-0.1), upper ≥ exact (0.7), and the
        // widening is tiny (well under 1e-3 for these O(1) magnitudes).
        let exact_lo = 0.1_f32 + (-0.2);
        let exact_up = 0.3_f32 + 0.4;
        assert!(
            out.lower_b()[0] <= exact_lo,
            "lower bias not widened down: {} > {}",
            out.lower_b()[0],
            exact_lo
        );
        assert!(
            out.upper_b()[0] >= exact_up,
            "upper bias not widened up: {} < {}",
            out.upper_b()[0],
            exact_up
        );
        assert!((out.lower_b()[0] - exact_lo).abs() < 1e-3);
        assert!((out.upper_b()[0] - exact_up).abs() < 1e-3);

        // Soundness at the box corners: the composed affine bound must contain the
        // sum of the two predecessor relaxations for every x in the box.
        for &x0 in &[-1.0_f32, 1.0] {
            for &x1 in &[-1.0_f32, 1.0] {
                let composed_lo =
                    out.lower_a()[[0, 0]] * x0 + out.lower_a()[[0, 1]] * x1 + out.lower_b()[0];
                let composed_up =
                    out.upper_a()[[0, 0]] * x0 + out.upper_a()[[0, 1]] * x1 + out.upper_b()[0];
                let sum_lo = (1.0 * x0 + 0.5 * x1 + 0.1) + (-0.5 * x0 + 2.0 * x1 - 0.2);
                let sum_up = (1.0 * x0 + 0.5 * x1 + 0.3) + (-0.5 * x0 + 2.0 * x1 + 0.4);
                assert!(
                    composed_lo <= sum_lo + 1e-5,
                    "unsound lower at ({x0},{x1}): {composed_lo} > {sum_lo}"
                );
                assert!(
                    composed_up >= sum_up - 1e-5,
                    "unsound upper at ({x0},{x1}): {composed_up} < {sum_up}"
                );
            }
        }
        Ok(())
    }

    /// A conservative (±∞-bias) predecessor — e.g. an Add whose own input is an
    /// unsupported upstream node — must produce no NaN and stay sound (the
    /// genuine ±∞ propagates; the other branch's finite contribution is kept).
    #[test]
    fn compose_add_forward_no_nan_on_conservative_pred() -> Result<()> {
        let pred_a = LinearBounds::conservative(1, 2);
        let pred_b = LinearBounds::new(
            ndarray::array![[0.5_f32, -1.0]],
            Array1::from(vec![0.25_f32]),
            ndarray::array![[0.5_f32, -1.0]],
            Array1::from(vec![0.75_f32]),
        )?;
        let abs_x = Array1::from(vec![1.0_f32, 1.0]);

        let out = compose_add_forward(&pred_a, &pred_b, &abs_x)?;

        assert!(
            out.lower_b().iter().all(|v| !v.is_nan()),
            "lower_b NaN: {:?}",
            out.lower_b()
        );
        assert!(
            out.upper_b().iter().all(|v| !v.is_nan()),
            "upper_b NaN: {:?}",
            out.upper_b()
        );
        // The conservative branch makes the sum genuinely unbounded (sound).
        assert_eq!(out.lower_b()[0], f32::NEG_INFINITY);
        assert_eq!(out.upper_b()[0], f32::INFINITY);
        Ok(())
    }

    /// Shape-mismatched (broadcasting) Add falls back to conservative bounds
    /// rather than guessing — sound by construction.
    #[test]
    fn compose_add_forward_shape_mismatch_is_conservative() -> Result<()> {
        let pred_a = LinearBounds::identity(2); // 2 outputs
        let pred_b = LinearBounds::identity(3); // 3 outputs — mismatched arity
        let abs_x = Array1::from(vec![1.0_f32, 1.0, 1.0]);
        let out = compose_add_forward(&pred_a, &pred_b, &abs_x)?;
        // Conservative: ±∞ biases, no NaN.
        assert!(out.lower_b().iter().all(|v| *v == f32::NEG_INFINITY));
        assert!(out.upper_b().iter().all(|v| *v == f32::INFINITY));
        Ok(())
    }
}
