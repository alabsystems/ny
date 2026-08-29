// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #alpha-step v4: analytical per-parameter α gradients for the
//! `relu_split_route` multi-objective optimizer.
//!
//! Replaces the shared-slope α SPSA (two extra spec-guided propagations per
//! α round, one scalar slope smeared across every parameter) with the exact
//! per-parameter derivative of the critical objective row's lower bound,
//! computed from intermediates the storing-intermediates pass already
//! produced this iteration — ZERO extra propagations.
//!
//! Math (sequential reference: `optimization/gradients/joint.rs:450-563`):
//! the spec row's lower bound is an affine function of the input built by the
//! backward pass substituting, at every ReLU neuron `j`, the relaxation
//! branch selected by the sign of the backward coefficient `A_j` (the
//! coefficient of the ReLU *output* after all output-side substitutions):
//! `A_j >= 0` → lower branch `α_j·ẑ_j`, `A_j < 0` → upper chord
//! `λ_j·ẑ_j + μ_j`. Concretizing at the corner
//! `x*_i = l_i if final_a_i >= 0 else u_i` gives the reported bound, and by
//! the envelope theorem
//!
//! ```text
//! d(lb)/d(α_j) = max(A_j, 0) · w_j
//! ```
//!
//! where `w_j` is the *pre-activation* value of neuron `j` at `x*` forward-
//! evaluated through the SAME branch-relaxed network (`w_j` read BEFORE the
//! ReLU transform — exactly `joint.rs`'s `w[neuron_idx]`).
//!
//! Scope: sequential single-input chains of Linear / ReLU / Flatten /
//! Reshape nodes (the relusplitter mnist_fc family). Anything else returns
//! `None` and the α round is a no-op — sound, since any α in [0, 1] is a
//! valid relaxation slope; only convergence speed is at stake.
//!
//! Soundness: unconditional. Gradients only choose the next stored α DELTA
//! candidate (`graph_runtime.rs` composes and clamps into [0, 1]); every
//! candidate's bound is re-evaluated through the sound spec-guided pass and
//! banked by the existing best-margin keeper.
//!
//! # STATUS: BANKED AND PARKED — deliberately unwired, not abandoned
//!
//! Nothing calls into this module today, so every item in it reports
//! `dead_code`. That is the INTENDED state, recorded by 862868d3c: "analytical
//! alpha gradient v1: FD-verified module banked (alpha_analytic.rs); greedy
//! critical-row stepping regresses (-12.31) and is parked — row-aggregated
//! follow-up scoped". The maths above is finite-difference verified; what
//! regressed was the greedy critical-row STEPPING policy built on top of it,
//! and the scoped follow-up is to aggregate over rows instead.
//!
//! The allow below is therefore a statement of that decision, not a way to make
//! a lint gate green. Deleting the module would discard verified work whose
//! successor is scoped; leaving it unannotated would make `-D warnings`
//! permanently red for every other lane. If the row-aggregated follow-up lands
//! and wires this in, DELETE the allow with it — a parked module that has been
//! un-parked must go back to being lint-visible.
#![allow(
    dead_code,
    reason = "banked FD-verified v1, parked per 862868d3c pending the row-aggregated follow-up"
)]

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::Array1;
use ny_tensor::BoundedTensor;

use crate::beta_crown::state::{GraphBetaState, GraphDomainAlphaState};
use crate::bounds::{nan_propagating_max, GraphAlphaCrownIntermediate};
use crate::layers::activations::relu_crossing_upper_chord;
use crate::layers::Layer;
use crate::network::NETWORK_INPUT;
use crate::GraphNetwork;

/// Per-ReLU data recorded by the backward reconstruction and consumed by the
/// forward sensitivity pass.
struct ReluRowData {
    /// Backward coefficient of the critical row at this ReLU's OUTPUT, before
    /// the relaxation is applied (`A_j`). Captured `a_at_relu` row when the
    /// pass stored one for this node, otherwise reconstructed.
    coeff: Array1<f32>,
    /// Composed lower-relaxation slopes (stable 0/1, crossing α) exactly as
    /// `build_alpha_array` served them to the backward pass.
    alphas: Array1<f32>,
    /// Upper-chord slope λ per neuron (identity/zero for stable neurons).
    lambda: Array1<f32>,
    /// Upper-chord intercept μ per neuron.
    mu: Array1<f32>,
    /// Pre-activation box (clamped, as the backward pass saw it).
    pre_lower: Array1<f32>,
    pre_upper: Array1<f32>,
}

/// Collect each ReLU node's pre-activation box (its single input node's
/// bounds) from this iterate's constrained per-node bounds map. Cheap Arc
/// clones; called only on relu-split α rounds, before the map moves into the
/// best-result keeper.
pub(in crate::beta_crown::engine::graph) fn collect_relu_pre_boxes(
    graph: &GraphNetwork,
    node_bounds: &HashMap<String, Arc<BoundedTensor>>,
) -> HashMap<String, Arc<BoundedTensor>> {
    let mut boxes = HashMap::new();
    for name in graph.node_names() {
        let Some(node) = graph.node(name) else {
            continue;
        };
        if !matches!(node.layer(), Layer::ReLU(_)) {
            continue;
        }
        let [input_name] = node.inputs() else {
            continue;
        };
        if let Some(bounds) = node_bounds.get(input_name) {
            boxes.insert(name.clone(), Arc::clone(bounds));
        }
    }
    boxes
}

/// The critical objective row driving the scalar margin: the unverified row
/// with the minimum (disjunctive) or maximum (conjunctive) finite
/// `lower - threshold`, mirroring `compute_margin`'s selection. NaN margins
/// are skipped (they cannot rank); `None` when no finite unverified row
/// exists.
fn critical_row(
    obj_bounds: &[(f32, f32)],
    thresholds: &[f32],
    verified_mask: &[bool],
    conjunctive: bool,
) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (idx, ((&(lower, _), &threshold), &verified)) in obj_bounds
        .iter()
        .zip(thresholds.iter())
        .zip(verified_mask.iter())
        .enumerate()
    {
        if verified {
            continue;
        }
        let margin = lower - threshold;
        if margin.is_nan() {
            continue;
        }
        let better = match best {
            None => true,
            Some((_, best_margin)) => {
                if conjunctive {
                    margin > best_margin
                } else {
                    margin < best_margin
                }
            }
        };
        if better {
            best = Some((idx, margin));
        }
    }
    best.map(|(idx, _)| idx)
}

/// Resolve the sequential Linear/ReLU/Flatten/Reshape chain from the network
/// input to the output node, in forward order. `None` when the graph is not
/// such a chain (multi-input nodes, unsupported layers, missing nodes).
fn sequential_chain(graph: &GraphNetwork) -> Option<Vec<&crate::network::GraphNode>> {
    let mut name = if graph.output_name().is_empty() {
        graph.node_names().last()?.as_str()
    } else {
        graph.output_name()
    };
    let mut chain = Vec::new();
    loop {
        let node = graph.node(name)?;
        match node.layer() {
            Layer::Linear(_) | Layer::ReLU(_) | Layer::Flatten(_) | Layer::Reshape(_) => {}
            _ => return None,
        }
        let [input_name] = node.inputs() else {
            return None;
        };
        chain.push(node);
        if input_name == NETWORK_INPUT {
            break;
        }
        name = input_name.as_str();
        // Defensive cycle guard: a chain longer than the node count repeats.
        if chain.len() > graph.num_nodes() {
            return None;
        }
    }
    chain.reverse();
    Some(chain)
}

/// This ReLU node's pre-activation box as flat `(lower, upper)` arrays:
/// prefer the pass's own captured (clamped) `pre_relu_bounds`, fall back to
/// the iterate's constrained per-node bounds map.
fn relu_pre_box(
    node_name: &str,
    intermediate: &GraphAlphaCrownIntermediate,
    relu_pre_boxes: &HashMap<String, Arc<BoundedTensor>>,
) -> Option<(Array1<f32>, Array1<f32>)> {
    if let Some((lower, upper)) = intermediate.pre_relu_bounds.get(node_name) {
        return Some((lower.clone(), upper.clone()));
    }
    let bounds = relu_pre_boxes.get(node_name)?;
    let flat = bounds.flatten();
    let n = flat.len();
    let mut lower = Array1::zeros(n);
    let mut upper = Array1::zeros(n);
    for i in 0..n {
        lower[i] = flat.lower()[[i]];
        upper[i] = flat.upper()[[i]];
    }
    Some((lower, upper))
}

/// Accumulate `d(lb_row)/d(α_j)` into `alpha_state`'s per-neuron lower-path
/// `.grad` fields for the given spec row.
///
/// Two passes over the sequential chain:
/// 1. Backward: reconstruct the critical row's backward coefficient at every
///    ReLU output (preferring the pass's captured `a_at_relu` rows), the
///    branch each neuron took, and the composed relaxation slopes.
/// 2. Forward: evaluate the branch-relaxed network at the concretization
///    corner `x*` (from `intermediate.final_bounds`), reading each neuron's
///    pre-activation `w_j` BEFORE its ReLU transform and accumulating
///    `max(A_j, 0) · w_j` for unstable neurons.
///
/// Returns `None` (leaving any already-accumulated grads harmlessly staged;
/// callers zero first) when the graph or intermediates do not support the
/// reconstruction.
pub(in crate::beta_crown::engine::graph) fn accumulate_alpha_gradients_for_row(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    beta_state: &GraphBetaState,
    alpha_state: &mut GraphDomainAlphaState,
    intermediate: &GraphAlphaCrownIntermediate,
    relu_pre_boxes: &HashMap<String, Arc<BoundedTensor>>,
    spec_row: ndarray::ArrayView1<'_, f32>,
    row_idx: usize,
) -> Option<()> {
    let chain = sequential_chain(graph)?;

    // Pass 1: backward coefficient reconstruction (output → input).
    let mut relu_rows: HashMap<&str, ReluRowData> = HashMap::new();
    let mut a: Array1<f32> = spec_row.to_owned();
    for node in chain.iter().rev() {
        match node.layer() {
            Layer::Linear(linear) => {
                let weight = linear.weight();
                if a.len() != weight.nrows() {
                    return None;
                }
                a = weight.t().dot(&a);
            }
            Layer::ReLU(_) => {
                let name = node.name();
                let (pre_lower, pre_upper) = relu_pre_box(name, intermediate, relu_pre_boxes)?;
                let n = pre_lower.len();
                if a.len() != n {
                    return None;
                }
                // Prefer the exact captured backward row (includes every
                // rounding and patch nuance of the real pass) when its shape
                // matches; otherwise use the reconstruction.
                let coeff: Array1<f32> = match intermediate.a_at_relu.get(name) {
                    Some(rows) if row_idx < rows.nrows() && rows.ncols() == n => {
                        rows.row(row_idx).to_owned()
                    }
                    _ => a.clone(),
                };
                let pre_tensor =
                    BoundedTensor::new(pre_lower.clone().into_dyn(), pre_upper.clone().into_dyn())
                        .ok()?;
                // The same composed slopes (delta-form under split_refresh)
                // the backward pass consumed this iteration.
                let alphas = alpha_state.build_alpha_array(name, &pre_tensor);
                let mut lambda = Array1::<f32>::zeros(n);
                let mut mu = Array1::<f32>::zeros(n);
                for j in 0..n {
                    let (l, u) = (pre_lower[j], pre_upper[j]);
                    if l.is_nan() || u.is_nan() {
                        return None;
                    }
                    if l >= 0.0 {
                        lambda[j] = 1.0;
                    } else if u <= 0.0 {
                        // λ = μ = 0.
                    } else {
                        let (slope, intercept) = relu_crossing_upper_chord(l, u, None);
                        lambda[j] = slope;
                        mu[j] = intercept;
                    }
                }
                // Branch-selected substitution: coefficients w.r.t. the
                // pre-activation.
                let mut a_in = Array1::<f32>::zeros(n);
                for j in 0..n {
                    a_in[j] = if coeff[j] >= 0.0 {
                        coeff[j] * alphas[j]
                    } else {
                        coeff[j] * lambda[j]
                    };
                }
                // β split contributions mutate the pre-activation columns
                // after the ReLU transform (`apply_beta_split_to_column`:
                // lower coefficient -= signed β).
                for entry in beta_state.entries_for_node(name) {
                    let idx = entry.neuron_idx();
                    if idx < n {
                        a_in[idx] -= entry.signed_value();
                    }
                }
                a = a_in;
                relu_rows.insert(
                    name,
                    ReluRowData {
                        coeff,
                        alphas,
                        lambda,
                        mu,
                        pre_lower,
                        pre_upper,
                    },
                );
            }
            Layer::Flatten(_) | Layer::Reshape(_) => {}
            _ => return None,
        }
    }

    // Concretization corner x* from the pass's own final input-level row.
    let final_bounds = &intermediate.final_bounds;
    let input_flat = input.flatten();
    let input_dim = input_flat.len();
    if row_idx >= final_bounds.num_outputs() || final_bounds.num_inputs() != input_dim {
        return None;
    }
    let final_a = final_bounds.lower_a().row(row_idx);
    let mut v = Array1::<f32>::zeros(input_dim);
    for i in 0..input_dim {
        v[i] = if final_a[i] >= 0.0 {
            input_flat.lower()[[i]]
        } else {
            input_flat.upper()[[i]]
        };
    }

    // Pass 2: forward sensitivity through the branch-relaxed network.
    for node in &chain {
        match node.layer() {
            Layer::Linear(linear) => {
                let weight = linear.weight();
                if v.len() != weight.ncols() {
                    return None;
                }
                let mut out = weight.dot(&v);
                if let Some(bias) = linear.bias() {
                    out += bias;
                }
                v = out;
            }
            Layer::ReLU(_) => {
                let name = node.name();
                let data = relu_rows.get(name)?;
                let n = data.coeff.len();
                if v.len() != n {
                    return None;
                }
                for j in 0..n {
                    let l = data.pre_lower[j];
                    let u = data.pre_upper[j];
                    if l < 0.0 && u > 0.0 {
                        // d(lb)/d(α_j) = max(A_j, 0) · w_j, with w_j read
                        // BEFORE the transform. NaN-safe max (#2643 analog);
                        // `accumulate_grad` drops non-finite values.
                        let grad = nan_propagating_max(data.coeff[j], 0.0) * v[j];
                        alpha_state.accumulate_grad(name, j, grad);
                    }
                    v[j] = if data.coeff[j] >= 0.0 {
                        data.alphas[j] * v[j]
                    } else {
                        data.lambda[j] * v[j] + data.mu[j]
                    };
                }
            }
            Layer::Flatten(_) | Layer::Reshape(_) => {}
            _ => return None,
        }
    }

    Some(())
}

/// One analytical α round for the relu-split multi-objective optimizer:
/// zero the α grads, accumulate the critical row's analytical gradients, and
/// ascent-step the stored lower-path DELTAS
/// (`set_alpha((stored + lr·grad).clamp(0, 1))` — `set_alpha` clamps).
///
/// Returns the max |grad| over stepped parameters for the convergence check,
/// or `None` (α state untouched beyond the zeroed grads) when the topology
/// or intermediates are unsupported.
#[allow(clippy::too_many_arguments)]
pub(in crate::beta_crown::engine::graph) fn relu_split_alpha_analytical_round(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    beta_state: &GraphBetaState,
    alpha_state: &mut GraphDomainAlphaState,
    intermediate: &GraphAlphaCrownIntermediate,
    relu_pre_boxes: &HashMap<String, Arc<BoundedTensor>>,
    spec_matrix: &ndarray::Array2<f32>,
    obj_bounds: &[(f32, f32)],
    thresholds: &[f32],
    verified_mask: &[bool],
    conjunctive: bool,
    lr: f32,
) -> Option<f32> {
    // ROW-AGGREGATED (#alpha-analytic v2, 2026-08-27): the critical-row-only
    // step was FD-correct yet regressed the frontier -5.59 → -12.31 — a
    // greedy step for ONE row wrecks the other eight, while the SPSA slope
    // it replaced was implicitly multi-row through the margin. Aggregate the
    // gradient over EVERY unverified row instead (uniform sum; `zero_grad`
    // once, accumulate per row — abCROWN's multi-spec pattern), so the step
    // serves the ensemble with per-parameter resolution. Rows past the spec
    // matrix and rows whose reconstruction fails are skipped (their
    // contribution is simply absent, never a poisoned step); if NO row
    // accumulates, the round no-ops exactly like the unsupported-topology
    // path.
    let _ = critical_row(obj_bounds, thresholds, verified_mask, conjunctive)?;
    alpha_state.zero_grad();
    let mut accumulated_rows = 0usize;
    for row in 0..obj_bounds.len().min(spec_matrix.nrows()) {
        if verified_mask.get(row).copied().unwrap_or(false) {
            continue;
        }
        if accumulate_alpha_gradients_for_row(
            graph,
            input,
            beta_state,
            alpha_state,
            intermediate,
            relu_pre_boxes,
            spec_matrix.row(row),
            row,
        )
        .is_some()
        {
            accumulated_rows += 1;
        }
    }
    if accumulated_rows == 0 {
        return None;
    }
    let row_norm = 1.0 / accumulated_rows as f32;
    for neuron_map in alpha_state.neurons_mut().values_mut() {
        for neuron in neuron_map.values_mut() {
            neuron.scale_grad(row_norm);
        }
    }

    let mut max_grad = 0.0f32;
    for neuron_map in alpha_state.neurons_mut().values_mut() {
        for neuron in neuron_map.values_mut() {
            let grad = neuron.grad();
            if !grad.is_finite() {
                continue;
            }
            max_grad = max_grad.max(grad.abs());
            neuron.set_alpha(neuron.alpha() + lr * grad);
        }
    }
    Some(max_grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    use crate::beta_crown::config::BetaCrownConfig;
    use crate::beta_crown::domain::GraphCrownContext;
    use crate::beta_crown::engine::BetaCrownVerifier;
    use crate::beta_crown::GraphSplitHistory;
    use crate::{GraphNode, Layer, LinearLayer, ReLULayer};

    /// Tiny 2-layer FC fixture: 2 inputs → 3 unstable hidden ReLUs → 1 output.
    /// All hidden boxes cross zero with `u > -l` (heuristic α = 1), and the
    /// stored deltas are chosen so every composed α is strictly inside (0, 1)
    /// — no clamp kinks near the finite-difference probes.
    fn fixture() -> (GraphNetwork, BoundedTensor) {
        let w1 = ndarray::arr2(&[[1.0f32, 0.4], [-0.5, 1.0], [0.6, -0.3]]);
        let b1 = arr1(&[0.05f32, 0.1, 0.15]);
        let w2 = ndarray::arr2(&[[1.0f32, -1.5, 0.75]]);
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "linear1",
            Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
        ));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["linear1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(LinearLayer::new(w2, None).unwrap()),
            vec!["relu1".to_string()],
        ));
        graph.set_output("linear2");
        let input = BoundedTensor::new(
            arr1(&[-1.0f32, -1.0]).into_dyn(),
            arr1(&[2.0f32, 2.0]).into_dyn(),
        )
        .unwrap();
        (graph, input)
    }

    fn delta_alpha_state(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        history: &GraphSplitHistory,
        stored: f32,
    ) -> GraphDomainAlphaState {
        let ibp = graph.collect_node_bounds(input).unwrap();
        let node_bounds: HashMap<String, Arc<BoundedTensor>> =
            ibp.into_iter().map(|(k, v)| (k, Arc::new(v))).collect();
        let mut alpha =
            GraphDomainAlphaState::from_graph_bounds(graph, &node_bounds, history, input);
        // Mirror the relu_split_route production configuration: delta-form
        // storage around the current-box heuristic.
        alpha.split_refresh = true;
        for map in [&mut alpha.neurons, &mut alpha.upper_neurons] {
            for node in map.values_mut() {
                for neuron in node.values_mut() {
                    neuron.set_alpha(stored);
                }
            }
        }
        alpha
    }

    /// Row-0 lower bound of the spec-guided pass under the given α state.
    fn lower_bound_for(
        verifier: &BetaCrownVerifier,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        history: &GraphSplitHistory,
        beta_state: &GraphBetaState,
        alpha_state: &GraphDomainAlphaState,
        spec_matrix: &ndarray::Array2<f32>,
    ) -> (
        f32,
        HashMap<String, Arc<BoundedTensor>>,
        GraphAlphaCrownIntermediate,
    ) {
        let context = GraphCrownContext::for_history(history).with_alpha(alpha_state);
        let (output, node_bounds, intermediate) = verifier
            .propagate_crown_with_graph_beta_and_spec_matrix_storing_intermediates(
                graph,
                input,
                &context,
                beta_state,
                spec_matrix,
            )
            .unwrap();
        let flat = output.flatten();
        (flat.lower()[[0]], node_bounds, intermediate)
    }

    /// The analytical per-parameter α gradient must match a central finite
    /// difference of the ACTUAL spec-guided lower bound to 1e-2 relative
    /// tolerance on the 2-layer FC fixture.
    #[test]
    fn analytical_alpha_gradient_matches_finite_difference_of_actual_bound() {
        let (graph, input) = fixture();
        let history = GraphSplitHistory::new();
        let beta_state = GraphBetaState::from_history(&history).unwrap();
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let spec_matrix = ndarray::arr2(&[[1.0f32]]);
        const STORED: f32 = 0.3; // composed α = heuristic 1.0 + (0.3 - 0.5) = 0.8

        let mut alpha_state = delta_alpha_state(&graph, &input, &history, STORED);
        let (_, node_bounds, intermediate) = lower_bound_for(
            &verifier,
            &graph,
            &input,
            &history,
            &beta_state,
            &alpha_state,
            &spec_matrix,
        );
        let relu_pre_boxes = collect_relu_pre_boxes(&graph, &node_bounds);
        assert!(
            relu_pre_boxes.contains_key("relu1"),
            "fixture must expose the ReLU pre-activation box"
        );

        accumulate_alpha_gradients_for_row(
            &graph,
            &input,
            &beta_state,
            &mut alpha_state,
            &intermediate,
            &relu_pre_boxes,
            spec_matrix.row(0),
            0,
        )
        .expect("sequential FC chain must be supported");

        // W2 = [1.0, -1.5, 0.75]: neurons 0 and 2 take the lower (α) branch
        // and must carry non-zero gradients; neuron 1's backward coefficient
        // is negative (upper chord) so its α gradient is exactly zero.
        let grads: Vec<f32> = (0..3)
            .map(|j| alpha_state.neuron("relu1", j).unwrap().grad())
            .collect();
        assert!(
            grads[0] != 0.0 && grads[2] != 0.0,
            "positive-coefficient neurons must have non-zero α gradients: {grads:?}"
        );
        assert_eq!(
            grads[1], 0.0,
            "negative-coefficient neuron must have zero α gradient: {grads:?}"
        );

        const H: f32 = 1e-3;
        for j in 0..3 {
            let mut plus = delta_alpha_state(&graph, &input, &history, STORED);
            plus.neuron_mut("relu1", j).unwrap().set_alpha(STORED + H);
            let (lb_plus, _, _) = lower_bound_for(
                &verifier,
                &graph,
                &input,
                &history,
                &beta_state,
                &plus,
                &spec_matrix,
            );
            let mut minus = delta_alpha_state(&graph, &input, &history, STORED);
            minus.neuron_mut("relu1", j).unwrap().set_alpha(STORED - H);
            let (lb_minus, _, _) = lower_bound_for(
                &verifier,
                &graph,
                &input,
                &history,
                &beta_state,
                &minus,
                &spec_matrix,
            );
            let fd = (lb_plus - lb_minus) / (2.0 * H);
            let analytical = grads[j];
            if fd.abs() < 1e-4 && analytical.abs() < 1e-4 {
                continue; // both ~zero: agreement
            }
            let rel = (analytical - fd).abs() / fd.abs().max(analytical.abs());
            assert!(
                rel <= 1e-2,
                "neuron {j}: analytical grad {analytical} vs finite difference {fd} \
                 (relative error {rel} > 1e-2)"
            );
        }
    }

    /// The round helper must step the stored deltas in the +LR·grad direction
    /// (clamped by `set_alpha`) and report the max |grad|.
    #[test]
    fn analytical_alpha_round_steps_stored_deltas() {
        let (graph, input) = fixture();
        let history = GraphSplitHistory::new();
        let beta_state = GraphBetaState::from_history(&history).unwrap();
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let spec_matrix = ndarray::arr2(&[[1.0f32]]);
        const STORED: f32 = 0.3;

        let mut alpha_state = delta_alpha_state(&graph, &input, &history, STORED);
        let (lb, node_bounds, intermediate) = lower_bound_for(
            &verifier,
            &graph,
            &input,
            &history,
            &beta_state,
            &alpha_state,
            &spec_matrix,
        );
        let relu_pre_boxes = collect_relu_pre_boxes(&graph, &node_bounds);
        let obj_bounds = vec![(lb, f32::INFINITY)];

        let max_grad = relu_split_alpha_analytical_round(
            &graph,
            &input,
            &beta_state,
            &mut alpha_state,
            &intermediate,
            &relu_pre_boxes,
            &spec_matrix,
            &obj_bounds,
            &[0.0],
            &[false],
            false,
            0.05,
        )
        .expect("sequential FC chain must be supported");
        assert!(max_grad > 0.0, "fixture must produce a non-zero max grad");

        for j in 0..3 {
            let neuron = alpha_state.neuron("relu1", j).unwrap();
            let expected = (STORED + 0.05 * neuron.grad()).clamp(0.0, 1.0);
            assert!(
                (neuron.alpha() - expected).abs() < 1e-6,
                "neuron {j}: stored delta {} != expected {expected}",
                neuron.alpha()
            );
        }
    }
}
