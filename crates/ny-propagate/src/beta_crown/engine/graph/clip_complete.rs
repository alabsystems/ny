// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph hidden-layer constrained tightening for complete clipping.
//!
//! Applies spec-derived constraints to tighten graph node bounds via the
//! Lagrangian dual LP solver. This is the graph equivalent of
//! `complete_clip_intermediate.rs` (sequential networks).
//!
//! Mirrors `apply_graph_clip_in_alpha_crown` from `clip_alpha.rs` with two
//! differences:
//! 1. Constraints come from CROWN output spec bounds, not split history
//! 2. Only unstable neurons are selected (via `selection_ratio`), not all
//!
//! ## References
//!
//! - `auto_LiRPA/concretize_bounds.py:concretize_bounds` — two-pass approach
//! - `designs/2026-03-17-issue-3552-complete-clipping-semantics-execution-packet.md` Packet 4

use std::collections::HashMap;

use ndarray::{Array1, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{BoundedTensor, RepairStrategy};
use tracing::{debug, trace};

use crate::batched_domain::CachedLinearBounds;
use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::clip_interm_domain::{
    merge_bounds, sort_out_constraints, tighten_with_constraints, PreprocessedConstraints,
    SplitConstraints,
};
use crate::cmp_utils::nan_last_descending_cmp;
use crate::GraphNetwork;
use crate::LinearBounds;

use super::clip_alpha::compute_forward_linear_bounds;

type BatchLinearParts = (
    ndarray::Array2<f32>,
    Array1<f32>,
    ndarray::Array2<f32>,
    Array1<f32>,
);

/// Build child-local node bounds for graph complete clipping.
///
/// Uses the clipped child input box together with the spec linear bounds that
/// produced that clip step. The spec rows remain sound on the clipped child
/// because the child box is a subset of the original domain the CROWN rows were
/// derived from.
pub(in crate::beta_crown::engine) fn build_graph_complete_clip_node_bounds(
    graph: &GraphNetwork,
    constrained_input: &BoundedTensor,
    linear_bounds: &LinearBounds,
    thresholds: &[f32],
    verify_upper: bool,
    selection_ratio: f32,
    engine: Option<&dyn ny_core::GemmEngine>,
) -> Result<Option<HashMap<String, BoundedTensor>>> {
    if thresholds.is_empty() {
        return Ok(None);
    }

    let mut bounds_cache = graph.collect_node_bounds_with_engine(constrained_input, engine)?;
    let exec_order = graph.exec_order()?;
    let forward_linear_bounds = compute_forward_linear_bounds(
        graph,
        &GraphSplitHistory::new(),
        exec_order,
        &bounds_cache,
        constrained_input,
    )?;

    // The spec rows may still carry their certified coefficient-error envelope;
    // fold it into the bias over the clipped child box before the raw
    // coefficients become constraint rows. The constraints are only ever
    // applied over this box, so the fold stays sound here, and an undischarged
    // envelope could otherwise cut off inputs the true coefficients admit.
    let folded_bounds;
    let linear_bounds = if linear_bounds.has_coeff_err() {
        let mut folded = linear_bounds.clone();
        BetaCrownVerifier::discharge_coeff_err_for_clip(&mut folded, constrained_input);
        folded_bounds = folded;
        &folded_bounds
    } else {
        linear_bounds
    };

    let spec_constraints =
        build_graph_complete_spec_constraints(linear_bounds, thresholds, verify_upper)?;

    let (input_lower, input_upper) =
        constrained_input.flatten_to_ix1("graph_complete_clip_build_input")?;
    let preprocessed = sort_out_constraints(&spec_constraints, &input_lower, &input_upper)?;
    if preprocessed.a_active.nrows() == 0 {
        return Ok(None);
    }

    apply_graph_complete_clip_constraints(
        &mut bounds_cache,
        exec_order,
        constrained_input,
        &forward_linear_bounds,
        &preprocessed,
        selection_ratio,
    )?;

    Ok(Some(bounds_cache))
}

fn build_graph_complete_spec_constraints(
    linear_bounds: &LinearBounds,
    thresholds: &[f32],
    verify_upper: bool,
) -> Result<SplitConstraints> {
    let (a_matrix, b_vector) = if verify_upper {
        (
            linear_bounds.upper_a().mapv(|value| -value),
            linear_bounds.upper_b().mapv(|value| -value),
        )
    } else {
        (
            linear_bounds.lower_a().clone(),
            linear_bounds.lower_b().clone(),
        )
    };

    if a_matrix.nrows() != thresholds.len() {
        return Err(NyError::shape_mismatch(
            vec![a_matrix.nrows()],
            vec![thresholds.len()],
        ));
    }

    let threshold_values = Array1::from_iter(thresholds.iter().copied().map(|threshold| {
        if verify_upper {
            -threshold
        } else {
            threshold
        }
    }));

    Ok(SplitConstraints {
        a_matrix,
        b_vector: &b_vector - &threshold_values,
        num_constraints: thresholds.len(),
    })
}

/// Tighten graph node bounds using spec-derived constraints.
///
/// For each node in `exec_order` that has forward linear bounds and concrete
/// bounds in the cache, selects unstable neurons based on `selection_ratio`,
/// then tightens them via constrained concretization using the Lagrangian dual
/// LP solver.
///
/// This is the graph variant of `tighten_intermediate_with_spec_constraints`
/// from `complete_clip_intermediate.rs`.
///
/// # Arguments
///
/// * `bounds_cache` - Per-node concrete bounds, updated in-place
/// * `exec_order` - Topological execution order of graph nodes
/// * `constrained_input` - Clipped input bounds
/// * `forward_linear_bounds` - Input-relative linear bounds per node
/// * `spec_constraints` - Preprocessed spec-derived constraints
/// * `selection_ratio` - Neuron selection ratio (<0 = all unstable)
///
/// Reference: `auto_LiRPA/concretize_bounds.py:concretize_bounds` (two-pass)
pub(in crate::beta_crown::engine::graph) fn apply_graph_complete_clip_constraints(
    bounds_cache: &mut HashMap<String, BoundedTensor>,
    exec_order: &[String],
    constrained_input: &BoundedTensor,
    forward_linear_bounds: &CachedLinearBounds,
    spec_constraints: &PreprocessedConstraints,
    selection_ratio: f32,
) -> Result<()> {
    if spec_constraints.a_active.nrows() == 0 {
        trace!("graph_complete_clip: no active constraints, skipping");
        return Ok(());
    }

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
        constrained_input.flatten_to_ix1("graph_complete_clip_apply_input")?;

    for node_name in &node_names {
        let Some(fwd_lb) = forward_linear_bounds.linear_bounds(node_name) else {
            continue;
        };
        let Some(old_bounds) = bounds_cache.get(*node_name) else {
            continue;
        };

        let (old_lower, old_upper) = old_bounds.flatten_to_ix1("graph_complete_clip_node")?;

        // Select unstable neurons using ratio-based selection
        let selected = select_graph_neurons_by_uncertainty(&old_lower, &old_upper, selection_ratio);
        if selected.is_empty() {
            continue;
        }

        // Extract forward linear bounds for selected neurons
        let Some((obj_lower_a, obj_lower_b, obj_upper_a, obj_upper_b)) =
            selected_neuron_linear_bounds(&fwd_lb, &selected)
        else {
            continue;
        };

        // Apply constrained tightening
        let (tightened_lower, tightened_upper) = match tighten_with_constraints(
            spec_constraints,
            &obj_lower_a,
            &obj_lower_b,
            &obj_upper_a,
            &obj_upper_b,
            &input_lower,
            &input_upper,
        ) {
            Ok(result) => result,
            Err(e) => {
                debug!(
                    "graph_complete_clip: tighten failed for '{}': {}",
                    node_name, e
                );
                continue;
            }
        };

        // Merge tightened bounds back
        let (merged_lower, merged_upper) = merge_bounds(
            &old_lower,
            &old_upper,
            &tightened_lower,
            &tightened_upper,
            &selected,
        );

        // Check if anything changed
        if bounds_changed(
            old_lower.iter().copied(),
            old_upper.iter().copied(),
            merged_lower.iter().copied(),
            merged_upper.iter().copied(),
        ) {
            let shape = old_bounds.shape().to_vec();
            let lower_arr = merged_lower
                .into_shape_clone(IxDyn(&shape))
                .map_err(|err| {
                    NyError::InternalError(format!(
                        "graph_complete_clip: reshape lower failed for '{}': {}",
                        node_name, err
                    ))
                })?;
            let upper_arr = merged_upper
                .into_shape_clone(IxDyn(&shape))
                .map_err(|err| {
                    NyError::InternalError(format!(
                        "graph_complete_clip: reshape upper failed for '{}': {}",
                        node_name, err
                    ))
                })?;
            // Use Widen repair instead of strict `new`: a tightened intermediate
            // bound may be non-finite (e.g. ±Inf from a CROWN backward through a
            // degenerate BatchNorm channel with var+eps ~= 0, or NaN that escaped
            // an upstream firewall). Widen replaces NaN with the conservative
            // direction (-inf for lower, +inf for upper) and keeps ±Inf as-is,
            // then repairs any inversion by widening. A ±Inf intermediate bound is
            // sound (the neuron is treated as unbounded), so BaB can still split or
            // time out instead of aborting the whole graph beta-CROWN attempt.
            bounds_cache.insert(
                (*node_name).to_string(),
                BoundedTensor::new_repaired(lower_arr, upper_arr, RepairStrategy::Widen)?,
            );
            trace!(
                "graph_complete_clip: tightened {} neurons at '{}'",
                selected.len(),
                node_name
            );
        }
    }

    Ok(())
}

/// Select unstable neurons for a graph node using ratio-based selection.
///
/// Same contract as `select_neurons_by_uncertainty` in
/// `complete_clip_intermediate.rs`:
/// - `ratio < 0`: all unstable neurons
/// - `ratio in [0, 1]`: ceil(unstable_count * ratio), clamped to >= 1
fn select_graph_neurons_by_uncertainty(
    lower: &Array1<f32>,
    upper: &Array1<f32>,
    selection_ratio: f32,
) -> Vec<usize> {
    let mut unstable: Vec<(usize, f32)> = lower
        .iter()
        .zip(upper.iter())
        .enumerate()
        .filter_map(|(i, (&l, &u))| {
            if l < 0.0 && u > 0.0 {
                let gap = u - l;
                if gap.is_finite() {
                    Some((i, gap))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    if unstable.is_empty() {
        return vec![];
    }

    let budget = if selection_ratio < 0.0 {
        unstable.len()
    } else {
        let k = (unstable.len() as f32 * selection_ratio).ceil() as usize;
        k.max(1).min(unstable.len())
    };

    // Sort descending by gap (largest uncertainty first).
    // NaN-safe: nan_last_descending_cmp sorts NaN last (#4288).
    // Defense-in-depth: the is_finite() filter above excludes NaN gaps,
    // but the safe comparator prevents silent corruption if the filter changes.
    unstable.sort_by(|a, b| nan_last_descending_cmp(&a.1, &b.1));

    unstable.iter().take(budget).map(|(idx, _)| *idx).collect()
}

/// Extract forward linear bounds for selected neuron indices.
///
/// Returns `(lower_A, lower_b, upper_A, upper_b)` matrices/vectors for the
/// selected neurons. Same semantics as `selected_neuron_linear_bounds` in
/// `clip_alpha.rs`.
fn selected_neuron_linear_bounds(
    linear_bounds: &LinearBounds,
    neuron_indices: &[usize],
) -> Option<BatchLinearParts> {
    let n_selected = neuron_indices.len();
    let n_inputs = linear_bounds.num_inputs();

    let mut lower_a = ndarray::Array2::zeros((n_selected, n_inputs));
    let mut lower_b = Array1::zeros(n_selected);
    let mut upper_a = ndarray::Array2::zeros((n_selected, n_inputs));
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

/// Check if any bounds changed (tightened or NaN appeared).
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
    use super::*;
    use ndarray::{arr1, arr2, array};

    use crate::{GraphNetwork, GraphNode, Layer, LinearLayer};

    fn two_stage_identity_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "hidden",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap()),
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap()),
            vec!["hidden".to_string()],
        ));
        graph.set_output("out");
        graph
    }

    #[test]
    fn test_select_graph_neurons_all_unstable_with_neg_ratio() {
        let lower = array![-1.0, -2.0, 0.5, -0.5];
        let upper = array![1.0, 2.0, 1.5, 0.5];
        // Unstable: idx 0 (gap=2), idx 1 (gap=4), idx 3 (gap=1)
        let selected = select_graph_neurons_by_uncertainty(&lower, &upper, -1.0);
        assert_eq!(selected.len(), 3); // All unstable
        assert_eq!(selected[0], 1); // Largest gap first
    }

    #[test]
    fn test_select_graph_neurons_half_ratio() {
        let lower = array![-1.0, -2.0, -3.0, -4.0];
        let upper = array![1.0, 2.0, 3.0, 4.0];
        // All unstable: gaps [2, 4, 6, 8]
        let selected = select_graph_neurons_by_uncertainty(&lower, &upper, 0.5);
        assert_eq!(selected.len(), 2); // ceil(4 * 0.5) = 2
        assert_eq!(selected[0], 3); // Gap=8, largest
        assert_eq!(selected[1], 2); // Gap=6
    }

    #[test]
    fn test_select_graph_neurons_no_unstable() {
        let lower = array![0.1, 0.2, 0.3];
        let upper = array![1.0, 2.0, 3.0];
        let selected = select_graph_neurons_by_uncertainty(&lower, &upper, -1.0);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_empty_constraints_skips() {
        let mut bounds_cache = HashMap::new();
        let input = BoundedTensor::new(array![0.0].into_dyn(), array![1.0].into_dyn()).unwrap();
        let fwd_lb = CachedLinearBounds::default();
        let preprocessed = PreprocessedConstraints {
            a_active: ndarray::Array2::zeros((0, 1)),
            b_active: Array1::zeros(0),
            d_active: Array1::zeros(0),
            infeasible_mask: vec![],
            fully_covered_mask: vec![],
        };

        let result = apply_graph_complete_clip_constraints(
            &mut bounds_cache,
            &[],
            &input,
            &fwd_lb,
            &preprocessed,
            -1.0,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_build_graph_complete_clip_node_bounds_tightens_hidden_cache() {
        let graph = two_stage_identity_graph();
        let input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
        let linear_bounds = LinearBounds {
            lower_a: arr2(&[[1.0_f32]]),
            lower_b: arr1(&[0.0_f32]),
            upper_a: arr2(&[[1.0_f32]]),
            upper_b: arr1(&[0.0_f32]),
            lower_a_err: None,
            upper_a_err: None,
        };

        let bounds_cache = build_graph_complete_clip_node_bounds(
            &graph,
            &input,
            &linear_bounds,
            &[0.2_f32],
            false,
            -1.0,
            None,
        )
        .unwrap()
        .expect("active spec constraint should produce child-local node bounds");

        let hidden = bounds_cache
            .get("hidden")
            .expect("hidden node bounds should be present")
            .flatten();
        assert!(
            hidden.upper()[[0]] <= 0.21,
            "spec-derived graph complete clipping should tighten hidden upper bound to the threshold region, got {}",
            hidden.upper()[[0]]
        );
        assert!(
            hidden.lower()[[0]] >= -1.01,
            "tightening should preserve the sound lower bound, got {}",
            hidden.lower()[[0]]
        );
    }
}
