// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{Axis, IxDyn};
use ny_core::{nan_propagating_max, nan_propagating_min, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::warn;

use crate::batched_domain::{DomainMetadata, PickedDomains};
use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::state::{GraphBetaState, GraphDomainAlphaState};
use crate::beta_crown::GraphBabDomain;
use crate::{GraphNetwork, Layer, NETWORK_INPUT};

use super::super::history::history_from_constraints;
use super::shared::array_element_at;

#[derive(Clone, Copy)]
enum ReluBranchDirection {
    Active,
    Inactive,
}

impl ReluBranchDirection {
    fn is_active(self) -> bool {
        matches!(self, Self::Active)
    }

    fn label(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Inactive => "inactive",
        }
    }
}

struct PickedReluBranchBase<'a> {
    idx: usize,
    metadata: &'a DomainMetadata,
    pre_name: &'a str,
    parent_history: GraphSplitHistory,
    parent_beta_state: GraphBetaState,
    parent_input_bounds: Arc<BoundedTensor>,
    priority: f32,
}

#[derive(Clone, Copy)]
struct PickedReluBranchSpec<'a> {
    node_name: &'a str,
    neuron_idx: usize,
    score: f32,
    direction: ReluBranchDirection,
}

fn read_branch_neuron_bounds(
    picked: &PickedDomains,
    idx: usize,
    pre_name: &str,
    branch_neuron_idx: usize,
) -> Result<Option<(f32, f32)>> {
    if pre_name == NETWORK_INPUT {
        let input_lower_view = picked.input_lowers.index_axis(Axis(0), idx);
        let input_upper_view = picked.input_uppers.index_axis(Axis(0), idx);
        let flat_len = input_lower_view.len();
        if branch_neuron_idx >= flat_len {
            return Ok(None);
        }
        let lower = array_element_at(
            &input_lower_view.into_dyn(),
            branch_neuron_idx,
            "branch_relu_from_picked: input lower",
        )?;
        let upper = array_element_at(
            &input_upper_view.into_dyn(),
            branch_neuron_idx,
            "branch_relu_from_picked: input upper",
        )?;
        Ok(Some((lower, upper)))
    } else {
        let pre_lowers = picked.layer_lowers.get(pre_name).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "branch_relu_from_picked: missing layer lower bounds for '{pre_name}'"
            ))
        })?;
        let pre_uppers = picked.layer_uppers.get(pre_name).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "branch_relu_from_picked: missing layer upper bounds for '{pre_name}'"
            ))
        })?;
        let lower_view = pre_lowers.index_axis(Axis(0), idx);
        let upper_view = pre_uppers.index_axis(Axis(0), idx);
        let flat_len = lower_view.len();
        if branch_neuron_idx >= flat_len {
            return Ok(None);
        }
        let lower = array_element_at(
            &lower_view.into_dyn(),
            branch_neuron_idx,
            "branch_relu_from_picked: layer lower",
        )?;
        let upper = array_element_at(
            &upper_view.into_dyn(),
            branch_neuron_idx,
            "branch_relu_from_picked: layer upper",
        )?;
        Ok(Some((lower, upper)))
    }
}

fn build_child_input_bounds(
    base: &PickedReluBranchBase<'_>,
    spec: PickedReluBranchSpec<'_>,
    had_propagation_failure: &mut bool,
) -> Result<Option<Arc<BoundedTensor>>> {
    if base.pre_name != NETWORK_INPUT {
        return Ok(Some(base.parent_input_bounds.clone()));
    }

    let shape = base.parent_input_bounds.shape().to_vec();
    let flat = base.parent_input_bounds.flatten();
    let mut lower_flat = flat.lower().clone();
    let mut upper_flat = flat.upper().clone();

    match spec.direction {
        ReluBranchDirection::Active => {
            lower_flat[[spec.neuron_idx]] = nan_propagating_max(lower_flat[[spec.neuron_idx]], 0.0);
        }
        ReluBranchDirection::Inactive => {
            upper_flat[[spec.neuron_idx]] = nan_propagating_min(upper_flat[[spec.neuron_idx]], 0.0);
        }
    }

    if lower_flat[[spec.neuron_idx]] > upper_flat[[spec.neuron_idx]] {
        return Ok(None);
    }

    let lower_new = match lower_flat.into_shape_clone(IxDyn(&shape)) {
        Ok(arr) => arr,
        Err(err) => {
            warn!(
                "branch_relu_from_picked: {} child lower reshape failed \
                 (idx={}, node={}, neuron={}, shape={:?}): {}",
                spec.direction.label(),
                base.idx,
                spec.node_name,
                spec.neuron_idx,
                shape,
                err
            );
            *had_propagation_failure = true;
            return Ok(None);
        }
    };
    let upper_new = match upper_flat.into_shape_clone(IxDyn(&shape)) {
        Ok(arr) => arr,
        Err(err) => {
            warn!(
                "branch_relu_from_picked: {} child upper reshape failed \
                 (idx={}, node={}, neuron={}, shape={:?}): {}",
                spec.direction.label(),
                base.idx,
                spec.node_name,
                spec.neuron_idx,
                shape,
                err
            );
            *had_propagation_failure = true;
            return Ok(None);
        }
    };
    match BoundedTensor::new(lower_new, upper_new) {
        Ok(bounds) => Ok(Some(Arc::new(bounds))),
        Err(err) => {
            warn!(
                "branch_relu_from_picked: {} child BoundedTensor::new failed \
                 (idx={}, node={}, neuron={}): {}",
                spec.direction.label(),
                base.idx,
                spec.node_name,
                spec.neuron_idx,
                err
            );
            *had_propagation_failure = true;
            Ok(None)
        }
    }
}

fn build_relu_child_from_picked(
    base: &PickedReluBranchBase<'_>,
    node_bounds: HashMap<String, Arc<BoundedTensor>>,
    graph: &GraphNetwork,
    spec: PickedReluBranchSpec<'_>,
    had_propagation_failure: &mut bool,
) -> Result<Option<GraphBabDomain>> {
    let constraint = GraphNeuronConstraint {
        node_name: spec.node_name.to_string(),
        neuron_idx: spec.neuron_idx,
        is_active: spec.direction.is_active(),
        score: spec.score,
    };
    let child_history = base.parent_history.with_constraint(constraint);
    let child_beta = GraphBetaState::from_history_with_warmup(
        &child_history,
        &base.parent_beta_state,
        GraphBetaState::DEFAULT_BETA_INIT,
    )?;

    let Some(input_bounds) = build_child_input_bounds(base, spec, had_propagation_failure)? else {
        return Ok(None);
    };

    let alpha_state = if let Some(parent_alpha) = base.metadata.require_runtime_alpha_state()? {
        GraphDomainAlphaState::from_parent(
            parent_alpha,
            graph,
            &node_bounds,
            &child_history,
            &input_bounds,
        )
    } else {
        GraphDomainAlphaState::from_graph_bounds(graph, &node_bounds, &child_history, &input_bounds)
    };

    GraphBabDomain::child_from_metadata(
        child_history,
        node_bounds,
        base.metadata.lower_bound,
        base.metadata.upper_bound,
        base.metadata.depth,
        base.priority,
        input_bounds,
        child_beta,
        alpha_state,
        base.metadata.cached_la.clone(),
    )
    .map(Some)
}

/// Branch a domain directly from batched `PickedDomains` for the ReLU GPU happy path,
/// without materializing an intermediate parent `GraphBabDomain`.
///
/// This is Direction 2 of #1668: instead of `graph_domain_from_picked` -> `with_constraint`,
/// this reads the single needed neuron value directly from the batched arrays for the
/// feasibility check, then materializes `node_bounds` only once and shares them between
/// both children via `Arc`. This eliminates:
/// - One full `graph_domain_from_picked` materialization for infeasible branches
/// - The HashMap clone in `with_constraint` (both children share the same HashMap)
/// - The intermediate parent `GraphBabDomain` allocation
///
/// # Arguments
/// * `idx` - Batch index within `PickedDomains`
/// * `picked` - The picked batch from DomainList
/// * `graph` - The graph network (for ReLU node -> pre-activation name lookup)
/// * `branch_node_name` - ReLU node to branch on
/// * `branch_neuron_idx` - Neuron index to branch on
/// * `branch_score` - Branching score
/// * `layer_names` - Ordered list of layer names
/// * `verify_upper` - Whether verifying upper bound
///
/// # Returns
/// `(Option<active_child>, Option<inactive_child>, propagation_failure)`:
/// either child is `None` if infeasible; `propagation_failure=true` means
/// branch child construction failed (for example NaN-contaminated input bounds).
///
/// # Reference
/// Issue: #1668 (zero-copy domain flow)
/// Design: `designs/2026-02-07-gpu-bab-zero-copy-domain-flow.md` Direction 2
// Justification: ReLU branching needs domain index, picked domains, graph, input,
// split info, context, config, beta state, and engine — all from BaB tree context.
#[allow(clippy::too_many_arguments)]
pub fn branch_relu_from_picked(
    idx: usize,
    picked: &PickedDomains,
    graph: &GraphNetwork,
    branch_node_name: &str,
    branch_neuron_idx: usize,
    branch_score: f32,
    layer_names: &[String],
    verify_upper: bool,
) -> Result<(Option<GraphBabDomain>, Option<GraphBabDomain>, bool)> {
    let metadata = picked.metadata.get(idx).ok_or_else(|| {
        NyError::InvalidSpec(format!("picked metadata missing entry for idx {idx}"))
    })?;

    let relu_node = graph.nodes.get(branch_node_name).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "branch_relu_from_picked: node '{branch_node_name}' not found in graph"
        ))
    })?;
    if !matches!(relu_node.layer, Layer::ReLU(_) | Layer::Sign(_)) {
        return Ok((None, None, false));
    }
    let pre_name = relu_node
        .inputs
        .first()
        .map(|name| name.as_str())
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ReLU node '{branch_node_name}' has no inputs — cannot determine pre-activation"
            ))
        })?;

    let Some((neuron_lower, neuron_upper)) =
        read_branch_neuron_bounds(picked, idx, pre_name, branch_neuron_idx)?
    else {
        return Ok((None, None, false));
    };

    if !neuron_lower.is_finite() || !neuron_upper.is_finite() {
        warn!(
            "branch_relu_from_picked: non-finite neuron bounds \
             (idx={}, node={}, neuron={}, lower={}, upper={}) — \
             treating as propagation failure",
            idx, branch_node_name, branch_neuron_idx, neuron_lower, neuron_upper
        );
        return Ok((None, None, true));
    }

    let active_feasible = neuron_upper >= 0.0;
    let inactive_feasible = neuron_lower <= 0.0;
    if !active_feasible && !inactive_feasible {
        return Ok((None, None, false));
    }

    let parent_history = history_from_constraints(&metadata.constraints)?;
    let parent_beta_state = GraphBetaState::from_history(&parent_history)?;
    let priority = BetaCrownConfig::domain_priority_for_mode(
        verify_upper,
        metadata.lower_bound,
        metadata.upper_bound,
    )?;

    let node_bounds: HashMap<String, Arc<BoundedTensor>> = {
        let mut bounds = HashMap::with_capacity(layer_names.len());
        for name in layer_names {
            let lowers = picked.layer_lowers.get(name.as_str()).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "branch_relu_from_picked: missing layer lower bounds for '{name}'"
                ))
            })?;
            let uppers = picked.layer_uppers.get(name.as_str()).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "branch_relu_from_picked: missing layer upper bounds for '{name}'"
                ))
            })?;
            let lower = lowers.index_axis(Axis(0), idx).to_owned().into_dyn();
            let upper = uppers.index_axis(Axis(0), idx).to_owned().into_dyn();
            let bounds_tensor = match BoundedTensor::new(lower, upper) {
                Ok(bt) => bt,
                Err(err) => {
                    warn!(
                        "branch_relu_from_picked: node bound materialization failed \
                         (idx={}, layer={}): {}",
                        idx, name, err
                    );
                    return Ok((None, None, true));
                }
            };
            bounds.insert(name.clone(), Arc::new(bounds_tensor));
        }
        bounds
    };

    let parent_input_lower = picked
        .input_lowers
        .index_axis(Axis(0), idx)
        .to_owned()
        .into_dyn();
    let parent_input_upper = picked
        .input_uppers
        .index_axis(Axis(0), idx)
        .to_owned()
        .into_dyn();
    let parent_input_bounds = match BoundedTensor::new(parent_input_lower, parent_input_upper) {
        Ok(bounds) => Arc::new(bounds),
        Err(err) => {
            warn!(
                "branch_relu_from_picked: parent input bound materialization failed \
                 (idx={}, node={}): {}",
                idx, branch_node_name, err
            );
            return Ok((None, None, true));
        }
    };

    let base = PickedReluBranchBase {
        idx,
        metadata,
        pre_name,
        parent_history,
        parent_beta_state,
        parent_input_bounds,
        priority,
    };
    let mut had_propagation_failure = false;

    let active_child = if active_feasible {
        let child = build_relu_child_from_picked(
            &base,
            node_bounds.clone(),
            graph,
            PickedReluBranchSpec {
                node_name: branch_node_name,
                neuron_idx: branch_neuron_idx,
                score: branch_score,
                direction: ReluBranchDirection::Active,
            },
            &mut had_propagation_failure,
        )?;
        if had_propagation_failure {
            return Ok((None, None, true));
        }
        child
    } else {
        None
    };

    let inactive_child = if inactive_feasible {
        let child = build_relu_child_from_picked(
            &base,
            node_bounds,
            graph,
            PickedReluBranchSpec {
                node_name: branch_node_name,
                neuron_idx: branch_neuron_idx,
                score: branch_score,
                direction: ReluBranchDirection::Inactive,
            },
            &mut had_propagation_failure,
        )?;
        if had_propagation_failure {
            return Ok((active_child, None, true));
        }
        child
    } else {
        None
    };

    Ok((active_child, inactive_child, had_propagation_failure))
}
