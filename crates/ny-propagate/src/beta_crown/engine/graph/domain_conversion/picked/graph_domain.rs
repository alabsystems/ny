// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::Axis;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use crate::batched_domain::PickedDomains;
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::state::GraphBetaState;
use crate::beta_crown::GraphBabDomain;
use crate::GraphNetwork;

use super::super::history::history_from_constraints;
use super::shared::alpha_state_from_metadata_or_graph;

/// Extract a single GraphBabDomain from a PickedDomains batch.
///
/// This is used when processing batches from DomainList to reconstruct the full
/// domain representation needed for CROWN propagation and branching.
///
/// When `graph` is provided, initializes alpha state from the domain's node bounds
/// and split history using the standard heuristic (alpha = 1 if u > -l, else 0). This
/// enables the batched backward pass to use per-neuron alpha values instead of the
/// fixed heuristic in `propagate_linear_with_bounds`.
///
/// # Arguments
/// * `idx` - Index within the batch to extract
/// * `picked` - The picked batch from DomainList
/// * `layer_names` - Ordered list of layer names
/// * `verify_upper` - Whether verifying upper bound (affects priority calculation)
/// * `graph` - Optional graph for alpha state initialization. Issue: #1841
pub fn graph_domain_from_picked(
    idx: usize,
    picked: &PickedDomains,
    layer_names: &[String],
    verify_upper: bool,
    graph: Option<&GraphNetwork>,
) -> Result<GraphBabDomain> {
    let metadata = picked.metadata.get(idx).ok_or_else(|| {
        NyError::InvalidSpec(format!("picked metadata missing entry for idx {idx}"))
    })?;

    let input_batch = picked.input_lowers.shape().first().copied().unwrap_or(0);
    if idx >= input_batch {
        return Err(NyError::InvalidSpec(format!(
            "graph_domain_from_picked: idx {idx} >= input batch size {input_batch}"
        )));
    }

    let mut node_bounds: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
    for name in layer_names {
        let lowers = picked.layer_lowers.get(name).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "missing layer lower bounds for '{name}' in picked batch"
            ))
        })?;
        let uppers = picked.layer_uppers.get(name).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "missing layer upper bounds for '{name}' in picked batch"
            ))
        })?;

        let layer_batch = lowers.shape().first().copied().unwrap_or(0);
        if idx >= layer_batch {
            return Err(NyError::InvalidSpec(format!(
                "graph_domain_from_picked: idx {idx} >= layer '{name}' batch size {layer_batch}"
            )));
        }
        let lower = lowers.index_axis(Axis(0), idx).to_owned().into_dyn();
        let upper = uppers.index_axis(Axis(0), idx).to_owned().into_dyn();
        let bounds = BoundedTensor::new(lower, upper)?;
        node_bounds.insert(name.clone(), Arc::new(bounds));
    }

    let input_lower = picked
        .input_lowers
        .index_axis(Axis(0), idx)
        .to_owned()
        .into_dyn();
    let input_upper = picked
        .input_uppers
        .index_axis(Axis(0), idx)
        .to_owned()
        .into_dyn();
    let input_bounds = Arc::new(BoundedTensor::new(input_lower, input_upper)?);

    let history = history_from_constraints(&metadata.constraints)?;
    let beta_state = GraphBetaState::from_history(&history)?;
    let priority = BetaCrownConfig::domain_priority_for_mode(
        verify_upper,
        metadata.lower_bound,
        metadata.upper_bound,
    )?;
    let alpha_state =
        alpha_state_from_metadata_or_graph(metadata, graph, &node_bounds, &history, &input_bounds)?;

    GraphBabDomain::from_metadata(
        history,
        node_bounds,
        metadata.lower_bound,
        metadata.upper_bound,
        metadata.depth,
        priority,
        input_bounds,
        beta_state,
        alpha_state,
        metadata.cached_la.clone(),
    )
}
