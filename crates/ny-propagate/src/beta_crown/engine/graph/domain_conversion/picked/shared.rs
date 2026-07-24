// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use crate::batched_domain::DomainMetadata;
use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::state::GraphDomainAlphaState;
use crate::GraphNetwork;

/// Read a single element from an ndarray view at the given flat index.
///
/// Tries `as_slice()` first for contiguous arrays (O(1) index), falls back to
/// `iter().nth()` for non-contiguous arrays. Returns `NyError::InternalError`
/// if the index is out of bounds — callers should bounds-check before calling.
pub(super) fn array_element_at(
    view: &ndarray::ArrayViewD<'_, f32>,
    idx: usize,
    context: &str,
) -> Result<f32> {
    if let Some(slice) = view.as_slice() {
        slice.get(idx).copied().ok_or_else(|| {
            NyError::InternalError(format!(
                "{context}: index {idx} out of bounds for len {}",
                slice.len()
            ))
        })
    } else {
        view.iter().nth(idx).copied().ok_or_else(|| {
            NyError::InternalError(format!(
                "{context}: nth({idx}) failed for len {}",
                view.len()
            ))
        })
    }
}

pub(super) fn alpha_state_from_metadata_or_graph(
    metadata: &DomainMetadata,
    graph: Option<&GraphNetwork>,
    node_bounds: &HashMap<String, Arc<BoundedTensor>>,
    history: &GraphSplitHistory,
    input_bounds: &Arc<BoundedTensor>,
) -> Result<GraphDomainAlphaState> {
    if let Some(persisted) = metadata.require_runtime_alpha_state()? {
        Ok(persisted.clone())
    } else {
        Ok(match graph {
            Some(graph) => {
                GraphDomainAlphaState::from_graph_bounds(graph, node_bounds, history, input_bounds)
            }
            None => GraphDomainAlphaState::empty(),
        })
    }
}
