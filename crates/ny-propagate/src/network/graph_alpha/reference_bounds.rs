// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Mutable reference-bound state for graph alpha-CROWN carry-forward.

use crate::layers::Layer;
use crate::network::core::{GraphNetwork, NETWORK_INPUT};
use ndarray::Zip;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::collections::{HashMap, HashSet};
use tracing::warn;

/// Graph-local reference bounds that carry tighter activation-input bounds
/// across DAG alpha-CROWN iterations.
///
/// This mirrors alpha-beta-CROWN's `best_intermediate_bounds` /
/// `reference_bounds` split: keep the immutable baseline for untargeted nodes,
/// monotonically tighten selected activation inputs element-wise, then promote
/// the tighter map into the next optimization iteration.
/// Source: `optimized_bounds.py:338-367,500-615`.
#[derive(Debug, Clone)]
pub(in crate::network::graph_alpha) struct GraphAlphaReferenceBounds {
    current: HashMap<String, BoundedTensor>,
    best: HashMap<String, BoundedTensor>,
    targets: Vec<String>,
}

impl GraphAlphaReferenceBounds {
    pub(in crate::network::graph_alpha) fn new(
        initial_bounds: HashMap<String, BoundedTensor>,
        targets: Vec<String>,
    ) -> Result<Self> {
        for target in &targets {
            if !initial_bounds.contains_key(target) {
                return Err(NyError::InvalidSpec(format!(
                    "graph alpha reference bounds missing target '{target}'"
                )));
            }
        }

        Ok(Self {
            current: initial_bounds.clone(),
            best: initial_bounds,
            targets,
        })
    }

    #[must_use]
    pub(in crate::network::graph_alpha) fn current(&self) -> &HashMap<String, BoundedTensor> {
        &self.current
    }

    /// Consume the optimizer-owned state and return its final sound reference
    /// map without cloning it.
    ///
    /// The DAG collection dispatcher uses this artifact to re-evaluate the
    /// returned alpha state against the exact map that initialized and survived
    /// the optimizer. This is especially important for default-uncached typed
    /// collectors, where recollecting would repeat the whole root transaction.
    #[must_use]
    pub(in crate::network::graph_alpha) fn into_current(self) -> HashMap<String, BoundedTensor> {
        self.current
    }

    #[must_use]
    pub(in crate::network::graph_alpha) fn targets(&self) -> &[String] {
        &self.targets
    }

    pub(in crate::network::graph_alpha) fn merge_candidate(
        &mut self,
        candidate: &HashMap<String, BoundedTensor>,
    ) -> Result<usize> {
        let mut tightened_targets = 0usize;

        for target in &self.targets {
            let best = self.best.get(target).ok_or_else(|| {
                NyError::InternalError(format!(
                    "graph alpha reference best map missing target '{target}'"
                ))
            })?;
            let candidate_bounds = candidate.get(target).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "graph alpha candidate map missing target '{target}'"
                ))
            })?;
            let (merged, tightened) = merge_tighter_bounds(target, best, candidate_bounds)?;
            if tightened {
                self.best.insert(target.clone(), merged);
                tightened_targets += 1;
            }
        }

        Ok(tightened_targets)
    }

    pub(in crate::network::graph_alpha) fn promote_best_to_current(&mut self) -> Result<()> {
        for target in &self.targets {
            let best = self.best.get(target).ok_or_else(|| {
                NyError::InternalError(format!(
                    "graph alpha reference best map missing target '{target}' during promote"
                ))
            })?;
            self.current.insert(target.clone(), best.clone());
        }
        Ok(())
    }

    #[cfg(test)]
    #[must_use]
    fn best(&self) -> &HashMap<String, BoundedTensor> {
        &self.best
    }
}

impl GraphNetwork {
    /// Collect activation-input nodes that should participate in graph alpha
    /// reference-bound carry-forward.
    ///
    /// Targets are the inputs to ReLU / Sigmoid / Tanh / Sqrt nodes, in
    /// topological order, with duplicates removed. The external network input
    /// is not optimized and therefore excluded.
    pub(in crate::network::graph_alpha) fn graph_alpha_reference_bound_targets(
        &self,
    ) -> Result<Vec<String>> {
        let exec_order = self.exec_order()?;
        let mut targets = Vec::new();
        let mut seen = HashSet::new();

        for node_name in exec_order {
            let node = self.node(node_name).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "graph alpha reference target node '{node_name}' missing from graph"
                ))
            })?;
            if !matches!(
                node.layer(),
                Layer::ReLU(_) | Layer::Sigmoid(_) | Layer::Tanh(_) | Layer::Sqrt(_)
            ) {
                continue;
            }

            let input_name = node.require_unary_input()?;
            if input_name == NETWORK_INPUT {
                continue;
            }

            if seen.insert(input_name.to_string()) {
                targets.push(input_name.to_string());
            }
        }

        Ok(targets)
    }
}

pub(crate) fn merge_reference_bound_maps(
    current: Option<&HashMap<String, BoundedTensor>>,
    candidate: Option<&HashMap<String, BoundedTensor>>,
) -> Result<Option<HashMap<String, BoundedTensor>>> {
    match (current, candidate) {
        (None, None) => Ok(None),
        (Some(bounds), None) | (None, Some(bounds)) => Ok(Some(bounds.clone())),
        (Some(current), Some(candidate)) => {
            let mut merged = current.clone();
            for (target, candidate_bounds) in candidate {
                if let Some(current_bounds) = merged.get(target) {
                    // Skip shape-mismatched nodes: DAG models can produce
                    // different shapes for the same node name when IBP and
                    // CROWN see pre- vs post-concat views. Keeping the
                    // current (wider) bounds is sound — the node just won't
                    // benefit from IBP tightening. (#4384)
                    if current_bounds.shape() != candidate_bounds.shape() {
                        tracing::debug!(
                            target = target.as_str(),
                            current_shape = ?current_bounds.shape(),
                            candidate_shape = ?candidate_bounds.shape(),
                            "skipping reference-bounds merge for shape-mismatched node"
                        );
                        continue;
                    }
                    let (tightened_bounds, _) =
                        merge_tighter_bounds(target, current_bounds, candidate_bounds)?;
                    merged.insert(target.clone(), tightened_bounds);
                } else {
                    merged.insert(target.clone(), candidate_bounds.clone());
                }
            }
            Ok(Some(merged))
        }
    }
}

fn merge_tighter_bounds(
    target: &str,
    current: &BoundedTensor,
    candidate: &BoundedTensor,
) -> Result<(BoundedTensor, bool)> {
    if current.shape() != candidate.shape() {
        return Err(NyError::shape_mismatch(
            current.shape().to_vec(),
            candidate.shape().to_vec(),
        ));
    }

    let mut lower = current.lower().clone();
    let mut upper = current.upper().clone();
    let mut tightened = false;
    let mut disjoint_count = 0usize;
    let mut nan_count = 0usize;

    Zip::from(lower.view_mut())
        .and(upper.view_mut())
        .and(current.lower())
        .and(current.upper())
        .and(candidate.lower())
        .and(candidate.upper())
        .for_each(
            |new_lower, new_upper, &cur_lower, &cur_upper, &cand_lower, &cand_upper| {
                // Defense-in-depth: NaN candidates are silently discarded by
                // f32::max/min (IEEE 754-2008 maxNum semantics), but we guard
                // explicitly and log for visibility (#3684).
                if cand_lower.is_nan() || cand_upper.is_nan() {
                    nan_count += 1;
                    *new_lower = cur_lower;
                    *new_upper = cur_upper;
                    return;
                }

                let merged_lower = cur_lower.max(cand_lower);
                let merged_upper = cur_upper.min(cand_upper);

                // Guard: disjoint intervals produce merged_lower > merged_upper.
                // Keep current bounds unchanged — they're still a valid
                // over-approximation. Adapts to BoundedTensor's strict
                // lower <= upper invariant (the Python reference proceeds
                // silently since PyTorch tensors have no validity check).
                // Source: #3684, optimized_bounds.py:356-361.
                if merged_lower > merged_upper {
                    disjoint_count += 1;
                    *new_lower = cur_lower;
                    *new_upper = cur_upper;
                    return;
                }

                if merged_lower > cur_lower || merged_upper < cur_upper {
                    tightened = true;
                }
                *new_lower = merged_lower;
                *new_upper = merged_upper;
            },
        );

    if disjoint_count > 0 {
        warn!(
            "reference-bound merge '{}': {} of {} elements had disjoint intervals, kept current",
            target,
            disjoint_count,
            lower.len()
        );
    }
    if nan_count > 0 {
        warn!(
            "reference-bound merge '{}': {} of {} elements had NaN candidates, kept current",
            target,
            nan_count,
            lower.len()
        );
    }

    let merged = BoundedTensor::new_allow_infinite(lower, upper).map_err(|e| {
        NyError::InvalidSpec(format!(
            "graph alpha reference-bound merge produced invalid interval for '{target}': {e}"
        ))
    })?;

    Ok((merged, tightened))
}

#[cfg(test)]
mod tests;
