// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::graph_history::GraphSplitHistory;
use super::split::{LayerRef, NeuronSplit};
use ny_core::Result;

/// Split history for GenBaB domains using NeuronSplit.
///
/// This is the GenBaB equivalent of `GraphSplitHistory`, supporting arbitrary
/// branching points instead of just ReLU's binary split at 0.
#[derive(Debug, Clone, Default)]
pub struct GeneralSplitHistory {
    /// All splits applied in this domain.
    pub(crate) splits: Vec<NeuronSplit>,
}

impl GeneralSplitHistory {
    /// Create empty split history.
    pub fn new() -> Self {
        Self { splits: Vec::new() }
    }

    /// Read-only access to the split list.
    #[inline]
    pub fn splits(&self) -> &[NeuronSplit] {
        &self.splits
    }

    /// Add a split to the history.
    pub fn add_split(&mut self, split: NeuronSplit) {
        self.splits.push(split);
    }

    /// Get the depth (number of splits).
    pub fn depth(&self) -> usize {
        self.splits.len()
    }

    /// Create a new history with an additional split.
    pub fn with_split(&self, split: NeuronSplit) -> Self {
        let mut new = self.clone();
        new.add_split(split);
        new
    }

    /// Convert from GraphSplitHistory (for backward compatibility).
    ///
    /// # Errors
    /// Returns `NyError::NumericalInstability` if any constraint has non-finite
    /// score or split point values.
    pub fn from_graph_history(history: &GraphSplitHistory) -> Result<Self> {
        let mut splits: Vec<NeuronSplit> = history
            .constraints
            .iter()
            .map(NeuronSplit::from_graph_constraint)
            .collect::<Result<Vec<_>>>()?;

        let relu_count = splits.len();
        let mut last_genbab_split_id: Option<usize> = None;
        for (idx, constraint) in history.genbab_constraints.iter().enumerate() {
            let (lower, upper) = if constraint.is_upper_branch {
                (Some(constraint.split_point), None)
            } else {
                (None, Some(constraint.split_point))
            };

            let split_id = history.genbab_split_ids.get(idx).copied();
            if splits.len() > relu_count && split_id.is_some() && split_id == last_genbab_split_id {
                if let Some(last) = splits.last_mut() {
                    let matches_neuron = matches!(&last.layer, LayerRef::Name(name) if name == &constraint.node_name)
                        && last.neuron_idx == constraint.neuron_idx;
                    if matches_neuron {
                        let can_merge = (lower.is_some() && last.lower_bound.is_none())
                            || (upper.is_some() && last.upper_bound.is_none());
                        if can_merge {
                            if let Some(bound) = lower {
                                last.lower_bound = Some(bound);
                            }
                            if let Some(bound) = upper {
                                last.upper_bound = Some(bound);
                            }
                            if constraint.score > last.score {
                                last.score = constraint.score;
                            }
                            continue;
                        }
                    }
                }
            }

            splits.push(NeuronSplit::new(
                LayerRef::Name(constraint.node_name.clone()),
                constraint.neuron_idx,
                lower,
                upper,
                constraint.score,
            )?);
            last_genbab_split_id = split_id;
        }

        Ok(Self { splits })
    }

    /// The effective bounds for a neuron after applying all splits.
    ///
    /// Returns (lower_bound, upper_bound) where None means use domain's original bound.
    pub fn neuron_bounds(&self, layer: &LayerRef, neuron_idx: usize) -> (Option<f32>, Option<f32>) {
        let mut lower = None;
        let mut upper = None;

        for split in &self.splits {
            if &split.layer == layer && split.neuron_idx == neuron_idx {
                if split.lower_bound.is_some() {
                    lower = split.lower_bound;
                }
                if split.upper_bound.is_some() {
                    upper = split.upper_bound;
                }
            }
        }

        (lower, upper)
    }
}
