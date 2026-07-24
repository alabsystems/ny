// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::core::GraphBetaState;
use super::entry::GraphBetaEntry;

impl GraphBetaState {
    fn entry_linear(&self, node_name: &str, neuron_idx: usize) -> Option<&GraphBetaEntry> {
        self.entries
            .iter()
            .find(|e| e.node_name == node_name && e.neuron_idx == neuron_idx)
    }

    /// β entry for a neuron, if constrained.
    pub fn entry(&self, node_name: &str, neuron_idx: usize) -> Option<&GraphBetaEntry> {
        if let Some(indices) = self.matching_indices(node_name, neuron_idx) {
            return indices.first().and_then(|&idx| self.entries.get(idx));
        }
        self.entry_linear(node_name, neuron_idx)
    }

    /// β entry for a specific constraint, matching split point and sign.
    pub fn entry_for_constraint(
        &self,
        node_name: &str,
        neuron_idx: usize,
        split_point: f32,
        sign: f32,
    ) -> Option<&GraphBetaEntry> {
        const EPS: f32 = 1e-6;
        if let Some(indices) = self.matching_indices(node_name, neuron_idx) {
            return indices.iter().find_map(|&idx| {
                let entry = &self.entries[idx];
                ((entry.split_point - split_point).abs() < EPS && (entry.sign - sign).abs() < EPS)
                    .then_some(entry)
            });
        }
        self.entries.iter().find(|e| {
            e.node_name == node_name
                && e.neuron_idx == neuron_idx
                && (e.split_point - split_point).abs() < EPS
                && (e.sign - sign).abs() < EPS
        })
    }

    /// Mutable β entry for a neuron, if constrained.
    pub fn entry_mut(&mut self, node_name: &str, neuron_idx: usize) -> Option<&mut GraphBetaEntry> {
        if !self.lookup_index_fresh() {
            self.rebuild_lookup_index();
        }
        if let Some(idx) = self
            .neuron_index
            .get(node_name)
            .and_then(|by_neuron| by_neuron.get(&neuron_idx))
            .and_then(|indices| indices.first())
            .copied()
        {
            return self.entries.get_mut(idx);
        }
        self.entries
            .iter_mut()
            .find(|e| e.node_name == node_name && e.neuron_idx == neuron_idx)
    }

    /// Signed β value for a neuron: β * sign.
    /// Returns None if neuron is not constrained.
    pub fn signed_beta(&self, node_name: &str, neuron_idx: usize) -> Option<f32> {
        if let Some(indices) = self.matching_indices(node_name, neuron_idx) {
            let total: f32 = indices
                .iter()
                .map(|&idx| self.entries[idx].signed_value())
                .sum();
            return (!indices.is_empty()).then_some(total);
        }
        let mut found = false;
        let mut total = 0.0;
        for entry in &self.entries {
            if entry.node_name == node_name && entry.neuron_idx == neuron_idx {
                found = true;
                total += entry.signed_value();
            }
        }
        found.then_some(total)
    }

    /// Check if any β entries exist for a given node. O(1) via index.
    pub fn has_node_entries(&self, node_name: &str) -> bool {
        if self.lookup_index_fresh() {
            return self
                .neuron_index
                .get(node_name)
                .is_some_and(|m| !m.is_empty());
        }
        self.entries.iter().any(|e| e.node_name == node_name)
    }

    /// Iterate over all entry indices for a given node.
    ///
    /// When the lookup index is fresh, this flattens the indexed entries for the
    /// node and restores original insertion order in O(k log k), where k is the
    /// number of constrained entries for the node. This still avoids the old O(B)
    /// full-state scan across all beta entries.
    pub fn entries_for_node(&self, node_name: &str) -> impl Iterator<Item = &GraphBetaEntry> {
        NodeEntryIter::new(self, node_name)
    }
}

/// Iterator over [`GraphBetaEntry`] references for a specific node.
///
/// When lookup indexes are fresh, uses `neuron_index` plus an index sort to
/// preserve original insertion order in O(k log k), where k is the number of
/// entries for the node. Falls back to linear scan otherwise.
///
/// Part of #2936: eliminates O(R*B) scans in backward CROWN passes.
enum NodeEntryIter<'a> {
    Indexed {
        /// Flattened entry indices for all neurons of the target node.
        indices: Vec<usize>,
        pos: usize,
        entries: &'a [GraphBetaEntry],
    },
    Linear {
        entries: std::slice::Iter<'a, GraphBetaEntry>,
        node_name: String,
    },
}

impl<'a> NodeEntryIter<'a> {
    fn new(state: &'a GraphBetaState, node_name: &str) -> Self {
        if state.lookup_index_fresh() {
            if let Some(by_neuron) = state.neuron_index.get(node_name) {
                let mut indices: Vec<usize> =
                    by_neuron.values().flat_map(|v| v.iter().copied()).collect();
                indices.sort_unstable();
                return Self::Indexed {
                    indices,
                    pos: 0,
                    entries: &state.entries,
                };
            }
            return Self::Indexed {
                indices: Vec::new(),
                pos: 0,
                entries: &state.entries,
            };
        }
        Self::Linear {
            entries: state.entries.iter(),
            node_name: node_name.to_string(),
        }
    }
}

impl<'a> Iterator for NodeEntryIter<'a> {
    type Item = &'a GraphBetaEntry;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Indexed {
                indices,
                pos,
                entries,
            } => {
                if *pos < indices.len() {
                    let idx = indices[*pos];
                    *pos += 1;
                    Some(&entries[idx])
                } else {
                    None
                }
            }
            Self::Linear { entries, node_name } => entries.find(|e| e.node_name == *node_name),
        }
    }
}
