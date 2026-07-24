// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use super::entry::GraphBetaEntry;

/// β parameters for constrained GraphNetwork CROWN propagation.
///
/// β values are Lagrangian multipliers in the dual formulation of split constraints.
/// The Lagrangian augmented bound is: lb = c^T * A * x + b + sum_i(β_i * sign_i * x_i)
///
/// This is the GraphNetwork equivalent of BetaState, using node_name instead of layer_idx.
#[derive(Debug, Clone, Default)]
pub struct GraphBetaState {
    /// Sparse beta entries for constrained neurons.
    pub(crate) entries: Vec<GraphBetaEntry>,
    /// Fast `(node_name, neuron_idx)` grouping for hot graph β-state lookups.
    pub(crate) neuron_index: HashMap<String, HashMap<usize, Vec<usize>>>,
    /// Number of entries covered by the lookup index.
    pub(crate) indexed_entries: usize,
}

impl GraphBetaState {
    /// Default initial β value for Lagrangian relaxation.
    ///
    /// Must be 0.0 so that β-CROWN bounds equal standard CROWN bounds when
    /// `beta_iterations = 0` (no optimization). A non-zero default widens bounds
    /// because the Lagrangian term `±β` is added to both lower_a and upper_a
    /// without compensation. The sequential `BetaState` uses `value: 0.0` for
    /// the same reason.
    ///
    /// When `beta_iterations > 0`, the optimizer will move β away from zero
    /// toward values that tighten bounds.
    ///
    /// Reference: α,β-CROWN paper (Xu et al. 2021), Section 3.2: β initialized
    /// at 0 and optimized via projected gradient ascent.
    ///
    /// Fix for #1817: Previously 0.1, which caused GPU BaB to produce wider
    /// bounds than unconstrained CROWN when beta_iterations=0 (default).
    pub const DEFAULT_BETA_INIT: f32 = 0.0;

    pub(crate) fn from_entries(entries: Vec<GraphBetaEntry>) -> Self {
        let mut state = Self {
            entries,
            ..Self::default()
        };
        state.rebuild_lookup_index();
        state
    }

    pub(super) fn rebuild_lookup_index(&mut self) {
        self.neuron_index.clear();
        for (idx, entry) in self.entries.iter().enumerate() {
            self.neuron_index
                .entry(entry.node_name.clone())
                .or_default()
                .entry(entry.neuron_idx)
                .or_default()
                .push(idx);
        }
        self.indexed_entries = self.entries.len();
    }

    pub(super) fn lookup_index_fresh(&self) -> bool {
        self.indexed_entries == self.entries.len()
    }

    pub(super) fn matching_indices(&self, node_name: &str, neuron_idx: usize) -> Option<&[usize]> {
        if !self.lookup_index_fresh() {
            return None;
        }
        self.neuron_index
            .get(node_name)
            .and_then(|by_neuron| by_neuron.get(&neuron_idx).map(Vec::as_slice))
    }

    /// Create empty beta state.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Check if the beta state is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
