// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::graph_constraints::GraphNeuronConstraint;
use super::graph_history::GraphSplitHistory;
use ny_core::{NyError, Result};
use std::collections::HashMap;

/// A constraint on a single ReLU neuron.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeuronConstraint {
    /// Layer index of the ReLU.
    pub(crate) layer_idx: usize,
    /// Neuron index within the layer.
    pub(crate) neuron_idx: usize,
    /// True if neuron is constrained to be active (x >= 0), false if inactive (x <= 0).
    pub(crate) is_active: bool,
    /// Influence score for this split (used by BICCOS constraint strengthening).
    /// Larger values indicate more impactful splits.
    pub(crate) score: f32,
}

impl NeuronConstraint {
    /// Create a new neuron constraint with an influence score.
    ///
    /// # Errors
    /// Returns `NyError::NumericalInstability` if `score` is not finite.
    pub fn new(layer_idx: usize, neuron_idx: usize, is_active: bool, score: f32) -> Result<Self> {
        if !score.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "NeuronConstraint score must be finite, got {score} \
                 for layer {layer_idx}[{neuron_idx}]"
            )));
        }
        Ok(Self {
            layer_idx,
            neuron_idx,
            is_active,
            score,
        })
    }

    /// Returns the layer index.
    #[inline]
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Returns the neuron index within the layer.
    #[inline]
    pub fn neuron_idx(&self) -> usize {
        self.neuron_idx
    }

    /// Returns whether the neuron is constrained to be active (x >= 0).
    #[inline]
    pub fn is_active(&self) -> bool {
        self.is_active
    }

    /// Returns the influence score for this split.
    #[inline]
    pub fn score(&self) -> f32 {
        self.score
    }
}

/// History of split decisions in a domain.
#[derive(Debug, Clone, Default)]
pub struct SplitHistory {
    /// All constraints applied in this domain.
    pub constraints: Vec<NeuronConstraint>,
    /// O(1) lookup cache: (layer_idx, neuron_idx) -> is_active.
    /// Kept in sync with `constraints` via add_constraint().
    constraint_lookup: HashMap<(usize, usize), bool>,
}

impl SplitHistory {
    /// Create empty split history.
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            constraint_lookup: HashMap::new(),
        }
    }

    /// Add a constraint to the history.
    pub fn add_constraint(&mut self, constraint: NeuronConstraint) {
        self.constraint_lookup.insert(
            (constraint.layer_idx, constraint.neuron_idx),
            constraint.is_active,
        );
        self.constraints.push(constraint);
    }

    /// Get the depth (number of splits).
    pub fn depth(&self) -> usize {
        self.constraints.len()
    }

    /// Check if a neuron is already constrained.
    /// O(1) complexity via HashMap lookup (was O(n) linear search).
    pub fn is_constrained(&self, layer_idx: usize, neuron_idx: usize) -> Option<bool> {
        self.constraint_lookup
            .get(&(layer_idx, neuron_idx))
            .copied()
    }

    /// Create a new history with an additional constraint.
    pub fn with_constraint(&self, constraint: NeuronConstraint) -> Self {
        let mut new = self.clone();
        new.add_constraint(constraint);
        new
    }

    /// Convert to a `GraphSplitHistory` for use with clip_interm_domain.
    ///
    /// Sequential network layer indices are converted to node names using the
    /// format "layer_{layer_idx}" (e.g., layer 0 becomes "layer_0").
    ///
    /// # Returns
    ///
    /// A `GraphSplitHistory` with the same constraints, using synthetic node names.
    ///
    /// # Errors
    /// Returns `NyError::NumericalInstability` if any constraint has a non-finite score.
    pub fn to_graph_split_history(&self) -> Result<GraphSplitHistory> {
        let mut graph_history = GraphSplitHistory::new();
        for constraint in &self.constraints {
            graph_history.add_constraint(GraphNeuronConstraint::new(
                format!("layer_{}", constraint.layer_idx),
                constraint.neuron_idx,
                constraint.is_active,
                constraint.score,
            )?);
        }
        Ok(graph_history)
    }
}
