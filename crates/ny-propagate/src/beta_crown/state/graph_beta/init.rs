// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::core::GraphBetaState;
use super::entry::GraphBetaEntry;
use crate::beta_crown::branching::GraphSplitHistory;
use ny_core::Result;

impl GraphBetaState {
    /// Create β state from GraphSplitHistory.
    ///
    /// # Errors
    /// Returns error if any constraint has a non-finite split point or invalid sign.
    pub fn from_history(history: &GraphSplitHistory) -> Result<Self> {
        Self::from_history_with_init(history, Self::DEFAULT_BETA_INIT)
    }

    /// Create β state from GraphSplitHistory with custom initial β value.
    ///
    /// Includes both ReLU and GenBaB constraints from the history.
    ///
    /// # Errors
    /// Returns error if any constraint has a non-finite split point or invalid sign.
    pub fn from_history_with_init(history: &GraphSplitHistory, init_beta: f32) -> Result<Self> {
        let entries = history
            .iter_all()
            .map(|c| {
                GraphBetaEntry::new(
                    c.node_name().to_string(),
                    c.neuron_idx(),
                    c.split_point(),
                    init_beta,
                    c.beta_sign(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::from_entries(entries))
    }

    /// Create β state from GraphSplitHistory with warmup from parent state.
    ///
    /// This inherits optimized β values from the parent domain for constraints
    /// that existed in the parent, while initializing the new constraint's β
    /// to the default value. This warmup is crucial for β-CROWN convergence
    /// and matches the α,β-CROWN behavior.
    ///
    /// For each constraint in the history (both ReLU and GenBaB):
    /// - If the parent had a β for this constraint, copy its value and Adam state (m, v, v_max)
    /// - Otherwise, initialize with the default value
    /// # Errors
    /// Returns error if any constraint has a non-finite split point or invalid sign.
    pub fn from_history_with_warmup(
        history: &GraphSplitHistory,
        parent_beta: &GraphBetaState,
        init_beta: f32,
    ) -> Result<Self> {
        let entries = history
            .iter_all()
            .map(|c| {
                let node_name = c.node_name();
                let neuron_idx = c.neuron_idx();
                let split_point = c.split_point();
                let sign = c.beta_sign();
                if let Some(parent_entry) =
                    parent_beta.entry_for_constraint(node_name, neuron_idx, split_point, sign)
                {
                    let mut entry = GraphBetaEntry::new(
                        node_name.to_string(),
                        neuron_idx,
                        split_point,
                        parent_entry.value,
                        sign,
                    )?;
                    entry.m = if parent_entry.m.is_finite() {
                        parent_entry.m
                    } else {
                        0.0
                    };
                    entry.v = if parent_entry.v.is_finite() {
                        parent_entry.v
                    } else {
                        0.0
                    };
                    entry.v_max = if parent_entry.v_max.is_finite() {
                        parent_entry.v_max
                    } else {
                        0.0
                    };
                    Ok(entry)
                } else {
                    GraphBetaEntry::new(
                        node_name.to_string(),
                        neuron_idx,
                        split_point,
                        init_beta,
                        sign,
                    )
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::from_entries(entries))
    }
}
