// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Vec-indexed storage for batched CROWN backward bounds.
//!
//! Replaces `HashMap<String, BatchedCrownBounds>` + `input_accumulated: bool`
//! with O(1) Vec-indexed access using the pre-compiled [`CrownDispatchPlan`].
//!
//! Design: `designs/2026-03-21-issue-4258-crown-dispatch-plan-compiled.md` Phase 3.
//! Part of #4297.

use std::collections::HashMap;
use std::time::Instant;

use ny_core::{NyError, Result};
use tracing::error;

use crate::bounds::patches_batched::BatchedCrownBounds;
use crate::bounds::BatchedLinearBounds;

use super::dispatch_plan::CrownDispatchPlan;

/// Vec-backed indexed storage for `BatchedCrownBounds` in the batched CROWN
/// backward loop.
///
/// All node-level bounds are stored in a `Vec<Option<BatchedCrownBounds>>`
/// indexed by the compact sequential indices from [`CrownDispatchPlan`].
/// Name-based access (needed by binary ops) uses an internal `name_to_idx` map.
///
/// The `NETWORK_INPUT` sentinel occupies the final slot in `storage`.
pub(crate) struct BatchedCrownAccumulator {
    storage: Vec<Option<BatchedCrownBounds>>,
    name_to_idx: HashMap<String, usize>,
    deadline: Option<Instant>,
}

impl BatchedCrownAccumulator {
    /// Build a new accumulator from a dispatch plan.
    ///
    /// Allocates `node_count + 1` slots (nodes + NETWORK_INPUT sentinel).
    #[cfg(test)]
    pub(crate) fn new(plan: &CrownDispatchPlan) -> Self {
        Self::new_with_deadline(plan, None)
    }

    /// Build an accumulator whose merge materializations inherit the caller's
    /// one absolute deadline.
    pub(crate) fn new_with_deadline(plan: &CrownDispatchPlan, deadline: Option<Instant>) -> Self {
        let capacity = plan.node_count() + 1; // +1 for NETWORK_INPUT
        Self {
            storage: (0..capacity).map(|_| None).collect(),
            name_to_idx: plan.name_to_idx.clone(),
            deadline,
        }
    }

    /// Insert bounds by name (used for initial output node and binary ops).
    #[inline]
    pub(crate) fn insert(&mut self, name: &str, bounds: BatchedCrownBounds) {
        if let Some(&idx) = self.name_to_idx.get(name) {
            debug_assert!(
                self.storage[idx].is_none(),
                "duplicate BatchedCrownAccumulator insert for key {name}",
            );
            self.storage[idx] = Some(bounds);
        } else {
            debug_assert!(
                false,
                "BatchedCrownAccumulator insert for unknown key {name}",
            );
        }
    }

    /// Take bounds by index (used in hot backward loop).
    #[inline]
    pub(crate) fn take_idx(&mut self, idx: usize) -> Option<BatchedCrownBounds> {
        self.storage[idx].take()
    }

    /// Take bounds by name.
    #[inline]
    pub(crate) fn take(&mut self, name: &str) -> Option<BatchedCrownBounds> {
        self.name_to_idx
            .get(name)
            .and_then(|&idx| self.storage[idx].take())
    }

    /// Check if a name has bounds stored.
    #[cfg(test)]
    #[inline]
    pub(crate) fn contains_key(&self, name: &str) -> bool {
        self.name_to_idx
            .get(name)
            .is_some_and(|&idx| self.storage[idx].is_some())
    }

    /// Check if any bounds are stored (excluding the current node being processed).
    ///
    /// Used by the partial CROWN fallback guard: if other paths have already
    /// accumulated, partial CROWN at this node would be unsound.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.storage.iter().all(|s| s.is_none())
    }

    /// Sum of memory bytes across all live bounds (for Patches diagnostics).
    pub(crate) fn total_memory_bytes(&self) -> usize {
        self.storage
            .iter()
            .filter_map(|s| s.as_ref())
            .map(|b| b.memory_bytes())
            .sum()
    }

    /// Accumulate `BatchedCrownBounds` for a named input.
    ///
    /// On first insertion, preserves the variant (Patches stays Patches).
    /// On merge (multiple paths converge), converts both to Dense and
    /// accumulates via `safe_add`.
    ///
    /// Replaces `GraphNetwork::accumulate_batched_crown_bounds_to_input`.
    pub(crate) fn accumulate(
        &mut self,
        input_name: &str,
        new_bounds: BatchedCrownBounds,
    ) -> Result<()> {
        if self.deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Err(NyError::DeadlineExceeded(
                "BatchedCrownAccumulator: deadline exceeded before accumulation".into(),
            ));
        }
        let idx = self.name_to_idx.get(input_name).copied().ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "BatchedCrownAccumulator: unknown input '{}'",
                input_name
            ))
        })?;

        if self.storage[idx].is_none() {
            // First insertion: preserve variant (Patches stays Patches).
            self.storage[idx] = Some(new_bounds);
        } else {
            // Merge point: convert new to Dense, then safe_add (#3550: checked).
            let new_blb = new_bounds.into_batched_dense_checked_with_deadline(
                "batched_accumulator:accumulate:new",
                self.deadline,
            )?;
            if let Some(ref mut existing_bcb) = self.storage[idx] {
                existing_bcb.merge_dense_checked_with_deadline(
                    new_blb,
                    "batched_accumulator:accumulate:existing",
                    self.deadline,
                )?;
            } else {
                error!(
                    "BatchedCrownAccumulator: merge expected but {} missing — bounds dropped",
                    input_name
                );
                debug_assert!(false, "BatchedCrownBounds entry missing during merge");
            }
        }
        Ok(())
    }

    /// Convenience: accumulate Dense `BatchedLinearBounds` for a named input.
    ///
    /// Wraps as `BatchedCrownBounds::Dense` and delegates to [`accumulate`].
    ///
    /// Replaces `GraphNetwork::accumulate_dense_batched_bounds_to_input`.
    pub(crate) fn accumulate_dense(
        &mut self,
        input_name: &str,
        new_bounds: impl Into<BatchedLinearBounds>,
    ) -> Result<()> {
        self.accumulate(input_name, BatchedCrownBounds::Dense(new_bounds.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bounds::BatchedLinearBounds;
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use crate::network::core::graph::{GraphNetwork, GraphNode};

    fn make_simple_graph_and_plan() -> (GraphNetwork, CrownDispatchPlan) {
        let mut g = GraphNetwork::new();
        g.try_add_node(GraphNode::from_input(
            "linear1",
            Layer::Linear(
                LinearLayer::new(
                    ndarray::Array2::zeros((4, 3)),
                    Some(ndarray::Array1::zeros(4)),
                )
                .unwrap(),
            ),
        ))
        .unwrap();
        g.try_add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["linear1".to_string()],
        ))
        .unwrap();
        g.set_output("relu1");
        let plan = CrownDispatchPlan::build(&g).unwrap();
        (g, plan)
    }

    #[test]
    fn test_batched_accumulator_insert_and_take_by_idx() {
        let (_g, plan) = make_simple_graph_and_plan();
        let mut acc = BatchedCrownAccumulator::new(&plan);
        let output_idx = plan.output_node_idx;

        assert!(acc.is_empty());
        let bounds = BatchedCrownBounds::Dense(BatchedLinearBounds::identity(&[4]).unwrap());
        acc.insert(plan.name_of(output_idx), bounds);
        assert!(!acc.is_empty());

        let taken = acc.take_idx(output_idx);
        assert!(taken.is_some());
        assert!(acc.take_idx(output_idx).is_none());
    }

    #[test]
    fn test_batched_accumulator_accumulate_first_preserves() {
        let (_g, plan) = make_simple_graph_and_plan();
        let mut acc = BatchedCrownAccumulator::new(&plan);
        let bounds = BatchedCrownBounds::Dense(BatchedLinearBounds::identity(&[4]).unwrap());

        acc.accumulate("linear1", bounds).unwrap();
        assert!(acc.contains_key("linear1"));
    }

    #[test]
    fn test_batched_accumulator_network_input_tracking() {
        let (_g, plan) = make_simple_graph_and_plan();
        let mut acc = BatchedCrownAccumulator::new(&plan);

        assert!(!acc.contains_key(super::super::NETWORK_INPUT));
        let bounds = BatchedCrownBounds::Dense(BatchedLinearBounds::identity(&[3]).unwrap());
        acc.accumulate(super::super::NETWORK_INPUT, bounds).unwrap();
        assert!(acc.contains_key(super::super::NETWORK_INPUT));
    }
}
