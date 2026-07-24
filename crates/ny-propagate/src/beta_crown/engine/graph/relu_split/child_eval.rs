// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Child-domain evaluation helpers for ReLU-split branch-and-bound.

use std::collections::HashMap;
use std::sync::Arc;

use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;

use crate::beta_crown::bab_cuts::GraphCutPool;
use crate::beta_crown::branching::GraphNeuronConstraint;
use crate::beta_crown::domain::GraphBabDomain;
use crate::GraphNetwork;

use super::super::super::BetaCrownVerifier;

/// Classification for a single child domain after constraint application and bounding.
pub(super) enum ChildOutcome {
    /// Child has valid bounds and a verification classification.
    Evaluated(Box<GraphBabDomain>, bool),
    /// Child domain is infeasible and can be pruned.
    Infeasible,
    /// Child could not be bounded and leaves the parent partially unresolved.
    Failed,
    /// Applying the split produced no child domain.
    NoChild,
}

impl BetaCrownVerifier {
    /// Evaluate a child whose constraints have already been applied.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate_existing_child(
        &self,
        graph: &GraphNetwork,
        mut child: GraphBabDomain,
        parent_node_bounds: &HashMap<String, Arc<BoundedTensor>>,
        objective: &[f32],
        threshold: f32,
        cut_pool: Option<&GraphCutPool>,
        engine: Option<&dyn GemmEngine>,
    ) -> ChildOutcome {
        match self.evaluate_graph_child_bounds(
            graph,
            &mut child,
            parent_node_bounds,
            objective,
            cut_pool,
            engine,
        ) {
            Ok(true) => {
                let verified =
                    self.config
                        .domain_is_verified(child.lower_bound, child.upper_bound, threshold);
                ChildOutcome::Evaluated(Box::new(child), verified)
            }
            Err(ref e) if e.is_infeasible_domain() => ChildOutcome::Infeasible,
            Ok(false) | Err(_) => ChildOutcome::Failed,
        }
    }

    /// Apply a split constraint to a domain, then evaluate the resulting child.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate_and_classify_child(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        constraint: GraphNeuronConstraint,
        objective: &[f32],
        threshold: f32,
        cut_pool: Option<&GraphCutPool>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<ChildOutcome> {
        let Some(child) =
            domain.with_constraint(graph, constraint, self.config.verify_upper_bound)?
        else {
            return Ok(ChildOutcome::NoChild);
        };

        Ok(self.evaluate_existing_child(
            graph,
            child,
            &domain.node_bounds,
            objective,
            threshold,
            cut_pool,
            engine,
        ))
    }
}
