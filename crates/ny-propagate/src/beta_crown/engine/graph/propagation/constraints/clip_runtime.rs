// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared graph clip-in-alpha preparation for constrained propagation.
//!
//! Computes forward linear bounds (input-relative) via topological accumulation,
//! then delegates to `clip_interm_domain_full` for Lagrangian-dual tightening.

use std::sync::Arc;

use ny_core::Result;
use ny_tensor::BoundedTensor;

use crate::beta_crown::domain::GraphCrownContext;
use crate::beta_crown::engine::graph::clip_alpha::{
    apply_graph_clip_in_alpha_crown, compute_forward_linear_bounds,
};
use crate::beta_crown::state::GraphBetaState;
use crate::GraphNetwork;

use super::super::super::super::BetaCrownVerifier;

impl BetaCrownVerifier {
    pub(in crate::beta_crown::engine::graph::propagation) fn prepare_constrained_graph_bounds(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: Option<&GraphBetaState>,
        objective: Option<&[f32]>,
    ) -> Result<(
        std::collections::HashMap<String, Arc<BoundedTensor>>,
        BoundedTensor,
        Vec<String>,
    )> {
        let (mut bounds_cache, constrained_input) = self.compute_constrained_forward_bounds(
            graph,
            input,
            context.history,
            context.base_bounds,
            context.delta_seeds, // #cone-delta: dark, NY_CONE_REFRESH-gated
        )?;
        let exec_order = graph.exec_order()?;

        self.maybe_apply_clip_in_alpha_crown(
            graph,
            &constrained_input,
            context,
            beta_state,
            objective,
            exec_order,
            &mut bounds_cache,
        )?;

        Ok((bounds_cache, constrained_input, exec_order.to_vec()))
    }

    #[allow(clippy::too_many_arguments)]
    fn maybe_apply_clip_in_alpha_crown(
        &self,
        graph: &GraphNetwork,
        constrained_input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        _beta_state: Option<&GraphBetaState>,
        _objective: Option<&[f32]>,
        exec_order: &[String],
        bounds_cache: &mut std::collections::HashMap<String, Arc<BoundedTensor>>,
    ) -> Result<()> {
        if !self.config.clip_in_alpha_crown || context.history.depth() == 0 {
            return Ok(());
        }

        // Compute forward linear bounds: each node expressed as a linear function
        // of the network input. This is the correct direction for clip_interm_domain_full
        // which uses these bounds to build Lagrangian constraints from split decisions.
        let forward_bounds = compute_forward_linear_bounds(
            graph,
            context.history,
            exec_order,
            bounds_cache,
            constrained_input,
        )?;

        apply_graph_clip_in_alpha_crown(
            graph,
            context.history,
            exec_order,
            bounds_cache,
            constrained_input,
            &forward_bounds,
            self.config.clip_interm_topk,
        )
    }
}
