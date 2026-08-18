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
    legacy_forward_affine_clipping_authorized,
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
        let (mut bounds_cache, constrained_input) = self
            .compute_constrained_forward_bounds_from_view(
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

    /// Apply the production Complete Clipping root bank to a freshly prepared
    /// one-domain constrained cache. Every sequential constrained backward
    /// variant calls this same seam so standard, intermediate-capture, scalar-
    /// objective, and spec-matrix evaluations cannot diverge.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn maybe_apply_complete_clip_root_bank(
        &self,
        graph: &GraphNetwork,
        context: &GraphCrownContext<'_>,
        beta_state: Option<&GraphBetaState>,
        objective: Option<&[f32]>,
        spec_matrix: Option<&ndarray::Array2<f32>>,
        constrained_input: &BoundedTensor,
        exec_order: &[String],
        bounds_cache: &mut std::collections::HashMap<String, Arc<BoundedTensor>>,
    ) {
        if !self.config.enable_clip_interm_domain {
            return;
        }
        let Some(engine) = context.engine else {
            return;
        };
        // The bounded facade intentionally exposes no broad GPU trait, while
        // Complete-Clip's sole publication path requires one. Avoid preparing
        // target weights/options for work that this engine can never consume.
        if engine.forbids_unbounded_cpu_fallback() {
            return;
        }
        let Some(output_node) = (if graph.output_name().is_empty() {
            exec_order.last().map(String::as_str)
        } else {
            Some(graph.output_name())
        }) else {
            return;
        };
        let Some(output_dim) = bounds_cache.get(output_node).map(|bounds| bounds.len()) else {
            return;
        };

        // Match constrained-backward seed precedence exactly:
        // multi-row spec > scalar objective > identity output seed.
        let owned_spec;
        let spec = if let Some(spec) = spec_matrix {
            if spec.nrows() == 0 || spec.ncols() != output_dim {
                return;
            }
            spec
        } else if let Some(objective) = objective {
            if objective.len() != output_dim {
                return;
            }
            let Ok(matrix) = ndarray::Array2::from_shape_vec((1, output_dim), objective.to_vec())
            else {
                return;
            };
            owned_spec = matrix;
            &owned_spec
        } else {
            // No exact seed was requested by this propagation call. Avoid
            // manufacturing an O(output_dim²) identity objective merely to
            // steer optional target selection.
            return;
        };

        let histories = [context.history];
        let beta_states = [beta_state];
        let alpha_states = [context.alpha_state];
        if let Some(mut outcome) = self.refine_last_relu_interm_bounds(
            graph,
            output_node,
            1,
            std::slice::from_ref(bounds_cache),
            std::slice::from_ref(constrained_input),
            &histories,
            &beta_states,
            &alpha_states,
            engine,
            spec,
        ) {
            if outcome.caches.len() == 1 {
                *bounds_cache = outcome.caches.remove(0);
            }
            // This scalar API has no infeasible bitmap. Discarding an
            // emptiness proof is conservative; the refined enclosure remains
            // sound and may merely do extra work.
        }
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
        if !self.config.clip_in_alpha_crown
            || context.history.depth() == 0
            || !legacy_forward_affine_clipping_authorized()
        {
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
