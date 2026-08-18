// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-row spec matrix support for constrained backward CROWN (#4306).
//!
//! Extracted from `mod.rs` to keep the constraints module within the 500-line limit.

use std::sync::Arc;

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use crate::batched_domain::CachedLinearBounds;
use crate::beta_crown::domain::GraphCrownContext;
use crate::beta_crown::engine::graph::DomainCrownResultWithIntermediates;
use crate::beta_crown::state::GraphBetaState;
use crate::GraphNetwork;

use super::super::super::super::BetaCrownVerifier;
use super::backward::{BackwardMode, BackwardParams};
use super::ensure_constrained_propagation_deadline;
use super::lookups::build_constraint_lookups;
use super::patches::ConstrainedPatchesPolicy;

impl BetaCrownVerifier {
    /// Constrained backward CROWN with a multi-row spec matrix seed (#4306).
    ///
    /// Identical to `propagate_crown_with_graph_constraints_with_cache` but
    /// seeds the backward pass with an (N, D) spec matrix instead of a single
    /// objective row. This allows one backward pass to compute bounds for all
    /// N objectives simultaneously.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn propagate_crown_with_graph_constraints_with_spec_matrix(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: Option<&GraphBetaState>,
        spec_matrix: &ndarray::Array2<f32>,
        seed_cache: Option<&CachedLinearBounds>,
        capture_linear_bounds: bool,
    ) -> Result<(
        BoundedTensor,
        std::collections::HashMap<String, Arc<BoundedTensor>>,
        Option<CachedLinearBounds>,
    )> {
        let deadline = self.effective_graph_bab_deadline();
        ensure_constrained_propagation_deadline(
            deadline,
            "before spec-matrix constrained forward preparation",
        )?;
        let (mut bounds_cache, constrained_input, exec_order) =
            self.prepare_constrained_graph_bounds(graph, input, context, beta_state, None)?;
        ensure_constrained_propagation_deadline(deadline, "before spec-matrix Complete Clip")?;
        self.maybe_apply_complete_clip_root_bank(
            graph,
            context,
            beta_state,
            None,
            Some(spec_matrix),
            &constrained_input,
            &exec_order,
            &mut bounds_cache,
        );
        ensure_constrained_propagation_deadline(
            deadline,
            "after spec-matrix Complete Clip and before constrained backward preparation",
        )?;

        let params = BackwardParams {
            graph,
            constrained_input: &constrained_input,
            exec_order: &exec_order,
            context,
            beta_state,
            objective: None,
            spec_matrix: Some(spec_matrix),
            seed_cache,
            capture_linear_bounds,
            deadline,
            patches_policy: ConstrainedPatchesPolicy::for_engine(context.engine),
        };
        let result =
            self.backward_crown_constrained(&params, &mut bounds_cache, BackwardMode::Standard)?;

        Ok((result.output_bounds, bounds_cache, result.captured_la))
    }

    /// Constrained backward CROWN with intermediate capture for batched spec rows.
    ///
    /// Mirrors `propagate_crown_with_graph_constraints_storing_intermediates` but
    /// seeds the backward pass with a dense `(num_specs, output_dim)` spec matrix
    /// so analytical beta optimization can evaluate all objectives in one pass.
    pub(in crate::beta_crown::engine::graph) fn propagate_crown_with_graph_constraints_storing_intermediates_with_spec_matrix(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: Option<&GraphBetaState>,
        spec_matrix: &ndarray::Array2<f32>,
    ) -> Result<DomainCrownResultWithIntermediates> {
        let deadline = self.effective_graph_bab_deadline();
        ensure_constrained_propagation_deadline(
            deadline,
            "before spec-matrix constrained forward preparation",
        )?;
        let (mut bounds_cache, constrained_input, exec_order) =
            self.prepare_constrained_graph_bounds(graph, input, context, beta_state, None)?;
        ensure_constrained_propagation_deadline(deadline, "before spec-matrix Complete Clip")?;
        self.maybe_apply_complete_clip_root_bank(
            graph,
            context,
            beta_state,
            None,
            Some(spec_matrix),
            &constrained_input,
            &exec_order,
            &mut bounds_cache,
        );
        ensure_constrained_propagation_deadline(
            deadline,
            "after spec-matrix Complete Clip and before constrained backward preparation",
        )?;

        let lookups = build_constraint_lookups(
            &context.history.constraints,
            &context.history.genbab_constraints,
            graph,
        )?;

        ensure_constrained_propagation_deadline(
            deadline,
            "before spec-matrix constrained backward dispatch",
        )?;
        let params = BackwardParams {
            graph,
            constrained_input: &constrained_input,
            exec_order: &exec_order,
            context,
            beta_state,
            objective: None,
            spec_matrix: Some(spec_matrix),
            seed_cache: None,
            capture_linear_bounds: false,
            deadline,
            patches_policy: ConstrainedPatchesPolicy::for_engine(context.engine),
        };
        let result = self.backward_crown_constrained(
            &params,
            &mut bounds_cache,
            BackwardMode::StoringIntermediates {
                lookups: Box::new(lookups),
            },
        )?;

        let intermediate = result.intermediate.ok_or_else(|| {
            NyError::InternalError(
                "StoringIntermediates mode did not produce intermediates".to_string(),
            )
        })?;

        Ok((result.output_bounds, bounds_cache, intermediate))
    }
}
