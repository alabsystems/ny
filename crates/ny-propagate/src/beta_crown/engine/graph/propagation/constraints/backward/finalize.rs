// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Final concretization and cut contribution application for constrained backward CROWN.
//!
//! Part of #4293 (directory-module split from former backward.rs monolith).

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::IxDyn;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use crate::batched_domain::CachedLinearBounds;
use crate::NETWORK_INPUT;

use super::super::super::super::super::BetaCrownVerifier;
use super::{
    BackwardCrownResult, BackwardParams, ConstrainedBackwardSetup, ConstrainedBackwardState,
};

impl BetaCrownVerifier {
    pub(super) fn apply_graph_cut_contribution_if_needed(
        &self,
        params: &BackwardParams<'_>,
        bounds_cache_mut: &mut HashMap<String, Arc<BoundedTensor>>,
        mut output_bounds: BoundedTensor,
    ) -> Result<BoundedTensor> {
        // GCP-CROWN: Apply cutting plane contributions to lower bound.
        // Only when objective is set — broadcasting to multi-output is unsound (#2400).
        if let (true, Some(pool)) = (params.objective.is_some(), params.context.cut_pool) {
            if !pool.is_empty() && self.config.enable_cuts {
                let relevant_cuts = pool.relevant_cuts_for(params.context.history);
                if !relevant_cuts.is_empty() {
                    let cut_contribution = self.compute_graph_cut_contribution_arc(
                        params.graph,
                        &relevant_cuts,
                        bounds_cache_mut,
                        params.constrained_input,
                    );

                    if cut_contribution > 0.0 {
                        let flat = output_bounds.flatten();
                        let shape = output_bounds.shape().to_vec();
                        let mut lower = flat.lower().clone();
                        for i in 0..lower.len() {
                            lower[[i]] += cut_contribution;
                        }
                        let lower_arr = lower
                            .into_shape_clone(IxDyn(&shape))
                            .map_err(|e| NyError::InvalidSpec(format!("shape error: {e}")))?;
                        let upper_arr = flat
                            .upper()
                            .clone()
                            .into_shape_clone(IxDyn(&shape))
                            .map_err(|e| NyError::InvalidSpec(format!("shape error: {e}")))?;
                        output_bounds = BoundedTensor::new(lower_arr, upper_arr)?;
                        debug!(
                            "Applied {} graph cuts, contribution: {}",
                            relevant_cuts.len(),
                            cut_contribution
                        );
                    }
                }
            }
        }

        Ok(output_bounds)
    }

    pub(super) fn finalize_constrained_backward(
        &self,
        params: &BackwardParams<'_>,
        is_standard: bool,
        bounds_cache_mut: &mut HashMap<String, Arc<BoundedTensor>>,
        setup: ConstrainedBackwardSetup<'_, '_>,
    ) -> Result<BackwardCrownResult> {
        let ConstrainedBackwardSetup {
            output_node,
            output_shape,
            state,
            ..
        } = setup;
        let ConstrainedBackwardState {
            mut node_crown_bounds,
            mut intermediate,
            mut captured_linear_bounds,
            input_accumulated,
        } = state;

        let mut output_bounds = if input_accumulated {
            let final_cb = node_crown_bounds
                .take(NETWORK_INPUT)?
                .ok_or_else(|| NyError::InvalidSpec("No linear bounds at input".to_string()))?;
            let final_lb = final_cb.into_dense()?;
            if is_standard
                && tracing::enabled!(tracing::Level::DEBUG)
                && params.context.history.constraints.len() >= 12
            {
                let gap = (final_lb.upper_a() - final_lb.lower_a())
                    .mapv(f32::abs)
                    .sum();
                let b_gap = (final_lb.upper_b() - final_lb.lower_b())
                    .mapv(f32::abs)
                    .sum();
                if gap > 1e-6 || b_gap > 1e-6 {
                    debug!(
                        "[#1817] CROWN backward A-gap={:.6}, b-gap={:.6} constraints={}",
                        gap,
                        b_gap,
                        params.context.history.constraints.len()
                    );
                }
            }
            if let Some(intermediate) = intermediate.as_mut() {
                intermediate.final_bounds = final_lb.clone();
            }
            if let Some(linear_bounds_map) = captured_linear_bounds.as_mut() {
                linear_bounds_map.insert(NETWORK_INPUT.to_string(), final_lb.clone());
            }
            final_lb
                .concretize_sound(params.constrained_input)
                .reshape(&output_shape)?
        } else {
            if is_standard {
                debug!(
                    "[#1817] CROWN backward did NOT reach input, falling back to IBP output bounds"
                );
            }
            // #cone-delta increment 2 residual copy: this IBP fallback (backward
            // did not reach the input) must hand out an OWNED tensor. The Arc is
            // usually uniquely held here (the output node is always in the
            // recompute cone), so `unwrap_or_clone` is a move; a shared Arc
            // deep-clones once — identical values either way.
            Arc::unwrap_or_clone(bounds_cache_mut.remove(output_node).ok_or_else(|| {
                NyError::InvalidSpec(format!("Output node {} not found", output_node))
            })?)
        };

        let captured_la = if input_accumulated {
            captured_linear_bounds.map(CachedLinearBounds::from_linear_bounds_map)
        } else {
            None
        };
        output_bounds =
            self.apply_graph_cut_contribution_if_needed(params, bounds_cache_mut, output_bounds)?;

        Ok(BackwardCrownResult {
            output_bounds,
            intermediate,
            captured_la,
        })
    }
}
