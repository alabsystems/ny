// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Linear layer override for constrained backward CROWN.
//!
//! Part of #4293 (directory-module split from former backward.rs monolith).

use std::collections::HashMap;
use std::sync::Arc;

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use crate::Layer;

use super::super::super::super::super::BetaCrownVerifier;
use super::super::patches::try_patches_step;
use super::dispatch::ConstrainedNodeContext;
use super::{resolve_pre_activation, BackwardParams, ConstrainedBackwardSetup};

impl BetaCrownVerifier {
    pub(super) fn process_linear_override(
        &self,
        params: &BackwardParams<'_>,
        current: &ConstrainedNodeContext<'_>,
        mut node_cb: crate::bounds::patches::CrownBounds,
        bounds_cache: &HashMap<String, Arc<BoundedTensor>>,
        setup: &mut ConstrainedBackwardSetup<'_, '_>,
        is_standard: bool,
    ) -> Result<()> {
        let Layer::Linear(linear) = &current.node.layer else {
            unreachable!("Constrained linear helper requires a Linear node");
        };
        let pre_activation =
            resolve_pre_activation(current.first_input, params.constrained_input, bounds_cache)?;

        if try_patches_step(
            params.graph,
            "Constrained CROWN",
            current.node_name,
            current.node,
            &mut node_cb,
            current.first_input,
            pre_activation,
            bounds_cache,
            &mut setup.state.node_crown_bounds,
            &mut setup.state.input_accumulated,
            params.context.engine,
            params.deadline,
            params.patches_policy,
        )? {
            return Ok(());
        }

        let new_lb = {
            let node_lb = node_cb.ensure_dense()?;
            if is_standard
                && tracing::enabled!(tracing::Level::DEBUG)
                && params.context.history.constraints.len() >= 12
            {
                let in_gap = (node_lb.upper_a() - node_lb.lower_a()).mapv(f32::abs).sum();
                if in_gap > 1e-6 {
                    debug!(
                        "[#1817 bwd] {} input A-gap={:.6}",
                        current.node_name, in_gap
                    );
                }
            }
            let new_lb = linear
                .propagate_linear_with_engine(node_lb, params.context.engine)
                .map_err(|error| {
                    NyError::InvalidSpec(format!(
                        "Constrained CROWN failed at node '{}' (Linear): {}",
                        current.node_name, error
                    ))
                })?;
            match new_lb {
                std::borrow::Cow::Borrowed(_) => node_lb.clone(),
                std::borrow::Cow::Owned(lb) => lb,
            }
        };

        if is_standard
            && tracing::enabled!(tracing::Level::DEBUG)
            && params.context.history.constraints.len() >= 12
        {
            let out_gap = (new_lb.upper_a() - new_lb.lower_a()).mapv(f32::abs).sum();
            if out_gap > 1e-6 {
                debug!(
                    "[#1817 bwd] {} output A-gap={:.6}",
                    current.node_name, out_gap
                );
            }
        }

        params.graph.accumulate_dense_bounds_to_input(
            current.first_input,
            new_lb,
            &mut setup.state.node_crown_bounds,
            setup.output_dim,
            setup.input_dim,
            &mut setup.state.input_accumulated,
        )?;
        Ok(())
    }
}
