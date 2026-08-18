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

use crate::bounds::patches::PatchesMaterializationPurpose;
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

        let propagated_owned = {
            let node_lb = node_cb.ensure_dense_with_deadline_for_purpose(
                params.deadline,
                PatchesMaterializationPurpose::Other,
            )?;
            if is_standard
                && params.deadline.is_none()
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
            match linear
                .propagate_linear_with_engine_and_deadline(
                    node_lb,
                    params.context.engine,
                    params.deadline,
                )
                .map_err(|error| {
                    if error.is_deadline_exceeded() {
                        error
                    } else {
                        NyError::InvalidSpec(format!(
                            "Constrained CROWN failed at node '{}' (Linear): {}",
                            current.node_name, error
                        ))
                    }
                })? {
                std::borrow::Cow::Borrowed(_) => None,
                std::borrow::Cow::Owned(lb) => Some(lb),
            }
        };
        // A borrowed result is exactly the input relation. Move that carrier
        // into publication instead of deep-cloning four potentially large
        // arrays after the deadline-aware materialization transaction.
        let new_lb = match propagated_owned {
            None => node_cb.into_dense_with_deadline_for_purpose(
                params.deadline,
                PatchesMaterializationPurpose::Other,
            )?,
            Some(lb) => lb,
        };
        super::super::ensure_constrained_propagation_deadline(
            params.deadline,
            "after constrained linear propagation",
        )?;

        if is_standard
            && params.deadline.is_none()
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
        super::super::ensure_constrained_propagation_deadline(
            params.deadline,
            "before constrained linear publication",
        )?;

        params
            .graph
            .accumulate_dense_bounds_to_input_with_deadline(
                current.first_input,
                new_lb,
                &mut setup.state.node_crown_bounds,
                setup.output_dim,
                setup.input_dim,
                &mut setup.state.input_accumulated,
                params.deadline,
            )?;
        Ok(())
    }
}
