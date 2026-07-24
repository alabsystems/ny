// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Constrained-backward Patches helpers for #3813.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, info};

use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
use crate::network::crown_memory::{cpu_crown_dense_budget_bytes, DenseMaterializationEstimate};
use crate::network::GraphNode;
use crate::network::{crown_backward_step_patches, CrownStepResult};
use crate::{GraphNetwork, Layer, LinearBounds};

#[derive(Debug, Clone, Copy)]
pub(in crate::beta_crown::engine::graph::propagation) struct ConstrainedPatchesPolicy {
    pub allow_initial_patches: bool,
    pub allow_conv_reentry: bool,
}

impl ConstrainedPatchesPolicy {
    pub(in crate::beta_crown::engine::graph::propagation) fn selective_matrix_reentry() -> Self {
        Self {
            allow_initial_patches: false,
            allow_conv_reentry: true,
        }
    }

    #[cfg(test)]
    pub(super) fn dense_only() -> Self {
        Self {
            allow_initial_patches: false,
            allow_conv_reentry: false,
        }
    }

    pub(in crate::beta_crown::engine::graph::propagation) fn seed_bounds(
        self,
        dense: LinearBounds,
    ) -> CrownBounds {
        debug_assert!(
            !self.allow_initial_patches,
            "#3813 constrained backward still starts from dense objective rows"
        );
        CrownBounds::Dense(dense)
    }
}

pub(super) fn patches_dense_fallback_details(
    bounds: &CrownBounds,
    site: &'static str,
) -> Result<Option<String>> {
    let CrownBounds::Patches(pb) = bounds else {
        return Ok(None);
    };
    let (rows, cols) = pb.dense_pair_shape()?;
    let budget = cpu_crown_dense_budget_bytes();
    let estimate = DenseMaterializationEstimate {
        site,
        rows,
        cols,
        required_bytes: pb.dense_pair_bytes()?,
    };
    if estimate.exceeds_budget(budget) {
        Ok(Some(estimate.budget_exceeded_details(budget)))
    } else {
        Ok(None)
    }
}

pub(super) fn maybe_reenter_dense_to_patches(
    policy: ConstrainedPatchesPolicy,
    node_name: &str,
    node: &GraphNode,
    node_cb: &mut CrownBounds,
    ibp_bounds: &HashMap<String, Arc<BoundedTensor>>,
) {
    let is_conv2d = matches!(&node.layer, Layer::Conv2d(_));
    let is_dense = matches!(node_cb, CrownBounds::Dense(_));
    let is_unary = node.inputs.len() == 1;

    if is_conv2d {
        debug!(
            node = node_name,
            allow_conv_reentry = policy.allow_conv_reentry,
            is_dense,
            is_unary,
            num_inputs = node.inputs.len(),
            "maybe_reenter: Conv2d candidate"
        );
    }

    if !policy.allow_conv_reentry || !is_dense || !is_unary || !is_conv2d {
        return;
    }

    let Some(current_bounds) = ibp_bounds.get(node_name) else {
        debug!(node = node_name, "maybe_reenter: no IBP bounds for Conv2d");
        return;
    };
    let current_shape = current_bounds.shape();
    if current_shape.len() != 3 {
        debug!(
            node = node_name,
            shape_len = current_shape.len(),
            "maybe_reenter: IBP shape not 3D"
        );
        return;
    }

    let spatial = (current_shape[0], current_shape[1], current_shape[2]);
    let spatial_dim = spatial.0 * spatial.1 * spatial.2;
    let CrownBounds::Dense(lb) = node_cb else {
        return;
    };
    if lb.num_inputs() != spatial_dim {
        debug!(
            node = node_name,
            num_inputs = lb.num_inputs(),
            spatial_dim,
            "maybe_reenter: dense columns != spatial dim"
        );
        return;
    }

    // #3813: Skip re-entry when the Dense carrier has few objective rows.
    // Patches mode iterates over (rows × spatial_positions) doing per-position
    // conv2d_transpose composition. With few rows (typical BaB constrained
    // backward has ≤10 objectives), Dense backward (small matrix × transposed
    // kernel via BLAS) is dramatically faster. The crossover depends on spatial
    // dims and model structure, but for ≤64 rows Dense wins for all practical
    // conv architectures due to BLAS optimization and zero per-position overhead.
    let num_rows = lb.num_outputs();
    if num_rows <= 64 {
        debug!(
            node = node_name,
            num_rows,
            spatial_dim,
            "maybe_reenter: skipping re-entry, Dense faster for {} rows",
            num_rows
        );
        return;
    }

    match PatchesLinearBounds::from_dense_spatial_rows(lb, spatial) {
        Ok(pb) => {
            info!(
                "Constrained CROWN: Dense->Patches re-entry at {} with {} rows over {:?}",
                node_name, pb.row_count, spatial
            );
            *node_cb = CrownBounds::Patches(Box::new(pb));
        }
        Err(err) => {
            debug!(
                "Constrained CROWN: Dense->Patches re-entry skipped at {}: {}",
                node_name, err
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_patches_step(
    graph: &GraphNetwork,
    label: &str,
    node_name: &str,
    node: &GraphNode,
    node_cb: &mut CrownBounds,
    first_input_name: &str,
    pre_activation: &BoundedTensor,
    ibp_bounds: &HashMap<String, Arc<BoundedTensor>>,
    node_crown_bounds: &mut crate::network::CrownMergeAccumulator,
    input_accumulated: &mut bool,
    engine: Option<&dyn GemmEngine>,
    per_node_deadline: Option<Instant>,
    policy: ConstrainedPatchesPolicy,
) -> Result<bool> {
    maybe_reenter_dense_to_patches(policy, node_name, node, node_cb, ibp_bounds);

    if !matches!(node_cb, CrownBounds::Patches(_)) {
        return Ok(false);
    }

    match crown_backward_step_patches(
        &node.layer,
        node_cb,
        pre_activation,
        engine,
        0,
        label,
        per_node_deadline,
    ) {
        Ok(CrownStepResult::Continue) => {
            graph.accumulate_crown_bounds_to_input(
                first_input_name,
                node_cb.clone(),
                node_crown_bounds,
                0,
                0,
                input_accumulated,
            )?;
            Ok(true)
        }
        Ok(CrownStepResult::IbpFallback(fallback)) => {
            if fallback.reason == crate::types::CrownIbpFallbackReason::MemoryBudgetExceeded {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "{label}: patches dispatch at '{node_name}' exceeded memory budget: {}",
                    fallback.details
                )));
            }
            debug!(
                "{}: Patches dispatch for {} ({}) requested Dense fallback: {}",
                label,
                node_name,
                node.layer.layer_type(),
                fallback.details
            );
            if let Some(details) =
                patches_dense_fallback_details(node_cb, "constraints::try_patches_step")?
            {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "{label}: {details}"
                )));
            }
            node_cb.ensure_dense()?;
            Ok(false)
        }
        Err(err) => {
            let err: NyError = err;
            if err.is_deadline_exceeded() {
                return Err(err);
            }
            debug!(
                "{}: Patches dispatch for {} ({}) failed: {}, falling back to Dense",
                label,
                node_name,
                node.layer.layer_type(),
                err
            );
            if let Some(details) =
                patches_dense_fallback_details(node_cb, "constraints::try_patches_step")?
            {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "{label}: {details}"
                )));
            }
            node_cb.ensure_dense()?;
            Ok(false)
        }
    }
}
