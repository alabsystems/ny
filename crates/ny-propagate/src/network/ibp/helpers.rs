// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared utility functions for IBP and CROWN-IBP propagation.

use crate::bounds::patches::CrownBounds;
use crate::layers::{BoundPropagation, Layer};
use crate::network::core::graph::ibp::dispatch::check_nan_firewall;
use crate::network::core::CrownStepFallback;
use crate::network::crown_memory::{cpu_crown_dense_budget_bytes, DenseMaterializationEstimate};
use crate::network::Network;
use crate::types::CrownIbpFallbackReason;
use ny_core::Result;
use ny_tensor::BoundedTensor;
use tracing::info;

pub(super) fn check_sequential_ibp_nan(
    bounds: &BoundedTensor,
    context: &str,
    layer_index: usize,
    layer_type: &str,
) -> Result<()> {
    let layer_name = format!("layer_{layer_index}");
    check_nan_firewall(bounds, context, &layer_name, layer_type)
}

pub(super) fn memory_budget_partial_fallback(
    estimate: DenseMaterializationEstimate,
    label: &str,
) -> CrownStepFallback {
    let details = estimate.budget_exceeded_details(cpu_crown_dense_budget_bytes());
    info!("{label}: {details}; using forward IBP bounds");
    CrownStepFallback {
        reason: CrownIbpFallbackReason::MemoryBudgetExceeded,
        details,
    }
}

pub(super) fn patches_dense_materialization_fallback(
    crown_bounds: &CrownBounds,
    site: &'static str,
    label: &str,
) -> Result<Option<CrownStepFallback>> {
    let CrownBounds::Patches(pb) = crown_bounds else {
        return Ok(None);
    };
    let (rows, cols) = pb.dense_pair_shape()?;
    let estimate = DenseMaterializationEstimate {
        site,
        rows,
        cols,
        required_bytes: pb.dense_pair_bytes()?,
    };
    if estimate.exceeds_budget(cpu_crown_dense_budget_bytes()) {
        return Ok(Some(memory_budget_partial_fallback(estimate, label)));
    }
    Ok(None)
}

/// Return true when a layer output needs a full partial CROWN pass.
///
/// Sequential CROWN consumes layer `k`'s output only as the pre-activation input
/// to layer `k + 1`. If that next backward step is exact and does not consult the
/// interval values, a full partial backward pass for layer `k` can be redundant.
///
/// Keep this shortcut deliberately narrow. The #3397 benchmark networks only need
/// ReLU/reshape/flatten/affine hops, and widening it to layers whose forward IBP
/// is merely sound (not exact) would silently loosen unrelated sequential models.
pub(super) fn layer_supports_forward_tightening_shortcut(layer: &Layer) -> bool {
    matches!(
        layer,
        Layer::Linear(_)
            | Layer::Conv1d(_)
            | Layer::Conv2d(_)
            | Layer::ConvTranspose1d(_)
            | Layer::ConvTranspose2d(_)
            | Layer::ReLU(_)
            | Layer::Flatten(_)
            | Layer::Reshape(_)
    )
}

/// Check if every element of the bounded tensor is in a stable ReLU region:
/// either lower >= 0 (identity) or upper <= 0 (zero).
///
/// This only justifies skipping a partial CROWN pass when the *successor* layer
/// is ReLU. Other nonlinear successors like Sqrt or Softplus still need tighter
/// pre-activation bounds even when the interval has a fixed sign.
pub(super) fn is_all_relu_stable(bounds: &BoundedTensor) -> bool {
    let lower = bounds.lower();
    let upper = bounds.upper();
    lower
        .iter()
        .zip(upper.iter())
        .all(|(&l, &u)| l >= 0.0 || u <= 0.0)
}

pub(super) fn layer_output_needs_partial_crown(layers: &[Layer], layer_index: usize) -> bool {
    match (layers.get(layer_index), layers.get(layer_index + 1)) {
        (Some(layer), Some(next_layer)) => {
            next_layer.requires_pre_activation_bounds()
                || !layer_supports_forward_tightening_shortcut(layer)
        }
        _ => true,
    }
}

/// Count the layer outputs that would trigger a partial CROWN pass during
/// sequential CROWN-IBP collection.
pub(crate) fn crown_ibp_partial_node_count(network: &Network) -> usize {
    network
        .layers()
        .iter()
        .enumerate()
        .filter(|(layer_index, _)| layer_output_needs_partial_crown(network.layers(), *layer_index))
        .count()
}
