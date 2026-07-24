// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Forward-propagation helper for tightened bounds in the CROWN-IBP hybrid loop.
//!
//! Extracted from `crown_ibp.rs` to keep that module under the 500-line limit.
//! Only called from [`super::crown_ibp::collect_core`] when a layer does not
//! need a full partial CROWN backward pass.

use crate::layers::{BoundPropagation, Layer};
use crate::types::{BoundsProvenance, CrownIbpFallbackReason};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

pub(super) type TightenedForwardResult = (
    BoundedTensor,
    BoundsProvenance,
    Option<(CrownIbpFallbackReason, String)>,
);

/// Forward-propagate a previously tightened bound when the current layer does
/// not need a full partial CROWN pass.
pub(super) fn propagate_forward_tightened_bound(
    layer: &Layer,
    layer_index: usize,
    input: &BoundedTensor,
    prior_bounds: &[BoundedTensor],
    ibp_bound: &BoundedTensor,
) -> Result<TightenedForwardResult> {
    let tightened_input = if layer_index == 0 {
        input
    } else {
        prior_bounds.get(layer_index - 1).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Missing prior tightened bound for layer {} ({})",
                layer_index,
                layer.layer_type()
            ))
        })?
    };

    let tightened = layer
        .propagate_ibp(tightened_input)
        .map_err(|e| NyError::LayerError {
            layer_index,
            layer_type: layer.layer_type().to_string(),
            source: Box::new(e),
        })?;

    let tightened = if tightened.shape() != ibp_bound.shape() && tightened.len() == ibp_bound.len()
    {
        match tightened.reshape(ibp_bound.shape()) {
            Ok(reshaped) => reshaped,
            Err(_) => tightened,
        }
    } else {
        tightened
    };

    if tightened.shape() != ibp_bound.shape() {
        let reason = CrownIbpFallbackReason::ShapeMismatch;
        return Ok((
            ibp_bound.clone(),
            BoundsProvenance::ForwardFallback(reason),
            Some((
                reason,
                format!(
                    "tightened forward shape {:?} does not match forward shape {:?}",
                    tightened.shape(),
                    ibp_bound.shape()
                ),
            )),
        ));
    }

    match ibp_bound.intersection_per_element(&tightened) {
        Some((tightened, disjoint)) => {
            if disjoint > 0 {
                debug!(
                    "CROWN-IBP layer {} ({}): tightened-forward/IBP intersection used union fallback for {} of {} elements",
                    layer_index,
                    layer.layer_type(),
                    disjoint,
                    tightened.len()
                );
            }
            Ok((tightened, BoundsProvenance::Crown, None))
        }
        None => {
            let reason = CrownIbpFallbackReason::EmptyIntersection;
            Ok((
                ibp_bound.clone(),
                BoundsProvenance::ForwardFallback(reason),
                Some((
                    reason,
                    format!(
                        "tightened-forward/IBP intersection failed (NaN) for shape {:?}",
                        ibp_bound.shape()
                    ),
                )),
            ))
        }
    }
}
