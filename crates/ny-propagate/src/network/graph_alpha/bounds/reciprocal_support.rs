// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::layers::ReciprocalLayer;

pub(super) fn backward_reciprocal_node(
    reciprocal: &ReciprocalLayer,
    alpha_state: Option<&GraphAlphaState>,
    node_name: &str,
    label: &str,
    node_lb: &LinearBounds,
    pre_activation: &BoundedTensor,
) -> Result<LinearBounds> {
    if let Some(alpha) = alpha_state.and_then(|state| state.reciprocal_alpha(node_name)) {
        reciprocal.propagate_linear_with_alpha(
            node_lb,
            pre_activation,
            &alpha.lower_path_mid,
            Some(&alpha.upper_path_mid),
        )
    } else {
        reciprocal
            .propagate_linear_with_bounds(node_lb, pre_activation)
            .map_err(|e| match e {
                NyError::UnsupportedConfiguration(_)
                | NyError::UnsupportedOp(_)
                | NyError::NumericalInstability(_)
                | NyError::ShapeMismatch { .. }
                | NyError::DeadlineExceeded(_)
                | NyError::SoundnessRefusal(_)
                | NyError::InternalError(_) => e,
                _ => NyError::InvalidSpec(format!(
                    "{} failed at '{}' (Reciprocal): {}",
                    label, node_name, e
                )),
            })
    }
}
