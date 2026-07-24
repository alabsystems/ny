// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::layers::SqrtLayer;

pub(super) fn backward_sqrt_node(
    sqrt: &SqrtLayer,
    alpha_state: Option<&GraphAlphaState>,
    node_name: &str,
    label: &str,
    node_lb: &LinearBounds,
    pre_activation: &BoundedTensor,
) -> Result<LinearBounds> {
    if let Some(alpha) = alpha_state.and_then(|state| state.sqrt_alpha(node_name)) {
        sqrt.propagate_linear_with_alpha(
            node_lb,
            pre_activation,
            &alpha.lower_path_mid,
            Some(&alpha.upper_path_mid),
        )
    } else {
        sqrt.propagate_linear_with_bounds(node_lb, pre_activation)
            .map_err(|e| {
                // Preserve structured error types (UnsupportedConfiguration,
                // NumericalInstability, etc.) so that CROWN-IBP tightening
                // in crown_tighten.rs can catch them and fall back to IBP.
                // Wrapping in InvalidSpec prevents the catch from matching.
                // Fix: same pattern as backward_dispatch/helpers.rs
                // preserve_structured_error (#3602).
                match e {
                    NyError::UnsupportedConfiguration(_)
                    | NyError::UnsupportedOp(_)
                    | NyError::NumericalInstability(_)
                    | NyError::ShapeMismatch { .. }
                    | NyError::DeadlineExceeded(_)
                    | NyError::SoundnessRefusal(_)
                    | NyError::InternalError(_) => e,
                    _ => NyError::InvalidSpec(format!(
                        "{} failed at '{}' (Sqrt): {}",
                        label, node_name, e
                    )),
                }
            })
    }
}
