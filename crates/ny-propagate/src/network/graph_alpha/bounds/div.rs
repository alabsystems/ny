// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::LinearBounds;
use ndarray::Array1;
use ny_core::Result;
use ny_tensor::BoundedTensor;

pub(super) enum DivBackwardResult {
    PropagateNumerator(Box<LinearBounds>),
    ConcretizeCurrentNode {
        lower: Box<Array1<f32>>,
        upper: Box<Array1<f32>>,
    },
}

/// Alpha-CROWN uses the same audited reciprocal center-radius certificate as
/// graph CROWN. Keeping one implementation prevents the two verdict paths from
/// drifting on reciprocal rounding, coefficient cast gaps, or error carriers.
pub(super) fn backward_div_to_numerator(
    _node_name: &str,
    node_lb: &LinearBounds,
    input_a_bounds: &BoundedTensor,
    input_b_bounds: &BoundedTensor,
    node_output_bounds: &BoundedTensor,
) -> Result<DivBackwardResult> {
    Ok(
        match crate::network::graph_crown::backward_div_to_numerator(
            node_lb,
            input_a_bounds,
            input_b_bounds,
            node_output_bounds,
        )? {
            crate::network::graph_crown::DivBackwardResult::PropagateNumerator(bounds) => {
                DivBackwardResult::PropagateNumerator(bounds)
            }
            crate::network::graph_crown::DivBackwardResult::ConcretizeCurrentNode(concrete) => {
                DivBackwardResult::ConcretizeCurrentNode {
                    lower: concrete.lower,
                    upper: concrete.upper,
                }
            }
        },
    )
}
