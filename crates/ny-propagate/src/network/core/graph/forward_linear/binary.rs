// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use ny_core::Result;
use ny_tensor::BoundedTensor;

use super::{
    compose_forward_relaxation, layer_debug_name, resolve_pre_activation_bounds,
    resolve_upstream_linear_bounds, sum_linear_bounds, unsupported_forward_linear_node,
    wrap_forward_linear_error,
};
use crate::bounds::LinearBounds;
use crate::layers::{BoundPropagation, Layer, MulBinaryLayer, PowConstantLayer};
use crate::MulBinaryRelaxationMode;

pub(super) fn compose_binary_forward<F>(
    node_name: &str,
    layer: &Layer,
    inputs: &[String],
    output_dim: usize,
    forward_bounds: &HashMap<String, LinearBounds>,
    ibp_node_bounds: &HashMap<String, BoundedTensor>,
    input: &BoundedTensor,
    input_dim: usize,
    local_forward: F,
) -> Result<LinearBounds>
where
    F: FnOnce(
        &LinearBounds,
        &BoundedTensor,
        &BoundedTensor,
    ) -> Result<(LinearBounds, LinearBounds)>,
{
    if inputs.len() != 2 {
        return Err(unsupported_forward_linear_node(
            node_name,
            layer,
            "binary nodes must have exactly 2 inputs",
        ));
    }

    let layer_name = layer_debug_name(layer);
    let input_a = &inputs[0];
    let input_b = &inputs[1];
    let upstream_a =
        resolve_upstream_linear_bounds(input_a, None, forward_bounds, input_dim, node_name)?;
    let upstream_b =
        resolve_upstream_linear_bounds(input_b, None, forward_bounds, input_dim, node_name)?;
    let input_a_bounds =
        resolve_pre_activation_bounds(input_a, ibp_node_bounds, input, node_name, &layer_name)?;
    let input_b_bounds =
        resolve_pre_activation_bounds(input_b, ibp_node_bounds, input, node_name, &layer_name)?;

    let identity = LinearBounds::identity(output_dim);
    let (local_a, local_b) = local_forward(&identity, input_a_bounds, input_b_bounds)
        .map_err(|error| wrap_forward_linear_error(node_name, layer, error))?;
    let composed_a = compose_forward_relaxation(&local_a, &upstream_a)?;
    let composed_b = compose_forward_relaxation(&local_b, &upstream_b)?;
    sum_linear_bounds(&[composed_a, composed_b])
}

pub(super) fn compose_div_forward(
    node_name: &str,
    layer: &Layer,
    inputs: &[String],
    output_dim: usize,
    forward_bounds: &HashMap<String, LinearBounds>,
    ibp_node_bounds: &HashMap<String, BoundedTensor>,
    input: &BoundedTensor,
    input_dim: usize,
) -> Result<LinearBounds> {
    if inputs.len() != 2 {
        return Err(unsupported_forward_linear_node(
            node_name,
            layer,
            "binary nodes must have exactly 2 inputs",
        ));
    }

    let layer_name = layer_debug_name(layer);
    let numerator_name = &inputs[0];
    let denominator_name = &inputs[1];
    let numerator_upstream =
        resolve_upstream_linear_bounds(numerator_name, None, forward_bounds, input_dim, node_name)?;
    let denominator_upstream = resolve_upstream_linear_bounds(
        denominator_name,
        None,
        forward_bounds,
        input_dim,
        node_name,
    )?;
    let numerator_bounds = resolve_pre_activation_bounds(
        numerator_name,
        ibp_node_bounds,
        input,
        node_name,
        &layer_name,
    )?;
    let denominator_bounds = resolve_pre_activation_bounds(
        denominator_name,
        ibp_node_bounds,
        input,
        node_name,
        &layer_name,
    )?;

    // Match alpha-beta-CROWN's graph rewrite for division: Div(a, b) becomes
    // Mul(a, Reciprocal(b)). This keeps the forward-linear packet narrow while
    // reusing the existing reciprocal and broadcast-aware Mul relaxations.
    // Reference: auto_LiRPA/optimize_graph.py::div_to_mul.
    let reciprocal_layer = PowConstantLayer::new(-1.0);
    let reciprocal_ibp = reciprocal_layer
        .propagate_ibp(denominator_bounds)
        .map_err(|error| wrap_forward_linear_error(node_name, layer, error))?;
    let reciprocal_local = reciprocal_layer
        .propagate_linear_with_bounds(
            &LinearBounds::identity(denominator_bounds.len()),
            denominator_bounds,
        )
        .map_err(|error| wrap_forward_linear_error(node_name, layer, error))?;
    let reciprocal_upstream = compose_forward_relaxation(&reciprocal_local, &denominator_upstream)?;

    let (local_numerator, local_reciprocal) = MulBinaryLayer
        .propagate_linear_binary(
            &LinearBounds::identity(output_dim),
            numerator_bounds,
            &reciprocal_ibp,
            MulBinaryRelaxationMode::Middle,
        )
        .map_err(|error| wrap_forward_linear_error(node_name, layer, error))?;

    let composed_numerator = compose_forward_relaxation(&local_numerator, &numerator_upstream)?;
    let composed_reciprocal = compose_forward_relaxation(&local_reciprocal, &reciprocal_upstream)?;
    sum_linear_bounds(&[composed_numerator, composed_reciprocal])
}
