// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GraphNetwork quantization analysis for residual/binary models.

use ny_propagate::{layers::Layer, BoundPropagation, GraphNetwork};
use ny_tensor::BoundedTensor;
use std::{borrow::Cow, collections::HashMap};
use tracing::{debug, info};

use super::{
    build_layer_quantization, make_default_input, tally_layer_quantization, QuantizeConfig,
    QuantizeError, QuantizeResult,
};
use crate::analysis_error::validate_analysis_epsilon;

/// Analyze quantization safety of a `GraphNetwork`.
pub fn analyze_quantization_graph(
    graph: &GraphNetwork,
    config: &QuantizeConfig,
    input_shape: &[usize],
) -> Result<QuantizeResult, QuantizeError> {
    let input = if let Some(ref inp) = config.input {
        inp.clone()
    } else {
        make_default_input(input_shape, config.epsilon, "quantize/graph")?
    };

    analyze_quantization_graph_with_input(graph, &input, config)
}

pub(super) fn analyze_quantization_graph_with_input(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    config: &QuantizeConfig,
) -> Result<QuantizeResult, QuantizeError> {
    validate_analysis_epsilon("quantize/graph", config.epsilon)?;

    info!(
        "Starting graph quantization analysis with input shape {:?}, epsilon {}",
        input.shape(),
        config.epsilon
    );

    let exec_order = graph
        .exec_order()
        .map_err(|e| QuantizeError::propagation("quantize/graph", e))?;

    if exec_order.is_empty() {
        return Err(QuantizeError::no_layers("quantize/graph"));
    }

    let mut bounds_cache: HashMap<String, BoundedTensor> = HashMap::with_capacity(exec_order.len());
    let mut layers = Vec::with_capacity(exec_order.len());
    let mut float16_overflow_count = 0;
    let mut int8_overflow_count = 0;
    let mut denormal_count = 0;

    for node_name in exec_order {
        let node = graph.node(node_name).ok_or_else(|| {
            QuantizeError::propagation_msg("quantize/graph", format!("Node not found: {node_name}"))
        })?;
        let mut propagation_failed = false;

        // Concat must be handled before `is_binary()` because n-ary concat
        // would otherwise drop trailing inputs.
        let output = if let Layer::Concat(concat) = node.layer() {
            let owned_inputs: Vec<BoundedTensor> = if let Some(ref constant_inputs) =
                concat.constant_inputs
            {
                let mut graph_idx = 0;
                constant_inputs
                    .iter()
                    .map(|constant| {
                        if let Some(value) = constant {
                            Ok(value.clone())
                        } else {
                            let input_name = node.inputs().get(graph_idx).ok_or_else(|| {
                                QuantizeError::propagation_msg(
                                    "quantize/graph",
                                    format!("Concat: ran out of graph inputs at idx {graph_idx}"),
                                )
                            })?;
                            graph_idx += 1;
                            get_bounds(input_name, input, &bounds_cache).map(Cow::into_owned)
                        }
                    })
                    .collect::<Result<Vec<_>, QuantizeError>>()?
            } else {
                node.inputs()
                    .iter()
                    .map(|name| get_bounds(name, input, &bounds_cache).map(Cow::into_owned))
                    .collect::<Result<Vec<_>, QuantizeError>>()?
            };
            let input_refs: Vec<&BoundedTensor> = owned_inputs.iter().collect();
            match concat.propagate_ibp_nary(&input_refs) {
                Ok(out) => out,
                Err(e) => {
                    debug!("Node {} propagation failed: {}", node_name, e);
                    if !config.continue_after_overflow {
                        return Err(QuantizeError::propagation("quantize/graph", e));
                    }
                    propagation_failed = true;
                    owned_inputs
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| input.clone())
                }
            }
        } else if node.layer().is_binary() {
            let (input_a_name, input_b_name) = node
                .require_binary_inputs()
                .map_err(|e| QuantizeError::propagation("quantize/graph", e))?;
            let input_a = get_bounds(input_a_name, input, &bounds_cache)?;
            let input_b = get_bounds(input_b_name, input, &bounds_cache)?;
            match node.layer().propagate_ibp_binary(&input_a, &input_b) {
                Ok(out) => out,
                Err(e) => {
                    debug!("Node {} propagation failed: {}", node_name, e);
                    if !config.continue_after_overflow {
                        return Err(QuantizeError::propagation("quantize/graph", e));
                    }
                    propagation_failed = true;
                    input_a.into_owned()
                }
            }
        } else {
            let input_name = node
                .require_unary_input()
                .map_err(|e| QuantizeError::propagation("quantize/graph", e))?;
            let node_input = get_bounds(input_name, input, &bounds_cache)?;
            match node.layer().propagate_ibp(&node_input) {
                Ok(out) => out,
                Err(e) => {
                    debug!("Node {} propagation failed: {}", node_name, e);
                    if !config.continue_after_overflow {
                        return Err(QuantizeError::propagation("quantize/graph", e));
                    }
                    propagation_failed = true;
                    node_input.into_owned()
                }
            }
        };

        let layer_result = build_layer_quantization(
            node.name().to_string(),
            node.layer().layer_type().to_string(),
            &output,
            propagation_failed,
        );
        tally_layer_quantization(
            &layer_result,
            &mut float16_overflow_count,
            &mut int8_overflow_count,
            &mut denormal_count,
        );

        debug!(
            "Node {}: bounds [{:.3e}, {:.3e}], f16={}, i8={}",
            layer_result.name,
            layer_result.min_bound,
            layer_result.max_bound,
            layer_result.float16_safety,
            layer_result.int8_safety
        );

        let has_overflow = layer_result.has_overflow;
        layers.push(layer_result);
        bounds_cache.insert(node.name().to_string(), output);

        if has_overflow && !config.continue_after_overflow {
            break;
        }
    }

    Ok(QuantizeResult {
        layers,
        float16_safe: float16_overflow_count == 0,
        int8_safe: int8_overflow_count == 0,
        float16_overflow_count,
        int8_overflow_count,
        denormal_count,
        input_epsilon: config.epsilon,
    })
}

fn get_bounds<'a>(
    input_name: &str,
    network_input: &BoundedTensor,
    cache: &'a HashMap<String, BoundedTensor>,
) -> Result<Cow<'a, BoundedTensor>, QuantizeError> {
    if input_name == "_input" {
        Ok(Cow::Owned(network_input.clone()))
    } else {
        cache.get(input_name).map(Cow::Borrowed).ok_or_else(|| {
            QuantizeError::propagation_msg(
                "quantize/graph",
                format!("Input {input_name} not found in cache"),
            )
        })
    }
}
