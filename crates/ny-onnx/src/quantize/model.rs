// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential/model-level quantization analysis.

use crate::analysis_error::validate_analysis_epsilon;
use crate::{load_onnx, OnnxModel};
use ny_propagate::{BoundPropagation, GraphNetwork};
use ny_tensor::BoundedTensor;
use std::path::Path;
use tracing::{debug, info};

use super::{
    build_layer_quantization, graph::analyze_quantization_graph_with_input, make_default_input,
    tally_layer_quantization, QuantizeConfig, QuantizeError, QuantizeResult,
};

/// Analyze quantization safety of a model loaded from ONNX file.
pub fn analyze_quantization(
    path: impl AsRef<Path>,
    config: &QuantizeConfig,
) -> Result<QuantizeResult, QuantizeError> {
    info!("Loading model: {}", path.as_ref().display());
    let onnx_model = load_onnx(path.as_ref()).map_err(|e| QuantizeError::load("quantize", e))?;

    analyze_quantization_model(&onnx_model, config)
}

/// Analyze quantization safety of an already-loaded ONNX model.
pub fn analyze_quantization_model(
    model: &OnnxModel,
    config: &QuantizeConfig,
) -> Result<QuantizeResult, QuantizeError> {
    validate_analysis_epsilon("quantize", config.epsilon)?;

    let input = if let Some(ref inp) = config.input {
        inp.clone()
    } else {
        let input_shape = input_shape(model)?;
        make_default_input(&input_shape, config.epsilon, "quantize")?
    };

    info!(
        "Starting quantization analysis with input shape {:?}, epsilon {}",
        input.shape(),
        config.epsilon
    );

    // Residual/binary DAGs require graph traversal because sequential IBP
    // uses `Layer::propagate_ibp`, and binary ops only implement the dedicated
    // multi-input entry points.
    if let Ok(graph) = model.to_graph_network() {
        if graph_requires_dag_quantization(&graph) {
            return analyze_quantization_graph_with_input(&graph, &input, config);
        }
    }

    analyze_quantization_sequential(model, &input, config)
}

fn graph_requires_dag_quantization(graph: &GraphNetwork) -> bool {
    graph
        .node_names()
        .iter()
        .filter_map(|name| graph.node(name))
        .any(|node| node.layer().is_binary())
}

fn input_shape(model: &OnnxModel) -> Result<Vec<usize>, QuantizeError> {
    let input_spec =
        model.network.inputs.first().ok_or_else(|| {
            QuantizeError::invalid_input_shape("quantize", "No input specification")
        })?;

    Ok(input_spec
        .shape
        .iter()
        .map(|&dim| if dim > 0 { dim as usize } else { 1 })
        .collect())
}

fn analyze_quantization_sequential(
    model: &OnnxModel,
    input: &BoundedTensor,
    config: &QuantizeConfig,
) -> Result<QuantizeResult, QuantizeError> {
    let network = model
        .to_propagate_network()
        .map_err(|e| QuantizeError::propagation("quantize", e))?;

    if network.layers().is_empty() {
        return Err(QuantizeError::no_layers("quantize"));
    }

    let mut layers = Vec::with_capacity(network.layers().len());
    let mut current = input.clone();
    let mut float16_overflow_count = 0;
    let mut int8_overflow_count = 0;
    let mut denormal_count = 0;

    for (layer, spec) in network.layers().iter().zip(model.network.layers.iter()) {
        let mut propagation_failed = false;
        let output = match layer.propagate_ibp(&current) {
            Ok(out) => out,
            Err(e) => {
                debug!("Layer {} propagation failed: {}", spec.name, e);
                if !config.continue_after_overflow {
                    return Err(QuantizeError::propagation("quantize", e));
                }
                propagation_failed = true;
                current.clone()
            }
        };

        let layer_result = build_layer_quantization(
            spec.name.clone(),
            format!("{:?}", spec.layer_type),
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
            "Layer {}: bounds [{:.3e}, {:.3e}], f16={}, i8={}",
            layer_result.name,
            layer_result.min_bound,
            layer_result.max_bound,
            layer_result.float16_safety,
            layer_result.int8_safety
        );

        let has_overflow = layer_result.has_overflow;
        layers.push(layer_result);

        if has_overflow && !config.continue_after_overflow {
            break;
        }

        current = output;
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
