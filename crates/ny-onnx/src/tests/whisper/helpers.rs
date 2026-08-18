// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::fixtures::{require_test_model_with_hint, WHISPER_TEST_MODEL_HINT};
use super::super::{load_whisper, WhisperModel};
use ndarray::ArrayD;
use ny_propagate::{BoundPropagation, GraphNetwork, Network as PropNetwork};
use ny_tensor::BoundedTensor;
use std::sync::OnceLock;

pub(super) fn debug_graph_ibp_failure(graph: &GraphNetwork, input: &BoundedTensor) -> String {
    use ny_propagate::Layer;
    use std::collections::HashMap;

    let exec_order = match graph.topological_sort() {
        Ok(order) => order,
        Err(e) => return format!("Topological sort failed: {:?}", e),
    };

    let mut bounds_cache: HashMap<String, BoundedTensor> = HashMap::new();

    for node_name in exec_order {
        let node = match graph.node(&node_name) {
            Some(node) => node,
            None => return format!("Node missing from graph: {}", node_name),
        };

        let resolve_input = |input_name: &str| -> Result<&BoundedTensor, String> {
            if input_name == "_input" {
                return Ok(input);
            }
            bounds_cache.get(input_name).ok_or_else(|| {
                let mut known: Vec<&str> = bounds_cache.keys().map(String::as_str).collect();
                known.sort_unstable();
                let preview: Vec<&str> = known.into_iter().take(8).collect();
                format!(
                    "Missing bounds for input {} while evaluating node {} ({}); cached={} preview={:?}",
                    input_name,
                    node_name,
                    node.layer().layer_type(),
                    bounds_cache.len(),
                    preview
                )
            })
        };

        let output_bounds = match node.layer() {
            Layer::Where(w) => {
                if w.has_embedded_constants() {
                    if node.inputs().is_empty() {
                        return format!(
                            "Where node {} has embedded constants but no inputs",
                            node_name
                        );
                    }
                    let cond = match resolve_input(&node.inputs()[0]) {
                        Ok(bounds) => bounds,
                        Err(err) => return err,
                    };
                    w.propagate_ibp_with_condition(cond)
                } else {
                    if node.inputs().len() < 3 {
                        return format!(
                            "Where node {} requires 3 inputs, got {}",
                            node_name,
                            node.inputs().len()
                        );
                    }
                    let cond = match resolve_input(&node.inputs()[0]) {
                        Ok(bounds) => bounds,
                        Err(err) => return err,
                    };
                    let x = match resolve_input(&node.inputs()[1]) {
                        Ok(bounds) => bounds,
                        Err(err) => return err,
                    };
                    let y = match resolve_input(&node.inputs()[2]) {
                        Ok(bounds) => bounds,
                        Err(err) => return err,
                    };
                    w.propagate_ibp_ternary(cond, x, y)
                }
            }
            Layer::Concat(concat) => {
                if node.inputs().len() < 2 {
                    return format!(
                        "Concat node {} requires at least 2 inputs, got {}",
                        node_name,
                        node.inputs().len()
                    );
                }
                let mut input_bounds = Vec::with_capacity(node.inputs().len());
                for (i, inp_name) in node.inputs().iter().enumerate() {
                    if let Some(constant) = concat.constant_input(i) {
                        input_bounds.push(constant);
                        continue;
                    }
                    let resolved = match resolve_input(inp_name) {
                        Ok(bounds) => bounds,
                        Err(err) => {
                            return format!(
                                "Concat input {} ({}) resolution failed: {}",
                                i, inp_name, err
                            )
                        }
                    };
                    input_bounds.push(resolved);
                }
                concat.propagate_ibp_nary(&input_bounds)
            }
            _ if node.layer().is_binary() => {
                if node.inputs().len() < 2 {
                    return format!(
                        "Binary node {} requires 2 inputs, got {}",
                        node_name,
                        node.inputs().len()
                    );
                }
                let input_a = match resolve_input(&node.inputs()[0]) {
                    Ok(bounds) => bounds,
                    Err(err) => {
                        return format!("Binary input 0 ({}) error: {}", node.inputs()[0], err)
                    }
                };
                let input_b = match resolve_input(&node.inputs()[1]) {
                    Ok(bounds) => bounds,
                    Err(err) => {
                        return format!("Binary input 1 ({}) error: {}", node.inputs()[1], err)
                    }
                };
                node.layer().propagate_ibp_binary(input_a, input_b)
            }
            _ => {
                if node.inputs().is_empty() {
                    return format!("Node {} has no inputs", node_name);
                }
                let node_input = match resolve_input(&node.inputs()[0]) {
                    Ok(bounds) => bounds,
                    Err(err) => {
                        return format!("Unary input 0 ({}) error: {}", node.inputs()[0], err)
                    }
                };
                node.layer().propagate_ibp(node_input)
            }
        };

        match output_bounds {
            Ok(bounds) => {
                bounds_cache.insert(node_name.clone(), bounds);
            }
            Err(e) => {
                return format!(
                    "IBP failed at node {} ({}) inputs {:?}: {:?}",
                    node_name,
                    node.layer().layer_type(),
                    node.inputs(),
                    e
                );
            }
        }
    }

    "IBP failed without identifying a node".to_string()
}

pub(super) fn whisper_tiny_encoder() -> &'static WhisperModel {
    static WHISPER_TINY: OnceLock<WhisperModel> = OnceLock::new();
    WHISPER_TINY.get_or_init(|| {
        let path =
            require_test_model_with_hint("whisper_tiny_encoder.onnx", WHISPER_TEST_MODEL_HINT);
        load_whisper(&path).expect("Failed to load Whisper model")
    })
}

pub(super) fn whisper_tiny_propagate_network() -> &'static PropNetwork {
    static WHISPER_TINY_NETWORK: OnceLock<PropNetwork> = OnceLock::new();
    WHISPER_TINY_NETWORK.get_or_init(|| {
        // Cache the full network once to keep test runtime under timeout limits.
        whisper_tiny_encoder()
            .model
            .to_propagate_network()
            .expect("Failed to convert Whisper model to PropNetwork")
    })
}

/// Build a zero-centered bounded input for Whisper encoder tests.
/// Standard shape: [1, seq_len, hidden_dim] with uniform epsilon perturbation.
pub(super) fn whisper_zero_input(hidden_dim: usize, seq_len: usize, epsilon: f32) -> BoundedTensor {
    BoundedTensor::from_epsilon(
        ArrayD::from_elem(ndarray::IxDyn(&[1, seq_len, hidden_dim]), 0.0f32),
        epsilon,
    )
    .expect("valid test input")
}

/// Assert that bounded tensor output is sound (lower <= upper for all elements).
pub(super) fn assert_bounds_sound(output: &BoundedTensor, label: &str) {
    assert!(
        output
            .lower()
            .iter()
            .zip(output.upper().iter())
            .all(|(l, u)| l <= u),
        "{label} output bounds must be sound (lower <= upper)"
    );
}

/// Assert that bounded tensor output is sound and all values are finite.
pub(super) fn assert_tensor_sound_finite(output: &BoundedTensor, label: &str) {
    assert_bounds_sound(output, label);
    assert!(
        output
            .lower()
            .iter()
            .chain(output.upper().iter())
            .all(|value| value.is_finite()),
        "{label} output bounds must be finite"
    );
}
