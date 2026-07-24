// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::fixtures::{require_test_model_with_hint, WHISPER_TEST_MODEL_HINT};
use super::super::{load_whisper, MultiBlockConfig, WhisperModel};
use ndarray::ArrayD;
use ny_propagate::{
    BoundPropagation, GraphNetwork, GraphNode, Network as PropNetwork, NETWORK_INPUT,
};
use ny_tensor::BoundedTensor;
use std::collections::HashSet;
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

/// Run a single Whisper encoder block with compositional GPU verification.
pub(super) fn run_whisper_block(
    whisper: &WhisperModel,
    index: usize,
    input: &BoundedTensor,
    cfg: &MultiBlockConfig,
    label: &str,
) -> (BoundedTensor, crate::GpuCompositionalDetails) {
    whisper
        .verify_block_compositional_gpu_with_config(index, input, None, cfg)
        .unwrap_or_else(|err| panic!("block {index} {label} failed: {err}"))
}

fn backward_closure_until(
    graph: &GraphNetwork,
    output_node: &str,
    root_input: &str,
) -> HashSet<String> {
    let mut closure = HashSet::new();
    let mut stack = vec![output_node.to_string()];
    while let Some(name) = stack.pop() {
        if name == root_input || !closure.insert(name.clone()) {
            continue;
        }
        if let Some(node) = graph.node(&name) {
            for input in node.inputs() {
                if input != NETWORK_INPUT {
                    stack.push(input.clone());
                }
            }
        }
    }
    closure
}

fn extract_single_input_subgraph(
    full_graph: &GraphNetwork,
    root_input: &str,
    output_node: &str,
) -> Result<GraphNetwork, String> {
    let topo_order = full_graph
        .topological_sort()
        .map_err(|err| format!("topological sort failed: {err}"))?;
    let closure = backward_closure_until(full_graph, output_node, root_input);

    let mut subgraph = GraphNetwork::new();
    for name in &topo_order {
        if !closure.contains(name.as_str()) {
            continue;
        }
        let node = full_graph
            .node(name)
            .ok_or_else(|| format!("missing node '{name}' in source graph"))?;
        let remapped_inputs: Vec<String> = node
            .inputs()
            .iter()
            .map(|input| {
                if input == root_input || input == NETWORK_INPUT {
                    NETWORK_INPUT.to_string()
                } else {
                    input.clone()
                }
            })
            .collect();
        for input in &remapped_inputs {
            if input != NETWORK_INPUT && !closure.contains(input.as_str()) {
                return Err(format!(
                    "node '{}' depends on '{}' outside the extracted suffix rooted at '{}'",
                    name, input, root_input
                ));
            }
        }
        subgraph.add_node(GraphNode::new(
            node.name().to_string(),
            node.layer().clone(),
            remapped_inputs,
        ));
    }
    subgraph.set_output(output_node);
    Ok(subgraph)
}

/// Run a single Whisper encoder block with the experimental attention-CROWN
/// seed from #318 while keeping the MLP suffix aligned with `mlp_crown`.
///
/// This stays test-only on purpose. The current #318 context/seed matrix
/// regressions show the short-sequence attention-CROWN retry collapses to the
/// IBP seed on real Whisper block-0 weights, so zonotope remains the only
/// promoted block-0 seed for production follow-ups.
pub(super) fn run_whisper_block_attention_crown_seed(
    whisper: &WhisperModel,
    index: usize,
    input: &BoundedTensor,
    cfg: &MultiBlockConfig,
    label: &str,
) -> (BoundedTensor, crate::GpuCompositionalDetails) {
    let artifacts = whisper
        .attention_subgraph_artifacts(index)
        .unwrap_or_else(|err| panic!("block {index} {label} attention artifacts failed: {err}"));
    let mut context_graph = artifacts.graph.clone();
    context_graph.set_output(artifacts.context_node.clone());
    if cfg.layernorm_forward_mode {
        context_graph.set_layernorm_forward_mode(true);
    }
    let context_bounds = context_graph
        .propagate_crown_batched_with_attention_full_composition(input)
        .map(|result| result.bounds)
        .unwrap_or_else(|err| {
            panic!("block {index} {label} attention context full-composition failed: {err}")
        });
    let output_suffix = extract_single_input_subgraph(
        &artifacts.graph,
        &artifacts.context_node,
        &artifacts.output_node,
    )
    .unwrap_or_else(|err| panic!("block {index} {label} output suffix extraction failed: {err}"));
    let attn_delta = output_suffix
        .propagate_ibp(&context_bounds)
        .unwrap_or_else(|err| {
            panic!("block {index} {label} attention output suffix failed: {err}")
        });

    let x_attn = input
        .add(&attn_delta)
        .unwrap_or_else(|err| panic!("block {index} {label} x_attn residual failed: {err}"));

    let mut mlp_graph = whisper
        .mlp_subgraph(index)
        .unwrap_or_else(|err| panic!("block {index} {label} mlp graph failed: {err}"));
    let mlp_forward_mode = cfg.layernorm_forward_mode && !cfg.use_crown_block_wise;
    if mlp_forward_mode {
        mlp_graph.set_layernorm_forward_mode(true);
    }
    let (mlp_delta, normalization_row_stats) = if cfg.use_crown_block_wise {
        let (bounds, stats) = mlp_graph
            .propagate_crown_within_graph_per_position_with_stats(&x_attn)
            .unwrap_or_else(|err| {
                panic!("block {index} {label} block-wise MLP CROWN failed: {err}")
            });
        let mapped = stats
            .into_iter()
            .map(|stat| crate::whisper::NormalizationRowStats {
                site_name: stat.node_name,
                fallback_rows: stat.fallback_rows,
                total_rows: stat.total_rows,
            })
            .collect();
        (bounds, mapped)
    } else {
        mlp_graph.set_layernorm_crown_mode(cfg.layernorm_crown_mode);
        let bounds = mlp_graph
            .propagate_crown_per_position(&x_attn)
            .unwrap_or_else(|err| panic!("block {index} {label} MLP CROWN failed: {err}"));
        (bounds, Vec::new())
    };

    let x_out = x_attn
        .add(&mlp_delta)
        .unwrap_or_else(|err| panic!("block {index} {label} output residual failed: {err}"));
    let seq_len = input.shape().get(1).copied().unwrap_or(1);
    let attention_delta_width = attn_delta.max_width();
    let x_attn_width = x_attn.max_width();
    let mlp_delta_width = mlp_delta.max_width();
    let output_width = x_out.max_width();
    (
        x_out,
        crate::GpuCompositionalDetails {
            attention_delta_width,
            x_attn_width,
            mlp_delta_width,
            output_width,
            used_gpu_attention: false,
            used_zonotope_attention: false,
            seq_len,
            normalization_row_stats,
        },
    )
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

/// Print block-level width metrics (attention delta, MLP delta, output).
pub(super) fn print_block_width_tuple(label: &str, details: &crate::GpuCompositionalDetails) {
    println!(
        "  [{label}] attn_delta={:.6e}  mlp_delta={:.6e}  output={:.6e}",
        details.attention_delta_width, details.mlp_delta_width, details.output_width,
    );
}

/// Assert block-level width metrics are finite and non-negative.
pub(super) fn assert_block_width_tuple_finite(
    output: &BoundedTensor,
    details: &crate::GpuCompositionalDetails,
    label: &str,
) {
    assert_tensor_sound_finite(output, label);
    for (metric, value) in [
        ("attention_delta_width", details.attention_delta_width),
        ("x_attn_width", details.x_attn_width),
        ("mlp_delta_width", details.mlp_delta_width),
        ("output_width", details.output_width),
    ] {
        assert!(
            value.is_finite() && value >= 0.0,
            "{label} {metric} must be finite and non-negative: {value:.6e}",
        );
    }
}
