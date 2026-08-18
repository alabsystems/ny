// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_test_utils::assert_bounded_tensor_close;

use super::super::fixtures::*;
use super::super::*;
use ndarray::ArrayD;
use ny_propagate::{
    GraphNetwork, GraphNode, ZonotopePropagationOptions, ZonotopeSoftmaxMode, NETWORK_INPUT,
};
use ny_tensor::{BoundedTensor, ZonotopeTensor};
use std::collections::HashSet;

struct ContextFallbackCase {
    graph: GraphNetwork,
    context_node: String,
    output_node: String,
    ln_output: BoundedTensor,
    context_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

fn load_context_fallback_case() -> ContextFallbackCase {
    let path = require_test_model_with_hint("whisper_tiny_encoder.onnx", WHISPER_TEST_MODEL_HINT);
    let whisper = load_whisper(&path).expect("Failed to load model");
    let hidden_dim = whisper.hidden_dim;
    let num_heads = whisper.num_heads;
    let head_dim = hidden_dim / num_heads;
    let artifacts = whisper
        .attention_suffix_subgraph_artifacts_from_layernorm_output(0)
        .expect("Failed to build suffix subgraph artifacts");

    let input_shape = vec![1, 2, hidden_dim];
    let input_data = ArrayD::from_elem(ndarray::IxDyn(&input_shape), 0.0f32);
    let block_input = BoundedTensor::from_epsilon(input_data, 0.01).expect("valid input");
    let ln_output = whisper
        .attention_layernorm_output_ibp(0, &block_input, false)
        .expect("LayerNorm IBP failed");

    ContextFallbackCase {
        graph: artifacts.graph,
        context_node: artifacts.context_node,
        output_node: artifacts.output_node,
        ln_output,
        context_shape: vec![1, num_heads, 2, head_dim],
        output_shape: vec![1, 2, hidden_dim],
    }
}

fn run_stage_with_zonotope_options(
    graph: &GraphNetwork,
    output_node: &str,
    input: &BoundedTensor,
    stage_name: &str,
    options: ZonotopePropagationOptions,
) -> (BoundedTensor, BoundedTensor) {
    let mut ibp_graph = graph.clone();
    ibp_graph.set_output(output_node);
    let ibp = ibp_graph
        .propagate_ibp(input)
        .unwrap_or_else(|e| panic!("{stage_name} IBP failed: {e:?}"));

    let mut zonotope_graph = graph.clone();
    zonotope_graph.set_output(output_node);
    let zonotope = zonotope_graph
        .propagate_zonotope_with_options(input, 0.0, options)
        .unwrap_or_else(|e| panic!("{stage_name} zonotope failed: {e:?}"));

    (ibp, zonotope)
}

fn assert_sound_finite(output: &BoundedTensor, context: &str, expected_shape: &[usize]) {
    assert_eq!(output.shape(), expected_shape, "{context} shape mismatch");
    assert!(
        output
            .lower()
            .iter()
            .zip(output.upper().iter())
            .all(|(lower, upper)| lower <= upper),
        "{context} bounds must be sound"
    );
    assert!(
        output
            .lower()
            .iter()
            .chain(output.upper().iter())
            .all(|value| value.is_finite()),
        "{context} bounds must be finite"
    );
}

fn assert_bounds_not_wider_than_reference(
    actual: &BoundedTensor,
    reference: &BoundedTensor,
    context: &str,
) {
    assert_eq!(
        actual.shape(),
        reference.shape(),
        "{context} shape mismatch"
    );
    for (idx, ((&actual_lo, &actual_hi), (&ref_lo, &ref_hi))) in actual
        .lower()
        .iter()
        .zip(actual.upper().iter())
        .zip(reference.lower().iter().zip(reference.upper().iter()))
        .enumerate()
    {
        assert!(
            actual_lo <= actual_hi + 1e-6,
            "{context} inverted at index {idx}: lower={actual_lo}, upper={actual_hi}"
        );
        assert!(
            actual_lo >= ref_lo - 1e-6,
            "{context} lower widened at index {idx}: actual={actual_lo}, reference={ref_lo}"
        );
        assert!(
            actual_hi <= ref_hi + 1e-6,
            "{context} upper widened at index {idx}: actual={actual_hi}, reference={ref_hi}"
        );
    }
}

fn backward_closure_until<'a>(
    graph: &'a GraphNetwork,
    output_node: &str,
    root_input: &str,
) -> HashSet<&'a str> {
    let mut visited = HashSet::new();
    let mut stack = vec![output_node.to_string()];
    while let Some(name) = stack.pop() {
        if name == NETWORK_INPUT || name == root_input {
            continue;
        }
        if let Some(node) = graph.node(&name) {
            let node_name = node.name();
            if !visited.insert(node_name) {
                continue;
            }
            for input in node.inputs() {
                stack.push(input.clone());
            }
        }
    }
    visited
}

fn extract_single_input_subgraph(
    full_graph: &GraphNetwork,
    root_input: &str,
    output_node: &str,
) -> Result<GraphNetwork, String> {
    let topo_order = full_graph
        .topological_sort()
        .map_err(|e| format!("topological sort failed: {e}"))?;
    let closure = backward_closure_until(full_graph, output_node, root_input);

    let mut subgraph = GraphNetwork::new();
    for name in &topo_order {
        if !closure.contains(name.as_str()) {
            continue;
        }
        let node = full_graph.node(name).expect("closure nodes must exist");
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
                    "node '{}' depends on '{}' outside context suffix rooted at '{}'",
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

/// Packet C for #318: make the `softmax @ V` fallback seam explicit.
///
/// The context node itself is computed by IBP fallback on the zonotope path, so
/// converting that interval back into a zonotope must preserve the context
/// bounds exactly. The only remaining question is whether the re-zonotized
/// affine suffix widens beyond a pure-IBP suffix rooted at the same context
/// bounds.
#[ntest::timeout(60000)]
#[cfg(feature = "external-whisper")]
#[test]
fn test_whisper_attention_softmax_cut_context_rezonotization_boundary_318() {
    crate::test_fixtures::assert_test_model_available!("whisper_tiny_encoder.onnx");
    let case = load_context_fallback_case();
    let options =
        ZonotopePropagationOptions::new().with_softmax_mode(ZonotopeSoftmaxMode::IntervalFallback);

    let (_, context_cut) = run_stage_with_zonotope_options(
        &case.graph,
        &case.context_node,
        &case.ln_output,
        "context",
        options,
    );
    assert_sound_finite(&context_cut, "context fallback", &case.context_shape);

    let reencoded_context = ZonotopeTensor::from_bounded_tensor(&context_cut)
        .to_bounded_tensor()
        .expect("single-error context re-zonotization should stay finite");
    assert_bounded_tensor_close(
        &context_cut,
        &reencoded_context,
        1e-6,
        "context fallback re-zonotization",
    );

    let context_suffix =
        extract_single_input_subgraph(&case.graph, &case.context_node, &case.output_node)
            .expect("context->output suffix graph should extract cleanly");
    let suffix_ibp = context_suffix
        .propagate_ibp(&context_cut)
        .expect("pure-IBP context suffix should succeed");
    assert_sound_finite(&suffix_ibp, "context suffix pure IBP", &case.output_shape);

    let (_, output_cut) = run_stage_with_zonotope_options(
        &case.graph,
        &case.output_node,
        &case.ln_output,
        "output",
        options,
    );
    assert_sound_finite(&output_cut, "softmax-cut output", &case.output_shape);
    assert_bounds_not_wider_than_reference(
        &output_cut,
        &suffix_ibp,
        "context suffix re-zonotization",
    );

    println!("\n=== Whisper Attention Context Fallback Boundary (#318) ===");
    println!("  context fallback width={:.6e}", context_cut.max_width());
    println!(
        "  context re-zonotized width={:.6e}",
        reencoded_context.max_width()
    );
    println!(
        "  context->output pure-IBP suffix width={:.6e}",
        suffix_ibp.max_width()
    );
    println!(
        "  context->output re-zonotized suffix width={:.6e}",
        output_cut.max_width()
    );
    if suffix_ibp.max_width() > 0.0 {
        println!(
            "  re-zonotized/pure-IBP suffix ratio={:.6e}",
            output_cut.max_width() / suffix_ibp.max_width()
        );
    }
}
