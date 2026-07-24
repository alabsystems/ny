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
use ny_tensor::BoundedTensor;
use std::collections::{BTreeSet, HashSet};

fn print_input_width(input: &BoundedTensor) {
    let input_width = input.max_width();
    assert!(input_width.is_finite(), "input width must be finite");
    println!(
        "  stage=input node=<graph-input> ibp={input_width:.6e} zonotope={input_width:.6e} ratio=1.000000e0"
    );
}

fn run_stage(
    graph: &GraphNetwork,
    output_node: &str,
    input: &BoundedTensor,
    stage_name: &str,
) -> (BoundedTensor, BoundedTensor) {
    run_stage_with_zonotope_options(
        graph,
        output_node,
        input,
        stage_name,
        ZonotopePropagationOptions::default(),
    )
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

fn assert_sound_finite(output: &BoundedTensor, stage_name: &str, label: &str, expected: &[usize]) {
    assert_eq!(
        output.shape(),
        expected,
        "{stage_name} {label} shape mismatch"
    );

    let bounds_sound = output
        .lower()
        .iter()
        .zip(output.upper().iter())
        .all(|(l, u)| l <= u);
    assert!(bounds_sound, "{stage_name} {label} bounds must be sound");

    let bounds_finite = output
        .lower()
        .iter()
        .chain(output.upper().iter())
        .all(|v| v.is_finite());
    assert!(bounds_finite, "{stage_name} {label} bounds must be finite");
}

fn assert_conservative_prefix_no_wider(
    stage_name: &str,
    forward_zonotope: &BoundedTensor,
    conservative_zonotope: &BoundedTensor,
) {
    let forward_width = forward_zonotope.max_width();
    let conservative_width = conservative_zonotope.max_width();
    assert!(
        conservative_width <= forward_width * 1.01,
        "conservative attention LayerNorm seam should stay no wider than the \
         forward-mode seam at {stage_name}: conservative={conservative_width:.6e}, \
         forward={forward_width:.6e}"
    );
}

fn softmax_cut_options() -> ZonotopePropagationOptions {
    ZonotopePropagationOptions::new().with_softmax_mode(ZonotopeSoftmaxMode::IntervalFallback)
}

fn print_stage_widths(
    stage_name: &str,
    output_node: &str,
    ibp: &BoundedTensor,
    zonotope: &BoundedTensor,
) {
    let ibp_width = ibp.max_width();
    let zonotope_width = zonotope.max_width();
    assert!(
        ibp_width.is_finite(),
        "{stage_name} IBP width must be finite"
    );
    assert!(
        zonotope_width.is_finite(),
        "{stage_name} zonotope width must be finite"
    );

    if ibp_width > 0.0 {
        println!(
            "  stage={stage_name} node={output_node} ibp={ibp_width:.6e} zonotope={zonotope_width:.6e} ratio={:.6e}",
            zonotope_width / ibp_width
        );
    } else {
        println!(
            "  stage={stage_name} node={output_node} ibp={ibp_width:.6e} zonotope={zonotope_width:.6e} ratio=zero-width-ibp"
        );
    }
}

struct PrefixCutStageLocalizationCase {
    graph: GraphNetwork,
    scores_node: String,
    softmax_node: String,
    context_node: String,
    output_node: String,
    ln_output: BoundedTensor,
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,
}

fn load_prefix_cut_stage_localization_case(
    layernorm_forward_mode: bool,
) -> PrefixCutStageLocalizationCase {
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
        .attention_layernorm_output_ibp(0, &block_input, layernorm_forward_mode)
        .expect("LayerNorm IBP failed");

    PrefixCutStageLocalizationCase {
        graph: artifacts.graph,
        scores_node: artifacts.scores_node,
        softmax_node: artifacts.softmax_node,
        context_node: artifacts.context_node,
        output_node: artifacts.output_node,
        ln_output,
        hidden_dim,
        num_heads,
        head_dim,
    }
}

fn assert_shared_prefix_topology(
    graph: &GraphNetwork,
    stage_nodes: &[(&str, &str); 6],
    label: &str,
) {
    let query_inputs = reachable_external_inputs(graph, stage_nodes[0].1);
    let key_inputs = reachable_external_inputs(graph, stage_nodes[1].1);
    let expected_inputs: BTreeSet<String> = std::iter::once(NETWORK_INPUT.to_string()).collect();
    assert_eq!(
        query_inputs, expected_inputs,
        "query must reach only _input in {label} graph, got {:?}",
        query_inputs
    );
    assert_eq!(
        key_inputs, expected_inputs,
        "key must reach only _input in {label} graph, got {:?}",
        key_inputs
    );
    println!("  topology: query and key both reach exactly {{\"_input\"}}");
}

/// Softmax-cut stage localization: keep the shared-source prefix cut from the
/// prior #318 packet, then intentionally degrade the Softmax node to IBP so the
/// next widening boundary can be measured downstream of Softmax.
#[ntest::timeout(60000)]
#[test]
fn test_whisper_attention_softmax_cut_stage_localization_318() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    let case = load_prefix_cut_stage_localization_case(false);

    println!("\n=== Whisper Attention Softmax-Cut Stage Localization (#318) ===");
    println!("Block 0 ln_output shape: {:?}", case.ln_output.shape());
    print_input_width(&case.ln_output);

    let stage_nodes = localized_stage_nodes(
        &case.graph,
        case.scores_node.as_str(),
        case.softmax_node.as_str(),
        case.context_node.as_str(),
        case.output_node.as_str(),
    );
    assert_shared_prefix_topology(&case.graph, &stage_nodes, "softmax-cut");

    for (stage_name, output_node, expected_shape) in [
        (
            "query",
            stage_nodes[0].1,
            vec![1, case.num_heads, 2, case.head_dim],
        ),
        (
            "key",
            stage_nodes[1].1,
            vec![1, case.num_heads, 2, case.head_dim],
        ),
        ("scores", stage_nodes[2].1, vec![1, case.num_heads, 2, 2]),
        ("softmax", stage_nodes[3].1, vec![1, case.num_heads, 2, 2]),
        (
            "context",
            stage_nodes[4].1,
            vec![1, case.num_heads, 2, case.head_dim],
        ),
        ("output", stage_nodes[5].1, vec![1, 2, case.hidden_dim]),
    ] {
        let (ibp, zonotope) = run_stage_with_zonotope_options(
            &case.graph,
            output_node,
            &case.ln_output,
            stage_name,
            softmax_cut_options(),
        );
        assert_sound_finite(&ibp, stage_name, "IBP", expected_shape.as_slice());
        assert_sound_finite(&zonotope, stage_name, "zonotope", expected_shape.as_slice());
        if stage_name == "softmax" {
            assert_bounded_tensor_close(&zonotope, &ibp, 1e-6, "softmax interval cut");
        }
        print_stage_widths(stage_name, output_node, &ibp, &zonotope);
    }
}

fn prefix_cut_stage_specs<'a>(
    case: &PrefixCutStageLocalizationCase,
    stage_nodes: [(&'static str, &'a str); 6],
) -> [(&'static str, &'a str, Vec<usize>); 6] {
    [
        (
            "query",
            stage_nodes[0].1,
            vec![1, case.num_heads, 2, case.head_dim],
        ),
        (
            "key",
            stage_nodes[1].1,
            vec![1, case.num_heads, 2, case.head_dim],
        ),
        ("scores", stage_nodes[2].1, vec![1, case.num_heads, 2, 2]),
        ("softmax", stage_nodes[3].1, vec![1, case.num_heads, 2, 2]),
        (
            "context",
            stage_nodes[4].1,
            vec![1, case.num_heads, 2, case.head_dim],
        ),
        ("output", stage_nodes[5].1, vec![1, 2, case.hidden_dim]),
    ]
}

fn assert_prefix_seam_stage(
    stage_name: &str,
    output_node: &str,
    expected_shape: &[usize],
    forward_case: &PrefixCutStageLocalizationCase,
    conservative_case: &PrefixCutStageLocalizationCase,
) {
    let (forward_ibp, forward_zonotope) = run_stage_with_zonotope_options(
        &forward_case.graph,
        output_node,
        &forward_case.ln_output,
        stage_name,
        softmax_cut_options(),
    );
    let (conservative_ibp, conservative_zonotope) = run_stage_with_zonotope_options(
        &conservative_case.graph,
        output_node,
        &conservative_case.ln_output,
        stage_name,
        softmax_cut_options(),
    );
    assert_sound_finite(&forward_ibp, stage_name, "forward-mode IBP", expected_shape);
    assert_sound_finite(
        &forward_zonotope,
        stage_name,
        "forward-mode zonotope",
        expected_shape,
    );
    assert_sound_finite(
        &conservative_ibp,
        stage_name,
        "conservative IBP",
        expected_shape,
    );
    assert_sound_finite(
        &conservative_zonotope,
        stage_name,
        "conservative zonotope",
        expected_shape,
    );
    if stage_name == "softmax" {
        assert_bounded_tensor_close(
            &forward_zonotope,
            &forward_ibp,
            1e-6,
            "forward-mode softmax interval cut",
        );
        assert_bounded_tensor_close(
            &conservative_zonotope,
            &conservative_ibp,
            1e-6,
            "conservative softmax interval cut",
        );
    }

    let forward_width = forward_zonotope.max_width();
    let conservative_width = conservative_zonotope.max_width();
    println!(
        "  stage={stage_name} forward={forward_width:.6e} conservative={conservative_width:.6e} ratio={:.6e}",
        conservative_width / forward_width,
    );
    if matches!(stage_name, "query" | "key" | "scores" | "output") {
        assert_conservative_prefix_no_wider(stage_name, &forward_zonotope, &conservative_zonotope);
    }
}

#[ntest::timeout(60000)]
#[test]
fn test_whisper_attention_softmax_cut_prefix_seam_comparison_318() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    let forward_case = load_prefix_cut_stage_localization_case(true);
    let conservative_case = load_prefix_cut_stage_localization_case(false);

    println!("\n=== Whisper Attention Prefix Seam Comparison (#318) ===");
    println!("forward-mode ln_output:");
    print_input_width(&forward_case.ln_output);
    println!("conservative ln_output:");
    print_input_width(&conservative_case.ln_output);

    let forward_stage_nodes = localized_stage_nodes(
        &forward_case.graph,
        forward_case.scores_node.as_str(),
        forward_case.softmax_node.as_str(),
        forward_case.context_node.as_str(),
        forward_case.output_node.as_str(),
    );
    let conservative_stage_nodes = localized_stage_nodes(
        &conservative_case.graph,
        conservative_case.scores_node.as_str(),
        conservative_case.softmax_node.as_str(),
        conservative_case.context_node.as_str(),
        conservative_case.output_node.as_str(),
    );
    assert_eq!(
        forward_stage_nodes, conservative_stage_nodes,
        "prefix-cut suffix graph topology should stay stable across LayerNorm seam modes"
    );
    assert_shared_prefix_topology(&forward_case.graph, &forward_stage_nodes, "forward-mode");
    assert_shared_prefix_topology(
        &conservative_case.graph,
        &conservative_stage_nodes,
        "conservative",
    );

    for (stage_name, output_node, expected_shape) in
        prefix_cut_stage_specs(&forward_case, forward_stage_nodes)
    {
        assert_prefix_seam_stage(
            stage_name,
            output_node,
            expected_shape.as_slice(),
            &forward_case,
            &conservative_case,
        );
    }
}

fn stage_node<'a>(graph: &'a GraphNetwork, stage_name: &str, output_node: &str) -> &'a GraphNode {
    graph.node(output_node).unwrap_or_else(|| {
        panic!("{stage_name} node {output_node} must exist in attention subgraph")
    })
}

fn assert_stage_nodes_present_and_unique(graph: &GraphNetwork, stage_nodes: &[(&str, &str)]) {
    let unique_nodes: HashSet<&str> = stage_nodes.iter().map(|(_, node)| *node).collect();
    assert_eq!(
        unique_nodes.len(),
        stage_nodes.len(),
        "stage-localization nodes must be distinct"
    );

    for (stage_name, output_node) in stage_nodes {
        stage_node(graph, stage_name, output_node);
    }
}

fn localized_stage_nodes<'a>(
    graph: &'a GraphNetwork,
    scores_name: &'a str,
    softmax_name: &'a str,
    context_name: &'a str,
    output_name: &'a str,
) -> [(&'static str, &'a str); 6] {
    let scores_node = stage_node(graph, "scores", scores_name);
    assert_eq!(
        scores_node.inputs().len(),
        2,
        "scores node must consume query/key prefix nodes"
    );

    let softmax_node = stage_node(graph, "softmax", softmax_name);
    assert_eq!(
        softmax_node.inputs().len(),
        1,
        "softmax must consume scores directly"
    );
    assert_eq!(
        softmax_node.inputs()[0],
        scores_name,
        "softmax must consume scores directly"
    );

    let context_node = stage_node(graph, "context", context_name);
    assert!(
        context_node
            .inputs()
            .iter()
            .any(|input| input == softmax_name),
        "context must consume softmax output directly"
    );

    let stage_nodes = [
        ("query", scores_node.inputs()[0].as_str()),
        ("key", scores_node.inputs()[1].as_str()),
        ("scores", scores_name),
        ("softmax", softmax_name),
        ("context", context_name),
        ("output", output_name),
    ];
    assert_stage_nodes_present_and_unique(graph, &stage_nodes);
    stage_nodes
}

#[ntest::timeout(60000)]
#[test]
fn test_whisper_attention_stage_localization_real_weights_318() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    let path = require_test_model_with_hint("whisper_tiny_encoder.onnx", WHISPER_TEST_MODEL_HINT);
    let whisper = load_whisper(&path).expect("Failed to load model");
    let hidden_dim = whisper.hidden_dim;
    let num_heads = whisper.num_heads;
    let head_dim = hidden_dim / num_heads;
    let artifacts = whisper
        .attention_subgraph_artifacts(0)
        .expect("Failed to build attention subgraph artifacts");

    let input_shape = vec![1, 2, hidden_dim];
    let input_data = ArrayD::from_elem(ndarray::IxDyn(&input_shape), 0.0f32);
    let input = BoundedTensor::from_epsilon(input_data, 0.01).expect("valid input");

    println!("\n=== Whisper Attention Stage Localization (#318) ===");
    println!("Block 0 input shape: {:?}", input_shape);
    print_input_width(&input);

    let stage_nodes = localized_stage_nodes(
        &artifacts.graph,
        artifacts.scores_node.as_str(),
        artifacts.softmax_node.as_str(),
        artifacts.context_node.as_str(),
        artifacts.output_node.as_str(),
    );

    for (stage_name, output_node, expected_shape) in [
        ("query", stage_nodes[0].1, vec![1, num_heads, 2, head_dim]),
        ("key", stage_nodes[1].1, vec![1, num_heads, 2, head_dim]),
        ("scores", stage_nodes[2].1, vec![1, num_heads, 2, 2]),
        ("softmax", stage_nodes[3].1, vec![1, num_heads, 2, 2]),
        ("context", stage_nodes[4].1, vec![1, num_heads, 2, head_dim]),
        ("output", stage_nodes[5].1, vec![1, 2, hidden_dim]),
    ] {
        let (ibp, zonotope) = run_stage(&artifacts.graph, output_node, &input, stage_name);
        assert_sound_finite(&ibp, stage_name, "IBP", expected_shape.as_slice());
        assert_sound_finite(&zonotope, stage_name, "zonotope", expected_shape.as_slice());
        print_stage_widths(stage_name, output_node, &ibp, &zonotope);
    }
}

/// Walk backward through the graph from `start` and collect all external input
/// sentinel names reachable from it. An external input is any node name that
/// does not exist in the graph (typically `_input` via `NETWORK_INPUT`).
///
/// Test-local helper for #318 prefix-cut topology assertion.
fn reachable_external_inputs(graph: &GraphNetwork, start: &str) -> BTreeSet<String> {
    let mut result = BTreeSet::new();
    let mut stack = vec![start.to_string()];
    let mut visited = HashSet::new();
    while let Some(name) = stack.pop() {
        if !visited.insert(name.clone()) {
            continue;
        }
        if let Some(node) = graph.node(&name) {
            for input in node.inputs() {
                stack.push(input.clone());
            }
        } else {
            // Node not in graph — it's an external input sentinel
            result.insert(name);
        }
    }
    result
}

/// Prefix-cut stage localization: build the attention suffix graph rooted at
/// `ln_output` and verify that both `query` and `key` trace back to one shared
/// `_input` sentinel. Then run IBP and zonotope through the suffix to measure
/// per-stage widths.
///
/// Part of #318: shared-source prefix cut.
/// Design: designs/2026-03-11-issue-318-whisper-shared-source-prefix-cut.md
#[ntest::timeout(60000)]
#[test]
fn test_whisper_attention_prefix_cut_stage_localization_318() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    let case = load_prefix_cut_stage_localization_case(false);

    println!("\n=== Whisper Attention Prefix-Cut Stage Localization (#318) ===");
    println!("Block 0 ln_output shape: {:?}", case.ln_output.shape());
    print_input_width(&case.ln_output);

    let stage_nodes = localized_stage_nodes(
        &case.graph,
        case.scores_node.as_str(),
        case.softmax_node.as_str(),
        case.context_node.as_str(),
        case.output_node.as_str(),
    );
    assert_shared_prefix_topology(&case.graph, &stage_nodes, "prefix-cut");

    // Run per-stage measurements on the prefix-cut suffix graph
    for (stage_name, output_node, expected_shape) in [
        (
            "query",
            stage_nodes[0].1,
            vec![1, case.num_heads, 2, case.head_dim],
        ),
        (
            "key",
            stage_nodes[1].1,
            vec![1, case.num_heads, 2, case.head_dim],
        ),
        ("scores", stage_nodes[2].1, vec![1, case.num_heads, 2, 2]),
        ("softmax", stage_nodes[3].1, vec![1, case.num_heads, 2, 2]),
        (
            "context",
            stage_nodes[4].1,
            vec![1, case.num_heads, 2, case.head_dim],
        ),
        ("output", stage_nodes[5].1, vec![1, 2, case.hidden_dim]),
    ] {
        let (ibp, zonotope) = run_stage(&case.graph, output_node, &case.ln_output, stage_name);
        assert_sound_finite(&ibp, stage_name, "IBP", expected_shape.as_slice());
        assert_sound_finite(&zonotope, stage_name, "zonotope", expected_shape.as_slice());
        print_stage_widths(stage_name, output_node, &ibp, &zonotope);
    }
}
