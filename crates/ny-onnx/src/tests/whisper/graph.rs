// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::helpers::whisper_tiny_encoder;
use ny_core::LayerType;

#[ntest::timeout(10000)]
#[test]
fn test_encoder_layer_graph_extraction() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    let whisper = whisper_tiny_encoder();

    // Extract block 0 as a GraphNetwork
    let graph = whisper
        .encoder_layer_graph(0)
        .expect("Failed to extract graph");

    println!("\n=== Block 0 GraphNetwork ===");
    println!("Number of nodes: {}", graph.num_nodes());
    println!("Node names:");
    for name in graph.node_names() {
        let node = graph.node(name).unwrap();
        println!(
            "  {} ({}) <- {:?}",
            name,
            node.layer().layer_type(),
            node.inputs()
        );
    }

    // Verify the graph has a stable node count after filtering shape-computing ops.
    // Node count increased from 25-27 to 27-29 after #697 fix: data-carrying Concat
    // nodes with evaluated_constants are now correctly retained in the graph instead
    // of being skipped by the shape-computing Concat filter.
    let node_count = graph.num_nodes();
    assert!(
        (27..=29).contains(&node_count),
        "Block should have 27 to 29 nodes, got {}",
        node_count
    );

    // Verify topological sort works (no cycles)
    let sorted = graph
        .exec_order()
        .expect("Topological sort failed - graph may have cycles");
    assert_eq!(
        sorted.len(),
        node_count,
        "Sorted order should include all {} nodes",
        node_count
    );

    // Check that there are nodes with multiple inputs (the residual Add nodes)
    let multi_input_count = graph
        .node_names()
        .iter()
        .filter_map(|name| graph.node(name))
        .filter(|node| node.inputs().len() >= 2)
        .count();

    println!(
        "\nNodes with multiple inputs (DAG nodes): {}",
        multi_input_count
    );
    // We expect at least 2 residual Add nodes with 2 inputs each
    assert!(
        multi_input_count >= 2,
        "Expected at least 2 multi-input nodes for residuals"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_encoder_layer_graph_ibp() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    // GraphNetwork connectivity for Whisper blocks.
    //
    // This Whisper model uses post-norm architecture (LayerNorm after residual),
    // NOT pre-norm (LayerNorm before attention). The block structure is:
    //
    //   _input → Q/K/V projections (3 parallel paths) → Attention → Add(residual) → LayerNorm → MLP → Add
    //
    // Multiple nodes legitimately consume _input:
    // - Q projection path entry (AddConstant for bias)
    // - K projection (Linear)
    // - V projection (Linear)
    // - First residual Add (skip connection)
    //
    // The tensor_producer map traces through intermediate ONNX ops (Cast, Transpose, Reshape)
    // to ensure proper connectivity.

    let whisper = whisper_tiny_encoder();
    let graph = whisper
        .encoder_layer_graph(0)
        .expect("Failed to extract graph");

    // Count nodes that depend on network input vs other nodes
    let mut input_dependent_names = Vec::new();
    let input_dependent = graph
        .node_names()
        .iter()
        .filter_map(|name| graph.node(name))
        .filter(|node| {
            let has_input = node.inputs().iter().any(|i| i == "_input");
            if has_input {
                input_dependent_names.push(node.name().to_string());
            }
            has_input
        })
        .count();

    println!("\n=== Block 0 GraphNetwork Connectivity Analysis ===");
    println!("Total nodes: {}", graph.num_nodes());
    println!("Nodes with _input dependency: {}", input_dependent);
    println!("_input-dependent nodes: {:?}", input_dependent_names);
    println!(
        "Nodes with inter-node dependency: {}",
        graph.num_nodes() - input_dependent
    );

    // Post-norm Whisper: Q/K/V projections + residual add all consume _input directly.
    // Expected: 4 nodes with _input dependency (3 projection entries + 1 residual).
    assert!(
        (3..=5).contains(&input_dependent),
        "Expected 3-5 nodes with _input dependency (Q/K/V projections + residual), got {}",
        input_dependent
    );
    let node_count = graph.num_nodes();
    let inter_node = node_count - input_dependent;
    // Range widened from 20..=25 to 22..=27 after #697: data-carrying Concat
    // nodes are now retained, increasing total node count by ~2.
    assert!(
        (22..=27).contains(&inter_node),
        "Expected 22-27 nodes with inter-node dependencies, got {} (total {}, input-dependent {})",
        inter_node,
        node_count,
        input_dependent
    );
}

// Budget: shares the 33MB dynamo whisper fixture; the first test to touch the
// cached model/network pays the full debug-build load/convert cost inside its
// timer, which exceeds the old 10s budget under parallel suite load. 120s
// matches the heavy whisper siblings and still guards against hangs.
#[ntest::timeout(120000)]
#[test]
fn test_encoder_sequential_subcomponents() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    // Test that we can at least run IBP through individual sublayers
    use ndarray::ArrayD;
    use ny_propagate::BoundPropagation;
    use ny_tensor::BoundedTensor;

    let whisper = whisper_tiny_encoder();
    let block = whisper.encoder_layer(0).expect("Failed to extract block");

    let hidden_dim = whisper.hidden_dim;

    // Test the first layer (LayerNorm) in isolation
    if let Some(first_layer) = block.layers().first() {
        let input_data = ArrayD::from_elem(ndarray::IxDyn(&[hidden_dim]), 0.0f32);
        let input = BoundedTensor::from_epsilon(input_data, 0.1).expect("valid test input");

        println!(
            "\n=== Testing First Layer ({}) ===",
            first_layer.layer_type()
        );
        println!("Input shape: {:?}", input.shape());

        let output = first_layer
            .propagate_ibp(&input)
            .expect("First layer IBP should succeed");
        println!("Output shape: {:?}", output.shape());
        println!("Max width: {:.4}", output.max_width());
        assert_eq!(output.shape(), input.shape());
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_encoder_layer_graph_network_ibp() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    // Test GraphNetwork IBP on a Whisper encoder block
    // This verifies the connectivity fix enables DAG propagation
    use ndarray::ArrayD;
    use ny_tensor::BoundedTensor;

    let whisper = whisper_tiny_encoder();
    let graph = whisper
        .encoder_layer_graph_full(0)
        .expect("Failed to extract full graph");

    let hidden_dim = whisper.hidden_dim;

    // Create input tensor with Whisper block input shape [batch, seq, hidden].
    // Use small seq length for test speed.
    let batch = 1;
    let seq_len = 4;
    let input_data = ArrayD::from_elem(ndarray::IxDyn(&[batch, seq_len, hidden_dim]), 0.0f32);
    let input = BoundedTensor::from_epsilon(input_data, 0.01).expect("valid test input");

    println!("\n=== Testing GraphNetwork IBP on Whisper Block 0 ===");
    println!("Input shape: {:?}", input.shape());
    println!("Input epsilon: 0.01");

    let output = graph
        .propagate_ibp(&input)
        .expect("Full block IBP should succeed");
    println!("SUCCESS: Full block GraphNetwork IBP completed");
    println!("Output shape: {:?}", output.shape());
    println!("Max width: {:.6}", output.max_width());

    assert_eq!(output.shape(), &[batch, seq_len, hidden_dim]);

    // Verify bounds are sound (lower <= upper for all elements)
    let sound = output
        .lower()
        .iter()
        .zip(output.upper().iter())
        .all(|(l, u)| l <= u);
    assert!(sound, "Bounds must be sound (lower <= upper)");
}

// Budget: the block graph extraction works against the dynamo fixture since
// 2026-07, so IBP + CROWN actually run (CROWN completes in ~16s debug); the
// old 10s budget dated from when the test died fast at extraction. 120s
// matches the heavy whisper siblings and leaves headroom under parallel load.
#[ntest::timeout(120000)]
#[test]
fn test_encoder_layer_graph_network_crown_limitation() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    // LIMITATION: CROWN does not work on full Whisper blocks with N-D batched inputs.
    //
    // Root cause: LinearBounds assume flattened tensors where the weight matrix
    // operates on all elements. But transformer Linear layers operate per-position
    // (last dimension only), so a [384, 384] weight applied to [1, 4, 384] input
    // processes each of 4 positions independently.
    //
    // CROWN backward propagation through Linear expects:
    //   new_A = A @ W  where A is [output_dim, layer_output_dim], W is [layer_output_dim, layer_input_dim]
    //
    // For full block with [1, 4, 384] = 1536 elements:
    //   - A starts as identity [1536, 1536]
    //   - Linear weight is [384, 384] (per-position operation)
    //   - A @ W would need [1536, 1536] @ [384, 384] → dimension mismatch!
    //
    // Solutions:
    // 1. Implement N-D batched LinearBounds (significant refactor)
    // 2. Use compositional verification (verify subgraphs, compose bounds)
    // 3. Use IBP for full blocks (current approach - sound but loose)
    //
    // For now, we verify that:
    // - IBP works and produces sound bounds
    // - CROWN correctly fails with dimension mismatch (expected behavior)

    use ndarray::ArrayD;
    use ny_tensor::BoundedTensor;

    let whisper = whisper_tiny_encoder();
    let graph = whisper
        .encoder_layer_graph_full(0)
        .expect("Failed to extract full graph");

    let hidden_dim = whisper.hidden_dim;

    // Create input tensor with Whisper block input shape [batch, seq, hidden].
    let batch = 1;
    let seq_len = 4;
    let input_data = ArrayD::from_elem(ndarray::IxDyn(&[batch, seq_len, hidden_dim]), 0.0f32);
    let input = BoundedTensor::from_epsilon(input_data, 0.01).expect("valid test input");

    println!("\n=== Testing CROWN Limitation on N-D Batched Transformer Block ===");
    println!("Input shape: {:?}", input.shape());
    println!("Total elements: {} (batch * seq * hidden)", input.len());

    // IBP works - verify this
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("Full block IBP should succeed");
    println!("IBP max width: {:.6e}", ibp_output.max_width());

    // Verify IBP bounds are sound
    let sound = ibp_output
        .lower()
        .iter()
        .zip(ibp_output.upper().iter())
        .all(|(l, u)| l <= u);
    assert!(sound, "IBP bounds must be sound");

    // CROWN is expected to fail due to N-D batched dimension mismatch
    let crown_result = std::panic::catch_unwind(|| graph.propagate_crown(&input));

    match crown_result {
        Ok(Ok(bounds)) => {
            // If CROWN somehow succeeds, that's fine - we've improved!
            println!("CROWN succeeded (unexpected - may have been fixed)");
            let sound = bounds
                .lower()
                .iter()
                .zip(bounds.upper().iter())
                .all(|(l, u)| l <= u);
            assert!(sound, "CROWN bounds must be sound");
            assert_eq!(bounds.shape(), input.shape());
        }
        Ok(Err(e)) => {
            println!("CROWN failed with error (expected): {:?}", e);
        }
        Err(_) => {
            // Panic (dimension mismatch in ndarray) is expected behavior for now
            println!("CROWN panicked due to dimension mismatch (expected limitation)");
        }
    }

    // Key insight: IBP gives very loose widths for full blocks.
    // Compositional verification of subgraphs gives much tighter bounds.
    // Full block bound explosion comes from composing these loose bounds sequentially.
    println!("\nRecommendation: Use compositional verification for tighter bounds");
}

#[ntest::timeout(10000)]
#[test]
fn test_encoder_mlp_subpath_ibp() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    // Test IBP on just the MLP subpath of a Whisper encoder block.
    // This demonstrates compositional verification: verify subcomponents that work.
    //
    // MLP path: LayerNorm → Linear → GELU → Linear → Add (bias)
    // No shape transformations, should work with [seq, hidden] input.
    use ndarray::ArrayD;
    use ny_propagate::{GraphNetwork, GraphNode};
    use ny_tensor::BoundedTensor;

    let whisper = whisper_tiny_encoder();

    // Extract just MLP layers from block 0 using layer-type ordering.
    // MLP sequence: LayerNorm -> MatMul -> Add -> GELU -> MatMul -> Add.
    let block_layers = whisper
        .block_layers_for_index(0)
        .expect("block layers should resolve");
    let ln_indices: Vec<usize> = block_layers
        .iter()
        .enumerate()
        .filter(|(_, spec)| spec.layer_type == LayerType::LayerNorm)
        .map(|(idx, _)| idx)
        .collect();
    assert!(
        ln_indices.len() >= 2,
        "Expected at least 2 LayerNorms in the block"
    );
    let mlp_start = *ln_indices.last().unwrap();
    let expected_types = [
        LayerType::LayerNorm,
        LayerType::MatMul,
        LayerType::Add,
        LayerType::GELU,
        LayerType::MatMul,
        LayerType::Add,
    ];

    // Build MLP subgraph
    let mut mlp_graph = GraphNetwork::new();
    let mut prev_node: Option<String> = None;

    for (offset, expected) in expected_types.iter().enumerate() {
        let spec = block_layers
            .get(mlp_start + offset)
            .unwrap_or_else(|| panic!("Missing MLP layer at offset {}", offset));
        assert_eq!(
            spec.layer_type, *expected,
            "MLP layer type mismatch at offset {}: expected {:?}, got {:?}",
            offset, expected, spec.layer_type
        );

        let layer = whisper
            .model
            .convert_layer(spec)
            .expect("Failed to convert layer");

        // Sequential input: previous node or graph input
        let inputs = match &prev_node {
            Some(name) => vec![name.clone()],
            None => vec!["_input".to_string()],
        };

        let node = GraphNode::new(spec.name.clone(), layer, inputs);
        mlp_graph.add_node(node);
        prev_node = Some(spec.name.clone());
    }

    if let Some(output_name) = prev_node {
        mlp_graph.set_output(&output_name);
    }

    println!("\n=== Testing MLP Subpath IBP ===");
    println!("MLP graph has {} nodes", mlp_graph.num_nodes());
    assert_eq!(
        mlp_graph.num_nodes(),
        expected_types.len(),
        "Expected all MLP layers to be captured"
    );

    // Create input with [seq, hidden] shape
    let hidden_dim = whisper.hidden_dim;
    let seq_len = 4;
    let input_data = ArrayD::from_elem(ndarray::IxDyn(&[seq_len, hidden_dim]), 0.0f32);
    let input = BoundedTensor::from_epsilon(input_data, 0.01).expect("valid test input");

    println!("Input shape: {:?}", input.shape());

    match mlp_graph.propagate_ibp(&input) {
        Ok(output) => {
            println!("SUCCESS: MLP subpath IBP completed");
            println!("Output shape: {:?}", output.shape());
            println!("Max width: {:.6}", output.max_width());

            // Output should be [seq, hidden] (after projection back)
            assert_eq!(output.shape()[0], seq_len);

            // Bounds should be sound
            let sound = output
                .lower()
                .iter()
                .zip(output.upper().iter())
                .all(|(l, u)| l <= u);
            assert!(sound, "Bounds must be sound");
        }
        Err(e) => {
            println!("MLP IBP failed: {:?}", e);
            // Print graph structure for debugging
            println!("\nGraph structure:");
            for node in mlp_graph.node_names() {
                println!("  {}", node);
            }
            panic!("MLP subpath should work without shape transformations");
        }
    }
}
