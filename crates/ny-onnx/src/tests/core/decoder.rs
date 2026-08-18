// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ny_core::{LayerType, NyError};
use ny_propagate::BoundPropagation;

#[ntest::timeout(10000)]
#[test]
fn test_load_decoder_block() {
    let path = require_test_model_with_hint("decoder_block.onnx", TRANSFORMER_TEST_MODEL_HINT);

    let model = load_onnx(&path).expect("Failed to load decoder_block model");

    // Check that we have a good number of layers
    assert!(
        model.network.layers.len() >= 3,
        "Expected at least 3 layers in decoder block, got {}",
        model.network.layers.len()
    );

    // Check for expected transformer components
    let layer_types: Vec<_> = model.network.layers.iter().map(|l| &l.layer_type).collect();
    println!("Decoder block layer types: {:?}", layer_types);

    // The fixture contains LayerNorm and an exact Erf decomposition of GELU.
    let has_layer_norm = model
        .network
        .layers
        .iter()
        .any(|l| l.layer_type == LayerType::LayerNorm);
    let has_erf = model
        .network
        .layers
        .iter()
        .any(|l| l.layer_type == LayerType::Erf);

    // Decoder should have transformer components
    let transformer_markers = [has_layer_norm, has_erf].iter().filter(|&&x| x).count();
    assert!(
        transformer_markers >= 1,
        "Expected at least 1 transformer marker (LayerNorm/Erf)"
    );
    assert!(has_erf, "decoder block must preserve decomposed GELU Erf");
}

#[ntest::timeout(10000)]
#[test]
fn test_decoder_block_structure() {
    // Test that decoder block loads correctly and has expected structure.
    // Note: Full E2E verification requires compositional approach due to
    // MatMul of two bounded tensors in attention (Q@K^T).
    let path = require_test_model_with_hint("decoder_block.onnx", TRANSFORMER_TEST_MODEL_HINT);

    let model = load_onnx(&path).expect("Failed to load decoder block model");
    let network = model
        .to_propagate_network()
        .expect("Failed to convert to propagate network");

    println!("\n=== Decoder Block Structure Test ===");
    println!("Network has {} layers", network.layers().len());

    // Print layer types
    for (i, layer) in network.layers().iter().enumerate() {
        println!("  Layer {}: {:?}", i, layer.layer_type());
    }

    // Verify expected layer types are present
    let has_layer_norm = network
        .layers()
        .iter()
        .any(|l| l.layer_type() == "LayerNorm");
    let has_causal_softmax = network
        .layers()
        .iter()
        .any(|l| l.layer_type() == "CausalSoftmax");
    let has_softmax = network.layers().iter().any(|l| l.layer_type() == "Softmax");
    let has_erf = network.layers().iter().any(|l| l.layer_type() == "Erf");
    // Since c93afde62, all activation-activation MatMuls produce BilinearCrown.
    let has_bilinear = network
        .layers()
        .iter()
        .any(|l| l.layer_type() == "BilinearCrown");

    println!("\nHas LayerNorm: {}", has_layer_norm);
    println!("Has CausalSoftmax: {}", has_causal_softmax);
    println!("Has Softmax: {}", has_softmax);
    println!("Has Erf: {}", has_erf);
    println!("Has BilinearCrown: {}", has_bilinear);

    assert!(has_layer_norm, "Decoder block should have LayerNorm");
    assert!(!has_causal_softmax, "finite mask must not hard-fuse");
    assert!(
        has_softmax,
        "decoder block should preserve ordinary Softmax"
    );
    assert!(has_erf, "Decoder block should preserve decomposed GELU Erf");
    assert!(
        has_bilinear,
        "Decoder block should have BilinearCrown (for attention MatMul)"
    );

    // Test IBP through pre-attention layers (LayerNorm + Q/K/V projections)
    // These are the first 7 layers before the attention MatMul
    use ndarray::ArrayD;
    use ny_tensor::BoundedTensor;

    let batch = 1;
    let seq = 4;
    let hidden = 4;
    let epsilon = 1e-3;

    let input_data = ArrayD::from_elem(IxDyn(&[batch, seq, hidden]), 0.0f32);
    let input = BoundedTensor::from_epsilon(input_data, epsilon).expect("valid test input");

    // Propagate through just LayerNorm (layer 0)
    let layer_norm = &network.layers()[0];
    let ln_output = layer_norm
        .propagate_ibp(&input)
        .expect("LayerNorm IBP failed");
    println!("\nLayerNorm output shape: {:?}", ln_output.shape());
    println!("LayerNorm output max_width: {:.2e}", ln_output.max_width());

    // LayerNorm should preserve shape
    assert_eq!(ln_output.shape(), input.shape());
}

#[ntest::timeout(10000)]
#[test]
fn test_decoder_compositional_verification_fails_closed() {
    // Subgraph inspection remains available, while proof-looking decoder
    // compatibility APIs fail closed.
    use ndarray::ArrayD;
    use ny_tensor::BoundedTensor;

    let path = require_test_model_with_hint("decoder_block.onnx", TRANSFORMER_TEST_MODEL_HINT);

    println!("\n=== Decoder Compositional Verification Test ===");

    // Load decoder model using new API
    let decoder = load_decoder(&path).expect("Failed to load decoder model");

    println!("Decoder structure:");
    println!("  Num blocks: {}", decoder.num_blocks);
    println!("  Hidden dim: {}", decoder.hidden_dim);
    println!("  Num heads: {}", decoder.num_heads);
    println!("  Head dim: {}", decoder.structure.head_dim);

    // Create input tensor matching the test model's dimensions
    // decoder_block.onnx uses hidden_dim=4 (test model with 4 heads, head_dim=1)
    let batch = 1;
    let seq = 4;
    let hidden = decoder.hidden_dim;
    let epsilon = 1e-3;

    let input_data = ArrayD::from_elem(IxDyn(&[batch, seq, hidden]), 0.0f32);
    let input = BoundedTensor::from_epsilon(input_data, epsilon).expect("valid test input");

    println!("\nInput:");
    println!("  Shape: {:?}", input.shape());
    println!("  Epsilon: {:.2e}", epsilon);
    println!("  Max width: {:.2e}", input.max_width());

    // Test subgraph extraction
    println!("\nTesting subgraph extraction...");

    let attn_graph = decoder
        .causal_attention_subgraph(0)
        .expect("Causal attention subgraph extraction should succeed");
    println!(
        "  Causal attention subgraph: {} nodes",
        attn_graph.num_nodes()
    );

    let mlp_graph = decoder
        .mlp_subgraph(0)
        .expect("MLP subgraph extraction should succeed");
    println!("  MLP subgraph: {} nodes", mlp_graph.num_nodes());

    for result in [
        decoder.causal_attention_subgraph(decoder.num_blocks),
        decoder.mlp_subgraph(decoder.num_blocks),
    ] {
        assert!(
            matches!(result, Err(NyError::InvalidSpec(message)) if message.contains("not found")),
            "an out-of-range block must not alias an unprefixed block-zero graph"
        );
    }

    for (label, result) in [
        (
            "single-block API",
            decoder.verify_block_compositional(0, &input).map(|_| ()),
        ),
        (
            "sequential API",
            decoder.verify_sequential(&input, 0, 1).map(|_| ()),
        ),
    ] {
        match result {
            Err(NyError::UnsupportedConfiguration(message)) => {
                assert!(
                    message.contains("no bounds or verification details were produced"),
                    "{label}: unexpected error: {message}"
                );
            }
            Err(other) => panic!("{label}: expected UnsupportedConfiguration, got {other:?}"),
            Ok(()) => panic!("{label}: unavailable decoder verification returned bounds"),
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_load_decoder_function() {
    // Test the load_decoder function with various models
    let decoder_path =
        require_test_model_with_hint("decoder_block.onnx", TRANSFORMER_TEST_MODEL_HINT);
    let decoder = load_decoder(&decoder_path);
    assert!(
        decoder.is_ok(),
        "load_decoder should succeed for decoder_block.onnx"
    );
    let decoder = decoder.unwrap();
    assert!(decoder.num_blocks >= 1, "Should have at least 1 block");
    assert!(decoder.hidden_dim > 0, "Should have positive hidden dim");
    println!(
        "Loaded decoder_block.onnx: {} blocks, hidden={}",
        decoder.num_blocks, decoder.hidden_dim
    );

    let enc_dec_path =
        require_test_model_with_hint("encoder_decoder_block.onnx", TRANSFORMER_TEST_MODEL_HINT);
    let decoder = load_decoder(&enc_dec_path);
    assert!(
        decoder.is_ok(),
        "load_decoder should succeed for encoder_decoder_block.onnx"
    );
    let decoder = decoder.unwrap();
    // Check if cross-attention was detected
    let has_cross = decoder
        .structure
        .blocks
        .iter()
        .any(|b| b.has_cross_attention);
    println!(
        "Loaded encoder_decoder_block.onnx: has_cross_attention={}",
        has_cross
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_decoder_gpu_verification_fails_closed() {
    // The accelerated compatibility surface must fail before device work.
    use ndarray::ArrayD;
    use ny_tensor::BoundedTensor;

    let path = require_test_model_with_hint("decoder_block.onnx", TRANSFORMER_TEST_MODEL_HINT);

    let decoder = load_decoder(&path).expect("Failed to load decoder");
    println!(
        "Loaded decoder: {} blocks, hidden={}, heads={}",
        decoder.num_blocks, decoder.hidden_dim, decoder.num_heads
    );

    // Create small test input
    let batch = 1;
    let seq = 4;
    let hidden = decoder.hidden_dim;
    let shape = vec![batch, seq, hidden];
    let eps = 0.01;

    let center = ArrayD::from_elem(IxDyn(&shape), 0.0_f32);
    let lower = center.mapv(|v| v - eps);
    let upper = center.mapv(|v| v + eps);
    let input = BoundedTensor::new(lower, upper).expect("Failed to create input");

    match decoder.verify_block_compositional_gpu(0, &input, None) {
        Err(NyError::UnsupportedConfiguration(message)) => {
            assert!(message.contains("no bounds or verification details were produced"));
        }
        Err(other) => panic!("expected UnsupportedConfiguration, got {other:?}"),
        Ok(_) => panic!("unavailable accelerated decoder verification returned bounds"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_decoder_sequential_gpu_fails_closed() {
    use ndarray::ArrayD;
    use ny_tensor::BoundedTensor;

    let path = require_test_model_with_hint("decoder_block.onnx", TRANSFORMER_TEST_MODEL_HINT);

    let decoder = load_decoder(&path).expect("Failed to load decoder");

    // Only test with single block to avoid overflow
    assert!(
        decoder.num_blocks >= 1,
        "Decoder should have at least 1 block"
    );

    let batch = 1;
    let seq = 4;
    let hidden = decoder.hidden_dim;
    let shape = vec![batch, seq, hidden];
    let eps = 0.01;

    let center = ArrayD::from_elem(IxDyn(&shape), 0.0_f32);
    let lower = center.mapv(|v| v - eps);
    let upper = center.mapv(|v| v + eps);
    let input = BoundedTensor::new(lower, upper).expect("Failed to create input");

    match decoder.verify_sequential_gpu(&input, 0, 1, None) {
        Err(NyError::UnsupportedConfiguration(message)) => {
            assert!(message.contains("no bounds or verification details were produced"));
        }
        Err(other) => panic!("expected UnsupportedConfiguration, got {other:?}"),
        Ok(_) => panic!("unavailable sequential accelerated decoder verification returned bounds"),
    }
}

// ================= SwiGLU MLP Subgraph Tests (#3181) =================
//
// Tests for the mlp_subgraph dispatcher: GELU backward compat, SwiGLU detection,
// and structural fallback.
// Reference: designs/2026-03-01-swiglu-mlp-decoder-block.md

/// Helper to build a minimal DecoderModel from LayerSpecs + WeightStore.
///
/// Constructs an OnnxModel with the given layers and weights, wraps it in a
/// DecoderModel with single-block structure. Useful for testing subgraph
/// extraction without needing a real ONNX file.
fn mock_decoder_model(
    layers: Vec<LayerSpec>,
    weights: WeightStore,
    hidden_dim: usize,
) -> crate::decoder::DecoderModel {
    use crate::decoder::{DecoderBlockInfo, DecoderModel, DecoderStructure};

    let network = Network {
        name: "test_decoder".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1, 4, hidden_dim as i64],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "output".to_string(),
            shape: vec![1, 4, hidden_dim as i64],
            dtype: DataType::Float32,
        }],
        layers,
        param_count: 0,
    };

    let model = OnnxModel::empty_with_network(network, weights);

    DecoderModel {
        model,
        structure: DecoderStructure {
            blocks: vec![DecoderBlockInfo {
                index: 0,
                has_cross_attention: false,
            }],
            num_heads: 2,
            hidden_dim,
            head_dim: hidden_dim / 2,
        },
        num_blocks: 1,
        hidden_dim,
        num_heads: 2,
    }
}

/// Helper to create a MatMul LayerSpec with a weight as second input.
fn matmul_spec(name: &str, act_input: &str, weight_name: &str, output: &str) -> LayerSpec {
    LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::MatMul,
        inputs: vec![act_input.to_string(), weight_name.to_string()],
        outputs: vec![output.to_string()],
        weights: None,
        attributes: std::collections::HashMap::new(),
    }
}

/// Helper to create a SiLU LayerSpec.
fn silu_spec(name: &str, input: &str, output: &str) -> LayerSpec {
    LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::SiLU,
        inputs: vec![input.to_string()],
        outputs: vec![output.to_string()],
        weights: None,
        attributes: std::collections::HashMap::new(),
    }
}

/// Helper to create a binary Mul LayerSpec (two activation inputs).
fn mul_binary_spec(name: &str, input_a: &str, input_b: &str, output: &str) -> LayerSpec {
    LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::Mul,
        inputs: vec![input_a.to_string(), input_b.to_string()],
        outputs: vec![output.to_string()],
        weights: None,
        attributes: std::collections::HashMap::new(),
    }
}

/// Helper to insert a 2D weight matrix into a WeightStore.
fn insert_weight(ws: &mut WeightStore, name: &str, rows: usize, cols: usize) {
    ws.insert(
        name.to_string(),
        ArrayD::from_shape_vec(IxDyn(&[rows, cols]), vec![0.1; rows * cols]).unwrap(),
    );
}

/// Assert MLP subgraph IBP produces ordered, NaN-free bounds.
fn assert_mlp_ibp_well_formed(graph: &ny_propagate::GraphNetwork, hidden: usize) {
    use ndarray::ArrayD;
    use ny_tensor::BoundedTensor;

    let input_data = ArrayD::from_elem(IxDyn(&[1, 4, hidden]), 0.5f32);
    let input = BoundedTensor::from_epsilon(input_data, 0.1).expect("valid test input");

    let output = graph
        .propagate_ibp(&input)
        .expect("MLP subgraph IBP should succeed");

    let ordered = output
        .lower()
        .iter()
        .zip(output.upper().iter())
        .all(|(l, u)| l <= u);
    assert!(ordered, "MLP output bounds must be ordered");

    let has_nan =
        output.lower().iter().any(|x| x.is_nan()) || output.upper().iter().any(|x| x.is_nan());
    assert!(!has_nan, "MLP output should not contain NaN");
}

#[ntest::timeout(10000)]
#[test]
fn test_mlp_subgraph_gelu_backward_compat() {
    // Verify that the existing decomposed-GELU decoder model still produces
    // the same MLP subgraph through the dispatcher without canonical fusion.
    let path = require_test_model_with_hint("decoder_block.onnx", TRANSFORMER_TEST_MODEL_HINT);
    let decoder = load_decoder(&path).expect("Failed to load decoder model");

    let mlp_graph = decoder
        .mlp_subgraph(0)
        .expect("MLP subgraph extraction should succeed after refactor");

    // The GELU-shaped path should include its exact Erf primitive.
    assert!(
        mlp_graph.num_nodes() >= 2,
        "decomposed GELU MLP should have at least 2 nodes (fc1 + fc2), got {}",
        mlp_graph.num_nodes()
    );
    assert!(
        mlp_graph
            .node_names()
            .iter()
            .filter_map(|name| mlp_graph.node(name))
            .any(|node| matches!(node.layer(), ny_propagate::Layer::Erf(_))),
        "decomposed GELU MLP subgraph must preserve its Erf primitive"
    );

    // Verify IBP works through the extracted subgraph
    use ndarray::ArrayD;
    use ny_tensor::BoundedTensor;

    let hidden = decoder.hidden_dim;
    let input_data = ArrayD::from_elem(IxDyn(&[1, 4, hidden]), 0.0f32);
    let input = BoundedTensor::from_epsilon(input_data, 1e-3).expect("valid test input");

    let mlp_output = mlp_graph
        .propagate_ibp(&input)
        .expect("decomposed GELU MLP subgraph IBP should succeed");

    // Bounds must be ordered (lower <= upper everywhere).
    let ordered = mlp_output
        .lower()
        .iter()
        .zip(mlp_output.upper().iter())
        .all(|(l, u)| l <= u);
    assert!(ordered, "decomposed GELU MLP output bounds must be ordered");

    // No NaN in output
    let has_nan = mlp_output.lower().iter().any(|x| x.is_nan())
        || mlp_output.upper().iter().any(|x| x.is_nan());
    assert!(
        !has_nan,
        "decomposed GELU MLP output should not contain NaN"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_mlp_subgraph_swiglu_detection() {
    // Build a mock model with SwiGLU naming convention (gate_proj, up_proj, down_proj)
    // and verify mlp_subgraph() dispatches to the SwiGLU path.
    let hidden = 4;
    let mlp_dim = 8;

    let layers = vec![
        matmul_spec(
            "/mlp/gate_proj/MatMul",
            "norm2_out",
            "gate_weight",
            "gate_out",
        ),
        silu_spec("/mlp/Silu", "gate_out", "silu_out"),
        matmul_spec("/mlp/up_proj/MatMul", "norm2_out", "up_weight", "up_out"),
        mul_binary_spec("/mlp/Mul", "silu_out", "up_out", "mul_out"),
        matmul_spec(
            "/mlp/down_proj/MatMul",
            "mul_out",
            "down_weight",
            "down_out",
        ),
    ];

    let mut weights = WeightStore::new();
    insert_weight(&mut weights, "gate_weight", hidden, mlp_dim);
    insert_weight(&mut weights, "up_weight", hidden, mlp_dim);
    insert_weight(&mut weights, "down_weight", mlp_dim, hidden);

    let decoder = mock_decoder_model(layers, weights, hidden);
    let mlp_graph = decoder
        .mlp_subgraph(0)
        .expect("SwiGLU MLP subgraph extraction should succeed");

    assert_eq!(
        mlp_graph.num_nodes(),
        5,
        "SwiGLU: gate, SiLU, up, Mul, down"
    );
    assert_mlp_ibp_well_formed(&mlp_graph, hidden);
}

#[ntest::timeout(10000)]
#[test]
fn test_mlp_subgraph_swiglu_alt_naming() {
    // Test SwiGLU with alternative naming (w1/w3/w2 instead of gate_proj/up_proj/down_proj).
    let hidden = 4;
    let mlp_dim = 8;

    let layers = vec![
        matmul_spec("/mlp/w1/MatMul", "norm2_out", "w1_weight", "w1_out"),
        silu_spec("/mlp/Silu", "w1_out", "silu_out"),
        matmul_spec("/mlp/w3/MatMul", "norm2_out", "w3_weight", "w3_out"),
        mul_binary_spec("/mlp/Mul", "silu_out", "w3_out", "mul_out"),
        matmul_spec("/mlp/w2/MatMul", "mul_out", "w2_weight", "w2_out"),
    ];

    let mut weights = WeightStore::new();
    insert_weight(&mut weights, "w1_weight", hidden, mlp_dim);
    insert_weight(&mut weights, "w3_weight", hidden, mlp_dim);
    insert_weight(&mut weights, "w2_weight", mlp_dim, hidden);

    let decoder = mock_decoder_model(layers, weights, hidden);
    let mlp_graph = decoder
        .mlp_subgraph(0)
        .expect("SwiGLU (w1/w3/w2) subgraph extraction should succeed");

    assert_eq!(mlp_graph.num_nodes(), 5, "SwiGLU alt naming: 5 nodes");
    assert_mlp_ibp_well_formed(&mlp_graph, hidden);
}

#[ntest::timeout(10000)]
#[test]
fn test_mlp_subgraph_structural_fallback() {
    // Test structural fallback when neither GELU nor SwiGLU naming is detected.
    let hidden = 4;
    let mlp_dim = 8;

    let layers = vec![
        matmul_spec("/mlp/linear1/MatMul", "norm2_out", "l1_weight", "l1_out"),
        silu_spec("/mlp/activation/Silu", "l1_out", "act_out"),
        matmul_spec("/mlp/linear2/MatMul", "act_out", "l2_weight", "l2_out"),
    ];

    let mut weights = WeightStore::new();
    insert_weight(&mut weights, "l1_weight", hidden, mlp_dim);
    insert_weight(&mut weights, "l2_weight", mlp_dim, hidden);

    let decoder = mock_decoder_model(layers, weights, hidden);
    let mlp_graph = decoder
        .mlp_subgraph(0)
        .expect("Structural fallback extraction should succeed");

    assert_eq!(mlp_graph.num_nodes(), 3, "Structural fallback: 3 nodes");
}
