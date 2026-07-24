// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use super::helpers::{whisper_tiny_encoder, whisper_tiny_propagate_network};
use ny_core::LayerType;
use ny_propagate::layers::GELULayer;
use ny_propagate::Layer as PropLayer;
use ny_propagate::{BoundPropagation, Network as PropNetwork};
use ny_tensor::BoundedTensor;

#[ntest::timeout(10000)]
#[test]
fn test_whisper_param_count() {
    // Create a minimal WhisperModel for testing
    let model = WhisperModel {
        model: OnnxModel {
            network: Network {
                name: "test".to_string(),
                inputs: vec![],
                outputs: vec![],
                layers: vec![],
                param_count: 0,
            },
            weights: WeightStore::new(),
            tensor_producer: std::collections::HashMap::new(),
            constant_tensors: std::collections::HashSet::new(),
            tensor_shapes: std::collections::HashMap::new(),
            original_float32_initializers: std::collections::HashMap::new(),
            original_network_topology: None,
            opset_imports: std::collections::HashMap::new(),
        },
        structure: WhisperEncoderStructure {
            stem_end_idx: 0,
            blocks: vec![],
            ln_post_start_idx: 0,
        },
        encoder_layers: 4,
        decoder_layers: 4,
        hidden_dim: 384,
        num_heads: 6,
    };

    assert_eq!(model.encoder_layers, 4);
    assert_eq!(model.hidden_dim, 384);
    assert_eq!(model.param_count(), 0);
}

// =========================================================================
// Transformer Model Tests
// =========================================================================

#[ntest::timeout(300000)]
#[test]
fn test_whisper_tiny_layernorm_fusion() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    // Test that LayerNorm fusion works on Whisper-tiny encoder
    let model = &whisper_tiny_encoder().model;

    // Count layer types
    let mut layer_norm_count = 0;
    let mut softmax_count = 0;
    let mut gelu_count = 0;
    let mut matmul_count = 0;
    let mut add_count = 0;
    let mut linear_count = 0;
    let mut conv_count = 0;

    for layer in &model.network.layers {
        match layer.layer_type {
            LayerType::LayerNorm => layer_norm_count += 1,
            LayerType::Softmax => softmax_count += 1,
            LayerType::GELU => gelu_count += 1,
            LayerType::MatMul => matmul_count += 1,
            LayerType::Add => add_count += 1,
            LayerType::Linear => linear_count += 1,
            LayerType::Conv1d => conv_count += 1,
            _ => {}
        }
    }

    println!("\n=== Whisper-tiny Encoder Statistics ===");
    println!("Total layers: {}", model.network.layers.len());
    println!("  LayerNorm (fused): {}", layer_norm_count);
    println!("  Softmax: {}", softmax_count);
    println!("  GELU (fused): {}", gelu_count);
    println!("  MatMul: {}", matmul_count);
    println!("  Add: {}", add_count);
    println!("  Linear: {}", linear_count);
    println!("  Conv1d: {}", conv_count);

    // Whisper-tiny encoder has 4 transformer blocks
    // Each block has:
    // - 2 LayerNorms (pre-attention and pre-FFN)
    // - 1 attention with softmax
    // - 1 FFN with GELU
    // Plus initial LayerNorm before first block = 2*4 + 1 = 9 LayerNorms
    // But the exact count depends on ONNX export

    // Test that we fused at least some LayerNorms
    assert!(
        layer_norm_count > 0,
        "Expected LayerNorm fusion to detect at least one LayerNorm in Whisper encoder"
    );

    // Test softmax count (one per attention layer)
    assert!(
        softmax_count >= 4,
        "Expected at least 4 Softmax layers (one per attention), got {}",
        softmax_count
    );

    // Test GELU count (one per FFN)
    assert!(
        gelu_count >= 4,
        "Expected at least 4 GELU activations (one per FFN), got {}",
        gelu_count
    );

    // Print fusion ratio
    let total_onnx_nodes = model.network.layers.len();
    let fused_ops = layer_norm_count + gelu_count;
    println!("\nFusion statistics:");
    println!("  Fused layer types: {} (LayerNorm + GELU)", fused_ops);
    println!("  Total layers after fusion: {}", total_onnx_nodes);
}

// Budget: shares the 33MB dynamo whisper fixture; the first test to touch the
// cached model/network pays the full debug-build load/convert cost inside its
// timer, which exceeds the old 10s budget under parallel suite load. 120s
// matches the heavy whisper siblings and still guards against hangs.
#[ntest::timeout(120000)]
#[test]
fn test_whisper_tiny_propagate_network_conversion() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    // Test that the Whisper model can be converted to a propagate network
    let model = &whisper_tiny_encoder().model;
    let result = model.to_propagate_network();

    // Conversion should now succeed with Conv1d support
    let network = result.expect("Failed to convert - Conv1d should be supported");

    println!("\nSuccessfully converted to propagate network!");
    println!("Total layers: {}", network.layers().len());

    // Count converted layer types
    let mut conv1d_count = 0;
    let mut linear_count = 0;
    let mut layer_norm_count = 0;
    let mut softmax_count = 0;
    let mut gelu_count = 0;
    let mut matmul_count = 0;
    let mut add_count = 0;

    for layer in network.layers() {
        match layer {
            PropLayer::Conv1d(_) => conv1d_count += 1,
            PropLayer::Linear(_) => linear_count += 1,
            PropLayer::LayerNorm(_) => layer_norm_count += 1,
            PropLayer::Softmax(_) => softmax_count += 1,
            PropLayer::GELU(_) => gelu_count += 1,
            PropLayer::MatMul(_) => matmul_count += 1,
            PropLayer::Add(_) => add_count += 1,
            _ => {}
        }
    }

    println!("Converted layers:");
    println!("  Conv1d: {}", conv1d_count);
    println!("  Linear: {}", linear_count);
    println!("  MatMul: {}", matmul_count);
    println!("  Add: {}", add_count);
    println!("  LayerNorm: {}", layer_norm_count);
    println!("  Softmax: {}", softmax_count);
    println!("  GELU: {}", gelu_count);

    // Verify we have the expected layers
    assert_eq!(
        conv1d_count, 2,
        "Expected 2 Conv1d layers in Whisper encoder"
    );
    assert!(
        linear_count > 0,
        "Expected Linear layers in Whisper encoder"
    );
    assert!(layer_norm_count > 0, "Expected LayerNorm layers");
    assert!(softmax_count > 0, "Expected Softmax layers for attention");
    assert!(gelu_count > 0, "Expected GELU activations");
}

// The Conv1d IBP itself is milliseconds; the budget is dominated by the shared
// whisper_tiny_propagate_network() OnceLock conversion (~6s debug), which can
// exceed 10s under a parallel full-workspace run. Same ~10x margin convention
// as the 120s/~30s timeouts elsewhere in this suite.
#[ntest::timeout(60000)]
#[test]
fn test_whisper_conv1d_ibp() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    // Test IBP propagation through the first Conv1d layer of Whisper
    let network = whisper_tiny_propagate_network();

    // Find and test the first Conv1d layer
    let first_conv1d = network
        .layers()
        .iter()
        .find(|l| matches!(l, PropLayer::Conv1d(_)))
        .expect("Expected Conv1d layer");

    // Create a small test input: (80 channels, 16 time steps)
    // Whisper expects 80 mel spectrogram channels
    let lower_data = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[80, 16]), -1.0f32);
    let upper_data = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[80, 16]), 1.0f32);
    let input = BoundedTensor::new(lower_data, upper_data).unwrap();

    // Propagate through the Conv1d
    let output = first_conv1d.propagate_ibp(&input).expect("IBP failed");

    println!("\nWhisper Conv1d IBP test:");
    println!("  Input shape: {:?}", input.shape());
    println!("  Output shape: {:?}", output.shape());

    // Verify output shape: Conv1d(80, 384, kernel=3, stride=1, padding=1)
    // With padding=1: output_len = (16 + 2*1 - 3) / 1 + 1 = 16
    assert_eq!(output.shape()[0], 384, "Expected 384 output channels");
    assert_eq!(
        output.shape()[1],
        16,
        "Expected same time dimension with padding"
    );

    // Verify bounds are sound (lower <= upper)
    for (l, u) in output.lower().iter().zip(output.upper().iter()) {
        assert!(l <= u, "Unsound bounds: lower {} > upper {}", l, u);
    }

    // Verify bounds are finite
    assert!(
        output.lower().iter().all(|&v| v.is_finite()),
        "Non-finite lower bounds"
    );
    assert!(
        output.upper().iter().all(|&v| v.is_finite()),
        "Non-finite upper bounds"
    );

    println!(
        "  Lower bound range: [{:.4}, {:.4}]",
        output.lower().iter().cloned().reduce(f32::min).unwrap(),
        output.lower().iter().cloned().reduce(f32::max).unwrap()
    );
    println!(
        "  Upper bound range: [{:.4}, {:.4}]",
        output.upper().iter().cloned().reduce(f32::min).unwrap(),
        output.upper().iter().cloned().reduce(f32::max).unwrap()
    );
}

#[ntest::timeout(120000)] // ~30s runtime; 4x margin for CI variability.
#[test]
fn test_whisper_first_layers_ibp() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    // Test IBP through Conv1d -> GELU sequence (first few layers)
    let network = whisper_tiny_propagate_network();

    // Create a small sequential network with just Conv1d + GELU
    let mut small_network = PropNetwork::new();

    // Find and add the first Conv1d
    for layer in network.layers() {
        if let PropLayer::Conv1d(c) = layer {
            small_network.add_layer(PropLayer::Conv1d(c.clone()));
            break;
        }
    }

    // Add a GELU after it
    small_network.add_layer(PropLayer::GELU(GELULayer::default()));

    // Create test input
    // Keep the sequence length short to keep IBP runtime bounded.
    let lower_data = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[80, 20]), -1.0f32);
    let upper_data = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[80, 20]), 1.0f32);
    let input = BoundedTensor::new(lower_data, upper_data).unwrap();

    // Propagate
    let output = small_network.propagate_ibp(&input).expect("IBP failed");

    println!("\nWhisper Conv1d -> GELU IBP test:");
    println!("  Input shape: {:?}", input.shape());
    println!("  Output shape: {:?}", output.shape());
    println!(
        "  Lower bound range: [{:.4}, {:.4}]",
        output.lower().iter().cloned().reduce(f32::min).unwrap(),
        output.lower().iter().cloned().reduce(f32::max).unwrap()
    );
    println!(
        "  Upper bound range: [{:.4}, {:.4}]",
        output.upper().iter().cloned().reduce(f32::min).unwrap(),
        output.upper().iter().cloned().reduce(f32::max).unwrap()
    );

    // GELU output is bounded (negative values are attenuated)
    // For any input, GELU(x) is roughly in [-0.17, x] for x > 0
    for (l, u) in output.lower().iter().zip(output.upper().iter()) {
        let l = *l;
        let u = *u;
        assert!(l <= u, "Unsound bounds: lower {} > upper {}", l, u);
        assert!(l.is_finite() && u.is_finite(), "Non-finite bounds");
    }
}
