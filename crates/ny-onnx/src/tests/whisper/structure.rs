// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::helpers::{whisper_tiny_encoder, whisper_tiny_propagate_network};
use ny_core::LayerType;
use ny_propagate::Layer as PropLayer;
use ny_tensor::BoundedTensor;

// =========================================================================
// Whisper Component Extraction Tests
// =========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_whisper_load_with_structure() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    let whisper = whisper_tiny_encoder();

    println!("\n=== Whisper Model Structure ===");
    println!("Encoder layers: {}", whisper.encoder_layers);
    println!("Hidden dimension: {}", whisper.hidden_dim);
    println!("Number of heads: {}", whisper.num_heads);
    println!("Stem end index: {}", whisper.structure.stem_end_idx);
    println!(
        "ln_post start index: {}",
        whisper.structure.ln_post_start_idx
    );
    println!("Number of blocks: {}", whisper.structure.blocks.len());

    // Verify expected structure for Whisper-tiny
    assert_eq!(whisper.encoder_layers, 4, "Expected 4 encoder layers");
    assert_eq!(
        whisper.hidden_dim, 384,
        "Expected hidden_dim=384 for Whisper-tiny"
    );
    assert_eq!(whisper.num_heads, 6, "Expected 6 attention heads");
    assert_eq!(whisper.structure.blocks.len(), 4, "Expected 4 blocks");

    // Verify block boundaries make sense
    for (i, block) in whisper.structure.blocks.iter().enumerate() {
        println!(
            "  Block {}: layers {}-{} ({} layers)",
            block.index, block.start_layer_idx, block.end_layer_idx, block.num_layers
        );
        assert_eq!(block.index, i, "Block index mismatch");
        assert!(block.num_layers > 0, "Block {} has no layers", i);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_whisper_encoder_stem_extraction() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    let whisper = whisper_tiny_encoder();
    let stem = whisper.encoder_stem().expect("Failed to extract stem");

    println!("\n=== Encoder Stem ===");
    println!("Stem layers: {}", stem.num_layers());

    // The stem should contain Conv1d, GELU, Conv1d, GELU, and possibly Add for positional embedding
    assert!(stem.num_layers() > 0, "Stem should have layers");

    // Count layer types
    let conv_count = stem
        .layers()
        .iter()
        .filter(|l| matches!(l, PropLayer::Conv1d(_)))
        .count();
    let gelu_count = stem
        .layers()
        .iter()
        .filter(|l| matches!(l, PropLayer::GELU(_)))
        .count();

    println!("  Conv1d layers: {}", conv_count);
    println!("  GELU activations: {}", gelu_count);

    // Expected: 2 Conv1d, 2 GELU (+ possibly transpose/add)
    assert!(
        conv_count >= 2,
        "Expected at least 2 Conv1d layers in stem, got {}",
        conv_count
    );
    assert!(
        gelu_count >= 2,
        "Expected at least 2 GELU activations in stem, got {}",
        gelu_count
    );
}

#[ntest::timeout(120000)] // ~30s runtime; 4x margin for CI variability.
#[test]
fn test_whisper_encoder_layer_extraction() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    let whisper = whisper_tiny_encoder();
    let full_network = whisper_tiny_propagate_network();

    // Extract each block and verify structure
    for block_idx in 0..whisper.encoder_layers {
        let block = whisper
            .encoder_layer_from_network(full_network, block_idx)
            .unwrap_or_else(|e| panic!("Failed to extract block {}: {}", block_idx, e));

        println!("\n=== Encoder Block {} ===", block_idx);
        println!("  Layers: {}", block.num_layers());

        // Count layer types in the block
        let layer_norm_count = block
            .layers()
            .iter()
            .filter(|l| matches!(l, PropLayer::LayerNorm(_)))
            .count();
        let softmax_count = block
            .layers()
            .iter()
            .filter(|l| matches!(l, PropLayer::Softmax(_)))
            .count();
        let gelu_count = block
            .layers()
            .iter()
            .filter(|l| matches!(l, PropLayer::GELU(_)))
            .count();
        let matmul_count = block
            .layers()
            .iter()
            .filter(|l| matches!(l, PropLayer::MatMul(_)))
            .count();
        let linear_count = block
            .layers()
            .iter()
            .filter(|l| matches!(l, PropLayer::Linear(_)))
            .count();

        println!("  LayerNorm: {}", layer_norm_count);
        println!("  Softmax: {}", softmax_count);
        println!("  GELU: {}", gelu_count);
        println!("  MatMul: {}", matmul_count);
        println!("  Linear: {}", linear_count);

        // Each block should have:
        // - 1-2 LayerNorms (post-norm Whisper has 1 LayerNorm between attention residual and MLP;
        //   pre-norm has 2 LayerNorms: attn_ln before attention, mlp_ln before MLP)
        // - 1 Softmax (attention)
        // - 1 GELU (MLP activation)
        assert!(
            layer_norm_count >= 1,
            "Block {} should have at least 1 LayerNorm, got {}",
            block_idx,
            layer_norm_count
        );
        assert_eq!(
            softmax_count, 1,
            "Block {} should have exactly 1 Softmax, got {}",
            block_idx, softmax_count
        );
        assert!(
            gelu_count >= 1,
            "Block {} should have at least 1 GELU, got {}",
            block_idx,
            gelu_count
        );
    }

    // Test out-of-bounds access
    let result = whisper.encoder_layer_from_network(full_network, 10);
    assert!(result.is_err(), "Should fail for out-of-bounds block index");
}

// Budget: shares the 33MB dynamo whisper fixture; the first test to touch the
// cached model/network pays the full debug-build load/convert cost inside its
// timer, which exceeds the old 10s budget under parallel suite load. 120s
// matches the heavy whisper siblings and still guards against hangs.
#[ntest::timeout(120000)]
#[test]
fn test_whisper_single_block_ibp() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    // Test IBP propagation through a single encoder block
    let whisper = whisper_tiny_encoder();
    let full_network = whisper_tiny_propagate_network();
    let block = whisper
        .encoder_layer_from_network(full_network, 0)
        .expect("Failed to extract block 0");

    println!("\n=== Block 0 IBP Test ===");
    println!("Block has {} layers", block.num_layers());

    // Create input matching the expected shape for a transformer block
    // Shape: [seq_len, hidden_dim] = [100, 384] for Whisper-tiny.
    // Use a small sequence for faster testing.
    let seq_len = 2;
    let hidden_dim = whisper.hidden_dim;

    let lower_data = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[seq_len, hidden_dim]), -1.0f32);
    let upper_data = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[seq_len, hidden_dim]), 1.0f32);
    let input = BoundedTensor::new(lower_data, upper_data).unwrap();

    println!("Input shape: {:?}", input.shape());

    match block.propagate_ibp(&input) {
        Ok(output) => {
            println!("Output shape: {:?}", output.shape());
            let sound = output
                .lower()
                .iter()
                .zip(output.upper().iter())
                .all(|(l, u)| l <= u);
            assert!(sound, "IBP bounds must be sound");
            assert_eq!(output.shape(), input.shape());
        }
        Err(err) => {
            // Sequential extraction doesn't model residual connections or
            // attention head reshapes; expect shape-related errors in IBP.
            // After #3312, the first failure is typically Reshape (head split)
            // rather than the old Concat error.
            let msg = format!("{:?}", err);
            assert!(
                msg.contains("Concat") || msg.contains("ShapeMismatch") || msg.contains("Reshape"),
                "Expected sequential IBP limitation (Concat/Reshape/ShapeMismatch), got: {}",
                msg
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_whisper_param_count_from_load() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    let whisper = whisper_tiny_encoder();
    let param_count = whisper.param_count();

    println!("\n=== Whisper-tiny Encoder Parameters ===");
    println!("Total parameters: {}", param_count);
    println!("Expected (approximate): ~9M for encoder only");

    // Whisper-tiny encoder has roughly 9M parameters
    // Full model (encoder+decoder) has ~39M
    assert!(
        param_count > 1_000_000,
        "Expected at least 1M parameters, got {}",
        param_count
    );
    assert!(
        param_count < 50_000_000,
        "Expected less than 50M parameters, got {}",
        param_count
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_block_layer_structure() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    let whisper = whisper_tiny_encoder();
    let block_info = whisper.block_info(0).expect("Block 0 not found");

    println!(
        "\n=== Block 0 Detailed Structure (layers {}-{}) ===",
        block_info.start_layer_idx, block_info.end_layer_idx
    );

    for idx in block_info.start_layer_idx..block_info.end_layer_idx {
        let layer = &whisper.model.network.layers[idx];
        println!("  [{}] {:?}: {}", idx, layer.layer_type, layer.name);
        println!("      inputs: {:?}", layer.inputs);
        println!("      outputs: {:?}", layer.outputs);
    }

    // Count layer types including Add
    let layers: Vec<_> = whisper
        .model
        .network
        .layers
        .iter()
        .skip(block_info.start_layer_idx)
        .take(block_info.num_layers)
        .collect();

    let add_count = layers
        .iter()
        .filter(|l| l.layer_type == LayerType::Add)
        .count();
    let transpose_count = layers
        .iter()
        .filter(|l| l.layer_type == LayerType::Transpose)
        .count();

    println!("\n=== Layer Counts ===");
    println!("Add operations: {}", add_count);
    println!("Transpose operations: {}", transpose_count);

    // Verify we see the expected structure for residual connections
    // Each block should have Add operations for residual connections
    assert!(
        add_count >= 2,
        "Expected at least 2 Add ops for residuals, got {}",
        add_count
    );
}
