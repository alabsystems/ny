// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::detect::extract_block_number;
use super::helpers::extract_layer_number;
use super::*;
use ny_propagate::layers::LayerNormMode;
use safetensors::tensor::TensorView;
use safetensors::{serialize, Dtype};
use std::collections::BTreeMap;
use tempfile::tempdir;

fn write_config_json(dir: &std::path::Path, contents: &str) -> std::path::PathBuf {
    let path = dir.join("config.json");
    std::fs::write(&path, contents).expect("Failed to write config.json");
    path
}

fn write_minimal_whisper_safetensors(
    dir: &std::path::Path,
    hidden_dim: usize,
    n_mels: usize,
) -> std::path::PathBuf {
    let conv1_len = hidden_dim * n_mels * 3;
    let conv2_len = hidden_dim * hidden_dim * 3;
    let conv1: Vec<f32> = (0..conv1_len).map(|i| i as f32 * 0.01).collect();
    let conv2: Vec<f32> = (0..conv2_len).map(|i| i as f32 * 0.01).collect();

    let conv1_view = TensorView::new(
        Dtype::F32,
        vec![hidden_dim, n_mels, 3],
        bytemuck::cast_slice(&conv1),
    )
    .expect("Failed to build conv1 TensorView");
    let conv2_view = TensorView::new(
        Dtype::F32,
        vec![hidden_dim, hidden_dim, 3],
        bytemuck::cast_slice(&conv2),
    )
    .expect("Failed to build conv2 TensorView");

    let mut tensors = BTreeMap::new();
    tensors.insert("conv1.weight".to_string(), conv1_view);
    tensors.insert("conv2.weight".to_string(), conv2_view);

    let data = serialize(tensors, None).expect("Failed to serialize safetensors");
    let path = dir.join("model.safetensors");
    std::fs::write(&path, data).expect("Failed to write safetensors file");
    path
}

fn write_whisper_fixture(
    hidden_dim: usize,
    num_heads: usize,
    num_layers: usize,
    n_mels: usize,
) -> tempfile::TempDir {
    let dir = tempdir().expect("Failed to create tempdir");
    let config = format!(
        r#"{{
  "architectures": ["WhisperForConditionalGeneration"],
  "model_type": "whisper",
  "d_model": {hidden_dim},
  "encoder_layers": {num_layers},
  "encoder_attention_heads": {num_heads},
  "num_mel_bins": {n_mels}
}}"#
    );
    write_config_json(dir.path(), &config);
    write_minimal_whisper_safetensors(dir.path(), hidden_dim, n_mels);
    dir
}

#[ntest::timeout(10000)]
#[test]
fn test_extract_block_number() {
    assert_eq!(
        extract_block_number("blocks.5.attn.weight", "blocks"),
        Some(5)
    );
    assert_eq!(
        extract_block_number("encoder.blocks.12.mlp.fc1.weight", "blocks"),
        Some(12)
    );
    assert_eq!(
        extract_block_number("layer.0.attention.weight", "layer"),
        Some(0)
    );
    assert_eq!(extract_block_number("no_match", "blocks"), None);
}

#[ntest::timeout(10000)]
#[test]
fn test_extract_layer_number() {
    assert_eq!(extract_layer_number("layer_3.weight"), Some(3));
    assert_eq!(extract_layer_number("fc2.weight"), Some(2));
    assert_eq!(extract_layer_number("linear1.weight"), Some(1));
    // This finds "fc1" first, so returns 1 (not the block number 5)
    assert_eq!(extract_layer_number("blocks.5.fc1.weight"), Some(1));
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_parse_whisper() {
    let json = r#"{
        "architectures": ["WhisperForConditionalGeneration"],
        "model_type": "whisper",
        "d_model": 1280,
        "encoder_layers": 32,
        "decoder_layers": 32,
        "encoder_attention_heads": 20,
        "decoder_attention_heads": 20,
        "num_mel_bins": 128,
        "vocab_size": 51866
    }"#;

    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse Whisper config");

    assert_eq!(
        hf_config.architecture_name(),
        Some("WhisperForConditionalGeneration")
    );
    assert_eq!(hf_config.model_type, "whisper");
    assert_eq!(hf_config.d_model, Some(1280));
    assert_eq!(hf_config.encoder_layers, Some(32));
    assert_eq!(hf_config.encoder_attention_heads, Some(20));
    assert_eq!(hf_config.num_mel_bins, Some(128));

    // Test architecture detection
    let config = hf_config.to_model_config();
    assert_eq!(config.architecture, Architecture::WhisperEncoder);
    assert_eq!(config.hidden_dim, 1280);
    assert_eq!(config.num_heads, Some(20));
    assert_eq!(config.num_layers, Some(32));
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_parse_dfine() {
    let json = r#"{
        "architectures": ["DFineForObjectDetection"],
        "model_type": "d_fine",
        "d_model": 256,
        "decoder_attention_heads": 8,
        "decoder_layers": 6,
        "encoder_layers": 1
    }"#;

    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse DFine config");

    assert_eq!(
        hf_config.architecture_name(),
        Some("DFineForObjectDetection")
    );
    assert_eq!(hf_config.model_type, "d_fine");
    assert_eq!(hf_config.d_model, Some(256));

    let config = hf_config.to_model_config();
    assert_eq!(config.architecture, Architecture::DFine);
    assert_eq!(config.hidden_dim, 256);
    assert_eq!(config.num_heads, Some(8));
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_parse_idefics3() {
    let json = r#"{
        "architectures": ["Idefics3ForConditionalGeneration"],
        "model_type": "idefics3",
        "text_config": {
            "hidden_size": 576,
            "num_attention_heads": 9,
            "num_hidden_layers": 30
        },
        "vision_config": {
            "hidden_size": 768,
            "image_size": 512
        }
    }"#;

    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse Idefics3 config");

    assert_eq!(
        hf_config.architecture_name(),
        Some("Idefics3ForConditionalGeneration")
    );
    assert_eq!(hf_config.model_type, "idefics3");

    // Check nested text_config
    let text_cfg = hf_config
        .text_config
        .as_ref()
        .expect("text_config should exist");
    assert_eq!(text_cfg.d_model, Some(576));
    assert_eq!(text_cfg.num_heads, Some(9));
    assert_eq!(text_cfg.num_hidden_layers, Some(30));

    let config = hf_config.to_model_config();
    assert_eq!(config.architecture, Architecture::Idefics3);
    // Should use text_config hidden_size
    assert_eq!(config.hidden_dim, 576);
    assert_eq!(config.num_heads, Some(9));
    assert_eq!(config.num_layers, Some(30));
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_parse_llama() {
    let json = r#"{
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "vocab_size": 32000
    }"#;

    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse Llama config");

    assert_eq!(hf_config.architecture_name(), Some("LlamaForCausalLM"));
    assert_eq!(hf_config.model_type, "llama");
    assert_eq!(hf_config.d_model, Some(4096)); // hidden_size alias

    let config = hf_config.to_model_config();
    assert_eq!(config.architecture, Architecture::Llama);
    assert_eq!(config.hidden_dim, 4096);
    assert_eq!(config.num_heads, Some(32));
    assert_eq!(config.num_layers, Some(32));
    assert_eq!(config.output_dim, Some(32000)); // vocab_size
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_from_file_whisper() {
    let dir = tempdir().expect("Failed to create tempdir");
    let config = r#"{
        "architectures": ["WhisperForConditionalGeneration"],
        "model_type": "whisper",
        "d_model": 16,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "encoder_attention_heads": 2,
        "decoder_attention_heads": 2,
        "num_mel_bins": 80
    }"#;
    let path = write_config_json(dir.path(), config);
    let hf_config = HfConfig::from_file(&path).expect("Failed to load config.json");

    assert_eq!(
        hf_config.architecture_name(),
        Some("WhisperForConditionalGeneration")
    );
    assert_eq!(hf_config.model_type, "whisper");
    assert_eq!(hf_config.d_model, Some(16));
    assert_eq!(hf_config.encoder_layers, Some(2));
    assert_eq!(hf_config.encoder_attention_heads, Some(2));
    assert_eq!(hf_config.num_mel_bins, Some(80));

    let config = hf_config.to_model_config();
    assert_eq!(config.architecture, Architecture::WhisperEncoder);
    assert_eq!(config.hidden_dim, 16);
    assert_eq!(config.num_heads, Some(2));
    assert_eq!(config.num_layers, Some(2));
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_from_file_dfine() {
    let dir = tempdir().expect("Failed to create tempdir");
    let config = r#"{
        "architectures": ["DFineForObjectDetection"],
        "model_type": "d_fine",
        "d_model": 256,
        "decoder_attention_heads": 8,
        "decoder_layers": 6,
        "encoder_layers": 1
    }"#;
    let path = write_config_json(dir.path(), config);
    let hf_config = HfConfig::from_file(&path).expect("Failed to load config.json");

    assert_eq!(
        hf_config.architecture_name(),
        Some("DFineForObjectDetection")
    );
    assert_eq!(hf_config.model_type, "d_fine");
    assert_eq!(hf_config.d_model, Some(256));

    let config = hf_config.to_model_config();
    assert_eq!(config.architecture, Architecture::DFine);
    assert_eq!(config.hidden_dim, 256);
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_from_file_smoldocling() {
    let dir = tempdir().expect("Failed to create tempdir");
    let config = r#"{
        "architectures": ["Idefics3ForConditionalGeneration"],
        "model_type": "idefics3",
        "text_config": {
            "d_model": 576,
            "num_heads": 8,
            "num_hidden_layers": 4
        }
    }"#;
    let path = write_config_json(dir.path(), config);
    let hf_config = HfConfig::from_file(&path).expect("Failed to load config.json");

    assert_eq!(
        hf_config.architecture_name(),
        Some("Idefics3ForConditionalGeneration")
    );
    assert_eq!(hf_config.model_type, "idefics3");

    // SmolDocling uses text_config for main dimensions
    let text_cfg = hf_config
        .text_config
        .as_ref()
        .expect("text_config should exist");
    assert_eq!(text_cfg.d_model, Some(576));

    let config = hf_config.to_model_config();
    assert_eq!(config.architecture, Architecture::Idefics3);
    assert_eq!(config.hidden_dim, 576);
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_from_directory() {
    let dir = tempdir().expect("Failed to create tempdir");
    let config = r#"{
        "architectures": ["WhisperForConditionalGeneration"],
        "model_type": "whisper",
        "d_model": 8,
        "encoder_layers": 1,
        "encoder_attention_heads": 1,
        "num_mel_bins": 80
    }"#;
    write_config_json(dir.path(), config);
    let hf_config = HfConfig::from_directory(dir.path())
        .expect("Failed to search directory")
        .expect("config.json should exist in directory");

    assert_eq!(
        hf_config.architecture_name(),
        Some("WhisperForConditionalGeneration")
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_architecture_detection_by_model_type() {
    // Test that model_type fallback works when architectures field is empty
    let json = r#"{
        "model_type": "bert",
        "hidden_size": 768
    }"#;

    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    let config = hf_config.to_model_config();
    assert_eq!(config.architecture, Architecture::TransformerEncoder);
}

#[ntest::timeout(10000)]
#[test]
fn test_native_model_load_with_hf_config() {
    // Test that NativeModel::load() uses HfConfig when config.json exists.
    // This verifies the integration of HfConfig-based architecture detection.
    let dir = write_whisper_fixture(2, 1, 0, 1);
    let config_path = dir.path().join("config.json");
    let model_path = dir.path().join("model.safetensors");

    // Load the model
    let native_model = NativeModel::load(&model_path).expect("Failed to load native model");

    // Verify model was loaded
    assert_eq!(
        native_model.config.architecture,
        Architecture::WhisperEncoder
    );

    // Fixture should always include config.json so missing config is a hard failure.
    assert!(
        config_path.exists(),
        "fixture should include config.json for HfConfig-based detection"
    );
    let hf_config = HfConfig::from_file(&config_path).expect("Failed to load config.json");
    if let Some(d_model) = hf_config.d_model {
        assert_eq!(
            native_model.config.hidden_dim, d_model,
            "HfConfig d_model should match NativeModel hidden_dim"
        );
    }
    println!(
        "Loaded model with HfConfig: architecture={:?}, hidden_dim={}",
        native_model.config.architecture, native_model.config.hidden_dim
    );

    // Verify weights were loaded
    assert!(
        !native_model.weights.is_empty(),
        "Model should have loaded weights"
    );
    println!("Loaded {} weight tensors", native_model.weights.len());
}

#[ntest::timeout(10000)]
#[test]
fn test_native_model_load_from_hf_directory() {
    // Test the explicit load_from_hf_directory method which requires config.json.
    let dir = write_whisper_fixture(2, 1, 0, 1);
    let native_model =
        NativeModel::load_from_hf_directory(dir.path()).expect("Failed to load HF directory");

    assert_eq!(
        native_model.config.architecture,
        Architecture::WhisperEncoder
    );
    assert_eq!(native_model.config.hidden_dim, 2);
    assert_eq!(native_model.config.num_heads, Some(1));
    assert_eq!(native_model.config.num_layers, Some(0));
}

#[ntest::timeout(10000)]
#[test]
fn test_native_model_to_graph_network() {
    // Test that to_graph_network works for native models (safetensors format).
    // Uses the whisper-tiny model if available.
    use ndarray::ArrayD;
    use ny_tensor::BoundedTensor;
    let dir = write_whisper_fixture(2, 1, 0, 1);
    let model_path = dir.path().join("model.safetensors");

    // Load native model
    let native_model = NativeModel::load(model_path).expect("Failed to load native model");

    assert_eq!(
        native_model.config.architecture,
        Architecture::WhisperEncoder
    );
    assert_eq!(native_model.config.hidden_dim, 2);

    println!(
        "Loaded native model: {:?}, layers: {}",
        native_model.config.architecture,
        native_model.network.layers.len()
    );

    // Convert to graph network
    let graph = native_model
        .to_graph_network()
        .expect("Failed to convert native model to graph network");

    println!("GraphNetwork nodes: {}", graph.num_nodes());

    // GraphNetwork should have nodes
    assert!(
        graph.num_nodes() > 0,
        "Expected graph network to have nodes"
    );

    // Test IBP propagation with correct input shape for Whisper encoder.
    // Note: these tests often use unbatched input [channels, length] even if the
    // original ONNX model includes a batch dimension.
    let n_mels = 80; // Standard mel spectrogram channels
    let time_frames = 100; // Small time dimension for testing
    let input_data = ArrayD::from_elem(ndarray::IxDyn(&[n_mels, time_frames]), 0.0f32);
    let input = BoundedTensor::from_epsilon(input_data, 0.01).expect("valid test input");

    println!(
        "Testing IBP propagation with input shape: {:?}",
        input.shape()
    );

    // Get topological order
    let node_names: Vec<String> = graph.topological_sort().unwrap_or_default();
    println!("Graph has {} nodes in topological order", node_names.len());
    for (i, name) in node_names.iter().take(5).enumerate() {
        println!("  Node {}: {}", i, name);
    }

    match graph.propagate_ibp(&input) {
        Ok(output) => {
            println!("IBP succeeded! Output shape: {:?}", output.shape());
            println!("Max width: {:.6e}", output.max_width());

            // Check for NaN/Inf
            let has_nan = output.lower().iter().any(|v| v.is_nan())
                || output.upper().iter().any(|v| v.is_nan());
            let has_inf = output.lower().iter().any(|v| v.is_infinite())
                || output.upper().iter().any(|v| v.is_infinite());

            assert!(
                !has_nan,
                "Output contains NaN values — bounds propagation is broken"
            );
            assert!(
                !has_inf,
                "Output contains Inf values — infinite bounds are vacuous \
                 and indicate a propagation gap"
            );

            // Count unsound bounds (lower > upper)
            let unsound_count = output
                .lower()
                .iter()
                .zip(output.upper().iter())
                .filter(|(l, u)| l > u)
                .count();
            if unsound_count > 0 {
                // Report details before failing
                eprintln!(
                    "UNSOUND: {} bounds have lower > upper out of {}",
                    unsound_count,
                    output.len()
                );
                for (i, (l, u)) in output
                    .lower()
                    .iter()
                    .zip(output.upper().iter())
                    .enumerate()
                    .take(10)
                {
                    if l > u {
                        eprintln!("  Element {}: lower={}, upper={}", i, l, u);
                    }
                }
            }

            assert_eq!(
                unsound_count, 0,
                "Unsound bounds detected: {unsound_count} elements have lower > upper. \
                 A verification tool must never produce unsound bounds."
            );
        }
        Err(e) => {
            // IBP fails for this fixture due to shape mismatches in the conv stem.
            // This is a known limitation tracked in #1721. Assert the specific
            // expected error to avoid silently passing when IBP breaks for
            // other reasons. When conv stem IBP is implemented, this branch
            // should never execute — the Ok branch has full soundness checks.
            let msg = format!("{}", e);
            assert!(
                msg.contains("Shape mismatch"),
                "IBP failed with unexpected error (expected shape mismatch \
                 from conv stem): {e}"
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_native_to_propagate_vs_graph_network() {
    // Compare to_propagate_network vs to_graph_network for native models.
    // For simple models, both should produce similar results.
    let dir = write_whisper_fixture(2, 1, 0, 1);
    let model_path = dir.path().join("model.safetensors");
    let native_model = NativeModel::load(&model_path).expect("Failed to load native model");

    // Get both network types
    let sequential = native_model.to_propagate_network();
    let graph = native_model.to_graph_network();

    // Both conversions should succeed
    assert!(
        sequential.is_ok(),
        "Sequential network conversion should succeed for native models"
    );
    assert!(
        graph.is_ok(),
        "Graph network conversion should succeed for native models"
    );
}

// ============================================================
// HfConfig edge cases
// ============================================================

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_default_values() {
    // Minimal config with all defaults
    let json = r#"{}"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");

    assert!(hf_config.architectures.is_empty());
    assert_eq!(hf_config.model_type, "");
    assert!(hf_config.d_model.is_none());
    assert!(hf_config.num_hidden_layers.is_none());
    assert!(hf_config.encoder_layers.is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_hidden_size_alias() {
    // Test that hidden_size works as alias for d_model
    let json = r#"{"hidden_size": 768}"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    assert_eq!(hf_config.d_model, Some(768));
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_hidden_dim_alias() {
    // Test that hidden_dim works as alias for d_model
    let json = r#"{"hidden_dim": 512}"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    assert_eq!(hf_config.d_model, Some(512));
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_num_attention_heads_alias() {
    // Test that num_attention_heads works as alias for num_heads
    let json = r#"{"num_attention_heads": 16}"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    assert_eq!(hf_config.num_heads, Some(16));
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_num_attention_head_alias() {
    // Some configs use singular num_attention_head.
    let json = r#"{"num_attention_head": 12}"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    assert_eq!(hf_config.num_heads, Some(12));
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_architecture_name_empty() {
    let json = r#"{"architectures": []}"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    assert!(hf_config.architecture_name().is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_architecture_name_multiple() {
    // When multiple architectures are specified, first one is used
    let json = r#"{"architectures": ["FirstArch", "SecondArch"]}"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    assert_eq!(hf_config.architecture_name(), Some("FirstArch"));
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_to_model_config_unknown_arch() {
    // Unknown architecture should default to Unknown
    let json = r#"{"architectures": ["SomeUnknownModel"], "hidden_size": 256}"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    let config = hf_config.to_model_config();
    assert_eq!(config.architecture, Architecture::Unknown);
    assert_eq!(config.hidden_dim, 256);
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_to_model_config_gpt2() {
    // GPT2 should be detected as TransformerDecoder
    let json = r#"{"architectures": ["GPT2LMHeadModel"], "hidden_size": 768}"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    let config = hf_config.to_model_config();
    assert_eq!(config.architecture, Architecture::TransformerDecoder);
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_to_model_config_layernorm_deept_alias_4176() {
    let json = r#"{
        "architectures": ["WhisperForConditionalGeneration"],
        "layer_norm_type": "deept"
    }"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    let config = hf_config.to_model_config();
    assert_eq!(config.layernorm_mode, LayerNormMode::MeanOnly);
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_to_model_config_bert() {
    // BERT should be detected as TransformerEncoder
    let json = r#"{"architectures": ["BertModel"], "hidden_size": 768}"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    let config = hf_config.to_model_config();
    assert_eq!(config.architecture, Architecture::TransformerEncoder);
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_to_model_config_efficientnet() {
    let json = r#"{"architectures": ["EfficientNetForImageClassification"], "hidden_size": 1280}"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    let config = hf_config.to_model_config();
    assert_eq!(config.architecture, Architecture::EfficientNet);
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_uses_encoder_layers_for_num_layers() {
    // When encoder_layers is specified, it should be used for num_layers
    let json = r#"{
        "architectures": ["WhisperForConditionalGeneration"],
        "model_type": "whisper",
        "encoder_layers": 12,
        "num_hidden_layers": 24
    }"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    let config = hf_config.to_model_config();
    // For WhisperEncoder, encoder_layers should take precedence
    assert_eq!(config.num_layers, Some(12));
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_whisper_fallback_num_hidden_layers() {
    let json = r#"{
        "architectures": ["WhisperForConditionalGeneration"],
        "model_type": "whisper",
        "num_hidden_layers": 16
    }"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    let config = hf_config.to_model_config();
    // Should fall back to num_hidden_layers when encoder/decoder layers are missing
    assert_eq!(config.num_layers, Some(16));
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_uses_encoder_attention_heads() {
    let json = r#"{
        "architectures": ["WhisperForConditionalGeneration"],
        "encoder_attention_heads": 8,
        "decoder_attention_heads": 16,
        "num_attention_heads": 4
    }"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    let config = hf_config.to_model_config();
    // For WhisperEncoder, encoder_attention_heads should be used
    assert_eq!(config.num_heads, Some(8));
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_whisper_fallback_num_attention_heads() {
    let json = r#"{
        "architectures": ["WhisperForConditionalGeneration"],
        "num_attention_heads": 6
    }"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    let config = hf_config.to_model_config();
    // Should fall back to generic num_attention_heads when encoder/decoder heads are missing
    assert_eq!(config.num_heads, Some(6));
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_model_type_fallback_gpt() {
    // When architectures is empty but model_type is set
    let json = r#"{"model_type": "gpt2", "hidden_size": 768}"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    let config = hf_config.to_model_config();
    assert_eq!(config.architecture, Architecture::TransformerDecoder);
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_model_type_fallback_llama() {
    let json = r#"{"model_type": "llama", "hidden_size": 4096}"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    let config = hf_config.to_model_config();
    assert_eq!(config.architecture, Architecture::Llama);
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_vision_config() {
    // Test that vision_config is parsed correctly
    let json = r#"{
        "architectures": ["Idefics3ForConditionalGeneration"],
        "vision_config": {
            "hidden_size": 768,
            "image_size": 512
        }
    }"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    let vision = hf_config.vision_config.expect("vision_config should exist");
    assert_eq!(vision.d_model, Some(768));
    assert_eq!(vision.image_size, Some(512));
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_intermediate_size() {
    let json = r#"{"intermediate_size": 3072}"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    assert_eq!(hf_config.intermediate_size, Some(3072));
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_encoder_ffn_dim() {
    let json = r#"{"encoder_ffn_dim": 1536, "decoder_ffn_dim": 2048}"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    assert_eq!(hf_config.encoder_ffn_dim, Some(1536));
    assert_eq!(hf_config.decoder_ffn_dim, Some(2048));
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_num_channels() {
    let json = r#"{"num_channels": 3}"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    assert_eq!(hf_config.num_channels, Some(3));
}

// ============================================================
// Error handling tests
// ============================================================

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_from_file_nonexistent() {
    let result = HfConfig::from_file("/nonexistent/path/config.json");
    assert!(result.is_err());
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_from_directory_nonexistent() {
    let result = HfConfig::from_directory("/nonexistent/path");
    // Should return Ok(None) for non-existent directory or error
    // depending on implementation
    assert!(result.is_err() || result.unwrap().is_none());
}

// ============================================================
// Helper function tests
// ============================================================

#[ntest::timeout(10000)]
#[test]
fn test_extract_block_number_various_patterns() {
    // Test with various block naming patterns
    assert_eq!(extract_block_number("transformer.h.0.attn", "h"), Some(0));
    assert_eq!(extract_block_number("transformer.h.10.attn", "h"), Some(10));
    assert_eq!(
        extract_block_number("model.layers.5.self_attn", "layers"),
        Some(5)
    );
    assert_eq!(
        extract_block_number("encoder.layer.3.attention", "layer"),
        Some(3)
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_extract_block_number_edge_cases() {
    // Empty pattern creates "." which matches first dot, then reads "5"
    // Pattern: format!("{}.", "") = "." -> finds "." at index 6 -> rest is "5.attn" -> returns 5
    assert_eq!(extract_block_number("blocks.5.attn", ""), Some(5));
    // Pattern at different positions - finds first match
    assert_eq!(extract_block_number("blocks.7.blocks.3", "blocks"), Some(7));
    // Pattern not followed by dot and number
    assert_eq!(extract_block_number("blocks_test", "blocks"), None);
}

#[ntest::timeout(10000)]
#[test]
fn test_extract_layer_number_edge_cases() {
    // No number pattern
    assert_eq!(extract_layer_number("weight"), None);
    // extract_layer_number tries patterns: ["layer_", "layer", "fc", "linear", "."]
    // "layer" is checked before "fc", so it finds "layer2" and returns 2
    assert_eq!(extract_layer_number("fc1.layer2.weight"), Some(2));
    // Just number suffix - finds "." pattern and then "123"
    assert_eq!(extract_layer_number("linear123"), Some(123));
}
