// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_propagate::layers::LayerNormMode;
use std::collections::HashMap;

/// Supported model architectures.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Architecture {
    /// Whisper encoder (speech-to-text)
    WhisperEncoder,
    /// Whisper decoder
    WhisperDecoder,
    /// Kokoro TTS model
    Kokoro,
    /// CosyVoice TTS model
    CosyVoice,
    /// Generic transformer encoder
    TransformerEncoder,
    /// Generic transformer decoder
    TransformerDecoder,
    /// Simple MLP/feedforward network
    MLP,
    /// Convolutional neural network
    CNN,
    /// EfficientNet (image classification)
    EfficientNet,
    /// DFine/RTDetr (object detection)
    DFine,
    /// Idefics3 (vision-language model)
    Idefics3,
    /// Llama (causal LM)
    Llama,
    /// Unknown architecture - use generic handling
    #[default]
    Unknown,
}

/// Configuration for a specific model architecture.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Architecture type
    pub architecture: Architecture,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads (for transformers)
    pub num_heads: Option<usize>,
    /// Number of layers/blocks
    pub num_layers: Option<usize>,
    /// Input dimension
    pub input_dim: Option<usize>,
    /// Output dimension
    pub output_dim: Option<usize>,
    /// Custom weight name mappings
    pub weight_mappings: HashMap<String, String>,
    /// LayerNorm mode (standard or mean-only).
    pub layernorm_mode: LayerNormMode,
}

impl ModelConfig {
    /// Create a new model config with architecture type.
    pub fn new(architecture: Architecture) -> Self {
        Self {
            architecture,
            hidden_dim: 512,
            num_heads: None,
            num_layers: None,
            input_dim: None,
            output_dim: None,
            weight_mappings: HashMap::new(),
            layernorm_mode: LayerNormMode::Standard,
        }
    }

    /// Configuration for Whisper tiny encoder.
    pub fn whisper_tiny() -> Self {
        Self {
            architecture: Architecture::WhisperEncoder,
            hidden_dim: 384,
            num_heads: Some(6),
            num_layers: Some(4),
            input_dim: Some(80), // mel channels
            output_dim: Some(384),
            weight_mappings: HashMap::new(),
            layernorm_mode: LayerNormMode::Standard,
        }
    }

    /// Configuration for Whisper base encoder.
    pub fn whisper_base() -> Self {
        Self {
            architecture: Architecture::WhisperEncoder,
            hidden_dim: 512,
            num_heads: Some(8),
            num_layers: Some(6),
            input_dim: Some(80), // mel channels
            output_dim: Some(512),
            weight_mappings: HashMap::new(),
            layernorm_mode: LayerNormMode::Standard,
        }
    }

    /// Configuration for Whisper small encoder.
    pub fn whisper_small() -> Self {
        Self {
            architecture: Architecture::WhisperEncoder,
            hidden_dim: 768,
            num_heads: Some(12),
            num_layers: Some(12),
            input_dim: Some(80),
            output_dim: Some(768),
            weight_mappings: HashMap::new(),
            layernorm_mode: LayerNormMode::Standard,
        }
    }

    /// Configuration for Whisper medium encoder.
    pub fn whisper_medium() -> Self {
        Self {
            architecture: Architecture::WhisperEncoder,
            hidden_dim: 1024,
            num_heads: Some(16),
            num_layers: Some(24),
            input_dim: Some(80),
            output_dim: Some(1024),
            weight_mappings: HashMap::new(),
            layernorm_mode: LayerNormMode::Standard,
        }
    }

    /// Configuration for Whisper large encoder.
    pub fn whisper_large() -> Self {
        Self {
            architecture: Architecture::WhisperEncoder,
            hidden_dim: 1280,
            num_heads: Some(20),
            num_layers: Some(32),
            input_dim: Some(128), // large uses 128 mel channels
            output_dim: Some(1280),
            weight_mappings: HashMap::new(),
            layernorm_mode: LayerNormMode::Standard,
        }
    }

    /// Configuration for Kokoro TTS.
    pub fn kokoro() -> Self {
        Self {
            architecture: Architecture::Kokoro,
            hidden_dim: 512,
            num_heads: Some(8),
            num_layers: Some(12),
            input_dim: Some(512),
            output_dim: Some(512),
            weight_mappings: HashMap::new(),
            layernorm_mode: LayerNormMode::Standard,
        }
    }

    /// Configuration for EfficientNet-B0 (DocumentFigureClassifier scale).
    pub fn efficientnet_b0() -> Self {
        Self {
            architecture: Architecture::EfficientNet,
            hidden_dim: 1280,               // EfficientNet-B0 final hidden dim
            num_heads: None,                // Not transformer-based
            num_layers: Some(64),           // Total blocks
            input_dim: Some(3 * 224 * 224), // RGB 224x224
            output_dim: Some(1000),         // ImageNet classes
            weight_mappings: HashMap::new(),
            layernorm_mode: LayerNormMode::Standard,
        }
    }

    /// Configuration for DFine/RTDetr object detection model.
    pub fn dfine() -> Self {
        Self {
            architecture: Architecture::DFine,
            hidden_dim: 256, // d_model
            num_heads: Some(8),
            num_layers: Some(6),            // decoder layers
            input_dim: Some(3 * 640 * 640), // RGB 640x640
            output_dim: None,               // Detection outputs vary
            weight_mappings: HashMap::new(),
            layernorm_mode: LayerNormMode::Standard,
        }
    }

    /// Configuration for Idefics3 (VLM like granite-docling-258M).
    pub fn idefics3_258m() -> Self {
        Self {
            architecture: Architecture::Idefics3,
            hidden_dim: 576, // From text_config.hidden_size
            num_heads: Some(9),
            num_layers: Some(30),
            input_dim: Some(3 * 512 * 512), // Vision input
            output_dim: Some(100352),       // vocab_size
            weight_mappings: HashMap::new(),
            layernorm_mode: LayerNormMode::Standard,
        }
    }

    /// Configuration for Llama-style decoder LLM.
    pub fn llama_7b() -> Self {
        Self {
            architecture: Architecture::Llama,
            hidden_dim: 4096,
            num_heads: Some(32),
            num_layers: Some(32),
            input_dim: Some(4096),   // Embedding dim
            output_dim: Some(32000), // Vocab size
            weight_mappings: HashMap::new(),
            layernorm_mode: LayerNormMode::Standard,
        }
    }

    /// Configuration for TinyLlama (1.1B params).
    pub fn tinyllama() -> Self {
        Self {
            architecture: Architecture::Llama,
            hidden_dim: 2048,
            num_heads: Some(32),
            num_layers: Some(22),
            input_dim: Some(2048),
            output_dim: Some(32000),
            weight_mappings: HashMap::new(),
            layernorm_mode: LayerNormMode::Standard,
        }
    }

    /// Set LayerNorm mode for native model loading.
    pub fn with_layernorm_mode(mut self, mode: LayerNormMode) -> Self {
        self.layernorm_mode = mode;
        self
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architecture: Architecture::Unknown,
            hidden_dim: 512,
            num_heads: None,
            num_layers: None,
            input_dim: None,
            output_dim: None,
            weight_mappings: HashMap::new(),
            layernorm_mode: LayerNormMode::Standard,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ntest::timeout(10000)]
    #[test]
    fn test_model_config_new() {
        let config = ModelConfig::new(Architecture::MLP);
        assert_eq!(config.architecture, Architecture::MLP);
        assert_eq!(config.hidden_dim, 512); // default value
        assert!(config.num_heads.is_none());
        assert!(config.num_layers.is_none());
        assert!(config.input_dim.is_none());
        assert!(config.output_dim.is_none());
        assert!(config.weight_mappings.is_empty());
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_model_config_whisper_tiny() {
        let config = ModelConfig::whisper_tiny();
        assert_eq!(config.architecture, Architecture::WhisperEncoder);
        assert_eq!(config.hidden_dim, 384);
        assert_eq!(config.num_heads, Some(6));
        assert_eq!(config.num_layers, Some(4));
        assert_eq!(config.input_dim, Some(80));
        assert_eq!(config.output_dim, Some(384));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_model_config_whisper_small() {
        let config = ModelConfig::whisper_small();
        assert_eq!(config.architecture, Architecture::WhisperEncoder);
        assert_eq!(config.hidden_dim, 768);
        assert_eq!(config.num_heads, Some(12));
        assert_eq!(config.num_layers, Some(12));
        assert_eq!(config.input_dim, Some(80));
        assert_eq!(config.output_dim, Some(768));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_model_config_whisper_medium() {
        let config = ModelConfig::whisper_medium();
        assert_eq!(config.architecture, Architecture::WhisperEncoder);
        assert_eq!(config.hidden_dim, 1024);
        assert_eq!(config.num_heads, Some(16));
        assert_eq!(config.num_layers, Some(24));
        assert_eq!(config.input_dim, Some(80));
        assert_eq!(config.output_dim, Some(1024));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_model_config_kokoro() {
        let config = ModelConfig::kokoro();
        assert_eq!(config.architecture, Architecture::Kokoro);
        assert_eq!(config.hidden_dim, 512);
        assert_eq!(config.num_heads, Some(8));
        assert_eq!(config.num_layers, Some(12));
        assert_eq!(config.input_dim, Some(512));
        assert_eq!(config.output_dim, Some(512));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_model_config_efficientnet_b0() {
        let config = ModelConfig::efficientnet_b0();
        assert_eq!(config.architecture, Architecture::EfficientNet);
        assert_eq!(config.hidden_dim, 1280);
        assert!(config.num_heads.is_none()); // Not transformer-based
        assert_eq!(config.num_layers, Some(64));
        assert_eq!(config.input_dim, Some(3 * 224 * 224));
        assert_eq!(config.output_dim, Some(1000)); // ImageNet classes
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_model_config_dfine() {
        let config = ModelConfig::dfine();
        assert_eq!(config.architecture, Architecture::DFine);
        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.num_heads, Some(8));
        assert_eq!(config.num_layers, Some(6));
        assert_eq!(config.input_dim, Some(3 * 640 * 640));
        assert!(config.output_dim.is_none()); // Detection outputs vary
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_model_config_idefics3_258m() {
        let config = ModelConfig::idefics3_258m();
        assert_eq!(config.architecture, Architecture::Idefics3);
        assert_eq!(config.hidden_dim, 576);
        assert_eq!(config.num_heads, Some(9));
        assert_eq!(config.num_layers, Some(30));
        assert_eq!(config.input_dim, Some(3 * 512 * 512));
        assert_eq!(config.output_dim, Some(100352)); // vocab_size
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_model_config_llama_7b() {
        let config = ModelConfig::llama_7b();
        assert_eq!(config.architecture, Architecture::Llama);
        assert_eq!(config.hidden_dim, 4096);
        assert_eq!(config.num_heads, Some(32));
        assert_eq!(config.num_layers, Some(32));
        assert_eq!(config.input_dim, Some(4096));
        assert_eq!(config.output_dim, Some(32000)); // vocab size
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_model_config_tinyllama() {
        let config = ModelConfig::tinyllama();
        assert_eq!(config.architecture, Architecture::Llama);
        assert_eq!(config.hidden_dim, 2048);
        assert_eq!(config.num_heads, Some(32));
        assert_eq!(config.num_layers, Some(22));
        assert_eq!(config.input_dim, Some(2048));
        assert_eq!(config.output_dim, Some(32000));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_model_config_debug() {
        let config = ModelConfig::whisper_tiny();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("WhisperEncoder"));
        assert!(debug_str.contains("384"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_model_config_clone() {
        let config = ModelConfig::whisper_base();
        let cloned = config.clone();
        assert_eq!(cloned.architecture, config.architecture);
        assert_eq!(cloned.hidden_dim, config.hidden_dim);
        assert_eq!(cloned.num_heads, config.num_heads);
        assert_eq!(cloned.num_layers, config.num_layers);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_architecture_clone() {
        let arch = Architecture::WhisperEncoder;
        let cloned = arch.clone();
        assert_eq!(arch, cloned);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_architecture_debug() {
        let arch = Architecture::Llama;
        let debug_str = format!("{:?}", arch);
        assert_eq!(debug_str, "Llama");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_architecture_eq() {
        assert_eq!(Architecture::WhisperEncoder, Architecture::WhisperEncoder);
        assert_ne!(Architecture::WhisperEncoder, Architecture::WhisperDecoder);
        assert_ne!(Architecture::MLP, Architecture::CNN);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_architecture_all_variants_distinct() {
        let variants = vec![
            Architecture::WhisperEncoder,
            Architecture::WhisperDecoder,
            Architecture::Kokoro,
            Architecture::CosyVoice,
            Architecture::TransformerEncoder,
            Architecture::TransformerDecoder,
            Architecture::MLP,
            Architecture::CNN,
            Architecture::EfficientNet,
            Architecture::DFine,
            Architecture::Idefics3,
            Architecture::Llama,
            Architecture::Unknown,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b, "Same variant should be equal");
                } else {
                    assert_ne!(a, b, "Different variants should not be equal");
                }
            }
        }
    }
}
