// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::config::{Architecture, ModelConfig};
use ny_core::{NyError, Result};
use ny_propagate::layers::LayerNormMode;
use serde::Deserialize;
use std::path::Path;

/// HuggingFace config.json structure.
///
/// This is used to determine model architecture when loading from
/// HuggingFace model directories.
#[derive(Debug, Clone, Deserialize)]
pub struct HfConfig {
    /// Architecture names (e.g., "WhisperForConditionalGeneration")
    #[serde(default)]
    pub architectures: Vec<String>,
    /// Model type (e.g., "whisper", "efficientnet")
    #[serde(default)]
    pub model_type: String,
    /// Hidden dimension (various names in different architectures)
    #[serde(alias = "hidden_size", alias = "hidden_dim")]
    pub d_model: Option<usize>,
    /// Number of hidden layers
    pub num_hidden_layers: Option<usize>,
    /// Encoder layers (for encoder-decoder models)
    pub encoder_layers: Option<usize>,
    /// Decoder layers (for encoder-decoder models)
    pub decoder_layers: Option<usize>,
    /// Encoder attention heads
    #[serde(alias = "encoder_attention_head")]
    pub encoder_attention_heads: Option<usize>,
    /// Decoder attention heads
    #[serde(alias = "decoder_attention_head")]
    pub decoder_attention_heads: Option<usize>,
    /// Number of attention heads (generic)
    #[serde(alias = "num_attention_heads", alias = "num_attention_head")]
    pub num_heads: Option<usize>,
    /// Number of mel bins (Whisper)
    pub num_mel_bins: Option<usize>,
    /// Image size (vision models)
    pub image_size: Option<usize>,
    /// Number of channels (vision models)
    pub num_channels: Option<usize>,
    /// Vocabulary size
    pub vocab_size: Option<usize>,
    /// Intermediate/FFN size (generic)
    pub intermediate_size: Option<usize>,
    /// Encoder FFN dimension (encoder-decoder models)
    pub encoder_ffn_dim: Option<usize>,
    /// Decoder FFN dimension (encoder-decoder models)
    pub decoder_ffn_dim: Option<usize>,
    /// Text config (for VLMs like Idefics3)
    pub text_config: Option<Box<HfConfig>>,
    /// Vision config (for VLMs like Idefics3)
    pub vision_config: Option<Box<HfConfig>>,
    /// Backbone config (for detection models like DFine)
    pub backbone_config: Option<serde_json::Value>,
    /// Optional LayerNorm mode (e.g., "mean_only" for DeepT-style normalization).
    #[serde(default, alias = "layer_norm_type", alias = "layer_norm_mode")]
    pub layer_norm_type: Option<String>,
}

impl HfConfig {
    /// Load HfConfig from a config.json file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| {
            NyError::ModelLoad(format!(
                "Failed to read config.json {}: {}",
                path.display(),
                e
            ))
        })?;

        serde_json::from_str(&content).map_err(|e| {
            NyError::ModelLoad(format!(
                "Failed to parse config.json {}: {}",
                path.display(),
                e
            ))
        })
    }

    /// Load HfConfig from a model directory (looks for config.json).
    pub fn from_directory<P: AsRef<Path>>(dir: P) -> Result<Option<Self>> {
        let config_path = dir.as_ref().join("config.json");
        if config_path.exists() {
            Ok(Some(Self::from_file(&config_path)?))
        } else {
            Ok(None)
        }
    }

    /// Get the primary architecture name.
    pub fn architecture_name(&self) -> Option<&str> {
        self.architectures.first().map(|s| s.as_str())
    }

    /// Convert to ModelConfig.
    pub fn to_model_config(&self) -> ModelConfig {
        let architecture = self.detect_architecture();

        // Determine hidden dimension
        let hidden_dim = if architecture == Architecture::Idefics3 {
            // For Idefics3, use text_config hidden_size
            self.text_config
                .as_ref()
                .and_then(|cfg| cfg.d_model)
                .or(self.d_model)
                .unwrap_or(576)
        } else {
            self.d_model.unwrap_or(512)
        };

        // Determine number of heads
        let num_heads = match architecture {
            Architecture::WhisperEncoder => self.encoder_attention_heads.or(self.num_heads),
            Architecture::WhisperDecoder => self.decoder_attention_heads.or(self.num_heads),
            Architecture::Idefics3 => self
                .text_config
                .as_ref()
                .and_then(|cfg| cfg.num_heads)
                .or(self.num_heads),
            Architecture::DFine => self
                .decoder_attention_heads
                .or(self.encoder_attention_heads)
                .or(self.num_heads),
            Architecture::Llama => self.num_heads,
            _ => self.num_heads,
        };

        // Determine number of layers
        let num_layers = match architecture {
            Architecture::WhisperEncoder => self.encoder_layers.or(self.num_hidden_layers),
            Architecture::WhisperDecoder => self.decoder_layers.or(self.num_hidden_layers),
            Architecture::Idefics3 => self
                .text_config
                .as_ref()
                .and_then(|cfg| cfg.num_hidden_layers)
                .or(self.num_hidden_layers),
            Architecture::DFine => self
                .decoder_layers
                .or(self.encoder_layers)
                .or(self.num_hidden_layers),
            _ => self.num_hidden_layers,
        };

        let mut config = ModelConfig::new(architecture.clone());
        config.hidden_dim = hidden_dim;
        config.num_heads = num_heads;
        config.num_layers = num_layers;
        if let Some(layer_norm_type) = self.layer_norm_type.as_deref() {
            if let Some(mode) = LayerNormMode::parse_alias(layer_norm_type) {
                config.layernorm_mode = mode;
            }
        }

        // Set output dimension based on vocab size for LLMs
        if matches!(
            architecture,
            Architecture::Llama | Architecture::TransformerDecoder
        ) {
            config.output_dim = self.vocab_size;
        }

        config
    }

    /// Detect architecture based on HF config.
    pub fn detect_architecture(&self) -> Architecture {
        // First check explicit architecture names
        for arch in &self.architectures {
            match arch.as_str() {
                "WhisperForConditionalGeneration" => return Architecture::WhisperEncoder,
                "WhisperModel" => return Architecture::WhisperEncoder,
                "CosyVoiceModel" | "CosyVoiceForConditionalGeneration" => {
                    return Architecture::CosyVoice
                }
                "KokoroTTS" | "KokoroModel" => return Architecture::Kokoro,
                "EfficientNetForImageClassification" => return Architecture::EfficientNet,
                "DFineForObjectDetection" => return Architecture::DFine,
                "RTDetrForObjectDetection" => return Architecture::DFine,
                "Idefics3ForConditionalGeneration" => return Architecture::Idefics3,
                "LlamaForCausalLM" | "MistralForCausalLM" | "GemmaForCausalLM" => {
                    return Architecture::Llama
                }
                "GPT2LMHeadModel" | "GPTNeoForCausalLM" | "GPTJForCausalLM" => {
                    return Architecture::TransformerDecoder
                }
                "BertModel" | "RobertaModel" | "DistilBertModel" => {
                    return Architecture::TransformerEncoder
                }
                _ => {}
            }
        }

        // Fall back to model_type
        match self.model_type.as_str() {
            "whisper" => Architecture::WhisperEncoder,
            "efficientnet" => Architecture::EfficientNet,
            "d_fine" | "rt_detr" => Architecture::DFine,
            "idefics3" => Architecture::Idefics3,
            "cosy_voice" => Architecture::CosyVoice,
            "kokoro" => Architecture::Kokoro,
            "llama" | "mistral" | "gemma" => Architecture::Llama,
            "gpt2" | "gpt_neo" | "gptj" => Architecture::TransformerDecoder,
            "bert" | "roberta" | "distilbert" => Architecture::TransformerEncoder,
            _ => Architecture::Unknown,
        }
    }
}
