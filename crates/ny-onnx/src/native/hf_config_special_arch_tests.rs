// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_to_model_config_cosy_voice_detected() {
    // CosyVoice can be detected from the HfConfig architectures field.
    let json = r#"{"architectures": ["CosyVoiceModel"], "hidden_size": 512}"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    let config = hf_config.to_model_config();
    assert_eq!(config.architecture, Architecture::CosyVoice);
    assert_eq!(config.hidden_dim, 512);
}

#[ntest::timeout(10000)]
#[test]
fn test_hf_config_to_model_config_kokoro_detected() {
    // Kokoro can be detected from the HfConfig architectures field.
    let json = r#"{"architectures": ["KokoroTTS"], "hidden_size": 512}"#;
    let hf_config: HfConfig = serde_json::from_str(json).expect("Failed to parse");
    let config = hf_config.to_model_config();
    assert_eq!(config.architecture, Architecture::Kokoro);
    assert_eq!(config.hidden_dim, 512);
}
