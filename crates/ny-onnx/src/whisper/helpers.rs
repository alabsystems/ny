// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::WhisperBlockScope;
use crate::LayerSpec;

fn name_is_block_scoped(lower: &str) -> bool {
    for token in [
        "blocks.", "blocks/", "blocks_", "block.", "block/", "block_", "layers.", "layers/",
        "layers_", "layer.", "layer/", "layer_",
    ] {
        if let Some(pos) = lower.find(token) {
            if lower[pos + token.len()..]
                .chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_digit())
            {
                return true;
            }
        }
    }
    false
}

fn name_is_ln_post(lower: &str) -> bool {
    lower.contains("ln_post") || lower.contains("/ln_post/")
}

fn contains_ln_f_token(lower: &str) -> bool {
    let mut search_start = 0;
    while let Some(pos) = lower[search_start..].find("ln_f") {
        let idx = search_start + pos;
        let before = idx.checked_sub(1).and_then(|i| lower.as_bytes().get(i));
        let after = lower.as_bytes().get(idx + "ln_f".len());
        let before_ok = before.map_or(true, |b| !b.is_ascii_alphanumeric());
        let after_ok = after.map_or(true, |b| !b.is_ascii_alphanumeric());
        if before_ok && after_ok {
            return true;
        }
        search_start = idx + 1;
    }
    false
}

fn is_explicit_ln_post(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("ln_post")
        || lower.contains("/ln_post/")
        || contains_ln_f_token(&lower)
        || lower.contains("post_norm")
        || lower.contains("post_layer_norm")
        || lower.contains("post_layernorm")
        || lower.contains("final_layer_norm")
        || lower.contains("final_layernorm")
}

fn is_layer_norm_fallback(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    (lower.contains("layer_norm") || lower.contains("layernorm"))
        && !lower.contains("self_attn_layer_norm")
        && !name_is_block_scoped(&lower)
}

fn in_scope(name: &str, scope: WhisperBlockScope) -> bool {
    match scope {
        WhisperBlockScope::All => true,
        WhisperBlockScope::Encoder | WhisperBlockScope::Decoder => {
            name_matches_scope(name, scope) || name_is_unscoped(name)
        }
    }
}

fn explicit_matches(name: &str, scope: WhisperBlockScope) -> bool {
    let lower = name.to_ascii_lowercase();
    if !is_explicit_ln_post(name) || !in_scope(name, scope) {
        return false;
    }
    if name_is_block_scoped(&lower) {
        return name_is_ln_post(&lower);
    }
    true
}

fn fallback_matches(name: &str, scope: WhisperBlockScope) -> bool {
    is_layer_norm_fallback(name) && in_scope(name, scope)
}

pub(super) fn layer_has_explicit_ln_post_marker(
    layer: &LayerSpec,
    scope: WhisperBlockScope,
) -> bool {
    if explicit_matches(&layer.name, scope) {
        return true;
    }
    if layer
        .inputs
        .iter()
        .any(|name| explicit_matches(name, scope))
    {
        return true;
    }
    if layer
        .outputs
        .iter()
        .any(|name| explicit_matches(name, scope))
    {
        return true;
    }
    if let Some(weights) = &layer.weights {
        if explicit_matches(&weights.name, scope) {
            return true;
        }
    }
    false
}

pub(super) fn layer_has_ln_post_marker(layer: &LayerSpec, scope: WhisperBlockScope) -> bool {
    if layer_has_explicit_ln_post_marker(layer, scope) {
        return true;
    }

    let looks_like_param = |name: &str| {
        let lower = name.to_ascii_lowercase();
        lower.contains("weight")
            || lower.contains("bias")
            || lower.contains("ny")
            || lower.contains("beta")
            || lower.contains("scale")
    };

    if fallback_matches(&layer.name, scope) {
        return true;
    }
    if let Some(weights) = &layer.weights {
        if fallback_matches(&weights.name, scope) {
            return true;
        }
    }
    for name in layer.inputs.iter().filter(|name| looks_like_param(name)) {
        if fallback_matches(name, scope) {
            return true;
        }
    }

    false
}

pub(super) fn name_is_unscoped(name: &str) -> bool {
    !name_matches_scope(name, WhisperBlockScope::Encoder)
        && !name_matches_scope(name, WhisperBlockScope::Decoder)
}

pub(super) fn name_matches_scope(name: &str, scope: WhisperBlockScope) -> bool {
    let lower = name.to_ascii_lowercase();
    match scope {
        WhisperBlockScope::All => true,
        WhisperBlockScope::Encoder => {
            lower.contains("/encoder/")
                || lower.contains("encoder/")
                || lower.contains("encoder.")
                || lower.contains("/encoder.")
        }
        WhisperBlockScope::Decoder => {
            lower.contains("/decoder/")
                || lower.contains("decoder/")
                || lower.contains("decoder.")
                || lower.contains("/decoder.")
        }
    }
}

pub(super) fn layer_any_name_matches<F: Fn(&str) -> bool>(layer: &LayerSpec, predicate: F) -> bool {
    if predicate(&layer.name) {
        return true;
    }
    if layer.inputs.iter().any(|name| predicate(name)) {
        return true;
    }
    if layer.outputs.iter().any(|name| predicate(name)) {
        return true;
    }
    if let Some(weights) = &layer.weights {
        if predicate(&weights.name) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod attention_tests {
    use super::*;
    use crate::{LayerSpec, WeightRef};
    use ny_core::LayerType;
    use std::collections::HashMap;

    fn layer_with_names(
        name: &str,
        inputs: &[&str],
        outputs: &[&str],
        weight: Option<&str>,
    ) -> LayerSpec {
        LayerSpec {
            name: name.to_string(),
            layer_type: LayerType::Linear,
            inputs: inputs.iter().map(|value| (*value).to_string()).collect(),
            outputs: outputs.iter().map(|value| (*value).to_string()).collect(),
            weights: weight.map(|name| WeightRef {
                name: name.to_string(),
                shape: vec![1],
                original_dtype: crate::DataType::Float32,
            }),
            attributes: HashMap::new(),
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_name_matches_scope_encoder_patterns() {
        let encoder_names = [
            "encoder/block",
            "/encoder/block",
            "encoder.block",
            "/encoder.block",
        ];
        for name in encoder_names {
            assert!(name_matches_scope(name, WhisperBlockScope::Encoder));
            assert!(!name_matches_scope(name, WhisperBlockScope::Decoder));
            assert!(name_matches_scope(name, WhisperBlockScope::All));
        }

        assert!(!name_matches_scope(
            "blocks/0/attn",
            WhisperBlockScope::Encoder
        ));
        assert!(!name_matches_scope(
            "blocks/0/attn",
            WhisperBlockScope::Decoder
        ));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_name_matches_scope_case_insensitive() {
        let encoder_name = "Encoder/blocks.0/attn";
        assert!(name_matches_scope(encoder_name, WhisperBlockScope::Encoder));
        assert!(!name_matches_scope(
            encoder_name,
            WhisperBlockScope::Decoder
        ));

        let decoder_name = "DECODER/blocks.1/attn";
        assert!(name_matches_scope(decoder_name, WhisperBlockScope::Decoder));
        assert!(!name_matches_scope(
            decoder_name,
            WhisperBlockScope::Encoder
        ));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_name_matches_scope_decoder_patterns() {
        let decoder_names = [
            "decoder/block",
            "/decoder/block",
            "decoder.block",
            "/decoder.block",
        ];
        for name in decoder_names {
            assert!(name_matches_scope(name, WhisperBlockScope::Decoder));
            assert!(!name_matches_scope(name, WhisperBlockScope::Encoder));
            assert!(name_matches_scope(name, WhisperBlockScope::All));
        }

        assert!(!name_matches_scope(
            "blocks/0/attn",
            WhisperBlockScope::Decoder
        ));
        assert!(!name_matches_scope(
            "blocks/0/attn",
            WhisperBlockScope::Encoder
        ));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_layer_has_ln_post_marker_unscoped() {
        let layer = layer_with_names("ln_post", &[], &[], None);
        assert!(layer_has_ln_post_marker(&layer, WhisperBlockScope::All));
        assert!(layer_has_ln_post_marker(&layer, WhisperBlockScope::Encoder));
        assert!(layer_has_ln_post_marker(&layer, WhisperBlockScope::Decoder));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_layer_has_ln_post_marker_ln_f_token() {
        let encoder_layer = layer_with_names("encoder/ln_f", &[], &[], None);
        assert!(layer_has_ln_post_marker(
            &encoder_layer,
            WhisperBlockScope::Encoder
        ));
        assert!(!layer_has_ln_post_marker(
            &encoder_layer,
            WhisperBlockScope::Decoder
        ));

        let negative_layer = layer_with_names("encoder/ln_ffn", &[], &[], None);
        assert!(!layer_has_ln_post_marker(
            &negative_layer,
            WhisperBlockScope::Encoder
        ));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_layer_has_ln_post_marker_layer_norm_fallback() {
        let layer = layer_with_names("layer_norm", &[], &[], None);
        assert!(layer_has_ln_post_marker(&layer, WhisperBlockScope::All));
        assert!(layer_has_ln_post_marker(&layer, WhisperBlockScope::Encoder));
        assert!(layer_has_ln_post_marker(&layer, WhisperBlockScope::Decoder));

        let final_layer = layer_with_names("final_layer_norm", &[], &[], None);
        assert!(layer_has_ln_post_marker(
            &final_layer,
            WhisperBlockScope::All
        ));
        assert!(layer_has_ln_post_marker(
            &final_layer,
            WhisperBlockScope::Encoder
        ));
        assert!(layer_has_ln_post_marker(
            &final_layer,
            WhisperBlockScope::Decoder
        ));

        let weight_layer = layer_with_names("noop", &[], &[], Some("layer_norm.weight"));
        assert!(layer_has_ln_post_marker(
            &weight_layer,
            WhisperBlockScope::All
        ));

        let block_final = layer_with_names("layers.0.final_layer_norm", &[], &[], None);
        assert!(!layer_has_ln_post_marker(
            &block_final,
            WhisperBlockScope::All
        ));

        let block_layer = layer_with_names("layers.0.self_attn_layer_norm", &[], &[], None);
        assert!(!layer_has_ln_post_marker(
            &block_layer,
            WhisperBlockScope::All
        ));

        let top_level = layer_with_names("layers_norm.layer_norm", &[], &[], None);
        assert!(layer_has_ln_post_marker(&top_level, WhisperBlockScope::All));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_layer_has_ln_post_marker_scoped_names() {
        let encoder_layer = layer_with_names("/encoder/ln_post", &[], &[], None);
        assert!(layer_has_ln_post_marker(
            &encoder_layer,
            WhisperBlockScope::Encoder
        ));
        assert!(!layer_has_ln_post_marker(
            &encoder_layer,
            WhisperBlockScope::Decoder
        ));

        let decoder_layer = layer_with_names("/decoder/ln_post", &[], &[], None);
        assert!(layer_has_ln_post_marker(
            &decoder_layer,
            WhisperBlockScope::Decoder
        ));
        assert!(!layer_has_ln_post_marker(
            &decoder_layer,
            WhisperBlockScope::Encoder
        ));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_layer_has_ln_post_marker_inputs_outputs_weights() {
        let input_layer = layer_with_names("noop", &["/decoder/ln_post/input"], &[], None);
        assert!(layer_has_ln_post_marker(
            &input_layer,
            WhisperBlockScope::Decoder
        ));
        assert!(!layer_has_ln_post_marker(
            &input_layer,
            WhisperBlockScope::Encoder
        ));

        let output_layer = layer_with_names("noop", &[], &["encoder/ln_post/output"], None);
        assert!(layer_has_ln_post_marker(
            &output_layer,
            WhisperBlockScope::Encoder
        ));
        assert!(!layer_has_ln_post_marker(
            &output_layer,
            WhisperBlockScope::Decoder
        ));

        let weight_layer = layer_with_names("noop", &[], &[], Some("encoder/ln_post.weight"));
        assert!(layer_has_ln_post_marker(
            &weight_layer,
            WhisperBlockScope::Encoder
        ));
        assert!(!layer_has_ln_post_marker(
            &weight_layer,
            WhisperBlockScope::Decoder
        ));

        let negative_layer = layer_with_names("ln_pre", &["encoder/ln_pre"], &[], None);
        assert!(!layer_has_ln_post_marker(
            &negative_layer,
            WhisperBlockScope::All
        ));

        let layer_norm_output = layer_with_names("noop", &["layer_norm_7"], &[], None);
        assert!(!layer_has_ln_post_marker(
            &layer_norm_output,
            WhisperBlockScope::All
        ));

        let ny_input = layer_with_names("noop", &["layer_norm.ny"], &[], None);
        assert!(layer_has_ln_post_marker(&ny_input, WhisperBlockScope::All));
    }
}
