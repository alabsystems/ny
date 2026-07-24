// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::helpers::layer_any_name_matches;
use crate::Network;
use tracing::warn;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WhisperBlockScope {
    All,
    Encoder,
    Decoder,
}

pub(crate) fn detect_block_scope(network: &Network) -> WhisperBlockScope {
    let encoder_prefixes = [
        "/encoder/blocks.",
        "/encoder/layers.",
        "encoder/blocks.",
        "encoder/layers.",
        "encoder.blocks.",
        "encoder.layers.",
    ];
    let decoder_prefixes = [
        "/decoder/blocks.",
        "/decoder/layers.",
        "decoder/blocks.",
        "decoder/layers.",
        "decoder.blocks.",
        "decoder.layers.",
    ];
    let mut has_encoder = false;
    let mut has_decoder = false;

    for layer in &network.layers {
        if layer_any_name_matches(layer, |name| {
            encoder_prefixes.iter().any(|prefix| name.contains(prefix))
        }) {
            has_encoder = true;
        }
        if layer_any_name_matches(layer, |name| {
            decoder_prefixes.iter().any(|prefix| name.contains(prefix))
        }) {
            has_decoder = true;
        }
        if has_encoder && has_decoder {
            break;
        }
    }

    match (has_encoder, has_decoder) {
        (true, true) => {
            warn!("Whisper encoder+decoder blocks detected; parsing encoder blocks only");
            WhisperBlockScope::Encoder
        }
        (true, false) => WhisperBlockScope::Encoder,
        (false, true) => {
            warn!("Whisper encoder blocks not found; parsing decoder-style blocks");
            WhisperBlockScope::Decoder
        }
        (false, false) => WhisperBlockScope::All,
    }
}
