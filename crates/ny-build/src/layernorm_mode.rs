// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_propagate::layers::LayerNormMode;

use crate::{AttributeValue, LayerSpec};

pub(crate) fn layernorm_mode_from_attrs(spec: &LayerSpec) -> LayerNormMode {
    let value = spec
        .attributes
        .get("layernorm_mode")
        .or_else(|| spec.attributes.get("mode"));
    match value {
        Some(AttributeValue::String(mode)) => {
            LayerNormMode::parse_alias(mode).unwrap_or(LayerNormMode::Standard)
        }
        Some(AttributeValue::Int(value)) if *value != 0 => LayerNormMode::MeanOnly,
        Some(AttributeValue::Float(value)) if *value != 0.0 => LayerNormMode::MeanOnly,
        _ => LayerNormMode::Standard,
    }
}
