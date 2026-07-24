// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::LayerSpec;
use std::collections::HashMap;

/// Match layer names between two models using heuristics.
///
/// Returns a list of (name_a, name_b) pairs for corresponding layers.
pub fn match_layer_names(
    layers_a: &[LayerSpec],
    layers_b: &[LayerSpec],
    explicit_mapping: &HashMap<String, String>,
) -> Vec<(String, Option<String>)> {
    let mut matches = Vec::new();

    // First, apply explicit mappings
    let b_names: HashMap<&str, &LayerSpec> =
        layers_b.iter().map(|l| (l.name.as_str(), l)).collect();

    for layer_a in layers_a {
        let name_a = &layer_a.name;

        // Check explicit mapping first
        if let Some(name_b) = explicit_mapping.get(name_a) {
            matches.push((name_a.clone(), Some(name_b.clone())));
            continue;
        }

        // Try exact match
        if b_names.contains_key(name_a.as_str()) {
            matches.push((name_a.clone(), Some(name_a.clone())));
            continue;
        }

        // Try fuzzy matching based on:
        // 1. Layer type match
        // 2. Position in network
        // 3. Suffix/prefix patterns (e.g., "_0" vs ".0")

        // Normalize name for matching
        let normalized_a = normalize_layer_name(name_a);

        let mut best_match: Option<&str> = None;
        for layer_b in layers_b {
            let normalized_b = normalize_layer_name(&layer_b.name);
            if normalized_a == normalized_b && layer_a.layer_type == layer_b.layer_type {
                best_match = Some(&layer_b.name);
                break;
            }
        }

        matches.push((name_a.clone(), best_match.map(|s| s.to_string())));
    }

    matches
}

/// Normalize a layer name for fuzzy matching.
pub(crate) fn normalize_layer_name(name: &str) -> String {
    let normalized: String = name
        .to_lowercase()
        .replace('_', ".")
        .replace("layer", "")
        .replace("block", "")
        .replace("module", "")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '.')
        .collect();
    // Remove consecutive dots and leading/trailing dots
    normalized
        .split('.')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(".")
}
