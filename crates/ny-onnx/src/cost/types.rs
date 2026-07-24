// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Core types for static compute-cost analysis.

use ny_core::truncate_name;
use serde::Serialize;

/// Errors that can occur during static cost analysis.
pub type CostError = crate::analysis_error::AnalysisError;

/// Static compute and memory estimate for one layer.
#[derive(Debug, Clone, Serialize)]
pub struct LayerCost {
    /// Layer name from the model.
    pub name: String,
    /// Layer type (for example `Conv2d` or `ReLU`).
    pub layer_type: String,
    /// All non-constant output shapes for this layer.
    pub output_shapes: Vec<Vec<usize>>,
    /// Total number of output elements across all non-constant outputs.
    pub output_elements: u64,
    /// Estimated floating-point operations for this layer.
    pub flops: u64,
    /// Bytes read from runtime activation inputs for this layer.
    pub activation_input_bytes: u64,
    /// Bytes read from parameter tensors (weights, bias) for this layer.
    pub parameter_input_bytes: u64,
    /// Bytes occupied by this layer's non-constant outputs.
    pub output_bytes: u64,
    /// Total activation/parameter/output tensor traffic for this layer.
    pub total_tensor_traffic_bytes: u64,
    /// Stable timing family used to look up conservative calibration data.
    pub timing_family: String,
    /// Peak live activation bytes during this layer's execution.
    pub peak_live_activation_bytes: u64,
    /// Total estimated FLOPs up to and including this layer.
    pub cumulative_flops: u64,
}

/// Result of analyzing a full model's static compute cost.
#[derive(Debug, Clone, Serialize)]
pub struct CostResult {
    /// Per-layer cost estimates.
    pub layers: Vec<LayerCost>,
    /// Sum of all estimated layer FLOPs.
    pub total_flops: u64,
    /// Total parameter storage in bytes.
    pub parameter_bytes: u64,
    /// Peak live activation memory in bytes.
    pub peak_activation_bytes: u64,
    /// Peak activation bytes plus parameter bytes.
    pub peak_total_bytes: u64,
    /// Analysis assumptions that callers should surface to users.
    pub assumptions: Vec<String>,
}

impl CostResult {
    /// Human-readable summary for CLI output.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Static Cost Estimate".to_string());
        lines.push("====================".to_string());
        lines.push(format!(
            "{:<40} | {:>12} | {:>10} | {:>10}",
            "Layer", "FLOPs", "Out Bytes", "Peak Live"
        ));
        lines.push(format!(
            "{:-<40}-+-{:-<12}-+-{:-<10}-+-{:-<10}",
            "", "", "", ""
        ));

        for layer in &self.layers {
            lines.push(format!(
                "{:<40} | {:>12} | {:>10} | {:>10}",
                truncate_name(&layer.name, 40),
                format_human_count(layer.flops),
                format_human_bytes(layer.output_bytes),
                format_human_bytes(layer.peak_live_activation_bytes),
            ));
        }

        lines.push(String::new());
        lines.push(format!(
            "Total FLOPs: {}",
            format_human_count(self.total_flops)
        ));
        lines.push(format!(
            "Parameter bytes: {}",
            format_human_bytes(self.parameter_bytes)
        ));
        lines.push(format!(
            "Peak activation bytes: {}",
            format_human_bytes(self.peak_activation_bytes)
        ));
        lines.push(format!(
            "Peak total bytes (params + activations): {}",
            format_human_bytes(self.peak_total_bytes)
        ));
        if !self.assumptions.is_empty() {
            lines.push("Assumptions:".to_string());
            for assumption in &self.assumptions {
                lines.push(format!("  - {assumption}"));
            }
        }

        lines.join("\n")
    }
}

pub(crate) fn format_human_count(value: u64) -> String {
    const K: f64 = 1_000.0;
    const M: f64 = 1_000_000.0;
    const G: f64 = 1_000_000_000.0;

    if value >= G as u64 {
        format!("{:.2}G", value as f64 / G)
    } else if value >= M as u64 {
        format!("{:.2}M", value as f64 / M)
    } else if value >= K as u64 {
        format!("{:.2}K", value as f64 / K)
    } else {
        value.to_string()
    }
}

pub(crate) fn format_human_bytes(value: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;

    if value >= GB as u64 {
        format!("{:.2} GiB", value as f64 / GB)
    } else if value >= MB as u64 {
        format!("{:.2} MiB", value as f64 / MB)
    } else if value >= KB as u64 {
        format!("{:.2} KiB", value as f64 / KB)
    } else {
        format!("{value} B")
    }
}
