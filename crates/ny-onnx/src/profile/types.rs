// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Core types for bound width profiling.

use std::cmp::Ordering;

use ny_tensor::BoundedTensor;

use super::truncate_name;

/// Errors that can occur during bound profiling.
///
/// Type alias for the shared [`AnalysisError`](crate::analysis_error::AnalysisError).
pub type ProfileError = crate::analysis_error::AnalysisError;

/// Bound width status indicators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundStatus {
    /// Bounds are tight and stable
    Tight,
    /// Bounds are moderate
    Moderate,
    /// Bounds are wide - verification getting harder
    Wide,
    /// Bounds are very wide - verification difficult
    VeryWide,
    /// Bounds have overflowed (infinity/NaN)
    Overflow,
}

impl std::fmt::Display for BoundStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BoundStatus::Tight => write!(f, "TIGHT"),
            BoundStatus::Moderate => write!(f, "MODERATE"),
            BoundStatus::Wide => write!(f, "WIDE"),
            BoundStatus::VeryWide => write!(f, "VERY WIDE"),
            BoundStatus::Overflow => write!(f, "OVERFLOW"),
        }
    }
}

impl BoundStatus {
    /// Determine status from bound width relative to input epsilon.
    pub(super) fn from_width(width: f32, input_epsilon: f32) -> Self {
        if !width.is_finite() {
            BoundStatus::Overflow
        } else {
            let ratio = width / (2.0 * input_epsilon);
            if ratio < 10.0 {
                BoundStatus::Tight
            } else if ratio < 100.0 {
                BoundStatus::Moderate
            } else if ratio < 10000.0 {
                BoundStatus::Wide
            } else {
                BoundStatus::VeryWide
            }
        }
    }
}

/// Result of profiling a single layer's bounds.
#[derive(Debug, Clone)]
pub struct LayerProfile {
    /// Layer name from the model.
    pub name: String,
    /// Layer type (e.g., "Linear", "ReLU", "Softmax").
    pub layer_type: String,
    /// Input bound width (max across all elements).
    pub input_width: f32,
    /// Output bound width (max across all elements).
    pub output_width: f32,
    /// Mean output bound width.
    pub mean_output_width: f32,
    /// Median output bound width.
    pub median_output_width: f32,
    /// Width growth ratio (output/input).
    pub growth_ratio: f32,
    /// Cumulative width from input (output_width / initial_epsilon * 2).
    pub cumulative_expansion: f32,
    /// Output shape.
    pub output_shape: Vec<usize>,
    /// Number of elements in output.
    pub num_elements: usize,
    /// Bound status indicator.
    pub status: BoundStatus,
}

impl LayerProfile {
    /// Check if this layer is a "choke point" where bounds explode.
    pub fn is_choke_point(&self, growth_threshold: f32) -> bool {
        self.growth_ratio > growth_threshold
    }
}

/// Result of a full bound profiling analysis.
#[derive(Debug, Clone)]
pub struct ProfileResult {
    /// Per-layer bound profiles.
    pub layers: Vec<LayerProfile>,
    /// Input perturbation epsilon.
    pub input_epsilon: f32,
    /// Initial input bound width (2 * epsilon).
    pub initial_width: f32,
    /// Final output bound width.
    pub final_width: f32,
    /// Total expansion (final_width / initial_width).
    pub total_expansion: f32,
    /// Layer with highest growth ratio.
    pub max_growth_layer: Option<usize>,
    /// Maximum single-layer growth ratio.
    pub max_growth_ratio: f32,
    /// Index of first layer with overflow.
    pub overflow_at_layer: Option<usize>,
    /// Verification difficulty score (0-100).
    pub difficulty_score: f32,
}

impl ProfileResult {
    /// Get a summary of the profiling.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Bound Width Profile".to_string());
        lines.push("===================".to_string());
        lines.push(format!(
            "{:<40} | {:>10} | {:>10} | {:>8} | {:>10} | Status",
            "Layer", "In Width", "Out Width", "Growth", "Cumul."
        ));
        lines.push(format!(
            "{:-<40}-+-{:-<10}-+-{:-<10}-+-{:-<8}-+-{:-<10}-+--------",
            "", "", "", "", ""
        ));

        for (i, layer) in self.layers.iter().enumerate() {
            let is_max = self.max_growth_layer == Some(i);
            let marker = if is_max { " <<<" } else { "" };

            lines.push(format!(
                "{:<40} | {:>10.3e} | {:>10.3e} | {:>8.2}x | {:>10.2}x | {}{}",
                truncate_name(&layer.name, 40),
                layer.input_width,
                layer.output_width,
                layer.growth_ratio,
                layer.cumulative_expansion,
                layer.status,
                marker
            ));
        }

        lines.push(String::new());
        lines.push(format!(
            "Initial width: {:.2e} (epsilon = {:.2e})",
            self.initial_width, self.input_epsilon
        ));
        lines.push(format!("Final width: {:.2e}", self.final_width));
        lines.push(format!("Total expansion: {:.2}x", self.total_expansion));
        lines.push(format!(
            "Max growth layer: {} ({:.2}x)",
            self.max_growth_layer
                .and_then(|i| self.layers.get(i))
                .map(|l| l.name.as_str())
                .unwrap_or("N/A"),
            self.max_growth_ratio
        ));
        lines.push(format!(
            "Verification difficulty: {:.0}/100",
            self.difficulty_score
        ));

        if let Some(idx) = self.overflow_at_layer {
            lines.push(format!(
                "WARNING: Overflow at layer {} ({})",
                idx,
                self.layers
                    .get(idx)
                    .map(|l| l.name.as_str())
                    .unwrap_or("unknown")
            ));
        }

        lines.join("\n")
    }

    /// Get layers sorted by growth ratio (highest first, NaN last).
    ///
    /// Uses NaN-safe descending ordering (#2601).
    pub fn layers_by_growth(&self) -> Vec<&LayerProfile> {
        let mut sorted: Vec<_> = self.layers.iter().collect();
        sorted.sort_by(|a, b| {
            // NaN-last descending: finite descending, then NaN
            match (a.growth_ratio.is_nan(), b.growth_ratio.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => b.growth_ratio.total_cmp(&a.growth_ratio),
            }
        });
        sorted
    }

    /// Get "choke points" - layers with high growth ratio.
    pub fn choke_points(&self, threshold: f32) -> Vec<&LayerProfile> {
        self.layers
            .iter()
            .filter(|l| l.growth_ratio > threshold)
            .collect()
    }

    /// Get layers with wide or very wide bounds.
    pub fn problematic_layers(&self) -> Vec<&LayerProfile> {
        self.layers
            .iter()
            .filter(|l| {
                matches!(
                    l.status,
                    BoundStatus::Wide | BoundStatus::VeryWide | BoundStatus::Overflow
                )
            })
            .collect()
    }
}

/// Configuration for bound profiling.
#[derive(Debug, Clone)]
pub struct ProfileConfig {
    /// Input perturbation epsilon.
    pub epsilon: f32,
    ///
    /// Defaults to `true` because profiling is a diagnostic pass and should
    /// continue reporting downstream width-growth sites after the first
    /// overflow.
    pub continue_after_overflow: bool,
    /// Custom input tensor (None = zeros with epsilon bounds).
    pub input: Option<BoundedTensor>,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.01,
            continue_after_overflow: true,
            input: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layer(name: &str, growth_ratio: f32) -> LayerProfile {
        LayerProfile {
            name: name.to_string(),
            layer_type: "Linear".to_string(),
            input_width: 1.0,
            output_width: growth_ratio,
            mean_output_width: growth_ratio * 0.8,
            median_output_width: growth_ratio * 0.7,
            growth_ratio,
            cumulative_expansion: growth_ratio,
            output_shape: vec![10],
            num_elements: 10,
            status: BoundStatus::Tight,
        }
    }

    /// Regression test for #2601: NaN growth ratios must sort last.
    #[test]
    fn test_layers_by_growth_nan_last_2601() {
        let result = ProfileResult {
            layers: vec![
                make_layer("nan_layer", f32::NAN),
                make_layer("high", 10.0),
                make_layer("low", 2.0),
            ],
            input_epsilon: 0.01,
            initial_width: 0.02,
            final_width: 10.0,
            total_expansion: 500.0,
            max_growth_layer: Some(0),
            max_growth_ratio: 10.0,
            overflow_at_layer: None,
            difficulty_score: 50.0,
        };

        let sorted = result.layers_by_growth();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].name, "high");
        assert_eq!(sorted[1].name, "low");
        assert_eq!(sorted[2].name, "nan_layer");
        assert!(sorted[2].growth_ratio.is_nan());
    }
}
