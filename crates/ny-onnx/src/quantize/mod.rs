// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Quantization safety analysis for neural networks.
//!
//! This module analyzes whether a neural network's layer outputs can safely
//! be quantized to lower precision formats (float16, int8) without overflow
//! or significant precision loss.
//!
//! ## Key Analysis
//!
//! For each layer, we compute output bounds and check:
//! - **float16 safety**: Can outputs fit in [-65504, 65504]?
//! - **int8 safety**: Can scaled outputs fit in [-128, 127]?
//! - **Denormal risk**: Are outputs in the denormal range for float16?
//!
//! ## Usage
//!
//! ```rust,no_run
//! use ny_onnx::quantize::{analyze_quantization, QuantizeConfig};
//!
//! let config = QuantizeConfig::default();
//! let result = analyze_quantization("model.onnx", &config).unwrap();
//! println!("{}", result.summary());
//! ```

mod graph;
mod model;

use ndarray::{ArrayD, IxDyn};
use ny_core::truncate_name;
use ny_tensor::BoundedTensor;

pub use graph::analyze_quantization_graph;
pub use model::{analyze_quantization, analyze_quantization_model};

/// float16 representation limits
const FLOAT16_MAX: f32 = 65504.0;
const FLOAT16_MIN_POSITIVE: f32 = 6.10e-5; // Minimum positive normal float16

/// Errors that can occur during quantization analysis.
///
/// Type alias for the shared [`AnalysisError`](crate::analysis_error::AnalysisError).
pub type QuantizeError = crate::analysis_error::AnalysisError;

/// Quantization format being checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    Float16,
    Int8,
}

impl std::fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantFormat::Float16 => write!(f, "float16"),
            QuantFormat::Int8 => write!(f, "int8"),
        }
    }
}

/// Safety status for a quantization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantSafety {
    /// Safe to quantize - all values within representable range
    Safe,
    /// Warning - values may be in denormal range (precision loss)
    Denormal,
    /// Warning - values require careful scaling for int8
    ScalingRequired,
    /// Unsafe - values may overflow the format
    Overflow,
    /// Unknown - bounds are infinite or NaN
    Unknown,
}

impl std::fmt::Display for QuantSafety {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantSafety::Safe => write!(f, "SAFE"),
            QuantSafety::Denormal => write!(f, "DENORMAL"),
            QuantSafety::ScalingRequired => write!(f, "SCALE"),
            QuantSafety::Overflow => write!(f, "OVERFLOW"),
            QuantSafety::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

/// Result of analyzing a single layer's quantization safety.
#[derive(Debug, Clone)]
pub struct LayerQuantization {
    /// Layer name from the model.
    pub name: String,
    /// Layer type (e.g., "Linear", "ReLU", "Softmax").
    pub layer_type: String,
    /// Minimum output bound across all elements.
    pub min_bound: f32,
    /// Maximum output bound across all elements.
    pub max_bound: f32,
    /// Maximum absolute value in bounds.
    pub max_abs: f32,
    /// Output shape.
    pub output_shape: Vec<usize>,
    /// float16 safety assessment.
    pub float16_safety: QuantSafety,
    /// int8 safety assessment.
    pub int8_safety: QuantSafety,
    /// Suggested int8 scale factor (if applicable).
    pub int8_scale: Option<f32>,
    /// Whether any output bounds are infinite or NaN (actual numerical overflow).
    pub has_overflow: bool,
    /// Whether propagation failed (e.g., shape mismatch).
    /// If true, safety assessments are unreliable (fallback input bounds used as output).
    pub propagation_failed: bool,
}

impl LayerQuantization {
    /// Check if layer is safe for the given format.
    ///
    /// Returns `false` if propagation failed, since the assessment is unreliable.
    pub fn is_safe_for(&self, format: QuantFormat) -> bool {
        if self.propagation_failed {
            return false;
        }
        match format {
            QuantFormat::Float16 => matches!(self.float16_safety, QuantSafety::Safe),
            QuantFormat::Int8 => matches!(
                self.int8_safety,
                QuantSafety::Safe | QuantSafety::ScalingRequired
            ),
        }
    }
}

/// Result of a full quantization analysis.
#[derive(Debug, Clone)]
pub struct QuantizeResult {
    /// Per-layer quantization analysis.
    pub layers: Vec<LayerQuantization>,
    /// Overall float16 safety (all layers must be safe).
    pub float16_safe: bool,
    /// Overall int8 safety (all layers must be safe with scaling).
    pub int8_safe: bool,
    /// Number of layers with float16 overflow risk.
    pub float16_overflow_count: usize,
    /// Number of layers with int8 overflow risk.
    pub int8_overflow_count: usize,
    /// Number of layers in float16 denormal range.
    pub denormal_count: usize,
    /// Input perturbation epsilon used.
    pub input_epsilon: f32,
}

#[deprecated(note = "use QuantizeResult")]
pub type QuantizationResult = QuantizeResult;

impl QuantizeResult {
    /// Get a summary of the analysis.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Quantization Safety Analysis".to_string());
        lines.push("============================".to_string());
        lines.push(format!(
            "{:<40} | {:>12} | {:>12} | {:>8} | {:>8}",
            "Layer", "Min", "Max", "F16", "I8"
        ));
        lines.push(format!(
            "{:-<40}-+-{:-<12}-+-{:-<12}-+-{:-<8}-+-{:-<8}",
            "", "", "", "", ""
        ));

        for layer in &self.layers {
            lines.push(format!(
                "{:<40} | {:>12.3e} | {:>12.3e} | {:>8} | {:>8}",
                truncate_name(&layer.name, 40),
                layer.min_bound,
                layer.max_bound,
                layer.float16_safety,
                layer.int8_safety
            ));
        }

        lines.push(String::new());
        lines.push(format!(
            "Float16: {} ({} layers with overflow risk, {} in denormal range)",
            if self.float16_safe { "SAFE" } else { "UNSAFE" },
            self.float16_overflow_count,
            self.denormal_count
        ));
        lines.push(format!(
            "Int8:    {} ({} layers with overflow risk)",
            if self.int8_safe { "SAFE" } else { "UNSAFE" },
            self.int8_overflow_count
        ));

        let skipped_count = self.layers.iter().filter(|l| l.propagation_failed).count();
        if skipped_count > 0 {
            lines.push(format!(
                "NOTE: {} layer(s) skipped due to propagation failure (assessments unreliable)",
                skipped_count
            ));
        }

        lines.join("\n")
    }

    /// Get layers that are unsafe for float16.
    pub fn float16_unsafe_layers(&self) -> Vec<&LayerQuantization> {
        self.layers
            .iter()
            .filter(|l| {
                matches!(
                    l.float16_safety,
                    QuantSafety::Overflow | QuantSafety::Unknown
                )
            })
            .collect()
    }

    /// Get layers that are unsafe for int8.
    pub fn int8_unsafe_layers(&self) -> Vec<&LayerQuantization> {
        self.layers
            .iter()
            .filter(|l| matches!(l.int8_safety, QuantSafety::Overflow | QuantSafety::Unknown))
            .collect()
    }

    /// Get layers with denormal warning.
    pub fn denormal_layers(&self) -> Vec<&LayerQuantization> {
        self.layers
            .iter()
            .filter(|l| matches!(l.float16_safety, QuantSafety::Denormal))
            .collect()
    }
}

/// Configuration for quantization analysis.
#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    /// Input perturbation epsilon.
    pub epsilon: f32,
    ///
    /// Defaults to `true` because quantization analysis is a diagnostic survey
    /// and should continue collecting per-layer failures after the first
    /// overflow.
    pub continue_after_overflow: bool,
    /// Custom input tensor (None = zeros with epsilon bounds).
    pub input: Option<BoundedTensor>,
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.01,
            continue_after_overflow: true,
            input: None,
        }
    }
}

/// Assess float16 safety for given bounds.
fn assess_float16(min_bound: f32, max_bound: f32) -> QuantSafety {
    if !min_bound.is_finite() || !max_bound.is_finite() {
        return QuantSafety::Unknown;
    }

    let max_abs = min_bound.abs().max(max_bound.abs());

    if max_abs > FLOAT16_MAX {
        QuantSafety::Overflow
    } else if max_abs > 0.0 && max_abs < FLOAT16_MIN_POSITIVE {
        // Values in denormal range - precision loss likely
        QuantSafety::Denormal
    } else {
        QuantSafety::Safe
    }
}

/// Assess int8 safety for given bounds.
fn assess_int8(min_bound: f32, max_bound: f32) -> (QuantSafety, Option<f32>) {
    if !min_bound.is_finite() || !max_bound.is_finite() {
        return (QuantSafety::Unknown, None);
    }

    // Calculate the scale needed to fit in int8
    let max_abs = min_bound.abs().max(max_bound.abs());

    if max_abs == 0.0 {
        // Zero tensor - safe
        return (QuantSafety::Safe, Some(1.0));
    }

    // Scale to fit in [-127, 127] (reserving -128 for special values)
    let scale = 127.0 / max_abs;

    if scale < 1e-10 {
        // Scale too small - overflow
        (QuantSafety::Overflow, None)
    } else if scale < 1.0 {
        // Needs scaling down
        (QuantSafety::ScalingRequired, Some(scale))
    } else if scale > 1e6 {
        // Very small values - precision issues
        (QuantSafety::ScalingRequired, Some(scale))
    } else {
        (QuantSafety::Safe, Some(scale))
    }
}

fn make_default_input(
    input_shape: &[usize],
    epsilon: f32,
    context: &'static str,
) -> Result<BoundedTensor, QuantizeError> {
    let data = ArrayD::zeros(IxDyn(input_shape));
    BoundedTensor::from_epsilon(data, epsilon).map_err(|e| QuantizeError::propagation(context, e))
}

fn build_layer_quantization(
    name: String,
    layer_type: String,
    output: &BoundedTensor,
    propagation_failed: bool,
) -> LayerQuantization {
    let min_bound = output.lower().iter().cloned().fold(f32::INFINITY, f32::min);
    let max_bound = output
        .upper()
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let max_abs = min_bound.abs().max(max_bound.abs());
    let has_overflow = !propagation_failed && (!min_bound.is_finite() || !max_bound.is_finite());
    let float16_safety = if propagation_failed {
        QuantSafety::Unknown
    } else {
        assess_float16(min_bound, max_bound)
    };
    let (int8_safety, int8_scale) = if propagation_failed {
        (QuantSafety::Unknown, None)
    } else {
        assess_int8(min_bound, max_bound)
    };

    LayerQuantization {
        name,
        layer_type,
        min_bound,
        max_bound,
        max_abs,
        output_shape: output.shape().to_vec(),
        float16_safety,
        int8_safety,
        int8_scale,
        has_overflow,
        propagation_failed,
    }
}

fn tally_layer_quantization(
    layer: &LayerQuantization,
    float16_overflow_count: &mut usize,
    int8_overflow_count: &mut usize,
    denormal_count: &mut usize,
) {
    if matches!(
        layer.float16_safety,
        QuantSafety::Overflow | QuantSafety::Unknown
    ) {
        *float16_overflow_count += 1;
    }
    if matches!(layer.float16_safety, QuantSafety::Denormal) {
        *denormal_count += 1;
    }
    if matches!(
        layer.int8_safety,
        QuantSafety::Overflow | QuantSafety::Unknown
    ) {
        *int8_overflow_count += 1;
    }
}

#[cfg(test)]
mod graph_tests;

#[cfg(test)]
mod tests;
