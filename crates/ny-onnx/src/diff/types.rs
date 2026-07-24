// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::{LayerSpec, TensorSpec};
use ndarray::ArrayD;
use ny_core::LayerType;
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during model diffing.
#[derive(Error, Debug)]
pub enum DiffError {
    #[error("Failed to load model: {0}")]
    LoadError(String),

    #[error("ONNX Runtime support not enabled. Rebuild with `--features ort`")]
    OrtUnavailable,

    #[cfg(feature = "ort")]
    #[error("ONNX Runtime error: {0}")]
    OrtError(#[from] ort::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("NPY read error: {0}")]
    NpyError(String),

    #[error("Input shape mismatch: model A {model_a:?} vs model B {model_b:?}")]
    InputShapeMismatch {
        model_a: Vec<i64>,
        model_b: Vec<i64>,
    },

    #[error("Layer not found: {0}")]
    LayerNotFound(String),

    #[error("No layers to compare")]
    NoLayers,
}

/// Result of comparing a single layer between two models.
#[derive(Debug, Clone)]
pub struct LayerComparison {
    /// Layer name (from model A, matched to model B).
    pub name: String,
    /// Name in model B (if different from model A).
    pub name_b: Option<String>,
    /// Maximum absolute difference between outputs.
    pub max_diff: f32,
    /// Mean absolute difference between outputs.
    pub mean_diff: f32,
    /// Whether this layer exceeds the tolerance.
    pub exceeds_tolerance: bool,
    /// Output shape from model A.
    pub shape_a: Vec<usize>,
    /// Output shape from model B.
    pub shape_b: Vec<usize>,
}

/// Status of a layer comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffStatus {
    /// Outputs match within tolerance.
    Ok,
    /// First layer where drift is detected (within 10x tolerance).
    DriftStarts,
    /// Layer exceeds tolerance.
    ExceedsTolerance,
    /// Shapes don't match.
    ShapeMismatch,
}

/// Pattern of divergence detected by root cause analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum DivergencePattern {
    /// exp() overflow/underflow differences (large logits near ±88)
    ExpPrecision {
        /// Maximum logit value observed before exp
        max_logit: f32,
        /// Whether overflow (>88) or underflow (<-88) boundary
        is_overflow: bool,
    },
    /// Softmax numerical instability
    SoftmaxInstability {
        /// Maximum score before softmax
        max_score: f32,
        /// Range of scores (max - min)
        score_range: f32,
    },
    /// Accumulation order differences (non-associative float ops)
    AccumulationOrder {
        /// Operation where accumulation differs (e.g., "matmul", "sum")
        operation: String,
        /// Whether diff correlates with tensor size
        size_correlated: bool,
    },
    /// Quantization truncation errors
    QuantizationError {
        /// Estimated bits of precision lost
        bits_lost: u8,
        /// Whether at power-of-2 boundaries
        at_power_boundary: bool,
    },
    /// Weight mismatch (not numerical - actual different values)
    WeightMismatch {
        /// Layer with mismatched weights
        layer: String,
        /// Maximum weight difference
        max_diff: f32,
    },
    /// GELU approximation method differs (tanh vs erf)
    GeluApproximation {
        /// Max difference in GELU region
        max_diff_in_region: f32,
    },
    /// LayerNorm epsilon or computation order differs
    LayerNormVariance {
        /// Whether epsilon values likely differ
        epsilon_differs: bool,
    },
    /// Unknown pattern (could not identify root cause)
    Unknown,
}

impl std::fmt::Display for DivergencePattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DivergencePattern::ExpPrecision {
                max_logit,
                is_overflow,
            } => {
                let boundary = if *is_overflow {
                    "overflow"
                } else {
                    "underflow"
                };
                write!(
                    f,
                    "exp() {} boundary: max_logit = {:.1} (boundary ~±88)",
                    boundary, max_logit
                )
            }
            DivergencePattern::SoftmaxInstability {
                max_score,
                score_range,
            } => {
                write!(
                    f,
                    "softmax instability: max_score = {:.1}, range = {:.1}",
                    max_score, score_range
                )
            }
            DivergencePattern::AccumulationOrder {
                operation,
                size_correlated,
            } => {
                let correlation = if *size_correlated {
                    " (diff grows with size)"
                } else {
                    ""
                };
                write!(f, "accumulation order in {}{}", operation, correlation)
            }
            DivergencePattern::QuantizationError {
                bits_lost,
                at_power_boundary,
            } => {
                let boundary = if *at_power_boundary {
                    " at power-of-2 boundary"
                } else {
                    ""
                };
                write!(f, "~{} bits precision lost{}", bits_lost, boundary)
            }
            DivergencePattern::WeightMismatch { layer, max_diff } => {
                write!(
                    f,
                    "weights differ in {}: max_diff = {:.2e}",
                    layer, max_diff
                )
            }
            DivergencePattern::GeluApproximation { max_diff_in_region } => {
                write!(
                    f,
                    "GELU approximation method differs: max_diff = {:.2e}",
                    max_diff_in_region
                )
            }
            DivergencePattern::LayerNormVariance { epsilon_differs } => {
                if *epsilon_differs {
                    write!(f, "LayerNorm epsilon values differ")
                } else {
                    write!(f, "LayerNorm variance computation order differs")
                }
            }
            DivergencePattern::Unknown => write!(f, "unknown pattern"),
        }
    }
}

/// Root cause diagnosis for model divergence.
#[derive(Debug, Clone)]
pub struct DiffDiagnosis {
    /// Layer where divergence first exceeds tolerance.
    pub divergence_layer: String,
    /// Layer type where divergence occurs.
    pub layer_type: Option<LayerType>,
    /// Detected pattern explaining the divergence.
    pub pattern: DivergencePattern,
    /// Human-readable explanation.
    pub explanation: String,
    /// Suggested fix if known.
    pub suggestion: Option<String>,
    /// Confidence level (0.0 - 1.0) in the diagnosis.
    pub confidence: f32,
    /// Supporting evidence for the diagnosis.
    pub evidence: Vec<String>,
}

impl DiffDiagnosis {
    /// Create a diagnosis for an unknown pattern.
    pub fn unknown(layer: &str, layer_type: Option<LayerType>) -> Self {
        Self {
            divergence_layer: layer.to_string(),
            layer_type,
            pattern: DivergencePattern::Unknown,
            explanation: "Could not identify a specific divergence pattern".to_string(),
            suggestion: None,
            confidence: 0.0,
            evidence: Vec::new(),
        }
    }

    /// Format the diagnosis for display.
    pub fn format_report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!("  Layer: {}\n", self.divergence_layer));
        if let Some(lt) = &self.layer_type {
            report.push_str(&format!("  Layer Type: {:?}\n", lt));
        }
        report.push_str(&format!("  Issue: {}\n", self.pattern));
        report.push_str(&format!("  Confidence: {:.0}%\n", self.confidence * 100.0));
        if !self.explanation.is_empty() {
            report.push_str(&format!("  Explanation: {}\n", self.explanation));
        }
        if !self.evidence.is_empty() {
            report.push_str("  Evidence:\n");
            for ev in &self.evidence {
                report.push_str(&format!("    - {}\n", ev));
            }
        }
        if let Some(ref sug) = self.suggestion {
            report.push_str(&format!("\n  Suggestion: {}\n", sug));
        }
        report
    }
}

/// Result of a full model diff operation.
#[derive(Debug, Clone)]
pub struct DiffResult {
    /// Per-layer comparison results.
    pub layers: Vec<LayerComparison>,
    /// Index of first layer that exceeded tolerance (if any).
    pub first_bad_layer: Option<usize>,
    /// Index of first layer where drift started (within 10x tolerance).
    pub drift_start_layer: Option<usize>,
    /// Overall maximum divergence across all layers.
    pub max_divergence: f32,
    /// Tolerance used for comparison.
    pub tolerance: f32,
    /// Suggested root cause (if identified) - legacy field for backwards compat.
    pub suggestion: Option<String>,
    /// Detailed root cause diagnosis (when --diagnose is enabled).
    pub diagnosis: Option<DiffDiagnosis>,
}

impl DiffResult {
    /// Get the status for each layer.
    pub fn statuses(&self) -> Vec<DiffStatus> {
        self.layers
            .iter()
            .enumerate()
            .map(|(i, l)| {
                if l.shape_a != l.shape_b {
                    DiffStatus::ShapeMismatch
                } else if l.exceeds_tolerance {
                    DiffStatus::ExceedsTolerance
                } else if self.drift_start_layer == Some(i) {
                    DiffStatus::DriftStarts
                } else {
                    DiffStatus::Ok
                }
            })
            .collect()
    }

    /// Check if models are equivalent within tolerance.
    pub fn is_equivalent(&self) -> bool {
        self.first_bad_layer.is_none()
    }

    /// Get the first bad layer name if any.
    pub fn first_bad_layer_name(&self) -> Option<&str> {
        self.first_bad_layer
            .and_then(|i| self.layers.get(i))
            .map(|l| l.name.as_str())
    }
}

/// Configuration for model diffing.
#[derive(Debug, Clone)]
pub struct DiffConfig {
    /// Maximum absolute difference allowed between outputs.
    pub tolerance: f32,
    /// Whether to continue comparing after first divergence.
    pub continue_after_divergence: bool,
    /// Input value for testing (None = zeros).
    pub input: Option<ArrayD<f32>>,
    /// Explicit layer name mappings (model_a_name -> model_b_name).
    pub layer_mapping: HashMap<String, String>,
    /// Enable root cause diagnosis analysis.
    pub diagnose: bool,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-5,
            continue_after_divergence: true,
            input: None,
            layer_mapping: HashMap::new(),
            diagnose: false,
        }
    }
}

/// Information about a model loaded for diffing.
#[derive(Debug)]
pub struct ModelInfo {
    /// Input specifications.
    pub inputs: Vec<TensorSpec>,
    /// Output specifications.
    pub outputs: Vec<TensorSpec>,
    /// All intermediate tensor names (node outputs).
    pub intermediate_names: Vec<String>,
    /// Layer specifications from ONNX parsing.
    pub layers: Vec<LayerSpec>,
}
