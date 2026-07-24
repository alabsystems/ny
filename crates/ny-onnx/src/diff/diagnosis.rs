// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::types::{DiffDiagnosis, DivergencePattern, LayerComparison};
use crate::LayerSpec;
use ndarray::ArrayD;
use ny_core::LayerType;
use std::collections::HashMap;

/// Suggest a root cause based on the layer type where divergence starts.
pub(crate) fn suggest_root_cause(layer: &LayerSpec) -> Option<String> {
    use ny_core::LayerType::{
        Add, CausalSoftmax, Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d, LayerNorm, Linear,
        MatMul, Mul, Softmax, GELU,
    };

    match layer.layer_type {
        Softmax | CausalSoftmax => Some(
            "Check softmax numerical precision (exp overflow handling, log-sum-exp trick)"
                .to_string(),
        ),
        LayerNorm => {
            Some("Check LayerNorm epsilon value and variance computation order".to_string())
        }
        GELU => Some("Check GELU approximation method (tanh vs erf)".to_string()),
        Conv1d | Conv2d | ConvTranspose1d | ConvTranspose2d => {
            Some("Check convolution padding mode and group handling".to_string())
        }
        Linear | MatMul => {
            Some("Check matrix multiplication transpose flags and bias handling".to_string())
        }
        Add | Mul => Some("Check broadcast semantics and tensor shapes".to_string()),
        _ => None,
    }
}

/// Context for diagnosing divergence patterns.
pub(crate) struct DiagnosisContext<'a> {
    /// Outputs from model A keyed by tensor name.
    pub(crate) outputs_a: &'a HashMap<String, ArrayD<f32>>,
    /// Outputs from model B keyed by tensor name.
    pub(crate) outputs_b: &'a HashMap<String, ArrayD<f32>>,
    /// Layer specs from model A.
    pub(crate) layers_a: &'a [LayerSpec],
    /// Layer comparisons computed so far.
    pub(crate) comparisons: &'a [LayerComparison],
    /// Tolerance used for comparison.
    pub(crate) tolerance: f32,
}

/// Detect the divergence pattern and generate a diagnosis.
pub(crate) fn diagnose_divergence(
    ctx: &DiagnosisContext,
    bad_layer_idx: usize,
    layer_spec: Option<&LayerSpec>,
) -> DiffDiagnosis {
    let comparison = &ctx.comparisons[bad_layer_idx];
    let layer_name = &comparison.name;
    let layer_type = layer_spec.map(|l| l.layer_type.clone());

    // Get the actual tensor data for analysis
    let out_a = ctx.outputs_a.get(layer_name);
    let out_b = ctx.outputs_b.get(layer_name);

    // Try to detect specific patterns based on layer type and tensor values
    if let (Some(arr_a), Some(arr_b)) = (out_a, out_b) {
        // Check for exp/softmax overflow pattern
        if let Some(ref lt) = layer_type {
            if matches!(lt, LayerType::Softmax | LayerType::CausalSoftmax) {
                if let Some(diag) = check_softmax_pattern(ctx, layer_name, arr_a, arr_b, lt.clone())
                {
                    return diag;
                }
            }

            // Check for GELU approximation differences
            if *lt == LayerType::GELU {
                if let Some(diag) =
                    check_gelu_pattern(layer_name, arr_a, arr_b, comparison.max_diff)
                {
                    return diag;
                }
            }

            // Check for LayerNorm variance issues
            if *lt == LayerType::LayerNorm {
                if let Some(diag) =
                    check_layernorm_pattern(layer_name, arr_a, arr_b, comparison.max_diff)
                {
                    return diag;
                }
            }

            // Check for accumulation order issues in matmul/linear
            if matches!(lt, LayerType::Linear | LayerType::MatMul) {
                if let Some(diag) =
                    check_accumulation_pattern(layer_name, arr_a, arr_b, comparison.max_diff)
                {
                    return diag;
                }
            }
        }

        // Check for quantization errors (independent of layer type)
        if let Some(diag) = check_quantization_pattern(
            layer_name,
            layer_type.as_ref(),
            arr_a,
            arr_b,
            comparison.max_diff,
        ) {
            return diag;
        }

        // Check for growing error pattern (accumulation order)
        if bad_layer_idx > 2 {
            if let Some(diag) =
                check_growing_error_pattern(ctx, bad_layer_idx, layer_name, layer_type.as_ref())
            {
                return diag;
            }
        }
    }

    // Fallback: return diagnosis based on layer type only
    let suggestion = layer_spec.and_then(suggest_root_cause);
    DiffDiagnosis {
        divergence_layer: layer_name.clone(),
        layer_type,
        pattern: DivergencePattern::Unknown,
        explanation: format!(
            "Divergence exceeds tolerance ({:.2e}) at layer {}",
            ctx.tolerance, layer_name
        ),
        suggestion,
        confidence: 0.2,
        evidence: vec![format!("max_diff = {:.2e}", comparison.max_diff)],
    }
}

/// Check for softmax numerical instability patterns.
fn check_softmax_pattern(
    ctx: &DiagnosisContext,
    layer_name: &str,
    _arr_a: &ArrayD<f32>,
    _arr_b: &ArrayD<f32>,
    layer_type: LayerType,
) -> Option<DiffDiagnosis> {
    // Find the input to this softmax layer by looking at the previous layer
    // Softmax input is typically from a matmul (attention scores)
    let mut prev_tensor_name: Option<String> = None;

    for layer in ctx.layers_a {
        for output in &layer.outputs {
            if output == layer_name {
                // Found the softmax layer, get its input
                if !layer.inputs.is_empty() {
                    prev_tensor_name = Some(layer.inputs[0].clone());
                }
                break;
            }
        }
    }

    // Get the pre-softmax values to analyze
    if let Some(ref input_name) = prev_tensor_name {
        if let Some(pre_softmax) = ctx.outputs_a.get(input_name) {
            let max_val = pre_softmax
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let min_val = pre_softmax.iter().cloned().fold(f32::INFINITY, f32::min);
            let range = max_val - min_val;

            // exp(88) is approximately f32::MAX, so logits > 80 are risky
            if max_val > 80.0 {
                return Some(DiffDiagnosis {
                    divergence_layer: layer_name.to_string(),
                    layer_type: Some(layer_type),
                    pattern: DivergencePattern::ExpPrecision {
                        max_logit: max_val,
                        is_overflow: true,
                    },
                    explanation: format!(
                        "Pre-softmax logits have max value {:.1}, near exp() overflow boundary (~88)",
                        max_val
                    ),
                    suggestion: Some(
                        "Apply log-sum-exp stabilization: subtract max(x) before exp()"
                            .to_string(),
                    ),
                    confidence: 0.9,
                    evidence: vec![
                        format!("max_logit = {:.1}", max_val),
                        format!("logit_range = {:.1}", range),
                        "exp(88) ~ f32::MAX".to_string(),
                    ],
                });
            } else if max_val < -80.0 {
                return Some(DiffDiagnosis {
                    divergence_layer: layer_name.to_string(),
                    layer_type: Some(layer_type),
                    pattern: DivergencePattern::ExpPrecision {
                        max_logit: max_val,
                        is_overflow: false,
                    },
                    explanation: format!(
                        "Pre-softmax logits have min value {:.1}, near exp() underflow boundary",
                        min_val
                    ),
                    suggestion: Some(
                        "Check if input normalization is missing or incorrect".to_string(),
                    ),
                    confidence: 0.85,
                    evidence: vec![
                        format!("min_logit = {:.1}", min_val),
                        format!("logit_range = {:.1}", range),
                    ],
                });
            } else if range > 50.0 {
                // Large range can cause precision loss in softmax
                return Some(DiffDiagnosis {
                    divergence_layer: layer_name.to_string(),
                    layer_type: Some(layer_type),
                    pattern: DivergencePattern::SoftmaxInstability {
                        max_score: max_val,
                        score_range: range,
                    },
                    explanation: format!(
                        "Large logit range ({:.1}) causes numerical instability in softmax",
                        range
                    ),
                    suggestion: Some(
                        "Use numerically stable softmax with max subtraction".to_string(),
                    ),
                    confidence: 0.75,
                    evidence: vec![
                        format!("max_score = {:.1}", max_val),
                        format!("min_score = {:.1}", min_val),
                        format!("range = {:.1}", range),
                    ],
                });
            }
        }
    }

    None
}

/// Check for GELU approximation differences.
pub(crate) fn check_gelu_pattern(
    layer_name: &str,
    arr_a: &ArrayD<f32>,
    arr_b: &ArrayD<f32>,
    max_diff: f32,
) -> Option<DiffDiagnosis> {
    // GELU has two common approximations:
    // 1. tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // 2. erf approximation: 0.5 * x * (1 + erf(x / sqrt(2)))
    //
    // These differ most around x = ±1 where the curve transitions

    // Check if differences are concentrated in the transition region
    let mut transition_diffs = 0;
    let mut total_significant = 0;

    for (a, b) in arr_a.iter().zip(arr_b.iter()) {
        let diff = (a - b).abs();
        if diff > max_diff * 0.1 {
            total_significant += 1;
            // Transition region is roughly |x| in [0.5, 2.0] where GELU derivative is significant
            // But we're looking at output values, which are similar magnitude for typical inputs
            if a.abs() > 0.1 && a.abs() < 2.0 {
                transition_diffs += 1;
            }
        }
    }

    if total_significant > 0 && transition_diffs as f64 / total_significant as f64 > 0.5 {
        return Some(DiffDiagnosis {
            divergence_layer: layer_name.to_string(),
            layer_type: Some(LayerType::GELU),
            pattern: DivergencePattern::GeluApproximation {
                max_diff_in_region: max_diff,
            },
            explanation: "GELU implementations use different approximation methods".to_string(),
            suggestion: Some(
                "Ensure both models use the same GELU variant (tanh vs erf approximation)"
                    .to_string(),
            ),
            confidence: 0.7,
            evidence: vec![
                format!("max_diff = {:.2e}", max_diff),
                format!(
                    "{:.0}% of significant diffs in transition region",
                    100.0 * transition_diffs as f64 / total_significant as f64
                ),
            ],
        });
    }

    None
}

/// Check for LayerNorm variance computation issues.
pub(crate) fn check_layernorm_pattern(
    layer_name: &str,
    arr_a: &ArrayD<f32>,
    arr_b: &ArrayD<f32>,
    max_diff: f32,
) -> Option<DiffDiagnosis> {
    // LayerNorm issues often show as:
    // 1. Consistent offset (epsilon difference)
    // 2. Scaling errors (variance computation order)

    // Check for consistent offset vs random errors
    let diffs: Vec<f32> = arr_a.iter().zip(arr_b.iter()).map(|(a, b)| a - b).collect();

    if diffs.is_empty() {
        return None;
    }

    let mean_diff = diffs.iter().sum::<f32>() / diffs.len() as f32;
    let variance: f32 =
        diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f32>() / diffs.len() as f32;
    let std_diff = variance.sqrt();

    // If std is very low relative to mean, it's likely a systematic error (epsilon)
    if std_diff < mean_diff.abs() * 0.1 && mean_diff.abs() > max_diff * 0.5 {
        return Some(DiffDiagnosis {
            divergence_layer: layer_name.to_string(),
            layer_type: Some(LayerType::LayerNorm),
            pattern: DivergencePattern::LayerNormVariance {
                epsilon_differs: true,
            },
            explanation: "Systematic offset suggests different epsilon values".to_string(),
            suggestion: Some(
                "Check LayerNorm epsilon parameter (common: 1e-5 vs 1e-6)".to_string(),
            ),
            confidence: 0.8,
            evidence: vec![
                format!("mean_diff = {:.2e}", mean_diff),
                format!("std_diff = {:.2e}", std_diff),
                "Low variance suggests systematic error".to_string(),
            ],
        });
    }

    // Otherwise could be variance computation order
    if std_diff > mean_diff.abs() * 2.0 {
        return Some(DiffDiagnosis {
            divergence_layer: layer_name.to_string(),
            layer_type: Some(LayerType::LayerNorm),
            pattern: DivergencePattern::LayerNormVariance {
                epsilon_differs: false,
            },
            explanation: "High variance suggests different computation order".to_string(),
            suggestion: Some(
                "Check variance computation: single-pass vs two-pass algorithm".to_string(),
            ),
            confidence: 0.6,
            evidence: vec![
                format!("std_diff = {:.2e}", std_diff),
                format!("mean_diff = {:.2e}", mean_diff),
            ],
        });
    }

    None
}

/// Check for accumulation order differences in matrix operations.
pub(crate) fn check_accumulation_pattern(
    layer_name: &str,
    arr_a: &ArrayD<f32>,
    arr_b: &ArrayD<f32>,
    max_diff: f32,
) -> Option<DiffDiagnosis> {
    // Accumulation order differences typically:
    // 1. Scale with tensor size (more ops = more drift)
    // 2. Are relatively uniform across the output

    let size = arr_a.len();
    if size < 100 {
        return None; // Too small to detect pattern
    }

    // Calculate diff statistics across the tensor
    let diffs: Vec<f32> = arr_a
        .iter()
        .zip(arr_b.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();

    let mean_diff = diffs.iter().sum::<f32>() / diffs.len() as f32;
    let variance: f32 =
        diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f32>() / diffs.len() as f32;
    let cv = variance.sqrt() / mean_diff.max(1e-10); // Coefficient of variation

    // Low CV suggests uniform error distribution (accumulation order)
    if cv < 0.5 && max_diff > 1e-6 {
        return Some(DiffDiagnosis {
            divergence_layer: layer_name.to_string(),
            layer_type: Some(LayerType::MatMul),
            pattern: DivergencePattern::AccumulationOrder {
                operation: "matmul".to_string(),
                size_correlated: size > 10000,
            },
            explanation: "Uniform error distribution suggests accumulation order difference"
                .to_string(),
            suggestion: Some(
                "Use Kahan summation or ensure consistent reduction order".to_string(),
            ),
            confidence: 0.65,
            evidence: vec![
                format!("coefficient_of_variation = {:.2}", cv),
                format!("tensor_size = {}", size),
                format!("mean_diff = {:.2e}", mean_diff),
            ],
        });
    }

    None
}

/// Check for quantization error patterns.
pub(crate) fn check_quantization_pattern(
    layer_name: &str,
    layer_type: Option<&LayerType>,
    arr_a: &ArrayD<f32>,
    arr_b: &ArrayD<f32>,
    _max_diff: f32,
) -> Option<DiffDiagnosis> {
    // Quantization errors show as:
    // 1. Step-like differences (rounding)
    // 2. Errors at power-of-2 boundaries

    // Check if differences are quantized (multiples of a base unit)
    let diffs: Vec<f32> = arr_a
        .iter()
        .zip(arr_b.iter())
        .map(|(a, b)| (a - b).abs())
        .filter(|d| *d > 1e-10)
        .collect();

    if diffs.len() < 10 {
        return None;
    }

    // Find the smallest non-zero difference as potential quantization step
    let min_diff = diffs.iter().cloned().fold(f32::INFINITY, f32::min);

    // Check how many diffs are close to multiples of min_diff
    let mut quantized_count = 0;
    for d in &diffs {
        let ratio = d / min_diff;
        let rounded = ratio.round();
        if (ratio - rounded).abs() < 0.1 {
            quantized_count += 1;
        }
    }

    let quantized_ratio = quantized_count as f64 / diffs.len() as f64;

    if quantized_ratio > 0.7 {
        // Estimate bits lost based on quantization step
        let bits_lost = if min_diff > 1e-3 {
            10
        } else if min_diff > 1e-5 {
            7
        } else if min_diff > 1e-7 {
            4
        } else {
            2
        };

        // Check for power-of-2 boundary effects
        let at_boundary = arr_a.iter().any(|v| {
            let exp = v.abs().log2().floor();
            (v.abs() - 2.0f32.powf(exp)).abs() < min_diff * 10.0
        });

        return Some(DiffDiagnosis {
            divergence_layer: layer_name.to_string(),
            layer_type: layer_type.cloned(),
            pattern: DivergencePattern::QuantizationError {
                bits_lost,
                at_power_boundary: at_boundary,
            },
            explanation: format!("Differences appear quantized with step ~{:.2e}", min_diff),
            suggestion: Some(
                "Check for fp16/fp32 mixed precision or INT8 quantization".to_string(),
            ),
            confidence: 0.75,
            evidence: vec![
                format!("quantization_step = {:.2e}", min_diff),
                format!("{:.0}% of diffs are quantized", quantized_ratio * 100.0),
                format!("estimated_bits_lost = {}", bits_lost),
            ],
        });
    }

    None
}

/// Check for growing error pattern (error accumulation across layers).
fn check_growing_error_pattern(
    ctx: &DiagnosisContext,
    bad_layer_idx: usize,
    layer_name: &str,
    layer_type: Option<&LayerType>,
) -> Option<DiffDiagnosis> {
    // Check if errors are growing across layers
    if bad_layer_idx < 3 {
        return None;
    }

    let recent_diffs: Vec<f32> = ctx.comparisons[bad_layer_idx.saturating_sub(5)..=bad_layer_idx]
        .iter()
        .map(|c| c.max_diff)
        .collect();

    if recent_diffs.len() < 3 {
        return None;
    }

    // Check if diffs are monotonically increasing
    let mut increasing = true;
    for i in 1..recent_diffs.len() {
        if recent_diffs[i] < recent_diffs[i - 1] * 0.9 {
            increasing = false;
            break;
        }
    }

    if increasing {
        // Calculate growth rate
        let first = recent_diffs.first().unwrap_or(&0.0);
        let last = recent_diffs.last().unwrap_or(&0.0);
        let growth = if *first > 1e-10 { last / first } else { 1.0 };

        return Some(DiffDiagnosis {
            divergence_layer: layer_name.to_string(),
            layer_type: layer_type.cloned(),
            pattern: DivergencePattern::AccumulationOrder {
                operation: "network".to_string(),
                size_correlated: true,
            },
            explanation: format!(
                "Errors grow {:.1}x across {} layers, suggesting accumulation order differences",
                growth,
                recent_diffs.len()
            ),
            suggestion: Some(
                "Check for non-associative operations: different reduction orders, fused ops"
                    .to_string(),
            ),
            confidence: 0.7,
            evidence: vec![
                format!("growth_factor = {:.1}x", growth),
                format!("layers_analyzed = {}", recent_diffs.len()),
                format!("first_diff = {:.2e}", first),
                format!("last_diff = {:.2e}", last),
            ],
        });
    }

    None
}
