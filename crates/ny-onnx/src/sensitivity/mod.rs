// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sensitivity analysis for neural networks.
//!
//! This module provides tools to analyze how each layer in a neural network
//! amplifies input uncertainty. This is useful for:
//! - Finding unstable layers that explode bounds
//! - Identifying where verification becomes difficult
//! - Pre-quantization analysis (high sensitivity = quantization risk)
//!
//! ## Key Metric: Sensitivity (Amplification Factor)
//!
//! For each layer, we compute:
//!   sensitivity = output_bound_width / input_bound_width
//!
//! - sensitivity < 1.0: Layer contracts bounds (stable)
//! - sensitivity = 1.0: Layer preserves bounds (neutral)
//! - sensitivity > 1.0: Layer amplifies bounds (potentially unstable)
//!
//! High sensitivity layers are "choke points" for verification and
//! may be problematic for quantization.

use crate::{load_onnx, OnnxModel};
use ndarray::{ArrayD, IxDyn};
use ny_core::truncate_name;
use ny_propagate::layers::Layer;
use ny_propagate::BoundPropagation;
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;
use std::cmp::Ordering;
use std::path::Path;
use tracing::{debug, info};

/// Errors that can occur during sensitivity analysis.
///
/// Type alias for the shared [`AnalysisError`](crate::analysis_error::AnalysisError).
pub type SensitivityError = crate::analysis_error::AnalysisError;

/// Result of analyzing a single layer's sensitivity.
#[derive(Debug, Clone)]
pub struct LayerSensitivity {
    /// Layer name from the model.
    pub name: String,
    /// Layer type (e.g., "Linear", "ReLU", "Softmax").
    pub layer_type: String,
    /// Input bound width (max width across all elements).
    pub input_width: f32,
    /// Output bound width (max width across all elements).
    pub output_width: f32,
    /// Sensitivity = output_width / input_width.
    /// >1 means the layer amplifies uncertainty.
    pub sensitivity: f32,
    /// Mean bound width at output.
    pub mean_output_width: f32,
    /// Output shape.
    pub output_shape: Vec<usize>,
    /// Whether any output bounds are infinite or NaN (actual numerical overflow).
    pub has_overflow: bool,
    /// Whether propagation failed (e.g., shape mismatch).
    /// If true, sensitivity is unreliable (fallback input used as output).
    pub propagation_failed: bool,
}

impl LayerSensitivity {
    /// Check if this layer is a high-sensitivity layer (amplifies significantly).
    pub fn is_high_sensitivity(&self, threshold: f32) -> bool {
        self.sensitivity > threshold
    }

    /// Check if this layer contracts bounds.
    pub fn is_contractive(&self) -> bool {
        self.sensitivity < 1.0
    }
}

/// Result of a full sensitivity analysis.
#[derive(Debug, Clone)]
pub struct SensitivityResult {
    /// Per-layer sensitivity analysis.
    pub layers: Vec<LayerSensitivity>,
    /// Total sensitivity (product of all layer sensitivities).
    /// This is the theoretical worst-case bound amplification.
    pub total_sensitivity: f32,
    /// Maximum single-layer sensitivity.
    pub max_sensitivity: f32,
    /// Index of the layer with maximum sensitivity.
    pub max_sensitivity_layer: Option<usize>,
    /// Initial input bound width.
    pub input_epsilon: f32,
    /// Final output bound width.
    pub final_width: f32,
    /// Index of first layer where overflow occurred (if any).
    pub overflow_at_layer: Option<usize>,
}

impl SensitivityResult {
    /// Get a summary of the analysis.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Sensitivity Analysis".to_string());
        lines.push("====================".to_string());
        lines.push(format!(
            "{:<40} | {:>10} | {:>10} | {:>10} | Status",
            "Layer", "In Width", "Out Width", "Sens."
        ));
        lines.push(format!(
            "{:-<40}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+--------",
            "", "", "", ""
        ));

        for (i, layer) in self.layers.iter().enumerate() {
            // Status priority: propagation_failed > has_overflow > sensitivity thresholds
            let status = if layer.propagation_failed {
                "SKIPPED" // Propagation failed (e.g., shape mismatch) - sensitivity is unreliable
            } else if layer.has_overflow {
                "OVERFLOW" // Actual numerical overflow (Inf/NaN)
            } else if layer.sensitivity > 10.0 {
                "HIGH"
            } else if layer.sensitivity > 2.0 {
                "MODERATE"
            } else if layer.sensitivity < 1.0 {
                "STABLE"
            } else {
                "OK"
            };

            let is_max = self.max_sensitivity_layer == Some(i);
            let marker = if is_max { " <<<" } else { "" };

            lines.push(format!(
                "{:<40} | {:>10.3e} | {:>10.3e} | {:>10.2} | {}{}",
                truncate_name(&layer.name, 40),
                layer.input_width,
                layer.output_width,
                layer.sensitivity,
                status,
                marker
            ));
        }

        lines.push(String::new());
        lines.push(format!(
            "Total sensitivity: {:.2e} (product of all layers)",
            self.total_sensitivity
        ));
        lines.push(format!(
            "Max single-layer sensitivity: {:.2} at layer {}",
            self.max_sensitivity,
            self.max_sensitivity_layer
                .and_then(|i| self.layers.get(i))
                .map(|l| l.name.as_str())
                .unwrap_or("N/A")
        ));
        lines.push(format!(
            "Input epsilon: {:.2e} → Final width: {:.2e}",
            self.input_epsilon, self.final_width
        ));

        // Count skipped (propagation failed) and overflow layers
        let skipped_count = self.layers.iter().filter(|l| l.propagation_failed).count();
        let overflow_count = self
            .layers
            .iter()
            .filter(|l| l.has_overflow && !l.propagation_failed)
            .count();

        if skipped_count > 0 {
            lines.push(format!(
                "NOTE: {} layer(s) skipped due to propagation failure (shape mismatch)",
                skipped_count
            ));
        }

        if overflow_count > 0 {
            lines.push(format!(
                "WARNING: {} layer(s) have numerical overflow (Inf/NaN values)",
                overflow_count
            ));
        }

        if let Some(overflow_idx) = self.overflow_at_layer {
            let layer = self.layers.get(overflow_idx);
            if layer
                .map(|l| l.has_overflow && !l.propagation_failed)
                .unwrap_or(false)
            {
                lines.push(format!(
                    "First overflow at layer {} ({})",
                    overflow_idx,
                    layer.map(|l| l.name.as_str()).unwrap_or("unknown")
                ));
            }
        }

        lines.join("\n")
    }

    /// Get layers sorted by sensitivity (highest first, NaN last).
    ///
    /// Uses NaN-safe descending ordering so NaN-contaminated entries
    /// sort deterministically to the end rather than corrupting rankings (#2601).
    pub fn layers_by_sensitivity(&self) -> Vec<&LayerSensitivity> {
        let mut sorted: Vec<_> = self.layers.iter().collect();
        sorted.sort_by(|a, b| {
            // NaN-last descending: finite descending, then NaN
            match (a.sensitivity.is_nan(), b.sensitivity.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => b.sensitivity.total_cmp(&a.sensitivity),
            }
        });
        sorted
    }

    /// Get "hot spots" - layers with sensitivity above a threshold.
    pub fn hot_spots(&self, threshold: f32) -> Vec<&LayerSensitivity> {
        self.layers
            .iter()
            .filter(|l| l.sensitivity > threshold)
            .collect()
    }
}

/// Configuration for sensitivity analysis.
#[derive(Debug, Clone)]
pub struct SensitivityConfig {
    /// Input perturbation epsilon.
    pub epsilon: f32,
    ///
    /// Defaults to `false` because downstream sensitivity magnitudes are no
    /// longer trustworthy once propagation has already overflowed.
    pub continue_after_overflow: bool,
    /// Custom input tensor (None = zeros with epsilon bounds).
    pub input: Option<BoundedTensor>,
}

impl Default for SensitivityConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.01,
            continue_after_overflow: false,
            input: None,
        }
    }
}

/// Analyze sensitivity of a model loaded from ONNX file.
pub fn analyze_sensitivity(
    path: impl AsRef<Path>,
    config: &SensitivityConfig,
) -> Result<SensitivityResult, SensitivityError> {
    info!("Loading model: {}", path.as_ref().display());
    let onnx_model =
        load_onnx(path.as_ref()).map_err(|e| SensitivityError::load("sensitivity", e))?;

    analyze_sensitivity_model(&onnx_model, config)
}

/// Analyze sensitivity of an already-loaded ONNX model.
pub fn analyze_sensitivity_model(
    model: &OnnxModel,
    config: &SensitivityConfig,
) -> Result<SensitivityResult, SensitivityError> {
    // Create input tensor
    let input = if let Some(ref inp) = config.input {
        inp.clone()
    } else {
        // Get input shape from model
        let input_spec = model.network.inputs.first().ok_or_else(|| {
            SensitivityError::invalid_input_shape("sensitivity", "No input specification")
        })?;

        let shape: Vec<usize> = input_spec
            .shape
            .iter()
            .map(|&d| if d > 0 { d as usize } else { 1 })
            .collect();

        let data = ArrayD::zeros(IxDyn(&shape));
        BoundedTensor::from_epsilon(data, config.epsilon)
            .map_err(|e| SensitivityError::propagation("sensitivity", e))?
    };

    info!(
        "Starting sensitivity analysis with input shape {:?}, epsilon {}",
        input.shape(),
        config.epsilon
    );

    // Prefer a graph-based analysis when the model contains binary nodes
    // (e.g., residual Add, bounded MatMul, MulBinary). Sequential sensitivity
    // uses `Network` and cannot represent DAGs correctly.
    // NOTE: Intentional fallback — graph conversion failure falls through to sequential analysis.
    if let Ok(graph) = model.to_graph_network() {
        if graph_requires_dag_sensitivity(&graph) {
            return analyze_sensitivity_graph(&graph, &input, config);
        }
    }

    // Fall back to sequential sensitivity analysis. If propagation panics (e.g., due to a
    // binary op slipping through), retry with DAG-based analysis instead of crashing.
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        analyze_sensitivity_sequential(model, &input, config)
    })) {
        Ok(res) => res,
        Err(_) => {
            debug!("Sequential sensitivity panicked; retrying with GraphNetwork analysis");
            let graph = model
                .to_graph_network()
                .map_err(|e| SensitivityError::propagation("sensitivity", e))?;
            analyze_sensitivity_graph(&graph, &input, config)
        }
    }
}

fn graph_requires_dag_sensitivity(graph: &GraphNetwork) -> bool {
    graph
        .node_names()
        .iter()
        .filter_map(|n| graph.node(n))
        .any(|node| node.layer().is_binary())
}

fn analyze_sensitivity_sequential(
    model: &OnnxModel,
    input: &BoundedTensor,
    config: &SensitivityConfig,
) -> Result<SensitivityResult, SensitivityError> {
    // Convert to propagate network
    let network = model
        .to_propagate_network()
        .map_err(|e| SensitivityError::propagation("sensitivity", e))?;

    if network.layers().is_empty() {
        return Err(SensitivityError::no_layers("sensitivity"));
    }

    // Track layer-by-layer sensitivity
    let mut layers = Vec::new();
    let mut current = input.clone();
    let mut total_sensitivity: f32 = 1.0;
    let mut max_sensitivity: f32 = 0.0;
    let mut max_sensitivity_layer: Option<usize> = None;
    let mut overflow_at_layer: Option<usize> = None;

    for (i, (layer, spec)) in network
        .layers()
        .iter()
        .zip(model.network.layers.iter())
        .enumerate()
    {
        let input_width = current.max_width();

        // Propagate through this layer
        let mut propagation_failed = false;
        let output = match layer.propagate_ibp(&current) {
            Ok(out) => out,
            Err(e) => {
                debug!("Layer {} propagation failed: {}", spec.name, e);
                if !config.continue_after_overflow {
                    return Err(SensitivityError::propagation("sensitivity", e));
                }
                // Use current as fallback and mark as propagation failure
                propagation_failed = true;
                current.clone()
            }
        };

        let output_width = output.max_width();
        let mean_width = output.width().iter().sum::<f32>() / output.width().len().max(1) as f32;
        // Only check for numerical overflow if propagation succeeded
        let has_overflow = !propagation_failed
            && (!output_width.is_finite() || output.width().iter().any(|w| !w.is_finite()));

        // Calculate sensitivity
        let sensitivity = if input_width > 0.0 && input_width.is_finite() {
            output_width / input_width
        } else if output_width == 0.0 {
            1.0
        } else {
            f32::INFINITY
        };

        // Track max sensitivity (only for successful propagations)
        if !propagation_failed && sensitivity > max_sensitivity && sensitivity.is_finite() {
            max_sensitivity = sensitivity;
            max_sensitivity_layer = Some(i);
        }

        // Accumulate total sensitivity (product) - skip failed propagations
        if !propagation_failed && sensitivity.is_finite() {
            total_sensitivity *= sensitivity;
        } else if has_overflow {
            total_sensitivity = f32::INFINITY;
        }

        // Check for actual overflow (not propagation failure)
        if has_overflow && overflow_at_layer.is_none() {
            overflow_at_layer = Some(i);
        }

        layers.push(LayerSensitivity {
            name: spec.name.clone(),
            layer_type: format!("{:?}", spec.layer_type),
            input_width,
            output_width,
            sensitivity,
            mean_output_width: mean_width,
            output_shape: output.shape().to_vec(),
            has_overflow,
            propagation_failed,
        });

        debug!(
            "Layer {}: {:?} -> sensitivity = {:.3}",
            spec.name, spec.layer_type, sensitivity
        );

        // Stop if overflow and not continuing
        if has_overflow && !config.continue_after_overflow {
            break;
        }

        current = output;
    }

    let final_width = current.max_width();

    Ok(SensitivityResult {
        layers,
        total_sensitivity,
        max_sensitivity,
        max_sensitivity_layer,
        input_epsilon: config.epsilon,
        final_width,
        overflow_at_layer,
    })
}

fn analyze_sensitivity_graph(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    config: &SensitivityConfig,
) -> Result<SensitivityResult, SensitivityError> {
    let exec_order = graph
        .exec_order()
        .map_err(|e| SensitivityError::propagation("sensitivity/graph", e))?;

    if exec_order.is_empty() {
        return Err(SensitivityError::no_layers("sensitivity/graph"));
    }

    let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
        std::collections::HashMap::with_capacity(exec_order.len());

    let mut layers = Vec::with_capacity(exec_order.len());
    let mut total_sensitivity: f32 = 1.0;
    let mut max_sensitivity: f32 = 0.0;
    let mut max_sensitivity_layer: Option<usize> = None;
    let mut overflow_at_layer: Option<usize> = None;

    let input_width0 = input.max_width();

    for (i, node_name) in exec_order.iter().enumerate() {
        let node = graph.node(node_name).ok_or_else(|| {
            SensitivityError::propagation_msg(
                "sensitivity/graph",
                format!("Node not found: {}", node_name),
            )
        })?;

        let input_width = if node.inputs().is_empty() {
            input_width0
        } else {
            node.inputs()
                .iter()
                .map(|inp| {
                    if inp == "_input" {
                        input_width0
                    } else {
                        bounds_cache
                            .get(inp)
                            .map(|b| b.max_width())
                            .unwrap_or(input_width0)
                    }
                })
                .fold(0.0_f32, f32::max)
        };

        let mut propagation_failed = false;

        // Concat MUST be checked before is_binary() because Layer::is_binary()
        // returns true for Concat. Without this ordering, n-ary Concat (3+ inputs)
        // would silently drop inputs beyond the first two. (#2405)
        let output = if let Layer::Concat(concat) = node.layer() {
            let resolve = |name: &str| -> Result<BoundedTensor, SensitivityError> {
                if name == "_input" {
                    Ok(input.clone())
                } else {
                    bounds_cache.get(name).cloned().ok_or_else(|| {
                        SensitivityError::propagation_msg(
                            "sensitivity/graph",
                            format!("Bounds for node {} not computed yet", name),
                        )
                    })
                }
            };
            // Handle constant_inputs interleaving (same pattern as dispatch.rs).
            let owned_inputs: Vec<BoundedTensor> = if let Some(ref ci) = concat.constant_inputs {
                let mut graph_idx = 0;
                ci.iter()
                    .map(|const_opt| {
                        if let Some(constant) = const_opt {
                            Ok(constant.clone())
                        } else {
                            let name = node.inputs().get(graph_idx).ok_or_else(|| {
                                SensitivityError::propagation_msg(
                                    "sensitivity/graph",
                                    format!("Concat: ran out of graph inputs at idx {}", graph_idx),
                                )
                            })?;
                            graph_idx += 1;
                            resolve(name)
                        }
                    })
                    .collect::<Result<Vec<_>, SensitivityError>>()?
            } else {
                node.inputs()
                    .iter()
                    .map(|name| resolve(name))
                    .collect::<Result<Vec<_>, SensitivityError>>()?
            };
            let input_refs: Vec<&BoundedTensor> = owned_inputs.iter().collect();
            match concat.propagate_ibp_nary(&input_refs) {
                Ok(out) => out,
                Err(e) => {
                    debug!("Node {} propagation failed: {}", node_name, e);
                    if !config.continue_after_overflow {
                        return Err(SensitivityError::propagation("sensitivity/graph", e));
                    }
                    propagation_failed = true;
                    owned_inputs
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| input.clone())
                }
            }
        } else if node.layer().is_binary() {
            if node.inputs().len() < 2 {
                return Err(SensitivityError::propagation_msg(
                    "sensitivity/graph",
                    format!(
                        "Binary node {} requires 2 inputs, got {}",
                        node_name,
                        node.inputs().len()
                    ),
                ));
            }

            let input_a = if node.inputs()[0] == "_input" {
                input
            } else {
                bounds_cache.get(&node.inputs()[0]).ok_or_else(|| {
                    SensitivityError::propagation_msg(
                        "sensitivity/graph",
                        format!("Bounds for node {} not computed yet", node.inputs()[0]),
                    )
                })?
            };
            let input_b = if node.inputs()[1] == "_input" {
                input
            } else {
                bounds_cache.get(&node.inputs()[1]).ok_or_else(|| {
                    SensitivityError::propagation_msg(
                        "sensitivity/graph",
                        format!("Bounds for node {} not computed yet", node.inputs()[1]),
                    )
                })?
            };

            match node.layer().propagate_ibp_binary(input_a, input_b) {
                Ok(out) => out,
                Err(e) => {
                    debug!("Node {} propagation failed: {}", node_name, e);
                    if !config.continue_after_overflow {
                        return Err(SensitivityError::propagation("sensitivity/graph", e));
                    }
                    propagation_failed = true;
                    input_a.clone()
                }
            }
        } else {
            let node_input = if node.inputs().is_empty() || node.inputs()[0] == "_input" {
                input
            } else {
                bounds_cache.get(&node.inputs()[0]).ok_or_else(|| {
                    SensitivityError::propagation_msg(
                        "sensitivity/graph",
                        format!("Bounds for node {} not computed yet", node.inputs()[0]),
                    )
                })?
            };

            match node.layer().propagate_ibp(node_input) {
                Ok(out) => out,
                Err(e) => {
                    debug!("Node {} propagation failed: {}", node_name, e);
                    if !config.continue_after_overflow {
                        return Err(SensitivityError::propagation("sensitivity/graph", e));
                    }
                    propagation_failed = true;
                    node_input.clone()
                }
            }
        };

        let output_width = output.max_width();
        let width_vec = output.width();
        let mean_width = width_vec.iter().sum::<f32>() / width_vec.len().max(1) as f32;
        let non_finite_count = width_vec.iter().filter(|w| !w.is_finite()).count();
        // Only check for numerical overflow if propagation succeeded
        let has_overflow =
            !propagation_failed && (!output_width.is_finite() || non_finite_count > 0);

        if non_finite_count > 0 && !propagation_failed {
            debug!(
                "Node {} has {}/{} non-finite width values",
                node_name,
                non_finite_count,
                width_vec.len()
            );
        }

        let sensitivity = if input_width > 0.0 && input_width.is_finite() {
            output_width / input_width
        } else if output_width == 0.0 {
            1.0
        } else {
            f32::INFINITY
        };

        // Track max sensitivity (only for successful propagations)
        if !propagation_failed && sensitivity > max_sensitivity && sensitivity.is_finite() {
            max_sensitivity = sensitivity;
            max_sensitivity_layer = Some(i);
        }

        // Accumulate total sensitivity (product) - skip failed propagations
        if !propagation_failed && sensitivity.is_finite() {
            total_sensitivity *= sensitivity;
        } else if has_overflow {
            // Only set to infinity for actual numerical overflow
            total_sensitivity = f32::INFINITY;
        }
        // Note: propagation failures are skipped (not counted in total)

        // Check for actual overflow (not propagation failure)
        if has_overflow && overflow_at_layer.is_none() {
            overflow_at_layer = Some(i);
        }

        layers.push(LayerSensitivity {
            name: node.name().to_string(),
            layer_type: node.layer().layer_type().to_string(),
            input_width,
            output_width,
            sensitivity,
            mean_output_width: mean_width,
            output_shape: output.shape().to_vec(),
            has_overflow,
            propagation_failed,
        });

        bounds_cache.insert(node.name().to_string(), output);

        // Stop early if overflow detected and not configured to continue.
        if has_overflow && !config.continue_after_overflow {
            break;
        }
    }

    let output_node = graph.output_name();
    let final_width = if !output_node.is_empty() {
        bounds_cache
            .get(output_node)
            .map(|b| b.max_width())
            .unwrap_or_else(|| {
                layers
                    .last()
                    .and_then(|l| bounds_cache.get(&l.name))
                    .map(|b| b.max_width())
                    .unwrap_or(input_width0)
            })
    } else {
        layers
            .last()
            .and_then(|l| bounds_cache.get(&l.name))
            .map(|b| b.max_width())
            .unwrap_or(input_width0)
    };

    Ok(SensitivityResult {
        layers,
        total_sensitivity,
        max_sensitivity,
        max_sensitivity_layer,
        input_epsilon: config.epsilon,
        final_width,
        overflow_at_layer,
    })
}

#[cfg(test)]
mod tests;
