// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::repr::repr_string;
use crate::utils::{export_torch_to_onnx_bytes, truncate_name, validate_epsilon};
use ny_onnx::{load_onnx_bytes, sensitivity};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Result of analyzing a single layer's sensitivity.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct LayerSensitivity {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub layer_type: String,
    #[pyo3(get)]
    pub input_width: f32,
    #[pyo3(get)]
    pub output_width: f32,
    #[pyo3(get)]
    pub sensitivity: f32,
    #[pyo3(get)]
    pub mean_output_width: f32,
    #[pyo3(get)]
    pub output_shape: Vec<usize>,
    #[pyo3(get)]
    pub has_overflow: bool,
    #[pyo3(get)]
    pub propagation_failed: bool,
}

#[pymethods]
impl LayerSensitivity {
    pub(crate) fn __repr__(&self) -> String {
        format!(
            "LayerSensitivity(name={}, sensitivity={:.2})",
            repr_string(&self.name),
            self.sensitivity
        )
    }

    /// Check if this layer amplifies significantly (sensitivity > threshold).
    pub(crate) fn is_high_sensitivity(&self, threshold: f32) -> bool {
        self.sensitivity > threshold
    }

    /// Check if this layer contracts bounds (sensitivity < 1.0).
    pub(crate) fn is_contractive(&self) -> bool {
        self.sensitivity < 1.0
    }
}

/// Result of a full sensitivity analysis.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct SensitivityResult {
    #[pyo3(get)]
    pub layers: Vec<LayerSensitivity>,
    #[pyo3(get)]
    pub total_sensitivity: f32,
    #[pyo3(get)]
    pub max_sensitivity: f32,
    #[pyo3(get)]
    pub max_sensitivity_layer: Option<usize>,
    #[pyo3(get)]
    pub input_epsilon: f32,
    #[pyo3(get)]
    pub final_width: f32,
    #[pyo3(get)]
    pub overflow_at_layer: Option<usize>,
}

#[pymethods]
impl SensitivityResult {
    pub(crate) fn __repr__(&self) -> String {
        format!(
            "SensitivityResult(layers={}, max_sensitivity={:.2}, total_sensitivity={:.2e})",
            self.layers.len(),
            self.max_sensitivity,
            self.total_sensitivity
        )
    }

    /// Get a formatted summary table.
    pub(crate) fn summary(&self) -> String {
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
            let status = if layer.propagation_failed {
                "FAILED"
            } else if layer.has_overflow {
                "OVERFLOW"
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
        lines.push(format!("Total sensitivity: {:.2e}", self.total_sensitivity));
        lines.push(format!(
            "Max single-layer sensitivity: {:.2} at layer {}",
            self.max_sensitivity,
            self.max_sensitivity_layer
                .and_then(|i| self.layers.get(i))
                .map(|l| l.name.as_str())
                .unwrap_or("N/A")
        ));

        lines.join("\n")
    }

    /// Get name of the layer with maximum sensitivity.
    #[getter]
    pub(crate) fn max_sensitivity_layer_name(&self) -> Option<String> {
        self.max_sensitivity_layer
            .and_then(|i| self.layers.get(i))
            .map(|l| l.name.clone())
    }

    /// Get high-sensitivity layers (above threshold).
    pub(crate) fn hot_spots(&self, threshold: f32) -> Vec<LayerSensitivity> {
        self.layers
            .iter()
            .filter(|l| l.sensitivity > threshold)
            .cloned()
            .collect()
    }

    /// Check if overflow occurred.
    #[getter]
    pub(crate) fn has_overflow(&self) -> bool {
        self.overflow_at_layer.is_some()
    }
}

/// Analyze layer-by-layer sensitivity (noise amplification).
///
/// Computes how each layer amplifies input uncertainty. High sensitivity
/// layers are "choke points" where verification becomes difficult.
///
/// Args:
///     model_path: Path to ONNX model
///     epsilon: Input perturbation size (default: 0.01)
///     continue_after_overflow: Keep going after overflow (default: False)
///
/// Returns:
///     SensitivityResult with per-layer analysis
///
/// Example:
///     >>> result = ny.sensitivity_analysis("model.onnx")
///     >>> print(f"Max sensitivity: {result.max_sensitivity:.2f}")
///     >>> for layer in result.hot_spots(10.0):
///     ...     print(f"  {layer.name}: {layer.sensitivity:.2f}x")
#[pyfunction]
#[pyo3(signature = (model_path, epsilon=0.01, continue_after_overflow=false))]
pub fn sensitivity_analysis(
    py: Python<'_>,
    model_path: &str,
    epsilon: f32,
    continue_after_overflow: bool,
) -> PyResult<SensitivityResult> {
    validate_epsilon(epsilon)?;

    let config = sensitivity::SensitivityConfig {
        epsilon,
        continue_after_overflow,
        input: None,
    };

    let result = Python::detach(py, || sensitivity::analyze_sensitivity(model_path, &config))
        .map_err(|e| PyValueError::new_err(format!("Sensitivity error: {}", e)))?;

    convert_sensitivity_result(result)
}

/// Analyze sensitivity from in-memory ONNX bytes.
///
/// Args:
///     model_bytes: ONNX model as bytes
///     epsilon: Input perturbation size (default: 0.01)
///     continue_after_overflow: Keep going after overflow (default: False)
///     name: Model name for diagnostics (default: "model")
///
/// Returns:
///     SensitivityResult with per-layer analysis
///
/// Example:
///     >>> with open("model.onnx", "rb") as f:
///     ...     data = f.read()
///     >>> result = ny.sensitivity_analysis_bytes(data)
#[pyfunction]
#[pyo3(signature = (model_bytes, epsilon=0.01, continue_after_overflow=false, name="model"))]
pub fn sensitivity_analysis_bytes(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    epsilon: f32,
    continue_after_overflow: bool,
    name: &str,
) -> PyResult<SensitivityResult> {
    validate_epsilon(epsilon)?;

    let config = sensitivity::SensitivityConfig {
        epsilon,
        continue_after_overflow,
        input: None,
    };

    let name = name.to_string();
    let result = Python::detach(py, || {
        let model = load_onnx_bytes(&name, &model_bytes)
            .map_err(|e| sensitivity::SensitivityError::load("sensitivity/python", e))?;
        sensitivity::analyze_sensitivity_model(&model, &config)
    })
    .map_err(|e| PyValueError::new_err(format!("Sensitivity error: {}", e)))?;

    convert_sensitivity_result(result)
}

/// Analyze sensitivity of a PyTorch model without writing to disk.
///
/// Args:
///     model: PyTorch model (nn.Module)
///     example_input: Example input tensor for ONNX export
///     epsilon: Input perturbation size (default: 0.01)
///     continue_after_overflow: Keep going after overflow (default: False)
///     opset: ONNX opset version (default: 17)
///
/// Returns:
///     SensitivityResult with per-layer analysis
///
/// Example:
///     >>> import torch
///     >>> model = MyModel()
///     >>> example = torch.randn(1, 3, 224, 224)
///     >>> result = ny.sensitivity_analysis_torch(model, example)
#[pyfunction]
#[pyo3(signature = (model, example_input, epsilon=0.01, continue_after_overflow=false, opset=17))]
pub fn sensitivity_analysis_torch(
    py: Python<'_>,
    model: &Bound<'_, PyAny>,
    example_input: &Bound<'_, PyAny>,
    epsilon: f32,
    continue_after_overflow: bool,
    opset: u32,
) -> PyResult<SensitivityResult> {
    let model_bytes = export_torch_to_onnx_bytes(
        py,
        model,
        example_input,
        opset,
        "sensitivity_analysis_torch",
    )?;
    sensitivity_analysis_bytes(
        py,
        model_bytes,
        epsilon,
        continue_after_overflow,
        "torch_model",
    )
}

fn convert_sensitivity_result(
    result: sensitivity::SensitivityResult,
) -> PyResult<SensitivityResult> {
    let layers: Vec<LayerSensitivity> = result
        .layers
        .into_iter()
        .map(|l| LayerSensitivity {
            name: l.name,
            layer_type: l.layer_type,
            input_width: l.input_width,
            output_width: l.output_width,
            sensitivity: l.sensitivity,
            mean_output_width: l.mean_output_width,
            output_shape: l.output_shape,
            has_overflow: l.has_overflow,
            propagation_failed: l.propagation_failed,
        })
        .collect();

    Ok(SensitivityResult {
        layers,
        total_sensitivity: result.total_sensitivity,
        max_sensitivity: result.max_sensitivity,
        max_sensitivity_layer: result.max_sensitivity_layer,
        input_epsilon: result.input_epsilon,
        final_width: result.final_width,
        overflow_at_layer: result.overflow_at_layer,
    })
}
