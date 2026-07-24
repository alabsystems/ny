// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::repr::repr_string;
use crate::utils::{export_torch_to_onnx_bytes, truncate_name, validate_epsilon};
use ny_onnx::load_onnx_bytes;
use ny_onnx::quantize::{self, QuantSafety as RustQuantSafety};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Quantization safety status.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum QuantSafety {
    Safe,
    Denormal,
    ScalingRequired,
    Overflow,
    Unknown,
}

#[pymethods]
impl QuantSafety {
    pub(crate) fn __repr__(&self) -> String {
        match self {
            QuantSafety::Safe => "QuantSafety.Safe".to_string(),
            QuantSafety::Denormal => "QuantSafety.Denormal".to_string(),
            QuantSafety::ScalingRequired => "QuantSafety.ScalingRequired".to_string(),
            QuantSafety::Overflow => "QuantSafety.Overflow".to_string(),
            QuantSafety::Unknown => "QuantSafety.Unknown".to_string(),
        }
    }

    pub(crate) fn __str__(&self) -> String {
        match self {
            QuantSafety::Safe => "SAFE".to_string(),
            QuantSafety::Denormal => "DENORMAL".to_string(),
            QuantSafety::ScalingRequired => "SCALE".to_string(),
            QuantSafety::Overflow => "OVERFLOW".to_string(),
            QuantSafety::Unknown => "UNKNOWN".to_string(),
        }
    }
}

impl From<RustQuantSafety> for QuantSafety {
    fn from(s: RustQuantSafety) -> Self {
        match s {
            RustQuantSafety::Safe => QuantSafety::Safe,
            RustQuantSafety::Denormal => QuantSafety::Denormal,
            RustQuantSafety::ScalingRequired => QuantSafety::ScalingRequired,
            RustQuantSafety::Overflow => QuantSafety::Overflow,
            RustQuantSafety::Unknown => QuantSafety::Unknown,
        }
    }
}

/// Result of analyzing a single layer's quantization safety.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct LayerQuantization {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub layer_type: String,
    #[pyo3(get)]
    pub min_bound: f32,
    #[pyo3(get)]
    pub max_bound: f32,
    #[pyo3(get)]
    pub max_abs: f32,
    #[pyo3(get)]
    pub output_shape: Vec<usize>,
    #[pyo3(get)]
    pub float16_safety: QuantSafety,
    #[pyo3(get)]
    pub int8_safety: QuantSafety,
    #[pyo3(get)]
    pub int8_scale: Option<f32>,
    #[pyo3(get)]
    pub has_overflow: bool,
    #[pyo3(get)]
    pub propagation_failed: bool,
}

#[pymethods]
impl LayerQuantization {
    pub(crate) fn __repr__(&self) -> String {
        format!(
            "LayerQuantization(name={}, f16={}, i8={})",
            repr_string(&self.name),
            self.float16_safety.__str__(),
            self.int8_safety.__str__()
        )
    }

    /// Check if safe for float16.
    ///
    /// Returns False if propagation failed, since the assessment is unreliable.
    pub fn is_float16_safe(&self) -> bool {
        !self.propagation_failed && matches!(self.float16_safety, QuantSafety::Safe)
    }

    /// Check if safe for int8 (with or without scaling).
    ///
    /// Returns False if propagation failed, since the assessment is unreliable.
    pub fn is_int8_safe(&self) -> bool {
        !self.propagation_failed
            && matches!(
                self.int8_safety,
                QuantSafety::Safe | QuantSafety::ScalingRequired
            )
    }
}

/// Result of a full quantization analysis.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct QuantizationResult {
    #[pyo3(get)]
    pub layers: Vec<LayerQuantization>,
    #[pyo3(get)]
    pub float16_safe: bool,
    #[pyo3(get)]
    pub int8_safe: bool,
    #[pyo3(get)]
    pub float16_overflow_count: usize,
    #[pyo3(get)]
    pub int8_overflow_count: usize,
    #[pyo3(get)]
    pub denormal_count: usize,
    #[pyo3(get)]
    pub input_epsilon: f32,
}

#[pymethods]
impl QuantizationResult {
    pub(crate) fn __repr__(&self) -> String {
        format!(
            "QuantizationResult(layers={}, float16_safe={}, int8_safe={})",
            self.layers.len(),
            self.float16_safe,
            self.int8_safe
        )
    }

    /// Get a formatted summary table.
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
                layer.float16_safety.__str__(),
                layer.int8_safety.__str__()
            ));
        }

        lines.push(String::new());
        lines.push(format!(
            "Float16: {} ({} overflow, {} denormal)",
            if self.float16_safe { "SAFE" } else { "UNSAFE" },
            self.float16_overflow_count,
            self.denormal_count
        ));
        lines.push(format!(
            "Int8:    {} ({} overflow)",
            if self.int8_safe { "SAFE" } else { "UNSAFE" },
            self.int8_overflow_count
        ));

        lines.join("\n")
    }

    /// Get layers that are unsafe for float16.
    pub fn float16_unsafe_layers(&self) -> Vec<LayerQuantization> {
        self.layers
            .iter()
            .filter(|l| {
                matches!(
                    l.float16_safety,
                    QuantSafety::Overflow | QuantSafety::Unknown
                )
            })
            .cloned()
            .collect()
    }

    /// Get layers that are unsafe for int8.
    pub fn int8_unsafe_layers(&self) -> Vec<LayerQuantization> {
        self.layers
            .iter()
            .filter(|l| matches!(l.int8_safety, QuantSafety::Overflow | QuantSafety::Unknown))
            .cloned()
            .collect()
    }
}

/// Check if model layers can safely be quantized to float16/int8.
///
/// Uses bound propagation to determine the output range of each layer,
/// then checks if those ranges fit within the target format.
///
/// Args:
///     model_path: Path to ONNX model
///     epsilon: Input perturbation size (default: 0.01)
///     check_float16: Check float16 safety (default: True)
///     check_int8: Check int8 safety (default: True)
///
/// Returns:
///     QuantizationResult with per-layer safety analysis
///
/// Example:
///     >>> result = ny.quantize_check("model.onnx")
///     >>> assert result.float16_safe, "Model has float16 overflow risk"
///     >>> for layer in result.float16_unsafe_layers():
///     ...     print(f"  Unsafe: {layer.name}")
#[pyfunction]
#[pyo3(signature = (model_path, epsilon=0.01, check_float16=true, check_int8=true))]
pub fn quantize_check(
    py: Python<'_>,
    model_path: &str,
    epsilon: f32,
    check_float16: bool,
    check_int8: bool,
) -> PyResult<QuantizationResult> {
    validate_epsilon(epsilon)?;

    let config = build_quantize_config(epsilon);

    let result = Python::detach(py, || quantize::analyze_quantization(model_path, &config))
        .map_err(|e| PyValueError::new_err(format!("Quantization error: {}", e)))?;

    convert_quantization_result(result, check_float16, check_int8)
}

/// Check quantization safety from in-memory ONNX bytes.
///
/// Args:
///     model_bytes: ONNX model as bytes
///     epsilon: Input perturbation size (default: 0.01)
///     check_float16: Check float16 safety (default: True)
///     check_int8: Check int8 safety (default: True)
///     name: Model name for diagnostics (default: "model")
///
/// Returns:
///     QuantizationResult with per-layer safety analysis
///
/// Example:
///     >>> with open("model.onnx", "rb") as f:
///     ...     data = f.read()
///     >>> result = ny.quantize_check_bytes(data)
#[pyfunction]
#[pyo3(signature = (model_bytes, epsilon=0.01, check_float16=true, check_int8=true, name="model"))]
pub fn quantize_check_bytes(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    epsilon: f32,
    check_float16: bool,
    check_int8: bool,
    name: &str,
) -> PyResult<QuantizationResult> {
    validate_epsilon(epsilon)?;

    let config = build_quantize_config(epsilon);
    let name = name.to_string();

    let result = Python::detach(py, || {
        let model = load_onnx_bytes(&name, &model_bytes)
            .map_err(|e| quantize::QuantizeError::load("quantize/python", e))?;
        quantize::analyze_quantization_model(&model, &config)
    })
    .map_err(|e| PyValueError::new_err(format!("Quantization error: {}", e)))?;

    convert_quantization_result(result, check_float16, check_int8)
}

/// Check quantization safety of a PyTorch model without writing to disk.
///
/// Args:
///     model: PyTorch model (nn.Module)
///     example_input: Example input tensor for ONNX export
///     epsilon: Input perturbation size (default: 0.01)
///     check_float16: Check float16 safety (default: True)
///     check_int8: Check int8 safety (default: True)
///     opset: ONNX opset version (default: 17)
///
/// Returns:
///     QuantizationResult with per-layer safety analysis
///
/// Example:
///     >>> import torch
///     >>> model = MyModel()
///     >>> example = torch.randn(1, 3, 224, 224)
///     >>> result = ny.quantize_check_torch(model, example)
#[pyfunction]
#[pyo3(signature = (model, example_input, epsilon=0.01, check_float16=true, check_int8=true, opset=17))]
pub fn quantize_check_torch(
    py: Python<'_>,
    model: &Bound<'_, PyAny>,
    example_input: &Bound<'_, PyAny>,
    epsilon: f32,
    check_float16: bool,
    check_int8: bool,
    opset: u32,
) -> PyResult<QuantizationResult> {
    let model_bytes =
        export_torch_to_onnx_bytes(py, model, example_input, opset, "quantize_check_torch")?;
    quantize_check_bytes(
        py,
        model_bytes,
        epsilon,
        check_float16,
        check_int8,
        "torch_model",
    )
}

fn build_quantize_config(epsilon: f32) -> quantize::QuantizeConfig {
    quantize::QuantizeConfig {
        epsilon,
        continue_after_overflow: true,
        input: None,
    }
}

fn convert_quantization_result(
    result: quantize::QuantizeResult,
    check_float16: bool,
    check_int8: bool,
) -> PyResult<QuantizationResult> {
    let layers: Vec<LayerQuantization> = result
        .layers
        .into_iter()
        .map(|l| LayerQuantization {
            name: l.name,
            layer_type: l.layer_type,
            min_bound: l.min_bound,
            max_bound: l.max_bound,
            max_abs: l.max_abs,
            output_shape: l.output_shape,
            float16_safety: if check_float16 {
                l.float16_safety.into()
            } else {
                QuantSafety::Safe
            },
            int8_safety: if check_int8 {
                l.int8_safety.into()
            } else {
                QuantSafety::Safe
            },
            int8_scale: if check_int8 { l.int8_scale } else { None },
            has_overflow: l.has_overflow,
            propagation_failed: l.propagation_failed,
        })
        .collect();

    Ok(QuantizationResult {
        layers,
        float16_safe: if check_float16 {
            result.float16_safe
        } else {
            true
        },
        int8_safe: if check_int8 { result.int8_safe } else { true },
        float16_overflow_count: if check_float16 {
            result.float16_overflow_count
        } else {
            0
        },
        int8_overflow_count: if check_int8 {
            result.int8_overflow_count
        } else {
            0
        },
        denormal_count: if check_float16 {
            result.denormal_count
        } else {
            0
        },
        input_epsilon: result.input_epsilon,
    })
}
