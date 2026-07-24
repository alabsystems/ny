// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::repr::repr_string;
use crate::utils::{export_torch_to_onnx_bytes, truncate_name, validate_epsilon};
use ny_onnx::load_onnx_bytes;
use ny_onnx::profile::{self, BoundStatus as RustBoundStatus};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Bound width status indicator.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum BoundStatus {
    Tight,
    Moderate,
    Wide,
    VeryWide,
    Overflow,
}

#[pymethods]
impl BoundStatus {
    pub(crate) fn __repr__(&self) -> String {
        match self {
            BoundStatus::Tight => "BoundStatus.Tight".to_string(),
            BoundStatus::Moderate => "BoundStatus.Moderate".to_string(),
            BoundStatus::Wide => "BoundStatus.Wide".to_string(),
            BoundStatus::VeryWide => "BoundStatus.VeryWide".to_string(),
            BoundStatus::Overflow => "BoundStatus.Overflow".to_string(),
        }
    }

    pub(crate) fn __str__(&self) -> String {
        match self {
            BoundStatus::Tight => "TIGHT".to_string(),
            BoundStatus::Moderate => "MODERATE".to_string(),
            BoundStatus::Wide => "WIDE".to_string(),
            BoundStatus::VeryWide => "VERY WIDE".to_string(),
            BoundStatus::Overflow => "OVERFLOW".to_string(),
        }
    }
}

impl From<RustBoundStatus> for BoundStatus {
    fn from(s: RustBoundStatus) -> Self {
        match s {
            RustBoundStatus::Tight => BoundStatus::Tight,
            RustBoundStatus::Moderate => BoundStatus::Moderate,
            RustBoundStatus::Wide => BoundStatus::Wide,
            RustBoundStatus::VeryWide => BoundStatus::VeryWide,
            RustBoundStatus::Overflow => BoundStatus::Overflow,
        }
    }
}

/// Result of profiling a single layer's bounds.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct LayerProfile {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub layer_type: String,
    #[pyo3(get)]
    pub input_width: f32,
    #[pyo3(get)]
    pub output_width: f32,
    #[pyo3(get)]
    pub mean_output_width: f32,
    #[pyo3(get)]
    pub median_output_width: f32,
    #[pyo3(get)]
    pub growth_ratio: f32,
    #[pyo3(get)]
    pub cumulative_expansion: f32,
    #[pyo3(get)]
    pub output_shape: Vec<usize>,
    #[pyo3(get)]
    pub num_elements: usize,
    #[pyo3(get)]
    pub status: BoundStatus,
}

#[pymethods]
impl LayerProfile {
    pub(crate) fn __repr__(&self) -> String {
        format!(
            "LayerProfile(name={}, growth={:.2}x, status={})",
            repr_string(&self.name),
            self.growth_ratio,
            self.status.__str__()
        )
    }

    /// Check if this layer is a choke point (high growth).
    pub(crate) fn is_choke_point(&self, threshold: f32) -> bool {
        self.growth_ratio > threshold
    }
}

/// Result of a full bound profiling analysis.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct ProfileResult {
    #[pyo3(get)]
    pub layers: Vec<LayerProfile>,
    #[pyo3(get)]
    pub input_epsilon: f32,
    #[pyo3(get)]
    pub initial_width: f32,
    #[pyo3(get)]
    pub final_width: f32,
    #[pyo3(get)]
    pub total_expansion: f32,
    #[pyo3(get)]
    pub max_growth_layer: Option<usize>,
    #[pyo3(get)]
    pub max_growth_ratio: f32,
    #[pyo3(get)]
    pub overflow_at_layer: Option<usize>,
    #[pyo3(get)]
    pub difficulty_score: f32,
}

#[pymethods]
impl ProfileResult {
    pub(crate) fn __repr__(&self) -> String {
        format!(
            "ProfileResult(layers={}, expansion={:.2}x, difficulty={:.0}/100)",
            self.layers.len(),
            self.total_expansion,
            self.difficulty_score
        )
    }

    /// Get a formatted summary table.
    pub(crate) fn summary(&self) -> String {
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
                layer.status.__str__(),
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
            "Verification difficulty: {:.0}/100",
            self.difficulty_score
        ));

        lines.join("\n")
    }

    /// Get name of layer with maximum growth.
    #[getter]
    pub(crate) fn max_growth_layer_name(&self) -> Option<String> {
        self.max_growth_layer
            .and_then(|i| self.layers.get(i))
            .map(|l| l.name.clone())
    }

    /// Get choke points (layers with growth above threshold).
    pub(crate) fn choke_points(&self, threshold: f32) -> Vec<LayerProfile> {
        self.layers
            .iter()
            .filter(|l| l.growth_ratio > threshold)
            .cloned()
            .collect()
    }

    /// Get problematic layers (wide or worse bounds).
    pub(crate) fn problematic_layers(&self) -> Vec<LayerProfile> {
        self.layers
            .iter()
            .filter(|l| {
                matches!(
                    l.status,
                    BoundStatus::Wide | BoundStatus::VeryWide | BoundStatus::Overflow
                )
            })
            .cloned()
            .collect()
    }

    /// Check if overflow occurred.
    #[getter]
    pub(crate) fn has_overflow(&self) -> bool {
        self.overflow_at_layer.is_some()
    }
}

/// Profile bound widths through the network.
///
/// Tracks how bound widths grow layer-by-layer, helping identify where
/// verification becomes difficult. Also computes a verification difficulty score.
///
/// Args:
///     model_path: Path to ONNX model
///     epsilon: Input perturbation size (default: 0.01)
///
/// Returns:
///     ProfileResult with per-layer bound analysis
///
/// Example:
///     >>> result = ny.profile_bounds("model.onnx")
///     >>> print(f"Difficulty: {result.difficulty_score:.0f}/100")
///     >>> for layer in result.choke_points(5.0):
///     ...     print(f"  {layer.name}: {layer.growth_ratio:.2f}x growth")
#[pyfunction]
#[pyo3(signature = (model_path, epsilon=0.01))]
pub fn profile_bounds(py: Python<'_>, model_path: &str, epsilon: f32) -> PyResult<ProfileResult> {
    validate_epsilon(epsilon)?;

    let config = build_profile_config(epsilon);

    let result = Python::detach(py, || profile::profile_bounds(model_path, &config))
        .map_err(|e| PyValueError::new_err(format!("Profile error: {}", e)))?;

    convert_profile_result(result)
}

/// Profile bound widths from in-memory ONNX bytes.
///
/// Args:
///     model_bytes: ONNX model as bytes
///     epsilon: Input perturbation size (default: 0.01)
///     name: Model name for diagnostics (default: "model")
///
/// Returns:
///     ProfileResult with per-layer bound analysis
///
/// Example:
///     >>> with open("model.onnx", "rb") as f:
///     ...     data = f.read()
///     >>> result = ny.profile_bounds_bytes(data)
#[pyfunction]
#[pyo3(signature = (model_bytes, epsilon=0.01, name="model"))]
pub fn profile_bounds_bytes(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    epsilon: f32,
    name: &str,
) -> PyResult<ProfileResult> {
    validate_epsilon(epsilon)?;

    let config = build_profile_config(epsilon);
    let name = name.to_string();

    let result = Python::detach(py, || {
        let model = load_onnx_bytes(&name, &model_bytes)
            .map_err(|e| profile::ProfileError::load("profile/python", e))?;
        profile::profile_bounds_model(&model, &config)
    })
    .map_err(|e| PyValueError::new_err(format!("Profile error: {}", e)))?;

    convert_profile_result(result)
}

/// Profile bound widths of a PyTorch model without writing to disk.
///
/// Args:
///     model: PyTorch model (nn.Module)
///     example_input: Example input tensor for ONNX export
///     epsilon: Input perturbation size (default: 0.01)
///     opset: ONNX opset version (default: 17)
///
/// Returns:
///     ProfileResult with per-layer bound analysis
///
/// Example:
///     >>> import torch
///     >>> model = MyModel()
///     >>> example = torch.randn(1, 3, 224, 224)
///     >>> result = ny.profile_bounds_torch(model, example)
#[pyfunction]
#[pyo3(signature = (model, example_input, epsilon=0.01, opset=17))]
pub fn profile_bounds_torch(
    py: Python<'_>,
    model: &Bound<'_, PyAny>,
    example_input: &Bound<'_, PyAny>,
    epsilon: f32,
    opset: u32,
) -> PyResult<ProfileResult> {
    let model_bytes =
        export_torch_to_onnx_bytes(py, model, example_input, opset, "profile_bounds_torch")?;
    profile_bounds_bytes(py, model_bytes, epsilon, "torch_model")
}

fn build_profile_config(epsilon: f32) -> profile::ProfileConfig {
    profile::ProfileConfig {
        epsilon,
        continue_after_overflow: true,
        input: None,
    }
}

fn convert_profile_result(result: profile::ProfileResult) -> PyResult<ProfileResult> {
    let layers: Vec<LayerProfile> = result
        .layers
        .into_iter()
        .map(|l| LayerProfile {
            name: l.name,
            layer_type: l.layer_type,
            input_width: l.input_width,
            output_width: l.output_width,
            mean_output_width: l.mean_output_width,
            median_output_width: l.median_output_width,
            growth_ratio: l.growth_ratio,
            cumulative_expansion: l.cumulative_expansion,
            output_shape: l.output_shape,
            num_elements: l.num_elements,
            status: l.status.into(),
        })
        .collect();

    Ok(ProfileResult {
        layers,
        input_epsilon: result.input_epsilon,
        initial_width: result.initial_width,
        final_width: result.final_width,
        total_expansion: result.total_expansion,
        max_growth_layer: result.max_growth_layer,
        max_growth_ratio: result.max_growth_ratio,
        overflow_at_layer: result.overflow_at_layer,
        difficulty_score: result.difficulty_score,
    })
}
