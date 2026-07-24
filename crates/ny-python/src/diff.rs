// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::repr::repr_string;
use crate::utils::{
    export_torch_to_onnx_bytes, truncate_name, validate_input_finite, validate_tolerance,
};
use numpy::{PyArrayDyn, PyArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

fn dtype_to_str(dtype: &ny_onnx::DataType) -> &'static str {
    match dtype {
        ny_onnx::DataType::Float32 => "float32",
        ny_onnx::DataType::Float16 => "float16",
        ny_onnx::DataType::Int64 => "int64",
        ny_onnx::DataType::Int32 => "int32",
    }
}

/// Result of comparing a single layer between two models.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct LayerComparison {
    #[pyo3(get)]
    pub name: String,

    #[pyo3(get)]
    pub name_b: Option<String>,

    #[pyo3(get)]
    pub max_diff: f32,

    #[pyo3(get)]
    pub mean_diff: f32,

    #[pyo3(get)]
    pub exceeds_tolerance: bool,

    #[pyo3(get)]
    pub shape_a: Vec<usize>,

    #[pyo3(get)]
    pub shape_b: Vec<usize>,
}

#[pymethods]
impl LayerComparison {
    pub(crate) fn __repr__(&self) -> String {
        format!(
            "LayerComparison(name={}, max_diff={:.2e}, exceeds={})",
            repr_string(&self.name),
            self.max_diff,
            self.exceeds_tolerance
        )
    }
}

/// Tensor metadata exposed by `load_model_info`.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct TensorSpec {
    #[pyo3(get)]
    pub name: String,

    #[pyo3(get)]
    pub shape: Vec<i64>,

    #[pyo3(get)]
    pub dtype: String,
}

#[pymethods]
impl TensorSpec {
    pub(crate) fn __repr__(&self) -> String {
        format!(
            "TensorSpec(name={}, shape={:?}, dtype={})",
            repr_string(&self.name),
            self.shape,
            repr_string(&self.dtype)
        )
    }
}

impl From<&ny_onnx::TensorSpec> for TensorSpec {
    fn from(spec: &ny_onnx::TensorSpec) -> Self {
        Self {
            name: spec.name.clone(),
            shape: spec.shape.clone(),
            dtype: dtype_to_str(&spec.dtype).to_string(),
        }
    }
}

/// Typed model metadata exposed by `load_model_info`.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct ModelInfo {
    #[pyo3(get)]
    pub inputs: Vec<TensorSpec>,

    #[pyo3(get)]
    pub outputs: Vec<TensorSpec>,

    #[pyo3(get)]
    pub layer_count: usize,

    #[pyo3(get)]
    pub layer_names: Vec<String>,
}

#[pymethods]
impl ModelInfo {
    pub(crate) fn __repr__(&self) -> String {
        format!(
            "ModelInfo(inputs={}, outputs={}, layer_count={})",
            self.inputs.len(),
            self.outputs.len(),
            self.layer_count
        )
    }
}

/// Status of a layer comparison.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum DiffStatus {
    Ok,
    DriftStarts,
    ExceedsTolerance,
    ShapeMismatch,
}

#[pymethods]
impl DiffStatus {
    pub(crate) fn __repr__(&self) -> String {
        match self {
            DiffStatus::Ok => "DiffStatus.Ok".to_string(),
            DiffStatus::DriftStarts => "DiffStatus.DriftStarts".to_string(),
            DiffStatus::ExceedsTolerance => "DiffStatus.ExceedsTolerance".to_string(),
            DiffStatus::ShapeMismatch => "DiffStatus.ShapeMismatch".to_string(),
        }
    }

    pub(crate) fn __str__(&self) -> String {
        match self {
            DiffStatus::Ok => "ok".to_string(),
            DiffStatus::DriftStarts => "drift_starts".to_string(),
            DiffStatus::ExceedsTolerance => "exceeds_tolerance".to_string(),
            DiffStatus::ShapeMismatch => "shape_mismatch".to_string(),
        }
    }
}

/// Result of a full model diff operation.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct DiffResult {
    #[pyo3(get)]
    pub layers: Vec<LayerComparison>,

    #[pyo3(get)]
    pub first_bad_layer: Option<usize>,

    #[pyo3(get)]
    pub drift_start_layer: Option<usize>,

    #[pyo3(get)]
    pub max_divergence: f32,

    #[pyo3(get)]
    pub tolerance: f32,

    #[pyo3(get)]
    pub suggestion: Option<String>,
}

#[pymethods]
impl DiffResult {
    /// Check if models are equivalent within tolerance.
    #[getter]
    pub(crate) fn is_equivalent(&self) -> bool {
        self.first_bad_layer.is_none()
    }

    /// Get the name of the first bad layer, if any.
    #[getter]
    pub(crate) fn first_bad_layer_name(&self) -> Option<String> {
        self.first_bad_layer
            .and_then(|i| self.layers.get(i))
            .map(|l| l.name.clone())
    }

    /// Get status for each layer.
    pub(crate) fn statuses(&self) -> Vec<DiffStatus> {
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

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "DiffResult(layers={}, max_divergence={:.2e}, is_equivalent={})",
            self.layers.len(),
            self.max_divergence,
            self.is_equivalent()
        )
    }

    /// Get a formatted summary table (like CLI output).
    pub(crate) fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Layer-by-Layer Comparison".to_string());
        lines.push("==========================".to_string());
        lines.push(format!(
            "{:<40} | {:>12} | {}",
            "Layer", "Max Diff", "Status"
        ));
        lines.push(format!("{:-<40}-+-{:-<12}-+--------", "", ""));

        let statuses = self.statuses();
        for (layer, status) in self.layers.iter().zip(statuses.iter()) {
            let status_str = match status {
                DiffStatus::Ok => "OK",
                DiffStatus::DriftStarts => "DRIFT STARTS",
                DiffStatus::ExceedsTolerance => "EXCEEDS",
                DiffStatus::ShapeMismatch => "SHAPE MISMATCH",
            };
            lines.push(format!(
                "{:<40} | {:>12.2e} | {}",
                truncate_name(&layer.name, 40),
                layer.max_diff,
                status_str
            ));
        }

        if let Some(ref suggestion) = self.suggestion {
            lines.push(String::new());
            lines.push(format!("Suggestion: {}", suggestion));
        }

        lines.join("\n")
    }
}

/// Compare two ONNX models layer-by-layer to find divergence.
///
/// This is the main entry point for model comparison. It runs inference on both
/// models with the same input and compares intermediate outputs at each layer.
///
/// Args:
///     model_a: Path to first ONNX model
///     model_b: Path to second ONNX model
///     tolerance: Maximum allowed difference (default: 1e-5)
///     input: Optional numpy array for input (default: zeros)
///     continue_after_divergence: Whether to continue after finding divergence (default: True)
///     layer_mapping: Optional dict mapping layer names from A to B
///     diagnose: Enable root cause diagnosis (default: False)
///
/// Returns:
///     DiffResult with comparison results
///
/// Example:
///     >>> diff = ny.diff("model_a.onnx", "model_b.onnx")
///     >>> assert diff.is_equivalent, f"Diverges at {diff.first_bad_layer_name}"
#[pyfunction]
#[pyo3(signature = (model_a, model_b, tolerance=1e-5, input=None, continue_after_divergence=true, layer_mapping=None, diagnose=false))]
// Justification: Python API binding — pyo3 requires all parameters as function arguments.
#[allow(clippy::too_many_arguments)]
pub fn diff(
    py: Python<'_>,
    model_a: &str,
    model_b: &str,
    tolerance: f32,
    input: Option<&Bound<'_, PyArrayDyn<f32>>>,
    continue_after_divergence: bool,
    layer_mapping: Option<HashMap<String, String>>,
    diagnose: bool,
) -> PyResult<DiffResult> {
    validate_tolerance(tolerance)?;

    // Convert numpy input if provided, rejecting NaN/Inf
    let input_array = input
        .map(|arr| -> PyResult<_> {
            let readonly = arr.readonly();
            let owned = readonly.as_array().to_owned();
            validate_input_finite(&owned)?;
            Ok(owned)
        })
        .transpose()?;

    // Build config
    let config = ny_onnx::diff::DiffConfig {
        tolerance,
        continue_after_divergence,
        input: input_array,
        layer_mapping: layer_mapping.unwrap_or_default(),
        diagnose,
    };

    // Run diff (release GIL during computation)
    let result = Python::detach(py, || ny_onnx::diff::diff_models(model_a, model_b, &config))
        .map_err(|e| PyValueError::new_err(format!("Diff error: {}", e)))?;

    // Convert to Python types
    let layers: Vec<LayerComparison> = result
        .layers
        .into_iter()
        .map(|l| LayerComparison {
            name: l.name,
            name_b: l.name_b,
            max_diff: l.max_diff,
            mean_diff: l.mean_diff,
            exceeds_tolerance: l.exceeds_tolerance,
            shape_a: l.shape_a,
            shape_b: l.shape_b,
        })
        .collect();

    Ok(DiffResult {
        layers,
        first_bad_layer: result.first_bad_layer,
        drift_start_layer: result.drift_start_layer,
        max_divergence: result.max_divergence,
        tolerance: result.tolerance,
        suggestion: result.suggestion,
    })
}

/// Run inference on an ONNX model and return all intermediate outputs.
///
/// This is useful for inspecting what's happening inside a model.
///
/// Args:
///     model_path: Path to ONNX model
///     input: Numpy array input
///
/// Returns:
///     Dict mapping layer names to numpy arrays
#[pyfunction]
pub fn run_with_intermediates<'py>(
    py: Python<'py>,
    model_path: &str,
    input: &Bound<'py, PyArrayDyn<f32>>,
) -> PyResult<HashMap<String, Py<PyArrayDyn<f32>>>> {
    // Convert input, rejecting NaN/Inf
    let readonly = input.readonly();
    let input_array = readonly.as_array().to_owned();
    validate_input_finite(&input_array)?;

    // Run inference (release GIL)
    let outputs = Python::detach(py, || {
        ny_onnx::diff::run_inference_with_intermediates(model_path, &input_array)
    })
    .map_err(|e| PyValueError::new_err(format!("Inference error: {}", e)))?;

    // Convert outputs to numpy arrays
    let mut result = HashMap::new();
    for (name, arr) in outputs {
        let py_arr = arr.to_pyarray(py).unbind();
        result.insert(name, py_arr);
    }

    Ok(result)
}

/// Load model info (inputs, outputs, layers).
///
/// Args:
///     model_path: Path to ONNX model
///
/// Returns:
///     ModelInfo with typed tensor metadata
#[pyfunction]
pub fn load_model_info(model_path: &str) -> PyResult<ModelInfo> {
    let info = ny_onnx::diff::load_model_info(model_path)
        .map_err(|e| PyValueError::new_err(format!("Load error: {}", e)))?;
    let layer_names: Vec<String> = info.layers.iter().map(|l| l.name.clone()).collect();

    Ok(ModelInfo {
        inputs: info.inputs.iter().map(TensorSpec::from).collect(),
        outputs: info.outputs.iter().map(TensorSpec::from).collect(),
        layer_count: info.layers.len(),
        layer_names,
    })
}

/// Load a numpy file (.npy).
#[pyfunction]
pub fn load_npy(py: Python<'_>, path: &str) -> PyResult<Py<PyArrayDyn<f32>>> {
    let arr = ny_onnx::diff::load_npy(path)
        .map_err(|e| PyValueError::new_err(format!("NPY load error: {}", e)))?;
    Ok(arr.to_pyarray(py).unbind())
}

/// Compare two ONNX models from in-memory bytes.
///
/// This is the bytes variant of diff() for in-memory ONNX models.
///
/// Args:
///     model_a_bytes: First ONNX model as bytes
///     model_b_bytes: Second ONNX model as bytes
///     tolerance: Maximum allowed difference (default: 1e-5)
///     input: Optional numpy array for input (default: zeros)
///     continue_after_divergence: Whether to continue after finding divergence (default: True)
///     layer_mapping: Optional dict mapping layer names from A to B
///     diagnose: Enable root cause diagnosis (default: False)
///     name_a: Name for model A (default: "model_a")
///     name_b: Name for model B (default: "model_b")
///
/// Returns:
///     DiffResult with comparison results
///
/// Example:
///     >>> with open("model_a.onnx", "rb") as f:
///     ...     bytes_a = f.read()
///     >>> with open("model_b.onnx", "rb") as f:
///     ...     bytes_b = f.read()
///     >>> diff = ny.diff_bytes(bytes_a, bytes_b)
///     >>> assert diff.is_equivalent
#[pyfunction]
#[pyo3(signature = (model_a_bytes, model_b_bytes, tolerance=1e-5, input=None, continue_after_divergence=true, layer_mapping=None, diagnose=false, name_a="model_a", name_b="model_b"))]
// Justification: Python API binding — pyo3 requires all parameters as function arguments.
#[allow(clippy::too_many_arguments)]
pub fn diff_bytes(
    py: Python<'_>,
    model_a_bytes: Vec<u8>,
    model_b_bytes: Vec<u8>,
    tolerance: f32,
    input: Option<&Bound<'_, PyArrayDyn<f32>>>,
    continue_after_divergence: bool,
    layer_mapping: Option<HashMap<String, String>>,
    diagnose: bool,
    name_a: &str,
    name_b: &str,
) -> PyResult<DiffResult> {
    validate_tolerance(tolerance)?;

    // Convert numpy input if provided, rejecting NaN/Inf
    let input_array = input
        .map(|arr| -> PyResult<_> {
            let readonly = arr.readonly();
            let owned = readonly.as_array().to_owned();
            validate_input_finite(&owned)?;
            Ok(owned)
        })
        .transpose()?;

    let config = ny_onnx::diff::DiffConfig {
        tolerance,
        continue_after_divergence,
        input: input_array,
        layer_mapping: layer_mapping.unwrap_or_default(),
        diagnose,
    };

    let name_a = name_a.to_string();
    let name_b = name_b.to_string();

    let result = Python::detach(py, || {
        ny_onnx::diff::diff_models_bytes(&name_a, &model_a_bytes, &name_b, &model_b_bytes, &config)
    })
    .map_err(|e| PyValueError::new_err(format!("Diff error: {}", e)))?;

    let layers: Vec<LayerComparison> = result
        .layers
        .into_iter()
        .map(|l| LayerComparison {
            name: l.name,
            name_b: l.name_b,
            max_diff: l.max_diff,
            mean_diff: l.mean_diff,
            exceeds_tolerance: l.exceeds_tolerance,
            shape_a: l.shape_a,
            shape_b: l.shape_b,
        })
        .collect();

    Ok(DiffResult {
        layers,
        first_bad_layer: result.first_bad_layer,
        drift_start_layer: result.drift_start_layer,
        max_divergence: result.max_divergence,
        tolerance: result.tolerance,
        suggestion: result.suggestion,
    })
}

/// Compare two PyTorch models layer-by-layer without writing to disk.
///
/// This exports both models to ONNX in memory and compares them.
///
/// Args:
///     model_a: First PyTorch model (nn.Module)
///     model_b: Second PyTorch model (nn.Module)
///     example_input: Example input tensor (used for tracing both models)
///     tolerance: Maximum allowed difference (default: 1e-5)
///     input: Optional numpy array for diff input (default: zeros)
///     continue_after_divergence: Whether to continue after finding divergence (default: True)
///     layer_mapping: Optional dict mapping layer names from A to B
///     diagnose: Enable root cause diagnosis (default: False)
///     opset: ONNX opset version (default: 17)
///
/// Returns:
///     DiffResult with comparison results
///
/// Example:
///     >>> model_a = MyModel()
///     >>> model_b = MyModel()
///     >>> model_b.load_state_dict(torch.load("checkpoint.pt"))
///     >>> example = torch.randn(1, 3, 224, 224)
///     >>> diff = ny.diff_torch(model_a, model_b, example)
///     >>> assert diff.is_equivalent
#[pyfunction]
#[pyo3(signature = (model_a, model_b, example_input, tolerance=1e-5, input=None, continue_after_divergence=true, layer_mapping=None, diagnose=false, opset=17))]
// Justification: Python API binding — pyo3 requires all parameters as function arguments.
#[allow(clippy::too_many_arguments)]
pub fn diff_torch(
    py: Python<'_>,
    model_a: &Bound<'_, PyAny>,
    model_b: &Bound<'_, PyAny>,
    example_input: &Bound<'_, PyAny>,
    tolerance: f32,
    input: Option<&Bound<'_, PyArrayDyn<f32>>>,
    continue_after_divergence: bool,
    layer_mapping: Option<HashMap<String, String>>,
    diagnose: bool,
    opset: u32,
) -> PyResult<DiffResult> {
    let bytes_a = export_torch_to_onnx_bytes(py, model_a, example_input, opset, "diff_torch")?;
    let bytes_b = export_torch_to_onnx_bytes(py, model_b, example_input, opset, "diff_torch")?;

    diff_bytes(
        py,
        bytes_a,
        bytes_b,
        tolerance,
        input,
        continue_after_divergence,
        layer_mapping,
        diagnose,
        "torch_model_a",
        "torch_model_b",
    )
}

#[cfg(test)]
mod tests {
    use super::{load_model_info, DiffStatus};

    fn simple_mlp_path() -> String {
        format!(
            "{}/../../tests/models/simple_mlp.onnx",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    #[test]
    fn test_diff_status_str_is_snake_case_3942() {
        assert_eq!(DiffStatus::Ok.__str__(), "ok");
        assert_eq!(DiffStatus::DriftStarts.__str__(), "drift_starts");
        assert_eq!(DiffStatus::ExceedsTolerance.__str__(), "exceeds_tolerance");
        assert_eq!(DiffStatus::ShapeMismatch.__str__(), "shape_mismatch");
    }

    #[test]
    fn test_load_model_info_returns_typed_surface_3942() {
        let info = load_model_info(&simple_mlp_path()).expect("load model info should succeed");

        assert!(info.layer_count > 0, "expected at least one layer");
        assert!(
            !info.inputs.is_empty(),
            "expected typed input tensor metadata to be present"
        );
        assert!(
            !info.outputs.is_empty(),
            "expected typed output tensor metadata to be present"
        );
        assert_eq!(
            info.inputs[0].dtype, "float32",
            "simple_mlp input dtype should stay typed"
        );
    }
}
