// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) use ny_core::truncate_name;

/// Validate that epsilon is non-negative and finite.
///
/// Returns `Err(PyValueError)` for NaN, negative, or infinite epsilon values.
/// This prevents downstream Rust panics (e.g., `BoundedTensor::from_epsilon`,
/// `Bound::new`) from surfacing as `PanicException` in Python.
pub(crate) fn validate_epsilon(epsilon: f32) -> PyResult<()> {
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(PyValueError::new_err(format!(
            "epsilon must be non-negative and finite, got {}",
            epsilon
        )));
    }
    Ok(())
}

/// Validate that tolerance is non-negative and finite.
///
/// NaN tolerance causes `diff > tolerance` to always be false, silently
/// accepting all comparisons regardless of actual divergence.
pub(crate) fn validate_tolerance(tolerance: f32) -> PyResult<()> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(PyValueError::new_err(format!(
            "tolerance must be non-negative and finite, got {}",
            tolerance
        )));
    }
    Ok(())
}

/// Validate that a numpy input array contains only finite values.
///
/// Returns `Err(PyValueError)` if any element is NaN or Inf. Without this check,
/// NaN inputs propagate silently through diff/compare, producing meaningless
/// max_diff values that mask real divergence.
pub(crate) fn validate_input_finite(arr: &ndarray::ArrayD<f32>) -> PyResult<()> {
    if arr.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(
            "Input array contains NaN or Inf values",
        ));
    }
    Ok(())
}

/// Export a PyTorch model to ONNX bytes in memory.
///
/// # Arguments
/// * `py` - Python interpreter reference
/// * `model` - PyTorch model to export
/// * `example_input` - Example input tensor for tracing
/// * `opset` - ONNX opset version
/// * `caller` - Name of calling function for error messages
pub(crate) fn export_torch_to_onnx_bytes(
    py: Python<'_>,
    model: &Bound<'_, PyAny>,
    example_input: &Bound<'_, PyAny>,
    opset: u32,
    caller: &str,
) -> PyResult<Vec<u8>> {
    let torch = PyModule::import(py, "torch").map_err(|_| {
        PyValueError::new_err(format!(
            "PyTorch is required for {}; install torch before using this API.",
            caller
        ))
    })?;
    let onnx = torch.getattr("onnx")?;
    let io = PyModule::import(py, "io")?;
    let buffer = io.call_method0("BytesIO")?;

    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("opset_version", opset)?;
    kwargs.set_item("export_params", true)?;
    kwargs.set_item("do_constant_folding", true)?;

    onnx.call_method(
        "export",
        (model, example_input, buffer.clone()),
        Some(&kwargs),
    )?;

    let data = buffer.call_method0("getvalue")?;
    data.extract::<Vec<u8>>()
}
