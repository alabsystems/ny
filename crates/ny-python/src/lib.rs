// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

//! Python bindings for ny neural network verification.
//!
//! Provides a `pytest`-friendly API for neural network testing and debugging.
//!
//! ## Example Usage
//!
//! ```python
//! import ny
//!
//! def test_port_equivalent():
//!     diff = ny.diff("model_torch.onnx", "model_coreml.onnx")
//!     assert diff.max_divergence < 1e-5, f"Diverges at {diff.first_bad_layer}"
//!
//! def test_specific_tolerance():
//!     diff = ny.diff("model_a.onnx", "model_b.onnx", tolerance=1e-4)
//!     assert diff.is_equivalent
//! ```

// Link macOS Accelerate BLAS for ndarray::dot() acceleration (#4259).
#[cfg(target_os = "macos")]
extern crate blas_src;

mod bench;
mod beta_config;
mod compare;
mod custom_ops;
mod diff;
mod profile;
mod quantize;
mod repr;
mod sensitivity;
mod utils;
mod verify;
mod weights;

#[cfg(test)]
mod stub_contract_tests;
#[cfg(test)]
mod stub_contract_typed_consumer_tests;
#[cfg(test)]
mod tests;

/// Benchmarking API: run bound propagation benchmarks with timing and dimensions.
pub use bench::{run_benchmark, BenchDimensions, BenchResult, BenchResultItem};
/// Python-facing β-CROWN configuration types for branch-and-bound verification.
pub use beta_config::{BetaCrownConfig, PyBranchingHeuristic, PyKfsbReduceOp};
/// Model comparison API: check bound violations between reference and target networks.
pub use compare::{BoundViolation, CompareResult};
/// Custom operator registration types for Python-defined ONNX operators.
pub use custom_ops::{
    CustomOpAttribute, CustomOpSchema, PyCustomOpAttributeType as CustomOpAttributeType,
};
/// Model diff API: layer-by-layer comparison identifying where outputs diverge.
pub use diff::{DiffResult, DiffStatus, LayerComparison, ModelInfo, TensorSpec};
/// Bound width profiling API: identify layers where verification becomes difficult.
pub use profile::{profile_bounds, BoundStatus, LayerProfile, ProfileResult};
/// Quantization safety analysis: check float16/int8 overflow risk per layer.
pub use quantize::{quantize_check, LayerQuantization, QuantSafety, QuantizationResult};
/// Sensitivity analysis API: measure per-layer uncertainty amplification.
pub use sensitivity::{sensitivity_analysis, LayerSensitivity, SensitivityResult};
/// Verification API: verify network properties with soundness tracking.
pub use verify::{HeuristicUsed, OutputBound, SoundnessProvenance, VerifyResult, VerifyStatus};
/// Weight inspection and comparison API: tensor info, norms, and diffs.
pub use weights::{
    weights_diff, weights_info, TensorComparison, TensorComparisonStatus, TensorInfo,
    WeightsDiffResult, WeightsInfo,
};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

fn register_diff_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LayerComparison>()?;
    m.add_class::<TensorSpec>()?;
    m.add_class::<ModelInfo>()?;
    m.add_class::<DiffResult>()?;
    m.add_class::<DiffStatus>()?;
    m.add_function(wrap_pyfunction!(crate::diff::diff, m)?)?;
    m.add_function(wrap_pyfunction!(crate::diff::diff_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(crate::diff::diff_torch, m)?)?;
    m.add_function(wrap_pyfunction!(crate::diff::run_with_intermediates, m)?)?;
    m.add_function(wrap_pyfunction!(crate::diff::load_model_info, m)?)?;
    m.add_function(wrap_pyfunction!(crate::diff::load_npy, m)?)?;
    Ok(())
}

fn register_sensitivity_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LayerSensitivity>()?;
    m.add_class::<SensitivityResult>()?;
    m.add_function(wrap_pyfunction!(
        crate::sensitivity::sensitivity_analysis,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::sensitivity::sensitivity_analysis_bytes,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::sensitivity::sensitivity_analysis_torch,
        m
    )?)?;
    Ok(())
}

fn register_quantization_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<QuantSafety>()?;
    m.add_class::<LayerQuantization>()?;
    m.add_class::<QuantizationResult>()?;
    m.add_function(wrap_pyfunction!(crate::quantize::quantize_check, m)?)?;
    m.add_function(wrap_pyfunction!(crate::quantize::quantize_check_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(crate::quantize::quantize_check_torch, m)?)?;
    Ok(())
}

fn register_profile_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BoundStatus>()?;
    m.add_class::<LayerProfile>()?;
    m.add_class::<ProfileResult>()?;
    m.add_function(wrap_pyfunction!(crate::profile::profile_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(crate::profile::profile_bounds_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(crate::profile::profile_bounds_torch, m)?)?;
    Ok(())
}

fn register_verify_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OutputBound>()?;
    m.add_class::<VerifyStatus>()?;
    m.add_class::<HeuristicUsed>()?;
    m.add_class::<SoundnessProvenance>()?;
    m.add_class::<VerifyResult>()?;
    m.add_class::<PyBranchingHeuristic>()?;
    m.add_class::<PyKfsbReduceOp>()?;
    m.add_class::<BetaCrownConfig>()?;
    m.add_function(wrap_pyfunction!(crate::verify::verify, m)?)?;
    m.add_function(wrap_pyfunction!(crate::verify::verify_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(crate::verify::verify_torch, m)?)?;
    Ok(())
}

fn register_compare_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CompareResult>()?;
    m.add_class::<BoundViolation>()?;
    m.add_function(wrap_pyfunction!(crate::compare::compare, m)?)?;
    m.add_function(wrap_pyfunction!(crate::compare::compare_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(crate::compare::compare_torch, m)?)?;
    Ok(())
}

fn register_weights_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TensorInfo>()?;
    m.add_class::<WeightsInfo>()?;
    m.add_class::<TensorComparisonStatus>()?;
    m.add_class::<TensorComparison>()?;
    m.add_class::<WeightsDiffResult>()?;
    m.add_function(wrap_pyfunction!(crate::weights::weights_info, m)?)?;
    m.add_function(wrap_pyfunction!(crate::weights::weights_diff, m)?)?;
    Ok(())
}

fn register_benchmark_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BenchResultItem>()?;
    m.add_class::<BenchDimensions>()?;
    m.add_class::<BenchResult>()?;
    m.add_function(wrap_pyfunction!(crate::bench::run_benchmark, m)?)?;
    Ok(())
}

/// ny Python module.
///
/// Neural network verification and testing library.
#[pymodule]
fn ny(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_diff_api(m)?;
    register_sensitivity_api(m)?;
    register_quantization_api(m)?;
    register_profile_api(m)?;
    register_verify_api(m)?;
    custom_ops::register_custom_ops_module(m)?;
    register_compare_api(m)?;
    register_weights_api(m)?;
    register_benchmark_api(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
