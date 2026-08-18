// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::utils::{
    export_torch_to_onnx_bytes, validate_epsilon, validate_input_finite, validate_tolerance,
};
use crate::verify::resolve_verify_backend;
use numpy::{PyArrayDyn, PyArrayMethods};
use ny_core::{
    nan_propagating_max, Bound as CoreBound, GemmEngine, VerificationResult,
    VerificationSoundnessMode, VerificationSpec,
};
use ny_onnx::{load_onnx, load_onnx_bytes};
use ny_propagate::{
    build_difference_network, GraphNetwork, Network, PropagationConfig, PropagationMethod, Verifier,
};
use ny_tensor::BoundedTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Result of comparing two models using bound propagation.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct CompareResult {
    #[pyo3(get)]
    pub is_equivalent: bool,

    #[pyo3(get)]
    pub max_lower_diff: f32,

    #[pyo3(get)]
    pub max_upper_diff: f32,

    #[pyo3(get)]
    pub tolerance: f32,

    #[pyo3(get)]
    pub overlap_pct: f32,

    #[pyo3(get)]
    pub ref_max_width: f32,

    #[pyo3(get)]
    pub target_max_width: f32,

    #[pyo3(get)]
    pub method: String,

    #[pyo3(get)]
    pub epsilon: f32,

    #[pyo3(get)]
    pub violations: Vec<BoundViolation>,
}

#[pymethods]
impl CompareResult {
    pub(crate) fn __repr__(&self) -> String {
        format!(
            "CompareResult(is_equivalent={}, max_lower_diff={:.2e}, max_upper_diff={:.2e})",
            self.is_equivalent, self.max_lower_diff, self.max_upper_diff
        )
    }

    /// Get a formatted summary.
    pub(crate) fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Model Comparison Result".to_string());
        lines.push("=======================".to_string());
        lines.push(format!(
            "Functionally equivalent (proved): {}",
            if self.is_equivalent { "YES" } else { "NO" }
        ));
        lines.push(format!("Method: {}", self.method));
        lines.push(format!("Epsilon: {:.2e}", self.epsilon));
        lines.push(format!("Tolerance: {:.2e}", self.tolerance));
        lines.push(format!("Max lower bound diff: {:.2e}", self.max_lower_diff));
        lines.push(format!("Max upper bound diff: {:.2e}", self.max_upper_diff));
        lines.push(format!("Bound overlap: {:.2}%", self.overlap_pct));
        lines.push(format!("Reference max width: {:.2e}", self.ref_max_width));
        lines.push(format!("Target max width: {:.2e}", self.target_max_width));

        if !self.violations.is_empty() {
            lines.push(format!("\nViolations ({}):", self.violations.len()));
            for v in self.violations.iter().take(10) {
                lines.push(format!(
                    "  [{}] ref=[{:.6}, {:.6}] target=[{:.6}, {:.6}]",
                    v.index, v.ref_lower, v.ref_upper, v.target_lower, v.target_upper
                ));
            }
            if self.violations.len() > 10 {
                lines.push(format!("  ... and {} more", self.violations.len() - 10));
            }
        }

        lines.join("\n")
    }
}

/// A single bound violation between two models.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct BoundViolation {
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub ref_lower: f32,
    #[pyo3(get)]
    pub ref_upper: f32,
    #[pyo3(get)]
    pub target_lower: f32,
    #[pyo3(get)]
    pub target_upper: f32,
    #[pyo3(get)]
    pub lower_diff: f32,
    #[pyo3(get)]
    pub upper_diff: f32,
}

#[pymethods]
impl BoundViolation {
    pub(crate) fn __repr__(&self) -> String {
        format!(
            "BoundViolation(idx={}, lower_diff={:.2e}, upper_diff={:.2e})",
            self.index, self.lower_diff, self.upper_diff
        )
    }
}

/// Parse a compare method string to a `PropagationMethod`.
///
/// Compare supports a subset of verification methods: ibp, crown, alpha.
fn parse_compare_method(method: &str) -> PyResult<PropagationMethod> {
    match method {
        "ibp" => Ok(PropagationMethod::Ibp),
        "crown" => Ok(PropagationMethod::Crown),
        "alpha" => Ok(PropagationMethod::AlphaCrown),
        _ => Err(PyValueError::new_err(format!(
            "Unknown method: {}. Use 'ibp', 'crown', or 'alpha'",
            method
        ))),
    }
}

/// Reject output layouts that an element-wise comparison cannot cover fully.
///
/// The comparison loops use `zip`, so checking first is essential: without it,
/// a shorter target output silently truncates the comparison and can make
/// different models appear equivalent. Empty outputs likewise have no
/// meaningful overlap percentage or maximum difference.
fn validate_comparable_outputs(
    reference: &BoundedTensor,
    target: &BoundedTensor,
) -> ny_core::Result<()> {
    if reference.shape() != target.shape() {
        return Err(ny_core::NyError::InvalidSpec(format!(
            "Output shape mismatch: reference {:?}, target {:?}",
            reference.shape(),
            target.shape()
        )));
    }
    if reference.lower().is_empty() {
        return Err(ny_core::NyError::InvalidSpec(
            "Models produced empty outputs; cannot compare bounds".to_string(),
        ));
    }
    Ok(())
}

/// The propagation facade accepts one bounded tensor. Selecting one input from
/// a genuinely multi-input model does not supply values for the remaining
/// inputs, so it cannot be made correct by an `input_index` choice.
fn validate_single_input_models(
    reference_count: usize,
    target_count: usize,
    api_name: &str,
) -> ny_core::Result<()> {
    if reference_count == 0 || target_count == 0 {
        return Err(ny_core::NyError::InvalidSpec(
            "Models must declare exactly one input".to_string(),
        ));
    }
    if reference_count != target_count {
        return Err(ny_core::NyError::InvalidSpec(format!(
            "Input count mismatch: reference has {reference_count}, target has {target_count}"
        )));
    }
    if reference_count != 1 {
        return Err(ny_core::NyError::UnsupportedConfiguration(format!(
            "{api_name} supports exactly one model input; both models declare \
             {reference_count}. Multi-input comparison requires joint bounds for \
             every input and cannot be emulated with input_index."
        )));
    }
    Ok(())
}

/// Prove pointwise equivalence on the shared input box.
///
/// Equal independently-propagated output ranges are only diagnostics: for
/// example, `f(x)=x` and `g(x)=-x` have the same range on a symmetric box. A
/// true equivalence claim therefore comes only from verifying the difference
/// graph `f(x)-g(x)` against `[-tolerance, tolerance]`, with sound provenance.
fn prove_functional_equivalence(
    reference: &Network,
    target: &Network,
    input: &BoundedTensor,
    output_elements: usize,
    tolerance: f32,
    method: PropagationMethod,
    engine: Option<&dyn GemmEngine>,
) -> ny_core::Result<bool> {
    let reference = GraphNetwork::from_sequential(reference)?;
    let target = GraphNetwork::from_sequential(target)?;
    let difference = build_difference_network(&reference, &target)?;
    let input_bounds = input
        .lower()
        .iter()
        .zip(input.upper())
        .map(|(&lower, &upper)| CoreBound::try_new_allow_infinite(lower, upper))
        .collect::<ny_core::Result<Vec<_>>>()?;
    let required = CoreBound::try_new(-tolerance, tolerance)?;
    let output_bounds = vec![required; output_elements];
    let spec = VerificationSpec::from_parts(
        input_bounds,
        output_bounds,
        None,
        Some(input.shape().to_vec()),
    )?;
    let verifier = Verifier::new(PropagationConfig {
        method,
        ..PropagationConfig::default()
    });
    let result = verifier.verify_graph_with_engine(&difference, &spec, engine)?;

    Ok(matches!(
        result,
        VerificationResult::Verified { provenance, .. }
            if provenance.mode() == VerificationSoundnessMode::Sound
    ))
}

/// Compare two models using bound propagation.
///
/// Runs bound propagation on both models with the same input perturbation
/// and compares the resulting output bounds element-wise.
///
/// Args:
///     reference: Path to reference ONNX model
///     target: Path to target ONNX model
///     tolerance: Maximum allowed difference in bounds (default: 0.001)
///     epsilon: Input perturbation radius (default: 0.01)
///     method: Verification method - 'ibp', 'crown', 'alpha' (default: 'crown')
///     input: Optional float32 numpy array for input center (default: zeros).
///         The shape must match the selected model input.
///     input_index: Retained for compatibility; only 0 is valid because compare
///         fails closed for multi-input models.
///     backend: Compute backend - 'auto', 'cpu', or 'wgpu'/'gpu' (default: 'cpu').
///         Only used for CROWN/alpha methods; ignored for IBP. Compare keeps both
///         reference and target propagation work in one call, so CPU stays the
///         conservative default to avoid unnecessary peak GPU memory pressure.
///         Pass `backend="auto"` to opt into GPU probing.
///
/// Notes:
///     compare currently propagates bounds for exactly one model input. Models
///     declaring multiple inputs are rejected until joint multi-input bounds are
///     supported. The endpoint-difference fields remain diagnostics;
///     `is_equivalent` is true only when a sound shared-input difference-network
///     proof establishes pointwise equivalence within tolerance.
///
/// Returns:
///     CompareResult with comparison results
///
/// Example:
///     >>> result = ny.compare("model_pytorch.onnx", "model_coreml.onnx")
///     >>> assert result.is_equivalent, f"Bounds differ: max diff = {result.max_lower_diff:.2e}"
///     >>> # Provide explicit input center
///     >>> # import numpy as np
///     >>> # my_input = np.zeros((1, 3, 224, 224), dtype=np.float32)
///     >>> # result = ny.compare("model_a.onnx", "model_b.onnx", input=my_input)
#[pyfunction]
#[pyo3(signature = (reference, target, tolerance=0.001, epsilon=0.01, method="crown", input=None, input_index=None, backend="cpu"))]
#[allow(clippy::too_many_arguments)] // Python API requires all parameters
pub fn compare(
    py: Python<'_>,
    reference: &str,
    target: &str,
    tolerance: f32,
    epsilon: f32,
    method: &str,
    input: Option<&Bound<'_, PyArrayDyn<f32>>>,
    input_index: Option<usize>,
    backend: &str,
) -> PyResult<CompareResult> {
    validate_epsilon(epsilon)?;
    validate_tolerance(tolerance)?;
    let prop_method = parse_compare_method(method)?;
    let backend = backend.to_string();

    let input_array = match input {
        Some(arr) => {
            let readonly = arr.readonly();
            let owned = readonly.as_array().to_owned();
            validate_input_finite(&owned)?;
            Some(owned)
        }
        None => None,
    };

    let result = Python::detach(py, || {
        // Resolve backend and GemmEngine (reuses verify.rs infrastructure)
        let verify_backend = resolve_verify_backend(&backend, prop_method)?;
        let engine = verify_backend.engine.as_deref();

        // Load both models
        let ref_model = load_onnx(reference)?;
        let target_model = load_onnx(target)?;

        // Convert to propagation networks
        let ref_network = ref_model.to_propagate_network()?;
        let target_network = target_model.to_propagate_network()?;

        let ref_inputs = &ref_model.network.inputs;
        let target_inputs = &target_model.network.inputs;

        validate_single_input_models(ref_inputs.len(), target_inputs.len(), "compare")?;

        let selected_index = match input_index {
            Some(idx) => {
                if idx >= ref_inputs.len() {
                    return Err(ny_core::NyError::InvalidSpec(format!(
                        "input_index {} out of range for {} inputs",
                        idx,
                        ref_inputs.len()
                    )));
                }
                idx
            }
            None => 0,
        };

        let ref_input = &ref_inputs[selected_index];
        let target_input = &target_inputs[selected_index];

        if ref_input.shape.len() != target_input.shape.len() {
            return Err(ny_core::NyError::InvalidSpec(format!(
                "Input shape mismatch at index {}: reference {:?}, target {:?}",
                selected_index, ref_input.shape, target_input.shape
            )));
        }

        for (dim_idx, (ref_dim, target_dim)) in
            ref_input.shape.iter().zip(target_input.shape.iter()).enumerate()
        {
            if *ref_dim > 0 && *target_dim > 0 && ref_dim != target_dim {
                return Err(ny_core::NyError::InvalidSpec(format!(
                    "Input shape mismatch at index {} dim {}: reference {}, target {}",
                    selected_index, dim_idx, ref_dim, target_dim
                )));
            }
        }

        let ref_input_shape: Vec<usize> = ref_input
            .shape
            .iter()
            .map(|&d| d.max(1) as usize)
            .collect();

        let input_center = match input_array {
            Some(arr) => {
                let input_shape = arr.shape();
                if input_shape.len() != ref_input.shape.len() {
                    return Err(ny_core::NyError::InvalidSpec(format!(
                        "Input rank mismatch at index {}: expected {:?}, got {:?}",
                        selected_index, ref_input.shape, input_shape
                    )));
                }
                for (idx, actual) in input_shape.iter().enumerate() {
                    let ref_dim = ref_input.shape[idx];
                    let target_dim = target_input.shape[idx];
                    if ref_dim > 0 && *actual != ref_dim as usize {
                        return Err(ny_core::NyError::InvalidSpec(format!(
                            "Input shape mismatch at index {} dim {}: expected {}, got {}",
                            selected_index, idx, ref_dim, actual
                        )));
                    }
                    if target_dim > 0 && *actual != target_dim as usize {
                        return Err(ny_core::NyError::InvalidSpec(format!(
                            "Input shape mismatch at index {} dim {}: target expects {}, got {}",
                            selected_index, idx, target_dim, actual
                        )));
                    }
                }
                arr
            }
            None => ndarray::ArrayD::from_elem(ndarray::IxDyn(&ref_input_shape), 0.0f32),
        };

        // Create bounded input
        let input = BoundedTensor::from_epsilon(input_center, epsilon)?;

        // Run bound propagation on both models (engine-aware for CROWN/alpha)
        let ref_output = match prop_method {
            PropagationMethod::Ibp => ref_network.propagate_ibp(&input)?,
            PropagationMethod::Crown => {
                ref_network.propagate_crown_with_engine(&input, engine)?
            }
            PropagationMethod::AlphaCrown => {
                ref_network.propagate_alpha_crown_with_engine(&input, engine)?
            }
            _ => {
                return Err(ny_core::NyError::InternalError(format!(
                    "Unsupported method '{:?}' reached propagation (should have been rejected earlier)",
                    prop_method
                )));
            }
        };

        let target_output = match prop_method {
            PropagationMethod::Ibp => target_network.propagate_ibp(&input)?,
            PropagationMethod::Crown => {
                target_network.propagate_crown_with_engine(&input, engine)?
            }
            PropagationMethod::AlphaCrown => {
                target_network.propagate_alpha_crown_with_engine(&input, engine)?
            }
            _ => {
                return Err(ny_core::NyError::InternalError(format!(
                    "Unsupported method '{:?}' reached propagation (should have been rejected earlier)",
                    prop_method
                )));
            }
        };

        validate_comparable_outputs(&ref_output, &target_output)?;

        // Compare outputs
        let ref_lower = ref_output.lower();
        let ref_upper = ref_output.upper();
        let target_lower = target_output.lower();
        let target_upper = target_output.upper();

        let mut max_lower_diff: f32 = 0.0;
        let mut max_upper_diff: f32 = 0.0;
        let mut violations = Vec::new();

        for (idx, (((&rl, &ru), &tl), &tu)) in ref_lower
            .iter()
            .zip(ref_upper.iter())
            .zip(target_lower.iter())
            .zip(target_upper.iter())
            .enumerate()
        {
            let lower_diff = (rl - tl).abs();
            let upper_diff = (ru - tu).abs();

            // nan_propagating_max: unlike f32::max, propagates NaN to surface
            // internal propagation failures instead of absorbing them (#2845).
            max_lower_diff = nan_propagating_max(max_lower_diff, lower_diff);
            max_upper_diff = nan_propagating_max(max_upper_diff, upper_diff);

            if lower_diff > tolerance || upper_diff > tolerance
                || !lower_diff.is_finite() || !upper_diff.is_finite()
            {
                violations.push(BoundViolation {
                    index: idx,
                    ref_lower: rl,
                    ref_upper: ru,
                    target_lower: tl,
                    target_upper: tu,
                    lower_diff,
                    upper_diff,
                });
            }
        }

        // Compute overlap metric
        let mut overlap_count = 0usize;
        let total = ref_lower.len();
        for (((&rl, &ru), &tl), &tu) in ref_lower
            .iter()
            .zip(ref_upper.iter())
            .zip(target_lower.iter())
            .zip(target_upper.iter())
        {
            let overlap = rl.max(tl) <= ru.min(tu);
            if overlap {
                overlap_count += 1;
            }
        }
        let overlap_pct = 100.0 * overlap_count as f32 / total as f32;

        let is_equivalent = prove_functional_equivalence(
            &ref_network,
            &target_network,
            &input,
            ref_output.lower().len(),
            tolerance,
            prop_method,
            engine,
        )?;

        Ok(CompareResult {
            is_equivalent,
            max_lower_diff,
            max_upper_diff,
            tolerance,
            overlap_pct,
            ref_max_width: ref_output.max_width(),
            target_max_width: target_output.max_width(),
            method: method.to_string(),
            epsilon,
            violations,
        })
    })
    .map_err(|e: ny_core::NyError| PyValueError::new_err(format!("Compare error: {}", e)))?;

    Ok(result)
}

/// Compare two models from in-memory ONNX bytes using bound propagation.
///
/// Args:
///     reference_bytes: Reference ONNX model as bytes
///     target_bytes: Target ONNX model as bytes
///     tolerance: Maximum allowed difference in bounds (default: 0.001)
///     epsilon: Input perturbation radius (default: 0.01)
///     method: Verification method - 'ibp', 'crown', 'alpha' (default: 'crown')
///     input: Optional float32 numpy array for input center (default: zeros)
///     input_index: Retained for compatibility; only 0 is valid because
///         multi-input models are rejected.
///     ref_name: Friendly name for reference model (default: "reference")
///     target_name: Friendly name for target model (default: "target")
///     backend: Compute backend - 'auto', 'cpu', or 'wgpu'/'gpu' (default: 'cpu').
///         Only used for CROWN/alpha methods; ignored for IBP. Compare keeps both
///         model propagations in one call, so CPU remains the conservative default;
///         pass `backend="auto"` to opt into GPU probing.
///
/// Notes:
///     Models declaring multiple inputs are rejected. The endpoint-difference
///     fields remain diagnostics; `is_equivalent` is true only when a sound
///     shared-input difference-network proof establishes pointwise equivalence
///     within tolerance.
///
/// Returns:
///     CompareResult with comparison results
#[pyfunction]
#[pyo3(signature = (reference_bytes, target_bytes, tolerance=0.001, epsilon=0.01, method="crown", input=None, input_index=None, ref_name="reference", target_name="target", backend="cpu"))]
// Justification: Python API binding — pyo3 requires all parameters as function arguments.
#[allow(clippy::too_many_arguments)]
pub fn compare_bytes(
    py: Python<'_>,
    reference_bytes: Vec<u8>,
    target_bytes: Vec<u8>,
    tolerance: f32,
    epsilon: f32,
    method: &str,
    input: Option<&Bound<'_, PyArrayDyn<f32>>>,
    input_index: Option<usize>,
    ref_name: &str,
    target_name: &str,
    backend: &str,
) -> PyResult<CompareResult> {
    validate_epsilon(epsilon)?;
    validate_tolerance(tolerance)?;
    let prop_method = parse_compare_method(method)?;
    let backend = backend.to_string();

    let input_array = match input {
        Some(arr) => {
            let readonly = arr.readonly();
            let owned = readonly.as_array().to_owned();
            validate_input_finite(&owned)?;
            Some(owned)
        }
        None => None,
    };

    let ref_name = ref_name.to_string();
    let target_name = target_name.to_string();

    let result = Python::detach(py, || {
        let verify_backend = resolve_verify_backend(&backend, prop_method)?;
        let engine = verify_backend.engine.as_deref();

        let ref_model = load_onnx_bytes(&ref_name, &reference_bytes)?;
        let target_model = load_onnx_bytes(&target_name, &target_bytes)?;

        let ref_network = ref_model.to_propagate_network()?;
        let target_network = target_model.to_propagate_network()?;

        let ref_inputs = &ref_model.network.inputs;
        let target_inputs = &target_model.network.inputs;

        validate_single_input_models(ref_inputs.len(), target_inputs.len(), "compare_bytes")?;

        let selected_index = match input_index {
            Some(idx) => {
                if idx >= ref_inputs.len() {
                    return Err(ny_core::NyError::InvalidSpec(format!(
                        "input_index {} out of range for {} inputs",
                        idx,
                        ref_inputs.len()
                    )));
                }
                idx
            }
            None => 0,
        };

        let ref_input = &ref_inputs[selected_index];
        let target_input = &target_inputs[selected_index];

        if ref_input.shape.len() != target_input.shape.len() {
            return Err(ny_core::NyError::InvalidSpec(format!(
                "Input shape mismatch at index {}: reference {:?}, target {:?}",
                selected_index, ref_input.shape, target_input.shape
            )));
        }
        for (dim_idx, (ref_dim, target_dim)) in
            ref_input.shape.iter().zip(target_input.shape.iter()).enumerate()
        {
            if *ref_dim > 0 && *target_dim > 0 && ref_dim != target_dim {
                return Err(ny_core::NyError::InvalidSpec(format!(
                    "Input shape mismatch at index {} dim {}: reference {}, target {}",
                    selected_index, dim_idx, ref_dim, target_dim
                )));
            }
        }

        let ref_input_shape: Vec<usize> = ref_input
            .shape
            .iter()
            .map(|&d| d.max(1) as usize)
            .collect();

        let input_center = match input_array {
            Some(arr) => {
                let input_shape = arr.shape();
                if input_shape.len() != ref_input.shape.len() {
                    return Err(ny_core::NyError::InvalidSpec(format!(
                        "Input rank mismatch at index {}: expected {:?}, got {:?}",
                        selected_index, ref_input.shape, input_shape
                    )));
                }
                for (idx, actual) in input_shape.iter().enumerate() {
                    let ref_dim = ref_input.shape[idx];
                    let target_dim = target_input.shape[idx];
                    if ref_dim > 0 && *actual != ref_dim as usize {
                        return Err(ny_core::NyError::InvalidSpec(format!(
                            "Input shape mismatch at index {} dim {}: expected {}, got {}",
                            selected_index, idx, ref_dim, actual
                        )));
                    }
                    if target_dim > 0 && *actual != target_dim as usize {
                        return Err(ny_core::NyError::InvalidSpec(format!(
                            "Input shape mismatch at index {} dim {}: target expects {}, got {}",
                            selected_index, idx, target_dim, actual
                        )));
                    }
                }
                arr
            }
            None => ndarray::ArrayD::from_elem(ndarray::IxDyn(&ref_input_shape), 0.0f32),
        };

        let input = BoundedTensor::from_epsilon(input_center, epsilon)?;

        let ref_output = match prop_method {
            PropagationMethod::Ibp => ref_network.propagate_ibp(&input)?,
            PropagationMethod::Crown => {
                ref_network.propagate_crown_with_engine(&input, engine)?
            }
            PropagationMethod::AlphaCrown => {
                ref_network.propagate_alpha_crown_with_engine(&input, engine)?
            }
            _ => {
                return Err(ny_core::NyError::InternalError(format!(
                    "Unsupported method '{:?}' reached propagation (should have been rejected earlier)",
                    prop_method
                )));
            }
        };

        let target_output = match prop_method {
            PropagationMethod::Ibp => target_network.propagate_ibp(&input)?,
            PropagationMethod::Crown => {
                target_network.propagate_crown_with_engine(&input, engine)?
            }
            PropagationMethod::AlphaCrown => {
                target_network.propagate_alpha_crown_with_engine(&input, engine)?
            }
            _ => {
                return Err(ny_core::NyError::InternalError(format!(
                    "Unsupported method '{:?}' reached propagation (should have been rejected earlier)",
                    prop_method
                )));
            }
        };

        validate_comparable_outputs(&ref_output, &target_output)?;

        let ref_lower = ref_output.lower();
        let ref_upper = ref_output.upper();
        let target_lower = target_output.lower();
        let target_upper = target_output.upper();

        let mut max_lower_diff: f32 = 0.0;
        let mut max_upper_diff: f32 = 0.0;
        let mut violations = Vec::new();

        for (idx, (((&rl, &ru), &tl), &tu)) in ref_lower
            .iter()
            .zip(ref_upper.iter())
            .zip(target_lower.iter())
            .zip(target_upper.iter())
            .enumerate()
        {
            let lower_diff = (rl - tl).abs();
            let upper_diff = (ru - tu).abs();

            // nan_propagating_max: unlike f32::max, propagates NaN to surface
            // internal propagation failures instead of absorbing them (#2845).
            max_lower_diff = nan_propagating_max(max_lower_diff, lower_diff);
            max_upper_diff = nan_propagating_max(max_upper_diff, upper_diff);

            if lower_diff > tolerance || upper_diff > tolerance
                || !lower_diff.is_finite() || !upper_diff.is_finite()
            {
                violations.push(BoundViolation {
                    index: idx,
                    ref_lower: rl,
                    ref_upper: ru,
                    target_lower: tl,
                    target_upper: tu,
                    lower_diff,
                    upper_diff,
                });
            }
        }

        let mut overlap_count = 0usize;
        let total = ref_lower.len();
        for (((&rl, &ru), &tl), &tu) in ref_lower
            .iter()
            .zip(ref_upper.iter())
            .zip(target_lower.iter())
            .zip(target_upper.iter())
        {
            if rl.max(tl) <= ru.min(tu) {
                overlap_count += 1;
            }
        }
        let overlap_pct = 100.0 * overlap_count as f32 / total as f32;

        let is_equivalent = prove_functional_equivalence(
            &ref_network,
            &target_network,
            &input,
            ref_output.lower().len(),
            tolerance,
            prop_method,
            engine,
        )?;

        Ok(CompareResult {
            is_equivalent,
            max_lower_diff,
            max_upper_diff,
            tolerance,
            overlap_pct,
            ref_max_width: ref_output.max_width(),
            target_max_width: target_output.max_width(),
            method: method.to_string(),
            epsilon,
            violations,
        })
    })
    .map_err(|e: ny_core::NyError| PyValueError::new_err(format!("Compare error: {}", e)))?;

    Ok(result)
}

/// Compare two PyTorch models using bound propagation without writing to disk.
///
/// Args:
///     reference: Reference PyTorch model (nn.Module)
///     target: Target PyTorch model (nn.Module)
///     example_input: Example input tensor for ONNX tracing
///     tolerance: Maximum allowed difference in bounds (default: 0.001)
///     epsilon: Input perturbation radius (default: 0.01)
///     method: Verification method - 'ibp', 'crown', 'alpha' (default: 'crown')
///     input: Optional float32 numpy array for input center (default: zeros)
///     input_index: Retained for compatibility; only 0 is valid because
///         multi-input models are rejected.
///     opset: ONNX opset version for export (default: 17)
///     backend: Compute backend - 'auto', 'cpu', or 'wgpu'/'gpu' (default: 'cpu').
///         Only used for CROWN/alpha methods; ignored for IBP. Compare defaults to
///         CPU because the dual-model path is more memory-hungry; pass
///         `backend="auto"` to opt into GPU probing.
///
/// Notes:
///     Models declaring multiple inputs are rejected. `is_equivalent` is true
///     only when a sound shared-input difference-network proof establishes
///     pointwise equivalence within tolerance.
///
/// Returns:
///     CompareResult with comparison results
#[pyfunction]
#[pyo3(signature = (reference, target, example_input, tolerance=0.001, epsilon=0.01, method="crown", input=None, input_index=None, opset=17, backend="cpu"))]
// Justification: Python API binding — pyo3 requires all parameters as function arguments.
#[allow(clippy::too_many_arguments)]
pub fn compare_torch(
    py: Python<'_>,
    reference: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
    example_input: &Bound<'_, PyAny>,
    tolerance: f32,
    epsilon: f32,
    method: &str,
    input: Option<&Bound<'_, PyArrayDyn<f32>>>,
    input_index: Option<usize>,
    opset: u32,
    backend: &str,
) -> PyResult<CompareResult> {
    let ref_bytes =
        export_torch_to_onnx_bytes(py, reference, example_input, opset, "compare_torch")?;
    let target_bytes =
        export_torch_to_onnx_bytes(py, target, example_input, opset, "compare_torch")?;

    compare_bytes(
        py,
        ref_bytes,
        target_bytes,
        tolerance,
        epsilon,
        method,
        input,
        input_index,
        "torch_reference",
        "torch_target",
        backend,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        nan_propagating_max, parse_compare_method, prove_functional_equivalence,
        validate_comparable_outputs, validate_single_input_models,
    };
    use ndarray::{arr1, arr2, ArrayD, IxDyn};
    use ny_propagate::layers::{Layer, LinearLayer};
    use ny_propagate::{Network, PropagationMethod};
    use ny_tensor::BoundedTensor;

    fn zero_bounds(shape: &[usize]) -> BoundedTensor {
        let values = ArrayD::zeros(IxDyn(shape));
        BoundedTensor::new(values.clone(), values).expect("valid zero-width bounds")
    }

    fn scalar_linear(weight: f32) -> Network {
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[weight]]), None).expect("valid scalar linear layer"),
        ));
        network
    }

    // Regression tests for nan_propagating_max (#2845, #2898).
    // This replaced unstable f32::maximum. If refactored to f32::max,
    // the NaN-absorption tests below will catch the regression.

    #[test]
    fn test_nan_propagating_max_normal_values() {
        assert_eq!(nan_propagating_max(3.0, 5.0), 5.0);
        assert_eq!(nan_propagating_max(5.0, 3.0), 5.0);
        assert_eq!(nan_propagating_max(-1.0, -2.0), -1.0);
    }

    #[test]
    fn test_nan_propagating_max_equal_values() {
        assert_eq!(nan_propagating_max(4.0, 4.0), 4.0);
        assert_eq!(nan_propagating_max(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_nan_propagating_max_nan_first_arg() {
        // CRITICAL: f32::max would return 5.0 here (absorbing NaN).
        // nan_propagating_max must return NaN to surface propagation failures.
        assert!(nan_propagating_max(f32::NAN, 5.0).is_nan());
    }

    #[test]
    fn test_nan_propagating_max_nan_second_arg() {
        assert!(nan_propagating_max(5.0, f32::NAN).is_nan());
    }

    #[test]
    fn test_nan_propagating_max_both_nan() {
        assert!(nan_propagating_max(f32::NAN, f32::NAN).is_nan());
    }

    #[test]
    fn test_nan_propagating_max_infinity() {
        assert_eq!(nan_propagating_max(f32::INFINITY, 5.0), f32::INFINITY);
        assert_eq!(nan_propagating_max(5.0, f32::NEG_INFINITY), 5.0);
    }

    #[test]
    fn test_nan_propagating_max_zero_initial_accumulator() {
        // Simulates the accumulation pattern: starts at 0.0, encounters NaN diff.
        let mut acc = 0.0_f32;
        acc = nan_propagating_max(acc, 0.5);
        assert_eq!(acc, 0.5);
        acc = nan_propagating_max(acc, f32::NAN);
        assert!(acc.is_nan());
        // Once NaN, stays NaN (the whole point of propagation).
        acc = nan_propagating_max(acc, 100.0);
        assert!(acc.is_nan());
    }

    #[test]
    fn comparable_outputs_reject_shape_mismatch_and_empty_outputs() {
        let two = zero_bounds(&[2]);
        let one = zero_bounds(&[1]);
        assert!(validate_comparable_outputs(&two, &one).is_err());

        let empty = zero_bounds(&[0]);
        assert!(validate_comparable_outputs(&empty, &empty).is_err());
        assert!(validate_comparable_outputs(&two, &two).is_ok());
    }

    #[test]
    fn single_tensor_facade_rejects_multi_input_models() {
        assert!(validate_single_input_models(1, 1, "compare").is_ok());
        assert!(validate_single_input_models(0, 0, "compare").is_err());
        assert!(validate_single_input_models(1, 2, "compare").is_err());
        let error = validate_single_input_models(2, 2, "compare")
            .expect_err("selecting one input cannot supply the other input");
        assert!(error.to_string().contains("exactly one model input"));
    }

    #[test]
    fn equal_independent_ranges_do_not_prove_functional_equivalence() {
        // x and -x both have output range [-1, 1] on this box, but differ at
        // every non-zero point. A range-endpoint comparison would claim a
        // perfect match; the shared-input difference graph must not.
        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
            .expect("valid input box");
        let equivalent = prove_functional_equivalence(
            &scalar_linear(1.0),
            &scalar_linear(-1.0),
            &input,
            1,
            0.1,
            PropagationMethod::Ibp,
            None,
        )
        .expect("difference verification");
        assert!(!equivalent);
    }

    #[test]
    fn sound_difference_proof_can_establish_box_equivalence() {
        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
            .expect("valid input box");
        let equivalent = prove_functional_equivalence(
            &scalar_linear(1.0),
            &scalar_linear(1.0),
            &input,
            1,
            1e-6,
            PropagationMethod::Crown,
            None,
        )
        .expect("difference verification");
        assert!(equivalent);
    }

    // Regression tests for #3622: compare backend routing.
    // Ensures parse_compare_method routes correctly and that
    // CROWN/alpha methods would receive an engine while IBP does not.

    #[test]
    fn test_parse_compare_method_ibp() {
        let method = parse_compare_method("ibp").expect("ibp should parse");
        assert_eq!(method, PropagationMethod::Ibp);
    }

    #[test]
    fn test_parse_compare_method_crown() {
        let method = parse_compare_method("crown").expect("crown should parse");
        assert_eq!(method, PropagationMethod::Crown);
    }

    #[test]
    fn test_parse_compare_method_alpha() {
        let method = parse_compare_method("alpha").expect("alpha should parse");
        assert_eq!(method, PropagationMethod::AlphaCrown);
    }

    #[test]
    fn test_parse_compare_method_unknown_rejected() {
        assert!(parse_compare_method("beta").is_err());
        assert!(parse_compare_method("sdp").is_err());
        assert!(parse_compare_method("").is_err());
    }

    #[test]
    fn test_resolve_verify_backend_cpu_returns_no_engine() {
        use crate::verify::resolve_verify_backend;

        let backend = resolve_verify_backend("cpu", PropagationMethod::Crown)
            .expect("cpu backend should resolve");
        assert!(
            backend.engine.is_none(),
            "CPU backend should not create a GemmEngine"
        );
    }

    #[test]
    fn test_resolve_verify_backend_ibp_ignores_backend() {
        use crate::verify::resolve_verify_backend;

        // Even with a non-CPU backend string, IBP should not create an engine
        // because IBP does not use GEMM-based propagation.
        let backend = resolve_verify_backend("cpu", PropagationMethod::Ibp)
            .expect("ibp backend should resolve");
        assert!(
            backend.engine.is_none(),
            "IBP should not create a GemmEngine regardless of backend"
        );
    }
}
