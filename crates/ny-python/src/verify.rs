// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::beta_config::BetaCrownConfig;
use crate::repr::repr_string;
use crate::utils::{export_torch_to_onnx_bytes, validate_epsilon};
use ny_core::{
    checked_shape_product, nan_propagating_max, nan_propagating_min, Bound as RustBound,
    GemmEngine, HeuristicUsed as RustHeuristicUsed, SoundnessProvenance as RustSoundnessProvenance,
    VerificationResult as RustVerificationResult, VerificationSoundnessMode, VerificationSpec,
};
use ny_gpu::{Backend, ComputeDevice};
use ny_onnx::{load_onnx, load_onnx_bytes};
use ny_propagate::{
    soundness::{count_sqrt_negative_domain_network, soundness_provenance_for_network},
    BetaCrownVerifier, MulBinaryRelaxationMode, PropagationConfig, PropagationMethod, Verifier,
};
use ny_tensor::BoundedTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

/// A single output bound (lower, upper).
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct OutputBound {
    #[pyo3(get)]
    pub lower: f32,
    #[pyo3(get)]
    pub upper: f32,
}

#[pymethods]
impl OutputBound {
    pub(crate) fn __repr__(&self) -> String {
        format!(
            "OutputBound(lower={:.6}, upper={:.6})",
            self.lower, self.upper
        )
    }

    /// Width of the bound interval.
    #[getter]
    pub(crate) fn width(&self) -> f32 {
        self.upper - self.lower
    }

    /// Midpoint of the bound.
    #[getter]
    pub(crate) fn midpoint(&self) -> f32 {
        f32::midpoint(self.lower, self.upper)
    }
}

impl From<RustBound> for OutputBound {
    fn from(b: RustBound) -> Self {
        OutputBound {
            lower: b.lower(),
            upper: b.upper(),
        }
    }
}

/// Status of a verification result.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum VerifyStatus {
    /// Property verified: all outputs within bounds for all inputs in region.
    Verified,
    /// Property violated: counterexample found.
    Violated,
    /// Verification inconclusive: bounds too loose, or no property was specified.
    Unknown,
    /// Verification timed out.
    Timeout,
}

#[pymethods]
impl VerifyStatus {
    pub(crate) fn __repr__(&self) -> String {
        match self {
            VerifyStatus::Verified => "VerifyStatus.Verified".to_string(),
            VerifyStatus::Violated => "VerifyStatus.Violated".to_string(),
            VerifyStatus::Unknown => "VerifyStatus.Unknown".to_string(),
            VerifyStatus::Timeout => "VerifyStatus.Timeout".to_string(),
        }
    }

    pub(crate) fn __str__(&self) -> String {
        match self {
            VerifyStatus::Verified => "VERIFIED".to_string(),
            VerifyStatus::Violated => "VIOLATED".to_string(),
            VerifyStatus::Unknown => "UNKNOWN".to_string(),
            VerifyStatus::Timeout => "TIMEOUT".to_string(),
        }
    }
}

/// A heuristic/approximation used by a verification run.
///
/// This mirrors `ny_core::HeuristicUsed` but is represented as a simple Python object.
#[pyclass(name = "HeuristicUsed", from_py_object)]
#[derive(Clone)]
pub struct HeuristicUsed {
    /// Type tag (snake_case), e.g. `instancenorm_forward_mode`.
    #[pyo3(get, name = "type")]
    pub type_: String,

    /// Optional number of nodes affected.
    #[pyo3(get)]
    pub num_nodes: Option<usize>,
}

impl From<RustHeuristicUsed> for HeuristicUsed {
    fn from(h: RustHeuristicUsed) -> Self {
        match h {
            RustHeuristicUsed::LayerNormForwardMode { num_nodes } => Self {
                type_: "layernorm_forward_mode".to_string(),
                num_nodes: Some(num_nodes),
            },
            RustHeuristicUsed::RmsNormForwardMode { num_nodes } => Self {
                type_: "rmsnorm_forward_mode".to_string(),
                num_nodes: Some(num_nodes),
            },
            RustHeuristicUsed::GroupNormForwardMode { num_nodes } => Self {
                type_: "groupnorm_forward_mode".to_string(),
                num_nodes: Some(num_nodes),
            },
            RustHeuristicUsed::InstanceNormForwardMode { num_nodes } => Self {
                type_: "instancenorm_forward_mode".to_string(),
                num_nodes: Some(num_nodes),
            },
            RustHeuristicUsed::AdaInForwardMode { num_nodes } => Self {
                type_: "adain_forward_mode".to_string(),
                num_nodes: Some(num_nodes),
            },
            RustHeuristicUsed::LayerNormCrownSampling { num_nodes } => Self {
                type_: "layernorm_crown_sampling".to_string(),
                num_nodes: Some(num_nodes),
            },
            RustHeuristicUsed::SoftmaxCrownSampling { num_nodes } => Self {
                type_: "softmax_crown_sampling".to_string(),
                num_nodes: Some(num_nodes),
            },
            RustHeuristicUsed::CausalSoftmaxCrownSampling { num_nodes } => Self {
                type_: "causal_softmax_crown_sampling".to_string(),
                num_nodes: Some(num_nodes),
            },
            RustHeuristicUsed::LogSoftmaxCrownSampling { num_nodes } => Self {
                type_: "logsoftmax_crown_sampling".to_string(),
                num_nodes: Some(num_nodes),
            },
            RustHeuristicUsed::SamplingBasedNonlinearRelaxations => Self {
                type_: "sampling_based_nonlinear_relaxations".to_string(),
                num_nodes: None,
            },
            RustHeuristicUsed::SqrtNegativeDomain { num_nodes } => Self {
                type_: "sqrt_negative_domain".to_string(),
                num_nodes: Some(num_nodes),
            },
            RustHeuristicUsed::ContinuousComparisonApproximation { num_nodes } => Self {
                type_: "continuous_comparison_approximation".to_string(),
                num_nodes: Some(num_nodes),
            },
            RustHeuristicUsed::ReduceExtremumFixedIndex { num_nodes } => Self {
                type_: "reduce_extremum_fixed_index".to_string(),
                num_nodes: Some(num_nodes),
            },
        }
    }
}

#[pymethods]
impl HeuristicUsed {
    pub(crate) fn __repr__(&self) -> String {
        match self.num_nodes {
            Some(n) => format!("HeuristicUsed(type='{}', num_nodes={})", self.type_, n),
            None => format!("HeuristicUsed(type='{}')", self.type_),
        }
    }
}

/// Machine-readable provenance for verification soundness semantics.
#[pyclass(name = "SoundnessProvenance", from_py_object)]
#[derive(Clone)]
pub struct SoundnessProvenance {
    #[pyo3(get)]
    pub mode: String,

    #[pyo3(get)]
    pub heuristics_used: Vec<HeuristicUsed>,
}

impl From<RustSoundnessProvenance> for SoundnessProvenance {
    fn from(p: RustSoundnessProvenance) -> Self {
        let mode = match p.mode() {
            VerificationSoundnessMode::Sound => "sound",
            VerificationSoundnessMode::Heuristic => "heuristic",
        }
        .to_string();
        Self {
            mode,
            heuristics_used: p
                .heuristics_used()
                .iter()
                .cloned()
                .map(Into::into)
                .collect(),
        }
    }
}

#[pymethods]
impl SoundnessProvenance {
    pub(crate) fn __repr__(&self) -> String {
        format!(
            "SoundnessProvenance(mode='{}', heuristics_used={})",
            self.mode,
            self.heuristics_used.len()
        )
    }
}

/// Result of neural network verification.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct VerifyResult {
    #[pyo3(get)]
    pub status: VerifyStatus,

    #[pyo3(get)]
    pub soundness: SoundnessProvenance,

    #[pyo3(get)]
    pub output_bounds: Option<Vec<OutputBound>>,

    #[pyo3(get)]
    pub counterexample: Option<Vec<f32>>,

    #[pyo3(get)]
    pub counterexample_output: Option<Vec<f32>>,

    #[pyo3(get)]
    pub reason: Option<String>,

    #[pyo3(get)]
    pub method: String,

    #[pyo3(get)]
    pub actual_method: Option<String>,

    #[pyo3(get)]
    pub epsilon: f32,
}

#[pymethods]
impl VerifyResult {
    pub(crate) fn __repr__(&self) -> String {
        match self.actual_method.as_deref() {
            Some(actual) if actual != self.method => format!(
                "VerifyResult(status={}, method={}, actual_method={}, epsilon={:.2e})",
                self.status.__str__(),
                repr_string(&self.method),
                repr_string(actual),
                self.epsilon
            ),
            _ => format!(
                "VerifyResult(status={}, method={}, epsilon={:.2e})",
                self.status.__str__(),
                repr_string(&self.method),
                self.epsilon
            ),
        }
    }

    /// Check if the property was verified.
    #[getter]
    pub(crate) fn is_verified(&self) -> bool {
        matches!(self.status, VerifyStatus::Verified)
    }

    /// Check if a violation was found.
    #[getter]
    pub(crate) fn is_violated(&self) -> bool {
        matches!(self.status, VerifyStatus::Violated)
    }

    /// Get max output bound width (for diagnostics).
    pub(crate) fn max_output_width(&self) -> Option<f32> {
        self.output_bounds.as_ref().map(|bounds| {
            bounds
                .iter()
                .map(|b| b.width())
                .fold(f32::NEG_INFINITY, nan_propagating_max)
        })
    }

    /// Get formatted summary.
    pub(crate) fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Verification Result".to_string());
        lines.push("===================".to_string());
        lines.push(format!("Status:  {}", self.status.__str__()));
        lines.push(format!("Method:  {}", self.method));
        lines.push(format!("Epsilon: {:.2e}", self.epsilon));

        if let Some(ref bounds) = self.output_bounds {
            let max_width = bounds
                .iter()
                .map(|b| b.width())
                .fold(0.0_f32, nan_propagating_max);
            let mean_width: f32 =
                bounds.iter().map(|b| b.width()).sum::<f32>() / bounds.len() as f32;
            lines.push(format!("Outputs: {} bounds", bounds.len()));
            lines.push(format!("Max width:  {:.2e}", max_width));
            lines.push(format!("Mean width: {:.2e}", mean_width));
        }

        if let Some(ref reason) = self.reason {
            lines.push(format!("Reason: {}", reason));
        }

        lines.join("\n")
    }
}

fn parse_mul_binary_relaxation(mode: &str) -> PyResult<MulBinaryRelaxationMode> {
    match mode {
        "mccormick" => Ok(MulBinaryRelaxationMode::McCormick),
        "middle" => Ok(MulBinaryRelaxationMode::Middle),
        _ => Err(PyValueError::new_err(format!(
            "Unknown mul_binary_relaxation: {}. Use 'mccormick' or 'middle'.",
            mode
        ))),
    }
}

/// Derive BaB threshold from spec output bounds (#3229).
///
/// Same pattern as `Verifier::verify_beta_crown` in
/// `ny-propagate/src/verifier/network.rs:183-195`.
///
/// Returns `min(all finite lower bounds)` from the output specification.
/// If all lower bounds are `-inf` (no finite lower constraints), returns
/// `-inf` so BaB trivially verifies the lower-bound direction.
///
/// Note: `Bound` rejects NaN at construction, so the `.is_finite()` filter
/// only distinguishes finite from infinite (defensive against future changes).
pub(crate) fn derive_bab_threshold(output_bounds: &[RustBound]) -> f32 {
    let from_spec = output_bounds
        .iter()
        .map(|b| b.lower())
        .filter(|l| l.is_finite())
        .fold(f32::INFINITY, nan_propagating_min);
    if from_spec == f32::INFINITY {
        f32::NEG_INFINITY
    } else {
        from_spec
    }
}

/// Reason attached to results computed without an output specification.
pub(crate) const NO_OUTPUT_SPEC_REASON: &str =
    "no output specification provided: certified output bounds were computed but no property \
     was checked; pass output_bounds=[(lower, upper), ...] to verify a property";

/// Convert user-facing (lower, upper) pairs into validated spec bounds.
///
/// Endpoints may be infinite for one-sided constraints, but the spec as a
/// whole must constrain something: an all-(-inf, +inf) requirement is
/// satisfied by every network, so it is rejected rather than reported as
/// `Verified` — the same triviality class as the empty-spec guard (#2266).
pub(crate) fn build_output_spec_bounds(
    bounds: &[(f32, f32)],
    output_dim: usize,
) -> ny_core::Result<Vec<RustBound>> {
    if bounds.len() != output_dim {
        return Err(ny_core::NyError::InvalidSpec(format!(
            "output_bounds has {} entries but the model has {} outputs",
            bounds.len(),
            output_dim
        )));
    }
    let mut has_finite_constraint = false;
    let converted = bounds
        .iter()
        .enumerate()
        .map(|(i, &(lower, upper))| {
            if lower.is_nan() || upper.is_nan() {
                return Err(ny_core::NyError::InvalidSpec(format!(
                    "output_bounds[{i}] contains NaN: [{lower}, {upper}]"
                )));
            }
            if lower > upper {
                return Err(ny_core::NyError::InvalidSpec(format!(
                    "output_bounds[{i}] is malformed: lower {lower} > upper {upper}"
                )));
            }
            has_finite_constraint |= lower.is_finite() || upper.is_finite();
            Ok(RustBound::new_allow_infinite(lower, upper))
        })
        .collect::<ny_core::Result<Vec<_>>>()?;
    if !has_finite_constraint {
        return Err(ny_core::NyError::InvalidSpec(
            "output_bounds are unconstrained (every bound is (-inf, +inf)) — nothing to verify"
                .to_string(),
        ));
    }
    Ok(converted)
}

/// Fold property verdicts for a run that carried no output specification.
///
/// The unconstrained (-inf, +inf) requirement is satisfied by any sound
/// bounds, so `Verified` would only state a vacuous property — and with no
/// requirement there is nothing for a counterexample to violate. Both fold
/// into `Unknown`, keeping the computed bounds; `Unknown` and `Timeout` pass
/// through unchanged.
pub(crate) fn fold_unspecified_property(result: RustVerificationResult) -> RustVerificationResult {
    match result {
        RustVerificationResult::Verified {
            provenance,
            output_bounds,
            actual_method,
            ..
        } => RustVerificationResult::Unknown {
            provenance,
            bounds: output_bounds,
            reason: ny_core::UnknownReason::Other {
                message: NO_OUTPUT_SPEC_REASON.to_string(),
            },
            actual_method,
        },
        RustVerificationResult::Violated {
            provenance,
            actual_method,
            ..
        } => RustVerificationResult::Unknown {
            provenance,
            bounds: Vec::new(),
            reason: ny_core::UnknownReason::Other {
                message: NO_OUTPUT_SPEC_REASON.to_string(),
            },
            actual_method,
        },
        other => other,
    }
}

/// Per-output spec check for a BaB `Verified` verdict (#2241).
///
/// BaB proves `min(all outputs) >= threshold`, where `threshold` is the
/// global minimum of the finite required lower bounds, so a `Verified` status
/// alone does not establish each output's own requirement (per-output lowers
/// can be tighter, and uppers are not covered at all). Checks every output's
/// computed bound — falling back to `[threshold, +inf)`, the bound the BaB
/// verdict itself justifies, where BaB produced no tensor bound — against its
/// requirement. Returns the first positive gap, or `None` when every
/// requirement is met. Same rule as `Verifier::finalize_beta_crown_result`.
pub(crate) fn bab_verified_spec_gap(
    required: &[RustBound],
    computed: Option<&[RustBound]>,
    threshold: f32,
) -> Option<f32> {
    let fallback = RustBound::new_allow_infinite(threshold, f32::INFINITY);
    for (idx, req) in required.iter().enumerate() {
        let bound = computed
            .and_then(|bounds| bounds.get(idx))
            .copied()
            .unwrap_or(fallback);
        let lower_gap = if bound.lower() < req.lower() {
            req.lower() - bound.lower()
        } else {
            0.0
        };
        let upper_gap = if bound.upper() > req.upper() {
            bound.upper() - req.upper()
        } else {
            0.0
        };
        let gap = lower_gap.max(upper_gap);
        if gap > 0.0 {
            return Some(gap);
        }
    }
    None
}

/// Concretely re-evaluate a BaB counterexample against per-output requirements.
///
/// BaB counterexamples come from f32 bound arithmetic that is not trusted
/// blindly: the candidate must lie inside the input box and its concrete
/// forward evaluation must violate at least one output requirement. Returns
/// the concrete output when the violation is confirmed, `None` when the
/// candidate fails validation (callers downgrade to Unknown). Mirrors
/// `Verifier::validate_beta_crown_counterexample`.
fn validate_bab_counterexample(
    network: &ny_propagate::Network,
    input: &BoundedTensor,
    required: &[RustBound],
    counterexample: &[f32],
) -> ny_core::Result<Option<Vec<f32>>> {
    const TOLERANCE: f32 = 1e-6;

    if counterexample.len() != input.len() {
        return Ok(None);
    }
    for (&value, (&lower, &upper)) in counterexample
        .iter()
        .zip(input.lower().iter().zip(input.upper().iter()))
    {
        if !value.is_finite() || value < lower - TOLERANCE || value > upper + TOLERANCE {
            return Ok(None);
        }
    }

    let candidate =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(input.shape()), counterexample.to_vec())
            .map_err(|e| ny_core::NyError::InvalidSpec(e.to_string()))?;
    let concrete = BoundedTensor::concrete(candidate)?;
    let output = network.propagate_ibp(&concrete)?;
    let actual_output: Vec<f32> = output.lower().iter().copied().collect();

    if actual_output.len() < required.len() || actual_output.iter().any(|value| !value.is_finite())
    {
        return Ok(None);
    }

    let violates = actual_output
        .iter()
        .zip(required.iter())
        .any(|(&value, req)| value < req.lower() || value > req.upper());
    Ok(violates.then_some(actual_output))
}

#[derive(Clone)]
pub(crate) struct VerifyBackend {
    pub(crate) use_gpu: bool,
    pub(crate) engine: Option<Arc<dyn GemmEngine>>,
}

fn method_uses_gemm_engine(method: PropagationMethod) -> bool {
    matches!(
        method,
        PropagationMethod::Crown | PropagationMethod::AlphaCrown | PropagationMethod::BetaCrown
    )
}

fn cpu_verify_backend() -> VerifyBackend {
    VerifyBackend {
        use_gpu: false,
        engine: None,
    }
}

fn auto_verify_backend_candidates() -> [Backend; 1] {
    // wgpu is the sole GPU backend with full GEMM + CROWN backward support.
    [Backend::Wgpu]
}

pub(crate) fn resolve_verify_backend_with_factory<F>(
    backend: &str,
    method: PropagationMethod,
    mut build_engine: F,
) -> ny_core::Result<VerifyBackend>
where
    F: FnMut(Backend) -> ny_core::Result<Arc<dyn GemmEngine>>,
{
    if !backend.eq_ignore_ascii_case("auto") {
        let backend = backend.parse::<Backend>().map_err(|_| {
            ny_core::NyError::InvalidSpec(format!(
                "Unknown backend: {backend}. Valid options: auto, cpu, wgpu"
            ))
        })?;
        if !method_uses_gemm_engine(method) || backend == Backend::Cpu {
            return Ok(cpu_verify_backend());
        }

        let engine = build_engine(backend)?;
        return Ok(VerifyBackend {
            use_gpu: true,
            engine: Some(engine),
        });
    }

    if !method_uses_gemm_engine(method) {
        return Ok(cpu_verify_backend());
    }

    for candidate in auto_verify_backend_candidates() {
        if let Ok(engine) = build_engine(candidate) {
            return Ok(VerifyBackend {
                use_gpu: true,
                engine: Some(engine),
            });
        }
    }
    Ok(cpu_verify_backend())
}

pub(crate) fn resolve_verify_backend(
    backend: &str,
    method: PropagationMethod,
) -> ny_core::Result<VerifyBackend> {
    resolve_verify_backend_with_factory(backend, method, |backend| {
        Ok(Arc::new(ComputeDevice::new(backend)?))
    })
}

pub(crate) fn build_standard_verifier(
    config: PropagationConfig,
    engine: Option<Arc<dyn GemmEngine>>,
) -> Verifier {
    match engine {
        Some(engine) => Verifier::new_with_engine(config, engine),
        None => Verifier::new(config),
    }
}

pub(crate) fn build_beta_crown_verifier(
    config: ny_propagate::BetaCrownConfig,
    engine: Option<Arc<dyn GemmEngine>>,
) -> BetaCrownVerifier {
    match engine {
        Some(engine) => BetaCrownVerifier::new_with_engine(config, engine),
        None => BetaCrownVerifier::new(config),
    }
}

fn resolve_verification_dim(
    dim: i64,
    dynamic_sub: usize,
    tensor_role: &str,
) -> ny_core::Result<(usize, bool)> {
    match dim {
        -1 | 0 => Ok((dynamic_sub, true)),
        value if value < 0 => Err(ny_core::NyError::InvalidSpec(format!(
            "Unsupported negative ONNX {tensor_role} dimension {value}; only -1 is permitted among negative dimensions"
        ))),
        value => Ok((value as usize, false)),
    }
}

fn resolve_verification_shape(
    shape: &[i64],
    dynamic_sub: usize,
    tensor_role: &str,
) -> ny_core::Result<(Vec<usize>, bool)> {
    let mut has_dynamic = false;
    let resolved = shape
        .iter()
        .map(|&dim| {
            let (resolved_dim, is_dynamic) =
                resolve_verification_dim(dim, dynamic_sub, tensor_role)?;
            has_dynamic |= is_dynamic;
            Ok(resolved_dim)
        })
        .collect::<ny_core::Result<Vec<_>>>()?;
    Ok((resolved, has_dynamic))
}

fn checked_verification_shape_product(
    shape: &[usize],
    tensor_role: &str,
) -> ny_core::Result<usize> {
    checked_shape_product(shape).ok_or_else(|| {
        ny_core::NyError::InvalidSpec(format!(
            "Verification {tensor_role} shape {:?} overflows usize",
            shape
        ))
    })
}

fn run_verification(
    onnx_model: ny_onnx::OnnxModel,
    prop_method: PropagationMethod,
    epsilon: f32,
    timeout: u64,
    output_bounds: Option<Vec<(f32, f32)>>,
    rust_beta_config: Option<ny_propagate::BetaCrownConfig>,
    mul_binary_relaxation: MulBinaryRelaxationMode,
    verify_backend: VerifyBackend,
    batch_size: Option<usize>,
) -> ny_core::Result<RustVerificationResult> {
    let VerifyBackend { use_gpu, engine } = verify_backend;
    let config = PropagationConfig {
        method: prop_method,
        max_iterations: 100,
        tolerance: 1e-4,
        use_gpu,
        mul_binary_relaxation,
        double_fp: false,
    };

    let onnx_network = &onnx_model.network;

    // Convert to propagate network
    let prop_network = onnx_model.to_propagate_network()?;

    // Create input shape, handling unresolved ONNX dimensions.
    // ny-onnx normalizes symbolic/missing dimensions to -1, and some
    // preserved shape metadata can still surface 0-valued unresolved dims.
    // Both use batch_size (default 1); other negative dims stay invalid.
    let dynamic_sub = batch_size.unwrap_or(1);
    if dynamic_sub == 0 {
        return Err(ny_core::NyError::InvalidSpec(
            "batch_size must be >= 1".to_string(),
        ));
    }
    let mut input_shape: Vec<usize> = match onnx_network.inputs.first() {
        Some(input_spec) => {
            let (shape, has_dynamic) =
                resolve_verification_shape(&input_spec.shape, dynamic_sub, "input")?;
            if has_dynamic {
                tracing::warn!(
                    "ONNX input has dynamic dimensions; substituting with batch_size={dynamic_sub}"
                );
            }
            shape
        }
        None => {
            tracing::warn!("ONNX model has no input specs; using fallback shape [100]");
            vec![100]
        }
    };

    // Squeeze leading batch dimension of 1
    if input_shape.len() >= 2 && input_shape[0] == 1 {
        input_shape.remove(0);
    }

    let input_dim = checked_verification_shape_product(&input_shape, "input")?;

    // Dynamic output dims also use batch_size; invalid negatives are errors (#2689).
    let output_dim = match onnx_network.outputs.first() {
        Some(output_spec) => {
            let (shape, has_dynamic) =
                resolve_verification_shape(&output_spec.shape, dynamic_sub, "output")?;
            let dim = checked_verification_shape_product(&shape, "output")?.max(1);
            if has_dynamic {
                tracing::warn!(
                    "ONNX output has dynamic dimensions; substituting with batch_size={dynamic_sub}"
                );
            }
            dim
        }
        None => {
            tracing::warn!("ONNX model has no output specs; using fallback output_dim=10");
            10
        }
    };

    // Build the required output bounds. A caller-supplied specification is
    // validated per output; without one the requirement is the unconstrained
    // (-inf, +inf) box, which any sound bounds satisfy — so the final verdict
    // is folded to Unknown below instead of being reported as a property result.
    let required_output_bounds = output_bounds
        .map(|bounds| build_output_spec_bounds(&bounds, output_dim))
        .transpose()?;
    let has_output_spec = required_output_bounds.is_some();
    let spec_output_bounds = required_output_bounds.unwrap_or_else(|| {
        vec![RustBound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); output_dim]
    });

    // Create specification
    let spec = VerificationSpec::from_parts(
        vec![RustBound::new(-epsilon, epsilon); input_dim],
        spec_output_bounds,
        Some(timeout * 1000),
        Some(input_shape.clone()),
    )?;

    // If beta-CROWN with custom config, use BetaCrownVerifier directly
    if let Some(mut beta_cfg) =
        rust_beta_config.filter(|_| matches!(prop_method, PropagationMethod::BetaCrown))
    {
        // Override timeout from function parameter
        beta_cfg.timeout = std::time::Duration::from_secs(timeout);

        let beta_verifier = build_beta_crown_verifier(beta_cfg, engine);

        // Create input bounds tensor
        let lower = vec![-epsilon; input_dim];
        let upper = vec![epsilon; input_dim];
        let input_tensor = BoundedTensor::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&input_shape), lower)
                .map_err(|e| ny_core::NyError::InvalidSpec(e.to_string()))?,
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&input_shape), upper)
                .map_err(|e| ny_core::NyError::InvalidSpec(e.to_string()))?,
        )?;
        // Derive BaB threshold from spec output bounds (#3229).
        let threshold = derive_bab_threshold(spec.output_bounds());
        let bab_result = beta_verifier.verify(&prop_network, &input_tensor, threshold)?;
        let mut provenance =
            soundness_provenance_for_network(&prop_network, &PropagationMethod::BetaCrown);
        let sqrt_negative_domain_nodes =
            count_sqrt_negative_domain_network(&prop_network, &input_tensor)?;
        if sqrt_negative_domain_nodes > 0 {
            let mut heuristics = provenance.heuristics_used().to_vec();
            heuristics.push(RustHeuristicUsed::SqrtNegativeDomain {
                num_nodes: sqrt_negative_domain_nodes,
            });
            provenance = RustSoundnessProvenance::from_heuristics(heuristics);
        }
        // Extract actual output bounds from BaB result when available (#2802).
        // NaN sanitization: NaN → conservative (-inf, +inf) per #2663.
        // Inverted bounds guard: fall back to [-inf, +inf] when lower > upper
        // (numerical corruption) — matches apply_bab_output_bounds in network.rs:352.
        let actual_output_bounds: Option<Vec<RustBound>> =
            bab_result.output_bounds.as_ref().map(|tensor| {
                let flat = tensor.flatten();
                let (lower, upper) = flat.lower_upper();
                lower
                    .iter()
                    .zip(upper.iter())
                    .map(|(&l, &u)| {
                        let safe_l = if l.is_nan() { f32::NEG_INFINITY } else { l };
                        let safe_u = if u.is_nan() { f32::INFINITY } else { u };
                        if safe_l <= safe_u {
                            RustBound::new_allow_infinite(safe_l, safe_u)
                        } else {
                            // Inverted bounds: numerical corruption, use conservative fallback.
                            RustBound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)
                        }
                    })
                    .collect()
            });

        // Convert BabResult to VerificationResult using actual bounds when
        // available, None when BaB did not produce tensor-shaped bounds.
        let result = match bab_result.result {
            ny_propagate::BabVerificationStatus::Verified => {
                match bab_verified_spec_gap(
                    spec.output_bounds(),
                    actual_output_bounds.as_deref(),
                    threshold,
                ) {
                    Some(gap) => RustVerificationResult::Unknown {
                        provenance,
                        bounds: actual_output_bounds.unwrap_or_default(),
                        reason: ny_core::UnknownReason::BoundsTooLoose { gap: Some(gap) },
                        actual_method: Some(ny_core::MethodUsed::BetaCrown),
                    },
                    None => RustVerificationResult::Verified {
                        provenance,
                        output_bounds: actual_output_bounds.unwrap_or_default(),
                        proof: None,
                        actual_method: Some(ny_core::MethodUsed::BetaCrown),
                    },
                }
            }
            ny_propagate::BabVerificationStatus::Violated {
                counterexample,
                output,
            } => {
                if has_output_spec {
                    // BaB flags Violated when any output dips below the global
                    // threshold, which an output without a finite lower
                    // requirement is allowed to do — and its counterexamples
                    // come from f32 arithmetic that is not trusted blindly.
                    // Confirm by concrete re-evaluation against the per-output
                    // requirements, downgrading to Unknown when unconfirmed.
                    match validate_bab_counterexample(
                        &prop_network,
                        &input_tensor,
                        spec.output_bounds(),
                        &counterexample,
                    ) {
                        Ok(Some(actual_output)) => RustVerificationResult::Violated {
                            provenance,
                            counterexample,
                            output: actual_output,
                            details: None,
                            actual_method: Some(ny_core::MethodUsed::BetaCrown),
                        },
                        Ok(None) => RustVerificationResult::Unknown {
                            provenance,
                            bounds: actual_output_bounds.unwrap_or_default(),
                            reason: ny_core::UnknownReason::PotentialViolation,
                            actual_method: Some(ny_core::MethodUsed::BetaCrown),
                        },
                        Err(err) => {
                            tracing::warn!(
                                ?err,
                                "beta-CROWN counterexample validation errored; \
                                 downgrading Violated to Unknown"
                            );
                            RustVerificationResult::Unknown {
                                provenance,
                                bounds: actual_output_bounds.unwrap_or_default(),
                                reason: ny_core::UnknownReason::PotentialViolation,
                                actual_method: Some(ny_core::MethodUsed::BetaCrown),
                            }
                        }
                    }
                } else {
                    RustVerificationResult::Violated {
                        provenance,
                        counterexample,
                        output,
                        details: None,
                        actual_method: Some(ny_core::MethodUsed::BetaCrown),
                    }
                }
            }
            ny_propagate::BabVerificationStatus::PotentialViolation => {
                RustVerificationResult::Unknown {
                    provenance,
                    bounds: actual_output_bounds.unwrap_or_default(),
                    reason: ny_core::UnknownReason::PotentialViolation,
                    actual_method: Some(ny_core::MethodUsed::BetaCrown),
                }
            }
            ny_propagate::BabVerificationStatus::Unknown { reason } => {
                RustVerificationResult::Unknown {
                    provenance,
                    bounds: actual_output_bounds.unwrap_or_default(),
                    reason: ny_core::UnknownReason::from(reason),
                    actual_method: Some(ny_core::MethodUsed::BetaCrown),
                }
            }
            ny_propagate::BabVerificationStatus::Timeout => RustVerificationResult::Timeout {
                provenance,
                partial_bounds: actual_output_bounds,
                actual_method: Some(ny_core::MethodUsed::BetaCrown),
            },
        };
        return Ok(if has_output_spec {
            result
        } else {
            fold_unspecified_property(result)
        });
    }

    // Standard verification path
    let verifier = build_standard_verifier(config, engine);
    let result = verifier.verify(&prop_network, &spec)?;
    Ok(if has_output_spec {
        result
    } else {
        fold_unspecified_property(result)
    })
}

pub(crate) fn build_verify_result(
    result: RustVerificationResult,
    method_str: String,
    epsilon: f32,
) -> VerifyResult {
    let soundness: SoundnessProvenance = result.provenance().clone().into();
    let (status, output_bounds, counterexample, counterexample_output, reason, actual_method) =
        match result {
            RustVerificationResult::Verified {
                output_bounds,
                actual_method,
                ..
            } => (
                VerifyStatus::Verified,
                Some(output_bounds.into_iter().map(|b| b.into()).collect()),
                None,
                None,
                None,
                actual_method,
            ),
            RustVerificationResult::Violated {
                counterexample,
                output,
                details,
                actual_method,
                ..
            } => {
                // Include violation explanation if available
                let reason = details.as_ref().map(|d| d.explanation().to_string());
                (
                    VerifyStatus::Violated,
                    None,
                    Some(counterexample),
                    Some(output),
                    reason,
                    actual_method,
                )
            }
            RustVerificationResult::Unknown {
                bounds,
                reason,
                actual_method,
                ..
            } => (
                VerifyStatus::Unknown,
                Some(bounds.into_iter().map(|b| b.into()).collect()),
                None,
                None,
                Some(reason.to_string()),
                actual_method,
            ),
            RustVerificationResult::Timeout {
                partial_bounds,
                actual_method,
                ..
            } => (
                VerifyStatus::Timeout,
                partial_bounds.map(|b| b.into_iter().map(|bound| bound.into()).collect()),
                None,
                None,
                Some("Verification timed out".to_string()),
                actual_method,
            ),
        };

    VerifyResult {
        status,
        soundness,
        output_bounds,
        counterexample,
        counterexample_output,
        reason,
        method: method_str,
        actual_method: actual_method.map(|m| m.to_string()),
        epsilon,
    }
}

/// Verify a neural network property using bound propagation.
///
/// Uses bound propagation (IBP, CROWN, α-CROWN, or β-CROWN) to compute
/// certified output bounds for all inputs within an epsilon ball, and checks
/// them against the required output_bounds when given.
///
/// Args:
///     model_path: Path to ONNX model
///     epsilon: Input perturbation radius (default: 0.01)
///     method: Verification method - 'ibp', 'crown', 'alpha', 'sdp-crown', or 'beta' (default: 'alpha').
///         'sdp-crown' is only valid over an ℓ2 input ball; this API perturbs inputs
///         over an ℓ∞ epsilon box, so 'sdp-crown' is refused at verify time with an
///         error. Use 'crown' or 'alpha' instead.
///     timeout: Timeout in seconds (default: 60)
///     beta_config: Optional BetaCrownConfig for β-CROWN configuration (only used when method='beta')
///     mul_binary_relaxation: MulBinary relaxation mode ("mccormick" or "middle"; default: "mccormick")
///     backend: Compute backend - 'auto', 'cpu', or 'wgpu' (default: 'auto').
///         Auto tries GPU GEMM backends for CROWN-family methods and falls back to CPU.
///     batch_size: Replacement for unresolved ONNX dimensions (-1 and preserved 0
///         values). Defaults to 1 if not specified. A warning is emitted when
///         dynamic dimensions are substituted.
///     output_bounds: The property to verify: one (lower, upper) requirement per
///         model output, e.g. [(0.0, 1.0), (float("-inf"), 0.5)]. Endpoints may
///         be infinite for one-sided constraints, but at least one endpoint
///         overall must be finite. When omitted, no property is checked and the
///         result status is Unknown with the certified bounds attached.
///
/// Returns:
///     VerifyResult with verification status and output bounds. Without
///     output_bounds the status is never Verified: bounds are computed, but
///     there is no property to verify.
///
/// Example:
///     >>> result = ny.verify("model.onnx", epsilon=0.01, output_bounds=[(-1.0, 1.0)] * 10)
///     >>> assert result.is_verified, f"Verification failed: {result.reason}"
///     >>> print(f"Output bounds certified with max width: {result.max_output_width():.2e}")
///
/// Example computing bounds only (no property):
///     >>> result = ny.verify("model.onnx", epsilon=0.01)
///     >>> print(result.status, result.max_output_width())  # Unknown, bounds available
///
/// Example with β-CROWN config:
///     >>> config = ny.BetaCrownConfig()
///     >>> config.branching = ny.BranchingHeuristic.Kfsb
///     >>> config.enable_proactive_cuts = True
///     >>> result = ny.verify("model.onnx", method="beta", beta_config=config,
///     ...                    output_bounds=[(0.0, float("inf"))] * 10)
#[pyfunction]
#[pyo3(signature = (model_path, epsilon=0.01, method="alpha", timeout=60, beta_config=None, mul_binary_relaxation="mccormick", backend="auto", batch_size=None, output_bounds=None))]
// Justification: Python API binding — pyo3 requires all parameters as function arguments.
#[allow(clippy::too_many_arguments)]
pub fn verify(
    py: Python<'_>,
    model_path: &str,
    epsilon: f32,
    method: &str,
    timeout: u64,
    beta_config: Option<BetaCrownConfig>,
    mul_binary_relaxation: &str,
    backend: &str,
    batch_size: Option<usize>,
    output_bounds: Option<Vec<(f32, f32)>>,
) -> PyResult<VerifyResult> {
    validate_epsilon(epsilon)?;

    // Parse method
    let prop_method = match method {
        "ibp" => PropagationMethod::Ibp,
        "crown" => PropagationMethod::Crown,
        "alpha" => PropagationMethod::AlphaCrown,
        "sdp" | "sdp-crown" => PropagationMethod::SdpCrown,
        "beta" => PropagationMethod::BetaCrown,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown method: {}. Use 'ibp', 'crown', 'alpha', 'sdp-crown' (ℓ2 input \
                 balls only; refused for ℓ∞ box specs), or 'beta'",
                method
            )));
        }
    };

    // Convert Python BetaCrownConfig to Rust if provided (validates float fields)
    let rust_beta_config = beta_config.map(|c| c.to_rust()).transpose()?;
    let mul_binary_relaxation = parse_mul_binary_relaxation(mul_binary_relaxation)?;
    let backend = backend.to_string();

    // Load and verify (release GIL during computation)
    let result = Python::detach(py, || {
        let verify_backend = resolve_verify_backend(&backend, prop_method)?;
        // Load ONNX model
        let onnx_model = load_onnx(model_path)?;
        run_verification(
            onnx_model,
            prop_method,
            epsilon,
            timeout,
            output_bounds,
            rust_beta_config,
            mul_binary_relaxation,
            verify_backend,
            batch_size,
        )
    })
    .map_err(|e| PyValueError::new_err(format!("Verification error: {}", e)))?;

    Ok(build_verify_result(result, method.to_string(), epsilon))
}

/// Verify a neural network property using in-memory ONNX bytes.
///
/// Accepts the same output_bounds specification as `verify`; when omitted, no
/// property is checked and the result status is Unknown with certified bounds
/// attached. Method restrictions also match `verify`: 'sdp-crown' requires an
/// ℓ2 input ball and is refused at verify time for the ℓ∞ epsilon box this
/// API builds.
#[pyfunction]
#[pyo3(signature = (model_bytes, epsilon=0.01, method="alpha", timeout=60, beta_config=None, mul_binary_relaxation="mccormick", name="in_memory", backend="auto", batch_size=None, output_bounds=None))]
// Justification: Python API binding — pyo3 requires all parameters as function arguments.
#[allow(clippy::too_many_arguments)]
pub fn verify_bytes(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    epsilon: f32,
    method: &str,
    timeout: u64,
    beta_config: Option<BetaCrownConfig>,
    mul_binary_relaxation: &str,
    name: &str,
    backend: &str,
    batch_size: Option<usize>,
    output_bounds: Option<Vec<(f32, f32)>>,
) -> PyResult<VerifyResult> {
    validate_epsilon(epsilon)?;

    let prop_method = match method {
        "ibp" => PropagationMethod::Ibp,
        "crown" => PropagationMethod::Crown,
        "alpha" => PropagationMethod::AlphaCrown,
        "sdp" | "sdp-crown" => PropagationMethod::SdpCrown,
        "beta" => PropagationMethod::BetaCrown,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown method: {}. Use 'ibp', 'crown', 'alpha', 'sdp-crown' (ℓ2 input \
                 balls only; refused for ℓ∞ box specs), or 'beta'",
                method
            )));
        }
    };

    let rust_beta_config = beta_config.map(|c| c.to_rust()).transpose()?;
    let mul_binary_relaxation = parse_mul_binary_relaxation(mul_binary_relaxation)?;
    let name = name.to_string();
    let backend = backend.to_string();

    let result = Python::detach(py, || {
        let verify_backend = resolve_verify_backend(&backend, prop_method)?;
        let onnx_model = load_onnx_bytes(&name, &model_bytes)?;
        run_verification(
            onnx_model,
            prop_method,
            epsilon,
            timeout,
            output_bounds,
            rust_beta_config,
            mul_binary_relaxation,
            verify_backend,
            batch_size,
        )
    })
    .map_err(|e| PyValueError::new_err(format!("Verification error: {}", e)))?;

    Ok(build_verify_result(result, method.to_string(), epsilon))
}

/// Verify a neural network property from a PyTorch module without writing ONNX to disk.
///
/// Accepts the same output_bounds specification as `verify`; when omitted, no
/// property is checked and the result status is Unknown with certified bounds
/// attached. Method restrictions also match `verify`: 'sdp-crown' requires an
/// ℓ2 input ball and is refused at verify time for the ℓ∞ epsilon box this
/// API builds.
#[pyfunction]
#[pyo3(signature = (model, example_input, epsilon=0.01, method="alpha", timeout=60, beta_config=None, mul_binary_relaxation="mccormick", opset=17, name="torch_in_memory", backend="auto", batch_size=None, output_bounds=None))]
// Justification: Python API binding — pyo3 requires all parameters as function arguments.
#[allow(clippy::too_many_arguments)]
pub fn verify_torch(
    py: Python<'_>,
    model: &Bound<'_, PyAny>,
    example_input: &Bound<'_, PyAny>,
    epsilon: f32,
    method: &str,
    timeout: u64,
    beta_config: Option<BetaCrownConfig>,
    mul_binary_relaxation: &str,
    opset: u32,
    name: &str,
    backend: &str,
    batch_size: Option<usize>,
    output_bounds: Option<Vec<(f32, f32)>>,
) -> PyResult<VerifyResult> {
    let model_bytes = export_torch_to_onnx_bytes(py, model, example_input, opset, "verify_torch")?;
    verify_bytes(
        py,
        model_bytes,
        epsilon,
        method,
        timeout,
        beta_config,
        mul_binary_relaxation,
        name,
        backend,
        batch_size,
        output_bounds,
    )
}

#[cfg(test)]
mod tests {
    use super::{checked_verification_shape_product, resolve_verification_shape};
    use ny_core::NyError;

    #[test]
    fn test_resolve_verification_shape_substitutes_zero_and_negative_one_dims_2883() {
        let (shape, has_dynamic) =
            resolve_verification_shape(&[0, 80, -1], 4, "input").expect("shape should resolve");

        assert_eq!(shape, vec![4, 80, 4]);
        assert!(has_dynamic, "0 and -1 dims should both count as dynamic");
    }

    #[test]
    fn test_resolve_verification_shape_rejects_other_negative_dims_2689() {
        let err = resolve_verification_shape(&[-2, 80], 4, "input")
            .expect_err("unexpected negative dims must stay invalid");

        assert!(
            matches!(err, NyError::InvalidSpec(ref message)
                if message.contains("Unsupported negative ONNX input dimension -2")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_checked_verification_shape_product_rejects_overflow_2602() {
        let err = checked_verification_shape_product(&[usize::MAX, 2], "input")
            .expect_err("overflowed verification input shape must be rejected");

        assert!(
            matches!(err, NyError::InvalidSpec(ref message)
                if message.contains("Verification input shape")
                    && message.contains("overflows usize")),
            "unexpected error: {err}"
        );
    }
}
