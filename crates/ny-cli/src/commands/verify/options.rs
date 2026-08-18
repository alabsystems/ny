// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Option resolution for the verify command — method, backend, soundness flags,
//! and verification spec construction.

use anyhow::Result;
use ny_core::{checked_shape_product, Bound, GemmEngine, VerificationSpec};
use ny_onnx::vnnlib::{load_vnnlib, VnnLibSpec};
use ny_propagate::layers::{LayerNormCrownMode, LayerNormMode};
use ny_propagate::{MulBinaryRelaxationMode, PropagationConfig, PropagationMethod};
use std::path::Path;
use tracing::{debug, info};

#[cfg(test)]
use super::super::backend::emit_backend_override;
use super::super::backend::{
    BackendRequest, BackendRequestSource, ProofBackendReceipt, WgpuProofRefusal,
};
use super::super::JsonCliError;
use crate::BackendArg;

/// Parse a CLI method string into a `PropagationMethod`.
///
/// SDP-CROWN is intentionally rejected here: the CLI currently constructs
/// only ℓ∞ box specifications, while SDP-CROWN requires an ℓ2 input ball.
pub(crate) fn parse_method(method: &str) -> Result<PropagationMethod> {
    match method {
        "ibp" => Ok(PropagationMethod::Ibp),
        "crown" => Ok(PropagationMethod::Crown),
        "alpha" => Ok(PropagationMethod::AlphaCrown),
        "sdp" | "sdp-crown" => anyhow::bail!(
            "SDP-CROWN requires an ℓ2 input-ball specification, which `ny verify` \
             does not yet expose; use crown or alpha"
        ),
        "beta" => Ok(PropagationMethod::BetaCrown),
        _ => {
            anyhow::bail!("Unknown method: {}. Use ibp, crown, alpha, or beta", method);
        }
    }
}

/// Validate option combinations that are mutually exclusive.
pub(crate) fn validate_option_compatibility(
    require_sound: bool,
    layer_by_layer: bool,
    use_block_wise: bool,
    allow_heuristic_logsoftmax: bool,
    allow_heuristic_softmax: bool,
    max_blocks: usize,
) -> Result<()> {
    if layer_by_layer && use_block_wise {
        return Err(JsonCliError::new(
            "incompatible_options",
            "--layer-by-layer cannot be combined with --block-wise or --checkpoint.",
        )
        .into());
    }

    if max_blocks > 0 && !use_block_wise {
        return Err(JsonCliError::new(
            "incompatible_options",
            "--max-blocks requires --block-wise or --checkpoint.",
        )
        .into());
    }

    if require_sound && (layer_by_layer || use_block_wise) {
        let mode = if layer_by_layer {
            "layer-by-layer"
        } else {
            "block-wise"
        };
        let message = format!(
            "Require-sound mode is not supported with --{}. \
            These modes don't produce full verification results with soundness provenance. \
            Use standard verification (without --{}) for sound verification.",
            mode, mode
        );
        return Err(JsonCliError::new("incompatible_options", message).into());
    }

    if require_sound && (allow_heuristic_logsoftmax || allow_heuristic_softmax) {
        return Err(JsonCliError::new(
            "incompatible_options",
            "Require-sound cannot be combined with heuristic softmax/logsoftmax relaxations.",
        )
        .into());
    }

    Ok(())
}

/// Resolve the effective propagation method, applying IBP fallback for
/// layer-by-layer and block-wise modes.
pub(crate) fn resolve_effective_method(
    requested: PropagationMethod,
    layer_by_layer: bool,
    use_block_wise: bool,
    strict: bool,
    json: bool,
) -> Result<PropagationMethod> {
    if (layer_by_layer || use_block_wise) && requested != PropagationMethod::Ibp {
        if strict {
            anyhow::bail!(
                "Strict mode: layer-by-layer/block-wise requires IBP, but requested {:?}",
                requested
            );
        }
        if !json {
            eprintln!(
                "Warning: layer-by-layer/block-wise uses IBP regardless of requested {:?}.",
                requested
            );
        }
        Ok(PropagationMethod::Ibp)
    } else {
        Ok(requested)
    }
}

/// Resolved backend configuration after live proof qualification and fallback.
///
/// Keeps an admitted device alive so standard verification can call the
/// explicit `*_with_engine` verifier APIs. `backend` always names execution,
/// not merely the request.
pub(crate) struct ResolvedBackend<T = ny_gpu::ComputeDevice> {
    pub(crate) backend: BackendArg,
    pub(crate) use_gpu: bool,
    pub(crate) device: Option<T>,
    pub(crate) receipt: ProofBackendReceipt,
}

/// Whether this verify execution path can consume the qualified f32 CROWN
/// engine. IBP-only diagnostic modes and the independent f64 implementation do
/// not reach that seam, so qualifying WGPU for them would create an unused
/// context and falsely report accelerated proof execution.
#[must_use]
pub(crate) const fn supports_qualified_wgpu_proof(
    method: PropagationMethod,
    layer_by_layer: bool,
    use_block_wise: bool,
    double_fp: bool,
) -> bool {
    !layer_by_layer
        && !use_block_wise
        && !double_fp
        && matches!(
            method,
            PropagationMethod::Crown | PropagationMethod::AlphaCrown | PropagationMethod::BetaCrown
        )
}

/// Resolve the effective backend and initialize a compute device if applicable.
///
/// WGPU uses the typed proof constructor, which eagerly qualifies the exact
/// device it returns. Any initialization/rung refusal emits the unconditional
/// override receipt and returns a concrete CPU device instead.
pub(crate) fn resolve_effective_backend(
    backend: BackendArg,
    backend_automatic: bool,
    gpu: bool,
    _json: bool,
    supports_wgpu_proof: bool,
) -> Result<ResolvedBackend> {
    use super::super::backend::{resolve_proof_backend, resolve_proof_backend_with_factories};

    let requested = super::super::backend::resolve_backend(backend, gpu);
    let source = if backend == BackendArg::Cpu && gpu {
        BackendRequestSource::LegacyGpuFlag
    } else if backend_automatic {
        BackendRequestSource::Auto
    } else {
        BackendRequestSource::DefaultedCliValue
    };
    let request = BackendRequest {
        backend: requested,
        source,
        selection_reason: None,
    };
    let request =
        super::super::backend::resolve_automatic_wgpu_request(request, supports_wgpu_proof)?;
    let requested = request.backend;
    let resolved = if requested == BackendArg::Wgpu && !supports_wgpu_proof {
        resolve_proof_backend_with_factories(
            request,
            "verify",
            || {
                let device = ny_gpu::ComputeDevice::new(ny_gpu::Backend::Cpu)?;
                let provenance = device.backend_provenance().to_string();
                Ok((device, provenance))
            },
            || {
                Err(WgpuProofRefusal {
                    reason: "selected verify mode has no qualified f32 CROWN consumer".to_string(),
                    failed_rung: Some("verify_mode_capability".to_string()),
                    qualification_provenance: None,
                })
            },
        )?
    } else {
        resolve_proof_backend(request, "verify")?
    };
    let use_gpu = resolved.receipt.qualified_wgpu_active();
    Ok(ResolvedBackend {
        backend: resolved.receipt.effective,
        use_gpu,
        device: Some(resolved.device),
        receipt: resolved.receipt,
    })
}

/// Resolve the effective backend using a caller-provided device factory.
///
/// The verify command uses this in tests to inject a counting `GemmEngine`
/// while keeping the production code on the normal `ComputeDevice` path.
#[cfg(test)]
pub(crate) fn resolve_effective_backend_with_factory<T, F>(
    backend: BackendArg,
    gpu: bool,
    _json: bool,
    supports_wgpu_proof: bool,
    build_device: F,
) -> ResolvedBackend<T>
where
    F: FnOnce(BackendArg) -> Result<T>,
{
    let requested = super::super::backend::resolve_backend(backend, gpu);
    let source = if backend == BackendArg::Cpu && gpu {
        BackendRequestSource::LegacyGpuFlag
    } else {
        BackendRequestSource::DefaultedCliValue
    };
    let request = BackendRequest {
        backend: requested,
        source,
        selection_reason: None,
    };

    match requested {
        BackendArg::Cpu => ResolvedBackend {
            backend: BackendArg::Cpu,
            use_gpu: false,
            device: None,
            receipt: ProofBackendReceipt::cpu(request, "cpu"),
        },
        BackendArg::Wgpu if !supports_wgpu_proof => {
            let receipt = ProofBackendReceipt::refused_wgpu(
                request,
                "cpu",
                None,
                Some("verify_mode_capability".to_string()),
                "selected verify mode has no qualified f32 CROWN consumer",
            );
            emit_backend_override("verify", &receipt);
            ResolvedBackend {
                backend: BackendArg::Cpu,
                use_gpu: false,
                device: None,
                receipt,
            }
        }
        BackendArg::Wgpu => match build_device(BackendArg::Wgpu) {
            Ok(device) => ResolvedBackend {
                backend: BackendArg::Wgpu,
                use_gpu: true,
                device: Some(device),
                receipt: ProofBackendReceipt::qualified_wgpu(
                    request,
                    "injected-qualified-proof-device",
                    "injected-test-adapter",
                ),
            },
            Err(error) => {
                let receipt = ProofBackendReceipt::refused_wgpu(
                    request,
                    "cpu",
                    None,
                    Some("proof_device_construction".to_string()),
                    error.to_string(),
                );
                emit_backend_override("verify", &receipt);
                ResolvedBackend {
                    backend: BackendArg::Cpu,
                    use_gpu: false,
                    device: None,
                    receipt,
                }
            }
        },
    }
}

/// Build a `PropagationConfig` from resolved options.
pub(crate) fn build_config(
    method: PropagationMethod,
    max_iterations: usize,
    tolerance: f32,
    use_gpu: bool,
    mul_binary_relaxation: MulBinaryRelaxationMode,
    double_fp: bool,
) -> PropagationConfig {
    PropagationConfig {
        method,
        max_iterations,
        tolerance,
        use_gpu,
        mul_binary_relaxation,
        double_fp,
    }
}

/// Apply LayerNorm configuration to a graph network.
///
/// Configures forward-mode, CROWN mode, and normalization mode based on CLI flags.
pub(crate) fn configure_layernorm(
    graph: &mut ny_propagate::GraphNetwork,
    conservative: bool,
    crown_mode: LayerNormCrownMode,
    norm_mode: LayerNormMode,
    effective_method: PropagationMethod,
    json: bool,
) {
    if !conservative {
        let num_modified = graph.set_layernorm_forward_mode(true);
        if num_modified > 0 && effective_method == PropagationMethod::Ibp && !json {
            eprintln!(
                "Warning: enabling LayerNorm forward-mode for {} LayerNorm nodes. \
                Forward-mode trades strict soundness for tighter bounds (may miss worst-case behavior for large perturbations). \
                Use --conservative-layernorm for sound verification.",
                num_modified
            );
        }
    }

    let num_ln_modified = graph.set_layernorm_crown_mode(crown_mode);
    if num_ln_modified > 0 && !json {
        match crown_mode {
            LayerNormCrownMode::Sampling => {
                eprintln!(
                    "Warning: LayerNorm CROWN using sampling mode for {} nodes (not provably sound).",
                    num_ln_modified
                );
            }
            LayerNormCrownMode::Cut => {
                debug!(
                    "LayerNorm CROWN using cut mode for {} nodes (sound but loses correlations).",
                    num_ln_modified
                );
            }
            LayerNormCrownMode::IbpValidated | LayerNormCrownMode::Sound => {}
        }
    }

    let num_norm_modified = graph.set_layernorm_norm_mode(norm_mode);
    if num_norm_modified > 0 && norm_mode == LayerNormMode::MeanOnly && !json {
        info!(
            "Using DeepT-style mean-only LayerNorm for {} nodes.",
            num_norm_modified
        );
    }
}

/// Result of building the verification spec from CLI inputs.
pub(crate) struct BuiltSpec {
    pub(crate) spec: VerificationSpec,
    pub(crate) vnnlib_spec: Option<VnnLibSpec>,
}

fn checked_verify_shape_product(shape: &[usize], shape_role: &str) -> Result<usize> {
    checked_shape_product(shape).ok_or_else(|| {
        anyhow::anyhow!(
            "Verification adapter {shape_role} shape {:?} overflows usize",
            shape
        )
    })
}

/// Bound-vector resource ceiling for the CLI adapter. Each entry is two f32
/// endpoints; larger model dimensions should use a streaming specification
/// path rather than one attacker-controlled monolithic allocation.
const MAX_VERIFICATION_BOUND_ELEMENTS: usize = 16 * 1024 * 1024;

fn validate_bound_count(count: usize, role: &str) -> Result<()> {
    if count > MAX_VERIFICATION_BOUND_ELEMENTS {
        anyhow::bail!(
            "Verification adapter {role} dimension {count} exceeds the \
             {MAX_VERIFICATION_BOUND_ELEMENTS}-element resource limit"
        );
    }
    Ok(())
}

fn repeated_bounds(bound: Bound, count: usize, role: &str) -> Result<Vec<Bound>> {
    validate_bound_count(count, role)?;
    let mut bounds = Vec::new();
    bounds.try_reserve_exact(count).map_err(|error| {
        anyhow::anyhow!("Verification adapter could not allocate {count} {role} bounds: {error}")
    })?;
    bounds.resize(count, bound);
    Ok(bounds)
}

fn shrink_vnnlib_input_bounds_checked(vnnlib: &mut VnnLibSpec, eps: f64) -> Result<()> {
    if !eps.is_finite() || eps <= 0.0 {
        anyhow::bail!("--shrink-eps must be positive and finite (got {eps})");
    }

    let invalid_after_shrink = |lower: f64, upper: f64| {
        let shrunk_lower = lower + eps;
        let shrunk_upper = upper - eps;
        shrunk_lower.is_nan() || shrunk_upper.is_nan() || shrunk_lower > shrunk_upper
    };

    for (input, &(lower, upper)) in vnnlib.input_bounds.iter().enumerate() {
        if invalid_after_shrink(lower, upper) {
            anyhow::bail!(
                "--shrink-eps {eps} would invalidate VNN-LIB input X_{input} bounds \
                 [{lower}, {upper}]"
            );
        }
    }
    for (clause, bounds) in vnnlib.per_clause_input_bounds.iter().enumerate() {
        for (&input, &(lower, upper)) in bounds {
            if invalid_after_shrink(lower, upper) {
                anyhow::bail!(
                    "--shrink-eps {eps} would invalidate VNN-LIB clause {clause} \
                     input X_{input} bounds [{lower}, {upper}]"
                );
            }
        }
    }

    vnnlib.shrink_input_bounds(eps);
    Ok(())
}

/// Build a `VerificationSpec` from CLI inputs: epsilon-ball or VNNLIB property.
///
/// Handles batch-dimension squeezing and VNNLIB dimension validation.
///
/// # Errors
/// Returns an error if the VNNLIB property dimensions don't match the model.
pub(crate) fn build_verification_spec(
    mut input_shape: Vec<usize>,
    output_dim: usize,
    epsilon: f32,
    timeout: u64,
    property: Option<&Path>,
    preloaded_vnnlib: Option<VnnLibSpec>,
    shrink_eps: Option<f64>,
    json: bool,
) -> Result<BuiltSpec> {
    let timeout_ms = timeout.checked_mul(1000).ok_or_else(|| {
        anyhow::anyhow!("Verification timeout {timeout} seconds overflows milliseconds")
    })?;
    let input_shape_original = input_shape.clone();
    let input_dim_original = checked_verify_shape_product(&input_shape, "input")?;
    validate_bound_count(input_dim_original, "input")?;
    validate_bound_count(output_dim, "output")?;

    // Squeeze out leading batch dimension of 1 for epsilon-ball input_dim computation.
    // Conv1d/Conv2d layers expect unbatched inputs.
    // VNNLIB path uses input_shape_original (pre-squeeze) for validation and spec shape.
    if input_shape.len() >= 2 && input_shape[0] == 1 {
        input_shape.remove(0);
        debug!("Squeezed batch dimension, input shape: {:?}", input_shape);
    }

    let input_dim = checked_verify_shape_product(&input_shape, "squeezed input")?;

    let (spec, vnnlib_spec) = if let Some(prop_path) = property {
        let mut vnnlib = if let Some(spec) = preloaded_vnnlib {
            spec
        } else {
            load_vnnlib(prop_path)?
        };

        // Apply shrink_eps at f64 precision before f32 conversion (#4299).
        // Reference: alpha-beta-CROWN `shrink_vnnlib` (`specifications.py:535-540`).
        if let Some(eps) = shrink_eps {
            shrink_vnnlib_input_bounds_checked(&mut vnnlib, eps)?;
            if !json {
                info!(
                    "Shrunk VNN-LIB input bounds by {eps}; verification now covers \
                     the resulting smaller property domain"
                );
            }
        }
        if !json {
            println!(
                "Loaded VNN-LIB property: {} inputs, {} outputs, {} constraints",
                vnnlib.num_inputs,
                vnnlib.num_outputs,
                vnnlib.output_constraints.len()
            );
        }

        if vnnlib.num_inputs != input_dim_original {
            anyhow::bail!(
                "Property file specifies {} inputs but model expects {} (shape {:?})",
                vnnlib.num_inputs,
                input_dim_original,
                input_shape_original
            );
        }
        if vnnlib.num_outputs != output_dim {
            anyhow::bail!(
                "Property file specifies {} outputs but model expects {}",
                vnnlib.num_outputs,
                output_dim
            );
        }

        let (lower, upper) = vnnlib.split_input_bounds_f32();
        let input_bounds: Vec<Bound> = lower
            .iter()
            .zip(upper.iter())
            .enumerate()
            .map(|(i, (&l, &u))| {
                Bound::try_new_allow_infinite(l, u).map_err(|e| {
                    anyhow::anyhow!(
                        "VNN-LIB input bound X_{i} is invalid (lower={l}, upper={u}): {e}"
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut vnnlib_shape = input_shape_original;
        if vnnlib_shape.len() >= 2 && vnnlib_shape[0] == 1 {
            vnnlib_shape.remove(0);
            debug!(
                "Squeezed batch dimension for VNNLIB, shape: {:?}",
                vnnlib_shape
            );
        }

        let spec = VerificationSpec::from_parts(
            input_bounds,
            repeated_bounds(
                Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY),
                output_dim,
                "output",
            )?,
            Some(timeout_ms),
            Some(vnnlib_shape),
        )?;

        (spec, Some(vnnlib))
    } else {
        let eps_bound = Bound::try_new(-epsilon, epsilon)
            .map_err(|e| anyhow::anyhow!("invalid epsilon {epsilon}: {e}"))?;
        let spec = VerificationSpec::from_parts(
            repeated_bounds(eps_bound, input_dim, "input")?,
            repeated_bounds(
                Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY),
                output_dim,
                "output",
            )?,
            Some(timeout_ms),
            Some(input_shape),
        )?;
        (spec, None)
    };

    Ok(BuiltSpec { spec, vnnlib_spec })
}

#[cfg(test)]
mod tests {
    use super::{
        build_verification_spec, parse_method, shrink_vnnlib_input_bounds_checked,
        supports_qualified_wgpu_proof, validate_option_compatibility,
        MAX_VERIFICATION_BOUND_ELEMENTS,
    };
    use ny_onnx::vnnlib::VnnLibSpec;
    use ny_propagate::PropagationMethod;
    use std::path::Path;

    #[test]
    fn test_build_verification_spec_rejects_overflowed_input_shape_2602() {
        let err = build_verification_spec(vec![usize::MAX, 2], 1, 0.1, 1, None, None, None, true)
            .err()
            .expect("overflowed verification input shape must be rejected");

        let message = err.to_string();
        assert!(
            message.contains("Verification adapter input shape [")
                && message.contains("overflows usize"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn build_verification_spec_rejects_timeout_conversion_overflow() {
        let err = build_verification_spec(vec![1], 1, 0.1, u64::MAX, None, None, None, true)
            .err()
            .expect("overflowed timeout must be rejected");
        assert!(err.to_string().contains("overflows milliseconds"), "{err}");
    }

    #[test]
    fn build_verification_spec_rejects_oversized_bound_vectors() {
        let err = build_verification_spec(
            vec![1],
            MAX_VERIFICATION_BOUND_ELEMENTS + 1,
            0.1,
            1,
            None,
            None,
            None,
            true,
        )
        .err()
        .expect("oversized output bounds must reject before allocation");
        assert!(err.to_string().contains("resource limit"), "{err}");
    }

    #[test]
    fn build_verification_spec_rejects_oversized_input_before_loading_property() {
        let err = build_verification_spec(
            vec![MAX_VERIFICATION_BOUND_ELEMENTS + 1],
            1,
            0.1,
            1,
            Some(Path::new("missing.vnnlib")),
            None,
            None,
            true,
        )
        .err()
        .expect("oversized input dimensions must reject before property parsing");
        assert!(err.to_string().contains("resource limit"), "{err}");
        assert!(
            !err.to_string().contains("missing.vnnlib"),
            "resource validation must precede property I/O: {err}"
        );
    }

    #[test]
    fn shrink_eps_rejects_invalid_values_and_inverted_domains() {
        for eps in [0.0, -1e-10, f64::NAN, f64::INFINITY] {
            let err = shrink_vnnlib_input_bounds_checked(&mut VnnLibSpec::new(), eps)
                .expect_err("invalid shrink epsilon must return an error, not panic");
            assert!(err.to_string().contains("positive and finite"), "{err}");
        }

        let mut spec = VnnLibSpec::new();
        spec.input_bounds = vec![(0.0, 0.1)];
        let err = shrink_vnnlib_input_bounds_checked(&mut spec, 0.051)
            .expect_err("shrinking past the midpoint must be rejected");
        assert!(err.to_string().contains("would invalidate"), "{err}");
        assert_eq!(
            spec.input_bounds,
            vec![(0.0, 0.1)],
            "failed validation must not partially mutate the property"
        );
    }

    #[test]
    fn parse_method_rejects_sdp_until_l2_specs_are_exposed() {
        let err = parse_method("sdp-crown").expect_err("box-only CLI must reject SDP-CROWN");
        assert!(
            err.to_string().contains("requires an ℓ2 input-ball"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn max_blocks_requires_block_wise_execution() {
        let err = validate_option_compatibility(false, false, false, false, false, 2)
            .expect_err("--max-blocks must not be silently ignored");
        assert!(
            err.to_string()
                .contains("--max-blocks requires --block-wise or --checkpoint"),
            "unexpected error: {err}"
        );

        validate_option_compatibility(false, false, true, false, false, 2)
            .expect("block-wise verification may cap the number of blocks");
    }

    #[test]
    fn checkpoint_cannot_be_combined_with_layer_by_layer() {
        let err = validate_option_compatibility(false, true, true, false, false, 0)
            .expect_err("checkpoint-backed block-wise mode must conflict with layer-by-layer");
        assert!(
            err.to_string()
                .contains("--layer-by-layer cannot be combined"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn only_standard_f32_crown_modes_can_consume_qualified_wgpu() {
        for method in [
            PropagationMethod::Crown,
            PropagationMethod::AlphaCrown,
            PropagationMethod::BetaCrown,
        ] {
            assert!(supports_qualified_wgpu_proof(method, false, false, false));
            assert!(!supports_qualified_wgpu_proof(method, true, false, false));
            assert!(!supports_qualified_wgpu_proof(method, false, true, false));
            assert!(!supports_qualified_wgpu_proof(method, false, false, true));
        }
        assert!(!supports_qualified_wgpu_proof(
            PropagationMethod::Ibp,
            false,
            false,
            false,
        ));
    }
}
