// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Option resolution for the verify command — method, backend, soundness flags,
//! and verification spec construction.

use anyhow::Result;
use ny_core::{checked_shape_product, Bound, VerificationSpec};
use ny_onnx::vnnlib::{load_vnnlib, VnnLibSpec};
use ny_propagate::layers::{LayerNormCrownMode, LayerNormMode};
use ny_propagate::{MulBinaryRelaxationMode, PropagationConfig, PropagationMethod};
use std::path::Path;
use tracing::{debug, info};

use super::super::JsonCliError;
use crate::BackendArg;

/// Parse a CLI method string into a `PropagationMethod`.
///
/// `sdp`/`sdp-crown` parses so the method stays addressable, but SDP-CROWN is
/// only valid over an ℓ2 input ball: verification refuses it for the ℓ∞ box
/// specs this command builds (epsilon balls and VNN-LIB boxes), so it fails
/// at verify time rather than here.
pub(crate) fn parse_method(method: &str) -> Result<PropagationMethod> {
    match method {
        "ibp" => Ok(PropagationMethod::Ibp),
        "crown" => Ok(PropagationMethod::Crown),
        "alpha" => Ok(PropagationMethod::AlphaCrown),
        "sdp" | "sdp-crown" => Ok(PropagationMethod::SdpCrown),
        "beta" => Ok(PropagationMethod::BetaCrown),
        _ => {
            anyhow::bail!(
                "Unknown method: {}. Use ibp, crown, alpha, sdp-crown (ℓ2 input balls only; \
                 refused for ℓ∞ box specs), or beta",
                method
            );
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
) -> Result<()> {
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

/// Resolved backend configuration after GPU fallback handling.
///
/// Wraps the shared `GemmBackendResolution` from `backend.rs`, keeping the
/// selected device alive so standard verification can call the explicit
/// `*_with_engine` verifier APIs.
pub(crate) struct ResolvedBackend<T = ny_gpu::ComputeDevice> {
    pub(crate) backend: BackendArg,
    pub(crate) use_gpu: bool,
    pub(crate) device: Option<T>,
}

/// Resolve the effective backend and initialize a compute device if applicable.
///
/// Delegates to the shared `resolve_gemm_backend` helper in `backend.rs` so the
/// GPU probe/fallback logic is not duplicated. The resolved `ComputeDevice` is
/// kept alive for downstream `verify_with_engine` / `verify_graph_with_engine`
/// calls (#3643).
pub(crate) fn resolve_effective_backend(
    backend: BackendArg,
    gpu: bool,
    json: bool,
) -> ResolvedBackend {
    resolve_effective_backend_with_factory(backend, gpu, json, |effective_backend| {
        Ok(ny_gpu::ComputeDevice::new(effective_backend.into())?)
    })
}

/// Resolve the effective backend using a caller-provided device factory.
///
/// The verify command uses this in tests to inject a counting `GemmEngine`
/// while keeping the production code on the normal `ComputeDevice` path.
pub(crate) fn resolve_effective_backend_with_factory<T, F>(
    backend: BackendArg,
    gpu: bool,
    json: bool,
    build_device: F,
) -> ResolvedBackend<T>
where
    F: FnOnce(BackendArg) -> Result<T>,
{
    let resolved =
        super::super::backend::resolve_gemm_backend_with_factory(backend, gpu, json, build_device);
    let use_gpu = resolved.device.is_some();
    ResolvedBackend {
        backend: resolved.backend,
        use_gpu,
        device: resolved.device,
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
    let input_shape_original = input_shape.clone();
    let input_dim_original = checked_verify_shape_product(&input_shape, "input")?;

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
            vnnlib.shrink_input_bounds(eps);
            if !json {
                info!("Shrunk VNN-LIB input bounds by {eps} (soundness defense)");
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
            vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); output_dim],
            Some(timeout * 1000),
            Some(vnnlib_shape),
        )?;

        (spec, Some(vnnlib))
    } else {
        let eps_bound = Bound::try_new(-epsilon, epsilon)
            .map_err(|e| anyhow::anyhow!("invalid epsilon {epsilon}: {e}"))?;
        let spec = VerificationSpec::from_parts(
            vec![eps_bound; input_dim],
            vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); output_dim],
            Some(timeout * 1000),
            Some(input_shape),
        )?;
        (spec, None)
    };

    Ok(BuiltSpec { spec, vnnlib_spec })
}

#[cfg(test)]
mod tests {
    use super::build_verification_spec;

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
}
