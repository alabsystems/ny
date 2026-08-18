// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Verification command handler — orchestration coordinator.
//!
//! Submodules:
//! - `options`: CLI option parsing, validation, and spec construction
//! - `model_load`: NNet/native/ONNX model loading
//! - `modes`: Layer-by-layer and block-wise execution
//! - `result`: JSON/text rendering and exit-code policy

mod model_load;
mod modes;
mod options;
mod request;
mod result;

#[cfg(test)]
pub(crate) use options::resolve_effective_backend_with_factory;
pub(crate) use request::VerificationConfig;
pub(crate) use result::exit_codes;
pub(crate) use result::json_f32;

use anyhow::Result;
use ny_core::GemmEngine;
use ny_propagate::Verifier;
use std::sync::Arc;
use tracing::info;

use crate::BackendArg;

mod f64_verify;

pub(crate) fn handle_verify_command(config: VerificationConfig) -> Result<()> {
    let requested_method = options::parse_method(&config.method)?;
    let supports_wgpu_proof = options::supports_qualified_wgpu_proof(
        requested_method,
        config.layer_by_layer,
        config.use_block_wise(),
        config.double_fp,
    );
    let resolved = options::resolve_effective_backend(
        config.backend,
        config.backend_automatic,
        config.gpu,
        config.json,
        supports_wgpu_proof,
    )?;
    // `main` deliberately defers this lazy auxiliary factory for proof-capable
    // commands. Install it only for a genuine CPU request; a retained WGPU
    // proof context owns the adapter alone, while a WGPU fallback receipt must
    // remain a real CPU execution rather than silently opening an auxiliary
    // WGPU context after qualification/mode admission refused.
    if resolved.receipt.requested == BackendArg::Cpu
        && !resolved.use_gpu
        && !crate::compute_backend::detect().wgpu_probe_skipped
    {
        ny_propagate::fl_value_gemm::set_fl_value_gemm_factory(|| {
            match ny_gpu::FlValueGemmDevice::new_wgpu() {
                Ok(device) => Some(Arc::new(device) as Arc<dyn GemmEngine>),
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "verify FL-value wgpu engine unavailable; certified value GEMMs \
                         keep their CPU tiers"
                    );
                    None
                }
            }
        });
    }
    handle_verify_command_with_resolved_backend(config, resolved)
}

fn handle_verify_command_with_resolved_backend<T>(
    config: VerificationConfig,
    resolved: options::ResolvedBackend<T>,
) -> Result<()>
where
    T: GemmEngine + 'static,
{
    if !config.json {
        info!("Verifying model: {}", config.model.display());
    }

    // --- Phase 1: Parse and validate options ---

    let requested_method = options::parse_method(&config.method)?;
    let mul_binary_relaxation =
        ny_propagate::MulBinaryRelaxationMode::from(config.mul_binary_relaxation);
    let use_block_wise = config.use_block_wise();

    options::validate_option_compatibility(
        config.require_sound,
        config.layer_by_layer,
        use_block_wise,
        config.allow_heuristic_logsoftmax,
        config.allow_heuristic_softmax,
        config.max_blocks,
    )?;

    let effective_method = options::resolve_effective_method(
        requested_method,
        config.layer_by_layer,
        use_block_wise,
        config.strict,
        config.json,
    )?;
    let options::ResolvedBackend {
        backend: effective_backend,
        use_gpu,
        device,
        receipt: backend_receipt,
    } = resolved;

    let propagation_config = options::build_config(
        effective_method,
        config.max_iterations,
        config.tolerance,
        use_gpu,
        mul_binary_relaxation,
        config.double_fp,
    );
    if !config.json {
        info!("Backend: {}", effective_backend);
    }

    // --- Phase 2: Validate mode/model compatibility and load model ---

    let is_nnet = model_load::is_nnet_format(&config.model);
    let use_native = model_load::should_use_native(&config.model, config.native, is_nnet);
    model_load::validate_mode_model_compat(config.layer_by_layer, use_block_wise, use_native)?;

    // Every currently supported non-NNet loader produces a GraphNetwork.
    // `Verifier::verify_graph` has a BetaCrown compatibility branch, but that
    // branch only computes alpha-CROWN/CROWN bounds and does not perform
    // branch-and-bound. Refuse before loading instead of silently running a
    // different method. NNet uses the real sequential β-CROWN implementation.
    if requested_method == ny_propagate::PropagationMethod::BetaCrown && !is_nnet {
        anyhow::bail!(
            "`ny verify --method beta` is not supported for graph-backed models \
             (including ONNX); use `ny beta-crown MODEL -p PROPERTY` for complete \
             graph branch-and-bound verification"
        );
    }

    let loaded = model_load::load_model(
        &config.model,
        config.native,
        config.conservative_layernorm,
        config.layernorm_mode,
        config.layernorm_norm_mode,
        effective_method,
        config.peel_off_last_softmax_layer,
        config.property.as_deref(),
        config.json,
    )?;
    let mut network = loaded.network;
    let input_shape = loaded.input_shape;
    let output_dim = loaded.output_dim;
    let preloaded_vnnlib = loaded.preloaded_vnnlib;
    let applied_terminal_peel = loaded.applied_terminal_peel;

    // --- Phase 3: Apply soundness mode flags ---

    model_load::apply_soundness_modes(
        &mut network,
        config.allow_heuristic_logsoftmax,
        config.allow_heuristic_softmax,
        config.require_sound,
        config.json,
    );

    // --- Phase 4: Build verification spec ---

    let verifier = match device {
        Some(device) => {
            let engine: Arc<dyn GemmEngine> = Arc::new(device);
            // #charged-metal-engagement: publish the exact qualified proof
            // device into the process-global sound CROWN slots so the
            // deadline-preinitialized routes and borrow-only consumers can
            // see it. Fail-closed no-op unless the receipt proves live
            // qualification AND the engine advertises the sound accessor
            // (test-injected devices refuse), so CPU/refused resolutions
            // remain byte-identical.
            crate::commands::backend::register_qualified_wgpu_proof_engine(
                &backend_receipt,
                &engine,
            );
            Verifier::new_with_engine(propagation_config, engine)
        }
        None => Verifier::new(propagation_config),
    };

    let built = options::build_verification_spec(
        input_shape,
        output_dim,
        config.epsilon,
        config.timeout,
        config.property.as_deref(),
        preloaded_vnnlib,
        config.shrink_eps,
        config.json,
    )?;
    let spec = built.spec;
    let vnnlib_spec = built.vnnlib_spec;

    // --- Phase 5: Require-sound sqrt domain check ---

    if config.require_sound {
        result::check_require_sound_sqrt(&network, &spec)?;
    }

    // --- Phase 6: Dispatch to execution mode ---

    if config.layer_by_layer {
        return modes::run_layer_by_layer(
            &network,
            spec.input_shape(),
            config.epsilon,
            config.progress,
            config.progress_json,
            config.json,
            &backend_receipt,
        );
    }

    if use_block_wise {
        return modes::run_block_wise(
            &network,
            spec.input_shape(),
            config.epsilon,
            effective_method,
            effective_backend,
            &config.model,
            config.progress,
            config.progress_json,
            config.max_blocks,
            config.checkpoint.as_deref(),
            config.json,
            &backend_receipt,
        );
    }

    // --- Phase 6b: f64 double-precision path ---
    //
    // When --double-fp is active, convert the sequential network to f64 types
    // and run the f64 propagation engine. This is required for VNN-COMP
    // soundnessbench and sat_relu where f32 rounding causes incorrect verdicts.
    // Reference: alpha-beta-CROWN `double_fp: true` (`abcrown.py:81-82`).

    if config.double_fp {
        return f64_verify::run_f64_verification(
            &network,
            &spec,
            requested_method,
            effective_method,
            vnnlib_spec.as_ref(),
            config.property.as_deref(),
            config.strict,
            config.allow_unknown,
            config.json,
            applied_terminal_peel,
            &backend_receipt,
        );
    }

    // --- Phase 7: Standard verification ---

    result::run_standard_verification(
        &network,
        &spec,
        &verifier,
        requested_method,
        &backend_receipt,
        config.epsilon,
        vnnlib_spec.as_ref(),
        config.property.as_deref(),
        config.strict,
        config.require_sound,
        config.allow_unknown,
        config.json,
        applied_terminal_peel,
    )
}

#[cfg(test)]
fn handle_verify_command_with_backend_factory<T, F>(
    config: VerificationConfig,
    build_device: F,
) -> Result<()>
where
    T: GemmEngine + 'static,
    F: FnOnce(BackendArg) -> Result<T>,
{
    let resolved = resolve_effective_backend_with_factory(
        config.backend,
        config.gpu,
        config.json,
        options::supports_qualified_wgpu_proof(
            options::parse_method(&config.method)?,
            config.layer_by_layer,
            config.use_block_wise(),
            config.double_fp,
        ),
        build_device,
    );
    handle_verify_command_with_resolved_backend(config, resolved)
}

#[cfg(test)]
mod tests {
    use super::{handle_verify_command_with_backend_factory, VerificationConfig};
    use crate::BackendArg;
    use ny_core::{GemmEngine, NaiveCpuGemmEngine, Result};
    use ny_test_utils::{require_model, test_models_dir};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct CountingGemmDevice {
        gemm_calls: Arc<AtomicUsize>,
    }

    impl CountingGemmDevice {
        fn new(gemm_calls: Arc<AtomicUsize>) -> Self {
            Self { gemm_calls }
        }
    }

    impl GemmEngine for CountingGemmDevice {
        fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            self.gemm_calls.fetch_add(1, Ordering::SeqCst);
            NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
        }
    }

    fn verify_command_counting_engine_calls(model_name: &str) -> usize {
        let model_path = test_models_dir().join(model_name);
        require_model(&model_path);

        let gemm_calls = Arc::new(AtomicUsize::new(0));
        handle_verify_command_with_backend_factory(
            VerificationConfig::builder(model_path, 0.01, "crown".to_string())
                .verification(crate::MulBinaryRelaxationArg::Mccormick, 10, 1e-4, 5)
                .backend_request(BackendArg::Wgpu, false, false)
                .layernorm(
                    false,
                    crate::LayerNormModeArg::IbpValidated,
                    crate::LayerNormNormModeArg::Standard,
                )
                .output(true, false, false, false)
                .double_fp(false, None)
                .build(),
            {
                let gemm_calls = Arc::clone(&gemm_calls);
                move |_| Ok(CountingGemmDevice::new(gemm_calls))
            },
        )
        .expect("verify command should preserve the counting GEMM backend");

        gemm_calls.load(Ordering::SeqCst)
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_handle_verify_command_refuses_opaque_gemm_for_deadline_scored_sequential_crown_4321() {
        let calls = verify_command_counting_engine_calls("simple_2layer.nnet");
        assert_eq!(
            calls, 0,
            "#4321 regression: a finite-deadline sequential CROWN fold must not launch a generic GEMM engine without a cooperative cancellation contract"
        );
    }

    /// #3866: a successfully qualified proof device must survive option
    /// resolution, model loading, verifier construction, and graph dispatch.
    /// The injected factory stands for the typed `new_for_proof` constructor;
    /// unlike a real adapter it is hermetic and lets this test observe the
    /// engine at the propagation seam.
    #[ntest::timeout(10000)]
    #[test]
    fn test_verify_command_retains_the_qualified_proof_device_3866() {
        let calls = verify_command_counting_engine_calls("linear_relu.onnx");
        assert!(
            calls > 0,
            "the qualified proof device was lost before graph propagation"
        );
    }
}
