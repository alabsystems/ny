// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

// Keep unsafe code denied throughout the CLI. The sole narrow exception is the
// `beta_crown::ay_tail_authority` verdict-capability bridge, whose unsafe calls
// attest that exact AY + ny-cert replay has already succeeded. Keeping that
// bridge in its own module makes the trust boundary mechanically auditable.
#![deny(unsafe_code)]

//! ny command-line interface bootstrap.
//!
//! This crate entry point owns process-wide setup for the CLI:
//! argument parsing, logging configuration, config/preset resolution, and
//! top-level dispatch into the command handlers.
//!
//! Source layout:
//! - `commands/` implements the executable command families
//! - `subcommands/` defines clap-facing CLI types and shared argument groups
//! - `config.rs` resolves verify-time YAML config, CSV instance, and CLI override state
//! - `preset/` applies competition and benchmark presets before command execution

// Link macOS Accelerate BLAS for ndarray::dot() acceleration (#4259).
#[cfg(target_os = "macos")]
extern crate blas_src;

// Global allocator: mimalloc's per-thread heaps eliminate the system-allocator
// LOCK CONTENTION measured (macOS `sample`, 2026-07-21) in the per-domain CROWN
// backward hot path of the relational input-split BaB — `_os_unfair_lock_lock_slow`
// + per-domain matrix alloc/memset were ~10% of the wall on the iso holdouts,
// caused by many rayon threads hammering the global malloc lock simultaneously.
// Pure performance; zero correctness/soundness impact (same computation, different
// memory manager). Benefits every allocation-heavy multi-threaded verify path.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Command handlers live in sibling modules to keep bootstrap and execution concerns separate.
mod commands;
mod config;
mod preset;
mod subcommands;

pub(crate) use subcommands::{
    AlphaGradientMethodArg, AlphaOptimizerArg, BackendArg, CompleteVerifierArg, LayerNormModeArg,
    LayerNormNormModeArg, MipSolverArg, MulBinaryRelaxationArg,
};

use anyhow::{Context, Result};
use clap::Parser;
use subcommands::{Cli, Commands, LogFormat};
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

const VNNCOMP_BUILD_PROVENANCE: &str = match option_env!("NY_VNNCOMP_BUILD_PROVENANCE_V1") {
    Some(value) => value,
    None => "ny.vnncomp.unsealed.v1|status=unsealed|",
};

fn main() -> Result<()> {
    // Windows gives a process's main thread only a 1 MiB stack by default,
    // where Linux and macOS give 8 MiB. ny's CROWN / β-CROWN branch-and-bound
    // bound propagation recurses deep enough to overflow 1 MiB on real models
    // (Windows aborts the process with STATUS_STACK_OVERFLOW, 0xC00000FD), so on
    // Windows we run the whole CLI on a worker thread with a generous explicit
    // stack. Unix/macOS keep the original direct call, so their behavior is
    // byte-for-byte unchanged.
    #[cfg(windows)]
    {
        // 256 MiB is a virtual reservation only — Windows commits stack pages on
        // demand — chosen to match the deep-recursion headroom the Linux/macOS
        // main thread already has. This does not mask runaway recursion: genuine
        // unbounded recursion still overflows any finite stack and aborts.
        const NY_MAIN_STACK_BYTES: usize = 256 * 1024 * 1024;
        let worker = std::thread::Builder::new()
            .name("ny-main".to_owned())
            .stack_size(NY_MAIN_STACK_BYTES)
            .spawn(run)
            .context("failed to spawn ny worker thread")?;
        match worker.join() {
            Ok(result) => result,
            // Propagate a worker panic on the main thread so the process still
            // exits via the normal panic path (identical to a panic in `run`).
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }
    #[cfg(not(windows))]
    {
        run()
    }
}

fn run() -> Result<()> {
    // Static, cross-architecture release identity. The submission packager
    // scans this exact string from the x86 ELF without executing foreign code;
    // the native builder also queries it before publishing the artifact.
    if std::env::args_os().nth(1).as_deref()
        == Some(std::ffi::OsStr::new("__vnncomp-build-provenance"))
    {
        println!("{VNNCOMP_BUILD_PROVENANCE}");
        return Ok(());
    }

    // Machine-readable compiled-feature report (`ny --build-info`). A plain
    // build (no cuda/mip) silently measures 0 on the GPU-dependent categories,
    // so scripts/measure_ny_scorecard.sh probes this line and refuses to
    // measure unless both features are compiled in. Intercepted before clap
    // parsing and logging setup so the output stays a single stable, greppable
    // stdout line with no side effects (no CUDA/adapter probing happens here).
    if std::env::args_os().nth(1).as_deref() == Some(std::ffi::OsStr::new("--build-info")) {
        println!(
            "ny {} features: cuda={} mip={}",
            env!("CARGO_PKG_VERSION"),
            if cfg!(feature = "cuda") { "on" } else { "off" },
            if cfg!(feature = "mip") { "on" } else { "off" },
        );
        return Ok(());
    }

    // Hidden subprocess entry (`ny __shape-infer`): serve one ONNX Runtime
    // shape-inference request over stdin/stdout, then exit. Intercepted before
    // clap parsing (it is not a user-facing command) and before logging setup,
    // so stdout carries nothing but the versioned response payload. See
    // `serve_shape_infer_subprocess` for why this exists.
    if std::env::args_os().nth(1).as_deref()
        == Some(std::ffi::OsStr::new(ny_onnx::SHAPE_INFER_SUBCOMMAND))
    {
        return serve_shape_infer_subprocess();
    }

    // Hidden subprocess entry (`ny __vnncomp-watchdog`): the OUT-OF-PROCESS
    // deadline backstop for `ny vnncomp` (see the spawn site in
    // `commands/vnncomp.rs`). Intercepted before clap parsing and logging
    // setup: the helper must stay free of GPU/ORT/preset/subscriber state —
    // its whole job is to keep functioning when the parent verifier process
    // has stopped scheduling entirely.
    if std::env::args_os().nth(1).as_deref()
        == Some(std::ffi::OsStr::new(
            commands::vnncomp::EXTERNAL_WATCHDOG_SUBCOMMAND,
        ))
    {
        return commands::vnncomp::serve_external_watchdog();
    }

    let cli = Cli::parse();
    let (verbose, log_format, command) = cli.into_parts();

    // Set up logging
    let level = match verbose {
        0 => Level::WARN,
        1 => Level::INFO,
        2 => Level::DEBUG,
        _ => Level::TRACE,
    };

    // Configure subscriber based on format
    // All log output goes to stderr to avoid polluting JSON/structured output on stdout
    match log_format {
        LogFormat::Text => {
            let subscriber = FmtSubscriber::builder()
                .with_max_level(level)
                .with_target(false)
                .with_writer(std::io::stderr)
                .finish();
            tracing::subscriber::set_global_default(subscriber)
                .context("Failed to set tracing subscriber")?;
        }
        LogFormat::Json => {
            let subscriber = FmtSubscriber::builder()
                .with_max_level(level)
                .with_target(false)
                .with_writer(std::io::stderr)
                .json()
                .finish();
            tracing::subscriber::set_global_default(subscriber)
                .context("Failed to set tracing subscriber")?;
        }
    }

    // Install the native CUDA / cuBLAS sound f64 GEMM accelerator if built with
    // `--features cuda` and a CUDA device is present. This routes the sound CPU
    // CROWN backward's f64 `A·W` / `|A|·|W|` products to cuBLAS Dgemm — a sound
    // (order-independent γ_n·S bound), ~18–34x faster verdict path that works
    // even under the sound_gpu_gate. No-op when no CUDA device is available.
    #[cfg(feature = "cuda")]
    if std::env::var_os("NY_NO_CUDA").is_none() {
        // Install LAZY factories: one shared CUDA engine (one GPU context + cuBLAS
        // handle), built on first use, drives BOTH the sound f64 `A·W` GEMM seam
        // AND the sound f64-exact GPU-resident CROWN backward. Lazy ⇒ attack-only /
        // CPU-trivial instances never pay the ~0.4s GPU init.
        use std::sync::{Arc, OnceLock};
        static CUDA_ENGINE: OnceLock<Option<Arc<ny_cuda::CudaGemmEngine>>> = OnceLock::new();
        fn shared_cuda_engine() -> Option<Arc<dyn ny_core::GemmEngine>> {
            CUDA_ENGINE
                .get_or_init(|| {
                    // ny-cuda probes for libcuda/libcublas before touching
                    // cudarc, so a genuinely CUDA-less host returns Err below
                    // and degrades cleanly to the sound CPU f64 path. cudarc's
                    // dynamic loader instead PANICS on a missing symbol in a
                    // library that did dlopen (partial/version-mismatched
                    // install). The release profile unwinds, so catch_unwind
                    // converts that Rust panic to the CPU fallback. A native
                    // abort, fault, or hang still needs process-level GPU
                    // isolation; the runner's pre-written `unknown` remains
                    // the fail-closed verdict if this process dies.
                    let constructed = std::panic::catch_unwind(ny_cuda::CudaGemmEngine::new);
                    match constructed {
                        Ok(Ok(engine)) => {
                            if ny_propagate::sound_gpu_gate::wide_sound_gpu_crown_requested() {
                                warn!(
                                    "CUDA wide CROWN factory ready: shared CUDA engine initialized \
                                     on {}",
                                    engine.device_name()
                                );
                            } else {
                                info!(
                                    "CUDA sound f64 engine installed ({}) — GEMM + resident CROWN",
                                    engine.device_name()
                                );
                            }
                            Some(Arc::new(engine))
                        }
                        Ok(Err(e)) => {
                            if ny_propagate::sound_gpu_gate::wide_sound_gpu_crown_requested() {
                                warn!(
                                    "CUDA wide CROWN factory unavailable: CUDA engine \
                                     initialization failed ({e}); using the fail-closed CPU path"
                                );
                            } else {
                                info!("CUDA acceleration unavailable; using CPU f64 path ({e})");
                            }
                            None
                        }
                        Err(panic) => {
                            let msg = panic
                                .downcast_ref::<String>()
                                .map(String::as_str)
                                .or_else(|| panic.downcast_ref::<&str>().copied())
                                .unwrap_or("<non-string panic>");
                            if ny_propagate::sound_gpu_gate::wide_sound_gpu_crown_requested() {
                                warn!(
                                    "CUDA wide CROWN factory unavailable: CUDA engine init \
                                     panicked ({msg}); using the fail-closed CPU path"
                                );
                            } else {
                                info!("CUDA engine init panicked; using CPU f64 path ({msg})");
                            }
                            None
                        }
                    }
                })
                .clone()
                .map(|e| e as Arc<dyn ny_core::GemmEngine>)
        }
        ny_propagate::sound_f64_gemm::set_sound_f64_gemm_factory(shared_cuda_engine);
        // Same shared engine drives the backend ComputeDevice's f32 GEMM seam
        // (IBP forward / PGD / BaB engine traffic): cuBLAS Sgemm is measured
        // 2-3.4x faster than the wgpu WGSL shader at every hotspot shape. IEEE
        // RN-f32 (math mode pinned), so the IBP call sites' order-independent
        // ULP widening stays valid. NY_NO_CUDA_F32 disables just this seam
        // (A/B measurement) without losing the sound f64 seam above.
        if std::env::var_os("NY_NO_CUDA_F32").is_none() {
            ny_propagate::fast_f32_gemm::set_fast_f32_gemm_factory(shared_cuda_engine);
        } else {
            info!("NY_NO_CUDA_F32 set; f32 GEMM engine offload disabled");
        }
        // Register CUDA separately for the domain-stacked proof forest.  Engine
        // construction stays lazy, and routing remains experimental until an
        // NVIDIA sealed A/B enables `NY_CUDA_WIDE=1` (or the master
        // `NY_HYDRA_CROWN=1`). This does not alter ordinary CROWN routing.
        ny_propagate::sound_gpu_gate::set_wide_sound_gpu_crown_factory(shared_cuda_engine);
        if ny_propagate::sound_gpu_gate::wide_sound_gpu_crown_requested() {
            warn!(
                "CUDA wide CROWN enabled: lazy factory registered; awaiting the first \
                 eligible domain batch (local/CPU fallback retained)"
            );
        } else {
            info!("CUDA wide CROWN registered but disabled; set NY_CUDA_WIDE=1 after A/B qualification");
        }
        // Preserve the legacy opt-in for non-wide, host-orchestrated CUDA CROWN,
        // which can be slower/weaker than CPU f64 on small networks.
        if std::env::var_os("NY_CUDA_CROWN").is_some() {
            info!("NY_CUDA_CROWN set; routing ordinary sound CROWN backward to CUDA");
            ny_propagate::sound_gpu_gate::set_sound_gpu_crown_factory(shared_cuda_engine);
        }
    } else {
        if ny_propagate::sound_gpu_gate::wide_sound_gpu_crown_requested() {
            warn!(
                "CUDA wide CROWN requested but NY_NO_CUDA is set; factory unavailable and \
                 the fail-closed CPU path remains active"
            );
        } else {
            info!("NY_NO_CUDA set; CUDA sound f64 GEMM disabled (CPU f64 path)");
        }
    }

    #[cfg(not(feature = "cuda"))]
    if ny_propagate::sound_gpu_gate::wide_sound_gpu_crown_requested() {
        warn!(
            "CUDA wide CROWN requested but this binary lacks the `cuda` feature; factory \
             unavailable and the fail-closed CPU path remains active"
        );
    }

    match command {
        Commands::Verify(args) => {
            let subcommands::VerifyArgs {
                model,
                model_flag,
                config,
                root_path,
                epsilon,
                property,
                peel_off_last_softmax_layer,
                method,
                mul_binary_relaxation,
                timeout,
                backend,
                gpu,
                native,
                conservative_layernorm,
                layernorm_mode,
                layernorm_norm_mode,
                layer_by_layer,
                block_wise,
                progress,
                progress_json,
                max_blocks,
                checkpoint,
                json,
                strict,
                require_sound,
                allow_heuristic_logsoftmax,
                allow_heuristic_softmax,
                allow_unknown,
                double_fp,
                shrink_eps,
            } = *args;
            let cli_model = model.or(model_flag);
            let cli_overrides = config::cli_overrides(
                cli_model.clone(),
                property.clone(),
                epsilon,
                method,
                Some(mul_binary_relaxation),
                timeout,
                backend,
                peel_off_last_softmax_layer,
            );

            let handle_result = |result: Result<()>| -> Result<()> {
                if let Err(err) = result {
                    if json {
                        // `--json` must be schema-stable even on failure (#395).
                        // Avoid Rust `main() -> Result<()>` termination printing `Error:` on stderr.
                        let error_output: serde_json::Value =
                            if let Some(json_err) = commands::find_json_cli_error(&err) {
                                json_err.payload().clone()
                            } else {
                                // Best-effort stable envelope for unexpected errors.
                                // Keep the schema simple (error + message) so callers can rely on it.
                                //
                                // (Issue #395)
                                serde_json::json!({
                                    "error": "verify_failed",
                                    "message": err.to_string(),
                                })
                            };
                        let json_payload =
                            serde_json::to_string_pretty(&error_output).or_else(|pretty_err| {
                                let fallback = serde_json::json!({
                                    "error": "verify_failed",
                                    "message": format!(
                                        "{err} (also failed to encode JSON: {pretty_err})"
                                    ),
                                });
                                serde_json::to_string(&fallback)
                            });
                        match json_payload {
                            Ok(s) => println!("{s}"),
                            Err(_) => println!(
                                "{{\"error\":\"verify_failed\",\"message\":\"failed to encode JSON\"}}"
                            ),
                        }
                        std::process::exit(1);
                    }
                    return Err(err);
                }
                Ok(())
            };

            let resolved_config = config::resolve_verify_config(config.clone(), root_path.clone())?;
            if let Some(resolved) = &resolved_config {
                if let Some(instances) = config::csv_instances(
                    &resolved.config,
                    &resolved.config_path,
                    root_path.as_deref(),
                )? {
                    if cli_model.is_some() || property.is_some() {
                        anyhow::bail!(
                            "Config CSV instances cannot be combined with --model or --property."
                        );
                    }
                    let instances =
                        config::select_instances(instances, resolved.config.data.as_ref());
                    if instances.is_empty() {
                        anyhow::bail!("No instances selected from config CSV.");
                    }
                    for (instance_index, instance) in instances.iter().enumerate() {
                        let instance_overrides = config::instance_overrides(instance);
                        let settings = config::resolve_verify_settings_from_config(
                            Some(resolved),
                            root_path.as_deref(),
                            cli_overrides.clone(),
                            instance_overrides,
                        )?;
                        let model = settings.model.clone().ok_or_else(|| {
                            anyhow::anyhow!(
                                "CSV instance missing model path after config resolution"
                            )
                        })?;
                        let property = settings.property.clone();

                        if !json {
                            info!(
                                "Instance {}/{}: model={}, property={}",
                                instance_index + 1,
                                instances.len(),
                                model.display(),
                                property
                                    .as_ref()
                                    .map(|path| path.display().to_string())
                                    .unwrap_or_else(|| "None".to_string())
                            );
                            if let Some(path) = settings.config_path.as_deref() {
                                info!("{}", config::config_path_hint(path));
                            }
                        }

                        let verify_config = commands::verify::VerificationConfig::builder(
                            model,
                            settings.epsilon,
                            settings.method.clone(),
                        )
                        .property(property)
                        .verification(
                            settings.mul_binary_relaxation,
                            settings.max_iterations,
                            settings.tolerance,
                            settings.timeout,
                        )
                        .backend(settings.backend, gpu)
                        .native(native)
                        .layernorm(conservative_layernorm, layernorm_mode, layernorm_norm_mode)
                        .modes(
                            layer_by_layer,
                            block_wise,
                            progress,
                            progress_json,
                            max_blocks,
                            checkpoint.clone(),
                        )
                        .output(json, strict, require_sound, allow_unknown)
                        .heuristics(
                            allow_heuristic_logsoftmax,
                            allow_heuristic_softmax,
                            settings.peel_off_last_softmax_layer,
                        )
                        .double_fp(double_fp || settings.double_fp, shrink_eps)
                        .build();
                        let result = commands::verify::handle_verify_command(verify_config);
                        handle_result(result)?;
                    }
                    return Ok(());
                }
            }

            let settings = config::resolve_verify_settings(config, root_path, cli_overrides)?;
            let model = settings.model.or(cli_model).ok_or_else(|| {
                anyhow::anyhow!("MODEL is required (positional, --model, or config)")
            })?;
            let property = settings.property;

            if !json {
                if let Some(path) = settings.config_path.as_deref() {
                    info!("{}", config::config_path_hint(path));
                }
            }
            let verify_config = commands::verify::VerificationConfig::builder(
                model,
                settings.epsilon,
                settings.method.clone(),
            )
            .property(property)
            .verification(
                settings.mul_binary_relaxation,
                settings.max_iterations,
                settings.tolerance,
                settings.timeout,
            )
            .backend(settings.backend, gpu)
            .native(native)
            .layernorm(conservative_layernorm, layernorm_mode, layernorm_norm_mode)
            .modes(
                layer_by_layer,
                block_wise,
                progress,
                progress_json,
                max_blocks,
                checkpoint,
            )
            .output(json, strict, require_sound, allow_unknown)
            .heuristics(
                allow_heuristic_logsoftmax,
                allow_heuristic_softmax,
                settings.peel_off_last_softmax_layer,
            )
            .double_fp(double_fp || settings.double_fp, shrink_eps)
            .build();
            let result = commands::verify::handle_verify_command(verify_config);
            handle_result(result)?;
        }

        Commands::Inspect {
            model,
            native,
            cost,
            timing_profile,
            json,
        } => {
            commands::inspect::handle_inspect_command(
                &model,
                native,
                cost,
                timing_profile.as_deref(),
                json,
            )?;
        }

        Commands::Coverage { json } => {
            commands::coverage::handle_coverage_command(json)?;
        }

        Commands::Lipschitz { model, json } => {
            commands::lipschitz::handle_lipschitz_command(&model, json)?;
        }

        Commands::Compare {
            reference,
            target,
            tolerance,
            epsilon,
            method,
            backend,
            gpu,
            verbose,
            json,
        } => {
            commands::inspect::handle_compare_command(
                &reference, &target, tolerance, epsilon, &method, backend, gpu, verbose, json,
            )?;
        }

        Commands::Diff {
            model_a,
            model_b,
            input,
            tolerance,
            layer_map,
            continue_after_divergence,
            diagnose,
            json,
        } => {
            commands::analysis::handle_diff_command(
                &model_a,
                &model_b,
                input.as_deref(),
                tolerance,
                layer_map.as_deref(),
                continue_after_divergence,
                diagnose,
                json,
            )?;
        }

        Commands::Sensitivity {
            model,
            epsilon,
            continue_after_overflow,
            threshold,
            json,
        } => {
            commands::analysis::handle_sensitivity_command(
                &model,
                epsilon,
                continue_after_overflow,
                threshold,
                json,
            )?;
        }

        Commands::QuantizeCheck {
            model,
            epsilon,
            continue_after_overflow,
            float16_only,
            int8_only,
            json,
        } => {
            let check_float16 = !int8_only;
            let check_int8 = !float16_only;
            commands::analysis::handle_quantize_check_command(
                &model,
                epsilon,
                continue_after_overflow,
                check_float16,
                check_int8,
                json,
            )?;
        }

        Commands::ProfileBounds {
            model,
            epsilon,
            continue_after_overflow,
            threshold,
            native,
            json,
            center_zeros,
        } => {
            commands::analysis::handle_profile_bounds_command(
                &model,
                epsilon,
                continue_after_overflow,
                threshold,
                native,
                json,
                center_zeros,
            )?;
        }

        Commands::Whisper {
            model,
            component,
            layer,
            epsilon,
            json,
        } => {
            commands::whisper::handle_whisper_command(model, component, layer, epsilon, json)?;
        }

        Commands::WhisperSeq {
            common,
            epsilon,
            mode,
            terminate_on_overflow,
            continue_after_overflow,
            overflow_clamp_value,
        } => {
            commands::whisper::handle_whisper_seq_command(
                common.model,
                common.start_block,
                common.end_block,
                common.include_stem,
                common.include_ln_post,
                common.batch,
                common.seq_len,
                common.n_mels,
                common.time,
                epsilon,
                common.backend,
                common.gpu,
                mode,
                common.max_bound_width,
                terminate_on_overflow,
                continue_after_overflow,
                overflow_clamp_value,
                common.reset_zonotope_blocks,
                common.json,
            )?;
        }

        Commands::WhisperSweep {
            common,
            epsilon_min,
            epsilon_max,
            steps,
            linear,
            mode,
            per_block,
        } => {
            commands::whisper::handle_whisper_sweep_command(
                common.model,
                common.start_block,
                common.end_block,
                common.include_stem,
                common.include_ln_post,
                common.batch,
                common.seq_len,
                common.n_mels,
                common.time,
                epsilon_min,
                epsilon_max,
                steps,
                linear,
                common.backend,
                common.gpu,
                mode,
                common.max_bound_width,
                common.reset_zonotope_blocks,
                per_block,
                common.json,
            )?;
        }

        Commands::WhisperEpsSearch {
            common,
            target_blocks,
            epsilon_min,
            epsilon_max,
            iterations,
            mode,
            verbose_search,
        } => {
            commands::whisper::handle_whisper_eps_search_command(
                common.model,
                common.start_block,
                common.end_block,
                target_blocks,
                common.include_stem,
                common.include_ln_post,
                common.batch,
                common.seq_len,
                common.n_mels,
                common.time,
                epsilon_min,
                epsilon_max,
                iterations,
                common.backend,
                common.gpu,
                mode,
                common.max_bound_width,
                common.reset_zonotope_blocks,
                verbose_search,
                common.json,
            )?;
        }

        Commands::Export {
            model_type,
            size,
            output,
        } => {
            commands::whisper::handle_export_command(model_type, size, output)?;
        }

        Commands::Bench(args) => {
            let subcommands::BenchArgs {
                benchmark,
                json,
                year,
                timeout,
                include_results,
                model_filter,
                property_filter,
                branching,
                max_domains,
                proactive_cuts,
                max_proactive_cuts,
                relaxed_clip,
                pgd_attack,
                pgd_restarts,
                gpu_bab,
                no_la_warm_start,
                backend,
                gpu,
            } = *args;
            if !json {
                info!("Running benchmark: {}", benchmark);
            }
            commands::bench::run_benchmarks(
                &benchmark,
                json,
                year,
                timeout,
                include_results,
                model_filter.as_deref(),
                property_filter.as_deref(),
                &branching,
                max_domains,
                proactive_cuts,
                max_proactive_cuts,
                relaxed_clip,
                pgd_attack,
                pgd_restarts,
                gpu_bab,
                no_la_warm_start,
                backend,
                gpu,
            )?;
        }

        Commands::VnncompAudit {
            year,
            timeout,
            json,
            verbose,
            category,
        } => {
            use commands::bench_vnncomp::{
                print_audit_summary, run_vnncomp_audit, VnncompAuditArgs,
            };

            let args = VnncompAuditArgs {
                year,
                timeout,
                json,
                category_filter: category,
            };

            match run_vnncomp_audit(args) {
                Ok(summary) => {
                    if json {
                        println!("{}", serde_json::to_string_pretty(&summary)?);
                    } else {
                        print_audit_summary(&summary, verbose);
                    }
                }
                Err(e) => {
                    if json {
                        println!(
                            "{}",
                            serde_json::json!({
                                "command": "vnncomp-audit",
                                "error": e.to_string()
                            })
                        );
                    } else {
                        eprintln!("VNN-COMP audit failed: {}", e);
                    }
                    anyhow::bail!("VNN-COMP audit failed: {}", e);
                }
            }
        }

        Commands::Benchmarks { action } => {
            commands::vnncomp_benchmarks::handle_benchmark_assets_command(action)?;
        }

        Commands::VnncompBenchmarks { years, all, json } => {
            commands::vnncomp_benchmarks::handle_vnncomp_benchmarks_command(years, all, json)?;
        }

        Commands::VnncompSubmit {
            output,
            no_build,
            dry_run,
            json,
        } => {
            commands::vnncomp_submit::handle_vnncomp_submit_command(
                output, no_build, dry_run, json,
            )?;
        }

        Commands::VnncompLateSubmit { action } => {
            commands::vnncomp_late_submit::handle_vnncomp_late_submit_command(action)?;
        }

        Commands::VnncompMatrix {
            year,
            tools,
            categories,
            sample_per_category,
            limit,
            timeout_override,
            skip_prepare,
            output_dir,
            json,
        } => {
            commands::vnncomp_matrix::handle_vnncomp_matrix_command(
                year,
                tools,
                categories,
                sample_per_category,
                limit,
                timeout_override,
                skip_prepare,
                output_dir,
                json,
            )?;
        }

        Commands::BetaCrown(args) => {
            let subcommands::BetaCrownArgs {
                model,
                property,
                preset,
                epsilon,
                threshold,
                peel_off_last_softmax_layer,
                allow_heuristic_logsoftmax,
                allow_heuristic_softmax,
                max_domains,
                timeout,
                max_depth,
                branching,
                fsb_candidates,
                no_alpha,
                alpha_iterations,
                input_split_alpha_iterations,
                input_split_lr_alpha,
                no_adaptive_alpha_skip,
                alpha_skip_depth,
                crown_ibp_intermediates,
                alpha_spsa_samples,
                alpha_lr,
                alpha_gradient_method,
                alpha_optimizer,
                invprop,
                invprop_apply,
                invprop_share_gammas,
                beta_iterations,
                beta_max_depth,
                lr_beta,
                crown_ibp,
                batch_size,
                sequential_children,
                enable_cuts,
                no_cuts,
                max_cuts,
                min_cut_depth,
                enable_near_miss_cuts,
                near_miss_margin,
                proactive_cuts,
                max_proactive_cuts,
                biccos_constraint_strengthening,
                biccos_drop_ratio,
                relaxed_clip,
                relaxed_clip_iterations,
                clip_interm_domain,
                clip_interm_topk,
                clip_in_alpha_crown,
                clip_interm_prune,
                clip_interm_use_final_layer,
                interm_transfer,
                pgd_attack,
                no_pgd_attack,
                pgd_restarts,
                pgd_steps,
                backend,
                gpu,
                input_split_metrics_jsonl,
                domain_batch_metrics_jsonl,
                json,
                gpu_bab,
                no_la_warm_start,
                complete_verifier,
                mip_solver,
                competition_mode,
                no_certificate,
                emit_certificate,
                allow_unsound_gpu_crown,
            } = *args;
            // PGD falsification is default-on; `--no-pgd-attack` (or
            // `--pgd-attack=false`) disables it. The explicit disable always wins.
            // SOUND: PGD only finds re-validated counterexamples, so toggling it
            // never changes a verified/unsat verdict.
            let pgd_attack = pgd_attack && !no_pgd_attack;
            // Resolve the proof/certificate policy. `--emit-certificate <path>`
            // forces emission ON (and sets the path); `--no-certificate` forces
            // it OFF; otherwise it is auto = NOT competition mode (ON by
            // default for interactive runs). SOUND: never affects the verdict.
            let emit_certificate_override = if emit_certificate.is_some() {
                Some(true)
            } else if no_certificate {
                Some(false)
            } else {
                None
            };
            let proof_opts = commands::beta_crown::ProofOpts {
                competition_mode,
                emit_certificate: emit_certificate_override,
                certificate_path: emit_certificate,
                allow_unsound_gpu_crown,
            };
            let beta_crown_outcome = commands::beta_crown::handle_beta_crown_command(
                model,
                property,
                preset,
                epsilon,
                threshold,
                peel_off_last_softmax_layer,
                allow_heuristic_logsoftmax,
                allow_heuristic_softmax,
                max_domains,
                timeout,
                max_depth,
                branching,
                fsb_candidates,
                no_alpha,
                alpha_iterations,
                input_split_alpha_iterations,
                input_split_lr_alpha,
                no_adaptive_alpha_skip,
                alpha_skip_depth,
                crown_ibp_intermediates,
                alpha_spsa_samples,
                alpha_lr,
                alpha_gradient_method,
                alpha_optimizer,
                invprop,
                invprop_apply,
                invprop_share_gammas,
                beta_iterations,
                beta_max_depth,
                lr_beta,
                crown_ibp,
                batch_size,
                sequential_children,
                enable_cuts,
                no_cuts,
                max_cuts,
                min_cut_depth,
                enable_near_miss_cuts,
                near_miss_margin,
                proactive_cuts,
                max_proactive_cuts,
                biccos_constraint_strengthening,
                biccos_drop_ratio,
                relaxed_clip,
                relaxed_clip_iterations,
                clip_interm_domain,
                clip_interm_topk,
                clip_in_alpha_crown,
                clip_interm_prune,
                clip_interm_use_final_layer,
                interm_transfer,
                pgd_attack,
                pgd_restarts,
                pgd_steps,
                backend,
                gpu,
                // Capability hint: default (compile-time wgpu availability). The
                // explicit human `--gpu` flag above is still an unconditional force.
                None,
                input_split_metrics_jsonl,
                domain_batch_metrics_jsonl,
                json,
                gpu_bab,
                no_la_warm_start,
                complete_verifier,
                mip_solver,
                proof_opts,
                commands::beta_crown::BetaCrownInstanceOverrides::default(),
            );
            // Verdict-emission guarantee (VNN-COMP): if verification errors out
            // (model-load failure, an internal propagation error during the
            // initial bound pass, etc.) we MUST still emit a valid competition
            // JSON verdict rather than dying with no output (which run_instance.sh
            // would score as "error"). A bare `unknown` is always sound — it never
            // claims a property is Verified. Without --json we keep the original
            // error so interactive/debug runs still surface the failure.
            if let Err(e) = beta_crown_outcome {
                if json {
                    println!(
                        "{}",
                        serde_json::json!({
                            "status": "unknown",
                            "reason": format!("verification aborted before producing a verdict: {e}"),
                            "counterexample": serde_json::Value::Null,
                            "counterexample_vnnlib": serde_json::Value::Null,
                        })
                    );
                } else {
                    return Err(e);
                }
            }
        }
        Commands::Vnncomp {
            version,
            category,
            onnx,
            vnnlib,
            results_file,
            timeout_secs,
            configs_dir,
        } => {
            commands::vnncomp::handle_vnncomp_command(
                version,
                category,
                onnx,
                vnnlib,
                results_file,
                timeout_secs,
                configs_dir,
            )?;
        }
        Commands::Weights { action } => {
            commands::weights::handle_weights_command(action)?;
        }
        Commands::Gt { action } => {
            // Exit codes mirror `ny verify`: 0 proved, 1 falsified, 2 unknown.
            let code = commands::gt::handle_gt_command(action)?;
            if code != 0 {
                std::process::exit(code);
            }
        }
        Commands::Tutorial { topic } => {
            commands::tutorial::run(topic.as_ref())?;
        }
    }

    Ok(())
}

/// Serve one ONNX Runtime shape-inference request over stdin/stdout.
///
/// This is the child half of [`ny_onnx::ShapeInferBackend::Subprocess`]
/// (selected by `commands::cli_shape_infer_backend` for CLI model loads): the
/// parent verifier streams the model bytes to our stdin and parses the
/// versioned shape table from our stdout. Running ORT's C++ shape inference
/// here means a native abort or fault kills only this short-lived child; the
/// parent observes the exit status and degrades to loading with no inferred
/// shapes. Errors exit non-zero through the normal `main() -> Result` path,
/// which the parent equally treats as inference-unavailable.
fn serve_shape_infer_subprocess() -> Result<()> {
    let mut stdin = std::io::stdin().lock();
    let mut stdout = std::io::stdout().lock();
    ny_onnx::serve_shape_infer_request(&mut stdin, &mut stdout)
        .context("shape-infer subprocess failed")
}
