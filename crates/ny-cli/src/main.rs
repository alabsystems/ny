// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

// Keep unsafe code denied throughout the CLI. The sole narrow exception is the
// `beta_crown::ay_tail_authority` verdict-capability bridge, whose unsafe calls
// attest that exact AY + ny-cert replay has already succeeded. Keeping that
// bridge in its own module makes the trust boundary mechanically auditable.
#![deny(unsafe_code)]
// Retained proof publications exceed the default auto-trait recursion budget.
#![recursion_limit = "256"]

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

// Use Accelerate for ndarray::dot() on macOS.
#[cfg(target_os = "macos")]
extern crate blas_src;

// Per-thread heaps reduce allocator contention during parallel bound propagation.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Command handlers live in sibling modules to keep bootstrap and execution concerns separate.
mod build_id;
mod commands;
mod compute_backend;
mod config;
mod flight;
#[cfg(feature = "mip")]
mod head_phase_advice;
mod plan_resolver;
mod preset;
// #lever-reachability: the typed preset ROUTE for env-gated verdict-affecting
// levers, and the single bridge that writes every promoted cell before any lane
// starts. `run_instance.sh` exports no `NY_*`, so a gate without a
// typed key is dead code on a scored run.
mod preset_gates;
// #router-r1 (ROUTER_PROGRAM R1): the shadow router's ny-cli glue — the one
// reader of `router.mode` / `optimizer.mode`, the capability registry, facts,
// telemetry and replay. It changes no verdict path.
mod router;
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CudaStartupAction {
    RegisterFactories,
    Disabled { requested: bool },
    Unavailable { requested: bool },
    NotCompiled { requested: bool },
}

const fn resolve_cuda_startup_action(
    compiled: bool,
    disabled: bool,
    engine_candidate: bool,
    requested: bool,
) -> CudaStartupAction {
    if !compiled {
        CudaStartupAction::NotCompiled { requested }
    } else if disabled {
        CudaStartupAction::Disabled { requested }
    } else if engine_candidate {
        CudaStartupAction::RegisterFactories
    } else {
        CudaStartupAction::Unavailable { requested }
    }
}

fn parse_exact_boolean_env(name: &str, raw: Option<&std::ffi::OsStr>) -> Result<bool> {
    match raw {
        None => Ok(false),
        Some(value) if value == std::ffi::OsStr::new("0") => Ok(false),
        Some(value) if value == std::ffi::OsStr::new("1") => Ok(true),
        Some(_) => anyhow::bail!("{name} must be exactly 0 or 1"),
    }
}

pub(crate) fn exact_boolean_env(name: &str) -> Result<bool> {
    parse_exact_boolean_env(name, std::env::var_os(name).as_deref())
}

#[cfg(test)]
#[test]
fn cuda_startup_action_matrix_covers_fallback_and_registration() {
    use CudaStartupAction::{Disabled, NotCompiled, RegisterFactories, Unavailable};

    let cases = [
        (true, false, true, false, RegisterFactories),
        (true, false, true, true, RegisterFactories),
        (true, true, false, false, Disabled { requested: false }),
        (true, true, false, true, Disabled { requested: true }),
        (true, false, false, false, Unavailable { requested: false }),
        (true, false, false, true, Unavailable { requested: true }),
        (false, false, false, false, NotCompiled { requested: false }),
        (false, false, false, true, NotCompiled { requested: true }),
    ];

    for (compiled, disabled, candidate, requested, expected) in cases {
        assert_eq!(
            resolve_cuda_startup_action(compiled, disabled, candidate, requested),
            expected,
            "compiled={compiled}, disabled={disabled}, candidate={candidate}, \
             requested={requested}"
        );
    }
}

#[cfg(test)]
#[test]
fn exact_boolean_environment_parser_rejects_malformed_and_non_unicode_values() {
    use std::ffi::OsStr;

    assert!(!parse_exact_boolean_env("TEST_GATE", None).expect("unset is false"));
    assert!(!parse_exact_boolean_env("TEST_GATE", Some(OsStr::new("0"))).expect("zero"));
    assert!(parse_exact_boolean_env("TEST_GATE", Some(OsStr::new("1"))).expect("one"));
    for invalid in ["", "true", "01", " 1", "1 "] {
        assert!(parse_exact_boolean_env("TEST_GATE", Some(OsStr::new(invalid))).is_err());
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        let non_unicode = std::ffi::OsString::from_vec(vec![0xff, b'1']);
        assert!(parse_exact_boolean_env("TEST_GATE", Some(&non_unicode)).is_err());
    }
}

type WgpuVerdictConstructor =
    fn(
        ny_gpu::WgpuVerdictRequest,
    ) -> std::result::Result<ny_gpu::WgpuDevice, ny_gpu::WgpuVerdictQualificationError>;

#[derive(Clone, Copy)]
enum WgpuCrownStartupAction {
    KeepCpu,
    RegisterVerdictFactory(WgpuVerdictConstructor),
}

fn resolve_wgpu_crown_startup_action(
    raw: Option<&std::ffi::OsStr>,
) -> Result<WgpuCrownStartupAction> {
    match raw {
        None => Ok(WgpuCrownStartupAction::KeepCpu),
        Some(value) if value == std::ffi::OsStr::new("0") => Ok(WgpuCrownStartupAction::KeepCpu),
        Some(value) if value == std::ffi::OsStr::new("auto") => Ok(WgpuCrownStartupAction::KeepCpu),
        Some(value) if value == std::ffi::OsStr::new("1") => Ok(
            WgpuCrownStartupAction::RegisterVerdictFactory(ny_gpu::WgpuDevice::new_for_verdict),
        ),
        Some(_) => anyhow::bail!("NY_WGPU_CROWN must be exactly auto, 0, or 1"),
    }
}

#[cfg(test)]
#[test]
fn wgpu_crown_startup_requires_exact_one_and_selects_the_verdict_constructor() {
    let WgpuCrownStartupAction::RegisterVerdictFactory(constructor) =
        resolve_wgpu_crown_startup_action(Some(std::ffi::OsStr::new("1"))).expect("one is valid")
    else {
        panic!("NY_WGPU_CROWN=1 must select the typed verdict constructor");
    };
    let expected: WgpuVerdictConstructor = ny_gpu::WgpuDevice::new_for_verdict;
    assert!(std::ptr::fn_addr_eq(constructor, expected));

    for raw in [None, Some("0"), Some("auto")] {
        assert!(matches!(
            resolve_wgpu_crown_startup_action(raw.map(std::ffi::OsStr::new)),
            Ok(WgpuCrownStartupAction::KeepCpu)
        ));
    }
    for invalid in ["", "true", " 1 ", "AUTO"] {
        assert!(resolve_wgpu_crown_startup_action(Some(std::ffi::OsStr::new(invalid))).is_err());
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        let non_unicode = std::ffi::OsString::from_vec(vec![0xff, b'1']);
        assert!(resolve_wgpu_crown_startup_action(Some(&non_unicode)).is_err());
    }
}

#[cfg(feature = "cuda")]
const CUDA_DEADLINE_F64_MARKER: &str = "NY_CUDA_DEADLINE_F64_GEMM_V1";

#[cfg(feature = "cuda")]
fn cuda_deadline_f64_telemetry_line(
    gemm: ny_cuda::CudaDeadlineF64GemmStats,
    admission: ny_propagate::sound_f64_gemm::DeadlineAdmissionStats,
) -> String {
    format!(
        "{CUDA_DEADLINE_F64_MARKER} calls={} dispatches={} wall_us={} \
         admission_ready={} admission_unavailable={} admission_timeouts={} \
         admission_wait_us={}",
        gemm.calls,
        gemm.dispatches,
        gemm.wall_us,
        admission.ready,
        admission.unavailable,
        admission.bounded_timeouts,
        admission.wait_us,
    )
}

/// Emit aggregate CUDA deadline telemetry only after the selected command has
/// completed. The authoritative GEMM/admission paths perform lock-free counter
/// updates only and therefore cannot block on a full diagnostic output pipe.
#[cfg(feature = "cuda")]
fn emit_cuda_deadline_f64_post_command_telemetry() {
    if std::env::var("NY_PHASE_TELEMETRY").ok().as_deref() == Some("1") {
        eprintln!(
            "{}",
            cuda_deadline_f64_telemetry_line(
                ny_cuda::cuda_deadline_f64_gemm_stats(),
                ny_propagate::sound_f64_gemm::deadline_admission_stats(),
            )
        );
        let packed = ny_cuda::packed_root::stats();
        eprintln!(
            "NY_PACKED_ROOT_CUDA_V1 completed={} dispatches={} wall_us={}",
            packed.completed, packed.dispatches, packed.wall_us
        );
    }
}

#[cfg(all(test, feature = "cuda"))]
#[test]
fn cuda_deadline_f64_telemetry_line_is_stable() {
    let line = cuda_deadline_f64_telemetry_line(
        ny_cuda::CudaDeadlineF64GemmStats {
            calls: 7,
            dispatches: 11,
            wall_us: 13,
        },
        ny_propagate::sound_f64_gemm::DeadlineAdmissionStats {
            ready: 2,
            unavailable: 3,
            bounded_timeouts: 5,
            wait_us: 17,
        },
    );
    assert_eq!(
        line,
        "NY_CUDA_DEADLINE_F64_GEMM_V1 calls=7 dispatches=11 wall_us=13 \
         admission_ready=2 admission_unavailable=3 admission_timeouts=5 admission_wait_us=17"
    );
}

fn main() -> std::process::ExitCode {
    match run_with_platform_stack() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {error:?}");
            std::process::ExitCode::from(commands::verify::exit_codes::ERROR as u8)
        }
    }
}

fn run_with_platform_stack() -> Result<()> {
    // Same-binary AY transport has no model, logger, device or ordinary solver
    // bootstrap. Bypass the Windows main-worker stack reservation as well.
    #[cfg(feature = "mip")]
    if head_phase_advice::helper_requested(std::env::args_os().skip(1))? {
        return head_phase_advice::run_helper();
    }
    #[cfg(feature = "mip")]
    if head_phase_advice::functional_helper_requested(std::env::args_os().skip(1))? {
        return head_phase_advice::run_functional_helper();
    }
    #[cfg(feature = "mip")]
    if head_phase_advice::phase_cover_helper_requested(std::env::args_os().skip(1))? {
        return head_phase_advice::run_phase_cover_helper();
    }

    // Recursive bound propagation can exhaust Windows' default main-thread
    // stack, so run the CLI on a worker with an explicit stack reservation.
    #[cfg(windows)]
    {
        // Windows commits reserved stack pages on demand.
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

    // #binary-identity: answer before clap, logging or backend probing.
    // --build-stamp uses compiled fields; --build-id also hashes the executable.
    // Per-instance receipts use the stamp to avoid executable I/O in the budget.
    {
        let mut args = std::env::args_os().skip(1);
        match args.next() {
            Some(first)
                if first == *std::ffi::OsStr::new("--build-id")
                    || first == *std::ffi::OsStr::new("build-id") =>
            {
                return build_id::run_cli(&args.collect::<Vec<_>>(), true);
            }
            Some(first)
                if first == *std::ffi::OsStr::new("--build-stamp")
                    || first == *std::ffi::OsStr::new("build-stamp") =>
            {
                return build_id::run_cli(&args.collect::<Vec<_>>(), false);
            }
            _ => {}
        }
    }

    // --build-info emits the derived version/features line and compiled stamp.
    // It performs no CUDA or adapter probe; runtime qualification is separate.
    if std::env::args_os().nth(1).as_deref() == Some(std::ffi::OsStr::new("--build-info")) {
        println!(
            "ny {} features: cuda={} mip={}",
            build_id::DISPLAY_VERSION,
            if cfg!(feature = "cuda") { "on" } else { "off" },
            if cfg!(feature = "mip") { "on" } else { "off" },
        );
        println!("{}", build_id::current_line());
        return Ok(());
    }

    // Hidden machine-readable CUDA runtime identity. Unlike `ldd`, `ldconfig`,
    // or an external loader probe, this observes the exact objects selected
    // inside the sealed NY process after cudarc has loaded and exercised the
    // driver/cuBLAS stack. Measurement provenance hashes these paths at start
    // and re-runs this command during completion.
    if std::env::args_os().nth(1).as_deref() == Some(std::ffi::OsStr::new("--cuda-runtime-info")) {
        #[cfg(feature = "cuda")]
        {
            let engine = match std::panic::catch_unwind(ny_cuda::CudaGemmEngine::new) {
                Ok(Ok(engine)) => engine,
                Ok(Err(error)) => {
                    anyhow::bail!("ny cuda runtime identity failed: {error}");
                }
                Err(_) => {
                    anyhow::bail!(
                        "ny cuda runtime identity failed: CUDA runtime loader panicked \
                         (missing or incompatible dynamic library)"
                    );
                }
            };
            engine
                .assert_deadline_f64_transport_bit_exact()
                .context("ny cuda runtime explicit-device deadline selfcheck failed")?;
            let identity = ny_cuda::cuda_runtime_identity()
                .context("ny cuda runtime mapped-object identity failed")?;
            let objects: Vec<_> = identity
                .objects
                .iter()
                .map(|object| {
                    serde_json::json!({
                        "role": object.role,
                        "provider_symbol": object.provider_symbol,
                        "mapped_path": object.mapped_path,
                        "resolved_path": object.resolved_path,
                        "mapped_device_major": object.mapped_device_major,
                        "mapped_device_minor": object.mapped_device_minor,
                        "mapped_inode": object.mapped_inode,
                        "size_bytes": object.size_bytes,
                        "sha256": object.sha256,
                        "fingerprint": {
                            "device": object.fingerprint.device,
                            "inode": object.fingerprint.inode,
                            "size_bytes": object.fingerprint.size_bytes,
                            "mtime_ns": object.fingerprint.mtime_ns,
                            "ctime_ns": object.fingerprint.ctime_ns,
                        },
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::json!({
                    "schema": ny_cuda::CUDA_RUNTIME_INFO_SCHEMA,
                    "device_name": engine.device_name(),
                    "pageable_host_ptr": engine.host_ptr_zero_copy(),
                    "pageable_memory_access": engine.pageable_memory_access(),
                    "pageable_access_uses_host_page_tables": engine.pageable_access_uses_host_page_tables(),
                    "integrated_device": engine.integrated_device(),
                    "ordinary_gemm_transport": engine.ordinary_gemm_transport_name(),
                    "ordinary_gemm_transport_policy": engine.ordinary_gemm_transport_policy_name(),
                    "ordinary_gemm_transport_reason": engine.ordinary_gemm_transport_reason(),
                    "explicit_device_copy": engine.discrete_mode_enabled(),
                    "discrete_mode": engine.discrete_mode_enabled(),
                    "deadline_f64_transport": engine.deadline_f64_transport_name(),
                    "candidates": {
                        "driver": identity.candidates.driver,
                        "cublas": identity.candidates.cublas,
                        "cublas_lt": identity.candidates.cublas_lt,
                        "nvrtc": identity.candidates.nvrtc,
                    },
                    "objects": objects,
                    "nvrtc_status": identity.nvrtc_status,
                })
            );
            return Ok(());
        }
        #[cfg(not(feature = "cuda"))]
        anyhow::bail!("ny cuda runtime identity failed: binary was built without the cuda feature");
    }

    // Runtime CUDA qualification (`ny --cuda-selfcheck`). `--build-info` proves
    // only that the feature was compiled; cudarc loads libcuda/libcublas at
    // runtime, so a missing or version-mismatched library could otherwise make a
    // supposedly GPU-qualified scorecard silently fall back to CPU. Construction
    // performs the on-device bit-exact Sgemm/Dgemm known-answer probes.
    if std::env::args_os().nth(1).as_deref() == Some(std::ffi::OsStr::new("--cuda-selfcheck")) {
        #[cfg(feature = "cuda")]
        {
            match std::panic::catch_unwind(ny_cuda::CudaGemmEngine::new) {
                Ok(Ok(engine)) => {
                    engine
                        .assert_deadline_f64_transport_bit_exact()
                        .context("ny cuda selfcheck explicit-device deadline KAT failed")?;
                    println!(
                        "ny cuda selfcheck: ok device={:?} pageable_host_ptr={} \
                         pageable_memory_access={} host_page_tables={} integrated_device={} \
                         ordinary_gemm_transport={} transport_policy={} transport_reason={} \
                         explicit_device_copy={} \
                         deadline_f64_transport={}",
                        engine.device_name(),
                        engine.host_ptr_zero_copy(),
                        engine.pageable_memory_access(),
                        engine.pageable_access_uses_host_page_tables(),
                        engine.integrated_device_state(),
                        engine.ordinary_gemm_transport_name(),
                        engine.ordinary_gemm_transport_policy_name(),
                        engine.ordinary_gemm_transport_reason(),
                        engine.discrete_mode_enabled(),
                        engine.deadline_f64_transport_name(),
                    );
                    return Ok(());
                }
                Ok(Err(error)) => {
                    anyhow::bail!("ny cuda selfcheck failed: {error}");
                }
                Err(_) => {
                    anyhow::bail!(
                        "ny cuda selfcheck failed: CUDA runtime loader panicked \
                         (missing or incompatible dynamic library)"
                    );
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        anyhow::bail!("ny cuda selfcheck failed: binary was built without the cuda feature");
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

    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            let clap_exit_code = error.exit_code();
            error
                .print()
                .context("failed to print command-line error")?;
            if clap_exit_code == 0 {
                return Ok(());
            }
            std::process::exit(commands::verify::exit_codes::ERROR);
        }
    };
    let (verbose, log_format, command) = cli.into_parts();
    // Global Clap options may precede this hidden command. Keep that route
    // just as free of normal backend/factory/subscriber setup as the exact
    // subprocess entry above.
    #[cfg(feature = "mip")]
    if matches!(&command, Commands::InternalHeadPhaseAyAdvice) {
        return head_phase_advice::run_helper();
    }
    #[cfg(feature = "mip")]
    if matches!(&command, Commands::InternalHeadFunctionalAyAdvice) {
        return head_phase_advice::run_functional_helper();
    }
    #[cfg(feature = "mip")]
    if matches!(&command, Commands::InternalHeadPhaseCoverAyAdvice) {
        return head_phase_advice::run_phase_cover_helper();
    }
    // Proof commands resolve and qualify their WGPU device in the handler.
    // Defer the auxiliary FL-value factory to avoid a rate probe creating a
    // second context before that decision; CPU/fallback handlers install it.
    let command_resolves_proof_backend = matches!(
        &command,
        Commands::Verify(_) | Commands::BetaCrown(_) | Commands::Vnncomp { .. }
    );
    #[cfg(feature = "cuda")]
    let prewarm_fast_f32_for_vnncomp = matches!(&command, Commands::Vnncomp { .. });

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

    // CUDA admission checks the feature, visible device, and loadable
    // driver/cuBLAS candidates. The engine later validates its retained
    // providers and device arithmetic; CROWN callers charge γ_n·S for Dgemm.
    let cuda_crown_requested = exact_boolean_env("NY_CUDA_CROWN")?;
    // These overrides are startup-admission hints. `ny-cuda` remains
    // authoritative over exact parsing, compatibility with the detected
    // topology, and the final transport selection.
    let cuda_discrete_override = ny_levers::read(&ny_levers::decls::cuda::CUDA_DISCRETE_MODE)
        .value
        .as_bool();
    let cuda_acceleration_requested =
        ny_propagate::sound_gpu_gate::wide_sound_gpu_crown_requested()
            || cuda_crown_requested
            || ny_levers::read_presence(&ny_levers::decls::cuda::CUDA_GEMM_TRANSPORT)
            || cuda_discrete_override;
    #[cfg(feature = "cuda")]
    let cuda_disabled = std::env::var_os("NY_NO_CUDA").is_some();
    #[cfg(feature = "cuda")]
    let cuda_engine_candidate = !cuda_disabled && compute_backend::detect().cuda_engine_candidate;
    #[cfg(feature = "cuda")]
    let cuda_startup_action = resolve_cuda_startup_action(
        true,
        cuda_disabled,
        cuda_engine_candidate,
        cuda_acceleration_requested,
    );
    #[cfg(not(feature = "cuda"))]
    let cuda_startup_action =
        resolve_cuda_startup_action(false, false, false, cuda_acceleration_requested);
    #[cfg(feature = "cuda")]
    if matches!(cuda_startup_action, CudaStartupAction::RegisterFactories) {
        // Share one lazy CUDA context and cuBLAS handle across GEMM and resident
        // CROWN. VNN-COMP prewarms the enabled f32 lane before its deadline.
        use std::sync::{Arc, OnceLock};
        static CUDA_ENGINE: OnceLock<Option<Arc<ny_cuda::CudaGemmEngine>>> = OnceLock::new();
        fn shared_cuda_engine() -> Option<Arc<dyn ny_core::GemmEngine>> {
            CUDA_ENGINE
                .get_or_init(|| {
                    // cudarc can panic on missing symbols even after dlopen
                    // succeeds. Unwinding maps that to CPU fallback; native
                    // faults or hangs still require process isolation.
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
                                info!("CUDA GEMM/CROWN unavailable; using CPU f64 path ({e})");
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
                                info!("CUDA GEMM/CROWN init panicked; using CPU f64 path ({msg})");
                            }
                            None
                        }
                    }
                })
                .clone()
                .map(|e| e as Arc<dyn ny_core::GemmEngine>)
        }
        ny_propagate::sound_f64_gemm::set_sound_f64_gemm_factory(shared_cuda_engine);
        // Scheduling metadata for structural raw loading only; the original
        // root later requires a ready engine with both checked capabilities.
        ny_propagate::beta_crown::certified_root_cut::note_checked_root_cut_factory_registered();
        // Share f32 GEMM with IBP, PGD, and BaB. Its pinned math mode supports
        // the bound callers' ULP widening. NY_NO_CUDA_F32 disables only f32.
        if std::env::var_os("NY_NO_CUDA_F32").is_none() {
            ny_propagate::fast_f32_gemm::set_fast_f32_gemm_factory(shared_cuda_engine);
            if prewarm_fast_f32_for_vnncomp {
                if ny_propagate::fast_f32_gemm::prewarm_fast_f32_gemm() {
                    info!("CUDA fast f32 GEMM materialized before VNN-COMP deadline authority");
                } else {
                    info!(
                        "CUDA fast f32 GEMM prewarm unavailable; finite calls will fail closed to \
                         their existing engine/CPU path"
                    );
                }
            } else {
                info!("CUDA fast f32 GEMM factory registered lazily for non-VNN-COMP command");
            }
        } else {
            info!("NY_NO_CUDA_F32 set; f32 GEMM engine offload disabled");
        }
        // The separate wide CROWN factory is opt-in via NY_CUDA_WIDE or
        // NY_HYDRA_CROWN. Prewarm a requested route before verifier deadlines.
        ny_propagate::sound_gpu_gate::set_wide_sound_gpu_crown_factory(shared_cuda_engine);
        if ny_propagate::sound_gpu_gate::wide_sound_gpu_crown_requested() {
            let ready = ny_propagate::sound_gpu_gate::prewarm_wide_sound_gpu_crown();
            if ready {
                warn!(
                    "CUDA wide CROWN enabled: backend materialized before command deadline \
                     authority (local/CPU fallback retained)"
                );
            } else {
                warn!(
                    "CUDA wide CROWN enabled but prewarm was unavailable; finite calls will \
                     fail closed to the local/CPU path"
                );
            }
        } else {
            info!("CUDA wide CROWN registered but disabled; set NY_CUDA_WIDE=1 after A/B qualification");
        }
        // Preserve the legacy opt-in for non-wide, host-orchestrated CUDA CROWN,
        // which can be slower/weaker than CPU f64 on small networks.
        if cuda_crown_requested {
            ny_propagate::sound_gpu_gate::set_sound_gpu_crown_factory(shared_cuda_engine);
            if ny_propagate::sound_gpu_gate::prewarm_sound_gpu_crown() {
                info!(
                    "NY_CUDA_CROWN=1; ordinary sound CUDA CROWN materialized before command \
                     deadline authority"
                );
            } else {
                warn!(
                    "NY_CUDA_CROWN=1 but prewarm was unavailable; finite ordinary CROWN calls \
                     will fail closed to the local/CPU path"
                );
            }
        }
    }

    // Proof handlers resolve NY_WGPU_CROWN=auto|0|1 on their retained device.
    // Other commands require the literal 1, qualified by WgpuVerdictRequest.
    // Requested CUDA CROWN registers first and takes precedence. Prewarm here
    // because finite calls only use initialized engines; proof handlers own theirs.
    if !command_resolves_proof_backend {
        match resolve_wgpu_crown_startup_action(std::env::var_os("NY_WGPU_CROWN").as_deref())? {
            WgpuCrownStartupAction::KeepCpu => {}
            WgpuCrownStartupAction::RegisterVerdictFactory(constructor) => {
                ny_propagate::sound_gpu_gate::set_sound_gpu_crown_factory(move || {
                    constructor(ny_gpu::WgpuVerdictRequest::new())
                        .ok()
                        .map(|device| {
                            std::sync::Arc::new(device) as std::sync::Arc<dyn ny_core::GemmEngine>
                        })
                });
                // Wide CROWN uses the same qualified constructor, independently
                // gated by NY_CUDA_WIDE/NY_HYDRA_CROWN despite their CUDA names.
                ny_propagate::sound_gpu_gate::set_wide_sound_gpu_crown_factory(move || {
                    constructor(ny_gpu::WgpuVerdictRequest::new())
                        .ok()
                        .map(|device| {
                            std::sync::Arc::new(device) as std::sync::Arc<dyn ny_core::GemmEngine>
                        })
                });
                if ny_propagate::sound_gpu_gate::wide_sound_gpu_crown_requested()
                    && !ny_propagate::sound_gpu_gate::prewarm_wide_sound_gpu_crown()
                {
                    warn!(
                        "NY_WGPU_CROWN=1 with the wide lane requested, but the wide WGPU \
                     prewarm was unavailable; wide CROWN calls fail closed to the \
                     serial/CPU path"
                    );
                }
                if ny_propagate::sound_gpu_gate::prewarm_sound_gpu_crown() {
                    warn!(
                        "NY_WGPU_CROWN=1: sound WGPU CROWN backend materialized before command \
                     deadline authority (authority-gated per adapter; CPU fallback retained)"
                    );
                } else {
                    warn!(
                        "NY_WGPU_CROWN=1 set but the WGPU sound CROWN prewarm was unavailable \
                     (adapter missing or the authority ladder refused); finite CROWN calls \
                     fail closed to the CPU path"
                    );
                }
            }
        }
    }

    // NY_MARGIN_ROW_GPU_BACKWARD=1 lazily creates a dedicated device. A
    // construction failure disables that channel and retains CPU backward.
    ny_propagate::margin_row::set_margin_row_lane_gpu_factory(|| {
        ny_gpu::MarginRowLaneDevice::new().ok().map(|device| {
            std::sync::Arc::new(device) as std::sync::Arc<dyn ny_core::MarginRowLaneGpu>
        })
    });

    // The head-group provider is optional advice. Register only a lazy factory
    // here; its first live armed complete-cover call binds this executable.
    #[cfg(feature = "mip")]
    head_phase_advice::install_lazy_factory();

    // joint_f32 root search proposes hints: slopes pass the certify-once gate,
    // and dual multipliers pass the certified node evaluator. Ascent is default.
    ny_propagate::margin_row::set_root_search_backend_factory(|| {
        Some(std::sync::Arc::new(ny_bbe::CpuF32Backend::new())
            as std::sync::Arc<dyn ny_core::search::SearchBackend>)
    });

    // Graph joint_f32 parameters pass the clamp gate and a certified fold;
    // keep them only if they improve the incumbent. Ascent remains default.
    ny_propagate::network::graph_root_search::set_graph_root_search_backend_factory(|| {
        Some(std::sync::Arc::new(ny_bbe::CpuF32Backend::new())
            as std::sync::Arc<dyn ny_core::search::SearchBackend>)
    });

    // Frozen-dual search defaults to f64. Optional f32 backends propose hints;
    // the tier certifies every prune in f64 and retains fallback on refusal.
    ny_propagate::margin_row::set_frozen_dual_search_backend_factory(
        ny_propagate::margin_row::FdBackendKind::CpuF32,
        || {
            Some(std::sync::Arc::new(ny_bbe::CpuF32Backend::new())
                as std::sync::Arc<dyn ny_core::search::SearchBackend>)
        },
    );
    // Packed root arithmetic owns a separately qualified driver-only context.
    // Prewarm before VNN-COMP deadlines; a finite call never initializes it.
    #[cfg(feature = "cuda")]
    if !cuda_disabled {
        ny_propagate::margin_row::packed_root_engine::set_factory(|| {
            match ny_cuda::packed_root::PackedRootEngine::new(0) {
                Ok(Some(engine)) => Some(std::sync::Arc::new(engine)
                    as std::sync::Arc<dyn ny_core::packed_root_conv::Engine>),
                Ok(None) => None,
                Err(error) => {
                    warn!(%error, "packed root CUDA qualification unavailable; using CPU contraction");
                    None
                }
            }
        });
        if prewarm_fast_f32_for_vnncomp {
            let ready = ny_propagate::margin_row::packed_root_engine::prewarm();
            info!(ready, "packed root CUDA startup qualification");
        }
    }

    // Level and wave search share one lazy CUDA context and fault latch.
    // They need libcuda only, so NY_NO_CUDA applies but cuBLAS admission does
    // not. Without a backend, cuda falls back to f64 and auto to cpu_f32.
    #[cfg(feature = "cuda")]
    let device_search =
        (!cuda_disabled).then(|| std::sync::Arc::new(ny_cuda::search::CudaSearchBackend::new()));
    #[cfg(feature = "cuda")]
    if let Some(device) = device_search.as_ref().map(std::sync::Arc::clone) {
        ny_propagate::margin_row::set_frozen_dual_search_backend_factory(
            ny_propagate::margin_row::FdBackendKind::Cuda,
            move || {
                Some(std::sync::Arc::clone(&device)
                    as std::sync::Arc<dyn ny_core::search::SearchBackend>)
            },
        );
    }

    // Wave search hints require the lane's certified evaluator. Refusal uses
    // the legacy tree, which is also the default.
    ny_propagate::margin_row::set_wave_search_backend_factory(|| {
        Some(std::sync::Arc::new(ny_bbe::CpuF32Backend::new())
            as std::sync::Arc<dyn ny_core::search::SearchBackend>)
    });
    // wave_cuda uses the shared device, then CPU search, then the legacy tree.
    #[cfg(feature = "cuda")]
    if let Some(device) = device_search {
        ny_propagate::margin_row::set_wave_cuda_search_backend_factory(move || {
            Some(std::sync::Arc::clone(&device)
                as std::sync::Arc<dyn ny_core::search::SearchBackend>)
        });
    }

    // Low-dimensional batch search re-bounds every hint with its f64 forms;
    // refusal retains the default legacy bounder.
    ny_propagate::lowdim::set_forward_linear_search_backend_factory(|| {
        Some(std::sync::Arc::new(ny_bbe::CpuF32Backend::new())
            as std::sync::Arc<dyn ny_core::search::SearchBackend>)
    });

    match cuda_startup_action {
        CudaStartupAction::RegisterFactories => {}
        CudaStartupAction::Disabled { requested: true } => warn!(
            "CUDA proof acceleration requested, but NY_NO_CUDA is set; CUDA factories are \
             unavailable and the fail-closed CPU path remains active"
        ),
        CudaStartupAction::Disabled { requested: false } => {
            info!("NY_NO_CUDA set; CUDA acceleration disabled (CPU f64 path)");
        }
        CudaStartupAction::Unavailable { requested: true } => warn!(
            "CUDA GEMM/CROWN requested, but startup admission found no device with loadable \
             libcuda/libcublas candidates; these paths use CPU fallback"
        ),
        CudaStartupAction::Unavailable { requested: false } => info!(
            "CUDA GEMM/CROWN admission found no device with loadable libcuda/libcublas \
             candidates; these paths use CPU fallback"
        ),
        CudaStartupAction::NotCompiled { requested: true } => warn!(
            "CUDA proof acceleration requested, but this binary lacks the `cuda` feature; CUDA \
             factories are unavailable and the fail-closed CPU path remains active"
        ),
        CudaStartupAction::NotCompiled { requested: false } => {}
    }

    // Apple Accelerate is opt-in via NY_ACCELERATE_F64/F32; NY_NO_ACCELERATE
    // disables both. The engine probes BLAS arithmetic and guards each call;
    // bound callers still charge rounding error. Register after CUDA and
    // before the CPU floor because the first factory wins.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        use std::sync::Arc;
        let (engine, outcome) = ny_accelerate::resolve_for_install();
        match (&engine, &outcome) {
            (Some(engine), ny_accelerate::InstallOutcome::Installed { summary, .. }) => {
                let shared = Arc::clone(engine) as Arc<dyn ny_core::GemmEngine>;
                if ny_accelerate::f64_seam_armed() {
                    let for_f64 = Arc::clone(&shared);
                    ny_propagate::sound_f64_gemm::set_sound_f64_gemm_factory(move || {
                        Some(Arc::clone(&for_f64))
                    });
                    warn!("ARMED sound f64 CROWN GEMM seam via Accelerate — {summary}");
                }
                if engine.f32_via_accelerate() {
                    let for_f32 = Arc::clone(&shared);
                    ny_propagate::fast_f32_gemm::set_fast_f32_gemm_factory(move || {
                        Some(Arc::clone(&for_f32))
                    });
                    info!("Accelerate f32 GEMM enabled for IBP, PGD, and BaB");
                }
            }
            (_, ny_accelerate::InstallOutcome::ProbeRefused(failures)) => {
                warn!(
                    ?failures,
                    "Accelerate conformance probe REFUSED this host's BLAS; keeping the \
                     incumbent faer engines (fail-closed)"
                );
            }
            (_, ny_accelerate::InstallOutcome::Disabled) => {
                info!("NY_NO_ACCELERATE set; Accelerate seam disabled");
            }
            _ => {}
        }
    }

    // Fill unclaimed GEMM factories with CPU engines, including the Patches
    // batched path. NY_CPU_GEMM_ENGINE=0 disables the f32 registration.
    ny_propagate::faer_parallelism::install_cpu_gemm_engine_if_absent();

    // NY_CPU_SOUND_F64_ENGINE=0 disables the corresponding f64 registration.
    ny_propagate::faer_parallelism::install_cpu_sound_f64_gemm_engine_if_absent();

    // Non-proof commands register FL-value acceleration before rate probes.
    // Proof handlers register after resolving ownership of their WGPU device.
    if !command_resolves_proof_backend && !compute_backend::detect().wgpu_probe_skipped {
        ny_propagate::fl_value_gemm::set_fl_value_gemm_factory(|| {
            match ny_gpu::FlValueGemmDevice::new_wgpu() {
                Ok(device) => {
                    Some(std::sync::Arc::new(device) as std::sync::Arc<dyn ny_core::GemmEngine>)
                }
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "wgpu unavailable for forward-linear value GEMMs; using the CPU"
                    );
                    None
                }
            }
        });
    }

    match command {
        #[cfg(feature = "mip")]
        Commands::InternalHeadPhaseAyAdvice => {
            unreachable!("head-phase helper must return before normal startup")
        }
        #[cfg(feature = "mip")]
        Commands::InternalHeadFunctionalAyAdvice => {
            unreachable!("head-functional helper must return before normal startup")
        }
        #[cfg(feature = "mip")]
        Commands::InternalHeadPhaseCoverAyAdvice => {
            unreachable!("head-phase-cover helper must return before normal startup")
        }
        Commands::Verify(args) => {
            let json_output = args.json;
            let handle_result = |result: Result<()>| -> Result<()> {
                if let Err(err) = result {
                    if json_output {
                        // `--json` must be schema-stable even on failure (#395).
                        // Avoid the top-level error renderer writing non-JSON text.
                        let error_output: serde_json::Value =
                            if let Some(json_err) = commands::find_json_cli_error(&err) {
                                json_err.payload().clone()
                            } else {
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
                        std::process::exit(commands::verify::exit_codes::ERROR);
                    }
                    return Err(err);
                }
                Ok(())
            };
            let verify_result = (|| -> Result<()> {
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

                let resolved_config =
                    config::resolve_verify_config(config.clone(), root_path.clone())?;
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
                            .backend_request(settings.backend, gpu, settings.backend_automatic)
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
                .backend_request(settings.backend, gpu, settings.backend_automatic)
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
                Ok(())
            })();
            handle_result(verify_result)?;
        }

        Commands::HeadVerify {
            request,
            replay,
            evidence,
            timeout,
            max_head_mb,
        } => {
            commands::head_verify::run(
                &request,
                replay.as_deref(),
                evidence.as_deref(),
                timeout,
                max_head_mb,
            )?;
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
                max_queue_bytes,
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
                proof_profile,
                competition_mode,
                require_certificate,
                no_certificate,
                emit_certificate,
                allow_unsound_gpu_crown,
            } = *args;
            // PGD falsification is default-on; `--no-pgd-attack` (or
            // `--pgd-attack=false`) disables it. The explicit disable always wins.
            // SOUND: PGD only finds re-validated counterexamples, so toggling it
            // never changes a verified/unsat verdict.
            let pgd_attack = pgd_attack && !no_pgd_attack;
            // Flags request or suppress an optional certificate attempt;
            // otherwise the proof profile supplies the default. Independently,
            // `--require-certificate` restricts verdicts to representable proofs
            // and overrides suppression through ProofOpts.
            let emit_certificate_override = if emit_certificate.is_some() {
                Some(true)
            } else if no_certificate {
                Some(false)
            } else {
                None
            };
            // #soundness-spine M2a: this entry point resolves to `proof`;
            // `--proof-profile` (or the deprecated `--competition-mode`)
            // overrides it. Frozen once, printed once.
            let profile_override = proof_profile.or_else(|| {
                competition_mode.then_some(ny_core::proof_profile::ProofProfile::Competition)
            });
            let profile = ny_core::proof_profile::freeze(
                ny_core::proof_profile::ResolvedProofProfile::resolve(
                    ny_core::proof_profile::EntryPoint::Interactive,
                    profile_override,
                ),
            );
            let _ = ny_core::proof_profile::announce(profile);
            let proof_opts = commands::beta_crown::ProofOpts {
                profile,
                require_certificate,
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
                max_queue_bytes,
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
                false, // direct beta-crown has no post-BaB attack consumer
                commands::beta_crown::BetaCrownInstanceOverrides::default(),
            );
            // Direct CLI operational failures are not verification outcomes.
            // Keep verdict codes 0-3 reserved for results produced by the
            // verifier. The `vnncomp` command owns its separate fail-closed
            // translation from operational failures to protocol `unknown`.
            if let Err(e) = beta_crown_outcome {
                if json {
                    println!(
                        "{}",
                        serde_json::json!({
                            "error": "beta_crown_failed",
                            "message": e.to_string(),
                        })
                    );
                    std::process::exit(commands::verify::exit_codes::ERROR);
                } else {
                    return Err(e);
                }
            }
        }
        Commands::Vnncomp { action } => match action {
            subcommands::VnncompAction::V1 {
                category,
                onnx,
                vnnlib,
                results_file,
                timeout_secs,
                configs_dir,
                proof_profile,
                phase_cost,
            } => {
                // #phase-cost: the cost census's detail level for this process
                // (default coarse). Observability only — no verdict path reads it.
                ny_core::phase_cost::set_level(
                    phase_cost.unwrap_or(ny_core::phase_cost::DEFAULT_LEVEL),
                );
                // #soundness-spine M2a: the scored entry point resolves to
                // `competition`; `--proof-profile` overrides it for banking and
                // moat runs. Frozen here, before any lane can read the rates;
                // printed and recorded inside the handler once the flight
                // recorder is armed.
                ny_core::proof_profile::freeze(
                    ny_core::proof_profile::ResolvedProofProfile::resolve(
                        ny_core::proof_profile::EntryPoint::VnncompV1,
                        proof_profile,
                    ),
                );
                // The handler keeps its own protocol-version check; the clap
                // subcommand name IS the version string, so pass it verbatim.
                commands::vnncomp::handle_vnncomp_command(
                    "v1".to_string(),
                    category,
                    onnx,
                    vnnlib,
                    results_file,
                    timeout_secs,
                    configs_dir,
                )?;
                // #attack-steering-segv: the verdict and the flight sidecar are
                // durable at this point, so nothing is left to flush and every
                // still-live thread (attack-steering arming inside the Vulkan
                // driver, ORT pools, CUDA, AY workers) is pure teardown risk.
                // Returning from here would end the process through libc
                // `exit`, whose `_dl_fini` runs the NVIDIA driver's destructors
                // CONCURRENTLY with those threads — measured as SIGSEGV in
                // `ny-attack-arming`, SIGABRT in fini's own `free`, and one
                // hang that cost a published `sat`. Ending with `_exit` instead
                // makes that race unreachable by construction.
                //
                // The post-command telemetry line still has to be emitted, so
                // it happens here rather than at the end of `run`.
                #[cfg(feature = "cuda")]
                emit_cuda_deadline_f64_post_command_telemetry();
                commands::vnncomp::exit_scored_instance_without_teardown();
            }
            subcommands::VnncompAction::Plan {
                category,
                onnx,
                vnnlib,
                budget_secs,
                configs_dir,
                json,
                router,
                router_target_host,
                replay,
            } => {
                if let Some(flight_json) = replay {
                    // #router-r1: `--replay` re-runs the router's pure plan and
                    // checkpoints on a flight record; no instance is resolved.
                    commands::vnncomp_plan::handle_router_replay(&flight_json)?;
                } else {
                    let (Some(category), Some(onnx), Some(vnnlib), Some(budget_secs)) =
                        (category, onnx, vnnlib, budget_secs)
                    else {
                        anyhow::bail!(
                            "ny vnncomp plan needs CATEGORY ONNX VNNLIB BUDGET (or --replay FLIGHT_JSON)"
                        );
                    };
                    commands::vnncomp_plan::handle_vnncomp_plan_command(
                        &category,
                        &onnx,
                        &vnnlib,
                        budget_secs,
                        configs_dir,
                        json,
                        router,
                        router_target_host.as_deref(),
                    )?;
                }
            }
            subcommands::VnncompAction::Census {
                manifest,
                out,
                envelope,
                router_target_host,
                row_deadline_secs,
                work_dir,
            } => {
                // #admission-census: CR-RS's admission per routable capability
                // over a corpus, as if each were sealed. No verification runs.
                let target_host = router_target_host
                    .as_deref()
                    .map(|token| {
                        ny_core::router::facts::HostClass::from_token(token).ok_or_else(|| {
                            anyhow::anyhow!(
                                "--router-target-host must be one of gb10, g5, m5, other; got {token:?}"
                            )
                        })
                    })
                    .transpose()?;
                let work_dir = work_dir.unwrap_or_else(|| {
                    let mut name = out.clone().into_os_string();
                    name.push(".work");
                    std::path::PathBuf::from(name)
                });
                let options = router::census::CensusOptions {
                    manifest,
                    envelope: router::census::parse_envelope(&envelope)?,
                    target_host,
                    row_deadline: std::time::Duration::from_secs(row_deadline_secs.max(1)),
                    work_dir,
                    out,
                };
                let summary = router::census::handle_census_command(&options)?;
                println!("{}", router::census::done_line(&summary, &options.out));
            }
        },
        Commands::VnncompResearch { action } => {
            commands::vnncomp::handle_vnncomp_research_command(action)?;
        }
        Commands::Weights { action } => {
            commands::weights::handle_weights_command(action)?;
        }
        Commands::Gt { action } => {
            // Verdict codes mirror `ny verify`: 0 proved, 1 falsified, 2 unknown.
            // Handler errors propagate to the top-level operational code 4.
            let code = commands::gt::handle_gt_command(action)?;
            if code != 0 {
                std::process::exit(code);
            }
        }
        Commands::Tutorial { topic } => {
            commands::tutorial::run(topic.as_ref())?;
        }
        Commands::ProofCheck(args) => {
            // ENGINE_PROGRAM §1.7 B2.1: the report's verdict code IS the exit
            // code (0 verified; 10/20/21/30/31/40 otherwise). An unreadable
            // input propagates to the top-level operational code 4.
            let code = commands::proof_check::handle_proof_check(&args)?;
            if code != 0 {
                std::process::exit(code);
            }
        }
        Commands::Claims(args) => {
            // The running process already knows its own compiled stamp, so
            // pass it rather than re-invoking `target/release/ny` by path:
            // the whole point of #binary-identity is that a path is not an
            // identity. `BuildStamp::current()` is free (no file hashing).
            let stamp = build_id::current_line();
            let binary = ny_claims::BinaryFacts::from_identity_line(stamp, "the running ny");
            let code = ny_claims::cli::run(&args, binary)?;
            if code != 0 {
                std::process::exit(code);
            }
        }
    }

    #[cfg(feature = "cuda")]
    emit_cuda_deadline_f64_post_command_telemetry();

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
