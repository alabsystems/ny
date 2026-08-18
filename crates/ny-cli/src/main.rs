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
mod compute_backend;
mod config;
mod flight;
mod plan_resolver;
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
    // Proof-capable commands decide WGPU ownership only after resolving their
    // CLI/config/preset request and running live qualification. Do not install
    // the independent lazy FL-value context before that decision: a plan-rate
    // probe could materialize it and then leave two WGPU contexts alive when
    // the qualified proof device is retained. CPU/fallback routes install the
    // auxiliary factory later inside their command handler.
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

    // Install the native CUDA / cuBLAS sound f64 GEMM accelerator if built with
    // `--features cuda`, a process-visible CUDA device exists, and candidate
    // libcuda/libcublas names are transiently dlopen-able. The engine
    // constructor remains authoritative: it selects and retains its actual
    // objects, validates their mapped providers on Linux, and runs its device
    // self-checks. This factory routes the sound CPU
    // CROWN backward's f64 `A·W` / `|A|·|W|` products to cuBLAS Dgemm with a
    // sound order-independent γ_n·S bound. Actual speed depends strongly on
    // topology and precision (especially consumer-GPU FP64). No-op when startup
    // admission fails.
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
        // Install LAZY factories: one shared CUDA engine (one GPU context + cuBLAS
        // handle), built on first use, drives BOTH the sound f64 `A·W` GEMM seam
        // AND the sound f64-exact GPU-resident CROWN backward. Lazy ⇒
        // CPU-trivial/small-input instances never pay the ~0.4s GPU init;
        // large-image attack steering materializes this same shared engine.
        // `vnncomp` explicitly prewarms the fast-f32 slot below before its
        // command handler creates finite verifier authority.
        use std::sync::{Arc, OnceLock};
        static CUDA_ENGINE: OnceLock<Option<Arc<ny_cuda::CudaGemmEngine>>> = OnceLock::new();
        fn shared_cuda_engine() -> Option<Arc<dyn ny_core::GemmEngine>> {
            CUDA_ENGINE
                .get_or_init(|| {
                    // Process-start admission observed a visible device and
                    // transiently dlopen-able driver/cuBLAS candidate names.
                    // This constructor selects and retains the actual cudarc
                    // objects, then validates their live mapped providers on
                    // Linux. cudarc's dynamic loader instead PANICS on a
                    // missing symbol in a partial/version-mismatched library
                    // that did dlopen. The release profile unwinds, so catch_unwind
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
        // Register CUDA separately for the domain-stacked proof forest. Engine
        // construction remains lazy while the experiment is off; an explicitly
        // requested wide route prewarms here, before command dispatch can create
        // verifier deadline authority. Routing remains experimental until an
        // NVIDIA sealed A/B enables `NY_CUDA_WIDE=1` (or the master
        // `NY_HYDRA_CROWN=1`). This does not alter ordinary CROWN routing.
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

    // #wgpu-crown-verdict: proof commands use `NY_WGPU_CROWN=auto|0|1` through
    // their cost/capability planner and exact typed resolver. Non-proof commands
    // retain the legacy exact `=1` factory below. Its construction consumes the typed
    // `WgpuVerdictRequest`. That constructor runs and stores the complete
    // five-rung authority report on the exact device returned to propagation.
    // Any initialization or qualification error maps to `None`, leaving the
    // fail-closed CPU verdict path unchanged.
    //
    // Ordering: AFTER the CUDA block — the factory is first-install-wins, so
    // NY_CUDA_CROWN keeps precedence when both are set. The prewarm is
    // MANDATORY here: under a finite deadline the route only consults the
    // PREINITIALIZED backend (select_lazy_backend_for_deadline), so a
    // registered-but-cold factory would be silently invisible in scored runs.
    // Proof commands construct and retain their one typed device in the shared
    // command resolver. Do not prewarm this older process-global factory there:
    // doing so creates a redundant second WGPU context and qualifies evidence
    // for a device other than the one the command executes on.
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
                // The WIDE (domain-stacked batched-BaB) lane rides the same typed
                // constructor: registration is deliberately independent of the
                // ordinary lane, and the request gate (`NY_CUDA_WIDE` /
                // `NY_HYDRA_CROWN` — backend-agnostic despite the names) still
                // applies, so this stays inert until explicitly asked. Prewarm
                // matters for the same deadline-preinit reason as above.
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
            "CUDA proof acceleration requested, but startup admission found no process-visible \
             CUDA device with transiently dlopen-able libcuda/libcublas candidates; CUDA \
             factories are unavailable and the fail-closed CPU path remains active"
        ),
        CudaStartupAction::Unavailable { requested: false } => info!(
            "CUDA startup admission found no process-visible device with transiently \
             dlopen-able libcuda/libcublas candidates; using CPU f64 path"
        ),
        CudaStartupAction::NotCompiled { requested: true } => warn!(
            "CUDA proof acceleration requested, but this binary lacks the `cuda` feature; CUDA \
             factories are unavailable and the fail-closed CPU path remains active"
        ),
        CudaStartupAction::NotCompiled { requested: false } => {}
    }

    // #accelerate-seam: Apple Accelerate (vecLib BLAS) on Apple silicon.
    //
    // DEFAULT OFF. Nothing below runs unless `NY_ACCELERATE_F64=1` (the SOUND
    // f64 seam that feeds the CROWN backward's `A·W` / `|A|·|W|`) or
    // `NY_ACCELERATE_F32=1` (the non-verdict IBP/PGD/BaB free-rider) is set;
    // `NY_NO_ACCELERATE` kills both regardless. Un-armed, this is a single
    // `getenv` and the process stays byte-identical to today.
    //
    // SOUNDNESS. `cblas_dgemm` is IEEE f64, and NY's certified error
    // `γ_n·S` is summation-order INDEPENDENT — the same Higham argument that
    // already licenses cuBLAS `Dgemm` above and faer's threaded reduction. The
    // engine additionally runs a one-shot 20-check conformance probe and refuses
    // to install on ANY failure, and enforces per call: G1 LP64 shape/stride,
    // G2 underflow domain (the one regime `γ_n·S` cannot cover), G3 symbol
    // provenance. Every refusal is a typed `Err`, so each call site keeps its
    // existing faer/CPU fallback unchanged.
    //
    // ORDERING: before the CPU floor below (first installation wins) and after
    // CUDA, so an actual accelerator still preempts it.
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
                    info!("Accelerate f32 free-rider armed (non-verdict IBP/PGD/BaB traffic)");
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

    // #cpu-gemm-engine: last resort, AFTER every accelerator above has had its
    // chance to register (first installation wins). Without this, a CPU-only
    // run left `fast_f32_gemm::is_installed()` false, and each engine-gated
    // fast path chose its scalar fallback — including the conv Patches
    // batched-GEMM seam, which is gated on an engine being reachable AT ALL and
    // was therefore structurally unreachable on any host without a GPU.
    // `NY_CPU_GEMM_ENGINE=0` restores the previous behaviour.
    ny_propagate::faer_parallelism::install_cpu_gemm_engine_if_absent();

    // #cpu-sound-f64-engine: same last-resort discipline for the SOUND f64 seam,
    // which until now only a CUDA build ever populated. On a CPU-only host
    // `sound_f64_gemm` stayed empty, so `aw_f64_with_abssum_unbounded` fell
    // through to a path that computes the abs-sum `S` with a SECOND FULL f64
    // GEMM; with any engine present it instead builds `S` from the cheap f32
    // seam. Measured worth of exactly this change: +2.4% end-to-end
    // (`FAER/OFF = 1.024x`, tests/audit_attribution.rs), with published bounds
    // bit-identical to the engine-absent arm. Note the vendor-BLAS variant
    // measured NEGATIVE on top of this (`ACC/FAER = 0.955x`) — the gain belongs
    // to having an engine at all, not to the kernel.
    // `NY_CPU_SOUND_F64_ENGINE=0` restores the previous behaviour.
    ny_propagate::faer_parallelism::install_cpu_sound_f64_gemm_engine_if_absent();

    // #fl-value-gpu-tier ORDERING FIX (2026-08-02): the factory was registered
    // inside `handle_beta_crown_command`, but the plan resolver's rate probe
    // (`resolve_and_materialize`, vnncomp.rs) runs BEFORE that handler — so the
    // probe measured the CPU-only chain, OnceLock-cached that rate, and the FL
    // admission gate never saw GPU speed even when the build later used it.
    // Register at startup, exactly like the CPU floor above: lazy (nothing
    // constructed until the seam's first consult) and fallible (None on hosts
    // without a usable adapter). The beta-crown registration remains as a
    // benign second `set` (first-install-wins).
    if !command_resolves_proof_backend && !compute_backend::detect().wgpu_probe_skipped {
        ny_propagate::fl_value_gemm::set_fl_value_gemm_factory(|| {
            match ny_gpu::FlValueGemmDevice::new_wgpu() {
                Ok(device) => {
                    Some(std::sync::Arc::new(device) as std::sync::Arc<dyn ny_core::GemmEngine>)
                }
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "FL-value wgpu f32 engine unavailable; forward-linear \
                         value GEMMs keep the tiled CPU tiers (#fl-value-gpu-tier)"
                    );
                    None
                }
            }
        });
    }

    match command {
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
            // default for interactive runs). A certificate request is threaded
            // into the verifier so verdict-only proof lanes without an external
            // transcript fail closed instead of producing an unexportable proof.
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
            } => {
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
            } => {
                commands::vnncomp_plan::handle_vnncomp_plan_command(
                    &category,
                    &onnx,
                    &vnnlib,
                    budget_secs,
                    configs_dir,
                    json,
                )?;
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
