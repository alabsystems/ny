// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Host compute-backend detection (#backend-detect).
//!
//! ny runs on three kinds of host: CUDA machines (the competition regime),
//! Apple-silicon laptops (Metal via wgpu), and CPU-only boxes. Which one a
//! measurement ran on has repeatedly decided what the measurement MEANT —
//! the cifar100/tinyimagenet timeout bank was frozen from a no-cuda regime
//! the repo itself documented as "meaningless" (0/99 objectives GPU-off vs
//! 98/99 in ~37s GPU-on, `CIFAR100_CUDA_BOOTSTRAP_BREAKTHROUGH_2026-07-15.md`),
//! and nothing in the rows records the difference. This module makes the
//! regime explicit: probe once per process, log one line that survives
//! `RUST_LOG=error` (the scored paths set it), and hand sweeps a string to
//! seal into their manifest.
//!
//! Detection is deliberately cheap. The CUDA half checks library presence and
//! asks the driver for its context-free visible-device count. On non-CUDA
//! hosts the wgpu half is an adapter query with no device or shader
//! compilation. CUDA hosts skip that redundant query: Vulkan
//! `request_adapter` may allocate an NVIDIA graphics context even without a
//! wgpu device, which adds process-start churn and can fail under driver-memory
//! pressure. Engine registration stays with `main.rs` and its lazy factories.
//! Enabled CUDA detection pays `cuInit`/`cuDeviceGetCount`, but attack-only
//! instances still never create a CUDA context or initialize the GEMM engine.
//!
//! The verdict-soundness story per backend (see `sound_gpu_gate.rs`):
//! - **cuda**: sound f64 GEMM seams (cuBLAS Dgemm, certified rounding bound);
//!   resident CUDA CROWN behind its qualification gates.
//! - **metal / other wgpu**: the public CLI route constructs a typed proof
//!   device and retains it only when every live qualification rung passes.
//!   WGSL has no f64 — but that
//!   is not the reason, and the older "the shaders carry no certified error
//!   term" claim here was simply wrong about the SOUND lane. The sound-resident
//!   shaders have carried a certified error term in pure f32 for a long time
//!   (Higham `γ_k·S`, host-side outward-rounded uniforms; ny-gpu
//!   `wgpu_device/sound_consts.rs:13` states the design as "no f64 ever enters a
//!   WGSL body"), and the EFT/double-single channel supplies an f64-grade
//!   compensated residual WITHOUT f64. Both EFT primitives measured bit-exact on
//!   an Apple M5 Max / Metal adapter on 2026-08-04 (fma TwoProduct 509/509
//!   lanes, fma-barrier TwoSum 307/307, 0 ULP).
//!
//!   CORRECTED 2026-08-06 (`#u2b`): the sentence that used to end the paragraph
//!   above — "and the shipped per-adapter gate `verify_eft_primitives()` passes
//!   there" — was true when written and is now FALSE BY CONSTRUCTION. Bit-exact
//!   primitives were never the whole precondition: the compensated channel's
//!   residual identity also requires GRADUAL UNDERFLOW, and this adapter is
//!   measured to flush subnormals (DAZ + FTZ, 7/15 probe lanes, and an EFT
//!   residual is among the silently-zeroed lanes). `verify_eft_primitives()`
//!   now ENTAILS `verify_gradual_underflow()`, so it reports `false` on this
//!   adapter and the compensated channel refuses itself here. The raw probes
//!   still pass — that is a measurement, not an authorization.
//!
//!   U1/U3/U4/U5/U6 and B0 are now discharged, and the raw `WgpuDevice` CROWN
//!   source gate and public integration are open. Authority still requires an
//!   explicit typed request and every live adapter probe; this Metal adapter
//!   fails the composed gradual-underflow gate and therefore emits a recorded
//!   CPU fallback. A conforming adapter retains the exact qualified context.
//!   Measured raw primitives alone are NOT verdict authority. See
//!   `docs/CURRENT_STATE_2026-08-10.md#wgpu-verdict-authority`.
//! - **cpu-only**: the proven-sound f64 CROWN path, at CPU speed.

use std::sync::OnceLock;

/// One host's compute regime, decided once per process.
#[derive(Debug)]
pub(crate) struct BackendReport {
    /// `cuda`, `metal`, `gpu` (non-Metal wgpu adapter), or `cpu-only`.
    pub(crate) kind: &'static str,
    /// One line for humans and manifests: kind, evidence for it, and what
    /// the verdict path soundly runs on as a result.
    pub(crate) summary: String,
    /// The one cached WGPU adapter observation, if probing was safe and found
    /// hardware. Never re-query Vulkan merely to render provenance.
    pub(crate) wgpu_adapter: Option<ny_gpu::AdapterProbe>,
    /// NVIDIA driver/device evidence made a Vulkan adapter query unsafe or
    /// redundant for this process.
    pub(crate) wgpu_probe_skipped: bool,
    /// A visible CUDA device plus the required driver/engine libraries makes
    /// the process-global CUDA GEMM factory a usable accelerator candidate.
    /// This is separate from WGPU adapter presence: NVIDIA hosts deliberately
    /// skip that redundant graphics-API probe.
    pub(crate) cuda_engine_candidate: bool,
}

/// Probe the host once and cache the answer for the process lifetime.
pub(crate) fn detect() -> &'static BackendReport {
    static REPORT: OnceLock<BackendReport> = OnceLock::new();
    REPORT.get_or_init(|| {
        let cuda = cuda_state();
        let adapter = maybe_probe_wgpu(cuda.avoid_wgpu_probe, ny_gpu::WgpuDevice::probe_adapter);

        let adapter_desc = if cuda.avoid_wgpu_probe {
            "skipped (NVIDIA driver/device evidence; avoid redundant graphics context)".to_string()
        } else {
            adapter.as_ref().map_or_else(
                || "none".to_string(),
                |probe| format!("{} ({}, {})", probe.backend, probe.name, probe.device_type),
            )
        };
        let metal = adapter
            .as_ref()
            .is_some_and(|probe| probe.backend == "Metal");

        let (kind, sound_path) = if cuda.engine_candidate {
            (
                "cuda",
                "CUDA f64 GEMM seams + CPU f64 CROWN (resident CUDA CROWN behind its gates)",
            )
        } else if metal {
            // #flush-charge: the charged-Metal narration is gated on the
            // build's charged source gate so the CLOSED state stays
            // byte-identical to the pre-chain summary. The gate state is a
            // compile-time fact (`ny_gpu::wgpu_charged_proof_authority`);
            // the per-run qualified/refused outcome is carried by the
            // ProofBackendReceipt and the NY-HARNESS override marker.
            let metal_sound_path = if ny_gpu::wgpu_charged_proof_authority() {
                "Metal CROWN qualified WITH FLUSH CHARGES when the typed charged \
                 constructor's live pure-flush ladder passes (charged source gate OPEN; \
                 fail-closed chain new_for_proof -> new_for_proof_flush_charged -> CPU f64; \
                 the gradual-underflow refusal of UNcharged authority is unchanged)"
            } else {
                "CPU f64 (Metal CROWN not qualified for verdicts. WGSL has no f64, but that is \
                 NOT the blocker: the EFT-f32 channel certifies error without f64 and its \
                 raw primitives measured bit-exact on Apple M5 Max/Metal 2026-08-04, \
                 but that adapter fails the composed gradual-underflow gate, so \
                 verify_eft_primitives() refuses there. U1/U3/U4/U5/U6 and B0 are discharged; \
                 the AUTO/default word route, ResNet-segment composition, and armed \
                 fail-closed C1 consult are landed, and the public typed proof constructor \
                 is open. This adapter still refuses at the gradual-underflow rung. \
                 docs/CURRENT_STATE_2026-08-10.md)"
            };
            ("metal", metal_sound_path)
        } else if adapter.is_some() {
            (
                "gpu",
                "typed WGPU proof qualification when all live rungs pass; CPU f64 fallback otherwise",
            )
        } else {
            ("cpu-only", "CPU f64")
        };

        BackendReport {
            kind,
            summary: format!(
                "{kind} [cuda: {}; wgpu adapter: {adapter_desc}; sound verdict path: {sound_path}]",
                cuda.describe
            ),
            wgpu_adapter: adapter,
            wgpu_probe_skipped: cuda.avoid_wgpu_probe,
            cuda_engine_candidate: cuda.engine_candidate,
        }
    })
}

/// A process with NVIDIA driver/device evidence does not need a second graphics
/// API merely to describe its compute regime. Keep the decision injectable so
/// tests prove that branch cannot execute the driver-facing probe.
fn maybe_probe_wgpu(
    skip_probe: bool,
    probe: impl FnOnce() -> Option<ny_gpu::AdapterProbe>,
) -> Option<ny_gpu::AdapterProbe> {
    (!skip_probe).then(probe).flatten()
}

#[cfg(test)]
fn render_wgpu_adapter_provenance(
    probe_skipped: bool,
    adapter: Option<&ny_gpu::AdapterProbe>,
) -> ny_gpu::WgpuAdapterProvenance {
    if probe_skipped {
        return ny_gpu::WgpuAdapterProvenance {
            hardware_available: false,
            description:
                "probe skipped: NVIDIA driver/device evidence; redundant graphics context avoided"
                    .to_string(),
        };
    }
    adapter.map_or_else(
        || ny_gpu::WgpuAdapterProvenance {
            hardware_available: false,
            description: "no hardware adapter".to_string(),
        },
        |adapter| ny_gpu::WgpuAdapterProvenance {
            hardware_available: true,
            description: format!(
                "{} ({}, {})",
                adapter.name, adapter.device_type, adapter.backend
            ),
        },
    )
}

/// Log the detected backend on the scored path.
///
/// `eprintln!`, not `tracing`: every scored entry point runs under
/// `RUST_LOG=error` (`run_instance.sh`, `measure_ny_scorecard.sh:667`,
/// `vnncomp_sweep.rs`), so an info/warn line is invisible exactly where the
/// regime matters. Stderr lands in the per-instance logs those harnesses
/// already capture.
pub(crate) fn log_once() {
    static LOGGED: OnceLock<()> = OnceLock::new();
    LOGGED.get_or_init(|| {
        eprintln!("ny compute backend: {}", detect().summary);
    });
}

/// Static host identity, probed once per process (#host-provenance).
///
/// Two timings are only comparable when they came from the same machine
/// class AND a known load regime — the 2026-07-29 audit called 900s timings
/// under load 63.7 "meaningless", and the cgan parity sweep proved the
/// sealed row-7 UNSAT was a different-host result. Identity is static and
/// cached; load is per-moment and deliberately NOT cached — see
/// [`load_average`].
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct HostProbe {
    pub(crate) hostname: String,
    pub(crate) cpu_model: String,
    pub(crate) logical_cores: usize,
    pub(crate) ram_bytes: u64,
}

impl HostProbe {
    /// One line for manifests: everything a reader needs to decide whether
    /// two timing measurements are even comparable.
    pub(crate) fn summary(&self) -> String {
        format!(
            "{} | {} | {} cores | {} GiB",
            self.hostname,
            self.cpu_model,
            self.logical_cores,
            self.ram_bytes >> 30
        )
    }
}

pub(crate) fn host() -> &'static HostProbe {
    static HOST: OnceLock<HostProbe> = OnceLock::new();
    HOST.get_or_init(|| HostProbe {
        hostname: command_line("uname", &["-n"]).unwrap_or_else(|| "unknown-host".to_string()),
        cpu_model: probe_cpu_model().unwrap_or_else(|| "unknown-cpu".to_string()),
        logical_cores: std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
        ram_bytes: probe_ram_bytes().unwrap_or(0),
    })
}

/// The 1/5/15-minute load averages RIGHT NOW — never cached, because load is
/// the per-moment fact that decides whether a wall-clock timing means
/// anything. Callers sample it at run start and again at artifact write, so
/// a row that ran on a busy box carries the evidence. Best-effort: `None` on
/// platforms without a probe, never an error.
pub(crate) fn load_average() -> Option<[f64; 3]> {
    #[cfg(target_os = "macos")]
    {
        // `sysctl -n vm.loadavg` prints "{ 1.74 1.88 2.08 }".
        let raw = command_line("sysctl", &["-n", "vm.loadavg"])?;
        parse_three_floats(raw.trim_matches(|c: char| c == '{' || c == '}' || c.is_whitespace()))
    }
    #[cfg(target_os = "linux")]
    {
        // /proc/loadavg: "1.74 1.88 2.08 2/1385 12345".
        let raw = std::fs::read_to_string("/proc/loadavg").ok()?;
        parse_three_floats(&raw)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// Pure split-and-parse helper behind [`load_average`].
///
/// Production callers exist only in the macOS and Linux arms above — no other
/// platform has a load-average source — so a Windows BIN build reaches none of
/// them and the function is dead there. It is deliberately NOT `#[cfg]`-gated:
/// the unit tests below exercise it on every platform, and keeping it compiled
/// everywhere means it cannot rot behind a gate that hides it from the build.
#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux")),
    allow(dead_code, reason = "no load-average source on this platform")
)]
fn parse_three_floats(raw: &str) -> Option<[f64; 3]> {
    let mut parts = raw.split_whitespace().map_while(|part| part.parse().ok());
    Some([parts.next()?, parts.next()?, parts.next()?])
}

/// Run a short probe command and return trimmed stdout. Best-effort: any
/// failure is `None`. Used only at process start / artifact write — never on
/// a hot path.
fn command_line(program: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let line = String::from_utf8(output.stdout).ok()?;
    let trimmed = line.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

#[cfg(target_os = "macos")]
fn probe_cpu_model() -> Option<String> {
    command_line("sysctl", &["-n", "machdep.cpu.brand_string"])
}

#[cfg(target_os = "linux")]
fn probe_cpu_model() -> Option<String> {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    cpuinfo
        .lines()
        .find(|line| line.starts_with("model name"))
        .and_then(|line| line.split(':').nth(1))
        .map(|name| name.trim().to_string())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn probe_cpu_model() -> Option<String> {
    None
}

#[cfg(target_os = "macos")]
fn probe_ram_bytes() -> Option<u64> {
    command_line("sysctl", &["-n", "hw.memsize"])?.parse().ok()
}

#[cfg(target_os = "linux")]
fn probe_ram_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: u64 = meminfo
        .lines()
        .find(|line| line.starts_with("MemTotal:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    Some(kb * 1024)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn probe_ram_bytes() -> Option<u64> {
    None
}

struct CudaState {
    /// A visible device and both libraries make lazy engine construction worth
    /// attempting. Context/cuBLAS allocation and self-check remain unconfirmed.
    engine_candidate: bool,
    /// Strong enough NVIDIA evidence to avoid asking Vulkan for a second
    /// graphics context. This is intentionally independent of NY_NO_CUDA and
    /// libcublas availability.
    avoid_wgpu_probe: bool,
    describe: String,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CudaDeviceProbe {
    DriverAbsent,
    Count(i32),
    Indeterminate,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaDecision {
    engine_candidate: bool,
    avoid_wgpu_probe: bool,
}

#[cfg(any(feature = "cuda", test))]
fn decide_cuda(
    disabled: bool,
    device_probe: CudaDeviceProbe,
    engine_libraries_present: bool,
    nvidia_kernel_driver_present: bool,
) -> CudaDecision {
    let visible_device = matches!(device_probe, CudaDeviceProbe::Count(count) if count > 0);
    let engine_candidate = !disabled && visible_device && engine_libraries_present;
    // TWO INDEPENDENT REASONS TO SKIP THE WGPU PROBE, and they are not the same
    // reason. Conflating them cost this repository the whole sound-f32 lane.
    //
    // 1. REDUNDANCY. A second graphics context beside a CUDA engine we are
    //    actually going to use is waste. That is `engine_candidate` — the engine
    //    is compiled in, enabled, has a visible device AND its libraries. If any
    //    of those is false the CUDA engine will not run, so there is nothing to
    //    be redundant WITH and every reason to look at the other adapter.
    //
    // 2. SAFETY. A driver-init/symbol error means the NVIDIA userspace stack is
    //    unhealthy, and Vulkan would provoke the same stack. That is
    //    `Indeterminate` specifically. `DriverAbsent` and a valid `Count(0)` are
    //    NOT errors: they say no NVIDIA device is available to this process, so
    //    an AMD/Intel/Metal adapter may legitimately be found.
    //
    // This used to read `nvidia_kernel_driver_present || ...`, which made mere
    // driver PRESENCE skip the probe unconditionally. On a healthy NVIDIA host
    // that meant the wgpu adapter was never probed even when CUDA was disabled
    // by NY_NO_CUDA, or absent from the build, or missing its libraries — so the
    // lane fell through to `cpu-only` / CPU f64 while a usable GPU sat idle, and
    // `NY_NO_CUDA=1 NY_WGPU_CROWN=1` could not reach the WGPU proof route at all.
    // docs/VNNCOMP_CRITICAL_PATH_2026-08-12.md recorded that as evidence there was
    // "no GPU f32 CROWN lane for the bound path at all"; it was this predicate.
    let unhealthy_nvidia_userspace = matches!(device_probe, CudaDeviceProbe::Indeterminate);
    let _ = nvidia_kernel_driver_present;
    CudaDecision {
        engine_candidate,
        avoid_wgpu_probe: unhealthy_nvidia_userspace || engine_candidate,
    }
}

#[cfg(target_os = "linux")]
fn nvidia_kernel_driver_present() -> bool {
    std::path::Path::new("/proc/driver/nvidia/version").exists()
        || std::path::Path::new("/dev/nvidiactl").exists()
}

#[cfg(not(target_os = "linux"))]
fn nvidia_kernel_driver_present() -> bool {
    false
}

#[cfg(feature = "cuda")]
fn cuda_state() -> CudaState {
    let disabled = std::env::var_os("NY_NO_CUDA").is_some();
    let kernel_driver_present = nvidia_kernel_driver_present();
    let driver_present = ny_cuda::CudaGemmEngine::runtime_driver_library_present();
    if disabled {
        return CudaState {
            engine_candidate: false,
            // NY_NO_CUDA is set precisely to make ny use something else. Skipping
            // the wgpu probe here left the operator with CPU f64 and no way to
            // reach the WGPU proof route — the opposite of what the flag asks for.
            avoid_wgpu_probe: false,
            describe: format!(
                "compiled, disabled by NY_NO_CUDA, NVIDIA kernel driver={}, libcuda={}",
                if kernel_driver_present {
                    "present"
                } else {
                    "not observed"
                },
                if driver_present { "present" } else { "absent" }
            ),
        };
    }
    let device_probe = if !driver_present {
        CudaDeviceProbe::DriverAbsent
    } else {
        ny_cuda::CudaGemmEngine::runtime_device_count()
            .map_or(CudaDeviceProbe::Indeterminate, CudaDeviceProbe::Count)
    };
    let engine_libraries_present = ny_cuda::CudaGemmEngine::runtime_libraries_present();
    let decision = decide_cuda(
        false,
        device_probe,
        engine_libraries_present,
        kernel_driver_present,
    );
    let device_desc = match device_probe {
        CudaDeviceProbe::DriverAbsent => "libcuda absent".to_string(),
        CudaDeviceProbe::Count(count) => format!("{count} process-visible device(s)"),
        CudaDeviceProbe::Indeterminate => "driver/device count indeterminate".to_string(),
    };
    let enablement = if decision.engine_candidate {
        "engine lazy/unconfirmed"
    } else {
        "engine unavailable"
    };
    CudaState {
        engine_candidate: decision.engine_candidate,
        avoid_wgpu_probe: decision.avoid_wgpu_probe,
        describe: format!(
            "compiled, NVIDIA kernel driver={}, {device_desc}, libcuda+libcublas={}, {enablement}",
            if kernel_driver_present {
                "present"
            } else {
                "not observed"
            },
            if engine_libraries_present {
                "present"
            } else {
                "incomplete"
            }
        ),
    }
}

#[cfg(not(feature = "cuda"))]
fn cuda_state() -> CudaState {
    let kernel_driver_present = nvidia_kernel_driver_present();
    CudaState {
        engine_candidate: false,
        // No CUDA engine in this build, so a "redundant" graphics context cannot
        // be redundant with anything. Probe the adapter we can actually use.
        avoid_wgpu_probe: false,
        describe: format!(
            "not compiled (build with --features cuda), NVIDIA kernel driver={}",
            if kernel_driver_present {
                "present"
            } else {
                "not observed"
            }
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_is_decisive_and_stable() {
        let report = detect();
        assert!(
            ["cuda", "metal", "gpu", "cpu-only"].contains(&report.kind),
            "kind must be one of the four regimes, got {}",
            report.kind
        );
        // The summary must carry all three provenance facts a reader needs.
        for needle in ["cuda:", "wgpu adapter:", "sound verdict path:"] {
            assert!(
                report.summary.contains(needle),
                "summary must state {needle}: {}",
                report.summary
            );
        }
        // Cached: a second call is the same allocation, not a re-probe.
        assert!(std::ptr::eq(report, detect()));
    }

    #[test]
    fn metal_summary_narrates_the_charged_gate_state() {
        // #flush-charge: the flight receipt seals `detect().summary`, so the
        // Metal line must carry the charged-route state — the refusal text
        // (byte-identical to pre-chain) while the source gate is closed, the
        // distinct charged narration once it opens.
        let report = detect();
        if report.kind != "metal" {
            return;
        }
        if ny_gpu::wgpu_charged_proof_authority() {
            assert!(
                report.summary.contains("WITH FLUSH CHARGES"),
                "an open charged gate must be narrated distinctly: {}",
                report.summary
            );
        } else {
            assert!(
                !report.summary.contains("WITH FLUSH CHARGES"),
                "a closed charged gate must leave today's refusal text untouched: {}",
                report.summary
            );
            assert!(
                report
                    .summary
                    .contains("refuses at the gradual-underflow rung"),
                "closed-gate Metal summary must keep the refusal narration: {}",
                report.summary
            );
        }
    }

    #[test]
    fn nvidia_evidence_never_calls_the_wgpu_adapter_probe() {
        let adapter = maybe_probe_wgpu(true, || {
            panic!("NVIDIA detection must not allocate a redundant graphics context")
        });
        assert!(adapter.is_none());

        let called = std::cell::Cell::new(false);
        let adapter = maybe_probe_wgpu(false, || {
            called.set(true);
            Some(ny_gpu::AdapterProbe {
                backend: "Metal".to_string(),
                name: "test-adapter".to_string(),
                device_type: "IntegratedGpu".to_string(),
            })
        });
        assert!(called.get());
        assert_eq!(
            adapter.as_ref().map(|probe| probe.name.as_str()),
            Some("test-adapter")
        );
    }

    #[test]
    fn explicit_wgpu_provenance_respects_the_nvidia_safety_skip() {
        let synthetic = ny_gpu::AdapterProbe {
            backend: "Vulkan".to_string(),
            name: "must-not-be-authoritative".to_string(),
            device_type: "DiscreteGpu".to_string(),
        };
        let provenance = render_wgpu_adapter_provenance(true, Some(&synthetic));
        assert!(!provenance.hardware_available);
        assert!(provenance.description.contains("probe skipped"));

        let provenance = render_wgpu_adapter_provenance(false, Some(&synthetic));
        assert!(provenance.hardware_available);
        assert!(provenance.description.contains("must-not-be-authoritative"));
    }

    #[test]
    fn the_wgpu_probe_skip_needs_a_reason_not_merely_a_driver() {
        // Was `loaded_nvidia_kernel_driver_forces_process_wide_wgpu_skip`, which
        // pinned the policy rather than an invariant: any loaded NVIDIA kernel
        // driver skipped the probe process-wide. That is why a build without
        // --features cuda, on a healthy NVIDIA host, reported `cpu-only` while a
        // usable GPU sat idle, and why `NY_NO_CUDA=1 NY_WGPU_CROWN=1` could not
        // reach the WGPU proof route.
        //
        // The invariant that actually matters: a skip must be justified by
        // REDUNDANCY (a CUDA engine we will really use) or by SAFETY (an
        // unhealthy NVIDIA userspace stack Vulkan would provoke). Driver
        // presence alone is neither.
        let report = detect();
        if report.wgpu_probe_skipped {
            assert!(
                report.cuda_engine_candidate || report.summary.contains("indeterminate"),
                "wgpu probe skipped with no CUDA engine to be redundant with and no \
                 driver/device error to fail safe on: {}",
                report.summary
            );
            assert!(report.wgpu_adapter.is_none());
        }
    }

    #[test]
    fn cuda_safety_and_engine_capability_are_independent() {
        let cases = [
            (
                false,
                CudaDeviceProbe::DriverAbsent,
                false,
                false,
                CudaDecision {
                    engine_candidate: false,
                    avoid_wgpu_probe: false,
                },
            ),
            (
                false,
                CudaDeviceProbe::Count(0),
                true,
                false,
                CudaDecision {
                    engine_candidate: false,
                    avoid_wgpu_probe: false,
                },
            ),
            (
                false,
                CudaDeviceProbe::Count(1),
                true,
                false,
                CudaDecision {
                    engine_candidate: true,
                    avoid_wgpu_probe: true,
                },
            ),
            // Device visible but CUDA libraries incomplete: the engine cannot
            // run, so there is nothing for a wgpu context to be redundant with.
            (
                false,
                CudaDeviceProbe::Count(1),
                false,
                false,
                CudaDecision {
                    engine_candidate: false,
                    avoid_wgpu_probe: false,
                },
            ),
            // NY_NO_CUDA: the operator disabled CUDA to use something else.
            // Skipping the probe here is how `NY_NO_CUDA=1 NY_WGPU_CROWN=1`
            // used to land on CPU f64.
            (
                true,
                CudaDeviceProbe::Count(1),
                true,
                false,
                CudaDecision {
                    engine_candidate: false,
                    avoid_wgpu_probe: false,
                },
            ),
            (
                false,
                CudaDeviceProbe::Indeterminate,
                true,
                false,
                CudaDecision {
                    engine_candidate: false,
                    avoid_wgpu_probe: true,
                },
            ),
            // Kernel driver present but libcuda absent. Driver PRESENCE alone
            // must not skip the probe: the kernel driver is healthy, Vulkan
            // reaches it through a different userspace library, and CUDA is
            // unusable. This is the case that left an RTX 5080 on cpu-only.
            (
                false,
                CudaDeviceProbe::DriverAbsent,
                false,
                true,
                CudaDecision {
                    engine_candidate: false,
                    avoid_wgpu_probe: false,
                },
            ),
            // The SAFETY case survives unchanged: a driver-init/symbol error
            // means the NVIDIA userspace stack is unhealthy and Vulkan would
            // provoke the same stack, so refuse the probe even though the CUDA
            // engine is not a candidate either.
            (
                false,
                CudaDeviceProbe::Indeterminate,
                false,
                true,
                CudaDecision {
                    engine_candidate: false,
                    avoid_wgpu_probe: true,
                },
            ),
        ];
        for (disabled, device_probe, libraries, kernel_driver, expected) in cases {
            assert_eq!(
                decide_cuda(disabled, device_probe, libraries, kernel_driver),
                expected,
                "disabled={disabled}, device_probe={device_probe:?}, libraries={libraries}, \
                 kernel_driver={kernel_driver}"
            );
        }
    }

    #[test]
    fn host_probe_is_populated_and_load_parses_on_this_platform() {
        let probe = host();
        assert!(!probe.hostname.is_empty());
        assert!(
            probe.logical_cores > 0,
            "available_parallelism should resolve on every dev/CI host"
        );
        // Identity is cached; the summary carries all four facts.
        assert!(std::ptr::eq(probe, host()));
        let summary = probe.summary();
        assert!(
            summary.matches(" | ").count() == 3,
            "summary must be 'host | cpu | cores | ram': {summary}"
        );
        // Load: macOS and Linux (every dev + CI + competition host) must
        // produce a real sample; a timing artifact without one is the
        // regression this module exists to prevent.
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            let load = load_average().expect("load average must parse on macOS/Linux");
            assert!(load.iter().all(|value| value.is_finite() && *value >= 0.0));
        }
    }

    #[test]
    fn three_float_parsing_handles_both_platform_shapes() {
        // macOS `sysctl -n vm.loadavg` (braces pre-stripped) and Linux
        // /proc/loadavg (trailing fields ignored by take-3).
        assert_eq!(
            parse_three_floats("1.74 1.88 2.08"),
            Some([1.74, 1.88, 2.08])
        );
        assert_eq!(
            parse_three_floats("1.74 1.88 2.08 2/1385 12345"),
            Some([1.74, 1.88, 2.08])
        );
        assert_eq!(parse_three_floats("1.74 1.88"), None);
        assert_eq!(parse_three_floats(""), None);
    }

    #[test]
    fn a_cuda_less_build_never_reports_cuda() {
        // This dev box builds without `--features cuda` (and Apple hardware
        // has no CUDA at all): claiming otherwise would repeat the frozen
        // GPU-off bank defect this module exists to prevent.
        #[cfg(not(feature = "cuda"))]
        assert_ne!(detect().kind, "cuda");
    }
}
