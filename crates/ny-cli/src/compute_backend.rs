// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Host compute-backend detection (#backend-detect).
//!
//! Cache host capability evidence once per process, log it even under
//! `RUST_LOG=error`, and supply it to sweep manifests. Detection is separate
//! from proof-backend qualification.
//!
//! CUDA detection checks library presence and the visible-device count.
//! WGPU independently queries an adapter without creating a device or compiling
//! shaders, including on healthy CUDA hosts: the two backends supply different
//! capabilities. An indeterminate CUDA device query suppresses the WGPU probe
//! to avoid re-entering unhealthy NVIDIA userspace. Engine registration and
//! lazy construction remain in `main.rs`.
//!
//! CUDA supplies certified f64 GEMM seams and separately gated resident CROWN.
//! For WGPU proof work, `commands/backend.rs` tries the typed uncharged
//! constructor, then the flush-charged constructor, retaining the exact device
//! and its qualification receipt on success. If both refuse, it records the
//! refusal and uses CPU f64. Adapter discovery alone grants no verdict authority.

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
    /// An indeterminate CUDA query suppressed WGPU adapter probing.
    pub(crate) wgpu_probe_skipped: bool,
    /// A visible CUDA device plus the required driver/engine libraries makes
    /// the process-global CUDA GEMM factory a usable accelerator candidate.
    /// WGPU adapter availability is independent of this candidate.
    pub(crate) cuda_engine_candidate: bool,
}

/// Probe the host once and cache the answer for the process lifetime.
pub(crate) fn detect() -> &'static BackendReport {
    static REPORT: OnceLock<BackendReport> = OnceLock::new();
    REPORT.get_or_init(|| {
        let cuda = cuda_state();
        let adapter = maybe_probe_wgpu(cuda.avoid_wgpu_probe, ny_gpu::WgpuDevice::probe_adapter);

        let adapter_desc = if cuda.avoid_wgpu_probe {
            "skipped (indeterminate CUDA query; avoid unhealthy NVIDIA userspace)".to_string()
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
            // Narrate the source gate here; ProofBackendReceipt and the
            // NY-HARNESS marker carry the run's actual qualification outcome.
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

/// Apply the CUDA safety skip before calling the adapter probe. The injectable
/// probe lets tests verify that a skipped query cannot enter the graphics API.
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
                "probe skipped: indeterminate CUDA query; unhealthy NVIDIA userspace avoided"
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

/// Log once to stderr so harnesses using `RUST_LOG=error` retain the evidence.
pub(crate) fn log_once() {
    static LOGGED: OnceLock<()> = OnceLock::new();
    LOGGED.get_or_init(|| {
        eprintln!("ny compute backend: {}", detect().summary);
    });
}

/// Cached host identity for measurement provenance. Sample changing load
/// separately with [`load_average`].
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct HostProbe {
    pub(crate) hostname: String,
    pub(crate) cpu_model: String,
    pub(crate) logical_cores: usize,
    pub(crate) ram_bytes: u64,
}

impl HostProbe {
    /// Host hardware summary; record resource limits, load and build identity separately.
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

/// Current 1/5/15-minute load averages, sampled at run start and artifact write.
/// Returns `None` on platforms without a probe; never cached.
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

/// Parse load averages; kept available to tests on every platform.
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
    /// An indeterminate device query suppresses re-entry into NVIDIA userspace,
    /// independently of NY_NO_CUDA and libcublas availability.
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
    // Only an indeterminate CUDA query suppresses WGPU probing, to avoid
    // re-entering unhealthy NVIDIA userspace. Healthy CUDA and WGPU supply
    // different capabilities, including WGPU's intermediate sweep. Engine
    // availability, disablement and kernel-driver presence do not govern this skip.
    let unhealthy_nvidia_userspace = matches!(device_probe, CudaDeviceProbe::Indeterminate);
    let _ = nvidia_kernel_driver_present;
    CudaDecision {
        engine_candidate,
        avoid_wgpu_probe: unhealthy_nvidia_userspace,
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
        // No CUDA userspace query ran, so no indeterminate result requires a skip.
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
        // Only an indeterminate CUDA query justifies the safety skip; driver
        // presence or a usable CUDA engine alone does not.
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
            // A healthy CUDA engine leaves WGPU's independent capabilities discoverable.
            (
                false,
                CudaDeviceProbe::Count(1),
                true,
                false,
                CudaDecision {
                    engine_candidate: true,
                    avoid_wgpu_probe: false,
                },
            ),
            // Incomplete CUDA engine libraries do not suppress WGPU discovery.
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
            // Explicitly disabling CUDA still permits WGPU discovery.
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
            // Kernel-driver presence without libcuda does not establish an
            // indeterminate query; keep WGPU discovery available.
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
            // An indeterminate query suppresses WGPU even without usable CUDA libraries.
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
        assert_ne!(probe.hostname, "");
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
