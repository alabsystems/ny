// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Backend selection helpers shared across CLI commands.

use ny_core::GemmEngine;
use ny_gpu::{ComputeDevice, WgpuVerdictRequest};
use tracing::{info, warn};

use crate::BackendArg;

/// Where the backend selected at the CLI/config boundary came from.
///
/// This is deliberately separate from the selected and effective backend:
/// `preset -> wgpu -> cpu` is materially different evidence from an explicit
/// CPU request even though both ultimately execute on CPU when WGPU
/// qualification refuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendRequestSource {
    ExplicitBackend,
    LegacyGpuFlag,
    Preset,
    Auto,
    /// A defaulted `BackendArg` erased whether `--backend cpu` was present.
    /// Use this honest value rather than claiming an explicit or automatic
    /// source that the caller can no longer prove.
    DefaultedCliValue,
}

impl BackendRequestSource {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitBackend => "explicit_backend",
            Self::LegacyGpuFlag => "legacy_gpu_flag",
            Self::Preset => "preset",
            Self::Auto => "auto",
            Self::DefaultedCliValue => "defaulted_cli_value",
        }
    }
}

/// Backend choice before runtime proof qualification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BackendRequest {
    pub(crate) backend: BackendArg,
    pub(crate) source: BackendRequestSource,
    /// Present only for an automatic choice; explicit and preset choices need
    /// no synthesized rationale.
    pub(crate) selection_reason: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutomaticWgpuPolicy {
    Auto,
    Disabled,
    Required,
}

fn parse_automatic_wgpu_policy(
    raw: Option<&std::ffi::OsStr>,
) -> anyhow::Result<AutomaticWgpuPolicy> {
    match raw.map(std::ffi::OsStr::to_str) {
        None | Some(Some("auto")) => Ok(AutomaticWgpuPolicy::Auto),
        Some(Some("0")) => Ok(AutomaticWgpuPolicy::Disabled),
        Some(Some("1")) => Ok(AutomaticWgpuPolicy::Required),
        Some(_) => anyhow::bail!("NY_WGPU_CROWN must be exactly auto, 0, or 1"),
    }
}

/// Apply the process authority policy to an AUTO backend request.
///
/// `auto_wgpu_candidate` is decided by the command's cost/capability planner:
/// beta-CROWN uses its size gate, while `verify` uses whether its selected mode
/// can consume the typed CROWN device. Explicit CLI/config backend requests are
/// never rewritten. A CUDA resident-CROWN request suppresses automatic WGPU
/// authority; asking for both authorities explicitly is rejected as ambiguous.
fn resolve_automatic_wgpu_request_from(
    mut request: BackendRequest,
    auto_wgpu_candidate: bool,
    raw_wgpu_policy: Option<&std::ffi::OsStr>,
    cuda_crown_requested: bool,
) -> anyhow::Result<BackendRequest> {
    let policy = parse_automatic_wgpu_policy(raw_wgpu_policy)?;
    if request.source != BackendRequestSource::Auto {
        if request.backend == BackendArg::Wgpu && cuda_crown_requested {
            anyhow::bail!(
                "an explicit WGPU proof backend conflicts with the explicit CUDA \
                 resident-CROWN authority request"
            );
        }
        return Ok(request);
    }

    let (backend, reason) = match (policy, cuda_crown_requested, auto_wgpu_candidate) {
        (AutomaticWgpuPolicy::Required, true, _) => anyhow::bail!(
            "NY_WGPU_CROWN=1 conflicts with the explicit CUDA resident-CROWN authority request"
        ),
        (AutomaticWgpuPolicy::Required, false, _) => (
            BackendArg::Wgpu,
            "NY_WGPU_CROWN=1 requires live typed WGPU qualification",
        ),
        (AutomaticWgpuPolicy::Disabled, _, _) => (
            BackendArg::Cpu,
            "NY_WGPU_CROWN=0 disables automatic WGPU proof qualification",
        ),
        (AutomaticWgpuPolicy::Auto, true, _) => (
            BackendArg::Cpu,
            "explicit CUDA resident-CROWN authority suppresses automatic WGPU authority",
        ),
        (AutomaticWgpuPolicy::Auto, false, true) => (
            BackendArg::Wgpu,
            "capability/cost profile selected live typed WGPU qualification",
        ),
        (AutomaticWgpuPolicy::Auto, false, false) => (
            BackendArg::Cpu,
            "capability/cost profile selected CPU proof execution",
        ),
    };
    request.backend = backend;
    request.selection_reason = Some(reason);
    Ok(request)
}

pub(crate) fn resolve_automatic_wgpu_request(
    request: BackendRequest,
    auto_wgpu_candidate: bool,
) -> anyhow::Result<BackendRequest> {
    let cuda_crown_requested = crate::exact_boolean_env("NY_CUDA_CROWN")?;
    resolve_automatic_wgpu_request_from(
        request,
        auto_wgpu_candidate,
        std::env::var_os("NY_WGPU_CROWN").as_deref(),
        cuda_crown_requested,
    )
}

/// Runtime qualification state of the selected proof backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProofBackendQualification {
    /// CPU needs no WGPU adapter qualification.
    NotRequested,
    /// The exact WGPU verdict request and every live adapter rung passed.
    Qualified,
    /// WGPU was selected, but construction or a qualification rung refused.
    Refused,
}

impl ProofBackendQualification {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::NotRequested => "not_requested",
            Self::Qualified => "qualified",
            Self::Refused => "refused",
        }
    }
}

/// Immutable receipt for one proof-backend decision.
///
/// The fields intentionally distinguish selection from execution. A caller
/// may select WGPU from a preset, fail a live qualification rung, and execute
/// with the CPU fallback; collapsing those facts into a single `backend`
/// string recreates the silent-override defect this receipt prevents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProofBackendReceipt {
    pub(crate) requested: BackendArg,
    pub(crate) request_source: BackendRequestSource,
    pub(crate) selection_reason: Option<String>,
    pub(crate) effective: BackendArg,
    pub(crate) qualification: ProofBackendQualification,
    /// Adapter identity tied to the live qualification attempt. This remains
    /// distinct from `provenance`, which names the engine that actually owns
    /// proof execution after any fallback.
    pub(crate) qualification_provenance: Option<String>,
    pub(crate) failed_rung: Option<String>,
    pub(crate) fallback_reason: Option<String>,
    pub(crate) provenance: String,
}

impl ProofBackendReceipt {
    pub(crate) fn cpu(request: BackendRequest, provenance: impl Into<String>) -> Self {
        debug_assert_eq!(request.backend, BackendArg::Cpu);
        Self {
            requested: request.backend,
            request_source: request.source,
            selection_reason: request.selection_reason.map(str::to_string),
            effective: BackendArg::Cpu,
            qualification: ProofBackendQualification::NotRequested,
            qualification_provenance: None,
            failed_rung: None,
            fallback_reason: None,
            provenance: provenance.into(),
        }
    }

    pub(crate) fn qualified_wgpu(
        request: BackendRequest,
        provenance: impl Into<String>,
        qualification_provenance: impl Into<String>,
    ) -> Self {
        debug_assert_eq!(request.backend, BackendArg::Wgpu);
        Self {
            requested: request.backend,
            request_source: request.source,
            selection_reason: request.selection_reason.map(str::to_string),
            effective: BackendArg::Wgpu,
            qualification: ProofBackendQualification::Qualified,
            qualification_provenance: Some(qualification_provenance.into()),
            failed_rung: None,
            fallback_reason: None,
            provenance: provenance.into(),
        }
    }

    pub(crate) fn refused_wgpu(
        request: BackendRequest,
        provenance: impl Into<String>,
        qualification_provenance: Option<String>,
        failed_rung: Option<String>,
        fallback_reason: impl Into<String>,
    ) -> Self {
        debug_assert_eq!(request.backend, BackendArg::Wgpu);
        Self {
            requested: request.backend,
            request_source: request.source,
            selection_reason: request.selection_reason.map(str::to_string),
            effective: BackendArg::Cpu,
            qualification: ProofBackendQualification::Refused,
            qualification_provenance,
            failed_rung,
            fallback_reason: Some(fallback_reason.into()),
            provenance: provenance.into(),
        }
    }

    #[must_use]
    pub(crate) fn qualified_wgpu_active(&self) -> bool {
        self.requested == BackendArg::Wgpu
            && self.effective == BackendArg::Wgpu
            && self.qualification == ProofBackendQualification::Qualified
    }
}

/// Render the unconditional stderr evidence required for any backend
/// substitution. Keeping formatting pure makes the JSON/scored-path contract
/// testable without redirecting process-global stderr.
#[must_use]
pub(crate) fn backend_override_message(
    command: &str,
    receipt: &ProofBackendReceipt,
) -> Option<String> {
    let reason = receipt.fallback_reason.as_deref()?;
    Some(format!(
        "{BACKEND_OVERRIDE_MARKER}: command={command} requested={} source={} effective={} \
         qualification={} failed_rung={} qualification_provenance={} provenance={} reason={reason}",
        receipt.requested,
        receipt.request_source.as_str(),
        receipt.effective,
        receipt.qualification.as_str(),
        receipt.failed_rung.as_deref().unwrap_or("none"),
        receipt
            .qualification_provenance
            .as_deref()
            .unwrap_or("unavailable"),
        receipt.provenance,
    ))
}

pub(crate) fn emit_backend_override(command: &str, receipt: &ProofBackendReceipt) {
    if let Some(message) = backend_override_message(command, receipt) {
        eprintln!("{message}");
    }
}

/// Qualified device plus the immutable receipt describing how it was chosen.
pub(crate) struct ProofBackendResolution<T> {
    pub(crate) receipt: ProofBackendReceipt,
    pub(crate) device: T,
}

/// Testable WGPU constructor refusal, preserving the same evidence the typed
/// ny-gpu error exposes without requiring a live adapter in CLI unit tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WgpuProofRefusal {
    pub(crate) reason: String,
    pub(crate) failed_rung: Option<String>,
    pub(crate) qualification_provenance: Option<String>,
}

/// Resolve a proof backend with injectable CPU/WGPU constructors.
///
/// WGPU is effective only when its proof constructor succeeds. Every refusal
/// constructs the always-sound CPU device, records both attempted and effective
/// provenance, and emits the override marker independently of log/JSON mode.
pub(crate) fn resolve_proof_backend_with_factories<T, C, W>(
    request: BackendRequest,
    command: &str,
    build_cpu: C,
    build_wgpu: W,
) -> anyhow::Result<ProofBackendResolution<T>>
where
    C: FnOnce() -> anyhow::Result<(T, String)>,
    W: FnOnce() -> Result<(T, String, String), WgpuProofRefusal>,
{
    match request.backend {
        BackendArg::Cpu => {
            let (device, provenance) = build_cpu()?;
            Ok(ProofBackendResolution {
                receipt: ProofBackendReceipt::cpu(request, provenance),
                device,
            })
        }
        BackendArg::Wgpu => match build_wgpu() {
            Ok((device, provenance, qualification_provenance)) => Ok(ProofBackendResolution {
                receipt: ProofBackendReceipt::qualified_wgpu(
                    request,
                    provenance,
                    qualification_provenance,
                ),
                device,
            }),
            Err(refusal) => {
                let (device, provenance) = build_cpu()?;
                let receipt = ProofBackendReceipt::refused_wgpu(
                    request,
                    provenance,
                    refusal.qualification_provenance,
                    refusal.failed_rung,
                    refusal.reason,
                );
                emit_backend_override(command, &receipt);
                Ok(ProofBackendResolution { receipt, device })
            }
        },
    }
}

/// #flush-charge: the fail-closed WGPU proof-qualification chain.
///
/// Order is part of the contract: the fully qualified (uncharged) constructor
/// is ALWAYS attempted first, so a conforming adapter can never be downgraded
/// into charged mode; the charged constructor is consulted only after the
/// uncharged ladder refuses. When the charged attempt also refuses, its
/// refusal is DISCARDED: the caller sees exactly the uncharged refusal
/// evidence, so refusal receipts and override markers stay byte-identical to
/// the pre-chain binary (pinned by
/// `closed_charged_gate_keeps_the_chain_byte_identical` and the live
/// `charged_chain_is_live_on_this_host_now_that_the_gate_is_open`). With the
/// reviewed charged source gate OPEN (2026-08-13 review), a charged admission
/// resolves the chain with the distinctly narrated charged evidence.
fn qualify_wgpu_proof_chain<T>(
    uncharged: impl FnOnce() -> Result<(T, String, String), WgpuProofRefusal>,
    charged: impl FnOnce() -> Result<(T, String, String), WgpuProofRefusal>,
) -> Result<(T, String, String), WgpuProofRefusal> {
    match uncharged() {
        Ok(qualified) => Ok(qualified),
        Err(uncharged_refusal) => match charged() {
            Ok(charged_qualified) => Ok(charged_qualified),
            // Fail-closed to CPU with the UNCHARGED evidence: with the charged
            // gate closed this arm always runs and must not perturb today's
            // receipt/marker bytes; with it open, the uncharged rung report
            // remains the more diagnostic refusal.
            Err(_charged_refusal) => Err(uncharged_refusal),
        },
    }
}

/// Shared translation of a typed qualification error into the testable refusal.
fn wgpu_refusal_from(error: &ny_gpu::WgpuVerdictQualificationError) -> WgpuProofRefusal {
    let report = error.report();
    WgpuProofRefusal {
        reason: format!(
            "{}; source error: {}",
            report.reason(),
            error.source_error()
        ),
        failed_rung: report.failed_rung().map(|rung| rung.to_string()),
        qualification_provenance: report.adapter().map(str::to_string),
    }
}

/// Success evidence for a qualified proof device: `(device, provenance,
/// qualification_provenance)`. A charged device narrates its outcome
/// distinctly in BOTH strings: `backend_provenance()` reports
/// `wgpu-qualified-crown-flush-charged`, and the adapter identity carries the
/// charged marker so receipts and manifests cannot read as fully qualified.
fn wgpu_qualified_evidence(
    device: ComputeDevice,
    charged: bool,
) -> (ComputeDevice, String, String) {
    let provenance = device.backend_provenance().to_string();
    let adapter = device
        .wgpu_verdict_report()
        .and_then(|report| report.adapter())
        .unwrap_or("qualified adapter identity unavailable")
        .to_string();
    let qualification_provenance = if charged {
        format!("{adapter} (CROWN qualified WITH FLUSH CHARGES)")
    } else {
        adapter
    };
    (device, provenance, qualification_provenance)
}

/// Production proof-device resolver shared by `verify` and `beta-crown`.
///
/// Ordinary `ComputeDevice::new(Wgpu)` is intentionally not reachable here:
/// only the typed constructors consume a request and retain the exact device
/// whose live admission report passed. The WGPU leg is the fail-closed chain
/// `new_for_proof` (fully qualified) -> `new_for_proof_flush_charged`
/// (charged-Metal; source gate OPEN since 2026-08-13, the live pure-flush
/// ladder decides per device) -> CPU. CUDA keeps
/// its existing precedence UPSTREAM of this resolver: an explicit CUDA
/// resident-CROWN authority request suppresses the automatic WGPU request
/// entirely (`resolve_automatic_wgpu_request`), so this chain only runs when
/// WGPU was actually selected.
pub(crate) fn resolve_proof_backend(
    request: BackendRequest,
    command: &str,
) -> anyhow::Result<ProofBackendResolution<ComputeDevice>> {
    resolve_proof_backend_with_factories(
        request,
        command,
        || {
            let device = ComputeDevice::new(ny_gpu::Backend::Cpu)?;
            let provenance = device.backend_provenance().to_string();
            Ok((device, provenance))
        },
        || {
            qualify_wgpu_proof_chain(
                || match ComputeDevice::new_for_proof(WgpuVerdictRequest::new()) {
                    Ok(device) => Ok(wgpu_qualified_evidence(device, false)),
                    Err(error) => Err(wgpu_refusal_from(&error)),
                },
                || match ComputeDevice::new_for_proof_flush_charged(
                    ny_gpu::WgpuChargedVerdictRequest::new(),
                ) {
                    Ok(device) => Ok(wgpu_qualified_evidence(device, true)),
                    Err(error) => Err(wgpu_refusal_from(&error)),
                },
            )
        },
    )
}

/// Decide, fail-closed, whether a resolved proof device may be published into
/// the process-global sound GPU CROWN engine slots.
///
/// `Some` only when the receipt proves a LIVE qualified WGPU resolution (the
/// charged-flush chain included: its receipt is also
/// `ProofBackendQualification::Qualified`) AND the exact engine advertises the
/// sound CROWN accessor (`as_gpu_crown_backward` +
/// `provides_sound_gpu_crown`). Everything else — CPU builds, refused
/// qualification, engines whose accessor is closed — returns `None` and the
/// caller installs NOTHING, keeping unqualified hosts byte-identical.
fn engine_for_sound_crown_slots(
    receipt: &ProofBackendReceipt,
    engine: &std::sync::Arc<dyn GemmEngine>,
) -> Option<std::sync::Arc<dyn GemmEngine>> {
    if !receipt.qualified_wgpu_active() {
        return None;
    }
    engine
        .as_gpu_crown_backward()
        .is_some_and(|gpu| gpu.provides_sound_gpu_crown())
        .then(|| std::sync::Arc::clone(engine))
}

/// #charged-metal-engagement: publish the exact QUALIFIED wgpu proof device
/// into the process-global sound GPU CROWN engine slots
/// (`ny_propagate::sound_gpu_gate`) that borrow-only consumers read — the
/// deadline-preinitialized sequential/margin-row routes, the resident cut
/// shadow, and the wide lanes.
///
/// The CUDA path fills these slots from `main` at startup. Proof commands
/// deliberately skip `main`'s legacy `NY_WGPU_CROWN=1` factory (it would build
/// a SECOND wgpu context and qualify evidence for a device other than the one
/// the command executes on), so on wgpu hosts the slots historically stayed
/// cold on the scored/verify routes. This closes that gap without a second
/// context: the registered factory returns a clone of the already-materialized
/// Arc, so the mandatory prewarm performs no GPU work and completes at
/// qualification time — before any finite deadline authority exists. Under a
/// deadline every route consults only the PREINITIALIZED slot
/// (`select_lazy_backend_for_deadline`), and the projected-M2 cut-shadow seam
/// explicitly cannot afford a lazy factory under its child deadline, hence
/// pre-materialization here rather than lazily.
///
/// Contracts respected, fail-closed:
/// - install-once / first-install-wins: on hosts where `main` already
///   installed a factory (CUDA), this registration silently loses and CUDA
///   precedence is untouched;
/// - nothing is installed unless [`engine_for_sound_crown_slots`] admits the
///   receipt+engine pair, so no-wgpu / refused-qualification hosts stay
///   byte-identical (no factory, no materialized slot);
/// - the wide slot mirrors `main`'s registration: the factory is installed,
///   but the lane stays dark unless its exact request gates
///   (`NY_CUDA_WIDE`/`NY_HYDRA_CROWN` — backend-agnostic despite the names)
///   are set, and then it too must be materialized pre-deadline;
/// - the sound-f64 GEMM seam (`sound_f64_gemm`) is deliberately NOT filled:
///   its contract assumes an IEEE f64 `Dgemm` (cuBLAS/Accelerate) and the
///   charged device is f32-only — substituting it there would be an assumption,
///   not a qualification. Only the capability-typed CROWN slots are published.
pub(crate) fn register_qualified_wgpu_proof_engine(
    receipt: &ProofBackendReceipt,
    engine: &std::sync::Arc<dyn GemmEngine>,
) {
    use std::sync::Arc;
    let Some(engine) = engine_for_sound_crown_slots(receipt, engine) else {
        return;
    };
    let ordinary = Arc::clone(&engine);
    ny_propagate::sound_gpu_gate::set_sound_gpu_crown_factory(move || Some(Arc::clone(&ordinary)));
    let wide = Arc::clone(&engine);
    ny_propagate::sound_gpu_gate::set_wide_sound_gpu_crown_factory(move || Some(Arc::clone(&wide)));
    if ny_propagate::sound_gpu_gate::prewarm_sound_gpu_crown() {
        // INFO, not WARN: registration SUCCEEDED. The fail-closed arm below is
        // the warning. At WARN this reached stderr at default verbosity and
        // broke `--json`'s empty-stderr contract (#395) on any machine with a
        // qualifying GPU.
        info!(
            "qualified WGPU proof engine registered: process-global sound CROWN slot \
             materialized before command deadline authority (borrow-only consumers may see it; \
             CPU fallback retained)"
        );
    } else {
        // First install wins: an earlier registration owns the slot and did
        // not materialize a sound backend. Fail closed to the existing paths.
        warn!(
            "process-global sound CROWN slot is owned by an earlier registration without a \
             sound backend; qualified WGPU engine not published (fail-closed)"
        );
    }
    if ny_propagate::sound_gpu_gate::wide_sound_gpu_crown_requested()
        && !ny_propagate::sound_gpu_gate::prewarm_wide_sound_gpu_crown()
    {
        warn!(
            "wide sound CROWN lane requested but its prewarm was unavailable; wide calls fail \
             closed to the serial/CPU path"
        );
    }
}

/// Loud, greppable marker printed whenever the binary REFUSES to execute a
/// requested backend and substitutes another one.
///
/// #capability-cost: the WGPU proof-adapter quarantine (1ede1d30) turned
/// `device: wgpu` in 16 shipped presets into "run the CPU verifier" and emitted
/// only a `tracing::warn`. `RUST_LOG` is ignored on the scored path, so no sweep
/// log carried the warn and two banked categories (vit_2023 9 unsat -> 1,
/// soundnessbench 3/3 sat -> 0) silently zeroed for months. Every future
/// runtime substitution goes through [`resolve_proof_backend`] and prints THIS marker
/// on stderr unconditionally, so `scripts/check_banked_rows.py` can name the
/// cause instead of reporting an unexplained verdict diff.
pub(crate) const BACKEND_OVERRIDE_MARKER: &str = "NY-HARNESS: BACKEND-OVERRIDE";

/// Outcome of asking the binary to honour a resolved backend choice.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HonouredBackend {
    /// The backend proofs will actually execute on.
    pub(crate) effective: BackendArg,
    /// `Some(reason)` exactly when `effective` differs from what was requested.
    pub(crate) override_reason: Option<&'static str>,
}

/// Static admission into runtime proof qualification.
///
/// Backend RESOLUTION (`--backend` > `--gpu` > preset `general.device` > auto)
/// answers what was *asked for*; this answers whether the binary has a public
/// proof construction route for that backend. Live adapter qualification and
/// fallback are deliberately not represented by this pure/static helper; they
/// are recorded by [`ProofBackendReceipt`]. Keeping the layers separate lets
/// `preset::backend_capability_tests` walk every shipped preset's declared
/// `general.device` through this function and fails unless the declaration is
/// either honoured or covered by a dated waiver in
/// `configs/backend_capability_waivers.toml`.
#[cfg(test)]
pub(crate) const fn honour_requested_backend(requested: BackendArg) -> HonouredBackend {
    match requested {
        BackendArg::Wgpu => HonouredBackend {
            effective: BackendArg::Wgpu,
            override_reason: None,
        },
        BackendArg::Cpu => HonouredBackend {
            effective: BackendArg::Cpu,
            override_reason: None,
        },
    }
}

/// Resolve the effective backend from --backend and --gpu flags.
///
/// A non-CPU `--backend` takes precedence. If the parsed backend is the CPU
/// default but legacy `--gpu` is true, use wgpu for backward compatibility.
/// Callers that store `--backend` as a defaulted value (rather than `Option`)
/// must make the two flags conflict in clap: this function cannot distinguish
/// an omitted backend from an explicit `--backend cpu`.
pub(crate) fn resolve_backend(backend: BackendArg, gpu: bool) -> BackendArg {
    if backend != BackendArg::Cpu {
        // --backend was explicitly specified
        backend
    } else if gpu {
        // Legacy --gpu flag, use wgpu for backward compat
        BackendArg::Wgpu
    } else {
        BackendArg::Cpu
    }
}

/// Apply preset `general.device` as a fallback when no CLI flag overrides.
///
/// Precedence: --backend > --gpu > preset general.device > CPU default.
/// Only activates when `cli_backend` is CPU and `gpu` is false (no explicit CLI override).
pub(crate) fn apply_preset_device(
    cli_backend: BackendArg,
    gpu: bool,
    preset_device: Option<&str>,
) -> BackendArg {
    if cli_backend != BackendArg::Cpu || gpu {
        // CLI explicitly set a backend — preset cannot override
        return cli_backend;
    }
    match preset_device {
        Some("wgpu") => BackendArg::Wgpu,
        Some("cpu") | None => BackendArg::Cpu,
        Some(other) => {
            warn!("Unknown preset device '{}', using CPU", other);
            BackendArg::Cpu
        }
    }
}

pub(crate) struct GemmBackendResolution<T> {
    pub(crate) backend: BackendArg,
    pub(crate) device: Option<T>,
}

impl<T> GemmBackendResolution<T> {
    pub(crate) const fn cpu() -> Self {
        Self {
            backend: BackendArg::Cpu,
            device: None,
        }
    }
}

impl<T: GemmEngine> GemmBackendResolution<T> {
    pub(crate) fn gemm_engine(&self) -> Option<&dyn GemmEngine> {
        self.device.as_ref().map(|device| device as &dyn GemmEngine)
    }
}

pub(crate) fn resolve_gemm_backend_with_factory<T, F>(
    backend: BackendArg,
    gpu: bool,
    json: bool,
    build_device: F,
) -> GemmBackendResolution<T>
where
    F: FnOnce(BackendArg) -> anyhow::Result<T>,
{
    let mut effective_backend = resolve_backend(backend, gpu);
    let device = match effective_backend {
        BackendArg::Cpu => None,
        BackendArg::Wgpu => match build_device(effective_backend) {
            Ok(device) => Some(device),
            Err(error) => {
                if !json {
                    warn!("WGPU backend not available: {error}. Using CPU.");
                }
                // Device-init/qualification fallbacks must remain visible even
                // when JSON output suppresses ordinary logging.
                eprintln!(
                    "{BACKEND_OVERRIDE_MARKER}: command=generic-gemm requested=wgpu \
                     source=defaulted_cli_value effective=cpu qualification=refused \
                     failed_rung=unavailable qualification_provenance=unavailable \
                     provenance=cpu-fallback-no-device reason={error}"
                );
                None
            }
        },
    };

    if device.is_none() && effective_backend != BackendArg::Cpu {
        effective_backend = BackendArg::Cpu;
    }

    GemmBackendResolution {
        backend: effective_backend,
        device,
    }
}

pub(crate) fn resolve_gemm_backend(
    backend: BackendArg,
    gpu: bool,
    json: bool,
) -> GemmBackendResolution<ComputeDevice> {
    resolve_gemm_backend_with_factory(backend, gpu, json, |effective_backend| {
        Ok(ComputeDevice::new(effective_backend.into())?)
    })
}

#[cfg(test)]
mod tests {
    use super::{
        apply_preset_device, backend_override_message, engine_for_sound_crown_slots,
        qualify_wgpu_proof_chain, resolve_automatic_wgpu_request_from, resolve_backend,
        resolve_gemm_backend_with_factory, resolve_proof_backend_with_factories, BackendRequest,
        BackendRequestSource, GemmBackendResolution, ProofBackendQualification,
        ProofBackendReceipt, WgpuProofRefusal,
    };
    use crate::BackendArg;

    /// Minimal engine that advertises a sound GPU CROWN backward, standing in
    /// for a qualified (charged or fully-qualified) proof device.
    struct MockSoundCrownEngine;

    impl ny_core::GemmEngine for MockSoundCrownEngine {
        fn gemm_f32(
            &self,
            m: usize,
            _k: usize,
            n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> ny_core::Result<Vec<f32>> {
            Ok(vec![0.0; m * n])
        }
        fn as_gpu_crown_backward(&self) -> Option<&dyn ny_core::GpuCrownBackward> {
            Some(self)
        }
    }

    impl ny_core::GpuCrownBackward for MockSoundCrownEngine {
        fn crown_backward_gpu(
            &self,
            _layers: &[ny_core::GpuCrownLayer],
            _spec: &[f32],
            num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> ny_core::Result<ny_core::GpuCrownResult> {
            Ok(ny_core::GpuCrownResult {
                lower_bounds: vec![0.0; num_specs],
                upper_bounds: vec![0.0; num_specs],
            })
        }
        fn provides_sound_gpu_crown(&self) -> bool {
            true
        }
    }

    fn wgpu_request_from(source: BackendRequestSource) -> BackendRequest {
        BackendRequest {
            backend: BackendArg::Wgpu,
            source,
            selection_reason: None,
        }
    }

    /// #charged-metal-engagement byte-identity pin: the slot-publication
    /// decision is fail-closed. Only a LIVE qualified WGPU receipt paired with
    /// an engine that actually advertises the sound CROWN accessor admits;
    /// CPU builds, refused qualification, and sound-accessor-less engines all
    /// refuse, so unqualified hosts install nothing and stay byte-identical.
    #[test]
    fn sound_crown_slot_publication_is_fail_closed_to_qualified_sound_devices() {
        use std::sync::Arc;
        let sound: Arc<dyn ny_core::GemmEngine> = Arc::new(MockSoundCrownEngine);
        let accessorless: Arc<dyn ny_core::GemmEngine> = Arc::new(ny_core::NaiveCpuGemmEngine);

        // CPU resolution: never published, even with a sound-capable engine.
        let cpu_receipt = ProofBackendReceipt::cpu(
            BackendRequest {
                backend: BackendArg::Cpu,
                source: BackendRequestSource::DefaultedCliValue,
                selection_reason: None,
            },
            "compute-device-cpu",
        );
        assert!(engine_for_sound_crown_slots(&cpu_receipt, &sound).is_none());

        // Refused qualification: the effective backend is CPU; never published.
        let refused = ProofBackendReceipt::refused_wgpu(
            wgpu_request_from(BackendRequestSource::Preset),
            "compute-device-cpu",
            Some("Apple M5 Max (Metal)".to_string()),
            Some("rung_3".to_string()),
            "denormal preservation refused",
        );
        assert!(engine_for_sound_crown_slots(&refused, &sound).is_none());

        // Qualified receipt but the engine exposes no sound CROWN accessor
        // (e.g. an injected test device): refuse rather than fill a slot with
        // an engine every consumer would have to reject.
        let qualified = ProofBackendReceipt::qualified_wgpu(
            wgpu_request_from(BackendRequestSource::Preset),
            "wgpu-qualified-crown-flush-charged",
            "Apple M5 Max (Metal) (CROWN qualified WITH FLUSH CHARGES)",
        );
        assert!(engine_for_sound_crown_slots(&qualified, &accessorless).is_none());

        // The one admitted shape: qualified receipt + sound CROWN accessor.
        let admitted = engine_for_sound_crown_slots(&qualified, &sound)
            .expect("qualified receipt with a sound CROWN accessor must be publishable");
        assert!(
            Arc::ptr_eq(&admitted, &sound),
            "the EXACT device is published"
        );
    }

    #[test]
    fn test_resolve_backend_explicit_wgpu() {
        let result = resolve_backend(BackendArg::Wgpu, false);
        assert_eq!(result, BackendArg::Wgpu);
    }

    #[test]
    fn test_resolve_backend_explicit_wgpu_with_gpu_flag() {
        // --backend takes precedence over --gpu
        let result = resolve_backend(BackendArg::Wgpu, true);
        assert_eq!(result, BackendArg::Wgpu);
    }

    #[test]
    fn test_resolve_backend_legacy_gpu_flag() {
        // When --backend is default (Cpu) and --gpu is true, use wgpu
        let result = resolve_backend(BackendArg::Cpu, true);
        assert_eq!(result, BackendArg::Wgpu);
    }

    #[test]
    fn test_resolve_backend_default_cpu() {
        let result = resolve_backend(BackendArg::Cpu, false);
        assert_eq!(result, BackendArg::Cpu);
    }

    #[test]
    fn automatic_wgpu_policy_preserves_explicit_choices_and_routes_auto() {
        let auto = |candidate, raw: Option<&str>, cuda| {
            resolve_automatic_wgpu_request_from(
                BackendRequest {
                    backend: BackendArg::Cpu,
                    source: BackendRequestSource::Auto,
                    selection_reason: None,
                },
                candidate,
                raw.map(std::ffi::OsStr::new),
                cuda,
            )
        };

        assert_eq!(
            auto(true, None, false)
                .expect("auto WGPU candidate")
                .backend,
            BackendArg::Wgpu
        );
        assert_eq!(
            auto(false, None, false)
                .expect("auto CPU cost profile")
                .backend,
            BackendArg::Cpu
        );
        assert_eq!(
            auto(true, Some("0"), false)
                .expect("auto kill switch")
                .backend,
            BackendArg::Cpu
        );
        assert_eq!(
            auto(false, Some("1"), false)
                .expect("required WGPU")
                .backend,
            BackendArg::Wgpu
        );
        assert_eq!(
            auto(true, None, true)
                .expect("CUDA authority suppresses auto WGPU")
                .backend,
            BackendArg::Cpu
        );
        assert!(auto(true, Some("1"), true).is_err());
        assert!(auto(true, Some("true"), false).is_err());

        for backend in [BackendArg::Cpu, BackendArg::Wgpu] {
            let explicit = BackendRequest {
                backend,
                source: BackendRequestSource::ExplicitBackend,
                selection_reason: None,
            };
            assert_eq!(
                resolve_automatic_wgpu_request_from(
                    explicit,
                    true,
                    Some(std::ffi::OsStr::new("0")),
                    false,
                )
                .expect("explicit backend survives automatic policy")
                .backend,
                backend
            );
        }
        assert!(resolve_automatic_wgpu_request_from(
            BackendRequest {
                backend: BackendArg::Wgpu,
                source: BackendRequestSource::ExplicitBackend,
                selection_reason: None,
            },
            true,
            None,
            true,
        )
        .is_err());

        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let non_unicode = std::ffi::OsString::from_vec(vec![0xff, b'1']);
            assert!(resolve_automatic_wgpu_request_from(
                BackendRequest {
                    backend: BackendArg::Cpu,
                    source: BackendRequestSource::Auto,
                    selection_reason: None,
                },
                true,
                Some(&non_unicode),
                false,
            )
            .is_err());
        }
    }

    #[test]
    fn cuda_crown_zero_does_not_suppress_the_auto_policy() {
        // The injectable seam's `false` is the exact parsed meaning of both
        // unset and `NY_CUDA_CROWN=0`; only exact `1` passes `true`.
        let request = BackendRequest {
            backend: BackendArg::Cpu,
            source: BackendRequestSource::Auto,
            selection_reason: None,
        };
        assert_eq!(
            resolve_automatic_wgpu_request_from(request, true, None, false)
                .expect("CUDA CROWN zero leaves WGPU auto available")
                .backend,
            BackendArg::Wgpu
        );
    }

    #[test]
    fn test_resolve_gemm_backend_with_factory_keeps_cpu_without_device() {
        let resolved: GemmBackendResolution<()> =
            resolve_gemm_backend_with_factory(BackendArg::Cpu, false, false, |_| {
                panic!("cpu backend should not build a device")
            });
        assert_eq!(resolved.backend, BackendArg::Cpu);
        assert!(resolved.device.is_none());
    }

    #[test]
    fn test_resolve_gemm_backend_with_factory_builds_non_cpu_device() {
        let resolved =
            resolve_gemm_backend_with_factory(BackendArg::Wgpu, false, false, |_| Ok(()));
        assert_eq!(resolved.backend, BackendArg::Wgpu);
        assert!(resolved.device.is_some());
    }

    #[test]
    fn test_resolve_gemm_backend_with_factory_falls_back_on_error() {
        let resolved: GemmBackendResolution<()> =
            resolve_gemm_backend_with_factory(BackendArg::Wgpu, false, true, |_| {
                Err(anyhow::anyhow!("wgpu unavailable"))
            });
        assert_eq!(resolved.backend, BackendArg::Cpu);
        assert!(resolved.device.is_none());
    }

    #[test]
    fn test_apply_preset_device_wgpu_when_no_cli_override() {
        // Preset `device: wgpu` activates GPU when CLI uses defaults
        let result = apply_preset_device(BackendArg::Cpu, false, Some("wgpu"));
        assert_eq!(result, BackendArg::Wgpu);
    }

    #[test]
    fn test_apply_preset_device_mlx_falls_back_to_cpu() {
        // mlx is no longer a supported backend — falls back to CPU
        let result = apply_preset_device(BackendArg::Cpu, false, Some("mlx"));
        assert_eq!(result, BackendArg::Cpu);
    }

    #[test]
    fn test_apply_preset_device_cli_backend_takes_precedence() {
        // --backend wgpu already set — preset cpu cannot downgrade
        let result = apply_preset_device(BackendArg::Wgpu, false, Some("cpu"));
        assert_eq!(result, BackendArg::Wgpu);
    }

    #[test]
    fn test_apply_preset_device_gpu_flag_takes_precedence() {
        // --gpu flag already set (resolved to Wgpu) — preset cpu cannot downgrade
        let result = apply_preset_device(BackendArg::Wgpu, true, Some("cpu"));
        assert_eq!(result, BackendArg::Wgpu);
    }

    #[test]
    fn test_apply_preset_device_none_stays_cpu() {
        // No preset device — stays CPU
        let result = apply_preset_device(BackendArg::Cpu, false, None);
        assert_eq!(result, BackendArg::Cpu);
    }

    #[test]
    fn test_apply_preset_device_unknown_falls_back_cpu() {
        // Unknown device string — falls back to CPU with warning
        let result = apply_preset_device(BackendArg::Cpu, false, Some("vulkan"));
        assert_eq!(result, BackendArg::Cpu);
    }

    #[test]
    fn test_apply_preset_device_explicit_cpu() {
        // Preset says cpu — stays CPU
        let result = apply_preset_device(BackendArg::Cpu, false, Some("cpu"));
        assert_eq!(result, BackendArg::Cpu);
    }

    #[test]
    fn proof_backend_receipt_keeps_request_and_refusal_distinct() {
        let request = BackendRequest {
            backend: BackendArg::Wgpu,
            source: BackendRequestSource::Preset,
            selection_reason: None,
        };
        let receipt = ProofBackendReceipt::refused_wgpu(
            request,
            "compute-device-cpu",
            Some("Apple M5 (IntegratedGpu, Metal)".to_string()),
            Some("gradual_underflow".to_string()),
            "live adapter did not preserve subnormals",
        );

        assert_eq!(receipt.requested, BackendArg::Wgpu);
        assert_eq!(receipt.request_source, BackendRequestSource::Preset);
        assert_eq!(receipt.effective, BackendArg::Cpu);
        assert_eq!(receipt.qualification, ProofBackendQualification::Refused);
        assert!(!receipt.qualified_wgpu_active());

        let marker = backend_override_message("beta-crown", &receipt)
            .expect("a fallback receipt must render an override marker");
        for expected in [
            "NY-HARNESS: BACKEND-OVERRIDE",
            "command=beta-crown",
            "requested=wgpu",
            "source=preset",
            "effective=cpu",
            "qualification=refused",
            "failed_rung=gradual_underflow",
            "qualification_provenance=Apple M5 (IntegratedGpu, Metal)",
            "provenance=compute-device-cpu",
            "reason=live adapter did not preserve subnormals",
        ] {
            assert!(
                marker.contains(expected),
                "missing {expected:?} in {marker}"
            );
        }
    }

    #[test]
    fn qualified_backend_receipt_has_no_override_marker() {
        let receipt = ProofBackendReceipt::qualified_wgpu(
            BackendRequest {
                backend: BackendArg::Wgpu,
                source: BackendRequestSource::ExplicitBackend,
                selection_reason: None,
            },
            "compute-device-wgpu-qualified",
            "NVIDIA GB10 (IntegratedGpu, Vulkan)",
        );
        assert!(receipt.qualified_wgpu_active());
        assert!(backend_override_message("verify", &receipt).is_none());
    }

    #[test]
    fn proof_resolution_retains_the_exact_qualified_device() {
        let request = BackendRequest {
            backend: BackendArg::Wgpu,
            source: BackendRequestSource::ExplicitBackend,
            selection_reason: None,
        };
        let resolved = resolve_proof_backend_with_factories(
            request,
            "verify",
            || panic!("qualified WGPU must not construct the CPU fallback"),
            || {
                Ok((
                    "qualified-device",
                    "wgpu-qualified-crown".to_string(),
                    "test adapter".to_string(),
                ))
            },
        )
        .expect("qualified route resolves");

        assert_eq!(resolved.device, "qualified-device");
        assert!(resolved.receipt.qualified_wgpu_active());
        assert_eq!(resolved.receipt.effective, BackendArg::Wgpu);
        assert_eq!(resolved.receipt.provenance, "wgpu-qualified-crown");
        assert_eq!(
            resolved.receipt.qualification_provenance.as_deref(),
            Some("test adapter")
        );
    }

    // -- #flush-charge chain (hermetic) ------------------------------------

    fn wgpu_request() -> BackendRequest {
        BackendRequest {
            backend: BackendArg::Wgpu,
            source: BackendRequestSource::ExplicitBackend,
            selection_reason: None,
        }
    }

    fn uncharged_refusal_fixture() -> WgpuProofRefusal {
        WgpuProofRefusal {
            reason: "gradual underflow refused; source error: subnormals flushed".to_string(),
            failed_rung: Some("gradual underflow".to_string()),
            qualification_provenance: Some("Apple M5 Max (IntegratedGpu, Metal)".to_string()),
        }
    }

    #[test]
    fn charged_attempt_never_preempts_uncharged_qualification() {
        // Order is the contract: a conforming adapter takes full (uncharged)
        // authority and the charged constructor is never consulted.
        let result: Result<(&str, String, String), WgpuProofRefusal> = qualify_wgpu_proof_chain(
            || {
                Ok((
                    "uncharged-device",
                    "wgpu-qualified-crown".to_string(),
                    "conforming adapter".to_string(),
                ))
            },
            || panic!("the charged constructor must not run when the uncharged ladder passes"),
        );
        let (device, provenance, _) = result.expect("uncharged qualification wins");
        assert_eq!(device, "uncharged-device");
        assert_eq!(provenance, "wgpu-qualified-crown");
    }

    #[test]
    fn closed_charged_gate_keeps_the_chain_byte_identical() {
        // The byte-identity pin: while the charged source gate is closed the
        // charged attempt refuses, its refusal is discarded, and the chain's
        // evidence is EXACTLY the uncharged refusal — receipts and override
        // markers cannot differ from the pre-chain binary.
        let expected = uncharged_refusal_fixture();
        let result: Result<((), String, String), WgpuProofRefusal> = qualify_wgpu_proof_chain(
            || Err(uncharged_refusal_fixture()),
            || {
                Err(WgpuProofRefusal {
                    reason: "charged verdict authority source gate is closed".to_string(),
                    failed_rung: None,
                    qualification_provenance: None,
                })
            },
        );
        assert_eq!(result.expect_err("both constructors refused"), expected);
    }

    #[test]
    fn charged_qualification_is_narrated_distinctly() {
        // Pre-flip acceptance evidence for the OPEN-gate state, hermetically:
        // when the uncharged ladder refuses and the charged constructor
        // qualifies, the receipt is active WGPU with charged provenance in
        // both strings — never readable as a fully qualified device.
        let resolved = resolve_proof_backend_with_factories(
            wgpu_request(),
            "verify",
            || panic!("a charged-qualified device must not construct the CPU fallback"),
            || {
                qualify_wgpu_proof_chain(
                    || Err(uncharged_refusal_fixture()),
                    || {
                        Ok((
                            "charged-device",
                            "wgpu-qualified-crown-flush-charged".to_string(),
                            "Apple M5 Max (IntegratedGpu, Metal) \
                             (CROWN qualified WITH FLUSH CHARGES)"
                                .to_string(),
                        ))
                    },
                )
            },
        )
        .expect("charged route resolves");

        assert_eq!(resolved.device, "charged-device");
        assert!(resolved.receipt.qualified_wgpu_active());
        assert_eq!(
            resolved.receipt.provenance,
            "wgpu-qualified-crown-flush-charged"
        );
        assert!(resolved
            .receipt
            .qualification_provenance
            .as_deref()
            .expect("charged qualification keeps adapter identity")
            .contains("WITH FLUSH CHARGES"));
        assert!(
            backend_override_message("verify", &resolved.receipt).is_none(),
            "charged qualification is not a fallback and must not emit the override marker"
        );
    }

    // -- #flush-charge chain (live constructors; box-quiet probe) ----------

    #[test]
    fn charged_chain_is_live_on_this_host_now_that_the_gate_is_open() {
        assert!(
            ny_gpu::wgpu_charged_proof_authority(),
            "the charged source gate closed again: this live-chain pin (and \
             the receipt contract it guards) must be revisited by that source \
             review"
        );

        // The production chain resolves exactly what the live constructors
        // imply, in chain order: uncharged qualification wins outright; a
        // charged admission is narrated distinctly; a double refusal falls
        // back to CPU carrying EXACTLY the uncharged refusal evidence (the
        // charged refusal is discarded).
        let uncharged = match ny_gpu::ComputeDevice::new_for_proof(super::WgpuVerdictRequest::new())
        {
            Ok(device) => {
                use ny_core::GemmEngine as _;
                Ok(device.backend_provenance().to_string())
            }
            Err(error) => Err(super::wgpu_refusal_from(&error)),
        };
        let charged_admits = ny_gpu::ComputeDevice::new_for_proof_flush_charged(
            ny_gpu::WgpuChargedVerdictRequest::new(),
        )
        .is_ok();
        let resolved = super::resolve_proof_backend(wgpu_request(), "test-charged-chain")
            .expect("chain always resolves a device");
        match uncharged {
            Ok(provenance) => {
                // Fully qualified host: the uncharged device wins, untouched
                // — a conforming adapter is never downgraded into charges.
                assert!(resolved.receipt.qualified_wgpu_active());
                assert_eq!(resolved.receipt.provenance, provenance);
            }
            Err(_) if charged_admits => {
                // This box today (Apple M5 Max/Metal): the uncharged ladder
                // refuses at rung 3, the charged ladder admits its own forced
                // plain-WGSL device, and the receipt narrates the charges in
                // BOTH strings — never readable as fully qualified.
                assert!(resolved.receipt.qualified_wgpu_active());
                assert_eq!(
                    resolved.receipt.provenance,
                    "wgpu-qualified-crown-flush-charged"
                );
                assert!(resolved
                    .receipt
                    .qualification_provenance
                    .as_deref()
                    .expect("charged qualification keeps adapter identity")
                    .contains("WITH FLUSH CHARGES"));
            }
            Err(refusal) => {
                assert_eq!(resolved.receipt.effective, BackendArg::Cpu);
                assert_eq!(
                    resolved.receipt.qualification,
                    ProofBackendQualification::Refused
                );
                assert_eq!(
                    resolved.receipt.fallback_reason.as_deref(),
                    Some(refusal.reason.as_str())
                );
                assert_eq!(resolved.receipt.failed_rung, refusal.failed_rung);
                assert_eq!(
                    resolved.receipt.qualification_provenance,
                    refusal.qualification_provenance
                );
                assert!(
                    !resolved
                        .receipt
                        .fallback_reason
                        .as_deref()
                        .unwrap_or_default()
                        .contains("FLUSH CHARGES"),
                    "a discarded charged refusal must leave no charged trace \
                     in the receipt"
                );
            }
        }
    }

    #[test]
    fn proof_resolution_falls_back_to_cpu_with_complete_refusal_evidence() {
        let request = BackendRequest {
            backend: BackendArg::Wgpu,
            source: BackendRequestSource::Auto,
            selection_reason: Some("large input"),
        };
        let resolved = resolve_proof_backend_with_factories(
            request,
            "beta-crown",
            || Ok(("cpu-device", "compute-device-cpu".to_string())),
            || {
                Err(WgpuProofRefusal {
                    reason: "gradual underflow refused".to_string(),
                    failed_rung: Some("gradual underflow".to_string()),
                    qualification_provenance: Some("test Metal adapter".to_string()),
                })
            },
        )
        .expect("CPU fallback resolves");

        assert_eq!(resolved.device, "cpu-device");
        assert_eq!(resolved.receipt.requested, BackendArg::Wgpu);
        assert_eq!(resolved.receipt.effective, BackendArg::Cpu);
        assert_eq!(
            resolved.receipt.qualification,
            ProofBackendQualification::Refused
        );
        assert_eq!(
            resolved.receipt.selection_reason.as_deref(),
            Some("large input")
        );
        assert_eq!(
            resolved.receipt.fallback_reason.as_deref(),
            Some("gradual underflow refused")
        );
    }
}
