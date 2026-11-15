// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU participation for embedded consumers — REACHABLE, measurement-qualified,
//! fail-closed.
//!
//! Enabled by the optional `gpu` feature (OFF by default; enabling it changes
//! no existing consumer's build). Without it, `ny-gpu` is not in this crate's
//! dependency graph at all and a facade consumer cannot construct, register, or
//! even NAME a ny GPU type. With it, the typed verdict request and the device
//! that consumes it become nameable through `ny_api::gpu::*` paths only.
//!
//! # Reachability is not trust
//!
//! Nothing here grants GPU authority. Every route below hands a typed REQUEST
//! to ny, which then runs its live qualification ladder on the exact device it
//! is about to return and refuses — typed, with a complete report — when a rung
//! does not pass. That refusal is the normal outcome on several real adapters
//! (see the platform notes), and it is CORRECT: a refused device means the
//! verdict comes from the proven-sound CPU fold instead.
//!
//! There is deliberately no route through this module that skips a rung, and no
//! route that turns a refusal into a downgrade you cannot see: an unqualified
//! device never reaches the process-global CROWN seam, because
//! [`install_gpu_crown_factory`] accepts only a [`QualifiedGpuCrown`], and that
//! type has no public constructor other than a passing run of the ladder.
//!
//! # The qualification ladder
//!
//! [`ComputeDevice::new_for_proof`] creates exactly one WGPU context, evaluates
//! all five [`WgpuVerdictRung`]s eagerly on it, and returns that same device
//! only when the whole conjunction passes. The five rungs are the IEEE f32
//! model, the error-free-transform primitives, gradual underflow, the host EFT
//! reference, and end-to-end sentinel-taint stickiness. Authority is retained
//! by the exact device that produced the passing report; it cannot be
//! transferred. Every error leaves no armed device behind.
//!
//! A refusal has TWO shapes and a caller must handle both. If the ladder ran,
//! the report names an adapter and [`WgpuVerdictReport::failed_rung`] names the
//! first rung that did not pass. If the device or its verdict-critical
//! pipelines never came up, the ladder never ran: the report's adapter is
//! `None`, every rung reads [`WgpuVerdictRungOutcome::NotRun`], and
//! `failed_rung()` is `None`. Code that assumes a named rung is always present
//! is wrong — that second shape is what any host with no adapter returns (and
//! what an `UnchargedOnly` request returned on Apple/Metal until 216f5aba1,
//! when a SPIR-V passthrough request Metal cannot load poisoned the loading
//! path before the ladder ran).
//!
//! Only the CROWN-backward trait seam opens on a qualified device. GEMM,
//! convolution, IBP, DAG-IBP, and attention stay QUARANTINED and return `Err`
//! on any WGPU device, qualified or not — those kernels carry no certified
//! error term and are not covered by the ladder. So "a qualified GPU device"
//! means "the CROWN backward lane may run GPU-resident", not "matmuls move to
//! the GPU".
//!
//! # What a caller can actually expect, by platform
//!
//! **macOS / Metal — the two routes give OPPOSITE answers, so name the one you
//! mean.** Measured live on 2026-08-27, Apple M4 Pro (IntegratedGpu, Metal), by
//! the two explicit `gpu-tests` probes in this module's tests:
//!
//! * [`FlushChargeAdmission::UnchargedOnly`] → **REFUSED by the ladder**, at
//!   a named rung. Measured 2026-10-02, Apple M5 Max (IntegratedGpu, Metal):
//!   [`WgpuVerdictReport::adapter`] names the adapter with
//!   `denorm_preserve=false policy=auto`, and
//!   [`WgpuVerdictReport::failed_rung`] is `EFT primitives` — the hardware is
//!   pure-flush, so the uncharged ladder cannot pass there (the same 3/5 shape
//!   as `docs/WGPU_METAL_CURRENT_QUALIFICATION_2026-08-11.md`: rungs 2 and 3
//!   fail). Until 216f5aba1 the refusal came BEFORE the ladder instead (adapter
//!   `None`, every rung `NotRun`, reason "WGPU device initialization did not
//!   complete"): AUTO requested DenormPreserve SPIR-V passthrough because Metal
//!   advertises `PASSTHROUGH_SHADERS` for MSL, Metal rejected the module ("does
//!   not contain any code compatible with this backend"), and the fallback
//!   poisoned the loading-path contract. Code must still handle both shapes.
//! * [`FlushChargeAdmission::AllowFlushCharged`] → **QUALIFIED** on that same
//!   host: `backend_provenance()` `== "wgpu-qualified-crown-flush-charged"`,
//!   [`QualifiedGpuCrown::flush_charged`] `== true`, and
//!   [`install_gpu_crown_factory`] materialized the process-global slot. So on
//!   Apple today the GPU CROWN lane is genuinely OPEN — through the charged
//!   route only.
//!
//! Do not read that qualification as "same bounds, now faster". The charged
//! route exists specifically to admit the pure-flush Apple/Metal class, and it
//! pays for it: flush losses become CERTIFIED ADDITIVE widening (looser
//! enclosures, never wrong ones) and four channels are hard-REFUSED because no
//! uniform bounds their loss — subnormal weights, subnormal bias, subnormal
//! activation slopes, and subnormal input-box endpoints — with the EFT
//! compensated channel forbidden outright ([`FlushChargePolicy`]). A layer
//! hitting one of those refusals falls back; a bound that comes back is sound
//! but may be looser than the CPU fold's. Charged admission is still fully
//! measured (rungs 1/4/5 pass, rung 3 refused, measured class PURE-FLUSH on the
//! exact device), so it is a qualification, not a bypass — and the distinct
//! provenance string exists so a receipt can never read as fully qualified.
//!
//! Both outcomes are MEASUREMENTS, not policy: there is no `Backend::Metal`
//! branch anywhere in the qualification path, and no environment variable can
//! move either one. A refusal is a correct outcome, and the right response to
//! it is to keep running on the CPU fold.
//!
//! **Linux / NVIDIA.** This feature routes the WGPU (Vulkan) lane, not cuBLAS.
//! On a Vulkan adapter that honours the `DenormPreserve` SPIR-V injection
//! (`ny-gpu-passthrough`) the full UNCHARGED ladder has measured 5/5 — the
//! GB10 record in `docs/CURRENT_STATE_2026-08-10.md` — and that is the only
//! configuration known to yield UNCHARGED WGPU CROWN authority. Nothing here
//! promises it for an arbitrary Linux adapter: the ladder still decides per
//! device. The native f64 cuBLAS route is a DIFFERENT crate
//! (`ny-cuda`) filling a DIFFERENT seam (`sound_f64_gemm`); it is not part of
//! this feature and is still not reachable from this facade. If you are on
//! Linux+CUDA and want cuBLAS `Dgemm` under the sound f64 folds, take a direct
//! `ny-cuda` dependency and register it through
//! `ny_propagate::sound_f64_gemm::set_sound_f64_gemm_factory`.
//!
//! **Any host with no adapter.** Device initialization fails before the ladder
//! runs. That is still a typed [`GpuCrownRefusal::Qualification`]; its report's
//! [`WgpuVerdictReport::adapter`] is `None` and every rung reads
//! [`WgpuVerdictRungOutcome::NotRun`].
//!
//! # Deliberately NOT exposed
//!
//! * `sound_gpu_gate::set_sound_gpu_crown_required(false)` — the switch that
//!   RELEASES the process-global soundness gate and lets an unqualified fast
//!   f32 GPU CROWN decide a verdict. The facade re-exports only the read-only
//!   [`is_sound_gpu_crown_required`].
//! * `WgpuBabBoundVerdictRequest` / `ComputeDevice::new_for_proof_bab_bound` —
//!   the retained-BaB channel is triple-gated dark and refuses in a static
//!   preflight before device creation. `ComputeDevice` carries the constructor
//!   as an inherent method, but its request type is not nameable through
//!   `ny_api`, so the route cannot be called from facade paths. A name that can
//!   never succeed is a fabricated capability.
//! * The `test_only_*` charged constructor, which is compiled out of every
//!   production build upstream.
//! * `set_sound_gpu_crown_factory` / `set_wide_sound_gpu_crown_factory` in raw
//!   form. [`install_gpu_crown_factory`] is the curated wrapper; it accepts only
//!   an already-qualified device. Consumers with their own accelerator crate can
//!   still reach the raw installers through a direct `ny-propagate` dependency.
//!
//! # One trap worth naming
//!
//! [`ComputeDevice::new`] is re-exported with the type, and
//! `ComputeDevice::new(Backend::Wgpu)` succeeds on any host with an adapter
//! WITHOUT running the ladder. The device it returns is QUARANTINED: every
//! GEMM, conv, IBP and attention entry point returns `Err`,
//! `provides_sound_gpu_crown()` is `false`, and `backend_provenance()` reports
//! `wgpu-quarantined`. It cannot carry a verdict — but reporting "GPU enabled"
//! because that constructor did not error is a fabricated capability report.
//! Decide from [`QualifiedGpuCrown`] instead; it exists only when the ladder
//! passed.
//!
//! # Example
//!
//! ```rust,no_run
//! use ny_api::gpu::{install_gpu_crown_factory, qualify_gpu_crown, FlushChargeAdmission};
//!
//! // Naming `AllowFlushCharged` is what admits the Apple/Metal charged route;
//! // `UnchargedOnly` refuses there, by measurement.
//! match qualify_gpu_crown(FlushChargeAdmission::AllowFlushCharged) {
//!     Ok(qualified) => {
//!         let install = install_gpu_crown_factory(&qualified);
//!         println!(
//!             "GPU CROWN qualified on {} ({}); charged: {}; slot materialized: {}",
//!             qualified.adapter(),
//!             qualified.backend_provenance(),
//!             qualified.flush_charged(),
//!             install.materialized(),
//!         );
//!     }
//!     // Keep running: the CPU fold decides the verdict. `failed_rung()` is
//!     // `None` when the device never came up, so narrate the refusal, do not
//!     // unwrap the rung.
//!     Err(refusal) => println!(
//!         "GPU CROWN refused (fail-closed): {refusal}; first failing rung: {:?}",
//!         refusal.failed_rung(),
//!     ),
//! }
//! ```

use std::sync::Arc;

use ny_core::GemmEngine;

/// Backend selector and the unified compute device that dispatches to it.
///
/// [`ComputeDevice::new_for_proof`] and
/// [`ComputeDevice::new_for_proof_flush_charged`] are the only constructors
/// that run the qualification ladder and retain authority on the exact device
/// they return; see the module docs for the trap in plain
/// [`ComputeDevice::new`].
pub use ny_gpu::{Backend, ComputeDevice};

/// The typed verdict REQUESTS. Neither carries authority: each is consumed
/// exactly once by a constructor that then runs the live ladder. Reaching
/// charged mode requires naming [`WgpuChargedVerdictRequest`] at the call site,
/// which is why neither type implements `Default`.
pub use ny_gpu::{WgpuChargedVerdictRequest, WgpuVerdictRequest};

/// The qualification EVIDENCE surface: the typed refusal, the complete
/// five-rung report it carries, the rung names, and the per-rung outcomes.
pub use ny_gpu::{
    WgpuVerdictQualificationError, WgpuVerdictReport, WgpuVerdictRung, WgpuVerdictRungOutcome,
};

/// The authority state a qualification attempt landed in, and the charged-flush
/// policy a `QualifiedWithFlushCharge` device carries (every field is either a
/// certified WIDENING factor or a hard REFUSAL — see the module docs).
pub use ny_gpu::{FlushChargePolicy, WgpuVerdictAuthority};

/// Build/source capability signals. All are read-only and grant nothing: they
/// report whether a route is COMPILED and whether its reviewed source gate is
/// open, never whether any device qualified.
pub use ny_gpu::{wgpu_backend_compiled, wgpu_charged_proof_authority, wgpu_proof_authority};

/// Adapter enumeration probe. Creates no device, compiles no pipeline, and
/// grants no authority; it exists so routing diagnostics can name the actual
/// Metal/Vulkan/DX12 adapter.
pub use ny_gpu::{wgpu_adapter_available, wgpu_adapter_provenance, WgpuAdapterProvenance};

/// The optional GPU CROWN-backward accessor on an engine, needed to read
/// `provides_sound_gpu_crown()` from an [`ny_core::GemmEngine`].
pub use ny_core::GpuCrownBackward;

/// Read-only: whether the process-global soundness gate is engaged (it is, by
/// default). When engaged, verdict-deciding CROWN routes accept only a backend
/// advertising a qualified SOUND implementation. The releasing setter is
/// deliberately not re-exported — see the module docs.
pub use ny_propagate::sound_gpu_gate::is_sound_gpu_crown_required;

/// Whether a facade qualification attempt may fall through to the
/// CHARGED-flush route after the uncharged ladder refuses.
///
/// This mirrors upstream's discipline that entering charged mode must be
/// visible at the call site. The uncharged constructor is ALWAYS attempted
/// first in both variants, so a conforming adapter can never be downgraded into
/// charged mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushChargeAdmission {
    /// Uncharged authority only. A pure-flush adapter (the Apple/Metal class)
    /// fails rungs 2 and 3 and the whole attempt refuses with that report.
    UnchargedOnly,
    /// After the uncharged ladder refuses, also attempt the CHARGED-flush
    /// route: certified additive widening plus four subnormal refusals, EFT
    /// forbidden. Still fully measured — a non-pure-flush adapter is refused.
    AllowFlushCharged,
}

/// A GPU CROWN engine that PASSED the live qualification ladder on the exact
/// device it wraps.
///
/// There is no public constructor: the only way to obtain one is
/// [`qualify_gpu_crown`], and it is produced only after the typed constructor
/// returned a device AND that device's engine advertised the sound CROWN
/// accessor. Holding this value is therefore evidence, not a request.
#[derive(Clone)]
pub struct QualifiedGpuCrown {
    engine: Arc<dyn GemmEngine>,
    charged: bool,
    adapter: String,
    provenance: &'static str,
}

impl QualifiedGpuCrown {
    /// The qualified device as an injectable engine.
    ///
    /// Suitable for `Verifier::new_with_engine` / `Session::with_engine`, but
    /// note the quarantine: only the CROWN-backward seam is open, so this
    /// engine returns `Err` from `gemm_f32` and friends and the caller's CPU
    /// fallbacks run for everything else. Publishing it to the process-global
    /// CROWN slot with [`install_gpu_crown_factory`] is what actually routes
    /// the sound GPU CROWN lane.
    #[must_use]
    pub fn engine(&self) -> &Arc<dyn GemmEngine> {
        &self.engine
    }

    /// Consume this evidence and recover the injectable engine.
    #[must_use]
    pub fn into_engine(self) -> Arc<dyn GemmEngine> {
        self.engine
    }

    /// Whether authority was granted under the CHARGED-flush policy (certified
    /// additive widening, four subnormal channels refused) rather than the full
    /// uncharged ladder. `true` is the admissible Apple/Metal outcome.
    #[must_use]
    pub const fn flush_charged(&self) -> bool {
        self.charged
    }

    /// Stable adapter identity recorded in the passing report. A charged
    /// qualification marks the string so a receipt cannot read as fully
    /// qualified.
    #[must_use]
    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    /// The engine's own provenance string: `wgpu-qualified-crown` or
    /// `wgpu-qualified-crown-flush-charged`.
    #[must_use]
    pub const fn backend_provenance(&self) -> &'static str {
        self.provenance
    }
}

impl std::fmt::Debug for QualifiedGpuCrown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `engine` is a trait object with no `Debug`; the provenance field
        // below is exactly the identity it would print.
        f.debug_struct("QualifiedGpuCrown")
            .field("engine", &"<Arc<dyn GemmEngine>>")
            .field("adapter", &self.adapter)
            .field("charged", &self.charged)
            .field("provenance", &self.provenance)
            .finish()
    }
}

/// Typed, fail-closed refusal from [`qualify_gpu_crown`].
///
/// Every arm means the same operationally: NO GPU authority was granted and the
/// verdict path is unchanged (the proven-sound CPU fold). The arms differ only
/// in what evidence they can show.
#[derive(Debug)]
pub enum GpuCrownRefusal {
    /// The live ladder refused, or the device never initialized. Carries the
    /// complete UNCHARGED five-rung report; when charged admission was allowed
    /// and also refused, the charged refusal is discarded in favour of this
    /// more diagnostic one (mirroring the upstream chain contract).
    Qualification(Box<WgpuVerdictQualificationError>),
    /// Defensive fail-closed: a typed constructor returned a device, but its
    /// engine did not advertise the sound CROWN accessor. Nothing is published.
    /// Reaching this arm would mean an upstream invariant broke; it is checked
    /// rather than assumed.
    SoundCrownAccessorClosed {
        /// Adapter identity from the device's report, when it had one.
        adapter: Option<String>,
        /// The engine's own provenance string at the moment of the check.
        backend_provenance: &'static str,
    },
}

impl GpuCrownRefusal {
    /// The complete five-rung report, when this refusal came from the ladder.
    #[must_use]
    pub fn report(&self) -> Option<&WgpuVerdictReport> {
        match self {
            Self::Qualification(error) => Some(error.report()),
            Self::SoundCrownAccessorClosed { .. } => None,
        }
    }

    /// First rung that did not pass. `None` when the device never initialized
    /// (so no rung ran) or when the refusal did not come from the ladder.
    #[must_use]
    pub fn failed_rung(&self) -> Option<WgpuVerdictRung> {
        self.report().and_then(WgpuVerdictReport::failed_rung)
    }
}

impl std::fmt::Display for GpuCrownRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Qualification(error) => write!(f, "{error}"),
            Self::SoundCrownAccessorClosed {
                adapter,
                backend_provenance,
            } => write!(
                f,
                "WGPU verdict qualification reported success but the sound CROWN accessor is \
                 closed on the returned engine (adapter: {}, provenance: {backend_provenance}); \
                 nothing was published (fail-closed)",
                adapter.as_deref().unwrap_or("unknown"),
            ),
        }
    }
}

impl std::error::Error for GpuCrownRefusal {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Qualification(error) => Some(error.as_ref()),
            Self::SoundCrownAccessorClosed { .. } => None,
        }
    }
}

/// Ask ny to qualify one WGPU device for verdict-bearing GPU CROWN.
///
/// Constructs exactly one device per attempt and runs the live ladder on it.
/// The uncharged constructor is attempted first in both admission modes; the
/// charged constructor is consulted only when `admission` is
/// [`FlushChargeAdmission::AllowFlushCharged`] AND the uncharged ladder
/// refused. When both refuse, the UNCHARGED refusal is returned — it carries
/// the more diagnostic rung report.
///
/// A refusal is a normal outcome, not an error in the caller's setup. See the
/// module docs for what each platform should expect.
pub fn qualify_gpu_crown(
    admission: FlushChargeAdmission,
) -> Result<QualifiedGpuCrown, GpuCrownRefusal> {
    let uncharged_refusal = match ComputeDevice::new_for_proof(WgpuVerdictRequest::new()) {
        Ok(device) => return admit(device, false),
        Err(error) => error,
    };

    match admission {
        FlushChargeAdmission::UnchargedOnly => {
            Err(GpuCrownRefusal::Qualification(Box::new(uncharged_refusal)))
        }
        FlushChargeAdmission::AllowFlushCharged => {
            match ComputeDevice::new_for_proof_flush_charged(WgpuChargedVerdictRequest::new()) {
                Ok(device) => admit(device, true),
                // The charged refusal is discarded on purpose: the uncharged
                // rung report is the more diagnostic evidence, and keeping it
                // makes the refusal bytes independent of the charged attempt.
                Err(_charged_refusal) => {
                    Err(GpuCrownRefusal::Qualification(Box::new(uncharged_refusal)))
                }
            }
        }
    }
}

/// Turn a device the ladder accepted into evidence — re-checking, rather than
/// assuming, that its engine really advertises the sound CROWN accessor.
fn admit(device: ComputeDevice, charged: bool) -> Result<QualifiedGpuCrown, GpuCrownRefusal> {
    let provenance = device.backend_provenance();
    let adapter = device
        .wgpu_verdict_report()
        .and_then(WgpuVerdictReport::adapter)
        .map(str::to_string);

    let sound_crown = device
        .as_gpu_crown_backward()
        .is_some_and(GpuCrownBackward::provides_sound_gpu_crown);
    if !sound_crown {
        return Err(GpuCrownRefusal::SoundCrownAccessorClosed {
            adapter,
            backend_provenance: provenance,
        });
    }

    let adapter = adapter.map_or_else(
        || "qualified adapter identity unavailable".to_string(),
        |adapter| {
            if charged {
                format!("{adapter} (CROWN qualified WITH FLUSH CHARGES)")
            } else {
                adapter
            }
        },
    );

    Ok(QualifiedGpuCrown {
        engine: Arc::new(device),
        charged,
        adapter,
        provenance,
    })
}

/// Result of publishing a qualified device into the process-global sound GPU
/// CROWN slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct GpuCrownInstall {
    materialized: bool,
    wide_lane_requested: bool,
    wide_lane_materialized: bool,
}

impl GpuCrownInstall {
    /// Whether a SOUND GPU CROWN backend now occupies the ordinary
    /// process-global slot.
    ///
    /// `false` means an earlier registration already owned the slot without a
    /// sound backend: first install wins, nothing was published, and every
    /// verdict-deciding route stays on the proven-sound CPU fold.
    #[must_use]
    pub const fn materialized(self) -> bool {
        self.materialized
    }

    /// Whether the wide, domain-stacked CROWN lane was requested for this
    /// process (`NY_CUDA_WIDE` / `NY_HYDRA_CROWN`; the names are historical,
    /// the lane is backend-agnostic).
    #[must_use]
    pub const fn wide_lane_requested(self) -> bool {
        self.wide_lane_requested
    }

    /// Whether the wide lane's backend materialized. Meaningful only when
    /// [`Self::wide_lane_requested`] is `true`; when the lane was requested and
    /// this is `false`, wide calls fail closed to the serial/CPU path.
    #[must_use]
    pub const fn wide_lane_materialized(self) -> bool {
        self.wide_lane_materialized
    }
}

/// Publish an already-qualified device into the process-global sound GPU CROWN
/// slots that borrow-only consumers read.
///
/// This is the seam that actually routes the sound GPU CROWN lane: the
/// deadline-preinitialized sequential and margin-row routes consult only the
/// PREINITIALIZED slot, so materialization has to happen before any finite
/// deadline authority exists — which is why this prewarms rather than
/// registering a lazy factory. The registered factory hands back a clone of the
/// already-materialized `Arc`, so the prewarm performs no GPU work and builds
/// no second context.
///
/// Fail-closed properties, all inherited rather than re-implemented:
///
/// * only a [`QualifiedGpuCrown`] can be passed, so an unqualified or
///   quarantined device can never reach the slot;
/// * installation is first-install-wins, so a process that already registered
///   an accelerator (e.g. CUDA at startup) keeps its precedence and this call
///   silently loses — reported as `materialized() == false` when that earlier
///   owner has no sound backend;
/// * `ny-propagate` re-filters the slot on `provides_sound_gpu_crown()` at
///   every read, so even a wrongly-published engine cannot decide a bound;
/// * the sound-f64 GEMM seam (`sound_f64_gemm`) is deliberately NOT filled: its
///   contract assumes an IEEE f64 `Dgemm` (cuBLAS/Accelerate) and a WGPU device
///   is f32-only. Substituting it there would be an assumption, not a
///   qualification.
pub fn install_gpu_crown_factory(qualified: &QualifiedGpuCrown) -> GpuCrownInstall {
    let ordinary = Arc::clone(&qualified.engine);
    ny_propagate::sound_gpu_gate::set_sound_gpu_crown_factory(move || Some(Arc::clone(&ordinary)));
    let wide = Arc::clone(&qualified.engine);
    ny_propagate::sound_gpu_gate::set_wide_sound_gpu_crown_factory(move || Some(Arc::clone(&wide)));

    let materialized = ny_propagate::sound_gpu_gate::prewarm_sound_gpu_crown();
    let wide_lane_requested = ny_propagate::sound_gpu_gate::wide_sound_gpu_crown_requested();
    let wide_lane_materialized =
        wide_lane_requested && ny_propagate::sound_gpu_gate::prewarm_wide_sound_gpu_crown();

    GpuCrownInstall {
        materialized,
        wide_lane_requested,
        wide_lane_materialized,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        install_gpu_crown_factory, is_sound_gpu_crown_required, qualify_gpu_crown,
        wgpu_backend_compiled, wgpu_charged_proof_authority, wgpu_proof_authority, Backend,
        ComputeDevice, FlushChargeAdmission, GpuCrownBackward, GpuCrownRefusal,
        WgpuChargedVerdictRequest, WgpuVerdictRequest, WgpuVerdictRung, WgpuVerdictRungOutcome,
    };
    use ny_core::GemmEngine;

    /// The whole point of the feature: the typed request is CONSTRUCTIBLE
    /// through `ny_api` paths, with no `ny_gpu` import at the call site.
    #[test]
    fn the_typed_verdict_requests_are_constructible_from_the_facade() {
        let uncharged = WgpuVerdictRequest::new();
        let charged = WgpuChargedVerdictRequest::new();
        assert_eq!(uncharged, WgpuVerdictRequest::new());
        assert_eq!(charged, WgpuChargedVerdictRequest::new());
    }

    /// The build/source signals are readable and say nothing about any device.
    #[test]
    fn build_capability_signals_are_readable_and_grant_nothing() {
        assert!(
            wgpu_backend_compiled(),
            "the `gpu` feature must pull ny-gpu WITH its wgpu backend, or the typed \
             constructors do not exist"
        );
        // Reviewed source gates. Both are open today; neither is authority.
        let _uncharged_gate: bool = wgpu_proof_authority();
        let _charged_gate: bool = wgpu_charged_proof_authority();
    }

    /// The soundness gate a verdict-deciding CROWN route consults is engaged by
    /// default, and the facade exposes only the READ.
    #[test]
    fn the_process_soundness_gate_is_engaged_by_default() {
        assert!(
            is_sound_gpu_crown_required(),
            "verdict-deciding CROWN must accept only a qualified sound backend by default"
        );
    }

    /// A CPU device is not a CROWN provider, so the admission predicate that
    /// guards [`QualifiedGpuCrown`] refuses it. Hermetic: no adapter needed.
    #[test]
    fn a_cpu_device_never_advertises_the_sound_crown_accessor() {
        let device = ComputeDevice::new(Backend::Cpu).expect("CPU device is always available");
        assert_eq!(device.backend(), Backend::Cpu);
        assert_eq!(device.backend_provenance(), "compute-device-cpu");
        assert!(
            device.as_gpu_crown_backward().is_none(),
            "CPU has no GPU CROWN backward accessor"
        );
        assert!(
            !device
                .as_gpu_crown_backward()
                .is_some_and(GpuCrownBackward::provides_sound_gpu_crown),
            "the admission predicate must refuse a CPU device"
        );
    }

    /// A CPU device also refuses the quarantined op family, so a caller cannot
    /// mistake "constructed" for "accelerated".
    #[test]
    fn a_cpu_device_reports_no_gpu_gemm() {
        let device = ComputeDevice::new(Backend::Cpu).expect("CPU device is always available");
        // Below the offload gate this is the documented Err; the point here is
        // that it is a typed Err, not a panic.
        let result = device.gemm_f32(2, 2, 2, &[1.0, 0.0, 0.0, 1.0], &[1.0, 0.0, 0.0, 1.0]);
        assert!(
            result.is_err(),
            "CPU ComputeDevice has no GPU GEMM to offer"
        );
    }

    /// LIVE: run the real UNCHARGED ladder on this host's adapter and narrate
    /// the five-rung report.
    ///
    /// This is the probe that says what an `UnchargedOnly` caller gets here.
    /// On the Apple/Metal class it REFUSES, and the assertion below is the one
    /// that matters: a refusal is a typed value with a complete report, not a
    /// panic and not a silently-downgraded success.
    ///
    /// Measured 2026-08-27, Apple M4 Pro (IntegratedGpu, Metal):
    /// refused; rungs 1/4/5 passed, rungs 2 and 3 failed, `failed_rung()`
    /// reported `EFT primitives` (the first non-passing rung in ladder order).
    ///
    /// Run with:
    /// `targo --unverified test -p ny-api --features gpu-tests --lib
    /// gpu::tests::live_uncharged_qualification_reports_its_rungs -- --exact --nocapture`
    #[cfg(feature = "gpu-tests")]
    #[test]
    fn live_uncharged_qualification_reports_its_rungs() {
        match qualify_gpu_crown(FlushChargeAdmission::UnchargedOnly) {
            Ok(qualified) => {
                println!(
                    "UNCHARGED QUALIFIED: adapter={} provenance={}",
                    qualified.adapter(),
                    qualified.backend_provenance()
                );
                assert!(
                    !qualified.flush_charged(),
                    "UnchargedOnly must never return a charged device"
                );
                assert_eq!(qualified.backend_provenance(), "wgpu-qualified-crown");
            }
            Err(GpuCrownRefusal::Qualification(error)) => {
                let report = error.report();
                println!(
                    "UNCHARGED REFUSED (fail-closed): adapter={:?} first_failed_rung={:?}\n  \
                     reason={}\n  source={}",
                    report.adapter(),
                    report.failed_rung().map(|rung| rung.to_string()),
                    report.reason(),
                    error.source_error(),
                );
                for rung in [
                    WgpuVerdictRung::IeeeF32Model,
                    WgpuVerdictRung::EftPrimitives,
                    WgpuVerdictRung::GradualUnderflow,
                    WgpuVerdictRung::HostEftReference,
                    WgpuVerdictRung::SentinelTaintSticky,
                ] {
                    println!("  rung {rung}: {:?}", report.outcome(rung));
                }
                assert!(
                    !report.qualified(),
                    "a refusal must not carry a qualified report"
                );
            }
            Err(refusal @ GpuCrownRefusal::SoundCrownAccessorClosed { .. }) => {
                panic!("upstream invariant broke: {refusal}");
            }
        }
    }

    /// LIVE: run the real ladder on this host's adapter, charged route allowed.
    ///
    /// Feature-gated because it creates a WGPU context and compiles the
    /// verdict-critical pipelines — deliberate, one at a time, never in an
    /// ordinary bulk run. What it pins is the CONTRACT, on any hardware: the attempt returns
    /// a typed `Result` and never panics; a refusal carries an inspectable
    /// report; a success really does advertise the sound CROWN accessor (the
    /// admission predicate in `admit` already enforced that) and can be
    /// published to the process-global slot.
    ///
    /// Measured 2026-08-27, Apple M4 Pro (IntegratedGpu, Metal): QUALIFIED
    /// with `wgpu-qualified-crown-flush-charged`, `flush_charged() == true`,
    /// process-global slot materialized. So on Apple today the charged route is
    /// OPEN — a refusal is the correct expectation only for `UnchargedOnly`.
    ///
    /// Run with:
    /// `targo --unverified test -p ny-api --features gpu-tests --lib
    /// gpu::tests::live_qualification_is_typed_and_fail_closed -- --exact --nocapture`
    #[cfg(feature = "gpu-tests")]
    #[test]
    fn live_qualification_is_typed_and_fail_closed() {
        match qualify_gpu_crown(FlushChargeAdmission::AllowFlushCharged) {
            Ok(qualified) => {
                assert!(
                    qualified
                        .engine()
                        .as_gpu_crown_backward()
                        .is_some_and(GpuCrownBackward::provides_sound_gpu_crown),
                    "a qualified device must advertise the sound CROWN accessor"
                );
                assert!(
                    qualified
                        .backend_provenance()
                        .starts_with("wgpu-qualified-crown"),
                    "provenance must narrate the qualified state, got {}",
                    qualified.backend_provenance()
                );
                assert_eq!(
                    qualified.flush_charged(),
                    qualified.backend_provenance() == "wgpu-qualified-crown-flush-charged",
                    "the charged flag and the provenance string must agree"
                );
                let install = install_gpu_crown_factory(&qualified);
                assert!(
                    install.materialized(),
                    "a freshly qualified engine must materialize the process-global slot"
                );
                // Narration, not decoration: an operator running this probe
                // deliberately needs to read WHAT this host admitted.
                println!(
                    "QUALIFIED: adapter={} provenance={} charged={} slot_materialized={}",
                    qualified.adapter(),
                    qualified.backend_provenance(),
                    qualified.flush_charged(),
                    install.materialized(),
                );
            }
            Err(GpuCrownRefusal::Qualification(error)) => {
                let report = error.report();
                println!(
                    "REFUSED (fail-closed): adapter={:?} first_failed_rung={:?} reason={}",
                    report.adapter(),
                    report.failed_rung().map(|rung| rung.to_string()),
                    report.reason(),
                );
                for rung in [
                    WgpuVerdictRung::IeeeF32Model,
                    WgpuVerdictRung::EftPrimitives,
                    WgpuVerdictRung::GradualUnderflow,
                    WgpuVerdictRung::HostEftReference,
                    WgpuVerdictRung::SentinelTaintSticky,
                ] {
                    println!("  rung {rung}: {:?}", report.outcome(rung));
                }
                // Either the device never initialized (no adapter: every rung
                // NotRun, no adapter identity) or a rung refused. Both are
                // complete, inspectable evidence — never a panic, never a
                // silent downgrade.
                if report.adapter().is_none() {
                    for rung in [
                        WgpuVerdictRung::IeeeF32Model,
                        WgpuVerdictRung::EftPrimitives,
                        WgpuVerdictRung::GradualUnderflow,
                        WgpuVerdictRung::HostEftReference,
                        WgpuVerdictRung::SentinelTaintSticky,
                    ] {
                        assert_eq!(report.outcome(rung), WgpuVerdictRungOutcome::NotRun);
                    }
                    assert!(report.failed_rung().is_none());
                } else {
                    assert!(
                        !report.qualified(),
                        "a refusal must not carry a qualified report"
                    );
                    assert!(
                        report.failed_rung().is_some(),
                        "a refusal on a live adapter must name the rung that did not pass"
                    );
                }
                assert!(!format!("{report:?}").is_empty());
            }
            Err(refusal @ GpuCrownRefusal::SoundCrownAccessorClosed { .. }) => {
                panic!("upstream invariant broke: {refusal}");
            }
        }
    }
}
