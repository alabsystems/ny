// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Process-global soundness gate for the GPU CROWN fast-path.
//!
//! # Why this exists (VNN-COMP 2026 soundness)
//!
//! The CPU CROWN path is PROVEN sound: it computes the backward `A·W` product in
//! f64 and adds a certified `γ_n·S` rounding-error term, with zero-tolerance
//! exact-rational oracle tests (see
//! `crates/ny-propagate/src/tests/proptest_soundness/crown_linear_aw_soundness.rs`).
//!
//! The GPU CROWN fast-path is NOT sound for verdicts. The
//! `CROWN_CONCRETIZE_SHADER` (ny-gpu `wgpu_device/shaders.rs`) computes
//! `a_pos·x_l + a_neg·x_u` in round-to-nearest f32, and the FAST backward `A·W`
//! shaders carry NO `γ_n·S` certified error.
//!
//! CORRECTED 2026-08-04 — this paragraph used to add "(WGSL has no f64, so the
//! CPU's exact approach cannot port directly)", and five downstream sites came
//! to quote that as the reason ALL GPU CROWN is off the verdict path. It is a
//! statement about the FAST lane only, and the parenthetical inference is false.
//! The SOUND-resident wgpu lane already carries a certified `γ_k·S` term in pure
//! f32 (`CROWN_AW_ERROR_COMBINE_SHADER`; host-side outward-rounded uniforms —
//! ny-gpu `wgpu_device/sound_consts.rs:13`: "no f64 ever enters a WGSL body"),
//! and the EFT/double-single channel supplies an f64-grade compensated residual
//! without f64, with primitives measured bit-exact on Apple M5 Max/Metal
//! (2026-08-04). U1/U3/U4/U5/U6 and B0 are now discharged, and the raw
//! `WgpuDevice` CROWN source gate is open. Its authority still requires an
//! explicit typed request and a passing live five-rung adapter ladder. The
//! public `ComputeDevice`/CLI proof router now exposes only that qualified
//! CROWN accessor and falls back to CPU on refusal; raw low-level WGPU
//! operations are not proof-authorized. This is not contingent on f64. See
//! `docs/CURRENT_STATE_2026-08-10.md`.
//!
//! An uncorrected fast-lane GPU-concretized bound can therefore be *tighter than
//! the true range* — and an over-tight bound on the verdict path can flip a
//! genuinely-violated instance to `Verified`/`unsat`/`hold`. In VNN-COMP one
//! incorrect verdict scores -150.
//!
//! Crucially this applies to *intermediate* GPU CROWN bounds as well, not only the
//! final concretization: an over-tight intermediate pre-activation bound silently
//! tightens every downstream bound, including the one that decides the verdict.
//! So when soundness is required, CROWN propagation that can decide a
//! `Verified`/`unsat`/`hold` must use a qualified sound GPU backward or the
//! proven-sound CPU fallback throughout.
//!
//! # How the gate works
//!
//! Verdict-deciding CROWN sites route through [`gpu_crown_backward_route`] or its
//! deadline/DAG siblings. With the gate engaged, those routers admit only an engine
//! that advertises a sound GPU implementation; otherwise they return `None` and the
//! caller takes the CPU sound fallback. [`sound_gpu_crown_backward`] is the stricter
//! test-only primitive that masks every GPU backward while the gate is engaged.
//!
//! The same process-global policy covers verdict-deciding GPU IBP routes. It does
//! not disable safe acceleration: qualified sound CROWN/IBP implementations remain
//! eligible, and the PGD/attack (sat-finding) path is unaffected. Sat-finding only
//! ever *under*-approximates (it exhibits a concrete counterexample that is
//! re-checked), so float speed there cannot produce an unsound `Verified`.
//!
//! Production CLI entry points require the gate unconditionally. The retired
//! `--allow-unsound-gpu-crown` compatibility value is rejected before model loading;
//! only an explicit programmatic/test caller can release the gate.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use ny_core::{
    certify_gpu_bab_bound_static_schedule, GemmEngine, GpuBabBoundPhaseDecline,
    GpuBabBoundProviderFailureKind, GpuBabBoundScheduleCertificate,
    GpuBabBoundScheduleCertification, GpuBabBoundStaticScheduleRequest, GpuCrownBackward,
    GpuCrownResult, GpuDagIbpForwardExt, GpuIbpForward,
};

/// Process-global LAZY factory for a sound GPU-resident CROWN backward engine,
/// such as a typed-request WGPU device or CUDA `CudaGemmEngine`. Consulted by
/// [`gpu_crown_backward_route`] as a fallback when the propagation's own
/// `engine` provides no sound GPU CROWN. The factory is built at most once and
/// every backend must advertise the sound capability before it can carry a
/// verdict.
type SharedEngine = Arc<dyn GemmEngine>;
type CrownFactory = Box<dyn Fn() -> Option<SharedEngine> + Send + Sync>;
static SOUND_GPU_CROWN_FACTORY: OnceLock<CrownFactory> = OnceLock::new();
static SOUND_GPU_CROWN_ENGINE: OnceLock<Option<SharedEngine>> = OnceLock::new();
// Separate registration for the domain-stacked ResNet path.  The ordinary
// process-global backend remains opt-in because its host-orchestrated small-net
// path can regress against CPU f64; the unmeasured CUDA-wide route is likewise
// dark until a sealed NVIDIA A/B enables it.
static WIDE_SOUND_GPU_CROWN_FACTORY: OnceLock<CrownFactory> = OnceLock::new();
static WIDE_SOUND_GPU_CROWN_ENGINE: OnceLock<Option<SharedEngine>> = OnceLock::new();
static WIDE_SOUND_GPU_CROWN_STATUS_REPORTED: AtomicBool = AtomicBool::new(false);

/// Explicit provider marker for the exact retained-WGPU BaB phase channel.
///
/// This is deliberately not a blanket trait over [`GemmEngine`] or
/// [`GpuCrownBackward`]. A reviewed `ny-gpu` integration must explicitly
/// implement it for the qualified WGPU owner that holds the stable numerical
/// TCB registration. CUDA-wide and ordinary sound-CROWN engines therefore
/// cannot enter this channel merely by advertising a generic core capability.
///
/// Implementing this trait is security-sensitive: the returned backend may be
/// offered to the verdict-bearing retained-BaB core after the channel rechecks
/// all three live capability predicates. Production must register only the
/// exact source-reviewed WGPU provider/device owner. The marker accessor and
/// the returned backend's `provides_sound_gpu_crown`,
/// `provides_sound_gpu_bab_bound_phase`, and
/// `gpu_bab_bound_numerical_tcb` queries must all be finite, nonblocking,
/// allocation-free, and unable to inspect or mutate accelerator resources.
pub trait RetainedWgpuBabPhaseProvider: Send + Sync {
    /// Borrow the exact provider-owned CROWN backend.
    ///
    /// The reference must remain stable for this provider's lifetime. This
    /// query must satisfy the finite, nonblocking, allocation-free, and
    /// resource-inaccessible live-observation contract above.
    fn gpu_crown_backward(&self) -> &dyn GpuCrownBackward;
}

type SharedRetainedWgpuBabProvider = Arc<dyn RetainedWgpuBabPhaseProvider>;
type RetainedWgpuBabFactory = Box<dyn Fn() -> Option<SharedRetainedWgpuBabProvider> + Send + Sync>;

#[derive(Clone, Copy)]
#[repr(u8)]
enum RetainedWgpuBabProviderFault {
    InvalidCapabilities = 1,
    Panicked = 2,
}

struct MaterializedRetainedWgpuBabProvider {
    provider: SharedRetainedWgpuBabProvider,
    // Zero means healthy/unclassified; nonzero is the absorbing first fault.
    // Atomic publication keeps the finite accessor wait-free with respect to
    // concurrent observers.
    fault: AtomicU8,
}

enum RetainedWgpuBabProviderMaterialization {
    Unavailable,
    FactoryPanicked,
    Present(MaterializedRetainedWgpuBabProvider),
}

// This exact-device channel is intentionally independent of both generic
// sound CROWN and CUDA-wide selection. It is default-dark: this crate contains
// no production registration call and consults no environment variable.
static RETAINED_WGPU_BAB_PROVIDER_FACTORY: OnceLock<RetainedWgpuBabFactory> = OnceLock::new();
static RETAINED_WGPU_BAB_PROVIDER: OnceLock<RetainedWgpuBabProviderMaterialization> =
    OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WideBackendStatus {
    Ready,
    Unavailable,
    Unsound,
}

/// Install the process-global sound GPU CROWN backward factory (idempotent; first
/// wins). The factory must yield an engine whose `as_gpu_crown_backward()` returns
/// a `provides_sound_gpu_crown()` backend (cuBLAS f64), or `None` if unavailable.
pub fn set_sound_gpu_crown_factory<F>(factory: F)
where
    F: Fn() -> Option<SharedEngine> + Send + Sync + 'static,
{
    let _ = SOUND_GPU_CROWN_FACTORY.set(Box::new(factory));
}

/// Install the process-global sound backend used specifically by wide,
/// domain-stacked CROWN calls (idempotent; first wins).
///
/// This registration is deliberately independent of
/// [`set_sound_gpu_crown_factory`]: callers may enable the CUDA proof forest
/// without routing ordinary small/non-wide CROWN calls away from their existing
/// CPU or WGPU path.
pub fn set_wide_sound_gpu_crown_factory<F>(factory: F)
where
    F: Fn() -> Option<SharedEngine> + Send + Sync + 'static,
{
    let _ = WIDE_SOUND_GPU_CROWN_FACTORY.set(Box::new(factory));
}

/// Install the exact retained-WGPU BaB provider factory (first install wins).
///
/// Installation is explicit and does not construct a device. The factory is
/// used only by [`prewarm_retained_wgpu_bab_phase_provider`]; finite verifier
/// work can observe only the already-materialized provider or its persisted
/// failure class. There is no generic/CUDA fallback and no environment-driven
/// activation.
///
/// A generic engine cannot be registered without a reviewed explicit marker
/// implementation:
///
/// ```compile_fail
/// use std::sync::Arc;
/// use ny_core::GemmEngine;
/// use ny_propagate::sound_gpu_gate::set_retained_wgpu_bab_phase_provider_factory;
///
/// fn register_generic(engine: Arc<dyn GemmEngine>) {
///     set_retained_wgpu_bab_phase_provider_factory(move || Some(engine.clone()));
/// }
/// ```
pub fn set_retained_wgpu_bab_phase_provider_factory<F>(factory: F)
where
    F: Fn() -> Option<SharedRetainedWgpuBabProvider> + Send + Sync + 'static,
{
    let _ = RETAINED_WGPU_BAB_PROVIDER_FACTORY.set(Box::new(factory));
}

/// Finite, noninitializing observation of the exact retained-WGPU channel.
///
/// Only [`Self::ColdOrUnconfigured`] and [`Self::Unavailable`] permit the
/// untouched legacy path before any core phase is opened: neither state owns a
/// provider or numerical-TCB authority. A present provider with invalid live
/// capabilities or a panic is an explicit authority contradiction and must be
/// mapped to a terminal, no-fallback outcome by the phase owner.
#[derive(Clone, Copy)]
#[must_use]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) enum RetainedWgpuBabProviderObservation<'a> {
    /// No materialization was attempted or no factory was explicitly installed.
    ColdOrUnconfigured,
    /// The installed factory ran and explicitly returned no provider.
    Unavailable,
    /// The exact marked provider passed every live capability query.
    Ready(&'a dyn GpuCrownBackward),
    /// A marked provider was present but failed at least one live predicate.
    InvalidCapabilities,
    /// The factory, marker accessor, or a live capability query panicked.
    Panicked,
}

impl RetainedWgpuBabProviderObservation<'_> {
    /// Whether an untouched pre-open caller may retain the legacy host path.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn allows_preopen_legacy_fallback(self) -> bool {
        matches!(self, Self::ColdOrUnconfigured | Self::Unavailable)
    }
}

/// Finite pre-descriptor schedule observation for the exact retained-WGPU slot.
///
/// A certificate owns no payload borrow. Only cold/unavailable state or a core-
/// validated clean decline permits the untouched legacy path. Provider
/// contradictions/panics are absorbed into the materialized slot's sticky
/// first-fault byte; request deadlines and any registration-unavailable result
/// remain request-local, nonsticky, terminal, and nonfallback.
#[must_use]
#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::large_enum_variant)] // Keep finite certification allocation-free.
pub(crate) enum RetainedWgpuBabScheduleObservation {
    ColdOrUnconfigured,
    Unavailable,
    CleanDecline(GpuBabBoundPhaseDecline),
    Certified(GpuBabBoundScheduleCertificate),
    /// Nonfallback failure for this request that does not fault the provider.
    Terminal(GpuBabBoundProviderFailureKind),
    InvalidCapabilities,
    Panicked,
}

impl RetainedWgpuBabScheduleObservation {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn allows_preopen_legacy_fallback(&self) -> bool {
        matches!(
            self,
            Self::ColdOrUnconfigured | Self::Unavailable | Self::CleanDecline(_)
        )
    }
}

fn retained_wgpu_bab_provider_fault_observation<'a>(
    fault: RetainedWgpuBabProviderFault,
) -> RetainedWgpuBabProviderObservation<'a> {
    match fault {
        RetainedWgpuBabProviderFault::InvalidCapabilities => {
            RetainedWgpuBabProviderObservation::InvalidCapabilities
        }
        RetainedWgpuBabProviderFault::Panicked => RetainedWgpuBabProviderObservation::Panicked,
    }
}

fn retained_wgpu_bab_provider_fault_code_observation<'a>(
    fault: u8,
) -> Option<RetainedWgpuBabProviderObservation<'a>> {
    match fault {
        0 => None,
        value if value == RetainedWgpuBabProviderFault::InvalidCapabilities as u8 => {
            Some(RetainedWgpuBabProviderObservation::InvalidCapabilities)
        }
        // `Panicked` and every impossible/corrupt nonzero state fail closed.
        _ => Some(RetainedWgpuBabProviderObservation::Panicked),
    }
}

/// Catch a provider-boundary unwind without running an arbitrary panic-payload
/// destructor after authority classification. The one-time payload is leaked
/// deliberately: a hostile or buggy `Drop` implementation can panic again and
/// escape an otherwise caught provider failure. Sticky materialization/fault
/// state ensures a classified boundary is not re-entered.
fn catch_quarantined_retained_wgpu_bab_unwind<T>(call: impl FnOnce() -> T) -> Result<T, ()> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)) {
        Ok(value) => Ok(value),
        Err(payload) => {
            std::mem::forget(payload);
            Err(())
        }
    }
}

fn record_retained_wgpu_bab_provider_fault(
    provider: &MaterializedRetainedWgpuBabProvider,
    fault: RetainedWgpuBabProviderFault,
) -> RetainedWgpuBabProviderObservation<'_> {
    let proposed = fault as u8;
    let winner =
        match provider
            .fault
            .compare_exchange(0, proposed, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => proposed,
            Err(existing) => existing,
        };
    retained_wgpu_bab_provider_fault_code_observation(winner)
        .unwrap_or_else(|| retained_wgpu_bab_provider_fault_observation(fault))
}

fn observe_materialized_retained_wgpu_bab_provider(
    materialized: &RetainedWgpuBabProviderMaterialization,
) -> RetainedWgpuBabProviderObservation<'_> {
    let provider = match materialized {
        RetainedWgpuBabProviderMaterialization::Unavailable => {
            return RetainedWgpuBabProviderObservation::Unavailable;
        }
        RetainedWgpuBabProviderMaterialization::FactoryPanicked => {
            return RetainedWgpuBabProviderObservation::Panicked;
        }
        RetainedWgpuBabProviderMaterialization::Present(provider) => provider,
    };
    if let Some(fault) =
        retained_wgpu_bab_provider_fault_code_observation(provider.fault.load(Ordering::Acquire))
    {
        return fault;
    }

    let queried = catch_quarantined_retained_wgpu_bab_unwind(|| {
        let gpu = provider.provider.gpu_crown_backward();
        // Do not short-circuit: every required live predicate is inside this
        // panic boundary and every successful observation checks all three.
        let sound_crown = gpu.provides_sound_gpu_crown();
        let phase_capability = gpu.provides_sound_gpu_bab_bound_phase();
        let exposes_tcb = gpu.gpu_bab_bound_numerical_tcb().is_some();
        (sound_crown && phase_capability && exposes_tcb).then_some(gpu)
    });
    match queried {
        Ok(Some(gpu)) => retained_wgpu_bab_provider_fault_code_observation(
            provider.fault.load(Ordering::Acquire),
        )
        .unwrap_or(RetainedWgpuBabProviderObservation::Ready(gpu)),
        Ok(None) => record_retained_wgpu_bab_provider_fault(
            provider,
            RetainedWgpuBabProviderFault::InvalidCapabilities,
        ),
        Err(()) => record_retained_wgpu_bab_provider_fault(
            provider,
            RetainedWgpuBabProviderFault::Panicked,
        ),
    }
}

fn materialize_registered_retained_wgpu_bab_provider<'a>(
    factory: &'a OnceLock<RetainedWgpuBabFactory>,
    provider: &'a OnceLock<RetainedWgpuBabProviderMaterialization>,
) -> RetainedWgpuBabProviderObservation<'a> {
    let Some(factory) = factory.get() else {
        return RetainedWgpuBabProviderObservation::ColdOrUnconfigured;
    };
    let materialized =
        provider.get_or_init(
            || match catch_quarantined_retained_wgpu_bab_unwind(factory) {
                Ok(Some(provider)) => RetainedWgpuBabProviderMaterialization::Present(
                    MaterializedRetainedWgpuBabProvider {
                        provider,
                        fault: AtomicU8::new(0),
                    },
                ),
                Ok(None) => RetainedWgpuBabProviderMaterialization::Unavailable,
                Err(()) => RetainedWgpuBabProviderMaterialization::FactoryPanicked,
            },
        );
    observe_materialized_retained_wgpu_bab_provider(materialized)
}

#[cfg_attr(not(test), allow(dead_code))]
fn preinitialized_retained_wgpu_bab_provider_from_slot(
    provider: &OnceLock<RetainedWgpuBabProviderMaterialization>,
) -> RetainedWgpuBabProviderObservation<'_> {
    match provider.get() {
        Some(materialized) => observe_materialized_retained_wgpu_bab_provider(materialized),
        None => RetainedWgpuBabProviderObservation::ColdOrUnconfigured,
    }
}

fn retained_wgpu_bab_schedule_fault(
    failure: GpuBabBoundProviderFailureKind,
) -> Option<RetainedWgpuBabProviderFault> {
    match failure {
        GpuBabBoundProviderFailureKind::SoundnessGatePanicked
        | GpuBabBoundProviderFailureKind::AccessorPanicked
        | GpuBabBoundProviderFailureKind::RegistrationPanicked
        | GpuBabBoundProviderFailureKind::SchedulePanicked
        | GpuBabBoundProviderFailureKind::PolicyPanicked
        | GpuBabBoundProviderFailureKind::PreparationPanicked
        | GpuBabBoundProviderFailureKind::DormantSessionDropPanicked => {
            Some(RetainedWgpuBabProviderFault::Panicked)
        }
        GpuBabBoundProviderFailureKind::DeadlineExpired
        | GpuBabBoundProviderFailureKind::RegistrationUnavailable => None,
        GpuBabBoundProviderFailureKind::InvalidScheduleEvidence
        | GpuBabBoundProviderFailureKind::RegistrationChanged
        | GpuBabBoundProviderFailureKind::InvalidPolicy => {
            Some(RetainedWgpuBabProviderFault::InvalidCapabilities)
        }
    }
}

fn retained_wgpu_bab_schedule_fault_observation(
    observation: RetainedWgpuBabProviderObservation<'_>,
) -> RetainedWgpuBabScheduleObservation {
    match observation {
        RetainedWgpuBabProviderObservation::InvalidCapabilities => {
            RetainedWgpuBabScheduleObservation::InvalidCapabilities
        }
        RetainedWgpuBabProviderObservation::Panicked => {
            RetainedWgpuBabScheduleObservation::Panicked
        }
        RetainedWgpuBabProviderObservation::ColdOrUnconfigured
        | RetainedWgpuBabProviderObservation::Unavailable
        | RetainedWgpuBabProviderObservation::Ready(_) => {
            // Only sticky fault observations enter this adapter. Treat an
            // impossible nonfault value as provider corruption.
            RetainedWgpuBabScheduleObservation::Panicked
        }
    }
}

fn postcheck_retained_wgpu_bab_schedule_provider(
    materialized: &RetainedWgpuBabProviderMaterialization,
    materialized_provider: &MaterializedRetainedWgpuBabProvider,
    expected_gpu: &dyn GpuCrownBackward,
) -> Option<RetainedWgpuBabScheduleObservation> {
    match observe_materialized_retained_wgpu_bab_provider(materialized) {
        RetainedWgpuBabProviderObservation::Ready(gpu) if std::ptr::eq(gpu, expected_gpu) => None,
        RetainedWgpuBabProviderObservation::InvalidCapabilities => {
            Some(RetainedWgpuBabScheduleObservation::InvalidCapabilities)
        }
        RetainedWgpuBabProviderObservation::Panicked => {
            Some(RetainedWgpuBabScheduleObservation::Panicked)
        }
        RetainedWgpuBabProviderObservation::ColdOrUnconfigured
        | RetainedWgpuBabProviderObservation::Unavailable
        | RetainedWgpuBabProviderObservation::Ready(_) => Some(
            retained_wgpu_bab_schedule_fault_observation(record_retained_wgpu_bab_provider_fault(
                materialized_provider,
                RetainedWgpuBabProviderFault::InvalidCapabilities,
            )),
        ),
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn certify_preinitialized_retained_wgpu_bab_schedule_from_slot(
    provider_slot: &OnceLock<RetainedWgpuBabProviderMaterialization>,
    request: &GpuBabBoundStaticScheduleRequest<'_>,
) -> RetainedWgpuBabScheduleObservation {
    let Some(materialized) = provider_slot.get() else {
        return RetainedWgpuBabScheduleObservation::ColdOrUnconfigured;
    };
    match materialized {
        RetainedWgpuBabProviderMaterialization::Unavailable => {
            return RetainedWgpuBabScheduleObservation::Unavailable;
        }
        RetainedWgpuBabProviderMaterialization::FactoryPanicked => {
            return RetainedWgpuBabScheduleObservation::Panicked;
        }
        RetainedWgpuBabProviderMaterialization::Present(provider) => {
            if let Some(fault) = retained_wgpu_bab_provider_fault_code_observation(
                provider.fault.load(Ordering::Acquire),
            ) {
                return retained_wgpu_bab_schedule_fault_observation(fault);
            }
        }
    }
    if std::time::Instant::now() >= request.deadline() {
        return RetainedWgpuBabScheduleObservation::Terminal(
            GpuBabBoundProviderFailureKind::DeadlineExpired,
        );
    }
    if request.logical_static_device_bytes() > request.requested_max_device_bytes() {
        return RetainedWgpuBabScheduleObservation::CleanDecline(
            GpuBabBoundPhaseDecline::InsufficientCapacity,
        );
    }
    let live = observe_materialized_retained_wgpu_bab_provider(materialized);
    let gpu = match live {
        RetainedWgpuBabProviderObservation::ColdOrUnconfigured => {
            return RetainedWgpuBabScheduleObservation::ColdOrUnconfigured;
        }
        RetainedWgpuBabProviderObservation::Unavailable => {
            return RetainedWgpuBabScheduleObservation::Unavailable;
        }
        RetainedWgpuBabProviderObservation::InvalidCapabilities => {
            return RetainedWgpuBabScheduleObservation::InvalidCapabilities;
        }
        RetainedWgpuBabProviderObservation::Panicked => {
            return RetainedWgpuBabScheduleObservation::Panicked;
        }
        RetainedWgpuBabProviderObservation::Ready(gpu) => gpu,
    };
    let RetainedWgpuBabProviderMaterialization::Present(materialized_provider) = materialized
    else {
        return RetainedWgpuBabScheduleObservation::Panicked;
    };

    let outcome = certify_gpu_bab_bound_static_schedule(gpu, request);
    match outcome {
        GpuBabBoundScheduleCertification::ProviderFailure(failure) => {
            let kind = failure.kind();
            match retained_wgpu_bab_schedule_fault(kind) {
                Some(fault) => retained_wgpu_bab_schedule_fault_observation(
                    record_retained_wgpu_bab_provider_fault(materialized_provider, fault),
                ),
                None => RetainedWgpuBabScheduleObservation::Terminal(kind),
            }
        }
        GpuBabBoundScheduleCertification::CleanDecline(reason) => {
            postcheck_retained_wgpu_bab_schedule_provider(materialized, materialized_provider, gpu)
                .unwrap_or(RetainedWgpuBabScheduleObservation::CleanDecline(reason))
        }
        GpuBabBoundScheduleCertification::Certified(certificate) => {
            postcheck_retained_wgpu_bab_schedule_provider(materialized, materialized_provider, gpu)
                .unwrap_or(RetainedWgpuBabScheduleObservation::Certified(certificate))
        }
    }
}

/// Explicitly materialize and validate the exact retained-WGPU BaB provider.
///
/// Call this before creating finite verifier authority. `false` means no
/// factory was installed, construction declined, or the resulting provider did
/// not expose all three required live capabilities, including by panicking. A
/// factory `None`, factory panic, invalid materialized provider, or provider
/// panic is persisted so the later typed finite observation cannot mistake it
/// for a cold channel. A call with no installed factory does not seal the
/// provider slot, so a later explicit startup registration can still be
/// prewarmed. This boolean is startup telemetry only and must never authorize
/// legacy fallback; only the later typed finite observation carries that
/// authority.
#[must_use]
pub fn prewarm_retained_wgpu_bab_phase_provider() -> bool {
    matches!(
        materialize_registered_retained_wgpu_bab_provider(
            &RETAINED_WGPU_BAB_PROVIDER_FACTORY,
            &RETAINED_WGPU_BAB_PROVIDER,
        ),
        RetainedWgpuBabProviderObservation::Ready(_)
    )
}

/// Borrow an already-materialized exact retained-WGPU BaB provider.
///
/// This finite-authority accessor reads only the materialized slot and the
/// provider's three required live capability queries. It may seal an
/// allocation-free fault classification with a nonblocking atomic first-writer
/// operation, but it never invokes or waits for the factory, never consults
/// generic/CUDA slots, and never reads an environment switch. Only
/// `ColdOrUnconfigured` and `Unavailable` authorize the untouched pre-open
/// legacy host path; `InvalidCapabilities` and `Panicked` are
/// terminal/no-fallback evidence for the future phase owner.
#[inline]
#[allow(dead_code)] // Default-dark bridge seam; the phase owner wires it in the next slice.
pub(crate) fn preinitialized_retained_wgpu_bab_phase_provider(
) -> RetainedWgpuBabProviderObservation<'static> {
    preinitialized_retained_wgpu_bab_provider_from_slot(&RETAINED_WGPU_BAB_PROVIDER)
}

/// Certify a static schedule through only the already-materialized exact WGPU
/// provider. This never invokes a factory, reads an environment switch, opens
/// a phase, or constructs a graph plan. Provider contradictions/panics covered
/// by the sticky mapping become sticky channel state. A deadline or any
/// registration-unavailable result remains request-local terminal/nonfallback;
/// only cold/unavailable/clean-decline permit fallback.
#[inline]
#[allow(dead_code)] // Default-dark pre-descriptor seam; no runtime caller yet.
pub(crate) fn preinitialized_retained_wgpu_bab_schedule(
    request: &GpuBabBoundStaticScheduleRequest<'_>,
) -> RetainedWgpuBabScheduleObservation {
    certify_preinitialized_retained_wgpu_bab_schedule_from_slot(
        &RETAINED_WGPU_BAB_PROVIDER,
        request,
    )
}

/// The lazily-materialized global sound GPU CROWN backend, if installed + sound.
fn global_sound_gpu_crown() -> Option<&'static dyn GpuCrownBackward> {
    let engine = SOUND_GPU_CROWN_ENGINE
        .get_or_init(|| SOUND_GPU_CROWN_FACTORY.get().and_then(|factory| factory()));
    engine
        .as_ref()
        .and_then(|e| e.as_gpu_crown_backward())
        .filter(|g| g.provides_sound_gpu_crown())
}

/// Observe the ordinary process-global sound backend without running its lazy
/// factory or waiting for another thread to finish initialization.
#[inline]
fn preinitialized_global_sound_gpu_crown() -> Option<&'static dyn GpuCrownBackward> {
    SOUND_GPU_CROWN_ENGINE
        .get()
        .and_then(Option::as_ref)
        .and_then(|engine| engine.as_gpu_crown_backward())
        .filter(|gpu| gpu.provides_sound_gpu_crown())
}

fn select_lazy_backend_for_deadline<T>(
    deadline: Option<std::time::Instant>,
    preinitialized: impl FnOnce() -> Option<T>,
    initialize: impl FnOnce() -> Option<T>,
) -> Option<T> {
    if deadline.is_some() {
        preinitialized()
    } else {
        initialize()
    }
}

/// Explicitly materialize the ordinary sound GPU CROWN backend before a
/// verifier creates finite deadline authority.
///
/// The CLI calls this after explicitly registering either the legacy CUDA route
/// or a typed-request WGPU verdict route. Finite propagation itself never
/// invokes this function.
#[must_use]
pub fn prewarm_sound_gpu_crown() -> bool {
    global_sound_gpu_crown().is_some()
}

fn env_switch_value(raw: Option<&str>) -> Option<bool> {
    raw.and_then(|raw| match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" => Some(true),
        "0" | "false" | "off" => Some(false),
        _ => None,
    })
}

fn wide_sound_gpu_crown_requested_from_values(
    cuda_wide: Option<&str>,
    hydra_crown: Option<&str>,
) -> bool {
    env_switch_value(cuda_wide).unwrap_or_else(|| env_switch_value(hydra_crown).unwrap_or(false))
}

fn resident_cut_shadow_requested_from_value(raw: Option<&str>) -> bool {
    raw == Some("1")
}

fn resident_cut_shadow_backend_from_gate<'a>(
    raw: Option<&str>,
    resolve: impl FnOnce() -> Option<&'a dyn GpuCrownBackward>,
) -> Option<&'a dyn GpuCrownBackward> {
    if !resident_cut_shadow_requested_from_value(raw) {
        return None;
    }
    resolve()
        .filter(|gpu| gpu.provides_sound_gpu_crown())
        .filter(|gpu| gpu.provides_resident_cut_shadow())
}

fn root_joint_deadline_lane_requested_from_values(
    root_joint: Option<&str>,
    deadline_ascent: Option<&str>,
) -> bool {
    root_joint == Some("1") && deadline_ascent == Some("1")
}

/// Headroom the GPU-suffix shortcut needs before a finite request may enter it.
///
/// The pre-dispatch host work (seed construction, input endpoint copies, the
/// extracted layer `Vec`) is infallible and O(N) with no poll point, so a GPU
/// deadline receipt cannot authorize it — that is the invariant the historical
/// decline was protecting, and it is a real one. Rather than ignore it, we admit
/// only when the remaining budget comfortably exceeds that unpollable prologue.
const GPU_SUFFIX_PREDISPATCH_HEADROOM: std::time::Duration = std::time::Duration::from_secs(3);

/// #gpu-suffix-expiry: does finite authority refuse the GPU-suffix shortcut?
///
/// Third member of the `deadline`-PRESENCE degradation class, and on cifar100 the
/// one that actually binds: the root phase is dominated by the GPU-resident lane
/// (`[root-comprehensive-gpu-interm-sweep]`), not the CPU DAG patches walk, so
/// gating the CPU set-mates alone left their engagement probe at zero across a
/// provenance-checked 12-run A/B.
///
/// The historical form was `if let Some(_) = deadline { return Ok(None) }` — i.e.
/// ANY live, unexpired deadline forced the CPU relation path. Since every scored
/// run carries one, the GPU shortcut was unreachable in competition and reachable
/// nowhere else.
///
/// * lever OFF (default): always refuses under a deadline. Byte-identical.
/// * lever ON: refuses only when expired or within
///   [`GPU_SUFFIX_PREDISPATCH_HEADROOM`] of expiry, so the unpollable prologue
///   cannot overrun the request.
#[must_use]
pub(crate) fn gpu_suffix_declines_under_finite_authority(limit: std::time::Instant) -> bool {
    // Reads the SAME lever declaration as the patches set-mates
    // (`patches_step::expiry_authority_armed`) rather than calling it, because
    // `network::core` is a private module. Latched identically: this predicate
    // runs per node per target per iteration, and `env::var_os` there is a lock
    // plus a scan of the whole environment block. Sharing the DECLARATION is what
    // keeps the lanes from drifting; sharing the function would be nicer but the
    // module graph does not allow it.
    static ARMED: OnceLock<bool> = OnceLock::new();
    let armed = *ARMED.get_or_init(|| {
        ny_levers::read(&ny_levers::decls::diagnostics::PATCHES_FINITE_EXPIRY)
            .value
            .as_bool()
    });
    if !armed {
        return true;
    }
    limit.saturating_duration_since(std::time::Instant::now()) < GPU_SUFFIX_PREDISPATCH_HEADROOM
}

/// Whether the finite-deadline root-joint lane has both of its exact, default-
/// dark gates armed.
///
/// Keep this check outside CUDA engine admission so a single gate can never
/// initialize an accelerator or consume verifier wall time.
#[must_use]
pub(crate) fn root_joint_deadline_lane_requested() -> bool {
    let root_joint = std::env::var("NY_ROOT_JOINT_INTERM_ALPHA").ok();
    let deadline_ascent = std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT").ok();
    root_joint_deadline_lane_requested_from_values(
        root_joint.as_deref(),
        deadline_ascent.as_deref(),
    )
}

/// Exact call-local capability admitted at the finite-deadline root-joint seam.
///
/// Construction is private to this module. A value therefore proves that the
/// selected engine exposes a verdict-sound CROWN backend, the bounded resident
/// joint adjoint, and a nonzero bounded-row sound-fold capacity no larger than
/// NY's audited K=8 contract. The candidate's general GemmEngine authority is
/// deliberately not retained, so an unsupported CUDA engine cannot fall back
/// to the quarantined WGPU proof adapter.
#[derive(Clone, Copy)]
pub(crate) struct RootJointDeadlineGpu<'a> {
    gpu: &'a dyn GpuCrownBackward,
    sound_fold_max_rows: usize,
}

impl<'a> RootJointDeadlineGpu<'a> {
    fn from_engine(engine: &'a dyn GemmEngine) -> Option<Self> {
        let gpu = engine
            .as_gpu_crown_backward()
            .filter(|gpu| gpu.provides_sound_gpu_crown())
            .filter(|gpu| gpu.provides_deadline_bounded_joint_alpha_gradient_resident())?;
        let sound_fold_max_rows = gpu.deadline_bounded_resnet_sound_max_rows();
        if !(1..=ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS).contains(&sound_fold_max_rows) {
            return None;
        }
        Some(Self {
            gpu,
            sound_fold_max_rows,
        })
    }

    #[inline]
    pub(crate) fn backend(self) -> &'a dyn GpuCrownBackward {
        self.gpu
    }

    #[inline]
    pub(crate) fn sound_fold_max_rows(self) -> usize {
        self.sound_fold_max_rows
    }
}

/// Invoke `f` only when one exact candidate engine exposes the sanctioned
/// root-joint CUDA capabilities.
///
/// The root may try its call-local engine first and then use this same selector
/// inside NY's deadline-safe sound-f64 factory closure. Returning `None` is a
/// sound no-op; this helper never substitutes an unsound backend.
pub(crate) fn with_root_joint_deadline_gpu<R>(
    engine: &dyn GemmEngine,
    f: impl FnOnce(RootJointDeadlineGpu<'_>) -> R,
) -> Option<R> {
    RootJointDeadlineGpu::from_engine(engine).map(f)
}

/// Whether the experimental CUDA-wide proof forest is explicitly requested.
///
/// An explicit `NY_CUDA_WIDE` value takes precedence over the Hydra master
/// switch, including `NY_CUDA_WIDE=0` disabling this component. Invalid values
/// are treated as absent and fall back to `NY_HYDRA_CROWN`, matching the routing
/// contract used by [`global_sound_gpu_crown_for_wide`].
#[must_use]
pub fn wide_sound_gpu_crown_requested() -> bool {
    let cuda_wide = std::env::var("NY_CUDA_WIDE").ok();
    let hydra_crown = std::env::var("NY_HYDRA_CROWN").ok();
    wide_sound_gpu_crown_requested_from_values(cuda_wide.as_deref(), hydra_crown.as_deref())
}

fn sound_wide_gpu_from_engine(engine: Option<&SharedEngine>) -> Option<&dyn GpuCrownBackward> {
    engine
        .and_then(|engine| engine.as_gpu_crown_backward())
        .filter(|gpu| gpu.provides_sound_gpu_crown())
}

/// Observe an already-materialized lazy engine without running its factory.
///
/// Finite-deadline optional work must use this seam instead of `get_or_init`:
/// a cold CUDA factory may perform driver discovery or device construction and
/// has no caller-supplied deadline.
#[inline]
fn preinitialized_engine(engine: &OnceLock<Option<SharedEngine>>) -> Option<&SharedEngine> {
    engine.get().and_then(Option::as_ref)
}

fn materialize_registered_engine<'a>(
    factory: &'a OnceLock<CrownFactory>,
    engine: &'a OnceLock<Option<SharedEngine>>,
) -> Option<&'a SharedEngine> {
    factory
        .get()
        .and_then(|factory| engine.get_or_init(factory).as_ref())
}

fn sound_wide_gpu_from_preinitialized_engines<'a>(
    requested: bool,
    registered_engine: Option<&'a SharedEngine>,
    legacy_engine: Option<&'a SharedEngine>,
) -> Option<&'a dyn GpuCrownBackward> {
    requested
        .then(|| {
            sound_wide_gpu_from_engine(registered_engine)
                .or_else(|| sound_wide_gpu_from_engine(legacy_engine))
        })
        .flatten()
}

fn wide_backend_status(engine: Option<&SharedEngine>) -> WideBackendStatus {
    match engine {
        None => WideBackendStatus::Unavailable,
        Some(engine) => match engine.as_gpu_crown_backward() {
            Some(gpu) if gpu.provides_sound_gpu_crown() => WideBackendStatus::Ready,
            Some(_) | None => WideBackendStatus::Unsound,
        },
    }
}

fn failed_wide_backend_status(
    wide_engine: Option<&SharedEngine>,
    legacy_engine: Option<&SharedEngine>,
) -> WideBackendStatus {
    if [wide_engine, legacy_engine]
        .into_iter()
        .any(|engine| wide_backend_status(engine) == WideBackendStatus::Unsound)
    {
        WideBackendStatus::Unsound
    } else {
        WideBackendStatus::Unavailable
    }
}

fn report_wide_backend_status_once(status: WideBackendStatus) {
    if WIDE_SOUND_GPU_CROWN_STATUS_REPORTED.swap(true, Ordering::SeqCst) {
        return;
    }
    match status {
        WideBackendStatus::Ready => tracing::warn!(
            "CUDA wide CROWN factory ready: certified GPU proof-forest backend resolved"
        ),
        WideBackendStatus::Unavailable => tracing::warn!(
            "CUDA wide CROWN factory unavailable: no backend was returned; wide dispatches \
             will fail closed to the existing local/CPU path"
        ),
        WideBackendStatus::Unsound => tracing::warn!(
            "CUDA wide CROWN factory rejected: backend does not advertise certified sound \
             GPU CROWN; wide dispatches will fail closed to the existing local/CPU path"
        ),
    }
}

/// Return the global sound backend when the experimental CUDA-wide preference
/// is explicitly enabled. `NY_HYDRA_CROWN=1` is the master experiment switch;
/// an explicit `NY_CUDA_WIDE=0` still disables this component for a factorial
/// A/B. CUDA's independent f64 GEMM acceleration is unaffected.
#[inline]
pub(crate) fn global_sound_gpu_crown_for_wide() -> Option<&'static dyn GpuCrownBackward> {
    if !wide_sound_gpu_crown_requested() {
        None
    } else {
        let registered_engine = materialize_registered_engine(
            &WIDE_SOUND_GPU_CROWN_FACTORY,
            &WIDE_SOUND_GPU_CROWN_ENGINE,
        );
        if let Some(gpu) = sound_wide_gpu_from_engine(registered_engine) {
            report_wide_backend_status_once(WideBackendStatus::Ready);
            return Some(gpu);
        }

        // A caller that explicitly installed only the legacy global backend
        // should still be able to use it for wide calls.
        let legacy_engine =
            materialize_registered_engine(&SOUND_GPU_CROWN_FACTORY, &SOUND_GPU_CROWN_ENGINE);
        if let Some(gpu) = sound_wide_gpu_from_engine(legacy_engine) {
            report_wide_backend_status_once(WideBackendStatus::Ready);
            return Some(gpu);
        }

        report_wide_backend_status_once(failed_wide_backend_status(
            registered_engine,
            legacy_engine,
        ));
        None
    }
}

/// Explicitly materialize the requested wide backend before a verifier creates
/// finite deadline authority.
///
/// Once this returns, finite wide accessors can resolve the backend with
/// `OnceLock::get` only. A refusal leaves all later deadline-bearing work on
/// the existing sound local/CPU fallback.
#[must_use]
pub fn prewarm_wide_sound_gpu_crown() -> bool {
    global_sound_gpu_crown_for_wide().is_some()
}

/// Return an already-materialized sound CUDA-wide backend without ever
/// initializing one.
///
/// This is the finite-deadline counterpart to
/// [`global_sound_gpu_crown_for_wide`]. It deliberately observes only the two
/// engine `OnceLock`s with [`OnceLock::get`]; registered factories are never
/// invoked and a cold or hung factory therefore cannot consume child authority.
/// Returning `None` is a deterministic, sound refusal of optional acceleration.
#[inline]
pub(crate) fn preinitialized_sound_gpu_crown_for_wide() -> Option<&'static dyn GpuCrownBackward> {
    sound_wide_gpu_from_preinitialized_engines(
        wide_sound_gpu_crown_requested(),
        preinitialized_engine(&WIDE_SOUND_GPU_CROWN_ENGINE),
        preinitialized_engine(&SOUND_GPU_CROWN_ENGINE),
    )
}

/// Select the wide backend without permitting lazy initialization once a
/// caller's finite authority has started.
///
/// Unbounded callers retain the historical lazy factory. Finite-deadline
/// callers observe only already-materialized engines and fail closed to their
/// existing local/CPU fallback when neither is ready.
#[inline]
pub(crate) fn sound_gpu_crown_for_wide_with_deadline(
    deadline: Option<std::time::Instant>,
) -> Option<&'static dyn GpuCrownBackward> {
    select_lazy_backend_for_deadline(
        deadline,
        preinitialized_sound_gpu_crown_for_wide,
        global_sound_gpu_crown_for_wide,
    )
}

/// Observe an already-materialized resident Cut-CROWN backend without invoking
/// either process-global factory.
#[inline]
pub(crate) fn preinitialized_sound_gpu_crown_for_cut_shadow(
) -> Option<&'static dyn GpuCrownBackward> {
    let requested = std::env::var("NY_CUT_CROWN_RESIDENT_SHADOW").ok();
    resident_cut_shadow_backend_from_gate(requested.as_deref(), || {
        sound_wide_gpu_from_engine(preinitialized_engine(&WIDE_SOUND_GPU_CROWN_ENGINE))
            .or_else(|| sound_wide_gpu_from_engine(preinitialized_engine(&SOUND_GPU_CROWN_ENGINE)))
    })
}

/// Non-initializing sibling of [`global_sound_gpu_crown_for_cut_shadow`].
///
/// Projected M2 uses this after the historical all-domain result is complete:
/// its small explicit child deadline cannot bound a lazy CUDA factory call, so
/// it may borrow only an engine already materialized by an earlier production
/// route. The already-selected local backend remains an independent fallback.
#[inline]
#[allow(dead_code)] // Non-initializing projected-M2 seam remains default-dark.
pub(crate) fn ready_global_sound_gpu_crown_for_cut_shadow() -> Option<&'static dyn GpuCrownBackward>
{
    let requested = std::env::var("NY_CUT_CROWN_RESIDENT_SHADOW").ok();
    resident_cut_shadow_backend_from_gate(requested.as_deref(), || {
        let registered_engine = WIDE_SOUND_GPU_CROWN_ENGINE
            .get()
            .and_then(|engine| engine.as_ref());
        sound_wide_gpu_from_engine(registered_engine).or_else(|| {
            SOUND_GPU_CROWN_ENGINE
                .get()
                .and_then(|engine| engine.as_ref())
                .and_then(|engine| sound_wide_gpu_from_engine(Some(engine)))
        })
    })
}

/// Process-global flag: when `true` (the DEFAULT), the *unsound* fast f32 GPU CROWN
/// backward is masked so every verdict-deciding CROWN bound comes from a proven-sound
/// path (the sound GPU-resident backward — still GPU-accelerated — or the CPU
/// f64+γ_n·S fallback).
///
/// DEFAULTS TO `true` (#gpu-crown-sound-default, 2026-07-05): a VERIFIER must not
/// decide a verdict on an unsound bound by default. Any entry point that does not
/// explicitly touch the gate is therefore sound. Production CLI entry points do
/// not expose an opt-out: beta-crown's retired compatibility value is rejected
/// before model loading, and verify/VNN-COMP cannot release the gate. Explicit
/// programmatic and test callers can still call `set_sound_gpu_crown_required(false)`.
/// Releasing or engaging this flag does not disable the PGD/attack (sat-finding)
/// path; verdict-deciding GPU CROWN and IBP routes separately require a qualified
/// sound implementation whenever the flag is engaged.
/// The DEFAULT of the process-global gate — ONE source of truth (the static
/// initialiser + the `production_gate_default_is_sound` regression test read it).
pub(crate) const DEFAULT_SOUND_GPU_CROWN: bool = true;
static SOUND_GPU_CROWN_REQUIRED: AtomicBool = AtomicBool::new(DEFAULT_SOUND_GPU_CROWN);

/// Engage or release the soundness gate on the GPU CROWN fast-path.
///
/// When `required` is `true`, verdict-deciding CROWN and IBP dispatches may use only
/// a backend that advertises the corresponding qualified sound implementation. A
/// route with no such backend returns `None`, so its caller takes the proven-sound
/// CPU fallback. The unqualified fast GPU f32 path cannot decide a verdict; the
/// PGD/attack path is unaffected.
///
/// This is a process-global switch (not per-call) so a single call at the
/// verification entry point covers every CROWN site — including ones reached
/// through BaB sub-verifiers and graph suffixes that do not thread an explicit
/// soundness flag. Idempotent; safe to call from the single-threaded setup phase
/// before propagation begins.
pub fn set_sound_gpu_crown_required(required: bool) {
    SOUND_GPU_CROWN_REQUIRED.store(required, Ordering::SeqCst);
}

/// Whether the soundness gate is currently engaged.
#[inline]
pub fn is_sound_gpu_crown_required() -> bool {
    SOUND_GPU_CROWN_REQUIRED.load(Ordering::SeqCst)
}

/// Resolve the GPU CROWN backward accelerator for a CROWN dispatch site,
/// honoring the soundness gate.
///
/// Returns `Some(&dyn GpuCrownBackward)` only when (a) the engine actually
/// provides a GPU CROWN backward AND (b) the soundness gate is NOT engaged. When
/// the gate is engaged this returns `None`, so the caller takes its proven-sound
/// CPU fallback — even if a fully-capable GPU engine was handed in.
///
/// NOTE (corrected 2026-07-05): the five verdict-relevant CROWN sites now call
/// [`gpu_crown_backward_route`] (which *permits* a sound GPU-resident backward to
/// decide a verdict under the gate), NOT this helper. This `None`-masking variant
/// has only test callers and is retained as the strict "GPU never decides a bound"
/// primitive; do not describe it as the production verdict router.
#[inline]
#[allow(dead_code)] // retained strict "GPU never decides a bound" primitive; only test callers (see NOTE above)
pub(crate) fn sound_gpu_crown_backward(
    engine: Option<&dyn GemmEngine>,
) -> Option<&dyn GpuCrownBackward> {
    if is_sound_gpu_crown_required() {
        // Soundness required: never let GPU f32 CROWN decide a bound.
        return None;
    }
    engine.and_then(|e| e.as_gpu_crown_backward())
}

/// Route a verdict-deciding CROWN backward to the GPU, honoring the gate.
///
/// Returns `Some((engine, use_sound))`:
/// - gate engaged AND the engine advertises a SOUND GPU-resident backward
///   (`provides_sound_gpu_crown`) → `(engine, true)`: the caller uses
///   `crown_backward_gpu_sound`, whose bounds are a certified enclosure, and
///   still falls back to the CPU sound path on `Err`/NaN (the 0-wrong moat holds);
/// - gate NOT engaged → `(engine, false)`: the existing fast (unsound) GPU path;
/// - gate engaged but no sound GPU path available → `None`: CPU sound fallback.
///
/// Keeps the gate's guarantee — GPU f32 round-to-nearest never decides a bound —
/// while letting the *sound* GPU-resident backward (directed/over-bounded error
/// throughout) carry verdicts at GPU speed.
#[inline]
#[allow(dead_code)] // retained as the public-in-crate unbounded lazy-routing primitive
pub(crate) fn gpu_crown_backward_route(
    engine: Option<&dyn GemmEngine>,
) -> Option<(&dyn GpuCrownBackward, bool)> {
    gpu_crown_backward_route_with_deadline(engine, None)
}

/// Validate the scalar interval payload returned by a GPU CROWN boundary.
///
/// Capability admission establishes which backend may be called; it does not
/// make a malformed device payload authoritative. Every verdict-adjacent
/// consumer must reject the whole payload when either side has the wrong row
/// count, a non-finite endpoint, or an inverted interval. Rejection means the
/// caller takes its established CPU/forward-bound fallback -- never that it
/// repairs a device result or turns a recoverable refusal into `Unknown`.
#[inline]
pub(crate) fn gpu_crown_result_is_publishable(
    result: &GpuCrownResult,
    expected_rows: usize,
) -> bool {
    gpu_interval_payload_is_publishable(&result.lower_bounds, &result.upper_bounds, expected_rows)
}

/// Slice-level form for GPU calls that return bounds alongside gradients or
/// other advisory data rather than as a standalone [`GpuCrownResult`].
#[inline]
pub(crate) fn gpu_interval_payload_is_publishable(
    lower_bounds: &[f32],
    upper_bounds: &[f32],
    expected_rows: usize,
) -> bool {
    lower_bounds.len() == expected_rows
        && upper_bounds.len() == expected_rows
        && lower_bounds
            .iter()
            .zip(upper_bounds)
            .all(|(&lower, &upper)| lower.is_finite() && upper.is_finite() && lower <= upper)
}

/// Deadline-aware form of [`gpu_crown_backward_route`].
///
/// A passed engine is already materialized and remains eligible. The optional
/// process-global fallback is different: under finite authority this resolver
/// observes only the published `OnceLock` value and never invokes or waits on
/// its factory. Truly unbounded callers retain the historical lazy behavior.
#[inline]
pub(crate) fn gpu_crown_backward_route_with_deadline(
    engine: Option<&dyn GemmEngine>,
    deadline: Option<std::time::Instant>,
) -> Option<(&dyn GpuCrownBackward, bool)> {
    if is_sound_gpu_crown_required() {
        // Gate engaged: only a SOUND GPU backward may decide a bound. Prefer the
        // propagation's own engine; else the process-global CUDA sound CROWN.
        if let Some(g) = engine
            .and_then(|e| e.as_gpu_crown_backward())
            .filter(|g| g.provides_sound_gpu_crown())
        {
            return Some((g, true));
        }
        return select_lazy_backend_for_deadline(
            deadline,
            preinitialized_global_sound_gpu_crown,
            global_sound_gpu_crown,
        )
        .map(|g| (g, true));
    }
    // Gate not engaged: the passed engine's (fast) GPU path; else fall back to the
    // process-global CUDA sound CROWN (always sound), if installed.
    if let Some(g) = engine.and_then(|e| e.as_gpu_crown_backward()) {
        return Some((g, false));
    }
    select_lazy_backend_for_deadline(
        deadline,
        preinitialized_global_sound_gpu_crown,
        global_sound_gpu_crown,
    )
    .map(|g| (g, true))
}

/// Route a verdict-deciding IBP FORWARD to the GPU, honoring the gate.
///
/// The IBP counterpart of [`gpu_crown_backward_route`] (`docs/SOUND_GPU_IBP_PLAN.md`
/// §6.2). It **reuses the SAME process-global flag** — one verdict switch covers
/// CROWN backward AND IBP forward — so `is_sound_gpu_crown_required()` also gates
/// the IBP path (the flag name keeps the `crown` spelling for API stability;
/// semantically it is now "sound GPU required" for every verdict op).
///
/// Returns `Some((engine, use_sound))`:
/// - gate engaged AND the engine advertises a SOUND GPU IBP forward
///   (`provides_sound_gpu_ibp`) → `(engine, true)`: the caller uses
///   `ibp_forward_gpu_sound` (a certified enclosure) and still falls back to the
///   CPU sound loop on `Err`/NaN (the 0-wrong moat holds);
/// - gate engaged but no sound GPU IBP available → `None`: CPU sound fallback;
/// - gate NOT engaged → `(engine, false)`: the existing fast (unsound) GPU IBP
///   speed path.
///
/// Unlike [`gpu_crown_backward_route`] there is no process-global sound-IBP factory
/// fallback here — a sound native (e.g. CUDA) IBP forward is reached through the
/// propagation's own `engine.as_gpu_ibp_forward()`.
///
/// NOTE (T1.0 dependency): callers must restrict the sound route to SEQUENTIAL
/// dense-chain networks until T1.0 forwards the sound flag through the DAG/graph
/// IBP accessor; graph verdicts must keep falling through to the CPU sound loop.
#[inline]
#[allow(dead_code)] // wired into `propagate_ibp_sound` in the §6.3 Keystone phase (T1.1)
pub(crate) fn gpu_ibp_forward_route(
    engine: Option<&dyn GemmEngine>,
) -> Option<(&dyn GpuIbpForward, bool)> {
    if is_sound_gpu_crown_required() {
        // Gate engaged: only a SOUND GPU IBP forward may decide a bound. Return
        // None (→ CPU sound fallback) unless the engine advertises one.
        return engine
            .and_then(|e| e.as_gpu_ibp_forward())
            .filter(|g| g.provides_sound_gpu_ibp())
            .map(|g| (g, true));
    }
    // Gate not engaged: the existing fast (unsound) GPU IBP speed path.
    engine
        .and_then(|e| e.as_gpu_ibp_forward())
        .map(|g| (g, false))
}

#[derive(Default)]
struct GpuCrownDeadlineLeaseState {
    by_backend: HashMap<usize, Vec<(Arc<()>, std::time::Instant)>>,
}

fn gpu_crown_deadline_leases() -> &'static Mutex<GpuCrownDeadlineLeaseState> {
    static LEASES: OnceLock<Mutex<GpuCrownDeadlineLeaseState>> = OnceLock::new();
    LEASES.get_or_init(|| Mutex::new(GpuCrownDeadlineLeaseState::default()))
}

fn gpu_crown_backend_identity(gpu: &dyn GpuCrownBackward) -> usize {
    let pointer: *const dyn GpuCrownBackward = gpu;
    pointer.cast::<()>() as usize
}

/// One compositional cooperative-deadline lease on an exact GPU backend.
///
/// The backend trait exposes one deadline slot, while verifier work can overlap
/// or nest. The process-local lease registry therefore retains every active
/// deadline for the same backend and writes their minimum into that slot. A
/// scope drop removes only its own token and restores the next-earliest active
/// deadline; only the final drop clears the backend. The registry lock remains
/// held while updating the backend slot so concurrent acquire/drop operations
/// cannot reorder their writes.
///
/// # Identity and ownership contract
///
/// The key is the data address of the exact `GpuCrownBackward` trait object
/// retained by the scope. Callers that already selected a backend must use
/// `GpuCrownBackendDeadlineScope` with that same object; routing again through
/// an engine can select a different process-global backend. While any lease is
/// live, this registry exclusively owns the backend deadline slot: production
/// callers must not invoke `set_crown_backward_deadline` directly or through a
/// wrapper with independent deadline state. The trait has no deadline getter,
/// so foreign writes cannot be detected or composed. All ny-propagate
/// production writes are therefore centralized below; direct ny-gpu writes are
/// confined to its deadline backend tests.
struct GpuCrownDeadlineLease<'a> {
    gpu: Option<&'a dyn GpuCrownBackward>,
    backend_identity: usize,
    token: Option<Arc<()>>,
}

impl<'a> GpuCrownDeadlineLease<'a> {
    fn set(gpu: Option<&'a dyn GpuCrownBackward>, deadline: Option<std::time::Instant>) -> Self {
        let Some(gpu) =
            gpu.filter(|gpu| deadline.is_some() && gpu.honors_crown_backward_deadline())
        else {
            return Self {
                gpu: None,
                backend_identity: 0,
                token: None,
            };
        };
        let deadline = deadline.expect("deadline-bearing GPU lease checked above");
        let backend_identity = gpu_crown_backend_identity(gpu);
        let token = Arc::new(());
        let mut leases = gpu_crown_deadline_leases()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let previous_effective = leases
            .by_backend
            .get(&backend_identity)
            .and_then(|active| active.iter().map(|(_, deadline)| *deadline).min());
        let effective = previous_effective.map_or(deadline, |active| active.min(deadline));

        // Publish to the backend before committing the token to the registry.
        // A buggy/custom backend setter may panic; catch it while the registry
        // lock is still healthy, best-effort restore the previously owned slot,
        // then propagate the original panic with no ghost lease installed.
        let set_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gpu.set_crown_backward_deadline(Some(effective));
        }));
        if let Err(payload) = set_result {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                gpu.set_crown_backward_deadline(previous_effective);
            }));
            drop(leases);
            std::panic::resume_unwind(payload);
        }

        leases
            .by_backend
            .entry(backend_identity)
            .or_default()
            .push((Arc::clone(&token), deadline));
        drop(leases);
        Self {
            gpu: Some(gpu),
            backend_identity,
            token: Some(token),
        }
    }
}

impl Drop for GpuCrownDeadlineLease<'_> {
    fn drop(&mut self) {
        let (Some(gpu), Some(token)) = (self.gpu, self.token.take()) else {
            return;
        };
        let mut leases = gpu_crown_deadline_leases()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let (effective, remove_backend) =
            if let Some(active) = leases.by_backend.get_mut(&self.backend_identity) {
                if let Some(position) = active
                    .iter()
                    .position(|(entry_token, _)| Arc::ptr_eq(entry_token, &token))
                {
                    active.swap_remove(position);
                }
                (
                    active.iter().map(|(_, deadline)| *deadline).min(),
                    active.is_empty(),
                )
            } else {
                (None, false)
            };
        if remove_backend {
            leases.by_backend.remove(&self.backend_identity);
        }
        gpu.set_crown_backward_deadline(effective);
    }
}

/// Scoped cooperative deadline on the routed GPU CROWN backward
/// (#w4-refresh-deadline).
///
/// Sets the deadline on the SAME backend the CROWN dispatch sites will route to
/// ([`gpu_crown_backward_route`]). Overlapping scopes compose by retaining the
/// earliest active deadline; the last scope to leave clears the backend slot.
#[must_use = "the routed GPU deadline is owned only while this scope remains alive"]
pub(crate) struct GpuCrownDeadlineScope<'a> {
    _lease: GpuCrownDeadlineLease<'a>,
}

impl<'a> GpuCrownDeadlineScope<'a> {
    pub(crate) fn set(
        engine: Option<&'a dyn GemmEngine>,
        deadline: Option<std::time::Instant>,
    ) -> Self {
        let gpu = if deadline.is_some() {
            gpu_crown_backward_route_with_deadline(engine, deadline)
                .map(|(g, _use_sound)| g)
                .filter(|g| g.honors_crown_backward_deadline())
        } else {
            None
        };
        Self {
            _lease: GpuCrownDeadlineLease::set(gpu, deadline),
        }
    }
}

/// Whether a routed GPU CROWN call can participate in a deadline-scored
/// operation. CPU/no-route calls are admissible; a routed backend must
/// explicitly advertise cooperative cancellation.
pub(crate) fn gpu_crown_route_honors_deadline(
    engine: Option<&dyn GemmEngine>,
    deadline: Option<std::time::Instant>,
) -> bool {
    deadline.is_none()
        || gpu_crown_backward_route_with_deadline(engine, deadline)
            .is_none_or(|(gpu, _use_sound)| gpu.honors_crown_backward_deadline())
}

/// Whether one already-selected GPU backend can participate in a
/// deadline-scored operation.
pub(crate) fn gpu_crown_backend_honors_deadline(
    gpu: &dyn GpuCrownBackward,
    deadline: Option<std::time::Instant>,
) -> bool {
    deadline.is_none() || gpu.honors_crown_backward_deadline()
}

/// Scoped deadline for an already-selected GPU backend.
///
/// Wide callers can prefer a process-global backend that differs from the
/// propagation engine, so routing again through [`gpu_crown_backward_route`]
/// can set the deadline on the wrong device. This guard targets the exact
/// trait object that will receive the dispatch and restores the earliest
/// remaining lease on drop.
#[must_use = "the exact-backend GPU deadline is owned only while this scope remains alive"]
pub(crate) struct GpuCrownBackendDeadlineScope<'a> {
    _lease: GpuCrownDeadlineLease<'a>,
}

impl<'a> GpuCrownBackendDeadlineScope<'a> {
    pub(crate) fn set(gpu: &'a dyn GpuCrownBackward, deadline: Option<std::time::Instant>) -> Self {
        Self {
            _lease: GpuCrownDeadlineLease::set(Some(gpu), deadline),
        }
    }
}

/// Route a verdict-deciding graph-DAG IBP FORWARD to the GPU, honoring the gate.
///
/// The DAG counterpart of [`gpu_ibp_forward_route`] (`docs/SOUND_GPU_IBP_PLAN.md`
/// T1.0). It reuses the SAME process-global flag, so one verdict switch covers CROWN
/// backward, sequential IBP forward AND graph-DAG IBP forward.
///
/// Returns `Some((ext, use_sound))`:
/// - gate engaged AND the engine advertises a SOUND graph-DAG IBP forward
///   (`provides_sound_gpu_dag_ibp`) → `(ext, true)`: the caller uses
///   `prepare_sound_dag_model_plan` (a certified enclosure per node) and still falls
///   back to the CPU graph loop on `Err`/`None` (the 0-wrong moat holds);
/// - gate engaged but no sound DAG path available → `None`: CPU sound fallback;
/// - gate NOT engaged → `(ext, false)`: the existing fast (unsound) DAG speed path.
///   Verdict-safe because, by the gate's global contract, no verdict is decided when
///   the gate is off.
#[inline]
#[allow(dead_code)] // wired into `propagate_ibp_core_inner` (graph sound branch, T1.0)
pub(crate) fn gpu_dag_ibp_forward_route(
    engine: Option<&dyn GemmEngine>,
) -> Option<(&dyn GpuDagIbpForwardExt, bool)> {
    if is_sound_gpu_crown_required() {
        // Gate engaged: only a SOUND graph-DAG IBP forward may decide a bound.
        return engine
            .and_then(|e| e.as_gpu_dag_ibp_forward_ext())
            .filter(|g| g.provides_sound_gpu_dag_ibp())
            .map(|g| (g, true));
    }
    // Gate not engaged: the existing fast (unsound) DAG speed path.
    engine
        .and_then(|e| e.as_gpu_dag_ibp_forward_ext())
        .map(|g| (g, false))
}

/// Test-only synchronization for the process-global gate.
///
/// ONE mutex shared by EVERY test — across modules — that mutates *or depends
/// on* the gate value. Per-module locks do not exclude each other: a module
/// flipping the gate under its own lock races a test elsewhere that merely
/// requires a KNOWN gate state to observe the GPU fast-path.
///
/// Acquiring the lock resets the gate to OFF (the fast, unsound path) — NOT the
/// production default, which is now ON/sound (see `SOUND_GPU_CROWN_REQUIRED`). This
/// is deliberate: a test that wants to exercise the FAST GPU CROWN gets it here
/// without an explicit set, while a test that needs the sound path engages the gate
/// itself. The guard restores OFF on drop.
#[cfg(test)]
pub(crate) mod test_lock {
    static GATE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    pub(crate) struct GateGuard(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);
    impl Drop for GateGuard {
        fn drop(&mut self) {
            super::set_sound_gpu_crown_required(false);
        }
    }

    pub(crate) fn lock_gate() -> GateGuard {
        let g = GATE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        super::set_sound_gpu_crown_required(false);
        GateGuard(g)
    }
}

#[cfg(test)]
mod tests {
    use super::test_lock::lock_gate;
    use super::*;
    use ny_core::{
        gpu_bab_bound_static_payload_identity_v1, GpuBabBoundBackendOpenPreparation,
        GpuBabBoundBackendRegistration, GpuBabBoundBackendScheduleDisposition,
        GpuBabBoundBackendScheduleEvidence, GpuBabBoundBackendScheduleIdentity,
        GpuBabBoundF32Tensor, GpuBabBoundF32TensorRole, GpuBabBoundNumericalTcb,
        GpuBabBoundOwnedSlice, GpuBabBoundPhaseDecline, GpuBabBoundPhasePolicy,
        GpuBabBoundScheduleTcbInvocation, GpuBabBoundStaticScheduleRequest,
        GpuBabBoundTcbInvocation, GpuBabBoundU32Tensor, GpuBabBoundU32TensorRole,
        NaiveCpuGemmEngine,
    };

    #[test]
    fn gpu_interval_publication_rejects_shape_nonfinite_and_inversion() {
        assert!(gpu_interval_payload_is_publishable(
            &[-1.0, 0.0],
            &[1.0, 2.0],
            2
        ));
        for (lower, upper, rows) in [
            (vec![-1.0], vec![1.0, 2.0], 2),
            (vec![-1.0, 0.0], vec![1.0], 2),
            (vec![f32::NAN], vec![1.0], 1),
            (vec![-1.0], vec![f32::INFINITY], 1),
            (vec![2.0], vec![1.0], 1),
        ] {
            assert!(!gpu_interval_payload_is_publishable(&lower, &upper, rows));
            assert!(!gpu_crown_result_is_publishable(
                &GpuCrownResult {
                    lower_bounds: lower,
                    upper_bounds: upper,
                },
                rows,
            ));
        }
    }

    #[test]
    fn wide_request_switch_precedence_is_deterministic() {
        assert!(!wide_sound_gpu_crown_requested_from_values(None, None));
        assert!(wide_sound_gpu_crown_requested_from_values(None, Some("1")));
        assert!(wide_sound_gpu_crown_requested_from_values(
            Some(" true "),
            Some("0")
        ));
        assert!(!wide_sound_gpu_crown_requested_from_values(
            Some("off"),
            Some("1")
        ));
        assert!(wide_sound_gpu_crown_requested_from_values(
            Some("invalid"),
            Some("ON")
        ));
        assert!(!wide_sound_gpu_crown_requested_from_values(
            Some("invalid"),
            Some("invalid")
        ));
    }

    #[test]
    fn resident_cut_shadow_switch_is_exact_and_default_dark() {
        for rejected in [
            None,
            Some(""),
            Some("0"),
            Some("true"),
            Some(" 1 "),
            Some("2"),
        ] {
            assert!(!resident_cut_shadow_requested_from_value(rejected));
        }
        assert!(resident_cut_shadow_requested_from_value(Some("1")));
    }

    #[test]
    fn root_joint_deadline_lane_requires_both_exact_default_dark_gates() {
        for (root_joint, deadline_ascent) in [
            (None, None),
            (Some("1"), None),
            (None, Some("1")),
            (Some("0"), Some("1")),
            (Some("1"), Some("0")),
            (Some("true"), Some("1")),
            (Some("1"), Some("true")),
            (Some(" 1 "), Some("1")),
            (Some("1"), Some(" 1 ")),
        ] {
            assert!(!root_joint_deadline_lane_requested_from_values(
                root_joint,
                deadline_ascent
            ));
        }
        assert!(root_joint_deadline_lane_requested_from_values(
            Some("1"),
            Some("1")
        ));
    }

    #[test]
    fn test_lock_resets_gate_to_off_for_isolation() {
        // `lock_gate` resets to OFF (fast path) for test isolation — this is the
        // test harness's reset, NOT the production default (which is ON/sound; see
        // `SOUND_GPU_CROWN_REQUIRED` and `gpu_crown_gate_defaults_to_sound` in the
        // ny-cli cert_adapter tests).
        let _g = lock_gate();
        assert!(!is_sound_gpu_crown_required());
    }

    /// The PRODUCTION default of the process-global gate is ON (sound) — a verifier
    /// never decides a verdict on the unsound fast GPU CROWN by default
    /// (#gpu-crown-sound-default). Verified without acquiring the test lock (which
    /// would reset it to OFF) by reading the static's initial value in a child
    /// process-free way: we can't observe it after other tests ran, so assert the
    /// documented invariant via a fresh load guarded by the lock set to the default.
    #[test]
    #[allow(clippy::assertions_on_constants)] // deliberately asserts the compile-time default so a regression to `false` fails CI
    fn production_gate_default_is_sound() {
        // The static initialiser is `AtomicBool::new(true)`. Any code path that does
        // not touch the gate (e.g. `ny verify`) therefore sees ON/sound. We assert
        // the constant contract here so a regression to `new(false)` fails a test.
        assert!(
            DEFAULT_SOUND_GPU_CROWN,
            "the GPU CROWN gate must DEFAULT to sound; a verifier must not decide a \
             verdict on the unsound fast f32 backward unless explicitly opted out"
        );
    }

    #[test]
    fn gate_set_and_clear() {
        let _g = lock_gate();
        set_sound_gpu_crown_required(true);
        assert!(is_sound_gpu_crown_required());
        set_sound_gpu_crown_required(false);
        assert!(!is_sound_gpu_crown_required());
    }

    #[test]
    fn cpu_engine_never_yields_gpu_crown_regardless_of_gate() {
        let _g = lock_gate();
        // A plain CPU engine has no GPU CROWN backward; it is `None` either way.
        let engine = NaiveCpuGemmEngine;
        set_sound_gpu_crown_required(false);
        assert!(sound_gpu_crown_backward(Some(&engine)).is_none());
        set_sound_gpu_crown_required(true);
        assert!(sound_gpu_crown_backward(Some(&engine)).is_none());
    }

    #[test]
    fn gate_masks_a_gpu_capable_engine() {
        let _g = lock_gate();
        // A mock engine that DOES advertise a GPU CROWN backward. With the gate
        // off it is visible; with the gate on the helper hides it (the verdict
        // path must take the CPU sound route).
        let engine = MockGpuCrownEngine { sound: false };

        set_sound_gpu_crown_required(false);
        assert!(
            sound_gpu_crown_backward(Some(&engine)).is_some(),
            "with the gate disabled, a GPU-capable engine exposes its GPU CROWN backward"
        );

        set_sound_gpu_crown_required(true);
        assert!(
            sound_gpu_crown_backward(Some(&engine)).is_none(),
            "with the gate enabled, the GPU CROWN backward MUST be masked so the \
             verdict bound comes from the proven-sound CPU path"
        );
    }

    /// Minimal mock engine that advertises (and would serve) a GPU CROWN backward.
    /// Used to prove the gate masks an engine that is otherwise GPU-eligible.
    struct MockGpuCrownEngine {
        sound: bool,
    }

    impl GemmEngine for MockGpuCrownEngine {
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
        fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
            Some(self)
        }
    }

    impl GpuCrownBackward for MockGpuCrownEngine {
        fn crown_backward_gpu(
            &self,
            _layers: &[ny_core::GpuCrownLayer],
            _spec: &[f32],
            num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> ny_core::Result<GpuCrownResult> {
            // Deliberately returns a bogus (over-tight) bound; if the gate ever
            // let this run on the verdict path, the verdict would be unsound.
            Ok(GpuCrownResult {
                lower_bounds: vec![0.0; num_specs],
                upper_bounds: vec![0.0; num_specs],
            })
        }

        fn provides_sound_gpu_crown(&self) -> bool {
            self.sound
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum RetainedCapabilityPanic {
        SoundCrown,
        PhaseCapability,
        NumericalTcb,
    }

    #[derive(Clone, Copy)]
    enum RetainedScheduleMode {
        Decline,
        Exact,
        InvalidKernelIdentity,
        Panic,
    }

    fn retained_schedule_identity() -> GpuBabBoundBackendScheduleIdentity {
        GpuBabBoundBackendScheduleIdentity {
            schema_bundle_version: 1,
            provider_abi_sha256: [41; 32],
            receipt_abi_sha256: [42; 32],
            kernel_sha256: [43; 32],
            topology_schema_sha256: [44; 32],
            selfcheck_schema_sha256: [45; 32],
            transcript_schema_sha256: [46; 32],
        }
    }

    struct RetainedBabCapabilityMock {
        sound_crown: bool,
        phase_capability: bool,
        exposes_tcb: bool,
        panic_at: Option<RetainedCapabilityPanic>,
        query_calls: std::sync::atomic::AtomicUsize,
        schedule_calls: std::sync::atomic::AtomicUsize,
        schedule_mode: RetainedScheduleMode,
        registration: GpuBabBoundBackendRegistration,
    }

    impl RetainedBabCapabilityMock {
        fn new(sound_crown: bool, phase_capability: bool, exposes_tcb: bool, tag: u8) -> Self {
            Self {
                sound_crown,
                phase_capability,
                exposes_tcb,
                panic_at: None,
                query_calls: std::sync::atomic::AtomicUsize::new(0),
                schedule_calls: std::sync::atomic::AtomicUsize::new(0),
                schedule_mode: RetainedScheduleMode::Decline,
                registration: GpuBabBoundBackendRegistration::new_with_schedule_identity(
                    [tag; 32],
                    retained_schedule_identity(),
                )
                .expect("nonzero mock registration identity and schedule bundle"),
            }
        }

        fn with_schedule(schedule_mode: RetainedScheduleMode, tag: u8) -> Self {
            Self {
                schedule_mode,
                ..Self::new(true, true, true, tag)
            }
        }

        fn panicking(panic_at: RetainedCapabilityPanic, tag: u8) -> Self {
            Self {
                panic_at: Some(panic_at),
                ..Self::new(true, true, true, tag)
            }
        }

        fn note_query(&self, query: RetainedCapabilityPanic) {
            self.query_calls.fetch_add(1, Ordering::SeqCst);
            assert!(self.panic_at != Some(query), "scripted capability panic");
        }
    }

    impl GpuCrownBackward for RetainedBabCapabilityMock {
        fn crown_backward_gpu(
            &self,
            _layers: &[ny_core::GpuCrownLayer],
            _spec: &[f32],
            num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> ny_core::Result<GpuCrownResult> {
            Ok(GpuCrownResult {
                lower_bounds: vec![0.0; num_specs],
                upper_bounds: vec![0.0; num_specs],
            })
        }

        fn provides_sound_gpu_crown(&self) -> bool {
            self.note_query(RetainedCapabilityPanic::SoundCrown);
            self.sound_crown
        }

        fn provides_sound_gpu_bab_bound_phase(&self) -> bool {
            self.note_query(RetainedCapabilityPanic::PhaseCapability);
            self.phase_capability
        }

        fn gpu_bab_bound_numerical_tcb(&self) -> Option<&dyn GpuBabBoundNumericalTcb> {
            self.note_query(RetainedCapabilityPanic::NumericalTcb);
            self.exposes_tcb.then_some(self)
        }
    }

    impl GpuBabBoundNumericalTcb for RetainedBabCapabilityMock {
        fn registration(&self) -> &GpuBabBoundBackendRegistration {
            &self.registration
        }

        fn certify_static_schedule(
            &self,
            invocation: &GpuBabBoundScheduleTcbInvocation<'_, '_>,
        ) -> GpuBabBoundBackendScheduleDisposition {
            self.schedule_calls.fetch_add(1, Ordering::SeqCst);
            match self.schedule_mode {
                RetainedScheduleMode::Decline => {
                    GpuBabBoundBackendScheduleDisposition::CleanDecline(
                        GpuBabBoundPhaseDecline::Unsupported,
                    )
                }
                RetainedScheduleMode::Exact | RetainedScheduleMode::InvalidKernelIdentity => {
                    let request = invocation.request();
                    let mut schedule_identity = retained_schedule_identity();
                    if matches!(
                        self.schedule_mode,
                        RetainedScheduleMode::InvalidKernelIdentity
                    ) {
                        schedule_identity.kernel_sha256 = [99; 32];
                    }
                    GpuBabBoundBackendScheduleDisposition::Certified(
                        GpuBabBoundBackendScheduleEvidence {
                            backend_issuer_sha256: *self.registration.backend_issuer_sha256(),
                            registration_epoch: self.registration.registration_epoch(),
                            static_payload_identity_sha256: *request
                                .static_payload_identity_sha256(),
                            topology_schema_version: request.topology_schema_version(),
                            schedule_identity,
                            requested_max_device_bytes: request.requested_max_device_bytes(),
                            phase_policy: GpuBabBoundPhasePolicy {
                                max_device_bytes: request.requested_max_device_bytes(),
                                preferred_domains_per_wave: 4,
                                minimum_domains_per_wave: 1,
                                maximum_domains_per_wave: 16,
                                maximum_objectives: 16,
                                maximum_dispatches_per_wave: 64,
                                maximum_submits_per_wave: 8,
                            },
                            dispatches_per_subchunk: 3,
                        },
                    )
                }
                RetainedScheduleMode::Panic => panic!("scripted schedule panic"),
            }
        }

        fn phase_policy(
            &self,
            _invocation: &GpuBabBoundTcbInvocation<'_>,
        ) -> Option<GpuBabBoundPhasePolicy> {
            None
        }

        fn prepare_phase<'a>(
            &'a self,
            _invocation: &GpuBabBoundTcbInvocation<'_>,
        ) -> GpuBabBoundBackendOpenPreparation<'a> {
            GpuBabBoundBackendOpenPreparation::CleanDecline(GpuBabBoundPhaseDecline::Unsupported)
        }
    }

    struct ExplicitRetainedWgpuProvider<G> {
        gpu: G,
    }

    impl<G: GpuCrownBackward> RetainedWgpuBabPhaseProvider for ExplicitRetainedWgpuProvider<G> {
        fn gpu_crown_backward(&self) -> &dyn GpuCrownBackward {
            &self.gpu
        }
    }

    struct PanickingRetainedWgpuProvider {
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl RetainedWgpuBabPhaseProvider for PanickingRetainedWgpuProvider {
        fn gpu_crown_backward(&self) -> &dyn GpuCrownBackward {
            self.calls.fetch_add(1, Ordering::SeqCst);
            panic!("scripted marker accessor panic")
        }
    }

    struct PanickingDropPayload {
        drop_calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Drop for PanickingDropPayload {
        fn drop(&mut self) {
            self.drop_calls.fetch_add(1, Ordering::SeqCst);
            panic!("panic payload destructor must remain quarantined")
        }
    }

    fn present_retained_wgpu_provider(
        provider: SharedRetainedWgpuBabProvider,
    ) -> RetainedWgpuBabProviderMaterialization {
        RetainedWgpuBabProviderMaterialization::Present(MaterializedRetainedWgpuBabProvider {
            provider,
            fault: AtomicU8::new(0),
        })
    }

    struct TestStaticSchedulePayload {
        topology: Vec<u8>,
        f32_tensors: Vec<GpuBabBoundF32Tensor>,
        u32_tensors: Vec<GpuBabBoundU32Tensor>,
        identity: [u8; 32],
    }

    impl TestStaticSchedulePayload {
        fn request(
            &self,
            deadline: std::time::Instant,
            max_device_bytes: usize,
        ) -> GpuBabBoundStaticScheduleRequest<'_> {
            GpuBabBoundStaticScheduleRequest::new(
                1,
                &self.topology,
                &self.f32_tensors,
                &self.u32_tensors,
                self.identity,
                deadline,
                max_device_bytes,
            )
            .unwrap()
        }
    }

    fn test_static_schedule_payload() -> TestStaticSchedulePayload {
        let topology = vec![1_u8, 2, 3, 4];
        let f32_tensor = |role, shape, values| GpuBabBoundF32Tensor {
            role,
            shape,
            values: GpuBabBoundOwnedSlice::new(values),
        };
        let f32_tensors = vec![
            f32_tensor(
                GpuBabBoundF32TensorRole::Parameters,
                vec![2],
                vec![0.5, -0.5],
            ),
            f32_tensor(
                GpuBabBoundF32TensorRole::CertifiedErrors,
                vec![1],
                vec![0.0],
            ),
            f32_tensor(GpuBabBoundF32TensorRole::Relaxations, vec![0], vec![]),
            f32_tensor(
                GpuBabBoundF32TensorRole::InputLower,
                vec![2],
                vec![-1.0, -1.0],
            ),
            f32_tensor(
                GpuBabBoundF32TensorRole::InputUpper,
                vec![2],
                vec![1.0, 1.0],
            ),
            f32_tensor(
                GpuBabBoundF32TensorRole::RootLower,
                vec![2],
                vec![-2.0, -2.0],
            ),
            f32_tensor(GpuBabBoundF32TensorRole::RootUpper, vec![2], vec![2.0, 2.0]),
            f32_tensor(
                GpuBabBoundF32TensorRole::ObjectiveCoefficients,
                vec![1, 2],
                vec![1.0, -1.0],
            ),
        ];
        let u32_tensors = vec![
            GpuBabBoundU32Tensor {
                role: GpuBabBoundU32TensorRole::ObjectiveIndices,
                shape: vec![1],
                values: GpuBabBoundOwnedSlice::new(vec![0]),
            },
            GpuBabBoundU32Tensor {
                role: GpuBabBoundU32TensorRole::TopologyMetadata,
                shape: vec![0],
                values: GpuBabBoundOwnedSlice::new(Vec::new()),
            },
        ];
        let mut check = |_| Ok(());
        let identity = gpu_bab_bound_static_payload_identity_v1(
            1,
            &topology,
            &f32_tensors,
            &u32_tensors,
            &mut check,
        )
        .unwrap();
        TestStaticSchedulePayload {
            topology,
            f32_tensors,
            u32_tensors,
            identity,
        }
    }

    struct LosingRetainedBabCapabilityMock {
        sound_queries: std::sync::atomic::AtomicUsize,
        query_calls: std::sync::atomic::AtomicUsize,
        registration: GpuBabBoundBackendRegistration,
    }

    impl LosingRetainedBabCapabilityMock {
        fn new() -> Self {
            Self {
                sound_queries: std::sync::atomic::AtomicUsize::new(0),
                query_calls: std::sync::atomic::AtomicUsize::new(0),
                registration: GpuBabBoundBackendRegistration::new_with_schedule_identity(
                    [61; 32],
                    retained_schedule_identity(),
                )
                .unwrap(),
            }
        }
    }

    impl GpuCrownBackward for LosingRetainedBabCapabilityMock {
        fn crown_backward_gpu(
            &self,
            _layers: &[ny_core::GpuCrownLayer],
            _spec: &[f32],
            num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> ny_core::Result<GpuCrownResult> {
            Ok(GpuCrownResult {
                lower_bounds: vec![0.0; num_specs],
                upper_bounds: vec![0.0; num_specs],
            })
        }

        fn provides_sound_gpu_crown(&self) -> bool {
            self.query_calls.fetch_add(1, Ordering::SeqCst);
            self.sound_queries.fetch_add(1, Ordering::SeqCst) == 0
        }

        fn provides_sound_gpu_bab_bound_phase(&self) -> bool {
            self.query_calls.fetch_add(1, Ordering::SeqCst);
            true
        }

        fn gpu_bab_bound_numerical_tcb(&self) -> Option<&dyn GpuBabBoundNumericalTcb> {
            self.query_calls.fetch_add(1, Ordering::SeqCst);
            Some(self)
        }
    }

    impl GpuBabBoundNumericalTcb for LosingRetainedBabCapabilityMock {
        fn registration(&self) -> &GpuBabBoundBackendRegistration {
            &self.registration
        }

        fn phase_policy(
            &self,
            _invocation: &GpuBabBoundTcbInvocation<'_>,
        ) -> Option<GpuBabBoundPhasePolicy> {
            None
        }

        fn prepare_phase<'a>(
            &'a self,
            _invocation: &GpuBabBoundTcbInvocation<'_>,
        ) -> GpuBabBoundBackendOpenPreparation<'a> {
            GpuBabBoundBackendOpenPreparation::CleanDecline(GpuBabBoundPhaseDecline::Unsupported)
        }
    }

    #[test]
    fn retained_wgpu_cold_finite_lookup_never_invokes_factory() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let factory: OnceLock<RetainedWgpuBabFactory> = OnceLock::new();
        let provider: OnceLock<RetainedWgpuBabProviderMaterialization> = OnceLock::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let factory_calls = Arc::clone(&calls);
        assert!(factory
            .set(Box::new(move || {
                factory_calls.fetch_add(1, Ordering::SeqCst);
                None
            }))
            .is_ok());

        let observation = preinitialized_retained_wgpu_bab_provider_from_slot(&provider);
        assert!(matches!(
            observation,
            RetainedWgpuBabProviderObservation::ColdOrUnconfigured
        ));
        assert!(observation.allows_preopen_legacy_fallback());
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "a finite lookup must not invoke even an installed factory"
        );
        assert!(
            provider.get().is_none(),
            "cold lookup must not seal the slot"
        );
    }

    #[test]
    fn retained_wgpu_selector_rejects_every_incomplete_generic_capability() {
        for (sound_crown, phase_capability, exposes_tcb, tag) in [
            (false, true, true, 11),
            (true, false, true, 12),
            (true, true, false, 13),
        ] {
            let candidate: SharedRetainedWgpuBabProvider = Arc::new(ExplicitRetainedWgpuProvider {
                gpu: RetainedBabCapabilityMock::new(
                    sound_crown,
                    phase_capability,
                    exposes_tcb,
                    tag,
                ),
            });
            let slot = OnceLock::new();
            assert!(slot.set(present_retained_wgpu_provider(candidate)).is_ok());
            let observation = preinitialized_retained_wgpu_bab_provider_from_slot(&slot);
            assert!(
                matches!(
                    observation,
                    RetainedWgpuBabProviderObservation::InvalidCapabilities
                ),
                "candidate ({sound_crown}, {phase_capability}, {exposes_tcb}) entered without all three exact capabilities"
            );
            assert!(!observation.allows_preopen_legacy_fallback());
            assert!(matches!(
                preinitialized_retained_wgpu_bab_provider_from_slot(&slot),
                RetainedWgpuBabProviderObservation::InvalidCapabilities
            ));
        }
    }

    #[test]
    fn retained_wgpu_explicit_exact_provider_prewarm_enables_finite_lookup() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let factory: OnceLock<RetainedWgpuBabFactory> = OnceLock::new();
        let provider_slot: OnceLock<RetainedWgpuBabProviderMaterialization> = OnceLock::new();
        let concrete = Arc::new(ExplicitRetainedWgpuProvider {
            gpu: RetainedBabCapabilityMock::new(true, true, true, 14),
        });
        let expected =
            std::ptr::from_ref::<dyn GpuCrownBackward>(concrete.gpu_crown_backward()).cast::<()>();
        let provider: SharedRetainedWgpuBabProvider = concrete;
        let factory_provider = Arc::clone(&provider);
        let calls = Arc::new(AtomicUsize::new(0));
        let factory_calls = Arc::clone(&calls);
        assert!(factory
            .set(Box::new(move || {
                factory_calls.fetch_add(1, Ordering::SeqCst);
                Some(Arc::clone(&factory_provider))
            }))
            .is_ok());

        assert!(matches!(
            preinitialized_retained_wgpu_bab_provider_from_slot(&provider_slot),
            RetainedWgpuBabProviderObservation::ColdOrUnconfigured
        ));
        let prewarmed =
            match materialize_registered_retained_wgpu_bab_provider(&factory, &provider_slot) {
                RetainedWgpuBabProviderObservation::Ready(gpu) => gpu,
                _ => panic!("an explicitly marked provider with all live capabilities must enter"),
            };
        assert_eq!(
            std::ptr::from_ref::<dyn GpuCrownBackward>(prewarmed).cast::<()>(),
            expected
        );

        let finite = match preinitialized_retained_wgpu_bab_provider_from_slot(&provider_slot) {
            RetainedWgpuBabProviderObservation::Ready(gpu) => gpu,
            _ => panic!("prewarm must publish the exact provider for finite lookup"),
        };
        assert_eq!(
            std::ptr::from_ref::<dyn GpuCrownBackward>(finite).cast::<()>(),
            expected
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "finite lookup must reuse the materialized provider without rerunning its factory"
        );
    }

    #[test]
    fn retained_wgpu_factory_unavailable_is_typed_and_allows_only_preopen_fallback() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let factory: OnceLock<RetainedWgpuBabFactory> = OnceLock::new();
        let provider_slot: OnceLock<RetainedWgpuBabProviderMaterialization> = OnceLock::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let factory_calls = Arc::clone(&calls);
        assert!(factory
            .set(Box::new(move || {
                factory_calls.fetch_add(1, Ordering::SeqCst);
                None
            }))
            .is_ok());

        let prewarmed = materialize_registered_retained_wgpu_bab_provider(&factory, &provider_slot);
        assert!(matches!(
            prewarmed,
            RetainedWgpuBabProviderObservation::Unavailable
        ));
        assert!(prewarmed.allows_preopen_legacy_fallback());
        assert!(matches!(
            preinitialized_retained_wgpu_bab_provider_from_slot(&provider_slot),
            RetainedWgpuBabProviderObservation::Unavailable
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn retained_wgpu_factory_panic_is_persisted_as_no_fallback() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let factory: OnceLock<RetainedWgpuBabFactory> = OnceLock::new();
        let provider_slot: OnceLock<RetainedWgpuBabProviderMaterialization> = OnceLock::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let factory_calls = Arc::clone(&calls);
        let payload_drop_calls = Arc::new(AtomicUsize::new(0));
        let factory_payload_drop_calls = Arc::clone(&payload_drop_calls);
        assert!(factory
            .set(Box::new(move || {
                factory_calls.fetch_add(1, Ordering::SeqCst);
                std::panic::panic_any(PanickingDropPayload {
                    drop_calls: Arc::clone(&factory_payload_drop_calls),
                })
            }))
            .is_ok());

        let prewarmed = materialize_registered_retained_wgpu_bab_provider(&factory, &provider_slot);
        assert!(matches!(
            prewarmed,
            RetainedWgpuBabProviderObservation::Panicked
        ));
        assert!(!prewarmed.allows_preopen_legacy_fallback());
        let finite = preinitialized_retained_wgpu_bab_provider_from_slot(&provider_slot);
        assert!(matches!(
            finite,
            RetainedWgpuBabProviderObservation::Panicked
        ));
        assert!(!finite.allows_preopen_legacy_fallback());
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "finite observation must use the persisted panic instead of rerunning the factory"
        );
        assert_eq!(
            payload_drop_calls.load(Ordering::SeqCst),
            0,
            "caught panic payload destructors must be quarantined"
        );
    }

    #[test]
    fn retained_wgpu_marker_and_each_capability_panic_are_absorbing_no_fallback() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let marker_calls = Arc::new(AtomicUsize::new(0));
        let marker: SharedRetainedWgpuBabProvider = Arc::new(PanickingRetainedWgpuProvider {
            calls: Arc::clone(&marker_calls),
        });
        let marker_slot = OnceLock::new();
        assert!(marker_slot
            .set(present_retained_wgpu_provider(marker))
            .is_ok());
        for _ in 0..2 {
            let observation = preinitialized_retained_wgpu_bab_provider_from_slot(&marker_slot);
            assert!(matches!(
                observation,
                RetainedWgpuBabProviderObservation::Panicked
            ));
            assert!(!observation.allows_preopen_legacy_fallback());
        }
        assert_eq!(marker_calls.load(Ordering::SeqCst), 1);

        for (panic_at, expected_calls, tag) in [
            (RetainedCapabilityPanic::SoundCrown, 1, 21),
            (RetainedCapabilityPanic::PhaseCapability, 2, 22),
            (RetainedCapabilityPanic::NumericalTcb, 3, 23),
        ] {
            let concrete = Arc::new(ExplicitRetainedWgpuProvider {
                gpu: RetainedBabCapabilityMock::panicking(panic_at, tag),
            });
            let candidate: SharedRetainedWgpuBabProvider = concrete.clone();
            let slot = OnceLock::new();
            assert!(slot.set(present_retained_wgpu_provider(candidate)).is_ok());
            for _ in 0..2 {
                let observation = preinitialized_retained_wgpu_bab_provider_from_slot(&slot);
                assert!(matches!(
                    observation,
                    RetainedWgpuBabProviderObservation::Panicked
                ));
                assert!(!observation.allows_preopen_legacy_fallback());
            }
            assert_eq!(
                concrete.gpu.query_calls.load(Ordering::SeqCst),
                expected_calls,
                "the first panic must be sealed without re-entering the provider"
            );
        }
    }

    #[test]
    fn retained_wgpu_concurrent_first_fault_is_nonblocking_and_absorbing() {
        use std::sync::Barrier;

        fn fault_code(observation: RetainedWgpuBabProviderObservation<'_>) -> u8 {
            match observation {
                RetainedWgpuBabProviderObservation::InvalidCapabilities => {
                    RetainedWgpuBabProviderFault::InvalidCapabilities as u8
                }
                RetainedWgpuBabProviderObservation::Panicked => {
                    RetainedWgpuBabProviderFault::Panicked as u8
                }
                _ => 0,
            }
        }

        let concrete = Arc::new(ExplicitRetainedWgpuProvider {
            gpu: RetainedBabCapabilityMock::new(true, true, true, 24),
        });
        let candidate: SharedRetainedWgpuBabProvider = concrete.clone();
        let provider = Arc::new(MaterializedRetainedWgpuBabProvider {
            provider: candidate,
            fault: AtomicU8::new(0),
        });
        let start = Arc::new(Barrier::new(3));

        std::thread::scope(|scope| {
            let invalid_provider = Arc::clone(&provider);
            let invalid_start = Arc::clone(&start);
            let invalid = scope.spawn(move || {
                invalid_start.wait();
                fault_code(record_retained_wgpu_bab_provider_fault(
                    &invalid_provider,
                    RetainedWgpuBabProviderFault::InvalidCapabilities,
                ))
            });

            let panicked_provider = Arc::clone(&provider);
            let panicked_start = Arc::clone(&start);
            let panicked = scope.spawn(move || {
                panicked_start.wait();
                fault_code(record_retained_wgpu_bab_provider_fault(
                    &panicked_provider,
                    RetainedWgpuBabProviderFault::Panicked,
                ))
            });

            start.wait();
            let invalid_observed = invalid.join().expect("invalid-fault recorder must return");
            let panicked_observed = panicked.join().expect("panic-fault recorder must return");
            assert_ne!(invalid_observed, 0);
            assert_eq!(invalid_observed, panicked_observed);

            let winner = provider.fault.load(Ordering::Acquire);
            assert_eq!(winner, invalid_observed);
            let losing_fault = if winner == RetainedWgpuBabProviderFault::InvalidCapabilities as u8
            {
                RetainedWgpuBabProviderFault::Panicked
            } else {
                RetainedWgpuBabProviderFault::InvalidCapabilities
            };
            assert_eq!(
                fault_code(record_retained_wgpu_bab_provider_fault(
                    &provider,
                    losing_fault,
                )),
                winner,
                "the first published fault must remain absorbing"
            );
            assert_eq!(provider.fault.load(Ordering::Acquire), winner);
        });

        assert_eq!(
            concrete.gpu.query_calls.load(Ordering::SeqCst),
            0,
            "fault publication must not enter the provider or its capabilities"
        );
    }

    fn materialized_schedule_fault(slot: &OnceLock<RetainedWgpuBabProviderMaterialization>) -> u8 {
        match slot.get().unwrap() {
            RetainedWgpuBabProviderMaterialization::Present(provider) => {
                provider.fault.load(Ordering::Acquire)
            }
            _ => panic!("test schedule slot must contain a provider"),
        }
    }

    #[test]
    fn retained_wgpu_schedule_exact_and_clean_decline_have_only_their_typed_fallback() {
        let payload = test_static_schedule_payload();
        let request = payload.request(
            std::time::Instant::now() + std::time::Duration::from_secs(10),
            4_096,
        );

        let exact = Arc::new(ExplicitRetainedWgpuProvider {
            gpu: RetainedBabCapabilityMock::with_schedule(RetainedScheduleMode::Exact, 31),
        });
        let exact_slot = OnceLock::new();
        let exact_provider: SharedRetainedWgpuBabProvider = exact.clone();
        assert!(exact_slot
            .set(present_retained_wgpu_provider(exact_provider))
            .is_ok());
        let exact_observation =
            certify_preinitialized_retained_wgpu_bab_schedule_from_slot(&exact_slot, &request);
        assert!(!exact_observation.allows_preopen_legacy_fallback());
        let certificate = match exact_observation {
            RetainedWgpuBabScheduleObservation::Certified(certificate) => certificate,
            _ => panic!("exact retained-WGPU schedule must certify"),
        };
        assert_ne!(certificate.certificate_identity_sha256(), &[0; 32]);
        assert_eq!(exact.gpu.schedule_calls.load(Ordering::SeqCst), 1);
        assert_eq!(materialized_schedule_fault(&exact_slot), 0);

        let declining = Arc::new(ExplicitRetainedWgpuProvider {
            gpu: RetainedBabCapabilityMock::new(true, true, true, 32),
        });
        let declining_slot = OnceLock::new();
        let declining_provider: SharedRetainedWgpuBabProvider = declining;
        assert!(declining_slot
            .set(present_retained_wgpu_provider(declining_provider))
            .is_ok());
        let declined =
            certify_preinitialized_retained_wgpu_bab_schedule_from_slot(&declining_slot, &request);
        assert!(declined.allows_preopen_legacy_fallback());
        assert!(matches!(
            declined,
            RetainedWgpuBabScheduleObservation::CleanDecline(GpuBabBoundPhaseDecline::Unsupported)
        ));
        assert_eq!(materialized_schedule_fault(&declining_slot), 0);
    }

    #[test]
    fn retained_wgpu_expired_request_is_nonsticky_and_never_queries_provider() {
        let payload = test_static_schedule_payload();
        let provider = Arc::new(ExplicitRetainedWgpuProvider {
            gpu: RetainedBabCapabilityMock::new(true, true, true, 33),
        });
        let slot = OnceLock::new();
        let marked: SharedRetainedWgpuBabProvider = provider.clone();
        assert!(slot.set(present_retained_wgpu_provider(marked)).is_ok());

        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(20);
        let expired_request = payload.request(deadline, 4_096);
        while std::time::Instant::now() < deadline {
            std::hint::spin_loop();
        }
        let expired =
            certify_preinitialized_retained_wgpu_bab_schedule_from_slot(&slot, &expired_request);
        assert!(!expired.allows_preopen_legacy_fallback());
        assert!(matches!(
            expired,
            RetainedWgpuBabScheduleObservation::Terminal(
                GpuBabBoundProviderFailureKind::DeadlineExpired
            )
        ));
        assert_eq!(provider.gpu.query_calls.load(Ordering::SeqCst), 0);
        assert_eq!(provider.gpu.schedule_calls.load(Ordering::SeqCst), 0);
        assert_eq!(materialized_schedule_fault(&slot), 0);

        let fresh_request = payload.request(
            std::time::Instant::now() + std::time::Duration::from_secs(10),
            4_096,
        );
        let fresh =
            certify_preinitialized_retained_wgpu_bab_schedule_from_slot(&slot, &fresh_request);
        assert!(fresh.allows_preopen_legacy_fallback());
        assert_eq!(materialized_schedule_fault(&slot), 0);
    }

    #[test]
    fn retained_wgpu_sticky_fault_wins_before_deadline_or_capacity_fallback() {
        let payload = test_static_schedule_payload();
        let provider = Arc::new(ExplicitRetainedWgpuProvider {
            gpu: RetainedBabCapabilityMock::new(true, true, true, 34),
        });
        let slot = OnceLock::new();
        let marked: SharedRetainedWgpuBabProvider = provider.clone();
        assert!(slot.set(present_retained_wgpu_provider(marked)).is_ok());
        let materialized = match slot.get().unwrap() {
            RetainedWgpuBabProviderMaterialization::Present(materialized) => materialized,
            _ => unreachable!(),
        };
        let _ = record_retained_wgpu_bab_provider_fault(
            materialized,
            RetainedWgpuBabProviderFault::InvalidCapabilities,
        );

        let low_cap = payload.request(
            std::time::Instant::now() + std::time::Duration::from_secs(10),
            1,
        );
        let observation =
            certify_preinitialized_retained_wgpu_bab_schedule_from_slot(&slot, &low_cap);
        assert!(!observation.allows_preopen_legacy_fallback());
        assert!(matches!(
            observation,
            RetainedWgpuBabScheduleObservation::InvalidCapabilities
        ));
        assert_eq!(provider.gpu.query_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn retained_wgpu_invalid_schedule_and_panic_are_sticky_without_reentry() {
        for (mode, expected_panicked, tag) in [
            (RetainedScheduleMode::InvalidKernelIdentity, false, 35),
            (RetainedScheduleMode::Panic, true, 36),
        ] {
            let payload = test_static_schedule_payload();
            let request = payload.request(
                std::time::Instant::now() + std::time::Duration::from_secs(10),
                4_096,
            );
            let provider = Arc::new(ExplicitRetainedWgpuProvider {
                gpu: RetainedBabCapabilityMock::with_schedule(mode, tag),
            });
            let slot = OnceLock::new();
            let marked: SharedRetainedWgpuBabProvider = provider.clone();
            assert!(slot.set(present_retained_wgpu_provider(marked)).is_ok());

            let first =
                certify_preinitialized_retained_wgpu_bab_schedule_from_slot(&slot, &request);
            assert!(!first.allows_preopen_legacy_fallback());
            assert_eq!(
                matches!(&first, RetainedWgpuBabScheduleObservation::Panicked),
                expected_panicked
            );
            if !expected_panicked {
                assert!(matches!(
                    &first,
                    RetainedWgpuBabScheduleObservation::InvalidCapabilities
                ));
            }
            let queries = provider.gpu.query_calls.load(Ordering::SeqCst);
            let schedules = provider.gpu.schedule_calls.load(Ordering::SeqCst);
            let second =
                certify_preinitialized_retained_wgpu_bab_schedule_from_slot(&slot, &request);
            assert_eq!(
                matches!(second, RetainedWgpuBabScheduleObservation::Panicked),
                expected_panicked
            );
            assert_eq!(provider.gpu.query_calls.load(Ordering::SeqCst), queries);
            assert_eq!(
                provider.gpu.schedule_calls.load(Ordering::SeqCst),
                schedules
            );
        }
    }

    #[test]
    fn retained_wgpu_ready_capability_loss_cannot_become_clean_fallback() {
        let payload = test_static_schedule_payload();
        let request = payload.request(
            std::time::Instant::now() + std::time::Duration::from_secs(10),
            4_096,
        );
        let provider = Arc::new(ExplicitRetainedWgpuProvider {
            gpu: LosingRetainedBabCapabilityMock::new(),
        });
        let slot = OnceLock::new();
        let marked: SharedRetainedWgpuBabProvider = provider.clone();
        assert!(slot.set(present_retained_wgpu_provider(marked)).is_ok());

        let first = certify_preinitialized_retained_wgpu_bab_schedule_from_slot(&slot, &request);
        assert!(!first.allows_preopen_legacy_fallback());
        assert!(matches!(
            first,
            RetainedWgpuBabScheduleObservation::InvalidCapabilities
        ));
        let queries = provider.gpu.query_calls.load(Ordering::SeqCst);
        assert!(
            queries > 3,
            "core/postcheck must observe the capability loss"
        );
        let second = certify_preinitialized_retained_wgpu_bab_schedule_from_slot(&slot, &request);
        assert!(matches!(
            second,
            RetainedWgpuBabScheduleObservation::InvalidCapabilities
        ));
        assert_eq!(provider.gpu.query_calls.load(Ordering::SeqCst), queries);
    }

    #[test]
    fn retained_wgpu_deadline_and_occupied_registration_kinds_are_nonsticky() {
        assert!(
            retained_wgpu_bab_schedule_fault(GpuBabBoundProviderFailureKind::DeadlineExpired)
                .is_none()
        );
        assert!(retained_wgpu_bab_schedule_fault(
            GpuBabBoundProviderFailureKind::RegistrationUnavailable
        )
        .is_none());
        assert!(retained_wgpu_bab_schedule_fault(
            GpuBabBoundProviderFailureKind::InvalidScheduleEvidence
        )
        .is_some());
    }

    struct DeadlineMockGpu {
        honors: bool,
        observed: Mutex<Vec<Option<std::time::Instant>>>,
    }

    impl GpuCrownBackward for DeadlineMockGpu {
        fn crown_backward_gpu(
            &self,
            _layers: &[ny_core::GpuCrownLayer],
            _spec: &[f32],
            num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> ny_core::Result<GpuCrownResult> {
            Ok(GpuCrownResult {
                lower_bounds: vec![0.0; num_specs],
                upper_bounds: vec![0.0; num_specs],
            })
        }

        fn honors_crown_backward_deadline(&self) -> bool {
            self.honors
        }

        fn set_crown_backward_deadline(&self, deadline: Option<std::time::Instant>) {
            self.observed.lock().unwrap().push(deadline);
        }
    }

    struct ExplicitResidentCutDeadlineMock;

    impl GpuCrownBackward for ExplicitResidentCutDeadlineMock {
        fn crown_backward_gpu(
            &self,
            _layers: &[ny_core::GpuCrownLayer],
            _spec: &[f32],
            num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> ny_core::Result<GpuCrownResult> {
            Ok(GpuCrownResult {
                lower_bounds: vec![0.0; num_specs],
                upper_bounds: vec![0.0; num_specs],
            })
        }

        fn provides_resident_cut_shadow(&self) -> bool {
            true
        }

        fn provides_sound_gpu_crown(&self) -> bool {
            true
        }
    }

    #[test]
    fn resident_cut_capability_is_narrower_than_general_deadline_capability() {
        let gpu = ExplicitResidentCutDeadlineMock;
        assert!(gpu.provides_resident_cut_shadow());
        assert!(!gpu.honors_crown_backward_deadline());
    }

    #[test]
    fn resident_cut_backend_resolver_never_touches_factory_while_gate_is_off() {
        let gpu = ExplicitResidentCutDeadlineMock;
        for raw in [None, Some("0"), Some("true"), Some(" 1 ")] {
            let calls = std::cell::Cell::new(0usize);
            let selected = resident_cut_shadow_backend_from_gate(raw, || {
                calls.set(calls.get() + 1);
                Some(&gpu)
            });
            assert!(selected.is_none());
            assert_eq!(calls.get(), 0, "raw gate {raw:?} resolved the factory");
        }

        let calls = std::cell::Cell::new(0usize);
        let selected = resident_cut_shadow_backend_from_gate(Some("1"), || {
            calls.set(calls.get() + 1);
            Some(&gpu)
        });
        assert!(selected.is_some());
        assert_eq!(calls.get(), 1);
    }

    struct RootJointCudaCapabilityMock {
        sound: bool,
        bounded_joint: bool,
        sound_fold_max_rows: usize,
        honors_backend_deadline: bool,
    }

    impl GemmEngine for RootJointCudaCapabilityMock {
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

        fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
            Some(self)
        }
    }

    impl GpuCrownBackward for RootJointCudaCapabilityMock {
        fn crown_backward_gpu(
            &self,
            _layers: &[ny_core::GpuCrownLayer],
            _spec: &[f32],
            num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> ny_core::Result<GpuCrownResult> {
            Ok(GpuCrownResult {
                lower_bounds: vec![0.0; num_specs],
                upper_bounds: vec![0.0; num_specs],
            })
        }

        fn provides_sound_gpu_crown(&self) -> bool {
            self.sound
        }

        fn provides_deadline_bounded_joint_alpha_gradient_resident(&self) -> bool {
            self.bounded_joint
        }

        fn deadline_bounded_resnet_sound_max_rows(&self) -> usize {
            self.sound_fold_max_rows
        }

        fn honors_crown_backward_deadline(&self) -> bool {
            self.honors_backend_deadline
        }
    }

    #[test]
    fn root_joint_factory_seam_reaches_exact_eligible_cuda_capability() {
        let cuda = RootJointCudaCapabilityMock {
            sound: true,
            bounded_joint: true,
            sound_fold_max_rows: ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
            honors_backend_deadline: false,
        };
        let calls = std::cell::Cell::new(0usize);
        let reached = with_root_joint_deadline_gpu(&cuda, |selected| {
            calls.set(calls.get() + 1);
            let selected_data =
                std::ptr::from_ref::<dyn GpuCrownBackward>(selected.backend()).cast::<()>();
            let expected_data = std::ptr::from_ref(&cuda).cast::<()>();
            assert_eq!(selected_data, expected_data);
            assert_eq!(
                selected.sound_fold_max_rows(),
                ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS
            );
            17usize
        });
        assert_eq!(reached, Some(17));
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn root_joint_factory_seam_quarantines_wgpu_like_unsound_backend() {
        let wgpu_like = RootJointCudaCapabilityMock {
            sound: false,
            bounded_joint: true,
            sound_fold_max_rows: ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
            honors_backend_deadline: true,
        };
        let called = std::cell::Cell::new(false);
        assert!(with_root_joint_deadline_gpu(&wgpu_like, |_| called.set(true)).is_none());
        assert!(
            !called.get(),
            "an unsound WGPU-like backend must never reach the root-joint call"
        );
    }

    #[test]
    fn root_joint_factory_seam_refuses_missing_or_out_of_contract_capabilities() {
        let cpu = NaiveCpuGemmEngine;
        assert!(with_root_joint_deadline_gpu(&cpu, |_| {
            panic!("an engine without GpuCrownBackward reached the root-joint seam")
        })
        .is_none());

        for candidate in [
            RootJointCudaCapabilityMock {
                sound: true,
                bounded_joint: false,
                sound_fold_max_rows: ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
                honors_backend_deadline: true,
            },
            RootJointCudaCapabilityMock {
                sound: true,
                bounded_joint: true,
                sound_fold_max_rows: 0,
                honors_backend_deadline: true,
            },
            RootJointCudaCapabilityMock {
                sound: true,
                bounded_joint: true,
                sound_fold_max_rows: ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS + 1,
                honors_backend_deadline: true,
            },
        ] {
            let called = std::cell::Cell::new(false);
            assert!(with_root_joint_deadline_gpu(&candidate, |_| called.set(true)).is_none());
            assert!(!called.get());
        }
    }

    /// #batched-bab arming pin (2026-08-11): the PRE-FIX WgpuDevice shape —
    /// sound, deadline-honoring, joint-capable, but WITHOUT the bounded-rows
    /// override (ny-core default ⇒ `deadline_bounded_resnet_sound_max_rows()==0`,
    /// gemm.rs:2087) — is refused by the K≤8 admission seam even though every
    /// other capability is green. The SAME backend with the honest full-contract
    /// override (K=8, now implemented on WgpuDevice) is admitted at exactly that
    /// capacity. This pins the decline point so a regression that drops the
    /// override silently re-darkens every bounded-rows lane.
    #[test]
    fn wgpu_shaped_backend_is_refused_without_the_bounded_rows_override_and_admitted_with_it() {
        let pre_fix_wgpu_shape = RootJointCudaCapabilityMock {
            sound: true,
            bounded_joint: true,
            sound_fold_max_rows: 0, // the ny-core default WgpuDevice used to inherit
            honors_backend_deadline: true,
        };
        let called = std::cell::Cell::new(false);
        assert!(
            with_root_joint_deadline_gpu(&pre_fix_wgpu_shape, |_| called.set(true)).is_none(),
            "a sound, deadline-honoring backend without the bounded-rows override must \
             still be refused (the pre-fix WgpuDevice decline point)"
        );
        assert!(!called.get());

        let armed_wgpu_shape = RootJointCudaCapabilityMock {
            sound: true,
            bounded_joint: true,
            sound_fold_max_rows: ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
            honors_backend_deadline: true,
        };
        let admitted_rows = with_root_joint_deadline_gpu(&armed_wgpu_shape, |selected| {
            selected.sound_fold_max_rows()
        });
        assert_eq!(
            admitted_rows,
            Some(ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS),
            "the honest K=8 override must be admitted at exactly the documented capacity"
        );
    }

    struct PanickingDeadlineMockGpu {
        panic_on_next_set: AtomicBool,
        observed: Mutex<Vec<Option<std::time::Instant>>>,
    }

    impl GpuCrownBackward for PanickingDeadlineMockGpu {
        fn crown_backward_gpu(
            &self,
            _layers: &[ny_core::GpuCrownLayer],
            _spec: &[f32],
            num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> ny_core::Result<GpuCrownResult> {
            Ok(GpuCrownResult {
                lower_bounds: vec![0.0; num_specs],
                upper_bounds: vec![0.0; num_specs],
            })
        }

        fn honors_crown_backward_deadline(&self) -> bool {
            true
        }

        fn set_crown_backward_deadline(&self, deadline: Option<std::time::Instant>) {
            self.observed.lock().unwrap().push(deadline);
            assert!(
                !self.panic_on_next_set.swap(false, Ordering::SeqCst),
                "injected backend deadline setter panic"
            );
        }
    }

    #[test]
    fn exact_backend_deadline_scope_sets_and_clears_only_capable_backends() {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        let capable = DeadlineMockGpu {
            honors: true,
            observed: Mutex::new(Vec::new()),
        };
        assert!(gpu_crown_backend_honors_deadline(&capable, Some(deadline)));
        {
            let _scope = GpuCrownBackendDeadlineScope::set(&capable, Some(deadline));
            assert_eq!(*capable.observed.lock().unwrap(), vec![Some(deadline)]);
        }
        assert_eq!(
            *capable.observed.lock().unwrap(),
            vec![Some(deadline), None]
        );

        let incapable = DeadlineMockGpu {
            honors: false,
            observed: Mutex::new(Vec::new()),
        };
        assert!(!gpu_crown_backend_honors_deadline(
            &incapable,
            Some(deadline)
        ));
        assert!(gpu_crown_backend_honors_deadline(&incapable, None));
        {
            let _scope = GpuCrownBackendDeadlineScope::set(&incapable, Some(deadline));
        }
        assert!(incapable.observed.lock().unwrap().is_empty());

        let unbounded = DeadlineMockGpu {
            honors: true,
            observed: Mutex::new(Vec::new()),
        };
        {
            let _scope = GpuCrownBackendDeadlineScope::set(&unbounded, None);
        }
        assert!(unbounded.observed.lock().unwrap().is_empty());
    }

    #[test]
    fn exact_backend_deadline_scopes_restore_earliest_active_lease() {
        let now = std::time::Instant::now();
        let later = now + std::time::Duration::from_secs(20);
        let earlier = now + std::time::Duration::from_secs(10);
        let capable = DeadlineMockGpu {
            honors: true,
            observed: Mutex::new(Vec::new()),
        };

        let later_scope = GpuCrownBackendDeadlineScope::set(&capable, Some(later));
        let earlier_scope = GpuCrownBackendDeadlineScope::set(&capable, Some(earlier));
        assert_eq!(
            *capable.observed.lock().unwrap(),
            vec![Some(later), Some(earlier)]
        );

        // Out-of-order drop: removing the later lease must retain the earlier
        // deadline owned by the still-live scope.
        drop(later_scope);
        assert_eq!(
            capable.observed.lock().unwrap().last().copied(),
            Some(Some(earlier))
        );
        drop(earlier_scope);
        assert_eq!(capable.observed.lock().unwrap().last().copied(), Some(None));
    }

    #[test]
    fn dropping_current_minimum_restores_later_active_lease() {
        let now = std::time::Instant::now();
        let later = now + std::time::Duration::from_secs(20);
        let earlier = now + std::time::Duration::from_secs(10);
        let capable = DeadlineMockGpu {
            honors: true,
            observed: Mutex::new(Vec::new()),
        };

        let later_scope = GpuCrownBackendDeadlineScope::set(&capable, Some(later));
        let earlier_scope = GpuCrownBackendDeadlineScope::set(&capable, Some(earlier));
        drop(earlier_scope);
        assert_eq!(
            capable.observed.lock().unwrap().last().copied(),
            Some(Some(later)),
            "dropping the current minimum must restore the remaining later lease"
        );
        drop(later_scope);
        assert_eq!(capable.observed.lock().unwrap().last().copied(), Some(None));
    }

    #[test]
    fn acquisition_setter_panic_does_not_leave_ghost_lease() {
        let now = std::time::Instant::now();
        let failed_earlier = now + std::time::Duration::from_secs(10);
        let live_later = now + std::time::Duration::from_secs(20);
        let gpu = PanickingDeadlineMockGpu {
            panic_on_next_set: AtomicBool::new(true),
            observed: Mutex::new(Vec::new()),
        };

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _scope = GpuCrownBackendDeadlineScope::set(&gpu, Some(failed_earlier));
        }));
        assert!(panic.is_err(), "injected setter panic must propagate");
        assert_eq!(
            *gpu.observed.lock().unwrap(),
            vec![Some(failed_earlier), None],
            "failed acquisition must best-effort restore the previously empty backend slot"
        );

        let live_scope = GpuCrownBackendDeadlineScope::set(&gpu, Some(live_later));
        assert_eq!(
            gpu.observed.lock().unwrap().last().copied(),
            Some(Some(live_later)),
            "failed earlier acquisition must not remain as a registry ghost minimum"
        );
        drop(live_scope);
        assert_eq!(gpu.observed.lock().unwrap().last().copied(), Some(None));
    }

    #[test]
    fn exact_backend_deadline_scopes_compose_across_threads() {
        use std::sync::mpsc;

        let now = std::time::Instant::now();
        let later = now + std::time::Duration::from_secs(20);
        let earlier = now + std::time::Duration::from_secs(10);
        let capable = Arc::new(DeadlineMockGpu {
            honors: true,
            observed: Mutex::new(Vec::new()),
        });
        let (later_ready_tx, later_ready_rx) = mpsc::channel();
        let (earlier_ready_tx, earlier_ready_rx) = mpsc::channel();
        let (drop_later_tx, drop_later_rx) = mpsc::channel();
        let (later_dropped_tx, later_dropped_rx) = mpsc::channel();
        let (drop_earlier_tx, drop_earlier_rx) = mpsc::channel();

        let later_gpu = Arc::clone(&capable);
        let later_worker = std::thread::spawn(move || {
            let scope = GpuCrownBackendDeadlineScope::set(later_gpu.as_ref(), Some(later));
            later_ready_tx.send(()).unwrap();
            drop_later_rx.recv().unwrap();
            drop(scope);
            later_dropped_tx.send(()).unwrap();
        });
        later_ready_rx.recv().unwrap();

        let earlier_gpu = Arc::clone(&capable);
        let earlier_worker = std::thread::spawn(move || {
            let scope = GpuCrownBackendDeadlineScope::set(earlier_gpu.as_ref(), Some(earlier));
            earlier_ready_tx.send(()).unwrap();
            drop_earlier_rx.recv().unwrap();
            drop(scope);
        });
        earlier_ready_rx.recv().unwrap();

        drop_later_tx.send(()).unwrap();
        later_dropped_rx.recv().unwrap();
        assert_eq!(
            capable.observed.lock().unwrap().last().copied(),
            Some(Some(earlier)),
            "one worker must not clear another worker's live deadline"
        );
        drop_earlier_tx.send(()).unwrap();
        later_worker.join().unwrap();
        earlier_worker.join().unwrap();
        assert_eq!(capable.observed.lock().unwrap().last().copied(), Some(None));
    }

    #[test]
    fn wide_backend_status_is_hermetic_and_fail_closed() {
        assert_eq!(wide_backend_status(None), WideBackendStatus::Unavailable);

        let cpu: SharedEngine = Arc::new(NaiveCpuGemmEngine);
        assert_eq!(
            wide_backend_status(Some(&cpu)),
            WideBackendStatus::Unsound,
            "a factory result without a certified GPU CROWN capability must be rejected"
        );

        let unsound: SharedEngine = Arc::new(MockGpuCrownEngine { sound: false });
        assert_eq!(
            wide_backend_status(Some(&unsound)),
            WideBackendStatus::Unsound
        );

        let sound: SharedEngine = Arc::new(MockGpuCrownEngine { sound: true });
        assert_eq!(wide_backend_status(Some(&sound)), WideBackendStatus::Ready);

        assert_eq!(
            failed_wide_backend_status(None, None),
            WideBackendStatus::Unavailable
        );
        assert_eq!(
            failed_wide_backend_status(None, Some(&unsound)),
            WideBackendStatus::Unsound,
            "an uncertified legacy fallback must be reported as rejected, not absent"
        );
    }

    #[test]
    fn preinitialized_wide_backend_never_invokes_a_cold_blocking_factory() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let factory: OnceLock<CrownFactory> = OnceLock::new();
        let engine: OnceLock<Option<SharedEngine>> = OnceLock::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let factory_calls = Arc::clone(&calls);
        let installed = factory.set(Box::new(move || {
            factory_calls.fetch_add(1, Ordering::SeqCst);
            // Models an unpollable CUDA constructor. Reaching this line would
            // hang the test, exactly the regression this preinitialized-only
            // seam prevents.
            std::thread::park();
            None
        }));
        assert!(installed.is_ok());

        let selected =
            sound_wide_gpu_from_preinitialized_engines(true, preinitialized_engine(&engine), None);
        assert!(selected.is_none());
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "observing a cold engine slot must never run its registered factory"
        );
    }

    #[test]
    fn explicit_prewarm_makes_backend_visible_to_finite_selector() {
        let factory: OnceLock<CrownFactory> = OnceLock::new();
        let engine: OnceLock<Option<SharedEngine>> = OnceLock::new();
        let sound: SharedEngine = Arc::new(MockGpuCrownEngine { sound: true });
        let factory_sound = Arc::clone(&sound);
        assert!(factory
            .set(Box::new(move || Some(Arc::clone(&factory_sound))))
            .is_ok());

        assert!(preinitialized_engine(&engine).is_none());
        assert!(materialize_registered_engine(&factory, &engine).is_some());

        let lazy_calls = std::cell::Cell::new(0usize);
        let selected = select_lazy_backend_for_deadline(
            Some(std::time::Instant::now()),
            || sound_wide_gpu_from_engine(preinitialized_engine(&engine)),
            || {
                lazy_calls.set(lazy_calls.get() + 1);
                None
            },
        );
        assert!(selected.is_some());
        assert_eq!(
            lazy_calls.get(),
            0,
            "a finite lookup after prewarm must use only the published engine slot"
        );
    }

    #[test]
    fn finite_deadline_selector_never_invokes_lazy_initializer() {
        let observed = std::cell::Cell::new(0usize);
        let initialized = std::cell::Cell::new(0usize);
        let selected = select_lazy_backend_for_deadline(
            Some(std::time::Instant::now()),
            || {
                observed.set(observed.get() + 1);
                Some("ready")
            },
            || {
                initialized.set(initialized.get() + 1);
                Some("initialized")
            },
        );
        assert_eq!(selected, Some("ready"));
        assert_eq!(observed.get(), 1);
        assert_eq!(initialized.get(), 0);

        let selected = select_lazy_backend_for_deadline(
            None,
            || {
                observed.set(observed.get() + 1);
                Some("ready")
            },
            || {
                initialized.set(initialized.get() + 1);
                Some("initialized")
            },
        );
        assert_eq!(selected, Some("initialized"));
        assert_eq!(observed.get(), 1);
        assert_eq!(initialized.get(), 1);
    }

    /// T2.3 (Metal verdict routing): a non-f64 GPU (e.g. Metal) has no cuBLAS-f64
    /// CROWN, so its verdict substitute is the f32-SOUND path. This proves
    /// [`gpu_dag_ibp_forward_route`] — the graph-DAG IBP counterpart — routes ONLY a
    /// SOUND DAG engine into a gate-engaged verdict and MASKS an unsound one
    /// (→ None → CPU sound fallback), exactly as the CROWN/IBP routes do.
    #[test]
    fn dag_route_gate_selects_sound_and_masks_unsound() {
        let _g = lock_gate();
        let sound = MockSoundDagEngine;
        let unsound = MockUnsoundDagEngine;
        let cpu = NaiveCpuGemmEngine;

        // Gate OFF (speed mode): the fast DAG path is exposed for any DAG-capable
        // engine; verdict-safe because no verdict is decided when the gate is off.
        set_sound_gpu_crown_required(false);
        assert!(matches!(
            gpu_dag_ibp_forward_route(Some(&sound)),
            Some((_, false))
        ));
        assert!(matches!(
            gpu_dag_ibp_forward_route(Some(&unsound)),
            Some((_, false))
        ));
        assert!(gpu_dag_ibp_forward_route(Some(&cpu)).is_none());

        // Gate ON (verdict mode): ONLY the SOUND DAG engine may decide a bound.
        set_sound_gpu_crown_required(true);
        assert!(matches!(
            gpu_dag_ibp_forward_route(Some(&sound)),
            Some((_, true))
        ));
        assert!(
            gpu_dag_ibp_forward_route(Some(&unsound)).is_none(),
            "gate on: an unsound DAG engine MUST be masked so no verdict uses it"
        );
        assert!(gpu_dag_ibp_forward_route(Some(&cpu)).is_none());
        assert!(gpu_dag_ibp_forward_route(None).is_none());
    }

    /// Mock engine advertising a SOUND graph-DAG IBP forward.
    struct MockSoundDagEngine;
    impl GemmEngine for MockSoundDagEngine {
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
        fn as_gpu_dag_ibp_forward_ext(&self) -> Option<&dyn GpuDagIbpForwardExt> {
            Some(self)
        }
    }
    impl GpuDagIbpForwardExt for MockSoundDagEngine {
        fn prepare_dag_model_plan(
            &self,
            _plan: &ny_core::GpuDagIbpPlanDesc,
        ) -> ny_core::Result<Option<Box<dyn ny_core::GpuDagIbpModelPlan>>> {
            Ok(None)
        }
        fn provides_sound_gpu_dag_ibp(&self) -> bool {
            true
        }
    }

    /// Mock engine whose graph-DAG IBP forward is NOT sound (fast path only).
    struct MockUnsoundDagEngine;
    impl GemmEngine for MockUnsoundDagEngine {
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
        fn as_gpu_dag_ibp_forward_ext(&self) -> Option<&dyn GpuDagIbpForwardExt> {
            Some(self)
        }
    }
    impl GpuDagIbpForwardExt for MockUnsoundDagEngine {
        fn prepare_dag_model_plan(
            &self,
            _plan: &ny_core::GpuDagIbpPlanDesc,
        ) -> ny_core::Result<Option<Box<dyn ny_core::GpuDagIbpModelPlan>>> {
            Ok(None)
        }
        // provides_sound_gpu_dag_ibp defaults to false.
    }
}
