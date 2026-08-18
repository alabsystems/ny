// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #alpha-steering-proposal: process-global PROPOSAL channel for the DAG
//! α-CROWN margin-gradient lane — the α-gradient sibling of
//! [`crate::fast_f32_gemm`] (same lazy first-install-wins factory pattern).
//!
//! WHY. On hosts without a verdict-authority resident backend (Metal), the
//! armed margin-gradient lane dies at the authority filter
//! (`engine.as_gpu_crown_backward()` is `None` on the quarantined proof
//! adapter; `WgpuDevice::provides_sound_gpu_crown()` is `false`) and every
//! iteration steers by the single-layer LOCAL gradient rule instead of the
//! true `∂(binding-row lower bound)/∂α` adjoint. Wrong-direction gradients
//! make the ascent patience-exit after a handful of cheap iterations
//! (MEASURED, `docs/MACOS_ACCELERATION_RESEARCH_2026-08-01.md` correction:
//! ~1.2s/iter, quit at 7).
//!
//! THE GATE THIS CHANNEL IS — AND IS NOT. `provides_sound_gpu_crown()` asks
//! "may this engine's numbers decide a verdict"; it stays exactly as-is for
//! every caller. This channel answers a DIFFERENT question — "may this
//! engine PROPOSE α gradients" — and the answer is yes for any adapter,
//! because of the consumer, not the device: gradients only steer α ∈ [0,1]
//! (design I3, consult #4 — "gradients may be approximate and used only to
//! propose alpha"), every iterate is re-evaluated by the certified CPU fold,
//! and best-state retention rejects regressions. Installation into this
//! channel is that consent; the channel is only ever populated with
//! proposal-grade wrappers (`ny_gpu::GradientSteeringDevice`) whose sole live
//! capability is the deadline-bounded joint-α adjoint — an API that returns
//! gradients only, never bounds.
//!
//! ROUTING CONTRACT. The engine stored here must NEVER be assignable to a
//! bounding engine slot (`DagAlphaLoopContext::engine`, `propagate_*`
//! engines, BaB bounding). The single consumer is the margin-gradient
//! proposal seam in `propagate_dag/gradients`, which forwards the adjoint
//! output exclusively as `MarginGradientRequest` gradient input.
//!
//! Lazy by design: the factory runs once, on the first armed margin-gradient
//! iteration that consults the channel, so runs that never arm the lane pay
//! no adapter/pipeline initialization.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, OnceLock,
};

use ny_core::GemmEngine;

type SharedEngine = Arc<dyn GemmEngine>;
type Factory = Box<dyn Fn() -> Option<SharedEngine> + Send + Sync>;

/// Installed factory (set once at startup; cheap — no device init).
static FACTORY: OnceLock<Factory> = OnceLock::new();
/// Lazily-built engine, materialized from the factory on first consult.
static ENGINE: OnceLock<Option<SharedEngine>> = OnceLock::new();
/// Process-global count of joint-α adjoint proposals actually dispatched
/// through this channel (provenance telemetry for the flight record).
static PROPOSAL_DISPATCHES: AtomicU64 = AtomicU64::new(0);
/// Process-global count of CPU binding-row replay gradients actually consumed
/// by the margin lane (#binding-row-replay — the engine-less sibling of
/// `PROPOSAL_DISPATCHES`; same truthful-outcome contract: incremented only
/// after accepted gradients).
static REPLAY_DISPATCHES: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
thread_local! {
    /// Test-only override so seam/loop tests can inject a scripted proposal
    /// engine without touching the process-global `OnceLock`s (which would
    /// leak across tests in the shared test process).
    static TEST_OVERRIDE: std::cell::RefCell<Option<SharedEngine>> =
        const { std::cell::RefCell::new(None) };
}

/// Install a process-global factory for the α-gradient proposal engine. The
/// factory is invoked at most once (on the first armed margin-gradient
/// iteration without a verdict-authority resident backend); it should
/// construct a proposal-grade steering wrapper, or return `None` when no
/// adapter is available. First installation wins (idempotent). Construction
/// failure must never fail the verification run — the lane keeps its bounded
/// local-gradient fallback.
pub fn set_alpha_gradient_steering_factory<F>(factory: F)
where
    F: Fn() -> Option<SharedEngine> + Send + Sync + 'static,
{
    let _ = FACTORY.set(Box::new(factory));
}

/// The proposal engine, materializing it from the factory on first consult.
/// `None` when no factory was installed or construction failed/declined.
pub(crate) fn steering_engine() -> Option<SharedEngine> {
    #[cfg(test)]
    {
        if let Some(engine) = TEST_OVERRIDE.with(|slot| slot.borrow().clone()) {
            return Some(engine);
        }
    }
    ENGINE
        .get_or_init(|| FACTORY.get().and_then(|factory| factory()))
        .clone()
}

/// Record one actual joint-α adjoint proposal dispatch (called by the seam
/// only after `Ok` gradients were accepted).
pub(crate) fn note_proposal_dispatch() {
    PROPOSAL_DISPATCHES.fetch_add(1, Ordering::Relaxed);
}

/// Record one accepted CPU binding-row replay gradient (#binding-row-replay;
/// called by the seam only after the replay's gradients were handed to the
/// margin lane).
pub(crate) fn note_replay_dispatch() {
    REPLAY_DISPATCHES.fetch_add(1, Ordering::Relaxed);
}

/// Non-forcing snapshot for the flight record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlphaGradientSteeringTelemetry {
    /// Joint-α adjoint proposals dispatched and accepted through the channel.
    pub proposal_dispatches: u64,
    /// CPU binding-row replay gradients dispatched and accepted by the margin
    /// lane (#binding-row-replay; no engine involved).
    pub replay_dispatches: u64,
    /// Backend identity of the materialized engine. `None` means the channel
    /// never materialized a usable engine; it must not be described as armed.
    pub backend: Option<&'static str>,
}

/// Snapshot the channel WITHOUT forcing factory materialization.
pub fn telemetry() -> AlphaGradientSteeringTelemetry {
    AlphaGradientSteeringTelemetry {
        proposal_dispatches: PROPOSAL_DISPATCHES.load(Ordering::Relaxed),
        replay_dispatches: REPLAY_DISPATCHES.load(Ordering::Relaxed),
        backend: ENGINE
            .get()
            .and_then(|engine| engine.as_ref())
            .map(|engine| engine.backend_provenance()),
    }
}

#[cfg(test)]
pub(crate) fn with_test_steering<R>(engine: SharedEngine, run: impl FnOnce() -> R) -> R {
    TEST_OVERRIDE.with(|slot| *slot.borrow_mut() = Some(engine));
    let result = run();
    TEST_OVERRIDE.with(|slot| *slot.borrow_mut() = None);
    result
}
