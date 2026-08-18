// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #wallhugger-arming-cost (defect A6): asynchronous arming for the
//! attack-steering accelerator.
//!
//! `b030e2a8` restored the falsification accelerator by constructing
//! [`ny_gpu::AttackSteeringDevice`] INLINE in `handle_beta_crown_command` —
//! `WgpuDevice::new` requests an adapter + device and compiles ~20 compute
//! pipelines, all serially on the instance's critical path, BEFORE any
//! verification work starts. On near-wall unsat rows (banked solves with ≤5s
//! of margin) that serial cost is pure loss: the accelerator's only power is
//! finding `sat` faster, and an unsat proof never consults it.
//!
//! This module removes the serial cost WITHOUT a routing heuristic:
//!
//! - [`AttackSteering`] arms the engine on a detached background thread
//!   started at command start, so construction overlaps model loading and
//!   bound setup. The attack wrapper shares ny-gpu's one ordinary WGPU context
//!   with the alpha-gradient and FL-value proposal wrappers.
//! - Attack lanes TAKE the engine through [`AttackEngineSource::take`]. By
//!   default this never blocks: if arming has not finished (or failed), the
//!   lane proceeds un-steered, exactly as if no accelerator existed. A
//!   measurement sweep may set `NY_ATTACK_ARMING_BLOCK=1` to wait for the
//!   one-shot arming result and remove scheduler timing from an A/B. The
//!   flight recorder captures that `NY_*` flag in the run artifact.
//!
//! Verdict-neutral by construction: this handle reaches ONLY the attack call
//! sites that previously received `attack_gemm_engine` (the
//! #attack-steering-unquarantine channel). Bound / precheck / BaB work never
//! sees it, and every candidate still passes the unchanged admission gates
//! (engine confirm + global-box guard; trusted-ORT + true-f64 on the scored
//! path).
//!
//! Process-exit note: the arming thread is detached. If the instance decides
//! before arming completes, process teardown terminates the thread inside
//! wgpu/driver setup — the same class of interruption as a Ctrl-C during
//! device creation, and it happens strictly AFTER the verdict is published.

use ny_core::GemmEngine;
use std::ffi::OsStr;
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use tracing::{info, warn};

fn attack_arming_block_from(value: Option<&OsStr>) -> bool {
    value == Some(OsStr::new("1"))
}

#[cfg(not(test))]
fn attack_arming_block_enabled() -> bool {
    attack_arming_block_from(std::env::var_os("NY_ATTACK_ARMING_BLOCK").as_deref())
}

// Unit tests exercise both branches through `take_with_block` without
// mutating the process-global environment (other tests run in parallel).
#[cfg(test)]
fn attack_arming_block_enabled() -> bool {
    false
}

/// One-shot arming state for the attack-steering accelerator.
///
/// The slot is written exactly once, by whichever constructor is used:
/// - unfilled          → still arming (takers proceed un-steered),
/// - `Some(engine)`    → armed,
/// - `None`            → disarmed for good (CPU route / construction failed).
pub(crate) struct AttackSteering {
    slot: Arc<OnceLock<Option<Arc<dyn GemmEngine>>>>,
}

impl AttackSteering {
    /// No accelerator, ever (CPU backend route; tests).
    pub(crate) fn disarmed() -> Self {
        let slot = Arc::new(OnceLock::new());
        let _ = slot.set(None);
        Self { slot }
    }

    /// An engine that already exists (shared-engine route): ready immediately,
    /// no thread, no arming cost.
    pub(crate) fn ready(engine: Arc<dyn GemmEngine>) -> Self {
        let slot = Arc::new(OnceLock::new());
        let _ = slot.set(Some(engine));
        Self { slot }
    }

    /// Arm on a detached background thread. `init` runs off the critical
    /// path; its failure disarms steering and must never fail the run.
    ///
    /// Callers use [`Self::arming_wgpu`]; this seam takes the closure so the
    /// never-blocks contract is testable with a mocked slow/stuck `init`.
    pub(crate) fn arming_in_background<F>(label: &'static str, init: F) -> Self
    where
        F: FnOnce() -> ny_core::Result<Arc<dyn GemmEngine>> + Send + 'static,
    {
        let slot = Arc::new(OnceLock::new());
        let thread_slot = Arc::clone(&slot);
        let spawned = std::thread::Builder::new()
            .name("ny-attack-arming".into())
            .spawn(move || {
                let started = Instant::now();
                let armed = match init() {
                    Ok(engine) => {
                        info!(
                            arming_ms = started.elapsed().as_millis() as u64,
                            label,
                            "attack-steering engine armed in background; falsification \
                             lanes take it from their next non-blocking take-point \
                             (#wallhugger-arming-cost)"
                        );
                        Some(engine)
                    }
                    Err(error) => {
                        warn!(
                            arming_ms = started.elapsed().as_millis() as u64,
                            label,
                            %error,
                            "attack-steering engine unavailable; attack lanes stay on \
                             CPU steering (#wallhugger-arming-cost)"
                        );
                        None
                    }
                };
                let _ = thread_slot.set(armed);
            });
        if let Err(error) = spawned {
            warn!(
                %error,
                "could not spawn the attack-steering arming thread; attack lanes \
                 stay on CPU steering (#wallhugger-arming-cost)"
            );
            let _ = slot.set(None);
        }
        Self { slot }
    }

    /// Arm the WGPU attack-steering device (the `AttackSteeringRoute::WgpuDevice`
    /// route) in the background. Same engine, same provenance, same quarantine
    /// posture as the former inline construction — only the WHERE of the cost
    /// changes.
    pub(crate) fn arming_wgpu() -> Self {
        Self::arming_in_background("wgpu-attack-steering", || {
            ny_gpu::AttackSteeringDevice::new_wgpu()
                .map(|device| Arc::new(device) as Arc<dyn GemmEngine>)
        })
    }

    /// Non-blocking: the armed engine, or `None` while arming is in flight /
    /// after it failed / when disarmed. `OnceLock::get` never waits.
    pub(crate) fn engine_if_ready(&self) -> Option<Arc<dyn GemmEngine>> {
        self.slot.get().and_then(Clone::clone)
    }

    /// Poll for readiness for at most `max_wait`, then answer whatever the slot
    /// holds. Returns as soon as the slot is filled (armed OR disarmed), so a
    /// construction failure costs the failure latency, never the full wait.
    ///
    /// #attack-steering-arming-race: a lane whose ENTIRE budget slice is the
    /// attack (soundnessbench spends 85% of the instance in upfront PGD) takes
    /// the engine exactly once, ~200 ms after command start — before WGPU
    /// arming (~360 ms) can finish. A pure non-blocking take therefore loses
    /// the accelerator for the whole slice on a coin-flip. Callers bound this
    /// wait by a small fraction of their OWN slice, so a near-wall row whose
    /// attack window is milliseconds still cannot pay a meaningful cost — the
    /// property `#wallhugger-arming-cost` (A6) was protecting.
    fn engine_within(&self, max_wait: std::time::Duration) -> Option<Arc<dyn GemmEngine>> {
        let deadline = Instant::now() + max_wait;
        loop {
            if let Some(armed) = self.slot.get() {
                return armed.clone();
            }
            if Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// Resolve according to the measurement determinism policy. The default
    /// remains non-blocking; an explicitly requested measurement may wait for
    /// the one-shot arming result.
    fn engine_for_take(&self, block: bool) -> Option<Arc<dyn GemmEngine>> {
        if block {
            self.slot.wait().clone()
        } else {
            self.engine_if_ready()
        }
    }
}

/// Process-global capability-bearing WGPU attack-wrapper armer.
///
/// The VNN-COMP wrapper's speculative exact-VJP pre-wave and the verifier's
/// ordinary attack lanes run at different times but need the same capability
/// view. A local timed-out armer would leave detached initialization overlapping
/// the sequential fallback and discard the completed wrapper. First use starts
/// exactly one background initializer; every later attack consumer takes from
/// the same state. The wrapper itself borrows ny-gpu's process-global ordinary
/// WGPU context, which is also reused by the alpha-gradient and FL-value views.
pub(crate) fn shared_wgpu_attack_steering() -> &'static AttackSteering {
    static SHARED: OnceLock<AttackSteering> = OnceLock::new();
    SHARED.get_or_init(AttackSteering::arming_wgpu)
}

/// The attack-engine channel threaded through dispatch into the falsification
/// lanes (successor of the resolved-up-front `attack_gemm_engine` handle).
/// `Copy` so per-group recursion and late take-points forward it freely.
#[derive(Clone, Copy)]
pub(crate) enum AttackEngineSource<'a> {
    /// A pre-resolved engine reference (test faces mirroring the proof
    /// channel), or a hard `None`. Production always routes through
    /// `Arming`; only the `#[cfg(test)]` faces construct this variant.
    #[cfg_attr(not(test), allow(dead_code))]
    Static(Option<&'a dyn GemmEngine>),
    /// Live arming state: every take re-checks readiness, so lanes pick the
    /// engine up as soon as the background thread finishes.
    Arming(&'a AttackSteering),
}

impl<'a> AttackEngineSource<'a> {
    /// Test-face constructor (`DispatchContext` fixtures).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn disarmed() -> Self {
        Self::Static(None)
    }

    /// Take the steering engine according to the measurement determinism
    /// policy. Historical/default behavior is non-blocking; the exact opt-in
    /// waits for the one-shot arming result.
    pub(crate) fn take(&self) -> Option<ResolvedAttackEngine<'a>> {
        self.take_with_block(attack_arming_block_enabled())
    }

    fn take_with_block(&self, block: bool) -> Option<ResolvedAttackEngine<'a>> {
        match self {
            Self::Static(engine) => engine.map(ResolvedAttackEngine::Borrowed),
            Self::Arming(steering) => steering
                .engine_for_take(block)
                .map(ResolvedAttackEngine::Owned),
        }
    }

    /// Bounded take for a lane that OWNS a large budget slice: wait up to
    /// `max_wait` for arming to settle (see [`AttackSteering::engine_within`]).
    /// `Static` faces answer immediately — there is nothing to wait for.
    pub(crate) fn take_within(
        &self,
        max_wait: std::time::Duration,
    ) -> Option<ResolvedAttackEngine<'a>> {
        match self {
            Self::Static(engine) => engine.map(ResolvedAttackEngine::Borrowed),
            Self::Arming(steering) => steering
                .engine_within(max_wait)
                .map(ResolvedAttackEngine::Owned),
        }
    }
}

/// A successfully taken attack engine. Owned when it came from the background
/// armer (the `Arc` keeps it alive for the take's scope), borrowed when the
/// caller supplied a pre-resolved reference.
pub(crate) enum ResolvedAttackEngine<'a> {
    Borrowed(&'a dyn GemmEngine),
    Owned(Arc<dyn GemmEngine>),
}

impl ResolvedAttackEngine<'_> {
    pub(crate) fn as_gemm(&self) -> &dyn GemmEngine {
        match self {
            Self::Borrowed(engine) => *engine,
            Self::Owned(engine) => engine.as_ref(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_core::NyError;
    use std::sync::mpsc;
    use std::time::Duration;

    /// Minimal engine stub; identity is carried by `backend_provenance`.
    struct StubEngine(&'static str);

    impl GemmEngine for StubEngine {
        fn backend_provenance(&self) -> &'static str {
            self.0
        }

        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> ny_core::Result<Vec<f32>> {
            Ok(Vec::new())
        }
    }

    fn poll_until_ready(steering: &AttackSteering) -> Arc<dyn GemmEngine> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(engine) = steering.engine_if_ready() {
                return engine;
            }
            assert!(
                Instant::now() < deadline,
                "arming thread never published its result"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// #wallhugger-arming-cost acceptance 1: arming NEVER blocks the caller.
    /// The mocked init is parked on a channel — a stand-in for a slow/stuck
    /// driver — and every take must return immediately, un-steered.
    #[test]
    fn arming_never_blocks_while_init_is_stuck() {
        let (release, gate) = mpsc::channel::<()>();
        let steering = AttackSteering::arming_in_background("stuck-init", move || {
            // Parked until the test releases it: simulates WgpuDevice::new
            // wedged in adapter/device/pipeline setup.
            let _ = gate.recv();
            Ok(Arc::new(StubEngine("stub-armed")) as Arc<dyn GemmEngine>)
        });

        let take_started = Instant::now();
        for _ in 0..3 {
            assert!(
                steering.engine_if_ready().is_none(),
                "a not-yet-armed engine must resolve to None, not wait"
            );
        }
        assert!(
            take_started.elapsed() < Duration::from_secs(1),
            "takes against a stuck init must return immediately (elapsed {:?})",
            take_started.elapsed()
        );

        // Release the init; a LATER take picks the engine up.
        release.send(()).expect("arming thread is alive");
        let engine = poll_until_ready(&steering);
        assert_eq!(engine.backend_provenance(), "stub-armed");
    }

    #[test]
    fn measurement_block_gate_is_exact() {
        assert!(!attack_arming_block_from(None));
        for value in ["", "0", "true", "01", " 1", "1 "] {
            assert!(!attack_arming_block_from(Some(OsStr::new(value))));
        }
        assert!(attack_arming_block_from(Some(OsStr::new("1"))));
    }

    #[test]
    fn measurement_take_waits_until_arming_resolves() {
        let (release, gate) = mpsc::channel::<()>();
        let steering = Arc::new(AttackSteering::arming_in_background(
            "measurement-block",
            move || {
                let _ = gate.recv();
                Ok(Arc::new(StubEngine("deterministic-armed")) as Arc<dyn GemmEngine>)
            },
        ));
        let (result_tx, result_rx) = mpsc::channel();
        let taker_steering = Arc::clone(&steering);
        let taker = std::thread::spawn(move || {
            let source = AttackEngineSource::Arming(taker_steering.as_ref());
            let provenance = source
                .take_with_block(true)
                .map(|engine| engine.as_gemm().backend_provenance().to_string());
            result_tx.send(provenance).expect("publish blocked take");
        });

        assert!(
            result_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "measurement take must wait while arming is unresolved"
        );
        release.send(()).expect("release arming");
        assert_eq!(
            result_rx
                .recv_timeout(Duration::from_secs(10))
                .expect("blocked take resolves"),
            Some("deterministic-armed".to_string())
        );
        taker.join().expect("taker thread");
    }

    /// #wallhugger-arming-cost acceptance 2: while the engine is not ready,
    /// the attack seam falls back to the un-steered channel — exactly the
    /// `take().or(proof_channel)` pattern the disjunctive lane uses — and a
    /// re-take after arming completes picks the accelerator up.
    #[test]
    fn unsteered_fallback_used_when_not_ready() {
        let (release, gate) = mpsc::channel::<()>();
        let steering = AttackSteering::arming_in_background("slow-init", move || {
            let _ = gate.recv();
            Ok(Arc::new(StubEngine("stub-armed")) as Arc<dyn GemmEngine>)
        });
        let source = AttackEngineSource::Arming(&steering);

        // Mirror of disjunctive.rs: attack_engine = take().or(gemm_engine).
        let proof_stub = StubEngine("cpu-proof-channel");
        let proof_engine: Option<&dyn GemmEngine> = Some(&proof_stub);

        let take = source.take();
        let attack_engine = take.as_ref().map(|t| t.as_gemm()).or(proof_engine);
        assert_eq!(
            attack_engine
                .expect("fallback channel present")
                .backend_provenance(),
            "cpu-proof-channel",
            "not-ready arming must fall back to the un-steered channel"
        );

        // With no fallback channel the lane runs fully un-steered (None) —
        // the pre-b030e2a8 sequential CPU behavior, not an error.
        let no_fallback = source.take();
        assert!(no_fallback.as_ref().map(|t| t.as_gemm()).or(None).is_none());

        release.send(()).expect("arming thread is alive");
        poll_until_ready(&steering);
        let retake = source.take();
        let attack_engine = retake.as_ref().map(|t| t.as_gemm()).or(proof_engine);
        assert_eq!(
            attack_engine.expect("armed engine").backend_provenance(),
            "stub-armed",
            "a take-point AFTER arming completes must pick the accelerator up"
        );
    }

    /// #attack-steering-arming-race acceptance: a lane that owns a large slice
    /// and takes ONCE must be able to wait a bounded moment for arming. The
    /// wait ends the instant the slot fills, and is capped when it does not.
    #[test]
    fn bounded_take_waits_for_a_late_arming_then_gives_up() {
        // Late-but-arriving init: the bounded take must PICK IT UP, where the
        // non-blocking take (the pre-fix behavior) returns `None` and loses
        // the accelerator for the whole slice.
        let steering = AttackSteering::arming_in_background("late-init", || {
            std::thread::sleep(Duration::from_millis(60));
            Ok(Arc::new(StubEngine("stub-armed")) as Arc<dyn GemmEngine>)
        });
        let source = AttackEngineSource::Arming(&steering);
        assert!(
            source.take().is_none(),
            "precondition: the plain take must miss a still-arming engine"
        );
        let taken = source
            .take_within(Duration::from_secs(5))
            .expect("bounded take must pick up a late arming");
        assert_eq!(taken.as_gemm().backend_provenance(), "stub-armed");

        // Stuck init: the bound is honoured and the lane proceeds un-steered.
        let (_release, gate) = mpsc::channel::<()>();
        let stuck = AttackSteering::arming_in_background("stuck-init", move || {
            let _ = gate.recv();
            Ok(Arc::new(StubEngine("never")) as Arc<dyn GemmEngine>)
        });
        let started = Instant::now();
        assert!(AttackEngineSource::Arming(&stuck)
            .take_within(Duration::from_millis(50))
            .is_none());
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "the bounded take must not outlast its bound (elapsed {:?})",
            started.elapsed()
        );
    }

    /// Failed arming disarms for good: takers see `None`, never an error.
    #[test]
    fn failed_arming_resolves_to_unsteered() {
        let steering = AttackSteering::arming_in_background("failing-init", || {
            Err(NyError::UnsupportedConfiguration("no adapter".into()))
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        while steering.slot.get().is_none() {
            assert!(Instant::now() < deadline, "failure never published");
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(steering.engine_if_ready().is_none());
        assert!(AttackEngineSource::Arming(&steering).take().is_none());
    }

    /// Pre-resolved states behave like the old inline construction.
    #[test]
    fn ready_and_disarmed_resolve_immediately() {
        let ready = AttackSteering::ready(Arc::new(StubEngine("shared")));
        assert_eq!(
            ready
                .engine_if_ready()
                .expect("ready engine")
                .backend_provenance(),
            "shared"
        );

        let disarmed = AttackSteering::disarmed();
        assert!(disarmed.engine_if_ready().is_none());
        assert!(AttackEngineSource::disarmed().take().is_none());

        let stub = StubEngine("static-borrowed");
        let taken = AttackEngineSource::Static(Some(&stub)).take();
        assert_eq!(
            taken.expect("static engine").as_gemm().backend_provenance(),
            "static-borrowed"
        );
    }
}
