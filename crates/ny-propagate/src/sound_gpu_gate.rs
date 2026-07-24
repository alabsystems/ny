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
//! `a_pos·x_l + a_neg·x_u` in round-to-nearest f32, and the GPU backward `A·W`
//! shaders carry NO `γ_n·S` certified error (WGSL has no f64, so the CPU's exact
//! approach cannot port directly). A GPU-concretized bound can therefore be
//! *tighter than the true range* — and an over-tight bound on the verdict path can
//! flip a genuinely-violated instance to `Verified`/`unsat`/`hold`. In VNN-COMP
//! one incorrect verdict scores -150.
//!
//! Crucially this applies to *intermediate* GPU CROWN bounds as well, not only the
//! final concretization: an over-tight intermediate pre-activation bound silently
//! tightens every downstream bound, including the one that decides the verdict.
//! So when soundness is required the CROWN propagation that decides a
//! `Verified`/`unsat`/`hold` must run on the proven-sound CPU path *throughout*.
//!
//! # How the gate works
//!
//! Every CROWN site that would dispatch to a GPU backward does so via
//! `engine.as_gpu_crown_backward()`. There are exactly five such sites in
//! ny-propagate (the sequential CROWN core, two graph-alpha GPU suffixes, the
//! per-node IBP CROWN-partial backward, and the constrained graph backward
//! dispatch). Each one is routed through [`sound_gpu_crown_backward`], which
//! returns `None` — forcing the CPU sound fallback — whenever the gate is engaged.
//!
//! This is the single chokepoint the analysis calls for: flipping the gate masks
//! the GPU CROWN accelerator at *all* five sites at once, while leaving every other
//! GPU capability untouched. GPU GEMM (`gemm_f32`/`gemm_f64`), GPU IBP forward
//! (`as_gpu_ibp_forward*`), and the PGD/attack (sat-finding) path do NOT go through
//! `as_gpu_crown_backward`, so they keep their acceleration. Sat-finding only ever
//! *under*-approximates (it exhibits a concrete counterexample that is re-checked),
//! so float speed there cannot produce an unsound `Verified`.
//!
//! The competition / verdict-deciding entry point (`ny vnncomp`, competition mode)
//! engages the gate via [`set_sound_gpu_crown_required`]; interactive / speed-only
//! callers leave it disabled and keep the GPU CROWN fast-path.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use ny_core::{GemmEngine, GpuCrownBackward, GpuDagIbpForwardExt, GpuIbpForward};

/// Process-global LAZY factory for a sound GPU-resident CROWN backward engine
/// (e.g. CUDA `CudaGemmEngine`), installed from the CLI behind `--features cuda`.
/// Consulted by [`gpu_crown_backward_route`] as a fallback when the propagation's
/// own `engine` provides no sound GPU CROWN — so a native CUDA sound f64 CROWN
/// drives verdicts without `ny-propagate` depending on the `unsafe` CUDA crate
/// (the engine is a `&dyn GemmEngine`). Mirrors `sound_f64_gemm`'s pattern: the
/// engine is built once, on the first CROWN backward, so attack-only instances
/// pay no GPU init.
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

/// The lazily-materialized global sound GPU CROWN backend, if installed + sound.
fn global_sound_gpu_crown() -> Option<&'static dyn GpuCrownBackward> {
    let engine = SOUND_GPU_CROWN_ENGINE
        .get_or_init(|| SOUND_GPU_CROWN_FACTORY.get().and_then(|factory| factory()));
    engine
        .as_ref()
        .and_then(|e| e.as_gpu_crown_backward())
        .filter(|g| g.provides_sound_gpu_crown())
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
        let registered_engine = WIDE_SOUND_GPU_CROWN_FACTORY
            .get()
            .and_then(|factory| WIDE_SOUND_GPU_CROWN_ENGINE.get_or_init(factory).as_ref());
        if let Some(gpu) = sound_wide_gpu_from_engine(registered_engine) {
            report_wide_backend_status_once(WideBackendStatus::Ready);
            return Some(gpu);
        }

        // A caller that explicitly installed only the legacy global backend
        // should still be able to use it for wide calls.
        let legacy_engine = SOUND_GPU_CROWN_FACTORY
            .get()
            .and_then(|factory| SOUND_GPU_CROWN_ENGINE.get_or_init(factory).as_ref());
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

/// Process-global flag: when `true` (the DEFAULT), the *unsound* fast f32 GPU CROWN
/// backward is masked so every verdict-deciding CROWN bound comes from a proven-sound
/// path (the sound GPU-resident backward — still GPU-accelerated — or the CPU
/// f64+γ_n·S fallback).
///
/// DEFAULTS TO `true` (#gpu-crown-sound-default, 2026-07-05): a VERIFIER must not
/// decide a verdict on an unsound bound by default. Any entry point that does not
/// explicitly touch the gate (e.g. `ny verify`) is therefore SOUND. Speed-only
/// callers who KNOWINGLY accept an unsound verdict opt out per-run — the CLI exposes
/// `--allow-unsound-gpu-crown` (`ny beta-crown` / `ny verify`), which calls
/// `set_sound_gpu_crown_required(false)`; `ny vnncomp` can never opt out. Masking the
/// fast path only affects the CROWN VERDICT bound: GPU GEMM, GPU IBP forward, and the
/// PGD/attack (sat-finding) path keep their acceleration regardless.
/// The DEFAULT of the process-global gate — ONE source of truth (the static
/// initialiser + the `production_gate_default_is_sound` regression test read it).
pub(crate) const DEFAULT_SOUND_GPU_CROWN: bool = true;
static SOUND_GPU_CROWN_REQUIRED: AtomicBool = AtomicBool::new(DEFAULT_SOUND_GPU_CROWN);

/// Engage or release the soundness gate on the GPU CROWN fast-path.
///
/// When `required` is `true`, all CROWN propagation in this process routes through
/// the proven-sound CPU backward/concretize path instead of the unsound GPU f32
/// concretize. GPU GEMM, GPU IBP forward, and the PGD/attack path are unaffected.
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
pub(crate) fn gpu_crown_backward_route(
    engine: Option<&dyn GemmEngine>,
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
        return global_sound_gpu_crown().map(|g| (g, true));
    }
    // Gate not engaged: the passed engine's (fast) GPU path; else fall back to the
    // process-global CUDA sound CROWN (always sound), if installed.
    if let Some(g) = engine.and_then(|e| e.as_gpu_crown_backward()) {
        return Some((g, false));
    }
    global_sound_gpu_crown().map(|g| (g, true))
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

/// Scoped cooperative deadline on the routed GPU CROWN backward
/// (#w4-refresh-deadline).
///
/// Sets the deadline on the SAME backend the CROWN dispatch sites will route to
/// ([`gpu_crown_backward_route`]) and ALWAYS clears it on drop, so a stale
/// deadline can never leak into an unrelated later GPU call. Constructed only
/// when a deadline exists (no-deadline callers touch nothing). Cancellation is
/// cooperative and fail-open: nested scopes or concurrent threads at worst
/// clear each other's deadline early (work then runs to completion — the
/// pre-existing behavior), never produce a wrong bound (an expired check
/// surfaces as `DeadlineExceeded`, which callers handle with sound fallbacks).
pub(crate) struct GpuCrownDeadlineScope<'a> {
    gpu: Option<&'a dyn GpuCrownBackward>,
}

impl<'a> GpuCrownDeadlineScope<'a> {
    pub(crate) fn set(
        engine: Option<&'a dyn GemmEngine>,
        deadline: Option<std::time::Instant>,
    ) -> Self {
        let gpu = if deadline.is_some() {
            gpu_crown_backward_route(engine).map(|(g, _use_sound)| g)
        } else {
            None
        };
        if let Some(g) = gpu {
            g.set_crown_backward_deadline(deadline);
        }
        Self { gpu }
    }
}

impl Drop for GpuCrownDeadlineScope<'_> {
    fn drop(&mut self) {
        if let Some(g) = self.gpu {
            g.set_crown_backward_deadline(None);
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
    use ny_core::NaiveCpuGemmEngine;

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
        ) -> ny_core::Result<ny_core::GpuCrownResult> {
            // Deliberately returns a bogus (over-tight) bound; if the gate ever
            // let this run on the verdict path, the verdict would be unsound.
            Ok(ny_core::GpuCrownResult {
                lower_bounds: vec![0.0; num_specs],
                upper_bounds: vec![0.0; num_specs],
            })
        }

        fn provides_sound_gpu_crown(&self) -> bool {
            self.sound
        }
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
