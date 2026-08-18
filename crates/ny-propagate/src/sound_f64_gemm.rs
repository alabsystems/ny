// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Process-global optional **sound** f64 GEMM accelerator (e.g. CUDA cuBLAS).
//!
//! The verdict-deciding CPU CROWN backward computes `A·W` and `|A|·|W|` in f64
//! (see [`crate::layers::linear::crown_single::aw_f64_with_abssum`]) and certifies
//! the f64 rounding with `γ_n·S`. On a datacenter GPU, cuBLAS `Dgemm` computes the
//! SAME f64 products far faster (~18–34× vs the single-threaded CPU loop on
//! CROWN-shaped matrices). Because Higham's `γ_n·S` bound is **summation-order
//! independent**, the certified error stays valid for the cuBLAS result — verified
//! against an exact-rational oracle (0 violations across cancellation + large-k
//! cases). Crucially this accelerates the *sound CPU f64 path* itself, so it is
//! valid even under [`crate::sound_gpu_gate`] (it does NOT use the unsound wgpu
//! f32 CROWN), i.e. it speeds up competition verdicts.
//!
//! # Lazy by design
//!
//! The accelerator is installed as a **factory** (`set_sound_f64_gemm_factory`),
//! not an engine: the factory is invoked **once, on the first large `A·W`** that
//! would benefit. Easy / sat-by-attack / conv-dominated instances therefore never
//! pay the GPU context/handle initialization (~0.4s) — important for per-instance
//! VNN-COMP processes. Mirrors `sound_gpu_gate`'s process-global pattern.
//!
//! The engine is a `&dyn ny_core::GemmEngine` (trait in `ny-core`), so this crate
//! does not depend on the `unsafe` CUDA FFI crate (`ny-cuda`); the CLI installs a
//! factory that constructs the concrete `CudaGemmEngine`.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use ny_core::{GemmEngine, NyError, Result};

type SharedEngine = Arc<dyn GemmEngine>;
type Factory = Box<dyn Fn() -> Option<SharedEngine> + Send + Sync>;

thread_local! {
    /// Depth of a call-local host-only certified-f64 scope.
    ///
    /// This is deliberately thread-local rather than process-global: an
    /// independently executing verifier lane may continue using its installed
    /// CUDA engine while a private CPU worker pool performs a comprehensive
    /// intermediate sweep. The pool installs the guard on every worker for the
    /// worker's full lifetime, so nested Rayon work inherits the same policy by
    /// executing on another guarded worker.
    static CPU_ONLY_F64_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Nest-safe, unwind-restoring authority for an explicitly CPU-only sound-f64
/// propagation scope.
///
/// The guard does not change any process-global engine slot. Narrow Linear and
/// Conv CROWN dispatch seams consult [`cpu_only_f64_active`] before global
/// admission and route to their existing deadline-aware faer implementation.
#[must_use = "hold the guard for the complete CPU-only propagation scope"]
pub(crate) struct CpuOnlyF64Guard {
    // A thread-local depth guard must be dropped on the thread that created it.
    // `Rc` makes the otherwise-zero-sized guard neither Send nor Sync.
    _thread_affine: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl CpuOnlyF64Guard {
    pub(crate) fn new() -> Self {
        CPU_ONLY_F64_DEPTH.with(|depth| {
            depth.set(
                depth
                    .get()
                    .checked_add(1)
                    .expect("CPU-only sound-f64 scope nesting overflow"),
            );
        });
        Self {
            _thread_affine: std::marker::PhantomData,
        }
    }
}

impl Drop for CpuOnlyF64Guard {
    fn drop(&mut self) {
        CPU_ONLY_F64_DEPTH.with(|depth| {
            let current = depth.get();
            debug_assert!(current > 0, "unbalanced CPU-only sound-f64 scope");
            depth.set(current.saturating_sub(1));
        });
    }
}

/// Whether the current thread is inside an explicit CPU-only certified-f64
/// propagation scope.
#[inline]
pub(crate) fn cpu_only_f64_active() -> bool {
    CPU_ONLY_F64_DEPTH.with(|depth| depth.get() > 0)
}

#[cfg(test)]
static CPU_ONLY_GLOBAL_ACCESSOR_ATTEMPTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static GLOBAL_ACCESSOR_ATTEMPTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static ACCESSOR_COUNTER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Record only attempts made while a CPU-only scope is active. This makes the
/// counter useful across all private-pool workers without counting unrelated
/// tests or verifier lanes that legitimately consult the global engine.
#[cfg(test)]
#[inline]
fn note_cpu_only_global_accessor_attempt() {
    GLOBAL_ACCESSOR_ATTEMPTS.fetch_add(1, Ordering::SeqCst);
    if cpu_only_f64_active() {
        CPU_ONLY_GLOBAL_ACCESSOR_ATTEMPTS.fetch_add(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
pub(crate) fn reset_cpu_only_global_accessor_attempts() {
    CPU_ONLY_GLOBAL_ACCESSOR_ATTEMPTS.store(0, Ordering::SeqCst);
}

#[cfg(test)]
pub(crate) fn cpu_only_global_accessor_attempts() -> usize {
    CPU_ONLY_GLOBAL_ACCESSOR_ATTEMPTS.load(Ordering::SeqCst)
}

#[cfg(test)]
pub(crate) fn global_accessor_attempts() -> usize {
    GLOBAL_ACCESSOR_ATTEMPTS.load(Ordering::SeqCst)
}

#[cfg(test)]
pub(crate) fn accessor_counter_test_lock() -> std::sync::MutexGuard<'static, ()> {
    ACCESSOR_COUNTER_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Installed factory (set once at startup; cheap — no device init).
static FACTORY: OnceLock<Factory> = OnceLock::new();
/// Lazily-built engine, materialized from the factory on first use.
static ENGINE: OnceLock<Option<SharedEngine>> = OnceLock::new();
/// Marks that some thread has entered (or spawned) the one-time initialization.
/// Deadline-bearing callers use this atomic instead of waiting on OnceLock's
/// internal initialization lock.
static INITIALIZATION_STARTED: AtomicBool = AtomicBool::new(false);
/// The bounded admission wait expired (or its background initializer failed).
/// Future deadline calls decline immediately while still checking whether the
/// background factory eventually published an engine.
static INITIALIZATION_ABANDONED: AtomicBool = AtomicBool::new(false);
static DEADLINE_ADMISSION_READY: AtomicU64 = AtomicU64::new(0);
static DEADLINE_ADMISSION_UNAVAILABLE: AtomicU64 = AtomicU64::new(0);
static DEADLINE_ADMISSION_TIMEOUTS: AtomicU64 = AtomicU64::new(0);
static DEADLINE_ADMISSION_WAIT_US: AtomicU64 = AtomicU64::new(0);

/// Lock-free telemetry for cold, deadline-bearing accelerator admission.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DeadlineAdmissionStats {
    pub ready: u64,
    pub unavailable: u64,
    pub bounded_timeouts: u64,
    pub wait_us: u64,
}

/// Read aggregate deadline-admission telemetry without performing I/O on a
/// deadline-bearing verifier thread.
#[must_use]
pub fn deadline_admission_stats() -> DeadlineAdmissionStats {
    DeadlineAdmissionStats {
        ready: DEADLINE_ADMISSION_READY.load(Ordering::Relaxed),
        unavailable: DEADLINE_ADMISSION_UNAVAILABLE.load(Ordering::Relaxed),
        bounded_timeouts: DEADLINE_ADMISSION_TIMEOUTS.load(Ordering::Relaxed),
        wait_us: DEADLINE_ADMISSION_WAIT_US.load(Ordering::Relaxed),
    }
}

fn record_deadline_admission(outcome: &AtomicU64, wait: Duration) {
    outcome.fetch_add(1, Ordering::Relaxed);
    DEADLINE_ADMISSION_WAIT_US.fetch_add(
        u64::try_from(wait.as_micros()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
}

/// Install a process-global factory for a sound f64 GEMM accelerator. The factory
/// is invoked at most once (on the first large `A·W`); it should construct an
/// engine whose `gemm_f64` is exact IEEE-f64 (e.g. cuBLAS `Dgemm`) so the `γ_n·S`
/// certified-error bound remains valid, or return `None` if unavailable. First
/// installation wins (idempotent).
pub fn set_sound_f64_gemm_factory<F>(factory: F)
where
    F: Fn() -> Option<SharedEngine> + Send + Sync + 'static,
{
    let _ = FACTORY.set(Box::new(factory));
}

/// Directly install a concrete engine (wraps it in a trivial factory). Useful for
/// tests / non-lazy callers.
pub fn set_sound_f64_gemm_engine(engine: SharedEngine) {
    set_sound_f64_gemm_factory(move || Some(engine.clone()));
}

/// Whether a factory is installed (does NOT force engine construction).
#[must_use]
pub fn is_installed() -> bool {
    FACTORY.get().is_some()
}

/// Run `f` with the engine, lazily materializing it from the factory on first
/// call. Returns `None` when no factory is installed or it yields no engine
/// (callers then use the CPU path).
pub(crate) fn with_engine<R>(f: impl FnOnce(&dyn GemmEngine) -> R) -> Option<R> {
    #[cfg(test)]
    note_cpu_only_global_accessor_attempt();
    let engine = ENGINE.get_or_init(|| {
        INITIALIZATION_STARTED.store(true, Ordering::Release);
        FACTORY.get().and_then(|factory| factory())
    });
    engine.as_ref().map(|e| f(e.as_ref()))
}

/// Force the one-time engine materialization so a test can observe WHICH
/// factory won the slot.
///
/// [`ENGINE`] is a `OnceLock`, so "the registered factory is the one that
/// materializes" is only observable in a process where nothing has materialized
/// yet — and `with_engine_deadline` sits on a dozen CROWN/BaB paths, so inside
/// the lib-test binary essentially any other test gets there first. A test that
/// needs this must therefore live in its own integration-test binary; see
/// `tests/cpu_sound_f64_floor.rs`.
///
/// `pub` only because that binary is a separate crate. Not part of the
/// supported API — production code wants [`with_engine`] or
/// `with_engine_deadline`, both of which materialize as a side effect anyway.
#[doc(hidden)]
pub fn force_engine_materialization_for_test() {
    let _ = with_engine(|_| ());
}

/// Observe the already-published engine slot without invoking or waiting for
/// the registered factory.
#[inline]
fn preinitialized_engine(engine: &OnceLock<Option<SharedEngine>>) -> Option<&SharedEngine> {
    engine.get().and_then(Option::as_ref)
}

/// Run `f` against an engine that is ALREADY materialized, or return `None`.
///
/// Unlike [`with_engine`] this never invokes the factory, and unlike
/// [`with_engine_deadline`] it never waits, never spawns, and never marks
/// initialization abandoned. It is the fail-closed accessor for *advisory*
/// questions ("would you want this shape?") that must not perturb the
/// established admission machinery: no engine ready ⇒ the caller keeps its
/// historical behaviour.
pub(crate) fn with_preinitialized_engine<R>(f: impl FnOnce(&dyn GemmEngine) -> R) -> Option<R> {
    #[cfg(test)]
    note_cpu_only_global_accessor_attempt();
    preinitialized_engine(&ENGINE).map(|engine| f(engine.as_ref()))
}

/// Start the ordinary one-time background materialization and return
/// IMMEDIATELY, without waiting for its outcome.
///
/// This is the non-blocking half of [`with_engine_deadline`]'s admission: it
/// lets an advisory caller (which may only consult an already-materialized
/// engine) break the chicken-and-egg where the engine is never materialized
/// because no call ever crosses the historical constant floor — while paying
/// none of the bounded wait, and crucially never setting
/// `INITIALIZATION_ABANDONED` on a timeout, which would disable the accelerator
/// for the calls that DO cross the floor.
pub(crate) fn start_background_initialization() {
    #[cfg(test)]
    note_cpu_only_global_accessor_attempt();
    if ENGINE.get().is_some() || FACTORY.get().is_none() {
        return;
    }
    if INITIALIZATION_ABANDONED.load(Ordering::Acquire) {
        return;
    }
    let _ = spawn_initializer_once();
}

/// Spawn the one-time background initializer if this thread wins the race.
///
/// Returns `false` only when the spawn itself failed, in which case
/// initialization is marked abandoned (the historical behaviour).
fn spawn_initializer_once() -> bool {
    if INITIALIZATION_STARTED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return true;
    }
    let spawn = std::thread::Builder::new()
        .name("ny-sound-f64-init".into())
        .spawn(|| {
            let initialized = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = ENGINE.get_or_init(|| FACTORY.get().and_then(|factory| factory()));
            }));
            if initialized.is_err() && ENGINE.get().is_none() {
                INITIALIZATION_ABANDONED.store(true, Ordering::Release);
            }
        });
    if spawn.is_err() {
        INITIALIZATION_ABANDONED.store(true, Ordering::Release);
        return false;
    }
    true
}

/// Admit an already-materialized sound-f64 engine for finite-deadline shared
/// graph BaB.
///
/// The shared executor can issue both long resident CROWN folds and generic
/// GEMM work, so the engine must advertise full-surface deadline safety in
/// addition to verdict-grade CROWN soundness and cooperative cancellation for
/// the broad deadline slot. A partial or non-cooperative engine remains usable
/// by its existing root/f64 call sites, but is deliberately refused here.
fn shared_bab_engine_from_preinitialized(engine: Option<&SharedEngine>) -> Option<&dyn GemmEngine> {
    let engine = engine?;
    if !engine.supports_deadline_safe_post_root_multi_objective_bab() {
        return None;
    }
    let gpu = engine.as_gpu_crown_backward()?;
    (gpu.provides_sound_gpu_crown() && gpu.honors_crown_backward_deadline())
        .then_some(engine.as_ref())
}

/// Return the already-materialized sound-f64 engine only when it is safe for a
/// finite-deadline shared graph-BaB handoff.
///
/// This is a read-only `OnceLock::get` seam. It never calls `FACTORY`, never
/// enters `get_or_init`, and never waits for another initializer.
#[must_use]
pub(crate) fn preinitialized_sound_gpu_engine() -> Option<&'static dyn GemmEngine> {
    shared_bab_engine_from_preinitialized(preinitialized_engine(&ENGINE))
}

/// Deadline-safe access to the lazily initialized engine.
///
/// `OnceLock::get_or_init` waits when another thread is constructing the
/// engine, and the CUDA factory itself can take hundreds of milliseconds. A
/// finite-deadline verifier must not enter either opaque wait. This accessor
/// therefore:
///
/// 1. uses only non-blocking `ENGINE.get()` reads on the calling thread;
/// 2. starts the ordinary one-time factory on a background thread at most once;
/// 3. polls readiness for a small, bounded admission window; and
/// 4. returns `Ok(None)` (the caller's CPU fallback) if initialization is still
///    unavailable while useful verifier budget remains.
///
/// A factory that hangs can strand only its background initializer. It cannot
/// strand the deadline-bearing verifier thread.
pub(crate) fn with_engine_deadline<R>(
    deadline: Instant,
    f: impl FnOnce(&dyn GemmEngine) -> R,
) -> Result<Option<R>> {
    #[cfg(test)]
    note_cpu_only_global_accessor_attempt();
    const MAX_INITIALIZATION_WAIT: Duration = Duration::from_secs(2);
    let admission_started = Instant::now();
    if Instant::now() >= deadline {
        return Err(NyError::DeadlineExceeded(
            "sound f64 GEMM: deadline exceeded before engine admission".into(),
        ));
    }
    if let Some(engine) = ENGINE.get() {
        return Ok(engine.as_ref().map(|engine| f(engine.as_ref())));
    }
    if FACTORY.get().is_none() {
        return Ok(None);
    }
    if INITIALIZATION_ABANDONED.load(Ordering::Acquire) {
        return Ok(None);
    }

    if !spawn_initializer_once() {
        return Ok(None);
    }

    let admission_end = deadline.min(Instant::now() + MAX_INITIALIZATION_WAIT);
    loop {
        if let Some(engine) = ENGINE.get() {
            if Instant::now() >= deadline {
                return Err(NyError::DeadlineExceeded(
                    "sound f64 GEMM: deadline exceeded during engine admission".into(),
                ));
            }
            let outcome = if engine.is_some() {
                &DEADLINE_ADMISSION_READY
            } else {
                &DEADLINE_ADMISSION_UNAVAILABLE
            };
            record_deadline_admission(outcome, admission_started.elapsed());
            return Ok(engine.as_ref().map(|engine| f(engine.as_ref())));
        }
        let now = Instant::now();
        if now >= deadline {
            return Err(NyError::DeadlineExceeded(
                "sound f64 GEMM: deadline exceeded during engine initialization".into(),
            ));
        }
        if now >= admission_end {
            INITIALIZATION_ABANDONED.store(true, Ordering::Release);
            record_deadline_admission(&DEADLINE_ADMISSION_TIMEOUTS, admission_started.elapsed());
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[cfg(test)]
mod preinitialized_shared_bab_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use ny_core::{GpuCrownBackward, GpuCrownLayer, GpuCrownResult};

    use super::*;

    struct MockGpuEngine {
        deadline_safe_bab_surface: bool,
        sound: bool,
        cooperative_deadline: bool,
    }

    impl GemmEngine for MockGpuEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            Err(NyError::UnsupportedOp("test engine".into()))
        }

        fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
            Some(self)
        }

        fn supports_deadline_safe_post_root_multi_objective_bab(&self) -> bool {
            self.deadline_safe_bab_surface
        }
    }

    impl GpuCrownBackward for MockGpuEngine {
        fn crown_backward_gpu(
            &self,
            _layers: &[GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<GpuCrownResult> {
            Err(NyError::UnsupportedOp("test engine".into()))
        }

        fn provides_sound_gpu_crown(&self) -> bool {
            self.sound
        }

        fn honors_crown_backward_deadline(&self) -> bool {
            self.cooperative_deadline
        }
    }

    #[test]
    fn cold_preinitialized_lookup_never_invokes_blocking_factory() {
        let factory: OnceLock<Factory> = OnceLock::new();
        let engine: OnceLock<Option<SharedEngine>> = OnceLock::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let factory_calls = Arc::clone(&calls);
        assert!(factory
            .set(Box::new(move || {
                factory_calls.fetch_add(1, Ordering::SeqCst);
                std::thread::park();
                None
            }))
            .is_ok());

        let selected = shared_bab_engine_from_preinitialized(preinitialized_engine(&engine));

        assert!(selected.is_none());
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "a cold get-only lookup must never enter the registered factory"
        );
    }

    #[test]
    fn shared_bab_handoff_requires_deadline_safe_sound_cooperative_surface() {
        for (deadline_safe_bab_surface, sound, cooperative_deadline, expected) in [
            (false, false, false, false),
            (false, false, true, false),
            (false, true, false, false),
            (false, true, true, false),
            (true, false, false, false),
            (true, false, true, false),
            (true, true, false, false),
            (true, true, true, true),
        ] {
            let slot: OnceLock<Option<SharedEngine>> = OnceLock::new();
            assert!(slot
                .set(Some(Arc::new(MockGpuEngine {
                    deadline_safe_bab_surface,
                    sound,
                    cooperative_deadline,
                })))
                .is_ok());

            assert_eq!(
                shared_bab_engine_from_preinitialized(preinitialized_engine(&slot)).is_some(),
                expected,
                "deadline_safe_bab_surface={deadline_safe_bab_surface} sound={sound} \
                 cooperative_deadline={cooperative_deadline}"
            );
        }
    }
}

#[cfg(test)]
mod cpu_only_scope_tests {
    use super::*;

    #[test]
    fn cpu_only_scope_is_nested_and_restored_on_unwind() {
        assert!(!cpu_only_f64_active());
        {
            let _outer = CpuOnlyF64Guard::new();
            assert!(cpu_only_f64_active());
            {
                let _inner = CpuOnlyF64Guard::new();
                assert!(cpu_only_f64_active());
            }
            assert!(cpu_only_f64_active());
            let unwind = std::panic::catch_unwind(|| {
                let _guard = CpuOnlyF64Guard::new();
                panic!("scripted unwind");
            });
            assert!(unwind.is_err());
            assert!(cpu_only_f64_active());
        }
        assert!(!cpu_only_f64_active());
    }

    #[test]
    fn guarded_accessor_counter_detects_an_accidental_global_lookup() {
        let _lock = accessor_counter_test_lock();
        reset_cpu_only_global_accessor_attempts();
        let _guard = CpuOnlyF64Guard::new();
        let _ = with_preinitialized_engine(|_| ());
        assert_eq!(cpu_only_global_accessor_attempts(), 1);
    }
}
