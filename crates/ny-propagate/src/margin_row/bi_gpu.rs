// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #bi-gpu: the `#backward-interm` prefix pass PAIRS on the certified sound
//! f64 GPU conv route.
//!
//! # RETIRED in production (ENGINE_PROGRAM R2, 2026-09-29)
//!
//! Every routed product lands in a CERTIFIED trunk box: the device's cuBLAS
//! `Dgemm` sum IS the certified value, covered only by the lane's a-priori
//! order-independent charges. Program rule R2 is "the device never certifies"
//! until the §1.9 preconditions hold (M9 is unscheduled: `-fmad=false` kernels,
//! SASS-only authority, a live-state KAT against exact rationals, an A10G moat
//! with a 100% CPU recheck), and none of them holds for this route; its gated
//! tests ran only emulated in-process engines. It is also the per-op offload
//! shape R1 retired after `margin_row.gpu_conv` lost c2050 and c1590. So:
//!
//! * the typed key `margin_row.bi_gpu` is retired (preset validation refuses
//!   it; its axis is `Unsafe` / `EnvOnly`);
//! * an armed policy (the environment is the only route left) is REFUSED at
//!   the phase state: [`BiGpu::armed`] prints `[bi-gpu] refused
//!   reason=r2_device_never_certifies scope= route=cpu` and returns `None`, so
//!   every pair runs the dark CPU pair, bit for bit;
//! * the code below stays compiled and tested DARK behind a test-only
//!   admission (`with_device_certification_admitted_for_test`) and the
//!   injected engines, as the reference for an M9 device certifier.
//!
//! # Why this phase
//!
//! The root-phase intermediate refinement is the phase cct never finishes
//! (`scratchpad/vbs/diag6/cct.md` §2a): on the 120 s narrow-CNN7 rows
//! `#backward-interm` spends ~38-40 s of `work_secs`, covers ~30% of the
//! unstable neurons and never reaches trunk layer 4, and its per-column cost
//! tracks the dense prefix MAC count (2.2 / 3.9-4.9 / 6.3-6.7 / 7.3 ms per
//! Lower+Upper column at relu_b..relu_e). Every one of those MACs is a conv
//! backward product — the trunk is conv/affine/relu only (`engine.rs` refuses
//! a trunk Gemm) — i.e. exactly the op `#mr-gpu-eval` (`super::gpu_conv`)
//! moves to the process-global sound f64 GEMM engine. The WAVE6 cct plan P3
//! and Vibecheck study_1 I3 both name this as the enabler for the joint loop:
//! a full-coverage refresh must fall from ~45 s to seconds before
//! `#joint-refresh` can alternate at abc's cadence.
//!
//! # What
//!
//! When `margin_row.bi_gpu` is armed, every prefix pass pair of the phase —
//! the baseline dense pair, the `#bi-window` cropped pair, every
//! `#interm-alpha` candidate pair (dense or cropped) — and every
//! `#joint-refresh` pair runs on an engine whose conv backward route is a
//! [`GpuConvRoute::Scoped`] over the GLOBAL sound-f64 route. The scope counts
//! its own calls, so this phase proves its own engagement (`[bi-gpu] done`)
//! without touching the `[mr-gpu-conv]` receipt, and it is independent of
//! `margin_row.gpu_conv` (which, armed alone, already routes these passes
//! through the lane route exactly as before; armed together, the phase's
//! calls are counted here instead).
//!
//! # Soundness — the certificate is the lane's own, unchanged
//!
//! The moved op, its admission and its per-op CPU fallback are exactly
//! `super::gpu_conv`'s (read its module docs): the device consumes the stored
//! f64 operands bit-for-bit, the engine contract is binary64 RN in ANY
//! summation order, and the pass's UNCHANGED order-independent charges
//! (`g = gamma(k_bwd + 2) + rel + gamma*rel` inside `comb`, `certify_up(.,
//! gamma(k_bwd + 8))` on the `|W|` products, the envelope's same
//! `certify_up`) cover the device sum as they cover the CPU kernel's. A
//! prefix pass is a `run_collect_impl` pass like any other: the prefix relus'
//! gates, the error carry from an EXACT identity seed and the outward
//! `concretize_*` are untouched host code. So each routed pair is a certified
//! outward enclosure of its seeded pre-activations, and the phase's
//! shrink-only intersection consumes it exactly as it consumes a CPU pair.
//!
//! # Fail-closed, per pass
//!
//! * Per OP: any refusal (no/non-CUDA engine, deadline, backoff, non-finite
//!   operand or output, wrong length, transposed conv, CPU-only scope) runs
//!   the unchanged CPU kernel, which overwrites every element — bit-identical
//!   to the dark pass for that op; a pass whose every product refuses is
//!   bit-identical to the dark pass.
//! * Per PAIR, counted: `pairs_gpu` (every product on device), `pairs_mixed`,
//!   `pairs_cpu` (every product refused), `pairs_noconv` (nothing to route).
//! * RSS ADMISSION, per pair: one routed call holds, beyond the pass's own
//!   `dst`, a gathered `B` tile (`GPU_CONV_TILE_ELEMS` f64 = 32 MiB, or ONE
//!   spatial row `8 * k * R` bytes when that alone is larger), its output tile
//!   `C` (`ci / k` times the `B` tile, and never larger than the pass's own
//!   `ci * ip * R` output), the `|W|` copy (`8 * ci * k` bytes) and the
//!   engine's device copies of all three (host-visible on a unified-memory
//!   box). So the bound is the tile constant (or one spatial row) plus terms
//!   set by the conv's own shape, the largest of which (`C`) never exceeds
//!   the pass's own output buffer — NOT a pure constant (on the cct CNN7
//!   trunk at `R = 256` it is ~32 MiB `B` + <= ~4 MiB `C`). A
//!   pair is admitted only while the process memory headroom (cgroup- or
//!   `MemAvailable`-derived, self-RSS backstop included; competition runs
//!   uncapped) keeps `headroom / 2 >= BI_GPU_ADMIT_BYTES`; otherwise the WHOLE
//!   pair runs on the CPU kernel (route `Off`, even when `margin_row.gpu_conv`
//!   is armed; counted `rss_refused`). An unobservable headroom (no readable
//!   source at all, e.g. off Linux) admits: the extra working set is the
//!   shape-bounded set of tiles above.
//!
//! # DARK by default
//!
//! `NY_MARGIN_ROW_BI_GPU` (declaration default `false`; the retired typed key
//! `margin_row.bi_gpu` is refused at preset validation), frozen into
//! `BackwardIntermRuntimePolicy::gpu` with the rest of the phase's policy. Dark, no [`BiGpu`] exists: every engine is
//! `BackwardEngine::new` and every pair is the historical pair, byte-identical.
//! A `#rebake` round's explicit config (`from_config`) runs on the CPU, like
//! its `#bi-window` flag.
//!
//! # Engagement observable (fixed schema, every armed phase)
//!
//! * `[bi-gpu] armed scope=<backward_interm|joint_refresh> route=sound_f64_engine
//!   backend_req=cuda admit_bytes=N source=policy` when the phase state is
//!   built;
//! * `[bi-gpu] done armed=1 pairs= gpu= cpu_fallback= secs= scope= pairs_gpu=
//!   pairs_mixed= pairs_cpu= pairs_noconv= rss_refused= convs= gpu_signed=
//!   gpu_abs= gpu_env= gpu_macs= gpu_secs= <refusal counts> backend=
//!   verdict=<VOID|engaged>` on EVERY exit of the armed state (printed on
//!   drop, so an early `?` return or a deadline error still prints it). `gpu=`
//!   counts device-carried conv products, `cpu_fallback=` refused ones, `secs=`
//!   the wall of the routed pairs. `gpu=0` prints `verdict=VOID`: it never
//!   engaged, which is not a negative.

use std::sync::Arc;
use std::time::Instant;

use super::engine::BackwardEngine;
use super::gpu_conv::{self, GpuConvRoute, ScopedRoute, TallySnapshot};
use super::net::TwinNet;
use super::root::RootGates;

/// RSS admission floor per routed pair: half the process memory headroom must
/// cover this many bytes (one routed call's host tiles plus the engine's
/// device copies, with margin: the `B` tile is capped at 32 MiB by
/// `gpu_conv::GPU_CONV_TILE_ELEMS` unless one spatial row is larger, and the
/// `C` tile scales with `ci / k` — see the module docs, "RSS ADMISSION").
pub(crate) const BI_GPU_ADMIT_BYTES: u64 = 512 << 20;

/// Which phase a [`BiGpu`] serves (telemetry only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BiGpuScope {
    /// The root build's interleaved `#backward-interm` phase.
    BackwardInterm,
    /// `#joint-refresh`'s re-run of that phase over the finished root.
    JointRefresh,
}

impl BiGpuScope {
    /// Stable token printed as `scope=`.
    #[must_use]
    pub(crate) fn tag(self) -> &'static str {
        match self {
            Self::BackwardInterm => "backward_interm",
            Self::JointRefresh => "joint_refresh",
        }
    }
}

/// Open accounting of one admitted pair.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PairMark {
    before: TallySnapshot,
    t0: Instant,
}

/// Per-scope totals (test surface; the `[bi-gpu] done` line prints them).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct BiGpuStats {
    /// Admitted pairs that ran (every class below).
    pub(crate) pairs: usize,
    /// ... every conv product on the device.
    pub(crate) pairs_gpu: usize,
    /// ... some products on the device, some refused (CPU kernel).
    pub(crate) pairs_mixed: usize,
    /// ... every product refused (bit-identical to the dark pair).
    pub(crate) pairs_cpu: usize,
    /// ... no conv product at all (nothing to route).
    pub(crate) pairs_noconv: usize,
    /// Pairs the RSS admission sent to the CPU kernel (not in `pairs`).
    pub(crate) rss_refused: usize,
    /// Wall seconds of the admitted pairs.
    pub(crate) secs: f64,
    /// The scope's conv-product counters.
    pub(crate) tally: TallySnapshot,
}

/// One armed phase's device route and its accounting.
pub(crate) struct BiGpu {
    scope: BiGpuScope,
    scoped: Arc<ScopedRoute>,
    /// Memory-headroom probe for the RSS admission (the live process probe in
    /// production; unit tests inject a fixed value).
    headroom: fn() -> Option<u64>,
    stats: BiGpuStats,
}

impl BiGpu {
    /// The ARMED state over the global sound-f64 route — REFUSED outside
    /// tests (R2, see the module docs): prints `[bi-gpu] refused
    /// reason=r2_device_never_certifies ...` and returns `None`, so the phase
    /// runs its dark CPU pairs bit for bit. Admitted (tests only): prints the
    /// armed line and starts the engine slot's one-time background
    /// materialization (non-blocking; the first pairs may refuse `no_engine`
    /// until it lands).
    #[must_use]
    pub(crate) fn armed(scope: BiGpuScope) -> Option<Self> {
        if !device_certification_admitted() {
            eprintln!("{}", refused_line(scope));
            return None;
        }
        crate::sound_f64_gemm::start_background_initialization();
        let gpu = Self::over(scope, GpuConvRoute::Global);
        eprintln!("{}", gpu.armed_line());
        Some(gpu)
    }

    /// Test injection: the same state over an explicit engine (admitted for
    /// every shape, like `GpuConvRoute::Engine`).
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_engine(scope: BiGpuScope, engine: Arc<dyn ny_core::GemmEngine>) -> Self {
        Self::over(scope, GpuConvRoute::Engine(engine))
    }

    /// Test injection: a fixed memory-headroom probe for the RSS admission.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_headroom(mut self, probe: fn() -> Option<u64>) -> Self {
        self.headroom = probe;
        self
    }

    fn over(scope: BiGpuScope, inner: GpuConvRoute) -> Self {
        Self {
            scope,
            scoped: ScopedRoute::new(inner),
            headroom: crate::network::crown_memory::process_memory_headroom_bytes,
            stats: BiGpuStats::default(),
        }
    }

    /// The `[bi-gpu] armed` line (fixed schema).
    #[must_use]
    pub(crate) fn armed_line(&self) -> String {
        format!(
            "[bi-gpu] armed scope={} route=sound_f64_engine backend_req=cuda \
             admit_bytes={BI_GPU_ADMIT_BYTES} source=policy",
            self.scope.tag()
        )
    }

    /// The engine route every admitted pair runs on.
    #[must_use]
    pub(crate) fn route(&self) -> GpuConvRoute {
        GpuConvRoute::Scoped(Arc::clone(&self.scoped))
    }

    /// RSS-admit ONE pair and open its accounting. `None`: the pair must run
    /// wholly on the CPU kernel (route `Off`; counted `rss_refused`).
    pub(crate) fn open_pair(&mut self) -> Option<PairMark> {
        if !admits((self.headroom)()) {
            self.stats.rss_refused += 1;
            return None;
        }
        Some(PairMark {
            before: self.scoped.tally(),
            t0: Instant::now(),
        })
    }

    /// Close an admitted pair: classify it by what its products did.
    pub(crate) fn close_pair(&mut self, mark: PairMark) {
        let d = self.scoped.tally().since(&mark.before);
        let s = &mut self.stats;
        s.secs += mark.t0.elapsed().as_secs_f64();
        s.pairs += 1;
        if d.convs == 0 {
            s.pairs_noconv += 1;
        } else if d.fallback() == 0 {
            s.pairs_gpu += 1;
        } else if d.gpu == 0 {
            s.pairs_cpu += 1;
        } else {
            s.pairs_mixed += 1;
        }
    }

    /// Totals so far.
    #[must_use]
    pub(crate) fn stats(&self) -> BiGpuStats {
        BiGpuStats {
            tally: self.scoped.tally(),
            ..self.stats
        }
    }

    /// The `[bi-gpu] done` receipt (fixed schema: every field on every line).
    #[must_use]
    pub(crate) fn done_line(&self) -> String {
        let s = self.stats();
        let t = s.tally;
        let verdict = if t.gpu == 0 { "VOID" } else { "engaged" };
        format!(
            "[bi-gpu] done armed=1 pairs={} gpu={} cpu_fallback={} secs={:.3} scope={} \
             pairs_gpu={} pairs_mixed={} pairs_cpu={} pairs_noconv={} rss_refused={} \
             convs={} gpu_signed={} gpu_abs={} gpu_env={} gpu_macs={} gpu_secs={:.3}{} \
             backend={} verdict={verdict}",
            s.pairs,
            t.gpu,
            t.fallback(),
            s.secs,
            self.scope.tag(),
            s.pairs_gpu,
            s.pairs_mixed,
            s.pairs_cpu,
            s.pairs_noconv,
            s.rss_refused,
            t.convs,
            t.gpu_signed,
            t.gpu_abs,
            t.gpu_env,
            t.macs,
            t.ns as f64 / 1e9,
            t.refusal_fields(),
            gpu_conv::backend(),
        )
    }
}

impl Drop for BiGpu {
    /// The receipt prints on EVERY exit of the armed state — a completed
    /// phase, an early `?` return, a deadline error, an unwinding panic — so
    /// "armed and never engaged" is always one grep away.
    fn drop(&mut self) {
        eprintln!("{}", self.done_line());
        #[cfg(test)]
        {
            let receipt = (self.scope, self.stats());
            let _ = THREAD_RECEIPTS.try_with(|cell| {
                if let Ok(mut receipts) = cell.try_borrow_mut() {
                    receipts.push(receipt);
                }
            });
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Test surface: the `(scope, totals)` of every armed state dropped on this
    /// thread, i.e. exactly what its `[bi-gpu] done` receipt printed. The phase
    /// state lives INSIDE a root build, so this is how a test proves that a
    /// policy-armed build really built the state and offered products to the
    /// route (engagement), instead of inferring it from equal outputs.
    static THREAD_RECEIPTS: std::cell::RefCell<Vec<(BiGpuScope, BiGpuStats)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Test surface: drain this thread's recorded `[bi-gpu] done` receipts.
#[cfg(test)]
#[must_use]
pub(crate) fn take_thread_receipts() -> Vec<(BiGpuScope, BiGpuStats)> {
    THREAD_RECEIPTS.with(|cell| std::mem::take(&mut *cell.borrow_mut()))
}

/// The refusal line of an armed phase (fixed schema; R2, see the module docs).
#[must_use]
pub(crate) fn refused_line(scope: BiGpuScope) -> String {
    format!(
        "[bi-gpu] refused reason=r2_device_never_certifies scope={} route=cpu \
         (margin_row.bi_gpu is retired: its cuBLAS products would enter certified boxes)",
        scope.tag()
    )
}

/// R2 (the device never certifies): production never admits the device route.
#[cfg(not(test))]
const fn device_certification_admitted() -> bool {
    false
}

/// Tests only: whether this thread admitted the device route
/// ([`with_device_certification_admitted_for_test`]).
#[cfg(test)]
fn device_certification_admitted() -> bool {
    TEST_DEVICE_ADMIT.with(std::cell::Cell::get)
}

#[cfg(test)]
thread_local! {
    static TEST_DEVICE_ADMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// TEST-ONLY: run `f` with the device route admitted on this thread (the
/// retired code path stays exercised as the reference for an M9 device
/// certifier). Restores the refusal afterwards, also on unwind.
#[cfg(test)]
pub(crate) fn with_device_certification_admitted_for_test<T>(f: impl FnOnce() -> T) -> T {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_DEVICE_ADMIT.with(|c| c.set(false));
        }
    }
    TEST_DEVICE_ADMIT.with(|c| c.set(true));
    let _restore = Restore;
    f()
}

/// The RSS admission rule (pure, for the unit tests): admit while half the
/// observed headroom covers [`BI_GPU_ADMIT_BYTES`]; an unobservable headroom
/// admits (one call's extra working set is the tile constant plus terms
/// bounded by the conv's own shape; module docs, "RSS ADMISSION").
#[must_use]
pub(crate) fn admits(headroom: Option<u64>) -> bool {
    headroom.is_none_or(|bytes| bytes / 2 >= BI_GPU_ADMIT_BYTES)
}

/// Which `(route, engine)` one prefix pair runs on: the scoped device route
/// when the pair was admitted; the CPU-only route when the phase is armed but
/// the RSS admission refused it; the phase's own dark engine (`route = None`,
/// exactly the historical pair) when `#bi-gpu` is dark.
#[must_use]
pub(crate) fn pick<'r, 'e, 'a>(
    admitted: bool,
    gpu_route: Option<&'r GpuConvRoute>,
    eng_gpu: Option<&'e BackwardEngine<'a>>,
    cpu_route: &'r GpuConvRoute,
    eng_cpu: Option<&'e BackwardEngine<'a>>,
    dark: &'e BackwardEngine<'a>,
) -> (Option<&'r GpuConvRoute>, &'e BackwardEngine<'a>) {
    match (gpu_route, eng_gpu, eng_cpu) {
        (Some(route), Some(routed), _) if admitted => (Some(route), routed),
        (Some(_), _, Some(cpu)) => (Some(cpu_route), cpu),
        _ => (None, dark),
    }
}

/// `BackwardEngine::new(net, root)` over `route` when given. `None` IS
/// `BackwardEngine::new` (the dark phase's engine, byte-identical).
#[must_use]
pub(crate) fn engine<'a>(
    net: &'a TwinNet,
    root: &'a RootGates,
    route: Option<&GpuConvRoute>,
) -> BackwardEngine<'a> {
    let eng = BackwardEngine::new(net, root);
    match route {
        Some(route) => eng.with_gpu_conv_route(route.clone()),
        None => eng,
    }
}
