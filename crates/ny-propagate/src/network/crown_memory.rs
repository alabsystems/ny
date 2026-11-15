// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared CPU dense-materialization policy for CROWN paths.
//!
//! Provides upfront memory estimates and budget checks for both sequential
//! (unbatched) and batched dense identity/densification allocations. All CPU
//! dense materialization shares a single budget: `NY_DENSE_BUDGET_MB`.

use ny_core::{checked_shape_product, NyError};

#[cfg(target_os = "linux")]
use std::ffi::OsString;
#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::os::unix::ffi::OsStringExt;
#[cfg(target_os = "linux")]
use std::os::unix::fs::FileExt;
#[cfg(target_os = "linux")]
use std::path::{Component, Path, PathBuf};
#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(target_os = "linux")]
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Default CPU Dense-materialization budget for sequential CROWN in MiB.
pub(crate) const DEFAULT_CROWN_DENSE_BUDGET_MB: usize = 2048;

const CROWN_DENSE_BUDGET_ENV: &str = "NY_DENSE_BUDGET_MB";

/// Upfront estimate for a Dense coefficient-pair materialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenseMaterializationEstimate {
    pub site: &'static str,
    pub rows: usize,
    pub cols: usize,
    pub required_bytes: usize,
}

impl DenseMaterializationEstimate {
    /// Create a Dense-materialization estimate, saturating to `usize::MAX`
    /// when the exact byte count overflows.
    pub(crate) fn new(site: &'static str, rows: usize, cols: usize) -> Self {
        Self {
            site,
            rows,
            cols,
            required_bytes: dense_pair_bytes(rows, cols).unwrap_or(usize::MAX),
        }
    }

    /// Whether this estimate exceeds the configured budget.
    pub(crate) fn exceeds_budget(self, budget_bytes: usize) -> bool {
        self.required_bytes > budget_bytes
    }

    /// Render a consistent fallback detail string for diagnostics.
    pub(crate) fn budget_exceeded_details(self, budget_bytes: usize) -> String {
        format!(
            "dense materialization at {} requires {} bytes for {}x{} coefficient pair; budget is {} bytes",
            self.site, self.required_bytes, self.rows, self.cols, budget_bytes
        )
    }
}

/// Share of the observed host memory the Dense budget may claim
/// (#adaptive-dense-budget). The observation is already conservative -- on a
/// host without a per-moment "available" probe it is half of physical RAM -- so
/// halving it again keeps a single materialization to roughly a quarter of the
/// machine.
const ADAPTIVE_DENSE_BUDGET_SHARE_DIVISOR: u64 = 2;

/// Share of LIVE process headroom -- cgroup/RLIMIT, or since #mem-governor the
/// self-governing `MemAvailable`/`VmRSS` term -- that one advertised Dense
/// coefficient-pair estimate may claim.
///
/// This is deliberately stricter than the host-wide heuristic above.  The
/// 7-D Patches -> Dense conversion can hold eight same-sized matrices while
/// its admission estimate names only the lower/upper f32 pair (two matrices):
/// retained coefficients + first error side + f64 count/accumulator scratch +
/// the second error side peak at four times the advertised pair.  Dividing
/// kernel-enforced headroom by eight leaves half of that headroom after the
/// largest presently known local peak, for graph state, allocator/GEMM
/// overhead, and admissions elsewhere in the process.
const PROCESS_ENVELOPE_DENSE_BUDGET_SHARE_DIVISOR: u64 = 8;

/// Hard ceiling on the ADAPTIVE budget, regardless of host size. An explicit
/// `NY_DENSE_BUDGET_MB` is not subject to it -- an operator stating a number is
/// making an assertion about their machine.
const MAX_ADAPTIVE_DENSE_BUDGET_MB: usize = 12 * 1024;

/// System memory withheld from `MemAvailable` before NY will call the remainder
/// "headroom" (#mem-governor).
///
/// # Why this exists at all
///
/// `process_memory_headroom_bytes` used to observe ONLY kernel-enforced
/// envelopes -- cgroup `memory.max` and `RLIMIT_AS` -- and to report `None` when
/// neither exists. `vnncomp_scripts/run_instance.sh` installs NEITHER: its only
/// `export` is `OMP_NUM_THREADS=1`, and it sets no `ulimit`, no cgroup and no
/// RLIMIT. So in COMPETITION the clamp that the whole Dense-admission stack
/// consults was inert by construction, every gate saw the full
/// `MAX_ADAPTIVE_DENSE_BUDGET_MB` ceiling for the entire run, and nothing at all
/// bounded total RSS. Measured 2026-08-26 on this 121 GiB host: one cgan row
/// reached anon-rss 118 GB and took a GLOBAL OOM kill (`task=ny_pin_base`).
/// A crashed run scores zero and can take the rest of the benchmark with it.
///
/// # Why a reserve rather than a fraction of RAM
///
/// `MemAvailable` already accounts for reclaimable page cache, so it is the
/// kernel's own estimate of what a new allocation can obtain without swapping or
/// reclaiming pressure. What it does NOT leave is room for the kernel itself and
/// for the allocator overshoot between one admission and the next, which is why
/// a flat reserve is subtracted before the figure is treated as headroom.
///
/// 4 GiB is chosen so the term is INERT on a healthy host and binding only under
/// real pressure: on this 121.7 GiB box a quiescent `MemAvailable` of 114.5 GiB
/// leaves 110.5 GiB, and 110.5 / `PROCESS_ENVELOPE_DENSE_BUDGET_SHARE_DIVISOR`
/// = 13.8 GiB, still above the 12 GiB adaptive ceiling. The clamp therefore
/// changes nothing until `MemAvailable` falls below 100 GiB here -- i.e. until
/// something really is eating the machine.
///
/// # Where it is NOT inert, measured rather than assumed
///
/// "Inert" is exact for the Dense budget and for
/// `runtime_host_budget_bytes`/`post_c_survivor`, whose other terms are already
/// far tighter. It is NOT exact for `narrowest_conv_crown_mem_cap`
/// (`conv2d/ops_transpose_gemm.rs`), which applies `PROCESS_ENVELOPE_SHARE_DIVISOR`
/// = 16 to any process envelope while charging an observed host only
/// `MIN_ENVELOPE_SHARE_DIVISOR` = 4. On this host that turns a previously
/// absent term into 110.5 GiB / 16 = 6.9 GiB, just under the 7.6 GiB RAM-adaptive
/// policy cap that used to bind, so the conv per-buffer cap tightens by ~9%.
/// That is the intended reading of its own comment -- the /16 exists because a
/// hard process peak deserves it, and a global OOM is exactly as hard as a
/// cgroup one -- but it is a real behavioural delta on conv benchmarks, not a
/// no-op, and it is the term to look at first if a conv row regresses.
#[cfg(target_os = "linux")]
const SYSTEM_MEMORY_RESERVE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Share of physical RAM above which NY's OWN resident set forces the cheap
/// representation outright, whatever `MemAvailable` still claims
/// (#mem-governor-backstop).
///
/// The `MemAvailable` term above decays as the process grows, but it decays
/// TOO LATE in one specific and observed shape: `MemAvailable` counts
/// reclaimable page cache, so a process holding 100 GiB of anonymous memory on a
/// box with 20 GiB of cache still reads ~20 GiB "available" and keeps admitting
/// multi-GiB materializations right up to the kill. This backstop contributes
/// ZERO headroom -- refuse everything, take the patches/IBP path -- once NY's
/// own `VmRSS` crosses three quarters of `MemTotal`.
///
/// Three quarters, not a half: the largest legitimate `ny` resident set on
/// record is 14.17 GiB on a 24 GiB host (yolo_2023, 2026-07-31), which is 59% of
/// RAM. A one-half ceiling would have fired against that run; three quarters
/// (18 GiB there) clears it, and still stops the 118 GB cgan shape at ~91 GiB on
/// this host, with 30 GiB of the machine left.
#[cfg(target_os = "linux")]
const SELF_RSS_CEILING_NUMERATOR: u64 = 3;
/// Denominator of [`SELF_RSS_CEILING_NUMERATOR`].
#[cfg(target_os = "linux")]
const SELF_RSS_CEILING_DENOMINATOR: u64 = 4;

/// Minimum spacing between `#mem-governor` trace lines (#mem-governor).
///
/// Dense admission is a hot loop -- `backward_step.rs` alone reaches it four
/// times in one function, and a SafeNLP proof was measured at 13,500 probes --
/// so the governor must not print per call. One line every five seconds is
/// enough to show the headroom TRAJECTORY across a 1200 s cgan budget (~240
/// lines) while costing nothing.
#[cfg(target_os = "linux")]
const MEMORY_GOVERNOR_LOG_INTERVAL_MS: u64 = 5_000;

/// Reclaimable-now figure below which the host is reported as critically low
/// (#dense-budget-valve, withdrawn as a clamp 2026-08-01 -- DIAGNOSTIC ONLY).
///
/// At the instant of `JetsamEvent-2026-07-30-210002` the host had 4668 pages
/// free, 541 speculative and 0 purgeable -- 79 MiB reclaimable-now. A single
/// `ny` run legitimately drives this below 1.5 GiB on a 24 GiB host, which is
/// exactly why this must not gate anything: the process reporting the condition
/// is the one causing it.
#[cfg(not(target_os = "linux"))]
const LOW_MEMORY_REPORT_THRESHOLD_BYTES: u64 = 1536 * 1024 * 1024;

/// How long a live host-availability reading may be reused (#dense-budget-valve).
///
/// Dense admission sits in hot loops -- `backward_step.rs` alone calls
/// `cpu_crown_dense_budget_bytes()` four times in one function -- and the
/// pre-existing macOS probe spawned a `sysctl` CHILD PROCESS on every one of
/// those calls, uncached.
///
/// Deliberately ten times the Linux `MEMORY_PROBE_CACHE_TTL`, because the two
/// probes do not cost the same thing. Linux reads `MemAvailable` from an
/// already-open handle; macOS has to fork `vm_stat`, measured at 1.34 ms a call
/// on this host. At a 100 ms TTL a 2400 s run would spawn up to 24,000 of them
/// (~32 s of overhead, ~1.3% of the budget); at 1 s it is 2,400 (~3 s, ~0.13%).
///
/// One second costs nothing in responsiveness: the valve guards a condition that
/// takes seconds to develop -- allocating the multi-GiB buffers that trigger it
/// is not instantaneous -- so it does not need sub-second resolution.
#[cfg(not(target_os = "linux"))]
const LIVE_MEMORY_PROBE_TTL: Duration = Duration::from_secs(1);

/// Get the sequential CPU Dense-materialization budget in bytes.
///
/// Priority: `NY_DENSE_BUDGET_MB` env var > host-adaptive share > 2 GiB floor,
/// then the live cgroup/RLIMIT envelope clamps every path.  An operator value
/// overrides a heuristic; it cannot override a limit the kernel will enforce.
pub fn cpu_crown_dense_budget_bytes() -> usize {
    // #poll-amortise: ONE observation of the memory terms, used for both the
    // adaptive policy and the process-envelope clamp. This function used to
    // derive them TWICE per call -- once inside `adaptive_..._bytes` and once
    // again for the clamp -- so the clamp could be computed from a different
    // instant than the budget it clamps. Sharing one observation is strictly
    // more coherent as well as half the cost.
    let terms = amortised_memory_terms();
    let requested = explicit_cpu_crown_dense_budget_bytes()
        .unwrap_or_else(|| adaptive_cpu_crown_dense_budget_from_terms(terms));
    let budget = clamp_dense_budget_to_process_envelope(requested, terms.headroom_bytes);
    // #kfsb-production: a fan-out SHARE installed on this thread narrows the
    // budget (never widens it). No share is installed anywhere but inside a
    // shared-budget fan-out's own worker threads, so everywhere else this is
    // `budget` unchanged.
    apply_dense_budget_thread_cap(budget, dense_budget_thread_cap())
}

// ---------------------------------------------------------------------------
// #kfsb-production -- a per-thread SHARE of the dense budget for fan-outs.
// ---------------------------------------------------------------------------

thread_local! {
    /// The dense-budget share installed on this thread by
    /// [`DenseBudgetThreadCap`] (`None`: no share; the live budget applies).
    static DENSE_BUDGET_THREAD_CAP: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// The dense-budget share installed on this thread, if any.
pub(crate) fn dense_budget_thread_cap() -> Option<usize> {
    DENSE_BUDGET_THREAD_CAP.with(std::cell::Cell::get)
}

/// `budget` narrowed to `cap`. Pure; a cap never widens the budget.
pub(crate) const fn apply_dense_budget_thread_cap(budget: usize, cap: Option<usize>) -> usize {
    match cap {
        Some(cap) if cap < budget => cap,
        _ => budget,
    }
}

/// RAII: while alive, [`cpu_crown_dense_budget_bytes`] on THIS thread returns
/// at most `bytes` (#kfsb-production).
///
/// # Why
///
/// The dense budget is "what ONE advertised dense pair may claim": a share of
/// LIVE headroom (`headroom / 8`), sized so a single materialization's real
/// peak (up to four times the advertised pair) leaves half the headroom for
/// everything else. A parallel fan-out of `k` independent CROWN passes read
/// that same figure `k` times, so `k` passes could each claim it at once —
/// on a unified-memory host with no cgroup (competition), a global OOM. A
/// fan-out that installs `total / width` on each of its `width` workers keeps
/// the sum of what its passes may claim at or under the ONE budget a single
/// pass gets.
///
/// # Scope
///
/// The share lives on the THREAD that installed it. A fan-out therefore
/// installs it on EVERY worker of a pool it owns (the thread wrapper of a
/// scoped Rayon pool), so a nested Rayon job one of its passes spawns — run
/// by whichever worker of that pool steals it — sees the share too. Nested
/// installations take the minimum; dropping restores the previous value.
///
/// # Never a verdict input
///
/// A budget is route information (dense vs. the cheap representation),
/// never a bound: a smaller share can only make a pass take its cheaper,
/// looser — still sound — representation.
#[must_use = "the dense-budget share applies only while the guard is alive"]
pub(crate) struct DenseBudgetThreadCap {
    previous: Option<usize>,
    // A thread-local guard must be dropped on the thread that created it.
    _thread_affine: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl DenseBudgetThreadCap {
    /// Install `bytes` (or keep a tighter share already installed).
    pub(crate) fn install(bytes: usize) -> Self {
        let previous = DENSE_BUDGET_THREAD_CAP.with(|cap| {
            let narrowed = Some(apply_dense_budget_thread_cap(bytes, cap.get()));
            cap.replace(narrowed)
        });
        Self {
            previous,
            _thread_affine: std::marker::PhantomData,
        }
    }
}

impl Drop for DenseBudgetThreadCap {
    fn drop(&mut self) {
        DENSE_BUDGET_THREAD_CAP.with(|cap| cap.set(self.previous));
    }
}

fn clamp_dense_budget_to_process_envelope(
    requested_bytes: usize,
    process_headroom_bytes: Option<u64>,
) -> usize {
    process_headroom_bytes.map_or(requested_bytes, |headroom| {
        let share = usize::try_from(headroom / PROCESS_ENVELOPE_DENSE_BUDGET_SHARE_DIVISOR)
            .unwrap_or(usize::MAX);
        requested_bytes.min(share)
    })
}

// ---------------------------------------------------------------------------
// #poll-amortise -- one observation of the live memory terms, reused for a
// bounded window.
// ---------------------------------------------------------------------------

/// How long ONE observation of the process/host memory terms may be reused
/// across dense-admission gates (#poll-amortise).
///
/// # Why this exists
///
/// `cpu_crown_dense_budget_bytes()` is reached from 87 call sites, several of
/// them per graph node per BaB domain -- `merge_accumulator.rs` alone holds 13.
/// Every call used to derive the terms TWICE (see that function), and each
/// derivation takes a process-global `Mutex`, preads `/proc/self/cgroup` to
/// revalidate cgroup membership, then reads `/proc/self/limits`,
/// `/proc/self/status` and `/proc/meminfo` LIVE. `process_memory_headroom_from_snapshot`
/// documents that `/proc/self/status` generation walks the mm and is therefore
/// the dominant per-admission cost; this module's own header already records
/// that a SafeNLP proof reached 13,500 probes, and that figure predates the
/// admission gates now sitting on the per-domain CROWN path.
///
/// # Why 100 ms is a safe stride
///
/// This is a NEW value-staleness, not one the module already had. The Linux
/// probe snapshot's own `MEMORY_PROBE_CACHE_TTL` (also 100 ms) bounds only the
/// reuse of open descriptors and cgroup topology; before #poll-amortise every
/// gate re-read the VALUES live through those handles. What makes 100 ms
/// acceptable is the rate of the condition the governor guards, which develops
/// over seconds, not milliseconds: the measured 2026-08-26 blowup reached
/// 118 GB over ~1150 s, i.e. ~100 MB/s on average, so a 100 ms stale headroom
/// is off by ~10 MB against a `SYSTEM_MEMORY_RESERVE_BYTES` reserve of 4 GiB --
/// roughly 400x of margin. That is an AVERAGE: one dense materialization can
/// allocate far faster, and during such a burst the error is the burst's own
/// allocation rate times the window. The 1/8 envelope share and the 4 GiB
/// reserve absorb that, exactly as they already absorbed the pre-existing
/// check-then-allocate window between an admission and its allocation. The
/// derived budget moves by at most the headroom error divided by
/// `PROCESS_ENVELOPE_DENSE_BUDGET_SHARE_DIVISOR`.
///
/// The window also delays, by at most one TTL, what the live read noticed at
/// the next gate: a lowered `RLIMIT_AS`, a migration into a tighter cgroup, and
/// the self-RSS backstop crossing its ceiling. [`process_memory_envelope`], the
/// strict fail-closed API, is NOT served from this cache and stays live per
/// call.
///
/// # Why this cannot change a verdict
///
/// A budget is scheduling and route information, never a bound. `instance_budget`
/// states the doctrine for the wall clock and it holds here verbatim: a wrong
/// value can only mis-schedule -- pick the cheap representation instead of the
/// dense one -- never mis-certify. The figure is also an inherently live
/// measurement: two runs of the same instance already read different headrooms
/// at every probe, so there is no bit-identity to lose.
const MEMORY_TERMS_AMORTISE_TTL: Duration = Duration::from_millis(100);

/// Sentinel for `None` in the lock-free term cache. No real byte count reaches
/// `u64::MAX`.
const MEMORY_TERM_NONE: u64 = u64::MAX;

/// Declared procfs reads avoided per amortised probe, for the observable only.
///
/// Two derivations per `cpu_crown_dense_budget_bytes()` call, each reading
/// `/proc/self/cgroup` (membership revalidation), `/proc/self/limits`,
/// `/proc/self/status` and `/proc/meminfo`, plus one cgroup counter pair:
/// 2 x 5. This is a DECLARED count read off the call graph, not a measurement.
const PROCFS_READS_PER_BUDGET_PROBE: u64 = 10;

/// The live memory readings every dense-admission gate needs, observed
/// together so one budget decision can never mix two instants.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AmortisedMemoryTerms {
    observed_host_bytes: Option<u64>,
    headroom_bytes: Option<u64>,
    live_available_bytes: Option<u64>,
}

/// Probes served (cache hits + observations).
static MEMORY_TERM_PROBES: AtomicU64 = AtomicU64::new(0);
/// Probes that actually went to the kernel.
static MEMORY_TERM_OBSERVATIONS: AtomicU64 = AtomicU64::new(0);
static MEMORY_TERM_VALID_UNTIL_NANOS: AtomicU64 = AtomicU64::new(0);
static MEMORY_TERM_OBSERVED_HOST: AtomicU64 = AtomicU64::new(MEMORY_TERM_NONE);
static MEMORY_TERM_HEADROOM: AtomicU64 = AtomicU64::new(MEMORY_TERM_NONE);
static MEMORY_TERM_LIVE_AVAILABLE: AtomicU64 = AtomicU64::new(MEMORY_TERM_NONE);

const fn encode_memory_term(value: Option<u64>) -> u64 {
    match value {
        Some(bytes) if bytes != MEMORY_TERM_NONE => bytes,
        Some(_) => MEMORY_TERM_NONE - 1,
        None => MEMORY_TERM_NONE,
    }
}

const fn decode_memory_term(value: u64) -> Option<u64> {
    if value == MEMORY_TERM_NONE {
        None
    } else {
        Some(value)
    }
}

/// Monotonic nanoseconds since the first probe.
fn memory_term_clock_nanos() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let epoch = EPOCH.get_or_init(Instant::now);
    u64::try_from(epoch.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// The memory terms, from the amortised observation when it is still inside
/// [`MEMORY_TERMS_AMORTISE_TTL`].
///
/// Two threads may refresh concurrently; both write legitimate live
/// observations and the last writer wins, so a reader can at worst combine two
/// readings taken inside one TTL window. That is not a new hazard: the
/// two-derivation shape this replaces mixed two instants inside a SINGLE
/// budget call.
fn amortised_memory_terms() -> AmortisedMemoryTerms {
    MEMORY_TERM_PROBES.fetch_add(1, Ordering::Relaxed);
    let now = memory_term_clock_nanos();
    if now < MEMORY_TERM_VALID_UNTIL_NANOS.load(Ordering::Acquire) {
        return AmortisedMemoryTerms {
            observed_host_bytes: decode_memory_term(
                MEMORY_TERM_OBSERVED_HOST.load(Ordering::Relaxed),
            ),
            headroom_bytes: decode_memory_term(MEMORY_TERM_HEADROOM.load(Ordering::Relaxed)),
            live_available_bytes: decode_memory_term(
                MEMORY_TERM_LIVE_AVAILABLE.load(Ordering::Relaxed),
            ),
        };
    }
    MEMORY_TERM_OBSERVATIONS.fetch_add(1, Ordering::Relaxed);
    let terms = observe_memory_terms();
    MEMORY_TERM_OBSERVED_HOST.store(
        encode_memory_term(terms.observed_host_bytes),
        Ordering::Relaxed,
    );
    MEMORY_TERM_HEADROOM.store(encode_memory_term(terms.headroom_bytes), Ordering::Relaxed);
    MEMORY_TERM_LIVE_AVAILABLE.store(
        encode_memory_term(terms.live_available_bytes),
        Ordering::Relaxed,
    );
    let ttl = u64::try_from(MEMORY_TERMS_AMORTISE_TTL.as_nanos()).unwrap_or(u64::MAX);
    MEMORY_TERM_VALID_UNTIL_NANOS.store(now.saturating_add(ttl), Ordering::Release);
    terms
}

/// Discard the amortised observation. Test-only: in production the TTL is the
/// whole contract, and nothing needs to force a live read.
#[cfg(test)]
pub(crate) fn invalidate_amortised_memory_terms() {
    MEMORY_TERM_VALID_UNTIL_NANOS.store(0, Ordering::Release);
}

/// The UNCONDITIONAL `#poll-amortise` observable.
///
/// Maintained with no feature gate, no env read and no preset key, so a null
/// measurement can be told from a dark one: `memory_probes=0` means the
/// amortiser was never reached, `factor=1.00` means it was reached and saved
/// nothing.
#[must_use]
pub fn amortisation_line() -> String {
    let probes = MEMORY_TERM_PROBES.load(Ordering::Relaxed);
    let observations = MEMORY_TERM_OBSERVATIONS.load(Ordering::Relaxed);
    let saved = probes.saturating_sub(observations);
    let hundredths = probes.saturating_mul(100) / observations.max(1);
    format!(
        "[poll-amortise] memory_probes={probes} memory_observations={observations} \
         memory_probes_saved={saved} memory_factor={}.{:02} memory_ttl_ms={} \
         procfs_reads_saved_estimate={} {}",
        hundredths / 100,
        hundredths % 100,
        MEMORY_TERMS_AMORTISE_TTL.as_millis(),
        saved.saturating_mul(PROCFS_READS_PER_BUDGET_PROBE),
        crate::amortise::poll_amortisation_fragment(),
    )
}

#[cfg(test)]
mod amortise_tests {
    use super::*;

    /// The counters are PROCESS-GLOBAL and this lib test binary runs thousands
    /// of tests in parallel, many of which reach `cpu_crown_dense_budget_bytes`
    /// through the CROWN admission gates. So an exact delta on
    /// `MEMORY_TERM_PROBES` is not a property of this test, a concurrent
    /// refresher can overwrite the cached terms, and a window that outlives the
    /// TTL may legitimately re-observe. Each of those is SPORADIC; every defect
    /// this test exists to catch (no caching, invalidation that does not force
    /// an observation, a cache hit that returns something other than the
    /// stored observation, an uncounted probe) is SYSTEMATIC and fails every
    /// attempt. So: retry a bounded number of windows and require ONE window in
    /// which every assertion holds; never pass without one.
    ///
    /// The value check compares TERMS, not only budgets. On a large host the
    /// budget saturates at `MAX_ADAPTIVE_DENSE_BUDGET_MB` (at quiescence the GB10
    /// reads 12 GiB whatever the headroom is), so a budget-only equality could
    /// not see a hit path that decodes the wrong bytes. The miss path returns
    /// the live observation ITSELF; a hit must reproduce it field for field.
    #[test]
    fn dense_budget_probe_is_served_from_the_amortised_cache() {
        const PROBES_PER_WINDOW: u64 = 64;
        const ATTEMPTS: usize = 32;
        // Hold the env lock with the operator override REMOVED, so the budget
        // is the adaptive value derived from the cached terms and no concurrent
        // `with_crown_dense_budget_mb` test can change it mid-window.
        crate::tests::with_serialized_env_vars_removed(&[CROWN_DENSE_BUDGET_ENV], || {
            let mut last_failure = String::new();
            for attempt in 0..ATTEMPTS {
                invalidate_amortised_memory_terms();
                let window = Instant::now();
                let before_observations = MEMORY_TERM_OBSERVATIONS.load(Ordering::Relaxed);
                // A miss: this is the kernel observation itself, not a decoded copy.
                let observed = amortised_memory_terms();
                let after_first = MEMORY_TERM_OBSERVATIONS.load(Ordering::Relaxed);
                let first = clamp_dense_budget_to_process_envelope(
                    adaptive_cpu_crown_dense_budget_from_terms(observed),
                    observed.headroom_bytes,
                );
                let before_probes = MEMORY_TERM_PROBES.load(Ordering::Relaxed);
                let repeats: Vec<usize> = (0..PROBES_PER_WINDOW)
                    .map(|_| cpu_crown_dense_budget_bytes())
                    .collect();
                // A hit: decoded from the lock-free cache.
                let hit = amortised_memory_terms();
                let after_observations = MEMORY_TERM_OBSERVATIONS.load(Ordering::Relaxed);
                let after_probes = MEMORY_TERM_PROBES.load(Ordering::Relaxed);
                let elapsed = window.elapsed();

                last_failure = if elapsed >= MEMORY_TERMS_AMORTISE_TTL / 2 {
                    format!("attempt {attempt}: window took {elapsed:?}, too close to the TTL")
                } else if after_first <= before_observations {
                    format!(
                        "attempt {attempt}: invalidation did not force a kernel observation \
                         ({before_observations} -> {after_first})"
                    )
                } else if after_observations != after_first {
                    format!(
                        "attempt {attempt}: {PROBES_PER_WINDOW} probes inside the TTL reached \
                         the kernel {} more time(s)",
                        after_observations - after_first
                    )
                } else if after_probes < before_probes + PROBES_PER_WINDOW + 1 {
                    format!(
                        "attempt {attempt}: only {} of {} probes were counted",
                        after_probes - before_probes,
                        PROBES_PER_WINDOW + 1
                    )
                } else if hit != observed {
                    format!(
                        "attempt {attempt}: a cache hit decoded {hit:?}, not the observation \
                         {observed:?}"
                    )
                } else if let Some(different) = repeats.iter().find(|&&budget| budget != first) {
                    format!(
                        "attempt {attempt}: an amortised probe returned {different}, not the \
                         {first} derived from the observation"
                    )
                } else {
                    return;
                };
            }
            panic!("no clean amortisation window in {ATTEMPTS} attempts; last: {last_failure}");
        });
    }

    #[test]
    fn amortisation_line_carries_both_halves_unconditionally() {
        let line = amortisation_line();
        for field in [
            "[poll-amortise]",
            "memory_probes=",
            "memory_observations=",
            "memory_factor=",
            "memory_ttl_ms=100",
            "procfs_reads_saved_estimate=",
            "poll_units=",
            "poll_chunk=4096",
        ] {
            assert!(line.contains(field), "missing {field} in {line}");
        }
    }

    #[test]
    fn memory_term_sentinel_round_trips() {
        assert_eq!(decode_memory_term(encode_memory_term(None)), None);
        assert_eq!(decode_memory_term(encode_memory_term(Some(0))), Some(0));
        assert_eq!(
            decode_memory_term(encode_memory_term(Some(1 << 40))),
            Some(1 << 40)
        );
        assert!(
            decode_memory_term(encode_memory_term(Some(MEMORY_TERM_NONE))).is_some(),
            "a saturated byte count must not decode as an absent observation"
        );
    }
}

/// Size the Dense budget from MEASURED host memory instead of a fixed constant
/// (#adaptive-dense-budget).
///
/// The 2 GiB default was a hardcoded number standing in for a measurement, and
/// it is the budget that actually does the refusing: every CROWN Dense
/// materialization gate reads this value, and on a 24 GiB host every refusal
/// logged `budget is 2147483648 bytes` no matter what else was configured.
/// Measured on yolo_2023 (2026-07-29), raising it to 12 GiB improved the only
/// output row that can decide the instance by 5.45x (Y[269] -42784.91 ->
/// -7847.13) with host memory pressure never leaving 80% free -- an improvement
/// no operator would ever have reached, because reaching it required knowing to
/// set an environment variable.
///
/// Three bounds still apply, in this order:
///   * the FLOOR is the historical 2 GiB default, so no host can regress;
///   * the SHARE is half of an already-conservative observation;
///   * `process_memory_headroom_bytes` clamps the result to one eighth of live
///     headroom -- exceeding it is a real OOM, not a policy choice. Its cgroup
///     and RLIMIT terms are KERNEL-ENFORCED; since #mem-governor it also
///     carries a self-governing `MemAvailable`/`VmRSS` term, for the
///     competition configuration in which neither kernel limit exists. The
///     public entry point reapplies that clamp to explicit operator budgets as
///     well.
///
/// With neither an observable host-memory value nor a process-envelope value,
/// this returns exactly the historical default.
fn adaptive_cpu_crown_dense_budget_from_terms(terms: AmortisedMemoryTerms) -> usize {
    let budget = adaptive_cpu_crown_dense_budget_from(
        terms.observed_host_bytes,
        terms.headroom_bytes,
        terms.live_available_bytes,
    );
    #[cfg(not(target_os = "linux"))]
    {
        // #dense-budget-valve: REPORT a critically low host once per process.
        // This deliberately does not change `budget` -- see the policy function's
        // doc comment for why the clamp was withdrawn.
        if let Some(available) = terms.live_available_bytes {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if available < LOW_MEMORY_REPORT_THRESHOLD_BYTES
                && !WARNED.swap(true, Ordering::Relaxed)
            {
                tracing::warn!(
                    "#dense-budget-valve: host has {available} bytes reclaimable-now, below \
                     {LOW_MEMORY_REPORT_THRESHOLD_BYTES}. The CROWN dense budget is UNCHANGED \
                     at {budget} -- this is a diagnostic, not a throttle. Concurrent heavy runs \
                     on this host are the usual cause. (Logged once per process.)"
                );
            }
        }
    }
    budget
}

/// Pure adaptive-budget policy, separated from the host probes so the
/// kernel-envelope precedence is regression-testable on every platform.
///
/// # `live_available_bytes` -- the low-memory safety valve (#dense-budget-valve)
///
/// `observed_host_bytes` is a STATIC figure on every non-Linux platform: it is
/// `hw.memsize / 2`, i.e. installed RAM, which reads the same whether the
/// machine has 20 GiB free or 60 MiB free. The kernel-envelope clamp that the
/// comment below calls the backstop is `None` on those platforms (there is no
/// cgroup), so before this valve existed macOS had NO term in this function
/// that could respond to the machine actually being full.
///
/// Measured, 2026-07-31 on a 24 GiB M4 Pro: one `ny` run of
/// `yolo_2023/TinyYOLO_prop_000001_eps_1_255` peaks at 14.17 GiB RSS and drives
/// host free memory to 65 MiB -- twice -- while this function keeps returning
/// 6 GiB. On 2026-07-30 that same shape, several copies of it running
/// concurrently in agent worktrees, took the machine into a jetsam kill
/// (`JetsamEvent-2026-07-30-210002`: `ny` at 44 GiB RSS, 4668 pages free).
///
/// # WITHDRAWN 2026-08-01: the valve no longer clamps, and here is why
///
/// This function briefly clamped the budget to 256 MiB when reclaimable-now fell
/// below 1.5 GiB. That was **wrong and is removed**. Two independent reasons,
/// both measured:
///
/// 1. **No benefit.** The premise was that the dense budget governs the
///    process footprint. It does not: cutting `NY_DENSE_BUDGET_MB` 4x moves peak
///    RSS by 5%, and cutting `NY_GPU_MEMORY_BUDGET_MB` 16x moves it 5%. The
///    ~14 GiB is a genuine in-use working set allocated through mimalloc arenas.
///    Clamping this number cannot prevent the OOM it was written to prevent.
/// 2. **Real harm.** The "engages only in a danger band ordinary operation does
///    not enter" claim was verified in unit tests and never end-to-end, and it
///    is false. `ny` is itself the process driving reclaimable memory down, so
///    on a 24 GiB host the valve fires *against the very run it is protecting*,
///    mid-proof. Observed on `yolo_2023` 2026-08-01: it clamped
///    6442450944 -> 268435456, after which Conv2d targets kept their IBP bound,
///    the f64 recompute degraded rows to +/-inf, and the instance returned
///    `unknown`.
///
/// A guard with no demonstrated benefit and a demonstrated cost is not something
/// to re-tune, so the threshold is gone rather than lowered. What survives is
/// worth keeping on its own: `hw.memsize` is cached instead of forking `sysctl`
/// per admission, and a critically low host is reported once per process.
///
/// The lesson worth keeping: a memory guard inside the process that consumes the
/// memory needs its trigger measured END TO END, not in a unit test where the
/// process is not running.
///
/// `live_available_bytes` must be a RECLAIMABLE-NOW figure (free +
/// speculative + purgeable), NOT a Linux-style `MemAvailable` that counts
/// inactive pages. Checked against the jetsam report: at the instant of the
/// kill the MemAvailable-style sum still read 4.29 GiB -- it would not have
/// noticed the machine dying -- while reclaimable-now read 79 MiB.
///
/// An explicit `NY_DENSE_BUDGET_MB` bypasses the host-adaptive heuristic (an
/// operator stating a number is making an assertion about their machine), but
/// the public entry point still clamps it by the kernel-enforced process
/// envelope.
///
/// This is an operational guard, not a soundness mechanism: it changes only how
/// much memory a materialization may claim, never what is proved. A refused
/// materialization takes the existing documented fallback path.
fn adaptive_cpu_crown_dense_budget_from(
    observed_host_bytes: Option<u64>,
    process_headroom_bytes: Option<u64>,
    live_available_bytes: Option<u64>,
) -> usize {
    let floor = DEFAULT_CROWN_DENSE_BUDGET_MB.saturating_mul(1024 * 1024);
    let ceiling = MAX_ADAPTIVE_DENSE_BUDGET_MB.saturating_mul(1024 * 1024);
    let mut budget = observed_host_bytes.map_or(floor, |observed| {
        let share = observed / ADAPTIVE_DENSE_BUDGET_SHARE_DIVISOR;
        usize::try_from(share)
            .unwrap_or(usize::MAX)
            .clamp(floor, ceiling)
    });
    // Kernel-enforced envelope wins over the host-wide observation, and over the
    // floor: if the cgroup says there is less than 2 GiB left, taking 2 GiB is
    // not a conservative default, it is an OOM.
    if let Some(headroom) = process_headroom_bytes {
        let headroom_share =
            usize::try_from(headroom / PROCESS_ENVELOPE_DENSE_BUDGET_SHARE_DIVISOR)
                .unwrap_or(usize::MAX);
        budget = budget.min(headroom_share);
    }
    // #dense-budget-valve WITHDRAWN 2026-08-01 -- see the doc comment above.
    // The live reading is still taken and still reported once per process,
    // because "the host is nearly out of memory" is worth saying. It no longer
    // CHANGES the budget.
    let _ = live_available_bytes;
    budget
}

#[cfg(test)]
mod adaptive_budget_tests {
    use super::*;

    #[test]
    fn missing_host_probe_still_honors_process_headroom() {
        let mib = 1024_u64 * 1024;
        assert_eq!(
            adaptive_cpu_crown_dense_budget_from(None, Some(512 * mib), None),
            64 * 1024 * 1024,
            "a missing host-wide probe must not bypass a kernel-enforced envelope"
        );
    }

    #[test]
    fn missing_host_and_process_probes_preserve_historical_floor() {
        assert_eq!(
            adaptive_cpu_crown_dense_budget_from(None, None, None),
            DEFAULT_CROWN_DENSE_BUDGET_MB * 1024 * 1024
        );
    }

    #[test]
    fn process_headroom_can_tighten_an_observed_host_budget_below_the_floor() {
        let gib = 1024_u64 * 1024 * 1024;
        assert_eq!(
            adaptive_cpu_crown_dense_budget_from(Some(128 * gib), Some(gib), None),
            128 * 1024 * 1024
        );
    }

    /// #dense-budget-valve WITHDRAWN: the live reading must never change the
    /// budget. Asserted across the whole range, including the reading recorded
    /// at the 2026-07-30 jetsam and a zero reading, because the withdrawn
    /// version clamped hardest exactly there -- and clamping there is what
    /// degraded a live proof to `unknown`.
    #[test]
    fn live_availability_never_changes_the_budget() {
        let gib = 1024_u64 * 1024 * 1024;
        let at_jetsam = (4668_u64 + 541) * 16384; // 79 MiB reclaimable-now
        let baseline = adaptive_cpu_crown_dense_budget_from(Some(12 * gib), None, None);
        assert_eq!(baseline, 6 * 1024 * 1024 * 1024);
        for reclaimable in [0, at_jetsam, 512 * 1024 * 1024, 2 * gib, 20 * gib] {
            assert_eq!(
                adaptive_cpu_crown_dense_budget_from(Some(12 * gib), None, Some(reclaimable)),
                baseline,
                "reclaimable={reclaimable}: the probe is diagnostic only and must not \
                 move the budget"
            );
        }
    }

    /// The kernel-enforced envelope is a different thing and still binds: it is
    /// a per-process limit the kernel will actually enforce, not a host-wide
    /// observation.
    #[test]
    fn kernel_envelope_still_binds_after_the_valve_withdrawal() {
        let gib = 1024_u64 * 1024 * 1024;
        assert_eq!(
            adaptive_cpu_crown_dense_budget_from(Some(12 * gib), Some(64 * 1024 * 1024), Some(0)),
            8 * 1024 * 1024,
            "a cgroup/RLIMIT envelope must survive, valve or no valve"
        );
    }

    #[test]
    fn explicit_budget_cannot_bypass_kernel_envelope() {
        let gib = 1024_u64 * 1024 * 1024;
        assert_eq!(
            clamp_dense_budget_to_process_envelope(12 * gib as usize, Some(4 * gib)),
            512 * 1024 * 1024
        );
        assert_eq!(
            clamp_dense_budget_to_process_envelope(64 * 1024 * 1024, Some(4 * gib)),
            64 * 1024 * 1024,
            "the envelope narrows an override but never inflates it"
        );
        assert_eq!(
            clamp_dense_budget_to_process_envelope(usize::MAX, Some(0)),
            0
        );
    }
}

/// #mem-governor: the self-governing memory term that competition actually
/// relies on, because `run_instance.sh` installs no cgroup and no RLIMIT.
///
/// These assert END-TO-END SHAPES, not just arithmetic. The withdrawn
/// `#dense-budget-valve` was "verified in unit tests and never end to end" and
/// its trigger turned out to fire against the very run it protected, so each
/// case below is a host state that was actually observed, named, and dated.
#[cfg(all(test, target_os = "linux"))]
mod memory_governor_tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;
    /// This GB10 host, `/proc/meminfo` 2026-08-26: 127_600_748 kB.
    const HOST_TOTAL: u64 = 127_600_748 * 1024;
    /// Same reading, quiescent: 120_036_632 kB.
    const HOST_AVAILABLE_QUIESCENT: u64 = 120_036_632 * 1024;

    fn dense_budget_under(headroom: Option<u64>) -> usize {
        // The adaptive path on this host: `MemAvailable / 2` clamped into
        // [2 GiB, 12 GiB], then the envelope share.
        adaptive_cpu_crown_dense_budget_from(Some(HOST_AVAILABLE_QUIESCENT), headroom, None)
    }

    /// The governor must change NOTHING on a healthy machine. If it did, every
    /// banked verdict on this box would be in question and the proof cost would
    /// be paid on rows that were never at risk.
    #[test]
    fn governor_is_inert_on_a_quiescent_host() {
        // Production cgan peak, measured 2026-08-26: 6.3 GB resident.
        let headroom = system_memory_headroom_from(
            Some(HOST_AVAILABLE_QUIESCENT),
            Some(HOST_TOTAL),
            Some(6_300_000_000),
        );
        let headroom = headroom.expect("procfs observations present");
        assert!(
            headroom / PROCESS_ENVELOPE_DENSE_BUDGET_SHARE_DIVISOR
                >= MAX_ADAPTIVE_DENSE_BUDGET_MB as u64 * 1024 * 1024,
            "quiescent headroom share {} must stay above the 12 GiB adaptive ceiling",
            headroom / PROCESS_ENVELOPE_DENSE_BUDGET_SHARE_DIVISOR
        );
        assert_eq!(
            dense_budget_under(Some(headroom)),
            MAX_ADAPTIVE_DENSE_BUDGET_MB * 1024 * 1024,
            "the governor must not move the dense budget on a healthy host"
        );
    }

    /// The shape that took the global OOM: `ny_pin_base` at anon-rss 118 GB on
    /// this 121 GiB host, 2026-08-26. By then `MemAvailable` is nearly gone, so
    /// the availability term alone already refuses everything.
    #[test]
    fn governor_binds_at_the_measured_oom_shape() {
        let headroom =
            system_memory_headroom_from(Some(2 * GIB), Some(HOST_TOTAL), Some(118_000_000_000));
        assert_eq!(
            headroom,
            Some(0),
            "with 2 GiB available and 118 GB resident there is no headroom to hand out"
        );
        assert_eq!(
            dense_budget_under(headroom),
            0,
            "a zero envelope must beat the 2 GiB floor -- taking 2 GiB here is the OOM"
        );
    }

    /// The case the availability term CANNOT see, and the reason the backstop
    /// exists: `MemAvailable` counts reclaimable page cache, so a process
    /// holding most of the machine can still read tens of GiB "available" and
    /// keep admitting multi-GiB materializations right up to the kill.
    #[test]
    fn self_rss_backstop_fires_while_mem_available_still_looks_healthy() {
        let available = 20 * GIB;
        let resident = 100 * GIB;
        assert!(
            available > SYSTEM_MEMORY_RESERVE_BYTES,
            "precondition: the availability term is NOT what binds here"
        );
        assert_eq!(
            system_memory_headroom_from(Some(available), Some(HOST_TOTAL), None),
            Some(available - SYSTEM_MEMORY_RESERVE_BYTES),
            "without a VmRSS reading this host looks fine, which is the defect"
        );
        assert_eq!(
            system_memory_headroom_from(Some(available), Some(HOST_TOTAL), Some(resident)),
            Some(0),
            "the backstop must force the cheap path once NY owns 3/4 of RAM"
        );
        assert_eq!(
            memory_governor_source(MemoryGovernorReport {
                cgroup: None,
                address_space: None,
                system: Some(0),
                headroom: Some(0),
                available: Some(available),
                total: Some(HOST_TOTAL),
                self_rss: Some(resident),
            }),
            "self-rss-backstop",
            "the observable must name the term that is actually holding the line"
        );
    }

    /// The largest legitimate `ny` resident set on record: 14.17 GiB on a
    /// 24 GiB host (yolo_2023, 2026-07-31), 59% of RAM. A one-half ceiling
    /// would have fired against it; three quarters must not.
    #[test]
    fn self_rss_backstop_clears_the_largest_legitimate_run() {
        let total = 24 * GIB;
        let resident = 15_214_921_646; // 14.17 GiB, 59% of a 24 GiB host
        assert_eq!(
            system_memory_headroom_from(Some(8 * GIB), Some(total), Some(resident)),
            Some(8 * GIB - SYSTEM_MEMORY_RESERVE_BYTES),
            "the backstop must not contribute a bound below its ceiling"
        );
    }

    /// LIVE, not one-shot: headroom must fall monotonically as the process
    /// grows and the host drains. A governor that samples once is a ceiling.
    #[test]
    fn governor_headroom_decays_as_the_process_grows() {
        let mut previous = u64::MAX;
        for step in 0..12_u64 {
            let resident = step * 10 * GIB;
            let available = HOST_AVAILABLE_QUIESCENT.saturating_sub(resident);
            let headroom =
                system_memory_headroom_from(Some(available), Some(HOST_TOTAL), Some(resident))
                    .expect("observations present");
            assert!(
                headroom <= previous,
                "headroom rose from {previous} to {headroom} at resident={resident}"
            );
            previous = headroom;
        }
        assert_eq!(previous, 0, "the trajectory must reach zero, not a floor");
    }

    /// Missing observations must contribute NO bound, exactly as an unreadable
    /// cgroup does. A `None` here must never be read as zero.
    #[test]
    fn absent_observations_contribute_no_bound() {
        assert_eq!(system_memory_headroom_from(None, None, None), None);
        assert_eq!(
            system_memory_headroom_from(None, Some(HOST_TOTAL), Some(1)),
            None,
            "a resident set below the ceiling with no availability reading bounds nothing"
        );
    }

    /// The governor is a peer term in a `min`. It must be able to tighten a
    /// kernel envelope and must never loosen one -- a looser envelope is the
    /// only way this could ever admit something it previously refused.
    #[test]
    fn governor_narrows_but_never_widens_a_kernel_envelope() {
        let system = system_memory_headroom_from(Some(6 * GIB), Some(HOST_TOTAL), None);
        assert_eq!(system, Some(2 * GIB));
        assert_eq!(
            narrowest_headroom(&[Some(64 * 1024 * 1024), None, system]),
            Some(64 * 1024 * 1024),
            "a tighter cgroup must still win"
        );
        assert_eq!(
            narrowest_headroom(&[Some(100 * GIB), None, system]),
            Some(2 * GIB),
            "and the governor must win when it is the tighter one"
        );
        for headroom in [Some(0), Some(GIB), Some(100 * GIB), None] {
            let with_governor = narrowest_headroom(&[headroom, None, system]);
            let without = narrowest_headroom(&[headroom, None, None]);
            let widened = match (with_governor, without) {
                (Some(new), Some(old)) => new > old,
                (Some(_), None) => false,
                (None, Some(_)) => true,
                (None, None) => false,
            };
            assert!(
                !widened,
                "adding the governor widened {without:?} to {with_governor:?}"
            );
        }
    }

    /// The behavioural change this whole task exists for: with no cgroup and no
    /// RLIMIT -- the competition configuration -- the probe used to report
    /// `None`, so the clamp every Dense gate consults was inert. It must now
    /// report a live bound on any Linux host.
    #[test]
    fn linux_headroom_is_no_longer_none_without_a_kernel_limit() {
        assert!(
            process_memory_headroom_bytes().is_some(),
            "the self-governing term must supply a bound when no kernel limit does"
        );
    }

    #[test]
    fn parses_vm_rss_and_mem_total() {
        let status = "Name:\tny\nVmSize:\t 8388608 kB\nVmRSS:\t 4194304 kB\nThreads:\t20\n";
        assert_eq!(parse_vm_rss_bytes(status), Some(4_194_304 * 1024));
        assert_eq!(parse_vm_size_bytes(status), Some(8_388_608 * 1024));
        assert_eq!(parse_vm_rss_bytes("VmSize:\t 8388608 kB\n"), None);
        assert_eq!(parse_vm_rss_bytes("VmRSS:\t notanumber kB\n"), None);
        assert_eq!(
            parse_vm_rss_bytes("VmRSS:\t 4194304 pages\n"),
            None,
            "a unit change must read as no observation, not as kiB"
        );

        let meminfo = "MemTotal:       127600748 kB\nMemFree:        103307692 kB\n\
                       MemAvailable:   120036632 kB\n";
        assert_eq!(
            parse_meminfo_field_bytes(meminfo, "MemTotal:"),
            Some(HOST_TOTAL)
        );
        assert_eq!(
            parse_meminfo_available_kib(meminfo),
            Some(HOST_AVAILABLE_QUIESCENT)
        );
        assert_eq!(parse_meminfo_field_bytes(meminfo, "MemAbsent:"), None);
    }
}

#[cfg(all(test, not(target_os = "linux")))]
mod vm_stat_parse_tests {
    use super::*;

    /// Real `vm_stat` output from the 24 GiB M4 Pro this defect was measured on.
    const SAMPLE: &str = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
Pages free:                                   184160.\n\
Pages active:                                 614765.\n\
Pages inactive:                               597542.\n\
Pages speculative:                             16310.\n\
Pages throttled:                                   0.\n\
Pages wired down:                             115640.\n\
Pages purgeable:                                3439.\n\
\"Translation faults\":                        7729185.\n";

    #[test]
    fn parses_reclaimable_pages_at_the_reported_page_size() {
        // free + purgeable + speculative, at 16 KiB pages.
        let expected = (184_160_u64 + 3_439 + 16_310) * 16384;
        assert_eq!(parse_vm_stat_available_bytes(SAMPLE), Some(expected));
    }

    /// The whole point of the metric: inactive pages must NOT be counted.
    /// Including them is what makes a dying host look healthy -- see the
    /// jetsam arithmetic in `adaptive_cpu_crown_dense_budget_from`.
    #[test]
    fn excludes_inactive_wired_and_active_pages() {
        let reclaimable = parse_vm_stat_available_bytes(SAMPLE).unwrap();
        let with_inactive = (184_160_u64 + 3_439 + 16_310 + 597_542) * 16384;
        assert!(
            reclaimable < with_inactive,
            "inactive pages must be excluded: {reclaimable} vs {with_inactive}"
        );
        assert_eq!(reclaimable + 597_542 * 16384, with_inactive);
    }

    #[test]
    fn unparseable_output_is_no_observation_not_zero() {
        assert_eq!(parse_vm_stat_available_bytes(""), None);
        assert_eq!(
            parse_vm_stat_available_bytes("Mach Virtual Memory Statistics: (page size of 0 bytes)"),
            None
        );
        assert_eq!(
            parse_vm_stat_available_bytes(
                "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nSomething else:  1.\n"
            ),
            None,
            "a wholesale format change must read as no observation"
        );
    }

    /// A future OS dropping one field must degrade to the fields that remain,
    /// not to `None` and not to a silent zero.
    #[test]
    fn missing_single_field_degrades_gracefully() {
        let partial = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
Pages free:                                   100.\n";
        assert_eq!(parse_vm_stat_available_bytes(partial), Some(100 * 16384));
    }
}

/// Host memory available to this process, in bytes, or `None` when no probe
/// exists on this platform (#adaptive-dense-budget).
///
/// Linux exposes a per-moment figure via `/proc/meminfo` `MemAvailable`. Other
/// platforms fall back to a conservative FRACTION of measured physical memory --
/// half, deliberately pessimistic since the rest of the machine may be busy --
/// which is still a measurement and strictly better than a fixed constant.
///
/// Dense admission sits in per-domain hot loops, where reopening procfs and
/// rediscovering the same cgroup mount thousands of times can cost more than
/// the proof itself. Linux keeps a short-lived snapshot of open handles and
/// reads `MemAvailable` from that handle at every OBSERVATION, without repeated
/// `openat`/`statx`/`close` traffic. Since #poll-amortise the dense-budget
/// gates share one observation for at most `MEMORY_TERMS_AMORTISE_TTL`.
#[cfg(not(target_os = "linux"))]
fn observed_host_memory_bytes() -> Option<u64> {
    physical_memory_bytes().map(|total| total / 2)
}

/// Parse `MemAvailable:` (reported in kiB) out of `/proc/meminfo` contents.
///
/// Pure so it can be tested without the file. Returns `None` when the field is
/// absent or unparseable, which the caller treats as "no observation" rather
/// than as zero -- a zero would silently clamp the budget to the floor.
#[cfg(target_os = "linux")]
fn parse_meminfo_available_kib(contents: &str) -> Option<u64> {
    parse_meminfo_field_bytes(contents, "MemAvailable:")
}

/// Parse a named `/proc/meminfo` field, reported in kiB, into bytes.
///
/// Returns `None` when the field is absent or unparseable, which callers treat
/// as "no observation" rather than as zero -- a zero would silently clamp the
/// budget to nothing on a host whose procfs format merely changed.
#[cfg(target_os = "linux")]
fn parse_meminfo_field_bytes(contents: &str, field: &str) -> Option<u64> {
    contents
        .lines()
        .find_map(|line| line.strip_prefix(field))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .and_then(|kib| kib.checked_mul(1024))
        .filter(|bytes| *bytes > 0)
}

/// Parse `VmRSS:` (reported in kiB) out of `/proc/self/status` contents.
///
/// RESIDENT set, not `VmSize`: the OOM killer charges resident anonymous pages,
/// and NY routinely reserves address space it never touches. Returns `None`
/// when absent or unparseable so an unreadable field contributes no bound
/// instead of a fabricated one.
#[cfg(target_os = "linux")]
fn parse_vm_rss_bytes(contents: &str) -> Option<u64> {
    let rest = contents
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))?;
    let mut fields = rest.split_whitespace();
    let kib: u64 = fields.next()?.parse().ok()?;
    if fields.next()? != "kB" {
        return None;
    }
    kib.checked_mul(1024)
}

/// Total physical memory, where the platform exposes it cheaply.
///
/// Cached: `hw.memsize` is immutable for the life of the process, and this used
/// to spawn a `sysctl` child on every dense-admission check (#dense-budget-valve).
#[cfg(not(target_os = "linux"))]
fn physical_memory_bytes() -> Option<u64> {
    static PHYSICAL: OnceLock<Option<u64>> = OnceLock::new();
    *PHYSICAL.get_or_init(|| {
        if !cfg!(any(
            target_os = "macos",
            target_os = "ios",
            target_vendor = "apple"
        )) {
            return None;
        }
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let value = String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse::<u64>()
            .ok()?;
        (value > 0).then_some(value)
    })
}

/// Live host-wide RECLAIMABLE-NOW memory in bytes, or `None` where no probe
/// exists (#dense-budget-valve).
///
/// Deliberately NOT the macOS analogue of Linux's `MemAvailable`. INACTIVE pages
/// are excluded: under compressor pressure macOS does not hand them back
/// promptly, and counting them hides exactly the condition this probe exists to
/// detect. Checked against `JetsamEvent-2026-07-30-210002`, which recorded the
/// page counts at the instant of the kill: including inactive reads 4.29 GiB
/// (the machine looks fine while it is being killed), excluding it reads
/// 79 MiB. Wired and active pages are excluded for the obvious reason.
///
/// Read from `vm_stat`, which reports every field in one shot and states its own
/// page size, so the reading cannot silently desynchronize from a host whose
/// page size is not 4 KiB (Apple silicon reports 16384).
#[cfg(not(target_os = "linux"))]
fn live_available_memory_bytes() -> Option<u64> {
    if !cfg!(any(
        target_os = "macos",
        target_os = "ios",
        target_vendor = "apple"
    )) {
        return None;
    }
    static CACHE: OnceLock<Mutex<Option<(Instant, Option<u64>)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    // A poisoned lock must not take down an admission check: fall back to
    // probing directly rather than propagating the panic.
    let Ok(mut slot) = cache.lock() else {
        return parse_vm_stat_available_bytes(&run_vm_stat()?);
    };
    if let Some((at, value)) = *slot {
        if at.elapsed() < LIVE_MEMORY_PROBE_TTL {
            return value;
        }
    }
    let value = run_vm_stat().and_then(|out| parse_vm_stat_available_bytes(&out));
    *slot = Some((Instant::now(), value));
    value
}

#[cfg(not(target_os = "linux"))]
fn run_vm_stat() -> Option<String> {
    let out = std::process::Command::new("vm_stat").output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Parse `vm_stat` output into available bytes.
///
/// Pure so the arithmetic is testable without the binary. Returns `None` when
/// the page size or every availability field is missing -- treated by the caller
/// as "no observation", never as zero, since a zero would clamp the budget to
/// the valve minimum on every call.
#[cfg(not(target_os = "linux"))]
fn parse_vm_stat_available_bytes(contents: &str) -> Option<u64> {
    let page_size = contents
        .lines()
        .next()
        .and_then(|line| line.rsplit_once("page size of "))
        .and_then(|(_, rest)| rest.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|size| *size > 0)?;

    let field = |name: &str| -> Option<u64> {
        contents
            .lines()
            .find_map(|line| line.trim().strip_prefix(name))
            .and_then(|rest| rest.trim().strip_prefix(':'))
            .and_then(|rest| rest.trim().trim_end_matches('.').parse::<u64>().ok())
    };

    // Any single field may be absent on a future OS revision; require at least
    // one to have parsed so a wholesale format change reads as "no observation".
    // `Pages inactive` is intentionally NOT summed here -- see the doc comment.
    let parts = [
        field("Pages free"),
        field("Pages purgeable"),
        field("Pages speculative"),
    ];
    if parts.iter().all(Option::is_none) {
        return None;
    }
    let pages: u64 = parts.into_iter().flatten().sum();
    pages.checked_mul(page_size)
}

/// Return the operator-configured Dense budget, distinguishing it from the
/// general 2 GiB fallback used by legacy materialization gates.
///
/// Conv CROWN uses this distinction because its independent RAM-adaptive
/// policy was introduced specifically to admit safe multi-GiB result buffers
/// on large hosts. An explicitly set `NY_DENSE_BUDGET_MB` remains authoritative
/// over host heuristics, while callers still enforce cgroup/RLIMIT headroom; an
/// invalid or non-Unicode value preserves the established 2 GiB fallback.
pub(crate) fn explicit_cpu_crown_dense_budget_bytes() -> Option<usize> {
    let raw = std::env::var_os(CROWN_DENSE_BUDGET_ENV)?;
    let parsed = raw
        .to_str()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_CROWN_DENSE_BUDGET_MB);
    Some(parsed.saturating_mul(1024 * 1024))
}

/// Remaining memory in the narrowest live envelope NY can observe without
/// adding a new configuration claim.
///
/// Linux exposes the two KERNEL-ENFORCED envelopes used by NY's guarded
/// runners:
///
/// - cgroup v2 `memory.max` (or v1 `memory.limit_in_bytes`), including tighter
///   ancestor limits and the corresponding hierarchical current usage;
/// - the process soft `RLIMIT_AS`, installed by `ny-safe-gpu-run` via
///   `ulimit -v`, less the process's current `VmSize`.
///
/// ...and, since #mem-governor, a third SELF-GOVERNING term for the case those
/// two do not cover, which is the case competition actually runs in:
///
/// - [`system_memory_headroom_from`], derived from `/proc/meminfo`
///   `MemAvailable` and this process's own `/proc/self/status` `VmRSS`.
///
/// The third term is a peer of the other two rather than a fallback used only
/// when they are absent. A global OOM kill is exactly as fatal as a cgroup one,
/// so a run that is inside its cgroup but outside what the machine can supply
/// must still degrade. It is deliberately sized to be INERT on a quiescent host
/// (see [`SYSTEM_MEMORY_RESERVE_BYTES`]).
///
/// The minimum remaining headroom is returned. An unreadable, malformed, or
/// unlimited source contributes no bound; callers must retain their own
/// conservative fallback. Every resolved cgroup limit/usage pair, both RLIMIT
/// inputs, and both self-governing inputs are read through cached open handles
/// at every OBSERVATION, and since #poll-amortise one observation is shared by
/// every gate for at most `MEMORY_TERMS_AMORTISE_TTL` (100 ms) -- so the value
/// returned here may be up to that old. Path discovery and handle opening use
/// a separate short, invalidating snapshot, so `openat`/`statx`/`close` never
/// become the dominant workload. Callers that must see a kernel envelope LIVE
/// use [`process_memory_envelope`], which bypasses the amortised cache.
///
/// This narrows; it never widens. See [`system_memory_headroom_from`] for why
/// that makes the governor incapable of tightening any bound.
///
/// [`ProcessMemoryEnvelope`] is deliberately NOT affected: that API's contract
/// is strictly about envelopes the kernel will enforce, and callers use it to
/// fail CLOSED on ambiguity. On Linux a host-derived observation has no place
/// in it — `MemAvailable` is advisory there, and the enforced envelopes are
/// cgroup/RLIMIT. The Darwin arm of that API (#darwin-memory-envelope) is not
/// an exception to this rule: `hw.memsize − phys_footprint` is the envelope
/// XNU's jetsam ledger enforces on a machine whose unified physical memory is
/// fixed, not an advisory availability heuristic.
pub fn process_memory_headroom_bytes() -> Option<u64> {
    amortised_memory_terms().headroom_bytes
}

/// One live observation of the memory terms, WITHOUT the amortised cache.
fn observe_memory_terms() -> AmortisedMemoryTerms {
    #[cfg(target_os = "linux")]
    {
        let snapshot = linux_memory_probe_snapshot();
        AmortisedMemoryTerms {
            observed_host_bytes: observed_host_memory_bytes_from_snapshot(&snapshot),
            headroom_bytes: process_memory_headroom_from_snapshot(&snapshot),
            // Linux already reads `MemAvailable` as its host observation, so
            // the observation IS the live figure; no separate valve is needed.
            live_available_bytes: None,
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        AmortisedMemoryTerms {
            observed_host_bytes: observed_host_memory_bytes(),
            headroom_bytes: None,
            live_available_bytes: live_available_memory_bytes(),
        }
    }
}

/// Strict classification of the live kernel-enforced memory envelope.
///
/// Unlike [`process_memory_headroom_bytes`], this API preserves the difference
/// between a process that is conclusively unbounded and one whose cgroup or
/// RLIMIT state could not be observed completely. Optional high-memory work
/// can therefore fail closed without changing the established Dense/Conv
/// budgeting semantics of the legacy `Option<u64>` probe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessMemoryEnvelope {
    /// At least one kernel limit is finite; this is the narrowest live
    /// headroom after accounting for the corresponding usage counters.
    Bounded { headroom_bytes: u64 },
    /// Every applicable kernel source was read successfully and was unlimited.
    Unbounded,
    /// At least one applicable source could not be discovered or read strictly.
    Unavailable,
}

/// Return a strict classification of the process's kernel-enforced envelope.
///
/// Linux requires complete, internally consistent observations from both
/// cgroups and `RLIMIT_AS`. Any ambiguous source makes the aggregate
/// unavailable.
///
/// macOS (#darwin-memory-envelope) derives the same classification from the
/// machine's fixed physical memory (`sysctl hw.memsize`) minus this process's
/// kernel-charged footprint (`task_info` `TASK_VM_INFO` `phys_footprint` — the
/// ledger jetsam enforces), narrowed by a finite `RLIMIT_AS` soft limit when
/// one is installed, mirroring the Linux RLIMIT term. Any failed syscall makes
/// the aggregate [`ProcessMemoryEnvelope::Unavailable`], so the fail-closed
/// contract for genuinely unknown envelopes is preserved; what changed is that
/// macOS stopped being "unknown BY CONSTRUCTION" — before this arm existed, no
/// measurement on a macOS host could ever exercise the optional-heavy lanes
/// behind this gate (the margin-row lane's flight sidecar showed
/// `outer_route=memory` on every cifar100 run at every commit).
///
/// Operating systems with no equivalent probe still report
/// [`ProcessMemoryEnvelope::Unavailable`].
pub fn process_memory_envelope() -> ProcessMemoryEnvelope {
    #[cfg(target_os = "linux")]
    {
        let snapshot = linux_memory_probe_snapshot();
        process_memory_envelope_from_snapshot(&snapshot)
    }
    #[cfg(target_os = "macos")]
    {
        let snapshot = darwin_memory_probe_snapshot();
        process_memory_envelope_from_darwin_snapshot(&snapshot)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        ProcessMemoryEnvelope::Unavailable
    }
}

#[cfg(all(test, not(any(target_os = "linux", target_os = "macos"))))]
#[test]
fn process_memory_envelope_is_unavailable_without_a_platform_probe() {
    assert_eq!(
        process_memory_envelope(),
        ProcessMemoryEnvelope::Unavailable
    );
}

// === Darwin strict memory envelope (#darwin-memory-envelope) ===
//
// macOS has no cgroup, so before this section existed `process_memory_envelope`
// returned `Unavailable` on every macOS host BY CONSTRUCTION, and the
// fail-closed consumers (`optional_heavy_memory_allowed_now`, the
// override-rebound admission) were structurally dark: no measurement taken on
// a macOS machine had ever exercised the lanes behind them. This section makes
// the gate MEASURE instead of failing closed per-platform, with the same
// strictness law as Linux: every applicable source must be observed
// completely, and any failed syscall is `Unavailable`.
//
// The probe deliberately uses direct libSystem calls declared here rather than
// a new dependency: ny-propagate does not link `libc` or `mach2`
// (Cargo.toml has neither), libSystem is always linked on Darwin, and the
// three calls needed (`sysctlbyname`, `task_info`, `getrlimit`) have stable
// documented ABIs. It also does not fork the `sysctl`/`vm_stat` child
// processes the legacy budget heuristics use — this is an admission gate, not
// a heuristic, and it must not cost a process spawn per check.

/// `sysctl` name of total physical memory, immutable for the process lifetime.
#[cfg(target_os = "macos")]
const DARWIN_HW_MEMSIZE_SYSCTL: &[u8] = b"hw.memsize\0";

/// `TASK_VM_INFO` flavor id from `mach/task_info.h`.
#[cfg(target_os = "macos")]
const DARWIN_TASK_VM_INFO_FLAVOR: u32 = 22;

/// `KERN_SUCCESS` from `mach/kern_return.h`.
#[cfg(target_os = "macos")]
const DARWIN_KERN_SUCCESS: i32 = 0;

/// `RLIMIT_AS` from Darwin's `sys/resource.h` (aliased there to `RLIMIT_RSS`).
#[cfg(target_os = "macos")]
const DARWIN_RLIMIT_AS: i32 = 5;

/// `RLIM_INFINITY` from Darwin's `sys/resource.h`: `(1 << 63) - 1`.
#[cfg(target_os = "macos")]
const DARWIN_RLIM_INFINITY: u64 = 0x7fff_ffff_ffff_ffff;

/// Prefix of XNU's `struct task_vm_info` (rev1) through `phys_footprint`.
///
/// Field order and widths mirror `osfmk/mach/task_info.h` exactly; only
/// `virtual_size` and `phys_footprint` are read, the rest exist to make the
/// kernel's write land at the documented offsets. `phys_footprint` is the
/// kernel-charged footprint ledger — the number jetsam enforces — which is why
/// it, and not `resident_size`, is subtracted from physical memory below.
#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DarwinTaskVmInfo {
    virtual_size: u64,
    _region_count: i32,
    _page_size: i32,
    _resident_size: u64,
    _resident_size_peak: u64,
    _device: u64,
    _device_peak: u64,
    _internal: u64,
    _internal_peak: u64,
    _external: u64,
    _external_peak: u64,
    _reusable: u64,
    _reusable_peak: u64,
    _purgeable_volatile_pmap: u64,
    _purgeable_volatile_resident: u64,
    _purgeable_volatile_virtual: u64,
    _compressed: u64,
    _compressed_peak: u64,
    _compressed_lifetime: u64,
    phys_footprint: u64,
}

/// Darwin's `struct rlimit`: two `rlim_t` (`u64`) fields.
#[cfg(target_os = "macos")]
#[repr(C)]
struct DarwinRlimit {
    rlim_cur: u64,
    rlim_max: u64,
}

#[cfg(target_os = "macos")]
extern "C" {
    /// This process's task port; libSystem exports it as a data symbol
    /// (`mach_task_self()` is a macro over it in C).
    static mach_task_self_: u32;
    fn task_info(
        target_task: u32,
        flavor: u32,
        task_info_out: *mut i32,
        task_info_out_cnt: *mut u32,
    ) -> i32;
    fn sysctlbyname(
        name: *const std::ffi::c_char,
        oldp: *mut std::ffi::c_void,
        oldlenp: *mut usize,
        newp: *mut std::ffi::c_void,
        newlen: usize,
    ) -> std::ffi::c_int;
    fn getrlimit(resource: std::ffi::c_int, rlp: *mut DarwinRlimit) -> std::ffi::c_int;
}

/// One complete observation of every source the Darwin envelope consults.
///
/// Mirrors the Linux `LinuxMemoryProbeSnapshot` split: the live syscalls fill
/// this struct, and [`process_memory_envelope_from_darwin_snapshot`] is a pure
/// derivation over it, so the arithmetic and the fail-closed law are
/// unit-testable without syscalls. `None` in any field means the corresponding
/// syscall failed — never zero, which would fabricate a bound.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DarwinMemoryProbeSnapshot {
    /// `sysctl hw.memsize`: total physical memory.
    total_physical_bytes: Option<u64>,
    /// `TASK_VM_INFO` `phys_footprint`: this process's jetsam-charged bytes.
    phys_footprint_bytes: Option<u64>,
    /// `getrlimit(RLIMIT_AS)` soft limit, with `RLIM_INFINITY` mapped to
    /// [`ParsedMemoryLimit::Unlimited`].
    address_space_limit: Option<ParsedMemoryLimit>,
    /// `TASK_VM_INFO` `virtual_size`, consulted only against a finite
    /// `RLIMIT_AS` (the same shape as Linux's `VmSize` subtraction).
    virtual_size_bytes: Option<u64>,
}

/// Live Darwin observation. Uncached except for the immutable `hw.memsize`:
/// unlike the Linux probe this is two cheap kernel calls, not procfs handle
/// discovery, and the strict envelope's consumers are admission gates rather
/// than per-allocation hot loops.
#[cfg(target_os = "macos")]
fn darwin_memory_probe_snapshot() -> DarwinMemoryProbeSnapshot {
    let task = darwin_task_vm_info();
    DarwinMemoryProbeSnapshot {
        total_physical_bytes: darwin_total_physical_memory_bytes(),
        phys_footprint_bytes: task.map(|info| info.phys_footprint),
        address_space_limit: darwin_address_space_soft_limit(),
        virtual_size_bytes: task.map(|info| info.virtual_size),
    }
}

/// `sysctl hw.memsize` via `sysctlbyname`, cached for the process lifetime
/// (physical memory cannot change under a live process).
// The crate denies unsafe_code; this FFI probe is one of the narrow allowed
// seams (same convention as `imb::from_independently_verified_parts`): a
// direct libSystem call whose invariants are stated at the call site.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
fn darwin_total_physical_memory_bytes() -> Option<u64> {
    static MEMSIZE: OnceLock<Option<u64>> = OnceLock::new();
    *MEMSIZE.get_or_init(|| {
        let mut value: u64 = 0;
        let mut len = size_of::<u64>();
        // SAFETY: the name is NUL-terminated; `oldp`/`oldlenp` describe the
        // valid 8-byte buffer `value`; no new value is written (`newp` null,
        // `newlen` 0).
        let rc = unsafe {
            sysctlbyname(
                DARWIN_HW_MEMSIZE_SYSCTL.as_ptr().cast(),
                (&raw mut value).cast(),
                &raw mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        // A shorter-than-expected reply would leave high bytes unwritten;
        // refuse it instead of parsing a permissive partial value (the same
        // truncation law as `OpenMemoryFile::parse` on Linux).
        (rc == 0 && len == size_of::<u64>() && value > 0).then_some(value)
    })
}

/// `task_info(TASK_VM_INFO)` on this process, or `None` on any failure.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
fn darwin_task_vm_info() -> Option<DarwinTaskVmInfo> {
    let mut info = DarwinTaskVmInfo::default();
    let mut count = u32::try_from(size_of::<DarwinTaskVmInfo>() / size_of::<i32>()).ok()?;
    // SAFETY: `info` is a valid, writable rev1-prefix buffer whose length in
    // `integer_t` units is exactly `count`, so the kernel writes at most
    // `count * 4` bytes into it; `mach_task_self_` is libSystem's own task
    // port for this process.
    let rc = unsafe {
        task_info(
            mach_task_self_,
            DARWIN_TASK_VM_INFO_FLAVOR,
            (&raw mut info).cast::<i32>(),
            &raw mut count,
        )
    };
    if rc != DARWIN_KERN_SUCCESS {
        return None;
    }
    // The kernel writes min(requested, supported) integers and stores that
    // number back into `count`. Require every field through `phys_footprint`
    // to have been filled: a hypothetical pre-rev1 kernel would leave it at
    // the zero this struct was initialized with, which must read as "no
    // observation", never as a zero footprint.
    let filled_bytes = usize::try_from(count).ok()?.checked_mul(size_of::<i32>())?;
    (filled_bytes >= size_of::<DarwinTaskVmInfo>()).then_some(info)
}

/// `getrlimit(RLIMIT_AS)` soft limit, mirroring the Linux
/// `parse_address_space_soft_limit_strict` semantics: `None` on observation
/// failure, `Unlimited` for `RLIM_INFINITY`, `Finite` otherwise. XNU enforces
/// a finite `RLIMIT_AS` against the task's map size, so this is a
/// kernel-enforced envelope in exactly the Linux arm's sense; no cgroup term
/// exists to mirror.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
fn darwin_address_space_soft_limit() -> Option<ParsedMemoryLimit> {
    let mut limit = DarwinRlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `limit` is a valid, writable `struct rlimit`-layout buffer.
    let rc = unsafe { getrlimit(DARWIN_RLIMIT_AS, &raw mut limit) };
    if rc != 0 {
        return None;
    }
    Some(if limit.rlim_cur >= DARWIN_RLIM_INFINITY {
        ParsedMemoryLimit::Unlimited
    } else {
        ParsedMemoryLimit::Finite(limit.rlim_cur)
    })
}

/// Pure derivation of the strict Darwin envelope from one snapshot.
///
/// The physical term is `hw.memsize − phys_footprint`, saturating at zero: a
/// footprint at or above physical memory reads as NO headroom (fail closed in
/// the bounded direction), never as unknown. The RLIMIT term mirrors Linux's:
/// an unlimited soft limit contributes no bound, a finite one contributes
/// `limit − virtual_size`. Either source observed incompletely makes the
/// aggregate `Unavailable`, via the same
/// [`combine_process_memory_envelopes`] law the Linux arm uses.
#[cfg(target_os = "macos")]
fn process_memory_envelope_from_darwin_snapshot(
    snapshot: &DarwinMemoryProbeSnapshot,
) -> ProcessMemoryEnvelope {
    combine_process_memory_envelopes(&[
        darwin_physical_memory_envelope_from_snapshot(snapshot),
        darwin_address_space_memory_envelope_from_snapshot(snapshot),
    ])
}

/// Physical-memory term: the envelope jetsam enforces. Both observations must
/// be present; there is no `Unbounded` case, because a machine's physical
/// memory is always finite.
#[cfg(target_os = "macos")]
fn darwin_physical_memory_envelope_from_snapshot(
    snapshot: &DarwinMemoryProbeSnapshot,
) -> ProcessMemoryEnvelope {
    match (snapshot.total_physical_bytes, snapshot.phys_footprint_bytes) {
        (Some(total), Some(footprint)) => ProcessMemoryEnvelope::Bounded {
            headroom_bytes: total.saturating_sub(footprint),
        },
        _ => ProcessMemoryEnvelope::Unavailable,
    }
}

/// Address-space term, the Darwin mirror of
/// `strict_address_space_memory_envelope_from_snapshot`: the usage
/// subtraction is required only once the limit is finite.
#[cfg(target_os = "macos")]
fn darwin_address_space_memory_envelope_from_snapshot(
    snapshot: &DarwinMemoryProbeSnapshot,
) -> ProcessMemoryEnvelope {
    match snapshot.address_space_limit {
        None => ProcessMemoryEnvelope::Unavailable,
        Some(ParsedMemoryLimit::Unlimited) => ProcessMemoryEnvelope::Unbounded,
        Some(ParsedMemoryLimit::Finite(limit)) => match snapshot.virtual_size_bytes {
            Some(used) => ProcessMemoryEnvelope::Bounded {
                headroom_bytes: limit.saturating_sub(used),
            },
            None => ProcessMemoryEnvelope::Unavailable,
        },
    }
}

#[cfg(all(test, target_os = "macos"))]
mod darwin_process_memory_tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn snapshot(
        total_physical_bytes: Option<u64>,
        phys_footprint_bytes: Option<u64>,
        address_space_limit: Option<ParsedMemoryLimit>,
        virtual_size_bytes: Option<u64>,
    ) -> DarwinMemoryProbeSnapshot {
        DarwinMemoryProbeSnapshot {
            total_physical_bytes,
            phys_footprint_bytes,
            address_space_limit,
            virtual_size_bytes,
        }
    }

    #[test]
    fn darwin_headroom_is_memsize_minus_footprint() {
        let observed = snapshot(
            Some(36 * GIB),
            Some(4 * GIB),
            Some(ParsedMemoryLimit::Unlimited),
            Some(400 * GIB),
        );
        assert_eq!(
            process_memory_envelope_from_darwin_snapshot(&observed),
            ProcessMemoryEnvelope::Bounded {
                headroom_bytes: 32 * GIB
            },
            "an unlimited RLIMIT_AS must leave the physical envelope as the bound"
        );
    }

    #[test]
    fn darwin_footprint_at_or_above_memsize_saturates_to_zero_headroom() {
        for footprint in [8 * GIB, 9 * GIB, u64::MAX] {
            let observed = snapshot(
                Some(8 * GIB),
                Some(footprint),
                Some(ParsedMemoryLimit::Unlimited),
                Some(0),
            );
            assert_eq!(
                process_memory_envelope_from_darwin_snapshot(&observed),
                ProcessMemoryEnvelope::Bounded { headroom_bytes: 0 },
                "a full machine reads as ZERO headroom (fail closed in the \
                 bounded direction), never as unknown"
            );
        }
    }

    #[test]
    fn darwin_extreme_values_stay_saturating_not_wrapping() {
        let observed = snapshot(
            Some(u64::MAX),
            Some(0),
            Some(ParsedMemoryLimit::Unlimited),
            Some(0),
        );
        assert_eq!(
            process_memory_envelope_from_darwin_snapshot(&observed),
            ProcessMemoryEnvelope::Bounded {
                headroom_bytes: u64::MAX
            }
        );
        let zero_machine = snapshot(
            Some(0),
            Some(0),
            Some(ParsedMemoryLimit::Unlimited),
            Some(0),
        );
        assert_eq!(
            process_memory_envelope_from_darwin_snapshot(&zero_machine),
            ProcessMemoryEnvelope::Bounded { headroom_bytes: 0 }
        );
    }

    #[test]
    fn darwin_any_missing_observation_is_unavailable() {
        let complete = snapshot(
            Some(36 * GIB),
            Some(GIB),
            Some(ParsedMemoryLimit::Unlimited),
            Some(400 * GIB),
        );
        assert_ne!(
            process_memory_envelope_from_darwin_snapshot(&complete),
            ProcessMemoryEnvelope::Unavailable,
            "control: the complete snapshot must resolve"
        );
        let cases = [
            snapshot(
                None,
                Some(GIB),
                Some(ParsedMemoryLimit::Unlimited),
                Some(GIB),
            ),
            snapshot(
                Some(36 * GIB),
                None,
                Some(ParsedMemoryLimit::Unlimited),
                Some(GIB),
            ),
            snapshot(Some(36 * GIB), Some(GIB), None, Some(GIB)),
            // A finite RLIMIT_AS needs the usage subtraction; without it the
            // aggregate is ambiguous even though the physical term resolved.
            snapshot(
                Some(36 * GIB),
                Some(GIB),
                Some(ParsedMemoryLimit::Finite(10 * GIB)),
                None,
            ),
            DarwinMemoryProbeSnapshot::default(),
        ];
        for observed in cases {
            assert_eq!(
                process_memory_envelope_from_darwin_snapshot(&observed),
                ProcessMemoryEnvelope::Unavailable,
                "incomplete observation must fail closed: {observed:?}"
            );
        }
    }

    #[test]
    fn darwin_finite_rlimit_narrows_the_physical_envelope_and_vice_versa() {
        // RLIMIT term (10 - 4 = 6 GiB) below the physical term (32 GiB).
        let rlimit_narrower = snapshot(
            Some(36 * GIB),
            Some(4 * GIB),
            Some(ParsedMemoryLimit::Finite(10 * GIB)),
            Some(4 * GIB),
        );
        assert_eq!(
            process_memory_envelope_from_darwin_snapshot(&rlimit_narrower),
            ProcessMemoryEnvelope::Bounded {
                headroom_bytes: 6 * GIB
            }
        );
        // Physical term (2 GiB) below the RLIMIT term (60 GiB).
        let physical_narrower = snapshot(
            Some(36 * GIB),
            Some(34 * GIB),
            Some(ParsedMemoryLimit::Finite(100 * GIB)),
            Some(40 * GIB),
        );
        assert_eq!(
            process_memory_envelope_from_darwin_snapshot(&physical_narrower),
            ProcessMemoryEnvelope::Bounded {
                headroom_bytes: 2 * GIB
            },
            "the narrowest live envelope must win, as on Linux"
        );
    }

    /// Live integration proof (#darwin-memory-envelope): on a real macOS host
    /// every syscall must resolve and the strict envelope must be `Bounded`
    /// with headroom inside `(0, hw.memsize]` — the exact opposite of the
    /// pre-existing law this section replaced (`Unavailable` by construction,
    /// pinned by the old `process_memory_envelope_is_unavailable_without_a_
    /// platform_probe` test, which now covers only probe-less platforms).
    /// `Unavailable` remains lawful ONLY as a syscall-failure outcome, and a
    /// syscall failure in this test process is itself a defect worth failing.
    #[test]
    fn darwin_live_envelope_is_bounded_within_physical_memory() {
        let total = darwin_total_physical_memory_bytes()
            .expect("hw.memsize must resolve on a live macOS host");
        assert!(total > 0);
        match process_memory_envelope() {
            ProcessMemoryEnvelope::Bounded { headroom_bytes } => {
                assert!(
                    headroom_bytes > 0,
                    "a test process cannot have consumed all {total} bytes of physical memory"
                );
                assert!(
                    headroom_bytes <= total,
                    "headroom {headroom_bytes} must never exceed physical memory {total}"
                );
            }
            other => panic!(
                "macOS must MEASURE its envelope instead of failing closed by \
                 construction; got {other:?}"
            ),
        }
    }

    #[test]
    fn darwin_live_task_vm_info_reports_a_plausible_footprint() {
        let info = darwin_task_vm_info().expect("task_info(TASK_VM_INFO) must resolve live");
        assert!(
            info.phys_footprint > 0,
            "a running process has a nonzero jetsam-charged footprint"
        );
        assert!(
            info.virtual_size >= info.phys_footprint,
            "virtual size {} cannot be below the physical footprint {}",
            info.virtual_size,
            info.phys_footprint
        );
    }
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CgroupMemoryVersion {
    V1,
    V2,
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParsedMemoryLimit {
    Finite(u64),
    Unlimited,
}

/// Maximum age of cgroup identity/topology validation and cached-handle
/// reopening.
///
/// Resource admission already has an unavoidable time-of-check/time-of-use
/// window. Keeping this at 100 ms preserves prompt response to a changed
/// cgroup identity while reducing a measured 13,500 full discoveries per
/// SafeNLP proof to at most roughly ten per second. This snapshot never caches
/// values: procfs plus every resolved `memory.max`/`memory.current` pair are
/// read live through the open handles on every call that consults it. Note the
/// separate VALUE cache in front of it (#poll-amortise,
/// `MEMORY_TERMS_AMORTISE_TTL`): the dense-budget and headroom probes consult
/// this snapshot at most once per that window; `process_memory_envelope` still
/// consults it on every call.
#[cfg(target_os = "linux")]
const MEMORY_PROBE_CACHE_TTL: Duration = Duration::from_millis(100);

#[cfg(target_os = "linux")]
const SMALL_MEMORY_FILE_BYTES: usize = 128;
#[cfg(target_os = "linux")]
const PROC_MEMORY_FILE_BYTES: usize = 16 * 1024;

#[cfg(target_os = "linux")]
#[derive(Clone, Debug, Eq, PartialEq)]
struct CgroupProbeIdentity {
    memberships: String,
    mountinfo: String,
}

#[cfg(target_os = "linux")]
#[derive(Clone, Debug, Eq, PartialEq)]
struct CgroupMemoryCounterPaths {
    version: CgroupMemoryVersion,
    limit_path: PathBuf,
    usage_path: PathBuf,
}

#[cfg(target_os = "linux")]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct CgroupMemoryTopology {
    counters: Vec<CgroupMemoryCounterPaths>,
}

#[cfg(target_os = "linux")]
#[derive(Clone, Debug)]
struct OpenMemoryFile {
    file: Arc<File>,
}

#[cfg(target_os = "linux")]
impl OpenMemoryFile {
    fn open(path: &Path) -> Option<Self> {
        File::open(path).ok().map(|file| Self {
            file: Arc::new(file),
        })
    }

    fn parse<const N: usize, T>(&self, parse: impl FnOnce(&str) -> Option<T>) -> Option<T> {
        let mut bytes = [0_u8; N];
        let len = self.file.read_at(&mut bytes, 0).ok()?;
        // Refuse a possibly truncated pseudo-file instead of parsing a
        // permissive partial value. All selected capacities are far above the
        // corresponding kernel file's normal size.
        if len == bytes.len() {
            return None;
        }
        parse(std::str::from_utf8(&bytes[..len]).ok()?)
    }
}

#[cfg(target_os = "linux")]
#[derive(Clone, Debug)]
struct OpenCgroupMemoryCounter {
    version: CgroupMemoryVersion,
    limit_path: PathBuf,
    usage_path: PathBuf,
    limit: OpenMemoryFile,
    usage: OpenMemoryFile,
}

#[cfg(target_os = "linux")]
#[derive(Clone, Debug, Default)]
struct LinuxMemoryProbeSnapshot {
    cgroup_identity: Option<CgroupProbeIdentity>,
    cgroup_topology: CgroupMemoryTopology,
    cgroup_counters: Vec<OpenCgroupMemoryCounter>,
    membership: Option<OpenMemoryFile>,
    meminfo: Option<OpenMemoryFile>,
    limits: Option<OpenMemoryFile>,
    status: Option<OpenMemoryFile>,
}

#[cfg(target_os = "linux")]
impl LinuxMemoryProbeSnapshot {
    fn cgroup_membership_changed(&self) -> bool {
        let Some(identity) = self.cgroup_identity.as_ref() else {
            return true;
        };
        let Some(membership) = self.membership.as_ref() else {
            return true;
        };
        membership
            .parse::<PROC_MEMORY_FILE_BYTES, _>(|live| Some(live != identity.memberships.as_str()))
            // Missing, unreadable, non-UTF8, or truncated membership data must
            // invalidate. Trusting stale topology on an observation failure
            // could miss migration into a tighter cgroup.
            .unwrap_or(true)
    }
}

#[cfg(target_os = "linux")]
#[derive(Default)]
struct LinuxMemoryProbeCache {
    pid: Option<u32>,
    sampled_at: Option<Instant>,
    snapshot: Option<Arc<LinuxMemoryProbeSnapshot>>,
}

#[cfg(target_os = "linux")]
impl LinuxMemoryProbeCache {
    fn invalidate(&mut self) {
        self.sampled_at = None;
        self.snapshot = None;
    }

    fn get_or_refresh<F>(
        &mut self,
        pid: u32,
        now: Instant,
        ttl: Duration,
        refresh: F,
    ) -> Arc<LinuxMemoryProbeSnapshot>
    where
        F: FnOnce(Option<&LinuxMemoryProbeSnapshot>) -> LinuxMemoryProbeSnapshot,
    {
        // Defense in depth for any non-global use (including deterministic
        // tests): invalidate before the TTL fast path on PID change.
        // `linux_memory_probe_snapshot` performs its OWNER_PID check before
        // acquiring the process-global mutex, which is the fork-deadlock and
        // inherited-`/proc/self/*` safety boundary.
        if self.pid != Some(pid) {
            self.pid = Some(pid);
            self.invalidate();
        }
        if let (Some(sampled_at), Some(snapshot)) = (self.sampled_at, self.snapshot.as_ref()) {
            if now
                .checked_duration_since(sampled_at)
                .is_some_and(|age| age < ttl)
            {
                return Arc::clone(snapshot);
            }
        }

        let refreshed = Arc::new(refresh(self.snapshot.as_deref()));
        self.sampled_at = Some(now);
        self.snapshot = Some(Arc::clone(&refreshed));
        refreshed
    }
}

#[cfg(target_os = "linux")]
fn linux_memory_probe_snapshot() -> Arc<LinuxMemoryProbeSnapshot> {
    static OWNER_PID: AtomicU32 = AtomicU32::new(0);
    static CACHE: OnceLock<Mutex<LinuxMemoryProbeCache>> = OnceLock::new();

    let pid = std::process::id();
    let owner = OWNER_PID.load(Ordering::Acquire);
    let owner = if owner == 0 {
        match OWNER_PID.compare_exchange(0, pid, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => pid,
            Err(installed) => installed,
        }
    } else {
        owner
    };
    if owner != pid {
        // PID validation happens BEFORE touching the global mutex or its TTL
        // fast path. A raw-fork child can inherit the mutex while a vanished
        // sibling held it, and inherited `/proc/self/*` descriptors still
        // expose the parent. Bypass both in that rare pre-exec child. Normal
        // NY subprocesses exec immediately (all Rust-opened descriptors are
        // CLOEXEC); a long-lived raw-fork child stays correct at the cost of
        // uncached discovery.
        return Arc::new(refresh_linux_memory_probe_snapshot(None));
    }

    // One process-wide snapshot is bounded to the resolved hierarchy's six
    // counter pairs plus four procfs handles on this host (16 descriptors
    // total, well below soft RLIMIT_NOFILE=1024) regardless of Rayon width.
    let cache = CACHE.get_or_init(|| Mutex::new(LinuxMemoryProbeCache::default()));
    let mut cache = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let now = Instant::now();
    let snapshot = cache.get_or_refresh(
        pid,
        now,
        MEMORY_PROBE_CACHE_TTL,
        refresh_linux_memory_probe_snapshot,
    );
    if snapshot.cgroup_membership_changed() {
        // A process can be migrated to a tighter cgroup without forking.
        // Compare the tiny membership file on EVERY admission, before
        // trusting cached topology. A mismatch atomically discards every old
        // counter pair and re-resolves the new hierarchy.
        cache.invalidate();
        cache.get_or_refresh(
            pid,
            now,
            MEMORY_PROBE_CACHE_TTL,
            refresh_linux_memory_probe_snapshot,
        )
    } else {
        snapshot
    }
}

#[cfg(target_os = "linux")]
fn refresh_linux_memory_probe_snapshot(
    previous: Option<&LinuxMemoryProbeSnapshot>,
) -> LinuxMemoryProbeSnapshot {
    let observed_identity = std::fs::read_to_string("/proc/self/cgroup")
        .ok()
        .zip(std::fs::read_to_string("/proc/self/mountinfo").ok())
        .map(|(memberships, mountinfo)| CgroupProbeIdentity {
            memberships,
            mountinfo,
        });

    // Never trust a previous identity when fresh discovery fails: the process
    // may have migrated into a tighter hierarchy. `None` reproduces the
    // baseline observation-failure behavior (no fabricated cgroup bound), and
    // membership validation retries discovery on the next admission.
    let cgroup_identity = observed_identity;
    let same_identity = previous
        .is_some_and(|snapshot| snapshot.cgroup_identity.as_ref() == cgroup_identity.as_ref());
    let cgroup_topology = match (&cgroup_identity, previous) {
        (Some(_), Some(previous)) if same_identity => previous.cgroup_topology.clone(),
        (Some(identity), _) => cgroup_memory_topology_from_identity(identity),
        (None, _) => CgroupMemoryTopology::default(),
    };
    let previous_counters: &[OpenCgroupMemoryCounter] = if same_identity {
        previous.map_or(&[], |snapshot| snapshot.cgroup_counters.as_slice())
    } else {
        &[]
    };
    let cgroup_counters = open_cgroup_memory_counters(
        &cgroup_topology,
        previous_counters,
        &mut OpenMemoryFile::open,
    );
    let keep_or_open = |path: &Path, previous: Option<&OpenMemoryFile>| {
        OpenMemoryFile::open(path).or_else(|| previous.cloned())
    };
    let meminfo = keep_or_open(
        Path::new("/proc/meminfo"),
        previous.and_then(|snapshot| snapshot.meminfo.as_ref()),
    );
    let membership = keep_or_open(
        Path::new("/proc/self/cgroup"),
        previous.and_then(|snapshot| snapshot.membership.as_ref()),
    );
    let limits = keep_or_open(
        Path::new("/proc/self/limits"),
        previous.and_then(|snapshot| snapshot.limits.as_ref()),
    );
    let status = keep_or_open(
        Path::new("/proc/self/status"),
        previous.and_then(|snapshot| snapshot.status.as_ref()),
    );

    LinuxMemoryProbeSnapshot {
        cgroup_identity,
        cgroup_topology,
        cgroup_counters,
        membership,
        meminfo,
        limits,
        status,
    }
}

#[cfg(target_os = "linux")]
fn narrowest_headroom(headrooms: &[Option<u64>]) -> Option<u64> {
    headrooms.iter().copied().flatten().min()
}

#[cfg(target_os = "linux")]
fn observed_host_memory_bytes_from_snapshot(snapshot: &LinuxMemoryProbeSnapshot) -> Option<u64> {
    snapshot
        .meminfo
        .as_ref()
        .and_then(|file| file.parse::<PROC_MEMORY_FILE_BYTES, _>(parse_meminfo_available_kib))
}

#[cfg(target_os = "linux")]
fn process_memory_headroom_from_snapshot(snapshot: &LinuxMemoryProbeSnapshot) -> Option<u64> {
    let cgroup = cgroup_memory_headroom_from_open(&snapshot.cgroup_counters);
    // Read both inputs live on every observation. In particular, an unlimited
    // RLIMIT_AS that is lowered mid-proof must become authoritative at the next
    // observation; the snapshot caches only the open descriptors. (The
    // #poll-amortise value cache in front of this function bounds that delay
    // by `MEMORY_TERMS_AMORTISE_TTL`.)
    let address_limit = snapshot.limits.as_ref().and_then(|file| {
        file.parse::<PROC_MEMORY_FILE_BYTES, _>(parse_address_space_soft_limit_bytes)
    });
    // ONE read of `/proc/self/status` for both fields. `VmSize` and `VmRSS`
    // live in the same generated file and that generation walks the mm, so a
    // second `read_at` would double the per-admission cost of the probe rather
    // than add a field to it.
    let (address_used, self_rss) = snapshot
        .status
        .as_ref()
        .and_then(|file| {
            file.parse::<PROC_MEMORY_FILE_BYTES, _>(|contents| {
                Some((parse_vm_size_bytes(contents), parse_vm_rss_bytes(contents)))
            })
        })
        .unwrap_or((None, None));
    let address_space = address_limit
        .zip(address_used)
        .map(|(limit, used)| limit.saturating_sub(used));
    let (available, total) = system_memory_reading_from_snapshot(snapshot);
    let system = system_memory_headroom_from(available, total, self_rss);
    let headroom = narrowest_headroom(&[cgroup, address_space, system]);
    report_memory_governor(MemoryGovernorReport {
        cgroup,
        address_space,
        system,
        headroom,
        available,
        total,
        self_rss,
    });
    headroom
}

/// `MemAvailable` and `MemTotal`, from a single read of `/proc/meminfo`.
#[cfg(target_os = "linux")]
fn system_memory_reading_from_snapshot(
    snapshot: &LinuxMemoryProbeSnapshot,
) -> (Option<u64>, Option<u64>) {
    snapshot
        .meminfo
        .as_ref()
        .and_then(|file| {
            file.parse::<PROC_MEMORY_FILE_BYTES, _>(|contents| {
                Some((
                    parse_meminfo_available_kib(contents),
                    parse_meminfo_field_bytes(contents, "MemTotal:"),
                ))
            })
        })
        .unwrap_or((None, None))
}

/// The self-governing memory term: what NY may still take when NO kernel
/// envelope exists to say (#mem-governor).
///
/// Pure, so the policy is regression-testable without a host in any particular
/// state -- which is precisely the discipline the WITHDRAWN
/// `#dense-budget-valve` failed ("verified in unit tests and never end to end").
/// The tests below therefore assert the END-TO-END shapes: the quiescent host
/// where this must be inert, the OOM host where it must bind, and the
/// page-cache host where only the backstop can see the danger.
///
/// Two terms, both narrowing, combined by `narrowest_headroom`:
///
/// * `MemAvailable` less [`SYSTEM_MEMORY_RESERVE_BYTES`]. LIVE: as NY grows,
///   `MemAvailable` falls and this term falls with it, exactly as cgroup
///   headroom (`limit - usage`) already does. This is what turns a fixed
///   ceiling into a governor.
/// * The [`SELF_RSS_CEILING_NUMERATOR`] backstop, which contributes `Some(0)`
///   -- refuse every Dense materialization, take the patches/IBP route -- once
///   NY's own resident set crosses its share of `MemTotal`, and contributes NO
///   bound below it. A hard backstop rather than another proportional term, so
///   that ordinary operation is bit-identical to before and only the runaway
///   shape is touched.
///
/// # This can only LOSE proofs, never mint one
///
/// The result is consumed at every call site as a `min` against an existing
/// budget (`clamp_dense_budget_to_process_envelope`,
/// `adaptive_cpu_crown_dense_budget_from`, `narrowest_conv_crown_mem_cap`,
/// `runtime_host_budget_bytes`, `admitted_memory_plan`,
/// `build_post_c_survivor_plan_with_headroom`). A SMALLER headroom can only
/// refuse or split a materialization that would otherwise have been admitted;
/// there is no path on which it admits one. A refused Dense materialization
/// takes the pre-existing documented fallback -- `CpuMemoryExceeded` mapped to
/// IBP for the target by the CROWN-IBP collector and the alpha paths, patches
/// re-entry, or simply declining an optional tightening sweep -- and every one
/// of those is a LOOSER sound bound, never a tighter one. So this governor can
/// cost a proof (`unsat` -> `timeout`/`unknown`) and cannot mint a verdict.
#[cfg(target_os = "linux")]
fn system_memory_headroom_from(
    available_bytes: Option<u64>,
    total_bytes: Option<u64>,
    self_rss_bytes: Option<u64>,
) -> Option<u64> {
    let available_term =
        available_bytes.map(|available| available.saturating_sub(SYSTEM_MEMORY_RESERVE_BYTES));
    let backstop_term = total_bytes.zip(self_rss_bytes).and_then(|(total, rss)| {
        let ceiling = total / SELF_RSS_CEILING_DENOMINATOR * SELF_RSS_CEILING_NUMERATOR;
        (rss >= ceiling).then_some(0)
    });
    narrowest_headroom(&[available_term, backstop_term])
}

/// Everything one `#mem-governor` line reports.
#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MemoryGovernorReport {
    cgroup: Option<u64>,
    address_space: Option<u64>,
    system: Option<u64>,
    headroom: Option<u64>,
    available: Option<u64>,
    total: Option<u64>,
    self_rss: Option<u64>,
}

/// Which term actually decided the headroom, for the governor's own observable.
#[cfg(target_os = "linux")]
fn memory_governor_source(report: MemoryGovernorReport) -> &'static str {
    let Some(headroom) = report.headroom else {
        return "none";
    };
    if report.cgroup == Some(headroom) {
        "cgroup"
    } else if report.address_space == Some(headroom) {
        "rlimit-as"
    } else if report.system == Some(headroom) {
        // A zero can only come from the backstop when `MemAvailable` alone
        // has not yet reached the reserve; report the term that is really
        // holding the line so the log distinguishes them.
        let backstop_only = headroom == 0
            && report
                .available
                .is_some_and(|available| available > SYSTEM_MEMORY_RESERVE_BYTES);
        if backstop_only {
            "self-rss-backstop"
        } else {
            "system-available"
        }
    } else {
        "none"
    }
}

/// Emit the governor's own observable (#mem-governor).
///
/// Prints on `stderr` via `eprintln!` rather than `tracing`, because
/// `RUST_LOG` is not honoured by the competition binary and a lever whose
/// telemetry is silent produces VOID measurements, not negative ones.
///
/// Exactly one line is guaranteed per process (the ARMED proof: it must be
/// possible to tell "the governor is present and inert" from "the governor is
/// not in this binary"), then at most one line per
/// [`MEMORY_GOVERNOR_LOG_INTERVAL_MS`] thereafter.
#[cfg(target_os = "linux")]
fn report_memory_governor(report: MemoryGovernorReport) {
    static START: OnceLock<Instant> = OnceLock::new();
    static LAST_MS: AtomicU64 = AtomicU64::new(u64::MAX);
    static ARMED: AtomicBool = AtomicBool::new(false);

    let first = !ARMED.swap(true, Ordering::Relaxed);
    let now = Instant::now();
    let start = *START.get_or_init(|| now);
    let elapsed_ms = u64::try_from(now.saturating_duration_since(start).as_millis())
        .unwrap_or(u64::MAX)
        .min(u64::MAX - 1);
    if !first {
        let last = LAST_MS.load(Ordering::Relaxed);
        if last != u64::MAX && elapsed_ms.saturating_sub(last) < MEMORY_GOVERNOR_LOG_INTERVAL_MS {
            return;
        }
        if LAST_MS
            .compare_exchange(last, elapsed_ms, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return;
        }
    } else {
        LAST_MS.store(elapsed_ms, Ordering::Relaxed);
    }

    let render = |value: Option<u64>| value.map_or(-1_i128, i128::from);
    let ceiling = report
        .total
        .map(|total| total / SELF_RSS_CEILING_DENOMINATOR * SELF_RSS_CEILING_NUMERATOR);
    eprintln!(
        "[mem-governor] armed=1 elapsed_ms={} source={} headroom={} dense_share={} \
         system={} cgroup={} rlimit_as={} mem_available={} mem_total={} self_rss={} \
         reserve={} rss_ceiling={}",
        elapsed_ms,
        memory_governor_source(report),
        render(report.headroom),
        render(
            report
                .headroom
                .map(|headroom| headroom / PROCESS_ENVELOPE_DENSE_BUDGET_SHARE_DIVISOR)
        ),
        render(report.system),
        render(report.cgroup),
        render(report.address_space),
        render(report.available),
        render(report.total),
        render(report.self_rss),
        SYSTEM_MEMORY_RESERVE_BYTES,
        render(ceiling),
    );
}

#[cfg(target_os = "linux")]
fn process_memory_envelope_from_snapshot(
    snapshot: &LinuxMemoryProbeSnapshot,
) -> ProcessMemoryEnvelope {
    combine_process_memory_envelopes(&[
        strict_cgroup_memory_envelope_from_snapshot(snapshot),
        strict_address_space_memory_envelope_from_snapshot(snapshot),
    ])
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn combine_process_memory_envelopes(envelopes: &[ProcessMemoryEnvelope]) -> ProcessMemoryEnvelope {
    if envelopes.is_empty() || envelopes.contains(&ProcessMemoryEnvelope::Unavailable) {
        return ProcessMemoryEnvelope::Unavailable;
    }

    envelopes
        .iter()
        .filter_map(|envelope| match envelope {
            ProcessMemoryEnvelope::Bounded { headroom_bytes } => Some(*headroom_bytes),
            ProcessMemoryEnvelope::Unbounded | ProcessMemoryEnvelope::Unavailable => None,
        })
        .min()
        .map_or(ProcessMemoryEnvelope::Unbounded, |headroom_bytes| {
            ProcessMemoryEnvelope::Bounded { headroom_bytes }
        })
}

#[cfg(target_os = "linux")]
fn strict_address_space_memory_envelope_from_snapshot(
    snapshot: &LinuxMemoryProbeSnapshot,
) -> ProcessMemoryEnvelope {
    let Some(limit) = snapshot.limits.as_ref().and_then(|file| {
        file.parse::<PROC_MEMORY_FILE_BYTES, _>(parse_address_space_soft_limit_strict)
    }) else {
        return ProcessMemoryEnvelope::Unavailable;
    };
    match limit {
        ParsedMemoryLimit::Unlimited => ProcessMemoryEnvelope::Unbounded,
        ParsedMemoryLimit::Finite(limit) => {
            let Some(used) = snapshot.status.as_ref().and_then(|file| {
                file.parse::<PROC_MEMORY_FILE_BYTES, _>(parse_vm_size_bytes_strict)
            }) else {
                return ProcessMemoryEnvelope::Unavailable;
            };
            ProcessMemoryEnvelope::Bounded {
                headroom_bytes: limit.saturating_sub(used),
            }
        }
    }
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StrictCgroupMembership<'a> {
    version: CgroupMemoryVersion,
    path: &'a str,
}

#[cfg(target_os = "linux")]
fn strict_cgroup_memberships(contents: &str) -> Option<Vec<StrictCgroupMembership<'_>>> {
    let mut memberships = Vec::new();
    let mut saw_line = false;
    let mut saw_v1_memory = false;
    let mut saw_v2 = false;

    for line in contents.lines() {
        saw_line = true;
        let mut fields = line.splitn(3, ':');
        let hierarchy = fields.next()?;
        let controllers = fields.next()?;
        let path = fields.next()?;
        let hierarchy: u64 = hierarchy.parse().ok()?;
        let non_root_segments_are_normal = path == "/"
            || path
                .strip_prefix('/')?
                .split('/')
                .all(|segment| !segment.is_empty() && segment != "." && segment != "..");
        if !non_root_segments_are_normal
            || Path::new(path).components().any(|component| {
                matches!(
                    component,
                    Component::CurDir | Component::ParentDir | Component::Prefix(_)
                )
            })
        {
            return None;
        }

        if hierarchy == 0 {
            if !controllers.is_empty() || saw_v2 {
                return None;
            }
            saw_v2 = true;
            memberships.push(StrictCgroupMembership {
                version: CgroupMemoryVersion::V2,
                path,
            });
            continue;
        }

        if controllers.is_empty()
            || controllers.split(',').any(|controller| {
                controller.is_empty() || controller.chars().any(char::is_whitespace)
            })
        {
            return None;
        }
        if controllers
            .split(',')
            .any(|controller| controller == "memory")
        {
            if saw_v1_memory {
                return None;
            }
            saw_v1_memory = true;
            memberships.push(StrictCgroupMembership {
                version: CgroupMemoryVersion::V1,
                path,
            });
        }
    }

    saw_line.then_some(memberships)
}

#[cfg(target_os = "linux")]
fn strict_cgroup_memory_envelope_from_snapshot(
    snapshot: &LinuxMemoryProbeSnapshot,
) -> ProcessMemoryEnvelope {
    let Some(identity) = snapshot.cgroup_identity.as_ref() else {
        return ProcessMemoryEnvelope::Unavailable;
    };
    let Some(memberships) = strict_cgroup_memberships(&identity.memberships) else {
        return ProcessMemoryEnvelope::Unavailable;
    };
    if memberships.is_empty() {
        return ProcessMemoryEnvelope::Unbounded;
    }

    let mut narrowest = None;
    for membership in memberships {
        let Some((current, mount)) =
            resolve_cgroup_mount(&identity.mountinfo, membership.path, membership.version)
        else {
            return ProcessMemoryEnvelope::Unavailable;
        };
        let expected = cgroup_memory_counter_paths_at(&current, &mount, membership.version);
        if expected.is_empty() {
            return ProcessMemoryEnvelope::Unavailable;
        }

        for paths in expected {
            let counter = snapshot.cgroup_counters.iter().find(|counter| {
                counter.version == paths.version
                    && counter.limit_path == paths.limit_path
                    && counter.usage_path == paths.usage_path
            });
            let Some(counter) = counter else {
                // A cgroup-v2 namespace mount root may legitimately expose no
                // memory controller files while descendants do. Only paired,
                // proven absence at that exact root is conclusively unbounded;
                // partial absence and every non-root omission are ambiguous.
                let is_v2_mount_root = membership.version == CgroupMemoryVersion::V2
                    && paths.limit_path == mount.join("memory.max")
                    && paths.usage_path == mount.join("memory.current");
                if is_v2_mount_root
                    && matches!(paths.limit_path.try_exists(), Ok(false))
                    && matches!(paths.usage_path.try_exists(), Ok(false))
                {
                    continue;
                }
                return ProcessMemoryEnvelope::Unavailable;
            };

            // Read both counters even for an unlimited limit so a malformed or
            // inaccessible usage file can never be mistaken for an unbounded
            // hierarchy.
            let limit = counter
                .limit
                .parse::<SMALL_MEMORY_FILE_BYTES, _>(|contents| {
                    parse_cgroup_memory_limit_strict(contents, counter.version)
                });
            let usage = counter
                .usage
                .parse::<SMALL_MEMORY_FILE_BYTES, _>(parse_memory_usage_bytes_strict);
            let (Some(limit), Some(usage)) = (limit, usage) else {
                return ProcessMemoryEnvelope::Unavailable;
            };
            if let ParsedMemoryLimit::Finite(limit) = limit {
                let headroom = limit.saturating_sub(usage);
                narrowest = Some(narrowest.map_or(headroom, |old: u64| old.min(headroom)));
            }
        }
    }

    narrowest.map_or(ProcessMemoryEnvelope::Unbounded, |headroom_bytes| {
        ProcessMemoryEnvelope::Bounded { headroom_bytes }
    })
}

#[cfg(target_os = "linux")]
fn parse_address_space_soft_limit_strict(contents: &str) -> Option<ParsedMemoryLimit> {
    let mut matches = contents.lines().filter_map(|line| {
        line.strip_prefix("Max address space")
            .filter(|rest| rest.starts_with(char::is_whitespace))
    });
    let rest = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    let mut fields = rest.split_whitespace();
    let soft = parse_memory_limit_field(fields.next()?)?;
    let hard = parse_memory_limit_field(fields.next()?)?;
    if fields.next()? != "bytes" || fields.next().is_some() {
        return None;
    }
    match (soft, hard) {
        (ParsedMemoryLimit::Unlimited, ParsedMemoryLimit::Finite(_)) => None,
        (ParsedMemoryLimit::Finite(soft), ParsedMemoryLimit::Finite(hard)) if soft > hard => None,
        (soft, _) => Some(soft),
    }
}

#[cfg(target_os = "linux")]
fn parse_memory_limit_field(field: &str) -> Option<ParsedMemoryLimit> {
    if field == "unlimited" {
        Some(ParsedMemoryLimit::Unlimited)
    } else {
        field.parse::<u64>().ok().map(ParsedMemoryLimit::Finite)
    }
}

#[cfg(target_os = "linux")]
fn parse_vm_size_bytes_strict(contents: &str) -> Option<u64> {
    let mut matches = contents
        .lines()
        .filter_map(|line| line.strip_prefix("VmSize:"));
    let rest = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    let mut fields = rest.split_whitespace();
    let kib: u64 = fields.next()?.parse().ok()?;
    if fields.next()? != "kB" || fields.next().is_some() {
        return None;
    }
    kib.checked_mul(1024)
}

#[cfg(target_os = "linux")]
fn parse_cgroup_memory_limit_strict(
    contents: &str,
    version: CgroupMemoryVersion,
) -> Option<ParsedMemoryLimit> {
    let value = contents.trim();
    if value.is_empty() {
        return None;
    }
    if value == "max" {
        return (version == CgroupMemoryVersion::V2).then_some(ParsedMemoryLimit::Unlimited);
    }
    let parsed: u64 = value.parse().ok()?;
    if version == CgroupMemoryVersion::V1 && parsed >= (1_u64 << 60) {
        Some(ParsedMemoryLimit::Unlimited)
    } else {
        Some(ParsedMemoryLimit::Finite(parsed))
    }
}

#[cfg(target_os = "linux")]
fn parse_memory_usage_bytes_strict(contents: &str) -> Option<u64> {
    contents.trim().parse().ok()
}

#[cfg(all(test, target_os = "linux"))]
fn address_space_headroom_from(limits: &str, status: &str) -> Option<u64> {
    let limit = parse_address_space_soft_limit_bytes(limits)?;
    let used = parse_vm_size_bytes(status)?;
    Some(limit.saturating_sub(used))
}

#[cfg(target_os = "linux")]
fn parse_address_space_soft_limit_bytes(contents: &str) -> Option<u64> {
    let line = contents
        .lines()
        .find(|line| line.starts_with("Max address space"))?;
    let rest = line.strip_prefix("Max address space")?;
    let mut fields = rest.split_whitespace();
    let soft = fields.next()?;
    let _hard = fields.next()?;
    let units = fields.next()?;
    if soft == "unlimited" || units != "bytes" {
        return None;
    }
    soft.parse().ok()
}

#[cfg(target_os = "linux")]
fn parse_vm_size_bytes(contents: &str) -> Option<u64> {
    let rest = contents
        .lines()
        .find_map(|line| line.strip_prefix("VmSize:"))?;
    let mut fields = rest.split_whitespace();
    let kib: u64 = fields.next()?.parse().ok()?;
    if fields.next()? != "kB" {
        return None;
    }
    kib.checked_mul(1024)
}

#[cfg(all(test, target_os = "linux"))]
fn cgroup_memory_headroom_from(memberships: &str, mountinfo: &str) -> Option<u64> {
    let identity = CgroupProbeIdentity {
        memberships: memberships.to_owned(),
        mountinfo: mountinfo.to_owned(),
    };
    let topology = cgroup_memory_topology_from_identity(&identity);
    let counters = open_cgroup_memory_counters(&topology, &[], &mut OpenMemoryFile::open);
    cgroup_memory_headroom_from_open(&counters)
}

#[cfg(target_os = "linux")]
fn cgroup_memory_topology_from_identity(identity: &CgroupProbeIdentity) -> CgroupMemoryTopology {
    let mut counters = Vec::new();
    for version in [CgroupMemoryVersion::V2, CgroupMemoryVersion::V1] {
        let Some(membership) = cgroup_membership_path(&identity.memberships, version) else {
            continue;
        };
        let Some((current, mount)) = resolve_cgroup_mount(&identity.mountinfo, membership, version)
        else {
            continue;
        };
        counters.extend(cgroup_memory_counter_paths_at(&current, &mount, version));
    }
    CgroupMemoryTopology { counters }
}

#[cfg(target_os = "linux")]
fn cgroup_memory_counter_paths_at(
    current: &Path,
    mount: &Path,
    version: CgroupMemoryVersion,
) -> Vec<CgroupMemoryCounterPaths> {
    if !current.starts_with(mount) {
        return Vec::new();
    }
    let (limit_name, usage_name) = match version {
        CgroupMemoryVersion::V2 => ("memory.max", "memory.current"),
        CgroupMemoryVersion::V1 => ("memory.limit_in_bytes", "memory.usage_in_bytes"),
    };
    let mut counters = Vec::new();
    let mut directory = current.to_path_buf();
    loop {
        counters.push(CgroupMemoryCounterPaths {
            version,
            limit_path: directory.join(limit_name),
            usage_path: directory.join(usage_name),
        });
        if directory == mount {
            break;
        }
        let Some(parent) = directory.parent() else {
            break;
        };
        if !parent.starts_with(mount) {
            break;
        }
        directory = parent.to_path_buf();
    }
    counters
}

#[cfg(target_os = "linux")]
fn open_cgroup_memory_counters<F>(
    topology: &CgroupMemoryTopology,
    previous: &[OpenCgroupMemoryCounter],
    open: &mut F,
) -> Vec<OpenCgroupMemoryCounter>
where
    F: FnMut(&Path) -> Option<OpenMemoryFile>,
{
    topology
        .counters
        .iter()
        .filter_map(|counter| {
            let old = previous.iter().find(|old| {
                old.version == counter.version
                    && old.limit_path == counter.limit_path
                    && old.usage_path == counter.usage_path
            });
            match (open(&counter.limit_path), open(&counter.usage_path)) {
                (Some(limit), Some(usage)) => Some(OpenCgroupMemoryCounter {
                    version: counter.version,
                    limit_path: counter.limit_path.clone(),
                    usage_path: counter.usage_path.clone(),
                    limit,
                    usage,
                }),
                // Treat the pair atomically. Mixing a newly opened limit with
                // stale usage (or vice versa) could fabricate headroom. Reuse
                // only a complete pair from the SAME cgroup identity; callers
                // pass an empty `previous` slice after identity invalidation.
                _ => old.cloned(),
            }
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn cgroup_memory_headroom_from_open(counters: &[OpenCgroupMemoryCounter]) -> Option<u64> {
    counters
        .iter()
        .filter_map(|counter| {
            // Read BOTH live even when the current limit is unlimited. This
            // preserves the original dynamic-envelope contract exactly:
            // unlimited -> finite, finite tightening, and changing usage all
            // take effect at the next admission. Only opening the files and
            // resolving their paths are cached.
            let limit = counter.limit.parse::<SMALL_MEMORY_FILE_BYTES, _>(|value| {
                parse_cgroup_memory_limit_bytes(value, counter.version)
            });
            let usage = counter
                .usage
                .parse::<SMALL_MEMORY_FILE_BYTES, _>(|value| value.trim().parse::<u64>().ok());
            let (Some(limit), Some(usage)) = (limit, usage) else {
                return None;
            };
            Some(limit.saturating_sub(usage))
        })
        .min()
}

#[cfg(target_os = "linux")]
fn cgroup_membership_path(contents: &str, version: CgroupMemoryVersion) -> Option<&str> {
    contents.lines().find_map(|line| {
        let mut fields = line.splitn(3, ':');
        let hierarchy = fields.next()?;
        let controllers = fields.next()?;
        let path = fields.next()?;
        let matches = match version {
            CgroupMemoryVersion::V2 => hierarchy == "0" && controllers.is_empty(),
            CgroupMemoryVersion::V1 => controllers.split(',').any(|name| name == "memory"),
        };
        matches.then_some(path)
    })
}

#[cfg(target_os = "linux")]
fn resolve_cgroup_mount(
    mountinfo: &str,
    membership: &str,
    version: CgroupMemoryVersion,
) -> Option<(PathBuf, PathBuf)> {
    let membership = Path::new(membership);
    let mut best: Option<(usize, PathBuf, PathBuf)> = None;
    for line in mountinfo.lines() {
        let Some((before_separator, after_separator)) = line.split_once(" - ") else {
            continue;
        };
        let mut after = after_separator.split_whitespace();
        let Some(fs_type) = after.next() else {
            continue;
        };
        let _source = after.next();
        let super_options = after.next().unwrap_or_default();
        let matches = match version {
            CgroupMemoryVersion::V2 => fs_type == "cgroup2",
            CgroupMemoryVersion::V1 => {
                fs_type == "cgroup" && super_options.split(',').any(|name| name == "memory")
            }
        };
        if !matches {
            continue;
        }

        let before: Vec<_> = before_separator.split_whitespace().collect();
        let (Some(raw_root), Some(raw_mount)) = (before.get(3), before.get(4)) else {
            continue;
        };
        let Some(root) = decode_mountinfo_path(raw_root) else {
            continue;
        };
        let Some(mount) = decode_mountinfo_path(raw_mount) else {
            continue;
        };
        let Ok(relative) = membership.strip_prefix(&root) else {
            continue;
        };
        if relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        }) {
            continue;
        }
        let current = mount.join(relative);
        let specificity = root.components().count();
        if best
            .as_ref()
            .is_none_or(|(best_specificity, _, _)| specificity > *best_specificity)
        {
            best = Some((specificity, current, mount));
        }
    }
    best.map(|(_, current, mount)| (current, mount))
}

#[cfg(target_os = "linux")]
fn decode_mountinfo_path(raw: &str) -> Option<PathBuf> {
    let raw = raw.as_bytes();
    let mut decoded = Vec::with_capacity(raw.len());
    let mut index = 0;
    while index < raw.len() {
        if raw[index] != b'\\' {
            decoded.push(raw[index]);
            index += 1;
            continue;
        }
        let octal = raw.get(index + 1..index + 4)?;
        if !octal.iter().all(|&byte| matches!(byte, b'0'..=b'7')) {
            return None;
        }
        let value = u16::from(octal[0] - b'0') * 64
            + u16::from(octal[1] - b'0') * 8
            + u16::from(octal[2] - b'0');
        decoded.push(u8::try_from(value).ok()?);
        index += 4;
    }
    Some(PathBuf::from(OsString::from_vec(decoded)))
}

#[cfg(all(test, target_os = "linux"))]
fn cgroup_memory_headroom_at(
    current: &Path,
    mount: &Path,
    version: CgroupMemoryVersion,
) -> Option<u64> {
    let topology = CgroupMemoryTopology {
        counters: cgroup_memory_counter_paths_at(current, mount, version),
    };
    let counters = open_cgroup_memory_counters(&topology, &[], &mut OpenMemoryFile::open);
    cgroup_memory_headroom_from_open(&counters)
}

#[cfg(target_os = "linux")]
fn parse_cgroup_memory_limit_bytes(contents: &str, version: CgroupMemoryVersion) -> Option<u64> {
    let value = contents.trim();
    if value.is_empty() || value == "max" {
        return None;
    }
    let parsed: u64 = value.parse().ok()?;
    // Cgroup v1 represents "unlimited" with a huge page-aligned sentinel
    // close to i64::MAX rather than a word such as cgroup v2's `max`.
    if version == CgroupMemoryVersion::V1 && parsed >= (1_u64 << 60) {
        None
    } else {
        Some(parsed)
    }
}

/// Estimate the heap bytes for a Dense lower/upper coefficient pair.
pub(crate) fn dense_pair_bytes(rows: usize, cols: usize) -> Option<usize> {
    rows.checked_mul(cols)?
        .checked_mul(size_of::<f32>())?
        .checked_mul(2)
}

/// Estimate the heap bytes for a Dense identity lower/upper coefficient pair.
pub(crate) fn identity_pair_bytes(dim: usize) -> Option<usize> {
    dense_pair_bytes(dim, dim)
}

/// Upfront estimate for a batched Dense coefficient-pair materialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BatchedDenseMaterializationEstimate {
    pub site: &'static str,
    pub batch_positions: usize,
    pub rows: usize,
    pub cols: usize,
    pub required_bytes: usize,
}

impl BatchedDenseMaterializationEstimate {
    /// Create estimate, saturating to `usize::MAX` on overflow.
    pub(crate) fn new(
        site: &'static str,
        batch_positions: usize,
        rows: usize,
        cols: usize,
    ) -> Self {
        Self {
            site,
            batch_positions,
            rows,
            cols,
            required_bytes: batched_dense_pair_bytes(batch_positions, rows, cols)
                .unwrap_or(usize::MAX),
        }
    }

    /// Whether this estimate exceeds the configured budget.
    pub(crate) fn exceeds_budget(self, budget_bytes: usize) -> bool {
        self.required_bytes > budget_bytes
    }

    /// Return `CpuMemoryExceeded` error if estimate exceeds budget, `Ok(())` otherwise.
    pub(crate) fn check_budget(self) -> ny_core::Result<()> {
        let budget = cpu_crown_dense_budget_bytes();
        if self.exceeds_budget(budget) {
            Err(NyError::CpuMemoryExceeded {
                required_bytes: self.required_bytes,
                budget_bytes: budget,
                site: self.site,
            })
        } else {
            Ok(())
        }
    }
}

/// Estimate heap bytes for a batched Dense lower/upper coefficient pair.
///
/// `required_bytes = 2 * batch_positions * rows * cols * sizeof(f32)`
pub(crate) fn batched_dense_pair_bytes(
    batch_positions: usize,
    rows: usize,
    cols: usize,
) -> Option<usize> {
    batch_positions
        .checked_mul(rows)?
        .checked_mul(cols)?
        .checked_mul(size_of::<f32>())?
        .checked_mul(2)
}

/// Estimate heap bytes for a batched Dense identity lower/upper pair.
///
/// For shape `[batch..., dim]`, `batch_positions = product(batch_dims)` and
/// `rows = cols = dim`.
pub(crate) fn batched_identity_pair_bytes(shape: &[usize]) -> Option<usize> {
    if shape.is_empty() {
        return Some(0);
    }
    let dim = shape[shape.len() - 1];
    let batch_positions = checked_shape_product(&shape[..shape.len() - 1])?.max(1);
    batched_dense_pair_bytes(batch_positions, dim, dim)
}

/// Check that a batched identity allocation for `shape` fits within the CPU budget.
///
/// Returns `Ok(())` if within budget, `Err(CpuMemoryExceeded)` otherwise.
pub(crate) fn check_batched_identity_budget(
    site: &'static str,
    shape: &[usize],
) -> ny_core::Result<()> {
    if shape.is_empty() {
        return Ok(());
    }
    let dim = shape[shape.len() - 1];
    let batch_positions = checked_shape_product(&shape[..shape.len() - 1])
        .unwrap_or(usize::MAX)
        .max(1);
    let estimate = BatchedDenseMaterializationEstimate::new(site, batch_positions, dim, dim);
    debug_assert_eq!(
        estimate.required_bytes,
        batched_identity_pair_bytes(shape).unwrap_or(usize::MAX)
    );
    estimate.check_budget()
}

#[cfg(all(test, target_os = "linux"))]
mod process_memory_tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn explicit_dense_budget_is_distinct_from_the_default() {
        crate::tests::with_serialized_env_vars_removed(&[CROWN_DENSE_BUDGET_ENV], || {
            assert_eq!(explicit_cpu_crown_dense_budget_bytes(), None);
            // The implicit value is host-adaptive. Its deterministic policy is
            // covered by `adaptive_budget_tests`; this test only establishes
            // that no explicit operator override is present.
        });
        crate::tests::with_crown_dense_budget_mb("3072", || {
            let explicit = 3072 * 1024 * 1024;
            assert_eq!(explicit_cpu_crown_dense_budget_bytes(), Some(explicit));
            assert!(
                cpu_crown_dense_budget_bytes() <= explicit,
                "a process envelope may narrow but never inflate an explicit budget"
            );
        });
        crate::tests::with_crown_dense_budget_mb("malformed", || {
            assert_eq!(
                explicit_cpu_crown_dense_budget_bytes(),
                Some(DEFAULT_CROWN_DENSE_BUDGET_MB * 1024 * 1024)
            );
        });
        let overflowing_mb = usize::MAX.to_string();
        crate::tests::with_crown_dense_budget_mb(&overflowing_mb, || {
            assert_eq!(explicit_cpu_crown_dense_budget_bytes(), Some(usize::MAX));
        });
    }

    #[test]
    fn parses_cgroup_memberships_and_mounts() {
        let memberships = concat!(
            "11:cpu,cpuacct:/legacy-cpu\n",
            "10:memory,blkio:/legacy-memory/job\n",
            "0::/user.slice/ny-build.slice/job.service\n",
        );
        assert_eq!(
            cgroup_membership_path(memberships, CgroupMemoryVersion::V2),
            Some("/user.slice/ny-build.slice/job.service")
        );
        assert_eq!(
            cgroup_membership_path(memberships, CgroupMemoryVersion::V1),
            Some("/legacy-memory/job")
        );

        let mountinfo = concat!(
            "39 29 0:33 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
            "40 29 0:34 /legacy-memory /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n",
        );
        let (v2_current, v2_mount) = resolve_cgroup_mount(
            mountinfo,
            "/user.slice/ny-build.slice/job.service",
            CgroupMemoryVersion::V2,
        )
        .expect("v2 mount resolves");
        assert_eq!(v2_mount, Path::new("/sys/fs/cgroup"));
        assert_eq!(
            v2_current,
            Path::new("/sys/fs/cgroup/user.slice/ny-build.slice/job.service")
        );

        let (v1_current, v1_mount) =
            resolve_cgroup_mount(mountinfo, "/legacy-memory/job", CgroupMemoryVersion::V1)
                .expect("v1 mount resolves");
        assert_eq!(v1_mount, Path::new("/sys/fs/cgroup/memory"));
        assert_eq!(v1_current, Path::new("/sys/fs/cgroup/memory/job"));
    }

    #[test]
    fn mountinfo_decoder_handles_kernel_octal_escapes() {
        assert_eq!(
            decode_mountinfo_path(r"/sys/fs/cgroup/with\040space"),
            Some(PathBuf::from("/sys/fs/cgroup/with space"))
        );
        assert!(decode_mountinfo_path(r"/bad\escape").is_none());
        assert!(decode_mountinfo_path(r"/bad\777").is_none());
    }

    #[test]
    fn parses_cgroup_limits_rlimit_and_vm_size() {
        assert_eq!(
            parse_cgroup_memory_limit_bytes("1073741824\n", CgroupMemoryVersion::V2),
            Some(1_073_741_824)
        );
        assert_eq!(
            parse_cgroup_memory_limit_bytes("max\n", CgroupMemoryVersion::V2),
            None
        );
        assert_eq!(
            parse_cgroup_memory_limit_bytes("9223372036854771712\n", CgroupMemoryVersion::V1),
            None,
            "the v1 unlimited sentinel is not a real envelope"
        );

        let limits = concat!(
            "Limit                     Soft Limit           Hard Limit           Units\n",
            "Max cpu time              unlimited            unlimited            seconds\n",
            "Max address space         8589934592           8589934592           bytes\n",
        );
        assert_eq!(
            parse_address_space_soft_limit_bytes(limits),
            Some(8_589_934_592)
        );
        assert_eq!(
            parse_address_space_soft_limit_bytes(
                "Max address space         unlimited            unlimited            bytes\n"
            ),
            None
        );
        assert_eq!(
            parse_vm_size_bytes("Name:\tny\nVmSize:\t   65536 kB\n"),
            Some(64 * 1024 * 1024)
        );
        assert_eq!(parse_vm_size_bytes("VmSize: 64 bytes\n"), None);
        assert_eq!(
            address_space_headroom_from(limits, "VmSize: 2097152 kB\n"),
            Some(6_442_450_944)
        );
        assert_eq!(address_space_headroom_from(limits, "Name:\tny\n"), None);
        assert_eq!(
            address_space_headroom_from(limits, "VmSize: malformed kB\n"),
            None,
            "an unreadable usage counter must not be treated as zero usage"
        );
        assert_eq!(
            address_space_headroom_from(limits, "VmSize: 16777216 kB\n"),
            Some(0),
            "usage above the soft limit saturates to exhausted headroom"
        );
    }

    #[test]
    fn strict_memory_envelope_parsers_preserve_unbounded_and_unavailable() {
        assert_eq!(
            parse_cgroup_memory_limit_strict("4096\n", CgroupMemoryVersion::V2),
            Some(ParsedMemoryLimit::Finite(4096))
        );
        assert_eq!(
            parse_cgroup_memory_limit_strict("max\n", CgroupMemoryVersion::V2),
            Some(ParsedMemoryLimit::Unlimited)
        );
        assert_eq!(
            parse_cgroup_memory_limit_strict("max\n", CgroupMemoryVersion::V1),
            None,
            "the v2 spelling is malformed in a v1 counter"
        );
        assert_eq!(
            parse_cgroup_memory_limit_strict("9223372036854771712\n", CgroupMemoryVersion::V1),
            Some(ParsedMemoryLimit::Unlimited)
        );
        assert_eq!(
            parse_cgroup_memory_limit_strict("malformed\n", CgroupMemoryVersion::V2),
            None
        );

        assert_eq!(
            parse_address_space_soft_limit_strict(
                "Max address space         unlimited            unlimited            bytes\n"
            ),
            Some(ParsedMemoryLimit::Unlimited)
        );
        assert_eq!(
            parse_address_space_soft_limit_strict(
                "Max address space         8192                 unlimited            bytes\n"
            ),
            Some(ParsedMemoryLimit::Finite(8192))
        );
        assert_eq!(
            parse_address_space_soft_limit_strict(
                "Max address space         8192                 unlimited            kbytes\n"
            ),
            None
        );
        assert_eq!(parse_vm_size_bytes_strict("VmSize: 8 kB\n"), Some(8192));
        assert_eq!(parse_vm_size_bytes_strict("VmSize: 8 kB trailing\n"), None);
    }

    #[test]
    fn strict_memory_envelope_fold_is_fail_closed_and_uses_the_narrowest_bound() {
        use ProcessMemoryEnvelope::{Bounded, Unavailable, Unbounded};

        assert_eq!(combine_process_memory_envelopes(&[]), Unavailable);
        assert_eq!(
            combine_process_memory_envelopes(&[Unbounded, Unbounded]),
            Unbounded
        );
        assert_eq!(
            combine_process_memory_envelopes(&[
                Unbounded,
                Bounded { headroom_bytes: 9 },
                Bounded { headroom_bytes: 4 },
            ]),
            Bounded { headroom_bytes: 4 }
        );
        assert_eq!(
            combine_process_memory_envelopes(&[
                Bounded { headroom_bytes: 0 },
                Unavailable,
                Unbounded,
            ]),
            Unavailable,
            "an unobserved source may be tighter than every known bound"
        );
    }

    #[test]
    fn strict_cgroup_membership_distinguishes_absent_ambiguous_and_duplicate_sources() {
        assert_eq!(
            strict_cgroup_memberships("2:cpu,cpuacct:/job\n"),
            Some(Vec::new()),
            "a valid membership with no memory hierarchy is conclusively unbounded"
        );
        for malformed in [
            "",
            "0:memory:/job\n",
            "0::relative\n",
            "0::/parent/../job\n",
            "0::/first\n0::/second\n",
            "7:memory:/first\n8:memory:/second\n",
        ] {
            assert_eq!(
                strict_cgroup_memberships(malformed),
                None,
                "malformed or ambiguous membership must fail closed: {malformed:?}"
            );
        }

        let no_memory = LinuxMemoryProbeSnapshot {
            cgroup_identity: Some(CgroupProbeIdentity {
                memberships: "2:cpu,cpuacct:/job\n".to_owned(),
                mountinfo: String::new(),
            }),
            ..LinuxMemoryProbeSnapshot::default()
        };
        assert_eq!(
            strict_cgroup_memory_envelope_from_snapshot(&no_memory),
            ProcessMemoryEnvelope::Unbounded
        );
        let unresolved = LinuxMemoryProbeSnapshot {
            cgroup_identity: Some(CgroupProbeIdentity {
                memberships: "0::/job\n".to_owned(),
                mountinfo: String::new(),
            }),
            ..LinuxMemoryProbeSnapshot::default()
        };
        assert_eq!(
            strict_cgroup_memory_envelope_from_snapshot(&unresolved),
            ProcessMemoryEnvelope::Unavailable
        );
    }

    #[test]
    fn strict_cgroup_envelope_handles_v2_mount_root_absence_without_hiding_failures() {
        let unique = format!(
            "ny-strict-cgroup-memory-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let mount = std::env::temp_dir().join(unique);
        let parent = mount.join("parent");
        let current = parent.join("job");
        std::fs::create_dir_all(&current).expect("create strict synthetic cgroup tree");
        std::fs::write(parent.join("memory.max"), "1000\n").expect("parent limit");
        std::fs::write(parent.join("memory.current"), "400\n").expect("parent usage");
        std::fs::write(current.join("memory.max"), "800\n").expect("current limit");
        std::fs::write(current.join("memory.current"), "300\n").expect("current usage");

        let identity = CgroupProbeIdentity {
            memberships: "0::/parent/job\n".to_owned(),
            mountinfo: format!("39 29 0:33 / {} rw - cgroup2 cgroup rw\n", mount.display()),
        };
        let topology = cgroup_memory_topology_from_identity(&identity);
        let counters = open_cgroup_memory_counters(&topology, &[], &mut OpenMemoryFile::open);
        let snapshot = LinuxMemoryProbeSnapshot {
            cgroup_identity: Some(identity),
            cgroup_topology: topology,
            cgroup_counters: counters,
            ..LinuxMemoryProbeSnapshot::default()
        };
        assert_eq!(
            strict_cgroup_memory_envelope_from_snapshot(&snapshot),
            ProcessMemoryEnvelope::Bounded {
                headroom_bytes: 500
            },
            "the exact v2 mount root may omit both files without masking descendant bounds"
        );
        assert_eq!(
            process_memory_headroom_from_snapshot(&snapshot),
            Some(500),
            "the legacy Dense/Conv probe must retain its independent Option semantics"
        );
        assert_eq!(
            process_memory_envelope_from_snapshot(&snapshot),
            ProcessMemoryEnvelope::Unavailable,
            "missing RLIMIT observation must dominate only in the strict aggregate"
        );

        std::fs::write(parent.join("memory.max"), "max\n").expect("unbound parent");
        std::fs::write(current.join("memory.max"), "max\n").expect("unbound current");
        assert_eq!(
            strict_cgroup_memory_envelope_from_snapshot(&snapshot),
            ProcessMemoryEnvelope::Unbounded
        );
        std::fs::write(parent.join("memory.max"), "malformed\n").expect("malformed ancestor limit");
        assert_eq!(
            strict_cgroup_memory_envelope_from_snapshot(&snapshot),
            ProcessMemoryEnvelope::Unavailable,
            "a malformed ancestor must dominate an otherwise readable hierarchy"
        );

        std::fs::write(parent.join("memory.max"), "max\n").expect("restore parent");
        std::fs::write(mount.join("memory.max"), "max\n").expect("one-sided root counter");
        assert_eq!(
            strict_cgroup_memory_envelope_from_snapshot(&snapshot),
            ProcessMemoryEnvelope::Unavailable,
            "one missing root counter is ambiguous, not structurally absent"
        );
        std::fs::remove_dir_all(&mount).expect("remove strict synthetic cgroup tree");
    }

    #[test]
    fn strict_rlimit_envelope_needs_vm_size_only_for_a_finite_limit() {
        let unique = format!(
            "ny-strict-rlimit-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let base = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&base).expect("create strict RLIMIT fixture");
        let limits_path = base.join("limits");
        let status_path = base.join("status");
        std::fs::write(
            &limits_path,
            "Max address space         8192                 unlimited            bytes\n",
        )
        .expect("write finite RLIMIT");
        std::fs::write(&status_path, "VmSize: 4 kB\n").expect("write VmSize");
        let snapshot = LinuxMemoryProbeSnapshot {
            limits: OpenMemoryFile::open(&limits_path),
            status: OpenMemoryFile::open(&status_path),
            ..LinuxMemoryProbeSnapshot::default()
        };
        assert_eq!(
            strict_address_space_memory_envelope_from_snapshot(&snapshot),
            ProcessMemoryEnvelope::Bounded {
                headroom_bytes: 4096
            }
        );

        let missing_status = LinuxMemoryProbeSnapshot {
            limits: snapshot.limits,
            ..LinuxMemoryProbeSnapshot::default()
        };
        assert_eq!(
            strict_address_space_memory_envelope_from_snapshot(&missing_status),
            ProcessMemoryEnvelope::Unavailable
        );
        std::fs::write(
            &limits_path,
            "Max address space         unlimited            unlimited            bytes\n",
        )
        .expect("write unlimited RLIMIT");
        assert_eq!(
            strict_address_space_memory_envelope_from_snapshot(&missing_status),
            ProcessMemoryEnvelope::Unbounded,
            "an unlimited address-space limit has no usage subtraction"
        );
        std::fs::remove_dir_all(&base).expect("remove strict RLIMIT fixture");
    }

    #[test]
    fn narrowest_headroom_ignores_unlimited_sources() {
        assert_eq!(
            narrowest_headroom(&[None, Some(9_000), Some(4_000)]),
            Some(4_000)
        );
        assert_eq!(narrowest_headroom(&[None, None]), None);
        assert_eq!(narrowest_headroom(&[Some(0), Some(4_000)]), Some(0));
    }

    #[test]
    fn cached_file_rejects_an_exact_capacity_parseable_prefix() {
        let unique = format!(
            "ny-memory-file-truncation-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let path = std::env::temp_dir().join(unique);
        std::fs::write(&path, "123\n").expect("write exact-capacity fixture");
        let file = OpenMemoryFile::open(&path).expect("open fixture");
        assert_eq!(
            file.parse::<4, u64>(|value| value.trim().parse().ok()),
            None,
            "an exact-size read may be truncated and must never be parsed"
        );
        std::fs::remove_file(&path).expect("remove fixture");
    }

    #[test]
    fn membership_mismatch_or_read_failure_forces_invalidation() {
        let unique = format!(
            "ny-cgroup-membership-live-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let path = std::env::temp_dir().join(unique);
        std::fs::write(&path, "0::/first\n").expect("write membership");
        let membership = OpenMemoryFile::open(&path).expect("open membership");
        let snapshot = LinuxMemoryProbeSnapshot {
            cgroup_identity: Some(CgroupProbeIdentity {
                memberships: "0::/first\n".to_owned(),
                mountinfo: "fixture".to_owned(),
            }),
            membership: Some(membership),
            ..LinuxMemoryProbeSnapshot::default()
        };
        assert!(!snapshot.cgroup_membership_changed());

        std::fs::write(&path, "0::/tighter\n").expect("migrate membership");
        assert!(snapshot.cgroup_membership_changed());

        let missing = LinuxMemoryProbeSnapshot {
            cgroup_identity: snapshot.cgroup_identity.clone(),
            ..LinuxMemoryProbeSnapshot::default()
        };
        assert!(
            missing.cgroup_membership_changed(),
            "missing live membership must not trust cached topology"
        );

        std::fs::write(&path, vec![b'x'; PROC_MEMORY_FILE_BYTES])
            .expect("write exact-capacity membership");
        assert!(
            snapshot.cgroup_membership_changed(),
            "a possibly truncated membership read must invalidate"
        );
        std::fs::remove_file(&path).expect("remove membership fixture");
    }

    #[test]
    fn bounded_probe_cache_invalidates_before_ttl_on_pid_change() {
        let now = Instant::now();
        let ttl = Duration::from_millis(100);
        let refreshes = Cell::new(0);
        let mut cache = LinuxMemoryProbeCache::default();
        let first = cache.get_or_refresh(17, now, ttl, |_| {
            refreshes.set(refreshes.get() + 1);
            LinuxMemoryProbeSnapshot {
                cgroup_identity: Some(CgroupProbeIdentity {
                    memberships: "0::/first\n".to_owned(),
                    mountinfo: "first".to_owned(),
                }),
                ..LinuxMemoryProbeSnapshot::default()
            }
        });
        let reused = cache.get_or_refresh(17, now + Duration::from_millis(99), ttl, |_| {
            panic!("a cache entry younger than the bounded TTL must be reused")
        });
        assert!(Arc::ptr_eq(&first, &reused));
        assert_eq!(refreshes.get(), 1);

        let child = cache.get_or_refresh(18, now + Duration::from_millis(99), ttl, |previous| {
            refreshes.set(refreshes.get() + 1);
            assert!(
                previous.is_none(),
                "a forked child must drop parent handles"
            );
            LinuxMemoryProbeSnapshot {
                cgroup_identity: Some(CgroupProbeIdentity {
                    memberships: "0::/child\n".to_owned(),
                    mountinfo: "child".to_owned(),
                }),
                ..LinuxMemoryProbeSnapshot::default()
            }
        });
        assert!(!Arc::ptr_eq(&first, &child));
        assert_eq!(refreshes.get(), 2);
        assert_eq!(
            child
                .cgroup_identity
                .as_ref()
                .map(|identity| identity.memberships.as_str()),
            Some("0::/child\n")
        );

        let refreshed =
            cache.get_or_refresh(18, now + Duration::from_millis(199), ttl, |previous| {
                refreshes.set(refreshes.get() + 1);
                assert_eq!(
                    previous.and_then(|snapshot| snapshot.cgroup_identity.as_ref()),
                    child.cgroup_identity.as_ref()
                );
                LinuxMemoryProbeSnapshot::default()
            });
        assert!(!Arc::ptr_eq(&child, &refreshed));
        assert_eq!(refreshes.get(), 3);
    }

    #[test]
    fn changed_cgroup_identity_re_resolves_counter_paths() {
        let mountinfo = "39 29 0:33 / /cg rw - cgroup2 cgroup rw\n";
        let first_identity = CgroupProbeIdentity {
            memberships: "0::/first\n".to_owned(),
            mountinfo: mountinfo.to_owned(),
        };
        let first = cgroup_memory_topology_from_identity(&first_identity);
        assert_eq!(
            first
                .counters
                .iter()
                .map(|counter| counter.usage_path.as_path())
                .collect::<Vec<_>>(),
            vec![
                Path::new("/cg/first/memory.current"),
                Path::new("/cg/memory.current")
            ]
        );

        let second_identity = CgroupProbeIdentity {
            memberships: "0::/second\n".to_owned(),
            mountinfo: mountinfo.to_owned(),
        };
        let second = cgroup_memory_topology_from_identity(&second_identity);
        assert_ne!(first_identity, second_identity);
        assert_eq!(
            second
                .counters
                .iter()
                .map(|counter| counter.usage_path.as_path())
                .collect::<Vec<_>>(),
            vec![
                Path::new("/cg/second/memory.current"),
                Path::new("/cg/memory.current")
            ]
        );
    }

    #[test]
    fn cached_handles_keep_unlimited_to_finite_limit_and_usage_live() {
        let unique = format!(
            "ny-cgroup-live-handle-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let base = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&base).expect("create synthetic cgroup");
        let limit_path = base.join("memory.max");
        let usage_path = base.join("memory.current");
        std::fs::write(&limit_path, "max\n").expect("initial unlimited limit");
        std::fs::write(&usage_path, "300\n").expect("initial usage");
        let topology = CgroupMemoryTopology {
            counters: vec![CgroupMemoryCounterPaths {
                version: CgroupMemoryVersion::V2,
                limit_path: limit_path.clone(),
                usage_path: usage_path.clone(),
            }],
        };
        let opens = Cell::new(0);
        let counters = open_cgroup_memory_counters(&topology, &[], &mut |path| {
            opens.set(opens.get() + 1);
            OpenMemoryFile::open(path)
        });
        assert_eq!(opens.get(), 2, "each path is opened exactly once");
        assert_eq!(cgroup_memory_headroom_from_open(&counters), None);

        std::fs::write(&limit_path, "1000\n").expect("tighten unlimited limit");
        assert_eq!(
            cgroup_memory_headroom_from_open(&counters),
            Some(700),
            "unlimited -> finite must become authoritative without reopening"
        );
        std::fs::write(&limit_path, "800\n").expect("tighten finite limit");
        std::fs::write(&usage_path, "700\n").expect("raise live usage");
        assert_eq!(
            cgroup_memory_headroom_from_open(&counters),
            Some(100),
            "finite tightening and changing usage must both remain live"
        );
        assert_eq!(opens.get(), 2, "live reads must reuse both open handles");

        let partial = open_cgroup_memory_counters(&topology, &counters, &mut |path| {
            (path == limit_path)
                .then(|| OpenMemoryFile::open(path))
                .flatten()
        });
        assert_eq!(partial.len(), 1);
        assert!(
            Arc::ptr_eq(&partial[0].limit.file, &counters[0].limit.file)
                && Arc::ptr_eq(&partial[0].usage.file, &counters[0].usage.file),
            "a partial reopen must reuse the old pair atomically"
        );
        assert!(
            open_cgroup_memory_counters(&topology, &[], &mut |path| {
                (path == limit_path)
                    .then(|| OpenMemoryFile::open(path))
                    .flatten()
            })
            .is_empty(),
            "an identity change must never reuse either half of an old pair"
        );
        std::fs::remove_dir_all(&base).expect("remove synthetic cgroup");
    }

    #[test]
    fn cgroup_headroom_includes_tighter_ancestor_and_current_usage() {
        let unique = format!(
            "ny-cgroup-memory-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let mount = std::env::temp_dir().join(unique);
        let parent = mount.join("parent");
        let current = parent.join("job");
        std::fs::create_dir_all(&current).expect("create synthetic cgroup tree");
        std::fs::write(parent.join("memory.max"), "1000\n").expect("parent limit");
        std::fs::write(parent.join("memory.current"), "400\n").expect("parent usage");
        std::fs::write(current.join("memory.max"), "800\n").expect("current limit");
        std::fs::write(current.join("memory.current"), "300\n").expect("current usage");

        assert_eq!(
            cgroup_memory_headroom_at(&current, &mount, CgroupMemoryVersion::V2),
            Some(500),
            "current has 500 bytes free, narrower than the parent's 600"
        );
        std::fs::write(current.join("memory.current"), "900\n").expect("current over limit");
        assert_eq!(
            cgroup_memory_headroom_at(&current, &mount, CgroupMemoryVersion::V2),
            Some(0),
            "usage above a limit saturates to exhausted headroom"
        );
        std::fs::remove_dir_all(&mount).expect("remove synthetic cgroup tree");
    }

    #[test]
    fn cgroup_headroom_requires_a_readable_usage_counter() {
        let unique = format!(
            "ny-cgroup-missing-usage-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let mount = std::env::temp_dir().join(unique);
        let current = mount.join("job");
        std::fs::create_dir_all(&current).expect("create synthetic cgroup tree");
        std::fs::write(current.join("memory.max"), "1000\n").expect("current limit");

        assert_eq!(
            cgroup_memory_headroom_at(&current, &mount, CgroupMemoryVersion::V2),
            None,
            "a missing usage counter must not be treated as zero usage"
        );
        std::fs::write(current.join("memory.current"), "malformed\n")
            .expect("malformed current usage");
        assert_eq!(
            cgroup_memory_headroom_at(&current, &mount, CgroupMemoryVersion::V2),
            None,
            "a malformed usage counter must not be treated as zero usage"
        );
        std::fs::remove_dir_all(&mount).expect("remove synthetic cgroup tree");
    }

    #[test]
    fn cgroup_headroom_uses_v1_when_v2_has_no_memory_envelope() {
        let unique = format!(
            "ny-cgroup-hybrid-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let base = std::env::temp_dir().join(unique);
        let v2_mount = base.join("v2");
        let v1_mount = base.join("v1");
        let v2_current = v2_mount.join("job");
        let v1_parent = v1_mount.join("parent");
        let v1_current = v1_parent.join("job");
        std::fs::create_dir_all(&v2_current).expect("create synthetic v2 tree");
        std::fs::create_dir_all(&v1_current).expect("create synthetic v1 tree");
        std::fs::write(v1_parent.join("memory.limit_in_bytes"), "1000\n").expect("v1 parent limit");
        std::fs::write(v1_parent.join("memory.usage_in_bytes"), "400\n").expect("v1 parent usage");
        std::fs::write(v1_current.join("memory.limit_in_bytes"), "800\n")
            .expect("v1 current limit");
        std::fs::write(v1_current.join("memory.usage_in_bytes"), "300\n")
            .expect("v1 current usage");

        let memberships = "0::/job\n10:memory,blkio:/parent/job\n";
        let mountinfo = format!(
            "39 29 0:33 / {} rw - cgroup2 cgroup rw\n\
             40 29 0:34 / {} rw - cgroup cgroup rw,memory\n",
            v2_mount.display(),
            v1_mount.display()
        );
        assert_eq!(
            cgroup_memory_headroom_from(memberships, &mountinfo),
            Some(500),
            "a present-but-unbounded v2 hierarchy must not mask v1 memory control"
        );

        std::fs::remove_dir_all(&base).expect("remove synthetic cgroup trees");
    }

    #[test]
    fn equally_specific_cgroup_mounts_resolve_deterministically() {
        let mountinfo = concat!(
            "39 29 0:33 /slice /sys/fs/cgroup/first rw - cgroup2 cgroup rw\n",
            "40 29 0:34 /slice /sys/fs/cgroup/second rw - cgroup2 cgroup rw\n",
        );
        assert_eq!(
            resolve_cgroup_mount(mountinfo, "/slice/job", CgroupMemoryVersion::V2),
            Some((
                PathBuf::from("/sys/fs/cgroup/first/job"),
                PathBuf::from("/sys/fs/cgroup/first")
            )),
            "mountinfo order breaks ties between equally specific roots"
        );
    }
}
