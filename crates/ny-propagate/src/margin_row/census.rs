// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #mr-census: an ALWAYS-ON, verdict-free cost census of the margin-row lane
//! (margin-row throughput plan, build B1(a)).
//!
//! # Why this exists
//!
//! Every tree-level speedup claimed for this lane so far converted a
//! STANDALONE pass wall into CONCURRENT-TREE throughput. That conversion is not
//! established at narrow pass widths (`par_min_len` runs an R=1 conv over a
//! tensor under 16Ki elements as ONE rayon task, so a standalone R=1 wall may
//! be serial wall rather than CPU work), and
//! nobody had measured whether the batch-synchronous tree loop leaves the pool
//! idle while a batch waits on its slowest domain. This module measures exactly
//! those quantities, on the scored route, in every run.
//!
//! # The one line
//!
//! Exactly ONE `[mr-census] ...` stderr line per lane invocation, printed on
//! EVERY stop path — a verdict, every `Unknown` reason, a fail-closed error,
//! and a panic (the line is emitted from a `Drop`, which runs during unwinding)
//! — not only at `TREE DONE`, which prints only when the closure ladder is
//! armed and only on the tree loop's normal exit. A lane that is still running
//! when the scored process `_exit`s (the detached CONCURRENT lane on every
//! timeout row) never reaches its `Drop`, so the exit path prints a SNAPSHOT
//! line for it ([`snapshot_if_live`], `outcome=snapshot`); the inline lane's
//! hard slice cap prints one from the abandoning thread as well. A row can
//! therefore carry more than one `[mr-census]` line (a cap snapshot, then an
//! exit snapshot or the worker's own final line): READ THE LAST ONE.
//!
//! Absence of the line therefore means "this binary lacks the census", never
//! "the lane stopped on a path the census forgot".
//!
//! # What each number is, and what it is NOT
//!
//! * `site.<name>=calls:sumR:wall_s:tcpu_s` — per engine call site: call count,
//!   total seed columns (R) across calls, summed wall, and summed CALLING-THREAD
//!   CPU. Walls of concurrent calls OVERLAP (a sum, not occupied wall). The
//!   thread-CPU figure is the CPU the calling thread consumed during the call;
//!   under rayon it INCLUDES unrelated jobs the caller stole while blocked in a
//!   join and EXCLUDES work other workers did for this call. It is a clean CPU
//!   figure only on a 1-thread pool — which is why the probe (`col-scaling-v2`)
//!   runs every arm at 1 thread as well as 20. Source: `/proc/thread-self/
//!   schedstat` (ns), read with `pread` on a per-thread cached descriptor;
//!   `na` when unavailable (non-Linux, the file is missing, or a kernel without
//!   sched_info that prints `0 0 0`). There is no `libc`/`rustix` dependency in
//!   this crate, so this is the std-only route. PRECISION: the kernel prints
//!   the thread's raw `sum_exec_runtime`, which for a RUNNING thread is only
//!   brought up to date at the next scheduler event (tick, switch), so each
//!   reading lags by up to one tick (1/CONFIG_HZ: 1-4 ms). Per-call error is
//!   therefore up to about one tick and roughly unbiased across many calls;
//!   treat per-site `tcpu` as a sum-level figure, never a per-call one
//!   (`clock_gettime(CLOCK_THREAD_CPUTIME_ID)` would be exact but needs a
//!   dependency this crate does not carry).
//! * `pool_util` = process CPU seconds over the tree / (tree wall x pool
//!   threads). The numerator is PROCESS-WIDE (`/proc/self/stat` utime+stime,
//!   converted at USER_HZ = 100 — the fixed userspace tick on Linux x86_64 and
//!   arm64; 10 ms granularity), so any other lane sharing the process (the
//!   internal verifier beside a CONCURRENT margin-row lane) inflates it.
//!   `cpu_src=` names the source; `pool_util=na` when unavailable.
//! * `straggler_s` = sum over frontier batches of (batch wall - MEDIAN domain
//!   wall): how long the batch-synchronous loop waited on its slowest domains.
//!   `slot_idle_frac` = 1 - (sum of domain walls) / (sum of batch wall x batch
//!   width): the fraction of domain-slot time spent waiting for the batch
//!   barrier. Measured on the default per-domain path only; stacked batches
//!   (`margin_row.domain_stack`) are counted separately.
//! * `tree_exp` / `tree_created` — the outermost trees' merged expansions and
//!   created child slots (the same increments as `BabStats`), so a SNAPSHOT
//!   line — a lane that never returned — still reports expansions at the wall.
//! * `close_*` — what closed each domain, counted in the deterministic serial
//!   merge of the OUTERMOST tree (nested epoch subtrees are one `close_epoch`):
//!   `eval` (the domain's own refreshed bound), `infeasible` (empty region at
//!   eval), `head_infeasible` (a head-split child slot with crossed bounds),
//!   `raw` (a child whose own scored bound closes), `lift` (a child whose raw
//!   score does NOT close but the parent lift does — the parent's own bound
//!   cannot close it there, it was checked at eval), `epoch` (a Tier-2 epoch
//!   subtree closed), `queued` (a popped entry whose queued bound already
//!   closed). Their sum equals the outermost trees' `closed` count.
//!
//! # Soundness
//!
//! Telemetry only. Nothing here is read by any bound, gate, closure decision or
//! verdict. Its only effect on a verdict path is timing: a few relaxed atomics,
//! two `Instant` reads and two `pread`s per engine call (engine calls cost
//! milliseconds), which is why it is identical in both arms of any A/B built
//! after it landed.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

use super::bab::MarginRowOutcome;

/// Census schema version printed as `v=` (bump on any field change).
/// v2 (#mr-stack, throughput plan B5): the `eval_stk` site and the `stk_*`
/// main-eval stacking fields.
const SCHEMA: u32 = 2;

/// Engine call sites (index = discriminant).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Site {
    /// Whole `root_eval` (includes its own `y_rows` and direct pass).
    RootEval = 0,
    /// Root alpha ascent evaluation pass (`alpha_opt::search_alpha`).
    RootAscent = 1,
    /// Root alpha ascent supergradient walks (mean/head supergradient).
    RootWalk = 2,
    /// y-row refresh: the 2 x n_y identity-seeded pass pair, in `bab::build_pack`
    /// (root, tree LRU misses, parallel prebuild) and in the root alpha ascent's
    /// frozen frame (`alpha_opt::search_alpha`).
    YRows = 3,
    /// Domain eval main certified pass (`eval_with_pack`).
    EvalMain = 4,
    /// #margin-row-beta-percol trial pass.
    BetaTrialPercol = 5,
    /// #margin-row-beta shared-vector trial pass.
    BetaTrialShared = 6,
    /// #margin-row-domain-alpha trial pass.
    AlphaTrial = 7,
    /// Batched candidate/child scoring pass (`score_candidates`).
    Score = 8,
    /// Cross-domain stacked scoring pass (`score_candidates_domain_stacked`).
    ScoreStacked = 9,
    /// β direction walk (`beta::step_scales` linearized walk).
    WalkBeta = 10,
    /// Domain-alpha direction walk (`alpha_opt::supergradient` from the tree).
    WalkAlpha = 11,
    /// #mr-stack (B5): ONE cross-domain stacked MAIN eval pass serving a
    /// frontier batch (`bab/stack_eval.rs`); `sumR` is the stacked width. The
    /// per-domain main passes it replaces are no longer counted under
    /// `eval_main`, which keeps only the per-domain passes (fallbacks and every
    /// non-stacked route).
    EvalStacked = 12,
}

const NSITE: usize = 13;

const SITE_NAMES: [&str; NSITE] = [
    "root_eval",
    "root_ascent",
    "root_walk",
    "y_rows",
    "eval_main",
    "beta_pc",
    "beta_shared",
    "dalpha",
    "score",
    "score_stk",
    "walk_beta",
    "walk_alpha",
    "eval_stk",
];

/// What closed a domain (index = discriminant). See the module docs.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum CloseKind {
    /// The domain's own refreshed bound closed it at eval.
    Eval = 0,
    /// Empty region discovered at eval.
    Infeasible = 1,
    /// A head-split child slot with crossed bounds.
    HeadInfeasible = 2,
    /// A child whose own scored (raw) bound closes.
    Raw = 3,
    /// A child closed only by the parent lift.
    Lift = 4,
    /// A Tier-2 epoch subtree closed the domain.
    Epoch = 5,
    /// A popped entry whose queued bound already closed.
    Queued = 6,
}

const NCLOSE: usize = 7;

const CLOSE_NAMES: [&str; NCLOSE] = [
    "eval",
    "infeasible",
    "head_infeasible",
    "raw",
    "lift",
    "epoch",
    "queued",
];

#[allow(clippy::declare_interior_mutable_const)]
const Z64: AtomicU64 = AtomicU64::new(0);

static SITE_CALLS: [AtomicU64; NSITE] = [Z64; NSITE];
static SITE_R: [AtomicU64; NSITE] = [Z64; NSITE];
static SITE_WALL_NS: [AtomicU64; NSITE] = [Z64; NSITE];
static SITE_TCPU_NS: [AtomicU64; NSITE] = [Z64; NSITE];
/// Calls whose thread-CPU could not be read (the site's `tcpu` prints `na`).
static SITE_TCPU_MISS: [AtomicU64; NSITE] = [Z64; NSITE];

static CLOSES: [AtomicU64; NCLOSE] = [Z64; NCLOSE];

static TREES: AtomicU64 = AtomicU64::new(0);
/// Expansions and created child slots merged by the OUTERMOST trees — the
/// same increments as `BabStats::{expansions, domains_created}`, kept here so a
/// SNAPSHOT line (a lane that never returned) still carries them.
static TREE_EXP: AtomicU64 = AtomicU64::new(0);
static TREE_CREATED: AtomicU64 = AtomicU64::new(0);
static TREE_WALL_NS: AtomicU64 = AtomicU64::new(0);
static TREE_CPU_NS: AtomicU64 = AtomicU64::new(0);
static TREE_CPU_MISS: AtomicU64 = AtomicU64::new(0);
static POOL_THREADS: AtomicU64 = AtomicU64::new(0);

static BATCHES: AtomicU64 = AtomicU64::new(0);
static BATCH_DOMAINS: AtomicU64 = AtomicU64::new(0);
static BATCH_WALL_NS: AtomicU64 = AtomicU64::new(0);
static STRAGGLER_NS: AtomicU64 = AtomicU64::new(0);
static DOMAIN_WALL_NS: AtomicU64 = AtomicU64::new(0);
static SLOT_WALL_NS: AtomicU64 = AtomicU64::new(0);
static STACKED_BATCHES: AtomicU64 = AtomicU64::new(0);
static STACKED_WALL_NS: AtomicU64 = AtomicU64::new(0);

/// Margin-row lanes currently inside [`LaneCensus`].
static LANES_LIVE: AtomicUsize = AtomicUsize::new(0);
/// Set when a lane began while another was still live: the process-global
/// counters then mix two lanes and the line prints `overlap=1` (per-lane
/// attribution VOID).
static OVERLAP: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// Nesting depth of tree loops on THIS thread. Nested Tier-2 epoch trees
    /// run on the outer tree's driver thread (the serial fill stage), so a
    /// thread-local depth identifies the outermost tree exactly.
    static TREE_DEPTH: Cell<usize> = const { Cell::new(0) };
}

fn reset() {
    for i in 0..NSITE {
        SITE_CALLS[i].store(0, Relaxed);
        SITE_R[i].store(0, Relaxed);
        SITE_WALL_NS[i].store(0, Relaxed);
        SITE_TCPU_NS[i].store(0, Relaxed);
        SITE_TCPU_MISS[i].store(0, Relaxed);
    }
    for c in &CLOSES {
        c.store(0, Relaxed);
    }
    for a in [
        &TREES,
        &TREE_EXP,
        &TREE_CREATED,
        &TREE_WALL_NS,
        &TREE_CPU_NS,
        &TREE_CPU_MISS,
        &POOL_THREADS,
        &BATCHES,
        &BATCH_DOMAINS,
        &BATCH_WALL_NS,
        &STRAGGLER_NS,
        &DOMAIN_WALL_NS,
        &SLOT_WALL_NS,
        &STACKED_BATCHES,
        &STACKED_WALL_NS,
    ] {
        a.store(0, Relaxed);
    }
    OVERLAP.store(false, Relaxed);
    super::bab::root_eval_reuse_reset();
    super::net::narrow_r_reset();
    super::net::errconv_screen_reset();
    super::bab::mr_stack_reset();
    super::errlane::reset();
    super::gpu_conv::reset();
    super::close_integrity::reset();
}

/// Whether a margin-row lane is currently inside its [`LaneCensus`] guard
/// (still running, or abandoned at its slice cap and still running).
#[must_use]
pub fn lane_live() -> bool {
    LANES_LIVE.load(Relaxed) > 0
}

/// Nanoseconds since `t0`, saturating into `u64`.
#[inline]
pub(crate) fn elapsed_ns(t0: Instant) -> u64 {
    u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

// --------------------------------------------------------------------------
//  CPU clocks (std-only; see the module docs for what they do and do not mean)
// --------------------------------------------------------------------------

/// Label of the process-CPU source printed as `cpu_src=`.
#[cfg(target_os = "linux")]
const CPU_SRC: &str = "proc_stat_userhz100";
#[cfg(not(target_os = "linux"))]
const CPU_SRC: &str = "none";

/// CPU nanoseconds consumed so far by the CALLING THREAD.
#[cfg(target_os = "linux")]
pub(crate) fn thread_cpu_ns() -> Option<u64> {
    use std::os::unix::fs::FileExt;
    thread_local! {
        // `None` = not yet opened; `Some(None)` = open failed (do not retry).
        #[expect(clippy::option_option, reason = "Distinguish unopened, failed open and cached descriptor")]
        static SCHED: std::cell::RefCell<Option<Option<std::fs::File>>> =
            const { std::cell::RefCell::new(None) };
    }
    SCHED.with(|cell| {
        let mut slot = cell.borrow_mut();
        let file = slot
            .get_or_insert_with(|| std::fs::File::open("/proc/thread-self/schedstat").ok())
            .as_ref()?;
        // `pread` at offset 0 regenerates a seq_file from its first record, so
        // one descriptor serves every read on this thread.
        let mut buf = [0u8; 96];
        let n = file.read_at(&mut buf, 0).ok()?;
        let text = std::str::from_utf8(buf.get(..n)?).ok()?;
        // A kernel built without sched_info prints the literal `0 0 0` here
        // (`proc_pid_schedstat`). A RUNNING thread cannot have consumed zero
        // CPU, so 0 means "unavailable" — reporting it as `tcpu=0.000` would be
        // a silent zero (an absence read as evidence), so it becomes `na`.
        text.split_ascii_whitespace()
            .next()?
            .parse::<u64>()
            .ok()
            .filter(|&ns| ns > 0)
    })
}

/// CPU nanoseconds consumed so far by the CALLING THREAD (unavailable here).
#[cfg(not(target_os = "linux"))]
pub(crate) fn thread_cpu_ns() -> Option<u64> {
    None
}

/// CPU nanoseconds consumed so far by the WHOLE PROCESS (all threads, dead
/// ones included), at USER_HZ = 100 tick granularity.
#[cfg(target_os = "linux")]
pub(crate) fn process_cpu_ns() -> Option<u64> {
    const NS_PER_TICK: u64 = 10_000_000; // USER_HZ = 100
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // Field 2 (`comm`) is parenthesised and may contain spaces; every field
    // after the LAST ')' is space-separated, starting at field 3 (`state`).
    let rest = stat.get(stat.rfind(')')? + 1..)?;
    let mut fields = rest.split_ascii_whitespace();
    // Field 14 (`utime`) is index 11 after `state`; field 15 (`stime`) follows.
    let utime: u64 = fields.nth(11)?.parse().ok()?;
    let stime: u64 = fields.next()?.parse().ok()?;
    utime.checked_add(stime)?.checked_mul(NS_PER_TICK)
}

/// CPU nanoseconds consumed so far by the whole process (unavailable here).
#[cfg(not(target_os = "linux"))]
pub(crate) fn process_cpu_ns() -> Option<u64> {
    None
}

// --------------------------------------------------------------------------
//  Per-site timer
// --------------------------------------------------------------------------

/// RAII engine-call-site timer: counts the call and its seed width on start,
/// adds wall and calling-thread CPU on drop (every exit path, `?` included).
pub(crate) struct SiteTimer {
    site: usize,
    t0: Instant,
    cpu0: Option<u64>,
}

impl SiteTimer {
    /// Start timing one call at `site` whose seed carries `r` columns.
    #[inline]
    pub(crate) fn start(site: Site, r: usize) -> Self {
        let idx = site as usize;
        SITE_CALLS[idx].fetch_add(1, Relaxed);
        SITE_R[idx].fetch_add(r as u64, Relaxed);
        Self {
            site: idx,
            cpu0: thread_cpu_ns(),
            t0: Instant::now(),
        }
    }
}

impl Drop for SiteTimer {
    #[inline]
    fn drop(&mut self) {
        SITE_WALL_NS[self.site].fetch_add(elapsed_ns(self.t0), Relaxed);
        match (self.cpu0, thread_cpu_ns()) {
            (Some(a), Some(b)) => {
                SITE_TCPU_NS[self.site].fetch_add(b.saturating_sub(a), Relaxed);
            }
            _ => {
                SITE_TCPU_MISS[self.site].fetch_add(1, Relaxed);
            }
        }
    }
}

// --------------------------------------------------------------------------
//  Tree accounting
// --------------------------------------------------------------------------

/// RAII tree-loop guard. Only the OUTERMOST tree on this thread records
/// (nested epoch subtrees would double-count wall that the outer tree already
/// spans); closure kinds and batch timings are gated on the same flag.
pub(crate) struct TreeGuard {
    outermost: bool,
    t0: Instant,
    cpu0: Option<u64>,
    threads: usize,
}

impl TreeGuard {
    /// Enter a tree loop.
    pub(crate) fn begin() -> Self {
        let depth = TREE_DEPTH.with(|d| {
            let v = d.get();
            d.set(v + 1);
            v
        });
        let outermost = depth == 0;
        Self {
            outermost,
            cpu0: if outermost { process_cpu_ns() } else { None },
            threads: rayon::current_num_threads().max(1),
            t0: Instant::now(),
        }
    }

    /// Record one per-domain frontier batch: its wall and each domain's own
    /// evaluate+expand wall.
    pub(crate) fn batch(&self, batch_ns: u64, domain_ns: &[u64]) {
        if !self.outermost {
            return;
        }
        let Some(terms) = batch_terms(batch_ns, domain_ns) else {
            return;
        };
        BATCHES.fetch_add(1, Relaxed);
        BATCH_DOMAINS.fetch_add(terms.width, Relaxed);
        BATCH_WALL_NS.fetch_add(batch_ns, Relaxed);
        STRAGGLER_NS.fetch_add(terms.straggler, Relaxed);
        DOMAIN_WALL_NS.fetch_add(terms.domain_total, Relaxed);
        SLOT_WALL_NS.fetch_add(terms.slot, Relaxed);
    }

    /// Whether this guard belongs to the OUTERMOST tree on its thread (the one
    /// whose once-per-lane receipts may be published).
    pub(crate) fn is_outermost(&self) -> bool {
        self.outermost
    }

    /// Record one cross-domain stacked batch (no per-domain walls exist: the
    /// score pass is shared).
    pub(crate) fn stacked_batch(&self, batch_ns: u64) {
        if !self.outermost {
            return;
        }
        STACKED_BATCHES.fetch_add(1, Relaxed);
        STACKED_WALL_NS.fetch_add(batch_ns, Relaxed);
    }

    /// Record one merged expansion and the child slots it created.
    pub(crate) fn expanded(&self, created: usize) {
        if self.outermost {
            TREE_EXP.fetch_add(1, Relaxed);
            TREE_CREATED.fetch_add(created as u64, Relaxed);
        }
    }

    /// Record `n` domains closed by `kind`.
    pub(crate) fn closed(&self, kind: CloseKind, n: usize) {
        if self.outermost && n > 0 {
            CLOSES[kind as usize].fetch_add(n as u64, Relaxed);
        }
    }
}

/// One batch's straggler terms (pure, so the arithmetic is unit-testable
/// without touching the process-global counters).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BatchTerms {
    width: u64,
    /// Batch wall minus the MEDIAN domain wall (upper median for even widths).
    straggler: u64,
    /// Sum of the domains' own walls.
    domain_total: u64,
    /// Batch wall x width: the domain-slot time the barrier held.
    slot: u64,
}

fn batch_terms(batch_ns: u64, domain_ns: &[u64]) -> Option<BatchTerms> {
    if domain_ns.is_empty() {
        return None;
    }
    let mut sorted = domain_ns.to_vec();
    sorted.sort_unstable();
    let median = sorted[sorted.len() / 2];
    let width = sorted.len() as u64;
    Some(BatchTerms {
        width,
        straggler: batch_ns.saturating_sub(median),
        domain_total: sorted.iter().fold(0u64, |acc, &v| acc.saturating_add(v)),
        slot: batch_ns.saturating_mul(width),
    })
}

impl Drop for TreeGuard {
    fn drop(&mut self) {
        TREE_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
        if !self.outermost {
            return;
        }
        TREES.fetch_add(1, Relaxed);
        TREE_WALL_NS.fetch_add(elapsed_ns(self.t0), Relaxed);
        POOL_THREADS.fetch_max(self.threads as u64, Relaxed);
        match (self.cpu0, process_cpu_ns()) {
            (Some(a), Some(b)) => {
                TREE_CPU_NS.fetch_add(b.saturating_sub(a), Relaxed);
            }
            _ => {
                TREE_CPU_MISS.fetch_add(1, Relaxed);
            }
        }
    }
}

// --------------------------------------------------------------------------
//  Lane guard + the line
// --------------------------------------------------------------------------

/// RAII lane guard: resets the counters when the lane starts and prints the
/// `[mr-census]` line when it is dropped — on every return path and during a
/// panic unwind.
pub(crate) struct LaneCensus {
    t0: Instant,
    outcome: Option<String>,
}

impl LaneCensus {
    /// Enter a lane invocation.
    pub(crate) fn begin() -> Self {
        if LANES_LIVE.fetch_add(1, Relaxed) == 0 {
            reset();
        } else {
            OVERLAP.store(true, Relaxed);
        }
        Self {
            t0: Instant::now(),
            outcome: None,
        }
    }

    /// Record the lane's outcome (printed by `Drop`).
    pub(crate) fn finish(&mut self, out: &MarginRowOutcome) {
        self.outcome = Some(outcome_fields(out));
    }
}

impl Drop for LaneCensus {
    fn drop(&mut self) {
        LANES_LIVE.fetch_sub(1, Relaxed);
        let outcome = self
            .outcome
            .take()
            .unwrap_or_else(|| "outcome=none stop=no_outcome_recorded(panic_or_early_drop)".into());
        eprintln!("{}", render(&outcome, Some(elapsed_ns(self.t0))));
        for extra in stop_lines() {
            eprintln!("{extra}");
        }
    }
}

/// [`snapshot_line`] only when a lane is still live (its own `Drop` has not
/// printed yet). The scored process calls this on its way out
/// (`vnncomp::exit_scored_instance_without_teardown`): a DETACHED lane that is
/// still running at `_exit` — the concurrent lane on every timeout row, whose
/// final join grace expires with the worker busy — would otherwise leave NO
/// census line. `None` (silent) when no lane is live.
pub fn snapshot_if_live(label: &str) -> Option<String> {
    (LANES_LIVE.load(Relaxed) > 0).then(|| snapshot_line(label))
}

/// A SNAPSHOT of the census for a lane that is still running (the hard slice
/// cap abandoned its worker, or the process is exiting around it). Safe from
/// any thread; reads relaxed atomics.
pub fn snapshot_line(label: &str) -> String {
    render(&format!("outcome=snapshot stop={}", sanitize(label)), None)
}

fn sanitize(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| {
            if c.is_whitespace() || c == '=' {
                '_'
            } else {
                c
            }
        })
        .collect();
    if out.len() > 96 {
        let mut cut = 96;
        while !out.is_char_boundary(cut) {
            cut -= 1;
        }
        out.truncate(cut);
    }
    if out.is_empty() {
        out.push('-');
    }
    out
}

fn outcome_fields(out: &MarginRowOutcome) -> String {
    match out {
        MarginRowOutcome::Unsat(stats, _) => format!(
            "outcome=unsat stop={} exp={} created={} closed={}",
            sanitize(&stats.stop),
            stats.expansions,
            stats.domains_created,
            stats.closed
        ),
        MarginRowOutcome::Unknown { reason, stats } => match stats {
            Some(stats) => format!(
                "outcome=unknown stop={} exp={} created={} closed={}",
                sanitize(reason),
                stats.expansions,
                stats.domains_created,
                stats.closed
            ),
            None => format!(
                "outcome=unknown stop={} exp=0 created=0 closed=0",
                sanitize(reason)
            ),
        },
    }
}

/// Extra lines printed at stop after the `[mr-census]` line (one per
/// mechanism that owns a stop receipt): `[root-eval-reuse] hits= ...` (B1(c)),
/// `[narrow-r] floor= ... calls= engaged=` (B1(d)), `[mr-errconv] screen=row-floor calls= fast_blocks=
/// slow_blocks=` (#mr-errconv-screen), `[mr-stack] main_eval K= nf= blocks= ...` (B5), `[mr-errlane] done
/// ...` (B3) and `[mr-close] min_close= ...` (#mr-close-telemetry). Each is
/// present on every stop of a binary that carries the mechanism, zero or not,
/// so its absence identifies a binary without it.
///
/// `pub` because the SNAPSHOT paths must print them too: a lane still running
/// at `_exit` (the detached concurrent worker on every TIMEOUT row) or
/// abandoned at its slice cap never reaches the guard's `Drop`, so without the
/// callers of [`snapshot_if_live`] / [`snapshot_line`] echoing these lines a
/// timeout row carries NO `[mr-errlane] done` receipt — and the B3 audit
/// numbers (`aud_*`) exist only there, so every timeout row would be VOID.
pub fn stop_lines() -> Vec<String> {
    vec![
        super::bab::root_eval_reuse_stop_line(),
        super::net::narrow_r_stop_line(),
        super::net::errconv_screen_stop_line(),
        super::errlane::stop_line(),
        super::bab::mr_stack_stop_line(),
        super::gpu_conv::stop_line(),
        super::close_integrity::stop_line(),
    ]
}

/// Extra `key=value` fields appended to the `[mr-census]` line.
fn extra_fields() -> String {
    let (narrow_calls, narrow_engaged) = super::net::narrow_r_counters();
    format!(
        "{} narrow_calls={narrow_calls} narrow_engaged={narrow_engaged}{}{}{}",
        super::bab::root_eval_reuse_fields(),
        super::bab::mr_stack_fields(),
        super::errlane::census_fields(),
        super::gpu_conv::census_fields()
    )
}

fn secs(ns: u64) -> f64 {
    ns as f64 / 1e9
}

fn render(outcome: &str, lane_ns: Option<u64>) -> String {
    let get = |a: &AtomicU64| a.load(Relaxed);
    let tree_ns = get(&TREE_WALL_NS);
    let threads = get(&POOL_THREADS);
    let tree_cpu = get(&TREE_CPU_NS);
    let cpu_ok = get(&TREES) > 0 && get(&TREE_CPU_MISS) == 0;
    let pool_util = if cpu_ok && tree_ns > 0 && threads > 0 {
        format!("{:.3}", tree_cpu as f64 / (tree_ns as f64 * threads as f64))
    } else {
        "na".into()
    };
    let batches = get(&BATCHES);
    let batch_ns = get(&BATCH_WALL_NS);
    let slot_ns = get(&SLOT_WALL_NS);
    let straggler_ns = get(&STRAGGLER_NS);
    let frac = |num: u64, den: u64| {
        if den > 0 {
            format!("{:.3}", num as f64 / den as f64)
        } else {
            "na".into()
        }
    };
    let width_mean = if batches > 0 {
        format!("{:.2}", get(&BATCH_DOMAINS) as f64 / batches as f64)
    } else {
        "na".into()
    };
    let slot_idle = if slot_ns > 0 {
        format!("{:.3}", 1.0 - get(&DOMAIN_WALL_NS) as f64 / slot_ns as f64)
    } else {
        "na".into()
    };
    let lane = lane_ns.map_or_else(|| "running".to_string(), |ns| format!("{:.3}", secs(ns)));
    let root = lane_ns.map_or_else(
        || "na".to_string(),
        |ns| format!("{:.3}", secs(ns.saturating_sub(tree_ns))),
    );
    let mut line = format!(
        "[mr-census] v={SCHEMA} {outcome} overlap={} lane_s={lane} pre_tree_s={root} \
         trees={} tree_exp={} tree_created={} tree_s={:.3} pool_threads={threads} \
         tree_cpu_s={} pool_util={pool_util} \
         cpu_src={CPU_SRC} batches={batches} width_mean={width_mean} batch_s={:.3} \
         straggler_s={:.3} straggler_frac={} slot_idle_frac={slot_idle} \
         stacked_batches={} stacked_s={:.3}",
        u8::from(OVERLAP.load(Relaxed)),
        get(&TREES),
        get(&TREE_EXP),
        get(&TREE_CREATED),
        secs(tree_ns),
        if cpu_ok {
            format!("{:.3}", secs(tree_cpu))
        } else {
            "na".into()
        },
        secs(batch_ns),
        secs(straggler_ns),
        frac(straggler_ns, batch_ns),
        get(&STACKED_BATCHES),
        secs(get(&STACKED_WALL_NS)),
    );
    let mut close_sum = 0u64;
    for (i, name) in CLOSE_NAMES.iter().enumerate() {
        let v = CLOSES[i].load(Relaxed);
        close_sum = close_sum.saturating_add(v);
        line.push_str(&format!(" close_{name}={v}"));
    }
    line.push_str(&format!(" close_sum={close_sum}"));
    for (i, name) in SITE_NAMES.iter().enumerate() {
        let calls = SITE_CALLS[i].load(Relaxed);
        let tcpu = if calls > 0 && SITE_TCPU_MISS[i].load(Relaxed) > 0 {
            "na".to_string()
        } else {
            format!("{:.3}", secs(SITE_TCPU_NS[i].load(Relaxed)))
        };
        line.push_str(&format!(
            " site.{name}={calls}:{}:{:.3}:{tcpu}",
            SITE_R[i].load(Relaxed),
            secs(SITE_WALL_NS[i].load(Relaxed)),
        ));
    }
    line.push_str(&extra_fields());
    line
}

#[cfg(test)]
mod census_tests {
    use super::*;

    /// `mr_census_line_has_fixed_schema`: every site and every close kind is
    /// printed on every line (a fixed schema is what lets a grep distinguish
    /// "zero" from "this binary lacks the field").
    #[test]
    fn mr_census_line_has_fixed_schema() {
        let line = snapshot_line("unit test");
        assert!(line.starts_with("[mr-census] v="), "{line}");
        assert!(line.contains("outcome=snapshot stop=unit_test"), "{line}");
        for name in SITE_NAMES {
            assert!(
                line.contains(&format!(" site.{name}=")),
                "missing site {name}: {line}"
            );
        }
        for name in CLOSE_NAMES {
            assert!(
                line.contains(&format!(" close_{name}=")),
                "missing close {name}: {line}"
            );
        }
        for key in [
            " tree_exp=",
            " tree_created=",
            " pool_util=",
            " straggler_frac=",
            " slot_idle_frac=",
            " cpu_src=",
            " reuse_hits=",
            " narrow_calls=",
            " narrow_engaged=",
            " site.eval_stk=",
            " stk_passes=",
            " stk_blocks=",
            " stk_cols=",
            " stk_fallback_beta_pc=",
            " errlane_mode=",
            " errlane_env_passes=",
            " errlane_pc_passes=",
            " gpuconv_armed=",
            " gpuconv_gpu=",
            " gpuconv_fallback=",
        ] {
            assert!(line.contains(key), "missing {key}: {line}");
        }
        assert!(!line.contains('\n'), "the census must be ONE line");
    }

    /// `mr_census_straggler_is_batch_minus_median`: the straggler term is the
    /// batch wall minus the MEDIAN domain wall, and the slot term is width x
    /// batch wall. Also proves a NESTED tree guard does not record (it would
    /// double-count wall the outer tree already spans).
    #[test]
    fn mr_census_straggler_is_batch_minus_median() {
        // Median of [1, 2, 9] is 2 -> straggler = 10 - 2 = 8; slot = 3 x 10.
        assert_eq!(
            batch_terms(10, &[9, 1, 2]),
            Some(BatchTerms {
                width: 3,
                straggler: 8,
                domain_total: 12,
                slot: 30,
            })
        );
        // Even width: the UPPER median (index len/2).
        assert_eq!(batch_terms(10, &[4, 1, 3, 2]).map(|t| t.straggler), Some(7));
        // A batch slower than its own wall reading cannot go negative.
        assert_eq!(batch_terms(1, &[5]).map(|t| t.straggler), Some(0));
        assert_eq!(batch_terms(10, &[]), None);
        // Fresh thread: the thread-local depth starts at zero there regardless
        // of what this test thread did before.
        std::thread::spawn(|| {
            let outer = TreeGuard::begin();
            assert!(outer.outermost, "the first tree on a thread is outermost");
            {
                let inner = TreeGuard::begin();
                assert!(!inner.outermost, "a nested tree must not record");
            }
            let again = TreeGuard::begin();
            assert!(
                !again.outermost,
                "depth must be restored by the inner drop, not reset"
            );
        })
        .join()
        .expect("census thread");
    }

    /// `mr_census_sanitize_keeps_one_token`: a stop reason with spaces and
    /// `=` must stay a single `key=value` token.
    #[test]
    fn mr_census_sanitize_keeps_one_token() {
        assert_eq!(sanitize("fail-closed: a = b"), "fail-closed:_a___b");
        assert_eq!(sanitize(""), "-");
        assert!(sanitize(&"x".repeat(500)).len() <= 96);
    }
}
