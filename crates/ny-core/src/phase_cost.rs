// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #phase-cost — the per-phase COST CENSUS of one scored run. Observability ONLY.
//!
//! Why this exists. The kFSB discovery — two thirds of every yolo BaB domain
//! spent choosing WHICH neuron to split (a 21125x21125 identity-seeded prescore
//! plus ~20 full alpha/beta optimizer runs of child simulation) — was found by a
//! hand-read `-vv` timeline of one row. Nothing in the bank could have shown it:
//! the flight record says which lanes ran, never what they cost. This module is
//! the missing ledger: per phase, the wall it occupied, the thread time it
//! burned, how often it was entered and what it processed, and the RSS observed
//! when it exited — recorded with cheap monotonic timers at phase boundaries
//! that already exist, and serialized into `<results>.flight.json` by ny-cli.
//!
//! # What a number means
//!
//! * `wall_s` — OCCUPIED wall: the union of the intervals during which at least
//!   one thread was inside the phase. Inclusive of nested phases, so
//!   `bab_loop` contains `branch_select`, which may contain `branch_sim`, which
//!   may contain `beta_opt`. A share of the row's wall is read directly off it.
//! * `thread_s` — inclusive time summed over threads (`thread_s / wall_s` is the
//!   phase's mean parallelism). A same-phase guard entered while the phase is
//!   already open ON THE SAME THREAD is a re-entry: counted, never re-timed, so
//!   recursion cannot double the total.
//! * `self_s` — `thread_s` minus the time of OTHER phases nested inside it on
//!   the same thread. A phase whose work fans out over rayon keeps the workers'
//!   time out of its own `self_s` only if the workers enter a phase; otherwise
//!   the wait is charged to the driver's `self_s`.
//! * `edges` — parent -> child inclusive thread time, where the parent is the
//!   innermost phase open on the SAME thread (`root` when none). Work a rayon
//!   worker does for a driver phase is attributed to `root`: cross-thread
//!   parentage is not propagated.
//! * `rss_peak_kb` — the largest resident set observed when this phase (or a
//!   phase nested in it on the same thread) EXITED, sampled at most every
//!   [`COARSE_RSS_PERIOD_US`] at the coarse level and at every exit at the
//!   detail level. It is a lower bound on the phase's true peak.
//!
//! # Levels
//!
//! [`DetailLevel::Coarse`] is the default and is what a scored run records:
//! timers at batch/phase granularity plus rate-limited RSS samples. Its cost
//! budget is < 0.5% of wall, MEASURED (the guard's per-enter cost x the
//! recorded enter counts — `docs/PHASE_COST_CENSUS_*.md`).
//! [`DetailLevel::Detail`] additionally times [`enter_fine`] sites and samples
//! RSS at EVERY exit; it is opt-in (`ny vnncomp v1 --phase-cost detail`) and
//! may cost more. [`DetailLevel::Off`] records nothing: every entry point is
//! then one relaxed atomic load.
//!
//! # Contract (enforced by review + the tests below + the bit-identity test in
//! ny-propagate)
//!
//! * NOTHING reads these cells to make a decision. No verdict, bound, deadline,
//!   gate or control-flow branch may depend on them; the only readers are the
//!   flight sidecar, the `[phase-cost]` stderr line and tests. The recorder
//!   writes no bound and reads no bound, so it cannot move a verdict except
//!   through the wall clock it consumes (the < 0.5% budget above).
//! * The write path never allocates, never locks, never formats and reads no
//!   environment. The only I/O is the rate-limited `/proc/self/statm` read.
//! * Process-global and monotonic (one instance per scored process, like the
//!   flight recorder). Tests build their own [`PhaseCostRecorder`].

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// Wire schema of [`PhaseCostSnapshot`]. Bump on any field change; the census
/// aggregator (`scripts/costcensus/`) refuses a schema it does not know.
pub const SCHEMA: &str = "ny-core/phase-cost/v1";

/// RSS sampling period at the coarse level, in microseconds.
pub const COARSE_RSS_PERIOD_US: u64 = 100_000;

/// The phases of a run. Ordering is stable and load-bearing only for the
/// serialized order; new variants must be APPENDED (the discriminant indexes
/// the cell arrays) and [`CostPhase::ALL`] / [`NPHASE`] extended with them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum CostPhase {
    /// Model load + lowering to the verifier's graph + property parse.
    Load = 0,
    /// Root bounding: the root IBP/CROWN/alpha evaluation and root gates.
    RootBounds = 1,
    /// An alpha optimizer run (graph alpha-CROWN loop, margin-row alpha ascent).
    AlphaOpt = 2,
    /// A beta optimizer run (graph beta ascent, joint alpha/beta, margin-row beta).
    BetaOpt = 3,
    /// Intermediate bound refresh / re-derivation for a domain or batch.
    IntermRefresh = 4,
    /// Choosing the split for a domain or a wave (umbrella of prescore + sim).
    BranchSelect = 5,
    /// Cheap per-candidate scoring of split candidates (BaBSR/kFSB prescore).
    BranchPrescore = 6,
    /// Child SIMULATION of shortlisted candidates (kFSB lookahead bounding).
    BranchSim = 7,
    /// Bounding the children a split produced (the BaB's main pass).
    ChildBound = 8,
    /// A whole BaB loop (umbrella; root bounding is outside it where the code
    /// separates the two).
    BabLoop = 9,
    /// A falsification / PGD lane.
    Attack = 10,
    /// MIP / LP solving.
    Mip = 11,
    /// The low-dimensional prover lane.
    LowDim = 12,
    /// Certification of a candidate verdict (cert re-verification, proof checks).
    Certify = 13,
    /// Writing RESULTS_FILE (verdict + witness).
    Publish = 14,
    /// The verifier invocation as a whole (umbrella over every lane the
    /// scored command dispatches to the beta-CROWN entry): what it contains
    /// beyond its instrumented sub-phases is an uninstrumented lane.
    Verify = 15,
    /// A structural EXACT driver: cell enumeration, the fractional-head
    /// driver, the nonlinear-VNN-LIB sat/unsat lanes.
    Exact = 16,
}

/// Number of [`CostPhase`] variants.
pub const NPHASE: usize = 17;

impl CostPhase {
    /// Every phase, in discriminant order.
    pub const ALL: [CostPhase; NPHASE] = [
        CostPhase::Load,
        CostPhase::RootBounds,
        CostPhase::AlphaOpt,
        CostPhase::BetaOpt,
        CostPhase::IntermRefresh,
        CostPhase::BranchSelect,
        CostPhase::BranchPrescore,
        CostPhase::BranchSim,
        CostPhase::ChildBound,
        CostPhase::BabLoop,
        CostPhase::Attack,
        CostPhase::Mip,
        CostPhase::LowDim,
        CostPhase::Certify,
        CostPhase::Publish,
        CostPhase::Verify,
        CostPhase::Exact,
    ];

    /// Stable snake_case wire name (the census aggregator keys on it).
    pub const fn as_str(self) -> &'static str {
        match self {
            CostPhase::Load => "load",
            CostPhase::RootBounds => "root_bounds",
            CostPhase::AlphaOpt => "alpha_opt",
            CostPhase::BetaOpt => "beta_opt",
            CostPhase::IntermRefresh => "interm_refresh",
            CostPhase::BranchSelect => "branch_select",
            CostPhase::BranchPrescore => "branch_prescore",
            CostPhase::BranchSim => "branch_sim",
            CostPhase::ChildBound => "child_bound",
            CostPhase::BabLoop => "bab_loop",
            CostPhase::Attack => "attack",
            CostPhase::Mip => "mip",
            CostPhase::LowDim => "lowdim",
            CostPhase::Certify => "certify",
            CostPhase::Publish => "publish",
            CostPhase::Verify => "verify",
            CostPhase::Exact => "exact",
        }
    }

    /// What one [`add_units`] unit of this phase is.
    pub const fn unit(self) -> &'static str {
        match self {
            CostPhase::Load => "models",
            CostPhase::RootBounds => "roots",
            CostPhase::AlphaOpt | CostPhase::BetaOpt => "iterations",
            CostPhase::IntermRefresh => "domains",
            CostPhase::BranchSelect => "domains",
            CostPhase::BranchPrescore => "candidates",
            CostPhase::BranchSim => "simulations",
            CostPhase::ChildBound => "domains",
            CostPhase::BabLoop => "domains",
            CostPhase::Attack => "runs",
            CostPhase::Mip => "solves",
            CostPhase::LowDim => "boxes",
            CostPhase::Certify => "checks",
            CostPhase::Publish => "files",
            CostPhase::Verify => "invocations",
            CostPhase::Exact => "attempts",
        }
    }
}

/// Run-level work counters (the denominators of "cost per useful unit").
/// Append-only, like [`CostPhase`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum CostCounter {
    /// Domains a BaB lane expanded/explored (graph BaB `domains_explored`,
    /// margin-row expansions).
    DomainsExplored = 0,
    /// Domains a BaB lane CLOSED (proved), graph `domains_verified`, margin-row
    /// `closed`.
    DomainsVerified = 1,
    /// Domains closed as infeasible (graph BaB).
    DomainsInfeasible = 2,
    /// BaB lifecycles that ended (one per `[bab-stop]` / `TREE DONE`).
    BabRuns = 3,
    /// Split candidates scored by a branching prescore.
    CandidatesScored = 4,
    /// Candidate children bounded by a branching simulation.
    CandidatesSimulated = 5,
    /// Child domains created by splits.
    ChildrenCreated = 6,
    /// Falsification lanes entered.
    AttackRuns = 7,
}

/// Number of [`CostCounter`] variants.
pub const NCOUNTER: usize = 8;

impl CostCounter {
    /// Every counter, in discriminant order.
    pub const ALL: [CostCounter; NCOUNTER] = [
        CostCounter::DomainsExplored,
        CostCounter::DomainsVerified,
        CostCounter::DomainsInfeasible,
        CostCounter::BabRuns,
        CostCounter::CandidatesScored,
        CostCounter::CandidatesSimulated,
        CostCounter::ChildrenCreated,
        CostCounter::AttackRuns,
    ];

    /// Stable snake_case wire name.
    pub const fn as_str(self) -> &'static str {
        match self {
            CostCounter::DomainsExplored => "domains_explored",
            CostCounter::DomainsVerified => "domains_verified",
            CostCounter::DomainsInfeasible => "domains_infeasible",
            CostCounter::BabRuns => "bab_runs",
            CostCounter::CandidatesScored => "candidates_scored",
            CostCounter::CandidatesSimulated => "candidates_simulated",
            CostCounter::ChildrenCreated => "children_created",
            CostCounter::AttackRuns => "attack_runs",
        }
    }
}

/// How much the recorder records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum DetailLevel {
    /// Nothing is recorded; every entry point is one relaxed atomic load.
    Off = 0,
    /// The default: phase timers at batch granularity, counters, RSS sampled
    /// at most every [`COARSE_RSS_PERIOD_US`].
    Coarse = 1,
    /// Coarse plus [`enter_fine`] sites plus an RSS sample at every exit.
    Detail = 2,
}

/// The level a scored run records at when nothing selects another.
pub const DEFAULT_LEVEL: DetailLevel = DetailLevel::Coarse;

impl DetailLevel {
    /// Parse `off` / `coarse` / `detail` (exactly; a typo is an error, never a
    /// silent default).
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "off" => Ok(DetailLevel::Off),
            "coarse" => Ok(DetailLevel::Coarse),
            "detail" => Ok(DetailLevel::Detail),
            other => Err(format!(
                "phase-cost level must be off, coarse or detail (got {other:?})"
            )),
        }
    }

    /// Stable wire token.
    pub const fn token(self) -> &'static str {
        match self {
            DetailLevel::Off => "off",
            DetailLevel::Coarse => "coarse",
            DetailLevel::Detail => "detail",
        }
    }

    const fn from_u8(raw: u8) -> Self {
        match raw {
            0 => DetailLevel::Off,
            1 => DetailLevel::Coarse,
            _ => DetailLevel::Detail,
        }
    }
}

/// Packed occupancy word: in-flight count in the top 16 bits, the start of the
/// currently open occupied interval (microseconds since the anchor) in the low
/// 48 bits. One CAS word makes "first in opens, last out closes" exact under
/// concurrency: a closer can never pair a newer opener's timestamp with an
/// older interval.
const OCC_TS_BITS: u32 = 48;
const OCC_TS_MASK: u64 = (1u64 << OCC_TS_BITS) - 1;
const OCC_COUNT_MAX: u64 = 0xFFFF;

/// "Never" for the first-enter cell.
const NEVER: u64 = u64::MAX;

struct PhaseCells {
    occ: AtomicU64,
    wall_us: AtomicU64,
    thread_ns: AtomicU64,
    self_ns: AtomicU64,
    enters: AtomicU64,
    reentries: AtomicU64,
    units: AtomicU64,
    max_inflight: AtomicU64,
    first_enter_us: AtomicU64,
    last_exit_us: AtomicU64,
    rss_peak_kb: AtomicU64,
}

impl PhaseCells {
    const fn new() -> Self {
        Self {
            occ: AtomicU64::new(0),
            wall_us: AtomicU64::new(0),
            thread_ns: AtomicU64::new(0),
            self_ns: AtomicU64::new(0),
            enters: AtomicU64::new(0),
            reentries: AtomicU64::new(0),
            units: AtomicU64::new(0),
            max_inflight: AtomicU64::new(0),
            first_enter_us: AtomicU64::new(NEVER),
            last_exit_us: AtomicU64::new(0),
            rss_peak_kb: AtomicU64::new(0),
        }
    }
}

/// Row index of the "no parent on this thread" edge source.
const ROOT_ROW: usize = NPHASE;

/// One recorder. Production uses the process-global [`global`]; tests build
/// their own so parallel tests never share cells.
pub struct PhaseCostRecorder {
    level: AtomicU8,
    phases: [PhaseCells; NPHASE],
    edges: [[AtomicU64; NPHASE]; NPHASE + 1],
    counters: [AtomicU64; NCOUNTER],
    rss_last_us: AtomicU64,
    rss_unavailable: AtomicBool,
    rss_samples: AtomicU64,
    rss_peak_kb: AtomicU64,
    unbalanced: AtomicU64,
    stack_overflow: AtomicU64,
    anchor: OnceLock<Instant>,
}

impl Default for PhaseCostRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Maximum same-thread nesting depth tracked for self time and edges. Deeper
/// guards still time their phase; they just do not subtract from a parent.
const MAX_DEPTH: usize = 16;

#[derive(Clone, Copy)]
struct Frame {
    rec: usize,
    phase: u8,
    child_ns: u64,
}

const EMPTY_FRAME: Frame = Frame {
    rec: 0,
    phase: 0,
    child_ns: 0,
};

struct Stack {
    depth: usize,
    frames: [Frame; MAX_DEPTH],
}

thread_local! {
    static STACK: RefCell<Stack> = const {
        RefCell::new(Stack {
            depth: 0,
            frames: [EMPTY_FRAME; MAX_DEPTH],
        })
    };
}

/// A slot index meaning "this guard pushed no frame".
const NO_FRAME: u8 = u8::MAX;

/// RAII guard for one phase interval. Inert (and free) when the level is off,
/// for a fine phase below the detail level, and for a same-thread re-entry.
///
/// `!Send`: a guard must drop on the thread that created it, which is what
/// keeps the per-thread nesting stack exact.
#[must_use = "a phase guard times the scope it is bound to; bind it to a named `_guard`"]
pub struct PhaseGuard<'a> {
    rec: Option<&'a PhaseCostRecorder>,
    phase: u8,
    slot: u8,
    start: Option<Instant>,
    _not_send: PhantomData<*const ()>,
}

impl PhaseGuard<'_> {
    const fn inert() -> Self {
        Self {
            rec: None,
            phase: 0,
            slot: NO_FRAME,
            start: None,
            _not_send: PhantomData,
        }
    }
}

impl PhaseGuard<'_> {
    /// Whether this guard is timing (false when off, below the detail level
    /// for a fine site, or a same-thread re-entry).
    pub const fn is_armed(&self) -> bool {
        self.rec.is_some()
    }
}

/// A `Send`, STACK-NEUTRAL phase interval: for a phase whose timer is stored in
/// a value that may be dropped on another thread (the margin-row lane's
/// `EvalNode` timer rides inside a domain evaluation that rayon may finish on a
/// different worker). It times occupied wall, thread time and enters exactly
/// like [`PhaseGuard`], but it neither pushes a frame nor reads the thread's
/// stack: its `self_s` equals its `thread_s`, it is never an edge parent, its
/// own edge is recorded under `root`, and a same-thread re-entry is not
/// detected (so a span nested in its own phase double-counts `thread_s`; the
/// occupied wall stays exact).
#[must_use = "a phase span times the scope it is bound to; bind it to a named `_span`"]
pub struct PhaseSpan<'a> {
    rec: Option<&'a PhaseCostRecorder>,
    phase: u8,
    start: Option<Instant>,
}

impl PhaseSpan<'_> {
    /// An inert span (records nothing).
    pub const fn inert() -> Self {
        Self {
            rec: None,
            phase: 0,
            start: None,
        }
    }

    /// Whether this span is timing.
    pub const fn is_armed(&self) -> bool {
        self.rec.is_some()
    }
}

impl Drop for PhaseSpan<'_> {
    #[inline]
    fn drop(&mut self) {
        if let (Some(rec), Some(start)) = (self.rec, self.start) {
            rec.exit_span(self.phase, start);
        }
    }
}

impl PhaseCostRecorder {
    /// A fresh recorder at [`DEFAULT_LEVEL`].
    pub const fn new() -> Self {
        Self {
            level: AtomicU8::new(DEFAULT_LEVEL as u8),
            phases: [const { PhaseCells::new() }; NPHASE],
            edges: [const { [const { AtomicU64::new(0) }; NPHASE] }; NPHASE + 1],
            counters: [const { AtomicU64::new(0) }; NCOUNTER],
            rss_last_us: AtomicU64::new(NEVER),
            rss_unavailable: AtomicBool::new(false),
            rss_samples: AtomicU64::new(0),
            rss_peak_kb: AtomicU64::new(0),
            unbalanced: AtomicU64::new(0),
            stack_overflow: AtomicU64::new(0),
            anchor: OnceLock::new(),
        }
    }

    /// Select the level. Takes effect for guards entered afterwards.
    pub fn set_level(&self, level: DetailLevel) {
        self.level.store(level as u8, Ordering::Relaxed);
    }

    /// The current level.
    pub fn level(&self) -> DetailLevel {
        DetailLevel::from_u8(self.level.load(Ordering::Relaxed))
    }

    /// Anchor the recorder's clock (idempotent: the first anchor wins). Every
    /// `*_s` offset in a snapshot is relative to it.
    pub fn begin(&self) {
        let _ = self.anchor();
    }

    fn anchor(&self) -> Instant {
        *self.anchor.get_or_init(Instant::now)
    }

    fn micros(&self, now: Instant) -> u64 {
        let us = now.saturating_duration_since(self.anchor()).as_micros();
        u64::try_from(us).unwrap_or(OCC_TS_MASK).min(OCC_TS_MASK)
    }

    /// Enter `phase`; the returned guard closes the interval when dropped.
    #[inline]
    pub fn enter(&self, phase: CostPhase) -> PhaseGuard<'_> {
        if self.level.load(Ordering::Relaxed) == DetailLevel::Off as u8 {
            return PhaseGuard::inert();
        }
        self.enter_armed(phase)
    }

    /// As [`Self::enter`], but recorded only at [`DetailLevel::Detail`]: for
    /// sites too hot for the default budget.
    #[inline]
    pub fn enter_fine(&self, phase: CostPhase) -> PhaseGuard<'_> {
        if self.level.load(Ordering::Relaxed) < DetailLevel::Detail as u8 {
            return PhaseGuard::inert();
        }
        self.enter_armed(phase)
    }

    fn enter_armed(&self, phase: CostPhase) -> PhaseGuard<'_> {
        let me = std::ptr::from_ref(self) as usize;
        let idx = phase as usize;
        let cells = &self.phases[idx];
        // Push a frame unless the phase is already open on this thread.
        let pushed = STACK
            .try_with(|stack| {
                let mut stack = stack.borrow_mut();
                let depth = stack.depth;
                if stack.frames[..depth.min(MAX_DEPTH)]
                    .iter()
                    .any(|f| f.rec == me && f.phase == phase as u8)
                {
                    return None;
                }
                if depth >= MAX_DEPTH {
                    stack.depth = depth + 1;
                    return Some(NO_FRAME);
                }
                stack.frames[depth] = Frame {
                    rec: me,
                    phase: phase as u8,
                    child_ns: 0,
                };
                stack.depth = depth + 1;
                Some(depth as u8)
            })
            .unwrap_or(Some(NO_FRAME));
        let Some(slot) = pushed else {
            // Same-thread re-entry: counted, never re-timed.
            cells.enters.fetch_add(1, Ordering::Relaxed);
            cells.reentries.fetch_add(1, Ordering::Relaxed);
            return PhaseGuard::inert();
        };
        if slot == NO_FRAME {
            self.stack_overflow.fetch_add(1, Ordering::Relaxed);
        }
        let now = Instant::now();
        let us = self.micros(now);
        let inflight = occ_enter(&cells.occ, us);
        cells.max_inflight.fetch_max(inflight, Ordering::Relaxed);
        cells.first_enter_us.fetch_min(us, Ordering::Relaxed);
        cells.enters.fetch_add(1, Ordering::Relaxed);
        PhaseGuard {
            rec: Some(self),
            phase: phase as u8,
            slot,
            start: Some(now),
            _not_send: PhantomData,
        }
    }

    /// Open a [`PhaseSpan`] (inert when the level is off).
    #[inline]
    pub fn span(&self, phase: CostPhase) -> PhaseSpan<'_> {
        if self.level.load(Ordering::Relaxed) == DetailLevel::Off as u8 {
            return PhaseSpan::inert();
        }
        let cells = &self.phases[phase as usize];
        let now = Instant::now();
        let us = self.micros(now);
        let inflight = occ_enter(&cells.occ, us);
        cells.max_inflight.fetch_max(inflight, Ordering::Relaxed);
        cells.first_enter_us.fetch_min(us, Ordering::Relaxed);
        cells.enters.fetch_add(1, Ordering::Relaxed);
        PhaseSpan {
            rec: Some(self),
            phase: phase as u8,
            start: Some(now),
        }
    }

    fn exit_span(&self, span_phase: u8, start: Instant) {
        let idx = usize::from(span_phase);
        let cells = &self.phases[idx];
        let now = Instant::now();
        let elapsed_ns =
            u64::try_from(now.saturating_duration_since(start).as_nanos()).unwrap_or(u64::MAX);
        cells.thread_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        cells.self_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        self.edges[ROOT_ROW][idx].fetch_add(elapsed_ns, Ordering::Relaxed);
        let us = self.micros(now);
        if let Some(interval) = occ_exit(&cells.occ, us) {
            cells.wall_us.fetch_add(interval, Ordering::Relaxed);
        }
        cells.last_exit_us.fetch_max(us, Ordering::Relaxed);
        self.maybe_sample_rss(us, idx);
    }

    fn exit(&self, guard_phase: u8, slot: u8, start: Instant) {
        let me = std::ptr::from_ref(self) as usize;
        let idx = usize::from(guard_phase);
        let cells = &self.phases[idx];
        let now = Instant::now();
        let elapsed_ns =
            u64::try_from(now.saturating_duration_since(start).as_nanos()).unwrap_or(u64::MAX);
        // Pop our frame and charge our elapsed time to the parent frame.
        let popped = STACK
            .try_with(|stack| {
                let mut stack = stack.borrow_mut();
                if slot == NO_FRAME {
                    // An overflowed guard: unwind the depth only.
                    stack.depth = stack.depth.saturating_sub(1);
                    return (0u64, None, true);
                }
                let at = usize::from(slot);
                let depth = stack.depth;
                if depth != at + 1 {
                    // Dropped out of order (fail-soft): count it, and never
                    // GROW the stack back over a frame already unwound.
                    self.unbalanced.fetch_add(1, Ordering::Relaxed);
                }
                if at >= depth {
                    return (0u64, None, true);
                }
                let child_ns = stack.frames[at].child_ns;
                stack.depth = at;
                let parent = if at > 0 {
                    let parent = &mut stack.frames[at - 1];
                    if parent.rec == me {
                        parent.child_ns = parent.child_ns.saturating_add(elapsed_ns);
                        Some(usize::from(parent.phase))
                    } else {
                        None
                    }
                } else {
                    None
                };
                (child_ns, parent, true)
            })
            .unwrap_or((0, None, false));
        let (child_ns, parent, _) = popped;
        cells.thread_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        cells
            .self_ns
            .fetch_add(elapsed_ns.saturating_sub(child_ns), Ordering::Relaxed);
        self.edges[parent.unwrap_or(ROOT_ROW)][idx].fetch_add(elapsed_ns, Ordering::Relaxed);
        let us = self.micros(now);
        if let Some(interval) = occ_exit(&cells.occ, us) {
            cells.wall_us.fetch_add(interval, Ordering::Relaxed);
        }
        cells.last_exit_us.fetch_max(us, Ordering::Relaxed);
        self.maybe_sample_rss(us, idx);
    }

    fn maybe_sample_rss(&self, now_us: u64, idx: usize) {
        if self.rss_unavailable.load(Ordering::Relaxed) {
            return;
        }
        let period = if self.level.load(Ordering::Relaxed) >= DetailLevel::Detail as u8 {
            0
        } else {
            COARSE_RSS_PERIOD_US
        };
        let last = self.rss_last_us.load(Ordering::Relaxed);
        if last != NEVER && now_us < last.saturating_add(period) {
            return;
        }
        // One sampler per period: whoever moves the stamp samples.
        if self
            .rss_last_us
            .compare_exchange(last, now_us, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return;
        }
        match rss_kb() {
            Some(kb) => self.note_rss_kb(idx, kb),
            // No statm (not Linux): stop trying, so a host without /proc pays
            // one failed open per process, not one per exit.
            None => self.rss_unavailable.store(true, Ordering::Relaxed),
        }
    }

    /// Record one RSS observation for `idx` and every phase of this recorder
    /// open on the calling thread (they contain the exit being sampled).
    fn note_rss_kb(&self, idx: usize, kb: u64) {
        let me = std::ptr::from_ref(self) as usize;
        self.rss_samples.fetch_add(1, Ordering::Relaxed);
        self.rss_peak_kb.fetch_max(kb, Ordering::Relaxed);
        self.phases[idx]
            .rss_peak_kb
            .fetch_max(kb, Ordering::Relaxed);
        let _ = STACK.try_with(|stack| {
            let stack = stack.borrow();
            for frame in &stack.frames[..stack.depth.min(MAX_DEPTH)] {
                if frame.rec == me {
                    self.phases[usize::from(frame.phase)]
                        .rss_peak_kb
                        .fetch_max(kb, Ordering::Relaxed);
                }
            }
        });
    }

    /// Add `n` work units (see [`CostPhase::unit`]) to `phase`.
    #[inline]
    pub fn add_units(&self, phase: CostPhase, n: u64) {
        if self.level.load(Ordering::Relaxed) != DetailLevel::Off as u8 {
            self.phases[phase as usize]
                .units
                .fetch_add(n, Ordering::Relaxed);
        }
    }

    /// Add `n` to a run-level counter.
    #[inline]
    pub fn count(&self, counter: CostCounter, n: u64) {
        if self.level.load(Ordering::Relaxed) != DetailLevel::Off as u8 {
            self.counters[counter as usize].fetch_add(n, Ordering::Relaxed);
        }
    }

    /// The census as of now. `read_proc` adds the process high-water mark
    /// (`/proc/self/status` `VmHWM`); the deadline watchdog passes `false`.
    pub fn snapshot(&self, read_proc: bool) -> PhaseCostSnapshot {
        let now = Instant::now();
        let now_us = self.micros(now);
        let secs = |us: u64| us as f64 / 1e6;
        let mut phases = Vec::new();
        for phase in CostPhase::ALL {
            let cells = &self.phases[phase as usize];
            let enters = cells.enters.load(Ordering::Relaxed);
            if enters == 0 {
                continue;
            }
            let occ = cells.occ.load(Ordering::Relaxed);
            let open = occ >> OCC_TS_BITS;
            let open_us = if open > 0 {
                now_us.saturating_sub(occ & OCC_TS_MASK)
            } else {
                0
            };
            let first = cells.first_enter_us.load(Ordering::Relaxed);
            let last = cells.last_exit_us.load(Ordering::Relaxed);
            let rss = cells.rss_peak_kb.load(Ordering::Relaxed);
            phases.push(PhaseCostRow {
                phase: phase.as_str(),
                wall_s: secs(
                    cells
                        .wall_us
                        .load(Ordering::Relaxed)
                        .saturating_add(open_us),
                ),
                thread_s: cells.thread_ns.load(Ordering::Relaxed) as f64 / 1e9,
                self_s: cells.self_ns.load(Ordering::Relaxed) as f64 / 1e9,
                enters,
                reentries: cells.reentries.load(Ordering::Relaxed),
                units: cells.units.load(Ordering::Relaxed),
                unit: phase.unit(),
                max_inflight: cells.max_inflight.load(Ordering::Relaxed),
                open,
                first_enter_s: (first != NEVER).then(|| secs(first)),
                last_exit_s: (last > 0).then(|| secs(last)),
                rss_peak_kb: (rss > 0).then_some(rss),
            });
        }
        let mut edges = Vec::new();
        for (row, cells) in self.edges.iter().enumerate() {
            for (col, cell) in cells.iter().enumerate() {
                let ns = cell.load(Ordering::Relaxed);
                if ns == 0 {
                    continue;
                }
                edges.push(PhaseEdge {
                    parent: if row == ROOT_ROW {
                        "root"
                    } else {
                        CostPhase::ALL[row].as_str()
                    },
                    child: CostPhase::ALL[col].as_str(),
                    thread_s: ns as f64 / 1e9,
                });
            }
        }
        let mut counters = BTreeMap::new();
        for counter in CostCounter::ALL {
            let value = self.counters[counter as usize].load(Ordering::Relaxed);
            if value > 0 {
                counters.insert(counter.as_str(), value);
            }
        }
        let peak = self.rss_peak_kb.load(Ordering::Relaxed);
        PhaseCostSnapshot {
            schema: SCHEMA,
            level: self.level().token(),
            elapsed_s: secs(now_us),
            phases,
            edges,
            counters,
            rss_samples: self.rss_samples.load(Ordering::Relaxed),
            rss_peak_kb: (peak > 0).then_some(peak),
            process_hwm_kb: if read_proc { process_hwm_kb() } else { None },
            unbalanced: self.unbalanced.load(Ordering::Relaxed),
            stack_overflow: self.stack_overflow.load(Ordering::Relaxed),
        }
    }
}

impl Drop for PhaseGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        if let (Some(rec), Some(start)) = (self.rec, self.start) {
            rec.exit(self.phase, self.slot, start);
        }
    }
}

/// Register one entry in the occupancy word; returns the in-flight count
/// INCLUDING this entry.
fn occ_enter(occ: &AtomicU64, us: u64) -> u64 {
    let mut cur = occ.load(Ordering::Relaxed);
    loop {
        let count = cur >> OCC_TS_BITS;
        let next = if count == 0 {
            (1u64 << OCC_TS_BITS) | (us & OCC_TS_MASK)
        } else {
            ((count + 1).min(OCC_COUNT_MAX) << OCC_TS_BITS) | (cur & OCC_TS_MASK)
        };
        match occ.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return (count + 1).min(OCC_COUNT_MAX),
            Err(seen) => cur = seen,
        }
    }
}

/// Register one exit; returns the closed occupied interval (microseconds) when
/// this was the last thread inside the phase.
fn occ_exit(occ: &AtomicU64, us: u64) -> Option<u64> {
    let mut cur = occ.load(Ordering::Relaxed);
    loop {
        let count = cur >> OCC_TS_BITS;
        if count == 0 {
            return None;
        }
        let opened = cur & OCC_TS_MASK;
        let next = if count == 1 {
            0
        } else {
            ((count - 1) << OCC_TS_BITS) | opened
        };
        match occ.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return (count == 1).then(|| us.saturating_sub(opened)),
            Err(seen) => cur = seen,
        }
    }
}

/// Resident set size in KiB from `/proc/self/statm` (field 2, pages) times the
/// page size from `/proc/self/auxv` (`AT_PAGESZ`, read once). `None` off Linux
/// or on any read/parse failure — the recorder then simply has no RSS.
fn rss_kb() -> Option<u64> {
    use std::io::Read;
    let mut buf = [0u8; 128];
    let mut file = std::fs::File::open("/proc/self/statm").ok()?;
    let n = file.read(&mut buf).ok()?;
    let text = std::str::from_utf8(&buf[..n]).ok()?;
    let pages: u64 = text.split_ascii_whitespace().nth(1)?.parse().ok()?;
    Some(pages.saturating_mul(page_size_bytes()) / 1024)
}

fn page_size_bytes() -> u64 {
    static PAGE: OnceLock<u64> = OnceLock::new();
    *PAGE.get_or_init(|| auxv_page_size().unwrap_or(4096))
}

/// `AT_PAGESZ` (type 6) from the native-endian `(u64 type, u64 value)` pairs of
/// `/proc/self/auxv`.
fn auxv_page_size() -> Option<u64> {
    const AT_PAGESZ: u64 = 6;
    let bytes = std::fs::read("/proc/self/auxv").ok()?;
    for pair in bytes.as_chunks::<16>().0 {
        let key = u64::from_ne_bytes(pair[..8].try_into().ok()?);
        let value = u64::from_ne_bytes(pair[8..].try_into().ok()?);
        if key == AT_PAGESZ && value.is_power_of_two() {
            return Some(value);
        }
        if key == 0 {
            break;
        }
    }
    None
}

/// `VmHWM` (KiB) from `/proc/self/status`.
fn process_hwm_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    parse_status_kb(&status, "VmHWM:")
}

fn parse_status_kb(status: &str, key: &str) -> Option<u64> {
    status
        .lines()
        .find_map(|line| line.strip_prefix(key))
        .and_then(|rest| rest.split_ascii_whitespace().next())
        .and_then(|v| v.parse().ok())
}

/// One phase's census row. See the module docs for what each number means.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct PhaseCostRow {
    /// [`CostPhase::as_str`].
    pub phase: &'static str,
    /// Occupied wall, seconds (union across threads; an interval still open at
    /// the snapshot is included up to the snapshot).
    pub wall_s: f64,
    /// Inclusive thread time summed over threads, seconds (closed guards only).
    pub thread_s: f64,
    /// Thread time minus same-thread nested phases, seconds.
    pub self_s: f64,
    /// Guards entered (re-entries included).
    pub enters: u64,
    /// Same-thread re-entries (counted, not timed).
    pub reentries: u64,
    /// Work units reported via `add_units`.
    pub units: u64,
    /// What a unit is.
    pub unit: &'static str,
    /// Largest number of threads inside the phase at once.
    pub max_inflight: u64,
    /// Guards still open at the snapshot.
    pub open: u64,
    /// First entry, seconds since the anchor.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_enter_s: Option<f64>,
    /// Last exit, seconds since the anchor.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_exit_s: Option<f64>,
    /// Largest RSS sampled at an exit inside this phase (KiB).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rss_peak_kb: Option<u64>,
}

/// Inclusive thread time of `child` entered while `parent` was the innermost
/// open phase on the same thread (`root`: none).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct PhaseEdge {
    /// Parent phase name or `root`.
    pub parent: &'static str,
    /// Child phase name.
    pub child: &'static str,
    /// Seconds.
    pub thread_s: f64,
}

/// The serialized census (the flight record's `phase_cost` field).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct PhaseCostSnapshot {
    /// [`SCHEMA`].
    pub schema: &'static str,
    /// [`DetailLevel::token`].
    pub level: &'static str,
    /// Seconds since the anchor at the snapshot.
    pub elapsed_s: f64,
    /// Phases entered at least once, in [`CostPhase`] order.
    pub phases: Vec<PhaseCostRow>,
    /// Nonzero parent -> child edges.
    pub edges: Vec<PhaseEdge>,
    /// Nonzero counters by [`CostCounter::as_str`].
    pub counters: BTreeMap<&'static str, u64>,
    /// RSS samples taken.
    pub rss_samples: u64,
    /// Largest sampled RSS (KiB).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rss_peak_kb: Option<u64>,
    /// Process `VmHWM` at the snapshot (KiB).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub process_hwm_kb: Option<u64>,
    /// Guards dropped out of LIFO order (should be 0).
    pub unbalanced: u64,
    /// Guards deeper than the tracked nesting depth (timed, not nested).
    pub stack_overflow: u64,
}

impl PhaseCostSnapshot {
    /// The one-line `[phase-cost]` stderr observable: level, elapsed, the five
    /// largest phases by occupied wall, total enters, peak RSS.
    pub fn summary_line(&self) -> String {
        let mut by_wall: Vec<&PhaseCostRow> = self.phases.iter().collect();
        by_wall.sort_by(|a, b| b.wall_s.total_cmp(&a.wall_s));
        let top: Vec<String> = by_wall
            .iter()
            .take(5)
            .map(|row| format!("{}:{:.2}", row.phase, row.wall_s))
            .collect();
        let enters: u64 = self.phases.iter().map(|row| row.enters).sum();
        let explored = self.counters.get("domains_explored").copied().unwrap_or(0);
        let verified = self.counters.get("domains_verified").copied().unwrap_or(0);
        format!(
            "[phase-cost] level={} elapsed_s={:.2} phases={} enters={} top={} \
             domains_explored={} domains_verified={} rss_peak_mb={}",
            self.level,
            self.elapsed_s,
            self.phases.len(),
            enters,
            if top.is_empty() {
                "none".to_string()
            } else {
                top.join(",")
            },
            explored,
            verified,
            self.rss_peak_kb.map_or_else(
                || "na".to_string(),
                |kb| format!("{:.0}", kb as f64 / 1024.0)
            ),
        )
    }
}

/// The process-global recorder the scored path writes through.
pub fn global() -> &'static PhaseCostRecorder {
    static GLOBAL: PhaseCostRecorder = PhaseCostRecorder::new();
    &GLOBAL
}

/// [`PhaseCostRecorder::enter`] on the global recorder.
#[inline]
pub fn enter(phase: CostPhase) -> PhaseGuard<'static> {
    global().enter(phase)
}

/// [`PhaseCostRecorder::span`] on the global recorder.
#[inline]
pub fn span(phase: CostPhase) -> PhaseSpan<'static> {
    global().span(phase)
}

/// [`PhaseCostRecorder::enter_fine`] on the global recorder.
#[inline]
pub fn enter_fine(phase: CostPhase) -> PhaseGuard<'static> {
    global().enter_fine(phase)
}

/// [`PhaseCostRecorder::add_units`] on the global recorder.
#[inline]
pub fn add_units(phase: CostPhase, n: u64) {
    global().add_units(phase, n);
}

/// [`PhaseCostRecorder::count`] on the global recorder.
#[inline]
pub fn count(counter: CostCounter, n: u64) {
    global().count(counter, n);
}

/// [`PhaseCostRecorder::set_level`] on the global recorder.
pub fn set_level(level: DetailLevel) {
    global().set_level(level);
}

/// [`PhaseCostRecorder::level`] of the global recorder.
pub fn level() -> DetailLevel {
    global().level()
}

/// [`PhaseCostRecorder::begin`] on the global recorder.
pub fn begin() {
    global().begin();
}

/// [`PhaseCostRecorder::snapshot`] of the global recorder.
pub fn snapshot(read_proc: bool) -> PhaseCostSnapshot {
    global().snapshot(read_proc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn spin(d: Duration) {
        let t = Instant::now();
        while t.elapsed() < d {
            std::hint::spin_loop();
        }
    }

    fn row<'a>(snap: &'a PhaseCostSnapshot, name: &str) -> &'a PhaseCostRow {
        snap.phases
            .iter()
            .find(|r| r.phase == name)
            .unwrap_or_else(|| panic!("phase {name} absent from {snap:?}"))
    }

    #[test]
    fn phase_cost_wire_names_are_unique_and_cover_every_variant() {
        let mut seen = std::collections::BTreeSet::new();
        for (i, phase) in CostPhase::ALL.iter().enumerate() {
            assert_eq!(*phase as usize, i, "ALL must be in discriminant order");
            assert!(
                seen.insert(phase.as_str()),
                "duplicate name {}",
                phase.as_str()
            );
        }
        assert_eq!(seen.len(), NPHASE);
        let mut seen = std::collections::BTreeSet::new();
        for (i, counter) in CostCounter::ALL.iter().enumerate() {
            assert_eq!(*counter as usize, i);
            assert!(seen.insert(counter.as_str()));
        }
        assert_eq!(seen.len(), NCOUNTER);
        assert!(!seen.contains("root"), "`root` is the edge source sentinel");
    }

    #[test]
    fn phase_cost_level_parse_is_exact_and_default_is_coarse() {
        assert_eq!(DetailLevel::parse("off"), Ok(DetailLevel::Off));
        assert_eq!(DetailLevel::parse("coarse"), Ok(DetailLevel::Coarse));
        assert_eq!(DetailLevel::parse("detail"), Ok(DetailLevel::Detail));
        assert!(DetailLevel::parse("Coarse").is_err());
        assert!(DetailLevel::parse("").is_err());
        assert_eq!(DEFAULT_LEVEL, DetailLevel::Coarse);
        assert_eq!(PhaseCostRecorder::new().level(), DetailLevel::Coarse);
    }

    #[test]
    fn phase_cost_off_records_nothing() {
        let rec = PhaseCostRecorder::new();
        rec.set_level(DetailLevel::Off);
        {
            let _g = rec.enter(CostPhase::BabLoop);
            let _f = rec.enter_fine(CostPhase::ChildBound);
            rec.add_units(CostPhase::BabLoop, 7);
            rec.count(CostCounter::DomainsExplored, 3);
            spin(Duration::from_millis(1));
        }
        let snap = rec.snapshot(false);
        assert!(snap.phases.is_empty(), "{snap:?}");
        assert_eq!(snap.edges.len(), 0);
        assert!(snap.counters.is_empty());
        assert_eq!(snap.rss_samples, 0);
        assert_eq!(snap.level, "off");
    }

    #[test]
    fn phase_cost_nesting_splits_self_time_and_records_the_edge() {
        let rec = PhaseCostRecorder::new();
        rec.begin();
        {
            let _outer = rec.enter(CostPhase::BranchSelect);
            spin(Duration::from_millis(4));
            {
                let _inner = rec.enter(CostPhase::BranchSim);
                spin(Duration::from_millis(8));
                rec.add_units(CostPhase::BranchSim, 5);
            }
            spin(Duration::from_millis(2));
        }
        let snap = rec.snapshot(false);
        let outer = row(&snap, "branch_select");
        let inner = row(&snap, "branch_sim");
        assert_eq!((outer.enters, inner.enters), (1, 1));
        assert_eq!(inner.units, 5);
        assert_eq!(inner.unit, "simulations");
        // Inclusive outer >= 14 ms; inner >= 8 ms; outer self excludes inner.
        assert!(outer.thread_s >= 0.014, "{outer:?}");
        assert!(inner.thread_s >= 0.008, "{inner:?}");
        assert!(
            outer.self_s >= 0.006 && outer.self_s < outer.thread_s - 0.007,
            "{outer:?}"
        );
        assert!((inner.self_s - inner.thread_s).abs() < 1e-9, "{inner:?}");
        // Single thread: occupied wall == thread time (to the microsecond).
        assert!((outer.wall_s - outer.thread_s).abs() < 2e-3, "{outer:?}");
        assert_eq!(outer.max_inflight, 1);
        assert_eq!((outer.open, inner.open), (0, 0));
        let edge = |p: &str, c: &str| {
            snap.edges
                .iter()
                .find(|e| e.parent == p && e.child == c)
                .map(|e| e.thread_s)
        };
        assert!(edge("root", "branch_select").is_some());
        assert!(edge("branch_select", "branch_sim").unwrap_or(0.0) >= 0.008);
        assert!(edge("root", "branch_sim").is_none());
        assert_eq!(snap.unbalanced, 0);
        assert!(inner.first_enter_s.unwrap() >= outer.first_enter_s.unwrap());
        assert!(outer.last_exit_s.unwrap() >= inner.last_exit_s.unwrap());
    }

    #[test]
    fn phase_cost_same_thread_reentry_is_counted_not_double_timed() {
        let rec = PhaseCostRecorder::new();
        {
            let _a = rec.enter(CostPhase::Attack);
            spin(Duration::from_millis(3));
            {
                let _b = rec.enter(CostPhase::Attack);
                spin(Duration::from_millis(3));
            }
        }
        let snap = rec.snapshot(false);
        let attack = row(&snap, "attack");
        assert_eq!(attack.enters, 2);
        assert_eq!(attack.reentries, 1);
        assert!(
            attack.thread_s >= 0.006 && attack.thread_s < 0.0115,
            "{attack:?}"
        );
    }

    #[test]
    fn phase_cost_occupied_wall_is_the_union_across_threads() {
        let rec = PhaseCostRecorder::new();
        // Four threads inside ChildBound at once for ~20 ms each: thread time
        // ~80 ms, occupied wall ~20 ms (not 80).
        let barrier = std::sync::Barrier::new(4);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    barrier.wait();
                    let _g = rec.enter(CostPhase::ChildBound);
                    spin(Duration::from_millis(20));
                });
            }
        });
        let snap = rec.snapshot(false);
        let cb = row(&snap, "child_bound");
        assert_eq!(cb.enters, 4);
        assert!(cb.thread_s >= 0.080, "{cb:?}");
        assert!(
            cb.wall_s >= 0.020 && cb.wall_s < cb.thread_s * 0.75,
            "{cb:?}"
        );
        assert!(cb.max_inflight >= 2, "{cb:?}");
        assert_eq!(cb.open, 0);
        // Worker-thread entries have no same-thread parent.
        assert!(snap
            .edges
            .iter()
            .all(|e| e.parent == "root" && e.child == "child_bound"));
    }

    #[test]
    fn phase_cost_disjoint_intervals_sum_and_an_open_interval_is_counted() {
        let rec = PhaseCostRecorder::new();
        for _ in 0..3 {
            let _g = rec.enter(CostPhase::BetaOpt);
            spin(Duration::from_millis(2));
            rec.add_units(CostPhase::BetaOpt, 10);
        }
        let open = rec.enter(CostPhase::BabLoop);
        spin(Duration::from_millis(5));
        let snap = rec.snapshot(false);
        drop(open);
        let beta = row(&snap, "beta_opt");
        assert_eq!((beta.enters, beta.units), (3, 30));
        assert!(beta.wall_s >= 0.006, "{beta:?}");
        let bab = row(&snap, "bab_loop");
        assert_eq!(bab.open, 1);
        assert!(
            bab.wall_s >= 0.005,
            "an open interval must count up to the snapshot: {bab:?}"
        );
        assert_eq!(bab.thread_s, 0.0, "thread time counts closed guards only");
    }

    #[test]
    fn phase_cost_span_is_send_stack_neutral_and_times_across_threads() {
        fn assert_send<T: Send>(_: &T) {}
        let rec = PhaseCostRecorder::new();
        let outer = rec.enter(CostPhase::BabLoop);
        // Opened on this thread, dropped on another.
        let span = rec.span(CostPhase::ChildBound);
        assert_send(&span);
        assert!(span.is_armed() && outer.is_armed());
        std::thread::scope(|scope| {
            scope.spawn(move || {
                spin(Duration::from_millis(3));
                drop(span);
            });
        });
        drop(outer);
        let snap = rec.snapshot(false);
        let cb = row(&snap, "child_bound");
        assert_eq!((cb.enters, cb.open), (1, 0));
        assert!(
            cb.thread_s >= 0.003 && (cb.self_s - cb.thread_s).abs() < 1e-12,
            "{cb:?}"
        );
        // Stack-neutral: the span is not nested under bab_loop, and bab_loop's
        // self time was not reduced by it.
        assert!(snap
            .edges
            .iter()
            .any(|e| e.parent == "root" && e.child == "child_bound"));
        assert!(!snap.edges.iter().any(|e| e.parent == "bab_loop"));
        let bab = row(&snap, "bab_loop");
        assert!((bab.self_s - bab.thread_s).abs() < 1e-12, "{bab:?}");
        assert_eq!(snap.unbalanced, 0);
        rec.set_level(DetailLevel::Off);
        assert!(!rec.span(CostPhase::Attack).is_armed());
    }

    #[test]
    fn phase_cost_fine_sites_record_only_at_detail() {
        let rec = PhaseCostRecorder::new();
        {
            let _f = rec.enter_fine(CostPhase::IntermRefresh);
        }
        assert_eq!(rec.snapshot(false).phases.len(), 0);
        rec.set_level(DetailLevel::Detail);
        {
            let _f = rec.enter_fine(CostPhase::IntermRefresh);
        }
        let snap = rec.snapshot(false);
        assert_eq!(row(&snap, "interm_refresh").enters, 1);
        assert_eq!(snap.level, "detail");
    }

    #[test]
    fn phase_cost_counters_and_out_of_order_drops_are_fail_soft() {
        let rec = PhaseCostRecorder::new();
        rec.count(CostCounter::DomainsExplored, 4);
        rec.count(CostCounter::DomainsExplored, 6);
        rec.count(CostCounter::DomainsVerified, 3);
        let a = rec.enter(CostPhase::RootBounds);
        let b = rec.enter(CostPhase::AlphaOpt);
        drop(a); // out of order
        drop(b);
        let snap = rec.snapshot(false);
        assert_eq!(snap.counters.get("domains_explored"), Some(&10));
        assert_eq!(snap.counters.get("domains_verified"), Some(&3));
        assert!(snap.unbalanced >= 1);
        assert_eq!(row(&snap, "root_bounds").open, 0);
        assert_eq!(row(&snap, "alpha_opt").open, 0);
        // The stack is usable again afterwards.
        {
            let _c = rec.enter(CostPhase::Publish);
        }
        assert_eq!(row(&rec.snapshot(false), "publish").enters, 1);
    }

    #[test]
    fn phase_cost_deep_nesting_beyond_the_tracked_depth_still_times() {
        // Two recorders share one thread's stack, so 2 x NPHASE distinct frames
        // exceed MAX_DEPTH: the overflowing guards still time their phase.
        let a = PhaseCostRecorder::new();
        let b = PhaseCostRecorder::new();
        let mut guards = Vec::new();
        for phase in CostPhase::ALL {
            guards.push(a.enter(phase));
        }
        for phase in CostPhase::ALL {
            guards.push(b.enter(phase));
        }
        while let Some(g) = guards.pop() {
            drop(g);
        }
        let (sa, sb) = (a.snapshot(false), b.snapshot(false));
        assert_eq!(sa.phases.len(), NPHASE);
        assert_eq!(sb.phases.len(), NPHASE);
        assert_eq!(sa.stack_overflow as usize, NPHASE.saturating_sub(MAX_DEPTH));
        assert_eq!(
            (sa.stack_overflow + sb.stack_overflow) as usize,
            2 * NPHASE - MAX_DEPTH
        );
        assert_eq!(sa.unbalanced + sb.unbalanced, 0);
        assert!(sa
            .phases
            .iter()
            .chain(&sb.phases)
            .all(|r| r.open == 0 && r.enters == 1));
        // The stack unwound to empty: a fresh guard nests at the root again.
        {
            let _g = a.enter(CostPhase::Publish);
        }
        assert!(a
            .snapshot(false)
            .edges
            .iter()
            .any(|e| e.parent == "root" && e.child == "publish"));
    }

    #[test]
    fn phase_cost_rss_is_sampled_and_rate_limited_at_coarse() {
        if rss_kb().is_none() {
            eprintln!("SKIP: no /proc/self/statm on this host");
            return;
        }
        let rec = PhaseCostRecorder::new();
        for _ in 0..50 {
            let _g = rec.enter(CostPhase::ChildBound);
        }
        let snap = rec.snapshot(false);
        // The first exit samples; the next 49 fall inside the 100 ms period.
        assert_eq!(snap.rss_samples, 1, "{snap:?}");
        assert!(row(&snap, "child_bound").rss_peak_kb.unwrap_or(0) > 0);
        rec.set_level(DetailLevel::Detail);
        for _ in 0..5 {
            let _g = rec.enter(CostPhase::ChildBound);
        }
        assert_eq!(rec.snapshot(false).rss_samples, 6);
    }

    #[test]
    fn phase_cost_status_parse_and_page_size_are_sane() {
        let status = "Name:\tny\nVmPeak:\t  100 kB\nVmHWM:\t   2048 kB\nVmRSS:\t 1024 kB\n";
        assert_eq!(parse_status_kb(status, "VmHWM:"), Some(2048));
        assert_eq!(parse_status_kb(status, "VmSwap:"), None);
        let page = page_size_bytes();
        assert!(page.is_power_of_two() && page >= 4096, "page size {page}");
    }

    #[test]
    fn phase_cost_snapshot_serializes_with_the_schema_and_a_summary_line() {
        let rec = PhaseCostRecorder::new();
        {
            let _g = rec.enter(CostPhase::Load);
            rec.add_units(CostPhase::Load, 1);
        }
        rec.count(CostCounter::BabRuns, 1);
        let snap = rec.snapshot(true);
        let json = serde_json::to_value(&snap).expect("serializes");
        assert_eq!(json["schema"], SCHEMA);
        assert_eq!(json["level"], "coarse");
        assert_eq!(json["phases"][0]["phase"], "load");
        assert_eq!(json["phases"][0]["unit"], "models");
        assert_eq!(json["counters"]["bab_runs"], 1);
        let line = snap.summary_line();
        assert!(line.starts_with("[phase-cost] level=coarse "), "{line}");
        assert!(line.contains(" top=load:"), "{line}");
        let empty = PhaseCostRecorder::new().snapshot(false).summary_line();
        assert!(empty.contains(" top=none "), "{empty}");
    }

    /// Per-enter cost of an armed guard on this host, printed for the census's
    /// overhead bound (`--nocapture`, ideally `--release`). The assertion is
    /// deliberately loose (a debug build on a loaded box must still pass); the
    /// printed figure is the measurement.
    #[test]
    fn phase_cost_guard_cost_is_printed() {
        let rec = PhaseCostRecorder::new();
        const N: u32 = 200_000;
        let t = Instant::now();
        for _ in 0..N {
            let _g = rec.enter(CostPhase::ChildBound);
        }
        let armed = t.elapsed().as_nanos() as f64 / f64::from(N);
        rec.set_level(DetailLevel::Off);
        let t = Instant::now();
        for _ in 0..N {
            let _g = rec.enter(CostPhase::ChildBound);
        }
        let off = t.elapsed().as_nanos() as f64 / f64::from(N);
        eprintln!(
            "PHASE_COST_GUARD_NS armed={armed:.1} off={off:.1} debug={}",
            cfg!(debug_assertions)
        );
        assert!(armed < 50_000.0, "an armed guard costs {armed} ns");
    }

    /// review(cost-census): the per-enter cost of an armed guard UNDER
    /// CONTENTION, and the occupancy word's exactness under it. `threads`
    /// threads enter and leave the SAME phase back to back with no work in
    /// between, so every enter/exit CAS on the phase's shared cells contends:
    /// the worst case for the census's own cost (a real phase does work between
    /// its enter and exit, and the scored rows' hottest site, safenlp's
    /// `branch_select`, peaks at 6 threads in flight). The printed
    /// `PHASE_COST_CONTENDED_GUARD_NS` figure is the measurement (`--release
    /// --nocapture`); the assertions are the exactness contract: every enter
    /// counted, no interval left open, and the occupied wall never exceeding
    /// the elapsed wall nor the summed thread time.
    #[test]
    fn phase_cost_contended_guard_cost_is_printed_and_occupancy_stays_exact() {
        const N: u64 = 50_000;
        let max_threads = std::thread::available_parallelism()
            .map_or(4, std::num::NonZeroUsize::get)
            .clamp(2, 32);
        for threads in [1, 6, max_threads] {
            let rec = PhaseCostRecorder::new();
            rec.begin();
            let barrier = std::sync::Barrier::new(threads);
            let t0 = Instant::now();
            std::thread::scope(|scope| {
                for _ in 0..threads {
                    scope.spawn(|| {
                        barrier.wait();
                        for _ in 0..N {
                            let _g = rec.enter(CostPhase::BranchSelect);
                        }
                    });
                }
            });
            let elapsed = t0.elapsed().as_secs_f64();
            let snap = rec.snapshot(false);
            let sel = row(&snap, "branch_select");
            assert_eq!(sel.enters, N * threads as u64, "{sel:?}");
            assert_eq!((sel.open, sel.reentries), (0, 0), "{sel:?}");
            assert!(sel.wall_s <= elapsed + 1e-3, "{sel:?} elapsed {elapsed}");
            assert!(sel.wall_s <= sel.thread_s + 1e-3, "{sel:?}");
            assert!(sel.max_inflight <= threads as u64, "{sel:?}");
            assert_eq!(snap.unbalanced, 0);
            // Each thread ran its N enters concurrently with the others, so the
            // per-enter cost a thread paid is the elapsed wall over N.
            let per_enter_ns = elapsed * 1e9 / N as f64;
            eprintln!(
                "PHASE_COST_CONTENDED_GUARD_NS threads={threads} per_enter_ns={per_enter_ns:.1} \
                 max_inflight={} debug={}",
                sel.max_inflight,
                cfg!(debug_assertions)
            );
        }
    }
}
