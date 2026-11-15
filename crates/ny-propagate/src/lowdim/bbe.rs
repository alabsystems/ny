// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! M8a — `#m8-input-split-batch`: the lowdim lane's UNTRUSTED forward-linear
//! wave search (ENGINE_PROGRAM §2 M8; `bab.branching.input_split.lowdim_batch_engine:
//! bbe`, default `legacy`).
//!
//! # What it does
//!
//! With the key armed and a backend registered
//! ([`set_forward_linear_search_backend_factory`], ny-cli registers ny-bbe's
//! `CpuF32Backend`), a lane run copies its lowered programs into an
//! [`FlProblem`] (the lane's `Arc` weights reused, zero copy on the lane side)
//! and hands every WAVE to one `forward_linear_batch` call (R1). Per domain the
//! backend returns, per requested objective row, a search lower bound, its
//! absolute-term sum and the bound form's input coefficients. Then:
//!
//! 1. INGEST (the ONE `into_ingest` caller in `lowdim`, [`ingest`]): the hint
//!    is shifted by the probe's `tau`, `s = lb32 + tau_rel * scale`
//!    (non-finite -> `-inf`), and the domain is CLAIMED iff its clause's OWN
//!    predicate holds on `s` (`s_i > thr_i` for some conjunct, or
//!    `FamClause::closes(s)`). A claim is optimistic by `tau`, as M4's is.
//! 2. CERTIFY: only the claimed domains are re-bounded — by the lane's own f64
//!    `search::bound_chunk` / `shared::bound_group` over each claimed domain's
//!    own box (claimed domains in domain order, cut at the run's chunk) — and
//!    decided by the UNCHANGED `search::fate` / `shared::decide` (strict
//!    `lb > thr`). A claim the f64 bound does not close is REFUTED and split.
//! 3. SPLIT: every unclaimed domain goes to `search::open_fate` /
//!    `shared::open_decide` — the lane's f64 violation-candidate test, then an
//!    exact-halves split (the hint only picks the dimension). It is never
//!    closed and never clipped. An unclaimed domain whose open decision is
//!    `Resolution` (it cannot be split) is NOT given up on the hint: it is
//!    certified exactly like a claim (step 2), so a hint that misses a close
//!    at the resolution floor never ends a clause the legacy bounder proves
//!    (review fix; such domains count as `claimed`).
//!
//! A hint never becomes a `DomainBound` (pinned by the source ratchet
//! `lowdim_bbe_has_one_ingest_gate_and_hints_never_build_a_domain_bound`).
//!
//! # Fallbacks (a refusal is never a verdict)
//!
//! * [`bbe_admits`] (the named admission, router C2) refuses a program the
//!   backend does not mirror — `refine_max_width > 0`, `joint_quadratic`,
//!   `alpha_iters > 0`, `clip`, any `Mul` or `MaxGlobal` op — and the
//!   unchanged legacy bounder runs (`fallback=unsupported:<why>`).
//! * No registered backend, an upload refusal or panic, a failed PROBE
//!   (liveness only: each family's first clause at its root box, both bounded
//!   by the backend and by the lane's f64 forms; `err_rel = max |lb32 - lb64|
//!   / max(scale, 1)`, fail on a non-finite search value where f64 is finite
//!   or `err_rel > 1.25e-3`; `tau_rel = clamp(8 err_rel, 1e-6, 1e-2)`) all run
//!   legacy for the whole run.
//! * LATCHES (the rest of the run runs legacy): a backend panic, an
//!   `Invalid` / `Memory` / `Unsupported` / `Nothing` refusal, a malformed
//!   output, and a REFUTATION STORM (`refuted / claimed > 0.5` after at least
//!   1024 claims).
//! * `Refusal::Deadline` stops the lane `Unproved(Deadline)`.
//!
//! # Observable
//!
//! Every armed lane run prints ONE line (append-only fields; the leading
//! marker and the trailing version tag are contiguous literals):
//!
//! `[bbe] phase=input_split lane=lowdim scheduler=<single|shared|disjunction>
//! backend=<name> probe=<ok|fail|skip> tau_rel=.. calls=.. batched_max=..
//! domains=.. claimed=.. certified=.. refuted=.. unclaimed=..
//! fallback=<none|reason> upload_s=.. probe_s=.. search_s=.. certify_s=..
//! m8-input-split-batch/v1`
//!
//! (`probe=skip`: the run fell back before a probe could run.) A run with
//! `fallback` other than `none` is VOID for measurement.
//!
//! # ROUTER (R1 pending)
//!
//! The router's `CapabilityDecl` type lives on `feat/router-r1` (0c084a7dd),
//! not on main; until it lands the rows are declared as plain data
//! ([`ROUTER_ROWS`]) and wired when R1 merges:
//!
//! | id | kind | admission | progress / unit | safe points | resources | stop reasons | knobs |
//! |---|---|---|---|---|---|---|---|
//! | `lowdim-bbe` | `Variant{slot:"lowdim.batch_engine", value:"bbe"}` | `lowdim::bbe::bbe_admits` | Volume / Wave | BetweenUnits (wave top) | all threads; device `Search`; mem = weights once per Arc + threads x `work_bytes`, within the lease | deadline, queue, resolution, candidate, nonfinite + fallback reasons | none (wave = `cfg.batch`; certify chunk = the `CHUNK_BYTES` clamp) |
//! | `input-split-worklist` | `Variant{slot:"input_split.engine", value:"bbe"}` | `lowdim::disjunction::disjunction_admits`; fold -> shared | Volume / Wave | BetweenUnits | CPU, all threads; queue <= `max_queue` / `SHARED_QUEUE_BYTES` | deadline, queue, memory, resolution, candidate (per clause), nonfinite | none |
//!
//! Progress is `LowDimStats::closed_volume` (`sum_closed prod_i (w_i /
//! w0_i)` per clause, clip-removed volume included); the safe point is the
//! top of every wave (the deadline poll). `first_rung: Unknown`; the
//! `upload_s` / `probe_s` fields exist so a cost model can be fitted later.
//! The deadline is still an `Instant` argument; R2 turns it into a monotone
//! cell.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use ny_core::search::{
    DeviceLease, FlBatch, FlFamily, FlHandle, FlLinear, FlOp, FlOpConsts, FlOut, FlProblem,
    FlProgram, Refusal, SearchBackend,
};
use rayon::prelude::*;

use super::search::{self, DomainNote, Fate, Wave};
use super::shared;
use super::{Chunk, FamClause, LowDimConfig, Op, Program};

// ---------------------------------------------------------------------------
//  The injected backend
// ---------------------------------------------------------------------------

type BackendFactory = Box<dyn Fn() -> Option<Arc<dyn SearchBackend>> + Send + Sync>;

static FACTORY: OnceLock<BackendFactory> = OnceLock::new();

/// Register the process's forward-linear search backend factory (ny-cli,
/// once, at startup). First install wins. The closure runs at most once, on
/// the first ARMED lane run, so a dark lane constructs nothing.
pub fn set_forward_linear_search_backend_factory<F>(factory: F)
where
    F: Fn() -> Option<Arc<dyn SearchBackend>> + Send + Sync + 'static,
{
    let _ = FACTORY.set(Box::new(factory));
}

fn registered_backend() -> Option<Arc<dyn SearchBackend>> {
    static HANDLE: OnceLock<Arc<dyn SearchBackend>> = OnceLock::new();
    if let Some(backend) = HANDLE.get() {
        return Some(Arc::clone(backend));
    }
    let built = FACTORY.get().and_then(|factory| factory())?;
    Some(Arc::clone(HANDLE.get_or_init(|| built)))
}

/// True when a factory is registered (the CLI's arrival test face).
pub fn forward_linear_search_backend_registered() -> bool {
    FACTORY.get().is_some()
}

#[cfg(test)]
thread_local! {
    static TEST_BACKEND: std::cell::RefCell<Option<Arc<dyn SearchBackend>>> =
        const { std::cell::RefCell::new(None) };
}

/// TEST ONLY: run `f` with `backend` injected as this thread's
/// forward-linear search backend.
#[cfg(test)]
pub(crate) fn with_test_backend<T>(backend: Arc<dyn SearchBackend>, f: impl FnOnce() -> T) -> T {
    TEST_BACKEND.with(|c| *c.borrow_mut() = Some(backend));
    let out = f();
    TEST_BACKEND.with(|c| *c.borrow_mut() = None);
    out
}

/// The backend a run uses: a test's thread-local injection, else the
/// registered one.
fn fl_backend() -> Option<Arc<dyn SearchBackend>> {
    #[cfg(test)]
    {
        if let Some(b) = TEST_BACKEND.with(|c| c.borrow().clone()) {
            return Some(b);
        }
    }
    registered_backend()
}

thread_local! {
    /// The last `[bbe] phase=input_split` line this THREAD printed (the lane
    /// runs on its calling thread). Telemetry only.
    static LAST_SUMMARY: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

/// Take the last `[bbe] phase=input_split ... m8-input-split-batch/v1` line
/// printed on this thread (`None` when no armed lane run printed since the
/// last take). The receipt a caller can check that the wave search — and not
/// its legacy fallback — ran.
pub fn take_last_bbe_summary() -> Option<String> {
    LAST_SUMMARY.with(|c| c.borrow_mut().take())
}

// ---------------------------------------------------------------------------
//  Router data (R1 pending)
// ---------------------------------------------------------------------------

/// One router capability row, as plain data until R1's `CapabilityDecl`
/// lands (see the module docs' table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouterRow {
    /// Capability id (the registry's `id`).
    pub id: &'static str,
    /// `Variant{slot, value}`: the typed slot and the value it selects.
    pub slot: &'static str,
    /// The value of `slot` this row is.
    pub value: &'static str,
    /// The named admission function the lane calls.
    pub admission: &'static str,
    /// Progress kind.
    pub progress: &'static str,
    /// Work unit.
    pub unit: &'static str,
    /// Where a router may stop, switch or re-plan.
    pub safe_points: &'static str,
    /// Resources the unit consumes.
    pub resources: &'static str,
    /// Stop reasons the unit reports.
    pub stop_reasons: &'static str,
    /// Tunable knobs (none: every cap is derived).
    pub knobs: &'static str,
}

/// M8a's two router rows.
pub const ROUTER_ROWS: [RouterRow; 2] = [
    RouterRow {
        id: "lowdim-bbe",
        slot: "lowdim.batch_engine",
        value: "bbe",
        admission: "lowdim::bbe::bbe_admits",
        progress: "Volume",
        unit: "Wave",
        safe_points: "BetweenUnits (wave top)",
        resources: "all threads; device Search; weights once per Arc + threads x work_bytes, within the lease",
        stop_reasons: "deadline, queue, resolution, candidate, nonfinite, fallback reasons",
        knobs: "none (wave = cfg.batch; certify chunk = the CHUNK_BYTES clamp)",
    },
    RouterRow {
        id: "input-split-worklist",
        slot: "input_split.engine",
        value: "bbe",
        admission: "lowdim::disjunction::disjunction_admits; fold -> shared",
        progress: "Volume",
        unit: "Wave",
        safe_points: "BetweenUnits",
        resources: "CPU, all threads; queue <= max_queue / SHARED_QUEUE_BYTES",
        stop_reasons: "deadline, queue, memory, resolution, candidate (per clause), nonfinite",
        knobs: "none",
    },
];

// ---------------------------------------------------------------------------
//  Admission, the problem copy, stats
// ---------------------------------------------------------------------------

/// The named admission (router C2): the programs and config the M8a backend
/// mirrors. `Err` is the reason (printed as `fallback=unsupported:<why>`).
pub(crate) fn bbe_admits(progs: &[&Program], cfg: &LowDimConfig) -> Result<(), &'static str> {
    if cfg.refine_max_width > 0 {
        return Err("refine");
    }
    if cfg.joint_quadratic {
        return Err("joint");
    }
    if cfg.alpha_iters > 0 {
        return Err("alpha");
    }
    if cfg.clip {
        return Err("clip");
    }
    for p in progs {
        for op in &p.ops {
            match op {
                Op::Mul { .. } => return Err("mul"),
                Op::MaxGlobal { .. } => return Err("maxglobal"),
                _ => {}
            }
        }
    }
    Ok(())
}

/// Which lane entry a run belongs to (the observable's `scheduler=`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Scheduler {
    /// One conjunction (`prove_conjunction_core`).
    Single,
    /// The fold's shared worklist (`shared::prove_families_shared`).
    Shared,
    /// The same-box disjunction tree (`disjunction`).
    Disjunction,
}

impl Scheduler {
    fn token(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Shared => "shared",
            Self::Disjunction => "disjunction",
        }
    }
}

/// Counters of one armed lane run (the `[bbe]` line). Telemetry only.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BbeStats {
    /// The backend's name (`none` when there was none).
    pub backend: &'static str,
    /// `Some(ok)` once the probe ran.
    pub probe: Option<bool>,
    /// The claim shift `tau_rel` the probe derived.
    pub tau_rel: f64,
    /// `forward_linear_batch` calls (the probe's included).
    pub calls: u64,
    /// Largest wave searched in one call.
    pub batched_max: usize,
    /// Domains searched.
    pub domains: u64,
    /// Domains a hint claimed, plus unclaimed domains certified because
    /// they could not be split (review fix: never a terminal stop on a hint).
    pub claimed: u64,
    /// Claims the lane's f64 bound closed.
    pub certified: u64,
    /// Claims the lane's f64 bound did not close (split).
    pub refuted: u64,
    /// Domains no hint claimed (split into exact halves).
    pub unclaimed: u64,
    /// Why the run (or its rest) ran legacy; `None` = never.
    pub fallback: Option<&'static str>,
    /// Seconds building and uploading the problem copy.
    pub upload_s: f64,
    /// Seconds in the probe.
    pub probe_s: f64,
    /// Seconds in `forward_linear_batch` calls.
    pub search_s: f64,
    /// Seconds re-bounding claims with the lane's f64 forms.
    pub certify_s: f64,
}

/// Print (and remember on this thread) the `[bbe]` line of one run.
pub(crate) fn emit_line(scheduler: Scheduler, st: &BbeStats) {
    let probe = match st.probe {
        Some(true) => "ok",
        Some(false) => "fail",
        None => "skip",
    };
    let line = format!(
        "[bbe] phase=input_split lane=lowdim scheduler={} backend={} probe={probe} tau_rel={:.3e} \
         calls={} batched_max={} domains={} claimed={} certified={} refuted={} unclaimed={} \
         fallback={} upload_s={:.3} probe_s={:.3} search_s={:.3} certify_s={:.3} \
         m8-input-split-batch/v1",
        scheduler.token(),
        if st.backend.is_empty() {
            "none"
        } else {
            st.backend
        },
        st.tau_rel,
        st.calls,
        st.batched_max,
        st.domains,
        st.claimed,
        st.certified,
        st.refuted,
        st.unclaimed,
        st.fallback.unwrap_or("none"),
        st.upload_s,
        st.probe_s,
        st.search_s,
        st.certify_s,
    );
    eprintln!("{line}");
    LAST_SUMMARY.with(|c| *c.borrow_mut() = Some(line));
}

/// A lane `Program`'s structure as an [`FlProgram`] (weights `Arc`-shared).
fn fl_program(p: &Program) -> FlProgram {
    let u = |i: usize| u32::try_from(i).unwrap_or(u32::MAX);
    let ops = p
        .ops
        .iter()
        .enumerate()
        .map(|(k, op)| match op {
            Op::Input => FlOp::Input,
            Op::Linear { src, n_in, w, .. } => FlOp::Linear(FlLinear {
                src: u(*src),
                n_in: u(*n_in),
                n_out: u(p.sizes[k]),
                w: Arc::clone(w),
            }),
            Op::Sparse { terms, .. } => FlOp::Sparse {
                terms: terms
                    .iter()
                    .map(|t| t.iter().map(|x| (x.src, x.idx)).collect())
                    .collect(),
            },
            Op::Relu { src } => FlOp::Relu { src: u(*src) },
            Op::Sigmoid { src } => FlOp::Sigmoid { src: u(*src) },
            Op::Slice { src, start } => FlOp::Slice {
                src: u(*src),
                start: u(*start),
            },
            Op::Concat { srcs } => FlOp::Concat {
                srcs: srcs.iter().map(|&s| u(s)).collect(),
            },
            Op::Identity { src } => FlOp::Identity { src: u(*src) },
            Op::Add { a, b } => FlOp::Add { a: u(*a), b: u(*b) },
            Op::Sub { a, b } => FlOp::Sub { a: u(*a), b: u(*b) },
            Op::AddConst { src, .. } => FlOp::AddConst { src: u(*src) },
            Op::RevSubConst { src, .. } => FlOp::RevSubConst { src: u(*src) },
            Op::MulConst { src, .. } => FlOp::MulConst { src: u(*src) },
            Op::MaxGlobal { src } => FlOp::MaxGlobal { src: u(*src) },
            Op::Mul { a, b } => FlOp::Mul { a: u(*a), b: u(*b) },
        })
        .collect();
    FlProgram {
        d: p.d,
        sizes: p.sizes.clone(),
        seed: p.seed,
        ops,
    }
}

/// `v` as an `Arc`, sharing `base`'s allocation when the contents are the
/// same bits (the deeper layers of a fold family).
fn shared_arc(v: &[f64], base: Option<&Arc<[f64]>>) -> Arc<[f64]> {
    if let Some(b) = base {
        if b.len() == v.len() && b.iter().zip(v).all(|(x, y)| x.to_bits() == y.to_bits()) {
            return Arc::clone(b);
        }
    }
    v.iter().copied().collect()
}

/// A clause program's constants, aligned with its ops (sharing `base`'s
/// `Arc`s where the values coincide).
fn fl_consts(p: &Program, base: Option<&[FlOpConsts]>) -> Vec<FlOpConsts> {
    p.ops
        .iter()
        .enumerate()
        .map(|(k, op)| {
            let b = base.and_then(|b| b.get(k));
            match op {
                Op::Linear {
                    bias_lo, bias_hi, ..
                } => {
                    let (blo, bhi) = match b {
                        Some(FlOpConsts::Bias { lo, hi }) => (Some(lo), Some(hi)),
                        _ => (None, None),
                    };
                    FlOpConsts::Bias {
                        lo: shared_arc(bias_lo, blo),
                        hi: shared_arc(bias_hi, bhi),
                    }
                }
                Op::Sparse {
                    terms, k_lo, k_hi, ..
                } => {
                    let coef: Vec<f64> = terms.iter().flatten().map(|t| t.coef).collect();
                    let (bc, bl, bh) = match b {
                        Some(FlOpConsts::Sparse { coef, k_lo, k_hi }) => {
                            (Some(coef), Some(k_lo), Some(k_hi))
                        }
                        _ => (None, None, None),
                    };
                    FlOpConsts::Sparse {
                        coef: shared_arc(&coef, bc),
                        k_lo: shared_arc(k_lo, bl),
                        k_hi: shared_arc(k_hi, bh),
                    }
                }
                Op::AddConst { k, .. } | Op::RevSubConst { k, .. } | Op::MulConst { k, .. } => {
                    let bk = match b {
                        Some(FlOpConsts::K(v)) => Some(v),
                        _ => None,
                    };
                    FlOpConsts::K(shared_arc(k, bk))
                }
                _ => FlOpConsts::None,
            }
        })
        .collect()
}

/// The search copy of `families` (per family, its clauses' programs; clause
/// 0 supplies the structure) with the objective table `objectives`.
pub(crate) fn fl_problem(families: &[Vec<&Program>], objectives: Vec<Vec<f64>>) -> FlProblem {
    let families = families
        .iter()
        .map(|progs| {
            let program = fl_program(progs[0]);
            let mut clauses: Vec<Vec<FlOpConsts>> = Vec::with_capacity(progs.len());
            for (c, p) in progs.iter().enumerate() {
                let base = (c > 0).then(|| clauses[0].as_slice());
                let consts = fl_consts(p, base);
                clauses.push(consts);
            }
            FlFamily { program, clauses }
        })
        .collect();
    FlProblem {
        families,
        objectives,
    }
}

/// The lease a lane run offers its backend: an eighth of the observed live
/// headroom, at most 8 GiB, or 2 GiB when the headroom cannot be observed
/// (ny has no governor without a cgroup).
fn lease() -> DeviceLease {
    const GIB: u64 = 1 << 30;
    let bytes = match crate::network::crown_memory::process_memory_headroom_bytes() {
        Some(h) => (h / 8).min(8 * GIB),
        None => 2 * GIB,
    };
    DeviceLease { bytes }
}

// ---------------------------------------------------------------------------
//  The ingest gate
// ---------------------------------------------------------------------------

/// One call's hints, sanitised: per CSR row the raw search bound (`NaN`
/// kept), its absolute-term sum, the SHIFTED claim value `lb32 + tau_rel *
/// scale` (non-finite -> `-inf`) and the input coefficients (non-finite ->
/// 0). Read only for claims (a boolean), split dimensions and a leaf-attack
/// ranking; never a bound.
pub(crate) struct Ingest {
    pub(crate) raw: Vec<f64>,
    pub(crate) scale: Vec<f64>,
    pub(crate) shifted: Vec<f64>,
    pub(crate) grads: Vec<f64>,
    pub(crate) d: usize,
}

impl Ingest {
    /// Row `r`'s sanitised raw bound (non-finite -> `-inf`), for split and
    /// ranking choices only.
    pub(crate) fn hint_lb(&self, r: usize) -> f64 {
        let v = self.raw[r];
        if v.is_finite() {
            v
        } else {
            f64::NEG_INFINITY
        }
    }
}

/// THE ingest gate: the only `into_ingest` caller in `lowdim`. `None` when
/// the backend wrote an output of another shape.
fn ingest(out: &FlOut, rows: usize, d: usize, tau_rel: f64) -> Option<Ingest> {
    let view = out.hint().into_ingest();
    if !view.well_formed() || view.rows() != rows || view.d() != d {
        return None;
    }
    let mut ing = Ingest {
        raw: Vec::with_capacity(rows),
        scale: Vec::with_capacity(rows),
        shifted: Vec::with_capacity(rows),
        grads: Vec::with_capacity(rows * d),
        d,
    };
    for r in 0..rows {
        let lb = f64::from(view.lb(r));
        let sc = f64::from(view.scale(r));
        let s = lb + tau_rel * sc.abs();
        ing.raw.push(lb);
        ing.scale.push(sc);
        ing.shifted
            .push(if s.is_finite() { s } else { f64::NEG_INFINITY });
        ing.grads.extend(view.grad(r).iter().map(|&g| {
            let g = f64::from(g);
            if g.is_finite() {
                g
            } else {
                0.0
            }
        }));
    }
    Some(ing)
}

// ---------------------------------------------------------------------------
//  The armed core
// ---------------------------------------------------------------------------

/// The claims of one searched wave.
pub(crate) enum Searched {
    /// The hints of every requested row. Boxed: an `Ingest` is 104 bytes
    /// against two unit variants (clippy::large_enum_variant, threshold 64).
    Hints(Box<Ingest>),
    /// The backend refused past the deadline: stop the lane unproved.
    Deadline,
    /// The backend latched: this wave (and the rest of the run) runs legacy.
    Latched,
}

/// An armed forward-linear search: backend, upload, claim shift, counters.
pub(crate) struct BbeCore {
    backend: Arc<dyn SearchBackend>,
    handle: FlHandle,
    dims: Vec<usize>,
    pub(crate) tau_rel: f64,
}

/// Minimum claims before the refutation-storm latch may fire.
const STORM_MIN_CLAIMS: u64 = 1024;

/// Probe failure threshold on `err_rel`.
const PROBE_MAX_ERR_REL: f64 = 1.25e-3;

/// One family's probe: its first live clause, that clause's root box and
/// its objective rows (indices into the problem's table).
pub(crate) struct ProbeSpec<'a> {
    pub(crate) family: u32,
    pub(crate) clause: u32,
    pub(crate) lo: &'a [f64],
    pub(crate) hi: &'a [f64],
    pub(crate) rows: Vec<u32>,
}

/// An armed lane run's state: the core while it is live, and the counters.
pub(crate) struct BbeRun {
    pub(crate) scheduler: Scheduler,
    core: Option<BbeCore>,
    pub(crate) stats: BbeStats,
}

impl BbeRun {
    /// A run that never armed (`fallback = why`).
    fn fallback(scheduler: Scheduler, mut stats: BbeStats, why: &'static str) -> Self {
        stats.fallback = Some(why);
        Self {
            scheduler,
            core: None,
            stats,
        }
    }

    /// Arm a run over `families` (per family, its clauses' programs, clause 0
    /// supplying the structure) and the objective table `objectives`,
    /// probing each family per `probes`. Never fails: any reason not to
    /// search is a legacy run with `fallback` set.
    pub(crate) fn arm(
        scheduler: Scheduler,
        families: &[Vec<&Program>],
        objectives: Vec<Vec<f64>>,
        probes: &[ProbeSpec<'_>],
        cfg: &LowDimConfig,
    ) -> Self {
        let mut stats = BbeStats {
            backend: "none",
            ..BbeStats::default()
        };
        let all: Vec<&Program> = families.iter().flatten().copied().collect();
        if let Err(why) = bbe_admits(&all, cfg) {
            return Self::fallback(scheduler, stats, unsupported(why));
        }
        if families.is_empty() || families.iter().any(Vec::is_empty) || objectives.is_empty() {
            return Self::fallback(scheduler, stats, "nothing");
        }
        let Some(backend) = fl_backend() else {
            return Self::fallback(scheduler, stats, "no_backend");
        };
        stats.backend = backend.name();
        let t_up = Instant::now();
        let problem = fl_problem(families, objectives.clone());
        let uploaded = catch_unwind(AssertUnwindSafe(|| {
            backend.upload_forward_linear(&problem, &lease())
        }));
        stats.upload_s = t_up.elapsed().as_secs_f64();
        let handle = match uploaded {
            Err(_) => return Self::fallback(scheduler, stats, "panic"),
            Ok(Err(r)) => return Self::fallback(scheduler, stats, refusal_reason(&r)),
            Ok(Ok(h)) => h,
        };
        let dims = families.iter().map(|f| f[0].d).collect();
        let mut core = BbeCore {
            backend,
            handle,
            dims,
            tau_rel: 1e-6,
        };
        // THE PROBE (liveness only; soundness never depends on it).
        let t_probe = Instant::now();
        let mut err_rel = 0.0f64;
        let mut ok = true;
        for p in probes {
            let f = p.family as usize;
            let Some(progs) = families.get(f) else {
                ok = false;
                break;
            };
            let c = p.clause as usize;
            let objs: Vec<Vec<f64>> = p
                .rows
                .iter()
                .map(|&r| objectives[r as usize].clone())
                .collect();
            let exact = shared::bound_group(progs, &[c], p.lo, p.hi, &[objs.as_slice()], 1, false);
            let off = [0u32, u32::try_from(p.rows.len()).unwrap_or(u32::MAX)];
            let clause = [p.clause];
            stats.calls += 1;
            let hint = core.call(f, p.lo, p.hi, 1, &clause, &off, &p.rows, None);
            let Ok(ing) = hint else {
                ok = false;
                break;
            };
            let Some(db) = exact.first() else {
                ok = false;
                break;
            };
            for (r, &lb64) in db.lbs.iter().enumerate() {
                if !lb64.is_finite() {
                    continue;
                }
                let lb32 = ing.raw.get(r).copied().unwrap_or(f64::NAN);
                if !lb32.is_finite() {
                    ok = false;
                    break;
                }
                let scale = ing.scale.get(r).copied().unwrap_or(f64::NAN).abs();
                let e = (lb32 - lb64).abs() / scale.max(1.0);
                if !e.is_finite() {
                    ok = false;
                    break;
                }
                err_rel = err_rel.max(e);
            }
            if !ok {
                break;
            }
        }
        stats.probe_s = t_probe.elapsed().as_secs_f64();
        if !ok || err_rel > PROBE_MAX_ERR_REL {
            stats.probe = Some(false);
            return Self::fallback(scheduler, stats, "probe");
        }
        stats.probe = Some(true);
        core.tau_rel = (8.0 * err_rel).clamp(1e-6, 1e-2);
        stats.tau_rel = core.tau_rel;
        Self {
            scheduler,
            core: Some(core),
            stats,
        }
    }

    /// Live (not fallen back).
    pub(crate) fn live(&self) -> bool {
        self.core.is_some()
    }

    fn latch(&mut self, why: &'static str) {
        self.core = None;
        if self.stats.fallback.is_none() {
            self.stats.fallback = Some(why);
        }
    }

    /// Search one wave of family `family` (`clause` per domain or empty;
    /// the CSR `off` / `rows`). Counts the call; latches on any failure but
    /// the deadline.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn search(
        &mut self,
        family: usize,
        lo: &[f64],
        hi: &[f64],
        count: usize,
        clause: &[u32],
        off: &[u32],
        rows: &[u32],
        deadline: Instant,
    ) -> Searched {
        let Some(core) = self.core.as_ref() else {
            return Searched::Latched;
        };
        let t = Instant::now();
        self.stats.calls += 1;
        self.stats.batched_max = self.stats.batched_max.max(count);
        self.stats.domains += count as u64;
        let res = core.call(family, lo, hi, count, clause, off, rows, Some(deadline));
        self.stats.search_s += t.elapsed().as_secs_f64();
        match res {
            Ok(ing) => Searched::Hints(Box::new(ing)),
            Err(Some(Refusal::Deadline)) => Searched::Deadline,
            Err(Some(r)) => {
                self.latch(refusal_reason(&r));
                Searched::Latched
            }
            Err(None) => {
                self.latch("panic_or_bad_shape");
                Searched::Latched
            }
        }
    }

    /// Record one wave's claim outcomes; latch on a refutation storm.
    pub(crate) fn note(&mut self, claimed: u64, certified: u64, refuted: u64, unclaimed: u64) {
        self.stats.claimed += claimed;
        self.stats.certified += certified;
        self.stats.refuted += refuted;
        self.stats.unclaimed += unclaimed;
        if self.stats.claimed >= STORM_MIN_CLAIMS && 2 * self.stats.refuted > self.stats.claimed {
            self.latch("refutation_storm");
        }
    }

    /// Print the run's `[bbe]` line.
    pub(crate) fn emit(&self) {
        emit_line(self.scheduler, &self.stats);
    }
}

impl BbeCore {
    /// One backend call, ingested. `Err(None)` = a panic or a malformed
    /// output; `Err(Some(r))` = a refusal.
    #[allow(clippy::too_many_arguments)]
    fn call(
        &self,
        family: usize,
        lo: &[f64],
        hi: &[f64],
        count: usize,
        clause: &[u32],
        off: &[u32],
        rows: &[u32],
        deadline: Option<Instant>,
    ) -> Result<Ingest, Option<Refusal>> {
        let d = self.dims.get(family).copied().unwrap_or(0);
        let batch = FlBatch {
            family: u32::try_from(family).unwrap_or(u32::MAX),
            count,
            d,
            lo,
            hi,
            clause,
            obj_offsets: off,
            obj_rows: rows,
            deadline,
        };
        let mut out = FlOut::default();
        let res = catch_unwind(AssertUnwindSafe(|| {
            self.backend
                .forward_linear_batch(&self.handle, &batch, &mut out)
        }));
        match res {
            Err(_) => Err(None),
            Ok(Err(r)) => Err(Some(r)),
            // The stats are telemetry; nothing reads them.
            Ok(Ok(_stats)) => ingest(&out, rows.len(), d, self.tau_rel).ok_or(None),
        }
    }
}

fn unsupported(why: &'static str) -> &'static str {
    match why {
        "refine" => "unsupported:refine",
        "joint" => "unsupported:joint",
        "alpha" => "unsupported:alpha",
        "clip" => "unsupported:clip",
        "mul" => "unsupported:mul",
        "maxglobal" => "unsupported:maxglobal",
        _ => "unsupported:other",
    }
}

fn refusal_reason(r: &Refusal) -> &'static str {
    match r {
        Refusal::Unsupported(_) => "unsupported:backend",
        Refusal::Invalid(_) => "invalid",
        Refusal::Nothing(_) => "nothing",
        Refusal::Deadline => "deadline",
        Refusal::Memory { .. } => "memory",
        // M6's device-fault refusal (ny_core::search, feat/wave-cuda-kernels):
        // the CPU f32 forward-linear search never returns it, but a device
        // backend may; it falls back like every other refusal.
        Refusal::Fault(_) => "fault",
    }
}

/// Every domain of a single-conjunction wave asks for all `k` objective rows.
fn all_rows_csr(count: usize, k: usize) -> (Vec<u32>, Vec<u32>) {
    let u = |i: usize| u32::try_from(i).unwrap_or(u32::MAX);
    let off = (0..=count).map(|b| u(b * k)).collect();
    let rows = (0..count).flat_map(|_| (0..k).map(u)).collect();
    (off, rows)
}

/// How a searched domain fared (per-clause counters; telemetry only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ClaimMark {
    /// Not searched (legacy bounder).
    #[default]
    None,
    /// Claimed and closed by the lane's f64 bound.
    Certified,
    /// Claimed and NOT closed by the lane's f64 bound (split).
    Refuted,
    /// Not claimed (split into exact halves).
    Unclaimed,
}

/// The per-domain decisions of a searched wave (`None`: run legacy).
pub(crate) enum WaveResult<T> {
    /// Decided in domain order.
    Done(T),
    /// The deadline passed inside the backend.
    Deadline,
    /// Not armed or latched: run the legacy path for this wave.
    Legacy,
}

// ---------------------------------------------------------------------------
//  Single conjunction (`prove_conjunction_core`)
// ---------------------------------------------------------------------------

/// `Bbe::wave` for [`super::prove_conjunction_core`]: one search call for
/// the whole wave, claims certified by `search::bound_chunk` + the unchanged
/// `search::fate`, every other domain `search::open_fate`. Results in domain
/// order (the lane's merge is unchanged).
pub(crate) fn single_wave(
    run: &mut BbeRun,
    w: &Wave<'_>,
    lo: &[f64],
    hi: &[f64],
    count: usize,
    deadline: Instant,
) -> WaveResult<Vec<(Fate, DomainNote)>> {
    if !run.live() {
        return WaveResult::Legacy;
    }
    let d = w.prog.d;
    let k = w.objs.len();
    let (off, rows) = all_rows_csr(count, k);
    let ing = match run.search(0, lo, hi, count, &[], &off, &rows, deadline) {
        Searched::Hints(ing) => *ing,
        Searched::Deadline => return WaveResult::Deadline,
        Searched::Latched => return WaveResult::Legacy,
    };
    // Claims: the clause's own predicate on the shifted hints.
    let mut is_claimed: Vec<bool> = (0..count)
        .map(|b| (0..k).any(|i| ing.shifted[b * k + i] > w.thrs[i]))
        .collect();
    // Every unclaimed domain: the lane's candidate test, then exact halves.
    let unclaimed: Vec<usize> = (0..count).filter(|&b| !is_claimed[b]).collect();
    let opened: Vec<(Fate, DomainNote)> = unclaimed
        .par_iter()
        .map(|&b| {
            let lbs: Vec<f64> = (0..k).map(|i| ing.hint_lb(b * k + i)).collect();
            let grads = &ing.grads[b * k * d..(b + 1) * k * d];
            let f = search::open_fate(
                w,
                lo[b * d..(b + 1) * d].to_vec(),
                hi[b * d..(b + 1) * d].to_vec(),
                false,
                grads,
                &lbs,
            );
            let margin = lbs
                .iter()
                .zip(w.thrs)
                .map(|(lb, t)| lb - t)
                .fold(f64::NEG_INFINITY, f64::max);
            let note = DomainNote {
                joint_hits: 0,
                alpha: search::AlphaNote::default(),
                margin,
                claim: ClaimMark::Unclaimed,
            };
            (f, note)
        })
        .collect();
    // Review fix (an untrusted miss is never a terminal stop): an unclaimed
    // domain whose open decision is `Resolution` would END the lane on a hint
    // alone, where the legacy bounder might close it (a missed claim at the
    // resolution floor) or split it along another coordinate (its f64
    // gradients). Such a domain is certified exactly like a claim: bounded by
    // the lane's own f64 forms and decided by the UNCHANGED `fate`, so the
    // lane stops at Resolution only when the legacy decision does.
    for (j, &b) in unclaimed.iter().enumerate() {
        if matches!(opened[j].0, Fate::Resolution) {
            is_claimed[b] = true;
        }
    }
    let claimed: Vec<usize> = (0..count).filter(|&b| is_claimed[b]).collect();
    // Certify the claims: claimed domains in domain order, cut at the run's
    // chunk, bounded by the lane's own f64 forms, decided by the UNCHANGED
    // `fate`.
    let t_cert = Instant::now();
    let chunk = w.chunk.max(1);
    let clo: Vec<f64> = claimed
        .iter()
        .flat_map(|&b| lo[b * d..(b + 1) * d].iter().copied())
        .collect();
    let chi: Vec<f64> = claimed
        .iter()
        .flat_map(|&b| hi[b * d..(b + 1) * d].iter().copied())
        .collect();
    let starts: Vec<usize> = (0..claimed.len()).step_by(chunk).collect();
    let certified: Vec<(Fate, DomainNote)> = starts
        .par_iter()
        .flat_map_iter(|&s| {
            let m = chunk.min(claimed.len() - s);
            let ch = Chunk::new(m, d, &clo[s * d..(s + m) * d], &chi[s * d..(s + m) * d]);
            let dbs = search::bound_chunk(w.prog, &ch, w.objs, w.cfg.clip, None);
            dbs.into_iter()
                .enumerate()
                .map(|(j, db)| {
                    let b = claimed[s + j];
                    let f = search::fate(w, &lo[b * d..(b + 1) * d], &hi[b * d..(b + 1) * d], &db);
                    let margin = if db.lbs.len() == w.thrs.len() {
                        db.lbs
                            .iter()
                            .zip(w.thrs)
                            .map(|(lb, t)| lb - t)
                            .fold(f64::NEG_INFINITY, f64::max)
                    } else {
                        f64::NAN
                    };
                    let mark = if matches!(f, Fate::Closed | Fate::ClipClosed) {
                        ClaimMark::Certified
                    } else {
                        ClaimMark::Refuted
                    };
                    let note = DomainNote {
                        joint_hits: db.joint_hits,
                        alpha: db.alpha,
                        margin,
                        claim: mark,
                    };
                    (f, note)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    run.stats.certify_s += t_cert.elapsed().as_secs_f64();
    if certified.len() != claimed.len() || opened.len() != unclaimed.len() {
        return WaveResult::Legacy;
    }
    // Merge in domain order (a domain certified at the resolution floor
    // takes its certified decision, not its open one).
    let mut ci = certified.into_iter();
    let mut open_of: Vec<Option<(Fate, DomainNote)>> = (0..count).map(|_| None).collect();
    for (&b, x) in unclaimed.iter().zip(opened) {
        open_of[b] = Some(x);
    }
    let mut out = Vec::with_capacity(count);
    for (b, &c) in is_claimed.iter().enumerate() {
        let next = if c { ci.next() } else { open_of[b].take() };
        match next {
            Some(x) => out.push(x),
            None => return WaveResult::Legacy,
        }
    }
    let n_cert = out
        .iter()
        .filter(|(_, n)| n.claim == ClaimMark::Certified)
        .count() as u64;
    let n_claimed = claimed.len() as u64;
    run.note(
        n_claimed,
        n_cert,
        n_claimed - n_cert,
        (count - claimed.len()) as u64,
    );
    WaveResult::Done(out)
}

/// Arm a single-conjunction run (probe: the root box, every objective).
pub(crate) fn arm_single(
    prog: &Program,
    lo0: &[f64],
    hi0: &[f64],
    objs: &[Vec<f64>],
    cfg: &LowDimConfig,
) -> BbeRun {
    let rows: Vec<u32> = (0..objs.len())
        .map(|i| u32::try_from(i).unwrap_or(u32::MAX))
        .collect();
    BbeRun::arm(
        Scheduler::Single,
        &[vec![prog]],
        objs.to_vec(),
        &[ProbeSpec {
            family: 0,
            clause: 0,
            lo: lo0,
            hi: hi0,
            rows,
        }],
        cfg,
    )
}

// ---------------------------------------------------------------------------
//  Family groups (`shared::prove_families_shared`)
// ---------------------------------------------------------------------------

/// One family group of a shared wave: its leaves' clause indices and boxes.
pub(crate) struct GroupIn {
    pub(crate) f: usize,
    pub(crate) which: Vec<usize>,
    pub(crate) lo: Vec<f64>,
    pub(crate) hi: Vec<f64>,
}

/// `Bbe::group` for the shared worklist: the search calls are issued
/// SEQUENTIALLY in group order (the backend is internally parallel); then
/// every group's claims are certified by `shared::bound_group(claimed,
/// planes = false)` and decided by the unchanged `shared::decide`, and every
/// unclaimed leaf by `shared::open_decide`, groups in parallel. Per group, per
/// leaf: its fate and claim mark.
#[allow(clippy::too_many_arguments)]
pub(crate) fn group_wave(
    run: &mut BbeRun,
    families: &[Vec<FamClause>],
    obj_base: &[Vec<u32>],
    groups: &[GroupIn],
    dims: &[usize],
    chunks: &[usize],
    w0s: &[Vec<Vec<f64>>],
    cfg: &LowDimConfig,
    deadline: Instant,
) -> WaveResult<Vec<Vec<(Fate, ClaimMark)>>> {
    if !run.live() {
        return WaveResult::Legacy;
    }
    let mut hints: Vec<Ingest> = Vec::with_capacity(groups.len());
    let mut csr: Vec<Vec<u32>> = Vec::with_capacity(groups.len());
    for g in groups {
        let fam = &families[g.f];
        let mut off = vec![0u32];
        let mut rows = Vec::new();
        for &c in &g.which {
            let base = obj_base[g.f][c];
            let k = fam[c].objs.len() as u32;
            rows.extend(base..base + k);
            off.push(u32::try_from(rows.len()).unwrap_or(u32::MAX));
        }
        let which: Vec<u32> = g
            .which
            .iter()
            .map(|&c| u32::try_from(c).unwrap_or(u32::MAX))
            .collect();
        match run.search(
            g.f,
            &g.lo,
            &g.hi,
            g.which.len(),
            &which,
            &off,
            &rows,
            deadline,
        ) {
            Searched::Hints(ing) => hints.push(*ing),
            Searched::Deadline => return WaveResult::Deadline,
            Searched::Latched => return WaveResult::Legacy,
        }
        csr.push(off);
    }
    let t_cert = Instant::now();
    let decided: Vec<Vec<(Fate, ClaimMark)>> = groups
        .par_iter()
        .zip(hints.par_iter())
        .zip(csr.par_iter())
        .map(|((g, ing), off)| {
            let fam = &families[g.f];
            let d = dims[g.f];
            let n = g.which.len();
            let hint_row = |b: usize| off[b] as usize..off[b + 1] as usize;
            let mut is_claimed: Vec<bool> = (0..n)
                .map(|b| fam[g.which[b]].closes(&ing.shifted[hint_row(b)]))
                .collect();
            let mut fates: Vec<Option<(Fate, ClaimMark)>> = (0..n).map(|_| None).collect();
            // Every unclaimed leaf: the clause's candidate test, then exact
            // halves on its hints.
            for b in 0..n {
                if is_claimed[b] {
                    continue;
                }
                let c = g.which[b];
                let r = hint_row(b);
                let lbs: Vec<f64> = r.clone().map(|q| ing.hint_lb(q)).collect();
                let grads = &ing.grads[r.start * ing.d..r.end * ing.d];
                let (l, h) = (&g.lo[b * d..(b + 1) * d], &g.hi[b * d..(b + 1) * d]);
                let fate = shared::open_decide(
                    &fam[c],
                    &w0s[g.f][c],
                    l.to_vec(),
                    h.to_vec(),
                    false,
                    grads,
                    &lbs,
                    cfg,
                );
                // Review fix (an untrusted miss is never a terminal stop): an
                // unsplittable unclaimed leaf would give its clause up on a
                // hint alone; it is certified like a claim instead (the f64
                // bound and the UNCHANGED `decide`).
                if matches!(fate, Fate::Resolution) {
                    is_claimed[b] = true;
                } else {
                    fates[b] = Some((fate, ClaimMark::Unclaimed));
                }
            }
            let claimed: Vec<usize> = (0..n).filter(|&b| is_claimed[b]).collect();
            if !claimed.is_empty() {
                let progs: Vec<&Program> = fam.iter().map(|cl| &cl.prog).collect();
                let which: Vec<usize> = claimed.iter().map(|&b| g.which[b]).collect();
                let lo: Vec<f64> = claimed
                    .iter()
                    .flat_map(|&b| g.lo[b * d..(b + 1) * d].iter().copied())
                    .collect();
                let hi: Vec<f64> = claimed
                    .iter()
                    .flat_map(|&b| g.hi[b * d..(b + 1) * d].iter().copied())
                    .collect();
                let objs: Vec<&[Vec<f64>]> =
                    which.iter().map(|&c| fam[c].objs.as_slice()).collect();
                let bounds =
                    shared::bound_group(&progs, &which, &lo, &hi, &objs, chunks[g.f], false);
                for (j, &b) in claimed.iter().enumerate() {
                    let c = g.which[b];
                    let (l, h) = (&g.lo[b * d..(b + 1) * d], &g.hi[b * d..(b + 1) * d]);
                    let fate = match bounds.get(j) {
                        Some(db) => shared::decide(&fam[c], &w0s[g.f][c], l, h, db, cfg, false),
                        None => Fate::NonFinite,
                    };
                    let mark = if matches!(fate, Fate::Closed | Fate::ClipClosed) {
                        ClaimMark::Certified
                    } else {
                        ClaimMark::Refuted
                    };
                    fates[b] = Some((fate, mark));
                }
            }
            fates
                .into_iter()
                .map(|f| f.unwrap_or((Fate::NonFinite, ClaimMark::None)))
                .collect()
        })
        .collect();
    run.stats.certify_s += t_cert.elapsed().as_secs_f64();
    let (mut claimed, mut cert, mut unclaimed) = (0u64, 0u64, 0u64);
    for (_, m) in decided.iter().flatten() {
        match m {
            ClaimMark::Certified => {
                claimed += 1;
                cert += 1;
            }
            ClaimMark::Refuted => claimed += 1,
            ClaimMark::Unclaimed => unclaimed += 1,
            ClaimMark::None => {}
        }
    }
    run.note(claimed, cert, claimed - cert, unclaimed);
    WaveResult::Done(decided)
}

/// Arm a shared-worklist run: every family's clauses, the objective table
/// (every clause's rows, clause after clause) and one probe per family on
/// its first LIVE clause (`live[f][c]`). `obj_base[f][c]` receives clause
/// `c`'s first row.
pub(crate) fn arm_shared(
    families: &[Vec<FamClause>],
    live: &[Vec<bool>],
    cfg: &LowDimConfig,
) -> (BbeRun, Vec<Vec<u32>>) {
    let mut objectives: Vec<Vec<f64>> = Vec::new();
    let mut obj_base: Vec<Vec<u32>> = Vec::with_capacity(families.len());
    for fam in families {
        let mut base = Vec::with_capacity(fam.len());
        for cl in fam {
            base.push(u32::try_from(objectives.len()).unwrap_or(u32::MAX));
            objectives.extend(cl.objs.iter().cloned());
        }
        obj_base.push(base);
    }
    let progs: Vec<Vec<&Program>> = families
        .iter()
        .map(|f| f.iter().map(|cl| &cl.prog).collect())
        .collect();
    let mut probes = Vec::new();
    for (f, fam) in families.iter().enumerate() {
        let Some(c) = (0..fam.len()).find(|&c| live[f].get(c).copied().unwrap_or(false)) else {
            continue;
        };
        let k = fam[c].objs.len() as u32;
        probes.push(ProbeSpec {
            family: u32::try_from(f).unwrap_or(u32::MAX),
            clause: u32::try_from(c).unwrap_or(u32::MAX),
            lo: &fam[c].lo,
            hi: &fam[c].hi,
            rows: (obj_base[f][c]..obj_base[f][c] + k).collect(),
        });
    }
    if probes.is_empty() {
        let stats = BbeStats {
            backend: "none",
            ..BbeStats::default()
        };
        return (
            BbeRun::fallback(Scheduler::Shared, stats, "nothing"),
            obj_base,
        );
    }
    let run = BbeRun::arm(Scheduler::Shared, &progs, objectives, &probes, cfg);
    (run, obj_base)
}
