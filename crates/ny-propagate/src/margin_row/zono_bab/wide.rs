// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#wide-fd-search` (ENGINE_PROGRAM §2 M4): the frozen-dual tier's level
//! search on an injected UNTRUSTED backend, certified here in CPU f64.
//!
//! # The split
//!
//! `margin_row.frozen_dual_bab_search: f64 | cpu_f32 | cuda | auto` (default
//! `f64`: the parent module's search, untouched — this module is not
//! entered). Otherwise, per class:
//!
//! 1. the class's substituted problem — of whatever state the tier built for
//!    it (`margin_row.frozen_dual_bab_state`: its own per-class build, or its
//!    assembly over the M3 shared trunk, which exists only once the trunk's
//!    class, layout and digest bindings accepted the class's tail) — is copied
//!    to f32 ([`search_copy`]) and uploaded to the backend for this class
//!    only (`SearchBackend::upload_fd`, under a memory lease carved from the
//!    frontier budget, which is already net of a resident trunk), and a
//!    lane-start KNOWN-ANSWER PROBE compares the backend's warm values on
//!    sixteen deterministic nodes against this crate's certified
//!    evaluator (ENGINE_PROGRAM G6). A refusal, a panic or a probe failure
//!    falls back to the f64 search for the class (`auto` first tries the next
//!    backend);
//! 2. each BFS level is handed to the backend in chunks (`fd_level`); per
//!    node it proposes the value and absolute-term sum `S` at the four dual
//!    points (warm, slack-clamped, floor, far — far-first), the clamped and
//!    far multipliers and the children's warm start;
//! 3. a node CLAIMS a certificate when some point's search value exceeds
//!    `-tau`, `tau = 4 gamma_f32(D) S` (the backend's own f32 error scale);
//!    claimed nodes are CERTIFIED on the lane's CPU pool by `eval_block` — the
//!    parent's certified evaluator, bit-identical to `eval_node` — at the
//!    winning point, then at the remaining distinct points in descending search
//!    value, and pruned on the first STRICTLY positive certified value;
//! 4. a claimed node no point certifies is REFUTED: kept and split, never
//!    pruned. An unclaimed node is kept and split. Kept nodes' children inherit
//!    the backend's proposed warm start (sanitised: finite, `>= 0`).
//!
//! # Why every prune is still sound
//!
//! * The backend never sees or changes the TREE: the level (which nodes
//!   exist, their sides and depth) is the parent's own, built from its own
//!   previous level; the sides certified are read from it, not from the
//!   backend.
//! * A prune is decided by `NodeBound::prunes` (strict) on `eval_block`'s
//!   output at the proposed multipliers after `widen` (a NaN, an infinity or
//!   a negative becomes 0). Weak duality makes the certified value a lower
//!   bound of the node's margin at EVERY non-negative point, so how the
//!   backend found the point — or whether its arithmetic was right — is
//!   irrelevant; a wrong proposal can only fail to certify.
//! * The level ledger (`LevelTrace`) counts only these certified prunes and
//!   the parent's `pruned + kept == frontier` check and `ledger_closes` run
//!   unchanged, so `Closed` still means "every leaf certified".
//! * Every other failure is a fallback, never a verdict: a malformed output, a
//!   refusal or a panic scores that chunk (and, having retired the backend,
//!   the rest of the class) with the f64 search; a deadline refusal stops the
//!   search OPEN. The sampled scalar cross-check (G4) of the parent runs on
//!   every prune here too.
//!
//! The device never certifies; ny-propagate never depends on ny-bbe or on a
//! CUDA crate (it sees `&dyn SearchBackend`, registered by ny-cli with
//! [`set_frozen_dual_search_backend_factory`]).
//!
//! # Observables
//!
//! Per class (appended to `[frozen-dual] class=`): `search=` (`f64` or the
//! backend's name), `probe=`, `claimed=`, `refuted=`, `fallback=` (chunks the
//! f64 search scored). Per armed tier, before the summary: `[bbe] armed=1
//! backend= probe= lease_mb= search= classes= fallback_classes= why=`. The
//! summary carries `search= claimed= refuted= fd-wide-search/v1`.

use std::ops::Range;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use rayon::prelude::*;

use ny_core::search::{
    DeviceLease, FdHandle, FdSearchProblem, LevelChunk, LevelOut, LevelView, Refusal,
    SearchBackend, FD_CLAMPED, FD_FAR, FD_FLOOR, FD_LOG_BUCKETS, FD_POINTS, FD_SEARCH_TOL, FD_WARM,
};

use crate::margin_row::zono_state::problem::block::{BlockScratch, BlockWant, BLOCK_NODES};
use crate::zono_core::rounding::{gamma, TINY};

use super::{
    admitted_bytes, block_eval, breakpoint, crosscheck_due, eval_chunk, narrow, past,
    refine_allowed, search_inner_with_relations, slack_decade, widen, Candidate, ChunkCtx,
    ChunkOut, ChunkWork, FrozenDualProblem, Level, NodeBound, NodeOut, NodeScratch, PrunedNode,
    SearchOptions, SearchReport, Side, TestHooks,
};

// A backend steps with ny-core's copies of the f64 search's step rules; a
// drift between them is a compile error, not a silent trajectory change.
const _: () = assert!(super::LOG_BUCKETS == FD_LOG_BUCKETS);
const _: () = assert!(super::SEARCH_TOL.to_bits() == FD_SEARCH_TOL.to_bits());

/// The f64 search's step rules `(SEARCH_TOL, LOG_BUCKETS)` (test access).
#[cfg(test)]
pub(crate) fn search_rules() -> (f64, usize) {
    (super::SEARCH_TOL, super::LOG_BUCKETS)
}

// ---------------------------------------------------------------------------
//  The selector
// ---------------------------------------------------------------------------

/// Which search scores the frozen-dual tier's nodes
/// (`margin_row.frozen_dual_bab_search`).
///
/// The serde tokens are the typed preset key's values and the lever
/// declaration's `Enum` members ([`Self::TOKENS`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum FrozenDualSearch {
    /// The f64 search of the parent module (default): bit-identical to M1, no
    /// backend is consulted.
    #[default]
    #[serde(rename = "f64")]
    F64,
    /// ny-bbe's CPU f32 SIMD level search, certified here.
    #[serde(rename = "cpu_f32")]
    CpuF32,
    /// ny-cuda's device level search, certified here.
    #[serde(rename = "cuda")]
    Cuda,
    /// The device search when it uploads and probes clean, else the CPU f32
    /// search, else f64.
    #[serde(rename = "auto")]
    Auto,
}

impl FrozenDualSearch {
    /// The preset / environment vocabulary, in declaration order.
    pub const TOKENS: [&'static str; 4] = ["f64", "cpu_f32", "cuda", "auto"];

    /// Parse one canonical token (exact spelling).
    #[must_use]
    pub fn from_token(token: &str) -> Option<Self> {
        match token {
            "f64" => Some(Self::F64),
            "cpu_f32" => Some(Self::CpuF32),
            "cuda" => Some(Self::Cuda),
            "auto" => Some(Self::Auto),
            _ => None,
        }
    }

    /// The canonical token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::CpuF32 => "cpu_f32",
            Self::Cuda => "cuda",
            Self::Auto => "auto",
        }
    }
}

// ---------------------------------------------------------------------------
//  The injected backends
// ---------------------------------------------------------------------------

/// Which registered backend a factory builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FdBackendKind {
    /// The CPU f32 search (ny-bbe).
    CpuF32,
    /// The device search (ny-cuda).
    Cuda,
}

type BackendFactory = Box<dyn Fn() -> Option<Arc<dyn SearchBackend>> + Send + Sync>;

static CPU_F32_FACTORY: OnceLock<BackendFactory> = OnceLock::new();
static CUDA_FACTORY: OnceLock<BackendFactory> = OnceLock::new();

/// Register the process's `kind` level-search backend factory (ny-cli, once,
/// at startup; first install wins). The closure runs on the first ARMED class
/// that asks for that backend, so a dark tier constructs nothing. A backend is
/// untrusted: nothing it returns is published without the certification this
/// module performs.
pub fn set_frozen_dual_search_backend_factory<F>(kind: FdBackendKind, factory: F)
where
    F: Fn() -> Option<Arc<dyn SearchBackend>> + Send + Sync + 'static,
{
    let slot = match kind {
        FdBackendKind::CpuF32 => &CPU_F32_FACTORY,
        FdBackendKind::Cuda => &CUDA_FACTORY,
    };
    let _ = slot.set(Box::new(factory));
}

/// The registered `kind` backend, built once per process on first use
/// (`None` when no factory is installed, it declined or it panicked — a
/// declined build is retried by the next class, a built one is kept).
fn registered_backend(kind: FdBackendKind) -> Option<Arc<dyn SearchBackend>> {
    static CPU_F32: OnceLock<Arc<dyn SearchBackend>> = OnceLock::new();
    static CUDA: OnceLock<Arc<dyn SearchBackend>> = OnceLock::new();
    let (factory, handle) = match kind {
        FdBackendKind::CpuF32 => (&CPU_F32_FACTORY, &CPU_F32),
        FdBackendKind::Cuda => (&CUDA_FACTORY, &CUDA),
    };
    if let Some(backend) = handle.get() {
        return Some(Arc::clone(backend));
    }
    let factory = factory.get()?;
    let built = catch_unwind(AssertUnwindSafe(factory)).ok().flatten()?;
    Some(Arc::clone(handle.get_or_init(|| built)))
}

// ---------------------------------------------------------------------------
//  The search copy, the lease, the probe
// ---------------------------------------------------------------------------

/// Nodes of the lane-start known-answer probe.
pub(crate) const WIDE_PROBE_NODES: usize = 16;

/// Depth of the probe's nodes (capped by the problem's split order).
pub(crate) const WIDE_PROBE_DEPTH: usize = 8;

/// The probe's tolerance on a warm value, relative to the certified path's
/// absolute-term sum `S` (a correct f32 search is below `2^-16 S` on every
/// fixture of ny-bbe's tracking test; a layout or sign bug is `O(S)`).
const PROBE_REL_TOL: f64 = 1.0 / 1024.0;

/// Bytes of one [`LevelOut`] chunk the tier aims for.
const WIDE_CHUNK_TARGET_BYTES: u64 = 64 << 20;

/// Nodes per backend call, bounds.
const WIDE_CHUNK_MIN: usize = 256;
const WIDE_CHUNK_MAX: usize = 16_384;

/// `gamma_k` of binary32 (`k u / (1 - k u)`, `u = 2^-24`).
fn gamma_f32(k: usize) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let ku = k as f64 * f64::from(f32::EPSILON) / 2.0;
    if ku < 1.0 {
        ku / (1.0 - ku)
    } else {
        f64::INFINITY
    }
}

/// The claim threshold's relative scale at depth `d`: a point claims a
/// certificate when its search value exceeds `-tau_rel(d) S`. SEARCH POLICY
/// ONLY: any threshold is sound (an unclaimed node is split, a claimed one is
/// certified or split).
pub(crate) fn tau_rel(depth: usize) -> f64 {
    4.0 * gamma_f32(depth)
}

/// Nodes per backend call at `depth`: about [`WIDE_CHUNK_TARGET_BYTES`] of
/// output, within `256..=16384`.
fn chunk_nodes(depth: usize) -> usize {
    let per_node = LevelOut::bytes_for(1, depth).max(1);
    let fit = usize::try_from(WIDE_CHUNK_TARGET_BYTES / per_node).unwrap_or(usize::MAX);
    fit.clamp(WIDE_CHUNK_MIN, WIDE_CHUNK_MAX)
}

/// The f32 SEARCH copy of `prob` (its split rows in split order). Fails when a
/// value does not survive the rounding (an f32 overflow) — the class then
/// runs the f64 search.
pub(crate) fn search_copy(prob: &FrozenDualProblem) -> Result<FdSearchProblem, String> {
    let n = prob.n();
    let depth = prob.max_depth();
    #[allow(clippy::cast_possible_truncation)]
    let f = |v: f64| v as f32;
    let pair = |v: [f64; 2]| [f(v[0]), f(v[1])];
    let abs_d: f64 = prob.d().iter().map(|v| v.abs()).sum();
    let mut copy = FdSearchProblem {
        n,
        depth,
        a: Vec::with_capacity(depth.saturating_mul(n)),
        enc: Vec::with_capacity(depth),
        rho: Vec::with_capacity(depth),
        c0s: Vec::with_capacity(depth),
        b0: Vec::with_capacity(depth),
        b1: Vec::with_capacity(depth),
        row_abs: Vec::with_capacity(depth),
        d: prob.d().iter().map(|&v| f(v)).collect(),
        c0: f(prob.c0()),
        abs_d: f(abs_d),
    };
    for i in 0..depth {
        let row = prob.split_row(i);
        copy.a.extend(row.a.iter().map(|&v| f(v)));
        copy.enc
            .push(u32::try_from(row.enc).map_err(|_| "own column beyond u32".to_string())?);
        copy.rho.push(pair(row.rho));
        copy.c0s.push(pair(row.c0s));
        copy.b0.push(pair(row.b0));
        copy.b1.push(pair(row.b1));
        copy.row_abs.push(f(row.row_abs));
    }
    copy.validate()?;
    Ok(copy)
}

/// splitmix64 step (the probe's deterministic nodes).
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The probe's nodes at `depth`: `(sides, l0, l1)`, row-major, a quarter of
/// the multipliers exact zeros (deterministic in `(n, depth)`).
fn probe_nodes(n: usize, depth: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let len = WIDE_PROBE_NODES * depth;
    let mut state = (n as u64) ^ ((depth as u64) << 40) ^ 0x00FD_4B0B;
    let (mut sides, mut l0, mut l1) = (
        Vec::with_capacity(len),
        Vec::with_capacity(len),
        Vec::with_capacity(len),
    );
    #[allow(clippy::cast_precision_loss)]
    let m = |r: u64, shift: u32| {
        if (r >> shift) & 3 == 0 {
            0.0f32
        } else {
            ((r >> (shift + 2)) & 0xffff) as f32 / 32_768.0
        }
    };
    for _ in 0..len {
        let r = splitmix(&mut state);
        sides.push(u8::from(r & 1 == 1));
        l0.push(m(r, 8));
        l1.push(m(r, 30));
    }
    (sides, l0, l1)
}

/// The lane-start known-answer probe (G6): the backend's WARM value and `S`
/// on the probe nodes against this crate's certified evaluator's raw value
/// and the `S` its charge was computed from. `Err` names the first mismatch.
fn probe(
    prob: &FrozenDualProblem,
    backend: &dyn SearchBackend,
    handle: &FdHandle,
    deadline: Option<Instant>,
) -> Result<(), String> {
    let d = prob.max_depth().min(WIDE_PROBE_DEPTH);
    if d == 0 {
        return Ok(());
    }
    let n = prob.n();
    let (sides, l0, l1) = probe_nodes(n, d);
    let chunk = LevelChunk {
        depth: d,
        count: WIDE_PROBE_NODES,
        sides: &sides,
        l0: &l0,
        l1: &l1,
        deadline,
    };
    let mut out = LevelOut::default();
    match catch_unwind(AssertUnwindSafe(|| {
        backend.fd_level(handle, &chunk, &mut out)
    })) {
        Ok(Ok(_)) => {}
        Ok(Err(why)) => return Err(format!("probe refused ({why})")),
        Err(_) => return Err("probe panicked".to_string()),
    }
    let view = out.hint().into_ingest();
    if !(view.well_formed() && view.count() == WIDE_PROBE_NODES && view.depth() == d) {
        return Err("probe output malformed".to_string());
    }
    let charge_per_s = gamma(n + 3 * d + 32) * 1.001;
    let mut scratch = NodeScratch::default();
    let mut s = vec![Side::Off; d];
    let (mut m0, mut m1) = (vec![0.0f64; d], vec![0.0f64; d]);
    let mut compared = 0usize;
    for k in 0..WIDE_PROBE_NODES {
        for i in 0..d {
            s[i] = if sides[k * d + i] == 0 {
                Side::Off
            } else {
                Side::On
            };
            m0[i] = widen(l0[k * d + i]);
            m1[i] = widen(l1[k * d + i]);
        }
        let reference = prob.eval_node(&s, &m0, &m1, &mut scratch, None);
        if !reference.value.is_finite() {
            continue;
        }
        let big_s = ((reference.charge - TINY) / charge_per_s).max(0.0);
        let v = f64::from(view.value(k, FD_WARM));
        let sv = f64::from(view.scale(k, FD_WARM));
        let tol = PROBE_REL_TOL * big_s + 1e-300;
        if !(v.is_finite() && (v - reference.value).abs() <= tol) {
            return Err(format!(
                "probe node {k}: warm value {v:e} vs the certified path's {:e} (S {big_s:e})",
                reference.value
            ));
        }
        if !(sv.is_finite() && (sv - big_s).abs() <= 4.0 * tol) {
            return Err(format!(
                "probe node {k}: warm S {sv:e} vs the certified path's {big_s:e}"
            ));
        }
        compared += 1;
    }
    if compared == 0 {
        return Err("probe compared no node".to_string());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
//  One class on one backend
// ---------------------------------------------------------------------------

/// What the level loop needs from the backend (read-only there, atomics for
/// the refusal latch so the context stays `Sync`).
pub(crate) struct WideCtx<'a> {
    backend: &'a dyn SearchBackend,
    handle: &'a FdHandle,
    chunk_nodes: usize,
    lease_bytes: u64,
    /// Set by the first non-deadline refusal, malformed output or panic: the
    /// rest of the class runs the f64 search.
    dead: AtomicBool,
    fallback_chunks: AtomicU64,
    why: OnceLock<String>,
}

impl WideCtx<'_> {
    /// Bytes the wide path adds to a level of `count` nodes of `depth` (one
    /// output chunk plus the backend's lease), admitted with the level.
    pub(crate) fn extra_bytes(&self, count: usize, depth: usize) -> u64 {
        LevelOut::bytes_for(self.chunk_nodes.min(count), depth).saturating_add(self.lease_bytes)
    }

    fn retire(&self, why: String) {
        self.dead.store(true, Ordering::Relaxed);
        let _ = self.why.set(why);
    }
}

/// A backend ready for a class.
struct Prepared {
    handle: FdHandle,
    lease_bytes: u64,
}

/// Why a backend could not take a class: `(probe status, reason)`.
type Failed = (&'static str, String);

/// Copy, upload and probe `prob` on `backend`.
fn prepare(
    prob: &FrozenDualProblem,
    opts: &SearchOptions,
    backend: &dyn SearchBackend,
) -> Result<Prepared, Failed> {
    let copy = search_copy(prob).map_err(|why| ("refused", why))?;
    let lease_bytes = admitted_bytes(opts.max_frontier_bytes) / 4;
    let lease = DeviceLease { bytes: lease_bytes };
    let handle = match catch_unwind(AssertUnwindSafe(|| backend.upload_fd(&copy, &lease))) {
        Ok(Ok(handle)) => handle,
        Ok(Err(why)) => return Err(("refused", why.to_string())),
        Err(_) => return Err(("refused", "upload panicked".to_string())),
    };
    drop(copy);
    probe(prob, backend, &handle, opts.deadline).map_err(|why| ("fail", why))?;
    Ok(Prepared {
        handle,
        lease_bytes,
    })
}

/// How one class went (for the tier's `[bbe]` line).
struct Outcome {
    /// The backend that searched, `None` when the f64 search ran.
    backend: Option<&'static str>,
    probe: &'static str,
    why: Option<String>,
    lease_bytes: u64,
}

/// True when the class needs no level (its root certifies, or it has no split
/// row): the f64 search decides it without a backend.
fn root_decides(prob: &FrozenDualProblem) -> bool {
    prob.max_depth() == 0
        || prob
            .eval_node(&[], &[], &[], &mut NodeScratch::default(), None)
            .prunes()
}

/// Search `prob` on the first of `backends` that uploads and probes clean,
/// else with the f64 search.
fn search_backends(
    prob: &FrozenDualProblem,
    opts: &SearchOptions,
    backends: &[&dyn SearchBackend],
    hooks: TestHooks,
) -> (SearchReport, Outcome) {
    search_backends_with_relations(prob, opts, backends, hooks, None)
}

fn search_backends_with_relations(
    prob: &FrozenDualProblem,
    opts: &SearchOptions,
    backends: &[&dyn SearchBackend],
    hooks: TestHooks,
    relations: Option<&super::relations::ClassHints>,
) -> (SearchReport, Outcome) {
    let registered_refiner = if opts.refine_steps > 0 {
        super::super::root_search::registered_backend()
    } else {
        None
    };
    let mut outcome = Outcome {
        backend: None,
        probe: "refused",
        why: backends
            .is_empty()
            .then(|| "no backend registered".to_string()),
        lease_bytes: 0,
    };
    if root_decides(prob) {
        let mut rep = search_inner_with_relations(
            prob,
            opts,
            hooks,
            None,
            registered_refiner.as_deref(),
            relations,
        );
        rep.probe = "-";
        outcome.probe = "-";
        outcome.why = None;
        return (rep, outcome);
    }
    for &backend in backends {
        match prepare(prob, opts, backend) {
            Ok(prepared) => {
                let ctx = WideCtx {
                    backend,
                    handle: &prepared.handle,
                    chunk_nodes: chunk_nodes(prob.max_depth()),
                    lease_bytes: prepared.lease_bytes,
                    dead: AtomicBool::new(false),
                    fallback_chunks: AtomicU64::new(0),
                    why: OnceLock::new(),
                };
                // A separately registered CPU refiner can compose with CUDA level
                // search. An injected backend can supply both surfaces when
                // there is no registered refiner; Unsupported stays a refusal.
                let refiner = if opts.refine_steps > 0 {
                    registered_refiner.as_deref().or(Some(backend))
                } else {
                    None
                };
                let mut rep =
                    search_inner_with_relations(prob, opts, hooks, Some(&ctx), refiner, relations);
                rep.search = backend.name();
                rep.probe = "ok";
                rep.fallback_chunks = ctx.fallback_chunks.load(Ordering::Relaxed);
                outcome.backend = Some(backend.name());
                outcome.probe = "ok";
                outcome.lease_bytes = prepared.lease_bytes;
                if outcome.why.is_none() {
                    outcome.why = ctx.why.get().cloned();
                }
                return (rep, outcome);
            }
            Err((status, why)) => {
                outcome.probe = status;
                if outcome.why.is_none() {
                    outcome.why = Some(format!("{}: {why}", backend.name()));
                }
            }
        }
    }
    let mut rep = search_inner_with_relations(
        prob,
        opts,
        hooks,
        None,
        registered_refiner.as_deref(),
        relations,
    );
    rep.probe = outcome.probe;
    (rep, outcome)
}

/// [`super::search`] with the M4 level search on `backend`: upload, probe,
/// then every level scored by the backend and certified here (the f64 search
/// when the backend refuses). The report's `search` / `probe` fields say which
/// ran.
#[must_use]
pub fn search_with_backend(
    prob: &FrozenDualProblem,
    opts: &SearchOptions,
    backend: &dyn SearchBackend,
) -> SearchReport {
    search_backends(prob, opts, &[backend], TestHooks::OFF).0
}

/// [`search_with_backend`] with the parent's test hooks: the charge-off and
/// fault-injection pins, and the claim-as-prune MUTANT (prunes a claimed node
/// on the backend's search value, no certification).
#[cfg(test)]
pub(crate) fn search_wide_with_test_hooks(
    prob: &FrozenDualProblem,
    opts: &SearchOptions,
    backend: &dyn SearchBackend,
    charge_off: bool,
    fault_ulp: bool,
    claim_prunes: bool,
) -> SearchReport {
    let hooks = TestHooks {
        charge_off,
        fault_ulp,
        claim_prunes,
    };
    search_backends(prob, opts, &[backend], hooks).0
}

// ---------------------------------------------------------------------------
//  The tier's backend state
// ---------------------------------------------------------------------------

/// A test injection: the backends to try for ONE class's problem (in order).
type InjectedBackends =
    Box<dyn Fn(&FrozenDualProblem) -> Vec<Arc<dyn SearchBackend>> + Send + Sync>;

/// The armed tier's M4 state across its classes (and the `[bbe]` line).
///
/// It carries no class's data across classes: every class's search copy is
/// made from, uploaded for and probed against that class's own problem inside
/// [`Self::search_class`], and the handle is dropped with it — whichever way
/// (per class, or over the M3 shared trunk) the tier built the class's state.
pub(crate) struct WideTier {
    mode: FrozenDualSearch,
    /// Test injection; `None` resolves the registered backends.
    injected: Option<InjectedBackends>,
    backend: &'static str,
    probe: &'static str,
    why: Option<String>,
    lease_mb: u64,
    searched: u64,
    fallback_classes: u64,
}

/// Severity order of the tier's probe summary.
fn probe_rank(status: &str) -> u8 {
    match status {
        "ok" => 1,
        "refused" => 2,
        "fail" => 3,
        _ => 0,
    }
}

impl WideTier {
    /// The state of a tier armed with `mode` (not `F64`).
    pub(crate) fn new(mode: FrozenDualSearch) -> Self {
        Self {
            mode,
            injected: None,
            backend: "-",
            probe: "none",
            why: None,
            lease_mb: 0,
            searched: 0,
            fallback_classes: 0,
        }
    }

    /// A tier searching on `backends` (in order) instead of the registry.
    #[cfg(test)]
    pub(crate) fn with_backends(
        mode: FrozenDualSearch,
        backends: Vec<Arc<dyn SearchBackend>>,
    ) -> Self {
        Self::with_backend_factory(mode, move |_| backends.clone())
    }

    /// A tier whose backends for each class are `factory(problem)` (in order)
    /// instead of the registry: a whole tier's classes on test backends built
    /// for each class's own problem (the reference and rogue backends score
    /// the problem they were built from).
    #[cfg(test)]
    pub(crate) fn with_backend_factory<F>(mode: FrozenDualSearch, factory: F) -> Self
    where
        F: Fn(&FrozenDualProblem) -> Vec<Arc<dyn SearchBackend>> + Send + Sync + 'static,
    {
        Self {
            injected: Some(Box::new(factory)),
            ..Self::new(mode)
        }
    }

    /// The backends to try for `prob`, in preference order.
    fn backends(&self, prob: &FrozenDualProblem) -> Vec<Arc<dyn SearchBackend>> {
        if let Some(factory) = &self.injected {
            return factory(prob);
        }
        let kinds: &[FdBackendKind] = match self.mode {
            FrozenDualSearch::F64 => &[],
            FrozenDualSearch::CpuF32 => &[FdBackendKind::CpuF32],
            FrozenDualSearch::Cuda => &[FdBackendKind::Cuda],
            FrozenDualSearch::Auto => &[FdBackendKind::Cuda, FdBackendKind::CpuF32],
        };
        kinds
            .iter()
            .filter_map(|&k| registered_backend(k))
            .collect()
    }

    /// Search one class (see the module docs).
    pub(crate) fn search_class(
        &mut self,
        prob: &FrozenDualProblem,
        opts: &SearchOptions,
    ) -> SearchReport {
        self.search_class_with_relations(prob, opts, None)
    }

    pub(super) fn search_class_with_relations(
        &mut self,
        prob: &FrozenDualProblem,
        opts: &SearchOptions,
        relations: Option<&super::relations::ClassHints>,
    ) -> SearchReport {
        let list = self.backends(prob);
        let refs: Vec<&dyn SearchBackend> = list.iter().map(|b| b.as_ref()).collect();
        let (rep, outcome) =
            search_backends_with_relations(prob, opts, &refs, TestHooks::OFF, relations);
        match outcome.backend {
            Some(name) => {
                self.backend = name;
                self.searched += 1;
                self.lease_mb = self.lease_mb.max(outcome.lease_bytes >> 20);
            }
            None if outcome.probe != "-" => self.fallback_classes += 1,
            None => {}
        }
        let status = if outcome.backend.is_some() {
            "ok"
        } else {
            outcome.probe
        };
        if probe_rank(status) > probe_rank(self.probe) {
            self.probe = status;
        }
        if self.why.is_none() {
            self.why = outcome.why;
        }
        rep
    }

    /// The `[bbe]` line (one per armed tier, before the summary).
    pub(crate) fn emit(&self) {
        let why: String = self
            .why
            .as_deref()
            .unwrap_or("-")
            .chars()
            .map(|c| if c.is_whitespace() { '_' } else { c })
            .take(160)
            .collect();
        eprintln!(
            "[bbe] armed=1 backend={} probe={} lease_mb={} search={} classes={} \
             fallback_classes={} why={why}",
            self.backend,
            self.probe,
            self.lease_mb,
            self.mode.token(),
            self.searched,
            self.fallback_classes,
        );
    }

    /// The tier's probe summary (`none`, `ok`, `refused`, `fail`).
    #[cfg(test)]
    pub(crate) fn probe_status(&self) -> &'static str {
        self.probe
    }
}

// ---------------------------------------------------------------------------
//  One level
// ---------------------------------------------------------------------------

/// A work item stopped by the deadline (the parent then stops OPEN).
fn aborted() -> ChunkOut {
    let mut out = ChunkOut::new();
    out.aborted = true;
    out
}

/// Score every node of `level` on the backend in chunks and certify its
/// proposals; a chunk the backend cannot score runs the f64 search. Work items
/// come back IN NODE ORDER, as the parent's own evaluation does.
pub(super) fn eval_level(
    prob: &FrozenDualProblem,
    level: &Level,
    ctx: &ChunkCtx,
    chunk: usize,
    w: &WideCtx<'_>,
) -> Vec<ChunkOut> {
    let (depth, count) = (level.depth, level.count);
    let mut outs: Vec<ChunkOut> = Vec::new();
    let mut buf = LevelOut::default();
    let mut lo = 0usize;
    while lo < count {
        let hi = lo.saturating_add(w.chunk_nodes.max(1)).min(count);
        if past(ctx.deadline) {
            outs.push(aborted());
            return outs;
        }
        let mut scored = false;
        if !w.dead.load(Ordering::Relaxed) {
            let r = lo * depth..hi * depth;
            let input = LevelChunk {
                depth,
                count: hi - lo,
                sides: &level.sides[r.clone()],
                l0: &level.l0[r.clone()],
                l1: &level.l1[r],
                deadline: ctx.deadline,
            };
            match catch_unwind(AssertUnwindSafe(|| {
                w.backend.fd_level(w.handle, &input, &mut buf)
            })) {
                Ok(Ok(_stats)) => {
                    let view = buf.hint().into_ingest();
                    if view.well_formed() && view.count() == hi - lo && view.depth() == depth {
                        outs.extend(certify_range(prob, level, &view, lo..hi, ctx, chunk));
                        scored = true;
                    } else {
                        w.retire("the backend returned a malformed level".to_string());
                    }
                }
                Ok(Err(Refusal::Deadline)) => {
                    outs.push(aborted());
                    return outs;
                }
                Ok(Err(why)) => w.retire(why.to_string()),
                Err(_) => w.retire("the backend panicked".to_string()),
            }
        }
        if !scored {
            w.fallback_chunks.fetch_add(1, Ordering::Relaxed);
            outs.extend(fallback_range(prob, level, lo..hi, ctx, chunk));
        }
        lo = hi;
    }
    outs
}

/// The parent's f64 evaluation of `range` (the unchanged M1 work items).
fn fallback_range(
    prob: &FrozenDualProblem,
    level: &Level,
    range: Range<usize>,
    ctx: &ChunkCtx,
    chunk: usize,
) -> Vec<ChunkOut> {
    let (n, depth) = (prob.n(), level.depth);
    let chunk = chunk.max(1);
    (0..range.len().div_ceil(chunk))
        .into_par_iter()
        .map_init(
            || ChunkWork::new(n, depth, ctx.evaluator),
            |cw, c| {
                let a = range.start + c * chunk;
                let b = (a + chunk).min(range.end);
                eval_chunk(prob, level, a..b, ctx, cw)
            },
        )
        .collect()
}

/// Certify the backend's proposals for `range` (level indices; the view's
/// node 0 is `range.start`) in work items of `chunk` nodes.
fn certify_range(
    prob: &FrozenDualProblem,
    level: &Level,
    view: &LevelView<'_>,
    range: Range<usize>,
    ctx: &ChunkCtx,
    chunk: usize,
) -> Vec<ChunkOut> {
    let depth = level.depth;
    let chunk = chunk.max(1);
    let view_lo = range.start;
    (0..range.len().div_ceil(chunk))
        .into_par_iter()
        .map_init(
            || CertWork::new(depth),
            |cw, c| {
                let a = range.start + c * chunk;
                let b = (a + chunk).min(range.end);
                certify_item(prob, level, view, view_lo, a..b, ctx, cw)
            },
        )
        .collect()
}

// ---------------------------------------------------------------------------
//  Certification of one work item
// ---------------------------------------------------------------------------

/// One node's certification plan.
#[derive(Debug, Clone, Copy)]
struct Plan {
    /// Level index.
    idx: usize,
    /// Index in the backend's view.
    k: usize,
    /// Points to certify, in order (the winner first), deduplicated.
    order: [usize; FD_POINTS],
    len: usize,
    /// Next point to try.
    pos: usize,
    /// Some point's search value claimed a certificate.
    claimed: bool,
    /// The point that certified and its certified value.
    won: Option<(usize, f64)>,
    /// The search values (search signal only).
    vals: [f64; FD_POINTS],
}

/// Per-thread buffers of the certification.
struct CertWork {
    ws: BlockScratch,
    scratch: NodeScratch,
    plans: Vec<Plan>,
    lanes: Vec<(usize, usize)>,
    g_sides: Vec<Side>,
    g_l0: Vec<f64>,
    g_l1: Vec<f64>,
    g_out: Vec<NodeBound>,
    /// Candidate multipliers of the node being planned (dedup), `4 x 2 depth`.
    cand: Vec<f64>,
    m0: Vec<f64>,
    m1: Vec<f64>,
}

impl CertWork {
    fn new(depth: usize) -> Self {
        let b = BLOCK_NODES;
        Self {
            ws: BlockScratch::default(),
            scratch: NodeScratch::default(),
            plans: Vec::new(),
            lanes: Vec::new(),
            g_sides: vec![Side::Off; b * depth],
            g_l0: vec![0.0; b * depth],
            g_l1: vec![0.0; b * depth],
            g_out: vec![NodeBound::refused(); b],
            cand: vec![0.0; FD_POINTS * 2 * depth],
            m0: vec![0.0; depth],
            m1: vec![0.0; depth],
        }
    }
}

/// The certification multipliers of `point` of the node at level index `idx`
/// (view index `k`): the INGEST GATE — every value through `widen` (finite,
/// `>= 0`; anything else becomes 0).
fn point_multipliers(
    level: &Level,
    view: &LevelView<'_>,
    idx: usize,
    k: usize,
    point: usize,
    m0: &mut [f64],
    m1: &mut [f64],
) {
    let d = level.depth;
    let (a, b): (&[f32], &[f32]) = match point {
        FD_WARM => (
            &level.l0[idx * d..(idx + 1) * d],
            &level.l1[idx * d..(idx + 1) * d],
        ),
        FD_CLAMPED => view.clamped(k),
        FD_FAR => view.far(k),
        _ => {
            m0.fill(0.0);
            m1.fill(0.0);
            return;
        }
    };
    for (dst, &v) in m0.iter_mut().zip(a) {
        *dst = widen(v);
    }
    for (dst, &v) in m1.iter_mut().zip(b) {
        *dst = widen(v);
    }
}

fn candidate_of(point: usize) -> Candidate {
    match point {
        FD_WARM => Candidate::Warm,
        FD_CLAMPED => Candidate::Clamped,
        FD_FLOOR => Candidate::Floor,
        _ => Candidate::Far,
    }
}

fn load_sides(level: &Level, idx: usize, out: &mut [Side]) {
    let d = level.depth;
    for (dst, &s) in out.iter_mut().zip(&level.sides[idx * d..(idx + 1) * d]) {
        *dst = if s == 0 { Side::Off } else { Side::On };
    }
}

/// Plan node `idx`: its claims, the certification order and the dedup.
fn plan_node(
    level: &Level,
    view: &LevelView<'_>,
    idx: usize,
    k: usize,
    rel: f64,
    cw: &mut CertWork,
) -> Plan {
    let d = level.depth;
    let mut vals = [f64::NAN; FD_POINTS];
    let mut winner: Option<usize> = None;
    // The claimed winner's value, tracked alongside `winner` so the argmax
    // (strict `>`: the earliest point keeps a tie) never re-reads `vals`
    // while it is mutably borrowed by the loop.
    let mut winner_val = f64::NEG_INFINITY;
    for (p, slot) in vals.iter_mut().enumerate() {
        let v = f64::from(view.value(k, p));
        let s = f64::from(view.scale(k, p));
        *slot = v;
        let claims = v.is_finite() && s.is_finite() && s >= 0.0 && v > -(rel * s);
        if claims && winner.is_none_or(|_| v > winner_val) {
            winner = Some(p);
            winner_val = v;
        }
    }
    let mut plan = Plan {
        idx,
        k,
        order: [0; FD_POINTS],
        len: 0,
        pos: 0,
        claimed: winner.is_some(),
        won: None,
        vals,
    };
    let Some(first) = winner else {
        return plan;
    };
    // The rest in descending search value, NaN last (stable by index).
    let mut rest = [usize::MAX; FD_POINTS - 1];
    let mut r = 0usize;
    for p in 0..FD_POINTS {
        if p != first {
            rest[r] = p;
            r += 1;
        }
    }
    let key = |p: usize| {
        let v = vals[p];
        if v.is_nan() {
            f64::NEG_INFINITY
        } else {
            v
        }
    };
    rest.sort_by(|&x, &y| key(y).total_cmp(&key(x)).then(x.cmp(&y)));
    // Keep each point whose multipliers differ from every earlier one.
    let span = 2 * d;
    for p in std::iter::once(first).chain(rest) {
        let slot = plan.len * span;
        let (c0, c1) = cw.cand[slot..slot + span].split_at_mut(d);
        point_multipliers(level, view, idx, k, p, c0, c1);
        let fresh = (0..plan.len).all(|q| {
            let prev = &cw.cand[q * span..(q + 1) * span];
            let this = &cw.cand[slot..slot + span];
            prev.iter()
                .zip(this)
                .any(|(a, b)| a.to_bits() != b.to_bits())
        });
        if fresh {
            plan.order[plan.len] = p;
            plan.len += 1;
        }
    }
    plan
}

/// Certify the claimed nodes of `range` (see the module docs) and emit the
/// work item's results in node order.
fn certify_item(
    prob: &FrozenDualProblem,
    level: &Level,
    view: &LevelView<'_>,
    view_lo: usize,
    range: Range<usize>,
    ctx: &ChunkCtx,
    cw: &mut CertWork,
) -> ChunkOut {
    let d = level.depth;
    let mut out = ChunkOut::new();
    if past(ctx.deadline) {
        out.aborted = true;
        return out;
    }
    let rel = tau_rel(d);
    cw.plans.clear();
    for idx in range.clone() {
        let plan = plan_node(level, view, idx, idx - view_lo, rel, cw);
        cw.plans.push(plan);
    }
    if ctx.hooks.claim_prunes() {
        // MUTANT (test builds only): the claim's own search value decides.
        for plan in &mut cw.plans {
            if plan.claimed && plan.len > 0 {
                let p = plan.order[0];
                plan.won = Some((p, plan.vals[p]));
            }
        }
    }
    // Rounds: every open claimed node's next point, BLOCK_NODES per block.
    loop {
        cw.lanes.clear();
        for (pi, plan) in cw.plans.iter().enumerate() {
            if plan.claimed && plan.won.is_none() && plan.pos < plan.len {
                cw.lanes.push((pi, plan.order[plan.pos]));
            }
        }
        if cw.lanes.is_empty() {
            break;
        }
        if past(ctx.deadline) {
            out.aborted = true;
            return out;
        }
        for group in cw.lanes.chunks(BLOCK_NODES) {
            let m = group.len();
            for (lane, &(pi, point)) in group.iter().enumerate() {
                let plan = cw.plans[pi];
                let r = lane * d..(lane + 1) * d;
                load_sides(level, plan.idx, &mut cw.g_sides[r.clone()]);
                point_multipliers(
                    level,
                    view,
                    plan.idx,
                    plan.k,
                    point,
                    &mut cw.g_l0[r.clone()],
                    &mut cw.g_l1[r],
                );
            }
            block_eval(
                prob,
                ctx.hooks,
                d,
                &cw.g_sides[..m * d],
                &cw.g_l0[..m * d],
                &cw.g_l1[..m * d],
                BlockWant {
                    rc: false,
                    subgrad: false,
                },
                &mut cw.ws,
                &mut [],
                &mut [],
                &mut cw.g_out[..m],
            );
            for (lane, &(pi, point)) in group.iter().enumerate() {
                let nb = cw.g_out[lane];
                let plan = &mut cw.plans[pi];
                if nb.prunes() {
                    plan.won = Some((point, nb.certified));
                } else {
                    plan.pos += 1;
                }
            }
        }
    }
    // Emit, in node order.
    for pi in 0..cw.plans.len() {
        let plan = cw.plans[pi];
        let v = plan.vals;
        let finite_max = |ps: &[usize]| {
            ps.iter()
                .map(|&p| v[p])
                .filter(|x| x.is_finite())
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let pre = finite_max(&[FD_WARM, FD_CLAMPED, FD_FLOOR]);
        let best = finite_max(&[FD_WARM, FD_CLAMPED, FD_FLOOR, FD_FAR]);
        let far_used = v[FD_FAR].is_finite() && v[FD_FAR] > pre;
        if far_used {
            out.far_used += 1;
        }
        if plan.claimed {
            out.claimed += 1;
        }
        let mut node = [NodeOut::refused()];
        if let Some((point, certified)) = plan.won {
            node[0] = NodeOut {
                best,
                certified,
                winner: candidate_of(point),
                far_used,
            };
            ctx.hooks.inject(&mut node);
        }
        let node = node[0];
        if plan.won.is_some() && node.prunes() {
            let point = plan.won.map_or(FD_WARM, |w| w.0);
            out.pruned += 1;
            out.cert_min = out.cert_min.min(node.certified);
            out.slack_hist[slack_decade(node.certified)] += 1;
            let sides = &mut cw.g_sides[..d];
            load_sides(level, plan.idx, sides);
            point_multipliers(level, view, plan.idx, plan.k, point, &mut cw.m0, &mut cw.m1);
            if crosscheck_due(ctx.crosscheck_every, d, plan.idx) {
                out.xchk += 1;
                let reference = prob.eval_node(sides, &cw.m0, &cw.m1, &mut cw.scratch, None);
                if reference.certified.to_bits() != node.certified.to_bits() {
                    if out.xchk_fail == 0 {
                        eprintln!(
                            "[frozen-dual] CROSSCHECK-MISMATCH depth={d} idx={} block={} \
                             ref={:e} got={:e} search=wide",
                            plan.idx, range.start, reference.certified, node.certified
                        );
                    }
                    out.xchk_fail += 1;
                }
            }
            if out.records.len() < ctx.record_cap {
                out.records.push(PrunedNode {
                    sides: sides.to_vec(),
                    l0: cw.m0.clone(),
                    l1: cw.m1.clone(),
                    certified: node.certified,
                    candidate: node.winner,
                });
            }
            continue;
        }
        // A successful wide call can still leave a node open. Its proposed
        // child warm start is a scheduling hint, so certify that exact point
        // from the original problem before using it as the refiner's incumbent.
        // The disabled path below keeps the original output bits unchanged.
        let allow_breakpoint = refine_allowed(ctx, prob.n(), d, &mut out.refinement);
        let relation_matches = ctx
            .relations
            .is_some_and(|h| h.matches_bytes(&level.sides[plan.idx * d..(plan.idx + 1) * d]));
        if relation_matches || allow_breakpoint {
            let (a, b) = view.next(plan.k);
            for (dst, &value) in cw.m0.iter_mut().zip(a) {
                *dst = widen(value);
            }
            for (dst, &value) in cw.m1.iter_mut().zip(b) {
                *dst = widen(value);
            }
            let sides = &mut cw.g_sides[..d];
            load_sides(level, plan.idx, sides);
            let warm = prob.eval_node(sides, &cw.m0, &cw.m1, &mut cw.scratch, None);
            let mut winner = Candidate::Relation;
            let mut accepted = ctx.relations.and_then(|relations| {
                relations.propose(
                    prob,
                    sides,
                    &cw.m0,
                    &cw.m1,
                    warm.certified,
                    ctx.deadline,
                    &mut cw.scratch,
                    &mut out.relations,
                )
            });
            if allow_breakpoint && accepted.as_ref().is_none_or(|h| !h.bound.prunes()) {
                let (m0, m1, incumbent) = accepted
                    .as_ref()
                    .map_or((&cw.m0[..], &cw.m1[..], warm.certified), |h| {
                        (&h.l0[..], &h.l1[..], h.bound.certified)
                    });
                if let Some(hint) = breakpoint::refine_with_policy(
                    prob,
                    sides,
                    m0,
                    m1,
                    incumbent,
                    ctx.refine_steps,
                    ctx.automatic_refine,
                    ctx.deadline,
                    ctx.backend,
                    &mut cw.scratch,
                    &mut out.refinement,
                ) {
                    accepted = Some(hint);
                    winner = breakpoint::CANDIDATE;
                }
            }
            if let Some(hint) = accepted {
                let mut refined = [NodeOut {
                    best: best.max(hint.bound.value),
                    certified: hint.bound.certified,
                    winner,
                    far_used,
                }];
                ctx.hooks.inject(&mut refined);
                let node = refined[0];
                if node.prunes() {
                    out.pruned += 1;
                    out.cert_min = out.cert_min.min(node.certified);
                    out.slack_hist[slack_decade(node.certified)] += 1;
                    if crosscheck_due(ctx.crosscheck_every, d, plan.idx) {
                        out.xchk += 1;
                        let reference =
                            prob.eval_node(sides, &hint.l0, &hint.l1, &mut cw.scratch, None);
                        if reference.certified.to_bits() != node.certified.to_bits() {
                            if out.xchk_fail == 0 {
                                eprintln!(
                                    "[frozen-dual] CROSSCHECK-MISMATCH depth={d} idx={} block={} \
                                     ref={:e} got={:e} search=wide-proposal",
                                    plan.idx, range.start, reference.certified, node.certified
                                );
                            }
                            out.xchk_fail += 1;
                        }
                    }
                    if out.records.len() < ctx.record_cap {
                        out.records.push(PrunedNode {
                            sides: sides.to_vec(),
                            l0: hint.l0,
                            l1: hint.l1,
                            certified: node.certified,
                            candidate: node.winner,
                        });
                    }
                    continue;
                }
                // An accepted, still-negative certificate improves the next
                // level's warm start without dropping any child from its cover.
                if plan.claimed {
                    out.refuted += 1;
                }
                out.kept.push(plan.idx);
                out.next_l0.extend(hint.l0.iter().map(|&x| narrow(x)));
                out.next_l1.extend(hint.l1.iter().map(|&x| narrow(x)));
                out.worst_open = out.worst_open.min(node.best);
                continue;
            }
        }
        if plan.claimed {
            out.refuted += 1;
        }
        let (a, b) = view.next(plan.k);
        out.kept.push(plan.idx);
        out.next_l0.extend(a.iter().map(|&x| narrow(widen(x))));
        out.next_l1.extend(b.iter().map(|&x| narrow(widen(x))));
        out.worst_open = out.worst_open.min(best);
    }
    out
}

// ---------------------------------------------------------------------------
//  Test support: the reference (honest f64) backend
// ---------------------------------------------------------------------------

/// A test backend that scores every node with the PARENT'S OWN f64 search
/// step (`eval_search_node`, all four points, no far-first) on a copy of the
/// problem: the reference the wide path must reproduce.
#[cfg(test)]
pub(crate) struct ReferenceBackend {
    prob: FrozenDualProblem,
}

#[cfg(test)]
impl ReferenceBackend {
    pub(crate) fn new(prob: &FrozenDualProblem) -> Self {
        Self { prob: prob.clone() }
    }
}

#[cfg(test)]
impl SearchBackend for ReferenceBackend {
    fn name(&self) -> &'static str {
        "reference_f64"
    }

    fn caps(&self) -> ny_core::search::BackendCaps {
        ny_core::search::BackendCaps::Cpu {
            threads: 1,
            lanes: 1,
            precision: ny_core::search::SearchPrecision::F64,
        }
    }

    fn upload_fd(
        &self,
        problem: &FdSearchProblem,
        _lease: &DeviceLease,
    ) -> Result<FdHandle, Refusal> {
        problem.validate().map_err(Refusal::Invalid)?;
        Ok(FdHandle::new("reference_f64", ()))
    }

    fn fd_level(
        &self,
        _handle: &FdHandle,
        chunk: &LevelChunk<'_>,
        out: &mut LevelOut,
    ) -> Result<ny_core::search::Untrusted<ny_core::search::LevelStats>, Refusal> {
        chunk.validate().map_err(Refusal::Invalid)?;
        let (d, count) = (chunk.depth, chunk.count);
        let prob = &self.prob;
        let charge_per_s = gamma(prob.n() + 3 * d + 32) * 1.001;
        let big_s = |nb: &NodeBound| ((nb.charge - TINY) / charge_per_s).max(0.0);
        out.prepare(count, d);
        let w = out.writer();
        let mut work = super::Work::new(prob.n(), d);
        let mut scratch = NodeScratch::default();
        for k in 0..count {
            for i in 0..d {
                work.sides[i] = if chunk.sides[k * d + i] == 0 {
                    Side::Off
                } else {
                    Side::On
                };
                work.l0[i] = widen(chunk.l0[k * d + i]);
                work.l1[i] = widen(chunk.l1[k * d + i]);
            }
            let node = super::eval_search_node(prob, &mut work);
            let at = k * 2 * d;
            if !node.best.is_finite() {
                // The f64 search's fail-closed branch (warm refused): it scores
                // no point — its clamped and far rows are the PREVIOUS node's,
                // stale — and the children inherit the parent's multipliers.
                // Propose exactly that: every value stays NaN (never claimed).
                for i in 0..d {
                    w.next[at + i] = narrow(work.next0[i]);
                    w.next[at + d + i] = narrow(work.next1[i]);
                }
                continue;
            }
            // The multipliers the lane will certify are the f32 roundings of
            // these; score each point AT them, as the lane will.
            let r32 = |v: &[f64]| -> Vec<f64> { v.iter().map(|&x| f64::from(x as f32)).collect() };
            let points = [
                (work.l0.clone(), work.l1.clone()),
                (r32(&work.c0), r32(&work.c1)),
                (vec![0.0; d], vec![0.0; d]),
                (r32(&work.f0), r32(&work.f1)),
            ];
            for (p, (a, b)) in points.iter().enumerate() {
                let nb = prob.eval_node(&work.sides, a, b, &mut scratch, None);
                if nb.value.is_finite() {
                    w.value[k * FD_POINTS + p] = nb.value as f32;
                    w.scale[k * FD_POINTS + p] = big_s(&nb) as f32;
                }
            }
            for i in 0..d {
                w.clamped[at + i] = work.c0[i] as f32;
                w.clamped[at + d + i] = work.c1[i] as f32;
                w.far[at + i] = work.f0[i] as f32;
                w.far[at + d + i] = work.f1[i] as f32;
                w.next[at + i] = narrow(work.next0[i]);
                w.next[at + d + i] = narrow(work.next1[i]);
            }
        }
        Ok(ny_core::search::Untrusted::new(
            ny_core::search::LevelStats {
                nodes: count as u64,
                ..ny_core::search::LevelStats::default()
            },
        ))
    }
}
