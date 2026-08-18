// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![cfg_attr(not(test), allow(dead_code))]

//! Best-first margin-row BaB (#twinwall).
//!
//! Port of the verified full-tree reference driver (validated against an
//! exact-rational falsifier harness during development):
//! pop the worst open domain; refresh its y-box from ITS OWN 2 x n_y-row
//! backward pass (LRU-cached per trunk-set); K=16 candidates (top-8 unstable
//! head neurons by exact single-gate variants + top-8 trunk by
//! |margin-row coef| x intercept from the domain's own seeded pass); score
//! ALL children exactly in ONE batched exception pass; push children with
//! bounds clamped to `max(child, parent)` (raw dips counted, never hidden);
//! close at a certified bound.
//!
//! SOUNDNESS: verdicts only ever come from `RoundMode::Outward` bounds
//! (rounded toward -inf), and production closure is STRICT (`bound > 0`,
//! vs the reference's `>= 0`) so `margin == 0` boundary points can never be
//! verified away. The tree is a per-point sign cover (every x belongs to a
//! child of any split), so closing every leaf closes the instance. This lane
//! can only return `Unsat` or `Unknown` — never `Sat`.

use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array2;
use ny_core::{NyError, Result};
use rayon::prelude::*;

use super::bounds::{
    compose_viay, head_gates, head_variant, margin_seed, per_class_direct, row_dots, trunk_variant,
    variant_state, HeadGates, MarginBatch, MarginSeed, RowDots, YBox,
};
use super::engine::{
    domain_gates, BackwardEngine, Collect, DomainGates, Exc, Exceptions, PassOut,
    RowDomainGateBlock, Seed,
};
use super::net::TwinNet;
use super::root::{RetainedRows, RootGates};
use super::rounding::{next_down, next_up, RoundMode, UNIT};

/// Trunk piece-fix: (layer, position in the layer's unstable list, direction).
pub type TrunkSplit = (usize, usize, i8);
/// Head fix: (head neuron, direction).
pub type HeadFix = (usize, i8);

/// Tuning knobs (defaults = the verified Part-B protocol).
#[derive(Clone)]
pub struct BabConfig {
    /// Head candidates per expansion.
    pub k_head: usize,
    /// Trunk candidates per expansion.
    pub k_trunk: usize,
    /// Expansion cap.
    pub max_expansions: usize,
    /// Wall-clock deadline.
    pub deadline: Option<Instant>,
    /// y-row LRU capacity (trunk-sets).
    pub lru_cap: usize,
    /// Parallel best-first frontier width: expand up to this many worst-open
    /// domains concurrently per step (rayon). `1` = the verified serial lane.
    /// Each domain's bound is computed bit-identically to serial; only the
    /// tree exploration order changes (still a sound per-point sign cover).
    pub frontier: usize,
    /// Default-off cross-domain candidate-score stacking canary.  Only the
    /// `score_candidates` backward passes of a parallel frontier batch are
    /// stacked; node evaluation, candidate selection, and heap merge retain
    /// their established per-domain paths and deterministic ordering.
    pub domain_stack: bool,
    /// Tier-0 (#epoch-bab): number of candidates that get the exact Tier-1
    /// pass after rank-1 variant ranking. `0` = legacy shortlist protocol
    /// (`k_head`/`k_trunk` candidates, no Tier-0). Requires `retained`.
    pub tier0_exact: usize,
    /// Tier-0 per-layer trunk universe cap (top by dynamic `|coef|·c`).
    pub tier0_universe: usize,
    /// Retained tableau rows for the Tier-0 trunk variant ranker
    /// (`RootGates::build_retaining`). Ranker-only; shared with the parallel
    /// frontier via `Arc`.
    pub retained: Option<Arc<RetainedRows>>,
    /// Tier-2 epochs (#epoch-bab): when a popped domain carries at least
    /// this many trunk splits, rebuild the trunk tableau with those splits
    /// BAKED (exact fixed lines -> every downstream gate re-derived from
    /// tighter boxes) and run the subtree as a nested lane under the epoch
    /// gates. `0` = off.
    pub epoch_depth: usize,
    /// Max epoch attempts per run that end `Unknown` (a closing epoch does
    /// not count against the budget; failed rebuilds are capped so the lane
    /// cannot thrash on expensive tableau builds).
    pub epoch_max_attempts: usize,
    /// Measured root tableau build seconds (set by the lane entry); an epoch
    /// is only attempted when the remaining deadline exceeds a multiple of
    /// this (the rebuild costs about one root build).
    pub root_build_secs: f64,
    /// Retention policy for epoch rebuilds (Tier-0 under the epoch).
    pub retain_cfg: Option<super::root::RetainCfg>,
    /// Nested-run plumbing: head sign-fixes inherited from the outer domain
    /// (semantic constraints on head pre-activations; gate-independent).
    pub initial_heads: Vec<HeadFix>,
    /// Nested-run plumbing: inherited certified y-box for the root entry
    /// (intersected on top of the epoch's own pack box at evaluation).
    pub initial_ybox: Option<(Vec<f64>, Vec<f64>)>,
}

impl Default for BabConfig {
    fn default() -> Self {
        Self {
            k_head: 8,
            k_trunk: 8,
            max_expansions: 20_000,
            deadline: None,
            lru_cap: 6,
            frontier: 1,
            domain_stack: false,
            tier0_exact: 0,
            tier0_universe: 64,
            retained: None,
            epoch_depth: 0,
            epoch_max_attempts: 2,
            root_build_secs: 0.0,
            retain_cfg: None,
            initial_heads: Vec::new(),
            initial_ybox: None,
        }
    }
}

/// One independently certified adversarial class in the classwise schedule.
///
/// These records are evidence/provenance only.  A classwise run returns
/// [`MarginRowOutcome::Unsat`] only when every record is `verified`; the first
/// non-verdict stops the schedule and returns `Unknown`.
#[derive(Debug, Clone)]
pub struct ClassBabStats {
    /// Adversarial output class `j` in the margin `Y_t - Y_j`.
    pub class: usize,
    /// Certified root lower bound used to order this class.
    pub root_bound: f64,
    /// Whether this class was certified over the complete input box.
    pub verified: bool,
    /// Expansions performed by this class tree.
    pub expansions: usize,
    /// Domains created by this class tree (including its root).
    pub domains_created: usize,
    /// Domains closed by this class tree.
    pub closed: usize,
    /// Maximum depth reached by this class tree.
    pub max_depth: usize,
    /// Raw child-below-parent dips observed by this class tree.
    pub mono_raw_dips: usize,
    /// Worst raw dip magnitude observed by this class tree.
    pub mono_worst: f64,
    /// Class stop reason (`verified_*`, budget, deadline, or fail-closed error).
    pub stop: String,
    /// Seconds spent in this class tree.
    pub elapsed_secs: f64,
    /// Tier-2 epochs attempted within this class tree.
    pub epochs_attempted: usize,
    /// Tier-2 epochs that closed a subtree within this class tree.
    pub epochs_closed: usize,
    /// This class tree's exact Kraft-ledger result when it verified.
    pub ledger_ok: Option<bool>,
}

/// Run statistics (evidence numbers, not adjectives).
#[derive(Debug, Clone)]
pub struct BabStats {
    /// Root direct bound over the tree classes (0.0 when closed at root).
    pub root_bound: f64,
    /// Classes that needed the tree.
    pub tree_classes: Vec<usize>,
    /// Classes closed at the root (of the adv list).
    pub root_closed_classes: usize,
    /// Expansions performed.
    pub expansions: usize,
    /// Domains created (incl. root).
    pub domains_created: usize,
    /// Domains closed.
    pub closed: usize,
    /// Max depth reached.
    pub max_depth: usize,
    /// Raw child-below-parent dips observed (clamped, logged).
    pub mono_raw_dips: usize,
    /// Worst raw dip magnitude.
    pub mono_worst: f64,
    /// Stop reason.
    pub stop: String,
    /// Seconds in the tree loop.
    pub elapsed_secs: f64,
    /// Per-class evidence for the opt-in classwise schedule. Empty for the
    /// unchanged joint schedule and for instances closed entirely at root.
    pub class_runs: Vec<ClassBabStats>,
    /// Tier-2 epochs attempted (#epoch-bab).
    pub epochs_attempted: usize,
    /// Tier-2 epochs that closed their subtree.
    pub epochs_closed: usize,
    /// Kraft-ledger verdict on Unsat (#epoch-bab Phase E): `Some(true)` when
    /// the closed leaves' prefix-free depths satisfy the exact-integer Kraft
    /// equality `sum 2^(D-d_i) == 2^D` (leaves partition the root domain)
    /// AND every nested epoch subtree's ledger verified too. Bookkeeping
    /// only — never moves a verdict; `None` when the lane did not verify.
    pub ledger_ok: Option<bool>,
}

/// Prefix-free leaf ledger (#epoch-bab Phase E). Records each closed leaf's
/// split depth; a complete binary split tree (every expansion produces
/// exactly two children) has `sum 2^(-d_i) == 1` over its leaves, checked in
/// exact integers below.
#[derive(Default)]
struct Ledger {
    depths: Vec<u32>,
    overflow: bool,
}

impl Ledger {
    fn leaf(&mut self, depth: usize) {
        if depth > 120 {
            self.overflow = true;
        } else {
            #[allow(clippy::cast_possible_truncation)]
            self.depths.push(depth as u32);
        }
    }

    /// Exact-integer Kraft equality over the recorded leaves.
    fn kraft_ok(&self) -> bool {
        if self.overflow || self.depths.is_empty() {
            return false;
        }
        let dmax = *self.depths.iter().max().expect("non-empty");
        let target: u128 = 1u128 << dmax;
        let mut sum: u128 = 0;
        for &d in &self.depths {
            let Some(s) = sum.checked_add(1u128 << (dmax - d)) else {
                return false;
            };
            sum = s;
        }
        sum == target
    }
}

/// Outcome of the lane. NEVER `Sat`: fail-closed to `Unknown`.
pub enum MarginRowOutcome {
    /// Every class certified: margin > 0 on the whole box.
    Unsat(BabStats),
    /// Not decided (budget, candidates exhausted, or fail-closed error).
    Unknown {
        /// Why.
        reason: String,
        /// Stats if the tree ran.
        stats: Option<BabStats>,
    },
}

struct YPack {
    al: PassOut,
    au: PassOut,
    ly0: Vec<f64>,
    uy0: Vec<f64>,
    al_dots: RowDots,
    au_dots: RowDots,
}

struct Lru {
    cap: usize,
    entries: Vec<(Vec<TrunkSplit>, Arc<YPack>)>,
}

impl Lru {
    fn get(&mut self, key: &[TrunkSplit]) -> Option<Arc<YPack>> {
        let pos = self.entries.iter().position(|(k, _)| k == key)?;
        let entry = self.entries.remove(pos);
        let pack = entry.1.clone();
        self.entries.push(entry);
        Some(pack)
    }

    fn put(&mut self, key: Vec<TrunkSplit>, pack: Arc<YPack>) {
        self.entries.retain(|(k, _)| k != &key);
        self.entries.push((key, pack));
        while self.entries.len() > self.cap {
            self.entries.remove(0);
        }
    }
}

struct DomainEntry {
    bound: f64,
    seq: u64,
    trunk: Vec<TrunkSplit>,
    heads: Vec<HeadFix>,
    ly: Vec<f64>,
    uy: Vec<f64>,
}

impl PartialEq for DomainEntry {
    fn eq(&self, other: &Self) -> bool {
        self.bound.total_cmp(&other.bound).is_eq() && self.seq == other.seq
    }
}
impl Eq for DomainEntry {}
impl PartialOrd for DomainEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for DomainEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap; invert so the WORST (lowest) bound pops
        // first, seq ascending as tiebreak (Python heapq parity).
        other
            .bound
            .total_cmp(&self.bound)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

struct NodeState {
    ybox: YBox,
    gates: HeadGates,
    ms: MarginSeed,
    coll: std::collections::BTreeMap<usize, Vec<f64>>,
    /// Tier-0 capture (#epoch-bab): layer -> `(n_retained, nf)` incoming
    /// coefficients at retained neurons from the direct pass.
    coll_rows: std::collections::BTreeMap<usize, Array2<f64>>,
    /// The domain's direct margin-seeded pass (rows/bias reused by the
    /// Tier-0 trunk variant ranker).
    pass: PassOut,
    dom: DomainGates,
    pack: Arc<YPack>,
}

/// A prospective child domain (heap `seq` assigned by the serial merge step).
struct ChildProto {
    bound: f64,
    trunk: Vec<TrunkSplit>,
    heads: Vec<HeadFix>,
    ly: Vec<f64>,
    uy: Vec<f64>,
}

/// Result of evaluating+expanding ONE popped domain in the parallel frontier.
/// Carries only data (no `&mut` state), so it is produced by a rayon worker
/// and applied deterministically by the serial merge, keeping every stats
/// delta and heap push bit-identical to what the serial lane would do for the
/// same domain.
enum ExpandStep {
    /// `eval_with_pack` returned `None` (provably empty box): closed.
    Infeasible,
    /// The refreshed bound already closes the domain.
    ClosedAtEval,
    /// No split candidates remain: the instance cannot close here (Unknown).
    NoCandidates,
    /// A verdict-bearing bound was non-finite: fail-closed to Unknown.
    NonFiniteBound,
    /// Expanded into children.
    Expanded {
        /// Raw child-below-parent dips observed.
        dips: usize,
        /// Worst (most negative) `raw - b` dip, or `0.0` if none.
        worst_dip: f64,
        /// Children to push (non-closed).
        pushes: Vec<ChildProto>,
        /// Children closed immediately (incl. head-infeasible slots).
        closed_children: usize,
        /// Child slots that incremented `domains_created`.
        created_children: usize,
    },
}

/// Gate-on two-phase expansion: domain evaluation and candidate selection stay
/// independent; only `Ready` records contribute columns to the shared score
/// pass. `Done` carries the unchanged terminal result for its batch position.
enum PreparedExpand {
    Done(ExpandStep),
    // Boxed: ReadyExpand (NodeState + candidate vecs) is ~13x the Done variant;
    // keeping it inline would size every batch slot at the large variant.
    Ready(Box<ReadyExpand>),
}

struct ReadyExpand {
    b: f64,
    st: NodeState,
    trunk_cands: Vec<(usize, usize)>,
    head_cands: Vec<usize>,
}

/// One domain's local score-candidate matrix and scalar side lanes.  Rows in
/// `exc` are local until `BackwardEngine::run_domain_stacked` assigns the
/// domain's validated contiguous range.
struct CandidateColumns {
    seed: Seed,
    cst: Vec<f64>,
    cst_err: Vec<f64>,
    m1: Vec<f64>,
    exc: Exceptions,
    n_candidates: usize,
}

/// The driver.
pub struct MarginRowBab<'a> {
    eng: BackwardEngine<'a>,
    mb: MarginBatch,
    cfg: BabConfig,
    lru: Lru,
    stats: BabStats,
    /// Failed (Unknown-ending) Tier-2 epoch attempts so far (#epoch-bab).
    epoch_failures: usize,
    /// Closed-leaf ledger (#epoch-bab Phase E).
    ledger: Ledger,
    /// AND of every nested epoch subtree's own ledger verdict.
    nested_ledgers_ok: bool,
}

/// Root evaluation shared by the driver and callers: via-y bounds for all
/// classes, then the direct pass for the via-y-failing set.
pub struct RootEval {
    /// Per-adv-class certified bound `max(m1, m2v, direct-if-computed)`.
    pub dj: Vec<f64>,
    /// Classes (values) still failing after the direct pass.
    pub tree_classes: Vec<usize>,
    /// The root y-pack (moved into the driver).
    pack: Arc<YPack>,
}

fn require_finite(context: &str, values: &[f64]) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(NyError::NumericalInstability(format!(
            "margin_row: non-finite {context}"
        )))
    }
}

fn closed(mode: RoundMode, b: f64) -> bool {
    // A non-finite arithmetic result is not a proof. In particular, `+inf`
    // must not satisfy the strict-positive closure predicate.
    if !b.is_finite() {
        return false;
    }
    // Production (outward) closure is STRICT: margin==0 admits a SAT point.
    if mode.outward() {
        b > 0.0
    } else {
        b >= 0.0
    }
}

fn build_pack(eng: &BackwardEngine<'_>, dom: Option<&DomainGates>) -> Result<YPack> {
    // The identity-seeded y-row refresh is the seam's cleanest admission: the
    // seed is f32-exact and carries no certified error, so no `y_abs` is
    // needed, and one device walk publishes BOTH lanes. Dark + fail-closed:
    // with `NY_MARGIN_ROW_GPU` unset this is the established `y_rows` call.
    let (al, au) = eng.y_rows_seamed(dom, &super::gpu_seam::SeamCtx::default())?;
    Ok(pack_from_rows(eng, al, au))
}

/// The y-pack's CPU tail: everything `build_pack` derives from the two
/// identity-seeded passes.
///
/// Split out so the DOMAIN-BATCHED prefill (#margin-row-gpu-batch) can build a
/// pack from rows a wide GPU call produced and get a BIT-IDENTICAL `YPack` for
/// identical `(al, au)` — the batched lane changes WHERE the rows come from and
/// nothing else about the pack.
fn pack_from_rows(eng: &BackwardEngine<'_>, al: PassOut, au: PassOut) -> YPack {
    let ybox = YBox::from_rows(eng, &al, &au);
    let al_dots = row_dots(eng.root, &al);
    let au_dots = row_dots(eng.root, &au);
    YPack {
        ly0: ybox.ly,
        uy0: ybox.uy,
        al_dots,
        au_dots,
        al,
        au,
    }
}

/// Root pass: mirrors `DirectBab.__init__` (via-y for all classes, direct for
/// the via-y-failing subset).
pub fn root_eval(
    eng: &BackwardEngine<'_>,
    net: &TwinNet,
    t: usize,
    adv: &[usize],
) -> Result<RootEval> {
    root_eval_impl(eng, net, t, adv)
}

fn root_eval_impl(
    eng: &BackwardEngine<'_>,
    net: &TwinNet,
    t: usize,
    adv: &[usize],
) -> Result<RootEval> {
    let mode = eng.root.mode;
    let pack = Arc::new(build_pack(eng, None)?);
    let ybox = YBox {
        ly: pack.ly0.clone(),
        uy: pack.uy0.clone(),
    };
    // #twin-head-probe (NY_MARGIN_ROW_HEAD_PROBE=1, print-only, verdict-neutral,
    // byte-identical when unset). Settles ONE question before anyone builds the
    // certified-head routing described in docs/MARGIN_ROW_ROOT_JOINT_COUPLING.md:
    // is this lane's OWN head box actually looser than the graph lane's tightened
    // one? The margin-row `YBox` is the same semantic tensor the CIFAR100 parity
    // oracle measures — `Flatten_55 -> Gemm_56 -> y -> Relu_57` — so these numbers
    // are directly comparable to that doc's `327.9822 -> 160.4943` and its
    // `unstable 44 -> 22`, and to the graph side's own
    // `[root-crown-interm-tighten] ... intersected_width=` line.
    //
    // If this lane already reports ~164, routing the graph box here buys NOTHING
    // and the patch must not be written. That is the whole point of measuring
    // first: `run_margin_row_lane_with_head` currently does `drop(external_head)`
    // (margin_row/mod.rs:135) as a deliberate quarantine, and lifting a quarantine
    // to gain zero is pure risk on a lane that decides verdicts.
    if std::env::var_os("NY_MARGIN_ROW_HEAD_PROBE").is_some() {
        let width: f64 = ybox.ly.iter().zip(ybox.uy.iter()).map(|(l, u)| u - l).sum();
        let unstable = ybox
            .ly
            .iter()
            .zip(ybox.uy.iter())
            .filter(|(l, u)| **l < 0.0 && **u > 0.0)
            .count();
        eprintln!(
            "[twin-head] n_y={} width_sum={:.4} mean_width={:.5} unstable={}",
            ybox.ly.len(),
            width,
            width / (ybox.ly.len().max(1) as f64),
            unstable
        );
    }
    let mb_all = MarginBatch::new(net, t, adv)?;
    let gates = head_gates(&ybox, mode);
    let ms_all = margin_seed(&mb_all, &gates, &ybox, mode);
    let m2v_all = compose_viay(
        eng,
        &mb_all,
        &gates,
        &pack.al,
        &pack.au,
        &pack.al_dots,
        &pack.au_dots,
        mode,
    );
    let mut dj = Vec::with_capacity(adv.len());
    for (&m1, &m2v) in ms_all.m1.iter().zip(&m2v_all) {
        require_finite("root component", &[m1, m2v])?;
        dj.push(m1.max(m2v));
    }
    let fail: Vec<usize> = (0..adv.len()).filter(|&k| !closed(mode, dj[k])).collect();
    if !fail.is_empty() {
        let fail_classes: Vec<usize> = fail.iter().map(|&k| adv[k]).collect();
        let mbf = MarginBatch::new(net, t, &fail_classes)?;
        let ms_f = margin_seed(&mbf, &gates, &ybox, mode);
        // The root direct pass. Its seed DOES carry a certified error and is
        // not f32-exact, so the seam needs the y-box magnitudes to concretize
        // both discrepancies into the bias-error lane; without them it refuses
        // and this is the established CPU call.
        let y_abs: Vec<f64> = ybox
            .ly
            .iter()
            .zip(&ybox.uy)
            .map(|(l, u)| l.abs().max(u.abs()))
            .collect();
        let seam = super::gpu_seam::SeamCtx {
            y_abs: Some(&y_abs),
            deadline: None,
        };
        let pass = eng.run_seamed(&ms_f.seed, None, super::engine::LaneDir::Lower, &seam)?;
        let direct = per_class_direct(eng, &pass, &ms_f, 0..fail.len());
        for (fi, &k) in fail.iter().enumerate() {
            require_finite("direct root bound", &[direct[fi]])?;
            dj[k] = dj[k].max(direct[fi]);
        }
    }
    require_finite("combined root bound", &dj)?;
    let tree_classes: Vec<usize> = (0..adv.len())
        .filter(|&k| !closed(mode, dj[k]))
        .map(|k| adv[k])
        .collect();
    Ok(RootEval {
        dj,
        tree_classes,
        pack,
    })
}

/// Build the deterministic classwise work list from root bounds aligned with
/// `adv`. Only a finite, strict-positive bound omits a class. Remaining
/// classes are hardest-root-first; class id is a data-independent tie-break.
pub(crate) fn classwise_schedule(t: usize, adv: &[usize], dj: &[f64]) -> Result<Vec<(usize, f64)>> {
    if adv.is_empty() || adv.len() != dj.len() {
        return Err(NyError::InvalidSpec(
            "margin_row classwise: invalid adversarial/root-bound cardinality".into(),
        ));
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut pending = Vec::new();
    for (&class, &bound) in adv.iter().zip(dj) {
        if class == t || !seen.insert(class) {
            return Err(NyError::InvalidSpec(
                "margin_row classwise: target/duplicate adversarial class".into(),
            ));
        }
        if !bound.is_finite() {
            return Err(NyError::NumericalInstability(
                "margin_row classwise: non-finite root bound".into(),
            ));
        }
        if bound <= 0.0 {
            pending.push((class, bound));
        }
    }
    pending.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    Ok(pending)
}

fn fresh_class_lru(cap: usize, root_pack: &Arc<YPack>) -> Lru {
    // Each class owns a fresh cache: an LRU key contains only trunk splits and
    // has no net/root/input identity. The immutable full-box root pack alone
    // is shared by Arc; no class partition, MarginBatch, candidate, or bound is.
    let mut lru = Lru {
        cap: cap.max(1),
        entries: Vec::new(),
    };
    lru.put(Vec::new(), root_pack.clone());
    lru
}

#[cfg(test)]
pub(crate) fn classwise_root_cache_isolated_for_test(re: &RootEval) -> bool {
    let mut a = fresh_class_lru(4, &re.pack);
    let mut b = fresh_class_lru(4, &re.pack);
    let Some(pa) = a.get(&[]) else { return false };
    let Some(pb) = b.get(&[]) else { return false };
    let same_pack = Arc::ptr_eq(&pa, &pb) && Arc::ptr_eq(&pa, &re.pack);
    a.entries.clear();
    same_pack && b.entries.len() == 1
}

fn class_record(class: usize, verified: bool, stats: &BabStats, stop: String) -> ClassBabStats {
    ClassBabStats {
        class,
        root_bound: stats.root_bound,
        verified,
        expansions: stats.expansions,
        domains_created: stats.domains_created,
        closed: stats.closed,
        max_depth: stats.max_depth,
        mono_raw_dips: stats.mono_raw_dips,
        mono_worst: stats.mono_worst,
        stop,
        elapsed_secs: stats.elapsed_secs,
        epochs_attempted: stats.epochs_attempted,
        epochs_closed: stats.epochs_closed,
        ledger_ok: stats.ledger_ok,
    }
}

pub(crate) fn absorb_class_stats(aggregate: &mut BabStats, record: ClassBabStats) {
    aggregate.expansions += record.expansions;
    aggregate.domains_created += record.domains_created;
    aggregate.closed += record.closed;
    aggregate.max_depth = aggregate.max_depth.max(record.max_depth);
    aggregate.mono_raw_dips += record.mono_raw_dips;
    aggregate.mono_worst = aggregate.mono_worst.min(record.mono_worst);
    aggregate.epochs_attempted += record.epochs_attempted;
    aggregate.epochs_closed += record.epochs_closed;
    aggregate.ledger_ok = Some(
        aggregate.ledger_ok == Some(true) && record.verified && record.ledger_ok == Some(true),
    );
    aggregate.class_runs.push(record);
}

pub(crate) fn classwise_conjunction_complete(
    tree_classes: &[usize],
    class_runs: &[ClassBabStats],
) -> bool {
    tree_classes.len() == class_runs.len()
        && tree_classes
            .iter()
            .zip(class_runs)
            .all(|(class, run)| *class == run.class && run.verified)
}

/// #parallel-prebuild gate. Exact `"1"`, read once -- it selects a scheduling
/// strategy, never a bound, and cannot change during a run.
fn parallel_prebuild_enabled() -> bool {
    static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var("NY_MARGIN_ROW_PARALLEL_PREBUILD").is_ok_and(|v| v == "1"))
}

impl<'a> MarginRowBab<'a> {
    /// Run the full lane: root pass, then best-first tree for failing classes.
    /// Authority granted 2026-07-18 (see `margin_row::margin_row_bab_enabled`):
    /// both proof obligations discharged, so production builds run the certified
    /// algorithm. Still fail-closed to `Unknown` on any internal error — it only
    /// ever returns `Unsat` or `Unknown`, never `Sat`.
    ///
    /// (This entry also used to carry a SECOND, undocumented `cfg(not(test))`
    /// quarantine that made the lane a no-op in every non-test build
    /// regardless of the gate above. It is gone: there is exactly ONE verdict
    /// gate now, and `run_entry_is_not_a_second_quarantine` fails if another
    /// appears.)
    pub fn run(
        net: &'a TwinNet,
        root: &'a RootGates,
        t: usize,
        adv: &[usize],
        cfg: BabConfig,
    ) -> MarginRowOutcome {
        match Self::run_inner(net, root, t, adv, cfg) {
            Ok(out) => out,
            Err(e) => MarginRowOutcome::Unknown {
                reason: format!("fail-closed: {e}"),
                stats: None,
            },
        }
    }

    /// Opt-in classwise conjunction: certify every adversarial margin over the
    /// complete input box in an independent tree. TwinNet, RootGates, the joint
    /// root evaluation, and its immutable root y-pack are built once; all
    /// class-dependent search state is fresh. The first non-verdict stops the
    /// schedule, so this can only strengthen `Unknown` to `Unsat`, never emit
    /// `Sat` or certify a subset of the robustness disjunction.
    pub fn run_classwise(
        net: &'a TwinNet,
        root: &'a RootGates,
        t: usize,
        adv: &[usize],
        cfg: BabConfig,
    ) -> MarginRowOutcome {
        if !root.mode.outward() {
            return MarginRowOutcome::Unknown {
                reason: "classwise requires certified outward rounding".into(),
                stats: None,
            };
        }
        match Self::run_classwise_inner(net, root, t, adv, cfg) {
            Ok(out) => out,
            Err(e) => MarginRowOutcome::Unknown {
                reason: format!("classwise fail-closed: {e}"),
                stats: None,
            },
        }
    }

    fn run_classwise_inner(
        net: &'a TwinNet,
        root: &'a RootGates,
        t: usize,
        adv: &[usize],
        cfg: BabConfig,
    ) -> Result<MarginRowOutcome> {
        let t0 = Instant::now();
        if cfg.deadline.is_some_and(|dl| Instant::now() > dl) {
            return Ok(MarginRowOutcome::Unknown {
                reason: "classwise wallclock before root evaluation".into(),
                stats: None,
            });
        }
        let root_eng = BackwardEngine::new(net, root);
        let re = {
            let _t = super::prof::Timer::start(super::prof::Phase::RootEval);
            root_eval_impl(&root_eng, net, t, adv)?
        };
        if cfg.deadline.is_some_and(|dl| Instant::now() > dl) {
            return Ok(MarginRowOutcome::Unknown {
                reason: "classwise wallclock after root evaluation".into(),
                stats: None,
            });
        }
        let schedule = classwise_schedule(t, adv, &re.dj)?;
        let root_closed_classes = adv.len() - schedule.len();
        let tree_classes: Vec<usize> = schedule.iter().map(|(class, _)| *class).collect();
        let root_bound = schedule.first().map_or(0.0, |(_, bound)| *bound);
        let mut aggregate = BabStats {
            root_bound,
            tree_classes,
            root_closed_classes,
            expansions: 0,
            domains_created: usize::from(schedule.is_empty()),
            closed: 0,
            max_depth: 0,
            mono_raw_dips: 0,
            mono_worst: 0.0,
            stop: "classwise_closed_at_root".into(),
            elapsed_secs: 0.0,
            class_runs: Vec::new(),
            epochs_attempted: 0,
            epochs_closed: 0,
            ledger_ok: Some(true),
        };
        if schedule.is_empty() {
            aggregate.elapsed_secs = t0.elapsed().as_secs_f64();
            if cfg.deadline.is_some_and(|dl| Instant::now() > dl) {
                aggregate.stop = "classwise_wallclock_before_root_verdict".into();
                return Ok(MarginRowOutcome::Unknown {
                    reason: "classwise wallclock before root-only verdict".into(),
                    stats: Some(aggregate),
                });
            }
            return Ok(MarginRowOutcome::Unsat(aggregate));
        }

        let mut remaining_expansions = cfg.max_expansions;
        for (class, class_root_bound) in schedule {
            if cfg.deadline.is_some_and(|dl| Instant::now() > dl) {
                aggregate.stop = format!("classwise_class_{class}_wallclock");
                aggregate.elapsed_secs = t0.elapsed().as_secs_f64();
                let empty = BabStats {
                    root_bound: class_root_bound,
                    tree_classes: vec![class],
                    root_closed_classes: 0,
                    expansions: 0,
                    domains_created: 0,
                    closed: 0,
                    max_depth: 0,
                    mono_raw_dips: 0,
                    mono_worst: 0.0,
                    stop: "wallclock_before_class".into(),
                    elapsed_secs: 0.0,
                    class_runs: Vec::new(),
                    epochs_attempted: 0,
                    epochs_closed: 0,
                    ledger_ok: None,
                };
                absorb_class_stats(
                    &mut aggregate,
                    class_record(class, false, &empty, empty.stop.clone()),
                );
                return Ok(MarginRowOutcome::Unknown {
                    reason: format!("classwise class {class}: wallclock"),
                    stats: Some(aggregate),
                });
            }
            if remaining_expansions == 0 {
                aggregate.stop = format!("classwise_class_{class}_max_expansions");
                aggregate.elapsed_secs = t0.elapsed().as_secs_f64();
                let empty = BabStats {
                    root_bound: class_root_bound,
                    tree_classes: vec![class],
                    root_closed_classes: 0,
                    expansions: 0,
                    domains_created: 0,
                    closed: 0,
                    max_depth: 0,
                    mono_raw_dips: 0,
                    mono_worst: 0.0,
                    stop: "global_max_expansions_before_class".into(),
                    elapsed_secs: 0.0,
                    class_runs: Vec::new(),
                    epochs_attempted: 0,
                    epochs_closed: 0,
                    ledger_ok: None,
                };
                absorb_class_stats(
                    &mut aggregate,
                    class_record(class, false, &empty, empty.stop.clone()),
                );
                return Ok(MarginRowOutcome::Unknown {
                    reason: format!("classwise class {class}: global max_expansions"),
                    stats: Some(aggregate),
                });
            }

            // Serial per-class trees make the one absolute/global expansion
            // budget exact. The existing parallel frontier may complete up to
            // frontier-1 extra expansions in its final batch, so it is not used
            // by this bounded experiment.
            let class_cfg = BabConfig {
                max_expansions: remaining_expansions,
                frontier: 1,
                ..cfg.clone()
            };
            let stats = BabStats {
                root_bound: class_root_bound,
                tree_classes: vec![class],
                root_closed_classes: 0,
                expansions: 0,
                domains_created: 1,
                closed: 0,
                max_depth: 0,
                mono_raw_dips: 0,
                mono_worst: 0.0,
                stop: "classwise_tree".into(),
                elapsed_secs: 0.0,
                class_runs: Vec::new(),
                epochs_attempted: 0,
                epochs_closed: 0,
                ledger_ok: None,
            };
            let class_lru = fresh_class_lru(class_cfg.lru_cap, &re.pack);
            let mut bab = Self {
                eng: BackwardEngine::new(net, root),
                mb: MarginBatch::new(net, t, &[class])?,
                cfg: class_cfg,
                lru: class_lru,
                stats,
                epoch_failures: 0,
                ledger: Ledger::default(),
                nested_ledgers_ok: true,
            };
            let class_t0 = Instant::now();
            let class_result = bab.tree_loop(class_root_bound, &re);
            let outcome = match class_result {
                Ok(outcome) => outcome,
                Err(e) => {
                    bab.stats.stop = format!("fail_closed: {e}");
                    bab.stats.elapsed_secs = class_t0.elapsed().as_secs_f64();
                    MarginRowOutcome::Unknown {
                        reason: format!("fail-closed: {e}"),
                        stats: Some(bab.stats.clone()),
                    }
                }
            };
            match outcome {
                MarginRowOutcome::Unsat(stats) => {
                    if stats.expansions > remaining_expansions {
                        return Err(NyError::NumericalInstability(
                            "margin_row classwise: global expansion budget overshoot".into(),
                        ));
                    }
                    remaining_expansions -= stats.expansions;
                    if cfg.deadline.is_some_and(|dl| Instant::now() > dl) {
                        let mut late = stats;
                        late.stop = "wallclock_after_class".into();
                        let stop = late.stop.clone();
                        absorb_class_stats(&mut aggregate, class_record(class, false, &late, stop));
                        aggregate.stop = format!("classwise_class_{class}_wallclock");
                        aggregate.elapsed_secs = t0.elapsed().as_secs_f64();
                        return Ok(MarginRowOutcome::Unknown {
                            reason: format!("classwise class {class}: wallclock"),
                            stats: Some(aggregate),
                        });
                    }
                    let stop = stats.stop.clone();
                    absorb_class_stats(&mut aggregate, class_record(class, true, &stats, stop));
                }
                MarginRowOutcome::Unknown { reason, stats } => {
                    let mut stats = stats.unwrap_or_else(|| bab.stats.clone());
                    if stats.stop == "classwise_tree" {
                        stats.stop = reason.clone();
                    }
                    if stats.expansions > remaining_expansions {
                        return Err(NyError::NumericalInstability(
                            "margin_row classwise: global expansion budget overshoot".into(),
                        ));
                    }
                    let stop = stats.stop.clone();
                    absorb_class_stats(&mut aggregate, class_record(class, false, &stats, stop));
                    aggregate.stop = format!("classwise_class_{class}_unknown");
                    aggregate.elapsed_secs = t0.elapsed().as_secs_f64();
                    return Ok(MarginRowOutcome::Unknown {
                        reason: format!("classwise class {class}: {reason}"),
                        stats: Some(aggregate),
                    });
                }
            }
        }
        if cfg.deadline.is_some_and(|dl| Instant::now() > dl) {
            aggregate.stop = "classwise_wallclock_before_verdict".into();
            aggregate.elapsed_secs = t0.elapsed().as_secs_f64();
            return Ok(MarginRowOutcome::Unknown {
                reason: "classwise wallclock before aggregate verdict".into(),
                stats: Some(aggregate),
            });
        }
        if !classwise_conjunction_complete(&aggregate.tree_classes, &aggregate.class_runs) {
            aggregate.stop = "classwise_incomplete_conjunction".into();
            aggregate.elapsed_secs = t0.elapsed().as_secs_f64();
            return Ok(MarginRowOutcome::Unknown {
                reason: "classwise incomplete conjunction".into(),
                stats: Some(aggregate),
            });
        }
        aggregate.stop = "classwise_verified_all".into();
        aggregate.elapsed_secs = t0.elapsed().as_secs_f64();
        Ok(MarginRowOutcome::Unsat(aggregate))
    }

    /// Full algorithm body. Production entry is [`Self::run`]; nested Tier-2
    /// epoch runs re-enter here directly (#epoch-bab).
    fn run_inner(
        net: &TwinNet,
        root: &RootGates,
        t: usize,
        adv: &[usize],
        cfg: BabConfig,
    ) -> Result<MarginRowOutcome> {
        let t0 = Instant::now();
        if cfg.deadline.is_some_and(|dl| Instant::now() > dl) {
            return Ok(MarginRowOutcome::Unknown {
                reason: "wallclock before root evaluation".into(),
                stats: None,
            });
        }
        let eng = BackwardEngine::new(net, root);
        let re = {
            let _t = super::prof::Timer::start(super::prof::Phase::RootEval);
            root_eval(&eng, net, t, adv)?
        };
        if cfg.deadline.is_some_and(|dl| Instant::now() > dl) {
            return Ok(MarginRowOutcome::Unknown {
                reason: "wallclock after root evaluation".into(),
                stats: None,
            });
        }
        let root_closed_classes = adv.len() - re.tree_classes.len();
        let mut stats = BabStats {
            root_bound: 0.0,
            tree_classes: re.tree_classes.clone(),
            root_closed_classes,
            expansions: 0,
            domains_created: 1,
            closed: 0,
            max_depth: 0,
            mono_raw_dips: 0,
            mono_worst: 0.0,
            stop: "closed_at_root".into(),
            elapsed_secs: 0.0,
            class_runs: Vec::new(),
            epochs_attempted: 0,
            epochs_closed: 0,
            ledger_ok: None,
        };
        if re.tree_classes.is_empty() {
            stats.elapsed_secs = t0.elapsed().as_secs_f64();
            // Single leaf at depth 0: Kraft trivially holds.
            stats.ledger_ok = Some(true);
            return Ok(MarginRowOutcome::Unsat(stats));
        }
        let mb = MarginBatch::new(net, t, &re.tree_classes)?;
        let mut bab = MarginRowBab {
            eng,
            mb,
            cfg,
            lru: Lru {
                cap: 6,
                entries: Vec::new(),
            },
            stats,
            epoch_failures: 0,
            ledger: Ledger::default(),
            nested_ledgers_ok: true,
        };
        bab.lru.cap = bab.cfg.lru_cap;
        bab.lru.put(Vec::new(), re.pack.clone());
        // Root bound over tree classes for the initial heap entry.
        let root_bound = re
            .dj
            .iter()
            .zip(adv)
            .filter(|(_, j)| re.tree_classes.contains(j))
            .map(|(b, _)| *b)
            .fold(f64::INFINITY, f64::min);
        bab.stats.root_bound = root_bound;
        if !root_bound.is_finite() {
            return Ok(MarginRowOutcome::Unknown {
                reason: "non-finite root bound".into(),
                stats: Some(bab.stats),
            });
        }
        let outcome = if bab.cfg.frontier > 1 {
            bab.tree_loop_parallel(root_bound, &re)?
        } else {
            bab.tree_loop(root_bound, &re)?
        };
        Ok(outcome)
    }

    /// Root heap entry: carries any nested-run inheritance (initial head
    /// fixes + certified y-box from the outer domain; #epoch-bab).
    fn root_entry(&self, root_bound: f64, re: &RootEval, seq: u64) -> DomainEntry {
        let (ly, uy) = match &self.cfg.initial_ybox {
            Some((ly, uy)) => (ly.clone(), uy.clone()),
            None => (re.pack.ly0.clone(), re.pack.uy0.clone()),
        };
        DomainEntry {
            bound: root_bound,
            seq,
            trunk: Vec::new(),
            heads: self.cfg.initial_heads.clone(),
            ly,
            uy,
        }
    }

    fn tree_loop(&mut self, root_bound: f64, re: &RootEval) -> Result<MarginRowOutcome> {
        let t0 = Instant::now();
        let mode = self.eng.root.mode;
        let mut heap: BinaryHeap<DomainEntry> = BinaryHeap::new();
        let mut seq: u64 = 0;
        heap.push(self.root_entry(root_bound, re, seq));
        seq += 1;
        let mut stop = "queue_empty".to_string();
        let trace = std::env::var("NY_MARGIN_ROW_TRACE").is_ok();
        while let Some(entry) = heap.pop() {
            if trace && self.stats.expansions.is_multiple_of(25) {
                eprintln!(
                    "[trace] exp {:>5} pop bound {:+.5} depth {} ({:.1}s)",
                    self.stats.expansions,
                    entry.bound,
                    entry.trunk.len() + entry.heads.len(),
                    t0.elapsed().as_secs_f64()
                );
            }
            if self.stats.expansions >= self.cfg.max_expansions {
                stop = "max_expansions".into();
                break;
            }
            if let Some(dl) = self.cfg.deadline {
                if Instant::now() > dl {
                    stop = "wallclock".into();
                    break;
                }
            }
            if !entry.bound.is_finite() {
                return Ok(MarginRowOutcome::Unknown {
                    reason: "non-finite queued domain bound".into(),
                    stats: Some(self.stats.clone()),
                });
            }
            if closed(mode, entry.bound) {
                self.stats.closed += 1 + heap.len();
                self.ledger.leaf(entry.trunk.len() + entry.heads.len());
                for e in heap.iter() {
                    self.ledger.leaf(e.trunk.len() + e.heads.len());
                }
                heap.clear();
                stop = "verified_bestfirst".into();
                break;
            }
            let depth = entry.trunk.len() + entry.heads.len();
            self.stats.max_depth = self.stats.max_depth.max(depth);
            // Tier-2 epoch attempt (#epoch-bab): rebuild-with-baked-splits +
            // nested subtree run. Closing the subtree closes this domain.
            if self.try_epoch(&entry) {
                self.stats.closed += 1;
                self.ledger.leaf(depth);
                continue;
            }
            if self.stats.expansions >= self.cfg.max_expansions {
                stop = "max_expansions_after_epoch_attempt".into();
                break;
            }
            if self.cfg.deadline.is_some_and(|dl| Instant::now() > dl) {
                stop = "wallclock_after_epoch_attempt".into();
                break;
            }
            let node = self.eval_node(&entry)?;
            let Some((b_eval, st)) = node else {
                // Infeasible domain: provably empty, closed.
                self.stats.closed += 1;
                self.ledger.leaf(depth);
                continue;
            };
            if !b_eval.is_finite() {
                return Ok(MarginRowOutcome::Unknown {
                    reason: "non-finite evaluated domain bound".into(),
                    stats: Some(self.stats.clone()),
                });
            }
            let b = b_eval.max(entry.bound);
            if closed(mode, b) {
                self.stats.closed += 1;
                self.ledger.leaf(depth);
                continue;
            }
            // ---------- candidates ----------
            let (trunk_cands, head_cands) = self.select_candidates(&st, &entry);
            if head_cands.is_empty() && trunk_cands.is_empty() {
                stop = "no_candidates".into();
                break;
            }
            let ch = {
                let _t = super::prof::Timer::start(super::prof::Phase::ScoreCands);
                self.score_candidates(&st, &trunk_cands, &head_cands)?
            };
            if ch
                .iter()
                .any(|(left, right)| !(left.is_finite() && right.is_finite()))
            {
                return Ok(MarginRowOutcome::Unknown {
                    reason: "non-finite child score".into(),
                    stats: Some(self.stats.clone()),
                });
            }
            // Pick: max by (min child, sum child); later index wins ties
            // (np.lexsort parity).
            let mut pick = 0usize;
            let mut best = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            for (i, pair) in ch.iter().enumerate() {
                let key = (pair.0.min(pair.1), pair.0 + pair.1);
                if key.0 > best.0 || (key.0 == best.0 && key.1 >= best.1) {
                    best = key;
                    pick = i;
                }
            }
            let n_t = trunk_cands.len();
            let chb = ch[pick];
            // ---------- push children ----------
            for (d_i, dr) in [(0usize, 1i8), (1usize, -1i8)] {
                let raw = if d_i == 0 { chb.0 } else { chb.1 };
                if !raw.is_finite() {
                    return Ok(MarginRowOutcome::Unknown {
                        reason: "non-finite child bound".into(),
                        stats: Some(self.stats.clone()),
                    });
                }
                if raw < b - 1e-9 {
                    self.stats.mono_raw_dips += 1;
                    self.stats.mono_worst = self.stats.mono_worst.min(raw - b);
                }
                let cb = raw.max(b);
                let (trunk_c, heads_c, ly_c, uy_c) = if pick < n_t {
                    super::prof::bump(super::prof::Counter::TrunkSplit, 1);
                    let (li, pos) = trunk_cands[pick];
                    let mut tc = entry.trunk.clone();
                    tc.push((li, pos, dr));
                    (
                        tc,
                        entry.heads.clone(),
                        st.ybox.ly.clone(),
                        st.ybox.uy.clone(),
                    )
                } else {
                    super::prof::bump(super::prof::Counter::HeadSplit, 1);
                    let i = head_cands[pick - n_t];
                    let mut ly = st.ybox.ly.clone();
                    let mut uy = st.ybox.uy.clone();
                    if dr > 0 {
                        ly[i] = ly[i].max(0.0);
                    } else {
                        uy[i] = uy[i].min(0.0);
                    }
                    if ly[i] > uy[i] + 1e-12 {
                        self.stats.closed += 1;
                        self.stats.domains_created += 1;
                        self.ledger.leaf(depth + 1);
                        continue;
                    }
                    let mut hc = entry.heads.clone();
                    hc.push((i, dr));
                    (entry.trunk.clone(), hc, ly, uy)
                };
                self.stats.domains_created += 1;
                if closed(mode, cb) {
                    self.stats.closed += 1;
                    self.ledger.leaf(depth + 1);
                } else {
                    heap.push(DomainEntry {
                        bound: cb,
                        seq,
                        trunk: trunk_c,
                        heads: heads_c,
                        ly: ly_c,
                        uy: uy_c,
                    });
                    seq += 1;
                }
            }
            self.stats.expansions += 1;
        }
        self.stats.stop = stop.clone();
        self.stats.elapsed_secs = t0.elapsed().as_secs_f64();
        let mut verified = matches!(stop.as_str(), "verified_bestfirst" | "queue_empty");
        if verified && self.cfg.deadline.is_some_and(|dl| Instant::now() > dl) {
            stop = "wallclock_before_verdict".into();
            self.stats.stop.clone_from(&stop);
            verified = false;
        }
        if verified {
            self.stats.ledger_ok = Some(self.ledger.kraft_ok() && self.nested_ledgers_ok);
            Ok(MarginRowOutcome::Unsat(self.stats.clone()))
        } else {
            Ok(MarginRowOutcome::Unknown {
                reason: stop,
                stats: Some(self.stats.clone()),
            })
        }
    }

    /// Parallel best-first frontier (`NY_MARGIN_ROW_PARALLEL`/`cfg.frontier`):
    /// pop up to `frontier` worst-open domains, evaluate+expand them
    /// CONCURRENTLY (rayon), then merge results serially and deterministically.
    ///
    /// SOUNDNESS / MOAT: each domain is evaluated by [`Self::expand_one`], a
    /// pure function of `(pack, entry)` — identical math and reduction order to
    /// the serial lane — so every bound and every `closed()` decision is
    /// BIT-IDENTICAL to serial. Only the exploration ORDER changes (a batch of
    /// N is expanded between re-sorts instead of one). The tree is still a
    /// per-point sign cover, so the instance closes iff every leaf closes; a
    /// SAT point survives in some leaf under both orders. The lane still only
    /// ever returns `Unsat` or `Unknown`.
    fn tree_loop_parallel(&mut self, root_bound: f64, re: &RootEval) -> Result<MarginRowOutcome> {
        let t0 = Instant::now();
        let mode = self.eng.root.mode;
        let width = self.cfg.frontier.max(1);
        let mut heap: BinaryHeap<DomainEntry> = BinaryHeap::new();
        let mut seq: u64 = 0;
        heap.push(self.root_entry(root_bound, re, seq));
        seq += 1;
        let mut stop = "queue_empty".to_string();
        let trace = std::env::var("NY_MARGIN_ROW_TRACE").is_ok();
        'outer: while !heap.is_empty() {
            if trace {
                if let Some(worst) = heap.peek() {
                    eprintln!(
                        "[trace] exp {:>5} frontier worst {:+.5} depth {} open {} ({:.1}s)",
                        self.stats.expansions,
                        worst.bound,
                        worst.trunk.len() + worst.heads.len(),
                        heap.len(),
                        t0.elapsed().as_secs_f64()
                    );
                }
            }
            if self.stats.expansions >= self.cfg.max_expansions {
                stop = "max_expansions".into();
                break;
            }
            if let Some(dl) = self.cfg.deadline {
                if Instant::now() > dl {
                    stop = "wallclock".into();
                    break;
                }
            }
            // Pop up to `width` worst-open domains into the batch. Open entries
            // are never closed (the push rule below never enqueues a closed
            // bound); the closed guard is defensive and, if it ever fires,
            // simply counts the domain closed (queue_empty then verifies).
            let mut batch: Vec<DomainEntry> = Vec::with_capacity(width);
            while batch.len() < width {
                let Some(entry) = heap.pop() else { break };
                if !entry.bound.is_finite() {
                    return Ok(MarginRowOutcome::Unknown {
                        reason: "non-finite queued domain bound".into(),
                        stats: Some(self.stats.clone()),
                    });
                }
                if closed(mode, entry.bound) {
                    self.stats.closed += 1;
                    self.ledger.leaf(entry.trunk.len() + entry.heads.len());
                    continue;
                }
                let depth = entry.trunk.len() + entry.heads.len();
                self.stats.max_depth = self.stats.max_depth.max(depth);
                // Tier-2 epoch attempt (#epoch-bab): handled in this serial
                // fill stage (the rebuild is internally rayon-parallel and
                // needs `&mut self` for the attempt accounting).
                if self.try_epoch(&entry) {
                    self.stats.closed += 1;
                    self.ledger.leaf(depth);
                    continue;
                }
                if self.stats.expansions >= self.cfg.max_expansions {
                    stop = "max_expansions_after_epoch_attempt".into();
                    break 'outer;
                }
                if self.cfg.deadline.is_some_and(|dl| Instant::now() > dl) {
                    stop = "wallclock_after_epoch_attempt".into();
                    break 'outer;
                }
                batch.push(entry);
            }
            if batch.is_empty() {
                break;
            }
            // Prebuild each domain's y-pack SERIALLY: this reads/populates the
            // shared LRU (no cross-thread cache sync) and dedups identical
            // trunk-sets within the batch. The builds are internally rayon-
            // parallel; the NEW parallelism is expanding the batch concurrently.
            // A deadline check between (potentially expensive) builds bounds the
            // prebuild wall overshoot to one y-refresh; un-built entries are
            // returned to the heap so no work is lost.
            // #parallel-prebuild (DARK, NY_MARGIN_ROW_PARALLEL_PREBUILD=1):
            // fill the batch's LRU misses CONCURRENTLY before the serial loop
            // below, which then hits on every entry.
            //
            // Why this is the batch-level barrier: the loop below builds up to
            // `frontier` (= rayon::current_num_threads()) y-packs one at a time
            // on the driver thread, each ~0.63 s measured
            // (docs/EPOCH_BAB_DESIGN.md:189, 77 s over 123 expansions), against
            // roughly 0.5-2 s of actual parallel expansion afterwards. So the
            // batch spends most of its wall clock with one thread building
            // packs while the rest of the pool idles.
            //
            // Bit-identical to the serial path by construction: `build_pack` is
            // a pure function of `(eng, domain_gates(root, trunk))`, and the LRU
            // is a pure cache -- a hit returns exactly what a miss would have
            // rebuilt. The only observable difference is LRU EVICTION ORDER at
            // the 64-entry cap, which can change later hit/miss timing but
            // never a bound, because a miss recomputes the identical pack.
            // #margin-row-gpu-batch (DARK, NY_MARGIN_ROW_GPU=1 AND
            // NY_MARGIN_ROW_GPU_BATCH=1): fold the batch's LRU misses in
            // chunked certified GPU calls instead of one per domain. This is
            // the batching shape intended to attack the measured 32x workload
            // gap — see `gpu_seam::batch` for what varies per domain and why
            // the batch is expressible. It only POPULATES the cache: the serial
            // `rows_for` loop below is unchanged and simply hits on every entry,
            // so a refusal (or a dark gate) costs nothing and the established
            // path rebuilds. Runs BEFORE the CPU parallel prebuild so the two
            // never do the same work twice.
            self.gpu_batch_prefill(&batch);
            if parallel_prebuild_enabled() && batch.len() > 1 {
                let mut missing: Vec<Vec<TrunkSplit>> = Vec::new();
                for e in &batch {
                    if self.lru.get(&e.trunk).is_none()
                        && !missing.iter().any(|m| m.as_slice() == e.trunk.as_slice())
                    {
                        missing.push(e.trunk.clone());
                    }
                }
                if !missing.is_empty() {
                    // Attribute the prebuild to YRefresh. Building the packs
                    // here bypasses `rows_for`, so without this the phase reads
                    // "0.000s n=0" and the work becomes invisible rather than
                    // free -- the exact failure mode this profiler was just
                    // fixed for at the slice cap. Counted as one miss per pack
                    // so lru_miss still reflects real rebuild work.
                    let _t = super::prof::Timer::start(super::prof::Phase::YRefresh);
                    super::prof::bump(super::prof::Counter::LruMiss, missing.len() as u64);
                    let eng = &self.eng;
                    let built_packs: Vec<Result<(Vec<TrunkSplit>, Arc<YPack>)>> = missing
                        .into_par_iter()
                        .map(|trunk| {
                            let dom = domain_gates(eng.root, &trunk);
                            let pack = Arc::new(build_pack(eng, Some(&dom))?);
                            Ok((trunk, pack))
                        })
                        .collect();
                    for entry in built_packs {
                        let (trunk, pack) = entry?;
                        self.lru.put(trunk, pack);
                    }
                }
            }
            let mut packs: Vec<Arc<YPack>> = Vec::with_capacity(batch.len());
            let mut built = 0usize;
            while built < batch.len() {
                if built > 0 {
                    if let Some(dl) = self.cfg.deadline {
                        if Instant::now() > dl {
                            stop = "wallclock".into();
                            break;
                        }
                    }
                }
                let trunk = batch[built].trunk.clone();
                packs.push(self.rows_for(&trunk)?);
                built += 1;
            }
            for e in batch.drain(built..) {
                heap.push(e);
            }
            // Gate OFF: retain the established per-domain expansion path
            // exactly. Gate ON: evaluate/select independently, stack only the
            // candidate-score rows, then restore batch order before the same
            // deterministic serial heap merge below.
            let steps: Vec<Result<ExpandStep>> = if self.cfg.domain_stack && batch.len() > 1 {
                self.expand_domain_stacked_batch(&batch, &packs)?
                    .into_iter()
                    .map(Ok)
                    .collect()
            } else {
                // Concurrent expansion (each domain's bound is bit-identical
                // to serial; see method docs). `this` is a shared &Self across
                // workers.
                let this: &Self = self;
                batch
                    .par_iter()
                    .zip(packs.par_iter())
                    .map(|(entry, pack)| this.expand_one(pack, entry))
                    .collect()
            };
            super::prof::bump(super::prof::Counter::FrontierBatch, 1);
            super::prof::bump(super::prof::Counter::FrontierPopped, batch.len() as u64);
            // Deterministic serial merge in pop order.
            for (step, entry) in steps.into_iter().zip(&batch) {
                let depth = entry.trunk.len() + entry.heads.len();
                match step? {
                    ExpandStep::Infeasible | ExpandStep::ClosedAtEval => {
                        self.stats.closed += 1;
                        self.ledger.leaf(depth);
                    }
                    ExpandStep::NoCandidates => {
                        stop = "no_candidates".into();
                        break 'outer;
                    }
                    ExpandStep::NonFiniteBound => {
                        return Ok(MarginRowOutcome::Unknown {
                            reason: "non-finite domain or child bound".into(),
                            stats: Some(self.stats.clone()),
                        });
                    }
                    ExpandStep::Expanded {
                        dips,
                        worst_dip,
                        pushes,
                        closed_children,
                        created_children,
                    } => {
                        self.stats.mono_raw_dips += dips;
                        self.stats.mono_worst = self.stats.mono_worst.min(worst_dip);
                        self.stats.domains_created += created_children;
                        self.stats.closed += closed_children;
                        for _ in 0..closed_children {
                            self.ledger.leaf(depth + 1);
                        }
                        for c in pushes {
                            heap.push(DomainEntry {
                                bound: c.bound,
                                seq,
                                trunk: c.trunk,
                                heads: c.heads,
                                ly: c.ly,
                                uy: c.uy,
                            });
                            seq += 1;
                        }
                        self.stats.expansions += 1;
                    }
                }
            }
        }
        self.stats.stop = stop.clone();
        self.stats.elapsed_secs = t0.elapsed().as_secs_f64();
        let mut verified = matches!(stop.as_str(), "verified_bestfirst" | "queue_empty");
        if verified && self.cfg.deadline.is_some_and(|dl| Instant::now() > dl) {
            stop = "wallclock_before_verdict".into();
            self.stats.stop.clone_from(&stop);
            verified = false;
        }
        if verified {
            self.stats.ledger_ok = Some(self.ledger.kraft_ok() && self.nested_ledgers_ok);
            Ok(MarginRowOutcome::Unsat(self.stats.clone()))
        } else {
            Ok(MarginRowOutcome::Unknown {
                reason: stop,
                stats: Some(self.stats.clone()),
            })
        }
    }

    /// Gate-on parallel-frontier expansion. Evaluation and candidate selection
    /// remain per-domain and rayon-parallel. Only ready domains' candidate
    /// rows enter one cross-domain engine pass; results are returned in the
    /// original pop order for the established serial heap merge.
    fn expand_domain_stacked_batch(
        &self,
        batch: &[DomainEntry],
        packs: &[Arc<YPack>],
    ) -> Result<Vec<ExpandStep>> {
        if batch.len() != packs.len() {
            return Err(NyError::InvalidSpec(
                "margin_row: domain-stack batch/pack length mismatch".into(),
            ));
        }
        let preps: Vec<Result<PreparedExpand>> = batch
            .par_iter()
            .zip(packs.par_iter())
            .map(|(entry, pack)| self.prepare_expand(pack, entry))
            .collect();
        let preps: Vec<PreparedExpand> = preps.into_iter().collect::<Result<_>>()?;
        let ready_idx: Vec<usize> = preps
            .iter()
            .enumerate()
            .filter_map(|(i, prep)| matches!(prep, PreparedExpand::Ready(_)).then_some(i))
            .collect();
        let mut scores: Vec<Option<Vec<(f64, f64)>>> = vec![None; preps.len()];
        if ready_idx.len() > 1 {
            let ready: Vec<&ReadyExpand> = ready_idx
                .iter()
                .map(|&i| match &preps[i] {
                    PreparedExpand::Ready(ready) => ready.as_ref(),
                    PreparedExpand::Done(_) => unreachable!("ready index"),
                })
                .collect();
            let stacked = {
                let _t = super::prof::Timer::start(super::prof::Phase::ScoreCands);
                self.score_candidates_domain_stacked(&ready)?
            };
            for (&i, score) in ready_idx.iter().zip(stacked) {
                scores[i] = Some(score);
            }
        } else if let Some(&i) = ready_idx.first() {
            let PreparedExpand::Ready(ready) = &preps[i] else {
                unreachable!("ready index")
            };
            let score = {
                let _t = super::prof::Timer::start(super::prof::Phase::ScoreCands);
                self.score_candidates(&ready.st, &ready.trunk_cands, &ready.head_cands)?
            };
            scores[i] = Some(score);
        }

        let mut out = Vec::with_capacity(preps.len());
        for (i, (prep, entry)) in preps.into_iter().zip(batch).enumerate() {
            match prep {
                PreparedExpand::Done(step) => out.push(step),
                PreparedExpand::Ready(ready) => {
                    let score = scores[i].take().ok_or_else(|| {
                        NyError::InvalidSpec(
                            "margin_row: missing domain-stack candidate score".into(),
                        )
                    })?;
                    out.push(self.finish_prepared_expand(entry, &ready, &score));
                }
            }
        }
        Ok(out)
    }

    fn prepare_expand(&self, pack: &Arc<YPack>, entry: &DomainEntry) -> Result<PreparedExpand> {
        let mode = self.eng.root.mode;
        let Some((b_eval, st)) = self.eval_with_pack(pack, entry)? else {
            return Ok(PreparedExpand::Done(ExpandStep::Infeasible));
        };
        if !(b_eval.is_finite() && entry.bound.is_finite()) {
            return Ok(PreparedExpand::Done(ExpandStep::NonFiniteBound));
        }
        let b = b_eval.max(entry.bound);
        if closed(mode, b) {
            return Ok(PreparedExpand::Done(ExpandStep::ClosedAtEval));
        }
        let (trunk_cands, head_cands) = self.select_candidates(&st, entry);
        if head_cands.is_empty() && trunk_cands.is_empty() {
            return Ok(PreparedExpand::Done(ExpandStep::NoCandidates));
        }
        Ok(PreparedExpand::Ready(Box::new(ReadyExpand {
            b,
            st,
            trunk_cands,
            head_cands,
        })))
    }

    fn finish_prepared_expand(
        &self,
        entry: &DomainEntry,
        ready: &ReadyExpand,
        ch: &[(f64, f64)],
    ) -> ExpandStep {
        if ch.len() != ready.trunk_cands.len() + ready.head_cands.len()
            || ch
                .iter()
                .any(|(left, right)| !(left.is_finite() && right.is_finite()))
        {
            return ExpandStep::NonFiniteBound;
        }
        // Pick: max by (min child, sum child); later index wins ties.
        let mut pick = 0usize;
        let mut best = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for (i, pair) in ch.iter().enumerate() {
            let key = (pair.0.min(pair.1), pair.0 + pair.1);
            if key.0 > best.0 || (key.0 == best.0 && key.1 >= best.1) {
                best = key;
                pick = i;
            }
        }
        let n_t = ready.trunk_cands.len();
        let chb = ch[pick];
        let mut pushes = Vec::new();
        let mut dips = 0usize;
        let mut worst_dip = 0.0f64;
        let mut closed_children = 0usize;
        let mut created_children = 0usize;
        for (d_i, dr) in [(0usize, 1i8), (1usize, -1i8)] {
            let raw = if d_i == 0 { chb.0 } else { chb.1 };
            if !raw.is_finite() {
                return ExpandStep::NonFiniteBound;
            }
            if raw < ready.b - 1e-9 {
                dips += 1;
                worst_dip = worst_dip.min(raw - ready.b);
            }
            let cb = raw.max(ready.b);
            let (trunk_c, heads_c, ly_c, uy_c) = if pick < n_t {
                super::prof::bump(super::prof::Counter::TrunkSplit, 1);
                let (li, pos) = ready.trunk_cands[pick];
                let mut tc = entry.trunk.clone();
                tc.push((li, pos, dr));
                (
                    tc,
                    entry.heads.clone(),
                    ready.st.ybox.ly.clone(),
                    ready.st.ybox.uy.clone(),
                )
            } else {
                super::prof::bump(super::prof::Counter::HeadSplit, 1);
                let i = ready.head_cands[pick - n_t];
                let mut ly = ready.st.ybox.ly.clone();
                let mut uy = ready.st.ybox.uy.clone();
                if dr > 0 {
                    ly[i] = ly[i].max(0.0);
                } else {
                    uy[i] = uy[i].min(0.0);
                }
                if ly[i] > uy[i] + 1e-12 {
                    closed_children += 1;
                    created_children += 1;
                    continue;
                }
                let mut hc = entry.heads.clone();
                hc.push((i, dr));
                (entry.trunk.clone(), hc, ly, uy)
            };
            created_children += 1;
            if closed(self.eng.root.mode, cb) {
                closed_children += 1;
            } else {
                pushes.push(ChildProto {
                    bound: cb,
                    trunk: trunk_c,
                    heads: heads_c,
                    ly: ly_c,
                    uy: uy_c,
                });
            }
        }
        ExpandStep::Expanded {
            dips,
            worst_dip,
            pushes,
            closed_children,
            created_children,
        }
    }

    /// Evaluate+expand one popped domain given a PREBUILT pack (pure, `&self`).
    /// Mirrors the serial expansion body exactly; see [`Self::tree_loop`].
    fn expand_one(&self, pack: &Arc<YPack>, entry: &DomainEntry) -> Result<ExpandStep> {
        let mode = self.eng.root.mode;
        let Some((b_eval, st)) = self.eval_with_pack(pack, entry)? else {
            return Ok(ExpandStep::Infeasible);
        };
        if !(b_eval.is_finite() && entry.bound.is_finite()) {
            return Ok(ExpandStep::NonFiniteBound);
        }
        let b = b_eval.max(entry.bound);
        if closed(mode, b) {
            return Ok(ExpandStep::ClosedAtEval);
        }
        let (trunk_cands, head_cands) = self.select_candidates(&st, entry);
        if head_cands.is_empty() && trunk_cands.is_empty() {
            return Ok(ExpandStep::NoCandidates);
        }
        let ch = {
            let _t = super::prof::Timer::start(super::prof::Phase::ScoreCands);
            self.score_candidates(&st, &trunk_cands, &head_cands)?
        };
        if ch
            .iter()
            .any(|(left, right)| !(left.is_finite() && right.is_finite()))
        {
            return Ok(ExpandStep::NonFiniteBound);
        }
        // Pick: max by (min child, sum child); later index wins ties.
        let mut pick = 0usize;
        let mut best = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for (i, pair) in ch.iter().enumerate() {
            let key = (pair.0.min(pair.1), pair.0 + pair.1);
            if key.0 > best.0 || (key.0 == best.0 && key.1 >= best.1) {
                best = key;
                pick = i;
            }
        }
        let n_t = trunk_cands.len();
        let chb = ch[pick];
        let mut pushes = Vec::new();
        let mut dips = 0usize;
        let mut worst_dip = 0.0f64;
        let mut closed_children = 0usize;
        let mut created_children = 0usize;
        for (d_i, dr) in [(0usize, 1i8), (1usize, -1i8)] {
            let raw = if d_i == 0 { chb.0 } else { chb.1 };
            if !raw.is_finite() {
                return Ok(ExpandStep::NonFiniteBound);
            }
            if raw < b - 1e-9 {
                dips += 1;
                worst_dip = worst_dip.min(raw - b);
            }
            let cb = raw.max(b);
            let (trunk_c, heads_c, ly_c, uy_c) = if pick < n_t {
                super::prof::bump(super::prof::Counter::TrunkSplit, 1);
                let (li, pos) = trunk_cands[pick];
                let mut tc = entry.trunk.clone();
                tc.push((li, pos, dr));
                (
                    tc,
                    entry.heads.clone(),
                    st.ybox.ly.clone(),
                    st.ybox.uy.clone(),
                )
            } else {
                super::prof::bump(super::prof::Counter::HeadSplit, 1);
                let i = head_cands[pick - n_t];
                let mut ly = st.ybox.ly.clone();
                let mut uy = st.ybox.uy.clone();
                if dr > 0 {
                    ly[i] = ly[i].max(0.0);
                } else {
                    uy[i] = uy[i].min(0.0);
                }
                if ly[i] > uy[i] + 1e-12 {
                    closed_children += 1;
                    created_children += 1;
                    continue;
                }
                let mut hc = entry.heads.clone();
                hc.push((i, dr));
                (entry.trunk.clone(), hc, ly, uy)
            };
            created_children += 1;
            if closed(mode, cb) {
                closed_children += 1;
            } else {
                pushes.push(ChildProto {
                    bound: cb,
                    trunk: trunk_c,
                    heads: heads_c,
                    ly: ly_c,
                    uy: uy_c,
                });
            }
        }
        Ok(ExpandStep::Expanded {
            dips,
            worst_dip,
            pushes,
            closed_children,
            created_children,
        })
    }

    /// VERIFIER DIFFERENTIAL (test-only). Reconstruct a driver and walk the
    /// tree by strictly descending into the first child of the best split for
    /// up to `steps` domains. At EACH visited domain compute the certified
    /// bound two ways: (a) serially via `eval_node` (the oracle path) and
    /// (b) concurrently from `WORKERS` rayon threads via `eval_with_pack` (the
    /// exact call the parallel frontier makes inside a worker). Returns one
    /// `(depth, serial_bits, every_worker_bit_identical_to_serial)` per domain.
    /// A `false` in the third slot is a soundness break: the SAME domain's
    /// certified bound moved when computed under concurrency.
    #[cfg(test)]
    pub(crate) fn diff_walk_serial_vs_worker(
        net: &'a TwinNet,
        root: &'a RootGates,
        t: usize,
        adv: &[usize],
        steps: usize,
    ) -> Vec<(usize, u64, bool)> {
        const WORKERS: usize = 24;
        let eng = BackwardEngine::new(net, root);
        let re = root_eval(&eng, net, t, adv).expect("root_eval");
        let mb = MarginBatch::new(net, t, &re.tree_classes).expect("margin batch");
        let root_bound = re
            .dj
            .iter()
            .zip(adv)
            .filter(|(_, j)| re.tree_classes.contains(j))
            .map(|(b, _)| *b)
            .fold(f64::INFINITY, f64::min);
        let mut bab = Self {
            eng,
            mb,
            cfg: BabConfig {
                lru_cap: 64,
                ..BabConfig::default()
            },
            lru: Lru {
                cap: 64,
                entries: Vec::new(),
            },
            stats: BabStats {
                root_bound,
                tree_classes: re.tree_classes.clone(),
                root_closed_classes: 0,
                expansions: 0,
                domains_created: 1,
                closed: 0,
                max_depth: 0,
                mono_raw_dips: 0,
                mono_worst: 0.0,
                stop: String::new(),
                elapsed_secs: 0.0,
                class_runs: Vec::new(),
                epochs_attempted: 0,
                epochs_closed: 0,
                ledger_ok: None,
            },
            epoch_failures: 0,
            ledger: Ledger::default(),
            nested_ledgers_ok: true,
        };
        bab.lru.put(Vec::new(), re.pack.clone());
        let mut entry = DomainEntry {
            bound: root_bound,
            seq: 0,
            trunk: Vec::new(),
            heads: Vec::new(),
            ly: re.pack.ly0.clone(),
            uy: re.pack.uy0.clone(),
        };
        let mut out = Vec::new();
        for _ in 0..steps {
            // (a) serial oracle bound (populates/reads the shared cache).
            let Some((b_ser, _st)) = bab.eval_node(&entry).expect("eval_node") else {
                break;
            };
            // (b) prebuild the pack once, then hammer eval_with_pack from many
            // rayon workers at once — the production frontier's concurrency.
            let pack = bab.rows_for(&entry.trunk).expect("rows_for");
            let bab_ref: &Self = &bab;
            let entry_ref = &entry;
            let worker_bits: Vec<u64> = (0..WORKERS)
                .into_par_iter()
                .map(|_| {
                    bab_ref
                        .eval_with_pack(&pack, entry_ref)
                        .expect("eval_with_pack")
                        .map_or(0xDEAD_BEEF, |(b, _)| b.to_bits())
                })
                .collect();
            let depth = entry.trunk.len() + entry.heads.len();
            let all_eq = worker_bits.iter().all(|&x| x == b_ser.to_bits());
            out.push((depth, b_ser.to_bits(), all_eq));
            // Descend into the first child of this domain's best split.
            match bab.expand_one(&pack, &entry).expect("expand_one") {
                ExpandStep::Expanded { mut pushes, .. } if !pushes.is_empty() => {
                    let c = pushes.swap_remove(0);
                    entry = DomainEntry {
                        bound: c.bound,
                        seq: 0,
                        trunk: c.trunk,
                        heads: c.heads,
                        ly: c.ly,
                        uy: c.uy,
                    };
                }
                _ => break,
            }
        }
        out
    }

    /// DOMAIN-BATCHED y-pack prefill (#margin-row-gpu-batch).
    ///
    /// Collects the frontier batch's DISTINCT LRU-missing trunk-sets, folds
    /// their identity-seeded y-row refreshes in chunked calls at a
    /// device-admitted width, and puts the resulting packs in the LRU. The
    /// established `rows_for` loop then hits on every entry.
    ///
    /// # Why this is the right seam
    ///
    /// `build_pack` is a pure function of `(eng, domain_gates(root, trunk))` and
    /// the LRU is a pure cache — a hit returns exactly what a miss would have
    /// rebuilt. So POPULATING the cache is the smallest possible blast radius:
    /// no bound, no candidate, no heap order and no stats delta is computed
    /// differently, and the only thing that changed is which arithmetic produced
    /// `(al, au)`. That arithmetic is authoritative and is guarded per domain
    /// (certified-error floor + realization probe against THAT domain's own
    /// gates) inside `gpu_seam::batch`.
    ///
    /// # Fail-closed
    ///
    /// Returns no value or error to its caller: any terminal refusal simply
    /// leaves the cache unpopulated, and the CPU path rebuilds. Counters and a
    /// one-time profile diagnostic may still record the refusal. There is no
    /// failure here that the established path does not already handle.
    fn gpu_batch_prefill(&mut self, batch: &[DomainEntry]) {
        if !super::gpu_seam::batch::enabled() || batch.len() < 2 {
            return;
        }
        // DISTINCT misses, in pop order. Deduping matters for correctness of the
        // accounting (two entries with the same trunk-set share one pack) and
        // for width (a duplicated domain wastes a batch slot).
        let mut missing: Vec<Vec<TrunkSplit>> = Vec::new();
        for e in batch {
            if self.lru.get(&e.trunk).is_none()
                && !missing.iter().any(|m| m.as_slice() == e.trunk.as_slice())
            {
                missing.push(e.trunk.clone());
            }
        }
        if missing.len() < 2 {
            return;
        }
        let gates: Vec<DomainGates> = missing
            .iter()
            .map(|trunk| domain_gates(self.eng.root, trunk))
            .collect();
        let refs: Vec<&DomainGates> = gates.iter().collect();
        // Attribute the batched refresh to YRefresh and count one miss per pack,
        // exactly as the CPU prebuild does, so the profile stays comparable
        // across an A/B of this gate.
        let _t = super::prof::Timer::start(super::prof::Phase::YRefresh);
        let Some(rows) = super::gpu_seam::batch::run_batch(&self.eng, &refs, self.cfg.deadline)
        else {
            return;
        };
        // The slot map, one last time: `run_batch` returns one pass pair per
        // input gate, in input order. Re-pair by ZIP (no index) and refuse the
        // whole prefill on any length drift rather than cache a pack under the
        // wrong trunk-set — that would be a wrong bound for every domain that
        // later hits it.
        if rows.len() != missing.len() {
            return;
        }
        super::prof::bump(super::prof::Counter::LruMiss, missing.len() as u64);
        for (trunk, (al, au)) in missing.into_iter().zip(rows) {
            let pack = Arc::new(pack_from_rows(&self.eng, al, au));
            self.lru.put(trunk, pack);
        }
    }

    fn rows_for(&mut self, trunk: &[TrunkSplit]) -> Result<Arc<YPack>> {
        if let Some(p) = self.lru.get(trunk) {
            super::prof::bump(super::prof::Counter::LruHit, 1);
            return Ok(p);
        }
        super::prof::bump(super::prof::Counter::LruMiss, 1);
        let _t = super::prof::Timer::start(super::prof::Phase::YRefresh);
        let dom = domain_gates(self.eng.root, trunk);
        let pack = Arc::new(build_pack(&self.eng, Some(&dom))?);
        self.lru.put(trunk.to_vec(), pack.clone());
        Ok(pack)
    }

    /// Refreshed bound + branching state; `None` = provably infeasible.
    /// Serial path: fetch (or build) the domain's y-pack, then evaluate.
    fn eval_node(&mut self, entry: &DomainEntry) -> Result<Option<(f64, NodeState)>> {
        let pack = self.rows_for(&entry.trunk)?;
        self.eval_with_pack(&pack, entry)
    }

    /// Pure domain evaluation given a PREBUILT y-pack (no cache, no `&mut`):
    /// the parallel-frontier lane reuses this across rayon workers. Every
    /// bound here is a deterministic function of `(pack, entry)`, so a domain's
    /// result is BIT-IDENTICAL whether evaluated serially or in a worker.
    fn eval_with_pack(
        &self,
        pack: &Arc<YPack>,
        entry: &DomainEntry,
    ) -> Result<Option<(f64, NodeState)>> {
        let mode = self.eng.root.mode;
        let _t = super::prof::Timer::start(super::prof::Phase::EvalNode);
        let dom = domain_gates(self.eng.root, &entry.trunk);
        let pack = pack.clone();
        let mut ybox = YBox {
            ly: pack.ly0.clone(),
            uy: pack.uy0.clone(),
        };
        ybox.clamp(&entry.heads);
        if ybox.is_empty() {
            return Ok(None);
        }
        ybox.intersect(&entry.ly, &entry.uy);
        if ybox.is_empty() {
            return Ok(None);
        }
        let gates = head_gates(&ybox, mode);
        let ms = margin_seed(&self.mb, &gates, &ybox, mode);
        let mut pass = self.eng.run_collect(
            &ms.seed,
            Some(&dom),
            super::engine::LaneDir::Lower,
            None,
            Collect {
                unst_abs: true,
                rows: if self.cfg.tier0_exact > 0 {
                    self.cfg.retained.as_deref()
                } else {
                    None
                },
            },
        )?;
        let per_j = per_class_direct(&self.eng, &pass, &ms, 0..self.mb.nf());
        let m2v = compose_viay(
            &self.eng,
            &self.mb,
            &gates,
            &pack.al,
            &pack.au,
            &pack.al_dots,
            &pack.au_dots,
            mode,
        );
        let mut b = f64::INFINITY;
        for r0 in 0..self.mb.nf() {
            if !(per_j[r0].is_finite() && ms.m1[r0].is_finite() && m2v[r0].is_finite()) {
                return Err(NyError::NumericalInstability(
                    "margin_row: non-finite per-class domain component".into(),
                ));
            }
            let pj = per_j[r0].max(ms.m1[r0]).max(m2v[r0]);
            if !pj.is_finite() {
                return Err(NyError::NumericalInstability(
                    "margin_row: non-finite per-class domain bound".into(),
                ));
            }
            b = b.min(pj);
        }
        if !b.is_finite() {
            return Err(NyError::NumericalInstability(
                "margin_row: non-finite node bound".into(),
            ));
        }
        let coll = pass.coll.take().unwrap_or_default();
        let coll_rows = pass.coll_rows.take().unwrap_or_default();
        Ok(Some((
            b,
            NodeState {
                ybox,
                gates,
                ms,
                coll,
                coll_rows,
                pass,
                dom,
                pack,
            },
        )))
    }

    /// Exact single-gate variant scores of every unstable, unused head
    /// neuron: `(min child, child sum, neuron)`, unsorted.
    fn head_variant_scores(&self, st: &NodeState, heads: &[HeadFix]) -> Vec<(f64, f64, usize)> {
        let used: std::collections::BTreeSet<usize> = heads.iter().map(|(i, _)| *i).collect();
        let cands: Vec<usize> = (0..self.mb.n_y)
            .filter(|&i| st.ybox.ly[i] < 0.0 && st.ybox.uy[i] > 0.0 && !used.contains(&i))
            .collect();
        if cands.is_empty() {
            return Vec::new();
        }
        let vs = variant_state(&self.mb, &st.gates, &st.ybox, &st.pack.al, &st.pack.au);
        cands
            .into_iter()
            .filter_map(|i| {
                let ba = head_variant(
                    &self.mb,
                    &vs,
                    &st.gates,
                    &st.ybox,
                    &st.pack.al,
                    &st.pack.au,
                    self.eng.root,
                    i,
                    1,
                );
                let bi = head_variant(
                    &self.mb,
                    &vs,
                    &st.gates,
                    &st.ybox,
                    &st.pack.al,
                    &st.pack.au,
                    self.eng.root,
                    i,
                    -1,
                );
                (ba.is_finite() && bi.is_finite()).then_some((ba.min(bi), ba + bi, i))
            })
            .collect()
    }

    /// Exact single-gate variant pre-rank of unstable head neurons.
    fn head_prerank(&self, st: &NodeState, heads: &[HeadFix]) -> Vec<usize> {
        let mut scored = self.head_variant_scores(st, heads);
        scored.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| b.1.total_cmp(&a.1))
                .then_with(|| b.2.cmp(&a.2))
        });
        scored
            .into_iter()
            .take(self.cfg.k_head)
            .map(|(_, _, i)| i)
            .collect()
    }

    /// Tier-2 epoch attempt (#epoch-bab, design doc §2.3): when the popped
    /// domain carries enough trunk splits, rebuild the trunk tableau with
    /// those splits BAKED (exact fixed lines at the split neurons, every
    /// downstream gate re-derived from the resulting tighter boxes) and run
    /// the domain's subtree as a NESTED lane over the epoch gates. Returns
    /// `true` iff the nested run certified the whole subtree (the domain is
    /// then closed). Every failure path — guards, rebuild error, nested
    /// `Unknown` — returns `false` and the caller expands the domain
    /// normally under the old gates (fail-open to the status quo, NEVER to
    /// a verdict).
    ///
    /// SOUNDNESS: the epoch gates are valid exactly on the split-halfspace
    /// intersection (same contract as `engine::domain_gates`, applied in the
    /// forward build — `RootGates::build_retaining` `splits` docs), which is
    /// precisely the popped domain. Inherited head fixes are semantic sign
    /// constraints on head pre-activations (gate-independent) and the
    /// inherited y-box is a certified box for a superset domain; both remain
    /// valid under the epoch. The nested lane runs the SAME certified
    /// Outward machinery, so a nested `Unsat` is a certified cover of the
    /// subtree by closed leaves.
    fn try_epoch(&mut self, entry: &DomainEntry) -> bool {
        if self.cfg.epoch_depth == 0 || entry.trunk.len() < self.cfg.epoch_depth {
            return false;
        }
        if self.epoch_failures >= self.cfg.epoch_max_attempts {
            return false;
        }
        if let Some(dl) = self.cfg.deadline {
            let rem = dl.saturating_duration_since(Instant::now()).as_secs_f64();
            if rem < 3.0 * self.cfg.root_build_secs.max(0.05) {
                return false;
            }
        }
        self.stats.epochs_attempted += 1;
        super::prof::bump(super::prof::Counter::EpochAttempt, 1);
        let _t = super::prof::Timer::start(super::prof::Phase::EpochBuild);
        // Splits by ABSOLUTE neuron index (the epoch derives its own unst
        // lists; positions are gate-set-relative and do not transfer).
        let abs: Vec<(usize, usize, i8)> = entry
            .trunk
            .iter()
            .map(|&(li, pos, d)| (li, self.eng.root.layers[li].unst[pos], d))
            .collect();
        let net = self.eng.net;
        let root = self.eng.root;
        let built = RootGates::build_retaining(
            net,
            &root.lo,
            &root.hi,
            root.mode,
            self.cfg.deadline,
            self.cfg.retain_cfg.as_ref(),
            &abs,
        );
        let Ok((eroot, eret)) = built else {
            // Rebuild failed (deadline / numeric): normal expansion instead.
            self.epoch_failures += 1;
            return false;
        };
        // Nested runs do NOT epoch again (`epoch_depth: 0`): re-linearization
        // is ITERATIVE through the outer loop — if the nested subtree stalls
        // and returns Unknown, the outer tree keeps expanding under the old
        // gates and its deeper domains re-trigger a fresh epoch with their
        // fuller split sets. Recursion depth is therefore exactly one, which
        // both bounds the stack and keeps the failed-attempt accounting in
        // one place.
        let ncfg = BabConfig {
            k_head: self.cfg.k_head,
            k_trunk: self.cfg.k_trunk,
            max_expansions: self
                .cfg
                .max_expansions
                .saturating_sub(self.stats.expansions),
            deadline: self.cfg.deadline,
            lru_cap: self.cfg.lru_cap,
            frontier: 1,
            domain_stack: false,
            tier0_exact: self.cfg.tier0_exact,
            tier0_universe: self.cfg.tier0_universe,
            retained: eret.map(Arc::new),
            epoch_depth: 0,
            epoch_max_attempts: 0,
            root_build_secs: self.cfg.root_build_secs,
            retain_cfg: self.cfg.retain_cfg,
            initial_heads: entry.heads.clone(),
            initial_ybox: Some((entry.ly.clone(), entry.uy.clone())),
        };
        let adv = self.stats.tree_classes.clone();
        let dbg = std::env::var("NY_EPOCH_DEBUG").is_ok();
        if dbg {
            eprintln!(
                "[epoch] attempt at depth {} (trunk {} heads {}): outer bound {:+.5}",
                entry.trunk.len() + entry.heads.len(),
                entry.trunk.len(),
                entry.heads.len(),
                entry.bound
            );
        }
        let t_epoch = Instant::now();
        match MarginRowBab::run_inner(net, &eroot, self.mb.t, &adv, ncfg) {
            Ok(MarginRowOutcome::Unsat(n)) => {
                super::prof::bump(super::prof::Counter::EpochClosed, 1);
                if dbg {
                    eprintln!(
                        "[epoch] CLOSED subtree in {:.1}s: nested root {:+.5} exp {} maxD {}",
                        t_epoch.elapsed().as_secs_f64(),
                        n.root_bound,
                        n.expansions,
                        n.max_depth
                    );
                }
                self.nested_ledgers_ok &= matches!(n.ledger_ok, Some(true));
                self.stats.epochs_closed += 1;
                self.stats.epochs_attempted += n.epochs_attempted;
                self.stats.epochs_closed += n.epochs_closed;
                self.stats.expansions += n.expansions;
                self.stats.domains_created += n.domains_created;
                self.stats.closed += n.closed;
                self.stats.mono_raw_dips += n.mono_raw_dips;
                self.stats.mono_worst = self.stats.mono_worst.min(n.mono_worst);
                self.stats.max_depth = self
                    .stats
                    .max_depth
                    .max(entry.trunk.len() + entry.heads.len() + n.max_depth);
                true
            }
            Ok(MarginRowOutcome::Unknown { stats, .. }) => {
                self.epoch_failures += 1;
                if dbg {
                    if let Some(n) = &stats {
                        eprintln!(
                            "[epoch] UNKNOWN after {:.1}s: nested root {:+.5} (outer {:+.5}) \
                             exp {} maxD {} stop {}",
                            t_epoch.elapsed().as_secs_f64(),
                            n.root_bound,
                            entry.bound,
                            n.expansions,
                            n.max_depth,
                            n.stop
                        );
                    }
                }
                if let Some(n) = stats {
                    self.stats.epochs_attempted += n.epochs_attempted;
                    self.stats.epochs_closed += n.epochs_closed;
                    self.stats.expansions += n.expansions;
                    self.stats.domains_created += n.domains_created;
                    self.stats.closed += n.closed;
                    self.stats.mono_raw_dips += n.mono_raw_dips;
                    self.stats.mono_worst = self.stats.mono_worst.min(n.mono_worst);
                }
                false
            }
            Err(_) => {
                self.epoch_failures += 1;
                false
            }
        }
    }

    /// Candidate selection for one expansion: the legacy shortlist protocol,
    /// or Tier-0 rank-1 ranking when configured (#epoch-bab).
    fn select_candidates(
        &self,
        st: &NodeState,
        entry: &DomainEntry,
    ) -> (Vec<(usize, usize)>, Vec<usize>) {
        if self.cfg.tier0_exact > 0 && self.cfg.retained.is_some() {
            return self.tier0_candidates(st, entry);
        }
        let head_cands = {
            let _t = super::prof::Timer::start(super::prof::Phase::HeadPrerank);
            self.head_prerank(st, &entry.heads)
        };
        let trunk_cands = {
            let _t = super::prof::Timer::start(super::prof::Phase::TrunkShortlist);
            self.trunk_shortlist(st, &entry.trunk)
        };
        (trunk_cands, head_cands)
    }

    /// Tier-0 (#epoch-bab): rank ALL unstable unused heads (exact variants)
    /// plus a wide trunk universe (rank-1 trunk variants against retained
    /// tableau rows), then keep only the strongest `tier0_exact` candidates
    /// overall for the exact Tier-1 pass. Ranker-only: every pushed bound
    /// and closure still comes from `score_candidates` (Outward, certified).
    fn tier0_candidates(
        &self,
        st: &NodeState,
        entry: &DomainEntry,
    ) -> (Vec<(usize, usize)>, Vec<usize>) {
        let _t = super::prof::Timer::start(super::prof::Phase::Tier0Rank);
        let ret = self.cfg.retained.as_deref().expect("checked by caller");
        // Pool entries: (min child, child sum, is_head, a, b) where
        // (a, b) = (layer, unst position) for trunks and (neuron, 0) heads.
        // Head and trunk scores are BOTH anchored on the parent's direct
        // pass (head_variant_direct / trunk_variant) so they pool on one
        // scale — the via-y-anchored head_variant systematically loses to
        // direct-anchored trunk scores and starves the tree of head splits
        // (measured on prop_1498).
        let mut pool: Vec<(f64, f64, bool, usize, usize)> = Vec::new();
        {
            let used: std::collections::BTreeSet<usize> =
                entry.heads.iter().map(|(i, _)| *i).collect();
            for i in (0..self.mb.n_y)
                .filter(|&i| st.ybox.ly[i] < 0.0 && st.ybox.uy[i] > 0.0 && !used.contains(&i))
            {
                let ba = super::bounds::head_variant_direct(
                    &self.mb,
                    &st.gates,
                    &st.ms,
                    &st.pass,
                    &st.pack.al,
                    &st.pack.au,
                    self.eng.root,
                    i,
                    1,
                );
                let bi = super::bounds::head_variant_direct(
                    &self.mb,
                    &st.gates,
                    &st.ms,
                    &st.pass,
                    &st.pack.al,
                    &st.pack.au,
                    self.eng.root,
                    i,
                    -1,
                );
                if ba.is_finite() && bi.is_finite() {
                    pool.push((ba.min(bi), ba + bi, true, i, 0));
                }
            }
        }
        let used: std::collections::BTreeSet<(usize, usize)> =
            entry.trunk.iter().map(|&(li, pos, _)| (li, pos)).collect();
        for (&li, coefs) in &st.coll {
            let Some(lr) = ret.layers.get(li) else {
                continue;
            };
            if lr.idx.is_empty() {
                continue;
            }
            let Some(vmat) = st.coll_rows.get(&li) else {
                continue;
            };
            let rec = &self.eng.root.layers[li];
            // Dynamic pre-rank within the retained set: |coef|-sum x c —
            // the same score the legacy shortlist uses (trunk_shortlist).
            let mut dyn_scored: Vec<(f64, usize)> = lr
                .unst_pos
                .iter()
                .enumerate()
                .filter(|&(_, &pos)| !used.contains(&(li, pos)))
                .map(|(ri, &pos)| (coefs[pos] * rec.c[lr.idx[ri]], ri))
                .filter(|&(m, _)| m.is_finite() && m > 0.0)
                .collect();
            dyn_scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
            dyn_scored.truncate(self.cfg.tier0_universe);
            let r = vmat.ncols();
            let vs = vmat.as_slice().expect("standard layout");
            for &(_, ri) in &dyn_scored {
                let vrow = &vs[ri * r..(ri + 1) * r];
                let ba = trunk_variant(self.eng.root, lr, li, ri, vrow, &st.pass, &st.ms, 1);
                let bi = trunk_variant(self.eng.root, lr, li, ri, vrow, &st.pass, &st.ms, -1);
                if ba.is_finite() && bi.is_finite() {
                    pool.push((ba.min(bi), ba + bi, false, li, lr.unst_pos[ri]));
                }
            }
        }
        // Strongest first — the same (min, sum) key the exact pick uses.
        pool.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| b.1.total_cmp(&a.1))
                .then_with(|| b.3.cmp(&a.3))
                .then_with(|| b.4.cmp(&a.4))
        });
        let k = self.cfg.tier0_exact.max(1);
        // Per-kind floor: keep at least one head and one trunk candidate in
        // the exact set when both kinds exist (guards residual scale bias
        // between the two rankers).
        let best_other = |pool: &[(f64, f64, bool, usize, usize)], kind: bool| {
            pool.iter().position(|&(_, _, h, _, _)| h == kind)
        };
        let mut take: Vec<usize> = (0..pool.len().min(k)).collect();
        if pool.len() > k {
            let has = |sel: &[usize], kind: bool| sel.iter().any(|&i| pool[i].2 == kind);
            for kind in [true, false] {
                if !has(&take, kind) {
                    if let Some(alt) = best_other(&pool, kind) {
                        let last = take.len() - 1;
                        take[last] = alt;
                    }
                }
            }
        }
        let mut trunks = Vec::new();
        let mut heads = Vec::new();
        for &i in &take {
            let (_, _, is_head, a, b) = pool[i];
            if is_head {
                heads.push(a);
            } else {
                trunks.push((a, b));
            }
        }
        (trunks, heads)
    }

    /// Trunk shortlist by |margin-row coef| x intercept from the seeded pass.
    fn trunk_shortlist(&self, st: &NodeState, trunk: &[TrunkSplit]) -> Vec<(usize, usize)> {
        let used: std::collections::BTreeSet<(usize, usize)> =
            trunk.iter().map(|&(li, pos, _)| (li, pos)).collect();
        let mut natives: Vec<(f64, usize, usize)> = Vec::new();
        for (&li, coefs) in &st.coll {
            let rec = &self.eng.root.layers[li];
            let mut sc: Vec<(f64, usize)> = coefs
                .iter()
                .enumerate()
                .map(|(pos, &m)| (m * rec.c[rec.unst[pos]], pos))
                .collect();
            // np.argsort(stable)[::-1] parity: descending score, ties by
            // descending position.
            sc.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
            for &(v, pos) in sc.iter().take(self.cfg.k_trunk + 4) {
                if v.is_finite() && v > 0.0 && !used.contains(&(li, pos)) {
                    natives.push((v, li, pos));
                }
            }
        }
        natives.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| b.1.cmp(&a.1))
                .then_with(|| b.2.cmp(&a.2))
        });
        natives
            .into_iter()
            .take(self.cfg.k_trunk)
            .map(|(_, li, pos)| (li, pos))
            .collect()
    }

    /// Build one domain's candidate columns without running the backward
    /// engine. This is the exact seed/side-lane/exception construction used by
    /// `score_candidates`, retained domain-local so ownership can be validated
    /// before cross-domain concatenation.
    fn candidate_columns(
        &self,
        st: &NodeState,
        trunk_cands: &[(usize, usize)],
        head_cands: &[usize],
    ) -> Result<CandidateColumns> {
        let nf = self.mb.nf();
        let n_y = self.mb.n_y;
        let n_t = trunk_cands.len();
        let n_h = head_cands.len();
        let n_candidates = n_t
            .checked_add(n_h)
            .ok_or_else(|| NyError::InvalidSpec("margin_row: candidate count overflow".into()))?;
        let total = n_candidates
            .checked_mul(2)
            .and_then(|v| v.checked_mul(nf))
            .ok_or_else(|| NyError::InvalidSpec("margin_row: candidate row overflow".into()))?;
        if total == 0 {
            return Err(NyError::InvalidSpec(
                "margin_row: empty candidate-score domain".into(),
            ));
        }
        let outward = self.eng.root.mode.outward();
        let mut seed = Array2::<f64>::zeros((n_y, total));
        let mut seed_e = outward.then(|| Array2::<f64>::zeros((n_y, total)));
        let mut cst = vec![0.0; total];
        let mut cst_err = vec![0.0; total];
        let mut m1 = vec![0.0; total];
        let mut exc = Exceptions::default();
        for (kc, &(li, pos)) in trunk_cands.iter().enumerate() {
            let rec = &self.eng.root.layers[li];
            let idx = rec.unst[pos];
            for (d_i, fix) in [(0usize, (1.0, 1.0, 0.0)), (1usize, (0.0, 0.0, 0.0))] {
                let r0 = (2 * kc + d_i) * nf;
                for j in 0..n_y {
                    for f in 0..nf {
                        seed[[j, r0 + f]] = st.ms.seed.s[[j, f]];
                        if let (Some(dst), Some(src)) = (seed_e.as_mut(), st.ms.seed.e.as_ref()) {
                            dst[[j, r0 + f]] = src[[j, f]];
                        }
                    }
                }
                for f in 0..nf {
                    cst[r0 + f] = st.ms.cst[f];
                    cst_err[r0 + f] = st.ms.cst_err[f];
                    m1[r0 + f] = st.ms.m1[f];
                    exc.by_layer.entry(li).or_default().push(Exc {
                        row: r0 + f,
                        neuron: idx,
                        a2: fix.0,
                        s2: fix.1,
                        c2: fix.2,
                    });
                }
            }
        }
        let base = 2 * n_t * nf;
        let mode = self.eng.root.mode;
        for (kc, &i) in head_cands.iter().enumerate() {
            for (d_i, dr) in [(0usize, 1i8), (1usize, -1i8)] {
                let r0 = base + (2 * kc + d_i) * nf;
                let mut gates2 = HeadGates {
                    alpha: st.gates.alpha.clone(),
                    s: st.gates.s.clone(),
                    c: st.gates.c.clone(),
                };
                if dr > 0 {
                    gates2.alpha[i] = 1.0;
                    gates2.s[i] = 1.0;
                    gates2.c[i] = 0.0;
                } else {
                    gates2.alpha[i] = 0.0;
                    gates2.s[i] = 0.0;
                    gates2.c[i] = 0.0;
                }
                let ms2 = margin_seed(&self.mb, &gates2, &st.ybox, mode);
                for j in 0..n_y {
                    for f in 0..nf {
                        seed[[j, r0 + f]] = ms2.seed.s[[j, f]];
                        if let (Some(dst), Some(src)) = (seed_e.as_mut(), ms2.seed.e.as_ref()) {
                            dst[[j, r0 + f]] = src[[j, f]];
                        }
                    }
                }
                for f in 0..nf {
                    cst[r0 + f] = ms2.cst[f];
                    cst_err[r0 + f] = ms2.cst_err[f];
                    m1[r0 + f] = if dr > 0 {
                        st.ms.m1[f]
                    } else {
                        let t = self.mb.wn[f * n_y + i] * (0.0 - st.ms.zu[i]);
                        let v = st.ms.m1[f] + t;
                        if outward {
                            next_down(v - UNIT * v.abs())
                        } else {
                            v
                        }
                    };
                }
            }
        }
        Ok(CandidateColumns {
            seed: Seed { s: seed, e: seed_e },
            cst,
            cst_err,
            m1,
            exc,
            n_candidates,
        })
    }

    /// One backward dispatch for the candidate-score columns of several
    /// independently gated domains. Arithmetic and reductions remain
    /// column-local; the engine validates the explicit ownership partition.
    fn score_candidates_domain_stacked(
        &self,
        ready: &[&ReadyExpand],
    ) -> Result<Vec<Vec<(f64, f64)>>> {
        if ready.len() < 2 {
            return Err(NyError::InvalidSpec(
                "margin_row: cross-domain score stack needs at least two domains".into(),
            ));
        }
        let columns: Vec<CandidateColumns> = ready
            .iter()
            .map(|ready| self.candidate_columns(&ready.st, &ready.trunk_cands, &ready.head_cands))
            .collect::<Result<_>>()?;
        let total = columns.iter().try_fold(0usize, |acc, domain| {
            acc.checked_add(domain.seed.s.ncols())
                .ok_or_else(|| NyError::InvalidSpec("margin_row: domain-stack overflow".into()))
        })?;
        let n_y = self.mb.n_y;
        let outward = self.eng.root.mode.outward();
        let mut seed = Array2::<f64>::zeros((n_y, total));
        let mut seed_e = outward.then(|| Array2::<f64>::zeros((n_y, total)));
        let mut offset = 0usize;
        for domain in &columns {
            let width = domain.seed.s.ncols();
            for j in 0..n_y {
                let src =
                    &domain.seed.s.as_slice().expect("standard layout")[j * width..(j + 1) * width];
                seed.as_slice_mut().expect("standard layout")
                    [j * total + offset..j * total + offset + width]
                    .copy_from_slice(src);
                if let (Some(dst), Some(src_e)) = (seed_e.as_mut(), domain.seed.e.as_ref()) {
                    let src =
                        &src_e.as_slice().expect("standard layout")[j * width..(j + 1) * width];
                    dst.as_slice_mut().expect("standard layout")
                        [j * total + offset..j * total + offset + width]
                        .copy_from_slice(src);
                }
            }
            offset += width;
        }
        let mut offset = 0usize;
        let blocks: Vec<RowDomainGateBlock<'_>> = ready
            .iter()
            .zip(&columns)
            .map(|(ready, domain)| {
                let start = offset;
                offset += domain.seed.s.ncols();
                RowDomainGateBlock {
                    columns: start..offset,
                    gates: &ready.st.dom,
                    exceptions: &domain.exc,
                }
            })
            .collect();
        if offset != total {
            return Err(NyError::InvalidSpec(
                "margin_row: domain-stack assembly mismatch".into(),
            ));
        }
        let pass = self.eng.run_domain_stacked(
            &Seed { s: seed, e: seed_e },
            &blocks,
            super::engine::LaneDir::Lower,
        )?;
        let low = self.eng.concretize_lower(&pass);
        if low.len() != total {
            return Err(NyError::InvalidSpec(
                "margin_row: domain-stack result shape mismatch".into(),
            ));
        }
        let nf = self.mb.nf();
        let mut all = Vec::with_capacity(columns.len());
        let mut offset = 0usize;
        for domain in &columns {
            let width = domain.seed.s.ncols();
            let mut out = Vec::with_capacity(domain.n_candidates);
            for kc in 0..domain.n_candidates {
                let mut pair = [f64::INFINITY, f64::INFINITY];
                for (d_i, p) in pair.iter_mut().enumerate() {
                    let r0 = (2 * kc + d_i) * nf;
                    let mut worst = f64::INFINITY;
                    for f in 0..nf {
                        let ri = r0 + f;
                        if !(low[offset + ri].is_finite()
                            && domain.cst[ri].is_finite()
                            && domain.cst_err[ri].is_finite()
                            && domain.m1[ri].is_finite())
                        {
                            return Err(NyError::NumericalInstability(
                                "margin_row: non-finite stacked child component".into(),
                            ));
                        }
                        let v = low[offset + ri] + domain.cst[ri];
                        let v = if outward {
                            next_down(next_down(v - next_up(domain.cst_err[ri])))
                        } else {
                            v
                        };
                        if !v.is_finite() {
                            return Err(NyError::NumericalInstability(
                                "margin_row: non-finite stacked child bound".into(),
                            ));
                        }
                        worst = worst.min(v.max(domain.m1[ri]));
                    }
                    if !worst.is_finite() {
                        return Err(NyError::NumericalInstability(
                            "margin_row: non-finite stacked aggregate child bound".into(),
                        ));
                    }
                    *p = worst;
                }
                out.push((pair[0], pair[1]));
            }
            all.push(out);
            offset += width;
        }
        #[cfg(test)]
        for (domain_i, (ready, stacked)) in ready.iter().zip(&all).enumerate() {
            let independent =
                self.score_candidates(&ready.st, &ready.trunk_cands, &ready.head_cands)?;
            assert_eq!(independent.len(), stacked.len());
            for (candidate, (a, b)) in independent.iter().zip(stacked).enumerate() {
                assert_eq!(
                    a.0.to_bits(),
                    b.0.to_bits(),
                    "domain-stack +child score moved: domain={domain_i} candidate={candidate}"
                );
                assert_eq!(
                    a.1.to_bits(),
                    b.1.to_bits(),
                    "domain-stack -child score moved: domain={domain_i} candidate={candidate}"
                );
            }
        }
        Ok(all)
    }

    /// ONE batched exception pass scoring all candidate children exactly.
    /// Returns per candidate `(bound(+1 child), bound(-1 child))`.
    fn score_candidates(
        &self,
        st: &NodeState,
        trunk_cands: &[(usize, usize)],
        head_cands: &[usize],
    ) -> Result<Vec<(f64, f64)>> {
        let nf = self.mb.nf();
        let n_y = self.mb.n_y;
        let n_t = trunk_cands.len();
        let n_h = head_cands.len();
        let total = (2 * n_t + 2 * n_h) * nf;
        let outward = self.eng.root.mode.outward();
        let mut seed = Array2::<f64>::zeros((n_y, total));
        let mut seed_e = outward.then(|| Array2::<f64>::zeros((n_y, total)));
        let mut cst = vec![0.0; total];
        let mut cst_err = vec![0.0; total];
        let mut m1 = vec![0.0; total];
        let mut exc = Exceptions::default();
        // Trunk candidate blocks: copies of the node seed + one exception.
        for (kc, &(li, pos)) in trunk_cands.iter().enumerate() {
            let rec = &self.eng.root.layers[li];
            let idx = rec.unst[pos];
            for (d_i, fix) in [(0usize, (1.0, 1.0, 0.0)), (1usize, (0.0, 0.0, 0.0))] {
                let r0 = (2 * kc + d_i) * nf;
                for j in 0..n_y {
                    for f in 0..nf {
                        seed[[j, r0 + f]] = st.ms.seed.s[[j, f]];
                        if let (Some(dst), Some(src)) = (seed_e.as_mut(), st.ms.seed.e.as_ref()) {
                            dst[[j, r0 + f]] = src[[j, f]];
                        }
                    }
                }
                for f in 0..nf {
                    cst[r0 + f] = st.ms.cst[f];
                    cst_err[r0 + f] = st.ms.cst_err[f];
                    m1[r0 + f] = st.ms.m1[f];
                    exc.by_layer.entry(li).or_default().push(Exc {
                        row: r0 + f,
                        neuron: idx,
                        a2: fix.0,
                        s2: fix.1,
                        c2: fix.2,
                    });
                }
            }
        }
        // Head candidate blocks: modified seeds (exact fixed gates).
        let base = 2 * n_t * nf;
        let mode = self.eng.root.mode;
        for (kc, &i) in head_cands.iter().enumerate() {
            for (d_i, dr) in [(0usize, 1i8), (1usize, -1i8)] {
                let r0 = base + (2 * kc + d_i) * nf;
                let mut gates2 = HeadGates {
                    alpha: st.gates.alpha.clone(),
                    s: st.gates.s.clone(),
                    c: st.gates.c.clone(),
                };
                if dr > 0 {
                    gates2.alpha[i] = 1.0;
                    gates2.s[i] = 1.0;
                    gates2.c[i] = 0.0;
                } else {
                    gates2.alpha[i] = 0.0;
                    gates2.s[i] = 0.0;
                    gates2.c[i] = 0.0;
                }
                let ms2 = margin_seed(&self.mb, &gates2, &st.ybox, mode);
                for j in 0..n_y {
                    for f in 0..nf {
                        seed[[j, r0 + f]] = ms2.seed.s[[j, f]];
                        if let (Some(dst), Some(src)) = (seed_e.as_mut(), ms2.seed.e.as_ref()) {
                            dst[[j, r0 + f]] = src[[j, f]];
                        }
                    }
                }
                for f in 0..nf {
                    cst[r0 + f] = ms2.cst[f];
                    cst_err[r0 + f] = ms2.cst_err[f];
                    // m1 for the child: dr>0 keeps m1; dr<0 zeroes z_i's upper
                    // contribution (exact; see bab_direct.py score_candidates).
                    m1[r0 + f] = if dr > 0 {
                        st.ms.m1[f]
                    } else {
                        let t = self.mb.wn[f * n_y + i] * (0.0 - st.ms.zu[i]);
                        let v = st.ms.m1[f] + t;
                        if outward {
                            next_down(v - UNIT * v.abs())
                        } else {
                            v
                        }
                    };
                }
            }
        }
        let pass = self.eng.run(
            &Seed { s: seed, e: seed_e },
            Some(&st.dom),
            super::engine::LaneDir::Lower,
            Some(&exc),
            false,
        )?;
        let low = self.eng.concretize_lower(&pass);
        let mut out = Vec::with_capacity(n_t + n_h);
        for kc in 0..(n_t + n_h) {
            let mut pair = [f64::INFINITY, f64::INFINITY];
            for (d_i, p) in pair.iter_mut().enumerate() {
                let r0 = (2 * kc + d_i) * nf;
                let mut worst = f64::INFINITY;
                for f in 0..nf {
                    if !(low[r0 + f].is_finite()
                        && cst[r0 + f].is_finite()
                        && cst_err[r0 + f].is_finite()
                        && m1[r0 + f].is_finite())
                    {
                        return Err(NyError::NumericalInstability(
                            "margin_row: non-finite child component".into(),
                        ));
                    }
                    let v = low[r0 + f] + cst[r0 + f];
                    let v = if outward {
                        next_down(next_down(v - next_up(cst_err[r0 + f])))
                    } else {
                        v
                    };
                    if !v.is_finite() {
                        return Err(NyError::NumericalInstability(
                            "margin_row: non-finite child bound".into(),
                        ));
                    }
                    worst = worst.min(v.max(m1[r0 + f]));
                }
                if !worst.is_finite() {
                    return Err(NyError::NumericalInstability(
                        "margin_row: non-finite aggregate child bound".into(),
                    ));
                }
                *p = worst;
            }
            out.push((pair[0], pair[1]));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod ledger_tests {
    use super::{closed, require_finite, Ledger, RoundMode};

    #[test]
    fn joint_nonfinite_bounds_fail_before_max_min_can_mask_them() {
        // Rust's floating max/min deliberately select the finite operand when
        // the other is NaN, so the firewall must run on every component first.
        assert_eq!(1.0f64.max(f64::NAN), 1.0);
        assert_eq!((-1.0f64).min(f64::NAN), -1.0);
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(require_finite("joint regression", &[1.0, invalid]).is_err());
            assert!(!closed(RoundMode::Outward, invalid));
            assert!(!closed(RoundMode::Parity, invalid));
        }
    }

    #[test]
    fn kraft_accepts_complete_trees() {
        for depths in [vec![0u32], vec![1, 1], vec![1, 2, 2], vec![2, 2, 2, 3, 3]] {
            let mut l = Ledger::default();
            for d in depths.iter() {
                l.leaf(*d as usize);
            }
            assert!(l.kraft_ok(), "complete tree {depths:?} rejected");
        }
    }

    #[test]
    fn kraft_rejects_missing_or_extra_leaves() {
        for depths in [
            vec![1u32],    // missing sibling
            vec![1, 2],    // missing 2
            vec![1, 1, 2], // extra leaf
            vec![0, 1],    // root plus child
            vec![],        // empty
        ] {
            let mut l = Ledger::default();
            for d in depths.iter() {
                l.leaf(*d as usize);
            }
            assert!(!l.kraft_ok(), "incomplete tree {depths:?} accepted");
        }
    }

    #[test]
    fn kraft_rejects_overflow_depth() {
        let mut l = Ledger::default();
        l.leaf(500);
        assert!(!l.kraft_ok());
    }
}

/// #margin-row-gpu-batch: the pack refactor's pin.
///
/// The batched prefill's whole safety story is "it only changes WHERE `(al,
/// au)` came from". That is only true if the y-pack's CPU tail is genuinely
/// shared, so this asserts `build_pack` is exactly `pack_from_rows` applied to
/// the lane's own `y_rows` — BIT-for-bit, on both the root gates and a
/// piece-fixed domain.
#[cfg(test)]
mod pack_from_rows_tests {
    use super::{build_pack, domain_gates, pack_from_rows, BackwardEngine};
    use crate::margin_row::net::TwinNet;
    use crate::margin_row::root::RootGates;
    use crate::margin_row::rounding::RoundMode;
    use crate::margin_row::spec::{TwinOpSpec, TwinSpec};

    /// Minimal VALID twin net: one trunk ReLU, then the head `Gemm -> Relu ->
    /// Gemm` pair the compiler requires. Mirrors `no_conv_chain_spec` at
    /// depth 1.
    #[allow(clippy::cast_precision_loss)]
    fn spec() -> TwinSpec {
        let n_in = 6usize;
        let (n_y, n_out) = (4usize, 3usize);
        let wh: Vec<f64> = (0..(n_y * n_in))
            .map(|i| ((i * 7) % 11) as f64 / 11.0 - 0.5)
            .collect();
        let wo: Vec<f64> = (0..(n_out * n_y))
            .map(|i| ((i * 5) % 13) as f64 / 13.0 - 0.5)
            .collect();
        TwinSpec {
            n_in,
            ops: vec![
                TwinOpSpec::Relu { input: 0 },    // t1, trunk relu 0
                TwinOpSpec::Flatten { input: 1 }, // t2
                TwinOpSpec::Gemm {
                    input: 2,
                    weight: wh,
                    bias: vec![0.1, -0.1, 0.05, 0.0],
                    shape: (n_y, n_in),
                }, // t3 = y
                TwinOpSpec::Relu { input: 3 },    // t4, head relu
                TwinOpSpec::Gemm {
                    input: 4,
                    weight: wo,
                    bias: vec![0.0, 0.1, -0.1],
                    shape: (n_out, n_y),
                }, // t5
            ],
        }
    }

    fn eq_rows(a: &[f64], b: &[f64], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length moved");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{what}: row {i} moved");
        }
    }

    #[test]
    fn build_pack_is_pack_from_rows_of_the_lanes_own_y_rows() {
        let spec = spec();
        let net = TwinNet::compile(&spec).expect("compiles");
        let lo = vec![-0.4; spec.n_in];
        let hi = vec![0.4; spec.n_in];
        let (root, _) =
            RootGates::build_retaining(&net, &lo, &hi, RoundMode::Outward, None, None, &[])
                .expect("root gates");
        let eng = BackwardEngine::new(&net, &root);
        let split = root
            .layers
            .iter()
            .enumerate()
            .find_map(|(li, rec)| (!rec.unst.is_empty()).then_some((li, 0usize, 1i8)));
        for dom in [None, split.map(|s| domain_gates(&root, &[s]))] {
            let dom_ref = dom.as_ref();
            let want = build_pack(&eng, dom_ref).expect("build_pack");
            let (al, au) = eng.y_rows(dom_ref).expect("y_rows");
            let got = pack_from_rows(&eng, al, au);
            eq_rows(&got.ly0, &want.ly0, "ly0");
            eq_rows(&got.uy0, &want.uy0, "uy0");
            eq_rows(
                &eng.concretize_lower(&got.al),
                &eng.concretize_lower(&want.al),
                "al",
            );
            eq_rows(
                &eng.concretize_upper(&got.au),
                &eng.concretize_upper(&want.au),
                "au",
            );
        }
    }
}
