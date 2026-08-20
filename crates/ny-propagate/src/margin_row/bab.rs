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

use ndarray::{s, Array2};
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
    /// #margin-row-col-retire (`NY_MARGIN_ROW_COL_RETIRE=1`, default OFF):
    /// retire margin columns whose certified bound crossed the closure
    /// threshold at an ancestor domain; descendants recompute only the
    /// SURVIVING columns. Cfg-carried (not read at eval) so tests can arm it
    /// without process-global env mutation.
    pub col_retire: bool,
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
            col_retire: col_retire_enabled(),
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

/// #margin-row-beta-percol: one column's inherited β state (trunk aligned
/// with the domain's `trunk`, heads with its `heads`; both resized with
/// zeros at eval exactly as the shared vectors are — a fresh split starts at
/// 0). Only columns that were among the K worst failing at some ancestor ever
/// get an entry, and all-zero entries are pruned before inheritance, so the
/// storage is O(K_active × splits), never O(nf × splits).
#[derive(Clone)]
struct PcBetas {
    /// Seed column (margin objective) this state prices.
    col: usize,
    /// Trunk-split multipliers, aligned with the domain's `trunk`.
    trunk: Vec<f64>,
    /// Head-split multipliers, aligned with the domain's `heads`.
    heads: Vec<f64>,
}

struct DomainEntry {
    bound: f64,
    seq: u64,
    trunk: Vec<TrunkSplit>,
    heads: Vec<HeadFix>,
    ly: Vec<f64>,
    uy: Vec<f64>,
    /// #margin-row-beta: split Lagrangian multipliers, aligned with `trunk`
    /// (child inherits parent's accepted β; the fresh split starts at 0).
    /// Empty/all-zero ⇒ the engine never sees a term (bit-identical passes).
    betas: Vec<f64>,
    /// #margin-row-beta C3: HEAD-split multipliers, aligned with `heads`
    /// (same inheritance contract as `betas`). Empty/all-zero ⇒ the seed is
    /// never shifted (bit-identical passes).
    head_betas: Vec<f64>,
    /// #margin-row-beta C2: per-node Polyak λ memory (kills the reset
    /// amnesia). Inherited by children exactly as `betas` is; accept ⇒
    /// ×1.5 capped, reject ⇒ ×0.5, floor `beta::LAMBDA_MIN`.
    beta_step: f64,
    /// #margin-row-beta-percol: per-column β state (see [`PcBetas`]).
    /// Empty unless `NY_MARGIN_ROW_BETA_PERCOL=1` ever accepted a column.
    pc: Vec<PcBetas>,
    /// #margin-row-col-retire: sorted ROOT-fail-set column indices whose
    /// bound crossed the closure threshold at an ANCESTOR of this domain.
    /// A certified per-column bound holds on the whole ancestor region, and
    /// every descendant region is a subset, so these columns are PROVEN here
    /// and need never be recomputed: the eval narrows its margin batch to the
    /// survivors. Inherits monotonically parent -> child (exactly like β).
    /// Always empty unless `NY_MARGIN_ROW_COL_RETIRE=1`.
    retired: Vec<u16>,
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
    /// #margin-row-beta: the β vector this node's ACCEPTED bound was scored
    /// under (children inherit it). Equals the entry's β when disarmed or when
    /// no trial improved.
    betas: Vec<f64>,
    /// #margin-row-beta C3: the HEAD-β vector the accepted bound was scored
    /// under (zeroed when the seed shift was refused — children must never
    /// inherit an unscored β).
    head_betas: Vec<f64>,
    /// #margin-row-beta C2: the node's outgoing Polyak λ memory.
    beta_step: f64,
    /// #margin-row-beta-percol: the per-column β state the accepted
    /// per-column bounds were scored under (pruned of all-zero entries;
    /// children inherit it verbatim — every entry is valid on every child
    /// region, a subset of this domain's). Column indices are ROOT-frame.
    pc: Vec<PcBetas>,
    /// #margin-row-col-retire: the OUT set for children — the entry's
    /// inherited set plus every column whose certified composite crossed the
    /// closure threshold at THIS eval (root frame, sorted). Equals the
    /// entry's set verbatim when the gate is off.
    retired: Vec<u16>,
    /// #margin-row-col-retire: the narrowed margin batch this node's `ms`,
    /// `pass` and per-column side lanes were computed under (`None` = the
    /// driver's full tree batch). Candidate selection/scoring for this node
    /// MUST use this batch — widths must agree with `ms`/`pass`.
    mb_local: Option<MarginBatch>,
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
    /// #margin-row-beta: inherited multipliers (see `DomainEntry::betas`).
    betas: Vec<f64>,
    /// #margin-row-beta C3: inherited head multipliers (see
    /// `DomainEntry::head_betas`); a head pick appends the fresh split's 0.
    head_betas: Vec<f64>,
    /// #margin-row-beta C2: inherited λ memory (see `DomainEntry::beta_step`).
    beta_step: f64,
    /// #margin-row-beta-percol: inherited per-column β (see
    /// `DomainEntry::pc`). The fresh split's zero is appended by the child's
    /// own eval resize, exactly as for the shared vectors.
    pc: Vec<PcBetas>,
    /// #margin-row-col-retire: inherited retirement set (see
    /// `DomainEntry::retired`).
    retired: Vec<u16>,
}

/// Result of one domain evaluation (see [`MarginRowBab::eval_with_pack`]).
enum EvalOut {
    /// Provably empty region: the domain is closed.
    Infeasible,
    /// #deadline-poll (D2): an intra-expansion poll fired before the next
    /// costly phase. The caller aborts CLEANLY with no verdict — never a
    /// closure, never an error — exactly as the between-expansion check
    /// would have at the next loop head, just up to one engine pass sooner.
    Deadline,
    /// Refreshed bound + branching state (boxed: `NodeState` dwarfs the other
    /// variants).
    Node(f64, Box<NodeState>),
}

/// Result of evaluating+expanding ONE popped domain in the parallel frontier.
/// Carries only data (no `&mut` state), so it is produced by a rayon worker
/// and applied deterministically by the serial merge, keeping every stats
/// delta and heap push bit-identical to what the serial lane would do for the
/// same domain.
enum ExpandStep {
    /// `eval_with_pack` returned `None` (provably empty box): closed.
    Infeasible,
    /// #deadline-poll (D2): an intra-expansion poll fired inside this
    /// domain's eval or candidate scoring. The merge stops the tree loop with
    /// a wallclock stop reason; the domain is NOT closed and NOT counted.
    DeadlineHit,
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
    /// #margin-row-col-retire: the margin width these columns were built at
    /// (the owning node's SURVIVOR count). The stacked decode must use this,
    /// not the driver's full width — stacked domains can differ.
    nf: usize,
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

/// #margin-row-adaptive-width: escalate the candidate budget when the frontier
/// shows the EXPLOSION signature.
///
/// The failure mode that costs cifar100 rows is measurable in-flight: a
/// proving row's frontier peaks around ~26 open domains and then DRAINS
/// (idx_6659: 26 -> 4), while a failing one grows without bound (idx_8600:
/// 18 -> 41 -> 156 -> 415). And the fix is measured: at k=16 that same row
/// PROVES in 186 expansions, frontier 424 -> 2. A fixed k=16 preset carries
/// transfer risk — rows the default serves fine pay double scoring cost for
/// nothing (idx_8762 proves in 9 expansions and never sees 32 open). This
/// policy spends width exactly where the signature appears:
///
///     open >= 32   ->  k = 16     (above every measured proving-row peak)
///     open >= 256  ->  k = 32
///
/// The ratchet is one-way (never narrows below the configured base), so a row
/// that never explodes runs the configured width bit-identically.
///
/// SOUNDNESS: width chooses only WHICH domains get split — every candidate is
/// scored by the same certified batched pass, and no bound computation reads
/// `k`. A wrong width costs proofs; it cannot manufacture one.
fn adaptive_width_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        match ny_levers::read(&ny_levers::decls::sound_channel::MARGIN_ROW_K_ADAPT).value {
            // Bool(_) only when the env supplied an admissible "1"/"0"; every
            // other resolution (absent, malformed) is Unset and defers to the
            // preset, preserving the reader's original three-state contract.
            ny_levers::LeverValue::Bool(forced) => forced,
            _ => super::k_adaptive_preset(),
        }
    })
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

/// #deadline-poll (D2) telemetry: an intra-expansion poll fired. One line for
/// the first few hits (parallel workers can race several polls into the same
/// instant), then silence — the stop reason carries the rest.
fn poll_note(site: &str) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 4 {
        eprintln!("[deadline-poll] intra-expansion exit at {site} (n={n})");
    }
}

fn build_pack(
    eng: &BackwardEngine<'_>,
    dom: Option<&DomainGates>,
    splits: &[TrunkSplit],
) -> Result<YPack> {
    // The identity-seeded y-row refresh is the seam's cleanest admission: the
    // seed is f32-exact and carries no certified error, so no `y_abs` is
    // needed, and one device walk publishes BOTH lanes. Dark + fail-closed:
    // with `NY_MARGIN_ROW_GPU` unset this is the established `y_rows` call.
    let (al, au) = eng.y_rows_seamed(dom, &super::gpu_seam::SeamCtx::default())?;
    let mut pack = pack_from_rows(eng, al, au);
    clip_tighten_pack(eng, splits, &mut pack);
    Ok(pack)
}

/// #clip-and-verify step 4: tighten this domain's y-box using the halfspaces its
/// own split history implies.
///
/// THE POINT. cifar100 rows fail by frontier explosion, not bound quality:
/// idx_6659 drains 26 -> 4 open domains and proves at 189 expansions, while
/// idx_8600 goes 18 -> 415 at depth 30 and times out 0.0129 from the threshold.
/// Both improve their bound at the SAME rate. Children were not closing because
/// a split fixed ONE neuron's gate and nothing else in the domain moved. This is
/// the mechanism that makes one split tighten the WHOLE domain.
///
/// SOUNDNESS, and none of it rests on the solver being optimal:
/// * every halfspace is a NECESSARY condition of its split, so the constrained
///   set OVER-approximates the true subdomain — minimising over the larger set
///   still lower-bounds it;
/// * `ybox_deltas` returns a delta that is non-negative BY CONSTRUCTION
///   (`beta = 0` is always a dual candidate and reproduces the box minimum), and
///   refuses the whole set if any delta is negative or non-finite;
/// * the commit is `YBox::intersect`, an elementwise max/min that cannot widen;
/// * the certified error and penalty terms already subtracted by `concretize`
///   bound the linear-vs-true gap independently of WHICH `x` attains the
///   minimum, so they stay valid verbatim under a shrunken feasible set.
///
/// Any refusal leaves the pack exactly as `pack_from_rows` produced it.
fn clip_tighten_pack(eng: &BackwardEngine<'_>, splits: &[TrunkSplit], pack: &mut YPack) {
    if !clip_tighten_enabled() || splits.is_empty() {
        return; // the root domain has no splits, hence no constraints
    }
    let root = eng.root;
    if !super::clip::any_rows_retained(root) {
        return;
    }
    // FAIL-CLOSED FRAME GATE. `clip_rows_frame_holds` re-concretizes every
    // retained row over the ROOT box and requires it to contain that neuron's
    // stored `[l, u]`. A row in the wrong frame (transposed, output-relative,
    // or missing the augmented bias) still produces plausible numbers, and the
    // resulting halfspace would cut away part of the true subdomain — a
    // false-`unsat` generator. Checked ONCE, and a failure disables the whole
    // path for the run rather than degrading it quietly.
    static FRAME_OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*FRAME_OK.get_or_init(|| {
        // TOLERANCE. `concretize_box` widens BOTH sides by an additive
        // certified slack (`gam*(tabs + |bl| + |bu|)`, plus the f32 error term
        // on that lane), which this reproduction does not re-derive. Measured
        // on cifar100 idx_8600: stored [-2.370757e-1, 1.126760e-2] against
        // reproduced [-2.370516e-1, 1.124350e-2] — symmetric, 2.4e-5, ~1e-4
        // relative. 1e-3 leaves an order of magnitude of headroom while still
        // being orders of magnitude tighter than any REAL frame error
        // (transposed, output-relative, or bias-dropped rows are wrong by
        // factors, not by one slack term in a symmetric direction).
        let ok = super::root::clip_rows_frame_holds(root, 0.5);
        if !ok {
            eprintln!("[clip] FRAME CHECK FAILED - clip-and-verify disabled for this run");
        }
        ok
    }) {
        return;
    }
    let hs = super::clip::halfspaces_for_splits(root, splits);
    if hs.is_empty() {
        return;
    }
    let (mid, rad) = (root.mid.as_slice(), root.rad.as_slice());
    let mut moved = 0usize;
    let mut total = 0.0f64;
    if let Some(dl) = super::clip::ybox_deltas(&pack.al.a, &hs, mid, rad, true) {
        if dl.len() == pack.ly0.len() {
            for (v, d) in pack.ly0.iter_mut().zip(&dl) {
                *v += *d; // delta >= 0 by construction: shrink-only
                if *d > 0.0 {
                    moved += 1;
                    total += *d;
                }
            }
        }
    }
    if let Some(du) = super::clip::ybox_deltas(&pack.au.a, &hs, mid, rad, false) {
        if du.len() == pack.uy0.len() {
            for (v, d) in pack.uy0.iter_mut().zip(&du) {
                *v -= *d; // delta >= 0 by construction: shrink-only
                if *d > 0.0 {
                    moved += 1;
                    total += *d;
                }
            }
        }
    }
    clip_probe(root, splits, &hs);
    clip_report(splits.len(), hs.len(), pack.ly0.len(), moved, total);
}

/// #clip-and-verify diagnostic: is each halfspace actually CUTTING the box?
///
/// A zero tightening has two very different causes, and only this tells them
/// apart: either the mechanism genuinely does not pay here, or the emitted line
/// is so loose that the whole box already satisfies it (`hi <= 0`), in which
/// case the dual returns `beta = 0` correctly and the measurement says nothing
/// about Clip-and-Verify at all. Printed for the first few calls only.
fn clip_probe(root: &RootGates, splits: &[TrunkSplit], hs: &[super::clip::Halfspace]) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    if N.fetch_add(1, Ordering::Relaxed) >= 3 {
        return;
    }
    let (mid, rad) = (root.mid.as_slice(), root.rad.as_slice());
    for (k, h) in hs.iter().enumerate() {
        let (lo, hi) = super::clip::halfspace_range(h, mid, rad);
        let (li, pos, sign) = splits[k.min(splits.len() - 1)];
        let nb = root
            .layers
            .get(li)
            .and_then(|l| l.unst.get(pos).map(|&i| (l.l[i], l.u[i])));
        eprintln!(
            "[clip-probe] hs={k} layer={li} pos={pos} sign={sign} range=[{lo:.6e},{hi:.6e}] \
neuron={nb:?} cuts={}",
            hi > 0.0
        );
    }
}

/// #clip-and-verify engagement telemetry. A null result from a lever that never
/// fired is VACUOUS, so this must be non-empty before any negative measurement
/// is believed: it prints the first few tightenings and then every 64th.
fn clip_report(splits: usize, hs: usize, rows: usize, moved: usize, total: f64) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    static MOVED: AtomicUsize = AtomicUsize::new(0);
    let n = CALLS.fetch_add(1, Ordering::Relaxed);
    MOVED.fetch_add(moved, Ordering::Relaxed);
    if n < 4 || n.is_multiple_of(64) {
        eprintln!(
            "[clip] call={n} splits={splits} halfspaces={hs} rows={rows} moved={moved} \
sum_delta={total:.6e} moved_total={}",
            MOVED.load(Ordering::Relaxed)
        );
    }
}

/// #clip-and-verify: default OFF. Env-only while it is being measured; if it
/// pays it becomes a typed preset key, because `run_instance.sh` exports exactly
/// one `NY_*` and an env setting cannot fire in competition.
fn clip_tighten_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        ny_levers::read(&ny_levers::decls::sound_channel::MARGIN_ROW_CLIP)
            .value
            .as_bool()
    })
}

/// #margin-row-col-retire (`NY_MARGIN_ROW_COL_RETIRE=1`, default OFF): retire
/// margin columns whose bound crossed the closure threshold at some ancestor,
/// so descendant evals/scorings pay only for the SURVIVING columns. This env
/// read seeds `BabConfig::default().col_retire`; the driver consults only the
/// cfg (tests arm the cfg directly, no process-global mutation).
///
/// Env-only while it is being measured (same contract as `NY_MARGIN_ROW_BETA`);
/// if it pays it ships as a typed preset key.
fn col_retire_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("NY_MARGIN_ROW_COL_RETIRE").ok().as_deref() == Some("1");
        if on {
            eprintln!("[col-retire] armed (NY_MARGIN_ROW_COL_RETIRE=1)");
        }
        on
    })
}

/// #margin-row-col-retire engagement telemetry (R9): `newly=` columns retired
/// by THIS eval, `retired=` the child-inherited set size, `survivors=` the
/// columns descendants will still pay for. Rate-limited like `[beta]`.
fn col_retire_report(depth: usize, newly: usize, retired: usize, survivors: usize) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    static TOTAL: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    TOTAL.fetch_add(newly, Ordering::Relaxed);
    if n < 8 || n.is_multiple_of(64) {
        eprintln!(
            "[col-retire] call={n} depth={depth} newly={newly} retired={retired} \
survivors={survivors} newly_total={}",
            TOTAL.load(Ordering::Relaxed)
        );
    }
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
    let pack = Arc::new(build_pack(eng, None, &[])?);
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
            betas: Vec::new(),
            head_betas: Vec::new(),
            beta_step: super::beta::lambda(),
            pc: Vec::new(),
            retired: Vec::new(),
        }
    }

    fn tree_loop(&mut self, root_bound: f64, re: &RootEval) -> Result<MarginRowOutcome> {
        // #margin-row-root-bound: announce where the search STARTS, before it can
        // time out.
        //
        // `root_bound` is the quantity that distinguishes a row this lane proves
        // from one it does not: idx_6659 proves from -0.4198 in 189 expansions,
        // while idx_8600 fails at 729 expansions AND at a 400 s budget (~3000),
        // so neither search volume nor split quality is its problem — narrowing
        // candidates actually LOSES proofs (k=8 proves, k=4 does 5x the search
        // and fails).
        //
        // Until now this number only appeared inside the completion summary, so
        // exactly the runs that need explaining — the timeouts — never printed
        // it. That is the wrong way round for a diagnostic. Emitting at entry
        // costs one line per lane invocation.
        //
        // NOTE this is the MARGIN-ROW lane's own root bound, not the internal
        // verifier's multi-objective census. The census is a different lane's
        // number, reads 85/99 on idx_8600, and does NOT predict verdicts
        // (idx_6659 proves at 1/99; idx_8762 proves at 0/99).
        eprintln!(
            "[margin-row-root] tree_loop entry root_bound={root_bound:.6} \
             (this lane's own bound; NOT the multi-objective census)"
        );
        let t0 = Instant::now();
        let mode = self.eng.root.mode;
        let mut heap: BinaryHeap<DomainEntry> = BinaryHeap::new();
        let mut seq: u64 = 0;
        heap.push(self.root_entry(root_bound, re, seq));
        seq += 1;
        let mut stop = "queue_empty".to_string();
        let trace = std::env::var("NY_MARGIN_ROW_TRACE").is_ok();
        while let Some(entry) = {
            self.adapt_width(heap.len() + 1);
            heap.pop()
        } {
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
            let (b_eval, st) = match self.eval_node(&entry)? {
                EvalOut::Infeasible => {
                    // Infeasible domain: provably empty, closed.
                    self.stats.closed += 1;
                    self.ledger.leaf(depth);
                    continue;
                }
                EvalOut::Deadline => {
                    // #deadline-poll (D2): clean early exit, no verdict.
                    stop = "wallclock_intra_expansion".into();
                    break;
                }
                EvalOut::Node(b_eval, st) => (b_eval, *st),
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
                match self.score_candidates(&st, &trunk_cands, &head_cands)? {
                    Some(ch) => ch,
                    None => {
                        // #deadline-poll (D2): clean early exit, no verdict.
                        stop = "wallclock_intra_expansion".into();
                        break;
                    }
                }
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
                let (trunk_c, heads_c, ly_c, uy_c, betas_c, head_betas_c) = if pick < n_t {
                    super::prof::bump(super::prof::Counter::TrunkSplit, 1);
                    let (li, pos) = trunk_cands[pick];
                    let mut tc = entry.trunk.clone();
                    tc.push((li, pos, dr));
                    // #margin-row-beta: inherit the parent's accepted β (valid
                    // on the child's smaller region); the fresh split starts
                    // at 0 (its term vanishes). Head β is inherited verbatim.
                    let mut bc = st.betas.clone();
                    bc.resize(entry.trunk.len(), 0.0);
                    bc.push(0.0);
                    let mut hb = st.head_betas.clone();
                    hb.resize(entry.heads.len(), 0.0);
                    (
                        tc,
                        entry.heads.clone(),
                        st.ybox.ly.clone(),
                        st.ybox.uy.clone(),
                        bc,
                        hb,
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
                    // #margin-row-beta C3: the fresh HEAD split starts at 0.
                    let mut hb = st.head_betas.clone();
                    hb.resize(entry.heads.len(), 0.0);
                    hb.push(0.0);
                    (entry.trunk.clone(), hc, ly, uy, st.betas.clone(), hb)
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
                        betas: betas_c,
                        head_betas: head_betas_c,
                        beta_step: st.beta_step,
                        // #margin-row-beta-percol: inherited verbatim; the
                        // child's own eval resizes each column's vectors
                        // (fresh split ⇒ appended 0), exactly like `betas`.
                        pc: st.pc.clone(),
                        // #margin-row-col-retire: monotone inheritance.
                        retired: st.retired.clone(),
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
    /// See [`adaptive_width_enabled`]. Called once per batch from the serial
    /// section; a no-op unless armed.
    fn adapt_width(&mut self, open: usize) {
        if !adaptive_width_enabled() {
            return;
        }
        let (base_h, base_t) = (self.cfg.k_head, self.cfg.k_trunk);
        let esc: usize = if open >= 256 {
            32
        } else if open >= 32 {
            16
        } else {
            0
        };
        let (nh, nt) = (base_h.max(esc), base_t.max(esc));
        if (nh, nt) != (base_h, base_t) {
            eprintln!(
                "[k-adapt] frontier open={open} -> k_head {base_h}->{nh} k_trunk {base_t}->{nt}                  at exp {}",
                self.stats.expansions
            );
            self.cfg.k_head = nh;
            self.cfg.k_trunk = nt;
        }
    }

    fn tree_loop_parallel(&mut self, root_bound: f64, re: &RootEval) -> Result<MarginRowOutcome> {
        // #margin-row-root-bound: announce where the search STARTS, before it can
        // time out.
        //
        // `root_bound` is the quantity that distinguishes a row this lane proves
        // from one it does not: idx_6659 proves from -0.4198 in 189 expansions,
        // while idx_8600 fails at 729 expansions AND at a 400 s budget (~3000),
        // so neither search volume nor split quality is its problem — narrowing
        // candidates actually LOSES proofs (k=8 proves, k=4 does 5x the search
        // and fails).
        //
        // Until now this number only appeared inside the completion summary, so
        // exactly the runs that need explaining — the timeouts — never printed
        // it. That is the wrong way round for a diagnostic. Emitting at entry
        // costs one line per lane invocation.
        //
        // NOTE this is the MARGIN-ROW lane's own root bound, not the internal
        // verifier's multi-objective census. The census is a different lane's
        // number, reads 85/99 on idx_8600, and does NOT predict verdicts
        // (idx_6659 proves at 1/99; idx_8762 proves at 0/99).
        eprintln!(
            "[margin-row-root] tree_loop entry root_bound={root_bound:.6} \
             (this lane's own bound; NOT the multi-objective census)"
        );
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
            self.adapt_width(heap.len());
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
                            let pack = Arc::new(build_pack(eng, Some(&dom), &trunk)?);
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
                    ExpandStep::DeadlineHit => {
                        // #deadline-poll (D2): clean early exit, no verdict.
                        // The domain is neither closed nor counted; results
                        // merged before this one stand, the rest are dropped
                        // (the end-of-run deadline guard would discard any
                        // late verdict anyway).
                        stop = "wallclock_intra_expansion".into();
                        break 'outer;
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
                                betas: c.betas,
                                head_betas: c.head_betas,
                                beta_step: c.beta_step,
                                pc: c.pc,
                                retired: c.retired,
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
        // #deadline-poll (D2): a poll inside the (single) scoring dispatch
        // turns every not-yet-scored Ready slot into a clean DeadlineHit.
        let mut score_deadline = false;
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
            match stacked {
                Some(stacked) => {
                    for (&i, score) in ready_idx.iter().zip(stacked) {
                        scores[i] = Some(score);
                    }
                }
                None => score_deadline = true,
            }
        } else if let Some(&i) = ready_idx.first() {
            let PreparedExpand::Ready(ready) = &preps[i] else {
                unreachable!("ready index")
            };
            let score = {
                let _t = super::prof::Timer::start(super::prof::Phase::ScoreCands);
                self.score_candidates(&ready.st, &ready.trunk_cands, &ready.head_cands)?
            };
            match score {
                Some(score) => scores[i] = Some(score),
                None => score_deadline = true,
            }
        }

        let mut out = Vec::with_capacity(preps.len());
        for (i, (prep, entry)) in preps.into_iter().zip(batch).enumerate() {
            match prep {
                PreparedExpand::Done(step) => out.push(step),
                PreparedExpand::Ready(ready) => match scores[i].take() {
                    Some(score) => out.push(self.finish_prepared_expand(entry, &ready, &score)),
                    None if score_deadline => out.push(ExpandStep::DeadlineHit),
                    None => {
                        return Err(NyError::InvalidSpec(
                            "margin_row: missing domain-stack candidate score".into(),
                        ))
                    }
                },
            }
        }
        Ok(out)
    }

    fn prepare_expand(&self, pack: &Arc<YPack>, entry: &DomainEntry) -> Result<PreparedExpand> {
        let mode = self.eng.root.mode;
        let (b_eval, st) = match self.eval_with_pack(pack, entry)? {
            EvalOut::Infeasible => return Ok(PreparedExpand::Done(ExpandStep::Infeasible)),
            EvalOut::Deadline => return Ok(PreparedExpand::Done(ExpandStep::DeadlineHit)),
            EvalOut::Node(b_eval, st) => (b_eval, *st),
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
            let (trunk_c, heads_c, ly_c, uy_c, betas_c, head_betas_c) = if pick < n_t {
                super::prof::bump(super::prof::Counter::TrunkSplit, 1);
                let (li, pos) = ready.trunk_cands[pick];
                let mut tc = entry.trunk.clone();
                tc.push((li, pos, dr));
                // #margin-row-beta: parent's accepted β + 0 for the new split;
                // head β inherited verbatim.
                let mut bc = ready.st.betas.clone();
                bc.resize(entry.trunk.len(), 0.0);
                bc.push(0.0);
                let mut hb = ready.st.head_betas.clone();
                hb.resize(entry.heads.len(), 0.0);
                (
                    tc,
                    entry.heads.clone(),
                    ready.st.ybox.ly.clone(),
                    ready.st.ybox.uy.clone(),
                    bc,
                    hb,
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
                // #margin-row-beta C3: the fresh HEAD split starts at 0.
                let mut hb = ready.st.head_betas.clone();
                hb.resize(entry.heads.len(), 0.0);
                hb.push(0.0);
                (entry.trunk.clone(), hc, ly, uy, ready.st.betas.clone(), hb)
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
                    betas: betas_c,
                    head_betas: head_betas_c,
                    beta_step: ready.st.beta_step,
                    // #margin-row-beta-percol: inherited verbatim (resized at
                    // the child's own eval, exactly like `betas`).
                    pc: ready.st.pc.clone(),
                    // #margin-row-col-retire: monotone inheritance.
                    retired: ready.st.retired.clone(),
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
        let (b_eval, st) = match self.eval_with_pack(pack, entry)? {
            EvalOut::Infeasible => return Ok(ExpandStep::Infeasible),
            EvalOut::Deadline => return Ok(ExpandStep::DeadlineHit),
            EvalOut::Node(b_eval, st) => (b_eval, *st),
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
            match self.score_candidates(&st, &trunk_cands, &head_cands)? {
                Some(ch) => ch,
                None => return Ok(ExpandStep::DeadlineHit),
            }
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
            let (trunk_c, heads_c, ly_c, uy_c, betas_c, head_betas_c) = if pick < n_t {
                super::prof::bump(super::prof::Counter::TrunkSplit, 1);
                let (li, pos) = trunk_cands[pick];
                let mut tc = entry.trunk.clone();
                tc.push((li, pos, dr));
                // #margin-row-beta: parent's accepted β + 0 for the new split;
                // head β inherited verbatim.
                let mut bc = st.betas.clone();
                bc.resize(entry.trunk.len(), 0.0);
                bc.push(0.0);
                let mut hb = st.head_betas.clone();
                hb.resize(entry.heads.len(), 0.0);
                (
                    tc,
                    entry.heads.clone(),
                    st.ybox.ly.clone(),
                    st.ybox.uy.clone(),
                    bc,
                    hb,
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
                // #margin-row-beta C3: the fresh HEAD split starts at 0.
                let mut hb = st.head_betas.clone();
                hb.resize(entry.heads.len(), 0.0);
                hb.push(0.0);
                (entry.trunk.clone(), hc, ly, uy, st.betas.clone(), hb)
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
                    betas: betas_c,
                    head_betas: head_betas_c,
                    beta_step: st.beta_step,
                    // #margin-row-beta-percol: inherited verbatim (resized at
                    // the child's own eval, exactly like `betas`).
                    pc: st.pc.clone(),
                    // #margin-row-col-retire: monotone inheritance.
                    retired: st.retired.clone(),
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
            betas: Vec::new(),
            head_betas: Vec::new(),
            beta_step: super::beta::lambda(),
            pc: Vec::new(),
            retired: Vec::new(),
        };
        let mut out = Vec::new();
        for _ in 0..steps {
            // (a) serial oracle bound (populates/reads the shared cache).
            let EvalOut::Node(b_ser, _st) = bab.eval_node(&entry).expect("eval_node") else {
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
                    match bab_ref
                        .eval_with_pack(&pack, entry_ref)
                        .expect("eval_with_pack")
                    {
                        EvalOut::Node(b, _) => b.to_bits(),
                        _ => 0xDEAD_BEEF,
                    }
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
                        betas: c.betas,
                        head_betas: c.head_betas,
                        beta_step: c.beta_step,
                        pc: c.pc,
                        retired: c.retired,
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
        let pack = Arc::new(build_pack(&self.eng, Some(&dom), trunk)?);
        self.lru.put(trunk.to_vec(), pack.clone());
        Ok(pack)
    }

    /// Refreshed bound + branching state; `Infeasible` = provably empty,
    /// `Deadline` = intra-expansion poll fired (D2).
    /// Serial path: fetch (or build) the domain's y-pack, then evaluate.
    fn eval_node(&mut self, entry: &DomainEntry) -> Result<EvalOut> {
        let pack = self.rows_for(&entry.trunk)?;
        self.eval_with_pack(&pack, entry)
    }

    /// #deadline-poll (D2): has the lane deadline passed? Polled at
    /// intra-expansion phase boundaries (each boundary guards a ~0.5–0.6 s
    /// certified pass), bounding the lane's deadline overshoot to ONE engine
    /// pass instead of one full expansion (seconds on cifar-med, minutes on
    /// resnet_large). SOUND by construction: a firing poll can only cause an
    /// EARLIER clean exit (Unknown) — it never closes a domain, never skips a
    /// verdict-bearing computation mid-way, and the existing end-of-run guard
    /// already discards any verdict that lands past the deadline.
    #[inline]
    fn deadline_hit(&self) -> bool {
        self.cfg.deadline.is_some_and(|dl| Instant::now() > dl)
    }

    /// Pure domain evaluation given a PREBUILT y-pack (no cache, no `&mut`):
    /// the parallel-frontier lane reuses this across rayon workers. Every
    /// bound here is a deterministic function of `(pack, entry)`, so a domain's
    /// result is BIT-IDENTICAL whether evaluated serially or in a worker.
    fn eval_with_pack(&self, pack: &Arc<YPack>, entry: &DomainEntry) -> Result<EvalOut> {
        // #deadline-poll (D2): guard the domain-gate rebuild + the main
        // certified pass about to be spent.
        if self.deadline_hit() {
            poll_note("eval_node");
            return Ok(EvalOut::Deadline);
        }
        let mode = self.eng.root.mode;
        let _t = super::prof::Timer::start(super::prof::Phase::EvalNode);
        let dom = domain_gates(self.eng.root, &entry.trunk);
        if dom.infeasible {
            // #clip-and-verify VERIFY: the split set's halfspaces make this
            // domain's region empty. Discharged without paying the ~600 ms
            // backward pass — the exact "child that never closes" converted
            // into an instant close.
            return Ok(EvalOut::Infeasible);
        }
        let pack = pack.clone();
        let mut ybox = YBox {
            ly: pack.ly0.clone(),
            uy: pack.uy0.clone(),
        };
        ybox.clamp(&entry.heads);
        if ybox.is_empty() {
            return Ok(EvalOut::Infeasible);
        }
        ybox.intersect(&entry.ly, &entry.uy);
        if ybox.is_empty() {
            return Ok(EvalOut::Infeasible);
        }
        let gates = head_gates(&ybox, mode);
        // #margin-row-col-retire: narrow the margin batch to the SURVIVING
        // columns. Every retired index carries a certified `closed()` bound
        // from an ANCESTOR region — a superset of this domain — so dropping
        // the row cannot lose an obligation: the verdict below becomes min
        // over survivors, and the retired columns are proven-by-ancestor.
        // `survivors` maps local column index -> root-frame index (`None` =
        // identity). Degenerate or malformed inherited sets fall back to full
        // width, which is always sound (it only recomputes proven columns).
        //
        // The y-pack is untouched: it is the IDENTITY-seeded y-row pack
        // (n_y-frame, no margin columns), so the LRU stays keyed by trunk
        // splits alone and packs are shared across any retirement state.
        let retire_on = self.cfg.col_retire;
        let mut mb_local: Option<MarginBatch> = None;
        let mut survivors: Option<Vec<usize>> = None;
        if retire_on && !entry.retired.is_empty() {
            let nf_root = self.mb.nf();
            let keep: Vec<usize> = (0..nf_root)
                .filter(|&c| !u16::try_from(c).is_ok_and(|c16| entry.retired.contains(&c16)))
                .collect();
            // A pushed domain always keeps >= 1 failing column (an all-closed
            // eval closes the parent instead of pushing children).
            if !keep.is_empty() && keep.len() < nf_root {
                if let Ok(sub) = self.mb.subset(&keep) {
                    mb_local = Some(sub);
                    survivors = Some(keep);
                }
            }
        }
        let mb: &MarginBatch = mb_local.as_ref().unwrap_or(&self.mb);
        let nf = mb.nf();
        let ms = margin_seed(mb, &gates, &ybox, mode);
        // #margin-row-beta (NY_MARGIN_ROW_BETA=1, default OFF): attach the
        // entry's inherited split Lagrangians before the pass. Disarmed, the
        // beta map stays empty, the seed is the untouched base margin seed,
        // and the engine's application site is never entered — the pass is
        // bit-identical to the pre-beta lane.
        //
        // Gate (C3): trunk OR head splits — head splits carry β via the seed
        // shift, so β now engages from depth 1 instead of waiting for the
        // first trunk split (measured: β used to engage only from depth 4–5,
        // leaving ~half of ancestral constraints unweighted).
        let beta_armed =
            super::beta::enabled() && (!entry.trunk.is_empty() || !entry.heads.is_empty());
        let mut betas = entry.betas.clone();
        betas.resize(entry.trunk.len(), 0.0);
        let mut head_betas = entry.head_betas.clone();
        head_betas.resize(entry.heads.len(), 0.0);
        // C2: per-node Polyak λ memory, inherited like `betas`; guarded so a
        // zero/non-finite inherited value can never wedge the schedule.
        let mut beta_step = entry.beta_step;
        if !(beta_step.is_finite() && beta_step > 0.0) {
            beta_step = super::beta::lambda();
        }
        // #margin-row-beta-percol: inherited per-column β, resized to the
        // current split lists exactly as the shared vectors above (the fresh
        // split starts at 0); out-of-range columns dropped defensively. Empty
        // unless NY_MARGIN_ROW_BETA_PERCOL=1 — the shared-vector lane is then
        // bit-identical (no pc terms are ever installed).
        let percol = beta_armed && super::beta::percol();
        // #margin-row-col-retire: entry `pc` columns are stored ROOT-frame
        // (stable as the survivor set shrinks down the tree); the ascent below
        // runs entirely in the LOCAL frame — the seed/pass columns the engine
        // prices (`beta_pc` columns index seed columns). A retired column's β
        // state is dropped: the column is proven, nothing prices it again.
        // Identity when nothing is retired (the pre-retirement behavior).
        let root_to_local = |root_c: usize| -> Option<usize> {
            match &survivors {
                Some(map) => map.binary_search(&root_c).ok(),
                None => (root_c < nf).then_some(root_c),
            }
        };
        let mut pc: Vec<PcBetas> = if percol {
            entry
                .pc
                .iter()
                .filter_map(|s| {
                    let col = root_to_local(s.col)?;
                    let mut t = s.trunk.clone();
                    t.resize(entry.trunk.len(), 0.0);
                    let mut h = s.heads.clone();
                    h.resize(entry.heads.len(), 0.0);
                    Some(PcBetas {
                        col,
                        trunk: t,
                        heads: h,
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        // C3 admissibility of the head seed shift: the shift charges its one
        // f64 add per entry into the seed's error lane, so OUTWARD mode
        // requires `ms.seed.e` — present by construction (bounds.rs::
        // margin_seed builds it exactly when `mode.outward()`); refused
        // defensively if that ever drifts. In parity mode (`e == None`) the
        // whole lane carries no certified error (the trunk-β site
        // `apply_beta_terms` also skips its charge there), so the shift is
        // applied charge-free; NOTE the β path does NOT refuse parity mode
        // upstream — parity is Python-parity testing only, never a
        // competition lane. See beta::seed_with_head_terms.
        let heads_requested = beta_armed && super::beta::heads_on() && !entry.heads.is_empty();
        let head_admissible = heads_requested && (ms.seed.e.is_some() || !mode.outward());
        if heads_requested && !head_admissible {
            super::beta::refuse("head-seed-error-lane-missing-outward");
            head_betas.iter_mut().for_each(|hb| *hb = 0.0);
            // Per-column head β share the admissibility gate (and the rule:
            // never inherit a β that was not scored).
            for s in &mut pc {
                s.heads.iter_mut().for_each(|hb| *hb = 0.0);
            }
        }
        let mut dom = dom;
        if beta_armed {
            super::beta::set_terms(&mut dom, self.eng.root, &entry.trunk, &betas);
        }
        // #margin-row-beta-percol: install the inherited per-column trunk
        // terms so the MAIN pass (and per_j) already reflects them — the same
        // contract as the shared terms above.
        if percol && !pc.is_empty() {
            let cols: Vec<(usize, &[f64])> =
                pc.iter().map(|s| (s.col, s.trunk.as_slice())).collect();
            super::beta::set_terms_pc(&mut dom, self.eng.root, &entry.trunk, &cols);
        }
        // Effective seed for the main pass: base, or the head-β-shifted
        // clone (weak duality one layer higher; the certified pass over the
        // shifted seed is a valid lower bound of f on this domain's region).
        let seed_main = if head_admissible && head_betas.iter().any(|&hb| hb > 0.0) {
            let shifted = super::beta::seed_with_head_terms(
                &ms.seed,
                &entry.heads,
                &head_betas,
                mode.outward(),
            );
            if shifted.is_none() {
                // Refused: run on the base seed and zero the inherited head
                // β — children must never inherit a β that was not scored.
                super::beta::refuse("head-seed-shift-main");
                head_betas.iter_mut().for_each(|hb| *hb = 0.0);
            }
            shifted
        } else {
            None
        };
        // #margin-row-beta-percol: per-column head shifts, chained AFTER the
        // shared shift (the effective multiplier per (head, col) is the sum
        // shared + own — a sum of valid `β >= 0` multipliers; each add is
        // separately charged). `seed_main` stays the shared-only base so the
        // ascent's trials can re-overlay candidate pc shifts on it.
        let seed_main_pc =
            if percol && head_admissible && pc.iter().any(|s| s.heads.iter().any(|&hb| hb > 0.0)) {
                let cols: Vec<(usize, &[f64])> =
                    pc.iter().map(|s| (s.col, s.heads.as_slice())).collect();
                let shifted = super::beta::seed_with_head_terms_pc(
                    seed_main.as_ref().unwrap_or(&ms.seed),
                    &entry.heads,
                    &cols,
                    mode.outward(),
                );
                if shifted.is_none() {
                    // Refused: run without the pc shift and zero the pc head β —
                    // children must never inherit a β that was not scored.
                    super::beta::refuse("head-seed-shift-pc-main");
                    for s in &mut pc {
                        s.heads.iter_mut().for_each(|hb| *hb = 0.0);
                    }
                }
                shifted
            } else {
                None
            };
        let collect = Collect {
            unst_abs: true,
            rows: if self.cfg.tier0_exact > 0 {
                self.cfg.retained.as_deref()
            } else {
                None
            },
            // Gradient capture for the beta proposal only; nothing
            // verdict-bearing reads it.
            unst_rows: beta_armed,
        };
        let mut pass = self.eng.run_collect(
            seed_main_pc
                .as_ref()
                .or(seed_main.as_ref())
                .unwrap_or(&ms.seed),
            Some(&dom),
            super::engine::LaneDir::Lower,
            None,
            collect,
        )?;
        let per_j = per_class_direct(&self.eng, &pass, &ms, 0..nf);
        let m2v = compose_viay(
            &self.eng,
            mb,
            &gates,
            &pack.al,
            &pack.au,
            &pack.al_dots,
            &pack.au_dots,
            mode,
        );
        // #margin-row-col-retire: certified per-column direct values backing
        // the retirement decision below. Accepted β trials refresh it — every
        // accepted `per_t[c]` is a valid column bound by per-column weak
        // duality, exactly the percol arm's acceptance argument.
        let mut per_track = retire_on.then(|| per_j.clone());
        let mut b = f64::INFINITY;
        let mut worst_col = 0usize;
        for r0 in 0..nf {
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
            if pj < b {
                worst_col = r0;
            }
            b = b.min(pj);
        }
        if !b.is_finite() {
            return Err(NyError::NumericalInstability(
                "margin_row: non-finite node bound".into(),
            ));
        }
        // #margin-row-beta ascent: propose beta from the pass's own gradient,
        // re-score with the UNCHANGED certified pass, monotone accept. Every
        // trial bound is valid for any beta >= 0 (weak duality), so taking the
        // max is sound; a wasted trial costs one pass, never a verdict.
        //
        // C1 (Polyak, default): step sized to close the DIRECT-PATH gap of
        // the worst column in one move. C2 (live gradient): trials keep the
        // gradient capture ON, and an ACCEPTED trial refreshes x*, the worst
        // column, the gap and the direction — plus per-node λ memory in
        // `beta_step` (accept ×1.5 capped, reject ×0.5, floor LAMBDA_MIN).
        // `NY_MARGIN_ROW_BETA_POLYAK=0` restores the legacy η sign-step
        // (reset each eval) so the A/B lives in one binary.
        if beta_armed && !closed(mode, b) && percol {
            // ---- #margin-row-beta-percol ascent ----
            // Each of the K worst failing columns steps ITS OWN β from ITS
            // OWN gap and supergradient; ONE certified pass per iteration
            // scores every candidate simultaneously (per_t is per-column).
            // Acceptance is MONOTONE PER COLUMN: per_best[c] is a running max
            // of certified values for column c, each valid by per-column weak
            // duality under the β pack in force at its trial — independent of
            // what any OTHER column's candidate did in the same pack. The
            // domain bound is then min_c max(per_best[c], m1[c], m2v[c]), a
            // min of valid column bounds. Cost per iteration: K single-column
            // linearized walks (ms each) + one certified pass (the dominant
            // term, unchanged from the shared arm).
            let heads_dir: &[HeadFix] = if head_admissible { &entry.heads } else { &[] };
            if !(entry.trunk.is_empty() && heads_dir.is_empty()) {
                let b0 = b;
                let mut per_best = per_j.clone();
                let comp = |pb: &[f64], c: usize| pb[c].max(ms.m1[c]).max(m2v[c]);
                let mut active: Vec<usize> = (0..nf)
                    .filter(|&c| !closed(mode, comp(&per_best, c)))
                    .collect();
                active.sort_by(|&x, &y| {
                    comp(&per_best, x)
                        .total_cmp(&comp(&per_best, y))
                        .then(x.cmp(&y))
                });
                active.truncate(super::beta::beta_cols());
                let cols0 = active.len();
                for &c in &active {
                    if !pc.iter().any(|s| s.col == c) {
                        pc.push(PcBetas {
                            col: c,
                            trunk: vec![0.0; entry.trunk.len()],
                            heads: vec![0.0; entry.heads.len()],
                        });
                    }
                }
                let mut lam = beta_step;
                let mut t_last = 0.0f64;
                let mut acc_events = 0usize;
                let mut closed_cols = 0usize;
                let n_t = entry.trunk.len();
                'ascent: for _ in 0..super::beta::iters() {
                    if self.cfg.deadline.is_some_and(|dl| Instant::now() > dl) {
                        break;
                    }
                    if active.is_empty() {
                        break;
                    }
                    // Per-column candidates. The walk is per column by
                    // construction: x* and the frozen-sign line selections
                    // both depend on the column (see alpha_opt), so K walks
                    // are needed — batching them into one matrix walk is a
                    // future optimization, not a correctness need.
                    let mut cands: Vec<(usize, Vec<f64>)> = Vec::new();
                    let mut dropped: Vec<usize> = Vec::new();
                    for &c in &active {
                        let si = pc.iter().position(|s| s.col == c).expect("slot exists");
                        // TARGET PAST THE THRESHOLD, not onto it. Closure is
                        // STRICT (`b > 0`, closed()), but a zero-target Polyak
                        // step converges TO zero and stalls infinitesimally
                        // under it — MEASURED: pc_PC2132 landed at -2.5e-5
                        // with step 1.7e-12 and the domain stayed open. The
                        // epsilon overshoot costs nothing (beta stays >= 0,
                        // the certified pass still scores) and turns
                        // threshold-graze into closure.
                        let gap = super::beta::CLOSE_EPS - per_best[c];
                        if !(gap.is_finite() && gap > 0.0) {
                            // Direct path no longer binding for this column
                            // (its composite is carried by m1/m2v).
                            dropped.push(c);
                            continue;
                        }
                        let Some(dir) = super::beta::step_scales(
                            self.eng.net,
                            self.eng.root,
                            &dom,
                            &entry.trunk,
                            heads_dir,
                            &ms.seed,
                            &pass,
                            c,
                        ) else {
                            super::beta::refuse("step-scales-pc");
                            dropped.push(c);
                            continue;
                        };
                        let slot = &pc[si];
                        // Concatenation trunk-then-heads, mirroring the
                        // direction's shape exactly as the shared arm does.
                        let cat_b: Vec<f64> = if dir.heads.is_empty() {
                            slot.trunk.clone()
                        } else {
                            slot.trunk
                                .iter()
                                .chain(slot.heads.iter())
                                .copied()
                                .collect()
                        };
                        let mut cat_d: Vec<(f64, f64)> = Vec::with_capacity(cat_b.len());
                        cat_d.extend_from_slice(&dir.trunk);
                        cat_d.extend_from_slice(&dir.heads);
                        match super::beta::polyak_step(&cat_b, &cat_d, gap, lam) {
                            Some((v, t)) => {
                                t_last = t;
                                cands.push((si, v));
                            }
                            None => {
                                // Structural refusal (S <= 0 for this
                                // column): re-walking next iteration cannot
                                // change it under the same pass.
                                super::beta::refuse("polyak-step-pc");
                                dropped.push(c);
                            }
                        }
                    }
                    active.retain(|c| !dropped.contains(c));
                    if cands.is_empty() {
                        break;
                    }
                    // Install every candidate over the frozen slots; ONE
                    // certified pass scores them all.
                    let mut trial_pc = pc.clone();
                    for (si, cat) in &cands {
                        let (t_part, h_part) = cat.split_at(n_t);
                        trial_pc[*si].trunk = t_part.to_vec();
                        if !h_part.is_empty() {
                            trial_pc[*si].heads = h_part.to_vec();
                        }
                    }
                    let cols_t: Vec<(usize, &[f64])> = trial_pc
                        .iter()
                        .map(|s| (s.col, s.trunk.as_slice()))
                        .collect();
                    super::beta::set_terms_pc(&mut dom, self.eng.root, &entry.trunk, &cols_t);
                    let seed_trial = if head_admissible
                        && trial_pc.iter().any(|s| s.heads.iter().any(|&hb| hb > 0.0))
                    {
                        let cols_h: Vec<(usize, &[f64])> = trial_pc
                            .iter()
                            .map(|s| (s.col, s.heads.as_slice()))
                            .collect();
                        match super::beta::seed_with_head_terms_pc(
                            seed_main.as_ref().unwrap_or(&ms.seed),
                            &entry.heads,
                            &cols_h,
                            mode.outward(),
                        ) {
                            Some(s) => Some(s),
                            None => {
                                super::beta::refuse("head-seed-shift-pc-trial");
                                break 'ascent;
                            }
                        }
                    } else {
                        None
                    };
                    let trial = self.eng.run_collect(
                        seed_trial
                            .as_ref()
                            .or(seed_main.as_ref())
                            .unwrap_or(&ms.seed),
                        Some(&dom),
                        super::engine::LaneDir::Lower,
                        None,
                        collect,
                    );
                    let Ok(mut trial) = trial else { break };
                    let per_t = per_class_direct(&self.eng, &trial, &ms, 0..nf);
                    let mut any_accept = false;
                    for (si, _) in &cands {
                        let c = trial_pc[*si].col;
                        let v = per_t.get(c).copied().unwrap_or(f64::NAN);
                        if v.is_finite() && v > per_best[c] {
                            per_best[c] = v;
                            pc[*si] = trial_pc[*si].clone();
                            any_accept = true;
                            acc_events += 1;
                        }
                    }
                    if any_accept {
                        // Heuristic consumers (shortlist, tier-0 ranker) read
                        // the latest any-accept pass; a rejected column's
                        // values in it reflect its rejected β — heuristic-
                        // plane only, the verdict-bearing values are per_best.
                        std::mem::swap(&mut pass, &mut trial);
                        lam = (lam * 1.5).min(super::beta::lambda_cap());
                    } else {
                        lam *= 0.5;
                        if lam < super::beta::LAMBDA_MIN {
                            break;
                        }
                    }
                    let mut bt = f64::INFINITY;
                    for c in 0..nf {
                        bt = bt.min(comp(&per_best, c));
                    }
                    if bt.is_finite() {
                        // per_best only ever grows, so bt >= the pre-ascent b.
                        b = b.max(bt);
                    }
                    if closed(mode, b) {
                        break;
                    }
                    let before = active.len();
                    active.retain(|&c| !closed(mode, comp(&per_best, c)));
                    closed_cols += before - active.len();
                }
                beta_step = lam;
                // #margin-row-col-retire: `per_best` is the running max of
                // certified per-column values — strictly the best retirement
                // evidence this eval produced.
                if let Some(tracked) = per_track.as_mut() {
                    tracked.clone_from(&per_best);
                }
                // Children inherit only real β: prune all-zero slots.
                pc.retain(|s| s.trunk.iter().chain(s.heads.iter()).any(|&v| v > 0.0));
                super::beta::report_pc(
                    entry.trunk.len() + entry.heads.len(),
                    entry.trunk.len(),
                    b0,
                    b,
                    cols0,
                    acc_events,
                    closed_cols,
                    t_last,
                    lam,
                );
            }
        } else if beta_armed && !closed(mode, b) {
            let heads_dir: &[HeadFix] = if head_admissible { &entry.heads } else { &[] };
            let dir0 = if entry.trunk.is_empty() && heads_dir.is_empty() {
                // Nothing movable (e.g. head-only domain with HEADS=0):
                // don't burn a walk to learn S == 0.
                None
            } else {
                super::beta::step_scales(
                    self.eng.net,
                    self.eng.root,
                    &dom,
                    &entry.trunk,
                    heads_dir,
                    &ms.seed,
                    &pass,
                    worst_col,
                )
                .or_else(|| {
                    super::beta::refuse("step-scales");
                    None
                })
            };
            if let Some(mut dir) = dir0 {
                let b0 = b;
                // Must-fix #3: the ascended function is the DIRECT-path value
                // of the worst column — per_j[worst_col] — NOT the composite
                // b = max(direct, m1, m2v): when m1/m2v carries the max, the
                // composite's gap is not the ascended function's gap.
                // Epsilon-target past the STRICT closure threshold (see the
                // per-column site): a zero target converges to a graze.
                let mut gap = super::beta::CLOSE_EPS - per_j[worst_col];
                let polyak = super::beta::polyak();
                let mut lam = beta_step;
                let mut eta = super::beta::eta();
                let mut t_last = 0.0f64;
                let n_t = entry.trunk.len();
                for _ in 0..super::beta::iters() {
                    if self.cfg.deadline.is_some_and(|dl| Instant::now() > dl) {
                        break;
                    }
                    // Concatenation trunk-then-heads: one step rule over both.
                    // The β vector must mirror the DIRECTION's shape: with
                    // heads inactive (`dir.heads` empty) the head β are
                    // provably all zero (zeroed on refusal; never raised with
                    // HEADS=0) and are excluded, or the step fn would refuse
                    // on a length mismatch.
                    let cat_b: Vec<f64> = if dir.heads.is_empty() {
                        betas.clone()
                    } else {
                        betas.iter().chain(head_betas.iter()).copied().collect()
                    };
                    let mut cat_d: Vec<(f64, f64)> = Vec::with_capacity(cat_b.len());
                    cat_d.extend_from_slice(&dir.trunk);
                    cat_d.extend_from_slice(&dir.heads);
                    let cand_cat = if polyak {
                        match super::beta::polyak_step(&cat_b, &cat_d, gap, lam) {
                            Some((v, t)) => {
                                t_last = t;
                                v
                            }
                            None => {
                                super::beta::refuse("polyak-step");
                                break;
                            }
                        }
                    } else {
                        let Some(v) = super::beta::apply_step(&cat_b, &cat_d, eta) else {
                            break;
                        };
                        v
                    };
                    let (cand_t, cand_h) = cand_cat.split_at(n_t);
                    super::beta::set_terms(&mut dom, self.eng.root, &entry.trunk, cand_t);
                    let seed_trial = if head_admissible && cand_h.iter().any(|&hb| hb > 0.0) {
                        match super::beta::seed_with_head_terms(
                            &ms.seed,
                            &entry.heads,
                            cand_h,
                            mode.outward(),
                        ) {
                            Some(s) => Some(s),
                            None => {
                                super::beta::refuse("head-seed-shift-trial");
                                break;
                            }
                        }
                    } else {
                        None
                    };
                    // Same seed family, same collect shape. C2: the gradient
                    // capture stays ON so an accepted trial carries the fresh
                    // x*/vsigns the direction refresh below reads (the capture
                    // is a read-only row copy; the trial bound is
                    // bit-identical to an uncaptured pass).
                    let trial = self.eng.run_collect(
                        seed_trial.as_ref().unwrap_or(&ms.seed),
                        Some(&dom),
                        super::engine::LaneDir::Lower,
                        None,
                        collect,
                    );
                    let Ok(mut trial) = trial else { break };
                    let per_t = per_class_direct(&self.eng, &trial, &ms, 0..nf);
                    let mut bt = f64::INFINITY;
                    let mut wc_t = 0usize;
                    for r0 in 0..nf {
                        let pj = per_t[r0].max(ms.m1[r0]).max(m2v[r0]);
                        if !pj.is_finite() {
                            bt = f64::NAN;
                            break;
                        }
                        if pj < bt {
                            wc_t = r0;
                        }
                        bt = bt.min(pj);
                    }
                    if bt.is_finite() && bt > b {
                        b = bt;
                        betas = cand_t.to_vec();
                        if !dir.heads.is_empty() {
                            head_betas = cand_h.to_vec();
                        }
                        // #margin-row-col-retire: each accepted `per_t[c]` is
                        // a valid column bound (per-column weak duality under
                        // the accepted shared β).
                        if let Some(tracked) = per_track.as_mut() {
                            for (tv, &nv) in tracked.iter_mut().zip(&per_t) {
                                if nv.is_finite() && nv > *tv {
                                    *tv = nv;
                                }
                            }
                        }
                        std::mem::swap(&mut pass, &mut trial);
                        if closed(mode, b) {
                            // The accepted β already closes this domain; the
                            // caller re-checks `closed` right after eval.
                            break;
                        }
                        // C2 live gradient: refresh the worst column, the gap
                        // and the direction from the ACCEPTED pass — x* and g
                        // go stale precisely when the step is material.
                        worst_col = wc_t;
                        gap = -per_t[worst_col];
                        if polyak {
                            lam = (lam * 1.5).min(super::beta::lambda_cap());
                        } else {
                            eta *= 2.0;
                        }
                        match super::beta::step_scales(
                            self.eng.net,
                            self.eng.root,
                            &dom,
                            &entry.trunk,
                            heads_dir,
                            &ms.seed,
                            &pass,
                            worst_col,
                        ) {
                            Some(d) => dir = d,
                            None => {
                                super::beta::refuse("dir-refresh");
                                break;
                            }
                        }
                    } else if polyak {
                        // Reject: keep the direction, halve λ (replaces the
                        // legacy `eta *= 0.5`), stop below the λ floor.
                        lam *= 0.5;
                        if lam < super::beta::LAMBDA_MIN {
                            break;
                        }
                    } else {
                        eta *= 0.5;
                        if eta < 1e-3 {
                            break;
                        }
                    }
                }
                if polyak {
                    // C2 step memory: children inherit where the schedule
                    // ended, not a reset (the measured cross-eval amnesia).
                    beta_step = lam;
                }
                // Leave the dom terms at the ACCEPTED beta: score_candidates
                // reuses `st.dom`, and children inherit `st.betas` /
                // `st.head_betas` (all valid on every child region — subsets
                // of this domain's). KNOWN GAP (accepted, design C3):
                // score_candidates does NOT see the head seed shift —
                // children are scored slightly loose and corrected at their
                // own eval.
                super::beta::set_terms(&mut dom, self.eng.root, &entry.trunk, &betas);
                super::beta::report(
                    entry.trunk.len() + entry.heads.len(),
                    entry.trunk.len(),
                    b0,
                    b,
                    b > b0,
                    &betas,
                    &head_betas,
                    t_last,
                    if polyak { lam } else { eta },
                );
            }
        }
        // #margin-row-beta-percol: strip the per-column terms before the dom
        // is stored — every downstream pass reusing `st.dom` (candidate
        // scoring, the stacked canary) runs on a DIFFERENT column layout
        // where these column indices would price the wrong objectives.
        // Children inherit the numeric `pc` state instead and re-install
        // terms at their own eval. (Same known-gap treatment as the C3 head
        // shift: children are scored slightly loose, corrected at their own
        // eval.) The stacked entry point additionally fails closed on any
        // `beta_pc`-carrying block.
        dom.beta_pc.clear();
        // #margin-row-col-retire: record every column whose certified
        // composite crossed the closure threshold at THIS eval; children
        // inherit the union. The predicate is the same strict `closed()` the
        // domain verdict uses, over the same certified components
        // (direct/m1/m2v), so a retired column is exactly one the ancestor
        // PROVED on a region containing every descendant.
        let mut retired_out = entry.retired.clone();
        if retire_on {
            let tracked = per_track.as_deref().unwrap_or(per_j.as_slice());
            let mut newly = 0usize;
            for c in 0..nf {
                let comp = tracked[c].max(ms.m1[c]).max(m2v[c]);
                if closed(mode, comp) {
                    let root_c = match &survivors {
                        Some(map) => map[c],
                        None => c,
                    };
                    if let Ok(c16) = u16::try_from(root_c) {
                        if !retired_out.contains(&c16) {
                            retired_out.push(c16);
                            newly += 1;
                        }
                    }
                }
            }
            if newly > 0 {
                retired_out.sort_unstable();
                col_retire_report(
                    entry.trunk.len() + entry.heads.len(),
                    newly,
                    retired_out.len(),
                    nf - newly,
                );
            }
            // pc state rests ROOT-frame (see the remap at the top).
            if let Some(map) = &survivors {
                for s in &mut pc {
                    s.col = map[s.col];
                }
            }
        }
        let coll = pass.coll.take().unwrap_or_default();
        let coll_rows = pass.coll_rows.take().unwrap_or_default();
        Ok(EvalOut::Node(
            b,
            Box::new(NodeState {
                ybox,
                gates,
                ms,
                betas,
                head_betas,
                beta_step,
                pc,
                retired: retired_out,
                mb_local,
                coll,
                coll_rows,
                pass,
                dom,
                pack,
            }),
        ))
    }

    /// #margin-row-col-retire: the margin batch a node's `ms`/`pass` were
    /// computed under. Everything that consumes a `NodeState`'s per-column
    /// tensors (candidate selection, candidate scoring, the variant rankers)
    /// MUST read the batch through this accessor — widths must agree.
    fn node_mb<'s>(&'s self, st: &'s NodeState) -> &'s MarginBatch {
        st.mb_local.as_ref().unwrap_or(&self.mb)
    }

    /// Exact single-gate variant scores of every unstable, unused head
    /// neuron: `(min child, child sum, neuron)`, unsorted.
    fn head_variant_scores(&self, st: &NodeState, heads: &[HeadFix]) -> Vec<(f64, f64, usize)> {
        let mb = self.node_mb(st);
        let used: std::collections::BTreeSet<usize> = heads.iter().map(|(i, _)| *i).collect();
        let cands: Vec<usize> = (0..mb.n_y)
            .filter(|&i| st.ybox.ly[i] < 0.0 && st.ybox.uy[i] > 0.0 && !used.contains(&i))
            .collect();
        if cands.is_empty() {
            return Vec::new();
        }
        let vs = variant_state(mb, &st.gates, &st.ybox, &st.pack.al, &st.pack.au);
        cands
            .into_iter()
            .filter_map(|i| {
                let ba = head_variant(
                    mb,
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
                    mb,
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
            // #margin-row-col-retire: the nested lane re-derives its own root
            // fail-set on the epoch gates; the outer retirement set is NOT
            // inherited (root frames differ), only the gate carries over.
            col_retire: self.cfg.col_retire,
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
            let mb = self.node_mb(st);
            let used: std::collections::BTreeSet<usize> =
                entry.heads.iter().map(|(i, _)| *i).collect();
            for i in (0..mb.n_y)
                .filter(|&i| st.ybox.ly[i] < 0.0 && st.ybox.uy[i] > 0.0 && !used.contains(&i))
            {
                let ba = super::bounds::head_variant_direct(
                    mb,
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
                    mb,
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
        let mb = self.node_mb(st);
        let nf = mb.nf();
        let n_y = mb.n_y;
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
                // #margin-row-seed-copy: one contiguous block assign instead of
                // `n_y * nf` bounds-checked 2-D writes. `seed` is (n_y, total)
                // and the source is (n_y, nf), so columns `r0..r0+nf` of EVERY
                // row are exactly the source. Value-identical -- same elements,
                // same order, no arithmetic -- but a strided copy rather than
                // element-by-element.
                //
                // Runs `2 * (n_t + n_h)` times per expansion. `score_candidates`
                // is 40.0% of the margin-row lane at 554 ms/call, measured with
                // NY_MARGIN_ROW_PROFILE=1 on idx_6659, and that lane is what
                // actually proves cifar100 rows.
                seed.slice_mut(s![.., r0..r0 + nf]).assign(&st.ms.seed.s);
                if let (Some(dst), Some(src)) = (seed_e.as_mut(), st.ms.seed.e.as_ref()) {
                    dst.slice_mut(s![.., r0..r0 + nf]).assign(src);
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
                let ms2 = margin_seed(mb, &gates2, &st.ybox, mode);
                // #margin-row-seed-copy: see the trunk block above.
                seed.slice_mut(s![.., r0..r0 + nf]).assign(&ms2.seed.s);
                if let (Some(dst), Some(src)) = (seed_e.as_mut(), ms2.seed.e.as_ref()) {
                    dst.slice_mut(s![.., r0..r0 + nf]).assign(src);
                }
                for f in 0..nf {
                    cst[r0 + f] = ms2.cst[f];
                    cst_err[r0 + f] = ms2.cst_err[f];
                    m1[r0 + f] = if dr > 0 {
                        st.ms.m1[f]
                    } else {
                        let t = mb.wn[f * n_y + i] * (0.0 - st.ms.zu[i]);
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
            nf,
        })
    }

    /// One backward dispatch for the candidate-score columns of several
    /// independently gated domains. Arithmetic and reductions remain
    /// column-local; the engine validates the explicit ownership partition.
    /// `Ok(None)` = intra-expansion deadline poll fired (#deadline-poll, D2).
    fn score_candidates_domain_stacked(
        &self,
        ready: &[&ReadyExpand],
    ) -> Result<Option<Vec<Vec<(f64, f64)>>>> {
        if ready.len() < 2 {
            return Err(NyError::InvalidSpec(
                "margin_row: cross-domain score stack needs at least two domains".into(),
            ));
        }
        // #deadline-poll (D2): guard the (large) stacked scoring dispatch.
        if self.deadline_hit() {
            poll_note("score_candidates_stacked");
            return Ok(None);
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
        let mut all = Vec::with_capacity(columns.len());
        let mut offset = 0usize;
        for domain in &columns {
            // #margin-row-col-retire: decode at the OWNING domain's width —
            // stacked domains can carry different survivor counts.
            let nf = domain.nf;
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
            // The differential's independent rescore can only return None on a
            // deadline poll; the stacked pass above already survived its own
            // poll and tests run without deadlines, so skip rather than panic.
            let Some(independent) =
                self.score_candidates(&ready.st, &ready.trunk_cands, &ready.head_cands)?
            else {
                continue;
            };
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
        Ok(Some(all))
    }

    /// ONE batched exception pass scoring all candidate children exactly.
    /// Returns per candidate `(bound(+1 child), bound(-1 child))`;
    /// `Ok(None)` = intra-expansion deadline poll fired (#deadline-poll, D2).
    fn score_candidates(
        &self,
        st: &NodeState,
        trunk_cands: &[(usize, usize)],
        head_cands: &[usize],
    ) -> Result<Option<Vec<(f64, f64)>>> {
        // #deadline-poll (D2): guard the column assembly + the ~0.55 s
        // scoring pass about to be spent (40% of the lane, measured).
        if self.deadline_hit() {
            poll_note("score_candidates");
            return Ok(None);
        }
        let mb = self.node_mb(st);
        let nf = mb.nf();
        let n_y = mb.n_y;
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
                // #margin-row-seed-copy: one contiguous block assign instead of
                // `n_y * nf` bounds-checked 2-D writes. `seed` is (n_y, total)
                // and the source is (n_y, nf), so columns `r0..r0+nf` of EVERY
                // row are exactly the source. Value-identical -- same elements,
                // same order, no arithmetic -- but a strided copy rather than
                // element-by-element.
                //
                // Runs `2 * (n_t + n_h)` times per expansion. `score_candidates`
                // is 40.0% of the margin-row lane at 554 ms/call, measured with
                // NY_MARGIN_ROW_PROFILE=1 on idx_6659, and that lane is what
                // actually proves cifar100 rows.
                seed.slice_mut(s![.., r0..r0 + nf]).assign(&st.ms.seed.s);
                if let (Some(dst), Some(src)) = (seed_e.as_mut(), st.ms.seed.e.as_ref()) {
                    dst.slice_mut(s![.., r0..r0 + nf]).assign(src);
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
                let ms2 = margin_seed(mb, &gates2, &st.ybox, mode);
                // #margin-row-seed-copy: see the trunk block above.
                seed.slice_mut(s![.., r0..r0 + nf]).assign(&ms2.seed.s);
                if let (Some(dst), Some(src)) = (seed_e.as_mut(), ms2.seed.e.as_ref()) {
                    dst.slice_mut(s![.., r0..r0 + nf]).assign(src);
                }
                for f in 0..nf {
                    cst[r0 + f] = ms2.cst[f];
                    cst_err[r0 + f] = ms2.cst_err[f];
                    // m1 for the child: dr>0 keeps m1; dr<0 zeroes z_i's upper
                    // contribution (exact; see bab_direct.py score_candidates).
                    m1[r0 + f] = if dr > 0 {
                        st.ms.m1[f]
                    } else {
                        let t = mb.wn[f * n_y + i] * (0.0 - st.ms.zu[i]);
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
        Ok(Some(out))
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
            let want = build_pack(&eng, dom_ref, &[]).expect("build_pack");
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

/// #margin-row-col-retire: GPU-free unit tests.
///
/// * `subset` row copies are bitwise the full batch's rows;
/// * a NARROWED eval reproduces the full-width per-column certified values
///   BIT-FOR-BIT for every surviving column (m1 / direct / m2v — the engine's
///   column-locality invariant, the same one the domain-stacked differential
///   pins);
/// * a column retired at an ancestor is empirically positive on sampled
///   points of the region (the certificate the retirement invariant banks);
/// * children inherit the retirement set monotonically and their evals narrow
///   (the engagement observable);
/// * with the cfg gate OFF an inherited set is inert (full width, passthrough).
#[cfg(test)]
mod col_retire_tests {
    use ndarray::Array2;

    use super::*;
    use crate::margin_row::engine::LaneDir;
    use crate::margin_row::spec::{TwinOpSpec, TwinSpec};

    const N_IN: usize = 6;
    const N_Y: usize = 4;
    const N_OUT: usize = 4;

    /// Twin net whose class-1 output row is IDENTICAL to class 0's with bias
    /// 4.0 lower: the margin `Y_0 - Y_1` is exactly +4.0 everywhere, so the
    /// column retires DETERMINISTICALLY at the first eval (m1 = 4.0 minus a
    /// tiny outward slack) regardless of the box. Class 2 is `-row0` (a real
    /// competitor: its margin goes negative in the box), class 3 is a third
    /// distinct pattern.
    #[allow(clippy::cast_precision_loss)]
    fn spec4() -> TwinSpec {
        let wh: Vec<f64> = (0..(N_Y * N_IN))
            .map(|i| ((i * 7) % 11) as f64 / 11.0 - 0.5)
            .collect();
        let row0: Vec<f64> = (0..N_Y).map(|k| ((k * 5) % 7) as f64 / 7.0 - 0.3).collect();
        let mut wo = Vec::with_capacity(N_OUT * N_Y);
        wo.extend_from_slice(&row0); // class 0
        wo.extend_from_slice(&row0); // class 1: identical weights
        wo.extend(row0.iter().map(|v| -v)); // class 2: strong competitor
        wo.extend((0..N_Y).map(|k| ((k * 3) % 5) as f64 / 5.0 - 0.4)); // class 3
        TwinSpec {
            n_in: N_IN,
            ops: vec![
                TwinOpSpec::Relu { input: 0 },
                TwinOpSpec::Flatten { input: 1 },
                TwinOpSpec::Gemm {
                    input: 2,
                    weight: wh,
                    bias: vec![0.1, -0.1, 0.05, 0.0],
                    shape: (N_Y, N_IN),
                },
                TwinOpSpec::Relu { input: 3 },
                TwinOpSpec::Gemm {
                    input: 4,
                    weight: wo,
                    bias: vec![0.0, -4.0, 0.0, -0.05],
                    shape: (N_OUT, N_Y),
                },
            ],
        }
    }

    fn build(net: &TwinNet) -> (Vec<f64>, Vec<f64>, RootGates) {
        let lo = vec![-0.4; N_IN];
        let hi = vec![0.4; N_IN];
        let root = RootGates::build(net, &lo, &hi, RoundMode::Outward, None).expect("root");
        (lo, hi, root)
    }

    fn driver<'a>(
        net: &'a TwinNet,
        root: &'a RootGates,
        adv: &[usize],
        col_retire: bool,
    ) -> (MarginRowBab<'a>, Arc<YPack>, f64) {
        let eng = BackwardEngine::new(net, root);
        let re = root_eval(&eng, net, 0, adv).expect("root_eval");
        let root_bound = re.dj.iter().copied().fold(f64::INFINITY, f64::min);
        let mut bab = MarginRowBab {
            eng: BackwardEngine::new(net, root),
            mb: MarginBatch::new(net, 0, adv).expect("mb"),
            cfg: BabConfig {
                lru_cap: 8,
                col_retire,
                ..BabConfig::default()
            },
            lru: Lru {
                cap: 8,
                entries: Vec::new(),
            },
            stats: BabStats {
                root_bound,
                tree_classes: adv.to_vec(),
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
        let pack = re.pack;
        bab.lru.put(Vec::new(), pack.clone());
        (bab, pack, root_bound)
    }

    fn root_entry_over(pack: &YPack, bound: f64, retired: Vec<u16>) -> DomainEntry {
        DomainEntry {
            bound,
            seq: 0,
            trunk: Vec::new(),
            heads: Vec::new(),
            ly: pack.ly0.clone(),
            uy: pack.uy0.clone(),
            betas: Vec::new(),
            head_betas: Vec::new(),
            beta_step: crate::margin_row::beta::lambda(),
            pc: Vec::new(),
            retired,
        }
    }

    #[test]
    fn subset_rows_are_bitwise_and_validation_rejects_bad_sets() {
        let spec = spec4();
        let net = TwinNet::compile(&spec).expect("compiles");
        let full = MarginBatch::new(&net, 0, &[1, 2, 3]).expect("full");
        let sub = full.subset(&[0, 2]).expect("subset");
        let direct = MarginBatch::new(&net, 0, &[1, 3]).expect("direct");
        assert_eq!(sub.adv, direct.adv);
        assert_eq!(sub.n_y, direct.n_y);
        assert_eq!(sub.nf(), direct.nf());
        assert_eq!(sub.wp.len(), direct.wp.len());
        assert_eq!(sub.wn.len(), direct.wn.len());
        for (a, b) in sub.wp.iter().zip(&direct.wp) {
            assert_eq!(a.to_bits(), b.to_bits(), "wp moved");
        }
        for (a, b) in sub.wn.iter().zip(&direct.wn) {
            assert_eq!(a.to_bits(), b.to_bits(), "wn moved");
        }
        for (a, b) in sub.cst.iter().zip(&direct.cst) {
            assert_eq!(a.to_bits(), b.to_bits(), "cst moved");
        }
        assert!(full.subset(&[2, 1]).is_err(), "unordered must refuse");
        assert!(full.subset(&[1, 1]).is_err(), "duplicate must refuse");
        assert!(full.subset(&[3]).is_err(), "out of range must refuse");
    }

    /// The bit-parity claim the whole design rests on: for a SURVIVING column,
    /// the narrowed batch's certified values (m1, direct, m2v) are bitwise the
    /// full-width batch's values for that column. Column arithmetic is
    /// column-local in the engine (the same invariant the domain-stacked
    /// differential pins), and `subset` copies rows verbatim.
    #[test]
    fn narrowed_batch_bit_parity_for_surviving_columns() {
        let spec = spec4();
        let net = TwinNet::compile(&spec).expect("compiles");
        let (_lo, _hi, root) = build(&net);
        let eng = BackwardEngine::new(&net, &root);
        let re = root_eval(&eng, &net, 0, &[1, 2, 3]).expect("root_eval");
        let pack = re.pack;
        let ybox = YBox {
            ly: pack.ly0.clone(),
            uy: pack.uy0.clone(),
        };
        let mode = RoundMode::Outward;
        let gates = head_gates(&ybox, mode);
        let full = MarginBatch::new(&net, 0, &[1, 2, 3]).expect("full");
        let keep = [0usize, 2];
        let sub = full.subset(&keep).expect("subset");

        let ms_full = margin_seed(&full, &gates, &ybox, mode);
        let ms_sub = margin_seed(&sub, &gates, &ybox, mode);
        for (i, &c) in keep.iter().enumerate() {
            assert_eq!(ms_sub.m1[i].to_bits(), ms_full.m1[c].to_bits(), "m1 moved");
            assert_eq!(
                ms_sub.cst[i].to_bits(),
                ms_full.cst[c].to_bits(),
                "cst moved"
            );
            assert_eq!(
                ms_sub.cst_err[i].to_bits(),
                ms_full.cst_err[c].to_bits(),
                "cst_err moved"
            );
            for j in 0..N_Y {
                assert_eq!(
                    ms_sub.seed.s[[j, i]].to_bits(),
                    ms_full.seed.s[[j, c]].to_bits(),
                    "seed moved"
                );
            }
        }

        let pass_full = eng
            .run(&ms_full.seed, None, LaneDir::Lower, None, false)
            .expect("full pass");
        let pass_sub = eng
            .run(&ms_sub.seed, None, LaneDir::Lower, None, false)
            .expect("sub pass");
        let per_full = per_class_direct(&eng, &pass_full, &ms_full, 0..full.nf());
        let per_sub = per_class_direct(&eng, &pass_sub, &ms_sub, 0..sub.nf());
        for (i, &c) in keep.iter().enumerate() {
            assert_eq!(
                per_sub[i].to_bits(),
                per_full[c].to_bits(),
                "direct bound moved for surviving column {c}"
            );
        }

        let m2v_full = compose_viay(
            &eng,
            &full,
            &gates,
            &pack.al,
            &pack.au,
            &pack.al_dots,
            &pack.au_dots,
            mode,
        );
        let m2v_sub = compose_viay(
            &eng,
            &sub,
            &gates,
            &pack.al,
            &pack.au,
            &pack.al_dots,
            &pack.au_dots,
            mode,
        );
        for (i, &c) in keep.iter().enumerate() {
            assert_eq!(
                m2v_sub[i].to_bits(),
                m2v_full[c].to_bits(),
                "m2v moved for surviving column {c}"
            );
        }
    }

    /// End-to-end: the deterministic +4.0 column retires at the first eval;
    /// its TRUE margin is positive on sampled region points (the certificate
    /// the invariant banks); children inherit the set; a child eval narrows
    /// its seed to the survivors and keeps the inherited set.
    #[test]
    fn retirement_populates_inherits_and_narrows() {
        let spec = spec4();
        let net = TwinNet::compile(&spec).expect("compiles");
        let (lo, hi, root) = build(&net);
        let adv = [1usize, 2, 3];
        let (mut bab, pack, root_bound) = driver(&net, &root, &adv, true);
        let mode = bab.eng.root.mode;
        let entry = root_entry_over(&pack, root_bound, Vec::new());
        let EvalOut::Node(b, st) = bab.eval_with_pack(&pack, &entry).expect("eval") else {
            panic!("root domain must evaluate")
        };
        assert!(
            st.retired.contains(&0u16),
            "column 0 (margin == +4.0 identically) must retire, got {:?}",
            st.retired
        );
        assert!(
            st.retired.len() < adv.len(),
            "a pushed frontier needs >= 1 survivor"
        );

        // Sampled soundness of every retired column over the region.
        let mbf = MarginBatch::new(&net, 0, &adv).expect("mb");
        let npts = 64usize;
        let mut xs = Array2::<f64>::zeros((N_IN, npts));
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for p in 0..npts {
            for i in 0..N_IN {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                #[allow(clippy::cast_precision_loss)]
                let u = ((state >> 11) as f64) / ((1u64 << 53) as f64);
                xs[[i, p]] = lo[i] + u * (hi[i] - lo[i]);
            }
        }
        let (y, _) = net
            .forward_points(&xs, &std::collections::BTreeMap::new())
            .expect("forward");
        for &c16 in &st.retired {
            let c = usize::from(c16);
            for p in 0..npts {
                let mut m = mbf.cst[c];
                for k in 0..N_Y {
                    m += (mbf.wp[c * N_Y + k] + mbf.wn[c * N_Y + k]) * y[[k, p]].max(0.0);
                }
                assert!(
                    m > 0.0,
                    "retired column {c} has non-positive sampled margin {m}"
                );
            }
        }

        // Inheritance + narrowed descendant eval.
        if !closed(mode, b.max(entry.bound)) {
            if let ExpandStep::Expanded { pushes, .. } =
                bab.expand_one(&pack, &entry).expect("expand")
            {
                assert!(!pushes.is_empty(), "open root must push children");
                for child in &pushes {
                    for c in &st.retired {
                        assert!(
                            child.retired.contains(c),
                            "child lost inherited retirement {c}"
                        );
                    }
                    let child_pack = bab.rows_for(&child.trunk).expect("child pack");
                    let centry = DomainEntry {
                        bound: child.bound,
                        seq: 1,
                        trunk: child.trunk.clone(),
                        heads: child.heads.clone(),
                        ly: child.ly.clone(),
                        uy: child.uy.clone(),
                        betas: child.betas.clone(),
                        head_betas: child.head_betas.clone(),
                        beta_step: child.beta_step,
                        pc: child.pc.clone(),
                        retired: child.retired.clone(),
                    };
                    if let EvalOut::Node(_, st2) = bab
                        .eval_with_pack(&child_pack, &centry)
                        .expect("child eval")
                    {
                        // Engagement observable: the child's pass is narrowed
                        // to the survivors.
                        assert_eq!(
                            st2.ms.seed.s.ncols(),
                            adv.len() - centry.retired.len(),
                            "child eval did not narrow to the survivors"
                        );
                        for c in &centry.retired {
                            assert!(st2.retired.contains(c), "child dropped {c}");
                        }
                    }
                }
            }
        }
    }

    /// Gate OFF: an inherited set is INERT — full width, passthrough — so the
    /// default lane is unchanged by construction.
    #[test]
    fn gate_off_ignores_inherited_retirement() {
        let spec = spec4();
        let net = TwinNet::compile(&spec).expect("compiles");
        let (_lo, _hi, root) = build(&net);
        let adv = [1usize, 2, 3];
        let (bab, pack, root_bound) = driver(&net, &root, &adv, false);
        let entry = root_entry_over(&pack, root_bound, vec![0u16]);
        let EvalOut::Node(_, st) = bab.eval_with_pack(&pack, &entry).expect("eval") else {
            panic!("root domain must evaluate")
        };
        assert_eq!(st.ms.seed.s.ncols(), adv.len(), "gate off must stay full");
        assert!(st.mb_local.is_none(), "gate off must not build a subset");
        assert_eq!(st.retired, vec![0u16], "gate off must pass through");
    }
}
