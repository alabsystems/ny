// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#frozen-dual-bab` — V1 STAGE B of the frozen alpha-zonotope dual-ascent
//! BaB (VIBECHECK_PLAN.md §6.V1): the certified SEARCH over stage A's query
//! state ([`super::zono_state`]), run as a TIER between the margin-row lane's
//! certified root and its existing best-first tree.
//!
//! # The search (port of the P1 replica's `bfs` / `node_bounds`)
//!
//! Per open class the tier builds the certified state, sorts the unstable rows
//! once by the static box + single-halfspace score ([`static_order`]) and runs
//! a level-synchronous BFS: at depth `D` every open node splits row
//! `order[D]` into its `Off` / `On` children. Each node carries one pair of
//! Lagrange multipliers per split (`L0` for the sign cut, `L1` for the far cut),
//! stored as non-negative `f32` search state, and is evaluated at FOUR dual
//! points, each through stage A's certified evaluator
//! [`FrozenDualProblem::eval_node`]:
//!
//! 1. the warm start inherited from the parent (plus its supergradient);
//! 2. the warm start with every multiplier whose cut is slack at the
//!    relaxation's minimiser set to 0;
//! 3. the BOX-CORNER FLOOR — all multipliers 0, i.e. the substituted
//!    objective minimised over the generator box;
//! 4. the far probe — the projected supergradient step to the first
//!    multiplier that hits 0 (or far along the ray when none does), CLAMPED
//!    to `>= 0` and evaluated with `rc` recomputed at the clamped point.
//!
//! A node is pruned ONLY when the largest CERTIFIED value of the four is
//! `> 0` (strict). The log-bucket line search along the projected
//! supergradient only sets the CHILDREN's warm start — one dual step per node,
//! inherited down the tree.
//!
//! # Why every prune is sound
//!
//! * Stage A's state contains the network: for every input `x` in the box
//!   there is a generator vector `e(x) in [-1, 1]^n` with the margin within
//!   `r_obj` of `c_obj + d . e(x)` and every unstable pre-activation within
//!   `r_k` of its row (exact-rational and double-double containment tests).
//! * A split partitions the box by the SIGN of a real pre-activation, so the
//!   two children cover the parent (`z = 0` lies in both); substituting
//!   `relu(z) = 0` / `= z` is exact on the child.
//! * Weak duality: `g(L)` lower-bounds the child's margin for EVERY `L >= 0`.
//!   How `L` was found — warm start, clamping, a line search or a far probe —
//!   is irrelevant. [`FrozenDualProblem::eval_node`] clamps `L` itself,
//!   recomputes `rc` at the clamped point, refuses a non-finite multiplier and
//!   subtracts `1.001 gamma_N sum|terms|` before rounding down.
//! * Vibecheck's own far probe (`rc` advanced with UNCLAMPED multipliers while
//!   the `b`-term uses clamped ones — not a dual bound even in exact
//!   arithmetic, DECISION.md U4b) and its slope-drop infeasibility test (U3) are
//!   NOT ported: the far point here is an ordinary clamped dual point.
//! * `unsat` for a class is published ONLY when the frontier is empty after
//!   evaluating every node of every level, and the LEVEL LEDGER proves it:
//!   level 1 holds the two root children, every level holds exactly twice the
//!   previous level's unpruned nodes, and the last level is fully pruned
//!   ([`ledger_closes`]). A deadline, node cap, memory refusal or exhausted
//!   split order leaves the class OPEN and hands it back to the margin-row
//!   tree unchanged. Nothing here can emit `sat`.
//!
//! # Memory
//!
//! ny has no memory governor without a cgroup, so every allocation that
//! scales with the search is admitted first: the state build by stage A's own
//! admission, the substituted problem (at most [`FROZEN_DUAL_MAX_DEPTH`] rows)
//! against half the admitted budget, and every BFS level's worst-case working
//! set (current level, kept warm starts, next level, per-thread scratch)
//! against [`admitted_bytes`] re-read per level. A refused level stops the
//! search OPEN (`reason=memory`), never partially.
//!
//! # Evaluation (`#fd-block-eval`, ENGINE_PROGRAM M1)
//!
//! [`Evaluator::Block`] (the default) evaluates a work item's nodes in blocks
//! of `BLOCK_NODES` same-depth nodes through `FrozenDualProblem::eval_block`,
//! which is BIT-IDENTICAL to `eval_node` node by node, so the certificate
//! arithmetic and the search trajectory are unchanged: every stage below is
//! the per-node formula of [`Evaluator::Node`] (kept, unchanged, as the
//! reference) applied lane by lane.
//!
//! * `far_first` evaluates the far point before the clamped and floor points
//!   and skips those two on a node the warm or far point already CERTIFIES
//!   closed. That is exact for the tree: the far point depends only on the
//!   warm point's `rc`, subgradient and line search (never on the clamped or
//!   floor value), a pruned node has no children, and a node neither point
//!   prunes still gets both evaluated, so its prune decision, `pending`, its
//!   children's warm start and `best` are the reference's. Only two
//!   statistics can move: the pruned node's certified value (the max over the
//!   points evaluated, so `cert_min_slack` can only go DOWN) and `far_used`
//!   (then `far > warm`).
//! * Every prune goes through ONE strict predicate (`NodeOut::prunes`).
//! * A deterministic sample of the block path's prunes (every prune at depth
//!   `<= 4`, and a hash-selected 1 in [`SearchOptions::crosscheck_every`]) is
//!   re-evaluated by the scalar `eval_node` at the winning multipliers; any
//!   bit difference is a soundness finding: the search stops OPEN
//!   (`reason=refused`) and prints `[frozen-dual] CROSSCHECK-MISMATCH`.
//!
//! # The shared trunk (`#fd-shared-state`, ENGINE_PROGRAM M3)
//!
//! Per class, the state build is ~all row passes, and the rows depend on the
//! class only through the slopes they read. `margin_row.frozen_dual_bab_state:
//! shared` (default `per_class`, the pre-M3 state build bit for bit) therefore
//! prepares ONE trunk under the TIER deadline before any fair share exists:
//! the shared slopes (the better of the joint margin's direction-adaptive
//! choice and the hardest class's own, priced by one-column margin passes),
//! then every row once. Each class then pays only a one-column margin tail
//! and a copy of the rows inside its share. On `t8853` every class's build
//! ran under a ~1.3 s share it could not fit (15/15 refused); on `t4863`
//! six builds took 11.41 s of the 27.2 s tier. Soundness is the builder's
//! (`zono_state/build.rs`, "shared trunk"): any slope in `[0, 1]` certifies,
//! and a tail that disagrees with the trunk on a slope a row reads is
//! refused. Observables: `[frozen-dual] trunk state=shared lambda= ...` and
//! the summary's `state= trunk_ms= classes_built= shared_root_loss_max=`.
//!
//! # Wide search (`#wide-fd-search`, ENGINE_PROGRAM M4)
//!
//! `margin_row.frozen_dual_bab_search` (default `f64`: everything above,
//! bit for bit) can hand the SCORING of each level to an injected UNTRUSTED
//! backend (`ny_core::search::SearchBackend`: ny-bbe's CPU f32 SIMD search, or
//! ny-cuda's device search), which proposes, per node, its best dual point, the
//! other points and the children's warm start. The lane then CERTIFIES a node
//! only by evaluating it itself — `eval_block`, the same certified f64
//! evaluator — at a proposed point, and splits every node no proposal
//! certifies. The level loop, admission, the strict prune predicate and the
//! level ledger are the ones above; see the `wide` submodule's docs for the
//! soundness argument and the fallbacks.
//!
//! # State x search (the two selectors compose)
//!
//! `frozen_dual_bab_state` decides how a class's certified STATE is built (its
//! own build, or the shared trunk plus the class's tail, assembled by
//! `FrozenDualTrunk::state`, which binds the tail to the class the tier asked
//! for and the trunk and the tail to one generator layout, and re-checks the
//! trunk's content digest); `frozen_dual_bab_search` decides which search
//! SCORES the levels of the problem substituted from that state. Both modes
//! meet in ONE place, `search_state`: order, substitute, admit, then either
//! the f64 search or `wide::WideTier::search_class`. So:
//!
//! * the wide search only ever sees the problem of a state that the builder
//!   produced for THIS class — in `shared` mode, one that passed the class,
//!   layout and digest bindings; a refused assembly never reaches a backend;
//! * its f32 search copy is made from that problem and uploaded per class (a
//!   handle never outlives the class it was uploaded for, so no class is
//!   searched on another class's copy), and the lane-start probe, the sampled
//!   cross-check and the level ledger all compare against that problem's own
//!   certified evaluator;
//! * the shared trunk's bytes (`resident`) are charged before the frontier
//!   budget the backend's lease is carved from, so a shared tier's lease
//!   shrinks with the trunk it keeps alive (the last class drops the trunk
//!   before its search).
//!
//! The defaults retain the original per-class state and certified f64
//! evaluator. Cheap checked phase relations may propose additional multipliers
//! on complete admitted charts; only the original evaluator can prune.

use std::time::{Duration, Instant};

use ny_core::search::SearchBackend;
use rayon::prelude::*;

mod automatic;
mod breakpoint;
#[allow(dead_code)] // Private cover prototype; no search schedule changes yet.
mod prefix_cover;
mod progressive;
pub use progressive::ProgressiveStats;
mod relations;
pub use relations::RelationStats;
#[cfg(test)]
mod breakpoint_tests;

#[cfg(test)]
use crate::zono_core::rounding::next_up;
use crate::zono_core::ZonoRefusal;

use super::bab::RootEval;
use super::bounds::YBox;
use super::net::TwinNet;
use super::root::RootGates;
use super::zono_state::build::estimate_trunk_state_bytes;
use super::zono_state::problem::block::{
    BlockGeom, BlockScratch, BlockWant, BLOCK_COLS, BLOCK_NODES,
};
use super::zono_state::{
    admitted_bytes, class_margin, estimate_ranked_state_bytes, estimate_state_bytes, static_order,
    ClassTail, CompactRelax, FrozenDualFrame, FrozenDualOptions, FrozenDualProblem,
    FrozenDualState, FrozenDualTrunk, NodeBound, NodeScratch, Side, Subgradient, TrunkRelax,
    FROZEN_DUAL_DEFAULT_CHUNK_COLS, FROZEN_DUAL_DEFAULT_MAX_STATE_BYTES,
};

/// M4 `#wide-fd-search`: the injected UNTRUSTED level search and the f64
/// certification of its proposals (see that module's docs).
mod wide;

/// #router-r5 CR-G: the gap-order contract that funds the classes when the
/// acting router arms it (see that module's docs).
mod gap_contract;

pub(crate) use wide::WideTier;
#[cfg(test)]
pub(crate) use wide::{
    search_copy, search_rules as wide_search_rules, search_wide_with_test_hooks,
    tau_rel as wide_tau_rel, ReferenceBackend,
};
pub use wide::{
    search_with_backend, set_frozen_dual_search_backend_factory, FdBackendKind, FrozenDualSearch,
};

/// `margin_row.frozen_dual_bab_state` (`#fd-shared-state`, ENGINE_PROGRAM
/// M3): how the tier builds its per-class states.
///
/// The serde tokens are the typed preset key's values and the lever
/// declaration's `Enum` members ([`Self::TOKENS`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum FrozenDualStateMode {
    /// One full state build per class, inside that class's fair share: the
    /// state as it was before M3, bit for bit (the declaration default).
    #[default]
    #[serde(rename = "per_class")]
    PerClass,
    /// ONE trunk (every unstable row, under slopes shared across the open
    /// classes) built BEFORE the fair shares are computed, then per class a
    /// one-column margin tail and a copy of the rows inside its share.
    #[serde(rename = "shared")]
    Shared,
}

impl FrozenDualStateMode {
    /// The preset / environment vocabulary, in declaration order.
    pub const TOKENS: [&'static str; 2] = ["per_class", "shared"];

    /// Parse one canonical token (exact spelling).
    #[must_use]
    pub fn from_token(token: &str) -> Option<Self> {
        match token {
            "per_class" => Some(Self::PerClass),
            "shared" => Some(Self::Shared),
            _ => None,
        }
    }

    /// The canonical token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::PerClass => "per_class",
            Self::Shared => "shared",
        }
    }
}

/// The shared-trunk root loss (a class's own per-class root minus its root
/// over the shared trunk) at or below which the class counts as served by
/// the trunk: ENGINE_PROGRAM M3's `0.01` line. It ranks the slope candidates
/// and is reported; it never decides a prune.
pub const FROZEN_DUAL_SHARED_LOSS_TOL: f64 = 0.01;

/// Declaration default of `margin_row.frozen_dual_bab_node_cap`: nodes one
/// objective's search may evaluate (P1x: cifar100 `idx_9585` closes at 1.13M
/// nodes under the gate arm, `idx_3494` at 55k).
pub const FROZEN_DUAL_DEFAULT_NODE_CAP: u64 = 2_000_000;

/// Reader clamp of the node cap.
pub const FROZEN_DUAL_MAX_NODE_CAP: u64 = 200_000_000;

/// Automatic share when `margin_row.frozen_dual_bab_secs` is zero, additionally
/// capped to four seconds while reserving one second for the established tree.
pub const FROZEN_DUAL_DEFAULT_SHARE: f64 = automatic::SHARE;

/// The explicit wall an in-process harness arming of the lane tier
/// (`BabConfig::frozen_dual_tier`) requests when the governed seconds select
/// the automatic route (see [`lane_tier`]): the tier's pre-#breakpoint-dual
/// no-deadline wall, capped like any explicit request at
/// [`FROZEN_DUAL_MAX_SHARE`] of the lane's remaining time.
pub const FROZEN_DUAL_HARNESS_SECS: f64 = 60.0;

/// Ceiling on the share an explicit `frozen_dual_bab_secs` request may take:
/// the tree always keeps at least a quarter of the time the root left.
pub const FROZEN_DUAL_MAX_SHARE: f64 = 0.75;

/// Tier wall when the lane has no deadline and no explicit request (tests and
/// diagnostic drivers only; every scored lane carries a deadline).
pub const FROZEN_DUAL_NO_DEADLINE_SECS: f64 = automatic::MAX_SECS;

/// Longest split order a problem is built with (rows copied per class). P1 and
/// P1x close at depth <= 53.
pub const FROZEN_DUAL_MAX_DEPTH: usize = 512;

/// Nodes per parallel work item (the deadline is also re-checked every
/// [`DEADLINE_STRIDE`] nodes inside one).
pub const FROZEN_DUAL_CHUNK: usize = 128;

/// Log-spaced buckets of the line search (the replica's `Kb`).
const LOG_BUCKETS: usize = 256;

/// Vibecheck's `_TOL`, used ONLY by search-step logic (which multipliers move,
/// whether a node keeps stepping). It never decides a prune.
const SEARCH_TOL: f64 = 1e-9;

/// Deadline re-check stride inside a work item.
const DEADLINE_STRIDE: usize = 16;

/// Declaration default of [`SearchOptions::crosscheck_every`] and of
/// [`TierPolicy::crosscheck_every`]: the block path's prunes re-checked
/// against the scalar evaluator, besides every prune at depth `<= 4`
/// (ENGINE_PROGRAM G4). The scored lane does not use this constant: its rate
/// arrives from the entry point's audit rates through `lane_tier`'s
/// caller (the Competition profile's rate is this same 256; Proof checks
/// every prune). This module never learns which profile chose it.
pub const FROZEN_DUAL_CROSSCHECK_EVERY: u32 = 256;

/// Prunes at this depth or shallower are ALWAYS cross-checked.
const CROSSCHECK_DEPTH: usize = 4;

/// Lower edges of the certified-slack histogram's decades
/// ([`SearchReport::slack_hist`]): bucket 0 is `< 1e-12`, bucket `k` is
/// `[1e-(13-k), 1e-(12-k))`, the last is `>= 1e-4`.
const SLACK_DECADES: [f64; 9] = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4];

/// Buckets of [`SearchReport::slack_hist`].
pub const SLACK_BUCKETS: usize = SLACK_DECADES.len() + 1;

/// The histogram bucket of a pruned node's certified value.
fn slack_decade(v: f64) -> usize {
    SLACK_DECADES.iter().take_while(|&&t| v >= t).count()
}

/// Which node evaluator the search runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Evaluator {
    /// The per-node reference (`FrozenDualProblem::eval_node`, one node at a
    /// time, all four dual points).
    Node,
    /// `#fd-block-eval`: same-depth blocks through the bit-identical
    /// `FrozenDualProblem::eval_block`. `far_first` skips the clamped and floor
    /// points of a node the warm or far point certifies closed (same tree, see
    /// the module docs).
    Block {
        /// Evaluate the far point first (exact for the tree).
        far_first: bool,
    },
}

impl Evaluator {
    /// Stable token for observables (`eval=` on the class line).
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            Self::Node => "node",
            Self::Block { .. } => "block",
        }
    }
}

/// The evaluator the lane tier runs.
pub const FROZEN_DUAL_DEFAULT_EVALUATOR: Evaluator = Evaluator::Block { far_first: true };

/// Why a search stopped without closing its objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenReason {
    /// The next level would exceed the node cap.
    NodeCap,
    /// The deadline passed (between or inside a level).
    Deadline,
    /// The next level's working set was not admitted.
    Memory,
    /// Every row of the split order is split and nodes remain open.
    SplitsExhausted,
    /// An internal consistency check failed (fail-closed).
    Refused,
}

impl OpenReason {
    /// Stable token for observables.
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            Self::NodeCap => "node_cap",
            Self::Deadline => "deadline",
            Self::Memory => "memory",
            Self::SplitsExhausted => "splits_exhausted",
            Self::Refused => "refused",
        }
    }
}

/// Outcome of one objective's search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchVerdict {
    /// Every leaf certified (`certified > 0`) and the level ledger closes:
    /// the margin is positive on the whole box.
    Closed,
    /// Not decided.
    Open(OpenReason),
}

/// Which dual point certified a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Candidate {
    /// The warm start inherited from the parent.
    Warm,
    /// The warm start with slack cuts' multipliers set to 0.
    Clamped,
    /// All multipliers 0 (the box-corner floor).
    Floor,
    /// The clamped far probe.
    Far,
    /// An untrusted breakpoint hint re-evaluated by the certified scalar evaluator.
    Breakpoint,
    /// A scoped pair-phase multiplier proposal independently re-evaluated.
    Relation,
}

/// Search options of one objective.
#[derive(Debug, Clone, Copy)]
pub struct SearchOptions {
    /// Apply bounded per-node automatic refinement admission and search slices.
    /// Explicit research callers default to the unrestricted legacy policy.
    pub automatic_refine: bool,
    /// Untrusted breakpoint refinement steps per open node (0 = disabled, max 32).
    pub refine_steps: usize,
    /// Stop OPEN once this passes.
    pub deadline: Option<Instant>,
    /// Stop OPEN before a level that would take the evaluated-node count
    /// above this.
    pub node_cap: u64,
    /// Configured ceiling of one level's working set, clamped per level by
    /// [`admitted_bytes`].
    pub max_frontier_bytes: u64,
    /// Nodes per parallel work item (>= 1).
    pub chunk: usize,
    /// Record up to this many pruned nodes (their sides and the certifying
    /// multipliers) for independent re-verification. Test hook; 0 in the lane.
    pub record_prunes: usize,
    /// The node evaluator ([`FROZEN_DUAL_DEFAULT_EVALUATOR`] by default).
    pub evaluator: Evaluator,
    /// Block path only: re-check one prune in this many (hash-selected by
    /// depth and level index, deterministic and thread-independent), besides every
    /// prune at depth `<= 4`, against the scalar `eval_node`; `0` = off,
    /// `1` = every prune.
    pub crosscheck_every: u32,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            automatic_refine: false,
            refine_steps: 0,
            deadline: None,
            node_cap: FROZEN_DUAL_DEFAULT_NODE_CAP,
            max_frontier_bytes: FROZEN_DUAL_DEFAULT_MAX_STATE_BYTES,
            chunk: FROZEN_DUAL_CHUNK,
            record_prunes: 0,
            evaluator: FROZEN_DUAL_DEFAULT_EVALUATOR,
            crosscheck_every: FROZEN_DUAL_CROSSCHECK_EVERY,
        }
    }
}

/// One pruned node, exactly as it was certified.
#[derive(Debug, Clone, PartialEq)]
pub struct PrunedNode {
    /// Sides along the split order.
    pub sides: Vec<Side>,
    /// The `L0` handed to the certifying evaluation.
    pub l0: Vec<f64>,
    /// The `L1` handed to the certifying evaluation.
    pub l1: Vec<f64>,
    /// The certified value (`> 0`).
    pub certified: f64,
    /// Which dual point it was.
    pub candidate: Candidate,
}

/// One BFS level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LevelTrace {
    /// Depth (number of splits) of every node of the level.
    pub depth: usize,
    /// Nodes evaluated.
    pub frontier: u64,
    /// Nodes certified closed.
    pub pruned: u64,
}

/// What one objective's search did.
#[derive(Debug, Clone)]
pub struct SearchReport {
    /// Checked-relation proposal work; the evaluator retains pruning authority.
    pub relations: RelationStats,
    /// Untrusted support compilation/work receipt, never pruning evidence.
    pub support: ny_core::search::SupportSearchStats,
    /// Open nodes skipped by automatic refinement admission or its pilot stop.
    pub refine_skipped: u64,
    /// Nodes offered to breakpoint search.
    pub refine_calls: u64,
    /// Improving steps reported by search (telemetry only).
    pub refine_steps: u64,
    /// Hints that improved the independently certified bound.
    pub refine_accepted: u64,
    /// Open nodes closed by independently certified breakpoint hints.
    pub refine_prunes: u64,
    /// Backend refusals, panics or malformed hints.
    pub refine_refused: u64,
    /// The verdict.
    pub verdict: SearchVerdict,
    /// Certified root value (depth 0).
    pub root_certified: f64,
    /// Nodes evaluated (root excluded).
    pub nodes: u64,
    /// Deepest level evaluated.
    pub depth: usize,
    /// Nodes certified closed.
    pub certified_prunes: u64,
    /// Smallest certified value among the prunes (`+inf` when none).
    pub cert_min_slack: f64,
    /// Nodes where the clamped far probe was the best raw value.
    pub far_used: u64,
    /// Largest level evaluated.
    pub peak_frontier: u64,
    /// Smallest raw value among the last level's open nodes (search signal).
    pub worst_open: f64,
    /// Every level, in order.
    pub levels: Vec<LevelTrace>,
    /// Recorded prunes (see [`SearchOptions::record_prunes`]).
    pub pruned: Vec<PrunedNode>,
    /// Wall seconds of the search alone (the state build, the order and the
    /// problem substitution precede it and are NOT included).
    pub secs: f64,
    /// Decades of the pruned nodes' certified values (see `SLACK_DECADES`):
    /// how close the closing margins sit to the certificate's charge.
    pub slack_hist: [u64; SLACK_BUCKETS],
    /// Prunes re-checked against the scalar evaluator (block path).
    pub crosschecks: u64,
    /// Re-checks whose bits differed (a soundness finding; the search then
    /// stops OPEN).
    pub crosscheck_failures: u64,
    /// [`Evaluator::token`] of the evaluator that ran.
    pub evaluator: &'static str,
    /// The block evaluator ran far-first.
    pub far_first: bool,
    /// M4 `#wide-fd-search`: which search scored the nodes — `f64` (this
    /// module's own evaluators, the default) or the name of the injected
    /// UNTRUSTED backend (`cpu_f32`, `cuda`) whose proposals were certified.
    pub search: &'static str,
    /// M4: the backend's lane-start known-answer probe (`ok`, `fail`,
    /// `refused`), `-` when no backend was consulted.
    pub probe: &'static str,
    /// M4: nodes whose search value claimed a certificate (`v > -tau`).
    pub claimed: u64,
    /// M4: claimed nodes that no proposed point certified: kept and SPLIT,
    /// never pruned.
    pub refuted: u64,
    /// M4: level chunks the backend refused (or panicked on) and the f64
    /// evaluator scored instead.
    pub fallback_chunks: u64,
}

impl SearchReport {
    /// An empty report (`Open(Refused)`, no level). `pub(crate)` so the
    /// `certify` adapter tests can build reports field-robustly.
    pub(crate) fn empty() -> Self {
        Self {
            relations: RelationStats::default(),
            support: ny_core::search::SupportSearchStats::default(),
            refine_skipped: 0,
            refine_calls: 0,
            refine_steps: 0,
            refine_accepted: 0,
            refine_prunes: 0,
            refine_refused: 0,
            verdict: SearchVerdict::Open(OpenReason::Refused),
            root_certified: f64::NEG_INFINITY,
            nodes: 0,
            depth: 0,
            certified_prunes: 0,
            cert_min_slack: f64::INFINITY,
            far_used: 0,
            peak_frontier: 0,
            worst_open: f64::NEG_INFINITY,
            levels: Vec::new(),
            pruned: Vec::new(),
            secs: 0.0,
            slack_hist: [0; SLACK_BUCKETS],
            crosschecks: 0,
            crosscheck_failures: 0,
            evaluator: "-",
            far_first: false,
            search: "f64",
            probe: "-",
            claimed: 0,
            refuted: 0,
            fallback_chunks: 0,
        }
    }

    /// Search-only rate: nodes per search second (`0` when no time passed).
    #[must_use]
    pub fn search_nps(&self) -> f64 {
        if self.secs > 0.0 {
            #[allow(clippy::cast_precision_loss)]
            let nodes = self.nodes as f64;
            nodes / self.secs
        } else {
            0.0
        }
    }

    /// True iff the objective is certified closed.
    #[must_use]
    pub fn closed(&self) -> bool {
        self.verdict == SearchVerdict::Closed
    }
}

/// The LEVEL LEDGER: true iff the recorded levels describe a complete binary
/// split tree whose every leaf was pruned — level 1 holds the root's two
/// children, level `d + 1` holds exactly twice level `d`'s unpruned nodes,
/// depths are consecutive, and the last level is fully pruned. Equivalent to
/// the exact Kraft equality over the pruned leaves; checked with overflow-safe
/// integers. An empty ledger never closes (a root closure is decided by the
/// root's own certified value, not here).
#[must_use]
pub fn ledger_closes(levels: &[LevelTrace]) -> bool {
    let Some(first) = levels.first() else {
        return false;
    };
    if first.depth != 1 || first.frontier != 2 {
        return false;
    }
    for pair in levels.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if b.depth != a.depth + 1 || a.pruned > a.frontier {
            return false;
        }
        match (a.frontier - a.pruned).checked_mul(2) {
            Some(next) if next == b.frontier => {}
            _ => return false,
        }
    }
    let last = levels[levels.len() - 1];
    last.frontier > 0 && last.pruned == last.frontier
}

fn past(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|d| Instant::now() >= d)
}

/// Stored multiplier -> evaluator input (non-negative, finite).
fn widen(v: f32) -> f64 {
    let x = f64::from(v);
    if x.is_finite() && x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Evaluator multiplier -> stored search state (non-negative, finite). Any
/// non-negative value is a valid dual point, so the `f32` rounding is free.
fn narrow(v: f64) -> f32 {
    if v.is_nan() || v <= 0.0 {
        0.0
    } else if v >= f64::from(f32::MAX) {
        f32::MAX
    } else {
        #[allow(clippy::cast_possible_truncation)]
        let x = v as f32;
        x
    }
}

/// One BFS level: `count` nodes of `depth` splits, row-major.
struct Level {
    depth: usize,
    count: usize,
    /// `0` = `Off`, `1` = `On`.
    sides: Vec<u8>,
    l0: Vec<f32>,
    l1: Vec<f32>,
}

/// Worst-case bytes of evaluating one level and building the next: the level
/// itself, every node's warm start kept, the next level if nothing prunes, and
/// the per-thread scratch (the node evaluator's AND the block evaluator's,
/// whichever runs, so both evaluators stop on memory at the same level).
pub(crate) fn level_bytes(count: usize, depth: usize, n: usize) -> u64 {
    let c = count as u64;
    let d = depth as u64;
    let current = c.saturating_mul(d).saturating_mul(9);
    let kept = c.saturating_mul(d.saturating_mul(8).saturating_add(8));
    let next = c
        .saturating_mul(2)
        .saturating_mul(d.saturating_add(1))
        .saturating_mul(9);
    let threads = rayon::current_num_threads().max(1) as u64;
    let work = threads.saturating_mul(
        (n as u64)
            .saturating_mul(8 * 3)
            .saturating_add(d.saturating_mul(8 * 16))
            .saturating_add(LOG_BUCKETS as u64 * 8)
            .saturating_add(block_work_bytes(n, depth)),
    );
    current
        .saturating_add(kept)
        .saturating_add(next)
        .saturating_add(work)
}

/// Per-thread upper bound on the block evaluator's buffers (`BlockWork` and
/// its `BlockScratch`, which allocate exactly once at these sizes): `3 B n`
/// floats (the warm points' `rc` rows, the `da` rows, `d_base`), `20 B D`
/// floats (the thirteen structure-of-arrays rows, the gathered lanes, the lane
/// multipliers and accumulators), `2 D` for the floor point and the row
/// kinds, plus the fixed column tile, the line-search buckets and the per-node
/// flags. `fd_block_level_bytes_covers_block_scratch` pins it against the
/// measured capacities.
pub(crate) fn block_work_bytes(n: usize, depth: usize) -> u64 {
    let b = BLOCK_NODES as u64;
    let n = n as u64;
    let d = depth as u64;
    b.saturating_mul(n)
        .saturating_mul(3 * 8)
        .saturating_add(b.saturating_mul(d).saturating_mul(20 * 8))
        .saturating_add(d.saturating_mul(2 * 8))
        .saturating_add((BLOCK_COLS as u64 * b + LOG_BUCKETS as u64) * 8 + 16 * 1024)
}

/// Per-thread buffers of the node evaluation.
struct Work {
    scratch: NodeScratch,
    sides: Vec<Side>,
    l0: Vec<f64>,
    l1: Vec<f64>,
    s0: Vec<f64>,
    s1: Vec<f64>,
    c0: Vec<f64>,
    c1: Vec<f64>,
    zero: Vec<f64>,
    f0: Vec<f64>,
    f1: Vec<f64>,
    sp0: Vec<f64>,
    sp1: Vec<f64>,
    next0: Vec<f64>,
    next1: Vec<f64>,
    rc: Vec<f64>,
    da: Vec<f64>,
    buckets: Vec<f64>,
}

impl Work {
    fn new(n: usize, depth: usize) -> Self {
        let dz = || vec![0.0f64; depth];
        Self {
            scratch: NodeScratch::default(),
            sides: vec![Side::Off; depth],
            l0: dz(),
            l1: dz(),
            s0: dz(),
            s1: dz(),
            c0: dz(),
            c1: dz(),
            zero: dz(),
            f0: dz(),
            f1: dz(),
            sp0: dz(),
            sp1: dz(),
            next0: dz(),
            next1: dz(),
            rc: Vec::with_capacity(n),
            da: vec![0.0f64; n],
            buckets: vec![0.0f64; LOG_BUCKETS],
        }
    }

    /// The multipliers `candidate` was evaluated at.
    fn multipliers(&self, candidate: Candidate) -> (&[f64], &[f64]) {
        match candidate {
            Candidate::Warm => (&self.l0, &self.l1),
            Candidate::Clamped => (&self.c0, &self.c1),
            Candidate::Floor => (&self.zero, &self.zero),
            Candidate::Far => (&self.f0, &self.f1),
            Candidate::Breakpoint | Candidate::Relation => (&self.next0, &self.next1),
        }
    }
}

/// One node's evaluation.
#[derive(Debug, Clone, Copy)]
struct NodeOut {
    /// Largest raw value of the four dual points (search signal).
    best: f64,
    /// Largest certified value (the ONLY prune authority).
    certified: f64,
    /// The dual point that achieved `certified`.
    winner: Candidate,
    /// The far probe had the largest raw value.
    far_used: bool,
}

impl NodeOut {
    /// A refused node: never pruned.
    fn refused() -> Self {
        Self {
            best: f64::NEG_INFINITY,
            certified: f64::NEG_INFINITY,
            winner: Candidate::Warm,
            far_used: false,
        }
    }

    /// THE prune predicate of the search, on both evaluators: the node's
    /// largest CERTIFIED value is `> 0`, strict. The only place this file
    /// compares a certified value against zero
    /// (`fd_prune_predicate_is_strict_and_unique` pins it).
    fn prunes(&self) -> bool {
        self.certified > 0.0
    }
}

/// `(eta, crossed, emax)`: the log-bucket line search of the replica's
/// `node_bound_logbucket` along the direction `da` in `rc` space, starting
/// with directional derivative `slope0`. SEARCH SIGNAL ONLY.
fn log_bucket_step(rc: &[f64], da: &[f64], slope0: f64, buckets: &mut [f64]) -> (f64, bool, f64) {
    let positive = |r: f64, v: f64| (v > 0.0 && r < 0.0) || (v < 0.0 && r > 0.0);
    let mut emin = f64::INFINITY;
    let mut emax = f64::NEG_INFINITY;
    for (&r, &v) in rc.iter().zip(da) {
        if positive(r, v) {
            let e = -r / v;
            emin = emin.min(e);
            emax = emax.max(e);
        }
    }
    let emin = emin.max(1e-12);
    let emax = emax.max(1e-12);
    let le = emin.ln();
    #[allow(clippy::cast_precision_loss)]
    let lw = (emax.ln() - le).max(1e-30) / LOG_BUCKETS as f64;
    buckets.fill(0.0);
    for (&r, &v) in rc.iter().zip(da) {
        if positive(r, v) {
            let e = -r / v;
            let raw = ((e.max(1e-12)).ln() - le) / lw;
            let f = raw.floor();
            #[allow(clippy::cast_precision_loss)]
            let top = (LOG_BUCKETS - 1) as f64;
            let idx = if f.is_nan() || f <= 0.0 {
                0
            } else if f >= top {
                LOG_BUCKETS - 1
            } else {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let i = f as usize;
                i
            };
            buckets[idx] += 2.0 * v.abs();
        }
    }
    let mut cum = 0.0f64;
    let mut first = None;
    for (k, &b) in buckets.iter().enumerate() {
        cum += b;
        if cum >= slope0 {
            first = Some(k);
            break;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let eta = first.map_or(emax, |k| (le + k as f64 * lw).exp());
    let eta = if eta.is_finite() { eta } else { 0.0 };
    (eta, first.is_some(), emax)
}

/// Evaluate the node in `w.sides` / `w.l0` / `w.l1` (see the module docs) and
/// write its children's warm start into `w.next0` / `w.next1`.
fn eval_search_node(prob: &FrozenDualProblem, w: &mut Work) -> NodeOut {
    let d = w.sides.len();
    let warm = prob.eval_node(
        &w.sides,
        &w.l0,
        &w.l1,
        &mut w.scratch,
        Some(Subgradient {
            s0: &mut w.s0,
            s1: &mut w.s1,
        }),
    );
    if !warm.value.is_finite() {
        // Refused (fail-closed): never pruned; the children inherit the
        // parent's multipliers unchanged.
        w.next0.copy_from_slice(&w.l0);
        w.next1.copy_from_slice(&w.l1);
        return NodeOut {
            best: f64::NEG_INFINITY,
            certified: f64::NEG_INFINITY,
            winner: Candidate::Warm,
            far_used: false,
        };
    }
    w.rc.clear();
    w.rc.extend_from_slice(w.scratch.rc());

    // (2) multipliers of cuts slack at the minimiser set to 0.
    let mut clamped_differs = false;
    for i in 0..d {
        let a = if w.s0[i] < 0.0 { 0.0 } else { w.l0[i] };
        let b = if w.s1[i] < 0.0 { 0.0 } else { w.l1[i] };
        clamped_differs |= a.to_bits() != w.l0[i].to_bits() || b.to_bits() != w.l1[i].to_bits();
        w.c0[i] = a;
        w.c1[i] = b;
    }
    let clamped = if clamped_differs {
        prob.eval_node(&w.sides, &w.c0, &w.c1, &mut w.scratch, None)
    } else {
        warm
    };
    // (3) the box-corner floor.
    let any_mult = w.l0.iter().chain(&w.l1).any(|v| v.to_bits() != 0);
    let floor = if any_mult {
        prob.eval_node(&w.sides, &w.zero, &w.zero, &mut w.scratch, None)
    } else {
        warm
    };
    let pre_best = warm.value.max(clamped.value).max(floor.value);
    let pending = pre_best <= SEARCH_TOL;

    // Projected supergradient and its directional derivative.
    let mut slope0 = 0.0f64;
    for i in 0..d {
        let p0 = if w.l0[i] <= SEARCH_TOL && w.s0[i] < 0.0 {
            0.0
        } else {
            w.s0[i]
        };
        let p1 = if w.l1[i] <= SEARCH_TOL && w.s1[i] < 0.0 {
            0.0
        } else {
            w.s1[i]
        };
        w.sp0[i] = p0;
        w.sp1[i] = p1;
        slope0 += p0 * w.s0[i] + p1 * w.s1[i];
    }
    // Its direction in rc space: da = sum_i sign_i (sp0_i - sp1_i) a_i.
    w.da.clear();
    w.da.resize(prob.n(), 0.0);
    for i in 0..d {
        let m = w.sides[i].sign() * (w.sp0[i] - w.sp1[i]);
        if m != 0.0 {
            let row = prob.split_row(i).a;
            for (acc, &v) in w.da.iter_mut().zip(row) {
                *acc += m * v;
            }
        }
    }
    let (eta, crossed, emax) = log_bucket_step(&w.rc, &w.da, slope0, &mut w.buckets);

    // (4) the clamped far probe.
    let mut ebnd = f64::INFINITY;
    for i in 0..d {
        if w.sp0[i] < -SEARCH_TOL {
            ebnd = ebnd.min(-w.l0[i] / w.sp0[i]);
        }
        if w.sp1[i] < -SEARCH_TOL {
            ebnd = ebnd.min(-w.l1[i] / w.sp1[i]);
        }
    }
    let far = if ebnd.is_finite() { ebnd } else { emax * 1e6 };
    let mut eta_c = if crossed { eta } else { far };
    if !(slope0 > SEARCH_TOL && eta_c.is_finite()) {
        eta_c = 0.0;
    }
    let eta_c = eta_c.max(0.0);
    let mut far_differs = false;
    for i in 0..d {
        let a = (w.l0[i] + eta_c * w.sp0[i]).max(0.0);
        let b = (w.l1[i] + eta_c * w.sp1[i]).max(0.0);
        far_differs |= a.to_bits() != w.l0[i].to_bits() || b.to_bits() != w.l1[i].to_bits();
        w.f0[i] = a;
        w.f1[i] = b;
    }
    let far_nb = if far_differs {
        prob.eval_node(&w.sides, &w.f0, &w.f1, &mut w.scratch, None)
    } else {
        warm
    };
    let best = pre_best.max(far_nb.value);
    let far_used = far_nb.value > pre_best;

    // The children's warm start: one line-search step, only while the node is
    // not already raw-closed (the replica's `pend`).
    let eta_s = if pending && slope0 > SEARCH_TOL {
        eta.max(0.0)
    } else {
        0.0
    };
    for i in 0..d {
        w.next0[i] = (w.l0[i] + eta_s * w.sp0[i]).max(0.0);
        w.next1[i] = (w.l1[i] + eta_s * w.sp1[i]).max(0.0);
    }

    let mut winner = Candidate::Warm;
    let mut certified = f64::NEG_INFINITY;
    for (candidate, nb) in [
        (Candidate::Warm, warm),
        (Candidate::Clamped, clamped),
        (Candidate::Floor, floor),
        (Candidate::Far, far_nb),
    ] {
        if nb.certified > certified {
            certified = nb.certified;
            winner = candidate;
        }
    }
    NodeOut {
        best,
        certified,
        winner,
        far_used,
    }
}

// ---------------------------------------------------------------------------
// `#fd-block-eval`: the block evaluation of a work item
// ---------------------------------------------------------------------------

/// Per-thread buffers of the block evaluation: the [`Work`] rows as
/// `BLOCK_NODES x depth` structure-of-arrays, the warm points' `rc` rows (in
/// `ws`) and line-search directions, and the lanes the far, clamped and floor
/// points are gathered into. Every buffer is allocated once, at its final
/// size (`block_work_bytes` bounds the total).
struct BlockWork {
    n: usize,
    depth: usize,
    /// Nodes loaded (`<= BLOCK_NODES`).
    count: usize,
    ws: BlockScratch,
    sides: Vec<Side>,
    l0: Vec<f64>,
    l1: Vec<f64>,
    s0: Vec<f64>,
    s1: Vec<f64>,
    c0: Vec<f64>,
    c1: Vec<f64>,
    f0: Vec<f64>,
    f1: Vec<f64>,
    sp0: Vec<f64>,
    sp1: Vec<f64>,
    next0: Vec<f64>,
    next1: Vec<f64>,
    /// Line-search direction multipliers `sign (sp0 - sp1)`.
    dam: Vec<f64>,
    /// The floor point's multipliers (`depth` zeros).
    zero: Vec<f64>,
    /// Line-search directions, `BLOCK_NODES x n`.
    da: Vec<f64>,
    buckets: Vec<f64>,
    warm: Vec<NodeBound>,
    far: Vec<NodeBound>,
    /// `None` = not evaluated (far-first skipped it on a pruned node).
    clamped: Vec<Option<NodeBound>>,
    floor: Vec<Option<NodeBound>>,
    active: Vec<bool>,
    clamped_differs: Vec<bool>,
    any_mult: Vec<bool>,
    far_differs: Vec<bool>,
    slope0: Vec<f64>,
    eta: Vec<f64>,
    /// The far step actually taken (read only by the U4b pin's probe).
    #[cfg(test)]
    eta_c: Vec<f64>,
    outs: Vec<NodeOut>,
    /// Gathered lanes (at most two per node: clamped and floor).
    g_sides: Vec<Side>,
    g_l0: Vec<f64>,
    g_l1: Vec<f64>,
    g_out: Vec<NodeBound>,
    g_who: Vec<(usize, Candidate)>,
}

impl BlockWork {
    fn new(n: usize, depth: usize) -> Self {
        let b = BLOCK_NODES;
        let bd = || vec![0.0f64; b * depth];
        Self {
            n,
            depth,
            count: 0,
            ws: BlockScratch::with_capacity(n, depth, b),
            sides: vec![Side::Off; b * depth],
            l0: bd(),
            l1: bd(),
            s0: bd(),
            s1: bd(),
            c0: bd(),
            c1: bd(),
            f0: bd(),
            f1: bd(),
            sp0: bd(),
            sp1: bd(),
            next0: bd(),
            next1: bd(),
            dam: bd(),
            zero: vec![0.0f64; depth],
            da: vec![0.0f64; b * n],
            buckets: vec![0.0f64; LOG_BUCKETS],
            warm: vec![NodeBound::refused(); b],
            far: vec![NodeBound::refused(); b],
            clamped: vec![None; b],
            floor: vec![None; b],
            active: vec![false; b],
            clamped_differs: vec![false; b],
            any_mult: vec![false; b],
            far_differs: vec![false; b],
            slope0: vec![0.0f64; b],
            eta: vec![0.0f64; b],
            #[cfg(test)]
            eta_c: vec![0.0f64; b],
            outs: vec![NodeOut::refused(); b],
            g_sides: vec![Side::Off; 2 * b * depth],
            g_l0: vec![0.0f64; 2 * b * depth],
            g_l1: vec![0.0f64; 2 * b * depth],
            g_out: vec![NodeBound::refused(); 2 * b],
            g_who: vec![(0, Candidate::Warm); 2 * b],
        }
    }

    /// The multipliers node `b` was evaluated at for `candidate`.
    fn multipliers(&self, b: usize, candidate: Candidate) -> (&[f64], &[f64]) {
        let r = b * self.depth..(b + 1) * self.depth;
        match candidate {
            Candidate::Warm => (&self.l0[r.clone()], &self.l1[r]),
            Candidate::Clamped => (&self.c0[r.clone()], &self.c1[r]),
            Candidate::Floor => (&self.zero[..], &self.zero[..]),
            Candidate::Far => (&self.f0[r.clone()], &self.f1[r]),
            Candidate::Breakpoint | Candidate::Relation => (&self.next0[r.clone()], &self.next1[r]),
        }
    }

    /// Copy node `b`'s sides and its `which` multipliers into gather lane
    /// `lane`.
    fn gather(&mut self, lane: usize, b: usize, which: Candidate) {
        let d = self.depth;
        let src = b * d..(b + 1) * d;
        let dst = lane * d..(lane + 1) * d;
        self.g_sides[dst.clone()].copy_from_slice(&self.sides[src.clone()]);
        let (m0, m1): (&[f64], &[f64]) = match which {
            Candidate::Warm => (&self.l0[src.clone()], &self.l1[src]),
            Candidate::Clamped => (&self.c0[src.clone()], &self.c1[src]),
            Candidate::Floor => (&self.zero[..], &self.zero[..]),
            Candidate::Far => (&self.f0[src.clone()], &self.f1[src]),
            Candidate::Breakpoint | Candidate::Relation => {
                (&self.next0[src.clone()], &self.next1[src])
            }
        };
        self.g_l0[dst.clone()].copy_from_slice(m0);
        self.g_l1[dst].copy_from_slice(m1);
        self.g_who[lane] = (b, which);
    }

    /// Evaluate the first `lanes` gathered lanes (no `rc`, no subgradient)
    /// into `g_out`.
    fn eval_gathered(&mut self, prob: &FrozenDualProblem, lanes: usize, hooks: TestHooks) {
        if lanes == 0 {
            return;
        }
        let gd = lanes * self.depth;
        block_eval(
            prob,
            hooks,
            self.depth,
            &self.g_sides[..gd],
            &self.g_l0[..gd],
            &self.g_l1[..gd],
            BlockWant {
                rc: false,
                subgrad: false,
            },
            &mut self.ws,
            &mut [],
            &mut [],
            &mut self.g_out[..lanes],
        );
    }

    /// Keep node `b`'s far step for the U4b pin's probe (test builds only).
    #[cfg(test)]
    fn note_eta_c(&mut self, b: usize, eta_c: f64) {
        self.eta_c[b] = eta_c;
    }

    #[cfg(not(test))]
    #[inline]
    #[allow(clippy::unused_self)]
    fn note_eta_c(&mut self, _b: usize, _eta_c: f64) {}

    /// Mark every loaded node refused (fail closed: never pruned, children
    /// inherit the parent's multipliers).
    fn refuse_all(&mut self) {
        let d = self.depth;
        for b in 0..self.count {
            let r = b * d..(b + 1) * d;
            self.active[b] = false;
            self.next0[r.clone()].copy_from_slice(&self.l0[r.clone()]);
            self.next1[r.clone()].copy_from_slice(&self.l1[r]);
            self.outs[b] = NodeOut::refused();
        }
    }

    /// Bytes reserved by every buffer (see `block_work_bytes`).
    #[cfg(test)]
    fn capacity_bytes(&self) -> u64 {
        #[allow(clippy::ptr_arg)]
        fn bytes<T>(v: &Vec<T>) -> u64 {
            (v.capacity() as u64).saturating_mul(size_of::<T>() as u64)
        }
        let mut total = self.ws.capacity_bytes();
        for v in [
            &self.l0,
            &self.l1,
            &self.s0,
            &self.s1,
            &self.c0,
            &self.c1,
            &self.f0,
            &self.f1,
            &self.sp0,
            &self.sp1,
            &self.next0,
            &self.next1,
            &self.dam,
            &self.zero,
            &self.da,
            &self.buckets,
            &self.slope0,
            &self.eta,
            &self.eta_c,
            &self.g_l0,
            &self.g_l1,
        ] {
            total = total.saturating_add(bytes(v));
        }
        for v in [&self.sides, &self.g_sides] {
            total = total.saturating_add(bytes(v));
        }
        for v in [&self.warm, &self.far, &self.g_out] {
            total = total.saturating_add(bytes(v));
        }
        for v in [&self.clamped, &self.floor] {
            total = total.saturating_add(bytes(v));
        }
        for v in [
            &self.active,
            &self.clamped_differs,
            &self.any_mult,
            &self.far_differs,
        ] {
            total = total.saturating_add(bytes(v));
        }
        total
            .saturating_add(bytes(&self.outs))
            .saturating_add(bytes(&self.g_who))
    }
}

/// Phase clock of the block evaluation (W, P, DA, LS, F, C). Test builds time
/// the phases for the integrator's rate probe; everywhere else it is a
/// zero-sized no-op, so no timer compiles into the lane.
#[cfg(test)]
pub(crate) mod phase_probe {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::Instant;

    /// Phase names, in order.
    #[cfg(feature = "benchmarks")]
    pub(crate) const PHASES: [&str; 6] = ["W", "P", "DA", "LS", "F", "C"];

    static ON: AtomicBool = AtomicBool::new(false);
    static NS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

    /// Arm (and zero) or disarm the phase timers (process-wide). Only the
    /// `benchmarks`-gated rate probe arms them.
    #[cfg(feature = "benchmarks")]
    pub(crate) fn enable(on: bool) {
        if on {
            for c in &NS {
                c.store(0, Ordering::SeqCst);
            }
        }
        ON.store(on, Ordering::SeqCst);
    }

    /// Accumulated thread-seconds per phase.
    #[cfg(feature = "benchmarks")]
    pub(crate) fn read_secs() -> [f64; 6] {
        let mut out = [0.0f64; 6];
        for (o, c) in out.iter_mut().zip(&NS) {
            #[allow(clippy::cast_precision_loss)]
            let ns = c.load(Ordering::SeqCst) as f64;
            *o = ns * 1e-9;
        }
        out
    }

    pub(crate) struct Clock {
        last: Option<Instant>,
    }

    impl Clock {
        pub(crate) fn start() -> Self {
            Self {
                last: ON.load(Ordering::Relaxed).then(Instant::now),
            }
        }

        pub(crate) fn lap(&mut self, phase: usize) {
            if let Some(t) = self.last {
                let now = Instant::now();
                let ns = u64::try_from(now.duration_since(t).as_nanos()).unwrap_or(u64::MAX);
                NS[phase].fetch_add(ns, Ordering::Relaxed);
                self.last = Some(now);
            }
        }
    }
}

#[cfg(not(test))]
mod phase_probe {
    pub(crate) struct Clock;

    impl Clock {
        #[inline]
        pub(crate) fn start() -> Self {
            Self
        }

        #[inline]
        #[allow(clippy::unused_self)]
        pub(crate) fn lap(&mut self, _phase: usize) {}
    }
}

/// Test-only knobs of the block path (the mutation and fault-injection pins).
/// Outside test builds it is empty and every hook is a constant.
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default)]
struct TestHooks {
    /// Evaluate with the certificate DISABLED (`charge_on = false`).
    charge_off: bool,
    /// Add one ulp to every block node's certified value AFTER evaluation.
    fault_ulp: bool,
    /// M4 MUTANT (wide path): prune a claimed node on the backend's SEARCH
    /// value, skipping the f64 certification.
    claim_prunes: bool,
}

#[cfg(not(test))]
#[derive(Debug, Clone, Copy, Default)]
struct TestHooks;

impl TestHooks {
    /// Every hook off: the production configuration. One spelling for both
    /// shapes of the type (a field struct in test builds, a unit struct
    /// otherwise), so production call sites compile identically in both.
    #[cfg(test)]
    const OFF: Self = Self {
        charge_off: false,
        fault_ulp: false,
        claim_prunes: false,
    };
    #[cfg(not(test))]
    const OFF: Self = Self;

    #[cfg(test)]
    fn inject(self, outs: &mut [NodeOut]) {
        if self.fault_ulp {
            for o in outs {
                o.certified = next_up(o.certified);
            }
        }
    }

    #[cfg(not(test))]
    #[inline]
    #[allow(clippy::unused_self)]
    fn inject(self, _outs: &mut [NodeOut]) {}

    /// The wide path's claim-as-prune mutant (test builds only).
    #[cfg(test)]
    fn claim_prunes(self) -> bool {
        self.claim_prunes
    }

    #[cfg(not(test))]
    #[inline]
    #[allow(clippy::unused_self)]
    fn claim_prunes(self) -> bool {
        false
    }
}

/// The lane's block evaluation at the lane geometry: `eval_block`, certificate
/// ON. Only a test build's charge-off hook (the mutation pin) reaches
/// `eval_block_inner(.., false)`; no other build contains that call.
#[cfg(not(test))]
#[inline]
#[allow(clippy::too_many_arguments)]
fn block_eval(
    prob: &FrozenDualProblem,
    _hooks: TestHooks,
    depth: usize,
    sides: &[Side],
    l0: &[f64],
    l1: &[f64],
    want: BlockWant,
    ws: &mut BlockScratch,
    s0: &mut [f64],
    s1: &mut [f64],
    out: &mut [NodeBound],
) {
    prob.eval_block(depth, sides, l0, l1, BlockGeom::LANE, want, ws, s0, s1, out);
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn block_eval(
    prob: &FrozenDualProblem,
    hooks: TestHooks,
    depth: usize,
    sides: &[Side],
    l0: &[f64],
    l1: &[f64],
    want: BlockWant,
    ws: &mut BlockScratch,
    s0: &mut [f64],
    s1: &mut [f64],
    out: &mut [NodeBound],
) {
    if hooks.charge_off {
        prob.eval_block_inner(
            depth,
            sides,
            l0,
            l1,
            BlockGeom::LANE,
            want,
            ws,
            s0,
            s1,
            out,
            false,
        );
    } else {
        prob.eval_block(depth, sides, l0, l1, BlockGeom::LANE, want, ws, s0, s1, out);
    }
}

/// [`eval_search_node`] for the `bw.count` nodes loaded in `bw`, stage by
/// stage (W, P, DA, LS, F, C, out), every per-node formula copied from it and
/// every certified value from the bit-identical `eval_block`. Writes
/// `bw.outs` and each node's children's warm start `bw.next0` / `bw.next1`.
fn eval_search_block(
    prob: &FrozenDualProblem,
    bw: &mut BlockWork,
    far_first: bool,
    hooks: TestHooks,
) {
    let d = bw.depth;
    let n = bw.n;
    let cnt = bw.count;
    let nd = cnt * d;
    let mut clk = phase_probe::Clock::start();

    // W: the warm points, with their rc rows and subgradients.
    block_eval(
        prob,
        hooks,
        d,
        &bw.sides[..nd],
        &bw.l0[..nd],
        &bw.l1[..nd],
        BlockWant {
            rc: true,
            subgrad: true,
        },
        &mut bw.ws,
        &mut bw.s0[..nd],
        &mut bw.s1[..nd],
        &mut bw.warm[..cnt],
    );
    clk.lap(0);

    // P: clamped multipliers, the projected supergradient, the direction's
    // multipliers (`eval_search_node` (2) and its slope loop).
    for b in 0..cnt {
        let r = b * d..(b + 1) * d;
        let warm = bw.warm[b];
        bw.clamped[b] = None;
        bw.floor[b] = None;
        if !warm.value.is_finite() {
            // Refused (fail-closed): never pruned; the children inherit the
            // parent's multipliers unchanged.
            bw.active[b] = false;
            bw.next0[r.clone()].copy_from_slice(&bw.l0[r.clone()]);
            bw.next1[r.clone()].copy_from_slice(&bw.l1[r.clone()]);
            bw.dam[r].fill(0.0);
            bw.outs[b] = NodeOut::refused();
            continue;
        }
        bw.active[b] = true;
        let mut clamped_differs = false;
        for i in r.clone() {
            let a = if bw.s0[i] < 0.0 { 0.0 } else { bw.l0[i] };
            let c = if bw.s1[i] < 0.0 { 0.0 } else { bw.l1[i] };
            clamped_differs |=
                a.to_bits() != bw.l0[i].to_bits() || c.to_bits() != bw.l1[i].to_bits();
            bw.c0[i] = a;
            bw.c1[i] = c;
        }
        bw.clamped_differs[b] = clamped_differs;
        bw.any_mult[b] = bw.l0[r.clone()]
            .iter()
            .chain(&bw.l1[r.clone()])
            .any(|v| v.to_bits() != 0);
        let mut slope0 = 0.0f64;
        for i in r {
            let p0 = if bw.l0[i] <= SEARCH_TOL && bw.s0[i] < 0.0 {
                0.0
            } else {
                bw.s0[i]
            };
            let p1 = if bw.l1[i] <= SEARCH_TOL && bw.s1[i] < 0.0 {
                0.0
            } else {
                bw.s1[i]
            };
            bw.sp0[i] = p0;
            bw.sp1[i] = p1;
            slope0 += p0 * bw.s0[i] + p1 * bw.s1[i];
            bw.dam[i] = bw.sides[i].sign() * (bw.sp0[i] - bw.sp1[i]);
        }
        bw.slope0[b] = slope0;
    }
    clk.lap(1);

    // DA: every node's direction in rc space, da = sum_i sign_i (sp0_i -
    // sp1_i) a_i, from +0.0 in row order (bit-identical to the per-node loop).
    if !prob.block_axpy_rows(
        d,
        cnt,
        &bw.dam[..nd],
        BlockGeom::LANE,
        &mut bw.ws,
        &mut bw.da[..cnt * n],
    ) {
        // Unreachable by construction (the shapes are the block's own);
        // fail closed rather than step on a wrong direction.
        bw.refuse_all();
        return;
    }
    clk.lap(2);

    // LS and the far point (`eval_search_node` (4), per node).
    for b in 0..cnt {
        if !bw.active[b] {
            continue;
        }
        let r = b * d..(b + 1) * d;
        let (eta, crossed, emax) = log_bucket_step(
            bw.ws.rc_row(b, n),
            &bw.da[b * n..(b + 1) * n],
            bw.slope0[b],
            &mut bw.buckets,
        );
        let mut ebnd = f64::INFINITY;
        for i in r.clone() {
            if bw.sp0[i] < -SEARCH_TOL {
                ebnd = ebnd.min(-bw.l0[i] / bw.sp0[i]);
            }
            if bw.sp1[i] < -SEARCH_TOL {
                ebnd = ebnd.min(-bw.l1[i] / bw.sp1[i]);
            }
        }
        let far = if ebnd.is_finite() { ebnd } else { emax * 1e6 };
        let mut eta_c = if crossed { eta } else { far };
        if !(bw.slope0[b] > SEARCH_TOL && eta_c.is_finite()) {
            eta_c = 0.0;
        }
        let eta_c = eta_c.max(0.0);
        let mut far_differs = false;
        for i in r {
            let a = (bw.l0[i] + eta_c * bw.sp0[i]).max(0.0);
            let c = (bw.l1[i] + eta_c * bw.sp1[i]).max(0.0);
            far_differs |= a.to_bits() != bw.l0[i].to_bits() || c.to_bits() != bw.l1[i].to_bits();
            bw.f0[i] = a;
            bw.f1[i] = c;
        }
        bw.far_differs[b] = far_differs;
        bw.eta[b] = eta;
        bw.note_eta_c(b, eta_c);
    }
    clk.lap(3);

    // F: the far points that differ from warm, gathered into lanes and
    // evaluated at the CLAMPED far multipliers with rc recomputed (never the
    // incremental rc + eta da: DECISION.md U4b).
    let mut lanes = 0usize;
    for b in 0..cnt {
        if bw.active[b] && bw.far_differs[b] {
            bw.gather(lanes, b, Candidate::Far);
            lanes += 1;
        }
    }
    bw.eval_gathered(prob, lanes, hooks);
    for b in 0..cnt {
        if bw.active[b] {
            bw.far[b] = bw.warm[b];
        }
    }
    for k in 0..lanes {
        let (b, _) = bw.g_who[k];
        bw.far[b] = bw.g_out[k];
    }
    clk.lap(4);

    // C: the clamped and floor points. Far-first skips them on a node the warm
    // or far point already certifies closed (see the module docs); a point
    // equal to warm is warm, as in the reference.
    let mut lanes = 0usize;
    for b in 0..cnt {
        if !bw.active[b] {
            continue;
        }
        let warm = bw.warm[b];
        bw.clamped[b] = (!bw.clamped_differs[b]).then_some(warm);
        bw.floor[b] = (!bw.any_mult[b]).then_some(warm);
        if far_first && (bw.far[b].prunes() || warm.prunes()) {
            continue;
        }
        if bw.clamped_differs[b] {
            bw.gather(lanes, b, Candidate::Clamped);
            lanes += 1;
        }
        if bw.any_mult[b] {
            bw.gather(lanes, b, Candidate::Floor);
            lanes += 1;
        }
    }
    bw.eval_gathered(prob, lanes, hooks);
    for k in 0..lanes {
        let (b, which) = bw.g_who[k];
        match which {
            Candidate::Clamped => bw.clamped[b] = Some(bw.g_out[k]),
            Candidate::Floor => bw.floor[b] = Some(bw.g_out[k]),
            Candidate::Warm | Candidate::Far | Candidate::Breakpoint | Candidate::Relation => {}
        }
    }
    clk.lap(5);

    // Out: exactly `eval_search_node`'s tail over the points evaluated.
    for b in 0..cnt {
        if !bw.active[b] {
            continue;
        }
        let warm = bw.warm[b];
        let far_nb = bw.far[b];
        let clamped = bw.clamped[b];
        let floor = bw.floor[b];
        let mut pre_best = warm.value;
        if let Some(c) = clamped {
            pre_best = pre_best.max(c.value);
        }
        if let Some(f) = floor {
            pre_best = pre_best.max(f.value);
        }
        let pending = pre_best <= SEARCH_TOL;
        let best = pre_best.max(far_nb.value);
        let far_used = far_nb.value > pre_best;
        // The children's warm start (only a node no point pruned has
        // children, and such a node had every point evaluated).
        let eta_s = if pending && bw.slope0[b] > SEARCH_TOL {
            bw.eta[b].max(0.0)
        } else {
            0.0
        };
        for i in b * d..(b + 1) * d {
            bw.next0[i] = (bw.l0[i] + eta_s * bw.sp0[i]).max(0.0);
            bw.next1[i] = (bw.l1[i] + eta_s * bw.sp1[i]).max(0.0);
        }
        let mut winner = Candidate::Warm;
        let mut certified = f64::NEG_INFINITY;
        for (candidate, nb) in [
            (Candidate::Warm, Some(warm)),
            (Candidate::Clamped, clamped),
            (Candidate::Floor, floor),
            (Candidate::Far, Some(far_nb)),
        ] {
            if let Some(nb) = nb {
                if nb.certified > certified {
                    certified = nb.certified;
                    winner = candidate;
                }
            }
        }
        bw.outs[b] = NodeOut {
            best,
            certified,
            winner,
            far_used,
        };
    }
    hooks.inject(&mut bw.outs[..cnt]);
}

// ---------------------------------------------------------------------------
// Work items and the level loop
// ---------------------------------------------------------------------------

/// One work item's results.
struct ChunkOut {
    relations: RelationStats,
    refinement: breakpoint::Stats,
    aborted: bool,
    pruned: u64,
    kept: Vec<usize>,
    next_l0: Vec<f32>,
    next_l1: Vec<f32>,
    cert_min: f64,
    worst_open: f64,
    far_used: u64,
    records: Vec<PrunedNode>,
    slack_hist: [u64; SLACK_BUCKETS],
    /// Block path: prunes re-checked against the scalar evaluator.
    xchk: u64,
    /// ... and how many differed.
    xchk_fail: u64,
    /// M4: nodes whose search value claimed a certificate.
    claimed: u64,
    /// M4: claimed nodes that did not certify (kept and split).
    refuted: u64,
}

impl ChunkOut {
    fn new() -> Self {
        Self {
            relations: RelationStats::default(),
            refinement: breakpoint::Stats::default(),
            aborted: false,
            pruned: 0,
            kept: Vec::new(),
            next_l0: Vec::new(),
            next_l1: Vec::new(),
            cert_min: f64::INFINITY,
            worst_open: f64::INFINITY,
            far_used: 0,
            records: Vec::new(),
            slack_hist: [0; SLACK_BUCKETS],
            xchk: 0,
            xchk_fail: 0,
            claimed: 0,
            refuted: 0,
        }
    }
}

/// What every work item of one level shares.
#[derive(Clone, Copy)]
struct ChunkCtx<'a> {
    relations: Option<relations::LevelHints<'a>>,
    automatic_refine: bool,
    refine_steps: usize,
    backend: Option<&'a dyn SearchBackend>,
    deadline: Option<Instant>,
    record_cap: usize,
    evaluator: Evaluator,
    crosscheck_every: u32,
    hooks: TestHooks,
}

/// Copy an incumbent only for a checked current literal match. Nonmatching
/// nodes must not perform depth-sized copies outside the replay reservation.
fn relation_incumbent<'a, T>(
    relations: Option<relations::LevelHints<'a>>,
    sides: &[u8],
    copy: impl FnOnce() -> T,
) -> Option<(relations::LevelHints<'a>, T)> {
    let hints = relations.filter(|h| h.matches_bytes(sides))?;
    Some((hints, copy()))
}

/// A declined optional path retains its receipt without consumer storage.
fn active_relation_hints(
    mut hints: Option<relations::ClassHints>,
) -> (Option<relations::ClassHints>, RelationStats) {
    let stats = hints
        .as_mut()
        .map_or(RelationStats::default(), |h| std::mem::take(&mut h.stats));
    (hints.filter(|h| h.active()), stats)
}

fn refine_allowed(
    ctx: &ChunkCtx<'_>,
    n: usize,
    depth: usize,
    stats: &mut breakpoint::Stats,
) -> bool {
    if ctx.refine_steps == 0 || ctx.backend.is_none() {
        return false;
    }
    if ctx.automatic_refine && !automatic::allow_refine(n, depth, stats) {
        stats.skipped += 1;
        false
    } else {
        true
    }
}

/// Per-thread state of a work item: the node evaluator's buffers (also the
/// cross-check's scratch) and, on the block path, the block buffers.
struct ChunkWork {
    node: Work,
    block: Option<BlockWork>,
}

impl ChunkWork {
    fn new(n: usize, depth: usize, evaluator: Evaluator) -> Self {
        Self {
            node: Work::new(n, depth),
            block: match evaluator {
                Evaluator::Node => None,
                Evaluator::Block { .. } => Some(BlockWork::new(n, depth)),
            },
        }
    }
}

/// True iff the block path's prune at `depth`, level index `idx` is
/// re-checked: always at depth `<= 4`, else 1 in `every` by a splitmix64
/// finalisation of `(depth, idx)` (deterministic, independent of chunking and
/// threads).
///
/// NOT the bare multiplicative hash `idx * K % every`: `K` is odd, so for the
/// power-of-two default (256) that selects exactly `idx % 256 == 0` — only
/// EVEN indices, i.e. only `Off` children (a level lists each kept parent's
/// `Off` child then its `On` child), and only index 0 of every level narrower
/// than 256. The full avalanche reaches both sides of every split and varies
/// by level (`fd_block_crosscheck_sample_is_spread` pins it).
pub(crate) fn crosscheck_due(every: u32, depth: usize, idx: usize) -> bool {
    if every == 0 {
        return false;
    }
    if depth <= CROSSCHECK_DEPTH {
        return true;
    }
    let mut z = (idx as u64) ^ ((depth as u64) << 48);
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    z.is_multiple_of(u64::from(every))
}

fn eval_chunk(
    prob: &FrozenDualProblem,
    level: &Level,
    range: std::ops::Range<usize>,
    ctx: &ChunkCtx<'_>,
    w: &mut ChunkWork,
) -> ChunkOut {
    match (ctx.evaluator, w.block.as_mut()) {
        (Evaluator::Block { far_first }, Some(bw)) => {
            eval_chunk_block(prob, level, range, ctx, far_first, &mut w.node.scratch, bw)
        }
        _ => eval_chunk_node(prob, level, range, ctx, &mut w.node),
    }
}

/// The reference: one node at a time through [`eval_search_node`].
fn eval_chunk_node(
    prob: &FrozenDualProblem,
    level: &Level,
    range: std::ops::Range<usize>,
    ctx: &ChunkCtx<'_>,
    w: &mut Work,
) -> ChunkOut {
    let d = level.depth;
    let mut out = ChunkOut::new();
    for (k, idx) in range.enumerate() {
        if k % DEADLINE_STRIDE == 0 && past(ctx.deadline) {
            out.aborted = true;
            return out;
        }
        let base = idx * d;
        for i in 0..d {
            w.sides[i] = if level.sides[base + i] == 0 {
                Side::Off
            } else {
                Side::On
            };
            w.l0[i] = widen(level.l0[base + i]);
            w.l1[i] = widen(level.l1[base + i]);
        }
        let mut node = eval_search_node(prob, w);
        if !node.prunes() {
            if let Some((relations, (m0, m1))) =
                relation_incumbent(ctx.relations, &level.sides[base..base + d], || {
                    let (m0, m1) = w.multipliers(node.winner);
                    (m0.to_vec(), m1.to_vec())
                })
            {
                if let Some(hint) = relations.propose(
                    prob,
                    &w.sides,
                    &m0,
                    &m1,
                    node.certified,
                    ctx.deadline,
                    &mut w.scratch,
                    &mut out.relations,
                ) {
                    node.best = node.best.max(hint.bound.value);
                    node.certified = hint.bound.certified;
                    node.winner = Candidate::Relation;
                    w.next0.copy_from_slice(&hint.l0);
                    w.next1.copy_from_slice(&hint.l1);
                }
            }
        }
        if !node.prunes() && refine_allowed(ctx, prob.n(), d, &mut out.refinement) {
            let (m0, m1) = w.multipliers(node.winner);
            // Tiny depth-sized copies avoid aliasing the certifier scratch.
            let (m0, m1) = (m0.to_vec(), m1.to_vec());
            if let Some(hint) = breakpoint::refine_with_policy(
                prob,
                &w.sides,
                &m0,
                &m1,
                node.certified,
                ctx.refine_steps,
                ctx.automatic_refine,
                ctx.deadline,
                ctx.backend,
                &mut w.scratch,
                &mut out.refinement,
            ) {
                node.best = node.best.max(hint.bound.value);
                node.certified = hint.bound.certified;
                node.winner = breakpoint::CANDIDATE;
                w.next0.copy_from_slice(&hint.l0);
                w.next1.copy_from_slice(&hint.l1);
            }
        }
        if node.far_used {
            out.far_used += 1;
        }
        if node.prunes() {
            out.pruned += 1;
            out.cert_min = out.cert_min.min(node.certified);
            out.slack_hist[slack_decade(node.certified)] += 1;
            if out.records.len() < ctx.record_cap {
                let (a, b) = w.multipliers(node.winner);
                out.records.push(PrunedNode {
                    sides: w.sides.clone(),
                    l0: a.to_vec(),
                    l1: b.to_vec(),
                    certified: node.certified,
                    candidate: node.winner,
                });
            }
        } else {
            out.kept.push(idx);
            out.next_l0.extend(w.next0.iter().map(|&v| narrow(v)));
            out.next_l1.extend(w.next1.iter().map(|&v| narrow(v)));
            out.worst_open = out.worst_open.min(node.best);
        }
    }
    out
}

/// `#fd-block-eval`: the work item in blocks of `BLOCK_NODES` nodes, results
/// scattered back IN NODE ORDER (so `kept`, `next_l0` and `next_l1`, hence the
/// next level, are the reference's), the deadline checked before each block
/// (the reference's stride), and the sampled scalar cross-check on prunes.
fn eval_chunk_block(
    prob: &FrozenDualProblem,
    level: &Level,
    range: std::ops::Range<usize>,
    ctx: &ChunkCtx<'_>,
    far_first: bool,
    scratch: &mut NodeScratch,
    bw: &mut BlockWork,
) -> ChunkOut {
    let d = level.depth;
    let mut out = ChunkOut::new();
    let mut start = range.start;
    while start < range.end {
        if past(ctx.deadline) {
            out.aborted = true;
            return out;
        }
        let end = (start + BLOCK_NODES).min(range.end);
        bw.count = end - start;
        for (b, idx) in (start..end).enumerate() {
            let src = idx * d;
            let dst = b * d;
            for i in 0..d {
                bw.sides[dst + i] = if level.sides[src + i] == 0 {
                    Side::Off
                } else {
                    Side::On
                };
                bw.l0[dst + i] = widen(level.l0[src + i]);
                bw.l1[dst + i] = widen(level.l1[src + i]);
            }
        }
        eval_search_block(prob, bw, far_first, ctx.hooks);
        for (b, idx) in (start..end).enumerate() {
            let mut node = bw.outs[b];
            if !node.prunes() {
                if let Some(relations) = ctx
                    .relations
                    .filter(|h| h.matches_bytes(&level.sides[idx * d..(idx + 1) * d]))
                {
                    let sides = &bw.sides[b * d..(b + 1) * d];
                    let (m0, m1) = bw.multipliers(b, node.winner);
                    if let Some(hint) = relations.propose(
                        prob,
                        sides,
                        m0,
                        m1,
                        node.certified,
                        ctx.deadline,
                        scratch,
                        &mut out.relations,
                    ) {
                        node.best = node.best.max(hint.bound.value);
                        node.certified = hint.bound.certified;
                        node.winner = Candidate::Relation;
                        bw.next0[b * d..(b + 1) * d].copy_from_slice(&hint.l0);
                        bw.next1[b * d..(b + 1) * d].copy_from_slice(&hint.l1);
                    }
                }
            }
            if !node.prunes() && refine_allowed(ctx, prob.n(), d, &mut out.refinement) {
                let sides = &bw.sides[b * d..(b + 1) * d];
                let (m0, m1) = bw.multipliers(b, node.winner);
                if let Some(hint) = breakpoint::refine_with_policy(
                    prob,
                    sides,
                    m0,
                    m1,
                    node.certified,
                    ctx.refine_steps,
                    ctx.automatic_refine,
                    ctx.deadline,
                    ctx.backend,
                    scratch,
                    &mut out.refinement,
                ) {
                    node.best = node.best.max(hint.bound.value);
                    node.certified = hint.bound.certified;
                    node.winner = breakpoint::CANDIDATE;
                    bw.next0[b * d..(b + 1) * d].copy_from_slice(&hint.l0);
                    bw.next1[b * d..(b + 1) * d].copy_from_slice(&hint.l1);
                }
            }
            if node.far_used {
                out.far_used += 1;
            }
            if node.prunes() {
                out.pruned += 1;
                out.cert_min = out.cert_min.min(node.certified);
                out.slack_hist[slack_decade(node.certified)] += 1;
                let sides = &bw.sides[b * d..(b + 1) * d];
                let (m0, m1) = bw.multipliers(b, node.winner);
                if crosscheck_due(ctx.crosscheck_every, d, idx) {
                    out.xchk += 1;
                    let reference = prob.eval_node(sides, m0, m1, scratch, None);
                    if reference.certified.to_bits() != node.certified.to_bits() {
                        if out.xchk_fail == 0 {
                            eprintln!(
                                "[frozen-dual] CROSSCHECK-MISMATCH depth={d} idx={idx} \
                                 block={start} ref={:e} got={:e}",
                                reference.certified, node.certified
                            );
                        }
                        out.xchk_fail += 1;
                    }
                }
                if out.records.len() < ctx.record_cap {
                    out.records.push(PrunedNode {
                        sides: sides.to_vec(),
                        l0: m0.to_vec(),
                        l1: m1.to_vec(),
                        certified: node.certified,
                        candidate: node.winner,
                    });
                }
            } else {
                out.kept.push(idx);
                out.next_l0
                    .extend(bw.next0[b * d..(b + 1) * d].iter().map(|&v| narrow(v)));
                out.next_l1
                    .extend(bw.next1[b * d..(b + 1) * d].iter().map(|&v| narrow(v)));
                out.worst_open = out.worst_open.min(node.best);
            }
        }
        start = end;
    }
    out
}

/// The certified BFS over one substituted problem (see the module docs).
/// Returns `Closed` only when every leaf is certified and the level ledger
/// closes; every other exit is `Open` with its reason.
#[must_use]
pub fn search(prob: &FrozenDualProblem, opts: &SearchOptions) -> SearchReport {
    search_inner(prob, opts, TestHooks::OFF, None)
}

/// Search with an injected untrusted backend. A missing backend retains the
/// legacy search; no search score can certify a prune.
#[must_use]
pub fn search_with_refinement_backend(
    prob: &FrozenDualProblem,
    opts: &SearchOptions,
    backend: Option<&dyn SearchBackend>,
) -> SearchReport {
    search_inner_with_backend(prob, opts, TestHooks::OFF, None, backend)
}

/// [`search`] with the test-only hooks of the block path (the charge-off
/// mutation pin and the cross-check fault injection).
#[cfg(test)]
pub(crate) fn search_with_test_hooks(
    prob: &FrozenDualProblem,
    opts: &SearchOptions,
    charge_off: bool,
    fault_ulp: bool,
) -> SearchReport {
    search_inner(
        prob,
        opts,
        TestHooks {
            charge_off,
            fault_ulp,
            claim_prunes: false,
        },
        None,
    )
}

/// The level loop. `wide_ctx` is M4's backend context: when present, each level
/// is scored by the injected UNTRUSTED search and only its proposals are
/// certified here (`wide::eval_level`); `None` is the f64 search, unchanged.
/// Admission, the prune predicate, the ledger and the next level are the same
/// code either way.
fn search_inner(
    prob: &FrozenDualProblem,
    opts: &SearchOptions,
    hooks: TestHooks,
    wide_ctx: Option<&wide::WideCtx<'_>>,
) -> SearchReport {
    let backend = if opts.refine_steps > 0 {
        super::root_search::registered_backend()
    } else {
        None
    };
    search_inner_with_backend(prob, opts, hooks, wide_ctx, backend.as_deref())
}

fn search_inner_with_backend(
    prob: &FrozenDualProblem,
    opts: &SearchOptions,
    hooks: TestHooks,
    wide_ctx: Option<&wide::WideCtx<'_>>,
    backend: Option<&dyn SearchBackend>,
) -> SearchReport {
    search_inner_with_relations(prob, opts, hooks, wide_ctx, backend, None)
}

fn search_with_relations(
    prob: &FrozenDualProblem,
    opts: &SearchOptions,
    relations: Option<&relations::ClassHints>,
) -> SearchReport {
    let backend = if opts.refine_steps > 0 {
        super::root_search::registered_backend()
    } else {
        None
    };
    search_inner_with_relations(
        prob,
        opts,
        TestHooks::OFF,
        None,
        backend.as_deref(),
        relations,
    )
}

fn search_inner_with_relations(
    prob: &FrozenDualProblem,
    opts: &SearchOptions,
    hooks: TestHooks,
    wide_ctx: Option<&wide::WideCtx<'_>>,
    backend: Option<&dyn SearchBackend>,
    relations: Option<&relations::ClassHints>,
) -> SearchReport {
    search_loop(
        &mut ProblemSource::Borrowed(prob),
        opts,
        hooks,
        wide_ctx,
        backend,
        relations,
        None,
    )
}

/// Borrowed ordinary problems never change. Only the private progressive
/// owner can offer a new original-source problem; the level loop checks every
/// prior numerical field before replacing its owned problem.
#[expect(
    clippy::large_enum_variant,
    reason = "Keep inline ownership and accounted storage; boxing adds an allocation"
)]
enum ProblemSource<'a> {
    Borrowed(&'a FrozenDualProblem),
    Owned(FrozenDualProblem),
}
impl ProblemSource<'_> {
    fn problem(&self) -> &FrozenDualProblem {
        match self {
            Self::Borrowed(p) => p,
            Self::Owned(p) => p,
        }
    }
}

type AppendProblem<'a> =
    dyn FnMut(&FrozenDualProblem) -> Result<Option<FrozenDualProblem>, ZonoRefusal> + 'a;

/// One BFS and one ledger across checked append-only original row batches.
/// Dynamic extension has no wide/relation handle to retain from an earlier
/// problem. Ordinary callers use the same loop with a fixed borrowed problem.
fn search_growing(
    prob: FrozenDualProblem,
    opts: &SearchOptions,
    append: &mut AppendProblem<'_>,
) -> (FrozenDualProblem, SearchReport) {
    let backend = if opts.refine_steps > 0 {
        super::root_search::registered_backend()
    } else {
        None
    };
    let mut source = ProblemSource::Owned(prob);
    let report = search_loop(
        &mut source,
        opts,
        TestHooks::OFF,
        None,
        backend.as_deref(),
        None,
        Some(append),
    );
    let ProblemSource::Owned(problem) = source else {
        unreachable!("growing problem is owned")
    };
    (problem, report)
}

#[cfg(test)]
pub(crate) fn search_growing_for_test(
    prob: FrozenDualProblem,
    opts: &SearchOptions,
    append: &mut dyn FnMut(&FrozenDualProblem) -> Result<Option<FrozenDualProblem>, ZonoRefusal>,
) -> (FrozenDualProblem, SearchReport) {
    search_growing(prob, opts, append)
}

fn search_loop(
    source: &mut ProblemSource<'_>,
    opts: &SearchOptions,
    hooks: TestHooks,
    wide_ctx: Option<&wide::WideCtx<'_>>,
    backend: Option<&dyn SearchBackend>,
    relations: Option<&relations::ClassHints>,
    mut append: Option<&mut AppendProblem<'_>>,
) -> SearchReport {
    let prob = source.problem();
    let extension_secs = std::cell::Cell::new(0.0);
    // #phase-cost: the frozen-dual level loop is a BaB loop; its node and
    // prune counts are the census's work denominators (write-only).
    let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::BabLoop);
    let t0 = Instant::now();
    let mut rep = SearchReport::empty();
    rep.relations = relations.map_or(RelationStats::default(), |h| h.stats);
    let relations = relations.filter(|h| h.active());
    rep.evaluator = opts.evaluator.token();
    rep.far_first = matches!(opts.evaluator, Evaluator::Block { far_first: true });
    let finish = |mut rep: SearchReport, verdict: SearchVerdict| {
        rep.verdict = verdict;
        rep.secs = (t0.elapsed().as_secs_f64() - extension_secs.get()).max(0.0);
        {
            use ny_core::phase_cost::{add_units, count, CostCounter, CostPhase};
            add_units(CostPhase::BabLoop, rep.nodes);
            count(CostCounter::DomainsExplored, rep.nodes);
            count(CostCounter::DomainsVerified, rep.certified_prunes);
            count(CostCounter::BabRuns, 1);
        }
        rep
    };
    let mut scratch = NodeScratch::default();
    let root: NodeBound = prob.eval_node(&[], &[], &[], &mut scratch, None);
    rep.root_certified = root.certified;
    rep.worst_open = root.value;
    if root.prunes() {
        rep.cert_min_slack = root.certified;
        return finish(rep, SearchVerdict::Closed);
    }
    if prob.max_depth() == 0 {
        return finish(rep, SearchVerdict::Open(OpenReason::SplitsExhausted));
    }
    let n = prob.n();
    let chunk = opts.chunk.max(1);
    let mut level = Level {
        depth: 1,
        count: 2,
        sides: vec![0, 1],
        l0: vec![0.0; 2],
        l1: vec![0.0; 2],
    };
    loop {
        let prob = source.problem();
        let depth = level.depth;
        let count = level.count;
        if rep.nodes.saturating_add(count as u64) > opts.node_cap {
            return finish(rep, SearchVerdict::Open(OpenReason::NodeCap));
        }
        if past(opts.deadline) {
            return finish(rep, SearchVerdict::Open(OpenReason::Deadline));
        }
        let refine_bytes = if opts.refine_steps > 0 && backend.is_some() {
            breakpoint::work_bytes(n, depth)
                .saturating_mul(rayon::current_num_threads().max(1) as u64)
        } else {
            0
        };
        let wide_bytes = wide_ctx.map_or(0, |w| w.extra_bytes(count, depth));
        let base_bytes = level_bytes(count, depth, n)
            .saturating_add(refine_bytes)
            .saturating_add(wide_bytes);
        let budget = admitted_bytes(opts.max_frontier_bytes);
        if base_bytes > budget {
            return finish(rep, SearchVerdict::Open(OpenReason::Memory));
        }
        let relation_bytes = relations::ClassHints::work_bytes(depth)
            .saturating_mul(rayon::current_num_threads().max(1) as u64);
        // Optional relation scratch cannot turn an admitted old level into a
        // memory refusal. Decline the proposal path and retain its old search.
        let level_relations = if relation_bytes <= budget.saturating_sub(base_bytes) {
            relations.and_then(|h| {
                h.admit_level(n, depth, &level.sides, opts.deadline, &mut rep.relations)
            })
        } else {
            if relations.is_some() {
                rep.relations.refused += 1;
            }
            None
        };
        rep.nodes += count as u64;
        rep.depth = depth;
        rep.peak_frontier = rep.peak_frontier.max(count as u64);
        let ctx = ChunkCtx {
            relations: level_relations,
            automatic_refine: opts.automatic_refine,
            refine_steps: opts.refine_steps.min(32),
            backend,
            deadline: opts.deadline,
            record_cap: opts.record_prunes.saturating_sub(rep.pruned.len()),
            evaluator: opts.evaluator,
            crosscheck_every: opts.crosscheck_every,
            hooks,
        };
        let outs: Vec<ChunkOut> = match wide_ctx {
            None => (0..count.div_ceil(chunk))
                .into_par_iter()
                .map_init(
                    || ChunkWork::new(n, depth, opts.evaluator),
                    |w, c| {
                        let lo = c * chunk;
                        let hi = (lo + chunk).min(count);
                        eval_chunk(prob, &level, lo..hi, &ctx, w)
                    },
                )
                .collect(),
            Some(w) => wide::eval_level(prob, &level, &ctx, chunk, w),
        };
        // Keep the engagement receipt even when a partial level later has to
        // be refused. These counters never participate in the cover ledger.
        for o in &outs {
            rep.relations.merge(o.relations);
            rep.support.merge(o.refinement.support);
            rep.refine_calls += o.refinement.calls;
            rep.refine_steps += o.refinement.steps;
            rep.refine_accepted += o.refinement.accepted;
            rep.refine_prunes += o.refinement.prunes;
            rep.refine_refused += o.refinement.refused;
            rep.refine_skipped += o.refinement.skipped;
        }
        let xfail: u64 = outs.iter().map(|o| o.xchk_fail).sum();
        rep.crosschecks += outs.iter().map(|o| o.xchk).sum::<u64>();
        rep.crosscheck_failures += xfail;
        if xfail > 0 {
            // The block path disagreed with the scalar reference: nothing it
            // computed may prune (a soundness finding, reported by the line).
            return finish(rep, SearchVerdict::Open(OpenReason::Refused));
        }
        if outs.iter().any(|o| o.aborted) {
            // A partially evaluated level proves nothing.
            return finish(rep, SearchVerdict::Open(OpenReason::Deadline));
        }
        let mut pruned = 0u64;
        let mut kept_total = 0usize;
        let mut worst_open = f64::INFINITY;
        for o in &outs {
            pruned += o.pruned;
            kept_total += o.kept.len();
            rep.cert_min_slack = rep.cert_min_slack.min(o.cert_min);
            rep.far_used += o.far_used;
            rep.claimed += o.claimed;
            rep.refuted += o.refuted;
            worst_open = worst_open.min(o.worst_open);
            for (h, &v) in rep.slack_hist.iter_mut().zip(&o.slack_hist) {
                *h += v;
            }
            for r in &o.records {
                if rep.pruned.len() < opts.record_prunes {
                    rep.pruned.push(r.clone());
                }
            }
        }
        rep.certified_prunes += pruned;
        rep.worst_open = worst_open;
        rep.levels.push(LevelTrace {
            depth,
            frontier: count as u64,
            pruned,
        });
        // #router-r1: write-only per-level progress tap (ROUTER_PROGRAM §4.3
        // Bfs: evaluated nodes with the root, certified prunes). Nothing reads
        // it back; one relaxed atomic load while the router is off.
        if ny_core::router::bus::enabled() {
            let evaluated = rep
                .levels
                .iter()
                .fold(1u64, |n, level| n.saturating_add(level.frontier));
            ny_core::router::bus::emit(ny_core::router::bus::ProgressSample::bfs(
                ny_core::router::decl::ids::FROZEN_DUAL_BAB,
                depth as u64,
                evaluated,
                rep.certified_prunes,
            ));
        }
        if pruned + kept_total as u64 != count as u64 {
            return finish(rep, SearchVerdict::Open(OpenReason::Refused));
        }
        if kept_total == 0 {
            let verdict = if ledger_closes(&rep.levels) {
                SearchVerdict::Closed
            } else {
                SearchVerdict::Open(OpenReason::Refused)
            };
            return finish(rep, verdict);
        }
        if depth >= prob.max_depth() {
            let Some(propose) = append.as_deref_mut() else {
                return finish(rep, SearchVerdict::Open(OpenReason::SplitsExhausted));
            };
            let began = Instant::now();
            let offered = propose(prob).and_then(|candidate| match candidate {
                Some(candidate) => {
                    prob.check_append(&candidate, opts.deadline)?;
                    Ok(Some(candidate))
                }
                None => Ok(None),
            });
            extension_secs.set(extension_secs.get() + began.elapsed().as_secs_f64());
            match offered {
                Ok(Some(candidate)) => {
                    *source = ProblemSource::Owned(candidate);
                }
                Ok(None) => return finish(rep, SearchVerdict::Open(OpenReason::SplitsExhausted)),
                Err(ZonoRefusal::Deadline) => {
                    return finish(rep, SearchVerdict::Open(OpenReason::Deadline))
                }
                Err(ZonoRefusal::Memory { .. }) => {
                    return finish(rep, SearchVerdict::Open(OpenReason::Memory))
                }
                Err(_) => return finish(rep, SearchVerdict::Open(OpenReason::Refused)),
            }
        }
        // Children of every kept node: `Off` then `On`, each inheriting the
        // parent's warm start plus a zero multiplier pair for the new split.
        let nd = depth + 1;
        let next_count = kept_total * 2;
        let mut sides = Vec::with_capacity(next_count * nd);
        let mut l0 = Vec::with_capacity(next_count * nd);
        let mut l1 = Vec::with_capacity(next_count * nd);
        for o in &outs {
            for (q, &idx) in o.kept.iter().enumerate() {
                let s = &level.sides[idx * depth..(idx + 1) * depth];
                let a = &o.next_l0[q * depth..(q + 1) * depth];
                let b = &o.next_l1[q * depth..(q + 1) * depth];
                for side in [0u8, 1u8] {
                    sides.extend_from_slice(s);
                    sides.push(side);
                    l0.extend_from_slice(a);
                    l0.push(0.0);
                    l1.extend_from_slice(b);
                    l1.push(0.0);
                }
            }
        }
        drop(outs);
        level = Level {
            depth: nd,
            count: next_count,
            sides,
            l0,
            l1,
        };
    }
}

// ---------------------------------------------------------------------------
// Test-only probes of the evaluators (pins and the integrator's rate probe)
// ---------------------------------------------------------------------------

/// One node's evaluation, as a test sees it.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct NodeProbe {
    pub(crate) best: f64,
    pub(crate) certified: f64,
    pub(crate) winner: Candidate,
    pub(crate) far_used: bool,
    pub(crate) next0: Vec<f64>,
    pub(crate) next1: Vec<f64>,
    /// Block path only (`None` on the node path and on a refused node): the
    /// far point's bound (warm's when it did not move).
    pub(crate) far: Option<NodeBound>,
    pub(crate) far_differs: bool,
    pub(crate) f0: Vec<f64>,
    pub(crate) f1: Vec<f64>,
    pub(crate) sp0: Vec<f64>,
    pub(crate) sp1: Vec<f64>,
    pub(crate) eta_c: f64,
    pub(crate) rc_warm: Vec<f64>,
    pub(crate) da: Vec<f64>,
}

/// Evaluate `sides.len() / depth` search nodes of one `depth >= 1` (row-major
/// inputs, multipliers passed as given) exactly as a level's work item would:
/// through [`eval_search_node`] one by one for [`Evaluator::Node`], or through
/// [`eval_search_block`] in blocks for [`Evaluator::Block`].
#[cfg(test)]
pub(crate) fn probe_nodes(
    prob: &FrozenDualProblem,
    depth: usize,
    sides: &[Side],
    l0: &[f64],
    l1: &[f64],
    evaluator: Evaluator,
) -> Vec<NodeProbe> {
    assert!(depth >= 1, "search nodes have depth >= 1");
    let count = sides.len() / depth;
    assert!(sides.len() == count * depth && l0.len() == sides.len() && l1.len() == sides.len());
    let n = prob.n();
    let mut out = Vec::with_capacity(count);
    match evaluator {
        Evaluator::Node => {
            let mut w = Work::new(n, depth);
            for b in 0..count {
                let r = b * depth..(b + 1) * depth;
                w.sides.copy_from_slice(&sides[r.clone()]);
                w.l0.copy_from_slice(&l0[r.clone()]);
                w.l1.copy_from_slice(&l1[r]);
                let o = eval_search_node(prob, &mut w);
                out.push(NodeProbe {
                    best: o.best,
                    certified: o.certified,
                    winner: o.winner,
                    far_used: o.far_used,
                    next0: w.next0.clone(),
                    next1: w.next1.clone(),
                    far: None,
                    far_differs: false,
                    f0: w.f0.clone(),
                    f1: w.f1.clone(),
                    sp0: w.sp0.clone(),
                    sp1: w.sp1.clone(),
                    eta_c: f64::NAN,
                    rc_warm: w.rc.clone(),
                    da: w.da.clone(),
                });
            }
        }
        Evaluator::Block { far_first } => {
            let mut bw = BlockWork::new(n, depth);
            let mut start = 0usize;
            while start < count {
                let end = (start + BLOCK_NODES).min(count);
                let cnt = end - start;
                bw.count = cnt;
                let src = start * depth..end * depth;
                bw.sides[..cnt * depth].copy_from_slice(&sides[src.clone()]);
                bw.l0[..cnt * depth].copy_from_slice(&l0[src.clone()]);
                bw.l1[..cnt * depth].copy_from_slice(&l1[src]);
                eval_search_block(prob, &mut bw, far_first, TestHooks::OFF);
                for b in 0..cnt {
                    let r = b * depth..(b + 1) * depth;
                    let o = bw.outs[b];
                    let active = bw.active[b];
                    out.push(NodeProbe {
                        best: o.best,
                        certified: o.certified,
                        winner: o.winner,
                        far_used: o.far_used,
                        next0: bw.next0[r.clone()].to_vec(),
                        next1: bw.next1[r.clone()].to_vec(),
                        far: active.then_some(bw.far[b]),
                        far_differs: active && bw.far_differs[b],
                        f0: bw.f0[r.clone()].to_vec(),
                        f1: bw.f1[r.clone()].to_vec(),
                        sp0: bw.sp0[r.clone()].to_vec(),
                        sp1: bw.sp1[r].to_vec(),
                        eta_c: bw.eta_c[b],
                        rc_warm: bw.ws.rc_row(b, n).to_vec(),
                        da: bw.da[b * n..(b + 1) * n].to_vec(),
                    });
                }
                start = end;
            }
        }
    }
    out
}

/// Exercise a fresh `BlockWork` on one full block of `depth` (both far-first
/// settings) and return the bytes its buffers then reserve.
#[cfg(test)]
pub(crate) fn exercised_block_work_bytes(prob: &FrozenDualProblem, depth: usize, seed: u64) -> u64 {
    let n = prob.n();
    let mut bw = BlockWork::new(n, depth);
    bw.count = BLOCK_NODES;
    let mut s = seed;
    for k in 0..BLOCK_NODES * depth {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        bw.sides[k] = if (s >> 33) & 1 == 0 {
            Side::Off
        } else {
            Side::On
        };
        #[allow(clippy::cast_precision_loss)]
        let a = ((s >> 40) & 0xff) as f64 / 64.0;
        #[allow(clippy::cast_precision_loss)]
        let c = ((s >> 48) & 0xff) as f64 / 64.0;
        bw.l0[k] = a;
        bw.l1[k] = c;
    }
    eval_search_block(prob, &mut bw, false, TestHooks::OFF);
    eval_search_block(prob, &mut bw, true, TestHooks::OFF);
    bw.capacity_bytes()
}

/// The integrator's rate probe: `count` random nodes at `prob.max_depth()`
/// (f32 multipliers, a quarter exact zeros), evaluated as one level through
/// the search's own parallel work items. Returns the wall seconds, the prune
/// count and (block path) the per-phase thread-seconds. Its only caller is the
/// `benchmarks`-gated `fd_block_rate_probe` evidence test.
#[cfg(all(test, feature = "benchmarks"))]
pub(crate) fn rate_probe(
    prob: &FrozenDualProblem,
    count: usize,
    seed: u64,
    evaluator: Evaluator,
) -> (f64, u64, [f64; 6]) {
    let depth = prob.max_depth();
    assert!(depth >= 1, "the rate probe needs split rows");
    let n = prob.n();
    let mut s = seed;
    let mut step = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        s
    };
    let mut sides = Vec::with_capacity(count * depth);
    let mut l0 = Vec::with_capacity(count * depth);
    let mut l1 = Vec::with_capacity(count * depth);
    for _ in 0..count * depth {
        let r = step();
        sides.push(u8::from((r >> 20) & 1 == 1));
        #[allow(clippy::cast_precision_loss)]
        let a = if (r >> 30) & 3 == 0 {
            0.0f32
        } else {
            ((r >> 40) & 0xffff) as f32 / 32_768.0
        };
        #[allow(clippy::cast_precision_loss)]
        let b = if (r >> 34) & 3 == 0 {
            0.0f32
        } else {
            ((r >> 24) & 0xffff) as f32 / 32_768.0
        };
        l0.push(a);
        l1.push(b);
    }
    let level = Level {
        depth,
        count,
        sides,
        l0,
        l1,
    };
    let ctx = ChunkCtx {
        relations: None,
        automatic_refine: false,
        refine_steps: 0,
        backend: None,
        deadline: None,
        record_cap: 0,
        evaluator,
        crosscheck_every: FROZEN_DUAL_CROSSCHECK_EVERY,
        hooks: TestHooks::OFF,
    };
    let timed = matches!(evaluator, Evaluator::Block { .. });
    phase_probe::enable(timed);
    let t0 = Instant::now();
    let outs: Vec<ChunkOut> = (0..count.div_ceil(FROZEN_DUAL_CHUNK))
        .into_par_iter()
        .map_init(
            || ChunkWork::new(n, depth, evaluator),
            |w, c| {
                let lo = c * FROZEN_DUAL_CHUNK;
                let hi = (lo + FROZEN_DUAL_CHUNK).min(count);
                eval_chunk(prob, &level, lo..hi, &ctx, w)
            },
        )
        .collect();
    let secs = t0.elapsed().as_secs_f64();
    let phases = if timed {
        phase_probe::read_secs()
    } else {
        [0.0; 6]
    };
    phase_probe::enable(false);
    let pruned = outs.iter().map(|o| o.pruned).sum();
    (secs, pruned, phases)
}

// ---------------------------------------------------------------------------
// The lane tier
// ---------------------------------------------------------------------------

/// The tier's resolved knobs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TierPolicy {
    /// Breakpoint refinement steps per open node, 0 = disabled.
    pub refine_steps: usize,
    /// Nodes per objective (`margin_row.frozen_dual_bab_node_cap`).
    pub node_cap: u64,
    /// Wall request in seconds, `0` = the bounded automatic route
    /// (`margin_row.frozen_dual_bab_secs`).
    pub secs: f64,
    /// How the per-class states are built
    /// (`margin_row.frozen_dual_bab_state`, default `per_class`).
    pub state: FrozenDualStateMode,
    /// M4: which search scores the nodes
    /// (`margin_row.frozen_dual_bab_search`, default `f64` = the M1 path).
    pub search: FrozenDualSearch,
    /// G4 (ENGINE_PROGRAM §1.7 (a′), M2d): the block path re-checks one prune
    /// in this many against the scalar `eval_node` (plus every prune at depth
    /// `<= 4`). NOT a key: the LANE sets it from the entry point's audit rates
    /// ([`Self::with_crosscheck_every`]); [`Self::clamped`] gives the
    /// declaration default [`FROZEN_DUAL_CROSSCHECK_EVERY`]. Always `>= 1`: a
    /// rate may sample a downgrade-only guard less often, never switch it off.
    pub crosscheck_every: u32,
}

impl TierPolicy {
    /// Clamp raw knob values into the reader's domain: node cap in
    /// `1..=FROZEN_DUAL_MAX_NODE_CAP`, seconds finite and `>= 0`. The state
    /// mode and the search are the declaration defaults,
    /// [`FrozenDualStateMode::PerClass`] and [`FrozenDualSearch::F64`].
    #[must_use]
    pub fn clamped(node_cap: u64, secs: f64) -> Self {
        Self {
            refine_steps: 0,
            node_cap: node_cap.clamp(1, FROZEN_DUAL_MAX_NODE_CAP),
            secs: if secs.is_finite() && secs > 0.0 {
                secs
            } else {
                0.0
            },
            state: FrozenDualStateMode::PerClass,
            search: FrozenDualSearch::F64,
            crosscheck_every: FROZEN_DUAL_CROSSCHECK_EVERY,
        }
    }

    /// This policy with the G4 cross-check rate `every` (one prune in `every`,
    /// besides every shallow prune), floored at `1`: a guard is never
    /// disabled, whatever the caller passes. Changes no prune, no child and no
    /// certified value (the cross-check re-evaluates a certified prune and can
    /// only REFUSE the search on a bit mismatch), so it cannot move a verdict
    /// the certificate supports.
    #[must_use]
    pub fn with_crosscheck_every(self, every: u32) -> Self {
        Self {
            crosscheck_every: every.max(1),
            ..self
        }
    }

    /// The same knobs with the state mode `state`.
    #[must_use]
    pub fn with_state(self, state: FrozenDualStateMode) -> Self {
        Self { state, ..self }
    }

    /// This policy with the M4 search selector `search`.
    #[must_use]
    pub fn with_search(self, search: FrozenDualSearch) -> Self {
        Self { search, ..self }
    }

    /// The governed typed-preset route (env > preset cell > declaration).
    /// An inadmissible state or search token cannot reach here (both keys are
    /// `Enum` declarations the resolver validates); a missing one resolves to
    /// its default (`per_class`, `f64`) all the same.
    #[must_use]
    pub fn from_gates() -> Self {
        let cap = crate::preset_gates::margin_row_frozen_dual_bab_node_cap();
        let secs = crate::preset_gates::margin_row_frozen_dual_bab_secs();
        let state = crate::preset_gates::margin_row_frozen_dual_bab_state()
            .as_deref()
            .and_then(FrozenDualStateMode::from_token)
            .unwrap_or_default();
        let search = crate::preset_gates::margin_row_frozen_dual_bab_search()
            .as_deref()
            .and_then(FrozenDualSearch::from_token)
            .unwrap_or_default();
        let mut policy = Self::clamped(u64::try_from(cap).unwrap_or(u64::MAX), secs)
            .with_state(state)
            .with_search(search);
        policy.refine_steps = crate::preset_gates::margin_row_frozen_dual_refine_steps().min(32);
        policy
    }

    /// Zero seconds selects bounded automatic routing; a positive request
    /// retains the explicit research route and its existing share ceiling.
    #[must_use]
    pub fn automatic(&self) -> bool {
        !(self.secs.is_finite() && self.secs > 0.0)
    }

    #[must_use]
    pub fn effective_node_cap(&self) -> u64 {
        if self.automatic() {
            self.node_cap.min(automatic::NODE_CAP)
        } else {
            self.node_cap
        }
    }

    fn state_byte_cap(&self) -> u64 {
        if self.automatic() {
            automatic::STATE_BYTES
        } else {
            FROZEN_DUAL_DEFAULT_MAX_STATE_BYTES
        }
    }

    /// Seconds the tier may spend when `left` seconds remain to the lane
    /// deadline (`None` = no deadline).
    #[must_use]
    pub fn budget_secs(&self, left: Option<f64>) -> f64 {
        if self.automatic() {
            return automatic::budget_secs(left);
        }
        left.map_or(self.secs, |left| {
            self.secs.min(FROZEN_DUAL_MAX_SHARE * left.max(0.0))
        })
    }
}

/// One objective's tier outcome.
#[derive(Debug, Clone)]
pub struct ClassReport {
    /// Adversarial class of the margin `y_t - y_class`.
    pub class: usize,
    /// The lane's certified root bound for it.
    pub lane_root: f64,
    /// The search, or the refusal that prevented it.
    pub result: Result<ClassRun, ZonoRefusal>,
}

/// A class whose state was built and searched.
#[derive(Debug, Clone)]
pub struct ClassRun {
    /// Generator count.
    pub state_gens: usize,
    /// Unstable rows of the state.
    pub unstable: usize,
    /// Split-order rows the problem was built with.
    pub depth_cap: usize,
    /// State build milliseconds.
    pub build_ms: f64,
    /// Stored bytes after streaming row selection.
    pub state_bytes: u64,
    /// Peak admission estimate of streaming state construction.
    pub build_estimate_bytes: u64,
    /// Seed columns built, including rows discarded only after scoring.
    pub build_columns: usize,
    /// Streaming was useful because the frame has more rows than search uses.
    pub streamed_rows: bool,
    /// Certified state root.
    pub root_g: f64,
    /// The search.
    pub search: SearchReport,
    /// How the state was built (per class, or over the shared trunk).
    pub state: FrozenDualStateMode,
    /// Shared trunk only: the class's own per-class root minus `root_g`
    /// (tightness the shared slopes cost; `None` when unknown or per class).
    pub root_loss: Option<f64>,
    /// B2.3 (Proof profile only, hint capture on): the class's write-only
    /// prune trace for the proof bundle; `None` under Competition. The
    /// prunes are MOVED here out of `search.pruned`, so `search` is the same
    /// report in both profiles. No verdict path reads it.
    pub proof: Option<Box<crate::certify::FdProofTrace>>,
    /// Progressive construction work; never consulted by closure authority.
    pub progressive: Option<ProgressiveStats>,
}

/// What the tier did (the lane reads only [`Self::closed`]).
#[derive(Debug, Clone, Default)]
pub struct TierReport {
    /// The bounded automatic route was selected by a zero-second wall request.
    pub automatic: bool,
    /// Classes deliberately handed back without a state build.
    pub automatic_skipped: usize,
    /// Automatic preflight refusal, or `none` when admitted.
    pub automatic_skip: &'static str,
    /// Retained-row preflight or exact streaming state estimate (bytes).
    pub state_estimate_bytes: u64,
    /// Cheap automatic estimate of state pass terms and row ordering work.
    pub compute_estimate: u64,
    /// Objectives the root left open.
    pub objectives: usize,
    /// Classes certified closed (margin > 0 on the whole box).
    pub closed: Vec<usize>,
    /// Classes refused before any search.
    pub refused: usize,
    /// Nodes evaluated over all objectives.
    pub nodes: u64,
    /// Deepest level over all objectives.
    pub depth_max: usize,
    /// Wall seconds.
    pub secs: f64,
    /// Seconds the tier was allowed.
    pub budget_secs: f64,
    /// Block-path cross-check mismatches over all objectives (any is a
    /// soundness finding).
    pub crosscheck_failures: u64,
    /// M4: claimed nodes over all objectives.
    pub claimed: u64,
    /// M4: claimed nodes no proposal certified (split) over all objectives.
    pub refuted: u64,
    /// Per objective.
    pub classes: Vec<ClassReport>,
    /// How the states were built (`margin_row.frozen_dual_bab_state`).
    pub state: FrozenDualStateMode,
    /// Shared mode: wall milliseconds of the trunk (slope selection plus the
    /// row passes), spent BEFORE the per-class fair shares were computed;
    /// `0` per class.
    pub trunk_ms: f64,
    /// Common frame preparation, charged to the original tier allowance.
    pub frame_ms: f64,
    /// Completed convolution magnitude majorants (diagnostic only).
    pub magnitude_envelope_convs: u64,
    /// Magnitude contraction products actually evaluated (diagnostic only).
    pub magnitude_products: u64,
    /// Products a full spatial contraction would evaluate (diagnostic only).
    pub magnitude_direct_products: u64,
    /// Gather entries inspected by the majorants (diagnostic only).
    pub magnitude_gathered: u64,
    /// Classes whose state was built and searched.
    pub classes_built: usize,
    /// Shared mode: the largest known per-class root loss of the chosen
    /// trunk slopes.
    pub shared_root_loss_max: Option<f64>,
    /// Shared mode: the chosen trunk slopes (`joint` or `anchor:<class>`),
    /// `-` otherwise.
    pub trunk_lambda: String,
}

/// Build, order, substitute and search one class (`per_class`: the tier
/// before ENGINE_PROGRAM M3, bit for bit for the state build). Optional
/// relations inspect this already materialized complete chart under the same
/// tier allowance; incomplete streamed charts are skipped. `wide_tier` is M4's backend
/// state when the tier's search is not `f64` (`None`: the f64 search).
fn run_class(
    net: &TwinNet,
    frame: &FrozenDualFrame,
    class: usize,
    deadline: Instant,
    policy: &TierPolicy,
    wide_tier: Option<&mut WideTier>,
    relation_cache: &mut relations::Cache,
) -> Result<ClassRun, ZonoRefusal> {
    let mut opts = FrozenDualOptions {
        deadline: Some(deadline),
        max_state_bytes: policy.state_byte_cap(),
        chunk_cols: if policy.automatic() {
            automatic::chunk_cols(net)
        } else {
            FrozenDualOptions::default().chunk_cols
        },
    };
    let threshold = fd_streamed_threshold();
    let streamed_rows = frame.n_unstable > threshold;
    let retained = frame.n_unstable.min(threshold);
    let estimate = if streamed_rows {
        estimate_ranked_state_bytes(frame, opts.chunk_cols, retained)
    } else {
        estimate_state_bytes(frame, opts.chunk_cols)
    };
    let admitted = admitted_bytes(opts.max_state_bytes);
    if estimate > admitted {
        return Err(ZonoRefusal::Memory {
            estimate_bytes: estimate,
            budget_bytes: admitted,
        });
    }
    // Keep the original build admissible when optional resident hints would
    // consume its spare envelope. Complete-chart relations cannot bind ranked
    // states without all original common slopes/global row metadata.
    if streamed_rows || relation_cache.bytes() > admitted.saturating_sub(estimate) {
        relation_cache.retire();
    }
    opts.max_state_bytes = opts.max_state_bytes.saturating_sub(relation_cache.bytes());
    if streamed_rows && policy.automatic() && wide_tier.is_none() {
        let estimate =
            super::zono_state::build::progressive::estimate_bytes(frame, &opts, retained);
        if estimate <= admitted_bytes(opts.max_state_bytes) {
            return progressive::run(net, frame, class, &opts, deadline, policy, retained);
        }
        eprintln!("[frozen-dual-progressive] class={class} admitted=0 reason=memory estimate_bytes={estimate} fallback=original_ranked");
    }
    let (state, order, table) = if streamed_rows {
        // #fd-compact-checkable: under the Proof profile only (hint capture
        // is switched on by the entry point's Proof arm alone) the compact
        // build also hands back its class relaxation for the class's proof
        // trace. Competition takes the historical build, which drops it.
        let (ranked, table) = if crate::certify::hint_capture_enabled() {
            compact_build_with_table(net, frame, class, &opts, retained, deadline)?
        } else {
            (
                FrozenDualState::build_ranked(net, frame, class, &opts, retained)?,
                FdTable::Refused("capture_off"),
            )
        };
        let order = (0..ranked.state.num_unstable()).collect();
        (ranked.state, Some(order), table)
    } else {
        (
            FrozenDualState::build(net, frame, class, &opts)?,
            None,
            FdTable::Full,
        )
    };
    let build_ms = state.stats.build_ms;
    search_state(
        state,
        frame,
        build_ms,
        deadline,
        policy,
        relation_cache.bytes(),
        FrozenDualStateMode::PerClass,
        None,
        wide_tier,
        order,
        frame.n_unstable,
        (!streamed_rows).then_some((relation_cache, frame)),
        table,
    )
}

/// Test entry for the same progressive owner called by the automatic route.
#[cfg(test)]
pub(crate) fn progressive_class_for_test(
    net: &TwinNet,
    frame: &FrozenDualFrame,
    class: usize,
    opts: &FrozenDualOptions,
    deadline: Instant,
    policy: &TierPolicy,
) -> Result<ClassRun, ZonoRefusal> {
    progressive::run(
        net,
        frame,
        class,
        opts,
        deadline,
        policy,
        FROZEN_DUAL_MAX_DEPTH,
    )
}

/// The unstable-row count above which a per-class state is built RANKED
/// (compact, [`FROZEN_DUAL_MAX_DEPTH`] retained rows). Tests may lower it on
/// their own thread to drive compact states through the tier on small
/// fixtures ([`force_fd_streamed_threshold_for_test`]).
fn fd_streamed_threshold() -> usize {
    #[cfg(test)]
    if let Some(v) = FD_STREAMED_THRESHOLD_FOR_TEST.with(std::cell::Cell::get) {
        return v.min(FROZEN_DUAL_MAX_DEPTH);
    }
    FROZEN_DUAL_MAX_DEPTH
}

#[cfg(test)]
thread_local! {
    /// Test-only override of [`fd_streamed_threshold`] for classes run on
    /// this thread (the real threshold is 512 unstable rows, more than any
    /// lane test fixture has).
    static FD_STREAMED_THRESHOLD_FOR_TEST: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    /// Test-only override of [`fd_proof_table_cap`] for classes run on this
    /// thread.
    static FD_PROOF_TABLE_CAP_FOR_TEST: std::cell::Cell<Option<u64>> =
        const { std::cell::Cell::new(None) };
}

/// Test-only: build per-class states RANKED above `rows` unstable rows on
/// this thread (`None`: the real [`FROZEN_DUAL_MAX_DEPTH`]).
#[cfg(test)]
pub(crate) fn force_fd_streamed_threshold_for_test(rows: Option<usize>) {
    FD_STREAMED_THRESHOLD_FOR_TEST.with(|c| c.set(rows));
}

/// Test-only: force [`fd_proof_table_cap`] to `bytes` (`None`: the real cap)
/// for classes run on this thread.
#[cfg(test)]
pub(crate) fn force_fd_proof_table_cap_for_test(bytes: Option<u64>) {
    FD_PROOF_TABLE_CAP_FOR_TEST.with(|c| c.set(bytes));
}

/// `#fd-compact-checkable`: the record budget (bytes) of one compact state's
/// class relaxation under the Proof profile, before the headroom clamp. A
/// table that does not fit is never recorded: the trace stays incomplete and
/// the class keeps its opaque claim (never a partial table, never a
/// mismatched certificate).
pub const FD_PROOF_TABLE_BYTES: u64 = 256 << 20;

/// The admitted table budget: [`FD_PROOF_TABLE_BYTES`] clamped to a quarter
/// of the admitted headroom.
fn fd_proof_table_cap() -> u64 {
    #[cfg(test)]
    if let Some(cap) = FD_PROOF_TABLE_CAP_FOR_TEST.with(std::cell::Cell::get) {
        return cap;
    }
    FD_PROOF_TABLE_BYTES.min(admitted_bytes(FD_PROOF_TABLE_BYTES) / 4)
}

/// Peak bytes of `frame`'s recorded class relaxation: the retained slope
/// tables (two `f64` per unstable neuron, four vector headers per layer)
/// while the trace's per-neuron records (a `u32` index and three `f64`, one
/// record header per layer) are written from them.
pub(crate) fn fd_proof_table_bytes(frame: &FrozenDualFrame) -> u64 {
    let per_neuron = 2 * size_of::<f64>() as u64 + size_of::<u32>() as u64 + 3 * 8;
    let per_layer =
        4 * size_of::<Vec<f64>>() as u64 + size_of::<crate::certify::FdLambdaRecord>() as u64;
    (frame.n_unstable as u64)
        .saturating_mul(per_neuron)
        .saturating_add((frame.layers.len() as u64).saturating_mul(per_layer))
        .saturating_add(size_of::<CompactRelax>() as u64)
}

/// `#fd-compact-checkable` (Proof profile only): the ranked compact build
/// with its class relaxation, VERIFIED ([`CompactRelax::verify`]) before a
/// trace may use it. Refusals of the build are the historical ranked build's
/// (the same build, operation for operation); a table over its cap is never
/// kept (the historical build runs instead), and a table that does not
/// verify is dropped. Either way the class searches exactly as it would
/// have, and only its trace learns why it cannot stream its prunes.
fn compact_build_with_table(
    net: &TwinNet,
    frame: &FrozenDualFrame,
    class: usize,
    opts: &FrozenDualOptions,
    retained: usize,
    deadline: Instant,
) -> Result<(super::zono_state::RankedFrozenDualState, FdTable), ZonoRefusal> {
    if fd_proof_table_bytes(frame) > fd_proof_table_cap() {
        let ranked = FrozenDualState::build_ranked(net, frame, class, opts, retained)?;
        return Ok((ranked, FdTable::Refused("table_cap")));
    }
    let (ranked, mut relax) =
        FrozenDualState::build_ranked_with_relax(net, frame, class, opts, retained)?;
    // The estimate admitted the table; the cap is re-checked on what the
    // build actually allocated (never a table past its cap).
    if relax.heap_bytes() > fd_proof_table_cap() {
        return Ok((ranked, FdTable::Refused("table_cap")));
    }
    let table = match relax.verify(net, frame, &ranked.state, Some(deadline)) {
        Ok(()) => FdTable::Compact(relax),
        Err(why) => FdTable::Refused(why),
    };
    Ok((ranked, table))
}

/// `#fd-compact-checkable`: what a class's proof trace knows about the
/// relaxation symbols of its frame.
pub(crate) enum FdTable {
    /// A full state: one row (and so one recorded slope) per relaxation
    /// symbol of the frame.
    Full,
    /// A ranked compact state with its VERIFIED class relaxation: every
    /// symbol's slope, the omitted rows' included.
    Compact(CompactRelax),
    /// A ranked compact state without a usable class relaxation (why): its
    /// trace is never complete.
    Refused(&'static str),
}

/// B2.3: the record budget (bytes) of one class's prune trace under the
/// Proof profile, before the headroom clamp. A class whose prunes do not fit
/// keeps its CLAIM in the bundle (an incomplete trace never streams).
pub const FD_PROOF_RECORD_BYTES: u64 = 1 << 30;

#[cfg(test)]
thread_local! {
    /// Test-only override of [`fd_proof_record_cap`] for classes searched on
    /// this thread (the real cap is ~1 GiB of records, which no test fixture
    /// can reach; the incomplete-trace pins need one they can).
    static FD_PROOF_RECORD_CAP_FOR_TEST: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// Test-only: force [`fd_proof_record_cap`] to `cap` (`None`: the real cap)
/// for classes searched on this thread.
#[cfg(test)]
pub(crate) fn force_fd_proof_record_cap_for_test(cap: Option<usize>) {
    FD_PROOF_RECORD_CAP_FOR_TEST.with(|c| c.set(cap));
}

/// B2.3: how many prunes one class may record at `depth_cap` (a prune holds
/// its sides and two multiplier vectors, at most `depth_cap` long).
fn fd_proof_record_cap(depth_cap: usize) -> usize {
    #[cfg(test)]
    if let Some(cap) = FD_PROOF_RECORD_CAP_FOR_TEST.with(std::cell::Cell::get) {
        return cap;
    }
    let budget = FD_PROOF_RECORD_BYTES.min(admitted_bytes(FD_PROOF_RECORD_BYTES) / 4);
    let per = 96u64.saturating_add(17u64.saturating_mul(depth_cap as u64));
    usize::try_from(budget / per.max(1))
        .unwrap_or(usize::MAX)
        .max(1)
}

/// B2.3: what a class's prune trace needs from its state before the state is
/// dropped: the slopes and boxes of every relaxation symbol (grouped by Relu
/// op) and the absolute site of every split-order row.
///
/// #breakpoint-dual x B2.3 (eng9): the prune Lagrangian is unchanged by the
/// breakpoint engine (an accepted hint is certified by the same
/// `FrozenDualProblem::eval_node` and recorded with ITS multipliers, on the
/// node, block and wide paths alike), but a RANKED (streamed) state keeps only
/// the retained split rows while every omitted unstable neuron still owns its
/// relaxation symbol at the class's own slope. Its rows' slopes therefore do
/// not describe the relaxation the tier certified with (the checker would
/// fill the gaps with its chord default and evaluate a DIFFERENT
/// computation).
///
/// `#fd-compact-checkable`: such a trace now carries the class relaxation
/// ([`FdTable::Compact`], a [`CompactRelax`] verified against the state):
/// every symbol's slope and box, the omitted rows' included, so its
/// `hint.fd_lambda` table is COMPLETE and the trace may stream its prunes. A
/// compact state WITHOUT a verified table ([`FdTable::Refused`], or a ranked
/// state labelled [`FdTable::Full`]) is never complete: the lane keeps the
/// class's opaque `fd` claim.
struct FdProofSeed {
    t: usize,
    class: usize,
    n_gens: usize,
    in_gens: usize,
    unstable: usize,
    /// Relaxation symbols the slope table covers.
    relaxed: usize,
    /// `None` when the slope table covers every relaxation symbol of the
    /// frame; otherwise why it cannot (the trace is then never complete).
    table_gap: Option<&'static str>,
    /// `full` or `compact` (the receipt).
    state_kind: &'static str,
    root: f64,
    order: Vec<crate::certify::AbsSite>,
    lambda: Vec<crate::certify::FdLambdaRecord>,
    /// The state hash, fed with the state's per-row relaxation (and, for a
    /// compact state, its class relaxation); the problem's rows are added
    /// after the search.
    hasher: sha2::Sha256,
}

impl FdProofSeed {
    fn of(
        state: &FrozenDualState,
        frame: &FrozenDualFrame,
        order: &[usize],
        table: FdTable,
    ) -> Self {
        use sha2::{Digest, Sha256};
        let to32 = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
        let op_of = |k: usize| {
            frame
                .layers
                .get(state.layer[k])
                .map_or(u32::MAX, |l| to32(l.op))
        };
        let record = |op: u32| crate::certify::FdLambdaRecord {
            op,
            idx: Vec::new(),
            val: Vec::new(),
            lo: Vec::new(),
            hi: Vec::new(),
        };
        let compact = matches!(table, FdTable::Compact(_));
        let mut by_op: std::collections::BTreeMap<u32, crate::certify::FdLambdaRecord> =
            std::collections::BTreeMap::new();
        // The historical state hash: the tag, then every row's relaxation.
        let mut hasher = Sha256::new();
        hasher.update(b"ny-fd-state/v1");
        for k in 0..state.num_unstable() {
            let op = op_of(k);
            if !compact {
                let rec = by_op.entry(op).or_insert_with(|| record(op));
                rec.idx.push(to32(state.neuron[k]));
                rec.val.push(state.lam[k]);
                rec.lo.push(state.lo[k]);
                rec.hi.push(state.hi[k]);
            }
            for v in [
                u64::from(op),
                state.neuron[k] as u64,
                state.enc[k] as u64,
                state.lam[k].to_bits(),
                state.mu[k].to_bits(),
                state.lo[k].to_bits(),
                state.hi[k].to_bits(),
            ] {
                hasher.update(v.to_le_bytes());
            }
        }
        let rows_cover = state.n_gens == frame.n_gens && state.num_unstable() == frame.n_unstable;
        let (relaxed, table_gap, state_kind) = match table {
            FdTable::Full => (
                state.num_unstable(),
                (!rows_cover).then_some("compact_no_table"),
                if rows_cover { "full" } else { "compact" },
            ),
            FdTable::Refused(why) => (state.num_unstable(), Some(why), "compact"),
            FdTable::Compact(cr) => {
                // Every relaxation symbol of the frame, in frame order: the
                // verified class relaxation (the retained rows' slopes are
                // its entries, checked bit for bit by `CompactRelax::verify`).
                hasher.update(b"ny-fd-class-relax/v1");
                let mut n = 0usize;
                let mut gap = None;
                for (li, layer) in frame.layers.iter().enumerate() {
                    let op = to32(layer.op);
                    for (p, &j) in layer.unst.iter().enumerate() {
                        let Some((lam, mu)) = cr.slope(li, p) else {
                            gap = Some("table_shape");
                            continue;
                        };
                        let rec = by_op.entry(op).or_insert_with(|| record(op));
                        rec.idx.push(to32(j));
                        rec.val.push(lam);
                        rec.lo.push(layer.l[j]);
                        rec.hi.push(layer.u[j]);
                        n += 1;
                        for v in [
                            u64::from(op),
                            j as u64,
                            (layer.col0 + p) as u64,
                            lam.to_bits(),
                            mu.to_bits(),
                            layer.l[j].to_bits(),
                            layer.u[j].to_bits(),
                        ] {
                            hasher.update(v.to_le_bytes());
                        }
                    }
                }
                if n != frame.n_unstable || n != cr.len() || state.n_gens != frame.n_gens {
                    gap = gap.or(Some("table_shape"));
                }
                (n, gap, "compact")
            }
        };
        let order = order
            .iter()
            .map(|&k| crate::certify::AbsSite {
                op: op_of(k),
                index: to32(state.neuron[k]),
            })
            .collect();
        Self {
            t: state.t,
            class: state.class,
            n_gens: state.n_gens,
            in_gens: state.n_in_gens,
            unstable: state.num_unstable(),
            relaxed,
            table_gap,
            state_kind,
            root: state.root_bound(),
            order,
            lambda: by_op.into_values().collect(),
            hasher,
        }
    }

    /// The trace of a FINISHED search: the prunes are MOVED out of the
    /// report. The state hash covers the relaxation, the margin and every
    /// split row the search reached (the certified content the prunes rest
    /// on). Prints one `[frozen-dual-proof]` receipt (the trace's own
    /// observable; capture is on whenever a seed exists).
    fn finish(
        self,
        prob: &FrozenDualProblem,
        search: &mut SearchReport,
    ) -> crate::certify::FdProofTrace {
        use sha2::Digest;
        let mut h = self.hasher;
        let put = |h: &mut sha2::Sha256, v: f64| h.update(v.to_bits().to_le_bytes());
        h.update((prob.n() as u64).to_le_bytes());
        put(&mut h, prob.c0());
        put(&mut h, prob.root_bound());
        for &v in prob.d() {
            put(&mut h, v);
        }
        let reached = search.depth.min(prob.max_depth());
        for i in 0..reached {
            let row = prob.split_row(i);
            h.update((row.state_row as u64).to_le_bytes());
            h.update((row.enc as u64).to_le_bytes());
            put(&mut h, row.row_abs);
            for v in row
                .rho
                .iter()
                .chain(&row.c0s)
                .chain(&row.b0)
                .chain(&row.b1)
                .chain(row.a)
            {
                put(&mut h, *v);
            }
        }
        let digest = h.finalize();
        let mut hex = String::with_capacity(64);
        for b in digest {
            use std::fmt::Write as _;
            let _ = write!(hex, "{b:02x}");
        }
        let prunes = std::mem::take(&mut search.pruned);
        let root_closed = search.closed() && search.levels.is_empty();
        let recorded = root_closed
            || (search.closed()
                && prunes.len() as u64 == search.certified_prunes
                && prunes.iter().all(|p| p.sides.len() <= self.order.len()));
        let complete = self.table_gap.is_none() && recorded;
        let reason = match (self.table_gap, search.closed(), recorded) {
            (Some(why), _, _) => why,
            (None, false, _) => "open",
            (None, true, false) => "record_cap",
            (None, true, true) => "-",
        };
        let relation_prunes = prunes
            .iter()
            .filter(|p| p.candidate == Candidate::Relation)
            .count();
        eprintln!(
            "[frozen-dual-proof] class={} state={} rows={} relaxed={} n_gens={} prunes={} \
             relation_prunes={relation_prunes} root_closed={} complete={} reason={reason}",
            self.class,
            self.state_kind,
            self.unstable,
            self.relaxed,
            self.n_gens,
            prunes.len(),
            u8::from(root_closed),
            u8::from(complete),
        );
        crate::certify::FdProofTrace::new(crate::certify::fd_proof::FdProofParts {
            class: self.class,
            t: self.t,
            state_sha256: hex,
            n_gens: self.n_gens,
            in_gens: self.in_gens,
            unstable: self.unstable,
            relaxed: self.relaxed,
            root: if root_closed {
                search.root_certified
            } else {
                self.root
            },
            order: self.order,
            lambda: self.lambda,
            prunes,
            root_closed,
            complete,
        })
    }
}

/// TEST-ONLY: the trace `search_state` builds for `state` searched along
/// `order` with the class's relaxation table `table` (#breakpoint-dual x
/// B2.3: ranked compact states).
#[cfg(test)]
pub(crate) fn fd_proof_trace_for_test(
    state: &FrozenDualState,
    frame: &FrozenDualFrame,
    order: &[usize],
    prob: &FrozenDualProblem,
    search: &mut SearchReport,
    table: FdTable,
) -> crate::certify::FdProofTrace {
    FdProofSeed::of(state, frame, order, table).finish(prob, search)
}

/// Order, substitute and search a state from either selector. Shared trunks
/// remain charged as resident memory. A ranked state already has the exact
/// static-order prefix; all other states retain the full static ordering.
#[allow(clippy::too_many_arguments)]
fn search_state(
    state: FrozenDualState,
    frame: &FrozenDualFrame,
    build_ms: f64,
    deadline: Instant,
    policy: &TierPolicy,
    resident: u64,
    mode: FrozenDualStateMode,
    root_loss: Option<f64>,
    wide_tier: Option<&mut WideTier>,
    ranked_order: Option<Vec<usize>>,
    unstable: usize,
    relation_cache: Option<(&mut relations::Cache, &FrozenDualFrame)>,
    table: FdTable,
) -> Result<ClassRun, ZonoRefusal> {
    let mut resident = resident;
    let streamed_rows = ranked_order.is_some();
    let build_columns = state.stats.columns;
    let build_estimate_bytes = state.stats.estimate_bytes;
    let state_bytes = state.bytes();
    let root_g = state.root_bound();
    let n = state.n_gens;
    if past(Some(deadline)) {
        return Err(ZonoRefusal::Deadline);
    }
    let order = ranked_order.unwrap_or_else(|| static_order(&state));
    if past(Some(deadline)) {
        return Err(ZonoRefusal::Deadline);
    }
    let budget = admitted_bytes(policy.state_byte_cap().saturating_sub(resident));
    let row_bytes = (n as u64).saturating_mul(8).max(1);
    let spare = budget.saturating_sub(state_bytes) / 2;
    let rows_fit = usize::try_from(spare / row_bytes).unwrap_or(usize::MAX);
    let depth_cap = state
        .num_unstable()
        .min(FROZEN_DUAL_MAX_DEPTH)
        .min(rows_fit);
    if depth_cap == 0
        && unstable > 0
        && root_g.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
    {
        return Err(ZonoRefusal::Memory {
            estimate_bytes: row_bytes,
            budget_bytes: spare,
        });
    }
    let prob_estimate = (depth_cap as u64)
        .saturating_add(2)
        .saturating_mul(row_bytes);
    let relation_cap = budget
        .saturating_sub(state_bytes)
        .saturating_sub(prob_estimate);
    let relation_hints = relation_cache.and_then(|(cache, frame)| {
        let before = cache.bytes();
        let hints = cache.bind(
            frame,
            &state,
            &order[..depth_cap],
            Some(deadline),
            relation_cap,
        );
        if !hints.as_ref().is_some_and(relations::ClassHints::active) {
            // Optional refusal must not retain an otherwise unused allocation
            // or reduce the old search's admitted frontier. Retirement is
            // terminal for this tier; it never replenishes the visit allowance.
            cache.retire();
        }
        resident = resident
            .saturating_sub(before)
            .saturating_add(cache.bytes());
        hints
    });
    let (relation_hints, relation_stats) = active_relation_hints(relation_hints);
    let relation_bytes = relation_hints
        .as_ref()
        .map_or(0, relations::ClassHints::bytes);
    let prob = FrozenDualProblem::new(&state, &order[..depth_cap])?;
    // B2.3 (ENGINE_PROGRAM §1.7 B1): under the Proof profile only — hint
    // capture is switched on by the entry point's Proof arm alone — the
    // search RECORDS its prunes (a write-only side channel: the records are
    // pushed after each prune decision and read by nothing in the search)
    // and the class's trace is assembled after it. Competition records
    // nothing (`record_prunes: 0`, exactly the historical value).
    let capture = crate::certify::hint_capture_enabled();
    let proof_seed = capture.then(|| FdProofSeed::of(&state, frame, &order[..depth_cap], table));
    drop(order);
    drop(state);
    if past(Some(deadline)) {
        return Err(ZonoRefusal::Deadline);
    }
    let prob_bytes = (depth_cap as u64)
        .saturating_add(2)
        .saturating_mul(row_bytes);
    let search_opts = SearchOptions {
        automatic_refine: policy.automatic(),
        refine_steps: policy.refine_steps.min(32),
        deadline: Some(deadline),
        node_cap: policy.effective_node_cap(),
        max_frontier_bytes: policy
            .state_byte_cap()
            .saturating_sub(prob_bytes)
            .saturating_sub(resident)
            .saturating_sub(relation_bytes),
        chunk: FROZEN_DUAL_CHUNK,
        record_prunes: if capture {
            fd_proof_record_cap(depth_cap)
        } else {
            0
        },
        evaluator: FROZEN_DUAL_DEFAULT_EVALUATOR,
        crosscheck_every: policy.crosscheck_every,
    };
    let mut search = match wide_tier {
        Some(tier) => match relation_hints.as_ref() {
            Some(hints) => tier.search_class_with_relations(&prob, &search_opts, Some(hints)),
            None => tier.search_class(&prob, &search_opts),
        },
        None => search_with_relations(&prob, &search_opts, relation_hints.as_ref()),
    };
    search.relations.merge(relation_stats);
    if past(Some(deadline)) {
        return Err(ZonoRefusal::Deadline);
    }
    let proof = proof_seed.map(|seed| Box::new(seed.finish(&prob, &mut search)));
    Ok(ClassRun {
        state_gens: n,
        unstable,
        depth_cap,
        build_ms,
        state_bytes,
        build_estimate_bytes,
        build_columns,
        streamed_rows,
        root_g,
        search,
        state: mode,
        root_loss,
        proof,
        progressive: None,
    })
}

/// A shared-trunk tier's per-class inputs (`#fd-shared-state`,
/// ENGINE_PROGRAM M3), prepared under the TIER deadline before any class's
/// fair share exists.
struct SharedTier {
    /// Private checked relation proposals in the exact shared generator chart.
    relations: relations::Cache,
    /// The rows; dropped once the last class's state is materialised.
    trunk: Option<FrozenDualTrunk>,
    /// Per class (tier order): its tail under the chosen slopes, taken once.
    tails: Vec<Option<Result<ClassTail, ZonoRefusal>>>,
    /// Per class (tier order): own per-class root minus shared root (NaN
    /// when either margin refused).
    losses: Vec<f64>,
    /// A refusal of the whole preparation: every class reports it.
    refusal: Option<ZonoRefusal>,
    /// The chosen slopes (`joint` or `anchor:<class>`), `-` if none.
    lambda: String,
    /// Per candidate `name:<within tol>/<classes>:<summed loss>`.
    candidates: String,
    /// Milliseconds choosing the slopes (one-column passes only).
    select_ms: f64,
    /// Milliseconds of the whole preparation (selection plus rows).
    trunk_ms: f64,
}

impl SharedTier {
    /// Every allocation retained between class runs, including tails that
    /// have not yet been consumed. Inline tail headers belong to `tails`.
    fn bytes(&self) -> u64 {
        fn refusal_heap_bytes(why: &ZonoRefusal) -> u64 {
            match why {
                ZonoRefusal::UnsupportedGraph(s)
                | ZonoRefusal::Shape(s)
                | ZonoRefusal::Internal(s) => s.capacity() as u64,
                ZonoRefusal::UnsupportedLayer { node, .. } => node.capacity() as u64,
                _ => 0,
            }
        }
        let tails = self.tails.iter().flatten().fold(0u64, |bytes, tail| {
            bytes.saturating_add(match tail {
                Ok(tail) => tail.heap_bytes(),
                Err(why) => refusal_heap_bytes(why),
            })
        });
        (size_of::<Self>() as u64)
            .saturating_add(self.trunk.as_ref().map_or(0, |trunk| {
                // The trunk's inline header is already part of `Self`.
                trunk
                    .bytes()
                    .saturating_sub(size_of::<FrozenDualTrunk>() as u64)
            }))
            .saturating_add(
                (self.tails.capacity() as u64)
                    .saturating_mul(size_of::<Option<Result<ClassTail, ZonoRefusal>>>() as u64),
            )
            .saturating_add(tails)
            .saturating_add((self.losses.capacity() as u64).saturating_mul(8))
            .saturating_add(self.relations.bytes())
            .saturating_add(self.lambda.capacity() as u64)
            .saturating_add(self.candidates.capacity() as u64)
            .saturating_add(self.refusal.as_ref().map_or(0, refusal_heap_bytes))
    }

    fn refused(mut self, why: ZonoRefusal, t0: Instant) -> Self {
        self.refusal = Some(why);
        self.trunk_ms = t0.elapsed().as_secs_f64() * 1e3;
        self
    }

    /// Class `class`, at `pos` in tier order: materialise its state over the
    /// trunk (the rows are copied) and search it; the last class drops the
    /// trunk first, so its search runs with nothing else resident. The tail
    /// at `pos` must BE `class`'s (the assembly refuses otherwise): the tier
    /// reports, and the cover adapter closes, the class it asked for. The
    /// search is `wide_tier`'s when the tier's search is not `f64`; it runs
    /// only on a state that assembled, so a refused binding never reaches a
    /// backend.
    #[allow(clippy::too_many_arguments)]
    fn run_class(
        &mut self,
        frame: &FrozenDualFrame,
        pos: usize,
        class: usize,
        deadline: Instant,
        policy: &TierPolicy,
        last: bool,
        wide_tier: Option<&mut WideTier>,
    ) -> Result<ClassRun, ZonoRefusal> {
        let materialised = self.class_state(frame, pos, class, deadline, policy.state_byte_cap());
        if last {
            self.trunk = None;
        }
        let state = materialised?;
        let build_ms = state.stats.build_ms;
        let resident = self.bytes();
        let loss = self.losses.get(pos).copied().filter(|l| l.is_finite());
        search_state(
            state,
            frame,
            build_ms,
            deadline,
            policy,
            resident,
            FrozenDualStateMode::Shared,
            loss,
            wide_tier,
            None,
            frame.n_unstable,
            Some((&mut self.relations, frame)),
            FdTable::Full,
        )
    }

    fn class_state(
        &mut self,
        frame: &FrozenDualFrame,
        pos: usize,
        class: usize,
        deadline: Instant,
        max_state_bytes: u64,
    ) -> Result<FrozenDualState, ZonoRefusal> {
        if let Some(why) = &self.refusal {
            return Err(why.clone());
        }
        // The selected tail remains live locally until assembly returns.
        // Capture residency before removing it from the retained tail table.
        let resident = self.bytes();
        let tail = self
            .tails
            .get_mut(pos)
            .and_then(Option::take)
            .ok_or_else(|| {
                ZonoRefusal::Internal("frozen-dual trunk: no tail for this class".to_string())
            })??;
        let trunk = self.trunk.as_ref().ok_or_else(|| {
            ZonoRefusal::Internal("frozen-dual trunk: dropped before its last class".to_string())
        })?;
        if past(Some(deadline)) {
            return Err(ZonoRefusal::Deadline);
        }
        // Rows, objective and flattened lam/mu are allocated while the trunk
        // and every tail, including the selected one, remain alive.
        let copy = trunk.state_copy_bytes(&tail);
        let budget = admitted_bytes(max_state_bytes.saturating_sub(resident));
        if copy > budget {
            return Err(ZonoRefusal::Memory {
                estimate_bytes: copy,
                budget_bytes: budget,
            });
        }
        trunk.state(frame, class, &tail)
    }
}

/// One scored slope candidate of [`prepare_shared`].
struct Scored {
    /// Index into the candidate list.
    cand: usize,
    /// Classes whose root loss is within [`FROZEN_DUAL_SHARED_LOSS_TOL`].
    within: usize,
    /// Summed finite root loss.
    sum: f64,
    /// Per class (tier order): its tail under the candidate.
    tails: Vec<Result<ClassTail, ZonoRefusal>>,
    /// Per class (tier order): own root minus the tail's root.
    losses: Vec<f64>,
}

/// Choose the shared slopes and build the trunk, under the TIER deadline and
/// before any class's fair share exists (ENGINE_PROGRAM M3: the build is
/// accounted OUTSIDE the per-class shares).
///
/// Two candidates, each priced by one-column margin passes only: the JOINT
/// margin's direction-adaptive slopes, and the hardest open class's OWN
/// slopes (its shared state is then its per-class state, bit for bit). Each
/// is scored by its per-class root losses against the classes' own roots —
/// most classes within [`FROZEN_DUAL_SHARED_LOSS_TOL`] first, then the
/// smaller summed loss, then the earlier candidate — and ONLY the winner's
/// rows are built. The choice is untrusted in the sense that matters: any
/// slope in `[0, 1]` gives a sound state, so a bad choice costs tightness.
fn prepare_shared(
    net: &TwinNet,
    frame: &FrozenDualFrame,
    classes: &[usize],
    deadline: Instant,
    max_state_bytes: u64,
) -> SharedTier {
    let t0 = Instant::now();
    let dl = Some(deadline);
    let m = classes.len();
    let tier = SharedTier {
        relations: relations::Cache::default(),
        trunk: None,
        tails: (0..m).map(|_| None).collect(),
        losses: vec![f64::NAN; m],
        refusal: None,
        lambda: "-".to_string(),
        candidates: "-".to_string(),
        select_ms: 0.0,
        trunk_ms: 0.0,
    };
    // Conservative preflight for selection and the row build: a candidate's
    // tails can coexist with the previous winner's tails. Include their
    // nested slope tables, candidate relaxations and retained tier metadata.
    let per_tail = ClassTail::estimated_bytes(frame).saturating_add(
        (size_of::<Result<ClassTail, ZonoRefusal>>() as u64)
            .saturating_sub(size_of::<ClassTail>() as u64),
    );
    let estimate = estimate_trunk_state_bytes(frame, FROZEN_DUAL_DEFAULT_CHUNK_COLS)
        .saturating_add(per_tail.saturating_mul((m as u64).saturating_mul(2)))
        .saturating_add(TrunkRelax::estimated_bytes(frame).saturating_mul(2))
        .saturating_add(tier.bytes())
        // Own roots and two candidate loss vectors; candidate/notes text.
        .saturating_add((m as u64).saturating_mul(3 * 8))
        .saturating_add(2 * size_of::<(String, TrunkRelax)>() as u64)
        .saturating_add(2 * size_of::<String>() as u64)
        .saturating_add(512);
    let budget = admitted_bytes(max_state_bytes);
    if estimate > budget {
        return tier.refused(
            ZonoRefusal::Memory {
                estimate_bytes: estimate,
                budget_bytes: budget,
            },
            t0,
        );
    }
    // 1. Every class's OWN margin: the loss reference and the anchor.
    let own: Vec<Result<ClassTail, ZonoRefusal>> = classes
        .iter()
        .map(|&j| class_margin(net, frame, j, dl))
        .collect();
    if past(dl) {
        return tier.refused(ZonoRefusal::Deadline, t0);
    }
    let own_root: Vec<f64> = own
        .iter()
        .map(|o| o.as_ref().map_or(f64::NAN, ClassTail::root_bound))
        .collect();
    // 2. The candidates.
    let mut cands: Vec<(String, TrunkRelax)> = Vec::with_capacity(2);
    match TrunkRelax::joint(net, frame, classes, dl) {
        Ok(joint) => cands.push(("joint".to_string(), joint)),
        Err(ZonoRefusal::Deadline) => return tier.refused(ZonoRefusal::Deadline, t0),
        Err(_) => {}
    }
    // The hardest class the tier reaches last (easiest-first order).
    if let Some((pos, Ok(hardest))) = own.iter().enumerate().rev().find(|(_, o)| o.is_ok()) {
        cands.push((
            format!("anchor:{}", classes[pos]),
            TrunkRelax::anchor(frame, hardest),
        ));
    }
    drop(own);
    // 3. Score each candidate by its tails.
    let mut notes = Vec::with_capacity(cands.len());
    let mut best: Option<Scored> = None;
    for (ci, (name, cand)) in cands.iter().enumerate() {
        let tails: Vec<Result<ClassTail, ZonoRefusal>> = classes
            .iter()
            .map(|&j| cand.tail(net, frame, j, dl))
            .collect();
        if past(dl) {
            return tier.refused(ZonoRefusal::Deadline, t0);
        }
        let losses: Vec<f64> = tails
            .iter()
            .zip(&own_root)
            .map(|(tail, &own)| tail.as_ref().map_or(f64::NAN, |tl| own - tl.root_bound()))
            .collect();
        let within = losses
            .iter()
            .filter(|&&l| l <= FROZEN_DUAL_SHARED_LOSS_TOL)
            .count();
        let sum: f64 = losses.iter().copied().filter(|l| l.is_finite()).sum();
        notes.push(format!("{name}:{within}/{m}:{sum:+.3e}"));
        let better = best
            .as_ref()
            .is_none_or(|b| within > b.within || (within == b.within && sum < b.sum));
        if better {
            best = Some(Scored {
                cand: ci,
                within,
                sum,
                tails,
                losses,
            });
        }
    }
    let mut tier = tier;
    tier.candidates = if notes.is_empty() {
        "-".to_string()
    } else {
        notes.join(",")
    };
    tier.select_ms = t0.elapsed().as_secs_f64() * 1e3;
    drop(notes);
    drop(own_root);
    let Some(Scored {
        cand: ci,
        tails,
        losses,
        ..
    }) = best
    else {
        return tier.refused(
            ZonoRefusal::Internal("frozen-dual trunk: no slope candidate".to_string()),
            t0,
        );
    };
    let (name, chosen) = cands.swap_remove(ci);
    drop(cands);
    tier.lambda = name;
    tier.tails = tails.into_iter().map(Some).collect();
    tier.losses = losses;
    // 4. The rows, ONCE. The selected tails stay resident during this build;
    // only the remaining configured allowance admits the new trunk peak.
    let opts = FrozenDualOptions {
        deadline: dl,
        max_state_bytes: max_state_bytes.saturating_sub(tier.bytes()),
        ..FrozenDualOptions::default()
    };
    match FrozenDualTrunk::build(net, frame, chosen, &opts) {
        Ok(trunk) => tier.trunk = Some(trunk),
        Err(why) => tier.refusal = Some(why),
    }
    tier.trunk_ms = t0.elapsed().as_secs_f64() * 1e3;
    tier
}

/// TEST-ONLY: exercise the production copy admission at its exact boundary.
#[cfg(test)]
pub(crate) fn shared_class_memory_for_test(
    net: &TwinNet,
    frame: &FrozenDualFrame,
    classes: &[usize],
    short_by: u64,
) -> (u64, u64, u64, Result<u64, ZonoRefusal>) {
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut tier = prepare_shared(
        net,
        frame,
        classes,
        deadline,
        FROZEN_DUAL_DEFAULT_MAX_STATE_BYTES,
    );
    assert!(
        tier.refusal.is_none(),
        "shared preparation: {:?}",
        tier.refusal
    );
    let trunk = tier.trunk.as_ref().expect("trunk");
    let tail = tier.tails[0]
        .as_ref()
        .expect("tail entry")
        .as_ref()
        .expect("tail");
    let resident = tier.bytes();
    let trunk_bytes = trunk.bytes();
    let copy = trunk.state_copy_bytes(tail);
    let cap = resident.saturating_add(copy).saturating_sub(short_by);
    let result = tier
        .class_state(frame, 0, classes[0], deadline, cap)
        .map(|state| state.bytes());
    (resident, trunk_bytes, copy, result)
}

/// `+x.xxxe+y`, or `-` when unknown.
fn fmt_loss(v: Option<f64>) -> String {
    v.map_or_else(|| "-".to_string(), |l| format!("{l:+.3e}"))
}

/// The one `[frozen-dual] trunk` line of a shared tier.
fn emit_trunk(sh: &SharedTier, frame: &FrozenDualFrame) {
    let (passes, trunk_mb, unstable) = sh.trunk.as_ref().map_or((0, 0, 0), |tr| {
        (tr.stats.passes, tr.bytes() >> 20, tr.num_unstable())
    });
    let known: Vec<f64> = sh
        .losses
        .iter()
        .copied()
        .filter(|l| l.is_finite())
        .collect();
    let loss_max = known.iter().copied().reduce(f64::max);
    let within = known
        .iter()
        .filter(|&&l| l <= FROZEN_DUAL_SHARED_LOSS_TOL)
        .count();
    let refused = sh.refusal.as_ref().map_or("-", ZonoRefusal::token);
    eprintln!(
        "[frozen-dual] trunk state=shared lambda={} candidates={} select_ms={:.0} trunk_ms={:.0} \
         passes={passes} unstable={unstable} state_gens={} trunk_mb={trunk_mb} \
         shared_root_loss_max={} within_tol={within}/{} refused={refused}",
        sh.lambda,
        sh.candidates,
        sh.select_ms,
        sh.trunk_ms,
        frame.n_gens,
        fmt_loss(loss_max),
        sh.losses.len(),
    );
}

/// The tier over the classes `tree_classes` the root left open (`dj` is the
/// root's certified bound per `adv` entry, `ybox` the root evaluation's head
/// box). Classes run easiest-first (largest lane root bound), each with a fair
/// share of the tier's remaining wall. Automatic routing selects one class
/// for at least 64MiB of dense rows, or at most two for smaller states, after
/// time, backend, memory and work admission; unselected classes
/// remain in the established tree. Prints a `[frozen-dual] class=` line
/// per objective and the `[frozen-dual] armed=1 objectives=...` summary,
/// unconditionally. Returns the classes certified closed.
///
/// `policy.state` (`margin_row.frozen_dual_bab_state`):
///
/// * `per_class` (default): each class builds its own state inside its share
///   (the original state build before ENGINE_PROGRAM M3, bit for bit). Cheap
///   relations inspect only complete admitted charts; the original evaluator
///   still certifies all additional multiplier proposals;
/// * `shared`: the slopes and the rows of ONE trunk are prepared under the
///   tier deadline BEFORE the first share is computed (one
///   `[frozen-dual] trunk ...` line), and each class then only runs its
///   one-column margin tail and copies the rows inside its share. Every class
///   still closes only through its own certified search (same prunes, same
///   ledger, same token adapter).
///
/// `policy.search` (`margin_row.frozen_dual_bab_search`, M4) is independent
/// of the state mode: `f64` (default) searches every class with this module's
/// own search; any other value resolves the registered UNTRUSTED backends
/// once per tier and scores each class's levels on them (one `[bbe] armed=1`
/// line before the summary), over whichever state the class was given.
#[allow(clippy::too_many_arguments)]
pub fn run_tier(
    net: &TwinNet,
    root: &RootGates,
    t: usize,
    adv: &[usize],
    dj: &[f64],
    tree_classes: &[usize],
    ybox: &YBox,
    deadline: Option<Instant>,
    policy: &TierPolicy,
) -> TierReport {
    let facts = automatic::Facts::from_root(net, root, ybox);
    run_tier_rigged(
        net,
        facts,
        |d| FrozenDualFrame::from_root_until(net, root, ybox, t, Some(d)),
        adv,
        dj,
        tree_classes,
        deadline,
        policy,
        TierRig::scored(policy),
    )
}

/// [`run_tier`] over a frame the CALLER built (`FrozenDualFrame::new` on its
/// own certified [`super::zono_state::FrameInput`]) instead of the lane
/// root's: ny-cli's real-network gates run the tier — the registered M4
/// backends and the M3 shared trunk included — over the P1 replica's
/// certified arm-A boxes, whose open objectives the search actually splits
/// down to certified prunes. The lane root's own boxes on that network leave
/// every class near `-1.6e3`, where no node ever approaches a prune, so a
/// claim-counting gate on them could only be VOID.
///
/// Everything after the frame is [`run_tier`]'s code, unchanged: the fair
/// shares, the state mode, the search selector resolved from the registry,
/// the lines and the report. The caller vouches for the frame exactly as a
/// caller of `FrozenDualFrame::new` does; the lane never calls this entry
/// (it reaches the tier only through [`run_tier`], from `lane_tier`).
#[must_use]
pub fn run_tier_on_frame(
    net: &TwinNet,
    frame: FrozenDualFrame,
    adv: &[usize],
    dj: &[f64],
    tree_classes: &[usize],
    deadline: Option<Instant>,
    policy: &TierPolicy,
) -> TierReport {
    let facts = automatic::Facts::from_frame(net, &frame);
    run_tier_rigged(
        net,
        facts,
        move |_| Ok(frame),
        adv,
        dj,
        tree_classes,
        deadline,
        policy,
        TierRig::scored(policy),
    )
}

/// What a tier runs with besides its policy: M4's backend state (resolved from
/// the registry by [`run_tier`], injected by tests) and, in TEST builds only,
/// the shared tier's binding mutants.
struct TierRig {
    /// M4's backend state; `None` is the f64 search.
    wide: Option<WideTier>,
    /// Test builds only: the M3 binding mutants.
    #[cfg(test)]
    mutant: TierMutant,
}

impl TierRig {
    /// A scored tier's rig: M4's backend state resolved from the registry
    /// when `policy.search` is not `f64`, and no mutant.
    fn scored(policy: &TierPolicy) -> Self {
        Self {
            wide: (policy.search != FrozenDualSearch::F64).then(|| WideTier::new(policy.search)),
            #[cfg(test)]
            mutant: TierMutant::default(),
        }
    }
}

/// TEST-ONLY mutants of a `shared` tier, applied once the trunk is prepared.
/// Each breaks one of the bindings `FrozenDualTrunk::state` enforces, so every
/// affected class must be REFUSED on every search path, before any backend is
/// consulted (the M3 bindings hold on the M4 wide path too).
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TierMutant {
    /// Swap the tails at these two tier positions (the CLASS binding).
    pub(crate) swap_tails: Option<(usize, usize)>,
    /// Assemble every class on a frame whose input centre moved (the LAYOUT
    /// binding).
    pub(crate) move_frame: bool,
    /// Flip one bit of the trunk's rows (the content DIGEST).
    pub(crate) corrupt_trunk: bool,
}

#[cfg(test)]
impl TierMutant {
    /// The shared tier as prepared, mutated.
    fn prepared(self, mut sh: SharedTier) -> SharedTier {
        if let Some((a, b)) = self.swap_tails {
            if a < sh.tails.len() && b < sh.tails.len() {
                sh.tails.swap(a, b);
            }
        }
        if self.corrupt_trunk {
            if let Some(trunk) = sh.trunk.as_mut() {
                trunk.corrupt_for_test(0);
            }
        }
        sh
    }

    /// The frame the classes of a SHARED tier assemble on, mutated.
    fn class_frame(
        self,
        shared: bool,
        frame: Result<FrozenDualFrame, ZonoRefusal>,
    ) -> Result<FrozenDualFrame, ZonoRefusal> {
        if !(shared && self.move_frame) {
            return frame;
        }
        frame.map(|mut fr| {
            if let Some(m) = fr.mid.first_mut() {
                *m += 0.5;
            }
            fr
        })
    }
}

/// [`run_tier`] with an injected M4 backend state and the shared tier's
/// binding mutants (test builds only).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_tier_with_wide(
    net: &TwinNet,
    root: &RootGates,
    t: usize,
    adv: &[usize],
    dj: &[f64],
    tree_classes: &[usize],
    ybox: &YBox,
    deadline: Option<Instant>,
    policy: &TierPolicy,
    wide_tier: Option<WideTier>,
    mutant: TierMutant,
) -> TierReport {
    let facts = automatic::Facts::from_root(net, root, ybox);
    run_tier_rigged(
        net,
        facts,
        |d| FrozenDualFrame::from_root_until(net, root, ybox, t, Some(d)),
        adv,
        dj,
        tree_classes,
        deadline,
        policy,
        TierRig {
            wide: wide_tier,
            mutant,
        },
    )
}

/// The tier body behind [`run_tier`], [`run_tier_on_frame`] and the test
/// entry: `frame_of` builds the class-independent frame, called once and only
/// when some objective is open (exactly where `run_tier` always built it).
#[allow(clippy::too_many_arguments)]
fn run_tier_rigged<F>(
    net: &TwinNet,
    facts: automatic::Facts,
    frame_of: F,
    adv: &[usize],
    dj: &[f64],
    tree_classes: &[usize],
    deadline: Option<Instant>,
    policy: &TierPolicy,
    rig: TierRig,
) -> TierReport
where
    F: FnOnce(Instant) -> Result<FrozenDualFrame, ZonoRefusal>,
{
    let t0 = Instant::now();
    let left = deadline.map(|d| d.saturating_duration_since(t0).as_secs_f64());
    let budget = policy.budget_secs(left);
    // `Duration::from_secs_f64` panics on NaN / negative / overflow.
    let budget = if budget.is_finite() {
        budget.clamp(0.0, 1e9)
    } else {
        0.0
    };
    let mut tier_deadline = t0 + Duration::from_secs_f64(budget);
    if let Some(d) = deadline {
        tier_deadline = tier_deadline.min(d);
    }
    let mut report = TierReport {
        automatic: policy.automatic(),
        automatic_skip: "none",
        objectives: tree_classes.len(),
        budget_secs: budget,
        ..TierReport::default()
    };
    let lane_root = |j: usize| {
        adv.iter()
            .position(|&c| c == j)
            .and_then(|i| dj.get(i).copied())
            .unwrap_or(f64::NAN)
    };
    let mut classes: Vec<(usize, f64)> = tree_classes.iter().map(|&j| (j, lane_root(j))).collect();
    classes.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if policy.automatic() {
        report.state_estimate_bytes = facts.retained_bytes;
        report.compute_estimate = facts.compute_work;
        let backend = super::root_search::registered_backend().is_some();
        if let Err(skip) = facts.admit(
            classes.len(),
            backend,
            tier_deadline
                .saturating_duration_since(Instant::now())
                .as_secs_f64(),
            admitted_bytes(automatic::STATE_BYTES),
        ) {
            report.automatic_skip = skip.token();
            report.automatic_skipped = report.objectives;
            report.secs = t0.elapsed().as_secs_f64();
            emit_summary(&report, policy);
            return report;
        }
        // Keep every unselected class in the caller's established tree. The
        // tier may only remove a class through its own certified cover.
        let class_limit = facts.class_limit();
        report.automatic_skipped = classes.len().saturating_sub(class_limit);
        classes.truncate(class_limit);
    }
    let frame_t0 = Instant::now();
    let frame = if classes.is_empty() {
        Err(ZonoRefusal::Internal("no open objective".to_string()))
    } else {
        frame_of(tier_deadline)
    };
    report.frame_ms = frame_t0.elapsed().as_secs_f64() * 1000.0;
    if let Ok(fr) = &frame {
        report.magnitude_envelope_convs = fr.magnitude_work.envelope_convs;
        report.magnitude_products = fr.magnitude_work.products;
        report.magnitude_direct_products = fr.magnitude_work.direct_products;
        report.magnitude_gathered = fr.magnitude_work.gathered;
    }
    if policy.automatic() {
        if let Ok(frame) = &frame {
            let estimate = if frame.n_unstable > FROZEN_DUAL_MAX_DEPTH {
                estimate_ranked_state_bytes(
                    frame,
                    automatic::chunk_cols(net),
                    FROZEN_DUAL_MAX_DEPTH,
                )
            } else {
                estimate_state_bytes(frame, automatic::chunk_cols(net))
            };
            report.state_estimate_bytes = estimate;
            if estimate > admitted_bytes(automatic::STATE_BYTES) {
                report.automatic_skip = "state_memory";
                report.automatic_skipped = report.objectives;
                report.secs = t0.elapsed().as_secs_f64();
                emit_summary(&report, policy);
                return report;
            }
        }
    }
    report.state = policy.state;
    report.trunk_lambda = "-".to_string();
    // #fd-shared-state (ENGINE_PROGRAM M3): the trunk is prepared HERE, before
    // the first class's share is computed, so its build is accounted outside
    // every per-class fair share. `per_class` runs nothing here.
    let mut shared = match (policy.state, &frame) {
        (FrozenDualStateMode::Shared, Ok(fr)) => {
            let ids: Vec<usize> = classes.iter().map(|&(j, _)| j).collect();
            let sh = prepare_shared(net, fr, &ids, tier_deadline, policy.state_byte_cap());
            #[cfg(test)]
            let sh = rig.mutant.prepared(sh);
            emit_trunk(&sh, fr);
            report.trunk_ms = sh.trunk_ms;
            report.trunk_lambda.clone_from(&sh.lambda);
            report.shared_root_loss_max = sh
                .losses
                .iter()
                .copied()
                .filter(|l| l.is_finite())
                .reduce(f64::max);
            Some(sh)
        }
        _ => None,
    };
    #[cfg(test)]
    let frame = rig.mutant.class_frame(shared.is_some(), frame);
    let total = classes.len();
    // #wide-fd-search (M4): one backend state for the whole tier, whichever
    // way the classes' states are built (`None`: the f64 search).
    let mut wide_tier = rig.wide;
    // #router-r5 CR-G: unarmed (every run without `router.mode: on`) the
    // funding hands back today's share untouched.
    let mut funding = gap_contract::Funding::from_authority(total);
    // One finite allowance across all ordinary per-class objectives. A changed
    // common chart refuses transfer; no target gets a freshly reset budget.
    let mut per_class_relations = relations::Cache::default();
    for (i, &(class, dj_j)) in classes.iter().enumerate() {
        let now = Instant::now();
        let left_tier = tier_deadline.saturating_duration_since(now);
        #[allow(clippy::cast_possible_truncation)]
        let share = left_tier / ((total - i).max(1) as u32);
        // #router-r5 CR-G: a contract >= build + 2 levels in gap order;
        // `None` declines this class and the ones behind it (they stay open).
        let Some(share) = funding.window(class, total - i, share, left_tier) else {
            break;
        };
        let class_deadline = now + share;
        let result = match &frame {
            Ok(fr) => match shared.as_mut() {
                None => run_class(
                    net,
                    fr,
                    class,
                    class_deadline,
                    policy,
                    wide_tier.as_mut(),
                    &mut per_class_relations,
                ),
                Some(sh) => sh.run_class(
                    fr,
                    i,
                    class,
                    class_deadline,
                    policy,
                    i + 1 == total,
                    wide_tier.as_mut(),
                ),
            },
            Err(e) => Err(e.clone()),
        };
        // #router-r5 CR-G: price the next class from this one (armed only).
        funding.observe(&result);
        match &result {
            Ok(run) => {
                let s = &run.search;
                let r = s.relations;
                let eligibility = if run.streamed_rows {
                    "streamed_chart_skip"
                } else {
                    "full_chart"
                };
                eprintln!("[frozen-dual-relations] class={class} state={} eligibility={eligibility} attempted={} admitted={} refused={} replayed={} used={} eval_improved={} eval_pruned={} work_reserved={} scope_census={} scope_over_cap={} authority=original_fd_eval",
                    run.state.token(), r.attempted, r.admitted, r.refused, r.replayed, r.used,
                    r.eval_improved, r.eval_pruned, r.work_reserved, r.scope_census, r.scope_over_cap);
                report.classes_built += 1;
                report.nodes += run
                    .progressive
                    .as_ref()
                    .map_or(s.nodes, |p| p.nodes_charged);
                report.depth_max = report.depth_max.max(s.depth);
                report.crosscheck_failures += s.crosscheck_failures;
                report.claimed += s.claimed;
                report.refuted += s.refuted;
                let hist = s
                    .slack_hist
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                let (verdict, reason) = match s.verdict {
                    SearchVerdict::Closed => ("closed", "-"),
                    SearchVerdict::Open(r) => ("open", r.token()),
                };
                if s.closed() {
                    report.closed.push(class);
                }
                eprintln!(
                    "[frozen-dual] class={class} state_gens={} unstable={} depth_cap={} \
                     build_ms={:.0} root_g={:+.6e} lane_root={dj_j:+.6e} nodes={} depth={} \
                     certified_prunes={} cert_min_slack={:.3e} far_used={} peak_frontier={} \
                     worst_open={:+.6e} verdict={verdict} reason={reason} secs={:.3} eval={} \
                     far_first={} search_nps={:.0} slack_hist={hist} xchk={}/{} \
                     breakpoint_calls={} breakpoint_steps={} breakpoint_accepted={} \
                     breakpoint_prunes={} breakpoint_refused={} breakpoint_skipped={} \
                     support_compiled={} support_fallback={} support_columns={}/{} \
                     support_untouched={} support_grouped={} support_sign_folded={} \
                     support_events={}/{} support_reads={}/{} support_build_reads={} \
                     support_peak_bytes={} support_dense_bytes={} \
                     state_rows={} state_bytes={} build_estimate_bytes={} build_columns={} \
                     state={} root_loss={} search={} probe={} claimed={} refuted={} fallback={}",
                    run.state_gens,
                    run.unstable,
                    run.depth_cap,
                    run.build_ms,
                    run.root_g,
                    s.nodes,
                    s.depth,
                    s.certified_prunes,
                    s.cert_min_slack,
                    s.far_used,
                    s.peak_frontier,
                    s.worst_open,
                    s.secs,
                    s.evaluator,
                    u8::from(s.far_first),
                    s.search_nps(),
                    s.crosschecks,
                    s.crosscheck_failures,
                    s.refine_calls,
                    s.refine_steps,
                    s.refine_accepted,
                    s.refine_prunes,
                    s.refine_refused,
                    s.refine_skipped,
                    s.support.compiled,
                    s.support.fallback,
                    s.support.active_columns,
                    s.support.input_columns,
                    s.support.untouched_columns,
                    s.support.grouped_columns,
                    s.support.sign_folded_columns,
                    s.support.events,
                    s.support.ungrouped_events,
                    s.support.search_reads,
                    s.support.dense_search_reads,
                    s.support.build_reads,
                    s.support.peak_bytes,
                    s.support.dense_bytes,
                    if run.streamed_rows {
                        "streamed"
                    } else {
                        "full"
                    },
                    run.state_bytes,
                    run.build_estimate_bytes,
                    run.build_columns,
                    run.state.token(),
                    fmt_loss(run.root_loss),
                    s.search,
                    s.probe,
                    s.claimed,
                    s.refuted,
                    s.fallback_chunks,
                );
                // #router-r1: write-only per-class progress tap (the class's
                // final counters; the per-level tap carries its trajectory).
                if ny_core::router::bus::enabled() {
                    ny_core::router::bus::emit(ny_core::router::bus::ProgressSample::bfs(
                        ny_core::router::decl::ids::FROZEN_DUAL_BAB,
                        s.depth as u64,
                        s.nodes.saturating_add(1),
                        s.certified_prunes,
                    ));
                }
            }
            Err(why) => {
                report.refused += 1;
                eprintln!(
                    "[frozen-dual] class={class} state_gens=0 unstable=0 depth_cap=0 build_ms=0 \
                     root_g=nan lane_root={dj_j:+.6e} nodes=0 depth=0 certified_prunes=0 \
                     cert_min_slack=inf far_used=0 peak_frontier=0 worst_open=nan \
                     verdict=refused reason={} secs=0.000 eval=- far_first=0 search_nps=0 \
                     slack_hist=0,0,0,0,0,0,0,0,0,0 xchk=0/0 state={} root_loss=- search=- \
                     probe=- claimed=0 refuted=0 fallback=0",
                    why.token(),
                    policy.state.token(),
                );
            }
        }
        report.classes.push(ClassReport {
            class,
            lane_root: dj_j,
            result,
        });
    }
    report.secs = t0.elapsed().as_secs_f64();
    if let Some(tier) = &wide_tier {
        tier.emit();
    }
    emit_summary(&report, policy);
    report
}

fn emit_summary(report: &TierReport, policy: &TierPolicy) {
    let closed = report.closed.len();
    let verdict = if report.objectives == 0 {
        "no_open_objective"
    } else if closed == report.objectives {
        "all_closed"
    } else if closed == 0 {
        "none_closed"
    } else {
        "partial"
    };
    let rss = super::zono_state::peak_rss_mb().map_or_else(|| "-".to_string(), |m| m.to_string());
    // Append-only: the M3 fields follow `xchk_fail=`, the M4 fields follow
    // `fd-shared-state/v1`, the G4 rate (`xchk_every=`, set by the lane from the
    // proof profile's audit rates) follows `fd-wide-search/v1`, and
    // `fd-shared-state/v1` / `fd-wide-search/v1` are each one contiguous
    // literal the integrator can grep a pinned binary for.
    eprintln!(
        "[frozen-dual] armed=1 objectives={} closed={} handed_back={} refused={} nodes={} \
         depth_max={} secs={:.3} budget_secs={:.3} node_cap={} peak_rss_mb={rss} verdict={verdict} \
         fd-block-eval/v1 xchk_fail={} breakpoint-dual/v1 refine_steps={} \
         route={} auto_skipped={} auto_skip={} state_estimate_bytes={} compute_estimate={} \
         state={} trunk_ms={:.0} classes_built={} shared_root_loss_max={} lambda={} \
         fd-shared-state/v1 search={} claimed={} refuted={} fd-wide-search/v1 xchk_every={} \
         fd-magnitude-envelope/v1 frame_ms={:.3} magnitude_convs={} magnitude_products={} \
         magnitude_direct_products={} magnitude_gathered={}",
        report.objectives,
        closed,
        report.objectives - closed,
        report.refused,
        report.nodes,
        report.depth_max,
        report.secs,
        report.budget_secs,
        policy.effective_node_cap(),
        report.crosscheck_failures,
        policy.refine_steps,
        if report.automatic {
            "automatic"
        } else {
            "manual"
        },
        report.automatic_skipped,
        report.automatic_skip,
        report.state_estimate_bytes,
        report.compute_estimate,
        report.state.token(),
        report.trunk_ms,
        report.classes_built,
        fmt_loss(report.shared_root_loss_max),
        report.trunk_lambda,
        policy.search.token(),
        report.claimed,
        report.refuted,
        policy.crosscheck_every,
        report.frame_ms,
        report.magnitude_envelope_convs,
        report.magnitude_products,
        report.magnitude_direct_products,
        report.magnitude_gathered,
    );
}

/// The lane hook (outer frame only, armed only): the tier over the classes
/// `re` leaves open, with the governed knobs. A panic inside the tier is
/// caught and hands every class back (fail-closed): a new tier must never be
/// able to take the established tree down with it.
///
/// `crosscheck_every` is G4's rate, chosen by the CALLER (the lane reads the
/// entry point's audit rates); this module stays profile-blind (the crate's
/// section-1.7 ratchet scans it for the profile's names).
///
/// `harness` is true only when an IN-PROCESS harness armed the tier
/// (`BabConfig::frozen_dual_tier`, never set by the scored lane). The
/// governed default (`frozen_dual_bab_secs: 0`) selects the #breakpoint-dual
/// bounded automatic route, which admits classes only with the CLI-registered
/// search backend that ny-propagate's in-process harnesses cannot register
/// (every class would be skipped as `no_backend`). A harness therefore keeps
/// the explicit research route at [`FROZEN_DUAL_HARNESS_SECS`] unless the
/// gate already requested explicit seconds; the tier certifies every class it
/// closes on either route.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lane_tier(
    net: &TwinNet,
    root: &RootGates,
    t: usize,
    adv: &[usize],
    re: &RootEval,
    deadline: Option<Instant>,
    crosscheck_every: u32,
    harness: bool,
) -> TierReport {
    let mut policy = TierPolicy::from_gates().with_crosscheck_every(crosscheck_every);
    if harness && policy.automatic() {
        policy.secs = FROZEN_DUAL_HARNESS_SECS;
    }
    let ybox = re.root_ybox();
    let run = std::panic::AssertUnwindSafe(|| {
        run_tier(
            net,
            root,
            t,
            adv,
            &re.dj,
            &re.tree_classes,
            &ybox,
            deadline,
            &policy,
        )
    });
    match std::panic::catch_unwind(run) {
        Ok(report) => report,
        Err(_) => {
            let report = TierReport {
                automatic: policy.automatic(),
                automatic_skip: "panic",
                automatic_skipped: re.tree_classes.len(),
                objectives: re.tree_classes.len(),
                refused: re.tree_classes.len(),
                state: policy.state,
                trunk_lambda: "-".to_string(),
                ..TierReport::default()
            };
            if policy.search != FrozenDualSearch::F64 {
                eprintln!(
                    "[bbe] armed=1 backend=- probe=panic lease_mb=0 search={} classes=0 \
                     fallback_classes=0 why=tier_panic",
                    policy.search.token()
                );
            }
            eprintln!("[frozen-dual] panic: the whole tier is refused");
            emit_summary(&report, &policy);
            report
        }
    }
}
