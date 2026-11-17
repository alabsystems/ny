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
use std::sync::atomic::{AtomicUsize as ClipHeadCounter, Ordering as ClipHeadOrder};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ndarray::{s, Array2};
use ny_core::{NyError, Result};
use rayon::prelude::*;

use super::bounds::{
    compose_viay, head_gates_with, head_variant, margin_seed, per_class_direct, row_dots,
    trunk_variant, variant_state, HeadGates, MarginBatch, MarginSeed, RowDots, YBox,
};
use super::close_integrity::{Cert, CertKind, CertTally, CloseIntegrityEvidence};
use super::engine::{
    domain_gates, domain_gates_acc, BackwardEngine, Collect, DomainGates, Exc, Exceptions, PassOut,
    RowDomainGateBlock, Seed,
};
use super::net::TwinNet;
use super::root::{RetainedRows, RootGates};
use super::rounding::{next_down, next_up, RoundMode, UNIT};
use crate::certify::margin_row::{self as cert_mr, ChildSplit};
use crate::certify::{
    authorize_unsat, AbsLit, ClassCover, CoverLedger, FdProofTrace, FrameSplit, GuardReport,
    HintLedger, LeafClose, LeafEvidence, LeafHint, PreHint, RetireHint, SiteMap, TokenScope,
    UnsatToken,
};

// #mr-stack (margin-row throughput plan B5): the domain-stacked frontier's
// shared main-eval pass, its receipt, and the trajectory hash.
mod prepared_domain_eval;
#[cfg(test)]
pub(in crate::margin_row) use prepared_domain_eval::evaluate_prepared_domain;
mod prepared_root_eval;
mod prepared_tree;
mod root_eval_progress;
pub use prepared_tree::{
    PreparedTreeCertificate, PreparedTreeOpen, PreparedTreeOpenReason, PreparedTreeOptions,
    PreparedTreeOutcome, PreparedTreeReceipt, PreparedTreeSchedule,
};
mod stack_eval;
pub(super) use prepared_root_eval::{
    prepared_root_eval_reuse, PreparedRootEvaluation, PreparedRootEvaluationReceipt,
    PreparedRootYBox,
};
#[cfg(test)]
pub(crate) use stack_eval::take_last_trajectory;

// #wave-beta-bab (ENGINE_PROGRAM M6): the wave tree engine behind
// `margin_row.tree_mode: wave` (dark; `legacy` never enters it).
pub mod wave;

// ENGINE_PROGRAM §1.7 B1 (proof bundle v2): the emitter's unit tests.
#[cfg(test)]
mod direct_publication_tests;
#[cfg(test)]
mod proof_hint_tests;
pub(crate) use stack_eval::{mr_stack_fields, mr_stack_reset, mr_stack_stop_line};

/// Trunk piece-fix: (layer, position in the layer's unstable list, direction).
pub type TrunkSplit = (usize, usize, i8);
/// Head fix: (head neuron, direction).
pub type HeadFix = (usize, i8);

/// Hard cap on incremental epoch frames simultaneously nested on one branch.
///
/// This is a trusted code limit, not a per-driver attempt counter: every edge
/// carries and decrements the same budget, and even a manually constructed
/// oversized [`BabConfig`] is clamped before it can recurse.
pub(crate) const EPOCH_PARENT_GENERATION_CAP: usize = 4;

/// Invocation frame used to keep outer treatment work distinct from work
/// performed inside an epoch-created subtree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpochFrameRoute {
    /// Top-level joint tree or one top-level classwise tree.
    Outer,
    /// Subtree created by a full-tableau epoch rebuild.
    NestedFull,
    /// Subtree created by an incremental parent-frame refresh.
    NestedIncremental,
}

/// Engagement counters for the `[domain-stack]` lane. A scored run is one
/// process, so process-global counters report it without threading fields
/// through the engine. Diagnostics only — never read by any bound, gate or
/// verdict.
///
/// `MAX_WIDTH` rather than a first-hit latch, deliberately: the first stacked
/// expansion occurs when the heap has barely filled and is therefore the
/// NARROWEST of the run, so reporting it understates the mechanism and would
/// make a width-sensitive A/B unreadable.
static DOMAIN_STACK_MAX_WIDTH: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static DOMAIN_STACK_EXPANSIONS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// #mr-stack (throughput plan B5) kill switch: with `margin_row.domain_stack`
/// armed, run a frontier batch's main eval passes as ONE stacked pass
/// (`bab/stack_eval.rs`). A trusted code constant, not a lever — the stacked
/// pass is bit-identical to the per-domain passes it replaces, so it selects a
/// schedule and cannot move a bound. `false` restores the per-domain main pass
/// exactly (score stacking is unaffected).
pub(crate) const DOMAIN_STACK_MAIN_EVAL: bool = true;

/// Tuning knobs (defaults = the verified Part-B protocol).
#[derive(Clone)]
pub struct BabConfig {
    /// Configured head candidates per expansion. This immutable base is kept
    /// separate from the one-way adaptive operational width below.
    pub k_head: usize,
    /// Exact lower-authority head width for execution counterfactuals.
    pub contextual_k_head: usize,
    /// Configured trunk candidates per expansion.
    pub k_trunk: usize,
    /// Exact lower-authority trunk width for execution counterfactuals.
    pub contextual_k_trunk: usize,
    /// Current operational head width. Public entries seed this from
    /// `k_head`; nested epoch entries inherit the parent's ratcheted value.
    pub effective_k_head: usize,
    /// Current lower-authority operational head width.
    pub effective_contextual_k_head: usize,
    /// Current operational trunk width.
    pub effective_k_trunk: usize,
    /// Current lower-authority operational trunk width.
    pub effective_contextual_k_trunk: usize,
    /// Receipt-owned adaptive-width authority.
    pub adaptive: bool,
    /// Exact lower-authority adaptive-width choice.
    pub contextual_adaptive: bool,
    /// Expansion cap.
    pub max_expansions: usize,
    /// Wall-clock deadline for SEARCHING.
    pub deadline: Option<Instant>,
    /// Optional source-owned joint head relaxation, once on the outer root.
    /// Its bounded slice is charged within `deadline`; the raw root memo is
    /// never changed by this refinement.
    pub head_groups: Option<super::HeadGroupPolicy>,
    /// #publish-deadline — wall-clock deadline for PUBLISHING a verdict whose
    /// proof is already complete.
    ///
    /// # Why this is separate from `deadline`
    ///
    /// `deadline` is the LANE deadline, set to `scored_deadline - 3s`
    /// (`ny-cli/src/commands/margin_row_bab.rs:2681`) to "leave the tail to the
    /// caller's own wrap-up (result write, witness gating)". That reserve is
    /// correct for SEARCH. But the same instant was also used to discard a
    /// verdict whose tree had ALREADY finished with a verdict-bearing stop
    /// (`queue_empty` / `verified_bestfirst`), and at that point the remaining
    /// work is a struct clone — microseconds, not seconds.
    ///
    /// MEASURED 2026-09-16 over 32 cifar100 resnet_large runs: 38% never publish
    /// a `TREE DONE` receipt, and `idx_1280_sidx_5310` published a genuine
    /// `UNSAT in 92.9s (ledger=Some(true))` — agreeing with alpha-beta-CROWN,
    /// which also proves it unsat — while returning `unknown` on five other runs
    /// with IDENTICAL tree statistics. `unspent_frac` was 0.020-0.031 on those
    /// rows: 2-3 s of the official budget was still unused at the moment the
    /// completed proof was thrown away.
    ///
    /// `None` falls back to `deadline`, which is byte-identical to the previous
    /// behaviour.
    pub publish_deadline: Option<Instant>,
    /// y-row LRU capacity (trunk-sets).
    pub lru_cap: usize,
    /// Parallel best-first frontier width: expand up to this many worst-open
    /// domains concurrently per step (rayon). `1` = the verified serial lane.
    /// Each domain's bound is computed bit-identically to serial; only the
    /// tree exploration order changes (still a sound per-point sign cover).
    pub frontier: usize,
    /// Effective frontier width the same invocation would have received after
    /// removing the `NY_MARGIN_ROW_PARALLEL` environment treatment while
    /// retaining its typed base preset. This is frozen beside `frontier` so
    /// post-run evidence compares one tree decision in the same Rayon context
    /// rather than reopening process-global state after the run.
    pub frontier_contextual: usize,
    /// Default-off cross-domain stacking (`margin_row.domain_stack`). The
    /// `score_candidates` backward passes of a parallel frontier batch are
    /// stacked, and (#mr-stack, throughput plan B5) so are the domains' MAIN
    /// eval passes while [`Self::domain_stack_main_eval`] holds; candidate
    /// selection, the β / domain-alpha trials and the heap merge retain their
    /// established per-domain paths and deterministic ordering.
    pub domain_stack: bool,
    /// #mr-stack (throughput plan B5): with `domain_stack`, run the batch's
    /// main eval passes as ONE stacked pass. NOT A LEVER: bit-identical by
    /// construction (each domain's block of the stacked pass equals its own
    /// pass bit-for-bit; pinned by `stacked_main_eval_bit_identical_to_per_domain`
    /// and `expand_batch_trajectory_hash_unchanged`), no preset key, no
    /// environment reader. Every production config carries the trusted code
    /// constant `bab::DOMAIN_STACK_MAIN_EVAL` — that constant is the kill switch —
    /// and the field exists only so the differential tests can run one tree
    /// both ways in one process.
    pub domain_stack_main_eval: bool,
    /// Immutable entry-time authority for per-domain alpha optimization.
    /// Every class and nested epoch driver inherits this policy unchanged.
    pub domain_alpha_policy: super::DomainAlphaRuntimePolicy,
    /// Immutable entry-time authority for the complete Tier-0/Tier-2 epoch
    /// family. Nested drivers carry this receipt-owned policy unchanged.
    pub epoch_policy: super::EpochRuntimePolicy,
    /// Evidence-only identity of this driver frame.
    pub epoch_frame_route: EpochFrameRoute,
    /// Retained tableau rows for the Tier-0 trunk variant ranker
    /// (`RootGates::build_retaining`). Ranker-only; shared with the parallel
    /// frontier via `Arc`.
    pub retained: Option<Arc<RetainedRows>>,
    /// Operational Tier-2 cadence for this tree. The entry tree seeds it from
    /// `epoch_policy.depth`; a nested full rebuild sets it to zero, while the
    /// closure controller may arm its own cadence only when the configured
    /// depth is zero. This separation prevents adaptive rollback from erasing
    /// an explicit scored treatment.
    pub epoch_trigger_depth: usize,
    /// Operational failed-attempt cap for this tree. The entry tree seeds it
    /// from `epoch_policy.max_attempts`; nested full rebuilds set it to zero.
    pub epoch_attempt_cap: usize,
    /// Remaining incremental-epoch nesting generations on this branch. The
    /// lane entry seeds a fixed finite budget; every nested refresh decrements
    /// it, making the maximum call-stack depth explicit and non-resetting.
    pub epoch_parent_generations_left: usize,
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
    /// Use independent beta state for selected failing output columns.
    pub beta_per_column: bool,
    /// Maximum live per-column beta states, clamped to the historical 1..=32.
    pub beta_columns: usize,
    /// Prebuild deterministic y-packs for multi-domain frontier batches.
    pub parallel_prebuild: bool,
    /// Immutable authority for rebakes and nested epoch roots. Execution
    /// evidence is deliberately not read back into this policy.
    pub root_precision_policy: super::RootPrecisionRuntimePolicy,
    /// Run's backward-intermediate policy, carried into rebuild requests.
    /// The root builder excludes split epoch rebuilds from BI.
    pub backward_interm_policy: super::BackwardIntermRuntimePolicy,
    /// Root-build execution evidence for the root-precision family.
    pub root_precision_execution: super::RootPrecisionExecutionEvidence,
    /// Root-build execution evidence for the frozen backward-intermediate
    /// family. Evidence-only; no tree decision reads it.
    pub backward_interm_execution: super::BackwardIntermExecutionEvidence,
    /// One closed receipt for the entry root's standalone alpha and rebake
    /// phases. Evidence-only and never recomputed by nested class/tree runs.
    pub root_alpha_execution: super::RootAlphaExecutionEvidence,
    /// #wave-beta-bab (ENGINE_PROGRAM M6): the tree engine of the OUTER frame
    /// (`margin_row.tree_mode`, `_tree_wave_batch`, `_tree_wave_cert_threads`).
    /// The default `legacy` is today's best-first tree, bit for bit; `wave`
    /// runs [`wave`]'s engine, whose every close is certified by this lane's
    /// own evaluator and whose verdict passes the same cover gate. Nested
    /// epoch frames and the classwise schedule always run the legacy tree.
    pub wave: wave::WavePolicy,
    /// B2.3: an IN-PROCESS HARNESS's explicit arming of the frozen-dual tier
    /// on the outer frame (the route e2e tests run the tier through the
    /// lane's own `authorize_frame` stream). NOT A LEVER and never fed by the
    /// scored lane: `None` (every production config) is the governed gate
    /// `margin_row.frozen_dual_bab`, latched once per process exactly as
    /// before. The tier certifies every class it closes either way, so this
    /// field can only choose WHETHER the certified tier runs.
    pub frozen_dual_tier: Option<bool>,
    /// #mr-close-telemetry TEST-ONLY switch: skip every proof-integrity
    /// recording site of this driver (and of its class / nested epoch
    /// drivers, which inherit it). NOT A LEVER: the field does not exist
    /// outside `cfg(test)`, where `close_telemetry_on` is the constant
    /// `true`. It exists only so `close_telemetry_trajectory_hash_unchanged`
    /// can run one tree with and without the telemetry in one process and
    /// prove the decision trajectory bit-identical.
    #[cfg(test)]
    pub(crate) close_telemetry_off: bool,
}

/// #mr-close-telemetry: whether this driver records proof-integrity
/// telemetry. Always `true` outside tests (see the test-only
/// `BabConfig::close_telemetry_off`).
#[inline]
fn close_telemetry_on(cfg: &BabConfig) -> bool {
    #[cfg(test)]
    {
        !cfg.close_telemetry_off
    }
    #[cfg(not(test))]
    {
        let _ = cfg;
        true
    }
}

impl Default for BabConfig {
    fn default() -> Self {
        Self {
            k_head: 8,
            contextual_k_head: 8,
            k_trunk: 8,
            contextual_k_trunk: 8,
            effective_k_head: 8,
            effective_contextual_k_head: 8,
            effective_k_trunk: 8,
            effective_contextual_k_trunk: 8,
            adaptive: false,
            contextual_adaptive: false,
            max_expansions: 20_000,
            deadline: None,
            head_groups: None,
            publish_deadline: None,
            lru_cap: 6,
            frontier: 1,
            frontier_contextual: 1,
            domain_stack: false,
            domain_stack_main_eval: DOMAIN_STACK_MAIN_EVAL,
            domain_alpha_policy: super::DomainAlphaRuntimePolicy::default(),
            epoch_policy: super::EpochRuntimePolicy::default(),
            epoch_frame_route: EpochFrameRoute::Outer,
            retained: None,
            epoch_trigger_depth: 0,
            epoch_attempt_cap: 2,
            epoch_parent_generations_left: 0,
            root_build_secs: 0.0,
            retain_cfg: None,
            initial_heads: Vec::new(),
            initial_ybox: None,
            col_retire: false,
            beta_per_column: false,
            beta_columns: 8,
            parallel_prebuild: false,
            root_precision_policy: super::RootPrecisionRuntimePolicy::default(),
            backward_interm_policy: super::BackwardIntermRuntimePolicy::default(),
            root_precision_execution: super::RootPrecisionExecutionEvidence::default(),
            backward_interm_execution: super::BackwardIntermExecutionEvidence::default(),
            root_alpha_execution: super::RootAlphaExecutionEvidence::default(),
            wave: wave::WavePolicy::default(),
            frozen_dual_tier: None,
            #[cfg(test)]
            close_telemetry_off: false,
        }
    }
}

impl BabConfig {
    /// The instant after which a COMPLETED proof may no longer be published.
    /// Falls back to the search deadline, so `None` is byte-identical.
    #[must_use]
    pub fn effective_publish_deadline(&self) -> Option<Instant> {
        self.publish_deadline.or(self.deadline)
    }

    /// Canonicalize the operational epoch state at a public outer-tree entry.
    ///
    /// Nested epoch drivers bypass `run` and `run_classwise` through
    /// `run_inner`, so their deliberately suppressed/decremented operational
    /// values are preserved. Public callers, however, must not be able to
    /// execute one cadence or attempt cap while reporting a different frozen
    /// receipt policy.
    fn seed_outer_epoch(mut self) -> Self {
        self.epoch_frame_route = EpochFrameRoute::Outer;
        self.epoch_trigger_depth = self.epoch_policy.depth;
        self.epoch_attempt_cap = self.epoch_policy.max_attempts;
        self.epoch_parent_generations_left = super::epoch_parent_generation_budget(
            self.epoch_policy.parent_topk,
            self.epoch_policy.depth,
            self.epoch_policy.max_attempts,
        );
        self.effective_k_head = self.k_head;
        self.effective_contextual_k_head = self.contextual_k_head;
        self.effective_k_trunk = self.k_trunk;
        self.effective_contextual_k_trunk = self.contextual_k_trunk;
        self
    }
}

/// Stable route carried by the margin-row frontier execution receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrontierRoute {
    /// The established joint multi-margin tree, where the configured frontier
    /// controls best-first batching.
    Joint,
    /// The experimental classwise schedule, which forcibly serializes every
    /// per-class tree for exact global expansion accounting.
    Classwise,
}

impl FrontierRoute {
    /// Stable machine token consumed by `scripts/ny_search.py`.
    #[must_use]
    pub const fn stable_token(self) -> &'static str {
        match self {
            Self::Joint => "joint",
            Self::Classwise => "classwise",
        }
    }
}

/// Per-invocation evidence that a frontier override changed a real tree
/// scheduling decision relative to the exact lower-layer width policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrontierExecutionEvidence {
    /// Tree route that owned these counters.
    pub route: FrontierRoute,
    /// Effective configured frontier width (`0` is normalized to serial `1`).
    pub configured: usize,
    /// Effective lower-layer width in this invocation's Rayon context.
    pub contextual: usize,
    /// Tree scheduling decisions that reached executable domain work.
    pub steps: usize,
    /// Decisions whose planned batch differs from the contextual policy.
    pub counterfactual_differences: usize,
    /// Largest open heap at an observed decision.
    pub max_open: usize,
    /// Largest configured planned batch at an observed decision.
    pub max_batch: usize,
}

impl FrontierExecutionEvidence {
    #[must_use]
    pub fn joint(configured: usize, contextual: usize) -> Self {
        Self {
            route: FrontierRoute::Joint,
            configured: configured.max(1),
            contextual: contextual.max(1),
            steps: 0,
            counterfactual_differences: 0,
            max_open: 0,
            max_batch: 0,
        }
    }

    #[must_use]
    pub fn classwise() -> Self {
        Self {
            route: FrontierRoute::Classwise,
            configured: 1,
            contextual: 1,
            steps: 0,
            counterfactual_differences: 0,
            max_open: 0,
            max_batch: 0,
        }
    }

    /// Observe one executable best-first decision.  `open` is captured before
    /// the selected domain(s) leave the heap, after the loop's budget/deadline
    /// gates.  The planned batch is therefore an exact `min(width, open)` for
    /// both the configured and exact lower-layer policies.
    pub(crate) fn observe(&mut self, open: usize) {
        if open == 0 {
            return;
        }
        let batch = self.configured.min(open);
        let contextual_batch = self.contextual.min(open);
        self.steps = self.steps.saturating_add(1);
        if batch != contextual_batch {
            self.counterfactual_differences = self.counterfactual_differences.saturating_add(1);
        }
        self.max_open = self.max_open.max(open);
        self.max_batch = self.max_batch.max(batch);
    }

    fn absorb(&mut self, other: Self) {
        debug_assert_eq!(self.route, other.route);
        debug_assert_eq!(self.configured, other.configured);
        debug_assert_eq!(self.contextual, other.contextual);
        if self.route != other.route
            || self.configured != other.configured
            || self.contextual != other.contextual
        {
            return;
        }
        self.steps = self.steps.saturating_add(other.steps);
        self.counterfactual_differences = self
            .counterfactual_differences
            .saturating_add(other.counterfactual_differences);
        self.max_open = self.max_open.max(other.max_open);
        self.max_batch = self.max_batch.max(other.max_batch);
    }
}

/// Frozen branch-width policy plus causal execution counters for the ordinary
/// head/trunk selectors. Tier-0 has its own ranker and deliberately contributes
/// no selector credit here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WidthExecutionEvidence {
    /// Configured immutable head-width base.
    pub configured_k_head: usize,
    /// Exact lower-authority head-width base.
    pub contextual_k_head: usize,
    /// Configured immutable trunk-width base.
    pub configured_k_trunk: usize,
    /// Exact lower-authority trunk-width base.
    pub contextual_k_trunk: usize,
    /// Configured adaptive-width authority.
    pub adaptive_enabled: bool,
    /// Exact lower-authority adaptive-width authority.
    pub contextual_adaptive: bool,
    /// Frozen Tier-0 exact width; a nonzero executable Tier-0 route bypasses
    /// both ordinary selectors.
    pub tier0_exact: usize,
    /// Largest operational head width reached by an ordinary selector route.
    pub effective_k_head_max: usize,
    /// Largest operational trunk width reached by an ordinary selector route.
    pub effective_k_trunk_max: usize,
    /// Largest exact lower-authority head width reached by an ordinary route.
    pub effective_contextual_k_head_max: usize,
    /// Largest exact lower-authority trunk width reached by an ordinary route.
    pub effective_contextual_k_trunk_max: usize,
    /// Ordinary outer/classwise head selector calls.
    pub outer_head_selection_calls: usize,
    /// Largest eligible head identity set seen by an outer selector.
    pub outer_head_available_max: usize,
    /// Head identities actually submitted across outer exact-score calls.
    pub outer_head_selected_total: usize,
    /// Successful exact-score calls whose actual head identity set differed
    /// from the exact contextual-width set.
    pub outer_head_binding_scored_calls: usize,
    /// Ordinary outer/classwise trunk selector calls.
    pub outer_trunk_selection_calls: usize,
    /// Largest eligible trunk identity set seen by an outer selector.
    pub outer_trunk_available_max: usize,
    /// Trunk identities actually submitted across outer exact-score calls.
    pub outer_trunk_selected_total: usize,
    /// Successful exact-score calls whose actual trunk identity set differed
    /// from the exact contextual-width set.
    pub outer_trunk_binding_scored_calls: usize,
    /// Ordinary head selector calls inside nested epoch trees.
    pub nested_head_selection_calls: usize,
    /// Ordinary trunk selector calls inside nested epoch trees.
    pub nested_trunk_selection_calls: usize,
    /// Frontier observations that reached a configured adaptive threshold on
    /// an ordinary selector route.
    pub configured_adaptive_threshold_opportunities: usize,
    /// Configured threshold observations that increased at least one
    /// operational treatment width.
    pub configured_adaptive_escalations: usize,
    /// Frontier observations that reached the exact lower-authority adaptive
    /// threshold on an ordinary selector route.
    pub contextual_adaptive_threshold_opportunities: usize,
    /// Contextual threshold observations that increased at least one exact
    /// lower-authority operational width.
    pub contextual_adaptive_escalations: usize,
    /// A class/nested merge observed a different supposedly-frozen policy.
    pub policy_mismatch: bool,
}

impl WidthExecutionEvidence {
    fn from_config(config: &BabConfig) -> Self {
        Self {
            configured_k_head: config.k_head,
            contextual_k_head: config.contextual_k_head,
            configured_k_trunk: config.k_trunk,
            contextual_k_trunk: config.contextual_k_trunk,
            adaptive_enabled: config.adaptive,
            contextual_adaptive: config.contextual_adaptive,
            tier0_exact: config.epoch_policy.tier0_exact,
            effective_k_head_max: config.effective_k_head,
            effective_k_trunk_max: config.effective_k_trunk,
            effective_contextual_k_head_max: config.effective_contextual_k_head,
            effective_contextual_k_trunk_max: config.effective_contextual_k_trunk,
            outer_head_selection_calls: 0,
            outer_head_available_max: 0,
            outer_head_selected_total: 0,
            outer_head_binding_scored_calls: 0,
            outer_trunk_selection_calls: 0,
            outer_trunk_available_max: 0,
            outer_trunk_selected_total: 0,
            outer_trunk_binding_scored_calls: 0,
            nested_head_selection_calls: 0,
            nested_trunk_selection_calls: 0,
            configured_adaptive_threshold_opportunities: 0,
            configured_adaptive_escalations: 0,
            contextual_adaptive_threshold_opportunities: 0,
            contextual_adaptive_escalations: 0,
            policy_mismatch: false,
        }
    }

    fn same_policy(&self, other: &Self) -> bool {
        self.configured_k_head == other.configured_k_head
            && self.contextual_k_head == other.contextual_k_head
            && self.configured_k_trunk == other.configured_k_trunk
            && self.contextual_k_trunk == other.contextual_k_trunk
            && self.adaptive_enabled == other.adaptive_enabled
            && self.contextual_adaptive == other.contextual_adaptive
            && self.tier0_exact == other.tier0_exact
    }

    fn absorb(&mut self, other: Self) {
        if !self.same_policy(&other) {
            self.policy_mismatch = true;
            return;
        }
        self.policy_mismatch |= other.policy_mismatch;
        self.effective_k_head_max = self.effective_k_head_max.max(other.effective_k_head_max);
        self.effective_k_trunk_max = self.effective_k_trunk_max.max(other.effective_k_trunk_max);
        self.effective_contextual_k_head_max = self
            .effective_contextual_k_head_max
            .max(other.effective_contextual_k_head_max);
        self.effective_contextual_k_trunk_max = self
            .effective_contextual_k_trunk_max
            .max(other.effective_contextual_k_trunk_max);
        self.outer_head_available_max = self
            .outer_head_available_max
            .max(other.outer_head_available_max);
        self.outer_trunk_available_max = self
            .outer_trunk_available_max
            .max(other.outer_trunk_available_max);
        macro_rules! add {
            ($($field:ident),+ $(,)?) => {
                $(self.$field = self.$field.saturating_add(other.$field);)+
            };
        }
        add!(
            outer_head_selection_calls,
            outer_head_selected_total,
            outer_head_binding_scored_calls,
            outer_trunk_selection_calls,
            outer_trunk_selected_total,
            outer_trunk_binding_scored_calls,
            nested_head_selection_calls,
            nested_trunk_selection_calls,
            configured_adaptive_threshold_opportunities,
            configured_adaptive_escalations,
            contextual_adaptive_threshold_opportunities,
            contextual_adaptive_escalations,
        );
    }

    fn observe_selection(
        &mut self,
        route: EpochFrameRoute,
        observation: WidthSelectionObservation,
    ) {
        match route {
            EpochFrameRoute::Outer => {
                self.outer_head_selection_calls = self.outer_head_selection_calls.saturating_add(1);
                self.outer_head_available_max = self
                    .outer_head_available_max
                    .max(observation.head_available);
                self.outer_head_selected_total = self
                    .outer_head_selected_total
                    .saturating_add(observation.head_selected);
                self.outer_trunk_selection_calls =
                    self.outer_trunk_selection_calls.saturating_add(1);
                self.outer_trunk_available_max = self
                    .outer_trunk_available_max
                    .max(observation.trunk_available);
                self.outer_trunk_selected_total = self
                    .outer_trunk_selected_total
                    .saturating_add(observation.trunk_selected);
            }
            EpochFrameRoute::NestedFull | EpochFrameRoute::NestedIncremental => {
                self.nested_head_selection_calls =
                    self.nested_head_selection_calls.saturating_add(1);
                self.nested_trunk_selection_calls =
                    self.nested_trunk_selection_calls.saturating_add(1);
            }
        }
    }

    fn observe_successful_score(
        &mut self,
        route: EpochFrameRoute,
        observation: WidthSelectionObservation,
    ) {
        if route != EpochFrameRoute::Outer {
            return;
        }
        if observation.head_counterfactual {
            self.outer_head_binding_scored_calls =
                self.outer_head_binding_scored_calls.saturating_add(1);
        }
        if observation.trunk_counterfactual {
            self.outer_trunk_binding_scored_calls =
                self.outer_trunk_binding_scored_calls.saturating_add(1);
        }
    }

    fn observe_adaptive_threshold(
        &mut self,
        ordinary_route: bool,
        configured_old: (usize, usize),
        configured_new: (usize, usize),
        contextual_old: (usize, usize),
        contextual_new: (usize, usize),
    ) {
        if !ordinary_route {
            return;
        }
        if self.adaptive_enabled {
            self.configured_adaptive_threshold_opportunities = self
                .configured_adaptive_threshold_opportunities
                .saturating_add(1);
            if configured_old != configured_new {
                self.configured_adaptive_escalations =
                    self.configured_adaptive_escalations.saturating_add(1);
                self.effective_k_head_max = self.effective_k_head_max.max(configured_new.0);
                self.effective_k_trunk_max = self.effective_k_trunk_max.max(configured_new.1);
            }
        }
        if self.contextual_adaptive {
            self.contextual_adaptive_threshold_opportunities = self
                .contextual_adaptive_threshold_opportunities
                .saturating_add(1);
            if contextual_old != contextual_new {
                self.contextual_adaptive_escalations =
                    self.contextual_adaptive_escalations.saturating_add(1);
                self.effective_contextual_k_head_max =
                    self.effective_contextual_k_head_max.max(contextual_new.0);
                self.effective_contextual_k_trunk_max =
                    self.effective_contextual_k_trunk_max.max(contextual_new.1);
            }
        }
    }
}

impl Default for WidthExecutionEvidence {
    fn default() -> Self {
        Self::from_config(&BabConfig::default())
    }
}

/// Selection-local identity comparison retained until the actual exact score
/// pass succeeds. The counterfactual booleans are computed from complete,
/// deterministic actual and contextual identity lists, not candidate counts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct WidthSelectionObservation {
    head_available: usize,
    head_selected: usize,
    head_counterfactual: bool,
    trunk_available: usize,
    trunk_selected: usize,
    trunk_counterfactual: bool,
}

/// Frozen policy plus counters proving that the new governed margin-row
/// controls reached executable work rather than merely appearing in a receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolicyExecutionEvidence {
    /// Frozen per-column beta gate.
    pub beta_per_column: bool,
    /// Frozen maximum beta-column width after the historical clamp.
    pub beta_columns: usize,
    /// Frozen certificate-backed column-retirement gate.
    pub column_retire: bool,
    /// Frozen parallel-prebuild gate.
    pub parallel_prebuild: bool,
    /// Frozen #clip-head gate (head-split halfspaces).
    pub clip_head: bool,
    /// Frozen per-domain alpha gate.
    pub domain_alpha_enabled: bool,
    /// Frozen per-domain alpha trial cap.
    pub domain_alpha_iters: usize,
    /// Frozen initial per-domain alpha step, stored as exact f64 bits.
    pub domain_alpha_eta_bits: u64,
    /// Evaluations where at least one head-split halfspace was actually built
    /// and offered to the dual. ZERO with the gate armed means the run is VOID,
    /// not negative.
    pub clip_head_evals: usize,
    /// Head columns whose interval strictly shrank under those halfspaces.
    pub clip_head_moved: usize,
    /// Head columns that became STABLE under them (abc's actual mechanism).
    pub clip_head_closes: usize,
    /// Head-clip steps refused: a crossed interval, a refused halfspace build,
    /// or a pack whose certified-slack frame check failed.
    pub clip_head_refused: usize,
    /// Live node evaluations whose first non-deadline ascent iteration
    /// consumed a per-column candidate set.
    pub beta_per_column_evals: usize,
    /// Per-column evaluations where this width changed the selected set
    /// relative to the historical width 8.
    pub beta_columns_binding_evals: usize,
    /// Largest live failing-column candidate set observed.
    pub beta_columns_max_candidates: usize,
    /// Columns newly certified and placed in an inherited retirement set.
    pub newly_retired_columns: usize,
    /// Descendant evaluations actually narrowed by an inherited retirement.
    pub inherited_retirement_evals: usize,
    /// Multi-domain batches for which at least one missing pack was prebuilt.
    pub parallel_prebuild_batches: usize,
    /// Missing domain packs actually built through the prebuild path.
    pub parallel_prebuild_domains: usize,
    /// Armed evaluations that reached the per-domain alpha admission site.
    pub domain_alpha_gate_evals: usize,
    /// Armed evaluations with at least one movable relaxation triangle.
    pub domain_alpha_movable_evals: usize,
    /// Movable relaxation triangles observed across those evaluations.
    pub domain_alpha_movable_slots: usize,
    /// Ancestor alpha entries actually applied to eligible child slots.
    pub domain_alpha_warm_applied: usize,
    /// Evaluations that started at least one certified alpha trial.
    pub domain_alpha_ascent_calls: usize,
    /// Certified alpha proposals that reached the scorer.
    pub domain_alpha_trials_started: usize,
    /// Trial passes that returned a scoreable result.
    pub domain_alpha_trial_passes_completed: usize,
    /// Trial passes that failed before producing a result.
    pub domain_alpha_trial_pass_failures: usize,
    /// Completed trial passes whose certified composite was finite.
    pub domain_alpha_finite_trial_scores: usize,
    /// Scored proposals whose certified composite improved and was retained.
    pub domain_alpha_trials_accepted: usize,
    /// Sparse accepted alpha entries carried from evaluated nodes to children.
    pub domain_alpha_carried_entries: usize,
    /// Domains where the configured trial cap causally differed from two.
    pub domain_alpha_iters_counterfactual_domains: usize,
    /// Domains where an ETA-different proposal received a finite score.
    pub domain_alpha_eta_counterfactual_scored_domains: usize,
    /// Outer epoch-frame domains that started at least one trial.
    pub domain_alpha_outer_ascent_calls: usize,
    /// Finite certified trial scores produced in outer epoch frames.
    pub domain_alpha_outer_finite_trial_scores: usize,
    /// Outer domains with a causally binding iteration counterfactual.
    pub domain_alpha_outer_iters_counterfactual_domains: usize,
    /// Outer domains with a finite-scored ETA counterfactual.
    pub domain_alpha_outer_eta_counterfactual_scored_domains: usize,
    /// A nested/classwise stats merge observed a different supposedly-frozen
    /// policy. Such a run may still carry a sound verdict, but its optimizer
    /// attribution must fail closed.
    pub policy_mismatch: bool,
}

impl PolicyExecutionEvidence {
    fn from_config(config: &BabConfig) -> Self {
        let domain_alpha = config.domain_alpha_policy.normalized();
        Self {
            beta_per_column: config.beta_per_column,
            beta_columns: config.beta_columns.clamp(1, 32),
            column_retire: config.col_retire,
            parallel_prebuild: config.parallel_prebuild,
            clip_head: crate::preset_gates::margin_row_clip_head_enabled(),
            domain_alpha_enabled: domain_alpha.enabled,
            domain_alpha_iters: domain_alpha.iters,
            domain_alpha_eta_bits: domain_alpha.eta.to_bits(),
            clip_head_evals: 0,
            clip_head_moved: 0,
            clip_head_closes: 0,
            clip_head_refused: 0,
            beta_per_column_evals: 0,
            beta_columns_binding_evals: 0,
            beta_columns_max_candidates: 0,
            newly_retired_columns: 0,
            inherited_retirement_evals: 0,
            parallel_prebuild_batches: 0,
            parallel_prebuild_domains: 0,
            domain_alpha_gate_evals: 0,
            domain_alpha_movable_evals: 0,
            domain_alpha_movable_slots: 0,
            domain_alpha_warm_applied: 0,
            domain_alpha_ascent_calls: 0,
            domain_alpha_trials_started: 0,
            domain_alpha_trial_passes_completed: 0,
            domain_alpha_trial_pass_failures: 0,
            domain_alpha_finite_trial_scores: 0,
            domain_alpha_trials_accepted: 0,
            domain_alpha_carried_entries: 0,
            domain_alpha_iters_counterfactual_domains: 0,
            domain_alpha_eta_counterfactual_scored_domains: 0,
            domain_alpha_outer_ascent_calls: 0,
            domain_alpha_outer_finite_trial_scores: 0,
            domain_alpha_outer_iters_counterfactual_domains: 0,
            domain_alpha_outer_eta_counterfactual_scored_domains: 0,
            policy_mismatch: false,
        }
    }

    fn absorb(&mut self, other: Self) {
        if self.beta_per_column != other.beta_per_column
            || self.beta_columns != other.beta_columns
            || self.column_retire != other.column_retire
            || self.parallel_prebuild != other.parallel_prebuild
            || self.domain_alpha_enabled != other.domain_alpha_enabled
            || self.domain_alpha_iters != other.domain_alpha_iters
            || self.domain_alpha_eta_bits != other.domain_alpha_eta_bits
        {
            self.policy_mismatch = true;
            return;
        }
        self.policy_mismatch |= other.policy_mismatch;
        // The #clip-head gate is process-wide and read per call, not frozen from
        // `BabConfig`, so a difference here is not a policy mismatch — it is a
        // nested driver that resolved the same global. OR it in rather than
        // failing the whole attribution closed.
        self.clip_head |= other.clip_head;
        self.clip_head_evals = self.clip_head_evals.saturating_add(other.clip_head_evals);
        self.clip_head_moved = self.clip_head_moved.saturating_add(other.clip_head_moved);
        self.clip_head_closes = self.clip_head_closes.saturating_add(other.clip_head_closes);
        self.clip_head_refused = self
            .clip_head_refused
            .saturating_add(other.clip_head_refused);
        self.beta_per_column_evals = self
            .beta_per_column_evals
            .saturating_add(other.beta_per_column_evals);
        self.beta_columns_binding_evals = self
            .beta_columns_binding_evals
            .saturating_add(other.beta_columns_binding_evals);
        self.beta_columns_max_candidates = self
            .beta_columns_max_candidates
            .max(other.beta_columns_max_candidates);
        self.newly_retired_columns = self
            .newly_retired_columns
            .saturating_add(other.newly_retired_columns);
        self.inherited_retirement_evals = self
            .inherited_retirement_evals
            .saturating_add(other.inherited_retirement_evals);
        self.parallel_prebuild_batches = self
            .parallel_prebuild_batches
            .saturating_add(other.parallel_prebuild_batches);
        self.parallel_prebuild_domains = self
            .parallel_prebuild_domains
            .saturating_add(other.parallel_prebuild_domains);
        self.domain_alpha_gate_evals = self
            .domain_alpha_gate_evals
            .saturating_add(other.domain_alpha_gate_evals);
        self.domain_alpha_movable_evals = self
            .domain_alpha_movable_evals
            .saturating_add(other.domain_alpha_movable_evals);
        self.domain_alpha_movable_slots = self
            .domain_alpha_movable_slots
            .saturating_add(other.domain_alpha_movable_slots);
        self.domain_alpha_warm_applied = self
            .domain_alpha_warm_applied
            .saturating_add(other.domain_alpha_warm_applied);
        self.domain_alpha_ascent_calls = self
            .domain_alpha_ascent_calls
            .saturating_add(other.domain_alpha_ascent_calls);
        self.domain_alpha_trials_started = self
            .domain_alpha_trials_started
            .saturating_add(other.domain_alpha_trials_started);
        self.domain_alpha_trial_passes_completed = self
            .domain_alpha_trial_passes_completed
            .saturating_add(other.domain_alpha_trial_passes_completed);
        self.domain_alpha_trial_pass_failures = self
            .domain_alpha_trial_pass_failures
            .saturating_add(other.domain_alpha_trial_pass_failures);
        self.domain_alpha_finite_trial_scores = self
            .domain_alpha_finite_trial_scores
            .saturating_add(other.domain_alpha_finite_trial_scores);
        self.domain_alpha_trials_accepted = self
            .domain_alpha_trials_accepted
            .saturating_add(other.domain_alpha_trials_accepted);
        self.domain_alpha_carried_entries = self
            .domain_alpha_carried_entries
            .saturating_add(other.domain_alpha_carried_entries);
        self.domain_alpha_iters_counterfactual_domains = self
            .domain_alpha_iters_counterfactual_domains
            .saturating_add(other.domain_alpha_iters_counterfactual_domains);
        self.domain_alpha_eta_counterfactual_scored_domains = self
            .domain_alpha_eta_counterfactual_scored_domains
            .saturating_add(other.domain_alpha_eta_counterfactual_scored_domains);
        self.domain_alpha_outer_ascent_calls = self
            .domain_alpha_outer_ascent_calls
            .saturating_add(other.domain_alpha_outer_ascent_calls);
        self.domain_alpha_outer_finite_trial_scores = self
            .domain_alpha_outer_finite_trial_scores
            .saturating_add(other.domain_alpha_outer_finite_trial_scores);
        self.domain_alpha_outer_iters_counterfactual_domains = self
            .domain_alpha_outer_iters_counterfactual_domains
            .saturating_add(other.domain_alpha_outer_iters_counterfactual_domains);
        self.domain_alpha_outer_eta_counterfactual_scored_domains = self
            .domain_alpha_outer_eta_counterfactual_scored_domains
            .saturating_add(other.domain_alpha_outer_eta_counterfactual_scored_domains);
    }
}

impl Default for PolicyExecutionEvidence {
    fn default() -> Self {
        Self::from_config(&BabConfig::default())
    }
}

/// Frozen epoch policy plus causal execution counters for the Tier-0 ranker
/// and Tier-2 rebuild paths. Evidence is accumulated beside, but never read by,
/// bound or verdict code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpochExecutionEvidence {
    /// Receipt-owned policy that every class and nested driver must inherit.
    pub policy: super::EpochRuntimePolicy,
    /// Popped live domains that reached the epoch gate.
    pub tree_entries: usize,
    /// Attempts admitted by the explicitly configured depth.
    pub configured_depth_admissions: usize,
    /// Configured-depth attempts whose nested run returned owned stats.
    pub configured_depth_nested_with_stats: usize,
    /// Attempts admitted only by the closure-adaptive cadence.
    pub adaptive_depth_admissions: usize,
    /// Failure-cap decisions made after depth/generation admission.
    pub attempt_cap_decisions: usize,
    /// Decisions that differ from the historical cap of two failures.
    pub attempt_cap_counterfactual_differences: usize,
    /// Attempts refused by the configured operational failure cap.
    pub attempt_cap_refusals: usize,
    /// Eligible attempts refused because incremental generations were spent.
    pub generation_refusals: usize,
    /// Eligible attempts refused by the rebuild affordability guard.
    pub deadline_refusals: usize,
    /// Epoch builds that began after every admission gate.
    pub attempts_started: usize,
    /// Started attempts that ended without closing the outer domain.
    pub attempt_failures: usize,
    /// Started attempts that certified and closed the outer domain.
    pub attempts_closed: usize,
    /// Full attempts started by an outer/class tree.
    pub outer_full_attempts: usize,
    /// Incremental attempts started by an outer/class tree.
    pub outer_incremental_attempts: usize,
    /// Full attempts started inside a nested epoch frame.
    pub nested_full_attempts: usize,
    /// Incremental attempts started inside a nested epoch frame.
    pub nested_incremental_attempts: usize,
    /// Full-tableau builds returned a candidate root frame.
    pub full_rebuilds_completed: usize,
    /// Full-tableau build or interval-inheritance refusals.
    pub full_rebuild_refusals: usize,
    /// Incremental parent-frame refreshes that started.
    pub parent_refreshes_started: usize,
    /// Incremental parent-frame refreshes returned a candidate root frame.
    pub parent_refreshes_completed: usize,
    /// Incremental refreshes accepted for a nested run.
    pub parent_refreshes_accepted: usize,
    /// Incremental refresh errors or zero-progress returns.
    pub parent_refresh_refusals: usize,
    /// Certified prefix columns completed by incremental refreshes.
    pub parent_refresh_columns: usize,
    /// Incremental columns whose inherited interval shrank.
    pub parent_refresh_tightened: usize,
    /// Incremental columns that became sign-stable.
    pub parent_refresh_stabilized: usize,
    /// Nested epoch trees actually started.
    pub nested_started: usize,
    /// Nested results that carried execution stats (Unsat or Unknown+stats).
    pub nested_runs_with_stats: usize,
    /// Nested trees that certified their subtree.
    pub nested_unsat: usize,
    /// Nested trees that returned Unknown.
    pub nested_unknown: usize,
    /// Nested trees that returned an internal error.
    pub nested_errors: usize,
    /// Outer domains closed by a nested epoch verdict.
    pub closures: usize,
    /// Debug trace attempts emitted under the frozen diagnostic bit.
    pub debug_emissions: usize,
    /// Outer/class candidate selections that entered Tier-0.
    pub outer_tier0_calls: usize,
    /// Outer Tier-0 selections whose certified candidate score pass completed.
    pub outer_tier0_scored_calls: usize,
    /// Completed outer calls where exact-width truncation removed candidates.
    pub outer_tier0_exact_binding_scored_calls: usize,
    /// Completed outer calls where universe selection differed from width 64.
    pub outer_universe_counterfactual_scored_calls: usize,
    /// Completed outer calls where consumed retention differed from 128.
    pub outer_retain_counterfactual_scored_calls: usize,
    /// Tier-0 selections executed inside a full nested frame.
    pub nested_full_tier0_calls: usize,
    /// Tier-0 selections executed inside an incremental nested frame.
    pub nested_incremental_tier0_calls: usize,
    /// Nested/classwise evidence carried a different supposedly-frozen policy.
    pub policy_mismatch: bool,
}

impl EpochExecutionEvidence {
    fn from_config(config: &BabConfig) -> Self {
        Self {
            policy: config.epoch_policy,
            tree_entries: 0,
            configured_depth_admissions: 0,
            configured_depth_nested_with_stats: 0,
            adaptive_depth_admissions: 0,
            attempt_cap_decisions: 0,
            attempt_cap_counterfactual_differences: 0,
            attempt_cap_refusals: 0,
            generation_refusals: 0,
            deadline_refusals: 0,
            attempts_started: 0,
            attempt_failures: 0,
            attempts_closed: 0,
            outer_full_attempts: 0,
            outer_incremental_attempts: 0,
            nested_full_attempts: 0,
            nested_incremental_attempts: 0,
            full_rebuilds_completed: 0,
            full_rebuild_refusals: 0,
            parent_refreshes_started: 0,
            parent_refreshes_completed: 0,
            parent_refreshes_accepted: 0,
            parent_refresh_refusals: 0,
            parent_refresh_columns: 0,
            parent_refresh_tightened: 0,
            parent_refresh_stabilized: 0,
            nested_started: 0,
            nested_runs_with_stats: 0,
            nested_unsat: 0,
            nested_unknown: 0,
            nested_errors: 0,
            closures: 0,
            debug_emissions: 0,
            outer_tier0_calls: 0,
            outer_tier0_scored_calls: 0,
            outer_tier0_exact_binding_scored_calls: 0,
            outer_universe_counterfactual_scored_calls: 0,
            outer_retain_counterfactual_scored_calls: 0,
            nested_full_tier0_calls: 0,
            nested_incremental_tier0_calls: 0,
            policy_mismatch: false,
        }
    }

    fn absorb(&mut self, other: Self) {
        if self.policy != other.policy {
            self.policy_mismatch = true;
            return;
        }
        self.policy_mismatch |= other.policy_mismatch;
        macro_rules! add {
            ($($field:ident),+ $(,)?) => {
                $(self.$field = self.$field.saturating_add(other.$field);)+
            };
        }
        add!(
            tree_entries,
            configured_depth_admissions,
            configured_depth_nested_with_stats,
            adaptive_depth_admissions,
            attempt_cap_decisions,
            attempt_cap_counterfactual_differences,
            attempt_cap_refusals,
            generation_refusals,
            deadline_refusals,
            attempts_started,
            attempt_failures,
            attempts_closed,
            outer_full_attempts,
            outer_incremental_attempts,
            nested_full_attempts,
            nested_incremental_attempts,
            full_rebuilds_completed,
            full_rebuild_refusals,
            parent_refreshes_started,
            parent_refreshes_completed,
            parent_refreshes_accepted,
            parent_refresh_refusals,
            parent_refresh_columns,
            parent_refresh_tightened,
            parent_refresh_stabilized,
            nested_started,
            nested_runs_with_stats,
            nested_unsat,
            nested_unknown,
            nested_errors,
            closures,
            debug_emissions,
            outer_tier0_calls,
            outer_tier0_scored_calls,
            outer_tier0_exact_binding_scored_calls,
            outer_universe_counterfactual_scored_calls,
            outer_retain_counterfactual_scored_calls,
            nested_full_tier0_calls,
            nested_incremental_tier0_calls,
        );
    }
}

impl Default for EpochExecutionEvidence {
    fn default() -> Self {
        Self::from_config(&BabConfig::default())
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct EpochSelectionObservation {
    tier0: bool,
    exact_binding: bool,
    universe_counterfactual: bool,
    retain_counterfactual: bool,
}

impl EpochExecutionEvidence {
    fn observe_selection(
        &mut self,
        route: EpochFrameRoute,
        observation: EpochSelectionObservation,
    ) {
        if !observation.tier0 {
            return;
        }
        match route {
            EpochFrameRoute::Outer => {
                self.outer_tier0_calls = self.outer_tier0_calls.saturating_add(1);
            }
            EpochFrameRoute::NestedFull => {
                self.nested_full_tier0_calls = self.nested_full_tier0_calls.saturating_add(1);
            }
            EpochFrameRoute::NestedIncremental => {
                self.nested_incremental_tier0_calls =
                    self.nested_incremental_tier0_calls.saturating_add(1);
            }
        }
    }

    fn observe_scored_selection(
        &mut self,
        route: EpochFrameRoute,
        observation: EpochSelectionObservation,
    ) {
        if route != EpochFrameRoute::Outer || !observation.tier0 {
            return;
        }
        self.outer_tier0_scored_calls = self.outer_tier0_scored_calls.saturating_add(1);
        if observation.exact_binding {
            self.outer_tier0_exact_binding_scored_calls = self
                .outer_tier0_exact_binding_scored_calls
                .saturating_add(1);
        }
        if observation.universe_counterfactual {
            self.outer_universe_counterfactual_scored_calls = self
                .outer_universe_counterfactual_scored_calls
                .saturating_add(1);
        }
        if observation.retain_counterfactual {
            self.outer_retain_counterfactual_scored_calls = self
                .outer_retain_counterfactual_scored_calls
                .saturating_add(1);
        }
    }
}

#[cfg(test)]
mod policy_execution_tests {
    use super::{BabConfig, PolicyExecutionEvidence};

    fn configured_policy() -> PolicyExecutionEvidence {
        PolicyExecutionEvidence::from_config(&BabConfig {
            beta_per_column: true,
            beta_columns: 4,
            col_retire: true,
            parallel_prebuild: true,
            domain_alpha_policy: super::super::DomainAlphaRuntimePolicy {
                enabled: true,
                iters: 4,
                eta: 0.5,
                min_depth: 0,
                stride: 1,
                gap: f64::INFINITY,
            },
            ..BabConfig::default()
        })
    }

    #[test]
    fn policy_execution_absorb_adds_maxima_and_poison_mismatches() {
        let mut aggregate = configured_policy();
        aggregate.beta_per_column_evals = 2;
        aggregate.beta_columns_binding_evals = 1;
        aggregate.beta_columns_max_candidates = 5;
        aggregate.newly_retired_columns = 3;
        aggregate.inherited_retirement_evals = 4;
        aggregate.parallel_prebuild_batches = 1;
        aggregate.parallel_prebuild_domains = 2;
        aggregate.domain_alpha_gate_evals = 1;
        aggregate.domain_alpha_finite_trial_scores = 2;
        aggregate.domain_alpha_trials_accepted = 1;
        aggregate.domain_alpha_outer_finite_trial_scores = 2;
        aggregate.domain_alpha_outer_iters_counterfactual_domains = 1;

        let mut other = configured_policy();
        other.beta_per_column_evals = 3;
        other.beta_columns_binding_evals = 2;
        other.beta_columns_max_candidates = 7;
        other.newly_retired_columns = 5;
        other.inherited_retirement_evals = 6;
        other.parallel_prebuild_batches = 2;
        other.parallel_prebuild_domains = 5;
        other.domain_alpha_gate_evals = 3;
        other.domain_alpha_finite_trial_scores = 6;
        other.domain_alpha_trials_accepted = 4;
        other.domain_alpha_outer_finite_trial_scores = 5;
        other.domain_alpha_outer_iters_counterfactual_domains = 2;
        aggregate.absorb(other);

        let mut merged = configured_policy();
        merged.beta_per_column_evals = 5;
        merged.beta_columns_binding_evals = 3;
        merged.beta_columns_max_candidates = 7;
        merged.newly_retired_columns = 8;
        merged.inherited_retirement_evals = 10;
        merged.parallel_prebuild_batches = 3;
        merged.parallel_prebuild_domains = 7;
        merged.domain_alpha_gate_evals = 4;
        merged.domain_alpha_finite_trial_scores = 8;
        merged.domain_alpha_trials_accepted = 5;
        merged.domain_alpha_outer_finite_trial_scores = 7;
        merged.domain_alpha_outer_iters_counterfactual_domains = 3;
        assert_eq!(aggregate, merged, "add/max aggregation drifted");

        let mut nested_mismatch = configured_policy();
        nested_mismatch.policy_mismatch = true;
        let mut propagated = merged;
        propagated.absorb(nested_mismatch);
        assert!(
            propagated.policy_mismatch,
            "a nested mismatch must poison its aggregate"
        );

        let mut different_policy = PolicyExecutionEvidence::from_config(&BabConfig {
            beta_per_column: true,
            beta_columns: 16,
            col_retire: true,
            parallel_prebuild: true,
            ..BabConfig::default()
        });
        different_policy.beta_per_column_evals = 99;
        different_policy.newly_retired_columns = 99;
        different_policy.parallel_prebuild_domains = 99;
        let mut mismatch_probe = merged;
        mismatch_probe.absorb(different_policy);
        let mut expected = merged;
        expected.policy_mismatch = true;
        assert_eq!(
            mismatch_probe, expected,
            "a frozen-policy mismatch must fail closed without mixing counters"
        );

        let mut domain_mismatch = configured_policy();
        domain_mismatch.domain_alpha_iters = 1;
        domain_mismatch.domain_alpha_finite_trial_scores = 99;
        let mut domain_probe = merged;
        domain_probe.absorb(domain_mismatch);
        let mut domain_expected = merged;
        domain_expected.policy_mismatch = true;
        assert_eq!(
            domain_probe, domain_expected,
            "domain-alpha policy drift must poison without mixing causal work"
        );
    }
}

#[cfg(test)]
mod width_execution_tests {
    use super::{
        ordinary_width_selectors_enabled, BabConfig, EpochFrameRoute, WidthExecutionEvidence,
        WidthSelectionObservation,
    };

    fn configured() -> WidthExecutionEvidence {
        WidthExecutionEvidence::from_config(&BabConfig {
            k_head: 16,
            contextual_k_head: 8,
            k_trunk: 12,
            contextual_k_trunk: 8,
            effective_k_head: 16,
            effective_contextual_k_head: 8,
            effective_k_trunk: 12,
            effective_contextual_k_trunk: 8,
            adaptive: true,
            contextual_adaptive: false,
            ..BabConfig::default()
        })
    }

    #[test]
    fn binding_requires_a_successful_exact_score_and_nested_counts_stay_separate() {
        let observation = WidthSelectionObservation {
            head_available: 24,
            head_selected: 16,
            head_counterfactual: true,
            trunk_available: 6,
            trunk_selected: 6,
            trunk_counterfactual: false,
        };
        let mut evidence = configured();
        evidence.observe_selection(EpochFrameRoute::Outer, observation);
        assert_eq!(evidence.outer_head_selection_calls, 1);
        assert_eq!(evidence.outer_head_available_max, 24);
        assert_eq!(evidence.outer_head_selected_total, 16);
        assert_eq!(evidence.outer_head_binding_scored_calls, 0);
        assert_eq!(evidence.outer_trunk_binding_scored_calls, 0);

        evidence.observe_successful_score(EpochFrameRoute::Outer, observation);
        assert_eq!(evidence.outer_head_binding_scored_calls, 1);
        assert_eq!(evidence.outer_trunk_binding_scored_calls, 0);

        evidence.observe_selection(EpochFrameRoute::NestedFull, observation);
        evidence.observe_successful_score(EpochFrameRoute::NestedFull, observation);
        assert_eq!(evidence.nested_head_selection_calls, 1);
        assert_eq!(evidence.nested_trunk_selection_calls, 1);
        assert_eq!(evidence.outer_head_selection_calls, 1);
        assert_eq!(evidence.outer_head_binding_scored_calls, 1);
    }

    #[test]
    fn aggregation_poison_preserves_owned_counters_on_policy_mismatch() {
        let mut aggregate = configured();
        aggregate.outer_head_selection_calls = 2;
        let mut nested = configured();
        nested.nested_head_selection_calls = 3;
        aggregate.absorb(nested);
        assert_eq!(aggregate.nested_head_selection_calls, 3);
        assert!(!aggregate.policy_mismatch);

        let mut mismatched = configured();
        mismatched.configured_k_head = 32;
        mismatched.outer_head_selection_calls = 99;
        aggregate.absorb(mismatched);
        assert!(aggregate.policy_mismatch);
        assert_eq!(aggregate.outer_head_selection_calls, 2);
    }

    #[test]
    fn adaptive_opportunities_escalate_once_and_tier0_bypasses_width_credit() {
        assert!(ordinary_width_selectors_enabled(0, true));
        assert!(ordinary_width_selectors_enabled(8, false));
        assert!(!ordinary_width_selectors_enabled(8, true));

        let mut evidence = configured();
        evidence.observe_adaptive_threshold(true, (16, 12), (16, 16), (8, 8), (8, 8));
        evidence.observe_adaptive_threshold(true, (16, 16), (16, 16), (8, 8), (8, 8));
        assert_eq!(evidence.configured_adaptive_threshold_opportunities, 2);
        assert_eq!(evidence.configured_adaptive_escalations, 1);
        assert_eq!(evidence.contextual_adaptive_threshold_opportunities, 0);
        assert_eq!(evidence.contextual_adaptive_escalations, 0);
        assert_eq!(evidence.effective_k_head_max, 16);
        assert_eq!(evidence.effective_k_trunk_max, 16);
        assert_eq!(evidence.effective_contextual_k_head_max, 8);
        assert_eq!(evidence.effective_contextual_k_trunk_max, 8);

        evidence.observe_adaptive_threshold(false, (16, 16), (32, 32), (8, 8), (8, 8));
        assert_eq!(evidence.configured_adaptive_threshold_opportunities, 2);
        assert_eq!(evidence.configured_adaptive_escalations, 1);
        assert_eq!(evidence.effective_k_head_max, 16);
    }

    #[test]
    fn contextual_adaptive_control_is_counted_independently() {
        let mut evidence = configured();
        evidence.configured_k_head = 8;
        evidence.configured_k_trunk = 8;
        evidence.adaptive_enabled = false;
        evidence.contextual_adaptive = true;
        evidence.effective_k_head_max = 8;
        evidence.effective_k_trunk_max = 8;
        evidence.effective_contextual_k_head_max = 8;
        evidence.effective_contextual_k_trunk_max = 8;

        evidence.observe_adaptive_threshold(true, (8, 8), (8, 8), (8, 8), (16, 16));
        assert_eq!(evidence.configured_adaptive_threshold_opportunities, 0);
        assert_eq!(evidence.configured_adaptive_escalations, 0);
        assert_eq!(evidence.contextual_adaptive_threshold_opportunities, 1);
        assert_eq!(evidence.contextual_adaptive_escalations, 1);
        assert_eq!(evidence.effective_k_head_max, 8);
        assert_eq!(evidence.effective_k_trunk_max, 8);
        assert_eq!(evidence.effective_contextual_k_head_max, 16);
        assert_eq!(evidence.effective_contextual_k_trunk_max, 16);
    }

    #[test]
    fn public_entry_reseeds_operational_widths_without_moving_frozen_policy() {
        let seeded = BabConfig {
            k_head: 16,
            contextual_k_head: 8,
            k_trunk: 12,
            contextual_k_trunk: 6,
            effective_k_head: 99,
            effective_contextual_k_head: 99,
            effective_k_trunk: 99,
            effective_contextual_k_trunk: 99,
            adaptive: true,
            contextual_adaptive: false,
            ..BabConfig::default()
        }
        .seed_outer_epoch();
        assert_eq!(
            (seeded.effective_k_head, seeded.effective_k_trunk),
            (16, 12)
        );
        assert_eq!(
            (
                seeded.effective_contextual_k_head,
                seeded.effective_contextual_k_trunk,
            ),
            (8, 6),
        );
        assert!(seeded.adaptive);
        assert!(!seeded.contextual_adaptive);
    }
}

#[cfg(test)]
mod epoch_execution_tests {
    use super::{BabConfig, EpochExecutionEvidence, EpochFrameRoute, EpochSelectionObservation};

    fn configured() -> EpochExecutionEvidence {
        EpochExecutionEvidence::from_config(&BabConfig {
            epoch_policy: super::super::EpochRuntimePolicy {
                depth: 3,
                max_attempts: 5,
                tier0_exact: 8,
                tier0_universe: 32,
                retain_per_layer: 64,
                parent_topk: 0,
                debug_trace: true,
            },
            ..BabConfig::default()
        })
    }

    #[test]
    fn public_outer_entry_reseeds_operational_epoch_state_from_policy() {
        let seeded = BabConfig {
            epoch_policy: super::super::EpochRuntimePolicy {
                depth: 3,
                max_attempts: 5,
                parent_topk: 16,
                ..super::super::EpochRuntimePolicy::default()
            },
            epoch_frame_route: EpochFrameRoute::NestedIncremental,
            epoch_trigger_depth: 99,
            epoch_attempt_cap: 101,
            epoch_parent_generations_left: 0,
            ..BabConfig::default()
        }
        .seed_outer_epoch();
        assert_eq!(seeded.epoch_frame_route, EpochFrameRoute::Outer);
        assert_eq!(seeded.epoch_trigger_depth, 3);
        assert_eq!(seeded.epoch_attempt_cap, 5);
        assert_eq!(
            seeded.epoch_parent_generations_left,
            super::EPOCH_PARENT_GENERATION_CAP
        );
    }

    #[test]
    fn tier0_credit_requires_outer_completed_scoring() {
        let observation = EpochSelectionObservation {
            tier0: true,
            exact_binding: true,
            universe_counterfactual: true,
            retain_counterfactual: true,
        };
        let mut outer = configured();
        outer.observe_selection(EpochFrameRoute::Outer, observation);
        assert_eq!(outer.outer_tier0_calls, 1);
        assert_eq!(outer.outer_tier0_scored_calls, 0);
        outer.observe_scored_selection(EpochFrameRoute::Outer, observation);
        assert_eq!(outer.outer_tier0_scored_calls, 1);
        assert_eq!(outer.outer_tier0_exact_binding_scored_calls, 1);
        assert_eq!(outer.outer_universe_counterfactual_scored_calls, 1);
        assert_eq!(outer.outer_retain_counterfactual_scored_calls, 1);

        let mut nested = configured();
        nested.observe_selection(EpochFrameRoute::NestedFull, observation);
        nested.observe_scored_selection(EpochFrameRoute::NestedFull, observation);
        assert_eq!(nested.nested_full_tier0_calls, 1);
        assert_eq!(nested.outer_tier0_calls, 0);
        assert_eq!(nested.outer_tier0_scored_calls, 0);
    }

    #[test]
    fn epoch_aggregation_adds_counters_and_poison_mismatches() {
        let mut aggregate = configured();
        aggregate.attempts_started = 2;
        aggregate.attempt_failures = 1;
        aggregate.attempts_closed = 1;
        aggregate.outer_full_attempts = 2;
        let mut nested = configured();
        nested.attempts_started = 3;
        nested.attempt_failures = 3;
        nested.nested_incremental_attempts = 3;
        aggregate.absorb(nested);
        assert_eq!(aggregate.attempts_started, 5);
        assert_eq!(aggregate.attempt_failures, 4);
        assert_eq!(aggregate.attempts_closed, 1);
        assert_eq!(aggregate.outer_full_attempts, 2);
        assert_eq!(aggregate.nested_incremental_attempts, 3);

        let mut different = configured();
        different.policy.max_attempts = 7;
        different.attempts_started = 99;
        aggregate.absorb(different);
        assert!(aggregate.policy_mismatch);
        assert_eq!(aggregate.attempts_started, 5);
    }
}

impl BabConfig {
    /// Epoch settings inherited by one nested subtree. The generation count is
    /// the non-resetting recursion budget: even though each nested driver owns
    /// its own failed-attempt counter, every recursive edge consumes one here.
    fn nested_epoch_limits(&self, incremental: bool) -> (usize, usize, usize) {
        let generations = self
            .epoch_parent_generations_left
            .min(EPOCH_PARENT_GENERATION_CAP);
        if !incremental || generations <= 1 {
            return (0, 0, 0);
        }
        (
            self.epoch_trigger_depth,
            self.epoch_attempt_cap,
            generations - 1,
        )
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
    /// This class tree's cover-gate result (see [`BabStats::ledger_ok`]).
    pub ledger_ok: Option<bool>,
    /// Forced-serial frontier decisions made by this class tree.
    pub frontier_execution: FrontierExecutionEvidence,
    /// Governed margin-row policy execution observed by this class tree.
    pub policy_execution: PolicyExecutionEvidence,
    /// Frozen epoch policy and causal work observed by this class tree.
    pub epoch_execution: EpochExecutionEvidence,
    /// Frozen width policy and ordinary-selector work observed by this class.
    pub width_execution: WidthExecutionEvidence,
}

/// #phase-cost: one finished margin-row tree's work denominators, added to the
/// run's cost census (expansions = domains explored, closed = domains proved).
/// Write-only telemetry: nothing reads these counters back.
fn note_phase_cost_tree(stats: &BabStats) {
    use ny_core::phase_cost::{count, CostCounter};
    count(CostCounter::DomainsExplored, stats.expansions as u64);
    count(CostCounter::DomainsVerified, stats.closed as u64);
    count(CostCounter::ChildrenCreated, stats.domains_created as u64);
    count(CostCounter::BabRuns, 1);
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
    /// Cover-gate verdict (#soundness-spine M2c; formerly the depth-only
    /// Kraft ledger of #epoch-bab Phase E, which never gated): `Some(true)`
    /// when the frame was AUTHORIZED — the certified leaves' PATHS cover the
    /// frame root (prefix-free, siblings split one site on opposite sides,
    /// nested epoch subtrees closed by their own sub-domain tokens) and every
    /// required class is covered; `Some(false)` when a verdict-bearing stop
    /// was REFUSED (the outcome is then `Unknown`); `None` when the lane did
    /// not reach a verdict-bearing stop.
    pub ledger_ok: Option<bool>,
    /// Frozen parallel-frontier policy plus observed tree decisions.
    pub frontier_execution: FrontierExecutionEvidence,
    /// Governed margin-row policy execution observed by this invocation.
    pub policy_execution: PolicyExecutionEvidence,
    /// Frozen epoch policy and causal Tier-0/Tier-2 execution.
    pub epoch_execution: EpochExecutionEvidence,
    /// Frozen width policy and causal ordinary-selector execution.
    pub width_execution: WidthExecutionEvidence,
    /// Governed root-precision execution.
    pub root_precision_execution: super::RootPrecisionExecutionEvidence,
    /// Governed backward-intermediate root-build execution.
    pub backward_interm_execution: super::BackwardIntermExecutionEvidence,
    /// Governed standalone-alpha and rebake root-phase execution.
    pub root_alpha_execution: super::RootAlphaExecutionEvidence,
    /// #mr-close-telemetry: the certified value of every close this
    /// invocation's proof rests on, by kind (root `dj`, tree bound closes by
    /// eval stage, child closes, emptiness slacks, retirements, closed epoch
    /// subtrees). Recorded beside every Kraft `ledger.leaf` in the serial
    /// merge. Telemetry only — never read by any bound, gate or verdict.
    pub close_integrity: CloseIntegrityEvidence,
}

/// The frame's cover ledger (#soundness-spine M2c). It replaces the
/// depth-only Kraft ledger of #epoch-bab Phase E, which could not see paths
/// (a duplicated leaf plus a missing sibling, or siblings that split different
/// neurons, both summed to 1) and never gated (`Unsat` was returned whatever
/// it said).
///
/// Every closed leaf is recorded with its PATH below the frame root and its
/// certified evidence; a verdict-bearing stop then goes through
/// [`authorize_frame`], which refuses unless the paths cover the root
/// ([`CoverLedger::prove`]) and every required class is covered
/// ([`authorize_unsat`]). Identical in both proof profiles.
#[derive(Default)]
struct Ledger {
    cover: CoverLedger,
    /// Who may consume this frame's token. `Default` is `Subdomain`, which can
    /// never be published: a driver that forgets to scope itself fails closed.
    scope: TokenScope,
    /// Classes the frame's token must cover (`None` = the tree's own classes).
    required: Option<Vec<usize>>,
    /// Classes discharged before the tree (root closes, frozen-dual tier).
    pre: Vec<ClassCover>,
    /// Original-frame authority for a direct-prepass contribution. Kept
    /// through tree work and checked after the common authorizer's own work.
    direct_guard: Option<super::head_group::direct::DirectPublicationGuard>,
    /// B2.3 x head routes (eng12b): the root-frame classes whose certified
    /// root bound a direct-head or head-group computation RAISED. Hint data
    /// only (read by nothing but the stream, after the token exists): such a
    /// `Root` cover is streamed as the ordinary root evaluation's own claim
    /// when that evaluation closed the class too, else as an opaque claim of
    /// the head route — never as a `close: root` node carrying a margin the
    /// checker's root rung does not compute.
    head_raised: Vec<HeadRaised>,
    /// A nested epoch subtree's token, parked by `try_epoch` for the leaf its
    /// caller records next.
    pending_subtree: Option<UnsatToken>,
    /// ENGINE_PROGRAM §1.7 B1 (proof bundle v2): the frame's WRITE-ONLY hint
    /// ledger. Captures only when the entry point's Proof arm switched hint
    /// capture on; nothing a hint says reaches a bound, a gate or a verdict.
    hints: HintLedger,
    /// The absolute literals of this frame's root (a nested frame's parent
    /// leaf), kept to translate a CHILD frame's prefix. Hint data only.
    hint_prefix: Vec<AbsLit>,
    /// B2.3: the frozen-dual tier's prune traces of the classes it closed
    /// (Proof profile only; empty under Competition). Hint data only: read
    /// by nothing but the stream, after the token exists.
    fd_proofs: Vec<FdProofTrace>,
    /// B2.3: the tree classes whose EVERY certified leaf went into this
    /// frame's hint ledger from a route with its own per-class ledgers (the
    /// wave engine): their `Tree` cover streams those leaves instead of an
    /// opaque claim. Hint data only.
    streamed_tree_classes: Vec<usize>,
}

/// What a leaf hint needs beyond the cover path (proof bundle v2): the
/// domain's split order and the β its closing bound was scored under, the
/// retired columns in force, whether the bound came from an ancestor region,
/// and whether the child split is the site found empty. Hint data only.
#[derive(Clone, Copy)]
struct HintCtx<'x> {
    order: u128,
    betas: &'x [f64],
    head_betas: &'x [f64],
    retired: &'x [u16],
    lift: bool,
    empty_child: bool,
}

impl<'x> HintCtx<'x> {
    fn of(entry: &'x DomainEntry, lift: bool) -> Self {
        Self {
            order: entry.split_order,
            betas: &entry.betas,
            head_betas: &entry.head_betas,
            retired: &entry.retired,
            lift,
            empty_child: false,
        }
    }
}

/// A frame's splits below its root in BRANCH order, with β aligned, from the
/// lane's own domain record: `order` bit `k` set means the `k`-th split of
/// the frame was a head split (see [`DomainEntry::split_order`]); a record
/// the mask cannot describe falls back to trunk-then-head order (which only
/// costs the checker's ancestor retries). The child split (fresh, β 0) goes
/// last. `None` when a split carries no side.
fn hint_splits(
    trunk: &[TrunkSplit],
    heads: &[HeadFix],
    root_heads: usize,
    order: u128,
    betas: &[f64],
    head_betas: &[f64],
    child: Option<ChildSplit>,
) -> Option<(Vec<FrameSplit>, Vec<f64>)> {
    let below = heads.get(root_heads..)?;
    let tsplit = |&(layer, pos, dir): &TrunkSplit| -> Option<FrameSplit> {
        if dir == 0 {
            return None;
        }
        Some(FrameSplit::Trunk {
            layer: u32::try_from(layer).ok()?,
            pos: u32::try_from(pos).ok()?,
            positive: dir > 0,
        })
    };
    let hsplit = |&(index, dir): &HeadFix| -> Option<FrameSplit> {
        if dir == 0 {
            return None;
        }
        Some(FrameSplit::Head {
            index: u32::try_from(index).ok()?,
            positive: dir > 0,
        })
    };
    let depth = trunk.len() + below.len();
    let described = depth <= 128
        && order.count_ones() as usize == below.len()
        && (depth == 128 || order >> depth == 0);
    let mut splits = Vec::with_capacity(depth + 1);
    let mut beta = Vec::with_capacity(depth + 1);
    let (mut ti, mut hi) = (0usize, 0usize);
    for k in 0..depth {
        let head = if described {
            order >> k & 1 == 1
        } else {
            k >= trunk.len()
        };
        if head {
            splits.push(hsplit(&below[hi])?);
            beta.push(head_betas.get(root_heads + hi).copied().unwrap_or(0.0));
            hi += 1;
        } else {
            splits.push(tsplit(&trunk[ti])?);
            beta.push(betas.get(ti).copied().unwrap_or(0.0));
            ti += 1;
        }
    }
    match child {
        Some(ChildSplit::Trunk(t)) => {
            splits.push(tsplit(&t)?);
            beta.push(0.0);
        }
        Some(ChildSplit::Head(h)) => {
            splits.push(hsplit(&h)?);
            beta.push(0.0);
        }
        None => {}
    }
    Some((splits, beta))
}

/// The split-order mask of a child of `entry` (see
/// [`DomainEntry::split_order`]).
fn child_split_order(entry: &DomainEntry, root_heads: usize, head: bool) -> u128 {
    let depth = entry.trunk.len() + entry.heads.len().saturating_sub(root_heads);
    if head && depth < 128 {
        entry.split_order | (1u128 << depth)
    } else {
        entry.split_order
    }
}

#[cfg(test)]
thread_local! {
    /// Test-only mutation hook: the next leaf this thread records is DROPPED.
    static COVER_FAULT_DROP_NEXT_LEAF: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Test-only: drop the next cover leaf recorded on this thread (the mutation
/// pin that proves the gate can SEE an incomplete tree).
#[cfg(test)]
pub(crate) fn inject_cover_fault_drop_next_leaf() {
    COVER_FAULT_DROP_NEXT_LEAF.with(|c| c.set(true));
}

#[cfg(test)]
fn cover_fault_drop_next_leaf() -> bool {
    COVER_FAULT_DROP_NEXT_LEAF.with(|c| c.replace(false))
}

#[cfg(not(test))]
#[inline(always)]
fn cover_fault_drop_next_leaf() -> bool {
    false
}

/// #soundness-spine M2c: a child closed at its parent's expansion — the split
/// that created it and its certified evidence — for the cover ledger.
struct ClosedLeaf {
    split: ChildSplit,
    evidence: Option<LeafEvidence>,
    /// Proof bundle v2: the closing bound came from the PARENT's region
    /// (a parent-lift or the parent's bound), not the child's own score.
    lift: bool,
}

/// Proof bundle v2: the hint context of one evaluated domain, carried from a
/// rayon worker to the serial merge (built only while the frame captures;
/// never read by any bound, gate or verdict).
struct EvalHints {
    /// The accepted β of this evaluation (aligned with the entry's trunk).
    betas: Vec<f64>,
    /// The accepted head β (aligned with the entry's heads).
    head_betas: Vec<f64>,
    /// The retirement set this evaluation passes to its children.
    retired_out: Vec<u16>,
    /// The closing bound came from the entry's inherited bound.
    lift_eval: bool,
}

/// The token scope of a driver frame: only the OUTER frame speaks for the
/// instance; a nested epoch frame proves a sub-domain.
fn frame_scope(cfg: &BabConfig) -> TokenScope {
    if cfg.epoch_frame_route == EpochFrameRoute::Outer {
        TokenScope::Instance
    } else {
        TokenScope::Subdomain
    }
}

/// The frozen-dual tier's closed classes as certified class covers, each
/// re-derived from its own search report
/// (`certify::margin_row::frozen_dual_class`) rather than read from the
/// tier's `closed` list.
fn frozen_dual_covers(tier: &super::zono_bab::TierReport) -> Vec<ClassCover> {
    tier.classes
        .iter()
        .filter(|report| tier.closed.contains(&report.class))
        .filter_map(cert_mr::frozen_dual_class)
        .collect()
}

/// Whether `covers` holds a frozen-dual cover for `class`.
fn fd_covered(covers: &[ClassCover], class: usize) -> bool {
    covers
        .iter()
        .any(|c| matches!(c, ClassCover::FrozenDual { class: k, .. } if *k == class))
}

/// Fraction of the remaining search wall the G2 containment guard may spend
/// when an instance-scoped proof is authorized. Downgrade-only work after the
/// proof is complete: a guard the budget cannot pay for is SKIPPED (recorded),
/// never allowed to delay publication.
const G2_BUDGET_FRACTION: f64 = 0.02;

/// G4 (#soundness-spine M2d, ENGINE_PROGRAM §1.7 B4): the frozen-dual tier's
/// block-path cross-check rate, taken from the audit rates of the proof profile
/// the ENTRY POINT froze — `Proof` re-checks every prune against the scalar
/// evaluator, `Competition` one in 256 (plus every prune at depth `<= 4` in
/// both). Read here, in the lane, and handed to the tier as a plain `u32`, so
/// the tier (`zono_bab`) never names the profile. Downgrade-only: a mismatch
/// refuses the search, never closes a class.
fn fd_crosscheck_every() -> u32 {
    let rates = ny_core::proof_profile::audit_rates();
    rates.crosscheck_every
}

/// G2 on the outer root frame (#soundness-spine M2d): the root's certified
/// boxes, root bounds and the verdict's margin claim against the lane's own
/// f64 point evaluator. `adv` must be aligned with `re.dj`.
/// Only authenticated producers construct this private view. It carries the
/// actual producer's head box, without requiring identity affine rows.
struct OwnedRootFrameView<'a> {
    dj: &'a [f64],
    ybox: YBox,
}

fn outer_frame_guards(
    net: &TwinNet,
    root: &RootGates,
    t: usize,
    adv: &[usize],
    frame: OwnedRootFrameView<'_>,
    cfg: &BabConfig,
) -> (GuardReport, String) {
    if frame.dj.len() != adv.len() {
        return (GuardReport::new(), "g2=skipped(misaligned)".to_string());
    }
    let rates = ny_core::proof_profile::audit_rates();
    let budget = cfg.deadline.map(|dl| {
        dl.saturating_duration_since(Instant::now())
            .mul_f64(G2_BUDGET_FRACTION)
    });
    let frame = super::containment_guard::RootFrame::new(net, root, t, adv, frame.dj, frame.ybox);
    let out = crate::guards::containment::run_containment(&frame, rates.containment_points, budget);
    (out.report(), out.fragment())
}

/// U5-U7 for one driver frame: prove the tree's cover (when a tree ran), add
/// the pre-tree class covers, run the frame's guards, and authorize a token of
/// the frame's scope — or refuse with a reason. Prints the G5 line
/// (`[cover-gate] authorized ...`) for an instance-scoped proof and a
/// `[cover-gate] REFUSED ...` line for every refusal.
#[allow(clippy::too_many_arguments)]
fn authorize_frame(
    ledger: &mut Ledger,
    tree_classes: &[usize],
    tree_ran: bool,
    net: &TwinNet,
    root: &RootGates,
    t: usize,
    re: &RootEval,
    cfg: &BabConfig,
) -> std::result::Result<UnsatToken, String> {
    authorize_frame_view(
        ledger,
        tree_classes,
        tree_ran,
        net,
        root,
        t,
        OwnedRootFrameView {
            dj: &re.dj,
            ybox: re.root_ybox(),
        },
        cfg,
    )
}

#[allow(clippy::too_many_arguments)]
fn authorize_frame_view(
    ledger: &mut Ledger,
    tree_classes: &[usize],
    tree_ran: bool,
    net: &TwinNet,
    root: &RootGates,
    t: usize,
    frame: OwnedRootFrameView<'_>,
    cfg: &BabConfig,
) -> std::result::Result<UnsatToken, String> {
    let scope = ledger.scope;
    let mut hist = crate::guards::margin_hist::MarginHistogram::new();
    let mut covers = std::mem::take(&mut ledger.pre);
    // ENGINE_PROGRAM §1.7 B1 (proof bundle v2): the pre-tree class
    // discharges as hints, so every class the token relies on is CLAIMED in
    // the stream (a class covered by a route that streams nothing is claimed
    // opaque). Hint data only; built only while capturing.
    let pre_hints: Vec<PreHint> = if ledger.hints.capturing() {
        let mut fd_proofs = std::mem::take(&mut ledger.fd_proofs);
        covers
            .iter()
            .filter_map(|cover| {
                head_route_pre_hint(cover, &ledger.head_raised).or_else(|| {
                    proof_pre_hint(
                        cover,
                        &mut fd_proofs,
                        &ledger.streamed_tree_classes,
                        ledger.hint_prefix.is_empty(),
                    )
                })
            })
            .collect()
    } else {
        Vec::new()
    };
    let mut tree_note = "tree=none".to_string();
    if tree_ran {
        let cover = match ledger.cover.prove(tree_classes) {
            Ok(cover) => cover,
            Err(refusal) => {
                let reason = format!("cover gate refused: {refusal}");
                eprintln!(
                    "[cover-gate] REFUSED scope={} leaves={} reason={refusal}",
                    scope.token(),
                    ledger.cover.recorded()
                );
                return Err(reason);
            }
        };
        for m in ledger.cover.bound_margins() {
            hist.note(m);
        }
        tree_note = format!(
            "tree_leaves={} tree_max_depth={} empty_leaves={} subtree_leaves={}",
            cover.leaves(),
            cover.max_depth(),
            cover.empty_leaves(),
            cover.subtree_leaves()
        );
        covers.push(ClassCover::Tree(cover));
    }
    for cover in &covers {
        match cover {
            ClassCover::Root { bound, .. } => hist.note(bound.value()),
            ClassCover::FrozenDual { cert, .. } => hist.note(cert.value()),
            ClassCover::Part(token) => hist.note(token.min_margin()),
            // Specification-emptiness covers carry no bound; the margin-row
            // lane never builds one.
            ClassCover::Tree(_) | ClassCover::Exact { .. } => {}
        }
    }
    let required: Vec<usize> = ledger
        .required
        .clone()
        .unwrap_or_else(|| tree_classes.to_vec());
    let (guards, g2) = if scope == TokenScope::Instance {
        outer_frame_guards(net, root, t, &required, frame, cfg)
    } else {
        (GuardReport::new(), "g2=skipped(frame)".to_string())
    };
    match authorize_unsat(scope, &required, &covers, &guards) {
        Ok(token) => {
            if scope == TokenScope::Instance && ledger.direct_guard.is_none() {
                eprintln!(
                    "[cover-gate] authorized scope={} classes={} {tree_note} {} {g2}",
                    scope.token(),
                    required.len(),
                    hist.fragment()
                );
            }
            // ENGINE_PROGRAM §1.7 B1: the PROVED frame, as hints, into the
            // process sink the entry point wired (a v2 bundle under Proof,
            // `NullSink` under Competition, nothing in unit tests) — for EVERY
            // authorized scope (instance, class part, nested subdomain), so a
            // nested frame's nodes cover its parent's subtree leaf. Write-only
            // and after the token exists: the token above is already minted,
            // so nothing a sink does can reach it.
            let capture = ledger.hints.capturing();
            if capture || (scope == TokenScope::Instance && tree_ran) {
                let mut hints = std::mem::take(&mut ledger.hints);
                for h in pre_hints {
                    hints.push_pre(h);
                }
                let sites = if capture {
                    SiteMap::from_root(root, net)
                } else {
                    SiteMap::default()
                };
                crate::certify::stream_frame(&ledger.cover, hints, tree_classes, &sites);
            }
            #[cfg(test)]
            if let Some(guard) = ledger.direct_guard.as_mut() {
                DIRECT_EXPIRE_AFTER_AUTHORIZER.with(|fault| {
                    if fault.replace(false) {
                        guard.expire_for_test();
                    }
                });
            }
            if ledger
                .direct_guard
                .as_ref()
                .is_some_and(|guard| !guard.valid(&BackwardEngine::new(net, root), t, &required))
            {
                eprintln!(
                    "[cover-gate] REFUSED scope={} classes={} reason=direct_source_or_deadline_after_authorizer",
                    scope.token(),
                    required.len()
                );
                return Err("direct head source/deadline/FP at publication".into());
            }
            if scope == TokenScope::Instance && ledger.direct_guard.is_some() {
                eprintln!(
                    "[cover-gate] authorized scope={} classes={} {tree_note} {} {g2}",
                    scope.token(),
                    required.len(),
                    hist.fragment()
                );
            }
            Ok(token)
        }
        Err(refusal) => {
            eprintln!(
                "[cover-gate] REFUSED scope={} classes={} reason={refusal} {g2}",
                scope.token(),
                required.len()
            );
            Err(format!("unsat refused: {refusal}"))
        }
    }
}

#[cfg(test)]
thread_local! {
    /// A deterministic deadline fault after the real containment/stream work.
    static DIRECT_EXPIRE_AFTER_AUTHORIZER: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// B2.3 x head routes (eng12b): one root-frame class whose certified root
/// bound a head route raised (see [`Ledger::head_raised`]).
#[derive(Clone, Copy, Debug, PartialEq)]
struct HeadRaised {
    class: usize,
    /// `"direct_head"` (the direct all-head prepass) or `"head_group"` (the
    /// source-owned head-group refinement of the root evaluation).
    route: &'static str,
    /// The ordinary root evaluation's own bound for the class, when THAT
    /// bound already closed it (`None`: the root evaluation alone did not
    /// close it, or never ran because the direct prepass closed the frame).
    root_margin: Option<f64>,
}

/// B2.3 x head routes (eng12b): which classes the head routes raised, from
/// the root evaluation's bounds (`root`), the bounds after the direct-prelude
/// merge (`merged`) and after the head-group refinement (`fin`), all aligned
/// with `adv`. Dark (no head route armed) the three are identical and the
/// result is empty. A non-finite or misaligned entry is never "raised".
fn head_raised_classes(
    mode: RoundMode,
    adv: &[usize],
    root: &[f64],
    merged: &[f64],
    fin: &[f64],
) -> Vec<HeadRaised> {
    adv.iter()
        .enumerate()
        .filter_map(|(i, &class)| {
            let (r, m, f) = (*root.get(i)?, *merged.get(i)?, *fin.get(i)?);
            (f > r).then(|| HeadRaised {
                class,
                route: if f > m { "head_group" } else { "direct_head" },
                root_margin: closed(mode, r).then_some(r),
            })
        })
        .collect()
}

/// B2.3 x head routes (eng12b): the hint of a `Root` cover a head route
/// raised (`None`: not one, so [`proof_pre_hint`] decides). A class the root
/// evaluation closed on its own keeps that evaluation's checkable claim with
/// ITS margin; any other is the head route's opaque claim — its phase-cover
/// multipliers stream nothing the checker can replay.
fn head_route_pre_hint(cover: &ClassCover, raised: &[HeadRaised]) -> Option<PreHint> {
    let ClassCover::Root { class, .. } = cover else {
        return None;
    };
    let head = raised.iter().find(|h| h.class == *class)?;
    Some(match head.root_margin {
        Some(margin) => PreHint::Root {
            class: *class,
            margin,
        },
        None => PreHint::Opaque {
            classes: vec![*class],
            route: head.route,
        },
    })
}

/// B2.3: a pre-tree class discharge as a proof hint, preferring a CHECKABLE
/// stream over a claim: a frozen-dual close with a complete prune trace
/// becomes its prunes (on a root frame: the tier runs only on the outer
/// frame, whose prefix is empty), and a tree cover whose every class streamed
/// its leaves into the frame's hint ledger (the wave engine) claims nothing
/// more. Everything else is [`pre_hint`]'s claim.
fn proof_pre_hint(
    cover: &ClassCover,
    fd_proofs: &mut Vec<FdProofTrace>,
    streamed_tree_classes: &[usize],
    root_frame: bool,
) -> Option<PreHint> {
    match cover {
        ClassCover::FrozenDual { class, .. } if root_frame => {
            if let Some(at) = fd_proofs
                .iter()
                .position(|p| p.class() == *class && p.complete())
            {
                return Some(PreHint::FrozenDualProof(Box::new(
                    fd_proofs.swap_remove(at),
                )));
            }
            pre_hint(cover)
        }
        ClassCover::Tree(tree)
            if !tree.classes().is_empty()
                && tree
                    .classes()
                    .iter()
                    .all(|c| streamed_tree_classes.contains(c)) =>
        {
            None
        }
        _ => pre_hint(cover),
    }
}

/// B2.3: the frozen-dual tier's prune traces of the classes `covers` closes,
/// MOVED out of the tier report. Empty under Competition (no trace is ever
/// built there).
fn take_fd_proofs(
    tier: &mut super::zono_bab::TierReport,
    covers: &[ClassCover],
) -> Vec<FdProofTrace> {
    let mut out = Vec::new();
    for report in &mut tier.classes {
        if !fd_covered(covers, report.class) {
            continue;
        }
        if let Ok(run) = report.result.as_mut() {
            if let Some(proof) = run.proof.take() {
                out.push(*proof);
            }
        }
    }
    out
}

/// A pre-tree class discharge as a proof hint (proof bundle v2).
fn pre_hint(cover: &ClassCover) -> Option<PreHint> {
    match cover {
        ClassCover::Root { class, bound } => Some(PreHint::Root {
            class: *class,
            margin: bound.value(),
        }),
        ClassCover::FrozenDual { class, cert } => Some(PreHint::FrozenDual {
            class: *class,
            margin: cert.value(),
        }),
        // A tree cover in `pre` is a route that proved its classes without
        // streaming leaves (the wave engine): claimed, never checkable.
        ClassCover::Tree(tree) => Some(PreHint::Opaque {
            classes: tree.classes().to_vec(),
            route: "wave",
        }),
        // A class part streamed its own frame when it was authorized.
        ClassCover::Part(_) => None,
        ClassCover::Exact { class, .. } => Some(PreHint::Opaque {
            classes: vec![*class],
            route: "exact",
        }),
    }
}

/// Outcome of the lane. NEVER `Sat`: fail-closed to `Unknown`.
pub enum MarginRowOutcome {
    /// Every class certified: margin > 0 on the whole box. The token is the
    /// frame's authorization (#soundness-spine M2b/M2c): minted only after
    /// the path-based cover gate and the class aggregation passed, scoped to
    /// the instance for an outer frame and to the sub-domain for a nested one.
    Unsat(BabStats, UnsatToken),
    /// Not decided (budget, candidates exhausted, or fail-closed error).
    Unknown {
        /// Why.
        reason: String,
        /// Stats if the tree ran.
        stats: Option<BabStats>,
    },
}

pub(super) struct YPack {
    pub(super) al: PassOut,
    pub(super) au: PassOut,
    pub(super) ly0: Vec<f64>,
    pub(super) uy0: Vec<f64>,
    /// Minted before the owned unconditional source passes, never by the raw
    /// row adapter. This is not an importable certificate or verdict token.
    head_group_source: Option<super::head_group_source::YPackSource>,
    /// #clip-head: the UNTIGHTENED concretization of `al`, kept verbatim as
    /// `pack_from_rows` produced it.
    ///
    /// `ly0` is NOT a safe calibration base: `clip_tighten_pack` adds trunk
    /// deltas into it in place whenever `margin_row.clip` is armed, and a slack
    /// recovered against a TIGHTENED bound under-recovers (clamps toward zero),
    /// which makes the halfspace claim the line bounds `y_i` more tightly than
    /// certified — a false-`unsat` generator. This is the ordering trap
    /// `root.rs`'s CALIBRATION INVARIANT records for the trunk rows.
    ly_fwd: Vec<f64>,
    /// #clip-head: the untightened concretization of `au`; see [`YPack::ly_fwd`].
    uy_fwd: Vec<f64>,
    /// #clip-head: certified slack recovered from the head lines, or `None`
    /// when the gate is dark or the per-pack frame check refused.
    head_slack: Option<super::clip::HeadSlack>,
    al_dots: RowDots,
    au_dots: RowDots,
    // Last: actual numerical buffers cease before their original owner claims.
    prepared_owner: Option<prepared_root_eval::PreparedPackOwner>,
}

impl YPack {
    pub(super) fn owned_for_head_groups(&self, eng: &BackwardEngine<'_>) -> bool {
        self.head_group_source
            .as_ref()
            .is_some_and(|source| source.matches(eng, &self.al, &self.au, &self.ly0, &self.uy0))
    }

    /// Freeze existing private provenance for an immediate prepared replay.
    /// Cloning this sealed value creates no authority for another root frame.
    pub(super) fn head_source_snapshot(
        &self,
        eng: &BackwardEngine<'_>,
    ) -> Option<super::head_group_source::YPackSource> {
        if !self.owned_for_head_groups(eng) {
            return None;
        }
        Some(self.head_group_source.as_ref()?.clone())
    }
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
    /// #margin-row-domain-alpha: the ancestor's ACCEPTED per-domain alpha, as a
    /// sparse `(layer, neuron, alpha)` list sorted by `(layer, neuron)` — the
    /// warm start alpha-beta-CROWN gets from `domain_updater.py:123-128`.
    ///
    /// Valid on this region for the same reason it was valid on the parent's:
    /// `alpha*y <= relu(y)` holds pointwise on all of R for any
    /// `alpha in [0, 1]`, so lower-line validity is BOX-INDEPENDENT. Entries
    /// naming a neuron this domain has since split or clip-stabilized are
    /// dropped rather than applied (`domain_alpha::apply_inherited` writes only
    /// through the movable-slot list). Always empty unless
    /// `NY_MARGIN_ROW_DOMAIN_ALPHA` is armed.
    alpha: Vec<super::domain_alpha::AlphaCarry>,
    /// #clip-stateful: the ACCUMULATED Clip-and-Verify intermediate tightening
    /// this domain inherits from its parent's evaluation — a sparse
    /// `(layer, neuron, lo, hi)` delta, not a dense interval copy, because the
    /// frontier reaches 400+ live entries.
    ///
    /// Same monotone parent -> child contract as `retired` and `betas`: every
    /// stored interval was proved over an ANCESTOR region, and this domain's
    /// region is a subset, so it stays a valid enclosure verbatim. The child's
    /// own evaluation intersects its fresh tightening into it shrink-only.
    /// `None` unless `margin_row.clip_stateful` (or `NY_MARGIN_ROW_CLIP_STATEFUL`)
    /// is armed, in which case the whole lane is byte-identical to its history.
    clip_acc: Option<Arc<super::clip::ClipAcc>>,
    /// Proof bundle v2 (ENGINE_PROGRAM §1.7 B1): the BRANCH ORDER of this
    /// domain's splits below its frame root — bit `k` set iff the frame's
    /// `k`-th split was a head split (trunk and head splits are kept in two
    /// vectors, which lose their interleaving). `Copy`, never read by any
    /// bound, gate or verdict: only [`hint_splits`] reads it, to order a
    /// leaf hint's literals.
    split_order: u128,
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
    /// Invocation-local execution counters produced by this pure evaluation.
    /// The parallel driver absorbs them only in deterministic pop order.
    policy_execution: PolicyExecutionEvidence,
    /// Invocation-local Tier-0 evidence, merged beside policy evidence.
    epoch_execution: EpochExecutionEvidence,
    /// Invocation-local ordinary width-selector evidence.
    width_execution: WidthExecutionEvidence,
    /// Selection-time facts held until the exact score pass completes.
    epoch_selection: EpochSelectionObservation,
    /// Ordinary width-selection identity facts held until exact scoring
    /// completes. Tier-0 leaves this at its all-zero default.
    width_selection: WidthSelectionObservation,
    /// #margin-row-domain-alpha: the per-domain alpha this node's ACCEPTED
    /// bound was scored under, sparse and root-relative (see
    /// `DomainEntry::alpha`). Children inherit it verbatim. Equals the entry's
    /// list when disarmed or when no trial improved.
    alpha: Vec<super::domain_alpha::AlphaCarry>,
    /// #clip-stateful: the OUT set for children — the entry's inherited
    /// accumulation intersected with what THIS eval's `domain_gates_acc`
    /// derived. `None` when the gate is dark.
    clip_acc: Option<Arc<super::clip::ClipAcc>>,
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
    /// #mr-close-telemetry: the eval STAGE whose bound first closed (`Eval`
    /// = the main pass, `Beta`, `DomainAlpha`). Meaningful only when the
    /// returned bound closes. Telemetry only.
    close_src: CertKind,
    /// #mr-close-telemetry: the certified composite of every column THIS eval
    /// newly retired (#margin-row-col-retire). Telemetry only.
    retire_certs: CertTally,
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
    /// #margin-row-domain-alpha: inherited per-domain alpha (see
    /// `DomainEntry::alpha`). Carried verbatim; the child re-filters it against
    /// its OWN movable slots at its own eval.
    alpha: Vec<super::domain_alpha::AlphaCarry>,
    /// #clip-stateful: inherited accumulation (see `DomainEntry::clip_acc`).
    clip_acc: Option<Arc<super::clip::ClipAcc>>,
    /// Proof bundle v2: see `DomainEntry::split_order`.
    split_order: u128,
}

/// #mr-stack (throughput plan B5): one domain evaluation paused at its main
/// certified pass — every local the unsplit `eval_with_pack` body computed
/// before the pass and read after it. Produced by
/// [`MarginRowBab::eval_prepare`], consumed by [`MarginRowBab::eval_finish`];
/// the pass in between is the domain's own ([`MarginRowBab::eval_main_pass`])
/// or its block of one domain-stacked pass (`bab/stack_eval.rs`).
struct EvalSeeded {
    /// `Phase::EvalNode` timer (profile-only), dropped at the end of the finish.
    t_node: Option<super::prof::Timer>,
    policy_execution: PolicyExecutionEvidence,
    epoch_execution: EpochExecutionEvidence,
    width_execution: WidthExecutionEvidence,
    clip_acc: Option<Arc<super::clip::ClipAcc>>,
    pack: Arc<YPack>,
    ybox: YBox,
    gates: HeadGates,
    mb_local: Option<MarginBatch>,
    survivors: Option<Vec<usize>>,
    ms: MarginSeed,
    beta_armed: bool,
    betas: Vec<f64>,
    head_betas: Vec<f64>,
    beta_step: f64,
    percol: bool,
    pc: Vec<PcBetas>,
    head_admissible: bool,
    /// The gates the main pass runs under (inherited β / per-column β /
    /// warm-started domain alpha already installed).
    dom: DomainGates,
    alpha_policy: super::DomainAlphaRuntimePolicy,
    alpha_armed: bool,
    alpha_slots: Vec<super::domain_alpha::Slot>,
    alpha_cur: Vec<f64>,
    alpha_warm: usize,
    seed_main: Option<Seed>,
    seed_main_pc: Option<Seed>,
    /// `max(|ly|, |uy|)` of the final y-box (the GPU seam's `y_abs`).
    eval_y_abs: Vec<f64>,
}

impl EvalSeeded {
    /// The seed of the main pass: the per-column head-shifted seed, else the
    /// shared head-shifted seed, else the base margin seed.
    fn main_seed(&self) -> &Seed {
        self.seed_main_pc
            .as_ref()
            .or(self.seed_main.as_ref())
            .unwrap_or(&self.ms.seed)
    }

    /// Whether this eval's passes capture `unst_rows` (the β / domain-alpha
    /// gradient input).
    fn unst_rows(&self) -> bool {
        self.beta_armed || self.alpha_armed
    }
}

/// [`MarginRowBab::eval_prepare`]'s result.
enum EvalPrep {
    /// An early exit, returned exactly as the unsplit body returned it
    /// (boxed like `PreparedExpand::Done`: the evidence record dwarfs a box).
    Done(Box<EvalOut>),
    /// Ready for its main pass (boxed: the payload is large).
    Seeded(Box<EvalSeeded>),
}

/// Result of one domain evaluation (see [`MarginRowBab::eval_with_pack`]).
enum EvalOut {
    /// Provably empty region: the domain is closed. The `f64` is the
    /// emptiness certificate's slack (#mr-close-telemetry; telemetry only).
    Infeasible(PolicyExecutionEvidence, f64),
    /// #deadline-poll (D2): an intra-expansion poll fired before the next
    /// costly phase. The caller aborts CLEANLY with no verdict — never a
    /// closure, never an error — exactly as the between-expansion check
    /// would have at the next loop head, just up to one engine pass sooner.
    Deadline(PolicyExecutionEvidence),
    /// Refreshed bound + branching state (boxed: `NodeState` dwarfs the other
    /// variants).
    Node(f64, Box<NodeState>),
}

/// Result of evaluating+expanding ONE popped domain in the parallel frontier.
/// Carries only data (no `&mut` state), so it is produced by a rayon worker
/// and applied deterministically by the serial merge, keeping every stats
/// delta and heap push bit-identical to what the serial lane would do for the
/// same domain.
#[derive(Clone, Copy)]
struct DomainExecutionEvidence {
    policy: PolicyExecutionEvidence,
    epoch: EpochExecutionEvidence,
    width: WidthExecutionEvidence,
}

impl DomainExecutionEvidence {
    fn before_selection(policy: PolicyExecutionEvidence, config: &BabConfig) -> Self {
        Self {
            policy,
            epoch: EpochExecutionEvidence::from_config(config),
            width: WidthExecutionEvidence::from_config(config),
        }
    }

    fn from_state(state: &NodeState) -> Self {
        Self {
            policy: state.policy_execution,
            epoch: state.epoch_execution,
            width: state.width_execution,
        }
    }
}

enum ExpandStep {
    /// `eval_with_pack` returned `None` (provably empty box): closed. The
    /// [`Cert`] carries the emptiness slack (#mr-close-telemetry).
    Infeasible(DomainExecutionEvidence, Cert),
    /// #deadline-poll (D2): an intra-expansion poll fired inside this
    /// domain's eval or candidate scoring. The merge stops the tree loop with
    /// a wallclock stop reason; the domain is NOT closed and NOT counted.
    DeadlineHit(DomainExecutionEvidence),
    /// The refreshed bound already closes the domain. The [`Cert`] carries
    /// the closing eval stage and bound (#mr-close-telemetry); the hints are
    /// proof bundle v2 data (`None` unless the frame captures).
    ClosedAtEval(DomainExecutionEvidence, Cert, Option<Box<EvalHints>>),
    /// No split candidates remain: the instance cannot close here (Unknown).
    NoCandidates(DomainExecutionEvidence),
    /// A verdict-bearing bound was non-finite: fail-closed to Unknown.
    NonFiniteBound(DomainExecutionEvidence),
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
        /// #mr-census: what closed the `closed_children` (telemetry only).
        close_kinds: ChildCloseKinds,
        /// #soundness-spine M2c: every closed child's split and certified
        /// evidence, in child order — the cover ledger's leaves. One entry per
        /// `closed_children`; the merge refuses the cover if they disagree.
        closed_leaves: Vec<ClosedLeaf>,
        /// Pure per-evaluation policy counters for deterministic serial merge.
        /// Boxed: with it inline `Expanded` was 1016 bytes against
        /// `ClosedAtEval`'s 848 (clippy::large_enum_variant, threshold 64).
        execution: Box<DomainExecutionEvidence>,
        /// Proof bundle v2 hint context (`None` unless the frame captures).
        hints: Option<Box<EvalHints>>,
    },
}

/// #mr-census: how an expansion's closed children closed. Telemetry only —
/// counted beside `closed_children`, never read by a bound or a verdict.
#[derive(Clone, Copy, Default)]
struct ChildCloseKinds {
    /// Child's own scored bound closes (`closed(raw)`).
    raw: usize,
    /// Closed only through `max(raw, b, lift)` with `raw` open: the parent
    /// lift (the parent bound `b` was already checked open at eval).
    lift: usize,
    /// Head-split child slot whose bounds crossed (provably empty).
    head_infeasible: usize,
    /// #mr-close-telemetry: each closed child's certificate, in child order
    /// (an expansion has at most two children). Not folded into the
    /// trajectory hash: telemetry only.
    certs: [Option<Cert>; 2],
    /// #mr-close-telemetry: the columns the PARENT retired while staying
    /// open (see `close_integrity`'s `retire`). Telemetry only.
    retire: CertTally,
}

impl ChildCloseKinds {
    fn note_scored(&mut self, raw_closes: bool) {
        if raw_closes {
            self.raw += 1;
        } else {
            self.lift += 1;
        }
    }

    /// #mr-close-telemetry: record one closed child's certificate.
    fn note_cert(&mut self, kind: CertKind, value: f64) {
        if let Some(slot) = self.certs.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some(Cert { kind, value });
        }
    }
}

/// #mr-close-telemetry: the certified slack of a crossed y-box — the largest
/// `ly_i - uy_i` (positive exactly when some interval is empty). The emptiness
/// DECISION stays `YBox::is_empty`, but since #soundness-spine M2c this slack
/// is also the emptiness leaf's cover-gate EVIDENCE (`certify::margin_row::
/// empty_evidence` refuses a slack that is not finite and `> 0`), so it must be
/// computed on every emptiness exit — a telemetry-off `NaN` here would make
/// the cover gate refuse an otherwise complete tree.
fn ybox_crossing(ybox: &YBox) -> f64 {
    ybox.ly
        .iter()
        .zip(&ybox.uy)
        .map(|(l, u)| l - u)
        .fold(f64::NEG_INFINITY, f64::max)
}

#[cfg(test)]
mod close_telemetry_unit_tests {
    use super::{ybox_crossing, CertKind, ChildCloseKinds, YBox};

    /// `ybox_crossing_prices_the_emptiness_decision`: the recorded slack is the
    /// largest `ly - uy`, and every box `YBox::is_empty` discharges carries a
    /// slack above that decision's own `1e-12` margin.
    #[test]
    fn ybox_crossing_prices_the_emptiness_decision() {
        let open = YBox {
            ly: vec![-1.0, 0.0, 0.5],
            uy: vec![1.0, 0.0, 0.5 + 1e-13],
        };
        assert!(!open.is_empty());
        assert!(ybox_crossing(&open) <= 1e-12);
        let crossed = YBox {
            ly: vec![-1.0, 0.3, 2.0],
            uy: vec![1.0, 0.1, 1.5],
        };
        assert!(crossed.is_empty());
        assert_eq!(ybox_crossing(&crossed).to_bits(), 0.5f64.to_bits());
        assert!(ybox_crossing(&crossed) > 1e-12);
    }

    /// `child_certs_fill_in_child_order_and_never_overwrite`: an expansion has
    /// at most two children; their certificates land in order, an
    /// (impossible) third never overwrites one, and the census counts are
    /// untouched.
    #[test]
    fn child_certs_fill_in_child_order_and_never_overwrite() {
        let mut k = ChildCloseKinds::default();
        k.note_cert(CertKind::Raw, 1e-3);
        k.note_cert(CertKind::HeadInfeasible, 2e-3);
        k.note_cert(CertKind::Lift, 3e-3);
        let got: Vec<(CertKind, u64)> = k
            .certs
            .iter()
            .flatten()
            .map(|c| (c.kind, c.value.to_bits()))
            .collect();
        assert_eq!(
            got,
            vec![
                (CertKind::Raw, 1e-3f64.to_bits()),
                (CertKind::HeadInfeasible, 2e-3f64.to_bits()),
            ]
        );
        assert_eq!((k.raw, k.lift, k.head_infeasible), (0, 0, 0));
        assert_eq!(k.retire.n, 0);
    }
}

impl ExpandStep {
    fn policy_execution(&self) -> PolicyExecutionEvidence {
        match self {
            Self::Infeasible(evidence, _)
            | Self::DeadlineHit(evidence)
            | Self::ClosedAtEval(evidence, _, _)
            | Self::NoCandidates(evidence)
            | Self::NonFiniteBound(evidence) => evidence.policy,
            Self::Expanded { execution, .. } => execution.policy,
        }
    }

    fn epoch_execution(&self) -> EpochExecutionEvidence {
        match self {
            Self::Infeasible(evidence, _)
            | Self::DeadlineHit(evidence)
            | Self::ClosedAtEval(evidence, _, _)
            | Self::NoCandidates(evidence)
            | Self::NonFiniteBound(evidence) => evidence.epoch,
            Self::Expanded { execution, .. } => execution.epoch,
        }
    }

    fn width_execution(&self) -> WidthExecutionEvidence {
        match self {
            Self::Infeasible(evidence, _)
            | Self::DeadlineHit(evidence)
            | Self::ClosedAtEval(evidence, _, _)
            | Self::NoCandidates(evidence)
            | Self::NonFiniteBound(evidence) => evidence.width,
            Self::Expanded { execution, .. } => execution.width,
        }
    }
}

/// Gate-on two-phase expansion: domain evaluation and candidate selection stay
/// independent; only `Ready` records contribute columns to the shared score
/// pass. `Done` carries the unchanged terminal result for its batch position.
enum PreparedExpand {
    // Boxed: ExpandStep carries the per-domain evidence records and is now the
    // LARGER variant (~848 bytes against Ready's boxed 8); keeping it inline
    // would size every batch slot at it. Layout only — the payload, its
    // construction order, and every verdict it carries are unchanged.
    Done(Box<ExpandStep>),
    // Boxed: ReadyExpand (NodeState + candidate vecs) is ~13x the unboxed Done
    // variant; keeping it inline would size every batch slot at the large variant.
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
    /// The frame's path-based cover ledger and authorization scope
    /// (#soundness-spine M2c).
    ledger: Ledger,
    /// AND of every nested epoch subtree's own ledger verdict. Telemetry only
    /// since M2c: a nested subtree now proves its leaf through its sub-domain
    /// token, which the cover gate checks.
    nested_ledgers_ok: bool,
    /// B1 plan-residency cache for the certified GPU lane, OWNED HERE.
    ///
    /// Ownership is the staleness argument: the cache lives exactly as long
    /// as the driver that owns `eng` (and therefore the `RootGates` its
    /// statics were built from), so a hit can never describe a different
    /// network. `PlanCache::probe` still re-validates a fail-closed identity
    /// fingerprint and every hit re-runs the per-dom gate admission. Inert
    /// (never probed) unless the lane is armed.
    lane_cache: super::gpu_backward::cache::PlanCache,
    /// #closure-adaptive: this tree's own child-closure observer. Owned per
    /// driver (so a nested epoch subtree measures itself) while the TIER it
    /// escalates to is process-global, which is what makes an escalation the
    /// parent earned stay in force inside the epoch.
    closure: super::closure::ClosureMonitor,
    /// #mr-stack (B5): this tree's main-eval stacking receipt and trajectory
    /// hash (telemetry only; see `bab/stack_eval.rs`).
    b5: stack_eval::B5Trace,
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
    /// #spec-alpha yield for THIS evaluation. Evidence only; never feeds a
    /// bound. `None` whenever the axis is dark, which is the shipped state.
    pub spec_alpha: Option<super::spec_alpha::SpecAlphaReport>,
}

impl RootEval {
    /// #frozen-dual-state: a copy of the certified head pre-activation box
    /// (y-box) THIS evaluation used — exactly the `YBox` `root_eval` built its
    /// head gates on. Read-only; the pack itself is untouched.
    #[must_use]
    pub fn root_ybox(&self) -> YBox {
        YBox {
            ly: self.pack.ly0.clone(),
            uy: self.pack.uy0.clone(),
        }
    }
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
/// ```text
/// open >= 32   ->  k = 16     (above every measured proving-row peak)
/// open >= 256  ->  k = 32
/// ```
///
/// The ratchet is one-way (never narrows below the configured base), so a row
/// that never explodes runs the configured width bit-identically.
///
/// SOUNDNESS: width chooses only WHICH domains get split — every candidate is
/// scored by the same certified batched pass, and no bound computation reads
/// `k`. A wrong width costs proofs; it cannot manufacture one.
const fn ordinary_width_selectors_enabled(tier0_exact: usize, retained: bool) -> bool {
    !(tier0_exact > 0 && retained)
}

/// #branch-reduceop: THE single candidate-pick rule, shared by all three
/// expansion paths.
///
/// WHY A HELPER AND NOT THREE COPIES. There are three pick sites — the serial
/// loop, the parallel per-domain step, and the domain-stacked batch step — and
/// the SCORED cifar100 lane runs the PARALLEL ones because the compiled
/// `frontier` default is `rayon::current_num_threads()` (mod.rs:416). The first
/// version of this lever patched only the
/// serial site; its R9 observable then printed ZERO lines on a scored run and
/// the whole A/B had to be voided. That is the campaign's rule 8c — "partial
/// fixes of a set measure exactly zero" — and the fix is to make the set
/// impossible to patch partially.
///
/// Returns `(pick, differs)`, where `differs` is true when the OTHER reduction
/// would have chosen a different candidate. That is the R9 observable: it
/// separates "the lever never ran" (no telemetry at all) from "it ran and the
/// two reductions agree on this row" (`differs=0`, a REAL null) from "it ran and
/// changed the branch" (`differs=1`).
///
/// `reduce_max` selects De Palma eq. (22), the original BaBSR s_SR score (rank a
/// candidate by its BETTER child); `false` selects eq. (23), the FSB balanced
/// score (rank by its WORSE child). abc's TUNED cifar100 and tinyimagenet
/// configs both choose the former, overriding abc's own global default, on
/// exactly the two benchmarks ny fails.
///
/// RANKER-ONLY, hence sound either way: every child bound still comes from the
/// same certified pass and `closed()` is untouched, so a bad choice costs tree
/// shape and can never manufacture a proof.
fn pick_candidate(ch: &[(f64, f64)], reduce_max: bool) -> (usize, bool) {
    let key_of = |pair: &(f64, f64), use_max: bool| {
        let reduced = if use_max {
            pair.0.max(pair.1)
        } else {
            pair.0.min(pair.1)
        };
        (reduced, pair.0 + pair.1)
    };
    let best_of = |use_max: bool| {
        let mut pick = 0usize;
        let mut best = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for (i, pair) in ch.iter().enumerate() {
            let key = key_of(pair, use_max);
            if key.0 > best.0 || (key.0 == best.0 && key.1 >= best.1) {
                best = key;
                pick = i;
            }
        }
        pick
    };
    let pick = best_of(reduce_max);
    let alt = best_of(!reduce_max);
    (pick, pick != alt)
}

/// Rate-limited R9 line for [`pick_candidate`]. First 8 picks, then every 64th.
fn note_reduceop(reduce_max: bool, site: &str, cands: usize, pick: usize, differs: bool) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 8 || n.is_multiple_of(64) {
        eprintln!(
            "[branch-reduceop] n={n} site={site} mode={} cands={cands} pick={pick} differs={}",
            if reduce_max { "max" } else { "min" },
            u8::from(differs),
        );
    }
}

/// #branch-parent-lift: recover the certified evidence the expansion throws away.
///
/// `score_candidates` runs a full certified backward pass over EVERY shortlisted
/// candidate, producing `ch[i] = (l_i, r_i)`: sound lower bounds on the two
/// children that candidate `i` would create. The expansion then splits on ONE
/// candidate and discards the other ~31 pairs. But each discarded pair still
/// certifies something about the PARENT:
///
/// Splitting `D` on candidate `i` partitions it into `D_i^+` and `D_i^-` with
/// `D = D_i^+ ∪ D_i^-` (a ReLU sign split covers both signs; a head split
/// covers `y_i >= 0` and `y_i <= 0`). Every `x ∈ D` therefore lies in at least
/// one child, so
///     margin(x) >= min(l_i, r_i)   for all x in D.
/// That holds for each candidate INDEPENDENTLY, so the best of them
///     lift = max_i min(l_i, r_i)
/// is a valid lower bound on the parent — available for free, already paid for.
///
/// SOUNDNESS, precisely. `l_i` and `r_i` are the same `concretize_lower`
/// outputs the expansion ALREADY relies on to CLOSE a child (`closed(mode, cb)`
/// on `cb = raw.max(b)`, pushed at the `closed_children` sites with no
/// re-evaluation). If they were not sound lower bounds ny would already be
/// unsound there, so that half is inherited, not new. `closed()` is untouched
/// and closure stays strict; the lift only raises a bound that is already
/// published, through the SAME monotone floor, never by weakening a test.
///
/// ⚠ BUT IT IS NOT TRUST-FREE. The lift needs ONE premise the existing closure
/// test does not: that candidate `i`'s two children COVER the parent. Closing a
/// child only needs that child's own bound; concluding something about D needs
/// the pair to leave no part of D unaccounted for. See [`CANDIDATES_ARE_A_COVER`]
/// — that invariant is what makes this sound, and it is the thing a future
/// candidate kind must preserve.
///
/// An empty child (`ly > uy`, the infeasible case the split loop skips) can only
/// make `min(l_i, r_i)` SMALLER than the truth, so the reduction stays
/// conservative there rather than over-claiming.
///
/// WHY IT PAYS. The frontier is `min` over open domains of their bounds, and
/// this lane is expansion-limited: lifting the parent lifts BOTH children, so
/// the gain propagates into the frontier instead of being spent on tree shape.
/// When `lift > 0` both children close, so the branch contributes no open
/// domain — though they are still CREATED and counted before closing, because
/// the parent's own `b` is not raised. That covers the closure blind spot of the
/// `max` reduction, which ranks by the BETTER child and can pass over a
/// candidate whose worse child also closes.
/// THE INVARIANT [`parent_lift`] RESTS ON: for every branching candidate, the
/// two children the expansion builds from it COVER the parent domain.
///
/// It holds for both candidate kinds the lane has, and each is a dichotomy:
/// * TRUNK — a ReLU sign split. `candidate_columns` fixes the gate to the exact
///   identity line `(1,1,0)` and the exact zero line `(0,0,0)`; the constructed
///   children are `trunk + (li,pos,+1)` and `trunk + (li,pos,-1)`. Every point
///   has a sign, so the pair covers.
/// * HEAD — `ly[i].max(0.0)` against `uy[i].min(0.0)`, i.e. `y_i >= 0` against
///   `y_i <= 0`. They overlap at zero and cover `[ly, uy]`. When one side is
///   infeasible the other IS the parent, and `min` (never `max`) keeps that
///   case conservative.
///
/// A candidate kind that is NOT a dichotomy — a 3-way interval split, a
/// tighten-both-sides split, a split that drops a boundary — would break this
/// while every existing test stayed green, because nothing else in the lane
/// needs the cover: closing a child only ever needs that child's own bound.
/// **Anything added to `select_candidates` must preserve this, or `parent_lift`
/// must be taught to skip it.**
#[allow(dead_code)]
const CANDIDATES_ARE_A_COVER: () = ();

fn parent_lift(ch: &[(f64, f64)]) -> f64 {
    let mut lift = f64::NEG_INFINITY;
    for &(l, r) in ch {
        if l.is_finite() && r.is_finite() {
            let both = l.min(r);
            if both > lift {
                lift = both;
            }
        }
    }
    lift
}

#[cfg(test)]
mod parent_lift_tests {
    use super::parent_lift;

    #[test]
    fn lift_is_the_best_worse_child_over_all_candidates() {
        // Candidate 0 has a spectacular better child and an open worse one;
        // candidate 1 is mediocre on both but its WORSE child is positive, so
        // it is the one that certifies the parent.
        let ch = [(5.0, -0.1), (0.001, 0.002)];
        assert_eq!(parent_lift(&ch), 0.001);
    }

    #[test]
    fn lift_ignores_non_finite_pairs_rather_than_propagating_them() {
        let ch = [
            (f64::INFINITY, f64::INFINITY),
            (-0.5, -0.25),
            (f64::NAN, 1.0),
        ];
        // -0.5 is the worse child of the only finite pair.
        assert_eq!(parent_lift(&ch), -0.5);
    }

    #[test]
    fn all_non_finite_yields_neg_infinity_which_a_max_floor_absorbs() {
        let ch = [(f64::NAN, 1.0), (f64::INFINITY, f64::NEG_INFINITY)];
        let lift = parent_lift(&ch);
        assert!(lift.is_infinite() && lift.is_sign_negative());
        // The floor it feeds must be a no-op in that case.
        let raw = -0.3f64;
        let b = -0.4f64;
        assert_eq!(raw.max(b).max(lift), raw.max(b));
    }

    #[test]
    fn empty_candidate_set_cannot_manufacture_a_bound() {
        let lift = parent_lift(&[]);
        assert!(lift.is_infinite() && lift.is_sign_negative());
    }

    #[test]
    fn the_reduction_is_min_not_max_so_an_infeasible_side_stays_conservative() {
        // If one child is infeasible its score is an ordinary finite number
        // that does NOT bound the parent; `min` keeps the pair honest, `max`
        // would not. Here 9.0 must never be the answer.
        let ch = [(9.0, -0.2)];
        assert_eq!(parent_lift(&ch), -0.2);
    }
}

/// #branch-parent-lift PERCOL probe. `score_candidates` collapses the COLUMN
/// axis with `min` before [`parent_lift`] ever sees a pair, so the shipped lift
/// is `max_i min_f`. But the domain's bound is itself `min_f`, and each COLUMN
/// may be certified by a DIFFERENT candidate, so the sound form is
///     lift_percol = min_f ( max_i min(l_{i,f}, r_{i,f}) )
/// and `max_i min_f <= min_f max_i` always, so this is >= the shipped lift and
/// equally sound. This probe MEASURES the headroom between the two before any
/// plumbing is paid for; it prints and returns nothing, so verdicts are
/// byte-identical whether or not it fires.
///
/// ANSWERED 2026-08-30, and the answer is ZERO. Across 71 sampled expansions on
/// the cifar100 pool the headroom was `0.000000` on EVERY one — because
/// `#margin-row-col-retire` has already collapsed the lane to `nf=1` (51 of 71
/// samples) or `nf=2` (20 of 71) before branching begins, and at `nf=1` the two
/// reduction orders are trivially identical. The refinement is real in theory
/// and worth nothing here. KEEP THIS PROBE: it is the cheap standing check that
/// would fire if a future change kept more columns alive into the tree, which
/// is the only world where re-plumbing the certified scorer would pay.
fn note_parent_lift_percol(site: &str, nf: usize, cands: usize, collapsed: f64, percol: f64) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 8 || n.is_multiple_of(64) {
        let head = if percol.is_finite() && collapsed.is_finite() {
            percol - collapsed
        } else {
            0.0
        };
        eprintln!(
            "[parent-lift-percol] n={n} site={site} nf={nf} cands={cands} \
             collapsed={collapsed:.6} percol={percol:.6} headroom={head:.6}"
        );
    }
}

/// Rate-limited observable for [`parent_lift`]: first 8 expansions, then every
/// 64th. `gain` is the amount the parent bound actually rose, so `gain=0` is a
/// REAL null (the split candidate was already the best certifier) and the
/// absence of the line entirely means the lever never ran.
fn note_parent_lift(site: &str, cands: usize, b: f64, lift: f64, closes: bool) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 8 || n.is_multiple_of(64) {
        let gain = if lift.is_finite() && lift > b {
            lift - b
        } else {
            0.0
        };
        eprintln!(
            "[branch-parent-lift] n={n} site={site} cands={cands} b={b:.6} lift={lift:.6} \
             gain={gain:.6} closes={}",
            u8::from(closes),
        );
    }
}

/// THE margin-row closure predicate (#soundness-spine M2b). It routes through
/// the certified facade: only an Outward engine value becomes a
/// `Certified<f64>`, and `certify::closed` on it is strict (`b > 0`) and finite
/// — bit-identical to the historical Outward branch. A non-finite result is
/// not a proof (`+inf` must not close). The historical Parity branch
/// (`b >= 0`) is GONE: a round-to-nearest value is a `SearchValue`, which has
/// no closure predicate. (The tree never ran in Parity — `run` and
/// `run_classwise` refuse it — so no verdict moves.)
pub(crate) fn closed(mode: RoundMode, b: f64) -> bool {
    cert_mr::closes(mode, b)
}

/// Root-evaluation SCHEDULING: which classes get the direct pass and which
/// form the tree set. Outward: exactly [`closed`]. Parity (the measurement
/// mode no verdict path accepts): the historical `>= 0` schedule on the
/// SEARCH value, so parity measurements against the reference stay
/// bit-identical. This is a schedule, never a closure: nothing that reads it
/// can publish.
fn root_schedules_closed(mode: RoundMode, b: f64) -> bool {
    match cert_mr::classify(mode, b) {
        cert_mr::MarginRowValue::Certified(c) => crate::certify::closed(&c),
        cert_mr::MarginRowValue::Search(v) => v.hint().is_finite() && v.hint() >= 0.0,
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

pub(super) fn build_pack(
    eng: &BackwardEngine<'_>,
    dom: Option<&DomainGates>,
    splits: &[TrunkSplit],
) -> Result<YPack> {
    build_pack_traced(eng, dom, splits, None)
}

fn build_pack_traced(
    eng: &BackwardEngine<'_>,
    dom: Option<&DomainGates>,
    splits: &[TrunkSplit],
    trace: Option<&root_eval_progress::Evaluation>,
) -> Result<YPack> {
    let head_group_source = if dom.is_none() && splits.is_empty() {
        super::head_group_source::YPackSource::capture(eng)
    } else {
        None
    };
    // The identity-seeded y-row refresh is the seam's cleanest admission: the
    // seed is f32-exact and carries no certified error, so no `y_abs` is
    // needed, and one device walk publishes BOTH lanes. Dark + fail-closed:
    // with `NY_MARGIN_ROW_GPU` unset this is the established `y_rows` call.
    let (al, au) = {
        let stage = trace.and_then(|t| t.stage("y_rows", 2 * eng.net.n_y));
        // #mr-census: two identity-seeded passes of n_y columns each.
        let _census = super::census::SiteTimer::start(super::census::Site::YRows, 2 * eng.net.n_y);
        let t_rows = Instant::now();
        let rows = if head_group_source.is_some() {
            // The source qualifier applies to the repaired CPU producer,
            // whose absolute underflow budget the optional replay relies on.
            // Separate device programs cannot inherit that qualification.
            BackwardEngine::new(eng.net, eng.root)
                .with_err_lane(eng.err_lane())
                .with_gpu_conv_route(super::gpu_conv::GpuConvRoute::Off)
                .y_rows(dom)?
        } else {
            eng.y_rows_seamed(dom, &super::gpu_seam::SeamCtx::default())?
        };
        // #root-eval-reuse: what a rows-memo hit saves (read by the store).
        LAST_Y_ROWS_NS.with(|c| c.set(super::census::elapsed_ns(t_rows)));
        root_eval_progress::Stage::finish(stage);
        rows
    };
    let stage = trace.and_then(|t| t.stage("pack_tail", eng.net.n_y));
    let mut pack = pack_from_rows(eng, al, au);
    clip_tighten_pack(eng, splits, &mut pack);
    pack.head_group_source = head_group_source.and_then(|mut source| {
        source
            .seal_rows(&pack.al, &pack.au, &pack.ly0, &pack.uy0)
            .then_some(source)
    });
    root_eval_progress::Stage::finish(stage);
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
    // FAIL-CLOSED FRAME GATE, now shared with the intermediate-gate consumer:
    // see `clip::frame_ok` for why it moved out of this function.
    if !super::clip::frame_ok(root) {
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

/// #clip-and-verify: default OFF. The typed preset route makes controlled scored
/// A/Bs possible, but no shipped preset arms it; qualification still requires
/// evidence rather than following automatically from delivery.
///
/// #closure-adaptive: deliberately NOT a rung of the runtime ladder. The 24-run
/// matrix measured `clip` BIT-IDENTICAL to baseline (same expansions, same
/// closures, 20 sample points) with the mechanism provably armed, and
/// `beta+clip` bit-identical to `beta` alone — see `closure.rs` for the
/// measurement and the high-dimensional geometry that explains it. Left on its
/// own gate; a rung that costs time and changes nothing is worse than no
/// rung. Note also that this particular sub-mechanism has never been VALIDLY
/// measured on its own: the y-pack LRU serves cached packs, so `[clip] call=`
/// tops out around 4 per run.
fn clip_tighten_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(crate::preset_gates::margin_row_clip_enabled)
}

/// #clip-head per-eval outcome: what the head-split halfspaces did to THIS
/// domain's y-box.
#[derive(Debug, Clone, Copy, Default)]
struct ClipHeadOut {
    /// Head sign fixes the domain carries.
    heads: usize,
    /// Halfspaces actually built from them.
    hs: usize,
    /// Head columns offered to the dual.
    cols: usize,
    /// Columns whose interval strictly shrank.
    moved: usize,
    /// Total shrink across those columns (both sides).
    sum_delta: f64,
    /// Columns that became STABLE (`ly >= 0` or `uy <= 0`) — abc's mechanism:
    /// 15/16 and 16/16 children verify at iteration <= 1 once enough head
    /// neurons close.
    closes: usize,
    /// Columns whose tightened interval would have CROSSED. Refused and kept at
    /// the parent values: reading a crossing as an emptiness certificate
    /// discharges a domain with no backward pass to disagree, which is the
    /// false-`unsat` surface the stateful trunk port deliberately left shut.
    refused_cross: usize,
    /// The halfspace build itself refused (index/finiteness).
    refused_hs: usize,
    /// The pack carried no recovered slack (gate dark, or frame check refused).
    frame_failed: usize,
}

/// #clip-head: announce the gate's resolved state ONCE per process, on the
/// first eval that reaches the insertion point — before any domain can
/// decline, exactly as `[beta] armed` does, and UNCONDITIONALLY: a dark run
/// prints `armed=false`, so a control log proves the same binary ran the same
/// site with the gate off, rather than merely being silent. A null from a
/// lever whose telemetry never printed is VOID, not negative.
fn clip_head_armed_once(on: bool) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "[clip-head] armed={on} h_cap={} c_cap={}",
            super::clip::HEAD_H_CAP,
            super::clip::HEAD_C_CAP
        );
    });
}

/// #clip-head process-global engagement counters. Written by every worker
/// (the eval is `&self` and runs across rayon workers), read once at lane exit.
static CLIP_HEAD_CALLS: ClipHeadCounter = ClipHeadCounter::new(0);
static CLIP_HEAD_ENGAGED: ClipHeadCounter = ClipHeadCounter::new(0);
static CLIP_HEAD_MOVED: ClipHeadCounter = ClipHeadCounter::new(0);
static CLIP_HEAD_CLOSES: ClipHeadCounter = ClipHeadCounter::new(0);
static CLIP_HEAD_REFUSED: ClipHeadCounter = ClipHeadCounter::new(0);
static CLIP_HEAD_REFUSED_HS: ClipHeadCounter = ClipHeadCounter::new(0);
static CLIP_HEAD_FRAME_FAILED: ClipHeadCounter = ClipHeadCounter::new(0);

/// #clip-head per-eval telemetry, rate-limited like `[clip] call=`.
fn clip_head_report(ch: &ClipHeadOut, us: u128) {
    let n = CLIP_HEAD_CALLS.fetch_add(1, ClipHeadOrder::Relaxed);
    if ch.hs > 0 {
        CLIP_HEAD_ENGAGED.fetch_add(1, ClipHeadOrder::Relaxed);
    }
    CLIP_HEAD_MOVED.fetch_add(ch.moved, ClipHeadOrder::Relaxed);
    CLIP_HEAD_CLOSES.fetch_add(ch.closes, ClipHeadOrder::Relaxed);
    CLIP_HEAD_REFUSED.fetch_add(ch.refused_cross, ClipHeadOrder::Relaxed);
    CLIP_HEAD_REFUSED_HS.fetch_add(ch.refused_hs, ClipHeadOrder::Relaxed);
    CLIP_HEAD_FRAME_FAILED.fetch_add(ch.frame_failed, ClipHeadOrder::Relaxed);
    if n < 4 || n.is_multiple_of(64) {
        eprintln!(
            "[clip-head] call={n} heads={} hs={} cols={} moved={} sum_delta={:.6e} \
             closes={} refused_cross={} refused_hs={} ms={:.3}",
            ch.heads,
            ch.hs,
            ch.cols,
            ch.moved,
            ch.sum_delta,
            ch.closes,
            ch.refused_cross,
            ch.refused_hs,
            us as f64 / 1000.0,
        );
    }
}

/// #clip-head lane-exit receipt. An armed run ALWAYS leaves one, so "ran and
/// found nothing" is distinguishable from "never ran" without opening the log.
fn clip_head_finish() {
    if !crate::preset_gates::margin_row_clip_head_enabled() {
        return;
    }
    let calls = CLIP_HEAD_CALLS.load(ClipHeadOrder::Relaxed);
    let engaged = CLIP_HEAD_ENGAGED.load(ClipHeadOrder::Relaxed);
    eprintln!(
        "[clip-head] done evals={calls} engaged={engaged} moved_total={} closes_total={} \
         refused_cross_total={} refused_hs_total={} frame_failed={}",
        CLIP_HEAD_MOVED.load(ClipHeadOrder::Relaxed),
        CLIP_HEAD_CLOSES.load(ClipHeadOrder::Relaxed),
        CLIP_HEAD_REFUSED.load(ClipHeadOrder::Relaxed),
        CLIP_HEAD_REFUSED_HS.load(ClipHeadOrder::Relaxed),
        CLIP_HEAD_FRAME_FAILED.load(ClipHeadOrder::Relaxed),
    );
    if engaged == 0 {
        eprintln!(
            "[clip-head] ARMED BUT INERT: no evaluated domain produced a head halfspace \
             - any measurement from this run is VACUOUS, not negative"
        );
    }
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
    // #clip-head: capture the FORWARD concretization here, at the one place the
    // rows and their untightened publication coexist. `clip_tighten_pack` runs
    // immediately after `build_pack` calls this (and the batched prefill does
    // the same) and mutates `ly0/uy0` in place, so anything recovered later
    // would be calibrated against a tightened bound.
    let ly_fwd = ybox.ly.clone();
    let uy_fwd = ybox.uy.clone();
    // Computed ONLY when the gate is armed: the dark lane then does no extra
    // work at all and prints no frame line, which is what makes the OFF arm
    // byte-identical AND free.
    let head_slack = if crate::preset_gates::margin_row_clip_head_enabled() {
        super::clip::head_slack(
            &al,
            &au,
            &ly_fwd,
            &uy_fwd,
            &eng.root.mid,
            &eng.root.rad,
            &eng.root.xabs,
            0.5,
        )
    } else {
        None
    };
    YPack {
        head_group_source: None,
        prepared_owner: None,
        ly0: ybox.ly,
        uy0: ybox.uy,
        ly_fwd,
        uy_fwd,
        head_slack,
        al_dots,
        au_dots,
        al,
        au,
    }
}

// ---------------------------------------------------------------------------
//  #root-eval-reuse — L1 root-eval dedupe (margin-row throughput plan B1(c))
// ---------------------------------------------------------------------------
//
// THE WASTE. With the root alpha ascent and its published-frame guard armed
// (the shipped cifar100/tinyimagenet presets), the root is fully evaluated up
// to THREE times, and its y-rows derived once more, on at most two gate states
// before the tree starts:
//   1. `alpha_opt`'s published-frame guard baseline (`published_before`,
//      `rebake::certified_root_bound`) on the entry gates G0;
//   2. the ascent's frozen frame re-derives G0's y-rows (`search_alpha`'s
//      `eng.y_rows(None)`) — the SAME 2 x n_y pass pair root_eval just ran;
//   3. after an accepted iterate, the guard's `published_after` evaluates G1;
//   4. `run_inner` then evaluates the published root AGAIN — G1 if the guard
//      kept the iterate (all 22 arms of the 2026-09-23 tinyimagenet head-alpha
//      A/B log `published_guard ... accepted=true`), G0 if it rolled back —
//      bit-for-bit the evaluation (1) or (3) already produced.
// (2) and (4) are pure recomputation.
//
// THE MEMO. `certified_root_bound` STORES its evaluation on the `RootGates` it
// evaluated (`RootGates::root_eval_memo`, at most two entries); `run_inner`
// and `run_classwise_inner` LOOK UP before evaluating and then release the
// memo; `search_alpha` borrows the stored pack's y-rows. Every lookup is keyed
// on the GATES GENERATION — a 128-bit content fingerprint of EVERY
// `RootGates` field `root_eval` reads (mode, box, per-layer op/n/l/u/alpha/s/
// c/ms/unst, clip-row presence and shape, head alpha, spec-alpha presence) —
// plus the net's identity and head fingerprint, the property `(t, adv)`, and
// the one process-global gate that shapes the pack (`clip_head`).
//
// WHY A CONTENT FINGERPRINT RATHER THAN A HAND-BUMPED COUNTER. `RootGates`'
// fields are `pub` and are written in place at no fewer than five sites (the
// ascent's commit and rollback, the rebake snapshot/restore and wholesale
// replacement, the spec-alpha install). A counter is one forgotten bump away
// from serving a stale pack; a fingerprint of the values themselves cannot be
// forgotten. A wholesale replacement (`*root = cand`) carries the candidate's
// OWN memo, which is exactly the evaluation of the root now published.
//
// BIT-IDENTITY. `root_eval` is a deterministic function of exactly those
// inputs (pinned by `root_eval_reuse_bit_identical`), so a hit returns the
// value a recomputation would. The memo REFUSES — never serves, never stores —
// when either GPU lane or the #mr-gpu-eval conv route is armed (device refusals
// make a pass run-dependent) or a spec-alpha policy is installed (its
// refinement is deadline-dependent).
//
// SOUNDNESS does not rest on the key. Every memoized `RootEval` is itself a
// certified evaluation of the SAME network over the SAME input box under a
// valid relaxation; even a (2^-128) fingerprint collision would publish a
// valid-but-different certified bound, never an unsound one.

thread_local! {
    /// Wall of the most recent `build_pack` y-row pair on this thread: what a
    /// rows-memo hit saves. Written by `build_pack`, read by the store.
    static LAST_Y_ROWS_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

static REUSE_LOOKUPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REUSE_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REUSE_ROW_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REUSE_SAVED_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REUSE_STORES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REUSE_REFUSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Two-lane word fingerprint (FNV-style xor-multiply beside a rotate-xor-
/// multiply lane), length-prefixed per field so concatenations cannot alias.
struct GateFingerprint {
    a: u64,
    b: u64,
    n: u64,
}

impl GateFingerprint {
    fn new() -> Self {
        Self {
            a: 0xcbf2_9ce4_8422_2325,
            b: 0x6a09_e667_f3bc_c909,
            n: 0,
        }
    }

    #[inline]
    fn word(&mut self, w: u64) {
        self.a = (self.a ^ w).wrapping_mul(0x0000_0100_0000_01b3);
        self.b = (self.b.rotate_left(29) ^ w).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.n = self.n.wrapping_add(1);
    }

    fn f64s(&mut self, v: &[f64]) {
        self.word(v.len() as u64);
        for x in v {
            self.word(x.to_bits());
        }
    }

    fn usizes(&mut self, v: &[usize]) {
        self.word(v.len() as u64);
        for &x in v {
            self.word(x as u64);
        }
    }

    fn finish(&self) -> [u64; 2] {
        [self.a ^ self.n.rotate_left(17), self.b.wrapping_add(self.n)]
    }
}

/// The GATES GENERATION: a content fingerprint of every `RootGates` field that
/// `root_eval` reads. Any in-place write to any of them moves it.
fn gates_generation(root: &RootGates) -> [u64; 2] {
    let mut fp = GateFingerprint::new();
    fp.word(u64::from(root.mode.outward()));
    fp.f64s(&root.mid);
    fp.f64s(&root.rad);
    fp.f64s(&root.xabs);
    fp.f64s(&root.lo);
    fp.f64s(&root.hi);
    fp.word(root.layers.len() as u64);
    for layer in &root.layers {
        fp.word(layer.op as u64);
        fp.word(layer.n as u64);
        fp.f64s(&layer.l);
        fp.f64s(&layer.u);
        fp.f64s(&layer.alpha);
        fp.f64s(&layer.s);
        fp.f64s(&layer.c);
        fp.f64s(&layer.ms);
        fp.usizes(&layer.unst);
        match &layer.clip_rows {
            Some(rows) => {
                fp.word(1);
                fp.word(rows.m.nrows() as u64);
                fp.word(rows.m.ncols() as u64);
            }
            None => fp.word(0),
        }
    }
    match &root.head_alpha {
        Some(head) => {
            fp.word(1);
            fp.f64s(head);
        }
        None => fp.word(0),
    }
    fp.word(u64::from(root.spec_alpha_policy.is_some()));
    fp.finish()
}

/// Net identity beyond its address: shape plus the head weights.
fn net_fingerprint(net: &TwinNet) -> [u64; 2] {
    let mut fp = GateFingerprint::new();
    fp.word(net.n_in as u64);
    fp.word(net.n_y as u64);
    fp.word(net.n_out as u64);
    fp.word(net.i_gemm1 as u64);
    fp.word(net.ops.len() as u64);
    fp.usizes(&net.tsize);
    fp.usizes(&net.trunk_relus);
    let (w1, b1, _) = net.gemm1();
    fp.f64s(w1);
    fp.f64s(b1);
    let (fold_bias_err, fold_weight_err) = net.gemm1_fold_err();
    fp.f64s(fold_bias_err);
    fp.word(fold_weight_err.to_bits());
    let (w2, b2, _) = net.gemm2();
    fp.f64s(w2);
    fp.f64s(b2);
    fp.finish()
}

#[derive(Clone, PartialEq, Eq)]
struct RootEvalKey {
    gates: [u64; 2],
    net_addr: usize,
    net: [u64; 2],
    t: usize,
    adv: Vec<usize>,
    clip_head: bool,
}

impl RootEvalKey {
    fn of(net: &TwinNet, root: &RootGates, t: usize, adv: &[usize]) -> Self {
        Self {
            gates: gates_generation(root),
            net_addr: std::ptr::from_ref(net) as usize,
            net: net_fingerprint(net),
            t,
            adv: adv.to_vec(),
            clip_head: crate::preset_gates::margin_row_clip_head_enabled(),
        }
    }

    /// The y-rows depend on the gates and the net only (identity seed: no
    /// property, no head gates).
    fn same_rows(&self, other: &Self) -> bool {
        self.gates == other.gates && self.net_addr == other.net_addr && self.net == other.net
    }
}

struct MemoEntry {
    key: RootEvalKey,
    dj: Vec<f64>,
    tree_classes: Vec<usize>,
    pack: Arc<YPack>,
    /// Wall of the evaluation that produced this entry (what a hit saves).
    eval_ns: u64,
    /// Wall of its y-row pair (what a rows hit saves).
    rows_ns: u64,
}

/// Per-`RootGates` memo of `root_eval` results (see the section comment).
/// Every `RootGates` constructor starts it EMPTY; it is never inherited.
#[derive(Default)]
pub(crate) struct RootEvalMemo(
    std::sync::Mutex<Vec<MemoEntry>>,
    std::sync::Mutex<prepared_root_eval::PreparedMemoSlots>,
);

impl RootEvalMemo {
    /// At most the entry gates' and one candidate's evaluation (the guard's
    /// before/after pair), so a rollback still hits.
    const CAP: usize = 2;

    fn put(&self, entry: MemoEntry) {
        if let Ok(mut entries) = self.0.lock() {
            entries.retain(|e| e.key != entry.key);
            entries.push(entry);
            while entries.len() > Self::CAP {
                entries.remove(0);
            }
        }
    }

    fn clear(&self) {
        if let Ok(mut entries) = self.0.lock() {
            entries.clear();
        }
        if let Ok(mut prepared) = self.1.lock() {
            prepared.clear();
        }
    }
}

/// The memo refuses (never serves, never stores) when an evaluation is not a
/// deterministic function of the key.
fn memo_eligible(eng: &BackwardEngine<'_>, net: &TwinNet) -> bool {
    std::ptr::eq(eng.net, net)
        && eng.root.spec_alpha_policy.is_none()
        && !super::gpu_seam::enabled()
        && !super::gpu_backward::enabled()
        // #mr-gpu-eval: a device refusal (lock budget, non-CUDA slot) sends a
        // conv to the CPU kernel, so an armed pass is run-dependent at the ulp
        // level — the same reason the two GPU lanes above are refused.
        && eng.gpu_conv_route().is_off()
}

/// Evaluate and STORE on the evaluated root (`rebake::certified_root_bound`,
/// i.e. the alpha guard's `published_before`/`published_after` and every
/// rebake round). Returns exactly what [`root_eval`] returns.
pub(crate) fn root_eval_store(
    eng: &BackwardEngine<'_>,
    net: &TwinNet,
    t: usize,
    adv: &[usize],
) -> Result<RootEval> {
    let t0 = Instant::now();
    LAST_Y_ROWS_NS.with(|c| c.set(0));
    let re = root_eval_impl(eng, net, t, adv)?;
    // #attack-open-disjuncts: this function is reached only through
    // `rebake::certified_root_bound`, i.e. the alpha guard's published-frame
    // evaluations and the rebake rounds — all of the lane's TOP-LEVEL root,
    // never a nested epoch subdomain — so `re.dj` is a certified full-box
    // bound. Attack scheduling only; one atomic load when unarmed.
    super::root_open::publish_outer_root(t, adv, &re.dj);
    let eval_ns = super::census::elapsed_ns(t0);
    if memo_eligible(eng, net) {
        eng.root.root_eval_memo.put(MemoEntry {
            key: RootEvalKey::of(net, eng.root, t, adv),
            dj: re.dj.clone(),
            tree_classes: re.tree_classes.clone(),
            pack: Arc::clone(&re.pack),
            eval_ns,
            rows_ns: LAST_Y_ROWS_NS.with(std::cell::Cell::get),
        });
        REUSE_STORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    Ok(re)
}

/// Serve a stored evaluation of THIS root, gates generation, net and property.
fn root_eval_lookup(
    eng: &BackwardEngine<'_>,
    net: &TwinNet,
    t: usize,
    adv: &[usize],
) -> Option<RootEval> {
    use std::sync::atomic::Ordering::Relaxed;
    REUSE_LOOKUPS.fetch_add(1, Relaxed);
    if !memo_eligible(eng, net) {
        REUSE_REFUSED.fetch_add(1, Relaxed);
        return None;
    }
    let key = RootEvalKey::of(net, eng.root, t, adv);
    let entries = eng.root.root_eval_memo.0.lock().ok()?;
    let hit = entries.iter().find(|e| e.key == key)?;
    let hits = REUSE_HITS.fetch_add(1, Relaxed) + 1;
    let saved = REUSE_SAVED_NS.fetch_add(hit.eval_ns, Relaxed) + hit.eval_ns;
    eprintln!(
        "[root-eval-reuse] hit kind=root_eval hits={hits} saved_ms={:.1} this_ms={:.1}",
        saved as f64 / 1e6,
        hit.eval_ns as f64 / 1e6
    );
    Some(RootEval {
        dj: hit.dj.clone(),
        tree_classes: hit.tree_classes.clone(),
        pack: Arc::clone(&hit.pack),
        // Eligibility requires `spec_alpha_policy == None`, under which a fresh
        // evaluation reports `None` too.
        spec_alpha: None,
    })
}

/// `run_inner`/`run_classwise_inner`'s root evaluation: a memo hit, else the
/// unchanged evaluation. The memo is then RELEASED — the tree holds the pack
/// it needs, and a retained sibling entry would only pin memory.
pub(super) fn root_eval_reuse(
    eng: &BackwardEngine<'_>,
    net: &TwinNet,
    t: usize,
    adv: &[usize],
) -> Result<RootEval> {
    let re = match root_eval_lookup(eng, net, t, adv) {
        Some(hit) => hit,
        None => root_eval_impl(eng, net, t, adv)?,
    };
    eng.root.root_eval_memo.clear();
    Ok(re)
}

/// Borrowed y-rows of a stored evaluation (see [`root_rows_memo`]).
pub(crate) struct MemoRows(Arc<YPack>);

impl MemoRows {
    /// Lower-lane y-rows, bit-identical to `BackwardEngine::y_rows(None).0`.
    pub(crate) fn al(&self) -> &PassOut {
        &self.0.al
    }

    /// Upper-lane y-rows, bit-identical to `BackwardEngine::y_rows(None).1`.
    pub(crate) fn au(&self) -> &PassOut {
        &self.0.au
    }
}

/// The root alpha ascent's frozen frame (`alpha_opt::search_alpha`) re-derives
/// the published root's y-rows. When a stored evaluation of the SAME gates
/// generation exists, its pack's rows ARE those rows (identity seed, no
/// splits, GPU lanes refused by eligibility), so borrow them.
pub(crate) fn root_rows_memo(net: &TwinNet, root: &RootGates) -> Option<MemoRows> {
    use std::sync::atomic::Ordering::Relaxed;
    let eng = BackwardEngine::new(net, root);
    REUSE_LOOKUPS.fetch_add(1, Relaxed);
    if !memo_eligible(&eng, net) {
        REUSE_REFUSED.fetch_add(1, Relaxed);
        return None;
    }
    // `t`/`adv` are not part of a rows match; any placeholder works.
    let key = RootEvalKey::of(net, root, 0, &[]);
    let entries = root.root_eval_memo.0.lock().ok()?;
    let hit = entries.iter().find(|e| e.key.same_rows(&key))?;
    let hits = REUSE_ROW_HITS.fetch_add(1, Relaxed) + 1;
    let saved = REUSE_SAVED_NS.fetch_add(hit.rows_ns, Relaxed) + hit.rows_ns;
    eprintln!(
        "[root-eval-reuse] hit kind=rows row_hits={hits} saved_ms={:.1} this_ms={:.1}",
        saved as f64 / 1e6,
        hit.rows_ns as f64 / 1e6
    );
    Some(MemoRows(Arc::clone(&hit.pack)))
}

/// Reset the process-global reuse counters (lane start; `census::reset`).
pub(crate) fn root_eval_reuse_reset() {
    use std::sync::atomic::Ordering::Relaxed;
    for counter in [
        &REUSE_LOOKUPS,
        &REUSE_HITS,
        &REUSE_ROW_HITS,
        &REUSE_SAVED_NS,
        &REUSE_STORES,
        &REUSE_REFUSED,
    ] {
        counter.store(0, Relaxed);
    }
}

/// The `[root-eval-reuse]` stop receipt — printed at every lane stop, so
/// `hits=0` (armed, nothing reusable) is distinguishable from a binary that
/// lacks the mechanism (no line at all).
pub(crate) fn root_eval_reuse_stop_line() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    format!(
        "[root-eval-reuse] hits={} row_hits={} saved_ms={:.1} lookups={} stores={} refused={}",
        REUSE_HITS.load(Relaxed),
        REUSE_ROW_HITS.load(Relaxed),
        REUSE_SAVED_NS.load(Relaxed) as f64 / 1e6,
        REUSE_LOOKUPS.load(Relaxed),
        REUSE_STORES.load(Relaxed),
        REUSE_REFUSED.load(Relaxed),
    )
}

/// The same counters as `[mr-census]` fields.
pub(crate) fn root_eval_reuse_fields() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    format!(
        " reuse_hits={} reuse_row_hits={} reuse_saved_ms={:.1}",
        REUSE_HITS.load(Relaxed),
        REUSE_ROW_HITS.load(Relaxed),
        REUSE_SAVED_NS.load(Relaxed) as f64 / 1e6,
    )
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
    // #mr-census: one call per root evaluation actually computed (the L1
    // root-eval dedupe shows up here as FEWER calls, not cheaper ones).
    let _census = super::census::SiteTimer::start(super::census::Site::RootEval, adv.len());
    let mut progress = root_eval_progress::Evaluation::new(net.n_in, net.n_y, adv.len());
    let mode = eng.root.mode;
    let pack = Arc::new(build_pack_traced(eng, None, &[], Some(&progress))?);
    let stage = progress.stage("head_seed", adv.len());
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
    let gates = head_gates_with(&ybox, mode, eng.root.head_alpha.as_deref());
    let ms_all = margin_seed(&mb_all, &gates, &ybox, mode);
    root_eval_progress::Stage::finish(stage);
    let stage = progress.stage("compose_viay", adv.len());
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
    root_eval_progress::Stage::finish(stage);
    let stage = progress.stage("screen_classes", adv.len());
    let mut dj = Vec::with_capacity(adv.len());
    for (&m1, &m2v) in ms_all.m1.iter().zip(&m2v_all) {
        require_finite("root component", &[m1, m2v])?;
        dj.push(m1.max(m2v));
    }
    let mut spec_alpha_report: Option<super::spec_alpha::SpecAlphaReport> = None;
    let fail: Vec<usize> = (0..adv.len())
        .filter(|&k| !root_schedules_closed(mode, dj[k]))
        .collect();
    root_eval_progress::Stage::finish(stage);
    if !fail.is_empty() {
        let stage = progress.stage("direct_seed", fail.len());
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
        root_eval_progress::Stage::finish(stage);
        let stage = progress.stage("direct_pass", fail.len());
        let pass = eng.run_seamed(&ms_f.seed, None, super::engine::LaneDir::Lower, &seam)?;
        root_eval_progress::Stage::finish(stage);
        let stage = progress.stage("direct_concretize", fail.len());
        let direct = per_class_direct(eng, &pass, &ms_f, 0..fail.len());
        for (fi, &k) in fail.iter().enumerate() {
            require_finite("direct root bound", &[direct[fi]])?;
            dj[k] = dj[k].max(direct[fi]);
        }
        root_eval_progress::Stage::finish(stage);
        // #spec-alpha: certified passes pair classes with candidate alphas.
        // Per-class max on the published frame can only raise `dj`; the
        // refinement leaves that frame unchanged, so no rollback is needed.
        if let Some(pol) = eng.root.spec_alpha_policy {
            let stage = progress.stage("spec_alpha", fail.len());
            spec_alpha_report = Some(super::spec_alpha::refine(
                eng,
                net,
                t,
                pol,
                &gates,
                &ybox,
                &fail,
                &fail_classes,
                &mut dj,
            ));
            root_eval_progress::Stage::finish(stage);
        }
    }
    let stage = progress.stage("publish", adv.len());
    require_finite("combined root bound", &dj)?;
    let tree_classes: Vec<usize> = (0..adv.len())
        .filter(|&k| !root_schedules_closed(mode, dj[k]))
        .map(|k| adv[k])
        .collect();
    let result = RootEval {
        dj,
        tree_classes,
        pack,
        spec_alpha: spec_alpha_report,
    };
    root_eval_progress::Stage::finish(stage);
    progress.finish(result.tree_classes.len());
    Ok(result)
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

fn classwise_lru_cap(cap: usize) -> usize {
    cap.max(1)
}

fn fresh_class_lru(cap: usize, root_pack: &Arc<YPack>) -> Lru {
    // Each class owns a fresh cache: an LRU key contains only trunk splits and
    // has no net/root/input identity. The immutable full-box root pack alone
    // is shared by Arc; no class partition, MarginBatch, candidate, or bound is.
    let mut lru = Lru {
        cap: classwise_lru_cap(cap),
        entries: Vec::new(),
    };
    lru.put(Vec::new(), root_pack.clone());
    lru
}

fn classwise_tree_config(cfg: &BabConfig, remaining_expansions: usize) -> BabConfig {
    // Serial per-class trees make the one absolute/global expansion budget
    // exact. A parallel frontier may complete frontier-1 extra expansions in
    // its final batch, so classwise deliberately suppresses it.
    BabConfig {
        max_expansions: remaining_expansions,
        frontier: 1,
        frontier_contextual: 1,
        ..cfg.clone()
    }
}

#[cfg(test)]
pub(crate) fn classwise_runtime_policy_for_test(
    lru_cap: usize,
    frontier: usize,
    frontier_contextual: usize,
) -> (usize, usize, usize) {
    let cfg = BabConfig {
        lru_cap,
        frontier,
        frontier_contextual,
        ..BabConfig::default()
    };
    let cfg = classwise_tree_config(&cfg, cfg.max_expansions);
    (
        classwise_lru_cap(cfg.lru_cap),
        cfg.frontier,
        cfg.frontier_contextual,
    )
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
        frontier_execution: stats.frontier_execution,
        policy_execution: stats.policy_execution,
        epoch_execution: stats.epoch_execution,
        width_execution: stats.width_execution,
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
    aggregate
        .frontier_execution
        .absorb(record.frontier_execution);
    aggregate.policy_execution.absorb(record.policy_execution);
    aggregate.epoch_execution.absorb(record.epoch_execution);
    aggregate.width_execution.absorb(record.width_execution);
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

fn closure_epoch_trigger(configured: usize, current: usize, tier: usize) -> usize {
    if configured > 0 {
        return current;
    }
    let adaptive = super::closure::EPOCH_DEPTH_ON_ESCALATION;
    if tier >= super::closure::TIER_EPOCH {
        adaptive
    } else if current == adaptive {
        0
    } else {
        current
    }
}

#[cfg(test)]
mod closure_epoch_trigger_tests {
    use super::closure_epoch_trigger;

    #[test]
    fn adaptive_tiers_never_overwrite_configured_depth() {
        let epoch_tier = super::super::closure::TIER_EPOCH;
        assert_eq!(closure_epoch_trigger(4, 4, epoch_tier), 4);
        assert_eq!(closure_epoch_trigger(4, 4, 0), 4);
        assert_eq!(closure_epoch_trigger(4, 0, epoch_tier), 0);
        let adaptive = super::super::closure::EPOCH_DEPTH_ON_ESCALATION;
        assert_eq!(closure_epoch_trigger(0, 0, epoch_tier), adaptive);
        assert_eq!(closure_epoch_trigger(0, adaptive, 0), 0);
    }
}

/// Create one optional full-root attempt under the original deadline. The
/// normal lane calls this before root owners; standalone BaB callers retain
/// the same entry. Private carry state prevents restarting a declined attempt.
pub(super) fn direct_root_attempt(
    eng: &BackwardEngine<'_>,
    t: usize,
    adv: &[usize],
    cfg: &BabConfig,
) -> Option<super::head_group::direct::DirectAttempt> {
    let policy = cfg.head_groups?;
    if eng.root.root_precision_policy.box_source == super::RootBoxSource::PatchBackward
        || cfg.epoch_frame_route != EpochFrameRoute::Outer
        || !cfg.initial_heads.is_empty()
        || cfg.initial_ybox.is_some()
    {
        return None;
    }
    let deadline = cfg.deadline?.min(cfg.effective_publish_deadline()?);
    let mut attempt = super::head_group::direct::attempt(eng, t, adv, policy, deadline);
    let start = Instant::now();
    attempt.receipt();
    attempt.charge_compute(start.elapsed());
    Some(attempt)
}

fn direct_root_stats(cfg: &BabConfig, bounds: &[f64], elapsed: Duration) -> BabStats {
    let mut stats = BabStats {
        root_bound: bounds.iter().copied().fold(f64::INFINITY, f64::min),
        tree_classes: Vec::new(),
        root_closed_classes: bounds.len(),
        expansions: 0,
        domains_created: 1,
        closed: 0,
        max_depth: 0,
        mono_raw_dips: 0,
        mono_worst: 0.0,
        stop: "direct_head_closed_at_root".into(),
        elapsed_secs: elapsed.as_secs_f64(),
        class_runs: Vec::new(),
        epochs_attempted: 0,
        epochs_closed: 0,
        ledger_ok: None,
        frontier_execution: FrontierExecutionEvidence::joint(cfg.frontier, cfg.frontier_contextual),
        policy_execution: PolicyExecutionEvidence::from_config(cfg),
        epoch_execution: EpochExecutionEvidence::from_config(cfg),
        width_execution: WidthExecutionEvidence::from_config(cfg),
        root_precision_execution: cfg.root_precision_execution,
        backward_interm_execution: cfg.backward_interm_execution,
        root_alpha_execution: cfg.root_alpha_execution,
        close_integrity: CloseIntegrityEvidence::default(),
    };
    if close_telemetry_on(cfg) {
        for &bound in bounds {
            stats.close_integrity.note(CertKind::Root, bound);
        }
        super::close_integrity::publish_live(&stats.close_integrity);
    }
    stats
}

/// Complete direct covers enter the same conjunction/containment authorizer.
/// There is no fabricated RootEval or identity-row pack on this path.
pub(super) fn direct_root_outcome(
    eng: &BackwardEngine<'_>,
    t: usize,
    adv: &[usize],
    cfg: &BabConfig,
    hint_prefix: &[AbsLit],
    direct: Option<&mut super::head_group::direct::DirectAttempt>,
    lane_start: Instant,
) -> Option<MarginRowOutcome> {
    let direct = direct?;
    let start = Instant::now();
    let result = (|| {
        let prelude = direct.prelude.as_ref()?;
        let deadline = cfg.deadline?.min(cfg.effective_publish_deadline()?);
        if !hint_prefix.is_empty() || !prelude.complete() || !prelude.matches(eng, t, adv, deadline)
        {
            return None;
        }
        let covers: Vec<_> = adv
            .iter()
            .zip(prelude.bounds())
            .filter_map(|(&class, &b)| cert_mr::root_class(eng.root.mode, class, b))
            .collect();
        if covers.len() != adv.len() || Instant::now() >= deadline {
            return None;
        }
        let mut ledger = Ledger {
            scope: TokenScope::Instance,
            required: Some(adv.to_vec()),
            pre: covers,
            direct_guard: Some(prelude.publication_guard(deadline)),
            // B2.3: no root evaluation ran; every class is the direct
            // prepass's opaque claim.
            head_raised: adv
                .iter()
                .map(|&class| HeadRaised {
                    class,
                    route: "direct_head",
                    root_margin: None,
                })
                .collect(),
            ..Ledger::default()
        };
        let mut stats = direct_root_stats(cfg, prelude.bounds(), lane_start.elapsed());
        let token = authorize_frame_view(
            &mut ledger,
            &[],
            false,
            eng.net,
            eng.root,
            t,
            OwnedRootFrameView {
                dj: prelude.bounds(),
                ybox: prelude.ybox(),
            },
            cfg,
        );
        // The common authorizer checks the private ledger guard after both
        // containment and hint-stream work, including subsequent mixed covers.
        stats.elapsed_secs = lane_start.elapsed().as_secs_f64();
        Some(match token {
            Ok(token) => {
                stats.ledger_ok = Some(true);
                MarginRowOutcome::Unsat(stats, token)
            }
            Err(reason) => {
                stats.ledger_ok = Some(false);
                stats.stop = "direct_head_unsat_refused".into();
                MarginRowOutcome::Unknown {
                    reason,
                    stats: Some(stats),
                }
            }
        })
    })();
    direct.charge_compute(start.elapsed());
    result
}

fn merge_direct_prelude(
    eng: &BackwardEngine<'_>,
    t: usize,
    adv: &[usize],
    re: &mut RootEval,
    cfg: &BabConfig,
    direct: Option<&mut super::head_group::direct::DirectAttempt>,
) -> Option<super::head_group::direct::DirectPublicationGuard> {
    let direct = direct?;
    let start = Instant::now();
    let merged = (|| {
        let prelude = direct.prelude.as_ref()?;
        let deadline = cfg.deadline?.min(cfg.effective_publish_deadline()?);
        if re.dj.len() != adv.len()
            || re.dj.iter().any(|b| !b.is_finite())
            || !prelude.matches(eng, t, adv, deadline)
        {
            return None;
        }
        // Only the returned clone changes; the ordinary root-eval memo and
        // its genuine source pack retain their original numerical contents.
        let mut improved = false;
        for (old, &new) in re.dj.iter_mut().zip(prelude.bounds()) {
            improved |= new > *old;
            *old = old.max(new);
        }
        re.tree_classes = adv
            .iter()
            .zip(&re.dj)
            .filter_map(|(&a, &b)| (!closed(eng.root.mode, b)).then_some(a))
            .collect();
        Some(improved.then(|| prelude.publication_guard(deadline)))
    })();
    direct.charge_compute(start.elapsed());
    if merged.is_some() {
        eprintln!("[direct-head] merge=owned_original_rows memo=unchanged");
    }
    merged.flatten()
}

pub(super) fn refine_head_groups(
    eng: &BackwardEngine<'_>,
    t: usize,
    adv: &[usize],
    re: &mut RootEval,
    cfg: &BabConfig,
) {
    refine_head_groups_with_direct(eng, t, adv, re, cfg, None);
}

fn refine_head_groups_with_direct(
    eng: &BackwardEngine<'_>,
    t: usize,
    adv: &[usize],
    re: &mut RootEval,
    cfg: &BabConfig,
    direct: Option<&mut super::head_group::direct::DirectAttempt>,
) {
    let Some(policy) = cfg.head_groups else {
        return;
    };
    if cfg.epoch_frame_route != EpochFrameRoute::Outer
        || !cfg.initial_heads.is_empty()
        || cfg.initial_ybox.is_some()
    {
        eprintln!("[head-groups] armed=1 status=ineligible_frame");
        return;
    }
    let Some(deadline) = cfg.deadline else {
        eprintln!("[head-groups] armed=1 status=missing_original_deadline");
        return;
    };
    let start = Instant::now();
    let result = match direct {
        Some(direct) => direct.refine_existing(eng, &re.pack, t, adv, &re.dj, policy, deadline),
        None => {
            super::head_group::improve_root_margins(eng, &re.pack, t, adv, &re.dj, policy, deadline)
        }
    };
    let Some(report) = result else {
        eprintln!(
            "[head-groups] armed=1 status=declined elapsed_s={:.6}",
            start.elapsed().as_secs_f64()
        );
        return;
    };
    if Instant::now() >= deadline
        || report.bounds.len() != adv.len()
        || report.bounds.len() != re.dj.len()
        || report
            .bounds
            .iter()
            .zip(&re.dj)
            .any(|(&new, &old)| !new.is_finite() || !old.is_finite() || new < old)
    {
        eprintln!("[head-groups] armed=1 status=publication_refused");
        return;
    }
    let closed_before = re.dj.iter().filter(|&&b| b > 0.0).count();
    let closed_after = report.bounds.iter().filter(|&&b| b > 0.0).count();
    for candidate in &report.candidates {
        eprintln!(
            "[head-group-bound] adversary={} group={:?} phases={} lower_bits={:016x}",
            candidate.adversary,
            candidate.group,
            candidate.phases,
            candidate.lower.to_bits()
        );
    }
    eprintln!(
        "[head-groups] armed=1 status=complete groups={} phases={} improved={} closed_before={} closed_after={} native_passes={} native_columns={} native_improved_phases={} advice_calls={} advice_candidates={} advice_improved_phases={} functional_advice_calls={} functional_advice_candidates={} functional_classes_improved={} elapsed_s={:.6}",
        report.groups_checked,
        report.phases_checked,
        report.classes_improved,
        closed_before,
        closed_after,
        report.native_passes,
        report.native_columns,
        report.native_improved_phases,
        report.advice_calls,
        report.advice_candidates,
        report.advice_improved_phases,
        report.functional_advice_calls,
        report.functional_advice_candidates,
        report.functional_classes_improved,
        report.elapsed_s
    );
    re.dj = report.bounds;
    re.tree_classes = adv
        .iter()
        .zip(&re.dj)
        .filter_map(|(&class, &bound)| (bound <= 0.0).then_some(class))
        .collect();
}

impl<'a> MarginRowBab<'a> {
    /// Enumerate authentic prepared descendants on their original ledger.
    /// Runtime eligibility and the complete legacy fallback are chosen by
    /// the source entry before construction; this method mints no token.
    pub fn run_prepared(
        open: super::support_program::PreparedRootOpen,
        options: PreparedTreeOptions,
        requested: Option<Instant>,
    ) -> Result<PreparedTreeOutcome> {
        prepared_tree::run(open, options, requested)
    }

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
        Self::run_with_direct(net, root, t, adv, cfg, None)
    }

    pub(super) fn run_with_direct(
        net: &'a TwinNet,
        root: &'a RootGates,
        t: usize,
        adv: &[usize],
        cfg: BabConfig,
        direct: Option<super::head_group::direct::DirectAttempt>,
    ) -> MarginRowOutcome {
        if !root.mode.outward() {
            return MarginRowOutcome::Unknown {
                reason: "margin-row BaB requires certified outward rounding".into(),
                stats: None,
            };
        }
        let cfg = cfg.seed_outer_epoch();
        // #closure-adaptive: one instance, one ladder. The tier is a
        // process-global (the mechanism gates live four modules down and take
        // no config), so a sweep that runs several rows in one process must not
        // let row N-1's escalation pre-arm row N — that would silently turn the
        // treatment arm into "always on" and destroy the measurement.
        super::closure::reset();
        match Self::run_inner_with_direct(net, root, t, adv, cfg, Vec::new(), direct) {
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
        Self::run_classwise_with_direct(net, root, t, adv, cfg, None)
    }

    pub(super) fn run_classwise_with_direct(
        net: &'a TwinNet,
        root: &'a RootGates,
        t: usize,
        adv: &[usize],
        cfg: BabConfig,
        direct: Option<super::head_group::direct::DirectAttempt>,
    ) -> MarginRowOutcome {
        if !root.mode.outward() {
            return MarginRowOutcome::Unknown {
                reason: "classwise requires certified outward rounding".into(),
                stats: None,
            };
        }
        let cfg = cfg.seed_outer_epoch();
        // #closure-adaptive: one instance, one ladder (see `run`). The classwise
        // schedule's per-class trees deliberately SHARE it — a tier class 1
        // earned by measuring itself is exactly the knowledge class 2 should
        // start from, and re-paying the warm-up per class would spend the
        // budget on rediscovering the same fact.
        super::closure::reset();
        match Self::run_classwise_inner(net, root, t, adv, cfg, direct) {
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
        direct: Option<super::head_group::direct::DirectAttempt>,
    ) -> Result<MarginRowOutcome> {
        let t0 = Instant::now();
        if cfg.deadline.is_some_and(|dl| Instant::now() > dl) {
            return Ok(MarginRowOutcome::Unknown {
                reason: "classwise wallclock before root evaluation".into(),
                stats: None,
            });
        }
        let root_eng = BackwardEngine::new(net, root);
        let mut direct = direct.or_else(|| direct_root_attempt(&root_eng, t, adv, &cfg));
        if let Some(attempt) = direct.as_mut() {
            attempt.retain_current_frame(
                &root_eng,
                t,
                adv,
                cfg.deadline
                    .zip(cfg.effective_publish_deadline())
                    .map(|(a, b)| a.min(b)),
            );
        }
        if let Some(outcome) =
            direct_root_outcome(&root_eng, t, adv, &cfg, &[], direct.as_mut(), t0)
        {
            return Ok(outcome);
        }
        if cfg.deadline.is_some_and(|dl| Instant::now() >= dl) {
            return Ok(MarginRowOutcome::Unknown {
                reason: "classwise wallclock after direct head prepass".into(),
                stats: None,
            });
        }
        let mut re = {
            let _t = super::prof::Timer::start(super::prof::Phase::RootEval);
            // #root-eval-reuse (B1(c)): as in `run_inner`.
            root_eval_reuse(&root_eng, net, t, adv)?
        };
        // B2.3 x head routes: the bounds each head route started from.
        let dj_root = re.dj.clone();
        let direct_guard = merge_direct_prelude(&root_eng, t, adv, &mut re, &cfg, direct.as_mut());
        let dj_merged = re.dj.clone();
        refine_head_groups_with_direct(&root_eng, t, adv, &mut re, &cfg, direct.as_mut());
        let head_raised = head_raised_classes(root.mode, adv, &dj_root, &dj_merged, &re.dj);
        if cfg.deadline.is_some_and(|dl| Instant::now() > dl) {
            return Ok(MarginRowOutcome::Unknown {
                reason: "classwise wallclock after root evaluation".into(),
                stats: None,
            });
        }
        // #attack-open-disjuncts: the joint root of the top-level classwise
        // schedule (see `run_inner`). Attack scheduling only.
        if cfg.epoch_frame_route == EpochFrameRoute::Outer {
            super::root_open::publish_outer_root(t, adv, &re.dj);
        }
        // #spec-alpha YIELD -> the STRUCTURED receipt, not only stderr.
        //
        // `RootAlphaExecutionEvidence` is built in the root driver immediately
        // after `#rebake` returns, which is BEFORE `#spec-alpha` runs inside
        // `root_eval`, so it can only carry the policy echo at that point. This
        // is the first site where the refinement has run and the receipt is
        // still assemblable. Without it the yield is stderr-only and every
        // sweep through `ny benchmarks run` (stdout nulled, `RUST_LOG=error`)
        // is uninterpretable by construction -- the documented `#interm-alpha`
        // gap. `None` (dark, or no failing class) leaves the fields at their
        // "did not run" values, which is a DIFFERENT fact from a zero delta.
        let cfg = BabConfig {
            root_alpha_execution: cfg
                .root_alpha_execution
                .with_spec_alpha_yield(re.spec_alpha.as_ref()),
            ..cfg
        };
        let mut schedule = classwise_schedule(t, adv, &re.dj)?;
        // #soundness-spine M2c: every class the root closed, as a certified
        // class cover (`classwise_schedule`'s own rule: finite and `> 0`).
        let mut pre_covers: Vec<ClassCover> = adv
            .iter()
            .zip(&re.dj)
            .filter_map(|(&class, &dj)| cert_mr::root_class(root.mode, class, dj))
            .collect();
        // #frozen-dual-bab (V1 stage B): the same outer-frame tier as
        // `run_inner`; a class it certifies on the whole box leaves the
        // schedule, every other class is scheduled unchanged. Called even when
        // the root closed every class, exactly like `run_inner`, so an armed
        // run ALWAYS prints its `[frozen-dual] armed=1` summary
        // (`verdict=no_open_objective` here) — a silent armed run reads VOID.
        let mut frozen_closed = 0usize;
        // B2.3: the tier's prune traces (Proof profile only; hint data).
        let mut fd_proofs: Vec<FdProofTrace> = Vec::new();
        // B2.3: a harness's explicit arming, else the governed gate (read
        // only when no harness chose: the dark lane is the historical one
        // latched bool).
        if cfg.epoch_frame_route == EpochFrameRoute::Outer
            && match cfg.frozen_dual_tier {
                Some(armed) => armed,
                None => super::zono_state::armed(),
            }
        {
            let pending: Vec<usize> = schedule.iter().map(|(class, _)| *class).collect();
            let open_re = RootEval {
                dj: re.dj.clone(),
                tree_classes: pending,
                pack: Arc::clone(&re.pack),
                spec_alpha: None,
            };
            let mut tier = super::zono_bab::lane_tier(
                net,
                root,
                t,
                adv,
                &open_re,
                cfg.deadline,
                fd_crosscheck_every(),
                cfg.frozen_dual_tier.is_some(),
            );
            // #soundness-spine M2c: a class leaves the schedule only when its
            // closure re-derives from its own search report.
            let fd = frozen_dual_covers(&tier);
            fd_proofs = take_fd_proofs(&mut tier, &fd);
            let before = schedule.len();
            schedule.retain(|(class, _)| !fd_covered(&fd, *class));
            frozen_closed = before - schedule.len();
            pre_covers.extend(fd);
        }
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
            frontier_execution: FrontierExecutionEvidence::classwise(),
            policy_execution: PolicyExecutionEvidence::from_config(&cfg),
            epoch_execution: EpochExecutionEvidence::from_config(&cfg),
            width_execution: WidthExecutionEvidence::from_config(&cfg),
            root_precision_execution: cfg.root_precision_execution,
            backward_interm_execution: cfg.backward_interm_execution,
            root_alpha_execution: cfg.root_alpha_execution,
            close_integrity: CloseIntegrityEvidence::default(),
        };
        // #mr-close-telemetry: the classes the root closed, at their own `dj`.
        // `classwise_schedule`'s rule exactly: a class leaves the schedule iff
        // its (finite, checked above) root bound is `> 0`. Frozen-dual closes
        // were scheduled (`dj <= 0`) and are counted apart. Telemetry only.
        if close_telemetry_on(&cfg) {
            for &dj in &re.dj {
                if dj > 0.0 {
                    aggregate.close_integrity.note(CertKind::Root, dj);
                }
            }
            aggregate.close_integrity.frozen_dual_closes = frozen_closed as u64;
            super::close_integrity::publish_live(&aggregate.close_integrity);
        }
        if frozen_closed > 0 && schedule.is_empty() {
            aggregate.stop = "classwise_frozen_dual_closed".into();
        }
        if schedule.is_empty() {
            aggregate.elapsed_secs = t0.elapsed().as_secs_f64();
            if cfg.deadline.is_some_and(|dl| Instant::now() > dl) {
                aggregate.stop = "classwise_wallclock_before_root_verdict".into();
                return Ok(MarginRowOutcome::Unknown {
                    reason: "classwise wallclock before root-only verdict".into(),
                    stats: Some(aggregate),
                });
            }
            // #soundness-spine M2c: the class covers ARE the proof.
            let mut ledger = Ledger {
                scope: frame_scope(&cfg),
                required: Some(adv.to_vec()),
                pre: pre_covers,
                fd_proofs,
                direct_guard,
                head_raised,
                ..Ledger::default()
            };
            return Ok(
                match authorize_frame(&mut ledger, &[], false, net, root, t, &re, &cfg) {
                    Ok(token) => MarginRowOutcome::Unsat(aggregate, token),
                    Err(reason) => {
                        aggregate.stop = "classwise_unsat_refused".into();
                        aggregate.ledger_ok = Some(false);
                        MarginRowOutcome::Unknown {
                            reason,
                            stats: Some(aggregate),
                        }
                    }
                },
            );
        }

        // #soundness-spine M2c: each class tree's class-part token.
        let mut part_covers: Vec<ClassCover> = Vec::new();
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
                    frontier_execution: FrontierExecutionEvidence::classwise(),
                    policy_execution: PolicyExecutionEvidence::from_config(&cfg),
                    epoch_execution: EpochExecutionEvidence::from_config(&cfg),
                    width_execution: WidthExecutionEvidence::from_config(&cfg),
                    root_precision_execution: cfg.root_precision_execution,
                    backward_interm_execution: cfg.backward_interm_execution,
                    root_alpha_execution: cfg.root_alpha_execution,
                    close_integrity: CloseIntegrityEvidence::default(),
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
                    frontier_execution: FrontierExecutionEvidence::classwise(),
                    policy_execution: PolicyExecutionEvidence::from_config(&cfg),
                    epoch_execution: EpochExecutionEvidence::from_config(&cfg),
                    width_execution: WidthExecutionEvidence::from_config(&cfg),
                    root_precision_execution: cfg.root_precision_execution,
                    backward_interm_execution: cfg.backward_interm_execution,
                    root_alpha_execution: cfg.root_alpha_execution,
                    close_integrity: CloseIntegrityEvidence::default(),
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

            let class_cfg = classwise_tree_config(&cfg, remaining_expansions);
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
                frontier_execution: FrontierExecutionEvidence::classwise(),
                policy_execution: PolicyExecutionEvidence::from_config(&class_cfg),
                epoch_execution: EpochExecutionEvidence::from_config(&class_cfg),
                width_execution: WidthExecutionEvidence::from_config(&class_cfg),
                root_precision_execution: class_cfg.root_precision_execution,
                backward_interm_execution: class_cfg.backward_interm_execution,
                root_alpha_execution: class_cfg.root_alpha_execution,
                // #mr-close-telemetry: CARRIED IN — the root closes and every
                // earlier class's certificates — so this class tree's live
                // publications show the whole proof so far. Handed back to the
                // aggregate below (replaced, never absorbed twice).
                close_integrity: aggregate.close_integrity,
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
                lane_cache: super::gpu_backward::cache::PlanCache::default(),
                // #closure-adaptive: a fresh observer per class tree (each tree
                // measures its OWN closure) over the shared process-global
                // tier, so a tier class 1 earned carries into class 2.
                closure: super::closure::ClosureMonitor::new(super::closure::armed()),
                b5: stack_eval::B5Trace::default(),
            };
            // #soundness-spine M2c: one class on the whole box; the
            // instance-level aggregation below consumes the part token.
            bab.ledger.scope = TokenScope::ClassPart;
            bab.ledger.required = Some(vec![class]);
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
                MarginRowOutcome::Unsat(stats, token) => {
                    // #mr-close-telemetry: the class tree carried the running
                    // tally in; take it back (replace, never absorb twice).
                    aggregate.close_integrity = stats.close_integrity;
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
                    part_covers.push(ClassCover::Part(token));
                }
                MarginRowOutcome::Unknown { reason, stats } => {
                    let mut stats = stats.unwrap_or_else(|| bab.stats.clone());
                    // #mr-close-telemetry: as in the Unsat arm.
                    aggregate.close_integrity = stats.close_integrity;
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
        // #soundness-spine M2c: root closes + frozen-dual closes + every
        // class tree's part token must cover every adversarial class.
        let mut covers = pre_covers;
        covers.extend(part_covers);
        let mut ledger = Ledger {
            scope: frame_scope(&cfg),
            required: Some(adv.to_vec()),
            pre: covers,
            fd_proofs,
            direct_guard,
            head_raised,
            ..Ledger::default()
        };
        match authorize_frame(&mut ledger, &[], false, net, root, t, &re, &cfg) {
            Ok(token) => {
                aggregate.stop = "classwise_verified_all".into();
                aggregate.elapsed_secs = t0.elapsed().as_secs_f64();
                Ok(MarginRowOutcome::Unsat(aggregate, token))
            }
            Err(reason) => {
                aggregate.stop = "classwise_unsat_refused".into();
                aggregate.ledger_ok = Some(false);
                aggregate.elapsed_secs = t0.elapsed().as_secs_f64();
                Ok(MarginRowOutcome::Unknown {
                    reason,
                    stats: Some(aggregate),
                })
            }
        }
    }

    /// Full algorithm body. Production entry is [`Self::run`]; nested Tier-2
    /// epoch runs re-enter here directly (#epoch-bab).
    fn run_inner(
        net: &TwinNet,
        root: &RootGates,
        t: usize,
        adv: &[usize],
        cfg: BabConfig,
        hint_prefix: Vec<AbsLit>,
    ) -> Result<MarginRowOutcome> {
        Self::run_inner_with_direct(net, root, t, adv, cfg, hint_prefix, None)
    }

    fn run_inner_with_direct(
        net: &TwinNet,
        root: &RootGates,
        t: usize,
        adv: &[usize],
        cfg: BabConfig,
        hint_prefix: Vec<AbsLit>,
        direct: Option<super::head_group::direct::DirectAttempt>,
    ) -> Result<MarginRowOutcome> {
        let t0 = Instant::now();
        if cfg.deadline.is_some_and(|dl| Instant::now() > dl) {
            return Ok(MarginRowOutcome::Unknown {
                reason: "wallclock before root evaluation".into(),
                stats: None,
            });
        }
        let eng = BackwardEngine::new(net, root);
        let mut direct = direct.or_else(|| direct_root_attempt(&eng, t, adv, &cfg));
        if let Some(attempt) = direct.as_mut() {
            attempt.retain_current_frame(
                &eng,
                t,
                adv,
                cfg.deadline
                    .zip(cfg.effective_publish_deadline())
                    .map(|(a, b)| a.min(b)),
            );
        }
        if let Some(outcome) =
            direct_root_outcome(&eng, t, adv, &cfg, &hint_prefix, direct.as_mut(), t0)
        {
            return Ok(outcome);
        }
        if cfg.deadline.is_some_and(|dl| Instant::now() >= dl) {
            return Ok(MarginRowOutcome::Unknown {
                reason: "wallclock after direct head prepass".into(),
                stats: None,
            });
        }
        let mut re = {
            let _t = super::prof::Timer::start(super::prof::Phase::RootEval);
            // #root-eval-reuse (B1(c)): the alpha guard's published evaluation
            // of these exact gates, when one was stored; else unchanged.
            root_eval_reuse(&eng, net, t, adv)?
        };
        // B2.3 x head routes: the bounds each head route started from.
        let dj_root = re.dj.clone();
        let direct_guard = merge_direct_prelude(&eng, t, adv, &mut re, &cfg, direct.as_mut());
        let dj_merged = re.dj.clone();
        refine_head_groups_with_direct(&eng, t, adv, &mut re, &cfg, direct.as_mut());
        let head_raised = head_raised_classes(root.mode, adv, &dj_root, &dj_merged, &re.dj);
        if cfg.deadline.is_some_and(|dl| Instant::now() > dl) {
            return Ok(MarginRowOutcome::Unknown {
                reason: "wallclock after root evaluation".into(),
                stats: None,
            });
        }
        // #attack-open-disjuncts: publish only the OUTER frame. A nested
        // epoch run re-enters here on a ReLU-split subdomain, where a closed
        // class is NOT closed on the box. Attack scheduling only; one atomic
        // load when unarmed.
        if cfg.epoch_frame_route == EpochFrameRoute::Outer {
            super::root_open::publish_outer_root(t, adv, &re.dj);
        }
        // #spec-alpha YIELD -> the STRUCTURED receipt, not only stderr.
        //
        // `RootAlphaExecutionEvidence` is built in the root driver immediately
        // after `#rebake` returns, which is BEFORE `#spec-alpha` runs inside
        // `root_eval`, so it can only carry the policy echo at that point. This
        // is the first site where the refinement has run and the receipt is
        // still assemblable. Without it the yield is stderr-only and every
        // sweep through `ny benchmarks run` (stdout nulled, `RUST_LOG=error`)
        // is uninterpretable by construction -- the documented `#interm-alpha`
        // gap. `None` (dark, or no failing class) leaves the fields at their
        // "did not run" values, which is a DIFFERENT fact from a zero delta.
        let cfg = BabConfig {
            root_alpha_execution: cfg
                .root_alpha_execution
                .with_spec_alpha_yield(re.spec_alpha.as_ref()),
            ..cfg
        };
        // #frozen-dual-bab (V1 stage B): the certified frozen alpha-zonotope
        // dual BaB as a TIER between this root and the tree, on the OUTER frame
        // only (a nested epoch run's frame is a ReLU-split subdomain) and only
        // when `margin_row.frozen_dual_bab` is armed by the category preset or
        // an explicit override. When disabled this is one latched bool and
        // `re` is untouched. A class it returns is certified
        // `margin > 0` on the WHOLE input box (every leaf certified, level
        // ledger closed), so dropping it from the tree's conjunction is exact;
        // every other class is handed to the tree unchanged. Counted with the
        // root closures below so `root_closed_classes + tree_classes == adv`.
        let mut re = re;
        // #soundness-spine M2c: the classes the root closed, as certified class
        // covers (`root_eval`'s own rule on this frame's Outward `dj`).
        let mut pre_covers: Vec<ClassCover> = adv
            .iter()
            .zip(&re.dj)
            .filter_map(|(&class, &dj)| cert_mr::root_class(root.mode, class, dj))
            .collect();
        let mut frozen_closed = 0usize;
        // B2.3: the tier's prune traces (Proof profile only; hint data).
        let mut fd_proofs: Vec<FdProofTrace> = Vec::new();
        // B2.3: a harness's explicit arming, else the governed gate (read
        // only when no harness chose: the dark lane is the historical one
        // latched bool).
        if cfg.epoch_frame_route == EpochFrameRoute::Outer
            && match cfg.frozen_dual_tier {
                Some(armed) => armed,
                None => super::zono_state::armed(),
            }
        {
            let mut tier = super::zono_bab::lane_tier(
                net,
                root,
                t,
                adv,
                &re,
                cfg.deadline,
                fd_crosscheck_every(),
                cfg.frozen_dual_tier.is_some(),
            );
            // #soundness-spine M2c: a class leaves the tree only when its
            // closure re-derives from its own search report.
            let fd = frozen_dual_covers(&tier);
            fd_proofs = take_fd_proofs(&mut tier, &fd);
            let before = re.tree_classes.len();
            re.tree_classes.retain(|j| !fd_covered(&fd, *j));
            frozen_closed = before - re.tree_classes.len();
            pre_covers.extend(fd);
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
            stop: if frozen_closed > 0 {
                "frozen_dual_closed".into()
            } else {
                "closed_at_root".into()
            },
            elapsed_secs: 0.0,
            class_runs: Vec::new(),
            epochs_attempted: 0,
            epochs_closed: 0,
            ledger_ok: None,
            frontier_execution: FrontierExecutionEvidence::joint(
                cfg.frontier,
                cfg.frontier_contextual,
            ),
            policy_execution: PolicyExecutionEvidence::from_config(&cfg),
            epoch_execution: EpochExecutionEvidence::from_config(&cfg),
            width_execution: WidthExecutionEvidence::from_config(&cfg),
            root_precision_execution: cfg.root_precision_execution,
            backward_interm_execution: cfg.backward_interm_execution,
            root_alpha_execution: cfg.root_alpha_execution,
            close_integrity: CloseIntegrityEvidence::default(),
        };
        // #mr-close-telemetry: every class this root closed, at its own `dj` —
        // `root_eval`'s `tree_classes` rule exactly (`closed(mode, dj)`), so
        // frozen-dual closes (scheduled, `dj` open) are counted apart. On a
        // nested epoch run these become the outer tally's `epoch_root` when
        // the subtree closes. Telemetry only.
        if close_telemetry_on(&cfg) {
            for &dj in &re.dj {
                if closed(root.mode, dj) {
                    stats.close_integrity.note(CertKind::Root, dj);
                }
            }
            stats.close_integrity.frozen_dual_closes = frozen_closed as u64;
            if cfg.epoch_frame_route == EpochFrameRoute::Outer {
                super::close_integrity::publish_live(&stats.close_integrity);
            }
        }
        if re.tree_classes.is_empty() {
            stats.elapsed_secs = t0.elapsed().as_secs_f64();
            // #frozen-dual-bab: a verdict the TIER completed is gated on the
            // publish deadline exactly like the tree's own verdict
            // (`tree_loop`'s #publish-deadline check); the tier spends lane
            // wall that the closed-at-root path never did. Dark
            // (`frozen_closed == 0`) this is not evaluated: bit-identical.
            if frozen_closed > 0
                && cfg
                    .effective_publish_deadline()
                    .is_some_and(|dl| Instant::now() > dl)
            {
                stats.stop = "frozen_dual_wallclock_before_verdict".into();
                return Ok(MarginRowOutcome::Unknown {
                    reason: "wallclock before the frozen-dual verdict".into(),
                    stats: Some(stats),
                });
            }
            // #soundness-spine M2c: every class closed before any tree, so
            // the class covers ARE the proof.
            let mut ledger = Ledger {
                scope: frame_scope(&cfg),
                required: Some(adv.to_vec()),
                pre: pre_covers,
                fd_proofs,
                direct_guard,
                head_raised,
                ..Ledger::default()
            };
            ledger.hints.set_prefix(hint_prefix.clone());
            ledger.hint_prefix = hint_prefix;
            return Ok(
                match authorize_frame(&mut ledger, &[], false, net, root, t, &re, &cfg) {
                    Ok(token) => {
                        stats.ledger_ok = Some(true);
                        MarginRowOutcome::Unsat(stats, token)
                    }
                    Err(reason) => {
                        stats.ledger_ok = Some(false);
                        stats.stop = "unsat_refused".into();
                        MarginRowOutcome::Unknown {
                            reason,
                            stats: Some(stats),
                        }
                    }
                },
            );
        }
        let mb = MarginBatch::new(net, t, &re.tree_classes)?;
        let mut ledger = Ledger {
            scope: frame_scope(&cfg),
            required: Some(adv.to_vec()),
            pre: pre_covers,
            fd_proofs,
            direct_guard,
            head_raised,
            ..Ledger::default()
        };
        ledger.hints.set_prefix(hint_prefix.clone());
        ledger.hint_prefix = hint_prefix;
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
            ledger,
            nested_ledgers_ok: true,
            lane_cache: super::gpu_backward::cache::PlanCache::default(),
            closure: super::closure::ClosureMonitor::new(super::closure::armed()),
            b5: stack_eval::B5Trace::default(),
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
        // #wave-beta-bab (ENGINE_PROGRAM M6): the wave engine only when armed
        // AND on the outer frame; dark (`legacy`, the default) this is the
        // established dispatch, bit for bit.
        let outcome = if bab.cfg.wave.armed() && bab.cfg.epoch_frame_route == EpochFrameRoute::Outer
        {
            bab.tree_loop_wave(root_bound, &re)?
        } else if bab.cfg.frontier > 1 {
            bab.tree_loop_parallel(root_bound, &re)?
        } else {
            bab.tree_loop(root_bound, &re)?
        };
        // #margin-row-domain-alpha R9: one end-of-lane ledger line, so the
        // mechanism's whole cost is readable from any run without waiting for a
        // rate-limited sample. Silent (and free) when the lane is disarmed.
        super::domain_alpha::cost_summary();
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
            // #margin-row-domain-alpha: the root domain has no ancestor, so it
            // starts at the root's own (possibly `#alpha-opt`-optimized) alpha.
            alpha: Vec::new(),
            // #clip-stateful: the root accumulates nothing (no splits, no
            // halfspaces); the induction that keeps every stored interval a
            // valid enclosure bottoms out exactly here.
            clip_acc: None,
            split_order: 0,
        }
    }

    /// #soundness-spine M2c: record one closed leaf in this frame's cover
    /// ledger — its path below the frame root, built from the SAME trunk
    /// splits and head fixes its certified bound was computed under (plus the
    /// child's split for a child closed at expansion), and its evidence. A
    /// leaf whose path or evidence cannot be built is recorded as a refusal.
    ///
    /// ENGINE_PROGRAM §1.7 B1 (proof bundle v2): AFTER the ledger counted the
    /// leaf, and only while the frame's hint ledger captures, the leaf's hint
    /// (branch-order splits, aligned β, the classes it discharges, close
    /// kind, margin) is pushed — write-only, never read back.
    fn cover_leaf(
        &mut self,
        trunk: &[TrunkSplit],
        heads: &[HeadFix],
        child: Option<ChildSplit>,
        evidence: Option<LeafEvidence>,
        hint: HintCtx<'_>,
    ) {
        if cover_fault_drop_next_leaf() {
            return;
        }
        let path = cert_mr::leaf_path(trunk, heads, &self.cfg.initial_heads, child);
        let info = if self.ledger.hints.capturing() && path.is_some() {
            evidence.as_ref().map(|e| match e {
                LeafEvidence::Bound(c) => (LeafClose::Bound, c.value()),
                LeafEvidence::Empty(c) => (LeafClose::Empty, c.value()),
                LeafEvidence::Subtree(t) => (LeafClose::Subtree, t.min_margin()),
                // Graph-lane evidence (never built by this lane): margin-free,
                // exactly as `CoverLedger::records` reports it.
                LeafEvidence::LaneClose(_) => (LeafClose::Lane, f64::NAN),
            })
        } else {
            None
        };
        self.ledger.cover.leaf(path, evidence);
        let Some((close, margin)) = info else {
            return;
        };
        let Some((splits, beta)) = hint_splits(
            trunk,
            heads,
            self.cfg.initial_heads.len(),
            hint.order,
            hint.betas,
            hint.head_betas,
            child,
        ) else {
            return;
        };
        let empty_split = (close == LeafClose::Empty && hint.empty_child && child.is_some())
            .then(|| splits.len() - 1);
        let dis = self.hint_dis(hint.retired);
        self.ledger.hints.push_leaf(LeafHint {
            splits,
            beta,
            close,
            dis,
            margin,
            lift: hint.lift,
            empty_split,
        });
    }

    /// Proof bundle v2: the hint context of an evaluated domain, for the
    /// serial merge (`None` unless this frame's hint ledger captures).
    fn eval_hints(&self, st: &NodeState, lift_eval: bool) -> Option<Box<EvalHints>> {
        self.ledger.hints.capturing().then(|| {
            Box::new(EvalHints {
                betas: st.betas.clone(),
                head_betas: st.head_betas.clone(),
                retired_out: st.retired.clone(),
                lift_eval,
            })
        })
    }

    /// The tree classes still open under `retired` (column indices into the
    /// tree's margin batch). Hint data only.
    fn hint_dis(&self, retired: &[u16]) -> Vec<usize> {
        self.mb
            .adv
            .iter()
            .enumerate()
            .filter(|(c, _)| !retired.iter().any(|&r| usize::from(r) == *c))
            .map(|(_, &j)| j)
            .collect()
    }

    /// Proof bundle v2: the columns `entry`'s evaluation newly retired, as a
    /// retirement hint on `entry`'s region (scored under `betas` /
    /// `head_betas`). Write-only; a no-op unless capturing.
    fn push_retire_hint(
        &mut self,
        entry: &DomainEntry,
        betas: &[f64],
        head_betas: &[f64],
        retired_out: &[u16],
    ) {
        if !self.ledger.hints.capturing() {
            return;
        }
        let newly: Vec<usize> = retired_out
            .iter()
            .filter(|c| !entry.retired.contains(c))
            .filter_map(|&c| self.mb.adv.get(usize::from(c)).copied())
            .collect();
        if newly.is_empty() {
            return;
        }
        let Some((splits, beta)) = hint_splits(
            &entry.trunk,
            &entry.heads,
            self.cfg.initial_heads.len(),
            entry.split_order,
            betas,
            head_betas,
            None,
        ) else {
            return;
        };
        self.ledger.hints.push_retire(RetireHint {
            splits,
            beta,
            dis: newly,
            margin: f64::NAN,
        });
    }

    fn tree_loop(&mut self, root_bound: f64, re: &RootEval) -> Result<MarginRowOutcome> {
        // #phase-cost: the margin-row tree is this lane's BaB loop (a nested
        // epoch subtree on the same thread is a counted re-entry).
        let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::BabLoop);
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
        // #mr-census: tree wall/CPU on every exit of this loop (Drop), closure
        // kinds in the serial merge below. Telemetry only.
        let census_tree = super::census::TreeGuard::begin();
        let mode = self.eng.root.mode;
        let mut heap: BinaryHeap<DomainEntry> = BinaryHeap::new();
        let mut seq: u64 = 0;
        heap.push(self.root_entry(root_bound, re, seq));
        seq += 1;
        // #mr-close-telemetry: record every close's certificate beside its
        // Kraft leaf below; only the OUTERMOST tree publishes the running tally
        // (a nested epoch tree's closes reach it through `try_epoch`). Kept
        // clear of the `NY_MARGIN_ROW_TRACE` read below so that site's
        // source-frontier fingerprint (+-12 tokens) is unchanged.
        let close_telemetry = close_telemetry_on(&self.cfg);
        let publish_close = close_telemetry && census_tree.is_outermost();
        let mut stop = "queue_empty".to_string();
        let trace = std::env::var("NY_MARGIN_ROW_TRACE").is_ok();
        while let Some(entry) = {
            self.adapt_width(heap.len() + 1);
            // #closure-adaptive: observe THIS lane's own child-closure fraction
            // and escalate the closure mechanisms if the children are not
            // closing. `heap.len()` is the open frontier at this instant (the
            // domain about to be popped is still in it).
            self.closure_step(heap.len(), t0.elapsed().as_secs_f64(), &re.pack);
            heap.pop()
        } {
            if publish_close {
                super::close_integrity::publish_live(&self.stats.close_integrity);
            }
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
                census_tree.closed(super::census::CloseKind::Queued, 1 + heap.len());
                self.stats.closed += 1 + heap.len();
                self.cover_leaf(
                    &entry.trunk,
                    &entry.heads,
                    None,
                    cert_mr::bound_evidence(mode, entry.bound),
                    HintCtx::of(&entry, true),
                );
                if close_telemetry {
                    self.stats
                        .close_integrity
                        .note(CertKind::Queued, entry.bound);
                }
                for e in heap.iter() {
                    self.cover_leaf(
                        &e.trunk,
                        &e.heads,
                        None,
                        cert_mr::bound_evidence(mode, e.bound),
                        HintCtx::of(e, true),
                    );
                    if close_telemetry {
                        self.stats.close_integrity.note(CertKind::Queued, e.bound);
                    }
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
                census_tree.closed(super::census::CloseKind::Epoch, 1);
                self.stats.closed += 1;
                let subtree = self
                    .ledger
                    .pending_subtree
                    .take()
                    .map(LeafEvidence::Subtree);
                self.cover_leaf(
                    &entry.trunk,
                    &entry.heads,
                    None,
                    subtree,
                    HintCtx::of(&entry, false),
                );
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
            // The entry survived the budget/deadline, finite-bound, closed,
            // and epoch gates and is about to enter executable domain work.
            // Include the already-popped entry in the pre-decision open count.
            self.stats.frontier_execution.observe(heap.len() + 1);
            let (b_eval, mut st) = match self.eval_node(&entry)? {
                EvalOut::Infeasible(policy_execution, slack) => {
                    self.stats.policy_execution.absorb(policy_execution);
                    // Infeasible domain: provably empty, closed.
                    census_tree.closed(super::census::CloseKind::Infeasible, 1);
                    self.stats.closed += 1;
                    self.cover_leaf(
                        &entry.trunk,
                        &entry.heads,
                        None,
                        cert_mr::empty_evidence(mode, slack),
                        HintCtx::of(&entry, false),
                    );
                    if close_telemetry {
                        self.stats.close_integrity.note(CertKind::Infeasible, slack);
                    }
                    continue;
                }
                EvalOut::Deadline(policy_execution) => {
                    self.stats.policy_execution.absorb(policy_execution);
                    // #deadline-poll (D2): clean early exit, no verdict.
                    stop = "wallclock_intra_expansion".into();
                    break;
                }
                EvalOut::Node(b_eval, st) => {
                    self.stats.policy_execution.absorb(st.policy_execution);
                    (b_eval, *st)
                }
            };
            if !b_eval.is_finite() {
                return Ok(MarginRowOutcome::Unknown {
                    reason: "non-finite evaluated domain bound".into(),
                    stats: Some(self.stats.clone()),
                });
            }
            // Proof bundle v2: columns this evaluation retired, as a hint on
            // this domain's region (write-only; a no-op unless capturing).
            self.push_retire_hint(&entry, &st.betas, &st.head_betas, &st.retired);
            let b = b_eval.max(entry.bound);
            if closed(mode, b) {
                census_tree.closed(super::census::CloseKind::Eval, 1);
                self.stats.closed += 1;
                self.cover_leaf(
                    &entry.trunk,
                    &entry.heads,
                    None,
                    cert_mr::bound_evidence(mode, b),
                    HintCtx {
                        order: entry.split_order,
                        betas: &st.betas,
                        head_betas: &st.head_betas,
                        retired: &entry.retired,
                        lift: b_eval < entry.bound,
                        empty_child: false,
                    },
                );
                if close_telemetry {
                    self.stats.close_integrity.note(st.close_src, b);
                }
                continue;
            }
            // ---------- candidates ----------
            let (trunk_cands, head_cands) = self.select_candidates(&mut st, &entry);
            if head_cands.is_empty() && trunk_cands.is_empty() {
                self.stats.epoch_execution.absorb(st.epoch_execution);
                self.stats.width_execution.absorb(st.width_execution);
                stop = "no_candidates".into();
                break;
            }
            let ch = {
                let _t = super::prof::Timer::start(super::prof::Phase::ScoreCands);
                match self.score_candidates(&st, &trunk_cands, &head_cands)? {
                    Some(ch) => ch,
                    None => {
                        // #deadline-poll (D2): clean early exit, no verdict.
                        self.stats.epoch_execution.absorb(st.epoch_execution);
                        self.stats.width_execution.absorb(st.width_execution);
                        stop = "wallclock_intra_expansion".into();
                        break;
                    }
                }
            };
            let epoch_selection = st.epoch_selection;
            st.epoch_execution
                .observe_scored_selection(self.cfg.epoch_frame_route, epoch_selection);
            self.stats.epoch_execution.absorb(st.epoch_execution);
            if ch
                .iter()
                .any(|(left, right)| !(left.is_finite() && right.is_finite()))
            {
                self.stats.width_execution.absorb(st.width_execution);
                return Ok(MarginRowOutcome::Unknown {
                    reason: "non-finite child score".into(),
                    stats: Some(self.stats.clone()),
                });
            }
            st.width_execution
                .observe_successful_score(self.cfg.epoch_frame_route, st.width_selection);
            self.stats.width_execution.absorb(st.width_execution);
            // #branch-reduceop: shared pick rule (see `pick_candidate`). This is
            // the SERIAL site; the scored cifar100 lane runs the parallel ones.
            let reduce_max = super::branch_reduce_max();
            let (pick, differs) = pick_candidate(&ch, reduce_max);
            note_reduceop(reduce_max, "serial", ch.len(), pick, differs);
            // #branch-parent-lift: every scored candidate certifies a bound on
            // the PARENT, not just on its own children. Recover the best one.
            let lift = if super::parent_lift_enabled() {
                let lift = parent_lift(&ch);
                note_parent_lift("serial", ch.len(), b, lift, closed(mode, lift));
                lift
            } else {
                f64::NEG_INFINITY
            };
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
                let cb = raw.max(b).max(lift);
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
                        census_tree.closed(super::census::CloseKind::HeadInfeasible, 1);
                        self.stats.closed += 1;
                        self.stats.domains_created += 1;
                        self.cover_leaf(
                            &entry.trunk,
                            &entry.heads,
                            Some(ChildSplit::Head((i, dr))),
                            cert_mr::empty_evidence(mode, ly[i] - uy[i]),
                            HintCtx {
                                order: entry.split_order,
                                betas: &st.betas,
                                head_betas: &st.head_betas,
                                retired: &st.retired,
                                lift: false,
                                empty_child: true,
                            },
                        );
                        if close_telemetry {
                            self.stats
                                .close_integrity
                                .note(CertKind::HeadInfeasible, ly[i] - uy[i]);
                        }
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
                    let raw_closes = closed(mode, raw);
                    census_tree.closed(
                        if raw_closes {
                            super::census::CloseKind::Raw
                        } else {
                            super::census::CloseKind::Lift
                        },
                        1,
                    );
                    self.stats.closed += 1;
                    self.cover_leaf(
                        &trunk_c,
                        &heads_c,
                        None,
                        cert_mr::bound_evidence(mode, cb),
                        HintCtx {
                            order: child_split_order(
                                &entry,
                                self.cfg.initial_heads.len(),
                                pick >= n_t,
                            ),
                            betas: &betas_c,
                            head_betas: &head_betas_c,
                            retired: &st.retired,
                            lift: !raw_closes,
                            empty_child: false,
                        },
                    );
                    if close_telemetry {
                        self.stats.close_integrity.note(
                            if raw_closes {
                                CertKind::Raw
                            } else {
                                CertKind::Lift
                            },
                            cb,
                        );
                    }
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
                        // #margin-row-domain-alpha: warm start (see
                        // `DomainEntry::alpha`).
                        alpha: st.alpha.clone(),
                        // #clip-stateful: monotone inheritance (see
                        // `DomainEntry::clip_acc`).
                        clip_acc: st.clip_acc.clone(),
                        split_order: child_split_order(
                            &entry,
                            self.cfg.initial_heads.len(),
                            pick >= n_t,
                        ),
                    });
                    seq += 1;
                }
            }
            self.stats.expansions += 1;
            // Both child slots were processed (a non-finite child returns
            // before this line), and each incremented `domains_created`.
            census_tree.expanded(2);
            // #mr-close-telemetry: columns this OPEN domain retired are
            // certified here for its whole subtree (its leaves certify only
            // the survivors).
            if close_telemetry {
                self.stats
                    .close_integrity
                    .note_tally(CertKind::Retire, &st.retire_certs);
            }
        }
        self.stats.stop = stop.clone();
        self.stats.elapsed_secs = t0.elapsed().as_secs_f64();
        // #router-r1: the tree's final progress sample (write-only).
        self.router_tap(heap.len(), true);
        // #phase-cost: the tree's work denominators for the run's cost census.
        note_phase_cost_tree(&self.stats);
        // R9: the armed policy always leaves a receipt, so a run that observed
        // nothing is distinguishable from a run where the policy never loaded.
        self.closure.finish(
            self.stats.expansions,
            self.stats.domains_created,
            self.stats.closed,
            &stop,
        );
        clip_head_finish();
        let mut verified = matches!(stop.as_str(), "verified_bestfirst" | "queue_empty");
        // #publish-deadline: the tree has ALREADY finished with a verdict-bearing
        // stop here, so gate publication on the publish deadline rather than on
        // the search deadline's 3 s wrap-up reserve.
        if verified
            && self
                .cfg
                .effective_publish_deadline()
                .is_some_and(|dl| Instant::now() > dl)
        {
            stop = "wallclock_before_verdict".into();
            self.stats.stop.clone_from(&stop);
            verified = false;
        }
        if verified {
            // #soundness-spine M2c: THE gate. The path-based cover, the class
            // aggregation and (instance scope) the G2 guard, in BOTH proof
            // profiles; a refusal is `Unknown`, never a verdict.
            match authorize_frame(
                &mut self.ledger,
                &self.stats.tree_classes,
                true,
                self.eng.net,
                self.eng.root,
                self.mb.t,
                re,
                &self.cfg,
            ) {
                Ok(token) => {
                    self.stats.ledger_ok = Some(true);
                    Ok(MarginRowOutcome::Unsat(self.stats.clone(), token))
                }
                Err(reason) => {
                    self.stats.ledger_ok = Some(false);
                    self.stats.stop = "unsat_refused".into();
                    Ok(MarginRowOutcome::Unknown {
                        reason,
                        stats: Some(self.stats.clone()),
                    })
                }
            }
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
    /// Called once per batch from the serial section. Both the configured and
    /// exact lower-authority ratchets advance from the same frontier fact, so
    /// selector counterfactuals never reopen process-global state.
    fn adapt_width(&mut self, open: usize) {
        let esc: usize = if open >= 256 {
            32
        } else if open >= 32 {
            16
        } else {
            0
        };
        if esc == 0 {
            return;
        }

        let ordinary_route = ordinary_width_selectors_enabled(
            self.cfg.epoch_policy.tier0_exact,
            self.cfg.retained.is_some(),
        );
        let (old_h, old_t) = (self.cfg.effective_k_head, self.cfg.effective_k_trunk);
        let (old_contextual_h, old_contextual_t) = (
            self.cfg.effective_contextual_k_head,
            self.cfg.effective_contextual_k_trunk,
        );
        if self.cfg.adaptive {
            self.cfg.effective_k_head = self.cfg.effective_k_head.max(esc);
            self.cfg.effective_k_trunk = self.cfg.effective_k_trunk.max(esc);
        }
        if self.cfg.contextual_adaptive {
            self.cfg.effective_contextual_k_head = self.cfg.effective_contextual_k_head.max(esc);
            self.cfg.effective_contextual_k_trunk = self.cfg.effective_contextual_k_trunk.max(esc);
        }
        let escalated = (self.cfg.effective_k_head, self.cfg.effective_k_trunk) != (old_h, old_t);
        self.stats.width_execution.observe_adaptive_threshold(
            ordinary_route,
            (old_h, old_t),
            (self.cfg.effective_k_head, self.cfg.effective_k_trunk),
            (old_contextual_h, old_contextual_t),
            (
                self.cfg.effective_contextual_k_head,
                self.cfg.effective_contextual_k_trunk,
            ),
        );
        if escalated {
            eprintln!(
                "[k-adapt] frontier open={open} -> k_head {old_h}->{} k_trunk {old_t}->{} at exp {}",
                self.cfg.effective_k_head,
                self.cfg.effective_k_trunk,
                self.stats.expansions
            );
        }
    }

    /// #closure-adaptive: one observation of THIS tree's child-closure
    /// fraction, and the escalation it implies.
    ///
    /// Called from the SERIAL section of either tree loop (never from a rayon
    /// worker), so a parallel frontier batch always runs at one tier for its
    /// whole batch and the deterministic merge is untouched. The policy writes
    /// nothing but an integer tier; it has no path to a published bound.
    fn closure_step(&mut self, open: usize, elapsed: f64, root_pack: &Arc<YPack>) {
        self.router_tap(open, false);
        let remaining = self
            .cfg
            .deadline
            .map(|dl| dl.saturating_duration_since(Instant::now()).as_secs_f64());
        let decision = self.closure.observe(
            self.stats.expansions,
            self.stats.domains_created,
            self.stats.closed,
            open,
            elapsed,
            remaining,
        );
        match decision {
            super::closure::Decision::Escalate(t) | super::closure::Decision::Rollback(t) => {
                self.set_epoch_tier(t);
                self.retier_packs(t, root_pack);
            }
            super::closure::Decision::Idle | super::closure::Decision::Hold => {}
        }
    }

    /// #router-r1: the margin-row tree's WRITE-ONLY progress tap
    /// (ROUTER_PROGRAM §11.1 file 23). Called from the SERIAL section of both
    /// tree loops (via `closure_step`) and once at `TREE DONE` (`force`). It
    /// pushes plain counters to `ny_core::router::bus`, which nothing on the
    /// verdict path reads back; while the router is off it is one relaxed
    /// atomic load. One sample per `FRONTIER_WINDOW_UNITS` expansions (the
    /// shipped closure window the FAR evidence was read in).
    fn router_tap(&self, open: usize, force: bool) {
        static CADENCE: ny_core::router::bus::Stride = ny_core::router::bus::Stride::new();
        let expansions = self.stats.expansions as u64;
        // Same evaluation order as before: the cadence stride is consulted
        // (and advanced) only when the sample is not forced.
        let forced = force && ny_core::router::bus::enabled();
        if !(forced || CADENCE.due(expansions, ny_core::router::project::FRONTIER_WINDOW_UNITS)) {
            return;
        }
        ny_core::router::bus::emit(ny_core::router::bus::ProgressSample::frontier(
            ny_core::router::decl::ids::MARGIN_ROW,
            expansions,
            self.stats.domains_created as u64,
            self.stats.closed as u64,
            open as u64,
        ));
    }

    /// #closure-adaptive top tier: the epoch rebuild is carried in `BabConfig`
    /// rather than behind a process-global gate, so the tier is applied here.
    /// The configured cadence remains immutable in `epoch_policy`; this method
    /// owns only the closure controller's operational cadence. Consequently a
    /// rollback can never disarm an explicit scored `NY_EPOCH_DEPTH`, including
    /// the formerly ambiguous case where it equals the adaptive depth.
    fn set_epoch_tier(&mut self, tier: usize) {
        let before = self.cfg.epoch_trigger_depth;
        let after = closure_epoch_trigger(self.cfg.epoch_policy.depth, before, tier);
        if after == before {
            return;
        }
        self.cfg.epoch_trigger_depth = after;
        if after > 0 {
            eprintln!(
                "[closure] tier={tier} epoch rebuilds ARMED at trunk depth >= {after} \
                 (max {} attempts)",
                self.cfg.epoch_attempt_cap
            );
        } else {
            eprintln!("[closure] tier={tier} epoch rebuilds disarmed");
        }
    }

    /// Drop y-packs built under a different tier.
    ///
    /// NOT a soundness requirement — a pack built at a lower tier is merely
    /// LOOSER, and the certified pass rescores every domain that uses it. It is
    /// an ENGAGEMENT requirement: without this the domains that triggered the
    /// escalation are exactly the ones that would keep hitting pre-escalation
    /// packs, and the lever would read as inert for reasons that have nothing
    /// to do with whether it works.
    ///
    /// The empty-trunk root pack is tier-invariant (no splits ⇒ no halfspaces ⇒
    /// every clip site returns early), so it is RE-SEEDED from the root
    /// evaluation rather than rebuilt — which also keeps `build_pack(.., None,
    /// ..)` the single producer of the root pack.
    fn retier_packs(&mut self, tier: usize, root_pack: &Arc<YPack>) {
        let dropped = self.lru.entries.len();
        self.lru.entries.clear();
        self.lru.put(Vec::new(), root_pack.clone());
        eprintln!(
            "[closure] tier={tier} ({}) y-pack cache retiered: {dropped} entries dropped, \
             root pack re-seeded",
            super::closure::tier_name(tier),
        );
    }

    fn tree_loop_parallel(&mut self, root_bound: f64, re: &RootEval) -> Result<MarginRowOutcome> {
        // #phase-cost: the margin-row tree is this lane's BaB loop (a nested
        // epoch subtree on the same thread is a counted re-entry).
        let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::BabLoop);
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
        // #mr-census: tree wall/CPU on every exit of this loop (Drop), batch
        // straggler timing and closure kinds in the serial merge below.
        // Telemetry only.
        let census_tree = super::census::TreeGuard::begin();
        let mode = self.eng.root.mode;
        let width = self.cfg.frontier.max(1);
        let mut heap: BinaryHeap<DomainEntry> = BinaryHeap::new();
        let mut seq: u64 = 0;
        heap.push(self.root_entry(root_bound, re, seq));
        seq += 1;
        // #mr-close-telemetry: record every close's certificate beside its
        // Kraft leaf in the serial fill stage and merge below; only the
        // OUTERMOST tree publishes the running tally, once per batch. Kept
        // clear of the `NY_MARGIN_ROW_TRACE` read below so that site's
        // source-frontier fingerprint (+-12 tokens) is unchanged.
        let close_telemetry = close_telemetry_on(&self.cfg);
        let publish_close = close_telemetry && census_tree.is_outermost();
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
            // #closure-adaptive: same observation as the serial lane, in the
            // serial section so the whole batch runs at one tier.
            self.closure_step(heap.len(), t0.elapsed().as_secs_f64(), &re.pack);
            // #mr-close-telemetry: publish the tally merged so far (outermost
            // tree only), so a SNAPSHOT of a lane that never returns is at
            // most one batch stale. Telemetry only.
            if publish_close {
                super::close_integrity::publish_live(&self.stats.close_integrity);
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
                    census_tree.closed(super::census::CloseKind::Queued, 1);
                    self.b5.fold_close(stack_eval::TRAJ_QUEUED, &entry);
                    self.stats.closed += 1;
                    self.cover_leaf(
                        &entry.trunk,
                        &entry.heads,
                        None,
                        cert_mr::bound_evidence(mode, entry.bound),
                        HintCtx::of(&entry, true),
                    );
                    if close_telemetry {
                        self.stats
                            .close_integrity
                            .note(CertKind::Queued, entry.bound);
                    }
                    continue;
                }
                let depth = entry.trunk.len() + entry.heads.len();
                self.stats.max_depth = self.stats.max_depth.max(depth);
                // Tier-2 epoch attempt (#epoch-bab): handled in this serial
                // fill stage (the rebuild is internally rayon-parallel and
                // needs `&mut self` for the attempt accounting).
                if self.try_epoch(&entry) {
                    census_tree.closed(super::census::CloseKind::Epoch, 1);
                    self.b5.fold_close(stack_eval::TRAJ_EPOCH, &entry);
                    self.stats.closed += 1;
                    let subtree = self
                        .ledger
                        .pending_subtree
                        .take()
                        .map(LeafEvidence::Subtree);
                    self.cover_leaf(
                        &entry.trunk,
                        &entry.heads,
                        None,
                        subtree,
                        HintCtx::of(&entry, false),
                    );
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
            // At least one domain survived the defensive closed/epoch gates,
            // so this planned frontier batch reached executable tree work.
            // Exclude entries already discharged by those gates; retain the
            // selected batch plus the still-open, uninspected heap.
            self.stats
                .frontier_execution
                .observe(batch.len() + heap.len());
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
            if self.cfg.parallel_prebuild && batch.len() > 1 {
                let mut missing: Vec<Vec<TrunkSplit>> = Vec::new();
                for e in &batch {
                    if self.lru.get(&e.trunk).is_none()
                        && !missing.iter().any(|m| m.as_slice() == e.trunk.as_slice())
                    {
                        missing.push(e.trunk.clone());
                    }
                }
                if !missing.is_empty() {
                    self.stats.policy_execution.parallel_prebuild_batches = self
                        .stats
                        .policy_execution
                        .parallel_prebuild_batches
                        .saturating_add(1);
                    self.stats.policy_execution.parallel_prebuild_domains = self
                        .stats
                        .policy_execution
                        .parallel_prebuild_domains
                        .saturating_add(missing.len());
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
            // exactly. Gate ON: stack the domains' MAIN eval passes (#mr-stack,
            // B5) and their candidate-score rows, keep selection and the β /
            // domain-alpha trials per domain, then restore batch order before
            // the same deterministic serial heap merge below.
            let batch_t0 = Instant::now();
            let steps: Vec<Result<ExpandStep>> = if self.cfg.domain_stack && batch.len() > 1 {
                // #domain-stack R9: the stacked and independent paths are
                // bit-identical by construction, so WITHOUT this line an armed
                // run and a dark run are indistinguishable in the log — which
                // makes any wall-clock A/B of this lever VOID rather than
                // negative. Emit once, on the first stacked expansion.
                // #domain-stack R9. Report the WIDEST stack seen, not the
                // first: the first stacked expansion happens when the heap has
                // barely filled, so it is necessarily the narrowest one this
                // run will ever do. Latching on it systematically understates
                // the mechanism — measured live, the first stack was 2 domains
                // on both cifar100 and tinyimagenet, which says nothing about
                // whether the column-scaling benefit was reached.
                {
                    use std::sync::atomic::Ordering::Relaxed;
                    let w = batch.len();
                    DOMAIN_STACK_EXPANSIONS.fetch_add(1, Relaxed);
                    let prev = DOMAIN_STACK_MAX_WIDTH.fetch_max(w, Relaxed);
                    if w > prev {
                        eprintln!(
                            "[domain-stack] armed: widest stack so far domains={w} \
                             (one wide certified pass replaces {w} independent ones; \
                             the pass is linear above ~64 columns)"
                        );
                    }
                }
                let (stacked_steps, stack_report) =
                    self.expand_domain_stacked_batch(&batch, &packs)?;
                // #mr-stack: this batch's main-eval stacking receipt (per tree
                // and process-wide; printed as `[mr-stack]` at every stop).
                self.b5.note_batch(stack_report);
                let stacked: Vec<Result<ExpandStep>> = stacked_steps.into_iter().map(Ok).collect();
                census_tree.stacked_batch(super::census::elapsed_ns(batch_t0));
                stacked
            } else {
                // Concurrent expansion (each domain's bound is bit-identical
                // to serial; see method docs). `this` is a shared &Self across
                // workers.
                let this: &Self = self;
                // #mr-census: each domain's own evaluate+expand wall, so the
                // batch barrier's straggler wait is measurable. The expansion
                // call itself is unchanged.
                let timed: Vec<(Result<ExpandStep>, u64)> = batch
                    .par_iter()
                    .zip(packs.par_iter())
                    .map(|(entry, pack)| {
                        let t_dom = Instant::now();
                        let step = this.expand_one(pack, entry);
                        (step, super::census::elapsed_ns(t_dom))
                    })
                    .collect();
                let (steps, domain_walls): (Vec<Result<ExpandStep>>, Vec<u64>) =
                    timed.into_iter().unzip();
                census_tree.batch(super::census::elapsed_ns(batch_t0), &domain_walls);
                steps
            };
            super::prof::bump(super::prof::Counter::FrontierBatch, 1);
            super::prof::bump(super::prof::Counter::FrontierPopped, batch.len() as u64);
            // Deterministic serial merge in pop order.
            for (step, entry) in steps.into_iter().zip(&batch) {
                let step = step?;
                // #mr-stack: fold every merged decision — its kind, the
                // popped domain, and each child's bound and state bits — into
                // the tree's trajectory hash (telemetry only).
                self.b5.fold_step(&step, entry);
                self.stats.policy_execution.absorb(step.policy_execution());
                self.stats.epoch_execution.absorb(step.epoch_execution());
                self.stats.width_execution.absorb(step.width_execution());
                match step {
                    ExpandStep::Infeasible(_, cert) => {
                        census_tree.closed(super::census::CloseKind::Infeasible, 1);
                        self.stats.closed += 1;
                        self.cover_leaf(
                            &entry.trunk,
                            &entry.heads,
                            None,
                            cert_mr::empty_evidence(mode, cert.value),
                            HintCtx::of(entry, false),
                        );
                        if close_telemetry {
                            self.stats.close_integrity.note(cert.kind, cert.value);
                        }
                    }
                    ExpandStep::ClosedAtEval(_, cert, hints) => {
                        census_tree.closed(super::census::CloseKind::Eval, 1);
                        self.stats.closed += 1;
                        let ctx = match hints.as_deref() {
                            Some(h) => {
                                self.push_retire_hint(
                                    entry,
                                    &h.betas,
                                    &h.head_betas,
                                    &h.retired_out,
                                );
                                HintCtx {
                                    order: entry.split_order,
                                    betas: &h.betas,
                                    head_betas: &h.head_betas,
                                    retired: &entry.retired,
                                    lift: h.lift_eval,
                                    empty_child: false,
                                }
                            }
                            None => HintCtx::of(entry, false),
                        };
                        self.cover_leaf(
                            &entry.trunk,
                            &entry.heads,
                            None,
                            cert_mr::bound_evidence(mode, cert.value),
                            ctx,
                        );
                        if close_telemetry {
                            self.stats.close_integrity.note(cert.kind, cert.value);
                        }
                    }
                    ExpandStep::DeadlineHit(_) => {
                        // #deadline-poll (D2): clean early exit, no verdict.
                        // The domain is neither closed nor counted; results
                        // merged before this one stand, the rest are dropped
                        // (the end-of-run deadline guard would discard any
                        // late verdict anyway).
                        stop = "wallclock_intra_expansion".into();
                        break 'outer;
                    }
                    ExpandStep::NoCandidates(_) => {
                        stop = "no_candidates".into();
                        break 'outer;
                    }
                    ExpandStep::NonFiniteBound(_) => {
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
                        close_kinds,
                        closed_leaves,
                        execution: _,
                        hints,
                    } => {
                        census_tree.closed(super::census::CloseKind::Raw, close_kinds.raw);
                        census_tree.closed(super::census::CloseKind::Lift, close_kinds.lift);
                        census_tree.closed(
                            super::census::CloseKind::HeadInfeasible,
                            close_kinds.head_infeasible,
                        );
                        // #mr-close-telemetry: the closed children's
                        // certificates and the parent's retirements.
                        if close_telemetry {
                            for cert in close_kinds.certs.iter().flatten() {
                                self.stats.close_integrity.note(cert.kind, cert.value);
                            }
                            self.stats
                                .close_integrity
                                .note_tally(CertKind::Retire, &close_kinds.retire);
                        }
                        census_tree.expanded(created_children);
                        self.stats.mono_raw_dips += dips;
                        self.stats.mono_worst = self.stats.mono_worst.min(worst_dip);
                        self.stats.domains_created += created_children;
                        self.stats.closed += closed_children;
                        if let Some(h) = hints.as_deref() {
                            self.push_retire_hint(entry, &h.betas, &h.head_betas, &h.retired_out);
                        }
                        if closed_leaves.len() == closed_children {
                            for leaf in closed_leaves {
                                let empty_child =
                                    matches!(leaf.evidence, Some(LeafEvidence::Empty(_)));
                                let ctx = match hints.as_deref() {
                                    Some(h) => HintCtx {
                                        order: entry.split_order,
                                        betas: &h.betas,
                                        head_betas: &h.head_betas,
                                        retired: &h.retired_out,
                                        lift: leaf.lift,
                                        empty_child,
                                    },
                                    None => HintCtx::of(entry, leaf.lift),
                                };
                                self.cover_leaf(
                                    &entry.trunk,
                                    &entry.heads,
                                    Some(leaf.split),
                                    leaf.evidence,
                                    ctx,
                                );
                            }
                        } else {
                            // A close count without its paths is a bookkeeping
                            // defect: record it so the cover gate refuses.
                            self.ledger.cover.leaf(None, None);
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
                                alpha: c.alpha,
                                clip_acc: c.clip_acc,
                                split_order: c.split_order,
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
        // #mr-stack: publish the OUTERMOST tree's trajectory hash and
        // main-eval stacking receipt (`[mr-stack]` at every lane stop).
        if census_tree.is_outermost() {
            self.b5.publish();
        }
        // #router-r1: the tree's final progress sample (write-only).
        self.router_tap(heap.len(), true);
        // #phase-cost: the tree's work denominators for the run's cost census.
        note_phase_cost_tree(&self.stats);
        // R9: the armed policy always leaves a receipt, so a run that observed
        // nothing is distinguishable from a run where the policy never loaded.
        self.closure.finish(
            self.stats.expansions,
            self.stats.domains_created,
            self.stats.closed,
            &stop,
        );
        clip_head_finish();
        let mut verified = matches!(stop.as_str(), "verified_bestfirst" | "queue_empty");
        // #publish-deadline: the tree has ALREADY finished with a verdict-bearing
        // stop here, so gate publication on the publish deadline rather than on
        // the search deadline's 3 s wrap-up reserve.
        if verified
            && self
                .cfg
                .effective_publish_deadline()
                .is_some_and(|dl| Instant::now() > dl)
        {
            stop = "wallclock_before_verdict".into();
            self.stats.stop.clone_from(&stop);
            verified = false;
        }
        if verified {
            // #soundness-spine M2c: THE gate. The path-based cover, the class
            // aggregation and (instance scope) the G2 guard, in BOTH proof
            // profiles; a refusal is `Unknown`, never a verdict.
            match authorize_frame(
                &mut self.ledger,
                &self.stats.tree_classes,
                true,
                self.eng.net,
                self.eng.root,
                self.mb.t,
                re,
                &self.cfg,
            ) {
                Ok(token) => {
                    self.stats.ledger_ok = Some(true);
                    Ok(MarginRowOutcome::Unsat(self.stats.clone(), token))
                }
                Err(reason) => {
                    self.stats.ledger_ok = Some(false);
                    self.stats.stop = "unsat_refused".into();
                    Ok(MarginRowOutcome::Unknown {
                        reason,
                        stats: Some(self.stats.clone()),
                    })
                }
            }
        } else {
            Ok(MarginRowOutcome::Unknown {
                reason: stop,
                stats: Some(self.stats.clone()),
            })
        }
    }

    /// Gate-on parallel-frontier expansion. Candidate selection remains
    /// per-domain and rayon-parallel. Two cross-domain engine passes serve the
    /// whole batch: the domains' MAIN eval passes (#mr-stack, throughput plan
    /// B5 — see `bab/stack_eval.rs`; β / domain-alpha trials stay per domain)
    /// and the ready domains' candidate-score rows. Results are returned in the
    /// original pop order for the established serial heap merge, with the
    /// batch's main-eval stacking receipt.
    fn expand_domain_stacked_batch(
        &self,
        batch: &[DomainEntry],
        packs: &[Arc<YPack>],
    ) -> Result<(Vec<ExpandStep>, stack_eval::StackReport)> {
        if batch.len() != packs.len() {
            return Err(NyError::InvalidSpec(
                "margin_row: domain-stack batch/pack length mismatch".into(),
            ));
        }
        // Each slot is exactly the `Result` the per-domain `eval_with_pack`
        // returns for that domain, so the first error in pop order below is
        // the one the unstaged path surfaced.
        let (evals, stack_report) = self.eval_batch_stacked(batch, packs);
        let preps: Vec<Result<PreparedExpand>> = evals
            .into_par_iter()
            .zip(batch.par_iter())
            .map(|(eval, entry)| eval.and_then(|eval| self.prepare_from_eval(entry, eval)))
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
                PreparedExpand::Done(step) => out.push(*step),
                PreparedExpand::Ready(ready) => match scores[i].take() {
                    Some(score) => out.push(self.finish_prepared_expand(entry, &ready, &score)),
                    None if score_deadline => {
                        out.push(ExpandStep::DeadlineHit(
                            DomainExecutionEvidence::from_state(&ready.st),
                        ));
                    }
                    None => {
                        return Err(NyError::InvalidSpec(
                            "margin_row: missing domain-stack candidate score".into(),
                        ))
                    }
                },
            }
        }
        Ok((out, stack_report))
    }

    /// The post-eval half of a gate-on expansion: the finite / closed /
    /// no-candidate exits and candidate selection, from one domain's
    /// [`EvalOut`] (however its main pass was dispatched).
    fn prepare_from_eval(&self, entry: &DomainEntry, eval: EvalOut) -> Result<PreparedExpand> {
        let mode = self.eng.root.mode;
        let (b_eval, mut st) = match eval {
            EvalOut::Infeasible(policy_execution, slack) => {
                return Ok(PreparedExpand::Done(Box::new(ExpandStep::Infeasible(
                    DomainExecutionEvidence::before_selection(policy_execution, &self.cfg),
                    Cert {
                        kind: CertKind::Infeasible,
                        value: slack,
                    },
                ))))
            }
            EvalOut::Deadline(policy_execution) => {
                return Ok(PreparedExpand::Done(Box::new(ExpandStep::DeadlineHit(
                    DomainExecutionEvidence::before_selection(policy_execution, &self.cfg),
                ))))
            }
            EvalOut::Node(b_eval, st) => (b_eval, *st),
        };
        if !(b_eval.is_finite() && entry.bound.is_finite()) {
            return Ok(PreparedExpand::Done(Box::new(ExpandStep::NonFiniteBound(
                DomainExecutionEvidence::from_state(&st),
            ))));
        }
        let b = b_eval.max(entry.bound);
        if closed(mode, b) {
            let hints = self.eval_hints(&st, b_eval < entry.bound);
            return Ok(PreparedExpand::Done(Box::new(ExpandStep::ClosedAtEval(
                DomainExecutionEvidence::from_state(&st),
                Cert {
                    kind: st.close_src,
                    value: b,
                },
                hints,
            ))));
        }
        let (trunk_cands, head_cands) = self.select_candidates(&mut st, entry);
        if head_cands.is_empty() && trunk_cands.is_empty() {
            return Ok(PreparedExpand::Done(Box::new(ExpandStep::NoCandidates(
                DomainExecutionEvidence::from_state(&st),
            ))));
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
        let mut execution = DomainExecutionEvidence::from_state(&ready.st);
        execution
            .epoch
            .observe_scored_selection(self.cfg.epoch_frame_route, ready.st.epoch_selection);
        if ch.len() != ready.trunk_cands.len() + ready.head_cands.len()
            || ch
                .iter()
                .any(|(left, right)| !(left.is_finite() && right.is_finite()))
        {
            return ExpandStep::NonFiniteBound(execution);
        }
        execution
            .width
            .observe_successful_score(self.cfg.epoch_frame_route, ready.st.width_selection);
        // #branch-reduceop: shared pick rule (see `pick_candidate`). THIS is a
        // scored-lane site — the concurrent cifar100 lane runs the parallel
        // expansion paths, not the serial loop.
        let reduce_max = super::branch_reduce_max();
        let (pick, differs) = pick_candidate(ch, reduce_max);
        note_reduceop(reduce_max, "parallel", ch.len(), pick, differs);
        // #branch-parent-lift: see `parent_lift`.
        let lift = if super::parent_lift_enabled() {
            let lift = parent_lift(ch);
            note_parent_lift(
                "parallel",
                ch.len(),
                ready.b,
                lift,
                closed(self.eng.root.mode, lift),
            );
            lift
        } else {
            f64::NEG_INFINITY
        };
        let n_t = ready.trunk_cands.len();
        let chb = ch[pick];
        let mut pushes = Vec::new();
        let mut dips = 0usize;
        let mut worst_dip = 0.0f64;
        let mut closed_children = 0usize;
        let mut created_children = 0usize;
        let mut close_kinds = ChildCloseKinds::default();
        let mut closed_leaves: Vec<ClosedLeaf> = Vec::new();
        let mode = self.eng.root.mode;
        for (d_i, dr) in [(0usize, 1i8), (1usize, -1i8)] {
            let raw = if d_i == 0 { chb.0 } else { chb.1 };
            if !raw.is_finite() {
                return ExpandStep::NonFiniteBound(execution);
            }
            // #soundness-spine M2c: the split this child slot carries.
            let split = if pick < n_t {
                let (li, pos) = ready.trunk_cands[pick];
                ChildSplit::Trunk((li, pos, dr))
            } else {
                ChildSplit::Head((ready.head_cands[pick - n_t], dr))
            };
            if raw < ready.b - 1e-9 {
                dips += 1;
                worst_dip = worst_dip.min(raw - ready.b);
            }
            let cb = raw.max(ready.b).max(lift);
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
                    close_kinds.head_infeasible += 1;
                    if close_telemetry_on(&self.cfg) {
                        close_kinds.note_cert(CertKind::HeadInfeasible, ly[i] - uy[i]);
                    }
                    closed_leaves.push(ClosedLeaf {
                        split,
                        evidence: cert_mr::empty_evidence(mode, ly[i] - uy[i]),
                        lift: false,
                    });
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
                closed_leaves.push(ClosedLeaf {
                    split,
                    evidence: cert_mr::bound_evidence(mode, cb),
                    lift: !closed(mode, raw),
                });
                let raw_closes = closed(self.eng.root.mode, raw);
                close_kinds.note_scored(raw_closes);
                if close_telemetry_on(&self.cfg) {
                    close_kinds.note_cert(
                        if raw_closes {
                            CertKind::Raw
                        } else {
                            CertKind::Lift
                        },
                        cb,
                    );
                }
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
                    // #margin-row-domain-alpha: warm start (see
                    // `DomainEntry::alpha`).
                    alpha: ready.st.alpha.clone(),
                    // #clip-stateful: monotone inheritance.
                    clip_acc: ready.st.clip_acc.clone(),
                    split_order: child_split_order(
                        entry,
                        self.cfg.initial_heads.len(),
                        pick >= n_t,
                    ),
                });
            }
        }
        // #mr-close-telemetry: this OPEN parent's retirements (telemetry only).
        if close_telemetry_on(&self.cfg) {
            close_kinds.retire = ready.st.retire_certs;
        }
        ExpandStep::Expanded {
            dips,
            worst_dip,
            pushes,
            closed_children,
            created_children,
            close_kinds,
            closed_leaves,
            execution: Box::new(execution),
            hints: self.eval_hints(&ready.st, false),
        }
    }

    /// Evaluate+expand one popped domain given a PREBUILT pack (pure, `&self`).
    /// Mirrors the serial expansion body exactly; see [`Self::tree_loop`].
    fn expand_one(&self, pack: &Arc<YPack>, entry: &DomainEntry) -> Result<ExpandStep> {
        let mode = self.eng.root.mode;
        let (b_eval, mut st) = match self.eval_with_pack(pack, entry)? {
            EvalOut::Infeasible(policy_execution, slack) => {
                return Ok(ExpandStep::Infeasible(
                    DomainExecutionEvidence::before_selection(policy_execution, &self.cfg),
                    Cert {
                        kind: CertKind::Infeasible,
                        value: slack,
                    },
                ))
            }
            EvalOut::Deadline(policy_execution) => {
                return Ok(ExpandStep::DeadlineHit(
                    DomainExecutionEvidence::before_selection(policy_execution, &self.cfg),
                ))
            }
            EvalOut::Node(b_eval, st) => (b_eval, *st),
        };
        if !(b_eval.is_finite() && entry.bound.is_finite()) {
            return Ok(ExpandStep::NonFiniteBound(
                DomainExecutionEvidence::from_state(&st),
            ));
        }
        let b = b_eval.max(entry.bound);
        if closed(mode, b) {
            let hints = self.eval_hints(&st, b_eval < entry.bound);
            return Ok(ExpandStep::ClosedAtEval(
                DomainExecutionEvidence::from_state(&st),
                Cert {
                    kind: st.close_src,
                    value: b,
                },
                hints,
            ));
        }
        let (trunk_cands, head_cands) = self.select_candidates(&mut st, entry);
        if head_cands.is_empty() && trunk_cands.is_empty() {
            return Ok(ExpandStep::NoCandidates(
                DomainExecutionEvidence::from_state(&st),
            ));
        }
        let ch = {
            let _t = super::prof::Timer::start(super::prof::Phase::ScoreCands);
            match self.score_candidates(&st, &trunk_cands, &head_cands)? {
                Some(ch) => ch,
                None => {
                    return Ok(ExpandStep::DeadlineHit(
                        DomainExecutionEvidence::from_state(&st),
                    ))
                }
            }
        };
        let epoch_selection = st.epoch_selection;
        st.epoch_execution
            .observe_scored_selection(self.cfg.epoch_frame_route, epoch_selection);
        if ch
            .iter()
            .any(|(left, right)| !(left.is_finite() && right.is_finite()))
        {
            return Ok(ExpandStep::NonFiniteBound(
                DomainExecutionEvidence::from_state(&st),
            ));
        }
        st.width_execution
            .observe_successful_score(self.cfg.epoch_frame_route, st.width_selection);
        // #branch-reduceop: shared pick rule (see `pick_candidate`). THIS is a
        // scored-lane site — the concurrent cifar100 lane runs the parallel
        // expansion paths, not the serial loop.
        let reduce_max = super::branch_reduce_max();
        let (pick, differs) = pick_candidate(&ch, reduce_max);
        note_reduceop(reduce_max, "parallel", ch.len(), pick, differs);
        // #branch-parent-lift: see `parent_lift`.
        let lift = if super::parent_lift_enabled() {
            let lift = parent_lift(&ch);
            note_parent_lift("parallel", ch.len(), b, lift, closed(mode, lift));
            lift
        } else {
            f64::NEG_INFINITY
        };
        let n_t = trunk_cands.len();
        let chb = ch[pick];
        let mut pushes = Vec::new();
        let mut dips = 0usize;
        let mut worst_dip = 0.0f64;
        let mut closed_children = 0usize;
        let mut created_children = 0usize;
        let mut close_kinds = ChildCloseKinds::default();
        let mut closed_leaves: Vec<ClosedLeaf> = Vec::new();
        for (d_i, dr) in [(0usize, 1i8), (1usize, -1i8)] {
            let raw = if d_i == 0 { chb.0 } else { chb.1 };
            if !raw.is_finite() {
                return Ok(ExpandStep::NonFiniteBound(
                    DomainExecutionEvidence::from_state(&st),
                ));
            }
            // #soundness-spine M2c: the split this child slot carries.
            let split = if pick < n_t {
                let (li, pos) = trunk_cands[pick];
                ChildSplit::Trunk((li, pos, dr))
            } else {
                ChildSplit::Head((head_cands[pick - n_t], dr))
            };
            if raw < b - 1e-9 {
                dips += 1;
                worst_dip = worst_dip.min(raw - b);
            }
            let cb = raw.max(b).max(lift);
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
                    close_kinds.head_infeasible += 1;
                    if close_telemetry_on(&self.cfg) {
                        close_kinds.note_cert(CertKind::HeadInfeasible, ly[i] - uy[i]);
                    }
                    closed_leaves.push(ClosedLeaf {
                        split,
                        evidence: cert_mr::empty_evidence(mode, ly[i] - uy[i]),
                        lift: false,
                    });
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
                closed_leaves.push(ClosedLeaf {
                    split,
                    evidence: cert_mr::bound_evidence(mode, cb),
                    lift: !closed(mode, raw),
                });
                let raw_closes = closed(mode, raw);
                close_kinds.note_scored(raw_closes);
                if close_telemetry_on(&self.cfg) {
                    close_kinds.note_cert(
                        if raw_closes {
                            CertKind::Raw
                        } else {
                            CertKind::Lift
                        },
                        cb,
                    );
                }
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
                    // #margin-row-domain-alpha: warm start (see
                    // `DomainEntry::alpha`).
                    alpha: st.alpha.clone(),
                    // #clip-stateful: monotone inheritance.
                    clip_acc: st.clip_acc.clone(),
                    split_order: child_split_order(
                        entry,
                        self.cfg.initial_heads.len(),
                        pick >= n_t,
                    ),
                });
            }
        }
        // #mr-close-telemetry: this OPEN parent's retirements (telemetry only).
        if close_telemetry_on(&self.cfg) {
            close_kinds.retire = st.retire_certs;
        }
        Ok(ExpandStep::Expanded {
            dips,
            worst_dip,
            pushes,
            closed_children,
            created_children,
            close_kinds,
            closed_leaves,
            execution: Box::new(DomainExecutionEvidence::from_state(&st)),
            hints: self.eval_hints(&st, false),
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
                frontier_execution: FrontierExecutionEvidence::joint(1, 1),
                policy_execution: PolicyExecutionEvidence::from_config(&BabConfig::default()),
                epoch_execution: EpochExecutionEvidence::default(),
                width_execution: WidthExecutionEvidence::default(),
                root_precision_execution: Default::default(),
                backward_interm_execution: Default::default(),
                root_alpha_execution: Default::default(),
                close_integrity: CloseIntegrityEvidence::default(),
            },
            epoch_failures: 0,
            ledger: Ledger::default(),
            nested_ledgers_ok: true,
            lane_cache: super::gpu_backward::cache::PlanCache::default(),
            // #closure-adaptive: this determinism harness predates the ladder
            // and must keep measuring the SHIPPED path, so the observer is off
            // and the tier stays at `TIER_BASE` — byte-identical to the
            // pre-policy driver rather than a hand-set tier.
            closure: super::closure::ClosureMonitor::new(false),
            b5: stack_eval::B5Trace::default(),
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
            alpha: Vec::new(),
            clip_acc: None,
            split_order: 0,
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
                        alpha: c.alpha,
                        clip_acc: c.clip_acc,
                        split_order: c.split_order,
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
            let mut pack = pack_from_rows(&self.eng, al, au);
            // #clip-and-verify / #closure-adaptive: the CPU producer
            // (`build_pack`) applies the split-derived y-box tightening at this
            // point, so the batched prefill must too. Otherwise a PREFILLED
            // pack differs from the one a cache MISS would have rebuilt, which
            // breaks the LRU's stated contract ("a hit returns exactly what a
            // miss would have rebuilt") and silently disarms tier 3 for every
            // domain the prefill reached. Sound either way — the tightening is
            // shrink-only — but only correct one way.
            clip_tighten_pack(&self.eng, &trunk, &mut pack);
            self.lru.put(trunk, Arc::new(pack));
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
    ///
    /// #mr-stack (throughput plan B5): the body is split at the domain's main
    /// certified pass — [`Self::eval_prepare`] (gates, y-box, seed, inherited
    /// β/α state), [`Self::eval_main_pass`] (the pass itself), then
    /// [`Self::eval_finish`] (per-column directs, via-y composition, both
    /// ascents, retirement) — so the domain-stacked frontier can run ONE main
    /// pass for a whole batch between the per-domain halves. This function is
    /// those three calls in sequence: the same statements in the same order as
    /// the unsplit body, so every existing caller is unchanged.
    fn eval_with_pack(&self, pack: &Arc<YPack>, entry: &DomainEntry) -> Result<EvalOut> {
        let seeded = match self.eval_prepare(pack, entry)? {
            EvalPrep::Done(out) => return Ok(*out),
            EvalPrep::Seeded(seeded) => seeded,
        };
        let pass = self.eval_main_pass(&seeded)?;
        self.eval_finish(entry, seeded, pass)
    }

    /// Everything a domain eval computes BEFORE its main certified pass (see
    /// [`Self::eval_with_pack`]). `Done` carries the early exits (deadline,
    /// empty region) exactly as the unsplit body returned them.
    fn eval_prepare(&self, pack: &Arc<YPack>, entry: &DomainEntry) -> Result<EvalPrep> {
        let mut policy_execution = PolicyExecutionEvidence::from_config(&self.cfg);
        let epoch_execution = EpochExecutionEvidence::from_config(&self.cfg);
        let width_execution = WidthExecutionEvidence::from_config(&self.cfg);
        // #deadline-poll (D2): guard the domain-gate rebuild + the main
        // certified pass about to be spent.
        if self.deadline_hit() {
            poll_note("eval_node");
            return Ok(EvalPrep::Done(Box::new(EvalOut::Deadline(
                policy_execution,
            ))));
        }
        let mode = self.eng.root.mode;
        // Rides in `EvalSeeded` to the end of `eval_finish`, so `eval_node`
        // still spans the whole evaluation (stacked or not).
        let t_node = super::prof::Timer::start(super::prof::Phase::EvalNode);
        // #clip-stateful: THE call site that gets the accumulated state. The
        // y-pack producers deliberately stay stateless (see `domain_gates_acc`);
        // this is the pass whose bound gates the frontier.
        let (dom, clip_acc) = {
            let _t = super::prof::Timer::start(super::prof::Phase::EvalGates);
            domain_gates_acc(self.eng.root, &entry.trunk, entry.clip_acc.as_deref())
        };
        // #mr-close-telemetry / #soundness-spine M2c: each emptiness exit
        // carries its certificate's slack — the cover leaf's evidence (the
        // emptiness DECISIONS are unchanged).
        if dom.infeasible {
            // #clip-and-verify VERIFY: the split set's halfspaces make this
            // domain's region empty. Discharged without paying the ~600 ms
            // backward pass — the exact "child that never closes" converted
            // into an instant close.
            return Ok(EvalPrep::Done(Box::new(EvalOut::Infeasible(
                policy_execution,
                dom.infeasible_slack,
            ))));
        }
        let pack = pack.clone();
        let mut ybox = YBox {
            ly: pack.ly0.clone(),
            uy: pack.uy0.clone(),
        };
        ybox.clamp(&entry.heads);
        // #soundness-spine M2c: the crossing slack is the emptiness leaf's
        // certified EVIDENCE now, not only telemetry, so it is computed on
        // every emptiness exit (O(n_y); the DECISION is still `is_empty`).
        if ybox.is_empty() {
            return Ok(EvalPrep::Done(Box::new(EvalOut::Infeasible(
                policy_execution,
                ybox_crossing(&ybox),
            ))));
        }
        ybox.intersect(&entry.ly, &entry.uy);
        if ybox.is_empty() {
            return Ok(EvalPrep::Done(Box::new(EvalOut::Infeasible(
                policy_execution,
                ybox_crossing(&ybox),
            ))));
        }
        // #clip-head: THE insertion point. A head sign fix moved exactly ONE
        // interval above (`YBox::clamp`); here its halfspace re-concretizes the
        // OTHER head columns before the relaxation gates are derived, which is
        // the term alpha-beta-CROWN's `clip_interm_domain` earns most of its
        // per-domain improvement from on the traced rows.
        //
        // Applied to the LOCAL box only. The pack is `Arc`-shared and its LRU is
        // keyed by TRUNK splits alone, so every sibling with a different head
        // state holds the same pack; writing back would leak one child's
        // constraints into all of them.
        let clip_head_on = crate::preset_gates::margin_row_clip_head_enabled();
        clip_head_armed_once(clip_head_on);
        if clip_head_on {
            let t_ch = Instant::now();
            let mut ch = ClipHeadOut {
                heads: entry.heads.len(),
                ..ClipHeadOut::default()
            };
            match pack.head_slack.as_ref() {
                Some(slack) => {
                    let step = super::clip::tighten_head_box(
                        &pack.al,
                        &pack.au,
                        slack,
                        &pack.ly_fwd,
                        &pack.uy_fwd,
                        &entry.heads,
                        self.eng.root.mid.as_slice(),
                        self.eng.root.rad.as_slice(),
                        &mut ybox.ly,
                        &mut ybox.uy,
                        super::clip::HEAD_H_CAP,
                        super::clip::HEAD_C_CAP,
                    );
                    ch.hs = step.hs;
                    ch.cols = step.cols;
                    ch.moved = step.moved;
                    ch.sum_delta = step.sum_delta;
                    ch.closes = step.closes;
                    ch.refused_cross = step.refused_cross;
                    ch.refused_hs = usize::from(step.refused_hs);
                }
                None => ch.frame_failed = 1,
            }
            clip_head_report(&ch, t_ch.elapsed().as_micros());
            policy_execution.clip_head_evals = policy_execution
                .clip_head_evals
                .saturating_add(usize::from(ch.hs > 0));
            policy_execution.clip_head_moved =
                policy_execution.clip_head_moved.saturating_add(ch.moved);
            policy_execution.clip_head_closes =
                policy_execution.clip_head_closes.saturating_add(ch.closes);
            policy_execution.clip_head_refused = policy_execution
                .clip_head_refused
                .saturating_add(ch.refused_cross + ch.refused_hs + ch.frame_failed);
        }
        let gates = head_gates_with(&ybox, mode, self.eng.root.head_alpha.as_deref());
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
                    policy_execution.inherited_retirement_evals = policy_execution
                        .inherited_retirement_evals
                        .saturating_add(1);
                }
            }
        }
        let mb: &MarginBatch = mb_local.as_ref().unwrap_or(&self.mb);
        let nf = mb.nf();
        let ms = {
            let _t = super::prof::Timer::start(super::prof::Phase::EvalSeed);
            margin_seed(mb, &gates, &ybox, mode)
        };
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
        let percol = beta_armed && self.cfg.beta_per_column;
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
        // #margin-row-domain-alpha WARM START (`NY_MARGIN_ROW_DOMAIN_ALPHA=1` /
        // `margin_row.domain_alpha`, default OFF): apply the ancestor's
        // ACCEPTED alpha to this domain's gates BEFORE the main pass, so a
        // child starts from the slope its parent paid for instead of resetting
        // to the frozen root value. Free (`dom/base` pass-cost ratio measures
        // 0.98-1.12 at every width) and sound on any region — lower-line
        // validity is box-independent. `movable` reads the gates
        // `domain_gates` already produced, INCLUDING clip's rebuilt
        // `(alpha, s, c)`, which is what makes the clip x alpha pair compound.
        //
        // Disarmed: `alpha_slots` stays empty, no gate is written, and every
        // pass below is bit-identical to the pre-domain-alpha lane.
        let alpha_policy = self.cfg.domain_alpha_policy.normalized();
        // #dom-alpha-closure-gate: the policy's `enabled` is FROZEN into the config
        // at `margin_row/mod.rs:537`, before any tree exists, so a tier check there
        // is always evaluated at TIER_BASE. This site is inside the expansion loop,
        // so the closure tier here is LIVE.
        //
        // MEASURED 2026-08-31: armed unconditionally, domain_alpha CONVERTS
        // idx_5127_sidx_993 (certified, 5/5 reps, under all six cost settings) but
        // LOSES 4 banked proofs to its own wall cost -- net -30, and -10 at best
        // across every stride/min_depth/iters dial, with two losses that never
        // return. The cost is structural, not a tuning artifact.
        //
        // The discriminator is the lane's OWN child-closure fraction: idx_5127
        // escalates to tier 3 because its closure collapses (which is exactly why
        // it cannot finish), while the rows domain_alpha loses are banked unsats
        // that PROVE -- healthy closure, ladder never leaves TIER_BASE. Gating here
        // spends the ascent only where the lane has already stalled, i.e. where the
        // row was going to be lost anyway and the cost is therefore free.
        //
        // A STATIC gate on banked wall was tested and REFUTED: eight held rows sit
        // at or above the lowest lost wall, and a row banked at 95.3s held while one
        // at 75.0s was lost. Only a runtime progress signal separates them.
        let alpha_armed =
            alpha_policy.enabled || super::closure::tier() >= super::closure::TIER_BETA_DEEP;
        let mut alpha_slots: Vec<super::domain_alpha::Slot> = Vec::new();
        let mut alpha_cur: Vec<f64> = Vec::new();
        let mut alpha_warm = 0usize;
        if alpha_armed {
            policy_execution.domain_alpha_gate_evals =
                policy_execution.domain_alpha_gate_evals.saturating_add(1);
            alpha_slots = super::domain_alpha::movable(self.eng.root, &dom, &entry.trunk);
            if alpha_slots.is_empty() {
                super::domain_alpha::refuse("no-movable-neuron");
            } else {
                policy_execution.domain_alpha_movable_evals = policy_execution
                    .domain_alpha_movable_evals
                    .saturating_add(1);
                policy_execution.domain_alpha_movable_slots = policy_execution
                    .domain_alpha_movable_slots
                    .saturating_add(alpha_slots.len());
                alpha_cur = super::domain_alpha::read_alphas(self.eng.root, &dom, &alpha_slots);
                alpha_warm = super::domain_alpha::apply_inherited(
                    self.eng.root,
                    &mut dom,
                    &alpha_slots,
                    &mut alpha_cur,
                    &entry.alpha,
                );
                policy_execution.domain_alpha_warm_applied = policy_execution
                    .domain_alpha_warm_applied
                    .saturating_add(alpha_warm);
            }
        }
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
        // The main pass's y-box magnitudes for the certified GPU seam (see
        // `eval_main_pass`); computed here, where the unsplit body computed
        // them, from the final local y-box.
        let eval_y_abs: Vec<f64> = ybox
            .ly
            .iter()
            .zip(&ybox.uy)
            .map(|(l, u)| l.abs().max(u.abs()))
            .collect();
        Ok(EvalPrep::Seeded(Box::new(EvalSeeded {
            t_node,
            policy_execution,
            epoch_execution,
            width_execution,
            clip_acc,
            pack,
            ybox,
            gates,
            mb_local,
            survivors,
            ms,
            beta_armed,
            betas,
            head_betas,
            beta_step,
            percol,
            pc,
            head_admissible,
            dom,
            alpha_policy,
            alpha_armed,
            alpha_slots,
            alpha_cur,
            alpha_warm,
            seed_main,
            seed_main_pc,
            eval_y_abs,
        })))
    }

    /// The collection spec every certified pass of one domain eval uses (the
    /// main pass and its β / domain-alpha trials).
    fn eval_collect(&self, unst_rows: bool) -> Collect<'_> {
        Collect {
            unst_abs: true,
            rows: if self.cfg.epoch_policy.tier0_exact > 0 {
                self.cfg.retained.as_deref()
            } else {
                None
            },
            // Gradient capture for the beta and per-domain-alpha proposals
            // only; nothing verdict-bearing reads it. `unst_rows` is a
            // read-only row copy taken BEFORE the domain-stack branch, so a
            // captured pass's bound is bit-identical to an uncaptured one — it
            // costs memory and a memcpy, and it refuses the dark GPU seam.
            unst_rows,
        }
    }

    /// One domain's OWN main certified pass (the per-domain half of
    /// [`Self::eval_with_pack`]; the domain-stacked frontier's fallback).
    fn eval_main_pass(&self, seeded: &EvalSeeded) -> Result<PassOut> {
        let nf = self.seeded_mb(seeded).nf();
        // B5 (#margin-row-gpu-eft, SITE_EVAL): the ~600 ms main pass — 43.4%
        // of the lane — routes through the certified GPU lane's own seam.
        // Dark and fail-closed: with the lane disarmed (or on ANY refusal —
        // beta-carrying dom, tier-0/unst_rows collection, Busy device, cap
        // overflow) this is exactly the established `run_collect` call,
        // bit-for-bit. The seed is the MARGIN seed, which carries certified
        // error, so the lane needs the y-box magnitudes to concretize both
        // discrepancies into the bias-error lane (the `seed_penalty`
        // contract, same as the root pass at bab.rs:910-919).
        let lane_ctx = super::gpu_backward::LaneCtx {
            y_abs: Some(&seeded.eval_y_abs),
            deadline: self.cfg.deadline,
            plan_cache: Some(&self.lane_cache),
        };
        let _t_pass = super::prof::Timer::start(super::prof::Phase::EvalPass);
        let _census_pass = super::census::SiteTimer::start(super::census::Site::EvalMain, nf);
        self.eng.run_collect_seamed(
            seeded.main_seed(),
            Some(&seeded.dom),
            super::engine::LaneDir::Lower,
            self.eval_collect(seeded.unst_rows()),
            &lane_ctx,
        )
    }

    /// The margin batch a seeded domain's seed was built under.
    fn seeded_mb<'s>(&'s self, seeded: &'s EvalSeeded) -> &'s MarginBatch {
        seeded.mb_local.as_ref().unwrap_or(&self.mb)
    }

    /// Everything a domain eval computes AFTER its main certified pass (see
    /// [`Self::eval_with_pack`]), given that pass — the domain's own, or its
    /// block of a domain-stacked pass (bit-identical by construction).
    fn eval_finish(
        &self,
        entry: &DomainEntry,
        seeded: Box<EvalSeeded>,
        mut pass: PassOut,
    ) -> Result<EvalOut> {
        let EvalSeeded {
            t_node: _t_node,
            mut policy_execution,
            epoch_execution,
            width_execution,
            clip_acc,
            pack,
            ybox,
            gates,
            mb_local,
            survivors,
            ms,
            beta_armed,
            mut betas,
            mut head_betas,
            mut beta_step,
            percol,
            mut pc,
            head_admissible,
            mut dom,
            alpha_policy,
            alpha_armed,
            alpha_slots,
            mut alpha_cur,
            alpha_warm,
            seed_main,
            seed_main_pc,
            eval_y_abs: _,
        } = *seeded;
        let mode = self.eng.root.mode;
        let retire_on = self.cfg.col_retire;
        let mb: &MarginBatch = mb_local.as_ref().unwrap_or(&self.mb);
        let nf = mb.nf();
        let collect = self.eval_collect(beta_armed || alpha_armed);
        let per_j = {
            let _t = super::prof::Timer::start(super::prof::Phase::EvalDirect);
            per_class_direct(&self.eng, &pass, &ms, 0..nf)
        };
        let _t_compose = super::prof::Timer::start(super::prof::Phase::EvalCompose);
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
        drop(_t_compose);
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
        //
        // #mr-close-telemetry: the main pass's bound, before any ascent (the
        // β and domain-alpha ascents below only ever raise `b`, and each runs
        // only while `b` is still open). Telemetry only.
        let b_main = b;
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
                let candidate_count = active.len();
                let configured_columns = self.cfg.beta_columns.clamp(1, 32);
                active.truncate(configured_columns);
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
                let mut execution_observed = false;
                'ascent: for _ in 0..super::beta::iters() {
                    let _t_beta = super::prof::Timer::start(super::prof::Phase::EvalBeta);
                    if self.cfg.deadline.is_some_and(|dl| Instant::now() > dl) {
                        break;
                    }
                    if active.is_empty() {
                        break;
                    }
                    if !execution_observed {
                        // Credit only after the first non-deadline iteration
                        // actually consumes the configured per-column set. A
                        // node that merely reached this block as its deadline
                        // expired is not an executed parameter experiment.
                        policy_execution.beta_per_column_evals =
                            policy_execution.beta_per_column_evals.saturating_add(1);
                        policy_execution.beta_columns_max_candidates = policy_execution
                            .beta_columns_max_candidates
                            .max(candidate_count);
                        if candidate_count.min(configured_columns) != candidate_count.min(8) {
                            policy_execution.beta_columns_binding_evals = policy_execution
                                .beta_columns_binding_evals
                                .saturating_add(1);
                        }
                        execution_observed = true;
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
                    let trial = {
                        let _census = super::census::SiteTimer::start(
                            super::census::Site::BetaTrialPercol,
                            nf,
                        );
                        self.eng.run_collect(
                            seed_trial
                                .as_ref()
                                .or(seed_main.as_ref())
                                .unwrap_or(&ms.seed),
                            Some(&dom),
                            super::engine::LaneDir::Lower,
                            None,
                            collect,
                        )
                    };
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
                    // #mr-census plan B1(a): the SHARED-β arm now carries the
                    // same per-iteration `EvalBeta` phase timer as the per-column
                    // arm, so the profile attributes its trials instead of
                    // leaving them in the eval_node residue.
                    let _t_beta = super::prof::Timer::start(super::prof::Phase::EvalBeta);
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
                    let trial = {
                        let _census = super::census::SiteTimer::start(
                            super::census::Site::BetaTrialShared,
                            nf,
                        );
                        self.eng.run_collect(
                            seed_trial.as_ref().unwrap_or(&ms.seed),
                            Some(&dom),
                            super::engine::LaneDir::Lower,
                            None,
                            collect,
                        )
                    };
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
        // #mr-close-telemetry: the bound after the β ascent (telemetry only).
        let b_beta = b;
        // ---- #margin-row-domain-alpha ascent ----
        //
        // Re-optimize this domain's lower-line slopes against ITS OWN gates.
        // Same shape as the β ascent directly above, and deliberately so: the
        // direction comes from `alpha_opt`'s frozen-sign supergradient
        // (`g = vp * U`, evaluated on the DOMAIN's effective alpha), every
        // proposal is re-scored by the UNCHANGED certified pass, and only a
        // measured improvement of the same certified composite the verdict uses
        // is kept. Runs AFTER β so it starts from the β-tightened bound and
        // composes with it; the accepted alpha then rides `st.dom` into
        // `score_candidates` and `st.alpha` into the children.
        //
        // SOUNDNESS is entirely in `domain_alpha`: `[0, 1]` clamp and the
        // `ms = max(alpha, s)` re-derivation at the single write site, exact
        // lines excluded by admission, certified pass as the only scorer. A bad
        // proposal costs one pass and nothing else. ON REJECTION THE GATES ARE
        // RESTORED to the last accepted vector, so the `dom` stored below always
        // describes the bound that was actually published.
        //
        // THE SCHEDULE decides WHICH domains pay. Default (`min_depth 0`,
        // `stride 1`, `gap 0` = unlimited) admits every domain, i.e. exactly the
        // arm the 2x2 measured; a non-default schedule declines domains whose
        // passes are predicted not to pay for themselves, and a declined domain
        // publishes the bound the disarmed lane would publish — from its
        // inheritance, which is free — so declining is tightness-only and never
        // a soundness event.
        //
        // The `&&` chain SHORT-CIRCUITS deliberately: a disarmed lane never
        // consults the schedule (`schedule_admits_with` is pure and takes its
        // three knobs from the SAME frozen `alpha_policy` as iters/eta, so the
        // knobs' values are resolved at entry like the rest of the family but
        // can only influence an ARMED ascent) — which is what their
        // `Requirement::Armed` edges assert — and the disarmed path is
        // byte-identical, not merely equivalent.
        let alpha_depth = entry.trunk.len() + entry.heads.len();
        let alpha_active = alpha_armed && !alpha_slots.is_empty() && !closed(mode, b);
        let alpha_scheduled = alpha_active
            && super::domain_alpha::schedule_admits_with(
                alpha_depth,
                b,
                alpha_policy.min_depth,
                alpha_policy.stride,
                alpha_policy.gap,
            );
        if alpha_active && !alpha_scheduled {
            super::domain_alpha::note_skipped();
        }
        if alpha_scheduled {
            let b0 = b;
            let mut eta_a = alpha_policy.eta;
            let mut eta_baseline = super::DomainAlphaRuntimePolicy::default().eta;
            let mut steps = 0usize;
            let mut accepts = 0usize;
            let mut cap_boundary_reachable = true;
            let outer_frame = self.cfg.epoch_frame_route == EpochFrameRoute::Outer;
            let mut last_trial_finite = false;
            let mut iters_counterfactual = false;
            let mut eta_counterfactual_scored = false;
            let mut eta_trace_live = alpha_policy.eta.to_bits()
                != super::DomainAlphaRuntimePolicy::default().eta.to_bits();
            let mut grad = {
                let t = Instant::now();
                let _census = super::census::SiteTimer::start(super::census::Site::WalkAlpha, 1);
                let g = super::alpha_opt::supergradient(
                    self.eng.net,
                    self.eng.root,
                    &super::domain_alpha::effective_alpha(self.eng.root, &dom),
                    &pass,
                    worst_col,
                );
                super::domain_alpha::charge_grad(t.elapsed().as_micros() as u64);
                g
            };
            if grad.is_none() {
                cap_boundary_reachable = false;
                super::domain_alpha::refuse("no-gradient-capture");
            }
            for trial_index in 0..alpha_policy.iters {
                if self.deadline_hit() {
                    cap_boundary_reachable = false;
                    break;
                }
                let Some(g) = grad.as_ref() else {
                    cap_boundary_reachable = false;
                    break;
                };
                let baseline_cand = eta_trace_live.then(|| {
                    super::domain_alpha::propose(&alpha_slots, &alpha_cur, g, eta_baseline)
                });
                let Some(cand) = super::domain_alpha::propose(&alpha_slots, &alpha_cur, g, eta_a)
                else {
                    cap_boundary_reachable = false;
                    super::domain_alpha::refuse("degenerate-direction");
                    break;
                };
                let eta_proposal_differs = baseline_cand
                    .as_ref()
                    .is_some_and(|baseline| baseline.as_ref() != Some(&cand));
                super::domain_alpha::write_alphas(self.eng.root, &mut dom, &alpha_slots, &cand);
                if steps == 0 {
                    policy_execution.domain_alpha_ascent_calls =
                        policy_execution.domain_alpha_ascent_calls.saturating_add(1);
                    if outer_frame {
                        policy_execution.domain_alpha_outer_ascent_calls = policy_execution
                            .domain_alpha_outer_ascent_calls
                            .saturating_add(1);
                    }
                }
                steps += 1;
                policy_execution.domain_alpha_trials_started = policy_execution
                    .domain_alpha_trials_started
                    .saturating_add(1);
                // Same seed family and collect shape as the main pass. NOTE the
                // seed is the one the MAIN pass used, i.e. it carries the
                // INHERITED head β rather than any the β ascent just accepted —
                // still a valid Lagrangian (`β >= 0`), so the trial bound is a
                // valid bound of this domain; at worst it is scored slightly
                // loose, and acceptance is `> b` regardless.
                let t_trial = Instant::now();
                let trial = {
                    let _census =
                        super::census::SiteTimer::start(super::census::Site::AlphaTrial, nf);
                    self.eng.run_collect(
                        seed_main_pc
                            .as_ref()
                            .or(seed_main.as_ref())
                            .unwrap_or(&ms.seed),
                        Some(&dom),
                        super::engine::LaneDir::Lower,
                        None,
                        collect,
                    )
                };
                super::domain_alpha::charge_trial(t_trial.elapsed().as_micros() as u64);
                let Ok(mut trial) = trial else {
                    cap_boundary_reachable = false;
                    policy_execution.domain_alpha_trial_pass_failures = policy_execution
                        .domain_alpha_trial_pass_failures
                        .saturating_add(1);
                    super::domain_alpha::refuse("trial-pass-failed");
                    super::domain_alpha::write_alphas(
                        self.eng.root,
                        &mut dom,
                        &alpha_slots,
                        &alpha_cur,
                    );
                    break;
                };
                policy_execution.domain_alpha_trial_passes_completed = policy_execution
                    .domain_alpha_trial_passes_completed
                    .saturating_add(1);
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
                last_trial_finite = bt.is_finite();
                if last_trial_finite {
                    policy_execution.domain_alpha_finite_trial_scores = policy_execution
                        .domain_alpha_finite_trial_scores
                        .saturating_add(1);
                    if outer_frame {
                        policy_execution.domain_alpha_outer_finite_trial_scores = policy_execution
                            .domain_alpha_outer_finite_trial_scores
                            .saturating_add(1);
                    }
                    if trial_index >= super::DomainAlphaRuntimePolicy::default().iters {
                        iters_counterfactual = true;
                    }
                    if eta_proposal_differs {
                        eta_counterfactual_scored = true;
                    }
                }
                if eta_proposal_differs {
                    // Once the proposal vector diverges, a shadow schedule can
                    // no longer follow the actual accepted/rejected states.
                    eta_trace_live = false;
                }
                if bt.is_finite() && bt > b {
                    b = bt;
                    alpha_cur = cand;
                    accepts += 1;
                    policy_execution.domain_alpha_trials_accepted = policy_execution
                        .domain_alpha_trials_accepted
                        .saturating_add(1);
                    // #margin-row-col-retire: each `per_t[c]` scored under an
                    // accepted (valid) relaxation is a valid column bound.
                    if let Some(tracked) = per_track.as_mut() {
                        for (tv, &nv) in tracked.iter_mut().zip(&per_t) {
                            if nv.is_finite() && nv > *tv {
                                *tv = nv;
                            }
                        }
                    }
                    std::mem::swap(&mut pass, &mut trial);
                    if closed(mode, b) {
                        cap_boundary_reachable = false;
                        break;
                    }
                    // Live gradient: x* and the frozen-sign selections go stale
                    // precisely when the step was material, so refresh both the
                    // worst column and the direction from the ACCEPTED pass.
                    worst_col = wc_t;
                    eta_a = (eta_a * 1.5).min(super::domain_alpha::ETA_MAX);
                    eta_baseline = (eta_baseline * 1.5).min(super::domain_alpha::ETA_MAX);
                    // NOTE: an earlier draft trimmed this refresh on the LAST
                    // trial as "nothing reads `grad` again". That is no longer
                    // true: the low-iters counterfactual probe after the loop
                    // consults `grad` to decide whether one more trial was
                    // possible, so the refresh must run on every accept.
                    let t = Instant::now();
                    let census_walk =
                        super::census::SiteTimer::start(super::census::Site::WalkAlpha, 1);
                    grad = super::alpha_opt::supergradient(
                        self.eng.net,
                        self.eng.root,
                        &super::domain_alpha::effective_alpha(self.eng.root, &dom),
                        &pass,
                        worst_col,
                    );
                    drop(census_walk);
                    super::domain_alpha::charge_grad(t.elapsed().as_micros() as u64);
                    if grad.is_none() {
                        cap_boundary_reachable = false;
                        super::domain_alpha::refuse("dir-refresh");
                    }
                } else {
                    // Restore the last ACCEPTED alpha before the next proposal:
                    // the stored `dom` must describe the published bound.
                    super::domain_alpha::write_alphas(
                        self.eng.root,
                        &mut dom,
                        &alpha_slots,
                        &alpha_cur,
                    );
                    eta_a *= 0.5;
                    eta_baseline *= 0.5;
                    if eta_a < super::domain_alpha::ETA_MIN {
                        cap_boundary_reachable = false;
                        break;
                    }
                }
            }
            if alpha_policy.iters < super::DomainAlphaRuntimePolicy::default().iters
                && cap_boundary_reachable
                && last_trial_finite
                && !self.deadline_hit()
                && grad.as_ref().is_some_and(|g| {
                    super::domain_alpha::propose(&alpha_slots, &alpha_cur, g, eta_a).is_some()
                })
            {
                iters_counterfactual = true;
            }
            if iters_counterfactual {
                policy_execution.domain_alpha_iters_counterfactual_domains = policy_execution
                    .domain_alpha_iters_counterfactual_domains
                    .saturating_add(1);
                if outer_frame {
                    policy_execution.domain_alpha_outer_iters_counterfactual_domains =
                        policy_execution
                            .domain_alpha_outer_iters_counterfactual_domains
                            .saturating_add(1);
                }
            }
            if eta_counterfactual_scored {
                policy_execution.domain_alpha_eta_counterfactual_scored_domains = policy_execution
                    .domain_alpha_eta_counterfactual_scored_domains
                    .saturating_add(1);
                if outer_frame {
                    policy_execution.domain_alpha_outer_eta_counterfactual_scored_domains =
                        policy_execution
                            .domain_alpha_outer_eta_counterfactual_scored_domains
                            .saturating_add(1);
                }
            }
            super::domain_alpha::report(
                alpha_depth,
                alpha_slots.len(),
                alpha_warm,
                steps,
                accepts,
                b0,
                b,
                eta_a,
            );
        }
        // #margin-row-domain-alpha: what the children inherit — every movable
        // slot whose accepted alpha differs from the ROOT's, capped by
        // `domain_alpha::CARRY_CAP`. Built from THIS domain's slot list, so an
        // inherited entry naming a neuron that has since been pinned is dropped
        // rather than carried forward stale. Empty (and the entry's list passes
        // through) when the gate is off.
        let alpha_out = if alpha_armed && !alpha_slots.is_empty() {
            super::domain_alpha::carry(self.eng.root, &alpha_slots, &alpha_cur)
        } else {
            entry.alpha.clone()
        };
        if alpha_armed && !alpha_slots.is_empty() {
            policy_execution.domain_alpha_carried_entries = policy_execution
                .domain_alpha_carried_entries
                .saturating_add(alpha_out.len());
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
        // #mr-close-telemetry: each newly retired column's certified composite
        // — its whole certificate on this domain's subtree when the domain
        // stays open. Telemetry only.
        let mut retire_certs = CertTally::default();
        if retire_on {
            let tracked = per_track.as_deref().unwrap_or(per_j.as_slice());
            let mut newly = 0usize;
            let close_telemetry = close_telemetry_on(&self.cfg);
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
                            if close_telemetry {
                                retire_certs.note(comp);
                            }
                        }
                    }
                }
            }
            if newly > 0 {
                policy_execution.newly_retired_columns =
                    policy_execution.newly_retired_columns.saturating_add(newly);
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
        // #mr-close-telemetry: the first stage whose bound closed (read only
        // when `b` closes; see `b_main` / `b_beta`). Telemetry only.
        let close_src = if closed(mode, b_main) {
            CertKind::Eval
        } else if closed(mode, b_beta) {
            CertKind::Beta
        } else {
            CertKind::DomainAlpha
        };
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
                policy_execution,
                epoch_execution,
                width_execution,
                epoch_selection: EpochSelectionObservation::default(),
                width_selection: WidthSelectionObservation::default(),
                alpha: alpha_out,
                clip_acc,
                mb_local,
                coll,
                coll_rows,
                pass,
                dom,
                pack,
                close_src,
                retire_certs,
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
    fn head_prerank(&self, st: &NodeState, heads: &[HeadFix]) -> (Vec<usize>, usize, bool) {
        let mut scored = self.head_variant_scores(st, heads);
        scored.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| b.1.total_cmp(&a.1))
                .then_with(|| b.2.cmp(&a.2))
        });
        let ranked: Vec<usize> = scored.into_iter().map(|(_, _, i)| i).collect();
        let selected: Vec<usize> = ranked
            .iter()
            .take(self.cfg.effective_k_head)
            .copied()
            .collect();
        let contextual: Vec<usize> = ranked
            .iter()
            .take(self.cfg.effective_contextual_k_head)
            .copied()
            .collect();
        let available = ranked.len();
        let counterfactual = selected != contextual;
        (selected, available, counterfactual)
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
    /// #epoch-monotone: intersect the PARENT frame's intermediate intervals
    /// into a freshly rebuilt epoch frame, and re-derive the affected gates.
    ///
    /// # Why this is sound
    ///
    /// The epoch is built over the SAME input box (`root.lo`/`root.hi`) with the
    /// domain's splits additionally baked in, so the epoch's feasible region is
    /// a SUBSET of the parent's. Every parent interval `[l, u]` is therefore a
    /// valid enclosure of that pre-activation over the epoch region too, and
    /// intersecting two valid enclosures is shrink-only — the same argument the
    /// #backward-interm pass already runs on (`.max(l)` / `.min(u)`). Nothing
    /// here can widen an interval, so the epoch can lose a proof but never
    /// manufacture one.
    ///
    /// The gates must be re-derived from the intersected box or they would
    /// describe a wider triangle than the interval claims: that direction is
    /// merely loose, not unsound, but it throws away the entire point. They are
    /// re-derived through the certified root path (`gates_from_box` then
    /// `repair_upper_lines` in outward mode) — the same functions the root
    /// itself used.
    ///
    /// # What it deliberately does NOT touch
    ///
    /// * **Split neurons.** `build_retaining` baked an EXACT fixed line at each
    ///   one; re-deriving a relaxation over them would delete the mechanism.
    /// * **`unst`.** A neuron that became stable under the intersected box stays
    ///   in the list. Leaving it is conservative (splitting a stable neuron is
    ///   sound — one child is simply empty) and, more importantly, `clip_rows`
    ///   is indexed by POSITION IN `unst`, so re-deriving the list here would
    ///   silently re-point every retained halfspace row at a different neuron.
    ///   That is the frame error `clip::frame_ok` exists to catch, and it must
    ///   not be introduced on purpose.
    ///
    /// Returns `false` — caller fails open to normal expansion — on any frame
    /// mismatch, or if the intersection CROSSES an interval. A crossing does
    /// certify the epoch region empty, but turning that into a closure is a new
    /// verdict path and this function is not allowed to open one.
    fn inherit_parent_intervals(
        &self,
        eroot: &mut RootGates,
        splits: &[(usize, usize, i8)],
    ) -> bool {
        let parent = self.eng.root;
        if eroot.layers.len() != parent.layers.len() {
            eprintln!(
                "[epoch-monotone] REFUSED: layer count {} vs parent {}",
                eroot.layers.len(),
                parent.layers.len()
            );
            return false;
        }
        let mut tightened = 0usize;
        let mut regated = 0usize;
        for (li, elayer) in eroot.layers.iter_mut().enumerate() {
            let player = &parent.layers[li];
            if elayer.n != player.n || elayer.l.len() != player.l.len() {
                eprintln!(
                    "[epoch-monotone] REFUSED: layer {li} width {} vs parent {}",
                    elayer.n, player.n
                );
                return false;
            }
            let mut touched: Vec<usize> = Vec::new();
            for i in 0..elayer.l.len() {
                let (pl, pu) = (player.l[i], player.u[i]);
                if !(pl.is_finite() && pu.is_finite()) {
                    continue;
                }
                let nl = elayer.l[i].max(pl);
                let nu = elayer.u[i].min(pu);
                if nl > nu {
                    eprintln!(
                        "[epoch-monotone] REFUSED: layer {li} neuron {i} crossed \
                         ({nl:.6e} > {nu:.6e}); the region is certified empty but this \
                         path may not open a verdict"
                    );
                    return false;
                }
                if nl > elayer.l[i] || nu < elayer.u[i] {
                    elayer.l[i] = nl;
                    elayer.u[i] = nu;
                    tightened += 1;
                    // The split neurons keep their baked exact lines.
                    if !splits.iter().any(|&(l2, i2, _)| l2 == li && i2 == i) {
                        touched.push(i);
                    }
                }
            }
            if touched.is_empty() {
                continue;
            }
            let (alpha2, mut s2, mut c2) = super::root::gates_from_box(&elayer.l, &elayer.u);
            if eroot.mode.outward() {
                super::root::repair_upper_lines(&elayer.l, &elayer.u, &mut s2, &mut c2);
            }
            for &i in &touched {
                elayer.alpha[i] = alpha2[i];
                elayer.s[i] = s2[i];
                elayer.c[i] = c2[i];
                // `ms = max(alpha, s)` is the certified error-carry contraction
                // factor; whoever writes `alpha` MUST re-derive it or the
                // carried error is UNDER-scaled, which is a false-UNSAT risk.
                elayer.ms[i] = alpha2[i].max(s2[i]);
                regated += 1;
            }
        }
        // R9: without this line an epoch arm cannot tell "inherited nothing"
        // from "never ran".
        eprintln!(
            "[epoch-monotone] inherited parent intervals: tightened={tightened} \
             regated={regated} (split neurons keep their baked exact lines)"
        );
        true
    }

    fn record_epoch_failure(&mut self) {
        self.epoch_failures = self.epoch_failures.saturating_add(1);
        self.stats.epoch_execution.attempt_failures = self
            .stats
            .epoch_execution
            .attempt_failures
            .saturating_add(1);
    }

    fn try_epoch(&mut self, entry: &DomainEntry) -> bool {
        self.stats.epoch_execution.tree_entries =
            self.stats.epoch_execution.tree_entries.saturating_add(1);
        if self.cfg.epoch_trigger_depth == 0 || entry.trunk.len() < self.cfg.epoch_trigger_depth {
            return false;
        }
        if self.cfg.epoch_policy.parent_topk > 0 && self.cfg.epoch_parent_generations_left == 0 {
            self.stats.epoch_execution.generation_refusals = self
                .stats
                .epoch_execution
                .generation_refusals
                .saturating_add(1);
            return false;
        }
        let actual_admit = self.epoch_failures < self.cfg.epoch_attempt_cap;
        let contextual_admit = self.epoch_failures < 2;
        self.stats.epoch_execution.attempt_cap_decisions = self
            .stats
            .epoch_execution
            .attempt_cap_decisions
            .saturating_add(1);
        if actual_admit != contextual_admit {
            self.stats
                .epoch_execution
                .attempt_cap_counterfactual_differences = self
                .stats
                .epoch_execution
                .attempt_cap_counterfactual_differences
                .saturating_add(1);
        }
        if !actual_admit {
            self.stats.epoch_execution.attempt_cap_refusals = self
                .stats
                .epoch_execution
                .attempt_cap_refusals
                .saturating_add(1);
            return false;
        }
        if let Some(dl) = self.cfg.deadline {
            let rem = dl.saturating_duration_since(Instant::now()).as_secs_f64();
            // The full build needs root-build-sized headroom. The incremental
            // arm pays only selected prefix rows; its own deadline checks make
            // a late attempt fail open, so retain a small fixed floor instead
            // of the resnet-large guard that otherwise makes the arm inert.
            let need = if self.cfg.epoch_policy.parent_topk > 0 {
                0.5
            } else {
                3.0 * self.cfg.root_build_secs.max(0.05)
            };
            if rem < need {
                self.stats.epoch_execution.deadline_refusals = self
                    .stats
                    .epoch_execution
                    .deadline_refusals
                    .saturating_add(1);
                return false;
            }
        }
        let configured_depth_attempt = self.cfg.epoch_policy.depth > 0
            && self.cfg.epoch_trigger_depth == self.cfg.epoch_policy.depth;
        let configured_depth_binding =
            configured_depth_attempt && entry.trunk.len() == self.cfg.epoch_policy.depth;
        if configured_depth_attempt {
            self.stats.epoch_execution.configured_depth_admissions = self
                .stats
                .epoch_execution
                .configured_depth_admissions
                .saturating_add(1);
        } else {
            self.stats.epoch_execution.adaptive_depth_admissions = self
                .stats
                .epoch_execution
                .adaptive_depth_admissions
                .saturating_add(1);
        }
        self.stats.epoch_execution.attempts_started = self
            .stats
            .epoch_execution
            .attempts_started
            .saturating_add(1);
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
        let incremental = self.cfg.epoch_policy.parent_topk > 0;
        match (self.cfg.epoch_frame_route, incremental) {
            (EpochFrameRoute::Outer, false) => {
                self.stats.epoch_execution.outer_full_attempts = self
                    .stats
                    .epoch_execution
                    .outer_full_attempts
                    .saturating_add(1);
            }
            (EpochFrameRoute::Outer, true) => {
                self.stats.epoch_execution.outer_incremental_attempts = self
                    .stats
                    .epoch_execution
                    .outer_incremental_attempts
                    .saturating_add(1);
            }
            (EpochFrameRoute::NestedFull, false) | (EpochFrameRoute::NestedIncremental, false) => {
                self.stats.epoch_execution.nested_full_attempts = self
                    .stats
                    .epoch_execution
                    .nested_full_attempts
                    .saturating_add(1);
            }
            (EpochFrameRoute::NestedFull, true) | (EpochFrameRoute::NestedIncremental, true) => {
                self.stats.epoch_execution.nested_incremental_attempts = self
                    .stats
                    .epoch_execution
                    .nested_incremental_attempts
                    .saturating_add(1);
            }
        }
        let (mut eroot, eret): (RootGates, Option<RetainedRows>) = if incremental {
            self.stats.epoch_execution.parent_refreshes_started = self
                .stats
                .epoch_execution
                .parent_refreshes_started
                .saturating_add(1);
            let refreshed = super::backward_interm::refresh_parent_frame(
                net,
                root,
                &abs,
                self.cfg.epoch_policy.parent_topk,
                self.cfg.epoch_policy.parent_topk.min(64),
                self.cfg.deadline,
            );
            let (eroot, refresh) = match refreshed {
                Ok(pair) => pair,
                Err(err) => {
                    // Any malformed frame, numerical refusal, or deadline
                    // leaves the established domain expansion in charge.
                    eprintln!("[epoch-parent] REFUSED refresh: {err}");
                    self.stats.epoch_execution.parent_refresh_refusals = self
                        .stats
                        .epoch_execution
                        .parent_refresh_refusals
                        .saturating_add(1);
                    self.record_epoch_failure();
                    return false;
                }
            };
            self.stats.epoch_execution.parent_refreshes_completed = self
                .stats
                .epoch_execution
                .parent_refreshes_completed
                .saturating_add(1);
            self.stats.epoch_execution.parent_refresh_columns = self
                .stats
                .epoch_execution
                .parent_refresh_columns
                .saturating_add(refresh.columns);
            self.stats.epoch_execution.parent_refresh_tightened = self
                .stats
                .epoch_execution
                .parent_refresh_tightened
                .saturating_add(refresh.tightened);
            self.stats.epoch_execution.parent_refresh_stabilized = self
                .stats
                .epoch_execution
                .parent_refresh_stabilized
                .saturating_add(refresh.stabilized);
            if refresh.tightened == 0 {
                eprintln!("[epoch-parent] REFUSED nested run: zero downstream intervals tightened");
                self.stats.epoch_execution.parent_refresh_refusals = self
                    .stats
                    .epoch_execution
                    .parent_refresh_refusals
                    .saturating_add(1);
                self.record_epoch_failure();
                return false;
            }
            self.stats.epoch_execution.parent_refreshes_accepted = self
                .stats
                .epoch_execution
                .parent_refreshes_accepted
                .saturating_add(1);
            let speedup = self.cfg.root_build_secs / refresh.elapsed_secs.max(1e-9);
            eprintln!(
                "[epoch-parent] accepted frame: refresh_secs={:.3} \
                 reference_root_secs={:.3} observed_ratio={:.2}x generations_left={}",
                refresh.elapsed_secs,
                self.cfg.root_build_secs,
                speedup,
                self.cfg.epoch_parent_generations_left
            );
            // Parent Tier-0 rows are indexed in the parent's unstable frame;
            // the refreshed frame deliberately changes that index space.
            (eroot, None)
        } else {
            // #epoch-carry: snapshot the root's per-layer alpha as the seed.
            let epoch_alpha_seed: Vec<Vec<f64>> =
                root.layers.iter().map(|l| l.alpha.clone()).collect();
            let built = RootGates::build_retaining_with_policy(
                net,
                &root.lo,
                &root.hi,
                root.mode,
                self.cfg.deadline,
                self.cfg.retain_cfg.as_ref(),
                &abs,
                self.cfg.root_precision_policy,
                // Carry the run policy; split epoch rebuilds do not rerun BI.
                self.cfg.backward_interm_policy,
                // #epoch-carry: preserve optimized alpha rather than restoring
                // the area heuristic. Each alpha in [0,1] is a valid lower line;
                // the builder recomputes the error factor ms = max(alpha, s).
                Some(&epoch_alpha_seed),
            );
            let Ok(pair) = built else {
                // Rebuild failed (deadline / numeric): normal expansion instead.
                self.stats.epoch_execution.full_rebuild_refusals = self
                    .stats
                    .epoch_execution
                    .full_rebuild_refusals
                    .saturating_add(1);
                self.record_epoch_failure();
                return false;
            };
            pair
        };
        // #epoch-monotone: intersect the parent's certified intervals into full
        // rebuilds before nested search. A fresh relaxation can be looser even
        // on a smaller domain; inheritance preserves the interval tightening.
        if !incremental && !self.inherit_parent_intervals(&mut eroot, &abs) {
            // Intersection found a crossed interval, or the frames did not
            // line up. Fail-open to normal expansion: this path may never
            // manufacture a closure.
            self.stats.epoch_execution.full_rebuild_refusals = self
                .stats
                .epoch_execution
                .full_rebuild_refusals
                .saturating_add(1);
            self.record_epoch_failure();
            return false;
        }
        if !incremental {
            self.stats.epoch_execution.full_rebuilds_completed = self
                .stats
                .epoch_execution
                .full_rebuilds_completed
                .saturating_add(1);
        }
        // Full-tableau epochs retain the historical one-level nesting. The
        // incremental parent-frame arm may recur after `epoch_trigger_depth`
        // NEW trunk
        // splits: every nested root has baked the prior constraints already,
        // so recursion makes strict domain-depth progress and refreshes the
        // gating path at the cadence measured necessary for idx_1298.
        let (nested_epoch_depth, nested_epoch_attempts, nested_epoch_generations) =
            self.cfg.nested_epoch_limits(incremental);
        let ncfg = BabConfig {
            // #publish-deadline: a nested epoch subtree inherits the parent's
            // publication authority unchanged.
            publish_deadline: self.cfg.publish_deadline,
            k_head: self.cfg.k_head,
            contextual_k_head: self.cfg.contextual_k_head,
            k_trunk: self.cfg.k_trunk,
            contextual_k_trunk: self.cfg.contextual_k_trunk,
            effective_k_head: self.cfg.effective_k_head,
            effective_contextual_k_head: self.cfg.effective_contextual_k_head,
            effective_k_trunk: self.cfg.effective_k_trunk,
            effective_contextual_k_trunk: self.cfg.effective_contextual_k_trunk,
            adaptive: self.cfg.adaptive,
            contextual_adaptive: self.cfg.contextual_adaptive,
            max_expansions: self
                .cfg
                .max_expansions
                .saturating_sub(self.stats.expansions),
            deadline: self.cfg.deadline,
            lru_cap: self.cfg.lru_cap,
            frontier: 1,
            frontier_contextual: 1,
            domain_stack: false,
            domain_stack_main_eval: self.cfg.domain_stack_main_eval,
            domain_alpha_policy: self.cfg.domain_alpha_policy,
            epoch_policy: self.cfg.epoch_policy,
            epoch_frame_route: if incremental {
                EpochFrameRoute::NestedIncremental
            } else {
                EpochFrameRoute::NestedFull
            },
            retained: eret.map(Arc::new),
            epoch_trigger_depth: nested_epoch_depth,
            epoch_attempt_cap: nested_epoch_attempts,
            epoch_parent_generations_left: nested_epoch_generations,
            root_build_secs: self.cfg.root_build_secs,
            retain_cfg: self.cfg.retain_cfg,
            initial_heads: entry.heads.clone(),
            initial_ybox: Some((entry.ly.clone(), entry.uy.clone())),
            // Head-group certificates cover only the unconditional outer box.
            head_groups: None,
            // #margin-row-col-retire: the nested lane re-derives its own root
            // fail-set on the epoch gates; the outer retirement set is NOT
            // inherited (root frames differ), only the gate carries over.
            col_retire: self.cfg.col_retire,
            beta_per_column: self.cfg.beta_per_column,
            beta_columns: self.cfg.beta_columns,
            parallel_prebuild: self.cfg.parallel_prebuild,
            root_precision_policy: self.cfg.root_precision_policy,
            backward_interm_policy: self.cfg.backward_interm_policy,
            root_precision_execution: self.cfg.root_precision_execution,
            backward_interm_execution: self.cfg.backward_interm_execution,
            root_alpha_execution: self.cfg.root_alpha_execution,
            // #wave-beta-bab: a nested epoch frame is a ReLU-split subdomain;
            // it always runs the legacy tree.
            wave: wave::WavePolicy::default(),
            // The frozen-dual tier runs on the outer frame only.
            frozen_dual_tier: self.cfg.frozen_dual_tier,
            #[cfg(test)]
            close_telemetry_off: self.cfg.close_telemetry_off,
        };
        let adv = self.stats.tree_classes.clone();
        // Entry-frozen presence semantics: `NY_EPOCH_DEBUG=0` and an empty
        // UTF-8 value arm the trace; a non-UTF-8 value remains dark. Nested
        // runs inherit this bit and never reopen process-global state.
        let dbg = self.cfg.epoch_policy.debug_trace;
        if dbg {
            self.stats.epoch_execution.debug_emissions =
                self.stats.epoch_execution.debug_emissions.saturating_add(1);
            eprintln!(
                "[epoch] attempt at depth {} (trunk {} heads {}): outer bound {:+.5}",
                entry.trunk.len() + entry.heads.len(),
                entry.trunk.len(),
                entry.heads.len(),
                entry.bound
            );
        }
        let t_epoch = Instant::now();
        self.stats.epoch_execution.nested_started =
            self.stats.epoch_execution.nested_started.saturating_add(1);
        // Proof bundle v2: the nested frame's root is this leaf's region; its
        // hints carry the leaf's ABSOLUTE literals (translated with THIS
        // frame's root) as their prefix. Hint data only.
        let nested_prefix = if self.ledger.hints.capturing() {
            let map = SiteMap::from_root(self.eng.root, self.eng.net);
            let own = hint_splits(
                &entry.trunk,
                &entry.heads,
                self.cfg.initial_heads.len(),
                entry.split_order,
                &[],
                &[],
                None,
            )
            .and_then(|(splits, _)| map.absolute(&splits))
            .unwrap_or_default();
            let mut p = self.ledger.hint_prefix.clone();
            p.extend(own);
            p
        } else {
            Vec::new()
        };
        match MarginRowBab::run_inner(net, &eroot, self.mb.t, &adv, ncfg, nested_prefix) {
            Ok(MarginRowOutcome::Unsat(n, token)) => {
                // #soundness-spine M2c: the nested frame's sub-domain token is
                // the evidence of the leaf the caller records next.
                self.ledger.pending_subtree = Some(token);
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
                self.stats.epoch_execution.nested_runs_with_stats = self
                    .stats
                    .epoch_execution
                    .nested_runs_with_stats
                    .saturating_add(1);
                if configured_depth_binding {
                    self.stats
                        .epoch_execution
                        .configured_depth_nested_with_stats = self
                        .stats
                        .epoch_execution
                        .configured_depth_nested_with_stats
                        .saturating_add(1);
                }
                self.stats.epoch_execution.nested_unsat =
                    self.stats.epoch_execution.nested_unsat.saturating_add(1);
                self.stats.epoch_execution.attempts_closed =
                    self.stats.epoch_execution.attempts_closed.saturating_add(1);
                self.stats.epoch_execution.closures =
                    self.stats.epoch_execution.closures.saturating_add(1);
                self.stats.epochs_closed += 1;
                self.stats.epochs_attempted += n.epochs_attempted;
                self.stats.epochs_closed += n.epochs_closed;
                self.stats.expansions += n.expansions;
                self.stats.domains_created += n.domains_created;
                self.stats.closed += n.closed;
                self.stats.mono_raw_dips += n.mono_raw_dips;
                self.stats.mono_worst = self.stats.mono_worst.min(n.mono_worst);
                self.stats.policy_execution.absorb(n.policy_execution);
                self.stats.epoch_execution.absorb(n.epoch_execution);
                self.stats.width_execution.absorb(n.width_execution);
                self.stats.max_depth = self
                    .stats
                    .max_depth
                    .max(entry.trunk.len() + entry.heads.len() + n.max_depth);
                // #mr-close-telemetry: the CLOSED subtree's certificates are
                // this domain's proof (its root closes become `epoch_root`).
                // A failed epoch's closes are not, and are never absorbed.
                if close_telemetry_on(&self.cfg) {
                    self.stats.close_integrity.absorb_nested(&n.close_integrity);
                    self.stats.close_integrity.epoch_closes =
                        self.stats.close_integrity.epoch_closes.saturating_add(1);
                }
                true
            }
            Ok(MarginRowOutcome::Unknown { stats, .. }) => {
                self.stats.epoch_execution.nested_unknown =
                    self.stats.epoch_execution.nested_unknown.saturating_add(1);
                self.record_epoch_failure();
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
                    self.stats.epoch_execution.nested_runs_with_stats = self
                        .stats
                        .epoch_execution
                        .nested_runs_with_stats
                        .saturating_add(1);
                    if configured_depth_binding {
                        self.stats
                            .epoch_execution
                            .configured_depth_nested_with_stats = self
                            .stats
                            .epoch_execution
                            .configured_depth_nested_with_stats
                            .saturating_add(1);
                    }
                    self.stats.epochs_attempted += n.epochs_attempted;
                    self.stats.epochs_closed += n.epochs_closed;
                    self.stats.expansions += n.expansions;
                    self.stats.domains_created += n.domains_created;
                    self.stats.closed += n.closed;
                    self.stats.mono_raw_dips += n.mono_raw_dips;
                    self.stats.mono_worst = self.stats.mono_worst.min(n.mono_worst);
                    self.stats.policy_execution.absorb(n.policy_execution);
                    self.stats.epoch_execution.absorb(n.epoch_execution);
                    self.stats.width_execution.absorb(n.width_execution);
                }
                false
            }
            Err(_) => {
                self.stats.epoch_execution.nested_errors =
                    self.stats.epoch_execution.nested_errors.saturating_add(1);
                self.record_epoch_failure();
                false
            }
        }
    }

    /// Candidate selection for one expansion: the legacy shortlist protocol,
    /// or Tier-0 rank-1 ranking when configured (#epoch-bab).
    fn select_candidates(
        &self,
        st: &mut NodeState,
        entry: &DomainEntry,
    ) -> (Vec<(usize, usize)>, Vec<usize>) {
        if !ordinary_width_selectors_enabled(
            self.cfg.epoch_policy.tier0_exact,
            self.cfg.retained.is_some(),
        ) {
            let (trunks, heads, observation) = self.tier0_candidates(st, entry);
            st.epoch_selection = observation;
            st.epoch_execution
                .observe_selection(self.cfg.epoch_frame_route, observation);
            return (trunks, heads);
        }
        let (head_cands, head_available, head_counterfactual) = {
            let _t = super::prof::Timer::start(super::prof::Phase::HeadPrerank);
            self.head_prerank(st, &entry.heads)
        };
        let (trunk_cands, trunk_available, trunk_counterfactual) = {
            let _t = super::prof::Timer::start(super::prof::Phase::TrunkShortlist);
            self.trunk_shortlist(st, &entry.trunk)
        };
        let observation = WidthSelectionObservation {
            head_available,
            head_selected: head_cands.len(),
            head_counterfactual,
            trunk_available,
            trunk_selected: trunk_cands.len(),
            trunk_counterfactual,
        };
        st.width_selection = observation;
        st.width_execution
            .observe_selection(self.cfg.epoch_frame_route, observation);
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
    ) -> (Vec<(usize, usize)>, Vec<usize>, EpochSelectionObservation) {
        let _t = super::prof::Timer::start(super::prof::Phase::Tier0Rank);
        let ret = self.cfg.retained.as_deref().expect("checked by caller");
        let mut observation = EpochSelectionObservation {
            tier0: true,
            ..EpochSelectionObservation::default()
        };
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
            let available = dyn_scored.len();
            let selected = available.min(self.cfg.epoch_policy.tier0_universe);
            let contextual = available.min(64);
            observation.universe_counterfactual |= selected != contextual;
            let retained = lr.idx.len();
            let configured_retained = lr.eligible.min(self.cfg.epoch_policy.retain_per_layer);
            let contextual_retained = lr.eligible.min(128);
            observation.retain_counterfactual |=
                available > 0 && retained == configured_retained && retained != contextual_retained;
            dyn_scored.truncate(self.cfg.epoch_policy.tier0_universe);
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
        let k = self.cfg.epoch_policy.tier0_exact.max(1);
        observation.exact_binding = pool.len() > k;
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
        (trunks, heads, observation)
    }

    /// Trunk shortlist by |margin-row coef| x intercept from the seeded pass.
    fn trunk_shortlist(
        &self,
        st: &NodeState,
        trunk: &[TrunkSplit],
    ) -> (Vec<(usize, usize)>, usize, bool) {
        let used: std::collections::BTreeSet<(usize, usize)> =
            trunk.iter().map(|&(li, pos, _)| (li, pos)).collect();
        let mut natives: Vec<(f64, usize, usize)> = Vec::new();
        let mut contextual_natives: Vec<(f64, usize, usize)> = Vec::new();
        let mut available = 0usize;
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
            available = available.saturating_add(
                sc.iter()
                    .filter(|&&(v, pos)| v.is_finite() && v > 0.0 && !used.contains(&(li, pos)))
                    .count(),
            );
            for &(v, pos) in sc.iter().take(self.cfg.effective_k_trunk.saturating_add(4)) {
                if v.is_finite() && v > 0.0 && !used.contains(&(li, pos)) {
                    natives.push((v, li, pos));
                }
            }
            for &(v, pos) in sc
                .iter()
                .take(self.cfg.effective_contextual_k_trunk.saturating_add(4))
            {
                if v.is_finite() && v > 0.0 && !used.contains(&(li, pos)) {
                    contextual_natives.push((v, li, pos));
                }
            }
        }
        let sort = |values: &mut Vec<(f64, usize, usize)>| {
            values.sort_by(|a, b| {
                b.0.total_cmp(&a.0)
                    .then_with(|| b.1.cmp(&a.1))
                    .then_with(|| b.2.cmp(&a.2))
            })
        };
        sort(&mut natives);
        sort(&mut contextual_natives);
        let selected: Vec<(usize, usize)> = natives
            .into_iter()
            .take(self.cfg.effective_k_trunk)
            .map(|(_, li, pos)| (li, pos))
            .collect();
        let contextual: Vec<(usize, usize)> = contextual_natives
            .into_iter()
            .take(self.cfg.effective_contextual_k_trunk)
            .map(|(_, li, pos)| (li, pos))
            .collect();
        let counterfactual = selected != contextual;
        (selected, available, counterfactual)
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
        let census_score =
            super::census::SiteTimer::start(super::census::Site::ScoreStacked, total);
        let pass = self.eng.run_domain_stacked(
            &Seed { s: seed, e: seed_e },
            &blocks,
            super::engine::LaneDir::Lower,
        )?;
        drop(census_score);
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
            // #branch-parent-lift PERCOL probe — THIS is the scored site: the
            // concurrent cifar100 lane runs the stacked scorer, so probing only
            // the serial one would print nothing on a scored run (the "partial
            // fixes of a set measure exactly zero" trap this file already
            // records for #branch-reduceop).
            let probe = super::parent_lift_enabled();
            let mut percol = if probe {
                vec![f64::NEG_INFINITY; nf]
            } else {
                Vec::new()
            };
            let mut side0 = if probe { vec![0.0f64; nf] } else { Vec::new() };
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
                        let vv = v.max(domain.m1[ri]);
                        if probe {
                            if d_i == 0 {
                                side0[f] = vv;
                            } else {
                                let both = side0[f].min(vv);
                                if both > percol[f] {
                                    percol[f] = both;
                                }
                            }
                        }
                        worst = worst.min(vv);
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
            if probe {
                let percol_lift = percol.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let collapsed = parent_lift(&out);
                note_parent_lift_percol("stacked", nf, out.len(), collapsed, percol_lift);
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
        // B4 (#margin-row-gpu-eft, SITE_SCORE): the ~554 ms batched-children
        // scoring pass — 40.0% of the lane — routes through the certified
        // lane's exception seam. VERDICT-BEARING: the values below become
        // child bounds (`cb = raw.max(b)`, :1612) that close children at pop
        // (:1555) without re-proving, which is why the exception table is
        // fully certified (FIX A1) rather than heuristic. Dark and
        // fail-closed: any refusal is exactly the established `run` call,
        // bit-for-bit. The per-block seeds are margin seeds built against
        // THIS node's y-box, so the lane takes that box's magnitudes.
        let score_y_abs: Vec<f64> = st
            .ybox
            .ly
            .iter()
            .zip(&st.ybox.uy)
            .map(|(l, u)| l.abs().max(u.abs()))
            .collect();
        let lane_ctx = super::gpu_backward::LaneCtx {
            y_abs: Some(&score_y_abs),
            deadline: self.cfg.deadline,
            plan_cache: Some(&self.lane_cache),
        };
        let census_score = super::census::SiteTimer::start(super::census::Site::Score, total);
        let pass = self.eng.run_exc_seamed(
            &Seed { s: seed, e: seed_e },
            Some(&st.dom),
            super::engine::LaneDir::Lower,
            &exc,
            &lane_ctx,
        )?;
        drop(census_score);
        let low = self.eng.concretize_lower(&pass);
        let mut out = Vec::with_capacity(n_t + n_h);
        // #branch-parent-lift PERCOL probe (see `note_parent_lift_percol`):
        // per COLUMN, the best over candidates of that candidate's worse child.
        // Read-only bookkeeping over values the fold below already validated.
        let probe = super::parent_lift_enabled();
        let mut percol = if probe {
            vec![f64::NEG_INFINITY; nf]
        } else {
            Vec::new()
        };
        let mut side0 = if probe { vec![0.0f64; nf] } else { Vec::new() };
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
                    let vv = v.max(m1[r0 + f]);
                    if probe {
                        if d_i == 0 {
                            side0[f] = vv;
                        } else {
                            let both = side0[f].min(vv);
                            if both > percol[f] {
                                percol[f] = both;
                            }
                        }
                    }
                    worst = worst.min(vv);
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
        if probe {
            let percol_lift = percol.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let collapsed = parent_lift(&out);
            note_parent_lift_percol("serial", nf, out.len(), collapsed, percol_lift);
        }
        Ok(Some(out))
    }
}

#[cfg(test)]
mod reduceop_tests {
    use super::pick_candidate;

    /// TEETH: the two reductions must actually be able to disagree, or the
    /// lever is decorative and every A/B on it is unfalsifiable. Candidate 0
    /// is balanced, candidate 1 is lopsided: `min` prefers 0 (worse child
    /// -0.10 beats -0.50), `max` prefers 1 (better child +0.40 beats -0.05).
    #[test]
    fn the_two_reductions_can_disagree_and_the_flag_reports_it() {
        let ch = [(-0.10_f64, -0.05_f64), (-0.50_f64, 0.40_f64)];
        let (pick_min, differs_min) = pick_candidate(&ch, false);
        let (pick_max, differs_max) = pick_candidate(&ch, true);
        assert_eq!(pick_min, 0, "min must rank by the WORSE child");
        assert_eq!(pick_max, 1, "max must rank by the BETTER child");
        assert!(differs_min && differs_max, "differs must be true both ways");
    }

    /// And they must AGREE when one candidate dominates on both children —
    /// that is the case where a `differs=0` telemetry line is a REAL null
    /// about the row rather than evidence of an inert lever.
    #[test]
    fn a_dominating_candidate_makes_both_reductions_agree() {
        let ch = [(-0.90_f64, -0.80_f64), (-0.10_f64, 0.20_f64)];
        for use_max in [false, true] {
            let (pick, differs) = pick_candidate(&ch, use_max);
            assert_eq!(pick, 1);
            assert!(!differs, "dominated candidate sets must not report differs");
        }
    }

    /// Ties resolve to the LATER index in both modes (np.lexsort parity), the
    /// behaviour every pick site had before they were unified.
    #[test]
    fn ties_resolve_to_the_later_index_in_both_modes() {
        let ch = [(-0.20_f64, -0.10_f64), (-0.20_f64, -0.10_f64)];
        for use_max in [false, true] {
            assert_eq!(pick_candidate(&ch, use_max).0, 1);
        }
    }
}

#[cfg(test)]
mod ledger_tests {
    use super::{
        closed, require_finite, BabConfig, ChildSplit, HeadFix, Ledger, RoundMode, TrunkSplit,
    };

    #[test]
    fn incremental_epoch_generation_budget_never_resets_in_nested_drivers() {
        let mut cfg = BabConfig {
            epoch_policy: super::super::EpochRuntimePolicy {
                depth: 3,
                parent_topk: 16,
                ..super::super::EpochRuntimePolicy::default()
            },
            epoch_trigger_depth: 3,
            epoch_attempt_cap: 2,
            epoch_parent_generations_left: 4,
            ..BabConfig::default()
        };
        for want_left in [3usize, 2, 1, 0] {
            let (depth, attempts, left) = cfg.nested_epoch_limits(true);
            assert_eq!(left, want_left);
            if want_left == 0 {
                assert_eq!((depth, attempts), (0, 0));
            } else {
                assert_eq!((depth, attempts), (3, 2));
            }
            cfg.epoch_trigger_depth = depth;
            cfg.epoch_attempt_cap = attempts;
            cfg.epoch_parent_generations_left = left;
        }
        assert_eq!(cfg.nested_epoch_limits(true), (0, 0, 0));
        assert_eq!(
            BabConfig {
                epoch_trigger_depth: 3,
                epoch_attempt_cap: 2,
                epoch_parent_generations_left: usize::MAX,
                ..BabConfig::default()
            }
            .nested_epoch_limits(true),
            (3, 2, 3),
            "an oversized public config is clamped to the hard four-frame cap before nesting"
        );
        assert_eq!(
            BabConfig {
                epoch_parent_generations_left: usize::MAX,
                ..BabConfig::default()
            }
            .nested_epoch_limits(false),
            (0, 0, 0),
            "the historical full epoch must stay one nesting level"
        );
    }

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

    /// #soundness-spine M2c: a margin-row tree recorded through the lane's
    /// own IR (trunk splits, head fixes, child splits) proves its cover.
    #[test]
    fn cover_ledger_accepts_complete_margin_row_trees() {
        use crate::certify::margin_row::{bound_evidence, empty_evidence, leaf_path};
        let mode = RoundMode::Outward;
        // Root split on head 3; the ON side splits trunk (0, 2); the OFF side
        // closes as a head-infeasible child, and the ON side's children close
        // as a scored child and an evaluated entry respectively.
        let mut l = Ledger::default();
        l.cover.leaf(
            leaf_path(&[], &[], &[], Some(ChildSplit::Head((3, -1)))),
            empty_evidence(mode, 1e-3),
        );
        l.cover.leaf(
            leaf_path(&[(0, 2, 1)], &[(3, 1)], &[], None),
            bound_evidence(mode, 0.25),
        );
        l.cover.leaf(
            leaf_path(&[], &[(3, 1)], &[], Some(ChildSplit::Trunk((0, 2, -1)))),
            bound_evidence(mode, 1e-9),
        );
        let cover = l.cover.prove(&[4, 7]).expect("complete tree");
        assert_eq!(cover.leaves(), 3);
        assert_eq!(cover.empty_leaves(), 1);
        assert_eq!(cover.min_margin().to_bits(), 1e-9f64.to_bits());
        // A nested frame records BELOW its inherited heads.
        let mut nested = Ledger::default();
        let root_heads = [(3usize, 1i8)];
        nested.cover.leaf(
            leaf_path(&[(1, 0, 1)], &root_heads, &root_heads, None),
            bound_evidence(mode, 0.5),
        );
        nested.cover.leaf(
            leaf_path(&[(1, 0, -1)], &root_heads, &root_heads, None),
            bound_evidence(mode, 0.5),
        );
        assert!(nested.cover.proves_cover(&[4]));
    }

    /// The depth-only Kraft ledger this replaces accepted both of these; the
    /// path gate refuses both, and refuses the ordinary incomplete shapes.
    #[test]
    fn cover_ledger_refuses_incomplete_margin_row_trees() {
        use crate::certify::margin_row::{bound_evidence, leaf_path};
        let mode = RoundMode::Outward;
        let record = |leaves: &[(Vec<TrunkSplit>, Vec<HeadFix>)]| {
            let mut l = Ledger::default();
            for (trunk, heads) in leaves {
                l.cover.leaf(
                    leaf_path(trunk, heads, &[], None),
                    bound_evidence(mode, 0.5),
                );
            }
            l
        };
        let different_sites = record(&[(vec![(0, 5, -1)], vec![]), (vec![(0, 7, 1)], vec![])]);
        assert!(different_sites.cover.kraft_ok());
        assert!(!different_sites.cover.proves_cover(&[1]));
        let duplicate = record(&[
            (vec![(0, 1, 1), (0, 2, 1)], vec![]),
            (vec![(0, 1, 1), (0, 2, 1)], vec![]),
            (vec![(0, 1, -1)], vec![]),
        ]);
        assert!(duplicate.cover.kraft_ok());
        assert!(!duplicate.cover.proves_cover(&[1]));
        let missing = record(&[(vec![(0, 1, 1)], vec![])]);
        assert!(!missing.cover.proves_cover(&[1]));
        let empty = Ledger::default();
        assert!(!empty.cover.proves_cover(&[1]));
        // An uncertified close (zero margin) is not a leaf.
        let mut zero = Ledger::default();
        zero.cover.leaf(
            leaf_path(&[(0, 1, 1)], &[], &[], None),
            bound_evidence(mode, 0.0),
        );
        zero.cover.leaf(
            leaf_path(&[(0, 1, -1)], &[], &[], None),
            bound_evidence(mode, 0.5),
        );
        assert!(!zero.cover.proves_cover(&[1]));
    }

    /// Only the outer frame may speak for the instance, and an unscoped
    /// ledger can never publish.
    #[test]
    fn frame_scope_is_instance_only_on_the_outer_frame() {
        use super::{frame_scope, EpochFrameRoute, TokenScope};
        let mut cfg = BabConfig {
            epoch_frame_route: EpochFrameRoute::Outer,
            ..BabConfig::default()
        };
        assert_eq!(frame_scope(&cfg), TokenScope::Instance);
        cfg.epoch_frame_route = EpochFrameRoute::NestedFull;
        assert_eq!(frame_scope(&cfg), TokenScope::Subdomain);
        cfg.epoch_frame_route = EpochFrameRoute::NestedIncremental;
        assert_eq!(frame_scope(&cfg), TokenScope::Subdomain);
        assert_eq!(Ledger::default().scope, TokenScope::Subdomain);
    }

    /// Parity values no longer close (no closure predicate), while the root
    /// SCHEDULE in Parity keeps its historical `>= 0` rule bit-for-bit.
    #[test]
    fn parity_never_closes_but_keeps_its_root_schedule() {
        use super::root_schedules_closed;
        for v in [0.0, -0.0, 1e-300, 0.5, f64::MAX] {
            assert!(!closed(RoundMode::Parity, v), "parity closed {v:e}");
            assert!(root_schedules_closed(RoundMode::Parity, v));
        }
        for v in [-1e-300, -0.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(!root_schedules_closed(RoundMode::Parity, v));
        }
        for v in [0.0, -0.0, f64::NAN, f64::INFINITY, -1.0] {
            assert!(!closed(RoundMode::Outward, v));
            assert!(!root_schedules_closed(RoundMode::Outward, v));
        }
        for v in [f64::from_bits(1), 1e-12, 0.5] {
            assert!(closed(RoundMode::Outward, v));
            assert!(root_schedules_closed(RoundMode::Outward, v));
        }
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
                    bias_err: Vec::new(),
                    weight_rel_err: 0.0,
                    shape: (n_y, n_in),
                }, // t3 = y
                TwinOpSpec::Relu { input: 3 },    // t4, head relu
                TwinOpSpec::Gemm {
                    input: 4,
                    weight: wo,
                    bias: vec![0.0, 0.1, -0.1],
                    bias_err: Vec::new(),
                    weight_rel_err: 0.0,
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

    use crate::margin_row::bounds::head_gates;

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
    pub(super) fn spec4() -> TwinSpec {
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
                    bias_err: Vec::new(),
                    weight_rel_err: 0.0,
                    shape: (N_Y, N_IN),
                },
                TwinOpSpec::Relu { input: 3 },
                TwinOpSpec::Gemm {
                    input: 4,
                    weight: wo,
                    bias: vec![0.0, -4.0, 0.0, -0.05],
                    bias_err: Vec::new(),
                    weight_rel_err: 0.0,
                    shape: (N_OUT, N_Y),
                },
            ],
        }
    }

    pub(super) fn build(net: &TwinNet) -> (Vec<f64>, Vec<f64>, RootGates) {
        let lo = vec![-0.4; N_IN];
        let hi = vec![0.4; N_IN];
        let root = RootGates::build(net, &lo, &hi, RoundMode::Outward, None).expect("root");
        (lo, hi, root)
    }

    pub(super) fn driver<'a>(
        net: &'a TwinNet,
        root: &'a RootGates,
        adv: &[usize],
        col_retire: bool,
    ) -> (MarginRowBab<'a>, Arc<YPack>, f64) {
        let eng = BackwardEngine::new(net, root);
        let re = root_eval(&eng, net, 0, adv).expect("root_eval");
        let root_bound = re.dj.iter().copied().fold(f64::INFINITY, f64::min);
        let cfg = BabConfig {
            lru_cap: 8,
            col_retire,
            ..BabConfig::default()
        };
        let policy_execution = PolicyExecutionEvidence::from_config(&cfg);
        let epoch_execution = EpochExecutionEvidence::from_config(&cfg);
        let width_execution = WidthExecutionEvidence::from_config(&cfg);
        let mut bab = MarginRowBab {
            eng: BackwardEngine::new(net, root),
            mb: MarginBatch::new(net, 0, adv).expect("mb"),
            cfg,
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
                frontier_execution: FrontierExecutionEvidence::joint(1, 1),
                policy_execution,
                epoch_execution,
                width_execution,
                root_precision_execution: Default::default(),
                backward_interm_execution: Default::default(),
                root_alpha_execution: Default::default(),
                close_integrity: CloseIntegrityEvidence::default(),
            },
            epoch_failures: 0,
            ledger: Ledger::default(),
            nested_ledgers_ok: true,
            lane_cache: crate::margin_row::gpu_backward::cache::PlanCache::default(),
            closure: crate::margin_row::closure::ClosureMonitor::new(false),
            b5: stack_eval::B5Trace::default(),
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
            alpha: Vec::new(),
            clip_acc: None,
            split_order: 0,
        }
    }

    #[test]
    fn frozen_domain_alpha_policy_drives_eval_and_closes_its_evidence_algebra() {
        let spec = spec4();
        let net = TwinNet::compile(&spec).expect("compiles");
        let (_lo, _hi, root) = build(&net);
        let adv = [1usize, 2, 3];
        let (mut bab, pack, root_bound) = driver(&net, &root, &adv, false);
        bab.cfg.domain_alpha_policy = super::super::DomainAlphaRuntimePolicy {
            enabled: true,
            iters: 4,
            eta: 0.5,
            min_depth: 0,
            stride: 1,
            gap: f64::INFINITY,
        };
        let entry = root_entry_over(&pack, root_bound, Vec::new());
        let EvalOut::Node(_, state) = bab.eval_with_pack(&pack, &entry).expect("eval") else {
            panic!("root domain must evaluate")
        };
        let evidence = state.policy_execution;
        assert!(evidence.domain_alpha_enabled);
        assert_eq!(evidence.domain_alpha_iters, 4);
        assert_eq!(evidence.domain_alpha_eta_bits, 0.5f64.to_bits());
        assert_eq!(evidence.domain_alpha_gate_evals, 1);
        assert_eq!(evidence.domain_alpha_movable_evals, 1);
        assert!(evidence.domain_alpha_movable_slots > 0);
        assert!(evidence.domain_alpha_ascent_calls <= evidence.domain_alpha_movable_evals);
        assert_eq!(
            evidence.domain_alpha_trials_started,
            evidence.domain_alpha_trial_passes_completed
                + evidence.domain_alpha_trial_pass_failures
        );
        assert!(
            evidence.domain_alpha_finite_trial_scores
                <= evidence.domain_alpha_trial_passes_completed
        );
        assert!(evidence.domain_alpha_trials_accepted <= evidence.domain_alpha_finite_trial_scores);
        assert_eq!(
            evidence.domain_alpha_outer_ascent_calls,
            evidence.domain_alpha_ascent_calls
        );
        assert_eq!(
            evidence.domain_alpha_outer_finite_trial_scores,
            evidence.domain_alpha_finite_trial_scores
        );
        assert_eq!(evidence.domain_alpha_iters_counterfactual_domains, 1);
        assert_eq!(evidence.domain_alpha_eta_counterfactual_scored_domains, 1);
        assert_eq!(
            evidence.domain_alpha_outer_iters_counterfactual_domains,
            evidence.domain_alpha_iters_counterfactual_domains
        );
        assert_eq!(
            evidence.domain_alpha_outer_eta_counterfactual_scored_domains,
            evidence.domain_alpha_eta_counterfactual_scored_domains
        );
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
        assert!(
            st.policy_execution.column_retire,
            "the evaluation evidence lost the armed retirement policy"
        );
        assert_eq!(
            st.policy_execution.newly_retired_columns,
            st.retired.len(),
            "the empty root retirement set must account for every new certificate"
        );
        assert_eq!(
            st.policy_execution.inherited_retirement_evals, 0,
            "the root has no inherited retirement to narrow"
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
        assert!(
            !closed(mode, b.max(entry.bound)),
            "the fixture root must stay open to exercise child inheritance"
        );
        let ExpandStep::Expanded { pushes, .. } = bab.expand_one(&pack, &entry).expect("expand")
        else {
            panic!("the fixture root must expand")
        };
        assert!(!pushes.is_empty(), "open root must push children");
        let mut narrowed_children = 0usize;
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
                clip_acc: child.clip_acc.clone(),
                alpha: child.alpha.clone(),
                split_order: child.split_order,
            };
            if let EvalOut::Node(_, st2) = bab
                .eval_with_pack(&child_pack, &centry)
                .expect("child eval")
            {
                narrowed_children += 1;
                // Engagement observable: the child's pass is narrowed to the
                // survivors and its per-eval evidence accounts for that work.
                assert_eq!(
                    st2.ms.seed.s.ncols(),
                    adv.len() - centry.retired.len(),
                    "child eval did not narrow to the survivors"
                );
                assert!(st2.policy_execution.column_retire);
                assert_eq!(
                    st2.policy_execution.inherited_retirement_evals, 1,
                    "one narrowed child evaluation must count once"
                );
                for c in &centry.retired {
                    assert!(st2.retired.contains(c), "child dropped {c}");
                }
                assert!(st2.retired.len() >= centry.retired.len());
                assert_eq!(
                    st2.policy_execution.newly_retired_columns,
                    st2.retired.len() - centry.retired.len(),
                    "child retirement certificates were not counted exactly"
                );
            }
        }
        assert!(
            narrowed_children > 0,
            "the fixture never produced a feasible narrowed child"
        );
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
        assert!(!st.policy_execution.column_retire);
        assert_eq!(st.policy_execution.inherited_retirement_evals, 0);
        assert_eq!(st.policy_execution.newly_retired_columns, 0);
    }
}

/// #closure-adaptive TEETH.
///
/// The policy's contract is that it selects INPUTS to the certified pass and
/// never produces a bound. The one place that contract could be broken without
/// anybody noticing is the y-pack cache: a pack is a certified enclosure, it is
/// keyed by trunk-set alone, and a tier change alters what `build_pack` would
/// produce for that same key. If a pack could survive a tier change, the lane
/// would evaluate a domain against numbers no pass scored under the tier in
/// force. These tests fail if that becomes possible.
#[cfg(test)]
mod closure_adaptive_tests {
    use super::col_retire_tests::{build, driver, spec4};
    use super::*;
    use crate::margin_row::closure::{
        tier, ClosureMonitor, Decision, TIER_BASE, TIER_BETA, TIER_TOP,
    };
    use crate::margin_row::net::TwinNet;

    /// A y-pack whose lower bounds are `+1e9` above anything the lane could
    /// certify: if this ever reaches an evaluation, an unscored bound was
    /// published.
    ///
    /// Built by REBUILDING the pack and mutating it rather than by cloning one.
    /// `YPack` deliberately has no `Clone`: it carries the two pass matrices and
    /// their row dots, and a derive would let a hot path clone them by accident
    /// — a throughput regression that is very hard to find again.
    fn poisoned_pack(bab: &MarginRowBab<'_>, trunk: &[TrunkSplit]) -> YPack {
        let dom = (!trunk.is_empty()).then(|| domain_gates(bab.eng.root, trunk));
        let mut p = build_pack(&bab.eng, dom.as_ref(), trunk).expect("pack");
        for v in &mut p.ly0 {
            *v += 1e9;
        }
        p
    }

    #[test]
    fn a_pack_the_certified_pass_did_not_score_cannot_survive_a_tier_change() {
        let net = TwinNet::compile(&spec4()).expect("compiles");
        let (_lo, _hi, root) = build(&net);
        let (mut bab, root_pack, _rb) = driver(&net, &root, &[1, 2, 3], false);

        // A domain that carries a split, so its pack is tier-DEPENDENT (a
        // split-free pack is tier-invariant and is the one we re-seed).
        let key: Vec<TrunkSplit> = vec![(0, 0, 1)];
        let honest = bab.rows_for(&key).expect("pack");
        let honest_ly = honest.ly0.clone();

        // THE HAZARD IS REAL: the cache is load-bearing, so a wrong pack in it
        // is a wrong bound out of it. (If this assert ever fails the rest of
        // the test proves nothing, so it is asserted rather than assumed.)
        let bad = poisoned_pack(&bab, &key);
        bab.lru.put(key.clone(), Arc::new(bad));
        let served = bab.rows_for(&key).expect("pack");
        assert!(
            served.ly0[0] > honest_ly[0] + 1e8,
            "the LRU must actually serve what was put in it, or this test is vacuous"
        );

        // THE TEETH: a tier change must retire it.
        bab.retier_packs(TIER_TOP, &root_pack);
        let after = bab.rows_for(&key).expect("pack");
        for (i, (a, b)) in after.ly0.iter().zip(&honest_ly).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "row {i}: a tier change left a pack the certified pass did not score \
                 (got {a}, certified {b})"
            );
        }
    }

    #[test]
    fn the_reseeded_root_pack_is_bit_identical_to_the_certified_root_pack() {
        let net = TwinNet::compile(&spec4()).expect("compiles");
        let (_lo, _hi, root) = build(&net);
        let (mut bab, root_pack, _rb) = driver(&net, &root, &[1, 2, 3], false);

        // `retier_packs` is the ONE site that inserts a pack without rebuilding
        // it. Poison the root slot first so a no-op re-seed cannot pass.
        let bad = poisoned_pack(&bab, &[]);
        bab.lru.put(Vec::new(), Arc::new(bad));
        bab.retier_packs(TIER_BETA, &root_pack);
        let after = bab.rows_for(&[]).expect("root pack");
        for (i, (a, b)) in after.ly0.iter().zip(&root_pack.ly0).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "root pack ly0 row {i} moved");
        }
        for (i, (a, b)) in after.uy0.iter().zip(&root_pack.uy0).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "root pack uy0 row {i} moved");
        }
    }

    /// The observer takes counters and clocks and NOTHING else — no bound, no
    /// pack, no gates. That is what makes "the policy cannot publish a bound" a
    /// structural claim rather than a review promise, and this pins it
    /// behaviourally: identical counters, identical decision, whatever the
    /// tree's bounds happen to be.
    #[test]
    fn the_decision_is_a_function_of_the_counters_alone() {
        let mut a = ClosureMonitor::with_params(8, 8, 0.40, TIER_BETA);
        let mut b = ClosureMonitor::with_params(8, 8, 0.40, TIER_BETA);
        crate::margin_row::closure::reset();
        // created 16 / closed 4 = window closure 0.25, inside the escalation
        // BAND that 756e2df14 introduced (`ESCALATION_CLOSURE_FLOOR <= closure
        // < thresh`). The old `closed = 0` driver now lands on the "stalled
        // below the escalation floor" refusal, which would leave this test
        // comparing two Holds and no longer witnessing that an ESCALATION --
        // the decision that moves the tier -- is determined by the counters
        // alone.
        let da = a.observe(8, 16, 4, 16, 0.8, Some(500.0));
        crate::margin_row::closure::reset();
        let db = b.observe(8, 16, 4, 16, 0.8, Some(500.0));
        assert_eq!(da, db);
        assert_eq!(da, Decision::Escalate(TIER_BETA));
        crate::margin_row::closure::reset();
        assert_eq!(tier(), TIER_BASE);
    }
}

#[cfg(test)]
mod publish_deadline_tests {
    use super::BabConfig;
    use std::time::{Duration, Instant};

    /// `None` must fall back to the SEARCH deadline — byte-identical to the
    /// behaviour before `#publish-deadline` existed.
    #[test]
    fn publish_deadline_none_falls_back_to_the_search_deadline() {
        let now = Instant::now();
        let cfg = BabConfig {
            deadline: Some(now),
            publish_deadline: None,
            ..BabConfig::default()
        };
        assert_eq!(cfg.effective_publish_deadline(), Some(now));
        let none = BabConfig {
            deadline: None,
            publish_deadline: None,
            ..BabConfig::default()
        };
        assert_eq!(none.effective_publish_deadline(), None);
    }

    /// When set, it must WIN over the search deadline — that is the whole point:
    /// a completed proof gets a later instant than the search did.
    #[test]
    fn publish_deadline_overrides_and_is_later_than_the_search_deadline() {
        let now = Instant::now();
        let cfg = BabConfig {
            deadline: Some(now),
            publish_deadline: Some(now + Duration::from_secs(2)),
            ..BabConfig::default()
        };
        let eff = cfg.effective_publish_deadline().expect("set");
        assert!(
            eff > now,
            "a completed proof must get MORE time than the search did, or the \
             38% of runs that finish their tree and lose the verdict stay lost"
        );
        assert_eq!(eff, now + Duration::from_secs(2));
    }

    /// The grace must never eat the caller's whole wrap-up reserve. The caller
    /// sets the lane deadline to `scored - 3s`; a grace >= 3s would push
    /// publication past the SCORED deadline, which is the overrun this reserve
    /// exists to prevent.
    #[test]
    fn the_publish_grace_constant_stays_inside_the_callers_reserve() {
        let g = crate::margin_row::publish_grace();
        if let Some(d) = g {
            assert!(
                d < Duration::from_secs(3),
                "publish grace {d:?} must stay strictly below the caller's 3 s \
                 wrap-up reserve (margin_row_bab.rs:2681)"
            );
        }
        // Shipped default is inert: a Some() grace implies a positive
        // constant or the acting router's armed authority (#router-r2 CR-R).
        assert!(
            g.is_none()
                || crate::margin_row::PUBLISH_GRACE_SECS > 0.0
                || ny_core::deadline_authority::global().is_armed(),
            "a Some() grace implies a positive constant or an armed authority"
        );
    }
}

/// #root-eval-reuse (throughput plan B1(c)): the L1 root-eval dedupe is
/// BIT-IDENTICAL to recomputing, and every in-place gate write invalidates it.
#[cfg(test)]
mod root_eval_reuse_tests {
    use super::col_retire_tests::{build, spec4};
    use super::*;

    fn eq_bits(a: &[f64], b: &[f64], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length moved");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{what}[{i}] moved: {x} vs {y}");
        }
    }

    fn eq_pass(a: &PassOut, b: &PassOut, what: &str) {
        assert_eq!(a.a.raw_dim(), b.a.raw_dim(), "{what}: shape moved");
        eq_bits(
            a.a.as_slice().expect("standard layout"),
            b.a.as_slice().expect("standard layout"),
            what,
        );
        match (&a.e, &b.e) {
            (Some(x), Some(y)) => eq_bits(
                x.as_slice().expect("standard layout"),
                y.as_slice().expect("standard layout"),
                what,
            ),
            (None, None) => {}
            _ => panic!("{what}: error-lane presence moved"),
        }
        eq_bits(&a.b, &b.b, what);
        eq_bits(&a.eb, &b.eb, what);
    }

    fn eq_root_eval(a: &RootEval, b: &RootEval) {
        eq_bits(&a.dj, &b.dj, "dj");
        assert_eq!(a.tree_classes, b.tree_classes, "tree_classes moved");
        assert!(a.spec_alpha.is_none() && b.spec_alpha.is_none());
        let (p, q) = (&a.pack, &b.pack);
        eq_bits(&p.ly0, &q.ly0, "ly0");
        eq_bits(&p.uy0, &q.uy0, "uy0");
        eq_bits(&p.ly_fwd, &q.ly_fwd, "ly_fwd");
        eq_bits(&p.uy_fwd, &q.uy_fwd, "uy_fwd");
        eq_pass(&p.al, &q.al, "al");
        eq_pass(&p.au, &q.au, "au");
        eq_bits(&p.al_dots.abs_dot, &q.al_dots.abs_dot, "al_dots.abs");
        eq_bits(&p.al_dots.err_dot, &q.al_dots.err_dot, "al_dots.err");
        eq_bits(&p.au_dots.abs_dot, &q.au_dots.abs_dot, "au_dots.abs");
        eq_bits(&p.au_dots.err_dot, &q.au_dots.err_dot, "au_dots.err");
        assert_eq!(p.head_slack.is_some(), q.head_slack.is_some());
    }

    /// `root_eval_reuse_bit_identical`: a stored evaluation served to the
    /// tree (`root_eval_reuse`) and to the ascent's frozen frame
    /// (`root_rows_memo`) is bit-for-bit what recomputing produces; the
    /// property is part of the key; an in-place alpha/ms write (what the
    /// ascent's commit does) and a head-alpha install (what #head-alpha does)
    /// both invalidate it; `run_inner`'s reuse releases the memo.
    #[test]
    fn root_eval_reuse_bit_identical() {
        let net = TwinNet::compile(&spec4()).expect("compile");
        let (_lo, _hi, mut root) = build(&net);
        let adv = [1usize, 2, 3];
        // The oracle: an unmemoized evaluation.
        let fresh = {
            let eng = BackwardEngine::new(&net, &root);
            root_eval(&eng, &net, 0, &adv).expect("fresh root_eval")
        };
        {
            let eng = BackwardEngine::new(&net, &root);
            if !memo_eligible(&eng, &net) {
                // A GPU lane is armed in this process: the memo must REFUSE,
                // never serve — checked first. The bit-identity half cannot run
                // here, and a test that returns early prints `... ok` exactly
                // like a real pass (libtest captures the note), which is the
                // VOID-read-as-PASS shape CLAUDE.md forbids. So it FAILS
                // CLOSED: rerun without the NY_MARGIN_ROW_GPU* arming.
                let _ = root_eval_store(&eng, &net, 0, &adv).expect("store");
                assert!(
                    root_eval_lookup(&eng, &net, 0, &adv).is_none(),
                    "an armed GPU lane must make the memo refuse"
                );
                panic!(
                    "root_eval_reuse_bit_identical: VOID — a GPU margin-row lane is armed in \
                     this test process, so the bit-identity half cannot run (the refusal \
                     half passed). Rerun with the GPU lanes dark."
                );
            }
        }
        // (1) The production store path (`certified_root_bound` = the alpha
        // guard's published_before) returns the same number and stores.
        let bound = crate::margin_row::rebake::certified_root_bound(&net, &root, 0, &adv)
            .expect("certified_root_bound");
        let fresh_min = fresh.dj.iter().copied().fold(f64::INFINITY, f64::min);
        assert_eq!(
            bound.to_bits(),
            fresh_min.to_bits(),
            "the store moved the bound"
        );
        {
            let eng = BackwardEngine::new(&net, &root);
            // (2) A hit, bit-identical to the oracle.
            let hit = root_eval_lookup(&eng, &net, 0, &adv)
                .expect("an unchanged gates generation must hit");
            eq_root_eval(&hit, &fresh);
            // (3) The ascent's frozen y-rows ARE `y_rows(None)`.
            let rows = root_rows_memo(&net, &root).expect("rows hit");
            let (al, au) = eng.y_rows(None).expect("y_rows");
            eq_pass(rows.al(), &al, "rows.al");
            eq_pass(rows.au(), &au, "rows.au");
            // (4) The property is part of the key.
            assert!(
                root_eval_lookup(&eng, &net, 0, &[1, 2]).is_none(),
                "a different adv set must miss"
            );
        }
        // (5) An in-place gate write (the ascent's commit) invalidates.
        let (li, j) = root
            .layers
            .iter()
            .enumerate()
            .find_map(|(li, l)| l.unst.first().map(|&j| (li, j)))
            .expect("spec4's trunk relu has unstable neurons");
        let saved = (root.layers[li].alpha[j], root.layers[li].ms[j]);
        root.layers[li].alpha[j] = if saved.0 == 0.75 { 0.25 } else { 0.75 };
        root.layers[li].ms[j] = root.layers[li].alpha[j].max(root.layers[li].s[j]);
        {
            let eng = BackwardEngine::new(&net, &root);
            assert!(
                root_eval_lookup(&eng, &net, 0, &adv).is_none(),
                "an in-place alpha/ms write must move the gates generation"
            );
            assert!(root_rows_memo(&net, &root).is_none());
            // The tree's path then recomputes: identical to an oracle on the
            // mutated gates.
            let via_reuse = root_eval_reuse(&eng, &net, 0, &adv).expect("reuse (miss)");
            let oracle = root_eval(&eng, &net, 0, &adv).expect("oracle");
            eq_root_eval(&via_reuse, &oracle);
        }
        // (6) Restoring the gates does NOT resurrect a hit: `root_eval_reuse`
        // released the memo. A new store then hits again (the rollback case of
        // the guard's before/after pair), bit-identically.
        root.layers[li].alpha[j] = saved.0;
        root.layers[li].ms[j] = saved.1;
        {
            let eng = BackwardEngine::new(&net, &root);
            assert!(
                root_eval_lookup(&eng, &net, 0, &adv).is_none(),
                "run_inner's reuse must release the memo"
            );
            let stored = root_eval_store(&eng, &net, 0, &adv).expect("store");
            eq_root_eval(&stored, &fresh);
            let via_reuse = root_eval_reuse(&eng, &net, 0, &adv).expect("reuse (hit)");
            eq_root_eval(&via_reuse, &fresh);
        }
        // (7) A head-alpha install (#head-alpha's commit) invalidates.
        {
            let eng = BackwardEngine::new(&net, &root);
            let _ = root_eval_store(&eng, &net, 0, &adv).expect("store");
        }
        root.head_alpha = Some(vec![0.5; net.n_y]);
        let eng = BackwardEngine::new(&net, &root);
        assert!(
            root_eval_lookup(&eng, &net, 0, &adv).is_none(),
            "a head-alpha install must move the gates generation"
        );
    }

    /// `root_eval_reuse_generation_covers_every_read_field`: each field the
    /// fingerprint claims to cover actually moves it (a field silently dropped
    /// from `gates_generation` would turn the memo stale on that field).
    #[test]
    fn root_eval_reuse_generation_covers_every_read_field() {
        let net = TwinNet::compile(&spec4()).expect("compile");
        let (_lo, _hi, mut root) = build(&net);
        let base = gates_generation(&root);
        let moved = |what: &str, root: &RootGates| {
            assert_ne!(
                gates_generation(root),
                base,
                "{what} is not in the gates generation"
            );
        };
        let li = 0usize;
        let bump = |v: &mut f64| *v = f64::from_bits(v.to_bits() ^ 1);
        for field in 0..6 {
            let layer = &mut root.layers[li];
            let vec = match field {
                0 => &mut layer.l,
                1 => &mut layer.u,
                2 => &mut layer.alpha,
                3 => &mut layer.s,
                4 => &mut layer.c,
                _ => &mut layer.ms,
            };
            let old = vec[0];
            bump(&mut vec[0]);
            moved(&format!("layers[0] field #{field}"), &root);
            let layer = &mut root.layers[li];
            let vec = match field {
                0 => &mut layer.l,
                1 => &mut layer.u,
                2 => &mut layer.alpha,
                3 => &mut layer.s,
                4 => &mut layer.c,
                _ => &mut layer.ms,
            };
            vec[0] = old;
            assert_eq!(gates_generation(&root), base, "restore must restore");
        }
        let old = root.mid[0];
        bump(&mut root.mid[0]);
        moved("mid", &root);
        root.mid[0] = old;
        let old = root.rad[0];
        bump(&mut root.rad[0]);
        moved("rad", &root);
        root.rad[0] = old;
        let old = root.xabs[0];
        bump(&mut root.xabs[0]);
        moved("xabs", &root);
        root.xabs[0] = old;
        let popped = root.layers[li].unst.pop();
        moved("unst", &root);
        if let Some(p) = popped {
            root.layers[li].unst.push(p);
        }
        root.head_alpha = Some(vec![0.0; net.n_y]);
        moved("head_alpha", &root);
        root.head_alpha = None;
        assert_eq!(gates_generation(&root), base);
    }
}
