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
    /// Wall-clock deadline.
    pub deadline: Option<Instant>,
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
    /// Default-off cross-domain candidate-score stacking canary.  Only the
    /// `score_candidates` backward passes of a parallel frontier batch are
    /// stacked; node evaluation, candidate selection, and heap merge retain
    /// their established per-domain paths and deterministic ordering.
    pub domain_stack: bool,
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
    /// Root-build execution evidence for the root-precision family.
    pub root_precision_execution: super::RootPrecisionExecutionEvidence,
    /// Root-build execution evidence for the frozen backward-intermediate
    /// family. Evidence-only; no tree decision reads it.
    pub backward_interm_execution: super::BackwardIntermExecutionEvidence,
    /// One closed receipt for the entry root's standalone alpha and rebake
    /// phases. Evidence-only and never recomputed by nested class/tree runs.
    pub root_alpha_execution: super::RootAlphaExecutionEvidence,
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
            lru_cap: 6,
            frontier: 1,
            frontier_contextual: 1,
            domain_stack: false,
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
            root_precision_execution: super::RootPrecisionExecutionEvidence::default(),
            backward_interm_execution: super::BackwardIntermExecutionEvidence::default(),
            root_alpha_execution: super::RootAlphaExecutionEvidence::default(),
        }
    }
}

impl BabConfig {
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
    /// Frozen per-domain alpha gate.
    pub domain_alpha_enabled: bool,
    /// Frozen per-domain alpha trial cap.
    pub domain_alpha_iters: usize,
    /// Frozen initial per-domain alpha step, stored as exact f64 bits.
    pub domain_alpha_eta_bits: u64,
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
            domain_alpha_enabled: domain_alpha.enabled,
            domain_alpha_iters: domain_alpha.iters,
            domain_alpha_eta_bits: domain_alpha.eta.to_bits(),
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
    /// This class tree's exact Kraft-ledger result when it verified.
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
    /// #margin-row-domain-alpha: inherited per-domain alpha (see
    /// `DomainEntry::alpha`). Carried verbatim; the child re-filters it against
    /// its OWN movable slots at its own eval.
    alpha: Vec<super::domain_alpha::AlphaCarry>,
}

/// Result of one domain evaluation (see [`MarginRowBab::eval_with_pack`]).
enum EvalOut {
    /// Provably empty region: the domain is closed.
    Infeasible(PolicyExecutionEvidence),
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
    /// `eval_with_pack` returned `None` (provably empty box): closed.
    Infeasible(DomainExecutionEvidence),
    /// #deadline-poll (D2): an intra-expansion poll fired inside this
    /// domain's eval or candidate scoring. The merge stops the tree loop with
    /// a wallclock stop reason; the domain is NOT closed and NOT counted.
    DeadlineHit(DomainExecutionEvidence),
    /// The refreshed bound already closes the domain.
    ClosedAtEval(DomainExecutionEvidence),
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
        /// Pure per-evaluation policy counters for deterministic serial merge.
        execution: DomainExecutionEvidence,
    },
}

impl ExpandStep {
    fn policy_execution(&self) -> PolicyExecutionEvidence {
        match self {
            Self::Infeasible(evidence)
            | Self::DeadlineHit(evidence)
            | Self::ClosedAtEval(evidence)
            | Self::NoCandidates(evidence)
            | Self::NonFiniteBound(evidence) => evidence.policy,
            Self::Expanded { execution, .. } => execution.policy,
        }
    }

    fn epoch_execution(&self) -> EpochExecutionEvidence {
        match self {
            Self::Infeasible(evidence)
            | Self::DeadlineHit(evidence)
            | Self::ClosedAtEval(evidence)
            | Self::NoCandidates(evidence)
            | Self::NonFiniteBound(evidence) => evidence.epoch,
            Self::Expanded { execution, .. } => execution.epoch,
        }
    }

    fn width_execution(&self) -> WidthExecutionEvidence {
        match self {
            Self::Infeasible(evidence)
            | Self::DeadlineHit(evidence)
            | Self::ClosedAtEval(evidence)
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
    /// Closed-leaf ledger (#epoch-bab Phase E).
    ledger: Ledger,
    /// AND of every nested epoch subtree's own ledger verdict.
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
        let cfg = cfg.seed_outer_epoch();
        // #closure-adaptive: one instance, one ladder (see `run`). The classwise
        // schedule's per-class trees deliberately SHARE it — a tier class 1
        // earned by measuring itself is exactly the knowledge class 2 should
        // start from, and re-paying the warm-up per class would spend the
        // budget on rediscovering the same fact.
        super::closure::reset();
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
            frontier_execution: FrontierExecutionEvidence::classwise(),
            policy_execution: PolicyExecutionEvidence::from_config(&cfg),
            epoch_execution: EpochExecutionEvidence::from_config(&cfg),
            width_execution: WidthExecutionEvidence::from_config(&cfg),
            root_precision_execution: cfg.root_precision_execution,
            backward_interm_execution: cfg.backward_interm_execution,
            root_alpha_execution: cfg.root_alpha_execution,
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
                    frontier_execution: FrontierExecutionEvidence::classwise(),
                    policy_execution: PolicyExecutionEvidence::from_config(&cfg),
                    epoch_execution: EpochExecutionEvidence::from_config(&cfg),
                    width_execution: WidthExecutionEvidence::from_config(&cfg),
                    root_precision_execution: cfg.root_precision_execution,
                    backward_interm_execution: cfg.backward_interm_execution,
                    root_alpha_execution: cfg.root_alpha_execution,
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
            lane_cache: super::gpu_backward::cache::PlanCache::default(),
            closure: super::closure::ClosureMonitor::new(super::closure::armed()),
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
            // #closure-adaptive: observe THIS lane's own child-closure fraction
            // and escalate the closure mechanisms if the children are not
            // closing. `heap.len()` is the open frontier at this instant (the
            // domain about to be popped is still in it).
            self.closure_step(heap.len(), t0.elapsed().as_secs_f64(), &re.pack);
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
            // The entry survived the budget/deadline, finite-bound, closed,
            // and epoch gates and is about to enter executable domain work.
            // Include the already-popped entry in the pre-decision open count.
            self.stats.frontier_execution.observe(heap.len() + 1);
            let (b_eval, mut st) = match self.eval_node(&entry)? {
                EvalOut::Infeasible(policy_execution) => {
                    self.stats.policy_execution.absorb(policy_execution);
                    // Infeasible domain: provably empty, closed.
                    self.stats.closed += 1;
                    self.ledger.leaf(depth);
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
            let b = b_eval.max(entry.bound);
            if closed(mode, b) {
                self.stats.closed += 1;
                self.ledger.leaf(depth);
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
                        // #margin-row-domain-alpha: warm start (see
                        // `DomainEntry::alpha`).
                        alpha: st.alpha.clone(),
                    });
                    seq += 1;
                }
            }
            self.stats.expansions += 1;
        }
        self.stats.stop = stop.clone();
        self.stats.elapsed_secs = t0.elapsed().as_secs_f64();
        // R9: the armed policy always leaves a receipt, so a run that observed
        // nothing is distinguishable from a run where the policy never loaded.
        self.closure.finish(
            self.stats.expansions,
            self.stats.domains_created,
            self.stats.closed,
            &stop,
        );
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
            // #closure-adaptive: same observation as the serial lane, in the
            // serial section so the whole batch runs at one tier.
            self.closure_step(heap.len(), t0.elapsed().as_secs_f64(), &re.pack);
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
                let step = step?;
                self.stats.policy_execution.absorb(step.policy_execution());
                self.stats.epoch_execution.absorb(step.epoch_execution());
                self.stats.width_execution.absorb(step.width_execution());
                match step {
                    ExpandStep::Infeasible(_) | ExpandStep::ClosedAtEval(_) => {
                        self.stats.closed += 1;
                        self.ledger.leaf(depth);
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
                        execution: _,
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
                                alpha: c.alpha,
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
        // R9: the armed policy always leaves a receipt, so a run that observed
        // nothing is distinguishable from a run where the policy never loaded.
        self.closure.finish(
            self.stats.expansions,
            self.stats.domains_created,
            self.stats.closed,
            &stop,
        );
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
        Ok(out)
    }

    fn prepare_expand(&self, pack: &Arc<YPack>, entry: &DomainEntry) -> Result<PreparedExpand> {
        let mode = self.eng.root.mode;
        let (b_eval, mut st) = match self.eval_with_pack(pack, entry)? {
            EvalOut::Infeasible(policy_execution) => {
                return Ok(PreparedExpand::Done(Box::new(ExpandStep::Infeasible(
                    DomainExecutionEvidence::before_selection(policy_execution, &self.cfg),
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
            return Ok(PreparedExpand::Done(Box::new(ExpandStep::ClosedAtEval(
                DomainExecutionEvidence::from_state(&st),
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
                return ExpandStep::NonFiniteBound(execution);
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
                    // #margin-row-domain-alpha: warm start (see
                    // `DomainEntry::alpha`).
                    alpha: ready.st.alpha.clone(),
                });
            }
        }
        ExpandStep::Expanded {
            dips,
            worst_dip,
            pushes,
            closed_children,
            created_children,
            execution,
        }
    }

    /// Evaluate+expand one popped domain given a PREBUILT pack (pure, `&self`).
    /// Mirrors the serial expansion body exactly; see [`Self::tree_loop`].
    fn expand_one(&self, pack: &Arc<YPack>, entry: &DomainEntry) -> Result<ExpandStep> {
        let mode = self.eng.root.mode;
        let (b_eval, mut st) = match self.eval_with_pack(pack, entry)? {
            EvalOut::Infeasible(policy_execution) => {
                return Ok(ExpandStep::Infeasible(
                    DomainExecutionEvidence::before_selection(policy_execution, &self.cfg),
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
            return Ok(ExpandStep::ClosedAtEval(
                DomainExecutionEvidence::from_state(&st),
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
                return Ok(ExpandStep::NonFiniteBound(
                    DomainExecutionEvidence::from_state(&st),
                ));
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
                    // #margin-row-domain-alpha: warm start (see
                    // `DomainEntry::alpha`).
                    alpha: st.alpha.clone(),
                });
            }
        }
        Ok(ExpandStep::Expanded {
            dips,
            worst_dip,
            pushes,
            closed_children,
            created_children,
            execution: DomainExecutionEvidence::from_state(&st),
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
    fn eval_with_pack(&self, pack: &Arc<YPack>, entry: &DomainEntry) -> Result<EvalOut> {
        let mut policy_execution = PolicyExecutionEvidence::from_config(&self.cfg);
        let epoch_execution = EpochExecutionEvidence::from_config(&self.cfg);
        let width_execution = WidthExecutionEvidence::from_config(&self.cfg);
        // #deadline-poll (D2): guard the domain-gate rebuild + the main
        // certified pass about to be spent.
        if self.deadline_hit() {
            poll_note("eval_node");
            return Ok(EvalOut::Deadline(policy_execution));
        }
        let mode = self.eng.root.mode;
        let _t = super::prof::Timer::start(super::prof::Phase::EvalNode);
        let dom = domain_gates(self.eng.root, &entry.trunk);
        if dom.infeasible {
            // #clip-and-verify VERIFY: the split set's halfspaces make this
            // domain's region empty. Discharged without paying the ~600 ms
            // backward pass — the exact "child that never closes" converted
            // into an instant close.
            return Ok(EvalOut::Infeasible(policy_execution));
        }
        let pack = pack.clone();
        let mut ybox = YBox {
            ly: pack.ly0.clone(),
            uy: pack.uy0.clone(),
        };
        ybox.clamp(&entry.heads);
        if ybox.is_empty() {
            return Ok(EvalOut::Infeasible(policy_execution));
        }
        ybox.intersect(&entry.ly, &entry.uy);
        if ybox.is_empty() {
            return Ok(EvalOut::Infeasible(policy_execution));
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
                    policy_execution.inherited_retirement_evals = policy_execution
                        .inherited_retirement_evals
                        .saturating_add(1);
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
        let alpha_armed = alpha_policy.enabled;
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
        let collect = Collect {
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
            unst_rows: beta_armed || alpha_armed,
        };
        // B5 (#margin-row-gpu-eft, SITE_EVAL): the ~600 ms main pass — 43.4%
        // of the lane — routes through the certified GPU lane's own seam.
        // Dark and fail-closed: with the lane disarmed (or on ANY refusal —
        // beta-carrying dom, tier-0/unst_rows collection, Busy device, cap
        // overflow) this is exactly the established `run_collect` call,
        // bit-for-bit. The seed is the MARGIN seed, which carries certified
        // error, so the lane needs the y-box magnitudes to concretize both
        // discrepancies into the bias-error lane (the `seed_penalty`
        // contract, same as the root pass at bab.rs:910-919).
        let eval_y_abs: Vec<f64> = ybox
            .ly
            .iter()
            .zip(&ybox.uy)
            .map(|(l, u)| l.abs().max(u.abs()))
            .collect();
        let lane_ctx = super::gpu_backward::LaneCtx {
            y_abs: Some(&eval_y_abs),
            deadline: self.cfg.deadline,
            plan_cache: Some(&self.lane_cache),
        };
        let mut pass = self.eng.run_collect_seamed(
            seed_main_pc
                .as_ref()
                .or(seed_main.as_ref())
                .unwrap_or(&ms.seed),
            Some(&dom),
            super::engine::LaneDir::Lower,
            collect,
            &lane_ctx,
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
                let trial = self.eng.run_collect(
                    seed_main_pc
                        .as_ref()
                        .or(seed_main.as_ref())
                        .unwrap_or(&ms.seed),
                    Some(&dom),
                    super::engine::LaneDir::Lower,
                    None,
                    collect,
                );
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
                    grad = super::alpha_opt::supergradient(
                        self.eng.net,
                        self.eng.root,
                        &super::domain_alpha::effective_alpha(self.eng.root, &dom),
                        &pass,
                        worst_col,
                    );
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
            let built = RootGates::build_retaining_with_policy(
                net,
                &root.lo,
                &root.hi,
                root.mode,
                self.cfg.deadline,
                self.cfg.retain_cfg.as_ref(),
                &abs,
                self.cfg.root_precision_policy,
                super::BackwardIntermRuntimePolicy::default(),
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
        // #epoch-monotone. MEASURED 2026-08-22 on cifar100 idx_4169 at the
        // official budget: the rebuild's root bound came out -0.389998 against
        // the parent tree's -0.371373 — WORSE, on a domain with four neurons
        // FIXED, which is a strict subset of the parent region and can only
        // have a better true bound. The cause is visible in the log ordering:
        // `build_retaining` is the ROOT BUILD, and the #backward-interm phase
        // that tightened 584 intermediate columns for 7.11 s is applied
        // separately at lane setup and never re-run for the nested tree. So the
        // rebuild was trading 584 tightened columns for 4 fixed neurons and
        // losing. That is the whole reason this mechanism was built, measured
        // once, and left dark.
        //
        // The fix costs nothing and makes the rebuild MONOTONE BY CONSTRUCTION.
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
            // #margin-row-col-retire: the nested lane re-derives its own root
            // fail-set on the epoch gates; the outer retirement set is NOT
            // inherited (root frames differ), only the gate carries over.
            col_retire: self.cfg.col_retire,
            beta_per_column: self.cfg.beta_per_column,
            beta_columns: self.cfg.beta_columns,
            parallel_prebuild: self.cfg.parallel_prebuild,
            root_precision_policy: self.cfg.root_precision_policy,
            root_precision_execution: self.cfg.root_precision_execution,
            backward_interm_execution: self.cfg.backward_interm_execution,
            root_alpha_execution: self.cfg.root_alpha_execution,
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
        let pass = self.eng.run_exc_seamed(
            &Seed { s: seed, e: seed_e },
            Some(&st.dom),
            super::engine::LaneDir::Lower,
            &exc,
            &lane_ctx,
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
    use super::{closed, require_finite, BabConfig, Ledger, RoundMode};

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
            },
            epoch_failures: 0,
            ledger: Ledger::default(),
            nested_ledgers_ok: true,
            lane_cache: crate::margin_row::gpu_backward::cache::PlanCache::default(),
            closure: crate::margin_row::closure::ClosureMonitor::new(false),
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
                alpha: child.alpha.clone(),
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
        let da = a.observe(8, 16, 0, 16, 0.8, Some(500.0));
        crate::margin_row::closure::reset();
        let db = b.observe(8, 16, 0, 16, 0.8, Some(500.0));
        assert_eq!(da, db);
        assert_eq!(da, Decision::Escalate(TIER_BETA));
        crate::margin_row::closure::reset();
        assert_eq!(tier(), TIER_BASE);
    }
}
