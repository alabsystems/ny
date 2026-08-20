// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Run-scoped observations of optional verification treatments.
//!
//! Configuration says what a caller requested; this recorder says what the
//! verifier actually reached. Recording is deliberately inert outside an
//! explicit [`begin_run`] scope. Overlapping scopes disable observation rather
//! than mixing events from two solves.
//!
//! Event producers must finish (or be synchronously joined) before their run
//! guard is dropped. The instrumented verifier paths satisfy that contract;
//! detached workers would need an explicit generation token before recording.

use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use serde::Serialize;

const EXECUTION_OBSERVATIONS_SCHEMA: &str = "ny_beta_crown_execution_observations_v5";

/// Structured execution evidence captured during one beta-CROWN command.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ExecutionObservations {
    pub schema: &'static str,
    /// Whether a non-conflicting run scope is active at snapshot time.
    pub run_active: bool,
    /// True when overlapping run scopes made attribution ambiguous.
    pub recording_conflict: bool,
    pub exact_c: ExactCObservations,
    pub root_spec_prune: RootSpecPruneObservations,
    pub invprop: InvpropObservations,
    pub fresh_domain_clip: FreshDomainClipObservations,
    /// Every full Patches-to-Dense materialization attempted by this solve.
    /// The section is always serialized, including an honest all-zero default.
    pub patches_materialization: PatchesMaterializationObservations,
}

impl Default for ExecutionObservations {
    fn default() -> Self {
        Self {
            schema: EXECUTION_OBSERVATIONS_SCHEMA,
            run_active: false,
            recording_conflict: false,
            exact_c: ExactCObservations::default(),
            root_spec_prune: RootSpecPruneObservations::default(),
            invprop: InvpropObservations::default(),
            fresh_domain_clip: FreshDomainClipObservations::default(),
            patches_materialization: PatchesMaterializationObservations::default(),
        }
    }
}

/// Semantic reason a caller crossed the Patches-to-Dense boundary.
///
/// Only a literal backward `Reshape` boundary and final `NETWORK_INPUT`
/// concretization receive dedicated classes. All cache captures, merge
/// promotions, unsupported operators, and other consumers must use `Other`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PatchesMaterializationPurpose {
    LatentInputCrossover,
    NetworkInputTerminal,
    Other,
}

/// Geometry carried by the lower/upper Patches pair at attempt time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PatchesMaterializationGeometry {
    Affine,
    Anchored,
    /// Lower and upper sides disagree on the geometry schema. The materializer
    /// will reject the malformed pair, but telemetry still accounts the attempt.
    Conflicting,
}

/// Successful disposition of the certified coefficient-error carrier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PatchesCoefficientErrorDisposition {
    /// The published Dense relation has no coefficient-error sidecar.
    Absent,
    /// The published Dense relation retains a certified error matrix.
    Materialized,
}

/// Typed refusal class for a failed full materialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PatchesMaterializationRefusal {
    Memory,
    Deadline,
    Semantic,
}

/// Exact total-live memory admission receipt from one successful materializer.
///
/// `admitted_bytes == nominal_required_bytes + capacity_overage_bytes` and
/// `admitted_bytes <= budget_bytes` are checked both by the producer and by the
/// aggregate snapshot validator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PatchesMaterializationMemoryReceipt {
    pub(crate) nominal_required_bytes: usize,
    pub(crate) capacity_overage_bytes: usize,
    pub(crate) admitted_bytes: usize,
    pub(crate) budget_bytes: usize,
}

/// Attempt/outcome counts for one semantic materialization purpose.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct PatchesMaterializationPurposeObservations {
    pub attempts: usize,
    pub succeeded: usize,
    pub refused: usize,
}

/// Run-scoped, fixed-size evidence for full Patches-to-Dense materialization.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct PatchesMaterializationObservations {
    /// True iff at least one full materialization attempt reached the recorder.
    pub observed: bool,
    pub attribution_conflict: bool,
    pub counter_overflow: bool,

    pub attempts: usize,
    pub succeeded: usize,
    pub refused: usize,

    pub latent_input_crossover: PatchesMaterializationPurposeObservations,
    pub network_input_terminal: PatchesMaterializationPurposeObservations,
    pub other: PatchesMaterializationPurposeObservations,

    pub finite_deadline_attempts: usize,
    pub no_deadline_attempts: usize,

    pub affine_geometry_attempts: usize,
    pub anchored_geometry_attempts: usize,
    pub conflicting_geometry_attempts: usize,

    /// Attempts whose input pair carried at least one row-level coefficient
    /// error. A 7D scatter can still generate a Dense error matrix without
    /// incrementing this counter, so successful output disposition is separate.
    pub input_coefficient_error_attempts: usize,
    pub coefficient_error_absent: usize,
    pub coefficient_error_materialized: usize,

    pub memory_refusals: usize,
    pub deadline_refusals: usize,
    pub semantic_refusals: usize,

    /// Exact checked sums of all successful admission receipts.
    pub memory_receipt_outcomes: usize,
    pub nominal_required_bytes: usize,
    pub capacity_overage_bytes: usize,
    pub admitted_bytes: usize,
    pub budget_bytes: usize,
}

/// Observed execution of the typed bounded exact-C root treatment.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ExactCObservations {
    /// True only after the typed route was selected at runtime.
    pub observed: bool,
    pub selections: usize,
    /// The common selected limit, or `None` when absent or inconsistent.
    pub selected_iteration_limit: Option<usize>,
    pub selected_iteration_limit_conflict: bool,
    /// The common row-selection disposition, or `None` when absent/conflicting.
    pub selected_compressed: Option<bool>,
    pub selected_compressed_conflict: bool,
    /// Source/evaluated/precertified layout observations at route selection.
    pub layout_observations: usize,
    pub source_rows: usize,
    pub evaluated_rows: usize,
    pub precertified_rows: usize,
    pub compressed_selections: usize,
    /// Compressed selections paired, in event order, with a finalized prune
    /// layout or explicitly rolled back before publication.
    pub compressed_layouts_finalized: usize,
    pub compressed_layouts_rolled_back: usize,
    /// Post-backend compact publication evidence.
    pub compact_commits: usize,
    pub compact_reconstruction_succeeded: usize,
    pub compact_reconstruction_failed: usize,
    pub compact_binding_map_succeeded: usize,
    pub compact_binding_map_failed: usize,
    pub compact_alpha_candidates: usize,
    pub compact_alpha_published: usize,
    pub compact_alpha_dropped: usize,
    /// An outcome arrived without a distinct preceding selection.
    pub attribution_conflict: bool,
    pub counter_overflow: bool,
    pub outcomes_observed: usize,
    pub refused_before_commit: usize,
    pub committed: usize,
    /// Outcomes that reported exact attempted/accepted iteration counts.
    pub iteration_count_outcomes: usize,
    pub iteration_count_conflict: bool,
    pub attempted_iterations: usize,
    pub accepted_iterations: usize,
    /// Outcomes authenticated by the typed multi-iteration child. These
    /// aggregates are fixed-size: runtime evidence must not grow memory with
    /// the number of outcomes.
    pub multi_iteration_evidence_outcomes: usize,
    /// Common exact child-gate value, or `None` before any authenticated
    /// multi-iteration outcome or after a conflict.
    pub multiplicative_weights_requested: Option<bool>,
    pub multiplicative_weights_requested_conflict: bool,
    /// Outcomes in which a full-row MW plan reached the proposal seam.
    pub multiplicative_weights_plan_dispatched_outcomes: usize,
    /// Outcomes in which a post-update MW plan returned a complete candidate
    /// pair. A completed first uniform proposal is deliberately excluded.
    pub multiplicative_weights_effective_outcomes: usize,
    /// Dispatched plans whose complete candidate pairs returned.
    pub completed_proposals: usize,
    /// Full-row plans dispatched with a post-update row-player distribution.
    pub adaptive_plan_dispatches: usize,
    /// Common planned `num_specs` among dispatched plans, or `None` when none
    /// were dispatched or the shape conflicted.
    pub gradient_plan_num_specs: Option<usize>,
    pub gradient_plan_num_specs_conflict: bool,
    /// Common exact-C row count among authenticated multi-iteration outcomes.
    pub gradient_row_count: Option<usize>,
    pub gradient_row_count_conflict: bool,
    /// A producer supplied an internally inconsistent per-outcome claim.
    pub multi_iteration_evidence_conflict: bool,
    pub stop_reasons: BTreeMap<String, usize>,
    /// Runtime-only event-order state. It is intentionally absent from JSON;
    /// a nonempty queue makes the serialized counters fail validation.
    #[serde(skip)]
    pending_compressed_layouts: VecDeque<RootRowLayout>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RootRowLayout {
    source_rows: usize,
    evaluated_rows: usize,
    precertified_rows: usize,
}

impl RootRowLayout {
    fn new(source_rows: usize, evaluated_rows: usize, precertified_rows: usize) -> Option<Self> {
        (source_rows > 0
            && evaluated_rows <= source_rows
            && precertified_rows == source_rows - evaluated_rows)
            .then_some(Self {
                source_rows,
                evaluated_rows,
                precertified_rows,
            })
    }

    fn compressed(self) -> bool {
        self.precertified_rows > 0
    }
}

/// Observed dispatch, planning, and publication of root specification pruning.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct RootSpecPruneObservations {
    /// True only after the runtime dispatcher was reached (including OFF).
    pub observed: bool,
    pub attribution_conflict: bool,
    pub counter_overflow: bool,
    pub route_observations: usize,
    /// Common request observed at the dispatcher. `None` is unobserved/conflict.
    pub configured: Option<bool>,
    pub route_conflict: bool,
    pub plans_built: usize,
    pub applied: usize,
    pub layout_observations: usize,
    pub source_rows: usize,
    pub evaluated_rows: usize,
    pub precertified_rows: usize,
    pub all_pruned: usize,
}

/// Observed execution of INVPROP initialization and gamma optimization.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct InvpropObservations {
    /// True only after at least one INVPROP runtime seam was reached.
    pub observed: bool,
    /// A rebind result or applied gamma step lacked a matching attempt.
    pub attribution_conflict: bool,
    pub counter_overflow: bool,
    pub clause_rebind_attempts: usize,
    pub clause_rebind_accepted: usize,
    pub clause_rebind_refused: usize,
    pub alpha_initializations: usize,
    pub gamma_steps_attempted: usize,
    pub gamma_steps_applied: usize,
    /// Successful nonzero folds into a full output identity seed. This can
    /// count both optimizer probes and later evaluated states: it proves the
    /// sound INVPROP algebra actually executed, not that a proposed parameter
    /// write was accepted or improved the returned bound.
    pub nonzero_output_seed_folds: usize,
    /// Nonzero folds from an explicitly scoped authoritative loop iterate.
    /// Discarded SPSA, finite-difference, and supplemental backwards are not
    /// evaluated folds.
    pub nonzero_evaluated_output_seed_folds: usize,
}

/// Observed dispatch and outcomes of the exact-current-domain LSNC clip.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct FreshDomainClipObservations {
    /// True after the runtime dispatcher or an outcome recorder was reached.
    /// Actual clip execution additionally requires `attempts > 0`.
    pub observed: bool,
    pub attribution_conflict: bool,
    pub counter_overflow: bool,
    pub route_observations: usize,
    /// Common typed request observed at the runtime dispatcher. `None` means
    /// not observed or conflicting, never an inferred `false`.
    pub configured: Option<bool>,
    /// Common runtime authorization decision. `None` is unobserved/conflicting.
    pub route_authorized: Option<bool>,
    pub route_conflict: bool,
    pub attempts: usize,
    pub applied: usize,
    pub all_clauses_refuted: usize,
    pub skipped: usize,
    pub tightened_dimensions: usize,
}

/// One disposition emitted by the fresh-domain clip's local telemetry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreshDomainClipDisposition {
    Applied,
    AllClausesRefuted,
    Skipped,
}

thread_local! {
    static INVPROP_EVALUATED_FOLD_SCOPES: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// Scoped marker for an authoritative loop backward whose bound is evaluated.
///
/// INVPROP folds are unclassified by default. This is intentional: alpha
/// finite-difference/SPSA and supplemental gradient paths also execute real
/// backwards, but discard their bounds. Requiring an explicit positive scope
/// makes those probes unable to inflate evaluated-fold evidence.
pub(crate) struct InvpropEvaluatedFoldGuard {
    active: bool,
}

impl Drop for InvpropEvaluatedFoldGuard {
    fn drop(&mut self) {
        if self.active {
            INVPROP_EVALUATED_FOLD_SCOPES.with(|scopes| {
                let _discarded_pending_folds = scopes.borrow_mut().pop();
            });
        }
    }
}

impl InvpropEvaluatedFoldGuard {
    /// Commit folds accumulated by a backward that reached authoritative final
    /// concretization. Dropping without commit discards only the evaluated
    /// attribution; the total real-fold counter was already recorded.
    pub(crate) fn commit(mut self) {
        let pending_folds = INVPROP_EVALUATED_FOLD_SCOPES
            .with(|scopes| scopes.borrow_mut().pop().unwrap_or_default());
        self.active = false;
        if pending_folds == 0 {
            return;
        }
        RECORDER.record(|all| {
            let invprop = &mut all.invprop;
            invprop.counter_overflow |= add_counter(
                &mut invprop.nonzero_evaluated_output_seed_folds,
                pending_folds,
            );
        });
    }
}

pub(crate) fn begin_invprop_evaluated_fold_scope() -> InvpropEvaluatedFoldGuard {
    INVPROP_EVALUATED_FOLD_SCOPES.with(|scopes| {
        scopes.borrow_mut().push(0);
    });
    InvpropEvaluatedFoldGuard { active: true }
}

fn accumulate_pending_evaluated_fold() -> bool {
    INVPROP_EVALUATED_FOLD_SCOPES.with(|scopes| {
        let mut scopes = scopes.borrow_mut();
        let Some(pending) = scopes.last_mut() else {
            return false;
        };
        let overflowed = pending.checked_add(1).is_none();
        *pending = pending.saturating_add(1);
        overflowed
    })
}

#[derive(Debug)]
struct ActiveRun {
    generation: u64,
    #[cfg(test)]
    owner_thread: std::thread::ThreadId,
    observations: ExecutionObservations,
}

#[derive(Debug)]
enum RecorderState {
    Inactive,
    Active(Box<ActiveRun>),
    /// Live generations whose observations cannot be attributed independently.
    Conflicted(Vec<u64>),
}

#[derive(Debug)]
struct Recorder {
    next_generation: AtomicU64,
    state: Mutex<RecorderState>,
}

impl Recorder {
    const fn new() -> Self {
        Self {
            next_generation: AtomicU64::new(1),
            state: Mutex::new(RecorderState::Inactive),
        }
    }

    // trust-1.99 deprecates `fetch_update` (renamed `try_update`); the public
    // 1.95 pin lacks `try_update` — keep the spelling both toolchains accept.
    #[allow(deprecated)]
    fn begin(&self) -> ExecutionTelemetryRun<'_> {
        let generation =
            self.next_generation
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    current.checked_add(1)
                });
        let (generation, generation_exhausted) = match generation {
            Ok(generation) => (generation, false),
            Err(exhausted) => (exhausted, true),
        };
        if let Ok(mut state) = self.state.lock() {
            if generation_exhausted {
                match &mut *state {
                    RecorderState::Inactive => {
                        *state = RecorderState::Conflicted(vec![generation]);
                    }
                    RecorderState::Active(active) => {
                        *state = RecorderState::Conflicted(vec![active.generation, generation]);
                    }
                    RecorderState::Conflicted(generations) => generations.push(generation),
                }
                return ExecutionTelemetryRun {
                    recorder: self,
                    generation,
                };
            }
            match &mut *state {
                RecorderState::Inactive => {
                    let observations = ExecutionObservations {
                        run_active: true,
                        ..ExecutionObservations::default()
                    };
                    *state = RecorderState::Active(Box::new(ActiveRun {
                        generation,
                        #[cfg(test)]
                        owner_thread: std::thread::current().id(),
                        observations,
                    }));
                }
                RecorderState::Active(active) => {
                    *state = RecorderState::Conflicted(vec![active.generation, generation]);
                }
                RecorderState::Conflicted(generations) => generations.push(generation),
            }
        }
        ExecutionTelemetryRun {
            recorder: self,
            generation,
        }
    }

    fn finish(&self, generation: u64) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        match &mut *state {
            RecorderState::Active(active) if active.generation == generation => {
                *state = RecorderState::Inactive;
            }
            RecorderState::Conflicted(generations) => {
                if let Some(index) = generations
                    .iter()
                    .position(|candidate| *candidate == generation)
                {
                    generations.remove(index);
                }
                if generations.is_empty() {
                    *state = RecorderState::Inactive;
                }
            }
            RecorderState::Inactive | RecorderState::Active(_) => {}
        }
    }

    fn snapshot(&self) -> ExecutionObservations {
        let Ok(state) = self.state.lock() else {
            return ExecutionObservations {
                recording_conflict: true,
                ..ExecutionObservations::default()
            };
        };
        match &*state {
            RecorderState::Inactive => ExecutionObservations::default(),
            RecorderState::Active(active) => validate_snapshot(active.observations.clone()),
            RecorderState::Conflicted(_) => ExecutionObservations {
                recording_conflict: true,
                ..ExecutionObservations::default()
            },
        }
    }

    fn record(&self, update: impl FnOnce(&mut ExecutionObservations)) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if let RecorderState::Active(active) = &mut *state {
            // Unit tests execute unrelated verifier cases in parallel inside
            // one process. Production command attribution is process-wide,
            // but test scopes must ignore events emitted by foreign harness
            // threads or zero-count assertions become scheduling-dependent.
            #[cfg(test)]
            if active.owner_thread != std::thread::current().id() {
                return;
            }
            update(&mut active.observations);
        }
    }
}

fn validate_snapshot(mut observations: ExecutionObservations) -> ExecutionObservations {
    let exact_c = &mut observations.exact_c;
    let classified_outcomes = checked_sum([exact_c.refused_before_commit, exact_c.committed]);
    let stop_reason_outcomes = checked_sum(exact_c.stop_reasons.values().copied());
    let exact_row_partition = checked_sum([exact_c.evaluated_rows, exact_c.precertified_rows]);
    let resolved_compressed_layouts = checked_sum([
        exact_c.compressed_layouts_finalized,
        exact_c.compressed_layouts_rolled_back,
    ]);
    let compact_reconstruction_outcomes = checked_sum([
        exact_c.compact_reconstruction_succeeded,
        exact_c.compact_reconstruction_failed,
    ]);
    let compact_binding_outcomes = checked_sum([
        exact_c.compact_binding_map_succeeded,
        exact_c.compact_binding_map_failed,
    ]);
    let compact_alpha_outcomes = checked_sum([
        exact_c.compact_alpha_published,
        exact_c.compact_alpha_dropped,
    ]);
    if exact_c.observed != (exact_c.selections > 0)
        || exact_c.selections != exact_c.outcomes_observed
        || classified_outcomes != Some(exact_c.outcomes_observed)
        || stop_reason_outcomes != Some(exact_c.outcomes_observed)
        || exact_c.layout_observations != exact_c.selections
        || exact_row_partition != Some(exact_c.source_rows)
        || exact_c.compressed_selections > exact_c.selections
        || resolved_compressed_layouts != Some(exact_c.compressed_selections)
        || !exact_c.pending_compressed_layouts.is_empty()
        || compact_reconstruction_outcomes != Some(exact_c.compact_commits)
        || compact_binding_outcomes != Some(exact_c.compact_commits)
        || exact_c.compact_commits > exact_c.committed
        || exact_c.compact_commits > exact_c.compressed_layouts_finalized
        || compact_alpha_outcomes != Some(exact_c.compact_alpha_candidates)
        || exact_c.compact_alpha_published > exact_c.compact_reconstruction_succeeded
        || exact_c.compact_alpha_published > exact_c.compact_binding_map_succeeded
        || exact_c.iteration_count_outcomes > exact_c.committed
        || exact_c.accepted_iterations > exact_c.attempted_iterations
        || !valid_exact_c_multi_iteration_aggregates(exact_c)
        || (exact_c.selections == 0
            && (exact_c.selected_iteration_limit.is_some()
                || exact_c.selected_compressed.is_some()))
        || (exact_c.selections > 0
            && (exact_c.selected_iteration_limit.is_none()
                || exact_c.selected_compressed.is_none()))
        || exact_c.selected_iteration_limit_conflict
        || exact_c.selected_compressed_conflict
        || exact_c.iteration_count_conflict
        || exact_c.counter_overflow
    {
        exact_c.attribution_conflict = true;
    }

    let prune = &mut observations.root_spec_prune;
    let prune_row_partition = checked_sum([prune.evaluated_rows, prune.precertified_rows]);
    if prune.observed != (prune.route_observations > 0)
        || (prune.route_observations == 0 && prune.configured.is_some())
        || (prune.route_observations > 0 && prune.configured.is_none())
        || (prune.configured != Some(true) && (prune.plans_built > 0 || prune.applied > 0))
        || prune.plans_built > prune.route_observations
        || prune.applied > prune.plans_built
        || prune.layout_observations != prune.applied
        || prune_row_partition != Some(prune.source_rows)
        || prune.all_pruned > prune.applied
        || prune.route_conflict
        || prune.counter_overflow
    {
        prune.attribution_conflict = true;
    }
    if exact_c.compressed_layouts_finalized > prune.applied {
        exact_c.attribution_conflict = true;
        prune.attribution_conflict = true;
    }
    // Aggregate equality is safe only when all exact selections are compressed
    // and every applied prune layout is one of their ordered finalizations.
    if exact_c.selected_compressed == Some(true)
        && exact_c.compressed_layouts_finalized > 0
        && exact_c.compressed_layouts_finalized == prune.applied
        && (exact_c.source_rows != prune.source_rows
            || exact_c.evaluated_rows != prune.evaluated_rows
            || exact_c.precertified_rows != prune.precertified_rows)
    {
        exact_c.attribution_conflict = true;
        prune.attribution_conflict = true;
    }

    let invprop = &mut observations.invprop;
    let rebind_outcomes = checked_sum([
        invprop.clause_rebind_accepted,
        invprop.clause_rebind_refused,
    ]);
    if rebind_outcomes != Some(invprop.clause_rebind_attempts)
        || invprop.gamma_steps_applied > invprop.gamma_steps_attempted
        || invprop.nonzero_evaluated_output_seed_folds > invprop.nonzero_output_seed_folds
        || (invprop.nonzero_evaluated_output_seed_folds > 0 && invprop.gamma_steps_applied == 0)
        || invprop.counter_overflow
    {
        invprop.attribution_conflict = true;
    }

    let fresh = &mut observations.fresh_domain_clip;
    let dispositions = checked_sum([fresh.applied, fresh.all_clauses_refuted, fresh.skipped]);
    if dispositions != Some(fresh.attempts)
        || (fresh.attempts > 0 && fresh.route_authorized != Some(true))
        || (fresh.route_authorized == Some(true) && fresh.configured != Some(true))
        || (fresh.route_observations == 0
            && (fresh.configured.is_some()
                || fresh.route_authorized.is_some()
                || fresh.attempts > 0))
        || (fresh.route_observations > 0
            && (fresh.configured.is_none() || fresh.route_authorized.is_none()))
        || (fresh.applied == 0 && fresh.tightened_dimensions > 0)
        || fresh.route_conflict
        || fresh.counter_overflow
    {
        fresh.attribution_conflict = true;
    }

    let patches = &mut observations.patches_materialization;
    let outcomes = checked_sum([patches.succeeded, patches.refused]);
    let purpose_attempts = checked_sum([
        patches.latent_input_crossover.attempts,
        patches.network_input_terminal.attempts,
        patches.other.attempts,
    ]);
    let purpose_succeeded = checked_sum([
        patches.latent_input_crossover.succeeded,
        patches.network_input_terminal.succeeded,
        patches.other.succeeded,
    ]);
    let purpose_refused = checked_sum([
        patches.latent_input_crossover.refused,
        patches.network_input_terminal.refused,
        patches.other.refused,
    ]);
    let deadline_attempts = checked_sum([
        patches.finite_deadline_attempts,
        patches.no_deadline_attempts,
    ]);
    let geometry_attempts = checked_sum([
        patches.affine_geometry_attempts,
        patches.anchored_geometry_attempts,
        patches.conflicting_geometry_attempts,
    ]);
    let refusal_dispositions = checked_sum([
        patches.memory_refusals,
        patches.deadline_refusals,
        patches.semantic_refusals,
    ]);
    let coefficient_error_dispositions = checked_sum([
        patches.coefficient_error_absent,
        patches.coefficient_error_materialized,
    ]);
    let admitted_receipt_bytes = checked_sum([
        patches.nominal_required_bytes,
        patches.capacity_overage_bytes,
    ]);
    if patches.observed != (patches.attempts > 0)
        || outcomes != Some(patches.attempts)
        || purpose_attempts != Some(patches.attempts)
        || purpose_succeeded != Some(patches.succeeded)
        || purpose_refused != Some(patches.refused)
        || deadline_attempts != Some(patches.attempts)
        || geometry_attempts != Some(patches.attempts)
        || patches.input_coefficient_error_attempts > patches.attempts
        || refusal_dispositions != Some(patches.refused)
        || coefficient_error_dispositions != Some(patches.succeeded)
        || patches.memory_receipt_outcomes != patches.succeeded
        || admitted_receipt_bytes != Some(patches.admitted_bytes)
        || patches.admitted_bytes > patches.budget_bytes
        || patches.conflicting_geometry_attempts > patches.refused
        || patches.counter_overflow
    {
        patches.attribution_conflict = true;
    }
    observations
}

fn valid_exact_c_multi_iteration_aggregates(exact_c: &ExactCObservations) -> bool {
    let evidence_outcomes = exact_c.multi_iteration_evidence_outcomes;
    if evidence_outcomes != exact_c.iteration_count_outcomes
        || evidence_outcomes > exact_c.committed
        || exact_c.completed_proposals > exact_c.attempted_iterations
        || exact_c.completed_proposals
            < exact_c
                .attempted_iterations
                .saturating_sub(evidence_outcomes)
        || exact_c.accepted_iterations > exact_c.completed_proposals
        || exact_c.multiplicative_weights_plan_dispatched_outcomes > evidence_outcomes
        || exact_c.multiplicative_weights_effective_outcomes
            > exact_c.multiplicative_weights_plan_dispatched_outcomes
        || exact_c.adaptive_plan_dispatches > exact_c.attempted_iterations
        || exact_c.multiplicative_weights_requested_conflict
        || exact_c.gradient_plan_num_specs_conflict
        || exact_c.gradient_row_count_conflict
        || exact_c.multi_iteration_evidence_conflict
    {
        return false;
    }
    if !exact_c.selected_iteration_limit_conflict {
        if let Some(iteration_limit) = exact_c.selected_iteration_limit {
            let Some(maximum_attempted_iterations) = evidence_outcomes.checked_mul(iteration_limit)
            else {
                return false;
            };
            if exact_c.attempted_iterations > maximum_attempted_iterations {
                return false;
            }
        }
    }

    if evidence_outcomes == 0 {
        return exact_c.attempted_iterations == 0
            && exact_c.accepted_iterations == 0
            && exact_c.multiplicative_weights_requested.is_none()
            && exact_c.multiplicative_weights_plan_dispatched_outcomes == 0
            && exact_c.multiplicative_weights_effective_outcomes == 0
            && exact_c.completed_proposals == 0
            && exact_c.adaptive_plan_dispatches == 0
            && exact_c.gradient_plan_num_specs.is_none()
            && exact_c.gradient_row_count.is_none();
    }
    let Some(row_count) = exact_c.gradient_row_count else {
        return false;
    };
    if row_count == 0
        || (exact_c.attempted_iterations == 0) != exact_c.gradient_plan_num_specs.is_none()
    {
        return false;
    }

    match exact_c.multiplicative_weights_requested {
        Some(true) => {
            let plan_outcomes = exact_c.multiplicative_weights_plan_dispatched_outcomes;
            if (exact_c.attempted_iterations > 0) != (plan_outcomes > 0)
                || exact_c.attempted_iterations < plan_outcomes
                || exact_c
                    .gradient_plan_num_specs
                    .is_some_and(|num_specs| num_specs != row_count)
            {
                return false;
            }
            let expected_adaptive = if row_count > 1 {
                exact_c.attempted_iterations - plan_outcomes
            } else {
                0
            };
            if exact_c.adaptive_plan_dispatches != expected_adaptive
                || exact_c.completed_proposals < exact_c.adaptive_plan_dispatches
                || exact_c.multiplicative_weights_effective_outcomes
                    > exact_c.adaptive_plan_dispatches
            {
                return false;
            }
            if row_count == 1 {
                return exact_c.completed_proposals >= exact_c.attempted_iterations - plan_outcomes;
            }

            // For every active multi-row outcome, attempts consist of one
            // uniform plan followed by D adaptive plans. Completed proposals
            // are D plus S completed final plans, with at most one S per
            // outcome. An effective outcome needs either one adaptive plan
            // and its final completion, or two adaptive plans.
            let Some(completed_final_plans) = exact_c
                .completed_proposals
                .checked_sub(exact_c.adaptive_plan_dispatches)
            else {
                return false;
            };
            if completed_final_plans > plan_outcomes {
                return false;
            }
            let effective_outcomes = exact_c.multiplicative_weights_effective_outcomes;
            if effective_outcomes == 0 {
                return exact_c.completed_proposals <= plan_outcomes;
            }
            let Some(minimum_adaptive_for_effective) = effective_outcomes
                .checked_mul(2)
                .and_then(|twice| twice.checked_sub(effective_outcomes.min(completed_final_plans)))
            else {
                return false;
            };
            exact_c.adaptive_plan_dispatches >= minimum_adaptive_for_effective
        }
        Some(false) => {
            exact_c.multiplicative_weights_plan_dispatched_outcomes == 0
                && exact_c.multiplicative_weights_effective_outcomes == 0
                && exact_c.adaptive_plan_dispatches == 0
                && exact_c
                    .gradient_plan_num_specs
                    .is_none_or(|num_specs| num_specs == 1)
        }
        None => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn valid_exact_c_multi_iteration_evidence(
    attempted_iterations: usize,
    accepted_iterations: usize,
    multiplicative_weights_requested: bool,
    multiplicative_weights_plan_dispatched: bool,
    multiplicative_weights_effective: bool,
    completed_proposals: usize,
    adaptive_plan_dispatches: usize,
    gradient_plan_num_specs: Option<usize>,
    gradient_row_count: usize,
) -> bool {
    if gradient_row_count == 0
        || accepted_iterations > completed_proposals
        || completed_proposals > attempted_iterations
        || completed_proposals < attempted_iterations.saturating_sub(1)
        || adaptive_plan_dispatches > attempted_iterations
        || (attempted_iterations == 0) != gradient_plan_num_specs.is_none()
    {
        return false;
    }
    if multiplicative_weights_requested {
        let expected_adaptive = if gradient_row_count > 1 {
            attempted_iterations.saturating_sub(1)
        } else {
            0
        };
        multiplicative_weights_plan_dispatched == (attempted_iterations > 0)
            && multiplicative_weights_effective
                == (gradient_row_count > 1 && completed_proposals > 1)
            && adaptive_plan_dispatches == expected_adaptive
            && gradient_plan_num_specs.is_none_or(|num_specs| num_specs == gradient_row_count)
    } else {
        !multiplicative_weights_plan_dispatched
            && !multiplicative_weights_effective
            && adaptive_plan_dispatches == 0
            && gradient_plan_num_specs.is_none_or(|num_specs| num_specs == 1)
    }
}

fn checked_sum(values: impl IntoIterator<Item = usize>) -> Option<usize> {
    values
        .into_iter()
        .try_fold(0usize, |sum, value| sum.checked_add(value))
}

static RECORDER: Recorder = Recorder::new();

#[cfg(test)]
pub(crate) static TEST_LOCK: Mutex<()> = Mutex::new(());

/// RAII boundary for one command-level solve.
#[derive(Debug)]
pub struct ExecutionTelemetryRun<'a> {
    recorder: &'a Recorder,
    generation: u64,
}

impl Drop for ExecutionTelemetryRun<'_> {
    fn drop(&mut self) {
        self.recorder.finish(self.generation);
    }
}

/// Start a fresh observation scope. Dropping the guard clears the scope.
pub fn begin_run() -> ExecutionTelemetryRun<'static> {
    RECORDER.begin()
}

/// Snapshot observations for the active solve, or honest unobserved defaults.
pub fn snapshot() -> ExecutionObservations {
    RECORDER.snapshot()
}

/// Record selection of the typed bounded exact-C route.
pub fn record_exact_c_selected(
    iteration_limit: usize,
    source_rows: usize,
    evaluated_rows: usize,
    precertified_rows: usize,
) {
    RECORDER.record(|all| {
        update_exact_c_selection(
            &mut all.exact_c,
            iteration_limit,
            source_rows,
            evaluated_rows,
            precertified_rows,
        );
    });
}

/// Resolve a compressed exact-C selection that could not publish a root-prune
/// layout. The supplied layout must match the oldest pending selection.
pub fn record_exact_c_compressed_selection_rolled_back(
    source_rows: usize,
    evaluated_rows: usize,
    precertified_rows: usize,
) {
    RECORDER.record(|all| {
        resolve_exact_c_compressed_layout(
            &mut all.exact_c,
            source_rows,
            evaluated_rows,
            precertified_rows,
            false,
        );
    });
}

/// Record post-commit publication checks for one compressed exact-C outcome.
pub fn record_exact_c_compact_commit(
    reconstruction_succeeded: bool,
    binding_map_valid: bool,
    alpha_candidate: bool,
    alpha_published: bool,
) {
    RECORDER.record(|all| {
        let exact_c = &mut all.exact_c;
        if exact_c.compact_commits >= exact_c.committed
            || exact_c.pending_compressed_layouts.is_empty()
            || (alpha_published
                && (!alpha_candidate || !reconstruction_succeeded || !binding_map_valid))
        {
            exact_c.attribution_conflict = true;
        }
        exact_c.counter_overflow |= increment_counter(&mut exact_c.compact_commits);
        if reconstruction_succeeded {
            exact_c.counter_overflow |=
                increment_counter(&mut exact_c.compact_reconstruction_succeeded);
        } else {
            exact_c.counter_overflow |=
                increment_counter(&mut exact_c.compact_reconstruction_failed);
        }
        if binding_map_valid {
            exact_c.counter_overflow |=
                increment_counter(&mut exact_c.compact_binding_map_succeeded);
        } else {
            exact_c.counter_overflow |= increment_counter(&mut exact_c.compact_binding_map_failed);
        }
        if alpha_candidate {
            exact_c.counter_overflow |= increment_counter(&mut exact_c.compact_alpha_candidates);
            if alpha_published {
                exact_c.counter_overflow |= increment_counter(&mut exact_c.compact_alpha_published);
            } else {
                exact_c.counter_overflow |= increment_counter(&mut exact_c.compact_alpha_dropped);
            }
        } else if alpha_published {
            exact_c.attribution_conflict = true;
        }
    });
}

/// Record the root-prune runtime dispatcher. This is honest OFF evidence too;
/// it does not imply that a plan was constructed or applied.
pub fn record_root_spec_prune_route(configured: bool) {
    RECORDER.record(|all| {
        let prune = &mut all.root_spec_prune;
        prune.observed = true;
        if prune.route_observations == 0 {
            prune.configured = Some(configured);
        } else if prune.configured != Some(configured) {
            prune.configured = None;
            prune.route_conflict = true;
            prune.attribution_conflict = true;
        }
        prune.counter_overflow |= increment_counter(&mut prune.route_observations);
    });
}

/// Record construction of one validated root-prune plan.
pub fn record_root_spec_prune_plan(
    source_rows: usize,
    evaluated_rows: usize,
    precertified_rows: usize,
) {
    RECORDER.record(|all| {
        let prune = &mut all.root_spec_prune;
        if prune.configured != Some(true)
            || RootRowLayout::new(source_rows, evaluated_rows, precertified_rows).is_none()
        {
            prune.attribution_conflict = true;
        }
        prune.counter_overflow |= increment_counter(&mut prune.plans_built);
    });
}

/// Record final publication of a root-prune layout. When `pair_exact_c` is
/// true, the same layout must be the oldest unresolved compressed selection.
pub fn record_root_spec_prune_applied(
    source_rows: usize,
    evaluated_rows: usize,
    precertified_rows: usize,
    all_pruned: bool,
    pair_exact_c: bool,
) {
    RECORDER.record(|all| {
        let layout = RootRowLayout::new(source_rows, evaluated_rows, precertified_rows);
        let prune = &mut all.root_spec_prune;
        if prune.configured != Some(true)
            || prune.applied >= prune.plans_built
            || layout.is_none()
            || all_pruned != (evaluated_rows == 0 && precertified_rows == source_rows)
        {
            prune.attribution_conflict = true;
        }
        prune.counter_overflow |= increment_counter(&mut prune.applied);
        prune.counter_overflow |= increment_counter(&mut prune.layout_observations);
        prune.counter_overflow |= add_counter(&mut prune.source_rows, source_rows);
        prune.counter_overflow |= add_counter(&mut prune.evaluated_rows, evaluated_rows);
        prune.counter_overflow |= add_counter(&mut prune.precertified_rows, precertified_rows);
        if all_pruned {
            prune.counter_overflow |= increment_counter(&mut prune.all_pruned);
        }
        if pair_exact_c {
            resolve_exact_c_compressed_layout(
                &mut all.exact_c,
                source_rows,
                evaluated_rows,
                precertified_rows,
                true,
            );
        }
    });
}

/// Record a typed exact-C refusal before backend commitment.
pub fn record_exact_c_refused_before_commit(stop_reason: &'static str) {
    RECORDER.record(|all| {
        let exact_c = &mut all.exact_c;
        exact_c.observed = true;
        if exact_c.outcomes_observed >= exact_c.selections {
            exact_c.attribution_conflict = true;
        }
        exact_c.counter_overflow |= increment_counter(&mut exact_c.outcomes_observed);
        exact_c.counter_overflow |= increment_counter(&mut exact_c.refused_before_commit);
        exact_c.counter_overflow |= increment_reason(&mut exact_c.stop_reasons, stop_reason);
    });
}

/// Record an impossible typed-route mismatch without classifying it as either
/// a refusal or a commit. Consumers must reject the attribution conflict.
pub fn record_exact_c_attribution_conflict(stop_reason: &'static str) {
    RECORDER.record(|all| {
        let exact_c = &mut all.exact_c;
        exact_c.observed = true;
        exact_c.attribution_conflict = true;
        exact_c.counter_overflow |= increment_counter(&mut exact_c.outcomes_observed);
        exact_c.counter_overflow |= increment_reason(&mut exact_c.stop_reasons, stop_reason);
    });
}

/// Record a typed exact-C committed outcome.
///
/// Iteration counts remain absent when the underlying outcome does not carry
/// them; zero is recorded only when the outcome explicitly reports zero.
pub fn record_exact_c_committed(
    attempted_iterations: Option<usize>,
    accepted_iterations: Option<usize>,
    stop_reason: &'static str,
) {
    RECORDER.record(|all| {
        update_exact_c_committed(
            &mut all.exact_c,
            attempted_iterations,
            accepted_iterations,
            stop_reason,
        );
    });
}

/// Record a typed multi-iteration outcome together with the exact child gate
/// and gradient shape that reached the proposal seam.
#[allow(clippy::too_many_arguments)]
pub fn record_exact_c_multi_iteration_committed(
    attempted_iterations: usize,
    accepted_iterations: usize,
    multiplicative_weights_requested: bool,
    multiplicative_weights_plan_dispatched: bool,
    multiplicative_weights_effective: bool,
    completed_proposals: usize,
    adaptive_plan_dispatches: usize,
    gradient_plan_num_specs: Option<usize>,
    gradient_row_count: usize,
    stop_reason: &'static str,
) {
    RECORDER.record(|all| {
        let exact_c = &mut all.exact_c;
        update_exact_c_committed(
            exact_c,
            Some(attempted_iterations),
            Some(accepted_iterations),
            stop_reason,
        );
        let first_evidence = exact_c.multi_iteration_evidence_outcomes == 0;
        if !valid_exact_c_multi_iteration_evidence(
            attempted_iterations,
            accepted_iterations,
            multiplicative_weights_requested,
            multiplicative_weights_plan_dispatched,
            multiplicative_weights_effective,
            completed_proposals,
            adaptive_plan_dispatches,
            gradient_plan_num_specs,
            gradient_row_count,
        ) {
            exact_c.multi_iteration_evidence_conflict = true;
        }
        exact_c.counter_overflow |=
            increment_counter(&mut exact_c.multi_iteration_evidence_outcomes);
        exact_c.counter_overflow |=
            add_counter(&mut exact_c.completed_proposals, completed_proposals);
        exact_c.counter_overflow |= add_counter(
            &mut exact_c.adaptive_plan_dispatches,
            adaptive_plan_dispatches,
        );
        if multiplicative_weights_plan_dispatched {
            exact_c.counter_overflow |=
                increment_counter(&mut exact_c.multiplicative_weights_plan_dispatched_outcomes);
        }
        if multiplicative_weights_effective {
            exact_c.counter_overflow |=
                increment_counter(&mut exact_c.multiplicative_weights_effective_outcomes);
        }

        if first_evidence {
            exact_c.multiplicative_weights_requested = Some(multiplicative_weights_requested);
            exact_c.gradient_row_count = Some(gradient_row_count);
        } else {
            if !exact_c.multiplicative_weights_requested_conflict
                && exact_c.multiplicative_weights_requested
                    != Some(multiplicative_weights_requested)
            {
                exact_c.multiplicative_weights_requested = None;
                exact_c.multiplicative_weights_requested_conflict = true;
            }
            if !exact_c.gradient_row_count_conflict
                && exact_c.gradient_row_count != Some(gradient_row_count)
            {
                exact_c.gradient_row_count = None;
                exact_c.gradient_row_count_conflict = true;
            }
        }
        if let Some(num_specs) = gradient_plan_num_specs {
            if !exact_c.gradient_plan_num_specs_conflict {
                match exact_c.gradient_plan_num_specs {
                    None => exact_c.gradient_plan_num_specs = Some(num_specs),
                    Some(existing) if existing == num_specs => {}
                    Some(_) => {
                        exact_c.gradient_plan_num_specs = None;
                        exact_c.gradient_plan_num_specs_conflict = true;
                    }
                }
            }
        }
    });
}

fn update_exact_c_committed(
    exact_c: &mut ExactCObservations,
    attempted_iterations: Option<usize>,
    accepted_iterations: Option<usize>,
    stop_reason: &'static str,
) {
    exact_c.observed = true;
    if exact_c.outcomes_observed >= exact_c.selections {
        exact_c.attribution_conflict = true;
    }
    exact_c.counter_overflow |= increment_counter(&mut exact_c.outcomes_observed);
    exact_c.counter_overflow |= increment_counter(&mut exact_c.committed);
    match (attempted_iterations, accepted_iterations) {
        (Some(attempted), Some(accepted)) if accepted <= attempted => {
            exact_c.counter_overflow |= increment_counter(&mut exact_c.iteration_count_outcomes);
            exact_c.counter_overflow |= add_counter(&mut exact_c.attempted_iterations, attempted);
            exact_c.counter_overflow |= add_counter(&mut exact_c.accepted_iterations, accepted);
        }
        (None, None) => {}
        _ => exact_c.iteration_count_conflict = true,
    }
    exact_c.counter_overflow |= increment_reason(&mut exact_c.stop_reasons, stop_reason);
}

pub fn record_invprop_clause_rebind_attempt() {
    RECORDER.record(|all| {
        let invprop = &mut all.invprop;
        invprop.observed = true;
        invprop.counter_overflow |= increment_counter(&mut invprop.clause_rebind_attempts);
    });
}

pub fn record_invprop_clause_rebind_accepted() {
    RECORDER.record(|all| {
        let invprop = &mut all.invprop;
        invprop.observed = true;
        if checked_sum([
            invprop.clause_rebind_accepted,
            invprop.clause_rebind_refused,
        ])
        .is_none_or(|outcomes| outcomes >= invprop.clause_rebind_attempts)
        {
            invprop.attribution_conflict = true;
        }
        invprop.counter_overflow |= increment_counter(&mut invprop.clause_rebind_accepted);
    });
}

pub fn record_invprop_clause_rebind_refused() {
    RECORDER.record(|all| {
        let invprop = &mut all.invprop;
        invprop.observed = true;
        if checked_sum([
            invprop.clause_rebind_accepted,
            invprop.clause_rebind_refused,
        ])
        .is_none_or(|outcomes| outcomes >= invprop.clause_rebind_attempts)
        {
            invprop.attribution_conflict = true;
        }
        invprop.counter_overflow |= increment_counter(&mut invprop.clause_rebind_refused);
    });
}

pub fn record_invprop_alpha_initialization() {
    RECORDER.record(|all| {
        let invprop = &mut all.invprop;
        invprop.observed = true;
        invprop.counter_overflow |= increment_counter(&mut invprop.alpha_initializations);
    });
}

pub fn record_invprop_gamma_step_attempted() {
    RECORDER.record(|all| {
        let invprop = &mut all.invprop;
        invprop.observed = true;
        invprop.counter_overflow |= increment_counter(&mut invprop.gamma_steps_attempted);
    });
}

pub fn record_invprop_gamma_step_applied() {
    RECORDER.record(|all| {
        let invprop = &mut all.invprop;
        invprop.observed = true;
        if invprop.gamma_steps_applied >= invprop.gamma_steps_attempted {
            invprop.attribution_conflict = true;
        }
        invprop.counter_overflow |= increment_counter(&mut invprop.gamma_steps_applied);
    });
}

/// Record a successfully constructed, genuinely changed output identity seed.
///
/// The producer calls this only after the conjunction, shape, and identity
/// admission gates and only when at least one coefficient/error or bias term
/// changed. Optimizer probes count because they execute the same certified
/// fold; consumers must keep this distinct from accepted gamma writes.
pub fn record_invprop_nonzero_output_seed_fold() {
    let pending_overflow = accumulate_pending_evaluated_fold();
    RECORDER.record(|all| {
        let invprop = &mut all.invprop;
        invprop.observed = true;
        invprop.counter_overflow |= increment_counter(&mut invprop.nonzero_output_seed_folds);
        invprop.counter_overflow |= pending_overflow;
    });
}

/// Record the runtime dispatch decision for the typed fresh-domain clip.
/// This is route evidence, not evidence that a clip attempt executed.
pub fn record_fresh_domain_clip_route(configured: bool, route_authorized: bool) {
    RECORDER.record(|all| {
        let fresh = &mut all.fresh_domain_clip;
        fresh.observed = true;
        if fresh.route_observations == 0 {
            fresh.configured = Some(configured);
            fresh.route_authorized = Some(route_authorized);
        } else if fresh.configured != Some(configured)
            || fresh.route_authorized != Some(route_authorized)
        {
            fresh.configured = None;
            fresh.route_authorized = None;
            fresh.route_conflict = true;
            fresh.attribution_conflict = true;
        }
        if route_authorized && !configured {
            fresh.attribution_conflict = true;
        }
        fresh.counter_overflow |= increment_counter(&mut fresh.route_observations);
    });
}

/// Record one atomic local clip disposition and its actual tightened width.
pub fn record_fresh_domain_clip_outcome(
    disposition: FreshDomainClipDisposition,
    tightened_dimensions: usize,
) {
    RECORDER.record(|all| {
        let fresh = &mut all.fresh_domain_clip;
        fresh.observed = true;
        fresh.counter_overflow |= increment_counter(&mut fresh.attempts);
        if fresh.route_authorized != Some(true) {
            fresh.attribution_conflict = true;
        }
        match disposition {
            FreshDomainClipDisposition::Applied => {
                fresh.counter_overflow |= increment_counter(&mut fresh.applied);
                fresh.counter_overflow |=
                    add_counter(&mut fresh.tightened_dimensions, tightened_dimensions);
            }
            FreshDomainClipDisposition::AllClausesRefuted => {
                fresh.counter_overflow |= increment_counter(&mut fresh.all_clauses_refuted);
                if tightened_dimensions > 0 {
                    fresh.attribution_conflict = true;
                }
            }
            FreshDomainClipDisposition::Skipped => {
                fresh.counter_overflow |= increment_counter(&mut fresh.skipped);
                if tightened_dimensions > 0 {
                    fresh.attribution_conflict = true;
                }
            }
        }
    });
}

fn patches_purpose_observations(
    patches: &mut PatchesMaterializationObservations,
    purpose: PatchesMaterializationPurpose,
) -> &mut PatchesMaterializationPurposeObservations {
    match purpose {
        PatchesMaterializationPurpose::LatentInputCrossover => &mut patches.latent_input_crossover,
        PatchesMaterializationPurpose::NetworkInputTerminal => &mut patches.network_input_terminal,
        PatchesMaterializationPurpose::Other => &mut patches.other,
    }
}

/// Record entry into one full Patches-to-Dense materializer.
pub(crate) fn record_patches_materialization_attempt(
    purpose: PatchesMaterializationPurpose,
    finite_deadline: bool,
    geometry: PatchesMaterializationGeometry,
    input_coefficient_error: bool,
) {
    RECORDER.record(|all| {
        let patches = &mut all.patches_materialization;
        patches.observed = true;
        patches.counter_overflow |= increment_counter(&mut patches.attempts);
        let purpose_overflow = {
            let purpose = patches_purpose_observations(patches, purpose);
            increment_counter(&mut purpose.attempts)
        };
        patches.counter_overflow |= purpose_overflow;
        if finite_deadline {
            patches.counter_overflow |= increment_counter(&mut patches.finite_deadline_attempts);
        } else {
            patches.counter_overflow |= increment_counter(&mut patches.no_deadline_attempts);
        }
        match geometry {
            PatchesMaterializationGeometry::Affine => {
                patches.counter_overflow |=
                    increment_counter(&mut patches.affine_geometry_attempts);
            }
            PatchesMaterializationGeometry::Anchored => {
                patches.counter_overflow |=
                    increment_counter(&mut patches.anchored_geometry_attempts);
            }
            PatchesMaterializationGeometry::Conflicting => {
                patches.counter_overflow |=
                    increment_counter(&mut patches.conflicting_geometry_attempts);
            }
        }
        if input_coefficient_error {
            patches.counter_overflow |=
                increment_counter(&mut patches.input_coefficient_error_attempts);
        }
    });
}

/// Record one successful full materialization and its exact admission receipt.
pub(crate) fn record_patches_materialization_success(
    purpose: PatchesMaterializationPurpose,
    coefficient_error: PatchesCoefficientErrorDisposition,
    receipt: PatchesMaterializationMemoryReceipt,
) {
    RECORDER.record(|all| {
        let patches = &mut all.patches_materialization;
        let purpose_attempts = patches_purpose_observations(patches, purpose).attempts;
        let purpose_outcomes = {
            let purpose = patches_purpose_observations(patches, purpose);
            checked_sum([purpose.succeeded, purpose.refused])
        };
        if checked_sum([patches.succeeded, patches.refused])
            .is_none_or(|outcomes| outcomes >= patches.attempts)
            || purpose_outcomes.is_none_or(|outcomes| outcomes >= purpose_attempts)
            || receipt
                .nominal_required_bytes
                .checked_add(receipt.capacity_overage_bytes)
                != Some(receipt.admitted_bytes)
            || receipt.admitted_bytes > receipt.budget_bytes
        {
            patches.attribution_conflict = true;
        }

        patches.counter_overflow |= increment_counter(&mut patches.succeeded);
        let purpose_overflow = {
            let purpose = patches_purpose_observations(patches, purpose);
            increment_counter(&mut purpose.succeeded)
        };
        patches.counter_overflow |= purpose_overflow;
        match coefficient_error {
            PatchesCoefficientErrorDisposition::Absent => {
                patches.counter_overflow |=
                    increment_counter(&mut patches.coefficient_error_absent);
            }
            PatchesCoefficientErrorDisposition::Materialized => {
                patches.counter_overflow |=
                    increment_counter(&mut patches.coefficient_error_materialized);
            }
        }
        patches.counter_overflow |= increment_counter(&mut patches.memory_receipt_outcomes);
        patches.counter_overflow |= add_counter(
            &mut patches.nominal_required_bytes,
            receipt.nominal_required_bytes,
        );
        patches.counter_overflow |= add_counter(
            &mut patches.capacity_overage_bytes,
            receipt.capacity_overage_bytes,
        );
        patches.counter_overflow |=
            add_counter(&mut patches.admitted_bytes, receipt.admitted_bytes);
        patches.counter_overflow |= add_counter(&mut patches.budget_bytes, receipt.budget_bytes);
    });
}

/// Record the typed failure of one full materialization attempt.
pub(crate) fn record_patches_materialization_refusal(
    purpose: PatchesMaterializationPurpose,
    refusal: PatchesMaterializationRefusal,
) {
    RECORDER.record(|all| {
        let patches = &mut all.patches_materialization;
        let purpose_attempts = patches_purpose_observations(patches, purpose).attempts;
        let purpose_outcomes = {
            let purpose = patches_purpose_observations(patches, purpose);
            checked_sum([purpose.succeeded, purpose.refused])
        };
        if checked_sum([patches.succeeded, patches.refused])
            .is_none_or(|outcomes| outcomes >= patches.attempts)
            || purpose_outcomes.is_none_or(|outcomes| outcomes >= purpose_attempts)
        {
            patches.attribution_conflict = true;
        }

        patches.counter_overflow |= increment_counter(&mut patches.refused);
        let purpose_overflow = {
            let purpose = patches_purpose_observations(patches, purpose);
            increment_counter(&mut purpose.refused)
        };
        patches.counter_overflow |= purpose_overflow;
        match refusal {
            PatchesMaterializationRefusal::Memory => {
                patches.counter_overflow |= increment_counter(&mut patches.memory_refusals);
            }
            PatchesMaterializationRefusal::Deadline => {
                patches.counter_overflow |= increment_counter(&mut patches.deadline_refusals);
            }
            PatchesMaterializationRefusal::Semantic => {
                patches.counter_overflow |= increment_counter(&mut patches.semantic_refusals);
            }
        }
    });
}

fn update_exact_c_selection(
    exact_c: &mut ExactCObservations,
    iteration_limit: usize,
    source_rows: usize,
    evaluated_rows: usize,
    precertified_rows: usize,
) {
    exact_c.observed = true;
    exact_c.counter_overflow |= increment_counter(&mut exact_c.selections);
    exact_c.counter_overflow |= increment_counter(&mut exact_c.layout_observations);
    exact_c.counter_overflow |= add_counter(&mut exact_c.source_rows, source_rows);
    exact_c.counter_overflow |= add_counter(&mut exact_c.evaluated_rows, evaluated_rows);
    exact_c.counter_overflow |= add_counter(&mut exact_c.precertified_rows, precertified_rows);

    if !exact_c.selected_iteration_limit_conflict {
        match exact_c.selected_iteration_limit {
            None => exact_c.selected_iteration_limit = Some(iteration_limit),
            Some(existing) if existing == iteration_limit => {}
            Some(_) => {
                exact_c.selected_iteration_limit = None;
                exact_c.selected_iteration_limit_conflict = true;
            }
        }
    }

    let compressed = precertified_rows > 0;
    if !exact_c.selected_compressed_conflict {
        match exact_c.selected_compressed {
            None => exact_c.selected_compressed = Some(compressed),
            Some(existing) if existing == compressed => {}
            Some(_) => {
                exact_c.selected_compressed = None;
                exact_c.selected_compressed_conflict = true;
            }
        }
    }

    let Some(layout) = RootRowLayout::new(source_rows, evaluated_rows, precertified_rows) else {
        exact_c.attribution_conflict = true;
        return;
    };
    if layout.compressed() {
        exact_c.counter_overflow |= increment_counter(&mut exact_c.compressed_selections);
        exact_c.pending_compressed_layouts.push_back(layout);
    }
}

fn resolve_exact_c_compressed_layout(
    exact_c: &mut ExactCObservations,
    source_rows: usize,
    evaluated_rows: usize,
    precertified_rows: usize,
    finalized: bool,
) {
    let supplied = RootRowLayout::new(source_rows, evaluated_rows, precertified_rows);
    let pending = exact_c.pending_compressed_layouts.pop_front();
    if supplied.is_none_or(|layout| !layout.compressed()) || pending != supplied {
        exact_c.attribution_conflict = true;
    }
    if finalized {
        exact_c.counter_overflow |= increment_counter(&mut exact_c.compressed_layouts_finalized);
    } else {
        exact_c.counter_overflow |= increment_counter(&mut exact_c.compressed_layouts_rolled_back);
    }
}

fn add_counter(counter: &mut usize, amount: usize) -> bool {
    if let Some(next) = counter.checked_add(amount) {
        *counter = next;
        false
    } else {
        *counter = usize::MAX;
        true
    }
}

fn increment_counter(counter: &mut usize) -> bool {
    add_counter(counter, 1)
}

fn increment_reason(reasons: &mut BTreeMap<String, usize>, reason: &'static str) -> bool {
    let count = reasons.entry(reason.to_string()).or_default();
    increment_counter(count)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::{
        begin_invprop_evaluated_fold_scope, begin_run, record_exact_c_compact_commit,
        record_exact_c_multi_iteration_committed, record_exact_c_refused_before_commit,
        record_exact_c_selected, record_fresh_domain_clip_outcome, record_fresh_domain_clip_route,
        record_invprop_alpha_initialization, record_invprop_clause_rebind_accepted,
        record_invprop_clause_rebind_attempt, record_invprop_gamma_step_applied,
        record_invprop_gamma_step_attempted, record_invprop_nonzero_output_seed_fold,
        record_patches_materialization_attempt, record_patches_materialization_refusal,
        record_patches_materialization_success, record_root_spec_prune_applied,
        record_root_spec_prune_plan, record_root_spec_prune_route, snapshot,
        update_exact_c_selection, ExactCObservations, FreshDomainClipDisposition,
        PatchesCoefficientErrorDisposition, PatchesMaterializationGeometry,
        PatchesMaterializationMemoryReceipt, PatchesMaterializationPurpose,
        PatchesMaterializationRefusal, Recorder,
    };

    #[test]
    fn observations_are_run_scoped_and_overlap_fails_closed() {
        let recorder = Recorder::new();

        recorder.record(|all| all.invprop.gamma_steps_applied = 99);
        assert_eq!(recorder.snapshot(), Default::default());

        {
            let first = recorder.begin();
            recorder.record(|all| {
                all.exact_c.observed = true;
                all.exact_c.selections = 1;
                all.invprop.observed = true;
                all.invprop.gamma_steps_attempted = 2;
            });
            let observed = recorder.snapshot();
            assert!(observed.run_active);
            assert!(!observed.recording_conflict);
            assert_eq!(observed.exact_c.selections, 1);
            assert_eq!(observed.invprop.gamma_steps_attempted, 2);

            let overlapping = recorder.begin();
            let conflicted = recorder.snapshot();
            assert!(!conflicted.run_active);
            assert!(conflicted.recording_conflict);
            assert!(!conflicted.exact_c.observed);
            assert!(!conflicted.invprop.observed);

            recorder.record(|all| all.exact_c.selections = 200);
            drop(first);
            assert!(recorder.snapshot().recording_conflict);
            drop(overlapping);
        }
        assert_eq!(recorder.snapshot(), Default::default());

        let fresh = recorder.begin();
        let reset = recorder.snapshot();
        assert!(reset.run_active);
        assert_eq!(reset.exact_c.selections, 0);
        assert_eq!(reset.invprop.gamma_steps_attempted, 0);
        drop(fresh);
        assert_eq!(recorder.snapshot(), Default::default());

        let exhausted = Recorder::new();
        exhausted.next_generation.store(u64::MAX, Ordering::Relaxed);
        let first_exhausted = exhausted.begin();
        let second_exhausted = exhausted.begin();
        assert!(exhausted.snapshot().recording_conflict);
        drop(first_exhausted);
        assert!(
            exhausted.snapshot().recording_conflict,
            "one duplicate exhausted generation must not clear another live guard"
        );
        drop(second_exhausted);
        assert_eq!(exhausted.snapshot(), Default::default());
    }

    #[test]
    fn test_recorder_ignores_foreign_harness_thread_events() {
        let recorder = Recorder::new();
        let _run = recorder.begin();
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    recorder.record(|all| {
                        all.invprop.observed = true;
                        all.invprop.gamma_steps_attempted = 99;
                    });
                })
                .join()
                .expect("foreign recorder thread");
        });
        recorder.record(|all| {
            all.invprop.observed = true;
            all.invprop.gamma_steps_attempted = 1;
        });

        let snapshot = recorder.snapshot();
        assert_eq!(snapshot.invprop.gamma_steps_attempted, 1);
        assert!(snapshot.invprop.observed);
    }

    #[test]
    fn exact_c_conflicting_limits_are_not_misattributed() {
        let mut exact_c = ExactCObservations::default();
        update_exact_c_selection(&mut exact_c, 2, 4, 4, 0);
        update_exact_c_selection(&mut exact_c, 4, 4, 4, 0);
        assert_eq!(exact_c.selections, 2);
        assert_eq!(exact_c.selected_iteration_limit, None);
        assert!(exact_c.selected_iteration_limit_conflict);

        let mut overflow = ExactCObservations {
            selections: usize::MAX,
            ..ExactCObservations::default()
        };
        update_exact_c_selection(&mut overflow, 4, 4, 4, 0);
        assert_eq!(overflow.selections, usize::MAX);
        assert!(overflow.counter_overflow);
    }

    #[test]
    fn public_events_are_attributed_and_cleared_after_the_solve() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        assert!(!snapshot().run_active);
        {
            let _run = begin_run();
            record_exact_c_selected(4, 4, 4, 0);
            record_exact_c_multi_iteration_committed(
                3,
                2,
                true,
                true,
                true,
                3,
                2,
                Some(4),
                4,
                "work_deadline_exceeded",
            );
            record_invprop_clause_rebind_attempt();
            record_invprop_clause_rebind_accepted();
            record_invprop_alpha_initialization();
            record_invprop_gamma_step_attempted();
            record_invprop_gamma_step_applied();
            {
                let evaluated = begin_invprop_evaluated_fold_scope();
                record_invprop_nonzero_output_seed_fold();
                evaluated.commit();
            }
            record_fresh_domain_clip_route(true, true);
            record_fresh_domain_clip_outcome(FreshDomainClipDisposition::Applied, 2);

            let observed = snapshot();
            assert!(observed.run_active);
            assert_eq!(observed.schema, "ny_beta_crown_execution_observations_v5");
            assert_eq!(observed.exact_c.selected_iteration_limit, Some(4));
            assert_eq!(observed.exact_c.attempted_iterations, 3);
            assert_eq!(observed.exact_c.accepted_iterations, 2);
            assert_eq!(observed.exact_c.multi_iteration_evidence_outcomes, 1);
            assert_eq!(
                observed.exact_c.multiplicative_weights_requested,
                Some(true)
            );
            assert_eq!(
                observed
                    .exact_c
                    .multiplicative_weights_plan_dispatched_outcomes,
                1
            );
            assert_eq!(
                observed.exact_c.multiplicative_weights_effective_outcomes,
                1
            );
            assert_eq!(observed.exact_c.completed_proposals, 3);
            assert_eq!(observed.exact_c.adaptive_plan_dispatches, 2);
            assert_eq!(observed.exact_c.gradient_plan_num_specs, Some(4));
            assert_eq!(observed.exact_c.gradient_row_count, Some(4));
            assert_eq!(observed.exact_c.stop_reasons["work_deadline_exceeded"], 1);
            assert_eq!(
                observed.exact_c.selections,
                observed.exact_c.outcomes_observed
            );
            assert_eq!(
                observed.exact_c.refused_before_commit + observed.exact_c.committed,
                observed.exact_c.outcomes_observed
            );
            assert_eq!(
                observed
                    .exact_c
                    .stop_reasons
                    .values()
                    .copied()
                    .sum::<usize>(),
                observed.exact_c.outcomes_observed
            );
            assert!(!observed.exact_c.attribution_conflict);
            assert_eq!(observed.invprop.clause_rebind_accepted, 1);
            assert_eq!(observed.invprop.alpha_initializations, 1);
            assert_eq!(observed.invprop.gamma_steps_applied, 1);
            assert_eq!(observed.invprop.nonzero_output_seed_folds, 1);
            assert_eq!(observed.invprop.nonzero_evaluated_output_seed_folds, 1);
            assert_eq!(
                observed.invprop.clause_rebind_accepted + observed.invprop.clause_rebind_refused,
                observed.invprop.clause_rebind_attempts
            );
            assert!(observed.invprop.gamma_steps_applied <= observed.invprop.gamma_steps_attempted);
            assert!(!observed.invprop.attribution_conflict);
            assert_eq!(observed.fresh_domain_clip.configured, Some(true));
            assert_eq!(observed.fresh_domain_clip.route_authorized, Some(true));
            assert_eq!(observed.fresh_domain_clip.attempts, 1);
            assert_eq!(observed.fresh_domain_clip.applied, 1);
            assert_eq!(observed.fresh_domain_clip.tightened_dimensions, 2);
            assert!(!observed.fresh_domain_clip.attribution_conflict);
        }
        assert_eq!(snapshot(), Default::default());
    }

    #[test]
    fn patches_materialization_records_typed_success_and_refusals() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let _run = begin_run();

        record_patches_materialization_attempt(
            PatchesMaterializationPurpose::NetworkInputTerminal,
            true,
            PatchesMaterializationGeometry::Anchored,
            true,
        );
        record_patches_materialization_success(
            PatchesMaterializationPurpose::NetworkInputTerminal,
            PatchesCoefficientErrorDisposition::Materialized,
            PatchesMaterializationMemoryReceipt {
                nominal_required_bytes: 100,
                capacity_overage_bytes: 8,
                admitted_bytes: 108,
                budget_bytes: 128,
            },
        );
        record_patches_materialization_attempt(
            PatchesMaterializationPurpose::LatentInputCrossover,
            false,
            PatchesMaterializationGeometry::Affine,
            false,
        );
        record_patches_materialization_refusal(
            PatchesMaterializationPurpose::LatentInputCrossover,
            PatchesMaterializationRefusal::Memory,
        );
        record_patches_materialization_attempt(
            PatchesMaterializationPurpose::Other,
            true,
            PatchesMaterializationGeometry::Conflicting,
            false,
        );
        record_patches_materialization_refusal(
            PatchesMaterializationPurpose::Other,
            PatchesMaterializationRefusal::Semantic,
        );

        let observed = snapshot().patches_materialization;
        assert!(observed.observed);
        assert!(!observed.attribution_conflict);
        assert!(!observed.counter_overflow);
        assert_eq!(observed.attempts, 3);
        assert_eq!(observed.succeeded, 1);
        assert_eq!(observed.refused, 2);
        assert_eq!(observed.network_input_terminal.attempts, 1);
        assert_eq!(observed.network_input_terminal.succeeded, 1);
        assert_eq!(observed.latent_input_crossover.attempts, 1);
        assert_eq!(observed.latent_input_crossover.refused, 1);
        assert_eq!(observed.other.attempts, 1);
        assert_eq!(observed.other.refused, 1);
        assert_eq!(observed.finite_deadline_attempts, 2);
        assert_eq!(observed.no_deadline_attempts, 1);
        assert_eq!(observed.anchored_geometry_attempts, 1);
        assert_eq!(observed.affine_geometry_attempts, 1);
        assert_eq!(observed.conflicting_geometry_attempts, 1);
        assert_eq!(observed.input_coefficient_error_attempts, 1);
        assert_eq!(observed.coefficient_error_materialized, 1);
        assert_eq!(observed.memory_refusals, 1);
        assert_eq!(observed.semantic_refusals, 1);
        assert_eq!(observed.deadline_refusals, 0);
        assert_eq!(observed.memory_receipt_outcomes, 1);
        assert_eq!(observed.nominal_required_bytes, 100);
        assert_eq!(observed.capacity_overage_bytes, 8);
        assert_eq!(observed.admitted_bytes, 108);
        assert_eq!(observed.budget_bytes, 128);
    }

    #[test]
    fn patches_materialization_invalid_receipt_fails_closed() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let _run = begin_run();
        record_patches_materialization_attempt(
            PatchesMaterializationPurpose::Other,
            false,
            PatchesMaterializationGeometry::Affine,
            false,
        );
        record_patches_materialization_success(
            PatchesMaterializationPurpose::Other,
            PatchesCoefficientErrorDisposition::Absent,
            PatchesMaterializationMemoryReceipt {
                nominal_required_bytes: 10,
                capacity_overage_bytes: 2,
                admitted_bytes: 11,
                budget_bytes: 11,
            },
        );

        let observed = snapshot().patches_materialization;
        assert_eq!(observed.succeeded, 1);
        assert!(observed.attribution_conflict);
    }

    #[test]
    fn patches_materialization_snapshot_rejects_incomplete_and_overflowed_streams() {
        let incomplete = Recorder::new();
        let _incomplete_run = incomplete.begin();
        incomplete.record(|all| {
            let patches = &mut all.patches_materialization;
            patches.observed = true;
            patches.attempts = 1;
            patches.other.attempts = 1;
            patches.no_deadline_attempts = 1;
            patches.affine_geometry_attempts = 1;
        });
        assert!(
            incomplete
                .snapshot()
                .patches_materialization
                .attribution_conflict
        );

        let overflow = Recorder::new();
        let _overflow_run = overflow.begin();
        overflow.record(|all| {
            let patches = &mut all.patches_materialization;
            patches.attempts = usize::MAX;
            patches.other.attempts = usize::MAX;
            patches.no_deadline_attempts = usize::MAX;
            patches.affine_geometry_attempts = usize::MAX;
        });
        overflow.record(|all| {
            let patches = &mut all.patches_materialization;
            patches.observed = true;
            patches.counter_overflow |= super::increment_counter(&mut patches.attempts);
        });
        let overflowed = overflow.snapshot().patches_materialization;
        assert_eq!(overflowed.attempts, usize::MAX);
        assert!(overflowed.counter_overflow);
        assert!(overflowed.attribution_conflict);
    }

    #[test]
    fn exact_c_compact_publication_pairs_with_the_same_prune_layout() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let _run = begin_run();
        record_root_spec_prune_route(true);
        record_root_spec_prune_plan(4, 2, 2);
        record_exact_c_selected(4, 4, 2, 2);
        record_exact_c_multi_iteration_committed(
            2,
            1,
            false,
            false,
            false,
            2,
            0,
            Some(1),
            2,
            "iteration_limit",
        );
        record_exact_c_compact_commit(true, true, true, true);
        record_root_spec_prune_applied(4, 2, 2, false, true);

        let observed = snapshot();
        assert!(!observed.exact_c.attribution_conflict);
        assert_eq!(observed.exact_c.selected_compressed, Some(true));
        assert_eq!(observed.exact_c.compressed_selections, 1);
        assert_eq!(observed.exact_c.compressed_layouts_finalized, 1);
        assert_eq!(observed.exact_c.compact_commits, 1);
        assert_eq!(observed.exact_c.compact_reconstruction_succeeded, 1);
        assert_eq!(observed.exact_c.compact_binding_map_succeeded, 1);
        assert_eq!(observed.exact_c.compact_alpha_published, 1);
        assert!(!observed.root_spec_prune.attribution_conflict);
        assert_eq!(observed.root_spec_prune.configured, Some(true));
        assert_eq!(observed.root_spec_prune.applied, 1);
        assert_eq!(observed.root_spec_prune.source_rows, 4);
        assert_eq!(observed.root_spec_prune.evaluated_rows, 2);
        assert_eq!(observed.root_spec_prune.precertified_rows, 2);
    }

    #[test]
    fn exact_c_multi_iteration_evidence_distinguishes_admission_from_completion() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let _run = begin_run();

        record_exact_c_selected(4, 4, 4, 0);
        record_exact_c_multi_iteration_committed(
            0,
            0,
            true,
            false,
            false,
            0,
            0,
            None,
            4,
            "invalid_multiplicative_weights",
        );
        record_exact_c_selected(4, 4, 4, 0);
        record_exact_c_multi_iteration_committed(
            1,
            0,
            true,
            true,
            false,
            0,
            0,
            Some(4),
            4,
            "joint_unavailable",
        );
        record_exact_c_selected(4, 4, 4, 0);
        record_exact_c_multi_iteration_committed(
            1,
            0,
            true,
            true,
            false,
            1,
            0,
            Some(4),
            4,
            "iteration_limit",
        );

        let observed = snapshot();
        assert!(!observed.exact_c.attribution_conflict);
        assert_eq!(observed.exact_c.multi_iteration_evidence_outcomes, 3);
        assert_eq!(
            observed.exact_c.multiplicative_weights_requested,
            Some(true)
        );
        assert_eq!(
            observed
                .exact_c
                .multiplicative_weights_plan_dispatched_outcomes,
            2
        );
        assert_eq!(
            observed.exact_c.multiplicative_weights_effective_outcomes,
            0
        );
        assert_eq!(observed.exact_c.attempted_iterations, 2);
        assert_eq!(observed.exact_c.completed_proposals, 1);
        assert_eq!(observed.exact_c.adaptive_plan_dispatches, 0);
        assert_eq!(observed.exact_c.gradient_plan_num_specs, Some(4));
        assert_eq!(observed.exact_c.gradient_row_count, Some(4));
    }

    #[test]
    fn exact_c_multi_iteration_evidence_rejects_uniform_only_effective_claim() {
        assert!(super::valid_exact_c_multi_iteration_evidence(
            1,
            0,
            true,
            true,
            false,
            1,
            0,
            Some(4),
            4,
        ));
        assert!(!super::valid_exact_c_multi_iteration_evidence(
            1,
            0,
            true,
            true,
            false,
            0,
            0,
            Some(1),
            4,
        ));
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let _run = begin_run();
        record_exact_c_selected(4, 4, 4, 0);
        record_exact_c_multi_iteration_committed(
            1,
            0,
            true,
            true,
            true,
            1,
            0,
            Some(4),
            4,
            "iteration_limit",
        );

        let observed = snapshot();
        assert!(observed.exact_c.attribution_conflict);
    }

    #[test]
    fn exact_c_aggregate_rejects_effective_free_completed_adaptive_plan() {
        let exact_c = ExactCObservations {
            committed: 1,
            iteration_count_outcomes: 1,
            attempted_iterations: 2,
            multi_iteration_evidence_outcomes: 1,
            multiplicative_weights_requested: Some(true),
            multiplicative_weights_plan_dispatched_outcomes: 1,
            completed_proposals: 2,
            adaptive_plan_dispatches: 1,
            gradient_plan_num_specs: Some(4),
            gradient_row_count: Some(4),
            ..ExactCObservations::default()
        };

        assert!(!super::valid_exact_c_multi_iteration_aggregates(&exact_c));
    }

    #[test]
    fn exact_c_aggregate_rejects_missing_per_outcome_completions() {
        let non_mw = ExactCObservations {
            committed: 2,
            iteration_count_outcomes: 2,
            attempted_iterations: 4,
            accepted_iterations: 1,
            multi_iteration_evidence_outcomes: 2,
            multiplicative_weights_requested: Some(false),
            completed_proposals: 1,
            gradient_plan_num_specs: Some(1),
            gradient_row_count: Some(4),
            ..ExactCObservations::default()
        };
        assert!(!super::valid_exact_c_multi_iteration_aggregates(&non_mw));

        let no_evidence = ExactCObservations {
            attempted_iterations: 1,
            accepted_iterations: 1,
            completed_proposals: 1,
            selected_iteration_limit_conflict: true,
            ..ExactCObservations::default()
        };
        assert!(!super::valid_exact_c_multi_iteration_aggregates(
            &no_evidence
        ));
    }

    #[test]
    fn exact_c_one_row_mw_aggregate_requires_active_outcome_completions() {
        let mut exact_c = ExactCObservations {
            committed: 2,
            iteration_count_outcomes: 2,
            attempted_iterations: 2,
            multi_iteration_evidence_outcomes: 2,
            multiplicative_weights_requested: Some(true),
            multiplicative_weights_plan_dispatched_outcomes: 1,
            gradient_plan_num_specs: Some(1),
            gradient_row_count: Some(1),
            ..ExactCObservations::default()
        };
        assert!(!super::valid_exact_c_multi_iteration_aggregates(&exact_c));

        exact_c.completed_proposals = 1;
        assert!(super::valid_exact_c_multi_iteration_aggregates(&exact_c));
    }

    #[test]
    fn exact_c_aggregate_respects_selected_iteration_limit() {
        let mut exact_c = ExactCObservations {
            selected_iteration_limit: Some(4),
            committed: 2,
            iteration_count_outcomes: 2,
            attempted_iterations: 9,
            multi_iteration_evidence_outcomes: 2,
            multiplicative_weights_requested: Some(false),
            completed_proposals: 7,
            gradient_plan_num_specs: Some(1),
            gradient_row_count: Some(4),
            ..ExactCObservations::default()
        };
        assert!(!super::valid_exact_c_multi_iteration_aggregates(&exact_c));

        exact_c.attempted_iterations = 8;
        exact_c.completed_proposals = 6;
        assert!(super::valid_exact_c_multi_iteration_aggregates(&exact_c));

        exact_c.attempted_iterations = 9;
        exact_c.completed_proposals = 7;
        exact_c.selected_iteration_limit_conflict = true;
        assert!(
            super::valid_exact_c_multi_iteration_aggregates(&exact_c),
            "the outer snapshot validator owns the selected-limit conflict"
        );

        let overflow = ExactCObservations {
            selected_iteration_limit: Some(2),
            committed: usize::MAX,
            iteration_count_outcomes: usize::MAX,
            multi_iteration_evidence_outcomes: usize::MAX,
            multiplicative_weights_requested: Some(false),
            gradient_row_count: Some(1),
            ..ExactCObservations::default()
        };
        assert!(!super::valid_exact_c_multi_iteration_aggregates(&overflow));
    }

    #[test]
    fn exact_c_multi_row_mw_aggregate_accepts_multiple_outcome_boundaries() {
        // (A, C, D, F): no effective event at C=P, then the exact lower
        // boundary for F=2 with respectively one and two completed finals.
        for (attempted, completed, adaptive, effective) in
            [(5, 3, 2, 0), (6, 4, 3, 2), (5, 4, 2, 2)]
        {
            let exact_c = ExactCObservations {
                committed: 3,
                iteration_count_outcomes: 3,
                attempted_iterations: attempted,
                multi_iteration_evidence_outcomes: 3,
                multiplicative_weights_requested: Some(true),
                multiplicative_weights_plan_dispatched_outcomes: 3,
                multiplicative_weights_effective_outcomes: effective,
                completed_proposals: completed,
                adaptive_plan_dispatches: adaptive,
                gradient_plan_num_specs: Some(4),
                gradient_row_count: Some(4),
                ..ExactCObservations::default()
            };
            assert!(
                super::valid_exact_c_multi_iteration_aggregates(&exact_c),
                "rejected feasible (A={attempted}, C={completed}, D={adaptive}, F={effective})"
            );
        }
    }

    #[test]
    fn compressed_layout_pairing_rejects_reordered_events_even_when_totals_match() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let _run = begin_run();
        for (evaluated, precertified) in [(1, 3), (3, 1)] {
            record_root_spec_prune_route(true);
            record_root_spec_prune_plan(4, evaluated, precertified);
            record_exact_c_selected(4, 4, evaluated, precertified);
            record_exact_c_refused_before_commit("missing_engine");
        }

        record_root_spec_prune_applied(4, 3, 1, false, true);
        record_root_spec_prune_applied(4, 1, 3, false, true);
        let observed = snapshot();
        assert_eq!(observed.exact_c.source_rows, 8);
        assert_eq!(observed.exact_c.evaluated_rows, 4);
        assert_eq!(observed.root_spec_prune.source_rows, 8);
        assert_eq!(observed.root_spec_prune.evaluated_rows, 4);
        assert!(observed.exact_c.attribution_conflict);
    }

    #[test]
    fn only_explicitly_scoped_folds_are_reported_as_evaluated_iterates() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let _run = begin_run();
        record_invprop_nonzero_output_seed_fold();
        let probe_only = snapshot();
        assert_eq!(probe_only.invprop.nonzero_output_seed_folds, 1);
        assert_eq!(probe_only.invprop.nonzero_evaluated_output_seed_folds, 0);

        {
            let _discarded = begin_invprop_evaluated_fold_scope();
            record_invprop_nonzero_output_seed_fold();
        }
        let discarded = snapshot();
        assert_eq!(discarded.invprop.nonzero_output_seed_folds, 2);
        assert_eq!(discarded.invprop.nonzero_evaluated_output_seed_folds, 0);

        record_invprop_gamma_step_attempted();
        record_invprop_gamma_step_applied();
        {
            let evaluated = begin_invprop_evaluated_fold_scope();
            record_invprop_nonzero_output_seed_fold();
            evaluated.commit();
        }
        let with_evaluated = snapshot();
        assert_eq!(with_evaluated.invprop.nonzero_output_seed_folds, 3);
        assert_eq!(
            with_evaluated.invprop.nonzero_evaluated_output_seed_folds,
            1
        );
        assert!(!with_evaluated.invprop.attribution_conflict);
    }

    #[test]
    fn snapshot_validation_rejects_incomplete_event_streams() {
        let missing_exact_outcome = Recorder::new();
        let _missing_exact_run = missing_exact_outcome.begin();
        missing_exact_outcome.record(|all| {
            all.exact_c.observed = true;
            all.exact_c.selections = 1;
            all.exact_c.selected_iteration_limit = Some(4);
        });
        assert!(
            missing_exact_outcome
                .snapshot()
                .exact_c
                .attribution_conflict
        );

        let missing_stop_reason = Recorder::new();
        let _missing_stop_run = missing_stop_reason.begin();
        missing_stop_reason.record(|all| {
            all.exact_c.observed = true;
            all.exact_c.selections = 1;
            all.exact_c.selected_iteration_limit = Some(4);
            all.exact_c.outcomes_observed = 1;
            all.exact_c.committed = 1;
        });
        assert!(missing_stop_reason.snapshot().exact_c.attribution_conflict);

        let incomplete_invprop = Recorder::new();
        let _incomplete_invprop_run = incomplete_invprop.begin();
        incomplete_invprop.record(|all| {
            all.invprop.observed = true;
            all.invprop.clause_rebind_attempts = 2;
            all.invprop.clause_rebind_accepted = 1;
            all.invprop.gamma_steps_applied = 1;
        });
        assert!(incomplete_invprop.snapshot().invprop.attribution_conflict);

        let evaluated_without_applied_gamma = Recorder::new();
        let _evaluated_without_gamma_run = evaluated_without_applied_gamma.begin();
        evaluated_without_applied_gamma.record(|all| {
            all.invprop.observed = true;
            all.invprop.nonzero_output_seed_folds = 1;
            all.invprop.nonzero_evaluated_output_seed_folds = 1;
        });
        assert!(
            evaluated_without_applied_gamma
                .snapshot()
                .invprop
                .attribution_conflict
        );

        let incomplete_fresh_clip = Recorder::new();
        let _incomplete_fresh_run = incomplete_fresh_clip.begin();
        incomplete_fresh_clip.record(|all| {
            all.fresh_domain_clip.observed = true;
            all.fresh_domain_clip.route_observations = 1;
            all.fresh_domain_clip.configured = Some(true);
            all.fresh_domain_clip.route_authorized = Some(true);
            all.fresh_domain_clip.attempts = 1;
        });
        assert!(
            incomplete_fresh_clip
                .snapshot()
                .fresh_domain_clip
                .attribution_conflict
        );
    }
}
