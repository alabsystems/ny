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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use serde::Serialize;

const EXECUTION_OBSERVATIONS_SCHEMA: &str = "ny_beta_crown_execution_observations_v8";
const MO_KFSB_OBSERVATIONS_SCHEMA: &str = "ny_beta_crown_mo_kfsb_observations_v1";
const MO_ALPHA_STEP_OBSERVATIONS_SCHEMA: &str = "ny_beta_crown_mo_alpha_step_observations_v1";
pub const MO_ALPHA_STEP_SCHEDULE_PERIOD: usize = 4;
pub const MO_ALPHA_STEP_ROUND_REMAINDER: usize = MO_ALPHA_STEP_SCHEDULE_PERIOD - 1;
const MO_KFSB_MAX_LEAVES_PER_PARENT: usize = 1 << 10;
const ROOT_JOINT_INTERM_ALPHA_OBSERVATIONS_SCHEMA: &str =
    "ny_beta_crown_root_joint_interm_alpha_observations_v1";
const ROOT_JOINT_DEFAULT_MIN_REMAINING_SECS: u64 = 240;

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
    /// Closed policy/admission/work evidence for the canonical multi-objective
    /// wave-kFSB selector. Certificate reuse is deliberately not represented:
    /// it remains an independent delivery-only policy.
    pub mo_kfsb: MoKfsbObservations,
    /// Closed admission/schedule/work evidence for relu-split delta-alpha
    /// optimization. Constants remain populated in the honest unobserved
    /// default so a consumer never has to infer the schedule from counters.
    pub mo_alpha_step: MoAlphaStepObservations,
    /// Closed route/configuration/work observations for the root joint
    /// intermediate-alpha family. Configuration alone is never work evidence.
    pub root_joint_interm_alpha: RootJointIntermAlphaObservations,
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
            mo_kfsb: MoKfsbObservations::default(),
            mo_alpha_step: MoAlphaStepObservations::default(),
            root_joint_interm_alpha: RootJointIntermAlphaObservations::default(),
            patches_materialization: PatchesMaterializationObservations::default(),
        }
    }
}

/// One closed invocation of the legacy multi-objective alpha/beta optimizer.
///
/// The recorder derives treatment validity from primitive route facts. In
/// particular, a configured alpha master cannot claim eligibility unless beta
/// SPSA, the explicit relu-split marker, real split constraints, and a
/// nonempty lower-alpha state were all present.
#[derive(Clone, Copy, Debug)]
pub struct MoAlphaStepInvocation {
    pub effective_alpha_step: bool,
    pub effective_beta_spsa: bool,
    pub relu_split_route: bool,
    pub has_split_constraints: bool,
    pub has_alpha_state: bool,
    pub alpha_coordinates: usize,
    pub eligible: bool,
    pub eligible_iterations: usize,
    pub fourth_iteration_opportunities: usize,
    pub alpha_rounds_scheduled: usize,
    pub beta_rounds_scheduled: usize,
    pub alpha_rounds_completed: usize,
    pub alpha_rounds_refused: usize,
    pub alpha_updates_applied: usize,
    pub perturbed_coordinates: usize,
}

/// Fixed-size aggregate for the every-fourth relu-split alpha schedule.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MoAlphaStepObservations {
    pub schema: &'static str,
    pub observed: bool,
    /// Run-common frozen policy. `None` is reserved for the unobserved default;
    /// mixed values within one solve poison attribution.
    pub effective_alpha_step: Option<bool>,
    pub effective_beta_spsa: Option<bool>,
    pub schedule_period: usize,
    /// Zero-based optimizer-iteration remainder: three means nominal rounds
    /// 4, 8, 12, ... are alpha-only when the master is effective.
    pub alpha_round_remainder: usize,
    pub attribution_conflict: bool,
    pub counter_overflow: bool,
    pub optimizer_invocations: usize,
    /// Structurally eligible invocations. This is deliberately independent of
    /// the alpha master so an exact-off arm retains its counterfactual witness.
    pub eligible_invocations: usize,
    /// Actual loop iterations entered across eligible invocations.
    pub eligible_iterations: usize,
    pub fourth_iteration_opportunities: usize,
    pub alpha_rounds_scheduled: usize,
    pub beta_rounds_scheduled: usize,
    pub alpha_rounds_completed: usize,
    pub alpha_rounds_refused: usize,
    pub alpha_updates_applied: usize,
    /// Sum of lower-alpha coordinates perturbed by scheduled alpha rounds.
    pub perturbed_coordinates: usize,
}

impl Default for MoAlphaStepObservations {
    fn default() -> Self {
        Self {
            schema: MO_ALPHA_STEP_OBSERVATIONS_SCHEMA,
            observed: false,
            effective_alpha_step: None,
            effective_beta_spsa: None,
            schedule_period: MO_ALPHA_STEP_SCHEDULE_PERIOD,
            alpha_round_remainder: MO_ALPHA_STEP_ROUND_REMAINDER,
            attribution_conflict: false,
            counter_overflow: false,
            optimizer_invocations: 0,
            eligible_invocations: 0,
            eligible_iterations: 0,
            fourth_iteration_opportunities: 0,
            alpha_rounds_scheduled: 0,
            beta_rounds_scheduled: 0,
            alpha_rounds_completed: 0,
            alpha_rounds_refused: 0,
            alpha_updates_applied: 0,
            perturbed_coordinates: 0,
        }
    }
}

/// Exact process-environment disposition of the canonical `NY_MO_KFSB`
/// master at one governed resolution boundary.
///
/// Absence preserves the lower typed/legacy policy. Exact enable and disable
/// override it in their respective directions. Rejection is distinct from an
/// exact disable so malformed input can never earn binding credit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MoKfsbCanonicalMaster {
    Absent,
    ExactEnable,
    ExactDisable,
    Rejected,
}

/// Exact configured/effective reduction policy for wave-kFSB ranking.
///
/// This receipt enum is intentionally separate from the verifier's config
/// enum so its terminal JSON spelling is permanently the closed lowercase
/// vocabulary consumed by search and transfer validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MoKfsbReduceOp {
    Min,
    Max,
    Mean,
}

impl From<crate::beta_crown::KfsbReduceOp> for MoKfsbReduceOp {
    fn from(value: crate::beta_crown::KfsbReduceOp) -> Self {
        match value {
            crate::beta_crown::KfsbReduceOp::Min => Self::Min,
            crate::beta_crown::KfsbReduceOp::Max => Self::Max,
            crate::beta_crown::KfsbReduceOp::Mean => Self::Mean,
        }
    }
}

/// One atomically closed opportunity at the historical wave-kFSB selector
/// seam. The recorder derives route and binding classifications from these
/// primitive facts; callers cannot assert credit directly.
#[derive(Clone, Copy, Debug)]
pub struct MoKfsbInvocation {
    pub canonical_master: MoKfsbCanonicalMaster,
    pub legacy_childsim: bool,
    pub configured_master: bool,
    pub configured_candidates: usize,
    pub effective_candidates: usize,
    pub configured_reduce_op: MoKfsbReduceOp,
    pub effective_reduce_op: MoKfsbReduceOp,
    pub heuristic_eligible: bool,
    pub typed_select: bool,
    pub lower_master: bool,
    pub effective_master: bool,
    pub bounded_shared_lane: bool,
    pub offered_domains: usize,
    pub selector_ran: bool,
    pub committed_parents: usize,
    /// A checked sum of committed leaves. `None` means the producer detected
    /// overflow before publication; zero is valid even with committed parents.
    pub committed_leaves: Option<usize>,
}

/// Fixed-size, validated execution evidence for multi-objective wave-kFSB.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MoKfsbObservations {
    pub schema: &'static str,
    pub observed: bool,
    pub attribution_conflict: bool,
    pub counter_overflow: bool,
    /// Run-common governed state. Every field is `None` before observation;
    /// variation within one solve poisons attribution rather than picking one.
    pub canonical_master: Option<MoKfsbCanonicalMaster>,
    pub legacy_childsim: Option<bool>,
    pub configured_master: Option<bool>,
    pub configured_candidates: Option<usize>,
    pub effective_candidates: Option<usize>,
    pub configured_reduce_op: Option<MoKfsbReduceOp>,
    pub effective_reduce_op: Option<MoKfsbReduceOp>,
    pub heuristic_eligible: Option<bool>,
    /// Invocation counts at the post-deadline branch-selection seam.
    pub branch_opportunities: usize,
    /// Checked sum of parent domains presented at that seam.
    pub offered_domains: usize,
    /// Equal to `branch_opportunities`: empty unstable waves return before the
    /// recorded selector seam.
    pub nonempty_domain_opportunities: usize,
    pub candidate_eligible_opportunities: usize,
    pub typed_select_opportunities: usize,
    pub lower_master_opportunities: usize,
    pub effective_master_opportunities: usize,
    /// Nonempty, unbounded, kFSB-heuristic opportunities with `K > 0`.
    pub route_eligible_opportunities: usize,
    /// Actual selector calls and domains passed to those calls. Every selector
    /// wave carries at least one domain.
    pub selector_waves: usize,
    pub selector_domains: usize,
    /// Returned parent-map entries and their feasible leaves. A parent entry
    /// with zero leaves is a valid complete infeasibility cover; depth is
    /// capped at ten, so each parent contributes at most 1024 feasible leaves.
    pub committed_waves: usize,
    pub committed_parents: usize,
    pub committed_leaves: usize,
    /// Exact-enable invocations that changed a lower-off policy and returned a
    /// nonempty committed parent map.
    pub force_on_binding_waves: usize,
    /// Exact-disable invocations that refused a lower-on eligible route.
    pub master_binding_refusals: usize,
}

impl Default for MoKfsbObservations {
    fn default() -> Self {
        Self {
            schema: MO_KFSB_OBSERVATIONS_SCHEMA,
            observed: false,
            attribution_conflict: false,
            counter_overflow: false,
            canonical_master: None,
            legacy_childsim: None,
            configured_master: None,
            configured_candidates: None,
            effective_candidates: None,
            configured_reduce_op: None,
            effective_reduce_op: None,
            heuristic_eligible: None,
            branch_opportunities: 0,
            offered_domains: 0,
            nonempty_domain_opportunities: 0,
            candidate_eligible_opportunities: 0,
            typed_select_opportunities: 0,
            lower_master_opportunities: 0,
            effective_master_opportunities: 0,
            route_eligible_opportunities: 0,
            selector_waves: 0,
            selector_domains: 0,
            committed_waves: 0,
            committed_parents: 0,
            committed_leaves: 0,
            force_on_binding_waves: 0,
            master_binding_refusals: 0,
        }
    }
}

/// Runtime route that consulted the root joint intermediate-alpha family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootJointIntermAlphaRoute {
    InputSplit,
    MultiObjective,
}

/// Closed disposition of one root joint intermediate-alpha invocation.
///
/// `Completed` means the bounded tightener was actually called. A completed
/// zero-tightening pass is honest live work, while every other variant proves
/// that no tightener attempt started.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootJointIntermAlphaOutcome {
    Completed,
    MasterDisabled,
    AdmissionFloor,
    DeadlineRouteDisabled,
    SliceTooSmall,
    NoTargets,
    CapabilityUnavailable,
    ExecutionRefused,
}

/// One independently closed root joint intermediate-alpha invocation.
///
/// Callers submit the effective values consumed by target selection and the
/// bounded tightener. `configured_lr` is converted to its exact IEEE-754 bit
/// pattern at the recording boundary; JSON and flight consumers never compare
/// unstable decimal formatting.
#[derive(Clone, Copy, Debug)]
pub struct RootJointIntermAlphaInvocation {
    pub route: RootJointIntermAlphaRoute,
    pub configured_master: bool,
    /// Explicit user/preset override, distinct from the route-context default.
    pub configured_max_dim_override: Option<u64>,
    pub configured_max_dim: u64,
    pub configured_secs: u64,
    pub configured_iters: u64,
    pub configured_lr: f32,
    pub configured_deadline_ascent: bool,
    pub configured_min_remaining_secs: u64,
    pub configured_max_sel: u64,
    pub configured_frozen_stop: bool,
    pub configured_pure_chain: bool,
    pub configured_bn_gpu_extract: bool,
    pub target_count: usize,
    pub available_budget: Duration,
    pub effective_slice: Duration,
    /// Top-level tightener dispatches. Current routes dispatch at most once.
    pub attempts: usize,
    pub tightened_count: usize,
    /// Input-split prerequisite shrinks produced inside this same exact-engine
    /// attempt. Multi-objective callers always report zero.
    pub outlier_tightened_count: usize,
    /// Actual target/backend work returned by the bounded tightener.
    pub target_attempts: usize,
    pub base_folds_completed: usize,
    /// Effective optimizer cap after policy/capability resolution.
    pub effective_iters: usize,
    pub ascent_iterations_attempted: usize,
    pub max_ascent_iterations_attempted: usize,
    pub optimizer_steps: usize,
    pub ascent_folds_completed: usize,
    /// Selection/preparation effects authenticated inside the tightener.
    pub max_selected_rows: usize,
    pub selection_cap_bound_targets: usize,
    pub frozen_stop_targets: usize,
    pub pure_chain_targets: usize,
    pub bn_extract_targets: usize,
    pub outcome: RootJointIntermAlphaOutcome,
}

/// Fixed-size, validated run aggregate for root joint intermediate alpha.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RootJointIntermAlphaObservations {
    pub schema: &'static str,
    pub observed: bool,
    pub attribution_conflict: bool,
    pub counter_overflow: bool,
    pub invocations: usize,
    pub input_split_invocations: usize,
    pub multi_objective_invocations: usize,
    /// Common effective configuration, or `None` before any invocation.
    pub configured_master: Option<bool>,
    /// Common explicit override. `None` honestly means contextual route values.
    pub configured_max_dim_override: Option<u64>,
    pub input_split_configured_max_dim: Option<u64>,
    pub multi_objective_configured_max_dim: Option<u64>,
    pub configured_secs: Option<u64>,
    pub configured_iters: Option<u64>,
    /// Exact IEEE-754 `f32::to_bits()` representation used by the tightener.
    pub configured_lr_bits: Option<u32>,
    pub configured_deadline_ascent: Option<bool>,
    pub configured_min_remaining_secs: Option<u64>,
    pub configured_max_sel: Option<u64>,
    pub configured_frozen_stop: Option<bool>,
    pub configured_pure_chain: Option<bool>,
    pub configured_bn_gpu_extract: Option<bool>,
    /// Maximum effective cap seen; per-invocation variation is valid.
    pub max_effective_iters: usize,
    /// Checked sums across independently closed invocations.
    pub target_count: usize,
    pub available_budget_ms: usize,
    pub effective_slice_ms: usize,
    /// Work invocations in which the configured slice was the actual bound.
    pub configured_slice_work_invocations: usize,
    /// Work invocations that reached the configured optimizer-iteration cap.
    pub configured_iters_binding_invocations: usize,
    /// Multi-objective work admitted by a lowered floor that the historical
    /// 240-second floor would have refused.
    pub lowered_min_remaining_work_invocations: usize,
    /// Multi-objective admission refusals caused by a raised floor in the
    /// window where the historical 240-second policy would have admitted.
    pub raised_min_remaining_binding_refusals: usize,
    pub work_started: usize,
    pub attempts: usize,
    pub completed: usize,
    pub tightened_count: usize,
    /// Input-split prerequisite shrinks published by the exact same qualified
    /// engine attempt. Kept separate so they cannot authenticate joint-target
    /// alpha work or its child knobs.
    pub outlier_tightened_count: usize,
    pub target_attempts: usize,
    pub base_folds_completed: usize,
    pub ascent_iterations_attempted: usize,
    pub max_ascent_iterations_attempted: usize,
    pub optimizer_steps: usize,
    pub ascent_folds_completed: usize,
    pub max_selected_rows: usize,
    pub selection_cap_bound_targets: usize,
    pub frozen_stop_targets: usize,
    pub pure_chain_targets: usize,
    pub bn_extract_targets: usize,
    pub master_disabled: usize,
    pub admission_floor_refusals: usize,
    pub deadline_route_refusals: usize,
    pub slice_refusals: usize,
    pub no_target_refusals: usize,
    pub capability_refusals: usize,
    pub execution_refusals: usize,
}

impl Default for RootJointIntermAlphaObservations {
    fn default() -> Self {
        Self {
            schema: ROOT_JOINT_INTERM_ALPHA_OBSERVATIONS_SCHEMA,
            observed: false,
            attribution_conflict: false,
            counter_overflow: false,
            invocations: 0,
            input_split_invocations: 0,
            multi_objective_invocations: 0,
            configured_master: None,
            configured_max_dim_override: None,
            input_split_configured_max_dim: None,
            multi_objective_configured_max_dim: None,
            configured_secs: None,
            configured_iters: None,
            configured_lr_bits: None,
            configured_deadline_ascent: None,
            configured_min_remaining_secs: None,
            configured_max_sel: None,
            configured_frozen_stop: None,
            configured_pure_chain: None,
            configured_bn_gpu_extract: None,
            max_effective_iters: 0,
            target_count: 0,
            available_budget_ms: 0,
            effective_slice_ms: 0,
            configured_slice_work_invocations: 0,
            configured_iters_binding_invocations: 0,
            lowered_min_remaining_work_invocations: 0,
            raised_min_remaining_binding_refusals: 0,
            work_started: 0,
            attempts: 0,
            completed: 0,
            tightened_count: 0,
            outlier_tightened_count: 0,
            target_attempts: 0,
            base_folds_completed: 0,
            ascent_iterations_attempted: 0,
            max_ascent_iterations_attempted: 0,
            optimizer_steps: 0,
            ascent_folds_completed: 0,
            max_selected_rows: 0,
            selection_cap_bound_targets: 0,
            frozen_stop_targets: 0,
            pure_chain_targets: 0,
            bn_extract_targets: 0,
            master_disabled: 0,
            admission_floor_refusals: 0,
            deadline_route_refusals: 0,
            slice_refusals: 0,
            no_target_refusals: 0,
            capability_refusals: 0,
            execution_refusals: 0,
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
    /// Lock-free hint for execution hot paths. The mutex remains authoritative;
    /// false means recording is certainly inert, while a stale true can only
    /// cause the recorder to take the lock and re-check state.
    recording_active: AtomicBool,
    state: Mutex<RecorderState>,
}

impl Recorder {
    const fn new() -> Self {
        Self {
            next_generation: AtomicU64::new(1),
            recording_active: AtomicBool::new(false),
            state: Mutex::new(RecorderState::Inactive),
        }
    }

    #[inline]
    fn is_active(&self) -> bool {
        self.recording_active.load(Ordering::Acquire)
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
                self.recording_active.store(false, Ordering::Release);
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
            self.recording_active.store(
                matches!(&*state, RecorderState::Active(_)),
                Ordering::Release,
            );
        } else {
            self.recording_active.store(false, Ordering::Release);
        }
        ExecutionTelemetryRun {
            recorder: self,
            generation,
        }
    }

    fn finish(&self, generation: u64) {
        let Ok(mut state) = self.state.lock() else {
            self.recording_active.store(false, Ordering::Release);
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
        self.recording_active.store(
            matches!(&*state, RecorderState::Active(_)),
            Ordering::Release,
        );
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
        if !self.is_active() {
            return;
        }
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

/// Minimum realizable overlap of two subsets of the same finite population.
///
/// This is the checked, overflow-free form of `left + right - total`: at most
/// `total - right` left members can sit outside `right`. Invalid subset sizes
/// return `None` so untrusted aggregate validation fails closed.
fn overlap_lower_bound(total: usize, left: usize, right: usize) -> Option<usize> {
    let outside_right = total.checked_sub(right)?;
    (left <= total).then(|| left.saturating_sub(outside_right))
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

    let mo_kfsb = &mut observations.mo_kfsb;
    let common_policy_complete = mo_kfsb.canonical_master.is_some()
        && mo_kfsb.legacy_childsim.is_some()
        && mo_kfsb.configured_master.is_some()
        && mo_kfsb.configured_candidates.is_some()
        && mo_kfsb.effective_candidates.is_some()
        && mo_kfsb.configured_reduce_op.is_some()
        && mo_kfsb.effective_reduce_op.is_some()
        && mo_kfsb.heuristic_eligible.is_some();
    let expected_candidate_eligible = mo_kfsb.effective_candidates.map(|candidates| {
        if candidates > 0 {
            mo_kfsb.branch_opportunities
        } else {
            0
        }
    });
    let expected_lower_master = mo_kfsb.configured_master.map(|configured| {
        if configured {
            mo_kfsb.branch_opportunities
        } else {
            mo_kfsb.typed_select_opportunities
        }
    });
    let expected_effective_master = match (
        mo_kfsb.canonical_master,
        mo_kfsb.legacy_childsim,
        expected_lower_master,
    ) {
        (Some(MoKfsbCanonicalMaster::ExactEnable), _, _) => Some(mo_kfsb.branch_opportunities),
        (Some(MoKfsbCanonicalMaster::ExactDisable | MoKfsbCanonicalMaster::Rejected), _, _) => {
            Some(0)
        }
        (Some(MoKfsbCanonicalMaster::Absent), Some(true), _) => Some(mo_kfsb.branch_opportunities),
        (Some(MoKfsbCanonicalMaster::Absent), Some(false), lower) => lower,
        _ => None,
    };
    let force_on_capacity = mo_kfsb
        .branch_opportunities
        .checked_sub(mo_kfsb.lower_master_opportunities);
    let committed_leaf_capacity = mo_kfsb
        .committed_parents
        .checked_mul(MO_KFSB_MAX_LEAVES_PER_PARENT);
    let selector_wave_lower_bound = overlap_lower_bound(
        mo_kfsb.branch_opportunities,
        mo_kfsb.route_eligible_opportunities,
        mo_kfsb.effective_master_opportunities,
    );
    let refusal_lower_bound = overlap_lower_bound(
        mo_kfsb.branch_opportunities,
        mo_kfsb.lower_master_opportunities,
        mo_kfsb.route_eligible_opportunities,
    );
    let force_on_lower_bound = mo_kfsb
        .committed_waves
        .saturating_sub(mo_kfsb.lower_master_opportunities);
    let selector_domain_capacity = mo_kfsb
        .branch_opportunities
        .checked_sub(mo_kfsb.selector_waves)
        .and_then(|unselected_waves| mo_kfsb.offered_domains.checked_sub(unselected_waves));
    let committed_parent_capacity = mo_kfsb
        .selector_waves
        .checked_sub(mo_kfsb.committed_waves)
        .and_then(|uncommitted_waves| mo_kfsb.selector_domains.checked_sub(uncommitted_waves));
    if mo_kfsb.schema != MO_KFSB_OBSERVATIONS_SCHEMA
        || mo_kfsb.observed != (mo_kfsb.branch_opportunities > 0)
        || common_policy_complete != mo_kfsb.observed
        || (mo_kfsb.branch_opportunities == 0
            && (mo_kfsb.canonical_master.is_some()
                || mo_kfsb.legacy_childsim.is_some()
                || mo_kfsb.configured_master.is_some()
                || mo_kfsb.configured_candidates.is_some()
                || mo_kfsb.effective_candidates.is_some()
                || mo_kfsb.configured_reduce_op.is_some()
                || mo_kfsb.effective_reduce_op.is_some()
                || mo_kfsb.heuristic_eligible.is_some()))
        || (mo_kfsb.canonical_master != Some(MoKfsbCanonicalMaster::Absent)
            && mo_kfsb.legacy_childsim == Some(true))
        || (mo_kfsb.observed
            && expected_candidate_eligible != Some(mo_kfsb.candidate_eligible_opportunities))
        || (mo_kfsb.observed && expected_lower_master != Some(mo_kfsb.lower_master_opportunities))
        || (mo_kfsb.observed
            && expected_effective_master != Some(mo_kfsb.effective_master_opportunities))
        || mo_kfsb.nonempty_domain_opportunities != mo_kfsb.branch_opportunities
        || mo_kfsb.candidate_eligible_opportunities > mo_kfsb.branch_opportunities
        || mo_kfsb.typed_select_opportunities > mo_kfsb.branch_opportunities
        || mo_kfsb.typed_select_opportunities > mo_kfsb.candidate_eligible_opportunities
        || (mo_kfsb.typed_select_opportunities > 0
            && mo_kfsb.effective_reduce_op != Some(MoKfsbReduceOp::Min))
        || mo_kfsb.lower_master_opportunities > mo_kfsb.branch_opportunities
        || mo_kfsb.effective_master_opportunities > mo_kfsb.branch_opportunities
        || mo_kfsb.route_eligible_opportunities > mo_kfsb.branch_opportunities
        || mo_kfsb.offered_domains < mo_kfsb.nonempty_domain_opportunities
        || (mo_kfsb.offered_domains > 0) != (mo_kfsb.nonempty_domain_opportunities > 0)
        || mo_kfsb.route_eligible_opportunities > mo_kfsb.nonempty_domain_opportunities
        || mo_kfsb.route_eligible_opportunities > mo_kfsb.candidate_eligible_opportunities
        || (mo_kfsb.heuristic_eligible == Some(false)
            && (mo_kfsb.typed_select_opportunities > 0
                || mo_kfsb.route_eligible_opportunities > 0
                || mo_kfsb.selector_waves > 0))
        || mo_kfsb.selector_waves > mo_kfsb.branch_opportunities
        || mo_kfsb.selector_waves > mo_kfsb.candidate_eligible_opportunities
        || mo_kfsb.selector_waves > mo_kfsb.effective_master_opportunities
        || mo_kfsb.selector_waves > mo_kfsb.route_eligible_opportunities
        || selector_wave_lower_bound.is_none_or(|lower_bound| mo_kfsb.selector_waves < lower_bound)
        || mo_kfsb.selector_domains > mo_kfsb.offered_domains
        || selector_domain_capacity.is_none_or(|capacity| mo_kfsb.selector_domains > capacity)
        || (mo_kfsb.selector_waves == mo_kfsb.branch_opportunities
            && mo_kfsb.selector_domains != mo_kfsb.offered_domains)
        || mo_kfsb.selector_waves > mo_kfsb.selector_domains
        || (mo_kfsb.selector_domains > 0 && mo_kfsb.selector_waves == 0)
        || (mo_kfsb.selector_domains > 0 && mo_kfsb.route_eligible_opportunities == 0)
        || mo_kfsb.committed_waves > mo_kfsb.selector_waves
        || mo_kfsb.committed_waves > mo_kfsb.route_eligible_opportunities
        || mo_kfsb.committed_waves > mo_kfsb.nonempty_domain_opportunities
        || mo_kfsb.committed_waves > mo_kfsb.committed_parents
        || mo_kfsb.committed_parents > mo_kfsb.selector_domains
        || committed_parent_capacity.is_none_or(|capacity| mo_kfsb.committed_parents > capacity)
        || (mo_kfsb.committed_waves > 0) != (mo_kfsb.committed_parents > 0)
        || (mo_kfsb.committed_leaves > 0 && mo_kfsb.committed_parents == 0)
        || committed_leaf_capacity.is_none_or(|capacity| mo_kfsb.committed_leaves > capacity)
        || mo_kfsb.force_on_binding_waves > mo_kfsb.committed_waves
        || force_on_capacity.is_none_or(|capacity| mo_kfsb.force_on_binding_waves > capacity)
        || (mo_kfsb.canonical_master == Some(MoKfsbCanonicalMaster::ExactEnable)
            && mo_kfsb.force_on_binding_waves < force_on_lower_bound)
        || (mo_kfsb.force_on_binding_waves > 0
            && mo_kfsb.canonical_master != Some(MoKfsbCanonicalMaster::ExactEnable))
        || mo_kfsb.master_binding_refusals > mo_kfsb.route_eligible_opportunities
        || mo_kfsb.master_binding_refusals > mo_kfsb.lower_master_opportunities
        || (mo_kfsb.canonical_master == Some(MoKfsbCanonicalMaster::ExactDisable)
            && refusal_lower_bound
                .is_none_or(|lower_bound| mo_kfsb.master_binding_refusals < lower_bound))
        || (mo_kfsb.master_binding_refusals > 0
            && (mo_kfsb.canonical_master != Some(MoKfsbCanonicalMaster::ExactDisable)
                || mo_kfsb
                    .effective_candidates
                    .is_none_or(|candidates| candidates == 0)))
        || (mo_kfsb.canonical_master == Some(MoKfsbCanonicalMaster::Rejected)
            && (mo_kfsb.force_on_binding_waves > 0 || mo_kfsb.master_binding_refusals > 0))
        || (mo_kfsb.effective_reduce_op == Some(MoKfsbReduceOp::Mean)
            && mo_kfsb.configured_reduce_op != Some(MoKfsbReduceOp::Mean))
        || mo_kfsb.counter_overflow
    {
        mo_kfsb.attribution_conflict = true;
    }

    let mo_alpha = &mut observations.mo_alpha_step;
    let alpha_outcomes = checked_sum([
        mo_alpha.alpha_rounds_completed,
        mo_alpha.alpha_rounds_refused,
    ]);
    let scheduled_rounds = checked_sum([
        mo_alpha.alpha_rounds_scheduled,
        mo_alpha.beta_rounds_scheduled,
    ]);
    let minimum_eligible_iterations = mo_alpha
        .fourth_iteration_opportunities
        .checked_mul(MO_ALPHA_STEP_SCHEDULE_PERIOD);
    let maximum_eligible_iterations = mo_alpha
        .eligible_invocations
        .checked_mul(MO_ALPHA_STEP_ROUND_REMAINDER)
        .and_then(|remainders| {
            minimum_eligible_iterations.and_then(|minimum| minimum.checked_add(remainders))
        });
    if mo_alpha.schema != MO_ALPHA_STEP_OBSERVATIONS_SCHEMA
        || mo_alpha.schedule_period != MO_ALPHA_STEP_SCHEDULE_PERIOD
        || mo_alpha.alpha_round_remainder != MO_ALPHA_STEP_ROUND_REMAINDER
        || mo_alpha.observed != (mo_alpha.optimizer_invocations > 0)
        || (mo_alpha.effective_alpha_step.is_some() != mo_alpha.observed)
        || (mo_alpha.effective_beta_spsa.is_some() != mo_alpha.observed)
        || mo_alpha.eligible_invocations > mo_alpha.optimizer_invocations
        || (mo_alpha.eligible_iterations > 0 && mo_alpha.eligible_invocations == 0)
        || mo_alpha.fourth_iteration_opportunities > mo_alpha.eligible_iterations
        || minimum_eligible_iterations.is_none_or(|minimum| mo_alpha.eligible_iterations < minimum)
        || maximum_eligible_iterations.is_none_or(|maximum| mo_alpha.eligible_iterations > maximum)
        || scheduled_rounds != Some(mo_alpha.eligible_iterations)
        || alpha_outcomes != Some(mo_alpha.alpha_rounds_scheduled)
        || mo_alpha.alpha_updates_applied != mo_alpha.alpha_rounds_completed
        || (mo_alpha.alpha_rounds_scheduled == 0) != (mo_alpha.perturbed_coordinates == 0)
        || mo_alpha.perturbed_coordinates < mo_alpha.alpha_rounds_scheduled
        || (mo_alpha.eligible_invocations == 1
            && mo_alpha.alpha_rounds_scheduled > 0
            && !mo_alpha
                .perturbed_coordinates
                .is_multiple_of(mo_alpha.alpha_rounds_scheduled))
        || (mo_alpha.effective_beta_spsa == Some(false)
            && (mo_alpha.eligible_invocations > 0
                || mo_alpha.eligible_iterations > 0
                || mo_alpha.fourth_iteration_opportunities > 0
                || mo_alpha.alpha_rounds_scheduled > 0
                || mo_alpha.beta_rounds_scheduled > 0))
        || (mo_alpha.effective_alpha_step == Some(true)
            && mo_alpha.alpha_rounds_scheduled != mo_alpha.fourth_iteration_opportunities)
        || (mo_alpha.effective_alpha_step == Some(false)
            && (mo_alpha.alpha_rounds_scheduled > 0
                || mo_alpha.alpha_rounds_completed > 0
                || mo_alpha.alpha_rounds_refused > 0
                || mo_alpha.alpha_updates_applied > 0
                || mo_alpha.perturbed_coordinates > 0))
        || mo_alpha.counter_overflow
    {
        mo_alpha.attribution_conflict = true;
    }

    let root_joint = &mut observations.root_joint_interm_alpha;
    let routes = checked_sum([
        root_joint.input_split_invocations,
        root_joint.multi_objective_invocations,
    ]);
    let classified = checked_sum([
        root_joint.completed,
        root_joint.master_disabled,
        root_joint.admission_floor_refusals,
        root_joint.deadline_route_refusals,
        root_joint.slice_refusals,
        root_joint.no_target_refusals,
        root_joint.capability_refusals,
        root_joint.execution_refusals,
    ]);
    let configuration_complete = root_joint.configured_master.is_some()
        && root_joint.configured_secs.is_some()
        && root_joint.configured_iters.is_some()
        && root_joint.configured_lr_bits.is_some()
        && root_joint.configured_deadline_ascent.is_some()
        && root_joint.configured_min_remaining_secs.is_some()
        && root_joint.configured_max_sel.is_some()
        && root_joint.configured_frozen_stop.is_some()
        && root_joint.configured_pure_chain.is_some()
        && root_joint.configured_bn_gpu_extract.is_some();
    let route_configuration_complete = (root_joint.input_split_invocations > 0)
        == root_joint.input_split_configured_max_dim.is_some()
        && (root_joint.multi_objective_invocations > 0)
            == root_joint.multi_objective_configured_max_dim.is_some();
    let maximum_ascent_attempts = root_joint
        .target_attempts
        .checked_mul(root_joint.max_effective_iters);
    if root_joint.observed != (root_joint.invocations > 0)
        || routes != Some(root_joint.invocations)
        || classified != Some(root_joint.invocations)
        || configuration_complete != (root_joint.invocations > 0)
        || !route_configuration_complete
        || root_joint.attempts != root_joint.completed
        || root_joint.attempts > root_joint.invocations
        || root_joint.work_started > root_joint.invocations
        || root_joint.configured_slice_work_invocations > root_joint.work_started
        || root_joint.configured_iters_binding_invocations > root_joint.work_started
        || root_joint.lowered_min_remaining_work_invocations > root_joint.work_started
        || root_joint.raised_min_remaining_binding_refusals > root_joint.admission_floor_refusals
        || (root_joint.work_started > 0) != (root_joint.target_attempts > 0)
        || root_joint.target_attempts > root_joint.target_count
        || root_joint.base_folds_completed > root_joint.target_attempts
        || root_joint.tightened_count > root_joint.base_folds_completed
        || (root_joint.outlier_tightened_count > 0
            && (root_joint.input_split_invocations == 0 || root_joint.attempts == 0))
        || root_joint.ascent_iterations_attempted > maximum_ascent_attempts.unwrap_or_default()
        || root_joint.max_ascent_iterations_attempted > root_joint.max_effective_iters
        || root_joint.max_ascent_iterations_attempted > root_joint.ascent_iterations_attempted
        || root_joint.optimizer_steps > root_joint.ascent_iterations_attempted
        || root_joint.ascent_folds_completed > root_joint.optimizer_steps
        || root_joint.selection_cap_bound_targets > root_joint.target_attempts
        || root_joint.frozen_stop_targets > root_joint.target_attempts
        || root_joint.pure_chain_targets > root_joint.target_attempts
        || root_joint.bn_extract_targets > root_joint.target_attempts
        || u64::try_from(root_joint.max_selected_rows)
            .ok()
            .is_none_or(|selected| {
                root_joint
                    .configured_max_sel
                    .is_none_or(|configured| selected > configured)
            })
        || (root_joint.selection_cap_bound_targets > 0
            && u64::try_from(root_joint.max_selected_rows).ok() != root_joint.configured_max_sel)
        || (root_joint.configured_max_sel == Some(0)
            && (root_joint.max_selected_rows > 0 || root_joint.selection_cap_bound_targets > 0))
        || (root_joint.configured_frozen_stop == Some(false) && root_joint.frozen_stop_targets > 0)
        || (root_joint.configured_pure_chain == Some(false) && root_joint.pure_chain_targets > 0)
        || (root_joint.configured_bn_gpu_extract == Some(false)
            && root_joint.bn_extract_targets > 0)
        || root_joint
            .configured_max_dim_override
            .is_some_and(|override_value| {
                root_joint
                    .input_split_configured_max_dim
                    .is_some_and(|value| value != override_value)
                    || root_joint
                        .multi_objective_configured_max_dim
                        .is_some_and(|value| value != override_value)
            })
        || (root_joint.configured_max_dim_override.is_none()
            && (root_joint
                .input_split_configured_max_dim
                .is_some_and(|value| value != 32_768)
                || root_joint
                    .multi_objective_configured_max_dim
                    .is_some_and(|value| {
                        value
                            != if root_joint.configured_master == Some(true)
                                && root_joint.configured_deadline_ascent == Some(true)
                            {
                                32_768
                            } else {
                                2_048
                            }
                    })))
        || (root_joint.lowered_min_remaining_work_invocations > 0
            && (root_joint.multi_objective_invocations == 0
                || root_joint
                    .configured_min_remaining_secs
                    .is_none_or(|floor| floor >= ROOT_JOINT_DEFAULT_MIN_REMAINING_SECS)))
        || (root_joint.raised_min_remaining_binding_refusals > 0
            && (root_joint.multi_objective_invocations == 0
                || root_joint
                    .configured_min_remaining_secs
                    .is_none_or(|floor| floor <= ROOT_JOINT_DEFAULT_MIN_REMAINING_SECS)))
        || (root_joint.max_effective_iters == 0
            && (root_joint.ascent_iterations_attempted > 0
                || root_joint.max_ascent_iterations_attempted > 0
                || root_joint.optimizer_steps > 0
                || root_joint.ascent_folds_completed > 0))
        || root_joint.tightened_count > root_joint.target_count
        || root_joint.effective_slice_ms > root_joint.available_budget_ms
        || (root_joint.attempts > 0
            && (root_joint.configured_master != Some(true)
                || root_joint.target_count == 0
                || root_joint.effective_slice_ms == 0))
        || (root_joint.configured_master == Some(false)
            && (root_joint.target_attempts > 0
                || root_joint.tightened_count > 0
                || root_joint.outlier_tightened_count > 0))
        || root_joint.counter_overflow
    {
        root_joint.attribution_conflict = true;
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

/// Lock-free hint for verifier paths that must retain primitive outcome data
/// before a later, non-critical recording seam. False guarantees that no
/// observation is attributable and lets callers skip all receipt construction.
#[inline]
pub(crate) fn observation_recording_active() -> bool {
    RECORDER.is_active()
}

/// Record one closed legacy multi-objective optimizer invocation.
///
/// Eligibility is recomputed from primitive route facts. A caller cannot earn
/// an alpha opportunity merely by asserting `eligible=true`, and an exact-off
/// alpha master retains eligible/fourth-iteration counterfactual evidence while
/// scheduling every reached iteration as beta work.
pub fn record_mo_alpha_step_invocation(invocation: MoAlphaStepInvocation) {
    RECORDER.record(|all| {
        let observed = &mut all.mo_alpha_step;
        observed.observed = true;
        observed.counter_overflow |= increment_counter(&mut observed.optimizer_invocations);
        observed.attribution_conflict |= observe_common(
            &mut observed.effective_alpha_step,
            invocation.effective_alpha_step,
        );
        observed.attribution_conflict |= observe_common(
            &mut observed.effective_beta_spsa,
            invocation.effective_beta_spsa,
        );

        let expected_eligible = invocation.effective_beta_spsa
            && invocation.relu_split_route
            && invocation.has_split_constraints
            && invocation.has_alpha_state
            && invocation.alpha_coordinates > 0;
        if invocation.eligible {
            observed.counter_overflow |= increment_counter(&mut observed.eligible_invocations);
        }
        for (counter, value) in [
            (
                &mut observed.eligible_iterations,
                invocation.eligible_iterations,
            ),
            (
                &mut observed.fourth_iteration_opportunities,
                invocation.fourth_iteration_opportunities,
            ),
            (
                &mut observed.alpha_rounds_scheduled,
                invocation.alpha_rounds_scheduled,
            ),
            (
                &mut observed.beta_rounds_scheduled,
                invocation.beta_rounds_scheduled,
            ),
            (
                &mut observed.alpha_rounds_completed,
                invocation.alpha_rounds_completed,
            ),
            (
                &mut observed.alpha_rounds_refused,
                invocation.alpha_rounds_refused,
            ),
            (
                &mut observed.alpha_updates_applied,
                invocation.alpha_updates_applied,
            ),
            (
                &mut observed.perturbed_coordinates,
                invocation.perturbed_coordinates,
            ),
        ] {
            observed.counter_overflow |= add_counter(counter, value);
        }

        let alpha_outcomes = invocation
            .alpha_rounds_completed
            .checked_add(invocation.alpha_rounds_refused);
        let scheduled_rounds = invocation
            .alpha_rounds_scheduled
            .checked_add(invocation.beta_rounds_scheduled);
        let expected_perturbed_coordinates = invocation
            .alpha_coordinates
            .checked_mul(invocation.alpha_rounds_scheduled);
        if invocation.eligible != expected_eligible
            || (!invocation.eligible
                && (invocation.eligible_iterations > 0
                    || invocation.fourth_iteration_opportunities > 0
                    || invocation.alpha_rounds_scheduled > 0
                    || invocation.beta_rounds_scheduled > 0
                    || invocation.alpha_rounds_completed > 0
                    || invocation.alpha_rounds_refused > 0
                    || invocation.alpha_updates_applied > 0
                    || invocation.perturbed_coordinates > 0))
            || invocation.fourth_iteration_opportunities
                != invocation.eligible_iterations / MO_ALPHA_STEP_SCHEDULE_PERIOD
            || scheduled_rounds != Some(invocation.eligible_iterations)
            || alpha_outcomes != Some(invocation.alpha_rounds_scheduled)
            || invocation.alpha_updates_applied != invocation.alpha_rounds_completed
            || expected_perturbed_coordinates != Some(invocation.perturbed_coordinates)
            || (invocation.alpha_rounds_scheduled == 0) != (invocation.perturbed_coordinates == 0)
            || (invocation.effective_alpha_step
                && invocation.alpha_rounds_scheduled != invocation.fourth_iteration_opportunities)
            || (!invocation.effective_alpha_step && invocation.alpha_rounds_scheduled > 0)
        {
            observed.attribution_conflict = true;
        }
    });
}

/// Record one closed multi-objective wave-kFSB opportunity.
///
/// Binding classifications are derived here from primitive governed state and
/// returned work. In particular, rejected canonical input cannot be mistaken
/// for an exact disable, and a force-on claim requires a nonempty committed
/// parent map (zero feasible leaves remains a valid commit).
pub fn record_mo_kfsb_invocation(invocation: MoKfsbInvocation) {
    RECORDER.record(|all| {
        let observations = &mut all.mo_kfsb;
        observations.observed = true;
        observations.counter_overflow |= increment_counter(&mut observations.branch_opportunities);
        observations.counter_overflow |= add_counter(
            &mut observations.offered_domains,
            invocation.offered_domains,
        );

        observations.attribution_conflict |= observe_common(
            &mut observations.canonical_master,
            invocation.canonical_master,
        );
        observations.attribution_conflict |= observe_common(
            &mut observations.legacy_childsim,
            invocation.legacy_childsim,
        );
        observations.attribution_conflict |= observe_common(
            &mut observations.configured_master,
            invocation.configured_master,
        );
        observations.attribution_conflict |= observe_common(
            &mut observations.configured_candidates,
            invocation.configured_candidates,
        );
        observations.attribution_conflict |= observe_common(
            &mut observations.effective_candidates,
            invocation.effective_candidates,
        );
        observations.attribution_conflict |= observe_common(
            &mut observations.configured_reduce_op,
            invocation.configured_reduce_op,
        );
        observations.attribution_conflict |= observe_common(
            &mut observations.effective_reduce_op,
            invocation.effective_reduce_op,
        );
        observations.attribution_conflict |= observe_common(
            &mut observations.heuristic_eligible,
            invocation.heuristic_eligible,
        );

        if invocation.offered_domains > 0 {
            observations.counter_overflow |=
                increment_counter(&mut observations.nonempty_domain_opportunities);
        }
        if invocation.effective_candidates > 0 {
            observations.counter_overflow |=
                increment_counter(&mut observations.candidate_eligible_opportunities);
        }
        if invocation.typed_select {
            observations.counter_overflow |=
                increment_counter(&mut observations.typed_select_opportunities);
        }
        if invocation.lower_master {
            observations.counter_overflow |=
                increment_counter(&mut observations.lower_master_opportunities);
        }
        if invocation.effective_master {
            observations.counter_overflow |=
                increment_counter(&mut observations.effective_master_opportunities);
        }

        let route_eligible = !invocation.bounded_shared_lane
            && invocation.heuristic_eligible
            && invocation.effective_candidates > 0
            && invocation.offered_domains > 0;
        if route_eligible {
            observations.counter_overflow |=
                increment_counter(&mut observations.route_eligible_opportunities);
        }

        if invocation.selector_ran {
            observations.counter_overflow |= increment_counter(&mut observations.selector_waves);
            observations.counter_overflow |= add_counter(
                &mut observations.selector_domains,
                invocation.offered_domains,
            );
        }

        observations.counter_overflow |= add_counter(
            &mut observations.committed_parents,
            invocation.committed_parents,
        );
        match invocation.committed_leaves {
            Some(committed_leaves) => {
                observations.counter_overflow |=
                    add_counter(&mut observations.committed_leaves, committed_leaves);
            }
            None => observations.counter_overflow = true,
        }
        if invocation.committed_parents > 0 {
            observations.counter_overflow |= increment_counter(&mut observations.committed_waves);
        }

        let force_on_binding = invocation.canonical_master == MoKfsbCanonicalMaster::ExactEnable
            && !invocation.lower_master
            && invocation.selector_ran
            && invocation.committed_parents > 0;
        if force_on_binding {
            observations.counter_overflow |=
                increment_counter(&mut observations.force_on_binding_waves);
        }
        let master_binding_refusal = invocation.canonical_master
            == MoKfsbCanonicalMaster::ExactDisable
            && invocation.lower_master
            && route_eligible;
        if master_binding_refusal {
            observations.counter_overflow |=
                increment_counter(&mut observations.master_binding_refusals);
        }

        let expected_lower_master = invocation.configured_master || invocation.typed_select;
        let expected_effective_master = match invocation.canonical_master {
            MoKfsbCanonicalMaster::Absent => expected_lower_master || invocation.legacy_childsim,
            MoKfsbCanonicalMaster::ExactEnable => true,
            MoKfsbCanonicalMaster::ExactDisable | MoKfsbCanonicalMaster::Rejected => false,
        };
        let expected_selector = route_eligible && invocation.effective_master;
        let committed_leaf_capacity = invocation
            .committed_parents
            .checked_mul(MO_KFSB_MAX_LEAVES_PER_PARENT);
        if invocation.offered_domains == 0
            || (invocation.effective_reduce_op == MoKfsbReduceOp::Mean
                && invocation.configured_reduce_op != MoKfsbReduceOp::Mean)
            || (invocation.typed_select && invocation.effective_reduce_op != MoKfsbReduceOp::Min)
            || (invocation.typed_select
                && (!invocation.heuristic_eligible || invocation.effective_candidates == 0))
            || invocation.lower_master != expected_lower_master
            || invocation.effective_master != expected_effective_master
            || (invocation.canonical_master != MoKfsbCanonicalMaster::Absent
                && invocation.legacy_childsim)
            || invocation.selector_ran != expected_selector
            || (!invocation.selector_ran
                && (invocation.committed_parents > 0
                    || invocation.committed_leaves.is_some_and(|leaves| leaves > 0)))
            || invocation.committed_parents > invocation.offered_domains
            || (invocation.committed_parents > 0 && !route_eligible)
            || (invocation.committed_parents == 0
                && invocation.committed_leaves.is_some_and(|leaves| leaves > 0))
            || invocation.committed_leaves.is_none_or(|leaves| {
                committed_leaf_capacity.is_none_or(|capacity| leaves > capacity)
            })
        {
            observations.attribution_conflict = true;
        }
    });
}

fn observe_common<T: Copy + Eq>(slot: &mut Option<T>, value: T) -> bool {
    match *slot {
        None => {
            *slot = Some(value);
            false
        }
        Some(existing) => existing != value,
    }
}

fn duration_millis(duration: Duration) -> (usize, bool) {
    match usize::try_from(duration.as_millis()) {
        Ok(millis) => (millis, false),
        Err(_) => (usize::MAX, true),
    }
}

/// Record one atomically closed root joint intermediate-alpha invocation.
///
/// This API intentionally has no separate "configured" or "started" calls:
/// an early return must still publish one refusal, and a missing terminal call
/// cannot leave a configuration-only fragment that downstream code mistakes
/// for work. Current callers dispatch at most one tightener per invocation.
pub fn record_root_joint_interm_alpha_invocation(invocation: RootJointIntermAlphaInvocation) {
    RECORDER.record(|all| {
        let root = &mut all.root_joint_interm_alpha;
        let first_invocation = root.invocations == 0;
        root.observed = true;
        root.counter_overflow |= increment_counter(&mut root.invocations);
        match invocation.route {
            RootJointIntermAlphaRoute::InputSplit => {
                root.counter_overflow |= increment_counter(&mut root.input_split_invocations);
            }
            RootJointIntermAlphaRoute::MultiObjective => {
                root.counter_overflow |= increment_counter(&mut root.multi_objective_invocations);
            }
        }

        root.attribution_conflict |=
            observe_common(&mut root.configured_master, invocation.configured_master);
        if first_invocation {
            root.configured_max_dim_override = invocation.configured_max_dim_override;
        } else if root.configured_max_dim_override != invocation.configured_max_dim_override {
            root.attribution_conflict = true;
        }
        root.attribution_conflict |= match invocation.route {
            RootJointIntermAlphaRoute::InputSplit => observe_common(
                &mut root.input_split_configured_max_dim,
                invocation.configured_max_dim,
            ),
            RootJointIntermAlphaRoute::MultiObjective => observe_common(
                &mut root.multi_objective_configured_max_dim,
                invocation.configured_max_dim,
            ),
        };
        root.attribution_conflict |=
            observe_common(&mut root.configured_secs, invocation.configured_secs);
        root.attribution_conflict |=
            observe_common(&mut root.configured_iters, invocation.configured_iters);
        root.attribution_conflict |= observe_common(
            &mut root.configured_lr_bits,
            invocation.configured_lr.to_bits(),
        );
        root.attribution_conflict |= observe_common(
            &mut root.configured_deadline_ascent,
            invocation.configured_deadline_ascent,
        );
        root.attribution_conflict |= observe_common(
            &mut root.configured_min_remaining_secs,
            invocation.configured_min_remaining_secs,
        );
        root.attribution_conflict |=
            observe_common(&mut root.configured_max_sel, invocation.configured_max_sel);
        root.attribution_conflict |= observe_common(
            &mut root.configured_frozen_stop,
            invocation.configured_frozen_stop,
        );
        root.attribution_conflict |= observe_common(
            &mut root.configured_pure_chain,
            invocation.configured_pure_chain,
        );
        root.attribution_conflict |= observe_common(
            &mut root.configured_bn_gpu_extract,
            invocation.configured_bn_gpu_extract,
        );
        root.max_effective_iters = root.max_effective_iters.max(invocation.effective_iters);

        let (available_budget_ms, available_overflow) =
            duration_millis(invocation.available_budget);
        let (effective_slice_ms, slice_overflow) = duration_millis(invocation.effective_slice);
        root.counter_overflow |= available_overflow | slice_overflow;
        root.counter_overflow |= add_counter(&mut root.target_count, invocation.target_count);
        root.counter_overflow |= add_counter(&mut root.available_budget_ms, available_budget_ms);
        root.counter_overflow |= add_counter(&mut root.effective_slice_ms, effective_slice_ms);
        root.counter_overflow |= add_counter(&mut root.attempts, invocation.attempts);
        root.counter_overflow |= add_counter(&mut root.tightened_count, invocation.tightened_count);
        root.counter_overflow |= add_counter(
            &mut root.outlier_tightened_count,
            invocation.outlier_tightened_count,
        );
        root.counter_overflow |= add_counter(&mut root.target_attempts, invocation.target_attempts);
        root.counter_overflow |= add_counter(
            &mut root.base_folds_completed,
            invocation.base_folds_completed,
        );
        root.counter_overflow |= add_counter(
            &mut root.ascent_iterations_attempted,
            invocation.ascent_iterations_attempted,
        );
        root.max_ascent_iterations_attempted = root
            .max_ascent_iterations_attempted
            .max(invocation.max_ascent_iterations_attempted);
        root.counter_overflow |= add_counter(&mut root.optimizer_steps, invocation.optimizer_steps);
        root.counter_overflow |= add_counter(
            &mut root.ascent_folds_completed,
            invocation.ascent_folds_completed,
        );
        root.max_selected_rows = root.max_selected_rows.max(invocation.max_selected_rows);
        root.counter_overflow |= add_counter(
            &mut root.selection_cap_bound_targets,
            invocation.selection_cap_bound_targets,
        );
        root.counter_overflow |= add_counter(
            &mut root.frozen_stop_targets,
            invocation.frozen_stop_targets,
        );
        root.counter_overflow |=
            add_counter(&mut root.pure_chain_targets, invocation.pure_chain_targets);
        root.counter_overflow |=
            add_counter(&mut root.bn_extract_targets, invocation.bn_extract_targets);
        if invocation.target_attempts > 0 {
            root.counter_overflow |= increment_counter(&mut root.work_started);
        }
        let backend_work = invocation.target_attempts > 0 && invocation.base_folds_completed > 0;
        if backend_work
            && invocation.effective_slice == Duration::from_secs(invocation.configured_secs)
        {
            root.counter_overflow |= increment_counter(&mut root.configured_slice_work_invocations);
        }
        if backend_work
            && invocation.configured_iters > 0
            && invocation.ascent_iterations_attempted > 0
            && u64::try_from(invocation.effective_iters).ok() == Some(invocation.configured_iters)
            && u64::try_from(invocation.max_ascent_iterations_attempted).ok()
                == Some(invocation.configured_iters)
        {
            root.counter_overflow |=
                increment_counter(&mut root.configured_iters_binding_invocations);
        }
        if backend_work
            && invocation.route == RootJointIntermAlphaRoute::MultiObjective
            && invocation.configured_min_remaining_secs < ROOT_JOINT_DEFAULT_MIN_REMAINING_SECS
            && invocation.available_budget
                >= Duration::from_secs(invocation.configured_min_remaining_secs)
            && invocation.available_budget
                < Duration::from_secs(ROOT_JOINT_DEFAULT_MIN_REMAINING_SECS)
        {
            root.counter_overflow |=
                increment_counter(&mut root.lowered_min_remaining_work_invocations);
        }
        if invocation.route == RootJointIntermAlphaRoute::MultiObjective
            && invocation.outcome == RootJointIntermAlphaOutcome::AdmissionFloor
            && invocation.configured_min_remaining_secs > ROOT_JOINT_DEFAULT_MIN_REMAINING_SECS
            && invocation.available_budget
                >= Duration::from_secs(ROOT_JOINT_DEFAULT_MIN_REMAINING_SECS)
            && invocation.available_budget
                < Duration::from_secs(invocation.configured_min_remaining_secs)
        {
            root.counter_overflow |=
                increment_counter(&mut root.raised_min_remaining_binding_refusals);
        }

        match invocation.outcome {
            RootJointIntermAlphaOutcome::Completed => {
                root.counter_overflow |= increment_counter(&mut root.completed);
            }
            RootJointIntermAlphaOutcome::MasterDisabled => {
                root.counter_overflow |= increment_counter(&mut root.master_disabled);
            }
            RootJointIntermAlphaOutcome::AdmissionFloor => {
                root.counter_overflow |= increment_counter(&mut root.admission_floor_refusals);
            }
            RootJointIntermAlphaOutcome::DeadlineRouteDisabled => {
                root.counter_overflow |= increment_counter(&mut root.deadline_route_refusals);
            }
            RootJointIntermAlphaOutcome::SliceTooSmall => {
                root.counter_overflow |= increment_counter(&mut root.slice_refusals);
            }
            RootJointIntermAlphaOutcome::NoTargets => {
                root.counter_overflow |= increment_counter(&mut root.no_target_refusals);
            }
            RootJointIntermAlphaOutcome::CapabilityUnavailable => {
                root.counter_overflow |= increment_counter(&mut root.capability_refusals);
            }
            RootJointIntermAlphaOutcome::ExecutionRefused => {
                root.counter_overflow |= increment_counter(&mut root.execution_refusals);
            }
        }

        let completed = invocation.outcome == RootJointIntermAlphaOutcome::Completed;
        let invocation_valid = invocation.configured_max_dim > 0
            && invocation.configured_lr.is_finite()
            && invocation
                .configured_max_dim_override
                .is_none_or(|value| value > 0 && value == invocation.configured_max_dim)
            && (invocation.configured_max_dim_override.is_some()
                || invocation.configured_max_dim
                    == match invocation.route {
                        RootJointIntermAlphaRoute::InputSplit => 32_768,
                        RootJointIntermAlphaRoute::MultiObjective
                            if invocation.configured_master
                                && invocation.configured_deadline_ascent =>
                        {
                            32_768
                        }
                        RootJointIntermAlphaRoute::MultiObjective => 2_048,
                    })
            && invocation.attempts <= 1
            && invocation.tightened_count <= invocation.target_count
            && invocation.target_attempts <= invocation.target_count
            && invocation.base_folds_completed <= invocation.target_attempts
            && invocation.tightened_count <= invocation.base_folds_completed
            && invocation.max_ascent_iterations_attempted <= invocation.effective_iters
            && invocation.max_ascent_iterations_attempted <= invocation.ascent_iterations_attempted
            && invocation
                .target_attempts
                .checked_mul(invocation.effective_iters)
                .is_some_and(|maximum| invocation.ascent_iterations_attempted <= maximum)
            && invocation.optimizer_steps <= invocation.ascent_iterations_attempted
            && invocation.ascent_folds_completed <= invocation.optimizer_steps
            && invocation.selection_cap_bound_targets <= invocation.target_attempts
            && invocation.frozen_stop_targets <= invocation.target_attempts
            && invocation.pure_chain_targets <= invocation.target_attempts
            && invocation.bn_extract_targets <= invocation.target_attempts
            && u64::try_from(invocation.max_selected_rows)
                .is_ok_and(|selected| selected <= invocation.configured_max_sel)
            && (invocation.selection_cap_bound_targets == 0
                || u64::try_from(invocation.max_selected_rows).ok()
                    == Some(invocation.configured_max_sel))
            && (invocation.configured_max_sel > 0
                || (invocation.max_selected_rows == 0
                    && invocation.selection_cap_bound_targets == 0))
            && (invocation.configured_frozen_stop || invocation.frozen_stop_targets == 0)
            && (invocation.configured_pure_chain || invocation.pure_chain_targets == 0)
            && (invocation.configured_bn_gpu_extract || invocation.bn_extract_targets == 0)
            && (invocation.effective_iters > 0
                || (invocation.ascent_iterations_attempted == 0
                    && invocation.max_ascent_iterations_attempted == 0
                    && invocation.optimizer_steps == 0
                    && invocation.ascent_folds_completed == 0))
            && invocation.effective_slice <= invocation.available_budget
            && completed == (invocation.attempts == 1)
            && (completed
                || (invocation.attempts == 0
                    && invocation.tightened_count == 0
                    && invocation.outlier_tightened_count == 0))
            && (invocation.outlier_tightened_count == 0
                || (invocation.route == RootJointIntermAlphaRoute::InputSplit && completed))
            && (!completed
                || (invocation.configured_master
                    && invocation.target_count > 0
                    && !invocation.effective_slice.is_zero()))
            && (invocation.outcome != RootJointIntermAlphaOutcome::MasterDisabled
                || (!invocation.configured_master
                    && invocation.target_count == 0
                    && invocation.attempts == 0))
            && (invocation.outcome == RootJointIntermAlphaOutcome::MasterDisabled
                || invocation.configured_master)
            && (invocation.outcome != RootJointIntermAlphaOutcome::NoTargets
                || invocation.target_count == 0);
        if !invocation_valid {
            root.attribution_conflict = true;
        }
    });
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
    use std::time::Duration;

    use super::{
        begin_invprop_evaluated_fold_scope, begin_run, observation_recording_active,
        record_exact_c_compact_commit, record_exact_c_multi_iteration_committed,
        record_exact_c_refused_before_commit, record_exact_c_selected,
        record_fresh_domain_clip_outcome, record_fresh_domain_clip_route,
        record_invprop_alpha_initialization, record_invprop_clause_rebind_accepted,
        record_invprop_clause_rebind_attempt, record_invprop_gamma_step_applied,
        record_invprop_gamma_step_attempted, record_invprop_nonzero_output_seed_fold,
        record_mo_alpha_step_invocation, record_mo_kfsb_invocation,
        record_patches_materialization_attempt, record_patches_materialization_refusal,
        record_patches_materialization_success, record_root_joint_interm_alpha_invocation,
        record_root_spec_prune_applied, record_root_spec_prune_plan, record_root_spec_prune_route,
        snapshot, update_exact_c_selection, ExactCObservations, FreshDomainClipDisposition,
        MoAlphaStepInvocation, MoKfsbCanonicalMaster, MoKfsbInvocation, MoKfsbReduceOp,
        PatchesCoefficientErrorDisposition, PatchesMaterializationGeometry,
        PatchesMaterializationMemoryReceipt, PatchesMaterializationPurpose,
        PatchesMaterializationRefusal, Recorder, RootJointIntermAlphaInvocation,
        RootJointIntermAlphaOutcome, RootJointIntermAlphaRoute,
    };

    fn root_joint_invocation(
        outcome: RootJointIntermAlphaOutcome,
    ) -> RootJointIntermAlphaInvocation {
        let completed = outcome == RootJointIntermAlphaOutcome::Completed;
        RootJointIntermAlphaInvocation {
            route: RootJointIntermAlphaRoute::InputSplit,
            configured_master: outcome != RootJointIntermAlphaOutcome::MasterDisabled,
            configured_max_dim_override: Some(32_768),
            configured_max_dim: 32_768,
            configured_secs: 30,
            configured_iters: 100,
            configured_lr: 0.1,
            configured_deadline_ascent: true,
            configured_min_remaining_secs: 240,
            configured_max_sel: 32,
            configured_frozen_stop: true,
            configured_pure_chain: true,
            configured_bn_gpu_extract: true,
            target_count: usize::from(completed) * 3,
            available_budget: Duration::from_mins(1),
            effective_slice: if completed {
                Duration::from_secs(30)
            } else {
                Duration::ZERO
            },
            attempts: usize::from(completed),
            tightened_count: usize::from(completed),
            outlier_tightened_count: usize::from(completed),
            target_attempts: usize::from(completed) * 3,
            base_folds_completed: usize::from(completed) * 3,
            effective_iters: usize::from(completed) * 100,
            ascent_iterations_attempted: usize::from(completed) * 100,
            max_ascent_iterations_attempted: usize::from(completed) * 100,
            optimizer_steps: usize::from(completed) * 100,
            ascent_folds_completed: usize::from(completed) * 100,
            max_selected_rows: usize::from(completed) * 32,
            selection_cap_bound_targets: usize::from(completed),
            frozen_stop_targets: usize::from(completed),
            pure_chain_targets: usize::from(completed),
            bn_extract_targets: usize::from(completed),
            outcome,
        }
    }

    fn mo_kfsb_invocation() -> MoKfsbInvocation {
        MoKfsbInvocation {
            canonical_master: MoKfsbCanonicalMaster::Absent,
            legacy_childsim: false,
            configured_master: false,
            configured_candidates: 8,
            effective_candidates: 8,
            configured_reduce_op: MoKfsbReduceOp::Max,
            effective_reduce_op: MoKfsbReduceOp::Max,
            heuristic_eligible: true,
            typed_select: false,
            lower_master: false,
            effective_master: false,
            bounded_shared_lane: false,
            offered_domains: 3,
            selector_ran: false,
            committed_parents: 0,
            committed_leaves: Some(0),
        }
    }

    fn mo_alpha_step_invocation(effective_alpha_step: bool) -> MoAlphaStepInvocation {
        MoAlphaStepInvocation {
            effective_alpha_step,
            effective_beta_spsa: true,
            relu_split_route: true,
            has_split_constraints: true,
            has_alpha_state: true,
            alpha_coordinates: 7,
            eligible: true,
            eligible_iterations: 15,
            fourth_iteration_opportunities: 3,
            alpha_rounds_scheduled: usize::from(effective_alpha_step) * 3,
            beta_rounds_scheduled: if effective_alpha_step { 12 } else { 15 },
            alpha_rounds_completed: usize::from(effective_alpha_step) * 3,
            alpha_rounds_refused: 0,
            alpha_updates_applied: usize::from(effective_alpha_step) * 3,
            perturbed_coordinates: usize::from(effective_alpha_step) * 21,
        }
    }

    #[test]
    fn mo_alpha_step_receipts_preserve_off_counterfactual_and_on_work() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let run = begin_run();
        record_mo_alpha_step_invocation(mo_alpha_step_invocation(false));
        let off = snapshot().mo_alpha_step;
        assert!(off.observed);
        assert_eq!(off.effective_alpha_step, Some(false));
        assert_eq!(off.effective_beta_spsa, Some(true));
        assert_eq!(off.schedule_period, 4);
        assert_eq!(off.alpha_round_remainder, 3);
        assert_eq!(off.eligible_invocations, 1);
        assert_eq!(off.eligible_iterations, 15);
        assert_eq!(off.fourth_iteration_opportunities, 3);
        assert_eq!(off.alpha_rounds_scheduled, 0);
        assert_eq!(off.beta_rounds_scheduled, 15);
        assert!(!off.attribution_conflict);
        drop(run);

        let run = begin_run();
        record_mo_alpha_step_invocation(mo_alpha_step_invocation(true));
        let on = snapshot().mo_alpha_step;
        assert_eq!(on.alpha_rounds_scheduled, 3);
        assert_eq!(on.beta_rounds_scheduled, 12);
        assert_eq!(on.alpha_rounds_completed, 3);
        assert_eq!(on.alpha_rounds_refused, 0);
        assert_eq!(on.alpha_updates_applied, 3);
        assert_eq!(on.perturbed_coordinates, 21);
        assert!(!on.attribution_conflict);
        drop(run);
    }

    #[test]
    fn mo_alpha_step_receipt_rejects_false_admission_and_unclosed_work() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let run = begin_run();
        let mut false_admission = mo_alpha_step_invocation(true);
        false_admission.relu_split_route = false;
        record_mo_alpha_step_invocation(false_admission);
        assert!(snapshot().mo_alpha_step.attribution_conflict);
        drop(run);

        let run = begin_run();
        let mut unclosed = mo_alpha_step_invocation(true);
        unclosed.alpha_rounds_completed = 2;
        unclosed.alpha_updates_applied = 2;
        record_mo_alpha_step_invocation(unclosed);
        assert!(snapshot().mo_alpha_step.attribution_conflict);
        drop(run);

        let run = begin_run();
        let mut wrong_coordinate_total = mo_alpha_step_invocation(true);
        wrong_coordinate_total.perturbed_coordinates -= 1;
        record_mo_alpha_step_invocation(wrong_coordinate_total);
        assert!(snapshot().mo_alpha_step.attribution_conflict);
        drop(run);
    }

    #[test]
    fn mo_alpha_step_snapshot_rejects_impossible_aggregate_remainder() {
        let mut observations = super::ExecutionObservations {
            run_active: true,
            ..Default::default()
        };
        observations.mo_alpha_step.observed = true;
        observations.mo_alpha_step.effective_alpha_step = Some(false);
        observations.mo_alpha_step.effective_beta_spsa = Some(true);
        observations.mo_alpha_step.optimizer_invocations = 1;
        observations.mo_alpha_step.eligible_invocations = 1;
        observations.mo_alpha_step.eligible_iterations = 8;
        observations.mo_alpha_step.fourth_iteration_opportunities = 1;
        observations.mo_alpha_step.beta_rounds_scheduled = 8;

        let validated = super::validate_snapshot(observations);
        assert!(validated.mo_alpha_step.attribution_conflict);
    }

    #[test]
    fn mo_alpha_step_snapshot_rejects_impossible_coordinate_aggregates() {
        let mut observations = super::ExecutionObservations {
            run_active: true,
            ..Default::default()
        };
        observations.mo_alpha_step.observed = true;
        observations.mo_alpha_step.effective_alpha_step = Some(true);
        observations.mo_alpha_step.effective_beta_spsa = Some(true);
        observations.mo_alpha_step.optimizer_invocations = 1;
        observations.mo_alpha_step.eligible_invocations = 1;
        observations.mo_alpha_step.eligible_iterations = 8;
        observations.mo_alpha_step.fourth_iteration_opportunities = 2;
        observations.mo_alpha_step.alpha_rounds_scheduled = 2;
        observations.mo_alpha_step.beta_rounds_scheduled = 6;
        observations.mo_alpha_step.alpha_rounds_completed = 2;
        observations.mo_alpha_step.alpha_updates_applied = 2;

        observations.mo_alpha_step.perturbed_coordinates = 1;
        let too_few = super::validate_snapshot(observations.clone());
        assert!(too_few.mo_alpha_step.attribution_conflict);

        observations.mo_alpha_step.perturbed_coordinates = 3;
        let indivisible = super::validate_snapshot(observations);
        assert!(indivisible.mo_alpha_step.attribution_conflict);
    }

    #[test]
    fn mo_kfsb_inactive_recorder_does_not_lock_or_build_observation() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        assert!(!super::RECORDER.is_active());
        let builder_ran = std::cell::Cell::new(false);
        let pending = observation_recording_active().then(|| {
            builder_ran.set(true);
            mo_kfsb_invocation()
        });
        assert!(pending.is_none());
        assert!(!builder_ran.get());
        assert_eq!(snapshot(), Default::default());

        let run = begin_run();
        let pending = observation_recording_active().then(|| {
            builder_ran.set(true);
            mo_kfsb_invocation()
        });
        record_mo_kfsb_invocation(pending.expect("active observation primitive"));
        assert!(builder_ran.get());
        assert_eq!(snapshot().mo_kfsb.branch_opportunities, 1);
        drop(run);
        assert!(!super::RECORDER.is_active());

        // Hold an inactive recorder's authoritative mutex. Its record call
        // must still return before the lock is released, proving the false
        // atomic hint is a genuine no-lock path rather than an inert closure
        // reached after mutex acquisition.
        let recorder = Recorder::new();
        let held_state = recorder.state.lock().expect("inactive recorder state");
        let closure_ran = std::sync::atomic::AtomicBool::new(false);
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let returned_without_lock = std::thread::scope(|scope| {
            let recorder = &recorder;
            let closure_ran = &closure_ran;
            let worker = scope.spawn(move || {
                started_tx.send(()).expect("announce inactive record");
                recorder.record(|_| closure_ran.store(true, Ordering::Relaxed));
                done_tx.send(()).expect("complete inactive record");
            });
            started_rx.recv().expect("inactive record started");
            let returned = done_rx.recv_timeout(Duration::from_secs(1)).is_ok();
            drop(held_state);
            worker.join().expect("inactive recorder worker");
            returned
        });
        assert!(returned_without_lock);
        assert!(!closure_ran.load(Ordering::Relaxed));
    }

    #[test]
    fn mo_kfsb_exact_enable_authenticates_nonempty_parent_map_with_zero_leaves() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let _run = begin_run();
        let mut invocation = mo_kfsb_invocation();
        invocation.canonical_master = MoKfsbCanonicalMaster::ExactEnable;
        invocation.effective_master = true;
        invocation.selector_ran = true;
        invocation.committed_parents = 1;
        record_mo_kfsb_invocation(invocation);

        let observed = snapshot().mo_kfsb;
        assert_eq!(observed.schema, "ny_beta_crown_mo_kfsb_observations_v1");
        assert!(observed.observed);
        assert!(!observed.attribution_conflict);
        assert!(!observed.counter_overflow);
        assert_eq!(
            observed.canonical_master,
            Some(MoKfsbCanonicalMaster::ExactEnable)
        );
        assert_eq!(observed.configured_reduce_op, Some(MoKfsbReduceOp::Max));
        assert_eq!(observed.effective_reduce_op, Some(MoKfsbReduceOp::Max));
        assert_eq!(observed.branch_opportunities, 1);
        assert_eq!(observed.offered_domains, 3);
        assert_eq!(observed.nonempty_domain_opportunities, 1);
        assert_eq!(observed.candidate_eligible_opportunities, 1);
        assert_eq!(observed.lower_master_opportunities, 0);
        assert_eq!(observed.effective_master_opportunities, 1);
        assert_eq!(observed.route_eligible_opportunities, 1);
        assert_eq!(observed.selector_waves, 1);
        assert_eq!(observed.selector_domains, 3);
        assert_eq!(observed.committed_waves, 1);
        assert_eq!(observed.committed_parents, 1);
        assert_eq!(observed.committed_leaves, 0);
        assert_eq!(observed.force_on_binding_waves, 1);
        assert_eq!(observed.master_binding_refusals, 0);

        let json = serde_json::to_value(&observed).expect("serialize MO-kFSB observations");
        assert_eq!(json["canonical_master"], "exact_enable");
        assert_eq!(json["configured_reduce_op"], "max");
        assert_eq!(json["effective_reduce_op"], "max");
        assert_eq!(json["committed_leaves"], 0);

        let mut corrupted = snapshot();
        corrupted.mo_kfsb.selector_waves = 0;
        assert!(
            super::validate_snapshot(corrupted)
                .mo_kfsb
                .attribution_conflict
        );
        let mut missing_route = snapshot();
        missing_route.mo_kfsb.route_eligible_opportunities = 0;
        assert!(
            super::validate_snapshot(missing_route)
                .mo_kfsb
                .attribution_conflict
        );
        let mut missing_selector_domains = snapshot();
        missing_selector_domains.mo_kfsb.selector_domains = 0;
        assert!(
            super::validate_snapshot(missing_selector_domains)
                .mo_kfsb
                .attribution_conflict
        );
    }

    #[test]
    fn mo_kfsb_force_on_credit_requires_lower_off_and_nonempty_parent_map() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let exact_enable = || {
            let mut invocation = mo_kfsb_invocation();
            invocation.canonical_master = MoKfsbCanonicalMaster::ExactEnable;
            invocation.effective_master = true;
            invocation.selector_ran = true;
            invocation
        };
        let assert_no_binding = |invocation| {
            let run = begin_run();
            record_mo_kfsb_invocation(invocation);
            let observed = snapshot().mo_kfsb;
            assert!(!observed.attribution_conflict);
            assert_eq!(observed.force_on_binding_waves, 0);
            drop(run);
        };

        let mut already_armed = exact_enable();
        already_armed.configured_master = true;
        already_armed.lower_master = true;
        already_armed.committed_parents = 1;
        assert_no_binding(already_armed);

        assert_no_binding(exact_enable());
    }

    #[test]
    fn mo_kfsb_committed_leaf_capacity_fails_closed() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let exact_enable = |committed_leaves| {
            let mut invocation = mo_kfsb_invocation();
            invocation.canonical_master = MoKfsbCanonicalMaster::ExactEnable;
            invocation.effective_master = true;
            invocation.selector_ran = true;
            invocation.committed_parents = 1;
            invocation.committed_leaves = Some(committed_leaves);
            invocation
        };

        let run = begin_run();
        record_mo_kfsb_invocation(exact_enable(1025));
        assert!(snapshot().mo_kfsb.attribution_conflict);
        drop(run);

        let run = begin_run();
        record_mo_kfsb_invocation(exact_enable(1024));
        let mut valid = snapshot();
        assert!(!valid.mo_kfsb.attribution_conflict);
        valid.mo_kfsb.offered_domains = usize::MAX;
        valid.mo_kfsb.selector_domains = usize::MAX;
        valid.mo_kfsb.committed_parents = usize::MAX;
        valid.mo_kfsb.committed_leaves = 0;
        assert!(super::validate_snapshot(valid).mo_kfsb.attribution_conflict);
        drop(run);
    }

    #[test]
    fn mo_kfsb_exact_disable_refuses_only_a_real_lower_on_route() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let _run = begin_run();
        let mut invocation = mo_kfsb_invocation();
        invocation.canonical_master = MoKfsbCanonicalMaster::ExactDisable;
        invocation.configured_master = true;
        invocation.lower_master = true;
        record_mo_kfsb_invocation(invocation);

        let observed = snapshot().mo_kfsb;
        assert!(!observed.attribution_conflict);
        assert_eq!(observed.lower_master_opportunities, 1);
        assert_eq!(observed.effective_master_opportunities, 0);
        assert_eq!(observed.route_eligible_opportunities, 1);
        assert_eq!(observed.selector_waves, 0);
        assert_eq!(observed.master_binding_refusals, 1);
        assert_eq!(observed.force_on_binding_waves, 0);
    }

    #[test]
    fn mo_kfsb_disable_credit_requires_unbounded_positive_k_and_rejects_empty_input() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let exact_disable = || {
            let mut invocation = mo_kfsb_invocation();
            invocation.canonical_master = MoKfsbCanonicalMaster::ExactDisable;
            invocation.configured_master = true;
            invocation.lower_master = true;
            invocation
        };
        let assert_no_refusal = |invocation| {
            let run = begin_run();
            record_mo_kfsb_invocation(invocation);
            let observed = snapshot().mo_kfsb;
            assert!(!observed.attribution_conflict);
            assert_eq!(observed.route_eligible_opportunities, 0);
            assert_eq!(observed.master_binding_refusals, 0);
            drop(run);
        };

        let mut bounded = exact_disable();
        bounded.bounded_shared_lane = true;
        assert_no_refusal(bounded);

        let mut zero_candidates = exact_disable();
        zero_candidates.effective_candidates = 0;
        assert_no_refusal(zero_candidates);

        let run = begin_run();
        let mut impossible_empty = exact_disable();
        impossible_empty.offered_domains = 0;
        record_mo_kfsb_invocation(impossible_empty);
        let observed = snapshot().mo_kfsb;
        assert!(observed.attribution_conflict);
        assert_eq!(observed.branch_opportunities, 1);
        assert_eq!(observed.nonempty_domain_opportunities, 0);
        assert_eq!(observed.master_binding_refusals, 0);
        drop(run);
    }

    #[test]
    fn mo_kfsb_round_varying_typed_select_is_not_a_policy_conflict() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let _run = begin_run();
        let mut ordinary = mo_kfsb_invocation();
        ordinary.effective_reduce_op = MoKfsbReduceOp::Min;
        record_mo_kfsb_invocation(ordinary);

        let mut typed = ordinary;
        typed.typed_select = true;
        typed.lower_master = true;
        typed.effective_master = true;
        typed.selector_ran = true;
        record_mo_kfsb_invocation(typed);

        let observed = snapshot().mo_kfsb;
        assert!(!observed.attribution_conflict);
        assert_eq!(observed.branch_opportunities, 2);
        assert_eq!(observed.offered_domains, 6);
        assert_eq!(observed.typed_select_opportunities, 1);
        assert_eq!(observed.lower_master_opportunities, 1);
        assert_eq!(observed.effective_master_opportunities, 1);
        assert_eq!(observed.selector_waves, 1);
        assert_eq!(observed.selector_domains, 3);
    }

    #[test]
    fn mo_kfsb_multi_wave_partition_bounds_fail_closed() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");

        // One ordinary lower-off wave and one typed Select wave realize the
        // exact route/effective intersection and domain partition bounds.
        let run = begin_run();
        let mut ordinary = mo_kfsb_invocation();
        ordinary.effective_reduce_op = MoKfsbReduceOp::Min;
        ordinary.offered_domains = 1;
        record_mo_kfsb_invocation(ordinary);
        let mut typed = ordinary;
        typed.offered_domains = 3;
        typed.typed_select = true;
        typed.lower_master = true;
        typed.effective_master = true;
        typed.selector_ran = true;
        record_mo_kfsb_invocation(typed);
        let valid = snapshot();
        assert!(!valid.mo_kfsb.attribution_conflict);
        assert_eq!(valid.mo_kfsb.branch_opportunities, 2);
        assert_eq!(valid.mo_kfsb.route_eligible_opportunities, 2);
        assert_eq!(valid.mo_kfsb.effective_master_opportunities, 1);
        assert_eq!(valid.mo_kfsb.selector_waves, 1);
        assert_eq!(valid.mo_kfsb.offered_domains, 4);
        assert_eq!(valid.mo_kfsb.selector_domains, 3);

        let mut missing_required_intersection = valid.clone();
        missing_required_intersection.mo_kfsb.selector_waves = 0;
        missing_required_intersection.mo_kfsb.selector_domains = 0;
        assert!(
            super::validate_snapshot(missing_required_intersection)
                .mo_kfsb
                .attribution_conflict
        );
        let mut impossible_domain_partition = valid;
        impossible_domain_partition.mo_kfsb.selector_domains = 4;
        assert!(
            super::validate_snapshot(impossible_domain_partition)
                .mo_kfsb
                .attribution_conflict
        );
        drop(run);

        // Exact disable records every lower/route intersection as a refusal.
        let run = begin_run();
        let mut disabled = mo_kfsb_invocation();
        disabled.canonical_master = MoKfsbCanonicalMaster::ExactDisable;
        disabled.configured_master = true;
        disabled.lower_master = true;
        record_mo_kfsb_invocation(disabled);
        let mut missing_refusal = snapshot();
        assert_eq!(missing_refusal.mo_kfsb.master_binding_refusals, 1);
        missing_refusal.mo_kfsb.master_binding_refusals = 0;
        assert!(
            super::validate_snapshot(missing_refusal)
                .mo_kfsb
                .attribution_conflict
        );
        drop(run);

        // Exact enable records every committed wave outside the lower-on set.
        let run = begin_run();
        let mut enabled = mo_kfsb_invocation();
        enabled.canonical_master = MoKfsbCanonicalMaster::ExactEnable;
        enabled.effective_master = true;
        enabled.selector_ran = true;
        enabled.committed_parents = 1;
        record_mo_kfsb_invocation(enabled);
        let mut missing_force_on = snapshot();
        assert_eq!(missing_force_on.mo_kfsb.force_on_binding_waves, 1);
        missing_force_on.mo_kfsb.force_on_binding_waves = 0;
        assert!(
            super::validate_snapshot(missing_force_on)
                .mo_kfsb
                .attribution_conflict
        );
        drop(run);

        // A selector wave with no committed parent consumes one selector
        // domain that no other wave's committed parent can occupy.
        let run = begin_run();
        let mut empty_commit = mo_kfsb_invocation();
        empty_commit.canonical_master = MoKfsbCanonicalMaster::ExactEnable;
        empty_commit.effective_master = true;
        empty_commit.selector_ran = true;
        empty_commit.offered_domains = 1;
        record_mo_kfsb_invocation(empty_commit);
        let mut committed = empty_commit;
        committed.committed_parents = 1;
        record_mo_kfsb_invocation(committed);
        let valid = snapshot();
        assert!(!valid.mo_kfsb.attribution_conflict);
        assert_eq!(valid.mo_kfsb.selector_waves, 2);
        assert_eq!(valid.mo_kfsb.selector_domains, 2);
        assert_eq!(valid.mo_kfsb.committed_waves, 1);
        assert_eq!(valid.mo_kfsb.committed_parents, 1);
        let mut impossible_parent_partition = valid;
        impossible_parent_partition.mo_kfsb.committed_parents = 2;
        assert!(
            super::validate_snapshot(impossible_parent_partition)
                .mo_kfsb
                .attribution_conflict
        );
        drop(run);

        // When every offered wave ran the selector, every offered domain must
        // belong to the selector partition. The looser upper bounds alone do
        // not reject a missing domain in this all-selected case.
        let run = begin_run();
        let mut selected = mo_kfsb_invocation();
        selected.canonical_master = MoKfsbCanonicalMaster::ExactEnable;
        selected.effective_master = true;
        selected.selector_ran = true;
        selected.offered_domains = 2;
        record_mo_kfsb_invocation(selected);
        record_mo_kfsb_invocation(selected);
        let valid = snapshot();
        assert!(!valid.mo_kfsb.attribution_conflict);
        assert_eq!(valid.mo_kfsb.branch_opportunities, 2);
        assert_eq!(valid.mo_kfsb.selector_waves, 2);
        assert_eq!(valid.mo_kfsb.offered_domains, 4);
        assert_eq!(valid.mo_kfsb.selector_domains, 4);
        let mut missing_selected_domain = valid;
        missing_selected_domain.mo_kfsb.selector_domains = 3;
        assert!(
            super::validate_snapshot(missing_selected_domain)
                .mo_kfsb
                .attribution_conflict
        );
        drop(run);
    }

    #[test]
    fn mo_kfsb_reducer_is_run_common_but_governed_override_may_differ() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let run = begin_run();
        let mut invocation = mo_kfsb_invocation();
        invocation.configured_reduce_op = MoKfsbReduceOp::Mean;
        invocation.effective_reduce_op = MoKfsbReduceOp::Min;
        record_mo_kfsb_invocation(invocation);
        let observed = snapshot().mo_kfsb;
        assert!(!observed.attribution_conflict);
        assert_eq!(observed.configured_reduce_op, Some(MoKfsbReduceOp::Mean));
        assert_eq!(observed.effective_reduce_op, Some(MoKfsbReduceOp::Min));
        let json = serde_json::to_value(&observed).expect("serialize reducer receipt");
        assert_eq!(json["configured_reduce_op"], "mean");
        assert_eq!(json["effective_reduce_op"], "min");

        invocation.effective_reduce_op = MoKfsbReduceOp::Max;
        record_mo_kfsb_invocation(invocation);
        assert!(snapshot().mo_kfsb.attribution_conflict);
        drop(run);

        // The only route to effective Mean is configured Mean: the governed
        // environment accepts only min/max and malformed input defaults min.
        let run = begin_run();
        let mut impossible_mean = mo_kfsb_invocation();
        impossible_mean.effective_reduce_op = MoKfsbReduceOp::Mean;
        record_mo_kfsb_invocation(impossible_mean);
        let impossible = snapshot().mo_kfsb;
        assert!(impossible.attribution_conflict);
        assert_eq!(impossible.configured_reduce_op, Some(MoKfsbReduceOp::Max));
        assert_eq!(impossible.effective_reduce_op, Some(MoKfsbReduceOp::Mean));
        drop(run);

        // Validate the same invariant on an otherwise producer-realizable
        // aggregate so imported/corrupted receipts also fail closed.
        let run = begin_run();
        let mut configured_mean = mo_kfsb_invocation();
        configured_mean.configured_reduce_op = MoKfsbReduceOp::Mean;
        configured_mean.effective_reduce_op = MoKfsbReduceOp::Mean;
        record_mo_kfsb_invocation(configured_mean);
        let mut corrupted = snapshot();
        assert!(!corrupted.mo_kfsb.attribution_conflict);
        corrupted.mo_kfsb.configured_reduce_op = Some(MoKfsbReduceOp::Max);
        assert!(
            super::validate_snapshot(corrupted)
                .mo_kfsb
                .attribution_conflict
        );
        drop(run);
    }

    #[test]
    fn mo_kfsb_typed_select_requires_candidate_and_eligible_heuristic() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let run = begin_run();
        let mut impossible = mo_kfsb_invocation();
        impossible.effective_reduce_op = MoKfsbReduceOp::Min;
        impossible.heuristic_eligible = false;
        impossible.typed_select = true;
        impossible.lower_master = true;
        impossible.effective_master = true;
        record_mo_kfsb_invocation(impossible);
        let observed = snapshot().mo_kfsb;
        assert!(observed.attribution_conflict);
        assert_eq!(observed.typed_select_opportunities, 1);
        assert_eq!(observed.heuristic_eligible, Some(false));
        drop(run);

        let run = begin_run();
        let mut impossible = mo_kfsb_invocation();
        impossible.effective_reduce_op = MoKfsbReduceOp::Min;
        impossible.effective_candidates = 0;
        impossible.typed_select = true;
        impossible.lower_master = true;
        impossible.effective_master = true;
        record_mo_kfsb_invocation(impossible);
        let observed = snapshot().mo_kfsb;
        assert!(observed.attribution_conflict);
        assert_eq!(observed.typed_select_opportunities, 1);
        assert_eq!(observed.candidate_eligible_opportunities, 0);
        drop(run);
    }

    #[test]
    fn mo_kfsb_typed_select_requires_effective_min_reducer() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");

        // A primitive with otherwise self-consistent typed-Select policy is
        // producer-impossible when the captured effective reducer is Max.
        let run = begin_run();
        let mut impossible = mo_kfsb_invocation();
        impossible.typed_select = true;
        impossible.lower_master = true;
        impossible.effective_master = true;
        impossible.selector_ran = true;
        record_mo_kfsb_invocation(impossible);
        let observed = snapshot().mo_kfsb;
        assert!(observed.attribution_conflict);
        assert_eq!(observed.typed_select_opportunities, 1);
        assert_eq!(observed.effective_reduce_op, Some(MoKfsbReduceOp::Max));
        drop(run);

        // The aggregate validator independently poisons an imported receipt
        // whose typed-Select count is paired with a non-Min reducer.
        let run = begin_run();
        let mut valid = mo_kfsb_invocation();
        valid.effective_reduce_op = MoKfsbReduceOp::Min;
        valid.typed_select = true;
        valid.lower_master = true;
        valid.effective_master = true;
        valid.selector_ran = true;
        record_mo_kfsb_invocation(valid);
        let mut corrupted = snapshot();
        assert!(!corrupted.mo_kfsb.attribution_conflict);
        corrupted.mo_kfsb.effective_reduce_op = Some(MoKfsbReduceOp::Max);
        assert!(
            super::validate_snapshot(corrupted)
                .mo_kfsb
                .attribution_conflict
        );
        drop(run);
    }

    #[test]
    fn mo_kfsb_rejection_never_earns_disable_credit_and_overflow_fails_closed() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let _run = begin_run();
        let mut rejected = mo_kfsb_invocation();
        rejected.canonical_master = MoKfsbCanonicalMaster::Rejected;
        rejected.configured_master = true;
        rejected.lower_master = true;
        record_mo_kfsb_invocation(rejected);
        let rejected_observation = snapshot().mo_kfsb;
        assert!(!rejected_observation.attribution_conflict);
        assert_eq!(rejected_observation.master_binding_refusals, 0);
        assert_eq!(rejected_observation.force_on_binding_waves, 0);

        let mut overflow = mo_kfsb_invocation();
        overflow.canonical_master = MoKfsbCanonicalMaster::Rejected;
        overflow.configured_master = true;
        overflow.lower_master = true;
        overflow.committed_leaves = None;
        record_mo_kfsb_invocation(overflow);
        let poisoned = snapshot().mo_kfsb;
        assert!(poisoned.counter_overflow);
        assert!(poisoned.attribution_conflict);
        assert_eq!(poisoned.master_binding_refusals, 0);
        assert_eq!(poisoned.force_on_binding_waves, 0);
    }

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
    fn root_joint_interm_alpha_records_exact_live_work() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let _run = begin_run();
        record_root_joint_interm_alpha_invocation(root_joint_invocation(
            RootJointIntermAlphaOutcome::Completed,
        ));

        let observed = snapshot().root_joint_interm_alpha;
        assert_eq!(
            observed.schema,
            "ny_beta_crown_root_joint_interm_alpha_observations_v1"
        );
        assert!(observed.observed);
        assert!(!observed.attribution_conflict);
        assert_eq!(observed.invocations, 1);
        assert_eq!(observed.input_split_invocations, 1);
        assert_eq!(observed.multi_objective_invocations, 0);
        assert_eq!(observed.configured_master, Some(true));
        assert_eq!(observed.configured_max_dim_override, Some(32_768));
        assert_eq!(observed.input_split_configured_max_dim, Some(32_768));
        assert_eq!(observed.multi_objective_configured_max_dim, None);
        assert_eq!(observed.configured_secs, Some(30));
        assert_eq!(observed.configured_iters, Some(100));
        assert_eq!(observed.configured_lr_bits, Some(0.1f32.to_bits()));
        assert_eq!(observed.configured_deadline_ascent, Some(true));
        assert_eq!(observed.configured_min_remaining_secs, Some(240));
        assert_eq!(observed.configured_max_sel, Some(32));
        assert_eq!(observed.configured_frozen_stop, Some(true));
        assert_eq!(observed.configured_pure_chain, Some(true));
        assert_eq!(observed.configured_bn_gpu_extract, Some(true));
        assert_eq!(observed.max_effective_iters, 100);
        assert_eq!(observed.target_count, 3);
        assert_eq!(observed.available_budget_ms, 60_000);
        assert_eq!(observed.effective_slice_ms, 30_000);
        assert_eq!(observed.configured_slice_work_invocations, 1);
        assert_eq!(observed.configured_iters_binding_invocations, 1);
        assert_eq!(observed.lowered_min_remaining_work_invocations, 0);
        assert_eq!(observed.work_started, 1);
        assert_eq!(observed.attempts, 1);
        assert_eq!(observed.completed, 1);
        assert_eq!(observed.tightened_count, 1);
        assert_eq!(observed.outlier_tightened_count, 1);
        assert_eq!(observed.target_attempts, 3);
        assert_eq!(observed.base_folds_completed, 3);
        assert_eq!(observed.ascent_iterations_attempted, 100);
        assert_eq!(observed.max_ascent_iterations_attempted, 100);
        assert_eq!(observed.optimizer_steps, 100);
        assert_eq!(observed.ascent_folds_completed, 100);
        assert_eq!(observed.max_selected_rows, 32);
        assert_eq!(observed.selection_cap_bound_targets, 1);
        assert_eq!(observed.frozen_stop_targets, 1);
        assert_eq!(observed.pure_chain_targets, 1);
        assert_eq!(observed.bn_extract_targets, 1);
        assert_eq!(observed.no_target_refusals, 0);
    }

    #[test]
    fn root_joint_interm_alpha_repeated_work_preserves_parameter_binding() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let _run = begin_run();
        let invocation = root_joint_invocation(RootJointIntermAlphaOutcome::Completed);
        record_root_joint_interm_alpha_invocation(invocation);
        record_root_joint_interm_alpha_invocation(invocation);

        let observed = snapshot().root_joint_interm_alpha;
        assert_eq!(observed.outlier_tightened_count, 2);
        assert!(!observed.attribution_conflict);
        assert_eq!(observed.invocations, 2);
        assert_eq!(observed.attempts, 2);
        assert_eq!(observed.target_attempts, 6);
        assert_eq!(observed.configured_slice_work_invocations, 2);
        assert_eq!(observed.configured_iters_binding_invocations, 2);
        assert_eq!(observed.selection_cap_bound_targets, 2);
    }

    #[test]
    fn root_joint_interm_alpha_mixed_contextual_routes_do_not_conflict() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let _run = begin_run();
        let mut input = root_joint_invocation(RootJointIntermAlphaOutcome::Completed);
        input.configured_max_dim_override = None;
        record_root_joint_interm_alpha_invocation(input);

        let mut multi = root_joint_invocation(RootJointIntermAlphaOutcome::NoTargets);
        multi.route = RootJointIntermAlphaRoute::MultiObjective;
        multi.configured_max_dim_override = None;
        multi.configured_max_dim = 32_768;
        multi.available_budget = Duration::from_secs(20);
        multi.effective_slice = Duration::from_secs(10);
        record_root_joint_interm_alpha_invocation(multi);

        let observed = snapshot().root_joint_interm_alpha;
        assert!(!observed.attribution_conflict);
        assert_eq!(observed.invocations, 2);
        assert_eq!(observed.input_split_configured_max_dim, Some(32_768));
        assert_eq!(observed.multi_objective_configured_max_dim, Some(32_768));
        assert_eq!(observed.configured_max_dim_override, None);
        assert_eq!(observed.max_effective_iters, 100);
    }

    #[test]
    fn root_joint_interm_alpha_records_raised_floor_binding_refusal() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let _run = begin_run();
        let mut refusal = root_joint_invocation(RootJointIntermAlphaOutcome::AdmissionFloor);
        refusal.route = RootJointIntermAlphaRoute::MultiObjective;
        refusal.configured_min_remaining_secs = 360;
        refusal.available_budget = Duration::from_mins(5);
        record_root_joint_interm_alpha_invocation(refusal);

        let observed = snapshot().root_joint_interm_alpha;
        assert!(!observed.attribution_conflict);
        assert_eq!(observed.admission_floor_refusals, 1);
        assert_eq!(observed.raised_min_remaining_binding_refusals, 1);
        assert_eq!(observed.work_started, 0);
    }

    #[test]
    fn root_joint_interm_alpha_refusals_and_contradictions_fail_closed() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let _run = begin_run();

        let mut no_targets = root_joint_invocation(RootJointIntermAlphaOutcome::NoTargets);
        no_targets.route = RootJointIntermAlphaRoute::MultiObjective;
        no_targets.available_budget = Duration::from_secs(20);
        no_targets.effective_slice = Duration::from_secs(10);
        record_root_joint_interm_alpha_invocation(no_targets);
        let refusal = snapshot().root_joint_interm_alpha;
        assert!(!refusal.attribution_conflict);
        assert_eq!(refusal.invocations, 1);
        assert_eq!(refusal.multi_objective_invocations, 1);
        assert_eq!(refusal.no_target_refusals, 1);
        assert_eq!(refusal.attempts, 0);
        assert_eq!(refusal.work_started, 0);

        let mut contradictory = root_joint_invocation(RootJointIntermAlphaOutcome::Completed);
        contradictory.target_count = 0;
        record_root_joint_interm_alpha_invocation(contradictory);
        let conflicted = snapshot().root_joint_interm_alpha;
        assert_eq!(conflicted.invocations, 2);
        assert!(conflicted.attribution_conflict);
    }

    #[test]
    fn root_joint_interm_alpha_rejects_work_beyond_selected_targets() {
        let _test_lock = super::TEST_LOCK.lock().expect("telemetry test lock");
        let _run = begin_run();
        let mut impossible = root_joint_invocation(RootJointIntermAlphaOutcome::Completed);
        impossible.target_count = 2;
        record_root_joint_interm_alpha_invocation(impossible);
        assert!(snapshot().root_joint_interm_alpha.attribution_conflict);
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
            assert_eq!(observed.schema, "ny_beta_crown_execution_observations_v8");
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
