// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Margin-row twin-wall UNSAT lane — vnncomp glue (#twinwall).
//!
//! ONNX -> [`TwinSpec`] extractor (f64 BN fold with certified error
//! bookkeeping, Python-reference parity: `core.py::Net.__init__`), the
//! robustness vnnlib parser, and the strictly-additive vnncomp hook: after
//! the generic verifier returns unknown/timeout and the attack found no CE,
//! run the margin-row BaB with the remaining budget. The authority gate is a
//! trusted code constant after the tier sweep and enclosure obligations passed;
//! the performance recipe remains independently kill-switchable.
//!
//! Everything here fails CLOSED: any structural deviation from the twin-wall
//! family (Conv+BN trunk, ReLU/Add, head Gemm->ReLU->Gemm) returns `None` and
//! the caller keeps its existing verdict.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::{Duration, Instant};

use prost::Message;

use ny_onnx::onnx_proto::{GraphProto, ModelProto, NodeProto, TensorProto};
use ny_propagate::margin_row::dense_bn_fold::{fold_dense_batchnorm, DenseBnFold, DenseBnParams};
use ny_propagate::margin_row::{
    run_margin_row_lane, run_margin_row_lane_with_policy, BabStats,
    BackwardIntermExecutionEvidence, EpochExecutionEvidence, FrontierExecutionEvidence,
    FrontierRoute, MarginRowOutcome, MarginRowRuntimePolicy, PolicyExecutionEvidence,
    RootAlphaExecutionEvidence, RootPrecisionExecutionEvidence, TwinOpSpec, TwinSpec,
    WidthExecutionEvidence,
};

use super::unsat_receipt::UnsatReceipt;
use super::vnncomp::VnncompResult;

/// f64 unit roundoff.
const U: f64 = 1.110_223_024_625_156_5e-16;

const MARGIN_ROW_FRONTIER_EXECUTION_METHOD: &str = "margin_row_frontier_execution";
const MARGIN_ROW_FRONTIER_EXECUTION_SCHEMA: &str = "ny-cli/margin-row-frontier-execution/v1";
const MARGIN_ROW_WIDTH_EXECUTION_METHOD: &str = "margin_row_width_execution";
const MARGIN_ROW_WIDTH_EXECUTION_SCHEMA: &str = "ny-margin-row-width-execution/v2";
const MARGIN_ROW_EPOCH_EXECUTION_METHOD: &str = "margin_row_epoch_execution";
const MARGIN_ROW_EPOCH_EXECUTION_SCHEMA: &str = "ny-cli/margin-row-epoch-execution/v1";
const MARGIN_ROW_DOMAIN_ALPHA_EXECUTION_METHOD: &str = "margin_row_domain_alpha_execution";
const MARGIN_ROW_DOMAIN_ALPHA_EXECUTION_SCHEMA: &str =
    "ny-cli/margin-row-domain-alpha-execution/v1";
const MARGIN_ROW_POLICY_EXECUTION_METHOD: &str = "margin_row_policy_execution";
const MARGIN_ROW_POLICY_EXECUTION_SCHEMA: &str = "ny-cli/margin-row-policy-execution/v1";
const MARGIN_ROW_BACKWARD_INTERM_EXECUTION_METHOD: &str = "margin_row_backward_interm_execution";
const MARGIN_ROW_BACKWARD_INTERM_EXECUTION_SCHEMA: &str =
    "ny-cli/margin-row-backward-interm-execution/v1";
const MARGIN_ROW_ROOT_PRECISION_EXECUTION_METHOD: &str = "margin_row_root_precision_execution";
const MARGIN_ROW_ROOT_PRECISION_EXECUTION_SCHEMA: &str =
    "ny-cli/margin-row-root-precision-execution/v1";
const MARGIN_ROW_ROOT_ALPHA_EXECUTION_METHOD: &str = "margin_row_root_alpha_execution";
const MARGIN_ROW_ROOT_ALPHA_EXECUTION_SCHEMA: &str = "ny-cli/margin-row-root-alpha-execution/v2";
const MARGIN_ROW_CONCURRENT_OVERLAP_EXECUTION_METHOD: &str =
    "margin_row_concurrent_overlap_execution";
const MARGIN_ROW_CONCURRENT_OVERLAP_EXECUTION_SCHEMA: &str =
    "ny-cli/margin-row-concurrent-overlap-execution/v1";
const MARGIN_ROW_CLOSE_INTEGRITY_METHOD: &str = "margin_row_close_integrity";
const MARGIN_ROW_CLOSE_INTEGRITY_SCHEMA: &str = "ny-cli/margin-row-close-integrity/v1";

/// #mr-close-telemetry: the PROOF-INTEGRITY receipt — the certified value of
/// every close the lane's verdict rests on (thinnest certificate as exact bits
/// and decimal, split by kind, counts below 1e-6 / 1e-8 / 1e-10, the smallest
/// root `dj`). Written into the FLIGHT RECORD because it is the one artifact
/// that survives `ny benchmarks run` (stdout nulled, stderr reduced to an
/// error tail). `phase=terminal` for a lane that returned (`source=stats`, or
/// `source=no_stats` for a lane that stopped before owning stats — every
/// field then reads empty); `phase=snapshot outcome=running` for a lane still
/// running when the scored command records its verdict (`source=` is where
/// the published tally came from: `live`, or `none` before the root closed
/// anything). Telemetry only.
fn render_margin_row_close_integrity(
    phase: &str,
    outcome: &str,
    source: &str,
    evidence: &ny_propagate::margin_row::close_integrity::CloseIntegrityEvidence,
) -> String {
    format!(
        "schema={MARGIN_ROW_CLOSE_INTEGRITY_SCHEMA} phase={phase} outcome={outcome} \
         source={source} {}",
        evidence.fields()
    )
}

/// #mr-close-telemetry: record a SNAPSHOT proof-integrity receipt when a
/// margin-row lane is still running as the scored command records its verdict
/// (the detached concurrent worker on a timeout row, an inline worker
/// abandoned at its slice cap, or any lane still live when another route
/// decided the row). A lane that returned has already recorded its terminal
/// receipt from `note_margin_row_execution`, so this is silent then. Called
/// once, from `flight::finish`, before the sidecar is written.
pub(crate) fn note_margin_row_close_integrity_snapshot_if_live() {
    if let Some((source, evidence)) = ny_propagate::margin_row::close_integrity::snapshot_if_live()
    {
        crate::flight::note(
            MARGIN_ROW_CLOSE_INTEGRITY_METHOD,
            crate::flight::FlightStatus::Ran,
            Some(render_margin_row_close_integrity(
                "snapshot",
                "running",
                source.token(),
                &evidence,
            )),
        );
    }
}

fn render_margin_row_concurrent_overlap_execution() -> String {
    format!(
        "schema={MARGIN_ROW_CONCURRENT_OVERLAP_EXECUTION_SCHEMA} phase=work \
         worker_active=true verifier_active=true"
    )
}

fn render_margin_row_frontier_execution(
    outcome: &str,
    evidence: FrontierExecutionEvidence,
) -> String {
    format!(
        "schema={MARGIN_ROW_FRONTIER_EXECUTION_SCHEMA} phase=terminal route={} outcome={outcome} \
         tree_started={} configured={} contextual={} steps={} counterfactual_differences={} \
         max_open={} max_batch={}",
        evidence.route.stable_token(),
        evidence.steps > 0,
        evidence.configured,
        evidence.contextual,
        evidence.steps,
        evidence.counterfactual_differences,
        evidence.max_open,
        evidence.max_batch,
    )
}

fn render_margin_row_width_execution(outcome: &str, evidence: WidthExecutionEvidence) -> String {
    format!(
        "schema={MARGIN_ROW_WIDTH_EXECUTION_SCHEMA} phase=terminal outcome={outcome} \
         configured_k_head={} contextual_k_head={} configured_k_trunk={} \
         contextual_k_trunk={} adaptive_enabled={} contextual_adaptive={} \
         tier0_exact={} effective_k_head_max={} effective_k_trunk_max={} \
         effective_contextual_k_head_max={} effective_contextual_k_trunk_max={} \
         outer_head_selection_calls={} outer_head_available_max={} \
         outer_head_selected_total={} outer_head_binding_scored_calls={} \
         outer_trunk_selection_calls={} outer_trunk_available_max={} \
         outer_trunk_selected_total={} outer_trunk_binding_scored_calls={} \
         nested_head_selection_calls={} nested_trunk_selection_calls={} \
         configured_adaptive_threshold_opportunities={} \
         configured_adaptive_escalations={} \
         contextual_adaptive_threshold_opportunities={} \
         contextual_adaptive_escalations={} policy_mismatch={}",
        evidence.configured_k_head,
        evidence.contextual_k_head,
        evidence.configured_k_trunk,
        evidence.contextual_k_trunk,
        evidence.adaptive_enabled,
        evidence.contextual_adaptive,
        evidence.tier0_exact,
        evidence.effective_k_head_max,
        evidence.effective_k_trunk_max,
        evidence.effective_contextual_k_head_max,
        evidence.effective_contextual_k_trunk_max,
        evidence.outer_head_selection_calls,
        evidence.outer_head_available_max,
        evidence.outer_head_selected_total,
        evidence.outer_head_binding_scored_calls,
        evidence.outer_trunk_selection_calls,
        evidence.outer_trunk_available_max,
        evidence.outer_trunk_selected_total,
        evidence.outer_trunk_binding_scored_calls,
        evidence.nested_head_selection_calls,
        evidence.nested_trunk_selection_calls,
        evidence.configured_adaptive_threshold_opportunities,
        evidence.configured_adaptive_escalations,
        evidence.contextual_adaptive_threshold_opportunities,
        evidence.contextual_adaptive_escalations,
        evidence.policy_mismatch,
    )
}

fn render_margin_row_epoch_execution(
    outcome: &str,
    route: FrontierRoute,
    evidence: &EpochExecutionEvidence,
) -> String {
    format!(
        "schema={MARGIN_ROW_EPOCH_EXECUTION_SCHEMA} phase=terminal route={} outcome={outcome} \
         configured_depth={} configured_attempts={} configured_tier0={} \
         configured_universe={} configured_retain={} configured_parent_topk={} \
         configured_debug={} policy_mismatch={} tree_entries={} \
         configured_depth_admissions={} configured_depth_nested_with_stats={} \
         adaptive_depth_admissions={} attempt_cap_decisions={} \
         attempt_cap_counterfactual_differences={} attempt_cap_refusals={} \
         generation_refusals={} deadline_refusals={} attempts_started={} \
         attempt_failures={} attempts_closed={} outer_full_attempts={} \
         outer_incremental_attempts={} nested_full_attempts={} \
         nested_incremental_attempts={} full_rebuilds_completed={} \
         full_rebuild_refusals={} parent_refreshes_started={} \
         parent_refreshes_completed={} parent_refreshes_accepted={} \
         parent_refresh_refusals={} parent_refresh_columns={} \
         parent_refresh_tightened={} parent_refresh_stabilized={} \
         nested_started={} nested_runs_with_stats={} nested_unsat={} \
         nested_unknown={} nested_errors={} closures={} debug_emissions={} \
         outer_tier0_calls={} outer_tier0_scored_calls={} \
         outer_tier0_exact_binding_scored_calls={} \
         outer_universe_counterfactual_scored_calls={} \
         outer_retain_counterfactual_scored_calls={} nested_full_tier0_calls={} \
         nested_incremental_tier0_calls={}",
        route.stable_token(),
        evidence.policy.depth,
        evidence.policy.max_attempts,
        evidence.policy.tier0_exact,
        evidence.policy.tier0_universe,
        evidence.policy.retain_per_layer,
        evidence.policy.parent_topk,
        evidence.policy.debug_trace,
        evidence.policy_mismatch,
        evidence.tree_entries,
        evidence.configured_depth_admissions,
        evidence.configured_depth_nested_with_stats,
        evidence.adaptive_depth_admissions,
        evidence.attempt_cap_decisions,
        evidence.attempt_cap_counterfactual_differences,
        evidence.attempt_cap_refusals,
        evidence.generation_refusals,
        evidence.deadline_refusals,
        evidence.attempts_started,
        evidence.attempt_failures,
        evidence.attempts_closed,
        evidence.outer_full_attempts,
        evidence.outer_incremental_attempts,
        evidence.nested_full_attempts,
        evidence.nested_incremental_attempts,
        evidence.full_rebuilds_completed,
        evidence.full_rebuild_refusals,
        evidence.parent_refreshes_started,
        evidence.parent_refreshes_completed,
        evidence.parent_refreshes_accepted,
        evidence.parent_refresh_refusals,
        evidence.parent_refresh_columns,
        evidence.parent_refresh_tightened,
        evidence.parent_refresh_stabilized,
        evidence.nested_started,
        evidence.nested_runs_with_stats,
        evidence.nested_unsat,
        evidence.nested_unknown,
        evidence.nested_errors,
        evidence.closures,
        evidence.debug_emissions,
        evidence.outer_tier0_calls,
        evidence.outer_tier0_scored_calls,
        evidence.outer_tier0_exact_binding_scored_calls,
        evidence.outer_universe_counterfactual_scored_calls,
        evidence.outer_retain_counterfactual_scored_calls,
        evidence.nested_full_tier0_calls,
        evidence.nested_incremental_tier0_calls,
    )
}

fn render_margin_row_domain_alpha_execution(
    outcome: &str,
    route: FrontierRoute,
    evidence: PolicyExecutionEvidence,
) -> String {
    format!(
        "schema={MARGIN_ROW_DOMAIN_ALPHA_EXECUTION_SCHEMA} phase=terminal route={} \
         outcome={outcome} configured_enabled={} configured_iters={} \
         configured_eta_bits={} policy_mismatch={} gate_evals={} \
         movable_domains={} movable_slots={} warm_entries_applied={} \
         ascent_domains={} outer_ascent_domains={} proposals={} \
         trial_passes_completed={} trial_pass_failures={} finite_trial_scores={} \
         accepted_trials={} carried_entries={} iters_counterfactual_domains={} \
         eta_counterfactual_scored_domains={} outer_finite_trial_scores={} \
         outer_iters_counterfactual_domains={} \
         outer_eta_counterfactual_scored_domains={}",
        route.stable_token(),
        evidence.domain_alpha_enabled,
        evidence.domain_alpha_iters,
        evidence.domain_alpha_eta_bits,
        evidence.policy_mismatch,
        evidence.domain_alpha_gate_evals,
        evidence.domain_alpha_movable_evals,
        evidence.domain_alpha_movable_slots,
        evidence.domain_alpha_warm_applied,
        evidence.domain_alpha_ascent_calls,
        evidence.domain_alpha_outer_ascent_calls,
        evidence.domain_alpha_trials_started,
        evidence.domain_alpha_trial_passes_completed,
        evidence.domain_alpha_trial_pass_failures,
        evidence.domain_alpha_finite_trial_scores,
        evidence.domain_alpha_trials_accepted,
        evidence.domain_alpha_carried_entries,
        evidence.domain_alpha_iters_counterfactual_domains,
        evidence.domain_alpha_eta_counterfactual_scored_domains,
        evidence.domain_alpha_outer_finite_trial_scores,
        evidence.domain_alpha_outer_iters_counterfactual_domains,
        evidence.domain_alpha_outer_eta_counterfactual_scored_domains,
    )
}

fn render_margin_row_policy_execution(outcome: &str, evidence: PolicyExecutionEvidence) -> String {
    format!(
        "schema={MARGIN_ROW_POLICY_EXECUTION_SCHEMA} phase=terminal outcome={outcome} \
         beta_per_column={} beta_columns={} column_retire={} parallel_prebuild={} \
         beta_per_column_evals={} beta_columns_binding_evals={} \
         beta_columns_max_candidates={} newly_retired_columns={} \
         inherited_retirement_evals={} parallel_prebuild_batches={} \
         parallel_prebuild_domains={} policy_mismatch={}",
        evidence.beta_per_column,
        evidence.beta_columns,
        evidence.column_retire,
        evidence.parallel_prebuild,
        evidence.beta_per_column_evals,
        evidence.beta_columns_binding_evals,
        evidence.beta_columns_max_candidates,
        evidence.newly_retired_columns,
        evidence.inherited_retirement_evals,
        evidence.parallel_prebuild_batches,
        evidence.parallel_prebuild_domains,
        evidence.policy_mismatch,
    )
}

fn render_margin_row_backward_interm_execution(
    evidence: BackwardIntermExecutionEvidence,
) -> String {
    format!(
        "schema={MARGIN_ROW_BACKWARD_INTERM_EXECUTION_SCHEMA} phase=terminal outcome={} \
         enabled={} configured_secs={} configured_chunk={} configured_chunk_secs={} \
         effective_chunk_secs={} configured_topk={} eligible_layers={} \
         suppressed_setup_layers={} layers_started={} chunks_completed={} \
         columns_completed={} deadline_capped={} budget_exhausted={} \
         chunk_secs_capped={} chunk_counterfactual_chunks={} chunk_ceiling_binding_chunks={} \
         fixed_width_chunks={} \
         adaptive_width_changes={} topk_binding_layers={} \
         policy_mismatch={}",
        evidence.outcome(),
        evidence.enabled,
        evidence.configured_secs,
        evidence.configured_chunk,
        evidence.configured_chunk_secs,
        evidence.effective_chunk_secs,
        evidence.configured_topk,
        evidence.eligible_layers,
        evidence.suppressed_setup_layers,
        evidence.layers_started,
        evidence.chunks_completed,
        evidence.columns_completed,
        evidence.deadline_capped,
        evidence.budget_exhausted,
        evidence.chunk_secs_capped,
        evidence.chunk_counterfactual_chunks,
        evidence.chunk_ceiling_binding_chunks,
        evidence.fixed_width_chunks,
        evidence.adaptive_width_changes,
        evidence.topk_binding_layers,
        evidence.policy_mismatch,
    )
}

fn render_margin_row_root_precision_execution(evidence: RootPrecisionExecutionEvidence) -> String {
    format!(
        "schema={MARGIN_ROW_ROOT_PRECISION_EXECUTION_SCHEMA} phase=terminal outcome=completed \
         configured_root_f32={} configured_tableau_f32={} outward_eligible={} \
         eligible_conv_ops={} root_f32_conv_ops={} root_f64_conv_ops={} \
         eligible_store_ops={} tableau_f32_store_ops={} tableau_f64_store_ops={} \
         tableau_refused={} policy_mismatch={}",
        evidence.configured_root_f32,
        evidence.configured_tableau_f32,
        evidence.outward_eligible,
        evidence.eligible_conv_ops,
        evidence.root_f32_conv_ops,
        evidence.root_f64_conv_ops,
        evidence.eligible_store_ops,
        evidence.tableau_f32_store_ops,
        evidence.tableau_f64_store_ops,
        evidence.tableau_refused,
        evidence.policy_mismatch,
    )
}

fn optional_bits(value: Option<u64>) -> String {
    value.map_or_else(|| "none".to_owned(), |bits| bits.to_string())
}

fn render_margin_row_root_alpha_execution(
    outcome: &str,
    evidence: &RootAlphaExecutionEvidence,
) -> String {
    format!(
        "schema={MARGIN_ROW_ROOT_ALPHA_EXECUTION_SCHEMA} phase=terminal outcome={outcome} \
         alpha_enabled={} alpha_iters={} alpha_secs_bits={} alpha_lr_bits={} \
         alpha_cols={} alpha_guard={} rebake_enabled={} rebake_rounds={} \
         rebake_secs_bits={} eligible={} standalone_suppressed={} \
         standalone_attempted={} standalone_completed={} \
         standalone_effective_secs_bits={} standalone_deadline_capped={} \
         standalone_budget_exhausted={} standalone_iters_cap_exhausted={} \
         standalone_fail_classes={} standalone_iters_attempted={} \
         standalone_candidate_steps={} \
         standalone_legacy_steps={} standalone_adam_steps={} standalone_passes={} \
         standalone_max_steer_cols={} standalone_frozen_improvements={} \
         standalone_accepted={} standalone_moved_neurons={} \
         standalone_guard_opportunities={} standalone_guard_baseline_evals={} \
         standalone_guard_candidate_evals={} standalone_guard_reversions={} \
         standalone_frozen_before_bits={} standalone_frozen_best_bits={} \
         standalone_published_before_bits={} standalone_published_after_bits={} \
         rebake_suppressed={} rebake_attempted={} rebake_completed={} \
         rebake_effective_secs_bits={} rebake_deadline_capped={} \
         rebake_budget_exhausted={} rebake_round_cap_exhausted={} \
         rebake_entry_unbaked_alpha={} rebake_baseline_evals={} \
         rebake_rounds_started={} rebake_alpha_completed={} \
         rebake_alpha_iters_attempted={} rebake_alpha_candidate_steps={} \
         rebake_alpha_legacy_steps={} \
         rebake_alpha_adam_steps={} rebake_alpha_passes={} \
         rebake_alpha_max_fail_classes={} rebake_alpha_max_steer_cols={} \
         rebake_alpha_moved_neurons={} rebake_rebuilds_completed={} \
         rebake_rounds_scored={} rebake_rounds_accepted={} \
         rebake_fixed_point_stops={} rebake_affordability_refusals={} \
         rebake_aborts={} rebake_published_before_bits={} \
         rebake_published_after_bits={} spec_alpha_enabled={} spec_alpha_width={} \
         spec_alpha_cands={} spec_alpha_rounds={} spec_alpha_entries={} \
         spec_alpha_ran={} spec_alpha_rounds_run={} spec_alpha_passes={} \
         spec_alpha_entries_committed={} spec_alpha_classes_selected={} \
         spec_alpha_incumbent_grads_ok={} \
         spec_alpha_classes_improved={} spec_alpha_distinct_winners={} \
         spec_alpha_incumbent_grad_improvements={} spec_alpha_closed_extra={} \
         spec_alpha_truncated={} spec_alpha_root_before_bits={} \
         spec_alpha_root_after_bits={} spec_alpha_best_gain_bits={} \
         policy_mismatch={}",
        evidence.alpha_enabled,
        evidence.alpha_iters,
        evidence.alpha_secs.to_bits(),
        evidence.alpha_lr.to_bits(),
        evidence.alpha_cols,
        evidence.alpha_guard,
        evidence.rebake_enabled,
        evidence.rebake_rounds,
        evidence.rebake_secs.to_bits(),
        evidence.eligible,
        evidence.standalone_suppressed,
        evidence.standalone_attempted,
        evidence.standalone_completed,
        evidence.standalone_effective_secs.to_bits(),
        evidence.standalone_deadline_capped,
        evidence.standalone_budget_exhausted,
        evidence.standalone_iters_cap_exhausted,
        evidence.standalone_fail_classes,
        evidence.standalone_iters_attempted,
        evidence.standalone_candidate_steps,
        evidence.standalone_legacy_steps,
        evidence.standalone_adam_steps,
        evidence.standalone_passes,
        evidence.standalone_max_steer_cols,
        evidence.standalone_frozen_improvements,
        evidence.standalone_accepted,
        evidence.standalone_moved_neurons,
        evidence.standalone_guard_opportunities,
        evidence.standalone_guard_baseline_evals,
        evidence.standalone_guard_candidate_evals,
        evidence.standalone_guard_reversions,
        optional_bits(evidence.standalone_frozen_before_bits),
        optional_bits(evidence.standalone_frozen_best_bits),
        optional_bits(evidence.standalone_published_before_bits),
        optional_bits(evidence.standalone_published_after_bits),
        evidence.rebake_suppressed,
        evidence.rebake_attempted,
        evidence.rebake_completed,
        evidence.rebake_effective_secs.to_bits(),
        evidence.rebake_deadline_capped,
        evidence.rebake_budget_exhausted,
        evidence.rebake_round_cap_exhausted,
        evidence.rebake_entry_unbaked_alpha,
        evidence.rebake_baseline_evals,
        evidence.rebake_rounds_started,
        evidence.rebake_alpha_completed,
        evidence.rebake_alpha_iters_attempted,
        evidence.rebake_alpha_candidate_steps,
        evidence.rebake_alpha_legacy_steps,
        evidence.rebake_alpha_adam_steps,
        evidence.rebake_alpha_passes,
        evidence.rebake_alpha_max_fail_classes,
        evidence.rebake_alpha_max_steer_cols,
        evidence.rebake_alpha_moved_neurons,
        evidence.rebake_rebuilds_completed,
        evidence.rebake_rounds_scored,
        evidence.rebake_rounds_accepted,
        evidence.rebake_fixed_point_stops,
        evidence.rebake_affordability_refusals,
        evidence.rebake_aborts,
        optional_bits(evidence.rebake_published_before_bits),
        optional_bits(evidence.rebake_published_after_bits),
        evidence.spec_alpha_enabled,
        evidence.spec_alpha_width,
        evidence.spec_alpha_cands,
        evidence.spec_alpha_rounds,
        evidence.spec_alpha_entries,
        evidence.spec_alpha_ran,
        evidence.spec_alpha_rounds_run,
        evidence.spec_alpha_passes,
        evidence.spec_alpha_entries_committed,
        evidence.spec_alpha_classes_selected,
        evidence.spec_alpha_incumbent_grads_ok,
        evidence.spec_alpha_classes_improved,
        evidence.spec_alpha_distinct_winners,
        evidence.spec_alpha_incumbent_grad_improvements,
        evidence.spec_alpha_closed_extra,
        evidence.spec_alpha_truncated,
        optional_bits(evidence.spec_alpha_root_before_bits),
        optional_bits(evidence.spec_alpha_root_after_bits),
        optional_bits(evidence.spec_alpha_best_gain_bits),
        evidence.policy_mismatch,
    )
}

/// Publish one terminal receipt only when the lane reached a stats-owning
/// state. Compile/root-build failures carry no tree decision and deliberately
/// earn no optimizer engagement credit.
fn note_margin_row_execution(out: &MarginRowOutcome) {
    let (outcome, stats) = match out {
        MarginRowOutcome::Unsat(stats, _) => ("unsat", Some(stats)),
        MarginRowOutcome::Unknown { stats, .. } => ("unknown", stats.as_ref()),
    };
    // #mr-close-telemetry: unlike the execution receipts below, EVERY returned
    // lane leaves a proof-integrity receipt — a stats-less stop says so
    // (`source=no_stats`) instead of leaving an absence a reader could mistake
    // for a binary without the receipt.
    crate::flight::note(
        MARGIN_ROW_CLOSE_INTEGRITY_METHOD,
        crate::flight::FlightStatus::Ran,
        Some(match stats {
            Some(stats) => render_margin_row_close_integrity(
                "terminal",
                outcome,
                "stats",
                &stats.close_integrity,
            ),
            None => render_margin_row_close_integrity(
                "terminal",
                outcome,
                "no_stats",
                &ny_propagate::margin_row::close_integrity::CloseIntegrityEvidence::EMPTY,
            ),
        }),
    );
    if let Some(stats) = stats {
        crate::flight::note(
            MARGIN_ROW_FRONTIER_EXECUTION_METHOD,
            crate::flight::FlightStatus::Ran,
            Some(render_margin_row_frontier_execution(
                outcome,
                stats.frontier_execution,
            )),
        );
        crate::flight::note(
            MARGIN_ROW_WIDTH_EXECUTION_METHOD,
            crate::flight::FlightStatus::Ran,
            Some(render_margin_row_width_execution(
                outcome,
                stats.width_execution,
            )),
        );
        crate::flight::note(
            MARGIN_ROW_EPOCH_EXECUTION_METHOD,
            crate::flight::FlightStatus::Ran,
            Some(render_margin_row_epoch_execution(
                outcome,
                stats.frontier_execution.route,
                &stats.epoch_execution,
            )),
        );
        crate::flight::note(
            MARGIN_ROW_DOMAIN_ALPHA_EXECUTION_METHOD,
            crate::flight::FlightStatus::Ran,
            Some(render_margin_row_domain_alpha_execution(
                outcome,
                stats.frontier_execution.route,
                stats.policy_execution,
            )),
        );
        crate::flight::note(
            MARGIN_ROW_POLICY_EXECUTION_METHOD,
            crate::flight::FlightStatus::Ran,
            Some(render_margin_row_policy_execution(
                outcome,
                stats.policy_execution,
            )),
        );
        crate::flight::note(
            MARGIN_ROW_BACKWARD_INTERM_EXECUTION_METHOD,
            crate::flight::FlightStatus::Ran,
            Some(render_margin_row_backward_interm_execution(
                stats.backward_interm_execution,
            )),
        );
        crate::flight::note(
            MARGIN_ROW_ROOT_PRECISION_EXECUTION_METHOD,
            crate::flight::FlightStatus::Ran,
            Some(render_margin_row_root_precision_execution(
                stats.root_precision_execution,
            )),
        );
        crate::flight::note(
            MARGIN_ROW_ROOT_ALPHA_EXECUTION_METHOD,
            crate::flight::FlightStatus::Ran,
            Some(render_margin_row_root_alpha_execution(
                outcome,
                &stats.root_alpha_execution,
            )),
        );
    }
}

/// #mr-close-telemetry: the flight receipt is versioned, carries its phase /
/// outcome / source, prints the thinnest certificate as exact bits AND a
/// round-trip decimal, and keeps a fixed schema (every per-kind key present
/// on an empty tally, so "zero" and "absent" stay distinguishable).
#[cfg(test)]
#[test]
fn margin_row_close_integrity_receipt_is_exact_and_versioned() {
    use ny_propagate::margin_row::close_integrity::{CertKind, CloseIntegrityEvidence};

    let empty = render_margin_row_close_integrity(
        "terminal",
        "unknown",
        "no_stats",
        &CloseIntegrityEvidence::EMPTY,
    );
    assert!(
        empty.starts_with(
            "schema=ny-cli/margin-row-close-integrity/v1 phase=terminal outcome=unknown \
             source=no_stats close_schema=1 closes=0 min_close_bits=none min_close=na \
             min_close_kind=none closes_lt_1e-6=0 closes_lt_1e-8=0 closes_lt_1e-10=0 \
             min_root_dj_bits=none min_root_dj=na root_closed=0 epoch_closes=0 \
             frozen_dual_closes=0 nonpositive=0 root_n=0 "
        ),
        "{empty}"
    );
    assert!(
        empty.ends_with(
            " epoch_root_n=0 epoch_root_min_bits=none epoch_root_min=na \
             epoch_root_lt_1e-6=0 epoch_root_lt_1e-8=0 epoch_root_lt_1e-10=0"
        ),
        "{empty}"
    );
    assert!(!empty.contains('\n'));

    let thin = 3.0e-9_f64;
    let mut evidence = CloseIntegrityEvidence::default();
    evidence.note(CertKind::Root, 0.125);
    evidence.note(CertKind::Lift, thin);
    evidence.note(CertKind::Eval, 2.0e-4);
    let rendered = render_margin_row_close_integrity("terminal", "unsat", "stats", &evidence);
    assert!(
        rendered.starts_with(&format!(
            "schema=ny-cli/margin-row-close-integrity/v1 phase=terminal outcome=unsat \
             source=stats close_schema=1 closes=3 min_close_bits={} min_close=3e-9 \
             min_close_kind=lift closes_lt_1e-6=1 closes_lt_1e-8=1 closes_lt_1e-10=0 \
             min_root_dj_bits={} min_root_dj=1.25e-1 root_closed=1 ",
            thin.to_bits(),
            0.125f64.to_bits(),
        )),
        "{rendered}"
    );
    assert!(rendered.contains(&format!(
        " lift_n=1 lift_min_bits={} lift_min=3e-9 lift_lt_1e-6=1 lift_lt_1e-8=1 \
         lift_lt_1e-10=0 ",
        thin.to_bits()
    )));
    assert!(rendered.contains(" eval_n=1 "), "{rendered}");
    assert!(rendered.contains(" eval_min=2e-4 "), "{rendered}");
    let snapshot = render_margin_row_close_integrity("snapshot", "running", "live", &evidence);
    assert!(snapshot.starts_with(
        "schema=ny-cli/margin-row-close-integrity/v1 phase=snapshot outcome=running \
         source=live close_schema=1 closes=3 "
    ));
}

#[cfg(test)]
#[test]
fn margin_row_frontier_execution_receipt_is_exact_and_versioned() {
    let evidence = FrontierExecutionEvidence {
        route: FrontierRoute::Joint,
        configured: 1,
        contextual: 8,
        steps: 4,
        counterfactual_differences: 3,
        max_open: 8,
        max_batch: 1,
    };
    assert_eq!(
        render_margin_row_frontier_execution("unknown", evidence),
        "schema=ny-cli/margin-row-frontier-execution/v1 phase=terminal route=joint \
         outcome=unknown tree_started=true configured=1 contextual=8 steps=4 \
         counterfactual_differences=3 max_open=8 max_batch=1",
    );
}

#[cfg(test)]
#[test]
fn margin_row_width_execution_receipt_is_exact_and_versioned() {
    let evidence = WidthExecutionEvidence {
        configured_k_head: 16,
        contextual_k_head: 8,
        configured_k_trunk: 12,
        contextual_k_trunk: 8,
        adaptive_enabled: true,
        contextual_adaptive: false,
        tier0_exact: 0,
        effective_k_head_max: 32,
        effective_k_trunk_max: 32,
        effective_contextual_k_head_max: 8,
        effective_contextual_k_trunk_max: 8,
        outer_head_selection_calls: 4,
        outer_head_available_max: 40,
        outer_head_selected_total: 64,
        outer_head_binding_scored_calls: 3,
        outer_trunk_selection_calls: 4,
        outer_trunk_available_max: 48,
        outer_trunk_selected_total: 48,
        outer_trunk_binding_scored_calls: 2,
        nested_head_selection_calls: 5,
        nested_trunk_selection_calls: 6,
        configured_adaptive_threshold_opportunities: 7,
        configured_adaptive_escalations: 2,
        contextual_adaptive_threshold_opportunities: 0,
        contextual_adaptive_escalations: 0,
        policy_mismatch: false,
    };
    assert_eq!(
        render_margin_row_width_execution("unknown", evidence),
        "schema=ny-margin-row-width-execution/v2 phase=terminal outcome=unknown \
         configured_k_head=16 contextual_k_head=8 configured_k_trunk=12 \
         contextual_k_trunk=8 adaptive_enabled=true contextual_adaptive=false \
         tier0_exact=0 effective_k_head_max=32 effective_k_trunk_max=32 \
         effective_contextual_k_head_max=8 effective_contextual_k_trunk_max=8 \
         outer_head_selection_calls=4 outer_head_available_max=40 \
         outer_head_selected_total=64 outer_head_binding_scored_calls=3 \
         outer_trunk_selection_calls=4 outer_trunk_available_max=48 \
         outer_trunk_selected_total=48 outer_trunk_binding_scored_calls=2 \
         nested_head_selection_calls=5 nested_trunk_selection_calls=6 \
         configured_adaptive_threshold_opportunities=7 \
         configured_adaptive_escalations=2 \
         contextual_adaptive_threshold_opportunities=0 \
         contextual_adaptive_escalations=0 policy_mismatch=false",
    );
}

#[cfg(test)]
#[test]
fn margin_row_epoch_execution_receipt_is_exact_and_versioned() {
    let evidence = EpochExecutionEvidence {
        policy: ny_propagate::margin_row::EpochRuntimePolicy {
            depth: 3,
            max_attempts: 4,
            tier0_exact: 8,
            tier0_universe: 128,
            retain_per_layer: 256,
            parent_topk: 16,
            debug_trace: true,
        },
        tree_entries: 1,
        configured_depth_admissions: 2,
        configured_depth_nested_with_stats: 3,
        adaptive_depth_admissions: 4,
        attempt_cap_decisions: 5,
        attempt_cap_counterfactual_differences: 6,
        attempt_cap_refusals: 7,
        generation_refusals: 8,
        deadline_refusals: 9,
        attempts_started: 10,
        attempt_failures: 11,
        attempts_closed: 12,
        outer_full_attempts: 13,
        outer_incremental_attempts: 14,
        nested_full_attempts: 15,
        nested_incremental_attempts: 16,
        full_rebuilds_completed: 17,
        full_rebuild_refusals: 18,
        parent_refreshes_started: 19,
        parent_refreshes_completed: 20,
        parent_refreshes_accepted: 21,
        parent_refresh_refusals: 22,
        parent_refresh_columns: 23,
        parent_refresh_tightened: 24,
        parent_refresh_stabilized: 25,
        nested_started: 26,
        nested_runs_with_stats: 27,
        nested_unsat: 28,
        nested_unknown: 29,
        nested_errors: 30,
        closures: 31,
        debug_emissions: 32,
        outer_tier0_calls: 33,
        outer_tier0_scored_calls: 34,
        outer_tier0_exact_binding_scored_calls: 35,
        outer_universe_counterfactual_scored_calls: 36,
        outer_retain_counterfactual_scored_calls: 37,
        nested_full_tier0_calls: 38,
        nested_incremental_tier0_calls: 39,
        policy_mismatch: true,
    };
    assert_eq!(
        render_margin_row_epoch_execution("unknown", FrontierRoute::Classwise, &evidence),
        "schema=ny-cli/margin-row-epoch-execution/v1 phase=terminal route=classwise \
         outcome=unknown configured_depth=3 configured_attempts=4 configured_tier0=8 \
         configured_universe=128 configured_retain=256 configured_parent_topk=16 \
         configured_debug=true policy_mismatch=true tree_entries=1 \
         configured_depth_admissions=2 configured_depth_nested_with_stats=3 \
         adaptive_depth_admissions=4 attempt_cap_decisions=5 \
         attempt_cap_counterfactual_differences=6 attempt_cap_refusals=7 \
         generation_refusals=8 deadline_refusals=9 attempts_started=10 \
         attempt_failures=11 attempts_closed=12 outer_full_attempts=13 \
         outer_incremental_attempts=14 nested_full_attempts=15 \
         nested_incremental_attempts=16 full_rebuilds_completed=17 \
         full_rebuild_refusals=18 parent_refreshes_started=19 \
         parent_refreshes_completed=20 parent_refreshes_accepted=21 \
         parent_refresh_refusals=22 parent_refresh_columns=23 \
         parent_refresh_tightened=24 parent_refresh_stabilized=25 nested_started=26 \
         nested_runs_with_stats=27 nested_unsat=28 nested_unknown=29 nested_errors=30 \
         closures=31 debug_emissions=32 outer_tier0_calls=33 \
         outer_tier0_scored_calls=34 outer_tier0_exact_binding_scored_calls=35 \
         outer_universe_counterfactual_scored_calls=36 \
         outer_retain_counterfactual_scored_calls=37 nested_full_tier0_calls=38 \
         nested_incremental_tier0_calls=39",
    );
}

#[cfg(test)]
#[test]
fn margin_row_domain_alpha_execution_receipt_is_exact_and_versioned() {
    let evidence = PolicyExecutionEvidence {
        beta_per_column: false,
        beta_columns: 8,
        column_retire: false,
        parallel_prebuild: false,
        clip_head: false,
        domain_alpha_enabled: true,
        domain_alpha_iters: 4,
        domain_alpha_eta_bits: 0.125f64.to_bits(),
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
        domain_alpha_gate_evals: 1,
        domain_alpha_movable_evals: 2,
        domain_alpha_movable_slots: 3,
        domain_alpha_warm_applied: 4,
        domain_alpha_ascent_calls: 5,
        domain_alpha_trials_started: 6,
        domain_alpha_trial_passes_completed: 7,
        domain_alpha_trial_pass_failures: 8,
        domain_alpha_finite_trial_scores: 9,
        domain_alpha_trials_accepted: 10,
        domain_alpha_carried_entries: 11,
        domain_alpha_iters_counterfactual_domains: 12,
        domain_alpha_eta_counterfactual_scored_domains: 13,
        domain_alpha_outer_ascent_calls: 14,
        domain_alpha_outer_finite_trial_scores: 15,
        domain_alpha_outer_iters_counterfactual_domains: 16,
        domain_alpha_outer_eta_counterfactual_scored_domains: 17,
        policy_mismatch: true,
    };
    assert_eq!(
        render_margin_row_domain_alpha_execution("unknown", FrontierRoute::Classwise, evidence,),
        "schema=ny-cli/margin-row-domain-alpha-execution/v1 phase=terminal \
         route=classwise outcome=unknown configured_enabled=true configured_iters=4 \
         configured_eta_bits=4593671619917905920 policy_mismatch=true gate_evals=1 \
         movable_domains=2 movable_slots=3 warm_entries_applied=4 ascent_domains=5 \
         outer_ascent_domains=14 proposals=6 trial_passes_completed=7 \
         trial_pass_failures=8 finite_trial_scores=9 accepted_trials=10 \
         carried_entries=11 iters_counterfactual_domains=12 \
         eta_counterfactual_scored_domains=13 outer_finite_trial_scores=15 \
         outer_iters_counterfactual_domains=16 \
         outer_eta_counterfactual_scored_domains=17",
    );
}

#[cfg(test)]
#[test]
fn margin_row_policy_execution_receipt_is_exact_and_versioned() {
    let evidence = PolicyExecutionEvidence {
        beta_per_column: true,
        beta_columns: 4,
        column_retire: true,
        parallel_prebuild: false,
        clip_head: false,
        domain_alpha_enabled: false,
        domain_alpha_iters: 2,
        domain_alpha_eta_bits: 0.25f64.to_bits(),
        clip_head_evals: 0,
        clip_head_moved: 0,
        clip_head_closes: 0,
        clip_head_refused: 0,
        beta_per_column_evals: 3,
        beta_columns_binding_evals: 2,
        beta_columns_max_candidates: 7,
        newly_retired_columns: 5,
        inherited_retirement_evals: 4,
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
    };
    assert_eq!(
        render_margin_row_policy_execution("unsat", evidence),
        "schema=ny-cli/margin-row-policy-execution/v1 phase=terminal outcome=unsat \
         beta_per_column=true beta_columns=4 column_retire=true parallel_prebuild=false \
         beta_per_column_evals=3 beta_columns_binding_evals=2 \
         beta_columns_max_candidates=7 newly_retired_columns=5 \
         inherited_retirement_evals=4 parallel_prebuild_batches=0 \
         parallel_prebuild_domains=0 policy_mismatch=false",
    );
}

#[cfg(test)]
#[test]
fn margin_row_backward_interm_execution_receipt_is_exact_and_versioned() {
    let evidence = BackwardIntermExecutionEvidence {
        enabled: true,
        // MEASURED phase work, distinct from the 20.0 s BUDGET below. This
        // receipt test pins the two as different numbers deliberately: reading
        // `configured_secs` as a cost is the mistake `#rebake`'s round-1 pricing
        // made.
        measured_work_secs: 12.5,
        configured_secs: 20.0,
        configured_chunk: 256,
        configured_chunk_secs: 2.0,
        effective_chunk_secs: 2.0,
        configured_topk: 1024,
        eligible_layers: 4,
        suppressed_setup_layers: 0,
        layers_started: 3,
        chunks_completed: 7,
        columns_completed: 769,
        deadline_capped: false,
        budget_exhausted: true,
        chunk_secs_capped: false,
        chunk_counterfactual_chunks: 2,
        chunk_ceiling_binding_chunks: 1,
        fixed_width_chunks: 0,
        adaptive_width_changes: 3,
        topk_binding_layers: 1,
        policy_mismatch: false,
    };
    assert_eq!(
        render_margin_row_backward_interm_execution(evidence),
        "schema=ny-cli/margin-row-backward-interm-execution/v1 phase=terminal \
         outcome=exhausted enabled=true configured_secs=20 configured_chunk=256 \
         configured_chunk_secs=2 effective_chunk_secs=2 configured_topk=1024 \
         eligible_layers=4 suppressed_setup_layers=0 layers_started=3 \
         chunks_completed=7 columns_completed=769 \
         deadline_capped=false budget_exhausted=true chunk_secs_capped=false \
         chunk_counterfactual_chunks=2 chunk_ceiling_binding_chunks=1 \
         fixed_width_chunks=0 adaptive_width_changes=3 \
         topk_binding_layers=1 policy_mismatch=false",
    );
}

#[cfg(test)]
#[test]
fn margin_row_root_precision_execution_receipt_is_exact_and_versioned() {
    let evidence = RootPrecisionExecutionEvidence {
        configured_root_f32: true,
        configured_tableau_f32: false,
        outward_eligible: true,
        eligible_conv_ops: 7,
        root_f32_conv_ops: 7,
        root_f64_conv_ops: 0,
        eligible_store_ops: 19,
        tableau_f32_store_ops: 0,
        tableau_f64_store_ops: 19,
        tableau_refused: false,
        policy_mismatch: false,
    };
    assert_eq!(
        render_margin_row_root_precision_execution(evidence),
        "schema=ny-cli/margin-row-root-precision-execution/v1 phase=terminal \
         outcome=completed configured_root_f32=true configured_tableau_f32=false \
         outward_eligible=true eligible_conv_ops=7 root_f32_conv_ops=7 \
         root_f64_conv_ops=0 eligible_store_ops=19 tableau_f32_store_ops=0 \
         tableau_f64_store_ops=19 tableau_refused=false policy_mismatch=false",
    );
}

#[cfg(test)]
#[test]
fn margin_row_root_alpha_execution_receipt_is_exact_and_versioned() {
    let evidence = RootAlphaExecutionEvidence {
        alpha_enabled: true,
        alpha_iters: 16,
        alpha_secs: 12.5,
        alpha_lr: 0.05,
        alpha_cols: 4,
        alpha_guard: true,
        rebake_enabled: true,
        rebake_rounds: 3,
        rebake_secs: 45.0,
        eligible: true,
        standalone_attempted: 1,
        standalone_completed: 1,
        standalone_effective_secs: 10.0,
        standalone_deadline_capped: true,
        standalone_iters_cap_exhausted: true,
        standalone_fail_classes: 9,
        standalone_iters_attempted: 16,
        standalone_candidate_steps: 16,
        standalone_adam_steps: 16,
        standalone_passes: 17,
        standalone_max_steer_cols: 4,
        standalone_frozen_improvements: 3,
        standalone_accepted: true,
        standalone_moved_neurons: 21,
        standalone_guard_opportunities: 1,
        standalone_guard_baseline_evals: 1,
        standalone_guard_candidate_evals: 1,
        standalone_frozen_before_bits: Some((-1.25f64).to_bits()),
        standalone_frozen_best_bits: Some((-1.0f64).to_bits()),
        standalone_published_before_bits: Some((-0.5f64).to_bits()),
        standalone_published_after_bits: Some(0.25f64.to_bits()),
        rebake_attempted: 1,
        rebake_completed: 1,
        rebake_effective_secs: 30.0,
        rebake_deadline_capped: true,
        rebake_budget_exhausted: true,
        rebake_entry_unbaked_alpha: true,
        rebake_baseline_evals: 1,
        rebake_rounds_started: 2,
        rebake_alpha_completed: 2,
        rebake_alpha_iters_attempted: 12,
        rebake_alpha_candidate_steps: 11,
        rebake_alpha_adam_steps: 11,
        rebake_alpha_passes: 13,
        rebake_alpha_max_fail_classes: 7,
        rebake_alpha_max_steer_cols: 4,
        rebake_alpha_moved_neurons: 17,
        rebake_rebuilds_completed: 2,
        rebake_rounds_scored: 2,
        rebake_rounds_accepted: 1,
        rebake_affordability_refusals: 1,
        rebake_published_before_bits: Some((-1.0f64).to_bits()),
        rebake_published_after_bits: Some((-0.5f64).to_bits()),
        // #spec-alpha: policy echo AND yield. Set explicitly so this exact
        // receipt test bites if either half stops being rendered -- the yield
        // is the half that makes an armed sweep interpretable.
        spec_alpha_enabled: true,
        spec_alpha_width: 4,
        spec_alpha_cands: 5,
        spec_alpha_rounds: 2,
        spec_alpha_entries: 4_096,
        spec_alpha_ran: true,
        spec_alpha_rounds_run: 2,
        spec_alpha_passes: 3,
        spec_alpha_entries_committed: 1_234,
        spec_alpha_classes_selected: 4,
        spec_alpha_incumbent_grads_ok: 4,
        spec_alpha_classes_improved: 3,
        spec_alpha_distinct_winners: 2,
        spec_alpha_incumbent_grad_improvements: 2,
        spec_alpha_closed_extra: 1,
        spec_alpha_truncated: 1,
        spec_alpha_root_before_bits: Some((-0.25f64).to_bits()),
        spec_alpha_root_after_bits: Some((-0.125f64).to_bits()),
        spec_alpha_best_gain_bits: Some(0.125f64.to_bits()),
        ..RootAlphaExecutionEvidence::default()
    };
    assert_eq!(
        render_margin_row_root_alpha_execution("unknown", &evidence),
        "schema=ny-cli/margin-row-root-alpha-execution/v2 phase=terminal outcome=unknown \
         alpha_enabled=true alpha_iters=16 alpha_secs_bits=4623226492472524800 \
         alpha_lr_bits=4587366580439587226 alpha_cols=4 alpha_guard=true \
         rebake_enabled=true rebake_rounds=3 rebake_secs_bits=4631530004285489152 \
         eligible=true standalone_suppressed=0 standalone_attempted=1 \
         standalone_completed=1 standalone_effective_secs_bits=4621819117588971520 \
         standalone_deadline_capped=true standalone_budget_exhausted=false \
         standalone_iters_cap_exhausted=true standalone_fail_classes=9 \
         standalone_iters_attempted=16 standalone_candidate_steps=16 \
         standalone_legacy_steps=0 standalone_adam_steps=16 standalone_passes=17 \
         standalone_max_steer_cols=4 standalone_frozen_improvements=3 \
         standalone_accepted=true standalone_moved_neurons=21 \
         standalone_guard_opportunities=1 standalone_guard_baseline_evals=1 \
         standalone_guard_candidate_evals=1 standalone_guard_reversions=0 \
         standalone_frozen_before_bits=13831680355561635840 \
         standalone_frozen_best_bits=13830554455654793216 \
         standalone_published_before_bits=13826050856027422720 \
         standalone_published_after_bits=4598175219545276416 \
         rebake_suppressed=0 rebake_attempted=1 rebake_completed=1 \
         rebake_effective_secs_bits=4629137466983448576 rebake_deadline_capped=true \
         rebake_budget_exhausted=true rebake_round_cap_exhausted=false \
         rebake_entry_unbaked_alpha=true rebake_baseline_evals=1 \
         rebake_rounds_started=2 rebake_alpha_completed=2 \
         rebake_alpha_iters_attempted=12 rebake_alpha_candidate_steps=11 \
         rebake_alpha_legacy_steps=0 rebake_alpha_adam_steps=11 \
         rebake_alpha_passes=13 rebake_alpha_max_fail_classes=7 \
         rebake_alpha_max_steer_cols=4 rebake_alpha_moved_neurons=17 \
         rebake_rebuilds_completed=2 rebake_rounds_scored=2 \
         rebake_rounds_accepted=1 rebake_fixed_point_stops=0 \
         rebake_affordability_refusals=1 rebake_aborts=0 \
         rebake_published_before_bits=13830554455654793216 \
         rebake_published_after_bits=13826050856027422720 \
         spec_alpha_enabled=true spec_alpha_width=4 spec_alpha_cands=5 \
         spec_alpha_rounds=2 spec_alpha_entries=4096 spec_alpha_ran=true \
         spec_alpha_rounds_run=2 spec_alpha_passes=3 \
         spec_alpha_entries_committed=1234 spec_alpha_classes_selected=4 \
         spec_alpha_incumbent_grads_ok=4 \
         spec_alpha_classes_improved=3 spec_alpha_distinct_winners=2 \
         spec_alpha_incumbent_grad_improvements=2 spec_alpha_closed_extra=1 \
         spec_alpha_truncated=1 spec_alpha_root_before_bits=13821547256400052224 \
         spec_alpha_root_after_bits=13817043656772681728 spec_alpha_best_gain_bits=4593671619917905920 \
         policy_mismatch=false",
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MarginRowReserveRoute {
    NotApplicable,
    Fixed,
    AdaptivePreserved,
    AdaptiveReleasedAlphaBetaTier,
}

/// Immutable entry-time policy for the margin-row lane scheduler.
///
/// The `configured_*` fields are the effective values consumed by this run.
/// Their `contextual_*` peers are the exact lower-layer values recorded in
/// [`ny_levers::Resolved::lower_value`]. Keeping both answers in one `Copy`
/// value lets execution evidence compare a treatment with the policy it
/// replaced without reopening either process environment or preset files.
///
/// Fractions are stored as IEEE-754 bits. Besides making this type `Eq`, that
/// preserves the exact f64 widened from a typed f32 preset and distinguishes
/// every receipt identity admitted by the declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MarginRowScheduleRuntimePolicy {
    pub(crate) configured_reserve_secs: u64,
    pub(crate) contextual_reserve_secs: u64,
    pub(crate) configured_adaptive_reserve: bool,
    pub(crate) contextual_adaptive_reserve: bool,
    pub(crate) configured_reserve_max_frac_bits: Option<u64>,
    pub(crate) contextual_reserve_max_frac_bits: Option<u64>,
    pub(crate) configured_adaptive_release_frac_bits: u64,
    pub(crate) contextual_adaptive_release_frac_bits: u64,
    pub(crate) configured_concurrent: bool,
    pub(crate) contextual_concurrent: bool,
    pub(crate) configured_early_exit: bool,
    pub(crate) contextual_early_exit: bool,
    pub(crate) configured_verifier_threads: Option<usize>,
    pub(crate) contextual_verifier_threads: Option<usize>,
    pub(crate) configured_concurrent_threads: Option<usize>,
    pub(crate) contextual_concurrent_threads: Option<usize>,
}

impl Default for MarginRowScheduleRuntimePolicy {
    fn default() -> Self {
        let release_bits = DEFAULT_ADAPTIVE_RELEASE_FRAC.to_bits();
        Self {
            configured_reserve_secs: 45,
            contextual_reserve_secs: 45,
            configured_adaptive_reserve: false,
            contextual_adaptive_reserve: false,
            configured_reserve_max_frac_bits: None,
            contextual_reserve_max_frac_bits: None,
            configured_adaptive_release_frac_bits: release_bits,
            contextual_adaptive_release_frac_bits: release_bits,
            configured_concurrent: false,
            contextual_concurrent: false,
            configured_early_exit: true,
            contextual_early_exit: true,
            configured_verifier_threads: None,
            contextual_verifier_threads: None,
            configured_concurrent_threads: None,
            contextual_concurrent_threads: None,
        }
    }
}

/// Result of the one process-wide attempt to configure Rayon for the internal
/// verifier. `actual_threads` is populated only when a global registry is
/// known to exist; observing it on the untouched route would initialize Rayon
/// and change the historical no-cap path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GlobalRayonOutcome {
    pub(crate) route: GlobalRayonRoute,
    pub(crate) requested_threads: Option<usize>,
    pub(crate) actual_threads: Option<usize>,
    pub(crate) host_threads: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GlobalRayonRoute {
    NotRequested,
    Applied,
    AlreadyInitialized,
}

impl GlobalRayonRoute {
    pub(crate) const fn stable_token(self) -> &'static str {
        match self {
            Self::NotRequested => "not_requested",
            Self::Applied => "applied",
            Self::AlreadyInitialized => "already_initialized",
        }
    }
}

impl GlobalRayonOutcome {
    pub(crate) const NOT_REQUESTED: Self = Self {
        route: GlobalRayonRoute::NotRequested,
        requested_threads: None,
        actual_threads: None,
        host_threads: None,
    };
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GlobalRayonBuildFailure {
    pub(crate) requested_threads: usize,
    pub(crate) host_threads: Option<usize>,
    pub(crate) error: String,
}

impl MarginRowScheduleRuntimePolicy {
    fn exact_u64(
        decl: &'static ny_levers::LeverDecl,
        value: &ny_levers::LeverValue,
    ) -> Result<u64, String> {
        match value {
            ny_levers::LeverValue::U64(value) => Ok(*value),
            other => Err(format!("{} resolved to non-u64 value {other:?}", decl.name)),
        }
    }

    fn exact_bool(
        decl: &'static ny_levers::LeverDecl,
        value: &ny_levers::LeverValue,
    ) -> Result<bool, String> {
        match value {
            ny_levers::LeverValue::Bool(value) => Ok(*value),
            other => Err(format!(
                "{} resolved to non-boolean value {other:?}",
                decl.name
            )),
        }
    }

    fn exact_open_fraction_bits(
        decl: &'static ny_levers::LeverDecl,
        value: &ny_levers::LeverValue,
    ) -> Result<u64, String> {
        match value {
            ny_levers::LeverValue::F64(value)
                if value.is_finite() && *value > 0.0 && *value < 1.0 =>
            {
                Ok(value.to_bits())
            }
            other => Err(format!(
                "{} resolved outside the finite open unit interval: {other:?}",
                decl.name
            )),
        }
    }

    fn exact_optional_open_fraction_bits(
        decl: &'static ny_levers::LeverDecl,
        value: &ny_levers::LeverValue,
    ) -> Result<Option<u64>, String> {
        match value {
            ny_levers::LeverValue::Unset => Ok(None),
            _ => Self::exact_open_fraction_bits(decl, value).map(Some),
        }
    }

    fn exact_optional_usize(
        decl: &'static ny_levers::LeverDecl,
        value: &ny_levers::LeverValue,
    ) -> Result<Option<usize>, String> {
        match value {
            ny_levers::LeverValue::Unset => Ok(None),
            ny_levers::LeverValue::U64(value) => usize::try_from(*value)
                .map(Some)
                .map_err(|_| format!("{} resolved outside platform usize: {value}", decl.name)),
            other => Err(format!(
                "{} resolved to non-optional-usize value {other:?}",
                decl.name
            )),
        }
    }

    fn from_resolved(
        reserve_secs: &ny_levers::Resolved,
        adaptive_reserve: &ny_levers::Resolved,
        reserve_max_frac: &ny_levers::Resolved,
        adaptive_release_frac: &ny_levers::Resolved,
        concurrent: &ny_levers::Resolved,
        early_exit: &ny_levers::Resolved,
        verifier_threads: &ny_levers::Resolved,
        concurrent_threads: &ny_levers::Resolved,
    ) -> Result<Self, String> {
        Ok(Self {
            configured_reserve_secs: Self::exact_u64(reserve_secs.decl, &reserve_secs.value)?,
            contextual_reserve_secs: Self::exact_u64(reserve_secs.decl, &reserve_secs.lower_value)?,
            configured_adaptive_reserve: Self::exact_bool(
                adaptive_reserve.decl,
                &adaptive_reserve.value,
            )?,
            contextual_adaptive_reserve: Self::exact_bool(
                adaptive_reserve.decl,
                &adaptive_reserve.lower_value,
            )?,
            configured_reserve_max_frac_bits: Self::exact_optional_open_fraction_bits(
                reserve_max_frac.decl,
                &reserve_max_frac.value,
            )?,
            contextual_reserve_max_frac_bits: Self::exact_optional_open_fraction_bits(
                reserve_max_frac.decl,
                &reserve_max_frac.lower_value,
            )?,
            configured_adaptive_release_frac_bits: Self::exact_open_fraction_bits(
                adaptive_release_frac.decl,
                &adaptive_release_frac.value,
            )?,
            contextual_adaptive_release_frac_bits: Self::exact_open_fraction_bits(
                adaptive_release_frac.decl,
                &adaptive_release_frac.lower_value,
            )?,
            configured_concurrent: Self::exact_bool(concurrent.decl, &concurrent.value)?,
            contextual_concurrent: Self::exact_bool(concurrent.decl, &concurrent.lower_value)?,
            configured_early_exit: Self::exact_bool(early_exit.decl, &early_exit.value)?,
            contextual_early_exit: Self::exact_bool(early_exit.decl, &early_exit.lower_value)?,
            configured_verifier_threads: Self::exact_optional_usize(
                verifier_threads.decl,
                &verifier_threads.value,
            )?,
            contextual_verifier_threads: Self::exact_optional_usize(
                verifier_threads.decl,
                &verifier_threads.lower_value,
            )?,
            configured_concurrent_threads: Self::exact_optional_usize(
                concurrent_threads.decl,
                &concurrent_threads.value,
            )?,
            contextual_concurrent_threads: Self::exact_optional_usize(
                concurrent_threads.decl,
                &concurrent_threads.lower_value,
            )?,
        })
    }

    /// Resolve the scheduler from the run's receipt-owning frozen lever set.
    /// Missing or ill-typed entries fail closed instead of consulting a second
    /// authority.
    pub(crate) fn from_levers(levers: &ny_levers::LeverSet) -> Result<Self, String> {
        use ny_levers::decls::margin_row::{
            MARGIN_ROW_ADAPTIVE_RELEASE_FRAC, MARGIN_ROW_ADAPTIVE_RESERVE, MARGIN_ROW_CONCURRENT,
            MARGIN_ROW_CONCURRENT_THREADS, MARGIN_ROW_EARLY_EXIT, MARGIN_ROW_RESERVE_MAX_FRAC,
            MARGIN_ROW_RESERVE_SECS, VERIFIER_THREADS,
        };

        let resolved = |decl: &'static ny_levers::LeverDecl| {
            levers
                .resolved(decl)
                .ok_or_else(|| format!("{} is missing from the frozen LeverSet", decl.name))
        };
        Self::from_resolved(
            resolved(&MARGIN_ROW_RESERVE_SECS)?,
            resolved(&MARGIN_ROW_ADAPTIVE_RESERVE)?,
            resolved(&MARGIN_ROW_RESERVE_MAX_FRAC)?,
            resolved(&MARGIN_ROW_ADAPTIVE_RELEASE_FRAC)?,
            resolved(&MARGIN_ROW_CONCURRENT)?,
            resolved(&MARGIN_ROW_EARLY_EXIT)?,
            resolved(&VERIFIER_THREADS)?,
            resolved(&MARGIN_ROW_CONCURRENT_THREADS)?,
        )
    }

    /// Resolve the historical process-over-preset layers once for compatibility
    /// and research entry points. Scored VNN-COMP execution uses
    /// [`Self::from_levers`] instead.
    pub(crate) fn from_layered_process(preset: Option<&Path>) -> Self {
        use ny_levers::decls::margin_row::{
            MARGIN_ROW_ADAPTIVE_RELEASE_FRAC, MARGIN_ROW_ADAPTIVE_RESERVE, MARGIN_ROW_CONCURRENT,
            MARGIN_ROW_CONCURRENT_THREADS, MARGIN_ROW_EARLY_EXIT, MARGIN_ROW_RESERVE_MAX_FRAC,
            MARGIN_ROW_RESERVE_SECS, VERIFIER_THREADS,
        };

        let loaded = preset.and_then(|path| crate::preset::load_preset(path).ok());
        let margin_row = loaded.as_ref().map(|config| &config.margin_row);

        // Compatibility callers still resolve process-over-preset layering at
        // entry, but the declaration registry remains the only environment
        // parser. Filter typed floats exactly as the historical preset readers
        // did before handing their admissible values to the shared resolver.
        let reserve_secs = ny_levers::read_over_config(
            &MARGIN_ROW_RESERVE_SECS,
            margin_row
                .and_then(|row| row.reserve_secs)
                .map(ny_levers::LeverValue::U64),
        )
        .expect("typed reserve seconds satisfy the scheduler declaration");
        let adaptive_reserve = ny_levers::read_over_config(
            &MARGIN_ROW_ADAPTIVE_RESERVE,
            margin_row
                .and_then(|row| row.adaptive_reserve)
                .map(ny_levers::LeverValue::Bool),
        )
        .expect("typed adaptive-reserve gate satisfies the scheduler declaration");
        let reserve_max_frac = ny_levers::read_over_config(
            &MARGIN_ROW_RESERVE_MAX_FRAC,
            margin_row
                .and_then(|row| row.reserve_max_frac)
                .map(f64::from)
                .and_then(valid_reserve_fraction)
                .map(ny_levers::LeverValue::F64),
        )
        .expect("filtered reserve ceiling satisfies the scheduler declaration");
        let adaptive_release_frac = ny_levers::read_over_config(
            &MARGIN_ROW_ADAPTIVE_RELEASE_FRAC,
            margin_row
                .and_then(|row| row.release_frac)
                .map(f64::from)
                .and_then(valid_reserve_fraction)
                .map(ny_levers::LeverValue::F64),
        )
        .expect("filtered release threshold satisfies the scheduler declaration");
        let concurrent = ny_levers::read_over_config(
            &MARGIN_ROW_CONCURRENT,
            margin_row
                .and_then(|row| row.concurrent)
                .map(ny_levers::LeverValue::Bool),
        )
        .expect("typed concurrent gate satisfies the scheduler declaration");
        let early_exit = ny_levers::read_over_config(
            &MARGIN_ROW_EARLY_EXIT,
            margin_row
                .and_then(|row| row.early_exit)
                .map(ny_levers::LeverValue::Bool),
        )
        .expect("typed early-exit gate satisfies the scheduler declaration");
        let verifier_threads = ny_levers::read_over_config(
            &VERIFIER_THREADS,
            margin_row
                .and_then(|row| row.verifier_threads)
                .map(|value| ny_levers::LeverValue::U64(value as u64)),
        )
        .expect("typed verifier-thread cap satisfies the scheduler declaration");
        let concurrent_threads = ny_levers::read_over_config(
            &MARGIN_ROW_CONCURRENT_THREADS,
            margin_row
                .and_then(|row| row.concurrent_threads)
                .map(|value| ny_levers::LeverValue::U64(value as u64)),
        )
        .expect("typed concurrent-thread cap satisfies the scheduler declaration");

        Self::from_resolved(
            &reserve_secs,
            &adaptive_reserve,
            &reserve_max_frac,
            &adaptive_release_frac,
            &concurrent,
            &early_exit,
            &verifier_threads,
            &concurrent_threads,
        )
        .expect("scheduler declarations preserve their exact runtime value types")
    }

    pub(crate) fn configured_reserve_max_frac(self) -> Option<f64> {
        self.configured_reserve_max_frac_bits.map(f64::from_bits)
    }

    pub(crate) fn configured_adaptive_release_frac(self) -> f64 {
        f64::from_bits(self.configured_adaptive_release_frac_bits)
    }

    pub(crate) const fn early_exit_armed(self) -> bool {
        self.configured_early_exit
    }

    pub(crate) fn verifier_thread_cap(self) -> Option<std::num::NonZeroUsize> {
        self.configured_verifier_threads
            .and_then(std::num::NonZeroUsize::new)
    }

    pub(crate) const fn concurrent_threads_override(self) -> Option<usize> {
        self.configured_concurrent_threads
    }
}

/// Apply the frozen verifier cap to Rayon's process-global registry exactly
/// once. The caller must invoke this on the main thread before any optional
/// scored Rayon work.
pub(crate) fn configure_global_rayon(
    policy: MarginRowScheduleRuntimePolicy,
) -> Result<GlobalRayonOutcome, GlobalRayonBuildFailure> {
    let host_threads = std::thread::available_parallelism()
        .ok()
        .map(std::num::NonZeroUsize::get);
    let Some(requested) = policy
        .verifier_thread_cap()
        .map(std::num::NonZeroUsize::get)
    else {
        return Ok(GlobalRayonOutcome {
            host_threads,
            ..GlobalRayonOutcome::NOT_REQUESTED
        });
    };

    match rayon::ThreadPoolBuilder::new()
        .num_threads(requested)
        .build_global()
    {
        Ok(()) => Ok(GlobalRayonOutcome {
            route: GlobalRayonRoute::Applied,
            requested_threads: Some(requested),
            actual_threads: Some(rayon::current_num_threads()),
            host_threads,
        }),
        Err(error) if std::error::Error::source(&error).is_none() => Ok(GlobalRayonOutcome {
            route: GlobalRayonRoute::AlreadyInitialized,
            requested_threads: Some(requested),
            actual_threads: Some(rayon::current_num_threads()),
            host_threads,
        }),
        Err(error) => Err(GlobalRayonBuildFailure {
            requested_threads: requested,
            host_threads,
            error: error.to_string(),
        }),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MarginRowReserveDecision {
    /// Configured reserve before the adaptive policy.
    pub(crate) configured_secs: u64,
    /// Reserve actually subtracted from the internal verifier.
    pub(crate) reserve_secs: u64,
    pub(crate) route: MarginRowReserveRoute,
}

impl MarginRowReserveDecision {
    /// The sealed adaptive route hands the internal verifier the full window;
    /// that is the only production route authorized to arm sparse root CROWN.
    pub(crate) fn enables_scored_sparse_crown(self) -> bool {
        matches!(
            self.route,
            MarginRowReserveRoute::AdaptiveReleasedAlphaBetaTier
        )
    }

    /// Apply the dark budget-proportional ceiling to an already-resolved
    /// decision. `route` is deliberately UNCHANGED: the sealed adaptive route
    /// is the only thing authorized to arm scored sparse root CROWN
    /// ([`Self::enables_scored_sparse_crown`]), and a scheduling ceiling must
    /// not smuggle that in.
    #[allow(dead_code)]
    pub(crate) fn capped_to_internal_budget(
        mut self,
        internal_budget_secs: u64,
        preset: Option<&Path>,
    ) -> Self {
        self.apply_internal_budget_cap(internal_budget_secs, margin_row_reserve_max_frac(preset))
    }

    /// Apply the cap from the immutable receipt-owned scheduler policy.
    pub(crate) fn capped_to_internal_budget_with_policy(
        mut self,
        internal_budget_secs: u64,
        policy: MarginRowScheduleRuntimePolicy,
    ) -> Self {
        self.apply_internal_budget_cap(internal_budget_secs, policy.configured_reserve_max_frac())
    }

    fn apply_internal_budget_cap(
        &mut self,
        internal_budget_secs: u64,
        max_fraction: Option<f64>,
    ) -> Self {
        // Nothing to cap: skip the preset load entirely. `min` against any
        // ceiling would return 0 anyway, so this is byte-identical and keeps
        // the non-reserving categories off the filesystem.
        if self.reserve_secs == 0 {
            return *self;
        }
        self.reserve_secs =
            capped_reserve_secs(self.reserve_secs, internal_budget_secs, max_fraction);
        *self
    }
}

/// Dark opt-in: cap the margin-row reserve at this fraction of the INTERNAL
/// budget. Absent or invalid ⇒ no ceiling ⇒ byte-identical to the shipped
/// fixed-seconds policy.
#[cfg(test)]
pub(crate) const RESERVE_MAX_FRAC_ENV: &str = "NY_MARGIN_ROW_RESERVE_MAX_FRAC";

/// Accept a ceiling fraction only if it is finite and strictly inside
/// `(0, 1)`. `1.0` is rejected on purpose — it is a no-op ceiling and
/// accepting it would invite "0 means release" confusion with
/// `NY_MARGIN_ROW_RESERVE_SECS`, which is the existing full-release knob.
fn valid_reserve_fraction(fraction: f64) -> Option<f64> {
    (fraction.is_finite() && fraction > 0.0 && fraction < 1.0).then_some(fraction)
}

/// Parse the ceiling fraction from its environment form. Everything the
/// parser or [`valid_reserve_fraction`] rejects (absent, malformed, `0`,
/// `>= 1`, non-finite) declines the ceiling and keeps the shipped policy.
#[cfg(test)]
fn reserve_max_fraction(raw: Option<&str>) -> Option<f64> {
    raw.and_then(|value| value.parse::<f64>().ok())
        .and_then(valid_reserve_fraction)
}

/// Resolve the reserve ceiling: `NY_MARGIN_ROW_RESERVE_MAX_FRAC` > per-category
/// `margin_row.reserve_max_frac` in the preset > no ceiling.
///
/// The environment wins wherever it is PRESENT, and a present-but-declined
/// value resolves to "no ceiling" WITHOUT consulting the preset. That is
/// deliberate: it keeps `NY_MARGIN_ROW_RESERVE_MAX_FRAC=0` usable as an exact
/// kill switch for a ceiling a shipped preset asked for, which is what the
/// dark-gate discipline requires of a scheduling knob. (It differs from
/// `margin_row_reserve_secs`, where an unparseable env value falls through —
/// there the fallback is a nonzero default, so falling through is the
/// conservative direction; here the fallback IS the shipped policy.)
///
/// Sound by construction: same argument as [`capped_reserve_secs`] — this only
/// schedules wall time between two independently sound lanes, so it is
/// verdict-neutral and can at worst make a lane fail to prove.
#[allow(dead_code)]
pub(crate) fn margin_row_reserve_max_frac(preset: Option<&Path>) -> Option<f64> {
    MarginRowScheduleRuntimePolicy::from_layered_process(preset).configured_reserve_max_frac()
}

/// The pure half of [`margin_row_reserve_max_frac`], kept separate so the
/// precedence rules are testable without mutating process-global environment.
#[cfg(test)]
fn resolve_reserve_max_frac(env_raw: Option<&str>, typed: Option<f32>) -> Option<f64> {
    match env_raw {
        // PRESENT (even if declined) ⇒ the environment decides, full stop.
        Some(raw) => reserve_max_fraction(Some(raw)),
        None => typed.and_then(|fraction| valid_reserve_fraction(f64::from(fraction))),
    }
}

/// Clamp a fixed-seconds reserve to a fraction of the internal budget.
///
/// WHY (measured 2026-07-26, cifar100_2024 `prop_idx_9502_sidx_7197` — a
/// winnable-60 row — via `NY_PHASE_TELEMETRY=1`): the reserve is a FIXED
/// number of seconds, so its share of the budget GROWS as the budget shrinks.
///
/// | scored budget | internal tier | ledger after reserve | effective BaB |
/// |---|---|---|---|
/// | 100 s | 95 s | 46 s | **34.2 s** |
/// | 200 s | 190 s | 141 s | **90.0 s** |
///
/// Doubling the scored budget yields 2.63x the BaB time, because the fixed
/// 45 s is 47% of the 95 s internal tier but only 24% of the 190 s one. That
/// non-proportionality — not per-domain GPU throughput — is why a scored 100 s
/// run behaves nothing like the first 100 s of a 200 s run. On that same run
/// the lane the reserve paid for reported `inline worker exceeded its 51.1s
/// hard slice cap; abandoning the detached worker`: 45 s bought nothing, while
/// the verifier that might have closed the row was left unable to finish its
/// root bootstrap.
///
/// Scope: 53 of the 60 cifar100 rows alpha-beta-CROWN proves and NY does not
/// pay the full fixed reserve.
///
/// The second half of this paragraph used to read "the shipped
/// `adaptive_reserve` release covers exactly 7 hard-coded rows (and none of
/// NY's 41 banked unsats)". That has been FALSE since #6569bfdc replaced the
/// seven-filename allowlist with pure budget arithmetic
/// ([`adaptive_release_target`]). On the scored cifar100 path every input is
/// fixed — reserve 45 s ([`margin_row_reserve_secs`], the preset sets no
/// `reserve_secs`), internal tier 95 s (`100 - max(5, 100/20)`), release_frac
/// 0.40 ([`DEFAULT_ADAPTIVE_RELEASE_FRAC`], not overridden) — so `45 >= 38`
/// holds for EVERY structurally admissible row, not seven. The banked-unsat
/// carve-out is therefore no longer true by construction; it can only be
/// established by measurement, and #6569bfdc's own message asks for exactly
/// that A/B ("must be A/B'd on the GB10 before the next bank of that
/// category") before the category is banked again.
///
/// Sound by construction: this only schedules wall time between two lanes that
/// are each independently sound. Shrinking the margin-row slice can only make
/// that lane fail to prove (fail-closed, verdict-neutral); it can never
/// produce a wrong verdict. Same argument as the surrounding reserve logic.
fn capped_reserve_secs(
    reserve_secs: u64,
    internal_budget_secs: u64,
    max_fraction: Option<f64>,
) -> u64 {
    let Some(fraction) = max_fraction else {
        return reserve_secs;
    };
    // f64 -> u64 via a saturating floor: the product is bounded by
    // `internal_budget_secs` for any accepted fraction (< 1.0), so this cannot
    // exceed u64, but keep the conversion total rather than relying on that.
    let ceiling = (internal_budget_secs as f64 * fraction).floor();
    let ceiling = if ceiling.is_finite() && ceiling >= 0.0 {
        ceiling as u64
    } else {
        return reserve_secs;
    };
    reserve_secs.min(ceiling)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
enum AdaptiveReserveGateEnv<'a> {
    Absent,
    Unicode(&'a str),
    NonUnicode,
}

#[cfg(test)]
fn adaptive_reserve_gate_env(raw: Option<&std::ffi::OsStr>) -> AdaptiveReserveGateEnv<'_> {
    match raw {
        None => AdaptiveReserveGateEnv::Absent,
        Some(value) => value.to_str().map_or(
            AdaptiveReserveGateEnv::NonUnicode,
            AdaptiveReserveGateEnv::Unicode,
        ),
    }
}

/// Resolve the typed production policy plus an exact sealed A/B override.
/// Only raw Unicode `"1"` enables and `"0"` disables. Any other present value
/// fails closed to the fixed-reserve policy, even if the preset is typed on.
#[cfg(test)]
fn adaptive_reserve_enabled(typed_enabled: bool, gate: AdaptiveReserveGateEnv<'_>) -> bool {
    match gate {
        AdaptiveReserveGateEnv::Absent => typed_enabled,
        AdaptiveReserveGateEnv::Unicode("1") => true,
        AdaptiveReserveGateEnv::Unicode(_) | AdaptiveReserveGateEnv::NonUnicode => false,
    }
}

/// Dark opt-in: the reserve share of the INTERNAL budget at or above which the
/// adaptive route releases the reserve entirely. Absent or invalid ⇒
/// [`DEFAULT_ADAPTIVE_RELEASE_FRAC`].
#[cfg(test)]
pub(crate) const RELEASE_FRAC_ENV: &str = "NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC";

/// Release when the fixed reserve would take >= 40% of the internal budget.
///
/// Chosen to sit between the two measured regimes rather than to hit a row
/// list: the shipped 45 s reserve is 47.4% of the 95 s internal tier at a
/// scored 100 s (release) and 23.7% of the 190 s tier at a scored 200 s
/// (preserve). See [`capped_reserve_secs`] for the measurement.
const DEFAULT_ADAPTIVE_RELEASE_FRAC: f64 = 0.40;

/// Resolve the release threshold: `NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC` >
/// per-category `margin_row.release_frac` > [`DEFAULT_ADAPTIVE_RELEASE_FRAC`].
///
/// Shares [`valid_reserve_fraction`]'s admission rule, so a malformed or
/// out-of-range value in either source falls back to the shipped default
/// rather than silently disabling or universalizing the release.
#[allow(dead_code)]
pub(crate) fn margin_row_release_frac(preset: Option<&Path>) -> f64 {
    MarginRowScheduleRuntimePolicy::from_layered_process(preset).configured_adaptive_release_frac()
}

/// The pure half of [`margin_row_release_frac`].
#[cfg(test)]
fn resolve_release_frac(env_raw: Option<&str>, typed: Option<f32>) -> f64 {
    env_raw
        .and_then(|raw| raw.parse::<f64>().ok())
        .and_then(valid_reserve_fraction)
        .or_else(|| typed.map(f64::from).and_then(valid_reserve_fraction))
        .unwrap_or(DEFAULT_ADAPTIVE_RELEASE_FRAC)
}

/// Should the adaptive route hand the internal verifier its whole window?
///
/// STRUCTURAL, not identity-based. The predicate is pure budget arithmetic:
/// release exactly when the configured fixed reserve would consume at least
/// `release_frac` of the internal budget. It therefore generalizes to every
/// category, model, and instance — including ones never measured — and depends
/// on nothing but the two numbers the scheduler already has.
///
/// This replaces a hardcoded seven-filename CIFAR100 allowlist. That list
/// encoded a real measurement (those rows paid a reserve that bought nothing
/// while the verifier missed its root bootstrap) but keyed it to the identity
/// of specific public benchmark files, so it could not fire on an unseen
/// instance with the same pathology and would silently stop firing if a file
/// were renamed. The share-of-budget condition is the actual mechanism behind
/// that measurement, expressed directly.
///
/// Sound by construction: this only schedules wall time between two
/// independently sound lanes, so it is verdict-neutral — at worst a lane fails
/// to prove. Same argument as [`capped_reserve_secs`].
fn adaptive_release_target(
    configured_secs: u64,
    internal_budget_secs: u64,
    release_frac: f64,
) -> bool {
    if internal_budget_secs == 0 {
        // No budget to take a share of; keep the established fixed lane.
        return false;
    }
    configured_secs as f64 >= internal_budget_secs as f64 * release_frac
}

fn reserve_policy(
    configured_secs: u64,
    adaptive_enabled: bool,
    release_target: bool,
) -> MarginRowReserveDecision {
    if configured_secs == 0 {
        return MarginRowReserveDecision {
            configured_secs,
            reserve_secs: 0,
            route: MarginRowReserveRoute::NotApplicable,
        };
    }
    if !adaptive_enabled {
        return MarginRowReserveDecision {
            configured_secs,
            reserve_secs: configured_secs,
            route: MarginRowReserveRoute::Fixed,
        };
    }
    if release_target {
        MarginRowReserveDecision {
            configured_secs,
            reserve_secs: 0,
            route: MarginRowReserveRoute::AdaptiveReleasedAlphaBetaTier,
        }
    } else {
        MarginRowReserveDecision {
            configured_secs,
            reserve_secs: configured_secs,
            route: MarginRowReserveRoute::AdaptivePreserved,
        }
    }
}

/// Does this [`TwinSpec`] satisfy the head contract the margin-row COMPILER
/// enforces (`ny_propagate::margin_row::net::TwinNet::compile`)?
///
/// #twinwall-reserve-admission. [`extract_twin_spec`] admits any net built from
/// the supported op set; the compiler additionally requires the tail to be
/// `... trunk-ReLU ... -> Gemm1 -> ReLU -> Gemm2` with exactly two Gemms. Those
/// two admissions were not the same predicate, so a net inside the op set but
/// outside the head shape ARMED the reserve and then declined at run time —
/// MEASURED on malbeware (`Conv -> ReLU -> Flatten -> Gemm`, ONE Gemm): the
/// reserve held back 45 s of a 95 s internal budget and the lane refused in
/// 0.0 s with `expected 2 gemms, got 1`, so the internal verifier ran on 46 s
/// instead of 95 s.
///
/// This mirrors the compiler's checks EXACTLY and allocates nothing: the op
/// index of a `TwinOpSpec` equals the op index of the `TwinOp` it compiles to
/// (the compiler pushes one op per spec op, in order), and op `k` publishes
/// tensor id `k + 1`. Fail-closed: any deviation returns `false`, which returns
/// the reserve to the internal verifier — never the other way round.
fn twin_spec_compiles_to_margin_row_head(spec: &TwinSpec) -> bool {
    twin_spec_has_margin_row_head(spec) && spec.root_tableau_within_cap()
}

fn twin_spec_compiles_with_root_policy(spec: &TwinSpec, policy: &MarginRowRuntimePolicy) -> bool {
    use ny_propagate::margin_row::root_box;
    if !twin_spec_has_margin_row_head(spec) {
        return false;
    }
    let conflicts = root_box::policy_conflicts(policy);
    let source = root_box::select_source(spec, policy.root_precision.box_source, conflicts);
    if !spec.root_tableau_within_cap() && !root_box::wide_consumers_supported(policy) {
        eprintln!("[root-source-admission] phase=cli_root requested={} effective={} status=refused reason=unpriced_wide_consumers working_set_bytes={}",
            policy.root_precision.box_source.token(), source.token(), root_box::wide_working_set_bytes());
        return false;
    }
    match root_box::source_admission(spec, source, conflicts) {
        Ok(_) => true,
        Err(error) => {
            if !spec.root_tableau_within_cap() {
                eprintln!("[root-source-admission] phase=cli_root requested={} effective={} status=refused working_set_bytes={} reason={error}",
                    policy.root_precision.box_source.token(), source.token(), root_box::wide_working_set_bytes());
            }
            false
        }
    }
}

/// A concurrent whole-lane admission must reserve its worst-case tree, not
/// merely hope that the new wide root closes before scoring allocates. Inline
/// root closure still uses the actual-survivor guard inside the lane.
fn twin_spec_compiles_with_tree_policy(
    spec: &TwinSpec,
    policy: &MarginRowRuntimePolicy,
    objectives: usize,
    frontier: usize,
    lru_cap: usize,
) -> bool {
    use ny_propagate::margin_row::root_box;
    if !twin_spec_compiles_with_root_policy(spec, policy) {
        return false;
    }
    if spec.root_tableau_within_cap() {
        return true; // preserve every established narrow admission
    }
    let Some(root) = root_box::source_census(spec) else {
        return false;
    };
    let Some(tree) = root_box::consumer_census(root, policy, objectives, frontier, lru_cap) else {
        return false;
    };
    let required = root.root_peak_bytes.max(tree.peak_bytes);
    eprintln!("[root-source-admission] phase=concurrent_preflight requested={} effective=patch_backward working_set_bytes={} root_peak_bytes={} tree_columns={} objectives={} workers={} cache_bytes={} tree_peak_bytes={} cap={}",
        policy.root_precision.box_source.token(), root_box::wide_working_set_bytes(), root.root_peak_bytes,
        tree.columns, objectives, tree.workers, tree.cache_bytes, tree.peak_bytes, root_box::ROOT_SOURCE_BYTES_CAP);
    required <= root_box::ROOT_SOURCE_BYTES_CAP && root_box::wide_memory_admission(required).is_ok()
}

fn twin_spec_has_margin_row_head(spec: &TwinSpec) -> bool {
    let gemms: Vec<usize> = spec
        .ops
        .iter()
        .enumerate()
        .filter(|(_, op)| matches!(op, TwinOpSpec::Gemm { .. }))
        .map(|(idx, _)| idx)
        .collect();
    let [i_gemm1, i_gemm2] = gemms[..] else {
        return false;
    };
    // `... -> Gemm1 -> Relu -> Gemm2` with Gemm2 last.
    if i_gemm2 != spec.ops.len() - 1 || i_gemm1 + 2 != i_gemm2 {
        return false;
    }
    // Head wiring: the head ReLU consumes Gemm1's output and Gemm2 consumes
    // the head ReLU's.
    match (spec.ops.get(i_gemm1 + 1), spec.ops.get(i_gemm2)) {
        (Some(TwinOpSpec::Relu { input: r_in }), Some(TwinOpSpec::Gemm { input: g2_in, .. }))
            if *r_in == i_gemm1 + 1 && *g2_in == i_gemm1 + 2 => {}
        _ => return false,
    }
    // #cct-margin-row-admission: the compiler refuses a BN fold on the FINAL
    // Gemm (its consumers read it as exact) ...
    if spec.ops[i_gemm2].is_folded_gemm() {
        return false;
    }
    // The compiler also requires at least one TRUNK ReLU (a ReLU seen before
    // the first Gemm); without one the lane has nothing to branch on.
    spec.ops[..i_gemm1]
        .iter()
        .any(|op| matches!(op, TwinOpSpec::Relu { .. }))
}

/// #admission-census: the margin-row lane's STRUCTURAL admission, for the
/// admission census (`ny vnncomp census`) — the conjunction of checks that
/// EVERY entry into the lane makes before any lane code runs, so a row this
/// refuses is a row on which neither the inline lane
/// (`try_margin_row_unsat`) nor the concurrent lane
/// (`spawn_concurrent_lane_impl`) ever starts, and on which no consumer
/// inside the lane (the frozen-dual tier, the wave tree, the
/// `backward_interm` window) can engage, whatever the router routes:
///
/// * the lane switch ([`ny_propagate::margin_row::margin_row_bab_enabled`]);
/// * the twin-family extraction ([`extract_twin_spec`]) — both entries;
/// * the head the COMPILER requires (`twin_spec_has_margin_row_head`) —
///   the concurrent entry checks it at admission and the inline lane's
///   `TwinNet::compile` refuses without it before the tree starts;
/// * the robustness property ([`parse_vnnlib_robustness`]) — both entries;
/// * the input width (`lo.len() == spec.n_in`) — both entries.
///
/// It deliberately OMITS every policy-dependent check (the root-tableau cap,
/// the wide-source admission, the device premise, the deadline): each of
/// those can admit under some policy, so leaving them out keeps this a
/// NECESSARY condition — it may admit a row the lane later declines (the
/// census then moats it needlessly), never refuse a row the lane runs on.
/// The same functions, not copies of them; a source ratchet
/// (`census_margin_row_probe_is_the_lanes_own_admission`) pins that both
/// entries still call each one. Split in a MODEL half (this function: the
/// switch, the extraction and the head, returning the net's input width) and
/// a PROPERTY half ([`lane_property_admission`]) so the census pays one
/// extraction per model, not one per row; a row is admitted iff both halves
/// admit it.
pub(crate) fn lane_model_admission(onnx: &Path) -> Result<usize, &'static str> {
    if !ny_propagate::margin_row::margin_row_bab_enabled() {
        return Err("lane-disabled");
    }
    let Some(spec) = extract_twin_spec(onnx) else {
        return Err("model-outside-twin-family");
    };
    if !twin_spec_has_margin_row_head(&spec) {
        return Err("no-compiled-head");
    }
    Ok(spec.n_in)
}

/// #admission-census: the PROPERTY half of the margin-row lane's structural
/// admission ([`lane_model_admission`]): the robustness parse and the input
/// width against the model half's `n_in`.
pub(crate) fn lane_property_admission(vnnlib: &Path, n_in: usize) -> Result<(), &'static str> {
    let Some((lo, _, _, _)) = parse_vnnlib_robustness(vnnlib) else {
        return Err("property-not-robustness");
    };
    if lo.len() != n_in {
        return Err("input-width");
    }
    Ok(())
}

#[cfg(test)]
mod source_admission_tests {
    use super::*;

    #[test]
    fn wide_source_admission_reaches_policy_aware_cli_and_keeps_product_refusals() {
        let _env = ny_test_utils::env::lock_env_shared();
        let n = 40_000;
        let spec = TwinSpec {
            n_in: n,
            ops: vec![
                TwinOpSpec::Relu { input: 0 },
                TwinOpSpec::Gemm {
                    input: 1,
                    weight: vec![0.0; n * 2],
                    bias: vec![0.0; 2],
                    bias_err: vec![],
                    weight_rel_err: 0.0,
                    shape: (2, n),
                },
                TwinOpSpec::Relu { input: 2 },
                TwinOpSpec::Gemm {
                    input: 3,
                    weight: vec![0.0; 4],
                    bias: vec![0.0; 2],
                    bias_err: vec![],
                    weight_rel_err: 0.0,
                    shape: (2, 2),
                },
            ],
        };
        assert!(!twin_spec_compiles_to_margin_row_head(&spec));
        let policy = MarginRowRuntimePolicy::default();
        assert!(twin_spec_compiles_with_root_policy(&spec, &policy));
        assert!(twin_spec_compiles_with_tree_policy(&spec, &policy, 1, 1, 1));
        assert!(!twin_spec_compiles_with_tree_policy(
            &spec, &policy, 200, 8, 64
        ));
        assert!(!twin_spec_compiles_with_tree_policy(
            &spec,
            &policy,
            usize::MAX,
            1,
            1
        ));
        let mut f32 = policy;
        f32.root_precision.root_f32 = true;
        assert!(!twin_spec_compiles_with_root_policy(&spec, &f32));
        let mut widened = policy;
        widened.parallel_prebuild = true;
        assert!(!twin_spec_compiles_with_root_policy(&spec, &widened));
    }
}

/// Resolve the per-instance reserve after structural admission.
///
/// A typed category preset or exact `NY_MARGIN_ROW_ADAPTIVE_RESERVE=1` enables
/// the adaptive route; within it, [`adaptive_release_target`] decides purely
/// from budget arithmetic. All parse, extraction, and environment uncertainty
/// retains the fixed policy. No verifier or margin-row bound/verdict code is
/// changed by this decision.
#[allow(dead_code)]
pub(crate) fn margin_row_reserve_decision(
    onnx: &Path,
    vnnlib: &Path,
    preset: Option<&Path>,
    internal_budget_secs: u64,
) -> MarginRowReserveDecision {
    margin_row_reserve_decision_with_policy(
        onnx,
        vnnlib,
        internal_budget_secs,
        MarginRowScheduleRuntimePolicy::from_layered_process(preset),
    )
}

/// Resolve the per-instance reserve from an immutable, receipt-owned scheduler
/// policy. This is the scored-use seam: it never consults process environment
/// or reloads the category preset.
pub(crate) fn margin_row_reserve_decision_with_policy(
    onnx: &Path,
    vnnlib: &Path,
    internal_budget_secs: u64,
    policy: MarginRowScheduleRuntimePolicy,
) -> MarginRowReserveDecision {
    let configured_secs = policy.configured_reserve_secs;
    // No reserve unless this category opted in: skip the expensive
    // structural probe entirely when there is nothing to reserve.
    //
    // #twinwall-reserve-admission: the probe must accept EXACTLY what the
    // margin-row compiler accepts. Extraction alone is not enough — see
    // [`twin_spec_compiles_to_margin_row_head`].
    if configured_secs == 0
        || !ny_propagate::margin_row::margin_row_bab_enabled()
        || parse_vnnlib_robustness(vnnlib).is_none()
        || !extract_twin_spec(onnx).is_some_and(|spec| twin_spec_compiles_to_margin_row_head(&spec))
    {
        return MarginRowReserveDecision {
            configured_secs,
            reserve_secs: 0,
            route: MarginRowReserveRoute::NotApplicable,
        };
    }
    reserve_policy(
        configured_secs,
        policy.configured_adaptive_reserve,
        adaptive_release_target(
            configured_secs,
            internal_budget_secs,
            policy.configured_adaptive_release_frac(),
        ),
    )
}

/// Seconds held back from the internal verifier for the margin-row lane.
///
/// **DEFAULT 45** — main's shipped value. The historical CIFAR100 +23
/// bookkeeping appears to have used that default, but the TinyImageNet +67
/// commit bodies explicitly record a non-default 82 s reserve. A later
/// blocked-backward 0/70 scratch rerun also used 82 s. Neither TinyImageNet run
/// has sealed provenance, so neither establishes a production reserve.
/// Resolution order:
/// `NY_MARGIN_ROW_RESERVE_SECS` > per-category `margin_row.reserve_secs` in the
/// preset > 45.
///
/// I briefly defaulted this to 0 on the theory that a 45 s reserve forfeits
/// solves — 28 on `sat_relu` and 10 on `cifar100_2024` by my reading of the
/// scorecards. That was WRONG on both counts and is corrected here:
///   * `sat_relu` never took the reserve at all: `extract_twin_spec` rejects
///     those nets (structural mismatch), so `margin_row_reserve_decision`
///     returns `NotApplicable` and the budget was never touched. 0 at risk,
///     not 28.
///   * Historical CIFAR100 bookkeeping records +23 certified UNSAT and appears
///     to use the 45 s default, but it is not a sealed A/B and therefore does
///     not establish the reserve's net production effect.
///
#[allow(dead_code)]
pub(crate) fn margin_row_reserve_secs(preset: Option<&Path>) -> u64 {
    MarginRowScheduleRuntimePolicy::from_layered_process(preset).configured_reserve_secs
}

fn log_classwise_stats(stats: &BabStats) {
    if stats.class_runs.is_empty() {
        return;
    }
    let order: Vec<usize> = stats.class_runs.iter().map(|run| run.class).collect();
    eprintln!(
        "margin-row classwise: order={order:?}, root_closed={}, completed={}/{}, stop={}",
        stats.root_closed_classes,
        stats.class_runs.iter().filter(|run| run.verified).count(),
        stats.tree_classes.len(),
        stats.stop,
    );
    for run in &stats.class_runs {
        eprintln!(
            "margin-row classwise class {}: verified={}, root_bound={:.6}, expansions={}, \
             domains={}, maxDepth={}, dips={}, epochs={}/{}, ledger={:?}, elapsed={:.3}s, stop={}",
            run.class,
            run.verified,
            run.root_bound,
            run.expansions,
            run.domains_created,
            run.max_depth,
            run.mono_raw_dips,
            run.epochs_closed,
            run.epochs_attempted,
            run.ledger_ok,
            run.elapsed_secs,
            run.stop,
        );
    }
}

#[derive(Debug, PartialEq, Eq)]
enum DetachedTake<T> {
    Finished(T),
    StillRunning,
    Failed,
}

/// Run `work` on a detached worker and bound how long the caller waits at
/// `cap` (#twinwall watchdog-kill class, banked 99ed4d42).
///
/// Distinguishes a cap expiry from a completed/failed worker.  The abandoned
/// worker keeps running until its own cooperative deadline fires or process
/// teardown; callers must not start another graph-heavy tail in that state.
///
/// WHY: `run_margin_row_lane`'s internal deadline checks are cooperative and
/// can be arbitrarily coarse between expensive tableau builds (measured on
/// metaroom: 113.1s against a 45s slice; one run overran until the process
/// watchdog killed it at budget+grace — the error-scoring risk class). The
/// only sound external enforcement is to abandon the work at the boundary.
fn run_capped_detached<T: Send + 'static>(
    cap: Duration,
    label: &'static str,
    work: impl FnOnce() -> T + Send + 'static,
) -> DetachedTake<T> {
    // Capacity-1 channel: the send never blocks, and a receiver gone after
    // the cap makes it fail — the expected abandoned-worker case.
    let (tx, rx) = std::sync::mpsc::sync_channel::<T>(1);
    let handle = match std::thread::Builder::new()
        .name(format!("margin-row-{label}"))
        .spawn(move || {
            // Evaluate the consumed FnOnce before publishing. Its captured
            // graph state is consequently dropped before `recv_timeout`
            // observes completion; joining below only waits for this bounded
            // send/epilogue, not for another proof phase.
            let out = work();
            let _ = tx.send(out);
        }) {
        Ok(handle) => handle,
        Err(_) => {
            eprintln!(
                "margin-row BaB: could not spawn {label} worker; skipping lane (fail-closed)"
            );
            return DetachedTake::Failed;
        }
    };
    match rx.recv_timeout(cap) {
        Ok(out) => match handle.join() {
            Ok(()) => DetachedTake::Finished(out),
            Err(_) => {
                eprintln!(
                    "margin-row BaB: {label} worker panicked after producing a result; \
                     discarding it fail-closed"
                );
                DetachedTake::Failed
            }
        },
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            eprintln!(
                "margin-row BaB: {label} worker exceeded its {:.1}s hard slice cap; abandoning \
                 the detached worker and returning in-budget (fail-closed, verdict-neutral)",
                cap.as_secs_f64()
            );
            // #mr-census: the abandoned worker's own `[mr-census]` line prints
            // only if it returns before the process exits, which it may never
            // do. Print a SNAPSHOT from the abandoning thread, unconditionally,
            // so no stop path of the lane goes without a census line.
            eprintln!(
                "{}",
                ny_propagate::margin_row::census::snapshot_line(&format!(
                    "abandoned_at_slice_cap_{label}"
                ))
            );
            // ...plus the per-mechanism stop receipts the worker's `Drop`
            // would have printed (`[mr-errlane] done` carries the B3 audit).
            for extra in ny_propagate::margin_row::census::stop_lines() {
                eprintln!("{extra}");
            }
            // #margin-row-profile-on-abandon: dump the phase profile HERE.
            //
            // `prof::dump()` normally runs at the end of `lane_impl`, but an
            // abandoned worker never returns, so the profiler could not report
            // on the one run you actually need to profile -- the overrun. The
            // counters are process-global atomics, so the abandoning thread can
            // read them while the detached worker is still going; the numbers
            // are a snapshot at the cap, which is exactly the question ("where
            // did the slice go?").
            if ny_propagate::margin_row::prof::enabled() {
                eprint!(
                    "margin-row profile AT SLICE CAP ({label}, {:.1}s, worker still running):\n{}",
                    cap.as_secs_f64(),
                    ny_propagate::margin_row::prof::dump()
                );
            }
            // Dropping the JoinHandle detaches the still-running worker.
            DetachedTake::StillRunning
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            let _ = handle.join();
            eprintln!(
                "margin-row BaB: {label} worker exited without a result (panicked); fail-closed"
            );
            DetachedTake::Failed
        }
    }
}

/// #soundness-spine M2d G3: the lane's hand-parsed objective rows `(t, adv)`
/// against the SAT GATE's own spec evaluator (`property_violated_f64` over the
/// full VNN-LIB parse). Evaluated at the spec box's own centre and at
/// synthetic output vectors that put every margin clearly on each side of
/// zero — adversarial classes above and below the true class, and every class
/// the lane does NOT treat as adversarial lifted above it — so a misread
/// class, a dropped disjunct or a flipped comparison disagrees in sign. No
/// network evaluation is involved: this checks the TRANSLATION, which every
/// forward self-check passes by construction. Microseconds.
pub(crate) fn margin_row_spec_agreement(
    vnnlib: &Path,
    t: usize,
    adv: &[usize],
) -> (ny_propagate::certify::GuardReport, String) {
    use ny_propagate::certify::GuardReport;
    let spec = match ny_onnx::vnnlib::load_vnnlib(vnnlib) {
        Ok(spec) => spec,
        Err(_) => return (GuardReport::new(), "g3=skipped(parse)".to_string()),
    };
    margin_row_spec_agreement_owned(&spec, t, adv)
}

/// Same ordinary G3 audit over the immutable original parse captured by a
/// source-authenticated producer; no filesystem reparse can replace it.
pub(crate) fn margin_row_spec_agreement_owned(
    spec: &ny_onnx::vnnlib::VnnLibSpec,
    t: usize,
    adv: &[usize],
) -> (ny_propagate::certify::GuardReport, String) {
    use ny_propagate::certify::{GuardId, GuardReport};
    use ny_propagate::guards::spec_agreement::SpecAgreementAudit;
    // Per-clause input boxes: the evaluator would gate each disjunct on its
    // own box, which the lane never parses (it refuses such specs). Nothing
    // to compare against here, so the guard is skipped, never tripped.
    if !spec.per_clause_input_bounds.is_empty() {
        return (
            GuardReport::new(),
            "g3=skipped(per_clause_boxes)".to_string(),
        );
    }
    let n_out = spec.num_outputs;
    if t >= n_out || adv.iter().any(|&j| j >= n_out || j == t) {
        let mut report = GuardReport::new();
        report.trip(
            GuardId::SpecAgreement,
            format!("lane classes (t={t}, adv={adv:?}) are not outputs of the spec (n={n_out})"),
        );
        return (report, "g3=ran g3_class_range=refused".to_string());
    }
    let x: Vec<f64> = spec
        .input_bounds
        .iter()
        .map(|&(lo, hi)| f64::midpoint(lo, hi))
        .collect();
    let mut audit = SpecAgreementAudit::default();
    let mut vectors: Vec<Vec<f64>> = Vec::new();
    let base = {
        let mut y = vec![0.0f64; n_out];
        y[t] = 1.0;
        y
    };
    vectors.push(base.clone());
    for &j in adv {
        for d in [0.5f64, -0.5] {
            let mut y = base.clone();
            y[j] = 1.0 + d;
            vectors.push(y);
        }
    }
    for c in (0..n_out).filter(|c| *c != t && !adv.contains(c)) {
        let mut y = base.clone();
        y[c] = 1.5;
        vectors.push(y);
    }
    for (i, y) in vectors.iter().enumerate() {
        let engine_min = adv
            .iter()
            .map(|&j| y[t] - y[j])
            .fold(f64::INFINITY, f64::min);
        let violated = super::vnncomp::property_violated_f64(spec, &x, y);
        audit.note(i, engine_min, violated, 1.5);
    }
    let fragment = format!("g3=ran {}", audit.fragment());
    (audit.report(), fragment)
}

/// #soundness-spine M2b: conversion from the TwinNet margin-row lane's `Unsat`
/// to a published `VnncompResult::Unsat`. G3 ratifies the token (downgrade
/// only); a refused or non-instance token yields `None`, and the caller then
/// publishes nothing from the lane.
pub(crate) fn margin_row_receipt(
    token: ny_propagate::certify::UnsatToken,
    vnnlib: &Path,
    t: usize,
    adv: &[usize],
) -> Option<UnsatReceipt> {
    let (report, fragment) = margin_row_spec_agreement(vnnlib, t, adv);
    match ny_propagate::certify::ratify(token, &report) {
        Ok(token) => {
            let receipt = UnsatReceipt::from_token(token);
            eprintln!(
                "[g3] lane=margin_row {fragment} receipt={}",
                if receipt.is_some() {
                    "ok"
                } else {
                    "not_publishable"
                }
            );
            receipt
        }
        Err(refusal) => {
            eprintln!("[g3] lane=margin_row REFUSED {refusal} {fragment}");
            None
        }
    }
}

#[cfg(test)]
mod spec_agreement_tests {
    use super::margin_row_spec_agreement;

    /// A standard robustness property: true class 0, adversarial {1, 2}.
    const ROBUSTNESS: &str = "\
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (>= X_1 0.0))
(assert (<= X_1 1.0))
(assert (or
    (and (>= Y_1 Y_0))
    (and (>= Y_2 Y_0))
))
";

    fn spec_file() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("prop.vnnlib");
        std::fs::write(&path, ROBUSTNESS).expect("write property");
        (dir, path)
    }

    /// G3 agrees with the lane's own (correct) parse, at every probe vector.
    #[test]
    fn g3_margin_row_spec_agreement_accepts_the_lanes_own_parse() {
        let (_dir, path) = spec_file();
        let (report, fragment) = margin_row_spec_agreement(&path, 0, &[1, 2]);
        assert!(report.is_clear(), "{fragment}");
        assert!(fragment.starts_with("g3=ran"), "{fragment}");
        assert!(fragment.contains("g3_disagree=0"), "{fragment}");
    }

    /// A dropped disjunct, a wrong true class and an out-of-range class all
    /// trip G3 — the misreads a forward self-check cannot see.
    #[test]
    fn g3_margin_row_spec_agreement_trips_on_a_misread_objective() {
        let (_dir, path) = spec_file();
        for (t, adv) in [(0usize, vec![1usize]), (1, vec![0, 2]), (0, vec![1, 7])] {
            let (report, fragment) = margin_row_spec_agreement(&path, t, &adv);
            assert!(
                !report.is_clear(),
                "misread (t={t}, adv={adv:?}) passed: {fragment}"
            );
        }
    }

    /// A spec the evaluator cannot parse skips the guard (downgrade-only: a
    /// guard that cannot run never trips).
    #[test]
    fn g3_margin_row_spec_agreement_skips_an_unreadable_spec() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (report, fragment) =
            margin_row_spec_agreement(&dir.path().join("missing.vnnlib"), 0, &[1]);
        assert!(report.is_clear());
        assert_eq!(fragment, "g3=skipped(parse)");
    }
}

pub(crate) enum MarginRowTry {
    Verdict(VnncompResult),
    FinishedWithoutVerdict,
    StillRunning,
}

/// Strictly-additive vnncomp hook: only ever turns unknown/timeout into
/// `Unsat`. The caller can distinguish an abandoned worker so it never starts
/// another graph-heavy optional tail while that worker remains live.
#[expect(
    clippy::large_types_passed_by_value,
    reason = "the runtime policy is MOVED into the lane it configures; a reference here would only add a clone at the move"
)]
pub(crate) fn try_margin_row_unsat(
    onnx: &Path,
    vnnlib: &Path,
    instance_deadline: Option<Instant>,
    runtime_policy: MarginRowRuntimePolicy,
) -> MarginRowTry {
    if !ny_propagate::margin_row::margin_row_bab_enabled() {
        return MarginRowTry::FinishedWithoutVerdict;
    }
    // No scored deadline (interactive runs): cap the lane at 10 min.
    let instance_deadline =
        instance_deadline.unwrap_or_else(|| Instant::now() + Duration::from_mins(10));
    let remaining = instance_deadline.saturating_duration_since(Instant::now());
    if remaining < Duration::from_secs(10) {
        return MarginRowTry::FinishedWithoutVerdict; // not enough budget for root gates + a useful tree
    }
    let deadline = instance_deadline
        .checked_sub(Duration::from_secs(3))
        .unwrap_or(instance_deadline);
    let dbg = std::env::var("NY_MARGIN_ROW_DEBUG").is_ok();
    let spec = match extract_twin_spec(onnx) {
        Some(s) => s,
        None => {
            if dbg {
                eprintln!("margin-row: extract_twin_spec returned None (structural mismatch)");
            }
            return MarginRowTry::FinishedWithoutVerdict;
        }
    };
    let (lo, hi, t, adv) = match parse_vnnlib_robustness(vnnlib) {
        Some(v) => v,
        None => {
            if dbg {
                eprintln!("margin-row: parse_vnnlib_robustness returned None");
            }
            return MarginRowTry::FinishedWithoutVerdict;
        }
    };
    // Proof bundle v2 (Proof profile only; write-only): the sites and the
    // clauses the lane's nodes will name.
    super::proof_bundle::note_sites_of(onnx);
    super::proof_bundle::note_clauses(t, &adv);
    if lo.len() != spec.n_in {
        if dbg {
            eprintln!(
                "margin-row: dim mismatch lo.len()={} != spec.n_in={}",
                lo.len(),
                spec.n_in
            );
        }
        return MarginRowTry::FinishedWithoutVerdict;
    }
    if dbg {
        eprintln!(
            "margin-row: spec OK (n_in={}, ops={}), running lane...",
            spec.n_in,
            spec.ops.len()
        );
    }
    // #router-r2 CR-1: the inline lane has no safe points; the authority
    // records the deadline its own rule derived and hands it back UNCHANGED.
    let deadline = crate::router::acting::claim_margin_row(
        ny_core::deadline_authority::LaneSlot::MarginRowInline,
        deadline,
    );
    let t0 = Instant::now();
    // #soundness-spine M2d: G3 needs the lane's own (t, adv) after the lane
    // consumed them.
    let (t_g3, adv_g3) = (t, adv.clone());
    // HARD SLICE ENFORCEMENT (banked 99ed4d42): the lane's cooperative
    // deadline checks are too coarse to trust with the caller's tail (they
    // overran a 45s slice to 113.1s on metaroom and once ran to the watchdog
    // kill). Detached worker + bounded join: on expiry the lane returns a
    // distinct `StillRunning` state in-budget, and the caller suppresses its
    // post-BaB graph tail. The worker still gets `Some(deadline)` so it normally
    // stops cooperatively well before the external cap fires.
    // #router-r2 CR-R: with a publish grace armed, a COMPLETED certificate may
    // publish up to `g` past the search deadline (`publish_grace`, read by the
    // lane itself), so the external cap waits the same `g` plus the collect
    // slack. Zero when unarmed: exactly today's cap.
    let cap = (deadline
        + crate::router::acting::collector_grace(
            ny_core::deadline_authority::LaneSlot::MarginRowInline,
        ))
    .saturating_duration_since(t0);
    let out = match run_capped_detached(cap, "inline", move || {
        let out = run_margin_row_lane_with_policy(
            &spec,
            &lo,
            &hi,
            t,
            &adv,
            Some(deadline),
            20_000,
            runtime_policy,
        );
        note_margin_row_execution(&out);
        out
    }) {
        DetachedTake::Finished(out) => out,
        DetachedTake::StillRunning => return MarginRowTry::StillRunning,
        DetachedTake::Failed => return MarginRowTry::FinishedWithoutVerdict,
    };
    match out {
        MarginRowOutcome::Unsat(stats, token) => {
            eprintln!(
                "margin-row BaB: UNSAT in {:.1}s (root_bound={:.4}, tree_classes={:?}, \
                 expansions={}, domains={}, maxDepth={}, dips={})",
                t0.elapsed().as_secs_f64(),
                stats.root_bound,
                stats.tree_classes,
                stats.expansions,
                stats.domains_created,
                stats.max_depth,
                stats.mono_raw_dips,
            );
            log_classwise_stats(&stats);
            match margin_row_receipt(token, vnnlib, t_g3, &adv_g3) {
                Some(receipt) => MarginRowTry::Verdict(VnncompResult::Unsat(receipt)),
                None => MarginRowTry::FinishedWithoutVerdict,
            }
        }
        MarginRowOutcome::Unknown { reason, stats } => {
            eprintln!(
                "margin-row BaB: no verdict in {:.1}s ({reason}{})",
                t0.elapsed().as_secs_f64(),
                stats
                    .as_ref()
                    .map(|s| format!(
                        ", root_bound={:.4}, expansions={}",
                        s.root_bound, s.expansions
                    ))
                    .unwrap_or_default(),
            );
            if let Some(stats) = &stats {
                log_classwise_stats(stats);
            }
            MarginRowTry::FinishedWithoutVerdict
        }
    }
}

#[derive(Default)]
struct ConcurrentLaneExecutionState {
    allocation_receipt_recorded: std::sync::atomic::AtomicBool,
    worker_active: std::sync::atomic::AtomicBool,
    verifier_active: std::sync::atomic::AtomicBool,
    overlap_recorded: std::sync::atomic::AtomicBool,
}

impl ConcurrentLaneExecutionState {
    fn new(allocation_receipt_recorded: bool) -> Self {
        Self {
            allocation_receipt_recorded: std::sync::atomic::AtomicBool::new(
                allocation_receipt_recorded,
            ),
            ..Self::default()
        }
    }

    fn mark_worker_started(&self) {
        self.worker_active
            .store(true, std::sync::atomic::Ordering::SeqCst);
        self.observe_overlap();
    }

    fn mark_worker_finished(&self) {
        self.worker_active
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    fn mark_verifier_started(&self) {
        self.verifier_active
            .store(true, std::sync::atomic::Ordering::SeqCst);
        self.observe_overlap();
    }

    fn mark_verifier_finished(&self) {
        self.verifier_active
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    fn observe_overlap(&self) {
        if self.worker_active.load(std::sync::atomic::Ordering::SeqCst)
            && self
                .verifier_active
                .load(std::sync::atomic::Ordering::SeqCst)
            && !self
                .overlap_recorded
                .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            crate::flight::note(
                MARGIN_ROW_CONCURRENT_OVERLAP_EXECUTION_METHOD,
                crate::flight::FlightStatus::Ran,
                Some(render_margin_row_concurrent_overlap_execution()),
            );
        }
    }
}

struct ConcurrentWorkerActiveGuard(std::sync::Arc<ConcurrentLaneExecutionState>);

impl Drop for ConcurrentWorkerActiveGuard {
    fn drop(&mut self) {
        self.0.mark_worker_finished();
    }
}

/// Stable result of concurrent-lane admission. A configured gate is not an
/// execution event unless this route reaches [`Self::Spawned`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConcurrentLaneAdmissionRoute {
    Disabled,
    AuthorityDisabled,
    CpuDevice,
    NoDeadline,
    ShortBudget,
    Model,
    Property,
    InputShape,
    NoSafeRemainder,
    GlobalUnavailable,
    PrivatePoolFailed,
    SpawnFailed,
    Spawned,
}

impl ConcurrentLaneAdmissionRoute {
    pub(crate) const fn stable_token(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::AuthorityDisabled => "authority_disabled",
            Self::CpuDevice => "cpu_device",
            Self::NoDeadline => "no_deadline",
            Self::ShortBudget => "short_budget",
            Self::Model => "model",
            Self::Property => "property",
            Self::InputShape => "input_shape",
            Self::NoSafeRemainder => "no_safe_remainder",
            Self::GlobalUnavailable => "global_unavailable",
            Self::PrivatePoolFailed => "private_pool_failed",
            Self::SpawnFailed => "spawn_failed",
            Self::Spawned => "spawned",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConcurrentLanePoolRoute {
    NotConsulted,
    SharedGlobal,
    ExplicitPrivate,
    VerifierRemainder,
    NoSafeRemainder,
    GlobalUnavailable,
}

impl ConcurrentLanePoolRoute {
    pub(crate) const fn stable_token(self) -> &'static str {
        match self {
            Self::NotConsulted => "not_consulted",
            Self::SharedGlobal => "shared_global",
            Self::ExplicitPrivate => "explicit_private",
            Self::VerifierRemainder => "verifier_remainder",
            Self::NoSafeRemainder => "no_safe_remainder",
            Self::GlobalUnavailable => "global_unavailable",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrivatePoolRoute {
    NotRequested,
    Built,
    BuildFailed,
}

impl PrivatePoolRoute {
    pub(crate) const fn stable_token(self) -> &'static str {
        match self {
            Self::NotRequested => "not_requested",
            Self::Built => "built",
            Self::BuildFailed => "build_failed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ConcurrentLaneResourceOutcome {
    pub(crate) pool_route: ConcurrentLanePoolRoute,
    pub(crate) lane_threads: Option<usize>,
    pub(crate) private_pool_route: PrivatePoolRoute,
}

impl ConcurrentLaneResourceOutcome {
    pub(crate) const NOT_CONSULTED: Self = Self {
        pool_route: ConcurrentLanePoolRoute::NotConsulted,
        lane_threads: None,
        private_pool_route: PrivatePoolRoute::NotRequested,
    };
}

fn concurrent_lane_pool_plan(
    policy: MarginRowScheduleRuntimePolicy,
    global: GlobalRayonOutcome,
) -> ConcurrentLaneResourceOutcome {
    if let Some(threads) = policy.concurrent_threads_override() {
        return ConcurrentLaneResourceOutcome {
            pool_route: if threads == 0 {
                ConcurrentLanePoolRoute::SharedGlobal
            } else {
                ConcurrentLanePoolRoute::ExplicitPrivate
            },
            lane_threads: Some(threads),
            private_pool_route: PrivatePoolRoute::NotRequested,
        };
    }

    if policy.verifier_thread_cap().is_none() {
        return ConcurrentLaneResourceOutcome {
            pool_route: ConcurrentLanePoolRoute::SharedGlobal,
            lane_threads: Some(0),
            private_pool_route: PrivatePoolRoute::NotRequested,
        };
    }

    let (Some(host_threads), Some(actual_threads)) = (global.host_threads, global.actual_threads)
    else {
        return ConcurrentLaneResourceOutcome {
            pool_route: ConcurrentLanePoolRoute::GlobalUnavailable,
            lane_threads: None,
            private_pool_route: PrivatePoolRoute::NotRequested,
        };
    };
    let Some(remainder) = host_threads
        .checked_sub(actual_threads)
        .filter(|value| *value > 0)
    else {
        return ConcurrentLaneResourceOutcome {
            pool_route: ConcurrentLanePoolRoute::NoSafeRemainder,
            lane_threads: None,
            private_pool_route: PrivatePoolRoute::NotRequested,
        };
    };
    ConcurrentLaneResourceOutcome {
        pool_route: ConcurrentLanePoolRoute::VerifierRemainder,
        lane_threads: Some(remainder),
        private_pool_route: PrivatePoolRoute::NotRequested,
    }
}

/// Admission route plus the worker handle, present exactly for `spawned`.
pub(crate) struct ConcurrentLaneAdmission {
    pub(crate) route: ConcurrentLaneAdmissionRoute,
    pub(crate) lane: Option<ConcurrentLane>,
    pub(crate) resources: ConcurrentLaneResourceOutcome,
}

impl ConcurrentLaneAdmission {
    fn refused(route: ConcurrentLaneAdmissionRoute) -> Self {
        debug_assert_ne!(route, ConcurrentLaneAdmissionRoute::Spawned);
        Self {
            route,
            lane: None,
            resources: ConcurrentLaneResourceOutcome::NOT_CONSULTED,
        }
    }

    fn refused_with_resources(
        route: ConcurrentLaneAdmissionRoute,
        resources: ConcurrentLaneResourceOutcome,
    ) -> Self {
        debug_assert_ne!(route, ConcurrentLaneAdmissionRoute::Spawned);
        Self {
            route,
            lane: None,
            resources,
        }
    }

    fn spawned(lane: ConcurrentLane, resources: ConcurrentLaneResourceOutcome) -> Self {
        Self {
            route: ConcurrentLaneAdmissionRoute::Spawned,
            lane: Some(lane),
            resources,
        }
    }
}

/// Handle to a margin-row lane running CONCURRENTLY with the internal
/// verifier (#epoch-bab).
pub(crate) struct ConcurrentLane {
    rx: std::sync::mpsc::Receiver<Option<VnncompResult>>,
    handle: std::thread::JoinHandle<()>,
    execution: std::sync::Arc<ConcurrentLaneExecutionState>,
}

pub(crate) enum ConcurrentLaneTake {
    Verdict(VnncompResult),
    FinishedWithoutVerdict,
    StillRunning,
}

impl ConcurrentLane {
    /// Release early-publication authority only after the main thread has
    /// durably recorded the scheduler allocation receipt. A worker that closes
    /// before this point sends its proof through the ordinary channel; it never
    /// waits for this bit.
    pub(crate) fn mark_allocation_receipt_recorded(&self) {
        self.execution
            .allocation_receipt_recorded
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Mark the internal verifier active. Whichever side becomes active second
    /// emits the one-shot exact overlap marker.
    pub(crate) fn mark_verifier_started(&self) {
        self.execution.mark_verifier_started();
    }

    /// Mark the internal verifier inactive after its invocation returns.
    pub(crate) fn mark_verifier_finished(&self) {
        self.execution.mark_verifier_finished();
    }

    /// Collect the concurrent lane's verdict, waiting up to `grace` for it to
    /// Peek for a verdict WITHOUT consuming the handle (#twinwall-join).
    /// The same lane is consulted twice per instance: once on the short grace
    /// right after the internal verifier returns, and once at the very end of
    /// the post-BaB tail. A message the first call did not wait long enough
    /// for stays queued and is still there for the second — the receiver is
    /// only drained by an actual `recv`. Moving the handle at the first
    /// consult is what silently discarded a certified proof on cifar100_2024.
    ///
    /// Because it borrows, this CANNOT join, so unlike [`Self::take`] it makes
    /// no promise that worker-owned graph memory is released. Callers needing
    /// that guarantee must still reach the consuming `take`.
    pub(crate) fn peek(&self, grace: Duration) -> ConcurrentLaneTake {
        match self.rx.recv_timeout(grace) {
            Ok(Some(verdict)) => ConcurrentLaneTake::Verdict(verdict),
            Ok(None) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                ConcurrentLaneTake::FinishedWithoutVerdict
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => ConcurrentLaneTake::StillRunning,
        }
    }

    /// land.  Keep timeout distinct from a completed no-verdict worker: only
    /// the former can still own/allocate graph memory after this handle drops.
    pub(crate) fn take(self, grace: Duration) -> ConcurrentLaneTake {
        match self.rx.recv_timeout(grace) {
            Ok(verdict) => {
                // The message is sent before the closure's captured model is
                // destroyed. Join on every completed path so `Finished*`
                // literally means all worker-owned graph memory is gone.
                if self.handle.join().is_err() {
                    return ConcurrentLaneTake::FinishedWithoutVerdict;
                }
                match verdict {
                    Some(verdict) => ConcurrentLaneTake::Verdict(verdict),
                    None => ConcurrentLaneTake::FinishedWithoutVerdict,
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                let _ = self.handle.join();
                ConcurrentLaneTake::FinishedWithoutVerdict
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Dropping the handle detaches the worker; the caller must
                // suppress every later graph-heavy tail for this instance.
                ConcurrentLaneTake::StillRunning
            }
        }
    }
}

/// Resolve the concurrent-lane gate: exact `NY_MARGIN_ROW_CONCURRENT=1|0`
/// wins wherever the variable is PRESENT (so it stays a kill switch for a
/// preset that asks for the lane); otherwise the typed
/// `margin_row.concurrent` preset key decides; absent from both ⇒ OFF, i.e.
/// byte-identical to the shipped reserve-only path.
#[allow(dead_code)]
pub(crate) fn concurrent_lane_armed(preset: Option<&Path>) -> bool {
    MarginRowScheduleRuntimePolicy::from_layered_process(preset).configured_concurrent
}

/// Read the concurrent selector exclusively from the immutable scheduler
/// policy used by scored execution.
pub(crate) const fn concurrent_lane_armed_with_policy(
    policy: MarginRowScheduleRuntimePolicy,
) -> bool {
    policy.configured_concurrent
}

/// Resolve the typed f32-root-tableau gate. Returns `None` when the preset
/// stays silent, so the caller leaves the propagate-side flag untouched.
pub(crate) fn root_f32_from_preset(preset: Option<&Path>) -> Option<bool> {
    preset
        .and_then(|p| crate::preset::load_preset(p).ok())
        .and_then(|c| c.margin_row.root_f32)
}

/// Resolve the typed backward-interm key (#backward-interm).
// STAGED AHEAD OF ITS CONSUMER. Added by 043d7ff75 but nothing calls it yet,
// so the crate fails -D warnings on dead_code. DELETE THIS ATTRIBUTE with the
// commit that wires the preset key through; it is not a permanent exemption.
#[allow(dead_code)]
pub(crate) fn backward_interm_from_preset(preset: Option<&Path>) -> Option<bool> {
    preset
        .and_then(|p| crate::preset::load_preset(p).ok())
        .and_then(|c| c.margin_row.backward_interm)
}

/// Resolve the typed STATEFUL Clip-and-Verify key (#clip-stateful).
pub(crate) fn clip_stateful_from_preset(preset: Option<&Path>) -> Option<bool> {
    preset
        .and_then(|p| crate::preset::load_preset(p).ok())
        .and_then(|c| c.margin_row.clip_stateful)
}

/// Resolve the typed closure-adaptive tier ladder gate (#closure-adaptive).
///
/// WITHOUT THIS READER AND ITS CALL SITE THE ENTIRE LADDER IS DEAD CODE IN
/// COMPETITION. `run_instance.sh` exports no `NY_*`, so the env half
/// (`NY_MARGIN_ROW_CLOSURE_ADAPTIVE`) can never arm it on a scored run; the
/// typed key is the only route. The ladder shipped without this bridge — its
/// own search-space entry (`ny-levers/src/space.rs:504-507`) records the gap
/// rather than closing it, which is exactly how three other mechanisms
/// (`NY_EPOCH_DEPTH`, `NY_CGAN_RECLAIM`, `NY_MARGIN_ROW_CLIP_ROWS`) ended up
/// built, correct and unreachable.
pub(crate) fn closure_adaptive_from_preset(preset: Option<&Path>) -> Option<bool> {
    preset
        .and_then(|p| crate::preset::load_preset(p).ok())
        .and_then(|c| c.margin_row.closure_adaptive)
}

/// Resolve the typed branching reduction (#branch-reduceop).
pub(crate) fn branch_reduce_max_from_preset(preset: Option<&Path>) -> Option<bool> {
    preset
        .and_then(|p| crate::preset::load_preset(p).ok())
        .and_then(|c| c.margin_row.branch_reduce_max)
}

/// Resolve the typed parent-lift route (#branch-parent-lift).
pub(crate) fn parent_lift_from_preset(preset: Option<&Path>) -> Option<bool> {
    preset
        .and_then(|p| crate::preset::load_preset(p).ok())
        .and_then(|c| c.margin_row.parent_lift)
}

/// Resolve the typed backward-interm SECONDS cap (#backward-interm budget).
pub(crate) fn bi_secs_from_preset(preset: Option<&Path>) -> Option<u64> {
    preset
        .and_then(|p| crate::preset::load_preset(p).ok())
        .and_then(|c| c.margin_row.bi_secs)
}

/// Start the margin-row lane on a BACKGROUND THREAD, concurrently with the
/// internal verifier (#epoch-bab).
///
/// WHY THIS IS ~FREE, and why it beats the budget reserve it replaces: the
/// wall presets (`cifar100_2024`, `tinyimagenet_2024`, `metaroom_2023`) all
/// run the internal verifier on `device: wgpu` — the GPU — while this lane is
/// pure CPU/rayon. They contend for memory bandwidth and host threads, not
/// for the resource the verifier is actually bound on. A reserve BUYS the
/// lane time by TAKING it from the verifier (measured: 45 s would forfeit 28
/// `sat_relu` and 10 `cifar100` solves); running concurrently gives the lane
/// the whole instance budget while the verifier keeps its own.
///
/// Soundness is unchanged: this is the same fail-closed lane
/// (`Unsat`/`Unknown` only, never `Sat`), its verdict is consumed ONLY when
/// the internal verifier came back undecided, and it is bounded by the same
/// instance deadline. `None` = not started (lane disabled, structural
/// mismatch, or too little budget), in which case the caller falls back to
/// the inline post-verifier attempt exactly as before.
#[allow(dead_code)]
#[expect(
    clippy::large_types_passed_by_value,
    reason = "the runtime policy is MOVED into the lane it configures; a reference here would only add a clone at the move"
)]
pub(crate) fn spawn_concurrent_lane(
    onnx: &Path,
    vnnlib: &Path,
    preset: Option<&Path>,
    instance_deadline: Option<Instant>,
    runtime_policy: MarginRowRuntimePolicy,
) -> Option<ConcurrentLane> {
    let schedule_policy = MarginRowScheduleRuntimePolicy::from_layered_process(preset);
    let global_rayon_outcome = configure_global_rayon(schedule_policy).ok()?;
    let device = preset
        .and_then(|path| crate::preset::load_preset(path).ok())
        .and_then(|config| config.general.device);
    let gpu_device_premise = device
        .as_deref()
        .is_some_and(|device| matches!(device, "wgpu" | "cuda" | "metal" | "gpu"));
    let admission = spawn_concurrent_lane_impl(
        onnx,
        vnnlib,
        instance_deadline,
        runtime_policy,
        schedule_policy,
        global_rayon_outcome,
        gpu_device_premise,
        true,
    );
    admission.lane
}

/// Policy-aware concurrent admission for scored execution. Every refusal has
/// a stable route, and the returned lane starts with early publication blocked
/// until [`ConcurrentLane::mark_allocation_receipt_recorded`] is called.
#[expect(
    clippy::large_types_passed_by_value,
    reason = "the runtime policy is MOVED into the lane it configures; a reference here would only add a clone at the move"
)]
pub(crate) fn spawn_concurrent_lane_with_schedule_policy(
    onnx: &Path,
    vnnlib: &Path,
    instance_deadline: Option<Instant>,
    runtime_policy: MarginRowRuntimePolicy,
    schedule_policy: MarginRowScheduleRuntimePolicy,
    global_rayon_outcome: GlobalRayonOutcome,
    gpu_device_premise: bool,
) -> ConcurrentLaneAdmission {
    spawn_concurrent_lane_impl(
        onnx,
        vnnlib,
        instance_deadline,
        runtime_policy,
        schedule_policy,
        global_rayon_outcome,
        gpu_device_premise,
        false,
    )
}

#[expect(
    clippy::large_types_passed_by_value,
    reason = "the runtime policy is MOVED into the lane it configures; a reference here would only add a clone at the move"
)]
fn spawn_concurrent_lane_impl(
    onnx: &Path,
    vnnlib: &Path,
    instance_deadline: Option<Instant>,
    runtime_policy: MarginRowRuntimePolicy,
    schedule_policy: MarginRowScheduleRuntimePolicy,
    global_rayon_outcome: GlobalRayonOutcome,
    gpu_device_premise: bool,
    allocation_receipt_recorded: bool,
) -> ConcurrentLaneAdmission {
    if !ny_propagate::margin_row::margin_row_bab_enabled() {
        return ConcurrentLaneAdmission::refused(ConcurrentLaneAdmissionRoute::AuthorityDisabled);
    }
    // OPT-IN (`NY_MARGIN_ROW_CONCURRENT=1`). Default OFF so the shipped path is
    // reserve-only: the 45s default feeds a lane that runs after the verifier.
    // Enabling both would tax the verifier twice — it would lose the reserved
    // 45s AND contend with the concurrent lane for the rest — and that
    // combination has no sealed measurement. Historical bookkeeping differs:
    // CIFAR100 +23 appears to use 45s, while the TinyImageNet +67 commit bodies
    // explicitly record 82s. Flip this on only with a sealed A/B that watches
    // the retained rows.
    //
    // MEASURED on cifar100_2024, 2026-08-02 (RTX 5080 + WSL2, 10-CPU cgroup,
    // rows 1..12 at the official 100 s budget, same binary, arms interleaved):
    //
    //   stock       solved 2/12 (sat 1, unsat 1)  timeout 10
    //   concurrent  solved 2/12 (sat 1, unsat 1)  timeout 10
    //
    // Identical -- no conversion and no regression. The premise below HELD, so
    // this is a fair test of the mechanism rather than a premise failure: the
    // verifier's sound f64 A.W GEMMs run on cuBLAS while the collector/alpha
    // work is CPU-bound. It simply does not pay on this category. Default OFF.
    //
    // TYPED ROUTE (#twinwall-provenance, 2026-08-03): `margin_row.concurrent`
    // in the category preset arms the same lane. tinyimagenet_2024 needs it,
    // because its whole +67 UNSAT bank was gathered through `sweep_targets`,
    // which hands the lane the FULL per-instance budget — a budget the
    // reserve-only route cannot reproduce at ANY reserve value (every banked
    // row recorded >= 50 s of lane time against a 45 s shipped reserve, and
    // the reserve slice only starts after the internal verifier returns).
    // The env var still wins wherever it is present, in both directions.
    if !concurrent_lane_armed_with_policy(schedule_policy) {
        return ConcurrentLaneAdmission::refused(ConcurrentLaneAdmissionRoute::Disabled);
    }
    // THE PREMISE MUST HOLD. Running concurrently is ~free only because the
    // internal verifier is on the GPU while this lane is on the CPU. On a
    // CPU-device category they contend for the SAME resource and the lane
    // would slow the verifier down for nothing — and `sat_relu` (no device
    // set, so CPU) is in the twin-wall structural class, scores 101/101 for
    // ny, and has 28 solves within 45s of its budget wall. Spawn only where
    // the premise is true; elsewhere the inline post-verifier attempt still
    // runs, exactly as before.
    if !gpu_device_premise {
        return ConcurrentLaneAdmission::refused(ConcurrentLaneAdmissionRoute::CpuDevice);
    }
    let Some(deadline) = instance_deadline else {
        return ConcurrentLaneAdmission::refused(ConcurrentLaneAdmissionRoute::NoDeadline);
    };
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining < Duration::from_secs(10) {
        return ConcurrentLaneAdmission::refused(ConcurrentLaneAdmissionRoute::ShortBudget);
    }
    // Structural admission happens on THIS thread (cheap, and keeps the
    // fail-closed decision on the same path the inline lane uses).
    let Some(spec) = extract_twin_spec(onnx) else {
        return ConcurrentLaneAdmission::refused(ConcurrentLaneAdmissionRoute::Model);
    };
    // #cct-margin-row-admission: refuse at ADMISSION what the compiler would
    // refuse inside the worker — in particular a net whose root tableau is
    // above the shared ceiling (the cct TinyImageNet CNN7 pair extracts
    // cleanly once the FC-BN fold admits it, and would OOM an uncapped host).
    // Every net the lane has banked evidence on satisfies this predicate.
    if !twin_spec_compiles_with_root_policy(&spec, &runtime_policy) {
        eprintln!(
            "margin-row BaB: CONCURRENT lane not started — net outside the compiled head \
             contract or above the root-tableau cap (entries={:?}, cap={})",
            spec.root_tableau_entries(),
            ny_propagate::margin_row::spec::MAX_ROOT_TABLEAU_ENTRIES,
        );
        return ConcurrentLaneAdmission::refused(ConcurrentLaneAdmissionRoute::Model);
    }
    let Some((lo, hi, t, adv)) = parse_vnnlib_robustness(vnnlib) else {
        return ConcurrentLaneAdmission::refused(ConcurrentLaneAdmissionRoute::Property);
    };
    if lo.len() != spec.n_in {
        return ConcurrentLaneAdmission::refused(ConcurrentLaneAdmissionRoute::InputShape);
    }
    // Proof bundle v2 (Proof profile only; write-only): the sites and the
    // clauses the lane's nodes will name.
    super::proof_bundle::note_sites_of(onnx);
    super::proof_bundle::note_clauses(t, &adv);
    // Leave the tail to the caller's own wrap-up (result write, witness
    // gating) exactly as the inline lane does.
    let lane_deadline = deadline
        .checked_sub(Duration::from_secs(3))
        .unwrap_or(deadline);
    // #router-r2 CR-1: the concurrent worker has no safe points after spawn;
    // the authority records its deadline and hands it back UNCHANGED.
    let lane_deadline = crate::router::acting::claim_margin_row(
        ny_core::deadline_authority::LaneSlot::MarginRowConcurrent,
        lane_deadline,
    );
    // BOUND THE CONTENTION. The lane is rayon-parallel and would otherwise
    // take the global pool, competing with the verifier's host-side threads
    // (the GPU is what the verifier is bound on, but it still needs CPU to
    // feed it). Confining the lane to a quarter of the cores keeps it useful
    // while leaving the verifier its orchestration headroom. Measured
    // unbounded: the lane runs ~2.4x slower concurrently than standalone
    // (prop_1498, 16.4s -> 40.1s), and a 69.5s standalone closure
    // (resnet_large prop_idx_4486) misses the budget entirely.
    // Default 0 = use the global rayon pool (all cores), which is the
    // configuration the conversion evidence was gathered under. Bounding the
    // lane cuts contention with the verifier BUT also slows the lane, and the
    // lane's speed is what decides whether it closes inside the budget — so
    // the trade is not obviously a win in either direction and is left as a
    // knob until measured. NY_MARGIN_ROW_CONCURRENT_THREADS=n to bound it.
    // #verifier-thread-cap: derive the lane only from the ACTUAL global pool.
    // A refused global cap must never manufacture a private remainder from the
    // requested value, and zero spare capacity is a refusal rather than an
    // oversubscribing one-thread pool.
    let mut resources = concurrent_lane_pool_plan(schedule_policy, global_rayon_outcome);
    if resources.pool_route == ConcurrentLanePoolRoute::NoSafeRemainder {
        return ConcurrentLaneAdmission::refused_with_resources(
            ConcurrentLaneAdmissionRoute::NoSafeRemainder,
            resources,
        );
    }
    if resources.pool_route == ConcurrentLanePoolRoute::GlobalUnavailable {
        return ConcurrentLaneAdmission::refused_with_resources(
            ConcurrentLaneAdmissionRoute::GlobalUnavailable,
            resources,
        );
    }
    let lane_threads = resources
        .lane_threads
        .expect("an admitted lane pool plan has an exact width");
    let available = if lane_threads == 0 {
        rayon::current_num_threads().max(1)
    } else {
        lane_threads
    };
    let frontier = ny_propagate::preset_gates::margin_row_frontier(available);
    let lru_cap = ny_propagate::preset_gates::margin_row_lru_cap();
    if !twin_spec_compiles_with_tree_policy(&spec, &runtime_policy, adv.len(), frontier, lru_cap) {
        eprintln!("margin-row BaB: CONCURRENT lane not started — wide root/tree/cache working set refused before worker allocation");
        return ConcurrentLaneAdmission::refused_with_resources(
            ConcurrentLaneAdmissionRoute::Model,
            resources,
        );
    }
    let private_pool = if lane_threads == 0 {
        None
    } else {
        match rayon::ThreadPoolBuilder::new()
            .num_threads(lane_threads)
            .thread_name(|i| format!("margin-row-{i}"))
            .build()
        {
            Ok(pool) => {
                resources.private_pool_route = PrivatePoolRoute::Built;
                Some(pool)
            }
            Err(error) => {
                eprintln!("margin-row BaB: private {lane_threads}-thread pool REFUSED: {error}");
                resources.private_pool_route = PrivatePoolRoute::BuildFailed;
                return ConcurrentLaneAdmission::refused_with_resources(
                    ConcurrentLaneAdmissionRoute::PrivatePoolFailed,
                    resources,
                );
            }
        }
    };
    if let Some(pool) = &private_pool {
        eprintln!(
            "margin-row BaB: CONCURRENT lane confined to a private {}-thread pool",
            pool.current_num_threads(),
        );
    }
    let early_exit = schedule_policy.early_exit_armed();
    // #soundness-spine M2d: G3 inputs, owned by the worker.
    let (vnnlib_g3, t_g3, adv_g3) = (vnnlib.to_path_buf(), t, adv.clone());
    let (tx, rx) = std::sync::mpsc::channel();
    let execution = std::sync::Arc::new(ConcurrentLaneExecutionState::new(
        allocation_receipt_recorded,
    ));
    let worker_execution = std::sync::Arc::clone(&execution);
    let handle = match std::thread::Builder::new()
        .name("margin-row-concurrent".into())
        .spawn(move || {
            let t0 = Instant::now();
            let run = || {
                worker_execution.mark_worker_started();
                let out = {
                    // Only the solver call owns the overlap claim. Private-pool
                    // construction before it and terminal receipt/log/channel
                    // bookkeeping after it are deliberately outside this guard.
                    let _active =
                        ConcurrentWorkerActiveGuard(std::sync::Arc::clone(&worker_execution));
                    run_margin_row_lane_with_policy(
                        &spec,
                        &lo,
                        &hi,
                        t,
                        &adv,
                        Some(lane_deadline),
                        20_000,
                        runtime_policy,
                    )
                };
                note_margin_row_execution(&out);
                out
            };
            let out = if let Some(pool) = private_pool {
                pool.install(run)
            } else {
                run()
            };
            let verdict = match out {
                MarginRowOutcome::Unsat(stats, token) => {
                    let lane_secs = t0.elapsed().as_secs_f64();
                    eprintln!(
                        "margin-row BaB (concurrent): UNSAT in {lane_secs:.1}s (root_bound={:.4}, \
                         expansions={}, maxDepth={}, ledger={:?})",
                        stats.root_bound, stats.expansions, stats.max_depth, stats.ledger_ok
                    );
                    // #soundness-spine M2b/M2d: only a G3-ratified,
                    // instance-scoped token may become a published `unsat`.
                    match margin_row_receipt(token, &vnnlib_g3, t_g3, &adv_g3) {
                        None => None,
                        Some(receipt) => {
                            // #twinwall-early-exit. The row is certified; the internal
                            // verifier's remaining budget can only re-derive the same
                            // answer or nothing at all. Publish and end the instance.
                            // Returns only if publication was declined (route not
                            // armed, or the main thread already claimed it), in which
                            // case the verdict goes down the channel as before.
                            if early_exit
                                && worker_execution
                                    .allocation_receipt_recorded
                                    .load(std::sync::atomic::Ordering::Acquire)
                            {
                                crate::commands::vnncomp::publish_lane_unsat_and_exit(
                                    lane_secs,
                                    receipt.clone(),
                                );
                            } else if early_exit {
                                eprintln!(
                                    "margin-row BaB: early exit DEFERRED — schedule allocation receipt \
                                     was not recorded when the proof completed; sending verdict through \
                                     the ordinary channel"
                                );
                            } else {
                                eprintln!(
                                    "margin-row BaB: early exit DISARMED \
                                     (NY_MARGIN_ROW_EARLY_EXIT/margin_row.early_exit); the verdict waits \
                                     for the internal verifier to return"
                                );
                            }
                            Some(VnncompResult::Unsat(receipt))
                        }
                    }
                }
                MarginRowOutcome::Unknown { reason, .. } => {
                    eprintln!(
                        "margin-row BaB (concurrent): no verdict in {:.1}s ({reason})",
                        t0.elapsed().as_secs_f64()
                    );
                    None
                }
            };
            let _ = tx.send(verdict);
        }) {
        Ok(handle) => handle,
        Err(_) => {
            return ConcurrentLaneAdmission::refused_with_resources(
                ConcurrentLaneAdmissionRoute::SpawnFailed,
                resources,
            );
        }
    };
    eprintln!("margin-row BaB: started CONCURRENT lane alongside the internal verifier");
    ConcurrentLaneAdmission::spawned(
        ConcurrentLane {
            rx,
            handle,
            execution,
        },
        resources,
    )
}

// --------------------------------------------------------------------------
//  ONNX -> TwinSpec (parity port of core.py::Net.__init__)
// --------------------------------------------------------------------------

fn tensor_f32(t: &TensorProto) -> Option<Vec<f64>> {
    // data_type 1 = FLOAT; reject external payloads.
    if t.data_type != 1 || t.data_location != 0 {
        return None;
    }
    if !t.raw_data.is_empty() {
        if !t.raw_data.len().is_multiple_of(4) {
            return None;
        }
        Some(
            t.raw_data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| ny_core::f32_to_f64_exact(f32::from_le_bytes(*c)))
                .collect(),
        )
    } else {
        Some(
            t.float_data
                .iter()
                .map(|&v| ny_core::f32_to_f64_exact(v))
                .collect(),
        )
    }
}

/// INT64 tensor payload (shape constants). Rejects external payloads.
fn tensor_i64(t: &TensorProto) -> Option<Vec<i64>> {
    // data_type 7 = INT64.
    if t.data_type != 7 || t.data_location != 0 {
        return None;
    }
    if !t.raw_data.is_empty() {
        if !t.raw_data.len().is_multiple_of(8) {
            return None;
        }
        Some(
            t.raw_data
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c))
                .collect(),
        )
    } else {
        Some(t.int64_data.clone())
    }
}

fn attr_ints<'a>(n: &'a NodeProto, name: &str) -> Option<&'a [i64]> {
    n.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.as_slice())
}

fn attr_f(n: &NodeProto, name: &str, default: f64) -> f64 {
    n.attribute
        .iter()
        .find(|a| a.name == name)
        .map_or(default, |a| f64::from(a.f_value()))
}

fn attr_i(n: &NodeProto, name: &str, default: i64) -> i64 {
    n.attribute
        .iter()
        .find(|a| a.name == name)
        .map_or(default, |a| a.i_value())
}

/// #cct-margin-row-admission: fold the inference-form `BatchNormalization`
/// node `bn`, which must consume `gemm_out` as its DATA input, into the Gemm
/// whose row-major `(n_out, n_in)` weights and bias are given. Returns the
/// certified fold and the BN's output name (which the folded Gemm now
/// publishes). `None` on anything outside the inference form or anything
/// [`fold_dense_batchnorm`] cannot certify — the caller then declines the
/// whole net.
///
/// The arithmetic is NOT abc's `merge_bn` (nearest-rounded fp32, no error
/// term): it is the f64 Conv+BN fold's own operation order with a carried
/// relative weight error and absolute bias error, charged downstream.
fn fold_gemm_batchnorm<'g>(
    bn: &'g NodeProto,
    gemm_out: &str,
    inits: &HashMap<&str, &TensorProto>,
    weight: &[f64],
    bias: &[f64],
    shape: (usize, usize),
) -> Option<(DenseBnFold, &'g str)> {
    // Inference form only: exactly (X, scale, B, mean, var) in and ONE output
    // out (training mode emits running statistics), X is this Gemm's output.
    if bn.input.len() != 5 || bn.output.len() != 1 || bn.input[0] != gemm_out {
        return None;
    }
    if attr_i(bn, "training_mode", 0) != 0 {
        return None;
    }
    let n_out = shape.0;
    let gamma = tensor_f32(inits.get(bn.input[1].as_str())?)?;
    let beta = tensor_f32(inits.get(bn.input[2].as_str())?)?;
    let mean = tensor_f32(inits.get(bn.input[3].as_str())?)?;
    let var = tensor_f32(inits.get(bn.input[4].as_str())?)?;
    if [&gamma, &beta, &mean, &var]
        .iter()
        .any(|v| v.len() != n_out)
    {
        return None;
    }
    // ONNX `epsilon` is a FLOAT attribute defaulting to 1e-5: an absent
    // attribute means the f32 value, widened exactly (cnn7 states it
    // explicitly as 9.999999747378752e-06).
    let eps = attr_f(bn, "epsilon", f64::from(1e-5_f32));
    let fold = fold_dense_batchnorm(
        weight,
        bias,
        shape,
        &DenseBnParams {
            gamma: &gamma,
            beta: &beta,
            mean: &mean,
            var: &var,
            eps,
        },
    )?;
    Some((fold, bn.output.first()?.as_str()))
}

/// Extract the twin-wall family net; `None` on ANY structural deviation.
pub(crate) fn extract_twin_spec(onnx: &Path) -> Option<TwinSpec> {
    extract_twin_spec_named(onnx).map(|(spec, _)| spec)
}

/// Proof bundle v2 (ENGINE_PROGRAM §1.7 B1): per Relu op of the extracted
/// spec, `(op index, ONNX name of the Relu node's input[0], flat size)` —
/// the ONNX-level identity an offline checker resolves literals against.
pub(crate) type TwinSiteNames = Vec<(usize, String, usize)>;

/// [`extract_twin_spec`] plus the Relu sites' ONNX names, recorded during
/// the SAME walk (the returned spec is bit-identical to `extract_twin_spec`'s:
/// it is that function's body).
pub(crate) fn extract_twin_spec_named(onnx: &Path) -> Option<(TwinSpec, TwinSiteNames)> {
    // Preserve the legacy early environmental refusal before source loading.
    ny_core::require_f64_interval_proof_environment().ok()?;
    let bytes = std::fs::read(onnx).ok()?;
    extract_twin_spec_named_bytes(&bytes)
}

/// The same original extraction walk over an immutable source snapshot.
/// This entry does not reload a pathname between source authentication and
/// extraction; callers retain and price the exact bytes themselves.
pub(crate) fn extract_twin_spec_named_bytes(bytes: &[u8]) -> Option<(TwinSpec, TwinSiteNames)> {
    // Certify the producer before decoding source constants or folding BN.
    // Both the ordinary and proof-named entry points share this gate. Root
    // qualification on a later worker cannot repair an earlier fold.
    ny_core::require_f64_interval_proof_environment().ok()?;
    let model = ModelProto::decode(bytes).ok()?;
    extract_legacy_twin_spec_named_model(&model)
}

/// Original-source facade extraction. Its native matrix-row branch keeps the
/// original MatMul/Add tensor trace and exact parameter bits; the old entry
/// above remains rank-four-only for existing callers. One decode selects the
/// branch, so the staged decoder census still covers three total decodes.
pub(crate) fn extract_original_twin_spec_named_bytes(
    bytes: &[u8],
    budget: &ny_propagate::certify::work::WorkBudget,
    at: Instant,
) -> Option<(TwinSpec, TwinSiteNames)> {
    super::prepared_root_source::checkpoint(budget, at).ok()?;
    let model = ModelProto::decode(bytes).ok()?;
    super::prepared_root_source::checkpoint(budget, at).ok()?;
    let rank = model
        .graph
        .as_ref()?
        .input
        .first()?
        .r#type
        .as_ref()?
        .tensor_type
        .as_ref()?
        .shape
        .as_ref()?
        .dim
        .len();
    let result = if rank == 2 {
        extract_native_affine_twin_spec_named_model(&model, budget, at)
    } else {
        extract_legacy_twin_spec_named_model(&model)
    };
    super::prepared_root_source::checkpoint(budget, at).ok()?;
    result
}

fn extract_native_affine_twin_spec_named_model(
    model: &ModelProto,
    budget: &ny_propagate::certify::work::WorkBudget,
    at: Instant,
) -> Option<(TwinSpec, TwinSiteNames)> {
    use ny_onnx::onnx_proto::tensor_shape_proto::dimension::Value;
    let g = model.graph.as_ref()?;
    if g.input.len() != 1 || g.output.len() != 1 || !g.sparse_initializer.is_empty() {
        return None;
    }
    let input = &g.input[0];
    let input_type = input.r#type.as_ref()?.tensor_type.as_ref()?;
    let dims = &input_type.shape.as_ref()?.dim;
    if input.name.is_empty() || input_type.elem_type != 1 || dims.len() != 2 {
        return None;
    }
    // A symbolic batch remains the original symbolic annotation. This
    // source's batch-separable row map is evaluated on input[0,j]; authentic
    // property admission later requires exactly n_in original X scalars.
    match dims[0].value.as_ref()? {
        Value::DimValue(1) => {}
        Value::DimParam(name) if !name.is_empty() => {}
        _ => return None,
    }
    let Value::DimValue(width) = dims[1].value.as_ref()? else {
        return None;
    };
    let n_in = usize::try_from(*width).ok()?;
    if n_in == 0 {
        return None;
    }
    let mut inits: HashMap<&str, &TensorProto> = HashMap::new();
    inits.try_reserve(g.initializer.len()).ok()?;
    for (i, tensor) in g.initializer.iter().enumerate() {
        super::prepared_root_source::poll(budget, at, i).ok()?;
        if tensor.name.is_empty()
            || tensor.name == input.name
            || inits.insert(tensor.name.as_str(), tensor).is_some()
        {
            return None;
        }
    }
    let mut ops = Vec::new();
    ops.try_reserve_exact(g.node.len()).ok()?;
    let mut sites = Vec::new();
    sites.try_reserve_exact(g.node.len()).ok()?;
    let mut current_name = input.name.as_str();
    let mut current_id = 0usize;
    let mut current_width = n_in;
    let mut ni = 0usize;
    while ni < g.node.len() {
        super::prepared_root_source::checkpoint(budget, at).ok()?;
        let n = &g.node[ni];
        if !matches!(n.domain.as_str(), "" | "ai.onnx")
            || !n.attribute.is_empty()
            || n.output.len() != 1
            || n.output[0].is_empty()
        {
            return None;
        }
        match n.op_type.as_str() {
            "MatMul" => {
                if n.input.len() != 2 || n.input[0] != current_name {
                    return None;
                }
                let raw = inits.get(n.input[1].as_str())?;
                if raw.dims.len() != 2 {
                    return None;
                }
                let nin = usize::try_from(raw.dims[0]).ok()?;
                let nout = usize::try_from(raw.dims[1]).ok()?;
                let cells = nin.checked_mul(nout)?;
                if nin != current_width || nout == 0 {
                    return None;
                }
                let raw_weight = native_affine_tensor_f32(raw, budget, at)?;
                if raw_weight.len() != cells {
                    return None;
                }
                let add = g.node.get(ni.checked_add(1)?)?;
                if add.op_type != "Add"
                    || !matches!(add.domain.as_str(), "" | "ai.onnx")
                    || !add.attribute.is_empty()
                    || add.input.len() != 2
                    || add.output.len() != 1
                    || add.output[0].is_empty()
                {
                    return None;
                }
                let matmul_out = n.output[0].as_str();
                let bias_name = match (add.input[0] == matmul_out, add.input[1] == matmul_out) {
                    (true, false) => add.input[1].as_str(),
                    (false, true) => add.input[0].as_str(),
                    _ => return None,
                };
                // Folding a real bias must not erase an independently used
                // intermediate MatMul, including an exported earlier output.
                let mut uses = 0usize;
                for (j, node) in g.node.iter().enumerate() {
                    super::prepared_root_source::poll(budget, at, j).ok()?;
                    for (k, name) in node.input.iter().enumerate() {
                        super::prepared_root_source::poll(budget, at, k).ok()?;
                        if name == matmul_out {
                            uses = uses.checked_add(1)?;
                        }
                    }
                }
                if uses != 1 || g.output[0].name == matmul_out {
                    return None;
                }
                let raw_bias = inits.get(bias_name)?;
                if !matches!(raw_bias.dims.as_slice(), [rows] | [1,rows] if usize::try_from(*rows).ok()==Some(nout))
                {
                    return None;
                }
                let bias = native_affine_tensor_f32(raw_bias, budget, at)?;
                if bias.len() != nout {
                    return None;
                }
                // Authored B is [n_in,n_out], whereas TwinOpSpec stores W
                // [n_out,n_in]. This is an exact permutation after exact bit
                // widening, with no rounded parameter fold or dummy layer.
                let mut weight = Vec::new();
                weight.try_reserve_exact(cells).ok()?;
                if weight.capacity() > cells {
                    return None;
                }
                for i in 0..cells {
                    super::prepared_root_source::poll(budget, at, i).ok()?;
                    weight.push(0.0);
                }
                for input_row in 0..nin {
                    for output_column in 0..nout {
                        super::prepared_root_source::poll(
                            budget,
                            at,
                            input_row * nout + output_column,
                        )
                        .ok()?;
                        weight[output_column * nin + input_row] =
                            raw_weight[input_row * nout + output_column];
                    }
                }
                ops.push(TwinOpSpec::Gemm {
                    input: current_id,
                    weight,
                    bias,
                    bias_err: Vec::new(),
                    weight_rel_err: 0.0,
                    shape: (nout, nin),
                });
                current_id = ops.len();
                current_name = add.output[0].as_str();
                current_width = nout;
                ni = ni.checked_add(2)?;
            }
            "Relu" => {
                if n.input.len() != 1 || n.input[0] != current_name {
                    return None;
                }
                sites.push((ops.len(), n.input[0].clone(), current_width));
                ops.push(TwinOpSpec::Relu { input: current_id });
                current_id = ops.len();
                current_name = n.output[0].as_str();
                ni = ni.checked_add(1)?;
            }
            _ => return None,
        }
    }
    if current_name != g.output[0].name
        || ops.len() != 3
        || !matches!(
            ops.as_slice(),
            [
                TwinOpSpec::Gemm { input: 0, .. },
                TwinOpSpec::Relu { input: 1 },
                TwinOpSpec::Gemm { input: 2, .. }
            ]
        )
    {
        return None;
    }
    Some((TwinSpec { n_in, ops }, sites))
}

fn native_affine_tensor_f32(
    tensor: &TensorProto,
    budget: &ny_propagate::certify::work::WorkBudget,
    at: Instant,
) -> Option<Vec<f64>> {
    if tensor.data_type != 1 || tensor.data_location != 0 {
        return None;
    }
    let raw = !tensor.raw_data.is_empty();
    if raw && !tensor.raw_data.len().is_multiple_of(4) {
        return None;
    }
    let len = if raw {
        tensor.raw_data.len() / 4
    } else {
        tensor.float_data.len()
    };
    let mut values = Vec::new();
    values.try_reserve_exact(len).ok()?;
    if values.capacity() > len {
        return None;
    }
    for i in 0..len {
        super::prepared_root_source::poll(budget, at, i).ok()?;
        let value = if raw {
            f32::from_le_bytes(tensor.raw_data[i * 4..i * 4 + 4].try_into().ok()?)
        } else {
            tensor.float_data[i]
        };
        let value = ny_core::f32_to_f64_exact(value);
        if !value.is_finite() {
            return None;
        }
        values.push(value);
    }
    super::prepared_root_source::checkpoint(budget, at).ok()?;
    Some(values)
}

// The historical decoded-model walk is preserved verbatim. Existing path
// and bytes callers therefore retain its rank-four grammar and numerics.
fn extract_legacy_twin_spec_named_model(model: &ModelProto) -> Option<(TwinSpec, TwinSiteNames)> {
    let g: &GraphProto = model.graph.as_ref()?;
    let inits: HashMap<&str, &TensorProto> =
        g.initializer.iter().map(|t| (t.name.as_str(), t)).collect();

    // Input (C, H, W).
    let inp = g.input.first()?;
    let dims: Vec<i64> = inp
        .r#type
        .as_ref()?
        .tensor_type
        .as_ref()?
        .shape
        .as_ref()?
        .dim
        .iter()
        .map(|d| match &d.value {
            Some(ny_onnx::onnx_proto::tensor_shape_proto::dimension::Value::DimValue(v)) => *v,
            _ => 0,
        })
        .collect();
    if dims.len() != 4 {
        return None;
    }
    let chw = (
        usize::try_from(dims[1]).ok()?,
        usize::try_from(dims[2]).ok()?,
        usize::try_from(dims[3]).ok()?,
    );
    if chw.0 == 0 || chw.1 == 0 || chw.2 == 0 {
        return None;
    }
    let n_in = chw.0 * chw.1 * chw.2;

    // Consumers of activation tensors (for the Conv+BN fold check).
    let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
    for (ni, n) in g.node.iter().enumerate() {
        for i in &n.input {
            if !inits.contains_key(i.as_str()) {
                consumers.entry(i.as_str()).or_default().push(ni);
            }
        }
    }

    // Tensor name -> id; shapes per id.
    let mut tid: HashMap<&str, usize> = HashMap::new();
    tid.insert(inp.name.as_str(), 0);
    let mut shape_chw: Vec<Option<(usize, usize, usize)>> = vec![Some(chw)];
    let mut flat: Vec<usize> = vec![n_in];
    let mut ops: Vec<TwinOpSpec> = Vec::new();
    let mut skip_bn: HashSet<usize> = HashSet::new();
    let mut site_names: TwinSiteNames = Vec::new();
    // Constant nodes referenced as Reshape shape inputs (name -> payload).
    let mut consts: HashMap<&str, &TensorProto> = HashMap::new();

    for (ni, n) in g.node.iter().enumerate() {
        if skip_bn.contains(&ni) {
            continue;
        }
        let out_id = ops.len() + 1;
        match n.op_type.as_str() {
            "Conv" => {
                let in_id = *tid.get(n.input.first()?.as_str())?;
                let ishape = shape_chw.get(in_id).copied().flatten()?;
                let w_t = inits.get(n.input.get(1)?.as_str())?;
                if w_t.dims.len() != 4 {
                    return None;
                }
                let (co, ci, kh, kw) = (
                    usize::try_from(w_t.dims[0]).ok()?,
                    usize::try_from(w_t.dims[1]).ok()?,
                    usize::try_from(w_t.dims[2]).ok()?,
                    usize::try_from(w_t.dims[3]).ok()?,
                );
                let mut weight = tensor_f32(w_t)?;
                if weight.len() != co * ci * kh * kw || ci != ishape.0 {
                    return None;
                }
                let cbias = match n.input.get(2) {
                    Some(name) => {
                        let bt = inits.get(name.as_str())?;
                        let b = tensor_f32(bt)?;
                        if b.len() != co {
                            return None;
                        }
                        Some(b)
                    }
                    None => None,
                };
                let dil = attr_ints(n, "dilations").unwrap_or(&[1, 1]);
                if dil.iter().any(|&d| d != 1) || attr_i(n, "group", 1) != 1 {
                    return None;
                }
                let strides = attr_ints(n, "strides").unwrap_or(&[1, 1]);
                let pads = attr_ints(n, "pads").unwrap_or(&[0, 0, 0, 0]);
                if strides.len() != 2 || pads.len() != 4 {
                    return None;
                }
                let stride = (
                    usize::try_from(strides[0]).ok()?,
                    usize::try_from(strides[1]).ok()?,
                );
                let pads = (
                    usize::try_from(pads[0]).ok()?,
                    usize::try_from(pads[1]).ok()?,
                    usize::try_from(pads[2]).ok()?,
                    usize::try_from(pads[3]).ok()?,
                );
                // Fold the (unique-consumer) BatchNormalization, if present.
                let conv_out = n.output.first()?;
                let mut out_name = conv_out.as_str();
                let (mut bias, mut bias_err, mut w_rel) = (
                    cbias.clone().unwrap_or_else(|| vec![0.0; co]),
                    vec![0.0; co],
                    0.0,
                );
                let cons = consumers.get(conv_out.as_str());
                let bn_idx = cons.and_then(|c| {
                    (c.len() == 1 && g.node[c[0]].op_type == "BatchNormalization").then(|| c[0])
                });
                if let Some(bi) = bn_idx {
                    let bn = &g.node[bi];
                    if bn.input.len() < 5 {
                        return None;
                    }
                    let w_bn = tensor_f32(inits.get(bn.input[1].as_str())?)?;
                    let b_bn = tensor_f32(inits.get(bn.input[2].as_str())?)?;
                    let mean = tensor_f32(inits.get(bn.input[3].as_str())?)?;
                    let var = tensor_f32(inits.get(bn.input[4].as_str())?)?;
                    if [&w_bn, &b_bn, &mean, &var].iter().any(|v| v.len() != co) {
                        return None;
                    }
                    let eps = attr_f(bn, "epsilon", 1e-5);
                    let mut a_c = vec![0.0; co];
                    for ch in 0..co {
                        let t1 = var[ch] + eps;
                        // NaN-rejecting positivity check (NaN compares false).
                        if t1 <= 0.0 || t1.is_nan() {
                            return None;
                        }
                        // Parity with core.py: a_c = w / sqrt(var + eps).
                        a_c[ch] = w_bn[ch] / t1.sqrt();
                        // b_c = b - a_c * mean (+ a_c * conv bias).
                        let m1 = a_c[ch] * mean[ch];
                        let mut b_c = b_bn[ch] - m1;
                        let mut cb_term = 0.0;
                        if let Some(cb) = &cbias {
                            cb_term = a_c[ch] * cb[ch];
                            b_c += cb_term;
                        }
                        bias[ch] = b_c;
                        // Certified absolute error: each of the ~5 rounded ops
                        // contributes <= u * |operand magnitude|.
                        bias_err[ch] = 8.0 * U * (m1.abs() + cb_term.abs() + b_c.abs()) + 1e-300;
                    }
                    for (i, w) in weight.iter_mut().enumerate() {
                        *w *= a_c[i / (ci * kh * kw)];
                    }
                    // Multiplicative chain (add, sqrt, div, mul): rel < 5u.
                    w_rel = 1e-15;
                    skip_bn.insert(bi);
                    out_name = bn.output.first()?.as_str();
                }
                let oh = (ishape.1 + pads.0 + pads.2).checked_sub(kh)? / stride.0 + 1;
                let ow = (ishape.2 + pads.1 + pads.3).checked_sub(kw)? / stride.1 + 1;
                let oshape = (co, oh, ow);
                ops.push(TwinOpSpec::Conv {
                    input: in_id,
                    weight,
                    bias,
                    bias_err,
                    weight_rel_err: w_rel,
                    kernel: (co, ci, kh, kw),
                    stride,
                    pads,
                    ishape,
                    oshape,
                });
                tid.insert(out_name, out_id);
                shape_chw.push(Some(oshape));
                flat.push(co * oh * ow);
            }
            // Standalone BN (not folded into a preceding Conv) becomes a
            // certified per-channel affine (#epoch-bab Phase D). Needs the
            // CHW shape of its input (else fail-closed).
            "BatchNormalization" => {
                let in_id = *tid.get(n.input.first()?.as_str())?;
                let shape = shape_chw.get(in_id).copied().flatten()?;
                let c = shape.0;
                if n.input.len() < 5 {
                    return None;
                }
                let w_bn = tensor_f32(inits.get(n.input[1].as_str())?)?;
                let b_bn = tensor_f32(inits.get(n.input[2].as_str())?)?;
                let mean = tensor_f32(inits.get(n.input[3].as_str())?)?;
                let var = tensor_f32(inits.get(n.input[4].as_str())?)?;
                if [&w_bn, &b_bn, &mean, &var].iter().any(|v| v.len() != c) {
                    return None;
                }
                let eps = attr_f(n, "epsilon", 1e-5);
                let mut scale = vec![0.0; c];
                let mut shift = vec![0.0; c];
                let mut shift_err = vec![0.0; c];
                for ch in 0..c {
                    let t1 = var[ch] + eps;
                    if t1 <= 0.0 || t1.is_nan() {
                        return None;
                    }
                    scale[ch] = w_bn[ch] / t1.sqrt();
                    let m1 = scale[ch] * mean[ch];
                    shift[ch] = b_bn[ch] - m1;
                    // Certified absolute error: ~4 rounded ops on operands of
                    // magnitude |m1| / |shift| (mirrors the Conv+BN fold).
                    shift_err[ch] = 8.0 * U * (m1.abs() + shift[ch].abs()) + 1e-300;
                    if !scale[ch].is_finite() || !shift[ch].is_finite() {
                        return None;
                    }
                }
                ops.push(TwinOpSpec::ChannelAffine {
                    input: in_id,
                    scale,
                    shift,
                    // Multiplicative chain (add, sqrt, div): rel < 5u.
                    scale_rel_err: 1e-15,
                    shift_err,
                    shape,
                });
                tid.insert(n.output.first()?.as_str(), out_id);
                shape_chw.push(Some(shape));
                flat.push(flat[in_id]);
            }
            "ConvTranspose" => {
                let in_id = *tid.get(n.input.first()?.as_str())?;
                let ishape = shape_chw.get(in_id).copied().flatten()?;
                let w_t = inits.get(n.input.get(1)?.as_str())?;
                if w_t.dims.len() != 4 {
                    return None;
                }
                // ONNX ConvTranspose weight layout is [cin][cout][kh][kw];
                // TwinOpSpec wants [cout][cin][kh][kw].
                let (ci, co, kh, kw) = (
                    usize::try_from(w_t.dims[0]).ok()?,
                    usize::try_from(w_t.dims[1]).ok()?,
                    usize::try_from(w_t.dims[2]).ok()?,
                    usize::try_from(w_t.dims[3]).ok()?,
                );
                let w_raw = tensor_f32(w_t)?;
                if w_raw.len() != ci * co * kh * kw || ci != ishape.0 {
                    return None;
                }
                let mut weight = vec![0.0; co * ci * kh * kw];
                for c in 0..ci {
                    for o in 0..co {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                weight[((o * ci + c) * kh + ky) * kw + kx] =
                                    w_raw[((c * co + o) * kh + ky) * kw + kx];
                            }
                        }
                    }
                }
                let cbias = match n.input.get(2) {
                    Some(name) => {
                        let b = tensor_f32(inits.get(name.as_str())?)?;
                        if b.len() != co {
                            return None;
                        }
                        b
                    }
                    None => vec![0.0; co],
                };
                let dil = attr_ints(n, "dilations").unwrap_or(&[1, 1]);
                if dil.iter().any(|&d| d != 1) || attr_i(n, "group", 1) != 1 {
                    return None;
                }
                let strides = attr_ints(n, "strides").unwrap_or(&[1, 1]);
                let pads_a = attr_ints(n, "pads").unwrap_or(&[0, 0, 0, 0]);
                let opad = attr_ints(n, "output_padding").unwrap_or(&[0, 0]);
                if strides.len() != 2 || pads_a.len() != 4 || opad.len() != 2 {
                    return None;
                }
                // output_shape / auto_pad are not modelled: fail closed.
                if attr_ints(n, "output_shape").is_some()
                    || n.attribute.iter().any(|a| a.name == "auto_pad")
                {
                    return None;
                }
                let stride = (
                    usize::try_from(strides[0]).ok()?,
                    usize::try_from(strides[1]).ok()?,
                );
                let pads = (
                    usize::try_from(pads_a[0]).ok()?,
                    usize::try_from(pads_a[1]).ok()?,
                    usize::try_from(pads_a[2]).ok()?,
                    usize::try_from(pads_a[3]).ok()?,
                );
                let out_pad = (
                    usize::try_from(opad[0]).ok()?,
                    usize::try_from(opad[1]).ok()?,
                );
                if stride.0 == 0 || stride.1 == 0 || ishape.1 == 0 || ishape.2 == 0 {
                    return None;
                }
                let oh =
                    ((ishape.1 - 1) * stride.0 + kh + out_pad.0).checked_sub(pads.0 + pads.2)?;
                let ow =
                    ((ishape.2 - 1) * stride.1 + kw + out_pad.1).checked_sub(pads.1 + pads.3)?;
                let oshape = (co, oh, ow);
                ops.push(TwinOpSpec::ConvTranspose {
                    input: in_id,
                    weight,
                    bias: cbias,
                    bias_err: vec![0.0; co],
                    weight_rel_err: 0.0,
                    kernel: (co, ci, kh, kw),
                    stride,
                    pads,
                    ishape,
                    oshape,
                    out_pad,
                });
                tid.insert(n.output.first()?.as_str(), out_id);
                shape_chw.push(Some(oshape));
                flat.push(co * oh * ow);
            }
            "Relu" => {
                let in_id = *tid.get(n.input.first()?.as_str())?;
                site_names.push((ops.len(), n.input.first()?.clone(), flat[in_id]));
                ops.push(TwinOpSpec::Relu { input: in_id });
                tid.insert(n.output.first()?.as_str(), out_id);
                shape_chw.push(shape_chw[in_id]);
                flat.push(flat[in_id]);
            }
            "Add" => {
                let a = *tid.get(n.input.first()?.as_str())?;
                let b = *tid.get(n.input.get(1)?.as_str())?;
                if flat[a] != flat[b] {
                    return None;
                }
                ops.push(TwinOpSpec::Add { lhs: a, rhs: b });
                tid.insert(n.output.first()?.as_str(), out_id);
                shape_chw.push(shape_chw[a]);
                flat.push(flat[a]);
            }
            "Flatten" => {
                let in_id = *tid.get(n.input.first()?.as_str())?;
                ops.push(TwinOpSpec::Flatten { input: in_id });
                tid.insert(n.output.first()?.as_str(), out_id);
                shape_chw.push(None);
                flat.push(flat[in_id]);
            }
            // Constant nodes only feed Reshape shape inputs here; record the
            // payload and emit no op (fail-closed: a Constant consumed as an
            // ACTIVATION would be an unknown tensor id downstream -> None).
            "Constant" => {
                let val = n
                    .attribute
                    .iter()
                    .find(|a| a.name == "value")
                    .and_then(|a| a.t.as_ref())?;
                consts.insert(n.output.first()?.as_str(), val);
            }
            // Reshape with a STATIC element-count-preserving target is a flat
            // identity in this row-major representation (ONNX Reshape is
            // C-order). Track the CHW shape when the target is [1, C, H, W]
            // (a conv may follow, e.g. the cgan generator); else None.
            "Reshape" => {
                let in_id = *tid.get(n.input.first()?.as_str())?;
                let shape_name = n.input.get(1)?.as_str();
                let shape_t = inits
                    .get(shape_name)
                    .copied()
                    .or_else(|| consts.get(shape_name).copied())?;
                let dims = tensor_i64(shape_t)?;
                // Resolve: 0 = copy input dim (only allowed at axis 0 with
                // batch 1 here), -1 = infer; all others positive.
                let mut resolved: Vec<i64> = Vec::with_capacity(dims.len());
                let mut infer = None;
                for (ax, &d) in dims.iter().enumerate() {
                    match d {
                        -1 => {
                            if infer.is_some() {
                                return None;
                            }
                            infer = Some(ax);
                            resolved.push(1);
                        }
                        0 => {
                            if ax != 0 {
                                return None;
                            }
                            resolved.push(1);
                        }
                        d if d > 0 => resolved.push(d),
                        _ => return None,
                    }
                }
                let known: i64 = resolved.iter().product();
                let total = i64::try_from(flat[in_id]).ok()?;
                if let Some(ax) = infer {
                    if known == 0 || total % known != 0 {
                        return None;
                    }
                    resolved[ax] = total / known;
                }
                if resolved.iter().product::<i64>() != total {
                    return None;
                }
                ops.push(TwinOpSpec::Flatten { input: in_id });
                tid.insert(n.output.first()?.as_str(), out_id);
                let chw_out = if resolved.len() == 4 && resolved[0] == 1 {
                    Some((
                        usize::try_from(resolved[1]).ok()?,
                        usize::try_from(resolved[2]).ok()?,
                        usize::try_from(resolved[3]).ok()?,
                    ))
                } else {
                    None
                };
                shape_chw.push(chw_out);
                flat.push(flat[in_id]);
            }
            "Gemm" => {
                let in_id = *tid.get(n.input.first()?.as_str())?;
                if (attr_f(n, "alpha", 1.0) - 1.0).abs() > 0.0
                    || (attr_f(n, "beta", 1.0) - 1.0).abs() > 0.0
                    || attr_i(n, "transA", 0) != 0
                {
                    return None;
                }
                let w_t = inits.get(n.input.get(1)?.as_str())?;
                if w_t.dims.len() != 2 {
                    return None;
                }
                let (d0, d1) = (
                    usize::try_from(w_t.dims[0]).ok()?,
                    usize::try_from(w_t.dims[1]).ok()?,
                );
                let w_raw = tensor_f32(w_t)?;
                if w_raw.len() != d0 * d1 {
                    return None;
                }
                let trans_b = attr_i(n, "transB", 0) != 0;
                let (weight, no, nin) = if trans_b {
                    (w_raw, d0, d1)
                } else {
                    // W stored (n_in, n_out): transpose to (n_out, n_in).
                    let mut w = vec![0.0; d0 * d1];
                    for i in 0..d0 {
                        for j in 0..d1 {
                            w[j * d0 + i] = w_raw[i * d1 + j];
                        }
                    }
                    (w, d1, d0)
                };
                if nin != flat[in_id] {
                    return None;
                }
                let bias = match n.input.get(2) {
                    Some(name) => {
                        let b = tensor_f32(inits.get(name.as_str())?)?;
                        if b.len() != no {
                            return None;
                        }
                        b
                    }
                    None => vec![0.0; no],
                };
                // #cct-margin-row-admission: fold a unique-consumer
                // BatchNormalization into this Gemm (CNN7's FC-BN head),
                // WITH its certified error (`dense_bn_fold`), exactly as the
                // Conv arm folds a conv BN. Anything the fold cannot certify
                // declines the whole net (fail-closed: the lane does not run).
                let gemm_out = n.output.first()?;
                let bn_idx = consumers.get(gemm_out.as_str()).and_then(|c| {
                    (c.len() == 1 && g.node[c[0]].op_type == "BatchNormalization").then(|| c[0])
                });
                let (weight, bias, bias_err, weight_rel_err, out_name) = match bn_idx {
                    None => (weight, bias, Vec::new(), 0.0, gemm_out.as_str()),
                    Some(bi) => {
                        let (fold, bn_out) = fold_gemm_batchnorm(
                            &g.node[bi],
                            gemm_out.as_str(),
                            &inits,
                            &weight,
                            &bias,
                            (no, nin),
                        )?;
                        eprintln!(
                            "[margin-row] fc-bn fold: Gemm node {ni} ({no}x{nin}) + \
                             BatchNormalization node {bi} folded with certified error \
                             (weight_rel_err={:.3e}, max bias_err={:.3e})",
                            fold.weight_rel_err,
                            fold.bias_err.iter().copied().fold(0.0_f64, f64::max),
                        );
                        skip_bn.insert(bi);
                        (
                            fold.weight,
                            fold.bias,
                            fold.bias_err,
                            fold.weight_rel_err,
                            bn_out,
                        )
                    }
                };
                ops.push(TwinOpSpec::Gemm {
                    input: in_id,
                    weight,
                    bias,
                    bias_err,
                    weight_rel_err,
                    shape: (no, nin),
                });
                tid.insert(out_name, out_id);
                shape_chw.push(None);
                flat.push(no);
            }
            other => {
                if std::env::var("NY_MARGIN_ROW_DEBUG").is_ok() {
                    eprintln!(
                        "margin-row extract_twin_spec: UNSUPPORTED op '{other}' at node {ni} \
                         (in={:?} out={:?})",
                        n.input, n.output
                    );
                }
                return None;
            }
        }
    }
    Some((TwinSpec { n_in, ops }, site_names))
}

// --------------------------------------------------------------------------
//  vnnlib robustness parser (parity port of core.py::parse_vnnlib)
// --------------------------------------------------------------------------

/// NCHW `(H, W)` from a `(declare-input X float32 [1, C, H, W])` header, needed to
/// flatten VNNLIB-2.0 bracket indices `X[0,c,h,w]` into the net's row-major input.
/// `None` when there is no 4-D declare-input (the flat `X_<idx>` form, where the
/// header is unused).
fn parse_declare_input_hw(text: &str) -> Option<(usize, usize)> {
    for line in text.lines() {
        let l = line.trim();
        if l.starts_with("(declare-input") {
            let lb = l.find('[')?;
            let rb = l[lb..].find(']')? + lb;
            let dims: Vec<usize> = l[lb + 1..rb]
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if dims.len() == 4 {
                return Some((dims[2], dims[3]));
            }
        }
    }
    None
}

/// Row-major flat index for a bracket tuple `"0,c,h,w"` given the NCHW `(H, W)`:
/// `c*H*W + h*W + w`. Matches the net's Flatten ordering in [`extract_twin_spec`].
fn flat_index_nchw(idx_str: &str, hw: Option<(usize, usize)>) -> Option<usize> {
    let parts: Vec<usize> = idx_str
        .split(',')
        .map(str::trim)
        .filter_map(|s| s.parse().ok())
        .collect();
    if parts.len() != 4 {
        return None;
    }
    let (h, w) = hw?;
    Some(parts[1] * h * w + parts[2] * w + parts[3])
}

/// Last comma-separated index in a bracket tuple, e.g. `"0,142"` -> `142`.
fn last_bracket_index(s: &str) -> Option<usize> {
    s.split(',').next_back()?.trim().parse().ok()
}

/// #cct-margin-row-admission: one output operand of a robustness disjunct —
/// flat `Y_<j>` or VNNLIB-2.0 bracket `Y[0,<j>]` — and the rest of the line.
/// Strict: the flat form needs at least one digit, the bracket form exactly
/// two indices with a leading `0` (the `[1, n_out]` output declaration).
fn parse_y_operand(s: &str) -> Option<(usize, &str)> {
    let s = s.trim_start();
    if let Some(rest) = s.strip_prefix("Y_") {
        let end = rest
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len());
        if end == 0 {
            return None;
        }
        Some((rest[..end].parse().ok()?, &rest[end..]))
    } else if let Some(rest) = s.strip_prefix("Y[") {
        let (idx, after) = rest.split_once(']')?;
        let parts: Vec<&str> = idx.split(',').map(str::trim).collect();
        if parts.len() != 2 || parts[0] != "0" {
            return None;
        }
        Some((parts[1].parse().ok()?, after))
    } else {
        None
    }
}

/// #cct-margin-row-admission: the `<=` spelling of a robustness disjunct,
/// `(and (<= Y_t Y_j))` (VNNLIB 1.0) or `(and (<= Y[0,t] Y[0,j]))` (2.0) —
/// the counterexample region "the TRUE class `t` does not beat class `j`",
/// so the true class is the LEFT operand. Returns `(t, j)`.
///
/// `None` unless the line is EXACTLY one such comparison between two DISTINCT
/// outputs: a conjunction of several comparisons, a constant operand, or any
/// trailing content is a different property, and the caller fails closed on
/// it (the trusted-parser cross-check would refuse it too).
fn parse_le_robustness_disjunct(line: &str) -> Option<(usize, usize)> {
    let rest = line.strip_prefix("(and (<=")?;
    let (t, rest) = parse_y_operand(rest)?;
    let (j, rest) = parse_y_operand(rest)?;
    let tail: String = rest.chars().filter(|c| !c.is_whitespace()).collect();
    if tail != "))" || t == j {
        return None;
    }
    Some((t, j))
}

/// Parse the cifar100/tinyimagenet robustness form: per-pixel X bounds and a
/// disjunction of `(and (>= Y_j Y_t))` clauses. Returns `(lo, hi, t, adv)`.
///
/// Accepts BOTH VNNLIB encodings of the identical property: the flat 2025 form
/// (`X_<flatidx>`, `Y_<j>`) and the VNNLIB-2.0 bracket form (`X[0,c,h,w]`,
/// `Y[0,j]`). Bracket input indices are flattened row-major (NCHW) using the
/// `declare-input` header, which is exactly the net's Flatten ordering, so the two
/// encodings yield identical `(lo, hi, t, adv)`.
///
/// #cct-margin-row-admission: also accepts the `<=` spelling the
/// challenging_certified_training corpora (and relusplitter's oval21 rows)
/// use, `(and (<= Y_t Y_j))` / `(and (<= Y[0,t] Y[0,j]))`, where the TRUE class
/// is the LEFT operand ([`parse_le_robustness_disjunct`]). The same
/// counterexample region as `(and (>= Y_j Y_t))`, and it passes through the
/// SAME trusted-parser cross-check below (which maps `LessEq(a, b)` to
/// adversarial `b`, true `a`), so a mis-ordered operand pair can only ever
/// fail closed.
#[allow(clippy::type_complexity)]
pub(crate) fn parse_vnnlib_robustness(
    path: &Path,
) -> Option<(Vec<f64>, Vec<f64>, usize, Vec<usize>)> {
    let text = std::fs::read_to_string(path).ok()?;
    let hw = parse_declare_input_hw(&text);
    let mut lo: HashMap<usize, f64> = HashMap::new();
    let mut hi: HashMap<usize, f64> = HashMap::new();
    let mut adv: Vec<usize> = Vec::new();
    let mut true_c: HashSet<usize> = HashSet::new();
    let mut le_disjuncts = 0usize;
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line
            .strip_prefix("(assert (<= X_")
            .or_else(|| line.strip_prefix("(assert (>= X_"))
        {
            // Flat 2025 form: `X_<flatidx>`.
            let upper = line.starts_with("(assert (<=");
            let mut parts = rest.split_whitespace();
            let idx: usize = parts.next()?.parse().ok()?;
            let val_s = parts.next()?;
            let val: f64 = val_s.trim_end_matches(')').parse().ok()?;
            if upper {
                hi.insert(idx, val);
            } else {
                lo.insert(idx, val);
            }
        } else if let Some(rest) = line
            .strip_prefix("(assert (<= X[")
            .or_else(|| line.strip_prefix("(assert (>= X["))
        {
            // VNNLIB-2.0 bracket form: `X[0,c,h,w] <val>`.
            let upper = line.starts_with("(assert (<=");
            let (idx_str, after) = rest.split_once(']')?;
            let idx = flat_index_nchw(idx_str, hw)?;
            let val: f64 = after
                .split_whitespace()
                .next()?
                .trim_end_matches(')')
                .parse()
                .ok()?;
            if upper {
                hi.insert(idx, val);
            } else {
                lo.insert(idx, val);
            }
        } else if let Some(rest) = line.strip_prefix("(and (>= Y_") {
            // Flat form: `(and (>= Y_j Y_t))`.
            let mut parts = rest.split_whitespace();
            let j: usize = parts.next()?.parse().ok()?;
            let t_s = parts.next()?;
            let t: usize = t_s
                .trim_start_matches("Y_")
                .trim_end_matches(')')
                .parse()
                .ok()?;
            adv.push(j);
            true_c.insert(t);
        } else if let Some(rest) = line.strip_prefix("(and (>= Y[") {
            // Bracket form: `(and (>= Y[0,j] Y[0,t]))`.
            let (j_str, after) = rest.split_once(']')?;
            let j = last_bracket_index(j_str)?;
            let t_inner = after.trim_start().strip_prefix("Y[")?;
            let (t_str, _) = t_inner.split_once(']')?;
            let t = last_bracket_index(t_str)?;
            adv.push(j);
            true_c.insert(t);
        } else if line.starts_with("(and (<= Y") {
            // #cct-margin-row-admission: `(and (<= Y_t Y_j))` /
            // `(and (<= Y[0,t] Y[0,j]))` — TRUE class on the LEFT. A line
            // that starts like a `<=` output disjunct but is not exactly one
            // clean comparison fails the whole parse closed.
            let (t, j) = parse_le_robustness_disjunct(line)?;
            adv.push(j);
            true_c.insert(t);
            le_disjuncts += 1;
        }
    }
    if lo.is_empty() || lo.len() != hi.len() || true_c.len() != 1 || adv.is_empty() {
        return None;
    }
    let n = lo.keys().max()? + 1;
    if lo.len() != n {
        return None;
    }
    let mut lov = vec![0.0; n];
    let mut hiv = vec![0.0; n];
    for i in 0..n {
        lov[i] = *lo.get(&i)?;
        hiv[i] = *hi.get(&i)?;
        if lov[i] > hiv[i] || lov[i].is_nan() || hiv[i].is_nan() {
            return None;
        }
    }
    adv.sort_unstable();
    adv.dedup();
    let t = true_c.into_iter().next()?;

    // SOUNDNESS GATE (#twinwall): the margin-row lane emits a CERTIFIED `unsat`, so a
    // mis-flattened or permuted input box parsed here would be a false verdict. Cross-
    // check the whole parsed box against ny's PRODUCTION VNNLIB parser (`load_vnnlib`,
    // the parser that drives the real verdicts and is fuzzed/tested for both the flat
    // 2025 and bracket 2.0 encodings) and FAIL-CLOSED on any divergence: the lane then
    // simply does not fire. This makes the fast hand-rolled parser above safe — its
    // certified output is trusted only when an independent, authoritative parser agrees
    // element-for-element. Class indices are additionally range-checked.
    match ny_onnx::vnnlib::load_vnnlib(path) {
        Ok(spec) => {
            if spec.input_bounds.len() != n {
                return None;
            }
            for (i, &(bl, bu)) in spec.input_bounds.iter().enumerate() {
                if (bl - lov[i]).abs() > 1e-6 || (bu - hiv[i]).abs() > 1e-6 {
                    return None; // input-box disagreement — never emit a certified unsat
                }
            }
            if t >= spec.num_outputs || adv.iter().any(|&j| j >= spec.num_outputs) {
                return None;
            }
            // OBLIGATION 2 — OUTPUT-SIDE fail-closed. The input-box cross-check
            // above proves the parsed X region matches the trusted parser, but
            // says NOTHING about the output property: a hand parser that silently
            // DROPS a `(and (>= Y_j Y_t))` disjunct would emit a certified `unsat`
            // against FEWER adversarial classes than the spec actually forbids —
            // a false verdict. Reconstruct (true class, adversarial set) from the
            // trusted spec's output constraints and require it to match `(t, adv)`
            // EXACTLY. Any divergence, or an output form that is not the recognized
            // robustness OR-of-single-comparisons, fails closed (lane does not fire).
            use ny_onnx::vnnlib::OutputConstraint;
            if !spec.is_disjunction {
                return None; // robustness unsafe region is an OR of disjuncts
            }
            if spec.output_constraint_clauses.is_empty() {
                return None; // no reconstructable disjuncts — fail closed
            }
            let mut rec_true: Option<usize> = None;
            let mut rec_adv: HashSet<usize> = HashSet::new();
            for clause in &spec.output_constraint_clauses {
                // A robustness disjunct is a SINGLE comparison Y_j vs Y_t.
                if clause.len() != 1 {
                    return None;
                }
                let (adv_class, true_class) = match &clause[0] {
                    // Y_a >= Y_t / Y_a > Y_t: a is adversarial, t is true.
                    OutputConstraint::GreaterEq(a, b) | OutputConstraint::GreaterThan(a, b) => {
                        (*a, *b)
                    }
                    // Y_t <= Y_a / Y_t < Y_a: a is adversarial, t is true.
                    OutputConstraint::LessEq(a, b) | OutputConstraint::LessThan(a, b) => (*b, *a),
                    // Constant/threshold forms are NOT robustness disjuncts.
                    _ => return None,
                };
                if adv_class == true_class {
                    return None; // degenerate Y_t vs Y_t
                }
                match rec_true {
                    None => rec_true = Some(true_class),
                    Some(tc) if tc != true_class => return None, // inconsistent true class
                    _ => {}
                }
                if !rec_adv.insert(adv_class) {
                    return None; // duplicate adversarial disjunct — not the clean OR form
                }
            }
            let rec_true = rec_true?;
            if rec_true != t {
                return None; // reconstructed true class disagrees with hand parse
            }
            // Set-equality of the adversarial set: a hand parse capturing a strict
            // SUBSET (or superset) of the trusted disjuncts diverges here → None.
            let adv_set: HashSet<usize> = adv.iter().copied().collect();
            if rec_adv != adv_set {
                return None;
            }
        }
        Err(_) => return None, // trusted parser could not read it — fail closed
    }

    // #cct-margin-row-admission observable: the `<=` spelling was admitted,
    // and only AFTER both cross-checks passed. Silent on the `>=` corpora
    // (cifar100 / tinyimagenet), whose output is unchanged.
    if le_disjuncts > 0 {
        eprintln!(
            "[margin-row] vnnlib robustness spec: form=le (true class LEFT of <=) \
             disjuncts={le_disjuncts} true_class={t} adversarial={} — trusted-parser \
             cross-check passed",
            adv.len()
        );
    }
    Some((lov, hiv, t, adv))
}

/// #cct-margin-row-admission pins: the `<=` robustness spelling, the certified
/// FC-BN fold, and the root-tableau admission ceiling.
#[cfg(test)]
mod cct_admission_tests;

/// #mr-ibp-seed real-ONNX gates on the cct IBP-trained eps8 CNN7s: seeded-box
/// containment of ORT-cross-checked samples (5323, 3144w) and the field-SAT
/// negative controls (8090, 8397). They need the gitignored corpus, ORT and a
/// release build, so they are in the explicit `external-vnncomp` lane (run it in
/// release with the corpus symlinked); a missing prerequisite is VOID and fails.
#[cfg(all(test, feature = "external-vnncomp"))]
mod ibp_seed_real_tests;

/// `col-scaling-v2`: the margin-row pass-cost probe v2 (throughput plan B1(b)).
/// Research-only; reached solely through `ny vnncomp-research margin-row`.
mod col_scaling_v2;

// --------------------------------------------------------------------------
//  Differential gates vs the verified Python reference (#twinwall INC2a/b/c).
//  Real-corpus measurements are exposed through the explicit
//  `ny vnncomp-research margin-row` lane and must run serially.
// --------------------------------------------------------------------------

pub(crate) mod research {
    use super::*;
    use ny_propagate::margin_row::bab::{BabConfig, MarginRowBab};
    use ny_propagate::margin_row::bounds::{
        compose_viay, head_gates, margin_seed, per_class_direct, row_dots, MarginBatch, YBox,
    };
    use ny_propagate::margin_row::engine::{domain_gates, BackwardEngine, LaneDir, Seed};
    use ny_propagate::margin_row::{RootGates, RoundMode, TwinNet};

    // --- Hard slice cap (banked 99ed4d42): the inline lane must RETURN at the
    // cap even when the underlying work ignores its cooperative deadline
    // entirely (the metaroom overrun class: 113.1s against a 45s slice, one
    // run killed by the watchdog at budget+grace). ---
    #[test]
    fn capped_detached_abandons_slow_work_at_the_cap() {
        let t0 = Instant::now();
        // Worker sleeps 30s — a stand-in for a lane whose internal deadline
        // checks are too coarse to fire. The cap is 200ms.
        let out = run_capped_detached(Duration::from_millis(200), "test-slow", || {
            std::thread::sleep(Duration::from_secs(30));
            42_u32
        });
        let elapsed = t0.elapsed();
        // Fail-closed abandon: no value, and the caller got control back at
        // the cap (generous 5s bound for CI schedulers), NOT after 30s.
        assert_eq!(out, DetachedTake::StillRunning);
        assert!(
            elapsed < Duration::from_secs(5),
            "cap not enforced: returned after {elapsed:?}"
        );
    }

    #[test]
    fn capped_detached_passes_through_fast_work() {
        let out = run_capped_detached(Duration::from_secs(30), "test-fast", || 7_u32);
        assert_eq!(out, DetachedTake::Finished(7));
    }

    #[test]
    fn capped_detached_panicking_worker_is_fail_closed() {
        let out = run_capped_detached(Duration::from_secs(30), "test-panic", || -> u32 {
            panic!("worker panic must map to None, never a verdict")
        });
        assert_eq!(out, DetachedTake::Failed);
    }

    #[test]
    fn concurrent_lane_joins_completed_worker_before_reporting_finished() {
        let (tx, rx) = std::sync::mpsc::channel();
        let epilogue_finished = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_flag = std::sync::Arc::clone(&epilogue_finished);
        let handle = std::thread::spawn(move || {
            tx.send(Some(VnncompResult::test_unsat()))
                .expect("receiver alive");
            std::thread::sleep(Duration::from_millis(20));
            worker_flag.store(true, std::sync::atomic::Ordering::Release);
        });
        let lane = ConcurrentLane {
            rx,
            handle,
            execution: std::sync::Arc::new(ConcurrentLaneExecutionState::default()),
        };
        assert!(matches!(
            lane.take(Duration::from_secs(1)),
            ConcurrentLaneTake::Verdict(VnncompResult::Unsat(_))
        ));
        assert!(
            epilogue_finished.load(std::sync::atomic::Ordering::Acquire),
            "a finished result must mean the worker and its captured graph are gone"
        );
    }

    #[test]
    fn concurrent_lane_joins_disconnected_worker_fail_closed() {
        let (tx, rx) = std::sync::mpsc::channel::<Option<VnncompResult>>();
        let handle = std::thread::spawn(move || drop(tx));
        let lane = ConcurrentLane {
            rx,
            handle,
            execution: std::sync::Arc::new(ConcurrentLaneExecutionState::default()),
        };
        assert!(matches!(
            lane.take(Duration::from_secs(1)),
            ConcurrentLaneTake::FinishedWithoutVerdict
        ));
    }

    #[test]
    fn concurrent_lane_timeout_remains_distinct_from_completion() {
        let (tx, rx) = std::sync::mpsc::channel::<Option<VnncompResult>>();
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            let _ = tx.send(None);
        });
        let lane = ConcurrentLane {
            rx,
            handle,
            execution: std::sync::Arc::new(ConcurrentLaneExecutionState::default()),
        };
        assert!(matches!(
            lane.take(Duration::from_millis(5)),
            ConcurrentLaneTake::StillRunning
        ));
    }

    // --- Bracket/flat VNNLIB index mapping (self-contained, no external files) ---
    // Guards the soundness-critical flatten used to parse VNNLIB-2.0 bracket specs
    // into the net's row-major NCHW input order (a mis-flatten would be a false unsat).
    #[test]
    fn bracket_flatten_is_row_major_nchw() {
        // tinyimagenet: C=3, H=W=56 -> n_in = 9408.
        assert_eq!(flat_index_nchw("0,0,0,0", Some((56, 56))), Some(0));
        assert_eq!(flat_index_nchw("0,1,0,0", Some((56, 56))), Some(56 * 56));
        assert_eq!(flat_index_nchw("0,0,1,0", Some((56, 56))), Some(56));
        assert_eq!(flat_index_nchw("0,0,0,1", Some((56, 56))), Some(1));
        assert_eq!(flat_index_nchw("0,2,55,55", Some((56, 56))), Some(9407));
        // cifar100: C=3, H=W=32 -> n_in = 3072; last pixel = 3071.
        assert_eq!(flat_index_nchw("0,2,31,31", Some((32, 32))), Some(3071));
        // Missing shape / wrong arity -> None (fail-closed, no unsat).
        assert_eq!(flat_index_nchw("0,2,31,31", None), None);
        assert_eq!(flat_index_nchw("0,2,31", Some((32, 32))), None);
    }

    #[test]
    fn declare_input_hw_and_class_index() {
        let hdr = "(declare-input  X float32 [1, 3, 56, 56])";
        assert_eq!(parse_declare_input_hw(hdr), Some((56, 56)));
        assert_eq!(
            parse_declare_input_hw("(declare-input X float32 [1, 3, 32, 32])"),
            Some((32, 32))
        );
        assert_eq!(parse_declare_input_hw("no header here"), None);
        assert_eq!(last_bracket_index("0,142"), Some(142));
        assert_eq!(last_bracket_index("0,0"), Some(0));
    }

    /// OBLIGATION 2 (output-side fail-closed): a hand parse that captures a
    /// STRICT SUBSET of the trusted spec's adversarial disjuncts must return
    /// `None` (never a spec), because a certified `unsat` against fewer
    /// adversarial classes than the property forbids is a FALSE verdict.
    ///
    /// The `(and (< Y_5 Y_2))` disjunct below is semantically a real robustness
    /// disjunct (Y_5 < Y_2, i.e. class 2 beats true class 5), and the trusted
    /// `load_vnnlib` parser records it — but the fast hand parser matches only
    /// the NON-strict spellings (`(and (>= Y_j Y_t))`, and since
    /// #cct-margin-row-admission `(and (<= Y_t Y_j))`), so it silently DROPS a
    /// strict one. The output-side set-equality cross-check must catch the
    /// divergence and fail closed. (This test used the `<=` spelling as its
    /// dropped disjunct until the parser learned it; the strict `<` keeps the
    /// obligation exercised.)
    #[test]
    fn output_side_subset_disjunct_fails_closed() {
        let dir = std::env::temp_dir();
        let stamp = format!(
            "{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );

        let header = "\
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(declare-const Y_3 Real)
(declare-const Y_4 Real)
(declare-const Y_5 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (>= X_1 0.0))
(assert (<= X_1 1.0))
";

        // POSITIVE CONTROL: all three disjuncts in the `>=` form the hand parser
        // recognizes, so hand parse == trusted spec == {adv 0,1,2 vs true 5}.
        let full = format!(
            "{header}(assert (or\n\
             (and (>= Y_0 Y_5))\n\
             (and (>= Y_1 Y_5))\n\
             (and (>= Y_2 Y_5))\n\
             ))\n"
        );
        let full_path = dir.join(format!("mr_full_{stamp}.vnnlib"));
        std::fs::write(&full_path, full).unwrap();
        let full_parsed = parse_vnnlib_robustness(&full_path);
        std::fs::remove_file(&full_path).ok();
        let (_, _, t, adv) =
            full_parsed.expect("full (matching) disjunction should parse to a spec");
        assert_eq!(t, 5, "true class");
        assert_eq!(adv, vec![0, 1, 2], "adversarial set");

        // SUBSET CASE: the third disjunct is written `(< Y_5 Y_2)` — a real
        // disjunct load_vnnlib captures, but the hand parser (matching only the
        // non-strict `>=` / `<=` spellings) drops, so its adv set is the strict
        // subset {0,1}.
        let subset = format!(
            "{header}(assert (or\n\
             (and (>= Y_0 Y_5))\n\
             (and (>= Y_1 Y_5))\n\
             (and (< Y_5 Y_2))\n\
             ))\n"
        );
        let subset_path = dir.join(format!("mr_subset_{stamp}.vnnlib"));
        std::fs::write(&subset_path, subset).unwrap();
        let subset_parsed = parse_vnnlib_robustness(&subset_path);
        std::fs::remove_file(&subset_path).ok();
        assert!(
            subset_parsed.is_none(),
            "hand parse dropped a disjunct (captured strict subset {{0,1}} of the trusted \
             {{0,1,2}}); the output-side cross-check must fail closed, got {subset_parsed:?}"
        );
    }

    const BENCH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../benchmarks/vnncomp2025/benchmarks/cifar100_2024"
    );
    const BENCH_ROOT: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../benchmarks/vnncomp2025/benchmarks"
    );

    /// Build an instance from an explicit onnx + vnnlib path pair.
    fn instance_at(onnx: &str, vnnlib: &str) -> (TwinSpec, Vec<f64>, Vec<f64>, usize, Vec<usize>) {
        let spec = extract_twin_spec(Path::new(onnx)).expect("extract twin spec");
        let (lo, hi, t, adv) = parse_vnnlib_robustness(Path::new(vnnlib)).expect("parse vnnlib");
        (spec, lo, hi, t, adv)
    }

    /// PROFILE / MEASURE harness driven by env (works for cifar100 +
    /// tinyimagenet). Set NY_MARGIN_ROW_PROFILE=1 for the phase breakdown.
    ///   NY_PROBE_CAT   = cifar100_2024 (default) | tinyimagenet_2024
    ///   NY_PROBE_ONNX  = onnx basename (default per category)
    ///   NY_PROBE_VNNLIB= vnnlib basename (required)
    ///   NY_PROBE_SECS  = wall budget (default 100)
    pub(crate) fn probe_env_instance() {
        let cat = std::env::var("NY_PROBE_CAT").unwrap_or_else(|_| "cifar100_2024".into());
        let default_onnx = if cat.starts_with("tiny") {
            "TinyImageNet_resnet_medium.onnx"
        } else {
            "CIFAR100_resnet_medium.onnx"
        };
        let onnx_base = std::env::var("NY_PROBE_ONNX").unwrap_or_else(|_| default_onnx.into());
        let vnnlib_base = std::env::var("NY_PROBE_VNNLIB").expect("set NY_PROBE_VNNLIB");
        let secs: u64 = std::env::var("NY_PROBE_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);
        let onnx = format!("{BENCH_ROOT}/{cat}/onnx/{onnx_base}");
        let vnnlib = format!("{BENCH_ROOT}/{cat}/vnnlib/{vnnlib_base}");
        let (spec, lo, hi, t, adv) = instance_at(&onnx, &vnnlib);
        let net = TwinNet::compile(&spec).expect("compile");
        eprintln!(
            "[probe] {cat}/{vnnlib_base}: n_in={} n_y={} n_out={} trunk_relus={} adv={} budget={secs}s cores={}",
            net.n_in, net.n_y, net.n_out, net.trunk_relus.len(), adv.len(),
            rayon::current_num_threads()
        );
        let t0 = Instant::now();
        let deadline = t0 + Duration::from_secs(secs);
        let out = run_margin_row_lane(&spec, &lo, &hi, t, &adv, Some(deadline), 20_000);
        let el = t0.elapsed().as_secs_f64();
        match out {
            MarginRowOutcome::Unsat(s, _) => eprintln!(
                "[probe] {vnnlib_base} -> UNSAT in {el:.1}s (root={:.4} exp={} domains={} closed={} maxD={} dips={} stop={})",
                s.root_bound, s.expansions, s.domains_created, s.closed, s.max_depth, s.mono_raw_dips, s.stop
            ),
            MarginRowOutcome::Unknown { reason, stats } => eprintln!(
                "[probe] {vnnlib_base} -> UNKNOWN({reason}) in {el:.1}s ({:?})",
                stats.map(|s| (s.root_bound, s.expansions, s.domains_created, s.max_depth, s.stop))
            ),
        }
    }

    /// ROOT-BUILD profile + ORACLE digest. Times `RootGates::build` (median of
    /// NY_ROOT_REPS, default 3), prints the per-op internal breakdown when
    /// NY_MARGIN_ROW_ROOT_TIMING=1, and emits a bit-exact FNV digest of EVERY
    /// frozen gate array (mid/rad/xabs + per-layer l/u/alpha/s/c/ms/unst) plus
    /// the verdict-feeding root_bound. Compare the digest across code versions:
    /// EQUAL => the entire downstream is bit-identical (strongest oracle). Set
    /// `NY_ROOT_DUMP=<path>` to also write every (l,u) as LE f64 for an
    /// outward-enclosure check (BLAS path). NY_ROOT_BLAS=1 selects the DGEMM
    /// forward-conv (provably-outward, not bit-identical).
    ///   NY_PROBE_CAT / NY_PROBE_ONNX / NY_PROBE_VNNLIB, NY_ROOT_REPS
    pub(crate) fn probe_root_build() {
        use ny_propagate::margin_row::bab::root_eval;
        let cat = std::env::var("NY_PROBE_CAT").unwrap_or_else(|_| "cifar100_2024".into());
        let default_onnx = if cat.starts_with("tiny") {
            "TinyImageNet_resnet_medium.onnx"
        } else {
            "CIFAR100_resnet_medium.onnx"
        };
        let onnx_base = std::env::var("NY_PROBE_ONNX").unwrap_or_else(|_| default_onnx.into());
        let vnnlib_base = std::env::var("NY_PROBE_VNNLIB").expect("set NY_PROBE_VNNLIB");
        let reps: usize = std::env::var("NY_ROOT_REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let onnx = format!("{BENCH_ROOT}/{cat}/onnx/{onnx_base}");
        let vnnlib = format!("{BENCH_ROOT}/{cat}/vnnlib/{vnnlib_base}");
        let (spec, lo, hi, t, adv) = instance_at(&onnx, &vnnlib);
        let net = TwinNet::compile(&spec).expect("compile");
        eprintln!(
            "[rootbuild] {cat}/{vnnlib_base}: n_in={} n_y={} n_out={} trunk_relus={} cores={}",
            net.n_in,
            net.n_y,
            net.n_out,
            net.trunk_relus.len(),
            rayon::current_num_threads()
        );
        // ONE warm build (also emits the per-op breakdown when profiling is on).
        // The digest + root_bound below come from THIS build and are printed
        // BEFORE the timing reps, so a wall-capped run still yields the oracle.
        let t_warm = Instant::now();
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let warm_secs = t_warm.elapsed().as_secs_f64();
        eprintln!("[rootbuild] warm build wall {warm_secs:.3}s");
        // Bit-exact digest of the full frozen-gate product (FNV-1a over bits).
        fn fnv(mut h: u64, xs: impl Iterator<Item = u64>) -> u64 {
            for b in xs {
                h ^= b;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            h
        }
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        h = fnv(h, root.mid.iter().map(|v| v.to_bits()));
        h = fnv(h, root.rad.iter().map(|v| v.to_bits()));
        h = fnv(h, root.xabs.iter().map(|v| v.to_bits()));
        let mut total_neurons = 0usize;
        for lg in &root.layers {
            total_neurons += lg.n;
            h = fnv(h, lg.l.iter().map(|v| v.to_bits()));
            h = fnv(h, lg.u.iter().map(|v| v.to_bits()));
            h = fnv(h, lg.alpha.iter().map(|v| v.to_bits()));
            h = fnv(h, lg.s.iter().map(|v| v.to_bits()));
            h = fnv(h, lg.c.iter().map(|v| v.to_bits()));
            h = fnv(h, lg.ms.iter().map(|v| v.to_bits()));
            h = fnv(h, lg.unst.iter().map(|&i| i as u64));
        }
        eprintln!(
            "[rootbuild] GATE_DIGEST = {h:#018x}  layers={} total_neurons={total_neurons}",
            root.layers.len()
        );
        for (i, lg) in root.layers.iter().enumerate() {
            let minl = lg.l.iter().copied().fold(f64::INFINITY, f64::min);
            let maxu = lg.u.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            eprintln!(
                "[rootbuild]   L{i:>2} n={:>7} unst={:>7} minl={minl:>12.6} maxu={maxu:>12.6}",
                lg.n,
                lg.unst.len()
            );
        }
        if let Ok(path) = std::env::var("NY_ROOT_DUMP") {
            use std::io::Write;
            let mut f = std::io::BufWriter::new(std::fs::File::create(&path).expect("dump create"));
            for lg in &root.layers {
                for &v in &lg.l {
                    f.write_all(&v.to_le_bytes()).unwrap();
                }
                for &v in &lg.u {
                    f.write_all(&v.to_le_bytes()).unwrap();
                }
            }
            f.flush().unwrap();
            eprintln!("[rootbuild] dumped per-layer l|u (LE f64) to {path}");
        }
        // Verdict-feeding root_bound (ties RootGates to the moat number).
        let eng = BackwardEngine::new(&net, &root);
        match root_eval(&eng, &net, t, &adv) {
            Ok(re) => {
                let rb = re
                    .dj
                    .iter()
                    .zip(&adv)
                    .filter(|(_, j)| re.tree_classes.contains(j))
                    .map(|(b, _)| *b)
                    .fold(f64::INFINITY, f64::min);
                eprintln!(
                    "[rootbuild] root_eval: adv={} tree_classes={} closed_at_root={} root_bound={rb:.17} bits={:#018x}",
                    adv.len(), re.tree_classes.len(), re.tree_classes.is_empty(), rb.to_bits()
                );
            }
            Err(e) => eprintln!("[rootbuild] root_eval err: {e}"),
        }
        // Best-effort timing (median of NY_ROOT_REPS extra builds). Comes LAST
        // so a wall-capped run still emitted the digest + root_bound above.
        if reps > 0 {
            let mut times = vec![warm_secs];
            for _ in 0..reps {
                let t0 = Instant::now();
                let r = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
                times.push(t0.elapsed().as_secs_f64());
                std::hint::black_box(&r.layers[0].l[0]);
            }
            times.sort_by(f64::total_cmp);
            eprintln!(
                "[rootbuild] build median {:.3}s over {} builds (min {:.3}s max {:.3}s)",
                times[times.len() / 2],
                times.len(),
                times[0],
                times[times.len() - 1]
            );
        }
    }

    /// Pass-cost scaling: is a backward pass cost proportional to column count
    /// R (batching only fills cores) or dominated by a fixed per-pass constant
    /// (batching amortizes real work)? Times single Lower passes at several R.
    pub(crate) fn probe_pass_scaling() {
        use ndarray::Array2;
        let vnnlib_base = std::env::var("NY_PROBE_VNNLIB").expect("set NY_PROBE_VNNLIB");
        let cat = std::env::var("NY_PROBE_CAT").unwrap_or_else(|_| "cifar100_2024".into());
        let default_onnx = if cat.starts_with("tiny") {
            "TinyImageNet_resnet_medium.onnx"
        } else {
            "CIFAR100_resnet_medium.onnx"
        };
        let onnx_base = std::env::var("NY_PROBE_ONNX").unwrap_or_else(|_| default_onnx.into());
        let onnx = format!("{BENCH_ROOT}/{cat}/onnx/{onnx_base}");
        let vnnlib = format!("{BENCH_ROOT}/{cat}/vnnlib/{vnnlib_base}");
        let (spec, lo, hi, _t, _adv) = instance_at(&onnx, &vnnlib);
        let net = TwinNet::compile(&spec).expect("compile");
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let eng = BackwardEngine::new(&net, &root);
        let n_y = net.n_y;
        eprintln!(
            "[scaling] {vnnlib_base} n_y={n_y} cores={}",
            rayon::current_num_threads()
        );
        eprintln!("[scaling]   R      ms   ms/col   (single Lower backward, outward)");
        for &r in &[1usize, 4, 16, 64, 128, 256] {
            // Seed (n_y, r): column c activates head neuron c%n_y (finite, dense-ish).
            let mut s = Array2::<f64>::zeros((n_y, r));
            for c in 0..r {
                s[[c % n_y, c]] = 1.0;
            }
            let e = Array2::<f64>::zeros((n_y, r));
            let seed = Seed { s, e: Some(e) };
            // Warm once, then median of 5.
            let _ = eng
                .run(&seed, None, LaneDir::Lower, None, false)
                .expect("run");
            let mut times = Vec::new();
            for _ in 0..5 {
                let t0 = Instant::now();
                let _ = eng
                    .run(&seed, None, LaneDir::Lower, None, false)
                    .expect("run");
                times.push(t0.elapsed().as_secs_f64());
            }
            times.sort_by(|a, b| a.total_cmp(b));
            let ms = times[2] * 1e3;
            eprintln!("[scaling] {r:>4} {ms:>8.2} {:>8.3}", ms / r as f64);
        }
    }

    // ----------------------------------------------------------------------
    //  #percol-alpha COLUMN-SCALING microbenchmark (`col-scaling`)
    // ----------------------------------------------------------------------
    //
    // THE QUESTION. Per-domain alpha (abc's `optimized_bounds.py`: 10 Adam
    // iterations, one private lower slope per unstable neuron per domain) needs
    // ONE FRESH CERTIFIED BACKWARD PASS PER OPTIMIZER STEP. abc only affords
    // that because a single pass serves a 512-child batch. So the whole design
    // hinges on: HOW DOES ny's CERTIFIED BACKWARD PASS COST SCALE WITH COLUMN
    // COUNT? Near-flat => stacking the frontier amortizes and per-domain alpha
    // is affordable. Roughly linear => amortization buys nothing and the alpha
    // inner loop must be made cheap some other way.
    //
    // WHY `pass-scaling` ABOVE IS NOT THE ANSWER. It times `eng.run(...)` with
    // dom=None and NO collection over R in 1..256. The SCORED eval pass
    // (`bab.rs:3236-3247`) runs with `Some(&dom)` and
    // `Collect { unst_abs: true, rows: .., unst_rows: beta_armed }`. Domain
    // gate overrides and the unstable-neuron collections are exactly the two
    // things a per-domain-alpha design would multiply, so they have to be in
    // the measurement. This probe adds them, plus R = 512 / 1280.
    //
    // NOT A LEVER. Read-only timing; nothing here is reachable from a verdict
    // path. Driven by NON-`NY_` env names on purpose: these are bench knobs,
    // not levers, and the `ny-levers` ratchet must not be fed a new raw `NY_*`
    // read for a diagnostic.
    //   COLSCALE_ONNX    absolute .onnx path   (default cifar100 resnet_medium)
    //   COLSCALE_VNNLIB  absolute .vnnlib path (default idx_4597, a wall row)
    //   COLSCALE_COLS    comma list of R       (default 1,2,4,8,16,32,64,128,256,512,1280)
    //   COLSCALE_REPS    timed reps per point  (default 5, median reported)
    //   COLSCALE_DEPTH   trunk splits in the deep-domain arm (default 20)
    pub(crate) fn probe_col_scaling() {
        use ndarray::Array2;

        fn env_or(key: &str, default: &str) -> String {
            std::env::var(key).unwrap_or_else(|_| default.to_string())
        }
        fn rss_gib() -> f64 {
            // statm field 2 = resident pages.
            std::fs::read_to_string("/proc/self/statm")
                .ok()
                .and_then(|s| s.split_whitespace().nth(1).map(str::to_owned))
                .and_then(|p| p.parse::<f64>().ok())
                .map_or(f64::NAN, |pages| {
                    pages * 4096.0 / (1024.0 * 1024.0 * 1024.0)
                })
        }

        let onnx = env_or(
            "COLSCALE_ONNX",
            &format!("{BENCH_ROOT}/cifar100_2024/onnx/CIFAR100_resnet_medium.onnx"),
        );
        let vnnlib = env_or(
            "COLSCALE_VNNLIB",
            &format!(
                "{BENCH_ROOT}/cifar100_2024/vnnlib/\
                 CIFAR100_resnet_medium_prop_idx_4597_sidx_282_eps_0.0039.vnnlib"
            ),
        );
        let cols: Vec<usize> = env_or("COLSCALE_COLS", "1,2,4,8,16,32,64,128,256,512,1280")
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let reps: usize = env_or("COLSCALE_REPS", "5").parse().unwrap_or(5);
        let depth: usize = env_or("COLSCALE_DEPTH", "20").parse().unwrap_or(20);

        let (spec, lo, hi, t, adv) = instance_at(&onnx, &vnnlib);
        let net = TwinNet::compile(&spec).expect("compile");
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let eng = BackwardEngine::new(&net, &root);
        let n_y = net.n_y;

        // ---- shape census: what the pass actually walks ----
        let n_unst: usize = root.layers.iter().map(|l| l.unst.len()).sum();
        let n_trunk: usize = root.layers.iter().map(|l| l.n).sum();
        let widest = net.tsize.iter().copied().max().unwrap_or(0);
        eprintln!(
            "[colscale] onnx={} vnnlib={}",
            Path::new(&onnx)
                .file_name()
                .map_or_else(|| "?".into(), |s| s.to_string_lossy()),
            Path::new(&vnnlib)
                .file_name()
                .map_or_else(|| "?".into(), |s| s.to_string_lossy())
        );
        eprintln!(
            "[colscale] n_in={} n_y={} trunk_relus={} trunk_neurons={n_trunk} unstable={n_unst} \
             widest_tensor={widest} cores={}",
            net.n_in,
            n_y,
            root.layers.len(),
            rayon::current_num_threads()
        );

        // ---- nf: the REAL column count of a scored eval pass ----
        // Root-fail set exactly as `Probe::new` / the lane derive it.
        let (al0, au0) = eng.y_rows(None).expect("y_rows");
        let ybox = YBox::from_rows(&eng, &al0, &au0);
        let mb_all = MarginBatch::new(&net, t, &adv).expect("mb_all");
        let gates0 = head_gates(&ybox, root.mode);
        let ms_all = margin_seed(&mb_all, &gates0, &ybox, root.mode);
        let ald = row_dots(&root, &al0);
        let aud = row_dots(&root, &au0);
        let m2v = compose_viay(&eng, &mb_all, &gates0, &al0, &au0, &ald, &aud, root.mode);
        let fail: Vec<usize> = (0..adv.len())
            .filter(|&k| ms_all.m1[k].max(m2v[k]) < 0.0)
            .map(|k| adv[k])
            .collect();
        let mb = MarginBatch::new(&net, t, &fail).expect("mb fail");
        let nf = mb.nf();
        let ms = margin_seed(&mb, &gates0, &ybox, root.mode);
        eprintln!(
            "[colscale] adv={} root_fail_nf={nf}  (a scored eval pass runs at R=nf unless \
             col-retire narrows it)",
            adv.len()
        );

        // ---- a DEEP domain: `depth` trunk splits spread over the layers ----
        // NOTE the convention: a trunk split is `(layer, POSITION-IN-`unst`,
        // sign)`, not `(layer, absolute-neuron, sign)` — `domain_gates_split_only`
        // (engine.rs:1690-1692) does `rec.unst[pos]` itself.
        let mut splits: Vec<(usize, usize, i8)> = Vec::new();
        'outer: loop {
            let before = splits.len();
            for (li, lg) in root.layers.iter().enumerate() {
                let taken = splits.iter().filter(|s| s.0 == li).count();
                if taken < lg.unst.len() {
                    let sign = if splits.len().is_multiple_of(2) {
                        1i8
                    } else {
                        -1i8
                    };
                    splits.push((li, taken, sign));
                    if splits.len() >= depth {
                        break 'outer;
                    }
                }
            }
            if splits.len() == before {
                break;
            }
        }
        let dom = domain_gates(&root, &splits);
        eprintln!(
            "[colscale] deep domain: {} trunk splits over {} layers (gate overrides on {} layers)",
            splits.len(),
            splits.iter().map(|s| s.0).collect::<HashSet<_>>().len(),
            dom.layers.len()
        );

        // ---- seed builder: tile the REAL margin seed's columns out to R ----
        // Column VALUES cannot change the cost (every skip in the walk tests a
        // WEIGHT against zero, never a seed entry), but using the real seed
        // keeps magnitudes — and therefore the certified error lane's work —
        // representative.
        let make_seed = |r: usize| -> Seed {
            let mut s = Array2::<f64>::zeros((n_y, r));
            let mut e = Array2::<f64>::zeros((n_y, r));
            let se = ms.seed.e.as_ref();
            for c in 0..r {
                let src = if nf == 0 { None } else { Some(c % nf) };
                for j in 0..n_y {
                    match src {
                        Some(sc) => {
                            s[[j, c]] = ms.seed.s[[j, sc]];
                            e[[j, c]] = se.map_or(0.0, |m| m[[j, sc]]);
                        }
                        None => s[[j, c]] = f64::from(u8::from(j == c % n_y)),
                    }
                }
            }
            Seed { s, e: Some(e) }
        };

        // ---- the four arms ----
        // base       : root gates, no collection      (== `pass-scaling`)
        // dom        : deep-domain gates, no collection
        // dom+abs    : + unst_abs                     (== the SCORED eval pass)
        // dom+rows   : + unst_rows                    (the alpha-gradient capture)
        let arms: [(&str, bool, bool, bool); 4] = [
            ("base", false, false, false),
            ("dom", true, false, false),
            ("dom+abs", true, true, false),
            ("dom+rows", true, true, true),
        ];
        eprintln!(
            "[colscale] median of {reps} timed reps (1 warm), single Lower pass, Outward \
             (certified error lane ON)"
        );
        eprintln!(
            "[colscale] {:>5} {:>10} {:>10} {:>10} {:>10}   {:>9} {:>9}",
            "R", "base_ms", "dom_ms", "domabs_ms", "domrows_ms", "abs/col", "rss_GiB"
        );
        let mut base_at_one = f64::NAN;
        let mut abs_at_one = f64::NAN;
        let mut rows: Vec<(usize, [f64; 4])> = Vec::new();
        for &r in &cols {
            if r == 0 {
                continue;
            }
            let seed = make_seed(r);
            let mut out = [f64::NAN; 4];
            for (ai, &(_, use_dom, unst_abs, unst_rows)) in arms.iter().enumerate() {
                let collect = ny_propagate::margin_row::engine::Collect {
                    unst_abs,
                    rows: None,
                    unst_rows,
                };
                let dref = use_dom.then_some(&dom);
                let warm = eng.run_collect(&seed, dref, LaneDir::Lower, None, collect);
                if let Err(e) = warm {
                    eprintln!("[colscale] R={r} arm={} ERR {e}", arms[ai].0);
                    continue;
                }
                let mut times = Vec::with_capacity(reps);
                for _ in 0..reps {
                    let t0 = Instant::now();
                    let p = eng
                        .run_collect(&seed, dref, LaneDir::Lower, None, collect)
                        .expect("pass");
                    times.push(t0.elapsed().as_secs_f64());
                    std::hint::black_box(&p.b[0]);
                }
                times.sort_by(f64::total_cmp);
                out[ai] = times[times.len() / 2] * 1e3;
            }
            if r == 1 {
                base_at_one = out[0];
                abs_at_one = out[2];
            }
            eprintln!(
                "[colscale] {r:>5} {:>10.2} {:>10.2} {:>10.2} {:>10.2}   {:>9.3} {:>9.2}",
                out[0],
                out[1],
                out[2],
                out[3],
                out[2] / r as f64,
                rss_gib()
            );
            rows.push((r, out));
        }

        // ---- the decision numbers ----
        eprintln!("[colscale] --- marginal cost of the 2nd..Nth column (scored arm dom+abs) ---");
        eprintln!(
            "[colscale] {:>5} {:>12} {:>12} {:>14}",
            "R", "x_vs_R=1", "marg_ms/col", "amortized?"
        );
        for &(r, out) in &rows {
            if r == 1 || !abs_at_one.is_finite() {
                continue;
            }
            let marg = (out[2] - abs_at_one) / (r as f64 - 1.0);
            eprintln!(
                "[colscale] {r:>5} {:>12.2} {:>12.3} {:>14}",
                out[2] / abs_at_one,
                marg,
                if out[2] / abs_at_one < 2.0 {
                    "FLAT"
                } else {
                    "not-flat"
                }
            );
        }
        if base_at_one.is_finite() && abs_at_one.is_finite() {
            eprintln!(
                "[colscale] collection overhead at R=1: unst_abs adds {:.1}% over base",
                100.0 * (abs_at_one / base_at_one - 1.0)
            );
        }
        for &(r, out) in &rows {
            eprintln!(
                "[colscale] R={r:<5} dom/base={:.3}  domabs/dom={:.3}  domrows/domabs={:.3}",
                out[1] / out[0],
                out[2] / out[1],
                out[3] / out[2]
            );
        }
    }

    /// ORACLE: the serial lane is the verified truth. Run serial (frontier=1)
    /// and parallel (frontier=N) from the SAME root and assert the root_bound
    /// is BIT-IDENTICAL and the verdict matches. Drive via NY_PROBE_VNNLIB /
    /// NY_ORACLE_FRONTIER / NY_PROBE_SECS.
    pub(crate) fn oracle_serial_vs_parallel() {
        let cat = std::env::var("NY_PROBE_CAT").unwrap_or_else(|_| "cifar100_2024".into());
        let default_onnx = if cat.starts_with("tiny") {
            "TinyImageNet_resnet_medium.onnx"
        } else {
            "CIFAR100_resnet_medium.onnx"
        };
        let onnx_base = std::env::var("NY_PROBE_ONNX").unwrap_or_else(|_| default_onnx.into());
        let vnnlib_base = std::env::var("NY_PROBE_VNNLIB").expect("set NY_PROBE_VNNLIB");
        let secs: u64 = std::env::var("NY_PROBE_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);
        let frontier: usize = std::env::var("NY_ORACLE_FRONTIER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(14);
        let onnx = format!("{BENCH_ROOT}/{cat}/onnx/{onnx_base}");
        let vnnlib = format!("{BENCH_ROOT}/{cat}/vnnlib/{vnnlib_base}");
        let (spec, lo, hi, t, adv) = instance_at(&onnx, &vnnlib);
        let net = TwinNet::compile(&spec).expect("compile");
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let run = |frontier: usize| {
            let deadline = Instant::now() + Duration::from_secs(secs);
            // #margin-row-branch-width: the candidate-width knobs live in
            // `margin_row/mod.rs`, NOT here. This diagnostic entry point builds
            // its own config; the SCORED concurrent lane does not come through
            // it, so an override added here is inert — measured exactly that way
            // (expansions identical at k=8, 4 and 2).
            let cfg = BabConfig {
                max_expansions: 20_000,
                deadline: Some(deadline),
                lru_cap: 64,
                frontier,
                ..BabConfig::default()
            };
            match MarginRowBab::run(&net, &root, t, &adv, cfg) {
                MarginRowOutcome::Unsat(s, _) => {
                    ("UNSAT".to_string(), s.root_bound, s.expansions, s.stop)
                }
                MarginRowOutcome::Unknown { reason, stats } => {
                    let rb = stats.as_ref().map_or(f64::NAN, |s| s.root_bound);
                    let exp = stats.as_ref().map_or(0, |s| s.expansions);
                    (format!("UNKNOWN({reason})"), rb, exp, "n/a".to_string())
                }
            }
        };
        let (sv, srb, sexp, sstop) = run(1);
        let (pv, prb, pexp, pstop) = run(frontier);
        eprintln!("[oracle] {vnnlib_base} frontier={frontier}");
        eprintln!("[oracle]   serial  : {sv:>10} root_bound={srb:.17} bits={:#018x} exp={sexp} stop={sstop}", srb.to_bits());
        eprintln!("[oracle]   parallel: {pv:>10} root_bound={prb:.17} bits={:#018x} exp={pexp} stop={pstop}", prb.to_bits());
        eprintln!(
            "[oracle]   root_bound diff = {:.3e}  bit-identical={}",
            (srb - prb).abs(),
            srb.to_bits() == prb.to_bits()
        );
        assert_eq!(
            srb.to_bits(),
            prb.to_bits(),
            "root_bound must be bit-identical (serial vs parallel)"
        );
        let serial_unsat = sv == "UNSAT";
        let parallel_unsat = pv == "UNSAT";
        // MOAT-safe: parallel may only be UNSAT if serial is (both sound); it
        // must NEVER claim UNSAT where serial did not.
        assert!(
            !parallel_unsat || serial_unsat,
            "parallel UNSAT but serial not: {pv} vs {sv}"
        );
        eprintln!("[oracle]   VERDICT MATCH: serial={sv} parallel={pv}");
    }

    fn instance(name: &str) -> (TwinSpec, Vec<f64>, Vec<f64>, usize, Vec<usize>) {
        let onnx = format!("{BENCH}/onnx/CIFAR100_resnet_medium.onnx");
        let vnnlib = format!("{BENCH}/vnnlib/{name}");
        let spec = extract_twin_spec(Path::new(&onnx)).expect("extract twin spec");
        let (lo, hi, t, adv) = parse_vnnlib_robustness(Path::new(&vnnlib)).expect("parse vnnlib");
        (spec, lo, hi, t, adv)
    }

    /// Probe-semantics bound at a (splits, head_clamp) prefix:
    /// `B = max(min_j direct_j, min_j max(m1_j, m2v_j))` over the root-fail
    /// class set (mirrors probe_backward.py's per-depth bound).
    #[allow(clippy::type_complexity)]
    struct Probe<'a> {
        eng: BackwardEngine<'a>,
        mb: MarginBatch,
    }

    impl<'a> Probe<'a> {
        fn new(net: &'a TwinNet, root: &'a RootGates, t: usize, adv: &[usize]) -> Self {
            let eng = BackwardEngine::new(net, root);
            // Root-fail set from the via-y bound over ALL classes (probe line
            // 62-64: b0 = mb_all.bounds(Al0, Au0); fail = b0 < 0).
            let (al0, au0) = eng.y_rows(None).expect("y_rows");
            let ybox = YBox::from_rows(&eng, &al0, &au0);
            let mb_all = MarginBatch::new(net, t, adv).expect("mb_all");
            let gates = head_gates(&ybox, root.mode);
            let ms = margin_seed(&mb_all, &gates, &ybox, root.mode);
            let ald = row_dots(root, &al0);
            let aud = row_dots(root, &au0);
            let m2v = compose_viay(&eng, &mb_all, &gates, &al0, &au0, &ald, &aud, root.mode);
            let fail: Vec<usize> = (0..adv.len())
                .filter(|&k| ms.m1[k].max(m2v[k]) < 0.0)
                .map(|k| adv[k])
                .collect();
            let mb = MarginBatch::new(net, t, &fail).expect("mb fail");
            Self { eng, mb }
        }

        fn bound_at(
            &self,
            splits: &[(usize, usize, i8)],
            head_clamp: &[(usize, i8)],
        ) -> (f64, f64) {
            let root = self.eng.root;
            let dom = domain_gates(root, splits);
            let dom_opt = (!splits.is_empty()).then_some(&dom);
            let (al, au) = self.eng.y_rows(dom_opt).expect("y_rows");
            let mut ybox = YBox::from_rows(&self.eng, &al, &au);
            ybox.clamp(head_clamp);
            assert!(!ybox.is_empty(), "probe domain must be feasible");
            let gates = head_gates(&ybox, root.mode);
            let ms = margin_seed(&self.mb, &gates, &ybox, root.mode);
            let ald = row_dots(root, &al);
            let aud = row_dots(root, &au);
            let m2v = compose_viay(&self.eng, &self.mb, &gates, &al, &au, &ald, &aud, root.mode);
            let b_viay = (0..self.mb.nf())
                .map(|k| ms.m1[k].max(m2v[k]))
                .fold(f64::INFINITY, f64::min);
            let pass = self
                .eng
                .run(&ms.seed, dom_opt, LaneDir::Lower, None, false)
                .expect("seeded pass");
            let direct = per_class_direct(&self.eng, &pass, &ms, 0..self.mb.nf());
            let b_direct = direct.iter().copied().fold(f64::INFINITY, f64::min);
            (b_direct.max(b_viay), b_viay)
        }
    }

    const P1498: &str = "CIFAR100_resnet_medium_prop_idx_1498_sidx_792_eps_0.0039.vnnlib";
    const P4429: &str = "CIFAR100_resnet_medium_prop_idx_4429_sidx_1471_eps_0.0039.vnnlib";
    const P2551: &str = "CIFAR100_resnet_medium_prop_idx_2551_sidx_9941_eps_0.0039.vnnlib";
    const P6232_SAT: &str = "CIFAR100_resnet_medium_prop_idx_6232_sidx_3020_eps_0.0039.vnnlib";

    /// Recorded 1498 worst path (out/probe_bw_prop_idx_1498_sidx_792_*.json).
    /// Events in chosen order; expected (bound, bound_viay) per depth.
    const TRAJ_1498: &[(Option<(&str, usize, usize, i8)>, f64, f64)] = &[
        (
            Some(("head", 23, 0, -1)),
            -0.127_445_187_049_127_2,
            -1.510_236_772_280_93,
        ),
        (
            Some(("head", 10, 0, 1)),
            -0.099_167_847_802_682_06,
            -1.472_712_865_338_950_3,
        ),
        (
            Some(("trunk", 5, 9, 1)),
            -0.073_089_561_916_132_14,
            -1.457_823_829_617_386_2,
        ),
        (
            Some(("head", 53, 0, -1)),
            -0.053_729_815_336_940_27,
            -1.469_321_617_079_468_3,
        ),
        (
            Some(("trunk", 5, 10, -1)),
            -0.037_234_894_794_964_585,
            -1.445_943_919_376_340_4,
        ),
        (
            Some(("head", 32, 0, 1)),
            -0.022_401_641_733_078_015,
            -1.425_016_855_247_147_6,
        ),
        (
            Some(("trunk", 5, 52, -1)),
            -0.008_241_903_027_566_416,
            -1.415_053_943_968_675_1,
        ),
        (None, 0.002_599_361_254_453_106, -1.384_457_873_124_884_9),
    ];

    /// INC2a gate (i): parity-mode root direct bound on prop_1498 must
    /// reproduce the Python reference to <= 1e-9.
    pub(crate) fn inc2a_root_parity_1498() {
        let (spec, lo, hi, t, adv) = instance(P1498);
        let net = TwinNet::compile(&spec).expect("compile");
        let t0 = Instant::now();
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Parity, None).expect("root");
        eprintln!(
            "[inc2a] parity root gates build: {:.1}s",
            t0.elapsed().as_secs_f64()
        );
        let probe = Probe::new(&net, &root, t, &adv);
        let (b, b_viay) = probe.bound_at(&[], &[]);
        eprintln!("[inc2a] root direct={b:.12} viay={b_viay:.12}");
        assert!(
            (b - (-0.127_445_187_049_127_2)).abs() <= 1e-9,
            "root direct {b:.12} vs reference -0.127445187049 (diff {})",
            (b - (-0.127_445_187_049_127_2)).abs()
        );
        assert!(
            (b_viay - (-1.510_236_772_280_93)).abs() <= 1e-9,
            "root viay {b_viay:.12} vs reference -1.51023677228093"
        );
    }

    /// INC2a gates (ii)+(iii): outward bound <= parity bound, within 1e-3;
    /// enclosure vs 200 sampled in-box margins (membership-filtered form is
    /// exercised in the INC2b replay; the root has no splits).
    pub(crate) fn inc2a_outward_bound_and_enclosure_1498() {
        use ndarray::Array2;
        let (spec, lo, hi, t, adv) = instance(P1498);
        let net = TwinNet::compile(&spec).expect("compile");
        let root_p = RootGates::build(&net, &lo, &hi, RoundMode::Parity, None).expect("root");
        let probe_p = Probe::new(&net, &root_p, t, &adv);
        let (b_par, _) = probe_p.bound_at(&[], &[]);
        let root_o = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let probe_o = Probe::new(&net, &root_o, t, &adv);
        let (b_out, _) = probe_o.bound_at(&[], &[]);
        eprintln!(
            "[inc2a] parity={b_par:.12} outward={b_out:.12} gap={:.3e}",
            b_par - b_out
        );
        assert!(b_out <= b_par, "outward {b_out} must be <= parity {b_par}");
        assert!(
            b_par - b_out <= 1e-3,
            "outward gap {:.3e} exceeds 1e-3",
            b_par - b_out
        );
        // Enclosure: min sampled margin over the fail set >= outward bound.
        let mut state = 271_828_182_u64;
        let mut next_unit = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
        };
        let n_in = net.n_in;
        let bsz = 200;
        let mut x = Array2::<f64>::zeros((n_in, bsz));
        for i in 0..n_in {
            for b in 0..bsz {
                let u = next_unit();
                x[[i, b]] = lo[i] + u * (hi[i] - lo[i]);
            }
        }
        let (y, _) = net
            .forward_points(&x, &std::collections::BTreeMap::new())
            .expect("forward");
        let (w2, b2, (_, n_y)) = net.gemm2();
        let fail = &probe_o.mb.adv;
        let mut min_margin = f64::INFINITY;
        for &j in fail {
            for b in 0..bsz {
                let mut m = b2[t] - b2[j];
                for k in 0..n_y {
                    m += (w2[t * n_y + k] - w2[j * n_y + k]) * y[[k, b]].max(0.0);
                }
                min_margin = min_margin.min(m);
            }
        }
        eprintln!("[inc2a] sampled min margin {min_margin:.6} vs outward bound {b_out:.6}");
        assert!(
            b_out <= min_margin,
            "outward bound {b_out} above sampled margin {min_margin}"
        );
    }

    /// INC2b gate: replay the RECORDED 1498 worst path. Parity mode must match
    /// every depth to <= 1e-9; outward mode must still cross 0 at the end.
    pub(crate) fn inc2b_trajectory_replay_1498() {
        let (spec, lo, hi, t, adv) = instance(P1498);
        let net = TwinNet::compile(&spec).expect("compile");
        for mode in [RoundMode::Parity, RoundMode::Outward] {
            let root = RootGates::build(&net, &lo, &hi, mode, None).expect("root");
            let probe = Probe::new(&net, &root, t, &adv);
            let mut splits: Vec<(usize, usize, i8)> = Vec::new();
            let mut clamp: Vec<(usize, i8)> = Vec::new();
            let mut final_bound = f64::NEG_INFINITY;
            for (depth, (event, want_b, want_viay)) in TRAJ_1498.iter().enumerate() {
                // Outward mode: recorded positions index the PARITY unstable
                // list; translate through neuron ids (outward unstable is a
                // superset).
                let (b, b_viay) = probe.bound_at(&splits, &clamp);
                eprintln!(
                    "[inc2b {mode:?}] d={depth} B={b:.12} (ref {want_b:.12}) viay={b_viay:.12}"
                );
                if mode == RoundMode::Parity {
                    assert!(
                        (b - want_b).abs() <= 1e-9,
                        "depth {depth}: bound {b:.12} vs recorded {want_b:.12} (diff {:.3e})",
                        (b - want_b).abs()
                    );
                    assert!(
                        (b_viay - want_viay).abs() <= 1e-9,
                        "depth {depth}: viay {b_viay:.12} vs recorded {want_viay:.12}"
                    );
                } else {
                    assert!(
                        b <= want_b + 1e-9,
                        "outward bound {b} above parity-recorded {want_b}"
                    );
                    assert!(
                        (b - want_b).abs() <= 1e-3,
                        "outward drifted {:.3e} from recorded at depth {depth}",
                        (b - want_b).abs()
                    );
                }
                final_bound = b;
                if let Some((kind, a, pos, dir)) = event {
                    if *kind == "trunk" {
                        splits.push((*a, *pos, *dir));
                    } else {
                        clamp.push((*a, *dir));
                    }
                }
            }
            if mode == RoundMode::Outward {
                assert!(
                    final_bound > 0.0,
                    "OUTWARD MODE LOST THE CROSSING: final bound {final_bound:.9} \
                     (parity reference +0.002599361254)"
                );
                eprintln!(
                    "[inc2b] outward final bound {final_bound:.12} still crosses 0 \
                     (margin kept: {:.3e} of +2.599e-3)",
                    final_bound
                );
            }
        }
    }

    /// Recorded positions are into the PARITY unstable list; the outward
    /// replay above relies on the unstable sets agreeing at those positions.
    /// This cross-checks that the (layer 5) ids match between modes.
    pub(crate) fn inc2b_unstable_position_parity_l5() {
        let (spec, lo, hi, _, _) = instance(P1498);
        let net = TwinNet::compile(&spec).expect("compile");
        let par = RootGates::build(&net, &lo, &hi, RoundMode::Parity, None).expect("root");
        let out = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        for li in 0..par.layers.len() {
            assert_eq!(
                par.layers[li].unst, out.layers[li].unst,
                "unstable sets diverge at layer {li}: recorded (li,pos) splits \
                 would not transfer between modes"
            );
        }
    }

    /// INC2c gate: end-to-end certified closures on the three probed
    /// pyrat-easy instances, and the MOAT gate on a known-SAT instance.
    pub(crate) fn inc2c_closes_4429() {
        assert_closes(P4429, 100);
    }

    pub(crate) fn inc2c_closes_1498() {
        assert_closes(P1498, 100);
    }

    pub(crate) fn inc2c_closes_2551() {
        assert_closes(P2551, 100);
    }

    /// MOAT: a banked-SAT instance must NEVER verify.
    pub(crate) fn inc2c_moat_sat_6232_not_verified() {
        let (spec, lo, hi, t, adv) = instance(P6232_SAT);
        let deadline = Instant::now() + Duration::from_secs(100);
        match run_margin_row_lane(&spec, &lo, &hi, t, &adv, Some(deadline), 20_000) {
            MarginRowOutcome::Unsat(stats, _) => panic!(
                "MOAT VIOLATION: SAT instance verified (root_bound={}, stop={})",
                stats.root_bound, stats.stop
            ),
            MarginRowOutcome::Unknown { reason, stats } => {
                eprintln!(
                    "[inc2c moat] 6232 correctly NOT verified: {reason} \
                     (root_bound={:?})",
                    stats.map(|s| s.root_bound)
                );
            }
        }
    }

    fn assert_closes(name: &str, budget_secs: u64) {
        let (spec, lo, hi, t, adv) = instance(name);
        let t0 = Instant::now();
        let deadline = t0 + Duration::from_secs(budget_secs);
        match run_margin_row_lane(&spec, &lo, &hi, t, &adv, Some(deadline), 20_000) {
            MarginRowOutcome::Unsat(stats, _) => {
                eprintln!(
                    "[inc2c] {name}: UNSAT in {:.1}s (root={:.4}, exp={}, domains={}, \
                     closed={}, maxD={}, dips={}, stop={})",
                    t0.elapsed().as_secs_f64(),
                    stats.root_bound,
                    stats.expansions,
                    stats.domains_created,
                    stats.closed,
                    stats.max_depth,
                    stats.mono_raw_dips,
                    stats.stop
                );
                assert!(
                    t0.elapsed().as_secs() < budget_secs,
                    "closure exceeded the {budget_secs}s budget"
                );
            }
            MarginRowOutcome::Unknown { reason, stats } => panic!(
                "{name} did not close: {reason} (stats: {:?})",
                stats.map(|s| (s.root_bound, s.expansions, s.stop))
            ),
        }
    }

    /// Extractor + parser smoke on the real files (fast; not ignored gating
    /// logic — still needs the benchmark checkout).
    pub(crate) fn extractor_and_parser_smoke() {
        let (spec, lo, hi, t, adv) = instance(P1498);
        assert_eq!(spec.n_in, 3072);
        assert_eq!(spec.ops.len(), 19 + 9 + 8 + 1 + 2 + 1); // conv+relu+add+flat+gemms+head relu
        assert_eq!(lo.len(), 3072);
        assert!(hi.iter().zip(&lo).all(|(h, l)| h >= l));
        assert_eq!(t, 72);
        assert_eq!(adv.len(), 99);
        let net = TwinNet::compile(&spec).expect("compile");
        assert_eq!(net.n_y, 100);
        assert_eq!(net.n_out, 100);
        assert_eq!(net.trunk_relus.len(), 9);
    }

    // ===================== ADVERSARIAL MOAT + CLOSURE HARNESS ==============
    // Ground truth: official 2025 results. GT-SAT set = clean-sat (abc/pyrat/
    // neuralsat, ORT-gated) PLUS conflict-sat (those tools sat, nnv the lone
    // false-unsat). The lane must return NON-Unsat on EVERY GT-sat row. UNSAT
    // tiers from pyrat(unsat)=pyrat-easy, abc(unsat)&!pyrat=abc-only.

    /// 14 GT-SAT CIFAR100_resnet_medium instances (7 clean, 7 nnv-conflict).
    const GT_SAT: &[(&str, &str)] = &[
        (
            "CIFAR100_resnet_medium_prop_idx_2697_sidx_4836_eps_0.0039.vnnlib",
            "clean",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_5001_sidx_1154_eps_0.0039.vnnlib",
            "clean",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_5634_sidx_9441_eps_0.0039.vnnlib",
            "clean",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_5894_sidx_5577_eps_0.0039.vnnlib",
            "clean",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_6573_sidx_5525_eps_0.0039.vnnlib",
            "clean",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_7431_sidx_7329_eps_0.0039.vnnlib",
            "clean",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_9845_sidx_200_eps_0.0039.vnnlib",
            "clean",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_1592_sidx_3741_eps_0.0039.vnnlib",
            "nnv-conflict",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_2352_sidx_2704_eps_0.0039.vnnlib",
            "nnv-conflict",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_4752_sidx_1800_eps_0.0039.vnnlib",
            "nnv-conflict",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_5973_sidx_7841_eps_0.0039.vnnlib",
            "nnv-conflict",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_6049_sidx_4719_eps_0.0039.vnnlib",
            "nnv-conflict",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_6232_sidx_3020_eps_0.0039.vnnlib",
            "nnv-conflict",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_7569_sidx_2906_eps_0.0039.vnnlib",
            "nnv-conflict",
        ),
    ];

    /// GT-UNSAT rows spanning both tiers (pyrat-easy + abc-only).
    const GT_UNSAT: &[(&str, &str)] = &[
        (
            "CIFAR100_resnet_medium_prop_idx_1498_sidx_792_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_2551_sidx_9941_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_4429_sidx_1471_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_54_sidx_9735_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_1588_sidx_6654_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_1642_sidx_7494_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_1798_sidx_1061_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_3343_sidx_1406_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_3418_sidx_4225_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_4605_sidx_6225_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_5127_sidx_993_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_5242_sidx_1208_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_5831_sidx_5506_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_1190_sidx_8846_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_1761_sidx_3933_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_2050_sidx_8228_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_2132_sidx_6868_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_230_sidx_1968_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_2477_sidx_388_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_2779_sidx_6416_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_2925_sidx_8815_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_3006_sidx_928_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_4191_sidx_3301_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_4760_sidx_4651_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_4921_sidx_3617_eps_0.0039.vnnlib",
            "abc-only",
        ),
    ];

    fn moat_deadline_secs() -> u64 {
        std::env::var("NY_MOAT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60)
    }

    /// Run the lane directly (the decisive path — no attack short-circuit).
    /// Returns (verdict, root_bound, stop, expansions, max_depth, dips, secs).
    fn run_named(name: &str, secs: u64) -> (String, f64, String, usize, usize, usize, f64) {
        let (spec, lo, hi, t, adv) = instance(name);
        let t0 = Instant::now();
        let deadline = t0 + Duration::from_secs(secs);
        match run_margin_row_lane(&spec, &lo, &hi, t, &adv, Some(deadline), 20_000) {
            MarginRowOutcome::Unsat(s, _) => (
                "UNSAT".into(),
                s.root_bound,
                s.stop,
                s.expansions,
                s.max_depth,
                s.mono_raw_dips,
                t0.elapsed().as_secs_f64(),
            ),
            MarginRowOutcome::Unknown { reason, stats } => match stats {
                Some(s) => (
                    format!("UNKNOWN({reason})"),
                    s.root_bound,
                    s.stop,
                    s.expansions,
                    s.max_depth,
                    s.mono_raw_dips,
                    t0.elapsed().as_secs_f64(),
                ),
                None => (
                    format!("UNKNOWN({reason})"),
                    f64::NAN,
                    "no-stats".into(),
                    0,
                    0,
                    0,
                    t0.elapsed().as_secs_f64(),
                ),
            },
        }
    }

    /// THE MOAT: every GT-sat row must return NON-Unsat. A single Unsat is
    /// catastrophic (a false-unsat where a real counterexample exists).
    pub(crate) fn moat_gt_sat_never_unsat() {
        let secs = moat_deadline_secs();
        let mut violations = Vec::new();
        for (name, kind) in GT_SAT {
            let (v, rb, stop, exp, md, dips, el) = run_named(name, secs);
            let short = name.split("prop_idx_").nth(1).unwrap_or(name);
            eprintln!(
                "[moat {kind:>12}] {short:>28} -> {v:>10} root_bound={rb:+.5} \
                 stop={stop} exp={exp} maxD={md} dips={dips} ({el:.1}s)"
            );
            if v == "UNSAT" {
                violations.push((*name, *kind, rb));
            }
            assert!(
                rb <= 0.0 || v.starts_with("UNKNOWN"),
                "GT-sat {short}: positive root_bound {rb} would imply a false close"
            );
        }
        assert!(
            violations.is_empty(),
            "MOAT BROKEN: lane returned UNSAT on GT-sat instances {violations:?}"
        );
        eprintln!(
            "[moat] PASS: 0 false-unsat across {} GT-sat rows",
            GT_SAT.len()
        );
    }

    /// Closure + cross-check contingency over GT-unsat tiers. Records how many
    /// each tier closes; every UNSAT here is GT-correct (all rows are GT-unsat).
    pub(crate) fn closure_gt_unsat_contingency() {
        let secs = moat_deadline_secs();
        let mut tally: std::collections::BTreeMap<&str, (usize, usize, usize)> = Default::default(); // tier -> (unsat, unknown, total)
        for (name, tier) in GT_UNSAT {
            let (v, rb, stop, exp, md, dips, el) = run_named(name, secs);
            let short = name.split("prop_idx_").nth(1).unwrap_or(name);
            eprintln!(
                "[closure {tier:>10}] {short:>28} -> {v:>10} root_bound={rb:+.5} \
                 stop={stop} exp={exp} maxD={md} dips={dips} ({el:.1}s)"
            );
            let e = tally.entry(tier).or_default();
            e.2 += 1;
            if v == "UNSAT" {
                e.0 += 1;
            } else {
                e.1 += 1;
            }
        }
        eprintln!("[closure] contingency (tier -> unsat/unknown/total):");
        for (tier, (u, k, tot)) in &tally {
            eprintln!("   {tier:>10}: unsat={u} unknown={k} total={tot}");
        }
    }

    /// Extraction safety on the NON-target onnx models: the lane must either
    /// extract them as valid twin-wall nets OR fail-closed (None). Either way,
    /// a GT-sat row on those models can never be falsely closed because the
    /// vnncomp hook only runs the lane when extraction succeeds AND the verdict
    /// is unknown/timeout. Reports which models extract.
    pub(crate) fn extraction_safety_other_models() {
        let base = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../benchmarks/vnncomp2025/benchmarks"
        );
        for (bench, onnx) in [
            ("cifar100_2024", "CIFAR100_resnet_large.onnx"),
            ("cifar100_2024", "test_nano.onnx"),
            ("tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx"),
            ("tinyimagenet_2024", "test_nano.onnx"),
        ] {
            let p = format!("{base}/{bench}/onnx/{onnx}");
            let got = extract_twin_spec(Path::new(&p));
            eprintln!(
                "[extract] {bench}/{onnx}: {}",
                got.as_ref()
                    .map_or("None (fail-closed)".to_string(), |s| format!(
                        "Some(n_in={}, ops={})",
                        s.n_in,
                        s.ops.len()
                    ))
            );
        }
    }
    /// EPOCH GAIN PROBE (#epoch-bab Phase C): on a real wall instance,
    /// measure per-layer epoch-rebuild gain — bake k same-direction splits
    /// of layer L's top unstable neurons (by retention score c*(u-l)) and
    /// compare the rebuilt-root domain bound against the frozen-gates
    /// domain-override bound for the SAME splits. Env:
    ///   NY_PROBE_VNNLIB, NY_GAIN_K (default 4), NY_GAIN_LAYERS (e.g. "0,2,4").
    pub(crate) fn epoch_gain_probe_real() {
        use ny_propagate::margin_row::root::RetainCfg;
        let vnnlib_base = std::env::var("NY_PROBE_VNNLIB").expect("set NY_PROBE_VNNLIB");
        let k: usize = std::env::var("NY_GAIN_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        let layers_env = std::env::var("NY_GAIN_LAYERS").unwrap_or_else(|_| "0,2,4,6,8".into());
        let (spec, lo, hi, t, adv) = instance(&vnnlib_base);
        let net = TwinNet::compile(&spec).expect("compile");
        let t0 = Instant::now();
        let cfg = RetainCfg {
            per_layer: 64,
            budget_bytes: 1 << 30,
        };
        let (root, _) =
            RootGates::build_retaining(&net, &lo, &hi, RoundMode::Outward, None, Some(&cfg), &[])
                .expect("root");
        eprintln!(
            "[gain] root build {:.1}s, {} trunk layers",
            t0.elapsed().as_secs_f64(),
            root.layers.len()
        );
        // Root-fail classes via the probe helper.
        let probe = Probe::new(&net, &root, t, &adv);
        let (b_root, _) = probe.bound_at(&[], &[]);
        eprintln!("[gain] {vnnlib_base} root bound {b_root:+.5}");
        for ls in layers_env.split(',') {
            let li: usize = ls.trim().parse().expect("layer index");
            let lg = &root.layers[li];
            // Top-k unstable by c*(u-l).
            let mut scored: Vec<(f64, usize, usize)> = lg
                .unst
                .iter()
                .enumerate()
                .map(|(pos, &j)| (lg.c[j] * (lg.u[j] - lg.l[j]), j, pos))
                .collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0));
            scored.truncate(k);
            if scored.is_empty() {
                eprintln!("[gain] layer {li}: no unstable neurons");
                continue;
            }
            for dir in [1i8, -1] {
                let splits_pos: Vec<(usize, usize, i8)> =
                    scored.iter().map(|&(_, _, pos)| (li, pos, dir)).collect();
                let splits_abs: Vec<(usize, usize, i8)> =
                    scored.iter().map(|&(_, j, _)| (li, j, dir)).collect();
                let (bf, _) = probe.bound_at(&splits_pos, &[]);
                let tb = Instant::now();
                let (eroot, _) = RootGates::build_retaining(
                    &net,
                    &lo,
                    &hi,
                    RoundMode::Outward,
                    None,
                    None,
                    &splits_abs,
                )
                .expect("epoch root");
                let eprobe = Probe::new(&net, &eroot, t, &adv);
                let (be, _) = eprobe.bound_at(&[], &[]);
                eprintln!(
                    "[gain] layer {li} k={} dir {dir:+}: frozen {bf:+.5} epoch {be:+.5} \
                     gain {:+.4} (rebuild {:.1}s)",
                    scored.len(),
                    be - bf,
                    tb.elapsed().as_secs_f64()
                );
            }
        }
    }
    // ================= METAROOM BANKED GAINS (#epoch-bab Phase D) ==========
    // The Constant/Reshape extractor extension unlocked the metaroom_2023
    // class (Conv-Relu...-Reshape-Gemm-Relu-Gemm). Measured 2026-07-18 over
    // the full 106-row instances.csv (M5 Max, release, official per-instance
    // budgets): 85 UNSAT, 0 incorrect — every one CONFIRMED against the
    // official 2025 results (>=1 tool unsat, 0 tools sat; cross-checked
    // against all 7 metaroom-reporting tools).
    //
    // FIVE of those are instances NY's PRODUCTION verifier does not decide
    // (reports/measured/metaroom_2023.csv: timeout). They are the banked
    // competitive gain of this arc — the lane closes each in ~8-9s where
    // production burns its budget and yields nothing.
    const METAROOM_PROD_TIMEOUT_GAINS: &[(&str, &str)] = &[
        (
            "6cnn_ry_1_0_no_custom_OP.onnx",
            "spec_idx_108_eps_0.00000436.vnnlib",
        ),
        (
            "6cnn_ry_64_10_no_custom_OP.onnx",
            "spec_idx_132_eps_0.00000436.vnnlib",
        ),
        (
            "6cnn_ry_4_0_no_custom_OP.onnx",
            "spec_idx_125_eps_0.00000436.vnnlib",
        ),
        (
            "6cnn_ry_52_8_no_custom_OP.onnx",
            "spec_idx_126_eps_0.00000436.vnnlib",
        ),
        (
            "6cnn_ry_28_4_no_custom_OP.onnx",
            "spec_idx_113_eps_0.00000436.vnnlib",
        ),
    ];

    /// The 5 banked metaroom gains still close (regression gate for the
    /// extractor + lane). Each is GT-unsat by >=6 official tools.
    pub(crate) fn metaroom_banked_gains_still_close() {
        let mut failures = Vec::new();
        for (onnx_base, vnnlib_base) in METAROOM_PROD_TIMEOUT_GAINS {
            let onnx = format!("{BENCH_ROOT}/metaroom_2023/onnx/{onnx_base}");
            let vnnlib = format!("{BENCH_ROOT}/metaroom_2023/vnnlib/{vnnlib_base}");
            let (spec, lo, hi, t, adv) = instance_at(&onnx, &vnnlib);
            let t0 = Instant::now();
            let deadline = t0 + Duration::from_mins(1);
            let out = run_margin_row_lane(&spec, &lo, &hi, t, &adv, Some(deadline), 20_000);
            match out {
                MarginRowOutcome::Unsat(s, _) => eprintln!(
                    "[metaroom] {vnnlib_base} -> UNSAT in {:.1}s (exp={} ledger={:?})",
                    t0.elapsed().as_secs_f64(),
                    s.expansions,
                    s.ledger_ok
                ),
                MarginRowOutcome::Unknown { reason, .. } => {
                    eprintln!("[metaroom] {vnnlib_base} -> UNKNOWN({reason})");
                    failures.push(*vnnlib_base);
                }
            }
        }
        assert!(
            failures.is_empty(),
            "banked metaroom gains regressed: {failures:?}"
        );
    }

    /// MOAT on the metaroom class: every GT-SAT metaroom row must return
    /// NON-Unsat. (Official 2025: 5 sat rows on this benchmark.)
    pub(crate) fn metaroom_moat_gt_sat_never_unsat() {
        const GT_SAT: &[(&str, &str)] = &[
            (
                "4cnn_ry_99_16_no_custom_OP.onnx",
                "spec_idx_43_eps_0.00000436.vnnlib",
            ),
            (
                "6cnn_ry_117_19_no_custom_OP.onnx",
                "spec_idx_101_eps_0.00000436.vnnlib",
            ),
            (
                "6cnn_ry_57_9_no_custom_OP.onnx",
                "spec_idx_129_eps_0.00000436.vnnlib",
            ),
            (
                "6cnn_ry_81_13_no_custom_OP.onnx",
                "spec_idx_144_eps_0.00000436.vnnlib",
            ),
            (
                "6cnn_ry_93_15_no_custom_OP.onnx",
                "spec_idx_148_eps_0.00000436.vnnlib",
            ),
        ];
        let mut violations = Vec::new();
        for (onnx_base, vnnlib_base) in GT_SAT {
            let onnx = format!("{BENCH_ROOT}/metaroom_2023/onnx/{onnx_base}");
            let vnnlib = format!("{BENCH_ROOT}/metaroom_2023/vnnlib/{vnnlib_base}");
            let Some(spec) = extract_twin_spec(Path::new(&onnx)) else {
                eprintln!("[metaroom moat] {vnnlib_base}: extraction fail-closed (safe)");
                continue;
            };
            let Some((lo, hi, t, adv)) = parse_vnnlib_robustness(Path::new(&vnnlib)) else {
                eprintln!("[metaroom moat] {vnnlib_base}: parse fail-closed (safe)");
                continue;
            };
            let deadline = Instant::now() + Duration::from_mins(1);
            match run_margin_row_lane(&spec, &lo, &hi, t, &adv, Some(deadline), 20_000) {
                MarginRowOutcome::Unsat(s, _) => {
                    violations.push((*vnnlib_base, s.root_bound));
                }
                MarginRowOutcome::Unknown { reason, .. } => {
                    eprintln!("[metaroom moat] {vnnlib_base}: correctly NOT verified ({reason})");
                }
            }
        }
        assert!(
            violations.is_empty(),
            "MOAT BROKEN on metaroom GT-sat rows: {violations:?}"
        );
    }

    #[cfg(test)]
    fn frozen_schedule_set(
        environment: &[(&str, &str)],
        config: &[(&str, ny_levers::LeverValue)],
    ) -> ny_levers::LeverSet {
        let registry = ny_levers::all();
        let raw = ny_levers::RawLeverInputs::capture_with(registry, |name| {
            environment
                .iter()
                .find(|(candidate, _)| *candidate == name)
                .map(|(_, value)| std::ffi::OsString::from(*value))
        });
        ny_levers::LeverSet::resolve_layered(registry, &raw, |decl| {
            config
                .iter()
                .find(|(candidate, _)| *candidate == decl.name)
                .map(|(_, value)| value.clone())
        })
        .expect("schedule fixture must satisfy the real declarations")
    }

    #[test]
    fn frozen_schedule_policy_preserves_effective_context_and_float_bits() {
        use ny_levers::decls::margin_row::{
            MARGIN_ROW_ADAPTIVE_RELEASE_FRAC, MARGIN_ROW_ADAPTIVE_RESERVE, MARGIN_ROW_CONCURRENT,
            MARGIN_ROW_CONCURRENT_THREADS, MARGIN_ROW_EARLY_EXIT, MARGIN_ROW_RESERVE_MAX_FRAC,
            MARGIN_ROW_RESERVE_SECS, VERIFIER_THREADS,
        };

        let typed_max = f64::from(0.25_f32);
        let typed_release = f64::from(0.47_f32);
        let env_max = 0.333_333_333_333_333_3_f64;
        let levers = frozen_schedule_set(
            &[
                (MARGIN_ROW_RESERVE_SECS.name, "82"),
                (MARGIN_ROW_ADAPTIVE_RESERVE.name, "0"),
                (MARGIN_ROW_RESERVE_MAX_FRAC.name, "0.3333333333333333"),
                (MARGIN_ROW_ADAPTIVE_RELEASE_FRAC.name, "0.375"),
                (MARGIN_ROW_CONCURRENT.name, " 1 "),
                (MARGIN_ROW_EARLY_EXIT.name, "garbage"),
                (VERIFIER_THREADS.name, "0"),
                (MARGIN_ROW_CONCURRENT_THREADS.name, "+3"),
            ],
            &[
                (MARGIN_ROW_RESERVE_SECS.name, ny_levers::LeverValue::U64(0)),
                (
                    MARGIN_ROW_ADAPTIVE_RESERVE.name,
                    ny_levers::LeverValue::Bool(true),
                ),
                (
                    MARGIN_ROW_RESERVE_MAX_FRAC.name,
                    ny_levers::LeverValue::F64(typed_max),
                ),
                (
                    MARGIN_ROW_ADAPTIVE_RELEASE_FRAC.name,
                    ny_levers::LeverValue::F64(typed_release),
                ),
                (
                    MARGIN_ROW_CONCURRENT.name,
                    ny_levers::LeverValue::Bool(false),
                ),
                (
                    MARGIN_ROW_EARLY_EXIT.name,
                    ny_levers::LeverValue::Bool(true),
                ),
                (VERIFIER_THREADS.name, ny_levers::LeverValue::U64(8)),
                (
                    MARGIN_ROW_CONCURRENT_THREADS.name,
                    ny_levers::LeverValue::U64(5),
                ),
            ],
        );
        let policy = MarginRowScheduleRuntimePolicy::from_levers(&levers).unwrap();

        assert_eq!(policy.configured_reserve_secs, 82);
        assert_eq!(policy.contextual_reserve_secs, 0);
        assert!(!policy.configured_adaptive_reserve);
        assert!(policy.contextual_adaptive_reserve);
        assert_eq!(
            policy.configured_reserve_max_frac_bits,
            Some(env_max.to_bits())
        );
        assert_eq!(
            policy.contextual_reserve_max_frac_bits,
            Some(typed_max.to_bits())
        );
        assert_eq!(
            policy.configured_adaptive_release_frac_bits,
            0.375_f64.to_bits()
        );
        assert_eq!(
            policy.contextual_adaptive_release_frac_bits,
            typed_release.to_bits()
        );
        assert!(policy.configured_concurrent);
        assert!(!policy.contextual_concurrent);
        assert!(!policy.configured_early_exit);
        assert!(policy.contextual_early_exit);
        assert_eq!(policy.configured_verifier_threads, Some(0));
        assert_eq!(policy.contextual_verifier_threads, Some(8));
        assert_eq!(policy.configured_concurrent_threads, Some(3));
        assert_eq!(policy.contextual_concurrent_threads, Some(5));
    }

    #[test]
    fn frozen_schedule_usize_ownership_preserves_kills_and_fallthrough() {
        use ny_levers::decls::margin_row::{MARGIN_ROW_CONCURRENT_THREADS, VERIFIER_THREADS};

        let verifier_rejected = frozen_schedule_set(
            &[(VERIFIER_THREADS.name, "malformed")],
            &[(VERIFIER_THREADS.name, ny_levers::LeverValue::U64(8))],
        );
        let policy = MarginRowScheduleRuntimePolicy::from_levers(&verifier_rejected).unwrap();
        assert_eq!(policy.configured_verifier_threads, None);
        assert_eq!(policy.contextual_verifier_threads, None);

        let concurrent_ignored = frozen_schedule_set(
            &[(MARGIN_ROW_CONCURRENT_THREADS.name, " 3 ")],
            &[(
                MARGIN_ROW_CONCURRENT_THREADS.name,
                ny_levers::LeverValue::U64(5),
            )],
        );
        let policy = MarginRowScheduleRuntimePolicy::from_levers(&concurrent_ignored).unwrap();
        assert_eq!(policy.configured_concurrent_threads, Some(5));
        assert_eq!(policy.contextual_concurrent_threads, None);
    }

    #[test]
    fn frozen_schedule_policy_rejects_an_ill_typed_runtime_value() {
        use ny_levers::decls::margin_row::{
            MARGIN_ROW_ADAPTIVE_RELEASE_FRAC, MARGIN_ROW_ADAPTIVE_RESERVE, MARGIN_ROW_CONCURRENT,
            MARGIN_ROW_CONCURRENT_THREADS, MARGIN_ROW_EARLY_EXIT, MARGIN_ROW_RESERVE_MAX_FRAC,
            MARGIN_ROW_RESERVE_SECS, VERIFIER_THREADS,
        };

        let levers = frozen_schedule_set(&[], &[]);
        let mut reserve = levers.resolved(&MARGIN_ROW_RESERVE_SECS).unwrap().clone();
        reserve.value = ny_levers::LeverValue::Bool(true);
        let error = MarginRowScheduleRuntimePolicy::from_resolved(
            &reserve,
            levers.resolved(&MARGIN_ROW_ADAPTIVE_RESERVE).unwrap(),
            levers.resolved(&MARGIN_ROW_RESERVE_MAX_FRAC).unwrap(),
            levers.resolved(&MARGIN_ROW_ADAPTIVE_RELEASE_FRAC).unwrap(),
            levers.resolved(&MARGIN_ROW_CONCURRENT).unwrap(),
            levers.resolved(&MARGIN_ROW_EARLY_EXIT).unwrap(),
            levers.resolved(&VERIFIER_THREADS).unwrap(),
            levers.resolved(&MARGIN_ROW_CONCURRENT_THREADS).unwrap(),
        )
        .unwrap_err();
        assert!(error.contains("resolved to non-u64"), "{error}");
    }

    #[test]
    fn schedule_policy_drives_cap_and_concurrent_without_second_authority() {
        let policy = MarginRowScheduleRuntimePolicy {
            configured_reserve_max_frac_bits: Some(0.25_f64.to_bits()),
            configured_concurrent: true,
            ..Default::default()
        };
        let decision = MarginRowReserveDecision {
            configured_secs: 45,
            reserve_secs: 45,
            route: MarginRowReserveRoute::Fixed,
        }
        .capped_to_internal_budget_with_policy(100, policy);
        assert_eq!(decision.reserve_secs, 25);
        assert!(concurrent_lane_armed_with_policy(policy));
    }

    #[test]
    fn concurrent_lane_pool_plan_uses_actual_capacity_and_exact_overrides() {
        let global = GlobalRayonOutcome {
            route: GlobalRayonRoute::Applied,
            requested_threads: Some(4),
            actual_threads: Some(4),
            host_threads: Some(20),
        };

        let shared = concurrent_lane_pool_plan(MarginRowScheduleRuntimePolicy::default(), global);
        assert_eq!(shared.pool_route, ConcurrentLanePoolRoute::SharedGlobal);
        assert_eq!(shared.lane_threads, Some(0));

        let explicit_zero = MarginRowScheduleRuntimePolicy {
            configured_verifier_threads: Some(4),
            configured_concurrent_threads: Some(0),
            ..Default::default()
        };
        let planned = concurrent_lane_pool_plan(explicit_zero, global);
        assert_eq!(planned.pool_route, ConcurrentLanePoolRoute::SharedGlobal);
        assert_eq!(planned.lane_threads, Some(0));

        let mut explicit = explicit_zero;
        explicit.configured_concurrent_threads = Some(7);
        let planned = concurrent_lane_pool_plan(explicit, global);
        assert_eq!(planned.pool_route, ConcurrentLanePoolRoute::ExplicitPrivate);
        assert_eq!(planned.lane_threads, Some(7));

        let derived = MarginRowScheduleRuntimePolicy {
            configured_verifier_threads: Some(4),
            ..Default::default()
        };
        for route in [
            GlobalRayonRoute::Applied,
            GlobalRayonRoute::AlreadyInitialized,
        ] {
            let planned =
                concurrent_lane_pool_plan(derived, GlobalRayonOutcome { route, ..global });
            assert_eq!(
                planned.pool_route,
                ConcurrentLanePoolRoute::VerifierRemainder
            );
            assert_eq!(planned.lane_threads, Some(16));
        }

        let exhausted = concurrent_lane_pool_plan(
            derived,
            GlobalRayonOutcome {
                actual_threads: Some(20),
                ..global
            },
        );
        assert_eq!(
            exhausted.pool_route,
            ConcurrentLanePoolRoute::NoSafeRemainder
        );
        assert_eq!(exhausted.lane_threads, None);

        let oversized = concurrent_lane_pool_plan(
            derived,
            GlobalRayonOutcome {
                actual_threads: Some(24),
                ..global
            },
        );
        assert_eq!(
            oversized.pool_route,
            ConcurrentLanePoolRoute::NoSafeRemainder
        );

        let unavailable = concurrent_lane_pool_plan(
            derived,
            GlobalRayonOutcome {
                host_threads: None,
                ..global
            },
        );
        assert_eq!(
            unavailable.pool_route,
            ConcurrentLanePoolRoute::GlobalUnavailable
        );
        assert_eq!(unavailable.lane_threads, None);
    }

    #[test]
    fn concurrent_admission_routes_and_overlap_state_are_exact() {
        assert_eq!(
            render_margin_row_concurrent_overlap_execution(),
            "schema=ny-cli/margin-row-concurrent-overlap-execution/v1 phase=work \
             worker_active=true verifier_active=true"
        );
        for (route, token) in [
            (ConcurrentLaneAdmissionRoute::Disabled, "disabled"),
            (
                ConcurrentLaneAdmissionRoute::AuthorityDisabled,
                "authority_disabled",
            ),
            (ConcurrentLaneAdmissionRoute::CpuDevice, "cpu_device"),
            (ConcurrentLaneAdmissionRoute::NoDeadline, "no_deadline"),
            (ConcurrentLaneAdmissionRoute::ShortBudget, "short_budget"),
            (ConcurrentLaneAdmissionRoute::Model, "model"),
            (ConcurrentLaneAdmissionRoute::Property, "property"),
            (ConcurrentLaneAdmissionRoute::InputShape, "input_shape"),
            (
                ConcurrentLaneAdmissionRoute::NoSafeRemainder,
                "no_safe_remainder",
            ),
            (
                ConcurrentLaneAdmissionRoute::GlobalUnavailable,
                "global_unavailable",
            ),
            (
                ConcurrentLaneAdmissionRoute::PrivatePoolFailed,
                "private_pool_failed",
            ),
            (ConcurrentLaneAdmissionRoute::SpawnFailed, "spawn_failed"),
            (ConcurrentLaneAdmissionRoute::Spawned, "spawned"),
        ] {
            assert_eq!(route.stable_token(), token);
        }
        for (route, token) in [
            (GlobalRayonRoute::NotRequested, "not_requested"),
            (GlobalRayonRoute::Applied, "applied"),
            (GlobalRayonRoute::AlreadyInitialized, "already_initialized"),
        ] {
            assert_eq!(route.stable_token(), token);
        }
        for (route, token) in [
            (ConcurrentLanePoolRoute::NotConsulted, "not_consulted"),
            (ConcurrentLanePoolRoute::SharedGlobal, "shared_global"),
            (ConcurrentLanePoolRoute::ExplicitPrivate, "explicit_private"),
            (
                ConcurrentLanePoolRoute::VerifierRemainder,
                "verifier_remainder",
            ),
            (
                ConcurrentLanePoolRoute::NoSafeRemainder,
                "no_safe_remainder",
            ),
            (
                ConcurrentLanePoolRoute::GlobalUnavailable,
                "global_unavailable",
            ),
        ] {
            assert_eq!(route.stable_token(), token);
        }
        for (route, token) in [
            (PrivatePoolRoute::NotRequested, "not_requested"),
            (PrivatePoolRoute::Built, "built"),
            (PrivatePoolRoute::BuildFailed, "build_failed"),
        ] {
            assert_eq!(route.stable_token(), token);
        }

        let state = ConcurrentLaneExecutionState::new(false);
        assert!(!state
            .allocation_receipt_recorded
            .load(std::sync::atomic::Ordering::Acquire));
        state.mark_worker_started();
        assert!(!state
            .overlap_recorded
            .load(std::sync::atomic::Ordering::SeqCst));
        state.mark_verifier_started();
        assert!(state
            .overlap_recorded
            .load(std::sync::atomic::Ordering::SeqCst));
        state.mark_verifier_started();
        assert!(state
            .overlap_recorded
            .load(std::sync::atomic::Ordering::SeqCst));
        state.mark_verifier_finished();
        state.mark_worker_finished();
    }

    /// The ceiling declines for every input that is not a finite fraction
    /// strictly inside `(0, 1)`, so an unset/typo'd/hostile value can never
    /// silently change the shipped scheduling policy.
    #[test]
    fn reserve_max_fraction_accepts_only_proper_fractions() {
        assert_eq!(reserve_max_fraction(Some("0.25")), Some(0.25));
        assert_eq!(reserve_max_fraction(Some("0.5")), Some(0.5));
        for declined in [
            None,
            Some(""),
            Some("0"),
            Some("0.0"),
            Some("1"),
            Some("1.0"),
            Some("1.5"),
            Some("-0.25"),
            Some("nan"),
            Some("inf"),
            Some("25%"),
            Some("a quarter"),
        ] {
            assert_eq!(
                reserve_max_fraction(declined),
                None,
                "{declined:?} must decline the ceiling"
            );
        }
    }

    /// No ceiling ⇒ the reserve is returned untouched. This is the shipped
    /// default path and it must stay byte-identical.
    #[test]
    fn capped_reserve_without_a_fraction_is_the_identity() {
        for budget in [0, 10, 95, 190, 600] {
            assert_eq!(capped_reserve_secs(45, budget, None), 45);
        }
    }

    /// The ceiling binds only when it is BELOW the configured reserve — it can
    /// never hand the lane MORE time than the fixed policy asked for.
    #[test]
    fn capped_reserve_only_ever_shrinks_the_reserve() {
        // The measured cifar100 case: 45 s of a 95 s internal tier is 47%.
        assert_eq!(capped_reserve_secs(45, 95, Some(0.25)), 23);
        // Same fixed reserve, double the budget: the ceiling stops binding.
        assert_eq!(capped_reserve_secs(45, 190, Some(0.25)), 45);
        // A generous ceiling is inert at either budget.
        assert_eq!(capped_reserve_secs(45, 95, Some(0.9)), 45);
        // Already-released reserves stay released.
        assert_eq!(capped_reserve_secs(0, 95, Some(0.25)), 0);
        // Degenerate budgets floor to zero rather than panicking.
        assert_eq!(capped_reserve_secs(45, 0, Some(0.25)), 0);
    }

    #[cfg(test)]
    fn twin_gemm(input: usize, n_out: usize, n_in: usize) -> TwinOpSpec {
        TwinOpSpec::Gemm {
            input,
            weight: vec![0.0; n_out * n_in],
            bias: vec![0.0; n_out],
            bias_err: Vec::new(),
            weight_rel_err: 0.0,
            shape: (n_out, n_in),
        }
    }

    /// #twinwall-reserve-admission: the reserve probe must accept EXACTLY the
    /// head shape `TwinNet::compile` accepts. malbeware
    /// (`Conv -> ReLU -> Flatten -> Gemm`) has ONE Gemm, so the compiler
    /// refuses it — the probe must refuse it too, or the reserve holds back
    /// 45 s the lane can never spend.
    #[test]
    fn reserve_admission_matches_the_compiler_head_contract() {
        // Twin-wall shape: trunk ReLU, then Gemm -> ReLU -> Gemm at the tail.
        let ok = TwinSpec {
            n_in: 4,
            ops: vec![
                TwinOpSpec::Relu { input: 0 },
                twin_gemm(1, 3, 4),
                TwinOpSpec::Relu { input: 2 },
                twin_gemm(3, 2, 3),
            ],
        };
        assert!(twin_spec_compiles_to_margin_row_head(&ok));

        // The malbeware shape, minus the Conv the predicate never inspects.
        let malbeware_shaped = TwinSpec {
            n_in: 4,
            ops: vec![
                TwinOpSpec::Relu { input: 0 },
                TwinOpSpec::Flatten { input: 1 },
                twin_gemm(2, 2, 4),
            ],
        };
        assert!(
            !twin_spec_compiles_to_margin_row_head(&malbeware_shaped),
            "one-Gemm head must NOT arm the reserve"
        );

        // Three Gemms: also outside the contract.
        let three_gemms = TwinSpec {
            n_in: 4,
            ops: vec![
                TwinOpSpec::Relu { input: 0 },
                twin_gemm(1, 3, 4),
                TwinOpSpec::Relu { input: 2 },
                twin_gemm(3, 3, 3),
                twin_gemm(4, 2, 3),
            ],
        };
        assert!(!twin_spec_compiles_to_margin_row_head(&three_gemms));

        // Right Gemm count, wrong wiring: no ReLU between the two heads.
        let no_head_relu = TwinSpec {
            n_in: 4,
            ops: vec![
                TwinOpSpec::Relu { input: 0 },
                twin_gemm(1, 3, 4),
                TwinOpSpec::Flatten { input: 2 },
                twin_gemm(3, 2, 3),
            ],
        };
        assert!(!twin_spec_compiles_to_margin_row_head(&no_head_relu));

        // Right head, but no TRUNK ReLU before Gemm1 — the compiler's
        // "no trunk relus" refusal.
        let no_trunk_relu = TwinSpec {
            n_in: 4,
            ops: vec![
                twin_gemm(0, 3, 4),
                TwinOpSpec::Relu { input: 1 },
                twin_gemm(2, 2, 3),
            ],
        };
        assert!(!twin_spec_compiles_to_margin_row_head(&no_trunk_relu));
    }

    /// The typed `margin_row.reserve_max_frac` key arms the ceiling only for
    /// proper fractions, exactly like the environment form. A malformed or
    /// out-of-range typed value must NOT arm the gate — it declines and the
    /// shipped fixed-seconds policy stands.
    #[test]
    fn typed_reserve_max_frac_arms_only_on_proper_fractions() {
        assert_eq!(resolve_reserve_max_frac(None, Some(0.25)), Some(0.25));
        assert_eq!(resolve_reserve_max_frac(None, Some(0.5)), Some(0.5));
        for declined in [
            None,
            Some(0.0f32),
            Some(-0.0),
            Some(-0.25),
            Some(1.0),
            Some(1.5),
            Some(f32::NAN),
            Some(f32::INFINITY),
            Some(f32::NEG_INFINITY),
        ] {
            assert_eq!(
                resolve_reserve_max_frac(None, declined),
                None,
                "typed {declined:?} must decline the ceiling"
            );
        }
    }

    /// Precedence: the environment wins wherever it is PRESENT, and a present
    /// value the parser declines resolves to "no ceiling" WITHOUT falling back
    /// to the preset — that is the exact kill switch for a shipped preset key.
    #[test]
    fn reserve_max_frac_env_overrides_and_kills_the_typed_key() {
        // Env absent ⇒ the typed key decides.
        assert_eq!(resolve_reserve_max_frac(None, Some(0.25)), Some(0.25));
        // Env present and valid ⇒ env wins over the typed key.
        assert_eq!(resolve_reserve_max_frac(Some("0.5"), Some(0.25)), Some(0.5));
        // Env present but declined ⇒ NO ceiling, typed key is not consulted.
        for kill in ["0", "0.0", "1", "1.0", "-0.25", "nan", "", "a quarter"] {
            assert_eq!(
                resolve_reserve_max_frac(Some(kill), Some(0.25)),
                None,
                "{kill:?} must kill the typed ceiling rather than fall through"
            );
        }
    }

    /// End-to-end through a real preset file: absent key ⇒ no ceiling and the
    /// reserve is untouched; a proper fraction ⇒ the ceiling binds. Reads the
    /// environment, which is unset in the test process (nothing here sets it).
    #[test]
    fn typed_reserve_max_frac_round_trips_through_a_preset_file() {
        let dir = std::env::temp_dir().join(format!("ny_mr_frac_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmpdir");

        // No preset at all ⇒ no ceiling.
        assert_eq!(margin_row_reserve_max_frac(None), None);

        // A preset without the key ⇒ still no ceiling (byte-identical path).
        let plain = dir.join("plain.yaml");
        std::fs::write(&plain, "margin_row:\n  reserve_secs: 45\n").expect("write");
        assert_eq!(margin_row_reserve_max_frac(Some(&plain)), None);
        let decision = MarginRowReserveDecision {
            configured_secs: 45,
            reserve_secs: 45,
            route: MarginRowReserveRoute::Fixed,
        };
        assert_eq!(
            decision
                .capped_to_internal_budget(95, Some(&plain))
                .reserve_secs,
            45,
            "an absent reserve_max_frac must leave the shipped reserve alone"
        );

        // A preset naming the key ⇒ the ceiling binds, route untouched.
        let capped = dir.join("capped.yaml");
        std::fs::write(
            &capped,
            "margin_row:\n  reserve_secs: 45\n  reserve_max_frac: 0.25\n",
        )
        .expect("write");
        assert_eq!(margin_row_reserve_max_frac(Some(&capped)), Some(0.25));
        let applied = decision.capped_to_internal_budget(95, Some(&capped));
        assert_eq!(applied.reserve_secs, 23);
        assert_eq!(applied.configured_secs, 45);
        assert_eq!(applied.route, MarginRowReserveRoute::Fixed);

        // A malformed typed value declines rather than arming anything.
        let bogus = dir.join("bogus.yaml");
        std::fs::write(&bogus, "margin_row:\n  reserve_max_frac: 1.0\n").expect("write");
        assert_eq!(margin_row_reserve_max_frac(Some(&bogus)), None);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Landing the KEY is the deliverable; ARMING it needs a measured A/B we
    /// have not run. No shipped preset may name `margin_row.reserve_max_frac`
    /// until that exists, so every shipped category stays byte-identical.
    #[test]
    fn no_shipped_preset_arms_the_reserve_ceiling() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
        let mut armed: Vec<String> = Vec::new();
        for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
            let dir = root.join(year);
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries {
                let path = entry.expect("dir entry").path();
                if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                    continue;
                }
                let text = std::fs::read_to_string(&path).expect("readable preset");
                let preset: crate::preset::PresetConfig = serde_yaml::from_str(&text)
                    .unwrap_or_else(|e| panic!("preset {} must parse: {e}", path.display()));
                if preset.margin_row.reserve_max_frac.is_some() {
                    armed.push(format!("{year}/{}", path.display()));
                }
            }
        }
        assert!(
            armed.is_empty(),
            "shipped presets must not arm the reserve ceiling without a measured A/B: {armed:?}"
        );
    }

    /// An already-released reserve skips the ceiling path entirely (and with it
    /// the preset load) and stays released.
    #[test]
    fn capped_to_internal_budget_is_inert_on_a_released_reserve() {
        let released = MarginRowReserveDecision {
            configured_secs: 45,
            reserve_secs: 0,
            route: MarginRowReserveRoute::AdaptiveReleasedAlphaBetaTier,
        };
        let capped = released.capped_to_internal_budget(95, Some(Path::new("/nonexistent.yaml")));
        assert_eq!(capped.reserve_secs, 0);
        assert!(capped.enables_scored_sparse_crown());
    }

    /// The ceiling is a SCHEDULING knob: it must not alter the route, because
    /// the route is what authorizes scored sparse root CROWN.
    #[test]
    fn capped_reserve_preserves_the_route_and_configured_seconds() {
        let decision = MarginRowReserveDecision {
            configured_secs: 45,
            reserve_secs: 45,
            route: MarginRowReserveRoute::AdaptivePreserved,
        };
        let capped = decision.capped_to_internal_budget(95, None);
        assert_eq!(capped.route, MarginRowReserveRoute::AdaptivePreserved);
        assert_eq!(capped.configured_secs, 45);
        assert!(!capped.enables_scored_sparse_crown());
        // With the env unset (the default in the test process) the ceiling
        // declines, so this is also an identity check on the shipped path.
        assert_eq!(capped.reserve_secs, 45);
    }

    /// The reserve resolves to the shipped 45 s default when neither the
    /// environment nor a category preset overrides it. This pins that
    /// resolution behavior (#epoch-bab); it does not establish score impact.
    #[test]
    fn margin_row_reserve_defaults_to_shipped_value_and_is_preset_scoped() {
        // No preset -> the shipped value.
        assert_eq!(margin_row_reserve_secs(None), 45);

        let dir = std::env::temp_dir().join(format!("ny_mr_preset_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmpdir");

        // A preset without a margin_row section -> still the default.
        let plain = dir.join("plain.yaml");
        std::fs::write(&plain, "general: {}\n").expect("write");
        assert_eq!(margin_row_reserve_secs(Some(&plain)), 45);

        // Opt-in is read from the preset.
        let opted = dir.join("opted.yaml");
        std::fs::write(&opted, "margin_row:\n  reserve_secs: 25\n").expect("write");
        assert_eq!(margin_row_reserve_secs(Some(&opted)), 25);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn adaptive_reserve_policy_is_typed_exact_and_fail_closed() {
        assert!(!adaptive_reserve_enabled(
            false,
            AdaptiveReserveGateEnv::Absent
        ));
        assert!(adaptive_reserve_enabled(
            true,
            AdaptiveReserveGateEnv::Absent
        ));
        assert!(adaptive_reserve_enabled(
            false,
            AdaptiveReserveGateEnv::Unicode("1")
        ));
        for malformed in ["0", "", " 1", "1 ", "true", "ON", "yes", "01"] {
            assert!(
                !adaptive_reserve_enabled(true, AdaptiveReserveGateEnv::Unicode(malformed)),
                "present non-exact gate {malformed:?} must retain the fixed reserve"
            );
        }
        assert!(!adaptive_reserve_enabled(
            true,
            AdaptiveReserveGateEnv::NonUnicode
        ));
        assert_eq!(
            adaptive_reserve_gate_env(None),
            AdaptiveReserveGateEnv::Absent
        );
    }

    /// The release predicate is budget arithmetic, so it must reproduce the
    /// two MEASURED regimes that motivated it and must not depend on the
    /// identity of any model or property file.
    #[test]
    fn adaptive_release_is_structural_budget_share_not_instance_identity() {
        let frac = DEFAULT_ADAPTIVE_RELEASE_FRAC;

        // Measured 2026-07-26 (cifar100_2024 prop_idx_9502_sidx_7197):
        // scored 100 s -> 95 s internal tier, 45 s reserve = 47.4% -> RELEASE.
        assert!(
            adaptive_release_target(45, 95, frac),
            "45s of a 95s internal tier is 47% — the regime where the reserve bought nothing"
        );
        // scored 200 s -> 190 s internal tier, 45 s reserve = 23.7% -> PRESERVE.
        assert!(
            !adaptive_release_target(45, 190, frac),
            "45s of a 190s internal tier is 24% — the regime where the lane had airtime"
        );

        // Exactly at the threshold releases; a hair under preserves.
        assert!(adaptive_release_target(40, 100, frac));
        assert!(!adaptive_release_target(39, 100, frac));

        // Degenerate budgets keep the established fixed lane.
        assert!(!adaptive_release_target(45, 0, frac));
        // A zero reserve is never a release target (the caller short-circuits
        // it to NotApplicable, but the predicate must agree).
        assert!(!adaptive_release_target(0, 95, frac));

        // Threshold resolution: env > preset > default, each fail-safe.
        assert_eq!(resolve_release_frac(None, None), frac);
        assert_eq!(resolve_release_frac(Some("0.25"), None), 0.25);
        assert_eq!(resolve_release_frac(None, Some(0.25)), 0.25);
        assert_eq!(resolve_release_frac(Some("0.25"), Some(0.9)), 0.25);
        for bad in ["", "abc", "0", "1", "1.5", "-0.2", "NaN", "inf"] {
            assert_eq!(
                resolve_release_frac(Some(bad), None),
                frac,
                "malformed threshold {bad:?} must fall back to the shipped default"
            );
        }
        for bad in [0.0f32, 1.0, 1.5, -0.2, f32::NAN, f32::INFINITY] {
            assert_eq!(resolve_release_frac(None, Some(bad)), frac);
        }
    }

    /// The removed allowlist keyed on `CIFAR100_resnet_medium.onnx` plus seven
    /// exact property basenames. Nothing in the reserve path may consult a
    /// model or property NAME again — the decision is budget-only.
    #[test]
    fn reserve_path_carries_no_instance_name_literals() {
        assert_eq!(RESERVE_MAX_FRAC_ENV, "NY_MARGIN_ROW_RESERVE_MAX_FRAC");
        assert_eq!(RELEASE_FRAC_ENV, "NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC");
        let src = include_str!("margin_row_bab.rs");
        let reserve_region = src
            .split_once("pub(crate) const RELEASE_FRAC_ENV")
            .expect("release threshold block present")
            .1
            .split_once("pub(crate) fn margin_row_reserve_secs")
            .expect("fixed-reserve resolver boundary present")
            .0;
        for needle in [
            "CIFAR100_resnet_medium.onnx",
            "prop_idx_",
            "ADAPTIVE_RELEASE_CIFAR100_MEDIUM",
        ] {
            assert!(
                !reserve_region.contains(needle),
                "instance-identity literal {needle:?} reintroduced on the reserve path"
            );
        }
    }

    #[test]
    fn adaptive_reserve_is_default_off_fail_open_and_preserves_closure_sentinels() {
        assert_eq!(
            reserve_policy(45, false, true),
            MarginRowReserveDecision {
                configured_secs: 45,
                reserve_secs: 45,
                route: MarginRowReserveRoute::Fixed,
            },
            "unset/default policy must be byte-for-policy identical even on a target row"
        );
        assert_eq!(
            reserve_policy(45, true, false),
            MarginRowReserveDecision {
                configured_secs: 45,
                reserve_secs: 45,
                route: MarginRowReserveRoute::AdaptivePreserved,
            },
            "unknown adaptive rows retain the established lane budget"
        );
        assert_eq!(
            reserve_policy(45, true, true),
            MarginRowReserveDecision {
                configured_secs: 45,
                reserve_secs: 0,
                route: MarginRowReserveRoute::AdaptiveReleasedAlphaBetaTier,
            }
        );
        assert!(reserve_policy(45, true, true).enables_scored_sparse_crown());
        assert!(!reserve_policy(45, true, false).enables_scored_sparse_crown());
        assert!(!reserve_policy(45, false, true).enables_scored_sparse_crown());
        assert_eq!(
            reserve_policy(0, true, true),
            MarginRowReserveDecision {
                configured_secs: 0,
                reserve_secs: 0,
                route: MarginRowReserveRoute::NotApplicable,
            }
        );

        // The historical margin-row closures were all measured at LONG scored
        // budgets, where the reserve is a small share of the internal tier and
        // the lane demonstrably had airtime. The structural predicate must
        // leave that whole regime on the fixed route, whatever the instance is
        // called. cifar100/tinyimagenet run a 200 s scored budget (190 s
        // internal) and metaroom 210 s (199 s internal).
        for internal_budget in [190u64, 199, 300, 600] {
            assert!(
                !adaptive_release_target(45, internal_budget, DEFAULT_ADAPTIVE_RELEASE_FRAC),
                "long-budget closure regime must retain the reserve ({internal_budget}s internal)"
            );
        }
    }

    /// The concurrent-lane gate: env wins in BOTH directions wherever present,
    /// the typed preset key decides when it is absent, and absent-from-both is
    /// OFF (byte-identical to the shipped reserve-only path).
    #[test]
    fn concurrent_lane_gate_resolution_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        let on = dir.path().join("on.yaml");
        std::fs::write(&on, "margin_row:\n  concurrent: true\n").expect("write");
        let off = dir.path().join("off.yaml");
        std::fs::write(&off, "margin_row:\n  concurrent: false\n").expect("write");
        let silent = dir.path().join("silent.yaml");
        std::fs::write(&silent, "margin_row:\n  reserve_secs: 45\n").expect("write");

        // Env absent: the preset decides; a preset that never mentions the key
        // keeps the shipped OFF behaviour.
        assert!(concurrent_lane_armed(Some(&on)));
        assert!(!concurrent_lane_armed(Some(&off)));
        assert!(!concurrent_lane_armed(Some(&silent)));
        assert!(!concurrent_lane_armed(None));
    }

    /// The typed f32 route must never be able to TIGHTEN anything: it only
    /// selects a tableau precision whose rounding is charged into a certified
    /// additive slack. Here we only pin the plumbing — that the preset key is
    /// read, and that a preset which stays silent leaves the flag alone.
    #[test]
    fn root_f32_preset_key_is_read() {
        let dir = tempfile::tempdir().expect("tempdir");
        let on = dir.path().join("on.yaml");
        std::fs::write(&on, "margin_row:\n  root_f32: true\n").expect("write");
        let off = dir.path().join("off.yaml");
        std::fs::write(&off, "margin_row:\n  root_f32: false\n").expect("write");
        let silent = dir.path().join("silent.yaml");
        std::fs::write(&silent, "margin_row:\n  reserve_secs: 0\n").expect("write");

        assert_eq!(root_f32_from_preset(Some(&on)), Some(true));
        assert_eq!(root_f32_from_preset(Some(&off)), Some(false));
        assert_eq!(root_f32_from_preset(Some(&silent)), None);
        assert_eq!(root_f32_from_preset(None), None);
    }

    /// `#tableau-f32` (`margin_row.tableau_f32`): pin the schema half of the
    /// delivery route — the key parses, in both directions, and a silent
    /// preset leaves it `None` so the propagate-side cell stays unwritten and
    /// the lever resolves exactly as before promotion. The cell write itself
    /// is `preset_gates::apply`'s one-liner, exercised by
    /// `validate_delivered_lever_values` on every preset load (a compiled
    /// `Deliver::PresetKey` axis whose dotted key is missing from the schema
    /// is a load-time error), so reachability cannot silently regress to
    /// `Source::LegacyEnv`-only.
    #[test]
    fn tableau_f32_preset_key_is_read() {
        let dir = tempfile::tempdir().expect("tempdir");
        let on = dir.path().join("on.yaml");
        std::fs::write(&on, "margin_row:\n  tableau_f32: true\n").expect("write");
        let off = dir.path().join("off.yaml");
        std::fs::write(&off, "margin_row:\n  tableau_f32: false\n").expect("write");
        let silent = dir.path().join("silent.yaml");
        std::fs::write(&silent, "margin_row:\n  reserve_secs: 0\n").expect("write");

        let load = |p: &Path| {
            crate::preset::load_preset(p)
                .expect("preset loads")
                .margin_row
                .tableau_f32
        };
        assert_eq!(load(&on), Some(true));
        assert_eq!(load(&off), Some(false));
        assert_eq!(load(&silent), None);
    }

    /// A preset that arms the concurrent lane must not ALSO bill the internal
    /// verifier for a reserve: it would lose the reserved slice AND contend
    /// with the lane, the untested combination 31949bcc refused.
    ///
    /// This checks the EFFECTIVE reserve at the category's official scored
    /// budget, through the real [`reserve_policy`] — not one preset key — so
    /// both shipped ways of satisfying the invariant are covered and neither
    /// can drift:
    ///   * `reserve_secs: 0` (tinyimagenet_2024) never takes a reserve at all;
    ///   * `adaptive_reserve: true` (cifar100_2024) RELEASES the whole reserve
    ///     whenever the configured seconds reach `release_frac` of the internal
    ///     budget — for the shipped 45 s / 0.40 pair that is every internal
    ///     tier up to 112 s, and a scored 100 s budget is a 95 s tier.
    ///
    /// Requiring the budget to be listed here is deliberate: a new concurrent
    /// preset must state the budget its evidence was gathered at, because the
    /// adaptive release is budget-dependent and silently stops firing on long
    /// tiers.
    #[test]
    fn concurrent_presets_never_also_hold_a_reserve() {
        // Official scored per-instance budgets from each category's
        // instances.csv. #cct-margin-row-admission: cct has THREE tiers, and
        // the adaptive release is budget-dependent, so every tier is checked.
        const OFFICIAL_SCORED_SECS: &[(&str, &[u64])] = &[
            ("cifar100_2024", &[100]),
            ("tinyimagenet_2024", &[100]),
            ("challenging_certified_training_2026", &[30, 120, 550]),
        ];

        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root");
        let mut concurrent_presets_checked = 0usize;
        // vnncomp26 SHADOWS vnncomp25; a concurrent preset in either year is a
        // scored configuration.
        for year in ["configs/vnncomp25", "configs/vnncomp26"] {
            let dir = root.join(year);
            let entries = std::fs::read_dir(&dir).unwrap_or_else(|error| {
                panic!("read shipped preset directory {}: {error}", dir.display())
            });
            for entry in entries {
                let entry = entry
                    .unwrap_or_else(|error| panic!("read entry under {}: {error}", dir.display()));
                let path = entry.path();
                if path.extension().is_none_or(|e| e != "yaml") {
                    continue;
                }
                // Only a preset with a `margin_row:` section can arm the lane;
                // skip the rest before loading (their loadability is pinned by
                // `every_shipped_vnncomp_preset_loads_with_no_unknown_keys`).
                let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
                    panic!("read shipped preset {}: {error}", path.display())
                });
                if !text.lines().any(|l| l.starts_with("margin_row:")) {
                    continue;
                }
                let cfg = crate::preset::load_preset(&path).unwrap_or_else(|error| {
                    panic!("load shipped preset {}: {error}", path.display())
                });
                if cfg.margin_row.concurrent != Some(true) {
                    continue;
                }
                let category = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .expect("preset file stem");
                // A non-resolvable canary mirrors its production preset
                // byte-for-byte outside its own keys; it is checked through the
                // production row, not registered separately.
                if category.ends_with("_canary") {
                    continue;
                }
                concurrent_presets_checked += 1;
                let tiers = OFFICIAL_SCORED_SECS
                    .iter()
                    .find(|(name, _)| *name == category)
                    .map(|(_, secs)| *secs)
                    .unwrap_or_else(|| {
                        panic!(
                            "{} arms the concurrent lane but has no official scored budget \
                             recorded here; the adaptive reserve release is budget-dependent, \
                             so the budget the evidence was gathered at must be stated",
                            path.display()
                        )
                    });
                for &scored in tiers {
                    // Bind to the PRODUCTION tier, not a local mirror. This test
                    // used to call a `#[cfg(test)] internal_timeout_tier` copy that
                    // main deleted; the copy was a drift hazard anyway (its own doc
                    // called it a "mirror of vnncomp::internal_timeout_secs"), and a
                    // stale mirror would let this reserve assertion pass against a
                    // rule production no longer uses.
                    let internal = crate::commands::vnncomp::internal_timeout_secs(scored);
                    let configured = cfg.margin_row.reserve_secs.unwrap_or(45);
                    let decision = reserve_policy(
                        configured,
                        cfg.margin_row.adaptive_reserve.unwrap_or(false),
                        adaptive_release_target(
                            configured,
                            internal,
                            resolve_release_frac(None, cfg.margin_row.release_frac),
                        ),
                    );
                    assert_eq!(
                        decision.reserve_secs,
                        0,
                        "{} arms the concurrent lane but still holds {}s of the {internal}s \
                         internal tier (scored {scored}s, route={:?}); the pair taxes the \
                         internal verifier twice",
                        path.display(),
                        decision.reserve_secs,
                        decision.route,
                    );
                }
            }
        }
        assert_eq!(
            concurrent_presets_checked,
            OFFICIAL_SCORED_SECS.len(),
            "the concurrent-preset invariant checked {concurrent_presets_checked} shipped \
             preset(s), but the sealed budget registry names {}; a missing/unreadable preset or \
             stale registry must not turn the test into a vacuous pass",
            OFFICIAL_SCORED_SECS.len()
        );
    }

    /// Shipped presets must retain the global reserve until a sealed A/B
    /// justifies a category override. Historical scorecards are not sealed
    /// production evidence, so merely fitting an override beneath an old
    /// solve time is not authority to ship it.
    #[test]
    fn shipped_presets_do_not_claim_unmeasured_reserve_overrides() {
        const PRESETS: &[&str] = &[
            "configs/vnncomp25/metaroom_2023.yaml",
            "configs/vnncomp25/sat_relu.yaml",
            "configs/vnncomp25/cifar100_2024.yaml",
        ];
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root")
            .to_path_buf();
        for preset_rel in PRESETS {
            let preset = root.join(preset_rel);
            assert!(
                preset.is_file(),
                "required shipped preset is missing: {}",
                preset.display()
            );
            let config = crate::preset::load_preset(&preset).unwrap_or_else(|error| {
                panic!("load shipped preset {}: {error}", preset.display())
            });
            assert_eq!(
                config.margin_row.reserve_secs, None,
                "{preset_rel} claims a category reserve override without a sealed A/B"
            );
        }
    }
    /// BATCH SWEEP (#epoch-bab): run the lane over a target list, one line of
    /// evidence per instance. Env:
    ///   NY_SWEEP_TARGETS = file of `onnx_basename,vnnlib_basename` lines
    ///   NY_SWEEP_CAT     = benchmark category dir (default cifar100_2024)
    ///   NY_SWEEP_SECS    = per-instance budget (default 100)
    ///   NY_SWEEP_SKIP/NY_SWEEP_TAKE = shard the list
    pub(crate) fn sweep_targets() {
        let targets = std::env::var("NY_SWEEP_TARGETS").expect("set NY_SWEEP_TARGETS");
        let cat = std::env::var("NY_SWEEP_CAT").unwrap_or_else(|_| "cifar100_2024".into());
        let secs: u64 = std::env::var("NY_SWEEP_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);
        let skip: usize = std::env::var("NY_SWEEP_SKIP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let take: usize = std::env::var("NY_SWEEP_TAKE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(usize::MAX);
        let text = std::fs::read_to_string(&targets).expect("target list");
        let rows: Vec<(String, String)> = text
            .lines()
            .filter_map(|l| l.split_once(','))
            .map(|(a, b)| (a.trim().to_string(), b.trim().to_string()))
            .skip(skip)
            .take(take)
            .collect();
        let (mut unsat, mut unknown, mut skipped) = (0usize, 0usize, 0usize);
        for (i, (ob, vb)) in rows.iter().enumerate() {
            let onnx = format!("{BENCH_ROOT}/{cat}/onnx/{ob}");
            let vnnlib = format!("{BENCH_ROOT}/{cat}/vnnlib/{vb}");
            if !Path::new(&onnx).exists() || !Path::new(&vnnlib).exists() {
                eprintln!("[sweep {i}] MISSING {ob} / {vb}");
                skipped += 1;
                continue;
            }
            let Some(spec) = extract_twin_spec(Path::new(&onnx)) else {
                eprintln!("[sweep {i}] {vb} -> extract fail-closed");
                skipped += 1;
                continue;
            };
            let Some((lo, hi, t, adv)) = parse_vnnlib_robustness(Path::new(&vnnlib)) else {
                eprintln!("[sweep {i}] {vb} -> parse fail-closed");
                skipped += 1;
                continue;
            };
            let t0 = Instant::now();
            let deadline = t0 + Duration::from_secs(secs);
            let out = run_margin_row_lane(&spec, &lo, &hi, t, &adv, Some(deadline), 20_000);
            let el = t0.elapsed().as_secs_f64();
            match out {
                MarginRowOutcome::Unsat(st, _) => {
                    unsat += 1;
                    eprintln!(
                        "[sweep {i}] {vb} -> UNSAT {el:.1}s root={:.4} exp={} ledger={:?}",
                        st.root_bound, st.expansions, st.ledger_ok
                    );
                }
                MarginRowOutcome::Unknown { reason, stats } => {
                    unknown += 1;
                    eprintln!(
                        "[sweep {i}] {vb} -> unknown({reason}) {el:.1}s root={:?}",
                        stats.map(|s| s.root_bound)
                    );
                }
            }
        }
        eprintln!(
            "[sweep] SUMMARY cat={cat} budget={secs}s rows={} UNSAT={unsat} unknown={unknown} skipped={skipped}",
            rows.len()
        );
    }

    fn require_probe_instance() -> anyhow::Result<()> {
        let category = std::env::var("NY_PROBE_CAT").unwrap_or_else(|_| "cifar100_2024".to_owned());
        let default_onnx = if category.starts_with("tiny") {
            "TinyImageNet_resnet_medium.onnx"
        } else {
            "CIFAR100_resnet_medium.onnx"
        };
        let onnx = std::env::var("NY_PROBE_ONNX").unwrap_or_else(|_| default_onnx.to_owned());
        let vnnlib = std::env::var("NY_PROBE_VNNLIB")
            .map_err(|_| anyhow::anyhow!("this probe requires NY_PROBE_VNNLIB"))?;
        for path in [
            Path::new(BENCH_ROOT)
                .join(&category)
                .join("onnx")
                .join(onnx),
            Path::new(BENCH_ROOT)
                .join(category)
                .join("vnnlib")
                .join(vnnlib),
        ] {
            anyhow::ensure!(
                path.is_file(),
                "missing margin-row prerequisite {}",
                path.display()
            );
        }
        Ok(())
    }

    /// Execute one explicit, serial real-corpus margin-row measurement.
    pub(crate) fn run(probe: &str) -> anyhow::Result<()> {
        // #col-scaling-v2 (plan B1(b)): arguments ride in the selector and the
        // corpus is addressed by ABSOLUTE path, so it runs from a worktree
        // build that has no in-tree `benchmarks/` — dispatched before that
        // precondition below.
        if let Some(args) = probe.strip_prefix("col-scaling-v2") {
            return col_scaling_v2::run(args);
        }
        anyhow::ensure!(
            Path::new(BENCH_ROOT).is_dir(),
            "missing VNN-COMP benchmark root {BENCH_ROOT}"
        );
        match probe {
            "instance" => {
                require_probe_instance()?;
                probe_env_instance();
            }
            "root-build" => {
                require_probe_instance()?;
                probe_root_build();
            }
            "pass-scaling" => {
                require_probe_instance()?;
                probe_pass_scaling();
            }
            // #percol-alpha: self-contained (defaults to cifar100 resnet_medium
            // idx_4597), so it does NOT take the NY_PROBE_* precondition.
            "col-scaling" => probe_col_scaling(),
            "serial-parallel-oracle" => {
                require_probe_instance()?;
                oracle_serial_vs_parallel();
            }
            "inc2a-root-parity-1498" => inc2a_root_parity_1498(),
            "inc2a-enclosure-1498" => inc2a_outward_bound_and_enclosure_1498(),
            "inc2b-trajectory-1498" => inc2b_trajectory_replay_1498(),
            "inc2b-l5-parity" => inc2b_unstable_position_parity_l5(),
            "inc2c-close-4429" => inc2c_closes_4429(),
            "inc2c-close-1498" => inc2c_closes_1498(),
            "inc2c-close-2551" => inc2c_closes_2551(),
            "inc2c-moat-6232" => inc2c_moat_sat_6232_not_verified(),
            "extractor-smoke" => extractor_and_parser_smoke(),
            "cifar-moat" => moat_gt_sat_never_unsat(),
            "cifar-closure" => closure_gt_unsat_contingency(),
            "extraction-safety" => extraction_safety_other_models(),
            "epoch-gain" => {
                require_probe_instance()?;
                epoch_gain_probe_real();
            }
            "metaroom-gains" => metaroom_banked_gains_still_close(),
            "metaroom-moat" => metaroom_moat_gt_sat_never_unsat(),
            "sweep" => {
                let targets = std::env::var("NY_SWEEP_TARGETS")
                    .map_err(|_| anyhow::anyhow!("sweep requires NY_SWEEP_TARGETS"))?;
                anyhow::ensure!(
                    Path::new(&targets).is_file(),
                    "missing sweep target list {targets}"
                );
                sweep_targets();
            }
            _ => anyhow::bail!(
                "unknown margin-row probe {probe:?}; expected instance, root-build, \
                 pass-scaling, col-scaling, serial-parallel-oracle, inc2a-root-parity-1498, \
                 inc2a-enclosure-1498, inc2b-trajectory-1498, inc2b-l5-parity, \
                 inc2c-close-4429, inc2c-close-1498, inc2c-close-2551, \
                 inc2c-moat-6232, extractor-smoke, cifar-moat, cifar-closure, \
                 extraction-safety, epoch-gain, metaroom-gains, metaroom-moat, or sweep"
            ),
        }
        Ok(())
    }
}
