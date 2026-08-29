// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lever-reachability — the TYPED PRESET ROUTE for compatibility-cell-backed
//! verdict-affecting gates. Receipt-owned full-policy families are frozen
//! directly from the run's [`ny_levers::LeverSet`] and bypass this bridge.
//!
//! # The defect
//!
//! `vnncomp_scripts/run_instance.sh` — the SCORED entry point — exports no
//! `NY_*` variables (only `OMP_NUM_THREADS`). Every `NY_*` gate therefore reads
//! "unset" on a scored run unless a caller explicitly supplies an override, so
//! its mechanism is dead code in competition however well it measured by hand.
//! Measured from the registry on 2026-08-23: 62 of the 75 axes that
//! `Class::VerdictAffecting` says can change an answer had no typed preset key.
//! `crates/ny-levers/tests/reachability_ratchet.rs` is the forcing function.
//!
//! That is not a bug in any single lever. It is a missing step in the workflow
//! — build behind an env gate, A/B it, bank it, **promote it**, ship — and the
//! fourth step has no deadline, so it is skipped and the mechanism is
//! remembered as "shipped".
//!
//! # What is here
//!
//! * the cells for the ny-cli-side gates (the attack / falsification slice, the
//!   SafeNLP cycle, the nn4sys context-group cover);
//! * [`apply`], the single bridge that loads the resolved preset ONCE and
//!   writes every compatibility-backed promoted cell — CLI-side and
//!   `ny-propagate`-side — before any lane can start.
//!
//! # The contract, identical for every gate
//!
//! * an UNWRITTEN cell resolves exactly as the pre-promotion
//!   `ny_levers::read(&DECL)` did, so a build that ships no preset key is
//!   byte-identical;
//! * an ADMISSIBLE environment value still WINS in BOTH directions, so every
//!   banked A/B recipe keeps working;
//! * a PRESENT but malformed environment value is a kill switch: it suppresses
//!   the preset and resolves to the declaration default;
//! * the precedence is [`ny_levers::gate`]'s, which is
//!   [`ny_levers::read_over_config`]'s. Nothing here re-implements it.
//!
//! PROMOTION IS NOT ARMING. No shipped yaml sets any of these keys, pinned by
//! `no_shipped_preset_arms_a_promoted_lever_delivery_key`.
//!
//! # Soundness
//!
//! Every gate delivered here is a ROUTE or a BUDGET choice between arms that
//! are each independently sound, so a wrong value costs proofs and cannot
//! manufacture one:
//!
//! * the falsification lanes (`attack.*`) return a CLAIMED counterexample input
//!   and never a verdict — a claim becomes `sat` only by passing the unchanged
//!   `gate_sat_with_trusted_oracle`, and none of them can produce `unsat`;
//! * the root phases and the batched clip are shrink-only and fail closed;
//! * `bab.nn4sys_ctx_group` changes WHICH boxes get bounded, under fail-closed
//!   cover/disjointness proofs — a doubtful cover keeps the obligation.
//!
//! `bab.safenlp_mip_cycle` is the one that needs its own sentence: it routes a
//! SafeNLP row to the MIP backend and to AY's margin reframe, i.e. it changes
//! WHICH prover answers. Both provers are verdict authorities in their own
//! right and the reframe's UNSAT is banked on a strict bound crossing with
//! original-model proof replay, so arming it can lose a row to the schedule but
//! cannot certify a false one.

use crate::preset::PresetConfig;
use ny_levers::gate::{resolve_bool, resolve_text, resolve_u64, BoolCell, TextCell, U64Cell};

/// Declare the cell + setter + resolver triple for one boolean lever.
///
/// Mechanical on purpose: the JUSTIFICATION for each promotion lives on the
/// typed preset field (`crates/ny-cli/src/preset/mod.rs`) and on the `space.rs`
/// axis, the two places a reviewer reads. A third copy would drift.
macro_rules! bool_gate {
    ($cell:ident, $setter:ident, $reader:ident, $decl:path) => {
        static $cell: BoolCell = BoolCell::new();

        /// Record the typed preset's answer. Called once, from `apply`.
        pub(crate) fn $setter(on: bool) {
            $cell.set(on);
        }

        /// Resolve the lever: environment first (both directions), then the
        /// typed preset, then the declaration default.
        pub(crate) fn $reader() -> bool {
            resolve_bool(&$decl, &$cell)
        }
    };
}

/// Integer form of `bool_gate`.
macro_rules! u64_gate {
    ($cell:ident, $setter:ident, $reader:ident, $decl:path) => {
        static $cell: U64Cell = U64Cell::new();

        /// Record the typed preset's answer. Called once, from `apply`.
        pub(crate) fn $setter(value: u64) {
            $cell.set(value);
        }

        /// Resolve the lever; `None` keeps the reader's contextual default.
        pub(crate) fn $reader() -> Option<u64> {
            resolve_u64(&$decl, &$cell)
        }
    };
}

// --- attack / falsification slice ------------------------------------------

bool_gate!(
    FALSIFY_PORTFOLIO,
    set_falsify_portfolio_preset,
    falsify_portfolio_armed,
    ny_levers::decls::dark_probes::FALSIFY_PORTFOLIO_LANE
);

u64_gate!(
    FALSIFY_PORTFOLIO_SECONDS,
    set_falsify_portfolio_secs_preset,
    falsify_portfolio_secs_override,
    ny_levers::decls::dark_probes::FALSIFY_PORTFOLIO_SECONDS
);

bool_gate!(
    ATTACK_PRE_SOFTMAX_OBJECTIVE,
    set_attack_pre_softmax_objective_preset,
    attack_pre_softmax_objective_armed,
    ny_levers::decls::dark_probes::ATTACK_PRE_SOFTMAX_OBJECTIVE
);

bool_gate!(
    LANE_BUDGET_FRAME,
    set_lane_budget_frame_preset,
    lane_budget_frame_armed,
    ny_levers::decls::dark_probes::LANE_BUDGET_FRAME
);

bool_gate!(
    LANE_BUDGET_ALLOCATOR,
    set_lane_budget_allocator_preset,
    lane_budget_allocator_armed,
    ny_levers::decls::dark_probes::LANE_BUDGET_ALLOCATOR
);

bool_gate!(
    LANE_VALUE_SCHEDULER,
    set_lane_value_scheduler_preset,
    lane_value_scheduler_armed,
    ny_levers::decls::dark_probes::LANE_VALUE_SCHEDULER
);

// --- category-scoped route selections --------------------------------------

bool_gate!(
    CPU_COMPUTE_ENGINE,
    set_cpu_compute_engine_preset,
    cpu_compute_engine_enabled,
    ny_levers::decls::collection::CPU_COMPUTE_ENGINE
);

bool_gate!(
    SAFENLP_MIP_CYCLE,
    set_safenlp_mip_cycle_preset,
    safenlp_mip_cycle_armed,
    ny_levers::decls::safenlp_cycle::SAFENLP_MIP_CYCLE
);

bool_gate!(
    NN4SYS_CTX_GROUP,
    set_nn4sys_ctx_group_preset,
    nn4sys_ctx_group_armed,
    ny_levers::decls::nn4sys_ctx_group::NN4SYS_CTX_GROUP
);

static BNN_SIGN_SPACE_MINIMAL_MOVE: BoolCell = BoolCell::new();

pub(crate) fn set_bnn_sign_space_minimal_move_preset(on: bool) {
    BNN_SIGN_SPACE_MINIMAL_MOVE.set(on);
}

/// Compatibility resolver for unit tests and non-scored helper callers.
/// Scored VNN-COMP execution consumes its entry-time frozen policy directly.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn bnn_sign_space_minimal_move_enabled() -> bool {
    resolve_bool(
        &ny_levers::decls::dark_probes::BNN_SIGN_SPACE_MINIMAL_MOVE,
        &BNN_SIGN_SPACE_MINIMAL_MOVE,
    )
}

static BNN_SIGN_SPACE_TRUST_REGION: TextCell = TextCell::new();

pub(crate) fn set_bnn_sign_space_trust_region_preset(value: impl Into<String>) {
    BNN_SIGN_SPACE_TRUST_REGION.set(value);
}

/// Compatibility resolver for unit tests and non-scored helper callers.
/// Scored VNN-COMP execution consumes its entry-time frozen policy directly.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn bnn_sign_space_trust_region_arm() -> Option<String> {
    resolve_text(
        &ny_levers::decls::dark_probes::BNN_SIGN_SPACE_TRUST_REGION,
        &BNN_SIGN_SPACE_TRUST_REGION,
    )
}

/// Write every compatibility-backed promoted lever cell from one preset.
///
/// Called once from `commands::vnncomp::run_and_translate`, BEFORE the attack
/// slice, the root phases and the margin-row lane — several of the readers
/// latch their answer in a `OnceLock`, so a later write would be invisible.
///
/// One preset LOAD for the whole set, deliberately: the older per-key readers
/// each re-read and re-parse the yaml from disk, which is both wasteful and a
/// TOCTOU seam. Absent keys leave their cells empty, which is the shipped,
/// byte-identical resolution.
pub(crate) fn apply(config: &PresetConfig) {
    // --- CLI-side: the attack / falsification slice ---
    if let Some(on) = config.attack.falsify_portfolio {
        set_falsify_portfolio_preset(on);
    }
    if let Some(secs) = config.attack.falsify_portfolio_secs {
        set_falsify_portfolio_secs_preset(secs);
    }
    if let Some(on) = config.attack.pre_softmax_objective {
        set_attack_pre_softmax_objective_preset(on);
    }
    if let Some(on) = config.attack.lane_budget_frame {
        set_lane_budget_frame_preset(on);
    }
    if let Some(on) = config.attack.lane_budget_allocator {
        set_lane_budget_allocator_preset(on);
    }
    if let Some(on) = config.attack.lane_value_scheduler {
        set_lane_value_scheduler_preset(on);
    }

    // --- CLI-side: category-scoped route selections ---
    if let Some(on) = config.bab.cpu_compute_engine {
        set_cpu_compute_engine_preset(on);
    }
    if let Some(on) = config.bab.safenlp_mip_cycle {
        set_safenlp_mip_cycle_preset(on);
    }
    if let Some(on) = config.bab.nn4sys_ctx_group {
        set_nn4sys_ctx_group_preset(on);
    }
    if let Some(on) = config.attack.bnn_sign_space_minimal_move {
        set_bnn_sign_space_minimal_move_preset(on);
    }
    if let Some(arm) = config.attack.bnn_sign_space_trust_region.as_deref() {
        set_bnn_sign_space_trust_region_preset(arm);
    }

    // --- ny-propagate: root-phase ownership lattice ---
    if let Some(on) = config.bab.root_phase_resident_crown {
        ny_propagate::preset_gates::set_root_phase_resident_crown_preset(on);
    }
    if let Some(on) = config.bab.root_wide_demanded_interm_crown {
        ny_propagate::preset_gates::set_root_wide_demanded_interm_crown_preset(on);
    }
    if let Some(on) = config.bab.root_cpu_parallel_interm_crown {
        ny_propagate::preset_gates::set_root_cpu_parallel_interm_crown_preset(on);
    }

    // --- ny-propagate: comprehensive-sweep sizing ---
    if let Some(rows) = config.bab.root_comprehensive_gpu_interm_rows {
        ny_propagate::preset_gates::set_root_comprehensive_gpu_interm_rows_preset(rows);
    }
    if let Some(secs) = config.bab.root_comprehensive_gpu_interm_secs {
        ny_propagate::preset_gates::set_root_comprehensive_gpu_interm_secs_preset(secs);
    }
    if let Some(on) = config.bab.root_objective_directed_rows {
        ny_propagate::preset_gates::set_root_objective_directed_rows_preset(on);
    }

    // --- ny-propagate: batched intermediate refinement ---
    if let Some(on) = config.bab.root_joint_interm_alpha {
        ny_propagate::preset_gates::set_root_joint_interm_alpha_preset(on);
    }
    if let Some(on) = config.bab.root_joint_interm_alpha_deadline_ascent {
        ny_propagate::preset_gates::set_root_joint_interm_alpha_deadline_ascent_preset(on);
    }
    if let Some(secs) = config.bab.root_joint_interm_alpha_secs {
        ny_propagate::preset_gates::set_root_joint_interm_alpha_secs_preset(secs);
    }
    if let Some(iters) = config.bab.root_joint_interm_alpha_iters {
        ny_propagate::preset_gates::set_root_joint_interm_alpha_iters_preset(iters);
    }
    if let Some(lr) = config.bab.root_joint_interm_alpha_lr {
        ny_propagate::preset_gates::set_root_joint_interm_alpha_lr_preset(lr);
    }
    if let Some(max_sel) = config.bab.root_joint_interm_alpha_max_sel {
        ny_propagate::preset_gates::set_root_joint_interm_alpha_max_sel_preset(max_sel);
    }
    if let Some(on) = config.bab.root_joint_interm_alpha_frozen_stop {
        ny_propagate::preset_gates::set_root_joint_interm_alpha_frozen_stop_preset(on);
    }
    if let Some(on) = config.bab.root_joint_interm_alpha_pure_chain {
        ny_propagate::preset_gates::set_root_joint_interm_alpha_pure_chain_preset(on);
    }
    if let Some(on) = config.bab.bn_gpu_extract {
        ny_propagate::preset_gates::set_bn_gpu_extract_preset(on);
    }
    if let Some(layers) = config.bab.root_joint_interm_alpha_layers.as_deref() {
        ny_propagate::preset_gates::set_root_joint_interm_alpha_layers_preset(layers);
    }
    if let Some(secs) = config.bab.root_joint_min_remaining_secs {
        ny_propagate::preset_gates::set_root_joint_min_remaining_secs_preset(secs);
    }
    if let Some(dim) = config.bab.root_joint_interm_alpha_max_dim {
        ny_propagate::preset_gates::set_root_joint_interm_alpha_max_dim_preset(dim);
    }
    if let Some(on) = config.bab.clip_host_mean_la {
        ny_propagate::preset_gates::set_clip_host_mean_la_preset(on);
    }
    if let Some(on) = config.bab.clip_interm_certified {
        ny_propagate::preset_gates::set_clip_interm_certified_preset(on);
    }

    // --- ny-propagate: wide multi-objective lane ---
    if let Some(on) = config.bab.mo_gpu_chunk_deadline {
        ny_propagate::preset_gates::set_mo_gpu_chunk_deadline_preset(on);
    }
    if let Some(on) = config.bab.mo_alpha_heuristic {
        ny_propagate::preset_gates::set_mo_alpha_heuristic_preset(on);
    }
    if let Some(on) = config.bab.mo_beta_spsa {
        ny_propagate::preset_gates::set_mo_beta_spsa_preset(on);
    }
    if let Some(on) = config.bab.mo_alpha_step {
        ny_propagate::preset_gates::set_mo_alpha_step_preset(on);
    }
    if let Some(share) = config.bab.kfsb_sim_share {
        ny_propagate::preset_gates::set_kfsb_sim_share_preset(share);
    }

    // --- ny-propagate: complete margin-row search delivery ---
    if let Some(cap) = config.margin_row.lru {
        ny_propagate::preset_gates::set_margin_row_lru_preset(cap);
    }
    if let Some(token) = config.margin_row.parallel.as_deref() {
        ny_propagate::preset_gates::set_margin_row_parallel_preset(token);
    }
    if let Some(on) = config.margin_row.gpu {
        ny_propagate::preset_gates::set_margin_row_gpu_preset(on);
    }
    if let Some(on) = config.margin_row.gpu_batch {
        ny_propagate::preset_gates::set_margin_row_gpu_batch_preset(on);
    }
    if let Some(on) = config.margin_row.gpu_backward {
        ny_propagate::preset_gates::set_margin_row_gpu_backward_preset(on);
    }
    if let Some(on) = config.margin_row.alpha_opt {
        ny_propagate::preset_gates::set_margin_row_alpha_opt_preset(on);
    }
    if let Some(iters) = config.margin_row.alpha_iters {
        ny_propagate::preset_gates::set_margin_row_alpha_iters_preset(iters);
    }
    if let Some(secs) = config.margin_row.alpha_secs {
        ny_propagate::preset_gates::set_margin_row_alpha_secs_preset(secs);
    }
    if let Some(lr) = config.margin_row.alpha_lr {
        ny_propagate::preset_gates::set_margin_row_alpha_lr_preset(lr);
    }
    if let Some(cols) = config.margin_row.alpha_cols {
        ny_propagate::preset_gates::set_margin_row_alpha_cols_preset(cols);
    }
    if let Some(on) = config.margin_row.alpha_guard {
        ny_propagate::preset_gates::set_margin_row_alpha_guard_preset(on);
    }
    if let Some(on) = config.margin_row.clip {
        ny_propagate::preset_gates::set_margin_row_clip_preset(on);
    }
    if let Some(on) = config.margin_row.clip_interm {
        ny_propagate::preset_gates::set_margin_row_clip_interm_preset(on);
    }
    if let Some(on) = config.margin_row.clip_rows {
        ny_propagate::preset_gates::set_margin_row_clip_rows_preset(on);
    }
    if let Some(topk) = config.margin_row.clip_topk {
        ny_propagate::preset_gates::set_margin_row_clip_topk_preset(topk);
    }
    if let Some(thresh) = config.margin_row.closure_thresh {
        ny_propagate::preset_gates::set_margin_row_closure_thresh_preset(thresh);
    }
    if let Some(warmup) = config.margin_row.closure_warmup {
        ny_propagate::preset_gates::set_margin_row_closure_warmup_preset(warmup);
    }
    if let Some(window) = config.margin_row.closure_window {
        ny_propagate::preset_gates::set_margin_row_closure_window_preset(window);
    }
    if let Some(max_tier) = config.margin_row.closure_max_tier {
        ny_propagate::preset_gates::set_margin_row_closure_max_tier_preset(max_tier);
    }
    if let Some(chunk) = config.margin_row.bi_chunk {
        ny_propagate::preset_gates::set_margin_row_bi_chunk_preset(chunk);
    }
    if let Some(secs) = config.margin_row.bi_chunk_secs {
        ny_propagate::preset_gates::set_margin_row_bi_chunk_secs_preset(secs);
    }
    if let Some(topk) = config.margin_row.bi_topk {
        ny_propagate::preset_gates::set_margin_row_bi_topk_preset(topk);
    }
    if let Some(on) = config.margin_row.beta {
        ny_propagate::preset_gates::set_margin_row_beta_preset(on);
    }
    if let Some(eta) = config.margin_row.beta_eta {
        ny_propagate::preset_gates::set_margin_row_beta_eta_preset(eta);
    }
    if let Some(iters) = config.margin_row.beta_iters {
        ny_propagate::preset_gates::set_margin_row_beta_iters_preset(iters);
    }
    if let Some(lambda) = config.margin_row.beta_lambda {
        ny_propagate::preset_gates::set_margin_row_beta_lambda_preset(lambda);
    }
    if let Some(on) = config.margin_row.beta_polyak {
        ny_propagate::preset_gates::set_margin_row_beta_polyak_preset(on);
    }
    if let Some(on) = config.margin_row.beta_heads {
        ny_propagate::preset_gates::set_margin_row_beta_heads_preset(on);
    }
    if let Some(on) = config.margin_row.rebake {
        ny_propagate::preset_gates::set_margin_row_rebake_preset(on);
    }
    if let Some(rounds) = config.margin_row.rebake_rounds {
        ny_propagate::preset_gates::set_margin_row_rebake_rounds_preset(rounds);
    }
    if let Some(secs) = config.margin_row.rebake_secs {
        ny_propagate::preset_gates::set_margin_row_rebake_secs_preset(secs);
    }
    // #tableau-f32: f32 STORAGE for the root tableau walk. Sibling of the
    // grandfathered `margin_row.root_f32` route (vnncomp.rs), delivered here
    // through the governed cell so env `1`/`0` wins in both directions and a
    // malformed env value kills the preset rather than riding it.
    if let Some(on) = config.margin_row.tableau_f32 {
        ny_propagate::preset_gates::set_margin_row_tableau_f32_preset(on);
    }
    // Compatibility delivery for callers that still construct a margin-row
    // policy from the process-layer preset cells. Scored VNN-COMP lanes consume
    // these same keys from their receipt-owning frozen LeverSet instead.
    if let Some(width) = config.margin_row.epoch_tier0 {
        ny_propagate::preset_gates::set_epoch_tier0_preset(width);
    }
    if let Some(universe) = config.margin_row.epoch_universe {
        ny_propagate::preset_gates::set_epoch_universe_preset(universe);
    }
    if let Some(rows) = config.margin_row.epoch_retain {
        ny_propagate::preset_gates::set_epoch_retain_preset(rows);
    }

    // --- ny-propagate: remaining competition-searchable controls ---
    if let Some(token) = config.bab.conv_ibp_blocked.as_deref() {
        ny_propagate::preset_gates::set_conv_ibp_blocked_preset(token);
    }
    if let Some(on) = config.bab.patches_finite_expiry {
        ny_propagate::preset_gates::set_patches_finite_expiry_preset(on);
    }
    if let Some(on) = config.bab.no_walk_record_admission {
        ny_propagate::preset_gates::set_no_walk_record_admission_preset(on);
    }
    if let Some(mib) = config.bab.cgan_stacked_budget_mb {
        ny_propagate::preset_gates::set_cgan_stacked_budget_mb_preset(mib);
    }
    if let Some(on) = config.bab.pollable_faer {
        ny_propagate::preset_gates::set_pollable_faer_preset(on);
    }

    // --- ny-propagate: MIP leaf oracle + true-gradient replay ---
    if let Some(on) = config.bab.graph_mip_leaf_sat {
        ny_propagate::preset_gates::set_graph_mip_leaf_sat_preset(on);
    }
    if let Some(on) = config.bab.true_grad_gpu_replay {
        ny_propagate::preset_gates::set_true_grad_gpu_replay_preset(on);
    }

    // --- batch 2: the ny-gpu CROWN lane (cells live in ny-propagate, which
    // ny-gpu depends on) ---
    if let Some(on) = config.bab.bab_resnet_wide {
        ny_propagate::preset_gates::set_bab_resnet_wide_preset(on);
    }
    if let Some(on) = config.bab.bab_resnet_wide_subgroup {
        ny_propagate::preset_gates::set_bab_resnet_wide_subgroup_preset(on);
    }
    if let Some(rows) = config.bab.sweep_class_rows {
        ny_propagate::preset_gates::set_sweep_class_rows_preset(rows);
    }
    if let Some(mib) = config.bab.sweep_class_mib {
        ny_propagate::preset_gates::set_sweep_class_mib_preset(mib);
    }
    if let Some(on) = config.bab.resnet_err_concretize {
        ny_propagate::preset_gates::set_resnet_err_concretize_preset(on);
    }
    if let Some(on) = config.bab.resnet_err_concretize_fine {
        ny_propagate::preset_gates::set_resnet_err_concretize_fine_preset(on);
    }

    // --- batch 2: the three `read_raw` predicate gates. Their arming RULES
    // stay at their read sites (each pure predicate IS the spec, and for the
    // opt-out one the declaration's parser is a DIFFERENT function); only the
    // cell beneath them is written here. ---
    if let Some(on) = config.bab.seq_weighted_budget {
        ny_propagate::preset_gates::set_seq_weighted_budget_preset(on);
    }
    if let Some(on) = config.bab.cgan_stacked_backward {
        ny_propagate::preset_gates::set_cgan_stacked_backward_preset(on);
    }
    if let Some(on) = config.bab.cgan_polish {
        ny_propagate::preset_gates::set_cgan_polish_preset(on);
    }
    if let Some(on) = config.bab.crown_partial_finite_expiry {
        ny_propagate::preset_gates::set_crown_partial_finite_expiry_preset(on);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_compute_engine_layers_exact_env_over_the_typed_kill_switch() {
        ny_test_utils::env::with_env_edits(|env| {
            let cell = BoolCell::new();
            let resolved =
                || resolve_bool(&ny_levers::decls::collection::CPU_COMPUTE_ENGINE, &cell);
            env.remove("NY_CPU_COMPUTE_ENGINE");
            assert!(resolved(), "absence keeps the legacy arm on");

            cell.set(false);
            assert!(!resolved(), "typed false suppresses CPU");

            env.set("NY_CPU_COMPUTE_ENGINE", "1");
            assert!(resolved(), "exact env 1 overrides the preset");
            env.set("NY_CPU_COMPUTE_ENGINE", "0");
            assert!(!resolved(), "exact env 0 stays the kill switch");
            env.set("NY_CPU_COMPUTE_ENGINE", "true");
            assert!(
                resolved(),
                "malformed presence suppresses the preset and restores the declaration default"
            );
        });
    }

    /// The property the promotion commit rests on: an EMPTY preset writes no
    /// cell, so every resolver still answers what its declaration says.
    ///
    /// Held under the shared test env lock (an empty edit list only takes the
    /// lock): sibling tests legitimately set these very names for their own
    /// arms, and a resolver read racing one would be flaky.
    #[test]
    fn an_empty_preset_leaves_every_promoted_lever_at_its_declaration_default() {
        ny_test_utils::env::with_serialized_env_vars(&[], || {
            let empty = PresetConfig::default();
            apply(&empty);

            for (decl, resolved) in [
                (
                    &ny_levers::decls::dark_probes::FALSIFY_PORTFOLIO_LANE,
                    falsify_portfolio_armed(),
                ),
                (
                    &ny_levers::decls::dark_probes::ATTACK_PRE_SOFTMAX_OBJECTIVE,
                    attack_pre_softmax_objective_armed(),
                ),
                (
                    &ny_levers::decls::dark_probes::LANE_BUDGET_FRAME,
                    lane_budget_frame_armed(),
                ),
                (
                    &ny_levers::decls::dark_probes::LANE_BUDGET_ALLOCATOR,
                    lane_budget_allocator_armed(),
                ),
                (
                    &ny_levers::decls::dark_probes::LANE_VALUE_SCHEDULER,
                    lane_value_scheduler_armed(),
                ),
                (
                    &ny_levers::decls::safenlp_cycle::SAFENLP_MIP_CYCLE,
                    safenlp_mip_cycle_armed(),
                ),
                (
                    &ny_levers::decls::nn4sys_ctx_group::NN4SYS_CTX_GROUP,
                    nn4sys_ctx_group_armed(),
                ),
                (
                    &ny_levers::decls::collection::CPU_COMPUTE_ENGINE,
                    cpu_compute_engine_enabled(),
                ),
                (
                    &ny_levers::decls::dark_probes::BNN_SIGN_SPACE_MINIMAL_MOVE,
                    bnn_sign_space_minimal_move_enabled(),
                ),
            ] {
                assert_eq!(
                    resolved,
                    ny_levers::read(decl).value.as_bool(),
                    "{} must resolve to its declaration default with no preset key",
                    decl.name
                );
            }

            assert_eq!(
                falsify_portfolio_secs_override(),
                ny_levers::read(&ny_levers::decls::dark_probes::FALSIFY_PORTFOLIO_SECONDS)
                    .value
                    .as_u64()
            );
            assert_eq!(
                bnn_sign_space_trust_region_arm().as_deref(),
                ny_levers::read(&ny_levers::decls::dark_probes::BNN_SIGN_SPACE_TRUST_REGION)
                    .value
                    .as_str()
            );
        });
    }
}
