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
//!   SafeNLP cycle, the nn4sys context-group cover, the nn4sys one-axis
//!   complete-cover prepass);
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
//! PROMOTION IS NOT ARMING. Shipped policy choices are restricted to the exact
//! category/key exceptions checked by
//! `no_shipped_preset_arms_a_promoted_lever_delivery_key` and its live-entry
//! ratchet; all other delivery keys retain their prior unarmed status.
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
//!   cover/disjointness proofs — a doubtful cover keeps the obligation;
//! * the three `bab.nn4sys_1d_cover*` keys gate an ALL-OR-NOTHING prepass that
//!   publishes only obligations it discharged itself through the unchanged
//!   certified f64 machinery: one undecided leaf, cap or deadline discards the
//!   whole pass and the unchanged screen decides every clause, and a `false`
//!   seed always means unproven.
//!
//! `bab.safenlp_mip_cycle` is the one that needs its own sentence: it routes a
//! SafeNLP row to the MIP backend and to AY's margin reframe, i.e. it changes
//! WHICH prover answers. Both provers are verdict authorities in their own
//! right and the reframe's UNSAT is banked on a strict bound crossing with
//! original-model proof replay, so arming it can lose a row to the schedule but
//! cannot certify a false one.

use crate::preset::PresetConfig;
use ny_levers::gate::{
    resolve_bool, resolve_f64, resolve_text, resolve_u64, BoolCell, F64Cell, TextCell, U64Cell,
};

/// `NY_NO_CELL_ENUM` — THE single reader for the cell-enumeration kill switch.
///
/// Deliberately NOT a `bool_gate!`: this lever has no typed preset key, so there
/// is no cell to layer over. It is an env-only debug kill switch and reads its
/// declaration directly, which is still a chokepoint — `ny_levers::read` looks
/// the name up DYNAMICALLY from the declaration, so the three byte-identical
/// direct-literal `var_os` reads of `NY_NO_CELL_ENUM` that it replaces leave the
/// raw-read ratchet entirely. (This sentence deliberately avoids spelling the
/// literal call form: the ratchet counts that substring anywhere in a `.rs`
/// file, comments included, so writing it here would re-add the debt it pays.)
///
/// Behaviour is unchanged: `LeverKind::Bool` with `DefaultSpec::Bool(false)`
/// answers exactly as `is_some_and(|v| v == "1")` did on every input — absent
/// and malformed fall to `false`, `"1"` arms, `"0"` disarms.
pub(crate) fn no_cell_enum() -> bool {
    ny_levers::read(&ny_levers::decls::no_cell_enum::NO_CELL_ENUM)
        .value
        .as_bool()
}

/// Declare the cell + setter + resolver triple for one boolean lever.
///
/// Mechanical on purpose: the JUSTIFICATION for each promotion lives on the
/// typed preset field (`crates/ny-cli/src/preset/mod.rs`) and on the `space.rs`
/// axis, the two places a reviewer reads. A third copy would drift.
macro_rules! bool_gate {
    // Attributes before the reader apply to the reader only (e.g. a
    // `#[cfg(feature = ..)]` when its only consumer is feature-gated).
    ($cell:ident, $setter:ident, $(#[$reader_meta:meta])* $reader:ident, $decl:path) => {
        static $cell: BoolCell = BoolCell::new();

        /// Record the typed preset's answer. Called once, from `apply`.
        pub(crate) fn $setter(on: bool) {
            $cell.set(on);
        }

        /// Resolve the lever: environment first (both directions), then the
        /// typed preset, then the declaration default.
        $(#[$reader_meta])*
        pub(crate) fn $reader() -> bool {
            resolve_bool(&$decl, &$cell)
        }
    };
}

/// Integer form of `bool_gate`.
macro_rules! u64_gate {
    // Attributes before the reader apply to the reader only (`bool_gate`).
    ($cell:ident, $setter:ident, $(#[$reader_meta:meta])* $reader:ident, $decl:path) => {
        static $cell: U64Cell = U64Cell::new();

        /// Record the typed preset's answer. Called once, from `apply`.
        pub(crate) fn $setter(value: u64) {
            $cell.set(value);
        }

        /// Resolve the lever; `None` keeps the reader's contextual default.
        $(#[$reader_meta])*
        pub(crate) fn $reader() -> Option<u64> {
            resolve_u64(&$decl, &$cell)
        }
    };
}

/// Floating-point form of `bool_gate`.
macro_rules! f64_gate {
    ($cell:ident, $setter:ident, $reader:ident, $decl:path) => {
        static $cell: F64Cell = F64Cell::new();

        /// Record the typed preset's answer. Called once, from `apply`.
        pub(crate) fn $setter(value: f64) {
            $cell.set(value);
        }

        /// Resolve the lever; `None` keeps the reader's contextual default.
        pub(crate) fn $reader() -> Option<f64> {
            resolve_f64(&$decl, &$cell)
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

u64_gate!(
    FLAT_SURROGATE_MAX_BLOCKS,
    set_flat_surrogate_max_blocks_preset,
    flat_surrogate_max_blocks_override,
    ny_levers::decls::dark_probes::FLAT_SURROGATE_MAX_BLOCKS
);

bool_gate!(
    FLAT_SURROGATE_TRANSFER,
    set_flat_surrogate_transfer_preset,
    flat_surrogate_transfer_armed,
    ny_levers::decls::dark_probes::FLAT_SURROGATE_TRANSFER
);

// #graph-free-candidates: a DIAGNOSTIC-ONLY instrument, not a route or budget
// choice. It rides this bridge only because a zero-NY_* diagnostic run needs a
// yaml route (run_instance.sh exports none). Armed, the graph-free lane writes
// the candidates its UNCHANGED gates judged to a side file; it cannot change a
// candidate, a gate or a verdict.
bool_gate!(
    GRAPH_FREE_CANDIDATE_SIDEFILE,
    set_graph_free_candidate_sidefile_preset,
    graph_free_candidate_sidefile_armed,
    ny_levers::decls::dark_probes::GRAPH_FREE_CANDIDATE_SIDEFILE
);

/// Which configuration layer decided the #graph-free-candidates side file.
///
/// Printed on the armed line and in the lane receipt so a live run with the
/// armed yaml and no `NY_*` set can prove the YAML route delivered it (an
/// env-armed run proves nothing about the preset bridge). Same token mapping
/// as [`nn4sys_1d_cover_source`].
pub(crate) fn graph_free_candidate_sidefile_source() -> &'static str {
    let decl = &ny_levers::decls::dark_probes::GRAPH_FREE_CANDIDATE_SIDEFILE;
    let cell = GRAPH_FREE_CANDIDATE_SIDEFILE
        .get()
        .map(ny_levers::LeverValue::Bool);
    match ny_levers::read_over_config(decl, cell) {
        Ok(resolved) => match resolved.source {
            ny_levers::Source::Default => "default",
            ny_levers::Source::Config => "preset",
            ny_levers::Source::LegacyEnv => "env",
            ny_levers::Source::LegacyEnvRejected => "env_rejected",
        },
        Err(_) => "refused",
    }
}

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

// --- MIP schedule governors (#metaroom-mip-governor) -----------------------
//
// Both lanes below can consume a large share of an instance budget and decide
// NOTHING, and both were reachable only through a raw `NY_*` read the scored
// harness does not export. The cells make them deliverable; no shipped yaml
// sets either key, so the shipped resolution is unchanged.

u64_gate!(
    MIP_LEAF_UNDECIDED_STREAK_CAP,
    set_mip_leaf_undecided_streak_cap_preset,
    // Read only by the MIP leaf lane (graph_mip_leaf.rs, feature `mip`).
    #[cfg(feature = "mip")]
    mip_leaf_undecided_streak_cap,
    ny_levers::decls::graph_mip::MIP_LEAF_UNDECIDED_STREAK_CAP
);

bool_gate!(
    MIP_ESCALATE_SKIP_CERTIFIED,
    set_mip_escalate_skip_certified_preset,
    // Read only by the HiGHS escalation (mip_highs.rs, feature `mip`).
    #[cfg(feature = "mip")]
    mip_escalate_skip_certified_armed,
    ny_levers::decls::graph_mip::MIP_ESCALATE_SKIP_CERTIFIED
);

// --- #nn4sys-1d-cover: the exact one-axis complete-cover prepass -----------
//
// Three gates, one mechanism. The master decides whether the prepass runs at
// all; the depth decides whether an armed prepass can finish (its pass is
// all-or-nothing, so an unreachable depth publishes nothing); the slice
// decides how much of the screen's own budget it may take before giving up.

bool_gate!(
    NN4SYS_1D_COVER,
    set_nn4sys_1d_cover_preset,
    nn4sys_1d_cover_armed,
    ny_levers::decls::nn4sys_1d_cover::NN4SYS_1D_COMPLETE_COVER
);

u64_gate!(
    NN4SYS_1D_COVER_DEPTH,
    set_nn4sys_1d_cover_depth_preset,
    nn4sys_1d_cover_depth_override,
    ny_levers::decls::nn4sys_1d_cover::NN4SYS_1D_COVER_DEPTH
);

f64_gate!(
    NN4SYS_1D_COVER_SLICE,
    set_nn4sys_1d_cover_slice_preset,
    nn4sys_1d_cover_slice_override,
    ny_levers::decls::nn4sys_1d_cover::NN4SYS_1D_COVER_SLICE
);

/// Which configuration layer decided the #nn4sys-1d-cover master.
///
/// A value alone is NOT a delivery proof: environment and preset converge on
/// the same `BoolCell`, so an `--lever`-armed gate proves nothing about the
/// yaml route, and the bridge that writes the cell is the thing that was once
/// silently clobbered in a merge. The screen's summary line carries this token
/// so a live run with the armed yaml and no `NY_*` set at all can assert the
/// route it actually took.
///
/// `Source::Config` is reported as `preset` because that is what the shipped
/// configuration layer is called everywhere else in this repo.
pub(crate) fn nn4sys_1d_cover_source() -> &'static str {
    let decl = &ny_levers::decls::nn4sys_1d_cover::NN4SYS_1D_COMPLETE_COVER;
    let cell = NN4SYS_1D_COVER.get().map(ny_levers::LeverValue::Bool);
    match ny_levers::read_over_config(decl, cell) {
        Ok(resolved) => match resolved.source {
            ny_levers::Source::Default => "default",
            ny_levers::Source::Config => "preset",
            ny_levers::Source::LegacyEnv => "env",
            ny_levers::Source::LegacyEnvRejected => "env_rejected",
        },
        Err(_) => "refused",
    }
}

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
    if let Some(cap) = config.attack.flat_surrogate_max_blocks {
        set_flat_surrogate_max_blocks_preset(cap);
    }
    if let Some(on) = config.attack.flat_surrogate_transfer {
        set_flat_surrogate_transfer_preset(on);
    }
    if let Some(on) = config.attack.graph_free_candidate_sidefile {
        set_graph_free_candidate_sidefile_preset(on);
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
    if let Some(cap) = config.bab.mip.leaf_undecided_streak_cap {
        set_mip_leaf_undecided_streak_cap_preset(cap as u64);
    }
    if let Some(on) = config.bab.mip.escalation_skips_certified_clauses {
        set_mip_escalate_skip_certified_preset(on);
    }
    if let Some(on) = config.bab.nn4sys_1d_cover {
        set_nn4sys_1d_cover_preset(on);
    }
    if let Some(depth) = config.bab.nn4sys_1d_cover_depth {
        set_nn4sys_1d_cover_depth_preset(depth);
    }
    if let Some(slice) = config.bab.nn4sys_1d_cover_slice {
        set_nn4sys_1d_cover_slice_preset(slice);
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

    // --- ny-propagate: cct engine-wiring pair (#patches-engine-refusal-
    // containment / #fast-f32-prewarm-all-hosts). The prewarm cell is READ
    // back by `commands::vnncomp`, which owns the actual materialization
    // call, immediately after this bridge runs.
    if let Some(on) = config.bab.patches_engine_refusal_containment {
        ny_propagate::preset_gates::set_patches_engine_refusal_containment_preset(on);
    }
    if let Some(on) = config.bab.fast_f32_prewarm {
        ny_propagate::preset_gates::set_fast_f32_prewarm_preset(on);
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
    if let Some(on) = config.margin_row.spec_alpha {
        ny_propagate::preset_gates::set_margin_row_spec_alpha_preset(on);
    }
    if let Some(width) = config.margin_row.spec_alpha_width {
        ny_propagate::preset_gates::set_margin_row_spec_alpha_width_preset(width);
    }
    if let Some(cands) = config.margin_row.spec_alpha_cands {
        ny_propagate::preset_gates::set_margin_row_spec_alpha_cands_preset(cands);
    }
    if let Some(rounds) = config.margin_row.spec_alpha_rounds {
        ny_propagate::preset_gates::set_margin_row_spec_alpha_rounds_preset(rounds);
    }
    if let Some(entries) = config.margin_row.spec_alpha_entries {
        ny_propagate::preset_gates::set_margin_row_spec_alpha_entries_preset(entries);
    }
    if let Some(on) = config.margin_row.clip {
        ny_propagate::preset_gates::set_margin_row_clip_preset(on);
    }
    if let Some(on) = config.margin_row.clip_head {
        ny_propagate::preset_gates::set_margin_row_clip_head_preset(on);
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
    if let Some(dose) = config.margin_row.interm_alpha {
        ny_propagate::preset_gates::set_margin_row_interm_alpha_preset(dose);
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
    // #bi-window / #joint-refresh: compatibility cells only; the scored lane
    // resolves these keys from its frozen LeverSet.
    if let Some(on) = config.margin_row.bi_window {
        ny_propagate::preset_gates::set_margin_row_bi_window_preset(on);
    }
    if let Some(rounds) = config.margin_row.joint_refresh {
        ny_propagate::preset_gates::set_margin_row_joint_refresh_preset(rounds);
    }
    if let Some(secs) = config.margin_row.joint_refresh_secs {
        ny_propagate::preset_gates::set_margin_row_joint_refresh_secs_preset(secs);
    }
    // Source-owned head groups: compatibility cell only. Scored runs resolve
    // this boolean from the same frozen LeverSet as their flight receipt.
    if let Some(on) = config.margin_row.head_groups {
        ny_propagate::preset_gates::set_margin_row_head_groups_preset(on);
    }
    if let Some(on) = config.margin_row.head_directional_cuts {
        ny_propagate::preset_gates::set_margin_row_head_directional_cuts_preset(on);
    }
    if let Some(max_heads) = config.margin_row.head_phase_cover {
        ny_propagate::preset_gates::set_margin_row_head_phase_cover_preset(max_heads);
    }
    // #root-certify-once: compatibility cells only; the scored lane resolves
    // these keys from its frozen LeverSet.
    if let Some(on) = config.margin_row.root_certify_once {
        ny_propagate::preset_gates::set_margin_row_root_certify_once_preset(on);
    }
    if let Some(iters) = config.margin_row.root_certify_once_iters {
        ny_propagate::preset_gates::set_margin_row_root_certify_once_iters_preset(iters);
    }
    if let Some(rounds) = config.margin_row.root_certify_once_refresh {
        ny_propagate::preset_gates::set_margin_row_root_certify_once_refresh_preset(rounds);
    }
    if let Some(secs) = config.margin_row.root_certify_once_secs {
        ny_propagate::preset_gates::set_margin_row_root_certify_once_secs_preset(secs);
    }
    // #bbe-joint-root: compatibility cell only; the scored lane resolves
    // `margin_row.root_search` from its frozen LeverSet.
    if let Some(search) = config.margin_row.root_search {
        ny_propagate::preset_gates::set_margin_row_root_search_preset(search.token());
    }
    // #root-box-source: compatibility cell only; the scored lane resolves
    // `margin_row.root_box_source` from its frozen LeverSet into the root's
    // precision policy.
    if let Some(source) = config.margin_row.root_box_source {
        ny_propagate::preset_gates::set_margin_row_root_box_source_preset(source.token());
    }
    // #bi-gpu: RETIRED (ENGINE_PROGRAM R2). No bridge write: preset
    // validation refuses `margin_row.bi_gpu`, so no admitted preset carries it.
    // #bi-patch: compatibility cells only, like #bi-window's; the scored lane
    // resolves the three keys from its frozen LeverSet.
    if let Some(on) = config.margin_row.bi_patch {
        ny_propagate::preset_gates::set_margin_row_bi_patch_preset(on);
    }
    if let Some(ratio) = config.margin_row.bi_patch_ratio {
        ny_propagate::preset_gates::set_margin_row_bi_patch_ratio_preset(ratio);
    }
    if let Some(mb) = config.margin_row.bi_patch_mb {
        ny_propagate::preset_gates::set_margin_row_bi_patch_mb_preset(mb);
    }
    // #mr-ibp-seed: compatibility cell only, like #bi-gpu's; the scored lane
    // resolves `margin_row.ibp_seed` from its frozen LeverSet into the root's
    // precision policy.
    if let Some(on) = config.margin_row.ibp_seed {
        ny_propagate::preset_gates::set_margin_row_ibp_seed_preset(on);
    }
    // #tableau-f32: f32 STORAGE for the root tableau walk. Sibling of the
    // grandfathered `margin_row.root_f32` route (vnncomp.rs), delivered here
    // through the governed cell so env `1`/`0` wins in both directions and a
    // malformed env value kills the preset rather than riding it.
    if let Some(on) = config.margin_row.tableau_f32 {
        ny_propagate::preset_gates::set_margin_row_tableau_f32_preset(on);
    }
    if let Some(on) = config.margin_row.domain_stack {
        ny_propagate::preset_gates::set_margin_row_domain_stack_preset(on);
    }
    // #mr-gpu-eval: RETIRED (ENGINE_PROGRAM R1 kill line + R2). No bridge
    // write: preset validation refuses `margin_row.gpu_conv`, and the lane never
    // arms the route (`margin_row::gpu_conv::armed` is false).
    // #frozen-dual-bab: latched once per process by
    // `margin_row::zono_state::armed` in the tree driver, which this write
    // precedes; the two knobs are read when the armed tier starts.
    if let Some(on) = config.margin_row.frozen_dual_bab {
        ny_propagate::preset_gates::set_margin_row_frozen_dual_bab_preset(on);
    }
    if let Some(cap) = config.margin_row.frozen_dual_bab_node_cap {
        ny_propagate::preset_gates::set_margin_row_frozen_dual_bab_node_cap_preset(cap);
    }
    if let Some(secs) = config.margin_row.frozen_dual_bab_secs {
        ny_propagate::preset_gates::set_margin_row_frozen_dual_bab_secs_preset(secs);
    }
    if let Some(steps) = config.margin_row.frozen_dual_refine_steps {
        ny_propagate::preset_gates::set_margin_row_frozen_dual_refine_steps_preset(steps);
    }
    // #fd-shared-state (ENGINE_PROGRAM M3): read with the two knobs above by
    // `zono_bab::TierPolicy::from_gates` when the armed tier starts.
    if let Some(mode) = config.margin_row.frozen_dual_bab_state {
        ny_propagate::preset_gates::set_margin_row_frozen_dual_bab_state_preset(mode.token());
    }
    // #wide-fd-search: the tier reads its search selector through this cell
    // (`zono_bab::TierPolicy::from_gates`), like its node cap and wall.
    if let Some(search) = config.margin_row.frozen_dual_bab_search {
        ny_propagate::preset_gates::set_margin_row_frozen_dual_bab_search_preset(search.token());
    }
    // #wave-beta-bab (ENGINE_PROGRAM M6): compatibility cells only, like
    // #root-box-source's; the scored lane resolves the tree engine and both
    // knobs from its frozen LeverSet (`MarginRowRuntimePolicy::wave`, the same
    // set the flight receipt serialises). Non-scored callers read these cells
    // through `bab::wave::WavePolicy::from_gates`.
    if let Some(mode) = config.margin_row.tree_mode {
        ny_propagate::preset_gates::set_margin_row_tree_mode_preset(mode.token());
    }
    if let Some(batch) = config.margin_row.tree_wave_batch {
        ny_propagate::preset_gates::set_margin_row_tree_wave_batch_preset(batch);
    }
    if let Some(threads) = config.margin_row.tree_wave_cert_threads {
        ny_propagate::preset_gates::set_margin_row_tree_wave_cert_threads_preset(threads);
    }
    // #mr-errlane: compatibility cell only; the scored lane resolves
    // `margin_row.err_envelope` from its frozen LeverSet.
    if let Some(lane) = config.margin_row.err_envelope {
        ny_propagate::preset_gates::set_margin_row_err_envelope_preset(lane.token());
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

    // --- ny-propagate: #dense-sparse-seed. The read site applies the value
    // only to Dense-seeded partial walks with a ReLU successor. ---
    if let Some(min_stable) = config.bab.crown_partial_dense_min_stable {
        ny_propagate::preset_gates::set_crown_partial_dense_min_stable_preset(min_stable);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_dual_refine_steps_reach_the_gate_through_apply() {
        let decl = &ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_REFINE_STEPS;
        ny_test_utils::env::with_env_edits(|env| {
            env.remove(decl.name);
            let armed: PresetConfig =
                serde_yaml::from_str("margin_row:\n  frozen_dual_refine_steps: 8\n")
                    .expect("the refinement key must be accepted by the scored preset schema");
            apply(&armed);
            let read = ny_propagate::preset_gates::margin_row_frozen_dual_refine_steps;
            assert_eq!(
                read(),
                8,
                "the yaml request must reach the tier's governed cell"
            );
            env.set(decl.name, " 16 ");
            assert_eq!(read(), 16, "an admissible environment request wins");
            env.set(decl.name, "0");
            assert_eq!(read(), 0, "zero disarms refinement over an armed preset");
            env.set(decl.name, "many");
            assert_eq!(read(), 0, "malformed presence suppresses the preset");
            env.remove(decl.name);
            ny_propagate::preset_gates::set_margin_row_frozen_dual_refine_steps_preset(0);
            assert_eq!(read(), 0, "restore the declaration's dark default");
        });
    }

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

    /// #flat-ladder-depth: the typed cap resolves preset-over-default, env
    /// wins in both directions, and a malformed env value is a kill switch
    /// back to the declaration default (0 = historical split-all schedule).
    #[test]
    fn flat_surrogate_max_blocks_layers_env_over_preset_over_default() {
        ny_test_utils::env::with_env_edits(|env| {
            let cell = U64Cell::new();
            let resolved = || {
                resolve_u64(
                    &ny_levers::decls::dark_probes::FLAT_SURROGATE_MAX_BLOCKS,
                    &cell,
                )
            };
            env.remove("NY_FLAT_SURROGATE_MAX_BLOCKS");
            assert_eq!(
                resolved(),
                Some(0),
                "absence resolves to the declaration default: cap disarmed"
            );
            cell.set(4096);
            assert_eq!(resolved(), Some(4096), "the typed preset arms the cap");
            env.set("NY_FLAT_SURROGATE_MAX_BLOCKS", "512");
            assert_eq!(resolved(), Some(512), "an admissible env value wins");
            env.set("NY_FLAT_SURROGATE_MAX_BLOCKS", "0");
            assert_eq!(resolved(), Some(0), "env 0 disarms over an armed preset");
            env.set("NY_FLAT_SURROGATE_MAX_BLOCKS", "many");
            assert_eq!(
                resolved(),
                Some(0),
                "malformed presence suppresses the preset and restores the default"
            );
        });
    }

    /// #flat-surrogate-transfer: the typed gate resolves preset-over-default,
    /// env wins in both directions, and a malformed env value is a kill switch
    /// back to the declaration default (false = transfer stage disarmed).
    #[test]
    fn flat_surrogate_transfer_layers_env_over_preset_over_default() {
        ny_test_utils::env::with_env_edits(|env| {
            let cell = BoolCell::new();
            let resolved = || {
                resolve_bool(
                    &ny_levers::decls::dark_probes::FLAT_SURROGATE_TRANSFER,
                    &cell,
                )
            };
            env.remove("NY_FLAT_SURROGATE_TRANSFER");
            assert!(
                !resolved(),
                "absence resolves to the declaration default: transfer disarmed"
            );
            cell.set(true);
            assert!(resolved(), "the typed preset arms the transfer stage");
            env.set("NY_FLAT_SURROGATE_TRANSFER", "0");
            assert!(!resolved(), "exact env 0 disarms over an armed preset");
            env.set("NY_FLAT_SURROGATE_TRANSFER", "1");
            assert!(resolved(), "exact env 1 arms");
            env.set("NY_FLAT_SURROGATE_TRANSFER", "true");
            assert!(
                !resolved(),
                "malformed presence suppresses the preset and restores the default"
            );
        });
    }

    /// #graph-free-candidates: the typed side-file gate resolves
    /// preset-over-default, env wins in both directions, and a malformed env
    /// value is a kill switch back to the declaration default (false = no
    /// recorder, the lane byte-identical). The name is taken from the
    /// declaration so this test adds no lever-name literal site.
    #[test]
    fn graph_free_candidate_sidefile_layers_env_over_preset_over_default() {
        let decl = &ny_levers::decls::dark_probes::GRAPH_FREE_CANDIDATE_SIDEFILE;
        ny_test_utils::env::with_env_edits(|env| {
            let cell = BoolCell::new();
            let resolved = || resolve_bool(decl, &cell);
            env.remove(decl.name);
            assert!(
                !resolved(),
                "absence resolves to the declaration default: side file dark"
            );
            cell.set(true);
            assert!(resolved(), "the typed preset arms the side file");
            env.set(decl.name, "0");
            assert!(!resolved(), "exact env 0 disarms over an armed preset");
            env.set(decl.name, "1");
            assert!(resolved(), "exact env 1 arms");
            env.set(decl.name, "true");
            assert!(
                !resolved(),
                "malformed presence suppresses the preset and restores the default"
            );
        });
    }

    /// #nn4sys-1d-cover: the three typed keys reach their gate cells through
    /// the REAL `apply` bridge, and the master's resolved SOURCE says so.
    ///
    /// A `Resolved::source` test on a local cell is necessary and not
    /// sufficient — environment and preset converge on the same cell, so an
    /// `--lever`-armed gate proves nothing about the yaml route, and this
    /// bridge is the thing that was once silently clobbered in a merge. This
    /// test exercises the bridge; the LIVE half (the canary yaml, zero `NY_*`,
    /// and the screen's own `cover_armed=1 cover_src=preset cover_depth=24`
    /// line) is what actually retires the obligation, and no unit test can
    /// stand in for it.
    ///
    /// The statics are process-global and `BoolCell::clear` is `#[cfg(test)]`
    /// of `ny-levers`, i.e. not compiled for this crate's harness. So the
    /// cells are RESTORED by writing each declaration's own default back into
    /// them: every value-based assertion in the sibling table test then reads
    /// exactly what an unwritten cell would give. The one assertion that
    /// cannot be restored is the SOURCE, which is why it is not in that table.
    #[test]
    fn nn4sys_1d_cover_keys_reach_their_gate_cells_through_apply() {
        ny_test_utils::env::with_env_edits(|env| {
            env.remove("NY_NN4SYS_1D_COMPLETE_COVER");
            env.remove("NY_NN4SYS_1D_COVER_DEPTH");
            env.remove("NY_NN4SYS_1D_COVER_SLICE");

            // A REAL yaml document, written as one: a backslash-continued string
            // literal folds its own indentation away and silently produces
            // `depth: 24 nn4sys_1d_cover_slice: 0.25` on one line, which then
            // fails as a type error rather than as the delivery failure this
            // test is for.
            let armed: PresetConfig = serde_yaml::from_str(
                r"
bab:
  nn4sys_1d_cover: true
  nn4sys_1d_cover_depth: 24
  nn4sys_1d_cover_slice: 0.25
",
            )
            .expect("the three #nn4sys-1d-cover keys must be spellable under deny_unknown_fields");
            assert_eq!(armed.bab.nn4sys_1d_cover, Some(true));
            assert_eq!(armed.bab.nn4sys_1d_cover_depth, Some(24));
            assert_eq!(armed.bab.nn4sys_1d_cover_slice, Some(0.25));

            apply(&armed);

            assert!(
                nn4sys_1d_cover_armed(),
                "the yaml master must reach the gate the cover reads"
            );
            assert_eq!(
                nn4sys_1d_cover_depth_override(),
                Some(24),
                "the depth must arrive as a VALUE, not merely as a key: the \
                 cover's pass is all-or-nothing, so a master delivered without \
                 its depth is a reachable inert mechanism"
            );
            assert_eq!(nn4sys_1d_cover_slice_override(), Some(0.25));
            assert_eq!(
                nn4sys_1d_cover_source(),
                "preset",
                "the master must resolve through the CONFIG layer, not the \
                 environment and not the declaration default"
            );

            // Restore: each declaration's own default, written explicitly.
            set_nn4sys_1d_cover_preset(
                ny_levers::read(&ny_levers::decls::nn4sys_1d_cover::NN4SYS_1D_COMPLETE_COVER)
                    .value
                    .as_bool(),
            );
            set_nn4sys_1d_cover_depth_preset(
                ny_levers::read(&ny_levers::decls::nn4sys_1d_cover::NN4SYS_1D_COVER_DEPTH)
                    .value
                    .as_u64()
                    .expect("the depth declaration ships a default"),
            );
            set_nn4sys_1d_cover_slice_preset(
                ny_levers::read(&ny_levers::decls::nn4sys_1d_cover::NN4SYS_1D_COVER_SLICE)
                    .value
                    .as_f64()
                    .expect("the slice declaration ships a default"),
            );
        });
    }

    /// #nn4sys-1d-cover: env wins over the typed preset in BOTH directions,
    /// and a malformed present value is a kill switch back to the declaration
    /// default. Local cells, so this cannot disturb the process-global ones.
    #[test]
    fn nn4sys_1d_cover_layers_env_over_preset_over_default() {
        ny_test_utils::env::with_env_edits(|env| {
            let cell = BoolCell::new();
            let resolved = || {
                resolve_bool(
                    &ny_levers::decls::nn4sys_1d_cover::NN4SYS_1D_COMPLETE_COVER,
                    &cell,
                )
            };
            env.remove("NY_NN4SYS_1D_COMPLETE_COVER");
            assert!(!resolved(), "absence resolves dark");
            cell.set(true);
            assert!(resolved(), "the typed preset arms the cover");
            env.set("NY_NN4SYS_1D_COMPLETE_COVER", "0");
            assert!(!resolved(), "exact env 0 disarms over an armed preset");
            env.set("NY_NN4SYS_1D_COMPLETE_COVER", "1");
            assert!(resolved(), "exact env 1 arms");
            env.set("NY_NN4SYS_1D_COMPLETE_COVER", "true");
            assert!(
                !resolved(),
                "malformed presence suppresses the preset and restores the default"
            );

            let depth = U64Cell::new();
            let depth_resolved = || {
                resolve_u64(
                    &ny_levers::decls::nn4sys_1d_cover::NN4SYS_1D_COVER_DEPTH,
                    &depth,
                )
            };
            env.remove("NY_NN4SYS_1D_COVER_DEPTH");
            assert_eq!(
                depth_resolved(),
                Some(8),
                "the declaration default must be the reader's own DEFAULT_DEPTH"
            );
            depth.set(24);
            assert_eq!(depth_resolved(), Some(24));
            env.set("NY_NN4SYS_1D_COVER_DEPTH", "12");
            assert_eq!(depth_resolved(), Some(12), "an admissible env value wins");

            let slice = F64Cell::new();
            let slice_resolved = || {
                resolve_f64(
                    &ny_levers::decls::nn4sys_1d_cover::NN4SYS_1D_COVER_SLICE,
                    &slice,
                )
            };
            env.remove("NY_NN4SYS_1D_COVER_SLICE");
            assert_eq!(slice_resolved(), Some(0.25));
            slice.set(0.5);
            assert_eq!(slice_resolved(), Some(0.5));
            env.set("NY_NN4SYS_1D_COVER_SLICE", "0.05");
            assert_eq!(slice_resolved(), Some(0.05));
            env.set("NY_NN4SYS_1D_COVER_SLICE", "0.9");
            assert_eq!(
                slice_resolved(),
                Some(0.25),
                "a value outside the declared [0.02, 0.5] interval is refused and \
                 falls back to the default rather than widening the slice"
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
                    &ny_levers::decls::dark_probes::FLAT_SURROGATE_TRANSFER,
                    flat_surrogate_transfer_armed(),
                ),
                (
                    &ny_levers::decls::dark_probes::GRAPH_FREE_CANDIDATE_SIDEFILE,
                    graph_free_candidate_sidefile_armed(),
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
                    &ny_levers::decls::nn4sys_1d_cover::NN4SYS_1D_COMPLETE_COVER,
                    nn4sys_1d_cover_armed(),
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
                flat_surrogate_max_blocks_override(),
                ny_levers::read(&ny_levers::decls::dark_probes::FLAT_SURROGATE_MAX_BLOCKS)
                    .value
                    .as_u64()
            );
            assert_eq!(
                nn4sys_1d_cover_depth_override(),
                ny_levers::read(&ny_levers::decls::nn4sys_1d_cover::NN4SYS_1D_COVER_DEPTH)
                    .value
                    .as_u64()
            );
            assert_eq!(
                nn4sys_1d_cover_slice_override(),
                ny_levers::read(&ny_levers::decls::nn4sys_1d_cover::NN4SYS_1D_COVER_SLICE)
                    .value
                    .as_f64()
            );
            assert_eq!(
                bnn_sign_space_trust_region_arm().as_deref(),
                ny_levers::read(&ny_levers::decls::dark_probes::BNN_SIGN_SPACE_TRUST_REGION)
                    .value
                    .as_str()
            );
        });
    }

    /// #dense-sparse-seed: `bab.crown_partial_dense_min_stable` reaches the
    /// ny-propagate cell the partial walk reads through the REAL `apply`
    /// bridge, as a VALUE (0.1 differs from the 0.9 default, so it cannot be
    /// the default answering), and the environment still layers over it in
    /// both directions. The LIVE half — the armed cct yaml, zero `NY_*`, and
    /// the walk's own `[dense-sparse-seed] ENGAGED rows=` line — is what
    /// retires the delivery obligation; no unit test stands in for it.
    ///
    /// The cell is process-global, so it is RESTORED by writing the
    /// declaration's own default back, exactly as the nn4sys bridge test does.
    #[test]
    fn crown_partial_dense_min_stable_reaches_its_gate_cell_through_apply() {
        let decl = &ny_levers::decls::crown_partial_seed::CROWN_PARTIAL_DENSE_MIN_STABLE;
        ny_test_utils::env::with_env_edits(|env| {
            env.remove(decl.name);

            let armed: PresetConfig = serde_yaml::from_str(
                r"
bab:
  crown_partial_dense_min_stable: 0.1
",
            )
            .expect(
                "bab.crown_partial_dense_min_stable must be spellable under deny_unknown_fields",
            );
            assert_eq!(armed.bab.crown_partial_dense_min_stable, Some(0.1));

            apply(&armed);
            assert_eq!(
                ny_propagate::preset_gates::crown_partial_dense_min_stable(),
                0.1,
                "the yaml threshold must reach the cell the partial walk reads"
            );

            env.set(decl.name, "0.9");
            assert_eq!(
                ny_propagate::preset_gates::crown_partial_dense_min_stable(),
                0.9,
                "an admissible environment value wins over the preset"
            );
            env.set(decl.name, "sparse");
            assert_eq!(
                ny_propagate::preset_gates::crown_partial_dense_min_stable(),
                0.9,
                "a malformed environment value is a kill switch back to the default"
            );
            env.remove(decl.name);

            // Restore: the declaration's own default, written explicitly.
            ny_propagate::preset_gates::set_crown_partial_dense_min_stable_preset(
                ny_levers::read(decl)
                    .value
                    .as_f64()
                    .expect("the declaration ships a default"),
            );
            assert_eq!(
                ny_propagate::preset_gates::crown_partial_dense_min_stable(),
                0.9
            );
        });
    }
}
