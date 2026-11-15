// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lever-reachability — the TYPED PRESET ROUTE for ny-propagate's
//! verdict-affecting gates.
//!
//! # Why every gate in here needed one
//!
//! `vnncomp_scripts/run_instance.sh` — the scored entry point — exports no
//! `NY_*` variables. Every environment-only gate therefore reads "unset" on a scored run, so a mechanism
//! that only an environment variable can arm is DEAD CODE in competition,
//! however well it measured by hand. Measured from the registry on 2026-08-23:
//! 62 of 75 axes that `Class::VerdictAffecting` says can change an answer had
//! no typed preset key at all. `ny-levers/tests/reachability_ratchet.rs` is the
//! forcing function; this module is one batch of the fix.
//!
//! # The contract, identical for every gate below
//!
//! * the CELL is empty on a build that ships no preset key, and an empty cell
//!   resolves EXACTLY as the pre-promotion `ny_levers::read(&DECL)` did — so a
//!   promotion commit is byte-identical on every shipped configuration;
//! * an ADMISSIBLE environment value still WINS, in BOTH directions: `"1"`
//!   arms and `"0"` disarms whatever a preset asked for, so every banked A/B
//!   recipe keeps working unchanged;
//! * a PRESENT but malformed environment value suppresses
//!   the preset and resolves to the declaration default, never to the preset;
//! * the layering is [`ny_levers::gate`]'s, which is
//!   [`ny_levers::read_over_config`]'s, which is the audited primitive the
//!   typed root-alpha seam already uses. No gate here re-implements precedence.
//!
//! Typed delivery alone does not change a default. The separately authorized
//! head-group and directional-cut defaults are true, including on malformed
//! legacy values; use exact `"0"` or typed `false` to disable them. Shipped
//! arming exceptions are pinned in the CLI preset delivery tests.
//!
//! # Ordering
//!
//! `crates/ny-cli/src/commands/vnncomp.rs` writes every cell once, from the
//! resolved preset, before any verification lane starts. Several readers here
//! latch their answer in a `OnceLock` (the historical hot-path shape), so a
//! write after the latch would be invisible — which is why the bridge block
//! sits ahead of the attack slice, the root phases and the margin-row lane
//! rather than next to any one of them.

use ny_levers::gate::{
    resolve_bool, resolve_f64, resolve_text, resolve_u64, BoolCell, F64Cell, TextCell, U64Cell,
};

/// Declare the cell + setter + resolver triple for one boolean lever.
///
/// Deliberately mechanical: the JUSTIFICATION for each promotion lives in the
/// typed preset field's doc comment (`crates/ny-cli/src/preset/mod.rs`) and in
/// the `space.rs` axis, which are the two places a reviewer reads. Repeating it
/// here would let the three copies drift.
macro_rules! bool_gate {
    ($cell:ident, $setter:ident, $reader:ident, $decl:path) => {
        static $cell: BoolCell = BoolCell::new();

        /// Record the typed preset's answer for this lever. Called once, from
        /// the CLI bridge, before any lane starts.
        pub fn $setter(on: bool) {
            $cell.set(on);
        }

        /// Resolve the lever: environment first (both directions), then the
        /// typed preset, then the declaration default.
        #[must_use]
        pub fn $reader() -> bool {
            resolve_bool(&$decl, &$cell)
        }
    };
}

/// Integer form of `bool_gate`. `None` means "no override": the reader keeps
/// whatever contextual default it already had.
macro_rules! u64_gate {
    ($cell:ident, $setter:ident, $reader:ident, $decl:path) => {
        static $cell: U64Cell = U64Cell::new();

        /// Record the typed preset's answer for this lever.
        pub fn $setter(value: u64) {
            $cell.set(value);
        }

        /// Resolve the lever; `None` keeps the reader's contextual default.
        #[must_use]
        pub fn $reader() -> Option<u64> {
            resolve_u64(&$decl, &$cell)
        }
    };
}

/// Integer form for declarations with a concrete shipped `u64` default.
macro_rules! u64_value_gate {
    ($cell:ident, $setter:ident, $reader:ident, $decl:path, $fallback:expr) => {
        static $cell: U64Cell = U64Cell::new();

        /// Record the typed preset's answer for this lever.
        pub fn $setter(value: u64) {
            $cell.set(value);
        }

        /// Resolve environment, typed preset, then the concrete declaration default.
        #[must_use]
        pub fn $reader() -> u64 {
            resolve_u64(&$decl, &$cell).unwrap_or($fallback)
        }
    };
}

/// Floating-point form of `bool_gate`.
macro_rules! f64_gate {
    ($cell:ident, $setter:ident, $reader:ident, $decl:path) => {
        static $cell: F64Cell = F64Cell::new();

        /// Record the typed preset's answer for this lever.
        pub fn $setter(value: f64) {
            $cell.set(value);
        }

        /// Resolve the lever; `None` keeps the reader's contextual default.
        #[must_use]
        pub fn $reader() -> Option<f64> {
            resolve_f64(&$decl, &$cell)
        }
    };
}

/// Integer form for declarations with a concrete shipped default.  The
/// subsystem keeps any algorithmic clamp at its historical read site; this
/// resolver owns only env > typed preset > declaration precedence.
macro_rules! usize_value_gate {
    ($cell:ident, $setter:ident, $reader:ident, $decl:path, $fallback:expr) => {
        static $cell: U64Cell = U64Cell::new();

        /// Record the typed preset's answer for this lever.
        pub fn $setter(value: usize) {
            $cell.set(u64::try_from(value).expect("usize must fit the governed u64 cell"));
        }

        /// Resolve the governed value before the consuming algorithm's clamp.
        #[must_use]
        pub fn $reader() -> usize {
            resolve_u64(&$decl, &$cell)
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or($fallback)
        }
    };
}

/// Floating-point form for declarations with a concrete shipped default.
macro_rules! f64_value_gate {
    ($cell:ident, $setter:ident, $reader:ident, $decl:path, $fallback:expr) => {
        static $cell: F64Cell = F64Cell::new();

        /// Record the typed preset's answer for this lever.
        pub fn $setter(value: f64) {
            $cell.set(value);
        }

        /// Resolve the governed value before any read-site policy filter.
        #[must_use]
        pub fn $reader() -> f64 {
            resolve_f64(&$decl, &$cell).unwrap_or($fallback)
        }
    };
}

/// Text/enum form for a typed preset selector.
macro_rules! text_gate {
    ($cell:ident, $setter:ident, $reader:ident, $decl:path) => {
        static $cell: TextCell = TextCell::new();

        /// Record the typed preset's answer for this lever.
        pub fn $setter(value: impl Into<String>) {
            $cell.set(value);
        }

        /// Resolve environment first, then typed preset, then declaration.
        #[must_use]
        pub fn $reader() -> Option<String> {
            resolve_text(&$decl, &$cell)
        }
    };
}

// --- root-phase ownership lattice (multi_objective/root.rs) ----------------
//
// These three OWN the root slot when armed, so they decide which intermediate
// tightening the whole instance gets. Every phase is shrink-only and fails
// closed, so a wrong value costs tightness and can lose a proof; none can
// manufacture one.

bool_gate!(
    ROOT_PHASE_RESIDENT_CROWN,
    set_root_phase_resident_crown_preset,
    root_phase_resident_crown_enabled,
    ny_levers::decls::collection::ROOT_PHASE_RESIDENT_CROWN
);

bool_gate!(
    ROOT_WIDE_DEMANDED_INTERM_CROWN,
    set_root_wide_demanded_interm_crown_preset,
    root_wide_demanded_interm_crown_enabled,
    ny_levers::decls::collection::ROOT_WIDE_DEMANDED_INTERM_CROWN
);

bool_gate!(
    ROOT_CPU_PARALLEL_INTERM_CROWN,
    set_root_cpu_parallel_interm_crown_preset,
    root_cpu_parallel_interm_crown_enabled,
    ny_levers::decls::collection::ROOT_CPU_PARALLEL_INTERM_CROWN
);

// --- comprehensive-sweep sizing (multi_objective/root.rs, root_phases.rs) ---

u64_gate!(
    ROOT_COMP_GPU_INTERM_ROWS,
    set_root_comprehensive_gpu_interm_rows_preset,
    root_comprehensive_gpu_interm_rows_override,
    ny_levers::decls::comprehensive_rows::ROOT_COMP_GPU_INTERM_ROWS
);

u64_gate!(
    ROOT_COMP_GPU_INTERM_SECS,
    set_root_comprehensive_gpu_interm_secs_preset,
    root_comprehensive_gpu_interm_secs_override,
    ny_levers::decls::comprehensive_rows::ROOT_COMP_GPU_INTERM_SECS
);

bool_gate!(
    ROOT_OBJECTIVE_DIRECTED_ROWS,
    set_root_objective_directed_rows_preset,
    root_objective_directed_rows_enabled,
    ny_levers::decls::comprehensive_rows::ROOT_OBJECTIVE_DIRECTED_ROWS
);

// --- batched intermediate refinement (propagation/batched/interm_refine.rs) -

bool_gate!(
    ROOT_JOINT_INTERM_ALPHA,
    set_root_joint_interm_alpha_preset,
    root_joint_interm_alpha_enabled,
    ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA
);

bool_gate!(
    ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT,
    set_root_joint_interm_alpha_deadline_ascent_preset,
    root_joint_interm_alpha_deadline_ascent_enabled,
    ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT
);

u64_value_gate!(
    ROOT_JOINT_INTERM_ALPHA_SECS,
    set_root_joint_interm_alpha_secs_preset,
    root_joint_interm_alpha_secs,
    ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_SECS,
    30
);

usize_value_gate!(
    ROOT_JOINT_INTERM_ALPHA_ITERS,
    set_root_joint_interm_alpha_iters_preset,
    root_joint_interm_alpha_iters,
    ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_ITERS,
    100
);

f64_value_gate!(
    ROOT_JOINT_INTERM_ALPHA_LR,
    set_root_joint_interm_alpha_lr_preset,
    root_joint_interm_alpha_lr,
    ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_LR,
    0.1
);

usize_value_gate!(
    ROOT_JOINT_INTERM_ALPHA_MAX_SEL,
    set_root_joint_interm_alpha_max_sel_preset,
    root_joint_interm_alpha_max_sel,
    ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_MAX_SEL,
    512
);

bool_gate!(
    ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP,
    set_root_joint_interm_alpha_frozen_stop_preset,
    root_joint_interm_alpha_frozen_stop_enabled,
    ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP
);

bool_gate!(
    ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN,
    set_root_joint_interm_alpha_pure_chain_preset,
    root_joint_interm_alpha_pure_chain_enabled,
    ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN
);

bool_gate!(
    BN_GPU_EXTRACT,
    set_bn_gpu_extract_preset,
    bn_gpu_extract_enabled,
    ny_levers::decls::collection::BN_GPU_EXTRACT
);

text_gate!(
    ROOT_JOINT_INTERM_ALPHA_LAYERS,
    set_root_joint_interm_alpha_layers_preset,
    root_joint_interm_alpha_layers,
    ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_LAYERS
);

/// Resolve the environment-only, print-only root-joint diagnostic arm.
///
/// It deliberately has no preset cell or competition axis: a logging switch is
/// governed for parser/receipt consistency, not promoted as a solver treatment.
#[must_use]
pub fn root_joint_interm_alpha_probe_enabled() -> bool {
    ny_levers::read(&ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_PROBE)
        .value
        .as_bool()
}

u64_value_gate!(
    ROOT_JOINT_MIN_REMAINING_SECS,
    set_root_joint_min_remaining_secs_preset,
    root_joint_min_remaining_secs,
    ny_levers::decls::collection::ROOT_JOINT_MIN_REMAINING_SECS,
    240
);

u64_gate!(
    ROOT_JOINT_INTERM_ALPHA_MAX_DIM,
    set_root_joint_interm_alpha_max_dim_preset,
    root_joint_interm_alpha_max_dim_override,
    ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_MAX_DIM
);

bool_gate!(
    CLIP_HOST_MEAN_LA,
    set_clip_host_mean_la_preset,
    clip_host_mean_la_enabled,
    ny_levers::decls::collection::CLIP_HOST_MEAN_LA
);

bool_gate!(
    CLIP_INTERM_CERTIFIED,
    set_clip_interm_certified_preset,
    clip_interm_certified_enabled,
    ny_levers::decls::collection::CLIP_INTERM_CERTIFIED
);

// --- wide multi-objective lane --------------------------------------------

bool_gate!(
    MO_GPU_CHUNK_DEADLINE,
    set_mo_gpu_chunk_deadline_preset,
    mo_gpu_chunk_deadline_enabled,
    ny_levers::decls::wide_lane::MO_GPU_CHUNK_DEADLINE
);

bool_gate!(
    MO_ALPHA_HEURISTIC,
    set_mo_alpha_heuristic_preset,
    mo_alpha_heuristic_enabled,
    ny_levers::decls::mo_alpha_heuristic::MO_ALPHA_HEURISTIC
);

bool_gate!(
    MO_BETA_SPSA,
    set_mo_beta_spsa_preset,
    mo_beta_spsa_enabled,
    ny_levers::decls::mo_optimizer::MO_BETA_SPSA
);

bool_gate!(
    MO_ALPHA_STEP,
    set_mo_alpha_step_preset,
    mo_alpha_step_enabled,
    ny_levers::decls::mo_optimizer::MO_ALPHA_STEP
);

f64_gate!(
    KFSB_SIM_SHARE,
    set_kfsb_sim_share_preset,
    kfsb_sim_share_override,
    ny_levers::decls::wide_lane::KFSB_SIM_SHARE
);

// --- complete margin-row search delivery ---------------------------------

bool_gate!(
    MARGIN_ROW_GPU,
    set_margin_row_gpu_preset,
    margin_row_gpu_enabled,
    ny_levers::decls::sound_channel::MARGIN_ROW_GPU
);

bool_gate!(
    MARGIN_ROW_GPU_BATCH,
    set_margin_row_gpu_batch_preset,
    margin_row_gpu_batch_enabled,
    ny_levers::decls::sound_channel::MARGIN_ROW_GPU_BATCH
);

bool_gate!(
    MARGIN_ROW_GPU_BACKWARD,
    set_margin_row_gpu_backward_preset,
    margin_row_gpu_backward_enabled,
    ny_levers::decls::margin_row::MARGIN_ROW_GPU_BACKWARD
);

bool_gate!(
    MARGIN_ROW_ALPHA_OPT,
    set_margin_row_alpha_opt_preset,
    margin_row_alpha_opt_enabled,
    ny_levers::decls::sound_channel::MARGIN_ROW_ALPHA_OPT
);

usize_value_gate!(
    MARGIN_ROW_ALPHA_ITERS,
    set_margin_row_alpha_iters_preset,
    margin_row_alpha_iters,
    ny_levers::decls::sound_channel::MARGIN_ROW_ALPHA_ITERS,
    20
);

f64_value_gate!(
    MARGIN_ROW_ALPHA_SECS,
    set_margin_row_alpha_secs_preset,
    margin_row_alpha_secs,
    ny_levers::decls::sound_channel::MARGIN_ROW_ALPHA_SECS,
    20.0
);

f64_value_gate!(
    MARGIN_ROW_ALPHA_LR,
    set_margin_row_alpha_lr_preset,
    margin_row_alpha_lr,
    ny_levers::decls::sound_channel::MARGIN_ROW_ALPHA_LR,
    0.1
);

usize_value_gate!(
    MARGIN_ROW_ALPHA_COLS,
    set_margin_row_alpha_cols_preset,
    margin_row_alpha_cols,
    ny_levers::decls::sound_channel::MARGIN_ROW_ALPHA_COLS,
    8
);

bool_gate!(
    MARGIN_ROW_ALPHA_GUARD,
    set_margin_row_alpha_guard_preset,
    margin_row_alpha_guard_enabled,
    ny_levers::decls::sound_channel::MARGIN_ROW_ALPHA_GUARD
);

bool_gate!(
    MARGIN_ROW_SPEC_ALPHA,
    set_margin_row_spec_alpha_preset,
    margin_row_spec_alpha_enabled,
    ny_levers::decls::sound_channel::MARGIN_ROW_SPEC_ALPHA
);

usize_value_gate!(
    MARGIN_ROW_SPEC_ALPHA_WIDTH,
    set_margin_row_spec_alpha_width_preset,
    margin_row_spec_alpha_width,
    ny_levers::decls::sound_channel::MARGIN_ROW_SPEC_ALPHA_WIDTH,
    3
);

usize_value_gate!(
    MARGIN_ROW_SPEC_ALPHA_CANDS,
    set_margin_row_spec_alpha_cands_preset,
    margin_row_spec_alpha_cands,
    ny_levers::decls::sound_channel::MARGIN_ROW_SPEC_ALPHA_CANDS,
    3
);

usize_value_gate!(
    MARGIN_ROW_SPEC_ALPHA_ROUNDS,
    set_margin_row_spec_alpha_rounds_preset,
    margin_row_spec_alpha_rounds,
    ny_levers::decls::sound_channel::MARGIN_ROW_SPEC_ALPHA_ROUNDS,
    1
);

usize_value_gate!(
    MARGIN_ROW_SPEC_ALPHA_ENTRIES,
    set_margin_row_spec_alpha_entries_preset,
    margin_row_spec_alpha_entries,
    ny_levers::decls::sound_channel::MARGIN_ROW_SPEC_ALPHA_ENTRIES,
    65_536
);

bool_gate!(
    MARGIN_ROW_CLIP,
    set_margin_row_clip_preset,
    margin_row_clip_enabled,
    ny_levers::decls::sound_channel::MARGIN_ROW_CLIP
);

bool_gate!(
    MARGIN_ROW_CLIP_HEAD,
    set_margin_row_clip_head_preset,
    margin_row_clip_head_enabled,
    ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_HEAD
);

bool_gate!(
    MARGIN_ROW_CLIP_INTERM,
    set_margin_row_clip_interm_preset,
    margin_row_clip_interm_enabled,
    ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_INTERM
);

bool_gate!(
    MARGIN_ROW_CLIP_ROWS,
    set_margin_row_clip_rows_preset,
    margin_row_clip_rows_enabled,
    ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_ROWS
);

usize_value_gate!(
    MARGIN_ROW_CLIP_TOPK,
    set_margin_row_clip_topk_preset,
    margin_row_clip_topk,
    ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_TOPK,
    20
);

f64_value_gate!(
    MARGIN_ROW_CLOSURE_THRESH,
    set_margin_row_closure_thresh_preset,
    margin_row_closure_thresh,
    ny_levers::decls::sound_channel::MARGIN_ROW_CLOSURE_THRESH,
    0.40
);

usize_value_gate!(
    MARGIN_ROW_CLOSURE_WARMUP,
    set_margin_row_closure_warmup_preset,
    margin_row_closure_warmup,
    ny_levers::decls::sound_channel::MARGIN_ROW_CLOSURE_WARMUP,
    24
);

usize_value_gate!(
    MARGIN_ROW_CLOSURE_WINDOW,
    set_margin_row_closure_window_preset,
    margin_row_closure_window,
    ny_levers::decls::sound_channel::MARGIN_ROW_CLOSURE_WINDOW,
    16
);

usize_value_gate!(
    MARGIN_ROW_CLOSURE_MAX_TIER,
    set_margin_row_closure_max_tier_preset,
    margin_row_closure_max_tier,
    ny_levers::decls::sound_channel::MARGIN_ROW_CLOSURE_MAX_TIER,
    3
);

usize_value_gate!(
    MARGIN_ROW_BI_CHUNK,
    set_margin_row_bi_chunk_preset,
    margin_row_bi_chunk,
    ny_levers::decls::margin_row::MARGIN_ROW_BI_CHUNK,
    256
);

f64_value_gate!(
    MARGIN_ROW_BI_CHUNK_SECS,
    set_margin_row_bi_chunk_secs_preset,
    margin_row_bi_chunk_secs,
    ny_levers::decls::margin_row::MARGIN_ROW_BI_CHUNK_SECS,
    0.0
);

usize_value_gate!(
    MARGIN_ROW_BI_TOPK,
    set_margin_row_bi_topk_preset,
    margin_row_bi_topk,
    ny_levers::decls::margin_row::MARGIN_ROW_BI_TOPK,
    1024
);

usize_value_gate!(
    MARGIN_ROW_INTERM_ALPHA,
    set_margin_row_interm_alpha_preset,
    margin_row_interm_alpha,
    ny_levers::decls::margin_row::MARGIN_ROW_INTERM_ALPHA,
    0
);

bool_gate!(
    MARGIN_ROW_BETA,
    set_margin_row_beta_preset,
    margin_row_beta_enabled,
    ny_levers::decls::margin_row::MARGIN_ROW_BETA
);

f64_value_gate!(
    MARGIN_ROW_BETA_ETA,
    set_margin_row_beta_eta_preset,
    margin_row_beta_eta,
    ny_levers::decls::margin_row::MARGIN_ROW_BETA_ETA,
    0.5
);

usize_value_gate!(
    MARGIN_ROW_BETA_ITERS,
    set_margin_row_beta_iters_preset,
    margin_row_beta_iters,
    ny_levers::decls::margin_row::MARGIN_ROW_BETA_ITERS,
    1
);

f64_value_gate!(
    MARGIN_ROW_BETA_LAMBDA,
    set_margin_row_beta_lambda_preset,
    margin_row_beta_lambda,
    ny_levers::decls::margin_row::MARGIN_ROW_BETA_LAMBDA,
    1.0
);

bool_gate!(
    MARGIN_ROW_BETA_POLYAK,
    set_margin_row_beta_polyak_preset,
    margin_row_beta_polyak_enabled,
    ny_levers::decls::margin_row::MARGIN_ROW_BETA_POLYAK
);

bool_gate!(
    MARGIN_ROW_BETA_HEADS,
    set_margin_row_beta_heads_preset,
    margin_row_beta_heads_enabled,
    ny_levers::decls::margin_row::MARGIN_ROW_BETA_HEADS
);

bool_gate!(
    MARGIN_ROW_REBAKE,
    set_margin_row_rebake_preset,
    margin_row_rebake_enabled,
    ny_levers::decls::margin_row::MARGIN_ROW_REBAKE
);

// #tableau-f32: f32 STORAGE for the root forward-tableau walk. Requires the
// f32 compute lane (`margin_row.root_f32`) — requested without it the build
// refuses loudly and stays f64 (`root.rs::build_retaining_inner`). Pure
// loosening: every store is exact, directed-up, or charged into the certified
// g_err concretize slack, so a wrong value costs proofs, never invents one.
bool_gate!(
    MARGIN_ROW_TABLEAU_F32,
    set_margin_row_tableau_f32_preset,
    margin_row_tableau_f32_enabled,
    ny_levers::decls::margin_row::MARGIN_ROW_TABLEAU_F32
);

// #mr-ibp-seed: intersect every trunk box of the root tableau walk with its
// certified IBP box, shrink-only, after the forward-only `bi_cal` capture and
// before #backward-interm (`margin_row/ibp_seed.rs`). This cell serves the
// COMPATIBILITY resolver only (`RootPrecisionRuntimePolicy::
// from_layered_process`); the scored route resolves the key from its frozen
// LeverSet. Dark = byte-identical; residual trunks decline.
bool_gate!(
    MARGIN_ROW_IBP_SEED,
    set_margin_row_ibp_seed_preset,
    margin_row_ibp_seed_enabled,
    ny_levers::decls::margin_row::MARGIN_ROW_IBP_SEED
);

// #domain-stack: score every domain in a frontier batch in ONE wide certified
// pass instead of one pass per domain. Column-local, so a stacked score is
// bit-identical to the independent one (pinned by
// `domain_stacked_candidate_scores_match_independent_frontier_bit_exact`); this
// selects a schedule, not a relaxation, and cannot move a published bound.
bool_gate!(
    MARGIN_ROW_DOMAIN_STACK,
    set_margin_row_domain_stack_preset,
    margin_row_domain_stack_enabled,
    ny_levers::decls::margin_row::MARGIN_ROW_DOMAIN_STACK
);

// #mr-errlane (margin-row throughput plan B3): the certified error-lane
// representation, `off` | `envelope` | `per_channel` | `audit`. A `text_gate!`
// rather than a `bool_gate!` because the key carries a second (per-channel)
// value and a measurement-only audit value on the same axis. This cell serves
// the COMPATIBILITY resolver only (`RootPrecisionRuntimePolicy::
// from_layered_process`); the scored route resolves the key from its frozen
// LeverSet and never reads it. Every value is sound: an envelope dominates the
// per-column lane, so a wrong value costs proofs (looseness, audit wall),
// never manufactures one.
text_gate!(
    MARGIN_ROW_ERR_ENVELOPE,
    set_margin_row_err_envelope_preset,
    margin_row_err_envelope,
    ny_levers::decls::margin_row::MARGIN_ROW_ERR_ENVELOPE
);

// #mr-gpu-eval: RETIRED 2026-09-29 (ENGINE_PROGRAM R1 kill line + R2). The
// cell survives only so an ENVIRONMENT request resolves and is REFUSED loudly
// (`margin_row::gpu_conv::requested` -> `[mr-gpu-conv] refused`); no bridge
// writes it (preset validation refuses `margin_row.gpu_conv`) and
// `gpu_conv::armed` is false in every build.
bool_gate!(
    MARGIN_ROW_GPU_CONV,
    set_margin_row_gpu_conv_preset,
    margin_row_gpu_conv_enabled,
    ny_levers::decls::margin_row::MARGIN_ROW_GPU_CONV
);

// #frozen-dual-bab (V1 stage B, on stage A's #frozen-dual-state): arm the
// certified frozen alpha-zonotope dual BaB as a tier between the margin-row
// root and its tree. A class it closes is certified on the whole box (every
// leaf `certified > 0`, level ledger closed); every other class is handed to
// the tree unchanged. Latched once per process by
// `margin_row::zono_state::armed`, after the CLI bridge writes the cell.
bool_gate!(
    MARGIN_ROW_FROZEN_DUAL_BAB,
    set_margin_row_frozen_dual_bab_preset,
    margin_row_frozen_dual_bab_enabled,
    ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB
);

// #frozen-dual-bab: nodes one objective's search may evaluate (reader clamp
// `1..=200M` in `zono_bab::TierPolicy`). Wall clock and memory only: a capped
// search is OPEN and handed back, never closed.
usize_value_gate!(
    MARGIN_ROW_FROZEN_DUAL_BAB_NODE_CAP,
    set_margin_row_frozen_dual_bab_node_cap_preset,
    margin_row_frozen_dual_bab_node_cap,
    ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB_NODE_CAP,
    2_000_000
);

// #frozen-dual-bab: the tier's wall request in seconds; 0 (the declaration
// default) takes half of the time the root left, a positive request is capped
// to three quarters of it (`zono_bab::TierPolicy::budget_secs`).
f64_value_gate!(
    MARGIN_ROW_FROZEN_DUAL_BAB_SECS,
    set_margin_row_frozen_dual_bab_secs_preset,
    margin_row_frozen_dual_bab_secs,
    ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB_SECS,
    0.0
);

// #breakpoint-dual: zero preserves the original frozen-dual evaluator;
// TierPolicy bounds a nonzero request to at most 32 refinement steps.
usize_value_gate!(
    MARGIN_ROW_FROZEN_DUAL_REFINE_STEPS,
    set_margin_row_frozen_dual_refine_steps_preset,
    margin_row_frozen_dual_refine_steps,
    ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_REFINE_STEPS,
    0
);

// #fd-shared-state (ENGINE_PROGRAM M3): how the frozen-dual tier builds its
// per-class states, `per_class` (the declaration default: one full build per
// class inside its fair share, the pre-M3 tier bit for bit) | `shared` (ONE
// trunk of every unstable row under slopes shared across the open classes,
// built before the shares, plus a one-column margin tail per class). A
// `text_gate!` because the key is an enum; read by
// `zono_bab::TierPolicy::from_gates`. Every value is sound: any slope in
// [0, 1] certifies, and a tail that disagrees with the trunk on a slope a row
// reads is refused.
text_gate!(
    MARGIN_ROW_FROZEN_DUAL_BAB_STATE,
    set_margin_row_frozen_dual_bab_state_preset,
    margin_row_frozen_dual_bab_state,
    ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB_STATE
);

// #wide-fd-search (ENGINE_PROGRAM M4): which search scores the frozen-dual
// tier's nodes, `f64` (default, the M1 path bit for bit) | `cpu_f32` |
// `cuda` | `auto` (`zono_bab::TierPolicy::from_gates`). A `text_gate!`
// because the key is an enum. Every value is sound: a backend only PROPOSES
// dual points; the tier certifies each prune in f64 with its own evaluator
// and splits every node no proposal certifies.
text_gate!(
    MARGIN_ROW_FROZEN_DUAL_BAB_SEARCH,
    set_margin_row_frozen_dual_bab_search_preset,
    margin_row_frozen_dual_bab_search,
    ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB_SEARCH
);

// #wave-beta-bab (ENGINE_PROGRAM M6): the margin-row tree engine of the outer
// frame, `legacy` (default: today's best-first tree bit for bit) | `wave`
// (`bab::wave`: batched untrusted beta-CROWN search, a streaming certify
// queue on a private pool, the path-based cover gate). Read by
// `bab::wave::WavePolicy::from_gates`. Every value is sound: the search only
// proposes; every close is certified by the lane's own evaluator.
text_gate!(
    MARGIN_ROW_TREE_MODE,
    set_margin_row_tree_mode_preset,
    margin_row_tree_mode,
    ny_levers::decls::margin_row::MARGIN_ROW_TREE_MODE
);

// #wave-beta-bab: domains per wave (reader clamp `1..=16384` in
// `bab::wave::WavePolicy::clamped`). Schedule only.
usize_value_gate!(
    MARGIN_ROW_TREE_WAVE_BATCH,
    set_margin_row_tree_wave_batch_preset,
    margin_row_tree_wave_batch,
    ny_levers::decls::margin_row::MARGIN_ROW_TREE_WAVE_BATCH,
    256
);

// #wave-beta-bab: threads of the private certify pool (0 = half of the lane's
// rayon threads; reader clamp `..=1024`). Schedule only.
usize_value_gate!(
    MARGIN_ROW_TREE_WAVE_CERT_THREADS,
    set_margin_row_tree_wave_cert_threads_preset,
    margin_row_tree_wave_cert_threads,
    ny_levers::decls::margin_row::MARGIN_ROW_TREE_WAVE_CERT_THREADS,
    0
);

usize_value_gate!(
    MARGIN_ROW_REBAKE_ROUNDS,
    set_margin_row_rebake_rounds_preset,
    margin_row_rebake_rounds,
    ny_levers::decls::margin_row::MARGIN_ROW_REBAKE_ROUNDS,
    2
);

f64_value_gate!(
    MARGIN_ROW_REBAKE_SECS,
    set_margin_row_rebake_secs_preset,
    margin_row_rebake_secs,
    ny_levers::decls::margin_row::MARGIN_ROW_REBAKE_SECS,
    30.0
);

// #bi-window: receptive-field-cropped backward-interm prefix passes. Each
// cropped pair is a certified pass of the unchanged engine on a net computing
// exactly the seeded pre-activations, never looser than the dense pair. This
// cell serves the COMPATIBILITY resolver only; the scored route resolves the
// key from its frozen LeverSet.
bool_gate!(
    MARGIN_ROW_BI_WINDOW,
    set_margin_row_bi_window_preset,
    margin_row_bi_window_enabled,
    ny_levers::decls::margin_row::MARGIN_ROW_BI_WINDOW
);

// #bi-gpu: RETIRED 2026-09-29 (ENGINE_PROGRAM R2, the device never
// certifies). The cell serves the COMPATIBILITY resolver only (the scored route
// resolves the flag from its frozen LeverSet); either way an armed phase is
// REFUSED at runtime (`[bi-gpu] refused`), and no bridge writes the cell
// (preset validation refuses `margin_row.bi_gpu`).
bool_gate!(
    MARGIN_ROW_BI_GPU,
    set_margin_row_bi_gpu_preset,
    margin_row_bi_gpu_enabled,
    ny_levers::decls::margin_row::MARGIN_ROW_BI_GPU
);

// #bi-patch: per-column receptive-field (patch-mode) backward-interm prefix
// passes — the arm, the crossover ratio and the per-pass working-set cap. Each
// patch pair is the dense pair restricted to its support (never looser). These
// cells serve the COMPATIBILITY resolver only; the scored route resolves the
// keys from its frozen LeverSet, like #bi-window's.
bool_gate!(
    MARGIN_ROW_BI_PATCH,
    set_margin_row_bi_patch_preset,
    margin_row_bi_patch_enabled,
    ny_levers::decls::margin_row::MARGIN_ROW_BI_PATCH
);

f64_value_gate!(
    MARGIN_ROW_BI_PATCH_RATIO,
    set_margin_row_bi_patch_ratio_preset,
    margin_row_bi_patch_ratio,
    ny_levers::decls::margin_row::MARGIN_ROW_BI_PATCH_RATIO,
    0.5
);

usize_value_gate!(
    MARGIN_ROW_BI_PATCH_MB,
    set_margin_row_bi_patch_mb_preset,
    margin_row_bi_patch_mb,
    ny_levers::decls::margin_row::MARGIN_ROW_BI_PATCH_MB,
    256
);

// #joint-refresh: rounds (0 = dark) and wall request of the backward-only joint
// intermediate refresh. Compatibility cells only, like #bi-window's.
usize_value_gate!(
    MARGIN_ROW_JOINT_REFRESH,
    set_margin_row_joint_refresh_preset,
    margin_row_joint_refresh,
    ny_levers::decls::margin_row::MARGIN_ROW_JOINT_REFRESH,
    0
);

f64_value_gate!(
    MARGIN_ROW_JOINT_REFRESH_SECS,
    set_margin_row_joint_refresh_secs_preset,
    margin_row_joint_refresh_secs,
    ny_levers::decls::margin_row::MARGIN_ROW_JOINT_REFRESH_SECS,
    30.0
);

// Source-owned head groups: compatibility cell only. Scored runs consume the
// same boolean from their frozen LeverSet and arm one fixed bounded policy.
bool_gate!(
    MARGIN_ROW_HEAD_GROUPS,
    set_margin_row_head_groups_preset,
    margin_row_head_groups_enabled,
    ny_levers::decls::margin_row::MARGIN_ROW_HEAD_GROUPS
);

bool_gate!(
    MARGIN_ROW_HEAD_DIRECTIONAL_CUTS,
    set_margin_row_head_directional_cuts_preset,
    margin_row_head_directional_cuts_enabled,
    ny_levers::decls::margin_row::MARGIN_ROW_HEAD_DIRECTIONAL_CUTS
);

// Optional complete Source phase cover; zero keeps existing head groups.
usize_value_gate!(
    MARGIN_ROW_HEAD_PHASE_COVER,
    set_margin_row_head_phase_cover_preset,
    margin_row_head_phase_cover,
    ny_levers::decls::margin_row::MARGIN_ROW_HEAD_PHASE_COVER,
    0
);

// #root-certify-once: arm, untrusted spec iterations, untrusted joint rounds and
// wall request of the search-untrusted / certify-once root alpha phase. The
// published root is only ever a CERTIFIED evaluation that strictly beats the
// certified incumbent. Compatibility cells only, like #joint-refresh's; the
// scored route resolves the keys from its frozen LeverSet.
bool_gate!(
    MARGIN_ROW_ROOT_CERTIFY_ONCE,
    set_margin_row_root_certify_once_preset,
    margin_row_root_certify_once_enabled,
    ny_levers::decls::margin_row::MARGIN_ROW_ROOT_CERTIFY_ONCE
);

usize_value_gate!(
    MARGIN_ROW_ROOT_CERTIFY_ONCE_ITERS,
    set_margin_row_root_certify_once_iters_preset,
    margin_row_root_certify_once_iters,
    ny_levers::decls::margin_row::MARGIN_ROW_ROOT_CERTIFY_ONCE_ITERS,
    20
);

usize_value_gate!(
    MARGIN_ROW_ROOT_CERTIFY_ONCE_REFRESH,
    set_margin_row_root_certify_once_refresh_preset,
    margin_row_root_certify_once_refresh,
    ny_levers::decls::margin_row::MARGIN_ROW_ROOT_CERTIFY_ONCE_REFRESH,
    0
);

f64_value_gate!(
    MARGIN_ROW_ROOT_CERTIFY_ONCE_SECS,
    set_margin_row_root_certify_once_secs_preset,
    margin_row_root_certify_once_secs,
    ny_levers::decls::margin_row::MARGIN_ROW_ROOT_CERTIFY_ONCE_SECS,
    20.0
);

// #bbe-joint-root (ENGINE_PROGRAM M5a): which UNTRUSTED search proposes the
// root slopes inside #root-certify-once, `ascent` (default, bit-identical) |
// `joint_f32` (the injected joint-alpha root search). A `text_gate!` because
// the key is an enum. Compatibility cell only; the scored route resolves the
// key from its frozen LeverSet. Every value is sound: the slopes reach the
// certified root only through the ingest gate and one certified evaluation.
text_gate!(
    MARGIN_ROW_ROOT_SEARCH,
    set_margin_row_root_search_preset,
    margin_row_root_search,
    ny_levers::decls::margin_row::MARGIN_ROW_ROOT_SEARCH
);

// #root-box-source (ENGINE_PROGRAM M5b): which certified construction supplies
// the root's trunk boxes, `tableau` (default, bit-identical) |
// `patch_backward` (`margin_row/root_box.rs`). A `text_gate!` because the key
// is an enum. Compatibility cell only; the scored route resolves the key from
// its frozen LeverSet into the root-precision policy. Every value is sound:
// both sources publish certified boxes only.
text_gate!(
    MARGIN_ROW_ROOT_BOX_SOURCE,
    set_margin_row_root_box_source_preset,
    margin_row_root_box_source,
    ny_levers::decls::margin_row::MARGIN_ROW_ROOT_BOX_SOURCE
);

usize_value_gate!(
    EPOCH_TIER0,
    set_epoch_tier0_preset,
    epoch_tier0,
    ny_levers::decls::margin_row::EPOCH_TIER0,
    0
);

usize_value_gate!(
    EPOCH_UNIVERSE,
    set_epoch_universe_preset,
    epoch_universe,
    ny_levers::decls::margin_row::EPOCH_UNIVERSE,
    64
);

usize_value_gate!(
    EPOCH_RETAIN,
    set_epoch_retain_preset,
    epoch_retain,
    ny_levers::decls::margin_row::EPOCH_RETAIN,
    128
);

// These two gates retain hand-written runtime semantics rather than using the
// mechanical macros above. LRU is an untrimmed usize with a concrete fallback;
// PARALLEL is a free-form legacy token whose meaning depends on the current
// Rayon worker count. Both still use the standard env > cell > declaration
// layering primitive, including non-UTF-8 presence suppressing the cell.
static MARGIN_ROW_LRU: U64Cell = U64Cell::new();

/// Record the typed y-pack LRU capacity before the margin-row lane starts.
pub fn set_margin_row_lru_preset(value: usize) {
    MARGIN_ROW_LRU.set(u64::try_from(value).expect("usize must fit the governed u64 cell"));
}

fn margin_row_lru_cap_from(cell: &U64Cell) -> usize {
    resolve_u64(&ny_levers::decls::margin_row::MARGIN_ROW_LRU, cell)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(64)
}

/// Resolve the y-pack LRU capacity: environment, typed preset, then 64.
#[must_use]
pub fn margin_row_lru_cap() -> usize {
    margin_row_lru_cap_from(&MARGIN_ROW_LRU)
}

static MARGIN_ROW_PARALLEL: TextCell = TextCell::new();

/// Record one validated canonical parallel-frontier token.
pub fn set_margin_row_parallel_preset(value: impl Into<String>) {
    MARGIN_ROW_PARALLEL.set(value);
}

fn margin_row_frontier_from(cell: &TextCell, available: usize) -> usize {
    let token = resolve_text(&ny_levers::decls::margin_row::MARGIN_ROW_PARALLEL, cell);
    crate::margin_row::margin_row_frontier_from_env(token.as_deref(), available)
}

fn margin_row_frontier_lower_from(cell: &TextCell, available: usize) -> usize {
    // Counterfactual for removing the scored environment arm: retain the
    // frozen typed base-preset layer, or the declaration's all-worker answer
    // when that key was absent. Do not reopen the environment here.
    let token = cell.get();
    crate::margin_row::margin_row_frontier_from_env(token.as_deref(), available)
}

/// Resolve the margin-row parallel frontier against the current worker count.
#[must_use]
pub fn margin_row_frontier(available: usize) -> usize {
    margin_row_frontier_from(&MARGIN_ROW_PARALLEL, available)
}

/// Effective frontier after removing only the environment treatment while
/// retaining the typed base-preset layer in the same Rayon context.
#[must_use]
pub fn margin_row_frontier_lower(available: usize) -> usize {
    margin_row_frontier_lower_from(&MARGIN_ROW_PARALLEL, available)
}

// --- remaining competition-searchable propagation controls ---------------

text_gate!(
    CONV_IBP_BLOCKED,
    set_conv_ibp_blocked_preset,
    conv_ibp_blocked_token,
    ny_levers::decls::conv_ibp_blocked::CONV_IBP_BLOCKED
);

bool_gate!(
    PATCHES_FINITE_EXPIRY,
    set_patches_finite_expiry_preset,
    patches_finite_expiry_enabled,
    ny_levers::decls::diagnostics::PATCHES_FINITE_EXPIRY
);

bool_gate!(
    NO_WALK_RECORD_ADMISSION,
    set_no_walk_record_admission_preset,
    no_walk_record_admission_enabled,
    ny_levers::decls::collection::NO_WALK_RECORD_ADMISSION
);

u64_gate!(
    CGAN_STACKED_BUDGET_MB,
    set_cgan_stacked_budget_mb_preset,
    cgan_stacked_budget_mb_override,
    ny_levers::decls::cgan_stacked::CGAN_STACKED_BUDGET_MB
);

// #patches-engine-refusal-containment + #fast-f32-prewarm-all-hosts
// (reports/measured-2026/cct_bound_gap_2026-08-31): the cct engine-wiring
// pair. Both DEFAULT OFF, set by no category-resolvable yaml, never
// autonomously armed. The containment reader is consulted per compose at the
// Conv2d Patches GEMM seam (`conv2d/ops.rs`); the prewarm reader is consulted
// once on the vnncomp route by the CLI, which owns the actual
// `fast_f32_gemm::prewarm_fast_f32_gemm()` call.
bool_gate!(
    PATCHES_ENGINE_REFUSAL_CONTAINMENT,
    set_patches_engine_refusal_containment_preset,
    patches_engine_refusal_containment_armed,
    ny_levers::decls::crown_ibp_budget::PATCHES_ENGINE_REFUSAL_CONTAINMENT
);

bool_gate!(
    FAST_F32_PREWARM,
    set_fast_f32_prewarm_preset,
    fast_f32_prewarm_armed,
    ny_levers::decls::crown_ibp_budget::FAST_F32_PREWARM
);

static POLLABLE_FAER: BoolCell = BoolCell::new();

/// Record the typed canonical Boolean arm for the legacy-truthy pollable-faer
/// selector. The raw predicate remains at its runtime read site.
pub fn set_pollable_faer_preset(on: bool) {
    POLLABLE_FAER.set(on);
}

/// Preset cell underneath the pollable-faer runtime's historical truthy
/// environment predicate.
#[must_use]
pub fn pollable_faer_cell() -> &'static BoolCell {
    &POLLABLE_FAER
}

/// Resolve the input-split nested-deadline selector over the typed
/// `BetaCrownConfig` value already carried to the consuming rebound.
/// Returns `(keep_deadline, source)`.
#[must_use]
pub fn input_split_nested_deadline_enabled(config: Option<bool>) -> (bool, &'static str) {
    use ny_levers::decls::dark_probes::INPUT_SPLIT_NESTED_DEADLINE;
    match ny_levers::read_over_config(
        &INPUT_SPLIT_NESTED_DEADLINE,
        config.map(ny_levers::LeverValue::Bool),
    ) {
        Ok(resolved) => (resolved.value.as_bool(), resolved.source.as_str()),
        // A Boolean config is admissible by construction. Retain a defensive
        // fail-closed answer for corrupted/internal callers.
        Err(_) => (true, "config-rejected"),
    }
}

// --- MIP leaf oracle + true-gradient replay --------------------------------

bool_gate!(
    GRAPH_MIP_LEAF_SAT,
    set_graph_mip_leaf_sat_preset,
    graph_mip_leaf_sat_enabled,
    ny_levers::decls::graph_mip::GRAPH_MIP_LEAF_SAT
);

bool_gate!(
    TRUE_GRAD_GPU_REPLAY,
    set_true_grad_gpu_replay_preset,
    true_grad_gpu_replay_enabled,
    ny_levers::decls::telemetry::TRUE_GRAD_GPU_REPLAY
);

// --- #lever-reachability batch 2 -------------------------------------------
//
// THE GPU CROWN LANE. These six are read in `ny-gpu`, which depends on this
// crate, so their cells live here rather than in a second copy of this module:
// one place the CLI bridge writes, one place a reviewer looks. `ny-gpu`'s
// readers call the resolvers below by their full path.

bool_gate!(
    BAB_RESNET_WIDE,
    set_bab_resnet_wide_preset,
    bab_resnet_wide_enabled,
    ny_levers::decls::wide_lane::BAB_RESNET_WIDE
);

bool_gate!(
    BAB_RESNET_WIDE_SUBGROUP,
    set_bab_resnet_wide_subgroup_preset,
    bab_resnet_wide_subgroup_enabled,
    ny_levers::decls::wide_lane::BAB_RESNET_WIDE_SUBGROUP
);

u64_gate!(
    SWEEP_CLASS_ROWS,
    set_sweep_class_rows_preset,
    sweep_class_rows_override,
    ny_levers::decls::comprehensive_rows::SWEEP_CLASS_ROWS
);

u64_gate!(
    SWEEP_CLASS_MIB,
    set_sweep_class_mib_preset,
    sweep_class_mib_override,
    ny_levers::decls::comprehensive_rows::SWEEP_CLASS_MIB
);

bool_gate!(
    RESNET_ERR_CONCRETIZE,
    set_resnet_err_concretize_preset,
    resnet_err_concretize_enabled,
    ny_levers::decls::sound_channel::RESNET_ERR_CONCRETIZE
);

bool_gate!(
    RESNET_ERR_CONCRETIZE_FINE,
    set_resnet_err_concretize_fine_preset,
    resnet_err_concretize_fine_enabled,
    ny_levers::decls::sound_channel::RESNET_ERR_CONCRETIZE_FINE
);

bool_gate!(
    CROWN_PARTIAL_FINITE_EXPIRY,
    set_crown_partial_finite_expiry_preset,
    crown_partial_finite_expiry_enabled,
    ny_levers::decls::diagnostics::CROWN_PARTIAL_FINITE_EXPIRY
);

// #dense-sparse-seed: the stability threshold of the sequential CROWN-IBP
// partial walk's Dense unstable-rows seed (`network/ibp/crown_partial.rs`).
// The declaration default 0.9 is that seed's historical literal, so an
// unwritten cell is byte-identical; the read site applies a written value only
// to targets whose successor layer is a ReLU.
f64_value_gate!(
    CROWN_PARTIAL_DENSE_MIN_STABLE,
    set_crown_partial_dense_min_stable_preset,
    crown_partial_dense_min_stable,
    ny_levers::decls::crown_partial_seed::CROWN_PARTIAL_DENSE_MIN_STABLE,
    0.9
);

// --- #lever-reachability batch 2: the `read_raw` predicate gates ------------
//
// These three keep their own unit-tested PURE PREDICATE at their own read site,
// and this module deliberately exposes only the CELL rather than a finished
// answer. Two reasons, both load-bearing:
//
// 1. for these levers the predicate IS the spec of the arming rule, and the
//    registry decls say so in as many words
//    (`decls/cgan_stacked.rs`: "the arming RULE stays in the pure predicate
//    below, which is unit-tested and is therefore the spec"). Moving the rule
//    here would leave that predicate reachable only from its own test — a
//    spec nothing executes;
// 2. the declaration's parser is NOT the same function. `NY_SEQ_WEIGHTED_BUDGET`
//    proves it: an OPT-OUT on `!= "0"` declared `Text` with effective default
//    `"1"`, so
//    resolving it through the parser would answer "off" where the shipped
//    behaviour is ON.
//
// `ny_levers::gate::resolve_raw_predicate` therefore layers UNDER the predicate
// at the call site: env token present, the predicate decides in both
// directions; absent, this cell; absent from both, the shipped default named
// there.

static SEQ_WEIGHTED_BUDGET: BoolCell = BoolCell::new();

/// Record the typed preset's answer for the sequential CROWN-IBP grant policy.
pub fn set_seq_weighted_budget_preset(on: bool) {
    SEQ_WEIGHTED_BUDGET.set(on);
}

/// The cell for `network::ibp::crown_ibp`'s opt-out grant-policy gate.
#[must_use]
pub fn seq_weighted_budget_cell() -> &'static BoolCell {
    &SEQ_WEIGHTED_BUDGET
}

static CGAN_STACKED_BACKWARD: BoolCell = BoolCell::new();

/// Record the typed preset's answer for the stacked cgan backward walk.
pub fn set_cgan_stacked_backward_preset(on: bool) {
    CGAN_STACKED_BACKWARD.set(on);
}

/// The cell for `graph_alpha::bounds::cgan_stacked`'s exact-`"1"` gate.
#[must_use]
pub fn cgan_stacked_backward_cell() -> &'static BoolCell {
    &CGAN_STACKED_BACKWARD
}

static CGAN_POLISH: BoolCell = BoolCell::new();

/// Record the typed preset's answer for the near-miss survivor alpha polish.
pub fn set_cgan_polish_preset(on: bool) {
    CGAN_POLISH.set(on);
}

/// The cell for `disjunctive_multi_clause`'s exact-`"1"` polish gate.
#[must_use]
pub fn cgan_polish_cell() -> &'static BoolCell {
    &CGAN_POLISH
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE PROPERTY THE PROMOTION COMMIT RESTS ON.
    ///
    /// With no preset key written — which is every shipped configuration, pinned
    /// by `no_shipped_preset_arms_a_promoted_lever_delivery_key` in
    /// `crates/ny-cli/src/preset/lever_delivery_pin_tests.rs` — every resolver here
    /// must return exactly what the declaration says, i.e. exactly what the
    /// pre-promotion `ny_levers::read(&DECL).value` returned.
    ///
    /// Deliberately asserted against the DECLARATION rather than against a
    /// hard-coded literal: a future default change should move both together,
    /// and a test that pins the literal would just have to be edited.
    ///
    /// Held under the shared test env lock (`with_serialized_env_vars(&[], ..)`
    /// edits nothing and only takes the lock): sibling tests in this crate
    /// legitimately set `NY_CLIP_INTERM_CERTIFIED` and friends for their own
    /// arms, and a resolver read racing one of those would be flaky.
    #[test]
    fn an_unwritten_cell_resolves_to_the_declaration_default() {
        ny_test_utils::env::with_serialized_env_vars(&[], declaration_default_parity);
    }

    fn declaration_default_parity() {
        use ny_levers::LeverValue;

        let bools: &[(&'static ny_levers::LeverDecl, bool)] = &[
            (
                &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA,
                root_joint_interm_alpha_enabled(),
            ),
            (
                &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT,
                root_joint_interm_alpha_deadline_ascent_enabled(),
            ),
            (
                &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP,
                root_joint_interm_alpha_frozen_stop_enabled(),
            ),
            (
                &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN,
                root_joint_interm_alpha_pure_chain_enabled(),
            ),
            (
                &ny_levers::decls::collection::BN_GPU_EXTRACT,
                bn_gpu_extract_enabled(),
            ),
            (
                &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_PROBE,
                root_joint_interm_alpha_probe_enabled(),
            ),
            (
                &ny_levers::decls::collection::ROOT_PHASE_RESIDENT_CROWN,
                root_phase_resident_crown_enabled(),
            ),
            (
                &ny_levers::decls::collection::ROOT_WIDE_DEMANDED_INTERM_CROWN,
                root_wide_demanded_interm_crown_enabled(),
            ),
            (
                &ny_levers::decls::collection::ROOT_CPU_PARALLEL_INTERM_CROWN,
                root_cpu_parallel_interm_crown_enabled(),
            ),
            (
                &ny_levers::decls::comprehensive_rows::ROOT_OBJECTIVE_DIRECTED_ROWS,
                root_objective_directed_rows_enabled(),
            ),
            (
                &ny_levers::decls::collection::CLIP_HOST_MEAN_LA,
                clip_host_mean_la_enabled(),
            ),
            (
                &ny_levers::decls::collection::CLIP_INTERM_CERTIFIED,
                clip_interm_certified_enabled(),
            ),
            (
                &ny_levers::decls::wide_lane::MO_GPU_CHUNK_DEADLINE,
                mo_gpu_chunk_deadline_enabled(),
            ),
            (
                &ny_levers::decls::mo_alpha_heuristic::MO_ALPHA_HEURISTIC,
                mo_alpha_heuristic_enabled(),
            ),
            (
                &ny_levers::decls::mo_optimizer::MO_BETA_SPSA,
                mo_beta_spsa_enabled(),
            ),
            (
                &ny_levers::decls::mo_optimizer::MO_ALPHA_STEP,
                mo_alpha_step_enabled(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_GPU,
                margin_row_gpu_enabled(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_GPU_BATCH,
                margin_row_gpu_batch_enabled(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_GPU_BACKWARD,
                margin_row_gpu_backward_enabled(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_ALPHA_OPT,
                margin_row_alpha_opt_enabled(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_ALPHA_GUARD,
                margin_row_alpha_guard_enabled(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_SPEC_ALPHA,
                margin_row_spec_alpha_enabled(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_CLIP,
                margin_row_clip_enabled(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_HEAD,
                margin_row_clip_head_enabled(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_INTERM,
                margin_row_clip_interm_enabled(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_ROWS,
                margin_row_clip_rows_enabled(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_BETA,
                margin_row_beta_enabled(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_BETA_POLYAK,
                margin_row_beta_polyak_enabled(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_BETA_HEADS,
                margin_row_beta_heads_enabled(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_REBAKE,
                margin_row_rebake_enabled(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_BI_WINDOW,
                margin_row_bi_window_enabled(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_HEAD_GROUPS,
                margin_row_head_groups_enabled(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_HEAD_DIRECTIONAL_CUTS,
                margin_row_head_directional_cuts_enabled(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_ROOT_CERTIFY_ONCE,
                margin_row_root_certify_once_enabled(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_BI_GPU,
                margin_row_bi_gpu_enabled(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_BI_PATCH,
                margin_row_bi_patch_enabled(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_TABLEAU_F32,
                margin_row_tableau_f32_enabled(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_IBP_SEED,
                margin_row_ibp_seed_enabled(),
            ),
            (
                &ny_levers::decls::diagnostics::PATCHES_FINITE_EXPIRY,
                patches_finite_expiry_enabled(),
            ),
            (
                &ny_levers::decls::collection::NO_WALK_RECORD_ADMISSION,
                no_walk_record_admission_enabled(),
            ),
            (
                &ny_levers::decls::graph_mip::GRAPH_MIP_LEAF_SAT,
                graph_mip_leaf_sat_enabled(),
            ),
            (
                &ny_levers::decls::telemetry::TRUE_GRAD_GPU_REPLAY,
                true_grad_gpu_replay_enabled(),
            ),
            (
                &ny_levers::decls::wide_lane::BAB_RESNET_WIDE,
                bab_resnet_wide_enabled(),
            ),
            (
                &ny_levers::decls::wide_lane::BAB_RESNET_WIDE_SUBGROUP,
                bab_resnet_wide_subgroup_enabled(),
            ),
            (
                &ny_levers::decls::sound_channel::RESNET_ERR_CONCRETIZE,
                resnet_err_concretize_enabled(),
            ),
            (
                &ny_levers::decls::sound_channel::RESNET_ERR_CONCRETIZE_FINE,
                resnet_err_concretize_fine_enabled(),
            ),
            (
                &ny_levers::decls::diagnostics::CROWN_PARTIAL_FINITE_EXPIRY,
                crown_partial_finite_expiry_enabled(),
            ),
        ];
        for &(decl, resolved) in bools {
            // The process environment is not touched here on purpose: this
            // asserts the SHIPPED resolution, which is what keeps the scored
            // path byte-identical.
            let expected = matches!(ny_levers::read(decl).value, LeverValue::Bool(true));
            assert_eq!(
                resolved, expected,
                "{} must resolve to its declaration default with no preset key",
                decl.name
            );
        }

        assert_eq!(
            root_comprehensive_gpu_interm_rows_override(),
            ny_levers::read(&ny_levers::decls::comprehensive_rows::ROOT_COMP_GPU_INTERM_ROWS)
                .value
                .as_u64()
        );
        assert_eq!(
            root_comprehensive_gpu_interm_secs_override(),
            ny_levers::read(&ny_levers::decls::comprehensive_rows::ROOT_COMP_GPU_INTERM_SECS)
                .value
                .as_u64()
        );
        assert_eq!(
            root_joint_interm_alpha_max_dim_override(),
            ny_levers::read(&ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_MAX_DIM)
                .value
                .as_u64()
        );
        assert_eq!(
            Some(root_joint_interm_alpha_secs()),
            ny_levers::read(&ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_SECS)
                .value
                .as_u64()
        );
        assert_eq!(
            Some(u64::try_from(root_joint_interm_alpha_max_sel()).unwrap()),
            ny_levers::read(&ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_MAX_SEL)
                .value
                .as_u64()
        );
        assert_eq!(
            root_joint_interm_alpha_layers(),
            ny_levers::read(&ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_LAYERS)
                .value
                .as_str()
                .map(str::to_owned)
        );
        assert_eq!(
            Some(root_joint_min_remaining_secs()),
            ny_levers::read(&ny_levers::decls::collection::ROOT_JOINT_MIN_REMAINING_SECS)
                .value
                .as_u64()
        );
        assert_eq!(
            kfsb_sim_share_override(),
            ny_levers::read(&ny_levers::decls::wide_lane::KFSB_SIM_SHARE)
                .value
                .as_f64()
        );
        for (decl, resolved) in [
            (
                &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_ITERS,
                root_joint_interm_alpha_iters(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_ALPHA_ITERS,
                margin_row_alpha_iters(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_ALPHA_COLS,
                margin_row_alpha_cols(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_SPEC_ALPHA_WIDTH,
                margin_row_spec_alpha_width(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_SPEC_ALPHA_CANDS,
                margin_row_spec_alpha_cands(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_SPEC_ALPHA_ROUNDS,
                margin_row_spec_alpha_rounds(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_SPEC_ALPHA_ENTRIES,
                margin_row_spec_alpha_entries(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_TOPK,
                margin_row_clip_topk(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_CLOSURE_WARMUP,
                margin_row_closure_warmup(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_CLOSURE_WINDOW,
                margin_row_closure_window(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_CLOSURE_MAX_TIER,
                margin_row_closure_max_tier(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_BI_CHUNK,
                margin_row_bi_chunk(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_BI_TOPK,
                margin_row_bi_topk(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_INTERM_ALPHA,
                margin_row_interm_alpha(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_BETA_ITERS,
                margin_row_beta_iters(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_REBAKE_ROUNDS,
                margin_row_rebake_rounds(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_JOINT_REFRESH,
                margin_row_joint_refresh(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_BI_PATCH_MB,
                margin_row_bi_patch_mb(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_ROOT_CERTIFY_ONCE_ITERS,
                margin_row_root_certify_once_iters(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_ROOT_CERTIFY_ONCE_REFRESH,
                margin_row_root_certify_once_refresh(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB_NODE_CAP,
                margin_row_frozen_dual_bab_node_cap(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_REFINE_STEPS,
                margin_row_frozen_dual_refine_steps(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_TREE_WAVE_BATCH,
                margin_row_tree_wave_batch(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_TREE_WAVE_CERT_THREADS,
                margin_row_tree_wave_cert_threads(),
            ),
            (&ny_levers::decls::margin_row::EPOCH_TIER0, epoch_tier0()),
            (
                &ny_levers::decls::margin_row::EPOCH_UNIVERSE,
                epoch_universe(),
            ),
            (&ny_levers::decls::margin_row::EPOCH_RETAIN, epoch_retain()),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_LRU,
                margin_row_lru_cap(),
            ),
        ] {
            assert_eq!(
                Some(u64::try_from(resolved).expect("resolved usize must fit u64")),
                ny_levers::read(decl).value.as_u64(),
                "{} must preserve its declaration default",
                decl.name
            );
        }
        for (decl, resolved) in [
            (
                &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_LR,
                root_joint_interm_alpha_lr(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_ALPHA_SECS,
                margin_row_alpha_secs(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_ALPHA_LR,
                margin_row_alpha_lr(),
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_CLOSURE_THRESH,
                margin_row_closure_thresh(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_BETA_ETA,
                margin_row_beta_eta(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_BETA_LAMBDA,
                margin_row_beta_lambda(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_REBAKE_SECS,
                margin_row_rebake_secs(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_JOINT_REFRESH_SECS,
                margin_row_joint_refresh_secs(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_BI_PATCH_RATIO,
                margin_row_bi_patch_ratio(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_ROOT_CERTIFY_ONCE_SECS,
                margin_row_root_certify_once_secs(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_BI_CHUNK_SECS,
                margin_row_bi_chunk_secs(),
            ),
            (
                &ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB_SECS,
                margin_row_frozen_dual_bab_secs(),
            ),
            (
                &ny_levers::decls::crown_partial_seed::CROWN_PARTIAL_DENSE_MIN_STABLE,
                crown_partial_dense_min_stable(),
            ),
        ] {
            assert_eq!(
                Some(resolved),
                ny_levers::read(decl).value.as_f64(),
                "{} must preserve its declaration default",
                decl.name
            );
        }
        assert_eq!(
            sweep_class_rows_override(),
            ny_levers::read(&ny_levers::decls::comprehensive_rows::SWEEP_CLASS_ROWS)
                .value
                .as_u64()
        );
        assert_eq!(
            sweep_class_mib_override(),
            ny_levers::read(&ny_levers::decls::comprehensive_rows::SWEEP_CLASS_MIB)
                .value
                .as_u64()
        );
        assert_eq!(
            conv_ibp_blocked_token().as_deref(),
            ny_levers::read(&ny_levers::decls::conv_ibp_blocked::CONV_IBP_BLOCKED)
                .value
                .as_str()
        );
        // #bbe-joint-root: the unwritten enum cell resolves to the declaration
        // default, which is the bit-identical `ascent` search.
        assert_eq!(
            margin_row_root_search().as_deref(),
            ny_levers::read(&ny_levers::decls::margin_row::MARGIN_ROW_ROOT_SEARCH)
                .value
                .as_str()
        );
        assert_eq!(
            margin_row_root_search().as_deref(),
            Some(crate::margin_row::RootSearch::Ascent.token())
        );
        // #fd-shared-state: the unwritten enum cell resolves to the declaration
        // default, which is the bit-identical `per_class` tier.
        assert_eq!(
            margin_row_frozen_dual_bab_state().as_deref(),
            ny_levers::read(&ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB_STATE)
                .value
                .as_str()
        );
        assert_eq!(
            margin_row_frozen_dual_bab_state().as_deref(),
            Some(crate::margin_row::zono_bab::FrozenDualStateMode::PerClass.token())
        );
        // #root-box-source: the unwritten enum cell resolves to the
        // declaration default, which is the bit-identical `tableau` source.
        assert_eq!(
            margin_row_root_box_source().as_deref(),
            ny_levers::read(&ny_levers::decls::margin_row::MARGIN_ROW_ROOT_BOX_SOURCE)
                .value
                .as_str()
        );
        assert_eq!(
            margin_row_root_box_source().as_deref(),
            Some(crate::margin_row::RootBoxSource::Tableau.token())
        );
        // #wide-fd-search: the unwritten enum cell resolves to the declaration
        // default, the bit-identical `f64` search.
        assert_eq!(
            margin_row_frozen_dual_bab_search().as_deref(),
            ny_levers::read(&ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB_SEARCH)
                .value
                .as_str()
        );
        assert_eq!(
            margin_row_frozen_dual_bab_search().as_deref(),
            Some(crate::margin_row::FrozenDualSearch::F64.token())
        );
        // #wave-beta-bab: the unwritten enum cell resolves to the declaration
        // default, the bit-identical `legacy` tree.
        assert_eq!(
            margin_row_tree_mode().as_deref(),
            ny_levers::read(&ny_levers::decls::margin_row::MARGIN_ROW_TREE_MODE)
                .value
                .as_str()
        );
        assert_eq!(
            margin_row_tree_mode().as_deref(),
            Some(crate::margin_row::TreeMode::Legacy.token())
        );
        let margin_row_parallel =
            ny_levers::read(&ny_levers::decls::margin_row::MARGIN_ROW_PARALLEL);
        let expected_frontier =
            crate::margin_row::margin_row_frontier_from_env(margin_row_parallel.value.as_str(), 12);
        assert_eq!(
            margin_row_frontier(12),
            expected_frontier,
            "the unwritten cell must preserve the environment/declaration resolution"
        );
        assert_eq!(
            cgan_stacked_budget_mb_override(),
            ny_levers::read(&ny_levers::decls::cgan_stacked::CGAN_STACKED_BUDGET_MB)
                .value
                .as_u64()
        );
        assert_eq!(
            pollable_faer_cell().get(),
            None,
            "the canonical typed arm must remain unwritten by default"
        );

        // The three `read_raw` predicate gates. Their cells live here but their
        // RULES live at their read sites, so what this pins is the property the
        // cells must have: unwritten, they must not change any of the three
        // answers — including the OPT-OUT one, where the shipped state is ARMED
        // and a naive `Option<bool>::unwrap_or_default()` would silently
        // disarm it.
        for (decl, cell, shipped_default) in [
            (
                &ny_levers::decls::seq_weighted_budget::SEQ_WEIGHTED_BUDGET,
                seq_weighted_budget_cell(),
                true,
            ),
            (
                &ny_levers::decls::cgan_stacked::CGAN_STACKED_BACKWARD,
                cgan_stacked_backward_cell(),
                false,
            ),
            (
                &ny_levers::decls::cgan_stacked::CGAN_POLISH,
                cgan_polish_cell(),
                false,
            ),
        ] {
            assert_eq!(
                cell.get(),
                None,
                "{} must have no preset cell written on a default run",
                decl.name
            );
            let predicate = |raw: Option<&str>| {
                if shipped_default {
                    raw != Some("0")
                } else {
                    raw == Some("1")
                }
            };
            assert_eq!(
                ny_levers::gate::resolve_raw_predicate(decl, cell, shipped_default, predicate),
                predicate(ny_levers::read_raw(decl).as_deref()),
                "{} must resolve exactly as its own predicate does with no cell",
                decl.name
            );
        }
    }

    #[test]
    fn root_joint_family_resolves_environment_over_preset_over_default() {
        let _env_lock = ny_test_utils::env::lock_env();
        let master = BoolCell::new();
        master.set(true);
        let deadline_ascent = BoolCell::new();
        deadline_ascent.set(false);
        let secs = U64Cell::new();
        secs.set(60);
        let iters = U64Cell::new();
        iters.set(12);
        let lr = F64Cell::new();
        lr.set(0.2);
        let max_sel = U64Cell::new();
        max_sel.set(8);
        let frozen_stop = BoolCell::new();
        frozen_stop.set(true);
        let pure_chain = BoolCell::new();
        pure_chain.set(true);
        let bn_extract = BoolCell::new();
        bn_extract.set(true);
        let layers = TextCell::new();
        layers.set("Relu_7,Relu_11");
        let min_remaining = U64Cell::new();
        min_remaining.set(120);

        {
            let _master = ny_test_utils::env::ScopedEnvVar::unset(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA.name,
            );
            let _deadline = ny_test_utils::env::ScopedEnvVar::unset(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT.name,
            );
            let _secs = ny_test_utils::env::ScopedEnvVar::unset(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_SECS.name,
            );
            let _iters = ny_test_utils::env::ScopedEnvVar::unset(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_ITERS.name,
            );
            let _lr = ny_test_utils::env::ScopedEnvVar::unset(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_LR.name,
            );
            let _max_sel = ny_test_utils::env::ScopedEnvVar::unset(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_MAX_SEL.name,
            );
            let _frozen_stop = ny_test_utils::env::ScopedEnvVar::unset(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP.name,
            );
            let _pure_chain = ny_test_utils::env::ScopedEnvVar::unset(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN.name,
            );
            let _bn_extract = ny_test_utils::env::ScopedEnvVar::unset(
                ny_levers::decls::collection::BN_GPU_EXTRACT.name,
            );
            let _layers = ny_test_utils::env::ScopedEnvVar::unset(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_LAYERS.name,
            );
            let _min = ny_test_utils::env::ScopedEnvVar::unset(
                ny_levers::decls::collection::ROOT_JOINT_MIN_REMAINING_SECS.name,
            );

            assert!(resolve_bool(
                &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA,
                &master
            ));
            assert!(!resolve_bool(
                &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT,
                &deadline_ascent
            ));
            assert_eq!(
                resolve_u64(
                    &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_SECS,
                    &secs
                ),
                Some(60)
            );
            assert_eq!(
                resolve_u64(
                    &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_ITERS,
                    &iters
                ),
                Some(12)
            );
            assert_eq!(
                resolve_f64(
                    &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_LR,
                    &lr
                ),
                Some(0.2)
            );
            assert_eq!(
                resolve_u64(
                    &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_MAX_SEL,
                    &max_sel
                ),
                Some(8)
            );
            assert!(resolve_bool(
                &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP,
                &frozen_stop
            ));
            assert!(resolve_bool(
                &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN,
                &pure_chain
            ));
            assert!(resolve_bool(
                &ny_levers::decls::collection::BN_GPU_EXTRACT,
                &bn_extract
            ));
            assert_eq!(
                resolve_text(
                    &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_LAYERS,
                    &layers
                )
                .as_deref(),
                Some("Relu_7,Relu_11")
            );
            assert_eq!(
                resolve_u64(
                    &ny_levers::decls::collection::ROOT_JOINT_MIN_REMAINING_SECS,
                    &min_remaining
                ),
                Some(120)
            );
        }

        {
            let _master = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA.name,
                "0",
            );
            let _deadline = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT.name,
                "1",
            );
            let _secs = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_SECS.name,
                "12",
            );
            let _iters = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_ITERS.name,
                "7",
            );
            let _lr = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_LR.name,
                "0.25",
            );
            let _max_sel = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_MAX_SEL.name,
                "16",
            );
            let _frozen_stop = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP.name,
                "0",
            );
            let _pure_chain = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN.name,
                "0",
            );
            let _bn_extract = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::BN_GPU_EXTRACT.name,
                "0",
            );
            let _layers = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_LAYERS.name,
                "all",
            );
            let _min = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::ROOT_JOINT_MIN_REMAINING_SECS.name,
                "0",
            );

            assert!(!resolve_bool(
                &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA,
                &master
            ));
            assert!(resolve_bool(
                &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT,
                &deadline_ascent
            ));
            assert_eq!(
                resolve_u64(
                    &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_SECS,
                    &secs
                ),
                Some(12)
            );
            assert_eq!(
                resolve_u64(
                    &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_ITERS,
                    &iters
                ),
                Some(7)
            );
            assert_eq!(
                resolve_f64(
                    &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_LR,
                    &lr
                ),
                Some(0.25)
            );
            assert_eq!(
                resolve_u64(
                    &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_MAX_SEL,
                    &max_sel
                ),
                Some(16)
            );
            assert!(!resolve_bool(
                &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP,
                &frozen_stop
            ));
            assert!(!resolve_bool(
                &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN,
                &pure_chain
            ));
            assert!(!resolve_bool(
                &ny_levers::decls::collection::BN_GPU_EXTRACT,
                &bn_extract
            ));
            assert_eq!(
                resolve_text(
                    &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_LAYERS,
                    &layers
                )
                .as_deref(),
                Some("all")
            );
            assert_eq!(
                resolve_u64(
                    &ny_levers::decls::collection::ROOT_JOINT_MIN_REMAINING_SECS,
                    &min_remaining
                ),
                Some(0)
            );
        }

        {
            let _master = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA.name,
                "true",
            );
            let _deadline = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT.name,
                "yes",
            );
            let _secs = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_SECS.name,
                "bad",
            );
            let _iters = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_ITERS.name,
                "-1",
            );
            let _lr = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_LR.name,
                "nan",
            );
            let _max_sel = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_MAX_SEL.name,
                "bad",
            );
            let _frozen_stop = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP.name,
                "yes",
            );
            let _pure_chain = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN.name,
                "true",
            );
            let _bn_extract = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::BN_GPU_EXTRACT.name,
                "on",
            );
            let _min = ny_test_utils::env::ScopedEnvVar::set(
                ny_levers::decls::collection::ROOT_JOINT_MIN_REMAINING_SECS.name,
                "bad",
            );

            assert!(!resolve_bool(
                &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA,
                &master
            ));
            assert!(!resolve_bool(
                &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT,
                &deadline_ascent
            ));
            assert_eq!(
                resolve_u64(
                    &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_SECS,
                    &secs
                ),
                Some(30)
            );
            assert_eq!(
                resolve_u64(
                    &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_ITERS,
                    &iters
                ),
                Some(100)
            );
            assert_eq!(
                resolve_f64(
                    &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_LR,
                    &lr
                ),
                Some(0.1)
            );
            assert_eq!(
                resolve_u64(
                    &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_MAX_SEL,
                    &max_sel
                ),
                Some(512)
            );
            assert!(!resolve_bool(
                &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP,
                &frozen_stop
            ));
            assert!(!resolve_bool(
                &ny_levers::decls::collection::ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN,
                &pure_chain
            ));
            assert!(!resolve_bool(
                &ny_levers::decls::collection::BN_GPU_EXTRACT,
                &bn_extract
            ));
            assert_eq!(
                resolve_u64(
                    &ny_levers::decls::collection::ROOT_JOINT_MIN_REMAINING_SECS,
                    &min_remaining
                ),
                Some(240)
            );
        }
    }

    #[test]
    fn margin_row_throughput_gates_preserve_legacy_grammar_and_layering() {
        let _env_lock = ny_test_utils::env::lock_env();
        let lru = U64Cell::new();
        lru.set(128);
        let parallel = TextCell::new();
        parallel.set("8");
        let empty_parallel = TextCell::new();

        {
            let _lru_env = ny_test_utils::env::ScopedEnvVar::unset("NY_MARGIN_ROW_LRU");
            let _parallel_env = ny_test_utils::env::ScopedEnvVar::unset("NY_MARGIN_ROW_PARALLEL");
            assert_eq!(margin_row_lru_cap_from(&lru), 128);
            assert_eq!(margin_row_frontier_from(&parallel, 12), 8);
            assert_eq!(margin_row_frontier_lower_from(&parallel, 12), 8);
            assert_eq!(margin_row_frontier_lower_from(&empty_parallel, 12), 12);
        }

        {
            let _lru_env = ny_test_utils::env::ScopedEnvVar::set("NY_MARGIN_ROW_LRU", "32");
            let _parallel_env =
                ny_test_utils::env::ScopedEnvVar::set("NY_MARGIN_ROW_PARALLEL", " 17 ");
            assert_eq!(margin_row_lru_cap_from(&lru), 32);
            assert_eq!(
                margin_row_frontier_from(&parallel, 12),
                17,
                "numeric widths are not clamped to the available worker count"
            );
            assert_eq!(
                margin_row_frontier_lower_from(&parallel, 12),
                8,
                "the counterfactual removes only the environment arm"
            );
            assert_eq!(
                margin_row_frontier_lower_from(&empty_parallel, 12),
                12,
                "without a typed base preset the lower layer is all workers"
            );
        }

        {
            let _lru_env = ny_test_utils::env::ScopedEnvVar::set("NY_MARGIN_ROW_LRU", " 32 ");
            let _parallel_env =
                ny_test_utils::env::ScopedEnvVar::set("NY_MARGIN_ROW_PARALLEL", "malformed");
            assert_eq!(
                margin_row_lru_cap_from(&lru),
                64,
                "untrimmed malformed env suppresses the typed LRU"
            );
            assert_eq!(
                margin_row_frontier_from(&parallel, 12),
                1,
                "malformed env suppresses the typed frontier and selects serial"
            );
        }

        for (token, expected) in [
            ("", 1),
            ("0", 1),
            ("false", 1),
            ("OFF", 1),
            ("no", 1),
            ("1", 12),
            ("true", 12),
            ("ON", 12),
            ("yes", 12),
        ] {
            let _parallel_env =
                ny_test_utils::env::ScopedEnvVar::set("NY_MARGIN_ROW_PARALLEL", token);
            assert_eq!(
                margin_row_frontier_from(&parallel, 12),
                expected,
                "{token:?}"
            );
        }
    }

    /// Non-UTF-8 is present to governance and suppresses the typed cell, but
    /// the old `std::env::var` parser saw absence: LRU returns 64 and PARALLEL
    /// returns all workers, not either typed treatment.
    #[cfg(unix)]
    #[test]
    fn margin_row_non_utf8_env_preserves_legacy_answer_and_suppresses_typed() {
        use std::os::unix::ffi::OsStrExt;

        let _env_lock = ny_test_utils::env::lock_env();
        let invalid = std::ffi::OsStr::from_bytes(&[0xff]);
        let _lru_env = ny_test_utils::env::ScopedEnvVar::set_os("NY_MARGIN_ROW_LRU", invalid);
        let _parallel_env =
            ny_test_utils::env::ScopedEnvVar::set_os("NY_MARGIN_ROW_PARALLEL", invalid);

        let lru = U64Cell::new();
        lru.set(128);
        let parallel = TextCell::new();
        parallel.set("8");
        assert_eq!(margin_row_lru_cap_from(&lru), 64);
        assert_eq!(margin_row_frontier_from(&parallel, 12), 12);
    }

    /// #spec-alpha REACHABILITY (T12). `run_instance.sh` exports ZERO `NY_*`,
    /// so a lever reachable only through the environment is dead capability in
    /// competition however well built. The assertion that matters is therefore
    /// on the SOURCE, not the value: arming through the typed cell must resolve
    /// `Source::Config`.
    ///
    /// NECESSARY BUT NOT SUFFICIENT, and the gap has shipped dead capability
    /// before: this exercises the RESOLVER, not the CLI bridge that writes the
    /// cell before the lane starts -- and that bridge has been silently
    /// clobbered in a merge before, turning every typed key dark while a test
    /// exactly like this one stayed green. The arrival proof is a RUN: armed
    /// yaml, zero `NY_*` set, and `[spec-alpha] armed:` plus `apply: round=1`
    /// on the row's own stderr.
    #[test]
    fn margin_row_spec_alpha_keys_are_delivered_by_preset_not_env() {
        use ny_levers::{LeverValue, Source};

        let decl = &ny_levers::decls::sound_channel::MARGIN_ROW_SPEC_ALPHA;
        let resolve = |preset: Option<bool>, raw: Option<&str>| {
            let raw = raw.map(str::to_owned);
            ny_levers::read_over_config_with(decl, move |_| raw, preset.map(LeverValue::Bool))
                .expect("Bool config is admissible for a Bool declaration")
        };

        // Dark by default: promotion is not arming, and no shipped yaml arms it.
        let absent = resolve(None, None);
        assert!(!absent.value.as_bool());
        assert_eq!(absent.source, Source::Default);

        let armed = resolve(Some(true), None);
        assert!(armed.value.as_bool());
        assert_ne!(
            armed.source,
            Source::LegacyEnv,
            "arming through the typed key must NOT report a legacy-env source; \
             that would mean the preset route is not what fired"
        );
        assert_eq!(armed.source, Source::Config);
        assert!(!resolve(Some(false), None).value.as_bool());

        // Env wins in BOTH directions, so a developer A/B keeps working and
        // `=0` stays a kill switch.
        assert!(resolve(Some(false), Some("1")).value.as_bool());
        let killed = resolve(Some(true), Some("0"));
        assert!(!killed.value.as_bool());
        assert_eq!(killed.source, Source::LegacyEnv);

        // The four child values: same source requirement, defaults intact.
        let children: [(&ny_levers::LeverDecl, u64); 4] = [
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_SPEC_ALPHA_WIDTH,
                3,
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_SPEC_ALPHA_CANDS,
                3,
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_SPEC_ALPHA_ROUNDS,
                1,
            ),
            (
                &ny_levers::decls::sound_channel::MARGIN_ROW_SPEC_ALPHA_ENTRIES,
                65_536,
            ),
        ];
        for (decl, want_default) in children {
            let bare = ny_levers::read_over_config_with(decl, |_| None, None)
                .expect("U64 config is admissible");
            assert_eq!(
                bare.value.as_u64(),
                Some(want_default),
                "{} default drifted",
                decl.name
            );
            assert_eq!(bare.source, Source::Default, "{}", decl.name);
            let set = ny_levers::read_over_config_with(decl, |_| None, Some(LeverValue::U64(2)))
                .expect("U64 config is admissible");
            assert_eq!(set.value.as_u64(), Some(2), "{}", decl.name);
            assert_ne!(set.source, Source::LegacyEnv, "{}", decl.name);
            assert_eq!(set.source, Source::Config, "{}", decl.name);
        }
    }

    /// #clip-head REACHABILITY. NECESSARY, NOT SUFFICIENT: this exercises the
    /// RESOLVER, and the resolver is not the CLI bridge that writes the cell
    /// before the lane starts. That bridge was once "silently clobbered in a
    /// merge", turning every typed `margin_row` key dark while a test of this
    /// exact shape stayed green. The arrival proof is a run with the ARMED YAML
    /// and ZERO `NY_*` set that prints `[clip-head] armed` — see the key's own
    /// doc. This test only rules out the OTHER failure: a key that can be armed
    /// by nothing but an environment variable `run_instance.sh` never exports.
    #[test]
    fn clip_head_is_reachable_by_preset_not_only_by_env() {
        use ny_levers::{LeverValue, Source};

        let decl = &ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_HEAD;
        let resolve = |preset: Option<bool>, raw: Option<&str>| {
            let raw = raw.map(str::to_owned);
            ny_levers::read_over_config_with(decl, move |_| raw, preset.map(LeverValue::Bool))
                .expect("Bool config is admissible for a Bool declaration")
        };

        let absent = resolve(None, None);
        assert!(!absent.value.as_bool(), "the head clip ships dark");
        assert_eq!(absent.source, Source::Default);

        let armed = resolve(Some(true), None);
        assert!(armed.value.as_bool());
        assert_ne!(
            armed.source,
            Source::LegacyEnv,
            "arming through the typed key must NOT report a legacy-env source; \
             that would mean the preset route is not what fired"
        );
        assert_eq!(armed.source, Source::Config);
        assert!(!resolve(Some(false), None).value.as_bool());

        // Env wins in BOTH directions, so an A/B recipe keeps working and `=0`
        // stays a kill switch on a yaml that arms it.
        assert!(resolve(Some(false), Some("1")).value.as_bool());
        let killed = resolve(Some(true), Some("0"));
        assert!(!killed.value.as_bool());
        assert_eq!(killed.source, Source::LegacyEnv);
    }

    /// #tableau-f32 REACHABILITY. The rule this repo learned the hard way is
    /// that a lever "measured as armed" through an environment variable is
    /// still dead code in competition, because `run_instance.sh` exports
    /// exactly one `NY_*`. So the assertion that matters is on the SOURCE, not
    /// the value: arming through the typed cell must resolve
    /// `Source::Config` — NOT `Source::LegacyEnv` — because only the former is
    /// a route the scored entry point can actually drive.
    ///
    /// Also pins the parser NARROWING this promotion performed. The historical
    /// ad-hoc reader accepted `1`/`true`/`on`; the declaration's rule is exact
    /// `"1"`/`"0"`, and every other present token is a KILL SWITCH that
    /// suppresses the preset and resolves dark rather than riding it.
    #[test]
    fn tableau_f32_is_reachable_by_preset_not_only_by_env() {
        use ny_levers::{LeverValue, Source};

        let decl = &ny_levers::decls::margin_row::MARGIN_ROW_TABLEAU_F32;
        let resolve = |preset: Option<bool>, raw: Option<&str>| {
            let raw = raw.map(str::to_owned);
            ny_levers::read_over_config_with(decl, move |_| raw, preset.map(LeverValue::Bool))
                .expect("Bool config is admissible for a Bool declaration")
        };

        // Dark by default: promotion is not arming.
        let absent = resolve(None, None);
        assert!(!absent.value.as_bool());
        assert_eq!(absent.source, Source::Default);

        // THE REACHABILITY ASSERTION: the typed key arms it, and the source is
        // the preset — the route a scored run has.
        let armed = resolve(Some(true), None);
        assert!(armed.value.as_bool());
        assert_ne!(
            armed.source,
            Source::LegacyEnv,
            "arming through the typed key must NOT report a legacy-env source; \
             that would mean the preset route is not what fired"
        );
        assert_eq!(armed.source, Source::Config);
        assert!(!resolve(Some(false), None).value.as_bool());

        // Env wins in BOTH directions over the preset, so every banked A/B
        // recipe keeps working and `=0` stays a kill switch.
        assert!(resolve(Some(false), Some("1")).value.as_bool());
        let killed = resolve(Some(true), Some("0"));
        assert!(!killed.value.as_bool());
        assert_eq!(killed.source, Source::LegacyEnv);

        // The narrowing: previously-accepted spellings are now REJECTIONS that
        // suppress the preset instead of arming it.
        for token in ["true", "on", "TRUE", " 1", "01", "", "2"] {
            let r = resolve(Some(true), Some(token));
            assert!(
                !r.value.as_bool(),
                "{token:?} must not arm the storage lane"
            );
            assert_eq!(r.source, Source::LegacyEnvRejected, "{token:?}");
        }
    }
}
