// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lever-reachability — the pin that keeps PROMOTION from becoming ARMING.
//!
//! # What a promotion is, and what it is not
//!
//! `vnncomp_scripts/run_instance.sh` exports no `NY_*` variables, so a
//! lever that only an environment variable can arm is dead code on a scored
//! run. The fix is a typed preset key — a DELIVERY ROUTE. Building the route is
//! not the same decision as driving down it, and conflating the two is how a
//! batch of plumbing turns into an unmeasured change to sixteen benchmarks.
//!
//! So the contract of the 2026-08-23 batch is: every promoted key is absent
//! from every shipped yaml, therefore every promoted lever resolves exactly as
//! it did before promotion, therefore the batch is byte-identical on every
//! scored run, except for the explicit owner-authorized arming entries below.
//! This file pins those exceptions mechanically.
//!
//! # Why the key list is written out rather than derived
//!
//! A derived check (`serde_json` round-trip, "no key under `bab` is set")
//! would drift silently the moment somebody adds a key that SHOULD ship armed —
//! and there are plenty. Naming each promoted key means arming one is a
//! deliberate edit to this list, in the same commit, with a measurement to point
//! at. That is the whole point.

use super::*;
use crate::commands::beta_crown::BetaCrownPresetSnapshot;
use std::{fs, path::Path};

/// Every configuration year that ships.
const YEARS: &[&str] = &["vnncomp24", "vnncomp25", "vnncomp26"];

/// One promoted key: its dotted preset path, and a probe that says whether a
/// given loaded preset sets it.
struct PromotedKey {
    path: &'static str,
    is_set: fn(&PresetConfig) -> bool,
}

/// The 2026-08-23 #lever-reachability batch, in the same order as
/// `preset_gates::apply` where a compatibility cell exists, followed by
/// receipt-owned fields that deliberately have no scored global setter.
///
/// `NY_MARGIN_ROW_K_ADAPT` (`margin_row.k_adaptive`) is deliberately NOT here:
/// its axis was corrected rather than promoted, it predates this batch, and
/// `cifar100_2024` may legitimately want it. `cgan_reclaim` is likewise absent
/// because it is a sibling commit's key with its own pin.
const PROMOTED: &[PromotedKey] = &[
    // Attack / falsification slice.
    PromotedKey {
        path: "attack.falsify_portfolio",
        is_set: |c| c.attack.falsify_portfolio.is_some(),
    },
    PromotedKey {
        path: "attack.falsify_portfolio_secs",
        is_set: |c| c.attack.falsify_portfolio_secs.is_some(),
    },
    PromotedKey {
        path: "attack.flat_surrogate_max_blocks",
        is_set: |c| c.attack.flat_surrogate_max_blocks.is_some(),
    },
    PromotedKey {
        path: "attack.flat_surrogate_transfer",
        is_set: |c| c.attack.flat_surrogate_transfer.is_some(),
    },
    // #graph-free-candidates: a diagnostic INSTRUMENT, pinned here so no
    // shipped yaml can arm a side file that costs wall clock on a scored run.
    PromotedKey {
        path: "attack.graph_free_candidate_sidefile",
        is_set: |c| c.attack.graph_free_candidate_sidefile.is_some(),
    },
    PromotedKey {
        path: "attack.pre_softmax_objective",
        is_set: |c| c.attack.pre_softmax_objective.is_some(),
    },
    PromotedKey {
        path: "attack.post_bab_restarts",
        is_set: |c| c.attack.post_bab_restarts.is_some(),
    },
    PromotedKey {
        path: "attack.lane_budget_frame",
        is_set: |c| c.attack.lane_budget_frame.is_some(),
    },
    PromotedKey {
        path: "attack.lane_budget_allocator",
        is_set: |c| c.attack.lane_budget_allocator.is_some(),
    },
    PromotedKey {
        path: "attack.lane_value_scheduler",
        is_set: |c| c.attack.lane_value_scheduler.is_some(),
    },
    PromotedKey {
        path: "attack.bnn_sign_space_minimal_move",
        is_set: |c| c.attack.bnn_sign_space_minimal_move.is_some(),
    },
    PromotedKey {
        path: "attack.bnn_sign_space_trust_region",
        is_set: |c| c.attack.bnn_sign_space_trust_region.is_some(),
    },
    PromotedKey {
        path: "attack.pgd_exact_batched",
        is_set: |c| c.attack.pgd_exact_batched.is_some(),
    },
    PromotedKey {
        path: "attack.pgd_fast_grad",
        is_set: |c| c.attack.pgd_fast_grad.is_some(),
    },
    PromotedKey {
        path: "attack.pgd_vjp_batch",
        is_set: |c| c.attack.pgd_vjp_batch.is_some(),
    },
    PromotedKey {
        path: "attack.pgd_vjp_k",
        is_set: |c| c.attack.pgd_vjp_k.is_some(),
    },
    PromotedKey {
        path: "attack.pgd_legacy_lanes",
        is_set: |c| c.attack.pgd_legacy_lanes.is_some(),
    },
    PromotedKey {
        path: "attack.pgd_odi_steps",
        is_set: |c| c.attack.pgd_odi_steps.is_some(),
    },
    PromotedKey {
        path: "attack.pgd_odi_eta",
        is_set: |c| c.attack.pgd_odi_eta.is_some(),
    },
    PromotedKey {
        path: "attack.pgd_vjp_steps",
        is_set: |c| c.attack.pgd_vjp_steps.is_some(),
    },
    PromotedKey {
        path: "attack.pgd_lane_window",
        is_set: |c| c.attack.pgd_lane_window.is_some(),
    },
    PromotedKey {
        path: "attack.pgd_exploit",
        is_set: |c| c.attack.pgd_exploit.is_some(),
    },
    PromotedKey {
        path: "attack.pgd_hinge_tau",
        is_set: |c| c.attack.pgd_hinge_tau.is_some(),
    },
    // #disj-odi SHIPS DARK: ODI + dangerous-class focus lanes in the batched
    // disjunctive exact-VJP driver. Graduating either key needs the yaml-route
    // arrival proof (`[pgd-vjp-disj] disj-odi schema=v1 odi=1` with
    // `odi_vjp_steps` > 0, zero `NY_*`), the tinyimagenet idx_1831/idx_5529
    // conversion A/B with x86-rig-valid witnesses and a both-year moat, in the
    // commit that arms a shipped yaml.
    PromotedKey {
        path: "attack.pgd_disj_odi",
        is_set: |c| c.attack.pgd_disj_odi.is_some(),
    },
    PromotedKey {
        path: "attack.pgd_disj_focus_lanes",
        is_set: |c| c.attack.pgd_disj_focus_lanes.is_some(),
    },
    // Root-phase ownership lattice.
    PromotedKey {
        path: "bab.root_phase_resident_crown",
        is_set: |c| c.bab.root_phase_resident_crown.is_some(),
    },
    PromotedKey {
        path: "bab.root_wide_demanded_interm_crown",
        is_set: |c| c.bab.root_wide_demanded_interm_crown.is_some(),
    },
    PromotedKey {
        path: "bab.root_cpu_parallel_interm_crown",
        is_set: |c| c.bab.root_cpu_parallel_interm_crown.is_some(),
    },
    // Comprehensive-sweep sizing.
    PromotedKey {
        path: "bab.root_comprehensive_gpu_interm_rows",
        is_set: |c| c.bab.root_comprehensive_gpu_interm_rows.is_some(),
    },
    PromotedKey {
        path: "bab.root_comprehensive_gpu_interm_secs",
        is_set: |c| c.bab.root_comprehensive_gpu_interm_secs.is_some(),
    },
    PromotedKey {
        path: "bab.root_objective_directed_rows",
        is_set: |c| c.bab.root_objective_directed_rows.is_some(),
    },
    PromotedKey {
        path: "bab.cpu_compute_engine",
        is_set: |c| c.bab.cpu_compute_engine.is_some(),
    },
    // Batched intermediate refinement.
    PromotedKey {
        path: "bab.root_joint_interm_alpha",
        is_set: |c| c.bab.root_joint_interm_alpha.is_some(),
    },
    PromotedKey {
        path: "bab.root_joint_interm_alpha_deadline_ascent",
        is_set: |c| c.bab.root_joint_interm_alpha_deadline_ascent.is_some(),
    },
    PromotedKey {
        path: "bab.root_joint_interm_alpha_secs",
        is_set: |c| c.bab.root_joint_interm_alpha_secs.is_some(),
    },
    PromotedKey {
        path: "bab.root_joint_interm_alpha_iters",
        is_set: |c| c.bab.root_joint_interm_alpha_iters.is_some(),
    },
    PromotedKey {
        path: "bab.root_joint_interm_alpha_lr",
        is_set: |c| c.bab.root_joint_interm_alpha_lr.is_some(),
    },
    PromotedKey {
        path: "bab.root_joint_interm_alpha_max_sel",
        is_set: |c| c.bab.root_joint_interm_alpha_max_sel.is_some(),
    },
    PromotedKey {
        path: "bab.root_joint_interm_alpha_frozen_stop",
        is_set: |c| c.bab.root_joint_interm_alpha_frozen_stop.is_some(),
    },
    PromotedKey {
        path: "bab.root_joint_interm_alpha_pure_chain",
        is_set: |c| c.bab.root_joint_interm_alpha_pure_chain.is_some(),
    },
    PromotedKey {
        path: "bab.bn_gpu_extract",
        is_set: |c| c.bab.bn_gpu_extract.is_some(),
    },
    PromotedKey {
        path: "bab.root_joint_interm_alpha_layers",
        is_set: |c| c.bab.root_joint_interm_alpha_layers.is_some(),
    },
    PromotedKey {
        path: "bab.root_joint_min_remaining_secs",
        is_set: |c| c.bab.root_joint_min_remaining_secs.is_some(),
    },
    PromotedKey {
        path: "bab.root_joint_interm_alpha_max_dim",
        is_set: |c| c.bab.root_joint_interm_alpha_max_dim.is_some(),
    },
    PromotedKey {
        path: "bab.phase_budget.post_bab_reserve_secs",
        is_set: |c| c.bab.phase_budget.post_bab_reserve_secs.is_some(),
    },
    PromotedKey {
        path: "bab.clip_host_mean_la",
        is_set: |c| c.bab.clip_host_mean_la.is_some(),
    },
    PromotedKey {
        path: "bab.clip_interm_certified",
        is_set: |c| c.bab.clip_interm_certified.is_some(),
    },
    // Wide multi-objective lane.
    PromotedKey {
        path: "bab.mo_gpu_chunk_deadline",
        is_set: |c| c.bab.mo_gpu_chunk_deadline.is_some(),
    },
    PromotedKey {
        path: "bab.mo_alpha_heuristic",
        is_set: |c| c.bab.mo_alpha_heuristic.is_some(),
    },
    PromotedKey {
        path: "bab.mo_beta_spsa",
        is_set: |c| c.bab.mo_beta_spsa.is_some(),
    },
    PromotedKey {
        path: "bab.mo_alpha_step",
        is_set: |c| c.bab.mo_alpha_step.is_some(),
    },
    PromotedKey {
        path: "bab.kfsb_sim_share",
        is_set: |c| c.bab.kfsb_sim_share.is_some(),
    },
    // Complete margin-row search surface.
    PromotedKey {
        path: "margin_row.lru",
        is_set: |c| c.margin_row.lru.is_some(),
    },
    PromotedKey {
        path: "margin_row.parallel",
        is_set: |c| c.margin_row.parallel.is_some(),
    },
    PromotedKey {
        path: "margin_row.gpu",
        is_set: |c| c.margin_row.gpu.is_some(),
    },
    PromotedKey {
        path: "margin_row.gpu_batch",
        is_set: |c| c.margin_row.gpu_batch.is_some(),
    },
    PromotedKey {
        path: "margin_row.gpu_backward",
        is_set: |c| c.margin_row.gpu_backward.is_some(),
    },
    // `margin_row.alpha_opt` was REMOVED from this list on 2026-08-25, by the
    // protocol this test's own failure message states: arming a promoted key
    // needs its own commit, its own measurement, and its own removal here.
    // Measurement: the results section of
    // `docs/ALPHA_OPT_MARGIN_ROW_ROOT_2026-08-18.md` — delivery proven with no
    // `NY_*` set (`iters=20`, +0.0225 published root), idx_9502 unsat 3/3 with
    // the disarmed control landing on the exact unarmed baseline and NOT
    // proving, moat unsat 9/9, and a wall-hugger sweep of every banked unsat
    // row whose headroom is under this mechanism's cost. cifar100_2024 sets it;
    // every other shipped preset stays pinned silent by
    // `margin_row_wall_preset_field_diff_is_deliberate`, and the five TUNING
    // keys below stay listed, so the armed configuration is the measured
    // default rather than a hand-tuned point.
    PromotedKey {
        path: "margin_row.alpha_iters",
        is_set: |c| c.margin_row.alpha_iters.is_some(),
    },
    // `margin_row.alpha_secs` and `margin_row.alpha_cols` LEFT this list
    // 2026-09-22, each armed on its own measurement:
    // - `alpha_cols: 1` on cifar100 (commit 6b52dcdc5) and tinyimagenet
    //   (c8230d7f5): the ascent's objective is the MIN column while its
    //   gradient was a MEAN over 8, so 1 column aligns the two. Tinyimagenet
    //   marginal rows convert 7/10 armed vs 1/10 shipped (p=0.0099).
    // - `alpha_secs: 35.0` on tinyimagenet only (c8230d7f5), the cap its
    //   ascent saturates at the 20 s default.
    PromotedKey {
        path: "margin_row.alpha_lr",
        is_set: |c| c.margin_row.alpha_lr.is_some(),
    },
    PromotedKey {
        path: "margin_row.alpha_guard",
        is_set: |c| c.margin_row.alpha_guard.is_some(),
    },
    // #spec-alpha SHIPS DARK: five typed keys, armed by no yaml. This is
    // deliberately `tableau_f32`'s old state -- promoted so a measurement can
    // reach it through the competition route, and unarmed until the arrival
    // proof (armed yaml, zero NY_*, `[spec-alpha] armed:` on the row's stderr)
    // and the moat run have both been done. Graduating a row here without that
    // evidence is the failure mode this manifest exists to catch.
    PromotedKey {
        path: "margin_row.spec_alpha",
        is_set: |c| c.margin_row.spec_alpha.is_some(),
    },
    PromotedKey {
        path: "margin_row.spec_alpha_width",
        is_set: |c| c.margin_row.spec_alpha_width.is_some(),
    },
    PromotedKey {
        path: "margin_row.spec_alpha_cands",
        is_set: |c| c.margin_row.spec_alpha_cands.is_some(),
    },
    PromotedKey {
        path: "margin_row.spec_alpha_rounds",
        is_set: |c| c.margin_row.spec_alpha_rounds.is_some(),
    },
    PromotedKey {
        path: "margin_row.spec_alpha_entries",
        is_set: |c| c.margin_row.spec_alpha_entries.is_some(),
    },
    PromotedKey {
        path: "margin_row.clip",
        is_set: |c| c.margin_row.clip.is_some(),
    },
    PromotedKey {
        path: "margin_row.clip_head",
        is_set: |c| c.margin_row.clip_head.is_some(),
    },
    // margin_row.clip_interm and margin_row.clip_rows GRADUATED 2026-08-31:
    // 048716cc6 arms both in configs/vnncomp25/cifar100_2024.yaml (the idx_1798
    // clip promotion), so they are no longer promoted-but-unarmed.
    PromotedKey {
        path: "margin_row.clip_topk",
        is_set: |c| c.margin_row.clip_topk.is_some(),
    },
    PromotedKey {
        path: "margin_row.closure_thresh",
        is_set: |c| c.margin_row.closure_thresh.is_some(),
    },
    PromotedKey {
        path: "margin_row.closure_warmup",
        is_set: |c| c.margin_row.closure_warmup.is_some(),
    },
    PromotedKey {
        path: "margin_row.closure_window",
        is_set: |c| c.margin_row.closure_window.is_some(),
    },
    PromotedKey {
        path: "margin_row.bi_chunk",
        is_set: |c| c.margin_row.bi_chunk.is_some(),
    },
    PromotedKey {
        path: "margin_row.bi_chunk_secs",
        is_set: |c| c.margin_row.bi_chunk_secs.is_some(),
    },
    PromotedKey {
        path: "margin_row.bi_topk",
        is_set: |c| c.margin_row.bi_topk.is_some(),
    },
    // margin_row.interm_alpha REMOVED from this pin 2026-08-30: armed in
    // configs/vnncomp25/cifar100_2024.yaml by e2186f8a4 (the measured
    // conversion pair, idx_5242 receipts in that commit).
    PromotedKey {
        path: "margin_row.beta",
        is_set: |c| c.margin_row.beta.is_some(),
    },
    PromotedKey {
        path: "margin_row.beta_eta",
        is_set: |c| c.margin_row.beta_eta.is_some(),
    },
    PromotedKey {
        path: "margin_row.beta_iters",
        is_set: |c| c.margin_row.beta_iters.is_some(),
    },
    PromotedKey {
        path: "margin_row.beta_per_column",
        is_set: |c| c.margin_row.beta_per_column.is_some(),
    },
    PromotedKey {
        path: "margin_row.beta_columns",
        is_set: |c| c.margin_row.beta_columns.is_some(),
    },
    PromotedKey {
        path: "margin_row.parallel_prebuild",
        is_set: |c| c.margin_row.parallel_prebuild.is_some(),
    },
    // The scored lane consumes this complete family directly from its frozen
    // LeverSet. Keep every shipped-year preset dark until a sealed measurement
    // deliberately removes a key from this all-year guard.
    PromotedKey {
        path: "margin_row.domain_alpha",
        is_set: |c| c.margin_row.domain_alpha.is_some(),
    },
    PromotedKey {
        path: "margin_row.domain_alpha_iters",
        is_set: |c| c.margin_row.domain_alpha_iters.is_some(),
    },
    PromotedKey {
        path: "margin_row.domain_alpha_eta",
        is_set: |c| c.margin_row.domain_alpha_eta.is_some(),
    },
    PromotedKey {
        path: "margin_row.domain_alpha_min_depth",
        is_set: |c| c.margin_row.domain_alpha_min_depth.is_some(),
    },
    PromotedKey {
        path: "margin_row.domain_alpha_stride",
        is_set: |c| c.margin_row.domain_alpha_stride.is_some(),
    },
    PromotedKey {
        path: "margin_row.domain_alpha_gap",
        is_set: |c| c.margin_row.domain_alpha_gap.is_some(),
    },
    PromotedKey {
        path: "margin_row.beta_lambda",
        is_set: |c| c.margin_row.beta_lambda.is_some(),
    },
    PromotedKey {
        path: "margin_row.beta_polyak",
        is_set: |c| c.margin_row.beta_polyak.is_some(),
    },
    PromotedKey {
        path: "margin_row.beta_heads",
        is_set: |c| c.margin_row.beta_heads.is_some(),
    },
    PromotedKey {
        path: "margin_row.rebake",
        is_set: |c| c.margin_row.rebake.is_some(),
    },
    PromotedKey {
        path: "margin_row.rebake_rounds",
        is_set: |c| c.margin_row.rebake_rounds.is_some(),
    },
    PromotedKey {
        path: "margin_row.rebake_secs",
        is_set: |c| c.margin_row.rebake_secs.is_some(),
    },
    // #bi-window / #joint-refresh (cct plan P5): promoted DARK, armed by no
    // shipped yaml until a measured A/B plus a moat say otherwise.
    PromotedKey {
        path: "margin_row.bi_window",
        is_set: |c| c.margin_row.bi_window.is_some(),
    },
    PromotedKey {
        path: "margin_row.joint_refresh",
        is_set: |c| c.margin_row.joint_refresh.is_some(),
    },
    PromotedKey {
        path: "margin_row.joint_refresh_secs",
        is_set: |c| c.margin_row.joint_refresh_secs.is_some(),
    },
    // Complete head-phase covers remain dark until production source tests
    // and a scored A/B justify an explicit shipped arming.
    PromotedKey {
        path: "margin_row.head_groups",
        is_set: |c| c.margin_row.head_groups.is_some(),
    },
    PromotedKey {
        path: "margin_row.head_directional_cuts",
        is_set: |c| c.margin_row.head_directional_cuts.is_some(),
    },
    PromotedKey {
        path: "margin_row.head_phase_cover",
        is_set: |c| c.margin_row.head_phase_cover.is_some(),
    },
    // #root-certify-once (Phase-0.5 root alpha, search-untrusted /
    // certify-once): promoted DARK, armed by no shipped yaml until a measured
    // A/B plus a moat say otherwise.
    PromotedKey {
        path: "margin_row.root_certify_once",
        is_set: |c| c.margin_row.root_certify_once.is_some(),
    },
    PromotedKey {
        path: "margin_row.root_certify_once_iters",
        is_set: |c| c.margin_row.root_certify_once_iters.is_some(),
    },
    PromotedKey {
        path: "margin_row.root_certify_once_refresh",
        is_set: |c| c.margin_row.root_certify_once_refresh.is_some(),
    },
    PromotedKey {
        path: "margin_row.root_certify_once_secs",
        is_set: |c| c.margin_row.root_certify_once_secs.is_some(),
    },
    // #bbe-joint-root (ENGINE_PROGRAM M5a): the search selector of
    // #root-certify-once, promoted DARK (`ascent`, bit-identical). Graduating
    // `joint_f32` needs the yaml-route arrival proof (`[bbe] phase=joint_root`
    // with zero `NY_*`), the S3 measurement, a ctl-vs-armed x3 and a both-year
    // moat, in the commit that arms a shipped yaml.
    PromotedKey {
        path: "margin_row.root_search",
        is_set: |c| c.margin_row.root_search.is_some(),
    },
    // #root-box-source (ENGINE_PROGRAM M5b): the root's trunk-box source,
    // promoted DARK (`tableau`, bit-identical). Graduating `patch_backward`
    // needs the yaml-route arrival proof (`[root-box] source=patch_backward`
    // with zero `NY_*`), the root-wall measurement against the kill line, a
    // ctl-vs-armed x3 and a both-year moat, in the commit that arms a yaml.
    PromotedKey {
        path: "margin_row.root_box_source",
        is_set: |c| c.margin_row.root_box_source.is_some(),
    },
    // margin_row.bi_gpu RETIRED 2026-09-29 (ENGINE_PROGRAM R2: its cuBLAS
    // products entered certified boxes): no longer a promotion; preset
    // validation refuses the key (`retired_gpu_keys_are_refused_at_validation`).
    // #bi-patch SHIPS DARK: the backward-interm / joint-refresh prefix pairs as
    // per-column receptive-field walks. Graduating it needs the yaml-route
    // arrival proof (`[bi-patch] armed`, `plan ... refused=-` and
    // `[bi-patch] done: pairs=` > 0 with zero `NY_*`), the cct wall A/B and a
    // moat, in the commit that arms a shipped yaml.
    PromotedKey {
        path: "margin_row.bi_patch",
        is_set: |c| c.margin_row.bi_patch.is_some(),
    },
    PromotedKey {
        path: "margin_row.bi_patch_ratio",
        is_set: |c| c.margin_row.bi_patch_ratio.is_some(),
    },
    PromotedKey {
        path: "margin_row.bi_patch_mb",
        is_set: |c| c.margin_row.bi_patch_mb.is_some(),
    },
    // #mr-ibp-seed SHIPS DARK: certified IBP boxes intersected shrink-only with
    // the root tableau's trunk boxes. Graduating it (cct only) needs the
    // yaml-route arrival proof (`[mr-ibp-seed] layer=` lines with zero `NY_*`),
    // the GAP conversion x3, the 8090/8397 negative controls and the cct moat,
    // in the commit that arms the yaml.
    PromotedKey {
        path: "margin_row.ibp_seed",
        is_set: |c| c.margin_row.ibp_seed.is_some(),
    },
    PromotedKey {
        path: "margin_row.epoch_tier0",
        is_set: |c| c.margin_row.epoch_tier0.is_some(),
    },
    PromotedKey {
        path: "margin_row.epoch_universe",
        is_set: |c| c.margin_row.epoch_universe.is_some(),
    },
    PromotedKey {
        path: "margin_row.epoch_retain",
        is_set: |c| c.margin_row.epoch_retain.is_some(),
    },
    // The scored lane consumes these directly from its frozen LeverSet. They
    // are pinned here with the compatibility-cell routes above, but must not
    // regain the old path-reread/setter bridge.
    PromotedKey {
        path: "margin_row.epoch_depth",
        is_set: |c| c.margin_row.epoch_depth.is_some(),
    },
    PromotedKey {
        path: "margin_row.epoch_attempts",
        is_set: |c| c.margin_row.epoch_attempts.is_some(),
    },
    PromotedKey {
        path: "margin_row.epoch_parent_topk",
        is_set: |c| c.margin_row.epoch_parent_topk.is_some(),
    },
    // Route selections.
    PromotedKey {
        path: "bab.graph_mip_leaf_sat",
        is_set: |c| c.bab.graph_mip_leaf_sat.is_some(),
    },
    PromotedKey {
        path: "bab.true_grad_gpu_replay",
        is_set: |c| c.bab.true_grad_gpu_replay.is_some(),
    },
    PromotedKey {
        path: "bab.safenlp_mip_cycle",
        is_set: |c| c.bab.safenlp_mip_cycle.is_some(),
    },
    PromotedKey {
        path: "bab.nn4sys_ctx_group",
        is_set: |c| c.bab.nn4sys_ctx_group.is_some(),
    },
    // #metaroom-mip-governor (2026-09-11): promoted at birth, armed by no
    // shipped yaml. The metaroom CANARY config that arms them lives outside the
    // tree, so production stays dark and this pin proves it.
    PromotedKey {
        path: "bab.mip.leaf_undecided_streak_cap",
        is_set: |c| c.bab.mip.leaf_undecided_streak_cap.is_some(),
    },
    PromotedKey {
        path: "bab.mip.escalation_skips_certified_clauses",
        is_set: |c| c.bab.mip.escalation_skips_certified_clauses.is_some(),
    },
    PromotedKey {
        path: "cz_hybrid.max_budget_fraction",
        is_set: |c| c.cz_hybrid.max_budget_fraction.is_some(),
    },
    // #nn4sys-1d-cover. Promoted as a SET: the master alone would deliver a
    // reachable but inert mechanism, because the cover's pass is
    // all-or-nothing and the depth decides whether it can finish
    // (crates/ny-levers/src/decls/nn4sys_1d_cover.rs).
    PromotedKey {
        path: "bab.nn4sys_1d_cover",
        is_set: |c| c.bab.nn4sys_1d_cover.is_some(),
    },
    PromotedKey {
        path: "bab.nn4sys_1d_cover_depth",
        is_set: |c| c.bab.nn4sys_1d_cover_depth.is_some(),
    },
    PromotedKey {
        path: "bab.nn4sys_1d_cover_slice",
        is_set: |c| c.bab.nn4sys_1d_cover_slice.is_some(),
    },
    // Batch 2: the ny-gpu CROWN lane.
    PromotedKey {
        path: "bab.bab_resnet_wide",
        is_set: |c| c.bab.bab_resnet_wide.is_some(),
    },
    PromotedKey {
        path: "bab.bab_resnet_wide_subgroup",
        is_set: |c| c.bab.bab_resnet_wide_subgroup.is_some(),
    },
    PromotedKey {
        path: "bab.sweep_class_rows",
        is_set: |c| c.bab.sweep_class_rows.is_some(),
    },
    PromotedKey {
        path: "bab.sweep_class_mib",
        is_set: |c| c.bab.sweep_class_mib.is_some(),
    },
    PromotedKey {
        path: "bab.resnet_err_concretize",
        is_set: |c| c.bab.resnet_err_concretize.is_some(),
    },
    PromotedKey {
        path: "bab.resnet_err_concretize_fine",
        is_set: |c| c.bab.resnet_err_concretize_fine.is_some(),
    },
    // Batch 2: the `read_raw` predicate gates.
    PromotedKey {
        path: "bab.seq_weighted_budget",
        is_set: |c| c.bab.seq_weighted_budget.is_some(),
    },
    PromotedKey {
        path: "bab.cgan_stacked_backward",
        is_set: |c| c.bab.cgan_stacked_backward.is_some(),
    },
    PromotedKey {
        path: "bab.cgan_polish",
        is_set: |c| c.bab.cgan_polish.is_some(),
    },
    PromotedKey {
        path: "bab.crown_partial_finite_expiry",
        is_set: |c| c.bab.crown_partial_finite_expiry.is_some(),
    },
    PromotedKey {
        path: "bab.cgan_stacked_budget_mb",
        is_set: |c| c.bab.cgan_stacked_budget_mb.is_some(),
    },
    PromotedKey {
        path: "bab.conv_ibp_blocked",
        is_set: |c| c.bab.conv_ibp_blocked.is_some(),
    },
    PromotedKey {
        path: "bab.patches_finite_expiry",
        is_set: |c| c.bab.patches_finite_expiry.is_some(),
    },
    PromotedKey {
        path: "bab.no_walk_record_admission",
        is_set: |c| c.bab.no_walk_record_admission.is_some(),
    },
    PromotedKey {
        path: "bab.pollable_faer",
        is_set: |c| c.bab.pollable_faer.is_some(),
    },
    // #mr-errlane (throughput plan B3) GRADUATED 2026-09-26: armed as a pair with
    // margin_row.domain_stack in cifar100_2024 + tinyimagenet_2024 on the
    // scratchpad/mrab A/B (idx_2050/idx_4921 timeout 3/3 -> unsat 3/3); removed
    // from this list in its arming commit, as the pin requires.
    // margin_row.gpu_conv RETIRED 2026-09-29 (ENGINE_PROGRAM R1 kill line: lost
    // c2050 and c1590; R2): no longer a promotion; preset validation refuses the
    // key (`retired_gpu_keys_are_refused_at_validation`).
    // #frozen-dual-bab + #breakpoint-dual: the parent and refinement keys leave
    // this list for user-directed CIFAR100/TinyImageNet default activation.
    // The tier uses bounded automatic mode and the unchanged certifier. Its
    // official-budget benefit and full both-year comparison remain unmeasured.
    // These two manual tuning keys remain absent from shipped presets.
    PromotedKey {
        path: "margin_row.frozen_dual_bab_node_cap",
        is_set: |c| c.margin_row.frozen_dual_bab_node_cap.is_some(),
    },
    PromotedKey {
        path: "margin_row.frozen_dual_bab_secs",
        is_set: |c| c.margin_row.frozen_dual_bab_secs.is_some(),
    },
    // #fd-shared-state (ENGINE_PROGRAM M3): the tier's state-build mode,
    // promoted DARK (`per_class`, bit-identical). Graduating `shared` needs the
    // yaml-route arrival proof (`[frozen-dual] trunk state=shared lambda=` with
    // zero `NY_*`), the root-loss / classes-built measurement, the tier-engaged
    // sat negative controls and a both-year moat, in the commit that arms a
    // shipped yaml.
    PromotedKey {
        path: "margin_row.frozen_dual_bab_state",
        is_set: |c| c.margin_row.frozen_dual_bab_state.is_some(),
    },
    // #wide-fd-search (ENGINE_PROGRAM M4): the tier's search selector,
    // promoted DARK (`f64`, bit-identical). Graduating `cpu_f32` / `cuda` /
    // `auto` needs the yaml-route arrival proof (`[bbe] armed=1 backend=
    // probe=ok` with zero `NY_*`), the certified nodes/s (S1), the
    // refuted/claimed rate, a ctl-vs-armed x3 and a both-year moat, in the
    // commit that arms a shipped yaml.
    PromotedKey {
        path: "margin_row.frozen_dual_bab_search",
        is_set: |c| c.margin_row.frozen_dual_bab_search.is_some(),
    },
    // #wave-beta-bab (ENGINE_PROGRAM M6): the outer frame's tree engine and
    // its two knobs, promoted DARK (`legacy`, bit-identical). Graduating
    // `wave` needs the yaml-route arrival proof (`[bbe] phase=wave armed=1
    // backend=` with zero `NY_*`), the certified closes/s, the refuted/claimed
    // rate, a ctl-vs-armed x3 with 0 banked rows lost and a both-year moat, in
    // the commit that arms a shipped yaml.
    PromotedKey {
        path: "margin_row.tree_mode",
        is_set: |c| c.margin_row.tree_mode.is_some(),
    },
    PromotedKey {
        path: "margin_row.tree_wave_batch",
        is_set: |c| c.margin_row.tree_wave_batch.is_some(),
    },
    PromotedKey {
        path: "margin_row.tree_wave_cert_threads",
        is_set: |c| c.margin_row.tree_wave_cert_threads.is_some(),
    },
    // #router-r1 (ROUTER_PROGRAM §2.5): the runtime router and optimizer
    // modes are promoted DARK and no category yaml may arm them. Rollout is a
    // compiled-default move after a both-year moat, never a per-category yaml
    // line; measurement arms `shadow` in scratch configs only.
    PromotedKey {
        path: "router.mode",
        is_set: |c| c.router.mode.is_some(),
    },
    PromotedKey {
        path: "optimizer.mode",
        is_set: |c| c.optimizer.mode.is_some(),
    },
    // #router-r2: the seven acting-rule ablation switches. Promoted DARK
    // (they participate only under `router.mode: on`) and pinned here so no
    // category yaml carries them.
    PromotedKey {
        path: "router.rules.admission",
        is_set: |c| c.router.rules.admission.is_some(),
    },
    PromotedKey {
        path: "router.rules.contracts",
        is_set: |c| c.router.rules.contracts.is_some(),
    },
    PromotedKey {
        path: "router.rules.cap_authority",
        is_set: |c| c.router.rules.cap_authority.is_some(),
    },
    PromotedKey {
        path: "router.rules.terminal",
        is_set: |c| c.router.rules.terminal.is_some(),
    },
    PromotedKey {
        path: "router.rules.hand_on",
        is_set: |c| c.router.rules.hand_on.is_some(),
    },
    PromotedKey {
        path: "router.rules.publish_grace",
        is_set: |c| c.router.rules.publish_grace.is_some(),
    },
    PromotedKey {
        path: "router.rules.derived_caps",
        is_set: |c| c.router.rules.derived_caps.is_some(),
    },
    // #router-r3: the four O-STOP optimizer-rule ablation switches. Promoted
    // DARK (they participate only under `optimizer.mode: on`) and pinned here
    // so no category yaml carries them.
    PromotedKey {
        path: "optimizer.rules.governor",
        is_set: |c| c.optimizer.rules.governor.is_some(),
    },
    PromotedKey {
        path: "optimizer.rules.ia_yield",
        is_set: |c| c.optimizer.rules.ia_yield.is_some(),
    },
    PromotedKey {
        path: "optimizer.rules.attack_defer",
        is_set: |c| c.optimizer.rules.attack_defer.is_some(),
    },
    PromotedKey {
        path: "optimizer.rules.root_admission",
        is_set: |c| c.optimizer.rules.root_admission.is_some(),
    },
    // #router-r6: the four O-TRIAL optimizer-rule ablation switches. Promoted
    // DARK (they participate only under `optimizer.mode: on`) and pinned here
    // so no category yaml carries them.
    PromotedKey {
        path: "optimizer.rules.trial_width",
        is_set: |c| c.optimizer.rules.trial_width.is_some(),
    },
    PromotedKey {
        path: "optimizer.rules.trial_lr_backoff",
        is_set: |c| c.optimizer.rules.trial_lr_backoff.is_some(),
    },
    PromotedKey {
        path: "optimizer.rules.trial_split_restart",
        is_set: |c| c.optimizer.rules.trial_split_restart.is_some(),
    },
    PromotedKey {
        path: "optimizer.rules.trial_attack_halving",
        is_set: |c| c.optimizer.rules.trial_attack_halving.is_some(),
    },
    // #router-r4: acting set 3's two ablation switches (CR-L, CR-F).
    // Promoted DARK (they participate only under `router.mode: on`) and
    // pinned here so no category yaml carries them.
    PromotedKey {
        path: "router.rules.lease",
        is_set: |c| c.router.rules.lease.is_some(),
    },
    PromotedKey {
        path: "router.rules.far_hand_on",
        is_set: |c| c.router.rules.far_hand_on.is_some(),
    },
    // #router-r5: acting set 4's two ablation switches (CR-RS, CR-G).
    // Promoted DARK (they participate only under `router.mode: on`) and
    // pinned here so no category yaml carries them.
    PromotedKey {
        path: "router.rules.route",
        is_set: |c| c.router.rules.route.is_some(),
    },
    PromotedKey {
        path: "router.rules.gap_contract",
        is_set: |c| c.router.rules.gap_contract.is_some(),
    },
    // #router-r7: the offline-prior switch. Promoted DARK (default false; it
    // is read only where the router runs) and pinned here so no category yaml
    // carries it.
    PromotedKey {
        path: "optimizer.priors",
        is_set: |c| c.optimizer.priors.is_some(),
    },
];

/// The ONE (yaml basename, promoted key) set allowed to arm.
///
/// `nn4sys_1dcover_canary.yaml` is unreachable by category resolution —
/// `preset_basename_candidates` yields only the lowercased category and its
/// year-stripped base, so nothing resolves to a `_canary` basename — and it
/// exists so the #nn4sys-1d-cover delivery route can be proved on the REAL
/// path with zero `NY_*` set. Arming it therefore cannot move a scored
/// verdict, which is the property the pin below actually protects.
///
/// This is an allowlist, not a skip: `every_canary_arming_is_live` asserts
/// each pair really is armed, so a stale entry fails instead of silently
/// widening the pin. The three #nn4sys-1d-cover keys are canary-only. The
/// separately authorized SafeNLP production defaults are the exact scored
/// exceptions listed below, restricted to vnncomp26.
const CANARY_ARMINGS: &[(&str, &str)] = &[
    ("nn4sys_1dcover_canary.yaml", "bab.nn4sys_1d_cover"),
    ("nn4sys_1dcover_canary.yaml", "bab.nn4sys_1d_cover_depth"),
    ("nn4sys_1dcover_canary.yaml", "bab.nn4sys_1d_cover_slice"),
    ("vnncomp26/safenlp_2024.yaml", "bab.safenlp_mip_cycle"),
    ("vnncomp26/safenlp_2024.yaml", "margin_row.head_phase_cover"),
    ("vnncomp26/safenlp_2024.yaml", "margin_row.head_groups"),
    (
        "vnncomp26/safenlp_2024.yaml",
        "margin_row.head_directional_cuts",
    ),
];

fn canary_arming_allowed(path: &str, key: &str) -> bool {
    CANARY_ARMINGS
        .iter()
        .any(|(file, allowed)| *allowed == key && Path::new(path).ends_with(file))
}

fn shipped_presets() -> Vec<(String, PresetConfig)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut out = Vec::new();
    for year in YEARS {
        let Ok(entries) = fs::read_dir(root.join(year)) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("preset directory entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                continue;
            }
            let text = fs::read_to_string(&path).expect("readable shipped preset");
            let preset: PresetConfig = serde_yaml::from_str(&text)
                .unwrap_or_else(|error| panic!("preset {} must parse: {error}", path.display()));
            out.push((path.display().to_string(), preset));
        }
    }
    out
}

/// PROMOTION IS NOT ARMING.
///
/// Every key in the 2026-08-23 batch must be absent from every shipped yaml.
/// While that holds, each promoted lever resolves through an EMPTY
/// `ny_levers::gate` cell, which is byte-identical to the pre-promotion
/// `ny_levers::read(&DECL)` — so landing the batch cannot have changed a single
/// scored verdict, and any future change of one is a measured, reviewable
/// decision rather than a side effect of plumbing.
#[test]
fn no_shipped_preset_arms_a_promoted_lever_delivery_key() {
    let presets = shipped_presets();
    assert!(
        presets.len() >= 20,
        "expected the shipped preset corpus, found {} files — a silent read failure \
         would make this pin vacuous",
        presets.len()
    );

    let mut armed = Vec::new();
    for (path, preset) in &presets {
        for key in PROMOTED {
            if (key.is_set)(preset) && !canary_arming_allowed(path, key.path) {
                armed.push(format!("{} sets {}", path, key.path));
            }
        }
    }

    assert!(
        armed.is_empty(),
        "a #lever-reachability delivery key is SET in a shipped preset:\n  {}\n\n\
         Promotion made these mechanisms deliverable; it did not qualify any of \
         them. Arming one needs its own commit, its own measurement, and its own \
         removal from `PROMOTED` in this file.",
        armed.join("\n  "),
    );
}

#[test]
fn no_shipped_preset_preselects_a_bnn_attack_budget_treatment() {
    let selected = shipped_presets()
        .into_iter()
        .filter_map(|(path, preset)| {
            (preset.attack.bnn_attack_cap_secs.is_some()
                || preset.attack.bnn_attack_fraction.is_some())
            .then_some(path)
        })
        .collect::<Vec<_>>();
    assert!(
        selected.is_empty(),
        "BNN budget governance must preserve the measured 240s/0.5 defaults until a \
         banked search result deliberately changes the traffic preset; set by {selected:?}"
    );
}

/// The canary allowlist may not go stale or vacuous: every pair it names must
/// be a key that the named file really sets. A rename, a deleted canary, or a
/// key that quietly stopped being armed would otherwise leave a live hole in
/// the pin above with nothing to say so.
#[test]
fn every_canary_arming_is_live() {
    let presets = shipped_presets();
    let mut dead = Vec::new();
    for (file, key) in CANARY_ARMINGS {
        let found = presets.iter().any(|(path, preset)| {
            Path::new(path).ends_with(file)
                && PROMOTED
                    .iter()
                    .any(|promoted| promoted.path == *key && (promoted.is_set)(preset))
        });
        if !found {
            dead.push(format!("{file} no longer sets {key}"));
        }
    }
    assert!(
        dead.is_empty(),
        "the canary allowlist has gone stale, which silently widens \
         `no_shipped_preset_arms_a_promoted_lever_delivery_key`:\n  {}",
        dead.join("\n  "),
    );
}

/// The pin above is only worth having if a set key would actually trip it.
#[test]
fn the_delivery_pin_is_not_vacuous() {
    let mut preset = PresetConfig::default();
    assert!(
        !PROMOTED.iter().any(|k| (k.is_set)(&preset)),
        "a default preset sets nothing"
    );
    preset.bab.clip_interm_certified = Some(true);
    assert_eq!(
        PROMOTED
            .iter()
            .filter(|k| (k.is_set)(&preset))
            .map(|k| k.path)
            .collect::<Vec<_>>(),
        vec!["bab.clip_interm_certified"],
        "the probe must detect exactly the key that was set"
    );
}

/// Every promoted key must survive a real yaml round-trip under
/// `deny_unknown_fields`, and must land where `preset_gates::apply` looks for
/// it. A typo in the field name would otherwise make the whole delivery route
/// unreachable in a different way — a preset that names it would be REJECTED at
/// load, which is at least loud, but a key nobody can spell is not a route.
#[test]
fn every_promoted_key_round_trips_through_the_real_schema() {
    let yaml = r"
attack:
  falsify_portfolio: true
  falsify_portfolio_secs: 15
  flat_surrogate_max_blocks: 4096
  flat_surrogate_transfer: true
  graph_free_candidate_sidefile: true
  pre_softmax_objective: true
  post_bab_restarts: 3
  lane_budget_frame: true
  lane_budget_allocator: true
  lane_value_scheduler: true
  bnn_sign_space_minimal_move: true
  bnn_sign_space_trust_region: linf
  pgd_exact_batched: false
  pgd_fast_grad: false
  pgd_vjp_batch: true
  pgd_vjp_k: 52
  pgd_legacy_lanes: 20
  pgd_odi_steps: 4
  pgd_odi_eta: 0.5
  pgd_vjp_steps: 250
  pgd_lane_window: 50
  pgd_exploit: false
  pgd_hinge_tau: 0.0001
  pgd_disj_odi: true
  pgd_disj_focus_lanes: 32
bab:
  root_phase_resident_crown: true
  root_wide_demanded_interm_crown: true
  root_cpu_parallel_interm_crown: true
  root_comprehensive_gpu_interm_rows: 128
  root_comprehensive_gpu_interm_secs: 40
  root_objective_directed_rows: false
  cpu_compute_engine: false
  root_joint_interm_alpha: true
  root_joint_interm_alpha_deadline_ascent: true
  root_joint_interm_alpha_secs: 60
  root_joint_interm_alpha_iters: 12
  root_joint_interm_alpha_lr: 0.2
  root_joint_interm_alpha_max_sel: 8
  root_joint_interm_alpha_frozen_stop: true
  root_joint_interm_alpha_pure_chain: false
  bn_gpu_extract: true
  root_joint_interm_alpha_layers: Relu_7,Relu_11
  root_joint_min_remaining_secs: 120
  root_joint_interm_alpha_max_dim: 1024
  clip_host_mean_la: true
  clip_interm_certified: true
  mo_gpu_chunk_deadline: true
  mo_alpha_heuristic: true
  mo_beta_spsa: false
  mo_alpha_step: true
  kfsb_sim_share: 0.6
  graph_mip_leaf_sat: false
  true_grad_gpu_replay: false
  safenlp_mip_cycle: true
  nn4sys_ctx_group: true
  nn4sys_1d_cover: true
  nn4sys_1d_cover_depth: 24
  nn4sys_1d_cover_slice: 0.25
  bab_resnet_wide: false
  bab_resnet_wide_subgroup: true
  sweep_class_rows: 1024
  sweep_class_mib: 256
  resnet_err_concretize: true
  resnet_err_concretize_fine: true
  seq_weighted_budget: false
  cgan_stacked_backward: true
  cgan_stacked_budget_mb: 4096
  conv_ibp_blocked: seq
  patches_finite_expiry: false
  no_walk_record_admission: true
  pollable_faer: true
  cgan_polish: true
  crown_partial_finite_expiry: true
  mip:
    leaf_undecided_streak_cap: 2
    escalation_skips_certified_clauses: true
  phase_budget:
    post_bab_reserve_secs: 100
margin_row:
  lru: 128
  parallel: '8'
  gpu: true
  gpu_batch: true
  gpu_backward: true
  frozen_dual_bab: true
  frozen_dual_bab_node_cap: 250000
  frozen_dual_bab_secs: 20.0
  frozen_dual_refine_steps: 8
  frozen_dual_bab_state: shared
  frozen_dual_bab_search: cpu_f32
  tree_mode: wave
  tree_wave_batch: 512
  tree_wave_cert_threads: 6
  alpha_opt: true
  alpha_iters: 16
  alpha_secs: 40.0
  alpha_lr: 0.1
  alpha_cols: 8
  spec_alpha: true
  spec_alpha_width: 4
  spec_alpha_cands: 5
  spec_alpha_rounds: 2
  spec_alpha_entries: 4096
  alpha_guard: false
  clip: true
  clip_head: true
  clip_interm: true
  clip_rows: true
  clip_topk: 50
  closure_thresh: 0.5
  closure_warmup: 32
  closure_window: 24
  closure_max_tier: 2
  bi_chunk: 1024
  bi_chunk_secs: 2.0
  bi_topk: 4096
  interm_alpha: 2
  beta: true
  beta_eta: 0.25
  beta_iters: 4
  beta_per_column: true
  beta_columns: 16
  column_retire: true
  parallel_prebuild: true
  domain_alpha: true
  domain_alpha_iters: 4
  domain_alpha_eta: 0.125
  domain_alpha_min_depth: 4
  domain_alpha_stride: 2
  domain_alpha_gap: 0.05
  beta_lambda: 2.0
  beta_polyak: false
  beta_heads: false
  rebake: true
  rebake_rounds: 3
  rebake_secs: 60.0
  bi_window: true
  joint_refresh: 3
  joint_refresh_secs: 45.0
  head_groups: true
  head_directional_cuts: false
  head_phase_cover: 7
  root_certify_once: true
  root_certify_once_iters: 30
  root_certify_once_refresh: 5
  root_certify_once_secs: 25.0
  root_search: joint_f32
  root_box_source: patch_backward
  bi_patch: true
  bi_patch_ratio: 0.25
  bi_patch_mb: 64
  ibp_seed: true
  epoch_tier0: 8
  epoch_universe: 128
  epoch_retain: 256
  epoch_depth: 3
  epoch_attempts: 4
  epoch_parent_topk: 16
  err_envelope: per_channel
cz_hybrid:
  max_budget_fraction: 0.5
router:
  mode: shadow
  rules:
    admission: true
    contracts: false
    cap_authority: true
    terminal: false
    hand_on: true
    publish_grace: false
    derived_caps: true
    lease: true
    far_hand_on: false
    route: false
    gap_contract: true
optimizer:
  mode: 'on'
  rules:
    governor: true
    ia_yield: false
    attack_defer: true
    root_admission: false
    trial_width: true
    trial_lr_backoff: false
    trial_split_restart: true
    trial_attack_halving: false
  priors: true
";
    let preset: PresetConfig = serde_yaml::from_str(yaml).expect("promoted keys must parse");
    let unset: Vec<&str> = PROMOTED
        .iter()
        .filter(|k| !(k.is_set)(&preset))
        .map(|k| k.path)
        .collect();
    assert!(
        unset.is_empty(),
        "these promoted keys did not round-trip from yaml: {unset:?}"
    );

    // Spot-check the non-boolean shapes: a `u64` key that silently parsed as
    // something else would deliver a value nobody asked for.
    assert_eq!(preset.attack.falsify_portfolio_secs, Some(15));
    // #nn4sys-1d-cover: the two non-boolean keys carry VALUES, and a value
    // that silently parsed as something else would deliver a pass nobody
    // asked for — on an all-or-nothing lane that means publishing nothing.
    assert_eq!(preset.bab.nn4sys_1d_cover, Some(true));
    assert_eq!(preset.bab.nn4sys_1d_cover_depth, Some(24));
    assert_eq!(preset.bab.nn4sys_1d_cover_slice, Some(0.25));
    assert_eq!(preset.attack.flat_surrogate_max_blocks, Some(4096));
    assert_eq!(preset.attack.post_bab_restarts, Some(3));
    assert_eq!(preset.attack.flat_surrogate_transfer, Some(true));
    assert_eq!(preset.attack.graph_free_candidate_sidefile, Some(true));
    assert_eq!(preset.attack.pgd_exact_batched, Some(false));
    assert_eq!(preset.attack.pgd_fast_grad, Some(false));
    assert_eq!(preset.attack.pgd_vjp_batch, Some(true));
    assert_eq!(preset.attack.pgd_vjp_k, Some(52));
    assert_eq!(preset.attack.pgd_legacy_lanes, Some(20));
    assert_eq!(preset.attack.pgd_odi_steps, Some(4));
    assert_eq!(preset.attack.pgd_odi_eta, Some(0.5));
    assert_eq!(preset.margin_row.beta_columns, Some(16));
    assert_eq!(preset.margin_row.domain_alpha, Some(true));
    assert_eq!(preset.margin_row.domain_alpha_iters, Some(4));
    assert_eq!(preset.margin_row.domain_alpha_eta, Some(0.125));
    assert_eq!(preset.margin_row.domain_alpha_min_depth, Some(4));
    assert_eq!(preset.margin_row.domain_alpha_stride, Some(2));
    assert_eq!(preset.margin_row.domain_alpha_gap, Some(0.05));
    assert_eq!(preset.attack.pgd_vjp_steps, Some(250));
    assert_eq!(preset.attack.pgd_lane_window, Some(50));
    assert_eq!(preset.attack.pgd_exploit, Some(false));
    assert_eq!(preset.attack.pgd_hinge_tau, Some(0.0001));
    assert_eq!(preset.attack.pgd_disj_odi, Some(true));
    assert_eq!(preset.attack.pgd_disj_focus_lanes, Some(32));
    assert_eq!(preset.bab.root_comprehensive_gpu_interm_rows, Some(128));
    assert_eq!(preset.bab.root_comprehensive_gpu_interm_secs, Some(40));
    assert_eq!(preset.bab.cpu_compute_engine, Some(false));
    assert_eq!(preset.bab.root_joint_interm_alpha, Some(true));
    assert_eq!(
        preset.bab.root_joint_interm_alpha_deadline_ascent,
        Some(true)
    );
    assert_eq!(preset.bab.root_joint_interm_alpha_secs, Some(60));
    assert_eq!(preset.bab.root_joint_interm_alpha_iters, Some(12));
    assert_eq!(preset.bab.root_joint_interm_alpha_lr, Some(0.2));
    assert_eq!(preset.bab.root_joint_interm_alpha_max_sel, Some(8));
    assert_eq!(preset.bab.root_joint_interm_alpha_frozen_stop, Some(true));
    assert_eq!(preset.bab.root_joint_interm_alpha_pure_chain, Some(false));
    assert_eq!(preset.bab.bn_gpu_extract, Some(true));
    assert_eq!(
        preset.bab.root_joint_interm_alpha_layers.as_deref(),
        Some("Relu_7,Relu_11")
    );
    assert_eq!(preset.bab.root_joint_min_remaining_secs, Some(120));
    assert_eq!(preset.bab.root_joint_interm_alpha_max_dim, Some(1024));
    assert_eq!(preset.bab.phase_budget.post_bab_reserve_secs, Some(100));
    assert_eq!(preset.bab.kfsb_sim_share, Some(0.6));
    assert_eq!(preset.bab.mo_beta_spsa, Some(false));
    assert_eq!(preset.bab.mo_alpha_step, Some(true));
    assert_eq!(
        preset.attack.bnn_sign_space_trust_region.as_deref(),
        Some("linf")
    );
    assert_eq!(preset.bab.cgan_stacked_budget_mb, Some(4096));
    assert_eq!(preset.bab.conv_ibp_blocked.as_deref(), Some("seq"));
    assert_eq!(preset.margin_row.lru, Some(128));
    assert_eq!(preset.margin_row.parallel.as_deref(), Some("8"));
    assert_eq!(preset.margin_row.alpha_iters, Some(16));
    assert_eq!(preset.margin_row.alpha_secs, Some(40.0));
    assert_eq!(preset.margin_row.alpha_lr, Some(0.1));
    assert_eq!(preset.margin_row.alpha_cols, Some(8));
    assert_eq!(preset.margin_row.alpha_guard, Some(false));
    assert_eq!(preset.margin_row.spec_alpha, Some(true));
    assert_eq!(preset.margin_row.spec_alpha_width, Some(4));
    assert_eq!(preset.margin_row.spec_alpha_cands, Some(5));
    assert_eq!(preset.margin_row.spec_alpha_rounds, Some(2));
    assert_eq!(preset.margin_row.spec_alpha_entries, Some(4096));
    assert_eq!(preset.margin_row.clip_topk, Some(50));
    assert_eq!(preset.margin_row.closure_thresh, Some(0.5));
    assert_eq!(preset.margin_row.bi_chunk, Some(1024));
    assert_eq!(preset.margin_row.bi_chunk_secs, Some(2.0));
    assert_eq!(preset.margin_row.bi_topk, Some(4096));
    assert_eq!(preset.margin_row.interm_alpha, Some(2));
    assert_eq!(preset.margin_row.beta_eta, Some(0.25));
    assert_eq!(preset.margin_row.beta_iters, Some(4));
    assert_eq!(preset.margin_row.beta_lambda, Some(2.0));
    assert_eq!(preset.margin_row.rebake_rounds, Some(3));
    assert_eq!(preset.margin_row.rebake_secs, Some(60.0));
    assert_eq!(preset.margin_row.bi_window, Some(true));
    assert_eq!(preset.margin_row.joint_refresh, Some(3));
    assert_eq!(preset.margin_row.joint_refresh_secs, Some(45.0));
    assert_eq!(preset.margin_row.head_groups, Some(true));
    assert_eq!(preset.margin_row.head_directional_cuts, Some(false));
    assert_eq!(preset.margin_row.head_phase_cover, Some(7));
    assert_eq!(preset.margin_row.root_certify_once, Some(true));
    assert_eq!(preset.margin_row.root_certify_once_iters, Some(30));
    assert_eq!(preset.margin_row.root_certify_once_refresh, Some(5));
    assert_eq!(preset.margin_row.root_certify_once_secs, Some(25.0));
    // #bbe-joint-root: the NON-default token lands as itself.
    assert_eq!(
        preset.margin_row.root_search,
        Some(ny_propagate::margin_row::RootSearch::JointF32)
    );
    // #root-box-source: the NON-default token lands as itself.
    assert_eq!(
        preset.margin_row.root_box_source,
        Some(ny_propagate::margin_row::RootBoxSource::PatchBackward)
    );
    assert_eq!(preset.margin_row.bi_patch, Some(true));
    assert_eq!(preset.margin_row.bi_patch_ratio, Some(0.25));
    assert_eq!(preset.margin_row.bi_patch_mb, Some(64));
    assert_eq!(preset.margin_row.ibp_seed, Some(true));
    assert_eq!(preset.margin_row.epoch_tier0, Some(8));
    assert_eq!(preset.margin_row.epoch_universe, Some(128));
    assert_eq!(preset.margin_row.epoch_retain, Some(256));
    assert_eq!(preset.margin_row.epoch_depth, Some(3));
    assert_eq!(preset.margin_row.epoch_attempts, Some(4));
    assert_eq!(preset.margin_row.epoch_parent_topk, Some(16));
    // #mr-errlane: the SECOND non-default value must land as itself (a
    // two-state probe could not tell `per_channel` from `envelope`).
    assert_eq!(
        preset.margin_row.err_envelope,
        Some(ny_propagate::margin_row::ErrLane::PerChannel)
    );
    // #frozen-dual-bab: the arming key and its two knobs land as themselves.
    assert_eq!(preset.margin_row.frozen_dual_bab, Some(true));
    assert_eq!(preset.margin_row.frozen_dual_bab_node_cap, Some(250_000));
    assert_eq!(preset.margin_row.frozen_dual_bab_secs, Some(20.0));
    assert_eq!(preset.margin_row.frozen_dual_refine_steps, Some(8));
    // #fd-shared-state: the NON-default token lands as itself.
    assert_eq!(
        preset.margin_row.frozen_dual_bab_state,
        Some(ny_propagate::margin_row::zono_bab::FrozenDualStateMode::Shared)
    );
    // #wide-fd-search: the NON-default token lands as itself.
    assert_eq!(
        preset.margin_row.frozen_dual_bab_search,
        Some(ny_propagate::margin_row::FrozenDualSearch::CpuF32)
    );
    // #wave-beta-bab: the NON-default engine token and both knobs land as
    // themselves.
    assert_eq!(
        preset.margin_row.tree_mode,
        Some(ny_propagate::margin_row::TreeMode::Wave)
    );
    assert_eq!(preset.margin_row.tree_wave_batch, Some(512));
    assert_eq!(preset.margin_row.tree_wave_cert_threads, Some(6));
    // ... and that the KILL-SWITCH direction survives, which is the only way a
    // default-ON declaration can be reached from a preset at all.
    assert_eq!(preset.bab.root_objective_directed_rows, Some(false));
    assert_eq!(preset.bab.graph_mip_leaf_sat, Some(false));
    assert_eq!(preset.bab.true_grad_gpu_replay, Some(false));
    assert_eq!(preset.bab.bab_resnet_wide, Some(false));
    assert_eq!(preset.bab.seq_weighted_budget, Some(false));
    assert_eq!(preset.margin_row.beta_polyak, Some(false));
    assert_eq!(preset.margin_row.beta_heads, Some(false));
    assert_eq!(preset.bab.sweep_class_rows, Some(1024));
    assert_eq!(preset.bab.sweep_class_mib, Some(256));
    // #metaroom-mip-governor: the nested `bab.mip` section and the CZ lane's
    // budget share, spot-checked for SHAPE as well as presence.
    assert_eq!(preset.bab.mip.leaf_undecided_streak_cap, Some(2));
    assert_eq!(
        preset.bab.mip.escalation_skips_certified_clauses,
        Some(true)
    );
    assert_eq!(preset.cz_hybrid.max_budget_fraction, Some(0.5));
}

fn assert_frozen_preset_semantics_invalid(yaml: &str, expected: &str) {
    let preset: PresetConfig =
        serde_yaml::from_str(yaml).expect("invalid semantics must still be valid typed YAML");
    let error = validate_preset(&preset).expect_err("semantic validation must reject preset");
    let direct = format!("{error:#}");
    assert!(
        direct.contains(expected),
        "direct validation error {direct:?} did not mention {expected:?}"
    );

    let dir = tempfile::tempdir().expect("create temporary preset directory");
    let path = dir.path().join("invalid.yaml");
    fs::write(&path, yaml).expect("write temporary preset");
    let snapshot = BetaCrownPresetSnapshot::load(&path);
    let frozen = snapshot
        .invalid_error()
        .expect("the frozen snapshot must retain invalid semantics");
    assert!(
        frozen.contains(expected),
        "snapshot validation error {frozen:?} did not mention {expected:?}"
    );
}

/// A typed preset is authority, not the legacy environment's best-effort input
/// layer. Values outside the declaration must fail the snapshot instead of
/// silently running the declaration default under a treatment label.
#[test]
fn frozen_preset_rejects_invalid_promoted_numeric_values() {
    let yaml = "bab:\n  batch_size: 123\nmargin_row:\n  closure_thresh: 1.5\n";
    assert_frozen_preset_semantics_invalid(yaml, "margin_row.closure_thresh");

    let preset: PresetConfig = serde_yaml::from_str(yaml).expect("typed preset must parse");
    let mut config = ny_propagate::BetaCrownConfig {
        batch_size: 777,
        ..Default::default()
    };
    apply_preset(&mut config, &preset).expect_err("invalid governed value must reject apply");
    assert_eq!(
        config.batch_size, 777,
        "governed validation must run before applying an otherwise-valid field"
    );
}

#[test]
fn frozen_preset_rejects_nonfinite_promoted_numeric_values() {
    let yaml = "margin_row:\n  closure_thresh: .nan\n";
    let parsed: PresetConfig = serde_yaml::from_str(yaml).expect("YAML supports NaN scalars");
    assert!(
        parsed
            .margin_row
            .closure_thresh
            .expect("closure threshold must be present")
            .is_nan(),
        "the regression probe must reach semantic validation with an actual NaN"
    );
    assert_frozen_preset_semantics_invalid(yaml, "margin_row.closure_thresh");
}

#[test]
fn frozen_preset_rejects_invalid_exact_vjp_policy_values() {
    for (key, value) in [("pgd_vjp_k", "0"), ("pgd_vjp_steps", "0")] {
        assert_frozen_preset_semantics_invalid(
            &format!("attack:\n  {key}: {value}\n"),
            &format!("attack.{key}"),
        );
    }
    for value in ["0", "-0", "-1", ".nan", ".inf", "1e-50"] {
        assert_frozen_preset_semantics_invalid(
            &format!("attack:\n  pgd_odi_eta: {value}\n"),
            "attack.pgd_odi_eta",
        );
    }
    for value in ["-1", ".nan", ".inf"] {
        assert_frozen_preset_semantics_invalid(
            &format!("attack:\n  pgd_hinge_tau: {value}\n"),
            "attack.pgd_hinge_tau",
        );
    }
}

#[test]
fn exact_vjp_f32_preset_values_are_canonicalized_and_applied() {
    let preset: PresetConfig = serde_yaml::from_str(
        "attack:\n  pgd_fast_grad: false\n  pgd_vjp_k: 52\n  pgd_odi_eta: 0.5\n  pgd_vjp_steps: 250\n  pgd_hinge_tau: 0.0001\n",
    )
    .unwrap();
    validate_preset(&preset).expect("representable positive/nonnegative f32 values are valid");
    let mut config = ny_propagate::BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(!config.pgd_exact_vjp.fast_gradient);
    assert_eq!(config.pgd_exact_vjp.batch_width, Some(52));
    assert_eq!(config.pgd_exact_vjp.odi_eta, 0.5_f32);
    assert_eq!(config.pgd_exact_vjp.wave_steps, Some(250));
    assert_eq!(config.pgd_exact_vjp.hinge_tau, 0.0001_f32);
}

/// #disj-odi: both keys reach the typed exact-VJP policy through
/// `apply_preset`, absence leaves the historical policy bit for bit, and the
/// two keys touch no other field of it.
#[test]
fn disj_odi_keys_apply_to_the_typed_exact_vjp_policy() {
    let preset: PresetConfig =
        serde_yaml::from_str("attack:\n  pgd_disj_odi: true\n  pgd_disj_focus_lanes: 48\n")
            .unwrap();
    validate_preset(&preset).expect("the #disj-odi keys are admissible");
    let mut config = ny_propagate::BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(config.pgd_exact_vjp.disj_odi);
    assert_eq!(config.pgd_exact_vjp.disj_focus_lanes, 48);
    assert_eq!(
        ny_propagate::ExactVjpPgdConfig {
            disj_odi: false,
            disj_focus_lanes: 0,
            ..config.pgd_exact_vjp
        },
        ny_propagate::ExactVjpPgdConfig::default()
    );

    let mut dark = ny_propagate::BetaCrownConfig::default();
    apply_preset(&mut dark, &PresetConfig::default()).unwrap();
    assert_eq!(
        dark.pgd_exact_vjp,
        ny_propagate::ExactVjpPgdConfig::default()
    );
    assert!(!dark.pgd_exact_vjp.disj_odi);
    assert_eq!(dark.pgd_exact_vjp.disj_focus_lanes, 0);
}

#[test]
fn frozen_preset_rejects_invalid_upfront_attack_fractions() {
    for key in ["upfront_attack_fraction", "bnn_attack_fraction"] {
        for value in ["0", "-0.1", "1.0001", ".nan", ".inf"] {
            assert_frozen_preset_semantics_invalid(
                &format!("attack:\n  {key}: {value}\n"),
                &format!("attack.{key}"),
            );
        }
    }
}

#[test]
fn frozen_preset_rejects_noncanonical_conv_ibp_blocked_tokens() {
    assert_frozen_preset_semantics_invalid(
        "bab:\n  conv_ibp_blocked: seqq\n",
        "noncanonical token",
    );
}

#[test]
fn frozen_preset_rejects_noncanonical_margin_row_parallel_tokens() {
    for token in ["1", "all", "true", "3", " 8 "] {
        assert_frozen_preset_semantics_invalid(
            &format!("margin_row:\n  parallel: '{token}'\n"),
            "noncanonical token",
        );
    }
}

/// #mr-errlane: the typed key admits exactly the declaration's four tokens
/// (serde rejects anything else at load, before any lane could run under a
/// mislabelled treatment), and each admitted token validates and projects to
/// the same `Enum` member the scored LeverSet resolves.
#[test]
fn margin_row_err_envelope_tokens_are_exactly_the_declaration() {
    use ny_propagate::margin_row::ErrLane;
    for token in ErrLane::TOKENS {
        let yaml = format!("margin_row:\n  err_envelope: {token}\n");
        let preset: PresetConfig =
            serde_yaml::from_str(&yaml).expect("every declared token must parse");
        let lane = preset
            .margin_row
            .err_envelope
            .expect("the token must land in the typed field");
        assert_eq!(lane.token(), token);
        validate_preset(&preset).expect("a declared token must pass governed validation");
    }
    for bogus in ["on", "true", "row", "Envelope", "per-channel", "1"] {
        let yaml = format!("margin_row:\n  err_envelope: {bogus}\n");
        assert!(
            serde_yaml::from_str::<PresetConfig>(&yaml).is_err(),
            "{bogus:?} must be rejected at load"
        );
    }
}

#[test]
fn margin_row_head_groups_preserves_typed_true_false_and_absence() {
    let absent: PresetConfig =
        serde_yaml::from_str("margin_row: {}\n").expect("an empty margin-row preset parses");
    assert_eq!(absent.margin_row.head_groups, None);
    assert_eq!(PresetConfig::default().margin_row.head_groups, None);
    validate_preset(&absent).expect("the absent key preserves the default preset");

    for enabled in [false, true] {
        let yaml = format!("margin_row:\n  head_groups: {enabled}\n");
        let preset: PresetConfig = serde_yaml::from_str(&yaml).expect("typed bool parses");
        assert_eq!(preset.margin_row.head_groups, Some(enabled));
        validate_preset(&preset).expect("either explicit boolean is admissible");
        let serialized = serde_json::to_value(&preset).expect("serialize the typed preset");
        assert_eq!(
            serialized["margin_row"]["head_groups"].as_bool(),
            Some(enabled)
        );
        let restored: PresetConfig =
            serde_json::from_value(serialized).expect("restore the typed preset");
        assert_eq!(restored.margin_row.head_groups, Some(enabled));
    }
    for invalid in ["1", "'true'", "'false'", "{}", "[]"] {
        let yaml = format!("margin_row:\n  head_groups: {invalid}\n");
        assert!(
            serde_yaml::from_str::<PresetConfig>(&yaml).is_err(),
            "{invalid:?}"
        );
    }
}

#[test]
fn margin_row_head_directional_cuts_preserves_typed_true_false_and_absence() {
    let absent: PresetConfig = serde_yaml::from_str("margin_row: {}\n").unwrap();
    assert_eq!(absent.margin_row.head_directional_cuts, None);
    assert_eq!(
        PresetConfig::default().margin_row.head_directional_cuts,
        None
    );
    validate_preset(&absent).unwrap();
    for enabled in [false, true] {
        let yaml = format!("margin_row:\n  head_directional_cuts: {enabled}\n");
        let preset: PresetConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(preset.margin_row.head_directional_cuts, Some(enabled));
        validate_preset(&preset).unwrap();
        let serialized = serde_json::to_value(&preset).unwrap();
        assert_eq!(
            serialized["margin_row"]["head_directional_cuts"].as_bool(),
            Some(enabled)
        );
        let restored: PresetConfig = serde_json::from_value(serialized).unwrap();
        assert_eq!(restored.margin_row.head_directional_cuts, Some(enabled));
    }
    for invalid in ["1", "'true'", "'false'", "{}", "[]"] {
        let yaml = format!("margin_row:\n  head_directional_cuts: {invalid}\n");
        assert!(
            serde_yaml::from_str::<PresetConfig>(&yaml).is_err(),
            "{invalid:?}"
        );
    }
}

#[test]
fn margin_row_head_phase_cover_preserves_off_and_rejects_unbounded_yaml() {
    let absent: PresetConfig = serde_yaml::from_str("margin_row: {}\n").unwrap();
    assert_eq!(absent.margin_row.head_phase_cover, None);
    assert_eq!(PresetConfig::default().margin_row.head_phase_cover, None);
    validate_preset(&absent).unwrap();
    for count in 0..=7 {
        let yaml = format!("margin_row:\n  head_groups: true\n  head_phase_cover: {count}\n");
        let preset: PresetConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(preset.margin_row.head_phase_cover, Some(count));
        validate_preset(&preset).unwrap();
        let serialized = serde_json::to_value(&preset).unwrap();
        assert_eq!(
            serialized["margin_row"]["head_phase_cover"].as_u64(),
            Some(count as u64)
        );
        let restored: PresetConfig = serde_json::from_value(serialized).unwrap();
        assert_eq!(restored.margin_row.head_phase_cover, Some(count));
    }
    for invalid in [
        "8",
        "128",
        "-1",
        "true",
        "'7'",
        "1.5",
        "{}",
        "[]",
        "18446744073709551616",
    ] {
        let yaml = format!("margin_row:\n  head_phase_cover: {invalid}\n");
        assert!(
            serde_yaml::from_str::<PresetConfig>(&yaml).is_err(),
            "{invalid:?}"
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("invalid-phase-cover.yaml");
        fs::write(&path, yaml).unwrap();
        assert!(BetaCrownPresetSnapshot::load(&path)
            .invalid_error()
            .is_some());
    }
}

/// #bbe-joint-root: the typed key admits exactly the declaration's two tokens
/// (serde rejects anything else at load), and each admitted token validates and
/// projects to the `Enum` member the scored LeverSet resolves.
#[test]
fn margin_row_root_search_tokens_are_exactly_the_declaration() {
    use ny_propagate::margin_row::RootSearch;
    let ny_levers::LeverKind::Enum(members) =
        ny_levers::decls::margin_row::MARGIN_ROW_ROOT_SEARCH.kind
    else {
        panic!("margin_row.root_search must be declared as an Enum");
    };
    assert_eq!(members, &RootSearch::TOKENS[..]);
    for token in RootSearch::TOKENS {
        let yaml = format!("margin_row:\n  root_search: {token}\n");
        let preset: PresetConfig =
            serde_yaml::from_str(&yaml).expect("every declared token must parse");
        let search = preset
            .margin_row
            .root_search
            .expect("the token must land in the typed field");
        assert_eq!(search.token(), token);
        validate_preset(&preset).expect("a declared token must pass governed validation");
    }
    for bogus in ["joint", "JointF32", "joint-f32", "f32", "1", "true"] {
        let yaml = format!("margin_row:\n  root_search: {bogus}\n");
        assert!(
            serde_yaml::from_str::<PresetConfig>(&yaml).is_err(),
            "{bogus:?} must be rejected at load"
        );
    }
}

/// The refinement key is a nonnegative work request. Its tier owns the upper
/// clamp; preserving oversized requests through typed delivery lets that clamp
/// be tested at the actual consumer instead of silently changing precedence.
#[test]
fn frozen_dual_refine_steps_are_typed_and_generic_default_is_zero() {
    assert_eq!(
        PresetConfig::default().margin_row.frozen_dual_refine_steps,
        None
    );
    for steps in [0, 1, 4, 8, 32, 4096] {
        let yaml = format!("margin_row:\n  frozen_dual_refine_steps: {steps}\n");
        let preset: PresetConfig = serde_yaml::from_str(&yaml)
            .expect("a nonnegative refinement request must reach the typed field");
        assert_eq!(preset.margin_row.frozen_dual_refine_steps, Some(steps));
        assert_eq!(preset.margin_row.frozen_dual_bab, None);
        validate_preset(&preset).expect("the governed refinement request must validate");
    }
    for bogus in ["-1", "1.5", "true", "many"] {
        let yaml = format!("margin_row:\n  frozen_dual_refine_steps: {bogus}\n");
        assert!(
            serde_yaml::from_str::<PresetConfig>(&yaml).is_err(),
            "{bogus:?} must not bypass the typed nonnegative request"
        );
    }
}

/// #fd-shared-state: the typed key admits exactly the declaration's two tokens
/// (serde rejects anything else at load), and each admitted token validates and
/// projects to the `Enum` member the tier resolves.
#[test]
fn margin_row_frozen_dual_bab_state_tokens_are_exactly_the_declaration() {
    use ny_propagate::margin_row::zono_bab::FrozenDualStateMode;
    let ny_levers::LeverKind::Enum(members) =
        ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB_STATE.kind
    else {
        panic!("margin_row.frozen_dual_bab_state must be declared as an Enum");
    };
    assert_eq!(members, &FrozenDualStateMode::TOKENS[..]);
    assert_eq!(
        ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB_STATE.default,
        ny_levers::DefaultSpec::Text(FrozenDualStateMode::PerClass.token()),
        "the declaration default must be the bit-identical per-class tier"
    );
    for token in FrozenDualStateMode::TOKENS {
        let yaml = format!("margin_row:\n  frozen_dual_bab_state: {token}\n");
        let preset: PresetConfig =
            serde_yaml::from_str(&yaml).expect("every declared token must parse");
        let mode = preset
            .margin_row
            .frozen_dual_bab_state
            .expect("the token must land in the typed field");
        assert_eq!(mode.token(), token);
        assert_eq!(FrozenDualStateMode::from_token(token), Some(mode));
        validate_preset(&preset).expect("a declared token must pass governed validation");
    }
    for bogus in ["perclass", "Shared", "per-class", "trunk", "1", "true"] {
        let yaml = format!("margin_row:\n  frozen_dual_bab_state: {bogus}\n");
        assert!(
            serde_yaml::from_str::<PresetConfig>(&yaml).is_err(),
            "{bogus:?} must be rejected at load"
        );
    }
}

/// #root-box-source: the typed key admits exactly the declaration's two tokens
/// (serde rejects anything else at load), and each admitted token validates and
/// projects to the `Enum` member the scored LeverSet resolves.
#[test]
fn margin_row_root_box_source_tokens_are_exactly_the_declaration() {
    use ny_propagate::margin_row::RootBoxSource;
    let ny_levers::LeverKind::Enum(members) =
        ny_levers::decls::margin_row::MARGIN_ROW_ROOT_BOX_SOURCE.kind
    else {
        panic!("margin_row.root_box_source must be declared as an Enum");
    };
    assert_eq!(members, &RootBoxSource::TOKENS[..]);
    for token in RootBoxSource::TOKENS {
        let yaml = format!("margin_row:\n  root_box_source: {token}\n");
        let preset: PresetConfig =
            serde_yaml::from_str(&yaml).expect("every declared token must parse");
        let source = preset
            .margin_row
            .root_box_source
            .expect("the token must land in the typed field");
        assert_eq!(source.token(), token);
        validate_preset(&preset).expect("a declared token must pass governed validation");
    }
    for bogus in [
        "patch",
        "PatchBackward",
        "patch-backward",
        "backward",
        "1",
        "true",
    ] {
        let yaml = format!("margin_row:\n  root_box_source: {bogus}\n");
        assert!(
            serde_yaml::from_str::<PresetConfig>(&yaml).is_err(),
            "{bogus:?} must be rejected at load"
        );
    }
}

/// RETIRED GPU keys (ENGINE_PROGRAM R1 kill line + R2): `margin_row.gpu_conv`
/// and `margin_row.bi_gpu` still PARSE (an old yaml gets the reason, not an
/// unknown-key error) but any value — `false` included — is refused at
/// validation and therefore by the frozen VNN-COMP snapshot; neither is a
/// `Deliver::PresetKey` axis any more, and the governed projection never sees
/// them.
#[test]
fn retired_gpu_keys_are_refused_at_validation() {
    for key in ["gpu_conv", "bi_gpu"] {
        for value in ["true", "false"] {
            let yaml = format!("margin_row:\n  {key}: {value}\n");
            assert_frozen_preset_semantics_invalid(&yaml, &format!("margin_row.{key} is RETIRED"));
        }
        let path = format!("margin_row.{key}");
        assert!(
            !ny_levers::space::axes()
                .iter()
                .chain(ny_levers::space::unsafe_axes())
                .any(|axis| matches!(
                    axis.deliver,
                    ny_levers::space::Deliver::PresetKey(key) if key == path
                )),
            "{path} must no longer be a delivered preset key"
        );
    }
    for decl in [
        &ny_levers::decls::margin_row::MARGIN_ROW_GPU_CONV,
        &ny_levers::decls::margin_row::MARGIN_ROW_BI_GPU,
    ] {
        let axis = ny_levers::space::unsafe_axes()
            .iter()
            .find(|axis| axis.name == decl.name)
            .unwrap_or_else(|| panic!("{} must be a withheld (Unsafe) axis", decl.name));
        assert_eq!(axis.deliver, ny_levers::space::Deliver::EnvOnly);
        assert_eq!(axis.class, ny_levers::space::Class::Unsafe);
    }
}

/// #root-box-source COMPOSITION (ENGINE_PROGRAM M5b integration): the
/// patch-backward source builds no clip rows, no Tier-0 retained rows and runs
/// no f32 tableau kernel, so a preset that arms `patch_backward` TOGETHER with
/// any of `clip_rows`, a retention-requesting epoch key, `root_f32` or
/// `tableau_f32` is refused at validation (and therefore by the frozen VNN-COMP
/// snapshot) with the conflict named — it can never ship with the product
/// silently switched off. The same keys without `patch_backward`, `tableau`
/// with all of them, `patch_backward` alone, a `false` value and the
/// incremental epoch arm (which retains nothing) all stay valid.
#[test]
fn root_box_patch_backward_composition_is_refused_at_validation() {
    for (conflict, reason) in [
        ("  clip_rows: true\n", "clip_rows"),
        ("  epoch_tier0: 8\n", "retain"),
        ("  epoch_depth: 3\n", "retain"),
        ("  root_f32: true\n", "root_f32"),
        (
            "  root_f32: true\n  tableau_f32: true\n",
            "root_f32+tableau_f32",
        ),
        (
            "  clip_rows: true\n  root_f32: true\n",
            "clip_rows+root_f32",
        ),
    ] {
        let yaml = format!("margin_row:\n  root_box_source: patch_backward\n{conflict}");
        assert_frozen_preset_semantics_invalid(
            &yaml,
            &format!("patch_backward cannot compose with {reason} "),
        );
        // The same product without the source, and with `tableau`, is valid.
        for base in ["", "  root_box_source: tableau\n"] {
            let yaml = format!("margin_row:\n{base}{conflict}");
            let preset: PresetConfig = serde_yaml::from_str(&yaml).expect("typed preset");
            validate_preset(&preset).unwrap_or_else(|e| {
                panic!("{yaml:?} must stay valid without patch_backward: {e:#}")
            });
        }
    }
    for fine in [
        "",
        "  clip_rows: false\n  root_f32: false\n  tableau_f32: false\n",
        "  epoch_tier0: 0\n",
        // The incremental epoch arm consumes no Tier-0 rows, so nothing is
        // retained (`epoch_retention_requested`).
        "  epoch_depth: 3\n  epoch_parent_topk: 16\n",
    ] {
        let yaml = format!("margin_row:\n  root_box_source: patch_backward\n{fine}");
        let preset: PresetConfig = serde_yaml::from_str(&yaml).expect("typed preset");
        validate_preset(&preset)
            .unwrap_or_else(|e| panic!("{yaml:?} composes and must stay valid: {e:#}"));
    }
}

/// #wide-fd-search: the typed key admits exactly the declaration's four
/// tokens (serde rejects anything else at load), and each admitted token
/// validates and projects to the `Enum` member the scored LeverSet resolves.
#[test]
fn margin_row_frozen_dual_bab_search_tokens_are_exactly_the_declaration() {
    use ny_propagate::margin_row::FrozenDualSearch;
    let ny_levers::LeverKind::Enum(members) =
        ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB_SEARCH.kind
    else {
        panic!("margin_row.frozen_dual_bab_search must be declared as an Enum");
    };
    assert_eq!(members, &FrozenDualSearch::TOKENS[..]);
    for token in FrozenDualSearch::TOKENS {
        let yaml = format!("margin_row:\n  frozen_dual_bab_search: {token}\n");
        let preset: PresetConfig =
            serde_yaml::from_str(&yaml).expect("every declared token must parse");
        let search = preset
            .margin_row
            .frozen_dual_bab_search
            .expect("the token must land in the typed field");
        assert_eq!(search.token(), token);
        assert_eq!(FrozenDualSearch::from_token(token), Some(search));
        validate_preset(&preset).expect("a declared token must pass governed validation");
    }
    for bogus in ["f32", "gpu", "CpuF32", "cpu-f32", "true", "1", "F64"] {
        let yaml = format!("margin_row:\n  frozen_dual_bab_search: {bogus}\n");
        assert!(
            serde_yaml::from_str::<PresetConfig>(&yaml).is_err(),
            "{bogus:?} must be rejected at load"
        );
    }
}

/// #wave-beta-bab: the typed key admits exactly the declaration's two tokens
/// (serde rejects anything else at load), and each admitted token validates
/// and projects to the `Enum` member the lane resolves.
#[test]
fn margin_row_tree_mode_tokens_are_exactly_the_declaration() {
    use ny_propagate::margin_row::TreeMode;
    let ny_levers::LeverKind::Enum(members) =
        ny_levers::decls::margin_row::MARGIN_ROW_TREE_MODE.kind
    else {
        panic!("margin_row.tree_mode must be declared as an Enum");
    };
    assert_eq!(members, &TreeMode::TOKENS[..]);
    for token in TreeMode::TOKENS {
        let yaml = format!("margin_row:\n  tree_mode: {token}\n");
        let preset: PresetConfig =
            serde_yaml::from_str(&yaml).expect("every declared token must parse");
        let mode = preset
            .margin_row
            .tree_mode
            .expect("the token must land in the typed field");
        assert_eq!(mode.token(), token);
        assert_eq!(TreeMode::from_token(token), Some(mode));
        validate_preset(&preset).expect("a declared token must pass governed validation");
    }
    for bogus in ["Wave", "WAVE", "waves", "beta", "true", "1", "legacy "] {
        let yaml = format!("margin_row:\n  tree_mode: \"{bogus}\"\n");
        assert!(
            serde_yaml::from_str::<PresetConfig>(&yaml).is_err(),
            "{bogus:?} must be rejected at load"
        );
    }
}

#[test]
fn frozen_preset_rejects_the_quarantined_const_matmul_interval_route() {
    for value in [false, true] {
        assert_frozen_preset_semantics_invalid(
            &format!("solver:\n  const_matmul_interval: {value}\n"),
            "quarantined and cannot be delivered by a preset",
        );
    }
}

#[test]
fn frozen_preset_accepts_valid_governed_representatives() {
    for conv_token in ["0", "1", "seq"] {
        for parallel_token in ["0", "2", "4", "8", "16"] {
            let yaml = format!(
                r#"attack:
  upfront_attack: true
  upfront_attack_cap_secs: 0
  upfront_attack_auto_cap_secs: 4
  upfront_attack_fraction: 0.08
  bnn_attack_cap_secs: 180
  bnn_attack_fraction: 0.5
bab:
  conv_ibp_blocked: '{conv_token}'
  cgan_stacked_budget_mb: 4096
margin_row:
  lru: 128
  parallel: '{parallel_token}'
  alpha_iters: 16
  alpha_secs: 40.0
  alpha_lr: 0.1
  alpha_cols: 8
  spec_alpha: true
  spec_alpha_width: 4
  spec_alpha_cands: 5
  spec_alpha_rounds: 2
  spec_alpha_entries: 4096
  alpha_guard: false
  bi_chunk_secs: 2.0
  closure_thresh: 0.5
  closure_warmup: 32
  epoch_tier0: 8
  epoch_universe: 128
  epoch_retain: 256
  epoch_depth: 3
  epoch_attempts: 4
  epoch_parent_topk: 16
"#
            );
            let preset: PresetConfig =
                serde_yaml::from_str(&yaml).expect("valid representatives must parse");
            validate_preset(&preset).expect("valid representatives must pass direct validation");

            let dir = tempfile::tempdir().expect("create temporary preset directory");
            let path = dir.path().join("valid.yaml");
            fs::write(&path, &yaml).expect("write temporary preset");
            assert!(
                BetaCrownPresetSnapshot::load(&path).loaded().is_some(),
                "canonical conv token {conv_token:?} and parallel token \
                 {parallel_token:?} must produce a valid frozen snapshot"
            );
        }
    }
}
