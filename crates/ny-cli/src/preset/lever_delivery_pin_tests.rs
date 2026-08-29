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
//! scored run. This file is that contract, mechanically.
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
        path: "attack.pre_softmax_objective",
        is_set: |c| c.attack.pre_softmax_objective.is_some(),
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
    PromotedKey {
        path: "margin_row.alpha_secs",
        is_set: |c| c.margin_row.alpha_secs.is_some(),
    },
    PromotedKey {
        path: "margin_row.alpha_lr",
        is_set: |c| c.margin_row.alpha_lr.is_some(),
    },
    PromotedKey {
        path: "margin_row.alpha_cols",
        is_set: |c| c.margin_row.alpha_cols.is_some(),
    },
    PromotedKey {
        path: "margin_row.alpha_guard",
        is_set: |c| c.margin_row.alpha_guard.is_some(),
    },
    PromotedKey {
        path: "margin_row.clip",
        is_set: |c| c.margin_row.clip.is_some(),
    },
    PromotedKey {
        path: "margin_row.clip_interm",
        is_set: |c| c.margin_row.clip_interm.is_some(),
    },
    PromotedKey {
        path: "margin_row.clip_rows",
        is_set: |c| c.margin_row.clip_rows.is_some(),
    },
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
        path: "margin_row.closure_max_tier",
        is_set: |c| c.margin_row.closure_max_tier.is_some(),
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
        path: "margin_row.column_retire",
        is_set: |c| c.margin_row.column_retire.is_some(),
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
];

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
            if (key.is_set)(preset) {
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
  pre_softmax_objective: true
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
  phase_budget:
    post_bab_reserve_secs: 100
margin_row:
  lru: 128
  parallel: '8'
  gpu: true
  gpu_batch: true
  gpu_backward: true
  alpha_opt: true
  alpha_iters: 16
  alpha_secs: 40.0
  alpha_lr: 0.1
  alpha_cols: 8
  alpha_guard: false
  clip: true
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
  epoch_tier0: 8
  epoch_universe: 128
  epoch_retain: 256
  epoch_depth: 3
  epoch_attempts: 4
  epoch_parent_topk: 16
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
    assert_eq!(preset.margin_row.clip_topk, Some(50));
    assert_eq!(preset.margin_row.closure_thresh, Some(0.5));
    assert_eq!(preset.margin_row.bi_chunk, Some(1024));
    assert_eq!(preset.margin_row.bi_chunk_secs, Some(2.0));
    assert_eq!(preset.margin_row.bi_topk, Some(4096));
    assert_eq!(preset.margin_row.beta_eta, Some(0.25));
    assert_eq!(preset.margin_row.beta_iters, Some(4));
    assert_eq!(preset.margin_row.beta_lambda, Some(2.0));
    assert_eq!(preset.margin_row.rebake_rounds, Some(3));
    assert_eq!(preset.margin_row.rebake_secs, Some(60.0));
    assert_eq!(preset.margin_row.epoch_tier0, Some(8));
    assert_eq!(preset.margin_row.epoch_universe, Some(128));
    assert_eq!(preset.margin_row.epoch_retain, Some(256));
    assert_eq!(preset.margin_row.epoch_depth, Some(3));
    assert_eq!(preset.margin_row.epoch_attempts, Some(4));
    assert_eq!(preset.margin_row.epoch_parent_topk, Some(16));
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
