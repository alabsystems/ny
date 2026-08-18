// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! VNN-COMP benchmark preset loading tests.
//!
//! Extracted from `tests.rs` to stay under the 1000-line limit.
//! These tests load actual YAML configs from `configs/vnncomp25/` and verify
//! that parsed preset values flow correctly to `BetaCrownConfig`.

use super::apply::apply_preset;
use super::*;
use crate::commands::beta_crown::branching::{
    apply_resolved_auto_branching_runtime_policy, resolve_auto_branching, AutoBranchingReason,
    AutoBranchingRequest, ModelStructure,
};
use ny_propagate::{
    BetaCrownConfig, BranchingHeuristic, InputClipType, PgdAlphaMode, PgdOptimizer,
};
use std::{fs, path::Path};

#[test]
fn only_vnncomp25_cifar100_arms_root_alpha_phase_checkpoint() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("preset directory entry").path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("yaml") {
                continue;
            }
            let text = fs::read_to_string(&path).expect("readable shipped preset");
            let preset: PresetConfig = serde_yaml::from_str(&text)
                .unwrap_or_else(|error| panic!("preset {} must parse: {error}", path.display()));
            if preset.bab.root_alpha_phase_checkpoint == Some(true) {
                armed.push(
                    path.strip_prefix(&root)
                        .expect("preset below config root")
                        .to_path_buf(),
                );
            }
        }
    }
    armed.sort();
    assert_eq!(
        armed,
        [Path::new("vnncomp25/cifar100_2024.yaml").to_path_buf()],
        "the root-alpha checkpoint production opt-in must remain scoped to measured CIFAR100"
    );
}

#[test]
fn effective_cifar100_2024_preset_arms_root_alpha_phase_checkpoint() {
    let configs_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let effective = crate::commands::vnncomp::resolve_preset_path(&configs_root, "cifar100_2024")
        .expect("CIFAR100 production preset resolves");
    let preset = load_preset(&effective).expect("effective CIFAR100 preset loads");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("effective CIFAR100 preset applies");
    assert!(
        config.root_alpha_phase_checkpoint,
        "newer vnncomp directories must not silently shadow the qualified CIFAR100 policy"
    );
}

#[test]
fn only_vnncomp25_cifar100_arms_kfsb_cert_reuse() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("preset directory entry").path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("yaml") {
                continue;
            }
            let text = fs::read_to_string(&path).expect("readable shipped preset");
            let preset: PresetConfig = serde_yaml::from_str(&text)
                .unwrap_or_else(|error| panic!("preset {} must parse: {error}", path.display()));
            if preset.bab.branching.kfsb_cert_reuse == Some(true) {
                armed.push(
                    path.strip_prefix(&root)
                        .expect("preset below config root")
                        .to_path_buf(),
                );
            }
        }
    }
    armed.sort();
    assert_eq!(
        armed,
        [Path::new("vnncomp25/cifar100_2024.yaml").to_path_buf()],
        "the proof-carrying kFSB reuse opt-in must remain scoped to qualified CIFAR100"
    );
}

#[test]
fn effective_cifar100_2024_preset_arms_kfsb_cert_reuse() {
    let configs_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let effective = crate::commands::vnncomp::resolve_preset_path(&configs_root, "cifar100_2024")
        .expect("CIFAR100 production preset resolves");
    assert!(
        effective.ends_with("vnncomp25/cifar100_2024.yaml"),
        "the missing vnncomp26 override must fall back to the vnncomp25 CIFAR100 preset: {}",
        effective.display()
    );
    let preset = load_preset(&effective).expect("effective CIFAR100 preset loads");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("effective CIFAR100 preset applies");
    assert!(
        config.kfsb_cert_reuse,
        "newer vnncomp directories must not silently shadow the CIFAR100 certificate policy"
    );
}

#[test]
fn only_vnncomp25_acas_arms_relational_exact_gradient_wrapper() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("preset directory entry").path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("yaml") {
                continue;
            }
            let text = fs::read_to_string(&path).expect("readable shipped preset");
            let preset: PresetConfig = serde_yaml::from_str(&text)
                .unwrap_or_else(|error| panic!("preset {} must parse: {error}", path.display()));
            if preset.attack.vnncomp_upfront_relational_exact_gradient == Some(true) {
                armed.push(
                    path.strip_prefix(&root)
                        .expect("preset below config root")
                        .to_path_buf(),
                );
            }
        }
    }
    armed.sort();
    assert_eq!(
        armed,
        [Path::new("vnncomp25/acasxu_2023.yaml").to_path_buf()],
        "the default-off relational wrapper opt-in must remain ACAS-only"
    );
}

#[test]
fn no_shipped_preset_arms_mo_cuda_factory_engine_handoff() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("preset directory entry").path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("yaml") {
                continue;
            }
            let text = fs::read_to_string(&path).expect("readable shipped preset");
            let preset: PresetConfig = serde_yaml::from_str(&text)
                .unwrap_or_else(|error| panic!("preset {} must parse: {error}", path.display()));
            if preset.bab.mo_cuda_factory_engine_handoff.is_some() {
                armed.push(path);
            }
        }
    }

    assert!(
        armed.is_empty(),
        "the default-dark handoff must remain absent from every shipped preset: {armed:?}"
    );
}

#[test]
fn no_shipped_preset_arms_mo_cuda_bounded_shared_executor() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("preset directory entry").path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("yaml") {
                continue;
            }
            let text = fs::read_to_string(&path).expect("readable shipped preset");
            let preset: PresetConfig = serde_yaml::from_str(&text)
                .unwrap_or_else(|error| panic!("preset {} must parse: {error}", path.display()));
            if preset.bab.mo_cuda_bounded_shared_executor.is_some() {
                armed.push(path);
            }
        }
    }

    assert!(
        armed.is_empty(),
        "the default-dark bounded shared executor must remain absent from every shipped preset: \
         {armed:?}"
    );
}

#[test]
fn cifar100_critical_alpha_bracket_uses_beta_crown_lr_not_adaptive_default() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cifar = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    assert_eq!(cifar.bab.beta_crown.lr_alpha, Some(0.1));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &cifar).unwrap();
    assert_eq!(
        config.alpha_lr.to_bits(),
        0.1_f32.to_bits(),
        "the bracket base must be the applied beta-CROWN alpha LR"
    );
    assert_eq!(
        config.adaptive_config.alpha_lr.to_bits(),
        0.01_f32.to_bits(),
        "the sealed adaptive default remains distinct and must not seed the bracket"
    );
    assert_eq!(
        [0.3_f32, 1.0, 2.0].map(|scale| (config.alpha_lr * scale).to_bits()),
        [0.1_f32 * 0.3, 0.1, 0.2].map(f32::to_bits),
        "the applied CIFAR base yields the staged 0.03/0.10/0.20 bracket"
    );
}

#[test]
fn cifar100_2024_arms_bounded_dense_head_crown_only_in_its_preset() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cifar = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    assert_eq!(
        cifar.general.device.as_deref(),
        Some("wgpu"),
        "the CIFAR preset must pass a WGPU engine into the ordinary verifier route"
    );
    assert_eq!(cifar.bab.root_crown_interm_dense_head, Some(true));
    assert_eq!(cifar.bab.root_crown_interm_max_secs, Some(2));
    assert_eq!(cifar.bab.root_crown_interm_max_dim, Some(512));

    let mut cifar_config = BetaCrownConfig::default();
    apply_preset(&mut cifar_config, &cifar).unwrap();
    assert!(cifar_config.root_crown_interm_dense_head);
    assert_eq!(cifar_config.root_crown_interm_max_secs, 2);
    assert_eq!(cifar_config.root_crown_interm_max_dim, 512);
    assert_eq!(
        cifar_config.alpha_config.iterations, 20,
        "CIFAR keeps its alpha-update schedule until a fast initial-CROWN route \
         preserves the useful state without regressing banked UNSAT rows"
    );

    let acas = load_preset(&repo_root.join("configs/vnncomp25/acasxu_2023.yaml")).unwrap();
    assert_eq!(acas.bab.root_crown_interm_dense_head, None);
    let mut acas_config = BetaCrownConfig::default();
    apply_preset(&mut acas_config, &acas).unwrap();
    assert!(
        !acas_config.root_crown_interm_dense_head,
        "representative non-cifar presets must remain default-off"
    );
}

#[test]
fn cifar100_2024_types_the_exact_open_row_adaptive_route() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cifar = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    assert_eq!(cifar.margin_row.adaptive_reserve, Some(true));

    let acas = load_preset(&repo_root.join("configs/vnncomp25/acasxu_2023.yaml")).unwrap();
    assert_eq!(
        acas.margin_row.adaptive_reserve, None,
        "unrelated categories retain the historical reserve policy"
    );
}

#[test]
fn cifar100_2024_enables_reference_adaptive_relu_split_depth() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cifar = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    assert_eq!(cifar.bab.min_batch_size_ratio, Some(0.1));
    assert_eq!(cifar.bab.max_split_depth, Some(4));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &cifar).unwrap();
    assert_eq!(config.min_batch_fill_ratio, 0.1);
    assert_eq!(config.max_relu_split_depth, 4);
    assert_eq!(
        config.effective_relu_split_depth(1),
        4,
        "singleton CIFAR root should commit the reference 16-leaf split"
    );
}

/// #a2-parity-not-exercised (measured 2026-08-09, GB10, `--features mip,cuda`).
///
/// The asymmetry pinned below is NOT an oversight waiting to be corrected by a
/// copy from the sibling preset. Arming the whole cifar100 bundle on
/// tinyimagenet — `batch_size` 128 -> 256, `candidates` 10 -> 7, the `clip`
/// block, `kfsb_multi`, `max_split_depth`, `min_batch_size_ratio`,
/// `alpha_crown.lr_alpha` 0.2 -> 0.25, `beta_crown.iterations` 15 -> 10 and
/// `root_crown_interm_dense_head` — was A/B'd at the official 100 s budget,
/// arms interleaved per instance, on both halves of the field (the GT-decided
/// rows ny misses and the rows it banks).
///
/// The arms were confirmed LIVE from the run's own effective-config receipt,
/// and the result was still null, because the internal verifier reports
/// `domains_explored: 0` on this category: every key in the bundle except the
/// dense head is consumed inside the BaB loop, which never runs. The verdicts
/// come from the concurrent margin-row lane, whose entry point
/// (`ny_propagate::margin_row::run_margin_row_lane`) takes no `BetaCrownConfig`
/// at all, so no `bab.*` key can reach it by signature.
///
/// See `docs/TINYIMAGENET_PARITY_A2_NOT_EXERCISED_2026-08-09.md`. Re-attempt
/// only after `domains_explored > 0` is demonstrated here.
#[test]
fn complete_clip_experiment_is_scoped_to_cifar100_not_tinyimagenet() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cifar = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    let tiny = load_preset(&repo_root.join("configs/vnncomp25/tinyimagenet_2024.yaml")).unwrap();

    assert_eq!(cifar.bab.clip.interm_domain, Some(true));
    assert_eq!(cifar.bab.clip.interm_topk, Some(20));
    assert_eq!(cifar.bab.max_split_depth, Some(4));
    assert_eq!(cifar.bab.branching.kfsb_multi, Some(true));
    assert_eq!(cifar.bab.branching.kfsb_cert_reuse, Some(true));
    assert_eq!(cifar.bab.root_alpha_phase_checkpoint, Some(true));

    let mut cifar_config = BetaCrownConfig::default();
    apply_preset(&mut cifar_config, &cifar).unwrap();
    assert!(cifar_config.enable_clip_interm_domain);
    assert_eq!(cifar_config.clip_interm_topk, 20);
    assert_eq!(cifar_config.max_relu_split_depth, 4);
    assert!(cifar_config.use_kfsb_multi_branching);
    assert!(cifar_config.kfsb_cert_reuse);
    assert!(cifar_config.root_alpha_phase_checkpoint);

    assert_eq!(tiny.bab.clip.interm_domain, None);
    assert_eq!(tiny.bab.clip.interm_topk, None);
    assert_eq!(tiny.bab.max_split_depth, None);
    assert_eq!(tiny.bab.branching.kfsb_multi, None);
    assert_eq!(tiny.bab.branching.kfsb_cert_reuse, None);
    assert_eq!(tiny.bab.root_alpha_phase_checkpoint, None);

    let mut tiny_config = BetaCrownConfig::default();
    apply_preset(&mut tiny_config, &tiny).unwrap();
    assert!(!tiny_config.enable_clip_interm_domain);
    assert_eq!(tiny_config.clip_interm_topk, 3);
    assert_eq!(tiny_config.max_relu_split_depth, 1);
    assert!(!tiny_config.use_kfsb_multi_branching);
    assert!(!tiny_config.kfsb_cert_reuse);
    assert!(!tiny_config.root_alpha_phase_checkpoint);
}

/// #tinyimagenet-alloc-parity-no-go: the CIFAR allocation trio regressed the
/// official-budget TinyImageNet sample from 5/10 to 2/10, including three lost
/// SAT rows. Pin the measured TinyImageNet baseline so a sibling-preset transfer
/// cannot silently override category-specific evidence again.
#[test]
fn tinyimagenet_2024_rejects_the_regressive_cifar100_allocation_trio() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let tiny = load_preset(&repo_root.join("configs/vnncomp25/tinyimagenet_2024.yaml")).unwrap();

    assert_eq!(tiny.bab.root_alpha_cap_secs, None);
    assert_eq!(tiny.bab.phase_budget.disjunctive_pgd_fraction, Some(0.40));
    assert_eq!(tiny.margin_row.adaptive_reserve, None);
    assert_eq!(
        tiny.margin_row.release_frac, None,
        "no margin-row reserve override should ride along with the rejected transfer"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &tiny).unwrap();
    assert_eq!(config.root_alpha_cap_secs, None);
    assert_eq!(config.phase_budget.disjunctive_pgd_fraction, 0.40);

    // Bound-quality knobs remain independently scoped and are not part of this rollback.
    assert_eq!(tiny.bab.root_crown_interm_dense_head, None);
    assert_eq!(tiny.bab.clip.interm_domain, None);
    assert_eq!(tiny.bab.branching.kfsb_multi, None);
}

#[test]
fn cersyve_preset_enables_complete_clipping() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/cersyve.yaml")).unwrap();

    assert_eq!(
        preset.bab.clip.clip_type.as_deref(),
        Some("complete"),
        "cersyve preset should enable complete clipping"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(
        config.input_clip_type,
        InputClipType::Complete,
        "complete clip_type should propagate to BetaCrownConfig"
    );
}

#[test]
fn lsnc_relu_stays_relaxed_while_cersyve_is_complete() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let lsnc_relu = load_preset(&repo_root.join("configs/vnncomp25/lsnc_relu.yaml")).unwrap();
    let cersyve = load_preset(&repo_root.join("configs/vnncomp25/cersyve.yaml")).unwrap();

    assert_eq!(
        lsnc_relu.bab.clip.clip_type.as_deref(),
        None,
        "lsnc_relu should rely on the default relaxed clip type"
    );
    assert_eq!(
        cersyve.bab.clip.clip_type.as_deref(),
        Some("complete"),
        "cersyve should keep its explicit complete clip type"
    );

    let mut lsnc_config = BetaCrownConfig::default();
    apply_preset(&mut lsnc_config, &lsnc_relu).unwrap();
    assert_eq!(
        lsnc_config.input_clip_type,
        InputClipType::Relaxed,
        "lsnc_relu should stay on the relaxed runtime clip path"
    );

    let mut cersyve_config = BetaCrownConfig::default();
    apply_preset(&mut cersyve_config, &cersyve).unwrap();
    assert_eq!(
        cersyve_config.input_clip_type,
        InputClipType::Complete,
        "cersyve should stay on the complete runtime clip path"
    );
}

#[test]
fn nn4sys_preset_matches_reference_input_split_schedule_4354() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/nn4sys.yaml")).unwrap();

    // pgd_order: skip — PGD disabled for nn4sys (Part of #4354)
    assert_eq!(preset.attack.pgd_order.as_deref(), Some("skip"));
    assert_eq!(
        preset.bab.phase_budget.post_bab_pgd_fraction,
        Some(0.0),
        "disabled NN4SYS PGD must not retain an engine tail reservation"
    );
    assert_eq!(
        preset.bab.phase_budget.vnncomp_post_bab_attack,
        Some(false),
        "the independent VNN-COMP post-BaB wrapper must follow the explicit proof-only policy"
    );
    assert_eq!(preset.bab.clip.relaxed, Some(true));
    assert_eq!(preset.bab.branching.method.as_deref(), Some("input"));
    assert_eq!(preset.bab.branching.input_split.enable, Some(true));
    assert_eq!(preset.bab.branching.input_split.sb_coeff_thresh, Some(0.1));
    assert_eq!(preset.bab.branching.input_split.reorder_bab, Some(true));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(matches!(
        config.branching_heuristic,
        BranchingHeuristic::InputSplit
    ));
    assert!(
        config.enable_relaxed_clip,
        "nn4sys preset should propagate relaxed clipping into BetaCrownConfig"
    );
    assert_eq!(config.input_clip_type, InputClipType::Relaxed);
    assert_eq!(config.input_split_coeff_thresh, 0.1);
    assert!(config.reorder_bab);
    assert_eq!(
        config.phase_budget.post_bab_pgd_fraction, 0.0,
        "the applied NN4SYS engine schedule must give its full slice to proof"
    );
}

/// #nn4sys-seb-dark (B8): the shipped nn4sys preset arms Saturation-Escape
/// Branching through the TYPED key, so the dual-pool input brancher (and the
/// disjunctive precheck budget cap that funds it) is reachable without any
/// ambient `NY_SAT_ESCAPE_BRANCH` export — while a preset that never names the
/// key keeps the dark default byte-identically.
#[test]
fn nn4sys_preset_threads_sat_escape_branch_and_absence_stays_dark() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    // Key parses from the shipped nn4sys preset. It is `false`: `2e4af94b`
    // DISARMED it on a scored-wall-time A/B over the first 12 `_dual` rows —
    // armed proved 1/12 unsat, unarmed 8/12, so arming LOSES 7 rows. Arming
    // caps the per-clause precheck slice at `.min(0.5)` where the unarmed path
    // takes `.max(0.95)` (verify/disjunctive.rs:549-558), and on the mscn dual
    // band that precheck lane is what closes the disjuncts.
    //
    // This test previously asserted `Some(true)` and went stale the moment the
    // preset was corrected — it was pinning an intent the measurement refuted.
    // What is worth pinning is the WIRING in both directions, which is what it
    // now does.
    let nn4sys = load_preset(&repo_root.join("configs/vnncomp25/nn4sys.yaml")).unwrap();
    assert_eq!(
        nn4sys.bab.branching.input_split.sat_escape_branch,
        Some(false),
        "nn4sys deliberately DISARMS SEB (2e4af94b); NY_SAT_ESCAPE_BRANCH=1 still \
         forces it for experiments"
    );

    // …and an explicit `false` threads through to the engine field as disarmed.
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &nn4sys).unwrap();
    assert!(
        !config.sat_escape_branch,
        "an explicit `false` must leave BetaCrownConfig::sat_escape_branch disarmed"
    );

    // Absent key = today's behavior: another input-split preset without the
    // key leaves the field at its dark default.
    let acasxu = load_preset(&repo_root.join("configs/vnncomp25/acasxu_2023.yaml")).unwrap();
    assert_eq!(acasxu.bab.branching.input_split.sat_escape_branch, None);
    let mut acasxu_config = BetaCrownConfig::default();
    apply_preset(&mut acasxu_config, &acasxu).unwrap();
    assert!(
        !acasxu_config.sat_escape_branch,
        "a preset that does not name the key must stay byte-identical (dark default)"
    );
}

#[test]
fn cersyve_and_lsnc_relu_presets_propagate_reference_input_split_tuning() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    let cersyve = load_preset(&repo_root.join("configs/vnncomp25/cersyve.yaml")).unwrap();
    assert_eq!(cersyve.bab.branching.input_split.enable, Some(true));
    assert_eq!(
        cersyve.bab.branching.input_split.sb_coeff_thresh,
        Some(1.0e-2)
    );
    assert_eq!(cersyve.bab.branching.input_split.sb_sum, Some(true));
    assert_eq!(
        cersyve.bab.branching.input_split.touch_zero_score,
        Some(0.1)
    );
    assert_eq!(
        cersyve.bab.branching.input_split.override_parallel,
        Some(true)
    );

    let lsnc_relu = load_preset(&repo_root.join("configs/vnncomp25/lsnc_relu.yaml")).unwrap();
    assert_eq!(lsnc_relu.bab.branching.input_split.enable, Some(true));
    assert_eq!(
        lsnc_relu.bab.branching.input_split.sb_coeff_thresh,
        Some(1.0e-2)
    );
    assert_eq!(lsnc_relu.bab.branching.input_split.sb_sum, Some(true));
    assert_eq!(
        lsnc_relu.bab.branching.input_split.touch_zero_score,
        Some(0.1)
    );
    assert_eq!(lsnc_relu.bab.branching.input_split.override_parallel, None);

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &cersyve).unwrap();
    assert!(config.input_split_sb_sum);
    assert_eq!(config.input_split_coeff_thresh, 1.0e-2);
    assert_eq!(config.input_split_touch_zero_score, 0.1);
    assert!(config.input_split_override_parallel);
}

#[test]
fn cersyve_and_lsnc_relu_presets_enable_pgd_alpha_scale() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    for preset_name in ["cersyve", "lsnc_relu"] {
        let preset =
            load_preset(&repo_root.join(format!("configs/vnncomp25/{preset_name}.yaml"))).unwrap();
        let mut config = BetaCrownConfig::default();
        apply_preset(&mut config, &preset).unwrap();
        assert_eq!(
            config.pgd_optimizer,
            PgdOptimizer::SignedGradient,
            "{preset_name} should switch PGD to signed-gradient when pgd_alpha_scale=true"
        );
        assert_eq!(
            config.pgd_alpha_mode,
            PgdAlphaMode::InputRangeScaled(0.01),
            "{preset_name} should propagate pgd_alpha_scale into the runtime PGD alpha mode"
        );
    }
}

#[test]
fn cora_2024_preset_enables_clip_in_alpha_crown() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/cora_2024.yaml")).unwrap();

    assert_eq!(preset.bab.clip.in_alpha_crown, Some(true));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        config.clip_in_alpha_crown,
        "cora_2024 preset should propagate clip_in_alpha_crown into BetaCrownConfig"
    );
}

#[test]
fn cora_and_metaroom_presets_explicitly_disable_interm_transfer_4358() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cora = load_preset(&repo_root.join("configs/vnncomp25/cora_2024.yaml")).unwrap();
    let metaroom = load_preset(&repo_root.join("configs/vnncomp25/metaroom_2023.yaml")).unwrap();

    assert_eq!(cora.bab.interm_transfer, Some(false));
    assert_eq!(metaroom.bab.interm_transfer, Some(false));

    let mut cora_config = BetaCrownConfig::default();
    apply_preset(&mut cora_config, &cora).unwrap();
    assert!(
        !cora_config.enable_interm_transfer,
        "cora_2024 preset should override the enabled default and keep interm_transfer disabled"
    );

    let mut metaroom_config = BetaCrownConfig::default();
    apply_preset(&mut metaroom_config, &metaroom).unwrap();
    assert!(
        !metaroom_config.enable_interm_transfer,
        "metaroom_2023 preset should keep its explicit interm_transfer=false override"
    );
}

#[test]
fn metaroom_preset_propagates_dd_zonotope_admission_overrides() {
    // #metaroom-ddzono: the scored entry point sets no env vars, so the
    // category preset is the only way the dd-zonotope admission caps can
    // reach `DdZonoConfig` for metaroom's 5376-input / k=161 instances.
    //
    // MECHANISM test: an inline section must propagate every knob.
    let inline: PresetConfig = serde_yaml::from_str(
        r#"
dd_zonotope:
  min_input_numel: 5000
  max_k: 192
  max_generators: 4096
  interm_intersect: true
"#,
    )
    .expect("dd_zonotope preset section must parse");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &inline).unwrap();
    assert_eq!(config.dd_zonotope_min_input_numel, Some(5000));
    assert_eq!(config.dd_zonotope_max_k, Some(192));
    assert_eq!(config.dd_zonotope_max_generators, Some(4096));
    assert_eq!(config.dd_zonotope_collect_interm, Some(true));

    // SHIPPED-STATE test: the metaroom yaml keeps the section DISARMED until
    // the near-wall A/B (spec_idx_28-class rows vs the ~18s/row cost) and the
    // campaign-GPU interm-intersect value test are both recorded — the
    // population-evidence discipline. Re-point these at Some(..) ONLY together
    // with those measurements.
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let metaroom = load_preset(&repo_root.join("configs/vnncomp25/metaroom_2023.yaml")).unwrap();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &metaroom).unwrap();
    assert_eq!(config.dd_zonotope_min_input_numel, None);
    assert_eq!(config.dd_zonotope_max_k, None);
    assert_eq!(config.dd_zonotope_max_generators, None);
    assert_eq!(config.dd_zonotope_collect_interm, None);

    // Every other shipped preset leaves the section absent, which must remain
    // byte-identical to the built-in caps: apply changes nothing.
    let acasxu = load_preset(&repo_root.join("configs/vnncomp25/acasxu_2023.yaml")).unwrap();
    let mut untouched = BetaCrownConfig::default();
    apply_preset(&mut untouched, &acasxu).unwrap();
    assert_eq!(untouched.dd_zonotope_min_input_numel, None);
    assert_eq!(untouched.dd_zonotope_max_k, None);
    assert_eq!(untouched.dd_zonotope_max_generators, None);
    assert_eq!(untouched.dd_zonotope_collect_interm, None);
}

#[test]
fn cersyve_and_lsnc_relu_presets_propagate_min_batch_fill_ratio() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    let cersyve = load_preset(&repo_root.join("configs/vnncomp25/cersyve.yaml")).unwrap();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &cersyve).unwrap();
    assert_eq!(
        config.min_batch_fill_ratio, 0.1,
        "cersyve min_batch_fill_ratio should match reference (0.1)"
    );

    let lsnc_relu = load_preset(&repo_root.join("configs/vnncomp25/lsnc_relu.yaml")).unwrap();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &lsnc_relu).unwrap();
    assert_eq!(
        config.min_batch_fill_ratio, 0.0,
        "lsnc_relu min_batch_fill_ratio should match reference (0.0)"
    );
}

#[test]
fn acasxu_2023_preset_matches_hardcoded_config() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/acasxu_2023.yaml")).unwrap();

    // Verify YAML fields parsed correctly
    assert_eq!(
        preset.solver.bound_prop_method.as_deref(),
        Some("crown"),
        "acasxu preset should use plain CROWN (not alpha-CROWN)"
    );
    assert_eq!(preset.solver.batch_size, Some(16384));
    assert_eq!(preset.attack.pgd_order.as_deref(), Some("after"));
    assert_eq!(preset.attack.pgd_restarts, Some(10000));
    assert_eq!(preset.bab.branching.method, Some("input".to_string()));
    assert_eq!(preset.bab.branching.input_split.reorder_bab, Some(true));
    assert_eq!(preset.bab.clip.relaxed, Some(true));

    // Apply and verify config matches BetaCrownConfig::acas_xu()
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    let reference = BetaCrownConfig::acas_xu();
    assert_eq!(
        config.use_alpha_crown, reference.use_alpha_crown,
        "preset should disable alpha-CROWN like acas_xu() config"
    );
    assert_eq!(
        config.batch_size, reference.batch_size,
        "preset batch_size should match acas_xu() config"
    );
    assert_eq!(
        config.branching_heuristic, reference.branching_heuristic,
        "preset branching should match acas_xu() config"
    );
    assert_eq!(
        config.reorder_bab, reference.reorder_bab,
        "preset input-split reordering should match acas_xu() config"
    );
    assert_eq!(
        config.input_split_depth, reference.input_split_depth,
        "preset multi-dim input_split_depth should match acas_xu() config"
    );
    assert_eq!(
        config.enable_relaxed_clip, reference.enable_relaxed_clip,
        "preset relaxed_clip should match acas_xu() config"
    );
    assert_eq!(
        config.enable_pgd_attack, reference.enable_pgd_attack,
        "preset pgd_attack should match acas_xu() config"
    );
    assert_eq!(
        config.pgd_restarts, reference.pgd_restarts,
        "preset pgd_restarts should match acas_xu() config"
    );
}

/// vit_2023 regression pin: softmax "complex" stays DISABLED in the shipped
/// preset — measured net-negative at vit scale (00338ba) — while the Phase A
/// alpha/share_alphas knobs stay in place and apply_preset succeeds cleanly.
#[test]
fn vit_2023_preset_keeps_softmax_complex_disabled() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/vit_2023.yaml")).unwrap();

    // softmax: complex is deliberately NOT set — measured net-negative at vit
    // scale (00338ba: ~3x per-iteration cost, best bound flat while the
    // direct-LSE arm improves). The machinery stays in-tree; the preset
    // re-enables it only after the recorded follow-ups land.
    assert_eq!(
        preset.solver.alpha_crown.softmax.as_deref(),
        None,
        "vit_2023 preset keeps softmax-complex disabled (measured net-negative; see 00338ba)"
    );
    assert_eq!(preset.solver.alpha_crown.iterations, Some(50));
    assert_eq!(preset.solver.alpha_crown.lr_alpha, Some(0.5));
    assert_eq!(preset.solver.alpha_crown.share_alphas, Some(true));

    // The softmax field is consumed at model load (graph rewrite), not by
    // BetaCrownConfig; applying the preset must still succeed cleanly.
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(config.alpha_config.iterations, 50);
    assert_eq!(config.alpha_config.learning_rate, 0.5);
}

/// cgan_2023 root-collection scheduling caps (#4413, #cgan-bn11-budget):
/// the preset's per-node CROWN-IBP cap and aggregate α reference-refresh cap
/// flow through to the exact typed configs consumed by the collectors. The
/// per-node floor and refresh fraction retain their built-in defaults.
#[test]
fn cgan_2023_preset_raises_crown_ibp_per_node_cap() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    let load_config =
        build_onnx_load_config(&preset).expect("official cGAN loader policy must be admitted");
    assert_eq!(
        load_config.batch_norm_folding_policy(),
        ny_onnx::BatchNormFoldingPolicy::PreserveRaw,
        "official cGAN verdicts must target the authored graph"
    );
    assert!(
        load_config.require_authored_float32_initializers(),
        "official cGAN loading must fail before verification on rewritten FLOAT initializers"
    );
    assert_eq!(
        preset.model.forward_linear_spec_alpha,
        Some(false),
        "the candidate remains typed and dark pending the physical moat"
    );

    assert_eq!(preset.bab.crown_ibp_per_node_cap_secs, Some(150.0));
    assert_eq!(preset.bab.crown_ibp_per_node_floor_secs, None);
    assert_eq!(preset.bab.branching.input_split.reorder_bab, Some(false));
    assert_eq!(preset.bab.branching.input_split.warm_parallel, None);
    assert_eq!(preset.bab.branching.input_split.alpha_iteration, Some(1));
    assert_eq!(preset.bab.branching.input_split.lr_alpha, Some(0.05));
    assert_eq!(preset.bab.alpha_crown.reference_refresh_max_secs, Some(12));
    assert_eq!(preset.bab.alpha_crown.reference_refresh_fraction, None);
    assert_eq!(
        preset
            .bab
            .alpha_crown
            .forward_linear_deadline_fallback_to_ibp,
        Some(true)
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(config.crown_ibp_per_node_cap_secs, Some(150.0));
    assert_eq!(config.crown_ibp_per_node_floor_secs, None);
    assert!(!config.reorder_bab);
    assert!(!config.input_split_warm_parallel);
    assert_eq!(config.input_split_alpha_iteration, 1);
    assert_eq!(config.input_split_lr_alpha, 0.05);
    assert_eq!(config.alpha_config.reference_refresh_max_secs, Some(12));
    assert!(config.alpha_config.forward_linear_deadline_fallback_to_ibp);
    assert_eq!(
        config.alpha_config.reference_refresh_fraction,
        ny_propagate::AlphaCrownConfig::DEFAULT_REFERENCE_REFRESH_FRACTION
    );

    let budget = config.crown_ibp_per_node_time_budget();
    assert_eq!(budget.cap_secs, Some(150.0));
    assert_eq!(budget.floor_secs, None);

    // Companion phase-budget levers: retained SAT instances fall to PGD in
    // under 2 s, while the official holdout otherwise spent 85.1 s before BaB
    // and reserved another 10% for an unsuccessful post-BaB attack.
    assert_eq!(config.phase_budget.disjunctive_pgd_fraction, 0.10);
    assert_eq!(config.phase_budget.disjunctive_pgd_max_secs, Some(5));
    assert_eq!(config.phase_budget.post_bab_pgd_fraction, 0.0);
    assert_eq!(config.phase_budget.initial_bounds_fraction, 0.45);
    // The current forward-linear precheck returns as soon as its shared map is
    // available, so this remains a ceiling rather than spent time.
    assert_eq!(config.phase_budget.disjunctive_precheck_fraction, 0.85);
}

/// The forward-alpha treatment is a typed, reviewable preset rather than an
/// environment overlay. Normalize its single treatment field and require the
/// full serialized presets to match, so sealed A/B runs cannot accidentally
/// drift a loader, attack, or solver knob.
#[test]
fn cgan_forward_alpha_surrogate_canary_differs_by_exactly_one_typed_field() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let production = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    let canary = load_preset(
        &repo_root.join("configs/vnncomp25/cgan_2023_forward_alpha_surrogate_canary.yaml"),
    )
    .unwrap();

    assert_eq!(production.model.forward_linear_spec_alpha, Some(false));
    assert_eq!(canary.model.forward_linear_spec_alpha, Some(true));

    for (name, preset) in [("production", &production), ("canary", &canary)] {
        let load_config = build_onnx_load_config(preset)
            .unwrap_or_else(|error| panic!("{name} cGAN loader policy must be admitted: {error}"));
        assert_eq!(
            load_config.batch_norm_folding_policy(),
            ny_onnx::BatchNormFoldingPolicy::PreserveRaw,
            "{name} cGAN preset must verify the authored graph"
        );
        assert!(
            load_config.require_authored_float32_initializers(),
            "{name} cGAN preset must reject rewritten authored FLOAT initializers"
        );
    }

    let mut normalized_canary = canary;
    normalized_canary.model.forward_linear_spec_alpha = production.model.forward_linear_spec_alpha;
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&production).unwrap(),
        "forward-alpha canary may differ from shipped cGAN only at forward_linear_spec_alpha"
    );
}

#[test]
fn cgan_2026_preset_preserves_authored_graph_and_keeps_surrogate_dark() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp26/cgan2026.yaml")).unwrap();
    let load_config = build_onnx_load_config(&preset).expect("official 2026 cGAN loader policy");
    assert_eq!(
        load_config.batch_norm_folding_policy(),
        ny_onnx::BatchNormFoldingPolicy::PreserveRaw
    );
    assert!(load_config.require_authored_float32_initializers());
    assert_eq!(preset.model.forward_linear_spec_alpha, Some(false));
}

/// The warm-parallel route is intentionally absent from the shipped cGAN
/// preset. Its canary changes exactly the reordered-loop and scoped activation
/// fields, so a row-7 serial/parallel A/B cannot drift any other solver knob.
/// The target-complete root collector remains a sidecar experiment. Keep its
/// treatment surface attributable: it may receive more initial-bound time,
/// request exact root intermediates, and arm the new typed policy, but must
/// preserve every other production cGAN setting.
#[test]
fn cgan_sparse_target_complete_canary_differs_by_exactly_three_knobs() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let production = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    let canary = load_preset(
        &repo_root.join("configs/vnncomp25/cgan_2023_sparse_target_complete_canary.yaml"),
    )
    .unwrap();

    assert_eq!(
        production.bab.phase_budget.initial_bounds_fraction,
        Some(0.45)
    );
    assert_eq!(canary.bab.phase_budget.initial_bounds_fraction, Some(0.85));
    assert_eq!(production.bab.alpha_crown.fix_interm_bounds, None);
    assert_eq!(canary.bab.alpha_crown.fix_interm_bounds, Some(false));
    assert_eq!(
        production.bab.alpha_crown.cgan_sparse_target_complete_root,
        None
    );
    assert_eq!(
        canary.bab.alpha_crown.cgan_sparse_target_complete_root,
        Some(true)
    );
    assert_eq!(
        canary
            .bab
            .alpha_crown
            .forward_linear_deadline_fallback_to_ibp,
        Some(true),
        "the canary must preserve production's certified deadline fallback"
    );

    let mut normalized_canary = canary;
    normalized_canary.bab.phase_budget.initial_bounds_fraction =
        production.bab.phase_budget.initial_bounds_fraction;
    normalized_canary.bab.alpha_crown.fix_interm_bounds =
        production.bab.alpha_crown.fix_interm_bounds;
    normalized_canary
        .bab
        .alpha_crown
        .cgan_sparse_target_complete_root =
        production.bab.alpha_crown.cgan_sparse_target_complete_root;
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&production).unwrap(),
        "canary may differ only at initial_bounds_fraction, fix_interm_bounds, and \
         cgan_sparse_target_complete_root"
    );
}

/// cgan2026 resamples properties over the same seven ONNX models as cgan_2023.
/// Keep model-cost and sound-fallback scheduling levers aligned without
/// transferring any result-count claim between competitions.
#[test]
fn cgan2026_mirrors_byte_identical_model_schedule_levers() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cgan_2025 = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    let cgan_2026 = load_preset(&repo_root.join("configs/vnncomp26/cgan2026.yaml")).unwrap();

    assert_ne!(cgan_2026.general.root_path, cgan_2025.general.root_path);
    assert!(
        cgan_2026
            .general
            .root_path
            .as_deref()
            .is_some_and(|path| path.ends_with("cgan2026/1.0")),
        "the regular-track cgan2026 preset must name the selected VNN-LIB 1.0 manifest"
    );
    assert_eq!(cgan_2026.general.device.as_deref(), Some("wgpu"));
    assert_eq!(
        cgan_2026.bab.crown_ibp_per_node_cap_secs,
        cgan_2025.bab.crown_ibp_per_node_cap_secs
    );
    assert_eq!(
        serde_json::to_value(&cgan_2026.bab.phase_budget).unwrap(),
        serde_json::to_value(&cgan_2025.bab.phase_budget).unwrap(),
        "byte-identical model schedules must retain the proven phase ceilings"
    );
    assert_eq!(
        serde_json::to_value(&cgan_2026.bab.branching).unwrap(),
        serde_json::to_value(&cgan_2025.bab.branching).unwrap(),
        "byte-identical model schedules must retain the certified input-split options"
    );
    assert_eq!(
        cgan_2026.bab.alpha_crown.reference_refresh_max_secs,
        Some(12)
    );
    assert_eq!(cgan_2026.bab.alpha_crown.reference_refresh_fraction, None);
    assert_eq!(
        cgan_2026
            .bab
            .alpha_crown
            .forward_linear_deadline_fallback_to_ibp,
        Some(true)
    );
    assert_eq!(
        cgan_2026.bab.alpha_crown.cgan_complete_crown_ibp_root,
        cgan_2025.bab.alpha_crown.cgan_complete_crown_ibp_root,
        "byte-identical cGAN models must retain the certified complete-root policy"
    );

    let mut normalized_2026 = cgan_2026.clone();
    normalized_2026.general.root_path = cgan_2025.general.root_path.clone();
    normalized_2026.general.device = cgan_2025.general.device.clone();
    assert_eq!(
        serde_json::to_value(&normalized_2026).unwrap(),
        serde_json::to_value(&cgan_2025).unwrap(),
        "byte-identical cGAN presets may differ only in corpus path and selected device"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &cgan_2026).unwrap();
    assert_eq!(config.crown_ibp_per_node_cap_secs, Some(150.0));
    assert_eq!(config.phase_budget.disjunctive_pgd_fraction, 0.10);
    assert_eq!(config.phase_budget.disjunctive_pgd_max_secs, Some(5));
    assert_eq!(config.phase_budget.attack_extension_fraction, 0.0);
    assert_eq!(config.phase_budget.initial_bounds_fraction, 0.45);
    assert_eq!(config.phase_budget.disjunctive_precheck_fraction, 0.85);
    assert_eq!(config.phase_budget.post_bab_pgd_fraction, 0.0);
    assert_eq!(config.branching_heuristic, BranchingHeuristic::InputSplit);
    assert_eq!(config.fsb_candidates, 5);
    assert!(!config.reorder_bab);
    assert_eq!(config.input_split_alpha_iteration, 1);
    assert_eq!(config.input_split_lr_alpha, 0.05);
    assert!(config.input_split_ibp_enhancement);
    assert!(config.input_split_stacked_rebound);
    assert_eq!(config.alpha_config.reference_refresh_max_secs, Some(12));
    assert!(config.alpha_config.forward_linear_deadline_fallback_to_ibp);
    assert!(config.alpha_config.cgan_complete_crown_ibp_root);
}

/// The warm-parallel route is intentionally absent from the shipped cGAN
/// preset. Its canary changes exactly the measured reordered-loop batch and
/// scoped activation fields, so a row-7 A/B cannot drift any other solver
/// knob.
#[test]
fn cgan_warm_parallel_batch8_canary_is_scoped_and_differs_by_exactly_three_knobs() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let production = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    let canary = load_preset(
        &repo_root.join("configs/vnncomp25/cgan_2023_reorder_warm_parallel_canary.yaml"),
    )
    .unwrap();

    assert_eq!(
        production.bab.branching.input_split.reorder_bab,
        Some(false)
    );
    assert_eq!(production.bab.branching.input_split.warm_parallel, None);
    assert_eq!(production.bab.batch_size, Some(64));
    assert_eq!(canary.bab.batch_size, Some(8));
    assert_eq!(canary.bab.branching.input_split.reorder_bab, Some(true));
    assert_eq!(canary.bab.branching.input_split.warm_parallel, Some(true));

    let mut normalized_canary = canary.clone();
    normalized_canary.bab.batch_size = production.bab.batch_size;
    normalized_canary.bab.branching.input_split.reorder_bab =
        production.bab.branching.input_split.reorder_bab;
    normalized_canary.bab.branching.input_split.warm_parallel =
        production.bab.branching.input_split.warm_parallel;
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&production).unwrap(),
        "canary may differ only at batch_size, reorder_bab, and warm_parallel"
    );

    let mut production_config = BetaCrownConfig::default();
    apply_preset(&mut production_config, &production).unwrap();
    let mut canary_config = BetaCrownConfig::default();
    apply_preset(&mut canary_config, &canary).unwrap();
    assert!(!production_config.reorder_bab);
    assert!(!production_config.input_split_warm_parallel);
    assert_eq!(production_config.batch_size, 64);
    assert!(canary_config.reorder_bab);
    assert!(canary_config.input_split_warm_parallel);
    assert_eq!(canary_config.batch_size, 8);
}

/// This default-off canary is a seven-knob translation of alpha-beta-CROWN's
/// current cGAN recipe. Keep the treatment surface exact: any extra drift
/// would make a sealed row-7 A/B impossible to attribute.
#[test]
fn cgan_abcrown_parity_canary_differs_by_exactly_seven_knobs() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let production = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    let canary =
        load_preset(&repo_root.join("configs/vnncomp25/cgan_2023_abcrown_parity_canary.yaml"))
            .unwrap();

    assert_eq!(production.general.device, None);
    assert_eq!(canary.general.device.as_deref(), Some("wgpu"));
    assert_eq!(production.general.conv_mode, None);
    assert_eq!(
        canary.general.conv_mode,
        Some(ny_propagate::ConvMode::Matrix)
    );
    assert_eq!(production.solver.bound_prop_method, None);
    assert_eq!(canary.solver.bound_prop_method.as_deref(), Some("crown"));
    assert_eq!(production.attack.pgd_restarts, Some(200));
    assert_eq!(canary.attack.pgd_restarts, Some(100));
    assert_eq!(production.bab.branching.input_split.sb_coeff_thresh, None);
    assert_eq!(
        canary.bab.branching.input_split.sb_coeff_thresh,
        Some(1.0e-2)
    );
    assert_eq!(
        production.bab.branching.input_split.reorder_bab,
        Some(false)
    );
    assert_eq!(canary.bab.branching.input_split.reorder_bab, Some(true));
    assert_eq!(production.bab.branching.input_split.adv_check, None);
    // Sentinel semantics are intentionally counterintuitive: zero enables the
    // periodic SAT-only probe immediately, while -1 disables it.
    assert_eq!(canary.bab.branching.input_split.adv_check, Some(0));

    let mut normalized_canary = canary.clone();
    normalized_canary.general.device = production.general.device.clone();
    normalized_canary.general.conv_mode = production.general.conv_mode;
    normalized_canary.solver.bound_prop_method = production.solver.bound_prop_method.clone();
    normalized_canary.attack.pgd_restarts = production.attack.pgd_restarts;
    normalized_canary.bab.branching.input_split.sb_coeff_thresh =
        production.bab.branching.input_split.sb_coeff_thresh;
    normalized_canary.bab.branching.input_split.reorder_bab =
        production.bab.branching.input_split.reorder_bab;
    normalized_canary.bab.branching.input_split.adv_check =
        production.bab.branching.input_split.adv_check;
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&production).unwrap(),
        "alpha-beta-CROWN parity canary may differ from shipped cGAN only at its seven declared treatment knobs"
    );

    let mut production_config = BetaCrownConfig::default();
    apply_preset(&mut production_config, &production).unwrap();
    let mut canary_config = BetaCrownConfig::default();
    apply_preset(&mut canary_config, &canary).unwrap();

    assert_eq!(production_config.conv_mode, ny_propagate::ConvMode::Auto);
    assert!(production_config.use_alpha_crown);
    assert_eq!(production_config.pgd_restarts, 200);
    assert_eq!(production_config.input_split_coeff_thresh, 1.0e-3);
    assert!(!production_config.reorder_bab);
    assert_eq!(production_config.adv_check, -1);

    assert_eq!(canary_config.conv_mode, ny_propagate::ConvMode::Matrix);
    assert!(!canary_config.use_alpha_crown);
    assert_eq!(canary_config.pgd_restarts, 100);
    assert_eq!(canary_config.input_split_coeff_thresh, 1.0e-2);
    assert!(canary_config.reorder_bab);
    assert_eq!(canary_config.adv_check, 0);
}

/// The default-off generator-prefix treatment must be a pure routing delta
/// from the alpha-beta-CROWN parity canary: explicit ReLU splitting replaces
/// explicit input splitting, and no input-split-only tuning leaks into the
/// treatment. Its sidecar filename also keeps normal cgan_2023 preset
/// discovery from selecting it.
#[test]
fn cgan_generator_relu_phase_split_canary_is_explicit_and_scoped() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let parity_path = repo_root.join("configs/vnncomp25/cgan_2023_abcrown_parity_canary.yaml");
    let canary_path =
        repo_root.join("configs/vnncomp25/cgan_2023_generator_relu_phase_split_canary.yaml");

    assert_ne!(
        canary_path.file_name().and_then(|name| name.to_str()),
        Some("cgan_2023.yaml"),
        "default category discovery must not silently select the canary"
    );

    let parity = load_preset(&parity_path).unwrap();
    let canary = load_preset(&canary_path).unwrap();
    assert_eq!(parity.bab.branching.method.as_deref(), Some("input"));
    assert_eq!(canary.bab.branching.method.as_deref(), Some("relu"));
    assert_eq!(
        serde_json::to_value(&canary.bab.branching.input_split).unwrap(),
        serde_json::to_value(InputSplitPreset::default()).unwrap(),
        "ReLU treatment must not retain inapplicable input-split tuning"
    );

    let resolved = resolve_branching(&canary)
        .unwrap()
        .expect("explicit relu method must resolve");
    assert_eq!(resolved.heuristic, BranchingHeuristic::LargestBoundWidth);
    assert!(
        resolved.use_relu_split,
        "explicit relu method must route to ReLU splitting, not auto-selection"
    );

    let mut normalized_canary = canary.clone();
    normalized_canary.bab.branching.method = parity.bab.branching.method.clone();
    normalized_canary.bab.branching.input_split = parity.bab.branching.input_split.clone();
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&parity).unwrap(),
        "generator-prefix canary may differ from the parity canary only at the explicit branching route and removed input-split stanza"
    );

    assert_eq!(canary.general.device.as_deref(), Some("wgpu"));
    assert_eq!(
        canary.general.conv_mode,
        Some(ny_propagate::ConvMode::Matrix)
    );
    assert_eq!(canary.solver.bound_prop_method.as_deref(), Some("crown"));
    assert_eq!(canary.attack.pgd_restarts, Some(100));
    assert_eq!(canary.bab.batch_size, Some(64));
    assert_eq!(canary.bab.phase_budget.disjunctive_pgd_fraction, Some(0.10));
    assert_eq!(canary.bab.phase_budget.attack_extension_fraction, Some(0.0));
    assert_eq!(canary.bab.phase_budget.initial_bounds_fraction, Some(0.45));
    assert_eq!(
        canary.bab.phase_budget.disjunctive_precheck_fraction,
        Some(0.85)
    );
    assert_eq!(canary.bab.clip.relaxed, Some(true));
    assert_eq!(canary.bab.clip.relaxed_iterations, Some(20));
    assert_eq!(canary.bab.alpha_crown.lr_alpha, Some(0.2));
    assert_eq!(canary.bab.alpha_crown.iterations, Some(100));
    assert_eq!(canary.bab.beta_crown.lr_alpha, Some(0.1));
    assert_eq!(canary.bab.beta_crown.lr_beta, Some(0.15));
    assert_eq!(canary.bab.beta_crown.iterations, Some(25));
}

/// The alpha-beta-CROWN VGG recipe is a default-off canary. Keep NY's shipped
/// WGPU/root phase-budget tuning and isolate exactly the five treatment knobs.
#[test]
fn vgg_abcrown_parity_canary_is_default_off_and_scoped() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let production = load_preset(&repo_root.join("configs/vnncomp25/vggnet16_2022.yaml")).unwrap();
    let canary =
        load_preset(&repo_root.join("configs/vnncomp25/vggnet16_2022_abcrown_parity_canary.yaml"))
            .unwrap();

    assert_eq!(production.model.vgg_abcrown_treatment, None);
    assert_eq!(canary.model.vgg_abcrown_treatment, Some(true));
    assert_eq!(
        canary.solver.bound_prop_method.as_deref(),
        Some("forward+backward")
    );
    assert_eq!(canary.bab.branching.method.as_deref(), Some("sb"));
    assert_eq!(canary.bab.branching.input_split.enable, Some(true));
    assert_eq!(canary.attack.pgd_order.as_deref(), Some("input_bab"));

    let mut normalized_canary = canary.clone();
    normalized_canary.model.vgg_abcrown_treatment = production.model.vgg_abcrown_treatment;
    normalized_canary.solver.bound_prop_method = production.solver.bound_prop_method.clone();
    normalized_canary.bab.branching.method = production.bab.branching.method.clone();
    normalized_canary.bab.branching.input_split.enable =
        production.bab.branching.input_split.enable;
    normalized_canary.attack.pgd_order = production.attack.pgd_order.clone();
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&production).unwrap(),
        "VGG canary may differ from production only at its five treatment fields"
    );

    let mut production_config = BetaCrownConfig::default();
    apply_preset(&mut production_config, &production).unwrap();
    let mut canary_config = BetaCrownConfig::default();
    apply_preset(&mut canary_config, &canary).unwrap();

    assert!(
        production_config.use_alpha_crown,
        "shipped VGG root optimizations remain unchanged"
    );
    assert!(!production_config.use_forward_bounds);
    assert_eq!(
        production_config.branching_heuristic,
        BranchingHeuristic::LargestBoundWidth
    );

    assert!(!canary_config.use_alpha_crown);
    assert!(canary_config.use_forward_bounds);
    assert_eq!(
        canary_config.branching_heuristic,
        BranchingHeuristic::InputSplit
    );
    assert!(canary_config.enable_pgd_attack);
    assert_eq!(canary_config.batch_size, production_config.batch_size);
    assert_eq!(
        serde_json::to_value(&canary_config.phase_budget).unwrap(),
        serde_json::to_value(&production_config.phase_budget).unwrap(),
        "existing root phase budgets must survive the treatment"
    );
}

/// The production YOLO preset deliberately omits a branching method. Once its
/// loaded-model dimensions reach auto-resolution, that must select kFSB and
/// carry the typed critical-row policy into the final verifier config. The
/// companion ny-propagate selector test proves this config bit reaches the real
/// scorer with `NY_MO_SCORER_FIX` absent.
#[test]
fn yolo_2023_preset_auto_kfsb_arms_typed_critical_row_policy() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/yolo_2023.yaml")).unwrap();
    assert_eq!(
        preset.bab.branching.method, None,
        "YOLO must reach model-aware auto resolution"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        !config.use_multi_objective_critical_kfsb,
        "the reusable preset alone must not globally arm the costly scorer"
    );

    let resolved = resolve_auto_branching(
        AutoBranchingRequest {
            mip_complete_verifier: false,
            spec_input_count: Some(8112),
        },
        ModelStructure {
            param_count: 5_000_000,
            has_conv: true,
            relu_node_count: 40,
            is_dag: true,
        },
        8112,
    );
    assert_eq!(resolved.heuristic, BranchingHeuristic::Kfsb);
    assert_eq!(resolved.reason, AutoBranchingReason::HighDimOrManyRelu);
    assert!(resolved.use_multi_objective_critical_kfsb);

    // Mirrors the two runtime stamps in handle_beta_crown_command after preset
    // application and auto resolution.
    config.branching_heuristic = resolved.heuristic.clone();
    apply_resolved_auto_branching_runtime_policy(&mut config, &resolved);
    assert_eq!(config.branching_heuristic, BranchingHeuristic::Kfsb);
    assert!(config.use_multi_objective_critical_kfsb);
}

/// yolo_2023 pins the historical 12-second base cap after an official 2026 row
/// showed that every useful target completed within 8.715 seconds while two
/// later targets consumed 51.772 and 31.142 seconds before falling back to IBP.
#[test]
fn yolo_2023_preset_caps_doomed_crown_ibp_targets() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/yolo_2023.yaml")).unwrap();

    assert_eq!(preset.bab.crown_ibp_per_node_cap_secs, Some(12.0));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(config.crown_ibp_per_node_cap_secs, Some(12.0));
    assert_eq!(config.crown_ibp_per_node_time_budget().cap_secs, Some(12.0));
}

/// Presets that do not opt in carry no per-node overrides: the collector keeps
/// the 2.0-second floor and adaptive remaining-budget cap (e.g. cersyve,
/// dist_shift).
#[test]
fn presets_without_crown_ibp_budget_knob_keep_defaults() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for name in ["cersyve.yaml", "dist_shift_2023.yaml"] {
        let preset = load_preset(&repo_root.join("configs/vnncomp25").join(name)).unwrap();
        assert_eq!(
            preset.bab.crown_ibp_per_node_cap_secs, None,
            "{name} must not set the cgan-only cap knob"
        );
        assert_eq!(preset.bab.crown_ibp_per_node_floor_secs, None);

        let mut config = BetaCrownConfig::default();
        apply_preset(&mut config, &preset).unwrap();
        let budget = config.crown_ibp_per_node_time_budget();
        assert_eq!(budget, ny_propagate::CrownIbpPerNodeTimeBudget::default());
    }
}

/// TEETH for #reclaim-pgd on relusplitter.
///
/// The disjunctive PGD lane on this benchmark is RESTART-bound, not
/// fraction-bound: `pgd_restarts: 50` x the default `pgd_steps: 50` x 9 clauses
/// exhausts restarts before the fractional deadline is reached (measured
/// 2026-08-01: 11.5-27.5 s used of a 30 s/90 s slice on GT-unsat oval21 rows). So
/// `disjunctive_pgd_fraction` is the WRONG control here and an ABSOLUTE cap is the
/// right one.
///
/// Deleting the preset's cap fails the first assertion. "Fixing" it by lowering the
/// fraction instead fails the second. Reaching for any other phase knob — in
/// particular one that shortens a BOUND phase, which could change a verdict — fails
/// the whole-struct comparison, which is the assertion that actually pins the blast
/// radius.
#[test]
fn relusplitter_caps_the_restart_bound_disjunctive_pgd_phase() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/relusplitter.yaml")).unwrap();

    assert_eq!(
        preset.bab.phase_budget.disjunctive_pgd_max_secs,
        Some(5),
        "relusplitter must cap the restart-bound disjunctive PGD phase in ABSOLUTE \
         seconds; the reclaimed budget flows to BaB via the ledger's remaining()"
    );
    assert_eq!(
        preset.bab.phase_budget.disjunctive_pgd_fraction, None,
        "the fraction is measured INERT on relusplitter (the lane never reaches its \
         fractional deadline) — it must not be used as the control here"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    // SCOPE, asserted as a whole-struct diff: the preset moves EXACTLY one phase
    // budget field. Every other phase — initial bounds, reduced verification, MIP,
    // precheck, post-BaB — keeps its sealed default, so no bound-producing phase is
    // shortened and no verdict can move. A cap on an ATTACK phase can only lose a
    // counterexample (`sat` -> `unknown`, a completeness cost); it can never mint an
    // `unsat`, because no `unsat` is ever concluded from a failed attack.
    let expected = ny_propagate::PhaseBudgetConfig {
        disjunctive_pgd_max_secs: Some(5),
        ..Default::default()
    };
    assert_eq!(
        config.phase_budget, expected,
        "relusplitter's phase budget must differ from the sealed defaults in exactly \
         disjunctive_pgd_max_secs"
    );
}

/// Collins' absolute cap suppresses an optional forward-linear warmer whose
/// cold build dominates the small CNN proof. The cap is schedule-only: neither
/// its attack work nor its optional bound-cache warmer can grant uncertified
/// authority, and the whole-struct assertion pins every other proof/attack
/// phase to its sealed default.
#[test]
fn collins_rul_caps_the_disjunctive_prewave_without_fraction_drift() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset =
        load_preset(&repo_root.join("configs/vnncomp25/collins_rul_cnn_2022.yaml")).unwrap();

    assert_eq!(
        preset.bab.phase_budget.disjunctive_pgd_max_secs,
        Some(15),
        "Collins must keep the measured absolute prewave ceiling at every timeout tier"
    );
    assert_eq!(
        preset.bab.phase_budget.disjunctive_pgd_fraction, None,
        "a fractional override grows at long tiers and can re-admit the optional warmer"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    let expected = ny_propagate::PhaseBudgetConfig {
        disjunctive_pgd_max_secs: Some(15),
        ..Default::default()
    };
    assert_eq!(
        config.phase_budget, expected,
        "Collins may change only the attack prewave's absolute ceiling"
    );
}

/// The cap is SCOPED: presets that never measured the restart-bound lane must keep
/// the pure-fraction behaviour, so landing relusplitter's cap cannot silently
/// shorten falsification on an unrelated benchmark.
#[test]
fn disjunctive_pgd_cap_stays_scoped_to_presets_that_measured_it() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for name in ["acasxu_2023.yaml", "cersyve.yaml", "vit_2023.yaml"] {
        let preset = load_preset(&repo_root.join("configs/vnncomp25").join(name)).unwrap();
        assert_eq!(
            preset.bab.phase_budget.disjunctive_pgd_max_secs, None,
            "{name} must keep the pure-fraction disjunctive PGD behaviour"
        );

        let mut config = BetaCrownConfig::default();
        apply_preset(&mut config, &preset).unwrap();
        assert_eq!(
            config.phase_budget.disjunctive_pgd_max_secs, None,
            "{name} must not inherit relusplitter's absolute cap"
        );
    }
}
