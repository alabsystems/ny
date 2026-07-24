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
use ny_propagate::{
    BetaCrownConfig, BranchingHeuristic, InputClipType, PgdAlphaMode, PgdOptimizer,
};
use std::path::Path;

#[test]
fn cifar100_2024_arms_bounded_dense_head_crown_only_in_its_preset() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cifar = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    assert_eq!(cifar.bab.root_crown_interm_dense_head, Some(true));
    assert_eq!(cifar.bab.root_crown_interm_max_secs, Some(2));
    assert_eq!(cifar.bab.root_crown_interm_max_dim, Some(512));

    let mut cifar_config = BetaCrownConfig::default();
    apply_preset(&mut cifar_config, &cifar).unwrap();
    assert!(cifar_config.root_crown_interm_dense_head);
    assert_eq!(cifar_config.root_crown_interm_max_secs, 2);
    assert_eq!(cifar_config.root_crown_interm_max_dim, 512);

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

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &cersyve).unwrap();
    assert!(config.input_split_sb_sum);
    assert_eq!(config.input_split_coeff_thresh, 1.0e-2);
    assert_eq!(config.input_split_touch_zero_score, 0.1);
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

/// cgan_2023 per-node CROWN-IBP time-budget cap (#4413, #cgan-bn11-budget):
/// the preset's `bab.crown_ibp_per_node_cap_secs: 150` flows through
/// `apply_preset` into `BetaCrownConfig` and out through
/// `crown_ibp_per_node_time_budget()` — the exact value the verifier stamps
/// onto the `GraphNetwork` for the collector's budget computation. The floor
/// stays unset (built-in 2.0 s).
#[test]
fn cgan_2023_preset_raises_crown_ibp_per_node_cap() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();

    assert_eq!(preset.bab.crown_ibp_per_node_cap_secs, Some(150.0));
    assert_eq!(preset.bab.crown_ibp_per_node_floor_secs, None);
    assert_eq!(preset.bab.branching.input_split.reorder_bab, Some(false));
    assert_eq!(preset.bab.branching.input_split.warm_parallel, None);
    assert_eq!(preset.bab.branching.input_split.alpha_iteration, Some(5));
    assert_eq!(preset.bab.branching.input_split.lr_alpha, Some(0.05));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(config.crown_ibp_per_node_cap_secs, Some(150.0));
    assert_eq!(config.crown_ibp_per_node_floor_secs, None);
    assert!(!config.reorder_bab);
    assert!(!config.input_split_warm_parallel);
    assert_eq!(config.input_split_alpha_iteration, 5);
    assert_eq!(config.input_split_lr_alpha, 0.05);

    let budget = config.crown_ibp_per_node_time_budget();
    assert_eq!(budget.cap_secs, Some(150.0));
    assert_eq!(budget.floor_secs, None);

    // Companion phase-budget levers (same lever chain, existing plumbing):
    // small upfront disjunctive PGD (the sat cgan instances fall to PGD in
    // <2 s) and a warmup fraction large enough for the repeated ~55-93 s
    // CROWN-IBP collections in the alpha warmup.
    assert_eq!(config.phase_budget.disjunctive_pgd_fraction, 0.10);
    assert_eq!(config.phase_budget.initial_bounds_fraction, 0.45);
    // #cgan-collection-cache companion: a precheck slice large enough that the
    // equal-share per-node budget lets BN_11 (~95 s chunked backward) finish,
    // so the FIRST collection is complete and the input-keyed cache serves it
    // to the alpha warmup / BaB bootstrap instead of ~80 s re-collections.
    // 0.85 (was 0.75): headroom for thermally-throttled runs where BN_11's
    // ~110 s share at 0.75 was measured too tight (share ~125 s at 0.85).
    assert_eq!(config.phase_budget.disjunctive_precheck_fraction, 0.85);
}

/// The warm-parallel route is intentionally absent from the shipped cGAN
/// preset. Its canary changes exactly the reordered-loop and scoped activation
/// fields, so a row-7 serial/parallel A/B cannot drift any other solver knob.
#[test]
fn cgan_warm_parallel_canary_is_scoped_and_differs_by_exactly_two_knobs() {
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
    assert_eq!(canary.bab.branching.input_split.reorder_bab, Some(true));
    assert_eq!(canary.bab.branching.input_split.warm_parallel, Some(true));

    let mut normalized_canary = canary.clone();
    normalized_canary.bab.branching.input_split.reorder_bab =
        production.bab.branching.input_split.reorder_bab;
    normalized_canary.bab.branching.input_split.warm_parallel =
        production.bab.branching.input_split.warm_parallel;
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&production).unwrap(),
        "canary preset may differ from shipped cGAN only at reorder_bab and warm_parallel"
    );

    let mut production_config = BetaCrownConfig::default();
    apply_preset(&mut production_config, &production).unwrap();
    let mut canary_config = BetaCrownConfig::default();
    apply_preset(&mut canary_config, &canary).unwrap();
    assert!(!production_config.reorder_bab);
    assert!(!production_config.input_split_warm_parallel);
    assert!(canary_config.reorder_bab);
    assert!(canary_config.input_split_warm_parallel);
}

/// This default-off canary is a six-knob translation of alpha-beta-CROWN's
/// current cGAN recipe. Keep the treatment surface exact: any extra drift
/// would make a sealed row-7 A/B impossible to attribute.
#[test]
fn cgan_abcrown_parity_canary_differs_by_exactly_six_knobs() {
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

    let mut normalized_canary = canary.clone();
    normalized_canary.general.device = production.general.device.clone();
    normalized_canary.general.conv_mode = production.general.conv_mode;
    normalized_canary.solver.bound_prop_method = production.solver.bound_prop_method.clone();
    normalized_canary.attack.pgd_restarts = production.attack.pgd_restarts;
    normalized_canary.bab.branching.input_split.sb_coeff_thresh =
        production.bab.branching.input_split.sb_coeff_thresh;
    normalized_canary.bab.branching.input_split.reorder_bab =
        production.bab.branching.input_split.reorder_bab;
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&production).unwrap(),
        "alpha-beta-CROWN parity canary may differ from shipped cGAN only at its six declared treatment knobs"
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

    assert_eq!(canary_config.conv_mode, ny_propagate::ConvMode::Matrix);
    assert!(!canary_config.use_alpha_crown);
    assert_eq!(canary_config.pgd_restarts, 100);
    assert_eq!(canary_config.input_split_coeff_thresh, 1.0e-2);
    assert!(canary_config.reorder_bab);
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

/// Default = old constants (#4413): a config untouched by any preset carries
/// no per-node budget overrides, so the collector keeps the built-in
/// 2.0 s floor / 12.0 s cap byte-identically (e.g. cersyve, dist_shift).
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
