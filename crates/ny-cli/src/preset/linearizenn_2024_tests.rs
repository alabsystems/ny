// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{apply_preset, load_preset};
use ny_propagate::{BetaCrownConfig, BranchingHeuristic};
use std::{fs, path::Path};

#[test]
fn linearizenn_2024_preset_uses_shared_root_crown_and_fresh_child_ibp() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/linearizenn_2024.yaml")).unwrap();

    // Bounded per-iteration batch for the reordered input-split BaB
    // (#linearizenn-bab-reorder): 512 keeps the transient workset small while
    // saturating the CPU cores.
    assert_eq!(preset.solver.batch_size, Some(512));
    assert_eq!(
        preset.solver.bound_prop_method.as_deref(),
        Some("alpha-crown")
    );
    assert_eq!(preset.solver.alpha_crown.iterations, Some(0));
    assert_eq!(preset.attack.pgd_order.as_deref(), Some("after"));
    // 100 restarts (#linearizenn-attack-budget): the lone sat instance is
    // caught by the first restart; 10_000 burned fixed work on every unsat.
    assert_eq!(preset.attack.pgd_restarts, Some(100));
    assert_eq!(
        preset.bab.batch_size, None,
        "solver batch size must not be shadowed"
    );
    assert_eq!(preset.bab.branching.method.as_deref(), Some("input"));
    assert_eq!(preset.bab.branching.candidates, Some(10));
    assert_eq!(preset.bab.branching.input_split.enable, Some(true));
    assert_eq!(preset.bab.branching.input_split.ibp_enhancement, Some(true));
    assert_eq!(preset.bab.branching.input_split.stacked_rebound, Some(true));
    assert_eq!(preset.bab.branching.input_split.alpha_iteration, Some(0));
    assert_eq!(
        preset
            .bab
            .branching
            .input_split
            .independent_singleton_disjunction,
        Some(true)
    );
    assert_eq!(preset.bab.alpha_crown.lr_alpha, None);
    assert_eq!(preset.bab.alpha_crown.iterations, None);

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    assert!(config.use_alpha_crown);
    assert_eq!(config.alpha_config.iterations, 0);
    assert!(config.input_split_ibp_enhancement);
    assert!(config.input_split_stacked_rebound);
    assert_eq!(config.input_split_alpha_iteration, 0);
    assert_eq!(config.batch_size, 512);
    assert!(matches!(
        config.branching_heuristic,
        BranchingHeuristic::InputSplit
    ));
    assert!(config.input_split_independent_singleton_disjunction);
    assert!(
        config.enable_pgd_attack,
        "linearizenn preset should keep the competition wrapper's PGD lane enabled"
    );
    assert_eq!(config.pgd_restarts, 100);
    assert_eq!(config.alpha_lr, 0.1);
    assert_eq!(config.beta_lr, 0.15);
    assert_eq!(config.beta_iterations, 15);
}

#[test]
fn singleton_disjunction_domain_list_opt_in_is_default_off_and_unique() {
    assert!(
        !BetaCrownConfig::default().input_split_independent_singleton_disjunction,
        "the engine config must fail closed when the typed preset field is absent"
    );

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let config_dir = repo_root.join("configs/vnncomp25");
    let mut armed = Vec::new();
    let mut cgan_presets = 0usize;
    for entry in fs::read_dir(&config_dir).expect("read vnncomp25 configs") {
        let path = entry.expect("config entry").path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("yaml") {
            continue;
        }
        let preset = load_preset(&path).unwrap_or_else(|error| {
            panic!("{} must remain a typed preset: {error}", path.display())
        });
        let enabled = preset
            .bab
            .branching
            .input_split
            .independent_singleton_disjunction
            .unwrap_or(false);
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("UTF-8 preset filename");
        if filename.contains("cgan") {
            cgan_presets += 1;
            assert!(
                !enabled,
                "{filename} must never inherit the LinearizeNN singleton decomposition"
            );
        }
        if enabled {
            armed.push(filename.to_string());
        }
    }

    armed.sort();
    assert_eq!(
        armed,
        ["linearizenn_2024.yaml"],
        "the new treatment is initially isolated to the production LinearizeNN preset"
    );
    assert!(
        cgan_presets > 0,
        "the static census must cover cGAN presets"
    );
}

/// `bab.branching.input_split.lowdim_prover` is the typed opt-in for the
/// `#lowdim-prover` lane at the DomainList input-split ENTRY. Pin three things:
/// the engine default fails closed, the yaml key actually ARRIVES in the engine
/// config (a key that parses but never reaches `BetaCrownConfig` is dead
/// capability that passes every other test), and no shipped preset sets it
/// together with `independent_singleton_disjunction` — that per-clause route
/// already runs the lane, so both would try it twice per clause.
#[test]
fn lowdim_prover_key_arrives_defaults_off_and_never_doubles_the_singleton_route() {
    assert!(
        !BetaCrownConfig::default().input_split_lowdim_prover,
        "the engine config must fail closed when the typed preset field is absent"
    );

    let mut preset = super::PresetConfig::default();
    preset.bab.branching.input_split.enable = Some(true);
    preset.bab.branching.input_split.lowdim_prover = Some(true);
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        config.input_split_lowdim_prover,
        "bab.branching.input_split.lowdim_prover must reach BetaCrownConfig"
    );
    assert_eq!(BetaCrownConfig::default().lowdim_prover_share(), 0.5);
    let mut shared = super::PresetConfig::default();
    shared.bab.branching.input_split.lowdim_prover_share = Some(0.9);
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &shared).unwrap();
    assert!(
        (config.lowdim_prover_share() - 0.9).abs() < 1e-6,
        "share must arrive"
    );
    config.input_split_lowdim_prover_share = 7.0;
    assert_eq!(config.lowdim_prover_share(), 0.95, "share must clamp high");
    config.input_split_lowdim_prover_share = f32::NAN;
    assert_eq!(
        config.lowdim_prover_share(),
        0.5,
        "non-finite share reads as default"
    );

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for year in ["configs/vnncomp25", "configs/vnncomp26"] {
        let Ok(entries) = fs::read_dir(repo_root.join(year)) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("config entry").path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("yaml") {
                continue;
            }
            let preset = load_preset(&path).unwrap_or_else(|error| {
                panic!("{} must remain a typed preset: {error}", path.display())
            });
            let split = &preset.bab.branching.input_split;
            assert!(
                !(split.lowdim_prover.unwrap_or(false)
                    && split.independent_singleton_disjunction.unwrap_or(false)),
                "{} arms lowdim_prover AND the singleton route: the lane would run twice per clause",
                path.display()
            );
        }
    }
}

/// `bab.branching.input_split.lowdim_fold` arrives in `BetaCrownConfig` and
/// defaults off (the nn4sys per-clause fold pre-pass).
#[test]
fn lowdim_fold_key_arrives_and_defaults_off() {
    assert!(!BetaCrownConfig::default().input_split_lowdim_fold);
    let mut preset = super::PresetConfig::default();
    preset.bab.branching.input_split.lowdim_fold = Some(true);
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        config.input_split_lowdim_fold,
        "bab.branching.input_split.lowdim_fold must reach BetaCrownConfig"
    );
}
