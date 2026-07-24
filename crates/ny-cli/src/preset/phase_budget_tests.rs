// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Phase budget preset tests (#2206 Packet E).

use super::apply::apply_preset;
use super::load_preset;
use super::{BabPreset, PhaseBudgetPreset, PresetConfig};
use ny_propagate::BetaCrownConfig;
use std::path::Path;

fn assert_phase_budget_pair(name: &str, path: &str, initial_bounds_fraction: f32) {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join(path)).unwrap();
    assert_eq!(
        preset.bab.phase_budget.initial_bounds_fraction,
        Some(initial_bounds_fraction),
        "{name} should set initial_bounds_fraction to {initial_bounds_fraction}"
    );
    assert_eq!(
        preset.bab.phase_budget.post_bab_pgd_fraction,
        Some(0.10),
        "{name} should set post_bab_pgd_fraction to 0.10"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        (config.phase_budget.initial_bounds_fraction - initial_bounds_fraction).abs() < 1e-6,
        "{name} initial_bounds_fraction should propagate to config as {initial_bounds_fraction}"
    );
    assert!(
        (config.phase_budget.post_bab_pgd_fraction - 0.10).abs() < 1e-6,
        "{name} post_bab_pgd_fraction should propagate to config"
    );
}

/// #2206 Packet E: phase_budget preset overrides flow to BetaCrownConfig.
#[test]
fn apply_preset_propagates_phase_budget_overrides_2206() {
    let preset = PresetConfig {
        bab: BabPreset {
            phase_budget: PhaseBudgetPreset {
                initial_bounds_fraction: Some(0.15),
                upfront_pgd_fraction: Some(0.10),
                mip_min_secs: Some(10),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    assert!(
        (config.phase_budget.initial_bounds_fraction - 0.15).abs() < 1e-6,
        "initial_bounds_fraction should be overridden to 0.15"
    );
    assert!(
        (config.phase_budget.upfront_pgd_fraction - 0.10).abs() < 1e-6,
        "upfront_pgd_fraction should be overridden to 0.10"
    );
    assert_eq!(
        config.phase_budget.mip_min_secs, 10,
        "mip_min_secs should be overridden to 10"
    );
    // Unset fields stay at defaults.
    assert!(
        (config.phase_budget.reduced_verification_fraction - 0.40).abs() < 1e-6,
        "unset reduced_verification_fraction should remain at default 0.40"
    );
    assert!(
        (config.phase_budget.disjunctive_pgd_fraction - 0.50).abs() < 1e-6,
        "unset disjunctive_pgd_fraction should remain at default 0.50"
    );
}

/// #2206 Packet E: phase_budget YAML section deserializes correctly.
#[test]
fn phase_budget_yaml_deserialization_2206() {
    let yaml = r#"
bab:
  timeout: 100
  phase_budget:
    initial_bounds_fraction: 0.15
    upfront_pgd_fraction: 0.10
    reduced_verification_fraction: 0.30
"#;
    let preset: PresetConfig = serde_yaml::from_str(yaml).unwrap();

    assert_eq!(preset.bab.phase_budget.initial_bounds_fraction, Some(0.15));
    assert_eq!(preset.bab.phase_budget.upfront_pgd_fraction, Some(0.10));
    assert_eq!(
        preset.bab.phase_budget.reduced_verification_fraction,
        Some(0.30)
    );
    // Omitted fields remain None (default).
    assert!(preset.bab.phase_budget.disjunctive_pgd_fraction.is_none());

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        (config.phase_budget.initial_bounds_fraction - 0.15).abs() < 1e-6,
        "YAML initial_bounds_fraction should flow to config"
    );
    assert!(
        (config.phase_budget.reduced_verification_fraction - 0.30).abs() < 1e-6,
        "YAML reduced_verification_fraction should flow to config"
    );
}

/// #2206 Packet E: soundnessbench and cifar100_2024 presets cap initial_bounds_fraction.
#[test]
fn soundnessbench_and_cifar100_presets_cap_initial_bounds_fraction_2206() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    let soundnessbench =
        load_preset(&repo_root.join("configs/vnncomp25/soundnessbench.yaml")).unwrap();
    assert_eq!(
        soundnessbench.bab.phase_budget.initial_bounds_fraction,
        Some(0.15),
        "soundnessbench should cap initial_bounds_fraction to 0.15"
    );
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &soundnessbench).unwrap();
    assert!(
        (config.phase_budget.initial_bounds_fraction - 0.15).abs() < 1e-6,
        "soundnessbench initial_bounds_fraction should propagate to config"
    );

    let cifar100 = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    assert_eq!(
        cifar100.bab.phase_budget.initial_bounds_fraction,
        Some(0.15),
        "cifar100_2024 should cap initial_bounds_fraction to 0.15"
    );
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &cifar100).unwrap();
    assert!(
        (config.phase_budget.initial_bounds_fraction - 0.15).abs() < 1e-6,
        "cifar100_2024 initial_bounds_fraction should propagate to config"
    );
    assert_eq!(
        cifar100.bab.phase_budget.mip_min_fraction,
        Some(0.0),
        "cifar100_2024 must not reserve BaB time for its ineligible 44M-NNZ Graph-MIP"
    );
    assert_eq!(
        cifar100.bab.phase_budget.mip_min_secs,
        Some(0),
        "cifar100_2024 must not impose a hidden minimum MIP reservation"
    );
    assert_eq!(
        config.phase_budget.mip_min_fraction, 0.0,
        "the zero MIP fraction should propagate to the runtime ledger"
    );
    assert_eq!(
        config.phase_budget.mip_min_secs, 0,
        "the zero MIP floor should propagate to the runtime ledger"
    );
    assert!(
        !config.phase_budget.requests_mip_reservation(),
        "CIFAR's zero policy must decline both the time reserve and root-bounds stash"
    );
}

#[test]
fn nn4sys_and_safenlp_retain_nonzero_mip_reservation_admission() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for (name, path) in [
        ("nn4sys/MSCN", "configs/vnncomp25/nn4sys.yaml"),
        ("safenlp", "configs/vnncomp26/safenlp_2024.yaml"),
    ] {
        let preset = load_preset(&repo_root.join(path)).unwrap();
        let mut config = BetaCrownConfig::default();
        apply_preset(&mut config, &preset).unwrap();
        assert!(
            config.phase_budget.requests_mip_reservation(),
            "{name} must retain its existing whole-net MIP reservation and stash"
        );
    }
}

/// #2206 Packet E: post_bab_pgd_fraction flows through preset system.
#[test]
fn post_bab_pgd_fraction_preset_override_2206() {
    let preset = PresetConfig {
        bab: BabPreset {
            phase_budget: PhaseBudgetPreset {
                post_bab_pgd_fraction: Some(0.25),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let mut config = BetaCrownConfig::default();
    // Default is 0.10 (matches reference competition configs)
    assert!(
        (config.phase_budget.post_bab_pgd_fraction - 0.10).abs() < 1e-6,
        "default post_bab_pgd_fraction should be 0.10"
    );

    apply_preset(&mut config, &preset).unwrap();
    assert!(
        (config.phase_budget.post_bab_pgd_fraction - 0.25).abs() < 1e-6,
        "post_bab_pgd_fraction should be overridden to 0.25"
    );
}

/// #2206 Packet E: post_bab_pgd_fraction deserializes from YAML.
#[test]
fn post_bab_pgd_fraction_yaml_deserialization_2206() {
    let yaml = r#"
bab:
  phase_budget:
    initial_bounds_fraction: 0.15
    post_bab_pgd_fraction: 0.10
"#;
    let preset: PresetConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(preset.bab.phase_budget.post_bab_pgd_fraction, Some(0.10));
    assert_eq!(preset.bab.phase_budget.initial_bounds_fraction, Some(0.15));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        (config.phase_budget.post_bab_pgd_fraction - 0.10).abs() < 1e-6,
        "YAML post_bab_pgd_fraction should flow to config"
    );
}

/// #2206 Packet E: VNN-COMP presets set post_bab_pgd_fraction for PGD reservation.
#[test]
fn vnncomp_presets_set_post_bab_pgd_fraction_2206() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    for (name, path) in [
        ("soundnessbench", "configs/vnncomp25/soundnessbench.yaml"),
        ("cifar100_2024", "configs/vnncomp25/cifar100_2024.yaml"),
        (
            "tinyimagenet_2024",
            "configs/vnncomp25/tinyimagenet_2024.yaml",
        ),
    ] {
        let preset = load_preset(&repo_root.join(path)).unwrap();
        assert_eq!(
            preset.bab.phase_budget.post_bab_pgd_fraction,
            Some(0.10),
            "{name} should set post_bab_pgd_fraction to 0.10"
        );

        let mut config = BetaCrownConfig::default();
        apply_preset(&mut config, &preset).unwrap();
        assert!(
            (config.phase_budget.post_bab_pgd_fraction - 0.10).abs() < 1e-6,
            "{name} post_bab_pgd_fraction should propagate to config"
        );
    }
}

/// #2206 Packet E: tinyimagenet_2024 preset caps initial_bounds_fraction.
#[test]
fn tinyimagenet_preset_caps_initial_bounds_fraction_2206() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/tinyimagenet_2024.yaml")).unwrap();
    assert_eq!(
        preset.bab.phase_budget.initial_bounds_fraction,
        Some(0.15),
        "tinyimagenet_2024 should cap initial_bounds_fraction to 0.15"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        (config.phase_budget.initial_bounds_fraction - 0.15).abs() < 1e-6,
        "tinyimagenet_2024 initial_bounds_fraction should propagate to config"
    );
}

/// #2206 Packet E: cersyve presets keep the competitive phase-budget pair.
#[test]
fn cersyve_presets_keep_competitive_phase_budgets_2206() {
    for (name, path) in [
        ("cersyve", "configs/vnncomp25/cersyve.yaml"),
        ("cersyve_gpu_bab", "configs/vnncomp25/cersyve_gpu_bab.yaml"),
    ] {
        assert_phase_budget_pair(name, path, 0.15);
    }
}

/// #4283 regression: lsnc_relu disables the alpha-CROWN warmup cap in both the
/// canonical preset and the standalone GPU-BaB sidecar.
#[test]
fn lsnc_relu_presets_disable_warmup_cap_4283() {
    for (name, path) in [
        ("lsnc_relu", "configs/vnncomp25/lsnc_relu.yaml"),
        (
            "lsnc_relu_gpu_bab",
            "configs/vnncomp25/lsnc_relu_gpu_bab.yaml",
        ),
    ] {
        assert_phase_budget_pair(name, path, 1.0);
    }
}

/// #2206: default PhaseBudgetPreset (all None) does not alter config defaults.
#[test]
fn empty_phase_budget_preset_preserves_defaults_2206() {
    let preset = PresetConfig::default();
    let mut config = BetaCrownConfig::default();
    let before = config.phase_budget.clone();
    apply_preset(&mut config, &preset).unwrap();

    assert!(
        (config.phase_budget.initial_bounds_fraction - before.initial_bounds_fraction).abs() < 1e-6,
        "empty preset should not alter initial_bounds_fraction"
    );
    assert!(
        (config.phase_budget.upfront_pgd_fraction - before.upfront_pgd_fraction).abs() < 1e-6,
        "empty preset should not alter upfront_pgd_fraction"
    );
    assert_eq!(
        config.phase_budget.mip_min_secs, before.mip_min_secs,
        "empty preset should not alter mip_min_secs"
    );
    assert!(
        (config.phase_budget.post_bab_pgd_fraction - before.post_bab_pgd_fraction).abs() < 1e-6,
        "empty preset should not alter post_bab_pgd_fraction"
    );
}
