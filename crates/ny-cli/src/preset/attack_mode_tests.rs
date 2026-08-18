// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for attack_mode preset parsing (#1449).

use super::apply::apply_preset;
use super::*;
use ny_propagate::{BetaCrownConfig, PgdInitialization};
use std::path::Path;

#[test]
fn vnncomp_relational_exact_gradient_opt_in_is_typed_default_off_and_closed() {
    let omitted: PresetConfig = serde_yaml::from_str("attack: {}\n").expect("empty attack map");
    assert_eq!(
        omitted.attack.vnncomp_upfront_relational_exact_gradient, None,
        "omission must preserve the default-off wrapper route"
    );

    let enabled: PresetConfig =
        serde_yaml::from_str("attack:\n  vnncomp_upfront_relational_exact_gradient: true\n")
            .expect("typed wrapper opt-in");
    assert_eq!(
        enabled.attack.vnncomp_upfront_relational_exact_gradient,
        Some(true)
    );

    let error = serde_yaml::from_str::<PresetConfig>(
        "attack:\n  vnncomp_upfront_relational_exact_gradients: true\n",
    )
    .expect_err("near-miss key must be rejected by the closed attack schema");
    assert!(
        error.to_string().contains("unknown field"),
        "closed-schema error should identify an unknown field: {error}"
    );
}

/// #1449: attack_mode 'diversed_PGD' maps to PgdInitialization::Osi.
#[test]
fn apply_preset_attack_mode_diversed_pgd_sets_osi_1449() {
    let preset = PresetConfig {
        attack: AttackPreset {
            attack_mode: Some("diversed_PGD".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(
        config.pgd_initialization,
        PgdInitialization::Osi,
        "diversed_PGD should map to Osi initialization"
    );
}

/// #1449: attack_mode 'PGD' maps to PgdInitialization::Uniform.
#[test]
fn apply_preset_attack_mode_pgd_sets_uniform_1449() {
    let preset = PresetConfig {
        attack: AttackPreset {
            attack_mode: Some("PGD".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(
        config.pgd_initialization,
        PgdInitialization::Uniform,
        "PGD should map to Uniform initialization"
    );
}

/// #1449: attack_mode 'diversed_GAMA_PGD' maps to OSI initialization + GAMA.
#[test]
fn apply_preset_attack_mode_gama_sets_osi_and_gama_1449() {
    let preset = PresetConfig {
        attack: AttackPreset {
            attack_mode: Some("diversed_GAMA_PGD".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(
        config.pgd_initialization,
        PgdInitialization::Osi,
        "diversed_GAMA_PGD should map to Osi initialization"
    );
    assert!(
        config.pgd_gama,
        "diversed_GAMA_PGD should enable the GAMA guidance loss"
    );
}

/// #1449: plain modes leave the GAMA guidance loss off.
#[test]
fn apply_preset_attack_mode_diversed_pgd_leaves_gama_off_1449() {
    let preset = PresetConfig {
        attack: AttackPreset {
            attack_mode: Some("diversed_PGD".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        !config.pgd_gama,
        "diversed_PGD must not enable the GAMA guidance loss"
    );
}

/// #1449: pgd_gama resolves to Some(GAMA_LAMBDA_DEFAULT) in the runtime attack config.
#[test]
fn pgd_gama_wires_to_pgd_attack_config_gama_lambda_1449() {
    let config = BetaCrownConfig {
        pgd_gama: true,
        ..BetaCrownConfig::default()
    };
    let pgd = config.pgd_attack_config(1, 1, None);
    assert_eq!(
        pgd.gama_lambda,
        Some(ny_propagate::GAMA_LAMBDA_DEFAULT),
        "pgd_gama=true must set PgdConfig.gama_lambda to the default λ₀"
    );
    let off = BetaCrownConfig::default().pgd_attack_config(1, 1, None);
    assert_eq!(
        off.gama_lambda, None,
        "pgd_gama=false must leave PgdConfig.gama_lambda unset"
    );
}

/// #1449: unknown attack_mode is rejected with a helpful message.
#[test]
fn apply_preset_attack_mode_unknown_rejected_1449() {
    let preset = PresetConfig {
        attack: AttackPreset {
            attack_mode: Some("fgsm".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    let err = apply_preset(&mut config, &preset).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("fgsm"),
        "error should mention the unknown mode: {msg}"
    );
}

/// #1449: osi_steps preset field wires through to BetaCrownConfig.pgd_osi_steps.
#[test]
fn apply_preset_osi_steps_wires_to_config_1449() {
    let preset = PresetConfig {
        attack: AttackPreset {
            attack_mode: Some("diversed_PGD".to_string()),
            osi_steps: Some(50),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(config.pgd_osi_steps, 50, "osi_steps should wire through");
    assert_eq!(config.pgd_initialization, PgdInitialization::Osi);
}

/// #surrogate-sign / #dense-sweep: the new attack keys wire preset →
/// BetaCrownConfig → PgdConfig, and stay off by default.
#[test]
fn apply_preset_surrogate_sign_and_dense_sweep_wire_to_pgd_config() {
    let preset = PresetConfig {
        attack: AttackPreset {
            surrogate_sign_gradient: Some(true),
            dense_low_dim_sweep: Some(true),
            dense_sweep_max_dims: Some(2),
            dense_sweep_points: Some(1234),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    let pgd = config.pgd_attack_config(1, 1, None);
    assert!(
        pgd.surrogate_sign_gradient,
        "surrogate key must wire through"
    );
    assert!(pgd.dense_low_dim_sweep, "dense sweep key must wire through");
    assert_eq!(pgd.dense_sweep_max_dims, 2);
    assert_eq!(pgd.dense_sweep_points, 1234);

    let off = BetaCrownConfig::default().pgd_attack_config(1, 1, None);
    assert!(
        !off.surrogate_sign_gradient && !off.dense_low_dim_sweep,
        "both features must default off"
    );
}

/// #surrogate-sign: traffic_signs_recognition_2023.yaml arms the STE
/// surrogate + diversed_GAMA_PGD (the falsifier lane for the BNN track).
#[test]
fn traffic_signs_preset_arms_surrogate_sign_gradient() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset =
        load_preset(&repo_root.join("configs/vnncomp25/traffic_signs_recognition_2023.yaml"))
            .unwrap();

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        config.pgd_surrogate_sign_gradient,
        "traffic_signs preset must arm the STE Sign surrogate"
    );
    assert!(
        config.pgd_gama,
        "traffic_signs preset must arm diversed_GAMA_PGD"
    );
    assert!(
        !config.pgd_dense_low_dim_sweep,
        "dense sweep must NOT be armed for traffic_signs"
    );
}

/// #1449: dist_shift_2023.yaml has attack_mode: diversed_PGD (reference parity).
#[test]
fn dist_shift_preset_has_diversed_pgd_attack_mode_1449() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/dist_shift_2023.yaml")).unwrap();

    assert_eq!(
        preset.attack.attack_mode.as_deref(),
        Some("diversed_PGD"),
        "dist_shift_2023 YAML must specify attack_mode: diversed_PGD"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(
        config.pgd_initialization,
        PgdInitialization::Osi,
        "dist_shift_2023 preset must resolve to OSI initialization"
    );
}

/// #1449: relusplitter.yaml has attack_mode: diversed_PGD that parses to Osi.
#[test]
fn relusplitter_preset_has_diversed_pgd_attack_mode_1449() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/relusplitter.yaml")).unwrap();

    assert_eq!(
        preset.attack.attack_mode.as_deref(),
        Some("diversed_PGD"),
        "relusplitter YAML must specify attack_mode: diversed_PGD"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(
        config.pgd_initialization,
        PgdInitialization::Osi,
        "relusplitter preset must resolve to OSI initialization"
    );
}
