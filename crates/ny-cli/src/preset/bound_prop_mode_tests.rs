// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ny_propagate::{BetaCrownConfig, BranchingHeuristic};

/// Test that solver.bound_prop_method: "crown" disables alpha-CROWN.
/// This is critical for ACAS-Xu where frozen root alpha regresses to 0% (#3453).
#[test]
fn bound_prop_method_crown_disables_alpha() {
    let preset = PresetConfig {
        solver: SolverPreset {
            bound_prop_method: Some("crown".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };

    let mut config = BetaCrownConfig::default();
    assert!(
        config.use_alpha_crown,
        "default config should have alpha-CROWN enabled"
    );

    apply_preset(&mut config, &preset).unwrap();
    assert!(
        !config.use_alpha_crown,
        "bound_prop_method=crown should disable alpha-CROWN"
    );
}

/// Test that solver.bound_prop_method: "alpha-crown" enables alpha-CROWN.
#[test]
fn bound_prop_method_alpha_crown_enables_alpha() {
    let preset = PresetConfig {
        solver: SolverPreset {
            bound_prop_method: Some("alpha-crown".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };

    let mut config = BetaCrownConfig {
        use_alpha_crown: false,
        ..Default::default()
    };
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        config.use_alpha_crown,
        "bound_prop_method=alpha-crown should enable alpha-CROWN"
    );
    assert!(
        !config.use_forward_bounds,
        "bound_prop_method=alpha-crown should keep forward bounds disabled"
    );
}

#[test]
fn bound_prop_method_forward_crown_enables_forward_bounds_4354() {
    let preset = PresetConfig {
        solver: SolverPreset {
            bound_prop_method: Some("forward+crown".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };

    let mut config = BetaCrownConfig::default();
    assert!(
        !config.use_forward_bounds,
        "default config should have forward bounds disabled"
    );
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        config.use_forward_bounds,
        "forward+crown should enable forward bounds"
    );
    assert!(
        !config.use_alpha_crown,
        "forward+crown should disable alpha-CROWN"
    );
}

#[test]
fn bound_prop_method_forward_backward_alias_enables_forward_bounds_4354() {
    let preset = PresetConfig {
        solver: SolverPreset {
            bound_prop_method: Some("forward+backward".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(config.use_forward_bounds);
    assert!(!config.use_alpha_crown);
}

#[test]
fn unsupported_bound_prop_method_is_rejected() {
    let preset = PresetConfig {
        solver: SolverPreset {
            bound_prop_method: Some("dynamic-forward".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };

    let mut config = BetaCrownConfig::default();
    let err = apply_preset(&mut config, &preset).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("dynamic-forward"),
        "error should mention the unsupported method: {msg}"
    );
    assert!(
        msg.contains("forward+crown"),
        "error should describe the supported methods: {msg}"
    );
}

/// Test that bound_prop_method is parsed from YAML, including the hyphenated alias.
#[test]
fn bound_prop_method_yaml_parsing() {
    let yaml = r#"
solver:
  bound_prop_method: crown
  batch_size: 16384
bab:
  branching:
    method: input
"#;
    let preset: PresetConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(preset.solver.bound_prop_method.as_deref(), Some("crown"));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        !config.use_alpha_crown,
        "YAML bound_prop_method=crown should disable alpha-CROWN"
    );
    assert_eq!(config.batch_size, 16384);
    assert!(matches!(
        config.branching_heuristic,
        BranchingHeuristic::InputSplit
    ));
    assert!(
        !config.use_forward_bounds,
        "YAML bound_prop_method=crown should keep forward bounds disabled"
    );
}
