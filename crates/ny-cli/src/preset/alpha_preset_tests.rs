// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::apply::apply_preset;
use super::*;
use ny_propagate::BetaCrownConfig;

/// Verify that solver.alpha-crown.full_conv_alpha flows through to
/// AlphaCrownConfig. The reference cifar100 config sets this to false
/// to enable channel-shared alpha (63x fewer parameters). #4404.
#[test]
fn apply_preset_maps_full_conv_alpha_into_alpha_config_4404() {
    let preset = PresetConfig {
        solver: SolverPreset {
            alpha_crown: AlphaCrownPreset {
                full_conv_alpha: Some(false),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = BetaCrownConfig::default();
    assert!(
        config.alpha_config.full_conv_alpha,
        "default should be true (per-neuron alpha)"
    );
    apply_preset(&mut config, &preset).expect("full_conv_alpha preset should apply");
    assert!(
        !config.alpha_config.full_conv_alpha,
        "full_conv_alpha should be false after preset application"
    );
}
