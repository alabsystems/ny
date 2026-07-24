// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use ny_propagate::{BetaCrownConfig, ConvMode};

use super::{apply_preset, load_preset};

#[test]
fn relusplitter_preset_sets_reference_conv_mode_auto_3813() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/relusplitter.yaml"))
        .expect("relusplitter preset should load");

    assert_eq!(
        preset.general.conv_mode,
        Some(ConvMode::Auto),
        "#3813: relusplitter preset should declare general.conv_mode: auto"
    );

    let mut config = BetaCrownConfig {
        enable_cuts: true,
        ..Default::default()
    };
    apply_preset(&mut config, &preset).expect("preset application should succeed");

    assert_eq!(config.conv_mode, ConvMode::Auto);
    assert!(
        !config.use_patches(),
        "#3813: relusplitter preset auto conv_mode should resolve to matrix mode when cuts are enabled"
    );
}

#[test]
fn relusplitter_preset_applies_truncated_crown_backward_layers_3813() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/relusplitter.yaml"))
        .expect("relusplitter preset should load");

    assert_eq!(
        preset.bab.crown_backward_layers,
        Some(6),
        "#3813: relusplitter preset should declare bab.crown_backward_layers: 6"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("preset application should succeed");

    assert_eq!(
        config.crown_backward_layers,
        Some(6),
        "#3813: relusplitter preset must carry bab.crown_backward_layers into BetaCrownConfig"
    );
}
