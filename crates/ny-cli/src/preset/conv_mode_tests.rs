// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use ny_propagate::{BetaCrownConfig, ConvMode};

use super::{apply_preset, load_preset};

#[test]
fn relusplitter_preset_keeps_matrix_mode_without_cut_authority_3813() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/relusplitter.yaml"))
        .expect("relusplitter preset should load");

    assert_eq!(
        preset.general.conv_mode,
        Some(ConvMode::Matrix),
        "#3813: relusplitter must retain matrix Conv2d throughput while cuts are quarantined"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("preset application should succeed");
    config
        .validate()
        .expect("the scored relusplitter preset must remain valid");

    assert_eq!(config.conv_mode, ConvMode::Matrix);
    assert!(!config.enable_cuts);
    assert!(!config.enable_near_miss_cuts);
    assert!(!config.enable_proactive_cuts);
    assert!(
        !config.use_patches(),
        "#3813: explicit matrix mode must preserve the measured Conv2d lane without cut authority"
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
