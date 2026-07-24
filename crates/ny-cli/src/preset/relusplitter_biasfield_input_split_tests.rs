// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{apply_preset, load_preset};
use ny_propagate::{BetaCrownConfig, BranchingHeuristic};
use std::path::Path;

#[test]
fn relusplitter_biasfield_input_split_preset_matches_pilot_contract() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset =
        load_preset(&repo_root.join("configs/vnncomp25/relusplitter_biasfield_input_split.yaml"))
            .unwrap();

    assert_eq!(
        preset.solver.bound_prop_method.as_deref(),
        Some("crown"),
        "pilot preset should force plain CROWN for the base biasfield lane"
    );
    assert_eq!(preset.solver.batch_size, Some(4));
    assert_eq!(preset.attack.pgd_order.as_deref(), Some("after"));
    assert_eq!(preset.attack.pgd_restarts, Some(300));
    assert_eq!(preset.attack.pgd_steps, Some(20));
    assert_eq!(preset.bab.branching.method.as_deref(), Some("sb"));
    assert_eq!(preset.bab.branching.input_split.enable, Some(true));
    assert_eq!(preset.bab.branching.input_split.sb_coeff_thresh, Some(2.0));
    assert_eq!(
        preset.general.conv_mode, None,
        "pilot preset should stay on the reference plain-CROWN path without conv_mode overrides"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    assert!(
        !config.use_alpha_crown,
        "pilot preset should disable alpha-CROWN"
    );
    assert_eq!(config.batch_size, 4);
    assert!(matches!(
        config.branching_heuristic,
        BranchingHeuristic::InputSplit
    ));
    assert!(
        !config.enable_cuts,
        "pilot preset should not inherit relusplitter cuts"
    );
    assert!(
        config.enable_pgd_attack,
        "pilot preset should keep PGD enabled for the reference-style after-bounding probe"
    );
    assert_eq!(config.pgd_restarts, 300);
    assert_eq!(config.pgd_steps, 20);
    assert_eq!(config.input_split_coeff_thresh, 2.0);
}
