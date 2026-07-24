// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{apply_preset, load_preset};
use ny_propagate::{BetaCrownConfig, BranchingHeuristic, ConvMode};
use std::path::Path;

/// Part of #3813: Stage 1 regression — matrix-mode input split preset contract.
///
/// Asserts the Stage 1 preset carries `conv_mode: matrix` into the no-cuts
/// input-split lane, matching the EXECUTE design requirement.
///
/// Design: designs/2026-03-14-issue-3813-ibp-first-input-split-alternative.md
#[test]
fn relusplitter_rsplitter_matrix_input_split_preset_contract_3813() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(
        &repo_root.join("configs/vnncomp25/relusplitter_rsplitter_matrix_input_split.yaml"),
    )
    .unwrap();

    assert_eq!(
        preset.general.conv_mode,
        Some(ConvMode::Matrix),
        "#3813: Stage 1 preset must force conv_mode=matrix"
    );
    assert_eq!(preset.attack.pgd_restarts, Some(300));
    assert_eq!(
        preset.solver.bound_prop_method.as_deref(),
        Some("crown"),
        "#3813: must use plain CROWN"
    );
    assert_eq!(preset.bab.branching.method.as_deref(), Some("sb"));
    assert_eq!(preset.bab.branching.input_split.enable, Some(true));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    assert_eq!(
        config.conv_mode,
        ConvMode::Matrix,
        "#3813: applied config must have matrix conv mode"
    );
    assert!(!config.enable_cuts, "#3813: graph input split forbids cuts");
    assert!(matches!(
        config.branching_heuristic,
        BranchingHeuristic::InputSplit
    ));
}
