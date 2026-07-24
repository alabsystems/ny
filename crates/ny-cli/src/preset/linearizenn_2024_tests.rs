// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{apply_preset, load_preset};
use ny_propagate::{BetaCrownConfig, BranchingHeuristic};
use std::path::Path;

#[test]
fn linearizenn_2024_preset_matches_reference_crown_translation_4323() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/linearizenn_2024.yaml")).unwrap();

    // Bounded per-iteration batch for the reordered input-split BaB
    // (#linearizenn-bab-reorder): 512 keeps the transient workset small while
    // saturating the CPU cores.
    assert_eq!(preset.solver.batch_size, Some(512));
    assert_eq!(preset.solver.bound_prop_method.as_deref(), Some("crown"));
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
    assert_eq!(preset.bab.alpha_crown.lr_alpha, None);
    assert_eq!(preset.bab.alpha_crown.iterations, None);

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    assert!(
        !config.use_alpha_crown,
        "linearizenn preset should force plain CROWN"
    );
    assert_eq!(config.batch_size, 512);
    assert!(matches!(
        config.branching_heuristic,
        BranchingHeuristic::InputSplit
    ));
    assert!(
        config.enable_pgd_attack,
        "linearizenn preset should keep PGD enabled for the after-bounding pass"
    );
    assert_eq!(config.pgd_restarts, 100);
    assert_eq!(config.alpha_lr, 0.1);
    assert_eq!(config.beta_lr, 0.15);
    assert_eq!(config.beta_iterations, 15);
}
