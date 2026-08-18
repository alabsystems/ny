// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{warm_start_alpha_config, BatchedScalarBounds};

#[test]
fn test_batched_scalar_bounds_field_roundtrip_4404() {
    let bounds = BatchedScalarBounds {
        lower_bounds: vec![-1.0_f32, 0.25],
        upper_bounds: vec![1.5_f32, 2.0],
        linear_bounds: vec![None, None],
    };

    assert_eq!(bounds.lower_bounds, vec![-1.0_f32, 0.25]);
    assert_eq!(bounds.upper_bounds, vec![1.5_f32, 2.0]);
    assert!(
        bounds.linear_bounds.iter().all(Option::is_none),
        "test carrier should preserve per-domain optional linear bounds"
    );
}

#[test]
fn child_warm_start_clears_root_only_cgan_collection_policy() {
    let mut config = crate::beta_crown::config::BetaCrownConfig::default();
    config.alpha_config.fix_interm_bounds = false;
    config.alpha_config.cgan_sparse_target_complete_root = true;
    config.alpha_config.cgan_complete_crown_ibp_root = true;
    config.input_split_alpha_iteration = 3;
    config.input_split_lr_alpha = 0.025;

    let child = warm_start_alpha_config(&config, None);
    assert!(
        child.fix_interm_bounds,
        "children must keep cheap intermediates"
    );
    assert!(
        !child.cgan_sparse_target_complete_root,
        "the root-only collector must not escape into BaB children"
    );
    assert!(
        !child.cgan_complete_crown_ibp_root,
        "the complete root-only collector must not escape into BaB children"
    );
    assert_eq!(child.iterations, 3);
    assert_eq!(child.learning_rate, 0.025);
}
