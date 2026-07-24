// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{BetaCrownConfig, PyBranchingHeuristic, PyKfsbReduceOp};

/// Round-trip all 44 Python-surface fields through to_rust() + from_rust().
///
/// Every field is set to a non-default value so that a silently-dropped field
/// produces a mismatch. PartialEq on BetaCrownConfig catches any field
/// difference, including fields added in the future (they will cause a compile
/// error in this struct literal until the test is updated).
#[test]
fn test_beta_crown_config_round_trip_preserves_python_surface() {
    // Construct with ALL fields explicitly non-default.
    let config = BetaCrownConfig {
        max_domains: 777,
        timeout_secs: 42,
        max_depth: 13,
        use_alpha_crown: false,
        use_forward_bounds: true,
        use_crown_ibp: true,
        branching: PyBranchingHeuristic::GenBaB,
        fsb_candidates: 11,
        kfsb_reduce_op: PyKfsbReduceOp::Mean,
        beta_lr: 0.125,
        beta_iterations: 99,
        beta_tolerance: 0.5,
        alpha_lr: 0.25,
        batch_size: 7,
        enable_cuts: true,
        max_cuts: 555,
        enable_proactive_cuts: true,
        max_proactive_cuts: 77,
        enable_biccos_constraint_strengthening: true,
        biccos_drop_ratio: 0.75,
        enable_biccos_cold_start: true,
        biccos_min_verified: 17,
        biccos_min_verified_rate: 0.125,
        biccos_verified_rate_window: 33,
        biccos_min_cuts: 9,
        biccos_min_bound_gain: 0.0625,
        biccos_bound_gain_window: 44,
        biccos_cold_max_iters: 55,
        biccos_cut_window: 66,
        biccos_min_cut_yield: 0.1875,
        biccos_cut_yield_window: 22,
        biccos_cut_yield_patience: 4,
        verify_upper_bound: true,
        enable_pgd_attack: true,
        pgd_restarts: 88,
        pgd_steps: 33,
        enable_relaxed_clip: true,
        relaxed_clip_iterations: 3,
        enable_interm_transfer: true,
        enable_clip_interm_domain: true,
        clip_interm_topk: 5,
        clip_in_alpha_crown: true,
        clip_interm_prune: true,
        clip_interm_use_final_layer: true,
    };

    let round_trip = BetaCrownConfig::from_rust(&config.to_rust().unwrap());

    assert_eq!(config, round_trip);
}

/// validate_inner() rejects NaN in each float field without needing pyo3. (#2899, #3305)
#[test]
fn test_beta_crown_config_validate_rejects_nan() {
    #[allow(clippy::type_complexity)] // Table-driven test: complex type is clearer than type alias
    let fields: &[(&str, fn(&mut BetaCrownConfig))] = &[
        ("beta_lr", |c| c.beta_lr = f32::NAN),
        ("alpha_lr", |c| c.alpha_lr = f32::NAN),
        ("beta_tolerance", |c| c.beta_tolerance = f32::NAN),
        ("biccos_drop_ratio", |c| c.biccos_drop_ratio = f32::NAN),
        ("biccos_min_verified_rate", |c| {
            c.biccos_min_verified_rate = f32::NAN
        }),
        ("biccos_min_bound_gain", |c| {
            c.biccos_min_bound_gain = f32::NAN
        }),
        ("biccos_min_cut_yield", |c| {
            c.biccos_min_cut_yield = f32::NAN
        }),
    ];

    for (name, mutate) in fields {
        let mut config = BetaCrownConfig::new();
        mutate(&mut config);
        let msg = config
            .validate_inner()
            .expect_err(&format!("{name} NaN should be rejected"));
        assert!(
            msg.contains(name),
            "Error for {name} should mention field name: {msg}",
        );
    }
}

/// validate() rejects Inf in float fields. (#2899)
#[test]
fn test_beta_crown_config_validate_rejects_inf() {
    let mut config = BetaCrownConfig::new();
    config.beta_lr = f32::INFINITY;
    config
        .validate()
        .expect_err("Inf beta_lr should be rejected");

    let mut config = BetaCrownConfig::new();
    config.alpha_lr = f32::NEG_INFINITY;
    config
        .validate()
        .expect_err("NEG_INFINITY alpha_lr should be rejected");
}

/// validate() rejects negative values in float fields. (#2899)
#[test]
fn test_beta_crown_config_validate_rejects_negative() {
    let mut config = BetaCrownConfig::new();
    config.beta_lr = -0.01;
    config
        .validate()
        .expect_err("negative beta_lr should be rejected");

    let mut config = BetaCrownConfig::new();
    config.biccos_drop_ratio = -1.0;
    config
        .validate()
        .expect_err("negative biccos_drop_ratio should be rejected");
}

/// to_rust() returns Err for invalid configs. (#2899)
#[test]
fn test_beta_crown_config_to_rust_rejects_invalid() {
    let mut config = BetaCrownConfig::new();
    config.beta_lr = f32::NAN;
    config
        .to_rust()
        .expect_err("to_rust should reject NaN beta_lr");
}

/// Default config passes validation. (#2899)
#[test]
fn test_beta_crown_config_default_validates() {
    let config = BetaCrownConfig::new();
    config.validate().expect("default config should be valid");
}

#[test]
fn test_beta_crown_config_validate_rejects_forward_alpha_combo_4354() {
    let mut config = BetaCrownConfig::new();
    config.use_forward_bounds = true;
    let err = config
        .validate_inner()
        .expect_err("forward+alpha combo should be rejected");
    assert!(
        err.contains("use_forward_bounds"),
        "combo validation should mention use_forward_bounds: {err}"
    );
}
