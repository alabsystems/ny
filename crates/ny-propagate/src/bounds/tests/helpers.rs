// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for helper functions and config types.

use crate::bounds::{
    batched_matvec, safe_add_for_bounds, safe_add_for_bounds_with_polarity, safe_array_add,
    safe_mul_for_bounds, AlphaCrownConfig, GradientMethod, MultiSpecKeep, Optimizer,
};
use crate::InvpropConfig;
use ndarray::{ArrayD, IxDyn};

// =========================================================================
// safe_mul_for_bounds tests
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_safe_mul_for_bounds_normal() {
    assert_eq!(safe_mul_for_bounds(2.0, 3.0), 6.0);
    assert_eq!(safe_mul_for_bounds(-2.0, 3.0), -6.0);
    assert_eq!(safe_mul_for_bounds(0.5, 4.0), 2.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_safe_mul_for_bounds_zero() {
    // 0 * anything = 0, even infinity
    assert_eq!(safe_mul_for_bounds(0.0, f32::INFINITY), 0.0);
    assert_eq!(safe_mul_for_bounds(0.0, f32::NEG_INFINITY), 0.0);
    assert_eq!(safe_mul_for_bounds(f32::INFINITY, 0.0), 0.0);
    assert_eq!(safe_mul_for_bounds(f32::NEG_INFINITY, 0.0), 0.0);
    assert_eq!(safe_mul_for_bounds(0.0, 0.0), 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_safe_mul_for_bounds_nan() {
    // NaN propagates (indicates invalid bounds)
    assert!(safe_mul_for_bounds(f32::NAN, 1.0).is_nan());
    assert!(safe_mul_for_bounds(1.0, f32::NAN).is_nan());
}

#[ntest::timeout(5000)]
#[test]
fn test_safe_mul_for_bounds_infinity() {
    // Non-zero * infinity = infinity
    assert_eq!(safe_mul_for_bounds(2.0, f32::INFINITY), f32::INFINITY);
    assert_eq!(safe_mul_for_bounds(-2.0, f32::INFINITY), f32::NEG_INFINITY);
    assert_eq!(
        safe_mul_for_bounds(2.0, f32::NEG_INFINITY),
        f32::NEG_INFINITY
    );
}

// =========================================================================
// safe_add_for_bounds tests
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_safe_add_for_bounds_with_polarity_normal() {
    assert_eq!(safe_add_for_bounds_with_polarity(1.0, 2.0, true), 3.0);
    assert_eq!(safe_add_for_bounds_with_polarity(1.0, 2.0, false), 3.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_safe_add_for_bounds_with_polarity_inf_minus_inf() {
    // inf + (-inf) = NaN in standard math, but we use conservative bounds
    // For lower bound: use -inf (conservative)
    let result_lower = safe_add_for_bounds_with_polarity(f32::INFINITY, f32::NEG_INFINITY, true);
    assert_eq!(result_lower, f32::NEG_INFINITY);

    // For upper bound: use +inf (conservative)
    let result_upper = safe_add_for_bounds_with_polarity(f32::INFINITY, f32::NEG_INFINITY, false);
    assert_eq!(result_upper, f32::INFINITY);
}

#[ntest::timeout(5000)]
#[test]
fn test_safe_add_for_bounds_default() {
    // Default (no polarity) should use upper bound (conservative = +inf for NaN)
    let result = safe_add_for_bounds(f32::INFINITY, f32::NEG_INFINITY);
    assert_eq!(result, f32::INFINITY);
}

// =========================================================================
// safe_array_add tests
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_safe_array_add_normal() {
    let a = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0_f32, 2.0, 3.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0_f32, 5.0, 6.0]).unwrap();

    let result_lower = safe_array_add(&a, &b, true).unwrap();
    let result_upper = safe_array_add(&a, &b, false).unwrap();

    assert_eq!(result_lower.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    assert_eq!(result_upper.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
}

#[ntest::timeout(5000)]
#[test]
fn test_safe_array_add_with_inf() {
    let a = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, 1.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, 2.0]).unwrap();

    let result_lower = safe_array_add(&a, &b, true).unwrap();
    let result_upper = safe_array_add(&a, &b, false).unwrap();

    // First element: inf + (-inf) = -inf for lower, +inf for upper
    assert_eq!(result_lower[[0]], f32::NEG_INFINITY);
    assert_eq!(result_upper[[0]], f32::INFINITY);

    // Second element: 1 + 2 = 3
    assert_eq!(result_lower[[1]], 3.0);
    assert_eq!(result_upper[[1]], 3.0);
}

// =========================================================================
// batched_matvec tests
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_batched_matvec_simple() {
    // A: [1, 2, 3] matrix, x: [1, 3] vector
    let a =
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f32]).unwrap();
    let x = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 1.0, 1.0_f32]).unwrap();

    let result = batched_matvec(&a, &x, false).unwrap();

    // Row 0: 1+2+3 = 6
    // Row 1: 4+5+6 = 15
    assert_eq!(result.shape(), &[1, 2]);
    let slice = result.as_slice().unwrap();
    assert!((slice[0] - 6.0).abs() < 1e-6);
    assert!((slice[1] - 15.0).abs() < 1e-6);
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_matvec_with_inf_and_zero() {
    // Test 0 * inf = 0 handling
    let a = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![0.0, 1.0_f32]).unwrap();
    let x = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![f32::INFINITY, 2.0]).unwrap();

    let result = batched_matvec(&a, &x, false).unwrap();

    // 0 * inf + 1 * 2 = 0 + 2 = 2
    assert_eq!(result[[0, 0]], 2.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_matvec_batched() {
    // Batch of 2, each 2x2 matrix
    let a = ArrayD::from_shape_vec(
        IxDyn(&[2, 2, 2]),
        vec![
            1.0, 0.0, 0.0, 1.0, // Identity batch 0
            2.0, 0.0, 0.0, 2.0, // 2*Identity batch 1
        ],
    )
    .unwrap();
    let x = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![3.0, 4.0, 3.0, 4.0_f32]).unwrap();

    let result = batched_matvec(&a, &x, false).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    // Batch 0: I @ [3,4] = [3,4]
    assert!((result[[0, 0]] - 3.0).abs() < 1e-6);
    assert!((result[[0, 1]] - 4.0).abs() < 1e-6);
    // Batch 1: 2I @ [3,4] = [6,8]
    assert!((result[[1, 0]] - 6.0).abs() < 1e-6);
    assert!((result[[1, 1]] - 8.0).abs() < 1e-6);
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_matvec_single_batch() {
    // Single batch with identity-like matrix
    let a = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0, 0.0, 0.0, 1.0_f32]).unwrap();
    let x = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![3.0, 4.0_f32]).unwrap();

    let result = batched_matvec(&a, &x, false).unwrap();

    assert_eq!(result.shape(), &[1, 2]);
    assert!((result[[0, 0]] - 3.0).abs() < 1e-6);
    assert!((result[[0, 1]] - 4.0).abs() < 1e-6);
}

// =========================================================================
// GradientMethod tests
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_gradient_method_default() {
    let method = GradientMethod::default();
    // Default changed from SPSA to AnalyticChain: SPSA with 1 sample is
    // noise-dominated for networks with many unstable neurons. AnalyticChain computes
    // true chain-rule gradients matching reference α,β-CROWN's loss.backward().
    assert_eq!(method, GradientMethod::AnalyticChain);
}

#[ntest::timeout(5000)]
#[test]
fn test_gradient_method_equality() {
    assert_eq!(GradientMethod::Spsa, GradientMethod::Spsa);
    assert_eq!(
        GradientMethod::FiniteDifferences,
        GradientMethod::FiniteDifferences
    );
    assert_ne!(GradientMethod::Spsa, GradientMethod::FiniteDifferences);
}

#[ntest::timeout(5000)]
#[test]
fn test_gradient_method_clone() {
    // GradientMethod implements Copy, so we test the Clone impl via Copy behavior
    let method = GradientMethod::Spsa;
    let cloned: GradientMethod = method; // Uses Copy (which implies Clone)
    assert_eq!(method, cloned);
}

// =========================================================================
// AlphaCrownConfig tests
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_alpha_crown_config_default() {
    let config = AlphaCrownConfig::default();

    // Default now matches α,β-CROWN's proven configuration
    assert_eq!(config.iterations, 100);
    assert_eq!(config.learning_rate, 0.1); // α,β-CROWN default
    assert_eq!(config.lr_decay, 0.98); // α,β-CROWN ExponentialLR decay
    assert_eq!(config.tolerance, 1e-4);
    assert!(config.use_momentum);
    assert_eq!(config.momentum, 0.9);
    assert_eq!(config.gradient_method, GradientMethod::AnalyticChain);
    assert_eq!(config.spsa_samples, 1);
    assert!(config.fix_interm_bounds);
    assert!(!config.cgan_sparse_target_complete_root);
    assert!(!config.cgan_complete_crown_ibp_root);
    assert!(!config.cgan_crown_root_map);
    assert!(!config.cgan_collector_dense);
    assert!(!config.cgan_unstable_rows);
    assert_eq!(config.cgan_unstable_rows_max_frac, 0.9);
    assert_eq!(config.cgan_child_interm_refine_max_secs, None);
    // #cgan-child-seed: all three child-cascade keys default-dark.
    assert!(!config.cgan_child_interm_refine_seed_from_parent);
    assert_eq!(config.cgan_child_interm_refine_parallel, 0);
    assert!(!config.cgan_child_interm_refine_fresh_clip_map);
    // #cgan-child-zono: default-dark with a non-binding generator budget.
    assert!(!config.cgan_child_zono_refresh.enabled);
    assert_eq!(config.cgan_child_zono_refresh.max_generators, 16_384);
    assert_eq!(config.cgan_child_zono_refresh.max_secs, None);
    assert_eq!(config.sparse_ratio, 0.3);
    assert!(!config.adaptive_skip); // #3918: disabled — reference has no depth gate, uses early_stop_patience
    assert_eq!(config.adaptive_skip_depth_threshold, 20); // Retained for explicit opt-in
    assert!(!config.adaptive_skip_pilot); // #3298: disabled by default
    assert_eq!(config.pilot_improvement_threshold, 1e-3);
    assert_eq!(config.early_stop_patience, 10); // #3298: matches α,β-CROWN reference
                                                // Adam optimizer settings (ported from α,β-CROWN)
    assert_eq!(config.optimizer, Optimizer::Adam);
    assert_eq!(config.adam_beta1, 0.9);
    assert_eq!(config.adam_beta2, 0.999);
    assert_eq!(config.adam_epsilon, 1e-8);
    assert!(!config.pruning_in_iteration);
    assert_eq!(config.pruning_in_iteration_threshold, 0.2);
    assert_eq!(config.multi_spec_keep_func, MultiSpecKeep::All);
}

#[ntest::timeout(5000)]
#[test]
fn test_alpha_crown_config_clone() {
    let config = AlphaCrownConfig::default();
    let cloned = config.clone();

    assert_eq!(config.iterations, cloned.iterations);
    assert_eq!(config.learning_rate, cloned.learning_rate);
    assert_eq!(config.gradient_method, cloned.gradient_method);
}

#[ntest::timeout(5000)]
#[test]
fn test_alpha_crown_config_custom() {
    let config = AlphaCrownConfig {
        // #joint-interm-grad: 0 = legacy frozen-intermediate ascent, which is
        // what these fixtures were written against.
        joint_interm_alpha_every: 0,
        // #cgan-narrow-target-budget: None = the shipped default (no ceiling).
        cgan_narrow_target_max_rows: None,
        // #envelope-grad: false = the shipped local rule these fixtures assume.
        alpha_envelope_grad: false,
        iterations: 10,
        alpha_spec_slots: 0,
        learning_rate: 0.1,
        lr_decay: 0.95,
        tolerance: 1e-6,
        use_momentum: false,
        momentum: 0.0,
        gradient_method: GradientMethod::FiniteDifferences,
        spsa_samples: 5,
        fix_interm_bounds: false,
        cgan_sparse_target_complete_root: false,
        cgan_complete_crown_ibp_root: false,
        cgan_crown_root_map: false,
        cgan_collector_dense: false,
        // #cgan-unstable-rows: false/0.9 = the shipped default these fixtures
        // assume (abc's `minimum_sparsity`).
        cgan_unstable_rows: false,
        cgan_unstable_rows_max_frac: 0.9,
        cgan_child_interm_refine_max_secs: None,
        // #cgan-child-seed: the shipped dark defaults.
        cgan_child_interm_refine_seed_from_parent: false,
        cgan_child_interm_refine_parallel: 0,
        cgan_child_interm_refine_fresh_clip_map: false,
        // #cgan-child-zono: the shipped dark default.
        cgan_child_zono_refresh: crate::bounds::CganChildZonoRefreshConfig::default(),
        sparse_ratio: 1.0,
        adaptive_skip: false,
        adaptive_skip_depth_threshold: 100,
        adaptive_skip_pilot: false,
        pilot_improvement_threshold: 0.0,
        early_stop_patience: 5,
        optimizer: Optimizer::Sgd,
        adam_beta1: 0.9,
        adam_beta2: 0.999,
        adam_epsilon: 1e-8,
        pruning_in_iteration: true,
        pruning_in_iteration_threshold: 0.5,
        multi_spec_keep_func: MultiSpecKeep::All,
        invprop: InvpropConfig::default(),
        output_constraints: None,
        deadline: None,
        start_save_best: 0.5,
        full_conv_alpha: true,
        reference_refresh_fraction: 0.25,
        reference_refresh_max_secs: None,
        forward_linear_deadline_fallback_to_ibp: false,
        skip_zero_iteration_collection_initial_bound: false,
        spec_early_exit: None,
        spec_ascent: None,
        root_alpha_margin: false,
        alpha_zero_yield_frac: None,
        inert_ascent_exit_iters: None,
        // #vit-crown-interm: None = the shipped uncapped reference route.
        crown_ibp_reference_cap_secs: None,
        // #vit-margin-alpha: the shipped defaults these fixtures assume.
        root_alpha_margin_gradient: false,
        matmul_alpha: false,
        // #vit-margin-alpha-complex: the shipped default (lane refuses
        // MulBinary/Reciprocal graphs).
        margin_alpha_freeze_mul_reciprocal: false,
        // #vit-root-alpha: the shipped default (the certified ascent).
        margin_alpha_search: crate::bounds::MarginAlphaSearch::Ascent,
    };

    assert_eq!(config.iterations, 10);
    assert!(!config.use_momentum);
    assert_eq!(config.gradient_method, GradientMethod::FiniteDifferences);
    assert_eq!(config.optimizer, Optimizer::Sgd);
}

#[ntest::timeout(5000)]
#[test]
fn test_alpha_crown_config_serialization() {
    let config = AlphaCrownConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: AlphaCrownConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(config.iterations, deserialized.iterations);
    assert_eq!(config.learning_rate, deserialized.learning_rate);
    assert_eq!(config.gradient_method, deserialized.gradient_method);

    let mut legacy = serde_json::to_value(&config).unwrap();
    legacy
        .as_object_mut()
        .expect("serialized config is an object")
        .remove("root_alpha_margin");
    let legacy: AlphaCrownConfig = serde_json::from_value(legacy).unwrap();
    assert!(
        !legacy.root_alpha_margin,
        "configs written before the typed delivery key must keep the legacy default"
    );

    let mut legacy = serde_json::to_value(&config).unwrap();
    legacy
        .as_object_mut()
        .expect("serialized config is an object")
        .remove("alpha_zero_yield_frac");
    let legacy: AlphaCrownConfig = serde_json::from_value(legacy).unwrap();
    assert!(
        legacy.alpha_zero_yield_frac.is_none(),
        "configs written before the typed delivery key must keep the legacy default"
    );

    let mut legacy = serde_json::to_value(&config).unwrap();
    legacy
        .as_object_mut()
        .expect("serialized config is an object")
        .remove("inert_ascent_exit_iters");
    let legacy: AlphaCrownConfig = serde_json::from_value(legacy).unwrap();
    assert!(
        legacy.inert_ascent_exit_iters.is_none(),
        "configs written before #inert-ascent-exit must keep the ten-iteration patience"
    );

    // #vit-crown-interm + #vit-margin-alpha: configs written before these typed keys keep them
    // at their historical defaults (uncapped reference route, both margin keys dark).
    let mut legacy = serde_json::to_value(&config).unwrap();
    let object = legacy
        .as_object_mut()
        .expect("serialized config is an object");
    object.remove("crown_ibp_reference_cap_secs");
    object.remove("root_alpha_margin_gradient");
    object.remove("matmul_alpha");
    object.remove("margin_alpha_freeze_mul_reciprocal");
    object.remove("margin_alpha_search");
    let legacy: AlphaCrownConfig = serde_json::from_value(legacy).unwrap();
    assert!(
        legacy.crown_ibp_reference_cap_secs.is_none(),
        "configs written before #vit-crown-interm must keep the uncapped reference route"
    );
    assert!(
        !legacy.root_alpha_margin_gradient && !legacy.matmul_alpha,
        "configs written before #vit-margin-alpha must keep both keys dark"
    );
    assert!(
        !legacy.margin_alpha_freeze_mul_reciprocal,
        "configs written before #vit-margin-alpha-complex must keep the freeze key dark"
    );
    assert_eq!(
        legacy.margin_alpha_search,
        crate::bounds::MarginAlphaSearch::Ascent,
        "configs written before #vit-root-alpha must keep the certified ascent"
    );
    // The typed tokens round-trip, and nothing else parses.
    for token in crate::bounds::MarginAlphaSearch::TOKENS {
        let parsed: crate::bounds::MarginAlphaSearch =
            serde_json::from_value(serde_json::Value::String(token.to_string())).unwrap();
        assert_eq!(parsed.token(), token);
    }
    assert!(
        serde_json::from_value::<crate::bounds::MarginAlphaSearch>(serde_json::Value::String(
            "joint".to_string()
        ))
        .is_err()
    );
}

/// #vit-crown-interm: the reference cap resolves ONLY on the
/// `fix_interm_bounds = false` route and only for admissible whole seconds;
/// every other shape is the historical uncapped route.
#[ntest::timeout(5000)]
#[test]
fn crown_ibp_reference_cap_resolves_only_on_the_typed_route() {
    let fi = |cap: Option<u64>| AlphaCrownConfig {
        fix_interm_bounds: false,
        crown_ibp_reference_cap_secs: cap,
        ..AlphaCrownConfig::default()
    };
    assert_eq!(AlphaCrownConfig::default().crown_ibp_reference_cap(), None);
    assert_eq!(fi(None).crown_ibp_reference_cap(), None);
    assert_eq!(
        fi(Some(12)).crown_ibp_reference_cap(),
        Some(std::time::Duration::from_secs(12))
    );
    assert_eq!(
        fi(Some(3600)).crown_ibp_reference_cap(),
        Some(std::time::Duration::from_secs(3600))
    );
    // Inadmissible values fail closed to the uncapped route.
    assert_eq!(fi(Some(0)).crown_ibp_reference_cap(), None);
    assert_eq!(fi(Some(3601)).crown_ibp_reference_cap(), None);
    // A config carrying fix_interm_bounds = true (e.g. an input-split
    // warm-start child) is inert whatever cap it inherits.
    let child = AlphaCrownConfig {
        fix_interm_bounds: true,
        ..fi(Some(12))
    };
    assert_eq!(child.crown_ibp_reference_cap(), None);
    assert!(AlphaCrownConfig::crown_ibp_reference_cap_secs_is_valid(1));
    assert!(!AlphaCrownConfig::crown_ibp_reference_cap_secs_is_valid(0));
}
