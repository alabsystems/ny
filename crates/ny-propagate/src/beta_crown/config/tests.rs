// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::NyError;

use super::beta_config::kfsb_cert_reuse_from_raw;
use super::*;
use crate::beta_crown::branching::BranchingHeuristic;
use crate::{PgdAlphaMode, PgdInitialization, PgdOptimizer};

fn verification_mode_label(verify_upper_bound: bool) -> &'static str {
    if verify_upper_bound {
        "upper-bound"
    } else {
        "lower-bound"
    }
}

#[track_caller]
fn assert_domain_not_verified(verify_upper_bound: bool, lower: f32, upper: f32, threshold: f32) {
    let result =
        BetaCrownConfig::domain_is_verified_for_mode(verify_upper_bound, lower, upper, threshold);
    assert!(
        !result,
        "domain_is_verified_for_mode unexpectedly returned true for mode={}, lower={lower}, upper={upper}, threshold={threshold}",
        verification_mode_label(verify_upper_bound)
    );
}

#[track_caller]
fn assert_domain_not_violation(verify_upper_bound: bool, lower: f32, upper: f32, threshold: f32) {
    let result =
        BetaCrownConfig::domain_is_violation_for_mode(verify_upper_bound, lower, upper, threshold);
    assert!(
        !result,
        "domain_is_violation_for_mode unexpectedly returned true for mode={}, lower={lower}, upper={upper}, threshold={threshold}",
        verification_mode_label(verify_upper_bound)
    );
}

#[track_caller]
fn assert_invalid_config(err: &NyError) {
    assert!(
        matches!(err, NyError::InvalidConfig(_)),
        "expected NyError::InvalidConfig, got {err:?}"
    );
}

#[track_caller]
fn assert_invalid_config_contains(err: &NyError, field: &str) {
    assert_invalid_config(err);
    let rendered = err.to_string();
    assert!(
        rendered.contains(field),
        "expected invalid-config error to mention `{field}`, got {rendered}"
    );
}

#[ntest::timeout(5000)]
#[test]
fn conv_mode_auto_matches_reference_cut_policy_3813() {
    let no_cuts = BetaCrownConfig {
        conv_mode: ConvMode::Auto,
        enable_cuts: false,
        ..Default::default()
    };
    assert!(
        no_cuts.use_patches(),
        "#3813: auto conv_mode should keep patches when cuts are disabled"
    );

    let with_cuts = BetaCrownConfig {
        conv_mode: ConvMode::Auto,
        enable_cuts: true,
        ..Default::default()
    };
    assert!(
        !with_cuts.use_patches(),
        "#3813: auto conv_mode should force matrix mode when cuts are enabled"
    );
}

#[ntest::timeout(5000)]
#[test]
fn conv_mode_explicit_overrides_cut_policy_3813() {
    let forced_patches = BetaCrownConfig {
        conv_mode: ConvMode::Patches,
        enable_cuts: true,
        ..Default::default()
    };
    assert!(
        forced_patches.use_patches(),
        "#3813: explicit patches mode must override the cuts-based auto policy"
    );

    let forced_matrix = BetaCrownConfig {
        conv_mode: ConvMode::Matrix,
        enable_cuts: false,
        ..Default::default()
    };
    assert!(
        !forced_matrix.use_patches(),
        "#3813: explicit matrix mode must override the no-cuts default"
    );
}

#[ntest::timeout(5000)]
#[test]
fn crown_backward_layers_defaults_to_full_backward_3813() {
    let default = BetaCrownConfig::default();
    assert_eq!(
        default.crown_backward_layers, None,
        "#3813: missing config must preserve full backward propagation"
    );

    let truncated = BetaCrownConfig {
        crown_backward_layers: Some(6),
        ..Default::default()
    };
    assert_eq!(truncated.crown_backward_layers, Some(6));
}

/// #kfsb-multi: the wave-batched multi-objective kFSB selector is OFF by
/// default. Complete-Clipping ResNet presets opt in
/// (`use_kfsb_multi_branching: true`); every other lane keeps the default so it
/// stays byte-identical to the
/// advisory path (env `NY_MO_KFSB` overrides either way).
#[ntest::timeout(5000)]
#[test]
fn use_kfsb_multi_branching_defaults_to_false() {
    let default = BetaCrownConfig::default();
    assert!(
        !default.use_kfsb_multi_branching,
        "#kfsb-multi must default OFF so unrelated lanes stay byte-identical"
    );
}

/// #nn4sys-seb-dark: Saturation-Escape Branching is OFF by default — the
/// nn4sys preset opts in (`bab.branching.input_split.sat_escape_branch`);
/// every config that does not set the field keeps the SEB scorer and the
/// disjunctive precheck budget cap byte-identical to today (env
/// `NY_SAT_ESCAPE_BRANCH` overrides either way).
#[ntest::timeout(5000)]
#[test]
fn sat_escape_branch_defaults_to_false() {
    let default = BetaCrownConfig::default();
    assert!(
        !default.sat_escape_branch,
        "#nn4sys-seb-dark must default OFF so unrelated lanes stay byte-identical"
    );

    // Serde default matches: a config document that never names the key
    // deserializes to the dark default.
    let parsed: BetaCrownConfig = serde_json::from_value(serde_json::json!({})).unwrap();
    assert!(!parsed.sat_escape_branch);

    let armed: BetaCrownConfig =
        serde_json::from_value(serde_json::json!({ "sat_escape_branch": true })).unwrap();
    assert!(armed.sat_escape_branch);
}

#[ntest::timeout(5000)]
#[test]
fn multi_objective_critical_kfsb_defaults_to_false() {
    assert!(
        !BetaCrownConfig::default().use_multi_objective_critical_kfsb,
        "full multi-objective kFSB must remain scoped to a typed auto-routing decision"
    );
}

#[ntest::timeout(5000)]
#[test]
fn depth_two_branch_lookahead_is_typed_and_default_off() {
    let default = BetaCrownConfig::default().depth_two_branch_lookahead;
    assert_eq!(default.mode, DepthTwoBranchLookaheadMode::Off);
    assert_eq!(default.candidates, 15);
    assert_eq!(default.top_rounds, 5);
    assert_eq!(default.discount, 0.5);
    assert!(!default.enabled_at_round(0));

    let parsed: BetaCrownConfig = serde_json::from_value(serde_json::json!({
        "depth_two_branch_lookahead": {
            "mode": "select"
        }
    }))
    .expect("partial typed lookahead config");
    assert_eq!(
        parsed.depth_two_branch_lookahead,
        DepthTwoBranchLookaheadConfig {
            mode: DepthTwoBranchLookaheadMode::Select,
            ..Default::default()
        }
    );
    assert!(parsed.depth_two_branch_lookahead.enabled_at_round(0));
    assert!(parsed.depth_two_branch_lookahead.enabled_at_round(4));
    assert!(!parsed.depth_two_branch_lookahead.enabled_at_round(5));
}

#[ntest::timeout(5000)]
#[test]
fn pgd_attack_config_preserves_runtime_attack_knobs() {
    let deadline = Some(std::time::Instant::now());
    let config = BetaCrownConfig {
        pgd_restart_when_stuck: true,
        pgd_initialization: PgdInitialization::Osi,
        pgd_osi_steps: 33,
        pgd_optimizer: PgdOptimizer::SignedGradient,
        pgd_alpha_mode: PgdAlphaMode::InputRangeScaled(0.01),
        pgd_lr_decay: 0.997,
        ..Default::default()
    };

    let pgd = config.pgd_attack_config(17, 29, deadline);

    assert_eq!(pgd.num_restarts, 17);
    assert_eq!(pgd.num_steps, 29);
    assert_eq!(pgd.deadline, deadline);
    assert!(pgd.restart_when_stuck);
    assert_eq!(pgd.initialization, PgdInitialization::Osi);
    assert_eq!(pgd.osi_steps, 33);
    assert_eq!(pgd.optimizer, PgdOptimizer::SignedGradient);
    assert_eq!(pgd.alpha_mode, PgdAlphaMode::InputRangeScaled(0.01));
    assert_eq!(
        pgd.adam.lr_decay, 0.997,
        "pgd_lr_decay must thread into AdamClippingParams.lr_decay"
    );
}

#[ntest::timeout(5000)]
#[test]
fn max_crown_ibp_nodes_defaults_to_none_and_preserves_override_4244() {
    let default = BetaCrownConfig::default();
    assert_eq!(
        default.max_crown_ibp_nodes, None,
        "#4244: missing config must preserve unbounded sequential CROWN-IBP"
    );

    let budgeted = BetaCrownConfig {
        max_crown_ibp_nodes: Some(7),
        ..Default::default()
    };
    assert_eq!(budgeted.max_crown_ibp_nodes, Some(7));
}

#[ntest::timeout(5000)]
#[test]
fn interm_transfer_defaults_to_reference_enabled_4358() {
    let default = BetaCrownConfig::default();
    assert!(
        default.enable_interm_transfer,
        "#4358: missing interm_transfer config must match the reference enabled default"
    );

    let disabled = BetaCrownConfig {
        enable_interm_transfer: false,
        ..Default::default()
    };
    assert!(
        !disabled.enable_interm_transfer,
        "#4358: explicit false override must still disable interm_transfer"
    );
}

#[ntest::timeout(5000)]
#[test]
fn early_stop_patience_defaults_to_reference_2418() {
    let default = BetaCrownConfig::default();
    assert_eq!(
        default.early_stop_patience, 10,
        "#2418: beta-CROWN must default to alpha-beta-CROWN's early_stop_patience=10"
    );

    let overridden = BetaCrownConfig {
        early_stop_patience: 3,
        ..Default::default()
    };
    assert_eq!(
        overridden.early_stop_patience, 3,
        "#2418: explicit early_stop_patience overrides must be preserved"
    );
}

#[ntest::timeout(5000)]
#[test]
fn acas_xu_preset_has_expected_values() {
    let config = BetaCrownConfig::acas_xu();
    assert_eq!(config.branching_heuristic, BranchingHeuristic::InputSplit);
    assert!(
        config.reorder_bab,
        "ACAS-Xu preset should enable input-split BAB reordering"
    );
    assert!(
        config.enable_relaxed_clip,
        "ACAS-Xu preset should enable relaxed clip"
    );
    assert!(
        config.enable_pgd_attack,
        "ACAS-Xu preset should enable PGD attack"
    );
    assert_eq!(config.pgd_restarts, 10_000);
    assert_eq!(config.batch_size, 16_384);

    // ACAS-Xu disables alpha-CROWN (reference uses plain CROWN).
    // Frozen root alpha (#3453) regresses ACAS-Xu from 99.8% to 0%.
    assert!(
        !config.use_alpha_crown,
        "ACAS-Xu preset should disable alpha-CROWN (#3453)"
    );

    // Verify non-overridden fields remain at defaults
    let default = BetaCrownConfig::default();
    assert_eq!(config.max_domains, default.max_domains);
    assert_eq!(config.timeout, default.timeout);
}

#[test]
fn fresh_domain_clip_is_default_dark_and_rejects_incompatible_routes() {
    let default = BetaCrownConfig::default();
    assert!(!default.input_split_fresh_domain_clip);

    let armed = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        reorder_bab: true,
        input_split_ibp_enhancement: true,
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Relaxed,
        input_split_fresh_domain_clip: true,
        ..Default::default()
    };
    armed.validate().expect("fully scoped fresh clip config");

    for incompatible in [
        BetaCrownConfig {
            branching_heuristic: BranchingHeuristic::LargestBoundWidth,
            ..armed.clone()
        },
        BetaCrownConfig {
            reorder_bab: false,
            ..armed.clone()
        },
        BetaCrownConfig {
            input_split_ibp_enhancement: false,
            ..armed.clone()
        },
        BetaCrownConfig {
            enable_relaxed_clip: false,
            ..armed.clone()
        },
        BetaCrownConfig {
            input_clip_type: InputClipType::Complete,
            ..armed.clone()
        },
        BetaCrownConfig {
            relaxed_clip_iterations: 0,
            ..armed
        },
    ] {
        let error = incompatible
            .validate()
            .expect_err("incompatible fresh clip route must fail validation");
        assert!(error.to_string().contains("input_split_fresh_domain_clip"));
    }
}

#[ntest::timeout(5000)]
#[test]
fn lookahead_config_new_sanitizes_nan_alpha() {
    let config = LookaheadConfig::new(7, f32::NAN);
    assert_eq!(config.sync_period, 7);
    assert!(
        (config.alpha - 0.5).abs() <= f32::EPSILON,
        "NaN alpha should fall back to 0.5, got {}",
        config.alpha
    );
}

// --- Domain check method tests (#2086) ---

#[ntest::timeout(5000)]
#[test]
fn domain_is_verified_for_mode_upper_bound_mode() {
    let threshold = 0.0;

    // upper < threshold → verified
    assert!(
        BetaCrownConfig::domain_is_verified_for_mode(true, -2.0, -0.1, threshold),
        "upper=-0.1 < threshold=0.0 should be verified"
    );

    // upper == threshold → NOT verified (strict <)
    assert!(
        !BetaCrownConfig::domain_is_verified_for_mode(true, -2.0, 0.0, threshold),
        "upper==threshold should not be verified (strict <)"
    );

    // upper > threshold → NOT verified
    assert!(
        !BetaCrownConfig::domain_is_verified_for_mode(true, -2.0, 0.5, threshold),
        "upper=0.5 > threshold=0.0 should not be verified"
    );
}

#[ntest::timeout(5000)]
#[test]
fn domain_is_verified_for_mode_lower_bound_mode() {
    let threshold = 0.0;

    // lower > threshold → verified
    assert!(
        BetaCrownConfig::domain_is_verified_for_mode(false, 0.1, 2.0, threshold),
        "lower=0.1 > threshold=0.0 should be verified"
    );

    // lower == threshold → NOT verified (strict >)
    assert!(
        !BetaCrownConfig::domain_is_verified_for_mode(false, 0.0, 2.0, threshold),
        "lower==threshold should not be verified (strict >)"
    );

    // lower < threshold → NOT verified
    assert!(
        !BetaCrownConfig::domain_is_verified_for_mode(false, -0.5, 2.0, threshold),
        "lower=-0.5 < threshold=0.0 should not be verified"
    );
}

/// An inverted interval cannot grant proof authority in either direction: at
/// least one endpoint is invalid, and the verifier cannot know which one.
#[ntest::timeout(5000)]
#[test]
fn domain_is_verified_for_mode_rejects_inverted_intervals() {
    assert!(!BetaCrownConfig::domain_is_verified_for_mode(
        false, 1.0, 0.5, 0.0,
    ));
    assert!(!BetaCrownConfig::domain_is_verified_for_mode(
        true, -0.5, -1.0, 0.0,
    ));
    assert!(BetaCrownConfig::domain_is_verified_for_mode(
        false, 1.0, 1.0, 0.0,
    ));
}

#[ntest::timeout(5000)]
#[test]
fn domain_is_violation_for_mode_upper_bound_mode() {
    let threshold = 0.0;

    // lower >= threshold → violation
    assert!(
        BetaCrownConfig::domain_is_violation_for_mode(true, 0.1, 2.0, threshold),
        "lower=0.1 >= threshold=0.0 should be violation"
    );

    // lower == threshold → violation (>= is inclusive)
    assert!(
        BetaCrownConfig::domain_is_violation_for_mode(true, 0.0, 2.0, threshold),
        "lower==threshold should be violation (>= inclusive)"
    );

    // lower < threshold → NOT violation
    assert!(
        !BetaCrownConfig::domain_is_violation_for_mode(true, -0.1, 2.0, threshold),
        "lower=-0.1 < threshold=0.0 should not be violation"
    );
}

#[ntest::timeout(5000)]
#[test]
fn domain_is_violation_for_mode_lower_bound_mode() {
    let threshold = 0.0;

    // upper < threshold → violation
    assert!(
        BetaCrownConfig::domain_is_violation_for_mode(false, -2.0, -0.1, threshold),
        "upper=-0.1 < threshold=0.0 should be violation"
    );

    // upper == threshold → NOT violation (strict <)
    assert!(
        !BetaCrownConfig::domain_is_violation_for_mode(false, -2.0, 0.0, threshold),
        "upper==threshold should not be violation (strict <)"
    );

    // upper > threshold → NOT violation
    assert!(
        !BetaCrownConfig::domain_is_violation_for_mode(false, -2.0, 0.5, threshold),
        "upper=0.5 > threshold=0.0 should not be violation"
    );
}

/// #violdrop: an INVERTED interval (`lower > upper`) is numerically
/// contradictory — a valid upper can never sit below a valid lower — so its
/// decision-relevant bound cannot PROVE a violation, in either mode.
/// `tighten_child_bounds_with_parent` and the α-CROWN best-bound merge both
/// document that inversions occur in this codebase.
#[ntest::timeout(5000)]
#[test]
fn domain_is_violation_for_mode_rejects_inverted_intervals() {
    let threshold = 0.0;

    // Lower-bound mode: upper=-0.1 < threshold would fire, but lower=1.0 > upper.
    assert!(
        !BetaCrownConfig::domain_is_violation_for_mode(false, 1.0, -0.1, threshold),
        "inverted interval [1.0, -0.1] must not prove a violation (lower mode)"
    );
    // Upper-bound mode: lower=0.5 >= threshold would fire, but lower > upper.
    assert!(
        !BetaCrownConfig::domain_is_violation_for_mode(true, 0.5, -3.0, threshold),
        "inverted interval [0.5, -3.0] must not prove a violation (upper mode)"
    );
    // The exactly-degenerate interval is NOT inverted and keeps firing.
    assert!(
        BetaCrownConfig::domain_is_violation_for_mode(false, -0.1, -0.1, threshold),
        "degenerate [-0.1, -0.1] is a valid interval below the threshold"
    );
    // Ordered intervals keep their historical answers (byte-identical path).
    assert!(BetaCrownConfig::domain_is_violation_for_mode(
        false, -2.0, -0.1, threshold
    ));
    assert!(!BetaCrownConfig::domain_is_violation_for_mode(
        false, -2.0, 0.5, threshold
    ));
}

#[ntest::timeout(5000)]
#[test]
fn domain_is_verified_instance_delegates_to_for_mode() {
    let mut config = BetaCrownConfig::default();
    let threshold = 1.0;

    // Default: verify_upper_bound = false → lower > threshold
    assert!(
        !config.verify_upper_bound,
        "default verify_upper_bound should be false"
    );
    assert!(
        config.domain_is_verified(1.5, 3.0, threshold),
        "lower=1.5 > threshold=1.0 should be verified in lower-bound mode"
    );
    assert!(
        !config.domain_is_verified(0.5, 3.0, threshold),
        "lower=0.5 < threshold=1.0 should not be verified in lower-bound mode"
    );

    // Switch to upper bound mode
    config.verify_upper_bound = true;
    assert!(
        config.domain_is_verified(-1.0, 0.5, threshold),
        "upper=0.5 < threshold=1.0 should be verified in upper-bound mode"
    );
    assert!(
        !config.domain_is_verified(-1.0, 1.5, threshold),
        "upper=1.5 > threshold=1.0 should not be verified in upper-bound mode"
    );
}

#[ntest::timeout(5000)]
#[test]
fn domain_is_violation_instance_delegates_to_for_mode() {
    let mut config = BetaCrownConfig::default();
    let threshold = 1.0;

    // Default: verify_upper_bound = false → upper < threshold
    assert!(
        !config.verify_upper_bound,
        "default verify_upper_bound should be false"
    );
    assert!(
        config.domain_is_violation(-2.0, 0.5, threshold),
        "upper=0.5 < threshold=1.0 should be violation in lower-bound mode"
    );
    assert!(
        !config.domain_is_violation(-2.0, 1.5, threshold),
        "upper=1.5 > threshold=1.0 should not be violation in lower-bound mode"
    );

    // Switch to upper bound mode
    config.verify_upper_bound = true;
    assert!(
        config.domain_is_violation(1.5, 3.0, threshold),
        "lower=1.5 >= threshold=1.0 should be violation in upper-bound mode"
    );
    assert!(
        !config.domain_is_violation(0.5, 3.0, threshold),
        "lower=0.5 < threshold=1.0 should not be violation in upper-bound mode"
    );
}

// --- Lambda optimization config regression tests (#2761) ---
// Prove BaB lambda params come from config, not hardcoded constants.

#[ntest::timeout(5000)]
#[test]
fn lambda_opt_defaults_match_prior_hardcoded_values() {
    // Before #2761 these were inline constants in bab_loop.rs and relu_split_bounds.rs.
    // Verify defaults preserve the original behavior.
    let config = BetaCrownConfig::default();
    assert_eq!(config.lambda_opt_interval, 20);
    assert!(
        (config.lambda_lr - 0.05).abs() < f32::EPSILON,
        "default lambda_lr: expected 0.05, got {}",
        config.lambda_lr
    );
    assert!(
        (config.adaptive_config.beta1 - 0.9).abs() < f32::EPSILON,
        "default beta1: expected 0.9, got {}",
        config.adaptive_config.beta1
    );
    assert!(
        (config.adaptive_config.beta2 - 0.999).abs() < f32::EPSILON,
        "default beta2: expected 0.999, got {}",
        config.adaptive_config.beta2
    );
    assert!(
        (config.adaptive_config.epsilon - 1e-8).abs() < f32::EPSILON,
        "default epsilon: expected 1e-8, got {}",
        config.adaptive_config.epsilon
    );
}

#[ntest::timeout(5000)]
#[test]
fn lambda_opt_non_default_config_propagates() {
    // Regression: non-default values must survive struct construction.
    // This proves the BaB loop reads from config fields (not hardcoded).
    let config = BetaCrownConfig {
        lambda_opt_interval: 5,
        lambda_lr: 0.1,
        adaptive_config: AdaptiveOptConfig {
            beta1: 0.85,
            beta2: 0.95,
            epsilon: 1e-6,
            ..AdaptiveOptConfig::default()
        },
        ..Default::default()
    };
    assert_eq!(config.lambda_opt_interval, 5);
    assert!(
        (config.lambda_lr - 0.1).abs() < f32::EPSILON,
        "non-default lambda_lr: expected 0.1, got {}",
        config.lambda_lr
    );
    assert!(
        (config.adaptive_config.beta1 - 0.85).abs() < f32::EPSILON,
        "non-default beta1: expected 0.85, got {}",
        config.adaptive_config.beta1
    );
    assert!(
        (config.adaptive_config.beta2 - 0.95).abs() < f32::EPSILON,
        "non-default beta2: expected 0.95, got {}",
        config.adaptive_config.beta2
    );
    assert!(
        (config.adaptive_config.epsilon - 1e-6).abs() < f32::EPSILON,
        "non-default epsilon: expected 1e-6, got {}",
        config.adaptive_config.epsilon
    );
}

#[ntest::timeout(5000)]
#[test]
fn lambda_opt_interval_zero_clamped_to_one() {
    // Both BaB loops use `.max(1)` to prevent division-by-zero.
    // Verify that even with interval=0, the clamped value is 1.
    let config = BetaCrownConfig {
        lambda_opt_interval: 0,
        ..Default::default()
    };
    assert_eq!(config.lambda_opt_interval.max(1), 1);
}

// --- NaN soundness tests for domain_is_verified / domain_is_violation ---
// NaN bounds must NEVER produce a Verified or Violation result.
// IEEE 754: NaN comparisons return false, so NaN should always fall through
// to Undecided (neither verified nor violated). This is the sound behavior.

#[ntest::timeout(5000)]
#[test]
fn domain_is_verified_nan_lower_returns_false() {
    // NaN lower bound must never be verified in either mode
    assert_domain_not_verified(true, f32::NAN, -0.5, 0.0);
    assert_domain_not_verified(false, f32::NAN, 2.0, 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn domain_is_verified_nan_upper_returns_false() {
    // NaN upper bound must never be verified in either mode
    assert_domain_not_verified(true, -2.0, f32::NAN, 0.0);
    assert_domain_not_verified(false, 0.5, f32::NAN, 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn domain_is_verified_both_nan_returns_false() {
    assert_domain_not_verified(true, f32::NAN, f32::NAN, 0.0);
    assert_domain_not_verified(false, f32::NAN, f32::NAN, 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn domain_is_violation_nan_lower_returns_false() {
    // NaN lower bound must never be a violation in either mode
    assert_domain_not_violation(true, f32::NAN, 2.0, 0.0);
    assert_domain_not_violation(false, -2.0, f32::NAN, 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn domain_is_violation_nan_upper_returns_false() {
    // NaN in the upper bound position must never produce a violation.
    // mode=true: violation checks `lower >= threshold`; upper is non-deciding.
    //   Use lower=-0.5 < threshold=0.0 so result is false independent of upper.
    //   Verifies NaN upper doesn't corrupt the comparison via side effects.
    assert_domain_not_violation(true, -0.5, f32::NAN, 0.0);
    // mode=false: violation checks `upper < threshold`; NaN upper IS the
    //   deciding parameter. NaN < 0.0 = false per IEEE 754.
    // Use lower=0.5 (different from nan_lower test's -2.0) for distinct coverage.
    assert_domain_not_violation(false, 0.5, f32::NAN, 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn domain_is_violation_both_nan_returns_false() {
    assert_domain_not_violation(true, f32::NAN, f32::NAN, 0.0);
    assert_domain_not_violation(false, f32::NAN, f32::NAN, 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn domain_is_verified_nan_threshold_returns_false() {
    // NaN threshold must never produce verified
    assert_domain_not_verified(true, -2.0, -0.1, f32::NAN);
    assert_domain_not_verified(false, 0.5, 2.0, f32::NAN);
}

#[ntest::timeout(5000)]
#[test]
fn domain_is_violation_nan_threshold_returns_false() {
    // NaN threshold must never produce violation
    assert_domain_not_violation(true, 0.5, 2.0, f32::NAN);
    assert_domain_not_violation(false, -2.0, -0.1, f32::NAN);
}

// --- Inf soundness tests for domain_is_verified / domain_is_violation (#2993) ---
// Inf bounds indicate propagation failure (reciprocal zero-crossing, non-convergence),
// not genuine verification/violation results. Both must return false.

#[ntest::timeout(5000)]
#[test]
fn domain_is_verified_inf_lower_returns_false_2993() {
    // +Inf lower must never be verified
    assert_domain_not_verified(true, f32::INFINITY, -0.5, 0.0);
    assert_domain_not_verified(false, f32::INFINITY, 2.0, 0.0);
    // -Inf lower
    assert_domain_not_verified(true, f32::NEG_INFINITY, -0.5, 0.0);
    assert_domain_not_verified(false, f32::NEG_INFINITY, 2.0, 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn domain_is_verified_inf_upper_returns_false_2993() {
    // +Inf upper must never be verified
    assert_domain_not_verified(true, -2.0, f32::INFINITY, 0.0);
    assert_domain_not_verified(false, 0.5, f32::INFINITY, 0.0);
    // -Inf upper: would-be-verified in upper-bound mode (NEG_INFINITY < threshold)
    // but Inf bounds must still return false
    assert_domain_not_verified(true, -2.0, f32::NEG_INFINITY, 0.0);
    assert_domain_not_verified(false, 0.5, f32::NEG_INFINITY, 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn domain_is_violation_inf_lower_returns_false_2993() {
    // +Inf lower: the exact bug from #2993. lower >= threshold would be true,
    // incorrectly returning Violation.
    assert_domain_not_violation(true, f32::INFINITY, 2.0, 0.0);
    assert_domain_not_violation(false, f32::INFINITY, -0.5, 0.0);
    // -Inf lower
    assert_domain_not_violation(true, f32::NEG_INFINITY, 2.0, 0.0);
    assert_domain_not_violation(false, f32::NEG_INFINITY, -0.5, 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn domain_is_violation_inf_upper_returns_false_2993() {
    // +Inf upper
    assert_domain_not_violation(true, 0.5, f32::INFINITY, 0.0);
    // -Inf upper: would-be-violated in lower-bound mode (NEG_INFINITY < threshold)
    // but Inf bounds must still return false
    assert_domain_not_violation(false, -2.0, f32::NEG_INFINITY, 0.0);
    assert_domain_not_violation(true, 0.5, f32::NEG_INFINITY, 0.0);
    assert_domain_not_violation(false, -2.0, f32::INFINITY, 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn domain_is_verified_both_inf_returns_false_2993() {
    assert_domain_not_verified(true, f32::INFINITY, f32::INFINITY, 0.0);
    assert_domain_not_verified(false, f32::INFINITY, f32::INFINITY, 0.0);
    assert_domain_not_verified(true, f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0);
    assert_domain_not_verified(false, f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn domain_is_violation_both_inf_returns_false_2993() {
    assert_domain_not_violation(true, f32::INFINITY, f32::INFINITY, 0.0);
    assert_domain_not_violation(false, f32::INFINITY, f32::INFINITY, 0.0);
    assert_domain_not_violation(true, f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0);
    assert_domain_not_violation(false, f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0);
}

// --- Inf threshold tests (#2993 self-audit finding) ---
// Inf threshold would trivially verify/violate all domains:
// upper < +Inf is always true → every domain "verified"
// lower >= -Inf is always true → every domain "violated"

#[ntest::timeout(5000)]
#[test]
fn domain_is_verified_inf_threshold_returns_false_2993() {
    // +Inf threshold: upper < Inf is always true for finite upper
    assert_domain_not_verified(true, -2.0, 0.5, f32::INFINITY);
    // -Inf threshold: lower > -Inf is always true for finite lower
    assert_domain_not_verified(false, 0.5, 2.0, f32::NEG_INFINITY);
}

#[ntest::timeout(5000)]
#[test]
fn domain_is_violation_inf_threshold_returns_false_2993() {
    // -Inf threshold: lower >= -Inf is always true → false violation
    assert_domain_not_violation(true, 2.0, 3.0, f32::NEG_INFINITY);
    // +Inf threshold: upper < +Inf is always true → false violation
    assert_domain_not_violation(false, -2.0, -0.5, f32::INFINITY);
}

#[ntest::timeout(5000)]
#[test]
fn domain_priority_for_mode_respects_verification_direction() {
    // verify_upper_bound=true: prioritize larger upper bound
    assert_eq!(
        BetaCrownConfig::domain_priority_for_mode(true, -5.0, 2.5).unwrap(),
        2.5
    );
    assert_eq!(
        BetaCrownConfig::domain_priority_for_mode(true, -9.0, -1.0).unwrap(),
        -1.0
    );

    // verify_upper_bound=false: prioritize smaller lower bound (max-heap via negation)
    assert_eq!(
        BetaCrownConfig::domain_priority_for_mode(false, 2.5, 9.0).unwrap(),
        -2.5
    );
    assert_eq!(
        BetaCrownConfig::domain_priority_for_mode(false, -5.0, -1.0).unwrap(),
        5.0
    );
}

#[ntest::timeout(5000)]
#[test]
fn domain_priority_for_mode_rejects_inverted_bounds() {
    assert!(BetaCrownConfig::domain_priority_for_mode(true, 1.0, 0.0).is_err());
    assert!(BetaCrownConfig::domain_priority_for_mode(false, 1.0, 0.0).is_err());
}

#[ntest::timeout(5000)]
#[test]
fn domain_priority_for_mode_rejects_nan_bounds() {
    // NaN lower bound must produce Err(NumericalInstability) (#2982)
    assert!(
        BetaCrownConfig::domain_priority_for_mode(true, f32::NAN, 1.0).is_err(),
        "NaN lower + upper-bound mode should return error"
    );
    assert!(
        BetaCrownConfig::domain_priority_for_mode(false, f32::NAN, 1.0).is_err(),
        "NaN lower + lower-bound mode should return error"
    );

    // NaN upper bound must produce Err(NumericalInstability)
    assert!(
        BetaCrownConfig::domain_priority_for_mode(true, -1.0, f32::NAN).is_err(),
        "NaN upper + upper-bound mode should return error"
    );
    assert!(
        BetaCrownConfig::domain_priority_for_mode(false, -1.0, f32::NAN).is_err(),
        "NaN upper + lower-bound mode should return error"
    );

    // Both NaN must produce Err
    assert!(
        BetaCrownConfig::domain_priority_for_mode(true, f32::NAN, f32::NAN).is_err(),
        "both NaN + upper-bound mode should return error"
    );
    assert!(
        BetaCrownConfig::domain_priority_for_mode(false, f32::NAN, f32::NAN).is_err(),
        "both NaN + lower-bound mode should return error"
    );
}

#[ntest::timeout(5000)]
#[test]
fn violation_priority_matches_domain_priority() {
    let config = BetaCrownConfig {
        verify_upper_bound: false,
        ..Default::default()
    };
    assert_eq!(
        config.violation_priority(0.25, 10.0).unwrap(),
        config.domain_priority(0.25, 10.0).unwrap()
    );

    let config = BetaCrownConfig {
        verify_upper_bound: true,
        ..Default::default()
    };
    assert_eq!(
        config.violation_priority(-3.0, 1.25).unwrap(),
        config.domain_priority(-3.0, 1.25).unwrap()
    );
}

// --- LR scheduler guard regression tests (#2840) ---

#[ntest::timeout(5000)]
#[test]
fn lr_step_decay_step_size_zero_returns_one() {
    // step_size=0 must not panic (division by zero). Guard returns 1.0
    // consistent with CosineAnnealing t_max=0 pattern.
    let sched = LRScheduler::StepDecay {
        ny: 0.5,
        step_size: 0,
    };
    assert_eq!(sched.lr_factor(0, 0.1), 1.0);
    assert_eq!(sched.lr_factor(100, 0.1), 1.0);
}

#[ntest::timeout(5000)]
#[test]
fn lr_step_decay_i32_saturation_no_wrap() {
    // For very large t/step_size, num_decays saturates to i32::MAX
    // instead of wrapping to negative (which would make LR grow).
    let sched = LRScheduler::StepDecay {
        ny: 0.5,
        step_size: 1,
    };
    let factor = sched.lr_factor(usize::MAX, 1.0);
    // ny^(i32::MAX) for ny=0.5 is effectively 0.0
    assert!(
        factor >= 0.0 && factor.is_finite(),
        "factor should be non-negative finite, got {factor}"
    );
    // Must NOT grow: ny < 1.0 means factor <= 1.0 always
    assert!(factor <= 1.0, "factor should not exceed 1.0, got {factor}");
}

#[ntest::timeout(5000)]
#[test]
fn lr_exponential_decay_i32_saturation_no_wrap() {
    // ny.powi(t as i32) must saturate, not wrap to negative exponent.
    let sched = LRScheduler::ExponentialDecay { ny: 0.99 };
    let factor = sched.lr_factor(usize::MAX, 1.0);
    assert!(
        factor >= 0.0 && factor.is_finite(),
        "factor should be non-negative finite, got {factor}"
    );
    assert!(factor <= 1.0, "factor should not exceed 1.0, got {factor}");
}

#[ntest::timeout(5000)]
#[test]
fn lr_step_decay_normal_operation() {
    // Verify normal operation is not affected by guards.
    let sched = LRScheduler::StepDecay {
        ny: 0.5,
        step_size: 10,
    };
    // t=0: 0 decays → factor=1.0
    assert!(
        (sched.lr_factor(0, 1.0) - 1.0).abs() < f32::EPSILON,
        "t=0: expected factor=1.0, got {}",
        sched.lr_factor(0, 1.0)
    );
    // t=9: still 0 decays → factor=1.0
    assert!(
        (sched.lr_factor(9, 1.0) - 1.0).abs() < f32::EPSILON,
        "t=9: expected factor=1.0, got {}",
        sched.lr_factor(9, 1.0)
    );
    // t=10: 1 decay → factor=0.5
    assert!(
        (sched.lr_factor(10, 1.0) - 0.5).abs() < f32::EPSILON,
        "t=10: expected factor=0.5, got {}",
        sched.lr_factor(10, 1.0)
    );
    // t=20: 2 decays → factor=0.25
    assert!(
        (sched.lr_factor(20, 1.0) - 0.25).abs() < f32::EPSILON,
        "t=20: expected factor=0.25, got {}",
        sched.lr_factor(20, 1.0)
    );
}

// --- AdaptiveOptConfig validation tests (#2942) ---

#[ntest::timeout(5000)]
#[test]
fn adaptive_opt_config_default_validates() {
    AdaptiveOptConfig::default().validate().unwrap();
}

#[ntest::timeout(5000)]
#[test]
fn adaptive_opt_config_negative_beta_lr_rejected() {
    let config = AdaptiveOptConfig {
        beta_lr: -0.01,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert_invalid_config_contains(&err, "beta_lr");
}

#[ntest::timeout(5000)]
#[test]
fn adaptive_opt_config_negative_alpha_lr_rejected() {
    let config = AdaptiveOptConfig {
        alpha_lr: -0.05,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert_invalid_config_contains(&err, "alpha_lr");
}

#[ntest::timeout(5000)]
#[test]
fn adaptive_opt_config_nan_beta_lr_rejected() {
    let config = AdaptiveOptConfig {
        beta_lr: f32::NAN,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert_invalid_config(&err);
}

#[ntest::timeout(5000)]
#[test]
fn adaptive_opt_config_zero_epsilon_rejected() {
    let config = AdaptiveOptConfig {
        epsilon: 0.0,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert_invalid_config_contains(&err, "epsilon");
}

#[ntest::timeout(5000)]
#[test]
fn adaptive_opt_config_negative_epsilon_rejected() {
    let config = AdaptiveOptConfig {
        epsilon: -1e-8,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert_invalid_config(&err);
}

#[ntest::timeout(5000)]
#[test]
fn adaptive_opt_config_beta1_out_of_range_rejected() {
    let config = AdaptiveOptConfig {
        beta1: 1.5,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert_invalid_config_contains(&err, "beta1");
}

#[ntest::timeout(5000)]
#[test]
fn adaptive_opt_config_beta2_negative_rejected() {
    let config = AdaptiveOptConfig {
        beta2: -0.1,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert_invalid_config_contains(&err, "beta2");
}

#[ntest::timeout(5000)]
#[test]
fn adaptive_opt_config_negative_grad_clip_rejected() {
    let config = AdaptiveOptConfig {
        grad_clip: -1.0,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert_invalid_config_contains(&err, "grad_clip");
}

#[ntest::timeout(5000)]
#[test]
fn adaptive_opt_config_negative_weight_decay_rejected() {
    let config = AdaptiveOptConfig {
        weight_decay: -0.01,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert_invalid_config_contains(&err, "weight_decay");
}

#[ntest::timeout(5000)]
#[test]
fn adaptive_opt_config_negative_lr_lambda_rejected() {
    let config = AdaptiveOptConfig {
        lr_lambda: Some(-0.01),
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert_invalid_config_contains(&err, "lr_lambda");
}

#[ntest::timeout(5000)]
#[test]
fn adaptive_opt_config_valid_custom_validates() {
    let config = AdaptiveOptConfig {
        beta_lr: 0.1,
        alpha_lr: 0.05,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        grad_clip: 10.0,
        weight_decay: 0.01,
        lr_lambda: Some(0.02),
        ..Default::default()
    };
    config.validate().unwrap();
}

// --- BetaCrownConfig::validate() tests (#2942) ---

#[ntest::timeout(5000)]
#[test]
fn beta_crown_config_default_validates() {
    BetaCrownConfig::default().validate().unwrap();
}

#[ntest::timeout(5000)]
#[test]
fn beta_crown_config_rejects_unbounded_or_nonfinite_depth_two_policy() {
    for invalid in [
        DepthTwoBranchLookaheadConfig {
            candidates: 0,
            ..Default::default()
        },
        DepthTwoBranchLookaheadConfig {
            candidates: 16,
            ..Default::default()
        },
        DepthTwoBranchLookaheadConfig {
            top_rounds: 0,
            ..Default::default()
        },
        DepthTwoBranchLookaheadConfig {
            top_rounds: 6,
            ..Default::default()
        },
        DepthTwoBranchLookaheadConfig {
            discount: f64::NAN,
            ..Default::default()
        },
        DepthTwoBranchLookaheadConfig {
            discount: 1.01,
            ..Default::default()
        },
    ] {
        let config = BetaCrownConfig {
            depth_two_branch_lookahead: invalid,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert_invalid_config_contains(&err, "depth_two_branch_lookahead");
    }
}

#[ntest::timeout(5000)]
#[test]
fn beta_crown_config_validate_rejects_forward_and_alpha_combo_4354() {
    let config = BetaCrownConfig {
        use_forward_bounds: true,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert_invalid_config_contains(&err, "use_forward_bounds");
}

#[ntest::timeout(5000)]
#[test]
fn beta_crown_config_negative_beta_lr_rejected() {
    let config = BetaCrownConfig {
        beta_lr: -0.01,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert_invalid_config_contains(&err, "beta_lr");
}

#[ntest::timeout(5000)]
#[test]
fn beta_crown_config_negative_alpha_lr_rejected() {
    let config = BetaCrownConfig {
        alpha_lr: -0.05,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert_invalid_config_contains(&err, "alpha_lr");
}

#[ntest::timeout(5000)]
#[test]
fn beta_crown_config_nan_alpha_lr_rejected() {
    let config = BetaCrownConfig {
        alpha_lr: f32::NAN,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert_invalid_config(&err);
}

#[ntest::timeout(5000)]
#[test]
fn beta_crown_config_zero_build_batch_size_rejected_4354() {
    let config = BetaCrownConfig {
        build_batch_size: Some(0),
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert_invalid_config_contains(&err, "build_batch_size");
}

/// Depth 0 selects no split dimensions, so InputSplit BaB could never branch:
/// the config must be rejected up front instead of silently disabling search.
#[ntest::timeout(5000)]
#[test]
fn beta_crown_config_zero_input_split_depth_rejected_with_input_split() {
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        input_split_depth: 0,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert_invalid_config_contains(&err, "input_split_depth");
}

/// `input_split_depth` is only used by the InputSplit heuristic; a zero value
/// under other heuristics is inert and must not fail validation.
#[ntest::timeout(5000)]
#[test]
fn beta_crown_config_zero_input_split_depth_allowed_without_input_split() {
    let config = BetaCrownConfig {
        input_split_depth: 0,
        ..Default::default()
    };
    config.validate().unwrap();
}

/// Cutting planes must remain research-only until their constraints are
/// proof-derived and their coefficients are folded through backward CROWN.
#[ntest::timeout(5000)]
#[test]
fn beta_crown_config_rejects_all_cut_authority_requests() {
    for config in [
        BetaCrownConfig {
            enable_cuts: true,
            ..Default::default()
        },
        BetaCrownConfig {
            enable_near_miss_cuts: true,
            ..Default::default()
        },
        BetaCrownConfig {
            enable_proactive_cuts: true,
            ..Default::default()
        },
    ] {
        let err = config.validate().unwrap_err();
        assert_invalid_config_contains(&err, "cut proof authority is quarantined");
        assert!(
            !config.cut_proof_authority_enabled(),
            "the internal authority gate must remain closed even when validation is bypassed"
        );
    }
}

#[ntest::timeout(5000)]
#[test]
fn beta_crown_config_validates_adaptive_config() {
    // Invalid adaptive_config should be caught by BetaCrownConfig::validate()
    let config = BetaCrownConfig {
        adaptive_config: AdaptiveOptConfig {
            epsilon: 0.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert_invalid_config_contains(&err, "epsilon");
}

// --- effective_relu_split_depth tests (#2767) ---
// Reference: alpha-beta-CROWN `get_split_depth()` in `bab.py:40-48`.

#[ntest::timeout(5000)]
#[test]
fn effective_relu_split_depth_disabled_returns_one() {
    // max_relu_split_depth <= 1 disables multi-depth splitting
    let config = BetaCrownConfig {
        max_relu_split_depth: 1,
        batch_size: 64,
        min_batch_fill_ratio: 0.5,
        ..Default::default()
    };
    assert_eq!(config.effective_relu_split_depth(0), 1);
    assert_eq!(config.effective_relu_split_depth(1), 1);
    assert_eq!(config.effective_relu_split_depth(100), 1);
}

#[ntest::timeout(5000)]
#[test]
fn effective_relu_split_depth_large_queue_returns_one() {
    // Queue >= min_batch → no multi-depth needed
    let config = BetaCrownConfig {
        max_relu_split_depth: 6,
        batch_size: 64,
        min_batch_fill_ratio: 0.5,
        ..Default::default()
    };
    // min_batch = 64 * 0.5 = 32
    assert_eq!(config.effective_relu_split_depth(32), 1); // queue == min_batch
    assert_eq!(config.effective_relu_split_depth(64), 1); // queue > min_batch
    assert_eq!(config.effective_relu_split_depth(1000), 1);
}

#[ntest::timeout(5000)]
#[test]
fn effective_relu_split_depth_small_queue_increases_depth() {
    let config = BetaCrownConfig {
        max_relu_split_depth: 6,
        batch_size: 64,
        min_batch_fill_ratio: 0.5,
        ..Default::default()
    };
    // min_batch = 32
    // queue=16: ratio=2, depth=floor(log2(2))=1
    assert_eq!(config.effective_relu_split_depth(16), 1);
    // queue=8: ratio=4, depth=floor(log2(4))=2
    assert_eq!(config.effective_relu_split_depth(8), 2);
    // queue=4: ratio=8, depth=floor(log2(8))=3
    assert_eq!(config.effective_relu_split_depth(4), 3);
    // queue=2: ratio=16, depth=floor(log2(16))=4
    assert_eq!(config.effective_relu_split_depth(2), 4);
    // queue=1: ratio=32, depth=floor(log2(32))=5
    assert_eq!(config.effective_relu_split_depth(1), 5);
}

#[ntest::timeout(5000)]
#[test]
fn effective_relu_split_depth_matches_reference_fractional_root() {
    let config = BetaCrownConfig {
        max_relu_split_depth: 10,
        batch_size: 256,
        min_batch_fill_ratio: 0.1,
        ..Default::default()
    };
    // alpha-beta-CROWN: int(log2(256 * 0.1 / 1)) = int(log2(25.6)) = 4.
    assert_eq!(config.effective_relu_split_depth(1), 4);
    // Floor, rather than ceil, is observable away from a power of two.
    assert_eq!(config.effective_relu_split_depth(2), 3);
    assert_eq!(config.effective_relu_split_depth(4), 2);
    assert_eq!(config.effective_relu_split_depth(8), 1);
}

#[ntest::timeout(5000)]
#[test]
fn effective_relu_split_depth_clamped_at_max() {
    let config = BetaCrownConfig {
        max_relu_split_depth: 3,
        batch_size: 64,
        min_batch_fill_ratio: 0.5,
        ..Default::default()
    };
    // min_batch = 32, queue=1: ratio=32, depth=5 but clamped to 3
    assert_eq!(config.effective_relu_split_depth(1), 3);
    // queue=2: ratio=16, depth=4 but clamped to 3
    assert_eq!(config.effective_relu_split_depth(2), 3);
}

#[ntest::timeout(5000)]
#[test]
fn effective_relu_split_depth_honors_defensive_truth_table_cap() {
    let config = BetaCrownConfig {
        max_relu_split_depth: usize::MAX,
        batch_size: usize::MAX,
        min_batch_fill_ratio: 1.0,
        ..Default::default()
    };
    assert_eq!(config.effective_relu_split_depth(1), 10);
}

#[ntest::timeout(5000)]
#[test]
fn effective_relu_split_depth_queue_zero_clamped() {
    // Queue size 0 should not panic (divides by max(1, queue))
    let config = BetaCrownConfig {
        max_relu_split_depth: 6,
        batch_size: 64,
        min_batch_fill_ratio: 0.5,
        ..Default::default()
    };
    let depth = config.effective_relu_split_depth(0);
    assert!(depth >= 1, "depth should be at least 1, got {depth}");
    assert!(depth <= 6, "depth should be at most max=6, got {depth}");
}

#[ntest::timeout(5000)]
#[test]
fn effective_relu_split_depth_default_config_disabled() {
    // Default config has max_relu_split_depth=1, always returns 1
    let config = BetaCrownConfig::default();
    assert_eq!(config.max_relu_split_depth, 1);
    assert_eq!(config.effective_relu_split_depth(0), 1);
    assert_eq!(config.effective_relu_split_depth(1), 1);
    assert_eq!(config.effective_relu_split_depth(100), 1);
}

// ============================================================
// auto_enlarge_batch_size (#4303)
// ============================================================

#[test]
fn maybe_enlarge_batch_size_disabled_by_default() {
    let config = BetaCrownConfig::default();
    assert!(!config.auto_enlarge_batch_size);
    assert_eq!(config.maybe_enlarge_batch_size(64, 64), None);
}

#[test]
fn maybe_enlarge_batch_size_doubles_on_full_batch_4303() {
    let config = BetaCrownConfig {
        auto_enlarge_batch_size: true,
        ..Default::default()
    };
    assert_eq!(config.maybe_enlarge_batch_size(64, 64), Some(128));
    assert_eq!(config.maybe_enlarge_batch_size(128, 128), Some(256));
    assert_eq!(config.maybe_enlarge_batch_size(256, 300), Some(512));
}

#[test]
fn maybe_enlarge_batch_size_no_change_on_partial_batch_4303() {
    let config = BetaCrownConfig {
        auto_enlarge_batch_size: true,
        ..Default::default()
    };
    assert_eq!(config.maybe_enlarge_batch_size(64, 63), None);
    assert_eq!(config.maybe_enlarge_batch_size(64, 1), None);
    assert_eq!(config.maybe_enlarge_batch_size(64, 0), None);
}

#[test]
fn maybe_enlarge_batch_size_caps_at_8192_4303() {
    use super::beta_config::AUTO_ENLARGE_BATCH_CAP;
    let config = BetaCrownConfig {
        auto_enlarge_batch_size: true,
        ..Default::default()
    };
    // At cap: no enlargement
    assert_eq!(
        config.maybe_enlarge_batch_size(AUTO_ENLARGE_BATCH_CAP, AUTO_ENLARGE_BATCH_CAP),
        None
    );
    // Just below cap: clamp to cap
    assert_eq!(
        config.maybe_enlarge_batch_size(AUTO_ENLARGE_BATCH_CAP / 2, AUTO_ENLARGE_BATCH_CAP / 2),
        Some(AUTO_ENLARGE_BATCH_CAP)
    );
    // Well below cap: normal doubling, capped
    assert_eq!(
        config.maybe_enlarge_batch_size(AUTO_ENLARGE_BATCH_CAP - 1, AUTO_ENLARGE_BATCH_CAP),
        Some(AUTO_ENLARGE_BATCH_CAP)
    );
}

// ============================================================
// try_enlarge_batch_size shared closeout helper (#4303)
// ============================================================

#[test]
fn try_enlarge_batch_size_updates_in_place_on_full_batch_4303() {
    let config = BetaCrownConfig {
        auto_enlarge_batch_size: true,
        ..Default::default()
    };
    let mut batch_size = 64;
    assert!(config.try_enlarge_batch_size(&mut batch_size, 64, "test"));
    assert_eq!(batch_size, 128);
    // Chain: second full batch doubles again
    assert!(config.try_enlarge_batch_size(&mut batch_size, 200, "test"));
    assert_eq!(batch_size, 256);
}

#[test]
fn try_enlarge_batch_size_noop_on_partial_batch_4303() {
    let config = BetaCrownConfig {
        auto_enlarge_batch_size: true,
        ..Default::default()
    };
    let mut batch_size = 64;
    assert!(!config.try_enlarge_batch_size(&mut batch_size, 63, "test"));
    assert_eq!(batch_size, 64);
}

#[test]
fn try_enlarge_batch_size_caps_at_limit_4303() {
    use super::beta_config::AUTO_ENLARGE_BATCH_CAP;
    let config = BetaCrownConfig {
        auto_enlarge_batch_size: true,
        ..Default::default()
    };
    let mut batch_size = AUTO_ENLARGE_BATCH_CAP;
    assert!(!config.try_enlarge_batch_size(&mut batch_size, AUTO_ENLARGE_BATCH_CAP, "test"));
    assert_eq!(batch_size, AUTO_ENLARGE_BATCH_CAP);
}

/// No-regression guarantee: per-sub-domain α refinement is OFF by default, so the
/// input-split BaB loop keeps doing exactly the single frozen-alpha pass it did
/// before this knob existed. The warm path is only taken when
/// `input_split_alpha_iteration > 0`.
#[test]
fn input_split_alpha_refinement_is_disabled_by_default() {
    let config = BetaCrownConfig::default();
    assert_eq!(
        config.input_split_alpha_iteration, 0,
        "default input_split_alpha_iteration must be 0 (frozen-alpha behavior, no regression)"
    );
    // The learning rate matches alpha-beta-CROWN's input_split_lr_alpha default
    // but is inert while the iteration count is 0.
    assert_eq!(config.input_split_lr_alpha, 0.05);
}

/// Serde round-trip with no explicit values keeps the refinement knobs at their
/// defaults (0 iterations), so deserializing an existing config never silently
/// flips on the new behavior.
#[test]
fn input_split_alpha_refinement_serde_defaults_to_disabled() {
    let config: BetaCrownConfig = serde_json::from_str("{}").expect("empty config deserializes");
    assert_eq!(config.input_split_alpha_iteration, 0);
    assert_eq!(config.input_split_lr_alpha, 0.05);
}

#[test]
fn root_alpha_phase_checkpoint_serde_is_typed_and_default_off() {
    let default: BetaCrownConfig = serde_json::from_str("{}").expect("empty config deserializes");
    assert!(!default.root_alpha_phase_checkpoint);

    let armed: BetaCrownConfig = serde_json::from_str(r#"{"root_alpha_phase_checkpoint": true}"#)
        .expect("typed root-alpha checkpoint config deserializes");
    assert!(armed.root_alpha_phase_checkpoint);

    let disabled: BetaCrownConfig =
        serde_json::from_str(r#"{"root_alpha_phase_checkpoint": false}"#)
            .expect("typed root-alpha checkpoint kill switch deserializes");
    assert!(!disabled.root_alpha_phase_checkpoint);
}

#[test]
fn kfsb_cert_reuse_serde_is_typed_and_default_off() {
    let default: BetaCrownConfig = serde_json::from_str("{}").expect("empty config deserializes");
    assert!(!default.kfsb_cert_reuse);

    let armed: BetaCrownConfig = serde_json::from_str(r#"{"kfsb_cert_reuse": true}"#)
        .expect("typed kFSB certificate-reuse config deserializes");
    assert!(armed.kfsb_cert_reuse);

    let disabled: BetaCrownConfig = serde_json::from_str(r#"{"kfsb_cert_reuse": false}"#)
        .expect("typed kFSB certificate-reuse kill switch deserializes");
    assert!(!disabled.kfsb_cert_reuse);
}

#[test]
fn kfsb_cert_reuse_raw_resolution_is_exact_and_fail_closed() {
    assert!(!kfsb_cert_reuse_from_raw(false, None));
    assert!(kfsb_cert_reuse_from_raw(true, None));
    assert!(kfsb_cert_reuse_from_raw(
        false,
        Some(std::ffi::OsStr::new("1"))
    ));
    for malformed in ["", "0", "01", "true", " 1", "1 "] {
        assert!(!kfsb_cert_reuse_from_raw(
            false,
            Some(std::ffi::OsStr::new(malformed))
        ));
        assert!(!kfsb_cert_reuse_from_raw(
            true,
            Some(std::ffi::OsStr::new(malformed))
        ));
    }
}

#[test]
fn kfsb_cert_reuse_armed_is_the_typed_environment_resolution_point() {
    crate::tests::with_serialized_env_vars_removed(&["NY_MO_KFSB_CERT_REUSE"], || {
        assert!(!BetaCrownConfig::default().kfsb_cert_reuse_armed());
        assert!(BetaCrownConfig {
            kfsb_cert_reuse: true,
            ..BetaCrownConfig::default()
        }
        .kfsb_cert_reuse_armed());
    });
    crate::tests::with_serialized_env_vars(&[("NY_MO_KFSB_CERT_REUSE", "1")], || {
        assert!(BetaCrownConfig::default().kfsb_cert_reuse_armed());
    });
    for value in ["0", "true", "malformed"] {
        crate::tests::with_serialized_env_vars(&[("NY_MO_KFSB_CERT_REUSE", value)], || {
            assert!(!BetaCrownConfig {
                kfsb_cert_reuse: true,
                ..BetaCrownConfig::default()
            }
            .kfsb_cert_reuse_armed());
        });
    }
}

#[cfg(unix)]
#[test]
fn kfsb_cert_reuse_non_unicode_override_disables_typed_policy() {
    use std::os::unix::ffi::OsStringExt;

    let non_unicode = std::ffi::OsString::from_vec(vec![b'1', 0xff]);
    assert!(!kfsb_cert_reuse_from_raw(
        false,
        Some(non_unicode.as_os_str())
    ));
    assert!(!kfsb_cert_reuse_from_raw(
        true,
        Some(non_unicode.as_os_str())
    ));
}

#[test]
fn verification_artifact_authority_is_runtime_only_and_fail_closed_under_serde() {
    let default: BetaCrownConfig = serde_json::from_str("{}").expect("empty config deserializes");
    assert_eq!(
        default.verification_artifact_authority,
        VerificationArtifactAuthority::CertificateExport
    );

    // A preset/config document cannot smuggle verdict-only authority through
    // the serde surface. The runtime-only field is not part of that schema,
    // and the direct typed schema rejects it instead of silently accepting
    // ambiguous material.
    assert!(
        serde_json::from_str::<BetaCrownConfig>(
            r#"{"verification_artifact_authority":"verdict_only"}"#
        )
        .is_err(),
        "runtime-only authority must be rejected by config deserialization"
    );

    let runtime = BetaCrownConfig {
        verification_artifact_authority: VerificationArtifactAuthority::VerdictOnly,
        ..Default::default()
    };
    let serialized = serde_json::to_value(&runtime).expect("config serializes");
    assert!(
        serialized.get("verification_artifact_authority").is_none(),
        "runtime authority must never leak into reusable preset/config material"
    );
    let cloned_runtime = runtime.clone();
    assert_eq!(
        runtime.verification_artifact_authority,
        VerificationArtifactAuthority::VerdictOnly,
        "cloning must not consume or mutate the resolved frontend request"
    );
    assert_eq!(
        cloned_runtime.verification_artifact_authority,
        VerificationArtifactAuthority::VerdictOnly,
        "in-process verifier clones must preserve the resolved frontend request"
    );
}

#[test]
fn input_split_conic_objective_is_default_dark_and_authority_scoped_under_serde() {
    let default: BetaCrownConfig = serde_json::from_str("{}").expect("empty config deserializes");
    assert!(!default.input_split_conic_objective);
    assert_eq!(default.input_split_conic_queue_refresh_batch_size, 512);
    assert!(!default.input_split_conic_objective_eligible());

    let default_json = serde_json::to_value(&default).expect("default config serializes");
    assert_eq!(default_json["input_split_conic_objective"], false);
    assert_eq!(
        default_json["input_split_conic_queue_refresh_batch_size"],
        512
    );

    let mut armed: BetaCrownConfig = serde_json::from_str(
        r#"{"input_split_conic_objective":true,"input_split_conic_queue_refresh_batch_size":1024}"#,
    )
    .expect("typed conic gate deserializes");
    assert!(armed.input_split_conic_objective);
    assert_eq!(armed.input_split_conic_queue_refresh_batch_size, 1024);
    assert!(
        !armed.input_split_conic_objective_eligible(),
        "serde restores certificate-export authority and must therefore decline"
    );

    armed.verification_artifact_authority = VerificationArtifactAuthority::VerdictOnly;
    assert!(armed.input_split_conic_objective_eligible());

    let round_trip: BetaCrownConfig =
        serde_json::from_value(serde_json::to_value(&armed).expect("armed config serializes"))
            .expect("armed config round-trips");
    assert!(round_trip.input_split_conic_objective);
    assert_eq!(round_trip.input_split_conic_queue_refresh_batch_size, 1024);
    assert!(
        !round_trip.input_split_conic_objective_eligible(),
        "runtime verdict-only authority must not survive reusable serialization"
    );
}

#[test]
fn input_split_conic_queue_refresh_batch_size_rejects_zero() {
    let config = BetaCrownConfig {
        input_split_conic_queue_refresh_batch_size: 0,
        ..Default::default()
    };
    let error = config
        .validate()
        .expect_err("a zero-sized tranche cannot run");
    assert_invalid_config_contains(&error, "input_split_conic_queue_refresh_batch_size");
}

#[test]
fn direct_beta_config_serde_rejects_unknown_outer_and_nested_fields() {
    assert!(
        serde_json::from_str::<BetaCrownConfig>(r#"{"mo_cuda_bounded_shared_executorr":true}"#)
            .is_err(),
        "an unknown outer authority key must not silently retain its default"
    );
    assert!(
        serde_json::from_str::<BetaCrownConfig>(
            r#"{"depth_two_branch_lookahead":{"mode":"select","candidatez":3}}"#
        )
        .is_err(),
        "an unknown bounded-lookahead resource key must be rejected"
    );
}

#[test]
fn root_crown_interm_serde_is_typed_bounded_and_default_off() {
    let default: BetaCrownConfig = serde_json::from_str("{}").expect("empty config deserializes");
    assert!(!default.root_crown_interm_dense_head);
    assert_eq!(default.root_crown_interm_max_secs, 2);
    assert_eq!(default.root_crown_interm_max_dim, 512);

    let armed: BetaCrownConfig = serde_json::from_str(
        r#"{
            "root_crown_interm_dense_head": true,
            "root_crown_interm_max_secs": 5,
            "root_crown_interm_max_dim": 100
        }"#,
    )
    .expect("typed root CROWN config deserializes");
    assert!(armed.root_crown_interm_dense_head);
    assert_eq!(armed.root_crown_interm_max_secs, 5);
    assert_eq!(armed.root_crown_interm_max_dim, 100);
}

#[test]
fn root_interm_cuda_factory_serde_is_typed_and_default_off() {
    let default: BetaCrownConfig = serde_json::from_str("{}").expect("empty config deserializes");
    assert!(!default.root_interm_cuda_factory);

    let armed: BetaCrownConfig = serde_json::from_str(r#"{"root_interm_cuda_factory": true}"#)
        .expect("typed root intermediate CUDA factory config deserializes");
    assert!(armed.root_interm_cuda_factory);

    let disabled: BetaCrownConfig = serde_json::from_str(r#"{"root_interm_cuda_factory": false}"#)
        .expect("typed root intermediate CUDA factory kill switch deserializes");
    assert!(!disabled.root_interm_cuda_factory);
}

#[test]
fn mo_cuda_factory_engine_handoff_serde_is_typed_and_default_off() {
    let default: BetaCrownConfig = serde_json::from_str("{}").expect("empty config deserializes");
    assert!(!default.mo_cuda_factory_engine_handoff);

    let armed: BetaCrownConfig =
        serde_json::from_str(r#"{"mo_cuda_factory_engine_handoff": true}"#)
            .expect("typed post-root CUDA factory handoff config deserializes");
    assert!(armed.mo_cuda_factory_engine_handoff);

    let disabled: BetaCrownConfig =
        serde_json::from_str(r#"{"mo_cuda_factory_engine_handoff": false}"#)
            .expect("typed post-root CUDA factory handoff kill switch deserializes");
    assert!(!disabled.mo_cuda_factory_engine_handoff);
}

#[test]
fn mo_cuda_bounded_shared_executor_serde_is_typed_and_default_off() {
    let default: BetaCrownConfig = serde_json::from_str("{}").expect("empty config deserializes");
    assert!(!default.mo_cuda_bounded_shared_executor);

    let armed: BetaCrownConfig =
        serde_json::from_str(r#"{"mo_cuda_bounded_shared_executor": true}"#)
            .expect("typed bounded shared-executor config deserializes");
    assert!(armed.mo_cuda_bounded_shared_executor);

    let disabled: BetaCrownConfig =
        serde_json::from_str(r#"{"mo_cuda_bounded_shared_executor": false}"#)
            .expect("typed bounded shared-executor kill switch deserializes");
    assert!(!disabled.mo_cuda_bounded_shared_executor);
}

#[test]
fn root_sparse_interm_crown_serde_is_typed_bounded_and_default_off() {
    let default: BetaCrownConfig = serde_json::from_str("{}").expect("empty config deserializes");
    assert!(!default.root_sparse_interm_crown);
    assert_eq!(default.root_sparse_interm_crown_max_secs, 2);
    assert_eq!(default.root_sparse_interm_crown_max_dim, 8_192);
    assert_eq!(default.root_sparse_interm_crown_max_rows, 512);
    assert_eq!(default.root_sparse_interm_crown_max_targets, 4);

    let armed: BetaCrownConfig = serde_json::from_str(
        r#"{
            "root_sparse_interm_crown": true,
            "root_sparse_interm_crown_max_secs": 3,
            "root_sparse_interm_crown_max_dim": 4096,
            "root_sparse_interm_crown_max_rows": 96,
            "root_sparse_interm_crown_max_targets": 2
        }"#,
    )
    .expect("typed root sparse intermediate CROWN config deserializes");
    assert!(armed.root_sparse_interm_crown);
    assert_eq!(armed.root_sparse_interm_crown_max_secs, 3);
    assert_eq!(armed.root_sparse_interm_crown_max_dim, 4096);
    assert_eq!(armed.root_sparse_interm_crown_max_rows, 96);
    assert_eq!(armed.root_sparse_interm_crown_max_targets, 2);
}

#[test]
fn complete_clip_pruning_is_explicitly_quarantined() {
    let config = BetaCrownConfig {
        enable_clip_interm_domain: true,
        clip_interm_prune: true,
        ..Default::default()
    };
    let error = config
        .validate()
        .expect_err("certificate-backed Complete Clip must reject pruning authority");
    assert!(
        matches!(&error, NyError::InvalidConfig(message) if message.contains("quarantined")),
        "unexpected validation error: {error:?}"
    );
}

#[test]
fn root_post_c_survivor_serde_is_typed_and_default_off() {
    let default = BetaCrownConfig::default();
    assert!(!default.root_post_c_survivor);

    let armed: BetaCrownConfig = serde_json::from_value(serde_json::json!({
        "root_post_c_survivor": true
    }))
    .expect("typed post-C survivor config deserializes");
    assert!(armed.root_post_c_survivor);
}

#[test]
fn atomic_root_c_margin_iterations_are_typed_dark_and_hard_capped() {
    let default = BetaCrownConfig::default();
    assert_eq!(default.atomic_root_c_margin_iterations, 0);
    default.validate().expect("dark default validates");

    let capped: BetaCrownConfig = serde_json::from_value(serde_json::json!({
        "atomic_root_c_margin_iterations": ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS
    }))
    .expect("typed exact-C iteration config deserializes");
    assert_eq!(
        capped.atomic_root_c_margin_iterations,
        ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS
    );
    capped.validate().expect("the hard cap is admitted");

    let over = BetaCrownConfig {
        atomic_root_c_margin_iterations: ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS + 1,
        ..Default::default()
    };
    let error = over.validate().expect_err("work above the cap must refuse");
    assert_invalid_config_contains(&error, "atomic_root_c_margin_iterations");
}
