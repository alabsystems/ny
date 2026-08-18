// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn early_exit_projection_stays_outward_under_catastrophic_cancellation() {
    let large = 2.0_f32.powi(30);
    let exit = AlphaSpecEarlyExit {
        objective: vec![large, 1.0, -large],
        threshold: 0.5,
        verify_upper_bound: true,
    };
    let point = [large, 1.0, large];
    let (lower, upper) = exit
        .project_bounds(&point, &point)
        .expect("matching widths");

    // Exact binary value is one. The upper endpoint must not collapse near
    // zero and falsely satisfy the `< 0.5` verdict early-exit.
    assert!(lower <= 1.0);
    assert!(upper >= 1.0);
    assert!(lower > 0.99 && upper < 1.01);
    assert!(!exit.is_verified(lower, upper));
}

#[test]
fn early_exit_rejects_inverted_projected_interval() {
    for verify_upper_bound in [false, true] {
        let exit = AlphaSpecEarlyExit {
            objective: vec![1.0],
            threshold: 0.0,
            verify_upper_bound,
        };
        assert!(!exit.is_verified(1.0, -1.0));
    }
}

#[test]
fn test_should_save_best_default_half() {
    let config = AlphaCrownConfig {
        iterations: 100,
        start_save_best: 0.5,
        ..Default::default()
    };
    // Iteration 0: always save (baseline).
    assert!(config.should_save_best(0, false));
    // Iterations 1..=50: skip (warmup window).
    for iter in 1..=50 {
        assert!(
            !config.should_save_best(iter, false),
            "should skip at iter {iter}"
        );
    }
    // Iterations 51+: save.
    for iter in 51..=100 {
        assert!(
            config.should_save_best(iter, false),
            "should save at iter {iter}"
        );
    }
}

#[test]
fn test_should_save_best_zero_saves_every_iteration() {
    let config = AlphaCrownConfig {
        iterations: 10,
        start_save_best: 0.0,
        ..Default::default()
    };
    for iter in 0..10 {
        assert!(
            config.should_save_best(iter, false),
            "should save at iter {iter} when start_save_best=0"
        );
    }
}

#[test]
fn test_should_save_best_one_skips_all_but_zero() {
    let config = AlphaCrownConfig {
        iterations: 10,
        start_save_best: 1.0,
        ..Default::default()
    };
    // iter 0: always save
    assert!(config.should_save_best(0, false));
    // iter 1..=10: skip (threshold = 10, so iter must be > 10)
    for iter in 1..=10 {
        assert!(
            !config.should_save_best(iter, false),
            "should skip at iter {iter} when start_save_best=1.0"
        );
    }
}

#[test]
fn test_should_save_best_force_overrides_warmup() {
    let config = AlphaCrownConfig {
        iterations: 100,
        start_save_best: 0.5,
        ..Default::default()
    };
    // Iteration 25 is in the warmup window — normally skipped.
    assert!(!config.should_save_best(25, false));
    // With force=true, always saves (patience/stop criterion exit).
    assert!(config.should_save_best(25, true));
}

#[test]
fn test_should_save_best_default_value() {
    let config = AlphaCrownConfig::default();
    assert!((config.start_save_best - 0.5).abs() < f32::EPSILON);
}

#[test]
fn reference_refresh_defaults_preserve_the_historical_fraction_only_policy() {
    let config = AlphaCrownConfig::default();
    assert_eq!(
        config.reference_refresh_fraction,
        AlphaCrownConfig::DEFAULT_REFERENCE_REFRESH_FRACTION
    );
    assert_eq!(config.reference_refresh_max_secs, None);
    assert!(!config.forward_linear_deadline_fallback_to_ibp);
    assert_eq!(
        config.resolved_reference_refresh_fraction(),
        AlphaCrownConfig::DEFAULT_REFERENCE_REFRESH_FRACTION
    );
}

#[test]
fn reference_refresh_fraction_resolves_invalid_direct_configs_fail_closed() {
    for invalid in [
        f32::NAN,
        f32::NEG_INFINITY,
        f32::INFINITY,
        -0.25,
        0.0,
        0.009,
        1.001,
    ] {
        let config = AlphaCrownConfig {
            reference_refresh_fraction: invalid,
            ..Default::default()
        };
        assert_eq!(
            config.resolved_reference_refresh_fraction(),
            AlphaCrownConfig::DEFAULT_REFERENCE_REFRESH_FRACTION,
            "invalid direct fraction {invalid:?} must resolve to the safe default"
        );
    }
}

#[test]
fn reference_refresh_fields_are_backward_compatible_when_absent_from_serde() {
    let mut encoded =
        serde_json::to_value(AlphaCrownConfig::default()).expect("default config serializes");
    let object = encoded
        .as_object_mut()
        .expect("AlphaCrownConfig serializes as an object");
    object.remove("reference_refresh_fraction");
    object.remove("reference_refresh_max_secs");
    object.remove("forward_linear_deadline_fallback_to_ibp");

    let decoded: AlphaCrownConfig =
        serde_json::from_value(encoded).expect("older config without refresh fields deserializes");
    assert_eq!(
        decoded.reference_refresh_fraction,
        AlphaCrownConfig::DEFAULT_REFERENCE_REFRESH_FRACTION
    );
    assert_eq!(decoded.reference_refresh_max_secs, None);
    assert!(!decoded.forward_linear_deadline_fallback_to_ibp);
}

#[test]
fn direct_alpha_config_serde_rejects_unknown_fields() {
    let mut encoded =
        serde_json::to_value(AlphaCrownConfig::default()).expect("default config serializes");
    encoded
        .as_object_mut()
        .expect("serialized alpha config is an object")
        .insert(
            "forward_linear_deadline_fallback_to_ibpp".into(),
            serde_json::Value::Bool(true),
        );
    assert!(
        serde_json::from_value::<AlphaCrownConfig>(encoded).is_err(),
        "an unknown fallback-authority key must not silently retain its default"
    );
}

// ---------------------------------------------------------------------------
// #root-alpha-margin: AlphaSpecAscent — the ranking objective for root-warmup
// alpha selection. SELECTION ONLY: none of this decides a verdict.
// ---------------------------------------------------------------------------

/// Two margin rows over a 3-dim output: `y0 - y1` and `y0 - y2`, both `>= 0`.
fn two_margin_rows() -> Vec<AlphaSpecEarlyExit> {
    vec![
        AlphaSpecEarlyExit {
            objective: vec![1.0, -1.0, 0.0],
            threshold: 0.0,
            verify_upper_bound: false,
        },
        AlphaSpecEarlyExit {
            objective: vec![1.0, 0.0, -1.0],
            threshold: 0.0,
            verify_upper_bound: false,
        },
    ]
}

#[test]
fn ascent_rejects_empty_and_ragged_rows() {
    assert!(AlphaSpecAscent::new(Vec::new()).is_none(), "empty");

    let ragged = vec![
        AlphaSpecEarlyExit {
            objective: vec![1.0, -1.0],
            threshold: 0.0,
            verify_upper_bound: false,
        },
        AlphaSpecEarlyExit {
            objective: vec![1.0, -1.0, 0.0],
            threshold: 0.0,
            verify_upper_bound: false,
        },
    ];
    assert!(AlphaSpecAscent::new(ragged).is_none(), "ragged widths");

    let zero_width = vec![AlphaSpecEarlyExit {
        objective: Vec::new(),
        threshold: 0.0,
        verify_upper_bound: false,
    }];
    assert!(AlphaSpecAscent::new(zero_width).is_none(), "zero width");
}

#[test]
fn hinge_score_is_zero_exactly_when_every_row_is_proven() {
    let ascent = AlphaSpecAscent::new(two_margin_rows()).expect("valid");

    // y0 in [10, 11], y1 in [0, 1], y2 in [0, 1]: both margins strictly positive.
    let lower = [10.0f32, 0.0, 0.0];
    let upper = [11.0f32, 1.0, 1.0];
    assert_eq!(
        ascent.hinge_score(&lower, &upper),
        Some(0.0),
        "all rows proven => hinge 0"
    );
    assert_eq!(ascent.verified_rows(&lower, &upper), 2);
}

#[test]
fn hinge_score_counts_only_unproven_rows() {
    let ascent = AlphaSpecAscent::new(two_margin_rows()).expect("valid");

    // Row 0 proven (y0-y1 >= 10-1 = 9 > 0); row 1 unproven (y0-y2 >= 10-20 = -10).
    let lower = [10.0f32, 0.0, 0.0];
    let upper = [11.0f32, 1.0, 20.0];
    let score = ascent.hinge_score(&lower, &upper).expect("finite");
    assert!(score < 0.0, "an unproven row must pull the score negative");
    assert_eq!(ascent.verified_rows(&lower, &upper), 1);

    // Widening the ALREADY-PROVEN row's slack must not change the score: effort
    // spent padding a proven margin is exactly the pathology this replaces.
    let wider_lower = [40.0f32, 0.0, 0.0];
    let wider_upper = [41.0f32, 1.0, 50.0];
    let wider = ascent
        .hinge_score(&wider_lower, &wider_upper)
        .expect("finite");
    assert!(
        wider < 0.0,
        "row 1 still unproven at y2 up to 50 (40 - 50 = -10)"
    );
    assert!(
        (wider - score).abs() < 1e-3,
        "proven-row slack must not move the hinge: {score} vs {wider}"
    );
}

#[test]
fn hinge_score_orders_iterates_by_worst_row_recovery() {
    let ascent = AlphaSpecAscent::new(two_margin_rows()).expect("valid");

    // Worse iterate: both rows badly violated.
    let bad = ascent
        .hinge_score(&[0.0, 5.0, 5.0], &[0.0, 5.0, 5.0])
        .expect("finite");
    // Better iterate: violations halved.
    let better = ascent
        .hinge_score(&[0.0, 2.5, 2.5], &[0.0, 2.5, 2.5])
        .expect("finite");

    assert!(
        better > bad,
        "recovering margin must raise the score: bad={bad} better={better}"
    );
}

#[test]
fn hinge_score_fails_closed_on_length_mismatch() {
    let ascent = AlphaSpecAscent::new(two_margin_rows()).expect("valid");
    assert_eq!(
        ascent.hinge_score(&[0.0, 0.0], &[0.0, 0.0]),
        None,
        "width mismatch must fail closed so the caller keeps its current best"
    );
}

#[test]
fn margin_slack_sign_agrees_with_is_verified() {
    // Cross-check the ranking scalar against the verdict-grade predicate: a
    // positive slack must mean `is_verified`, and vice versa. If these ever
    // disagree the ranking is measuring something other than the property.
    let row = AlphaSpecEarlyExit {
        objective: vec![1.0, -1.0],
        threshold: 0.0,
        verify_upper_bound: false,
    };
    for (lo0, up1) in [(10.0f32, 1.0f32), (1.0, 10.0), (5.0, 5.0)] {
        let lower = [lo0, 0.0];
        let upper = [lo0 + 1.0, up1];
        let slack = row.margin_slack(&lower, &upper).expect("finite");
        let (plo, phi) = row.project_bounds(&lower, &upper).expect("projects");
        assert_eq!(
            slack > 0.0,
            row.is_verified(plo, phi),
            "slack sign must match is_verified for lo0={lo0} up1={up1}"
        );
    }
}

#[test]
fn spec_ascent_defaults_to_none_so_production_is_unchanged() {
    // The gate is the only thing that can populate this; a default config must
    // leave the warmup on its legacy last-iterate path.
    assert!(AlphaCrownConfig::default().spec_ascent.is_none());
}
