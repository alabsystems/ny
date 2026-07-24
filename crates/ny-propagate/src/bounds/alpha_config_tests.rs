// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

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
