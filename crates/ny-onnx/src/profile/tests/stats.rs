// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_median() {
    assert_eq!(median(&[1.0, 2.0, 3.0]), 2.0);
    assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    assert_eq!(median(&[5.0]), 5.0);
    assert_eq!(median(&[]), 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_difficulty_score() {
    // No expansion
    assert_eq!(difficulty_score(1.0, 1.0, false), 0.0);

    // Overflow = max difficulty
    assert_eq!(difficulty_score(1.0, 1.0, true), 100.0);

    // Some expansion
    let score = difficulty_score(100.0, 10.0, false);
    assert!(score > 0.0 && score < 100.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_truncate_name() {
    assert_eq!(truncate_name("short", 10), "short");
    assert_eq!(truncate_name("very_long_layer_name", 10), "...er_name");
}

#[ntest::timeout(10000)]
#[test]
fn test_median_edge_cases() {
    // Empty input
    assert_eq!(median(&[]), 0.0);

    // Single element
    assert_eq!(median(&[5.0]), 5.0);

    // Two elements (even)
    assert_eq!(median(&[1.0, 3.0]), 2.0);

    // Odd number
    assert_eq!(median(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0);

    // Even number (already sorted)
    assert_eq!(median(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 3.5);

    // Unsorted input
    assert_eq!(median(&[5.0, 1.0, 3.0]), 3.0);

    // With infinite values (should filter them)
    assert_eq!(median(&[1.0, f32::INFINITY, 3.0, 2.0]), 2.0);

    // All infinite
    assert_eq!(median(&[f32::INFINITY, f32::NEG_INFINITY]), f32::INFINITY);

    // With NaN (should filter)
    assert_eq!(median(&[1.0, f32::NAN, 3.0, 2.0]), 2.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_difficulty_score_edge_cases() {
    // Both under 1.0
    assert_eq!(difficulty_score(0.5, 0.5, false), 0.0);

    // Expansion exactly 1.0
    assert_eq!(difficulty_score(1.0, 1.0, false), 0.0);

    // High expansion
    let high_expansion = difficulty_score(1e6, 1.0, false);
    assert!(high_expansion > 40.0);
    assert!(high_expansion <= 50.0);

    // High growth
    let high_growth = difficulty_score(1.0, 1e6, false);
    assert!(high_growth > 40.0);
    assert!(high_growth <= 50.0);

    // Both high
    let both_high = difficulty_score(1e6, 1e6, false);
    assert_eq!(both_high, 100.0);

    // Overflow always 100
    assert_eq!(difficulty_score(1.0, 1.0, true), 100.0);
    assert_eq!(difficulty_score(0.1, 0.1, true), 100.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_make_unit_variance_input() {
    let shape = &[2, 4];
    let epsilon = 0.01;
    let input = make_unit_variance_input(shape, epsilon).expect("valid unit variance input");

    assert_eq!(input.shape(), shape);

    // Check alternating pattern
    let data: Vec<f32> = input.lower().iter().cloned().collect();
    assert_eq!(data[0], 1.0 - epsilon);
    assert_eq!(data[1], -1.0 - epsilon);
    assert_eq!(data[2], 1.0 - epsilon);
    assert_eq!(data[3], -1.0 - epsilon);

    // Check that bounds have correct width
    let width = input.max_width();
    assert!((width - 2.0 * epsilon).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_truncate_name_edge_cases() {
    // Empty string
    assert_eq!(truncate_name("", 10), "");

    // Exactly width length
    assert_eq!(truncate_name("1234567890", 10), "1234567890");

    // One over: 11 chars, width 10 -> keep last 7 = "5678901"
    assert_eq!(truncate_name("12345678901", 10), "...5678901");

    // Width smaller than "...": width < 4 early-return takes first `width` chars
    assert_eq!(truncate_name("hello", 3), "hel");

    // Width 4 with "hello": 5-4+3=4, &name[4..] = "o" -> "...o"
    assert_eq!(truncate_name("hello", 4), "...o");
}
