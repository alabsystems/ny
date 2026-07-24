// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for precomputed tangent point tables for sound GELU relaxation.

use super::super::eval::{
    check_lower_gelu_f64, check_lower_gelu_tanh_f64, check_upper_gelu_f64,
    check_upper_gelu_tanh_f64, gelu_critical_point, gelu_derivative, gelu_erf, gelu_tanh,
    gelu_tanh_inflection_point,
};
use super::super::GeluApproximation;
use super::*;

const TABLE_VALUE_TOL: f32 = 1e-4;
const TABLE_ENTRY_SAMPLE_STRIDE: usize = 10;
const INTERVAL_SOUND_GRID_INTERVALS: u32 = 20;

fn push_unique_index(indices: &mut Vec<usize>, len: usize, idx: usize) {
    if idx < len && !indices.contains(&idx) {
        indices.push(idx);
    }
}

fn representative_lower_table_indices(len: usize) -> Vec<usize> {
    let mut indices = Vec::new();
    for idx in [
        0,
        1,
        2,
        10,
        100,
        1_000,
        len / 4,
        len / 2,
        (3 * len) / 4,
        len.saturating_sub(1),
    ] {
        push_unique_index(&mut indices, len, idx);
    }
    indices
}

fn representative_clamped_table_indices(len: usize, active_span: usize) -> Vec<usize> {
    let mut indices = Vec::new();
    for idx in [
        0,
        1,
        2,
        10,
        active_span.saturating_sub(1),
        active_span,
        active_span.saturating_add(1),
        active_span.saturating_mul(2),
        len / 2,
        len.saturating_sub(1),
    ] {
        push_unique_index(&mut indices, len, idx);
    }
    indices
}

fn assert_table_value_close(table_name: &str, index: usize, actual: f32, expected: f32) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= TABLE_VALUE_TOL,
        "{table_name}[{index}] expected {expected}, got {actual} (diff={diff:.2e}, tol={TABLE_VALUE_TOL})",
    );
}

fn expected_erf_lower_right(upper: f32) -> f32 {
    let upper = upper as f64;
    let mut r = 1.0_f64;
    let mut l = -1.0_f64;

    for _ in 0..200 {
        if check_lower_gelu_f64(upper, l) {
            break;
        }
        l *= 2.0;
    }

    for _ in 0..100 {
        let m = f64::midpoint(l, r);
        if check_lower_gelu_f64(upper, m) {
            l = m;
        } else {
            r = m;
        }
    }

    l as f32
}

fn expected_erf_upper_right(lower: f32) -> f32 {
    let lower = lower as f64;
    let mut l = std::f64::consts::SQRT_2;
    let mut r = 1000.0_f64;

    for _ in 0..200 {
        if check_upper_gelu_f64(lower, r) {
            break;
        }
        r *= 2.0;
    }

    for _ in 0..100 {
        let m = f64::midpoint(l, r);
        if check_upper_gelu_f64(lower, m) {
            r = m;
        } else {
            l = m;
        }
    }

    r as f32
}

fn expected_erf_lower_left(upper: f32) -> f32 {
    let upper = upper as f64;
    let mut l = -std::f64::consts::SQRT_2;
    let mut r = GELU_MINIMIZER_X as f64;

    for _ in 0..200 {
        if check_lower_gelu_f64(upper, r) {
            break;
        }
        r *= 2.0;
    }

    for _ in 0..100 {
        let m = f64::midpoint(l, r);
        if check_lower_gelu_f64(upper, m) {
            r = m;
        } else {
            l = m;
        }
    }

    r as f32
}

fn expected_erf_upper_left(lower: f32) -> f32 {
    let lower = lower as f64;
    let mut l = -1000.0_f64;
    let mut r = -std::f64::consts::SQRT_2;

    for _ in 0..200 {
        if check_upper_gelu_f64(lower, l) {
            break;
        }
        l *= 2.0;
    }

    for _ in 0..100 {
        let m = f64::midpoint(l, r);
        if check_upper_gelu_f64(lower, m) {
            r = m;
        } else {
            l = m;
        }
    }

    r as f32
}

fn expected_tanh_lower_right(upper: f32) -> f32 {
    let upper = upper as f64;
    let mut r = 1.0_f64;
    let mut l = -1.0_f64;

    for _ in 0..200 {
        if check_lower_gelu_tanh_f64(upper, l) {
            break;
        }
        l *= 2.0;
    }

    for _ in 0..100 {
        let m = f64::midpoint(l, r);
        if check_lower_gelu_tanh_f64(upper, m) {
            l = m;
        } else {
            r = m;
        }
    }

    l as f32
}

fn expected_tanh_upper_right(lower: f32) -> f32 {
    let lower = lower as f64;
    let split = gelu_tanh_inflection_point() as f64;
    let mut l = split;
    let mut r = 1000.0_f64;

    for _ in 0..200 {
        if check_upper_gelu_tanh_f64(lower, r) {
            break;
        }
        r *= 2.0;
    }

    for _ in 0..100 {
        let m = f64::midpoint(l, r);
        if check_upper_gelu_tanh_f64(lower, m) {
            r = m;
        } else {
            l = m;
        }
    }

    r as f32
}

fn expected_tanh_lower_left(upper: f32) -> f32 {
    let upper = upper as f64;
    let split = gelu_tanh_inflection_point() as f64;
    let mut l = -split;
    let mut r = gelu_critical_point(GeluApproximation::Tanh) as f64;

    for _ in 0..200 {
        if check_lower_gelu_tanh_f64(upper, r) {
            break;
        }
        r *= 2.0;
    }

    for _ in 0..100 {
        let m = f64::midpoint(l, r);
        if check_lower_gelu_tanh_f64(upper, m) {
            r = m;
        } else {
            l = m;
        }
    }

    r as f32
}

fn expected_tanh_upper_left(lower: f32) -> f32 {
    let lower = lower as f64;
    let split = gelu_tanh_inflection_point() as f64;
    let mut l = -1000.0_f64;
    let mut r = -split;

    for _ in 0..200 {
        if check_upper_gelu_tanh_f64(lower, l) {
            break;
        }
        l *= 2.0;
    }

    for _ in 0..100 {
        let m = f64::midpoint(l, r);
        if check_upper_gelu_tanh_f64(lower, m) {
            r = m;
        } else {
            l = m;
        }
    }

    r as f32
}

fn assert_lower_bound_sound_on_grid(
    interval_lower: f32,
    interval_upper: f32,
    slope: f32,
    intercept: f32,
    gelu_fn: fn(f32) -> f32,
    table_name: &str,
    index: usize,
) {
    for sample in 0..=INTERVAL_SOUND_GRID_INTERVALS {
        let t = sample as f32 / INTERVAL_SOUND_GRID_INTERVALS as f32;
        let x = interval_lower + (interval_upper - interval_lower) * t;
        let line = slope * x + intercept;
        let gx = gelu_fn(x);
        assert!(
            line <= gx + POSTHOC_TOL,
            "{table_name}[{index}] lower bound {line} > gelu({x})={gx} on [{interval_lower}, {interval_upper}]",
        );
    }
}

fn assert_upper_bound_sound_on_grid(
    interval_lower: f32,
    interval_upper: f32,
    slope: f32,
    intercept: f32,
    gelu_fn: fn(f32) -> f32,
    table_name: &str,
    index: usize,
) {
    for sample in 0..=INTERVAL_SOUND_GRID_INTERVALS {
        let t = sample as f32 / INTERVAL_SOUND_GRID_INTERVALS as f32;
        let x = interval_lower + (interval_upper - interval_lower) * t;
        let line = slope * x + intercept;
        let gx = gelu_fn(x);
        assert!(
            line >= gx - POSTHOC_TOL,
            "{table_name}[{index}] upper bound {line} < gelu({x})={gx} on [{interval_lower}, {interval_upper}]",
        );
    }
}

// =========================================================================
// Table construction — basic sanity
// =========================================================================

#[test]
fn test_erf_table_construction() {
    let tables = get_gelu_precompute();
    assert_eq!(tables.step_pre, 0.01);
    assert!(!tables.d_lower_right.is_empty());
    assert!(!tables.d_lower_left.is_empty());
    assert!(!tables.d_upper_right.is_empty());
    assert!(!tables.d_upper_left.is_empty());
    // All four tables should have same length
    let n = tables.d_lower_right.len();
    assert_eq!(tables.d_lower_left.len(), n);
    assert_eq!(tables.d_upper_right.len(), n);
    assert_eq!(tables.d_upper_left.len(), n);

    for i in representative_lower_table_indices(n) {
        let upper = tables.step_pre * i as f32 + SQRT_2;
        let expected = expected_erf_lower_right(upper);
        assert_table_value_close("d_lower_right", i, tables.d_lower_right[i], expected);
    }

    let erf_active_span = (SQRT_2 / tables.step_pre).ceil() as usize;
    for i in representative_clamped_table_indices(n, erf_active_span) {
        let lower = (SQRT_2 - tables.step_pre * i as f32).max(tables.step_pre);
        let expected = expected_erf_upper_right(lower);
        assert_table_value_close("d_upper_right", i, tables.d_upper_right[i], expected);
    }

    for i in representative_lower_table_indices(n) {
        let upper = -(tables.step_pre * i as f32) - SQRT_2;
        let expected = expected_erf_lower_left(upper);
        assert_table_value_close("d_lower_left", i, tables.d_lower_left[i], expected);
    }

    for i in representative_clamped_table_indices(n, erf_active_span) {
        let lower = (tables.step_pre * i as f32 - SQRT_2).min(0.0);
        let expected = expected_erf_upper_left(lower);
        assert_table_value_close("d_upper_left", i, tables.d_upper_left[i], expected);
    }
}

#[test]
fn test_tanh_table_construction() {
    let tables = get_gelu_tanh_precompute();
    assert_eq!(tables.step_pre, 0.01);
    assert!(!tables.d_lower_right.is_empty());
    assert!(!tables.d_lower_left.is_empty());
    assert!(!tables.d_upper_right.is_empty());
    assert!(!tables.d_upper_left.is_empty());
    let n = tables.d_lower_right.len();
    assert_eq!(tables.d_lower_left.len(), n);
    assert_eq!(tables.d_upper_right.len(), n);
    assert_eq!(tables.d_upper_left.len(), n);

    let split = gelu_tanh_inflection_point();
    for i in representative_lower_table_indices(n) {
        let upper = tables.step_pre * i as f32 + split;
        let expected = expected_tanh_lower_right(upper);
        assert_table_value_close("tanh d_lower_right", i, tables.d_lower_right[i], expected);
    }

    let tanh_active_span = (split / tables.step_pre).ceil() as usize;
    for i in representative_clamped_table_indices(n, tanh_active_span) {
        let lower = (split - tables.step_pre * i as f32).max(tables.step_pre);
        let expected = expected_tanh_upper_right(lower);
        assert_table_value_close("tanh d_upper_right", i, tables.d_upper_right[i], expected);
    }

    for i in representative_lower_table_indices(n) {
        let upper = -(tables.step_pre * i as f32) - split;
        let expected = expected_tanh_lower_left(upper);
        assert_table_value_close("tanh d_lower_left", i, tables.d_lower_left[i], expected);
    }

    for i in representative_clamped_table_indices(n, tanh_active_span) {
        let lower = (tables.step_pre * i as f32 - split).min(0.0);
        let expected = expected_tanh_upper_left(lower);
        assert_table_value_close("tanh d_upper_left", i, tables.d_upper_left[i], expected);
    }
}

#[test]
fn test_table_entries_are_finite() {
    let erf_tables = get_gelu_precompute();
    for v in &erf_tables.d_lower_right {
        assert!(v.is_finite(), "d_lower_right has non-finite entry: {}", v);
    }
    for v in &erf_tables.d_lower_left {
        assert!(v.is_finite(), "d_lower_left has non-finite entry: {}", v);
    }
    for v in &erf_tables.d_upper_right {
        assert!(v.is_finite(), "d_upper_right has non-finite entry: {}", v);
    }
    for v in &erf_tables.d_upper_left {
        assert!(v.is_finite(), "d_upper_left has non-finite entry: {}", v);
    }

    let tanh_tables = get_gelu_tanh_precompute();
    for v in &tanh_tables.d_lower_right {
        assert!(
            v.is_finite(),
            "tanh d_lower_right has non-finite entry: {}",
            v
        );
    }
    for v in &tanh_tables.d_lower_left {
        assert!(
            v.is_finite(),
            "tanh d_lower_left has non-finite entry: {}",
            v
        );
    }
    for v in &tanh_tables.d_upper_right {
        assert!(
            v.is_finite(),
            "tanh d_upper_right has non-finite entry: {}",
            v
        );
    }
    for v in &tanh_tables.d_upper_left {
        assert!(
            v.is_finite(),
            "tanh d_upper_left has non-finite entry: {}",
            v
        );
    }
}

// =========================================================================
// retrieve() — index boundary conditions
// =========================================================================

#[test]
fn test_retrieve_zero_bound() {
    let tables = get_gelu_precompute();
    // bound=0.0 -> idx = floor(0.0/0.01) + 1 = 1
    let val = tables.retrieve(&tables.d_lower_right, 0.0, -999.0);
    assert_eq!(val, tables.d_lower_right[1], "bound=0.0 should read idx=1");
}

#[test]
fn test_retrieve_negative_bound_clamps_to_zero() {
    let tables = get_gelu_precompute();
    // bound=-1.0 -> idx = floor(-100) + 1 = -99 -> clamped to 0
    let val = tables.retrieve(&tables.d_lower_right, -1.0, -999.0);
    assert_eq!(
        val, tables.d_lower_right[0],
        "negative bound should clamp idx to 0, not use default"
    );
}

#[test]
fn test_retrieve_very_large_bound_returns_default() {
    let tables = get_gelu_precompute();
    // bound way beyond table range -> default
    let val = tables.retrieve(&tables.d_lower_right, 1e6, -999.0);
    assert_eq!(val, -999.0, "huge bound should return default");
}

#[test]
fn test_retrieve_extreme_bound_no_overflow_panic() {
    // Regression: bounds from multi-block IBP can be enormous (1e15+).
    // Before fix, (bound / 0.01).floor() as isize + 1 overflowed isize.
    let erf_tables = get_gelu_precompute();
    let tanh_tables = get_gelu_tanh_precompute();

    for &extreme in &[1e10_f32, 1e15, 1e30, f32::MAX, f32::INFINITY, f32::NAN] {
        let val = erf_tables.retrieve(&erf_tables.d_lower_right, extreme, -999.0);
        assert_eq!(
            val, -999.0,
            "erf retrieve({}) should return default",
            extreme
        );
        let val = tanh_tables.retrieve(&tanh_tables.d_lower_right, extreme, -999.0);
        assert_eq!(
            val, -999.0,
            "tanh retrieve({}) should return default",
            extreme
        );
    }
}

#[test]
fn test_retrieve_at_step_boundaries() {
    let tables = get_gelu_precompute();
    // Exact step boundary: bound=0.01 -> idx = floor(1.0) + 1 = 2
    let val_exact = tables.retrieve(&tables.d_lower_right, 0.01, -999.0);
    // Just below: bound=0.009 -> idx = floor(0.9) + 1 = 1
    let val_below = tables.retrieve(&tables.d_lower_right, 0.009, -999.0);
    assert_eq!(
        val_exact, tables.d_lower_right[2],
        "bound=0.01 should read idx=2"
    );
    assert_eq!(
        val_below, tables.d_lower_right[1],
        "bound=0.009 should read idx=1"
    );
}

// =========================================================================
// Tangent line functions
// =========================================================================

#[test]
fn test_gelu_tangent_at_zero() {
    let (slope, lower_i, upper_i) = gelu_tangent_at(0.0, 10.0);
    // GELU(0) = 0, GELU'(0) = 0.5
    assert!(
        (slope - 0.5).abs() < 1e-3,
        "GELU'(0) should be ~0.5, got {}",
        slope
    );
    // With directed rounding, lower_i <= true intercept <= upper_i.
    // True intercept at 0 is 0.
    assert!(
        lower_i <= 1e-3,
        "lower intercept at 0 should be <= ~0, got {}",
        lower_i
    );
    assert!(
        upper_i >= -1e-3,
        "upper intercept at 0 should be >= ~0, got {}",
        upper_i
    );
}

#[test]
fn test_gelu_tangent_at_large_positive() {
    // For large x, GELU(x) ≈ x, so GELU'(x) ≈ 1.
    let (slope, lower_i, upper_i) = gelu_tangent_at(5.0, 10.0);
    assert!(
        (slope - 1.0).abs() < 0.01,
        "GELU'(5) should be ~1.0, got {}",
        slope
    );
    // Tangent line at x=5 should bracket GELU(5).
    let gelu_at_5 = gelu_erf(5.0);
    assert!(
        slope * 5.0 + lower_i <= gelu_at_5 + 1e-3,
        "lower tangent should be <= GELU(5)"
    );
    assert!(
        slope * 5.0 + upper_i >= gelu_at_5 - 1e-3,
        "upper tangent should be >= GELU(5)"
    );
}

#[test]
fn test_gelu_tangent_at_large_negative() {
    // For large negative x, GELU(x) ≈ 0, so GELU'(x) ≈ 0.
    let (slope, lower_i, upper_i) = gelu_tangent_at(-5.0, 10.0);
    assert!(slope.abs() < 0.01, "GELU'(-5) should be ~0, got {}", slope);
    let gelu_at_neg5 = gelu_erf(-5.0);
    assert!(
        slope * (-5.0) + lower_i <= gelu_at_neg5 + 1e-3,
        "lower tangent should be <= GELU(-5)"
    );
    assert!(
        slope * (-5.0) + upper_i >= gelu_at_neg5 - 1e-3,
        "upper tangent should be >= GELU(-5)"
    );
}

#[test]
fn test_gelu_tanh_tangent_at_zero() {
    let (slope, lower_i, upper_i) = gelu_tanh_tangent_at(0.0, 10.0);
    assert!(
        (slope - 0.5).abs() < 0.01,
        "GELU_tanh'(0) should be ~0.5, got {}",
        slope
    );
    assert!(
        lower_i <= 1e-3,
        "lower intercept should be <= ~0, got {}",
        lower_i
    );
    assert!(
        upper_i >= -1e-3,
        "upper intercept should be >= ~0, got {}",
        upper_i
    );
}

#[test]
fn test_tangent_line_brackets_point() {
    // For any d, the tangent line at d should bracket GELU(d):
    // slope * d + lower_i <= GELU(d) <= slope * d + upper_i
    for &d in &[-2.0_f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
        let max_abs_x = d.abs().max(5.0);
        let (slope, lower_i, upper_i) = gelu_tangent_at(d, max_abs_x);
        let gelu_val = gelu_erf(d);
        assert!(
            slope * d + lower_i <= gelu_val + 1e-4,
            "lower tangent at d={} should be <= GELU(d)={}, got {}",
            d,
            gelu_val,
            slope * d + lower_i,
        );
        assert!(
            slope * d + upper_i >= gelu_val - 1e-4,
            "upper tangent at d={} should be >= GELU(d)={}, got {}",
            d,
            gelu_val,
            slope * d + upper_i,
        );
    }
}

#[test]
fn test_tanh_tangent_line_brackets_point() {
    for &d in &[-2.0_f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
        let max_abs_x = d.abs().max(5.0);
        let (slope, lower_i, upper_i) = gelu_tanh_tangent_at(d, max_abs_x);
        let gelu_val = gelu_tanh(d);
        assert!(
            slope * d + lower_i <= gelu_val + 1e-4,
            "tanh lower tangent at d={} should be <= GELU_tanh(d)={}, got {}",
            d,
            gelu_val,
            slope * d + lower_i,
        );
        assert!(
            slope * d + upper_i >= gelu_val - 1e-4,
            "tanh upper tangent at d={} should be >= GELU_tanh(d)={}, got {}",
            d,
            gelu_val,
            slope * d + upper_i,
        );
    }
}

// =========================================================================
// gelu_posthoc_adjust — soundness property
// =========================================================================

#[test]
fn test_posthoc_adjust_no_violation() {
    // Tangent at x=0 for interval [-1, 1]:
    // GELU'(0) = 0.5, GELU(0) = 0 -> line: y = 0.5x
    // This is actually a good lower bound near 0 for GELU over [-1,1].
    let (ls, li, us, ui) = gelu_posthoc_adjust(
        -1.0, 1.0, 0.5, 0.0, // lower: y = 0.5x
        1.0, 0.0, // upper: y = x (always >= GELU for x in [-1,1])
        gelu_erf,
    );
    // Lower intercept should be adjusted down (safety eps)
    assert!(
        li <= 0.0,
        "lower intercept should be <= 0 after adjust, got {}",
        li
    );
    // Upper intercept should be adjusted up (safety eps)
    assert!(
        ui >= 0.0,
        "upper intercept should be >= 0 after adjust, got {}",
        ui
    );
    // Slopes should be unchanged
    assert_eq!(ls, 0.5);
    assert_eq!(us, 1.0);
}

#[test]
fn test_posthoc_adjust_corrects_lower_violation() {
    // Intentionally create a lower bound that violates soundness:
    // line y = 0.5x + 0.5 is ABOVE GELU(x) at x=0 (line=0.5, GELU(0)=0).
    let (ls, li, us, ui) = gelu_posthoc_adjust(
        -1.0, 1.0, 0.5, 0.5, // lower: y = 0.5x + 0.5 (too high at x=0)
        1.0, 1.0, // upper: y = x + 1 (safely above)
        gelu_erf,
    );
    // After adjustment, lower line should be valid at all sample points
    let points = [-1.0_f32, 1.0, -0.25, 0.0, 0.25];
    for &x in &points {
        let lower_line = ls * x + li;
        let gx = gelu_erf(x);
        assert!(
            lower_line <= gx + 1e-4,
            "adjusted lower at x={}: line={} > gelu={}",
            x,
            lower_line,
            gx,
        );
    }
    assert_eq!(ls, 0.5, "slope should be preserved");
    // Upper should still be valid
    for &x in &points {
        let upper_line = us * x + ui;
        let gx = gelu_erf(x);
        assert!(
            upper_line >= gx - 1e-4,
            "adjusted upper at x={}: line={} < gelu={}",
            x,
            upper_line,
            gx,
        );
    }
}

#[test]
fn test_posthoc_adjust_corrects_upper_violation() {
    // Intentionally create an upper bound that violates soundness:
    // line y = 0.5x - 0.5 is BELOW GELU(x) at x=1 (line=0, GELU(1)≈0.841).
    let (_ls, _li, us, ui) = gelu_posthoc_adjust(
        -1.0, 1.0, 0.0, -1.0, // lower: y = -1 (safely below)
        0.5, -0.5, // upper: y = 0.5x - 0.5 (too low)
        gelu_erf,
    );
    let points = [-1.0_f32, 1.0, -0.25, 0.0, 0.25];
    for &x in &points {
        let upper_line = us * x + ui;
        let gx = gelu_erf(x);
        assert!(
            upper_line >= gx - 1e-4,
            "adjusted upper at x={}: line={} < gelu={}",
            x,
            upper_line,
            gx,
        );
    }
}

#[test]
fn test_posthoc_adjust_with_tanh_gelu() {
    let (ls, li, us, ui) = gelu_posthoc_adjust(-2.0, 2.0, 0.5, 0.0, 1.0, 0.5, gelu_tanh);
    // Verify soundness at sample points
    for &x in &[-2.0_f32, -1.0, 0.0, 1.0, 2.0] {
        let lower_line = ls * x + li;
        let upper_line = us * x + ui;
        let gx = gelu_tanh(x);
        assert!(
            lower_line <= gx + 1e-4,
            "tanh lower at x={}: {} > {}",
            x,
            lower_line,
            gx,
        );
        assert!(
            upper_line >= gx - 1e-4,
            "tanh upper at x={}: {} < {}",
            x,
            upper_line,
            gx,
        );
    }
}

fn numeric_second_derivative_gelu(x: f32, approximation: GeluApproximation) -> f32 {
    // Scale h with |x| to keep finite differences stable across wide domains.
    let h = 1e-4_f32 * (1.0 + x.abs());
    let d_plus = gelu_derivative(x + h, approximation);
    let d_minus = gelu_derivative(x - h, approximation);
    (d_plus - d_minus) / (2.0 * h)
}

#[test]
fn test_gelu_erf_curvature_bound_global() {
    // For GELU_erf(x) = x * Phi(x), the closed-form second derivative is:
    //   GELU''(x) = phi(x) * (2 - x^2)
    // with global max absolute value at x = 0: sqrt(2/pi) ≈ 0.79788456.
    let inv_sqrt_2pi = 1.0_f32 / (2.0 * std::f32::consts::PI).sqrt();
    let mut max_abs_d2 = 0.0_f32;
    for i in -12000..=12000 {
        let x = i as f32 / 1000.0;
        let pdf = inv_sqrt_2pi * (-0.5 * x * x).exp();
        let d2 = pdf * (2.0 - x * x);
        max_abs_d2 = max_abs_d2.max(d2.abs());
    }
    assert!(
        max_abs_d2 <= 0.8,
        "erf GELU curvature exceeds 0.8: max |GELU''| = {}",
        max_abs_d2,
    );
}

#[test]
fn test_gelu_tanh_curvature_bound_global_sampled() {
    // The tanh approximation tracks erf closely; sampled finite differences
    // verify its global second-derivative magnitude remains <= 0.8.
    let mut max_abs_d2 = 0.0_f32;
    for i in -12000..=12000 {
        let x = i as f32 / 1000.0;
        let d2 = numeric_second_derivative_gelu(x, GeluApproximation::Tanh);
        max_abs_d2 = max_abs_d2.max(d2.abs());
    }
    assert!(
        max_abs_d2 <= 0.8,
        "tanh GELU sampled curvature exceeds 0.8: max |GELU''| = {}",
        max_abs_d2,
    );
}

// =========================================================================
// Table soundness: production path (table + posthoc_adjust) produces valid bounds
// =========================================================================
//
// Raw table entries store f32-truncated tangent points from f64 bisection.
// The f32 truncation can shift the tangent point enough that the raw entry
// violates the f64 soundness check. This is expected and handled by the
// production code via `gelu_posthoc_adjust`, which adds safety margins.
//
// These tests verify the end-to-end production path: table lookup → tangent
// computation → posthoc adjustment → soundness check. This is what matters
// for correctness of bound propagation.
//
// Tolerance: 1e-5 matches SAFETY_EPS in gelu_posthoc_adjust. After adjustment,
// bounds should be sound within f32 rounding (~1e-7 for typical GELU values).
// Using 1e-5 ensures the test would fail if posthoc_adjust stopped applying
// its safety margin.
const POSTHOC_TOL: f32 = 1e-5;

/// Verify that the erf GELU sound relaxation (table + posthoc) produces valid
/// lower and upper bounds at sampled boundary points.
#[test]
fn test_erf_d_lower_right_soundness_sampled() {
    let tables = get_gelu_precompute();
    let step = tables.step_pre;
    let sqrt2 = SQRT_2;

    for i in (0..tables.d_lower_right.len()).step_by(TABLE_ENTRY_SAMPLE_STRIDE) {
        let upper = step * (i as f32) + sqrt2;
        // Use a fixed lower bound below sqrt2 to form a valid interval
        let lower = -1.0_f32;
        let max_abs_x = lower.abs().max(upper.abs());
        let (slope, lower_i, _upper_i) = gelu_tangent_at(tables.d_lower_right[i], max_abs_x);
        // Apply posthoc adjustment as the production code does
        let (ls, li, _us, _ui) =
            gelu_posthoc_adjust(lower, upper, slope, lower_i, 1.0, 1.0, gelu_erf);
        assert_lower_bound_sound_on_grid(lower, upper, ls, li, gelu_erf, "d_lower_right", i);
    }
}

#[test]
fn test_erf_d_upper_right_soundness_sampled() {
    let tables = get_gelu_precompute();
    let step = tables.step_pre;
    let sqrt2 = SQRT_2;

    for i in (0..tables.d_upper_right.len()).step_by(TABLE_ENTRY_SAMPLE_STRIDE) {
        let lower = (sqrt2 - step * (i as f32)).max(step);
        let upper = 3.0_f32; // fixed upper to form a valid interval
        let max_abs_x = lower.abs().max(upper.abs());
        let (_slope, _lower_i, upper_i) = gelu_tangent_at(tables.d_upper_right[i], max_abs_x);
        let (_ls, _li, us, ui) =
            gelu_posthoc_adjust(lower, upper, 0.0, -1.0, _slope, upper_i, gelu_erf);
        assert_upper_bound_sound_on_grid(lower, upper, us, ui, gelu_erf, "d_upper_right", i);
    }
}

#[test]
fn test_erf_d_lower_left_soundness_sampled() {
    let tables = get_gelu_precompute();
    let step = tables.step_pre;
    let sqrt2 = SQRT_2;

    for i in (0..tables.d_lower_left.len()).step_by(TABLE_ENTRY_SAMPLE_STRIDE) {
        let upper = -(step * (i as f32)) - sqrt2;
        let lower = upper - 1.0; // form valid interval below upper
        let max_abs_x = lower.abs().max(upper.abs());
        let (slope, lower_i, _upper_i) = gelu_tangent_at(tables.d_lower_left[i], max_abs_x);
        let (ls, li, _us, _ui) =
            gelu_posthoc_adjust(lower, upper, slope, lower_i, 1.0, 1.0, gelu_erf);
        assert_lower_bound_sound_on_grid(lower, upper, ls, li, gelu_erf, "d_lower_left", i);
    }
}

#[test]
fn test_erf_d_upper_left_soundness_sampled() {
    let tables = get_gelu_precompute();
    let step = tables.step_pre;
    let sqrt2 = SQRT_2;

    for i in (0..tables.d_upper_left.len()).step_by(TABLE_ENTRY_SAMPLE_STRIDE) {
        let lower = (step * (i as f32) - sqrt2).min(0.0);
        let upper = lower + 1.0; // form valid interval above lower
        let max_abs_x = lower.abs().max(upper.abs());
        let (_slope, _lower_i, upper_i) = gelu_tangent_at(tables.d_upper_left[i], max_abs_x);
        let (_ls, _li, us, ui) =
            gelu_posthoc_adjust(lower, upper, 0.0, -1.0, _slope, upper_i, gelu_erf);
        assert_upper_bound_sound_on_grid(lower, upper, us, ui, gelu_erf, "d_upper_left", i);
    }
}

/// Same soundness check for tanh-approximation tables via posthoc-adjusted path.
#[test]
fn test_tanh_d_lower_right_soundness_sampled() {
    let tables = get_gelu_tanh_precompute();
    let split = gelu_tanh_inflection_point();
    let step = tables.step_pre;

    for i in (0..tables.d_lower_right.len()).step_by(TABLE_ENTRY_SAMPLE_STRIDE) {
        let upper = step * (i as f32) + split;
        let lower = -1.0_f32;
        let max_abs_x = lower.abs().max(upper.abs());
        let (slope, lower_i, _upper_i) = gelu_tanh_tangent_at(tables.d_lower_right[i], max_abs_x);
        let (ls, li, _us, _ui) =
            gelu_posthoc_adjust(lower, upper, slope, lower_i, 1.0, 1.0, gelu_tanh);
        assert_lower_bound_sound_on_grid(lower, upper, ls, li, gelu_tanh, "tanh d_lower_right", i);
    }
}

#[test]
fn test_tanh_d_upper_right_soundness_sampled() {
    let tables = get_gelu_tanh_precompute();
    let split = gelu_tanh_inflection_point();
    let step = tables.step_pre;

    for i in (0..tables.d_upper_right.len()).step_by(TABLE_ENTRY_SAMPLE_STRIDE) {
        let lower = (split - step * (i as f32)).max(step);
        let upper = 3.0_f32;
        let max_abs_x = lower.abs().max(upper.abs());
        let (_slope, _lower_i, upper_i) = gelu_tanh_tangent_at(tables.d_upper_right[i], max_abs_x);
        let (_ls, _li, us, ui) =
            gelu_posthoc_adjust(lower, upper, 0.0, -1.0, _slope, upper_i, gelu_tanh);
        assert_upper_bound_sound_on_grid(lower, upper, us, ui, gelu_tanh, "tanh d_upper_right", i);
    }
}

/// Soundness check for tanh d_lower_left table via posthoc-adjusted path.
/// Part of #1905.
#[test]
fn test_tanh_d_lower_left_soundness_sampled() {
    let tables = get_gelu_tanh_precompute();
    let split = gelu_tanh_inflection_point();
    let step = tables.step_pre;

    for i in (0..tables.d_lower_left.len()).step_by(TABLE_ENTRY_SAMPLE_STRIDE) {
        let upper = -(step * (i as f32)) - split;
        let lower = upper - 1.0;
        let max_abs_x = lower.abs().max(upper.abs());
        let (slope, lower_i, _upper_i) = gelu_tanh_tangent_at(tables.d_lower_left[i], max_abs_x);
        let (ls, li, _us, _ui) =
            gelu_posthoc_adjust(lower, upper, slope, lower_i, 1.0, 1.0, gelu_tanh);
        assert_lower_bound_sound_on_grid(lower, upper, ls, li, gelu_tanh, "tanh d_lower_left", i);
    }
}

/// Soundness check for tanh d_upper_left table via posthoc-adjusted path.
/// Part of #1905.
#[test]
fn test_tanh_d_upper_left_soundness_sampled() {
    let tables = get_gelu_tanh_precompute();
    let split = gelu_tanh_inflection_point();
    let step = tables.step_pre;

    for i in (0..tables.d_upper_left.len()).step_by(TABLE_ENTRY_SAMPLE_STRIDE) {
        let lower = (step * (i as f32) - split).min(0.0);
        let upper = lower + 1.0;
        let max_abs_x = lower.abs().max(upper.abs());
        let (_slope, _lower_i, upper_i) = gelu_tanh_tangent_at(tables.d_upper_left[i], max_abs_x);
        let (_ls, _li, us, ui) =
            gelu_posthoc_adjust(lower, upper, 0.0, -1.0, _slope, upper_i, gelu_tanh);
        assert_upper_bound_sound_on_grid(lower, upper, us, ui, gelu_tanh, "tanh d_upper_left", i);
    }
}

// =========================================================================
// Constants
// =========================================================================

#[test]
fn test_gelu_minimizer_is_near_global_min() {
    // The GELU global minimum for x<0 is near x ≈ -0.7518.
    // gelu_critical_point should agree with our constant.
    let critical = gelu_critical_point(GeluApproximation::Erf);
    assert!(
        (critical - GELU_MINIMIZER_X).abs() < 0.01,
        "GELU_MINIMIZER_X={} should be near critical point {}",
        GELU_MINIMIZER_X,
        critical,
    );
}

#[test]
fn test_sqrt2_constant() {
    assert!((SQRT_2 - std::f32::consts::SQRT_2).abs() < f32::EPSILON);
}
