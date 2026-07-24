// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Equivalence tests for the fast-path NaN/Inf firewall scan in
//! `LinearBounds::validate_no_nan` / `new_or_conservative`.
//!
//! `validate_no_nan` was changed to scan a flat contiguous `&[f32]` slice
//! (via `as_slice()`) on the common standard-layout path, falling back to
//! ndarray's strided element iterator for non-contiguous arrays. This is a
//! PERF-ONLY change: the firewall must catch the EXACT same set of values
//! (NaN/Inf in coefficients, NaN in biases) as the original
//! `arr.iter().any(...)` and must still fall back to conservative bounds.
//!
//! These tests assert:
//! 1. Finite inputs pass through unchanged (no firewall trip).
//! 2. NaN/Inf in coefficients trips the firewall -> conservative bounds.
//! 3. The contiguous fast path and the strided fallback path agree, for both
//!    clean and poisoned inputs, against a reference `.iter().any()` oracle.

use crate::bounds::LinearBounds;
use ndarray::{Array1, Array2};
use proptest::prelude::*;

/// Reference oracle matching the original (pre-optimization) firewall
/// predicate: coefficients reject any non-finite value, biases reject NaN.
fn oracle_trips(la: &Array2<f32>, lb: &Array1<f32>, ua: &Array2<f32>, ub: &Array1<f32>) -> bool {
    la.iter().any(|v| !v.is_finite())
        || ua.iter().any(|v| !v.is_finite())
        || lb.iter().any(|v| v.is_nan())
        || ub.iter().any(|v| v.is_nan())
}

/// True when `new_or_conservative` returned conservative bounds (A == 0,
/// lower_b == -inf, upper_b == +inf for all elements).
fn is_conservative(b: &LinearBounds) -> bool {
    b.lower_a().iter().all(|&v| v == 0.0)
        && b.upper_a().iter().all(|&v| v == 0.0)
        && b.lower_b().iter().all(|&v| v == f32::NEG_INFINITY)
        && b.upper_b().iter().all(|&v| v == f32::INFINITY)
}

/// Finite contiguous inputs pass the fast-path firewall unchanged.
#[ntest::timeout(10000)]
#[test]
fn fastpath_finite_contiguous_passes_through() {
    let la = Array2::from_shape_vec((2, 2), vec![1.0, -0.5, 0.3, 2.0]).unwrap();
    let ua = Array2::from_shape_vec((2, 2), vec![1.2, -0.3, 0.5, 2.1]).unwrap();
    let lb = Array1::from_vec(vec![0.1, -0.2]);
    let ub = Array1::from_vec(vec![0.3, 0.0]);

    let out = LinearBounds::new_or_conservative(la.clone(), lb.clone(), ua.clone(), ub.clone())
        .expect("finite inputs must construct");
    assert!(
        !is_conservative(&out),
        "finite inputs must not trip firewall"
    );
    assert_eq!(out.lower_a(), &la, "lower_a must pass through unchanged");
    assert_eq!(out.upper_a(), &ua, "upper_a must pass through unchanged");
    assert_eq!(out.lower_b(), &lb, "lower_b must pass through unchanged");
    assert_eq!(out.upper_b(), &ub, "upper_b must pass through unchanged");
    // ±Inf in the BIAS is allowed (conservative-bound encoding) and must NOT
    // trip the firewall — verify new() accepts it directly.
    let inf_bias = LinearBounds::new(
        la,
        Array1::from_vec(vec![f32::NEG_INFINITY, 0.0]),
        ua,
        Array1::from_vec(vec![f32::INFINITY, 0.0]),
    );
    assert!(
        inf_bias.is_ok(),
        "±Inf bias must remain allowed by fast path"
    );
}

/// NaN / Inf coefficients on the contiguous fast path trip the firewall.
#[ntest::timeout(10000)]
#[test]
fn fastpath_nonfinite_contiguous_trips_firewall() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let la = Array2::from_shape_vec((2, 2), vec![1.0, bad, 0.3, 2.0]).unwrap();
        let ua = Array2::from_shape_vec((2, 2), vec![1.2, -0.3, 0.5, 2.1]).unwrap();
        let lb = Array1::zeros(2);
        let ub = Array1::zeros(2);
        // new() must hard-error...
        assert!(
            LinearBounds::new(la.clone(), lb.clone(), ua.clone(), ub.clone()).is_err(),
            "new() must reject non-finite coeff {bad}"
        );
        // ...and new_or_conservative must fall back to conservative bounds.
        let out = LinearBounds::new_or_conservative(la, lb, ua, ub)
            .expect("firewall must produce conservative bounds, not Err");
        assert!(
            is_conservative(&out),
            "non-finite coeff {bad} must trip firewall -> conservative"
        );
    }
}

/// The strided (non-contiguous) fallback path must catch the same poison as
/// the contiguous fast path. We build a non-standard-layout matrix via
/// transpose so `as_slice()` returns `None` and the `.iter()` branch runs.
#[ntest::timeout(10000)]
#[test]
fn fallback_strided_layout_trips_firewall() {
    // Build [3,2] then transpose to a [2,3] non-contiguous view, materialized.
    let base = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, f32::NAN, 4.0, 5.0, 6.0]).unwrap();
    let strided = base.reversed_axes(); // [2,3], column-major => as_slice() == None
    assert!(
        strided.as_slice().is_none(),
        "test precondition: transposed array must be non-contiguous"
    );
    let ua = Array2::zeros((2, 3));
    let lb = Array1::zeros(2);
    let ub = Array1::zeros(2);
    assert!(
        LinearBounds::new(strided.to_owned(), lb.clone(), ua, ub.clone()).is_err(),
        "strided fallback must reject NaN coefficient"
    );
    // Clean strided array must pass.
    let clean_base = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let clean_strided = clean_base.reversed_axes();
    assert!(clean_strided.as_slice().is_none());
    let out =
        LinearBounds::new_or_conservative(clean_strided.to_owned(), lb, Array2::zeros((2, 3)), ub)
            .expect("clean strided inputs must construct");
    assert!(
        !is_conservative(&out),
        "clean strided inputs must not trip firewall"
    );
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(256) })]

    /// Property: fast-path firewall decision matches the reference oracle for
    /// random finite values with optional NaN/Inf injection, on contiguous
    /// arrays. `new_or_conservative` is conservative IFF the oracle trips.
    #[ntest::timeout(10000)]
    #[test]
    fn prop_contiguous_matches_oracle(
        rows in 1_usize..=6,
        cols in 1_usize..=6,
        vals in proptest::collection::vec(-10.0_f32..10.0, 1..=200),
        inject in proptest::option::of((0_usize..4, 0_usize..36)),
        // 0 = NaN, 1 = +Inf, 2 = -Inf
        kind in 0_usize..3,
    ) {
        let n = rows * cols;
        let mut la: Vec<f32> = (0..n).map(|i| vals[i % vals.len()]).collect();
        let mut ua: Vec<f32> = (0..n).map(|i| vals[(i + 1) % vals.len()]).collect();
        let mut lb: Vec<f32> = (0..rows).map(|i| vals[(i + 2) % vals.len()]).collect();
        let mut ub: Vec<f32> = (0..rows).map(|i| vals[(i + 3) % vals.len()]).collect();

        let bad = match kind { 0 => f32::NAN, 1 => f32::INFINITY, _ => f32::NEG_INFINITY };
        if let Some((target, pos)) = inject {
            match target {
                0 => la[pos % n] = bad,
                1 => ua[pos % n] = bad,
                2 => lb[pos % rows] = bad,
                _ => ub[pos % rows] = bad,
            }
        }

        let la_a = Array2::from_shape_vec((rows, cols), la).unwrap();
        let ua_a = Array2::from_shape_vec((rows, cols), ua).unwrap();
        let lb_a = Array1::from_vec(lb);
        let ub_a = Array1::from_vec(ub);

        let expected_trip = oracle_trips(&la_a, &lb_a, &ua_a, &ub_a);

        // new(): Err IFF oracle trips.
        let new_res = LinearBounds::new(la_a.clone(), lb_a.clone(), ua_a.clone(), ub_a.clone());
        prop_assert_eq!(new_res.is_err(), expected_trip,
            "new() error state must match oracle ({})", expected_trip);

        // new_or_conservative(): conservative IFF oracle trips; otherwise identity passthrough.
        let out = LinearBounds::new_or_conservative(la_a.clone(), lb_a.clone(), ua_a.clone(), ub_a.clone()).unwrap();
        prop_assert_eq!(is_conservative(&out), expected_trip,
            "firewall fallback must match oracle ({})", expected_trip);
        if !expected_trip {
            prop_assert_eq!(out.lower_a(), &la_a);
            prop_assert_eq!(out.upper_a(), &ua_a);
            prop_assert_eq!(out.lower_b(), &lb_a);
            prop_assert_eq!(out.upper_b(), &ub_a);
        }
    }
}
