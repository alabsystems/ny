// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use num_rational::BigRational;

fn problem(constant: f64, upper: f64) -> HullMarginProblem {
    HullMarginProblem::new(
        vec![1.0],
        vec![0.0],
        1,
        &[-1.0],
        &[constant],
        1,
        &[(0, 1.0)],
        0.0,
        vec![-1.0],
        vec![upper],
    )
    .unwrap()
}

#[test]
fn hull_goal_strict_positive_stops_on_original_triangle_before_lp() {
    let original = problem(2.0, 1.0);
    let deadline = Instant::now() + Duration::from_secs(10);
    let report = hull_root_bound_for_goal(&original, deadline, HullBoundGoal::StrictPositive);
    assert!(report.verified());
    assert_eq!(report.neg_unstable, 1);
    assert_eq!(report.route, Some(HullBoundRoute::Triangle));
    assert_eq!(report.lp_status, HullLpStatus::SkippedByGoal);
    assert_eq!(report.lp_status.label(), "skipped_by_goal");
    assert!(report.lp_value.is_none());
    assert_eq!(report.subgradient_iterations, 0);
    let lower = BigRational::from_float(report.certified.unwrap()).unwrap();
    for x in [-1.0_f64, -0.5, 0.0, 0.5, 1.0] {
        let margin = BigRational::from_float(2.0 - x.max(0.0)).unwrap();
        assert!(lower <= margin);
    }
    // The same positive triangle is insufficient to stop the historical
    // full-bound contract, which still obtains the original LP bound.
    let full = hull_root_bound(&original, deadline);
    assert_eq!(full.lp_status, HullLpStatus::Solved);
    assert!(full.lp_value.is_some());
    assert!(full.certified.unwrap() >= report.certified.unwrap());
}

#[test]
fn hull_goal_strict_tie_and_witnessed_sat_never_skip_or_close() {
    for constant in [1.0, 0.5] {
        let original = problem(constant, 1.0);
        let report = hull_root_bound_for_goal(
            &original,
            Instant::now() + Duration::from_secs(10),
            HullBoundGoal::StrictPositive,
        );
        assert!(!report.verified());
        assert_ne!(report.lp_status, HullLpStatus::SkippedByGoal);
        assert!(report.certified.is_none_or(|value| value <= constant - 1.0));
        // Original x=1 witnesses the strict tie or negative margin.
        assert!(constant - 1.0 <= 0.0);
    }
}

#[test]
fn hull_goal_expired_positive_problem_refuses_before_original_scans() {
    let original = problem(2.0, 1.0);
    let report = hull_root_bound_for_goal(
        &original,
        Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("overflow when subtracting duration from instant"),
        HullBoundGoal::StrictPositive,
    );
    assert!(!report.verified());
    assert_eq!(report.lp_status, HullLpStatus::Skipped);
    assert_eq!(report.unstable, 0);
    assert!(report.certified.is_none() && report.route.is_none() && report.lp_value.is_none());
    assert_eq!(report.subgradient_iterations, 0);
}

#[test]
fn hull_goal_full_bound_wrapper_keeps_original_no_lp_result_identical() {
    let original = problem(2.0, -0.25);
    let deadline = Instant::now() + Duration::from_secs(10);
    let old = hull_root_bound(&original, deadline);
    let explicit = hull_root_bound_for_goal(&original, deadline, HullBoundGoal::FullBound);
    assert_eq!(old, explicit);
    assert_eq!(old.lp_status, HullLpStatus::Empty);
    assert!(old.verified());
}
