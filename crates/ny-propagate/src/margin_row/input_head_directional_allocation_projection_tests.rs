// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Nominal optimizer controls; no proposal is a numerical certificate.
use super::*;

fn col(values: Vec<f64>, mass: f64) -> Column {
    let norm = values.iter().map(|v| v * v).sum();
    Column {
        head: 0,
        mass,
        values,
        norm,
    }
}

fn end() -> Instant {
    Instant::now() + Duration::from_secs(10)
}

#[test]
fn allocation_projection_active_set_satisfies_exact_dyadic_kkt() {
    let z = [3.0, -2.0, 1.0, 0.0];
    let a = project(&z, 2.0).unwrap();
    assert_eq!(a, vec![2.0, 0.0, 0.0, 0.0]);
    assert_eq!(a.iter().sum::<f64>(), 2.0);
    for (&a, &z) in a.iter().zip(&z) {
        if a > 0.0 {
            assert_eq!(z - a, 1.0);
        } else {
            assert!(z <= 1.0);
        }
    }
}

#[test]
fn allocation_projection_interior_and_ties_preserve_mass_without_index_bias() {
    assert_eq!(project(&[1.0, 0.5, 0.0], 3.0).unwrap(), vec![1.5, 1.0, 0.5]);
    assert_eq!(project(&[0.25; 12], 3.0).unwrap(), vec![0.25; 12]);
    assert_eq!(descending(&[2.0, 1.0, 2.0, 1.0]).unwrap(), vec![0, 2, 1, 3]);
}

#[test]
fn allocation_projection_bounded_mergesort_orders_every_supported_cardinality() {
    for k in 1..=12 {
        let z = (0..k).map(|g| ((7 * g + 3) % 5) as f64).collect::<Vec<_>>();
        let order = descending(&z).unwrap();
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..k).collect::<Vec<_>>());
        for pair in order.windows(2) {
            assert!(z[pair[0]] > z[pair[1]] || (z[pair[0]] == z[pair[1]] && pair[0] < pair[1]));
        }
    }
    assert!(descending(&[]).is_none());
    assert!(descending(&[0.0; 13]).is_none());
}

#[test]
fn allocation_projection_rejects_nonfinite_mass_geometry_and_overflow() {
    for mass in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert!(project(&[0.0; 3], mass).is_none());
    }
    for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        assert!(project(&[0.0, value, 1.0], 1.0).is_none());
    }
    assert!(project(&[1e100; 3], 1.0).is_none());
    assert!(cost(3, usize::MAX).is_none());
    assert!(cost(2, 30).is_none() && cost(13, 30).is_none() && cost(3, 0).is_none());
}

#[test]
fn allocation_projection_full_coordinate_reduces_exact_quadratic_and_keeps_inputs() {
    let column = col(vec![1.0, 0.0], 3.0);
    let old = vec![1.0, 1.0, 1.0];
    let residual = vec![vec![2.0, 0.0], vec![-1.0, 0.0], vec![-1.0, 0.0]];
    let update = coordinate(&column, &old, &residual).unwrap();
    assert_eq!(update.masses, vec![3.0, 0.0, 0.0]);
    assert_eq!(update.residual, vec![vec![0.0; 2]; 3]);
    assert!(update.changed);
    assert_eq!(old, vec![1.0; 3]);
    assert_eq!(residual[2], vec![-1.0, 0.0]);
}

#[test]
fn allocation_projection_bad_last_coordinate_is_transactional() {
    let column = col(vec![1.0, 1.0], 3.0);
    let old = vec![1.0; 3];
    let residual = vec![vec![0.0; 2], vec![0.0; 2], vec![0.0, f64::NAN]];
    let before = residual
        .iter()
        .flatten()
        .map(|x| x.to_bits())
        .collect::<Vec<_>>();
    assert!(coordinate(&column, &old, &residual).is_none());
    assert_eq!(
        before,
        residual
            .iter()
            .flatten()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(old, vec![1.0; 3]);
    assert!(coordinate(&column, &old, &residual[..2]).is_none());
    assert!(coordinate(&column, &old, &vec![vec![0.0]; 3]).is_none());
}

#[test]
fn allocation_projection_zero_norm_and_wrong_group_dimensions_decline() {
    let old = vec![1.0; 3];
    assert!(coordinate(&col(vec![0.0; 2], 3.0), &old, &vec![vec![0.0; 2]; 3]).is_none());
    assert!(coordinate(&col(vec![1.0], 3.0), &[1.0; 2], &vec![vec![0.0]; 2]).is_none());
    assert!(coordinate(&col(vec![1.0], 3.0), &[1.0, -1.0, 1.0], &vec![vec![0.0]; 3]).is_none());
}

#[test]
fn allocation_projection_credit_ledger_reserves_the_original_tail() {
    assert_eq!(cost(6, 30), Some(478));
    let initial = 6 * 30 + 30 * 6 + 48 * 31 + 30 * 6 * 32;
    let tail = 30 * (7 * 6 + 4) + 48 * 4 + 32;
    assert_eq!(initial, 7608);
    assert!(initial + 8 * 30 * 478 + tail <= 125_000);
    assert_eq!(initial + 8 * 30 * 478 + 30 * 40 + 6 * 62, 123_900);
}

#[test]
fn allocation_projection_optional_room_stop_normalizes_last_complete_allocation() {
    let negatives = vec![
        col(vec![3.0], 1.0),
        col(vec![0.0], 1.0),
        col(vec![0.0], 1.0),
    ];
    let positives = vec![col(vec![1.0], 3.0)];
    let groups = vec![vec![0], vec![1], vec![2]];
    let mut budget = Budget {
        deadline: end(),
        cells: 0,
        masks: 0,
    };
    let mut work = Search {
        budget: &mut budget,
        start: 0,
        maximum: 100,
    };
    let masses = fit(&negatives, &groups, &positives, &mut work).unwrap();
    assert!(masses[0].iter().all(|&a| a > 0.99 && a <= 1.0));
    assert_eq!(work.budget.cells, 43);
    let total = masses[0]
        .iter()
        .try_fold(I::p(0.0), |s, &a| s.add(I::p(a)))
        .unwrap();
    assert!(total.hi <= 3.0);
}

#[test]
fn allocation_projection_two_group_pair_path_retains_original_credit_and_solution() {
    let negatives = vec![col(vec![3.0, 0.0], 1.0), col(vec![-1.0, 0.0], 1.0)];
    let positives = vec![col(vec![1.0, 0.0], 2.0)];
    let mut budget = Budget {
        deadline: end(),
        cells: 0,
        masks: 0,
    };
    let mut work = Search {
        budget: &mut budget,
        start: 0,
        maximum: 125_000,
    };
    let masses = fit(&negatives, &[vec![0], vec![1]], &positives, &mut work).unwrap();
    assert!(masses[0][0] > 1.99 && masses[0][0] <= 2.0 && masses[0][1] == 0.0);
    assert_eq!(work.budget.cells, 64);
}

#[test]
fn allocation_projection_expired_optional_coordinate_keeps_complete_arrays() {
    let positives = vec![col(vec![1.0], 3.0)];
    let mut masses = vec![vec![1.0; 3]];
    let mut residual = vec![vec![0.0]; 3];
    let mut budget = Budget {
        deadline: Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
        cells: 0,
        masks: 0,
    };
    let mut work = Search {
        budget: &mut budget,
        start: 0,
        maximum: 125_000,
    };
    assert!(improve(&positives, &mut masses, &mut residual, &mut work, 100).is_some());
    assert_eq!(masses, vec![vec![1.0; 3]]);
    assert_eq!(residual, vec![vec![0.0]; 3]);
    assert_eq!(work.budget.cells, 0);
}
