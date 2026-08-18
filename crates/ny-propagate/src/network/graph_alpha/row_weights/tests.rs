// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #row-weights tests.
//!
//! The load-bearing properties, in order of how badly a violation would hurt:
//!  1. STRICT POSITIVITY — this fixed-shape route promises to preserve every
//!     row's sign branch, so a zero weight violates its admission contract.
//!     Tested against `exp` underflow explicitly.
//!  2. SIMPLEX — the weights are a distribution every round.
//!  3. NO COLLAPSE — MW must not degenerate to the point mass whose
//!     oscillation it exists to avoid.
//!  4. Positive-affine scale invariance, direction of mass flow, and typed
//!     refusals.

use super::*;

fn assert_simplex(w: &RowWeights) {
    let total: f64 = w.weights().iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-9,
        "weights must sum to 1, got {total}"
    );
    assert!(
        w.weights().iter().all(|p| *p > 0.0),
        "every weight must be strictly positive: {:?}",
        w.weights()
    );
}

#[test]
fn starts_uniform_on_the_simplex() {
    let w = RowWeights::new(99, 0.5).unwrap();
    assert_simplex(&w);
    assert_eq!(w.rows(), 99);
    for p in w.weights() {
        assert!((p - 1.0 / 99.0).abs() < 1e-12);
    }
}

#[test]
fn mass_flows_to_the_worst_row() {
    let mut w = RowWeights::new(3, 1.0).unwrap();
    // Row 1 is doing worst (most negative slack) and must gain weight.
    w.update(&[1.0, -2.0, 0.5]).unwrap();
    assert_simplex(&w);
    let p = w.weights();
    assert!(p[1] > p[0] && p[1] > p[2], "worst row must dominate: {p:?}");
    assert!(p[0] < 1.0 / 3.0, "a comfortable row must lose weight");
}

#[test]
fn equal_slacks_leave_the_distribution_uniform() {
    let mut w = RowWeights::new(5, 2.0).unwrap();
    w.update(&[0.25; 5]).unwrap();
    assert_simplex(&w);
    for p in w.weights() {
        assert!((p - 0.2).abs() < 1e-12, "{:?}", w.weights());
    }
}

#[test]
fn update_is_invariant_to_a_constant_shift_of_all_slacks() {
    // The shift cancels in the bounded-loss normalisation.
    let mut a = RowWeights::new(4, 0.7).unwrap();
    let mut b = RowWeights::new(4, 0.7).unwrap();
    a.update(&[-1.0, 0.0, 2.0, 0.5]).unwrap();
    b.update(&[999.0, 1000.0, 1002.0, 1000.5]).unwrap();
    for (x, y) in a.weights().iter().zip(b.weights()) {
        assert!(
            (x - y).abs() < 1e-9,
            "{:?} vs {:?}",
            a.weights(),
            b.weights()
        );
    }
}

#[test]
fn update_is_invariant_to_large_positive_rescaling() {
    let mut baseline = RowWeights::new(4, 0.7).unwrap();
    let mut rescaled = RowWeights::new(4, 0.7).unwrap();
    baseline.update(&[-1.0, 0.0, 2.0, 0.5]).unwrap();
    rescaled.update(&[-1.0e20, 0.0, 2.0e20, 0.5e20]).unwrap();
    for (x, y) in baseline.weights().iter().zip(rescaled.weights()) {
        assert!(
            (x - y).abs() < 1e-7,
            "raw margin scale changed the MW decision: {:?} vs {:?}",
            baseline.weights(),
            rescaled.weights()
        );
    }
}

#[test]
fn exp_underflow_is_typed_and_transactional() {
    // A manually supplied extreme eta can underflow exp(-eta). Projection
    // would change the Hedge algorithm, so refuse without mutating instead.
    let mut w = RowWeights::new(4, 1_000.0).unwrap();
    let before = w.weights().to_vec();
    assert!(w.update(&[0.0, 1.0, 2.0, 3.0]).is_err());
    assert_eq!(w.weights(), before);
    assert_eq!(w.rounds(), 0);
}

#[test]
fn repeated_extreme_updates_keep_the_simplex_invariant() {
    let mut w = RowWeights::new(6, 10.0).unwrap();
    for round in 0..50 {
        let slacks: Vec<f32> = (0..6)
            .map(|r| if r == round % 6 { -100.0 } else { 100.0 })
            .collect();
        w.update(&slacks).unwrap();
        assert_simplex(&w);
    }
    assert_eq!(w.rounds(), 50);
}

#[test]
fn does_not_collapse_to_a_point_mass_under_alternating_binding_rows() {
    // The failure mode MW exists to avoid: single-row (best-response) steering
    // fixes row A, row B becomes binding, it fixes B, and A returns. A point
    // mass would sit at 1.0 on whichever row is currently worst. MW must keep
    // meaningful weight on both.
    let mut w = RowWeights::new(2, 1.0).unwrap();
    let mut worst_max = 0.0f64;
    for round in 0..40 {
        let slacks = if round % 2 == 0 {
            [-1.0f32, 0.0]
        } else {
            [0.0f32, -1.0]
        };
        w.update(&slacks).unwrap();
        assert_simplex(&w);
        worst_max = worst_max.max(w.weights().iter().copied().fold(0.0, f64::max));
    }
    assert!(
        worst_max < 0.95,
        "MW collapsed to a near point mass (max weight {worst_max}), which is \
         the oscillation it is supposed to prevent"
    );
}

#[test]
fn horizon_eta_comes_from_the_regret_bound() {
    // eta = sqrt(8 ln R / T): derived, not tuned.
    let w = RowWeights::with_horizon(99, 8).unwrap();
    let expected = (8.0 * 99.0f64.ln() / 8.0).sqrt();
    assert!((w.eta() - expected).abs() < 1e-12);
    assert!(w.eta().is_finite() && w.eta() > 0.0);
    assert_simplex(&w);
}

#[test]
fn horizon_handles_the_degenerate_single_row_case() {
    // ln(1) = 0 would give eta = 0, which `new` rejects.
    let w = RowWeights::with_horizon(1, 20).unwrap();
    assert_eq!(w.rows(), 1);
    assert!(w.eta() > 0.0);
    assert_simplex(&w);
}

#[test]
fn horizon_refuses_an_empty_row_set() {
    assert!(RowWeights::with_horizon(0, 20).is_err());
    assert!(RowWeights::with_horizon(99, 0).is_err());
}

#[test]
fn production_horizon_stays_positive_and_is_enforced() {
    let mut w = RowWeights::with_horizon(256, 8).unwrap();
    let slacks: Vec<f32> = (0..256)
        .map(|row| if row == 0 { 1.0 } else { 0.0 })
        .collect();
    for _ in 0..8 {
        w.update(&slacks).unwrap();
        assert!(w.min_weight() > 0.0);
    }
    let before = w.weights().to_vec();
    assert!(w.update(&slacks).is_err());
    assert_eq!(w.rounds(), 8);
    assert_eq!(w.weights(), before);
    assert!(RowWeights::with_horizon(99, 20).is_ok());
}

#[test]
fn weighted_slack_is_the_scalar_handed_to_the_alpha_player() {
    let w = RowWeights::new(2, 1.0).unwrap();
    // Uniform start: the mean.
    let v = w.weighted_slack(&[-1.0, 3.0]).unwrap();
    assert!((v - 1.0).abs() < 1e-12);
}

#[test]
fn scaled_seed_scales_each_row_by_its_weight() {
    let w = RowWeights::new(2, 1.0).unwrap(); // uniform 0.5 / 0.5
    let objectives = vec![vec![1.0f32, 0.0, -1.0], vec![0.0, 2.0, 0.0]];
    let seed = w.scaled_seed(&objectives).unwrap();
    assert_eq!(seed.len(), 6);
    assert!((seed[0] - 0.5).abs() < 1e-6);
    assert!((seed[2] + 0.5).abs() < 1e-6);
    assert!((seed[4] - 1.0).abs() < 1e-6);
}

#[test]
fn refusals_are_typed_not_silent() {
    assert!(RowWeights::new(0, 1.0).is_err(), "empty row set");
    assert!(RowWeights::new(3, 0.0).is_err(), "non-positive eta");
    assert!(RowWeights::new(3, f64::NAN).is_err(), "non-finite eta");

    let mut w = RowWeights::new(3, 1.0).unwrap();
    assert!(w.update(&[1.0, 2.0]).is_err(), "wrong slack length");
    // A diverged fold must not be allowed to poison the distribution.
    assert!(w.update(&[1.0, f64::NAN as f32, 2.0]).is_err(), "NaN slack");
    assert!(
        w.update(&[1.0, f32::INFINITY, 2.0]).is_err(),
        "infinite slack"
    );
    // ...and none of those refusals may have mutated the weights.
    assert_simplex(&w);
    assert_eq!(w.rounds(), 0);

    let objectives = vec![vec![1.0f32]; 2];
    assert!(
        w.scaled_seed(&objectives).is_err(),
        "objective count must match row count"
    );
    assert!(
        RowWeights::new(1, 1.0)
            .unwrap()
            .scaled_seed(&[Vec::new()])
            .is_err(),
        "empty objective rows must not produce an empty resident seed"
    );
    assert!(
        RowWeights::new(1, 1.0)
            .unwrap()
            .scaled_seed(&[vec![f32::NAN]])
            .is_err(),
        "non-finite objective coefficients must fail closed"
    );
    assert!(
        RowWeights::new(2, 1.0)
            .unwrap()
            .scaled_seed(&[vec![f32::from_bits(1)], vec![1.0]])
            .is_err(),
        "a positive scale that rounds a nonzero coefficient to zero must fail closed"
    );
}
