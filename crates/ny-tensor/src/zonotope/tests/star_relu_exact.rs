// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Exactness tests for the star ReLU transformer (`Star::relu_split` / `relu_exact`).
//!
//! The property under test is not merely soundness (the union CONTAINS the ReLU image)
//! but EXACTNESS (the union EQUALS it). That distinction is the whole reason the star
//! path is a complete method: an over-approximation can only ever say "unknown", while an
//! exact transformer can refute.

use ndarray::{array, ArrayD, IxDyn};

use crate::zonotope::{Star, StarReluSplit, ZonotopeTensor};

/// Build a star over `m` symbols from an explicit `(1 + m, n)` coefficient table.
fn star_from_rows(rows: Vec<Vec<f32>>) -> Star {
    let m1 = rows.len();
    let n = rows[0].len();
    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    let coeffs = ArrayD::from_shape_vec(IxDyn(&[m1, n]), flat).expect("coeff shape");
    Star::from_zonotope(ZonotopeTensor::new(coeffs).expect("zonotope"))
}

/// Value of flat coordinate `idx` at a given alpha.
fn value_at(star: &Star, idx: usize, alpha: &[f32]) -> f32 {
    let (c, g) = star.coordinate_form(idx).expect("coordinate form");
    c + g.iter().zip(alpha).map(|(gi, ai)| gi * ai).sum::<f32>()
}

/// Does `alpha` satisfy the star's predicate `A·α <= b`?
fn satisfies(star: &Star, alpha: &[f32]) -> bool {
    let (a, b) = star.constraints();
    (0..a.nrows()).all(|r| {
        let lhs: f32 = (0..a.ncols()).map(|c| a[[r, c]] * alpha[c]).sum();
        lhs <= b[r] + 1e-5
    })
}

#[test]
fn stable_active_neuron_passes_through_without_branching() {
    // x = 5 + 1·e  over e in [-1,1]  =>  range [4,6], strictly positive.
    let star = star_from_rows(vec![vec![5.0], vec![1.0]]);
    match star.relu_split(0).expect("split") {
        StarReluSplit::Active(s) => {
            assert_eq!(s.num_constraints(), 0, "no predicate row should be added");
            assert_eq!(value_at(&s, 0, &[0.5]), 5.5, "identity on an active neuron");
        }
        other => panic!("expected Active, got {other:?}"),
    }
}

#[test]
fn stable_inactive_neuron_is_zeroed_without_branching() {
    // x = -5 + 1·e  =>  range [-6,-4], strictly negative.
    let star = star_from_rows(vec![vec![-5.0], vec![1.0]]);
    match star.relu_split(0).expect("split") {
        StarReluSplit::Inactive(s) => {
            assert_eq!(s.num_constraints(), 0);
            for a in [-1.0, 0.0, 1.0] {
                assert_eq!(
                    value_at(&s, 0, &[a]),
                    0.0,
                    "inactive neuron must be exactly 0"
                );
            }
        }
        other => panic!("expected Inactive, got {other:?}"),
    }
}

#[test]
fn unstable_split_is_exact_on_a_dense_alpha_grid() {
    // Two coordinates over two symbols; coordinate 0 straddles zero.
    //   x0 = 0.5 + 1.0·e0 - 0.5·e1     (range [-1.0, 2.0] -> unstable)
    //   x1 = 2.0 + 0.25·e0 + 0.25·e1   (carried along untouched)
    let star = star_from_rows(vec![vec![0.5, 2.0], vec![1.0, 0.25], vec![-0.5, 0.25]]);
    let StarReluSplit::Split { inactive, active } = star.relu_split(0).expect("split") else {
        panic!("coordinate 0 must be unstable");
    };

    let steps = 41;
    let mut covered = 0usize;
    for i in 0..steps {
        for j in 0..steps {
            let a0 = -1.0 + 2.0 * (i as f32) / ((steps - 1) as f32);
            let a1 = -1.0 + 2.0 * (j as f32) / ((steps - 1) as f32);
            let alpha = [a0, a1];

            let pre = value_at(&star, 0, &alpha);
            let want = pre.max(0.0);

            // EXACTNESS: alpha lands in exactly one branch (up to the boundary), and that
            // branch's value at the same alpha equals ReLU of the original.
            let in_inactive = satisfies(&inactive, &alpha);
            let in_active = satisfies(&active, &alpha);
            assert!(
                in_inactive || in_active,
                "alpha ({a0},{a1}) escaped BOTH branches — the union would miss a reachable point"
            );
            if in_inactive {
                assert!(
                    (value_at(&inactive, 0, &alpha) - want).abs() < 1e-4,
                    "inactive branch value mismatch at ({a0},{a1})"
                );
                // Untouched coordinate must be preserved exactly.
                assert!((value_at(&inactive, 1, &alpha) - value_at(&star, 1, &alpha)).abs() < 1e-5);
            }
            if in_active {
                assert!(
                    (value_at(&active, 0, &alpha) - want).abs() < 1e-4,
                    "active branch value mismatch at ({a0},{a1})"
                );
            }
            covered += 1;
        }
    }
    assert_eq!(covered, steps * steps);
}

#[test]
fn split_branches_carry_the_expected_predicate_rows() {
    // x = 0.5 + 1.0·e  => inactive needs  1.0·e <= -0.5 ; active needs -1.0·e <= 0.5
    let star = star_from_rows(vec![vec![0.5], vec![1.0]]);
    let StarReluSplit::Split { inactive, active } = star.relu_split(0).expect("split") else {
        panic!("must be unstable");
    };
    let (ai, bi) = inactive.constraints();
    assert_eq!(ai.nrows(), 1);
    assert!((ai[[0, 0]] - 1.0).abs() < 1e-6 && (bi[0] + 0.5).abs() < 1e-6);

    let (aa, ba) = active.constraints();
    assert_eq!(aa.nrows(), 1);
    assert!((aa[[0, 0]] + 1.0).abs() < 1e-6 && (ba[0] - 0.5).abs() < 1e-6);
}

#[test]
fn relu_exact_enumerates_the_full_population() {
    // Three independent straddling coordinates => 2^3 = 8 stars.
    let star = star_from_rows(vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ]);
    let stars = star.relu_exact(64).expect("enumeration");
    assert_eq!(
        stars.len(),
        8,
        "three unstable neurons must yield 2^3 stars"
    );
}

#[test]
fn relu_exact_fails_closed_rather_than_truncating() {
    // The same 2^3 population, but capped below it. Dropping a branch would drop part of
    // the reachable set, so this must ERROR rather than return a partial answer.
    let star = star_from_rows(vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ]);
    let err = star.relu_exact(4).expect_err("must refuse to truncate");
    let msg = format!("{err}");
    assert!(
        msg.contains("refusing to truncate"),
        "error must say it refused rather than silently truncating: {msg}"
    );
}

#[test]
fn stable_coordinates_do_not_multiply_the_population() {
    // One unstable + two stable coordinates => 2 stars, not 8.
    let star = star_from_rows(vec![vec![0.0, 9.0, -9.0], vec![1.0, 0.5, 0.5]]);
    let stars = star.relu_exact(64).expect("enumeration");
    assert_eq!(stars.len(), 2, "stable neurons must not branch");
}

#[test]
fn with_constraint_rejects_shape_and_non_finite_rows() {
    let star = star_from_rows(vec![vec![0.0], vec![1.0]]);
    assert!(
        star.with_constraint(&array![1.0, 2.0], 0.0).is_err(),
        "wrong width"
    );
    assert!(
        star.with_constraint(&array![f32::NAN], 0.0).is_err(),
        "NaN row"
    );
    assert!(
        star.with_constraint(&array![1.0], f32::INFINITY).is_err(),
        "inf rhs"
    );
}

// ---------------------------------------------------------------------------
// Input-box bisection: the branching axis that scales with INPUT dimension
// rather than neuron count.
// ---------------------------------------------------------------------------

#[test]
fn input_bisection_halves_the_box_and_covers_it_exactly() {
    // x = 0 + 1·e0  over e0 in [-1,1]  =>  range [-1, 1].
    let star = star_from_rows(vec![vec![0.0], vec![1.0]]);
    let (lo_half, hi_half) = star.split_input_symbol(0).expect("bisect");

    // Each half must span exactly one side of the original range.
    let lb = lo_half.interval_bounds().expect("bounds");
    assert!((lb.lower()[[0]] - -1.0).abs() < 1e-6, "{:?}", lb.lower());
    assert!((lb.upper()[[0]] - 0.0).abs() < 1e-6, "{:?}", lb.upper());

    let hb = hi_half.interval_bounds().expect("bounds");
    assert!((hb.lower()[[0]] - 0.0).abs() < 1e-6);
    assert!((hb.upper()[[0]] - 1.0).abs() < 1e-6);
}

#[test]
fn input_bisection_tightens_the_interval_bound_unlike_a_predicate_row() {
    // The whole point: bisection shrinks the REPRESENTATION, so the cheap interval bound
    // improves. A predicate row would leave it untouched.
    let star = star_from_rows(vec![vec![2.0], vec![1.0]]);
    let before = star.interval_bounds().expect("b");
    let width_before = before.upper()[[0]] - before.lower()[[0]];

    let (half, _) = star.split_input_symbol(0).expect("bisect");
    let after = half.interval_bounds().expect("b");
    let width_after = after.upper()[[0]] - after.lower()[[0]];

    assert!(
        (width_after - width_before / 2.0).abs() < 1e-6,
        "bisection must halve the interval width: {width_before} -> {width_after}"
    );
}

#[test]
fn input_bisection_is_exact_on_a_dense_grid() {
    // Every point of the original must be reachable in exactly one half, at the SAME value.
    let star = star_from_rows(vec![vec![0.5, -1.0], vec![1.0, 0.25], vec![-0.5, 2.0]]);
    let (lo_half, hi_half) = star.split_input_symbol(0).expect("bisect");

    let steps = 41;
    for i in 0..steps {
        for j in 0..steps {
            let a0 = -1.0 + 2.0 * (i as f32) / ((steps - 1) as f32);
            let a1 = -1.0 + 2.0 * (j as f32) / ((steps - 1) as f32);
            for coord in 0..2 {
                let want = value_at(&star, coord, &[a0, a1]);
                // a0 in [-1,0] lives in the low half at a0' = 2·a0 + 1; the high half maps
                // a0 in [0,1] to a0' = 2·a0 - 1.
                let got = if a0 <= 0.0 {
                    value_at(&lo_half, coord, &[2.0 * a0 + 1.0, a1])
                } else {
                    value_at(&hi_half, coord, &[2.0 * a0 - 1.0, a1])
                };
                assert!(
                    (got - want).abs() < 1e-5,
                    "coord {coord} at ({a0},{a1}): half gave {got}, original {want}"
                );
            }
        }
    }
}

#[test]
fn input_bisection_rescales_existing_predicate_rows() {
    // A constraint must stay exactly as binding on a half as it was on the whole.
    // Star: x = e0, predicate e0 <= 0.5.
    let base = star_from_rows(vec![vec![0.0], vec![1.0]]);
    let star = base.with_constraint(&array![1.0], 0.5).expect("constrain");
    let (lo_half, hi_half) = star.split_input_symbol(0).expect("bisect");

    // Low half: e0 = (e' - 1)/2 <= 0.5  =>  0.5·e' <= 1.0
    let (a_lo, b_lo) = lo_half.constraints();
    assert!((a_lo[[0, 0]] - 0.5).abs() < 1e-6, "{a_lo:?}");
    assert!((b_lo[0] - 1.0).abs() < 1e-6, "{b_lo:?}");

    // High half: e0 = (e' + 1)/2 <= 0.5  =>  0.5·e' <= 0.0
    let (a_hi, b_hi) = hi_half.constraints();
    assert!((a_hi[[0, 0]] - 0.5).abs() < 1e-6);
    assert!((b_hi[0] - 0.0).abs() < 1e-6);
}

#[test]
fn widest_input_symbol_picks_the_largest_generator() {
    let star = star_from_rows(vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![3.0, 0.0]]);
    assert_eq!(star.widest_input_symbol(), Some(1), "symbol 1 dominates");

    let degenerate = star_from_rows(vec![vec![1.0], vec![0.0]]);
    assert_eq!(degenerate.widest_input_symbol(), None, "nothing to bisect");
}

// ---------------------------------------------------------------------------
// Zonotope ReLU relaxation: sound, non-branching, and TIGHTER than intervals.
// ---------------------------------------------------------------------------

#[test]
fn relu_overapprox_contains_the_relu_image_on_a_dense_grid() {
    // Soundness is the whole claim: every point's ReLU value must lie inside the relaxed
    // star's range at the SAME alpha (with the fresh symbol free over [-1,1]).
    let star = star_from_rows(vec![vec![0.25], vec![1.0]]);
    let relaxed = star.relu_overapprox(0).expect("relax");
    assert_eq!(
        relaxed.alpha_dim(),
        2,
        "one fresh symbol for the relaxation"
    );

    let steps = 81;
    for i in 0..steps {
        let a = -1.0 + 2.0 * (i as f32) / ((steps - 1) as f32);
        let want = value_at(&star, 0, &[a]).max(0.0);
        // Range over the fresh symbol at this alpha.
        let lo = value_at(&relaxed, 0, &[a, -1.0]);
        let hi = value_at(&relaxed, 0, &[a, 1.0]);
        let (lo, hi) = (lo.min(hi), lo.max(hi));
        assert!(
            want >= lo - 1e-5 && want <= hi + 1e-5,
            "ReLU value {want} at alpha {a} escaped the relaxation [{lo}, {hi}]"
        );
    }
}

#[test]
fn relu_overapprox_is_tighter_than_the_interval_image() {
    // x = 0.25 + 1.0*e over [-1,1] => pre-activation [-0.75, 1.25], ReLU image [0, 1.25].
    // Interval-of-ReLU gives exactly [0, 1.25]; the zonotope relaxation must not be WIDER,
    // and it retains correlation the interval throws away.
    let star = star_from_rows(vec![vec![0.25], vec![1.0]]);
    let relaxed = star.relu_overapprox(0).expect("relax");
    let b = relaxed.interval_bounds().expect("bounds");
    let (lo, hi) = (b.lower()[[0]], b.upper()[[0]]);
    assert!(lo <= 0.0 + 1e-5, "lower {lo} must cover 0");
    assert!(hi >= 1.25 - 1e-5, "upper {hi} must cover the true max 1.25");
    assert!(
        hi <= 1.6,
        "relaxation should stay near the true range, got {hi}"
    );
}

#[test]
fn relu_overapprox_handles_stable_coordinates_without_a_new_symbol() {
    let active = star_from_rows(vec![vec![5.0], vec![1.0]]);
    let r = active.relu_overapprox(0).expect("relax");
    assert_eq!(r.alpha_dim(), 1, "stable-active adds no symbol");
    assert_eq!(value_at(&r, 0, &[0.5]), 5.5);

    let inactive = star_from_rows(vec![vec![-5.0], vec![1.0]]);
    let r = inactive.relu_overapprox(0).expect("relax");
    assert_eq!(r.alpha_dim(), 1, "stable-inactive adds no symbol");
    assert_eq!(value_at(&r, 0, &[0.5]), 0.0);
}

#[test]
fn relu_overapprox_preserves_other_coordinates_and_the_predicate() {
    let base = star_from_rows(vec![vec![0.0, 3.0], vec![1.0, 0.5]]);
    let star = base.with_constraint(&array![1.0], 0.25).expect("constrain");
    let relaxed = star.relu_overapprox(0).expect("relax");

    // Coordinate 1 is untouched at the same alpha.
    let want = value_at(&star, 1, &[0.5]);
    let got = value_at(&relaxed, 1, &[0.5, 0.0]);
    assert!(
        (got - want).abs() < 1e-6,
        "coordinate 1 changed: {got} vs {want}"
    );

    // The predicate carries over, widened by a zero column for the fresh symbol.
    let (a, b) = relaxed.constraints();
    assert_eq!(a.nrows(), 1);
    assert_eq!(a.ncols(), 2, "widened to the new alpha dim");
    assert!(
        (a[[0, 0]] - 1.0).abs() < 1e-6 && a[[0, 1]].abs() < 1e-9,
        "fresh symbol unconstrained"
    );
    assert!((b[0] - 0.25).abs() < 1e-6);
}
