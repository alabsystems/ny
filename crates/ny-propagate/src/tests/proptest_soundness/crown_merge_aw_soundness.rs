// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! STRICT, ZERO-TOLERANCE soundness tests for the graph CROWN merge accumulator's
//! certified coefficient error (#vnncomp-aw-soundness — DAG-merge err-drop bug).
//!
//! The bug: at every DAG merge point (residual blocks) the graph CROWN merge
//! accumulator folded each contribution's f32 coefficients into an f64 sidecar via
//! `LinearBounds64::from_f32`, which DROPPED the contribution's certified
//! per-coefficient error (`lower_a_err`/`upper_a_err`), and `accumulate_linear_bounds64`
//! never re-accumulated it. The merged bounds therefore concretized with NO
//! coefficient-error penalty even though the merged coefficient genuinely differs
//! from the true real coefficient by the SUM of the inputs' errors. On a deep
//! residual net this under-counts the certified error → the concretized bound can
//! be tighter than the true reachable value = FALSE PROOF.
//!
//! These tests assert the CERTIFICATE CONTRACT directly: after merging two
//! error-carrying contributions, the merged certified error must cover
//!
//!     |stored_f32_merged_coeff − true_real_merged_coeff|
//!
//! where `true_real_merged_coeff` is the SUM of each contribution's known true
//! coefficient (the value its own certified error certifies it against). The OLD
//! code attaches ZERO error here → the assertion fails (reproduces the bug); the
//! patched code carries `err1 + err2 + roundoff` → the assertion passes.

use crate::bounds::patches::CrownBounds;
use crate::network::CrownMergeAccumulator;
use crate::LinearBounds;
use ndarray::{Array1, Array2};
use proptest::prelude::*;

/// Build a single-output `LinearBounds` whose STORED f32 coefficients are `stored`
/// and which carries a certified per-coefficient error `err` (so the invariant
/// `|stored - true| <= err` holds for some unknown true real coefficient). Bias is
/// zero. Lower == upper coefficients (exact-linear style contribution).
fn contribution_with_err(stored: &[f32], err: &[f32]) -> LinearBounds {
    let n = stored.len();
    let a = Array2::from_shape_vec((1, n), stored.to_vec()).unwrap();
    let e = Array2::from_shape_vec((1, n), err.to_vec()).unwrap();
    LinearBounds::new_or_conservative_with_err(
        a.clone(),
        Array1::zeros(1),
        a,
        Array1::zeros(1),
        e.clone(),
        e,
    )
    .unwrap()
}

/// Merge two error-carrying contributions through the graph CROWN merge
/// accumulator and return the merged `LinearBounds` (downcast back to f32).
fn merge_two(c1: LinearBounds, c2: LinearBounds) -> LinearBounds {
    let mut acc = CrownMergeAccumulator::new();
    acc.insert("residual".to_string(), CrownBounds::Dense(c1));
    acc.merge_dense("residual", c2)
        .expect("merge_dense should succeed");
    acc.take("residual")
        .expect("take should succeed")
        .expect("residual entry should exist")
        .into_dense()
        .expect("merged entry should be dense")
}

/// REPRODUCE + VERIFY: the merge accumulator must carry the certified coefficient
/// error across a DAG merge. Construct two contributions whose stored coefficients
/// differ from their known true coefficients by a certified error; after the merge
/// the merged certified error MUST cover the merged stored-vs-true gap.
///
/// On the OLD (buggy) accumulator the merged bounds carry NO error → the certified
/// error is 0 < gap → this assertion FAILS (reproduces the false-proof risk). On
/// the patched accumulator the error is summed (err1 + err2 + roundoff) → PASSES.
#[test]
fn merge_accumulator_carries_coeff_err_across_dag_merge() {
    // Two contributions on 3 inputs. Pick true coefficients that, when added,
    // genuinely cancel/round, and stored f32 values that differ from truth by a
    // certified error (modeling an upstream f64-accumulated A·W coefficient).
    let true1 = [1.000_000_1_f64, -2.000_000_3, 0.5];
    let true2 = [-1.000_000_05_f64, 2.000_000_1, -0.5];
    let stored1 = [true1[0] as f32, true1[1] as f32, true1[2] as f32];
    let stored2 = [true2[0] as f32, true2[1] as f32, true2[2] as f32];
    let err1: Vec<f32> = (0..3)
        .map(|j| ((stored1[j] as f64 - true1[j]).abs() as f32) * 2.0 + 1e-7)
        .collect();
    let err2: Vec<f32> = (0..3)
        .map(|j| ((stored2[j] as f64 - true2[j]).abs() as f32) * 2.0 + 1e-7)
        .collect();

    let c1 = contribution_with_err(&stored1, &err1);
    let c2 = contribution_with_err(&stored2, &err2);
    let merged = merge_two(c1, c2);

    let merged_err = merged.lower_a_err().expect(
        "patched merge accumulator must carry a certified coefficient error \
             across a DAG merge (the OLD code drops it → None → certified 0)",
    );

    let true_merged: Vec<f64> = (0..3).map(|j| true1[j] + true2[j]).collect();
    for j in 0..3 {
        let stored_merged = merged.lower_a()[[0, j]] as f64;
        let gap = (stored_merged - true_merged[j]).abs();
        let cert = merged_err[[0, j]] as f64;
        assert!(
            cert >= gap,
            "UNSOUND merge certificate at col {j}: certified {cert:.3e} < \
             |stored_f32_merged − true_real_merged| {gap:.3e}"
        );
    }
}

/// End-to-end zero-tolerance enclosure: the merged bounds, concretized over a wide
/// input box, must enclose the true reachable value of the SUMMED true-coefficient
/// linear function for every corner. On the OLD accumulator the dropped error makes
/// the concretized interval too tight, crossing the true value on the worst corner.
#[test]
fn merge_accumulator_concretized_bound_encloses_true_value() {
    use ny_tensor::BoundedTensor;

    let true1 = [3.000_000_2_f64, -1.000_000_4, 2.000_000_05];
    let true2 = [-3.000_000_1_f64, 1.000_000_2, -2.000_000_02];
    let stored1 = [true1[0] as f32, true1[1] as f32, true1[2] as f32];
    let stored2 = [true2[0] as f32, true2[1] as f32, true2[2] as f32];
    let err1: Vec<f32> = (0..3)
        .map(|j| ((stored1[j] as f64 - true1[j]).abs() as f32) * 2.0 + 1e-7)
        .collect();
    let err2: Vec<f32> = (0..3)
        .map(|j| ((stored2[j] as f64 - true2[j]).abs() as f32) * 2.0 + 1e-7)
        .collect();

    let merged = merge_two(
        contribution_with_err(&stored1, &err1),
        contribution_with_err(&stored2, &err2),
    );

    // Wide input box [-100, 100]^3.
    let lo = Array1::from_elem(3, -100.0_f32).into_dyn();
    let hi = Array1::from_elem(3, 100.0_f32).into_dyn();
    let input = BoundedTensor::new(lo, hi).unwrap();
    let out = merged.concretize_sound(&input);

    // True reachable extrema of f(x) = Σ_j true_merged[j]·x_j over the box.
    let true_merged: Vec<f64> = (0..3).map(|j| true1[j] + true2[j]).collect();
    let (mut true_lo, mut true_hi) = (0.0f64, 0.0f64);
    for j in 0..3 {
        let coeff = true_merged[j];
        if coeff >= 0.0 {
            true_lo += coeff * -100.0;
            true_hi += coeff * 100.0;
        } else {
            true_lo += coeff * 100.0;
            true_hi += coeff * -100.0;
        }
    }

    let concrete_lo = out.lower()[0] as f64;
    let concrete_hi = out.upper()[0] as f64;
    assert!(
        concrete_lo <= true_lo,
        "UNSOUND: concretized lower {concrete_lo:.6} > true reachable min {true_lo:.6} \
         (the dropped DAG-merge coefficient error makes the bound too tight)"
    );
    assert!(
        concrete_hi >= true_hi,
        "UNSOUND: concretized upper {concrete_hi:.6} < true reachable max {true_hi:.6}"
    );
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 4000, ..ProptestConfig::with_cases(200) })]

    /// SOUNDNESS PROPTEST: random pairs of error-carrying contributions merged
    /// through the accumulator — the merged certified error must cover the merged
    /// stored-vs-true coefficient gap for EVERY coefficient. Zero tolerance.
    #[test]
    fn proptest_merge_accumulator_covers_coeff_err(
        n in 1usize..=6,
        seed in any::<u64>(),
    ) {
        // Deterministic pseudo-random fill.
        let mut s = seed | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as i64 % 2_000_001 - 1_000_000) as f64 / 7919.0
        };
        let true1: Vec<f64> = (0..n).map(|_| next()).collect();
        let true2: Vec<f64> = (0..n).map(|_| next()).collect();
        let stored1: Vec<f32> = true1.iter().map(|&t| t as f32).collect();
        let stored2: Vec<f32> = true2.iter().map(|&t| t as f32).collect();
        let err1: Vec<f32> = (0..n)
            .map(|j| ((stored1[j] as f64 - true1[j]).abs() as f32) * 2.0 + 1e-7)
            .collect();
        let err2: Vec<f32> = (0..n)
            .map(|j| ((stored2[j] as f64 - true2[j]).abs() as f32) * 2.0 + 1e-7)
            .collect();

        let merged = merge_two(
            contribution_with_err(&stored1, &err1),
            contribution_with_err(&stored2, &err2),
        );
        let lower_err = merged.lower_a_err().expect("merged err must be present");
        let upper_err = merged.upper_a_err().expect("merged err must be present");

        for j in 0..n {
            let stored_merged = merged.lower_a()[[0, j]] as f64;
            let true_merged = true1[j] + true2[j];
            let gap = (stored_merged - true_merged).abs();
            prop_assert!(
                lower_err[[0, j]] as f64 >= gap,
                "UNSOUND lower merge cert at {} seed {}: {:.3e} < gap {:.3e}",
                j, seed, lower_err[[0, j]], gap
            );
            let stored_merged_u = merged.upper_a()[[0, j]] as f64;
            let gap_u = (stored_merged_u - true_merged).abs();
            prop_assert!(
                upper_err[[0, j]] as f64 >= gap_u,
                "UNSOUND upper merge cert at {} seed {}: {:.3e} < gap {:.3e}",
                j, seed, upper_err[[0, j]], gap_u
            );
        }
    }
}
