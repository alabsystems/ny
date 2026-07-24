// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certificate-contract soundness tests for the β-CROWN split-constraint term.
//!
//! BUG (false-proof risk): `apply_constrained_relu_beta_contribution`
//! (constraints/backward/relu.rs) and `apply_beta_contribution`
//! (batched/backward_core.rs) mutate every coefficient in a neuron's column with
//! a single f32 op `lower_a := fl32(a - β)` / `upper_a := fl32(a + β)`. When the
//! bounds object already carries a certified coefficient error
//! (`#vnncomp-aw-soundness`, e.g. it came from a wide linear/conv backward whose
//! `A·W` was f64-accumulated), the f32 rounding of THAT mutation was dropped from
//! the err. The certificate then UNDER-counts `|stored_f32_coeff − true_coeff|`,
//! so `concretize` subtracts/adds too small an `err·|input|` penalty and can
//! produce a bound TIGHTER than the true reachable value = FALSE PROOF.
//!
//! These tests assert the CERTIFICATE CONTRACT directly (mirrors the conv repro
//! `conv2d_aw_cert_covers_f32_error_and_old_undercounts`, commit becc501):
//!
//!   certified_err[i, j]  >=  |stored_f32_coeff[i, j] − exact_f64_recompute[i, j]|
//!
//! for the β-split op on a WIDE coefficient where the mutation rounds. The fix is
//! `LinearBounds::apply_beta_split_to_column`, which folds the f32 rounding gap
//! into the err with an outward (`next_up_f32`) round.

use ndarray::{arr1, Array1, Array2};
use proptest::prelude::*;

use crate::LinearBounds;

/// The OLD (buggy) β-split: mutate the coefficient in-place but DO NOT touch the
/// certified error. This is exactly what both call sites did before the fix
/// (`lb.lower_a_mut()[[i,j]] -= β; lb.upper_a_mut()[[i,j]] += β;`).
fn apply_beta_split_buggy(lb: &mut LinearBounds, neuron_idx: usize, signed_beta: f32) {
    for i in 0..lb.num_outputs() {
        lb.lower_a_mut()[[i, neuron_idx]] -= signed_beta;
        lb.upper_a_mut()[[i, neuron_idx]] += signed_beta;
    }
    // err deliberately left unchanged — the bug.
}

/// Build a single-row, single-column `LinearBounds` that already carries a
/// certified coefficient error, where the *true* coefficient equals the stored
/// coefficient (err = 0 entries but the err arrays ARE attached, so
/// `has_coeff_err()` is true — exactly the "came from a tracked wide backward"
/// case). After a β subtraction that rounds, the unchanged err must under-count.
fn tracked_bounds_with(coeff: f32, init_err: f32) -> (LinearBounds, f64) {
    let lower_a = Array2::from_elem((1, 1), coeff);
    let upper_a = Array2::from_elem((1, 1), coeff);
    let mut lb = LinearBounds::new(lower_a, arr1(&[0.0_f32]), upper_a, arr1(&[0.0_f32])).unwrap();
    lb.set_coeff_err(
        Array2::from_elem((1, 1), init_err),
        Array2::from_elem((1, 1), init_err),
    );
    // The exact real lower/upper coefficient the certificate must track. With
    // init_err=0 the stored f32 IS the true coefficient.
    (lb, coeff as f64)
}

/// REPRODUCE (a): the OLD β-split (err untouched) UNDER-counts the f32 rounding
/// of `a − β`. VERIFY (b): the patched `apply_beta_split_to_column` covers it.
#[test]
fn beta_split_cert_covers_f32_error_and_old_undercounts() {
    // Large coefficient, small β → fl32(a − β) loses the low-order part of β.
    let coeff = 1.0e7_f32;
    let beta = 0.3_f32;
    let (_, true_lower) = tracked_bounds_with(coeff, 0.0);
    let true_upper = coeff as f64;

    // The exact real coefficients after the β split (β is a fixed real applied to
    // both stored and true coefficient).
    let exact_lower_after = true_lower - beta as f64;
    let exact_upper_after = true_upper + beta as f64;

    // ---- (a) REPRODUCE: OLD path under-counts. ----
    let (mut buggy, _) = tracked_bounds_with(coeff, 0.0);
    apply_beta_split_buggy(&mut buggy, 0, beta);
    let buggy_stored_lower = buggy.lower_a()[[0, 0]] as f64;
    let buggy_stored_upper = buggy.upper_a()[[0, 0]] as f64;
    let buggy_cert_lower = buggy.lower_a_err().unwrap()[[0, 0]] as f64;
    let buggy_cert_upper = buggy.upper_a_err().unwrap()[[0, 0]] as f64;

    let true_gap_lower = (buggy_stored_lower - exact_lower_after).abs();
    let true_gap_upper = (buggy_stored_upper - exact_upper_after).abs();
    eprintln!(
        "[beta-split repro] lower: stored={buggy_stored_lower:.6} exact={exact_lower_after:.6} \
         true_gap={true_gap_lower:.3e} OLD_cert={buggy_cert_lower:.3e}; \
         upper true_gap={true_gap_upper:.3e} OLD_cert={buggy_cert_upper:.3e}"
    );
    assert!(
        true_gap_lower > buggy_cert_lower,
        "EXPECTED OLD β-split certificate to UNDER-COUNT the lower f32 gap: \
         true_gap {true_gap_lower:.3e} should exceed OLD cert {buggy_cert_lower:.3e}"
    );
    assert!(
        true_gap_upper > buggy_cert_upper,
        "EXPECTED OLD β-split certificate to UNDER-COUNT the upper f32 gap: \
         true_gap {true_gap_upper:.3e} should exceed OLD cert {buggy_cert_upper:.3e}"
    );

    // ---- (b) VERIFY: patched path covers the f32 gap. ----
    let (mut fixed, _) = tracked_bounds_with(coeff, 0.0);
    fixed.apply_beta_split_to_column(0, beta);
    let fixed_stored_lower = fixed.lower_a()[[0, 0]] as f64;
    let fixed_stored_upper = fixed.upper_a()[[0, 0]] as f64;
    let fixed_cert_lower = fixed.lower_a_err().unwrap()[[0, 0]] as f64;
    let fixed_cert_upper = fixed.upper_a_err().unwrap()[[0, 0]] as f64;

    // Stored coefficient must be unchanged from the (correctly rounded) mutation.
    assert_eq!(fixed_stored_lower, buggy_stored_lower);
    assert_eq!(fixed_stored_upper, buggy_stored_upper);

    let fixed_gap_lower = (fixed_stored_lower - exact_lower_after).abs();
    let fixed_gap_upper = (fixed_stored_upper - exact_upper_after).abs();
    assert!(
        fixed_cert_lower >= fixed_gap_lower,
        "UNSOUND patched β-split lower certificate: cert {fixed_cert_lower:.3e} < \
         |stored_f32 − exact_f64| {fixed_gap_lower:.3e}"
    );
    assert!(
        fixed_cert_upper >= fixed_gap_upper,
        "UNSOUND patched β-split upper certificate: cert {fixed_cert_upper:.3e} < \
         |stored_f32 − exact_f64| {fixed_gap_upper:.3e}"
    );
}

/// f64-reference penalty soundness: the certified coefficient error is consumed
/// by `concretize` as an `err·max(|in_l|,|in_u|)` penalty (subtracted from lower,
/// added to upper). For the bound to enclose the true value, the SUMMED penalty
/// must cover the SUMMED true coefficient distance `Σ |stored_after − true_after|
/// · |x|`. This test computes both penalty sums in f64 (the exact arithmetic
/// `concretize` uses internally before the final directed-f32 round) over a WIDE
/// many-column column-vector, and asserts:
///   - the BUGGY penalty (err untouched) UNDER-counts the true distance → the
///     inner f64 bound would CROSS the true value (false-proof at the penalty
///     level — only the coarse outer f32 round can mask it, which is exactly the
///     latent danger when these errs are propagated/summed across deep nets);
///   - the PATCHED penalty COVERS the true distance with ZERO tolerance.
#[test]
fn beta_split_penalty_covers_true_coeff_distance_zero_tol() {
    // Many columns, each a coefficient large enough that fl32(c − β) rounds, so
    // every column drops the SAME ~0.3 gap. Summed over the column dimension the
    // dropped distance is large and coherent.
    let n_cols = 4096usize;
    let coeff = 1.0e7_f32;
    let beta = 0.3_f32;
    let mag = 1.0e3_f64; // max(|in_l|, |in_u|) per column

    // True (real) coefficient per column == stored (init err 0).
    let lower_a = Array2::from_elem((1, n_cols), coeff);
    let upper_a = Array2::from_elem((1, n_cols), coeff);
    let make = || {
        let mut lb = LinearBounds::new(
            lower_a.clone(),
            arr1(&[0.0_f32]),
            upper_a.clone(),
            arr1(&[0.0_f32]),
        )
        .unwrap();
        lb.set_coeff_err(Array2::zeros((1, n_cols)), Array2::zeros((1, n_cols)));
        lb
    };

    // Per-column true post-β coefficients (exact real).
    let exact_lower_after = coeff as f64 - beta as f64;
    let exact_upper_after = coeff as f64 + beta as f64;

    // Sum in f64 the certified penalty vs the true required distance over columns.
    let penalty_sums = |lb: &LinearBounds| -> (f64, f64, f64, f64) {
        let le = lb.lower_a_err().unwrap();
        let ue = lb.upper_a_err().unwrap();
        let mut cert_l = 0.0;
        let mut cert_u = 0.0;
        let mut true_l = 0.0;
        let mut true_u = 0.0;
        for j in 0..n_cols {
            let stored_lo = lb.lower_a()[[0, j]] as f64;
            let stored_hi = lb.upper_a()[[0, j]] as f64;
            cert_l += le[[0, j]] as f64 * mag;
            cert_u += ue[[0, j]] as f64 * mag;
            true_l += (stored_lo - exact_lower_after).abs() * mag;
            true_u += (stored_hi - exact_upper_after).abs() * mag;
        }
        (cert_l, cert_u, true_l, true_u)
    };

    // Each column is a distinct constrained neuron getting its own β split.
    // concretize then sums all their err penalties — the coherent accumulation
    // that makes the dropped per-coefficient gap dangerous in a real bound.
    let split_all = |lb: &mut LinearBounds, helper: fn(&mut LinearBounds, usize, f32)| {
        for j in 0..n_cols {
            helper(lb, j, beta);
        }
    };

    // BUGGY: err untouched ⇒ certified penalty 0 but true distance is large.
    let mut buggy = make();
    split_all(&mut buggy, apply_beta_split_buggy);
    let (b_cert_l, b_cert_u, b_true_l, b_true_u) = penalty_sums(&buggy);
    eprintln!(
        "[beta-split penalty] BUGGY cert=({b_cert_l:.3e},{b_cert_u:.3e}) \
         true=({b_true_l:.3e},{b_true_u:.3e})"
    );
    assert!(
        b_cert_l < b_true_l && b_cert_u < b_true_u,
        "EXPECTED buggy β-split penalty to UNDER-COUNT the true coeff distance: \
         cert=({b_cert_l:.3e},{b_cert_u:.3e}) should be < true=({b_true_l:.3e},{b_true_u:.3e})"
    );

    // PATCHED: certified penalty must cover the true distance with zero tolerance.
    let mut fixed = make();
    split_all(&mut fixed, LinearBounds::apply_beta_split_to_column);
    let (f_cert_l, f_cert_u, f_true_l, f_true_u) = penalty_sums(&fixed);
    eprintln!(
        "[beta-split penalty] FIXED cert=({f_cert_l:.3e},{f_cert_u:.3e}) \
         true=({f_true_l:.3e},{f_true_u:.3e})"
    );
    assert!(
        f_cert_l >= f_true_l,
        "UNSOUND: patched lower penalty {f_cert_l:.3e} < true distance {f_true_l:.3e}"
    );
    assert!(
        f_cert_u >= f_true_u,
        "UNSOUND: patched upper penalty {f_cert_u:.3e} < true distance {f_true_u:.3e}"
    );
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 500,
        max_shrink_time: 5000,
        ..ProptestConfig::default()
    })]

    /// For ANY tracked bounds and ANY finite β, the patched β-split certificate
    /// must cover the actual f32 rounding gap of the mutation, per coefficient.
    #[test]
    fn proptest_beta_split_cert_covers_f32_error(
        coeffs in proptest::collection::vec(-1.0e7_f32..1.0e7_f32, 3),
        init_errs in proptest::collection::vec(0.0_f32..1.0_f32, 3),
        signed_beta in -1.0e6_f32..1.0e6_f32,
    ) {
        let n_out = coeffs.len();
        let lower_a = Array2::from_shape_vec((n_out, 1), coeffs.clone()).unwrap();
        let upper_a = Array2::from_shape_vec((n_out, 1), coeffs.clone()).unwrap();
        let mut lb = LinearBounds::new(
            lower_a,
            Array1::zeros(n_out),
            upper_a,
            Array1::zeros(n_out),
        ).unwrap();
        lb.set_coeff_err(
            Array2::from_shape_vec((n_out, 1), init_errs.clone()).unwrap(),
            Array2::from_shape_vec((n_out, 1), init_errs.clone()).unwrap(),
        );

        // Exact pre-mutation truth interval per coeff: [stored - err, stored + err].
        // The β subtraction applies an exact real β to BOTH bounds of that interval.
        let pre_lower_stored: Vec<f64> = coeffs.iter().map(|&c| c as f64).collect();

        lb.apply_beta_split_to_column(0, signed_beta);

        for i in 0..n_out {
            let stored_lower_after = lb.lower_a()[[i, 0]] as f64;
            let stored_upper_after = lb.upper_a()[[i, 0]] as f64;
            let cert_lower = lb.lower_a_err().unwrap()[[i, 0]] as f64;
            let cert_upper = lb.upper_a_err().unwrap()[[i, 0]] as f64;

            // Worst-case exact true coefficient after the split: the original true
            // coeff lies in [stored - err, stored + err]; β shifts it by ∓/±β.
            // The certificate must cover the maximum |stored_after − true_after|.
            let e0 = init_errs[i] as f64;
            // Lower relaxation: stored_after = fl32(c - β); true_after ∈
            // [(c - e0) - β, (c + e0) - β]. Max distance = |fl32(c-β) - (c-β)| + e0.
            let round_gap_lower = (stored_lower_after - (pre_lower_stored[i] - signed_beta as f64)).abs();
            let required_lower = round_gap_lower + e0;
            // Upper: stored_after = fl32(c + β); true_after ∈ [(c-e0)+β,(c+e0)+β].
            let round_gap_upper = (stored_upper_after - (pre_lower_stored[i] + signed_beta as f64)).abs();
            let required_upper = round_gap_upper + e0;

            prop_assert!(
                cert_lower >= required_lower,
                "lower cert {cert_lower:.3e} < required {required_lower:.3e} \
                 (round_gap {round_gap_lower:.3e} + e0 {e0:.3e}) at row {i}"
            );
            prop_assert!(
                cert_upper >= required_upper,
                "upper cert {cert_upper:.3e} < required {required_upper:.3e} \
                 (round_gap {round_gap_upper:.3e} + e0 {e0:.3e}) at row {i}"
            );
        }
    }
}
