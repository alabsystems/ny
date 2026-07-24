// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Spectral norm helpers for linear layers.

use ndarray::Array2;
use ny_core::{nan_propagating_max_f64, nan_propagating_min_f64};
use ny_tensor::next_up_f32;

/// Compute a sound upper bound on the spectral norm (largest singular value).
///
/// We avoid fixed-step power iteration here because it converges from below and can
/// underestimate `sigma_max`, which is unsound for SDP-CROWN radius propagation.
///
/// Instead we use two matrix-norm upper bounds and take the tighter one:
/// - `sigma_max(A) <= ||A||_F`
/// - `sigma_max(A)^2 = rho(A^T A) <= ||A^T A||_inf <= ||A||_1 * ||A||_inf`
///
/// Reference: Golub & Van Loan, _Matrix Computations_ (4th ed.), Ch. 2 and Ch. 8.
pub(super) fn compute_spectral_norm(weight: &Array2<f32>) -> f32 {
    let (m, n) = (weight.nrows(), weight.ncols());
    if m == 0 || n == 0 {
        return 0.0;
    }

    let mut frobenius_sq = 0.0_f64;
    let mut max_row_abs_sum = 0.0_f64;
    let mut col_abs_sums = vec![0.0_f64; n];

    for i in 0..m {
        let mut row_abs_sum = 0.0_f64;
        for j in 0..n {
            let entry = weight[[i, j]];
            if !entry.is_finite() {
                // Non-finite weights imply an unbounded operator norm.
                return f32::INFINITY;
            }

            let abs_entry = f64::from(entry.abs());
            frobenius_sq += abs_entry * abs_entry;
            row_abs_sum += abs_entry;
            col_abs_sums[j] += abs_entry;
        }
        max_row_abs_sum = nan_propagating_max_f64(max_row_abs_sum, row_abs_sum);
    }

    let max_col_abs_sum = col_abs_sums
        .into_iter()
        .fold(0.0_f64, nan_propagating_max_f64);
    let frobenius_upper = frobenius_sq.sqrt();
    let induced_upper = (max_row_abs_sum * max_col_abs_sum).sqrt();
    let upper = nan_propagating_min_f64(frobenius_upper, induced_upper);

    if !upper.is_finite() {
        return f32::INFINITY;
    }
    if upper == 0.0 {
        return 0.0;
    }

    // Round upward to preserve the upper-bound guarantee through f64 -> f32 conversion.
    next_up_f32(upper as f32)
}
