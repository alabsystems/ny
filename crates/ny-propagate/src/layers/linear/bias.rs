// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bias accumulation and directed-rounding finalization for linear CROWN.
//!
//! Centralizes the f64 bias accumulation pattern shared by CPU CROWN,
//! GEMM CROWN, and batched CROWN backward paths.
//!
//! # Precision rationale
//!
//! Bias accumulation uses f64 to prevent catastrophic cancellation when
//! large positive and negative contributions nearly cancel. After
//! accumulation, the result is cast back to f32 with directed rounding
//! (`next_down_f32` for lower bounds, `next_up_f32` for upper bounds)
//! to maintain soundness.
//!
//! References:
//! - Issue #1863: f64 accumulation requirement
//! - Issue #2164: directed-rounding finalization
//! - `common.rs:170-175`: nonlinear path's equivalent precision standard

use ndarray::Array1;
use ny_tensor::{next_down_f32, next_up_f32};

/// Parameters for one position block of bias accumulation.
pub(crate) struct BiasBlockParams {
    /// Number of output rows in the A matrix.
    pub num_outputs: usize,
    /// Number of features in this position block.
    pub out_features: usize,
    /// Column offset into the A matrix for this position.
    pub col_offset: usize,
}

/// Accumulate the bias contribution for one position block of the A matrix.
///
/// For each output row `i`, computes:
///   `accum.0[i] += sum_j A_lower[i, offset+j] * bias[j]`  (lower)
///   `accum.1[i] += sum_j A_upper[i, offset+j] * bias[j]`  (upper)
///
/// All arithmetic is in f64 to prevent catastrophic cancellation (#1863).
pub(crate) fn accumulate_bias_f64(
    accum: &mut (&mut [f64], &mut [f64]),
    lower_a_val: impl Fn(usize, usize) -> f32,
    upper_a_val: impl Fn(usize, usize) -> f32,
    bias: &Array1<f32>,
    block: &BiasBlockParams,
) {
    for i in 0..block.num_outputs {
        for j in 0..block.out_features {
            let col = block.col_offset + j;
            accum.0[i] += lower_a_val(i, col) as f64 * bias[j] as f64;
            accum.1[i] += upper_a_val(i, col) as f64 * bias[j] as f64;
        }
    }
}

/// Finalize bias accumulators with directed rounding.
///
/// Adds the original bias vector (promoted to f64) to the accumulated
/// bias contributions, then casts back to f32 with directed rounding:
/// - Lower bounds use `next_down_f32` (round toward -inf)
/// - Upper bounds use `next_up_f32` (round toward +inf)
///
/// When the layer has no bias, returns `(old_lower_b.clone(), old_upper_b.clone())`
/// without any rounding (since the f32->f64->f32 round-trip is exact and
/// `next_down`/`next_up` would needlessly widen by 1 ULP).
pub(crate) fn finalize_bias_directed(
    lower_accum: &Array1<f64>,
    upper_accum: &Array1<f64>,
    old_lower_b: &Array1<f32>,
    old_upper_b: &Array1<f32>,
) -> (Array1<f32>, Array1<f32>) {
    let lb_f64 = old_lower_b.mapv(|x| x as f64) + lower_accum;
    let ub_f64 = old_upper_b.mapv(|x| x as f64) + upper_accum;
    (
        lb_f64.mapv(|x| next_down_f32(x as f32)),
        ub_f64.mapv(|x| next_up_f32(x as f32)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_accumulate_bias_basic() {
        // A = [[1, 0], [0, 1]], bias = [2, 3]
        // lower_accum[0] += 1*2 + 0*3 = 2
        // lower_accum[1] += 0*2 + 1*3 = 3
        let mut lower_accum = [0.0_f64; 2];
        let mut upper_accum = [0.0_f64; 2];
        let bias = array![2.0_f32, 3.0];

        let a = [[1.0_f32, 0.0], [0.0_f32, 1.0]];
        let block = BiasBlockParams {
            num_outputs: 2,
            out_features: 2,
            col_offset: 0,
        };
        accumulate_bias_f64(
            &mut (&mut lower_accum[..], &mut upper_accum[..]),
            |i, j| a[i][j],
            |i, j| a[i][j],
            &bias,
            &block,
        );

        assert!((lower_accum[0] - 2.0).abs() < 1e-10);
        assert!((lower_accum[1] - 3.0).abs() < 1e-10);
        assert!((upper_accum[0] - 2.0).abs() < 1e-10);
        assert!((upper_accum[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_finalize_directed_rounding_widens() {
        // With non-zero bias contribution, directed rounding should widen bounds
        let lower_accum = array![1.0_f64];
        let upper_accum = array![1.0_f64];
        let old_lower_b = array![0.0_f32];
        let old_upper_b = array![0.0_f32];

        let (lb, ub) =
            finalize_bias_directed(&lower_accum, &upper_accum, &old_lower_b, &old_upper_b);

        // next_down_f32(1.0) < 1.0, next_up_f32(1.0) > 1.0
        assert!(lb[0] <= 1.0_f32, "lower should be <= 1.0, got {}", lb[0]);
        assert!(ub[0] >= 1.0_f32, "upper should be >= 1.0, got {}", ub[0]);
    }

    #[test]
    fn test_finalize_no_bias_exact() {
        // When bias accum is zero, the result is old_b with directed rounding.
        // next_down_f32(5.0) rounds down by 1 ULP and next_up_f32(5.0) rounds up by 1 ULP.
        let lower_accum = array![0.0_f64];
        let upper_accum = array![0.0_f64];
        let old_lower_b = array![5.0_f32];
        let old_upper_b = array![5.0_f32];

        let (lb, ub) =
            finalize_bias_directed(&lower_accum, &upper_accum, &old_lower_b, &old_upper_b);

        // Directed rounding: lower bound goes down, upper goes up
        assert!(lb[0] <= 5.0_f32, "lower must be <= 5.0, got {}", lb[0]);
        assert!(ub[0] >= 5.0_f32, "upper must be >= 5.0, got {}", ub[0]);
        assert_eq!(lb[0], next_down_f32(5.0_f32));
        assert_eq!(ub[0], next_up_f32(5.0_f32));
    }
}
