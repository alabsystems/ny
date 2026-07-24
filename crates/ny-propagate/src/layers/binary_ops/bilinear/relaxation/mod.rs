// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

mod broadcast;
mod build;
#[cfg(test)]
mod interval;

/// Element-wise McCormick relaxation with per-batch coefficients (#286 Approach A).
///
/// For z = X @ Y^T (or X @ Y) where X: [batch..., m, k], Y: [batch..., n, k]:
/// - Each element z[b,i,j] = sum_l x[b,i,l] * y[b,j,l] is relaxed via McCormick
/// - Coefficients are stored **per-batch-position** (not batch-reduced global intervals)
///
/// For each contraction index l:
///   z[b,i,j] >= alpha_l[b,i,j,l] * x[b,i,l] + beta_l[b,i,j,l] * y[b,j,l] + ny_l_per_l
///   z[b,i,j] <= alpha_u[b,i,j,l] * x[b,i,l] + beta_u[b,i,j,l] * y[b,j,l] + ny_u_per_l
///
/// The bias (ny) is pre-summed over l for storage efficiency.
///
/// Storage: O(batch_size * m * n * k) per coefficient array.
/// For batch_size>1, produces tighter bounds than batch-reduced global intervals.
///
/// # Reference
/// - Design: `designs/2026-03-04-286-attention-bilinear-alternative.md` Approach A
/// - auto_LiRPA: `operators/bivariate.py` MulHelper.interpolated_relaxation
/// - McCormick (1976): "Computability of global solutions to factorable nonconvex programs"
#[derive(Debug, Clone)]
pub(crate) struct BilinearRelaxation {
    /// Q-coefficient for lower bound: shape [batch_size, m, n, k]
    alpha_lower: ndarray::Array4<f32>,
    /// Q-coefficient for upper bound: shape [batch_size, m, n, k]
    alpha_upper: ndarray::Array4<f32>,
    /// K-coefficient for lower bound: shape [batch_size, m, n, k]
    beta_lower: ndarray::Array4<f32>,
    /// K-coefficient for upper bound: shape [batch_size, m, n, k]
    beta_upper: ndarray::Array4<f32>,
    /// Bias for lower bound: shape [batch_size, m, n] (summed over l, f64→f32 directed rounding)
    bias_lower: ndarray::Array3<f32>,
    /// Bias for upper bound: shape [batch_size, m, n] (summed over l, f64→f32 directed rounding)
    bias_upper: ndarray::Array3<f32>,
    /// Original batch dimensions (e.g. [B, H] for attention), used for output reshaping
    batch_dims: Vec<usize>,
    m: usize,
    n: usize,
    k: usize,
    transpose_b: bool,
}

// CompactMcCormick removed (#286 Phase 5): superseded by BilinearRelaxation with
// per-batch coefficients and broadcast sign-split composition. See git history for
// the original global-interval implementation used as a test baseline.
