// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::super::super::matmul::{decode_batch_index_into_buf, parse_matmul_dims};
use super::super::interpolated_mccormick;
use super::BilinearRelaxation;

enum RelaxationBuildMode<'a> {
    Midpoint,
    Alpha(&'a ndarray::Array4<f32>),
}

struct RelaxationBuildContext {
    batch_dims: Vec<usize>,
    batch_size: usize,
    m: usize,
    n: usize,
    k: usize,
    transpose_b: bool,
    scale: f32,
}

impl RelaxationBuildContext {
    fn new(
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        transpose_b: bool,
        scale: Option<f32>,
    ) -> Result<Self> {
        let dims = parse_matmul_dims(transpose_b, input_a_bounds.shape(), input_b_bounds.shape())?;
        if dims.batch_dims.contains(&0) {
            return Err(NyError::InvalidSpec(format!(
                "BilinearRelaxation: zero-valued batch dimension in {:?}",
                dims.batch_dims
            )));
        }

        let batch_size = dims.batch_size()?;
        let batch_dims = dims.batch_dims;
        let m = dims.m;
        let n = dims.n;
        let k = dims.k;

        let scale = scale.unwrap_or(1.0);
        if scale < 0.0 {
            return Err(NyError::UnsupportedOp(
                "BilinearRelaxation does not support negative scale".to_string(),
            ));
        }

        Ok(Self {
            batch_dims,
            batch_size,
            m,
            n,
            k,
            transpose_b,
            scale,
        })
    }

    fn validate_alpha_shape(&self, alphas: &ndarray::Array4<f32>) -> Result<()> {
        if alphas.shape() != [2, self.m, self.n, self.k] {
            return Err(NyError::ShapeMismatch {
                expected: vec![2, self.m, self.n, self.k],
                got: alphas.shape().to_vec(),
            });
        }
        Ok(())
    }
}

fn build_relaxation(
    input_a_bounds: &BoundedTensor,
    input_b_bounds: &BoundedTensor,
    transpose_b: bool,
    scale: Option<f32>,
    mode: RelaxationBuildMode<'_>,
) -> Result<BilinearRelaxation> {
    let context = RelaxationBuildContext::new(input_a_bounds, input_b_bounds, transpose_b, scale)?;
    if let RelaxationBuildMode::Alpha(alphas) = mode {
        context.validate_alpha_shape(alphas)?;
    }

    let batch_size = context.batch_size;
    let m = context.m;
    let n = context.n;
    let k = context.k;
    let scale = context.scale;
    let transpose_b = context.transpose_b;
    let batch_dims = context.batch_dims;
    let batch_index_len = batch_dims.len();

    let mut alpha_lower = ndarray::Array4::<f32>::zeros((batch_size, m, n, k));
    let mut alpha_upper = ndarray::Array4::<f32>::zeros((batch_size, m, n, k));
    let mut beta_lower = ndarray::Array4::<f32>::zeros((batch_size, m, n, k));
    let mut beta_upper = ndarray::Array4::<f32>::zeros((batch_size, m, n, k));
    let mut bias_lower_f64 = ndarray::Array3::<f64>::zeros((batch_size, m, n));
    let mut bias_upper_f64 = ndarray::Array3::<f64>::zeros((batch_size, m, n));
    // Stack-allocated index buffers — batch_index_len + 2 is always small
    // (batch dims 0-5 + m,k indices). Eliminates per-call heap allocations (#2237 F4).
    assert!(
        batch_index_len + 2 <= 8,
        "BilinearRelaxation: batch_index_len + 2 exceeds stack buffer"
    );
    let mut a_idx = [0usize; 8];
    let mut b_idx = [0usize; 8];
    let idx_len = batch_index_len + 2;

    for batch_idx in 0..batch_size {
        decode_batch_index_into_buf(batch_idx, &batch_dims, &mut a_idx[..batch_index_len])?;
        b_idx[..batch_index_len].copy_from_slice(&a_idx[..batch_index_len]);
        a_idx[batch_index_len + 1] = 0;
        b_idx[batch_index_len] = 0;
        b_idx[batch_index_len + 1] = 0;

        for i in 0..m {
            a_idx[batch_index_len] = i;
            for j in 0..n {
                if transpose_b {
                    b_idx[batch_index_len] = j;
                } else {
                    b_idx[batch_index_len + 1] = j;
                }
                for l in 0..k {
                    a_idx[batch_index_len + 1] = l;
                    let q_l = input_a_bounds.lower()[&a_idx[..idx_len]];
                    let q_u = input_a_bounds.upper()[&a_idx[..idx_len]];

                    if transpose_b {
                        b_idx[batch_index_len + 1] = l;
                    } else {
                        b_idx[batch_index_len] = l;
                    }
                    let k_l = input_b_bounds.lower()[&b_idx[..idx_len]];
                    let k_u = input_b_bounds.upper()[&b_idx[..idx_len]];

                    let (ax_l, ay_l, c_l, ax_u, ay_u, c_u) = match mode {
                        RelaxationBuildMode::Midpoint => {
                            // Bit-identical McCormick anchors: f32::midpoint rounds differently at overflow/subnormal edges.
                            #[allow(clippy::manual_midpoint)]
                            let q0 = (q_l + q_u) * 0.5;
                            #[allow(clippy::manual_midpoint)]
                            let k0 = (k_l + k_u) * 0.5;

                            let l1_val = k_l * q0 + q_l * k0 - q_l * k_l;
                            let l2_val = k_u * q0 + q_u * k0 - q_u * k_u;
                            let (ax_l, ay_l, c_l) = if l1_val >= l2_val {
                                (k_l, q_l, -q_l * k_l)
                            } else {
                                (k_u, q_u, -q_u * k_u)
                            };

                            let u1_val = k_u * q0 + q_l * k0 - q_l * k_u;
                            let u2_val = k_l * q0 + q_u * k0 - q_u * k_l;
                            let (ax_u, ay_u, c_u) = if u1_val <= u2_val {
                                (k_u, q_l, -q_l * k_u)
                            } else {
                                (k_l, q_u, -q_u * k_l)
                            };

                            (ax_l, ay_l, c_l, ax_u, ay_u, c_u)
                        }
                        RelaxationBuildMode::Alpha(alphas) => {
                            let r_l = alphas[[0, i, j, l]].clamp(0.0, 1.0);
                            let r_u = alphas[[1, i, j, l]].clamp(0.0, 1.0);
                            interpolated_mccormick(q_l, q_u, k_l, k_u, r_l, r_u)
                        }
                    };

                    alpha_lower[[batch_idx, i, j, l]] = scale * ax_l;
                    alpha_upper[[batch_idx, i, j, l]] = scale * ax_u;
                    beta_lower[[batch_idx, i, j, l]] = scale * ay_l;
                    beta_upper[[batch_idx, i, j, l]] = scale * ay_u;
                    bias_lower_f64[[batch_idx, i, j]] += scale as f64 * c_l as f64;
                    bias_upper_f64[[batch_idx, i, j]] += scale as f64 * c_u as f64;
                }
            }
        }
    }

    let bias_lower = bias_lower_f64.mapv(|v| next_down_f32(v as f32));
    let bias_upper = bias_upper_f64.mapv(|v| next_up_f32(v as f32));

    Ok(BilinearRelaxation {
        alpha_lower,
        alpha_upper,
        beta_lower,
        beta_upper,
        bias_lower,
        bias_upper,
        batch_dims,
        m,
        n,
        k,
        transpose_b,
    })
}

impl BilinearRelaxation {
    /// Build per-batch McCormick coefficients from Q and K interval bounds.
    ///
    /// Computes McCormick planes per-batch position, which is tighter than
    /// batch-reduced global intervals when different positions have different widths.
    ///
    /// Uses the midpoint heuristic for plane selection.
    /// For alpha-optimized variants, use `from_bounds_with_alpha`.
    pub(crate) fn from_bounds(
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        transpose_b: bool,
        scale: Option<f32>,
    ) -> Result<Self> {
        build_relaxation(
            input_a_bounds,
            input_b_bounds,
            transpose_b,
            scale,
            RelaxationBuildMode::Midpoint,
        )
    }

    /// Build per-batch McCormick coefficients with alpha-parameterized interpolation.
    ///
    /// Uses `interpolated_mccormick` instead of the fixed midpoint heuristic,
    /// enabling gradient-based optimization of McCormick face selection.
    ///
    /// # Alpha Parameters
    /// `alphas` has shape [2, m, n, k] where:
    /// - alphas[[0, i, j, l]] = r_l: interpolation parameter for lower bound
    /// - alphas[[1, i, j, l]] = r_u: interpolation parameter for upper bound
    ///
    /// # Reference
    /// auto_LiRPA/operators/bivariate.py:MulHelper.get_relaxation
    pub(crate) fn from_bounds_with_alpha(
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        transpose_b: bool,
        scale: Option<f32>,
        alphas: &ndarray::Array4<f32>,
    ) -> Result<Self> {
        build_relaxation(
            input_a_bounds,
            input_b_bounds,
            transpose_b,
            scale,
            RelaxationBuildMode::Alpha(alphas),
        )
    }
}
