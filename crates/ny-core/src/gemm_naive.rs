// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Naive triple-loop CPU GEMM for testing and fallback.

use crate::{checked_dim_product, NyError, Result};

use super::{ConvTranspose2dParams, GemmEngine};

fn zeroed_f32(elements: usize, context: &str) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values.try_reserve_exact(elements).map_err(|error| {
        NyError::InvalidSpec(format!(
            "{context}: allocation failed for {elements} elements: {error}"
        ))
    })?;
    values.resize(elements, 0.0);
    Ok(values)
}

/// Naive triple-loop CPU GEMM for testing and fallback.
///
/// O(m*k*n) with no SIMD, tiling, or parallelism. Suitable for small matrices
/// in tests and as a reference implementation. For production use, prefer
/// GPU-accelerated `ComputeDevice` from ny-gpu.
#[derive(Debug, Clone, Copy)]
pub struct NaiveCpuGemmEngine;

impl GemmEngine for NaiveCpuGemmEngine {
    fn backend_provenance(&self) -> &'static str {
        "naive-cpu"
    }

    fn conv_transpose_2d(
        &self,
        a_reshaped: &[f32],
        weight_col: &[f32],
        params: &ConvTranspose2dParams,
    ) -> Result<Vec<f32>> {
        let s = params.num_specs;
        let oc = params.out_channels;
        let ic = params.in_channels;
        let (oh, ow) = (params.out_h, params.out_w);
        let (ih, iw) = (params.in_h, params.in_w);
        let (kh, kw) = (params.kernel_h, params.kernel_w);
        let (sh, sw) = (params.stride_h, params.stride_w);
        let (ph, pw) = (params.pad_h, params.pad_w);
        let spatial = checked_dim_product(&[oh, ow], "conv_transpose_2d output spatial")?;
        let total_rows = checked_dim_product(&[s, spatial], "conv_transpose_2d GEMM rows")?;
        let kernel_cols = checked_dim_product(&[ic, kh, kw], "conv_transpose_2d kernel columns")?;
        let a_len = checked_dim_product(&[total_rows, oc], "conv_transpose_2d a_reshaped")?;
        let weight_len = checked_dim_product(&[oc, kernel_cols], "conv_transpose_2d weight_col")?;

        if a_reshaped.len() != a_len {
            return Err(NyError::InvalidSpec(format!(
                "conv_transpose_2d: a_reshaped.len()={} != S*OH*OW*OC={}",
                a_reshaped.len(),
                a_len,
            )));
        }
        if weight_col.len() != weight_len {
            return Err(NyError::InvalidSpec(format!(
                "conv_transpose_2d: weight_col.len()={} != OC*IC*KH*KW={}",
                weight_col.len(),
                weight_len,
            )));
        }

        let flat_input_dim = checked_dim_product(&[ic, ih, iw], "conv_transpose_2d flat input")?;
        let result_len = checked_dim_product(&[s, flat_input_dim], "conv_transpose_2d output")?;

        // Prove the loop's coordinate arithmetic cannot wrap before using plain
        // multiplication/addition in the hot path.
        let _max_y_padded = oh
            .saturating_sub(1)
            .checked_mul(sh)
            .and_then(|v| v.checked_add(kh.saturating_sub(1)))
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "conv_transpose_2d: vertical stride/kernel coordinate overflow".to_string(),
                )
            })?;
        let _max_x_padded = ow
            .saturating_sub(1)
            .checked_mul(sw)
            .and_then(|v| v.checked_add(kw.saturating_sub(1)))
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "conv_transpose_2d: horizontal stride/kernel coordinate overflow".to_string(),
                )
            })?;

        if total_rows == 0 || oc == 0 || kernel_cols == 0 {
            return zeroed_f32(result_len, "conv_transpose_2d output");
        }

        // Step 1: GEMM — (S*OH*OW, OC) × (OC, IC*KH*KW) → (S*OH*OW, IC*KH*KW)
        let gemm_out = self.gemm_f32(total_rows, oc, kernel_cols, a_reshaped, weight_col)?;

        // Step 2: col2im scatter → (S, IC*IH*IW)
        // Reference: ops_transpose_gemm.rs col2im loop (lines 226-250).
        let mut result = zeroed_f32(result_len, "conv_transpose_2d output")?;

        for spec in 0..s {
            for gy in 0..oh {
                for gx in 0..ow {
                    let gemm_row = spec * spatial + gy * ow + gx;
                    for ic_idx in 0..ic {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let padded_y = gy * sh + ki;
                                let padded_x = gx * sw + kj;
                                if padded_y < ph || padded_x < pw {
                                    continue;
                                }
                                let iy = padded_y - ph;
                                let ix = padded_x - pw;
                                if iy >= ih || ix >= iw {
                                    continue;
                                }
                                let col_idx = ic_idx * kh * kw + ki * kw + kj;
                                let out_idx = ic_idx * ih * iw + iy * iw + ix;
                                result[spec * flat_input_dim + out_idx] +=
                                    gemm_out[gemm_row * kernel_cols + col_idx];
                            }
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        let a_len = checked_dim_product(&[m, k], "GEMM f64 lhs")?;
        let b_len = checked_dim_product(&[k, n], "GEMM f64 rhs")?;
        let output_len = checked_dim_product(&[m, n], "GEMM f64 output")?;
        if a.len() != a_len {
            return Err(NyError::InvalidSpec(format!(
                "GEMM f64: a.len()={} != m*k={}*{}={}",
                a.len(),
                m,
                k,
                a_len
            )));
        }
        if b.len() != b_len {
            return Err(NyError::InvalidSpec(format!(
                "GEMM f64: b.len()={} != k*n={}*{}={}",
                b.len(),
                k,
                n,
                b_len
            )));
        }

        let mut c = vec![0.0f64; output_len];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        Ok(c)
    }

    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        let a_len = checked_dim_product(&[m, k], "GEMM lhs")?;
        let b_len = checked_dim_product(&[k, n], "GEMM rhs")?;
        let output_len = checked_dim_product(&[m, n], "GEMM output")?;
        if a.len() != a_len {
            return Err(NyError::InvalidSpec(format!(
                "GEMM: a.len()={} != m*k={}*{}={}",
                a.len(),
                m,
                k,
                a_len
            )));
        }
        if b.len() != b_len {
            return Err(NyError::InvalidSpec(format!(
                "GEMM: b.len()={} != k*n={}*{}={}",
                b.len(),
                k,
                n,
                b_len
            )));
        }

        let mut c = vec![0.0f32; output_len];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        Ok(c)
    }
}
