// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Naive triple-loop CPU GEMM for testing and fallback.

use crate::{NyError, Result};

use super::{ConvTranspose2dParams, GemmEngine};

/// Naive triple-loop CPU GEMM for testing and fallback.
///
/// O(m*k*n) with no SIMD, tiling, or parallelism. Suitable for small matrices
/// in tests and as a reference implementation. For production use, prefer
/// GPU-accelerated `ComputeDevice` from ny-gpu.
#[derive(Debug, Clone, Copy)]
pub struct NaiveCpuGemmEngine;

impl GemmEngine for NaiveCpuGemmEngine {
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
        let spatial = oh * ow;
        let total_rows = s * spatial;
        let kernel_cols = ic * kh * kw;

        if a_reshaped.len() != total_rows * oc {
            return Err(NyError::InvalidSpec(format!(
                "conv_transpose_2d: a_reshaped.len()={} != S*OH*OW*OC={}",
                a_reshaped.len(),
                total_rows * oc,
            )));
        }
        if weight_col.len() != oc * kernel_cols {
            return Err(NyError::InvalidSpec(format!(
                "conv_transpose_2d: weight_col.len()={} != OC*IC*KH*KW={}",
                weight_col.len(),
                oc * kernel_cols,
            )));
        }

        // Step 1: GEMM — (S*OH*OW, OC) × (OC, IC*KH*KW) → (S*OH*OW, IC*KH*KW)
        let gemm_out = self.gemm_f32(total_rows, oc, kernel_cols, a_reshaped, weight_col)?;

        // Step 2: col2im scatter → (S, IC*IH*IW)
        // Reference: ops_transpose_gemm.rs col2im loop (lines 226-250).
        let flat_input_dim = ic * ih * iw;
        let mut result = vec![0.0f32; s * flat_input_dim];

        for spec in 0..s {
            for gy in 0..oh {
                for gx in 0..ow {
                    let gemm_row = spec * spatial + gy * ow + gx;
                    for ic_idx in 0..ic {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let iy = (gy * sh + ki) as isize - ph as isize;
                                let ix = (gx * sw + kj) as isize - pw as isize;
                                if iy >= 0 && iy < ih as isize && ix >= 0 && ix < iw as isize {
                                    let col_idx = ic_idx * kh * kw + ki * kw + kj;
                                    let out_idx = ic_idx * ih * iw + iy as usize * iw + ix as usize;
                                    result[spec * flat_input_dim + out_idx] +=
                                        gemm_out[gemm_row * kernel_cols + col_idx];
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        if a.len() != m * k {
            return Err(NyError::InvalidSpec(format!(
                "GEMM f64: a.len()={} != m*k={}*{}={}",
                a.len(),
                m,
                k,
                m * k
            )));
        }
        if b.len() != k * n {
            return Err(NyError::InvalidSpec(format!(
                "GEMM f64: b.len()={} != k*n={}*{}={}",
                b.len(),
                k,
                n,
                k * n
            )));
        }

        let mut c = vec![0.0f64; m * n];
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
        if a.len() != m * k {
            return Err(NyError::InvalidSpec(format!(
                "GEMM: a.len()={} != m*k={}*{}={}",
                a.len(),
                m,
                k,
                m * k
            )));
        }
        if b.len() != k * n {
            return Err(NyError::InvalidSpec(format!(
                "GEMM: b.len()={} != k*n={}*{}={}",
                b.len(),
                k,
                n,
                k * n
            )));
        }

        let mut c = vec![0.0f32; m * n];
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
