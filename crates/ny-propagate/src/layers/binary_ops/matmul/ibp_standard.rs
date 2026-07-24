// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Standard interval arithmetic IBP propagation for MatMul.

use ndarray::{ArrayD, IxDyn};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};

use super::shape::{decode_batch_index_into, parse_matmul_dims};
use super::{MatMulLayer, Result};

impl MatMulLayer {
    /// Propagate IBP bounds through matrix multiplication using standard
    /// interval arithmetic.
    ///
    /// For each element-wise product in the sum, computes min/max over the four
    /// corner products. NaN from 0*inf is widened during accumulation; final
    /// non-finite repair happens at the type boundary via `new_repaired` (#3423).
    /// Reference: alpha-beta-CROWN interval_bound.py:73-93.
    pub(super) fn propagate_ibp_standard(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        let dims = parse_matmul_dims(self.transpose_b, input_a.shape(), input_b.shape())?;

        let mut out_shape = dims.batch_dims.clone();
        out_shape.push(dims.m);
        out_shape.push(dims.n);

        let mut out_lower = ArrayD::zeros(IxDyn(&out_shape));
        let mut out_upper = ArrayD::zeros(IxDyn(&out_shape));

        let batch_size = dims.batch_size()?;
        let bd_len = dims.batch_dims.len();

        // Reusable index buffers: batch prefix + 2 suffix slots.
        // Eliminates batch_size × m × n × k clone allocations.
        let mut batch_scratch = Vec::with_capacity(bd_len);
        let mut idx_buf = Vec::with_capacity(bd_len + 2);

        for batch_idx in 0..batch_size {
            decode_batch_index_into(batch_idx, &dims.batch_dims, &mut batch_scratch)?;

            for i in 0..dims.m {
                for j in 0..dims.n {
                    // SOUNDNESS (#vnncomp-aw-soundness): accumulate the interval
                    // dot product in f64 and directed-round only at the final
                    // f64→f32 store, mirroring
                    // `bounds/batched/interval.rs::batched_interval_matvec_finite`.
                    // Products are formed in f64 with EXACT f32→f64 widening
                    // (f32×f32 fits in 48 < 53 significand bits, so each product
                    // is exact), so the ONLY rounding is the f64 accumulation
                    // (which keeps the running sum the true real value up to f64
                    // precision) plus the single directed-rounded final cast,
                    // which `next_down_f32`/`next_up_f32` cover OUTWARD.
                    let mut sum_lower = 0.0_f64;
                    let mut sum_upper = 0.0_f64;

                    for l in 0..dims.k {
                        // Get A[..., i, l]
                        idx_buf.clear();
                        idx_buf.extend_from_slice(&batch_scratch);
                        idx_buf.push(i);
                        idx_buf.push(l);
                        let a_l = input_a.lower()[idx_buf.as_slice()];
                        let a_u = input_a.upper()[idx_buf.as_slice()];

                        // Get B[..., l, j] or B[..., j, l] if transposed
                        idx_buf.clear();
                        idx_buf.extend_from_slice(&batch_scratch);
                        if self.transpose_b {
                            idx_buf.push(j);
                            idx_buf.push(l);
                        } else {
                            idx_buf.push(l);
                            idx_buf.push(j);
                        }
                        let b_l = input_b.lower()[idx_buf.as_slice()];
                        let b_u = input_b.upper()[idx_buf.as_slice()];

                        // Interval multiplication: [a_l, a_u] * [b_l, b_u].
                        // EXACT f32→f64 widening before multiplying, so each f64
                        // product equals the true real product (no rounding); the
                        // accumulation then rounds only in f64.
                        let p1 = (a_l as f64) * (b_l as f64);
                        let p2 = (a_l as f64) * (b_u as f64);
                        let p3 = (a_u as f64) * (b_l as f64);
                        let p4 = (a_u as f64) * (b_u as f64);

                        // NaN-safe min/max: if ANY product is NaN (e.g., 0*inf),
                        // conservatively widen to [-inf, +inf].
                        let (prod_min, prod_max) =
                            if p1.is_nan() || p2.is_nan() || p3.is_nan() || p4.is_nan() {
                                (f64::NEG_INFINITY, f64::INFINITY)
                            } else {
                                (p1.min(p2).min(p3.min(p4)), p1.max(p2).max(p3.max(p4)))
                            };

                        // NaN-safe accumulation: inf + -inf -> conservatively widen.
                        let new_lower = sum_lower + prod_min;
                        let new_upper = sum_upper + prod_max;
                        sum_lower = if new_lower.is_nan() {
                            f64::NEG_INFINITY
                        } else {
                            new_lower
                        };
                        sum_upper = if new_upper.is_nan() {
                            f64::INFINITY
                        } else {
                            new_upper
                        };
                    }

                    // Apply optional scaling (NaN-safe). Scale is exact-widened to
                    // f64; the f64 multiply is the last arithmetic step before the
                    // directed-rounded store below.
                    if let Some(scale) = self.scale {
                        let scale = scale as f64;
                        if scale >= 0.0 {
                            sum_lower *= scale;
                            sum_upper *= scale;
                        } else {
                            let tmp = sum_lower;
                            sum_lower = sum_upper * scale;
                            sum_upper = tmp * scale;
                        }
                        if sum_lower.is_nan() {
                            sum_lower = f64::NEG_INFINITY;
                        }
                        if sum_upper.is_nan() {
                            sum_upper = f64::INFINITY;
                        }
                    }

                    // Directed-round OUTWARD at the f64→f32 store: `next_down_f32`
                    // for the lower endpoint, `next_up_f32` for the upper, so the
                    // stored f32 interval encloses the true real interval despite
                    // the f64→f32 cast (both pass ±inf through unchanged).
                    // Non-finite endpoints stored as-is; repaired at type boundary (#3423).
                    idx_buf.clear();
                    idx_buf.extend_from_slice(&batch_scratch);
                    idx_buf.push(i);
                    idx_buf.push(j);
                    out_lower[idx_buf.as_slice()] = next_down_f32(sum_lower as f32);
                    out_upper[idx_buf.as_slice()] = next_up_f32(sum_upper as f32);
                }
            }
        }

        // Repair NaN/Inf at the type boundary instead of per-element (#3423).
        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }
}
