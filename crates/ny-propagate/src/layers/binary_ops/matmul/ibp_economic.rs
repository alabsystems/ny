// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Economic IBP propagation for MatMul (memory-efficient, potentially looser).

use ndarray::{Array2, ArrayD, IxDyn};
use ny_tensor::{BoundedTensor, RepairStrategy};

use super::helpers::{apply_scale, view_batch_2d};
use super::shape::{decode_batch_index_into, MatMulDims};
use super::{MatMulLayer, NyError, Result};
use crate::bounds::{nan_propagating_max_zero, nan_propagating_min_zero};

impl MatMulLayer {
    // Justification: Economic IBP for two-perturbed-input matmul requires both input bound
    // tensors, parsed dimensions struct, and output shape — these represent distinct
    // mathematical/structural parameters of the batched matmul computation.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn propagate_ibp_economic(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
        dims: &MatMulDims,
        out_shape: &[usize],
    ) -> Result<BoundedTensor> {
        let mut out_lower = ArrayD::zeros(IxDyn(out_shape));
        let mut out_upper = ArrayD::zeros(IxDyn(out_shape));

        let batch_size = dims.batch_size()?;

        // Reusable batch index buffer — eliminates batch_size clone allocations.
        let mut batch_scratch = Vec::with_capacity(dims.batch_dims.len());
        let mut idx_buf = Vec::with_capacity(dims.batch_dims.len() + 2);

        for batch_idx in 0..batch_size {
            decode_batch_index_into(batch_idx, &dims.batch_dims, &mut batch_scratch)?;

            let (a_lower, a_upper) =
                view_batch_2d(input_a, &batch_scratch, "MatMul IBP economic A")?;
            let (b_lower, b_upper) =
                view_batch_2d(input_b, &batch_scratch, "MatMul IBP economic B")?;

            if a_lower.nrows() != dims.m || a_lower.ncols() != dims.k {
                return Err(NyError::shape_mismatch(
                    vec![dims.m, dims.k],
                    vec![a_lower.nrows(), a_lower.ncols()],
                ));
            }

            let (b_rows, b_cols) = if self.transpose_b {
                (dims.n, dims.k)
            } else {
                (dims.k, dims.n)
            };
            if b_lower.nrows() != b_rows || b_lower.ncols() != b_cols {
                return Err(NyError::shape_mismatch(
                    vec![b_rows, b_cols],
                    vec![b_lower.nrows(), b_lower.ncols()],
                ));
            }

            let x_l = a_lower.to_owned();
            let x_u = a_upper.to_owned();
            let (y_l, y_u) = if self.transpose_b {
                (b_lower.t().to_owned(), b_upper.t().to_owned())
            } else {
                (b_lower.to_owned(), b_upper.to_owned())
            };

            // NaN-propagating max(0) so NaN bounds poison rather than silently vanish (#2432).
            let dx = (&x_u - &x_l).mapv(nan_propagating_max_zero);
            let dy = (&y_u - &y_l).mapv(nan_propagating_max_zero);
            let base = x_l.dot(&y_l);

            let mask_xp = x_l.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
            let mask_xn = x_u.mapv(|v| if v < 0.0 { 1.0 } else { 0.0 });
            let mask_xpn = Array2::<f32>::from_elem(x_l.raw_dim(), 1.0) - &mask_xp - &mask_xn;

            let mask_yp = y_l.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
            let mask_yn = y_u.mapv(|v| if v < 0.0 { 1.0 } else { 0.0 });
            let mask_ypn = Array2::<f32>::from_elem(y_l.raw_dim(), 1.0) - &mask_yp - &mask_yn;

            let mut lower = base.clone();
            let mut upper = base;

            let y_l_neg = y_l.mapv(nan_propagating_min_zero);
            let y_l_pos = y_l.mapv(nan_propagating_max_zero);
            lower += &dx.dot(&y_l_neg);
            upper += &dx.dot(&y_l_pos);

            let dx_xn = &dx * &mask_xn;
            let dx_xp = &dx * &mask_xp;
            let y_l_ypn = &y_l * &mask_ypn;
            lower -= &dx_xn.dot(&y_l_ypn);
            upper += &dx_xp.dot(&y_l_ypn);

            let x_l_neg = x_l.mapv(nan_propagating_min_zero);
            let x_l_pos = x_l.mapv(nan_propagating_max_zero);
            lower += &x_l_neg.dot(&dy);
            upper += &x_l_pos.dot(&dy);

            let x_l_xpn = &x_l * &mask_xpn;
            let dy_yn = &dy * &mask_yn;
            let dy_yp = &dy * &mask_yp;
            lower -= &x_l_xpn.dot(&dy_yn);
            upper += &x_l_xpn.dot(&dy_yp);

            lower += &dx_xn.dot(&dy_yn);
            let dx_xpn_xp = &dx * (&mask_xpn + &mask_xp);
            let dy_ypn_yp = &dy * (&mask_ypn + &mask_yp);
            upper += &dx_xpn_xp.dot(&dy_ypn_yp);

            apply_scale(&mut lower, &mut upper, self.scale);

            for i in 0..dims.m {
                for j in 0..dims.n {
                    idx_buf.clear();
                    idx_buf.extend_from_slice(&batch_scratch);
                    idx_buf.push(i);
                    idx_buf.push(j);
                    out_lower[idx_buf.as_slice()] = lower[[i, j]];
                    out_upper[idx_buf.as_slice()] = upper[[i, j]];
                }
            }
        }

        // Repair NaN/Inf at the type boundary instead of per-batch (#3423).
        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }
}
