// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_up_f32, BoundedTensor};

use super::cumsum::CumsumLayer;
use crate::BatchedLinearBounds;

impl CumsumLayer {
    /// Batched CROWN backward propagation through CumSum for grouped last-axis
    /// layouts.
    ///
    /// This is the grouped analogue of `propagate_linear_with_bounds()`: apply
    /// the same suffix/prefix scan independently to every grouped coefficient
    /// row in `[..., out_dim, in_dim]`.
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        let input_shape = pre_activation.shape();
        if input_shape.is_empty() {
            return Err(NyError::InvalidSpec(
                "CumSum batched CROWN requires rank >= 1".to_string(),
            ));
        }
        if input_shape.contains(&0) {
            return Err(NyError::InvalidSpec(
                "CumSum batched CROWN: input has zero-sized dimension".to_string(),
            ));
        }

        let ndim = input_shape.len();
        let axis = self.resolve_axis(ndim)?;
        if axis != ndim - 1 {
            return Err(NyError::UnsupportedOp(format!(
                "CumSum batched CROWN requires last-axis grouped layout, \
                 got axis {axis} for input shape {input_shape:?}"
            )));
        }

        if bounds.input_shape() != input_shape || bounds.output_shape() != input_shape {
            return Err(NyError::UnsupportedOp(format!(
                "CumSum batched CROWN requires shape-preserving grouped bounds \
                 matching {input_shape:?}, got input {:?}, output {:?}",
                bounds.input_shape(),
                bounds.output_shape()
            )));
        }

        let axis_len = input_shape[ndim - 1];
        let mut expected_a_shape = input_shape[..ndim - 1].to_vec();
        expected_a_shape.push(axis_len);
        expected_a_shape.push(axis_len);
        if bounds.lower_a().shape() != expected_a_shape.as_slice() {
            return Err(NyError::UnsupportedOp(format!(
                "CumSum batched CROWN requires coefficient shape {:?}, got {:?}",
                expected_a_shape,
                bounds.lower_a().shape()
            )));
        }

        // Scan the coefficients AND emit a certified coefficient error per stored
        // partial-sum coeff (the batched analogue of the non-batched cumsum.rs fix): the
        // f32 prefix/suffix accumulation can drop a unit coefficient under cancellation
        // (e.g. suffix-sum of [-2^24, 2^24, 1] stores 0 at col0 while the true coeff is 1),
        // and BatchedLinearBounds::new_or_conservative trusts the coeffs as EXACT (err=None),
        // so the verdict box would exclude a reachable output. (#vnncomp-aw-soundness.)
        let (new_lower_a, lower_err) = self.scan_grouped_coefficients(
            bounds.lower_a(),
            bounds.lower_a_err.as_ref(),
            axis_len,
        )?;
        let (new_upper_a, upper_err) = self.scan_grouped_coefficients(
            bounds.upper_a(),
            bounds.upper_a_err.as_ref(),
            axis_len,
        )?;

        let mut out = BatchedLinearBounds::new_or_conservative(
            new_lower_a,
            bounds.lower_b().clone(),
            new_upper_a,
            bounds.upper_b().clone(),
            bounds.input_shape().to_vec(),
            bounds.output_shape().to_vec(),
        )?;
        out.set_coeff_err(lower_err, upper_err);
        Ok(out)
    }

    /// Scan (prefix/suffix sum) the coefficients in f32 AND build the certified per-coeff
    /// error `next_up(γ_axislen·S + prop)` over EXACTLY the terms folded into each stored
    /// coefficient — `S` = f64 abs-sum of the folded original coeffs (cancellation-safe),
    /// `prop` = re-summed incoming coeff err. The abs/prop are taken BEFORE folding the
    /// current term for the EXCLUSIVE variants (stored coeff excludes it) and AFTER for the
    /// INCLUSIVE variants, matching the non-batched cumsum.rs scan exactly.
    fn scan_grouped_coefficients(
        &self,
        coefficients: &ArrayD<f32>,
        incoming_err: Option<&ArrayD<f32>>,
        axis_len: usize,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
        let original_shape = coefficients.shape().to_vec();
        let outer = checked_shape_product(&original_shape[..original_shape.len() - 1]).ok_or_else(
            || {
                NyError::InvalidSpec(format!(
                    "CumSum batched CROWN: coefficient rows overflow for shape {original_shape:?}"
                ))
            },
        )?;
        let mut reshaped = coefficients
            .clone()
            .into_shape_with_order((outer, axis_len))
            .map_err(|e| {
                NyError::InvalidSpec(format!("CumSum batched CROWN: reshape to 2D failed: {e}"))
            })?;
        let in_err =
            match incoming_err {
                Some(e) => Some(e.clone().into_shape_with_order((outer, axis_len)).map_err(
                    |e| {
                        NyError::InvalidSpec(format!(
                            "CumSum batched CROWN: err reshape to 2D failed: {e}"
                        ))
                    },
                )?),
                None => None,
            };
        let gamma = crate::layers::linear::crown_single_gamma_n_f32(axis_len);
        let mut err = ndarray::Array2::<f32>::zeros((outer, axis_len));

        for row in 0..outer {
            let mut abss = 0.0f64;
            let mut prop = 0.0f64;
            let in_at = |col: usize| in_err.as_ref().map_or(0.0, |e| e[[row, col]] as f64);
            if self.reverse {
                if self.exclusive {
                    let mut acc = 0.0f32;
                    for col in 0..axis_len {
                        let original = reshaped[[row, col]];
                        reshaped[[row, col]] = acc;
                        err[[row, col]] = next_up_f32((gamma * abss + prop) as f32);
                        acc += original;
                        abss += (original as f64).abs();
                        prop += in_at(col);
                    }
                } else {
                    let mut acc = 0.0f32;
                    for col in 0..axis_len {
                        let original = reshaped[[row, col]];
                        acc += original;
                        abss += (original as f64).abs();
                        prop += in_at(col);
                        reshaped[[row, col]] = acc;
                        err[[row, col]] = next_up_f32((gamma * abss + prop) as f32);
                    }
                }
            } else if self.exclusive {
                let mut acc = 0.0f32;
                for col in (0..axis_len).rev() {
                    let original = reshaped[[row, col]];
                    reshaped[[row, col]] = acc;
                    err[[row, col]] = next_up_f32((gamma * abss + prop) as f32);
                    acc += original;
                    abss += (original as f64).abs();
                    prop += in_at(col);
                }
            } else {
                let mut acc = 0.0f32;
                for col in (0..axis_len).rev() {
                    let original = reshaped[[row, col]];
                    acc += original;
                    abss += (original as f64).abs();
                    prop += in_at(col);
                    reshaped[[row, col]] = acc;
                    err[[row, col]] = next_up_f32((gamma * abss + prop) as f32);
                }
            }
        }

        let scanned = reshaped
            .into_shape_with_order(IxDyn(&original_shape))
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "CumSum batched CROWN: reshape back to {original_shape:?} failed: {e}"
                ))
            })?;
        let err_d = err
            .into_dyn()
            .into_shape_with_order(IxDyn(&original_shape))
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "CumSum batched CROWN: err reshape back to {original_shape:?} failed: {e}"
                ))
            })?;
        Ok((scanned, err_d))
    }
}
