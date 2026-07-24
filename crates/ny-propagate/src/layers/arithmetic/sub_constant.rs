// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Subtract constant layer: y = x - c or y = c - x (element-wise).

use ndarray::{Array1, ArrayD, Axis, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use std::borrow::Cow;
use tracing::debug;

use super::common::{dot_bias_f64, extract_scalar_constant_for_batched, scalar_row_sum_f64};
use super::validate::validate_finite_array;
use crate::layers::common::BoundPropagation;
use crate::{BatchedLinearBounds, LinearBounds};

/// Subtract constant layer: y = x - c or y = c - x (element-wise).
///
/// Used in LayerNorm for mean subtraction.
#[derive(Debug, Clone)]
pub struct SubConstantLayer {
    /// The constant tensor.
    pub(crate) constant: ArrayD<f32>,
    /// If true: y = constant - x, if false: y = x - constant
    pub reverse: bool,
}

impl SubConstantLayer {
    /// Validate and create a new layer for y = x - constant.
    pub fn try_new(constant: ArrayD<f32>) -> Result<Self> {
        validate_finite_array(&constant, "SubConstantLayer", "constant")?;
        Ok(Self {
            constant,
            reverse: false,
        })
    }

    /// Create a new layer for y = x - constant.
    pub fn new(constant: ArrayD<f32>) -> Self {
        Self::try_new(constant).expect("invariant: SubConstantLayer::new requires finite constant")
    }

    /// Validate and create a new layer for y = constant - x.
    pub fn try_new_reverse(constant: ArrayD<f32>) -> Result<Self> {
        let mut layer = Self::try_new(constant)?;
        layer.reverse = true;
        Ok(layer)
    }

    /// Create a new layer for y = constant - x.
    pub fn new_reverse(constant: ArrayD<f32>) -> Self {
        Self::try_new_reverse(constant)
            .expect("invariant: SubConstantLayer::new_reverse requires finite constant")
    }

    /// Create a scalar subtraction layer (y = x - scalar).
    pub fn scalar(value: f32) -> Self {
        Self::new(ArrayD::from_elem(IxDyn(&[]), value))
    }

    /// Return the constant tensor.
    pub fn constant(&self) -> &ArrayD<f32> {
        &self.constant
    }
}

impl BoundPropagation for SubConstantLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let input_shape = input.shape();
        let const_shape = self.constant.shape();

        // Compute broadcast output shape (ONNX semantics).
        // Handles both constant→input and input→constant broadcast.
        let output_shape =
            crate::shape::broadcast_shapes(input_shape, const_shape).ok_or_else(|| {
                NyError::ShapeMismatch {
                    expected: input_shape.to_vec(),
                    got: const_shape.to_vec(),
                }
            })?;

        // Broadcast constant and input bounds to output shape.
        let c = self
            .constant
            .broadcast(IxDyn(&output_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: output_shape.clone(),
                got: const_shape.to_vec(),
            })?;
        let lower_in = input
            .lower()
            .broadcast(IxDyn(&output_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: output_shape.clone(),
                got: input_shape.to_vec(),
            })?;
        let upper_in = input
            .upper()
            .broadcast(IxDyn(&output_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: output_shape.clone(),
                got: input_shape.to_vec(),
            })?;

        // Directed rounding (#vnncomp-softmax-complex): plain `propagate_ibp`
        // is a production node-bound source, so the subtraction must round
        // OUTWARD. f64 subtraction of two f32 values is exact; only the
        // f64→f32 cast rounds, so next_down/next_up on the cast guarantees
        // enclosure of the real-arithmetic interval.
        let (out_lower, out_upper) = if self.reverse {
            // y = c - x: when x is large, y is small and vice versa
            // y_lower = c - x_upper
            // y_upper = c - x_lower
            let lower = ndarray::Zip::from(&c)
                .and(&upper_in)
                .map_collect(|&ci, &ui| next_down_f32((ci as f64 - ui as f64) as f32));
            let upper = ndarray::Zip::from(&c)
                .and(&lower_in)
                .map_collect(|&ci, &li| next_up_f32((ci as f64 - li as f64) as f32));
            (lower, upper)
        } else {
            // y = x - c: subtraction preserves order
            // y_lower = x_lower - c
            // y_upper = x_upper - c
            let lower = ndarray::Zip::from(&lower_in)
                .and(&c)
                .map_collect(|&li, &ci| next_down_f32((li as f64 - ci as f64) as f32));
            let upper = ndarray::Zip::from(&upper_in)
                .and(&c)
                .map_collect(|&ui, &ci| next_up_f32((ui as f64 - ci as f64) as f32));
            (lower, upper)
        };

        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // For y = x - c: just shift the bias by -c
        // For y = c - x: negate coefficients and adjust bias

        // Flatten constant to 1D and convert to Array1 for compatibility with LinearBounds
        // bounds.lower_b is Array1<f32>, so we need c as Array1<f32> for subtraction
        let c_flat: Array1<f32> = self
            .constant
            .clone()
            .into_shape_with_order((self.constant.len(),))
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![self.constant.len()],
                got: self.constant.shape().to_vec(),
            })?
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![self.constant.len()],
                got: self.constant.shape().to_vec(),
            })?;

        // Debug shapes
        debug!(
            "SubConstant propagate_linear: constant shape {:?} (len {}), lower_b len {}, c_flat len {}",
            self.constant.shape(),
            self.constant.len(),
            bounds.lower_b().len(),
            c_flat.len()
        );

        // CROWN backward propagation for SubConstant.
        //
        // For y = x - c (where y is output, x is input):
        // - Before: lA @ y + lb <= output (where y is this layer's output)
        // - Substitute y = x - c: lA @ (x - c) + lb = lA @ x + (lb - lA @ c)
        // - After: new_lA = lA, new_lb = lb - lA @ c
        //
        // The bias adjustment requires lA @ c, NOT lb - c!
        // lA has shape (num_outputs, layer_dim), c has shape (layer_dim,)
        // lA @ c gives shape (num_outputs,) which matches lb
        //
        // Special case: scalar constant (broadcasts to input shape)
        // If c is scalar, c_flat has len=1 but layer_dim > 1
        // In this case, lA @ c_broadcast = c * sum(lA, axis=1)
        // Compute A @ c with f64 accumulation + directed rounding (#3157, #1863).
        let layer_dim = bounds.lower_a().ncols();
        let (lower_c_f64, upper_c_f64) = if c_flat.len() == 1 && layer_dim > 1 {
            scalar_row_sum_f64(bounds.lower_a(), bounds.upper_a(), c_flat[0])
        } else if c_flat.len() == layer_dim {
            dot_bias_f64(bounds.lower_a(), bounds.upper_a(), &c_flat)
        } else if c_flat.len() > layer_dim {
            // Broadcast expansion: constant is larger than input. CROWN backward
            // through broadcast expansion requires column aggregation, not yet
            // implemented — fall back to IBP at this layer boundary.
            return Err(NyError::UnsupportedOp(format!(
                "SubConstant CROWN backward: broadcast expansion \
                 (constant {} elements > layer dim {}) not supported",
                c_flat.len(),
                layer_dim
            )));
        } else {
            return Err(NyError::ShapeMismatch {
                expected: vec![layer_dim],
                got: vec![c_flat.len()],
            });
        };

        if self.reverse {
            // y = c - x: CROWN backward substitution (no swap).
            // Before: lA @ y + lb, uA @ y + ub
            // Substitute y = c - x: lA @ (c - x) + lb = -lA @ x + (lb + lA @ c)
            // No swap: CROWN composes by substitution, not IBP.
            // Reference: designs/2026-01-29-crown-affine-negative-scale.md
            let lb_f64 = bounds.lower_b().mapv(|x| x as f64) + &lower_c_f64;
            let ub_f64 = bounds.upper_b().mapv(|x| x as f64) + &upper_c_f64;
            let new_lower_b = lb_f64.mapv(|x| next_down_f32(x as f32));
            let new_upper_b = ub_f64.mapv(|x| next_up_f32(x as f32));
            Ok(Cow::Owned(LinearBounds::new_or_conservative(
                -bounds.lower_a(),
                new_lower_b,
                -bounds.upper_a(),
                new_upper_b,
            )?))
        } else {
            // y = x - c: new_lb = old_lb - lA @ c, new_ub = old_ub - uA @ c
            let lb_f64 = bounds.lower_b().mapv(|x| x as f64) - &lower_c_f64;
            let ub_f64 = bounds.upper_b().mapv(|x| x as f64) - &upper_c_f64;
            let new_lower_b = lb_f64.mapv(|x| next_down_f32(x as f32));
            let new_upper_b = ub_f64.mapv(|x| next_up_f32(x as f32));
            Ok(Cow::Owned(LinearBounds::new_or_conservative(
                bounds.lower_a().clone(),
                new_lower_b,
                bounds.upper_a().clone(),
                new_upper_b,
            )?))
        }
    }
}

impl SubConstantLayer {
    /// Batched CROWN backward propagation through SubConstant.
    ///
    /// For y = x - c: coefficient matrices unchanged, bias shifts by -A @ c
    /// For y = c - x: coefficient matrices negated, bias shifts by +A @ c
    ///
    /// Currently only supports scalar constants in batched mode.
    #[inline]
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        // Compute the bias contribution: A @ c
        // For scalar c: A @ c = c * sum(A, axis=-1)
        // For non-scalar c: Not yet supported in batched mode
        let c_val = extract_scalar_constant_for_batched(&self.constant, "SubConstant")?;
        debug!("SubConstant batched CROWN: scalar c = {}", c_val);

        // f64 accumulation + directed rounding to match non-batched path (#2423).
        // Previous f32 sum_axis lost precision in large reductions.
        let ndim = bounds.lower_a.ndim();
        let c_f64 = c_val as f64;

        // Row sums in f64 for precision.
        let lower_sum_f64 = bounds.lower_a.mapv(|v| v as f64).sum_axis(Axis(ndim - 1));
        let upper_sum_f64 = bounds.upper_a.mapv(|v| v as f64).sum_axis(Axis(ndim - 1));

        // Bias contributions: c * sum(A_row) in f64.
        let lower_contrib_f64 = lower_sum_f64.mapv(|v| v * c_f64);
        let upper_contrib_f64 = upper_sum_f64.mapv(|v| v * c_f64);

        if self.reverse {
            // y = c - x: CROWN backward substitution (no swap).
            // Substitute y = c - x: lA @ (c - x) + lb = -lA @ x + (lb + lA @ c)
            // Reference: designs/2026-01-29-crown-affine-negative-scale.md
            debug!("SubConstant batched CROWN: reverse mode (c - x)");
            // Add bias + contribution in f64, directed rounding, NaN → conservative.
            let lb_f64 = bounds.lower_b.mapv(|v| v as f64) + &lower_contrib_f64;
            let ub_f64 = bounds.upper_b.mapv(|v| v as f64) + &upper_contrib_f64;

            let new_lower_b = lb_f64.mapv(|v| {
                let f = v as f32;
                if f.is_nan() {
                    f32::NEG_INFINITY
                } else {
                    next_down_f32(f)
                }
            });
            let new_upper_b = ub_f64.mapv(|v| {
                let f = v as f32;
                if f.is_nan() {
                    f32::INFINITY
                } else {
                    next_up_f32(f)
                }
            });

            BatchedLinearBounds::new_or_conservative(
                bounds.lower_a.mapv(|v| -v),
                new_lower_b,
                bounds.upper_a.mapv(|v| -v),
                new_upper_b,
                bounds.input_shape.clone(),
                bounds.output_shape.clone(),
            )
        } else {
            // y = x - c: CROWN backward substitution.
            // Substitute y = x - c: lA @ (x - c) + lb = lA @ x + (lb - lA @ c)
            debug!("SubConstant batched CROWN: standard mode (x - c)");
            // Subtract bias contribution in f64, directed rounding, NaN → conservative.
            let lb_f64 = bounds.lower_b.mapv(|v| v as f64) - &lower_contrib_f64;
            let ub_f64 = bounds.upper_b.mapv(|v| v as f64) - &upper_contrib_f64;

            let new_lower_b = lb_f64.mapv(|v| {
                let f = v as f32;
                if f.is_nan() {
                    f32::NEG_INFINITY
                } else {
                    next_down_f32(f)
                }
            });
            let new_upper_b = ub_f64.mapv(|v| {
                let f = v as f32;
                if f.is_nan() {
                    f32::INFINITY
                } else {
                    next_up_f32(f)
                }
            });

            BatchedLinearBounds::new_or_conservative(
                bounds.lower_a.clone(),
                new_lower_b,
                bounds.upper_a.clone(),
                new_upper_b,
                bounds.input_shape.clone(),
                bounds.output_shape.clone(),
            )
        }
    }
}
