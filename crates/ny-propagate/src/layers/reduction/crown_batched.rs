// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN backward pass for reduction operations (ReduceMean, ReduceSum,
//! ReduceMax, ReduceMin).
//!
//! Implements N-D batched CROWN propagation for the common case of single
//! last-axis reduction with `keepdims=true`. This covers the primary use case:
//! LayerNorm, RMSNorm, and InstanceNorm1d all reduce over the hidden/feature
//! dimension (last axis) in their ONNX decomposition.
//!
//! Part of #3221 (CROWN tensor bounds through reduction ops degenerate).

use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::BoundedTensor;

use super::{ReduceMaxLayer, ReduceMeanLayer, ReduceMinLayer, ReduceSumLayer};
use crate::{contiguous_flat_slice, contiguous_flat_slice_mut, BatchedLinearBounds};

/// Shared batched CROWN backward pass for reduction operations.
///
/// Expands batched linear bound coefficients from the reduced output dimensions
/// back to the original input dimensions, applying a per-element `scale` factor.
///
/// **Restriction:** Only supports single last-axis reduction with `keepdims=true`.
///
/// For last-axis reduction with keepdims=true:
/// - Input: [..., N], Output: [..., 1]
/// - Forward: y[..., 0] = scale * sum(x[..., k] for k in 0..N)
/// - Backward: broadcast A[..., out_dim, 1] → A[..., out_dim, N], scale by `scale`
///
/// Non-last-axis reductions cannot be correctly represented in the batched CROWN
/// framework because batch dimensions are treated as independent positions, but
/// reductions across batch positions violate this independence assumption.
///
/// Reference:
/// `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/reduce.py:167-184`
pub(super) fn reduce_backward_batched(
    bounds: &BatchedLinearBounds,
    pre_activation: &BoundedTensor,
    axes: &[usize],
    keepdims: bool,
    scale: f32,
) -> Result<BatchedLinearBounds> {
    let input_shape = pre_activation.shape();
    let ndim = input_shape.len();

    // Validate: single last-axis reduction with keepdims=true.
    // Multi-axis and non-last-axis reductions require cross-batch-position
    // aggregation which the batched CROWN framework does not support.
    if axes.len() != 1 || axes[0] != ndim - 1 {
        return Err(NyError::UnsupportedOp(format!(
            "Batched CROWN for reduction requires single last-axis reduction, \
             got axes={axes:?} for {ndim}D input"
        )));
    }
    if !keepdims {
        return Err(NyError::UnsupportedOp(
            "Batched CROWN for reduction requires keepdims=true".to_string(),
        ));
    }

    let in_dim = input_shape[ndim - 1]; // N (original feature dim)

    // Validate the incoming bounds match the reduced output shape.
    // With last-axis reduction and keepdims, the output's last dim is 1.
    let a_shape = bounds.lower_a().shape();
    let current_in_dim = a_shape[a_shape.len() - 1];
    if current_in_dim != 1 {
        return Err(NyError::ShapeMismatch {
            expected: vec![1],
            got: vec![current_in_dim],
        });
    }

    // Build target coefficient shape: expand last dim from 1 to in_dim.
    // A shape: [...batch, out_dim, 1] → [...batch, out_dim, in_dim]
    //
    // Safety: a_shape is guaranteed non-empty (BatchedLinearBounds requires ndim >= 2).
    let mut target_a_shape: Vec<usize> = a_shape.to_vec();
    let last = target_a_shape
        .last_mut()
        .ok_or_else(|| NyError::InvalidSpec("reduce_backward_batched: empty A shape".into()))?;
    *last = in_dim;

    // Broadcast and scale. ndarray broadcasts size-1 dims to target size.
    // The broadcast produces a view; .mapv() materializes it with scaling.
    let new_lower_a: ArrayD<f32> = bounds
        .lower_a()
        .broadcast(IxDyn(&target_a_shape))
        .ok_or_else(|| NyError::ShapeMismatch {
            expected: target_a_shape.clone(),
            got: a_shape.to_vec(),
        })?
        .mapv(|v| v * scale);

    let new_upper_a: ArrayD<f32> = bounds
        .upper_a()
        .broadcast(IxDyn(&target_a_shape))
        .ok_or_else(|| NyError::ShapeMismatch {
            expected: target_a_shape.clone(),
            got: a_shape.to_vec(),
        })?
        .mapv(|v| v * scale);

    // Certify the f32 `coeff * scale` multiply error for ReduceMean (scale = 1/n); ReduceSum
    // (scale == 1.0) multiplies exactly. err = gamma_2 * |scaled coeff|, OUTWARD.
    // (#vnncomp-aw-soundness self-audit — matches the scalar reduce_backward fix.)
    let coeff_err = if scale != 1.0 {
        let gamma2 = crate::layers::linear::crown_single_gamma_n_f32(2);
        let le = new_lower_a.mapv(|c| ny_tensor::next_up_f32((gamma2 * (c as f64).abs()) as f32));
        let ue = new_upper_a.mapv(|c| ny_tensor::next_up_f32((gamma2 * (c as f64).abs()) as f32));
        Some((le, ue))
    } else {
        None
    };

    // Bias unchanged — reduction scaling affects only the coefficient matrix.
    // (The constant term in the linear bound is independent of the Jacobian.)
    let mut out = BatchedLinearBounds::new_or_conservative(
        new_lower_a,
        bounds.lower_b().clone(),
        new_upper_a,
        bounds.upper_b().clone(),
        input_shape.to_vec(),
        bounds.output_shape().to_vec(),
    )?;
    if let Some((le, ue)) = coeff_err {
        out.set_coeff_err(le, ue);
    }
    Ok(out)
}

impl ReduceMeanLayer {
    /// Batched CROWN backward propagation through ReduceMean layer.
    ///
    /// Delegates to [`reduce_backward_batched`] with `scale = 1/n`.
    /// Only supports single last-axis reduction with keepdims=true.
    ///
    /// Reference:
    /// `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/reduce.py:167-184`
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        let axes = self.resolve_axes(pre_activation.shape().len())?;
        let reduction_count = axes
            .iter()
            .map(|&a| pre_activation.shape()[a])
            .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "ReduceMean batched CROWN: reduction axes product overflows: {:?}",
                    axes
                ))
            })?;
        if reduction_count == 0 {
            return Err(NyError::InvalidSpec(
                "ReduceMean: reduction over zero-sized axes".to_string(),
            ));
        }
        let scale = 1.0 / (reduction_count as f32);
        reduce_backward_batched(bounds, pre_activation, &axes, self.keepdims, scale)
    }
}

impl ReduceSumLayer {
    /// Batched CROWN backward propagation through ReduceSum layer.
    ///
    /// Delegates to [`reduce_backward_batched`] with `scale = 1.0`.
    /// Only supports single last-axis reduction with keepdims=true.
    ///
    /// Reference:
    /// `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/reduce.py:207-227`
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        let axes = self.resolve_axes(pre_activation.shape().len())?;
        reduce_backward_batched(bounds, pre_activation, &axes, self.keepdims, 1.0)
    }
}

/// Batched CROWN backward for ReduceMax/ReduceMin using fixed argext index.
///
/// Unlike `reduce_backward_batched` (which broadcasts to all positions),
/// this scatters each output coefficient to the single argmax/argmin position
/// at the center point. The result is a sparse selection per batch position.
///
/// **Restriction:** Single last-axis reduction with `keepdims=true`.
///
/// For last-axis reduction with keepdims=true:
/// - Input: [..., N], Output: [..., 1]
/// - A shape: [...batch, out_dim, 1] → [...batch, out_dim, N]
/// - At each batch position, only A[..., out_dim, argmax_idx] is non-zero.
///
/// Reference: `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/reduce.py:40-93`
fn reduce_extremum_backward_batched(
    bounds: &BatchedLinearBounds,
    pre_activation: &BoundedTensor,
    axes: &[usize],
    keepdims: bool,
    use_argmax: bool,
) -> Result<BatchedLinearBounds> {
    let input_shape = pre_activation.shape();
    let ndim = input_shape.len();

    if axes.len() != 1 || axes[0] != ndim - 1 {
        return Err(NyError::UnsupportedOp(format!(
            "Batched CROWN for ReduceMax/Min requires single last-axis reduction, \
             got axes={axes:?} for {ndim}D input"
        )));
    }
    if !keepdims {
        return Err(NyError::UnsupportedOp(
            "Batched CROWN for ReduceMax/Min requires keepdims=true".to_string(),
        ));
    }

    let in_dim = input_shape[ndim - 1]; // N (feature dim)

    let a_shape = bounds.lower_a().shape();
    let current_in_dim = a_shape[a_shape.len() - 1];
    if current_in_dim != 1 {
        return Err(NyError::ShapeMismatch {
            expected: vec![1],
            got: vec![current_in_dim],
        });
    }

    // Target A shape: [...batch, out_dim, in_dim]
    let mut target_a_shape: Vec<usize> = a_shape.to_vec();
    let last = target_a_shape.last_mut().ok_or_else(|| {
        NyError::InvalidSpec("reduce_extremum_backward_batched: empty A shape".into())
    })?;
    *last = in_dim;

    // A has shape [...batch_dims, out_dim, in_dim].
    // The batch dims in A correspond to the leading dims of input_shape (all except last).
    // We need to find argmax/argmin at each batch position.
    let center = pre_activation.center();

    // Compute argmax/argmin index at each batch position.
    // center shape: [..., N] where ... = batch dims matching A's leading dims.
    // For each batch position, find which of the N elements is max/min.
    let batch_shape = &input_shape[..ndim - 1]; // all dims except last
    let batch_count: usize = checked_shape_product(batch_shape).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "reduce_extremum_backward_batched: batch shape {batch_shape:?} overflow usize",
        ))
    })?;

    // Flatten center to [batch_count, in_dim] for easy iteration
    let center_flat = center.to_shape((batch_count, in_dim)).map_err(|_| {
        NyError::InvalidSpec(format!(
            "reduce_extremum_backward_batched: cannot reshape center {:?} to [{}, {}]",
            center.shape(),
            batch_count,
            in_dim
        ))
    })?;

    // Compute argmax/argmin for each batch position
    let argext_indices: Vec<usize> = (0..batch_count)
        .map(|b| {
            let row = center_flat.row(b);
            if use_argmax {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| crate::cmp_utils::nan_least_cmp(a, b))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            } else {
                row.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| crate::cmp_utils::nan_propagating_cmp(a, b))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            }
        })
        .collect();

    // Build output A tensors: zeros with coefficients scattered to argext positions.
    // A shape: [...batch, out_dim, in_dim]
    // The batch dims in A may differ from batch_count because A has extra out_dim.
    // A layout: [b0, b1, ..., out_dim, in_dim] where b0*b1*... = batch_count.
    let a_ndim = a_shape.len();
    let out_dim = a_shape[a_ndim - 2]; // second-to-last dim

    let mut new_lower_a = ArrayD::<f32>::zeros(IxDyn(&target_a_shape));
    let mut new_upper_a = ArrayD::<f32>::zeros(IxDyn(&target_a_shape));

    // Flatten A to [batch_count, out_dim, in_dim] for easy indexing
    let flat_size = batch_count * out_dim * in_dim;
    let new_lower_flat = contiguous_flat_slice_mut(&mut new_lower_a)?;
    let new_upper_flat = contiguous_flat_slice_mut(&mut new_upper_a)?;

    debug_assert_eq!(new_lower_flat.len(), flat_size);
    debug_assert_eq!(new_upper_flat.len(), flat_size);

    // Source A is [...batch, out_dim, 1], flatten to [batch_count * out_dim]
    let src_lower = contiguous_flat_slice(bounds.lower_a());
    let src_upper = contiguous_flat_slice(bounds.upper_a());

    for (b, &argext_idx) in argext_indices.iter().enumerate() {
        for o in 0..out_dim {
            let src_idx = b * out_dim + o; // source flat index (in_dim=1 so no last dim stride)
            let dst_offset = (b * out_dim + o) * in_dim + argext_idx;
            new_lower_flat[dst_offset] = src_lower[src_idx];
            new_upper_flat[dst_offset] = src_upper[src_idx];
        }
    }

    BatchedLinearBounds::new_or_conservative(
        new_lower_a,
        bounds.lower_b().clone(),
        new_upper_a,
        bounds.upper_b().clone(),
        input_shape.to_vec(),
        bounds.output_shape().to_vec(),
    )
}

impl ReduceMaxLayer {
    /// Batched CROWN backward propagation through ReduceMax layer.
    ///
    /// Uses fixed_max_index assumption: scatters coefficients to argmax positions.
    /// Only supports single last-axis reduction with keepdims=true.
    ///
    /// Reference: `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/reduce.py:40-93`
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        if !self.fixed_max_index {
            return Err(NyError::UnsupportedOp(
                "ReduceMax batched CROWN requires fixed_max_index=true".to_string(),
            ));
        }
        let axes = self.resolve_axes(pre_activation.shape().len())?;
        reduce_extremum_backward_batched(bounds, pre_activation, &axes, self.keepdims, true)
    }
}

impl ReduceMinLayer {
    /// Batched CROWN backward propagation through ReduceMin layer.
    ///
    /// Uses fixed_min_index assumption: scatters coefficients to argmin positions.
    /// Only supports single last-axis reduction with keepdims=true.
    ///
    /// Reference: `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/reduce.py:40-93`
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        if !self.fixed_min_index {
            return Err(NyError::UnsupportedOp(
                "ReduceMin batched CROWN requires fixed_min_index=true".to_string(),
            ));
        }
        let axes = self.resolve_axes(pre_activation.shape().len())?;
        reduce_extremum_backward_batched(bounds, pre_activation, &axes, self.keepdims, false)
    }
}
