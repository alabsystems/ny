// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Reduction layers for bound propagation.

use ndarray::{Array2, Axis, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{
    cast_f64_to_f32_down, cast_f64_to_f32_up, next_up_f32, BoundedTensor, RepairStrategy,
};
use std::borrow::Cow;

use super::common::{compute_strides, BoundPropagation};
use crate::LinearBounds;

mod certified_mean;
mod crown_batched;
mod cumsum;
mod cumsum_batched;
mod extremum;
mod selection;

pub use cumsum::CumsumLayer;
pub use extremum::{ReduceMaxLayer, ReduceMinLayer};
pub use selection::{ArgMaxLayer, ArgMinLayer, ArgSortLayer, TopkLayer, TopkOutputKind};

/// Resolve negative axis indices to positive ones with bounds validation.
///
/// Shared implementation for multi-axis reduction layers. Delegates per-axis
/// validation to `common::resolve_axis`. Returns `Err` if any axis is out of range.
fn resolve_reduction_axes(axes: &[i64], ndim: usize, layer_name: &str) -> Result<Vec<usize>> {
    if axes.is_empty() {
        return Ok((0..ndim).collect());
    }
    let resolved: Vec<usize> = axes
        .iter()
        .map(|&axis| super::common::resolve_axis(axis, ndim, layer_name))
        .collect::<Result<Vec<_>>>()?;

    // Reject duplicate axes — double-reduction produces wrong bounds or panics
    // when `output_shape.remove(axis)` shifts indices. (#2946)
    let mut seen = std::collections::HashSet::new();
    for &axis in &resolved {
        if !seen.insert(axis) {
            return Err(NyError::InvalidSpec(format!(
                "{layer_name}: duplicate reduction axis {axis} after resolution"
            )));
        }
    }
    Ok(resolved)
}

/// Fold the extremum (min or max) of the terms feeding each reduced output
/// element, mirroring the keepdims/reshape flow of the f64 accumulator loops
/// in `ReduceMean`/`ReduceSum` IBP so shapes stay aligned.
///
/// This is the sign witness for the outward directed cast: when every LOWER
/// term of an output element is >= 0, the true sum/mean there is >= 0, so a
/// downward-cast result may be clamped at zero without losing soundness.
/// Without the clamp, an exactly-zero lower bound (sum of squares, variance)
/// is pushed one denormal below zero, which spuriously triggers
/// sqrt-negative-domain handling and fails `>= 0` output specs downstream.
/// Symmetrically for UPPER terms that are all <= 0.
fn fold_extremum_over_axes(
    values: &ndarray::ArrayD<f32>,
    sorted_axes_desc: &[usize],
    keepdims: bool,
    minimum: bool,
) -> Result<ndarray::ArrayD<f32>> {
    let identity = if minimum {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    };
    let mut acc = values.clone();
    for &axis in sorted_axes_desc {
        let folded = acc.fold_axis(Axis(axis), identity, |&a, &b| {
            if minimum {
                a.min(b)
            } else {
                a.max(b)
            }
        });
        if keepdims {
            let folded_shape = folded.shape().to_vec();
            let mut new_shape = folded_shape.clone();
            new_shape.insert(axis, 1);
            acc = folded
                .into_shape_with_order(IxDyn(&new_shape))
                .map_err(|_| NyError::ShapeMismatch {
                    expected: new_shape,
                    got: folded_shape,
                })?;
        } else {
            acc = folded;
        }
    }
    Ok(acc)
}

/// Directed-cast the f64 reduction accumulators outward to f32, preserving
/// sign bounds certified by [`fold_extremum_over_axes`] witnesses. NaN values
/// pass through untouched (the `stepped < 0.0` / `stepped > 0.0` guards are
/// false for NaN) so the centralized repair still sees them.
fn directed_cast_with_sign_witness(
    lower: &ndarray::ArrayD<f64>,
    upper: &ndarray::ArrayD<f64>,
    lower_min: &ndarray::ArrayD<f32>,
    upper_max: &ndarray::ArrayD<f32>,
) -> (ndarray::ArrayD<f32>, ndarray::ArrayD<f32>) {
    let lower = ndarray::Zip::from(lower)
        .and(lower_min)
        .map_collect(|&x, &witness| {
            let stepped = cast_f64_to_f32_down(x);
            if witness >= 0.0 && stepped < 0.0 {
                0.0
            } else {
                stepped
            }
        });
    let upper = ndarray::Zip::from(upper)
        .and(upper_max)
        .map_collect(|&x, &witness| {
            let stepped = cast_f64_to_f32_up(x);
            if witness <= 0.0 && stepped > 0.0 {
                0.0
            } else {
                stepped
            }
        });
    (lower, upper)
}

/// Reduce `values` along `axis` by a CERTIFIED mean: each output element is the
/// endpoint of a rigorous enclosure of the exact arithmetic mean of the terms
/// that fed it (see [`certified_mean::certified_mean_enclosure`]).
///
/// `take_lower` selects which endpoint is kept. The mean is monotone in every
/// argument, so folding the lower array with `take_lower = true` and the upper
/// array with `take_lower = false` composes correctly across successive axes:
/// each intermediate stays a valid endpoint of the running interval.
fn certified_mean_axis(
    values: &ndarray::ArrayD<f64>,
    axis: Axis,
    take_lower: bool,
) -> ndarray::ArrayD<f64> {
    values.map_axis(axis, |lane| {
        let (lo, hi) = certified_mean::certified_mean_enclosure(lane.iter().copied());
        if take_lower {
            lo
        } else {
            hi
        }
    })
}

/// Shared CROWN backward pass for reduction operations (ReduceMean and ReduceSum).
///
/// Expands linear bound coefficients from the reduced output dimensions back to the
/// original input dimensions, applying a per-element `scale` factor.
///
/// Math:
/// - Forward: `y[j] = scale * sum(x[k] for k in reduction_set_j)`
/// - Jacobian: `J[j,k] = scale` if k contributes to `y[j]`, else 0
/// - Backward: `new_A = A @ J` (expands columns, each scaled by `scale`)
///
/// For ReduceMean, `scale = 1/n` where n is the product of reduced axis sizes.
/// For ReduceSum, `scale = 1.0`.
///
/// Reference:
/// `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/reduce.py:167-227`
fn reduce_backward(
    bounds: &LinearBounds,
    pre_activation: &BoundedTensor,
    axes: &[usize],
    keepdims: bool,
    scale: f32,
) -> Result<LinearBounds> {
    let input_shape = pre_activation.shape();
    let input_len = checked_shape_product(input_shape).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "reduce_backward: input shape product overflows: {:?}",
            input_shape
        ))
    })?;
    let ndim = input_shape.len();

    // Guard: zero-sized dimension in input causes zero strides, which panics
    // in the flat-index decomposition below (`remaining / stride` where stride=0).
    // Part of #2816.
    if input_shape.contains(&0) {
        return Err(NyError::InvalidSpec(
            "reduce_backward: input has zero-sized dimension".to_string(),
        ));
    }

    // Compute output shape after reduction
    let mut output_shape: Vec<usize> = input_shape.to_vec();
    for &axis in axes {
        if keepdims {
            output_shape[axis] = 1;
        }
    }

    // For !keepdims, remove the axes (in descending order to avoid index shift)
    if !keepdims {
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_by(|a, b| b.cmp(a));
        for &axis in &sorted_axes {
            output_shape.remove(axis);
        }
    }

    let output_len = checked_shape_product(&output_shape).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "reduce_backward: output shape product overflows: {:?}",
            output_shape
        ))
    })?;

    // Verify dimensions match
    if bounds.num_inputs() != output_len {
        return Err(NyError::ShapeMismatch {
            expected: vec![output_len],
            got: vec![bounds.num_inputs()],
        });
    }

    let num_outputs = bounds.num_outputs();

    // Build index mapping: for each input index, find corresponding output index
    let input_strides = compute_strides(input_shape)?;
    let output_strides = compute_strides(&output_shape)?;

    // Create new coefficient matrices (expanded from output to input dimensions)
    let mut new_lower_a = Array2::<f32>::zeros((num_outputs, input_len));
    let mut new_upper_a = Array2::<f32>::zeros((num_outputs, input_len));

    // The `coeff * scale` multiply rounds in f32 for ReduceMean (scale = 1/n); certify it.
    // (ReduceSum scale == 1.0 multiplies exactly, so no error is emitted.)
    let gamma2 = crate::layers::linear::crown_single_gamma_n_f32(2);
    let scale_abs = (scale as f64).abs();
    let emit_err = scale != 1.0;
    let mut lower_err = Array2::<f32>::zeros((num_outputs, input_len));
    let mut upper_err = Array2::<f32>::zeros((num_outputs, input_len));

    // Map each input position to its output position
    for input_idx in 0..input_len {
        // Convert flat index to multi-dimensional index
        let mut coords = vec![0usize; ndim];
        let mut remaining = input_idx;
        for d in 0..ndim {
            coords[d] = remaining / input_strides[d];
            remaining %= input_strides[d];
        }

        // Compute output coordinates (reduce axes to 0 or remove)
        let output_coords: Vec<usize> = if keepdims {
            coords
                .iter()
                .enumerate()
                .map(|(d, &c)| if axes.contains(&d) { 0 } else { c })
                .collect()
        } else {
            coords
                .iter()
                .enumerate()
                .filter(|(d, _)| !axes.contains(d))
                .map(|(_, &c)| c)
                .collect()
        };

        // Convert output coordinates to flat index
        let output_idx: usize = output_coords
            .iter()
            .zip(output_strides.iter())
            .map(|(&c, &s)| c * s)
            .sum();

        // Copy coefficients with scaling
        for row in 0..num_outputs {
            new_lower_a[[row, input_idx]] = bounds.lower_a()[[row, output_idx]] * scale;
            new_upper_a[[row, input_idx]] = bounds.upper_a()[[row, output_idx]] * scale;
            if emit_err {
                // Certified error of the `coeff * scale` f32 multiply (#vnncomp-aw-soundness):
                // for ReduceMean scale = fl(1/n), so stored = fl(coeff * fl(1/n)) differs from
                // the true coeff * (1/n) by up to ~1.5u·|coeff·scale|; gamma_2 covers the
                // precompute + the multiply. ReduceSum (scale == 1.0) multiplies exactly, so
                // gamma is 0 there. Rounded OUTWARD.
                let g = if scale == 1.0 { 0.0 } else { gamma2 };
                let cl = (bounds.lower_a()[[row, output_idx]] as f64).abs() * scale_abs;
                let cu = (bounds.upper_a()[[row, output_idx]] as f64).abs() * scale_abs;
                lower_err[[row, input_idx]] = next_up_f32((g * cl) as f32);
                upper_err[[row, input_idx]] = next_up_f32((g * cu) as f32);
            }
        }
    }

    if emit_err {
        return LinearBounds::new_or_conservative_with_err(
            new_lower_a,
            bounds.lower_b().clone(),
            new_upper_a,
            bounds.upper_b().clone(),
            lower_err,
            upper_err,
        );
    }

    // Bias remains unchanged
    LinearBounds::new_or_conservative(
        new_lower_a,
        bounds.lower_b().clone(),
        new_upper_a,
        bounds.upper_b().clone(),
    )
}

/// Reduce mean layer: computes mean over specified axes.
///
/// Used in unfused LayerNorm for computing mean(x).
#[derive(Debug, Clone)]
pub struct ReduceMeanLayer {
    /// Axes to reduce over (e.g., [-1] for last axis).
    pub axes: Vec<i64>,
    /// Whether to keep reduced dimensions (size 1) in output.
    pub keepdims: bool,
}

impl ReduceMeanLayer {
    /// Create a new reduce mean layer.
    pub fn new(axes: Vec<i64>, keepdims: bool) -> Self {
        Self { axes, keepdims }
    }

    /// Create a reduce mean layer for the last axis (common in LayerNorm).
    pub fn last_axis() -> Self {
        Self {
            axes: vec![-1],
            keepdims: true,
        }
    }

    /// Resolve negative axis indices to positive ones.
    fn resolve_axes(&self, ndim: usize) -> Result<Vec<usize>> {
        resolve_reduction_axes(&self.axes, ndim, "ReduceMean")
    }
}

impl BoundPropagation for ReduceMeanLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Mean is a linear operation: mean(x) = sum(x) / n
        // For bounded inputs:
        // mean_lower = sum(lower) / n = mean(lower)
        // mean_upper = sum(upper) / n = mean(upper)

        let ndim = input.lower().ndim();
        let axes = self.resolve_axes(ndim)?;

        // Accumulate the reduction in f64, then directed-cast OUTWARD to f32. The f32
        // `mean_axis` is round-to-NEAREST over axis_len terms, so it can produce a box that
        // EXCLUDES the true value under cancellation/absorption (e.g. mean over [2^24, 1]
        // gives 2^24 < true 2^24+1) — unsound as a node bound feeding ReLU stability.
        // (#vnncomp-aw-soundness self-audit; mirrors linear/ibp.rs.)
        //
        // The f64 accumulation CERTIFIES ITSELF: `certified_mean_axis` charges the exact
        // TwoSum residual and the exact fma division remainder rather than assuming the
        // subsequent f32 ULP step covers them. It reports zero width whenever the reduction
        // was exact — every size-1 axis, which ONNX emits as an identity — so the cast below
        // is the only place a bound can move, and it moves only when it must.
        let mut lower = input.lower().mapv(f64::from);
        let mut upper = input.upper().mapv(f64::from);

        // Sort axes in descending order to avoid index shifting issues
        let mut sorted_axes = axes;
        sorted_axes.sort_by(|a, b| b.cmp(a));

        for &axis in &sorted_axes {
            // Compute mean along this axis (in f64)
            let axis_obj = Axis(axis);

            if lower.len_of(axis_obj) == 0 {
                return Err(NyError::ShapeMismatch {
                    expected: vec![],
                    got: lower.shape().to_vec(),
                });
            }

            let new_lower = certified_mean_axis(&lower, axis_obj, true);
            let new_upper = certified_mean_axis(&upper, axis_obj, false);

            if self.keepdims {
                // Insert a dimension of size 1 at the reduced axis
                let mut new_shape: Vec<usize> = new_lower.shape().to_vec();
                let lower_shape = new_lower.shape().to_vec();
                let upper_shape = new_upper.shape().to_vec();
                new_shape.insert(axis, 1);
                lower = new_lower
                    .into_shape_with_order(IxDyn(&new_shape))
                    .map_err(|_| NyError::ShapeMismatch {
                        expected: new_shape.clone(),
                        got: lower_shape,
                    })?;
                upper = new_upper
                    .into_shape_with_order(IxDyn(&new_shape))
                    .map_err(|_| NyError::ShapeMismatch {
                        expected: new_shape,
                        got: upper_shape,
                    })?;
            } else {
                lower = new_lower;
                upper = new_upper;
            }
        }

        // Directed-cast the f64 accumulators OUTWARD to f32 (lower down, upper up), then the
        // centralized NaN/Inf repair (#3423). The directed cast is what makes the box ENCLOSE
        // the true value past the f32 grid (#vnncomp-aw-soundness self-audit). Sign witnesses
        // keep provably sign-bounded reductions (mean of non-negatives, e.g. variances) from
        // being stepped across zero — see `directed_cast_with_sign_witness`.
        let lower_min = fold_extremum_over_axes(input.lower(), &sorted_axes, self.keepdims, true)?;
        let upper_max = fold_extremum_over_axes(input.upper(), &sorted_axes, self.keepdims, false)?;
        let (lower, upper) =
            directed_cast_with_sign_witness(&lower, &upper, &lower_min, &upper_max);
        BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // ReduceMean backward needs pre-activation shape to expand reduced dimensions.
        Err(NyError::UnsupportedOp(
            "ReduceMean linear propagation requires pre-activation bounds - use propagate_crown_backward"
                .to_string(),
        ))
    }

    fn requires_pre_activation_bounds(&self) -> bool {
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        ReduceMeanLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

impl ReduceMeanLayer {
    /// CROWN backward propagation through ReduceMean layer.
    ///
    /// Delegates to [`reduce_backward`] with `scale = 1/n`.
    ///
    /// Reference implementation:
    /// `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/reduce.py:167-184`
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        let axes = self.resolve_axes(pre_activation.shape().len())?;
        let reduction_count = axes
            .iter()
            .map(|&a| pre_activation.shape()[a])
            .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "ReduceMean CROWN: reduction axes product overflows: {:?}",
                    axes
                ))
            })?;
        // Guard: zero reduction count produces Inf scale. (#2816)
        if reduction_count == 0 {
            return Err(NyError::InvalidSpec(
                "ReduceMean: reduction over zero-sized axes".to_string(),
            ));
        }
        let scale = 1.0 / (reduction_count as f32);
        reduce_backward(bounds, pre_activation, &axes, self.keepdims, scale)
    }
}

/// Reduce sum layer: computes sum over specified axes.
///
/// Used in various reduction patterns and unfused operations.
#[derive(Debug, Clone)]
pub struct ReduceSumLayer {
    /// Axes to reduce over (e.g., [-1] for last axis).
    pub axes: Vec<i64>,
    /// Whether to keep reduced dimensions (size 1) in output.
    pub keepdims: bool,
}

impl ReduceSumLayer {
    /// Create a new reduce sum layer.
    pub fn new(axes: Vec<i64>, keepdims: bool) -> Self {
        Self { axes, keepdims }
    }

    /// Create a reduce sum layer for the last axis.
    pub fn last_axis() -> Self {
        Self {
            axes: vec![-1],
            keepdims: true,
        }
    }

    /// Resolve negative axis indices to positive ones.
    fn resolve_axes(&self, ndim: usize) -> Result<Vec<usize>> {
        resolve_reduction_axes(&self.axes, ndim, "ReduceSum")
    }
}

impl BoundPropagation for ReduceSumLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Sum is a linear operation: sum(x) = sum(x)
        // For bounded inputs:
        // sum_lower = sum(lower)
        // sum_upper = sum(upper)

        let ndim = input.lower().ndim();
        let axes = self.resolve_axes(ndim)?;

        // Accumulate in f64 then directed-cast OUTWARD: the f32 `sum_axis` is round-to-NEAREST
        // over axis_len terms and can EXCLUDE the true value under cancellation/absorption
        // (e.g. sum over [2^24, 1] = 2^24 < true 2^24+1) — unsound as a node bound.
        // (#vnncomp-aw-soundness self-audit; mirrors linear/ibp.rs:157-170.)
        let mut lower = input.lower().mapv(|x| x as f64);
        let mut upper = input.upper().mapv(|x| x as f64);

        // Sort axes in descending order to avoid index shifting issues
        let mut sorted_axes = axes;
        sorted_axes.sort_by(|a, b| b.cmp(a));

        for &axis in &sorted_axes {
            // Compute sum along this axis (in f64)
            let axis_obj = Axis(axis);

            let new_lower = lower.sum_axis(axis_obj);
            let new_upper = upper.sum_axis(axis_obj);

            if self.keepdims {
                // Insert a dimension of size 1 at the reduced axis
                let lower_shape = new_lower.shape().to_vec();
                let upper_shape = new_upper.shape().to_vec();
                let mut new_shape: Vec<usize> = new_lower.shape().to_vec();
                new_shape.insert(axis, 1);
                lower = new_lower
                    .into_shape_with_order(IxDyn(&new_shape))
                    .map_err(|_| NyError::ShapeMismatch {
                        expected: new_shape.clone(),
                        got: lower_shape,
                    })?;
                upper = new_upper
                    .into_shape_with_order(IxDyn(&new_shape))
                    .map_err(|_| NyError::ShapeMismatch {
                        expected: new_shape,
                        got: upper_shape,
                    })?;
            } else {
                lower = new_lower;
                upper = new_upper;
            }
        }

        // Directed-cast the f64 accumulators OUTWARD to f32 (lower down, upper up), then the
        // centralized NaN/Inf repair (#3423). The directed cast is what makes the box ENCLOSE
        // the true value past the f32 grid (#vnncomp-aw-soundness self-audit). Sign witnesses
        // keep provably sign-bounded reductions (sums of non-negatives, e.g. sums of squares)
        // from being stepped across zero — see `directed_cast_with_sign_witness`.
        let lower_min = fold_extremum_over_axes(input.lower(), &sorted_axes, self.keepdims, true)?;
        let upper_max = fold_extremum_over_axes(input.upper(), &sorted_axes, self.keepdims, false)?;
        let (lower, upper) =
            directed_cast_with_sign_witness(&lower, &upper, &lower_min, &upper_max);
        BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // ReduceSum backward needs pre-activation shape to expand reduced dimensions.
        Err(NyError::UnsupportedOp(
            "ReduceSum linear propagation requires pre-activation bounds - use propagate_crown_backward"
                .to_string(),
        ))
    }

    fn requires_pre_activation_bounds(&self) -> bool {
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        ReduceSumLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

impl ReduceSumLayer {
    /// CROWN backward propagation through ReduceSum layer.
    ///
    /// Delegates to [`reduce_backward`] with `scale = 1.0`.
    ///
    /// Reference implementation:
    /// `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/reduce.py:207-227`
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        let axes = self.resolve_axes(pre_activation.shape().len())?;
        reduce_backward(bounds, pre_activation, &axes, self.keepdims, 1.0)
    }
}

// ReduceMax and ReduceMin live in extremum.rs (split for file size limit).

#[cfg(test)]
mod cumsum_tests;
#[cfg(test)]
mod tests;
