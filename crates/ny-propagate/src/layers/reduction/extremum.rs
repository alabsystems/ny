// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ReduceMax and ReduceMin layers for bound propagation.
//!
//! IBP is exact due to monotonicity. CROWN backward uses fixed index assumption
//! (argmax/argmin position at center point is assumed not to change under perturbation).
//!
//! Reference: `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/reduce.py:40-93`

use std::borrow::Cow;

use ndarray::{Array2, ArrayD, Axis, IxDyn};
use ny_core::{checked_shape_product, nan_propagating_max, nan_propagating_min, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};

use super::{compute_strides, resolve_reduction_axes};
use crate::{BoundPropagation, LinearBounds};

/// Compute max reduction along an axis, returning the reduced array.
///
/// ndarray does not have `max_axis()` like `mean_axis()` or `sum_axis()`,
/// so we use `fold_axis()` with NaN-propagating max.
///
/// Uses `nan_propagating_max` instead of `f32::max` because IEEE 754-2008
/// `maxNum` silently absorbs NaN: `NaN.max(2.0) = 2.0`. If NaN bounds leak
/// through, this would produce silently unsound (too-tight) bounds.
/// Reference: ny_core::nan_math, binary_ops/elementwise.rs (same pattern).
fn max_axis(arr: &ArrayD<f32>, axis: Axis) -> ArrayD<f32> {
    arr.fold_axis(axis, f32::NEG_INFINITY, |&acc, &x| {
        nan_propagating_max(acc, x)
    })
}

/// Compute min reduction along an axis, returning the reduced array.
///
/// Uses `nan_propagating_min` instead of `f32::min` for the same reason
/// as `max_axis`: IEEE 754 `minNum` silently absorbs NaN.
fn min_axis(arr: &ArrayD<f32>, axis: Axis) -> ArrayD<f32> {
    arr.fold_axis(axis, f32::INFINITY, |&acc, &x| nan_propagating_min(acc, x))
}

/// CROWN backward for ReduceMax/ReduceMin using fixed index assumption.
///
/// Instead of broadcasting coefficients to all reduced positions (like reduce_backward),
/// this scatters each output coefficient to the single argmax/argmin position at the
/// center point. The Jacobian is a sparse selection matrix.
///
/// Reference: `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/reduce.py:40-93`
pub(super) fn reduce_extremum_backward(
    bounds: &LinearBounds,
    pre_activation: &BoundedTensor,
    axes: &[usize],
    keepdims: bool,
    use_argmax: bool,
) -> Result<LinearBounds> {
    let input_shape = pre_activation.shape();
    let input_len = checked_shape_product(input_shape).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "reduce_extremum_backward: input shape product overflows: {:?}",
            input_shape
        ))
    })?;

    if input_shape.contains(&0) {
        return Err(NyError::InvalidSpec(
            "reduce_extremum_backward: input has zero-sized dimension".to_string(),
        ));
    }

    // Compute output shape after reduction
    let mut output_shape: Vec<usize> = input_shape.to_vec();
    for &axis in axes {
        if keepdims {
            output_shape[axis] = 1;
        }
    }
    if !keepdims {
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_by(|a, b| b.cmp(a));
        for &axis in &sorted_axes {
            output_shape.remove(axis);
        }
    }

    let output_len = checked_shape_product(&output_shape).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "reduce_extremum_backward: output shape product overflows: {:?}",
            output_shape
        ))
    })?;

    if bounds.num_inputs() != output_len {
        return Err(NyError::ShapeMismatch {
            expected: vec![output_len],
            got: vec![bounds.num_inputs()],
        });
    }

    let num_outputs = bounds.num_outputs();

    let input_strides = compute_strides(input_shape)?;
    let output_strides = compute_strides(&output_shape)?;

    // Per-output reduced groups: ALL flat input indices reduced into each output.
    // (Replaces the old single-center-argmax map: the backward must know the whole
    // group to check whether the extremum index is provably stable over the box.)
    let groups = compute_argext_groups(
        input_shape,
        &output_shape,
        axes,
        keepdims,
        &input_strides,
        &output_strides,
    )?;

    // Pre-activation interval bounds in standard (row-major) flat order, matching
    // `input_strides`, so a group flat index addresses them directly.
    let pre_lower_std = pre_activation.lower().as_standard_layout();
    let pre_upper_std = pre_activation.upper().as_standard_layout();
    let pre_lower = pre_lower_std.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("reduce_extremum_backward: non-contiguous pre-activation lower".into())
    })?;
    let pre_upper = pre_upper_std.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("reduce_extremum_backward: non-contiguous pre-activation upper".into())
    })?;

    let mut new_lower_a = Array2::<f32>::zeros((num_outputs, input_len));
    let mut new_upper_a = Array2::<f32>::zeros((num_outputs, input_len));
    // Bias accumulators in f64: the stable (definite-winner) case adds nothing; the
    // unstable case folds the sound IBP interval in. Directed-rounded at the end.
    let mut new_lower_b = bounds.lower_b().mapv(|x| x as f64);
    let mut new_upper_b = bounds.upper_b().mapv(|x| x as f64);

    for (output_idx, group) in groups.iter().enumerate() {
        if group.is_empty() {
            continue;
        }

        // Sound IBP interval for this reduction output over the whole input box:
        //   ReduceMax: y in [max_j lower_j, max_j upper_j]
        //   ReduceMin: y in [min_j lower_j, min_j upper_j]
        // NaN-propagating folds (IEEE 754-2008), matching the IBP / MaxPool2d paths.
        let (ext_lower, ext_upper) = if use_argmax {
            (
                group
                    .iter()
                    .map(|&i| pre_lower[i])
                    .fold(f32::NEG_INFINITY, nan_propagating_max),
                group
                    .iter()
                    .map(|&i| pre_upper[i])
                    .fold(f32::NEG_INFINITY, nan_propagating_max),
            )
        } else {
            (
                group
                    .iter()
                    .map(|&i| pre_lower[i])
                    .fold(f32::INFINITY, nan_propagating_min),
                group
                    .iter()
                    .map(|&i| pre_upper[i])
                    .fold(f32::INFINITY, nan_propagating_min),
            )
        };

        // Definite winner: a group member whose extremum is provably selected over
        // the ENTIRE box, so the fixed-index linear selection is exact and SOUND.
        //   ReduceMax: lower_j >= upper_k for all k != j  (j is always the max)
        //   ReduceMin: upper_j <= lower_k for all k != j  (j is always the min)
        // Mirrors pooling/max.rs `definite_winner`. Without this guard the previous
        // code scattered the center-argmax UNCONDITIONALLY into both bounds, which is
        // unsound when the extremum index moves within the box and (via tighter-wins
        // CROWN/IBP intersection) could yield a falsely-tight bound -> wrong `unsat`.
        let definite_winner = group.iter().copied().find(|&j| {
            group.iter().all(|&k| {
                k == j
                    || if use_argmax {
                        pre_lower[j] >= pre_upper[k]
                    } else {
                        pre_upper[j] <= pre_lower[k]
                    }
            })
        });

        match definite_winner {
            Some(winner) => {
                // Stable extremum: the selection Jacobian is exact; gradient flows
                // entirely through the winning input (identity-like), as before.
                for row in 0..num_outputs {
                    new_lower_a[[row, winner]] += bounds.lower_a()[[row, output_idx]];
                    new_upper_a[[row, winner]] += bounds.upper_a()[[row, output_idx]];
                }
            }
            None => {
                // Unstable extremum: no member provably dominates, so a single fixed
                // index is unsound. Cut the backward pass here and fold the sound IBP
                // interval [ext_lower, ext_upper] into the bias (zero A contribution).
                for row in 0..num_outputs {
                    let la = bounds.lower_a()[[row, output_idx]];
                    let ua = bounds.upper_a()[[row, output_idx]];
                    // Lower row: minimize la*y over y in [ext_lower, ext_upper].
                    if la > 0.0 {
                        new_lower_b[row] += la as f64 * ext_lower as f64;
                    } else if la < 0.0 {
                        new_lower_b[row] += la as f64 * ext_upper as f64;
                    }
                    // Upper row: maximize ua*y over y in [ext_lower, ext_upper].
                    if ua > 0.0 {
                        new_upper_b[row] += ua as f64 * ext_upper as f64;
                    } else if ua < 0.0 {
                        new_upper_b[row] += ua as f64 * ext_lower as f64;
                    }
                }
            }
        }
    }

    LinearBounds::new_or_conservative(
        new_lower_a,
        new_lower_b.mapv(|x| next_down_f32(x as f32)),
        new_upper_a,
        new_upper_b.mapv(|x| next_up_f32(x as f32)),
    )
}

/// Compute, for each output position, the full set of flat input indices reduced
/// into it (the reduction group).
///
/// Same geometry as the former center-argmax map, but returns EVERY group member
/// instead of one selected index, so [`reduce_extremum_backward`] can check
/// extremum stability (a definite winner) over the whole input box and fall back to
/// sound constant bounds when the argmax/argmin is not stable.
fn compute_argext_groups(
    input_shape: &[usize],
    output_shape: &[usize],
    axes: &[usize],
    keepdims: bool,
    input_strides: &[usize],
    output_strides: &[usize],
) -> Result<Vec<Vec<usize>>> {
    let ndim = input_shape.len();
    let output_len: usize = checked_shape_product(output_shape).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "compute_argext_groups: output shape {output_shape:?} overflow usize",
        ))
    })?;

    let reduction_dims: Vec<usize> = axes.iter().map(|&a| input_shape[a]).collect();
    let reduction_count: usize = checked_shape_product(&reduction_dims).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "compute_argext_groups: reduction dims {reduction_dims:?} overflow usize",
        ))
    })?;

    let mut groups = vec![Vec::new(); output_len];

    for (output_idx, group) in groups.iter_mut().enumerate() {
        // Convert output flat index to multi-dimensional output coordinates
        let mut output_coords = vec![0usize; output_shape.len()];
        let mut remaining = output_idx;
        for (d, coord) in output_coords.iter_mut().enumerate() {
            *coord = remaining / output_strides[d];
            remaining %= output_strides[d];
        }

        // Map output coords back to input coords (non-reduced dims)
        let base_input_coords: Vec<usize> = if keepdims {
            output_coords.clone()
        } else {
            // Insert 0s at the removed axis positions
            let mut coords = Vec::with_capacity(ndim);
            let mut out_d = 0;
            for d in 0..ndim {
                if axes.contains(&d) {
                    coords.push(0); // placeholder, iterated over below
                } else {
                    coords.push(output_coords[out_d]);
                    out_d += 1;
                }
            }
            coords
        };

        group.reserve(reduction_count);
        for r in 0..reduction_count {
            // Decompose r into per-axis indices
            let mut input_coords = base_input_coords.clone();
            let mut rem = r;
            for (i, &axis) in axes.iter().enumerate() {
                let stride: usize = checked_shape_product(&reduction_dims[i + 1..]).unwrap_or(0);
                let stride = if stride == 0 { 1 } else { stride };
                input_coords[axis] = rem / stride;
                rem %= stride;
            }

            // Convert to flat input index (standard row-major, matching input_strides)
            let flat_idx: usize = input_coords
                .iter()
                .zip(input_strides.iter())
                .map(|(&c, &s)| c * s)
                .sum();
            group.push(flat_idx);
        }
    }

    Ok(groups)
}

/// IBP propagation for extremum (max/min) reduction.
///
/// Factored out to avoid duplication between ReduceMaxLayer and ReduceMinLayer.
fn propagate_extremum_ibp(
    input: &BoundedTensor,
    axes: &[usize],
    keepdims: bool,
    reduce_fn: fn(&ArrayD<f32>, Axis) -> ArrayD<f32>,
) -> Result<BoundedTensor> {
    let mut lower = input.lower().clone();
    let mut upper = input.upper().clone();

    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort_by(|a, b| b.cmp(a));

    for &axis in &sorted_axes {
        let axis_obj = Axis(axis);
        let new_lower = reduce_fn(&lower, axis_obj);
        let new_upper = reduce_fn(&upper, axis_obj);

        if keepdims {
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

    // Centralized NaN/Inf repair at constructor (#3423).
    BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative)
}

/// ReduceMax layer: computes element-wise maximum over specified axes.
///
/// IBP is exact (monotonicity): max(lower) <= max(x) <= max(upper).
///
/// CROWN backward uses the fixed_max_index assumption: the argmax position
/// is assumed fixed (not perturbed), making ReduceMax a linear selection op.
/// This is standard in VNN-COMP tools (alpha-beta-CROWN, etc.).
///
/// Reference: `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/reduce.py:40-93`
#[derive(Debug, Clone)]
pub struct ReduceMaxLayer {
    /// Axes to reduce over (e.g., [-1] for last axis).
    pub axes: Vec<i64>,
    /// Whether to keep reduced dimensions (size 1) in output.
    pub keepdims: bool,
    /// When true, assume the argmax index doesn't change under perturbation.
    /// Required for CROWN backward. Default: true (matches alpha-beta-CROWN).
    pub fixed_max_index: bool,
}

impl ReduceMaxLayer {
    /// Create a new reduce max layer with fixed_max_index=true (default).
    pub fn new(axes: Vec<i64>, keepdims: bool) -> Self {
        Self {
            axes,
            keepdims,
            fixed_max_index: true,
        }
    }

    /// Create a reduce max layer for the last axis.
    pub fn last_axis() -> Self {
        Self {
            axes: vec![-1],
            keepdims: true,
            fixed_max_index: true,
        }
    }

    /// Resolve negative axis indices to positive ones.
    pub(crate) fn resolve_axes(&self, ndim: usize) -> Result<Vec<usize>> {
        resolve_reduction_axes(&self.axes, ndim, "ReduceMax")
    }
}

impl BoundPropagation for ReduceMaxLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let axes = self.resolve_axes(input.lower().ndim())?;
        propagate_extremum_ibp(input, &axes, self.keepdims, max_axis)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "ReduceMax linear propagation requires pre-activation bounds".to_string(),
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
        ReduceMaxLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

impl ReduceMaxLayer {
    /// CROWN backward propagation through ReduceMax layer.
    ///
    /// Uses fixed_max_index assumption: coefficients scatter to argmax positions
    /// at the center point. The Jacobian is J[j,k] = 1 if k = argmax(center)[j].
    ///
    /// Reference: `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/reduce.py:40-93`
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        if !self.fixed_max_index {
            return Err(NyError::UnsupportedOp(
                "ReduceMax CROWN requires fixed_max_index=true".to_string(),
            ));
        }
        let axes = self.resolve_axes(pre_activation.shape().len())?;
        reduce_extremum_backward(bounds, pre_activation, &axes, self.keepdims, true)
    }
}

/// ReduceMin layer: computes element-wise minimum over specified axes.
///
/// IBP is exact (monotonicity): min(lower) <= min(x) <= min(upper).
///
/// CROWN backward uses the fixed_min_index assumption (argmin at center point).
///
/// Reference: `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/reduce.py:40-93`
#[derive(Debug, Clone)]
pub struct ReduceMinLayer {
    /// Axes to reduce over (e.g., [-1] for last axis).
    pub axes: Vec<i64>,
    /// Whether to keep reduced dimensions (size 1) in output.
    pub keepdims: bool,
    /// When true, assume the argmin index doesn't change under perturbation.
    pub fixed_min_index: bool,
}

impl ReduceMinLayer {
    /// Create a new reduce min layer with fixed_min_index=true (default).
    pub fn new(axes: Vec<i64>, keepdims: bool) -> Self {
        Self {
            axes,
            keepdims,
            fixed_min_index: true,
        }
    }

    /// Create a reduce min layer for the last axis.
    pub fn last_axis() -> Self {
        Self {
            axes: vec![-1],
            keepdims: true,
            fixed_min_index: true,
        }
    }

    /// Resolve negative axis indices to positive ones.
    pub(crate) fn resolve_axes(&self, ndim: usize) -> Result<Vec<usize>> {
        resolve_reduction_axes(&self.axes, ndim, "ReduceMin")
    }
}

impl BoundPropagation for ReduceMinLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let axes = self.resolve_axes(input.lower().ndim())?;
        propagate_extremum_ibp(input, &axes, self.keepdims, min_axis)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "ReduceMin linear propagation requires pre-activation bounds".to_string(),
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
        ReduceMinLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

impl ReduceMinLayer {
    /// CROWN backward propagation through ReduceMin layer.
    ///
    /// Uses fixed_min_index assumption: coefficients scatter to argmin positions.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        if !self.fixed_min_index {
            return Err(NyError::UnsupportedOp(
                "ReduceMin CROWN requires fixed_min_index=true".to_string(),
            ));
        }
        let axes = self.resolve_axes(pre_activation.shape().len())?;
        reduce_extremum_backward(bounds, pre_activation, &axes, self.keepdims, false)
    }
}
