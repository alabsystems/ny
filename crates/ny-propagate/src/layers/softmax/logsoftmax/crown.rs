// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward linearization for LogSoftmax.
//!
//! Contains the sound (LSE-based) and heuristic (sampling-based) CROWN
//! backward passes, plus helper functions (eval, softmax, jacobian).

use ndarray::{Array1, Array2, ArrayD};
use ny_core::{checked_shape_product, NyError, Result, VerificationSoundnessMode};
use ny_tensor::{next_up_f32, BoundedTensor};
use tracing::debug;

use super::super::bounds::constant_bounds_from_output;
use super::super::utils;
use super::LogSoftmaxLayer;
use crate::bounds::nan_propagating_max;
use crate::LinearBounds;

impl LogSoftmaxLayer {
    pub(crate) fn resolve_axis(&self, ndim: usize) -> Result<usize> {
        crate::layers::common::resolve_axis_i32(self.axis, ndim, "LogSoftmax")
    }

    /// Evaluate logsoftmax at a point.
    ///
    /// logsoftmax(x) = x - logsumexp(x)
    pub(crate) fn eval(&self, x: &Array1<f32>) -> Array1<f32> {
        let max_val = x
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, nan_propagating_max);
        let exp_sum: f32 = x.iter().map(|&v| (v - max_val).exp()).sum();
        let logsumexp = max_val + exp_sum.ln();
        x.mapv(|v| v - logsumexp)
    }

    /// Compute softmax at a point.
    pub(crate) fn softmax(&self, x: &Array1<f32>) -> Array1<f32> {
        let max_val = x
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, nan_propagating_max);
        let exp_vals: Array1<f32> = x.mapv(|v| (v - max_val).exp());
        let sum: f32 = exp_vals.sum();
        exp_vals / sum
    }

    /// Compute Jacobian of logsoftmax at a point.
    ///
    /// J[i,j] = δ_ij - softmax[j]
    ///
    /// The Jacobian is: I - 1 * softmax^T
    pub(crate) fn jacobian(&self, x: &Array1<f32>) -> Array2<f32> {
        let n = x.len();
        let s = self.softmax(x);

        let mut j = Array2::<f32>::eye(n);
        for i in 0..n {
            for k in 0..n {
                j[[i, k]] -= s[k];
            }
        }
        j
    }

    /// CROWN backward propagation with pre-activation bounds.
    ///
    /// LogSoftmax has global dependencies through the logsumexp term.
    /// We use a Jacobian-based linear approximation at the interval center
    /// with sampling to estimate the approximation error (heuristic).
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        soundness: VerificationSoundnessMode,
    ) -> Result<LinearBounds> {
        debug!("LogSoftmax layer CROWN backward propagation with pre-activation bounds");
        let shape = pre_activation.shape();
        if shape.is_empty() {
            return Err(NyError::InvalidSpec(
                "LogSoftmax requires at least 1D input".to_string(),
            ));
        }
        let ndim = shape.len();
        let axis = self.resolve_axis(ndim)?;

        let effective_soundness = if self.sound {
            if soundness == VerificationSoundnessMode::Heuristic {
                debug!("LogSoftmax heuristic requested, but layer is in sound mode; using IBP constant bounds");
            }
            VerificationSoundnessMode::Sound
        } else {
            soundness
        };

        if effective_soundness == VerificationSoundnessMode::Sound {
            debug!("LogSoftmax sound mode: using LSE-based affine bounds");
            return self.propagate_linear_with_bounds_sound(bounds, pre_activation, axis);
        }
        debug!("LogSoftmax heuristic mode: sampling-based bounds (not sound)");
        self.propagate_linear_with_bounds_heuristic(bounds, pre_activation, axis)
    }

    fn propagate_linear_with_bounds_sound(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        axis: usize,
    ) -> Result<LinearBounds> {
        let shape = pre_activation.shape();
        let ndim = shape.len();
        let total_size: usize = checked_shape_product(shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "LogSoftmax CROWN: shape product overflows usize: {:?}",
                shape,
            ))
        })?;
        if bounds.num_inputs() != total_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![total_size],
                got: vec![bounds.num_inputs()],
            });
        }

        let has_non_finite = pre_activation.lower().iter().any(|&v| !v.is_finite())
            || pre_activation.upper().iter().any(|&v| !v.is_finite());
        if has_non_finite {
            let lower = ArrayD::from_elem(pre_activation.lower().raw_dim(), f32::NEG_INFINITY);
            let upper = ArrayD::from_elem(pre_activation.upper().raw_dim(), f32::INFINITY);
            let fallback = BoundedTensor::new_allow_infinite(lower, upper)?;
            return constant_bounds_from_output(bounds, &fallback);
        }

        let mut lower_a = Array2::<f32>::zeros((total_size, total_size));
        let mut upper_a = Array2::<f32>::zeros((total_size, total_size));
        let mut lower_b = Array1::<f32>::zeros(total_size);
        let mut upper_b = Array1::<f32>::zeros(total_size);

        let softmax_size = shape[axis];
        let num_outputs = bounds.num_outputs();

        // Compute strides for converting between flat and multi-dimensional indices.
        let mut strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        // Number of groups = product of all dimensions except axis
        let non_axis_dims: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, &d)| d)
            .collect();
        let num_groups: usize = checked_shape_product(&non_axis_dims).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "LogSoftmax CROWN: non-axis dimensions {non_axis_dims:?} overflow usize",
            ))
        })?;

        if num_groups == 0 {
            // Empty non-axis dimension means there are no independent logsoftmax groups.
            return LinearBounds::new_or_conservative(
                Array2::<f32>::zeros((num_outputs, total_size)),
                bounds.lower_b().clone(),
                Array2::<f32>::zeros((num_outputs, total_size)),
                bounds.upper_b().clone(),
            );
        }

        // Helper: convert multi-dim index to flat index
        let multi_to_flat =
            |idx: &[usize]| -> usize { idx.iter().zip(strides.iter()).map(|(&i, &s)| i * s).sum() };

        // Compute "group shape" (all dims except axis)
        let group_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, &d)| d)
            .collect();

        // Stack-allocated index buffers — ndim is always small (2-5).
        assert!(ndim <= 8, "logsoftmax N-D indexing assumes ndim <= 8");
        let mut group_strides = [1usize; 8];
        if !group_shape.is_empty() {
            for d in (0..group_shape.len() - 1).rev() {
                group_strides[d] = group_strides[d + 1] * group_shape[d + 1];
            }
        }

        let mut group_multi = [0usize; 8];
        let mut full_idx = [0usize; 8];
        for group_idx in 0..num_groups {
            // Convert group_idx to indices on non-axis dimensions
            let mut remaining = group_idx;
            for d in 0..group_shape.len() {
                group_multi[d] = remaining / group_strides[d];
                remaining %= group_strides[d];
            }

            let mut group_lower = Array1::<f32>::zeros(softmax_size);
            let mut group_upper = Array1::<f32>::zeros(softmax_size);
            let mut flat_indices_for_group = Vec::with_capacity(softmax_size);
            let mut gm_pos = 0;
            for d in 0..ndim {
                if d == axis {
                    full_idx[d] = 0;
                } else {
                    full_idx[d] = group_multi[gm_pos];
                    gm_pos += 1;
                }
            }

            for s in 0..softmax_size {
                full_idx[axis] = s;
                let flat = multi_to_flat(&full_idx[..ndim]);
                flat_indices_for_group.push(flat);
                group_lower[s] = pre_activation.lower()[&full_idx[..ndim]];
                group_upper[s] = pre_activation.upper()[&full_idx[..ndim]];
            }

            // Bit-identical linearization center: f32::midpoint rounds differently at overflow edges.
            #[allow(clippy::manual_midpoint)]
            let x_center: Array1<f32> = group_lower
                .iter()
                .zip(group_upper.iter())
                .map(|(&l, &u)| (l + u) / 2.0)
                .collect();

            // Directed rounding: lse_upper feeds lower_b = -lse_upper, so understating
            // lse_upper overstates -lse_upper, making the lower bound unsound.
            // Match softmax sound path pattern (linear/sound.rs:88, #3275 Gap 1).
            let lse_upper = next_up_f32(utils::logsumexp_1d(&group_upper));
            let lse_center = utils::logsumexp_1d(&x_center);
            if !lse_upper.is_finite() || !lse_center.is_finite() {
                let lower = ArrayD::from_elem(pre_activation.lower().raw_dim(), f32::NEG_INFINITY);
                let upper = ArrayD::from_elem(pre_activation.upper().raw_dim(), f32::INFINITY);
                let fallback = BoundedTensor::new_allow_infinite(lower, upper)?;
                return constant_bounds_from_output(bounds, &fallback);
            }

            let softmax_center = utils::softmax_1d(&x_center);
            if softmax_center.iter().any(|&v| !v.is_finite()) {
                let lower = ArrayD::from_elem(pre_activation.lower().raw_dim(), f32::NEG_INFINITY);
                let upper = ArrayD::from_elem(pre_activation.upper().raw_dim(), f32::INFINITY);
                let fallback = BoundedTensor::new_allow_infinite(lower, upper)?;
                return constant_bounds_from_output(bounds, &fallback);
            }
            let lse_lower_b = lse_center - softmax_center.dot(&x_center);

            for (local_i, &global_i) in flat_indices_for_group.iter().enumerate() {
                lower_a[[global_i, global_i]] = 1.0;
                lower_b[global_i] = -lse_upper;

                for (local_k, &global_k) in flat_indices_for_group.iter().enumerate() {
                    upper_a[[global_i, global_k]] =
                        (if local_k == local_i { 1.0 } else { 0.0 }) - softmax_center[local_k];
                }
                upper_b[global_i] = -lse_lower_b;
            }
        }

        self.apply_affine_bounds(bounds, &lower_a, &lower_b, &upper_a, &upper_b)
    }

    fn propagate_linear_with_bounds_heuristic(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        axis: usize,
    ) -> Result<LinearBounds> {
        let shape = pre_activation.shape();
        let ndim = shape.len();
        let total_size: usize = checked_shape_product(shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "LogSoftmax CROWN: shape product overflows usize: {:?}",
                shape,
            ))
        })?;
        if bounds.num_inputs() != total_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![total_size],
                got: vec![bounds.num_inputs()],
            });
        }

        // Fall back to constant [-inf, +inf] bounds for non-finite pre-activations (#2591).
        // Returning bounds.clone() (identity passthrough) was unsound because
        // LogSoftmax is not the identity function. Constant bounds are trivially
        // sound and match the sound path's fallback (lines 127-134).
        if pre_activation.lower().iter().any(|&v| !v.is_finite())
            || pre_activation.upper().iter().any(|&v| !v.is_finite())
        {
            debug!("LogSoftmax heuristic CROWN: non-finite pre-activation, falling back to constant bounds");
            let lower = ArrayD::from_elem(pre_activation.lower().raw_dim(), f32::NEG_INFINITY);
            let upper = ArrayD::from_elem(pre_activation.upper().raw_dim(), f32::INFINITY);
            let fallback = BoundedTensor::new_allow_infinite(lower, upper)?;
            return constant_bounds_from_output(bounds, &fallback);
        }

        let mut lower_a = Array2::<f32>::zeros((total_size, total_size));
        let mut upper_a = Array2::<f32>::zeros((total_size, total_size));
        let mut lower_b = Array1::<f32>::zeros(total_size);
        let mut upper_b = Array1::<f32>::zeros(total_size);

        let softmax_size = shape[axis];
        let num_outputs = bounds.num_outputs();

        // Compute strides for converting between flat and multi-dimensional indices.
        let mut strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        // Number of groups = product of all dimensions except axis
        let non_axis_dims: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, &d)| d)
            .collect();
        let num_groups: usize = checked_shape_product(&non_axis_dims).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "LogSoftmax CROWN: non-axis dimensions {non_axis_dims:?} overflow usize",
            ))
        })?;

        if num_groups == 0 {
            // Empty non-axis dimension means there are no independent logsoftmax groups.
            return LinearBounds::new_or_conservative(
                Array2::<f32>::zeros((num_outputs, total_size)),
                bounds.lower_b().clone(),
                Array2::<f32>::zeros((num_outputs, total_size)),
                bounds.upper_b().clone(),
            );
        }

        // Helper: convert multi-dim index to flat index
        let multi_to_flat =
            |idx: &[usize]| -> usize { idx.iter().zip(strides.iter()).map(|(&i, &s)| i * s).sum() };

        // Compute "group shape" (all dims except axis)
        let group_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, &d)| d)
            .collect();

        // Stack-allocated index buffers — ndim is always small (2-5).
        assert!(ndim <= 8, "logsoftmax N-D indexing assumes ndim <= 8");
        let mut group_strides = [1usize; 8];
        if !group_shape.is_empty() {
            for d in (0..group_shape.len() - 1).rev() {
                group_strides[d] = group_strides[d + 1] * group_shape[d + 1];
            }
        }

        let mut group_multi = [0usize; 8];
        let mut full_idx = [0usize; 8];
        for group_idx in 0..num_groups {
            // Convert group_idx to indices on non-axis dimensions
            let mut remaining = group_idx;
            for d in 0..group_shape.len() {
                group_multi[d] = remaining / group_strides[d];
                remaining %= group_strides[d];
            }

            let mut group_lower = Array1::<f32>::zeros(softmax_size);
            let mut group_upper = Array1::<f32>::zeros(softmax_size);
            let mut flat_indices_for_group = Vec::with_capacity(softmax_size);
            let mut gm_pos = 0;
            for d in 0..ndim {
                if d == axis {
                    full_idx[d] = 0;
                } else {
                    full_idx[d] = group_multi[gm_pos];
                    gm_pos += 1;
                }
            }

            for s in 0..softmax_size {
                full_idx[axis] = s;
                let flat = multi_to_flat(&full_idx[..ndim]);
                flat_indices_for_group.push(flat);
                group_lower[s] = pre_activation.lower()[&full_idx[..ndim]];
                group_upper[s] = pre_activation.upper()[&full_idx[..ndim]];
            }

            // Compute center point and evaluate
            // Bit-identical linearization center: f32::midpoint rounds differently at overflow edges.
            #[allow(clippy::manual_midpoint)]
            let x_center: Array1<f32> = group_lower
                .iter()
                .zip(group_upper.iter())
                .map(|(&l, &u)| (l + u) / 2.0)
                .collect();

            let y_center = self.eval(&x_center);
            let jacobian = self.jacobian(&x_center);

            // Linear approximation: y ≈ J @ x + (y_c - J @ x_c)
            let jx_center = jacobian.dot(&x_center);
            let b_approx: Array1<f32> = &y_center - &jx_center;

            // Sample to find max error from linear approximation
            let num_samples = 50;
            let mut max_error_above: Array1<f32> = Array1::zeros(softmax_size);
            let mut max_error_below: Array1<f32> = Array1::zeros(softmax_size);

            let mut x_sample = x_center.clone();

            // Sample random points in the hypercube
            for sample_idx in 0..num_samples {
                x_sample.assign(&x_center);
                for i in 0..softmax_size {
                    // Pseudo-random sampling with fixed seed for reproducibility
                    let t = ((sample_idx as u32).wrapping_mul(2654435761_u32) ^ (i as u32))
                        .wrapping_mul(2654435761_u32) as f32
                        / u32::MAX as f32;
                    x_sample[i] = group_lower[i] + (group_upper[i] - group_lower[i]) * t;
                }

                // Also sample corners for first few samples
                if sample_idx < softmax_size * 2 {
                    let dim = sample_idx / 2;
                    if dim < softmax_size {
                        x_sample.assign(&x_center);
                        x_sample[dim] = if sample_idx % 2 == 0 {
                            group_lower[dim]
                        } else {
                            group_upper[dim]
                        };
                    }
                }

                let y_actual = self.eval(&x_sample);
                let y_approx: Array1<f32> = jacobian.dot(&x_sample) + &b_approx;

                for i in 0..softmax_size {
                    let error = y_actual[i] - y_approx[i];
                    if error > max_error_above[i] {
                        max_error_above[i] = error;
                    }
                    if -error > max_error_below[i] {
                        max_error_below[i] = -error;
                    }
                }
            }

            // Add safety margin (10% extra for unsampled regions)
            let safety_factor = 1.1;
            let min_margin = 1e-6_f32;
            for i in 0..softmax_size {
                max_error_above[i] = (max_error_above[i] * safety_factor).max(min_margin);
                max_error_below[i] = (max_error_below[i] * safety_factor).max(min_margin);
            }

            let lower_b_slice = &b_approx - &max_error_below;
            let upper_b_slice = &b_approx + &max_error_above;

            for (local_i, &global_i) in flat_indices_for_group.iter().enumerate() {
                lower_b[global_i] = lower_b_slice[local_i];
                upper_b[global_i] = upper_b_slice[local_i];
                for (local_k, &global_k) in flat_indices_for_group.iter().enumerate() {
                    let val = jacobian[[local_i, local_k]];
                    lower_a[[global_i, global_k]] = val;
                    upper_a[[global_i, global_k]] = val;
                }
            }
        }

        self.apply_affine_bounds(bounds, &lower_a, &lower_b, &upper_a, &upper_b)
    }

    pub(crate) fn apply_affine_bounds(
        &self,
        bounds: &LinearBounds,
        lower_a: &Array2<f32>,
        lower_b: &Array1<f32>,
        upper_a: &Array2<f32>,
        upper_b: &Array1<f32>,
    ) -> Result<LinearBounds> {
        utils::apply_affine_bounds_f64(bounds, lower_a, lower_b, upper_a, upper_b)
    }
}
