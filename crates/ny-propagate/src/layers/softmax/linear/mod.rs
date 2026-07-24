// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN linear bounds propagation for softmax.
//!
//! Implements backward bound propagation through softmax using local
//! linearization (heuristic mode) or LSE-based affine bounds (sound mode).
//! Supports 1D, 2D, and general N-D inputs with independent group decomposition
//! along the softmax axis.

mod batched;
mod heuristic;
mod sound;
#[cfg(test)]
mod tests;

use ndarray::{Array1, Array2};
use ny_core::{checked_shape_product, NyError, Result, VerificationSoundnessMode};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use tracing::debug;

use crate::LinearBounds;

use super::super::common::BoundPropagation;
use super::bounds::constant_bounds_from_output;
use super::layer::SoftmaxLayer;

impl SoftmaxLayer {
    /// Compute CROWN linear bounds for softmax with pre-activation bounds.
    ///
    /// Uses local linearization at the center point with sampling-based error estimates.
    /// Returns linear bounds: y_lower >= A_l @ x + b_l, y_upper <= A_u @ x + b_u
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        soundness: VerificationSoundnessMode,
    ) -> Result<LinearBounds> {
        let ndim = pre_activation.shape().len();
        if ndim == 0 {
            return Err(NyError::InvalidSpec(
                "Softmax requires at least 1D input".to_string(),
            ));
        }
        let effective_soundness = if self.sound {
            if soundness == VerificationSoundnessMode::Heuristic {
                debug!("Softmax heuristic requested, but layer is in sound mode; using IBP constant bounds");
            }
            VerificationSoundnessMode::Sound
        } else {
            soundness
        };

        let use_sound = effective_soundness == VerificationSoundnessMode::Sound;
        if use_sound {
            debug!("Softmax sound mode: using LSE-based affine bounds");
        } else {
            debug!("Softmax heuristic mode: sampling-based bounds (not sound)");
        }

        if use_sound {
            let has_non_finite = pre_activation.lower().iter().any(|&v| !v.is_finite())
                || pre_activation.upper().iter().any(|&v| !v.is_finite());
            if has_non_finite {
                debug!("Softmax sound mode: non-finite pre-activation bounds; falling back to IBP constant bounds");
                let output_bounds = self.propagate_ibp(pre_activation)?;
                return constant_bounds_from_output(bounds, &output_bounds);
            }
        }

        let shape = pre_activation.shape();
        let ndim = shape.len();
        let axis = crate::layers::common::resolve_axis_i32(self.axis, ndim, "Softmax")?;

        match ndim {
            1 => {
                let pre_lower = pre_activation
                    .lower()
                    .clone()
                    .into_dimensionality::<ndarray::Ix1>()
                    .map_err(|_| NyError::ShapeMismatch {
                        expected: vec![pre_activation.len()],
                        got: pre_activation.lower().shape().to_vec(),
                    })?;
                let pre_upper = pre_activation
                    .upper()
                    .clone()
                    .into_dimensionality::<ndarray::Ix1>()
                    .map_err(|_| NyError::ShapeMismatch {
                        expected: vec![pre_activation.len()],
                        got: pre_activation.upper().shape().to_vec(),
                    })?;
                if use_sound {
                    self.propagate_linear_with_bounds_1d_sound(bounds, &pre_lower, &pre_upper)
                } else {
                    self.propagate_linear_with_bounds_1d_heuristic(bounds, &pre_lower, &pre_upper)
                }
            }
            2 => {
                self.propagate_linear_with_bounds_2d(bounds, pre_activation, shape, axis, use_sound)
            }
            _ => {
                self.propagate_linear_with_bounds_nd(bounds, pre_activation, shape, axis, use_sound)
            }
        }
    }

    /// 2D softmax CROWN backward: decompose into independent 1D groups along axis.
    fn propagate_linear_with_bounds_2d(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        shape: &[usize],
        axis: usize,
        use_sound: bool,
    ) -> Result<LinearBounds> {
        if bounds.num_inputs() != shape[0] * shape[1] {
            return Err(NyError::ShapeMismatch {
                expected: vec![shape[0] * shape[1]],
                got: vec![bounds.num_inputs()],
            });
        }

        let pre_lower = pre_activation
            .lower()
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| NyError::InvalidSpec("Softmax pre-activation must be 2D".to_string()))?;
        let pre_upper = pre_activation
            .upper()
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| NyError::InvalidSpec("Softmax pre-activation must be 2D".to_string()))?;

        let rows = shape[0];
        let cols = shape[1];
        let num_outputs = bounds.num_outputs();

        let num_groups = if axis == 0 { cols } else { rows };
        if num_groups == 0 {
            // Empty non-axis dimension means there are no independent softmax groups.
            // The affine form has no coefficient contributions and should remain
            // constant-only (preserve incoming bias) without dividing by 0.
            return LinearBounds::new_or_conservative(
                Array2::<f32>::zeros((num_outputs, rows * cols)),
                bounds.lower_b().clone(),
                Array2::<f32>::zeros((num_outputs, rows * cols)),
                bounds.upper_b().clone(),
            );
        }
        let num_groups_f = num_groups as f32;
        let bias_split_lower = bounds.lower_b().mapv(|v| v / num_groups_f);
        let bias_split_upper = bounds.upper_b().mapv(|v| v / num_groups_f);

        let mut out_lower_a = Array2::<f32>::zeros((num_outputs, rows * cols));
        let mut out_upper_a = Array2::<f32>::zeros((num_outputs, rows * cols));
        // Cross-group bias accumulation in f64 to prevent catastrophic cancellation (#2489).
        // Each group's bias is computed in f64 with directed rounding, but accumulating
        // many groups back in f32 negates the per-group precision. For column-wise softmax
        // over sequence_length=512, this is 512 f32 additions.
        let mut out_lower_b = Array1::<f64>::zeros(num_outputs);
        let mut out_upper_b = Array1::<f64>::zeros(num_outputs);

        if axis == 0 {
            // Column-wise softmax: operate independently on each column.
            for j in 0..cols {
                let mut group_lower = Array1::<f32>::zeros(rows);
                let mut group_upper = Array1::<f32>::zeros(rows);
                for i in 0..rows {
                    group_lower[i] = pre_lower[[i, j]];
                    group_upper[i] = pre_upper[[i, j]];
                }

                let mut group_lower_a = Array2::<f32>::zeros((num_outputs, rows));
                let mut group_upper_a = Array2::<f32>::zeros((num_outputs, rows));
                for out_idx in 0..num_outputs {
                    for i in 0..rows {
                        let flat = i * cols + j;
                        group_lower_a[[out_idx, i]] = bounds.lower_a()[[out_idx, flat]];
                        group_upper_a[[out_idx, i]] = bounds.upper_a()[[out_idx, flat]];
                    }
                }

                let group_bounds = LinearBounds::new_or_conservative(
                    group_lower_a,
                    bias_split_lower.clone(),
                    group_upper_a,
                    bias_split_upper.clone(),
                )?;

                let group_result = if use_sound {
                    self.propagate_linear_with_bounds_1d_sound(
                        &group_bounds,
                        &group_lower,
                        &group_upper,
                    )?
                } else {
                    self.propagate_linear_with_bounds_1d_heuristic(
                        &group_bounds,
                        &group_lower,
                        &group_upper,
                    )?
                };

                for out_idx in 0..num_outputs {
                    for i in 0..rows {
                        let flat = i * cols + j;
                        out_lower_a[[out_idx, flat]] += group_result.lower_a()[[out_idx, i]];
                        out_upper_a[[out_idx, flat]] += group_result.upper_a()[[out_idx, i]];
                    }
                }
                out_lower_b += &group_result.lower_b().mapv(|v| v as f64);
                out_upper_b += &group_result.upper_b().mapv(|v| v as f64);
            }
        } else {
            // Row-wise softmax: operate independently on each row.
            for i in 0..rows {
                let group_lower = pre_lower.row(i).to_owned();
                let group_upper = pre_upper.row(i).to_owned();

                let mut group_lower_a = Array2::<f32>::zeros((num_outputs, cols));
                let mut group_upper_a = Array2::<f32>::zeros((num_outputs, cols));
                for out_idx in 0..num_outputs {
                    for j in 0..cols {
                        let flat = i * cols + j;
                        group_lower_a[[out_idx, j]] = bounds.lower_a()[[out_idx, flat]];
                        group_upper_a[[out_idx, j]] = bounds.upper_a()[[out_idx, flat]];
                    }
                }

                let group_bounds = LinearBounds::new_or_conservative(
                    group_lower_a,
                    bias_split_lower.clone(),
                    group_upper_a,
                    bias_split_upper.clone(),
                )?;

                let group_result = if use_sound {
                    self.propagate_linear_with_bounds_1d_sound(
                        &group_bounds,
                        &group_lower,
                        &group_upper,
                    )?
                } else {
                    self.propagate_linear_with_bounds_1d_heuristic(
                        &group_bounds,
                        &group_lower,
                        &group_upper,
                    )?
                };

                for out_idx in 0..num_outputs {
                    for j in 0..cols {
                        let flat = i * cols + j;
                        out_lower_a[[out_idx, flat]] += group_result.lower_a()[[out_idx, j]];
                        out_upper_a[[out_idx, flat]] += group_result.upper_a()[[out_idx, j]];
                    }
                }
                out_lower_b += &group_result.lower_b().mapv(|v| v as f64);
                out_upper_b += &group_result.upper_b().mapv(|v| v as f64);
            }
        }

        // Directed rounding on f64→f32 downcast (#2489):
        // lower bounds round toward -inf, upper bounds round toward +inf.
        let out_lower_b = out_lower_b.mapv(|v| next_down_f32(v as f32));
        let out_upper_b = out_upper_b.mapv(|v| next_up_f32(v as f32));

        LinearBounds::new_or_conservative(out_lower_a, out_lower_b, out_upper_a, out_upper_b)
    }

    /// General N-D softmax CROWN backward: decompose into independent 1D groups.
    fn propagate_linear_with_bounds_nd(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        shape: &[usize],
        axis: usize,
        use_sound: bool,
    ) -> Result<LinearBounds> {
        let total_size: usize = checked_shape_product(shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Softmax CROWN: shape product overflows usize: {:?}",
                shape,
            ))
        })?;
        if bounds.num_inputs() != total_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![total_size],
                got: vec![bounds.num_inputs()],
            });
        }

        let ndim = shape.len();
        let softmax_size = shape[axis];
        let num_outputs = bounds.num_outputs();

        // Compute strides for converting between flat and multi-dimensional indices
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
                "Softmax CROWN: non-axis dimensions {non_axis_dims:?} overflow usize",
            ))
        })?;

        if num_groups == 0 {
            // Empty non-axis dimension means there are no independent softmax groups.
            // Preserve incoming bias and keep coefficient matrices zero instead of
            // forcing one fake group via `.max(1)`.
            return LinearBounds::new_or_conservative(
                Array2::<f32>::zeros((num_outputs, total_size)),
                bounds.lower_b().clone(),
                Array2::<f32>::zeros((num_outputs, total_size)),
                bounds.upper_b().clone(),
            );
        }

        let num_groups_f = num_groups as f32;
        let bias_split_lower = bounds.lower_b().mapv(|v| v / num_groups_f);
        let bias_split_upper = bounds.upper_b().mapv(|v| v / num_groups_f);

        let mut out_lower_a = Array2::<f32>::zeros((num_outputs, total_size));
        let mut out_upper_a = Array2::<f32>::zeros((num_outputs, total_size));
        // Cross-group bias accumulation in f64 (#2489), same rationale as 2D path.
        let mut out_lower_b = Array1::<f64>::zeros(num_outputs);
        let mut out_upper_b = Array1::<f64>::zeros(num_outputs);

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

        // Stack-allocated index buffers — ndim is always small (2-5 for
        // typical tensors). Eliminates num_groups × ndim heap allocations.
        assert!(ndim <= 8, "softmax N-D indexing assumes ndim <= 8");
        let mut group_strides = [1usize; 8];
        if !group_shape.is_empty() {
            for d in (0..group_shape.len() - 1).rev() {
                group_strides[d] = group_strides[d + 1] * group_shape[d + 1];
            }
        }

        let mut group_multi = [0usize; 8];
        let mut full_idx = [0usize; 8];
        // Pre-allocate flat index buffer once outside the group loop to avoid
        // num_groups heap allocations (#2237 Finding 3).
        let mut flat_indices_for_group = Vec::with_capacity(softmax_size);
        for group_idx in 0..num_groups {
            // Convert group_idx to indices on non-axis dimensions
            let mut remaining = group_idx;
            for d in 0..group_shape.len() {
                group_multi[d] = remaining / group_strides[d];
                remaining %= group_strides[d];
            }

            // Extract 1D slice bounds for this group
            let mut group_lower = Array1::<f32>::zeros(softmax_size);
            let mut group_upper = Array1::<f32>::zeros(softmax_size);

            // Map flat indices for this group (reuses pre-allocated buffer).
            flat_indices_for_group.clear();
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

            // Extract coefficients for this group
            let mut group_lower_a = Array2::<f32>::zeros((num_outputs, softmax_size));
            let mut group_upper_a = Array2::<f32>::zeros((num_outputs, softmax_size));
            for out_idx in 0..num_outputs {
                for (s, &flat) in flat_indices_for_group.iter().enumerate() {
                    group_lower_a[[out_idx, s]] = bounds.lower_a()[[out_idx, flat]];
                    group_upper_a[[out_idx, s]] = bounds.upper_a()[[out_idx, flat]];
                }
            }

            let group_bounds = LinearBounds::new_or_conservative(
                group_lower_a,
                bias_split_lower.clone(),
                group_upper_a,
                bias_split_upper.clone(),
            )?;

            let group_result = if use_sound {
                self.propagate_linear_with_bounds_1d_sound(
                    &group_bounds,
                    &group_lower,
                    &group_upper,
                )?
            } else {
                self.propagate_linear_with_bounds_1d_heuristic(
                    &group_bounds,
                    &group_lower,
                    &group_upper,
                )?
            };

            // Embed results back into full tensor
            for out_idx in 0..num_outputs {
                for (s, &flat) in flat_indices_for_group.iter().enumerate() {
                    out_lower_a[[out_idx, flat]] += group_result.lower_a()[[out_idx, s]];
                    out_upper_a[[out_idx, flat]] += group_result.upper_a()[[out_idx, s]];
                }
            }
            out_lower_b += &group_result.lower_b().mapv(|v| v as f64);
            out_upper_b += &group_result.upper_b().mapv(|v| v as f64);
        }

        // Directed rounding on f64→f32 downcast (#2489).
        let out_lower_b = out_lower_b.mapv(|v| next_down_f32(v as f32));
        let out_upper_b = out_upper_b.mapv(|v| next_up_f32(v as f32));

        LinearBounds::new_or_conservative(out_lower_a, out_lower_b, out_upper_a, out_upper_b)
    }
}
