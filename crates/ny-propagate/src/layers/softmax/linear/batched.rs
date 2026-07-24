// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN backward propagation through softmax.
//!
//! Operates on N-D batched bounds, preserving batch structure. Softmax is
//! applied independently along the last dimension of pre_activation (axis=-1).
//! Each batch element is processed via the corresponding 1D method (heuristic
//! or sound).

use ndarray::{s, Array1, Array2, Array3, ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result, VerificationSoundnessMode};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use tracing::debug;

use crate::{BatchedLinearBounds, LinearBounds};

use super::super::bounds::batched_constant_bounds_from_output;
use super::super::layer::SoftmaxLayer;
use crate::layers::common::BoundPropagation;

impl SoftmaxLayer {
    /// Batched CROWN backward propagation through Softmax with pre-activation bounds.
    ///
    /// Same as `propagate_linear_with_bounds` but operates on N-D batched bounds,
    /// preserving batch structure. Softmax is applied independently along the
    /// last dimension of pre_activation (axis=-1).
    ///
    /// # Arguments
    /// - `bounds`: BatchedLinearBounds with shape [...batch_dims, out_dim, softmax_size]
    /// - `pre_activation`: Input bounds with shape [...batch_dims, softmax_size]
    ///
    /// # Returns
    /// New BatchedLinearBounds with softmax backward propagation applied.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
        soundness: VerificationSoundnessMode,
    ) -> Result<BatchedLinearBounds> {
        // Reconcile external soundness directive with layer's own sound flag,
        // matching the non-batched version (propagate_linear_with_bounds).
        let effective_soundness = if self.sound {
            if soundness == VerificationSoundnessMode::Heuristic {
                debug!("Softmax batched: heuristic requested, but layer is in sound mode; using IBP constant bounds");
            }
            VerificationSoundnessMode::Sound
        } else {
            soundness
        };

        let use_sound = effective_soundness == VerificationSoundnessMode::Sound;

        if use_sound {
            let has_non_finite = pre_activation.lower().iter().any(|&v| !v.is_finite())
                || pre_activation.upper().iter().any(|&v| !v.is_finite());
            if has_non_finite {
                debug!("Softmax sound mode: non-finite pre-activation bounds; falling back to IBP constant bounds (batched)");
                let output_bounds = self.propagate_ibp(pre_activation)?;
                return batched_constant_bounds_from_output(bounds, &output_bounds);
            }
            debug!("Softmax sound mode: using LSE-based affine bounds (batched)");
            return self.propagate_linear_batched_with_bounds_sound(bounds, pre_activation);
        }

        debug!("Softmax layer batched CROWN backward propagation (heuristic)");

        let pre_shape = pre_activation.shape();
        let a_shape = bounds.lower_a.shape();

        if a_shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        // Detect flat-with-groups case: bounds are 2D flat [out_dim, total_in]
        // but pre_activation is multi-dimensional [num_groups, softmax_size].
        // This occurs in attention CROWN backward where BilinearCrown flattens
        // batched bounds to block-diagonal format for broadcast composition.
        let pre_softmax_size = *pre_shape.last().unwrap_or(&0);
        let a_in_dim = a_shape[a_shape.len() - 1];
        let is_flat_with_groups = a_shape.len() == 2
            && pre_shape.len() >= 2
            && pre_softmax_size > 0
            && a_in_dim != pre_softmax_size
            && a_in_dim.is_multiple_of(pre_softmax_size);

        if is_flat_with_groups {
            return self.propagate_linear_flat_grouped_heuristic(bounds, pre_activation);
        }

        let out_dim = a_shape[a_shape.len() - 2];
        let softmax_size = a_in_dim;
        let batch_dims = &a_shape[..a_shape.len() - 2];
        let total_batch: usize = checked_shape_product(batch_dims).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Softmax batched CROWN: batch dims product overflows usize: {:?}",
                batch_dims,
            ))
        })?;
        let total_batch = total_batch.max(1);

        // Verify pre_activation shape matches
        if pre_softmax_size != softmax_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![softmax_size],
                got: vec![pre_softmax_size],
            });
        }

        // Reshape pre-activation to [batch, softmax_size]
        let pre_lower_flat = pre_activation
            .lower()
            .view()
            .into_shape_with_order((total_batch, softmax_size))
            .map_err(|_| {
                NyError::InvalidSpec("Cannot reshape pre_lower for softmax".to_string())
            })?;
        let pre_upper_flat = pre_activation
            .upper()
            .view()
            .into_shape_with_order((total_batch, softmax_size))
            .map_err(|_| {
                NyError::InvalidSpec("Cannot reshape pre_upper for softmax".to_string())
            })?;

        // Reshape bounds to [batch, out_dim, softmax_size]
        let lower_a_3d = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_batch, out_dim, softmax_size))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_a for softmax".to_string()))?;
        let upper_a_3d = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_batch, out_dim, softmax_size))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_a for softmax".to_string()))?;
        let lower_b_2d = bounds
            .lower_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_b for softmax".to_string()))?;
        let upper_b_2d = bounds
            .upper_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_b for softmax".to_string()))?;

        // Output arrays
        let mut new_lower_a = Array3::<f32>::zeros((total_batch, out_dim, softmax_size));
        let mut new_upper_a = Array3::<f32>::zeros((total_batch, out_dim, softmax_size));
        let mut new_lower_b = Array2::<f32>::zeros((total_batch, out_dim));
        let mut new_upper_b = Array2::<f32>::zeros((total_batch, out_dim));

        // Process each batch position independently using the 1D softmax backward
        for b in 0..total_batch {
            // Extract 1D pre-activation bounds for this batch
            let pre_lower_1d = pre_lower_flat.row(b).to_owned();
            let pre_upper_1d = pre_upper_flat.row(b).to_owned();

            // Extract 2D coefficient matrix for this batch: [out_dim, softmax_size]
            let lower_a_slice = lower_a_3d.slice(ndarray::s![b, .., ..]).to_owned();
            let upper_a_slice = upper_a_3d.slice(ndarray::s![b, .., ..]).to_owned();
            let lower_b_slice = lower_b_2d.row(b).to_owned();
            let upper_b_slice = upper_b_2d.row(b).to_owned();

            let batch_bounds = LinearBounds::new_or_conservative(
                lower_a_slice,
                lower_b_slice,
                upper_a_slice,
                upper_b_slice,
            )?;

            // Apply 1D softmax backward
            let result = self.propagate_linear_with_bounds_1d_heuristic(
                &batch_bounds,
                &pre_lower_1d,
                &pre_upper_1d,
            )?;

            // Copy results back
            for j in 0..out_dim {
                for k in 0..softmax_size {
                    new_lower_a[[b, j, k]] = result.lower_a()[[j, k]];
                    new_upper_a[[b, j, k]] = result.upper_a()[[j, k]];
                }
                new_lower_b[[b, j]] = result.lower_b()[j];
                new_upper_b[[b, j]] = result.upper_b()[j];
            }
        }

        // Reshape back to original batch dims
        let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
        let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();
        let (new_lower_b_vec, _) = new_lower_b.into_raw_vec_and_offset();
        let (new_upper_b_vec, _) = new_upper_b.into_raw_vec_and_offset();

        let out_a_shape: Vec<usize> = batch_dims
            .iter()
            .cloned()
            .chain([out_dim, softmax_size])
            .collect();
        let out_b_shape: Vec<usize> = batch_dims.iter().cloned().chain([out_dim]).collect();

        // CROWN backward NaN firewall (#2812): conservative fallback instead of hard error.
        // Reassembled from per-batch heuristic results (#3033).
        BatchedLinearBounds::new_or_conservative(
            ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?,
            ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_b_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?,
            ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?,
            ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_b_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?,
            bounds.input_shape.clone(),
            bounds.output_shape.clone(),
        )
    }

    fn propagate_linear_batched_with_bounds_sound(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        let pre_shape = pre_activation.shape();
        let a_shape = bounds.lower_a.shape();

        if a_shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        // Detect flat-with-groups case (same as heuristic path)
        let pre_softmax_size = *pre_shape.last().unwrap_or(&0);
        let a_in_dim = a_shape[a_shape.len() - 1];
        let is_flat_with_groups = a_shape.len() == 2
            && pre_shape.len() >= 2
            && pre_softmax_size > 0
            && a_in_dim != pre_softmax_size
            && a_in_dim.is_multiple_of(pre_softmax_size);

        if is_flat_with_groups {
            return self.propagate_linear_flat_grouped_sound(bounds, pre_activation);
        }

        let out_dim = a_shape[a_shape.len() - 2];
        let softmax_size = a_in_dim;
        let batch_dims = &a_shape[..a_shape.len() - 2];
        let total_batch: usize = checked_shape_product(batch_dims).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Softmax batched CROWN: batch dims product overflows usize: {:?}",
                batch_dims,
            ))
        })?;
        let total_batch = total_batch.max(1);

        if pre_softmax_size != softmax_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![softmax_size],
                got: vec![pre_softmax_size],
            });
        }

        let pre_lower_flat = pre_activation
            .lower()
            .view()
            .into_shape_with_order((total_batch, softmax_size))
            .map_err(|_| {
                NyError::InvalidSpec("Cannot reshape pre_lower for softmax".to_string())
            })?;
        let pre_upper_flat = pre_activation
            .upper()
            .view()
            .into_shape_with_order((total_batch, softmax_size))
            .map_err(|_| {
                NyError::InvalidSpec("Cannot reshape pre_upper for softmax".to_string())
            })?;

        let lower_a = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_batch, out_dim, softmax_size))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_a for softmax".to_string()))?;
        let upper_a = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_batch, out_dim, softmax_size))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_a for softmax".to_string()))?;
        let lower_b = bounds
            .lower_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_b for softmax".to_string()))?;
        let upper_b = bounds
            .upper_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_b for softmax".to_string()))?;

        let mut new_lower_a = Array3::<f32>::zeros((total_batch, out_dim, softmax_size));
        let mut new_upper_a = Array3::<f32>::zeros((total_batch, out_dim, softmax_size));
        let mut new_lower_b = Array2::<f32>::zeros((total_batch, out_dim));
        let mut new_upper_b = Array2::<f32>::zeros((total_batch, out_dim));

        for b in 0..total_batch {
            let pre_lower = pre_lower_flat.row(b).to_owned();
            let pre_upper = pre_upper_flat.row(b).to_owned();
            let group_bounds = LinearBounds::new_or_conservative(
                lower_a.slice(s![b, .., ..]).to_owned(),
                lower_b.row(b).to_owned(),
                upper_a.slice(s![b, .., ..]).to_owned(),
                upper_b.row(b).to_owned(),
            )?;

            let result =
                self.propagate_linear_with_bounds_1d_sound(&group_bounds, &pre_lower, &pre_upper)?;

            new_lower_a
                .slice_mut(s![b, .., ..])
                .assign(result.lower_a());
            new_upper_a
                .slice_mut(s![b, .., ..])
                .assign(result.upper_a());
            new_lower_b.row_mut(b).assign(result.lower_b());
            new_upper_b.row_mut(b).assign(result.upper_b());
        }

        let out_a_shape: Vec<usize> = batch_dims
            .iter()
            .cloned()
            .chain([out_dim, softmax_size])
            .collect();
        let out_b_shape: Vec<usize> = batch_dims.iter().cloned().chain([out_dim]).collect();

        let (lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
        let (lower_b_vec, _) = new_lower_b.into_raw_vec_and_offset();
        let (upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();
        let (upper_b_vec, _) = new_upper_b.into_raw_vec_and_offset();

        // Validated construction: reassembled from per-batch sound results,
        // NaN firewall catches corruption during reassembly (#3033).
        BatchedLinearBounds::new_or_conservative(
            ArrayD::from_shape_vec(IxDyn(&out_a_shape), lower_a_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?,
            ArrayD::from_shape_vec(IxDyn(&out_b_shape), lower_b_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?,
            ArrayD::from_shape_vec(IxDyn(&out_a_shape), upper_a_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?,
            ArrayD::from_shape_vec(IxDyn(&out_b_shape), upper_b_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?,
            bounds.input_shape.clone(),
            bounds.output_shape.clone(),
        )
    }

    /// Softmax backward for flat bounds with embedded group structure.
    ///
    /// When BilinearCrown flattens batched bounds to block-diagonal format,
    /// the A matrix arrives as `[out_dim, total_in]` where `total_in = num_groups * softmax_size`.
    /// Each group of `softmax_size` columns corresponds to an independent softmax row.
    ///
    /// For each group g, we extract columns `[g*softmax_size, (g+1)*softmax_size)`,
    /// apply 1D softmax backward, and write the result back to those columns.
    fn propagate_linear_flat_grouped_heuristic(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        let pre_shape = pre_activation.shape();
        let a_shape = bounds.lower_a.shape();

        let out_dim = a_shape[0];
        let total_in = a_shape[1];
        let softmax_size = *pre_shape.last().unwrap_or(&0);
        let num_groups = total_in / softmax_size;

        debug!(
            "Softmax flat-grouped heuristic: out_dim={}, total_in={}, softmax_size={}, num_groups={}",
            out_dim, total_in, softmax_size, num_groups
        );

        // Reshape pre-activation to [num_groups, softmax_size]
        let pre_lower_2d = pre_activation
            .lower()
            .view()
            .into_shape_with_order((num_groups, softmax_size))
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "Softmax flat-grouped: reshape pre_lower to [{num_groups}, {softmax_size}] failed: {e}"
                ))
            })?
            .to_owned();
        let pre_upper_2d = pre_activation
            .upper()
            .view()
            .into_shape_with_order((num_groups, softmax_size))
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "Softmax flat-grouped: reshape pre_upper to [{num_groups}, {softmax_size}] failed: {e}"
                ))
            })?
            .to_owned();

        // View A as [out_dim, total_in] (already 2D)
        let lower_a_2d = bounds
            .lower_a
            .view()
            .into_shape_with_order((out_dim, total_in))
            .map_err(|e| {
                NyError::InvalidSpec(format!("Softmax flat-grouped: reshape lower_a failed: {e}"))
            })?;
        let upper_a_2d = bounds
            .upper_a
            .view()
            .into_shape_with_order((out_dim, total_in))
            .map_err(|e| {
                NyError::InvalidSpec(format!("Softmax flat-grouped: reshape upper_a failed: {e}"))
            })?;

        let mut new_lower_a = Array2::<f32>::zeros((out_dim, total_in));
        let mut new_upper_a = Array2::<f32>::zeros((out_dim, total_in));
        // Bias: accumulated across all groups
        let lower_b_1d = bounds
            .lower_b
            .view()
            .into_shape_with_order(out_dim)
            .map_err(|e| {
                NyError::InvalidSpec(format!("Softmax flat-grouped: reshape lower_b failed: {e}"))
            })?
            .to_owned();
        let upper_b_1d = bounds
            .upper_b
            .view()
            .into_shape_with_order(out_dim)
            .map_err(|e| {
                NyError::InvalidSpec(format!("Softmax flat-grouped: reshape upper_b failed: {e}"))
            })?
            .to_owned();

        // Split bias equally among groups so each 1D backward gets its share.
        // Use directed rounding: lower rounds down, upper rounds up.
        let inv_groups = 1.0 / num_groups as f32;
        let group_lower_b = lower_b_1d.mapv(|v| next_down_f32(v * inv_groups));
        let group_upper_b = upper_b_1d.mapv(|v| next_up_f32(v * inv_groups));

        // f64 bias accumulators to prevent catastrophic cancellation (#2336, #1745).
        // Pattern matches non-batched softmax backward at mod.rs:217-272.
        let mut new_lower_b_f64 = Array1::<f64>::zeros(out_dim);
        let mut new_upper_b_f64 = Array1::<f64>::zeros(out_dim);

        for g in 0..num_groups {
            let col_start = g * softmax_size;
            let col_end = col_start + softmax_size;

            // Extract [out_dim, softmax_size] slice for this group
            let group_la = lower_a_2d.slice(s![.., col_start..col_end]).to_owned();
            let group_ua = upper_a_2d.slice(s![.., col_start..col_end]).to_owned();

            let group_bounds = LinearBounds::new_or_conservative(
                group_la,
                group_lower_b.clone(),
                group_ua,
                group_upper_b.clone(),
            )?;

            let result = self.propagate_linear_with_bounds_1d_heuristic(
                &group_bounds,
                &pre_lower_2d.row(g).to_owned(),
                &pre_upper_2d.row(g).to_owned(),
            )?;

            // Write result coefficients back to the group's column range
            for j in 0..out_dim {
                for k in 0..softmax_size {
                    new_lower_a[[j, col_start + k]] = result.lower_a()[[j, k]];
                    new_upper_a[[j, col_start + k]] = result.upper_a()[[j, k]];
                }
                new_lower_b_f64[j] += result.lower_b()[j] as f64;
                new_upper_b_f64[j] += result.upper_b()[j] as f64;
            }
        }

        // Directed rounding on f64→f32 downcast (#2489, #2336):
        let new_lower_b = new_lower_b_f64.mapv(|v| next_down_f32(v as f32));
        let new_upper_b = new_upper_b_f64.mapv(|v| next_up_f32(v as f32));

        BatchedLinearBounds::new_or_conservative(
            new_lower_a.into_dyn(),
            new_lower_b.into_dyn(),
            new_upper_a.into_dyn(),
            new_upper_b.into_dyn(),
            bounds.input_shape.clone(),
            bounds.output_shape.clone(),
        )
    }

    /// Sound variant of flat-grouped softmax backward.
    fn propagate_linear_flat_grouped_sound(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        let pre_shape = pre_activation.shape();
        let a_shape = bounds.lower_a.shape();

        let out_dim = a_shape[0];
        let total_in = a_shape[1];
        let softmax_size = *pre_shape.last().unwrap_or(&0);
        let num_groups = total_in / softmax_size;

        debug!(
            "Softmax flat-grouped sound: out_dim={}, total_in={}, softmax_size={}, num_groups={}",
            out_dim, total_in, softmax_size, num_groups
        );

        let pre_lower_2d = pre_activation
            .lower()
            .view()
            .into_shape_with_order((num_groups, softmax_size))
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "Softmax flat-grouped sound: reshape pre_lower failed: {e}"
                ))
            })?
            .to_owned();
        let pre_upper_2d = pre_activation
            .upper()
            .view()
            .into_shape_with_order((num_groups, softmax_size))
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "Softmax flat-grouped sound: reshape pre_upper failed: {e}"
                ))
            })?
            .to_owned();

        let lower_a_2d = bounds
            .lower_a
            .view()
            .into_shape_with_order((out_dim, total_in))
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "Softmax flat-grouped sound: reshape lower_a failed: {e}"
                ))
            })?;
        let upper_a_2d = bounds
            .upper_a
            .view()
            .into_shape_with_order((out_dim, total_in))
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "Softmax flat-grouped sound: reshape upper_a failed: {e}"
                ))
            })?;

        let lower_b_1d = bounds
            .lower_b
            .view()
            .into_shape_with_order(out_dim)
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "Softmax flat-grouped sound: reshape lower_b failed: {e}"
                ))
            })?
            .to_owned();
        let upper_b_1d = bounds
            .upper_b
            .view()
            .into_shape_with_order(out_dim)
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "Softmax flat-grouped sound: reshape upper_b failed: {e}"
                ))
            })?
            .to_owned();

        let inv_groups = 1.0 / num_groups as f32;
        let group_lower_b = lower_b_1d.mapv(|v| next_down_f32(v * inv_groups));
        let group_upper_b = upper_b_1d.mapv(|v| next_up_f32(v * inv_groups));

        let mut new_lower_a = Array2::<f32>::zeros((out_dim, total_in));
        let mut new_upper_a = Array2::<f32>::zeros((out_dim, total_in));
        // f64 bias accumulators to prevent catastrophic cancellation (#2336, #1745).
        // Pattern matches non-batched softmax backward at mod.rs:217-272.
        let mut new_lower_b_f64 = Array1::<f64>::zeros(out_dim);
        let mut new_upper_b_f64 = Array1::<f64>::zeros(out_dim);

        for g in 0..num_groups {
            let col_start = g * softmax_size;
            let col_end = col_start + softmax_size;

            let group_la = lower_a_2d.slice(s![.., col_start..col_end]).to_owned();
            let group_ua = upper_a_2d.slice(s![.., col_start..col_end]).to_owned();

            let group_bounds = LinearBounds::new_or_conservative(
                group_la,
                group_lower_b.clone(),
                group_ua,
                group_upper_b.clone(),
            )?;

            let result = self.propagate_linear_with_bounds_1d_sound(
                &group_bounds,
                &pre_lower_2d.row(g).to_owned(),
                &pre_upper_2d.row(g).to_owned(),
            )?;

            for j in 0..out_dim {
                for k in 0..softmax_size {
                    new_lower_a[[j, col_start + k]] = result.lower_a()[[j, k]];
                    new_upper_a[[j, col_start + k]] = result.upper_a()[[j, k]];
                }
                new_lower_b_f64[j] += result.lower_b()[j] as f64;
                new_upper_b_f64[j] += result.upper_b()[j] as f64;
            }
        }

        // Directed rounding on f64→f32 downcast (#2489, #2336):
        let new_lower_b = new_lower_b_f64.mapv(|v| next_down_f32(v as f32));
        let new_upper_b = new_upper_b_f64.mapv(|v| next_up_f32(v as f32));

        BatchedLinearBounds::new_or_conservative(
            new_lower_a.into_dyn(),
            new_lower_b.into_dyn(),
            new_upper_a.into_dyn(),
            new_upper_b.into_dyn(),
            bounds.input_shape.clone(),
            bounds.output_shape.clone(),
        )
    }
}
