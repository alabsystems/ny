// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tile layer: repeats tensor along specified axis for bound propagation.

use ndarray::{Array2, ArrayD, Axis, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;
use tracing::debug;

use super::super::common::BoundPropagation;
use crate::{contiguous_flat_slice, contiguous_flat_slice_mut, BatchedLinearBounds, LinearBounds};

/// Certified coefficient error for the dense (Array2) Tile backward
/// (#vnncomp-aw-soundness).
///
/// The Tile backward sums `reps` replica coefficients per output column in f32,
/// a contraction of width `reps`. The certified error per `(row, i)` is
/// `γ_reps·S + prop` with the EXACT abs-sums over the same `reps` source columns:
///   S[row,i]    = Σ_rep |A[row, src_rep]|
///   prop[row,i] = Σ_rep |err_in[row, src_rep]|
/// rounded OUTWARD. Source column for replica `rep` of input column `i` is
/// `(i / block_size)·out_block_size + rep·block_size + (i % block_size)`.
#[allow(clippy::too_many_arguments)]
fn tile_backward_coeff_err(
    lower_a: &Array2<f32>,
    upper_a: &Array2<f32>,
    lower_a_err: Option<&Array2<f32>>,
    upper_a_err: Option<&Array2<f32>>,
    input_size: usize,
    block_size: usize,
    out_block_size: usize,
    reps: usize,
) -> (Array2<f32>, Array2<f32>) {
    let num_outputs = lower_a.nrows();
    let mut lower_err = Array2::<f32>::zeros((num_outputs, input_size));
    let mut upper_err = Array2::<f32>::zeros((num_outputs, input_size));
    // reps < 2 introduces no summation rounding; leave err at zero.
    if reps < 2 {
        return (lower_err, upper_err);
    }
    // SOUNDNESS (#vnncomp-aw-soundness): the replica sum `new_A[:, i] = Σ_rep
    // A[:, src_rep]` is accumulated in f32 (see the `+=` loops in
    // `propagate_linear*`), so its rounding error is bounded by the **f32**
    // growth factor `γ_reps^f32 ≈ reps·2^-24` — NOT the f64 factor `≈ reps·2^-53`
    // (~2^29× smaller), which would UNDER-count the real f32 coefficient error.
    let gamma = crate::layers::linear::crown_single_gamma_n_f32(reps);
    for i in 0..input_size {
        let prefix = i / block_size;
        let within_block = i % block_size;
        for row in 0..num_outputs {
            let mut s_l = 0.0f64;
            let mut s_u = 0.0f64;
            let mut prop_l = 0.0f64;
            let mut prop_u = 0.0f64;
            for rep in 0..reps {
                let src = prefix * out_block_size + rep * block_size + within_block;
                s_l += (lower_a[[row, src]] as f64).abs();
                s_u += (upper_a[[row, src]] as f64).abs();
                if let Some(e) = lower_a_err {
                    prop_l += (e[[row, src]] as f64).abs();
                }
                if let Some(e) = upper_a_err {
                    prop_u += (e[[row, src]] as f64).abs();
                }
            }
            lower_err[[row, i]] = ny_tensor::next_up_f32((gamma * s_l + prop_l) as f32);
            upper_err[[row, i]] = ny_tensor::next_up_f32((gamma * s_u + prop_u) as f32);
        }
    }
    (lower_err, upper_err)
}

/// Tile layer: repeats tensor along specified axis.
///
/// Used in GQA (Grouped Query Attention) to expand KV heads to match Q heads.
/// For example, if K has shape [seq, num_kv_heads, head_dim] and we need to
/// tile by `reps` along axis 1, the output is [seq, num_kv_heads * reps, head_dim].
#[derive(Debug, Clone)]
pub struct TileLayer {
    /// Axis along which to repeat (supports negative indexing).
    pub axis: i32,
    /// Number of times to repeat.
    pub reps: usize,
    /// Input shape (required for CROWN backward propagation).
    /// Set via `set_input_shape()` before calling `propagate_linear()`.
    input_shape: Option<Vec<usize>>,
}

impl TileLayer {
    /// Create a new tile layer.
    pub fn new(axis: i32, reps: usize) -> Self {
        Self {
            axis,
            reps,
            input_shape: None,
        }
    }

    /// Set the input shape (required for CROWN backward propagation).
    pub fn set_input_shape(&mut self, shape: Vec<usize>) {
        self.input_shape = Some(shape);
    }

    /// Normalize axis to positive index given number of dimensions.
    fn normalize_axis(&self, ndim: usize) -> Result<usize> {
        super::super::common::resolve_axis_i32(self.axis, ndim, "Tile")
    }

    /// CROWN backward propagation through Tile layer.
    ///
    /// For y = tile(x, axis, reps), each input position is replicated `reps` times
    /// along the specified axis. In the backward pass, each input position receives
    /// contributions from all its replicated output positions.
    ///
    /// Math:
    /// - Forward: y[..., i*reps+r, ...] = x[..., i, ...] for r in 0..reps
    /// - Jacobian: `J[j,k] = 1` if output j is a replica of input k, else 0
    /// - Backward: new_A[:, k] = sum(A[:, j] for j in replicas_of_k)
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        let input_shape = pre_activation.shape();
        let ndim = input_shape.len();
        let axis = self.normalize_axis(ndim)?;

        if self.reps == 0 {
            return Err(NyError::InvalidSpec(
                "Tile reps must be at least 1".to_string(),
            ));
        }

        // Guard: zero-valued dimensions cause division-by-zero in block index
        // arithmetic below. (#2806)
        if input_shape.contains(&0) {
            return Err(NyError::InvalidSpec(
                "Tile: zero-valued dimension in input shape".to_string(),
            ));
        }

        if self.reps == 1 {
            // No-op: return input unchanged
            return Ok(bounds.clone());
        }

        // Compute sizes for index mapping
        let n_axis = input_shape[axis]; // Size along tile axis before tiling
        let n_axis_out = n_axis * self.reps; // Size along tile axis after tiling

        // Suffix size: product of dimensions after axis
        let suffix_size: usize =
            checked_shape_product(&input_shape[(axis + 1)..]).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Tile: suffix shape product overflows usize: {:?}",
                    &input_shape[(axis + 1)..],
                ))
            })?;

        // Total input size (flattened)
        let input_size: usize = checked_shape_product(input_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Tile: input shape product overflows usize: {:?}",
                input_shape,
            ))
        })?;

        // Total output size (flattened)
        let output_size = input_size / n_axis * n_axis_out;

        // Validate bounds dimensions
        let num_outputs = bounds.num_outputs();
        let num_current_inputs = bounds.num_inputs();

        if num_current_inputs != output_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_outputs, output_size],
                got: vec![num_outputs, num_current_inputs],
            });
        }

        // Block size for one tile copy (elements in one repetition block)
        let block_size = n_axis * suffix_size;
        let out_block_size = n_axis_out * suffix_size;

        // Build new coefficient matrices by summing contributions from all replicas
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, input_size));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, input_size));

        // For each input index, find all output indices that map to it and sum coefficients
        // Output index j maps to input index:
        //   prefix = j / out_block_size
        //   remainder = j % out_block_size
        //   within_block = remainder % block_size  (position within one tile block)
        //   input_index = prefix * block_size + within_block
        //
        // Equivalently, for each input index i:
        //   prefix = i / block_size
        //   within_block = i % block_size
        //   output indices = prefix * out_block_size + rep * block_size + within_block
        //     for rep in 0..reps

        for i in 0..input_size {
            let prefix = i / block_size;
            let within_block = i % block_size;

            for rep in 0..self.reps {
                let output_idx = prefix * out_block_size + rep * block_size + within_block;

                // Sum coefficients from this output position to input position i
                for row in 0..num_outputs {
                    new_lower_a[[row, i]] += bounds.lower_a()[[row, output_idx]];
                    new_upper_a[[row, i]] += bounds.upper_a()[[row, output_idx]];
                }
            }
        }

        // SOUND Tile coefficient error (#vnncomp-aw-soundness): γ_reps·S + prop over
        // the same `reps` summed source columns. Tile now lives in
        // `propagates_coeff_err` (query.rs), so incoming err arrives here and MUST be
        // propagated via the `prop` term inside the helper.
        let (lower_err, upper_err) = tile_backward_coeff_err(
            bounds.lower_a(),
            bounds.upper_a(),
            bounds.lower_a_err(),
            bounds.upper_a_err(),
            input_size,
            block_size,
            out_block_size,
            self.reps,
        );

        // Bias terms are unchanged (they don't depend on input positions)
        LinearBounds::new_or_conservative_with_err(
            new_lower_a,
            bounds.lower_b().clone(),
            new_upper_a,
            bounds.upper_b().clone(),
            lower_err,
            upper_err,
        )
    }

    /// Batched CROWN backward propagation through Tile layer.
    ///
    /// For y = tile(x, axis, reps), each input position is replicated `reps` times
    /// along the specified axis. In the backward pass, each input position receives
    /// contributions from all its replicated output positions (sum over replicas).
    ///
    /// A matrices: shape [...batch, out_dim, output_flat] -> [...batch, out_dim, input_flat]
    ///
    /// Supports the standard per-position batched form when tiling the last
    /// logical axis (`[..., out_dim, in_dim]`) in addition to flattened-column
    /// bounds used by broader graph shape transforms.
    ///
    /// Reference:
    /// `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/tile.py:29-47`
    /// The reference reshapes A to interleave reps dims then sums; our flattened
    /// approach is equivalent for single-axis tiling.
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        let input_shape = pre_activation.shape();
        let ndim = input_shape.len();
        let axis = self.normalize_axis(ndim)?;

        if self.reps == 0 {
            return Err(NyError::InvalidSpec(
                "Tile reps must be at least 1".to_string(),
            ));
        }

        if input_shape.contains(&0) {
            return Err(NyError::InvalidSpec(
                "Tile: zero-valued dimension in input shape".to_string(),
            ));
        }

        if self.reps == 1 {
            return BatchedLinearBounds::new_or_conservative(
                bounds.lower_a().clone(),
                bounds.lower_b().clone(),
                bounds.upper_a().clone(),
                bounds.upper_b().clone(),
                input_shape.to_vec(),
                bounds.output_shape().to_vec(),
            );
        }

        let n_axis = input_shape[axis];
        let n_axis_out = n_axis * self.reps;

        let suffix_size: usize =
            checked_shape_product(&input_shape[(axis + 1)..]).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Tile batched CROWN: suffix shape product overflows: {:?}",
                    &input_shape[(axis + 1)..],
                ))
            })?;

        let flat_input_size: usize = checked_shape_product(input_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Tile batched CROWN: input shape product overflows: {:?}",
                input_shape,
            ))
        })?;

        let flat_output_size = flat_input_size / n_axis * n_axis_out;

        let a_shape = bounds.lower_a().shape();
        let a_ndim = a_shape.len();
        if a_ndim < 2 {
            return Err(NyError::InvalidSpec(
                "Tile batched CROWN: A matrices must have at least 2 dimensions".to_string(),
            ));
        }
        let in_dim = a_shape[a_ndim - 1];
        let last_axis_mode = axis + 1 == ndim && in_dim == n_axis_out;
        let (input_size, output_size) = if last_axis_mode {
            (n_axis, n_axis_out)
        } else if in_dim == flat_output_size {
            (flat_input_size, flat_output_size)
        } else {
            return Err(NyError::InvalidSpec(format!(
                "Tile batched CROWN expects last A dim {} (last-axis mode) or {} (flattened), got {}",
                n_axis_out, flat_output_size, in_dim
            )));
        };

        let block_size = n_axis * suffix_size;
        let out_block_size = n_axis_out * suffix_size;

        let mut new_a_shape = a_shape.to_vec();
        new_a_shape[a_ndim - 1] = input_size;
        let mut new_lower_a = ArrayD::<f32>::zeros(IxDyn(&new_a_shape));
        let mut new_upper_a = ArrayD::<f32>::zeros(IxDyn(&new_a_shape));

        let outer_size: usize = checked_shape_product(&a_shape[..a_ndim - 1]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Tile batched CROWN: outer shape product overflows: {:?}",
                &a_shape[..a_ndim - 1],
            ))
        })?;

        let flat_lower = contiguous_flat_slice(bounds.lower_a());
        let flat_upper = contiguous_flat_slice(bounds.upper_a());
        // Incoming certified err (if any) for the SOUND `prop` term
        // (#vnncomp-aw-soundness). Normally None here: in the BATCHED pipeline Tile is
        // a coeff-err CARRIER (dispatch.rs), so the dispatcher carries any incoming err
        // via a separate carrier re-run (ADDED to this fresh err) and this `real` run
        // sees err-free bounds. Read defensively anyway so a non-empty incoming err can
        // never be under-counted on any path that reaches here with err attached.
        let in_lower_err = bounds
            .lower_a_err
            .as_ref()
            .map(|e| contiguous_flat_slice(e));
        let in_upper_err = bounds
            .upper_a_err
            .as_ref()
            .map(|e| contiguous_flat_slice(e));
        // reps < 2 sums nothing → no fresh rounding error (handled by zeros below).
        // SOUNDNESS (#vnncomp-aw-soundness): the batched replica sum
        // `new_a[.., i] += flat_*[.., output_idx]` is accumulated in f32, so its
        // rounding error must be certified with the **f32** growth factor
        // `γ_reps^f32 ≈ reps·2^-24` (the f64 factor under-counts by ~2^29×).
        let gamma = if self.reps >= 2 {
            crate::layers::linear::crown_single_gamma_n_f32(self.reps)
        } else {
            0.0
        };
        let mut lower_err_flat = vec![0.0f32; outer_size * input_size];
        let mut upper_err_flat = vec![0.0f32; outer_size * input_size];
        let new_lower_flat = contiguous_flat_slice_mut(&mut new_lower_a)?;
        let new_upper_flat = contiguous_flat_slice_mut(&mut new_upper_a)?;

        // For each row (outer dims), sum coefficients from all replicas.
        // Input position i maps to output positions:
        //   prefix * out_block_size + rep * block_size + within_block
        // for rep in 0..reps, where prefix = i / block_size, within_block = i % block_size
        for row in 0..outer_size {
            let old_base = row * output_size;
            let new_base = row * input_size;
            for i in 0..input_size {
                let prefix = i / block_size;
                let within_block = i % block_size;
                let mut s = 0.0f64; // abs-sum over replicas (lower)
                let mut s_u = 0.0f64; // abs-sum over replicas (upper)
                let mut prop_l = 0.0f64;
                let mut prop_u = 0.0f64;
                for rep in 0..self.reps {
                    let output_idx = prefix * out_block_size + rep * block_size + within_block;
                    let lv = flat_lower[old_base + output_idx];
                    let uv = flat_upper[old_base + output_idx];
                    new_lower_flat[new_base + i] += lv;
                    new_upper_flat[new_base + i] += uv;
                    s += (lv as f64).abs();
                    s_u += (uv as f64).abs();
                    if let Some(e) = in_lower_err.as_ref() {
                        prop_l += (e[old_base + output_idx] as f64).abs();
                    }
                    if let Some(e) = in_upper_err.as_ref() {
                        prop_u += (e[old_base + output_idx] as f64).abs();
                    }
                }
                lower_err_flat[new_base + i] = ny_tensor::next_up_f32((gamma * s + prop_l) as f32);
                upper_err_flat[new_base + i] =
                    ny_tensor::next_up_f32((gamma * s_u + prop_u) as f32);
            }
        }

        debug!(
            "Tile batched CROWN: reduced {} -> {} columns across {} rows (axis={}, reps={}, last_axis_mode={})",
            output_size, input_size, outer_size, axis, self.reps, last_axis_mode
        );

        let mut out = BatchedLinearBounds::new_or_conservative(
            new_lower_a,
            bounds.lower_b().clone(),
            new_upper_a,
            bounds.upper_b().clone(),
            input_shape.to_vec(),
            bounds.output_shape().to_vec(),
        )?;
        // SOUND Tile coefficient error (#vnncomp-aw-soundness): γ_reps·S (+ prop)
        // over the same `reps` summed source columns, computed in the loop above.
        let lower_err =
            ArrayD::from_shape_vec(IxDyn(&new_a_shape), lower_err_flat).map_err(|_| {
                NyError::InvalidSpec("Tile batched CROWN: cannot reshape lower err".to_string())
            })?;
        let upper_err =
            ArrayD::from_shape_vec(IxDyn(&new_a_shape), upper_err_flat).map_err(|_| {
                NyError::InvalidSpec("Tile batched CROWN: cannot reshape upper err".to_string())
            })?;
        out.set_coeff_err(lower_err, upper_err);
        Ok(out)
    }
}

impl BoundPropagation for TileLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let shape = input.shape();
        let ndim = shape.len();
        let axis = self.normalize_axis(ndim)?;

        if self.reps == 0 {
            return Err(NyError::InvalidSpec(
                "Tile reps must be at least 1".to_string(),
            ));
        }

        // Guard: zero-valued dimensions cause degenerate concatenation. (#2806)
        if shape.contains(&0) {
            return Err(NyError::InvalidSpec(
                "Tile: zero-valued dimension in input shape".to_string(),
            ));
        }

        if self.reps == 1 {
            // No-op: return input unchanged
            return Ok(input.clone());
        }

        // Compute output shape: multiply axis dimension by reps
        let mut output_shape = shape.to_vec();
        output_shape[axis] *= self.reps;

        // Tile the lower and upper bounds
        // Strategy: concatenate `reps` copies along the axis
        use ndarray::concatenate;

        let lower_views: Vec<_> = (0..self.reps).map(|_| input.lower().view()).collect();
        let upper_views: Vec<_> = (0..self.reps).map(|_| input.upper().view()).collect();

        let lower_tiled = concatenate(Axis(axis), &lower_views).map_err(|e| {
            NyError::InvalidSpec(format!("Tile lower bound concatenation failed: {}", e))
        })?;

        let upper_tiled = concatenate(Axis(axis), &upper_views).map_err(|e| {
            NyError::InvalidSpec(format!("Tile upper bound concatenation failed: {}", e))
        })?;

        BoundedTensor::new(lower_tiled, upper_tiled)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // For CROWN backward propagation through Tile:
        // Tile replicates data along an axis: output[..., i, ...] = input[..., i % N, ...]
        // where N is the input size along the tile axis and output size is N * reps.
        //
        // In the backward pass, each input position receives contributions from all
        // its replicated output positions: new_A[:, input_i] = sum_j A[:, output_j]
        // for all output_j that map to input_i.

        // Get input shape (required for CROWN)
        let input_shape = self.input_shape.as_ref().ok_or_else(|| {
            NyError::UnsupportedConfiguration(
                "Tile CROWN requires input_shape to be set. Use set_input_shape().".to_string(),
            )
        })?;

        let ndim = input_shape.len();
        let axis = self.normalize_axis(ndim)?;

        if self.reps == 0 {
            return Err(NyError::InvalidSpec(
                "Tile reps must be at least 1".to_string(),
            ));
        }

        // Guard: zero-valued dimensions cause division-by-zero in block index
        // arithmetic below. (#2806)
        if input_shape.contains(&0) {
            return Err(NyError::InvalidSpec(
                "Tile: zero-valued dimension in input shape".to_string(),
            ));
        }

        if self.reps == 1 {
            // No-op: return input unchanged
            return Ok(Cow::Borrowed(bounds));
        }

        // Compute sizes for index mapping
        let n_axis = input_shape[axis]; // Size along tile axis before tiling
        let n_axis_out = n_axis * self.reps; // Size along tile axis after tiling

        // Suffix size: product of dimensions after axis
        let suffix_size: usize =
            checked_shape_product(&input_shape[(axis + 1)..]).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Tile: suffix shape product overflows usize: {:?}",
                    &input_shape[(axis + 1)..],
                ))
            })?;

        // Total input size (flattened)
        let input_size: usize = checked_shape_product(input_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Tile: input shape product overflows usize: {:?}",
                input_shape,
            ))
        })?;

        // Total output size (flattened)
        let output_size = input_size / n_axis * n_axis_out;

        // Validate bounds dimensions
        let num_outputs = bounds.num_outputs();
        let num_current_inputs = bounds.num_inputs();

        if num_current_inputs != output_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_outputs, output_size],
                got: vec![num_outputs, num_current_inputs],
            });
        }

        // Block size for one tile copy
        let block_size = n_axis * suffix_size;
        let out_block_size = n_axis_out * suffix_size;

        // Build new coefficient matrices by summing contributions from all replicas
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, input_size));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, input_size));

        // For each input index, sum coefficients from all its replica outputs
        for i in 0..input_size {
            let prefix = i / block_size;
            let within_block = i % block_size;

            for rep in 0..self.reps {
                let output_idx = prefix * out_block_size + rep * block_size + within_block;

                // Sum coefficients from this output position to input position i
                for row in 0..num_outputs {
                    new_lower_a[[row, i]] += bounds.lower_a()[[row, output_idx]];
                    new_upper_a[[row, i]] += bounds.upper_a()[[row, output_idx]];
                }
            }
        }

        // SOUND Tile coefficient error (#vnncomp-aw-soundness): γ_reps·S + prop over
        // the same `reps` summed source columns. Tile lives in
        // `propagates_coeff_err` (query.rs), so incoming err is propagated here.
        let (lower_err, upper_err) = tile_backward_coeff_err(
            bounds.lower_a(),
            bounds.upper_a(),
            bounds.lower_a_err(),
            bounds.upper_a_err(),
            input_size,
            block_size,
            out_block_size,
            self.reps,
        );

        // Bias terms are unchanged (they don't depend on input positions)
        Ok(Cow::Owned(LinearBounds::new_or_conservative_with_err(
            new_lower_a,
            bounds.lower_b().clone(),
            new_upper_a,
            bounds.upper_b().clone(),
            lower_err,
            upper_err,
        )?))
    }
}

#[cfg(test)]
mod tests;
