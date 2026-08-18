// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shape transforms for batched linear bounds (flatten, tile, dimension queries).
//!
//! Extracted from `mod.rs` as part of #4212.

use super::BatchedLinearBounds;
#[cfg(test)]
use ndarray::IxDyn;
use ndarray::{Array2, ArrayD};
use ny_core::{checked_shape_product, NyError, Result};
use std::mem::size_of;

impl BatchedLinearBounds {
    /// Output dimension (last dimension of output shape).
    pub fn out_dim(&self) -> usize {
        *self.output_shape.last().unwrap_or(&1)
    }

    /// Input dimension (last dimension of input shape).
    pub fn in_dim(&self) -> usize {
        *self.input_shape.last().unwrap_or(&1)
    }

    /// Total logical heap payload used by this batched bounds struct, in bytes.
    ///
    /// Includes both A-matrices, both bias vectors, and any certified
    /// per-coefficient error carriers across all batch dimensions, plus the
    /// logical input/output shape vectors. ndarray does not expose backing-vector
    /// capacity, so allocator slack is intentionally not claimed here.
    pub fn memory_bytes(&self) -> usize {
        let array_elements = self
            .lower_a
            .len()
            .saturating_add(self.upper_a.len())
            .saturating_add(self.lower_b.len())
            .saturating_add(self.upper_b.len())
            .saturating_add(self.lower_a_err.as_ref().map_or(0, |error| error.len()))
            .saturating_add(self.upper_a_err.as_ref().map_or(0, |error| error.len()));
        array_elements
            .saturating_mul(size_of::<f32>())
            .saturating_add(
                self.input_shape
                    .len()
                    .saturating_add(self.output_shape.len())
                    .saturating_mul(size_of::<usize>()),
            )
    }

    /// Flatten batched bounds to a block-diagonal 2D representation.
    ///
    /// Converts `[batch..., out_dim, in_dim]` coefficient matrices into
    /// `[total_out, total_in]` where `total_out = product(batch) * out_dim`
    /// and `total_in = product(batch) * in_dim`. Each batch position becomes
    /// a block on the diagonal; off-diagonal blocks are zero.
    ///
    /// This is needed for attention CROWN backward: `BilinearCrown` constructs
    /// flat McCormick matrices `[m*n, m*k]` while downstream identity bounds
    /// are batched `[m, n, n]`. Flattening the downstream to `[m*n, m*n]`
    /// block-diagonal makes the shapes compatible for `compose()`.
    ///
    /// The bias is simply concatenated: `[batch..., out_dim]` → `[total_out]`.
    pub fn flatten_to_block_diagonal(&self) -> Result<BatchedLinearBounds> {
        let a_shape = self.lower_a.shape();
        let ndim = a_shape.len();

        // Already flat (2D) — no-op
        if ndim <= 2 {
            return Ok(self.clone());
        }

        let out_dim = a_shape[ndim - 2];
        let in_dim = a_shape[ndim - 1];
        let batch_dims = &a_shape[..ndim - 2];
        let batch_size: usize = checked_shape_product(batch_dims)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "flatten_to_block_diagonal: batch dims overflow: {batch_dims:?}",
                ))
            })?
            .max(1);

        let total_out = batch_size * out_dim;
        let total_in = batch_size * in_dim;

        // Build block-diagonal coefficient matrices.
        // Each batch position i places its [out_dim, in_dim] block at
        // rows [i*out_dim..(i+1)*out_dim], cols [i*in_dim..(i+1)*in_dim].
        let build_block_diag = |a: &ArrayD<f32>| -> Result<ArrayD<f32>> {
            let a3 = a
                .view()
                .into_shape_with_order((batch_size, out_dim, in_dim))
                .map_err(|e| {
                    NyError::InvalidSpec(format!(
                        "flatten_to_block_diagonal: reshape A to 3D failed: {e}"
                    ))
                })?;
            let mut flat = Array2::<f32>::zeros((total_out, total_in));
            for b in 0..batch_size {
                for r in 0..out_dim {
                    for c in 0..in_dim {
                        flat[[b * out_dim + r, b * in_dim + c]] = a3[[b, r, c]];
                    }
                }
            }
            Ok(flat.into_dyn())
        };

        let flat_lower_a = build_block_diag(&self.lower_a)?;
        let flat_upper_a = build_block_diag(&self.upper_a)?;

        // Bias: concatenate batched [batch..., out_dim] to flat [total_out]
        let flat_lower_b = self
            .lower_b
            .view()
            .into_shape_with_order(total_out)
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "flatten_to_block_diagonal: reshape lower_b failed: {e}"
                ))
            })?
            .into_dyn()
            .to_owned();
        let flat_upper_b = self
            .upper_b
            .view()
            .into_shape_with_order(total_out)
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "flatten_to_block_diagonal: reshape upper_b failed: {e}"
                ))
            })?
            .into_dyn()
            .to_owned();

        BatchedLinearBounds::new(
            flat_lower_a,
            flat_lower_b,
            flat_upper_a,
            flat_upper_b,
            vec![total_in],
            vec![total_out],
        )
    }

    /// Tile unbatched (2D) bounds to match a target batch shape.
    ///
    /// Given current A shape `[out_dim, in_dim]`, creates
    /// `[batch_dims..., out_dim, in_dim]` by repeating the same matrix for
    /// each batch position. This enables composition with batched downstream
    /// bounds that carry `[batch, heads, ...]` dimensions.
    ///
    /// Superseded in production by `BilinearRelaxation` (#286 Approach A), which
    /// composes per-batch coefficients directly without tiling. Retained for tests.
    ///
    /// # Errors
    /// Returns an error if the source bounds are not 2D or if the batch product
    /// overflows `usize`.
    #[cfg(test)]
    pub fn tile_to_batch(&self, batch_shape: &[usize]) -> Result<Self> {
        let a_shape = self.lower_a.shape();
        if a_shape.len() != 2 {
            return Err(NyError::InvalidSpec(format!(
                "tile_to_batch requires 2D source, got {}D",
                a_shape.len()
            )));
        }
        if batch_shape.is_empty() {
            return Ok(self.clone());
        }

        let out_dim = a_shape[0];
        let in_dim = a_shape[1];
        let batch_size: usize = checked_shape_product(batch_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "tile_to_batch: batch dims {batch_shape:?} overflow usize",
            ))
        })?;

        // Tile A matrices: repeat [out_dim, in_dim] -> [batch..., out_dim, in_dim]
        let a_slice_len = out_dim * in_dim;
        let mut new_a_shape: Vec<usize> = batch_shape.to_vec();
        new_a_shape.push(out_dim);
        new_a_shape.push(in_dim);

        let mut lower_a_data = Vec::with_capacity(batch_size * a_slice_len);
        let lower_a_slice: Vec<f32> = self.lower_a.iter().copied().collect();
        for _ in 0..batch_size {
            lower_a_data.extend_from_slice(&lower_a_slice);
        }
        let mut upper_a_data = Vec::with_capacity(batch_size * a_slice_len);
        let upper_a_slice: Vec<f32> = self.upper_a.iter().copied().collect();
        for _ in 0..batch_size {
            upper_a_data.extend_from_slice(&upper_a_slice);
        }

        // Tile biases: repeat [out_dim] -> [batch..., out_dim]
        let mut new_b_shape: Vec<usize> = batch_shape.to_vec();
        new_b_shape.push(out_dim);

        let mut lower_b_data = Vec::with_capacity(batch_size * out_dim);
        let lower_b_slice: Vec<f32> = self.lower_b.iter().copied().collect();
        for _ in 0..batch_size {
            lower_b_data.extend_from_slice(&lower_b_slice);
        }
        let mut upper_b_data = Vec::with_capacity(batch_size * out_dim);
        let upper_b_slice: Vec<f32> = self.upper_b.iter().copied().collect();
        for _ in 0..batch_size {
            upper_b_data.extend_from_slice(&upper_b_slice);
        }

        // Prepend batch dims to input/output shapes
        let mut new_input_shape: Vec<usize> = batch_shape.to_vec();
        new_input_shape.extend_from_slice(&self.input_shape);
        let mut new_output_shape: Vec<usize> = batch_shape.to_vec();
        new_output_shape.extend_from_slice(&self.output_shape);

        // KEEP unchecked: source bounds were already validated, and tiling only
        // duplicates existing entries into new shapes.
        Ok(Self::from_parts_unchecked(
            ArrayD::from_shape_vec(IxDyn(&new_a_shape), lower_a_data).map_err(|e| {
                NyError::InternalError(format!("tile_to_batch: lower_a reshape: {e}"))
            })?,
            ArrayD::from_shape_vec(IxDyn(&new_b_shape), lower_b_data).map_err(|e| {
                NyError::InternalError(format!("tile_to_batch: lower_b reshape: {e}"))
            })?,
            ArrayD::from_shape_vec(IxDyn(&new_a_shape), upper_a_data).map_err(|e| {
                NyError::InternalError(format!("tile_to_batch: upper_a reshape: {e}"))
            })?,
            ArrayD::from_shape_vec(IxDyn(&new_b_shape), upper_b_data).map_err(|e| {
                NyError::InternalError(format!("tile_to_batch: upper_b reshape: {e}"))
            })?,
            new_input_shape,
            new_output_shape,
        ))
    }
}
