// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Transpose layer: permutes tensor axes for bound propagation.

use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;
use tracing::debug;

use super::super::common::BoundPropagation;
use crate::{contiguous_flat_slice, contiguous_flat_slice_mut, BatchedLinearBounds, LinearBounds};

/// Transpose layer: permutes tensor axes.
///
/// For attention patterns, this is used for the K^T in Q @ K^T.
/// For example, transposing a (batch, seq, heads, dim) tensor to (batch, heads, seq, dim).
#[derive(Debug, Clone)]
pub struct TransposeLayer {
    /// Axes permutation. For 2D transpose, this is [1, 0].
    /// For batched transpose of last two dims in 3D: [0, 2, 1].
    pub axes: Vec<usize>,
    /// Input shape (required for CROWN backward propagation).
    /// Set via `set_input_shape()` before calling `propagate_linear()`.
    input_shape: Option<Vec<usize>>,
}

impl TransposeLayer {
    /// Create a new transpose layer with specified axes permutation.
    pub fn new(axes: Vec<usize>) -> Self {
        Self {
            axes,
            input_shape: None,
        }
    }

    /// Create a simple 2D transpose (swap last two dimensions).
    pub fn transpose_2d() -> Self {
        Self {
            axes: vec![1, 0],
            input_shape: None,
        }
    }

    /// Create a batched transpose that swaps the last two dimensions.
    /// For 3D input (batch, m, n), produces (batch, n, m).
    /// For 4D input (a, b, m, n), produces (a, b, n, m).
    pub fn batched_transpose() -> Self {
        // Axes will be computed dynamically based on input dimension
        Self {
            axes: Vec::new(),
            input_shape: None,
        }
    }

    /// Set the input shape for CROWN backward propagation.
    pub fn set_input_shape(&mut self, shape: Vec<usize>) {
        self.input_shape = Some(shape);
    }

    /// Compute the flat index mapping from output (transposed) indices to input indices.
    /// Returns a vector where mapping[output_flat_idx] = input_flat_idx.
    fn compute_index_mapping(&self, input_shape: &[usize], perm: &[usize]) -> Result<Vec<usize>> {
        // Compute output shape
        let output_shape: Vec<usize> = perm.iter().map(|&p| input_shape[p]).collect();
        let total_elems: usize = checked_shape_product(input_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Transpose: input shape product overflows usize: {:?}",
                input_shape,
            ))
        })?;

        // Compute inverse permutation
        let mut inv_perm = vec![0usize; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            inv_perm[p] = i;
        }

        // Compute strides for input (row-major)
        let mut input_strides = vec![1usize; input_shape.len()];
        for i in (0..input_shape.len() - 1).rev() {
            input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
        }

        // Compute strides for output (row-major)
        let mut output_strides = vec![1usize; output_shape.len()];
        for i in (0..output_shape.len() - 1).rev() {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
        }

        // Build mapping: for each output flat index, find corresponding input flat index
        let mut mapping = vec![0usize; total_elems];
        for (out_flat, mapping_entry) in mapping.iter_mut().enumerate() {
            // Convert out_flat to output multi-index
            let mut out_idx = vec![0usize; output_shape.len()];
            let mut remainder = out_flat;
            for (i, &stride) in output_strides.iter().enumerate() {
                out_idx[i] = remainder / stride;
                remainder %= stride;
            }

            // Apply inverse permutation to get input multi-index
            let mut in_idx = vec![0usize; input_shape.len()];
            for (i, &ip) in inv_perm.iter().enumerate() {
                in_idx[i] = out_idx[ip];
            }

            // Convert input multi-index to flat index
            let mut in_flat = 0usize;
            for (i, &idx) in in_idx.iter().enumerate() {
                in_flat += idx * input_strides[i];
            }

            *mapping_entry = in_flat;
        }

        Ok(mapping)
    }

    /// Resolve the permutation given input dimensionality.
    fn resolve_perm(&self, ndim: usize) -> Result<Vec<usize>> {
        if self.axes.is_empty() {
            // Batched transpose: swap last two dimensions
            if ndim < 2 {
                return Err(NyError::InvalidSpec(
                    "Transpose requires at least 2D input".to_string(),
                ));
            }
            let mut p: Vec<usize> = (0..ndim).collect();
            p.swap(ndim - 2, ndim - 1);
            Ok(p)
        } else {
            // Explicit permutation. It may be rank-consistent already, or it may
            // carry extra leading axes that were squeezed out (e.g. ny strips the
            // batch dimension before propagation, so an ONNX perm authored for the
            // batched rank arrives here longer than the unbatched `ndim`). It may
            // also be a meaningless over-ranked perm on a rank-≤1 tensor (transpose
            // of a rank-≤1 tensor is the identity). `normalize_transpose_perm_for_rank`
            // resolves all of these into a valid permutation of `0..ndim` that
            // preserves the transpose's element ordering, or returns `None` when no
            // equivalence can be proven.
            normalize_transpose_perm_for_rank(&self.axes, ndim).ok_or(NyError::ShapeMismatch {
                expected: vec![ndim],
                got: vec![self.axes.len()],
            })
        }
    }
}

/// Normalize an ONNX `Transpose` `perm` so that it is a valid permutation of
/// `0..rank` *and* computes the same logical transpose as the original `perm`
/// would on the corresponding full-rank tensor.
///
/// ny-propagate operates on **unbatched** tensors (the synthetic leading batch
/// dimension is stripped) and graph normalization can collapse axes, so a
/// `Transpose` whose ONNX `perm` was authored for rank `k` frequently arrives
/// applied to a tensor of smaller rank `r < k`. Likewise, an exporter can emit
/// a `Transpose` whose `perm` references axes that do not exist for a rank-≤1
/// tensor. Both forms must be reconciled with the *actual* tensor rank, never
/// applied verbatim (which panics / is rejected by ONNX Runtime shape
/// inference with `Invalid attribute perm`).
///
/// Returns `Some(perm)` (a permutation of `0..rank`) or `None` when no
/// rank-consistent rewrite can be *proven* equivalent — the caller must then
/// fail closed rather than guess.
///
/// Soundness:
/// - `perm.len() == rank`: already consistent; returned verbatim after
///   validating it is a genuine permutation of `0..rank`.
/// - `rank <= 1`: a rank-0 or rank-1 tensor has no pair of axes to swap, so the
///   transpose is necessarily the identity. The only value-preserving result is
///   the identity perm `0..rank`; we return it regardless of the (meaningless)
///   original perm length. (Element ordering is provably unchanged.)
/// - `perm.len() > rank` with `d = perm.len() - rank` dropped leading axes:
///   valid **iff** the full `perm` is a permutation of `0..perm.len()` (so the
///   `d` smallest labels `0..d` — the dropped/leading dims — are exactly the
///   entries removed, and the survivors are `d..perm.len()`). The rewrite drops
///   the entries labelled `< d` and subtracts `d` from the rest, yielding a
///   permutation of `0..rank`. This generalizes the established batch-strip
///   convention (#3602, the `d == 1` case) to multiple dropped leading dims and
///   preserves the surviving axes' element ordering because the dropped leading
///   axes were pure passthroughs.
/// - otherwise: `None`.
pub fn normalize_transpose_perm_for_rank(perm: &[usize], rank: usize) -> Option<Vec<usize>> {
    // Rank-≤1: transpose is the identity (no axis pair to reorder).
    if rank <= 1 {
        return Some((0..rank).collect());
    }

    if perm.len() == rank {
        // Validate it is a genuine permutation of 0..rank before trusting it.
        return is_permutation_of_range(perm, rank).then(|| perm.to_vec());
    }

    if perm.len() > rank {
        let d = perm.len() - rank;
        // The dropped leading axes are the `d` smallest labels `0..d`. Requiring
        // the full perm to be a permutation of `0..perm.len()` guarantees those
        // `d` labels are present and the survivors (entries `>= d`) are exactly
        // `d..perm.len()`.
        if !is_permutation_of_range(perm, perm.len()) {
            return None;
        }
        let adjusted: Vec<usize> = perm.iter().filter(|&&a| a >= d).map(|&a| a - d).collect();
        // After subtracting `d`, `adjusted` is a permutation of `0..rank` exactly
        // when the survivors were `d..perm.len()` (guaranteed above).
        if adjusted.len() == rank && is_permutation_of_range(&adjusted, rank) {
            return Some(adjusted);
        }
        return None;
    }

    // perm.len() < rank (and rank >= 2): cannot extend a shorter perm soundly.
    None
}

/// Returns `true` iff `values` is a permutation of `0..len` (each of the `len`
/// labels appears exactly once).
fn is_permutation_of_range(values: &[usize], len: usize) -> bool {
    if values.len() != len {
        return false;
    }
    let mut seen = vec![false; len];
    for &v in values {
        match seen.get_mut(v) {
            Some(slot) if !*slot => *slot = true,
            _ => return false,
        }
    }
    true
}

impl BoundPropagation for TransposeLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let ndim = input.shape().len();
        let perm = self.resolve_perm(ndim)?;

        // Apply permutation to lower and upper bounds
        // Use as_standard_layout().into_owned() to ensure contiguous memory layout
        // after the permutation, since permuted_axes only changes strides
        let lower_t = input
            .lower()
            .clone()
            .permuted_axes(perm.clone())
            .as_standard_layout()
            .into_owned();
        let upper_t = input
            .upper()
            .clone()
            .permuted_axes(perm)
            .as_standard_layout()
            .into_owned();

        // Pure layout op (axis permutation): value-preserving, so infinite bounds
        // pass through soundly. Allow `±inf` to flow without tripping the NaN
        // firewall; NaN is still rejected.
        BoundedTensor::new_allow_infinite(lower_t, upper_t)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // For CROWN backward propagation through Transpose:
        // If we have bounds A @ y + b where y is the transposed output,
        // and y = Transpose(x), we need to compute A' such that A @ y = A' @ x.
        //
        // The transpose permutes elements, so the columns of A need to be
        // reordered according to the inverse permutation.

        let input_shape = match &self.input_shape {
            Some(shape) => shape.clone(),
            None => {
                return Err(NyError::InvalidSpec(
                    "Transpose CROWN backward requires input_shape to be set via set_input_shape()"
                        .to_string(),
                ));
            }
        };

        let ndim = input_shape.len();
        let perm = self.resolve_perm(ndim)?;

        // Compute the index mapping: mapping[output_flat] = input_flat
        let mapping = self.compute_index_mapping(&input_shape, &perm)?;
        let total_elems = mapping.len();

        // Check bounds dimensions match
        if bounds.num_inputs() != total_elems {
            return Err(NyError::ShapeMismatch {
                expected: vec![total_elems],
                got: vec![bounds.num_inputs()],
            });
        }

        debug!(
            "Transpose CROWN: {} bound outputs over {} layer inputs (permuting columns)",
            bounds.num_outputs(),
            bounds.num_inputs()
        );

        // Permute columns of the coefficient matrices
        // New column j comes from old column mapping[j] (i.e., where output j maps to in input)
        // But we need the inverse: for input position i, which output position maps to it?
        // mapping[out] = in, so we need inv_mapping[in] = out
        let mut inv_mapping = vec![0usize; total_elems];
        for (out, &in_idx) in mapping.iter().enumerate() {
            inv_mapping[in_idx] = out;
        }

        // Create new coefficient matrices with permuted columns
        let num_outputs = bounds.num_outputs();
        let mut new_lower_a = ndarray::Array2::zeros((num_outputs, total_elems));
        let mut new_upper_a = ndarray::Array2::zeros((num_outputs, total_elems));

        for in_col in 0..total_elems {
            let out_col = inv_mapping[in_col];
            for row in 0..num_outputs {
                new_lower_a[[row, in_col]] = bounds.lower_a()[[row, out_col]];
                new_upper_a[[row, in_col]] = bounds.upper_a()[[row, out_col]];
            }
        }

        Ok(Cow::Owned(LinearBounds::new_or_conservative(
            new_lower_a,
            bounds.lower_b().clone(),
            new_upper_a,
            bounds.upper_b().clone(),
        )?))
    }
}

impl TransposeLayer {
    /// Batched CROWN backward propagation through Transpose.
    ///
    /// For CROWN backward: if y = transpose(x), and we have bounds A @ y + b,
    /// we need A' @ x + b where A' has columns permuted by the inverse of
    /// the transpose permutation.
    ///
    /// For flattened bounds (`in_dim == total_elems`), this permutes columns
    /// directly. For grouped multi-dimensional batched bounds, it first flattens
    /// the coefficients to an exact block-diagonal `[total_out, total_in]`
    /// representation, then applies the same flat column permutation. The result
    /// stays flat-grouped because transpose can swap the linear axis with a
    /// grouped axis.
    ///
    /// Reference: same column permutation as non-batched `propagate_linear` path.
    #[inline]
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        let input_shape = match &self.input_shape {
            Some(shape) => shape.clone(),
            None => {
                return Err(NyError::InvalidSpec(
                    "Transpose batched CROWN backward requires input_shape to be set via set_input_shape()"
                        .to_string(),
                ));
            }
        };

        let ndim = input_shape.len();
        let perm = self.resolve_perm(ndim)?;

        // Compute the index mapping: mapping[output_flat] = input_flat
        let mapping = self.compute_index_mapping(&input_shape, &perm)?;
        let total_elems = mapping.len();

        // The last dimension of the A matrices is in_dim (the columns). If the
        // incoming bounds are still grouped by batch position, flatten them to an
        // exact block-diagonal matrix first so the existing flat column
        // permutation remains correct.
        let a_shape = bounds.lower_a.shape();
        let in_dim = a_shape[a_shape.len() - 1];
        let flat_bounds = if in_dim == total_elems {
            Cow::Borrowed(bounds)
        } else {
            debug!(
                "Transpose batched CROWN: flattening grouped bounds before exact column permutation \
                 (in_dim={}, total_elems={})",
                in_dim, total_elems
            );
            Cow::Owned(bounds.flatten_to_block_diagonal()?)
        };

        let flat_a_shape = flat_bounds.lower_a.shape();
        let flat_in_dim = flat_a_shape[flat_a_shape.len() - 1];
        if flat_in_dim != total_elems {
            return Err(NyError::ShapeMismatch {
                expected: vec![total_elems],
                got: vec![flat_in_dim],
            });
        }

        // Build inverse mapping: inv_mapping[input_flat] = output_flat
        let mut inv_mapping = vec![0usize; total_elems];
        for (out, &in_idx) in mapping.iter().enumerate() {
            inv_mapping[in_idx] = out;
        }

        // Permute columns (last dimension) of the coefficient matrices.
        // For each position [...batch, row, col], the new value at col=c
        // comes from the old value at col=inv_mapping[c].
        let mut new_lower_a = flat_bounds.lower_a.clone();
        let mut new_upper_a = flat_bounds.upper_a.clone();

        // Iterate over all elements, permuting only the last dimension.
        // Total elements excluding last dim = product of all dims except last.
        let outer_size: usize = checked_shape_product(&flat_a_shape[..flat_a_shape.len() - 1])
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Transpose batched CROWN: outer shape product overflows usize: {:?}",
                    &flat_a_shape[..flat_a_shape.len() - 1],
                ))
            })?;

        // Reshape to 2D: [outer, in_dim] for easier column permutation
        let flat_lower = contiguous_flat_slice(&flat_bounds.lower_a);
        let flat_upper = contiguous_flat_slice(&flat_bounds.upper_a);

        let new_lower_flat = contiguous_flat_slice_mut(&mut new_lower_a)?;
        let new_upper_flat = contiguous_flat_slice_mut(&mut new_upper_a)?;

        for row in 0..outer_size {
            let base = row * total_elems;
            for in_col in 0..total_elems {
                let out_col = inv_mapping[in_col];
                new_lower_flat[base + in_col] = flat_lower[base + out_col];
                new_upper_flat[base + in_col] = flat_upper[base + out_col];
            }
        }

        debug!(
            "Transpose batched CROWN: permuted {} columns across {} positions",
            total_elems, outer_size
        );

        // Phase 4 audit: per-layer column permutation — no NaN introduction.
        BatchedLinearBounds::new_or_conservative(
            new_lower_a,
            flat_bounds.lower_b.clone(),
            new_upper_a,
            flat_bounds.upper_b.clone(),
            flat_bounds.input_shape.clone(),
            flat_bounds.output_shape.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::common::BoundPropagation;
    use crate::{BatchedLinearBounds, LinearBounds};
    use ndarray::{ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    // ── Construction ───────────────────────────────────────────────────

    #[ntest::timeout(5000)]
    #[test]
    fn test_transpose_2d_creates_correct_axes() {
        let t = TransposeLayer::transpose_2d();
        assert_eq!(t.axes, vec![1, 0]);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_batched_transpose_has_empty_axes() {
        let t = TransposeLayer::batched_transpose();
        assert!(t.axes.is_empty());
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_new_with_explicit_axes() {
        let t = TransposeLayer::new(vec![2, 0, 1]);
        assert_eq!(t.axes, vec![2, 0, 1]);
    }

    // ── resolve_perm ───────────────────────────────────────────────────

    #[ntest::timeout(5000)]
    #[test]
    fn test_resolve_perm_explicit_match() {
        let t = TransposeLayer::new(vec![1, 0]);
        assert_eq!(t.resolve_perm(2).unwrap(), vec![1, 0]);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_resolve_perm_batched_transpose_2d() {
        let t = TransposeLayer::batched_transpose();
        assert_eq!(t.resolve_perm(2).unwrap(), vec![1, 0]);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_resolve_perm_batched_transpose_3d() {
        let t = TransposeLayer::batched_transpose();
        assert_eq!(t.resolve_perm(3).unwrap(), vec![0, 2, 1]);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_resolve_perm_batch_dim_squeeze() {
        // axes [0, 2, 1] for 2D input → [1, 0] (shift by -1, skip axis 0)
        let t = TransposeLayer::new(vec![0, 2, 1]);
        assert_eq!(t.resolve_perm(2).unwrap(), vec![1, 0]);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_resolve_perm_errors_on_mismatch() {
        let t = TransposeLayer::new(vec![2, 1, 0]);
        assert!(t.resolve_perm(4).is_err());
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_resolve_perm_errors_for_1d_batched() {
        let t = TransposeLayer::batched_transpose();
        assert!(t.resolve_perm(1).is_err());
    }

    // ── compute_index_mapping ──────────────────────────────────────────

    #[ntest::timeout(5000)]
    #[test]
    fn test_index_mapping_2d_transpose() {
        let t = TransposeLayer::transpose_2d();
        // Input shape [2, 3]: elements indexed as:
        //   (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5
        // After transpose to [3, 2]:
        //   (0,0)→from(0,0)=0, (0,1)→from(1,0)=3, (1,0)→from(0,1)=1,
        //   (1,1)→from(1,1)=4, (2,0)→from(0,2)=2, (2,1)→from(1,2)=5
        let mapping = t.compute_index_mapping(&[2, 3], &[1, 0]).unwrap();
        assert_eq!(mapping, vec![0, 3, 1, 4, 2, 5]);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_index_mapping_identity_perm() {
        let t = TransposeLayer::new(vec![0, 1]);
        let mapping = t.compute_index_mapping(&[2, 3], &[0, 1]).unwrap();
        // Identity permutation: mapping[i] = i
        assert_eq!(mapping, vec![0, 1, 2, 3, 4, 5]);
    }

    // ── IBP propagation ────────────────────────────────────────────────

    #[ntest::timeout(5000)]
    #[test]
    fn test_ibp_2d_transpose() {
        let t = TransposeLayer::transpose_2d();
        // Input: 2x3 tensor with distinct values
        let lower =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let upper =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let result = t.propagate_ibp(&input).unwrap();

        // Output should be 3x2
        assert_eq!(result.shape(), &[3, 2]);

        // Check specific elements: result[i,j] = input[j,i]
        // result[0,0] = input[0,0] = [1.0, 1.5]
        assert_eq!(result.lower()[[0, 0]], 1.0);
        assert_eq!(result.upper()[[0, 0]], 1.5);
        // result[0,1] = input[1,0] = [4.0, 4.5]
        assert_eq!(result.lower()[[0, 1]], 4.0);
        assert_eq!(result.upper()[[0, 1]], 4.5);
        // result[2,1] = input[1,2] = [6.0, 6.5]
        assert_eq!(result.lower()[[2, 1]], 6.0);
        assert_eq!(result.upper()[[2, 1]], 6.5);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_ibp_batched_transpose_3d() {
        let t = TransposeLayer::batched_transpose();
        // Input: [2, 2, 3] — batch of 2, each is 2x3
        let lower =
            ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), (0..12).map(|i| i as f32).collect()).unwrap();
        let upper =
            ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), (0..12).map(|i| i as f32 + 0.5).collect())
                .unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let result = t.propagate_ibp(&input).unwrap();
        // Output should be [2, 3, 2] (swapping last two dims)
        assert_eq!(result.shape(), &[2, 3, 2]);
    }

    // ── CROWN backward propagation ─────────────────────────────────────

    #[ntest::timeout(5000)]
    #[test]
    fn test_crown_backward_2d_transpose() {
        let mut t = TransposeLayer::transpose_2d();
        t.set_input_shape(vec![2, 3]);

        // Identity bounds on output (6 elements flattened)
        let bounds = LinearBounds::identity(6);

        let result = t.propagate_linear(&bounds).unwrap();
        let result = result.as_ref();

        // After transpose [2,3]->[3,2], the columns of A get permuted
        // by the inverse of the transpose. For identity A, this permutes columns.
        // Check that result is a permutation matrix (each row/col has exactly one 1.0).
        assert_eq!(result.lower_a.shape(), &[6, 6]);

        // Verify it's a permutation: each row and column has exactly one non-zero
        for row in 0..6 {
            let nonzero_count = (0..6)
                .filter(|&col| result.lower_a[[row, col]].abs() > 1e-10)
                .count();
            assert_eq!(nonzero_count, 1, "Row {row} should have exactly 1 nonzero");
        }

        // Bias should be unchanged
        assert_eq!(result.lower_b, bounds.lower_b);
        assert_eq!(result.upper_b, bounds.upper_b);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_crown_backward_requires_input_shape() {
        let t = TransposeLayer::transpose_2d();
        let bounds = LinearBounds::identity(6);
        let result = t.propagate_linear(&bounds);
        assert!(result.is_err(), "Should error without input_shape set");
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_crown_backward_shape_mismatch_errors() {
        let mut t = TransposeLayer::transpose_2d();
        t.set_input_shape(vec![2, 3]); // 6 elements
        let bounds = LinearBounds::identity(4); // Wrong size
        let result = t.propagate_linear(&bounds);
        assert!(result.is_err(), "Should error on shape mismatch");
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_transpose_batched_multidim_exact_flat_grouped_4171() {
        let mut t = TransposeLayer::batched_transpose();
        t.set_input_shape(vec![1, 8, 3]);
        let bounds = BatchedLinearBounds::identity(&[1, 3, 8])
            .expect("grouped identity bounds should construct");
        let actual = t
            .propagate_linear_batched(&bounds)
            .expect("grouped multi-dimensional transpose should stay exact after flattening");

        let scalar_bounds = LinearBounds::identity(24);
        let expected = t
            .propagate_linear(&scalar_bounds)
            .expect("scalar transpose oracle should succeed")
            .into_owned();

        assert_eq!(
            actual.input_shape(),
            &[24],
            "grouped transpose should expose the exact flattened input shape"
        );
        assert_eq!(
            actual.output_shape(),
            &[24],
            "grouped transpose should expose the exact flattened output shape"
        );

        let expected_lower_a = expected.lower_a().clone().into_dyn();
        let expected_upper_a = expected.upper_a().clone().into_dyn();
        let expected_lower_b = expected.lower_b().clone().into_dyn();
        let expected_upper_b = expected.upper_b().clone().into_dyn();

        assert_eq!(
            actual.lower_a(),
            &expected_lower_a,
            "grouped transpose lower_a should match the scalar exact oracle"
        );
        assert_eq!(
            actual.upper_a(),
            &expected_upper_a,
            "grouped transpose upper_a should match the scalar exact oracle"
        );
        assert_eq!(
            actual.lower_b(),
            &expected_lower_b,
            "grouped transpose lower_b should match the scalar exact oracle"
        );
        assert_eq!(
            actual.upper_b(),
            &expected_upper_b,
            "grouped transpose upper_b should match the scalar exact oracle"
        );
    }

    // ── CROWN backward soundness: transpose then inverse ───────────────

    #[ntest::timeout(5000)]
    #[test]
    fn test_crown_backward_soundness_round_trip() {
        // Applying transpose and its inverse should give identity
        let mut t = TransposeLayer::transpose_2d();
        t.set_input_shape(vec![2, 3]);

        let bounds = LinearBounds::identity(6);
        let after_transpose = t.propagate_linear(&bounds).unwrap().into_owned();

        // Now apply inverse: transpose [3,2] -> [2,3]
        let mut t_inv = TransposeLayer::transpose_2d();
        t_inv.set_input_shape(vec![3, 2]);
        let round_trip = t_inv
            .propagate_linear(&after_transpose)
            .unwrap()
            .into_owned();

        // Should be back to identity
        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (round_trip.lower_a[[i, j]] - expected).abs() < 1e-6,
                    "Round-trip lower_a[{},{}] = {}, expected {}",
                    i,
                    j,
                    round_trip.lower_a[[i, j]],
                    expected
                );
            }
        }
    }

    // ── normalize_transpose_perm_for_rank: soundness ───────────────────

    /// Rank-≤1 transpose is the identity regardless of the (over-ranked) perm.
    #[ntest::timeout(5000)]
    #[test]
    fn test_normalize_perm_rank1_is_identity() {
        // vit case: perm={0,2,1} on a rank-1 {48} tensor.
        assert_eq!(
            normalize_transpose_perm_for_rank(&[0, 2, 1], 1),
            Some(vec![0])
        );
        // rank-0 scalar.
        assert_eq!(normalize_transpose_perm_for_rank(&[0, 1], 0), Some(vec![]));
        // even a "valid" rank-1 perm stays identity.
        assert_eq!(normalize_transpose_perm_for_rank(&[0], 1), Some(vec![0]));
    }

    /// Established batch-strip convention (#3602, one dropped leading dim).
    #[ntest::timeout(5000)]
    #[test]
    fn test_normalize_perm_drops_one_leading_dim() {
        // perm={0,2,1} authored for rank-3, applied to unbatched rank-2 => [1,0].
        assert_eq!(
            normalize_transpose_perm_for_rank(&[0, 2, 1], 2),
            Some(vec![1, 0])
        );
        // perm={0,1,3,2} (rank-4) on unbatched rank-3 => [0,2,1].
        assert_eq!(
            normalize_transpose_perm_for_rank(&[0, 1, 3, 2], 3),
            Some(vec![0, 2, 1])
        );
    }

    /// Multiple dropped leading dims (generalization beyond #3602).
    #[ntest::timeout(5000)]
    #[test]
    fn test_normalize_perm_drops_two_leading_dims() {
        // perm={0,1,3,2} (rank-4) on rank-2 => drop labels {0,1}, survivors {3,2}-2=[1,0].
        assert_eq!(
            normalize_transpose_perm_for_rank(&[0, 1, 3, 2], 2),
            Some(vec![1, 0])
        );
    }

    /// Already-consistent perm is validated and returned verbatim.
    #[ntest::timeout(5000)]
    #[test]
    fn test_normalize_perm_consistent_passthrough() {
        assert_eq!(
            normalize_transpose_perm_for_rank(&[2, 0, 1], 3),
            Some(vec![2, 0, 1])
        );
        // A same-length non-permutation (duplicate axis) is rejected.
        assert_eq!(normalize_transpose_perm_for_rank(&[0, 0, 2], 3), None);
    }

    /// Unprovable rewrites fail closed (None), never guessed.
    #[ntest::timeout(5000)]
    #[test]
    fn test_normalize_perm_unprovable_returns_none() {
        // perm references a leading axis that is NOT one of the dropped dims:
        // {0,2,1} on rank-1 falls to the rank<=1 identity rule (handled), but
        // {1,2,0} on rank-2 with d=1 is NOT a leading-passthrough (label 0 is not
        // in the dropped prefix position pattern) — still a permutation of 0..3 so
        // it rewrites; instead test a genuine non-permutation over-ranked perm.
        assert_eq!(normalize_transpose_perm_for_rank(&[0, 2, 3], 2), None);
        // perm shorter than rank (rank>=2) cannot be soundly extended.
        assert_eq!(normalize_transpose_perm_for_rank(&[1, 0], 3), None);
    }

    /// DENSE-SAMPLING SOUNDNESS: the normalized perm applied to the unbatched
    /// tensor (leading dropped dims squeezed) produces the SAME element ordering
    /// as the original perm applied to the full-rank tensor. This is the core
    /// guarantee that the rewrite preserves the transpose's mathematical function.
    #[ntest::timeout(10000)]
    #[test]
    fn test_normalize_perm_preserves_element_ordering_dense() {
        // (full_perm, full_shape with leading dropped dims of extent 1, dropped d)
        let cases: &[(Vec<usize>, Vec<usize>, usize)] = &[
            // perm={0,2,1}: rank-3 full [1,3,4] -> unbatched [3,4], d=1.
            (vec![0, 2, 1], vec![1, 3, 4], 1),
            // perm={0,1,3,2}: rank-4 full [1,1,3,4] -> unbatched [3,4], d=2.
            (vec![0, 1, 3, 2], vec![1, 1, 3, 4], 2),
            // perm={0,1,3,2}: rank-4 full [1,5,3,4] -> unbatched [5,3,4], d=1.
            (vec![0, 1, 3, 2], vec![1, 5, 3, 4], 1),
            // perm={0,2,3,1}: rank-4 full [1,2,3,4] -> unbatched [2,3,4], d=1.
            (vec![0, 2, 3, 1], vec![1, 2, 3, 4], 1),
            // Dropped (size-1) label NOT at a leading OUTPUT position: perm={1,0,2}
            // on full [1,3,4] outputs [3,1,4] (size-1 axis in the middle). Row-major
            // flatten is unaffected by a size-1 axis anywhere, so equivalence holds.
            (vec![1, 0, 2], vec![1, 3, 4], 1),
            // perm={2,0,1} on full [1,3,4] outputs [4,1,3]; size-1 trails-ish.
            (vec![2, 0, 1], vec![1, 3, 4], 1),
        ];
        for (full_perm, full_shape, d) in cases {
            let rank = full_shape.len() - d;
            let normalized =
                normalize_transpose_perm_for_rank(full_perm, rank).expect("case should normalize");

            // Build a full-rank tensor with distinct sequential values.
            let total: usize = full_shape.iter().product();
            let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
            let full = ArrayD::from_shape_vec(IxDyn(full_shape), data.clone()).unwrap();

            // Original transpose on the full-rank tensor.
            let full_t = full
                .view()
                .permuted_axes(full_perm.clone())
                .as_standard_layout()
                .into_owned();

            // Unbatched tensor: leading `d` dims have extent 1, so squeezing them
            // is value-preserving. Build the rank-`rank` view directly.
            let unbatched_shape: Vec<usize> = full_shape[*d..].to_vec();
            let unbatched = ArrayD::from_shape_vec(IxDyn(&unbatched_shape), data).unwrap();
            let unbatched_t = unbatched
                .view()
                .permuted_axes(normalized.clone())
                .as_standard_layout()
                .into_owned();

            // The leading `d` dims of `full_t` are all extent 1 (identity-mapped),
            // so the flattened contents of `full_t` and `unbatched_t` must match
            // element-for-element.
            assert_eq!(
                full_t.as_slice().unwrap(),
                unbatched_t.as_slice().unwrap(),
                "perm {:?} full {:?} d={} normalized {:?}: element ordering diverged",
                full_perm,
                full_shape,
                d,
                normalized
            );
        }
    }

    /// IBP end-to-end: a rank-1 bounded tensor through a TransposeLayer carrying
    /// the over-ranked vit perm {0,2,1} now resolves (identity) instead of erroring.
    #[ntest::timeout(5000)]
    #[test]
    fn test_transpose_layer_rank1_overranked_perm_is_identity() {
        let t = TransposeLayer::new(vec![0, 2, 1]);
        let lo = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 2.0]).unwrap();
        let hi = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 0.5, 3.0]).unwrap();
        let input = BoundedTensor::new(lo.clone(), hi.clone()).unwrap();
        let out = t
            .propagate_ibp(&input)
            .expect("rank-1 transpose should resolve");
        assert_eq!(out.lower(), &lo);
        assert_eq!(out.upper(), &hi);
    }
}
