// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array2, Array3, Array4, ArrayD, Axis, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};

use crate::BoundedTensor;

use super::ZonotopeTensor;

impl ZonotopeTensor {
    /// Create a zonotope from input with epsilon perturbation.
    ///
    /// Each element of the input gets its own error symbol, allowing
    /// the zonotope to track how perturbations to each element propagate.
    ///
    /// # Arguments
    /// * `values` - Center values for the input
    /// * `epsilon` - Maximum perturbation for each element
    ///
    /// # Note
    ///
    /// This creates n_elements error terms, which can be memory-intensive
    /// for large inputs. For large models, consider `from_input_shared`
    /// which uses fewer symbols.
    pub fn from_input_elementwise(values: &ArrayD<f32>, epsilon: f32) -> Self {
        let n_elements = values.len();

        // coeffs shape: (1 + n_elements, flat_size)
        // We flatten for simplicity; can reshape later
        let flat_values = values.iter().cloned().collect::<Vec<_>>();

        let mut coeffs = Array2::<f32>::zeros((1 + n_elements, n_elements));

        // Center row = values
        for (i, &v) in flat_values.iter().enumerate() {
            coeffs[[0, i]] = v;
        }

        // Each element gets epsilon coefficient at its own error term
        for i in 0..n_elements {
            coeffs[[1 + i, i]] = epsilon;
        }

        Self {
            coeffs: coeffs.into_dyn(),
            n_error_terms: n_elements,
            element_shape: vec![n_elements], // Flattened
        }
    }

    /// Create a zonotope with a single shared error symbol.
    ///
    /// All elements share one error symbol, representing uniform perturbation.
    /// This is memory-efficient but doesn't track element-specific correlations.
    ///
    /// # Arguments
    /// * `values` - Center values
    /// * `epsilon` - Maximum perturbation (same for all elements)
    pub fn from_input_shared(values: &ArrayD<f32>, epsilon: f32) -> Self {
        let element_shape = values.shape().to_vec();

        // coeffs shape: (2, ...element_shape)
        let mut coeffs_shape = vec![2];
        coeffs_shape.extend_from_slice(&element_shape);

        let mut coeffs = ArrayD::zeros(IxDyn(&coeffs_shape));

        // Center = values
        coeffs.index_axis_mut(Axis(0), 0).assign(values);

        // Single error term with coefficient = epsilon for all elements
        coeffs.index_axis_mut(Axis(0), 1).fill(epsilon);

        Self {
            coeffs,
            n_error_terms: 1,
            element_shape,
        }
    }

    /// Create a 2D zonotope from a matrix with per-element error symbols.
    ///
    /// Each element (i,j) gets its own error symbol. This is needed for
    /// operations like Q@K^T where we want to track correlations between
    /// all elements of Q and K.
    ///
    /// # Arguments
    /// * `values` - Center values with shape (rows, cols)
    /// * `epsilon` - Maximum perturbation for each element
    ///
    /// # Layout
    /// * coeffs shape: `(1 + rows*cols, rows, cols)`
    /// * `coeffs[0]` = center matrix
    /// * `coeffs[1 + i*cols + j]` has epsilon at position `(i,j)`, zero elsewhere
    pub fn from_input_2d(values: &Array2<f32>, epsilon: f32) -> Self {
        let rows = values.nrows();
        let cols = values.ncols();
        let n_elements = rows * cols;

        // coeffs shape: (1 + n_elements, rows, cols)
        let mut coeffs = Array3::<f32>::zeros((1 + n_elements, rows, cols));

        // Center = values
        coeffs.index_axis_mut(Axis(0), 0).assign(values);

        // Each element (i,j) gets its own error symbol
        for i in 0..rows {
            for j in 0..cols {
                let error_idx = 1 + i * cols + j;
                coeffs[[error_idx, i, j]] = epsilon;
            }
        }

        Self {
            coeffs: coeffs.into_dyn(),
            n_error_terms: n_elements,
            element_shape: vec![rows, cols],
        }
    }

    /// Create a zonotope with per-position error symbols (for sequence data).
    ///
    /// For shape (..., seq_len, embed_dim), creates seq_len error symbols.
    /// All elements at each position share a symbol.
    ///
    /// # Arguments
    /// * `values` - Center values with shape (..., seq_len, embed_dim)
    /// * `epsilon` - Maximum perturbation
    pub fn from_input_per_position(values: &ArrayD<f32>, epsilon: f32) -> Result<Self> {
        let shape = values.shape();
        if shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "from_input_per_position requires at least 2 dimensions".to_string(),
            ));
        }

        let unsupported_rank_error = || {
            NyError::InvalidSpec(
                "from_input_per_position currently only supports 2D or 3D tensors".to_string(),
            )
        };

        let seq_len = shape[shape.len() - 2];
        let element_shape = shape.to_vec();

        let n_error_terms = match shape.len() {
            2 => seq_len,
            3 => shape[0] * seq_len,
            _ => return Err(unsupported_rank_error()),
        };

        // coeffs shape: (1 + n_error_terms, ...element_shape)
        let mut coeffs_shape = vec![1 + n_error_terms];
        coeffs_shape.extend_from_slice(&element_shape);

        let mut coeffs = ArrayD::zeros(IxDyn(&coeffs_shape));

        // Center = values
        coeffs.index_axis_mut(Axis(0), 0).assign(values);

        match shape.len() {
            2 => {
                // For position i, set coeffs[1+i, i, :] = epsilon
                for pos in 0..seq_len {
                    for emb in 0..shape[1] {
                        coeffs[[1 + pos, pos, emb]] = epsilon;
                    }
                }
            }
            3 => {
                let batch = shape[0];
                let dim = shape[2];
                // For (b,pos), set coeffs[1 + b*seq + pos, b, pos, :] = epsilon
                for b in 0..batch {
                    for pos in 0..seq_len {
                        let err = 1 + b * seq_len + pos;
                        for d in 0..dim {
                            coeffs[[err, b, pos, d]] = epsilon;
                        }
                    }
                }
            }
            _ => return Err(unsupported_rank_error()),
        }

        Ok(Self {
            coeffs,
            n_error_terms,
            element_shape,
        })
    }

    /// Create zonotope from a BoundedTensor with a single shared error term.
    ///
    /// This is a lossy conversion - we can't recover the original correlations.
    /// The resulting zonotope has center = (lower+upper)/2 and radius = (upper-lower)/2.
    pub fn from_bounded_tensor(bounds: &BoundedTensor) -> Self {
        let center = (bounds.lower() + bounds.upper()) / 2.0;
        let radius = (bounds.upper() - bounds.lower()) / 2.0;

        let element_shape = bounds.shape().to_vec();
        let mut coeffs_shape = vec![2]; // center + 1 error term
        coeffs_shape.extend_from_slice(&element_shape);

        let mut coeffs = ArrayD::zeros(IxDyn(&coeffs_shape));
        coeffs.index_axis_mut(Axis(0), 0).assign(&center);
        coeffs.index_axis_mut(Axis(0), 1).assign(&radius);

        Self {
            coeffs,
            n_error_terms: 1,
            element_shape,
        }
    }

    /// Create a per-position zonotope from bounds with shape (..., seq, dim).
    ///
    /// Uses one error symbol per leading-batch+sequence position pair, with per-feature radii:
    /// - center = (lower + upper) / 2
    /// - error term for position (batch_idx, seq_idx) has coefficients radius[batch_idx, seq_idx, :]
    ///
    /// This is used for sequence data when we want to preserve correlations across multiple
    /// projections (Q/K/V) that share the same sequence position.
    pub fn from_bounded_tensor_per_position(bounds: &BoundedTensor) -> Result<Self> {
        let shape = bounds.shape();
        if shape.len() < 2 {
            return Err(NyError::InvalidSpec(format!(
                "from_bounded_tensor_per_position requires at least 2D bounds, got shape {:?}",
                shape
            )));
        }

        let dim = shape[shape.len() - 1];
        let seq = shape[shape.len() - 2];
        let batch_shape = &shape[..shape.len() - 2];
        let batch_size = checked_shape_product(batch_shape)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "from_bounded_tensor_batched: batch shape product overflows: {:?}",
                    batch_shape
                ))
            })?
            .max(1);

        let center = (bounds.lower() + bounds.upper()) / 2.0;
        let radius = (bounds.upper() - bounds.lower()) / 2.0;

        let center_3d = center
            .into_shape_with_order(IxDyn(&[batch_size, seq, dim]))
            .map_err(|e| NyError::InvalidSpec(format!("reshape center failed: {}", e)))?
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|_| NyError::InvalidSpec("Cannot view center as 3D".to_string()))?;
        let radius_3d = radius
            .into_shape_with_order(IxDyn(&[batch_size, seq, dim]))
            .map_err(|e| NyError::InvalidSpec(format!("reshape radius failed: {}", e)))?
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|_| NyError::InvalidSpec("Cannot view radius as 3D".to_string()))?;

        let n_error_terms = batch_size * seq;
        let mut coeffs_flat = Array4::<f32>::zeros((1 + n_error_terms, batch_size, seq, dim));
        coeffs_flat.index_axis_mut(Axis(0), 0).assign(&center_3d);

        for b in 0..batch_size {
            for s in 0..seq {
                let err = 1 + b * seq + s;
                for d in 0..dim {
                    coeffs_flat[[err, b, s, d]] = radius_3d[[b, s, d]];
                }
            }
        }

        let mut out_shape = vec![1 + n_error_terms];
        out_shape.extend_from_slice(batch_shape);
        out_shape.push(seq);
        out_shape.push(dim);

        let coeffs = coeffs_flat
            .into_dyn()
            .into_shape_with_order(IxDyn(&out_shape))
            .map_err(|e| NyError::InvalidSpec(format!("reshape coeffs failed: {}", e)))?;

        Ok(Self {
            coeffs,
            n_error_terms,
            element_shape: shape.to_vec(),
        })
    }

    /// Create a per-position zonotope from 2D bounds (seq, dim).
    ///
    /// Uses one error symbol per sequence position, with per-feature radii:
    /// - center = (lower + upper) / 2
    /// - error term for position i has coefficients radius[i, :]
    ///
    /// This preserves correlations between multiple projections (Q/K/V) that share the same
    /// input sequence position, without introducing per-element error symbols.
    pub fn from_bounded_tensor_per_position_2d(bounds: &BoundedTensor) -> Result<Self> {
        let shape = bounds.shape();
        if shape.len() != 2 {
            return Err(NyError::InvalidSpec(format!(
                "from_bounded_tensor_per_position_2d requires 2D bounds, got shape {:?}",
                shape
            )));
        }
        Self::from_bounded_tensor_per_position(bounds)
    }
}
