// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ONNX NonZero layer for bound propagation.

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;

use crate::layers::common::BoundPropagation;
use crate::LinearBounds;

/// NonZero: returns indices of non-zero elements.
///
/// ONNX NonZero returns a 2D tensor of shape [rank(input), num_nonzero] where:
/// - Row i contains the indices along dimension i for each non-zero element
/// - num_nonzero is the count of non-zero elements (data-dependent)
///
/// For bound propagation, since the output shape is data-dependent:
/// - We compute the maximum possible number of non-zero elements (elements where
///   the interval could contain non-zero values, i.e., lower < 0 or upper > 0)
/// - We return index bounds: lower = 0, upper = dim_size - 1 for each dimension
///
/// This is sound but conservative - downstream operations (like Gather) will
/// see the full range of possible indices.
#[derive(Debug, Clone)]
pub struct NonZeroLayer;

impl NonZeroLayer {
    /// Count elements that could possibly be non-zero.
    /// An element could be non-zero if its interval doesn't contain exactly 0.
    fn count_possibly_nonzero(input: &BoundedTensor) -> usize {
        ndarray::Zip::from(input.lower())
            .and(input.upper())
            .fold(0, |count, &l, &u| {
                // Element is possibly non-zero if interval is not exactly [0, 0]
                // and the interval doesn't exclude all non-zero values
                if l > 0.0 || u < 0.0 || (l != 0.0 || u != 0.0) {
                    count + 1
                } else {
                    count
                }
            })
    }

    /// Propagate IBP bounds through NonZero.
    ///
    /// Returns index bounds with shape [rank(input), max_possibly_nonzero].
    /// All index values are bounded by [0, dim_size - 1] for the corresponding dimension.
    pub fn propagate_ibp_unary(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let shape = input.shape();
        let rank = shape.len();

        // Count maximum possible non-zero elements
        // In the worst case, any element that could be non-zero will be
        let max_nonzero = Self::count_possibly_nonzero(input);

        // If no elements could be non-zero, return empty result
        if max_nonzero == 0 {
            // Shape: [rank, 0] - no non-zero elements
            let out_shape = IxDyn(&[rank, 0]);
            let out_lower = ArrayD::<f32>::zeros(out_shape.clone());
            let out_upper = ArrayD::<f32>::zeros(out_shape);
            return BoundedTensor::new(out_lower, out_upper);
        }

        // Output shape: [rank, max_nonzero]
        let out_shape = IxDyn(&[rank, max_nonzero]);

        // Lower bounds: all 0s (minimum possible index is 0)
        let out_lower = ArrayD::<f32>::zeros(out_shape.clone());

        // Upper bounds: [dim_size - 1] for each dimension, replicated across columns
        let mut out_upper = ArrayD::<f32>::zeros(out_shape);
        for (dim_idx, &dim_size) in shape.iter().enumerate() {
            let max_idx = (dim_size.saturating_sub(1)) as f32;
            for col in 0..max_nonzero {
                out_upper[[dim_idx, col]] = max_idx;
            }
        }

        BoundedTensor::new(out_lower, out_upper)
    }
}

impl BoundPropagation for NonZeroLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_ibp_unary(input)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // NonZero has data-dependent output shape, cannot use linear bounds
        Err(NyError::UnsupportedOp(
            "NonZero has data-dependent output shape - use propagate_ibp".to_string(),
        ))
    }
}
