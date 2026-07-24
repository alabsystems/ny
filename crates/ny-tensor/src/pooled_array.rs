// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Pooled ArrayD storage that returns buffers to TensorPool on drop.
//!
//! This wraps an `ndarray::ArrayD<f32>` and returns its underlying Vec back to
//! `TensorPool` when dropped, enabling reuse of large CPU-side batched tensors.

use crate::{PooledBuffer, TensorPool};
use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};

/// ArrayD storage that returns its backing buffer to TensorPool on drop.
#[derive(Debug)]
pub struct PooledArray {
    array: Option<ArrayD<f32>>,
}

impl PooledArray {
    /// Create an empty pooled array (shape [0]).
    pub fn empty() -> Self {
        Self {
            array: Some(ArrayD::zeros(IxDyn(&[0]))),
        }
    }

    /// Wrap an existing ArrayD (will return buffer to pool on drop).
    pub fn from_array(array: ArrayD<f32>) -> Self {
        Self { array: Some(array) }
    }

    /// Build a pooled array from a PooledBuffer and shape.
    ///
    /// # Panics
    ///
    /// Panics if the buffer length doesn't match `shape`. This is a programmer
    /// error: callers must ensure the buffer was acquired with the correct size.
    /// Use [`try_from_pooled_buffer`](Self::try_from_pooled_buffer) for fallible construction.
    pub fn from_pooled_buffer(buffer: PooledBuffer, shape: &[usize]) -> Self {
        Self::try_from_pooled_buffer(buffer, shape)
            .expect("invariant: buffer length must match shape product")
    }

    /// Build a pooled array from a PooledBuffer and shape.
    pub fn try_from_pooled_buffer(buffer: PooledBuffer, shape: &[usize]) -> Result<Self> {
        let array = buffer.try_into_arrayd(shape)?;
        Ok(Self::from_array(array))
    }

    /// Build a pooled array from a raw Vec and shape.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        let expected_len = checked_shape_product(shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "PooledArray::from_vec: shape product overflows: {:?}",
                shape
            ))
        })?;
        if expected_len != data.len() {
            return Err(NyError::InvalidSpec(format!(
                "PooledArray::from_vec: shape {:?} requires {} elements but vec has {}",
                shape,
                expected_len,
                data.len()
            )));
        }
        let array = ArrayD::from_shape_vec(IxDyn(shape), data).map_err(|_| {
            NyError::InvalidSpec("PooledArray::from_vec failed to build ArrayD".to_string())
        })?;
        Ok(Self::from_array(array))
    }

    /// Allocate a pooled array with the given shape, filled with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        let total =
            checked_shape_product(shape).expect("PooledArray::zeros: shape product overflows");
        let mut buffer = TensorPool::acquire(total);
        buffer.truncate(total);
        Self::from_pooled_buffer(buffer, shape)
    }

    /// Borrow the underlying array.
    ///
    /// # Panics
    ///
    /// Panics if the internal `Option` is `None`. This is unreachable in safe
    /// Rust: the `Option` is `Some` at construction and only set to `None` by
    /// `into_array()` (which consumes `self`) or `Drop`. The panic is a
    /// defensive guard for internal invariant safety.
    #[inline]
    pub fn as_array(&self) -> &ArrayD<f32> {
        // Invariant: array is Some at construction, only None after into_array()
        // consumes self or Drop runs — unreachable in safe Rust.
        self.array
            .as_ref()
            .expect("invariant: PooledArray array is always Some")
    }

    /// Mutably borrow the underlying array.
    ///
    /// # Panics
    ///
    /// Panics if the internal `Option` is `None`. Unreachable in safe Rust —
    /// see [`as_array`](Self::as_array) for explanation.
    #[inline]
    pub fn as_array_mut(&mut self) -> &mut ArrayD<f32> {
        self.array
            .as_mut()
            .expect("invariant: PooledArray array is always Some")
    }

    /// Consume without returning buffer to the pool.
    ///
    /// # Panics
    ///
    /// Panics if the internal `Option` is `None`. Unreachable in safe Rust:
    /// `into_array()` consumes `self` by value, so it cannot be called twice.
    /// The panic is a defensive guard matching the `Option` unwrap pattern.
    pub fn into_array(mut self) -> ArrayD<f32> {
        self.array
            .take()
            .expect("invariant: PooledArray array is always Some")
    }
}

impl Clone for PooledArray {
    fn clone(&self) -> Self {
        Self::from_array(self.as_array().clone())
    }
}

impl Drop for PooledArray {
    fn drop(&mut self) {
        if let Some(array) = self.array.take() {
            let (data, offset) = array.into_raw_vec_and_offset();
            if let Some(offset) = offset {
                debug_assert!(
                    offset <= data.len(),
                    "PooledArray offset {} exceeds data len {}",
                    offset,
                    data.len()
                );
            }
            // Return the original backing storage; the pool clears contents on reuse.
            let _buffer = PooledBuffer::from_vec(data);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;

    #[ntest::timeout(10000)]
    #[test]
    fn test_pooled_array_returns_to_pool_on_drop() {
        TensorPool::clear();
        TensorPool::reset_stats();

        {
            let mut buffer = TensorPool::acquire(16);
            buffer.truncate(16);
            let array = PooledArray::from_pooled_buffer(buffer, &[4, 4]);
            assert_eq!(array.as_array().shape(), &[4, 4]);
        }

        let stats = TensorPool::stats();
        assert!(stats.returns > 0, "expected pooled array to return buffer");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_pooled_array_drop_handles_offset_slices_without_pooling() {
        TensorPool::clear();
        TensorPool::reset_stats();

        let base = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0])
            .expect("base array should construct");
        let sliced = base.slice_move(s![1..]).into_dyn();

        {
            let _array = PooledArray::from_array(sliced);
        }

        let stats = TensorPool::stats();
        assert_eq!(
            stats.returns, 0,
            "offset slices from non-pooled arrays should be freed, not pooled"
        );
        assert_eq!(stats.pooled_buffers, 0);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_pooled_array_drop_skips_non_pool_backing_capacity() {
        TensorPool::clear();
        TensorPool::reset_stats();

        let mut data = Vec::with_capacity(100);
        data.extend(std::iter::repeat_n(1.0, 32));
        let array = ArrayD::from_shape_vec(IxDyn(&[32]), data)
            .expect("array with non-pool capacity should construct");

        {
            let _array = PooledArray::from_array(array);
        }

        let stats = TensorPool::stats();
        assert_eq!(
            stats.returns, 0,
            "arbitrary backing capacities should not be recycled into TensorPool"
        );
        assert_eq!(stats.pooled_buffers, 0);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_pooled_array_drop_returns_empty_buffer() {
        TensorPool::clear();
        TensorPool::reset_stats();

        {
            let _array = PooledArray::zeros(&[0]);
        }

        let stats = TensorPool::stats();
        assert!(
            stats.returns > 0,
            "expected empty pooled array to return buffer"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_pooled_array_into_array_does_not_return_to_pool() {
        TensorPool::clear();
        TensorPool::reset_stats();

        let mut buffer = TensorPool::acquire(4);
        buffer.truncate(4);
        let array = PooledArray::from_pooled_buffer(buffer, &[4]);
        let _owned = array.into_array();

        let stats = TensorPool::stats();
        assert_eq!(
            stats.returns, 0,
            "expected into_array to consume without returning to pool"
        );
    }
}
