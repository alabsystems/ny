// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dynamic tensor storage with append/pop operations for branch-and-bound.
//!
//! This module provides TensorStorage-style pooling following alpha-beta-CROWN's
//! pattern. Tensors are stored with a batch dimension and can efficiently append
//! new entries and pop entries from either end (LIFO for DFS, FIFO for BFS).
//!
//! # Design
//!
//! Key features from alpha-beta-CROWN's tensor_storage.py:
//! - Dynamic capacity: exponential growth until `switching_size`, then linear
//! - Stack (LIFO) mode: pop from end for depth-first search
//! - Queue (FIFO) mode: pop from start for breadth-first search
//! - Reorder support: sort domains by lower bound
//! - Memory pooling: reuse underlying buffers via TensorPool
//!
//! # Example
//!
//! ```rust,no_run
//! use ny_tensor::tensor_storage::{TensorStorage, StackTensorStorage};
//! use ndarray::{Array2, ArrayD, IxDyn};
//!
//! fn example() -> ny_core::Result<()> {
//!     // Create storage for [batch, 10] shaped tensors
//!     let mut storage = StackTensorStorage::new(&[0, 10])?;
//!
//!     // Append some data
//!     let data = Array2::zeros((3, 10)).into_dyn();
//!     storage.append(&data)?;
//!     assert_eq!(storage.len(), 3);
//!
//!     // Pop 2 entries from the end (LIFO)
//!     let popped = storage.pop(2)?;
//!     assert_eq!(popped.shape()[0], 2);
//!     assert_eq!(storage.len(), 1);
//!     Ok(())
//! }
//! ```
//!
//! # Reference
//!
//! Based on alpha-beta-CROWN: complete_verifier/tensor_storage.py

#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::mem::size_of;

use ndarray::{ArrayD, ArrayView, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};

/// Default initial capacity for storage.
const DEFAULT_INITIAL_SIZE: usize = 1024;

/// Capacity at which growth switches from exponential to linear.
const DEFAULT_SWITCHING_SIZE: usize = 65536;

/// Linear growth factor after switching_size.
const LINEAR_GROWTH_FACTOR: usize = 32;

/// Return tensor elements in logical (row-major index) order.
///
/// Most owned tensors are already contiguous and borrow directly.
/// Non-contiguous inputs are flattened into a temporary buffer.
fn flatten_tensor_data(tensor: &ArrayD<f32>) -> Cow<'_, [f32]> {
    // Use as_slice() (not as_slice_memory_order()) to guarantee row-major order.
    // as_slice_memory_order() returns data in whatever layout the array uses,
    // which is column-major for Fortran-order arrays — contradicting this
    // function's contract. Fortran-layout arrays fall through to the iterator
    // path which always yields logical (row-major) order. Part of #2257.
    if let Some(slice) = tensor.as_slice() {
        Cow::Borrowed(slice)
    } else {
        Cow::Owned(tensor.iter().copied().collect())
    }
}

/// Trait for dynamic tensor storage with append/pop operations.
pub trait TensorStorage {
    /// Append a tensor to the storage along the batch dimension.
    ///
    /// The tensor must have shape `[n, ...rest]` where `rest` matches
    /// the storage's element shape. Returns an error if the element shape
    /// does not match.
    fn append(&mut self, tensor: &ArrayD<f32>) -> Result<()>;

    /// Pop `size` entries from the storage.
    ///
    /// Returns a tensor of shape `[size, ...rest]`.
    /// For StackTensorStorage: pops from end (LIFO).
    /// For QueueTensorStorage: pops from start (FIFO).
    ///
    /// Returns an error if internal bookkeeping is corrupted (shape
    /// computation yields an invalid array layout).
    fn pop(&mut self, size: usize) -> Result<ArrayD<f32>>;

    /// Get a view of all stored entries as a tensor.
    ///
    /// Returns an error if internal bookkeeping is corrupted or if
    /// `QueueTensorStorage`'s circular buffer has wrapped — call
    /// `reorder()` first to make data contiguous.
    fn tensor(&self) -> Result<ArrayView<'_, f32, IxDyn>>;

    /// Reorder entries based on indices.
    ///
    /// Returns an error if `num_domains` exceeds stored entries, `indices`
    /// length does not match `num_domains`, or any index is out of bounds.
    ///
    /// # Arguments
    /// * `num_domains` - Number of entries to reorder
    /// * `indices` - Permutation indices
    fn reorder(&mut self, num_domains: usize, indices: &[usize]) -> Result<()>;

    /// Current number of entries stored.
    fn len(&self) -> usize;

    /// Check if storage is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Current storage capacity.
    fn capacity(&self) -> usize;

    /// Calculate memory usage in bytes (allocated, used).
    fn memory_usage(&self) -> (usize, usize);
}

/// Stack-based tensor storage (LIFO) for depth-first search.
///
/// Entries are popped from the end, matching depth-first traversal order.
pub struct StackTensorStorage {
    /// Backing storage: shape [capacity, *element_shape]
    storage: Vec<f32>,
    /// Shape of the full storage (including capacity dimension)
    shape: Vec<usize>,
    /// Element shape (without batch dimension)
    element_shape: Vec<usize>,
    /// Number of elements per entry (product of element_shape)
    elements_per_entry: usize,
    /// Number of entries currently used
    num_used: usize,
    /// Current capacity (number of entries)
    current_capacity: usize,
    /// Size at which to switch from exponential to linear growth
    switching_size: usize,
}

impl StackTensorStorage {
    /// Create a new stack storage for tensors with the given full shape.
    ///
    /// # Arguments
    /// * `full_shape` - Shape including batch dimension, e.g., `[0, 10]` for [batch, 10]
    ///
    /// The first dimension is the batch size and can be 0 (will be set to initial_size).
    /// Returns an error if `full_shape` is empty.
    pub fn new(full_shape: &[usize]) -> Result<Self> {
        Self::with_options(full_shape, DEFAULT_INITIAL_SIZE, DEFAULT_SWITCHING_SIZE)
    }

    /// Create storage with custom initial size and switching size.
    ///
    /// Returns an error if `full_shape` is empty.
    pub fn with_options(
        full_shape: &[usize],
        initial_size: usize,
        switching_size: usize,
    ) -> Result<Self> {
        if full_shape.is_empty() {
            return Err(NyError::InvalidSpec(
                "TensorStorage requires at least one dimension".into(),
            ));
        }

        let element_shape: Vec<usize> = full_shape[1..].to_vec();
        let elements_per_entry = checked_shape_product(&element_shape)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "TensorStorage: element shape product overflows: {:?}",
                    element_shape
                ))
            })?
            .max(1);

        // Allocate initial storage
        let initial_capacity = initial_size.max(1);
        let total_elements = initial_capacity * elements_per_entry;
        let storage = vec![0.0f32; total_elements];

        let mut shape = vec![initial_capacity];
        shape.extend_from_slice(&element_shape);

        Ok(Self {
            storage,
            shape,
            element_shape,
            elements_per_entry,
            num_used: 0,
            current_capacity: initial_capacity,
            switching_size,
        })
    }

    /// Calculate new capacity given a request size.
    fn compute_new_capacity(&self, request_size: usize) -> usize {
        if self.current_capacity < self.switching_size {
            // Exponential growth
            (self.current_capacity * 2).max(self.num_used + request_size)
        } else {
            // Linear growth
            self.current_capacity + request_size * LINEAR_GROWTH_FACTOR
        }
    }

    /// Reallocate storage to new capacity.
    fn reallocate(&mut self, new_capacity: usize) {
        let new_total = new_capacity * self.elements_per_entry;
        let mut new_storage = vec![0.0f32; new_total];

        // Copy existing data
        let used_elements = self.num_used * self.elements_per_entry;
        new_storage[..used_elements].copy_from_slice(&self.storage[..used_elements]);

        self.storage = new_storage;
        self.current_capacity = new_capacity;
        self.shape[0] = new_capacity;
    }
}

impl TensorStorage for StackTensorStorage {
    fn append(&mut self, tensor: &ArrayD<f32>) -> Result<()> {
        let append_size = tensor.shape().first().copied().unwrap_or(0);
        if append_size == 0 {
            return Ok(());
        }

        // Validate shape matches
        let tensor_element_shape = &tensor.shape()[1..];
        if tensor_element_shape != &self.element_shape[..] {
            return Err(NyError::ShapeMismatch {
                expected: self.element_shape.clone(),
                got: tensor_element_shape.to_vec(),
            });
        }

        // Reallocate if needed
        if self.num_used + append_size > self.current_capacity {
            let new_capacity = self.compute_new_capacity(append_size);
            self.reallocate(new_capacity);
        }

        // Copy data
        let start = self.num_used * self.elements_per_entry;
        let append_elements = append_size * self.elements_per_entry;
        let tensor_data = flatten_tensor_data(tensor);
        debug_assert_eq!(tensor_data.len(), append_elements);
        self.storage[start..start + append_elements].copy_from_slice(&tensor_data);

        self.num_used += append_size;
        Ok(())
    }

    fn pop(&mut self, size: usize) -> Result<ArrayD<f32>> {
        let size = size.min(self.num_used);
        if size == 0 {
            let mut empty_shape = vec![0];
            empty_shape.extend_from_slice(&self.element_shape);
            return Ok(ArrayD::zeros(IxDyn(&empty_shape)));
        }

        // Pop from end (LIFO)
        let start = (self.num_used - size) * self.elements_per_entry;
        let elements = size * self.elements_per_entry;

        let mut result_shape = vec![size];
        result_shape.extend_from_slice(&self.element_shape);

        let result = ArrayD::from_shape_vec(
            IxDyn(&result_shape),
            self.storage[start..start + elements].to_vec(),
        )
        .map_err(|e| {
            NyError::InternalError(format!("StackTensorStorage::pop shape mismatch: {e}"))
        })?;

        self.num_used -= size;
        Ok(result)
    }

    fn tensor(&self) -> Result<ArrayView<'_, f32, IxDyn>> {
        let mut used_shape = vec![self.num_used];
        used_shape.extend_from_slice(&self.element_shape);

        let used_elements = self.num_used * self.elements_per_entry;
        ArrayView::from_shape(IxDyn(&used_shape), &self.storage[..used_elements]).map_err(|e| {
            NyError::InternalError(format!("StackTensorStorage::tensor shape mismatch: {e}"))
        })
    }

    fn reorder(&mut self, num_domains: usize, indices: &[usize]) -> Result<()> {
        if num_domains > self.num_used {
            return Err(NyError::InternalError(format!(
                "StackTensorStorage::reorder num_domains ({num_domains}) > num_used ({})",
                self.num_used
            )));
        }
        if indices.len() != num_domains {
            return Err(NyError::InternalError(format!(
                "StackTensorStorage::reorder indices.len() ({}) != num_domains ({num_domains})",
                indices.len()
            )));
        }

        // Create temporary buffer for reordering
        let reorder_elements = num_domains * self.elements_per_entry;
        let mut temp = vec![0.0f32; reorder_elements];

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            if old_idx >= num_domains {
                return Err(NyError::InternalError(format!(
                    "StackTensorStorage::reorder index {old_idx} out of bounds \
                     (num_domains={num_domains})"
                )));
            }
            let src_start = old_idx * self.elements_per_entry;
            let dst_start = new_idx * self.elements_per_entry;
            temp[dst_start..dst_start + self.elements_per_entry]
                .copy_from_slice(&self.storage[src_start..src_start + self.elements_per_entry]);
        }

        self.storage[..reorder_elements].copy_from_slice(&temp);
        Ok(())
    }

    fn len(&self) -> usize {
        self.num_used
    }

    fn capacity(&self) -> usize {
        self.current_capacity
    }

    fn memory_usage(&self) -> (usize, usize) {
        let elem_size = size_of::<f32>();
        let allocated = self.storage.len() * elem_size;
        let used = self.num_used * self.elements_per_entry * elem_size;
        (allocated, used)
    }
}

/// Queue-based tensor storage (FIFO) for breadth-first search.
///
/// Entries are popped from the start, matching breadth-first traversal order.
/// Uses a circular buffer internally for efficient FIFO operations.
pub struct QueueTensorStorage {
    /// Backing storage: shape [capacity, *element_shape]
    storage: Vec<f32>,
    /// Shape of the full storage
    shape: Vec<usize>,
    /// Element shape (without batch dimension)
    element_shape: Vec<usize>,
    /// Number of elements per entry
    elements_per_entry: usize,
    /// Number of entries currently used
    num_used: usize,
    /// Current capacity
    current_capacity: usize,
    /// Start index in circular buffer
    usage_start: usize,
    /// Switching size
    switching_size: usize,
}

impl QueueTensorStorage {
    /// Create a new queue storage for tensors with the given full shape.
    ///
    /// Returns an error if `full_shape` is empty.
    pub fn new(full_shape: &[usize]) -> Result<Self> {
        Self::with_options(full_shape, DEFAULT_INITIAL_SIZE, DEFAULT_SWITCHING_SIZE)
    }

    /// Create storage with custom options.
    ///
    /// Returns an error if `full_shape` is empty.
    pub fn with_options(
        full_shape: &[usize],
        initial_size: usize,
        switching_size: usize,
    ) -> Result<Self> {
        if full_shape.is_empty() {
            return Err(NyError::InvalidSpec(
                "TensorStorage requires at least one dimension".into(),
            ));
        }

        let element_shape: Vec<usize> = full_shape[1..].to_vec();
        let elements_per_entry = checked_shape_product(&element_shape)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "TensorStorage: element shape product overflows: {:?}",
                    element_shape
                ))
            })?
            .max(1);

        let initial_capacity = initial_size.max(1);
        let total_elements = initial_capacity * elements_per_entry;
        let storage = vec![0.0f32; total_elements];

        let mut shape = vec![initial_capacity];
        shape.extend_from_slice(&element_shape);

        Ok(Self {
            storage,
            shape,
            element_shape,
            elements_per_entry,
            num_used: 0,
            current_capacity: initial_capacity,
            usage_start: 0,
            switching_size,
        })
    }

    fn compute_new_capacity(&self, request_size: usize) -> usize {
        if self.current_capacity < self.switching_size {
            (self.current_capacity * 2).max(self.num_used + request_size)
        } else {
            self.current_capacity + request_size * LINEAR_GROWTH_FACTOR
        }
    }

    /// Move data to a new storage buffer, making it contiguous from index 0.
    fn move_to_new_storage(&mut self, new_capacity: usize) {
        let new_total = new_capacity * self.elements_per_entry;
        let mut new_storage = vec![0.0f32; new_total];

        // Handle circular buffer wrap-around
        let entries_to_end = (self.current_capacity - self.usage_start).min(self.num_used);
        let entries_at_start = self.num_used.saturating_sub(entries_to_end);

        // Copy entries from usage_start to end of buffer
        let src_start = self.usage_start * self.elements_per_entry;
        let copy_elements = entries_to_end * self.elements_per_entry;
        new_storage[..copy_elements]
            .copy_from_slice(&self.storage[src_start..src_start + copy_elements]);

        // Copy wrapped entries from start of buffer
        if entries_at_start > 0 {
            let dst_start = entries_to_end * self.elements_per_entry;
            let wrap_elements = entries_at_start * self.elements_per_entry;
            new_storage[dst_start..dst_start + wrap_elements]
                .copy_from_slice(&self.storage[..wrap_elements]);
        }

        self.storage = new_storage;
        self.current_capacity = new_capacity;
        self.shape[0] = new_capacity;
        self.usage_start = 0;
    }
}

impl TensorStorage for QueueTensorStorage {
    fn append(&mut self, tensor: &ArrayD<f32>) -> Result<()> {
        let append_size = tensor.shape().first().copied().unwrap_or(0);
        if append_size == 0 {
            return Ok(());
        }

        let tensor_element_shape = &tensor.shape()[1..];
        if tensor_element_shape != &self.element_shape[..] {
            return Err(NyError::ShapeMismatch {
                expected: self.element_shape.clone(),
                got: tensor_element_shape.to_vec(),
            });
        }

        // Reallocate if needed
        if self.num_used + append_size > self.current_capacity {
            let new_capacity = self.compute_new_capacity(append_size);
            self.move_to_new_storage(new_capacity);
        }

        let tensor_data = flatten_tensor_data(tensor);
        debug_assert_eq!(tensor_data.len(), append_size * self.elements_per_entry);

        // Find first free index in circular buffer
        let first_free = (self.usage_start + self.num_used) % self.current_capacity;
        let entries_at_tail = self.current_capacity - first_free;

        // Copy to tail of buffer
        let entries_to_tail = entries_at_tail.min(append_size);
        let dst_start = first_free * self.elements_per_entry;
        let copy_elements = entries_to_tail * self.elements_per_entry;
        self.storage[dst_start..dst_start + copy_elements]
            .copy_from_slice(&tensor_data[..copy_elements]);

        // Copy wrap-around to start of buffer
        if entries_to_tail < append_size {
            let entries_to_start = append_size - entries_to_tail;
            let src_start = entries_to_tail * self.elements_per_entry;
            let wrap_elements = entries_to_start * self.elements_per_entry;
            self.storage[..wrap_elements]
                .copy_from_slice(&tensor_data[src_start..src_start + wrap_elements]);
        }

        self.num_used += append_size;
        Ok(())
    }

    fn pop(&mut self, size: usize) -> Result<ArrayD<f32>> {
        let size = size.min(self.num_used);
        if size == 0 {
            let mut empty_shape = vec![0];
            empty_shape.extend_from_slice(&self.element_shape);
            return Ok(ArrayD::zeros(IxDyn(&empty_shape)));
        }

        let mut result_shape = vec![size];
        result_shape.extend_from_slice(&self.element_shape);
        let result_elements = size * self.elements_per_entry;

        // Pop from start (FIFO)
        let entries_to_end = (self.current_capacity - self.usage_start).min(size);

        if entries_to_end >= size {
            // All entries are contiguous
            let src_start = self.usage_start * self.elements_per_entry;
            let result = ArrayD::from_shape_vec(
                IxDyn(&result_shape),
                self.storage[src_start..src_start + result_elements].to_vec(),
            )
            .map_err(|e| {
                NyError::InternalError(format!("QueueTensorStorage::pop shape mismatch: {e}"))
            })?;

            self.num_used -= size;
            self.usage_start = (self.usage_start + size) % self.current_capacity;
            Ok(result)
        } else {
            // Need to concatenate from end and start of buffer
            let mut data = Vec::with_capacity(result_elements);

            // Copy from usage_start to end of buffer
            let src_start = self.usage_start * self.elements_per_entry;
            let first_part = entries_to_end * self.elements_per_entry;
            data.extend_from_slice(&self.storage[src_start..src_start + first_part]);

            // Copy from start of buffer
            let second_part = (size - entries_to_end) * self.elements_per_entry;
            data.extend_from_slice(&self.storage[..second_part]);

            let result = ArrayD::from_shape_vec(IxDyn(&result_shape), data).map_err(|e| {
                NyError::InternalError(format!(
                    "QueueTensorStorage::pop wrap-around shape mismatch: {e}"
                ))
            })?;

            self.num_used -= size;
            self.usage_start = (self.usage_start + size) % self.current_capacity;
            Ok(result)
        }
    }

    fn tensor(&self) -> Result<ArrayView<'_, f32, IxDyn>> {
        // For a view, we need contiguous data. If wrapped, caller must
        // call reorder() first to make data contiguous.
        let end_index = self.usage_start + self.num_used;
        if end_index > self.current_capacity {
            return Err(NyError::InternalError(
                "QueueTensorStorage::tensor() requires contiguous data. \
                 Call reorder() first."
                    .into(),
            ));
        }

        let mut used_shape = vec![self.num_used];
        used_shape.extend_from_slice(&self.element_shape);

        let start = self.usage_start * self.elements_per_entry;
        let end = end_index * self.elements_per_entry;
        ArrayView::from_shape(IxDyn(&used_shape), &self.storage[start..end]).map_err(|e| {
            NyError::InternalError(format!("QueueTensorStorage::tensor shape mismatch: {e}"))
        })
    }

    fn reorder(&mut self, num_domains: usize, indices: &[usize]) -> Result<()> {
        if num_domains > self.num_used {
            return Err(NyError::InternalError(format!(
                "QueueTensorStorage::reorder num_domains ({num_domains}) > num_used ({})",
                self.num_used
            )));
        }
        if indices.len() != num_domains {
            return Err(NyError::InternalError(format!(
                "QueueTensorStorage::reorder indices.len() ({}) != num_domains ({num_domains})",
                indices.len()
            )));
        }

        // First, make storage contiguous
        if self.usage_start != 0 {
            self.move_to_new_storage(self.current_capacity);
        }

        // Then reorder like stack storage
        let reorder_elements = num_domains * self.elements_per_entry;
        let mut temp = vec![0.0f32; reorder_elements];

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            if old_idx >= num_domains {
                return Err(NyError::InternalError(format!(
                    "QueueTensorStorage::reorder index {old_idx} out of bounds \
                     (num_domains={num_domains})"
                )));
            }
            let src_start = old_idx * self.elements_per_entry;
            let dst_start = new_idx * self.elements_per_entry;
            temp[dst_start..dst_start + self.elements_per_entry]
                .copy_from_slice(&self.storage[src_start..src_start + self.elements_per_entry]);
        }

        self.storage[..reorder_elements].copy_from_slice(&temp);
        Ok(())
    }

    fn len(&self) -> usize {
        self.num_used
    }

    fn capacity(&self) -> usize {
        self.current_capacity
    }

    fn memory_usage(&self) -> (usize, usize) {
        let elem_size = size_of::<f32>();
        let allocated = self.storage.len() * elem_size;
        let used = self.num_used * self.elements_per_entry * elem_size;
        (allocated, used)
    }
}

/// Tree traversal mode for domain list operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TreeTraversal {
    /// Depth-first search: uses stack (LIFO)
    #[default]
    DepthFirst,
    /// Breadth-first search: uses queue (FIFO)
    BreadthFirst,
}

/// Create a TensorStorage with the appropriate type based on traversal mode.
///
/// Returns an error if `full_shape` is empty.
pub fn create_tensor_storage(
    full_shape: &[usize],
    traversal: TreeTraversal,
) -> Result<Box<dyn TensorStorage + Send>> {
    match traversal {
        TreeTraversal::DepthFirst => Ok(Box::new(StackTensorStorage::new(full_shape)?)),
        TreeTraversal::BreadthFirst => Ok(Box::new(QueueTensorStorage::new(full_shape)?)),
    }
}
