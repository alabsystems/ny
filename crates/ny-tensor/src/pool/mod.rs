// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Thread-local memory pool for tensor buffers.
//!
//! This module provides a memory pool that reuses `Vec<f32>` buffers to reduce
//! allocation overhead during bound propagation. Buffers are organized into
//! size classes (powers of 2) for efficient reuse.
//!
//! # Usage
//!
//! ```rust,no_run
//! use ny_tensor::pool::TensorPool;
//!
//! // Acquire a buffer with at least 1000 elements
//! let mut buffer = TensorPool::acquire(1000);
//!
//! // Fill with data
//! let data = buffer.as_mut_slice();
//! for (i, v) in data.iter_mut().enumerate() {
//!     *v = i as f32;
//! }
//!
//! // Truncate to exact size and convert to ndarray
//! buffer.truncate(1000);
//! let array = buffer
//!     .into_arrayd(&[10, 100])
//!     .expect("into_arrayd failed");
//! ```
//!
//! # Performance
//!
//! Expected benefits:
//! - 30-50% memory reduction from buffer reuse
//! - 10-20% speedup from reduced allocation overhead
//! - No locking: thread-local pools avoid synchronization cost

use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use std::{cell::RefCell, mem::size_of};

/// Minimum size class: 64 elements (256 bytes for f32)
const MIN_SIZE_CLASS: usize = 64;

/// Maximum size class exponent (2^30 = ~1B elements = 4GB per buffer)
const MAX_SIZE_CLASS_EXP: usize = 30;

/// Maximum buffers to keep per size class
const MAX_BUFFERS_PER_CLASS: usize = 16;

/// Maximum total cached bytes per thread-local pool.
///
/// This bounds retention after one-off large verification spikes without
/// removing pooling entirely for steady-state workloads.
const MAX_POOLED_BYTES: usize = 32 * 1024 * 1024;

thread_local! {
    static POOL: RefCell<PoolStorage> = RefCell::new(PoolStorage::new());
}

/// Internal storage for the thread-local pool.
struct PoolStorage {
    /// buckets[i] holds buffers of size MIN_SIZE_CLASS * 2^i
    /// bucket 0: 64 elements
    /// bucket 1: 128 elements
    /// bucket 2: 256 elements
    /// ...
    buckets: Vec<Vec<Vec<f32>>>,
    /// Total bytes currently retained across all buckets.
    pooled_bytes: usize,
    /// Statistics for monitoring pool usage
    stats: PoolStatsInternal,
}

#[derive(Default, Clone)]
struct PoolStatsInternal {
    /// Total allocations requested
    allocations: usize,
    /// Allocations satisfied from pool (cache hits)
    pool_hits: usize,
    /// New allocations required (cache misses)
    pool_misses: usize,
    /// Total buffers returned to pool
    returns: usize,
    /// Buffers discarded (pool was full)
    discards: usize,
}

impl PoolStorage {
    fn num_buckets() -> usize {
        MAX_SIZE_CLASS_EXP - MIN_SIZE_CLASS.trailing_zeros() as usize + 1
    }

    fn new() -> Self {
        // Pre-create empty buckets for each size class
        Self {
            buckets: vec![Vec::new(); Self::num_buckets()],
            pooled_bytes: 0,
            stats: PoolStatsInternal::default(),
        }
    }

    /// Get the bucket index for a given capacity.
    fn bucket_index(capacity: usize) -> usize {
        if capacity <= MIN_SIZE_CLASS {
            return 0;
        }
        // Round up to next power of 2, then compute index
        let rounded = capacity.next_power_of_two();
        let exp = rounded.trailing_zeros() as usize;
        let min_exp = MIN_SIZE_CLASS.trailing_zeros() as usize;
        exp.saturating_sub(min_exp)
    }

    /// Get the actual size for a bucket index.
    fn size_for_bucket(bucket: usize) -> usize {
        MIN_SIZE_CLASS << bucket
    }

    fn buffer_bytes(buffer: &Vec<f32>) -> usize {
        buffer.capacity() * size_of::<f32>()
    }

    fn aligned_bucket_index(capacity: usize) -> Option<usize> {
        if capacity == 0 {
            return None;
        }

        let bucket = Self::bucket_index(capacity);
        if bucket >= Self::num_buckets() || capacity != Self::size_for_bucket(bucket) {
            return None;
        }

        Some(bucket)
    }

    /// Acquire a buffer from the pool or allocate a new one.
    fn acquire(&mut self, capacity: usize) -> PooledBuffer {
        self.stats.allocations += 1;

        let bucket = Self::bucket_index(capacity);
        let actual_size = Self::size_for_bucket(bucket);

        // Try to get a buffer from this bucket
        if bucket < self.buckets.len() {
            if let Some(mut data) = self.buckets[bucket].pop() {
                self.stats.pool_hits += 1;
                let pooled_bytes = Self::buffer_bytes(&data);
                debug_assert!(
                    self.pooled_bytes >= pooled_bytes,
                    "pool byte accounting underflow: pooled={} < popped={}",
                    self.pooled_bytes,
                    pooled_bytes
                );
                self.pooled_bytes = self.pooled_bytes.saturating_sub(pooled_bytes);
                // Clear the buffer for reuse (fill with zeros)
                data.clear();
                data.resize(actual_size, 0.0);
                return PooledBuffer {
                    capacity: data.capacity(),
                    data,
                    size_class: bucket,
                };
            }
        }

        // Need to allocate a new buffer
        self.stats.pool_misses += 1;
        let data = vec![0.0f32; actual_size];
        PooledBuffer {
            capacity: data.capacity(),
            data,
            size_class: bucket,
        }
    }

    /// Return a buffer to the pool.
    fn release(&mut self, mut buffer: Vec<f32>, size_class: usize) {
        self.stats.returns += 1;

        let buffer_bytes = Self::buffer_bytes(&buffer);
        let within_byte_budget = self
            .pooled_bytes
            .checked_add(buffer_bytes)
            .is_some_and(|total| total <= MAX_POOLED_BYTES);

        if size_class < self.buckets.len()
            && self.buckets[size_class].len() < MAX_BUFFERS_PER_CLASS
            && within_byte_budget
        {
            // Clear capacity but keep allocation
            buffer.clear();
            self.pooled_bytes += buffer_bytes;
            self.buckets[size_class].push(buffer);
        } else {
            // Pool is full or invalid size class, discard
            self.stats.discards += 1;
        }
    }

    /// Get current pool statistics.
    fn stats(&self) -> PoolStats {
        let total_pooled: usize = self.buckets.iter().map(Vec::len).sum();

        PoolStats {
            allocations: self.stats.allocations,
            pool_hits: self.stats.pool_hits,
            pool_misses: self.stats.pool_misses,
            returns: self.stats.returns,
            discards: self.stats.discards,
            pooled_buffers: total_pooled,
            pooled_bytes: self.pooled_bytes,
            hit_rate: if self.stats.allocations > 0 {
                self.stats.pool_hits as f64 / self.stats.allocations as f64
            } else {
                0.0
            },
        }
    }

    /// Clear all pooled buffers.
    fn clear(&mut self) {
        for bucket in &mut self.buckets {
            bucket.clear();
        }
        self.pooled_bytes = 0;
    }
}

/// Thread-local pool of reusable f32 buffers.
///
/// Buffers are organized into size classes (powers of 2) starting at 64 elements.
/// Each thread has its own pool to avoid synchronization overhead.
pub struct TensorPool;

impl TensorPool {
    /// Acquire a buffer with at least `capacity` f32 elements.
    ///
    /// The returned buffer may have more capacity than requested (rounded up to
    /// the next power of 2). The buffer is zero-initialized.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use ny_tensor::pool::TensorPool;
    ///
    /// let buffer = TensorPool::acquire(100);
    /// assert!(buffer.len() >= 100);
    /// ```
    #[inline]
    pub fn acquire(capacity: usize) -> PooledBuffer {
        POOL.with(|pool| pool.borrow_mut().acquire(capacity))
    }

    /// Get statistics about the current thread's pool.
    pub fn stats() -> PoolStats {
        POOL.with(|pool| pool.borrow().stats())
    }

    /// Clear all pooled buffers in the current thread's pool.
    ///
    /// This is mainly useful for testing to ensure clean state.
    pub fn clear() {
        POOL.with(|pool| pool.borrow_mut().clear())
    }

    /// Reset statistics counters (for benchmarking).
    pub fn reset_stats() {
        POOL.with(|pool| {
            pool.borrow_mut().stats = PoolStatsInternal::default();
        })
    }
}

/// A buffer acquired from the pool that auto-returns on drop.
///
/// The buffer can be used as a mutable slice, then either:
/// - Dropped to return to the pool for reuse
/// - Converted to an `ArrayD` via `into_arrayd()`
pub struct PooledBuffer {
    data: Vec<f32>,
    size_class: usize,
    capacity: usize,
}

impl PooledBuffer {
    /// Get the buffer contents as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get the buffer contents as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get the buffer length.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the actual capacity (may be larger than requested).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Truncate the buffer to a specific length.
    ///
    /// This is useful when you need exactly `len` elements for an array reshape.
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        if len < self.data.len() {
            self.data.truncate(len);
        }
    }

    /// Convert the buffer into an ndarray ArrayD with the given shape.
    ///
    /// The buffer is consumed and will NOT be returned to the pool. Use this
    /// only when you need the data as an ndarray for the rest of its lifetime.
    pub fn into_arrayd(mut self, shape: &[usize]) -> Result<ArrayD<f32>> {
        let expected_len = checked_shape_product(shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "PooledBuffer::into_arrayd: shape product overflows: {:?}",
                shape
            ))
        })?;
        if expected_len != self.data.len() {
            // Adjust length if needed (truncate or return error)
            if expected_len < self.data.len() {
                self.data.truncate(expected_len);
            } else {
                return Err(NyError::InvalidSpec(format!(
                    "PooledBuffer::into_arrayd: shape {:?} requires {} elements but buffer has {}",
                    shape,
                    expected_len,
                    self.data.len()
                )));
            }
        }

        // Take ownership of the data, preventing return to pool
        let data = std::mem::take(&mut self.data);
        // Mark as already consumed so Drop doesn't try to return it
        self.size_class = usize::MAX;

        // IxDyn construction only fails on a length mismatch, which the checked
        // shape product and truncation logic above already ruled out.
        Ok(ArrayD::from_shape_vec(IxDyn(shape), data)
            .expect("invariant: checked shape product matches buffer length"))
    }

    /// Convert the buffer into an ndarray ArrayD with the given shape.
    ///
    /// Retained for compatibility; prefer `into_arrayd`.
    pub fn try_into_arrayd(self, shape: &[usize]) -> Result<ArrayD<f32>> {
        self.into_arrayd(shape)
    }

    /// Convert the buffer into a raw `Vec<f32>`, preventing return to pool.
    ///
    /// Use this when you need the raw Vec for other purposes.
    pub fn into_vec(mut self) -> Vec<f32> {
        let data = std::mem::take(&mut self.data);
        self.size_class = usize::MAX; // Mark as consumed
        data
    }

    /// Create a new PooledBuffer from an existing Vec (wraps it for potential pooling).
    ///
    /// Only bucket-aligned capacities are eligible for reuse. Arbitrary Vecs
    /// that did not originate from TensorPool are freed on drop instead of
    /// being recycled into the wrong size class.
    pub fn from_vec(data: Vec<f32>) -> Self {
        let capacity = data.capacity();
        let size_class = PoolStorage::aligned_bucket_index(capacity).unwrap_or(usize::MAX);
        Self {
            data,
            size_class,
            capacity,
        }
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // Don't return if already consumed (size_class == MAX)
        if self.size_class != usize::MAX && self.capacity > 0 {
            let data = std::mem::take(&mut self.data);
            // Use try_with + try_borrow_mut to avoid panics when:
            // 1. Thread-local is already destroyed (thread shutdown ordering)
            // 2. Pool is already borrowed (re-entrant drop during pool operation)
            // In either case, the buffer is simply freed instead of returned to pool.
            let _ = POOL.try_with(|pool| {
                if let Ok(mut pool) = pool.try_borrow_mut() {
                    pool.release(data, self.size_class);
                }
                // If borrow fails, `data` drops here — buffer is freed, not pooled.
            });
            // If try_with fails, `data` was already moved into the closure or
            // drops here — buffer is freed, not pooled.
        }
    }
}

impl std::ops::Deref for PooledBuffer {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::DerefMut for PooledBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

/// Statistics about the memory pool.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total allocations requested.
    pub allocations: usize,
    /// Allocations satisfied from pool (cache hits).
    pub pool_hits: usize,
    /// New allocations required (cache misses).
    pub pool_misses: usize,
    /// Total buffers returned to pool.
    pub returns: usize,
    /// Buffers discarded (pool was full).
    pub discards: usize,
    /// Current number of buffers in the pool.
    pub pooled_buffers: usize,
    /// Current bytes held in the pool.
    pub pooled_bytes: usize,
    /// Hit rate (pool_hits / allocations).
    pub hit_rate: f64,
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TensorPool: {} allocs ({:.1}% hits), {} pooled ({} KB)",
            self.allocations,
            self.hit_rate * 100.0,
            self.pooled_buffers,
            self.pooled_bytes / 1024
        )
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_mutation;
