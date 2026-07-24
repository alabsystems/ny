// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_acquire_returns_zeroed_buffer() {
    TensorPool::clear();
    let buffer = TensorPool::acquire(100);
    assert!(
        buffer.len() >= 100,
        "acquire(100) should return len >= request, got {}",
        buffer.len()
    );
    assert!(
        buffer.as_slice().iter().all(|&v| v == 0.0),
        "acquire(100) should zero-fill the buffer, got {:?}",
        buffer.as_slice()
    );
}

#[test]
fn test_size_class_bucketing() {
    // 64 elements -> bucket 0
    assert_eq!(PoolStorage::bucket_index(1), 0);
    assert_eq!(PoolStorage::bucket_index(64), 0);
    // 65-128 -> bucket 1
    assert_eq!(PoolStorage::bucket_index(65), 1);
    assert_eq!(PoolStorage::bucket_index(128), 1);
    // 129-256 -> bucket 2
    assert_eq!(PoolStorage::bucket_index(129), 2);
    assert_eq!(PoolStorage::bucket_index(256), 2);
}

#[test]
fn test_buffer_reuse() {
    TensorPool::clear();
    TensorPool::reset_stats();

    // First allocation: miss
    let buffer1 = TensorPool::acquire(100);
    let ptr1 = buffer1.as_slice().as_ptr();
    drop(buffer1);

    // Second allocation of same size: should hit
    let buffer2 = TensorPool::acquire(100);
    let ptr2 = buffer2.as_slice().as_ptr();

    // Same memory should be reused
    assert_eq!(ptr1, ptr2);

    let stats = TensorPool::stats();
    assert_eq!(stats.allocations, 2);
    assert_eq!(stats.pool_hits, 1);
    assert_eq!(stats.pool_misses, 1);
}

#[test]
fn test_pool_stats_display() {
    let stats = PoolStats {
        allocations: 10,
        pool_hits: 7,
        pool_misses: 3,
        returns: 5,
        discards: 1,
        pooled_buffers: 2,
        pooled_bytes: 2048,
        hit_rate: 0.7,
    };

    assert_eq!(
        stats.to_string(),
        "TensorPool: 10 allocs (70.0% hits), 2 pooled (2 KB)"
    );
}

#[test]
fn test_into_arrayd() {
    TensorPool::clear();

    let mut buffer = TensorPool::acquire(12);
    for (i, v) in buffer.as_mut_slice().iter_mut().enumerate() {
        *v = i as f32;
    }
    buffer.truncate(12);

    let array = buffer.into_arrayd(&[3, 4]).expect("into_arrayd failed");
    assert_eq!(array.shape(), &[3, 4]);
    assert_eq!(array[[0, 0]], 0.0);
    assert_eq!(array[[2, 3]], 11.0);
}

#[test]
fn test_try_into_arrayd_shape_mismatch_errors() {
    TensorPool::clear();

    let buffer = PooledBuffer::from_vec(vec![0.0f32; 3]);
    let err = buffer.try_into_arrayd(&[2, 2]).unwrap_err();
    let msg = format!("{}", err);
    assert!(
        msg.contains("into_arrayd"),
        "shape mismatch error should mention into_arrayd: {msg}"
    );
    assert!(
        msg.contains("requires 4 elements"),
        "shape mismatch error should describe the required element count: {msg}"
    );
}

#[test]
fn test_into_arrayd_does_not_return_to_pool() {
    TensorPool::clear();
    TensorPool::reset_stats();

    // Acquire a pool-backed buffer (bucket-aligned capacity 64)
    let mut buffer = TensorPool::acquire(64);
    for i in 0..64 {
        buffer.as_mut_slice()[i] = i as f32;
    }

    // into_arrayd consumes the buffer via mem::take — it must NOT return to pool
    let array = buffer.into_arrayd(&[8, 8]).expect("into_arrayd failed");
    assert_eq!(array.shape(), &[8, 8]);
    drop(array);

    let stats = TensorPool::stats();
    assert_eq!(
        stats.returns, 0,
        "into_arrayd should consume buffer without returning to pool"
    );
}

#[test]
fn test_into_vec_prevents_pool_return() {
    TensorPool::clear();
    TensorPool::reset_stats();

    let buffer = TensorPool::acquire(100);
    let _vec = buffer.into_vec();

    // No return since we took the vec
    let stats = TensorPool::stats();
    assert_eq!(
        stats.returns, 0,
        "into_vec should consume the buffer without returning it"
    );
}

#[test]
fn test_from_vec() {
    TensorPool::clear();
    TensorPool::reset_stats();

    // Use a bucket-aligned capacity (128) so the buffer is eligible for pooling.
    // Non-bucket capacities are correctly discarded on drop per the reuse invariant.
    let vec = vec![1.0f32; 128];
    let buffer = PooledBuffer::from_vec(vec);
    assert_eq!(buffer.len(), 128);
    assert!(
        buffer.as_slice().iter().all(|&v| v == 1.0),
        "from_vec should preserve element values, got {:?}",
        buffer.as_slice()
    );

    // Bucket-aligned buffer should return to pool on drop
    drop(buffer);

    let stats = TensorPool::stats();
    assert_eq!(stats.returns, 1);
}

#[test]
fn test_from_vec_preserves_capacity() {
    let mut data = Vec::with_capacity(128);
    data.extend(std::iter::repeat_n(0.0, 32));

    let buffer = PooledBuffer::from_vec(data);

    assert!(
        buffer.capacity() >= 128,
        "expected from_vec to preserve Vec capacity"
    );
}

#[test]
fn test_from_vec_non_bucket_capacity_is_not_pooled() {
    TensorPool::clear();
    TensorPool::reset_stats();

    let mut data = Vec::with_capacity(100);
    data.extend(std::iter::repeat_n(1.0, 32));

    let buffer = PooledBuffer::from_vec(data);
    assert_eq!(buffer.capacity(), 100);
    drop(buffer);

    let stats = TensorPool::stats();
    assert_eq!(stats.returns, 0);
    assert_eq!(stats.pooled_buffers, 0);
}

#[test]
fn test_max_buffers_per_class() {
    TensorPool::clear();

    // Fill pool beyond capacity
    let buffers: Vec<_> = (0..MAX_BUFFERS_PER_CLASS + 5)
        .map(|_| TensorPool::acquire(100))
        .collect();

    // Drop all buffers
    drop(buffers);

    let stats = TensorPool::stats();
    // Only MAX_BUFFERS_PER_CLASS should be kept
    assert!(
        stats.pooled_buffers <= MAX_BUFFERS_PER_CLASS,
        "pool kept {} buffers, exceeding the per-class cap {}",
        stats.pooled_buffers,
        MAX_BUFFERS_PER_CLASS
    );
    assert!(
        stats.discards >= 5,
        "dropping {} extra buffers should discard at least 5, got {}",
        MAX_BUFFERS_PER_CLASS + 5,
        stats.discards
    );
}

#[test]
fn test_different_size_classes_separate() {
    TensorPool::clear();
    TensorPool::reset_stats();

    // Allocate small buffer
    let small = TensorPool::acquire(50);
    let small_ptr = small.as_slice().as_ptr();
    drop(small);

    // Allocate larger buffer - should NOT reuse small buffer
    let large = TensorPool::acquire(200);
    let large_ptr = large.as_slice().as_ptr();

    // Different addresses (different size classes)
    assert_ne!(small_ptr, large_ptr);

    let stats = TensorPool::stats();
    assert_eq!(stats.pool_misses, 2); // Both were misses (different sizes)
}

#[test]
fn test_pool_stats_display_contains_summary() {
    TensorPool::clear();
    TensorPool::reset_stats();

    let _buf1 = TensorPool::acquire(100);
    let buf2 = TensorPool::acquire(100);
    drop(buf2);

    let stats = TensorPool::stats();
    let display = format!("{}", stats);
    assert!(
        display.contains("allocs"),
        "stats display should include allocation summary: {display}"
    );
    assert!(
        display.contains("hits"),
        "stats display should include hit-rate summary: {display}"
    );
}

#[test]
fn test_pooled_buffer_is_empty_exact() {
    TensorPool::clear();
    // Non-empty buffer should return false
    let buffer = TensorPool::acquire(100);
    assert!(
        !buffer.is_empty(),
        "acquired buffer should not report empty"
    );

    // Empty buffer should return true
    let empty_buffer = PooledBuffer::from_vec(vec![]);
    assert!(
        empty_buffer.is_empty(),
        "empty Vec-backed pooled buffer should report empty"
    );
}

#[test]
fn test_pooled_buffer_capacity_exact() {
    TensorPool::clear();
    let buffer = TensorPool::acquire(100);
    // Capacity should be at least 100, not 0 or 1
    assert!(
        buffer.capacity() >= 100,
        "capacity {} should cover the 100-element request",
        buffer.capacity()
    );
    assert!(
        buffer.capacity() >= 64,
        "capacity {} should be at least the minimum size class",
        buffer.capacity()
    ); // At least MIN_SIZE_CLASS

    // More specific test: capacity should match a power of 2 >= 100
    let cap = buffer.capacity();
    assert!(
        cap >= 128,
        "capacity {cap} should round the 100-element request up to 128"
    ); // Should be rounded up to 128 minimum
}

#[test]
fn test_pooled_buffer_truncate_boundary() {
    TensorPool::clear();
    let mut buffer = TensorPool::acquire(100);
    assert!(
        buffer.len() >= 100,
        "acquire(100) should return len >= request before truncation, got {}",
        buffer.len()
    );

    // Truncate to less than current length should work
    let _original_len = buffer.len();
    buffer.truncate(50);
    assert_eq!(buffer.len(), 50);

    // Truncate to greater than current length should be a no-op
    buffer.truncate(200);
    assert_eq!(buffer.len(), 50); // Still 50, not 200

    // Truncate to 0 should work
    buffer.truncate(0);
    assert_eq!(buffer.len(), 0);
    assert!(
        buffer.is_empty(),
        "truncate(0) should leave the buffer empty"
    );
}

#[test]
fn test_pooled_buffer_into_arrayd_truncates() {
    TensorPool::clear();
    let mut buffer = TensorPool::acquire(20);
    for i in 0..20 {
        buffer.as_mut_slice()[i] = i as f32;
    }

    // Should truncate to fit 3x4 = 12
    buffer.truncate(20);
    let array = buffer.into_arrayd(&[3, 4]).expect("into_arrayd failed");
    assert_eq!(array.shape(), &[3, 4]);
    assert_eq!(array[[0, 0]], 0.0);
    assert_eq!(array[[2, 3]], 11.0);
}

#[test]
fn test_clear_actually_clears() {
    // Create some buffers and return them
    {
        let _b1 = TensorPool::acquire(100);
        let _b2 = TensorPool::acquire(100);
    }

    let stats_before = TensorPool::stats();
    assert!(
        stats_before.returns >= 2,
        "dropping two acquired buffers should record at least two returns, got {}",
        stats_before.returns
    );

    TensorPool::clear();

    // After clear, pool should be empty
    let stats_after = TensorPool::stats();
    assert_eq!(stats_after.pooled_buffers, 0);
    assert_eq!(stats_after.pooled_bytes, 0);
}

#[test]
fn test_reset_stats_actually_resets() {
    TensorPool::clear();

    // Do some operations
    let _b = TensorPool::acquire(100);

    let stats_before = TensorPool::stats();
    assert!(
        stats_before.allocations >= 1,
        "acquire(100) should increment allocations before reset, got {}",
        stats_before.allocations
    );

    TensorPool::reset_stats();

    let stats_after = TensorPool::stats();
    assert_eq!(stats_after.allocations, 0);
    assert_eq!(stats_after.pool_hits, 0);
    assert_eq!(stats_after.pool_misses, 0);
}

#[test]
fn test_drop_returns_to_pool() {
    TensorPool::clear();
    TensorPool::reset_stats();

    {
        let buffer = TensorPool::acquire(100);
        drop(buffer);
    }

    let stats = TensorPool::stats();
    assert_eq!(stats.returns, 1);
}

#[test]
fn test_drop_doesnt_return_consumed_buffer() {
    TensorPool::clear();
    TensorPool::reset_stats();

    {
        let buffer = TensorPool::acquire(100);
        let _vec = buffer.into_vec(); // Consume it
    }

    let stats = TensorPool::stats();
    assert_eq!(stats.returns, 0); // Should not return since we consumed it
}

#[test]
fn test_byte_budget_discards_excess_buffers() {
    use std::mem::size_of;

    TensorPool::clear();
    TensorPool::reset_stats();

    let requested_elements = MAX_POOLED_BYTES / (8 * size_of::<f32>());
    let probe = TensorPool::acquire(requested_elements);
    let cached_bytes_per_buffer = probe.capacity() * size_of::<f32>();
    let keepable = MAX_POOLED_BYTES / cached_bytes_per_buffer;

    assert!(
        cached_bytes_per_buffer > 0,
        "probe capacity {} should produce a positive byte count",
        probe.capacity()
    );
    assert!(
        keepable > 0,
        "MAX_POOLED_BYTES should keep at least one buffer of {} bytes",
        cached_bytes_per_buffer
    );
    assert!(
        keepable < MAX_BUFFERS_PER_CLASS,
        "test setup expects budget to bind before per-class cap"
    );

    drop(probe);
    TensorPool::clear();
    TensorPool::reset_stats();

    let buffers: Vec<_> = (0..=keepable)
        .map(|_| TensorPool::acquire(requested_elements))
        .collect();
    drop(buffers);

    let stats = TensorPool::stats();
    assert_eq!(stats.pooled_buffers, keepable);
    assert_eq!(stats.pooled_bytes, keepable * cached_bytes_per_buffer);
    assert_eq!(stats.discards, 1);
}
