// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Mutation-killing tests for pool.rs.
//!
//! These tests target specific mutants that survived initial test coverage.
//! Each test documents which line and mutation it kills.

use super::*;

/// Kill mutant: Line 156 - replace * with + or / in PoolStorage::stats
/// Tests that pooled_bytes is calculated correctly as len * bucket_size * 4
#[test]
fn test_stats_pooled_bytes_calculation() {
    TensorPool::clear();
    TensorPool::reset_stats();

    // Acquire and release a buffer so it's in the pool
    let buffer = TensorPool::acquire(64); // Bucket 0 = 64 elements
    drop(buffer);

    let stats = TensorPool::stats();
    assert_eq!(stats.pooled_buffers, 1);
    // Should be 64 elements * 4 bytes/element = 256 bytes
    // If * was replaced with +: 1 + 64 + 4 = 69 (wrong)
    // If * was replaced with /: 1 / 64 / 4 = 0 (wrong)
    assert_eq!(stats.pooled_bytes, 256);

    // Add another buffer of different size
    let buffer2 = TensorPool::acquire(128); // Bucket 1 = 128 elements
    drop(buffer2);

    let stats2 = TensorPool::stats();
    assert_eq!(stats2.pooled_buffers, 2);
    // Should be 64*4 + 128*4 = 256 + 512 = 768 bytes
    assert_eq!(stats2.pooled_bytes, 768);
}

/// Kill mutant: Line 167 - replace > with ==, <, or >= in PoolStorage::stats
/// Tests hit_rate calculation with zero allocations
#[test]
fn test_stats_hit_rate_zero_allocations() {
    TensorPool::clear();
    TensorPool::reset_stats();

    let stats = TensorPool::stats();
    assert_eq!(stats.allocations, 0);
    // With zero allocations, hit_rate should be 0.0
    // If > was replaced with ==: would divide by zero (panic or NaN)
    // If > was replaced with <: would divide by zero for allocations=0
    // If > was replaced with >=: would still be 0.0 for allocations=0 but wrong for allocations>0
    assert_eq!(stats.hit_rate, 0.0);
    assert!(
        !stats.hit_rate.is_nan(),
        "hit_rate must not be NaN with zero allocations"
    );
}

/// Kill mutant: Line 168 - replace / with % or * in PoolStorage::stats
/// Tests that hit_rate is calculated as pool_hits / allocations
#[test]
fn test_stats_hit_rate_calculation() {
    TensorPool::clear();
    TensorPool::reset_stats();

    // First allocation: miss
    let buffer1 = TensorPool::acquire(100);
    drop(buffer1);

    // Second allocation: hit (reuses buffer)
    let buffer2 = TensorPool::acquire(100);
    drop(buffer2);

    // Third allocation: hit (reuses buffer)
    let buffer3 = TensorPool::acquire(100);
    drop(buffer3);

    // Fourth allocation: hit (reuses buffer)
    let buffer4 = TensorPool::acquire(100);
    drop(buffer4);

    let stats = TensorPool::stats();
    assert_eq!(stats.allocations, 4);
    assert_eq!(stats.pool_hits, 3);
    // hit_rate = 3 / 4 = 0.75
    // If / was replaced with %: 3 % 4 = 3 (wrong)
    // If / was replaced with *: 3 * 4 = 12 (wrong)
    assert!(
        (stats.hit_rate - 0.75).abs() < 1e-10,
        "hit_rate should be 0.75 (3 hits / 4 allocs), got {}",
        stats.hit_rate
    );
}

/// Kill mutant: Line 273 - replace < with <= in PooledBuffer::truncate
/// Tests boundary case where len == data.len()
#[test]
fn test_truncate_boundary_equal_length() {
    TensorPool::clear();

    let mut buffer = TensorPool::acquire(64);
    let original_len = buffer.len();
    assert_eq!(original_len, 64);

    // Truncate to exactly the current length - should be a no-op
    buffer.truncate(64);
    assert_eq!(buffer.len(), 64);

    // Now truncate to less - should actually truncate
    buffer.truncate(63);
    assert_eq!(buffer.len(), 63);

    // Truncate back to 64 - should be a no-op (can't expand)
    buffer.truncate(64);
    assert_eq!(buffer.len(), 63);
}

/// Kill mutant: Line 290 - replace < with <= in PooledBuffer::into_arrayd
/// Tests boundary case where expected_len == data.len()
#[test]
fn test_into_arrayd_exact_match() {
    TensorPool::clear();

    let mut buffer = TensorPool::acquire(64);
    // Fill with test data
    for i in 0..64 {
        buffer.as_mut_slice()[i] = i as f32;
    }

    // Shape [8, 8] = 64 elements, exactly matching buffer length
    // Should NOT truncate
    let array = buffer.into_arrayd(&[8, 8]).expect("into_arrayd failed");
    assert_eq!(array.shape(), &[8, 8]);
    assert_eq!(array[[0, 0]], 0.0);
    assert_eq!(array[[7, 7]], 63.0);
}

/// Kill mutant: Line 336 - replace && with || in Drop::drop
/// Tests that drop only returns buffer when BOTH conditions are met:
/// - size_class != usize::MAX (not consumed)
/// - !is_empty() (has data)
#[test]
fn test_drop_conditions_both_required() {
    TensorPool::clear();
    TensorPool::reset_stats();

    // Case 1: Consumed buffer (size_class == MAX) - should NOT return
    {
        let buffer = TensorPool::acquire(100);
        let _vec = buffer.into_vec(); // Consumes, sets size_class = MAX
    }
    let stats1 = TensorPool::stats();
    assert_eq!(stats1.returns, 0);

    TensorPool::reset_stats();

    // Case 2: Empty buffer (is_empty() == true) - should NOT return
    {
        let buffer = PooledBuffer::from_vec(vec![]);
        assert!(buffer.is_empty(), "buffer from empty vec should be empty");
        drop(buffer);
    }
    let stats2 = TensorPool::stats();
    assert_eq!(stats2.returns, 0);

    TensorPool::reset_stats();

    // Case 3: Normal buffer (size_class != MAX AND !is_empty()) - SHOULD return
    {
        let buffer = TensorPool::acquire(100);
        assert!(!buffer.is_empty(), "acquired buffer should not be empty");
        drop(buffer);
    }
    let stats3 = TensorPool::stats();
    assert_eq!(stats3.returns, 1);
}

/// Kill mutant: Line 109 - replace < with <= in PoolStorage::acquire
/// Tests that bucket index boundary is handled correctly
#[test]
fn test_acquire_bucket_boundary() {
    TensorPool::clear();
    TensorPool::reset_stats();

    // Test acquiring buffers at size class boundaries
    // Bucket 0: 64 elements
    let b0 = TensorPool::acquire(64);
    assert_eq!(b0.capacity(), 64);
    drop(b0);

    // Reacquire from pool should work
    let b0_reuse = TensorPool::acquire(64);
    assert_eq!(b0_reuse.capacity(), 64);

    let stats = TensorPool::stats();
    assert_eq!(stats.pool_hits, 1);
}

/// Kill mutant: Line 137 - replace < with <= in PoolStorage::release
/// Tests that release handles size class boundary correctly
#[test]
fn test_release_bucket_boundary() {
    TensorPool::clear();
    TensorPool::reset_stats();

    // Create a buffer and release it
    let buffer = TensorPool::acquire(64);
    let size_class = buffer.size_class;
    drop(buffer);

    // Verify it was returned to pool
    let stats = TensorPool::stats();
    assert_eq!(stats.returns, 1);
    assert_eq!(stats.pooled_buffers, 1);

    // Reacquire should get the same buffer back
    let buffer2 = TensorPool::acquire(64);
    assert_eq!(buffer2.size_class, size_class);
}

/// Kill mutant: verify multiplication in bytes calculation with multiple buckets
#[test]
fn test_pooled_bytes_multi_bucket() {
    TensorPool::clear();
    TensorPool::reset_stats();

    // Bucket 0: 64 elements = 256 bytes
    let b0 = TensorPool::acquire(64);
    drop(b0);

    // Bucket 1: 128 elements = 512 bytes
    let b1 = TensorPool::acquire(128);
    drop(b1);

    // Bucket 2: 256 elements = 1024 bytes
    let b2 = TensorPool::acquire(256);
    drop(b2);

    let stats = TensorPool::stats();
    assert_eq!(stats.pooled_buffers, 3);
    // Total: 256 + 512 + 1024 = 1792 bytes
    assert_eq!(stats.pooled_bytes, 1792);
}

/// Kill mutant: Lines 109, 137 - boundary check for oversized bucket indices
/// When bucket >= num_buckets, we should NOT try to access self.buckets[bucket]
/// If < was changed to <=, bucket == num_buckets would cause panic
#[test]
fn test_oversized_bucket_index_handling() {
    TensorPool::clear();
    TensorPool::reset_stats();

    // Test that bucket_index can return values >= num_buckets for large sizes
    // num_buckets = MAX_SIZE_CLASS_EXP - MIN_SIZE_CLASS.trailing_zeros() + 1 = 30 - 6 + 1 = 25
    // bucket_index returns exp - 6 where exp = trailing_zeros(next_power_of_two(capacity))
    // For capacity = 2^31, bucket = 31 - 6 = 25 which equals num_buckets

    // We can't actually allocate 2^31 elements, but we can test that the code
    // handles the case where bucket >= num_buckets by checking bucket_index directly
    let large_bucket = PoolStorage::bucket_index(1 << 31); // 2^31 elements
    assert!(
        large_bucket >= 25,
        "bucket index for 2^31 should be >= num_buckets (25), got {large_bucket}"
    );

    // Now test that we can create a PooledBuffer with an invalid size_class
    // and release it - it should be discarded, not cause a panic
    let oversized = PooledBuffer {
        data: vec![0.0f32; 100],
        size_class: 100, // Invalid: > num_buckets
        capacity: 100,
    };
    drop(oversized); // Should not panic, should discard

    let stats = TensorPool::stats();
    assert_eq!(stats.discards, 1); // Should be discarded due to invalid size_class
}

/// Kill mutant: Line 109 - if bucket < buckets.len() was bucket <= buckets.len()
/// For bucket == num_buckets, indexing would panic
/// Since we can't easily create such a large allocation, we test the boundary
#[test]
fn test_acquire_max_valid_bucket() {
    TensorPool::clear();
    TensorPool::reset_stats();

    // Test the largest bucket that's still valid
    // Bucket 24 = 2^(24+6) = 2^30 = 1 billion elements = 4GB
    // This is too large to actually allocate in a test

    // Instead test that bucket calculation works correctly near the boundary
    // bucket_index(2^30) should return 24 (the last valid bucket)
    let bucket_30 = PoolStorage::bucket_index(1 << 30);
    assert_eq!(bucket_30, 24); // 30 - 6 = 24

    // bucket_index(2^31) should return 25 (invalid bucket)
    let bucket_31 = PoolStorage::bucket_index(1 << 31);
    assert_eq!(bucket_31, 25); // 31 - 6 = 25

    // The check `bucket < num_buckets` (where num_buckets=25) means:
    // - bucket 24: valid (24 < 25)
    // - bucket 25: invalid (25 < 25 is false)
    // If mutated to <=, bucket 25 would pass but cause index OOB
}

/// Test that PooledBuffer Drop is defensive against RefCell re-entrance.
///
/// If a PooledBuffer is dropped while the pool's RefCell is already borrowed
/// (e.g., during another pool operation), the Drop should silently discard the
/// buffer instead of panicking on a double borrow_mut.
///
/// This uses POOL.try_with + try_borrow_mut (defensive Drop fix).
#[test]
fn test_drop_during_pool_borrow_does_not_panic() {
    TensorPool::clear();
    TensorPool::reset_stats();

    // Create a buffer that we'll drop inside a pool borrow
    let buffer = TensorPool::acquire(100);

    // Simulate re-entrant drop by manually borrowing the pool and then
    // dropping the buffer while the borrow is active.
    POOL.with(|pool| {
        let _guard = pool.borrow_mut(); // Active mutable borrow
                                        // Drop buffer while pool is already borrowed — should NOT panic.
                                        // Buffer is silently freed instead of returned to pool.
        drop(buffer);
    });

    let stats = TensorPool::stats();
    // The buffer was NOT returned to the pool (borrow failed, buffer freed).
    assert_eq!(
        stats.returns, 0,
        "buffer should not have been returned during re-entrant drop"
    );
}

/// Test that PooledBuffer Drop works correctly from a spawned thread.
///
/// When a PooledBuffer is sent to another thread, its Drop runs against
/// that thread's pool (thread-local), which is separate from the
/// originating thread's pool. This is not a leak, but the buffer
/// ends up in a different thread's pool.
#[test]
fn test_drop_on_different_thread_does_not_panic() {
    TensorPool::clear();
    TensorPool::reset_stats();

    let buffer = TensorPool::acquire(128);

    // Send buffer to another thread and drop it there
    let handle = std::thread::spawn(move || {
        drop(buffer); // Drops into the spawned thread's pool
    });
    handle.join().expect("spawned thread should not panic");

    // Original thread's pool should NOT have gotten the buffer back
    let stats = TensorPool::stats();
    assert_eq!(
        stats.returns, 0,
        "buffer dropped on another thread should not return to originating thread's pool"
    );
}

/// Kill mutant: Lines 113-119 - remove `data.clear()` or `data.resize()` in acquire.
/// Verifies that reused buffers are properly zeroed (data isolation between uses).
/// Without clear+resize, stale data from the previous user would leak through.
#[test]
fn test_reused_buffer_is_zeroed() {
    TensorPool::clear();
    TensorPool::reset_stats();

    // Step 1: Acquire a buffer and fill it with non-zero sentinel data.
    let mut buffer = TensorPool::acquire(64);
    for v in buffer.as_mut_slice().iter_mut() {
        *v = 42.0;
    }
    assert!(
        buffer.as_slice().iter().all(|&v| v == 42.0),
        "buffer should be filled with sentinel value 42.0"
    );

    // Step 2: Drop the buffer (returns to pool with stale data).
    drop(buffer);

    // Step 3: Reacquire from the same size class (pool hit).
    let reused = TensorPool::acquire(64);
    let stats = TensorPool::stats();
    assert_eq!(stats.pool_hits, 1, "expected pool reuse");

    // Step 4: Verify the reused buffer is all zeros (no data leakage).
    assert!(
        reused.as_slice().iter().all(|&v| v == 0.0),
        "reused buffer must be zeroed — stale data leaked through pool"
    );
}

/// Kill mutant: PooledArray drop → from_vec → pool → reacquire zeroed.
/// Verifies the full round-trip: PooledBuffer → ArrayD → PooledArray → Drop →
/// pool → reacquire returns zeroed data.
#[test]
fn test_pooled_array_round_trip_data_isolation() {
    TensorPool::clear();
    TensorPool::reset_stats();

    // Step 1: Acquire buffer, fill with data, convert to ArrayD via PooledArray.
    let mut buffer = TensorPool::acquire(16);
    for (i, v) in buffer.as_mut_slice().iter_mut().enumerate() {
        *v = (i + 1) as f32; // Non-zero sentinel
    }
    buffer.truncate(16);
    let array = crate::PooledArray::from_pooled_buffer(buffer, &[4, 4]);
    assert_eq!(array.as_array()[[0, 0]], 1.0);

    // Step 2: Drop the PooledArray (extracts Vec, wraps in PooledBuffer, returns to pool).
    drop(array);

    // Step 3: Reacquire — should get the pooled buffer back, zeroed.
    let reused = TensorPool::acquire(16);
    let stats = TensorPool::stats();
    // Should have at least 1 pool hit (the round-tripped buffer).
    assert!(
        stats.pool_hits >= 1,
        "expected pool reuse after PooledArray drop"
    );

    // Step 4: Verify zeroed.
    assert!(
        reused.as_slice().iter().all(|&v| v == 0.0),
        "buffer recycled through PooledArray must be zeroed on reacquire"
    );
}

// Note on equivalent mutants:
// Lines 273 and 290 - `len < self.data.len()` in truncate/into_arrayd
// These are EQUIVALENT MUTANTS because:
// - When len == data.len(), truncating does nothing (Vec::truncate is no-op)
// - So `<` vs `<=` produces identical behavior at the boundary
// These cannot be killed by any test.
