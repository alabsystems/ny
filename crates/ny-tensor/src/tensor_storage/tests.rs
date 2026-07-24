// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for tensor_storage module.

use super::*;
use ndarray::Array2;

#[test]
fn test_stack_basic_operations() {
    let mut storage = StackTensorStorage::new(&[0, 4]).unwrap();

    // Append 3 entries
    let data = Array2::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f32).into_dyn();
    storage.append(&data).unwrap();
    assert_eq!(storage.len(), 3);

    // Pop 2 entries (LIFO - should get last 2)
    let popped = storage.pop(2).unwrap();
    assert_eq!(popped.shape(), &[2, 4]);
    assert_eq!(popped[[0, 0]], 4.0); // Entry 1
    assert_eq!(popped[[1, 0]], 8.0); // Entry 2
    assert_eq!(storage.len(), 1);

    // Pop remaining
    let remaining = storage.pop(1).unwrap();
    assert_eq!(remaining[[0, 0]], 0.0); // Entry 0
    assert_eq!(storage.len(), 0);
}

#[test]
fn test_stack_reallocation() {
    let mut storage = StackTensorStorage::with_options(&[0, 2], 4, 16).unwrap();
    assert_eq!(storage.capacity(), 4);

    // Append more than initial capacity
    let data = Array2::zeros((10, 2)).into_dyn();
    storage.append(&data).unwrap();
    assert_eq!(storage.len(), 10);
    assert!(
        storage.capacity() >= 10,
        "capacity should grow to fit 10 entries, got {}",
        storage.capacity()
    );
}

#[test]
fn test_stack_reorder() {
    let mut storage = StackTensorStorage::new(&[0, 2]).unwrap();

    let data = Array2::from_shape_fn((4, 2), |(i, _)| i as f32).into_dyn();
    storage.append(&data).unwrap();

    // Reorder: [0,1,2,3] -> [3,1,2,0]
    storage.reorder(4, &[3, 1, 2, 0]).unwrap();

    let tensor = storage.tensor().unwrap();
    assert_eq!(tensor[[0, 0]], 3.0);
    assert_eq!(tensor[[1, 0]], 1.0);
    assert_eq!(tensor[[2, 0]], 2.0);
    assert_eq!(tensor[[3, 0]], 0.0);
}

#[test]
fn test_queue_basic_operations() {
    let mut storage = QueueTensorStorage::new(&[0, 4]).unwrap();

    // Append 3 entries
    let data = Array2::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f32).into_dyn();
    storage.append(&data).unwrap();
    assert_eq!(storage.len(), 3);

    // Pop 2 entries (FIFO - should get first 2)
    let popped = storage.pop(2).unwrap();
    assert_eq!(popped.shape(), &[2, 4]);
    assert_eq!(popped[[0, 0]], 0.0); // Entry 0
    assert_eq!(popped[[1, 0]], 4.0); // Entry 1
    assert_eq!(storage.len(), 1);
}

#[test]
fn test_queue_circular_buffer() {
    let mut storage = QueueTensorStorage::with_options(&[0, 2], 4, 16).unwrap();

    // Fill to capacity
    let data1 = Array2::from_shape_fn((4, 2), |(i, _)| i as f32).into_dyn();
    storage.append(&data1).unwrap();
    assert_eq!(storage.len(), 4);

    // Pop 2 (creates space at start)
    let _ = storage.pop(2).unwrap();
    assert_eq!(storage.len(), 2);

    // Append 2 more (should wrap around)
    let data2 = Array2::from_shape_fn((2, 2), |(i, _)| (10 + i) as f32).into_dyn();
    storage.append(&data2).unwrap();
    assert_eq!(storage.len(), 4);

    // Pop all and verify order
    let all = storage.pop(4).unwrap();
    assert_eq!(all[[0, 0]], 2.0); // Original entry 2
    assert_eq!(all[[1, 0]], 3.0); // Original entry 3
    assert_eq!(all[[2, 0]], 10.0); // New entry 0
    assert_eq!(all[[3, 0]], 11.0); // New entry 1
}

#[test]
fn test_empty_operations() {
    let mut stack = StackTensorStorage::new(&[0, 4]).unwrap();
    let mut queue = QueueTensorStorage::new(&[0, 4]).unwrap();

    // Pop from empty should return empty tensor
    let empty_stack = stack.pop(5).unwrap();
    let empty_queue = queue.pop(5).unwrap();

    assert_eq!(empty_stack.shape(), &[0, 4]);
    assert_eq!(empty_queue.shape(), &[0, 4]);
}

#[test]
fn test_memory_usage() {
    let mut storage = StackTensorStorage::new(&[0, 100]).unwrap();

    let data = Array2::zeros((10, 100)).into_dyn();
    storage.append(&data).unwrap();

    let (allocated, used) = storage.memory_usage();
    assert!(
        allocated >= used,
        "Allocated bytes ({allocated}) should be >= used bytes ({used})"
    );
    assert_eq!(used, 10 * 100 * 4); // 10 entries * 100 elements * 4 bytes
}

#[test]
fn test_create_tensor_storage_factory() {
    let stack = create_tensor_storage(&[0, 4], TreeTraversal::DepthFirst).unwrap();
    let queue = create_tensor_storage(&[0, 4], TreeTraversal::BreadthFirst).unwrap();

    assert_eq!(stack.len(), 0);
    assert_eq!(queue.len(), 0);
}

// --- Error path regression tests for #2896 ---

#[test]
fn test_stack_new_empty_shape_returns_err() {
    let result = StackTensorStorage::new(&[]);
    assert!(
        result.is_err(),
        "StackTensorStorage::new should reject an empty shape, got Ok(_)"
    );
}

#[test]
fn test_queue_new_empty_shape_returns_err() {
    let result = QueueTensorStorage::new(&[]);
    assert!(
        result.is_err(),
        "QueueTensorStorage::new should reject an empty shape, got Ok(_)"
    );
}

#[test]
fn test_create_tensor_storage_empty_shape_returns_err() {
    let result = create_tensor_storage(&[], TreeTraversal::DepthFirst);
    assert!(
        result.is_err(),
        "create_tensor_storage should reject an empty shape, got Ok(_)"
    );
}

#[test]
fn test_stack_append_shape_mismatch_returns_err() {
    let mut storage = StackTensorStorage::new(&[0, 4]).unwrap();
    // Append tensor with wrong element shape [3, 5] instead of [3, 4]
    let wrong = Array2::zeros((3, 5)).into_dyn();
    let result = storage.append(&wrong);
    assert!(
        result.is_err(),
        "append should reject mismatched stack element shapes, got {result:?}"
    );
    // Storage should be unchanged
    assert_eq!(storage.len(), 0);
}

#[test]
fn test_queue_append_shape_mismatch_returns_err() {
    let mut storage = QueueTensorStorage::new(&[0, 4]).unwrap();
    let wrong = Array2::zeros((3, 5)).into_dyn();
    let result = storage.append(&wrong);
    assert!(
        result.is_err(),
        "append should reject mismatched queue element shapes, got {result:?}"
    );
    assert_eq!(storage.len(), 0);
}

#[test]
fn test_stack_reorder_num_domains_exceeds_used_returns_err() {
    let mut storage = StackTensorStorage::new(&[0, 2]).unwrap();
    let data = Array2::zeros((3, 2)).into_dyn();
    storage.append(&data).unwrap();

    // num_domains=5 but only 3 stored
    let result = storage.reorder(5, &[0, 1, 2, 0, 1]);
    assert!(
        result.is_err(),
        "stack reorder should reject num_domains larger than used entries, got {result:?}"
    );
}

#[test]
fn test_stack_reorder_indices_length_mismatch_returns_err() {
    let mut storage = StackTensorStorage::new(&[0, 2]).unwrap();
    let data = Array2::zeros((3, 2)).into_dyn();
    storage.append(&data).unwrap();

    // 3 domains but only 2 indices
    let result = storage.reorder(3, &[0, 1]);
    assert!(
        result.is_err(),
        "stack reorder should reject indices shorter than num_domains, got {result:?}"
    );
}

#[test]
fn test_stack_reorder_index_out_of_bounds_returns_err() {
    let mut storage = StackTensorStorage::new(&[0, 2]).unwrap();
    let data = Array2::zeros((3, 2)).into_dyn();
    storage.append(&data).unwrap();

    // Index 5 is out of bounds for num_domains=3
    let result = storage.reorder(3, &[0, 1, 5]);
    assert!(
        result.is_err(),
        "stack reorder should reject out-of-bounds indices, got {result:?}"
    );
}

#[test]
fn test_queue_reorder_num_domains_exceeds_used_returns_err() {
    let mut storage = QueueTensorStorage::new(&[0, 2]).unwrap();
    let data = Array2::zeros((3, 2)).into_dyn();
    storage.append(&data).unwrap();

    let result = storage.reorder(5, &[0, 1, 2, 0, 1]);
    assert!(
        result.is_err(),
        "queue reorder should reject num_domains larger than used entries, got {result:?}"
    );
}

#[test]
fn test_queue_reorder_indices_length_mismatch_returns_err() {
    let mut storage = QueueTensorStorage::new(&[0, 2]).unwrap();
    let data = Array2::zeros((3, 2)).into_dyn();
    storage.append(&data).unwrap();

    let result = storage.reorder(3, &[0, 1]);
    assert!(
        result.is_err(),
        "queue reorder should reject indices shorter than num_domains, got {result:?}"
    );
}

#[test]
fn test_queue_reorder_index_out_of_bounds_returns_err() {
    let mut storage = QueueTensorStorage::new(&[0, 2]).unwrap();
    let data = Array2::zeros((3, 2)).into_dyn();
    storage.append(&data).unwrap();

    let result = storage.reorder(3, &[0, 1, 5]);
    assert!(
        result.is_err(),
        "queue reorder should reject out-of-bounds indices, got {result:?}"
    );
}

/// Regression test for #2257: Fortran-layout arrays must round-trip correctly.
/// Before the fix, `flatten_tensor_data` used `as_slice_memory_order()` which
/// returned column-major data for Fortran-layout arrays. The storage buffer
/// interpreted this as row-major, producing silently wrong results on pop.
#[test]
fn test_stack_append_fortran_layout_roundtrip_2257() {
    use ndarray::ShapeBuilder;

    let mut storage = StackTensorStorage::new(&[0, 3]).unwrap();

    // Create a Fortran-layout array with known values:
    // Logical (row-major) view:
    //   [[1, 2, 3],
    //    [4, 5, 6]]
    let c_arr = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mut f_arr = ArrayD::zeros(IxDyn(&[2, 3]).f());
    f_arr.assign(&c_arr);
    assert!(
        f_arr.as_slice().is_none(),
        "precondition: Fortran-layout array"
    );

    // Append Fortran-layout array
    storage.append(&f_arr).unwrap();
    assert_eq!(storage.len(), 2);

    // Pop and verify data matches LOGICAL order, not memory order.
    // Before #2257 fix, this would return [[1, 4, 2], [5, 3, 6]] (column-major
    // data misinterpreted as row-major).
    let popped = storage.pop(2).unwrap();
    assert_eq!(popped.shape(), &[2, 3]);
    assert_eq!(popped[[0, 0]], 1.0);
    assert_eq!(popped[[0, 1]], 2.0);
    assert_eq!(popped[[0, 2]], 3.0);
    assert_eq!(popped[[1, 0]], 4.0);
    assert_eq!(popped[[1, 1]], 5.0);
    assert_eq!(popped[[1, 2]], 6.0);
}

#[test]
fn test_queue_tensor_wrapped_returns_err() {
    let mut storage = QueueTensorStorage::with_options(&[0, 2], 4, 16).unwrap();

    // Fill to capacity
    let data = Array2::zeros((4, 2)).into_dyn();
    storage.append(&data).unwrap();

    // Pop 2 from front, creating space at start
    let _ = storage.pop(2).unwrap();

    // Append 2 more, which wraps around
    let data2 = Array2::zeros((2, 2)).into_dyn();
    storage.append(&data2).unwrap();

    // Now buffer is wrapped: tensor() should return Err
    let result = storage.tensor();
    assert!(
        result.is_err(),
        "tensor() should reject wrapped queue buffers, got {result:?}"
    );
}
