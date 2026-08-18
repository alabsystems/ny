// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::validate_cpu_buffer;
use ny_propagate::beta_crown::constraint_store::{
    ConstraintHeader, ConstraintOrigin, ConstraintSense, DomainConstraintStore,
};
use ny_propagate::BatchedConstraintBuffer;
use std::mem::{align_of, size_of};

fn make_valid_cpu_buffer() -> BatchedConstraintBuffer {
    let mut store = DomainConstraintStore::new();
    store
        .delta_mut()
        .add_constraint(
            &[0, 1],
            &[1.0, -1.0],
            0.0,
            ConstraintSense::Le,
            ConstraintOrigin::Split,
        )
        .unwrap();

    let stores = vec![&store];
    BatchedConstraintBuffer::from_domain_stores(&stores).unwrap()
}

#[test]
fn test_constraint_header_abi_layout() {
    assert_eq!(
        size_of::<ConstraintHeader>(),
        16,
        "ConstraintHeader must be 16 bytes for GPU buffer layout"
    );
    assert_eq!(
        align_of::<ConstraintHeader>(),
        4,
        "ConstraintHeader must be 4-byte aligned for GPU buffer layout"
    );
    assert_eq!(std::mem::offset_of!(ConstraintHeader, data_start), 0);
    assert_eq!(std::mem::offset_of!(ConstraintHeader, data_len), 4);
    assert_eq!(std::mem::offset_of!(ConstraintHeader, origin), 6);
    assert_eq!(std::mem::offset_of!(ConstraintHeader, sense), 7);
    assert_eq!(std::mem::offset_of!(ConstraintHeader, bias), 8);
}

#[test]
fn test_from_cpu_buffer_rejects_invalid_offsets() {
    let mut cpu_buffer = make_valid_cpu_buffer();
    cpu_buffer.domain_header_offsets = vec![0, 2];

    let err = match validate_cpu_buffer(&cpu_buffer) {
        Err(err) => err,
        Ok(()) => panic!("expected invalid domain_header_offsets to be rejected"),
    };
    assert!(matches!(
        err,
        ny_core::NyError::InvalidSpec(ref msg)
            if msg.contains("last domain_header_offsets entry")
    ));
}

#[test]
fn test_from_cpu_buffer_rejects_invalid_header_range() {
    let mut cpu_buffer = make_valid_cpu_buffer();
    cpu_buffer.headers[0] = ConstraintHeader::new(
        1,
        2,
        cpu_buffer.headers[0].bias,
        ConstraintSense::Le,
        ConstraintOrigin::Split,
    );

    let err = match validate_cpu_buffer(&cpu_buffer) {
        Err(err) => err,
        Ok(()) => panic!("expected invalid header term range to be rejected"),
    };
    assert!(matches!(
        err,
        ny_core::NyError::InvalidSpec(ref msg)
            if msg.contains("header 0 term range 1..3 exceeds coeffs length 2")
    ));
}
