// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Actual allocator requests for the public prepared affine context and all
//! three kernel routes. Compare identical observations across representations.

#[path = "support/certificate_allocations.rs"]
mod allocations;

use allocations::{measure, Counts};
use ndarray::array;
use ny_propagate::certify::scope::Digest;
use ny_propagate::certify::work::{LeaseClass, WorkBudget, WorkReceipt};
use ny_propagate::layers::{PointLinearBuildPolicy, PointLinearLayer};
use ny_propagate::network::original_affine_proof::{
    OriginalAffineIds, OriginalAffineOptions, OriginalLogitDifference, PreparedOriginalAffine,
};
use ny_propagate::LinearBounds;
use ny_tensor::BoundedTensor;
use std::hint::black_box;
use std::time::Duration;

fn options(budget: &WorkBudget) -> OriginalAffineOptions<'_> {
    OriginalAffineOptions {
        budget,
        class: LeaseClass::Optional,
        deadline: None,
    }
}

fn emit(
    phase: &str,
    context_bytes: usize,
    owner_bytes: usize,
    counts: &Counts,
    receipt: WorkReceipt,
    output: Digest,
) {
    assert!(!counts.overflowed);
    println!(
        "{{\"phase\":\"{phase}\",\"context_bytes\":{context_bytes},\"prepared_owner_request_bytes\":{owner_bytes},\"allocations\":{},\"reallocations\":{},\"deallocations\":{},\"requested_bytes\":{},\"work_reserved\":{},\"proposals_reserved\":{},\"live_persistent_bytes\":{},\"live_scratch_bytes\":{},\"output_digest\":{:?}}}",
        counts.allocations,
        counts.reallocations,
        counts.deallocations,
        counts.requested_bytes,
        receipt.work_reserved,
        receipt.proposals_reserved,
        receipt.live_persistent_bytes,
        receipt.live_scratch_bytes,
        output.0,
    );
}

#[test]
fn prepared_affine_native_allocation_scope_clone_and_last_alias_observation() {
    let budget = WorkBudget::new(None, 1 << 24, 1 << 26, 1 << 32, 1024, Duration::ZERO).unwrap();
    let weights = array![[1.0f32, 0.5, -0.25], [-0.5, 0.25, 2.0]];
    let biases = array![0.25f32, -0.125];
    let original = PointLinearLayer::from_original_view(
        weights.view(),
        false,
        Some(biases.view().into_dyn()),
        PointLinearBuildPolicy {
            budget: &budget,
            deadline: None,
            max_resident_bytes: 1 << 26,
            max_scratch_bytes: 1 << 24,
            max_work: 1 << 32,
        },
    )
    .unwrap();
    let ids = OriginalAffineIds {
        lowering: Digest::of_bytes(b"allocation-lowering", b"original affine"),
        node: Digest::of_bytes(b"allocation-node", b"affine"),
        input: Digest::of_bytes(b"allocation-input", b"original input"),
        output: Digest::of_bytes(b"allocation-output", b"original output"),
    };
    let (context, prepared_counts) =
        measure(|| PreparedOriginalAffine::prepare(&original, ids, options(&budget)).unwrap())
            .unwrap();
    let scope = std::ptr::from_ref(context.scope());
    let owner = prepared_counts
        .allocation_containing(scope.addr())
        .expect("scope remains inside the actual original prepared owner allocation");
    let context_bytes = size_of_val(&context);
    let prepared_bytes = context.preparation_reserved_bytes();
    emit(
        "prepared_affine_prepare",
        context_bytes,
        owner.bytes,
        &prepared_counts,
        budget.receipt().unwrap(),
        context.scope().model,
    );
    let input = BoundedTensor::new(
        array![[-1.0, -0.5, 0.0], [0.25, -2.0, 1.0]].into_dyn(),
        array![[1.0, 0.75, 1.0], [0.5, 0.0, 2.0]].into_dyn(),
    )
    .unwrap();
    let (interval, counts) =
        measure(|| context.interval(&input, options(&budget)).unwrap()).unwrap();
    assert!(std::ptr::eq(interval.scope(), context.scope()));
    emit(
        "prepared_affine_interval",
        context_bytes,
        owner.bytes,
        &counts,
        budget.receipt().unwrap(),
        interval.binding().output,
    );
    let incoming = LinearBounds::new(
        array![[1.0, -0.5, 0.25, -0.75]],
        array![0.125],
        array![[-0.25, 0.5, 1.0, -0.5]],
        array![-0.25],
    )
    .unwrap();
    let (signed, counts) =
        measure(|| context.signed_linear(&incoming, options(&budget)).unwrap()).unwrap();
    assert!(std::ptr::eq(signed.scope(), context.scope()));
    emit(
        "prepared_affine_signed",
        context_bytes,
        owner.bytes,
        &counts,
        budget.receipt().unwrap(),
        signed.binding().output,
    );
    let directions = [OriginalLogitDifference {
        target: 1,
        competitor: 0,
        position: 0,
    }];
    let (sparse, counts) = measure(|| {
        context
            .sparse_margins(2, &directions, options(&budget))
            .unwrap()
    })
    .unwrap();
    assert!(std::ptr::eq(sparse.scope(), context.scope()));
    emit(
        "prepared_affine_sparse",
        context_bytes,
        owner.bytes,
        &counts,
        budget.receipt().unwrap(),
        sparse.binding().output,
    );
    let (retained_interval, counts) = measure(|| black_box(&interval).clone()).unwrap();
    assert!(counts.is_empty());
    let (retained_signed, counts) = measure(|| black_box(&signed).clone()).unwrap();
    assert!(counts.is_empty());
    let (retained_sparse, counts) = measure(|| black_box(&sparse).clone()).unwrap();
    assert!(counts.is_empty());
    let expected = prepared_bytes
        + interval.original_owner_bytes()
        + interval.reserved_bytes()
        + signed.reserved_bytes()
        + sparse.reserved_bytes();
    drop(interval);
    drop(signed);
    drop(sparse);
    drop(context);
    drop(original);
    assert!(std::ptr::eq(retained_interval.scope(), scope));
    assert!(std::ptr::eq(retained_signed.scope(), scope));
    assert!(std::ptr::eq(retained_sparse.scope(), scope));
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, expected);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    let spent_work = budget.receipt().unwrap().work_reserved;
    drop(retained_interval);
    drop(retained_signed);
    assert!(budget.receipt().unwrap().live_persistent_bytes > 0);
    drop(retained_sparse);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(budget.receipt().unwrap().work_reserved, spent_work);
}
