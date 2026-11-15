// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Actual public prepared-tree cover allocation and final-drop observation.
//! Totals are successful current-thread System callback records, not RSS or
//! a process-wide live-allocation census. Residual records after query drop
//! have unresolved provenance; initialization or cross-thread frees may remain.

#[path = "support/certificate_allocations.rs"]
mod allocations;

use allocations::{measure, snapshot, watch_tracked_allocation};
use ny_propagate::certify::residency::{CancellationToken, MemoryAuthority};
use ny_propagate::certify::work::{WorkBudget, WorkLimits, WorkReserve};
use ny_propagate::margin_row::bab::{
    MarginRowBab, PreparedTreeCertificate, PreparedTreeOptions, PreparedTreeOutcome,
    PreparedTreeSchedule,
};
use ny_propagate::margin_row::{
    PreparedRootOutcome, PreparedRootProofOptions, PreparedRootSession, RootPrecisionRuntimePolicy,
    TwinOpSpec, TwinSpec,
};
use std::time::{Duration, Instant};

const CAP: usize = 32 << 20;

// The same declared one-input/two-head model used by the original prepared-tree
// control. Its duplicate nonlinear heads need two genuine conditional leaves.
fn fixture() -> TwinSpec {
    TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0],
                bias: vec![0.0],
                bias_err: vec![0.0],
                weight_rel_err: 0.0,
                kernel: (1, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, 1),
                oshape: (1, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: vec![1.0, 1.0],
                bias: vec![-1.0, -1.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![1.0, -1.0, 0.0, 0.0],
                bias: vec![0.75, 0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 2),
            },
        ],
    }
}

#[test]
fn actual_public_prepared_tree_complete_retained_graph_and_final_cover_drop() {
    let memory = MemoryAuthority::new(CAP);
    let cancel = CancellationToken::default();
    let deadline = Instant::now() + Duration::from_secs(60);
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: Some(deadline),
            scratch_bytes: CAP,
            persistent_bytes: CAP,
            max_work: 100_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    budget.tighten_total_byte_limit(CAP).unwrap();
    let ((source, compiled, receipt, held, released, work, bytes, cover_request), counts) =
        measure(|| {
            // Construct all moved model vectors inside the measured transaction.
            let session = PreparedRootSession::compile_declared_twin(
                fixture(),
                &[0.0],
                &[3.0],
                RootPrecisionRuntimePolicy::default(),
                PreparedRootProofOptions {
                    prefer_patch: false,
                    ..PreparedRootProofOptions::default()
                },
                &budget,
                Some(deadline),
            )
            .unwrap();
            let source = *session.source_digest();
            let compiled = *session.compiled_digest();
            let PreparedRootOutcome::Open(open) = session.try_prove_root(0, &[1], None).unwrap()
            else {
                panic!("original duplicate-head root must remain open")
            };
            assert!(open.receipt().ordinary_completed);
            assert!(!open.receipt().terminal_closed);
            drop(session);
            let PreparedTreeOutcome::Covered(cert) = MarginRowBab::run_prepared(
                open,
                PreparedTreeOptions {
                    max_expansions: 2,
                    lru_capacity: 4,
                    schedule: PreparedTreeSchedule::Serial,
                },
                None,
            )
            .unwrap() else {
                panic!("two genuine original head children must close")
            };
            // Source binding: PreparedTreeState::finish calls ledger.prove last,
            // which commits the cover's original Arc before returning Covered.
            // Confirm that actual final event is live before using its metadata.
            let returned = snapshot().unwrap();
            assert!(
                !returned.overflowed,
                "TLS request-census overflow: allocations={} reallocations={} deallocations={} requested_bytes={} recorded_live={} recorded_requested_bytes={:?}",
                returned.allocations,
                returned.reallocations,
                returned.deallocations,
                returned.requested_bytes,
                returned.live_allocations(),
                returned.live_requested_bytes(),
            );
            let cover_request = returned.last_allocation().unwrap();
            assert_eq!(
                returned.allocation_containing(cover_request.address),
                Some(cover_request)
            );
            assert!(watch_tracked_allocation(cover_request.address));
            assert_eq!(cert.target(), 0);
            assert_eq!(cert.classes(), [1]);
            let receipt = cert.receipt();
            assert_eq!(receipt.expansions, 1);
            assert_eq!(receipt.bound_leaves, 2);
            assert_eq!(receipt.parallel_batches, 0);
            assert_eq!(receipt.wave_batches, 0);
            cert.checkpoint(None).unwrap();
            let held = snapshot().unwrap();
            assert_eq!(held.allocations, returned.allocations);
            assert_eq!(held.reallocations, returned.reallocations);
            assert_eq!(held.requested_bytes, returned.requested_bytes);
            assert_eq!(held.last_allocation(), Some(cover_request));
            assert_eq!(held.watched_deallocations(cover_request.address), Some(0));
            let work = budget.receipt().unwrap();
            let bytes = memory.receipt().unwrap();
            assert_eq!(work.live_scratch_bytes, 0);
            assert!(work.live_persistent_bytes > 0);
            assert_eq!(bytes.live_pending_bytes, 0);
            assert_eq!(bytes.live_materialized_bytes, work.live_persistent_bytes);
            drop(cert);
            let released = snapshot().unwrap();
            assert_eq!(
                released.watched_deallocations(cover_request.address),
                Some(1)
            );
            assert!(released
                .allocation_containing(cover_request.address)
                .is_none());
            let after = budget.receipt().unwrap();
            let after_bytes = memory.receipt().unwrap();
            assert_eq!(after.live_persistent_bytes, 0);
            assert_eq!(after.live_scratch_bytes, 0);
            assert_eq!(after.work_reserved, work.work_reserved);
            assert_eq!(after.proposals_reserved, work.proposals_reserved);
            assert_eq!(after_bytes.live_pending_bytes, 0);
            assert_eq!(after_bytes.live_materialized_bytes, 0);
            assert_eq!(held.allocations, released.allocations);
            assert_eq!(held.reallocations, released.reallocations);
            assert!(!held.overflowed && !released.overflowed);
            (
                source,
                compiled,
                receipt,
                held,
                released,
                work,
                bytes,
                cover_request,
            )
        })
        .unwrap();
    assert_eq!(counts, released);
    assert!(!counts.overflowed);
    let held_bytes = held.live_requested_bytes().unwrap();
    let residual_bytes = released.live_requested_bytes().unwrap();
    let graph_bytes = held_bytes.checked_sub(residual_bytes).unwrap();
    let report_bytes = size_of::<PreparedTreeCertificate>();
    let complete_bytes = graph_bytes.checked_add(report_bytes).unwrap();
    let cover_request_bytes = cover_request.bytes;
    let inline_plus_cover_request_bytes = report_bytes.checked_add(cover_request_bytes).unwrap();
    assert_eq!(counts.watched_deallocations(cover_request.address), Some(1));
    println!(
        "{{\"phase\":\"actual_public_prepared_tree\",\"certificate_bytes\":{report_bytes},\"cover_request_bytes\":{cover_request_bytes},\"inline_plus_cover_request_bytes\":{inline_plus_cover_request_bytes},\"cover_watched_deallocations\":1,\"outcome_bytes\":{},\"allocations\":{},\"reallocations\":{},\"held_deallocations\":{},\"final_deallocations\":{},\"requested_bytes\":{},\"held_live_allocations\":{},\"residual_live_allocations\":{},\"held_live_requested_bytes\":{held_bytes},\"residual_live_requested_bytes\":{residual_bytes},\"report_graph_live_requested_bytes\":{graph_bytes},\"report_graph_plus_inline_bytes\":{complete_bytes},\"work\":{},\"proposals\":{},\"held_ledger_bytes\":{},\"peak_leased_bytes\":{},\"expansions\":{},\"evaluated_domains\":{},\"bound_leaves\":{},\"empty_leaves\":{},\"cache_hits\":{},\"cache_misses\":{},\"source\":{source:?},\"compiled\":{compiled:?}}}",
        size_of::<PreparedTreeOutcome>(),
        counts.allocations,
        counts.reallocations,
        held.deallocations,
        counts.deallocations,
        counts.requested_bytes,
        held.live_allocations(),
        released.live_allocations(),
        work.work_reserved,
        work.proposals_reserved,
        work.live_persistent_bytes,
        bytes.peak_leased_bytes,
        receipt.expansions,
        receipt.evaluated_domains,
        receipt.bound_leaves,
        receipt.empty_leaves,
        receipt.cache_hits,
        receipt.cache_misses,
    );
}
