// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use std::time::Duration;

fn owned_head(
    weights: Vec<f64>,
    bias: Vec<f64>,
    hidden: usize,
) -> (SharedAffineHead, VocabOptions) {
    let ledger =
        WorkBudget::new(None, 32 << 20, 256 << 20, 1 << 34, 100_000, Duration::ZERO).unwrap();
    let limits = VocabOptions {
        budget: Some(ledger),
        ..VocabOptions::default()
    };
    let owner = SharedAffineHead::new(
        bias.len(),
        hidden,
        weights,
        bias,
        HeadArithmetic::ExactReal,
        &limits,
    )
    .unwrap();
    (owner, limits)
}

fn fixture(vocab: usize, target: usize) -> (SharedAffineHead, VocabOptions) {
    let weights = (0..vocab)
        .flat_map(|id| {
            [
                (id % 7) as f64 * 0.0625 + 2f64.powi(-40),
                -((id % 5) as f64) * 0.03125,
                (id % 3) as f64 * 0.125,
            ]
        })
        .collect();
    let mut bias = vec![-10.0; vocab];
    bias[target] = 0.0;
    owned_head(weights, bias, 3)
}

fn normal() -> FinalNormalization<'static> {
    FinalNormalization {
        kind: FinalNormalizationKind::LayerNorm,
        gamma: &[1.0, -0.5, 0.0],
        beta: Some(&[0.375, -0.125, 0.25]),
        epsilon: 1e-5,
    }
}
fn pre() -> HiddenBox<'static> {
    HiddenBox {
        lo: &[-0.25, 0.125, -0.5],
        hi: &[-0.125, 0.25, -0.375],
    }
}
fn query(
    owner: &SharedAffineHead,
    norm: FinalNormalization<'_>,
    pre: HiddenBox<'_>,
    target: usize,
    width: usize,
    limits: &VocabOptions,
) -> NormalizedGroupedHeadReport {
    let ledger = owner.budget();
    verify_shared_grouped_normalized_next_token(
        owner,
        norm,
        pre,
        target,
        NormalizedGroupPolicy { group_width: width },
        NormalizedHeadOptions {
            budget: &ledger,
            limits,
        },
    )
    .unwrap()
}
fn proof(report: &NormalizedGroupedHeadReport) -> &CertifiedGroupedNormalizedNextToken {
    match &report.outcome {
        NormalizedGroupedNextTokenOutcome::Certified(proof) => proof,
        _ => panic!("complete original positive control must certify"),
    }
}

#[test]
fn decoder_n12_groups_wire_complete_original_cover_target_in_group_and_fresh_replay() {
    let (owner, limits) = fixture(257, 13);
    let report = query(&owner, normal(), pre(), 13, 64, &limits);
    assert_eq!(proof(&report).covered_competitors(), 256);
    assert!(matches!(
        report.evidence.entries[0],
        NormalizedTerminalEvidence::OriginalGroup {
            start: 0,
            end: 64,
            ..
        }
    ));
    let export = export_shared_grouped_normalized_json(&owner, &report.evidence, &limits).unwrap();
    assert_eq!(export.stats().coverage_visits, 256);
    assert_eq!(
        export.stats().validated_entries,
        report.evidence.entries.len()
    );
    assert_eq!(
        export.stats().exported_entries,
        report.evidence.entries.len()
    );
    let plain: NormalizedGroupedHeadEvidence = serde_json::from_slice(export.as_bytes()).unwrap();
    assert_eq!(plain, report.evidence);
    let ledger = owner.budget();
    let before = ledger.receipt().unwrap();
    let replay = replay_shared_grouped_normalized_json(
        &owner,
        normal(),
        pre(),
        13,
        export.as_bytes(),
        &limits,
    )
    .unwrap();
    assert_eq!(replay.report.evidence, report.evidence);
    assert_eq!(proof(&replay.report).covered_competitors(), 256);
    assert_eq!(
        proof(&replay.report).provenance(),
        NextTokenProvenance::OriginalReplay
    );
    assert_eq!(proof(&replay.report).checked_stats(), &replay.report.stats);
    assert_eq!(replay.stats.input_byte_visits, export.as_bytes().len() * 2);
    assert_eq!(
        replay.stats.parsed_entry_visits,
        report.evidence.entries.len() * 2
    );
    assert_eq!(replay.stats.coverage_visits, 512);
    assert_eq!(
        replay.stats.original_identity_visits,
        (257 * 3 + 257 + 3 + 3 + 3) as u64
    );
    assert_eq!(
        ledger.receipt().unwrap().work_reserved - before.work_reserved,
        replay.stats.reserved_work + replay.report.stats.reserved_work
    );
    assert!(
        grouped_publication_work_bound(257, 3, 64).unwrap()
            >= export.stats().reserved_work
                + replay.stats.reserved_work
                + replay.report.stats.reserved_work
    );
    assert_eq!(ledger.receipt().unwrap().live_scratch_bytes, 0);
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Keep original certified evidence intact while mutating an independent proposal."
)]
fn decoder_n12_groups_wire_original_rows_ignore_taylor_method_as_arithmetic_authority() {
    let (owner, limits) = owned_head(vec![0.0, 1.0], vec![0.0, -1.1], 1);
    let norm = FinalNormalization {
        kind: FinalNormalizationKind::RmsNorm,
        gamma: &[1.0],
        beta: None,
        epsilon: 1e-5,
    };
    let pre = HiddenBox {
        lo: &[-1.0],
        hi: &[1.0],
    };
    let report = query(&owner, norm, pre, 0, 8, &limits);
    proof(&report);
    assert_eq!(report.stats.direct_rows, 1);
    let mut proposal = report.evidence.clone();
    let NormalizedTerminalEvidence::OriginalRow { method, .. } = &mut proposal.entries[0] else {
        panic!("paid direct row expected");
    };
    *method = NormalizedSupportMethod::RadialTaylor;
    let export = export_shared_grouped_normalized_json(&owner, &proposal, &limits).unwrap();
    let replay =
        replay_shared_grouped_normalized_json(&owner, norm, pre, 0, export.as_bytes(), &limits)
            .unwrap();
    proof(&replay.report);
    assert_eq!(replay.report.evidence, proposal);
    assert_eq!(replay.report.stats.original.cauchy_screened_rows, 1);
    assert_eq!(replay.report.stats.original.radial_taylor_attempted_rows, 0);
    assert_eq!(
        proof(&replay.report)
            .checked_stats()
            .original
            .radial_taylor_attempted_rows,
        0
    );
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Keep original certified evidence intact while mutating an independent proposal."
)]
fn decoder_n12_groups_wire_member_rescue_is_recomputed_from_original_rows() {
    let (owner, limits) = owned_head(vec![0.0, 0.0, 2.0], vec![0.0, -0.125, -2.0], 1);
    let norm = FinalNormalization {
        kind: FinalNormalizationKind::RmsNorm,
        gamma: &[1.0],
        beta: None,
        epsilon: 1.0,
    };
    let pre = HiddenBox {
        lo: &[-0.25],
        hi: &[0.25],
    };
    let report = query(&owner, norm, pre, 0, 8, &limits);
    assert_eq!(report.stats.member_rescues_closed_groups, 1);
    let export = export_shared_grouped_normalized_json(&owner, &report.evidence, &limits).unwrap();
    let replay =
        replay_shared_grouped_normalized_json(&owner, norm, pre, 0, export.as_bytes(), &limits)
            .unwrap();
    assert_eq!(proof(&replay.report).covered_competitors(), 2);
    assert_eq!(replay.report.stats.member_rescues_completed, 1);
    assert_eq!(replay.report.evidence, report.evidence);
    let mut forged = report.evidence.clone();
    let NormalizedTerminalEvidence::OriginalGroup { upper, .. } = &mut forged.entries[0] else {
        panic!("rescued original group expected");
    };
    *upper = -f64::MAX;
    let bad = export_shared_grouped_normalized_json(&owner, &forged, &limits).unwrap();
    assert!(
        replay_shared_grouped_normalized_json(&owner, norm, pre, 0, bad.as_bytes(), &limits)
            .is_err()
    );
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Keep original certified evidence intact while mutating an independent proposal."
)]
fn decoder_n12_groups_wire_forged_negative_bounds_and_exact_ties_cannot_mint() {
    let (owner, limits) = owned_head(vec![0.0; 4], vec![0.0, -1.0, -1.0, -1.0], 1);
    let norm = FinalNormalization {
        kind: FinalNormalizationKind::LayerNorm,
        gamma: &[1.0],
        beta: None,
        epsilon: 1e-5,
    };
    let pre = HiddenBox {
        lo: &[0.0],
        hi: &[0.0],
    };
    let report = query(&owner, norm, pre, 0, 8, &limits);
    let prior = proof(&report).clone();
    let mut forged = report.evidence.clone();
    for entry in &mut forged.entries {
        match entry {
            NormalizedTerminalEvidence::OriginalGroup { upper, .. }
            | NormalizedTerminalEvidence::OriginalRow { upper, .. } => *upper = -1000.0,
        }
    }
    let export = export_shared_grouped_normalized_json(&owner, &forged, &limits).unwrap();
    assert!(replay_shared_grouped_normalized_json(
        &owner,
        norm,
        pre,
        0,
        export.as_bytes(),
        &limits
    )
    .is_err());
    assert_eq!(prior.provenance(), NextTokenProvenance::CheckedQuery);
    let (tie_owner, tie_limits) = owned_head(vec![0.0; 2], vec![0.0; 2], 1);
    let ties = query(&tie_owner, norm, pre, 0, 8, &tie_limits);
    assert!(matches!(
        ties.outcome,
        NormalizedGroupedNextTokenOutcome::Unknown
    ));
    let export =
        export_shared_grouped_normalized_json(&tie_owner, &ties.evidence, &tie_limits).unwrap();
    let replay = replay_shared_grouped_normalized_json(
        &tie_owner,
        norm,
        pre,
        0,
        export.as_bytes(),
        &tie_limits,
    )
    .unwrap();
    assert!(matches!(
        replay.report.outcome,
        NormalizedGroupedNextTokenOutcome::Unknown
    ));
    let mut attempt = ties.evidence.clone();
    attempt.entries = vec![NormalizedTerminalEvidence::OriginalGroup {
        start: 1,
        end: 2,
        upper: -f64::MIN_POSITIVE,
    }];
    let export = export_shared_grouped_normalized_json(&tie_owner, &attempt, &tie_limits).unwrap();
    assert!(replay_shared_grouped_normalized_json(
        &tie_owner,
        norm,
        pre,
        0,
        export.as_bytes(),
        &tie_limits
    )
    .is_err());
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Keep original certified evidence intact while mutating an independent proposal."
)]
fn decoder_n12_groups_wire_canonical_membership_and_closed_header_reject_before_materialization() {
    let (owner, limits) = fixture(17, 5);
    let report = query(&owner, normal(), pre(), 5, 8, &limits);
    let text = serde_json::to_string(&report.evidence).unwrap();
    let mut cases = Vec::new();
    let mut e = report.evidence.clone();
    e.entries.pop();
    cases.push(serde_json::to_vec(&e).unwrap());
    e = report.evidence.clone();
    e.entries.swap(0, 1);
    cases.push(serde_json::to_vec(&e).unwrap());
    e = report.evidence.clone();
    e.entries.push(e.entries[0].clone());
    cases.push(serde_json::to_vec(&e).unwrap());
    e = report.evidence.clone();
    e.entries[0] = NormalizedTerminalEvidence::OriginalGroup {
        start: 1,
        end: 8,
        upper: -1.0,
    };
    cases.push(serde_json::to_vec(&e).unwrap());
    e = report.evidence.clone();
    e.entries[0] = NormalizedTerminalEvidence::OriginalGroup {
        start: 0,
        end: usize::MAX,
        upper: -1.0,
    };
    cases.push(serde_json::to_vec(&e).unwrap());
    e = report.evidence.clone();
    e.entries[0] = NormalizedTerminalEvidence::OriginalRow {
        competitor: 5,
        upper: -1.0,
        method: NormalizedSupportMethod::ProjectedDense,
    };
    cases.push(serde_json::to_vec(&e).unwrap());
    for (from, to) in [
        ("\"schema_version\":2", "\"schema_version\":1"),
        ("\"group_width\":8", "\"group_width\":0"),
        ("\"vocab_size\":17", "\"vocab_size\":1000000000"),
        ("\"target\":5", "\"target\":0"),
        ("OriginalGroup", "CachedGeometry"),
        ("scope_digest", "scope\\u005fdigest"),
        ("LayerNorm", "RmsNorm"),
    ] {
        cases.push(text.replace(from, to).into_bytes());
    }
    cases.push(text.replacen("{", "{\"schema_version\":2,", 1).into_bytes());
    cases.push(
        text.replace("\"entries\":[", "\"extra\":[],\"entries\":[")
            .into_bytes(),
    );
    cases.push(format!("{text}[]").into_bytes());
    cases.push(
        text.replace("\"upper\":", "\"upper\":NaN,\"ignored\":")
            .into_bytes(),
    );
    for bytes in cases {
        let ledger = owner.budget();
        let before = ledger.receipt().unwrap();
        assert!(
            replay_shared_grouped_normalized_json(&owner, normal(), pre(), 5, &bytes, &limits)
                .is_err()
        );
        let after = ledger.receipt().unwrap();
        assert_eq!(
            after.admitted,
            before.admitted + 1,
            "metadata-valid input spends one bounded cover phase, not a replay"
        );
        assert_eq!(after.completed, before.completed);
        assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
        assert_eq!(after.live_scratch_bytes, before.live_scratch_bytes);
        assert!(after.work_reserved > before.work_reserved);
    }
}

#[test]
fn decoder_n12_groups_wire_original_head_norm_box_scope_is_checked_before_expansion() {
    let (owner, limits) = fixture(17, 0);
    let report = query(&owner, normal(), pre(), 0, 8, &limits);
    let bytes = serde_json::to_vec(&report.evidence).unwrap();
    for norm in [
        FinalNormalization {
            gamma: &[1.0, -0.25, 0.0],
            ..normal()
        },
        FinalNormalization {
            epsilon: 2e-5,
            ..normal()
        },
        FinalNormalization {
            beta: None,
            ..normal()
        },
    ] {
        let before = owner.budget().receipt().unwrap();
        assert!(
            replay_shared_grouped_normalized_json(&owner, norm, pre(), 0, &bytes, &limits).is_err()
        );
        let after = owner.budget().receipt().unwrap();
        assert_eq!(after.admitted, before.admitted + 1);
        assert_eq!(after.completed, before.completed);
        assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    }
    assert!(replay_shared_grouped_normalized_json(
        &owner,
        normal(),
        HiddenBox {
            lo: &[-0.5, 0.125, -0.5],
            ..pre()
        },
        0,
        &bytes,
        &limits
    )
    .is_err());
    let mut changed = owner.as_head().weights.to_vec();
    changed[2] += 0.125;
    let replacement = SharedAffineHead::new(
        17,
        3,
        changed,
        owner.as_head().bias.to_vec(),
        HeadArithmetic::ExactReal,
        &limits,
    )
    .unwrap();
    let before = owner.budget().receipt().unwrap();
    assert!(replay_shared_grouped_normalized_json(
        &replacement,
        normal(),
        pre(),
        0,
        &bytes,
        &limits
    )
    .is_err());
    assert_eq!(
        owner.budget().receipt().unwrap().admitted,
        before.admitted + 1
    );
    assert_eq!(
        proof(&report).provenance(),
        NextTokenProvenance::CheckedQuery
    );
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Keep the original options and ledger owner while exercising refusing alternatives."
)]
fn decoder_n12_groups_wire_limits_foreign_ledger_and_encoded_caps_decline_without_new_allowance() {
    let (owner, limits) = fixture(257, 0);
    let report = query(&owner, normal(), pre(), 0, 64, &limits);
    let export = export_shared_grouped_normalized_json(&owner, &report.evidence, &limits).unwrap();
    let ledger = owner.budget();
    let before = ledger.receipt().unwrap();
    let mut tight = limits.clone();
    tight.max_evidence_bytes = 32;
    assert!(export_shared_grouped_normalized_json(&owner, &report.evidence, &tight).is_err());
    assert!(replay_shared_grouped_normalized_json(
        &owner,
        normal(),
        pre(),
        0,
        export.as_bytes(),
        &tight
    )
    .is_err());
    // The original decoded evidence fits16KiB, but whitespace cannot hide an
    // oversized borrowed byte stream before the two-pass admission.
    tight = limits.clone();
    tight.max_evidence_bytes = 16 * 1024;
    let mut padded = vec![b' '; 32 * 1024];
    padded.extend_from_slice(export.as_bytes());
    assert!(matches!(
        replay_shared_grouped_normalized_json(&owner, normal(), pre(), 0, &padded, &tight),
        Err(NormalizedHeadRefusal::Invalid(DecoderRefusal::Resource(
            "grouped wire input byte cap"
        )))
    ));
    tight = limits.clone();
    tight.max_persistent_bytes =
        (before.live_persistent_bytes + grouped_wire_size_bound(257).unwrap() - 1) as u64;
    assert!(export_shared_grouped_normalized_json(&owner, &report.evidence, &tight).is_err());
    let request = decode_request(export.as_bytes().len(), 257, 3).unwrap();
    tight = limits.clone();
    tight.max_persistent_bytes =
        (before.live_persistent_bytes + request.persistent_bytes - 1) as u64;
    assert!(replay_shared_grouped_normalized_json(
        &owner,
        normal(),
        pre(),
        0,
        export.as_bytes(),
        &tight
    )
    .is_err());
    tight = limits.clone();
    tight.max_scratch_bytes = (request.scratch_bytes - 1) as u64;
    assert!(replay_shared_grouped_normalized_json(
        &owner,
        normal(),
        pre(),
        0,
        export.as_bytes(),
        &tight
    )
    .is_err());
    tight = limits.clone();
    tight.deadline = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
    assert!(matches!(
        replay_shared_grouped_normalized_json(
            &owner,
            normal(),
            pre(),
            0,
            export.as_bytes(),
            &tight
        ),
        Err(NormalizedHeadRefusal::Invalid(DecoderRefusal::Deadline))
    ));
    tight = limits.clone();
    tight.budget =
        Some(WorkBudget::new(None, 32 << 20, 256 << 20, 1 << 34, 100_000, Duration::ZERO).unwrap());
    assert!(export_shared_grouped_normalized_json(&owner, &report.evidence, &tight).is_err());
    assert!(replay_shared_grouped_normalized_json(
        &owner,
        normal(),
        pre(),
        0,
        export.as_bytes(),
        &tight
    )
    .is_err());
    assert_eq!(ledger.receipt().unwrap().admitted, before.admitted);
    assert_eq!(
        ledger.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
}

#[test]
fn decoder_n12_groups_wire_replayed_last_clone_retains_exact_owner_and_report_capacity() {
    let (owner, limits) = fixture(17, 0);
    let ledger = owner.budget();
    let report = query(&owner, normal(), pre(), 0, 8, &limits);
    let export = export_shared_grouped_normalized_json(&owner, &report.evidence, &limits).unwrap();
    assert!(export.reserved_bytes() >= export.as_bytes().len());
    let mut replay = replay_shared_grouped_normalized_json(
        &owner,
        normal(),
        pre(),
        0,
        export.as_bytes(),
        &limits,
    )
    .unwrap();
    let held = proof(&replay.report).clone();
    replay.report.stats.original.coverage_visits = 0;
    replay.report.evidence.scope_digest[0] ^= 1;
    assert_eq!(held.checked_stats().original.coverage_visits, 16);
    assert_eq!(held.provenance(), NextTokenProvenance::OriginalReplay);
    let expected = held.reserved_bytes() + held.shared_head_bytes();
    drop(replay);
    drop(export);
    drop(report);
    drop(owner);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, expected);
    assert_eq!(held.provenance(), NextTokenProvenance::OriginalReplay);
    let last = held.clone();
    drop(held);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, expected);
    let spent = ledger.receipt().unwrap().work_reserved;
    drop(last);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(ledger.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(ledger.receipt().unwrap().work_reserved, spent);
}

#[test]
fn decoder_n12_groups_wire_late_second_pass_cancel_preserves_prior_sealed_proof_and_spent_ledger() {
    let memory = MemoryAuthority::new(8 << 20);
    let cancellation = CancellationToken::default();
    let ledger = WorkBudget::new_in(
        &memory,
        &cancellation,
        WorkLimits {
            deadline: None,
            scratch_bytes: 2 << 20,
            persistent_bytes: 4 << 20,
            max_work: 1 << 34,
            max_proposals: 100_000,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    let limits = VocabOptions {
        budget: Some(ledger.clone()),
        ..VocabOptions::default()
    };
    let mut bias = vec![-1.0; 257];
    bias[0] = 0.0;
    let owner = SharedAffineHead::new(
        257,
        3,
        vec![0.0; 257 * 3],
        bias,
        HeadArithmetic::ExactReal,
        &limits,
    )
    .unwrap();
    let report = query(&owner, normal(), pre(), 0, 64, &limits);
    let sealed = proof(&report).clone();
    let export = export_shared_grouped_normalized_json(&owner, &report.evidence, &limits).unwrap();
    let before = ledger.receipt().unwrap();
    let result = cancel_after(Pass::Materialize, 32, cancellation.clone(), || {
        replay_shared_grouped_normalized_json(
            &owner,
            normal(),
            pre(),
            0,
            export.as_bytes(),
            &limits,
        )
    });
    assert!(result.is_err());
    assert!(cancellation.is_cancelled());
    let after = ledger.receipt().unwrap();
    assert_eq!(
        after.admitted,
        before.admitted + 1,
        "late parse failure never starts arithmetic replay"
    );
    assert!(after.work_reserved > before.work_reserved);
    assert_eq!(after.completed, before.completed);
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(after.live_scratch_bytes, before.live_scratch_bytes);
    assert_eq!(sealed.provenance(), NextTokenProvenance::CheckedQuery);
    assert_eq!(sealed.covered_competitors(), 256);
    drop(export);
    drop(report);
    drop(owner);
    drop(sealed);
    let released = memory.receipt().unwrap();
    assert_eq!(
        released.live_pending_bytes + released.live_materialized_bytes,
        0
    );
}

#[test]
fn decoder_n12_groups_wire_singleton_and_overflow_admission_are_bounded() {
    assert!(grouped_wire_size_bound(0).is_err());
    assert!(grouped_wire_size_bound(usize::MAX).is_err());
    assert!(grouped_publication_work_bound(usize::MAX, 3, 64).is_err());
    assert!(decode_request(0, 0, 1).is_err());
    assert!(decode_request(usize::MAX, 2, 1).is_err());
    let (owner, limits) = owned_head(vec![0.0], vec![0.0], 1);
    let norm = FinalNormalization {
        kind: FinalNormalizationKind::LayerNorm,
        gamma: &[1.0],
        beta: None,
        epsilon: 1e-5,
    };
    let pre = HiddenBox {
        lo: &[0.0],
        hi: &[0.0],
    };
    let report = query(&owner, norm, pre, 0, 64, &limits);
    let export = export_shared_grouped_normalized_json(&owner, &report.evidence, &limits).unwrap();
    let replay =
        replay_shared_grouped_normalized_json(&owner, norm, pre, 0, export.as_bytes(), &limits)
            .unwrap();
    assert_eq!(proof(&replay.report).covered_competitors(), 0);
    assert_eq!(replay.stats.parsed_entry_visits, 0);
    assert_eq!(replay.stats.coverage_visits, 0);
}

#[test]
fn decoder_n12_groups_wire_exhausted_original_ledger_cannot_refill_on_decode_retry() {
    let (owner, limits) = fixture(17, 0);
    let report = query(&owner, normal(), pre(), 0, 8, &limits);
    let export = export_shared_grouped_normalized_json(&owner, &report.evidence, &limits).unwrap();
    let ledger = owner.budget();
    let decode = decode_request(export.as_bytes().len(), 17, 3).unwrap();
    let spent = ledger.receipt().unwrap().work_reserved;
    let drain = ledger
        .lease_for(
            LeaseClass::Replay,
            WorkRequest {
                // Leave one fewer unit than the exact prepaid wire request. This is
                // a reservation-only negative control, not executed arithmetic work.
                work: (1 << 34) - spent - decode.work + 1,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    drop(drain);
    let before = ledger.receipt().unwrap();
    for _ in 0..2 {
        assert!(matches!(
            replay_shared_grouped_normalized_json(
                &owner,
                normal(),
                pre(),
                0,
                export.as_bytes(),
                &limits
            ),
            Err(NormalizedHeadRefusal::Invalid(DecoderRefusal::Budget(
                WorkRefusal::WorkLimit
            )))
        ));
        let after = ledger.receipt().unwrap();
        assert_eq!(after.admitted, before.admitted);
        assert_eq!(after.work_reserved, before.work_reserved);
        assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
        assert_eq!(after.completed, before.completed);
    }
    assert_eq!(proof(&report).covered_competitors(), 16);
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Keep original certified evidence intact while mutating an independent proposal."
)]
fn decoder_n12_groups_wire_finite_ryu_extremes_fit_bounded_output_without_proof_authority() {
    let (owner, limits) = owned_head(vec![0.0; 2], vec![0.0, -1.0], 1);
    let norm = FinalNormalization {
        kind: FinalNormalizationKind::LayerNorm,
        gamma: &[1.0],
        beta: None,
        epsilon: 1e-5,
    };
    let pre = HiddenBox {
        lo: &[0.0],
        hi: &[0.0],
    };
    let report = query(&owner, norm, pre, 0, usize::MAX, &limits);
    let mut proposal = report.evidence.clone();
    proposal.entries = vec![NormalizedTerminalEvidence::OriginalGroup {
        start: 1,
        end: 2,
        upper: f64::MAX,
    }];
    let export = export_shared_grouped_normalized_json(&owner, &proposal, &limits).unwrap();
    assert!(export.as_bytes().len() <= grouped_wire_size_bound(2).unwrap());
    let replay =
        replay_shared_grouped_normalized_json(&owner, norm, pre, 0, export.as_bytes(), &limits)
            .unwrap();
    assert!(matches!(
        replay.report.outcome,
        NormalizedGroupedNextTokenOutcome::Unknown
    ));
    proposal.entries[0] = NormalizedTerminalEvidence::OriginalGroup {
        start: 1,
        end: 2,
        upper: -f64::MAX,
    };
    let export = export_shared_grouped_normalized_json(&owner, &proposal, &limits).unwrap();
    assert!(replay_shared_grouped_normalized_json(
        &owner,
        norm,
        pre,
        0,
        export.as_bytes(),
        &limits
    )
    .is_err());
}
