// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use std::time::Duration;
fn setup(v: usize, h: usize) -> (SharedAffineHead, VocabOptions) {
    let budget =
        WorkBudget::new(None, 32 << 20, 256 << 20, 1 << 34, 10_000, Duration::ZERO).unwrap();
    let limits = VocabOptions {
        budget: Some(budget),
        ..VocabOptions::default()
    };
    let mut bias = vec![-1.0; v];
    bias[0] = 0.0;
    let owner = SharedAffineHead::new(
        v,
        h,
        vec![0.0; v * h],
        bias,
        HeadArithmetic::ExactReal,
        &limits,
    )
    .unwrap();
    (owner, limits)
}
fn norm(gamma: &[f64]) -> FinalNormalization<'_> {
    FinalNormalization {
        kind: FinalNormalizationKind::LayerNorm,
        gamma,
        beta: None,
        epsilon: 1e-5,
    }
}
#[test]
fn decoder_n12_wire_positive_complete_original_replay_and_observed_visits() {
    let (owner, limits) = setup(257, 3);
    let normal = norm(&[1.0, -2.0, 0.0]);
    let pre = HiddenBox {
        lo: &[-2.0, 0.0, -0.5],
        hi: &[1.0, 3.0, 0.25],
    };
    let report =
        shared::verify_shared_normalized_next_token(&owner, normal, pre, 0, &limits).unwrap();
    let exported = export_shared_normalized_json(&owner, &report.evidence, &limits).unwrap();
    let plain: NormalizedHeadEvidence = serde_json::from_slice(exported.as_bytes()).unwrap();
    assert_eq!(plain, report.evidence);
    assert_eq!(exported.stats().validated_rows, 256);
    assert_eq!(exported.stats().exported_rows, 256);
    let replay =
        replay_shared_normalized_json(&owner, normal, pre, 0, exported.as_bytes(), &limits)
            .unwrap();
    assert_eq!(replay.report.evidence, report.evidence);
    assert_eq!(
        replay.report.provenance(),
        NextTokenProvenance::OriginalReplay
    );
    assert!(matches!(
        replay.report.outcome,
        NormalizedNextTokenOutcome::Certified(_)
    ));
    assert_eq!(
        replay.stats.input_byte_visits,
        exported.as_bytes().len() * 2
    );
    assert_eq!(replay.stats.parsed_row_visits, 512);
    assert_eq!(
        replay.stats.original_identity_visits,
        (257 * 3 + 257 + 3 + 3) as u64
    );
    assert!(
        normalized_publication_work_bound(257, 3).unwrap()
            >= exported.stats().reserved_work
                + replay.stats.reserved_work
                + replay.report.stats.reserved_work
    );
    assert_eq!(owner.budget().receipt().unwrap().live_scratch_bytes, 0);
}
#[test]
fn decoder_n12_wire_rejects_forged_bounds_and_scope_before_any_private_mint() {
    let (owner, limits) = setup(3, 1);
    let normal = norm(&[1.0]);
    let pre = HiddenBox {
        lo: &[-1.0],
        hi: &[1.0],
    };
    let report =
        shared::verify_shared_normalized_next_token(&owner, normal, pre, 0, &limits).unwrap();
    let mut evidence = report.evidence.clone();
    evidence.rows[0].upper = -1000.0;
    let bytes = serde_json::to_vec(&evidence).unwrap();
    assert!(replay_shared_normalized_json(&owner, normal, pre, 0, &bytes, &limits).is_err());
    evidence = report.evidence.clone();
    evidence.scope_digest[0] ^= 1;
    let before = owner.budget().receipt().unwrap().admitted;
    assert!(replay_shared_normalized_json(
        &owner,
        normal,
        pre,
        0,
        &serde_json::to_vec(&evidence).unwrap(),
        &limits
    )
    .is_err());
    assert_eq!(
        owner.budget().receipt().unwrap().admitted,
        before + 1,
        "original binding refuses before core replay allocation"
    );
    let NormalizedNextTokenOutcome::Certified(sealed) = &report.outcome else {
        panic!("safe")
    };
    assert_eq!(sealed.provenance(), NextTokenProvenance::CheckedQuery);
}
#[test]
fn decoder_n12_wire_refuses_duplicate_missing_extra_reordered_and_hidden_fields() {
    let (owner, limits) = setup(4, 1);
    let normal = norm(&[1.0]);
    let pre = HiddenBox {
        lo: &[0.0],
        hi: &[0.0],
    };
    let report =
        shared::verify_shared_normalized_next_token(&owner, normal, pre, 0, &limits).unwrap();
    let mut cases = Vec::new();
    let mut e = report.evidence.clone();
    e.rows[1].competitor = 1;
    cases.push(serde_json::to_vec(&e).unwrap());
    e = report.evidence.clone();
    e.rows.pop();
    cases.push(serde_json::to_vec(&e).unwrap());
    e = report.evidence.clone();
    e.rows.push(e.rows[0].clone());
    cases.push(serde_json::to_vec(&e).unwrap());
    e = report.evidence.clone();
    e.rows.swap(0, 1);
    cases.push(serde_json::to_vec(&e).unwrap());
    let text = serde_json::to_string(&report.evidence).unwrap();
    cases.push(text.replacen("{", "{\"target\":0,", 1).into_bytes());
    cases.push(text.replacen("{", "{\"hidden_extra\":[],", 1).into_bytes());
    cases.push(text.replace("\"target\":0", "\"target\":1").into_bytes());
    cases.push(
        text.replace("\"vocab_size\":4", "\"vocab_size\":400000")
            .into_bytes(),
    );
    cases.push(text.replace("LayerNorm", "RmsNorm").into_bytes());
    cases.push(
        text.replace("schema_version", "schema\\u005fversion")
            .into_bytes(),
    );
    for bytes in cases {
        let before = owner.budget().receipt().unwrap().admitted;
        assert!(replay_shared_normalized_json(&owner, normal, pre, 0, &bytes, &limits).is_err());
        assert_eq!(owner.budget().receipt().unwrap().admitted, before + 1);
    }
}
#[test]
fn decoder_n12_wire_original_norm_and_box_parameter_changes_refuse_replay() {
    let (owner, limits) = setup(2, 2);
    let normal = norm(&[1.0, 1.0]);
    let pre = HiddenBox {
        lo: &[-1.0, -1.0],
        hi: &[1.0, 1.0],
    };
    let report =
        shared::verify_shared_normalized_next_token(&owner, normal, pre, 0, &limits).unwrap();
    let bytes = serde_json::to_vec(&report.evidence).unwrap();
    for changed in [
        norm(&[1.0, 2.0]),
        FinalNormalization {
            epsilon: 2e-5,
            ..normal
        },
        FinalNormalization {
            beta: Some(&[0.0, 0.0]),
            ..normal
        },
    ] {
        assert!(replay_shared_normalized_json(&owner, changed, pre, 0, &bytes, &limits).is_err());
    }
    assert!(replay_shared_normalized_json(
        &owner,
        normal,
        HiddenBox {
            lo: &[-2.0, -1.0],
            hi: &[1.0, 1.0]
        },
        0,
        &bytes,
        &limits
    )
    .is_err());
}
#[test]
fn decoder_n12_wire_aggregate_caps_deadlines_and_original_ledger_refuse_before_allocations() {
    let (owner, limits) = setup(257, 1);
    let normal = norm(&[1.0]);
    let pre = HiddenBox {
        lo: &[0.0],
        hi: &[0.0],
    };
    let report =
        shared::verify_shared_normalized_next_token(&owner, normal, pre, 0, &limits).unwrap();
    let budget = owner.budget();
    let before = budget.receipt().unwrap();
    let mut tight = limits.clone();
    tight.max_persistent_bytes =
        (before.live_persistent_bytes + normalized_wire_size_bound(257).unwrap() - 1) as u64;
    assert!(export_shared_normalized_json(&owner, &report.evidence, &tight).is_err());
    tight = limits.clone();
    tight.max_evidence_bytes = 32;
    assert!(export_shared_normalized_json(&owner, &report.evidence, &tight).is_err());
    tight = limits.clone();
    tight.deadline = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
    assert!(matches!(
        export_shared_normalized_json(&owner, &report.evidence, &tight),
        Err(NormalizedHeadRefusal::Invalid(DecoderRefusal::Deadline))
    ));
    tight = limits.clone();
    tight.budget =
        Some(WorkBudget::new(None, 32 << 20, 256 << 20, 1 << 34, 100, Duration::ZERO).unwrap());
    assert!(export_shared_normalized_json(&owner, &report.evidence, &tight).is_err());
    assert!(replay_shared_normalized_json(
        &owner,
        normal,
        pre,
        0,
        &serde_json::to_vec(&report.evidence).unwrap(),
        &tight
    )
    .is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    // A succinct declared giant vocabulary cannot widen the request's actual
    // dimensions or allocate a giant decoded row cohort.
    let fake = b"{\"vocab_size\":1000000,\"rows\":[]}";
    assert!(replay_shared_normalized_json(&owner, normal, pre, 0, fake, &limits).is_err());
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
}
#[test]
fn decoder_n12_wire_finite_ryu_extremes_fit_bound_but_do_not_gain_arithmetic_authority() {
    let (owner, limits) = setup(2, 1);
    let normal = norm(&[1.0]);
    let pre = HiddenBox {
        lo: &[0.0],
        hi: &[0.0],
    };
    let report =
        shared::verify_shared_normalized_next_token(&owner, normal, pre, 0, &limits).unwrap();
    let mut e = report.evidence;
    e.rows[0].upper = f64::MAX;
    let exported = export_shared_normalized_json(&owner, &e, &limits).unwrap();
    assert!(exported.as_bytes().len() <= normalized_wire_size_bound(2).unwrap());
    let replay =
        replay_shared_normalized_json(&owner, normal, pre, 0, exported.as_bytes(), &limits)
            .unwrap();
    assert!(matches!(
        replay.report.outcome,
        NormalizedNextTokenOutcome::Unknown
    ));
    e.rows[0].upper = -f64::MAX;
    let exported = export_shared_normalized_json(&owner, &e, &limits).unwrap();
    assert!(
        replay_shared_normalized_json(&owner, normal, pre, 0, exported.as_bytes(), &limits)
            .is_err()
    );
}
#[test]
fn decoder_n12_wire_writer_retains_typed_deadline_diagnostics() {
    let (owner, mut limits) = setup(2, 1);
    let budget = owner.budget();
    let lease = budget
        .lease_for(LeaseClass::Replay, WorkRequest::default())
        .unwrap();
    limits.deadline = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
    let mut writer = Writer {
        bytes: Vec::new(),
        cap: 32,
        lease: &lease,
        limits: &limits,
    };
    assert!(writer.write_all(b"{}").is_err());
    assert!(matches!(
        write_error(&lease, &limits),
        NormalizedHeadRefusal::Invalid(DecoderRefusal::Deadline)
    ));
}
