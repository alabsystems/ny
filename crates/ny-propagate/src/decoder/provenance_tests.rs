// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;

fn fixture() -> SharedAffineHead {
    SharedAffineHead::new(
        4,
        1,
        vec![0.0; 4],
        vec![2.0, 0.0, 0.0, 0.0],
        HeadArithmetic::ExactReal,
        &VocabOptions::default(),
    )
    .unwrap()
}

#[test]
fn decoder_provenance_checked_query_and_original_replay_are_valid_distinct_receipts() {
    let head = fixture();
    let options = VocabOptions::default();
    let hidden = HiddenBox {
        lo: &[-1.0],
        hi: &[1.0],
    };
    let query = head.verify_next_token(hidden, &[0.0], 0, &options).unwrap();
    assert!(
        query.is_certified(),
        "the checked query retains certificate authority"
    );
    assert_eq!(query.provenance(), NextTokenProvenance::CheckedQuery);
    let NextTokenOutcome::Certified(query_proof) = &query.outcome else {
        unreachable!()
    };
    assert_eq!(query_proof.provenance(), NextTokenProvenance::CheckedQuery);
    let retained_query = query_proof.clone();
    let replay = head
        .replay_next_token(hidden, &[0.0], 0, &query.evidence, &options)
        .unwrap();
    assert!(replay.is_certified());
    assert_eq!(replay.provenance(), NextTokenProvenance::OriginalReplay);
    let NextTokenOutcome::Certified(replay_proof) = &replay.outcome else {
        unreachable!()
    };
    let retained_replay = replay_proof.clone();
    assert_eq!(retained_replay.scope(), retained_query.scope());
    assert_eq!(
        retained_replay.covered_competitors(),
        retained_query.covered_competitors()
    );
    drop(query);
    drop(replay);
    drop(head);
    assert_eq!(
        retained_query.provenance(),
        NextTokenProvenance::CheckedQuery
    );
    assert_eq!(
        retained_replay.provenance(),
        NextTokenProvenance::OriginalReplay
    );
    assert!(retained_query.shared_head_bytes() > 0);
    assert_eq!(
        retained_query.shared_head_bytes(),
        retained_replay.shared_head_bytes()
    );
}

#[test]
fn decoder_provenance_poisoned_raw_export_never_changes_sealed_query_or_mints_replay() {
    let head = fixture();
    let options = VocabOptions::default();
    let hidden = HiddenBox {
        lo: &[-1.0],
        hi: &[1.0],
    };
    let mut query = head.verify_next_token(hidden, &[0.0], 0, &options).unwrap();
    let NextTokenOutcome::Certified(proof) = &query.outcome else {
        unreachable!()
    };
    let sealed = proof.clone();
    query.evidence.groups[0].upper = -99.0;
    assert!(
        query.is_certified(),
        "sealed decision remains valid in its private scope"
    );
    assert_eq!(query.provenance(), NextTokenProvenance::CheckedQuery);
    assert_eq!(sealed.provenance(), NextTokenProvenance::CheckedQuery);
    assert!(head
        .replay_next_token(hidden, &[0.0], 0, &query.evidence, &options)
        .is_err());
    let raw = query.evidence.to_json().unwrap();
    assert!(head
        .replay_next_token_json(hidden, &[0.0], 0, raw.as_bytes(), &options)
        .is_err());
    let fresh = head.verify_next_token(hidden, &[0.0], 0, &options).unwrap();
    let replay = head
        .replay_next_token(hidden, &[0.0], 0, &fresh.evidence, &options)
        .unwrap();
    assert_eq!(replay.provenance(), NextTokenProvenance::OriginalReplay);
    assert!(replay.is_certified());
    assert!(head
        .replay_next_token(hidden, &[0.5], 0, &fresh.evidence, &options)
        .is_err());
    assert_eq!(
        sealed.scope(),
        match &fresh.outcome {
            NextTokenOutcome::Certified(p) => p.scope(),
            _ => unreachable!(),
        }
    );
}

#[test]
fn decoder_provenance_model_queries_and_replayed_unknowns_cannot_forge_origin_flags() {
    let head = fixture();
    let options = VocabOptions::default();
    let hidden = HiddenBox {
        lo: &[-1.0],
        hi: &[1.0],
    };
    let tree = head.prepare_model_vocab_tree(&options).unwrap();
    let query = tree.verify(hidden, &[0.0], 0, &options).unwrap();
    assert!(query.is_certified());
    assert_eq!(query.report.provenance(), NextTokenProvenance::CheckedQuery);
    let mut replay = tree
        .replay(hidden, &[0.0], 0, &query.report.evidence, &options)
        .unwrap();
    assert_eq!(replay.provenance(), NextTokenProvenance::OriginalReplay);
    let NextTokenOutcome::Certified(query_proof) = &query.report.outcome else {
        unreachable!()
    };
    replay.outcome = NextTokenOutcome::Certified(query_proof.clone());
    assert_eq!(
        replay.provenance(),
        NextTokenProvenance::CheckedQuery,
        "a public outcome replacement cannot borrow another sealed receipt's origin"
    );
    // Caller-supplied JSON flags are not a private checker receipt.
    let mut value: serde_json::Value =
        serde_json::from_str(&query.report.evidence.to_json().unwrap()).unwrap();
    value["provenance"] = serde_json::json!("OriginalReplay");
    assert!(head
        .replay_next_token_json(
            hidden,
            &[0.0],
            0,
            &serde_json::to_vec(&value).unwrap(),
            &options
        )
        .is_err());
    let tie = SharedAffineHead::new(
        2,
        1,
        vec![0.0; 2],
        vec![0.0; 2],
        HeadArithmetic::ExactReal,
        &options,
    )
    .unwrap();
    let open = tie.verify_next_token(hidden, &[0.0], 0, &options).unwrap();
    assert!(!open.is_certified());
    assert_eq!(open.provenance(), NextTokenProvenance::CheckedQuery);
    let open_replay = tie
        .replay_next_token(hidden, &[0.0], 0, &open.evidence, &options)
        .unwrap();
    assert!(!open_replay.is_certified());
    assert_eq!(
        open_replay.provenance(),
        NextTokenProvenance::OriginalReplay
    );
}
