// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::certify::work::{WorkBudget, WorkRefusal};
use std::time::Duration;

fn fixture() -> (Vec<f64>, Vec<f64>, VocabOptions, WorkBudget) {
    let (n, d) = (64usize, 512usize);
    let mut weights = vec![0.0; n * d];
    weights[0] = 1.0;
    weights[d] = -1.0;
    let mut bias = vec![-10.0; n];
    bias[0] = 1.0;
    bias[1] = 1.0;
    let head_bytes = (weights.len() + bias.len()) * 8;
    let budget = WorkBudget::new(
        None,
        32 << 20,
        head_bytes * 3 / 2,
        1 << 28,
        128,
        Duration::ZERO,
    )
    .unwrap();
    let options = VocabOptions {
        budget: Some(budget.clone()),
        max_persistent_bytes: (head_bytes * 3 / 2) as u64,
        leaf_size: n,
        ..VocabOptions::default()
    };
    (weights, bias, options, budget)
}

fn box_for(target: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut lo = vec![0.0; 512];
    let mut hi = vec![0.0; 512];
    let mut anchor = vec![0.0; 512];
    if target == 0 {
        lo[0] = 2.0;
        hi[0] = 3.0;
        anchor[0] = 2.5;
    } else {
        lo[0] = -3.0;
        hi[0] = -2.0;
        anchor[0] = -2.5;
    }
    (lo, hi, anchor)
}

#[test]
fn decoder_n8_shared_head_moves_vectors_and_prices_spare_capacity_and_owner_lifetime() {
    let weights = vec![0.0; 32];
    let bias = vec![0.0; 8];
    let pointer = weights.as_ptr();
    let bias_pointer = bias.as_ptr();
    let budget = WorkBudget::new(None, 1 << 20, 1 << 20, 1 << 20, 16, Duration::ZERO).unwrap();
    let options = VocabOptions {
        budget: Some(budget.clone()),
        ..VocabOptions::default()
    };
    let head =
        SharedAffineHead::new(8, 4, weights, bias, HeadArithmetic::ExactReal, &options).unwrap();
    assert_eq!(
        head.as_head().weights.as_ptr(),
        pointer,
        "no giant matrix copy"
    );
    assert_eq!(head.as_head().bias.as_ptr(), bias_pointer);
    assert_eq!(head.identity_terms(), 40);
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        head.reserved_bytes()
    );
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    let retained = head.clone();
    drop(head);
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        retained.reserved_bytes()
    );
    drop(retained);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);

    let mut spare = vec![0.0; 32];
    spare.reserve_exact(10_000);
    let limited = VocabOptions {
        max_persistent_bytes: 4096,
        ..options
    };
    assert!(matches!(
        SharedAffineHead::new(
            8,
            4,
            spare,
            vec![0.0; 8],
            HeadArithmetic::ExactReal,
            &limited
        ),
        Err(DecoderRefusal::Resource("shared head memory cap"))
    ));
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn decoder_n8_shared_head_only_private_exact_owner_can_suppress_original_bytes() {
    let head = SharedAffineHead::new(
        2,
        1,
        vec![0.0; 2],
        vec![2.0, 0.0],
        HeadArithmetic::ExactReal,
        &VocabOptions::default(),
    )
    .unwrap();
    assert_eq!(
        shared_head::incremental_head_bytes(head.as_head(), Some(&head)).unwrap(),
        0
    );
    assert_eq!(
        shared_head::incremental_head_bytes(head.as_head(), None).unwrap(),
        32
    );
    let same_values = [0.0, 0.0];
    let substituted = AffineHead {
        weights: &same_values,
        ..head.as_head()
    };
    assert!(matches!(
        shared_head::incremental_head_bytes(substituted, Some(&head)),
        Err(DecoderRefusal::Evidence("shared original-head ownership"))
    ));
    let mut changed_identity = head.head_identity().clone();
    changed_identity.model.0[0] ^= 1;
    assert!(head.validate_identity(&changed_identity).is_err());
    let invalid = SharedAffineHead::new(
        2,
        1,
        vec![f64::NAN, 0.0],
        vec![0.0; 2],
        HeadArithmetic::ExactReal,
        &VocabOptions::default(),
    );
    assert!(matches!(
        invalid,
        Err(DecoderRefusal::NonFinite("original head"))
    ));
}

#[test]
fn decoder_n8_shared_head_avoids_repeated_head_charge_and_keeps_original_replay() {
    let (weights, bias, options, budget) = fixture();
    let head =
        SharedAffineHead::new(64, 512, weights, bias, HeadArithmetic::ExactReal, &options).unwrap();
    let shared_bytes = head.reserved_bytes();
    let tree = head.prepare_model_vocab_tree(&options).unwrap();
    assert_eq!(tree.shared_head_bytes(), shared_bytes);
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        shared_bytes + tree.reserved_bytes()
    );
    for target in [0, 1] {
        let (lo, hi, anchor) = box_for(target);
        let hidden = HiddenBox { lo: &lo, hi: &hi };
        let report = tree.verify(hidden, &anchor, target, &options).unwrap();
        assert!(report.is_certified());
        assert_eq!(report.report.shared_head_bytes(), shared_bytes);
        let encoded = head.export_json(&report.report.evidence, &options).unwrap();
        let replay = head
            .replay_next_token_json(hidden, &anchor, target, encoded.as_bytes(), &options)
            .unwrap();
        assert!(replay.is_certified());
        assert_eq!(replay.evidence, report.report.evidence);
        assert_eq!(replay.shared_head_bytes(), shared_bytes);
        assert!(
            replay.stats.query_terms >= head.identity_terms(),
            "independent replay still scans/hashes every original coefficient"
        );
        assert_eq!(
            budget.receipt().unwrap().live_persistent_bytes,
            shared_bytes
                + tree.reserved_bytes()
                + report.report.reserved_bytes()
                + replay.reserved_bytes()
        );
    }
    drop(tree);
    drop(head);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);

    // The unchanged borrowed API still prices its external original head for
    // both prepared geometry and independently replayed proof ownership.
    let (weights, bias, options, budget) = fixture();
    let borrowed = AffineHead {
        vocab_size: 64,
        hidden_size: 512,
        weights: &weights,
        bias: &bias,
        arithmetic: HeadArithmetic::ExactReal,
    };
    let tree = prepare_model_vocab_tree(borrowed, &options).unwrap();
    let (lo, hi, anchor) = box_for(0);
    let hidden = HiddenBox { lo: &lo, hi: &hi };
    let report = tree.verify(hidden, &anchor, 0, &options).unwrap();
    assert!(report.is_certified());
    assert!(matches!(
        tree.replay(hidden, &anchor, 0, &report.report.evidence, &options),
        Err(DecoderRefusal::Budget(WorkRefusal::PersistentLimit))
    ));
    drop(report);
    drop(tree);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn decoder_n8_shared_original_head_survives_tree_eviction_and_final_proof_owner() {
    let (weights, bias, options, budget) = fixture();
    let head =
        SharedAffineHead::new(64, 512, weights, bias, HeadArithmetic::ExactReal, &options).unwrap();
    let shared_bytes = head.reserved_bytes();
    let tree = head.prepare_vocab_tree(&box_for(0).2, 0, &options).unwrap();
    let (lo, hi, _) = box_for(0);
    let report = tree
        .verify(HiddenBox { lo: &lo, hi: &hi }, &options)
        .unwrap();
    let NextTokenOutcome::Certified(proof) = &report.outcome else {
        panic!("certified control")
    };
    let retained = proof.clone();
    let incremental = retained.reserved_bytes();
    assert_eq!(retained.shared_head_bytes(), shared_bytes);
    drop(report);
    drop(tree);
    drop(head);
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        shared_bytes + incremental
    );
    let another = retained.clone();
    drop(retained);
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        shared_bytes + incremental
    );
    drop(another);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn decoder_n8_shared_head_uses_its_ledger_and_refuses_changed_original_scope() {
    let (weights, bias, options, budget) = fixture();
    let head =
        SharedAffineHead::new(64, 512, weights, bias, HeadArithmetic::ExactReal, &options).unwrap();
    let fresh = WorkBudget::new(None, 32 << 20, 1 << 30, 1 << 30, 128, Duration::ZERO).unwrap();
    let replacement = VocabOptions {
        budget: Some(fresh.clone()),
        ..options
    };
    let (lo, hi, anchor) = box_for(0);
    let hidden = HiddenBox { lo: &lo, hi: &hi };
    let before = budget.receipt().unwrap();
    assert!(matches!(
        head.verify_next_token(hidden, &anchor, 0, &replacement),
        Err(DecoderRefusal::Resource(
            "shared head caller ledger mismatch"
        ))
    ));
    assert_eq!(budget.receipt().unwrap(), before);
    let report = head
        .verify_next_token(hidden, &anchor, 0, &options)
        .unwrap();
    assert!(report.is_certified());
    let replay = head
        .replay_next_token(hidden, &anchor, 0, &report.evidence, &options)
        .unwrap();
    assert!(replay.is_certified());
    assert_eq!(
        fresh.receipt().unwrap().attempted,
        0,
        "no reset of resident head ledger"
    );
    let live = budget.receipt().unwrap().live_persistent_bytes;
    let strict = VocabOptions {
        max_persistent_bytes: 1,
        ..options.clone()
    };
    assert!(head
        .replay_next_token(hidden, &anchor, 0, &report.evidence, &strict)
        .is_err());
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, live);
    let changed_anchor = vec![0.0; 512];
    assert!(head
        .replay_next_token(hidden, &changed_anchor, 0, &report.evidence, &options)
        .is_err());
    let changed = SharedAffineHead::new(
        64,
        512,
        vec![0.0; 64 * 512],
        vec![0.0; 64],
        HeadArithmetic::SequentialBinary64,
        &VocabOptions::default(),
    )
    .unwrap();
    assert!(changed
        .replay_next_token(
            hidden,
            &anchor,
            0,
            &report.evidence,
            &VocabOptions::default()
        )
        .is_err());
}

#[test]
fn decoder_n8_shared_dense_cold_route_replays_full_cover_and_retains_one_original_owner() {
    let (weights, bias, options, budget) = fixture();
    let head =
        SharedAffineHead::new(64, 512, weights, bias, HeadArithmetic::ExactReal, &options).unwrap();
    let shared = head.reserved_bytes();
    for target in [0, 1] {
        let (lo, hi, anchor) = box_for(target);
        let hidden = HiddenBox { lo: &lo, hi: &hi };
        let report = head
            .verify_dense_next_token(hidden, &anchor, target, &options)
            .unwrap();
        assert!(report.is_certified());
        assert_eq!(report.provenance(), NextTokenProvenance::OriginalReplay);
        assert_eq!(report.stats.dense_rows, 63);
        assert_eq!(report.evidence.groups.len(), 1);
        assert!(report.evidence.groups[0].group_bounds.is_none());
        assert_eq!(report.shared_head_bytes(), shared);
        assert_eq!(
            budget.receipt().unwrap().live_persistent_bytes,
            shared + report.reserved_bytes()
        );
        let encoded = head.export_json(&report.evidence, &options).unwrap();
        let replay = head
            .replay_next_token_json(hidden, &anchor, target, encoded.as_bytes(), &options)
            .unwrap();
        assert!(replay.is_certified());
        assert_eq!(replay.evidence, report.evidence);
        let mut poisoned = report.evidence;
        poisoned.groups[0].upper = f64::NEG_INFINITY;
        assert!(head
            .replay_next_token(hidden, &anchor, target, &poisoned, &options)
            .is_err());
    }
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, shared);
    drop(head);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn decoder_n8_shared_dense_cold_work_avoids_prepared_geometry_and_ties_stay_unknown() {
    let (n, d) = (256usize, 8usize);
    let weights = vec![0.0; n * d];
    let mut bias = vec![-2.0; n];
    bias[17] = 1.0;
    let options = VocabOptions::default();
    let head =
        SharedAffineHead::new(n, d, weights, bias, HeadArithmetic::ExactReal, &options).unwrap();
    let lo = vec![-1.0; d];
    let hi = vec![2.0; d];
    let anchor = vec![0.0; d];
    let hidden = HiddenBox { lo: &lo, hi: &hi };
    let dense = head
        .verify_dense_next_token(hidden, &anchor, 17, &options)
        .unwrap();
    let tree = head
        .verify_next_token(hidden, &anchor, 17, &options)
        .unwrap();
    assert!(dense.is_certified() && tree.is_certified());
    assert_eq!(dense.evidence.scope_digest, tree.evidence.scope_digest);
    let tree_replay = head
        .replay_next_token(hidden, &anchor, 17, &tree.evidence, &options)
        .unwrap();
    let dense_work = dense.stats.preprocessing_terms + dense.stats.query_terms;
    let tree_work =
        tree.stats.preprocessing_terms + tree.stats.query_terms + tree_replay.stats.query_terms;
    assert!(
        dense_work < tree_work,
        "same full cover and original replay must include cold geometry"
    );
    let tied = SharedAffineHead::new(
        2,
        1,
        vec![0.0; 2],
        vec![1.0; 2],
        HeadArithmetic::ExactReal,
        &VocabOptions::default(),
    )
    .unwrap();
    let tied_report = tied
        .verify_dense_next_token(
            HiddenBox {
                lo: &[0.0],
                hi: &[0.0],
            },
            &[0.0],
            0,
            &VocabOptions::default(),
        )
        .unwrap();
    assert!(!tied_report.is_certified());
    assert_eq!(tied_report.unresolved, vec![1]);
}

#[test]
fn decoder_n8_shared_session_exhaustion_cannot_be_reset_by_a_new_caller_ledger() {
    let budget = WorkBudget::new(None, 1 << 20, 1 << 20, 512, 128, Duration::ZERO).unwrap();
    let options = VocabOptions {
        budget: Some(budget.clone()),
        max_terms: 512,
        ..VocabOptions::default()
    };
    let head = SharedAffineHead::new(
        4,
        1,
        vec![0.0; 4],
        vec![2.0, 0.0, -1.0, -2.0],
        HeadArithmetic::ExactReal,
        &options,
    )
    .unwrap();
    let mut successes = 0;
    loop {
        match head.verify_dense_next_token(
            HiddenBox {
                lo: &[0.0],
                hi: &[0.0],
            },
            &[0.0],
            0,
            &options,
        ) {
            Ok(report) => {
                assert!(report.is_certified());
                successes += 1;
                drop(report);
            }
            Err(DecoderRefusal::Budget(WorkRefusal::WorkLimit)) => break,
            other => panic!("unexpected session result: {other:?}"),
        }
        assert!(successes < 100, "the bounded original session must exhaust");
    }
    assert!(successes > 0);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        head.reserved_bytes()
    );
    let fresh = WorkBudget::new(None, 1 << 20, 1 << 20, 1 << 20, 128, Duration::ZERO).unwrap();
    let replacement = VocabOptions {
        budget: Some(fresh.clone()),
        ..options
    };
    assert!(matches!(
        head.verify_dense_next_token(
            HiddenBox {
                lo: &[0.0],
                hi: &[0.0]
            },
            &[0.0],
            0,
            &replacement
        ),
        Err(DecoderRefusal::Resource(
            "shared head caller ledger mismatch"
        ))
    ));
    assert_eq!(fresh.receipt().unwrap().attempted, 0);
    drop(head);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
}
