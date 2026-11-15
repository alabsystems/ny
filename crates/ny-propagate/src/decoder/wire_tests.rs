// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::certify::work::{WorkBudget, WorkRefusal};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

fn head<'a>(n: usize, d: usize, weights: &'a [f64], bias: &'a [f64]) -> AffineHead<'a> {
    AffineHead {
        vocab_size: n,
        hidden_size: d,
        weights,
        bias,
        arithmetic: HeadArithmetic::ExactReal,
    }
}

fn fixture() -> (Vec<f64>, Vec<f64>, VocabOptions) {
    let weights = vec![0.0; 8];
    let mut bias = vec![0.0; 8];
    bias[3] = 2.0;
    (weights, bias, VocabOptions::default())
}

fn compact_fixture() -> Value {
    let (weights, bias, options) = fixture();
    let report = verify_dense_next_token(
        head(8, 1, &weights, &bias),
        HiddenBox {
            lo: &[-0.1],
            hi: &[0.1],
        },
        &[0.0],
        3,
        &options,
    )
    .unwrap();
    serde_json::from_str(&report.evidence.to_json().unwrap()).unwrap()
}

fn replay_fixture(json: &[u8], options: &VocabOptions) -> Result<NextTokenReport, DecoderRefusal> {
    let (weights, bias, _) = fixture();
    replay_next_token_json(
        head(8, 1, &weights, &bias),
        HiddenBox {
            lo: &[-0.1],
            hi: &[0.1],
        },
        &[0.0],
        3,
        json,
        options,
    )
}

#[test]
fn decoder_n8_v1_compatibility_and_v2_replay_preserve_every_original_bit() {
    let weights = [0.1, -0.3, 0.1, 0.2, -0.7, 0.4, 0.2, -0.5];
    let bias = [-2.0, 0.7, -2.0, -2.0];
    let options = VocabOptions {
        leaf_size: 2,
        ..VocabOptions::default()
    };
    for arithmetic in [
        HeadArithmetic::ExactReal,
        HeadArithmetic::SequentialBinary64,
    ] {
        let h = AffineHead {
            arithmetic,
            ..head(4, 2, &weights, &bias)
        };
        let hidden = HiddenBox {
            lo: &[-0.4, -0.2],
            hi: &[0.5, 0.25],
        };
        let report = verify_next_token(h, hidden, &[0.0, 0.1], 1, &options).unwrap();
        assert!(report.is_certified());
        assert!(report
            .evidence
            .groups
            .iter()
            .any(|g| g.group_bounds.is_some()));
        let v1 = report.evidence.to_json_v1().unwrap();
        let direct_v1 = serde_json::to_string(&report.evidence).unwrap();
        assert_eq!(
            v1, direct_v1,
            "plain serde remains the v1 compatibility API"
        );
        let v2 = report.evidence.to_json().unwrap();
        let value: Value = serde_json::from_str(&v2).unwrap();
        assert_eq!(value["schema_version"], 2);
        assert!(value["groups"]
            .as_array()
            .unwrap()
            .iter()
            .all(|g| g.get("members").is_none()));
        let mut streamed = Vec::new();
        report.evidence.write_json(&mut streamed).unwrap();
        assert_eq!(streamed, v2.as_bytes());
        for encoded in [v1.as_bytes(), v2.as_bytes()] {
            let replay =
                replay_next_token_json(h, hidden, &[0.0, 0.1], 1, encoded, &options).unwrap();
            assert!(replay.is_certified());
            assert_eq!(replay.evidence, report.evidence);
            for (expected, actual) in report.evidence.groups.iter().zip(&replay.evidence.groups) {
                assert_eq!(expected.upper.to_bits(), actual.upper.to_bits());
            }
        }
        let mut poisoned: Value = serde_json::from_str(&v2).unwrap();
        let summary = poisoned["groups"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|group| !group["group_bounds"].is_null())
            .unwrap();
        summary["group_bounds"]["radius_l1"] = json!(99.0);
        assert!(replay_next_token_json(
            h,
            hidden,
            &[0.0, 0.1],
            1,
            &serde_json::to_vec(&poisoned).unwrap(),
            &options,
        )
        .is_err());
    }
}

#[test]
fn decoder_n8_large_original_heads_replay_compact_wire_with_unchanged_defaults() {
    for n in [320_000usize, 1_000_000] {
        let weights = vec![0.0; n];
        let mut bias = vec![0.0; n];
        let target = n / 2;
        bias[target] = 2.0;
        let budget =
            WorkBudget::new(None, 32 << 20, 2 << 30, 1 << 34, 1_000_000, Duration::ZERO).unwrap();
        let options = VocabOptions {
            budget: Some(budget.clone()),
            ..VocabOptions::default()
        };
        let h = head(n, 1, &weights, &bias);
        let hidden = HiddenBox {
            lo: &[0.0],
            hi: &[0.0],
        };
        let report = verify_dense_next_token(h, hidden, &[0.0], target, &options).unwrap();
        assert!(report.is_certified());
        let v1 = report.evidence.to_json_v1().unwrap();
        assert!(v1.len() > options.max_scratch_bytes as usize / 16);
        assert!(matches!(
            replay_next_token_json(h, hidden, &[0.0], target, v1.as_bytes(), &options),
            Err(DecoderRefusal::Resource("JSON admission"))
        ));
        let v2 = report.evidence.export_json(&options).unwrap();
        assert!(
            v2.as_bytes().len() < 1024,
            "two original-ID ranges, no explicit vocabulary IDs"
        );
        assert_eq!(v2.stats().member_visits, n - 1);
        assert_eq!(v2.stats().ranges_emitted, 2);
        assert_eq!(v2.stats().groups_emitted, 1);
        assert_eq!(v2.stats().encoded_bytes, v2.as_bytes().len());
        assert_eq!(
            budget.receipt().unwrap().live_scratch_bytes,
            v2.stats().leased_scratch_bytes
        );
        let replay =
            replay_next_token_json(h, hidden, &[0.0], target, v2.as_bytes(), &options).unwrap();
        assert!(replay.is_certified());
        let NextTokenOutcome::Certified(proof) = &replay.outcome else {
            unreachable!()
        };
        assert_eq!(proof.covered_competitors(), n - 1);
        assert_eq!(
            replay.stats.dense_rows,
            n - 1,
            "full ORIGINAL arithmetic is still performed"
        );
        assert_eq!(replay.evidence, report.evidence);
        assert_eq!(
            budget.receipt().unwrap().live_scratch_bytes,
            v2.stats().leased_scratch_bytes
        );
        drop(replay);
        drop(report);
        drop(v2);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    }
}

#[test]
fn decoder_n8_ranges_reject_overflow_order_overlap_target_holes_and_noncanonical_runs() {
    let original = compact_fixture();
    assert_eq!(
        original["groups"][0]["member_ranges"],
        json!([[0, 3], [4, 8]])
    );
    let poisons = [
        json!([[0, 3], [4, usize::MAX]]),
        json!([[0, 3], [8, 4]]),
        json!([[4, 8], [0, 3]]),
        json!([[0, 3], [2, 8]]),
        json!([[0, 8]]),
        json!([[0, 2], [4, 8]]),
        json!([[0, 1], [1, 3], [4, 8]]),
        json!([[0, 0], [0, 3], [4, 8]]),
        json!([]),
    ];
    for ranges in poisons {
        let mut value = original.clone();
        value["groups"][0]["member_ranges"] = ranges;
        assert!(replay_fixture(
            &serde_json::to_vec(&value).unwrap(),
            &VocabOptions::default()
        )
        .is_err());
    }
    // Total counts still match, but two groups overlap and leave a hole.
    let mut overlap = original;
    let mut second = overlap["groups"][0].clone();
    overlap["groups"][0]["member_ranges"] = json!([[0, 3]]);
    second["member_ranges"] = json!([[2, 3], [4, 7]]);
    second["anchor_token"] = json!(2);
    overlap["groups"].as_array_mut().unwrap().push(second);
    assert!(matches!(
        replay_fixture(
            &serde_json::to_vec(&overlap).unwrap(),
            &VocabOptions::default()
        ),
        Err(DecoderRefusal::Evidence("overlapping range groups"))
    ));
}

#[test]
fn decoder_n8_shape_and_expansion_caps_refuse_before_proportional_storage() {
    let original = compact_fixture();
    let budget = WorkBudget::new(None, 1 << 20, 1 << 20, 1 << 20, 32, Duration::ZERO).unwrap();
    let options = VocabOptions {
        budget: Some(budget.clone()),
        ..VocabOptions::default()
    };
    for field in ["vocab_size", "hidden_size", "target"] {
        let mut value = original.clone();
        value[field] = json!(usize::MAX);
        assert!(matches!(
            replay_fixture(&serde_json::to_vec(&value).unwrap(), &options),
            Err(DecoderRefusal::Evidence("JSON request or schema binding"))
        ));
    }
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);

    let n = 4096;
    let weights = vec![0.0; n];
    let mut bias = vec![0.0; n];
    bias[0] = 2.0;
    let h = head(n, 1, &weights, &bias);
    let hidden = HiddenBox {
        lo: &[0.0],
        hi: &[0.0],
    };
    let report = verify_dense_next_token(h, hidden, &[0.0], 0, &VocabOptions::default()).unwrap();
    let encoded = report.evidence.to_json().unwrap();
    let tiny = VocabOptions {
        max_scratch_bytes: (encoded.len() * 16 + 1024) as u64,
        ..VocabOptions::default()
    };
    assert!(matches!(
        replay_next_token_json(h, hidden, &[0.0], 0, encoded.as_bytes(), &tiny),
        Err(DecoderRefusal::Resource("range expansion memory cap"))
    ));
    // The cap belongs to the shared ledger too, rather than to fresh per-call allowances.
    let shared = WorkBudget::new(None, 20_000, 1 << 20, 1 << 24, 32, Duration::ZERO).unwrap();
    let shared_options = VocabOptions {
        budget: Some(shared.clone()),
        ..VocabOptions::default()
    };
    assert!(matches!(
        replay_next_token_json(h, hidden, &[0.0], 0, encoded.as_bytes(), &shared_options),
        Err(DecoderRefusal::Budget(WorkRefusal::ScratchLimit))
    ));
    assert_eq!(shared.receipt().unwrap().live_scratch_bytes, 0);
}

#[test]
fn decoder_n8_compact_input_never_authorizes_changed_scope_or_arithmetic() {
    let original = compact_fixture();
    for field in ["anchor_token", "upper"] {
        let mut value = original.clone();
        value["groups"][0][field] = if field == "anchor_token" {
            json!(3)
        } else {
            json!(-99.0)
        };
        assert!(replay_fixture(
            &serde_json::to_vec(&value).unwrap(),
            &VocabOptions::default()
        )
        .is_err());
    }
    let mut scope = original.clone();
    scope["scope_digest"][0] = json!(scope["scope_digest"][0].as_u64().unwrap() ^ 1);
    assert!(replay_fixture(
        &serde_json::to_vec(&scope).unwrap(),
        &VocabOptions::default()
    )
    .is_err());
    let encoded = serde_json::to_vec(&original).unwrap();
    let (weights, mut bias, options) = fixture();
    bias[4] = 3.0;
    assert!(replay_next_token_json(
        head(8, 1, &weights, &bias),
        HiddenBox {
            lo: &[-0.1],
            hi: &[0.1]
        },
        &[0.0],
        3,
        &encoded,
        &options,
    )
    .is_err());
    let (_, original_bias, _) = fixture();
    let h = AffineHead {
        arithmetic: HeadArithmetic::SequentialBinary64,
        ..head(8, 1, &weights, &original_bias)
    };
    assert!(replay_next_token_json(
        h,
        HiddenBox {
            lo: &[-0.1],
            hi: &[0.1]
        },
        &[0.0],
        3,
        &encoded,
        &options
    )
    .is_err());
    let expired = VocabOptions {
        deadline: Some(Instant::now()),
        ..options
    };
    assert!(replay_fixture(&encoded, &expired).is_err());
}

#[test]
fn decoder_n8_ties_and_unsafe_original_rows_remain_unknown_and_singleton_is_vacuous() {
    for n in [1, 8] {
        let weights = vec![0.0; n];
        for challenger in [0.0, 3.0] {
            let mut bias = vec![challenger; n];
            bias[0] = 0.0;
            let h = head(n, 1, &weights, &bias);
            let options = VocabOptions::default();
            let hidden = HiddenBox {
                lo: &[0.0],
                hi: &[0.0],
            };
            let report = verify_dense_next_token(h, hidden, &[0.0], 0, &options).unwrap();
            let encoded = report.evidence.to_json().unwrap();
            let replay =
                replay_next_token_json(h, hidden, &[0.0], 0, encoded.as_bytes(), &options).unwrap();
            assert_eq!(replay.is_certified(), n == 1);
            assert_eq!(replay.evidence, report.evidence);
        }
    }
}

#[test]
fn decoder_n8_budgeted_export_polls_inside_one_long_run_and_retains_actual_byte_lease() {
    let n = 10_000usize;
    let evidence = VocabEvidence {
        schema_version: 1,
        scope_digest: [0; 32],
        target: 0,
        vocab_size: n,
        hidden_size: 1,
        groups: vec![GroupEvidence {
            members: (1..n).collect(),
            anchor_token: 1,
            group_bounds: None,
            upper: -1.0,
        }],
    };
    let budget = WorkBudget::new(None, 32 << 20, 1 << 20, 1 << 24, 32, Duration::ZERO).unwrap();
    let options = VocabOptions {
        budget: Some(budget.clone()),
        ..VocabOptions::default()
    };
    assert!(
        matches!(
            wire::with_export_stop_after_members(4096, || evidence.export_json(&options)),
            Err(DecoderRefusal::Deadline)
        ),
        "the long contiguous run must checkpoint before emitting its first range"
    );
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(budget.receipt().unwrap().completed, 0);
    let exported = evidence.export_json(&options).unwrap();
    assert_eq!(exported.as_bytes(), evidence.to_json().unwrap().as_bytes());
    assert_eq!(exported.stats().member_visits, n - 1);
    assert_eq!(exported.stats().ranges_emitted, 1);
    assert_eq!(exported.stats().groups_emitted, 1);
    assert_eq!(
        budget.receipt().unwrap().live_scratch_bytes,
        exported.stats().leased_scratch_bytes
    );
    drop(exported);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(budget.receipt().unwrap().completed, 1);
    assert!(matches!(
        evidence.export_json(&VocabOptions::default()),
        Err(DecoderRefusal::Resource("export requires shared ledger"))
    ));
    let tiny = VocabOptions {
        max_evidence_bytes: 8,
        ..options.clone()
    };
    assert!(matches!(
        evidence.export_json(&tiny),
        Err(DecoderRefusal::Resource("JSON export byte cap"))
    ));
    let small_budget = WorkBudget::new(None, 128, 1 << 20, 1 << 24, 32, Duration::ZERO).unwrap();
    let low = VocabOptions {
        budget: Some(small_budget.clone()),
        ..options
    };
    assert!(matches!(
        evidence.export_json(&low),
        Err(DecoderRefusal::Budget(WorkRefusal::ScratchLimit))
    ));
    assert_eq!(small_budget.receipt().unwrap().live_scratch_bytes, 0);
}
