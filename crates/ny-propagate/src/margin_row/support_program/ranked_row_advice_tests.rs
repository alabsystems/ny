// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Public policy/ownership flows use genuine completed sessions only.
//! Advice is compared as nominal data, never treated as numerical proof.

use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{LeaseClass, WorkBudget, WorkLimits, WorkRequest, WorkReserve};
use crate::margin_row::{
    PreparedRootCompilation, PreparedRootProofOptions, PreparedRootRetention, PreparedRootSession,
    RootPrecisionRuntimePolicy, TwinOpSpec, TwinSpec,
};
use std::mem::size_of;
use std::time::{Duration, Instant};

const CAP: usize = 128 << 20;

fn at() -> Instant {
    Instant::now() + Duration::from_secs(60)
}

fn resources() -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(2 * CAP);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: None,
            scratch_bytes: CAP,
            persistent_bytes: CAP,
            max_work: 100_000_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    (budget, memory, cancel)
}

fn live(memory: &MemoryAuthority) -> usize {
    let receipt = memory.receipt().unwrap();
    receipt.live_pending_bytes + receipt.live_materialized_bytes
}

fn conv(n: usize, input: usize, weight: f64, bias: f64) -> TwinOpSpec {
    TwinOpSpec::Conv {
        input,
        weight: vec![weight],
        bias: vec![bias],
        bias_err: vec![0.0],
        weight_rel_err: 0.0,
        kernel: (1, 1, 1, 1),
        stride: (1, 1),
        pads: (0, 0, 0, 0),
        ishape: (1, 1, n),
        oshape: (1, 1, n),
    }
}

fn fixture(n: usize, bias: f64) -> TwinSpec {
    TwinSpec {
        n_in: n,
        ops: vec![
            conv(n, 0, 1.0, bias),
            TwinOpSpec::Relu { input: 1 },
            conv(n, 2, 1.0, -0.1),
            TwinOpSpec::Relu { input: 3 },
            TwinOpSpec::Flatten { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![0.25; n],
                bias: vec![0.5],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (1, n),
            },
            TwinOpSpec::Relu { input: 6 },
            TwinOpSpec::Gemm {
                input: 7,
                weight: vec![0.5, -0.25],
                bias: vec![0.0; 2],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
        ],
    }
}

fn varied_fixture() -> TwinSpec {
    let mut spec = fixture(5, 0.0);
    for (op, scales, biases) in [
        (0, [0.0, 2.0, 0.0, 4.0, 0.0], [2.0, 0.0, 2.0, 0.0, 2.0]),
        (2, [1.0; 5], [-0.1; 5]),
    ] {
        let mut weight = vec![0.0; 25];
        for (i, &scale) in scales.iter().enumerate() {
            weight[i * 5 + i] = scale;
        }
        spec.ops[op] = TwinOpSpec::Conv {
            input: if op == 0 { 0 } else { 2 },
            weight,
            bias: biases.to_vec(),
            bias_err: vec![0.0; 5],
            weight_rel_err: 0.0,
            kernel: (5, 5, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            ishape: (5, 1, 1),
            oshape: (5, 1, 1),
        };
    }
    spec
}

fn compilation(spec: TwinSpec, budget: &WorkBudget) -> PreparedRootCompilation {
    PreparedRootCompilation::compile_declared_twin(spec, budget, Some(at())).unwrap()
}

fn domain(
    compiled: PreparedRootCompilation,
    budget: &WorkBudget,
    width: f64,
    keep_input_rows: bool,
) -> PreparedRootSession {
    let n = compiled.n_in();
    compiled
        .prepare_declared_domain(
            &vec![-width; n],
            &vec![width; n],
            RootPrecisionRuntimePolicy::default(),
            PreparedRootProofOptions {
                keep_input_rows,
                prefer_patch: false,
                dense_tile: 2,
                ..PreparedRootProofOptions::default()
            },
            budget,
            None,
        )
        .unwrap()
}

fn session(spec: TwinSpec, budget: &WorkBudget, keep: bool) -> PreparedRootSession {
    domain(compilation(spec, budget), budget, 1.0, keep)
}

fn request(per_layer: usize, rows: usize, n_in: usize) -> PreparedRootRetention {
    PreparedRootRetention {
        per_layer,
        budget_bytes: rows * 2 * (n_in + 1) * size_of::<f32>(),
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn ranked_advice_public_ties_global_policy_and_packs_match_original_producer() {
    let _env = crate::tests::lock_env_shared();
    assert!(!PreparedRootProofOptions::default().keep_input_rows);
    assert!(!PreparedRootProofOptions::default().row_f32);
    let (budget, memory, _) = resources();
    let source = session(fixture(5, -0.25), &budget, true);
    let row_bytes = 2 * 6 * size_of::<f32>();
    let cfg = PreparedRootRetention {
        per_layer: 3,
        budget_bytes: row_bytes * 4 + row_bytes - 1,
    };
    let advice = source.ranked_row_advice(cfg, None).unwrap();
    advice.validate_session(&source, None).unwrap();
    assert_eq!(advice.request().per_layer, cfg.per_layer);
    assert_eq!(advice.request().budget_bytes, cfg.budget_bytes);
    assert_eq!(advice.layer_count(), 2);
    assert_eq!(advice.layer(0, None).unwrap().ids(), &[0, 1, 2]);
    assert_eq!(advice.layer(1, None).unwrap().ids(), &[0]);
    assert_eq!(advice.payload_bytes(), row_bytes * 4);
    assert!(advice.payload_bytes() <= cfg.budget_bytes);
    assert!(advice.resident_bytes() >= advice.payload_bytes());
    let direct = super::build_prepared_retained(
        source.source(),
        super::RetainCfg {
            per_layer: cfg.per_layer,
            budget_bytes: cfg.budget_bytes,
        },
        None,
    )
    .unwrap();
    assert_eq!(advice.content_fingerprint(), direct.digest());
    let before = budget.receipt().unwrap();
    let bytes = live(&memory);
    let mut payload = 0;
    for layer_id in 0..advice.layer_count() {
        let view = advice.layer(layer_id, None).unwrap();
        let original = &direct.layers()[layer_id];
        assert_eq!(view.original_layer(), layer_id);
        assert_eq!(view.eligible(), 5);
        assert_eq!(view.ids(), original.ids());
        assert_eq!(view.unstable_positions(), original.unstable_positions());
        assert_eq!(view.selected_count(), view.ids().len());
        assert_eq!(view.row_width(), 6);
        assert_eq!(
            view.lower_advice().len(),
            view.selected_count() * view.row_width()
        );
        assert_eq!(view.upper_advice().len(), view.lower_advice().len());
        assert_eq!(bits(view.lower_advice()), bits(original.lower_advice()));
        assert_eq!(bits(view.upper_advice()), bits(original.upper_advice()));
        payload += view.payload_bytes();
    }
    assert_eq!(payload, advice.payload_bytes());
    assert!(advice.layer(2, None).is_err());
    assert!(advice.layer(usize::MAX, None).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    assert_eq!(
        budget.receipt().unwrap().work_reserved,
        before.work_reserved
    );
    assert_eq!(live(&memory), bytes);
}

#[test]
fn ranked_advice_slack_selection_returns_original_ids_and_distinct_unstable_positions() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let source = session(varied_fixture(), &budget, true);
    let advice = source.ranked_row_advice(request(2, 2, 5), None).unwrap();
    let first = advice.layer(0, None).unwrap();
    assert_eq!(first.eligible(), 2);
    assert_eq!(first.ids(), &[1, 3]);
    assert_eq!(first.unstable_positions(), &[0, 1]);
    // The greater-slack original ID 3 wins a one-row policy. With two rows,
    // the final stored order reverses score order into ascending original IDs.
    let one = source.ranked_row_advice(request(1, 1, 5), None).unwrap();
    assert_eq!(one.layer(0, None).unwrap().ids(), &[3]);
    let later = advice.layer(1, None).unwrap();
    assert_eq!(later.eligible(), 2);
    assert_eq!(later.selected_count(), 0);
    assert_eq!((later.lower_advice()).len(), 0);
    assert_eq!((later.upper_advice()).len(), 0);
}

#[test]
fn ranked_advice_zero_policy_keeps_census_and_missing_retention_still_refuses() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let source = session(fixture(3, -0.25), &budget, true);
    for cfg in [request(0, 10, 3), request(3, 0, 3)] {
        let advice = source.ranked_row_advice(cfg, None).unwrap();
        assert_eq!(advice.layer_count(), 2);
        assert_eq!(advice.payload_bytes(), 0);
        for layer in 0..2 {
            let view = advice.layer(layer, None).unwrap();
            assert_eq!(view.eligible(), 3);
            assert_eq!(view.selected_count(), 0);
            assert_eq!((view.ids()).len(), 0);
            assert_eq!((view.unstable_positions()).len(), 0);
            assert_eq!((view.lower_advice()).len(), 0);
            assert_eq!((view.upper_advice()).len(), 0);
        }
    }
    for spec in [fixture(3, -0.25), fixture(3, 2.0)] {
        let missing = session(spec, &budget, false);
        assert!(missing.ranked_row_advice(request(0, 0, 3), None).is_err());
        missing.checkpoint(None).unwrap();
    }
}

#[test]
fn ranked_advice_requires_exact_source_domain_and_original_ledger() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let compiled = compilation(fixture(3, -0.25), &budget);
    let first = domain(compiled.clone(), &budget, 1.0, true);
    let other_domain = domain(compiled, &budget, 0.5, true);
    assert_eq!(first.compiled_digest(), other_domain.compiled_digest());
    assert!(!first.same_source(&other_domain));
    let advice = first.ranked_row_advice(request(1, 2, 3), None).unwrap();
    advice
        .validate_session(&first.restricted(None).unwrap(), None)
        .unwrap();
    assert!(advice.validate_session(&other_domain, None).is_err());
    let equal_content = session(fixture(3, -0.25), &budget, true);
    assert_eq!(first.compiled_digest(), equal_content.compiled_digest());
    assert!(advice.validate_session(&equal_content, None).is_err());
    let (foreign_budget, _, _) = resources();
    let foreign = session(fixture(3, -0.25), &foreign_budget, true);
    assert!(!budget.shares_ledger(&foreign_budget));
    assert!(advice.validate_session(&foreign, None).is_err());
}

#[test]
fn ranked_advice_aliases_share_real_packs_and_release_only_with_last_source_owner() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let source = session(fixture(3, -0.25), &budget, true);
    let cfg = request(1, 2, 3);
    let advice = source.ranked_row_advice(cfg, None).unwrap();
    let other = source.ranked_row_advice(cfg, None).unwrap();
    assert!(!advice.same_owner(&other));
    assert_eq!(advice.content_fingerprint(), other.content_fingerprint());
    drop(other);
    let before = live(&memory);
    let receipt = budget.receipt().unwrap();
    let alias = advice.clone();
    assert!(advice.same_owner(&alias));
    assert_eq!(
        advice.layer(0, None).unwrap().lower_advice().as_ptr(),
        alias.layer(0, None).unwrap().lower_advice().as_ptr()
    );
    assert_eq!(budget.receipt().unwrap().admitted, receipt.admitted);
    assert_eq!(
        budget.receipt().unwrap().work_reserved,
        receipt.work_reserved
    );
    assert_eq!(live(&memory), before);
    drop(source);
    assert_eq!(live(&memory), before);
    drop(advice);
    {
        let snapshot = alias.layer(0, None).unwrap();
        assert_eq!(snapshot.selected_count(), 1);
        assert_eq!(snapshot.row_width(), 4);
        assert!(live(&memory) > 0);
    }
    drop(alias);
    assert_eq!(live(&memory), 0);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
}

#[test]
fn ranked_advice_preserves_restricted_and_requested_clocks_and_shared_cancellation() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources();
    let source = session(fixture(3, -0.25), &budget, true);
    let cfg = request(1, 2, 3);
    let session_at = source
        .deadline(None)
        .unwrap()
        .checked_sub(Duration::from_secs(10))
        .expect("test clock supports an earlier deadline");
    let restricted = source.restricted(Some(session_at)).unwrap();
    let advice = restricted
        .ranked_row_advice(cfg, Some(session_at + Duration::from_secs(1)))
        .unwrap();
    assert_eq!(advice.deadline(None).unwrap(), session_at);
    assert_eq!(
        advice
            .deadline(Some(session_at + Duration::from_secs(2)))
            .unwrap(),
        session_at
    );
    let requested = session_at
        .checked_sub(Duration::from_secs(1))
        .expect("test clock supports an earlier deadline");
    let earlier = source.ranked_row_advice(cfg, Some(requested)).unwrap();
    assert_eq!(earlier.deadline(None).unwrap(), requested);
    let expired = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("test clock supports an expired deadline"),
    );
    let receipt = budget.receipt().unwrap();
    let bytes = live(&memory);
    assert!(source.ranked_row_advice(cfg, expired).is_err());
    assert!(advice.layer(0, expired).is_err());
    assert!(advice.validate_session(&source, expired).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, receipt.admitted);
    assert_eq!(live(&memory), bytes);
    cancel.cancel();
    assert!(source.ranked_row_advice(cfg, None).is_err());
    assert!(advice.layer(0, None).is_err());
    assert!(advice.validate_session(&source, None).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, receipt.admitted);
    assert_eq!(live(&memory), bytes);
}

#[test]
fn ranked_advice_original_capacity_and_unrepresentable_features_refuse_without_leaks() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let source = session(fixture(3, -0.25), &budget, true);
    let cfg = request(1, 2, 3);
    let cap = budget.tighten_total_byte_limit(CAP / 2).unwrap();
    let original = budget.receipt().unwrap();
    let occupied = cap - original.live_persistent_bytes - original.live_scratch_bytes;
    let blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: occupied,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let before = budget.receipt().unwrap();
    let bytes = live(&memory);
    let error = source.ranked_row_advice(cfg, None).err().unwrap();
    assert!(error.to_string().contains("MemoryLimit"));
    // Existing coverage/metadata work may run before its first new allocation.
    // The advice request is not a separately renewable allowance.
    let refused = budget.receipt().unwrap();
    assert!(refused.work_reserved >= before.work_reserved);
    assert_eq!(refused.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(refused.live_scratch_bytes, before.live_scratch_bytes);
    assert_eq!(live(&memory), bytes);
    drop(blocker);
    source.ranked_row_advice(cfg, None).unwrap();
    source.checkpoint(None).unwrap();

    let (large_budget, large_memory, _) = resources();
    let mut large_spec = fixture(1, 0.0);
    large_spec.ops[0] = conv(1, 0, 1e100, 0.0);
    let large = session(large_spec, &large_budget, true);
    let before = large_budget.receipt().unwrap();
    let bytes = live(&large_memory);
    let error = large
        .ranked_row_advice(request(1, 2, 1), None)
        .err()
        .unwrap();
    assert!(error.to_string().contains("ranker advice cannot represent"));
    let refused = large_budget.receipt().unwrap();
    assert!(refused.work_reserved > before.work_reserved);
    assert_eq!(refused.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(refused.live_scratch_bytes, 0);
    assert_eq!(large_memory.receipt().unwrap().live_pending_bytes, 0);
    assert_eq!(live(&large_memory), bytes);
    large.checkpoint(None).unwrap();
}

#[test]
fn ranked_original_rows_keep_error_planes_separate_from_nominal_advice() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let mut spec = fixture(3, 1.0 / 7.0);
    let weight = f64::from_bits(1.0_f64.to_bits() + 1);
    if let TwinOpSpec::Conv {
        weight: weights,
        weight_rel_err,
        bias_err,
        ..
    } = &mut spec.ops[0]
    {
        weights[0] = weight;
        *weight_rel_err = 1e-8;
        bias_err[0] = 1e-9;
    } else {
        unreachable!();
    }
    let source = session(spec, &budget, true);
    let advice = source.ranked_row_advice(request(1, 2, 3), None).unwrap();
    let before = budget.receipt().unwrap();
    let bytes = live(&memory);
    {
        let lower = advice.original_lower_row(0, 0, None).unwrap();
        let upper = advice.original_upper_row(0, 0, None).unwrap();
        assert_eq!((lower.original_layer(), lower.original_target()), (0, 0));
        assert_eq!((upper.original_layer(), upper.original_target()), (0, 0));
        let nominal = advice.layer(0, None).unwrap();
        assert_ne!(
            lower.bias().to_bits(),
            f64::from(nominal.lower_advice()[3]).to_bits()
        );
        assert!(lower.bias_error() > 0.0);
        assert!(upper.bias_error() > 0.0);
        assert!(lower.entries().any(|(_, _, error)| error > 0.0));
        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let value = weight * x + 1.0 / 7.0;
            let evaluate = |row: &super::PreparedRankedInputRow<'_>, sign: f64| {
                row.bias()
                    + sign * row.bias_error()
                    + row
                        .entries()
                        .map(|(i, a, error)| {
                            let coordinate: f64 = if i == 0 { x } else { 0.0 };
                            a * coordinate + sign * error * coordinate.abs()
                        })
                        .sum::<f64>()
            };
            assert!(evaluate(&lower, -1.0) <= value);
            assert!(value <= evaluate(&upper, 1.0));
        }
    }
    let after = budget.receipt().unwrap();
    assert_eq!(after.admitted, before.admitted + 2);
    assert_eq!(
        after.work_reserved - before.work_reserved,
        2 * (1024 + 128 * (u64::from(usize::BITS) + 1))
    );
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(after.live_scratch_bytes, before.live_scratch_bytes);
    assert_eq!(live(&memory), bytes);
    drop(source);
    {
        let original = advice.original_lower_row(0, 0, None).unwrap();
        assert_eq!(original.original_target(), 0);
        assert!(live(&memory) > 0);
    }
    drop(advice);
    assert_eq!(live(&memory), 0);
}

#[test]
fn ranked_original_rows_refuse_unselected_ids_expiry_and_cancellation() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources();
    let source = session(fixture(3, -0.25), &budget, true);
    let advice = source.ranked_row_advice(request(1, 2, 3), None).unwrap();
    assert_eq!(advice.layer(0, None).unwrap().ids(), &[0]);
    let bytes = live(&memory);
    assert!(advice.original_lower_row(0, 1, None).is_err());
    assert!(advice.original_upper_row(0, 3, None).is_err());
    assert!(advice.original_lower_row(2, 0, None).is_err());
    let before = budget.receipt().unwrap();
    let expired = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("test clock supports an expired deadline"),
    );
    assert!(advice.original_lower_row(0, 0, expired).is_err());
    assert!(advice.original_upper_row(0, 0, expired).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    cancel.cancel();
    assert!(advice.original_lower_row(0, 0, None).is_err());
    assert!(advice.original_upper_row(0, 0, None).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    assert_eq!(live(&memory), bytes);
}

#[test]
fn ranked_original_rows_cannot_renew_the_shared_work_allowance() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let source = session(fixture(3, -0.25), &budget, true);
    let advice = source.ranked_row_advice(request(1, 2, 3), None).unwrap();
    let reserved = budget.receipt().unwrap().work_reserved;
    let mut consume = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                work: 100_000_000_000_000 - reserved,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    consume.complete().unwrap();
    let before = budget.receipt().unwrap();
    let bytes = live(&memory);
    let error = advice.original_lower_row(0, 0, None).err().unwrap();
    assert!(error.to_string().contains("WorkLimit"));
    assert!(advice.original_upper_row(0, 0, None).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    assert_eq!(
        budget.receipt().unwrap().work_reserved,
        before.work_reserved
    );
    assert_eq!(live(&memory), bytes);
}
