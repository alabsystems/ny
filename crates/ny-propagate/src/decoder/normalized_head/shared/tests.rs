// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use std::time::Duration;
fn setup(v: usize, h: usize) -> (SharedAffineHead, VocabOptions) {
    let ledger =
        WorkBudget::new(None, 32 << 20, 64 << 20, 1 << 34, 10_000, Duration::ZERO).unwrap();
    let limits = VocabOptions {
        budget: Some(ledger),
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
#[test]
fn decoder_n12_owner_shared_and_raw_original_replay_have_identical_scope_and_math() {
    let (owner, limits) = setup(257, 3);
    let gamma = [1.0, -2.0, 0.0];
    let norm = FinalNormalization {
        kind: FinalNormalizationKind::LayerNorm,
        gamma: &gamma,
        beta: Some(&[0.125, 1.0, -0.75]),
        epsilon: 1e-5,
    };
    let pre = HiddenBox {
        lo: &[-2.0, -0.25, 0.125],
        hi: &[1.0, 3.0, 0.75],
    };
    let budget = owner.budget();
    let raw = verify_normalized_next_token(
        owner.as_head(),
        norm,
        pre,
        0,
        NormalizedHeadOptions {
            budget: &budget,
            limits: &limits,
        },
    )
    .unwrap();
    let shared = verify_shared_normalized_next_token(&owner, norm, pre, 0, &limits).unwrap();
    assert_eq!(raw.evidence, shared.evidence);
    assert_eq!(raw.stats.reserved_work, shared.stats.reserved_work);
    assert_eq!(raw.stats.scratch_bytes, shared.stats.scratch_bytes);
    assert_eq!(
        raw.reserved_bytes() - shared.reserved_bytes(),
        8 * (owner.as_head().weights.len() + owner.as_head().bias.len())
    );
    assert_eq!(shared.shared_head_bytes(), owner.reserved_bytes());
    let replay = shared
        .replay(owner.as_head(), norm, pre, 0, &limits)
        .unwrap();
    assert_eq!(replay.evidence, shared.evidence);
    assert_eq!(replay.provenance(), NextTokenProvenance::OriginalReplay);
    let NormalizedNextTokenOutcome::Certified(proof) = &shared.outcome else {
        panic!("safe")
    };
    assert_eq!(
        proof.scope().identity(),
        match &raw.outcome {
            NormalizedNextTokenOutcome::Certified(p) => p.scope().identity(),
            _ => panic!("safe"),
        }
    );
}
#[test]
fn decoder_n12_owner_certificate_keeps_original_reservation_until_final_clone_drop() {
    let (owner, limits) = setup(2, 1);
    let budget = owner.budget();
    let norm = FinalNormalization {
        kind: FinalNormalizationKind::RmsNorm,
        gamma: &[1.0],
        beta: None,
        epsilon: 1e-5,
    };
    let report = verify_shared_normalized_next_token(
        &owner,
        norm,
        HiddenBox {
            lo: &[0.0],
            hi: &[0.0],
        },
        0,
        &limits,
    )
    .unwrap();
    let NormalizedNextTokenOutcome::Certified(proof) = &report.outcome else {
        panic!("safe")
    };
    let retained = proof.clone();
    let bytes = retained.shared_head_bytes() + retained.reserved_bytes();
    drop(report);
    drop(owner);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, bytes);
    let clone = retained.clone();
    drop(retained);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, bytes);
    drop(clone);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
}
#[test]
fn decoder_n12_owner_exact_binding_and_original_ledger_are_required_before_admission() {
    let (owner, limits) = setup(2, 1);
    let budget = owner.budget();
    let norm = FinalNormalization {
        kind: FinalNormalizationKind::LayerNorm,
        gamma: &[1.0],
        beta: None,
        epsilon: 1e-5,
    };
    let pre = HiddenBox {
        lo: &[-1.0],
        hi: &[1.0],
    };
    let copied = owner.as_head().weights.to_vec();
    let fake = AffineHead {
        weights: &copied,
        ..owner.as_head()
    };
    let before = budget.receipt().unwrap();
    assert!(shape_with_owner(
        fake,
        norm,
        pre,
        0,
        NormalizedHeadOptions {
            budget: &budget,
            limits: &limits
        },
        Some(&owner)
    )
    .is_err());
    let other = WorkBudget::new(None, 32 << 20, 64 << 20, 1 << 34, 10_000, Duration::ZERO).unwrap();
    assert!(shape_with_owner(
        owner.as_head(),
        norm,
        pre,
        0,
        NormalizedHeadOptions {
            budget: &other,
            limits: &limits
        },
        Some(&owner)
    )
    .is_err());
    let mut substitute = limits;
    substitute.budget = Some(other);
    assert!(verify_shared_normalized_next_token(&owner, norm, pre, 0, &substitute).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
}
#[test]
fn decoder_n12_owner_sharing_admits_near_cap_without_discounting_work_or_rows() {
    let (owner, mut limits) = setup(128, 4);
    let budget = owner.budget();
    let norm = FinalNormalization {
        kind: FinalNormalizationKind::LayerNorm,
        gamma: &[1.0; 4],
        beta: None,
        epsilon: 1e-5,
    };
    let pre = HiddenBox {
        lo: &[-1.0; 4],
        hi: &[1.0; 4],
    };
    let request = shape_with_owner(
        owner.as_head(),
        norm,
        pre,
        0,
        NormalizedHeadOptions {
            budget: &budget,
            limits: &limits,
        },
        Some(&owner),
    )
    .unwrap();
    limits.max_persistent_bytes = (owner.reserved_bytes() + request.persistent_bytes) as u64;
    assert!(verify_normalized_next_token(
        owner.as_head(),
        norm,
        pre,
        0,
        NormalizedHeadOptions {
            budget: &budget,
            limits: &limits
        }
    )
    .is_err());
    let report = verify_shared_normalized_next_token(&owner, norm, pre, 0, &limits).unwrap();
    assert_eq!(
        report.stats.reserved_work,
        normalized_work_bound(128, 4).unwrap()
    );
    assert_eq!(report.evidence.rows.len(), 127);
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        limits.max_persistent_bytes as usize
    );
}
