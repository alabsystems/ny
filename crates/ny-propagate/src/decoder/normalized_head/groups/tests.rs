// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;
use std::time::Duration;

const TEST_WORK: u64 = 1 << 34;
fn setup(weights: Vec<f64>, bias: Vec<f64>, hidden: usize) -> (SharedAffineHead, VocabOptions) {
    let ledger =
        WorkBudget::new(None, 8 << 20, 32 << 20, TEST_WORK, 100_000, Duration::ZERO).unwrap();
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
fn normalized(kind: FinalNormalizationKind, gamma: &[f64]) -> FinalNormalization<'_> {
    FinalNormalization {
        kind,
        gamma,
        beta: None,
        epsilon: 1e-5,
    }
}
fn query(
    owner: &SharedAffineHead,
    norm: FinalNormalization<'_>,
    pre: HiddenBox<'_>,
    target: usize,
    width: usize,
    limits: &VocabOptions,
) -> Result<NormalizedGroupedHeadReport> {
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
}
fn certified(report: &NormalizedGroupedHeadReport) -> &CertifiedGroupedNormalizedNextToken {
    match &report.outcome {
        NormalizedGroupedNextTokenOutcome::Certified(proof) => proof,
        NormalizedGroupedNextTokenOutcome::Unknown => {
            panic!("checked positive control must certify")
        }
    }
}
fn q(value: f64) -> BigRational {
    BigRational::from_float(value).unwrap()
}
fn proposed_upper(evidence: &NormalizedGroupedHeadEvidence, competitor: usize) -> f64 {
    for entry in &evidence.entries {
        match entry {
            NormalizedTerminalEvidence::OriginalGroup { start, end, upper }
                if (*start..*end).contains(&competitor) =>
            {
                return *upper;
            }
            NormalizedTerminalEvidence::OriginalRow {
                competitor: id,
                upper,
                ..
            } if *id == competitor => return *upper,
            _ => {}
        }
    }
    panic!("missing original competitor {competitor}");
}
/// Independent rational original algebra; squared signed comparisons eliminate
/// sqrt without sharing the interval kernel or its stored group geometry.
fn original_containment(
    evidence: &NormalizedGroupedHeadEvidence,
    head: AffineHead<'_>,
    norm: FinalNormalization<'_>,
    point: &[f64],
) {
    let n = BigRational::from_integer(BigInt::from(head.hidden_size));
    let mut z: Vec<_> = point.iter().copied().map(q).collect();
    if norm.kind == FinalNormalizationKind::LayerNorm {
        let mean = z.iter().cloned().sum::<BigRational>() / &n;
        for value in &mut z {
            *value -= &mean;
        }
    }
    let denominator2 = q(norm.epsilon) + z.iter().map(|v| v * v / &n).sum::<BigRational>();
    assert!(denominator2 > BigRational::zero());
    for competitor in 0..head.vocab_size {
        if competitor == evidence.target {
            continue;
        }
        let mut numerator = BigRational::zero();
        let mut constant = q(head.bias[competitor]) - q(head.bias[evidence.target]);
        for i in 0..head.hidden_size {
            let delta = q(head.row(competitor)[i]) - q(head.row(evidence.target)[i]);
            numerator += &delta * q(norm.gamma[i]) * &z[i];
            if let Some(beta) = norm.beta {
                constant += delta * q(beta[i]);
            }
        }
        let allowance = q(proposed_upper(evidence, competitor)) - constant;
        if numerator >= BigRational::zero() {
            assert!(allowance >= BigRational::zero());
            assert!(&allowance * &allowance * &denominator2 >= &numerator * &numerator);
        } else if allowance < BigRational::zero() {
            assert!(&allowance * &allowance * &denominator2 <= &numerator * &numerator);
        }
    }
}
fn nonzero_fixture(vocab: usize) -> (SharedAffineHead, VocabOptions) {
    let weights = (0..vocab)
        .flat_map(|i| {
            [
                (i % 7) as f64 * 0.0625 + 2f64.powi(-40),
                -((i % 5) as f64) * 0.03125,
                (i % 3) as f64 * 0.125,
            ]
        })
        .collect();
    let mut bias = vec![-10.0; vocab];
    bias[0] = 0.0;
    setup(weights, bias, 3)
}

#[test]
fn decoder_n12_groups_original_signed_zero_gamma_beta_and_nondyadic_mean_are_rational() {
    let (owner, limits) = nonzero_fixture(17);
    let gamma = [1.0, -0.5, 0.0];
    let beta = [0.375, -0.125, 0.25];
    let pre = HiddenBox {
        lo: &[-0.25, 0.125, -0.5],
        hi: &[-0.125, 0.25, -0.375],
    };
    for kind in [
        FinalNormalizationKind::LayerNorm,
        FinalNormalizationKind::RmsNorm,
    ] {
        let norm = FinalNormalization {
            beta: Some(&beta),
            ..normalized(kind, &gamma)
        };
        let report = query(&owner, norm, pre, 0, 8, &limits).unwrap();
        let proof = certified(&report);
        assert_eq!(proof.covered_competitors(), 16);
        assert!(report.stats.terminal_groups > 0 && report.stats.grouped_competitors == 16);
        for mask in 0..8 {
            let point: Vec<_> = (0..3)
                .map(|i| {
                    if mask & (1 << i) == 0 {
                        pre.lo[i]
                    } else {
                        pre.hi[i]
                    }
                })
                .collect();
            original_containment(&report.evidence, owner.as_head(), norm, &point);
        }
        let replay = report.replay(norm, pre, 0, &limits).unwrap();
        assert_eq!(replay.evidence, report.evidence);
        assert_eq!(replay.provenance(), NextTokenProvenance::OriginalReplay);
        assert_eq!(replay.stats.proposal_coverage_visits, 16);
    }
}

#[test]
fn decoder_n12_groups_nonzero_complete_groups_reduce_completed_visits_without_skipping_hashes() {
    let (owner, limits) = nonzero_fixture(257);
    let norm = normalized(FinalNormalizationKind::LayerNorm, &[1.0, -0.5, 0.0]);
    let pre = HiddenBox {
        lo: &[-0.25, 0.125, -0.5],
        hi: &[-0.125, 0.25, -0.375],
    };
    let direct =
        shared::verify_shared_normalized_next_token(&owner, norm, pre, 0, &limits).unwrap();
    let grouped = query(&owner, norm, pre, 0, 32, &limits).unwrap();
    certified(&grouped);
    assert_eq!(direct.stats.dense_screened_rows, 256);
    assert_eq!(grouped.stats.grouped_competitors, 256);
    assert_eq!(grouped.stats.direct_rows, 0);
    assert_eq!(grouped.stats.unobserved_failed_group_attempts, 0);
    assert_eq!(
        grouped.stats.original.original_head_visits,
        direct.stats.original_head_visits
    );
    assert_eq!(
        grouped.stats.original.normalization_parameter_visits,
        direct.stats.normalization_parameter_visits
    );
    let completed = grouped.stats.original.kernel_coordinate_visits
        + grouped.stats.completed_group_coordinate_visits;
    assert!(completed < direct.stats.kernel_coordinate_visits);
    assert_eq!(grouped.stats.original.coverage_visits, 256);
    assert!(grouped.stats.reserved_work > direct.stats.reserved_work);
    assert_eq!(grouped.evidence.scope_digest, direct.evidence.scope_digest);
}

#[test]
fn decoder_n12_groups_loose_bound_uses_paid_direct_rows_and_old_tight_taylor_replay() {
    let (owner, limits) = setup(vec![0.0, 1.0], vec![0.0, -1.1], 1);
    let norm = normalized(FinalNormalizationKind::RmsNorm, &[1.0]);
    let pre = HiddenBox {
        lo: &[-1.0],
        hi: &[1.0],
    };
    let report = query(&owner, norm, pre, 0, 8, &limits).unwrap();
    certified(&report);
    assert_eq!(report.stats.groups_completed, 1);
    assert_eq!(report.stats.terminal_groups, 0);
    assert_eq!(report.stats.direct_rows, 1);
    // The original RMS support is bounded by one, so the paid direct row
    // closes at its Cauchy screen when the dense screen cannot close. Fallback
    // must retain that original route and its complete row evidence.
    assert_eq!(report.stats.original.cauchy_screened_rows, 1);
    assert_eq!(report.stats.original.dense_screened_rows, 0);
    assert_eq!(report.stats.original.radial_taylor_attempted_rows, 0);
    assert_eq!(report.stats.original.coverage_visits, 1);
    for point in [-1.0, 0.0, 1.0] {
        original_containment(&report.evidence, owner.as_head(), norm, &[point]);
    }
    let fallback_replay = report.replay(norm, pre, 0, &limits).unwrap();
    assert_eq!(fallback_replay.evidence, report.evidence);
    assert_eq!(fallback_replay.stats.direct_rows, 1);
    assert_eq!(fallback_replay.stats.original.cauchy_screened_rows, 1);

    let (owner, limits) = setup(vec![0.0, 1.0], vec![0.0, -0.999997], 1);
    let norm = normalized(FinalNormalizationKind::RmsNorm, &[1.0]);
    let pre = HiddenBox {
        lo: &[0.9999],
        hi: &[1.0001],
    };
    let direct =
        shared::verify_shared_normalized_next_token(&owner, norm, pre, 0, &limits).unwrap();
    assert_eq!(direct.stats.radial_taylor_rows, 1);
    let proposal = NormalizedGroupedHeadEvidence {
        schema_version: GROUP_SCHEMA,
        scope_digest: direct.evidence.scope_digest,
        target: 0,
        vocab_size: 2,
        hidden_size: 1,
        normalization: norm.kind,
        group_width: 8,
        entries: direct
            .evidence
            .rows
            .iter()
            .map(|row| NormalizedTerminalEvidence::OriginalRow {
                competitor: row.competitor,
                upper: row.upper,
                method: row.method,
            })
            .collect(),
    };
    let ledger = owner.budget();
    let replay = replay_shared_grouped_normalized_next_token(
        &owner,
        norm,
        pre,
        0,
        &proposal,
        NormalizedHeadOptions {
            budget: &ledger,
            limits: &limits,
        },
    )
    .unwrap();
    certified(&replay);
    assert_eq!(replay.evidence, proposal);
    assert_eq!(replay.stats.groups_attempted, 0);
    assert_eq!(replay.stats.original.radial_taylor_attempted_rows, 1);
    assert_eq!(
        direct
            .replay(owner.as_head(), norm, pre, 0, &limits)
            .unwrap()
            .evidence,
        direct.evidence
    );
}

#[test]
fn decoder_n12_groups_target_at_every_boundary_is_excluded_exactly_once() {
    for target in [0, 1, 3, 4, 8] {
        let mut bias = vec![-2.0; 9];
        bias[target] = 0.0;
        let (owner, limits) = setup(vec![0.0; 27], bias, 3);
        let norm = normalized(FinalNormalizationKind::LayerNorm, &[1.0, -1.0, 0.0]);
        let pre = HiddenBox {
            lo: &[-1.0; 3],
            hi: &[1.0; 3],
        };
        let report = query(&owner, norm, pre, target, 4, &limits).unwrap();
        let proof = certified(&report);
        assert_eq!(proof.target(), target);
        assert_eq!(proof.covered_competitors(), 8);
        assert_eq!(report.stats.original.coverage_visits, 8);
        assert_eq!(report.stats.grouped_competitors, 8);
        assert_eq!((report.unresolved).len(), 0);
        certified(&report.replay(norm, pre, target, &limits).unwrap());
    }
}

#[test]
fn decoder_n12_groups_tie_and_last_unsafe_original_cannot_be_certified() {
    for last in [0.0, 0.25] {
        let mut bias = vec![-2.0; 257];
        bias[0] = 0.0;
        bias[256] = last;
        let (owner, limits) = setup(vec![0.0; 771], bias, 3);
        let norm = normalized(FinalNormalizationKind::LayerNorm, &[1.0, 1.0, 1.0]);
        let pre = HiddenBox {
            lo: &[-1.0; 3],
            hi: &[1.0; 3],
        };
        let report = query(&owner, norm, pre, 0, 32, &limits).unwrap();
        assert!(matches!(
            report.outcome,
            NormalizedGroupedNextTokenOutcome::Unknown
        ));
        assert_eq!(report.unresolved, vec![256]);
        assert_eq!(report.stats.original.coverage_visits, 256);
        let replay = report.replay(norm, pre, 0, &limits).unwrap();
        assert!(matches!(
            replay.outcome,
            NormalizedGroupedNextTokenOutcome::Unknown
        ));
    }
}

#[test]
fn decoder_n12_groups_malformed_bounds_ranges_coverage_and_original_scope_refuse() {
    let (owner, limits) = nonzero_fixture(17);
    let norm = normalized(FinalNormalizationKind::LayerNorm, &[1.0, -0.5, 0.0]);
    let pre = HiddenBox {
        lo: &[-0.25, 0.125, -0.5],
        hi: &[-0.125, 0.25, -0.375],
    };
    let report = query(&owner, norm, pre, 0, 8, &limits).unwrap();
    let ledger = owner.budget();
    for mutation in 0..9 {
        let mut evidence = report.evidence.clone();
        match mutation {
            0 => evidence.schema_version = 1,
            1 => evidence.scope_digest[0] ^= 1,
            2 => evidence.target = 1,
            3 => evidence.entries.pop().map(|_| ()).unwrap(),
            4 => evidence.entries.swap(0, 1),
            5 => {
                if let NormalizedTerminalEvidence::OriginalGroup { start, .. } =
                    &mut evidence.entries[0]
                {
                    *start += 1;
                }
            }
            6 => {
                if let NormalizedTerminalEvidence::OriginalGroup { end, .. } =
                    &mut evidence.entries[0]
                {
                    *end -= 1;
                }
            }
            7 => {
                if let NormalizedTerminalEvidence::OriginalGroup { upper, .. } =
                    &mut evidence.entries[0]
                {
                    *upper = f64::NAN;
                }
            }
            _ => {
                if let NormalizedTerminalEvidence::OriginalGroup { upper, .. } =
                    &mut evidence.entries[0]
                {
                    *upper = -f64::MAX;
                }
            }
        }
        assert!(
            replay_shared_grouped_normalized_next_token(
                &owner,
                norm,
                pre,
                0,
                &evidence,
                NormalizedHeadOptions {
                    budget: &ledger,
                    limits: &limits
                }
            )
            .is_err(),
            "mutation{mutation}"
        );
    }
    for changed in [
        FinalNormalization {
            epsilon: 2e-5,
            ..norm
        },
        normalized(norm.kind, &[1.0, -0.25, 0.0]),
    ] {
        assert!(report.replay(changed, pre, 0, &limits).is_err());
    }
    assert!(report
        .replay(
            norm,
            HiddenBox {
                lo: &[-0.5, 0.125, -0.5],
                hi: pre.hi
            },
            0,
            &limits
        )
        .is_err());
    let original = owner.as_head();
    let mut weights = original.weights.to_vec();
    weights[0] = f64::from_bits(weights[0].to_bits() + 1);
    let changed_owner = SharedAffineHead::new(
        original.vocab_size,
        original.hidden_size,
        weights,
        original.bias.to_vec(),
        HeadArithmetic::ExactReal,
        &limits,
    )
    .unwrap();
    assert!(replay_shared_grouped_normalized_next_token(
        &changed_owner,
        norm,
        pre,
        0,
        &report.evidence,
        NormalizedHeadOptions {
            budget: &ledger,
            limits: &limits,
        },
    )
    .is_err());
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Keep original certified evidence intact while mutating an independent proposal."
)]
fn decoder_n12_groups_positive_proposal_and_method_tags_cannot_replace_checked_stages() {
    let (owner, limits) = setup(vec![0.0, 1.0], vec![0.0, -1.1], 1);
    let norm = normalized(FinalNormalizationKind::RmsNorm, &[1.0]);
    let pre = HiddenBox {
        lo: &[-1.0],
        hi: &[1.0],
    };
    let report = query(&owner, norm, pre, 0, 8, &limits).unwrap();
    let mut raw = report.evidence.clone();
    if let NormalizedTerminalEvidence::OriginalRow { method, .. } = &mut raw.entries[0] {
        *method = NormalizedSupportMethod::RadialTaylor;
    }
    let ledger = owner.budget();
    let replay = replay_shared_grouped_normalized_next_token(
        &owner,
        norm,
        pre,
        0,
        &raw,
        NormalizedHeadOptions {
            budget: &ledger,
            limits: &limits,
        },
    )
    .unwrap();
    certified(&replay);
    assert_eq!(replay.evidence, raw);
    assert_eq!(replay.stats.original.radial_taylor_attempted_rows, 0);

    let (owner, limits) = setup(vec![0.0, 0.0], vec![0.0, -1.0], 1);
    let norm = normalized(FinalNormalizationKind::LayerNorm, &[1.0]);
    let pre = HiddenBox {
        lo: &[-1.0],
        hi: &[1.0],
    };
    let report = query(&owner, norm, pre, 0, 8, &limits).unwrap();
    let mut raw = report.evidence.clone();
    if let NormalizedTerminalEvidence::OriginalGroup { upper, .. } = &mut raw.entries[0] {
        *upper = 0.0;
    }
    let ledger = owner.budget();
    let replay = replay_shared_grouped_normalized_next_token(
        &owner,
        norm,
        pre,
        0,
        &raw,
        NormalizedHeadOptions {
            budget: &ledger,
            limits: &limits,
        },
    )
    .unwrap();
    assert!(matches!(
        replay.outcome,
        NormalizedGroupedNextTokenOutcome::Unknown
    ));
    assert_eq!(replay.unresolved, vec![1]);
}

#[test]
fn decoder_n12_groups_same_owner_ledger_and_finite_caps_precede_admission() {
    let (owner, limits) = nonzero_fixture(17);
    let norm = normalized(FinalNormalizationKind::LayerNorm, &[1.0, -0.5, 0.0]);
    let pre = HiddenBox {
        lo: &[-0.25, 0.125, -0.5],
        hi: &[-0.125, 0.25, -0.375],
    };
    let ledger = owner.budget();
    let before = ledger.receipt().unwrap();
    let other =
        WorkBudget::new(None, 8 << 20, 32 << 20, TEST_WORK, 100_000, Duration::ZERO).unwrap();
    assert!(verify_shared_grouped_normalized_next_token(
        &owner,
        norm,
        pre,
        0,
        NormalizedGroupPolicy { group_width: 8 },
        NormalizedHeadOptions {
            budget: &other,
            limits: &limits
        }
    )
    .is_err());
    assert_eq!(other.receipt().unwrap().admitted, 0);
    for which in 0..5 {
        let mut tight = limits.clone();
        match which {
            0 => tight.max_evidence_bytes = 1,
            1 => tight.max_terms = 1,
            2 => tight.max_scratch_bytes = 1,
            3 => tight.max_persistent_bytes = 1,
            _ => tight.deadline = Some(Instant::now()),
        }
        assert!(query(&owner, norm, pre, 0, 8, &tight).is_err());
    }
    assert!(query(&owner, norm, pre, 0, 0, &limits).is_err());
    assert_eq!(ledger.receipt().unwrap().admitted, before.admitted);
}

#[test]
fn decoder_n12_groups_local_group_prediction_denial_preserves_the_prepaid_direct_route() {
    let (owner, limits) = nonzero_fixture(17);
    let norm = normalized(FinalNormalizationKind::LayerNorm, &[1.0, -0.5, 0.0]);
    let pre = HiddenBox {
        lo: &[-0.25, 0.125, -0.5],
        hi: &[-0.125, 0.25, -0.375],
    };
    let mut tight = limits.clone();
    tight.max_terms = normalized_work_bound(17, 3).unwrap();
    let report = query(&owner, norm, pre, 0, 8, &tight).unwrap();
    certified(&report);
    assert!(!report.stats.group_profile_enabled);
    assert_eq!(report.stats.groups_attempted, 0);
    assert_eq!(report.stats.direct_rows, 16);
    let grouped = query(&owner, norm, pre, 0, 8, &limits).unwrap();
    assert!(grouped.replay(norm, pre, 0, &tight).is_err());
    certified(&report.replay(norm, pre, 0, &tight).unwrap());
}

#[test]
fn decoder_n12_groups_global_work_denial_does_not_mint_a_fallback_allowance() {
    let (owner, limits) = nonzero_fixture(17);
    let norm = normalized(FinalNormalizationKind::LayerNorm, &[1.0, -0.5, 0.0]);
    let pre = HiddenBox {
        lo: &[-0.25, 0.125, -0.5],
        hi: &[-0.125, 0.25, -0.375],
    };
    let ledger = owner.budget();
    let remaining =
        TEST_WORK - ledger.receipt().unwrap().work_reserved - normalized_work_bound(17, 3).unwrap();
    let spent = ledger
        .lease_for(
            LeaseClass::Replay,
            WorkRequest {
                work: remaining,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let report = query(&owner, norm, pre, 0, 8, &limits).unwrap();
    certified(&report);
    assert!(report.stats.group_profile_enabled);
    assert!(report.stats.groups_admission_refused > 0);
    assert_eq!(report.stats.groups_admitted, 0);
    assert_eq!(report.stats.direct_rows, 16);
    assert_eq!(ledger.receipt().unwrap().work_reserved, TEST_WORK);
    drop(spent);
    assert_eq!(ledger.receipt().unwrap().work_reserved, TEST_WORK);
    assert!(report.replay(norm, pre, 0, &limits).is_err());
}

#[test]
fn decoder_n12_groups_original_model_and_query_claims_survive_until_last_certificate_clone() {
    let (owner, limits) = nonzero_fixture(17);
    let ledger = owner.budget();
    let norm = normalized(FinalNormalizationKind::LayerNorm, &[1.0, -0.5, 0.0]);
    let pre = HiddenBox {
        lo: &[-0.25, 0.125, -0.5],
        hi: &[-0.125, 0.25, -0.375],
    };
    let mut report = query(&owner, norm, pre, 0, 8, &limits).unwrap();
    let proof = certified(&report).clone();
    let checked = proof.checked_stats().clone();
    report.stats.terminal_groups = usize::MAX;
    report.evidence.entries.clear();
    assert_eq!(proof.checked_stats(), &checked);
    assert_eq!(proof.covered_competitors(), 16);
    let bytes = proof.reserved_bytes() + proof.shared_head_bytes();
    drop(report);
    drop(owner);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, bytes);
    let last = proof.clone();
    drop(proof);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, bytes);
    drop(last);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn decoder_n12_groups_mints_have_distinct_scope_owners_and_clones_share_checked_state() {
    let (owner, limits) = nonzero_fixture(17);
    let norm = normalized(FinalNormalizationKind::LayerNorm, &[1.0, -0.5, 0.0]);
    let pre = HiddenBox {
        lo: &[-0.25, 0.125, -0.5],
        hi: &[-0.125, 0.25, -0.375],
    };
    let first = query(&owner, norm, pre, 0, 8, &limits).unwrap();
    let second = query(&owner, norm, pre, 0, 8, &limits).unwrap();
    let replay = first.replay(norm, pre, 0, &limits).unwrap();
    let proof = certified(&first);
    let alias = proof.clone();
    assert_eq!(size_of_val(proof), size_of::<Arc<()>>());
    assert!(Arc::ptr_eq(&proof.inner, &alias.inner));
    assert!(std::ptr::eq(proof.scope(), alias.scope()));
    assert!(std::ptr::eq(proof.checked_stats(), alias.checked_stats()));
    for other in [certified(&second), certified(&replay)] {
        assert_eq!(proof.scope(), other.scope());
        assert!(!Arc::ptr_eq(&proof.inner, &other.inner));
        assert!(!std::ptr::eq(proof.scope(), other.scope()));
    }
    assert_eq!(proof.provenance(), NextTokenProvenance::CheckedQuery);
    assert_eq!(
        certified(&replay).provenance(),
        NextTokenProvenance::OriginalReplay
    );
    assert!(
        size_of::<NormalizedGroupedHeadReport>()
            + size_of_val(proof.inner.as_ref())
            + 2 * size_of::<usize>()
            <= 4096
    );
}

#[test]
fn decoder_n12_groups_subnormal_originals_nonfinite_normalization_and_arithmetic_profiles() {
    let tiny = f64::from_bits(1);
    let (owner, limits) = setup(vec![0.0, tiny, -tiny], vec![0.0, -1.0, -1.0], 1);
    let pre = HiddenBox {
        lo: &[tiny],
        hi: &[tiny],
    };
    for kind in [
        FinalNormalizationKind::LayerNorm,
        FinalNormalizationKind::RmsNorm,
    ] {
        let norm = normalized(kind, &[-1.0]);
        let report = query(&owner, norm, pre, 0, 8, &limits).unwrap();
        certified(&report);
        original_containment(&report.evidence, owner.as_head(), norm, pre.lo);
        certified(&report.replay(norm, pre, 0, &limits).unwrap());
    }
    for poison in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let gamma = [poison];
        let norm = normalized(FinalNormalizationKind::LayerNorm, &gamma);
        assert!(query(&owner, norm, pre, 0, 8, &limits).is_err());
        let norm = FinalNormalization {
            beta: Some(&gamma),
            ..normalized(FinalNormalizationKind::LayerNorm, &[0.0])
        };
        assert!(query(&owner, norm, pre, 0, 8, &limits).is_err());
    }
    let ledger = owner.budget();
    let rounded = SharedAffineHead::new(
        3,
        1,
        vec![0.0, tiny, -tiny],
        vec![0.0, -1.0, -1.0],
        HeadArithmetic::SequentialBinary64,
        &limits,
    )
    .unwrap();
    let before = ledger.receipt().unwrap().admitted;
    assert!(matches!(
        query(
            &rounded,
            normalized(FinalNormalizationKind::LayerNorm, &[0.0]),
            pre,
            0,
            8,
            &limits
        ),
        Err(NormalizedHeadRefusal::UnsupportedHeadArithmetic(
            HeadArithmetic::SequentialBinary64
        ))
    ));
    assert_eq!(ledger.receipt().unwrap().admitted, before);
}

#[test]
fn decoder_n12_groups_cancelled_complete_cover_cannot_publish() {
    let authority = MemoryAuthority::new(8 << 20);
    let cancel = CancellationToken::default();
    let ledger = WorkBudget::new_in(
        &authority,
        &cancel,
        WorkLimits {
            deadline: None,
            scratch_bytes: 1 << 20,
            persistent_bytes: 4 << 20,
            max_work: TEST_WORK,
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
    let mut bias = vec![-1.0; 258];
    bias[0] = 0.0;
    let owner = SharedAffineHead::new(
        258,
        1,
        vec![0.0; 258],
        bias,
        HeadArithmetic::ExactReal,
        &limits,
    )
    .unwrap();
    let norm = normalized(FinalNormalizationKind::LayerNorm, &[1.0]);
    let pre = HiddenBox {
        lo: &[-1.0],
        hi: &[1.0],
    };
    let report = query(&owner, norm, pre, 0, 512, &limits).unwrap();
    let lease = ledger
        .lease_for(
            LeaseClass::Replay,
            WorkRequest {
                work: 4096,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let mut visited = 0;
    let result = coverage(&report.evidence, &lease, &limits, |_, _| {
        visited += 1;
        if visited == 3 {
            cancel.cancel();
        }
        Ok(())
    });
    assert!(result.is_err());
    assert!(visited <= 256);
    assert!(report.replay(norm, pre, 0, &limits).is_err());
}

#[test]
fn decoder_n12_groups_member_rescue_closes_and_independently_replays_original_pairs() {
    let (owner, limits) = setup(vec![0.0, 0.0, 2.0], vec![0.0, -0.125, -2.0], 1);
    let norm = FinalNormalization {
        epsilon: 1.0,
        ..normalized(FinalNormalizationKind::RmsNorm, &[1.0])
    };
    let pre = HiddenBox {
        lo: &[-0.25],
        hi: &[0.25],
    };
    let report = query(&owner, norm, pre, 0, 8, &limits).unwrap();
    certified(&report);
    assert_eq!(report.stats.groups_completed, 1);
    assert_eq!(report.stats.member_rescues_attempted, 1);
    assert_eq!(report.stats.member_rescues_admitted, 1);
    assert_eq!(report.stats.member_rescues_completed, 1);
    assert_eq!(report.stats.member_rescues_improved, 1);
    assert_eq!(report.stats.member_rescues_closed_groups, 1);
    assert_eq!(report.stats.member_rescue_original_visits, 6);
    assert_eq!(report.stats.terminal_groups, 1);
    assert_eq!(report.stats.grouped_competitors, 2);
    assert_eq!(report.stats.direct_rows, 0);
    assert!(proposed_upper(&report.evidence, 1) < 0.0);
    for point in [-0.25, -0.125, 0.0, 0.125, 0.25] {
        original_containment(&report.evidence, owner.as_head(), norm, &[point]);
    }
    let ledger = owner.budget();
    let before = ledger.receipt().unwrap().work_reserved;
    let replay = report.replay(norm, pre, 0, &limits).unwrap();
    certified(&replay);
    assert_eq!(replay.evidence, report.evidence);
    assert_eq!(replay.stats.member_rescues_completed, 1);
    assert_eq!(replay.stats.member_rescues_closed_groups, 1);
    assert_eq!(replay.provenance(), NextTokenProvenance::OriginalReplay);
    assert_eq!(
        ledger.receipt().unwrap().work_reserved - before,
        replay.stats.reserved_work
    );
    assert!(report.stats.reserved_work <= normalized_group_work_bound(3, 1, 8).unwrap());
    let mut forged = report.evidence.clone();
    if let NormalizedTerminalEvidence::OriginalGroup { upper, .. } = &mut forged.entries[0] {
        *upper = -f64::MAX;
    }
    assert!(replay_shared_grouped_normalized_next_token(
        &owner,
        norm,
        pre,
        0,
        &forged,
        NormalizedHeadOptions {
            budget: &ledger,
            limits: &limits
        }
    )
    .is_err());
    // A stronger original-group proposal cannot evade reconstruction by
    // disabling the new optional profile in the replay caller's local caps.
    let mut old_only = limits.clone();
    old_only.max_terms = normalized_work_bound(3, 1).unwrap() + extra_group_work(3, 1, 8).unwrap();
    let before = ledger.receipt().unwrap().work_reserved;
    assert!(matches!(
        report.replay(norm, pre, 0, &old_only),
        Err(NormalizedHeadRefusal::Invalid(DecoderRefusal::Resource(
            "grouped normalized replay member prediction"
        )))
    ));
    // Replay has checked the older group on the same original ledger. Its
    // spent admission cannot be refunded by refusing the unpriced rescue.
    assert!(ledger.receipt().unwrap().work_reserved > before);
    assert_eq!(
        report.replay(norm, pre, 0, &limits).unwrap().evidence,
        report.evidence
    );
}

#[test]
fn decoder_n12_groups_member_old_closing_groups_preserve_evidence_and_actual_charges() {
    let (owner, limits) = nonzero_fixture(17);
    let norm = normalized(FinalNormalizationKind::LayerNorm, &[1.0, -0.5, 0.0]);
    let pre = HiddenBox {
        lo: &[-0.25, 0.125, -0.5],
        hi: &[-0.125, 0.25, -0.375],
    };
    let ledger = owner.budget();
    let mut old_only = limits.clone();
    old_only.max_terms =
        normalized_work_bound(17, 3).unwrap() + extra_group_work(17, 3, 8).unwrap();
    let before = ledger.receipt().unwrap().work_reserved;
    let old = query(&owner, norm, pre, 0, 8, &old_only).unwrap();
    let old_work = ledger.receipt().unwrap().work_reserved - before;
    certified(&old);
    assert!(old.stats.group_profile_enabled);
    assert!(!old.stats.member_rescue_profile_enabled);
    assert_eq!(old.stats.direct_rows, 0);
    let before = ledger.receipt().unwrap().work_reserved;
    let current = query(&owner, norm, pre, 0, 8, &limits).unwrap();
    certified(&current);
    assert!(current.stats.member_rescue_profile_enabled);
    assert_eq!(current.evidence, old.evidence);
    assert_eq!(current.stats.groups_completed, old.stats.groups_completed);
    assert_eq!(current.stats.member_rescues_attempted, 0);
    assert_eq!(current.stats.member_rescues_admitted, 0);
    assert_eq!(current.stats.reserved_work, old.stats.reserved_work);
    assert_eq!(ledger.receipt().unwrap().work_reserved - before, old_work);
    assert_eq!(
        current.stats.completed_group_coordinate_visits,
        old.stats.completed_group_coordinate_visits
    );
    assert_eq!(
        current.stats.completed_group_consumed_work,
        old.stats.completed_group_consumed_work
    );
    assert_eq!(
        current.replay(norm, pre, 0, &old_only).unwrap().evidence,
        current.evidence
    );
}

#[test]
fn decoder_n12_groups_member_denied_rescue_retains_the_prepaid_direct_fallback_without_refill() {
    let (owner, limits) = setup(vec![0.0, 0.0, 2.0], vec![0.0, -0.125, -2.0], 1);
    let norm = FinalNormalization {
        epsilon: 1.0,
        ..normalized(FinalNormalizationKind::RmsNorm, &[1.0])
    };
    let pre = HiddenBox {
        lo: &[-0.25],
        hi: &[0.25],
    };
    let ledger = owner.budget();
    let options = NormalizedHeadOptions {
        budget: &ledger,
        limits: &limits,
    };
    let (baseline, _, old_enabled, _, member_enabled) =
        shape_grouped(&owner, norm, pre, 0, 8, options).unwrap();
    assert!(old_enabled && member_enabled);
    let original = kernel_request(owner.as_head(), norm, 0, 1, 3);
    let old_phase = group_support_request(&original).unwrap();
    let remaining =
        TEST_WORK - ledger.receipt().unwrap().work_reserved - baseline.work - old_phase.work;
    let spent = ledger
        .lease_for(
            LeaseClass::Replay,
            WorkRequest {
                work: remaining,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let report = query(&owner, norm, pre, 0, 8, &limits).unwrap();
    certified(&report);
    assert_eq!(report.stats.groups_completed, 1);
    assert_eq!(report.stats.member_rescues_attempted, 1);
    assert_eq!(report.stats.member_rescues_admission_refused, 1);
    assert_eq!(report.stats.member_rescues_admitted, 0);
    assert_eq!(report.stats.member_rescues_completed, 0);
    assert_eq!(report.stats.terminal_groups, 0);
    assert_eq!(report.stats.direct_rows, 2);
    assert_eq!(report.stats.reserved_work, baseline.work + old_phase.work);
    for point in [-0.25, 0.0, 0.25] {
        original_containment(&report.evidence, owner.as_head(), norm, &[point]);
    }
    assert_eq!(ledger.receipt().unwrap().work_reserved, TEST_WORK);
    drop(spent);
    assert_eq!(ledger.receipt().unwrap().work_reserved, TEST_WORK);
    assert!(report.replay(norm, pre, 0, &limits).is_err());
}
