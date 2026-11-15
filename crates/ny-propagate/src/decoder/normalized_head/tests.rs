// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;
use std::time::Duration;

fn budget() -> WorkBudget {
    WorkBudget::new(None, 32 << 20, 256 << 20, 1 << 34, 10_000, Duration::ZERO).unwrap()
}
fn options<'a>(ledger: &'a WorkBudget, limits: &'a VocabOptions) -> NormalizedHeadOptions<'a> {
    NormalizedHeadOptions {
        budget: ledger,
        limits,
    }
}
fn affine<'a>(dim: usize, weights: &'a [f64], bias: &'a [f64]) -> AffineHead<'a> {
    AffineHead {
        vocab_size: bias.len(),
        hidden_size: dim,
        weights,
        bias,
        arithmetic: HeadArithmetic::ExactReal,
    }
}
fn norm<'a>(
    kind: FinalNormalizationKind,
    gamma: &'a [f64],
    beta: Option<&'a [f64]>,
) -> FinalNormalization<'a> {
    FinalNormalization {
        kind,
        gamma,
        beta,
        epsilon: 1e-5,
    }
}
fn q(value: f64) -> BigRational {
    BigRational::from_float(value).unwrap()
}

/// Exact original rational arithmetic; no interval or libm sqrt implementation
/// is shared with the checker. Square comparison proves the signed inequality.
fn original_containment(
    upper: f64,
    head: AffineHead<'_>,
    norm: FinalNormalization<'_>,
    target: usize,
    competitor: usize,
    point: &[f64],
) {
    let n = BigRational::from_integer(BigInt::from(point.len()));
    let mut z: Vec<_> = point.iter().copied().map(q).collect();
    if norm.kind == FinalNormalizationKind::LayerNorm {
        let mean = z.iter().cloned().sum::<BigRational>() / &n;
        for value in &mut z {
            *value -= &mean;
        }
    }
    let mut numerator = BigRational::zero();
    let mut constant = q(head.bias[competitor]) - q(head.bias[target]);
    let mut denominator2 = q(norm.epsilon);
    for i in 0..point.len() {
        let difference = q(head.row(competitor)[i]) - q(head.row(target)[i]);
        numerator += &difference * q(norm.gamma[i]) * &z[i];
        if let Some(beta) = norm.beta {
            constant += difference * q(beta[i]);
        }
        denominator2 += &z[i] * &z[i] / &n;
    }
    assert!(denominator2 > BigRational::zero());
    let bound = q(upper) - constant;
    if numerator >= BigRational::zero() {
        assert!(
            bound >= BigRational::zero(),
            "negative bound for positive numerator"
        );
        assert!(
            &bound * &bound * &denominator2 >= &numerator * &numerator,
            "original positive normalized value exceeds upper {upper}"
        );
    } else if bound < BigRational::zero() {
        assert!(
            &bound * &bound * &denominator2 <= &numerator * &numerator,
            "original negative normalized value exceeds upper {upper}"
        );
    }
}

#[test]
fn decoder_n12_rational_original_f64_containment_asymmetric_signed_zero_gamma_and_projection() {
    let weights = [
        1.0 + 2f64.powi(-40),
        0.75,
        -0.6,
        1.0,
        -1000.0,
        -0.1,
        -2.0 - 2f64.powi(-42),
        0.01,
        3.0 + 2f64.powi(-43),
    ];
    let biases = [1.5, -3.0, -2.75];
    let head = affine(3, &weights, &biases);
    let gamma = [-1.5, 0.0, 2.0 + 2f64.powi(-41)];
    let beta = [-0.125, 0.25, 2f64.powi(-22)];
    let lo = [-0.375, 0.1875, -0.25];
    let hi = [0.3125, 0.4375, 0.125];
    let pre = HiddenBox { lo: &lo, hi: &hi };
    for kind in [
        FinalNormalizationKind::LayerNorm,
        FinalNormalizationKind::RmsNorm,
    ] {
        let ledger = budget();
        let limits = VocabOptions::default();
        let normal = norm(kind, &gamma, Some(&beta));
        let report =
            verify_normalized_next_token(head, normal, pre, 0, options(&ledger, &limits)).unwrap();
        assert_eq!(report.evidence.rows.len(), 2);
        for mask in 0..8 {
            let point: Vec<_> = (0..3)
                .map(|i| if mask & (1 << i) == 0 { lo[i] } else { hi[i] })
                .collect();
            for row in &report.evidence.rows {
                original_containment(row.upper, head, normal, 0, row.competitor, &point);
            }
        }
        let midpoint = [0.0, 0.25, -0.0625];
        for row in &report.evidence.rows {
            original_containment(row.upper, head, normal, 0, row.competitor, &midpoint);
        }
        let replay = report.replay(head, normal, pre, 0, &limits).unwrap();
        assert_eq!(replay.evidence, report.evidence);
        assert_eq!(replay.provenance(), NextTokenProvenance::OriginalReplay);
        assert!(report.stats.kernel_coordinate_visits > 0);
        assert_eq!(
            report.stats.original_head_visits,
            (weights.len() + biases.len()) as u64
        );
    }
}

#[test]
fn decoder_n12_layernorm_dead_direction_closes_a_margin_lost_by_separate_output_boxes() {
    let weights = [0.0, 0.0, 1.0, 1.0];
    let biases = [0.0, -0.25];
    let head = affine(2, &weights, &biases);
    let gamma = [1.0, 1.0];
    let pre = HiddenBox {
        lo: &[-100.0, -3.0],
        hi: &[20.0, 70.0],
    };
    let ledger = budget();
    let limits = VocabOptions::default();
    let normal = norm(FinalNormalizationKind::LayerNorm, &gamma, None);
    let composed =
        verify_normalized_next_token(head, normal, pre, 0, options(&ledger, &limits)).unwrap();
    let NormalizedNextTokenOutcome::Certified(proof) = &composed.outcome else {
        panic!("checked mean direction should cancel")
    };
    assert_eq!(proof.maximum_competitor_upper(), Some(-0.25));
    assert_eq!(composed.stats.projected_zero_rows, 1);
    let separate = super::super::verify_dense_next_token(
        head,
        HiddenBox {
            lo: &[-1.0, -1.0],
            hi: &[1.0, 1.0],
        },
        &[0.0, 0.0],
        0,
        &limits,
    )
    .unwrap();
    assert!(matches!(
        separate.outcome,
        super::super::NextTokenOutcome::Unknown
    ));
    let rms = verify_normalized_next_token(
        head,
        norm(FinalNormalizationKind::RmsNorm, &gamma, None),
        pre,
        0,
        options(&ledger, &limits),
    )
    .unwrap();
    assert!(matches!(rms.outcome, NormalizedNextTokenOutcome::Unknown));
    assert_eq!(rms.stats.projected_zero_rows, 0);
}

#[test]
fn decoder_n12_zero_gamma_preserves_original_beta_and_bias_constants() {
    let weights = [0.0, 0.0, 7.0, -13.0];
    let biases = [0.0, -4.0];
    let gamma = [0.0, -0.0];
    let beta = [0.25, -0.125];
    let head = affine(2, &weights, &biases);
    let pre = HiddenBox {
        lo: &[-8.0, 1.0],
        hi: &[2.0, 11.0],
    };
    let ledger = budget();
    let limits = VocabOptions::default();
    let normal = norm(FinalNormalizationKind::LayerNorm, &gamma, Some(&beta));
    let report =
        verify_normalized_next_token(head, normal, pre, 0, options(&ledger, &limits)).unwrap();
    assert!(matches!(
        report.outcome,
        NormalizedNextTokenOutcome::Certified(_)
    ));
    assert_eq!(report.stats.projected_zero_rows, 1);
    original_containment(
        report.evidence.rows[0].upper,
        head,
        normal,
        0,
        1,
        &[-8.0, 11.0],
    );
}

#[test]
fn decoder_n12_original_f64_low_bits_and_exact_ties_cannot_be_narrowed_into_strict_proofs() {
    let weights = [1.0, 1.0 + 2f64.powi(-40)];
    let biases = [0.0, -2f64.powi(-42)];
    let gamma = [1.0];
    let normal = norm(FinalNormalizationKind::RmsNorm, &gamma, None);
    let head = affine(1, &weights, &biases);
    let pre = HiddenBox {
        lo: &[1.0],
        hi: &[1.0],
    };
    let ledger = budget();
    let limits = VocabOptions::default();
    let report =
        verify_normalized_next_token(head, normal, pre, 0, options(&ledger, &limits)).unwrap();
    assert!(matches!(
        report.outcome,
        NormalizedNextTokenOutcome::Unknown
    ));
    assert_eq!(report.unresolved, [1]);
    original_containment(report.evidence.rows[0].upper, head, normal, 0, 1, &[1.0]);
    let equal = affine(1, &[1.0, 1.0], &[0.0, 0.0]);
    let tie =
        verify_normalized_next_token(equal, normal, pre, 0, options(&ledger, &limits)).unwrap();
    assert!(matches!(tie.outcome, NormalizedNextTokenOutcome::Unknown));
    assert_eq!(tie.evidence.rows[0].upper, 0.0);
}

#[test]
fn decoder_n12_scope_replay_invalidates_original_parameters_region_target_and_type() {
    let weights = [0.0, 0.0, 1.0, 1.0];
    let biases = [0.0, -0.25];
    let gamma = [1.0, 1.0];
    let beta = [0.0, 0.0];
    let head = affine(2, &weights, &biases);
    let normal = norm(FinalNormalizationKind::LayerNorm, &gamma, Some(&beta));
    let pre = HiddenBox {
        lo: &[-1.0, 0.0],
        hi: &[0.5, 2.0],
    };
    let ledger = budget();
    let limits = VocabOptions::default();
    let report =
        verify_normalized_next_token(head, normal, pre, 0, options(&ledger, &limits)).unwrap();
    let changed_gamma = [1.0, 1.0 + 2f64.powi(-40)];
    let changed_beta = [0.0, f64::from_bits(1)];
    for changed in [
        FinalNormalization {
            gamma: &changed_gamma,
            ..normal
        },
        FinalNormalization {
            beta: Some(&changed_beta),
            ..normal
        },
        FinalNormalization {
            epsilon: f64::from_bits(normal.epsilon.to_bits() + 1),
            ..normal
        },
        FinalNormalization {
            kind: FinalNormalizationKind::RmsNorm,
            ..normal
        },
    ] {
        assert!(report.replay(head, changed, pre, 0, &limits).is_err());
    }
    let changed_weights = [0.0, 0.0, 1.0 + 2f64.powi(-40), 1.0];
    assert!(report
        .replay(
            affine(2, &changed_weights, &biases),
            normal,
            pre,
            0,
            &limits
        )
        .is_err());
    assert!(report
        .replay(
            head,
            normal,
            HiddenBox {
                lo: &[-1.0, 0.0],
                hi: &[0.5, 2.0 + 2f64.powi(-40)]
            },
            0,
            &limits
        )
        .is_err());
    assert!(report.replay(head, normal, pre, 1, &limits).is_err());
}

#[test]
fn decoder_n12_original_complete_cohort_includes_the_last_unsafe_competitor() {
    let mut weights = vec![0.0; 257];
    weights[256] = 1.0;
    let mut biases = vec![-0.25; 257];
    biases[0] = 0.0;
    let gamma = [1.0];
    let normal = norm(FinalNormalizationKind::RmsNorm, &gamma, None);
    let pre = HiddenBox {
        lo: &[1.0],
        hi: &[1.0],
    };
    let ledger = budget();
    let limits = VocabOptions::default();
    let source = affine(1, &weights, &biases);
    let report =
        verify_normalized_next_token(source, normal, pre, 0, options(&ledger, &limits)).unwrap();
    assert_eq!(report.evidence.rows.len(), 256);
    assert_eq!(report.stats.coverage_visits, 256);
    assert!(matches!(
        report.outcome,
        NormalizedNextTokenOutcome::Unknown
    ));
    assert_eq!(report.unresolved, [256]);
    assert_eq!(report.stats.dense_screened_rows, 255);
    assert_eq!(report.stats.radial_taylor_attempted_rows, 1);
    assert_eq!(report.stats.original_head_visits, 514);
    let replay = report.replay(source, normal, pre, 0, &limits).unwrap();
    assert_eq!(replay.unresolved, [256]);
    biases[256] = -2.0;
    let safe = affine(1, &weights, &biases);
    let closed =
        verify_normalized_next_token(safe, normal, pre, 0, options(&ledger, &limits)).unwrap();
    let NormalizedNextTokenOutcome::Certified(proof) = &closed.outcome else {
        panic!("full safe cohort")
    };
    assert_eq!(proof.covered_competitors(), 256);
    assert_eq!(closed.stats.dense_screened_rows, 256);
    assert_eq!(closed.stats.radial_taylor_attempted_rows, 0);
    assert_eq!(proof.scope().region().lo(), &[1.0]);
}

#[test]
fn decoder_n12_poisoned_evidence_does_not_change_sealed_provenance_or_mint_replay() {
    let source = affine(1, &[0.0, 0.0, 0.0], &[0.0, -0.25, -0.5]);
    let gamma = [1.0];
    let normal = norm(FinalNormalizationKind::LayerNorm, &gamma, None);
    let pre = HiddenBox {
        lo: &[-1.0],
        hi: &[2.0],
    };
    let ledger = budget();
    let limits = VocabOptions::default();
    let mut report =
        verify_normalized_next_token(source, normal, pre, 0, options(&ledger, &limits)).unwrap();
    let NormalizedNextTokenOutcome::Certified(proof) = &report.outcome else {
        panic!("constant normalized margin")
    };
    let sealed = proof.clone();
    let replay = report.replay(source, normal, pre, 0, &limits).unwrap();
    let NormalizedNextTokenOutcome::Certified(replayed) = &replay.outcome else {
        panic!("independent replay")
    };
    assert_eq!(replayed.provenance(), NextTokenProvenance::OriginalReplay);
    assert_eq!(replayed.scope(), sealed.scope());
    report.evidence.rows[0].upper = -99.0;
    assert!(report.replay(source, normal, pre, 0, &limits).is_err());
    assert_eq!(sealed.provenance(), NextTokenProvenance::CheckedQuery);
    assert_eq!(sealed.maximum_competitor_upper(), Some(-0.25));
    let mut duplicate = replay.evidence.clone();
    duplicate.rows[1].competitor = 1;
    assert!(replay_normalized_next_token(
        source,
        normal,
        pre,
        0,
        &duplicate,
        options(&ledger, &limits)
    )
    .is_err());
    duplicate.rows.pop();
    assert!(replay_normalized_next_token(
        source,
        normal,
        pre,
        0,
        &duplicate,
        options(&ledger, &limits)
    )
    .is_err());
    drop(report);
    drop(replay);
    assert!(ledger.receipt().unwrap().live_persistent_bytes > 0);
    drop(sealed);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn decoder_n12_resource_deadline_and_ledger_substitution_refuse_before_allocation() {
    let source = affine(1, &[0.0, 0.0], &[0.0, -0.25]);
    let gamma = [1.0];
    let normal = norm(FinalNormalizationKind::RmsNorm, &gamma, None);
    let pre = HiddenBox {
        lo: &[-1.0],
        hi: &[1.0],
    };
    let ledger = budget();
    let limits = VocabOptions::default();
    let required = shape(source, normal, pre, 0, options(&ledger, &limits)).unwrap();
    let local = VocabOptions {
        max_terms: required.work - 1,
        ..limits.clone()
    };
    let before = ledger.receipt().unwrap();
    assert!(
        verify_normalized_next_token(source, normal, pre, 0, options(&ledger, &local)).is_err()
    );
    assert_eq!(ledger.receipt().unwrap(), before);
    let expired = VocabOptions {
        deadline: Some(Instant::now()),
        ..limits.clone()
    };
    assert!(
        verify_normalized_next_token(source, normal, pre, 0, options(&ledger, &expired)).is_err()
    );
    assert_eq!(ledger.receipt().unwrap(), before);
    let substitute = VocabOptions {
        budget: Some(budget()),
        ..limits
    };
    assert!(
        verify_normalized_next_token(source, normal, pre, 0, options(&ledger, &substitute))
            .is_err()
    );
    assert_eq!(ledger.receipt().unwrap(), before);
    let starved = WorkBudget::new(
        None,
        required.scratch_bytes - 1,
        1 << 20,
        1 << 30,
        32,
        Duration::ZERO,
    )
    .unwrap();
    assert!(
        verify_normalized_next_token(source, normal, pre, 0, options(&starved, &limits)).is_err()
    );
    let refused = starved.receipt().unwrap();
    assert_eq!(refused.attempted, 1);
    assert_eq!(refused.admitted, 0);
    assert_eq!(refused.live_persistent_bytes, 0);
    assert_eq!(refused.live_scratch_bytes, 0);
    let report =
        verify_normalized_next_token(source, normal, pre, 0, options(&ledger, &limits)).unwrap();
    let before = ledger.receipt().unwrap();
    let replay = report.replay(source, normal, pre, 0, &limits).unwrap();
    assert_eq!(
        ledger.receipt().unwrap().work_reserved - before.work_reserved,
        required.work
    );
    assert_eq!(replay.reserved_bytes(), required.persistent_bytes);
    drop(report);
    drop(replay);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(ledger.receipt().unwrap().live_scratch_bytes, 0);
}

#[test]
fn decoder_n12_nonexact_heads_nonfinite_parameters_and_changed_epsilon_fail_closed() {
    let source = affine(1, &[0.0, 0.0], &[0.0, -0.25]);
    let gamma = [1.0];
    let normal = norm(FinalNormalizationKind::LayerNorm, &gamma, None);
    let pre = HiddenBox {
        lo: &[-1.0],
        hi: &[1.0],
    };
    let ledger = budget();
    let limits = VocabOptions::default();
    for arithmetic in [
        HeadArithmetic::SequentialBinary64,
        HeadArithmetic::Binary32 {
            reduction: super::super::Binary32Reduction::SequentialSeparate,
        },
    ] {
        assert!(matches!(
            verify_normalized_next_token(
                AffineHead {
                    arithmetic,
                    ..source
                },
                normal,
                pre,
                0,
                options(&ledger, &limits)
            ),
            Err(NormalizedHeadRefusal::UnsupportedHeadArithmetic(_))
        ));
    }
    for epsilon in [0.0, 1e-15, f64::NAN, f64::INFINITY] {
        assert!(verify_normalized_next_token(
            source,
            FinalNormalization { epsilon, ..normal },
            pre,
            0,
            options(&ledger, &limits)
        )
        .is_err());
    }
    for changed in [
        FinalNormalization {
            gamma: &[f64::INFINITY],
            ..normal
        },
        FinalNormalization {
            beta: Some(&[f64::NAN]),
            ..normal
        },
    ] {
        assert!(
            verify_normalized_next_token(source, changed, pre, 0, options(&ledger, &limits))
                .is_err()
        );
    }
    let invalid = affine(1, &[f64::INFINITY, f64::INFINITY], &[0.0, -0.25]);
    assert!(
        verify_normalized_next_token(invalid, normal, pre, 0, options(&ledger, &limits)).is_err()
    );
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn decoder_n12_subnormal_anchor_and_kernel_shortcuts_have_checked_original_premises() {
    let tiny = f64::from_bits(1);
    let source = affine(2, &[0.0, 0.0, 1e200, -1e200], &[0.0, -0.25]);
    let gamma = [1.0, 1.0];
    let normal = norm(FinalNormalizationKind::LayerNorm, &gamma, None);
    let uniform = HiddenBox {
        lo: &[tiny, tiny],
        hi: &[tiny, tiny],
    };
    let pre = HiddenBox {
        lo: &[tiny, 2.0 * tiny],
        hi: &[tiny, 2.0 * tiny],
    };
    let ledger = budget();
    let limits = VocabOptions::default();
    let admitted =
        verify_normalized_next_token(source, normal, uniform, 0, options(&ledger, &limits))
            .unwrap();
    assert!(matches!(
        admitted.outcome,
        NormalizedNextTokenOutcome::Certified(_)
    ));
    assert_eq!(admitted.evidence.rows[0].upper, -0.25);
    original_containment(
        admitted.evidence.rows[0].upper,
        source,
        normal,
        0,
        1,
        uniform.lo,
    );
    // The checked square-root may conservatively refuse a subnormal square
    // enclosure. A decided result must still contain the exact original map.
    if let Ok(report) =
        verify_normalized_next_token(source, normal, pre, 0, options(&ledger, &limits))
    {
        original_containment(report.evidence.rows[0].upper, source, normal, 0, 1, pre.lo);
    }
    let mut check = |_: usize| Ok::<_, DecoderRefusal>(());
    let geometry = GeometryF64::new(&[-1.0], &[1.0], 1e-5, true, &mut check).unwrap();
    assert!(geometry
        .upper(
            &[f64::INFINITY],
            &[f64::INFINITY],
            &[1.0],
            None,
            0.0,
            0.0,
            &mut check
        )
        .is_err());
    assert!(geometry
        .upper(&[0.0], &[0.0], &[f64::INFINITY], None, 0.0, 0.0, &mut check)
        .is_err());
    assert!(geometry
        .upper(
            &[0.0],
            &[0.0],
            &[1.0],
            Some(&[f64::NAN]),
            0.0,
            0.0,
            &mut check
        )
        .is_err());
    assert!(geometry
        .upper(
            &[0.0],
            &[0.0],
            &[1.0],
            None,
            f64::INFINITY,
            f64::INFINITY,
            &mut check
        )
        .is_err());
    let mut calls = 0;
    let mut interrupted = |_: usize| {
        calls += 1;
        if calls >= 3 {
            Err(DecoderRefusal::Deadline)
        } else {
            Ok(())
        }
    };
    assert!(matches!(
        GeometryF64::new(
            &vec![-1.0; 1024],
            &vec![1.0; 1024],
            1e-5,
            true,
            &mut interrupted
        ),
        Err(SupportError::Poll(DecoderRefusal::Deadline))
    ));
}

#[test]
fn decoder_n12_demand_dense_screen_preserves_full_taylor_and_tighter_legacy_replay() {
    let source = affine(3, &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[0.0, -0.47]);
    let gamma = [1.0, 1.0, 1.0];
    let normal = norm(FinalNormalizationKind::RmsNorm, &gamma, None);
    let lo = [0.9999, 1.9999, 2.9999];
    let hi = [1.0001, 2.0001, 3.0001];
    let pre = HiddenBox { lo: &lo, hi: &hi };
    let ledger = budget();
    let limits = VocabOptions::default();
    let report =
        verify_normalized_next_token(source, normal, pre, 0, options(&ledger, &limits)).unwrap();
    assert!(matches!(
        report.outcome,
        NormalizedNextTokenOutcome::Certified(_)
    ));
    assert_eq!(
        report.stats.radial_taylor_rows, 0,
        "a sufficient dense support should avoid stronger Taylor work"
    );
    assert_eq!(
        report.evidence.rows[0].method,
        NormalizedSupportMethod::ProjectedDense
    );
    assert_eq!(report.stats.dense_screened_rows, 1);
    assert_eq!(report.stats.radial_taylor_attempted_rows, 0);
    // The generic kernel still returns its full original minimum. Rebuild a
    // historical tighter Taylor proposal rather than weakening it to the new
    // query's sufficient upper or changing its serialized method tag.
    let mut full_visits = 0u64;
    let mut poll = |count| visits(&mut full_visits, count);
    let geometry = GeometryF64::new(pre.lo, pre.hi, normal.epsilon, false, &mut poll).unwrap();
    let full = geometry
        .upper(
            source.row(1),
            source.row(0),
            &gamma,
            None,
            source.bias[1],
            source.bias[0],
            &mut poll,
        )
        .unwrap();
    assert!(full.taylor_upper.unwrap() < full.dense_upper);
    assert_eq!(full.upper, full.taylor_upper.unwrap());
    assert!(full.upper < report.evidence.rows[0].upper);
    assert!(full_visits > report.stats.kernel_coordinate_visits);
    let mut historical = report.evidence.clone();
    historical.rows[0].upper = full.upper;
    historical.rows[0].method = NormalizedSupportMethod::RadialTaylor;
    for mask in 0..8 {
        let point: Vec<_> = (0..3)
            .map(|i| if mask & (1 << i) == 0 { lo[i] } else { hi[i] })
            .collect();
        original_containment(report.evidence.rows[0].upper, source, normal, 0, 1, &point);
        original_containment(full.upper, source, normal, 0, 1, &point);
    }
    let replay = report.replay(source, normal, pre, 0, &limits).unwrap();
    assert_eq!(replay.evidence, report.evidence);
    assert_eq!(replay.stats.radial_taylor_attempted_rows, 0);
    let replay = replay_normalized_next_token(
        source,
        normal,
        pre,
        0,
        &historical,
        options(&ledger, &limits),
    )
    .unwrap();
    assert_eq!(replay.evidence, historical);
    assert_eq!(replay.stats.radial_taylor_rows, 1);
    assert_eq!(replay.stats.radial_taylor_attempted_rows, 1);
    assert!(matches!(
        replay.outcome,
        NormalizedNextTokenOutcome::Certified(_)
    ));
}

#[test]
fn decoder_n12_radial_taylor_closes_a_margin_the_checked_dense_fallback_leaves_open() {
    let source = affine(1, &[0.0, 1.0], &[0.0, -0.999997]);
    let gamma = [1.0];
    let normal = norm(FinalNormalizationKind::RmsNorm, &gamma, None);
    let pre = HiddenBox {
        lo: &[0.9999],
        hi: &[1.0001],
    };
    let ledger = budget();
    let limits = VocabOptions::default();
    let report =
        verify_normalized_next_token(source, normal, pre, 0, options(&ledger, &limits)).unwrap();
    assert!(matches!(
        report.outcome,
        NormalizedNextTokenOutcome::Certified(_)
    ));
    assert_eq!(report.stats.radial_taylor_rows, 1);
    assert_eq!(
        report.evidence.rows[0].method,
        NormalizedSupportMethod::RadialTaylor
    );
    let mut poll = |_: usize| Ok::<_, DecoderRefusal>(());
    let geometry = GeometryF64::new(pre.lo, pre.hi, normal.epsilon, false, &mut poll).unwrap();
    let independent_candidates = geometry
        .upper(
            source.row(1),
            source.row(0),
            &gamma,
            None,
            source.bias[1],
            source.bias[0],
            &mut poll,
        )
        .unwrap();
    assert!(independent_candidates.dense_upper >= 0.0);
    assert!(independent_candidates.taylor_upper.unwrap() < 0.0);
    for x in [pre.lo[0], 1.0, pre.hi[0]] {
        original_containment(report.evidence.rows[0].upper, source, normal, 0, 1, &[x]);
    }
    let replay = report.replay(source, normal, pre, 0, &limits).unwrap();
    assert!(matches!(
        replay.outcome,
        NormalizedNextTokenOutcome::Certified(_)
    ));
    assert_eq!(replay.evidence, report.evidence);
}

#[test]
fn decoder_n12_negative_numerator_negative_bound_original_rational_taylor_control() {
    let source = affine(1, &[0.0, 1.0], &[0.0, 0.0]);
    let gamma = [1.0];
    let normal = norm(FinalNormalizationKind::RmsNorm, &gamma, None);
    let pre = HiddenBox {
        lo: &[-1.0001],
        hi: &[-0.9999],
    };
    let ledger = budget();
    let limits = VocabOptions::default();
    let report =
        verify_normalized_next_token(source, normal, pre, 0, options(&ledger, &limits)).unwrap();
    assert!(report.evidence.rows[0].upper < 0.0);
    assert_eq!(report.stats.dense_screened_rows, 1);
    assert_eq!(report.stats.radial_taylor_attempted_rows, 0);
    let mut poll = |_: usize| Ok::<_, DecoderRefusal>(());
    let geometry = GeometryF64::new(pre.lo, pre.hi, normal.epsilon, false, &mut poll).unwrap();
    let full = geometry
        .upper(
            source.row(1),
            source.row(0),
            &gamma,
            None,
            source.bias[1],
            source.bias[0],
            &mut poll,
        )
        .unwrap();
    assert!(full.upper < report.evidence.rows[0].upper);
    let mut historical = report.evidence.clone();
    historical.rows[0].upper = full.upper;
    historical.rows[0].method = NormalizedSupportMethod::RadialTaylor;
    for x in [pre.lo[0], -1.0, pre.hi[0]] {
        // Here B<0 and N<0, exercising the reversed exact square inequality.
        original_containment(report.evidence.rows[0].upper, source, normal, 0, 1, &[x]);
        original_containment(full.upper, source, normal, 0, 1, &[x]);
    }
    assert_eq!(
        report
            .replay(source, normal, pre, 0, &limits)
            .unwrap()
            .evidence,
        report.evidence
    );
    let replay = replay_normalized_next_token(
        source,
        normal,
        pre,
        0,
        &historical,
        options(&ledger, &limits),
    )
    .unwrap();
    assert_eq!(replay.evidence, historical);
    assert_eq!(replay.stats.radial_taylor_rows, 1);
    assert_eq!(replay.stats.radial_taylor_attempted_rows, 1);
}

#[test]
fn decoder_n12_demand_signed_dense_screen_counts_actual_scans_and_preserves_rational_support() {
    let weights = [0.0, 0.0, 0.0, 1.0, -0.5, 0.25];
    let biases = [0.0, -10.0];
    let gamma = [-1.5, 0.0, 2.0];
    let beta = [0.25, -0.125, 0.5];
    let normal = norm(FinalNormalizationKind::LayerNorm, &gamma, Some(&beta));
    let source = affine(3, &weights, &biases);
    let lo = [-2.0, -0.5, 0.5];
    let hi = [-1.75, -0.25, 0.75];
    let pre = HiddenBox { lo: &lo, hi: &hi };
    let ledger = budget();
    let limits = VocabOptions::default();
    let report =
        verify_normalized_next_token(source, normal, pre, 0, options(&ledger, &limits)).unwrap();
    let NormalizedNextTokenOutcome::Certified(proof) = &report.outcome else {
        panic!("checked signed dense support should decide this margin")
    };
    assert_eq!(report.stats.dense_screened_rows, 1);
    assert_eq!(report.stats.cauchy_screened_rows, 0);
    assert_eq!(report.stats.radial_taylor_attempted_rows, 0);
    assert_eq!(report.stats.kernel_coordinate_visits, 12 * 3);
    assert_eq!(report.stats.original_head_visits, 8);
    assert_eq!(report.stats.coverage_visits, 1);
    assert_eq!(proof.checked_stats(), &report.stats);
    assert_eq!(
        report.stats.reserved_work,
        normalized_work_bound(2, 3).unwrap()
    );
    assert_eq!(
        ledger.receipt().unwrap().work_reserved,
        report.stats.reserved_work
    );
    for vertex in 0..8 {
        let point: Vec<_> = (0..3)
            .map(|i| if vertex & (1 << i) == 0 { lo[i] } else { hi[i] })
            .collect();
        original_containment(report.evidence.rows[0].upper, source, normal, 0, 1, &point);
    }
    let replay = report.replay(source, normal, pre, 0, &limits).unwrap();
    assert_eq!(replay.evidence, report.evidence);
    assert_eq!(
        replay.stats.kernel_coordinate_visits,
        report.stats.kernel_coordinate_visits
    );
    assert_eq!(replay.stats.radial_taylor_attempted_rows, 0);
}

#[test]
fn decoder_n12_demand_cauchy_screens_a_wide_region_without_taylor_or_sampling() {
    let source = affine(1, &[0.0, 1.0], &[0.0, -2.0]);
    let gamma = [-1.0];
    let normal = norm(FinalNormalizationKind::RmsNorm, &gamma, None);
    let pre = HiddenBox {
        lo: &[-1.0],
        hi: &[1.0],
    };
    let ledger = budget();
    let limits = VocabOptions::default();
    let report =
        verify_normalized_next_token(source, normal, pre, 0, options(&ledger, &limits)).unwrap();
    assert!(matches!(
        report.outcome,
        NormalizedNextTokenOutcome::Certified(_)
    ));
    assert_eq!(report.stats.dense_screened_rows, 0);
    assert_eq!(report.stats.cauchy_screened_rows, 1);
    assert_eq!(report.stats.radial_taylor_attempted_rows, 0);
    assert_eq!(report.stats.kernel_coordinate_visits, 10);
    assert!(report.evidence.rows[0].upper < 0.0);
    for point in [-1.0, -0.75, 0.0, 0.25, 1.0] {
        original_containment(
            report.evidence.rows[0].upper,
            source,
            normal,
            0,
            1,
            &[point],
        );
    }
    let replay = report.replay(source, normal, pre, 0, &limits).unwrap();
    assert_eq!(replay.evidence, report.evidence);
    assert_eq!(replay.stats.cauchy_screened_rows, 1);
    assert_eq!(replay.stats.radial_taylor_attempted_rows, 0);
}

#[test]
fn decoder_n12_demand_replay_tags_do_not_authorize_bounds_or_replace_sealed_execution() {
    let source = affine(1, &[0.0, 0.0], &[0.0, -0.25]);
    let gamma = [1.0];
    let normal = norm(FinalNormalizationKind::RmsNorm, &gamma, None);
    let pre = HiddenBox {
        lo: &[-1.0],
        hi: &[2.0],
    };
    let ledger = budget();
    let limits = VocabOptions::default();
    let mut report =
        verify_normalized_next_token(source, normal, pre, 0, options(&ledger, &limits)).unwrap();
    let NormalizedNextTokenOutcome::Certified(proof) = &report.outcome else {
        panic!("checked constant margin")
    };
    let sealed = proof.clone();
    let saved = report.stats.clone();
    report.evidence.rows[0].method = NormalizedSupportMethod::RadialTaylor;
    let replay = report.replay(source, normal, pre, 0, &limits).unwrap();
    // The legacy wire tag remains a proposal. Fresh measured authority records
    // the dense stage which actually authorized its upper.
    assert_eq!(replay.evidence, report.evidence);
    assert_eq!(replay.stats.projected_dense_rows, 1);
    assert_eq!(replay.stats.radial_taylor_rows, 0);
    assert_eq!(replay.stats.radial_taylor_attempted_rows, 0);
    let NormalizedNextTokenOutcome::Certified(replayed) = &replay.outcome else {
        panic!("original checked replay")
    };
    assert_eq!(replayed.checked_stats(), &replay.stats);
    assert_eq!(replayed.provenance(), NextTokenProvenance::OriginalReplay);
    report.outcome = NormalizedNextTokenOutcome::Certified(replayed.clone());
    assert_eq!(report.provenance(), NextTokenProvenance::OriginalReplay);
    assert_eq!(sealed.provenance(), NextTokenProvenance::CheckedQuery);
    report.stats.radial_taylor_rows = usize::MAX;
    report.stats.kernel_coordinate_visits = 0;
    assert_eq!(sealed.checked_stats(), &saved);
    report.evidence.rows[0].upper = -99.0;
    assert!(report.replay(source, normal, pre, 0, &limits).is_err());
    assert_eq!(sealed.maximum_competitor_upper(), Some(-0.25));
}

#[test]
fn decoder_n12_normalized_mints_have_distinct_scope_owners_and_clones_share_checked_state() {
    let source = affine(1, &[0.0, 0.0], &[0.0, -0.25]);
    let normal = norm(FinalNormalizationKind::RmsNorm, &[1.0], None);
    let pre = HiddenBox {
        lo: &[-1.0],
        hi: &[2.0],
    };
    let ledger = budget();
    let limits = VocabOptions::default();
    let first =
        verify_normalized_next_token(source, normal, pre, 0, options(&ledger, &limits)).unwrap();
    let second =
        verify_normalized_next_token(source, normal, pre, 0, options(&ledger, &limits)).unwrap();
    let replay = first.replay(source, normal, pre, 0, &limits).unwrap();
    let NormalizedNextTokenOutcome::Certified(proof) = &first.outcome else {
        panic!("checked constant margin")
    };
    let alias = proof.clone();
    assert!(Arc::ptr_eq(&proof.inner, &alias.inner));
    assert!(std::ptr::eq(proof.scope(), alias.scope()));
    assert!(std::ptr::eq(proof.checked_stats(), alias.checked_stats()));
    for report in [&second, &replay] {
        let NormalizedNextTokenOutcome::Certified(other) = &report.outcome else {
            panic!("checked constant margin")
        };
        assert_eq!(proof.scope(), other.scope());
        assert!(!Arc::ptr_eq(&proof.inner, &other.inner));
        assert!(!std::ptr::eq(proof.scope(), other.scope()));
    }
    assert!(
        size_of::<NormalizedHeadReport>()
            + size_of_val(proof.inner.as_ref())
            + 2 * size_of::<usize>()
            <= 4096
    );
}

#[test]
fn decoder_n12_demand_ties_never_take_the_strict_query_screen() {
    let source = affine(1, &[1.0, 1.0], &[0.0, 0.0]);
    let gamma = [-1.0];
    let normal = norm(FinalNormalizationKind::RmsNorm, &gamma, None);
    let pre = HiddenBox {
        lo: &[-1.0],
        hi: &[2.0],
    };
    let ledger = budget();
    let limits = VocabOptions::default();
    let report =
        verify_normalized_next_token(source, normal, pre, 0, options(&ledger, &limits)).unwrap();
    assert_eq!(report.evidence.rows[0].upper, 0.0);
    assert_eq!(report.stats.dense_screened_rows, 0);
    assert_eq!(report.stats.cauchy_screened_rows, 0);
    assert_eq!(report.stats.radial_taylor_attempted_rows, 1);
    assert!(matches!(
        report.outcome,
        NormalizedNextTokenOutcome::Unknown
    ));
    // Replay may authorize equality to a proposed zero bound cheaply. Equality
    // to the proposed bound still supplies no strict argmax decision.
    let replay = report.replay(source, normal, pre, 0, &limits).unwrap();
    assert_eq!(replay.stats.dense_screened_rows, 1);
    assert_eq!(replay.stats.radial_taylor_attempted_rows, 0);
    assert_eq!(replay.unresolved, [1]);
    assert!(matches!(
        replay.outcome,
        NormalizedNextTokenOutcome::Unknown
    ));
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "retain the genuine replay evidence while corrupting an independent copy"
)]
fn decoder_n12_full_support_retains_all_stages_and_replays_the_tighter_margin() {
    let source = affine(1, &[0.0, 1.0], &[0.0, -3.0]);
    let gamma = [1.0];
    let normal = norm(FinalNormalizationKind::RmsNorm, &gamma, None);
    let pre = HiddenBox {
        lo: &[0.75],
        hi: &[1.25],
    };
    let ledger = budget();
    let limits = VocabOptions::default();
    let decision =
        verify_normalized_next_token(source, normal, pre, 0, options(&ledger, &limits)).unwrap();
    let full = verify_normalized_next_token_full_support(
        source,
        normal,
        pre,
        0,
        options(&ledger, &limits),
    )
    .unwrap();
    assert_eq!(decision.stats.dense_screened_rows, 1);
    assert_eq!(decision.stats.radial_taylor_attempted_rows, 0);
    assert_eq!(full.stats.dense_screened_rows, 0);
    assert_eq!(full.stats.cauchy_screened_rows, 0);
    assert_eq!(full.stats.radial_taylor_attempted_rows, 1);
    assert!(full.stats.kernel_coordinate_visits > decision.stats.kernel_coordinate_visits);
    assert_eq!(full.stats.reserved_work, decision.stats.reserved_work);
    assert!(full.evidence.rows[0].upper < decision.evidence.rows[0].upper);
    assert!(matches!(
        full.outcome,
        NormalizedNextTokenOutcome::Certified(_)
    ));
    for point in [0.75, 1.0, 1.25] {
        original_containment(full.evidence.rows[0].upper, source, normal, 0, 1, &[point]);
    }
    let replay = full.replay(source, normal, pre, 0, &limits).unwrap();
    assert_eq!(replay.evidence, full.evidence);
    assert_eq!(replay.provenance(), NextTokenProvenance::OriginalReplay);
    let mut false_margin = full.evidence.clone();
    false_margin.rows[0].upper = -4.0;
    assert!(replay_normalized_next_token(
        source,
        normal,
        pre,
        0,
        &false_margin,
        options(&ledger, &limits),
    )
    .is_err());
}

#[test]
fn decoder_n12_full_support_preserves_arithmetic_work_and_ledger_refusals() {
    let source = affine(1, &[0.0, 1.0], &[0.0, -3.0]);
    let gamma = [1.0];
    let normal = norm(FinalNormalizationKind::RmsNorm, &gamma, None);
    let pre = HiddenBox {
        lo: &[0.75],
        hi: &[1.25],
    };
    let ledger = budget();
    let limits = VocabOptions::default();
    assert!(matches!(
        verify_normalized_next_token_full_support(
            AffineHead {
                arithmetic: HeadArithmetic::SequentialBinary64,
                ..source
            },
            normal,
            pre,
            0,
            options(&ledger, &limits),
        ),
        Err(NormalizedHeadRefusal::UnsupportedHeadArithmetic(_))
    ));
    let too_little_work = VocabOptions {
        max_terms: normalized_work_bound(2, 1).unwrap() - 1,
        ..limits.clone()
    };
    assert!(verify_normalized_next_token_full_support(
        source,
        normal,
        pre,
        0,
        options(&ledger, &too_little_work),
    )
    .is_err());
    let foreign_ledger = VocabOptions {
        budget: Some(budget()),
        ..limits
    };
    assert!(verify_normalized_next_token_full_support(
        source,
        normal,
        pre,
        0,
        options(&ledger, &foreign_ledger),
    )
    .is_err());
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(ledger.receipt().unwrap().live_scratch_bytes, 0);
}

#[test]
fn decoder_n12_full_support_keeps_complete_coverage_and_exact_ties_unknown() {
    let source = affine(1, &[1.0, 1.0, 1.0], &[0.0, -0.25, 0.0]);
    let gamma = [-1.0];
    let normal = norm(FinalNormalizationKind::RmsNorm, &gamma, None);
    let pre = HiddenBox {
        lo: &[-1.0],
        hi: &[2.0],
    };
    let ledger = budget();
    let limits = VocabOptions::default();
    let report = verify_normalized_next_token_full_support(
        source,
        normal,
        pre,
        0,
        options(&ledger, &limits),
    )
    .unwrap();
    assert_eq!(report.evidence.rows.len(), 2);
    assert_eq!(report.evidence.rows[0].competitor, 1);
    assert_eq!(report.evidence.rows[1].competitor, 2);
    assert_eq!(report.evidence.rows[1].upper, 0.0);
    assert_eq!(report.stats.coverage_visits, 2);
    assert_eq!(report.unresolved, [2]);
    assert!(matches!(
        report.outcome,
        NormalizedNextTokenOutcome::Unknown
    ));
    let replay = report.replay(source, normal, pre, 0, &limits).unwrap();
    assert_eq!(replay.evidence, report.evidence);
    assert!(matches!(
        replay.outcome,
        NormalizedNextTokenOutcome::Unknown
    ));
    drop(report);
    assert!(ledger.receipt().unwrap().live_persistent_bytes > 0);
    drop(replay);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
}
