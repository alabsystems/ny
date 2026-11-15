// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Authored Source controls; no compiler-proof or competition-score claim.
use super::*;

fn end() -> Instant {
    Instant::now() + Duration::from_secs(10)
}
fn work() -> Budget {
    Budget {
        deadline: end(),
        cells: 0,
        masks: 0,
    }
}

fn dense_spec(bias: f32) -> InputHeadSpec {
    let (d, h) = (65, 73);
    let (epsilon, delta) = (1.0f32 / 4096.0, 1.0f32 / 256.0);
    let common = 1.0 + 65.0 * epsilon + delta * epsilon;
    let mut first = vec![0.0; d * h];
    for i in 0..d {
        for j in 0..65 {
            first[i * h + j] = epsilon + if i == j { 1.0 } else { 0.0 };
        }
        for k in 0..8 {
            first[i * h + 65 + k] = common + if i == k { delta } else { 0.0 };
        }
    }
    let last = (0..h)
        .flat_map(|j| {
            [
                if j < 8 {
                    8.0 + delta
                } else if j < 65 {
                    8.0
                } else {
                    -1.0
                },
                0.0,
            ]
        })
        .collect();
    InputHeadSpec::new(d, h, first, vec![0.0; h], last, vec![bias, 0.0], [117; 32]).unwrap()
}

fn dense(bias: f32) -> Source {
    Source::build(&dense_spec(bias), &[-1.0; 65], &[1.0; 65], 0, 1, end()).unwrap()
}

fn small(bias: f32) -> Source {
    let spec = InputHeadSpec::new(
        1,
        2,
        vec![1.0, 1.0],
        vec![0.0; 2],
        vec![1.0, 0.0, -1.0, 0.0],
        vec![bias, 0.0],
        [118; 32],
    )
    .unwrap();
    Source::build(&spec, &[-1.0], &[1.0], 0, 1, end()).unwrap()
}

fn simple_cut<'a>(source: &'a Source, budget: &mut Budget) -> Option<CertifiedDirection<'a>> {
    certify(
        source,
        Direction {
            heads: vec![(1, 1.0)],
        },
        PositiveTerms(vec![(0, 1.0)]),
        vec![],
        budget,
    )
}

fn dense_cut<'a>(source: &'a Source, budget: &mut Budget) -> CertifiedDirection<'a> {
    let (negative, positive) = signed_search::anchor(source, budget).unwrap();
    assert_eq!(negative.heads.len(), 8);
    assert_eq!(positive.0.len(), 65);
    assert_eq!(positive.0.last().unwrap().0, 64); // beyond the old32-positive family
    certify(source, negative, positive, vec![], budget).unwrap()
}

fn floor(source: &Source) -> f64 {
    let mut budget = work();
    let cut = dense_cut(source, &mut budget);
    assert_eq!(budget.masks, 256); // includes the empty algebraic mask
    assert!(
        cut.lower > -0.04 && cut.lower <= 0.0,
        "geometry floor {}",
        cut.lower
    );
    let cuts = vec![cut];
    let model = triangle::Model::from_root_portfolio(source, &mut None, &cuts, end()).unwrap();
    let hint = native::portfolio_hint(&model, &cuts, &vec![0.0; 65], &mut budget).unwrap();
    assert!(hint.row_duals.last().copied().unwrap() > 0.99);
    model.replay(source, &hint, 1.0, end()).unwrap()
}

#[test]
fn signed_dense_65_input_73_head_common_slope_has_exact_mask_reference() {
    let spec = dense_spec(0.5);
    assert!(spec.first.iter().all(|&a| a != 0.0));
    let mut maximum = 0i64;
    for mask in 0..256usize {
        let k = i64::from(mask.count_ones());
        let mut numerator = 0i64;
        for i in 0..65 {
            let all = 8 * 1_065_217i64 + if i < 8 { 4096 } else { 0 };
            let selected = k * 1_065_217
                + if i < 8 && mask & (1 << i) != 0 {
                    4096
                } else {
                    0
                };
            numerator += (k * all - 8 * selected).abs();
        }
        assert_eq!(numerator, 8192 * k * (8 - k));
        maximum = maximum.max(numerator);
    }
    assert_eq!(maximum, (1 << 23) / 64); // every ideal mask floor >= -1/64
    let denominator = 1i64 << 23;
    let biased_floor = denominator / 2 - maximum;
    assert_eq!(64 * biased_floor, 31 * denominator); // ideal >=31/64
    for a in 0..73 {
        for b in a + 1..73 {
            // Exact products of these small dyadics fit f64's53-bit significand.
            assert!((1..65).any(|i| {
                f64::from(spec.first[i * 73 + a]) * f64::from(spec.first[b])
                    != f64::from(spec.first[i * 73 + b]) * f64::from(spec.first[a])
            }));
        }
    }
}

#[test]
fn signed_dense_source_replay_closes_safe_and_keeps_unsafe_below_true_endpoint() {
    let safe = floor(&dense(0.5));
    assert!(safe > 0.45 && safe <= 0.5, "safe Source floor {safe}");
    let unsafe_floor = floor(&dense(-0.5));
    assert!(unsafe_floor <= -0.5, "unsafe Source floor {unsafe_floor}");
}

#[test]
fn signed_dense_old_triangle_and_seven_head_cover_have_explicit_negative_points() {
    let source = dense(0.5);
    let old = triangle::Model::build(&source, end()).unwrap();
    let cover = phase_cover::Model::build_with_prefix(
        &source,
        PhaseCoverPolicy::new(7).unwrap(),
        Some(old.into_phase_prefix()),
        end(),
    )
    .unwrap();
    let request = cover.request();
    assert_eq!(request.leaves.len(), 128);
    for leaf in &request.leaves {
        let mut point = vec![0.0; 65 + 73];
        for j in 65..73 {
            point[65 + j] = 1.0;
        }
        for row in &leaf.rows {
            for &(j, a) in &row.coefficients {
                if j >= 65 && a == -1.0 {
                    point[j] = 0.0;
                }
            }
        }
        assert!(request
            .bounds
            .iter()
            .zip(&point)
            .all(|(&[lo, hi], &x)| lo <= x && x <= hi));
        for row in request.rows.iter().chain(&leaf.rows) {
            let lhs = row.coefficients.iter().fold(I::p(0.0), |sum, &(j, a)| {
                sum.add(I::p(a).mul(I::p(point[j])).unwrap()).unwrap()
            });
            assert!(
                lhs.lo >= row.lower,
                "old leaf{} row excludes point",
                leaf.mask
            );
        }
        let margin = request
            .objective
            .iter()
            .zip(&point)
            .fold(I::p(0.5), |sum, (&a, &x)| {
                sum.add(I::p(a).mul(I::p(x)).unwrap()).unwrap()
            });
        assert!(margin.hi < 0.0); // outward check, with no feasibility tolerance
    }
    // Actual old private triangle+fixed7 LP rows, not an executed LP solve.
    // The independent exact reference covers arbitrary negative-only rows;
    // no dominance over arbitrary mixed-sign/global NY cuts is claimed here.
}

#[test]
fn signed_original_q_anchor_is_automatically_admitted_before_local_shortlist() {
    let source = dense(0.5);
    let (report, prefix) = refine_with_prefix(&source, -1000.0, None, end(), end(), &|| None);
    assert!(
        report.lower > 0.45 && report.lower <= 0.5,
        "automatic floor {}",
        report.lower
    );
    assert_eq!(report.rounds.len(), 1);
    assert_eq!(report.cuts, 1);
    assert_eq!(report.attempted, 1);
    assert_eq!(report.masks, 256);
    assert_eq!(report.calls, 0); // native closure does not fabricate AY engagement
    assert_eq!(report.native_replayed, 1);
    assert!(report.work_cells <= MAX_CELLS && prefix.is_some());
}

#[test]
fn signed_unsafe_automatic_refinement_never_crosses_original_counterexample() {
    let source = dense(-0.5);
    let (report, _) = refine_with_prefix(&source, -1000.0, None, end(), end(), &|| None);
    assert!(report.lower <= -0.5);
    assert!(report.cuts >= 1 && report.cuts <= MAX_CUTS * MAX_ROUNDS);
    assert!(
        report.masks <= MAX_MASKS
            && report.work_cells <= MAX_CELLS
            && report.attempted <= MAX_ATTEMPTS
    );
    assert!(report.calls <= MAX_ROUNDS && report.native_replayed <= MAX_ROUNDS);
}

#[test]
fn signed_row_exports_positive_and_negative_heads_with_full_original_objective() {
    let source = small(0.5);
    let cut = simple_cut(&source, &mut work()).unwrap();
    let model = triangle::Model::from_root_portfolio(&source, &mut None, &[cut], end()).unwrap();
    let request = model.request();
    assert_eq!(request.feature_count, 1);
    assert_eq!(request.objective.len(), 3);
    assert_eq!(request.objective[0], 0.0);
    assert!(request.objective[1] > 0.0 && request.objective[2] < 0.0);
    assert_eq!(
        request.rows.last().unwrap().coefficients,
        vec![(1, 1.0), (2, -1.0)]
    );
}

#[test]
fn signed_geometry_rejects_foreign_source_and_changed_exact_coefficients() {
    let source = small(0.5);
    let other = small(0.5);
    assert!(simple_cut(&source, &mut work())
        .unwrap()
        .owned_row(&other, end())
        .is_none());
    for kind in 0..3 {
        let mut cut = simple_cut(&source, &mut work()).unwrap();
        match kind {
            0 => cut.direction.heads[0].1 = 0.5,
            1 => {
                if let GeometryKind::Signed(proof) = &mut cut.geometry {
                    proof.terms.0[0].1 = 0.5;
                }
            }
            _ => cut.eta.push((0, 0.5)),
        }
        assert!(cut.owned_row(&source, end()).is_none());
    }
}

#[test]
fn signed_geometry_refuses_overlapping_duplicate_and_invalid_terms() {
    let source = small(0.5);
    for positive in [
        vec![],
        vec![(1, 1.0)],
        vec![(0, -1.0)],
        vec![(0, f64::NAN)],
        vec![(0, 1.0), (0, 1.0)],
        vec![(2, 1.0)],
    ] {
        assert!(certify(
            &source,
            Direction {
                heads: vec![(1, 1.0)]
            },
            PositiveTerms(positive),
            vec![],
            &mut work()
        )
        .is_none());
    }
    assert!(certify(
        &source,
        Direction { heads: vec![] },
        PositiveTerms(vec![(0, 1.0)]),
        vec![],
        &mut work()
    )
    .is_none());
    assert!(certify(
        &source,
        Direction {
            heads: vec![(1, 0.0)]
        },
        PositiveTerms(vec![(0, 1.0)]),
        vec![],
        &mut work()
    )
    .is_none());
}

#[test]
fn signed_positive_and_negative_first_errors_both_reduce_geometry_floor() {
    let source = small(0.5);
    let before = simple_cut(&source, &mut work()).unwrap().lower;
    for head in [0, 1] {
        let mut changed = small(0.5);
        changed.first_error[head] += 0.25;
        let after = simple_cut(&changed, &mut work()).unwrap().lower;
        assert!(after < before - 0.24, "head{head}: {before} -> {after}");
    }
    let changed_bias = small(100.0);
    assert_eq!(
        simple_cut(&changed_bias, &mut work())
            .unwrap()
            .lower
            .to_bits(),
        before.to_bits()
    );
}

#[test]
fn signed_certificate_includes_last_original_input_residual() {
    let mut spec = dense_spec(0.5);
    spec.first[64 * 73 + 65] += 1.0;
    let source = Source::build(&spec, &[-1.0; 65], &[1.0; 65], 0, 1, end()).unwrap();
    let cut = dense_cut(&source, &mut work());
    assert!(cut.lower <= -1.0, "last input omitted: {}", cut.lower);
}

#[test]
fn signed_empty_mask_limits_positive_floor_and_is_not_zero_clamped() {
    let spec = InputHeadSpec::new(
        1,
        2,
        vec![0.0, 1.0],
        vec![-10.0, -2.0],
        vec![1.0, 0.0, -1.0, 0.0],
        vec![0.0; 2],
        [119; 32],
    )
    .unwrap();
    let source = Source::build(&spec, &[1.0], &[3.0], 0, 1, end()).unwrap();
    let mut budget = work();
    let cut = certify(
        &source,
        Direction {
            heads: vec![(1, 1.0)],
        },
        PositiveTerms(vec![(0, 1.0)]),
        vec![(0, 1.0)],
        &mut budget,
    )
    .unwrap();
    assert_eq!(budget.masks, 2);
    assert!(cut.lower > 0.99 && cut.lower <= 1.0, "{}", cut.lower);
}

#[test]
fn signed_mask_work_deadline_and_partial_cover_fail_closed() {
    let source = small(0.5);
    let mut budget = Budget {
        cells: MAX_CELLS - 1,
        ..work()
    };
    assert!(simple_cut(&source, &mut budget).is_none());
    assert!(budget.cells <= MAX_CELLS);
    let mut budget = Budget {
        masks: MAX_MASKS - 1,
        ..work()
    };
    assert!(simple_cut(&source, &mut budget).is_none());
    assert_eq!(budget.masks, MAX_MASKS - 1);
    for count in [1, 2] {
        EXPIRE_AFTER_MASK.with(|fault| fault.set(Some(count)));
        let mut budget = work();
        let result = simple_cut(&source, &mut budget);
        EXPIRE_AFTER_MASK.with(|fault| fault.set(None));
        assert!(result.is_none());
        assert_eq!(budget.masks, count);
    }
    let mut budget = Budget {
        deadline: Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap(),
        ..work()
    };
    assert!(simple_cut(&source, &mut budget).is_none());
}

#[test]
fn signed_failed_cover_retains_completed_scalar_prefix() {
    let source = small(0.5);
    let model = triangle::Model::build_phase_base(&source, end()).unwrap();
    let prefix = model.into_phase_prefix();
    let previous = prefix.root_work_hint();
    EXPIRE_AFTER_MASK.with(|fault| fault.set(Some(1)));
    let (report, retained) =
        refine_with_prefix(&source, -3.0, Some(prefix), end(), end(), &|| None);
    EXPIRE_AFTER_MASK.with(|fault| fault.set(None));
    assert_eq!(report.lower, -3.0);
    assert_eq!(report.cuts, 0);
    assert_eq!(retained.unwrap().root_work_hint(), previous);
}

#[test]
fn signed_bad_advice_retains_completed_signed_rows_and_native_floor() {
    struct Bad;
    impl HeadPhaseAdvice for Bad {
        fn propose(
            &self,
            _: &ny_core::head_phase_advice::HeadPhaseLpRequest,
            _: Instant,
        ) -> Option<ny_core::head_phase_advice::HeadPhaseLpHint> {
            None
        }
        fn supports_functional_advice(&self) -> bool {
            true
        }
        fn propose_functional(
            &self,
            request: &HeadFunctionalLpRequest,
            _: Instant,
        ) -> Option<HeadFunctionalLpHint> {
            Some(HeadFunctionalLpHint {
                row_duals: vec![f64::NAN; request.rows.len()],
            })
        }
    }
    let source = small(-0.5);
    let baseline = triangle::Model::build_phase_base(&source, end())
        .unwrap()
        .into_phase_prefix()
        .root_work_hint()
        .unwrap();
    let advice: Arc<dyn HeadPhaseAdvice> = Arc::new(Bad);
    let (report, prefix) = refine_with_prefix(&source, -3.0, None, end(), end(), &|| {
        Some(Arc::clone(&advice))
    });
    assert!(report.lower > -3.0 && report.lower <= -0.5);
    assert_eq!(report.cuts, 1);
    assert_eq!(report.calls, 1);
    assert_eq!(report.native_replayed, 1);
    assert_eq!(report.ay_replayed, 0);
    assert!(prefix.unwrap().root_work_hint().unwrap() > baseline);
}

#[test]
fn signed_empty_positive_family_skips_without_changing_negative_validity() {
    let spec = InputHeadSpec::new(
        1,
        7,
        vec![1.0; 7],
        vec![0.0; 7],
        (0..7).flat_map(|_| [-1.0, 0.0]).collect(),
        vec![100.0, 0.0],
        [120; 32],
    )
    .unwrap();
    let source = Source::build(&spec, &[-1.0], &[1.0], 0, 1, end()).unwrap();
    let mut budget = work();
    assert!(signed_search::anchor(&source, &mut budget).is_none());
    assert_eq!(budget.masks, 0);
    assert!(Direction {
        heads: (0..7).map(|j| (j, 1.0)).collect()
    }
    .valid(&source));
}

#[path = "input_head_directional_signed_production_tests.rs"]
mod production_tests;
