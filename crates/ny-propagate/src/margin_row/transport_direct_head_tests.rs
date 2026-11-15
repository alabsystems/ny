// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::certify::{TokenScope, UnsatSource};
use crate::margin_row::bab::{BabConfig, MarginRowOutcome};
use crate::margin_row::head_group_source::HeadGroupSourceGuard;
use crate::margin_row::net::TwinNet;
use crate::margin_row::root::RootGates;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use num_rational::BigRational;
use num_traits::Zero;

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(10)
}
fn policy() -> HeadGroupPolicy {
    HeadGroupPolicy::new(5.0, 2, 4).unwrap()
}
fn rat(x: f64) -> BigRational {
    BigRational::from_float(x).unwrap()
}

// The extra live unstable trunk channel is authored, not an inserted layer;
// Gemm1 ignores it. This remains genuinely eligible for ordinary root alpha.
fn twins(pairs: usize, b: f64) -> TwinSpec {
    let width = pairs + 1;
    let heads = 2 * pairs;
    let mut trunk = vec![0.0; width * pairs];
    for j in 0..pairs {
        trunk[j * pairs + j] = 1.0;
    }
    trunk[pairs * pairs] = 1.0;
    let mut first = vec![0.0; heads * width];
    for j in 0..pairs {
        first[j * width + j] = 1.0;
        first[(pairs + j) * width + j] = 1.0;
    }
    let q: Vec<_> = (0..heads)
        .map(|j| if j < pairs { 1.0 } else { -1.0 })
        .collect();
    TwinSpec {
        n_in: pairs,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: trunk,
                bias: (0..width)
                    .map(|j| if j < pairs { 2.0 } else { 0.0 })
                    .collect(),
                bias_err: vec![0.0; width],
                weight_rel_err: 0.0,
                kernel: (width, pairs, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (pairs, 1, 1),
                oshape: (width, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: first,
                bias: vec![-2.0; heads],
                bias_err: vec![0.0; heads],
                weight_rel_err: 0.0,
                shape: (heads, width),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: q
                    .iter()
                    .copied()
                    .chain(vec![0.0; heads])
                    .chain(q.iter().copied())
                    .collect(),
                bias: vec![b, 0.0, b - 1.0],
                bias_err: vec![0.0; 3],
                weight_rel_err: 0.0,
                shape: (3, heads),
            },
        ],
    }
}

fn source(eng: &BackwardEngine<'_>) -> DirectScalarHeadSource {
    DirectScalarHeadSource::build(eng, 0, &[1, 2], deadline()).unwrap()
}
fn margins(eng: &BackwardEngine<'_>, source: &DirectScalarHeadSource) -> Vec<(Vec<I>, I)> {
    (0..source.adversaries.len())
        .map(|r| source.margin(eng, r, deadline()).unwrap())
        .collect()
}
fn floors(source: &DirectScalarHeadSource, margins: &[(Vec<I>, I)]) -> Vec<f64> {
    margins
        .iter()
        .map(|(q, b)| source.baseline(q, *b, deadline()).unwrap())
        .collect()
}
fn checked(
    eng: &BackwardEngine<'_>,
    source: &DirectScalarHeadSource,
    q: &[I],
    b: I,
    pairs: &[Pair],
) -> Option<f64> {
    candidate(
        &mut adaptive::Contractions::new(eng, source),
        source,
        q,
        b,
        pairs,
        deadline(),
        &mut adaptive::Report::default(),
    )
}

#[test]
fn transport_five_independent_twins_beat_real_depth_four_and_authorize_instance() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&twins(5, 1.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 5], &[1.0; 5], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    assert!(root.layers.iter().any(|l| !l.unst.is_empty()));
    let source = source(&eng);
    let margins = margins(&eng, &source);
    let mut old = floors(&source, &margins);
    let old_report = adaptive::refine(
        &eng,
        &source,
        &margins,
        &mut old,
        policy(),
        deadline(),
        deadline(),
    );
    assert!(old_report.contraction_counts().0 > 0);
    // Every depth<=4 cell contains x=0 and an unfixed independent negative
    // head. Its scalar intercept is <=-1/2; the global point functional
    // lower is <=its value0 there, and all positive intercepts are <=0.
    assert!(old[0] <= -7.0 / 16.0);
    let reference = attempt_portfolio(&eng, 0, &[1, 2], policy(), deadline(), || None, false);
    assert!(!reference.prelude.unwrap().complete());
    let mut out = attempt_with_provider(&eng, 0, &[1, 2], policy(), deadline(), || {
        panic!("complete transport may not construct advice")
    });
    let prelude = out.prelude.as_ref().unwrap();
    assert!(prelude.complete());
    assert!(prelude.bounds[0] > 1.0 / 32.0 && prelude.bounds[0] <= 1.0 / 16.0);
    assert_eq!(out.transport.pairs_checked, 5);
    assert_eq!(out.transport.seed_checks, 5);
    assert_eq!(out.adaptive.leaves, vec![1, 1]);
    assert_eq!(out.adaptive.contraction_counts().0, 0);
    assert_eq!((out.calls, out.passes, out.columns), (0, 0, 0));
    for x in [-1.0, 0.0, 1.0] {
        let rho = rat(x).max(BigRational::zero());
        assert_eq!(
            rat(1.0 / 16.0) + rat(5.0) * (&rho - rho.clone()),
            rat(1.0 / 16.0)
        );
    }
    let cfg = BabConfig {
        deadline: Some(deadline()),
        head_groups: Some(policy()),
        ..BabConfig::default()
    };
    let outcome = crate::margin_row::bab::direct_root_outcome(
        &eng,
        0,
        &[1, 2],
        &cfg,
        &[],
        Some(&mut out),
        Instant::now(),
    )
    .unwrap();
    let MarginRowOutcome::Unsat(stats, token) = outcome else {
        panic!("owned full transport must reach ordinary instance authorizer")
    };
    assert_eq!(stats.root_closed_classes, 2);
    assert_eq!(stats.expansions, 0);
    assert_eq!(stats.ledger_ok, Some(true));
    assert_eq!(token.scope(), TokenScope::Instance);
    assert_eq!(token.source(), UnsatSource::Authorized);
    assert!(token.covers_all(&[1, 2]));
}

#[test]
fn transport_near_twins_with_independent_fold_and_bias_errors_remain_sound() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut model = twins(5, 1.0 / 16.0);
    let epsilon = 2.0f64.powi(-12);
    let dust = 2.0f64.powi(-16);
    if let TwinOpSpec::Gemm {
        weight,
        bias,
        bias_err,
        weight_rel_err,
        ..
    } = &mut model.ops[3]
    {
        for j in 0..5 {
            weight[(5 + j) * 6 + j] = 1.0 + epsilon;
            bias[5 + j] = -2.0 * (1.0 + epsilon) + epsilon;
        }
        *weight_rel_err = dust;
        bias_err.fill(dust);
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0; 5], &[1.0; 5], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let margins = margins(&eng, &source);
    let mut bounds = floors(&source, &margins);
    let report = refine(&eng, &source, &margins, &mut bounds, deadline(), deadline());
    assert_eq!(report.pairs_checked, 5);
    assert!(bounds[0] > 1.0 / 32.0 && bounds[0] < 1.0 / 16.0);
    for x in [-1.0, 0.0, 1.0] {
        for sign in [-1.0, 1.0] {
            // Independent worst directions, including both bias errors.
            let pos = ((rat(1.0) - rat(sign * dust)) * rat(x + 2.0) - rat(2.0) - rat(sign * dust))
                .max(BigRational::zero());
            let neg = ((rat(1.0 + epsilon) * (rat(1.0) + rat(sign * dust))) * rat(x + 2.0)
                - rat(2.0 * (1.0 + epsilon))
                + rat(epsilon)
                + rat(sign * dust))
            .max(BigRational::zero());
            assert!(rat(bounds[0]) <= rat(1.0 / 16.0) + rat(5.0) * (pos - neg));
        }
    }
}

#[test]
fn transport_full_q_uncertainty_and_overdrawn_mass_are_repriced() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut model = twins(1, 0.25);
    if let TwinOpSpec::Gemm { weight, bias, .. } = &mut model.ops[3] {
        weight.fill(0.0);
        bias.fill(1.0);
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let q = [I { lo: 0.5, hi: 1.5 }, I { lo: -1.5, hi: -0.5 }];
    for tau in [0.5, 1.0, 2.0] {
        let lower = checked(
            &eng,
            &source,
            &q,
            I::p(0.25),
            &[Pair {
                positive: 0,
                negative: 1,
                tau,
            }],
        )
        .unwrap();
        assert!(lower <= -0.75);
        for p in [q[0].lo, q[0].hi] {
            for n in [q[1].lo, q[1].hi] {
                assert!(rat(lower) <= rat(0.25) + rat(p) + rat(n));
            }
        }
    }
}

#[test]
fn transport_unequal_authored_masses_leave_the_uncancelled_negative_head() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut model = twins(1, 1.0 / 16.0);
    if let TwinOpSpec::Gemm { weight, .. } = &mut model.ops[5] {
        weight[0] = 0.5;
        weight[4] = 0.5;
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let (q, b) = source.margin(&eng, 0, deadline()).unwrap();
    let lower = checked(
        &eng,
        &source,
        &q,
        b,
        &[Pair {
            positive: 0,
            negative: 1,
            tau: 0.5,
        }],
    )
    .unwrap();
    assert!(lower <= -7.0 / 16.0);
    let mut out = attempt_with_provider(&eng, 0, &[1, 2], policy(), deadline(), || None);
    assert!(out.prelude.as_ref().unwrap().bounds[0] <= -7.0 / 16.0);
    let cfg = BabConfig {
        deadline: Some(deadline()),
        head_groups: Some(policy()),
        ..BabConfig::default()
    };
    assert!(crate::margin_row::bab::direct_root_outcome(
        &eng,
        0,
        &[1, 2],
        &cfg,
        &[],
        Some(&mut out),
        Instant::now()
    )
    .is_none());
}

#[test]
fn transport_subnormal_pair_product_is_charged_before_rounding_to_zero() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let eta = f64::from_bits(1);
    let mut model = twins(1, 0.0);
    if let TwinOpSpec::Gemm { weight, bias, .. } = &mut model.ops[3] {
        weight.fill(0.0);
        bias[0] = 0.0;
        bias[1] = eta;
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    assert_eq!((-eta * 0.5).to_bits(), (-0.0f64).to_bits());
    let lower = checked(
        &eng,
        &source,
        &[I::p(0.5), I::p(-0.5)],
        I::p(0.0),
        &[Pair {
            positive: 0,
            negative: 1,
            tau: 0.5,
        }],
    )
    .unwrap();
    assert!(rat(lower) <= -rat(eta) / rat(2.0));
    assert!(lower < 0.0);
}

#[test]
fn transport_fold_underflow_is_charged_before_large_mass_amplification() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let eta = f64::from_bits(1);
    let large = 2.0f64.powi(256);
    let mut model = twins(1, 0.0);
    if let TwinOpSpec::Gemm {
        weight,
        bias,
        weight_rel_err,
        ..
    } = &mut model.ops[3]
    {
        weight.fill(0.0);
        weight[0] = eta;
        weight[2] = eta;
        bias.fill(0.0);
        *weight_rel_err = eta;
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let lower = checked(
        &eng,
        &source,
        &[I::p(large), I::p(-large)],
        I::p(0.0),
        &[Pair {
            positive: 0,
            negative: 1,
            tau: large,
        }],
    )
    .unwrap();
    assert!(lower <= -(large * eta));
    assert!(rat(lower) <= -rat(2.0) * rat(large) * rat(eta) * rat(eta) * rat(3.0));
}

#[test]
fn transport_sampled_geometry_cannot_omit_an_unseen_feature_difference() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let inputs = 130;
    let mut model = twins(inputs, 1.0 / 16.0);
    // Keep two heads and130 original input coordinates (plus live trunk).
    let width = inputs + 1;
    let n = width.min(FUNCTIONAL_MAX_FEATURES);
    let sample: Vec<_> = (0..n).map(|i| i * (width - 1) / (n - 1)).collect();
    let omitted = (1..inputs).find(|i| !sample.contains(i)).unwrap();
    if let TwinOpSpec::Gemm {
        weight,
        bias,
        bias_err,
        shape,
        ..
    } = &mut model.ops[3]
    {
        *weight = vec![0.0; 2 * width];
        weight[0] = 1.0;
        weight[width] = 1.0;
        weight[width + omitted] = 1.0;
        *bias = vec![-2.0, -4.0];
        *bias_err = vec![0.0; 2];
        *shape = (2, width);
    }
    if let TwinOpSpec::Gemm {
        weight,
        bias_err,
        shape,
        ..
    } = &mut model.ops[5]
    {
        *weight = vec![1.0, -1.0, 0.0, 0.0, 1.0, -1.0];
        *bias_err = vec![0.0; 3];
        *shape = (3, 2);
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(
        &net,
        &vec![-1.0; inputs],
        &vec![1.0; inputs],
        RoundMode::Outward,
        None,
    )
    .unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let (q, b) = source.margin(&eng, 0, deadline()).unwrap();
    let lower = checked(
        &eng,
        &source,
        &q,
        b,
        &[Pair {
            positive: 0,
            negative: 1,
            tau: 1.0,
        }],
    )
    .unwrap();
    let mut geometry_report = Report::default();
    let geometry = Geometry::build(&eng, &source, deadline(), &mut geometry_report).unwrap();
    assert_eq!(geometry.rows[0].len(), FUNCTIONAL_MAX_FEATURES);
    assert_eq!(geometry_report.geometry_terms, 2 * FUNCTIONAL_MAX_FEATURES);
    // x0=0,x_omitted=1: relu(x0)-relu(x0+x_omitted)=-1.
    assert!(lower <= -15.0 / 16.0);
}

#[test]
fn transport_pair_and_seed_caps_keep_unpaired_debt_in_complete_bound() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&twins(MAX_PAIRS + 1, 1.0 / 16.0)).unwrap();
    let root = RootGates::build(
        &net,
        &[-1.0; MAX_PAIRS + 1],
        &[1.0; MAX_PAIRS + 1],
        RoundMode::Outward,
        None,
    )
    .unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let margins = margins(&eng, &source);
    let mut bounds = floors(&source, &margins);
    let report = refine(&eng, &source, &margins, &mut bounds, deadline(), deadline());
    assert_eq!(report.pairs_checked, MAX_PAIRS);
    assert_eq!(report.seed_checks, MAX_PAIRS);
    assert!(bounds[0] <= -15.0 / 16.0);
    assert!(bounds[1] > 0.0);
    let mut cache = adaptive::Contractions::new(&eng, &source);
    let mut work = adaptive::Report::default();
    let pairs: Vec<_> = (0..=MAX_PAIRS)
        .map(|j| Pair {
            positive: j,
            negative: MAX_PAIRS + 1 + j,
            tau: 1.0,
        })
        .collect();
    assert!(candidate(
        &mut cache,
        &source,
        &margins[0].0,
        margins[0].1,
        &pairs,
        deadline(),
        &mut work
    )
    .is_none());
    assert_eq!(work.contraction_counts().0, 0);
}

#[test]
fn transport_expired_partial_candidate_and_original_authority_publish_no_gain() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&twins(5, 1.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 5], &[1.0; 5], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let margins = margins(&eng, &source);
    let before = floors(&source, &margins);
    let mut bounds = before.clone();
    STOP_AFTER_PAIR.with(|stop| stop.set(true));
    let report = refine(&eng, &source, &margins, &mut bounds, deadline(), deadline());
    STOP_AFTER_PAIR.with(|stop| stop.set(false));
    assert!(report.seed_checks > 0);
    assert_eq!(report.candidates, 0);
    assert_eq!(bounds, before);
    for (slice, authority) in [(Instant::now(), deadline()), (deadline(), Instant::now())] {
        let report = refine(&eng, &source, &margins, &mut bounds, slice, authority);
        assert_eq!(report.seed_checks, 0);
        assert_eq!(bounds, before);
    }
}

#[test]
fn transport_identical_bytes_foreign_source_generation_and_class_order_decline() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&twins(1, 1.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let margins = margins(&eng, &source);
    let before = floors(&source, &margins);
    let rebuilt = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let foreign = BackwardEngine::new(&net, &rebuilt);
    assert!(checked(
        &foreign,
        &source,
        &margins[0].0,
        margins[0].1,
        &[Pair {
            positive: 0,
            negative: 1,
            tau: 1.0
        }]
    )
    .is_none());
    let mut bounds = before.clone();
    let report = refine(
        &foreign,
        &source,
        &margins,
        &mut bounds,
        deadline(),
        deadline(),
    );
    assert_eq!(report.seed_checks, 0);
    assert_eq!(bounds, before);
    let out = attempt_with_provider(&eng, 0, &[1, 2], policy(), deadline(), || None);
    let prelude = out.prelude.unwrap();
    assert!(!prelude.matches(&eng, 0, &[2, 1], deadline()));
    assert!(!prelude.matches(&foreign, 0, &[1, 2], deadline()));
}

#[test]
fn transport_zero_tie_and_negative_bias_never_authorize_unsat() {
    let _capture = HeadGroupSourceGuard::enter(true);
    for b in [0.0, -1.0 / 16.0] {
        let net = TwinNet::compile(&twins(5, b)).unwrap();
        let root = RootGates::build(&net, &[-1.0; 5], &[1.0; 5], RoundMode::Outward, None).unwrap();
        let eng = BackwardEngine::new(&net, &root);
        let mut out = attempt_with_provider(&eng, 0, &[1, 2], policy(), deadline(), || None);
        assert!(out.transport.pairs_checked > 0);
        assert!(out.prelude.as_ref().unwrap().bounds[0] <= b);
        let cfg = BabConfig {
            deadline: Some(deadline()),
            head_groups: Some(policy()),
            ..BabConfig::default()
        };
        assert!(crate::margin_row::bab::direct_root_outcome(
            &eng,
            0,
            &[1, 2],
            &cfg,
            &[],
            Some(&mut out),
            Instant::now()
        )
        .is_none());
    }
}

#[test]
fn transport_multiclass_cache_reprices_own_mass_bias_and_original_slots() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut model = twins(2, 1.0 / 16.0);
    let q = [1.0, 1.0, -1.0, -1.0];
    if let TwinOpSpec::Gemm {
        weight,
        bias,
        bias_err,
        shape,
        ..
    } = &mut model.ops[5]
    {
        *weight = q
            .iter()
            .copied()
            .chain(vec![0.0; 4])
            .chain(q.iter().copied())
            .chain(q.iter().copied())
            .chain(q.iter().map(|q| q * 0.5))
            .collect();
        *bias = vec![1.0 / 16.0, 0.0, -15.0 / 16.0, 17.0 / 16.0, 1.0 / 16.0];
        *bias_err = vec![0.0; 5];
        *shape = (5, 4);
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0; 2], &[1.0; 2], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = DirectScalarHeadSource::build(&eng, 0, &[4, 2, 1, 3], deadline()).unwrap();
    let margins = margins(&eng, &source);
    let mut bounds = floors(&source, &margins);
    let report = refine(&eng, &source, &margins, &mut bounds, deadline(), deadline());
    assert_eq!(report.seed_checks, 2);
    assert_eq!(report.cache_hits, 2);
    assert!(bounds[0] <= 0.0);
    assert!(bounds[1] > 0.5 && bounds[1] <= 1.0);
    assert!(bounds[2] > 1.0 / 32.0 && bounds[2] <= 1.0 / 16.0);
    assert!(bounds[3] <= -1.0);
    assert_eq!(source.adversaries, vec![4, 2, 1, 3]);
}

#[test]
fn transport_invalid_or_reused_mass_indices_refuse_before_contraction() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&twins(1, 1.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let (q, b) = source.margin(&eng, 0, deadline()).unwrap();
    for pairs in [
        vec![Pair {
            positive: 0,
            negative: 1,
            tau: -1.0,
        }],
        vec![Pair {
            positive: 0,
            negative: 1,
            tau: f64::NAN,
        }],
        vec![Pair {
            positive: 0,
            negative: 2,
            tau: 1.0,
        }],
        vec![Pair {
            positive: 0,
            negative: 0,
            tau: 1.0,
        }],
        vec![
            Pair {
                positive: 0,
                negative: 1,
                tau: 1.0
            };
            2
        ],
    ] {
        let mut work = adaptive::Report::default();
        assert!(candidate(
            &mut adaptive::Contractions::new(&eng, &source),
            &source,
            &q,
            b,
            &pairs,
            deadline(),
            &mut work
        )
        .is_none());
        assert_eq!(work.contraction_counts().0, 0);
    }
    let zero = checked(
        &eng,
        &source,
        &q,
        b,
        &[Pair {
            positive: 0,
            negative: 1,
            tau: 0.0,
        }],
    )
    .unwrap();
    assert!(zero <= source.baseline(&q, b, deadline()).unwrap());
}

#[test]
fn transport_closed_form_signed_support_encloses_all_activation_quadrants() {
    for (p, n) in [
        (I { lo: -2.0, hi: 3.0 }, I { lo: -1.0, hi: 4.0 }),
        (I { lo: 1.0, hi: 3.0 }, I { lo: 0.0, hi: 2.0 }),
        (I { lo: -2.0, hi: -1.0 }, I { lo: -4.0, hi: -3.0 }),
    ] {
        for delta in [-6.0, -1.0, 0.0, 1.0] {
            let lower = difference_floor(p, n, delta).unwrap();
            for a in [p.lo, p.hi, 0.0] {
                for b in [n.lo, n.hi, 0.0] {
                    if a >= p.lo && a <= p.hi && b >= n.lo && b <= n.hi && a - b >= delta {
                        assert!(rat(lower) <= rat(a.max(0.0)) - rat(b.max(0.0)));
                    }
                }
            }
        }
    }
    assert_eq!(difference_floor(I::p(0.0), I::p(1.0), -1.0), Some(-1.0));
}

// Shared mass fixtures have original inputs and three original outputs. Every
// star is one positive mass2 and two negative mass1 heads of the same x_i.
fn fractional_stars(stars: usize, b: f64) -> TwinSpec {
    let heads = 3 * stars;
    let mut trunk = vec![0.0; stars * stars];
    let mut first = vec![0.0; heads * stars];
    for i in 0..stars {
        trunk[i * stars + i] = 1.0;
        for j in [i, stars + i, 2 * stars + i] {
            first[j * stars + i] = 1.0;
        }
    }
    let q: Vec<_> = (0..heads)
        .map(|j| if j < stars { 2.0 } else { -1.0 })
        .collect();
    TwinSpec {
        n_in: stars,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: trunk,
                bias: vec![2.0; stars],
                bias_err: vec![0.0; stars],
                weight_rel_err: 0.0,
                kernel: (stars, stars, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (stars, 1, 1),
                oshape: (stars, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: first,
                bias: vec![-2.0; heads],
                bias_err: vec![0.0; heads],
                weight_rel_err: 0.0,
                shape: (heads, stars),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: q
                    .iter()
                    .copied()
                    .chain(vec![0.0; heads])
                    .chain(q.iter().copied())
                    .collect(),
                bias: vec![b, 0.0, b - 1.0],
                bias_err: vec![0.0; 3],
                weight_rel_err: 0.0,
                shape: (3, heads),
            },
        ],
    }
}

fn star_pairs(stars: usize, tau: f64) -> Vec<Pair> {
    (0..stars)
        .flat_map(|i| {
            [
                Pair {
                    positive: i,
                    negative: stars + i,
                    tau,
                },
                Pair {
                    positive: i,
                    negative: 2 * stars + i,
                    tau,
                },
            ]
        })
        .collect()
}

#[test]
fn fractional_five_stars_beat_actual_disjoint_and_depth_four_and_authorize_instance() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&fractional_stars(5, 1.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 5], &[1.0; 5], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let margins = margins(&eng, &source);
    let mut geometry_report = Report::default();
    let geometry = Geometry::build(&eng, &source, deadline(), &mut geometry_report).unwrap();
    let disjoint = geometry
        .propose(&source, &margins[0].0, deadline(), &mut geometry_report)
        .unwrap();
    assert_eq!(disjoint.len(), 5);
    let old = checked(&eng, &source, &margins[0].0, margins[0].1, &disjoint).unwrap();
    assert!(old <= 1.0 / 16.0 - 5.0);
    let mut scalar_phase = floors(&source, &margins);
    let old_report = adaptive::refine(
        &eng,
        &source,
        &margins,
        &mut scalar_phase,
        policy(),
        deadline(),
        deadline(),
    );
    assert!(old_report.contraction_counts().0 > 0);
    assert!(scalar_phase[0] <= 1.0 / 16.0 - 3.0);
    // Exact dyadic authored margin: each independent star cancels at every x.
    for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
        let rho = rat(x).max(BigRational::zero());
        assert_eq!(
            rat(1.0 / 16.0) + rat(5.0) * (rat(2.0) * &rho - &rho - &rho),
            rat(1.0 / 16.0)
        );
    }
    let mut out = attempt_with_provider(&eng, 0, &[1, 2], policy(), deadline(), || {
        panic!("complete fractional transport may not request advice")
    });
    let prelude = out.prelude.as_ref().unwrap();
    assert!(prelude.complete());
    assert!(prelude.bounds[0] > 1.0 / 32.0 && prelude.bounds[0] <= 1.0 / 16.0);
    assert_eq!(out.transport.fractional_candidates, 1);
    assert_eq!(out.transport.fractional_edges, 10);
    assert_eq!(out.transport.fractional_shared_endpoints, 5);
    assert_eq!(out.transport.seed_checks, 10);
    assert_eq!(out.transport.fractional_seed_checks, 5);
    assert_eq!(out.transport.fractional_cache_hits, 5);
    assert_eq!(out.transport.classes_improved, 1);
    assert_eq!(out.transport.fractional_classes_improved, 1);
    assert_eq!(out.adaptive.leaves, vec![1, 1]);
    assert_eq!(out.adaptive.contraction_counts().0, 0);
    assert_eq!((out.calls, out.passes, out.columns), (0, 0, 0));
    let cfg = BabConfig {
        deadline: Some(deadline()),
        head_groups: Some(policy()),
        ..BabConfig::default()
    };
    let outcome = crate::margin_row::bab::direct_root_outcome(
        &eng,
        0,
        &[1, 2],
        &cfg,
        &[],
        Some(&mut out),
        Instant::now(),
    )
    .unwrap();
    let MarginRowOutcome::Unsat(stats, token) = outcome else {
        panic!("full fractional cover must reach instance authorizer")
    };
    assert_eq!(stats.root_closed_classes, 2);
    assert_eq!(stats.expansions, 0);
    assert_eq!(stats.ledger_ok, Some(true));
    assert_eq!(token.scope(), TokenScope::Instance);
    assert_eq!(token.source(), UnsatSource::Authorized);
    assert!(token.covers_all(&[1, 2]));
}

#[test]
fn fractional_shared_negative_and_mixed_endpoint_roles_retain_complete_residuals() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&fractional_stars(1, 1.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let negative_shared = [
        Pair {
            positive: 0,
            negative: 2,
            tau: 1.0,
        },
        Pair {
            positive: 1,
            negative: 2,
            tau: 1.0,
        },
    ];
    let lower = checked(
        &eng,
        &source,
        &[I::p(1.0), I::p(1.0), I::p(-2.0)],
        I::p(1.0 / 16.0),
        &negative_shared,
    )
    .unwrap();
    assert!(lower > 1.0 / 32.0 && lower <= 1.0 / 16.0);
    let mixed = [
        Pair {
            positive: 0,
            negative: 1,
            tau: 1.0,
        },
        Pair {
            positive: 1,
            negative: 2,
            tau: 1.0,
        },
    ];
    let q = [I::p(2.0), I::p(-1.0), I::p(-1.0)];
    let lower = checked(&eng, &source, &q, I::p(1.0 / 16.0), &mixed).unwrap();
    // The middle endpoint has both roles; its negative residual cannot vanish.
    assert!(lower <= -15.0 / 16.0);
    for x in [-1.0, 0.0, 1.0] {
        let rho = rat(x).max(BigRational::zero());
        assert!(rat(lower) <= rat(1.0 / 16.0) + (rat(2.0) - rat(1.0) - rat(1.0)) * rho);
    }
}

#[test]
fn fractional_uncertain_and_overdrawn_shared_mass_reprices_every_head() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut model = fractional_stars(1, 0.25);
    if let TwinOpSpec::Gemm { weight, bias, .. } = &mut model.ops[3] {
        weight.fill(0.0);
        bias.fill(1.0);
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let q = [
        I { lo: 1.5, hi: 2.5 },
        I { lo: -1.5, hi: -0.5 },
        I { lo: -1.5, hi: 0.5 },
    ];
    for tau in [0.5, 1.0, 2.0, 4.0] {
        let lower = checked(&eng, &source, &q, I::p(0.25), &star_pairs(1, tau)).unwrap();
        assert!(lower <= -1.25);
        for p in [q[0].lo, q[0].hi] {
            for n1 in [q[1].lo, q[1].hi] {
                for n2 in [q[2].lo, q[2].hi] {
                    assert!(rat(lower) <= rat(0.25) + rat(p) + rat(n1) + rat(n2));
                }
            }
        }
    }
    // Exact zero masses make no contractions and retain the original full Q.
    let mut work = adaptive::Report::default();
    let zero = candidate(
        &mut adaptive::Contractions::new(&eng, &source),
        &source,
        &q,
        I::p(0.25),
        &star_pairs(1, 0.0),
        deadline(),
        &mut work,
    )
    .unwrap();
    assert_eq!(work.contraction_counts().0, 0);
    assert!(zero <= source.baseline(&q, I::p(0.25), deadline()).unwrap());
}

#[test]
fn fractional_sequential_mass_rounding_cannot_erase_exact_negative_residual() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut model = fractional_stars(1, 0.0);
    if let TwinOpSpec::Gemm { weight, bias, .. } = &mut model.ops[3] {
        weight.fill(0.0);
        bias.fill(1.0);
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let exact = rat(0.3) - rat(0.1) - rat(0.2);
    assert!(exact < BigRational::zero());
    let lower = checked(
        &eng,
        &source,
        &[I::p(0.3), I::p(-0.1), I::p(-0.2)],
        I::p(0.0),
        &[
            Pair {
                positive: 0,
                negative: 1,
                tau: 0.1,
            },
            Pair {
                positive: 0,
                negative: 2,
                tau: 0.2,
            },
        ],
    )
    .unwrap();
    assert!(lower < 0.0);
    assert!(rat(lower) <= exact);
}

#[test]
fn fractional_independent_fold_and_bias_uncertainty_survives_shared_rows() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let dust = 2.0f64.powi(-16);
    let mut model = fractional_stars(1, 1.0 / 16.0);
    if let TwinOpSpec::Gemm {
        weight_rel_err,
        bias_err,
        ..
    } = &mut model.ops[3]
    {
        *weight_rel_err = dust;
        bias_err.fill(dust);
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let (q, b) = source.margin(&eng, 0, deadline()).unwrap();
    let lower = checked(&eng, &source, &q, b, &star_pairs(1, 1.0)).unwrap();
    assert!(lower > 1.0 / 32.0 && lower < 1.0 / 16.0);
    for x in [-1.0, 0.0, 1.0] {
        for direction in [-1.0, 1.0] {
            let pos = ((rat(1.0) - rat(direction * dust)) * rat(x + 2.0)
                - rat(2.0)
                - rat(direction * dust))
            .max(BigRational::zero());
            let neg = ((rat(1.0) + rat(direction * dust)) * rat(x + 2.0) - rat(2.0)
                + rat(direction * dust))
            .max(BigRational::zero());
            assert!(rat(lower) <= rat(1.0 / 16.0) + rat(2.0) * pos - rat(2.0) * neg);
        }
    }
}

#[test]
fn fractional_underflow_and_large_shared_mass_keep_absolute_fold_charge() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let eta = f64::from_bits(1);
    let large = 2.0f64.powi(256);
    let mut model = fractional_stars(1, 0.0);
    if let TwinOpSpec::Gemm {
        weight,
        bias,
        weight_rel_err,
        ..
    } = &mut model.ops[3]
    {
        weight.fill(eta);
        bias.fill(0.0);
        *weight_rel_err = eta;
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let lower = checked(
        &eng,
        &source,
        &[I::p(large), I::p(-large / 2.0), I::p(-large / 2.0)],
        I::p(0.0),
        &star_pairs(1, large / 2.0),
    )
    .unwrap();
    assert!(lower <= -(large * eta));
    assert!(rat(lower) <= -rat(2.0) * rat(large) * rat(eta) * rat(eta) * rat(3.0));
    let mut product_model = fractional_stars(1, 0.0);
    if let TwinOpSpec::Gemm { weight, bias, .. } = &mut product_model.ops[3] {
        weight.fill(0.0);
        *bias = vec![0.0, eta, eta];
    }
    let product_net = TwinNet::compile(&product_model).unwrap();
    let product_root =
        RootGates::build(&product_net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let product_eng = BackwardEngine::new(&product_net, &product_root);
    let product_source = self::source(&product_eng);
    let lower = checked(
        &product_eng,
        &product_source,
        &[I::p(1.0), I::p(-0.5), I::p(-0.5)],
        I::p(0.0),
        &star_pairs(1, 0.5),
    )
    .unwrap();
    assert!(lower < 0.0 && rat(lower) <= -rat(eta));
}

#[test]
fn fractional_all_disjoint_recipients_run_before_new_seeds_and_keep_completed_floors() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let stars = 9;
    let heads = 3 * stars;
    let mut model = fractional_stars(stars, 1.0 / 16.0);
    let q: Vec<_> = (0..heads)
        .map(|j| if j < stars { 2.0 } else { -1.0 })
        .collect();
    let mut g = vec![0.0; heads];
    g[stars - 1] = 1.0;
    g[heads - 1] = -1.0;
    if let TwinOpSpec::Gemm {
        weight,
        bias,
        bias_err,
        shape,
        ..
    } = &mut model.ops[5]
    {
        *weight = q
            .iter()
            .copied()
            .chain(vec![0.0; heads])
            .chain(q.iter().copied())
            .chain(q.iter().zip(&g).map(|(q, g)| q - g))
            .collect();
        *bias = vec![1.0 / 16.0, 0.0, -15.0 / 16.0, 0.0];
        *bias_err = vec![0.0; 4];
        *shape = (4, heads);
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0; 9], &[1.0; 9], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = DirectScalarHeadSource::build(&eng, 0, &[1, 3, 2], deadline()).unwrap();
    let margins = margins(&eng, &source);
    let mut bounds = floors(&source, &margins);
    let mut geometry_report = Report::default();
    let geometry = Geometry::build(&eng, &source, deadline(), &mut geometry_report).unwrap();
    let old_pairs = geometry
        .propose(&source, &margins[0].0, deadline(), &mut geometry_report)
        .unwrap();
    let old = checked(&eng, &source, &margins[0].0, margins[0].1, &old_pairs).unwrap();
    let report = refine(&eng, &source, &margins, &mut bounds, deadline(), deadline());
    assert_eq!(report.seed_checks, MAX_PAIRS);
    assert_eq!(report.fractional_candidates, 0);
    assert_eq!(bounds[0].to_bits(), old.to_bits());
    // This later old candidate needs a seed absent from the first row's old
    // matching. It closes before new shared seeds exhaust the original cap.
    assert!(bounds[1] > 1.0 / 32.0 && bounds[1] <= 1.0 / 16.0);
    assert!(bounds[2] > 0.0);
    assert_eq!(source.adversaries, vec![1, 3, 2]);
}

#[test]
fn fractional_local_expiry_retains_disjoint_floors_but_original_expiry_restores_inputs() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&fractional_stars(5, 1.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 5], &[1.0; 5], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let margins = margins(&eng, &source);
    let before = floors(&source, &margins);
    let mut geometry_report = Report::default();
    let geometry = Geometry::build(&eng, &source, deadline(), &mut geometry_report).unwrap();
    let pairs = geometry
        .propose(&source, &margins[0].0, deadline(), &mut geometry_report)
        .unwrap();
    let old = checked(&eng, &source, &margins[0].0, margins[0].1, &pairs).unwrap();
    let mut bounds = before.clone();
    STOP_AFTER_DISJOINT.with(|stop| stop.set(1));
    let report = refine(&eng, &source, &margins, &mut bounds, deadline(), deadline());
    assert!(report.slice_expired && !report.authority_expired);
    assert_eq!(bounds[0].to_bits(), old.to_bits());
    assert!(bounds[0] > before[0]);
    assert_eq!(report.fractional_seed_checks, 0);
    // Expire a partial shared candidate only after its first NEW contraction,
    // rather than before entry or on a reused old edge. Its full floor cannot
    // publish, and the already completed disjoint certificate stays intact.
    bounds.copy_from_slice(&before);
    STOP_AFTER_DISJOINT.with(|stop| stop.set(3));
    let report = refine(&eng, &source, &margins, &mut bounds, deadline(), deadline());
    STOP_AFTER_NEW_PAIR.with(|stop| stop.set(false));
    assert_eq!(report.fractional_seed_checks, 1);
    assert_eq!(report.fractional_candidates, 0);
    assert_eq!(bounds[0].to_bits(), old.to_bits());
    bounds.copy_from_slice(&before);
    STOP_AFTER_DISJOINT.with(|stop| stop.set(2));
    let report = refine(&eng, &source, &margins, &mut bounds, deadline(), deadline());
    assert!(report.authority_expired);
    assert_eq!(report.classes_improved, 0);
    assert_eq!(report.fractional_classes_improved, 0);
    assert_eq!(bounds, before);
}

#[test]
fn fractional_multiclass_own_q_bias_order_and_negative_recipient_prevent_instance_token() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut model = fractional_stars(2, 1.0 / 16.0);
    let q = [2.0, 2.0, -1.0, -1.0, -1.0, -1.0];
    if let TwinOpSpec::Gemm {
        weight,
        bias,
        bias_err,
        shape,
        ..
    } = &mut model.ops[5]
    {
        *weight = q
            .iter()
            .copied()
            .chain([0.0; 6])
            .chain(q.iter().copied())
            .chain(q.iter().map(|q| -q))
            .chain(q.iter().map(|q| q * 0.5))
            .collect();
        *bias = vec![1.0 / 16.0, 0.0, -15.0 / 16.0, -1.0 / 16.0, 3.0 / 32.0];
        *bias_err = vec![0.0; 5];
        *shape = (5, 6);
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0; 2], &[1.0; 2], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = DirectScalarHeadSource::build(&eng, 0, &[3, 4, 1, 2], deadline()).unwrap();
    let margins = margins(&eng, &source);
    let mut bounds = floors(&source, &margins);
    let report = refine(&eng, &source, &margins, &mut bounds, deadline(), deadline());
    assert_eq!(report.seed_checks, 4);
    assert_eq!(report.fractional_candidates, 3);
    assert!(bounds[0] > 1.0 / 16.0 && bounds[0] <= 1.0 / 8.0);
    assert!(bounds[1] <= -1.0 / 32.0);
    assert!(bounds[2] > 1.0 / 32.0 && bounds[2] <= 1.0 / 16.0);
    assert!(bounds[3] > 0.0);
    assert_eq!(source.adversaries, vec![3, 4, 1, 2]);
    let mut out = attempt_with_provider(&eng, 0, &[3, 4, 1, 2], policy(), deadline(), || None);
    assert!(!out.prelude.as_ref().unwrap().complete());
    let cfg = BabConfig {
        deadline: Some(deadline()),
        head_groups: Some(policy()),
        ..BabConfig::default()
    };
    assert!(crate::margin_row::bab::direct_root_outcome(
        &eng,
        0,
        &[3, 4, 1, 2],
        &cfg,
        &[],
        Some(&mut out),
        Instant::now()
    )
    .is_none());
}

#[test]
fn fractional_nonpositive_unequal_and_missing_mass_never_authorize_instance() {
    let _capture = HeadGroupSourceGuard::enter(true);
    for (b, positive, negative2, exact_min) in [
        (0.0, 2.0, -1.0, 0.0),
        (-1.0 / 16.0, 2.0, -1.0, -1.0 / 16.0),
        (1.0 / 16.0, 1.5, -1.0, -7.0 / 16.0),
        (1.0 / 16.0, 2.0, -1.5, -7.0 / 16.0),
    ] {
        let mut model = fractional_stars(1, b);
        if let TwinOpSpec::Gemm { weight, .. } = &mut model.ops[5] {
            weight[0] = positive;
            weight[2] = negative2;
            weight[6] = positive;
            weight[8] = negative2;
        }
        let net = TwinNet::compile(&model).unwrap();
        let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
        let eng = BackwardEngine::new(&net, &root);
        let mut out = attempt_with_provider(&eng, 0, &[1, 2], policy(), deadline(), || None);
        assert!(out.transport.fractional_candidates > 0);
        assert!(out.prelude.as_ref().unwrap().bounds[0] <= exact_min);
        let cfg = BabConfig {
            deadline: Some(deadline()),
            head_groups: Some(policy()),
            ..BabConfig::default()
        };
        assert!(crate::margin_row::bab::direct_root_outcome(
            &eng,
            0,
            &[1, 2],
            &cfg,
            &[],
            Some(&mut out),
            Instant::now()
        )
        .is_none());
    }
}

#[test]
fn fractional_foreign_generation_invalid_edges_and_magnitude_refuse_before_publication() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&fractional_stars(1, 1.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let margins = margins(&eng, &source);
    let rebuilt = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let foreign = BackwardEngine::new(&net, &rebuilt);
    assert!(checked(
        &foreign,
        &source,
        &margins[0].0,
        margins[0].1,
        &star_pairs(1, 1.0)
    )
    .is_none());
    for invalid in [f64::NAN, f64::INFINITY, -1.0] {
        let mut pairs = star_pairs(1, 1.0);
        pairs[1].tau = invalid;
        let mut work = adaptive::Report::default();
        assert!(candidate(
            &mut adaptive::Contractions::new(&eng, &source),
            &source,
            &margins[0].0,
            margins[0].1,
            &pairs,
            deadline(),
            &mut work
        )
        .is_none());
        assert_eq!(work.contraction_counts().0, 0);
    }
    let mut duplicate = star_pairs(1, 1.0);
    duplicate.push(duplicate[0]);
    let mut work = adaptive::Report::default();
    assert!(candidate(
        &mut adaptive::Contractions::new(&eng, &source),
        &source,
        &margins[0].0,
        margins[0].1,
        &duplicate,
        deadline(),
        &mut work
    )
    .is_none());
    assert_eq!(work.contraction_counts().0, 0);
    let mut bounds = floors(&source, &margins);
    let before = bounds.clone();
    let report = refine(
        &foreign,
        &source,
        &margins,
        &mut bounds,
        deadline(),
        deadline(),
    );
    assert_eq!(report.seed_checks, 0);
    assert_eq!(bounds, before);
    // A mass inside the inherited magnitude cap can create a residual outside
    // that cap. The whole candidate declines rather than publishing a prefix.
    assert!(good(MAX_MAG));
    assert!(checked(
        &eng,
        &source,
        &margins[0].0,
        I::p(0.0),
        &star_pairs(1, MAX_MAG)
    )
    .is_none());
    let out = attempt_with_provider(&eng, 0, &[1, 2], policy(), deadline(), || None);
    let prelude = out.prelude.unwrap();
    assert!(!prelude.matches(&eng, 0, &[2, 1], deadline()));
    assert!(!prelude.matches(&foreign, 0, &[1, 2], deadline()));
}

// Five genuinely distinct affine combinations; every block has TWO original
// inputs and three distinct head rows, rather than duplicate-row cancellation.
fn jensen_blocks(blocks: usize, b: f64) -> TwinSpec {
    let width = 2 * blocks;
    let heads = 3 * blocks;
    let mut trunk = vec![0.0; width * width];
    let mut first = vec![0.0; heads * width];
    for k in 0..width {
        trunk[k * width + k] = 1.0;
    }
    for i in 0..blocks {
        first[(2 * i) * width + 2 * i] = 1.0;
        first[(2 * i + 1) * width + 2 * i + 1] = 1.0;
        first[(2 * blocks + i) * width + 2 * i] = 0.5;
        first[(2 * blocks + i) * width + 2 * i + 1] = 0.5;
    }
    let q: Vec<_> = (0..heads)
        .map(|j| if j < 2 * blocks { 0.5 } else { -1.0 })
        .collect();
    TwinSpec {
        n_in: width,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: trunk,
                bias: vec![2.0; width],
                bias_err: vec![0.0; width],
                weight_rel_err: 0.0,
                kernel: (width, width, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (width, 1, 1),
                oshape: (width, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: first,
                bias: vec![-2.0; heads],
                bias_err: vec![0.0; heads],
                weight_rel_err: 0.0,
                shape: (heads, width),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: q
                    .iter()
                    .copied()
                    .chain(vec![0.0; heads])
                    .chain(q.iter().copied())
                    .collect(),
                bias: vec![b, 0.0, b - 1.0],
                bias_err: vec![0.0; 3],
                weight_rel_err: 0.0,
                shape: (3, heads),
            },
        ],
    }
}

fn hyperedges(blocks: usize, tau: f64) -> Vec<Hyperedge> {
    (0..blocks)
        .map(|i| Hyperedge {
            positive: vec![(2 * i, 0.5), (2 * i + 1, 0.5)],
            negative: 2 * blocks + i,
            tau,
        })
        .collect()
}
fn checked_jensen(
    eng: &BackwardEngine<'_>,
    source: &DirectScalarHeadSource,
    q: &[I],
    b: I,
    edges: &[Hyperedge],
) -> Option<f64> {
    jensen_candidate(
        &mut adaptive::Contractions::new(eng, source),
        source,
        q,
        b,
        edges,
        deadline(),
        &mut adaptive::Report::default(),
    )
}

#[test]
fn jensen_five_distinct_blocks_beat_actual_pair_shared_depth_four_and_authorize_instance() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&jensen_blocks(5, 1.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 10], &[1.0; 10], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let margins = margins(&eng, &source);
    let baseline = floors(&source, &margins);
    let mut old = baseline.clone();
    let reference = refine_portfolio(
        &eng,
        &source,
        &margins,
        &mut old,
        deadline(),
        deadline(),
        false,
    );
    assert_eq!(reference.seed_checks, 10);
    assert_eq!(reference.jensen_seed_checks, 0);
    assert_eq!(reference.fractional_candidates, 1);
    assert!(old[0] <= 1.0 / 16.0 - 5.0);
    let mut old_phase = baseline;
    let phase = adaptive::refine(
        &eng,
        &source,
        &margins,
        &mut old_phase,
        policy(),
        deadline(),
        deadline(),
    );
    assert!(phase.contraction_counts().0 > 0);
    assert!(old_phase[0] <= -7.0 / 16.0);
    // rho(x)+rho(z)>=rho(x+z): dyadic exact source margin is nonnegative
    // above the authored1/16, with equality at all-zero original inputs.
    for x in [-1.0f64, 0.0, 1.0] {
        for z in [-1.0f64, 0.0, 1.0] {
            let j = rat(0.5) * rat(x.max(0.0)) + rat(0.5) * rat(z.max(0.0))
                - rat((0.5 * x + 0.5 * z).max(0.0));
            assert!(j >= BigRational::zero());
        }
    }
    let mut out = attempt_with_provider(&eng, 0, &[1, 2], policy(), deadline(), || {
        panic!("complete Jensen may not request advice")
    });
    assert!(out.prelude.as_ref().unwrap().complete());
    let lower = out.prelude.as_ref().unwrap().bounds[0];
    assert!(lower > 1.0 / 32.0 && lower <= 1.0 / 16.0);
    assert_eq!(out.transport.seed_checks, 15);
    assert_eq!(out.transport.jensen_seed_checks, 5);
    assert_eq!(out.transport.jensen_hyperedges, 5);
    assert_eq!(out.transport.jensen_candidates, 1);
    assert_eq!(out.transport.jensen_classes_improved, 1);
    assert!(out.transport.jensen_rank_trials <= MAX_JENSEN_RANK_TRIALS);
    assert_eq!((out.calls, out.passes, out.columns), (0, 0, 0));
    assert_eq!(out.adaptive.contraction_counts().0, 0);
    let cfg = BabConfig {
        deadline: Some(deadline()),
        head_groups: Some(policy()),
        ..BabConfig::default()
    };
    let result = crate::margin_row::bab::direct_root_outcome(
        &eng,
        0,
        &[1, 2],
        &cfg,
        &[],
        Some(&mut out),
        Instant::now(),
    )
    .unwrap();
    let MarginRowOutcome::Unsat(stats, token) = result else {
        panic!("real original Instance token required")
    };
    assert_eq!(stats.root_closed_classes, 2);
    assert_eq!(stats.expansions, 0);
    assert_eq!(stats.ledger_ok, Some(true));
    assert_eq!(token.scope(), TokenScope::Instance);
    assert_eq!(token.source(), UnsatSource::Authorized);
    assert!(token.covers_all(&[1, 2]));
}

#[test]
fn jensen_normalized_seed_reuses_across_scaled_reordered_original_recipients() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut model = jensen_blocks(5, 1.0 / 16.0);
    let q: Vec<_> = (0..15).map(|j| if j < 10 { 0.5 } else { -1.0 }).collect();
    if let TwinOpSpec::Gemm {
        weight,
        bias,
        bias_err,
        shape,
        ..
    } = &mut model.ops[5]
    {
        *weight = q
            .iter()
            .copied()
            .chain([0.0; 15])
            .chain(q.iter().map(|q| -q))
            .chain(q.iter().copied())
            .collect();
        *bias = vec![1.0 / 16.0, 0.0, 0.0, -15.0 / 16.0];
        *bias_err = vec![0.0; 4];
        *shape = (4, 15);
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0; 10], &[1.0; 10], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = DirectScalarHeadSource::build(&eng, 0, &[2, 1, 3], deadline()).unwrap();
    let margins = margins(&eng, &source);
    let mut bounds = floors(&source, &margins);
    let report = refine(&eng, &source, &margins, &mut bounds, deadline(), deadline());
    assert_eq!(source.adversaries, vec![2, 1, 3]);
    assert_eq!(report.seed_checks, 15);
    assert_eq!(report.jensen_seed_checks, 5);
    assert_eq!(report.jensen_cache_hits, 5);
    assert_eq!(report.jensen_candidates, 2);
    assert_eq!(report.jensen_classes_improved, 2);
    assert!(bounds[0] > 1.0 / 32.0 && bounds[0] <= 1.0 / 16.0);
    assert!(bounds[1] > 1.0 / 32.0 && bounds[1] <= 1.0 / 16.0);
    assert!(bounds[2] > 0.5 && bounds[2] <= 1.0);
}

#[test]
fn jensen_full_q_bias_uncertainty_nonunit_weights_and_overdraw_reprice_every_head() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut model = jensen_blocks(1, 0.0);
    if let TwinOpSpec::Gemm { weight, bias, .. } = &mut model.ops[3] {
        weight.fill(0.0);
        bias.fill(1.0);
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0; 2], &[1.0; 2], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let q = [
        I { lo: 0.25, hi: 0.75 },
        I { lo: 0.25, hi: 0.75 },
        I { lo: -1.5, hi: -0.5 },
    ];
    let b = I {
        lo: -0.25,
        hi: 0.25,
    };
    for tau in [0.5, 1.0, 4.0] {
        let lower = checked_jensen(
            &eng,
            &source,
            &q,
            b,
            &[Hyperedge {
                positive: vec![(0, 0.1), (1, 0.2)],
                negative: 2,
                tau,
            }],
        )
        .unwrap();
        assert!(lower <= -1.25);
        for p in [q[0].lo, q[0].hi] {
            for r in [q[1].lo, q[1].hi] {
                for n in [q[2].lo, q[2].hi] {
                    assert!(rat(lower) <= rat(b.lo) + rat(p) + rat(r) + rat(n));
                }
            }
        }
    }
}

#[test]
fn jensen_rounded_recipient_mass_product_is_not_exact_certificate_authority() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut model = jensen_blocks(1, 0.0);
    if let TwinOpSpec::Gemm { weight, bias, .. } = &mut model.ops[3] {
        weight.fill(0.0);
        bias.fill(1.0);
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0; 2], &[1.0; 2], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let q = [I::p(0.1), I::p(0.2), I::p(-0.30000000000000004)];
    let exact = rat(0.1) + rat(0.2) - rat(0.30000000000000004);
    assert!(exact < BigRational::zero());
    let lower = checked_jensen(
        &eng,
        &source,
        &q,
        I::p(0.0),
        &[Hyperedge {
            positive: vec![(0, 1.0 / 3.0), (1, 2.0 / 3.0)],
            negative: 2,
            tau: 0.3,
        }],
    )
    .unwrap();
    assert!(lower < 0.0 && rat(lower) <= exact);
}

#[test]
fn jensen_subnormal_mass_product_charge_precedes_large_head_amplification() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let eta = f64::from_bits(1);
    let large = 2.0f64.powi(256);
    assert_eq!(0.5 * eta, 0.0);
    let mut model = jensen_blocks(1, 0.0);
    if let TwinOpSpec::Gemm { weight, bias, .. } = &mut model.ops[3] {
        weight.fill(0.0);
        *bias = vec![large, 0.0, 0.0];
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0; 2], &[1.0; 2], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let lower = checked_jensen(
        &eng,
        &source,
        &[I::p(0.0); 3],
        I::p(0.0),
        &[Hyperedge {
            positive: vec![(0, eta), (1, 0.0)],
            negative: 2,
            tau: 0.5,
        }],
    )
    .unwrap();
    // The original margin is exactly zero. RN(tau*lambda)==0 would retain a
    // positive scaled floor without its exact negative residual cancellation.
    assert!(lower <= -large * eta / 4.0);
    assert!(rat(lower) <= BigRational::zero());
}

#[test]
fn jensen_whole_fold_subnormal_error_survives_large_recipient_scale() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let eta = f64::from_bits(1);
    let large = 2.0f64.powi(256);
    let mut model = jensen_blocks(1, 0.0);
    if let TwinOpSpec::Gemm {
        weight,
        bias,
        weight_rel_err,
        ..
    } = &mut model.ops[3]
    {
        weight.fill(0.0);
        weight[0] = eta;
        weight[2] = eta;
        weight[4] = 2.0 * eta;
        // The nominal normalized seed(1,1,-1) cancels. Integer seed products
        // avoid nominal half-eta rounding masking the separate fold charge.
        bias.fill(0.0);
        *weight_rel_err = eta;
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0; 2], &[1.0; 2], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let lower = checked_jensen(
        &eng,
        &source,
        &[I::p(large), I::p(large), I::p(-large)],
        I::p(0.0),
        &[Hyperedge {
            positive: vec![(0, 1.0), (1, 1.0)],
            negative: 2,
            tau: large,
        }],
    )
    .unwrap();
    // Nominal-only directed addition yields coefficient lower-2eta and box
    // floor-7eta; full UP(rel*|W|) dust before scaling charges strictly more.
    assert!(lower <= -9.0 * large * eta);
    let actual_extreme = -rat(4.0) * rat(large) * rat(eta) * rat(eta) * rat(3.0);
    assert!(rat(lower) <= actual_extreme);
}

#[test]
fn jensen_independent_fold_and_bias_balls_do_not_nominally_cancel() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let dust = 2.0f64.powi(-16);
    let mut model = jensen_blocks(1, 1.0 / 16.0);
    if let TwinOpSpec::Gemm {
        weight_rel_err,
        bias_err,
        ..
    } = &mut model.ops[3]
    {
        *weight_rel_err = dust;
        bias_err.fill(dust);
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0; 2], &[1.0; 2], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let (q, b) = source.margin(&eng, 0, deadline()).unwrap();
    let lower = checked_jensen(&eng, &source, &q, b, &hyperedges(1, 1.0)).unwrap();
    assert!(lower > 1.0 / 32.0 && lower < 1.0 / 16.0);
    for x in [-1.0f64, 0.0, 1.0] {
        for z in [-1.0f64, 0.0, 1.0] {
            for sign in [-1.0, 1.0] {
                let xp =
                    ((rat(1.0) - rat(sign * dust)) * rat(x + 2.0) - rat(2.0) - rat(sign * dust))
                        .max(BigRational::zero());
                let zp =
                    ((rat(1.0) - rat(sign * dust)) * rat(z + 2.0) - rat(2.0) - rat(sign * dust))
                        .max(BigRational::zero());
                let n = (rat(0.5) * (rat(1.0) + rat(sign * dust)) * (rat(x + 2.0) + rat(z + 2.0))
                    - rat(2.0)
                    + rat(sign * dust))
                .max(BigRational::zero());
                assert!(rat(lower) <= rat(1.0 / 16.0) + rat(0.5) * xp + rat(0.5) * zp - n);
            }
        }
    }
}

#[test]
fn jensen_invalid_duplicate_heads_foreign_generation_and_zero_scale_refuse_safely() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&jensen_blocks(1, 1.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 2], &[1.0; 2], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let (q, b) = source.margin(&eng, 0, deadline()).unwrap();
    for edge in [
        Hyperedge {
            positive: vec![(0, 0.5), (0, 0.5)],
            negative: 2,
            tau: 1.0,
        },
        Hyperedge {
            positive: vec![(2, 0.5)],
            negative: 2,
            tau: 1.0,
        },
        Hyperedge {
            positive: vec![(0, -0.5)],
            negative: 2,
            tau: 1.0,
        },
        Hyperedge {
            positive: vec![(0, f64::NAN)],
            negative: 2,
            tau: 1.0,
        },
        Hyperedge {
            positive: vec![(0, 0.5)],
            negative: 3,
            tau: 1.0,
        },
        Hyperedge {
            positive: vec![(0, 0.5)],
            negative: 2,
            tau: -1.0,
        },
        Hyperedge {
            positive: vec![],
            negative: 2,
            tau: 1.0,
        },
    ] {
        let mut work = adaptive::Report::default();
        assert!(jensen_candidate(
            &mut adaptive::Contractions::new(&eng, &source),
            &source,
            &q,
            b,
            &[edge],
            deadline(),
            &mut work
        )
        .is_none());
        assert_eq!(work.contraction_counts().0, 0);
    }
    let rebuilt = RootGates::build(&net, &[-1.0; 2], &[1.0; 2], RoundMode::Outward, None).unwrap();
    let foreign = BackwardEngine::new(&net, &rebuilt);
    assert!(checked_jensen(&foreign, &source, &q, b, &hyperedges(1, 1.0)).is_none());
    let mut work = adaptive::Report::default();
    let zero = jensen_candidate(
        &mut adaptive::Contractions::new(&eng, &source),
        &source,
        &q,
        b,
        &hyperedges(1, 0.0),
        deadline(),
        &mut work,
    )
    .unwrap();
    assert_eq!(work.contraction_counts().0, 0);
    assert!(zero <= source.baseline(&q, b, deadline()).unwrap());
}

#[test]
fn jensen_partial_candidate_and_original_expiry_preserve_required_completed_floors() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&jensen_blocks(5, 1.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 10], &[1.0; 10], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let margins = margins(&eng, &source);
    let before = floors(&source, &margins);
    let mut old = before.clone();
    refine_portfolio(
        &eng,
        &source,
        &margins,
        &mut old,
        deadline(),
        deadline(),
        false,
    );
    for mode in [1, 2, 3] {
        let mut bounds = before.clone();
        STOP_AFTER_FRACTIONAL.with(|s| s.set(mode));
        let report = refine(&eng, &source, &margins, &mut bounds, deadline(), deadline());
        STOP_AFTER_NEW_HYPEREDGE.with(|s| s.set(false));
        assert_eq!(report.jensen_candidates, 0);
        if mode == 2 {
            assert!(report.authority_expired);
            assert_eq!(bounds, before);
            assert_eq!(report.classes_improved, 0);
        } else {
            assert_eq!(bounds, old);
            assert_eq!(report.jensen_seed_checks, usize::from(mode == 3));
        }
    }
}

#[test]
fn jensen_same_transport_seed_cap_discards_incomplete_six_block_candidate() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&jensen_blocks(6, 1.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 12], &[1.0; 12], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let margins = margins(&eng, &source);
    let mut old = floors(&source, &margins);
    let old_report = refine_portfolio(
        &eng,
        &source,
        &margins,
        &mut old,
        deadline(),
        deadline(),
        false,
    );
    assert_eq!(old_report.seed_checks, 12);
    let mut bounds = floors(&source, &margins);
    let report = refine(&eng, &source, &margins, &mut bounds, deadline(), deadline());
    assert_eq!(report.seed_checks, MAX_PAIRS);
    assert_eq!(report.jensen_seed_checks, 4);
    assert_eq!(report.jensen_candidates, 0);
    assert_eq!(bounds, old);
}

#[test]
fn jensen_zero_negative_and_missing_positive_mass_never_authorize_original_instance() {
    let _capture = HeadGroupSourceGuard::enter(true);
    for (b, positive, exact_min) in [
        (0.0, 0.5, 0.0),
        (-1.0 / 16.0, 0.5, -1.0 / 16.0),
        (1.0 / 16.0, 0.25, -7.0 / 16.0),
    ] {
        let mut model = jensen_blocks(1, b);
        if let TwinOpSpec::Gemm { weight, .. } = &mut model.ops[5] {
            weight[0] = positive;
            weight[1] = positive;
            weight[6] = positive;
            weight[7] = positive;
        }
        let net = TwinNet::compile(&model).unwrap();
        let root = RootGates::build(&net, &[-1.0; 2], &[1.0; 2], RoundMode::Outward, None).unwrap();
        let eng = BackwardEngine::new(&net, &root);
        let mut out = attempt_with_provider(&eng, 0, &[1, 2], policy(), deadline(), || None);
        // x=z=0 witnesses the ties/negative bias; x=z=1 witnesses missing mass.
        assert!(out.prelude.as_ref().unwrap().bounds[0] <= exact_min);
        let cfg = BabConfig {
            deadline: Some(deadline()),
            head_groups: Some(policy()),
            ..BabConfig::default()
        };
        assert!(crate::margin_row::bab::direct_root_outcome(
            &eng,
            0,
            &[1, 2],
            &cfg,
            &[],
            Some(&mut out),
            Instant::now()
        )
        .is_none());
    }
}

#[test]
fn jensen_unnormalized_aggregate_support_covers_all_activation_quadrants() {
    for lambdas in [[0.25, 0.75], [2.0, 0.5], [0.0, 0.0]] {
        for a in [-1.0f64, 0.0, 1.0] {
            for b in [-1.0f64, 0.0, 1.0] {
                for n in [-1.0f64, 0.0, 1.0] {
                    let u = lambdas[0] * a + lambdas[1] * b;
                    let floor = difference_floor(I::p(u), I::p(n), u - n).unwrap();
                    let exact = rat(lambdas[0]) * rat(a.max(0.0))
                        + rat(lambdas[1]) * rat(b.max(0.0))
                        - rat(n.max(0.0));
                    assert!(rat(floor) <= exact);
                }
            }
        }
    }
}

#[test]
fn jensen_sampled_omitted_feature_cannot_replace_full_width_source_replay() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let width = 129;
    let n = width.min(FUNCTIONAL_MAX_FEATURES);
    let sampled: Vec<_> = (0..n).map(|k| k * (width - 1) / (n - 1)).collect();
    let omitted = (2..width).find(|k| !sampled.contains(k)).unwrap();
    let mut model = jensen_blocks(1, 1.0 / 16.0);
    model.n_in = width;
    if let TwinOpSpec::Conv {
        weight,
        bias,
        bias_err,
        kernel,
        ishape,
        oshape,
        ..
    } = &mut model.ops[0]
    {
        *weight = vec![0.0; width * width];
        for k in 0..width {
            weight[k * width + k] = 1.0;
        }
        *bias = vec![2.0; width];
        *bias_err = vec![0.0; width];
        *kernel = (width, width, 1, 1);
        *ishape = (width, 1, 1);
        *oshape = (width, 1, 1);
    }
    if let TwinOpSpec::Gemm {
        weight,
        bias,
        shape,
        ..
    } = &mut model.ops[3]
    {
        *weight = vec![0.0; 3 * width];
        weight[0] = 1.0;
        weight[width + 1] = 1.0;
        weight[2 * width] = 0.5;
        weight[2 * width + 1] = 0.5;
        weight[2 * width + omitted] = 1.0;
        *bias = vec![-2.0, -2.0, -4.0];
        *shape = (3, width);
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(
        &net,
        &vec![-1.0; width],
        &vec![1.0; width],
        RoundMode::Outward,
        None,
    )
    .unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = source(&eng);
    let (q, b) = source.margin(&eng, 0, deadline()).unwrap();
    let mut report = Report::default();
    let geometry = Geometry::build(&eng, &source, deadline(), &mut report).unwrap();
    assert_eq!(
        geometry.combination_score(&[(0, 0.5), (1, 0.5)], 2, deadline(), &mut report),
        Some(0.0)
    );
    let mut work = adaptive::Report::default();
    let lower = jensen_candidate(
        &mut adaptive::Contractions::new(&eng, &source),
        &source,
        &q,
        b,
        &hyperedges(1, 1.0),
        deadline(),
        &mut work,
    )
    .unwrap();
    // Actual unsafe original point: x0=x1=0, omitted input=1, hence F=B-1.
    assert!(lower <= -15.0 / 16.0);
    assert_eq!(work.contraction_counts().2, 3 * width);
}

#[test]
fn jensen_complete_positive_subset_cannot_publish_a_negative_original_class() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut model = jensen_blocks(2, 1.0 / 16.0);
    let q = [0.5, 0.5, 0.5, 0.5, -1.0, -1.0];
    if let TwinOpSpec::Gemm {
        weight,
        bias,
        bias_err,
        shape,
        ..
    } = &mut model.ops[5]
    {
        *weight = q
            .iter()
            .copied()
            .chain([0.0; 6])
            .chain(q.iter().map(|q| -q))
            .chain(q.iter().copied())
            .collect();
        *bias = vec![1.0 / 16.0, 0.0, 0.0, 1.0 / 8.0];
        *bias_err = vec![0.0; 4];
        *shape = (4, 6);
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0; 4], &[1.0; 4], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let mut out = attempt_with_provider(&eng, 0, &[2, 3, 1], policy(), deadline(), || None);
    let prelude = out.prelude.as_ref().unwrap();
    assert!(!prelude.complete());
    assert_eq!(prelude.source.adversaries, vec![2, 3, 1]);
    assert!(prelude.bounds[0] > 1.0 / 32.0 && prelude.bounds[0] <= 1.0 / 16.0);
    assert!(prelude.bounds[1] <= -1.0 / 16.0);
    assert!(prelude.bounds[2] > 1.0 / 32.0 && prelude.bounds[2] <= 1.0 / 16.0);
    let cfg = BabConfig {
        deadline: Some(deadline()),
        head_groups: Some(policy()),
        ..BabConfig::default()
    };
    assert!(crate::margin_row::bab::direct_root_outcome(
        &eng,
        0,
        &[2, 3, 1],
        &cfg,
        &[],
        Some(&mut out),
        Instant::now()
    )
    .is_none());
}

/// review(eng14): a seeded xorshift stream for the enumeration-exact tests
/// below (deterministic, so a failure names its trial).
struct ExactRng(u64);

impl ExactRng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    /// A dyadic `k / den` with `|k| <= span`.
    fn dyadic(&mut self, span: usize, den: f64) -> f64 {
        (self.below(2 * span + 1) as f64 - span as f64) / den
    }
}

/// review(eng14) — ENUMERATION-EXACT difference floor. `difference_floor(P,
/// N, delta)` claims `ReLU(a) - ReLU(b) >= floor` for EVERY `a in P`, `b in N`
/// with `a - b >= delta`, and all three transport covers (pairs, fractional
/// shared mass, Jensen hyperedges) price their masses with it. On that
/// feasible polygon `ReLU(a) - ReLU(b)` is affine on each cell cut by `a = 0`
/// and `b = 0`, so its minimum is attained at a crossing of two of the seven
/// lines below; enumerating every feasible crossing gives the EXACT minimum.
/// (The quadrant test above samples only `{lo, hi, 0}^2`, which misses the
/// crossings on `a - b = delta` — e.g. P=[1,3], N=[0,2], delta=0 has minimum
/// 0 at a = b, where that sample's smallest value is 1.)
#[test]
fn difference_floor_never_exceeds_the_exact_minimum_over_its_polygon() {
    use crate::margin_row::exact_relu_oracle::relu;
    let grid = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.25, 1.0, 2.0, 3.0];
    let deltas = [-7.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0, 7.0];
    let mut checked = 0usize;
    let mut tight = 0usize;
    for (i, &pl) in grid.iter().enumerate() {
        for &ph in &grid[i..] {
            for (k, &nl) in grid.iter().enumerate() {
                for &nh in &grid[k..] {
                    for &delta in &deltas {
                        let Some(floor) =
                            difference_floor(I { lo: pl, hi: ph }, I { lo: nl, hi: nh }, delta)
                        else {
                            continue;
                        };
                        // alpha * a + beta * b = gamma
                        let lines = [
                            (1.0, 0.0, pl),
                            (1.0, 0.0, ph),
                            (0.0, 1.0, nl),
                            (0.0, 1.0, nh),
                            (1.0, -1.0, delta),
                            (1.0, 0.0, 0.0),
                            (0.0, 1.0, 0.0),
                        ];
                        let mut exact: Option<BigRational> = None;
                        for (u, l1) in lines.iter().enumerate() {
                            for l2 in &lines[u + 1..] {
                                let det = rat(l1.0 * l2.1 - l1.1 * l2.0);
                                if det.is_zero() {
                                    continue;
                                }
                                let a = (rat(l1.2) * rat(l2.1) - rat(l1.1) * rat(l2.2)) / &det;
                                let b = (rat(l1.0) * rat(l2.2) - rat(l1.2) * rat(l2.0)) / &det;
                                if a < rat(pl)
                                    || a > rat(ph)
                                    || b < rat(nl)
                                    || b > rat(nh)
                                    || &a - &b < rat(delta)
                                {
                                    continue;
                                }
                                let v = relu(a) - relu(b);
                                if exact.as_ref().is_none_or(|e| v < *e) {
                                    exact = Some(v);
                                }
                            }
                        }
                        let Some(exact) = exact else {
                            continue; // an empty polygon: nothing to bound
                        };
                        assert!(
                            rat(floor) <= exact,
                            "P=[{pl},{ph}] N=[{nl},{nh}] delta={delta}: floor {floor} > exact \
                             minimum {exact}"
                        );
                        checked += 1;
                        tight += usize::from(rat(floor) == exact);
                    }
                }
            }
        }
    }
    assert!(checked > 5_000, "VOID: only {checked} feasible cases");
    assert!(
        tight > 0,
        "positive control: some floor is the exact minimum"
    );
}

/// review(eng14) — ENUMERATION-EXACT soundness of the three transport covers
/// AND of their seed REUSE. Over seeded random direct heads (two original
/// inputs, an identity trunk kept active by its bias, six Gemm1 heads whose
/// second half are shifted twins and Jensen combinations of the first, and
/// three outputs), the exact minimum of each ORIGINAL margin over the box
/// comes from the arrangement oracle, and none of these may exceed it: the
/// scalar baseline; ARBITRARY advice — random directed pairs and random
/// hyperedges (any `tau >= 0`, any `lambda >= 0`, normalised or not) —
/// replayed through `candidate` / `jensen_candidate` with ONE contraction
/// cache shared by both classes, so the second class reads every seed bound
/// back from the cache (a seed's certified lower bound is a property of the
/// source box alone, never of the recipient); and the production portfolio
/// `refine` (disjoint, fractional and Jensen passes). Fails VOID when no pair
/// or hyperedge replays or the cache is never re-read.
#[test]
fn transport_covers_and_reused_seeds_never_exceed_the_exact_arrangement_minimum() {
    use crate::margin_row::exact_relu_oracle::{min_over_box, Line};
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut rng = ExactRng(0x9e37_79b9_7f4a_7c15);
    let (heads, width, outputs) = (6usize, 2usize, 3usize);
    let (mut sources, mut pairs_ok, mut jensen_ok, mut reused, mut improved) = (0, 0, 0, 0, 0);
    for trial in 0..40 {
        let mut first = vec![0.0; heads * width];
        let mut bias = vec![0.0; heads];
        for j in 0..3 {
            for k in 0..width {
                first[j * width + k] = rng.dyadic(8, 8.0);
            }
            bias[j] = rng.dyadic(8, 16.0);
        }
        // The negative partners: a shifted twin of head 0 and two Jensen
        // combinations, each with a small dyadic bias shift.
        for (j, mix) in [
            (3, [1.0, 0.0, 0.0]),
            (4, [0.5, 0.5, 0.0]),
            (5, [0.0, 0.25, 0.75]),
        ] {
            for k in 0..width {
                first[j * width + k] = (0..3).map(|p| mix[p] * first[p * width + k]).sum();
            }
            bias[j] = (0..3).map(|p| mix[p] * bias[p]).sum::<f64>() + rng.dyadic(1, 16.0);
        }
        let mut last = vec![0.0; outputs * heads];
        for j in 0..heads {
            let magnitude = (1 + rng.below(8)) as f64 / 8.0;
            last[j] = if j < 3 { magnitude } else { -magnitude };
            last[2 * heads + j] = rng.dyadic(8, 8.0);
        }
        let last_bias = vec![rng.dyadic(8, 16.0), 0.0, rng.dyadic(8, 16.0)];
        let lo: Vec<f64> = (0..width).map(|_| -(rng.below(9) as f64) / 8.0).collect();
        let hi: Vec<f64> = (0..width).map(|_| rng.below(9) as f64 / 8.0).collect();
        let mut trunk = vec![0.0; width * width];
        for k in 0..width {
            trunk[k * width + k] = 1.0;
        }
        let spec = TwinSpec {
            n_in: width,
            ops: vec![
                TwinOpSpec::Conv {
                    input: 0,
                    weight: trunk,
                    bias: vec![2.0; width],
                    bias_err: vec![0.0; width],
                    weight_rel_err: 0.0,
                    kernel: (width, width, 1, 1),
                    stride: (1, 1),
                    pads: (0, 0, 0, 0),
                    ishape: (width, 1, 1),
                    oshape: (width, 1, 1),
                },
                TwinOpSpec::Relu { input: 1 },
                TwinOpSpec::Flatten { input: 2 },
                TwinOpSpec::Gemm {
                    input: 3,
                    weight: first.clone(),
                    bias: bias.clone(),
                    bias_err: vec![0.0; heads],
                    weight_rel_err: 0.0,
                    shape: (heads, width),
                },
                TwinOpSpec::Relu { input: 4 },
                TwinOpSpec::Gemm {
                    input: 5,
                    weight: last.clone(),
                    bias: last_bias.clone(),
                    bias_err: vec![0.0; outputs],
                    weight_rel_err: 0.0,
                    shape: (outputs, heads),
                },
            ],
        };
        let net = TwinNet::compile(&spec).unwrap();
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).unwrap();
        let eng = BackwardEngine::new(&net, &root);
        let Some(source) = DirectScalarHeadSource::build(&eng, 0, &[1, 2], deadline()) else {
            continue;
        };
        sources += 1;
        let margins = margins(&eng, &source);
        // The trunk keeps every feature at x + 2 (>= 1 on the box), so head j
        // is the exact affine preactivation a_j . x + (b_j + 2 * sum a_j).
        let lines: Vec<Line> = (0..heads)
            .map(|j| {
                let a = &first[j * width..(j + 1) * width];
                Line::of(a, bias[j] + 2.0 * a.iter().sum::<f64>())
            })
            .collect();
        let rlo: Vec<BigRational> = lo.iter().map(|&v| rat(v)).collect();
        let rhi: Vec<BigRational> = hi.iter().map(|&v| rat(v)).collect();
        let exact: Vec<BigRational> = [1usize, 2]
            .iter()
            .map(|&o| {
                let q: Vec<BigRational> = (0..heads)
                    .map(|j| rat(last[j]) - rat(last[o * heads + j]))
                    .collect();
                let c = rat(last_bias[0]) - rat(last_bias[o]);
                min_over_box(&rlo, &rhi, &c, &q, &lines)
            })
            .collect();
        for (r, (q, b)) in margins.iter().enumerate() {
            let floor = source.baseline(q, *b, deadline()).unwrap();
            assert!(rat(floor) <= exact[r], "trial {trial} class {r}: baseline");
        }
        // Arbitrary advice: distinct directed pairs and hyperedges.
        let mut pairs: Vec<Pair> = Vec::new();
        while pairs.len() < 3 {
            let positive = rng.below(heads);
            let negative = (positive + 1 + rng.below(heads - 1)) % heads;
            if pairs
                .iter()
                .any(|p| p.positive == positive && p.negative == negative)
            {
                continue;
            }
            let tau = rng.below(17) as f64 / 8.0;
            pairs.push(Pair {
                positive,
                negative,
                tau,
            });
        }
        let hyperedges: Vec<Hyperedge> = (0..2)
            .map(|_| {
                let negative = rng.below(heads);
                let mut others: Vec<usize> = (0..heads).filter(|&j| j != negative).collect();
                let mut positive = Vec::new();
                for _ in 0..2 + rng.below(2) {
                    let head = others.remove(rng.below(others.len()));
                    positive.push((head, rng.below(13) as f64 / 8.0));
                }
                Hyperedge {
                    positive,
                    negative,
                    tau: (1 + rng.below(16)) as f64 / 8.0,
                }
            })
            .collect();
        let mut cache = adaptive::Contractions::new(&eng, &source);
        let mut work = adaptive::Report::default();
        for (r, (q, b)) in margins.iter().enumerate() {
            if let Some(lower) =
                candidate(&mut cache, &source, q, *b, &pairs, deadline(), &mut work)
            {
                assert!(
                    rat(lower) <= exact[r],
                    "trial {trial} class {r}: pair cover"
                );
                pairs_ok += 1;
            }
            if let Some(lower) = jensen_candidate(
                &mut cache,
                &source,
                q,
                *b,
                &hyperedges,
                deadline(),
                &mut work,
            ) {
                assert!(
                    rat(lower) <= exact[r],
                    "trial {trial} class {r}: Jensen cover"
                );
                jensen_ok += 1;
            }
        }
        reused += work.contraction_counts().1;
        let mut bounds = floors(&source, &margins);
        let report = refine(&eng, &source, &margins, &mut bounds, deadline(), deadline());
        for (r, bound) in bounds.iter().enumerate() {
            assert!(
                rat(*bound) <= exact[r],
                "trial {trial} class {r}: refine portfolio"
            );
        }
        improved += report.classes_improved;
    }
    eprintln!(
        "[transport-exact] sources={sources} pairs={pairs_ok} jensen={jensen_ok} \
         cache_hits={reused} refine_improved={improved}"
    );
    assert!(sources >= 20, "VOID: only {sources} direct sources built");
    assert!(
        pairs_ok > 0 && jensen_ok > 0,
        "VOID: pairs={pairs_ok} jensen={jensen_ok}"
    );
    assert!(
        reused > 0,
        "VOID: no seed bound was re-read from the shared cache"
    );
}
