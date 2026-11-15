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

// z=x+2 is a stable trunk ReLU. Head offsets are expressed in the
// original x coordinates; the final third output adds a second real margin.
fn spec(inputs: usize, weights: Vec<f64>, shifts: Vec<f64>, q: Vec<f64>, b: f64) -> TwinSpec {
    let heads = q.len();
    let mut trunk = vec![0.0; inputs * inputs];
    for i in 0..inputs {
        trunk[i * inputs + i] = 1.0;
    }
    let biases: Vec<_> = (0..heads)
        .map(|j| shifts[j] - 2.0 * weights[j * inputs..(j + 1) * inputs].iter().sum::<f64>())
        .collect();
    TwinSpec {
        n_in: inputs,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: trunk,
                bias: vec![2.0; inputs],
                bias_err: vec![0.0; inputs],
                weight_rel_err: 0.0,
                kernel: (inputs, inputs, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (inputs, 1, 1),
                oshape: (inputs, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: weights,
                bias: biases,
                bias_err: vec![0.0; heads],
                weight_rel_err: 0.0,
                shape: (heads, inputs),
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

fn twins(b: f64) -> TwinSpec {
    spec(1, vec![1.0; 2], vec![0.0; 2], vec![1.0, -1.0], b)
}

fn run(eng: &BackwardEngine<'_>, source: &DirectScalarHeadSource) -> (Vec<f64>, Report) {
    let margins: Vec<_> = (0..source.adversaries.len())
        .map(|r| source.margin(eng, r, deadline()).unwrap())
        .collect();
    let mut bounds: Vec<_> = margins
        .iter()
        .map(|(q, b)| source.baseline(q, *b, deadline()).unwrap())
        .collect();
    let report = refine(
        eng,
        source,
        &margins,
        &mut bounds,
        policy(),
        deadline(),
        deadline(),
    );
    (bounds, report)
}

#[test]
fn adaptive_identical_twins_close_two_phases_and_publish_the_real_instance() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&twins(1.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = DirectScalarHeadSource::build(&eng, 0, &[1, 2], deadline()).unwrap();
    let (q, b) = source.margin(&eng, 0, deadline()).unwrap();
    let geometry = Geometry::build(&eng, &source, deadline()).unwrap();
    let candidate = geometry
        .propose(&source, &q, b, &[], 4, deadline(), &mut Report::default())
        .unwrap();
    let mut cache = Contractions {
        eng: &eng,
        source: &source,
        completed: Vec::new(),
        attempted: 0,
    };
    let scalar = I::p(
        cache
            .lower(&candidate.seed, deadline(), &mut Report::default())
            .unwrap(),
    )
    .add(candidate.offset)
    .unwrap()
    .lo;
    assert!(scalar <= -7.0 / 16.0);
    // Exact authored twins cancel for every real x, not merely sampled x.
    for x in [-1.0, 0.0, 1.0] {
        let rho = rat(x).max(BigRational::zero());
        assert_eq!(rat(1.0 / 16.0) + &rho - rho, rat(1.0 / 16.0));
    }
    let mut out =
        adaptive_only_attempt_with_provider(&eng, 0, &[1, 2], policy(), deadline(), || {
            panic!("closed adaptive cover must not initialize advice")
        });
    let prelude = out.prelude.as_ref().unwrap();
    assert!(prelude.complete());
    assert!(prelude.bounds()[0] > 1.0 / 32.0 && prelude.bounds()[0] <= 1.0 / 16.0);
    assert_eq!(out.adaptive.splits, 1);
    assert_eq!(out.adaptive.leaves, vec![2, 1]);
    assert_eq!(out.calls, 0);
    assert_eq!(out.budget.calls, 0);
    assert_eq!(out.passes, 0);
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
        panic!("full owned cover must reach the ordinary authorizer")
    };
    assert_eq!(stats.stop, "direct_head_closed_at_root");
    assert_eq!(stats.root_closed_classes, 2);
    assert_eq!(stats.expansions, 0);
    assert_eq!(stats.ledger_ok, Some(true));
    assert_eq!(token.scope(), TokenScope::Instance);
    assert_eq!(token.source(), UnsatSource::Authorized);
    assert!(token.covers_all(&[1, 2]));
}

#[test]
fn adaptive_heterogeneous_three_leaf_cover_beats_unconditional_and_one_split() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&spec(
        1,
        vec![1.0; 4],
        vec![0.0, 0.0, -0.5, -0.5],
        vec![1.0, -1.0, 1.0, -1.0],
        3.0 / 8.0,
    ))
    .unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = DirectScalarHeadSource::build(&eng, 0, &[1, 2], deadline()).unwrap();
    let (q, b) = source.margin(&eng, 0, deadline()).unwrap();
    let geometry = Geometry::build(&eng, &source, deadline()).unwrap();
    let mut cache = Contractions {
        eng: &eng,
        source: &source,
        completed: Vec::new(),
        attempted: 0,
    };
    let mut report = Report::default();
    let baseline = source.baseline(&q, b, deadline()).unwrap();
    let mut cover = Cover::new(baseline).unwrap();
    cover.improve(
        0,
        check_cell(
            &mut cache,
            &geometry,
            &q,
            b,
            &cover.cells[0],
            4,
            deadline(),
            &mut report,
        )
        .unwrap(),
    );
    assert!(cover.lower() < 0.0);
    let (left, right) = cover.split(0, 1).unwrap();
    for child in [left, right] {
        cover.improve(
            child,
            check_cell(
                &mut cache,
                &geometry,
                &q,
                b,
                &cover.cells[child],
                4,
                deadline(),
                &mut report,
            )
            .unwrap(),
        );
    }
    assert!(cover.cells[left].floor > 0.0);
    assert!(cover.cells[right].floor <= 0.0);
    assert!(cover.lower() <= 0.0);
    let (bounds, report) = run(&eng, &source);
    assert_eq!(report.splits, 2);
    assert_eq!(report.leaves, vec![3, 1]);
    assert_eq!(report.max_depth, 2);
    assert!(bounds[0] > 1.0 / 16.0 && bounds[0] <= 3.0 / 8.0);
    assert!(report.fold_terms > 0 && report.seed_checks <= MAX_SEEDS);
}

#[test]
fn adaptive_negative_selected_sign_and_zero_ties_cannot_authorize() {
    let _capture = HeadGroupSourceGuard::enter(true);
    for model in [
        twins(0.0),
        spec(1, vec![1.0; 2], vec![0.0; 2], vec![0.0, -1.0], 1.0 / 16.0),
    ] {
        let net = TwinNet::compile(&model).unwrap();
        let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
        let eng = BackwardEngine::new(&net, &root);
        let source = DirectScalarHeadSource::build(&eng, 0, &[1, 2], deadline()).unwrap();
        let (bounds, report) = run(&eng, &source);
        assert!(bounds[0] <= 0.0);
        assert_eq!(report.leaves, vec![2, 1]);
        let mut out =
            adaptive_only_attempt_with_provider(&eng, 0, &[1, 2], policy(), deadline(), || None);
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
fn adaptive_expiry_after_first_child_keeps_the_missing_sibling_floor() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&twins(1.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = DirectScalarHeadSource::build(&eng, 0, &[1, 2], deadline()).unwrap();
    STOP_AFTER_FIRST_CHILD.with(|s| s.set(1));
    let (bounds, report) = run(&eng, &source);
    STOP_AFTER_FIRST_CHILD.with(|s| s.set(0));
    assert!(report.slice_expired && !report.authority_expired);
    assert_eq!(report.leaves, vec![2, 1]);
    assert_eq!(report.splits, 1);
    assert!(bounds[0] <= -7.0 / 16.0);
    assert!(bounds[1] > 0.0);
    STOP_AFTER_FIRST_CHILD.with(|s| s.set(1));
    let mut out =
        adaptive_only_attempt_with_provider(&eng, 0, &[1, 2], policy(), deadline(), || None);
    STOP_AFTER_FIRST_CHILD.with(|s| s.set(0));
    assert!(out.adaptive.slice_expired);
    assert!(!out.prelude.as_ref().unwrap().complete());
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
fn adaptive_whole_contraction_prices_full_fold_bias_and_coefficients() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut model = twins(1.0 / 16.0);
    let dust = 2.0f64.powi(-10);
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
    let source = DirectScalarHeadSource::build(&eng, 0, &[1, 2], deadline()).unwrap();
    let mut cache = Contractions {
        eng: &eng,
        source: &source,
        completed: Vec::new(),
        attempted: 0,
    };
    let mut report = Report::default();
    let lower = cache.lower(&[1.0, -1.0], deadline(), &mut report).unwrap();
    assert!(lower < 0.0); // nominal identical rows cannot cancel independent balls
    for z in [1.0, 3.0] {
        for a in [-1.0, 1.0] {
            for b in [-1.0, 1.0] {
                let y1 = (rat(1.0) + rat(a * dust)) * rat(z) - rat(2.0) + rat(a * dust);
                let y2 = (rat(1.0) + rat(b * dust)) * rat(z) - rat(2.0) + rat(b * dust);
                assert!(rat(lower) <= y1 - y2);
            }
        }
    }
    let again = cache.lower(&[1.0, -1.0], deadline(), &mut report).unwrap();
    assert_eq!(lower.to_bits(), again.to_bits());
    assert_eq!(report.seed_checks, 1);
    assert_eq!(report.cache_hits, 1);
    // A deliberately nonpoint full Q proposal must retain all selected debt.
    let q = vec![
        I {
            lo: 0.875,
            hi: 1.125,
        },
        I {
            lo: -1.125,
            hi: -0.875,
        },
    ];
    let b = I { lo: 0.0, hi: 0.125 };
    let path = [Phase {
        head: 1,
        active: true,
    }];
    let c = column(&source, &q, b, &path, &[1.0, 0.0], &[0.0], deadline()).unwrap();
    let bound = I::p(cache.lower(&c.seed, deadline(), &mut report).unwrap())
        .add(c.offset)
        .unwrap()
        .lo;
    for z in [1.0, 2.0, 3.0] {
        for u in [q[0].lo, q[0].hi] {
            for v in [q[1].lo, q[1].hi] {
                let y = rat(z) - rat(2.0);
                if y >= BigRational::zero() {
                    assert!(rat(bound) <= rat(b.lo) + rat(u + v) * y);
                }
            }
        }
    }
}

#[test]
fn adaptive_source_generation_order_and_original_authority_refuse_foreign_replay() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let model = twins(1.0 / 16.0);
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = DirectScalarHeadSource::build(&eng, 0, &[1, 2], deadline()).unwrap();
    let rebuilt = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let foreign = BackwardEngine::new(&net, &rebuilt);
    let margins: Vec<_> = (0..2)
        .map(|r| source.margin(&eng, r, deadline()).unwrap())
        .collect();
    for (receiver, authority) in [(&foreign, deadline()), (&eng, Instant::now())] {
        let mut bounds = vec![-1.0, 1.0];
        let report = refine(
            receiver,
            &source,
            &margins,
            &mut bounds,
            policy(),
            deadline(),
            authority,
        );
        assert_eq!(report.seed_checks, 0);
        assert_eq!(report.splits, 0);
        assert_eq!(bounds, vec![-1.0, 1.0]);
    }
    let out = adaptive_only_attempt_with_provider(&eng, 0, &[1, 2], policy(), deadline(), || None);
    let prelude = out.prelude.unwrap();
    assert!(!prelude.matches(&eng, 0, &[2, 1], deadline()));
    assert!(!prelude.matches(&eng, 1, &[0, 2], deadline()));
    assert!(!prelude.matches(&foreign, 0, &[1, 2], deadline()));
    assert!(!prelude.matches(&eng, 0, &[1, 2], Instant::now()));
}

#[test]
fn adaptive_depth_split_seed_caps_preserve_an_open_complete_cover() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut w = vec![0.0; 50];
    for i in 0..5 {
        w[(2 * i) * 5 + i] = 1.0;
        w[(2 * i + 1) * 5 + i] = 1.0;
    }
    let net = TwinNet::compile(&spec(
        5,
        w,
        vec![0.0; 10],
        (0..10)
            .map(|j| if j % 2 == 0 { 1.0 } else { -1.0 })
            .collect(),
        1.0 / 16.0,
    ))
    .unwrap();
    let root = RootGates::build(&net, &[-1.0; 5], &[1.0; 5], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = DirectScalarHeadSource::build(&eng, 0, &[1, 2], deadline()).unwrap();
    let (bounds, report) = run(&eng, &source);
    assert!(report.splits > 0 && report.splits <= MAX_SPLITS);
    assert!(report.max_depth <= MAX_DEPTH);
    assert!(report.leaves.iter().sum::<usize>() <= MAX_LEAVES);
    assert!(report.seed_checks <= MAX_SEEDS);
    assert!(bounds[0] <= -7.0 / 16.0);
    // The connected attempt leaves advice allowance untouched when absent.
    let out = adaptive_only_attempt_with_provider(&eng, 0, &[1, 2], policy(), deadline(), || None);
    assert!(out.elapsed > Duration::ZERO);
    assert_eq!(out.budget.calls, 0);
    assert!(out.budget.charged < Duration::from_millis(250));
    assert_eq!(out.calls, 0);
    assert!(!out.prelude.unwrap().complete());
}

#[test]
fn adaptive_negative_multipliers_and_duplicate_or_out_of_range_paths_decline() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&twins(1.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = DirectScalarHeadSource::build(&eng, 0, &[1, 2], deadline()).unwrap();
    let (q, b) = source.margin(&eng, 0, deadline()).unwrap();
    let p = Phase {
        head: 1,
        active: true,
    };
    assert!(column(&source, &q, b, &[p], &[1.0, 0.0], &[-1.0], deadline()).is_none());
    assert!(column(&source, &q, b, &[p, p], &[1.0, 0.0], &[0.0; 2], deadline()).is_none());
    assert!(column(
        &source,
        &q,
        b,
        &[Phase {
            head: 2,
            active: false
        }],
        &[1.0, 0.0],
        &[0.0],
        deadline()
    )
    .is_none());
    let mut cover = Cover::new(-1.0).unwrap();
    cover.split(0, 1).unwrap();
    assert!(cover.split(0, 1).is_none());
    assert_eq!(cover.cells.len(), 2);
    cover.improve(0, 1.0 / 16.0);
    assert_eq!(cover.lower(), -1.0); // no missing-sibling publication seam
}

#[test]
fn adaptive_zero_point_seed_retains_full_selected_q_residual() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&twins(1.0 / 16.0)).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = DirectScalarHeadSource::build(&eng, 0, &[1, 2], deadline()).unwrap();
    let q = [
        I {
            lo: 0.875,
            hi: 1.125,
        },
        I::p(0.0),
    ];
    let path = [Phase {
        head: 0,
        active: true,
    }];
    let candidate = column(&source, &q, I::p(0.0), &path, &[0.0; 2], &[1.0], deadline()).unwrap();
    assert_eq!(candidate.seed, vec![0.0; 2]);
    assert!(candidate.offset.lo <= -0.125);
    let mut cache = Contractions {
        eng: &eng,
        source: &source,
        completed: Vec::new(),
        attempted: 0,
    };
    let mut report = Report::default();
    assert_eq!(
        cache.lower(&candidate.seed, deadline(), &mut report),
        Some(0.0)
    );
    assert_eq!(report.fold_terms, 0);
    assert!(candidate.offset.lo <= 0.0);
}

#[test]
fn adaptive_underflowing_fold_error_is_charged_before_seed_amplification() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let eta = f64::from_bits(1);
    let large = 2.0f64.powi(256);
    let mut model = spec(1, vec![eta; 2], vec![0.0; 2], vec![1.0, -1.0], 1.0 / 16.0);
    if let TwinOpSpec::Gemm { weight_rel_err, .. } = &mut model.ops[3] {
        *weight_rel_err = eta;
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = DirectScalarHeadSource::build(&eng, 0, &[1, 2], deadline()).unwrap();
    let mut cache = Contractions {
        eng: &eng,
        source: &source,
        completed: Vec::new(),
        attempted: 0,
    };
    let lower = cache
        .lower(&[large, -large], deadline(), &mut Report::default())
        .unwrap();
    assert!(lower.is_finite() && lower < 0.0);
    // Distinguish the pre-amplification UP eta error charge from ordinary
    // normal-product ulp widening alone, which is much smaller here.
    assert!(lower <= -(large * eta));
    let exact_negative = -rat(2.0) * rat(large) * rat(eta) * rat(eta) * rat(3.0);
    assert!(rat(lower) <= exact_negative);
}

#[test]
fn adaptive_large_head_search_reprices_the_full_recipient_only_once() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut q = vec![1.0 / 1024.0; MAX_HEAD];
    q[0] = 1.0;
    q[MAX_HEAD - 1] = -1.0;
    let model = spec(1, vec![1.0; MAX_HEAD], vec![0.0; MAX_HEAD], q, 1.0 / 16.0);
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = DirectScalarHeadSource::build(&eng, 0, &[1, 2], deadline()).unwrap();
    let (q, b) = source.margin(&eng, 0, deadline()).unwrap();
    let geometry = Geometry::build(&eng, &source, deadline()).unwrap();
    let mut report = Report::default();
    let path = [Phase {
        head: MAX_HEAD - 1,
        active: false,
    }];
    let candidate = geometry
        .propose(&source, &q, b, &path, 4, deadline(), &mut report)
        .unwrap();
    assert_eq!(candidate.seed.len(), MAX_HEAD);
    assert!(report.search_trials > MAX_COORDINATES);
    assert_eq!(report.recipient_reprices, 1);
    assert_eq!(report.seed_checks, 0); // no speculative full Gemm contraction
    let rebuilt = column(
        &source,
        &q,
        b,
        &path,
        &candidate.seed,
        &[candidate.seed[MAX_HEAD - 1]],
        deadline(),
    )
    .unwrap();
    assert_eq!(candidate.seed, rebuilt.seed);
    assert_eq!(candidate.offset.lo.to_bits(), rebuilt.offset.lo.to_bits());
}

#[test]
fn adaptive_reordered_nine_class_cover_keeps_zero_and_negative_recipients_open() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let adversaries = [9, 2, 7, 1, 8, 3, 6, 4, 5];
    for unsafe_margin in [None, Some(0.0), Some(-1.0 / 16.0)] {
        let mut model = twins(1.0 / 16.0);
        if let TwinOpSpec::Gemm {
            weight,
            bias,
            bias_err,
            shape,
            ..
        } = &mut model.ops[5]
        {
            *weight = vec![0.0; 20];
            *bias = vec![-1.0; 10];
            bias[0] = 0.0;
            for (adversary, scale) in [(2, 1.0), (9, 2.0)] {
                weight[2 * adversary] = -scale;
                weight[2 * adversary + 1] = scale;
                bias[adversary] = -1.0 / 16.0;
            }
            if let Some(margin) = unsafe_margin {
                bias[7] = -margin;
            }
            *bias_err = vec![0.0; 10];
            *shape = (10, 2);
        }
        let net = TwinNet::compile(&model).unwrap();
        let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
        let eng = BackwardEngine::new(&net, &root);
        let mut factories = 0;
        let mut out = adaptive_only_attempt_with_provider(
            &eng,
            0,
            &adversaries,
            policy(),
            deadline(),
            || {
                factories += 1;
                assert!(unsafe_margin.is_some(), "complete cover must skip advice");
                None
            },
        );
        let prelude = out.prelude.as_ref().unwrap();
        assert_eq!(prelude.bounds().len(), 9);
        assert!(prelude.bounds()[0] > 1.0 / 32.0);
        assert!(prelude.bounds()[1] > 1.0 / 32.0);
        assert_eq!(out.adaptive.leaves, vec![2, 2, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(out.adaptive.splits, 2);
        assert_eq!(out.calls, 0);
        assert_eq!(out.passes, 0);
        assert!(factories <= 1);
        if unsafe_margin.is_none() {
            assert_eq!(factories, 0);
        }
        let mut reordered = adversaries;
        reordered.swap(0, 1);
        assert!(!prelude.matches(&eng, 0, &reordered, deadline()));
        let cfg = BabConfig {
            deadline: Some(deadline()),
            head_groups: Some(policy()),
            ..BabConfig::default()
        };
        let outcome = crate::margin_row::bab::direct_root_outcome(
            &eng,
            0,
            &adversaries,
            &cfg,
            &[],
            Some(&mut out),
            Instant::now(),
        );
        if unsafe_margin.is_some() {
            assert!(outcome.is_none());
            assert!(out.prelude.as_ref().unwrap().bounds()[2] <= 0.0);
        } else {
            let MarginRowOutcome::Unsat(stats, token) = outcome.unwrap() else {
                panic!("all nine original margins must reach the instance authorizer")
            };
            assert_eq!(stats.root_closed_classes, 9);
            assert_eq!(stats.expansions, 0);
            assert_eq!(stats.ledger_ok, Some(true));
            assert_eq!(token.scope(), TokenScope::Instance);
            assert_eq!(token.source(), UnsatSource::Authorized);
            assert!(token.covers_all(&adversaries));
        }
    }
}

#[test]
fn adaptive_feature_cap_refuses_before_contraction_and_preserves_complete_floors() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let width = MAX_CONTRACTION_FEATURES + 1;
    let mut model = twins(1.0 / 16.0);
    if let TwinOpSpec::Conv {
        weight,
        bias,
        bias_err,
        kernel,
        oshape,
        ..
    } = &mut model.ops[0]
    {
        *weight = vec![1.0; width];
        *bias = vec![2.0; width];
        *bias_err = vec![0.0; width];
        *kernel = (width, 1, 1, 1);
        *oshape = (width, 1, 1);
    }
    if let TwinOpSpec::Gemm { weight, shape, .. } = &mut model.ops[3] {
        *weight = vec![0.0; 2 * width];
        weight[0] = 1.0;
        weight[width] = 1.0;
        *shape = (2, width);
    }
    let net = TwinNet::compile(&model).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = DirectScalarHeadSource::build(&eng, 0, &[1, 2], deadline()).unwrap();
    assert_eq!(source.features.len(), width);
    let mut cache = Contractions {
        eng: &eng,
        source: &source,
        completed: Vec::new(),
        attempted: 0,
    };
    let mut report = Report::default();
    assert!(cache.lower(&[1.0, -1.0], deadline(), &mut report).is_none());
    assert_eq!(report.seed_checks, 1);
    assert_eq!(report.fold_terms, 0);
    assert_eq!((cache.completed).len(), 0);
    let margins: Vec<_> = (0..2)
        .map(|row| source.margin(&eng, row, deadline()).unwrap())
        .collect();
    let baseline: Vec<_> = margins
        .iter()
        .map(|(q, b)| source.baseline(q, *b, deadline()).unwrap())
        .collect();
    let (bounds, report) = run(&eng, &source);
    assert_eq!(bounds, baseline);
    assert_eq!(report.fold_terms, 0);
    assert_eq!(report.cells_checked, 0);
    assert!(report.seed_checks > 0 && report.seed_checks <= MAX_SEEDS);
    assert!(bounds[0] <= 0.0 && bounds[1] > 0.0);
}
