// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::certify::{TokenScope, UnsatSource};
use crate::margin_row::bab::{BabConfig, MarginRowBab, MarginRowOutcome};
use crate::margin_row::head_group_source::HeadGroupSourceGuard;
use crate::margin_row::net::TwinNet;
use crate::margin_row::root::RootGates;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;
use ny_core::head_phase_advice::{HeadPhaseLpHint, HeadPhaseLpRequest};
use std::sync::atomic::{AtomicUsize, Ordering};

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(10)
}

fn policy() -> HeadGroupPolicy {
    HeadGroupPolicy::new(5.0, 2, 4).unwrap()
}

fn rational(x: f64) -> BigRational {
    BigRational::from_float(x).unwrap()
}

fn relu(x: BigRational) -> BigRational {
    x.max(BigRational::zero())
}

fn contains(interval: I, value: &BigRational) -> bool {
    rational(interval.lo) <= *value && *value <= rational(interval.hi)
}

/// Six independent unstable heads, one stable cancellation head, and two
/// original margins. The second margin is identically one. In the first,
/// each term rho(x-c)-3*x/8 is minimized at x=c.
fn six_spec(positive: bool) -> TwinSpec {
    let mut trunk = vec![0.0; 36];
    let mut head = vec![0.0; 42];
    for i in 0..6 {
        trunk[i * 6 + i] = 1.0;
        head[i * 6 + i] = 1.0;
        head[36 + i] = 1.0;
    }
    let q = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -3.0 / 8.0];
    let bias = if positive { 181.0 / 64.0 } else { 173.0 / 64.0 };
    TwinSpec {
        n_in: 6,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: trunk,
                bias: vec![2.0; 6],
                bias_err: vec![0.0; 6],
                weight_rel_err: 0.0,
                kernel: (6, 6, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (6, 1, 1),
                oshape: (6, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: head,
                bias: vec![-33.0 / 16.0; 6].into_iter().chain([-5.0]).collect(),
                bias_err: vec![0.0; 7],
                weight_rel_err: 0.0,
                shape: (7, 6),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: q.into_iter().chain([0.0; 7]).chain(q).collect(),
                bias: vec![bias, 0.0, bias - 1.0],
                bias_err: vec![0.0; 3],
                weight_rel_err: 0.0,
                shape: (3, 7),
            },
        ],
    }
}

struct AllSlopes {
    negate: bool,
    calls: AtomicUsize,
}

impl HeadPhaseAdvice for AllSlopes {
    fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
        None
    }

    fn supports_functional_advice(&self) -> bool {
        true
    }

    fn propose_functional(
        &self,
        request: &HeadFunctionalLpRequest,
        deadline: Instant,
    ) -> Option<HeadFunctionalLpHint> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        assert!(request.validate());
        assert!(live(deadline));
        assert_eq!((request.phase_rows).len(), 0);
        let sign = if self.negate { -1.0 } else { 1.0 };
        let mut row_duals = vec![0.0; request.rows.len()];
        for &row in request.slope_rows.iter().flatten() {
            let variable = request.rows[row].coefficients.last().unwrap().0;
            row_duals[row] = sign * request.objective[variable] * 3.0 / 8.0;
        }
        Some(HeadFunctionalLpHint { row_duals })
    }
}

fn provider(negate: bool) -> Arc<AllSlopes> {
    Arc::new(AllSlopes {
        negate,
        calls: AtomicUsize::new(0),
    })
}

fn direct_with(
    eng: &BackwardEngine<'_>,
    target: usize,
    adversaries: &[usize],
    provider: &Arc<AllSlopes>,
) -> DirectAttempt {
    let backend: Arc<dyn HeadPhaseAdvice> = provider.clone();
    attempt_with_provider(eng, target, adversaries, policy(), deadline(), || {
        Some(backend)
    })
}

#[test]
fn full_width_direct_y_encloses_exact_fold_and_bias_extremes() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut spec = six_spec(true);
    let rel = 2.0f64.powi(-20);
    let bias_error = 2.0f64.powi(-19);
    if let TwinOpSpec::Gemm {
        weight_rel_err,
        bias_err,
        ..
    } = &mut spec.ops[3]
    {
        *weight_rel_err = rel;
        bias_err.fill(bias_error);
    }
    let net = TwinNet::compile(&spec).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = DirectScalarHeadSource::build(&eng, 0, &[1, 2], deadline()).unwrap();
    let (w, b, (heads, width)) = net.gemm1();
    // Every sampled value and every parameter endpoint is evaluated as a
    // rational binary64 input; the oracle performs no floating dot product.
    for mask in 0..64 {
        let z: Vec<_> = (0..6)
            .map(|i| rational(if mask & (1 << i) == 0 { 1.0 } else { 3.0 }))
            .collect();
        for sign in [-1.0, 1.0] {
            for j in 0..heads {
                let mut exact = rational(b[j]) + rational(sign * bias_error);
                for k in 0..width {
                    let coefficient = rational(w[j * width + k])
                        + rational(sign) * rational(rel) * rational(w[j * width + k].abs());
                    exact += coefficient * &z[k];
                }
                assert!(
                    contains(source.y[j], &exact),
                    "head={j} mask={mask} sign={sign}"
                );
            }
        }
    }
}

#[test]
fn exact_shifted_six_dimensional_minima_and_both_hint_signs_replay_locally() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let center = rational(1.0 / 16.0);
    for positive in [true, false] {
        let net = TwinNet::compile(&six_spec(positive)).unwrap();
        let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
        let eng = BackwardEngine::new(&net, &root);
        let b = if positive { 181.0 / 64.0 } else { 173.0 / 64.0 };
        let exact_min =
            rational(b) - rational(3.0 / 8.0) * (rational(7.0) + rational(6.0) * &center);
        assert_eq!(
            exact_min,
            rational(if positive { 1.0 / 16.0 } else { -1.0 / 16.0 })
        );
        // The scalar theorem is checked at every breakpoint and box endpoint:
        // rho(x-c)-3*x/8 >= -3*c/8 on both affine pieces.
        for x in [-1.0, 0.0, 1.0 / 16.0, 1.0] {
            let x = rational(x);
            assert!(
                relu(&x - &center) - rational(3.0 / 8.0) * &x >= -rational(3.0 / 8.0) * &center
            );
        }
        for negate in [false, true] {
            let backend = provider(negate);
            let out = direct_with(&eng, 0, &[1, 2], &backend);
            let prelude = out.prelude.as_ref().unwrap();
            assert!(rational(prelude.bounds()[0]) <= exact_min);
            assert!(prelude.bounds()[1] > 0.0 && prelude.bounds()[1] <= 1.0);
            assert_eq!(prelude.complete(), positive);
            if positive {
                assert!(prelude.bounds()[0] > 1.0 / 32.0);
            } else {
                assert!(prelude.bounds()[0] <= -1.0 / 16.0);
            }
            assert_eq!(backend.calls.load(Ordering::Relaxed), 1);
            assert_eq!(out.calls, 1);
            assert_eq!(out.passes, 1);
            assert_eq!(out.columns, 1);
        }
    }
}

#[test]
fn all_original_classes_use_the_shared_instance_authorizer_without_a_pack() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&six_spec(true)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let mut out = direct_with(&eng, 0, &[1, 2], &provider(false));
    let cfg = BabConfig {
        deadline: Some(deadline()),
        head_groups: Some(policy()),
        frozen_dual_tier: Some(false),
        ..BabConfig::default()
    };
    let verdict = crate::margin_row::bab::direct_root_outcome(
        &eng,
        0,
        &[1, 2],
        &cfg,
        &[],
        Some(&mut out),
        Instant::now(),
    )
    .unwrap();
    match verdict {
        MarginRowOutcome::Unsat(stats, token) => {
            assert_eq!(stats.stop, "direct_head_closed_at_root");
            assert_eq!(stats.expansions, 0);
            assert_eq!(stats.root_closed_classes, 2);
            assert_eq!(stats.ledger_ok, Some(true));
            assert_eq!(token.scope(), TokenScope::Instance);
            assert_eq!(token.source(), UnsatSource::Authorized);
            assert!(token.covers_all(&[1, 2]));
        }
        MarginRowOutcome::Unknown { reason, .. } => {
            panic!("complete directed cover refused: {reason}")
        }
    }
}

#[test]
fn partial_cover_negative_recipient_and_foreign_order_cannot_publish() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&six_spec(false)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let mut out = direct_with(&eng, 0, &[1, 2], &provider(false));
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
        Instant::now(),
    )
    .is_none());
    let prelude = out.prelude.as_ref().unwrap();
    assert!(!prelude.matches(&eng, 0, &[2, 1], deadline()));
    assert!(!prelude.matches(&eng, 1, &[0, 2], deadline()));
    assert!(!prelude.matches(&eng, 0, &[2], deadline()));
    // A positive donor never transfers its bound or class identity.
    assert!(prelude.bounds()[1] > 0.0);
    assert!(prelude.bounds()[0] < 0.0);
}

#[test]
fn private_generation_refuses_identical_rebuild_and_foreign_network() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let spec = six_spec(true);
    let net = TwinNet::compile(&spec).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let out = direct_with(&eng, 0, &[1, 2], &provider(false));
    let prelude = out.prelude.as_ref().unwrap();
    let rebuilt = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    assert!(!prelude.matches(&BackwardEngine::new(&net, &rebuilt), 0, &[1, 2], deadline()));
    let foreign = TwinNet::compile(&spec).unwrap();
    let foreign_root =
        RootGates::build(&foreign, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    assert!(!prelude.matches(
        &BackwardEngine::new(&foreign, &foreign_root),
        0,
        &[1, 2],
        deadline()
    ));
}

#[test]
fn scalar_intercepts_retain_cross_zero_negative_and_tiny_full_q() {
    for q in [
        I {
            lo: -1.0 / 8.0,
            hi: 1.0,
        },
        I {
            lo: -2.0,
            hi: -1.0 / 8.0,
        },
        I {
            lo: f64::from_bits(1),
            hi: f64::from_bits(4),
        },
    ] {
        let y = I {
            lo: -17.0 / 16.0,
            hi: 15.0 / 16.0,
        };
        for slope in [-3.0 / 8.0, 0.0, 3.0 / 8.0, 1.0] {
            let intercept = scalar_intercept(q, slope, y).unwrap();
            for coefficient in [q.lo, q.hi] {
                for value in [y.lo, 0.0, y.hi] {
                    let lhs = rational(slope) * rational(value) + rational(intercept);
                    let rhs = rational(coefficient) * relu(rational(value));
                    assert!(lhs <= rhs, "q={q:?} slope={slope} y={value}");
                }
            }
        }
    }
}

#[test]
fn positive_underflow_before_large_head_amplifier_is_charged_in_direct_y() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let eta = f64::from_bits(1);
    let amplifier = 2.0f64.powi(256);
    let spec = TwinSpec {
        n_in: 4,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![0.0; 16],
                bias: vec![3.0 / 8.0; 4],
                bias_err: vec![0.0; 4],
                weight_rel_err: 0.0,
                kernel: (4, 4, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (4, 1, 1),
                oshape: (4, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: vec![eta, eta, eta, eta, 0.0, 0.0, 0.0, 0.0],
                bias: vec![0.0; 2],
                bias_err: vec![0.0; 2],
                weight_rel_err: 0.0,
                shape: (2, 4),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![-amplifier, 0.0, 0.0, 0.0],
                bias: vec![0.0; 2],
                bias_err: vec![0.0; 2],
                weight_rel_err: 0.0,
                shape: (2, 2),
            },
        ],
    };
    let net = TwinNet::compile(&spec).unwrap();
    let root = RootGates::build(&net, &[0.0; 4], &[0.0; 4], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = DirectScalarHeadSource::build(&eng, 0, &[1], deadline()).unwrap();
    let real_y = BigRational::new(BigInt::from(3), BigInt::from(2)) * rational(eta);
    assert!(contains(source.y[0], &real_y));
    assert!(source.y[0].hi > 0.0);
    let (q, b) = source.margin(&eng, 0, deadline()).unwrap();
    let lower = source.baseline(&q, b, deadline()).unwrap();
    let exact_margin = -rational(amplifier) * real_y;
    assert!(rational(lower) <= exact_margin);
    assert!(
        lower < 0.0,
        "RN nominal zero must not erase the absolute charge"
    );
}

#[test]
fn expired_authority_refuses_before_provider_and_after_complete_cover() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&six_spec(true)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let calls = AtomicUsize::new(0);
    let expired = attempt_with_provider(&eng, 0, &[1, 2], policy(), Instant::now(), || {
        calls.fetch_add(1, Ordering::Relaxed);
        None
    });
    assert!(expired.prelude.is_none());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let mut completed = direct_with(&eng, 0, &[1, 2], &provider(false));
    let cfg = BabConfig {
        deadline: Some(Instant::now()),
        head_groups: Some(policy()),
        ..BabConfig::default()
    };
    assert!(crate::margin_row::bab::direct_root_outcome(
        &eng,
        0,
        &[1, 2],
        &cfg,
        &[],
        Some(&mut completed),
        Instant::now(),
    )
    .is_none());
    let cfg = BabConfig {
        deadline: Some(deadline()),
        publish_deadline: Some(Instant::now()),
        head_groups: Some(policy()),
        ..BabConfig::default()
    };
    assert!(crate::margin_row::bab::direct_root_outcome(
        &eng,
        0,
        &[1, 2],
        &cfg,
        &[],
        Some(&mut completed),
        Instant::now(),
    )
    .is_none());
}

#[test]
fn malformed_or_panicking_advice_keeps_complete_baselines() {
    // Capability refusal preserves the complete production native portfolio;
    // it supplies no proposal that could alter the expected certificate.
    struct Unsupported(AtomicUsize);
    impl HeadPhaseAdvice for Unsupported {
        fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
            self.0.fetch_add(1, Ordering::Relaxed);
            panic!("unsupported baseline must never propose");
        }
    }
    struct Broken {
        panic: bool,
        calls: AtomicUsize,
        valid_requests: AtomicUsize,
    }
    impl HeadPhaseAdvice for Broken {
        fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
            None
        }
        fn supports_functional_advice(&self) -> bool {
            true
        }
        fn propose_functional(
            &self,
            request: &HeadFunctionalLpRequest,
            deadline: Instant,
        ) -> Option<HeadFunctionalLpHint> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if request.validate() && live(deadline) {
                self.valid_requests.fetch_add(1, Ordering::Relaxed);
            }
            assert!(!self.panic, "untrusted optional backend");
            Some(HeadFunctionalLpHint {
                row_duals: vec![f64::NAN],
            })
        }
    }
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&six_spec(false)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let source = DirectScalarHeadSource::build(&eng, 0, &[1, 2], deadline()).unwrap();
    let scalar_baseline: Vec<_> = (0..2)
        .map(|row| {
            let (q, b) = source.margin(&eng, row, deadline()).unwrap();
            source.baseline(&q, b, deadline()).unwrap()
        })
        .collect();
    // Use identical policy/caps and one original authority deadline for all
    // attempts. Native transport/adaptive can strengthen the scalar floor
    // before the provider is considered, so that floor alone is not the
    // fault-injection oracle.
    let same_policy = policy();
    let authority = deadline();
    let unsupported = Arc::new(Unsupported(AtomicUsize::new(0)));
    let backend: Arc<dyn HeadPhaseAdvice> = unsupported.clone();
    let baseline =
        attempt_with_provider(&eng, 0, &[1, 2], same_policy, authority, || Some(backend));
    let expected = baseline.prelude.as_ref().unwrap().bounds().to_vec();
    assert!(expected[0] > scalar_baseline[0]);
    assert!(expected.iter().zip(&scalar_baseline).all(|(a, b)| a >= b));
    assert_eq!(unsupported.0.load(Ordering::Relaxed), 0);
    assert_eq!(baseline.budget.calls, 0);
    assert_eq!(baseline.calls, 0);
    assert_eq!(baseline.columns, 0);
    assert_eq!(baseline.passes, 0);
    assert!(!baseline.prelude.as_ref().unwrap().complete());
    assert!(baseline.transport.candidates > 0);
    let native_counts = (
        baseline.transport.pairs_checked,
        baseline.transport.candidates,
        baseline.transport.classes_improved,
        baseline.transport.seed_checks,
        baseline.adaptive.contraction_counts(),
    );
    for panic in [false, true] {
        let broken = Arc::new(Broken {
            panic,
            calls: AtomicUsize::new(0),
            valid_requests: AtomicUsize::new(0),
        });
        let backend: Arc<dyn HeadPhaseAdvice> = broken.clone();
        let out = attempt_with_provider(&eng, 0, &[1, 2], same_policy, authority, || Some(backend));
        let actual = out.prelude.as_ref().unwrap().bounds();
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            (
                out.transport.pairs_checked,
                out.transport.candidates,
                out.transport.classes_improved,
                out.transport.seed_checks,
                out.adaptive.contraction_counts(),
            ),
            native_counts
        );
        assert_eq!(broken.calls.load(Ordering::Relaxed), 1);
        assert_eq!(broken.valid_requests.load(Ordering::Relaxed), 1);
        assert_eq!(out.budget.calls, 1);
        assert_eq!(out.calls, 1);
        assert_eq!(out.columns, 0);
        assert_eq!(out.passes, 0);
        assert!(!out.prelude.as_ref().unwrap().complete());
    }
}

#[test]
fn unsupported_capability_preserves_v1_calls_and_charges_attempt_wall() {
    struct V1Only;
    impl HeadPhaseAdvice for V1Only {
        fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
            None
        }
    }
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&six_spec(false)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let backend: Arc<dyn HeadPhaseAdvice> = Arc::new(V1Only);
    let out = attempt_with_provider(&eng, 0, &[1, 2], policy(), deadline(), || Some(backend));
    assert_eq!(out.budget.calls, 0);
    assert!(out.budget.charged > Duration::ZERO);
    assert!(out
        .budget
        .call_deadline(Instant::now(), deadline())
        .is_some());
    assert_eq!(out.calls, 0);
    assert!(out.prelude.is_some());
}

#[test]
fn actual_joint_and_classwise_entries_close_direct_baseline_before_identity_root() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut spec = six_spec(true);
    if let TwinOpSpec::Gemm { weight, bias, .. } = &mut spec.ops[5] {
        weight.fill(0.0);
        *bias = vec![1.0, 0.0, 0.0];
    }
    let net = TwinNet::compile(&spec).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    for classwise in [false, true] {
        let cfg = BabConfig {
            deadline: Some(deadline()),
            head_groups: Some(policy()),
            frozen_dual_tier: Some(false),
            ..BabConfig::default()
        };
        let verdict = if classwise {
            MarginRowBab::run_classwise(&net, &root, 0, &[1, 2], cfg)
        } else {
            MarginRowBab::run(&net, &root, 0, &[1, 2], cfg)
        };
        match verdict {
            MarginRowOutcome::Unsat(stats, token) => {
                assert_eq!(stats.stop, "direct_head_closed_at_root");
                assert_eq!(stats.expansions, 0);
                assert_eq!(stats.root_closed_classes, 2);
                assert!(token.publishable() && token.covers_all(&[1, 2]));
            }
            MarginRowOutcome::Unknown { reason, .. } => {
                panic!("direct all-margin baseline did not close: {reason}")
            }
        }
    }
}

#[test]
fn rejected_tail_cannot_invoke_provider_or_create_direct_authority() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut spec = six_spec(true);
    // Remove the trunk ReLU from the input dependency, while leaving a valid
    // twin-wall shape. A frozen preactivation box is not a direct feature box.
    if let TwinOpSpec::Flatten { input } = &mut spec.ops[2] {
        *input = 1;
    }
    let net = TwinNet::compile(&spec).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let constructed = AtomicUsize::new(0);
    let out = attempt_with_provider(&eng, 0, &[1, 2], policy(), deadline(), || {
        constructed.fetch_add(1, Ordering::Relaxed);
        None
    });
    assert!(out.prelude.is_none());
    assert_eq!(constructed.load(Ordering::Relaxed), 0);
}

#[test]
fn cross_margin_candidates_recompute_recipient_q_bias_and_order() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut spec = six_spec(true);
    if let TwinOpSpec::Gemm {
        weight,
        bias,
        bias_err,
        shape,
        ..
    } = &mut spec.ops[5]
    {
        let target = weight[..7].to_vec();
        let b = bias[0];
        // Recipient margins are F, 2F and 2F-1/4. The last is unsafe;
        // passing the first row's positive bound to it would be unsound.
        *weight = target
            .iter()
            .copied()
            .chain([0.0; 7])
            .chain(target.iter().map(|q| -*q))
            .chain(target.iter().map(|q| -*q))
            .collect();
        *bias = vec![b, 0.0, -b, -b + 1.0 / 4.0];
        *bias_err = vec![0.0; 4];
        *shape = (4, 7);
    }
    let net = TwinNet::compile(&spec).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let out = direct_with(&eng, 0, &[3, 1, 2], &provider(false));
    let prelude = out.prelude.as_ref().unwrap();
    assert!(prelude.bounds()[0] <= -1.0 / 8.0);
    assert!(prelude.bounds()[1] > 1.0 / 32.0 && prelude.bounds()[1] <= 1.0 / 16.0);
    assert!(prelude.bounds()[2] > 1.0 / 16.0 && prelude.bounds()[2] <= 1.0 / 8.0);
    assert!(!prelude.complete());
    assert_eq!(out.calls, 1);
    assert_eq!(out.passes, 1);
    assert_eq!(out.columns, 3);
}

#[test]
fn large_authored_bias_uncertainty_defeats_nominal_cancellation() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut spec = six_spec(true);
    if let TwinOpSpec::Gemm { bias_err, .. } = &mut spec.ops[3] {
        bias_err.fill(1.0);
    }
    let net = TwinNet::compile(&spec).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let out = direct_with(&eng, 0, &[1, 2], &provider(false));
    // Choose admissible bias errors -1 for unstable heads and +1 for the
    // stable head. At x_i=c all unstable outputs are zero, and margin=-5/16.
    let witness =
        rational(181.0 / 64.0) - rational(3.0 / 8.0) * (rational(8.0) + rational(6.0 / 16.0));
    assert_eq!(witness, rational(-5.0 / 16.0));
    let prelude = out.prelude.as_ref().unwrap();
    assert!(rational(prelude.bounds()[0]) <= witness);
    assert!(!prelude.complete());
}

#[test]
fn exhausted_direct_compute_cannot_restart_the_postpack_five_second_slice() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&six_spec(false)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let mut out = direct_with(&eng, 0, &[1, 2], &provider(false));
    out.charge_compute(Duration::from_secs(5));
    let pack = crate::margin_row::bab::build_pack(&eng, None, &[]).unwrap();
    let existing = out.prelude.as_ref().unwrap().bounds().to_vec();
    let prior_calls = out.budget.calls;
    assert!(out
        .refine_existing(&eng, &pack, 0, &[1, 2], &existing, policy(), deadline())
        .is_none());
    assert_eq!(out.budget.calls, prior_calls);
}

#[test]
fn changed_nonfinite_network_cannot_reuse_frozen_source_generation() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let mut net = TwinNet::compile(&six_spec(true)).unwrap();
    let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let head_index = net.i_gemm1;
    if let TwinOp::Gemm { weight, .. } = &mut net.ops[head_index] {
        weight[0] = f64::NAN;
    }
    let eng = BackwardEngine::new(&net, &root);
    assert!(DirectScalarHeadSource::build(&eng, 0, &[1, 2], deadline()).is_none());
}

#[test]
fn nine_open_reordered_margins_share_one_hint_and_one_complete_native_batch() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let scales = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 0.75];
    let order = [9, 4, 1, 8, 2, 7, 3, 6, 5];
    for negative_recipient in [false, true] {
        let mut spec = six_spec(true);
        if let TwinOpSpec::Gemm {
            weight,
            bias,
            bias_err,
            shape,
            ..
        } = &mut spec.ops[5]
        {
            let target = weight[..7].to_vec();
            let b = bias[0];
            *weight = target.clone();
            *bias = vec![b];
            for &scale in &scales {
                weight.extend(target.iter().map(|q| (1.0 - scale) * q));
                bias.push((1.0 - scale) * b);
            }
            if negative_recipient {
                // The original class at slot zero now has a true negative
                // minimum. No donor endpoint may conceal this obligation.
                bias[order[0]] += 0.75;
            }
            *bias_err = vec![0.0; 10];
            *shape = (10, 7);
        }
        let net = TwinNet::compile(&spec).unwrap();
        let root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
        let eng = BackwardEngine::new(&net, &root);
        let source = DirectScalarHeadSource::build(&eng, 0, &order, deadline()).unwrap();
        for row in 0..9 {
            let (q, b) = source.margin(&eng, row, deadline()).unwrap();
            assert!(source.baseline(&q, b, deadline()).unwrap() <= 0.0);
        }
        let backend = provider(false);
        let mut out = direct_with(&eng, 0, &order, &backend);
        assert_eq!(backend.calls.load(Ordering::Relaxed), 1);
        assert_eq!(out.calls, 1);
        assert_eq!(out.passes, 1);
        assert_eq!(out.columns, 9);
        let prelude = out.prelude.as_ref().unwrap();
        assert_eq!(prelude.complete(), !negative_recipient);
        for (row, &class) in order.iter().enumerate() {
            let scale = scales[class - 1];
            let exact = rational(scale / 16.0)
                - if negative_recipient && row == 0 {
                    rational(0.75)
                } else {
                    BigRational::zero()
                };
            assert!(rational(prelude.bounds()[row]) <= exact);
            if !(negative_recipient && row == 0) {
                assert!(prelude.bounds()[row] > scale / 32.0);
            }
        }
        let cfg = BabConfig {
            deadline: Some(deadline()),
            head_groups: Some(policy()),
            ..BabConfig::default()
        };
        let published = crate::margin_row::bab::direct_root_outcome(
            &eng,
            0,
            &order,
            &cfg,
            &[],
            Some(&mut out),
            Instant::now(),
        );
        if negative_recipient {
            assert!(published.is_none());
        } else {
            let Some(MarginRowOutcome::Unsat(stats, token)) = published else {
                panic!("every one of the nine original margins must authorize");
            };
            assert_eq!(token.scope(), TokenScope::Instance);
            assert_eq!(stats.root_closed_classes, 9);
            assert_eq!(stats.expansions, 0);
            assert_eq!(stats.ledger_ok, Some(true));
        }
    }
}

#[test]
fn stale_gate_or_generation_discards_only_numeric_prelude_and_keeps_spent_budgets() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&six_spec(false)).unwrap();
    let mut root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let mut first = direct_with(
        &BackwardEngine::new(&net, &root),
        0,
        &[1, 2],
        &provider(false),
    );
    let mut second = direct_with(
        &BackwardEngine::new(&net, &root),
        0,
        &[1, 2],
        &provider(false),
    );
    assert_eq!(first.budget.calls, 1);
    assert_eq!(second.budget.calls, 1);
    let spent = first.elapsed;
    let charged = first.budget.charged;
    let old = root.layers[0].alpha[0];
    root.layers[0].alpha[0] = if old == 0.0 { 0.5 } else { 0.0 };
    root.layers[0].ms[0] = root.layers[0].alpha[0].max(root.layers[0].s[0]);
    first.retain_current_frame(
        &BackwardEngine::new(&net, &root),
        0,
        &[1, 2],
        Some(deadline()),
    );
    assert!(first.prelude.is_none());
    assert_eq!(first.budget.calls, 1);
    assert_eq!(first.budget.charged, charged);
    assert!(first.elapsed >= spent);
    // A newly built frame refuses even if it restores the original bytes.
    let rebuilt = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let spent = second.elapsed;
    let charged = second.budget.charged;
    second.retain_current_frame(
        &BackwardEngine::new(&net, &rebuilt),
        0,
        &[1, 2],
        Some(deadline()),
    );
    assert!(second.prelude.is_none());
    assert_eq!(second.budget.calls, 1);
    assert_eq!(second.budget.charged, charged);
    assert!(second.elapsed >= spent);
    second.charge_compute(Duration::from_secs(5));
    let eng = BackwardEngine::new(&net, &rebuilt);
    let pack = crate::margin_row::bab::build_pack(&eng, None, &[]).unwrap();
    assert!(second
        .refine_existing(&eng, &pack, 0, &[1, 2], &[0.0, 1.0], policy(), deadline())
        .is_none());
    assert_eq!(second.budget.calls, 1);
}

#[test]
fn exact_gate_rollback_retains_the_same_private_generation_prelude() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&six_spec(false)).unwrap();
    let mut root = RootGates::build(&net, &[-1.0; 6], &[1.0; 6], RoundMode::Outward, None).unwrap();
    let mut out = direct_with(
        &BackwardEngine::new(&net, &root),
        0,
        &[1, 2],
        &provider(false),
    );
    let alpha = root.layers[0].alpha[0];
    let ms = root.layers[0].ms[0];
    root.layers[0].alpha[0] = 0.5;
    root.layers[0].ms[0] = 0.5_f64.max(root.layers[0].s[0]);
    root.layers[0].alpha[0] = alpha;
    root.layers[0].ms[0] = ms;
    let spent = out.elapsed;
    out.retain_current_frame(
        &BackwardEngine::new(&net, &root),
        0,
        &[1, 2],
        Some(deadline()),
    );
    assert!(out.prelude.is_some());
    assert_eq!(out.budget.calls, 1);
    assert!(out.elapsed >= spent);
}
