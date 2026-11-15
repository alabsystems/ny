// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ny_core::head_functional_advice::HeadFunctionalLpHint;
use ny_core::head_phase_advice::{HeadPhaseLpHint, HeadPhaseLpRequest};
use std::sync::atomic::{AtomicUsize, Ordering};

fn coupled(bias: f32) -> InputHeadSpec {
    InputHeadSpec::new(
        2,
        3,
        vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.5],
        vec![0.0; 3],
        vec![0.5, 0.0, 0.5, 0.0, -1.0, 0.0],
        vec![bias, 0.0],
        [13; 32],
    )
    .unwrap()
}

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(3)
}

fn policy() -> HeadGroupPolicy {
    // Isolate the historical two-call triangle and fault-replay contract.
    HeadGroupPolicy::new(1.0, 1, 1)
        .unwrap()
        .with_directional_cuts(false)
}

#[derive(Clone, Copy)]
enum Fault {
    Valid,
    Missing,
    Nan,
    Panic,
    None,
}

struct Advice {
    sign: f64,
    fault: Fault,
    legacy: AtomicUsize,
    triangle: AtomicUsize,
}

impl Advice {
    fn new(sign: f64, fault: Fault) -> Self {
        Self {
            sign,
            fault,
            legacy: AtomicUsize::new(0),
            triangle: AtomicUsize::new(0),
        }
    }
}

impl HeadPhaseAdvice for Advice {
    fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
        panic!("the input source uses genuine functional requests")
    }

    fn supports_functional_advice(&self) -> bool {
        true
    }

    fn propose_functional(
        &self,
        request: &HeadFunctionalLpRequest,
        deadline: Instant,
    ) -> Option<HeadFunctionalLpHint> {
        assert!(request.validate() && live(deadline));
        assert_eq!(request.feature_count, 2);
        let mut row_duals = vec![0.0; request.rows.len()];
        if request.bounds.len() == 4 {
            self.legacy.fetch_add(1, Ordering::SeqCst);
            // A complete first-call replay is deliberately left open. It
            // remains the control floor when the second hint is malformed.
            return Some(HeadFunctionalLpHint { row_duals });
        }
        assert_eq!(request.bounds.len(), 5);
        assert!(request.slope_rows.iter().all(Option::is_some));
        assert!(request.objective[4] < 0.0); // negative head is a real variable
        self.triangle.fetch_add(1, Ordering::SeqCst);
        match self.fault {
            Fault::Panic => panic!("second-call provider fault"),
            Fault::None => return None,
            Fault::Missing => {
                row_duals.pop();
            }
            Fault::Nan => row_duals[0] = f64::NAN,
            Fault::Valid => {
                // For this authored source the last row is the independently
                // priced midpoint Jensen cut; no imported objective is used.
                let cut = request.rows.last().unwrap();
                assert_eq!(cut.coefficients.len(), 3);
                assert!(cut.coefficients[0].1 > 0.0);
                assert!(cut.coefficients[1].1 > 0.0);
                assert!(cut.coefficients[2].1 < 0.0);
                *row_duals.last_mut().unwrap() = self.sign;
            }
        }
        Some(HeadFunctionalLpHint { row_duals })
    }
}

fn run(spec: &InputHeadSpec, advice: &Arc<Advice>) -> InputHeadOutcome {
    verify_with_provider(
        spec,
        &[-1.0; 2],
        &[1.0; 2],
        0,
        1,
        policy(),
        deadline(),
        || Some(Arc::clone(advice) as Arc<dyn HeadPhaseAdvice>),
    )
}

#[test]
fn source_coupling_closes_after_the_real_legacy_call_and_original_singleton_authorizer() {
    let spec = coupled(1.0 / 16.0);
    let source = Source::build(&spec, &[-1.0; 2], &[1.0; 2], 0, 1, deadline()).unwrap();
    assert!(source.baseline(deadline()).unwrap() < 0.0);
    assert!(
        source
            .replay(&source.slopes().unwrap(), deadline())
            .unwrap()
            < 0.0
    );
    for sign in [1.0, -1.0] {
        let advice = Arc::new(Advice::new(sign, Fault::Valid));
        let InputHeadOutcome::Unsat(report, token) = run(&spec, &advice) else {
            panic!("owned midpoint cut did not close")
        };
        let lower = report.lower.unwrap();
        assert!(lower > 1.0 / 32.0 && lower <= 1.0 / 16.0);
        assert_eq!(token.scope(), TokenScope::Instance);
        assert_eq!(token.classes(), &[1]);
        assert!(token.covers_all(&[1]));
        assert!(!token.covers_all(&[0, 1]));
        assert_eq!(report.advice_calls, 2);
        assert_eq!(report.triangle_advice_calls, 1);
        assert_eq!(report.replayed_candidates, 5);
        assert_eq!(report.triangle_replayed_candidates, 2);
        assert_eq!(report.triangle_variables, 5);
        assert_eq!(report.triangle_rows, 7);
        assert_eq!(report.triangle_cuts, 1);
        assert_eq!(report.triangle_classes_improved, 1);
        assert_eq!(advice.legacy.load(Ordering::SeqCst), 1);
        assert_eq!(advice.triangle.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn malformed_second_hints_preserve_the_exact_completed_same_portfolio_floor() {
    let spec = coupled(1.0 / 16.0);
    let control = Arc::new(Advice::new(1.0, Fault::None));
    let InputHeadOutcome::Unknown(base) = run(&spec, &control) else {
        panic!("legacy-only completed control unexpectedly closed")
    };
    assert_eq!(base.advice_calls, 2);
    assert_eq!(base.replayed_candidates, 3);
    for fault in [Fault::Missing, Fault::Nan, Fault::Panic] {
        let advice = Arc::new(Advice::new(1.0, fault));
        let InputHeadOutcome::Unknown(report) = run(&spec, &advice) else {
            panic!("faulty row dual acquired authority")
        };
        assert_eq!(
            report.lower.unwrap().to_bits(),
            base.lower.unwrap().to_bits()
        );
        assert_eq!(report.advice_calls, 2);
        assert_eq!(report.replayed_candidates, 3);
        assert_eq!(report.triangle_replayed_candidates, 0);
        assert_eq!(report.triangle_classes_improved, 0);
        assert_eq!(advice.legacy.load(Ordering::SeqCst), 1);
        assert_eq!(advice.triangle.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn second_factory_or_capability_panic_cannot_erase_a_completed_legacy_replay() {
    let spec = coupled(1.0 / 16.0);
    let control = Arc::new(Advice::new(1.0, Fault::None));
    let InputHeadOutcome::Unknown(base) = run(&spec, &control) else {
        panic!("control closed")
    };
    let advice = Arc::new(Advice::new(1.0, Fault::Valid));
    let factories = AtomicUsize::new(0);
    let InputHeadOutcome::Unknown(report) = verify_with_provider(
        &spec,
        &[-1.0; 2],
        &[1.0; 2],
        0,
        1,
        policy(),
        deadline(),
        || {
            assert!(
                factories.fetch_add(1, Ordering::SeqCst) != 1,
                "second factory"
            );
            Some(Arc::clone(&advice) as Arc<dyn HeadPhaseAdvice>)
        },
    ) else {
        panic!("factory panic closed")
    };
    assert_eq!(factories.load(Ordering::SeqCst), 2);
    assert_eq!(report.advice_calls, 1);
    assert_eq!(
        report.lower.unwrap().to_bits(),
        base.lower.unwrap().to_bits()
    );

    struct CapabilityFault {
        calls: AtomicUsize,
        legacy: Advice,
    }
    impl HeadPhaseAdvice for CapabilityFault {
        fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
            None
        }
        fn supports_functional_advice(&self) -> bool {
            assert!(
                self.calls.fetch_add(1, Ordering::SeqCst) != 1,
                "second capability"
            );
            true
        }
        fn propose_functional(
            &self,
            r: &HeadFunctionalLpRequest,
            d: Instant,
        ) -> Option<HeadFunctionalLpHint> {
            self.legacy.propose_functional(r, d)
        }
    }
    let capability = Arc::new(CapabilityFault {
        calls: AtomicUsize::new(0),
        legacy: Advice::new(1.0, Fault::Valid),
    });
    let InputHeadOutcome::Unknown(report) = verify_with_provider(
        &spec,
        &[-1.0; 2],
        &[1.0; 2],
        0,
        1,
        policy(),
        deadline(),
        || Some(Arc::clone(&capability) as Arc<dyn HeadPhaseAdvice>),
    ) else {
        panic!("capability panic closed")
    };
    assert_eq!(capability.calls.load(Ordering::SeqCst), 2);
    assert_eq!(report.advice_calls, 1);
    assert_eq!(
        report.lower.unwrap().to_bits(),
        base.lower.unwrap().to_bits()
    );
}

#[test]
fn triangle_provider_arrives_on_tie_and_negative_without_a_false_instance_token() {
    for bias in [0.0, -1.0 / 16.0] {
        let advice = Arc::new(Advice::new(1.0, Fault::Valid));
        let InputHeadOutcome::Unknown(report) = run(&coupled(bias), &advice) else {
            panic!("unsafe boundary was certified")
        };
        assert!(report.lower.unwrap() <= f64::from(bias));
        assert_eq!(report.advice_calls, 2);
        assert_eq!(report.triangle_advice_calls, 1);
        assert_eq!(report.triangle_replayed_candidates, 2);
        assert_eq!(report.triangle_classes_improved, 1);
        assert_eq!(advice.triangle.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn replay_budget_has_no_third_grant_and_preserves_time_for_second_call_replay() {
    let now = Instant::now();
    let outer = now + Duration::from_secs(10);
    let mut budget = AdviceBudget::new();
    budget.calls = 2;
    budget.charge_elapsed(Duration::from_millis(230));
    assert!(budget.call_deadline(now, outer).is_none());
    assert_eq!(
        budget.replay_deadline(now, outer),
        Some(now + Duration::from_millis(20))
    );
    assert_eq!(
        budget.replay_deadline(now, now + Duration::from_millis(7)),
        Some(now + Duration::from_millis(7))
    );
    budget.charge_elapsed(Duration::from_millis(20));
    assert!(budget.replay_deadline(now, outer).is_none());
    assert!(budget.replay_deadline(now, now).is_none());
}
