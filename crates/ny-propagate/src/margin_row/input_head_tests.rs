// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use ny_core::head_functional_advice::HeadFunctionalLpHint;
use ny_core::head_phase_advice::{HeadPhaseLpHint, HeadPhaseLpRequest};

fn ridge(bias: f32) -> InputHeadSpec {
    InputHeadSpec::new(
        2,
        3,
        vec![1.0, -1.0, 0.0, 1.0, 1.0, 1.0],
        vec![1.0 / 16.0, 1.0 / 16.0, 2.0],
        vec![1.0, 0.0, 1.0, 0.0, -0.75, 0.0],
        vec![bias, 0.0],
        [7; 32],
    )
    .unwrap()
}
fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(2)
}
fn policy() -> HeadGroupPolicy {
    HeadGroupPolicy::new(1.0, 1, 1).unwrap()
}
struct Advice {
    sign: f64,
    fault: u8,
}
impl HeadPhaseAdvice for Advice {
    fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
        None
    }
    fn supports_functional_advice(&self) -> bool {
        assert!(self.fault != 3, "capability fault");
        self.fault != 5
    }
    fn propose_functional(
        &self,
        r: &HeadFunctionalLpRequest,
        _: Instant,
    ) -> Option<HeadFunctionalLpHint> {
        assert!(self.fault != 2, "provider fault");
        let mut row_duals = vec![0.0; r.rows.len()];
        for row in r.slope_rows.iter().flatten() {
            row_duals[*row] = self.sign * 3.0 / 8.0;
        }
        if self.fault == 1 {
            row_duals.pop();
        }
        if self.fault == 4 {
            row_duals[0] = f64::NAN;
        }
        Some(HeadFunctionalLpHint { row_duals })
    }
}
fn run(spec: &InputHeadSpec, a: Advice) -> InputHeadOutcome {
    // The provider closure is `Fn`: each call builds a fresh `Advice` from the
    // copied fields (moving `a` itself would make the closure `FnOnce`).
    let Advice { sign, fault } = a;
    verify_with_provider(
        spec,
        &[-1.0; 2],
        &[1.0; 2],
        0,
        1,
        policy(),
        deadline(),
        || Some(Arc::new(Advice { sign, fault })),
    )
}

#[test]
fn owned_dense_ridge_advice_closes_the_original_singleton_with_both_dual_signs() {
    let spec = ridge(97.0 / 64.0);
    let source = Source::build(&spec, &[-1.0; 2], &[1.0; 2], 0, 1, deadline()).unwrap();
    assert!(source.baseline(deadline()).unwrap() < 0.0);
    assert!(
        source
            .replay(&source.slopes().unwrap(), deadline())
            .unwrap()
            < 0.0
    );
    for sign in [1.0, -1.0] {
        let InputHeadOutcome::Unsat(report, token) = run(&spec, Advice { sign, fault: 0 }) else {
            panic!("ridge stayed open")
        };
        assert!(report.lower.unwrap() > 1.0 / 32.0 && report.lower.unwrap() <= 1.0 / 16.0);
        assert_eq!(token.scope(), TokenScope::Instance);
        assert_eq!(report.advice_calls, 1);
    }
}

#[test]
fn negative_and_tie_dense_sources_never_publish_unsat() {
    for bias in [89.0 / 64.0, 93.0 / 64.0] {
        let InputHeadOutcome::Unknown(report) = run(
            &ridge(bias),
            Advice {
                sign: 1.0,
                fault: 0,
            },
        ) else {
            panic!("unsafe/tie closed")
        };
        let exact = bias as f64 - 93.0 / 64.0;
        assert!(report.lower.unwrap() <= exact);
    }
}

#[test]
fn provider_dimension_nan_and_panic_faults_preserve_owned_baseline() {
    let spec = ridge(97.0 / 64.0);
    let InputHeadOutcome::Unknown(base) = verify_with_provider(
        &spec,
        &[-1.0; 2],
        &[1.0; 2],
        0,
        1,
        policy(),
        deadline(),
        || None,
    ) else {
        panic!("native control unexpectedly closed")
    };
    for fault in [1, 2, 3, 4, 5] {
        let InputHeadOutcome::Unknown(report) = run(&spec, Advice { sign: 1.0, fault }) else {
            panic!("fault closed")
        };
        assert_eq!(
            report.lower.unwrap().to_bits(),
            base.lower.unwrap().to_bits()
        );
        if matches!(fault, 3 | 5) {
            assert_eq!(report.advice_calls, 0);
        }
    }
    let InputHeadOutcome::Unknown(factory_fault) = verify_with_provider(
        &spec,
        &[-1.0; 2],
        &[1.0; 2],
        0,
        1,
        policy(),
        deadline(),
        || panic!("factory fault"),
    ) else {
        panic!("factory fault closed")
    };
    assert_eq!(factory_fault.advice_calls, 0);
    assert_eq!(
        factory_fault.lower.unwrap().to_bits(),
        base.lower.unwrap().to_bits()
    );
}

#[test]
fn f32_input_cast_and_first_layer_rounding_are_owned_before_replay() {
    let spec = InputHeadSpec::new(
        1,
        1,
        vec![1.0],
        vec![0.0],
        vec![1.0, 0.0],
        vec![0.0; 2],
        [0; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[0.1], &[0.1], 0, 1, deadline()).unwrap();
    let executed = f64::from(0.1f32);
    assert!(s.y[0].lo <= executed && s.y[0].hi >= executed);
    assert!(s.first_error[0] > 0.0 && s.output_error > 0.0);
    assert!(s.replay(&[1.0], deadline()).unwrap() <= executed);
}

#[test]
fn rounded_zero_subnormal_product_is_charged_before_large_final_amplification() {
    let amplifier = f32::from_bits(227 << 23); // exact2^100
    let spec = InputHeadSpec::new(
        1,
        1,
        vec![f32::from_bits(1)],
        vec![0.0],
        vec![amplifier, 0.0],
        vec![0.0; 2],
        [0; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[0.5], &[0.5], 0, 1, deadline()).unwrap();
    assert_eq!(f32::from_bits(1) * 0.5, 0.0);
    assert!(s.first_error[0] >= F32_ETA);
    assert!(s.replay(&[f64::from(amplifier)], deadline()).unwrap() <= 0.0);
}

#[test]
fn complete_box_last_feature_and_source_copy_are_retained() {
    let mut spec = InputHeadSpec::new(
        2,
        1,
        vec![0.0, 1.0],
        vec![0.0],
        vec![-1.0, 0.0],
        vec![0.5, 0.0],
        [0; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[0.0, -1.0], &[0.0, 1.0], 0, 1, deadline()).unwrap();
    assert!(s.baseline(deadline()).unwrap() <= -0.5);
    let bound = s.replay(&[-0.5], deadline()).unwrap();
    spec.first[1] = 0.0; // producer snapshot cannot be changed by its source caller
    assert_eq!(
        bound.to_bits(),
        s.replay(&[-0.5], deadline()).unwrap().to_bits()
    );
    let other = Source::build(&spec, &[0.0, -1.0], &[0.0, 1.0], 0, 1, deadline()).unwrap();
    assert!(!Arc::ptr_eq(&s.spec, &other.spec));
    assert!(other.baseline(deadline()).unwrap() > 0.0);
}

#[test]
fn expiry_overflow_wrong_classes_and_crossed_boxes_fail_closed() {
    let spec = ridge(97.0 / 64.0);
    assert!(Source::build(&spec, &[-1.0; 2], &[1.0; 2], 0, 1, Instant::now()).is_none());
    assert!(Source::build(&spec, &[2.0; 2], &[1.0; 2], 0, 1, deadline()).is_none());
    assert!(Source::build(&spec, &[-1.0; 2], &[1.0; 2], 0, 0, deadline()).is_none());
    let overflow = InputHeadSpec::new(
        1,
        1,
        vec![f32::MAX],
        vec![0.0],
        vec![1.0, 0.0],
        vec![0.0; 2],
        [0; 32],
    )
    .unwrap();
    assert!(Source::build(&overflow, &[1.0], &[1.0], 0, 1, deadline()).is_none());
    let s = Source::build(&spec, &[-1.0; 2], &[1.0; 2], 0, 1, deadline()).unwrap();
    assert!(s.replay(&[0.375, 0.375, -0.75], Instant::now()).is_none());
}

#[test]
fn genuine_f32_g2_checks_the_complete_margin_claim_and_head_boxes() {
    let s = Source::build(&ridge(97.0 / 64.0), &[-1.0; 2], &[1.0; 2], 0, 1, deadline()).unwrap();
    let points = Array2::from_shape_vec((2, 1), vec![0.0, -1.0 / 16.0]).unwrap();
    let mut audit = ContainmentAudit::default();
    Frame {
        source: &s,
        lower: 0.2,
        deadline: deadline(),
    }
    .audit(&points, &mut audit)
    .unwrap();
    assert!(audit.violations() > 0);
}
