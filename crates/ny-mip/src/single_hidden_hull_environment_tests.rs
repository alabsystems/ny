// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Rejection wiring controls do not mutate hardware floating-point state.
//! The shared ny-core guard separately tests its actual environment probes.
//! Supported-environment arithmetic controls below use exact original dyadics.

use super::*;
use bab::{certify_hull_node_lower_bound, hull_bab, HullBabOptions, HullBabStop, HullSplit};
use num_rational::BigRational;
use num_traits::Zero;
use std::cell::Cell;

thread_local! {
    static REFUSE_ENVIRONMENT: Cell<bool> = const { Cell::new(false) };
}

pub(super) fn environment_refused_for_test() -> bool {
    REFUSE_ENVIRONMENT.with(Cell::get)
}

struct RefusalScope(bool);
impl RefusalScope {
    fn new(refuse: bool) -> Self {
        Self(REFUSE_ENVIRONMENT.with(|state| state.replace(refuse)))
    }
}
impl Drop for RefusalScope {
    fn drop(&mut self) {
        REFUSE_ENVIRONMENT.with(|state| state.set(self.0));
    }
}

fn simple_problem(coefficient: f64) -> Result<HullMarginProblem, HullDecline> {
    HullMarginProblem::new(
        vec![1.0],
        vec![0.0],
        1,
        &[coefficient],
        &[0.25],
        1,
        &[(0, 1.0)],
        0.0,
        vec![-1.0],
        vec![1.0],
    )
}

#[test]
fn hull_environment_constructor_refuses_before_original_arithmetic() {
    let _refusal = RefusalScope::new(true);
    assert!(matches!(
        simple_problem(1.0),
        Err(HullDecline::UnsupportedEnvironment)
    ));
    assert_eq!(
        HullDecline::UnsupportedEnvironment.label(),
        "unsupported_environment"
    );
}

#[test]
fn hull_environment_fresh_root_and_signed_node_replays_refuse() {
    let problems = [simple_problem(1.0).unwrap(), simple_problem(-1.0).unwrap()];
    let _refusal = RefusalScope::new(true);
    for problem in &problems {
        let multipliers = HullMultipliers::zeros(problem);
        assert!(certify_hull_lower_bound(problem, &multipliers).is_none());
        for active in [false, true] {
            assert!(certify_hull_node_lower_bound(
                problem,
                &[HullSplit { neuron: 0, active }],
                &multipliers,
                &[f64::from_bits(1)],
            )
            .is_none());
        }
    }
}

#[test]
fn hull_environment_root_and_bab_never_publish_an_unsupported_positive_report() {
    let problem = simple_problem(1.0).unwrap();
    let _refusal = RefusalScope::new(true);
    let deadline = Instant::now() + Duration::from_secs(20);
    let root = hull_root_bound(&problem, deadline);
    assert!(!root.verified());
    assert_eq!(root.lp_status, HullLpStatus::UnsupportedEnvironment);
    assert!(root.certified.is_none() && root.route.is_none() && root.lp_value.is_none());
    assert_eq!(root.subgradient_iterations, 0);

    let tree = hull_bab(&problem, deadline, HullBabOptions::default());
    assert!(!tree.verified());
    assert_eq!(tree.stop, HullBabStop::UnsupportedEnvironment);
    assert_eq!(tree.root_lp_status, HullLpStatus::UnsupportedEnvironment);
    assert!(tree.root_certified.is_none() && tree.root_lp_value.is_none());
    assert_eq!(tree.nodes, 0);
    assert_eq!(tree.closed, 0);
    assert_eq!(tree.open, 1);
}

#[test]
fn hull_environment_rechecks_after_proposer_before_accepting_its_positive_result() {
    let _scope = RefusalScope::new(false);
    let called = Cell::new(false);
    let outcome = guarded_hull_proposal(|| {
        called.set(true);
        // Simulate a foreign proposer returning under an unsupported mode.
        REFUSE_ENVIRONMENT.with(|state| state.set(true));
        f64::MAX
    });
    assert!(called.get());
    assert_eq!(outcome, Err(HullDecline::UnsupportedEnvironment));
}

#[test]
fn hull_environment_unsupported_entry_never_calls_the_proposer() {
    let _refusal = RefusalScope::new(true);
    let called = Cell::new(false);
    let outcome = guarded_hull_proposal(|| {
        called.set(true);
        1.0
    });
    assert_eq!(outcome, Err(HullDecline::UnsupportedEnvironment));
    assert!(!called.get());
}

#[test]
fn hull_environment_supported_tiny_signed_duals_match_original_rational_controls() {
    ny_core::require_f64_interval_proof_environment().unwrap();
    let tiny = f64::from_bits(1);
    let problem = HullMarginProblem::new(
        vec![f64::from_bits(17)],
        vec![f64::from_bits(3)],
        1,
        &[-1.0],
        &[2.0],
        1,
        &[(0, 1.0)],
        0.0,
        vec![-0.5],
        vec![0.5],
    )
    .unwrap();
    let q = |x: f64| BigRational::from_float(x).unwrap();
    for slope in [tiny, -tiny, f64::from_bits(5), -f64::from_bits(5)] {
        let multipliers = HullMultipliers {
            alpha: vec![0.0],
            g: vec![slope],
        };
        let root = certify_hull_lower_bound(&problem, &multipliers).unwrap();
        assert!(root > 0.0);
        for x in [-0.5, 0.0, 0.5] {
            let z = q(f64::from_bits(17)) * q(x) + q(f64::from_bits(3));
            let original_margin = q(2.0) - z.clone().max(BigRational::zero());
            assert!(q(root) <= original_margin);
            for active in [false, true] {
                if (active && z < BigRational::zero()) || (!active && z > BigRational::zero()) {
                    continue;
                }
                let node = certify_hull_node_lower_bound(
                    &problem,
                    &[HullSplit { neuron: 0, active }],
                    &multipliers,
                    &[tiny],
                )
                .unwrap();
                assert!(q(node) <= original_margin);
            }
        }
    }
}
