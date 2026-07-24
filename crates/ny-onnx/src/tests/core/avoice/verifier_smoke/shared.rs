// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::common;
use ndarray::{ArrayD, IxDyn};
use ny_core::{Bound, MethodUsed, NaiveCpuGemmEngine, VerificationResult};
use ny_propagate::{GraphNetwork, PropagationConfig, PropagationMethod, Verifier};
use ny_tensor::BoundedTensor;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub(super) enum VerifierSmokeRoute {
    Ibp,
    Crown,
}

impl VerifierSmokeRoute {
    fn propagation_method(self) -> PropagationMethod {
        match self {
            Self::Ibp => PropagationMethod::Ibp,
            Self::Crown => PropagationMethod::Crown,
        }
    }

    fn assert_actual_method(self, result: &VerificationResult, label: &str) {
        let expected_actual_method = match self {
            Self::Ibp => MethodUsed::Ibp,
            Self::Crown => MethodUsed::Crown,
        };
        let actual = result.actual_method_tag();
        if matches!(self, Self::Crown) {
            assert!(
                actual.is_some(),
                "{label}: actual_method_tag should be populated, got None"
            );
            assert_eq!(
                actual,
                Some(&expected_actual_method),
                "{label}: expected actual_method_tag={expected_actual_method:?}; if this regresses to \
                 MethodUsed::Ibp, the graph CROWN route likely fell back unexpectedly"
            );
            return;
        }
        assert_eq!(
            actual,
            Some(&expected_actual_method),
            "{label}: expected actual_method_tag={expected_actual_method:?}"
        );
    }
}

fn verified_bounds_tensor(bounds: &[Bound], shape: &[usize], label: &str) -> BoundedTensor {
    let lower = ArrayD::from_shape_vec(
        IxDyn(shape),
        bounds.iter().map(|bound| bound.lower()).collect(),
    )
    .unwrap_or_else(|e| panic!("{label}: verifier lower bounds shape mismatch: {e}"));
    let upper = ArrayD::from_shape_vec(
        IxDyn(shape),
        bounds.iter().map(|bound| bound.upper()).collect(),
    )
    .unwrap_or_else(|e| panic!("{label}: verifier upper bounds shape mismatch: {e}"));
    BoundedTensor::new(lower, upper)
        .unwrap_or_else(|e| panic!("{label}: verifier bounds tensor invalid: {e}"))
}

/// Run one verifier-smoke route. `timeout_ms` is the RELEASE wall-clock
/// budget; debug builds get an effectively unbounded budget instead (see
/// `common::release_budget_ms`) so the verdict assertions still run.
pub(super) fn run_verifier_smoke_route(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    output_bounds: Vec<Bound>,
    timeout_ms: u64,
    route: VerifierSmokeRoute,
    label: &str,
) -> VerificationResult {
    let spec = common::verifier_spec_from_bounded_input(
        input,
        output_bounds,
        common::release_budget_ms(timeout_ms),
    );
    let config = PropagationConfig {
        method: route.propagation_method(),
        ..Default::default()
    };
    let verifier = Verifier::new_with_engine(config, Arc::new(NaiveCpuGemmEngine));
    let result = verifier
        .verify_graph(graph, &spec)
        .unwrap_or_else(|e| panic!("{label} should not error: {e}"));
    route.assert_actual_method(&result, label);
    result
}

pub(super) fn assert_center_contained_in_verified_bounds(
    output_bounds: &[Bound],
    graph: &GraphNetwork,
    input: &BoundedTensor,
    label: &str,
) {
    let center_label = format!("{label} center");
    let concrete = common::evaluate_graph_at_center(graph, input, &center_label);
    let verifier_bounds = verified_bounds_tensor(output_bounds, concrete.shape(), label);
    common::assert_concrete_contained_in_bounds(&concrete, &verifier_bounds, label);
}

pub(super) fn assert_verified_result_contains_center(
    result: &VerificationResult,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    label: &str,
) {
    match result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert_center_contained_in_verified_bounds(output_bounds, graph, input, label);
        }
        _ => panic!(
            "{label}: expected Verified result so center containment can be checked, got {result:?}"
        ),
    }
}
