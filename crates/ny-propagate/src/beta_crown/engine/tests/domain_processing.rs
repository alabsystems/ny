// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::prelude::*;
use crate::beta_crown::domain::DomainProcessingConfig;

fn domain_processing_verifier() -> BetaCrownVerifier {
    BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::Sequential,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        timeout: Duration::from_secs(5),
        ..Default::default()
    })
}

fn test_input_bounds() -> BoundedTensor {
    BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap()
}

fn unstable_relu_domain() -> BabDomain {
    let pre_relu_bounds =
        BoundedTensor::new(arr1(&[-1.0, -0.5]).into_dyn(), arr1(&[1.0, 0.5]).into_dyn()).unwrap();
    let post_relu_bounds =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 0.5]).into_dyn()).unwrap();
    let output_bounds =
        BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.5]).into_dyn()).unwrap();

    BabDomain::root(
        vec![pre_relu_bounds, post_relu_bounds, output_bounds],
        0.0,
        1.5,
    )
    .unwrap()
}

fn stable_relu_domain() -> BabDomain {
    let pre_relu_bounds =
        BoundedTensor::new(arr1(&[0.2, 0.4]).into_dyn(), arr1(&[1.0, 1.2]).into_dyn()).unwrap();
    let post_relu_bounds =
        BoundedTensor::new(arr1(&[0.2, 0.4]).into_dyn(), arr1(&[1.0, 1.2]).into_dyn()).unwrap();
    let output_bounds =
        BoundedTensor::new(arr1(&[0.6]).into_dyn(), arr1(&[2.2]).into_dyn()).unwrap();

    BabDomain::root(
        vec![pre_relu_bounds, post_relu_bounds, output_bounds],
        0.6,
        2.2,
    )
    .unwrap()
}

fn child_for_activity(children: &[BabDomain], is_active: bool) -> &BabDomain {
    children
        .iter()
        .find(|child| {
            child
                .history
                .constraints
                .last()
                .is_some_and(|constraint| constraint.is_active() == is_active)
        })
        .unwrap_or_else(|| panic!("missing child with is_active={is_active}"))
}

#[ntest::timeout(5000)]
#[test]
fn test_process_domain_sequential_reports_no_branch_for_stable_domain_3089() {
    let network = simple_network();
    let input = test_input_bounds();
    let domain = stable_relu_domain();
    let verifier = domain_processing_verifier();
    let mut cut_pool = CutPool::new(0);

    let result = verifier.process_domain_sequential(
        &network,
        &input,
        &domain,
        0.5,
        &mut cut_pool,
        None,
        None,
    );

    assert!(
        result.children.is_empty(),
        "stable domain should not emit children"
    );
    assert!(
        result.had_no_branch,
        "stable domain should be marked unresolved due to no unstable neurons"
    );
    assert!(
        !result.had_propagation_failure,
        "no-branch path should not be reported as a propagation failure"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_process_domain_sequential_creates_relu_children_3089() {
    let network = simple_network();
    let input = test_input_bounds();
    let domain = unstable_relu_domain();
    let verifier = domain_processing_verifier();
    let mut cut_pool = CutPool::new(0);

    let result = verifier.process_domain_sequential(
        &network,
        &input,
        &domain,
        0.5,
        &mut cut_pool,
        None,
        None,
    );

    assert_eq!(
        result.children.len(),
        2,
        "one unstable neuron should yield two children"
    );
    assert!(
        !result.had_no_branch,
        "unstable domain should not take the no-branch path"
    );
    assert!(
        !result.had_propagation_failure,
        "successful child creation should not report propagation failure"
    );

    let active_child = child_for_activity(&result.children, true);
    let inactive_child = child_for_activity(&result.children, false);

    let active_constraint = active_child.history.constraints.last().unwrap();
    assert_eq!(active_constraint.layer_idx(), 1);
    assert_eq!(active_constraint.neuron_idx(), 0);
    assert_eq!(active_child.depth(), 1);

    let active_bounds = active_child.layer_bounds[0].flatten();
    assert!(
        active_bounds.lower()[[0]] >= 0.0,
        "active split must clamp the constrained neuron lower bound to >= 0"
    );

    let inactive_constraint = inactive_child.history.constraints.last().unwrap();
    assert_eq!(inactive_constraint.layer_idx(), 1);
    assert_eq!(inactive_constraint.neuron_idx(), 0);
    assert_eq!(inactive_child.depth(), 1);

    let inactive_bounds = inactive_child.layer_bounds[0].flatten();
    assert!(
        inactive_bounds.upper()[[0]] <= 0.0,
        "inactive split must clamp the constrained neuron upper bound to <= 0"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_process_domain_parallel_creates_children_when_enabled_3089() {
    let network = simple_network();
    let input = test_input_bounds();
    let domain = unstable_relu_domain();
    let verifier = domain_processing_verifier();
    let mut cut_pool = CutPool::new(0);
    let config = DomainProcessingConfig::new(0.5, true);

    let result =
        verifier.process_domain_parallel(&network, &input, &domain, &config, &mut cut_pool, None);

    assert_eq!(
        result.children.len(),
        2,
        "parallel child creation should preserve both feasible ReLU branches"
    );
    assert!(
        !result.had_no_branch,
        "parallel path should branch on the unstable neuron"
    );
    assert!(
        !result.had_propagation_failure,
        "parallel path should surface two successful child computations"
    );
    assert_eq!(child_for_activity(&result.children, true).depth(), 1);
    assert_eq!(child_for_activity(&result.children, false).depth(), 1);
}

#[ntest::timeout(5000)]
#[test]
fn test_process_domain_parallel_falls_back_when_parallel_disabled_3089() {
    let network = simple_network();
    let input = test_input_bounds();
    let domain = unstable_relu_domain();
    let verifier = domain_processing_verifier();
    let mut cut_pool = CutPool::new(0);
    let config = DomainProcessingConfig::new(0.5, false);

    let result =
        verifier.process_domain_parallel(&network, &input, &domain, &config, &mut cut_pool, None);

    assert_eq!(
        result.children.len(),
        2,
        "fallback path should delegate to sequential child creation"
    );
    assert!(
        !result.had_no_branch,
        "fallback path still has an unstable neuron to split"
    );
    assert!(
        !result.had_propagation_failure,
        "fallback path should not report propagation failures for feasible children"
    );
    assert_eq!(child_for_activity(&result.children, true).depth(), 1);
    assert_eq!(child_for_activity(&result.children, false).depth(), 1);
}
