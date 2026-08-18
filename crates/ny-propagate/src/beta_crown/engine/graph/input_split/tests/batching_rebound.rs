// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::super::batching::{bound_deferred_domains_batch, force_override_rebound_parallel};
use super::*;
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::engine::graph::propagation::batched::SPEC_GATE_TEST_LOCK;

#[test]
fn test_bound_deferred_domains_batch_uses_node_bounds_override() {
    let graph = build_complete_clip_override_graph();
    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("finite input bounds");
    let mut domains = vec![GraphInputDomain {
        input_bounds: Arc::new(input_bounds),
        lower_bound: -1.0,
        upper_bound: 1.0,
        depth: 1,
        priority: 1.0,
        linear_bounds: None,
        needs_bounding: true,
        node_bounds_override: Some(build_complete_clip_override_bounds()),
        inherited_alpha_state: None,
    }];

    bound_deferred_domains_batch(
        &mut domains,
        &graph,
        &arr2(&[[1.0_f32]]),
        None,
        None,
        None,
        None,
        None,
        None,
        &BetaCrownConfig::default(),
    )
    .expect("deferred bound pass should honor the node-bounds override");

    let domain = &domains[0];
    assert!(
        domain.upper_bound <= 0.21,
        "deferred complete-clipping override should tighten the child upper bound, got {}",
        domain.upper_bound
    );
    assert!(
        !domain.needs_bounding,
        "deferred domain should be marked bounded after the override-backed pass"
    );
    assert!(
        domain.node_bounds_override.is_none(),
        "override should be consumed after the deferred bound pass"
    );
}

#[test]
fn override_parallel_scalar_domains_match_serial() {
    let _gate_guard = SPEC_GATE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    struct OverrideForceReset;
    impl Drop for OverrideForceReset {
        fn drop(&mut self) {
            force_override_rebound_parallel(None);
        }
    }
    let _force_reset = OverrideForceReset;

    let graph = build_complete_clip_override_graph();
    let spec_matrix = arr2(&[[1.0_f32]]);
    let override_bounds = build_complete_clip_override_bounds();
    let child_a = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[0.6_f32]).into_dyn())
        .expect("valid child_a");
    let child_b = BoundedTensor::new(arr1(&[-0.4_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid child_b");
    let make_domains = || {
        [child_a.clone(), child_b.clone()]
            .into_iter()
            .map(|input_bounds| GraphInputDomain {
                input_bounds: Arc::new(input_bounds),
                lower_bound: -1.0,
                upper_bound: 1.0,
                depth: 1,
                priority: 1.0,
                linear_bounds: None,
                needs_bounding: true,
                node_bounds_override: Some(Arc::clone(&override_bounds)),
                inherited_alpha_state: None,
            })
            .collect::<Vec<_>>()
    };
    let config = BetaCrownConfig {
        input_split_override_parallel: true,
        ..BetaCrownConfig::default()
    };
    let run = |domains: &mut [GraphInputDomain]| {
        bound_deferred_domains_batch(
            domains,
            &graph,
            &spec_matrix,
            None,
            None,
            None,
            None,
            None,
            None,
            &config,
        )
    };

    force_override_rebound_parallel(Some(false));
    let mut serial = make_domains();
    run(&mut serial).expect("serial override rebound should succeed");

    force_override_rebound_parallel(Some(true));
    let mut parallel = make_domains();
    run(&mut parallel).expect("parallel override rebound should succeed");

    for (idx, (serial_domain, parallel_domain)) in serial.iter().zip(parallel.iter()).enumerate() {
        assert_eq!(parallel_domain.lower_bound, serial_domain.lower_bound);
        assert_eq!(parallel_domain.upper_bound, serial_domain.upper_bound);
        assert_eq!(parallel_domain.priority, serial_domain.priority);
        assert_eq!(parallel_domain.needs_bounding, serial_domain.needs_bounding);
        assert_eq!(
            parallel_domain.node_bounds_override.is_some(),
            serial_domain.node_bounds_override.is_some()
        );
        match (&parallel_domain.linear_bounds, &serial_domain.linear_bounds) {
            (Some(parallel_linear), Some(serial_linear)) => {
                assert_linear_bounds_match(parallel_linear, serial_linear);
            }
            (None, None) => {}
            _ => panic!("linear-bound availability changed at domain {idx}"),
        }
    }
}

/// Monotonicity guard: when CROWN produces a worse lower bound than the
/// parent domain, the parent's bound is retained. This prevents relaxation
/// noise from regressing the certification bound on child subdomains.
/// Reference: alpha-beta-CROWN input_split/bounding.py:154
///   lb = torch.max(lb, dm_lb)
#[test]
fn test_bound_deferred_domains_batch_monotonicity_guard() {
    let graph = build_complete_clip_override_graph();
    let input_bounds =
        BoundedTensor::new(arr1(&[-0.5_f32]).into_dyn(), arr1(&[0.5_f32]).into_dyn())
            .expect("finite input bounds");
    let parent_lower_bound = 0.1_f32;
    let mut domains = vec![GraphInputDomain {
        input_bounds: Arc::new(input_bounds),
        lower_bound: parent_lower_bound,
        upper_bound: 1.0,
        depth: 1,
        priority: 1.0,
        linear_bounds: None,
        needs_bounding: true,
        node_bounds_override: None,
        inherited_alpha_state: None,
    }];

    bound_deferred_domains_batch(
        &mut domains,
        &graph,
        &arr2(&[[1.0_f32]]),
        None,
        None,
        None,
        None,
        None,
        None,
        &BetaCrownConfig::default(),
    )
    .expect("deferred bound pass should succeed");

    let domain = &domains[0];
    assert!(
        domain.lower_bound >= parent_lower_bound,
        "monotonicity guard must prevent lower bound regression: got {} but parent had {}",
        domain.lower_bound,
        parent_lower_bound
    );
    assert!(!domain.needs_bounding);
}

#[test]
fn test_bound_deferred_domains_batch_override_path_applies_parent_floor_4354() {
    let graph = build_complete_clip_override_graph();
    let node_bounds_override = build_complete_clip_override_bounds();
    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("finite input bounds");
    let direct_bounds = compute_crown_or_ibp_bounds_with_node_bounds(
        &graph,
        &input_bounds,
        &arr2(&[[1.0_f32]]),
        None,
        None,
        Some(node_bounds_override.as_ref()),
        None,
        None,
        None,
        None,
        false,
    )
    .expect("direct override-backed rebound should succeed");
    let parent_lower_bound = 0.1_f32;
    let mut domains = vec![GraphInputDomain {
        input_bounds: Arc::new(input_bounds),
        lower_bound: parent_lower_bound,
        upper_bound: 1.0,
        depth: 1,
        priority: 1.0,
        linear_bounds: None,
        needs_bounding: true,
        node_bounds_override: Some(node_bounds_override),
        inherited_alpha_state: None,
    }];

    assert!(
        direct_bounds.0.lower_scalar() < parent_lower_bound,
        "fixture should exercise the override monotonicity guard: direct lower={} parent={}",
        direct_bounds.0.lower_scalar(),
        parent_lower_bound
    );

    let config = BetaCrownConfig::default();
    bound_deferred_domains_batch(
        &mut domains,
        &graph,
        &arr2(&[[1.0_f32]]),
        None,
        None,
        None,
        None,
        None,
        None,
        &config,
    )
    .expect("override-backed deferred bound pass should apply the parent floor");

    let domain = &domains[0];
    assert!(
        (domain.lower_bound - parent_lower_bound).abs() <= 1e-6,
        "override-backed monotonicity guard should keep parent lower bound: actual={} expected={}",
        domain.lower_bound,
        parent_lower_bound
    );
    assert!(
        (domain.upper_bound - direct_bounds.0.upper_scalar()).abs() <= 1e-6,
        "override-backed rebound should preserve the fresh upper bound: actual={} expected={}",
        domain.upper_bound,
        direct_bounds.0.upper_scalar()
    );
    assert_eq!(
        domain.priority,
        config
            .domain_priority(domain.lower_bound, domain.upper_bound)
            .expect("priority recomputation should succeed"),
        "override-backed rebound should recompute priority from the clamped child bounds"
    );
    assert!(!domain.needs_bounding);
    assert!(domain.node_bounds_override.is_none());
}
