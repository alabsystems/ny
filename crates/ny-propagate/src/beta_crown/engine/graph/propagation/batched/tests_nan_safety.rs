// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! NaN-safe accumulation tests for batched CROWN backward pass (#2102).
//!
//! Tests that the indexed pending-bounds helper correctly handles:
//! - INF-cancellation (INF + NEG_INFINITY → sound conservative bound, not NaN)
//! - NaN-preservation (upstream NaN propagates through accumulation)

use ndarray::{arr1, arr2};

use crate::bounds::LinearBounds;
use crate::layers::{Layer, LinearLayer};
use crate::{GraphNetwork, GraphNode};

use super::gpu_objective_intervals_valid;
use super::indexed_pending::IndexedPendingLinearBounds;

#[test]
fn gpu_objective_interval_gate_rejects_malformed_passes() {
    assert!(gpu_objective_intervals_valid(&[-1.0], &[1.0], 1));
    assert!(!gpu_objective_intervals_valid(&[], &[], 0));
    assert!(!gpu_objective_intervals_valid(&[-1.0], &[1.0, 2.0], 1));
    assert!(!gpu_objective_intervals_valid(
        &[f32::NEG_INFINITY],
        &[1.0],
        1
    ));
    assert!(!gpu_objective_intervals_valid(&[2.0], &[1.0], 1));
}

fn make_pending(node_name: &str, n_domains: usize) -> IndexedPendingLinearBounds {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        node_name,
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0_f32]))).expect("linear layer"),
        ),
    ));
    graph.set_output(node_name);
    let plan = graph.dispatch_plan().expect("dispatch plan should build");
    IndexedPendingLinearBounds::new(plan, n_domains)
}

/// Test for #2102: INF-cancellation in batched CROWN accumulation must produce
/// sound conservative bounds, not NaN.
///
/// The batched variant stores `Vec<Option<LinearBounds>>` per node, indexed by
/// domain_idx. This test exercises the accumulation path at domain_idx=0 in a
/// batch of 2 domains.
#[test]
fn test_accumulate_crown_bounds_batched_inf_cancellation_safe_2102() {
    let existing = LinearBounds {
        lower_a: arr2(&[[f32::NEG_INFINITY, 1.0]]),
        lower_b: arr1(&[f32::NEG_INFINITY]),
        upper_a: arr2(&[[f32::INFINITY, 2.0]]),
        upper_b: arr1(&[f32::INFINITY]),
        lower_a_err: None,
        upper_a_err: None,
    };
    let new_bounds = LinearBounds {
        lower_a: arr2(&[[f32::INFINITY, 3.0]]),
        lower_b: arr1(&[f32::INFINITY]),
        upper_a: arr2(&[[f32::NEG_INFINITY, 4.0]]),
        upper_b: arr1(&[f32::NEG_INFINITY]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let mut pending = make_pending("out", 2);
    pending
        .seed_name(crate::NETWORK_INPUT, 0, existing)
        .expect("existing bounds should seed");
    pending
        .accumulate_name(crate::NETWORK_INPUT, new_bounds, 0)
        .expect("accumulation should succeed");

    let entry = pending
        .get_name(crate::NETWORK_INPUT)
        .expect("_input key must exist after accumulation");
    let result = entry[0]
        .as_ref()
        .expect("domain_idx=0 slot must contain bounds after accumulation");

    // INF + (-INF) = NaN under IEEE 754. NaN-safe addition should recover:
    assert_eq!(
        result.lower_a[[0, 0]],
        f32::NEG_INFINITY,
        "lower_a INF-cancellation should recover to NEG_INFINITY, not NaN"
    );
    assert_eq!(
        result.lower_b[0],
        f32::NEG_INFINITY,
        "lower_b INF-cancellation should recover to NEG_INFINITY, not NaN"
    );
    assert_eq!(
        result.upper_a[[0, 0]],
        f32::INFINITY,
        "upper_a INF-cancellation should recover to INFINITY, not NaN"
    );
    assert_eq!(
        result.upper_b[0],
        f32::INFINITY,
        "upper_b INF-cancellation should recover to INFINITY, not NaN"
    );

    // Normal additions should still work correctly
    assert!(
        (result.lower_a[[0, 1]] - 4.0).abs() < 1e-6,
        "Normal lower_a addition should produce 1.0 + 3.0 = 4.0, got {}",
        result.lower_a[[0, 1]]
    );
    assert!(
        (result.upper_a[[0, 1]] - 6.0).abs() < 1e-6,
        "Normal upper_a addition should produce 2.0 + 4.0 = 6.0, got {}",
        result.upper_a[[0, 1]]
    );

    // Other domain slot must remain unaffected
    assert!(entry[1].is_none(), "domain_idx=1 slot should remain None");
    assert!(
        pending.input_accumulated()[0],
        "network-input accumulation should preserve input tracking"
    );
}

/// Test for #2102: NaN from upstream becomes conservative infinity in batched accumulation.
///
/// safe_add_* replaces all NaN with conservative bounds (NEG_INFINITY for lower,
/// INFINITY for upper). Exercises the intermediate node path (not "_input").
#[test]
fn test_accumulate_crown_bounds_batched_nan_input_preserved_2102() {
    let existing = LinearBounds {
        lower_a: arr2(&[[f32::NAN]]),
        lower_b: arr1(&[1.0]),
        upper_a: arr2(&[[2.0]]),
        upper_b: arr1(&[f32::NAN]),
        lower_a_err: None,
        upper_a_err: None,
    };
    let new_bounds = LinearBounds {
        lower_a: arr2(&[[5.0]]),
        lower_b: arr1(&[3.0]),
        upper_a: arr2(&[[4.0]]),
        upper_b: arr1(&[6.0]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let mut pending = make_pending("node1", 1);
    pending
        .seed_name("node1", 0, existing)
        .expect("existing bounds should seed");
    pending
        .accumulate_name("node1", new_bounds, 0)
        .expect("accumulation should succeed");

    let result = pending
        .get_name("node1")
        .expect("node1 key must exist after accumulation")[0]
        .as_ref()
        .expect("domain_idx=0 slot must contain bounds after accumulation");

    // NaN input is replaced with conservative infinity by safe_add_*
    // (NEG_INFINITY for lower bounds, INFINITY for upper bounds).
    // This is the sound behavior: NaN is not a valid bound, so we widen
    // to the most conservative value. See safe_add in
    // network/graph_crown/utils.rs.
    assert_eq!(
        result.lower_a[[0, 0]],
        f32::NEG_INFINITY,
        "NaN in lower_a should become NEG_INFINITY (conservative lower bound)"
    );
    assert_eq!(
        result.upper_b[0],
        f32::INFINITY,
        "NaN in upper_b should become INFINITY (conservative upper bound)"
    );
    // Non-NaN additions should still work
    assert!(
        (result.lower_b[0] - 4.0).abs() < 1e-6,
        "Normal lower_b: 1.0 + 3.0 = 4.0, got {}",
        result.lower_b[0]
    );
    assert!(
        (result.upper_a[[0, 0]] - 6.0).abs() < 1e-6,
        "Normal upper_a: 2.0 + 4.0 = 6.0, got {}",
        result.upper_a[[0, 0]]
    );

    // input_accumulated should remain false (not "_input" key)
    assert!(
        !pending.input_accumulated()[0],
        "intermediate node accumulation must not set input_accumulated"
    );
}

/// Test for #2102: NaN in upper_a and lower_b positions (complementary to the test above
/// which only covers lower_a and upper_b).
///
/// A bug in safe_add polarity that only affects upper_a or lower_b would not be caught
/// by the existing test. This test covers all 4 NaN positions across the two tests.
#[test]
fn test_accumulate_crown_bounds_batched_nan_in_upper_a_and_lower_b_2102() {
    let existing = LinearBounds {
        lower_a: arr2(&[[1.0]]),
        lower_b: arr1(&[f32::NAN]),
        upper_a: arr2(&[[f32::NAN]]),
        upper_b: arr1(&[2.0]),
        lower_a_err: None,
        upper_a_err: None,
    };
    let new_bounds = LinearBounds {
        lower_a: arr2(&[[3.0]]),
        lower_b: arr1(&[4.0]),
        upper_a: arr2(&[[5.0]]),
        upper_b: arr1(&[6.0]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let mut pending = make_pending("node2", 1);
    pending
        .seed_name("node2", 0, existing)
        .expect("existing bounds should seed");
    pending
        .accumulate_name("node2", new_bounds, 0)
        .expect("accumulation should succeed");

    let result = pending
        .get_name("node2")
        .expect("node2 key must exist after accumulation")[0]
        .as_ref()
        .expect("domain_idx=0 slot must contain bounds after accumulation");

    // NaN in upper_a should become INFINITY (conservative upper bound)
    assert_eq!(
        result.upper_a[[0, 0]],
        f32::INFINITY,
        "NaN in upper_a should become INFINITY (conservative upper bound)"
    );
    // NaN in lower_b should become NEG_INFINITY (conservative lower bound)
    assert_eq!(
        result.lower_b[0],
        f32::NEG_INFINITY,
        "NaN in lower_b should become NEG_INFINITY (conservative lower bound)"
    );
    // Non-NaN additions should still be correct
    assert!(
        (result.lower_a[[0, 0]] - 4.0).abs() < 1e-6,
        "Normal lower_a: 1.0 + 3.0 = 4.0, got {}",
        result.lower_a[[0, 0]]
    );
    assert!(
        (result.upper_b[0] - 8.0).abs() < 1e-6,
        "Normal upper_b: 2.0 + 6.0 = 8.0, got {}",
        result.upper_b[0]
    );
}
