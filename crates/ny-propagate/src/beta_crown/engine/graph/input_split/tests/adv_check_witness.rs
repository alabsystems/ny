// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #advcheck-witness: the adv_check PGD probe must CARRY the concrete point it
//! validated, and the input-split loop must hand that point to the confirmer
//! inside the `PotentialViolation`.
//!
//! Before this, `try_adv_check_on_domain` ran a true concrete forward, found a
//! genuine violation, and returned a bare `true` — dropping `x` and `output`
//! on the floor. The confirmer then re-attacked the ROOT box for a point the
//! code had already held, so a validated counterexample downgraded to Unknown.
//!
//! These are deterministic: the probe's SPSA step on an identity net is exactly
//! `-step_size` per step regardless of the RNG sign draw (grad = pert * pert),
//! so five steps always drive `x` to the lower corner of the domain.

use ndarray::{arr2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use std::time::Duration;

use super::super::adv_check::try_adv_check_on_domain;
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::beta_crown::result::BabVerificationStatus;
use crate::beta_crown::BranchingHeuristic;
use crate::{GraphNetwork, GraphNode, Layer, LinearLayer};

/// y = x on one scalar input.
fn identity_graph() -> GraphNetwork {
    let identity = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("identity layer should build");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("id", Layer::Linear(identity)));
    graph.set_output("id");
    graph
}

fn box_1d(lo: f32, hi: f32) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![lo]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![hi]).unwrap(),
    )
    .unwrap()
}

/// A hit returns the CONCRETE POINT, not a bare `true`: the carried point is
/// inside the searched domain, genuinely violating under the same
/// objective/threshold test the probe accepted it with, and accompanied by the
/// output its own concrete forward produced.
#[test]
fn adv_check_hit_carries_the_validated_point_advcheck_witness() {
    let graph = identity_graph();
    let domain = box_1d(-1.0, -0.6);
    let objective = [1.0_f32];
    let threshold = -0.5_f32;

    let witness =
        try_adv_check_on_domain(&graph, &domain, &objective, threshold, false, None, 0, None)
            .expect("probe runs")
            .expect("a domain that violates everywhere must produce a witness");

    assert_eq!(
        witness.input_shape,
        vec![1_usize],
        "carries the input shape"
    );
    assert_eq!(witness.input.len(), 1);
    assert_eq!(witness.output.len(), 1);

    let x = witness.input[0];
    assert!(
        (-1.0..=-0.6).contains(&x),
        "carried point {x} escaped the searched sub-box"
    );

    let obj: f32 = objective
        .iter()
        .zip(witness.output.iter())
        .map(|(a, b)| a * b)
        .sum();
    assert!(
        obj <= threshold,
        "carried point fails the violation test it was accepted under \
         (obj={obj}, threshold={threshold})"
    );

    let point = BoundedTensor::concrete(ArrayD::from_shape_vec(IxDyn(&[1]), vec![x]).unwrap())
        .expect("concrete point");
    let fresh = graph
        .propagate_concrete_point(&point, None, None)
        .expect("concrete forward");
    assert!(
        (fresh.lower()[0] - witness.output[0]).abs() < 1e-6,
        "carried output {} disagrees with a fresh forward {}",
        witness.output[0],
        fresh.lower()[0]
    );
}

/// No violation in the domain: still nothing carried — exactly the old
/// `Ok(false)`. A miss must never fabricate a witness.
#[test]
fn adv_check_miss_carries_nothing_advcheck_witness() {
    let graph = identity_graph();
    let witness = try_adv_check_on_domain(
        &graph,
        &box_1d(-1.0, 1.0),
        &[1.0_f32],
        -1000.0,
        false,
        None,
        0,
        None,
    )
    .expect("probe runs");
    assert!(witness.is_none(), "a miss must not fabricate a witness");
}

/// End-to-end through the input-split BaB loop: when adv_check is the stage
/// that decides, the returned `PotentialViolation` CARRIES the point.
///
/// The root box `[-1, 1]` is deliberately undecided at threshold `-0.5`
/// (root_lower = -1 < -0.5 < 1 = root_upper), so neither root early-exit fires
/// and the loop reaches the adv_check arm on its first batch (`adv_check = 0`).
#[test]
fn input_split_potential_violation_carries_the_adv_check_witness() {
    let graph = identity_graph();
    let input = box_1d(-1.0, 1.0);
    let threshold = -0.5_f32;

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        adv_check: 0,
        max_domains: 64,
        max_depth: 8,
        timeout: Duration::from_secs(10),
        ..Default::default()
    });

    let result = verifier
        .verify_graph_input_split(&graph, &input, &[1.0_f32], threshold)
        .expect("input split completes");

    let BabVerificationStatus::PotentialViolation { witness } = &result.result else {
        panic!("expected PotentialViolation, got {:?}", result.result);
    };
    let witness = witness
        .as_deref()
        .expect("the adv_check arm must carry its concrete point to the confirmer");
    assert_eq!(witness.input.len(), 1);
    assert!(
        (-1.0..=1.0).contains(&witness.input[0]),
        "carried point {} escaped the root box",
        witness.input[0]
    );
    assert!(
        witness.output[0] <= threshold,
        "carried witness does not violate (y={}, threshold={threshold})",
        witness.output[0]
    );
}
