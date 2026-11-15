// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::layers::{LinearLayer, ReLULayer, SubLayer};
use crate::{GraphNode, NETWORK_INPUT};
use ndarray::{array, Array1, Array2};
use std::time::{Duration, Instant};

fn fork_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    for (name, weights, bias) in [
        (
            "left",
            array![[1.0_f32, 0.0], [0.0, 2.0]],
            array![0.25_f32, -0.25],
        ),
        (
            "right",
            array![[0.0_f32, 1.0], [-1.0, 0.0]],
            array![0.5_f32, 0.0],
        ),
    ] {
        graph
            .try_add_node(GraphNode::from_input(
                name,
                Layer::Linear(LinearLayer::new(weights, Some(bias)).unwrap()),
            ))
            .unwrap();
    }
    graph
        .try_add_node(GraphNode::new(
            "left_relu",
            Layer::ReLU(ReLULayer),
            vec!["left".into()],
        ))
        .unwrap();
    graph
        .try_add_node(GraphNode::new(
            "difference",
            Layer::Sub(SubLayer),
            vec!["left_relu".into(), "right".into()],
        ))
        .unwrap();
    graph.set_output("difference");
    graph
}

fn inputs() -> Vec<BoundedTensor> {
    vec![
        BoundedTensor::new(
            array![-2.0_f32, -0.5].into_dyn(),
            array![0.5_f32, 1.0].into_dyn(),
        )
        .unwrap(),
        BoundedTensor::new(
            array![2.0_f32, 1.0].into_dyn(),
            array![3.0_f32, 2.0].into_dyn(),
        )
        .unwrap(),
    ]
}

fn caches(
    graph: &GraphNetwork,
    inputs: &[BoundedTensor],
) -> Vec<HashMap<String, Arc<BoundedTensor>>> {
    inputs
        .iter()
        .map(|input| {
            graph
                .collect_node_bounds_with_engine_and_deadline(input, None, None)
                .unwrap()
                .into_iter()
                .map(|(name, bounds)| (name, Arc::new(bounds)))
                .collect()
        })
        .collect()
}

fn same_bits(actual: &LinearBounds, expected: &LinearBounds) {
    for (a, b) in [
        (actual.lower_a(), expected.lower_a()),
        (actual.upper_a(), expected.upper_a()),
    ] {
        assert_eq!(a.shape(), b.shape());
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    for (a, b) in [
        (actual.lower_b(), expected.lower_b()),
        (actual.upper_b(), expected.upper_b()),
    ] {
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    for (a, b) in [
        (actual.lower_a_err(), expected.lower_a_err()),
        (actual.upper_a_err(), expected.upper_a_err()),
    ] {
        assert_eq!(a.is_some(), b.is_some());
        if let (Some(a), Some(b)) = (a, b) {
            assert_eq!(a.shape(), b.shape());
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn prepared_targets_match_independently_retargeted_output_bits_and_keep_domain_maps() {
    let graph = fork_graph();
    let inputs = inputs();
    let caches = caches(&graph, &inputs);
    let refs: Vec<_> = caches.iter().collect();
    let original_plan = graph.dispatch_plan().unwrap();
    let original_output = original_plan.output_node_idx;
    let mut relu_first_coeffs = Vec::new();
    for name in ["left", "left_relu", "right", "difference"] {
        let target = PreparedSoaTarget::new(&graph, name).unwrap();
        assert!(std::ptr::eq(target.plan, original_plan));
        assert_eq!(target.name(), name);
        let mut reference_graph = graph.clone();
        reference_graph.set_output(name);
        let reference_plan = reference_graph.dispatch_plan().unwrap();
        let subset = LinearBounds::new(
            array![[0.0_f32, 1.0]],
            Array1::zeros(1),
            array![[0.0_f32, 1.0]],
            Array1::zeros(1),
        )
        .unwrap();
        for (seed_index, seed) in [LinearBounds::identity(2), subset].into_iter().enumerate() {
            let actual = target
                .try_backward(
                    2,
                    &refs,
                    &inputs,
                    &[None, None],
                    &seed,
                    &ny_core::NaiveCpuGemmEngine,
                    None,
                    None,
                )
                .unwrap()
                .expect("prepared target must execute");
            let expected = try_backward_soa(
                &reference_graph,
                reference_plan,
                2,
                &refs,
                &inputs,
                &[None, None],
                &seed,
                &ny_core::NaiveCpuGemmEngine,
                None,
                None,
            )
            .unwrap()
            .expect("historical retargeted output must execute");
            assert_eq!(actual.0, expected.0);
            assert_eq!(actual.0, vec![true, true]);
            let actual = actual.1.unwrap();
            let expected = expected.1.unwrap();
            assert_eq!(actual.len(), 2);
            assert_eq!(expected.len(), 2);
            for (actual, expected) in actual.iter().zip(&expected) {
                let actual = actual.as_ref().unwrap();
                assert!(actual
                    .lower_b()
                    .iter()
                    .chain(actual.upper_b())
                    .all(|x| x.is_finite()));
                assert!(actual
                    .lower_a()
                    .iter()
                    .chain(actual.upper_a())
                    .all(|x| x.is_finite()));
                assert!(actual
                    .lower_a()
                    .iter()
                    .chain(actual.upper_a())
                    .any(|x| *x != 0.0));
                same_bits(actual, expected.as_ref().unwrap());
            }
            if name == "left" && seed_index == 0 {
                // Analytic identity through this affine target: seeding at the
                // graph's difference output would involve the other branch.
                assert_eq!(
                    actual[0].as_ref().unwrap().lower_a(),
                    &array![[1.0_f32, 0.0], [0.0, 2.0]]
                );
            }
            if name == "left_relu" && seed_index == 0 {
                relu_first_coeffs = actual
                    .iter()
                    .map(|x| x.as_ref().unwrap().lower_a()[[0, 0]].to_bits())
                    .collect();
            }
        }
    }
    assert_eq!(relu_first_coeffs.len(), 2);
    assert_ne!(
        relu_first_coeffs[0], relu_first_coeffs[1],
        "crossing/stable domains must retain different ReLU relaxations"
    );
    assert_eq!(graph.output_name(), "difference");
    assert_eq!(
        graph.dispatch_plan().unwrap().output_node_idx,
        original_output
    );
    assert!(std::ptr::eq(graph.dispatch_plan().unwrap(), original_plan));
}

#[test]
fn prepared_target_keeps_finite_deadline_refusal_before_domain_storage() {
    let graph = fork_graph();
    let target = PreparedSoaTarget::new(&graph, "left").unwrap();
    let seed = LinearBounds::identity(2);
    let live = Some(Instant::now() + Duration::from_secs(60));
    assert!(target
        .try_backward(
            0,
            &[],
            &[],
            &[],
            &seed,
            &ny_core::NaiveCpuGemmEngine,
            live,
            None
        )
        .unwrap()
        .is_none());
    let expired = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
    let error = target
        .try_backward(
            0,
            &[],
            &[],
            &[],
            &seed,
            &ny_core::NaiveCpuGemmEngine,
            expired,
            None,
        )
        .expect_err("expired deadline is terminal");
    assert!(error.is_deadline_exceeded());
}

#[test]
fn prepared_target_rejects_unknown_input_sentinel_and_malformed_domain_layouts() {
    let graph = fork_graph();
    assert!(PreparedSoaTarget::new(&graph, "unknown").is_err());
    assert!(PreparedSoaTarget::new(&graph, NETWORK_INPUT).is_err());
    let target = PreparedSoaTarget::new(&graph, "left").unwrap();
    let seed = LinearBounds::identity(2);
    assert!(target
        .try_backward(
            0,
            &[],
            &[],
            &[],
            &seed,
            &ny_core::NaiveCpuGemmEngine,
            None,
            None
        )
        .unwrap()
        .is_none());
    assert!(target
        .try_backward(
            1,
            &[],
            &[],
            &[],
            &seed,
            &ny_core::NaiveCpuGemmEngine,
            None,
            None
        )
        .is_err());
    let inputs = inputs();
    let mut caches = caches(&graph, &inputs);
    let refs: Vec<_> = caches.iter().collect();
    assert!(target
        .try_backward(
            2,
            &refs,
            &inputs,
            &[None],
            &seed,
            &ny_core::NaiveCpuGemmEngine,
            None,
            None
        )
        .is_err());
    assert!(target
        .try_backward(
            2,
            &refs,
            &inputs,
            &[None, None],
            &LinearBounds::identity(1),
            &ny_core::NaiveCpuGemmEngine,
            None,
            None
        )
        .is_err());
    // Retargeting must not broaden the historical plain-spec seed contract.
    for lower in [true, false] {
        let mut nonplain = LinearBounds::identity(2);
        if lower {
            nonplain.lower_a_err = Some(Array2::zeros((2, 2)));
        } else {
            nonplain.upper_a_err = Some(Array2::zeros((2, 2)));
        }
        assert!(target
            .try_backward(
                2,
                &refs,
                &inputs,
                &[None, None],
                &nonplain,
                &ny_core::NaiveCpuGemmEngine,
                None,
                None
            )
            .is_err());
    }
    caches[1].remove("left");
    let refs: Vec<_> = caches.iter().collect();
    assert!(target
        .try_backward(
            2,
            &refs,
            &inputs,
            &[None, None],
            &seed,
            &ny_core::NaiveCpuGemmEngine,
            None,
            None
        )
        .is_err());
}

#[test]
fn prepared_target_refuses_point_only_node_outside_target_ancestry() {
    use crate::certify::work::WorkBudget;
    use crate::layers::{PointLinearBuildPolicy, PointLinearLayer};
    let mut graph = fork_graph();
    let budget = WorkBudget::new(None, 1 << 24, 1 << 28, 1 << 30, 1_024, Duration::ZERO).unwrap();
    let weights = Array2::zeros((1, 2));
    let diagnostic = PointLinearLayer::from_original_view(
        weights.view(),
        false,
        None,
        PointLinearBuildPolicy {
            budget: &budget,
            deadline: None,
            max_resident_bytes: 1 << 28,
            max_scratch_bytes: 1 << 24,
            max_work: 1 << 30,
        },
    )
    .unwrap();
    graph
        .try_add_node(GraphNode::from_input(
            "unrelated-diagnostic",
            Layer::PointLinear(diagnostic),
        ))
        .unwrap();
    let error = PreparedSoaTarget::new(&graph, "left")
        .err()
        .expect("whole original graph must remain proof eligible");
    assert!(matches!(error, NyError::UnsupportedOp(message) if message.contains("point-only")));
}
