// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::layers::{AddLayer, Layer, LinearLayer};
use crate::{GraphNetwork, GraphNode};
use ndarray::{ArrayD, IxDyn};
use ny_core::NyError;
use ny_tensor::BoundedTensor;
use std::time::Instant;

fn branches() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    for (name, weights) in [("a", vec![1.0, -2.0]), ("b", vec![0.5, 0.25])] {
        graph.add_node(GraphNode::from_input(
            name,
            Layer::Linear(LinearLayer::from_flat(weights, 1, 2, None).unwrap()),
        ));
    }
    graph.add_node(GraphNode::binary("out", Layer::Add(AddLayer), "a", "b"));
    graph.set_output("out");
    graph
}

#[test]
fn graph_point_observer_sees_completed_cached_points_once_without_changing_output() {
    let graph = branches();
    let input =
        BoundedTensor::concrete(ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 0.75]).unwrap())
            .unwrap();
    let (ordinary, ordinary_trace) = graph
        .propagate_concrete_point_with_trace(&input, None, None)
        .unwrap();
    let mut seen = Vec::new();
    let (observed, trace) = graph
        .propagate_concrete_point_with_observer(&input, None, |name, bounds| {
            assert_eq!(bounds.lower(), bounds.upper());
            seen.push((name.to_string(), bounds.lower().clone()));
            Ok(())
        })
        .unwrap();
    assert_eq!(seen.len(), 3);
    assert_eq!(trace.steps().len(), ordinary_trace.steps().len());
    assert_eq!(observed.lower(), ordinary.lower());
    assert_eq!(observed.upper(), ordinary.upper());
    assert_eq!(seen.last().unwrap().0, "out");
    assert_eq!(&seen.last().unwrap().1, observed.lower());
    assert_eq!(seen.iter().filter(|(name, _)| name == "a").count(), 1);
    assert_eq!(seen.iter().filter(|(name, _)| name == "b").count(), 1);
    let interval = graph.propagate_ibp_sound(&input).unwrap();
    assert!(interval
        .lower()
        .iter()
        .zip(interval.upper())
        .any(|(l, u)| l != u));
}

#[test]
fn graph_point_observer_cancellation_refuses_before_later_node_or_result_publication() {
    let graph = branches();
    let input = BoundedTensor::concrete(ArrayD::from_elem(IxDyn(&[1, 2]), 1.0)).unwrap();
    let mut seen = 0;
    let refusal = graph.propagate_concrete_point_with_observer(&input, None, |_, _| {
        seen += 1;
        Err(NyError::DeadlineExceeded("injected observer cutoff".into()))
    });
    assert!(matches!(refusal, Err(NyError::DeadlineExceeded(_))));
    assert_eq!(seen, 1);
    let mut expired_seen = 0;
    let refusal =
        graph.propagate_concrete_point_with_observer(&input, Some(Instant::now()), |_, _| {
            expired_seen += 1;
            Ok(())
        });
    assert!(matches!(refusal, Err(NyError::DeadlineExceeded(_))));
    assert_eq!(expired_seen, 0);
}
