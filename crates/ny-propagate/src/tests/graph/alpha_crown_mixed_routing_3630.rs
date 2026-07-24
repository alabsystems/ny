// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Mixed sequential alpha-CROWN routing regressions for #3630.
//!
//! Verifies that `Linear -> ReLU -> Linear -> Sigmoid/Tanh -> Linear` graphs
//! route to DAG alpha-CROWN rather than getting stranded on the sequential
//! ReLU-only path.

use crate::bounds::{AlphaCrownConfig, GradientMethod};
use crate::tests::crown::helpers::total_width;
use crate::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

#[derive(Clone, Copy)]
enum MixedMonotoneKind {
    Sigmoid,
    Tanh,
}

impl MixedMonotoneKind {
    fn layer(self) -> Layer {
        match self {
            Self::Sigmoid => Layer::Sigmoid(SigmoidLayer::new()),
            Self::Tanh => Layer::Tanh(TanhLayer::new()),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Sigmoid => "Sigmoid",
            Self::Tanh => "Tanh",
        }
    }
}

/// Mixed sequential graph for #3630:
/// Linear -> ReLU -> Linear -> Sigmoid/Tanh -> Linear.
///
/// The first linear keeps every ReLU pre-activation strictly positive across the
/// input box, so the ReLU is present in the graph but contributes zero unstable
/// alpha state. Any improvement over fixed-slope CROWN must therefore come from
/// the DAG monotone-alpha route, not from sequential ReLU alpha optimization.
fn build_mixed_relu_monotone_graph(kind: MixedMonotoneKind) -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[0.6_f32, 0.2], [0.4, 0.5], [0.3, 0.1]]);
    let b1 = arr1(&[0.9_f32, 0.95, 1.0]);
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer::new()),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.9_f32, 0.4, 0.2], [0.3, 0.8, 0.5], [0.6, 0.2, 0.7]]);
    let b2 = arr1(&[0.05_f32, 0.05, 0.05]);
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "monotone1",
        kind.layer(),
        vec!["linear2".to_string()],
    ));

    let w3 = arr2(&[[0.9_f32, -0.6, 0.4], [-0.3, 1.1, -0.5]]);
    let b3 = arr1(&[0.0_f32, 0.1]);
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(LinearLayer::new(w3, Some(b3)).unwrap()),
        vec!["monotone1".to_string()],
    ));
    graph.set_output("linear3");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[2]), 1.0_f32),
    )
    .unwrap();

    (graph, input)
}

/// Mixed ReLU + Sigmoid graphs must route to DAG alpha-CROWN even when the
/// ReLU contributes no unstable alpha state. Without the mixed-activation DAG
/// route, this graph would collapse to fixed-slope CROWN because the sequential
/// optimizer would see zero unstable ReLUs and exit early.
#[ntest::timeout(60000)]
#[test]
fn test_sequential_mixed_relu_sigmoid_routes_to_dag_alpha_crown_3630() {
    let (graph, input) = build_mixed_relu_monotone_graph(MixedMonotoneKind::Sigmoid);

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let ibp_width = total_width(&ibp_bounds);
    let crown_bounds = graph
        .propagate_crown(&input)
        .expect("fixed-slope CROWN should succeed on mixed ReLU+Sigmoid graph");
    let crown_width = total_width(&crown_bounds);

    let config = AlphaCrownConfig {
        iterations: 50,
        gradient_method: GradientMethod::Spsa,
        spsa_samples: 4,
        fix_interm_bounds: false,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..AlphaCrownConfig::default()
    };
    let alpha_bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .expect("alpha-CROWN should succeed on mixed ReLU+Sigmoid graph");
    let alpha_width = total_width(&alpha_bounds);
    eprintln!(
        "mixed ReLU+Sigmoid: IBP={ibp_width:.6}, CROWN={crown_width:.6}, alpha={alpha_width:.6}"
    );

    assert!(
        alpha_width <= ibp_width + 1e-4,
        "mixed ReLU+Sigmoid alpha-CROWN ({alpha_width:.6}) wider than IBP \
         ({ibp_width:.6}) — unsound (#3630)"
    );
    assert!(
        alpha_width + 1e-6 < crown_width,
        "mixed ReLU+Sigmoid alpha-CROWN should beat fixed-slope CROWN when DAG \
         monotone routing is active. fixed={crown_width:.6}, alpha={alpha_width:.6} (#3630)"
    );
}

/// Default AnalyticChain must also route the mixed monotone shape into DAG
/// alpha-CROWN rather than leaving it stranded on the sequential ReLU-only
/// path. This is the direct soundness guard for the default gradient mode.
#[ntest::timeout(60000)]
#[test]
fn test_sequential_mixed_relu_tanh_analytic_chain_soundness_3630() {
    let (graph, input) = build_mixed_relu_monotone_graph(MixedMonotoneKind::Tanh);

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let ibp_width = total_width(&ibp_bounds);

    let config = AlphaCrownConfig {
        iterations: 20,
        gradient_method: GradientMethod::AnalyticChain,
        fix_interm_bounds: false,
        ..AlphaCrownConfig::default()
    };
    let alpha_bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .unwrap_or_else(|error| {
            panic!(
                "{} AnalyticChain alpha-CROWN should succeed on mixed sequential graph: {error}",
                MixedMonotoneKind::Tanh.name()
            )
        });
    let alpha_width = total_width(&alpha_bounds);
    eprintln!(
        "mixed ReLU+{} AnalyticChain: IBP={ibp_width:.6}, alpha={alpha_width:.6}",
        MixedMonotoneKind::Tanh.name()
    );

    assert!(
        alpha_width <= ibp_width + 1e-4,
        "mixed ReLU+{} AnalyticChain alpha-CROWN ({alpha_width:.6}) wider than \
         IBP ({ibp_width:.6}) — unsound (#3630)",
        MixedMonotoneKind::Tanh.name()
    );
}
