// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use ndarray::{arr1, arr2, array};
use ny_core::NyError;

use super::*;
use crate::bounds::{GradientMethod, MonotoneSShapedAlpha};
use crate::layers::{LinearLayer, SigmoidLayer, TanhLayer};
use crate::network::core::GraphNode;
use crate::LinearBounds;

#[derive(Clone, Copy)]
enum SShaped {
    Sigmoid,
    Tanh,
}

impl SShaped {
    fn layer(self) -> Layer {
        match self {
            Self::Sigmoid => Layer::Sigmoid(SigmoidLayer::new()),
            Self::Tanh => Layer::Tanh(TanhLayer::new()),
        }
    }

    fn eval(self, value: f64) -> f64 {
        match self {
            Self::Sigmoid => 1.0 / (1.0 + (-value).exp()),
            Self::Tanh => value.tanh(),
        }
    }

    fn relaxation(
        self,
        preactivation: &BoundedTensor,
        alpha: &MonotoneSShapedAlpha,
    ) -> (f64, f64, f64, f64) {
        let carrier = LinearBounds::identity(1);
        let relaxed = match self {
            Self::Sigmoid => {
                SigmoidLayer::new().propagate_linear_with_alpha(&carrier, preactivation, alpha)
            }
            Self::Tanh => {
                TanhLayer::new().propagate_linear_with_alpha(&carrier, preactivation, alpha)
            }
        }
        .expect("S-shaped alpha relaxation");
        (
            f64::from(relaxed.lower_a()[[0, 0]]),
            f64::from(relaxed.lower_b()[0]),
            f64::from(relaxed.upper_a()[[0, 0]]),
            f64::from(relaxed.upper_b()[0]),
        )
    }
}

fn interval(lower: f32, upper: f32) -> BoundedTensor {
    BoundedTensor::new(array![lower].into_dyn(), array![upper].into_dyn()).unwrap()
}

fn build_graph(kind: SShaped) -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "pre",
        Layer::Linear(LinearLayer::new(arr2(&[[4.0]]), Some(arr1(&[0.0]))).unwrap()),
    ));
    graph.add_node(GraphNode::new("act", kind.layer(), vec!["pre".to_string()]));
    graph.add_node(GraphNode::new(
        "readout",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap()),
        vec!["act".to_string()],
    ));
    graph.set_output("readout");
    (graph, interval(-1.0, 1.0))
}

fn assert_registered_relaxation_encloses(
    kind: SShaped,
    bounds: &HashMap<String, BoundedTensor>,
    state: &GraphAlphaState,
    lane: &str,
) {
    let preactivation = bounds.get("pre").expect("producer bounds");
    let lower = preactivation.lower()[[0]];
    let upper = preactivation.upper()[[0]];
    assert!(lower <= -4.0 && upper >= 4.0, "{lane}: producer box moved");

    let alpha = state
        .monotone_s_shaped_alpha("act")
        .expect("registered activation alpha");
    assert!(
        alpha.mask_cross[0],
        "{lane}: alpha was initialized from the activation output instead of [-4, 4]"
    );
    let (lower_slope, lower_bias, upper_slope, upper_bias) = kind.relaxation(preactivation, alpha);
    for step in 0..=2_000 {
        let value =
            f64::from(lower) + (f64::from(upper) - f64::from(lower)) * f64::from(step) / 2_000.0;
        let truth = kind.eval(value);
        let published_lower = lower_slope * value + lower_bias;
        let published_upper = upper_slope * value + upper_bias;
        assert!(
            published_lower <= truth + 1e-6,
            "{lane}: lower line exceeds the activation at {value}: {published_lower} > {truth}"
        );
        assert!(
            published_upper >= truth - 1e-6,
            "{lane}: upper line falls below the activation at {value}: {published_upper} < {truth}"
        );
    }
}

#[test]
fn resolver_uses_exact_unary_producer_identity() {
    let input = interval(-9.0, 9.0);
    let producer = interval(-4.0, 4.0);
    let activation_output = interval(0.017, 0.983);
    let node = GraphNode::new(
        "act",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["pre".to_string()],
    );
    let bounds = HashMap::from([
        ("pre".to_string(), producer),
        ("act".to_string(), activation_output),
    ]);
    let resolved =
        s_shaped_preactivation::resolve(&node, "act", &input, &bounds).expect("unique producer");
    assert!(std::ptr::eq(resolved, bounds.get("pre").unwrap()));

    let direct = GraphNode::from_input("direct", Layer::Sigmoid(SigmoidLayer::new()));
    let resolved = s_shaped_preactivation::resolve(&direct, "direct", &input, &bounds)
        .expect("network input producer");
    assert!(std::ptr::eq(resolved, &raw const input));
}

#[test]
fn resolver_fails_closed_on_missing_or_ambiguous_producer() {
    let input = interval(-1.0, 1.0);
    let missing = GraphNode::new(
        "missing",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["absent".to_string()],
    );
    let error = s_shaped_preactivation::resolve(&missing, "missing", &input, &HashMap::new())
        .expect_err("missing producer must be refused");
    assert!(matches!(error, NyError::InvalidSpec(message) if message.contains("absent")));

    let ambiguous = GraphNode::new(
        "ambiguous",
        Layer::Tanh(TanhLayer::new()),
        vec!["left".to_string(), "right".to_string()],
    );
    let error = s_shaped_preactivation::resolve(&ambiguous, "ambiguous", &input, &HashMap::new())
        .expect_err("ambiguous producer must be refused");
    assert!(matches!(error, NyError::InvalidSpec(message) if message.contains("exactly 1 input")));

    // Bypass the validating constructor to model a corrupt in-memory graph.
    let malformed = GraphNode {
        name: "malformed".to_string(),
        layer: Layer::Sigmoid(SigmoidLayer::new()),
        inputs: Vec::new(),
    };
    let error = s_shaped_preactivation::resolve(&malformed, "malformed", &input, &HashMap::new())
        .expect_err("missing unary identity must be refused");
    assert!(matches!(error, NyError::InvalidSpec(message) if message.contains("exactly 1 input")));
}

fn run_dag(kind: SShaped, lane: &str) {
    let (graph, input) = build_graph(kind);
    let config = AlphaCrownConfig {
        iterations: 2,
        gradient_method: GradientMethod::Spsa,
        ..AlphaCrownConfig::default()
    };
    let (bounds, state) = graph
        .collect_alpha_crown_bounds_dag(&input, &config)
        .expect("DAG-SPSA alpha collection");
    assert_registered_relaxation_encloses(kind, &bounds, &state, lane);
}

fn run_warm(kind: SShaped, lane: &str) {
    let (graph, input) = build_graph(kind);
    let config = AlphaCrownConfig {
        iterations: 2,
        ..AlphaCrownConfig::default()
    };
    let (bounds, state) = graph
        .collect_alpha_crown_bounds_dag_warm(&input, &config, &GraphAlphaState::new())
        .expect("warm-start alpha collection");
    assert_registered_relaxation_encloses(kind, &bounds, &state, lane);
}

#[test]
fn dag_spsa_sigmoid_and_tanh_lines_enclose_over_the_producer_box() {
    run_dag(SShaped::Sigmoid, "DAG-SPSA/Sigmoid");
    run_dag(SShaped::Tanh, "DAG-SPSA/Tanh");
}

#[test]
fn warm_start_sigmoid_and_tanh_lines_enclose_over_the_producer_box() {
    run_warm(SShaped::Sigmoid, "warm-start/Sigmoid");
    run_warm(SShaped::Tanh, "warm-start/Tanh");
}
