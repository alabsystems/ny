// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential alpha-CROWN routing tests.
//!
//! Verifies that the sequential entry point correctly routes models with
//! non-ReLU activations (Sigmoid, Tanh) to DAG alpha-CROWN rather than
//! falling back to fixed-slope CROWN with no alpha optimization.
//!
//! Part of #3619.

use crate::bounds::{AlphaCrownConfig, GradientMethod};
use crate::tests::crown::helpers::{assert_bounds_finite, CountingGemmEngine};
use crate::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

fn total_width(bounds: &BoundedTensor) -> f32 {
    let lower = bounds.lower();
    let upper = bounds.upper();
    (upper - lower).iter().sum::<f32>()
}

/// Pure Sigmoid sequential model: alpha-CROWN with SPSA gradients MUST produce
/// tighter bounds than fixed-slope CROWN via DAG alpha-CROWN routing.
///
/// The default gradient method (AnalyticChain) only computes ReLU gradients, so
/// monotone Sigmoid/Tanh tangent-point alphas get zero gradients. SPSA is the
/// only gradient method that perturbs monotone alpha parameters today (#3619).
#[ntest::timeout(60000)]
#[test]
fn test_sequential_pure_sigmoid_routes_to_dag_alpha_crown_3619() {
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[1.2_f32, -0.5], [-0.8, 1.3], [0.4, 0.7]]);
    let b1 = arr1(&[0.15_f32, -0.1, 0.05]);
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "sigmoid1",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.9_f32, -0.6, 0.4], [-0.3, 1.1, -0.5]]);
    let b2 = arr1(&[0.0_f32, 0.1]);
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()),
        vec!["sigmoid1".to_string()],
    ));
    graph.set_output("linear2");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[2]), 1.0_f32),
    )
    .unwrap();

    // IBP baseline.
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let ibp_width = total_width(&ibp_bounds);

    // Fixed-slope CROWN (no alpha optimization).
    let crown_bounds = graph
        .propagate_crown(&input)
        .expect("fixed-slope CROWN should succeed");
    let crown_width = total_width(&crown_bounds);
    eprintln!("pure Sigmoid seq: IBP={ibp_width:.6}, CROWN={crown_width:.6}");

    // Alpha-CROWN with SPSA gradients — the only gradient method that perturbs
    // monotone tangent-point alphas in the DAG verifier today.
    let config = AlphaCrownConfig {
        iterations: 50,
        gradient_method: GradientMethod::Spsa,
        spsa_samples: 4,
        fix_interm_bounds: false,
        ..AlphaCrownConfig::default()
    };
    let alpha_bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .expect("alpha-CROWN should succeed on pure Sigmoid sequential");
    let alpha_width = total_width(&alpha_bounds);
    let improvement_pct = 100.0 * (1.0 - alpha_width / crown_width);
    eprintln!("pure Sigmoid seq: alpha-CROWN(SPSA)={alpha_width:.6}, impr={improvement_pct:.2}%");

    // Soundness: alpha-CROWN must not produce wider bounds than IBP.
    assert!(
        alpha_width <= ibp_width + 1e-4,
        "alpha-CROWN ({alpha_width:.6}) wider than IBP ({ibp_width:.6}) — unsound"
    );

    // SPSA should produce measurable improvement over fixed-slope CROWN.
    assert!(
        alpha_width < crown_width - 1e-6,
        "alpha-CROWN SPSA ({alpha_width:.6}) not tighter than fixed-slope \
         CROWN ({crown_width:.6}). Either the routing fix is not active, or \
         SPSA cannot improve monotone tangent points on this model. (#3619)"
    );
}

/// AnalyticChain (default gradient method) routes to DAG alpha-CROWN and the
/// monotone SPSA supplement runs without error on a pure Sigmoid model.
///
/// Soundness test: alpha-CROWN with AnalyticChain MUST produce bounds no wider
/// than IBP. Tightness is NOT asserted because the crossing-zero Sigmoid tangent-
/// point constraint (`.min(d_lower)` / `.max(d_upper)`) prevents alpha optimization
/// from improving over the precomputed table defaults for crossing-zero intervals.
/// When all neurons cross zero (as in this symmetric-input model), the alpha
/// parameterization is fundamentally unable to improve element-wise bounds.
///
/// For tightness verification, see the SPSA test above which achieves 2.97%
/// improvement by perturbing all parameters jointly.
#[ntest::timeout(60000)]
#[test]
fn test_sequential_sigmoid_analytic_chain_soundness_3619() {
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[1.2_f32, -0.5], [-0.8, 1.3], [0.4, 0.7]]);
    let b1 = arr1(&[0.15_f32, -0.1, 0.05]);
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "sigmoid1",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.9_f32, -0.6, 0.4], [-0.3, 1.1, -0.5]]);
    let b2 = arr1(&[0.0_f32, 0.1]);
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()),
        vec!["sigmoid1".to_string()],
    ));
    graph.set_output("linear2");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[2]), 1.0_f32),
    )
    .unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let ibp_width = total_width(&ibp_bounds);

    // AnalyticChain: routes to DAG alpha-CROWN, runs monotone SPSA supplement.
    let config = AlphaCrownConfig {
        iterations: 20,
        gradient_method: GradientMethod::AnalyticChain,
        fix_interm_bounds: false,
        ..AlphaCrownConfig::default()
    };
    let alpha_bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .expect("alpha-CROWN AnalyticChain should succeed on pure Sigmoid");
    let alpha_width = total_width(&alpha_bounds);
    eprintln!("Sigmoid AnalyticChain: IBP={ibp_width:.6}, alpha={alpha_width:.6}");

    // Soundness: DAG alpha-CROWN with monotone supplement must not widen
    // bounds beyond IBP, even when tangent-point optimization oscillates.
    assert!(
        alpha_width <= ibp_width + 1e-4,
        "AnalyticChain alpha-CROWN ({alpha_width:.6}) wider than IBP \
         ({ibp_width:.6}) — unsound (#3619)"
    );
}

/// Build a Linear→LayerNorm→ReLU→Linear sequential graph for routing tests.
fn build_layernorm_relu_graph() -> (GraphNetwork, BoundedTensor) {
    use ndarray::Array1;
    let mut graph = GraphNetwork::new();
    let w1 = arr2(&[[1.2_f32, -0.5], [-0.8, 1.3], [0.4, 0.7], [-0.3, 0.9]]);
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(arr1(&[0.15_f32, -0.1, 0.05, 0.2]))).unwrap()),
    ));
    let ln = LayerNormLayer::new(Array1::ones(4), Array1::zeros(4), 1e-5).unwrap();
    graph.add_node(GraphNode::new(
        "layernorm1",
        Layer::LayerNorm(ln),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer::new()),
        vec!["layernorm1".to_string()],
    ));
    let w2 = arr2(&[[0.9_f32, -0.6, 0.4, -0.2], [-0.3, 1.1, -0.5, 0.7]]);
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, Some(arr1(&[0.0_f32, 0.1]))).unwrap()),
        vec!["relu1".to_string()],
    ));
    graph.set_output("linear2");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[2]), 1.0_f32),
    )
    .unwrap();
    (graph, input)
}

fn make_layernorm_alpha_routing_config(iterations: usize) -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations,
        gradient_method: GradientMethod::Spsa,
        spsa_samples: 4,
        fix_interm_bounds: false,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..AlphaCrownConfig::default()
    }
}

/// Sequential LayerNorm + ReLU: alpha-CROWN must not collapse to fixed-slope
/// CROWN when normalization is present. The old #3825 bug routed this shape to
/// plain CROWN, so equal widths alone are not enough to prove the fix: some
/// networks legitimately see zero width improvement even when alpha runs, and
/// the shared CROWN-IBP setup happens before the routing decision.
///
/// Instead, compare zero-iteration alpha-CROWN against a live optimizing run.
/// Both paths pay the same setup cost, so any extra GEMM work must come from
/// the routed DAG alpha optimizer rather than the pre-routing intermediates.
#[ntest::timeout(60000)]
#[test]
fn test_sequential_layernorm_relu_keeps_alpha_path_live_3825() {
    let (graph, input) = build_layernorm_relu_graph();
    let ibp_width = total_width(&graph.propagate_ibp(&input).unwrap());
    let crown_width = total_width(
        &graph
            .propagate_crown(&input)
            .expect("CROWN should succeed with LayerNorm"),
    );
    eprintln!("LayerNorm+ReLU seq: IBP={ibp_width:.6}, CROWN={crown_width:.6}");

    let baseline_engine = CountingGemmEngine::new();
    let baseline_bounds = graph
        .propagate_alpha_crown_with_config_and_engine(
            &input,
            &make_layernorm_alpha_routing_config(0),
            Some(&baseline_engine),
        )
        .expect("zero-iteration alpha-CROWN should succeed on LayerNorm + ReLU");
    assert_bounds_finite(
        &baseline_bounds,
        "#3825 zero-iteration LayerNorm alpha-CROWN with engine output",
    );
    let baseline_width = total_width(&baseline_bounds);

    let optimized_engine = CountingGemmEngine::new();
    let alpha_bounds = graph
        .propagate_alpha_crown_with_config_and_engine(
            &input,
            &make_layernorm_alpha_routing_config(2),
            Some(&optimized_engine),
        )
        .expect("alpha-CROWN should succeed on LayerNorm + ReLU sequential");
    assert_bounds_finite(
        &alpha_bounds,
        "#3825 routed LayerNorm alpha-CROWN with engine output",
    );
    let alpha_width = total_width(&alpha_bounds);
    eprintln!(
        "LayerNorm+ReLU: alpha0={baseline_width:.6}, alpha={alpha_width:.6}, impr={:.2}%",
        100.0 * (1.0 - alpha_width / crown_width)
    );

    assert!(
        (baseline_width - crown_width).abs() <= 1e-5,
        "#3825 setup regression: zero-iteration alpha-CROWN should match fixed-slope CROWN; alpha0={baseline_width:.6}, crown={crown_width:.6}"
    );
    assert!(
        alpha_width <= ibp_width + 1e-4,
        "alpha-CROWN ({alpha_width:.6}) wider than IBP ({ibp_width:.6}) — unsound"
    );
    assert!(
        alpha_width <= crown_width + 1e-5,
        "alpha-CROWN ({alpha_width:.6}) wider than CROWN ({crown_width:.6}) — regression"
    );
    assert!(
        baseline_engine.gemm_calls() > 0,
        "#3825 setup regression: zero-iteration LayerNorm alpha-CROWN should exercise GEMM"
    );
    assert!(
        optimized_engine.gemm_calls() > baseline_engine.gemm_calls(),
        "#3825 regression: routed LayerNorm alpha-CROWN should do more GEMM-backed \
         work than the zero-iteration alpha baseline; optimized_calls={} baseline_calls={}",
        optimized_engine.gemm_calls(),
        baseline_engine.gemm_calls(),
    );
}

/// Pure Sqrt sequential model: alpha-CROWN with SPSA gradients must route into
/// DAG alpha-CROWN so the optimizable tangent point is used instead of the
/// fixed tangent-at-upper relaxation.
#[ntest::timeout(60000)]
#[test]
fn test_sequential_pure_sqrt_routes_to_dag_alpha_crown_3773() {
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[0.6_f32, -0.2], [0.1, 0.4], [0.3, 0.1]]);
    let b1 = arr1(&[1.8_f32, 1.4, 2.0]);
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "sqrt1",
        Layer::Sqrt(SqrtLayer::new()),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.9_f32, -0.7, 0.5], [-0.4, 0.8, -0.6]]);
    let b2 = arr1(&[0.0_f32, 0.1]);
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()),
        vec!["sqrt1".to_string()],
    ));
    graph.set_output("linear2");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[2]), 1.0_f32),
    )
    .unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let ibp_width = total_width(&ibp_bounds);

    let crown_bounds = graph
        .propagate_crown(&input)
        .expect("fixed-slope sqrt CROWN should succeed");
    let crown_width = total_width(&crown_bounds);

    let config = AlphaCrownConfig {
        iterations: 40,
        gradient_method: GradientMethod::Spsa,
        spsa_samples: 4,
        fix_interm_bounds: false,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..AlphaCrownConfig::default()
    };
    let alpha_bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .expect("sqrt alpha-CROWN should succeed on pure sequential sqrt");
    let alpha_width = total_width(&alpha_bounds);

    assert!(
        alpha_width <= ibp_width + 1e-4,
        "sqrt alpha-CROWN ({alpha_width:.6}) wider than IBP ({ibp_width:.6}) — unsound"
    );
    assert!(
        alpha_width + 1e-5 < crown_width,
        "sqrt alpha-CROWN should beat fixed-slope CROWN. fixed={crown_width:.6}, alpha={alpha_width:.6}"
    );
}
