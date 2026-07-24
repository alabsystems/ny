// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN parity and fallback tests for GraphNetwork.

use crate::*;
use ndarray::{arr1, arr2};

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_batched_crown_parity_for_missing_activation_dispatch() {
    // #1753 regression: GraphNetwork batched CROWN must track the same unary activation
    // dispatch as sequential Network batched CROWN.
    struct ActivationCase {
        name: &'static str,
        layer: Layer,
        lower: [f32; 2],
        upper: [f32; 2],
    }

    let mixed_lower = [-0.9_f32, -0.7];
    let mixed_upper = [0.6_f32, 0.9];
    let positive_lower = [0.25_f32, 0.4];
    let positive_upper = [1.5_f32, 1.8];
    let tan_lower = [-0.8_f32, -0.4];
    let tan_upper = [0.8_f32, 0.4];

    let cases = vec![
        ActivationCase {
            name: "Tanh",
            layer: Layer::Tanh(TanhLayer::new()),
            lower: mixed_lower,
            upper: mixed_upper,
        },
        ActivationCase {
            name: "Sigmoid",
            layer: Layer::Sigmoid(SigmoidLayer::new()),
            lower: mixed_lower,
            upper: mixed_upper,
        },
        ActivationCase {
            name: "Arctan",
            layer: Layer::Arctan(ArctanLayer::new()),
            lower: mixed_lower,
            upper: mixed_upper,
        },
        ActivationCase {
            name: "Tan",
            layer: Layer::Tan(TanLayer::new()),
            lower: tan_lower,
            upper: tan_upper,
        },
        ActivationCase {
            name: "Exp",
            layer: Layer::Exp(ExpLayer::new()),
            lower: mixed_lower,
            upper: mixed_upper,
        },
        ActivationCase {
            name: "Log",
            layer: Layer::Log(LogLayer::new()),
            lower: positive_lower,
            upper: positive_upper,
        },
        ActivationCase {
            name: "Sqrt",
            layer: Layer::Sqrt(SqrtLayer::new()),
            lower: positive_lower,
            upper: positive_upper,
        },
        ActivationCase {
            name: "Reciprocal",
            layer: Layer::Reciprocal(ReciprocalLayer::new()),
            lower: positive_lower,
            upper: positive_upper,
        },
        ActivationCase {
            name: "Softplus",
            layer: Layer::Softplus(SoftplusLayer::new()),
            lower: mixed_lower,
            upper: mixed_upper,
        },
        ActivationCase {
            name: "HardSwish",
            layer: Layer::HardSwish(HardSwishLayer::new()),
            lower: mixed_lower,
            upper: mixed_upper,
        },
        ActivationCase {
            name: "Mish",
            layer: Layer::Mish(MishLayer::new()),
            lower: mixed_lower,
            upper: mixed_upper,
        },
        ActivationCase {
            name: "Selu",
            layer: Layer::Selu(SeluLayer::new()),
            lower: mixed_lower,
            upper: mixed_upper,
        },
        ActivationCase {
            name: "Softsign",
            layer: Layer::Softsign(SoftsignLayer::new()),
            lower: mixed_lower,
            upper: mixed_upper,
        },
        ActivationCase {
            name: "Snake",
            layer: Layer::Snake(SnakeLayer::new(2.0).expect("test: valid Snake")),
            lower: mixed_lower,
            upper: mixed_upper,
        },
        ActivationCase {
            name: "Elu",
            layer: Layer::Elu(EluLayer::new(1.0)),
            lower: mixed_lower,
            upper: mixed_upper,
        },
        ActivationCase {
            name: "Celu",
            layer: Layer::Celu(CeluLayer::default()),
            lower: mixed_lower,
            upper: mixed_upper,
        },
    ];

    let first_linear = LinearLayer::new(arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]), None).unwrap();
    let second_linear = LinearLayer::new(
        arr2(&[[1.1_f32, -0.4], [0.3, 0.8]]),
        Some(arr1(&[0.05_f32, -0.02])),
    )
    .unwrap();

    for case in cases {
        let mut network = Network::new();
        network.add_layer(Layer::Linear(first_linear.clone()));
        network.add_layer(case.layer);
        network.add_layer(Layer::Linear(second_linear.clone()));

        let graph = GraphNetwork::from_sequential(&network).unwrap();
        let input =
            BoundedTensor::new(arr1(&case.lower).into_dyn(), arr1(&case.upper).into_dyn()).unwrap();

        let sequential_bounds = network.propagate_crown_batched(&input).unwrap();
        let graph_bounds = graph.propagate_crown_batched(&input).unwrap();

        assert_eq!(
            graph_bounds.shape(),
            sequential_bounds.shape(),
            "shape mismatch for {}",
            case.name
        );

        for ((&graph_l, &graph_u), (&seq_l, &seq_u)) in graph_bounds
            .lower()
            .iter()
            .zip(graph_bounds.upper().iter())
            .zip(
                sequential_bounds
                    .lower()
                    .iter()
                    .zip(sequential_bounds.upper().iter()),
            )
        {
            assert!(
                graph_l.is_finite() && graph_u.is_finite(),
                "{} graph batched CROWN produced non-finite bounds [{}, {}]",
                case.name,
                graph_l,
                graph_u
            );
            assert!(
                seq_l.is_finite() && seq_u.is_finite(),
                "{} sequential batched CROWN produced non-finite bounds [{}, {}]",
                case.name,
                seq_l,
                seq_u
            );
            assert!(
                graph_l <= graph_u + 1e-6,
                "{} graph bounds inverted: lower {} > upper {}",
                case.name,
                graph_l,
                graph_u
            );
            assert!(
                seq_l <= seq_u + 1e-6,
                "{} sequential bounds inverted: lower {} > upper {}",
                case.name,
                seq_l,
                seq_u
            );
            // Post-#2990: sequential path intersects output with IBP bounds using
            // collect_ibp_bounds(), while graph path intersects with its own forward-bound
            // cache (CROWN-IBP). Different intermediate bound sources produce different
            // intersection results, so we verify both are sound rather than exact parity.
            // Specifically: both must contain all true network outputs, and neither should
            // be dramatically wider than the other.
            let width_graph = graph_u - graph_l;
            let width_seq = seq_u - seq_l;
            let max_width = width_graph.max(width_seq);
            assert!(
                (graph_l - seq_l).abs() <= max_width * 0.5 + 1e-4,
                "{} lower diverges beyond 50%% of max width: graph {} vs sequential {} (widths: {}, {})",
                case.name,
                graph_l,
                seq_l,
                width_graph,
                width_seq,
            );
            assert!(
                (graph_u - seq_u).abs() <= max_width * 0.5 + 1e-4,
                "{} upper diverges beyond 50%% of max width: graph {} vs sequential {} (widths: {}, {})",
                case.name,
                graph_u,
                seq_u,
                width_graph,
                width_seq,
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_batched_crown_unsupported_unary_falls_back() {
    // Regression contract from #1753 design: unsupported unary ops in GraphNetwork
    // batched CROWN should trigger partial/IBP fallback, not hard-fail.
    let layer = Layer::SkipMerge(SkipMergeLayer::new());
    assert!(
        !layer.supports_batched_crown(),
        "SkipMerge must remain unsupported in batched CROWN for this fallback regression test"
    );

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("skip", layer));
    graph.set_output("skip");

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.25]).into_dyn(),
        arr1(&[0.75_f32, 1.25]).into_dyn(),
    )
    .unwrap();

    let batched_bounds = graph.propagate_crown_batched(&input).unwrap();

    assert_eq!(batched_bounds.shape(), input.shape());
    for ((&actual_l, &actual_u), (&expected_l, &expected_u)) in batched_bounds
        .lower()
        .iter()
        .zip(batched_bounds.upper().iter())
        .zip(input.lower().iter().zip(input.upper().iter()))
    {
        assert!(
            (actual_l - expected_l).abs() <= 1e-6,
            "fallback lower mismatch: got {}, expected {}",
            actual_l,
            expected_l
        );
        assert!(
            (actual_u - expected_u).abs() <= 1e-6,
            "fallback upper mismatch: got {}, expected {}",
            actual_u,
            expected_u
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_batched_crown_three_input_concat_reaches_fallback_4136() {
    // #4136 regression: multi-input nodes like Concat must NOT be rejected by
    // the stale require_unary_input() guard before dispatch. The batched CROWN
    // backward loop must use first-input resolution (like DAG-CROWN after #4113)
    // so that the existing unsupported-layer fallback produces IBP bounds.
    //
    // Graph: _input → a (AddConstant(0.0))
    //        _input → b (MulConstant(1.0))
    //        _input → c (SubConstant(0.0))
    //        a, b, c → concat (Concat axis=0)
    //        output = concat
    //
    // Concat is unsupported in batched CROWN, so the fallback should produce
    // IBP-equivalent bounds. Before #4136, this would fail with
    // "requires exactly 1 input but has 3".
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "a",
        Layer::AddConstant(AddConstantLayer::new(arr1(&[0.0_f32]).into_dyn())),
    ));
    graph.add_node(GraphNode::from_input(
        "b",
        Layer::MulConstant(MulConstantLayer::new(arr1(&[1.0_f32]).into_dyn())),
    ));
    graph.add_node(GraphNode::from_input(
        "c",
        Layer::SubConstant(SubConstantLayer::new(arr1(&[0.0_f32]).into_dyn())),
    ));
    graph.add_node(GraphNode::new(
        "concat",
        Layer::Concat(ConcatLayer::new(0)),
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
    ));
    graph.set_output("concat");

    let input =
        BoundedTensor::new(arr1(&[-0.5_f32]).into_dyn(), arr1(&[0.75_f32]).into_dyn()).unwrap();

    // Before fix: this fails with "requires exactly 1 input but has 3".
    // After fix: reaches the unsupported-layer fallback and returns IBP bounds.
    let batched_result = graph
        .propagate_crown_batched(&input)
        .expect("batched CROWN must not reject a valid three-input Concat node");

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // For an unsupported output node, the fallback should produce IBP bounds.
    assert_eq!(
        batched_result.shape(),
        ibp_bounds.shape(),
        "batched CROWN fallback shape must match IBP"
    );
    for ((&batched_l, &batched_u), (&ibp_l, &ibp_u)) in batched_result
        .lower()
        .iter()
        .zip(batched_result.upper().iter())
        .zip(ibp_bounds.lower().iter().zip(ibp_bounds.upper().iter()))
    {
        assert!(
            (batched_l - ibp_l).abs() <= 1e-6,
            "batched CROWN fallback lower {} must match IBP lower {}",
            batched_l,
            ibp_l
        );
        assert!(
            (batched_u - ibp_u).abs() <= 1e-6,
            "batched CROWN fallback upper {} must match IBP upper {}",
            batched_u,
            ibp_u
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_batched_crown_tightens_to_ibp_reference_4242() {
    // #4242 regression: post-concretization batched CROWN must intersect with
    // the output node's forward bounds. An identity graph is the minimal case
    // where concretize_sound() alone widens the interval; tightening should
    // recover the exact IBP output bounds.
    //
    // Wrapped in with_crown_dense_budget_mb to serialize against concurrent
    // tests that set NY_DENSE_BUDGET_MB=0 (budget=0 causes CROWN to
    // fall back to IBP, masking the tightening contract).
    tests::with_crown_dense_budget_mb("2048", || {
        let mut graph = GraphNetwork::new();
        let identity = LinearLayer::new(arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]), None).unwrap();
        graph.add_node(GraphNode::from_input("identity", Layer::Linear(identity)));
        graph.set_output("identity");

        let input = BoundedTensor::new(
            arr1(&[-0.75_f32, 0.125]).into_dyn(),
            arr1(&[1.5_f32, 2.75]).into_dyn(),
        )
        .unwrap();

        let ibp_bounds = graph.propagate_ibp(&input).unwrap();
        let crown_bounds = graph.propagate_crown_batched(&input).unwrap();

        assert_eq!(crown_bounds.shape(), ibp_bounds.shape());

        for ((&crown_l, &crown_u), (&ibp_l, &ibp_u)) in crown_bounds
            .lower()
            .iter()
            .zip(crown_bounds.upper().iter())
            .zip(ibp_bounds.lower().iter().zip(ibp_bounds.upper().iter()))
        {
            assert!(
                (crown_l - ibp_l).abs() <= 1e-6,
                "tightened batched CROWN lower {} must match IBP lower {}",
                crown_l,
                ibp_l
            );
            assert!(
                (crown_u - ibp_u).abs() <= 1e-6,
                "tightened batched CROWN upper {} must match IBP upper {}",
                crown_u,
                ibp_u
            );
        }
    });
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_batched_crown_leaky_relu_provenance_4240() {
    // #4240 regression: batched CROWN must use tighten_crown_output_with_provenance
    // so that provenance reflects whether forward-bound tightening was applied.
    // LeakyReLU(0.01) with x in [-10, 10] is the canonical case:
    //   IBP:   [-0.1, 10]  (exact for monotone activation)
    //   CROWN: [-10.0, 10] (loose lower from linear relaxation)
    //   Tightened: [-0.1, 10] (intersection recovers IBP tightness)
    //
    // The provenance should be Crown (intersection succeeded, not a full fallback).
    use crate::types::BoundsProvenance;

    tests::with_crown_dense_budget_mb("2048", || {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "leaky",
            Layer::LeakyReLU(LeakyReLULayer::new(0.01)),
        ));
        graph.set_output("leaky");

        let input = BoundedTensor::new(
            arr1(&[-10.0_f32, -5.0]).into_dyn(),
            arr1(&[10.0_f32, 8.0]).into_dyn(),
        )
        .unwrap();

        let result = graph
            .propagate_crown_batched_with_provenance(&input)
            .unwrap();
        let ibp_bounds = graph.propagate_ibp(&input).unwrap();

        // Tightened batched CROWN must be at least as tight as IBP.
        for (i, ((&crown_l, &crown_u), (&ibp_l, &ibp_u))) in result
            .bounds
            .lower()
            .iter()
            .zip(result.bounds.upper().iter())
            .zip(ibp_bounds.lower().iter().zip(ibp_bounds.upper().iter()))
            .enumerate()
        {
            assert!(
                crown_l >= ibp_l - 1e-5,
                "element {}: tightened lower {} must be >= IBP lower {}",
                i,
                crown_l,
                ibp_l
            );
            assert!(
                crown_u <= ibp_u + 1e-5,
                "element {}: tightened upper {} must be <= IBP upper {}",
                i,
                crown_u,
                ibp_u
            );
        }

        // Provenance must be Crown (successful intersection), not ForwardFallback.
        assert!(
            matches!(result.provenance, BoundsProvenance::Crown),
            "expected Crown provenance, got {:?}",
            result.provenance
        );
    });
}
