// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Facade contract for the layer and graph-inspection surface a graph
//! *translator* needs.
//!
//! Every symbol here is reached through an `ny_api::*` path only — there is no
//! `ny_propagate` import in this file, and that is the point. A downstream
//! translator (nn's `graph_translate` / `graph_tensor_*` modules) has to name a
//! layer type to build a [`Layer`]; a type missing from `ny_api::layers` is the
//! whole reason such a consumer keeps a direct `ny-propagate` dependency
//! alongside its facade-only architecture rule.
//!
//! The layer assertions are deliberately CONSTRUCTIVE: each type is built and
//! wrapped in a `Layer`, so an export that survived as a name but lost its
//! constructor still fails here.

#![allow(unused_qualifications)]

use ndarray::{arr1, arr2, ArrayD, IxDyn};

use ny_api::graph::{
    count_sqrt_negative_domain_graph, count_sqrt_negative_domain_network,
    soundness_provenance_for_graph, soundness_provenance_for_network, BoundsProvenance,
    CrownIbpFallbackEvent, CrownIbpFallbackReason, GraphCrownIbpBoundsResult, GraphNetwork,
    GraphNode, SequentialNetwork, NETWORK_INPUT,
};
use ny_api::layers::{
    AbsLayer, AddConstantLayer, AddLayer, Atan2Layer, AveragePoolLayer, CausalSoftmaxLayer,
    ClipLayer, ConcatLayer, ConvTranspose2dLayer, CosLayer, DivConstantLayer, DivLayer,
    EpsilonExpandLayer, ExpLayer, Layer, LinearLayer, LogSoftmaxLayer, MatMulLayer, MaxBinaryLayer,
    MaxPool2dLayer, MinBinaryLayer, MulBinaryLayer, MulConstantLayer, PowConstantLayer, ReLULayer,
    ReciprocalLayer, ReduceMaxLayer, ReduceMeanLayer, ReduceMinLayer, ReduceSumLayer, ReshapeLayer,
    SinLayer, SliceLayer, SqrtLayer, SqueezeLayer, SubConstantLayer, SubLayer, TanhLayer,
    TileLayer, TransposeLayer, UnsqueezeLayer, WhereLayer,
};
use ny_api::verify::{PropagationMethod, Verifier};
use ny_api::{Bound, BoundedTensor, VerificationSpec};

fn scalar_constant(value: f32) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(&[1]), value)
}

/// Build every layer type that `nn-verify` names directly, through the facade.
///
/// The returned list is what the facade must be able to express; the test below
/// only has to observe that each entry exists and reports a layer kind.
fn every_translator_layer() -> Vec<Layer> {
    vec![
        // --- the 20 kinds a translator previously had to reach past the
        // facade for ---
        Layer::Atan2(Atan2Layer),
        Layer::AveragePool(AveragePoolLayer::new((2, 2), (2, 2), (0, 0), false)),
        Layer::CausalSoftmax(CausalSoftmaxLayer::new(-1)),
        Layer::ConvTranspose2d(
            ConvTranspose2dLayer::new(
                ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), 0.5),
                Some(arr1(&[0.0])),
                (1, 1),
                (0, 0),
            )
            .expect("valid transposed convolution"),
        ),
        Layer::Cos(CosLayer::new()),
        Layer::DivConstant(DivConstantLayer::new(scalar_constant(2.0))),
        Layer::Div(DivLayer),
        Layer::LogSoftmax(LogSoftmaxLayer::new(-1)),
        Layer::MaxBinary(MaxBinaryLayer),
        Layer::MaxPool2d(MaxPool2dLayer::new((2, 2), (2, 2), (0, 0))),
        Layer::MinBinary(MinBinaryLayer),
        Layer::Reciprocal(ReciprocalLayer),
        Layer::ReduceMax(ReduceMaxLayer::new(vec![-1], false)),
        Layer::ReduceMean(ReduceMeanLayer::new(vec![-1], false)),
        Layer::ReduceMin(ReduceMinLayer::new(vec![-1], false)),
        Layer::ReduceSum(ReduceSumLayer::new(vec![-1], false)),
        Layer::Sin(SinLayer::new()),
        Layer::Squeeze(SqueezeLayer::new(0)),
        Layer::SubConstant(SubConstantLayer::new(scalar_constant(1.0))),
        Layer::Unsqueeze(UnsqueezeLayer::new(0)),
        // --- kinds the facade already carried; kept so a regression that
        // narrows the surface is caught here too ---
        Layer::Abs(AbsLayer),
        Layer::Add(AddLayer),
        Layer::AddConstant(AddConstantLayer::new(scalar_constant(1.0))),
        Layer::Clip(ClipLayer::new(0.0, 1.0)),
        Layer::Concat(ConcatLayer::new(0)),
        Layer::EpsilonExpand(EpsilonExpandLayer::uniform(0.1).expect("valid epsilon")),
        Layer::Exp(ExpLayer::new()),
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0]))).expect("valid linear"),
        ),
        Layer::MatMul(MatMulLayer::new(false, None)),
        Layer::MulBinary(MulBinaryLayer),
        Layer::MulConstant(MulConstantLayer::scalar(2.0)),
        Layer::PowConstant(PowConstantLayer::new(2.0)),
        Layer::ReLU(ReLULayer::new()),
        Layer::Reshape(ReshapeLayer::new(vec![1])),
        Layer::Slice(SliceLayer::new(0, 0, 1)),
        Layer::Sqrt(SqrtLayer::new()),
        Layer::Sub(SubLayer),
        Layer::Tanh(TanhLayer::new()),
        Layer::Tile(TileLayer::new(0, 2)),
        Layer::Transpose(TransposeLayer::new(vec![0])),
        Layer::Where(WhereLayer::new()),
    ]
}

#[test]
fn every_translator_layer_is_constructible_through_the_facade() {
    let layers = every_translator_layer();
    assert_eq!(
        layers.len(),
        41,
        "the constructive list must cover every layer this contract pins"
    );
    for layer in &layers {
        assert!(
            !layer.layer_type().is_empty(),
            "a facade layer must report a kind"
        );
    }
}

/// The 20 kinds that used to force a direct `ny-propagate` dependency, named
/// one by one so a silent removal from `ny_api::layers` fails loudly here.
#[test]
fn the_twenty_previously_missing_kinds_are_present_by_name() {
    let expected = [
        "Atan2",
        "AveragePool",
        "CausalSoftmax",
        "ConvTranspose2d",
        "Cos",
        "DivConstant",
        "Div",
        "LogSoftmax",
        "MaxBinary",
        "MaxPool2d",
        "MinBinary",
        "Reciprocal",
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceSum",
        "Sin",
        "Squeeze",
        "SubConstant",
        "Unsqueeze",
    ];
    let kinds: Vec<&str> = every_translator_layer()
        .iter()
        .map(Layer::layer_type)
        .collect();
    for kind in expected {
        assert!(
            kinds.contains(&kind),
            "'{kind}' must be constructible through ny_api::layers; kinds = {kinds:?}"
        );
    }
}

/// A newly exported layer must not merely be nameable — it must propagate
/// through the facade verifier as part of a real graph.
#[test]
fn newly_exported_layers_propagate_through_the_facade_verifier() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "sub",
        Layer::SubConstant(SubConstantLayer::new(scalar_constant(1.0))),
    ));
    graph.add_node(GraphNode::new(
        "div",
        Layer::DivConstant(DivConstantLayer::new(scalar_constant(2.0))),
        vec!["sub".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "sum",
        Layer::ReduceSum(ReduceSumLayer::new(vec![-1], false)),
        vec!["div".to_string()],
    ));
    graph.set_output("sum");

    // x in [0, 2] -> (x - 1) / 2 in [-0.5, 0.5] -> sum over the single axis.
    let spec = VerificationSpec::new(
        vec![Bound::new(0.0, 2.0), Bound::new(0.0, 2.0)],
        vec![Bound::new(-1.5, 1.5)],
    )
    .expect("valid spec");
    let verifier = Verifier::new(ny_api::verify::PropagationConfig {
        method: PropagationMethod::Ibp,
        ..Default::default()
    });
    let result = verifier
        .verify_graph(&graph, &spec)
        .expect("a graph built only from facade layer types propagates");
    assert!(
        result.is_verified(),
        "the true range is [-1, 1], so the spec must verify: {result:?}"
    );
}

/// The soundness-provenance pair: a graph carrier AND a sequential carrier.
///
/// `soundness_provenance_for_network` is the one a streaming consumer needs,
/// and it had no `ny_api` equivalent before this contract.
#[test]
fn soundness_provenance_is_reachable_for_both_carriers() {
    let linear = LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0]))).expect("valid linear");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("sqrt", Layer::Sqrt(SqrtLayer::new())));
    graph.set_output("sqrt");
    let graph_provenance = soundness_provenance_for_graph(&graph, &PropagationMethod::Crown);

    let mut network = SequentialNetwork::new();
    network.add_layer(Layer::Linear(linear));
    let network_provenance = soundness_provenance_for_network(&network, &PropagationMethod::Crown);
    assert!(
        network_provenance.heuristics_used().is_empty(),
        "a plain Linear chain uses no heuristics: {network_provenance:?}"
    );
    // Both carriers answer the same question; only the input type differs.
    let _ = graph_provenance;

    // The sqrt-domain guards, same pairing.
    let bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1]), 1.0_f32),
    )
    .expect("valid bounds");
    let graph_negative =
        count_sqrt_negative_domain_graph(&graph, &bounds).expect("graph sqrt scan runs");
    assert_eq!(
        graph_negative, 1,
        "a [-1, 1] input reaches the negative sqrt domain once"
    );

    let mut sqrt_network = SequentialNetwork::new();
    sqrt_network.add_layer(Layer::Sqrt(SqrtLayer::new()));
    let network_negative = count_sqrt_negative_domain_network(&sqrt_network, &bounds)
        .expect("sequential sqrt scan runs");
    assert_eq!(
        network_negative, graph_negative,
        "the two carriers must agree on the same layer"
    );
}

/// The CROWN-IBP node-bound collection result and its component types.
///
/// The collection API was always callable through the exported `GraphNetwork`,
/// but its RESULT was unnameable through `ny_api`, so a consumer that wanted
/// per-node bound reporting had to import `ny_propagate::types`.
#[test]
fn crown_ibp_collection_result_is_nameable_and_inspectable() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "l1",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0_f32, -1.0], [-1.0, 1.0]]),
                Some(arr1(&[0.0, 0.0])),
            )
            .expect("valid linear"),
        ),
    ));
    graph.add_node(GraphNode::new(
        "r1",
        Layer::ReLU(ReLULayer::new()),
        vec!["l1".to_string()],
    ));
    graph.set_output("r1");
    assert_eq!(
        graph.node("l1").expect("node exists").inputs(),
        [NETWORK_INPUT.to_string()]
    );

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[2]), 1.0_f32),
    )
    .expect("valid input box");

    // The binding is the contract: this type has to be nameable through the
    // facade for a consumer to write this function signature at all.
    let result: GraphCrownIbpBoundsResult = graph
        .collect_crown_ibp_bounds_dag_with_status(&input)
        .expect("collection runs");

    assert!(
        result.bounds.contains_key("l1") && result.bounds.contains_key("r1"),
        "every node must be keyed in the result"
    );
    let provenance: BoundsProvenance = result
        .provenance_for_node("l1")
        .expect("the linear node has provenance");
    // Both variants are reachable through the facade; only one holds here.
    match provenance {
        BoundsProvenance::Crown => {}
        BoundsProvenance::ForwardFallback(reason) => {
            let _: CrownIbpFallbackReason = reason;
        }
    }
    let events: &Vec<CrownIbpFallbackEvent> = &result.fallback_events;
    assert_eq!(
        events.len(),
        result.fallback_count(),
        "the event list and the count must agree"
    );
    // The question a consumer actually needs to ask: structural omissions are
    // not incomplete work.
    let _completed: bool = result.all_requested_crown_targets_completed();
}
