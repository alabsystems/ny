// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for the batched resnet point-VJP plan (#batched-vjp-resnet): plan
//! structure (segments + flat mask slots), template-forward ↔ graph-forward
//! parity (outputs and masks), and the unified wave-plan routing.

use super::{point_vjp_resnet_forward_masks, PointVjpWavePlan};
use crate::layers::{AddLayer, Conv2dLayer, FlattenLayer, Layer, LinearLayer, ReLULayer};
use crate::network::core::{GraphNetwork, GraphNode, NETWORK_INPUT};
use ndarray::{Array1, Array2, Array4, ArrayD, IxDyn};
use ny_core::GpuResnetSegment;
use ny_tensor::BoundedTensor;

/// Deterministic xorshift for reproducible fixtures.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn conv(rng: &mut Rng, out_c: usize, in_c: usize) -> Conv2dLayer {
    let kernel = Array4::from_shape_fn((out_c, in_c, 3, 3), |_| rng.next_f32() * 0.4).into_dyn();
    let bias = Array1::from_shape_fn(out_c, |_| rng.next_f32() * 0.1);
    Conv2dLayer::new(kernel, Some(bias), (1, 1), (1, 1)).expect("conv layer")
}

/// Small conv RESNET: input [1,4,4] → conv1(1→2) → relu1 → [F: conv2(2→2) →
/// relu2] → add(relu2, relu1) → flatten → lin1(32→3) → relu3 → lin2(3→2).
/// One identity-skip residual block; three ReLU mask slots.
fn conv_resnet_fixture() -> (GraphNetwork, BoundedTensor) {
    let mut rng = Rng(0x5EED_BEEF);
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "conv1",
        Layer::Conv2d(conv(&mut rng, 2, 1)),
    ));
    g.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["conv1".into()],
    ));
    g.add_node(GraphNode::new(
        "conv2",
        Layer::Conv2d(conv(&mut rng, 2, 2)),
        vec!["relu1".into()],
    ));
    g.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["conv2".into()],
    ));
    g.add_node(GraphNode::new(
        "add",
        Layer::Add(AddLayer),
        vec!["relu2".into(), "relu1".into()],
    ));
    g.add_node(GraphNode::new(
        "flat",
        Layer::Flatten(FlattenLayer::new(0)),
        vec!["add".into()],
    ));
    let w1 = Array2::from_shape_fn((3, 32), |_| rng.next_f32() * 0.5);
    g.add_node(GraphNode::new(
        "lin1",
        Layer::Linear(
            LinearLayer::new(w1, Some(Array1::from_vec(vec![0.1, -0.2, 0.05]))).expect("lin1"),
        ),
        vec!["flat".into()],
    ));
    g.add_node(GraphNode::new(
        "relu3",
        Layer::ReLU(ReLULayer),
        vec!["lin1".into()],
    ));
    let w2 = Array2::from_shape_fn((2, 3), |_| rng.next_f32());
    g.add_node(GraphNode::new(
        "lin2",
        Layer::Linear(LinearLayer::new(w2, None).expect("lin2")),
        vec!["relu3".into()],
    ));
    g.set_output("lin2");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), 1.0_f32),
    )
    .expect("input box");
    (g, input)
}

#[test]
fn resnet_plan_builds_with_expected_segments_and_slots() {
    let (g, input) = conv_resnet_fixture();
    let plan = g.build_point_vjp_resnet_plan(&input).expect("resnet plan");
    assert_eq!(plan.input_dim, 16);
    assert_eq!(plan.output_dim, 2);
    // Backward segments: Chain([lin2, relu3, lin1]) (flatten folds away),
    // Residual([relu2, conv2]), Chain([relu1, conv1]).
    assert_eq!(plan.segments_backward.len(), 3);
    match &plan.segments_backward[0] {
        GpuResnetSegment::Chain(l) => assert_eq!(l.len(), 3),
        _ => panic!("expected Chain"),
    }
    match &plan.segments_backward[1] {
        GpuResnetSegment::Residual(f) => assert_eq!(f.len(), 2),
        _ => panic!("expected Residual"),
    }
    match &plan.segments_backward[2] {
        GpuResnetSegment::Chain(l) => assert_eq!(l.len(), 2),
        _ => panic!("expected Chain"),
    }
    // Flat traversal: 0=lin2 1=relu3 2=lin1 | 3=relu2 4=conv2 | 5=relu1 6=conv1.
    assert_eq!(plan.mask_flat_positions, vec![1, 3, 5]);
    assert_eq!(
        plan.relu_nodes,
        vec![
            "relu3".to_string(),
            "relu2".to_string(),
            "relu1".to_string()
        ]
    );
    // The pure-chain plan must refuse this graph (fan-in), and the unified plan
    // must route to the resnet template.
    assert!(g.build_point_vjp_batch_plan(&input).is_none());
    match PointVjpWavePlan::build(&g, &input) {
        Some(PointVjpWavePlan::Resnet(_)) => {}
        _ => panic!("wave plan must pick the resnet template for a residual DAG"),
    }
}

#[test]
fn resnet_forward_masks_match_graph_forward_signs_and_outputs() {
    let (g, input) = conv_resnet_fixture();
    let plan = g.build_point_vjp_resnet_plan(&input).expect("resnet plan");

    let mut rng = Rng(0xD1CE_D00D);
    let points: Vec<Vec<f32>> = (0..4)
        .map(|_| (0..plan.input_dim).map(|_| rng.next_f32()).collect())
        .collect();
    let (masks, outputs) = point_vjp_resnet_forward_masks(&plan, &points).expect("batched forward");

    for (k, point) in points.iter().enumerate() {
        let x = ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), point.clone()).expect("shape");
        let node_bounds = g
            .collect_node_bounds(&BoundedTensor::concrete(x.clone()).expect("concrete"))
            .expect("node bounds");
        // Graph point forward output vs template forward output.
        let out = node_bounds.get("lin2").expect("output node").center();
        for (a, b) in out.iter().zip(outputs[k].iter()) {
            assert!(
                (a - b).abs() <= 1e-4 * (1.0 + a.abs()),
                "output mismatch at restart {k}: graph={a} template={b}"
            );
        }
        // Mask parity per ReLU slot: the ReLU's INPUT node signs.
        for (r, relu_name) in plan.relu_nodes.iter().enumerate() {
            let pre_name = g
                .node(relu_name)
                .expect("relu node")
                .require_unary_input()
                .expect("unary");
            let pre = node_bounds.get(pre_name).expect("pre node").center();
            assert_eq!(masks[k][r].len(), pre.len());
            for (i, (&m, &v)) in masks[k][r].iter().zip(pre.iter()).enumerate() {
                if v.abs() < 1e-5 {
                    continue; // razor-edge: rounding may differ across 0
                }
                let expected = if v > 0.0 { 1.0 } else { 0.0 };
                assert_eq!(
                    m, expected,
                    "mask mismatch restart {k} relu {relu_name} neuron {i} (pre={v})"
                );
            }
        }
    }
}

/// Projection-skip variant: add(conv2(relu1), conv3(relu1)) — both branches
/// non-empty → ResidualProj with F-then-P flat slot ordering.
#[test]
fn resnet_plan_handles_projection_skip() {
    let mut rng = Rng(0xFEED_5EED);
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "conv1",
        Layer::Conv2d(conv(&mut rng, 2, 1)),
    ));
    g.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["conv1".into()],
    ));
    g.add_node(GraphNode::new(
        "conv2",
        Layer::Conv2d(conv(&mut rng, 2, 2)),
        vec!["relu1".into()],
    ));
    g.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["conv2".into()],
    ));
    g.add_node(GraphNode::new(
        "conv3",
        Layer::Conv2d(conv(&mut rng, 2, 2)),
        vec!["relu1".into()],
    ));
    g.add_node(GraphNode::new(
        "add",
        Layer::Add(AddLayer),
        vec!["relu2".into(), "conv3".into()],
    ));
    g.add_node(GraphNode::new(
        "flat",
        Layer::Flatten(FlattenLayer::new(0)),
        vec!["add".into()],
    ));
    let w = Array2::from_shape_fn((2, 32), |_| rng.next_f32() * 0.3);
    g.add_node(GraphNode::new(
        "lin",
        Layer::Linear(LinearLayer::new(w, None).expect("lin")),
        vec!["flat".into()],
    ));
    g.set_output("lin");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), 1.0_f32),
    )
    .expect("input box");
    let plan = g.build_point_vjp_resnet_plan(&input).expect("proj plan");
    // Segments: Chain([lin]), ResidualProj(F=[relu2, conv2], P=[conv3]),
    // Chain([relu1, conv1]).
    assert_eq!(plan.segments_backward.len(), 3);
    match &plan.segments_backward[1] {
        GpuResnetSegment::ResidualProj(f, p) => {
            assert_eq!(f.len(), 2);
            assert_eq!(p.len(), 1);
        }
        _ => panic!("expected ResidualProj"),
    }
    // Flat: 0=lin | 1=relu2 2=conv2 3=conv3 | 4=relu1 5=conv1.
    assert_eq!(plan.mask_flat_positions, vec![1, 4]);
    assert_eq!(
        plan.relu_nodes,
        vec!["relu2".to_string(), "relu1".to_string()]
    );

    // Forward parity at one point.
    let point: Vec<f32> = (0..plan.input_dim).map(|_| rng.next_f32()).collect();
    let (_masks, outputs) =
        point_vjp_resnet_forward_masks(&plan, std::slice::from_ref(&point)).expect("forward");
    let x = ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), point).expect("shape");
    let node_bounds = g
        .collect_node_bounds(&BoundedTensor::concrete(x).expect("concrete"))
        .expect("node bounds");
    let out = node_bounds.get("lin").expect("output").center();
    for (a, b) in out.iter().zip(outputs[0].iter()) {
        assert!(
            (a - b).abs() <= 1e-4 * (1.0 + a.abs()),
            "proj output mismatch: graph={a} template={b}"
        );
    }
}

// Silence unused-import warning for NETWORK_INPUT if the fixture never names it
// explicitly (kept for readability parity with the chain tests).
#[allow(unused)]
const _: &str = NETWORK_INPUT;
