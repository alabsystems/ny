// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for the batched point-VJP plan (#batched-vjp): plan structure, mask ↔
//! CPU-forward sign parity, and template-forward ↔ graph-forward output parity.

use super::point_vjp_forward_masks;
use crate::layers::{Conv2dLayer, FlattenLayer, Layer, LinearLayer, ReLULayer};
use crate::network::core::{GraphNetwork, GraphNode};
use ndarray::{Array1, Array2, Array4, ArrayD, IxDyn};
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

/// Small conv chain: input [1,4,4] → Conv2d(1→2, 3x3, pad 1) → ReLU → Flatten →
/// Linear(32→3) → ReLU → Linear(3→2). Two ReLU mask slots.
fn conv_chain_fixture() -> (GraphNetwork, BoundedTensor) {
    let mut rng = Rng(0x5EED_CAFE);
    let mut g = GraphNetwork::new();

    let kernel = Array4::from_shape_fn((2, 1, 3, 3), |_| rng.next_f32()).into_dyn();
    let conv = Conv2dLayer::new(
        kernel,
        Some(Array1::from_vec(vec![0.05, -0.03])),
        (1, 1),
        (1, 1),
    )
    .expect("conv layer");
    g.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv)));
    g.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["conv1".into()],
    ));
    g.add_node(GraphNode::new(
        "flat",
        Layer::Flatten(FlattenLayer::new(0)),
        vec!["relu1".into()],
    ));
    let w1 = Array2::from_shape_fn((3, 32), |_| rng.next_f32() * 0.5);
    let b1 = Array1::from_vec(vec![0.1, -0.2, 0.05]);
    g.add_node(GraphNode::new(
        "lin1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("lin1")),
        vec!["flat".into()],
    ));
    g.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["lin1".into()],
    ));
    let w2 = Array2::from_shape_fn((2, 3), |_| rng.next_f32());
    g.add_node(GraphNode::new(
        "lin2",
        Layer::Linear(LinearLayer::new(w2, None).expect("lin2")),
        vec!["relu2".into()],
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
fn plan_builds_on_conv_chain_with_expected_slots() {
    let (g, input) = conv_chain_fixture();
    let plan = g
        .build_point_vjp_batch_plan(&input)
        .expect("conv chain plan");
    assert_eq!(plan.input_dim, 16);
    assert_eq!(plan.output_dim, 2);
    // Backward order: lin2, relu2(Act), lin1, relu1(Act), conv1 — Flatten folded.
    assert_eq!(plan.layers_backward.len(), 5);
    assert_eq!(plan.mask_positions, vec![1, 3]);
    assert_eq!(
        plan.relu_nodes_backward,
        vec!["relu2".to_string(), "relu1".to_string()],
        "fold-order ReLU names must be backward order (output→input)"
    );
}

#[test]
fn plan_refuses_residual_add() {
    // input → l1 → relu → l2 → add(l2, l1): fan-in must refuse (fail-closed).
    let mut g = GraphNetwork::new();
    let w = Array2::from_shape_fn((4, 4), |(i, j)| if i == j { 0.9 } else { 0.1 });
    g.add_node(GraphNode::from_input(
        "l1",
        Layer::Linear(LinearLayer::new(w.clone(), None).expect("l1")),
    ));
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["l1".into()],
    ));
    g.add_node(GraphNode::new(
        "l2",
        Layer::Linear(LinearLayer::new(w, None).expect("l2")),
        vec!["relu".into()],
    ));
    g.add_node(GraphNode::new(
        "add",
        Layer::Add(crate::layers::AddLayer),
        vec!["l2".into(), "l1".into()],
    ));
    g.set_output("add");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[4]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[4]), 1.0_f32),
    )
    .expect("input box");
    assert!(g.build_point_vjp_batch_plan(&input).is_none());
}

/// INC4 parity leg: masks captured by the template forward must match the graph
/// forward's pre-activation signs at every ReLU, and the template output must
/// match the graph point forward.
#[test]
fn forward_masks_match_graph_forward_signs_and_outputs() {
    let (g, input) = conv_chain_fixture();
    let plan = g
        .build_point_vjp_batch_plan(&input)
        .expect("conv chain plan");

    let mut rng = Rng(0xD1CE_F00D);
    let points: Vec<Vec<f32>> = (0..4)
        .map(|_| (0..plan.input_dim).map(|_| rng.next_f32()).collect())
        .collect();
    let (masks, outputs) = point_vjp_forward_masks(&plan, &points).expect("batched forward");

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
        // Mask parity per ReLU: the ReLU's INPUT node signs.
        for (r, relu_name) in plan.relu_nodes_backward.iter().enumerate() {
            let pre_name = g
                .node(relu_name)
                .expect("relu node")
                .require_unary_input()
                .expect("unary");
            let pre = node_bounds.get(pre_name).expect("pre node").center();
            assert_eq!(masks[k][r].len(), pre.len());
            for (i, (&m, &v)) in masks[k][r].iter().zip(pre.iter()).enumerate() {
                let expected = if v > 0.0 { 1.0 } else { 0.0 };
                // Skip razor-edge pre-activations (|v| tiny): the template
                // forward and the certified graph forward may round across 0.
                if v.abs() < 1e-5 {
                    continue;
                }
                assert_eq!(
                    m, expected,
                    "mask mismatch restart {k} relu {relu_name} neuron {i} (pre={v})"
                );
            }
        }
    }
}
