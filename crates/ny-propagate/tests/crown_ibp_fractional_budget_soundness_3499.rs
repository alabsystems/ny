// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{arr1, arr2, array, ArrayD, IxDyn};
use ny_propagate::layers::{Conv2dLayer, LinearLayer, ReLULayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer, Network};
use ny_tensor::BoundedTensor;
use ny_test_utils::assert_bounded_tensor_close;
use std::time::{Duration, Instant};

fn build_fractional_budget_parity_graph_3499() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0_f32, -0.5], [0.4, 0.8], [-0.3, 0.6]]),
                Some(arr1(&[0.1_f32, -0.2, 0.05])),
            )
            .expect("lin1 should be valid"),
        ),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["lin1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin2",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[0.5_f32, -0.3, 0.7], [-0.2, 0.9, 0.1], [0.6, -0.4, 0.2]]),
                Some(arr1(&[0.0_f32, 0.15, -0.1])),
            )
            .expect("lin2 should be valid"),
        ),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["lin2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(arr2(&[[0.8_f32, -0.1, 0.2]]), Some(arr1(&[0.05_f32])))
                .expect("out should be valid"),
        ),
        vec!["relu2".to_string()],
    ));
    graph.set_output("out");
    graph
}

fn eval_fractional_budget_parity_graph_3499(x0: f32, x1: f32) -> f32 {
    let z1_0 = x0 - (0.5 * x1) + 0.1;
    let z1_1 = (0.4 * x0) + (0.8 * x1) - 0.2;
    let z1_2 = (-0.3 * x0) + (0.6 * x1) + 0.05;
    let r1_0 = z1_0.max(0.0);
    let r1_1 = z1_1.max(0.0);
    let r1_2 = z1_2.max(0.0);

    let z2_0 = (0.5 * r1_0) - (0.3 * r1_1) + (0.7 * r1_2);
    let z2_1 = (-0.2 * r1_0) + (0.9 * r1_1) + (0.1 * r1_2) + 0.15;
    let z2_2 = (0.6 * r1_0) - (0.4 * r1_1) + (0.2 * r1_2) - 0.1;
    let r2_0 = z2_0.max(0.0);
    let r2_1 = z2_1.max(0.0);
    let r2_2 = z2_2.max(0.0);

    (0.8 * r2_0) - (0.1 * r2_1) + (0.2 * r2_2) + 0.05
}

fn assert_budgeted_matches_unbounded_and_ibp(
    ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
    unbounded: &ny_propagate::types::GraphCrownIbpBoundsResult,
    budgeted: &ny_propagate::types::GraphCrownIbpBoundsResult,
) {
    for (name, ibp_bound) in ibp_bounds {
        let budgeted_bound = budgeted
            .bounds
            .get(name)
            .unwrap_or_else(|| panic!("budgeted bounds missing node '{name}'"));
        let unbounded_bound = unbounded
            .bounds
            .get(name)
            .unwrap_or_else(|| panic!("unbounded bounds missing node '{name}'"));

        assert_bounded_tensor_close(
            budgeted_bound,
            unbounded_bound,
            1e-6,
            &format!("#3499 node '{name}' bounded/unbounded parity"),
        );

        for (idx, (&budgeted_l, &ibp_l)) in budgeted_bound
            .lower()
            .iter()
            .zip(ibp_bound.lower().iter())
            .enumerate()
        {
            assert!(
                budgeted_l >= ibp_l - 1e-6,
                "#3499 node '{name}' lower[{idx}] loosened under fractional deadline: budgeted={budgeted_l}, ibp={ibp_l}"
            );
        }
        for (idx, (&budgeted_u, &ibp_u)) in budgeted_bound
            .upper()
            .iter()
            .zip(ibp_bound.upper().iter())
            .enumerate()
        {
            assert!(
                budgeted_u <= ibp_u + 1e-6,
                "#3499 node '{name}' upper[{idx}] loosened under fractional deadline: budgeted={budgeted_u}, ibp={ibp_u}"
            );
        }
    }
}

fn assert_fractional_budget_output_sound(input: &BoundedTensor, output: &BoundedTensor) {
    let lower = input.lower();
    let upper = input.upper();
    for &x0 in &[
        lower[[0]],
        f32::midpoint(lower[[0]], upper[[0]]),
        upper[[0]],
    ] {
        for &x1 in &[
            lower[[1]],
            f32::midpoint(lower[[1]], upper[[1]]),
            upper[[1]],
        ] {
            let y = eval_fractional_budget_parity_graph_3499(x0, x1);
            assert!(
                y >= output.lower()[[0]] - 1e-5 && y <= output.upper()[[0]] + 1e-5,
                "#3499 fractional deadline output {} not in [{}, {}] at ({}, {})",
                y,
                output.lower()[[0]],
                output.upper()[[0]],
                x0,
                x1
            );
        }
    }
}

fn exact_grouped_conv_chain_bounds_3777(input: &BoundedTensor) -> BoundedTensor {
    let weights = [
        [-0.25_f32, -0.75, 2.25, -0.375],
        [-0.6875, 1.53125, -3.5, 1.625],
    ];
    let mut lower = ArrayD::<f32>::zeros(IxDyn(&[2, 3, 3]));
    let mut upper = ArrayD::<f32>::zeros(IxDyn(&[2, 3, 3]));

    for oc in 0..2 {
        for oh in 0..3 {
            for ow in 0..3 {
                let mut lower_sum = 0.0_f32;
                let mut upper_sum = 0.0_f32;

                for (ic, weight) in weights[oc].iter().copied().enumerate() {
                    let input_lower = input.lower()[[ic, oh, ow]];
                    let input_upper = input.upper()[[ic, oh, ow]];
                    if weight >= 0.0 {
                        lower_sum += weight * input_lower;
                        upper_sum += weight * input_upper;
                    } else {
                        lower_sum += weight * input_upper;
                        upper_sum += weight * input_lower;
                    }
                }

                lower[[oc, oh, ow]] = lower_sum;
                upper[[oc, oh, ow]] = upper_sum;
            }
        }
    }

    BoundedTensor::new(lower, upper).expect("exact grouped-conv chain bounds should be valid")
}

#[test]
fn test_crown_ibp_fractional_deadline_matches_unbounded_bounds_3499() {
    let graph = build_fractional_budget_parity_graph_3499();
    let input = BoundedTensor::new(
        array![-0.5_f32, -0.25].into_dyn(),
        array![0.75_f32, 0.6].into_dyn(),
    )
    .expect("bounded input should be valid");
    let ibp_bounds = graph
        .collect_node_bounds(&input)
        .expect("IBP bounds should succeed");

    let unbounded = graph
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp(&input, ibp_bounds.clone(), None)
        .expect("unbounded CROWN-IBP should succeed");
    let budgeted = graph
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp(
            &input,
            ibp_bounds.clone(),
            Some(Instant::now() + Duration::from_secs(20)),
        )
        .expect("fraction-of-remaining deadline should match unbounded results on a small graph");

    assert_budgeted_matches_unbounded_and_ibp(&ibp_bounds, &unbounded, &budgeted);
    let output = budgeted
        .bounds
        .get("out")
        .expect("budgeted output bounds should exist");
    assert_fractional_budget_output_sound(&input, output);
}

#[test]
fn test_collect_crown_ibp_bounds_grouped_conv_chain_matches_exact_chain_3777() {
    let grouped_kernel = ArrayD::from_shape_vec(
        IxDyn(&[4, 2, 1, 1]),
        vec![
            1.0_f32, -2.0, //
            0.5, 0.25, //
            -1.5, 0.75, //
            2.0, -0.5, //
        ],
    )
    .expect("grouped kernel should be valid");
    let grouped_conv =
        Conv2dLayer::with_input_shape_full(grouped_kernel, None, (1, 1), (0, 0), 2, 3, 3)
            .expect("grouped conv should be valid");

    let downstream_kernel = ArrayD::from_shape_vec(
        IxDyn(&[2, 4, 1, 1]),
        vec![
            0.25_f32, -1.0, 0.5, 1.5, //
            -0.75, 0.125, 2.0, -0.25, //
        ],
    )
    .expect("downstream kernel should be valid");
    let downstream_conv =
        Conv2dLayer::with_input_shape(downstream_kernel, None, (1, 1), (0, 0), 3, 3)
            .expect("downstream conv should be valid");

    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(grouped_conv));
    network.add_layer(Layer::Conv2d(downstream_conv));

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[4, 3, 3]),
            (0..36).map(|idx| -1.0_f32 + idx as f32 * 0.05).collect(),
        )
        .expect("lower input tensor should be valid"),
        ArrayD::from_shape_vec(
            IxDyn(&[4, 3, 3]),
            (0..36).map(|idx| -0.6_f32 + idx as f32 * 0.05).collect(),
        )
        .expect("upper input tensor should be valid"),
    )
    .expect("bounded grouped-conv input should be valid");

    let ibp_bounds = network
        .collect_ibp_bounds(&input)
        .expect("grouped-conv IBP should succeed");
    let crown_ibp_bounds = network
        .collect_crown_ibp_bounds(&input)
        .expect("grouped-conv CROWN-IBP should succeed");
    let exact_output = exact_grouped_conv_chain_bounds_3777(&input);

    assert_eq!(ibp_bounds.len(), 2);
    assert_eq!(crown_ibp_bounds.len(), 2);
    assert_bounded_tensor_close(
        &crown_ibp_bounds[0],
        &ibp_bounds[0],
        1e-6,
        "#3777 grouped Conv2d first-layer bound",
    );
    // The dense Conv2d CROWN backward carries a per-row coefficient-error
    // envelope (the f64-recomputed coefficient error, ~1 ULP of the composed
    // coefficient) which concretize discharges OUTWARD against sum_j max|x_j|
    // over all 36 inputs — measured 8e-6..1.1e-5 on this chain (largest on
    // channel 2, whose composed coefficient magnitude peaks at 3.5). The
    // envelope only ever widens, so pin the direction (no element may cut
    // inside the exact chain bound beyond fp noise) and then bound the
    // outward slack at 2e-5.
    for (idx, (&actual_l, &exact_l)) in crown_ibp_bounds[1]
        .lower()
        .iter()
        .zip(exact_output.lower().iter())
        .enumerate()
    {
        assert!(
            actual_l <= exact_l + 1e-6,
            "#3777 grouped Conv2d chain lower[{idx}] cut inside the exact bound: actual={actual_l} exact={exact_l}"
        );
    }
    for (idx, (&actual_u, &exact_u)) in crown_ibp_bounds[1]
        .upper()
        .iter()
        .zip(exact_output.upper().iter())
        .enumerate()
    {
        assert!(
            actual_u >= exact_u - 1e-6,
            "#3777 grouped Conv2d chain upper[{idx}] cut inside the exact bound: actual={actual_u} exact={exact_u}"
        );
    }
    assert_bounded_tensor_close(
        &crown_ibp_bounds[1],
        &exact_output,
        2e-5,
        "#3777 grouped Conv2d chain output bound",
    );
}
