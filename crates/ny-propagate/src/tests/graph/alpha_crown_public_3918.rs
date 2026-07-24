// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Public GraphNetwork alpha-CROWN regressions for #3918.

use crate::bounds::AlphaCrownConfig;
use crate::tests::crown::helpers::{assert_bounds_finite, CountingGemmEngine};
use crate::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

fn total_width(bounds: &BoundedTensor) -> f32 {
    (bounds.upper() - bounds.lower()).iter().sum::<f32>()
}

fn run_engine_width(
    label: &str,
    run: impl FnOnce(&CountingGemmEngine) -> BoundedTensor,
) -> (CountingGemmEngine, f32) {
    let engine = CountingGemmEngine::new();
    let bounds = run(&engine);
    assert_bounds_finite(&bounds, label);
    (engine, total_width(&bounds))
}

fn build_sigmoid_tanh_dag() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    let hidden_w = arr2(&[[1.6_f32, -0.9], [-1.1, 1.4], [0.7, 1.2]]);
    let hidden_b = arr1(&[0.15_f32, -0.2, 0.05]);
    graph.add_node(GraphNode::from_input(
        "linear_hidden",
        Layer::Linear(LinearLayer::new(hidden_w, Some(hidden_b)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "sigmoid_hidden",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["linear_hidden".to_string()],
    ));

    let skip_w = arr2(&[[0.4_f32, -0.3], [0.2, 0.5], [-0.6, 0.1]]);
    let skip_b = arr1(&[0.0_f32, 0.1, -0.05]);
    graph.add_node(GraphNode::from_input(
        "linear_skip",
        Layer::Linear(LinearLayer::new(skip_w, Some(skip_b)).unwrap()),
    ));

    graph.add_node(GraphNode::new(
        "merge",
        Layer::Add(AddLayer),
        vec!["sigmoid_hidden".to_string(), "linear_skip".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "tanh_hidden",
        Layer::Tanh(TanhLayer::new()),
        vec!["merge".to_string()],
    ));

    let out_w = arr2(&[[1.3_f32, -0.8, 0.6], [-0.7, 1.1, -0.9]]);
    let out_b = arr1(&[0.0_f32, 0.05]);
    graph.add_node(GraphNode::new(
        "linear_out",
        Layer::Linear(LinearLayer::new(out_w, Some(out_b)).unwrap()),
        vec!["tanh_hidden".to_string()],
    ));
    graph.set_output("linear_out");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    (graph, input)
}

fn sigmoid_tanh_dag_forward(x: &[f32; 2]) -> [f32; 2] {
    let hidden_w = arr2(&[[1.6_f32, -0.9], [-1.1, 1.4], [0.7, 1.2]]);
    let hidden_b = arr1(&[0.15_f32, -0.2, 0.05]);
    let skip_w = arr2(&[[0.4_f32, -0.3], [0.2, 0.5], [-0.6, 0.1]]);
    let skip_b = arr1(&[0.0_f32, 0.1, -0.05]);
    let out_w = arr2(&[[1.3_f32, -0.8, 0.6], [-0.7, 1.1, -0.9]]);
    let out_b = arr1(&[0.0_f32, 0.05]);

    let x_arr = arr1(&[x[0], x[1]]);
    let hidden_logits = hidden_w.dot(&x_arr) + &hidden_b;
    let hidden = hidden_logits.mapv(|v| 1.0 / (1.0 + (-v).exp()));
    let skip = skip_w.dot(&x_arr) + &skip_b;
    let merged = &hidden + &skip;
    let tanh_hidden = merged.mapv(f32::tanh);
    let out = out_w.dot(&tanh_hidden) + &out_b;
    [out[0], out[1]]
}

fn assert_sigmoid_tanh_dag_bounds_sound(bounds: &BoundedTensor) {
    for i in 0..=10 {
        for j in 0..=10 {
            let x0 = -1.0 + 2.0 * (i as f32) / 10.0;
            let x1 = -1.0 + 2.0 * (j as f32) / 10.0;
            let out = sigmoid_tanh_dag_forward(&[x0, x1]);
            for (dim, &out_val) in out.iter().enumerate() {
                assert!(
                    out_val >= bounds.lower()[[dim]] - 1e-5,
                    "Sigmoid/Tanh DAG soundness: output[{dim}]={out_val} < lower {} at ({x0}, {x1})",
                    bounds.lower()[[dim]],
                );
                assert!(
                    out_val <= bounds.upper()[[dim]] + 1e-5,
                    "Sigmoid/Tanh DAG soundness: output[{dim}]={out_val} > upper {} at ({x0}, {x1})",
                    bounds.upper()[[dim]],
                );
            }
        }
    }
}

fn build_deep_relu_chain_graph(num_relu_layers: usize) -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    let mut prev_node: Option<String> = None;

    for layer_idx in 0..num_relu_layers {
        let linear_name = format!("linear{}", layer_idx + 1);
        let relu_name = format!("relu{}", layer_idx + 1);
        let (weights, bias) = if layer_idx % 2 == 0 {
            (
                arr2(&[
                    [0.42_f32, -0.31, 0.18, -0.27],
                    [-0.24, 0.37, -0.29, 0.16],
                    [0.15, -0.22, 0.34, -0.11],
                    [-0.19, 0.12, -0.26, 0.41],
                ]),
                arr1(&[0.05_f32, -0.04, 0.03, -0.02]),
            )
        } else {
            (
                arr2(&[
                    [0.33_f32, -0.18, 0.24, -0.15],
                    [-0.21, 0.29, -0.17, 0.26],
                    [0.27, -0.25, 0.14, -0.19],
                    [-0.16, 0.23, -0.31, 0.28],
                ]),
                arr1(&[-0.03_f32, 0.04, -0.05, 0.02]),
            )
        };
        let linear = Layer::Linear(LinearLayer::new(weights, Some(bias)).unwrap());

        if let Some(prev) = &prev_node {
            graph.add_node(GraphNode::new(
                linear_name.clone(),
                linear,
                vec![prev.clone()],
            ));
        } else {
            graph.add_node(GraphNode::from_input(linear_name.clone(), linear));
        }
        graph.add_node(GraphNode::new(
            relu_name.clone(),
            Layer::ReLU(ReLULayer::new()),
            vec![linear_name],
        ));
        prev_node = Some(relu_name);
    }

    let output = Layer::Linear(
        LinearLayer::new(
            arr2(&[[0.48_f32, -0.37, 0.21, -0.14], [-0.28, 0.44, -0.19, 0.32]]),
            Some(arr1(&[0.01_f32, -0.02])),
        )
        .unwrap(),
    );
    graph.add_node(GraphNode::new(
        "linear_out",
        output,
        vec![prev_node.expect("deep ReLU chain should create at least one ReLU node")],
    ));
    graph.set_output("linear_out");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[4]), -0.5_f32),
        ArrayD::from_elem(IxDyn(&[4]), 0.5_f32),
    )
    .unwrap();
    (graph, input)
}

/// Add one residual block: linear→relu on the main path, linear (skip) on the
/// bypass, add→relu to merge. Returns the name of the merge-relu output node.
fn add_residual_block(
    graph: &mut GraphNetwork,
    block_idx: usize,
    current: &str,
    w_main: &ndarray::Array2<f32>,
    w_skip: &ndarray::Array2<f32>,
) -> String {
    let linear_name = format!("block{block_idx}_linear");
    let relu_name = format!("block{block_idx}_relu");
    let skip_name = format!("block{block_idx}_skip");
    let add_name = format!("block{block_idx}_add");
    let merge_relu_name = format!("block{block_idx}_merge_relu");

    graph.add_node(GraphNode::new(
        linear_name.clone(),
        Layer::Linear(LinearLayer::new(w_main.clone(), None).unwrap()),
        vec![current.to_string()],
    ));
    graph.add_node(GraphNode::new(
        relu_name.clone(),
        Layer::ReLU(ReLULayer::new()),
        vec![linear_name],
    ));
    graph.add_node(GraphNode::new(
        skip_name.clone(),
        Layer::Linear(LinearLayer::new(w_skip.clone(), None).unwrap()),
        vec![current.to_string()],
    ));
    graph.add_node(GraphNode::new(
        add_name.clone(),
        Layer::Add(AddLayer),
        vec![relu_name, skip_name],
    ));
    graph.add_node(GraphNode::new(
        merge_relu_name.clone(),
        Layer::ReLU(ReLULayer::new()),
        vec![add_name],
    ));
    merge_relu_name
}

fn build_deep_relu_skip_dag(num_relu_layers: usize) -> (GraphNetwork, BoundedTensor) {
    assert!(
        num_relu_layers >= 22,
        "deep residual DAG fixture expects at least 22 ReLU layers to exceed the legacy threshold"
    );

    let mut graph = GraphNetwork::new();
    let w0 = arr2(&[
        [0.6_f32, -0.3, 0.4],
        [-0.2, 0.5, 0.3],
        [0.1, -0.4, 0.7],
        [0.3, 0.2, -0.5],
    ]);
    let b0 = arr1(&[0.1_f32, -0.05, 0.05, -0.1]);
    let w_main = arr2(&[
        [0.4_f32, -0.2, 0.3, -0.1],
        [-0.1, 0.5, -0.2, 0.3],
        [0.2, -0.3, 0.6, -0.2],
        [-0.3, 0.1, -0.1, 0.4],
    ]);
    let w_skip = arr2(&[
        [0.9_f32, 0.05, -0.05, 0.0],
        [0.0, 0.85, 0.1, -0.05],
        [-0.05, 0.0, 0.9, 0.05],
        [0.05, -0.05, 0.0, 0.85],
    ]);
    let w_tail = arr2(&[
        [0.5_f32, -0.2, 0.1, 0.3],
        [-0.1, 0.4, -0.3, 0.2],
        [0.3, -0.1, 0.5, -0.2],
        [-0.2, 0.3, -0.1, 0.4],
    ]);
    let w_out = arr2(&[[0.4_f32, -0.3, 0.2, 0.1], [-0.2, 0.5, -0.1, 0.3]]);

    graph.add_node(GraphNode::from_input(
        "linear0",
        Layer::Linear(LinearLayer::new(w0, Some(b0)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu0",
        Layer::ReLU(ReLULayer::new()),
        vec!["linear0".to_string()],
    ));

    let residual_blocks = (num_relu_layers - 2) / 2;
    let mut current = "relu0".to_string();
    for block_idx in 0..residual_blocks {
        current = add_residual_block(&mut graph, block_idx, &current, &w_main, &w_skip);
    }

    graph.add_node(GraphNode::new(
        "tail_linear",
        Layer::Linear(LinearLayer::new(w_tail, None).unwrap()),
        vec![current],
    ));
    graph.add_node(GraphNode::new(
        "tail_relu",
        Layer::ReLU(ReLULayer::new()),
        vec!["tail_linear".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear_out",
        Layer::Linear(LinearLayer::new(w_out, None).unwrap()),
        vec!["tail_relu".to_string()],
    ));
    graph.set_output("linear_out");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[3]), -0.5_f32),
        ArrayD::from_elem(IxDyn(&[3]), 0.5_f32),
    )
    .unwrap();
    (graph, input)
}

#[ntest::timeout(10000)]
#[test]
fn test_public_graph_crown_entrypoint_beats_fixed_crown_on_sigmoid_tanh_dag_3918() {
    let (graph, input) = build_sigmoid_tanh_dag();

    let fixed_bounds = graph.propagate_crown_fixed_slope(&input).unwrap();
    let public_bounds = graph.propagate_crown(&input).unwrap();

    assert_sigmoid_tanh_dag_bounds_sound(&public_bounds);

    let fixed_width = total_width(&fixed_bounds);
    let public_width = total_width(&public_bounds);
    assert!(
        public_width + 1e-4 < fixed_width,
        "Public GraphNetwork::propagate_crown should preserve alpha tightening for downstream callers. fixed={fixed_width:.6}, public={public_width:.6}"
    );
}

#[ntest::timeout(60000)]
#[test]
fn test_public_graph_crown_keeps_alpha_live_past_legacy_skip_threshold_3918() {
    let (graph, input) = build_deep_relu_chain_graph(21);

    let (fixed_engine, fixed_width) = run_engine_width(
        "#3918 fixed-slope graph CROWN with engine output",
        |engine| {
            graph
                .propagate_crown_fixed_slope_with_engine(&input, Some(engine))
                .expect("fixed-slope CROWN should succeed on deep ReLU chain")
        },
    );

    let (public_engine, public_width) =
        run_engine_width("#3918 public graph CROWN with engine output", |engine| {
            graph
                .propagate_crown_with_engine(&input, Some(engine))
                .expect("public CROWN should keep alpha-CROWN live on deep ReLU chain")
        });

    let (_legacy_skip_engine, legacy_skip_width) = run_engine_width(
        "#3918 legacy adaptive-skip graph CROWN with engine output",
        |engine| {
            graph
                .propagate_alpha_crown_with_config_and_engine(
                    &input,
                    &AlphaCrownConfig {
                        adaptive_skip: true,
                        adaptive_skip_depth_threshold: 20,
                        ..AlphaCrownConfig::default()
                    },
                    Some(engine),
                )
                .expect("legacy adaptive-skip config should still execute")
        },
    );

    assert!(
        (legacy_skip_width - fixed_width).abs() <= 1e-5,
        "Legacy adaptive_skip should reproduce fixed-slope CROWN on a 21-ReLU graph; skip={legacy_skip_width:.6}, fixed={fixed_width:.6}"
    );
    assert!(
        public_width <= fixed_width + 1e-5,
        "Public CROWN widened bounds relative to fixed-slope baseline on deep ReLU chain; public={public_width:.6}, fixed={fixed_width:.6}"
    );
    assert!(
        public_engine.gemm_calls() > fixed_engine.gemm_calls(),
        "Public CROWN should do more GEMM-backed work than fixed-slope CROWN once adaptive_skip is disabled by default; public_calls={}, fixed_calls={}",
        public_engine.gemm_calls(),
        fixed_engine.gemm_calls(),
    );
    // The legacy skip path can have MORE GEMM calls than the public path because
    // CROWN-IBP init runs before adaptive_skip fires, followed by a redundant full
    // CROWN backward fallback. The meaningful check is bounds quality: with alpha-CROWN
    // enabled, public bounds should be at least as tight as the skip path's bounds.
    assert!(
        public_width <= legacy_skip_width + 1e-5,
        "Public CROWN (alpha-CROWN enabled) should produce bounds at least as tight as \
         legacy adaptive_skip fallback; public_width={public_width:.6}, skip_width={legacy_skip_width:.6}",
    );
}

#[ntest::timeout(60000)]
#[test]
fn test_public_graph_crown_keeps_alpha_live_past_legacy_skip_threshold_on_dag_3918() {
    let (graph, input) = build_deep_relu_skip_dag(22);

    let (fixed_engine, fixed_width) =
        run_engine_width("#3918 fixed-slope DAG CROWN with engine output", |engine| {
            graph
                .propagate_crown_fixed_slope_with_engine(&input, Some(engine))
                .expect("fixed-slope CROWN should succeed on deep ReLU DAG")
        });

    let (public_engine, public_width) =
        run_engine_width("#3918 public DAG CROWN with engine output", |engine| {
            graph
                .propagate_crown_with_engine(&input, Some(engine))
                .expect("public CROWN should keep alpha-CROWN live on deep ReLU DAG")
        });

    let (legacy_skip_engine, legacy_skip_width) = run_engine_width(
        "#3918 legacy adaptive-skip DAG CROWN with engine output",
        |engine| {
            graph
                .propagate_alpha_crown_with_config_and_engine(
                    &input,
                    &AlphaCrownConfig {
                        adaptive_skip: true,
                        adaptive_skip_depth_threshold: 20,
                        ..AlphaCrownConfig::default()
                    },
                    Some(engine),
                )
                .expect("legacy adaptive-skip config should still execute on DAG")
        },
    );

    assert!(
        (legacy_skip_width - fixed_width).abs() <= 1e-5,
        "Legacy adaptive_skip should reproduce fixed-slope CROWN on a >20-ReLU DAG; skip={legacy_skip_width:.6}, fixed={fixed_width:.6}"
    );
    assert!(
        public_width <= fixed_width + 1e-5,
        "Public CROWN widened bounds relative to fixed-slope baseline on deep ReLU DAG; public={public_width:.6}, fixed={fixed_width:.6}"
    );
    assert!(
        public_engine.gemm_calls() > fixed_engine.gemm_calls(),
        "Public CROWN should do more GEMM-backed work than fixed-slope CROWN once adaptive_skip is disabled on DAGs; public_calls={}, fixed_calls={}",
        public_engine.gemm_calls(),
        fixed_engine.gemm_calls(),
    );
    assert!(
        public_width <= legacy_skip_width + 1e-5,
        "Public CROWN should be no worse than the legacy adaptive_skip fallback on a >20-ReLU DAG; public_width={public_width:.6}, skip_width={legacy_skip_width:.6}",
    );
    assert!(
        legacy_skip_engine.gemm_calls() >= fixed_engine.gemm_calls(),
        "Legacy adaptive-skip path should still incur at least the fixed-slope DAG work before skipping; skip_calls={}, fixed_calls={}",
        legacy_skip_engine.gemm_calls(),
        fixed_engine.gemm_calls(),
    );
}
