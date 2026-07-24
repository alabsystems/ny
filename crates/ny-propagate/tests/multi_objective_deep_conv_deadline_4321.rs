// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression test for #4321: the multi-objective graph verifier must reliably
//! self-terminate at its own deadline on deep conv models.
//!
//! Symptom (TinyImageNet ResNet): the root/output-bound phase is dominated by
//! repeated graph IBP forward passes over a deep conv DAG. Those passes did not
//! check the wall-clock deadline, so one long uninterrupted phase overran the
//! `--timeout`, and the verifier was killed externally with no JSON verdict.
//!
//! This builds a synthetic deep+wide conv graph whose unbounded root IBP pass
//! takes well over 2s on CPU, runs multi-objective verification with a SHORT
//! (1s) timeout, and asserts it returns a Timeout/Unknown verdict PROMPTLY
//! (well under a generous wall limit) rather than running to completion.

use std::time::{Duration, Instant};

use ndarray::{Array1, ArrayD, IxDyn};
use ny_propagate::layers::{Conv2dLayer, ReLULayer, ReduceSumLayer};
use ny_propagate::{
    BabVerificationStatus, BetaCrownConfig, BetaCrownVerifier, GraphNetwork, GraphNode, Layer,
};
use ny_tensor::BoundedTensor;

/// Build a deep, spatial-preserving conv stack. Each block is a 3x3
/// `channels`->`channels` Conv2d (padding 1, stride 1) followed by ReLU, so the
/// spatial size stays `hw`x`hw` through all `depth` blocks. The graph ends with
/// a `channels`->`out_logits` 1x1 Conv2d and a ReduceSum over the spatial axes,
/// giving an output of shape `[out_logits]` for multi-objective specs.
fn build_deep_conv_graph(
    channels: usize,
    hw: usize,
    depth: usize,
    out_logits: usize,
) -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    // 3x3 same-padding conv kernel [out=channels, in=channels, 3, 3].
    let make_conv_kernel = |out_ch: usize, in_ch: usize, k: usize| -> ArrayD<f32> {
        let numel = out_ch * in_ch * k * k;
        // Small alternating weights keep bounds finite while still exercising the
        // full conv IBP cost (the runtime is dominated by the GEMM size, not the
        // values).
        let data: Vec<f32> = (0..numel)
            .map(|i| if i % 2 == 0 { 0.03 } else { -0.025 })
            .collect();
        ArrayD::from_shape_vec(IxDyn(&[out_ch, in_ch, k, k]), data).expect("valid conv kernel")
    };

    let mut prev = String::from(ny_propagate::NETWORK_INPUT);
    for layer_idx in 0..depth {
        let conv = Conv2dLayer::with_input_shape(
            make_conv_kernel(channels, channels, 3),
            Some(Array1::zeros(channels)),
            (1, 1),
            (1, 1),
            hw,
            hw,
        )
        .expect("valid same-padding conv");
        let conv_name = format!("conv_{layer_idx}");
        let relu_name = format!("relu_{layer_idx}");
        graph.add_node(GraphNode::new(
            &conv_name,
            Layer::Conv2d(conv),
            vec![prev.clone()],
        ));
        graph.add_node(GraphNode::new(
            &relu_name,
            Layer::ReLU(ReLULayer),
            vec![conv_name.clone()],
        ));
        prev = relu_name;
    }

    // 1x1 conv to reduce to `out_logits` channels.
    let head = Conv2dLayer::with_input_shape(
        make_conv_kernel(out_logits, channels, 1),
        Some(Array1::zeros(out_logits)),
        (1, 1),
        (0, 0),
        hw,
        hw,
    )
    .expect("valid 1x1 head conv");
    graph.add_node(GraphNode::new("head", Layer::Conv2d(head), vec![prev]));

    // ReduceSum over the spatial axes (1=H, 2=W) of the [C, H, W] tensor, leaving
    // a `[out_logits]` output for the multi-objective spec.
    graph.add_node(GraphNode::new(
        "out",
        Layer::ReduceSum(ReduceSumLayer::new(vec![1, 2], false)),
        vec!["head".to_string()],
    ));
    graph.set_output("out");

    let numel = channels * hw * hw;
    let lower = ArrayD::from_shape_vec(IxDyn(&[channels, hw, hw]), vec![-0.5_f32; numel])
        .expect("valid lower input");
    let upper = ArrayD::from_shape_vec(IxDyn(&[channels, hw, hw]), vec![0.5_f32; numel])
        .expect("valid upper input");
    let input = BoundedTensor::new(lower, upper).expect("valid input bounds");

    (graph, input)
}

#[ntest::timeout(30000)]
#[test]
fn test_multi_objective_deep_conv_self_terminates_at_deadline_4321() {
    // Deep + wide enough that a single unbounded root IBP forward pass costs well
    // over 2s on CPU: 32 channels at 64x64 through 24 conv+relu blocks.
    let channels = 32;
    let hw = 64;
    let depth = 24;
    let out_logits = 4;
    let (graph, input) = build_deep_conv_graph(channels, hw, depth, out_logits);

    // Sanity: this MUST be classified as a large conv graph (input > 5000 elems)
    // so it exercises the IBP-bootstrap path that previously ran without a deadline.
    assert!(
        input.len() > 5000,
        "test graph must be a large conv graph (input {} elems) to exercise the IBP bootstrap path",
        input.len()
    );

    // Four objectives over the 4 output logits → spec_matrix path (multi-objective).
    let objectives = vec![
        vec![1.0_f32, -1.0, 0.0, 0.0],
        vec![1.0_f32, 0.0, -1.0, 0.0],
        vec![1.0_f32, 0.0, 0.0, -1.0],
        vec![0.0_f32, 1.0, -1.0, 0.0],
    ];
    let thresholds = vec![0.0_f32; objectives.len()];

    let timeout = Duration::from_secs(1);
    let config = BetaCrownConfig {
        timeout,
        max_domains: 100_000,
        max_depth: 64,
        batch_size: 1,
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);
    let start = Instant::now();
    let result = verifier
        .verify_graph_relu_split_multi_objective(&graph, &input, &objectives, &thresholds)
        .expect("verification must return a verdict (not error out) on deadline");
    let elapsed = start.elapsed();

    // PROMPT termination: the verifier's own deadline must fire. Without the fix
    // the unbounded root IBP forward alone takes >2s; we allow a generous margin
    // (timeout + 2s) for the pass to notice the deadline between nodes.
    assert!(
        elapsed < timeout + Duration::from_secs(2),
        "verifier did not self-terminate promptly: elapsed {:?} for a {:?} timeout (deadline not threaded into root IBP passes?)",
        elapsed,
        timeout
    );

    // Soundness: a deadline abort is always Timeout/Unknown, never Verified.
    assert!(
        matches!(
            result.result,
            BabVerificationStatus::Timeout | BabVerificationStatus::Unknown { .. }
        ),
        "expected Timeout/Unknown on deadline abort, got {:?}",
        result.result
    );
}
