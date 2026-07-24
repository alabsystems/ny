// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::batching::RestartBatchLayout;
use super::{
    batched_spsa_objective, evaluate_graph_batch, try_graph_pgd_upfront_batched, BatchForward,
    BatchedGraphPgdOutcome, SpsaTarget, CHUNK_TARGET_ELEMS,
};
use crate::commands::beta_crown::verify::graph_pgd::GraphPgdTarget;
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_onnx::vnnlib::parse_vnnlib;
use ny_propagate::{
    layers::{AddConstantLayer, Conv2dLayer, LinearLayer},
    GraphNetwork, Layer, Network,
};
use ny_tensor::BoundedTensor;
use ny_test_utils::CountingGemmEngine;

fn make_positive_conv2d_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(
        Conv2dLayer::with_input_shape(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0_f32]).unwrap(),
            Some(arr1(&[2.0_f32])),
            (1, 1),
            (0, 0),
            1,
            1,
        )
        .expect("conv2d kernel should be valid"),
    ));
    GraphNetwork::from_sequential(&network).expect("single conv2d network should convert to graph")
}

fn make_add_constant_broadcast_linear_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::AddConstant(AddConstantLayer::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![0.5_f32, -0.25_f32]).unwrap(),
    )));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32, 1.0_f32]]), Some(arr1(&[0.0_f32]))).unwrap(),
    ));
    GraphNetwork::from_sequential(&network)
        .expect("broadcast add_constant+linear network should convert to graph")
}

fn make_flat_then_activate_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32, 0.0_f32]]), Some(arr1(&[-0.9_f32]))).unwrap(),
    ));
    network.add_layer(Layer::ReLU(Default::default()));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[10.0_f32]]), Some(arr1(&[0.0_f32]))).unwrap(),
    ));
    GraphNetwork::from_sequential(&network)
        .expect("flat-then-activate network should convert to graph")
}

fn make_scalar_upper_bound_spec(
    lower: f32,
    upper: f32,
    threshold: f32,
) -> ny_onnx::vnnlib::VnnLibSpec {
    parse_vnnlib(&format!(
        "\
(declare-const X_0 Real)\n\
(declare-const Y_0 Real)\n\
(assert (>= X_0 {lower}))\n\
(assert (<= X_0 {upper}))\n\
(assert (<= Y_0 {threshold}))\n"
    ))
    .expect("scalar VNN-LIB spec should parse")
}

fn make_multi_input_upper_bound_spec(
    inputs: usize,
    lower: f32,
    upper: f32,
    threshold: f32,
) -> ny_onnx::vnnlib::VnnLibSpec {
    let mut spec = String::new();
    for index in 0..inputs {
        let _ = std::fmt::Write::write_fmt(
            &mut spec,
            format_args!(
                "(declare-const X_{index} Real)\n(declare-const Y_{index} Real)\n(assert (>= X_{index} {lower}))\n(assert (<= X_{index} {upper}))\n(assert (<= Y_{index} {threshold}))\n"
            ),
        );
    }
    parse_vnnlib(&spec).expect("multi-input VNN-LIB spec should parse")
}

/// A 1x1 conv graph whose per-item input is large enough that a multi-restart
/// batch exceeds `CHUNK_TARGET_ELEMS`, forcing the chunked forward path.
fn make_large_input_conv2d_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(
        Conv2dLayer::with_input_shape(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![2.0_f32]).unwrap(),
            Some(arr1(&[0.5_f32])),
            (1, 1),
            (0, 0),
            1,
            1,
        )
        .expect("conv2d kernel should be valid"),
    ));
    GraphNetwork::from_sequential(&network).expect("single conv2d network should convert to graph")
}

#[test]
fn generic_batched_objective_is_bit_identical_raw_when_gama_is_off() {
    let output = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0f32, 0.0]).unwrap();
    let legacy = SpsaTarget::Constant(0, 2.0);
    let conjunct_targets = [GraphPgdTarget::Constant(0, 2.0)];
    let raw = legacy.margin(&output);
    let default_off = batched_spsa_objective(&legacy, &conjunct_targets, &output, None);
    assert_eq!(
        default_off.to_bits(),
        raw.to_bits(),
        "default-off generic batching must preserve the historical raw objective bit-for-bit"
    );

    let guided = batched_spsa_objective(
        &legacy,
        &conjunct_targets,
        &output,
        Some((&[1.0f32, 0.0], 10.0)),
    );
    assert!(
        guided > 0.0 && raw < 0.0,
        "configured guidance must be numerically active without redefining raw success"
    );
}

#[test]
fn chunked_batched_forward_matches_single_dispatch_four_walls() {
    let graph = make_large_input_conv2d_graph();
    let item_shape = [1usize, 256, 256];
    let item_elems: usize = item_shape.iter().product();
    let num_items = 8usize;
    assert!(
        item_elems * num_items > CHUNK_TARGET_ELEMS && item_elems < CHUNK_TARGET_ELEMS,
        "test premise: batch must exceed the chunk target so chunking engages"
    );

    // Deterministic pseudo-random batch on the prepend-axis layout.
    let mut batch_shape = vec![num_items];
    batch_shape.extend_from_slice(&item_shape);
    let mut value = 0.1_f32;
    let batch = ArrayD::from_shape_fn(IxDyn(&batch_shape), |_| {
        value = (value * 1.31 + 0.017) % 1.0;
        value
    });

    let chunked = match evaluate_graph_batch(
        &graph,
        &batch,
        None,
        num_items,
        RestartBatchLayout::PrependAxis,
        None,
    )
    .expect("chunked batched forward should succeed")
    {
        BatchForward::Output(output) => output,
        BatchForward::DeadlineExceeded => panic!("no deadline was set"),
    };

    let single = graph
        .propagate_concrete_point_preserve_leading_axis(
            &BoundedTensor::concrete(batch).unwrap(),
            None,
        )
        .expect("single-dispatch forward should succeed")
        .center();

    assert_eq!(
        chunked.shape(),
        single.shape(),
        "chunked forward must preserve the batched output shape"
    );
    assert!(
        chunked.iter().zip(single.iter()).all(|(a, b)| a == b),
        "chunked forward must be bit-identical to the single dispatch"
    );
}

#[test]
fn chunked_batched_forward_respects_expired_deadline_four_walls() {
    let graph = make_large_input_conv2d_graph();
    let num_items = 8usize;
    let batch = ArrayD::from_elem(IxDyn(&[num_items, 1, 256, 256]), 0.25_f32);
    let expired = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_millis(10))
        .unwrap();

    let outcome = evaluate_graph_batch(
        &graph,
        &batch,
        None,
        num_items,
        RestartBatchLayout::PrependAxis,
        Some(expired),
    )
    .expect("chunked batched forward should not error on an expired deadline");
    assert!(
        matches!(outcome, BatchForward::DeadlineExceeded),
        "an expired deadline must preempt the chunk loop instead of running the full batch"
    );
}

#[test]
fn graph_pgd_batched_threads_gemm_engine_for_conv2d_nodes_4081() {
    let graph = make_positive_conv2d_graph();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1, 1]), 0.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1, 1]), 1.0_f32),
    )
    .unwrap();
    let spec = make_scalar_upper_bound_spec(0.0, 1.0, 0.0);
    let engine = CountingGemmEngine::new();

    let result = try_graph_pgd_upfront_batched(
        &graph,
        &input,
        &spec,
        4,
        3,
        Default::default(),
        20,
        None,
        Some(&engine),
        true,
        false,
    )
    .expect("graph PGD should accept Conv2d restart batches");
    let BatchedGraphPgdOutcome::Completed(result) = result else {
        panic!("Conv2d graph PGD should stay on the batched path");
    };

    assert!(
        result.is_none(),
        "x + 2 stays above zero across [0, 1], so Conv2d graph PGD should not find a violation"
    );
    assert!(
        engine.gemm_calls() > 0,
        "#4081 regression: batched graph PGD should thread GemmEngine through Conv2d IBP nodes"
    );
}

#[test]
fn graph_pgd_batched_returns_fallback_for_shape_mismatch_4093() {
    let graph = make_add_constant_broadcast_linear_graph();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0_f32, 0.0_f32]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0_f32, 1.0_f32]).unwrap(),
    )
    .unwrap();
    let spec = make_multi_input_upper_bound_spec(2, 0.0, 1.0, -1.0);
    let engine = CountingGemmEngine::new();

    let outcome = try_graph_pgd_upfront_batched(
        &graph,
        &input,
        &spec,
        4,
        2,
        Default::default(),
        20,
        None,
        Some(&engine),
        true,
        false,
    )
    .expect("shape-mismatch graphs should report fallback, not error");

    assert!(
        matches!(outcome, BatchedGraphPgdOutcome::FallbackToSequential),
        "#4093 regression: shape-mismatch batched graphs must signal fallback so the caller can run sequential PGD"
    );
    // No gemm_calls assertion: since #cgan-eval the batched attempt is a point
    // forward that hits the broadcast Add's shape mismatch BEFORE the first
    // Linear engine GEMM, so "attempted the batched path" is no longer
    // observable through the engine counter — the FallbackToSequential outcome
    // (vs an Err) is the attempt evidence.
}

#[test]
fn graph_pgd_batched_restart_when_stuck_resamples_live_rng_4278() {
    let graph = make_flat_then_activate_graph();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0_f32, 0.0_f32]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0_f32, 1.0_f32]).unwrap(),
    )
    .unwrap();
    let spec = parse_vnnlib(
        "\
(declare-const X_0 Real)\n\
(declare-const X_1 Real)\n\
(declare-const Y_0 Real)\n\
(assert (>= X_0 0.0))\n\
(assert (<= X_0 1.0))\n\
(assert (>= X_1 0.0))\n\
(assert (<= X_1 1.0))\n\
(assert (>= Y_0 0.05))\n",
    )
    .expect("graph PGD restart_when_stuck spec should parse");
    let engine = CountingGemmEngine::new();

    let without_restart = try_graph_pgd_upfront_batched(
        &graph,
        &input,
        &spec,
        1,
        4,
        Default::default(),
        20,
        None,
        Some(&engine),
        true,
        false,
    )
    .expect("batched graph PGD without restart_when_stuck should not error");
    let with_restart = try_graph_pgd_upfront_batched(
        &graph,
        &input,
        &spec,
        1,
        4,
        Default::default(),
        20,
        None,
        Some(&engine),
        true,
        true,
    )
    .expect("batched graph PGD with restart_when_stuck should not error");

    let BatchedGraphPgdOutcome::Completed(without_restart) = without_restart else {
        panic!("flat graph should stay on the batched graph PGD path");
    };
    let BatchedGraphPgdOutcome::Completed(with_restart) = with_restart else {
        panic!("flat graph should stay on the batched graph PGD path");
    };

    assert!(
        without_restart.is_none(),
        "without restart_when_stuck, batched graph PGD should stay pinned in the flat dead region"
    );
    let witness = with_restart.expect(
        "restart_when_stuck must resample with the live RNG state; recreating the original seed leaves restart 0 stuck forever",
    );
    assert!(
        witness.0[0] > 0.9,
        "the restarted witness should leave the dead ReLU region, got x0={}",
        witness.0[0]
    );
    assert!(
        witness.1[0] >= 0.05,
        "the restarted witness must satisfy the unsafe output constraint, got {}",
        witness.1[0]
    );
}
