// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::{
    fmt::Write as _,
    time::{Duration, Instant},
};

use ndarray::{arr1, arr2, ArrayD, Axis, IxDyn};
use ny_core::{GemmEngine, NaiveCpuGemmEngine};
use ny_onnx::vnnlib::{parse_vnnlib, VnnLibSpec};
use ny_propagate::{
    layers::{
        AbsLayer, AddConstantLayer, AddLayer, BatchNormLayer, CausalSoftmaxLayer, ConcatLayer,
        Conv2dLayer, DivConstantLayer, DivLayer, ExpLayer, FlattenLayer, LeakyReLULayer,
        LinearLayer, LogLayer, LogSoftmaxLayer, MaxPool2dLayer, MulBinaryLayer, MulConstantLayer,
        PadLayer, PadMode, PowConstantLayer, ReLULayer, ReshapeLayer, SigmoidLayer, SliceLayer,
        SoftmaxLayer, SqrtLayer, SubConstantLayer, SubLayer, TanhLayer, TransposeLayer,
    },
    GraphNetwork, GraphNode, Layer, Network, PgdConfig,
};
use ny_tensor::BoundedTensor;
use ny_test_utils::CountingGemmEngine;

use super::graph_pgd::{
    evaluate_graph, graph_supports_exact_gradients, independent_graph_forward,
    try_graph_pgd_upfront, try_graph_pgd_upfront_with_config, GAMA_ENV_LOCK,
};
use super::graph_pgd_batched::GENERIC_BATCHED_GAMA_OBJECTIVE_EVALS;

#[path = "tests_graph_pgd_round_five.rs"]
mod round_five;

// Batched-vs-sequential parity is ULP-scale, not exact: the batched eval is a
// true point forward (#cgan-eval, `propagate_concrete_point_preserve_leading_axis`
// + `.center()`), whose per-node reduction order differs from the per-sample
// sequential forward — softmax's exp/sum chain accumulates a few tens of f32
// ULPs (observed 2.3e-6 at output 1.0).
const PRESERVE_LEADING_AXIS_TOL: f32 = 1e-5;

/// Ceiling on engine GEMM dispatches for one BATCHED graph-PGD run with the
/// 5-step configs used below: 1 initial batched forward + 2 per step (the
/// stacked SPSA eval and the post-step recompute) = 11 since #cgan-eval made
/// every eval a point forward through the engine. The guarded regression —
/// falling back to per-restart dispatch — costs `num_restarts` times this.
const BATCHED_GEMM_DISPATCH_MAX: usize = 1 + 2 * 5;

/// Serializes the tests that pin `NY_PGD_EXACT_BATCHED` (a process-global env
/// var) so they cannot interleave under the parallel runner.
static EXACT_BATCHED_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// RAII guard that forces the batched routing (`NY_PGD_EXACT_BATCHED=0`) for the
/// duration of a test, then clears it. Exact-eligible graphs (single Linear,
/// Reshape/Flatten + Linear) route to the sequential EXACT-gradient lane by
/// default now (#soundnessbench); these `*_batched_*` regressions still need to
/// exercise the BATCHED lane, so they opt back into it via the documented kill
/// switch — which this also covers.
/// Mutation routed through the blessed env choke point (clippy env wall);
/// field order matters: `_var` restores before `_lock` releases.
struct ForceBatchedRouting {
    _var: ny_test_utils::env::ScopedEnvVar,
    _lock: std::sync::MutexGuard<'static, ()>,
}
impl ForceBatchedRouting {
    fn new() -> Self {
        let lock = EXACT_BATCHED_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let var = ny_test_utils::env::ScopedEnvVar::set("NY_PGD_EXACT_BATCHED", "0");
        Self {
            _var: var,
            _lock: lock,
        }
    }
}

/// Keeps config-driven GAMA tests independent of process-global environment
/// overrides and serializes them with the precedence regression. Mutation
/// routed through the blessed env choke point (clippy env wall); field order
/// matters: `_var` restores before `_lock` releases.
struct IsolatedGamaEnv {
    _var: ny_test_utils::env::ScopedEnvVar,
    _lock: std::sync::MutexGuard<'static, ()>,
}
impl IsolatedGamaEnv {
    fn new() -> Self {
        let lock = GAMA_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let var = ny_test_utils::env::ScopedEnvVar::unset("NY_PGD_GAMA");
        Self {
            _var: var,
            _lock: lock,
        }
    }
}

fn make_single_linear_graph(weight: f32, bias: f32) -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[weight]]), Some(arr1(&[bias]))).unwrap(),
    ));
    GraphNetwork::from_sequential(&network).expect("single linear network should convert to graph")
}

fn make_two_output_linear_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0], [-1.0]]), Some(arr1(&[0.0, 0.0]))).unwrap(),
    ));
    GraphNetwork::from_sequential(&network)
        .expect("two-output linear network should convert to graph")
}

fn make_two_output_sigmoid_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0], [-1.0]]), Some(arr1(&[0.0, 0.0]))).unwrap(),
    ));
    network.add_layer(Layer::Sigmoid(SigmoidLayer::new()));
    GraphNetwork::from_sequential(&network)
        .expect("two-output sigmoid network should convert to graph")
}

pub(super) fn make_interval_input(lower: f32, upper: f32) -> BoundedTensor {
    BoundedTensor::new(arr1(&[lower]).into_dyn(), arr1(&[upper]).into_dyn()).unwrap()
}

pub(super) fn make_upper_bound_spec(lower: f32, upper: f32, threshold: f32) -> VnnLibSpec {
    parse_vnnlib(&format!(
        "\
(declare-const X_0 Real)\n\
(declare-const Y_0 Real)\n\
(assert (>= X_0 {lower}))\n\
(assert (<= X_0 {upper}))\n\
(assert (<= Y_0 {threshold}))\n"
    ))
    .unwrap()
}

fn make_maxpool_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::MaxPool2d(MaxPool2dLayer::new(
        (2, 2),
        (2, 2),
        (0, 0),
    )));
    GraphNetwork::from_sequential(&network)
        .expect("single max-pool network should convert to graph")
}

fn make_relu_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::ReLU(ReLULayer));
    GraphNetwork::from_sequential(&network).expect("single relu network should convert to graph")
}

fn make_conv2d_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(
        Conv2dLayer::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0, -0.5, 0.25, 0.75]).unwrap(),
            Some(arr1(&[0.1])),
            (1, 1),
            (0, 0),
        )
        .expect("conv2d kernel should be valid"),
    ));
    GraphNetwork::from_sequential(&network).expect("single conv2d network should convert to graph")
}

fn make_add_graph() -> GraphNetwork {
    let linear_a = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.25]))).unwrap();
    let linear_b = LinearLayer::new(arr2(&[[2.0]]), Some(arr1(&[-0.5]))).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear_a", Layer::Linear(linear_a)));
    graph.add_node(GraphNode::from_input("linear_b", Layer::Linear(linear_b)));
    graph.add_node(GraphNode::binary(
        "add",
        Layer::Add(AddLayer),
        "linear_a",
        "linear_b",
    ));
    graph.set_output("add");
    graph
}

fn make_fixed_reshape_linear_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Reshape(ReshapeLayer::new(vec![4])));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 1.0, 1.0, 1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    GraphNetwork::from_sequential(&network).expect("reshape+linear network should convert to graph")
}

fn make_fixed_flatten_linear_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Flatten(FlattenLayer::new(0)));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 1.0, 1.0, 1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    GraphNetwork::from_sequential(&network).expect("flatten+linear network should convert to graph")
}

fn make_fixed_transpose_linear_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Transpose(TransposeLayer::new(vec![1, 0])));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    GraphNetwork::from_sequential(&network)
        .expect("transpose+linear network should convert to graph")
}

/// ONNX-style batch-preserving transpose: stored perm [0, 2, 1].
///
/// On sequential rank-2 input [1, 4], resolve_perm squeezes out the batch axis
/// and produces [1, 0] (swap the two sample dims).
/// On restart-batched rank-3 input [restart, 1, 4], the stored perm matches
/// ndim directly and resolves to [0, 2, 1] (preserve leading axis, swap last two).
fn make_batch_preserving_transpose_linear_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Transpose(TransposeLayer::new(vec![0, 2, 1])));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    GraphNetwork::from_sequential(&network)
        .expect("batch-preserving transpose+linear network should convert to graph")
}

fn make_add_constant_broadcast_linear_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::AddConstant(AddConstantLayer::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![0.5, -0.25]).unwrap(),
    )));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    GraphNetwork::from_sequential(&network)
        .expect("broadcast add_constant+linear network should convert to graph")
}

fn make_fixed_slice_linear_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Slice(SliceLayer::new(0, 0, 1)));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    GraphNetwork::from_sequential(&network).expect("slice+linear network should convert to graph")
}

fn make_fixed_concat_linear_graph() -> GraphNetwork {
    let linear_a = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.25]))).unwrap();
    let linear_b = LinearLayer::new(arr2(&[[2.0]]), Some(arr1(&[0.5]))).unwrap();
    let linear_out = LinearLayer::new(arr2(&[[1.0, 1.0]]), Some(arr1(&[0.0]))).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear_a", Layer::Linear(linear_a)));
    graph.add_node(GraphNode::from_input("linear_b", Layer::Linear(linear_b)));
    graph.add_node(GraphNode::new(
        "concat",
        Layer::Concat(ConcatLayer::new(0).with_restored_batch_axis_shift(true)),
        vec!["linear_a".to_string(), "linear_b".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear_out",
        Layer::Linear(linear_out),
        vec!["concat".to_string()],
    ));
    graph.set_output("linear_out");
    graph
}

pub(super) fn make_tensor_interval_input(shape: &[usize], lower: f32, upper: f32) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), lower),
        ArrayD::from_elem(IxDyn(shape), upper),
    )
    .unwrap()
}

fn make_rank1_interval_input(lower: f32, upper: f32) -> BoundedTensor {
    make_tensor_interval_input(&[4], lower, upper)
}

fn make_rank4_interval_input(lower: f32, upper: f32) -> BoundedTensor {
    make_tensor_interval_input(&[1, 1, 2, 2], lower, upper)
}

fn make_rank2_interval_input(lower: f32, upper: f32) -> BoundedTensor {
    make_tensor_interval_input(&[1, 4], lower, upper)
}

fn assert_arrays_close(actual: &ArrayD<f32>, expected: &ArrayD<f32>, context: &str) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{context}: shape mismatch, got {:?}, expected {:?}",
        actual.shape(),
        expected.shape()
    );
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected.iter()).enumerate()
    {
        assert!(
            (actual_value - expected_value).abs() <= PRESERVE_LEADING_AXIS_TOL,
            "{context}: element {index} mismatch, got {actual_value}, expected {expected_value}"
        );
    }
}

fn assert_preserve_leading_axis_matches_sequential(
    graph: &GraphNetwork,
    samples: &[ArrayD<f32>],
    engine: Option<&dyn GemmEngine>,
) {
    let sample_views: Vec<_> = samples.iter().map(|sample| sample.view()).collect();
    let batched_samples = ndarray::stack(Axis(0), &sample_views)
        .expect("stacked test inputs should form a valid restart batch");
    let batched_input =
        BoundedTensor::concrete(batched_samples).expect("batched concrete input should be valid");
    let batched_output = graph
        .propagate_ibp_with_engine_preserve_leading_axis(&batched_input, engine)
        .expect("preserve-leading-axis IBP should succeed");

    for (batch_index, sample) in samples.iter().enumerate() {
        let sequential_input =
            BoundedTensor::concrete(sample.clone()).expect("concrete sample should be valid");
        let sequential_output = graph
            .propagate_ibp_with_engine(&sequential_input, engine)
            .expect("sequential IBP should succeed");
        let batched_lower = batched_output
            .lower()
            .index_axis(Axis(0), batch_index)
            .to_owned()
            .into_dyn();
        let batched_upper = batched_output
            .upper()
            .index_axis(Axis(0), batch_index)
            .to_owned()
            .into_dyn();

        assert_arrays_close(
            &batched_lower,
            sequential_output.lower(),
            "lower bounds should match the sequential graph IBP output",
        );
        assert_arrays_close(
            &batched_upper,
            sequential_output.upper(),
            "upper bounds should match the sequential graph IBP output",
        );
    }
}

pub(super) fn make_multi_input_upper_bound_spec(
    input_count: usize,
    lower: f32,
    upper: f32,
    threshold: f32,
) -> VnnLibSpec {
    let mut spec = String::new();
    for index in 0..input_count {
        let _ = writeln!(&mut spec, "(declare-const X_{index} Real)");
    }
    spec.push_str("(declare-const Y_0 Real)\n");
    for index in 0..input_count {
        let _ = writeln!(&mut spec, "(assert (>= X_{index} {lower}))");
        let _ = writeln!(&mut spec, "(assert (<= X_{index} {upper}))");
    }
    let _ = writeln!(&mut spec, "(assert (<= Y_0 {threshold}))");
    parse_vnnlib(&spec).unwrap()
}

#[test]
fn graph_pgd_batched_finds_counterexample_3955() {
    let graph = make_single_linear_graph(2.0, 0.0);
    let input = make_interval_input(-1.0, 1.0);
    let spec = make_upper_bound_spec(-1.0, 1.0, 0.0);
    let engine = NaiveCpuGemmEngine;

    let result = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        8,
        20,
        Default::default(),
        20,
        None,
        Some(&engine),
        true,
        false,
    )
    .expect("batched graph PGD should not error")
    .expect("batched graph PGD should find an unsafe point");

    let (counterexample, output) = result;
    assert!(
        counterexample[[0]] <= 0.0,
        "counterexample should stay on the unsafe half-space"
    );
    assert!(
        output.iter().next().copied().unwrap_or(f32::INFINITY) <= 0.0,
        "reported output should satisfy the unsafe upper-bound constraint"
    );
}

#[test]
fn graph_pgd_conjunctive_gama_honors_expired_deadline_without_model_work() {
    let graph = make_two_output_linear_graph();
    let input = make_interval_input(0.0, 1.0);
    let spec = parse_vnnlib(
        "(declare-const X_0 Real)\n\
         (declare-const Y_0 Real)\n\
         (declare-const Y_1 Real)\n\
         (assert (>= X_0 0.0))\n\
         (assert (<= X_0 1.0))\n\
         (assert (>= Y_0 2.0))\n\
         (assert (<= Y_1 -2.0))\n",
    )
    .unwrap();
    let engine = CountingGemmEngine::new();
    let config = PgdConfig {
        num_restarts: 1,
        num_steps: 10,
        gama_lambda: Some(50.0),
        // checked_sub + now-fallback (the ny-cert `past_instant` house pattern):
        // `now` itself is `<=` any deadline check that runs after this line, so
        // the fallback deadline is just as expired as the 1 ms-past one.
        deadline: Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .unwrap_or_else(Instant::now),
        ),
        ..PgdConfig::default()
    };

    let result =
        try_graph_pgd_upfront_with_config(&graph, &input, &spec, config, Some(&engine), true)
            .expect("expired GAMA attack should return cleanly");

    assert!(result.is_none());
    assert_eq!(
        engine.gemm_calls(),
        0,
        "an expired deadline must stop before initialization or guidance evaluation"
    );
}

#[test]
fn graph_pgd_configured_gama_reaches_generic_batched_spsa_route() {
    let _gama_env = IsolatedGamaEnv::new();
    let graph = make_two_output_sigmoid_graph();
    assert!(
        !graph_supports_exact_gradients(&graph),
        "Sigmoid keeps this regression on the generic batched SPSA dispatcher"
    );
    let input = make_interval_input(0.0, 1.0);
    let spec = parse_vnnlib(
        "(declare-const X_0 Real)\n\
         (declare-const Y_0 Real)\n\
         (declare-const Y_1 Real)\n\
         (assert (>= X_0 0.0))\n\
         (assert (<= X_0 1.0))\n\
         (assert (<= Y_0 -1.0))\n\
         (assert (>= Y_1 2.0))\n",
    )
    .unwrap();
    let engine = NaiveCpuGemmEngine;

    let run = |graph: &GraphNetwork, gama_lambda: Option<f32>| {
        GENERIC_BATCHED_GAMA_OBJECTIVE_EVALS.store(0, std::sync::atomic::Ordering::Relaxed);
        let config = PgdConfig {
            num_restarts: 2,
            num_steps: 2,
            osi_steps: 0,
            gama_lambda,
            ..PgdConfig::default()
        };
        let result =
            try_graph_pgd_upfront_with_config(graph, &input, &spec, config, Some(&engine), true)
                .expect("generic batched graph PGD should complete");
        assert!(
            result.is_none(),
            "impossible raw thresholds must never become a GAMA-authorized candidate"
        );
        GENERIC_BATCHED_GAMA_OBJECTIVE_EVALS.load(std::sync::atomic::Ordering::Relaxed)
    };

    assert!(
        run(&graph, Some(50.0)) > 0,
        "configured GAMA must be consumed inside the generic batched SPSA loop"
    );
    let exact_graph = make_two_output_linear_graph();
    assert!(
        graph_supports_exact_gradients(&exact_graph),
        "linear graph must exercise the exact-eligible override case"
    );
    {
        let _force_batched = ForceBatchedRouting::new();
        assert!(
            run(&exact_graph, Some(50.0)) > 0,
            "NY_PGD_EXACT_BATCHED=0 must not bypass configured GAMA"
        );
    }
    assert_eq!(
        run(&graph, None),
        0,
        "default-off batching must retain the historical raw objective"
    );
    assert_eq!(
        run(&graph, Some(f32::NAN)),
        0,
        "invalid guidance configuration must fall back to the raw objective"
    );
}

#[test]
fn graph_pgd_batched_reduces_gemm_dispatches_3955() {
    let _force_batched = ForceBatchedRouting::new();
    let graph = make_single_linear_graph(1.0, 10.0);
    let input = make_interval_input(0.0, 1.0);
    let spec = make_upper_bound_spec(0.0, 1.0, 0.0);
    let engine = CountingGemmEngine::new();

    let result = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        10,
        5,
        Default::default(),
        20,
        None,
        Some(&engine),
        true,
        false,
    )
    .expect("batched graph PGD should not error");

    assert!(
        result.is_none(),
        "network output stays above zero, so no unsafe counterexample should exist"
    );
    assert!(
        engine.gemm_calls() <= BATCHED_GEMM_DISPATCH_MAX,
        "#3955 regression: batched graph PGD should use one GEMM dispatch per restart phase, got {}",
        engine.gemm_calls()
    );
}

#[test]
fn graph_pgd_batched_preserves_rank4_inputs_3955() {
    let graph = make_maxpool_graph();
    let input = make_rank4_interval_input(-1.0, -0.5);
    let spec = make_multi_input_upper_bound_spec(4, -1.0, -0.5, -0.25);
    let engine = NaiveCpuGemmEngine;

    let result = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        4,
        0,
        Default::default(),
        20,
        None,
        Some(&engine),
        true,
        false,
    )
    .expect("batched graph PGD should keep rank-4 inputs compatible with graph IBP")
    .expect("uniformly unsafe rank-4 inputs should be detected during the batched random sample");

    let (counterexample, output) = result;
    assert_eq!(
        counterexample.shape(),
        &[1, 1, 2, 2],
        "#3955 regression: batched graph PGD should preserve the original rank-4 input shape"
    );
    assert!(
        output.iter().next().copied().unwrap_or(f32::INFINITY) <= -0.25,
        "max-pool output should satisfy the unsafe upper-bound constraint"
    );
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_relu_4096() {
    let graph = make_relu_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        arr1(&[-0.75]).into_dyn(),
        arr1(&[0.25]).into_dyn(),
        arr1(&[1.5]).into_dyn(),
    ];

    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_add_4096() {
    let graph = make_add_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        arr1(&[-1.0]).into_dyn(),
        arr1(&[0.5]).into_dyn(),
        arr1(&[2.0]).into_dyn(),
    ];

    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_conv2d_4096() {
    let graph = make_conv2d_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        ArrayD::from_shape_vec(
            IxDyn(&[1, 3, 3]),
            vec![1.0, -1.0, 0.5, 0.0, 2.0, -0.5, 1.5, 0.25, -0.75],
        )
        .unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[1, 3, 3]),
            vec![-0.5, 0.75, 1.25, 1.0, -1.5, 0.0, 0.5, -0.25, 2.0],
        )
        .unwrap(),
    ];

    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_batched_handles_fixed_reshape_graphs_4093() {
    let _force_batched = ForceBatchedRouting::new();
    let graph = make_fixed_reshape_linear_graph();
    let input = make_rank4_interval_input(0.0, 1.0);
    let spec = make_multi_input_upper_bound_spec(4, 0.0, 1.0, -1.0);
    let engine = CountingGemmEngine::new();

    let result = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        10,
        5,
        Default::default(),
        20,
        None,
        Some(&engine),
        true,
        false,
    )
    .expect("reshape graphs should use the batched graph PGD path");

    assert!(
        result.is_none(),
        "the fixed reshape graph should stay safely above the impossible threshold"
    );
    assert!(
        (1..=BATCHED_GEMM_DISPATCH_MAX).contains(&engine.gemm_calls()),
        "#4093 regression: reshape-compatible graph PGD should keep batched GEMM dispatch count low, got {}",
        engine.gemm_calls()
    );
}

#[test]
fn graph_pgd_batched_handles_fixed_flatten_graphs_4093() {
    let _force_batched = ForceBatchedRouting::new();
    let graph = make_fixed_flatten_linear_graph();
    let input = make_rank4_interval_input(0.0, 1.0);
    let spec = make_multi_input_upper_bound_spec(4, 0.0, 1.0, -1.0);
    let engine = CountingGemmEngine::new();

    let result = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        10,
        5,
        Default::default(),
        20,
        None,
        Some(&engine),
        true,
        false,
    )
    .expect("flatten graphs should use the batched graph PGD path");

    assert!(
        result.is_none(),
        "the fixed flatten graph should stay safely above the impossible threshold"
    );
    assert!(
        (1..=BATCHED_GEMM_DISPATCH_MAX).contains(&engine.gemm_calls()),
        "#4093 regression: flatten-compatible graph PGD should keep batched GEMM dispatch count low, got {}",
        engine.gemm_calls()
    );
}

/// #4094: Prove ONNX-style batch-preserving transpose (perm=[0,2,1]) produces
/// identical batched-vs-sequential IBP output via preserve_leading_axis.
#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_batch_preserving_transpose_4094() {
    let graph = make_batch_preserving_transpose_linear_graph();
    let engine = NaiveCpuGemmEngine;
    // Rank-2 samples [1, 4]: transpose [0,2,1] resolves to [1,0] on 2D
    // (batch axis squeezed), so each sample transposes [1, 4] → [4, 1].
    let samples = vec![
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0, -0.5, 0.25, 0.75]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![-1.0, 2.0, 0.5, -0.25]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.0, 0.0, 1.0, -1.0]).unwrap(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

/// #4094: Prove batch-preserving transpose graphs now take the batched restart
/// PGD lane (low GEMM dispatch count), joining reshape/flatten.
#[test]
fn graph_pgd_batched_handles_batch_preserving_transpose_graphs_4094() {
    let graph = make_batch_preserving_transpose_linear_graph();
    let input = make_rank2_interval_input(0.0, 1.0);
    let spec = make_multi_input_upper_bound_spec(4, 0.0, 1.0, -1.0);
    let engine = CountingGemmEngine::new();

    let result = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        10,
        5,
        Default::default(),
        20,
        None,
        Some(&engine),
        true,
        false,
    )
    .expect("batch-preserving transpose graphs should use the batched graph PGD path");

    assert!(
        result.is_none(),
        "the batch-preserving transpose graph should stay safely above the impossible threshold"
    );
    assert!(
        (1..=BATCHED_GEMM_DISPATCH_MAX).contains(&engine.gemm_calls()),
        "#4094 regression: batch-preserving transpose graph PGD should keep batched GEMM dispatch count low, got {}",
        engine.gemm_calls()
    );
}

/// #4094: Manual unbatched transpose (perm=[1,0]) enters the batched path
/// (Transpose is now whitelisted) but the runtime resolve_perm raises
/// ShapeMismatch on rank-3 batched input, triggering FallbackToSequential.
/// Sequential fallback produces >10 GEMM calls.
#[test]
fn graph_pgd_manual_transpose_falls_back_via_runtime_shape_mismatch_4094() {
    let graph = make_fixed_transpose_linear_graph();
    let input = make_rank2_interval_input(0.0, 1.0);
    let spec = make_multi_input_upper_bound_spec(4, 0.0, 1.0, -1.0);
    let engine = CountingGemmEngine::new();

    let result = try_graph_pgd_upfront(&graph, &input, &spec, 10, 5, Default::default(), 20, None, Some(&engine), true, false)
        .expect("manual unbatched transpose graphs should fall back to sequential via runtime ShapeMismatch");

    assert!(
        result.is_none(),
        "the manual transpose graph should stay safely above the impossible threshold"
    );
    assert!(
        engine.gemm_calls() > 10,
        "#4094 contract: manual unbatched transpose (perm=[1,0]) should fall back to sequential via runtime ShapeMismatch, got {} GEMM calls",
        engine.gemm_calls()
    );
}

#[test]
fn graph_pgd_batched_shape_mismatch_falls_back_to_sequential_4093() {
    let graph = make_add_constant_broadcast_linear_graph();
    let input = make_tensor_interval_input(&[2], 0.0, 1.0);
    let spec = make_multi_input_upper_bound_spec(2, 0.0, 1.0, -1.0);
    let engine = CountingGemmEngine::new();

    let result = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        10,
        5,
        Default::default(),
        20,
        None,
        Some(&engine),
        true,
        false,
    )
    .expect("shape-mismatch batched graphs should fall back to sequential PGD");

    assert!(
        result.is_none(),
        "the broadcast add_constant graph should stay safely above the impossible threshold"
    );
    assert!(
        engine.gemm_calls() > 10,
        "#4093 regression: shape-mismatch batched graphs should still execute sequential PGD after fallback, got {} GEMM calls",
        engine.gemm_calls()
    );
}

#[test]
fn graph_pgd_whitelist_rejects_unproven_operators_4096() {
    let graph = make_fixed_slice_linear_graph();
    let input = make_rank1_interval_input(0.0, 1.0);
    let spec = make_multi_input_upper_bound_spec(4, 0.0, 1.0, -1.0);
    let engine = CountingGemmEngine::new();

    let result = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        10,
        5,
        Default::default(),
        20,
        None,
        Some(&engine),
        true,
        false,
    )
    .expect("slice graphs should stay on the sequential fallback path");

    assert!(
        result.is_none(),
        "the fixed slice graph should stay safely above the impossible threshold"
    );
    assert!(
        engine.gemm_calls() > 10,
        "#4096 regression: unwhitelisted slice graphs should still miss the batched restart path, got {} GEMM calls",
        engine.gemm_calls()
    );
}

#[test]
fn graph_pgd_whitelist_rejects_concat_graphs_4096() {
    let graph = make_fixed_concat_linear_graph();
    let input = make_interval_input(0.0, 1.0);
    let spec = make_upper_bound_spec(0.0, 1.0, -1.0);
    let engine = CountingGemmEngine::new();

    let result = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        10,
        5,
        Default::default(),
        20,
        None,
        Some(&engine),
        true,
        false,
    )
    .expect("concat graphs should stay on the sequential fallback path");

    assert!(
        result.is_none(),
        "the fixed concat graph should stay safely above the impossible threshold"
    );
    assert!(
        engine.gemm_calls() > 10,
        "#4096 regression: concat graphs should still miss the batched restart path, got {} GEMM calls",
        engine.gemm_calls()
    );
}

// --- #4096 equivalence tests: whitelisted families ---
// Each test proves batched preserve-leading-axis output matches sequential per-sample output.
// Acceptance criteria #2: "Each allowed family has a direct batched-vs-sequential regression."

fn make_linear_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, -0.5], [0.25, 0.75]]), Some(arr1(&[0.1, -0.2]))).unwrap(),
    ));
    GraphNetwork::from_sequential(&network).expect("single linear network should convert to graph")
}

fn make_sigmoid_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Sigmoid(SigmoidLayer::new()));
    GraphNetwork::from_sequential(&network).expect("single sigmoid network should convert to graph")
}

fn make_tanh_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Tanh(TanhLayer::new()));
    GraphNetwork::from_sequential(&network).expect("single tanh network should convert to graph")
}

fn make_batchnorm_graph() -> GraphNetwork {
    let ny = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 0.5]).unwrap();
    let beta = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.1]).unwrap();
    let mean = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, -0.25]).unwrap();
    let var = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap();
    let bn = BatchNormLayer::new(&ny, &beta, &mean, &var, 1e-5)
        .expect("batchnorm params should be valid");
    let mut network = Network::new();
    network.add_layer(Layer::BatchNorm(bn));
    GraphNetwork::from_sequential(&network)
        .expect("single batchnorm network should convert to graph")
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_linear_4096() {
    let graph = make_linear_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, -0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_sigmoid_4096() {
    let graph = make_sigmoid_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        arr1(&[-2.0]).into_dyn(),
        arr1(&[0.0]).into_dyn(),
        arr1(&[3.0]).into_dyn(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_tanh_4096() {
    let graph = make_tanh_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        arr1(&[-1.5]).into_dyn(),
        arr1(&[0.5]).into_dyn(),
        arr1(&[2.5]).into_dyn(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_batchnorm_4096() {
    let graph = make_batchnorm_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, -0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 0.25]).unwrap(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_maxpool2d_4096() {
    let graph = make_maxpool_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0, -0.5, 0.25, 0.75]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![-1.0, 2.0, 0.5, -0.25]).unwrap(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

// --- #4096 rejection tests: axis-sensitive blocked families ---
// Acceptance criteria #3: "Each disallowed family is explicitly blocked by the guard."

fn make_softmax_linear_graph() -> GraphNetwork {
    let mut network = Network::new();
    // Stored axis 0 models a positive ONNX axis after ny's unbatched
    // `axis - 1` rewrite. The preserve-leading-axis path must shift it back to
    // the sample axis when restart batching prepends a leading dimension.
    network.add_layer(Layer::Softmax(SoftmaxLayer::new(0)));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    GraphNetwork::from_sequential(&network).expect("softmax+linear network should convert to graph")
}

// --- #4096 equivalence tests: softmax/logsoftmax whitelisted families ---

fn make_logsoftmax_linear_graph() -> GraphNetwork {
    let mut network = Network::new();
    // Same stored-axis contract as `make_softmax_linear_graph()`: use a
    // non-negative stored axis so the graph-level regression exercises the
    // restart-axis restoration logic instead of the already-correct `axis=-1`
    // path.
    network.add_layer(Layer::LogSoftmax(LogSoftmaxLayer::new(0)));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    GraphNetwork::from_sequential(&network)
        .expect("logsoftmax+linear network should convert to graph")
}

fn make_causal_softmax_linear_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::CausalSoftmax(CausalSoftmaxLayer::new(-1)));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    GraphNetwork::from_sequential(&network)
        .expect("causal_softmax+linear network should convert to graph")
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_softmax_4096() {
    let graph = make_softmax_linear_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_logsoftmax_4096() {
    let graph = make_logsoftmax_linear_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_whitelist_rejects_causal_softmax_4096() {
    let graph = make_causal_softmax_linear_graph();
    // CausalSoftmax requires at least 2D input (it applies a causal mask).
    let input = make_tensor_interval_input(&[1, 2], 0.0, 1.0);
    let spec = make_multi_input_upper_bound_spec(2, 0.0, 1.0, -1.0);
    let engine = CountingGemmEngine::new();

    let result = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        10,
        5,
        Default::default(),
        20,
        None,
        Some(&engine),
        true,
        false,
    )
    .expect("causal_softmax graphs should stay on the sequential fallback path");

    assert!(
        result.is_none(),
        "the fixed causal_softmax graph should stay safely above the impossible threshold"
    );
    assert!(
        engine.gemm_calls() > 10,
        "#4096 regression: CausalSoftmax graphs should miss the batched restart path, got {} GEMM calls",
        engine.gemm_calls()
    );
}

#[path = "tests_graph_pgd_round_two.rs"]
mod round_two;

#[path = "tests_graph_pgd_round_three.rs"]
mod round_three;

#[path = "tests_graph_pgd_round_four.rs"]
mod round_four;

#[path = "tests_graph_pgd_round_six.rs"]
mod round_six;
