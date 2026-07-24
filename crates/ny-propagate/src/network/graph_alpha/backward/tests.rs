// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::layers::{LinearLayer, ReLULayer, SigmoidLayer};
use crate::network::core::GraphNode;
use ndarray::{arr1, arr2, Array1, ArrayD, IxDyn};
use ny_core::{
    GemmEngine, GpuCrownBackward, GpuCrownLayer, GpuCrownResult, GpuCrownSeed, NaiveCpuGemmEngine,
    Result as NyResult,
};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

fn test_input() -> BoundedTensor {
    BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap()
}

fn empty_dag_alpha_context_4205<'a>(
    input: &'a BoundedTensor,
    relu_name_to_idx: &'a HashMap<String, usize>,
    alpha_state: &'a GraphAlphaState,
    gradients: &'a mut [Array1<f32>],
    gradients_upper: &'a mut [Array1<f32>],
    node_crown_bounds: &'a mut CrownMergeAccumulator,
    input_accumulated: &'a mut bool,
) -> DagAlphaNodeContext<'a> {
    DagAlphaNodeContext {
        input,
        relu_name_to_idx,
        alpha_state,
        invprop_state: None,
        gradients,
        gradients_upper,
        track_gradients: true,
        node_crown_bounds,
        intermediate: None,
        output_dim: 1,
        input_dim: 1,
        input_accumulated,
        engine: None,
        deadline: None,
    }
}

fn assert_linear_bounds_eq_4205(actual: &LinearBounds, expected: &LinearBounds) {
    assert_eq!(actual.lower_a(), expected.lower_a());
    assert_eq!(actual.lower_b(), expected.lower_b());
    assert_eq!(actual.upper_a(), expected.upper_a());
    assert_eq!(actual.upper_b(), expected.upper_b());
}

#[ntest::timeout(10000)]
#[test]
fn test_chain_rule_gradients_missing_a_matrix_returns_correct_length_1937() {
    let graph = GraphNetwork::new();
    let mut intermediate = GraphAlphaCrownIntermediate::new();
    intermediate.pre_relu_bounds.insert(
        "relu1".to_string(),
        (arr1(&[-1.0_f32, -0.2, 0.1]), arr1(&[0.5_f32, 0.8, 1.2])),
    );

    let gradients = graph.compute_graph_chain_rule_gradients(
        &test_input(),
        &["relu1".to_string()],
        &intermediate,
    );

    assert_eq!(gradients.len(), 1);
    assert_eq!(gradients[0].len(), 3);
    assert!(
        gradients[0].iter().all(|&v| v == 0.0),
        "Expected zero fallback gradient when A matrix is missing"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_chain_rule_gradients_missing_both_a_and_bounds_returns_empty_1937() {
    let graph = GraphNetwork::new();
    let intermediate = GraphAlphaCrownIntermediate::new();

    let gradients = graph.compute_graph_chain_rule_gradients(
        &test_input(),
        &["relu1".to_string()],
        &intermediate,
    );

    assert_eq!(gradients.len(), 1);
    assert!(
        gradients[0].is_empty(),
        "Expected empty gradient when both A matrix and pre-ReLU bounds are missing"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_chain_rule_gradients_a_present_but_bounds_missing() {
    let graph = GraphNetwork::new();
    let mut intermediate = GraphAlphaCrownIntermediate::new();
    intermediate.a_at_relu.insert(
        "relu1".to_string(),
        arr2(&[[1.0_f32, -2.0, 0.5, 0.0], [0.3_f32, 1.1, -0.4, 2.2]]),
    );

    let gradients = graph.compute_graph_chain_rule_gradients(
        &test_input(),
        &["relu1".to_string()],
        &intermediate,
    );

    assert_eq!(gradients.len(), 1);
    assert_eq!(gradients[0].len(), 4);
    assert!(
        gradients[0].iter().all(|&v| v == 0.0),
        "Expected zero fallback gradient when pre-ReLU bounds are missing"
    );
}

/// Regression test: NaN pre-ReLU bounds must produce zero gradients, not NaN.
///
/// Before the #2809 parity fix, IEEE-754 NaN comparisons caused NaN bounds
/// to fall through the `l >= 0.0 || u <= 0.0` guard as "unstable", producing
/// NaN gradient contributions that corrupt the Adam optimizer state.
/// The sequential path (helpers.rs) was fixed in #2809 but the graph path
/// was not updated until this regression was caught.
#[ntest::timeout(10000)]
#[test]
fn test_chain_rule_gradients_nan_pre_relu_bounds_produce_zero_gradients_2809() {
    let graph = GraphNetwork::new();
    let mut intermediate = GraphAlphaCrownIntermediate::new();

    // A matrix with positive entries that would produce non-zero gradients
    // for any unstable neuron.
    intermediate.a_at_relu.insert(
        "relu1".to_string(),
        arr2(&[[1.0_f32, 2.0, 3.0], [0.5_f32, 1.5, 2.5]]),
    );

    // Pre-ReLU bounds: neuron 0 is unstable (finite), neuron 1 has NaN lower,
    // neuron 2 has NaN upper. Only neuron 0 should contribute to gradients.
    intermediate.pre_relu_bounds.insert(
        "relu1".to_string(),
        (
            arr1(&[-0.5_f32, f32::NAN, -0.3]),
            arr1(&[0.5_f32, 0.8, f32::NAN]),
        ),
    );

    let gradients = graph.compute_graph_chain_rule_gradients(
        &test_input(),
        &["relu1".to_string()],
        &intermediate,
    );

    assert_eq!(gradients.len(), 1);
    assert_eq!(gradients[0].len(), 3);

    // Neuron 0: unstable (l=-0.5, u=0.5), grad = 1.0*(-0.5) + 0.5*(-0.5) = -0.75
    assert!(
        gradients[0][0].is_finite(),
        "Neuron 0 gradient should be finite: {}",
        gradients[0][0]
    );
    let expected_grad_0 = 1.0_f32 * (-0.5) + 0.5 * (-0.5);
    assert!(
        (gradients[0][0] - expected_grad_0).abs() < 1e-6,
        "Neuron 0 gradient should be {expected_grad_0}, got {}",
        gradients[0][0]
    );

    // Neuron 1: NaN lower bound -> must be zero, not NaN
    assert_eq!(
        gradients[0][1], 0.0,
        "#2809 regression: NaN lower bound must produce zero gradient, got {}",
        gradients[0][1]
    );

    // Neuron 2: NaN upper bound -> must be zero, not NaN
    assert_eq!(
        gradients[0][2], 0.0,
        "#2809 regression: NaN upper bound must produce zero gradient, got {}",
        gradients[0][2]
    );

    // Belt-and-suspenders: no NaN anywhere in the gradient vector
    assert!(
        gradients[0].iter().all(|v| v.is_finite()),
        "#2809 regression: all gradients must be finite, got {:?}",
        gradients[0]
    );
}

/// Regression test: NaN A matrix coefficients must not corrupt gradients.
///
/// The A coefficient NaN case is accidentally safe via IEEE-754 (NaN > 0.0
/// returns false), but the explicit guard from #2809 makes this behavior
/// documented and robust rather than relying on IEEE-754 subtleties.
#[ntest::timeout(10000)]
#[test]
fn test_chain_rule_gradients_nan_a_coefficient_produces_zero_contribution_2809() {
    let graph = GraphNetwork::new();
    let mut intermediate = GraphAlphaCrownIntermediate::new();

    // A matrix: row 0 has NaN at column 0, row 1 has a finite positive value.
    intermediate.a_at_relu.insert(
        "relu1".to_string(),
        arr2(&[[f32::NAN, 1.0_f32], [2.0_f32, 0.5]]),
    );

    // Neuron 0 and 1 are both unstable.
    intermediate.pre_relu_bounds.insert(
        "relu1".to_string(),
        (arr1(&[-0.5_f32, -0.3]), arr1(&[0.5_f32, 0.7])),
    );

    let gradients = graph.compute_graph_chain_rule_gradients(
        &test_input(),
        &["relu1".to_string()],
        &intermediate,
    );

    assert_eq!(gradients.len(), 1);
    assert_eq!(gradients[0].len(), 2);

    // Neuron 0: A[0,0]=NaN (skipped), A[1,0]=2.0 > 0 -> grad = 2.0 * (-0.5) = -1.0
    assert!(
        gradients[0][0].is_finite(),
        "#2809 regression: NaN A coefficient must not produce NaN gradient, got {}",
        gradients[0][0]
    );
    let expected_grad_0 = 2.0_f32 * (-0.5);
    assert!(
        (gradients[0][0] - expected_grad_0).abs() < 1e-6,
        "Neuron 0 gradient should be {expected_grad_0}, got {}",
        gradients[0][0]
    );

    // Neuron 1: A[0,1]=1.0 > 0, A[1,1]=0.5 > 0 -> grad = 1.0*(-0.3) + 0.5*(-0.3) = -0.45
    let expected_grad_1 = 1.0_f32 * (-0.3) + 0.5 * (-0.3);
    assert!(
        (gradients[0][1] - expected_grad_1).abs() < 1e-6,
        "Neuron 1 gradient should be {expected_grad_1}, got {}",
        gradients[0][1]
    );

    assert!(
        gradients[0].iter().all(|v| v.is_finite()),
        "#2809 regression: all gradients must be finite, got {:?}",
        gradients[0]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_monotone_shape_mismatch_retries_fixed_slope_4118() {
    let node_lb = LinearBounds::identity(1);
    let pre_activation =
        BoundedTensor::new(arr1(&[-0.5_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("bounded tensor should construct");
    let expected = LinearBounds::new(
        arr2(&[[0.25_f32]]),
        arr1(&[0.1_f32]),
        arr2(&[[0.5_f32]]),
        arr1(&[0.2_f32]),
    )
    .expect("expected fixed-slope bounds should construct");
    let alpha_called = Cell::new(false);
    let fixed_called = Cell::new(false);

    let result = retry_monotone_shape_mismatch_with_fixed_slope(
        "sigmoid_hidden",
        "Sigmoid",
        &node_lb,
        &pre_activation,
        |_, _| {
            alpha_called.set(true);
            Err(NyError::ShapeMismatch {
                expected: vec![1],
                got: vec![2],
            })
        },
        |_, _| {
            fixed_called.set(true);
            Ok(expected.clone())
        },
    )
    .expect("ShapeMismatch should retry the local fixed-slope path");

    assert!(alpha_called.get(), "alpha path should be attempted first");
    assert!(
        fixed_called.get(),
        "fixed-slope retry should run after alpha ShapeMismatch"
    );
    assert_eq!(result.lower_a, expected.lower_a);
    assert_eq!(result.lower_b, expected.lower_b);
    assert_eq!(result.upper_a, expected.upper_a);
    assert_eq!(result.upper_b, expected.upper_b);
}

/// Regression test for #4118: when both alpha AND fixed-slope retry fail with
/// ShapeMismatch, the error must propagate out (so the outer backward-pass
/// match can catch it and fall back to plain CROWN, rather than cascading to
/// graph-wide IBP).
#[ntest::timeout(10000)]
#[test]
fn test_monotone_shape_mismatch_both_paths_fail_propagates_error_4118() {
    let node_lb = LinearBounds::identity(1);
    let pre_activation =
        BoundedTensor::new(arr1(&[-0.5_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("bounded tensor should construct");
    let alpha_called = Cell::new(false);
    let fixed_called = Cell::new(false);

    let result = retry_monotone_shape_mismatch_with_fixed_slope(
        "sigmoid_hidden",
        "Sigmoid",
        &node_lb,
        &pre_activation,
        |_, _| {
            alpha_called.set(true);
            Err(NyError::ShapeMismatch {
                expected: vec![1],
                got: vec![2],
            })
        },
        |_, _| {
            fixed_called.set(true);
            Err(NyError::ShapeMismatch {
                expected: vec![1],
                got: vec![3],
            })
        },
    );

    assert!(alpha_called.get(), "alpha path should be attempted first");
    assert!(
        fixed_called.get(),
        "fixed-slope retry should run after alpha ShapeMismatch"
    );
    assert!(
        result.is_err(),
        "both paths failing must propagate the error for CROWN fallback"
    );
    assert!(
        matches!(result, Err(NyError::ShapeMismatch { .. })),
        "error should be ShapeMismatch from the fixed-slope retry"
    );
}

/// Regression test for #4118: UnsupportedConfiguration from fixed-slope retry
/// must also propagate so the backward-pass match catches it for CROWN fallback.
#[ntest::timeout(10000)]
#[test]
fn test_monotone_unsupported_config_propagates_for_crown_fallback_4118() {
    let node_lb = LinearBounds::identity(1);
    let pre_activation =
        BoundedTensor::new(arr1(&[-0.5_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("bounded tensor should construct");

    let result = retry_monotone_shape_mismatch_with_fixed_slope(
        "tanh_hidden",
        "Tanh",
        &node_lb,
        &pre_activation,
        |_, _| {
            Err(NyError::ShapeMismatch {
                expected: vec![1],
                got: vec![2],
            })
        },
        |_, _| {
            Err(NyError::UnsupportedConfiguration(
                "fixed-slope Tanh shape not supported".to_string(),
            ))
        },
    );

    assert!(
        result.is_err(),
        "UnsupportedConfiguration from fixed-slope must propagate"
    );
    assert!(
        matches!(result, Err(NyError::UnsupportedConfiguration(_))),
        "error should be UnsupportedConfiguration from the fixed-slope retry"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_handle_nonlinear_node_sigmoid_without_alpha_returns_not_handled_4205() {
    let graph = GraphNetwork::new();
    let node = GraphNode::from_input("sigmoid_hidden", Layer::Sigmoid(SigmoidLayer::new()));
    let expected = LinearBounds::identity(1);
    let pre_activation =
        BoundedTensor::new(arr1(&[-0.5_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("bounded tensor should construct");
    let relu_name_to_idx = HashMap::new();
    let alpha_state = GraphAlphaState::new();
    let mut gradients: Vec<Array1<f32>> = Vec::new();
    let mut gradients_upper: Vec<Array1<f32>> = Vec::new();
    let mut node_crown_bounds = CrownMergeAccumulator::new();
    let mut input_accumulated = false;

    let result = handle_nonlinear_node(
        &graph,
        "sigmoid_hidden",
        &node,
        NETWORK_INPUT,
        CrownBounds::Dense(expected.clone()),
        &pre_activation,
        empty_dag_alpha_context_4205(
            &pre_activation,
            &relu_name_to_idx,
            &alpha_state,
            gradients.as_mut_slice(),
            gradients_upper.as_mut_slice(),
            &mut node_crown_bounds,
            &mut input_accumulated,
        ),
    )
    .expect("missing monotone alpha should not error");

    let NonlinearNodeResult::NotHandled(returned_cb) = result else {
        panic!("missing monotone alpha should leave nonlinear handling to the caller");
    };
    let returned = returned_cb
        .into_dense()
        .expect("not-handled result should preserve dense bounds");
    assert_linear_bounds_eq_4205(&returned, &expected);
    assert!(
        !input_accumulated,
        "missing alpha must not accumulate input bounds"
    );
    assert!(
        node_crown_bounds
            .take(NETWORK_INPUT)
            .expect("merge accumulator lookup should succeed")
            .is_none(),
        "missing alpha must not mutate accumulated crown bounds"
    );
}

// ========================================================================
// GPU suffix integration tests for try_alpha_backward_gpu_suffix
// ========================================================================

/// Scripted GPU engine that records seeded suffix calls and returns
/// pre-configured bounds. Modeled on SeededSuffixScriptedEngine in
/// `gpu_suffix_tests.rs` but simplified for backward-pass integration.
struct BackwardGpuSuffixEngine {
    seeded_calls: AtomicUsize,
    result_lower: Vec<f32>,
    result_upper: Vec<f32>,
}

impl BackwardGpuSuffixEngine {
    fn new(result_lower: Vec<f32>, result_upper: Vec<f32>) -> Self {
        Self {
            seeded_calls: AtomicUsize::new(0),
            result_lower,
            result_upper,
        }
    }

    fn seeded_call_count(&self) -> usize {
        self.seeded_calls.load(Ordering::SeqCst)
    }
}

impl GemmEngine for BackwardGpuSuffixEngine {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> NyResult<Vec<f32>> {
        NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
    }

    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}

impl GpuCrownBackward for BackwardGpuSuffixEngine {
    fn crown_backward_gpu(
        &self,
        _layers: &[GpuCrownLayer],
        _spec: &[f32],
        _num_specs: usize,
        _input_lower: &[f32],
        _input_upper: &[f32],
    ) -> NyResult<GpuCrownResult> {
        panic!("unexpected full GPU CROWN dispatch in backward suffix test");
    }

    fn crown_backward_gpu_seeded(
        &self,
        layers: &[GpuCrownLayer],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        _input_upper: &[f32],
    ) -> NyResult<GpuCrownResult> {
        self.seeded_calls.fetch_add(1, Ordering::SeqCst);

        assert!(
            seed.lower_a.iter().all(|v| v.is_finite()),
            "seeded GPU suffix should receive finite lower_a"
        );
        assert!(
            seed.upper_a.iter().all(|v| v.is_finite()),
            "seeded GPU suffix should receive finite upper_a"
        );
        assert!(
            input_lower.iter().all(|v| v.is_finite()),
            "seeded GPU suffix should receive finite input bounds"
        );
        assert!(
            !layers.is_empty(),
            "seeded GPU suffix should receive at least one GPU layer"
        );

        Ok(GpuCrownResult {
            lower_bounds: self.result_lower.clone(),
            upper_bounds: self.result_upper.clone(),
        })
    }
}

/// Build a simple chain graph: Input(2) → Linear1(2→2) → ReLU → Linear2(2→1).
/// This topology allows the GPU suffix to fire after Linear2 backward produces
/// LinearBounds — the remaining chain (ReLU → Linear1) is unary and GPU-extractable.
fn build_chain_graph_for_gpu_suffix() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[1.0_f32, -0.5], [0.3, 0.8]]);
    let b1 = arr1(&[0.1_f32, -0.2]);
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("valid Linear1")),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.6_f32, -0.4]]);
    let b2 = arr1(&[0.05_f32]);
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).expect("valid Linear2")),
        vec!["relu1".to_string()],
    ));
    graph.set_output("linear2");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -0.5_f32),
        ArrayD::from_elem(IxDyn(&[2]), 0.5_f32),
    )
    .expect("valid input");

    (graph, input)
}

/// Shared setup for GPU suffix backward tests: builds graph, IBP bounds,
/// alpha state, and runs the backward pass with the given engine.
fn run_chain_graph_backward(engine: Option<&dyn GemmEngine>, context: &str) -> BoundedTensor {
    run_chain_graph_backward_mode(engine, context, true)
}

fn run_chain_graph_backward_mode(
    engine: Option<&dyn GemmEngine>,
    context: &str,
    track_gradients: bool,
) -> BoundedTensor {
    let (graph, input) = build_chain_graph_for_gpu_suffix();
    let ibp_bounds = graph
        .collect_node_bounds(&input)
        .expect("IBP forward should succeed");
    let exec_order = graph.exec_order().expect("exec_order").to_vec();
    let relu_pre = graph
        .relu_preactivation_bounds("relu1", &input, &ibp_bounds, context)
        .expect("ReLU pre-activation bounds");
    let mut alpha_state = GraphAlphaState::new();
    alpha_state
        .add_relu_node("relu1", relu_pre, false)
        .expect("alpha init");
    let relu_name_to_idx: HashMap<String, usize> = [("relu1".to_string(), 0)].into_iter().collect();
    let mut gradients = vec![Array1::<f32>::zeros(2)];
    let mut gradients_upper = vec![Array1::<f32>::zeros(2)];
    let result = if track_gradients {
        graph.dag_alpha_backward_pass_with_engine(
            &input,
            &ibp_bounds,
            &exec_order,
            1,
            2,
            &relu_name_to_idx,
            &alpha_state,
            None,
            &mut gradients,
            &mut gradients_upper,
            engine,
            None,
            None,
            None,
        )
    } else {
        graph.dag_alpha_bound_pass_with_engine(
            &input,
            &ibp_bounds,
            &exec_order,
            1,
            2,
            &relu_name_to_idx,
            &alpha_state,
            None,
            engine,
            None,
            None,
            None,
        )
    };
    result.unwrap_or_else(|e| panic!("{context} backward failed: {e}"))
}

/// Regression test: `try_alpha_backward_gpu_suffix` fires on a linear chain
/// graph when a GPU engine is provided.
#[ntest::timeout(10000)]
#[test]
fn test_dag_backward_gpu_suffix_fires_on_chain_graph() {
    // Fast-mock dispatch test — gate OFF so the fast seeded GPU suffix runs (the
    // production default is now sound, which would mask this non-sound mock).
    // #gpu-crown-sound-default.
    let _gate = crate::sound_gpu_gate::test_lock::lock_gate();
    let cpu_bounds = run_chain_graph_backward(None, "cpu-baseline");

    let engine = BackwardGpuSuffixEngine::new(
        cpu_bounds.lower().iter().copied().collect(),
        cpu_bounds.upper().iter().copied().collect(),
    );
    let gpu_bounds = run_chain_graph_backward(Some(&engine), "gpu-suffix");

    assert_eq!(
        engine.seeded_call_count(),
        1,
        "GPU suffix should fire exactly once on a chain graph"
    );
    let cpu_lower: Vec<f32> = cpu_bounds.lower().iter().copied().collect();
    let cpu_upper: Vec<f32> = cpu_bounds.upper().iter().copied().collect();
    let gpu_lower: Vec<f32> = gpu_bounds.lower().iter().copied().collect();
    let gpu_upper: Vec<f32> = gpu_bounds.upper().iter().copied().collect();
    assert_eq!(gpu_lower, cpu_lower, "GPU suffix lower must match CPU");
    assert_eq!(gpu_upper, cpu_upper, "GPU suffix upper must match CPU");
}

/// Regression test: GPU suffix does NOT fire when engine is None.
#[ntest::timeout(10000)]
#[test]
fn test_dag_backward_no_gpu_suffix_without_engine() {
    let bounds = run_chain_graph_backward(None, "no-engine");
    assert!(
        bounds.lower().iter().all(|v| v.is_finite()),
        "CPU-only bounds must be finite"
    );
    assert!(
        bounds.upper().iter().all(|v| v.is_finite()),
        "CPU-only bounds must be finite"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_dag_alpha_bound_only_is_byte_identical_without_engine() {
    let with_grad = run_chain_graph_backward_mode(None, "with-gradient", true);
    let bound_only = run_chain_graph_backward_mode(None, "bound-only", false);
    assert_eq!(
        bound_only
            .lower()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        with_grad
            .lower()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        bound_only
            .upper()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        with_grad
            .upper()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

// #margin-subset-alpha: DAG alpha per-iteration backward margin-subset tests.

/// input(2) -> Linear(2->3) "pre" -> ReLU "act" -> Linear(3->600) "out".
/// 600 outputs put the OUTPUT node at/above the margin-subset engagement width.
fn wide_output_net_margin_subset() -> (GraphNetwork, BoundedTensor) {
    let pre = LinearLayer::new(
        arr2(&[[1.0_f32, -0.5], [0.25, 0.75], [-0.6, 0.4]]),
        Some(arr1(&[0.05_f32, -0.1, 0.02])),
    )
    .expect("pre");
    let weights = ndarray::Array2::from_shape_fn((600, 3), |(i, j)| {
        let v = ((i * 7 + j * 13) % 11) as f32 / 11.0 - 0.5;
        if v == 0.0 {
            0.3
        } else {
            v
        }
    });
    let out = LinearLayer::new(weights, None).expect("out");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("pre", Layer::Linear(pre)));
    graph.add_node(GraphNode::new(
        "act",
        Layer::ReLU(ReLULayer),
        vec!["pre".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(out),
        vec!["act".to_string()],
    ));
    graph.set_output("out");
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -0.5]).into_dyn(),
        arr1(&[1.0_f32, 0.75]).into_dyn(),
    )
    .expect("input");
    (graph, input)
}

fn run_wide_backward(publish: bool) -> (BoundedTensor, HashMap<String, BoundedTensor>) {
    let (graph, input) = wide_output_net_margin_subset();
    let ibp_bounds = graph.collect_node_bounds(&input).expect("IBP forward");
    let exec_order = graph.exec_order().expect("exec_order").to_vec();
    let relu_pre = graph
        .relu_preactivation_bounds("act", &input, &ibp_bounds, "margin-subset-test")
        .expect("ReLU pre-activation bounds");
    let mut alpha_state = GraphAlphaState::new();
    alpha_state
        .add_relu_node("act", relu_pre, false)
        .expect("alpha init");
    let relu_name_to_idx: HashMap<String, usize> = [("act".to_string(), 0)].into_iter().collect();
    let mut gradients = vec![Array1::<f32>::zeros(3)];
    let mut gradients_upper = vec![Array1::<f32>::zeros(3)];
    let _guard =
        publish.then(|| crate::output_margin_seed::MarginOutputSeedGuard::publish(vec![5, 200]));
    let bounds = graph
        .dag_alpha_backward_pass_with_engine(
            &input,
            &ibp_bounds,
            &exec_order,
            600,
            2,
            &relu_name_to_idx,
            &alpha_state,
            None,
            &mut gradients,
            &mut gradients_upper,
            None,
            None,
            None,
            None,
        )
        .expect("dag alpha backward");
    (bounds, ibp_bounds)
}

/// With published indices the DAG alpha backward's referenced rows are
/// BIT-IDENTICAL to the full-width backward (row-independence); unreferenced
/// rows keep the output node's sound reference (IBP) enclosure; and the
/// returned tensor is always full output width.
#[ntest::timeout(30000)]
#[test]
fn dag_alpha_backward_scatters_published_margin_rows() {
    let (full, _) = run_wide_backward(false);
    let (subset, ibp_bounds) = run_wide_backward(true);
    let out_ibp = ibp_bounds.get("out").expect("ibp out");

    assert_eq!(subset.shape(), full.shape(), "full output width preserved");
    for i in 0..600 {
        if i == 5 || i == 200 {
            assert_eq!(
                subset.lower()[[i]],
                full.lower()[[i]],
                "referenced lower row {i} must match the full-width backward"
            );
            assert_eq!(
                subset.upper()[[i]],
                full.upper()[[i]],
                "referenced upper row {i} must match the full-width backward"
            );
        } else {
            assert_eq!(
                subset.lower()[[i]],
                out_ibp.lower()[[i]],
                "unreferenced lower row {i} must keep the reference enclosure"
            );
            assert_eq!(
                subset.upper()[[i]],
                out_ibp.upper()[[i]],
                "unreferenced upper row {i} must keep the reference enclosure"
            );
        }
    }
    // Meaningfulness guard: the backward actually tightens a referenced row
    // past IBP (otherwise the equalities above are vacuous).
    assert!(
        [5_usize, 200].iter().any(|&i| {
            full.lower()[[i]] > out_ibp.lower()[[i]] || full.upper()[[i]] < out_ibp.upper()[[i]]
        }),
        "CROWN must beat IBP on a referenced row"
    );
}
