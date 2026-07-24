// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Constrained patches regressions (#3813, #4138).

use ndarray::{arr1, ArrayD, IxDyn};
use ny_test_utils::{assert_bounded_tensor_close, CountingGemmEngine};
use std::collections::HashMap;

use crate::beta_crown::state::{
    AlphaNeuronState, GraphBetaEntry, GraphBetaState, GraphDomainAlphaState,
};
use crate::beta_crown::{GraphCrownContext, GraphNeuronConstraint, GraphSplitHistory};
use crate::bounds::patches::{patches_to_dense_call_sites, reset_patches_to_dense_call_count};
use crate::bounds::GraphAlphaCrownIntermediate;
use crate::layers::Conv2dLayer;
use crate::{
    BetaCrownConfig, BetaCrownVerifier, BoundedTensor, GraphNetwork, GraphNode, Layer, ReLULayer,
};

use super::super::backward::{BackwardCrownResult, BackwardMode, BackwardParams};
use super::super::lookups::build_constraint_lookups;
use super::super::patches::ConstrainedPatchesPolicy;
pub(super) fn build_two_conv_relu_graph_3813() -> GraphNetwork {
    let conv1_kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.45_f32, -0.2, 0.7, 0.35])
            .expect("valid conv1 kernel");
    let conv1 =
        Conv2dLayer::with_input_shape(conv1_kernel, Some(arr1(&[0.05_f32])), (1, 1), (0, 0), 4, 4)
            .expect("valid conv1");

    let conv2_kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.3_f32, 0.15, -0.4, 0.25])
            .expect("valid conv2 kernel");
    let conv2 =
        Conv2dLayer::with_input_shape(conv2_kernel, Some(arr1(&[-0.03_f32])), (1, 1), (0, 0), 3, 3)
            .expect("valid conv2");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["conv1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "conv2",
        Layer::Conv2d(conv2),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["conv2".to_string()],
    ));
    graph.set_output("relu2");
    graph.set_use_patches_mode(false);
    graph
}

pub(super) fn build_two_conv_relu_input_3813() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), -0.25_f32),
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.6_f32),
    )
    .expect("valid two-conv input")
}

fn build_large_two_conv_relu_graph_4138() -> GraphNetwork {
    let conv1_kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.45_f32, -0.2, 0.7, 0.35])
            .expect("valid conv1 kernel");
    let conv1 = Conv2dLayer::with_input_shape(
        conv1_kernel,
        Some(arr1(&[0.05_f32])),
        (1, 1),
        (0, 0),
        11,
        11,
    )
    .expect("valid conv1");

    let conv2_kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.3_f32, 0.15, -0.4, 0.25])
            .expect("valid conv2 kernel");
    let conv2 = Conv2dLayer::with_input_shape(
        conv2_kernel,
        Some(arr1(&[-0.03_f32])),
        (1, 1),
        (0, 0),
        10,
        10,
    )
    .expect("valid conv2");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["conv1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "conv2",
        Layer::Conv2d(conv2),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["conv2".to_string()],
    ));
    graph.set_output("relu2");
    graph.set_use_patches_mode(false);
    graph
}

fn build_large_two_conv_relu_input_4138() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 11, 11]), -0.25_f32),
        ArrayD::from_elem(IxDyn(&[1, 11, 11]), 0.6_f32),
    )
    .expect("valid large two-conv input")
}

fn constrained_patches_call_sites_4138(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    context: &GraphCrownContext<'_>,
    beta_state: Option<&GraphBetaState>,
) -> Vec<String> {
    reset_patches_to_dense_call_count();
    let (_result, _) = run_constrained_backward_with_policy(
        verifier,
        graph,
        input,
        context,
        beta_state,
        None,
        BackwardMode::Standard,
        ConstrainedPatchesPolicy::selective_matrix_reentry(),
    );
    patches_to_dense_call_sites()
}

// Justification: this test helper threads verifier state, graph/input/context,
// optional beta/objective configuration, backward mode, and the patches policy.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_constrained_backward_with_policy(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    context: &GraphCrownContext<'_>,
    beta_state: Option<&GraphBetaState>,
    objective: Option<&[f32]>,
    mode: BackwardMode,
    patches_policy: ConstrainedPatchesPolicy,
) -> (
    BackwardCrownResult,
    HashMap<String, std::sync::Arc<BoundedTensor>>,
) {
    let (mut bounds_cache, constrained_input, exec_order) = verifier
        .prepare_constrained_graph_bounds(graph, input, context, beta_state, objective)
        .expect("constrained forward bounds should succeed");
    let params = BackwardParams {
        graph,
        constrained_input: &constrained_input,
        exec_order: &exec_order,
        context,
        beta_state,
        objective,
        spec_matrix: None,
        seed_cache: None,
        capture_linear_bounds: false,
        deadline: verifier.config.alpha_config.deadline,
        patches_policy,
    };
    let result = verifier
        .backward_crown_constrained(&params, &mut bounds_cache, mode)
        .expect("constrained backward should succeed");
    (result, bounds_cache)
}

pub(super) fn storing_intermediates_mode_3813(
    graph: &GraphNetwork,
    history: &GraphSplitHistory,
) -> BackwardMode {
    let lookups =
        build_constraint_lookups(&history.constraints, &history.genbab_constraints, graph)
            .expect("constraint lookups should succeed");
    BackwardMode::StoringIntermediates {
        lookups: Box::new(lookups),
    }
}

pub(super) fn assert_storing_intermediate_capture_3813(intermediate: &GraphAlphaCrownIntermediate) {
    let a_at_relu = intermediate
        .a_at_relu
        .get("relu1")
        .expect("relu1 A matrix should be captured");
    assert_eq!(
        a_at_relu.nrows(),
        4,
        "relu1 A matrix row count should match output dim"
    );
    assert_eq!(
        a_at_relu.ncols(),
        9,
        "relu1 A matrix column count should match the 1x3x3 pre-activation"
    );
    assert!(
        a_at_relu.iter().all(|value| value.is_finite()),
        "relu1 A matrix should stay finite after patches re-entry"
    );

    let (lower, upper) = intermediate
        .pre_relu_bounds
        .get("relu1")
        .expect("relu1 pre-ReLU bounds should be captured");
    assert_eq!(
        lower.len(),
        9,
        "relu1 pre-lower should flatten to 9 neurons"
    );
    assert_eq!(
        upper.len(),
        9,
        "relu1 pre-upper should flatten to 9 neurons"
    );
    assert!(
        !intermediate.final_bounds.lower_b.is_empty(),
        "final_bounds should be populated in storing intermediates mode"
    );
}

#[test]
fn test_constrained_patches_matches_dense_baseline_3813() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_two_conv_relu_graph_3813();
    let input = build_two_conv_relu_input_3813();
    let history = GraphSplitHistory::new();
    let context = GraphCrownContext::for_history(&history);

    let (dense_result, _) = run_constrained_backward_with_policy(
        &verifier,
        &graph,
        &input,
        &context,
        None,
        None,
        BackwardMode::Standard,
        ConstrainedPatchesPolicy::dense_only(),
    );
    let (patches_result, _) = run_constrained_backward_with_policy(
        &verifier,
        &graph,
        &input,
        &context,
        None,
        None,
        BackwardMode::Standard,
        ConstrainedPatchesPolicy::selective_matrix_reentry(),
    );

    assert_bounded_tensor_close(
        &dense_result.output_bounds,
        &patches_result.output_bounds,
        1e-5,
        "selective constrained patches parity",
    );
}

#[test]
fn test_constrained_patches_alpha_matches_dense_baseline_3813() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_two_conv_relu_graph_3813();
    let input = build_two_conv_relu_input_3813();
    let history = GraphSplitHistory::new();
    let mut alpha_state = GraphDomainAlphaState::empty();
    alpha_state.insert("relu1".to_string(), 0, AlphaNeuronState::new(0.37));
    alpha_state.insert("relu1".to_string(), 2, AlphaNeuronState::new(0.61));

    let context = GraphCrownContext::for_history(&history).with_alpha(&alpha_state);
    let (dense_result, _) = run_constrained_backward_with_policy(
        &verifier,
        &graph,
        &input,
        &context,
        None,
        None,
        BackwardMode::Standard,
        ConstrainedPatchesPolicy::dense_only(),
    );
    let (patches_result, _) = run_constrained_backward_with_policy(
        &verifier,
        &graph,
        &input,
        &context,
        None,
        None,
        BackwardMode::Standard,
        ConstrainedPatchesPolicy::selective_matrix_reentry(),
    );

    assert_bounded_tensor_close(
        &dense_result.output_bounds,
        &patches_result.output_bounds,
        1e-5,
        "selective constrained patches alpha parity",
    );
}

#[test]
fn test_constrained_patches_beta_matches_dense_baseline_3813() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_two_conv_relu_graph_3813();
    let input = build_two_conv_relu_input_3813();
    let history = GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    let context = GraphCrownContext::for_history(&history);
    let beta_state = GraphBetaState {
        entries: vec![GraphBetaEntry {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            split_point: 0.0,
            value: 0.35,
            sign: 1.0,
            grad: 0.0,
            m: 0.0,
            v: 0.0,
            v_max: 0.0,
        }],
        ..GraphBetaState::empty()
    };

    let (dense_result, _) = run_constrained_backward_with_policy(
        &verifier,
        &graph,
        &input,
        &context,
        Some(&beta_state),
        None,
        BackwardMode::Standard,
        ConstrainedPatchesPolicy::dense_only(),
    );
    let (patches_result, _) = run_constrained_backward_with_policy(
        &verifier,
        &graph,
        &input,
        &context,
        Some(&beta_state),
        None,
        BackwardMode::Standard,
        ConstrainedPatchesPolicy::selective_matrix_reentry(),
    );

    assert_bounded_tensor_close(
        &dense_result.output_bounds,
        &patches_result.output_bounds,
        1e-5,
        "selective constrained patches beta parity",
    );
}

#[test]
fn test_constrained_patches_ignore_unrelated_beta_entries_for_relu_densification_4138() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_large_two_conv_relu_graph_4138();
    let input = build_large_two_conv_relu_input_4138();
    let history = GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    let context = GraphCrownContext::for_history(&history);
    let unrelated_beta_state = GraphBetaState {
        entries: vec![GraphBetaEntry {
            node_name: "relu2".to_string(),
            neuron_idx: 0,
            split_point: 0.0,
            value: 0.35,
            sign: 1.0,
            grad: 0.0,
            m: 0.0,
            v: 0.0,
            v_max: 0.0,
        }],
        ..GraphBetaState::empty()
    };

    // The patches→dense call-site recorder is thread-local (#4138): each test
    // runs on its own thread and the CROWN propagation here is synchronous on
    // that thread, so concurrently running tests cannot contaminate these two
    // recording windows.
    let baseline_call_sites =
        constrained_patches_call_sites_4138(&verifier, &graph, &input, &context, None);
    let unrelated_beta_call_sites = constrained_patches_call_sites_4138(
        &verifier,
        &graph,
        &input,
        &context,
        Some(&unrelated_beta_state),
    );

    assert_eq!(
        unrelated_beta_call_sites,
        baseline_call_sites,
        "beta entries for another node must not densify relu1 patches path: baseline={baseline_call_sites:?} unrelated_beta={unrelated_beta_call_sites:?}",
    );
}

#[test]
fn test_constrained_patches_reduce_gemm_calls_3813() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_two_conv_relu_graph_3813();
    let input = build_two_conv_relu_input_3813();
    let history = GraphSplitHistory::new();

    let dense_engine = CountingGemmEngine::new();
    let dense_context = GraphCrownContext::for_history_and_engine(&history, Some(&dense_engine));
    let (_dense_result, _) = run_constrained_backward_with_policy(
        &verifier,
        &graph,
        &input,
        &dense_context,
        None,
        None,
        BackwardMode::Standard,
        ConstrainedPatchesPolicy::dense_only(),
    );

    let patches_engine = CountingGemmEngine::new();
    let patches_context =
        GraphCrownContext::for_history_and_engine(&history, Some(&patches_engine));
    let (_patches_result, _) = run_constrained_backward_with_policy(
        &verifier,
        &graph,
        &input,
        &patches_context,
        None,
        None,
        BackwardMode::Standard,
        ConstrainedPatchesPolicy::selective_matrix_reentry(),
    );

    // After #3813 row-count threshold: for small models (≤64 objective rows),
    // Patches re-entry is skipped because Dense BLAS backward is faster than
    // per-position Patches composition. Both policies now produce the same
    // GEMM call count. Patches re-entry activates for large-row carriers
    // (e.g., CROWN-IBP collector with thousands of rows).
    assert_eq!(
        patches_engine.gemm_calls(),
        dense_engine.gemm_calls(),
        "#3813 small-model constrained backward stays Dense (≤64 rows): dense={} patches={}",
        dense_engine.gemm_calls(),
        patches_engine.gemm_calls()
    );
}

#[test]
fn test_constrained_patches_storing_intermediates_preserves_dense_relu_storage_3813() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_two_conv_relu_graph_3813();
    let input = build_two_conv_relu_input_3813();
    let history = GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    let context = GraphCrownContext::for_history(&history);

    let (standard_result, _) = run_constrained_backward_with_policy(
        &verifier,
        &graph,
        &input,
        &context,
        None,
        None,
        BackwardMode::Standard,
        ConstrainedPatchesPolicy::selective_matrix_reentry(),
    );
    let (intermediate_result, _) = run_constrained_backward_with_policy(
        &verifier,
        &graph,
        &input,
        &context,
        None,
        None,
        storing_intermediates_mode_3813(&graph, &history),
        ConstrainedPatchesPolicy::selective_matrix_reentry(),
    );

    assert_bounded_tensor_close(
        &standard_result.output_bounds,
        &intermediate_result.output_bounds,
        1e-5,
        "selective constrained patches storing-intermediates parity",
    );

    let intermediate = intermediate_result
        .intermediate
        .expect("storing intermediates mode should populate intermediate state");
    assert_storing_intermediate_capture_3813(&intermediate);
}
