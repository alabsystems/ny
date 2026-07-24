// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::gpu_suffix::{
    take_only_gpu_layer, try_finish_target_gpu_suffix_with_pending_input, GpuSuffixPlan,
};
use super::*;
use crate::bounds::patches::CrownBounds;
use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::network::core::{GraphTargetShapeContract, NETWORK_INPUT};
use crate::network::CrownMergeAccumulator;
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_core::{
    GemmEngine, GpuCrownBackward, GpuCrownLayer, GpuCrownResult, GpuCrownSeed, NaiveCpuGemmEngine,
    Result,
};
use ny_test_utils::assert_bounded_tensor_close;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::*;

struct SeededSuffixScriptedEngine {
    seeded_calls: AtomicUsize,
    expected_lower: Vec<f32>,
    expected_upper: Vec<f32>,
    expected_num_specs: usize,
    expected_current_dim: usize,
    expected_input_lower: Vec<f32>,
    expected_input_upper: Vec<f32>,
    expected_activation_variant: ExpectedActivationVariant,
}

#[derive(Clone, Copy)]
enum ExpectedActivationVariant {
    Either,
    Legacy,
    DualAlpha,
}

impl SeededSuffixScriptedEngine {
    fn new(
        expected_lower: Vec<f32>,
        expected_upper: Vec<f32>,
        expected_num_specs: usize,
        expected_current_dim: usize,
        expected_input_lower: Vec<f32>,
        expected_input_upper: Vec<f32>,
    ) -> Self {
        Self {
            seeded_calls: AtomicUsize::new(0),
            expected_lower,
            expected_upper,
            expected_num_specs,
            expected_current_dim,
            expected_input_lower,
            expected_input_upper,
            expected_activation_variant: ExpectedActivationVariant::Either,
        }
    }

    fn seeded_calls(&self) -> usize {
        self.seeded_calls.load(Ordering::SeqCst)
    }

    fn with_dual_alpha_expectation(mut self) -> Self {
        self.expected_activation_variant = ExpectedActivationVariant::DualAlpha;
        self
    }

    fn with_legacy_activation_expectation(mut self) -> Self {
        self.expected_activation_variant = ExpectedActivationVariant::Legacy;
        self
    }
}

fn gpu_layer_kinds(layers: &[GpuCrownLayer]) -> Vec<&'static str> {
    layers
        .iter()
        .map(|layer| match layer {
            GpuCrownLayer::Linear { .. } => "Linear",
            GpuCrownLayer::Activation { .. } | GpuCrownLayer::ActivationReluDualAlpha { .. } => {
                "Activation"
            }
            GpuCrownLayer::MaxPool2d { .. } => "MaxPool2d",
            GpuCrownLayer::Conv2d { .. } => "Conv2d",
        })
        .collect()
}

impl GemmEngine for SeededSuffixScriptedEngine {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
    }

    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}

impl GpuCrownBackward for SeededSuffixScriptedEngine {
    fn crown_backward_gpu(
        &self,
        _layers: &[GpuCrownLayer],
        _spec: &[f32],
        _num_specs: usize,
        _input_lower: &[f32],
        _input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        panic!("unexpected full GPU CROWN dispatch in seeded-suffix regression");
    }

    fn crown_backward_gpu_seeded(
        &self,
        layers: &[GpuCrownLayer],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        let call_idx = self.seeded_calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(
            call_idx,
            0,
            "seeded GPU suffix should run exactly once, got call #{}",
            call_idx + 1
        );
        // With alpha-aware GPU suffix (#4312), the ReLU node is extracted as
        // an Activation layer. Exact layer sequence varies by graph topology
        // (diamond vs skip-merge), so validate Activation presence, not order.
        assert!(
            gpu_layer_kinds(layers).contains(&"Activation"),
            "seeded GPU suffix should include alpha-aware ReLU Activation (#4312), got: {:?}",
            gpu_layer_kinds(layers),
        );
        match self.expected_activation_variant {
            ExpectedActivationVariant::Either => {}
            ExpectedActivationVariant::Legacy => {
                assert!(
                    layers
                        .iter()
                        .any(|layer| matches!(layer, GpuCrownLayer::Activation { .. })),
                    "seeded GPU suffix should preserve the legacy Activation fast path when alpha_lower == alpha_upper, got: {:?}",
                    gpu_layer_kinds(layers),
                );
                assert!(
                    !layers
                        .iter()
                        .any(|layer| matches!(layer, GpuCrownLayer::ActivationReluDualAlpha { .. })),
                    "seeded GPU suffix should not emit ActivationReluDualAlpha when alpha_lower == alpha_upper, got: {:?}",
                    gpu_layer_kinds(layers),
                );
            }
            ExpectedActivationVariant::DualAlpha => {
                assert!(
                    layers
                        .iter()
                        .any(|layer| matches!(layer, GpuCrownLayer::ActivationReluDualAlpha { .. })),
                    "seeded GPU suffix should emit ActivationReluDualAlpha when alpha_upper != alpha_lower, got: {:?}",
                    gpu_layer_kinds(layers),
                );
            }
        }
        assert_eq!(
            seed.num_specs, self.expected_num_specs,
            "seeded GPU suffix num_specs mismatch"
        );
        assert_eq!(
            seed.current_dim, self.expected_current_dim,
            "seeded GPU suffix current_dim mismatch"
        );
        assert!(
            seed.lower_a.iter().all(|value| value.is_finite())
                && seed.upper_a.iter().all(|value| value.is_finite())
                && seed.lower_b.iter().all(|value| value.is_finite())
                && seed.upper_b.iter().all(|value| value.is_finite()),
            "seeded GPU suffix should receive finite linear bounds"
        );
        assert_eq!(
            input_lower, self.expected_input_lower,
            "seeded GPU suffix input lower mismatch"
        );
        assert_eq!(
            input_upper, self.expected_input_upper,
            "seeded GPU suffix input upper mismatch"
        );

        Ok(GpuCrownResult {
            lower_bounds: self.expected_lower.clone(),
            upper_bounds: self.expected_upper.clone(),
        })
    }
}

fn init_relu_alpha_state(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    ibp_bounds: &HashMap<String, BoundedTensor>,
    relu_name: &str,
    context: &str,
) -> GraphAlphaState {
    let relu_pre = graph
        .relu_preactivation_bounds(relu_name, input, ibp_bounds, context)
        .expect("ReLU pre-activation bounds should exist");
    let mut alpha_state = GraphAlphaState::new();
    alpha_state
        .add_relu_node(relu_name, relu_pre, false)
        .expect("ReLU alpha state should initialize");
    alpha_state
}

fn set_divergent_relu_alpha_state(
    alpha_state: &mut GraphAlphaState,
    relu_name: &str,
    context: &str,
) {
    let unstable_indices: Vec<usize> = alpha_state
        .relu_unstable_mask(relu_name)
        .expect("ReLU unstable mask should exist")
        .iter()
        .enumerate()
        .filter_map(|(idx, unstable)| unstable.then_some(idx))
        .collect();
    assert!(
        !unstable_indices.is_empty(),
        "{context}: fixture must include at least one unstable ReLU neuron"
    );

    let (alpha_lower, alpha_upper) = alpha_state
        .relu_alpha_pair_mut(relu_name)
        .expect("ReLU alpha pair should exist");
    for &idx in &unstable_indices {
        alpha_lower[idx] = 0.15;
        alpha_upper[idx] = 0.85;
    }
    assert!(
        unstable_indices
            .iter()
            .any(|&idx| (alpha_lower[idx] - alpha_upper[idx]).abs() > 1e-6),
        "{context}: divergent test setup must set alpha_upper != alpha_lower"
    );
}

fn init_divergent_relu_alpha_state(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    ibp_bounds: &HashMap<String, BoundedTensor>,
    relu_name: &str,
    context: &str,
) -> GraphAlphaState {
    let mut alpha_state = init_relu_alpha_state(graph, input, ibp_bounds, relu_name, context);
    set_divergent_relu_alpha_state(&mut alpha_state, relu_name, context);
    alpha_state
}

fn assert_bounds_change_from_equal_alpha(
    divergent_bounds: &BoundedTensor,
    equal_bounds: &BoundedTensor,
    context: &str,
) {
    assert!(
        divergent_bounds
            .lower()
            .iter()
            .zip(equal_bounds.lower().iter())
            .chain(
                divergent_bounds
                    .upper()
                    .iter()
                    .zip(equal_bounds.upper().iter())
            )
            .any(|(divergent, equal)| (divergent - equal).abs() > 1e-6),
        "{context}: divergent alpha_upper must change the CPU baseline bounds"
    );
}

fn seeded_suffix_engine_for_bounds(
    bounds: &BoundedTensor,
    input: &BoundedTensor,
) -> SeededSuffixScriptedEngine {
    SeededSuffixScriptedEngine::new(
        bounds.lower().iter().copied().collect(),
        bounds.upper().iter().copied().collect(),
        bounds.len(),
        input.len(),
        input.lower().iter().copied().collect(),
        input.upper().iter().copied().collect(),
    )
    .with_dual_alpha_expectation()
}

fn legacy_seeded_suffix_engine_for_bounds(
    bounds: &BoundedTensor,
    input: &BoundedTensor,
) -> SeededSuffixScriptedEngine {
    SeededSuffixScriptedEngine::new(
        bounds.lower().iter().copied().collect(),
        bounds.upper().iter().copied().collect(),
        bounds.len(),
        input.len(),
        input.lower().iter().copied().collect(),
        input.upper().iter().copied().collect(),
    )
    .with_legacy_activation_expectation()
}

fn propagate_alpha_target(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    target: &str,
    ibp_bounds: &HashMap<String, BoundedTensor>,
    alpha_state: &GraphAlphaState,
    engine: Option<&dyn GemmEngine>,
) -> BoundedTensor {
    graph
        .propagate_crown_to_node_with_alpha(
            input,
            target,
            &HashMap::new(),
            ibp_bounds,
            alpha_state,
            engine,
            None,
        )
        .unwrap_or_else(|error| panic!("alpha target backward for '{target}' failed: {error}"))
}

fn build_alpha_suffix_diamond_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0]]);
    let b1 = arr1(&[0.1_f32, -0.1]);
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("valid Linear1")),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let w2a = arr2(&[[0.8_f32, -0.3], [-0.2, 0.6]]);
    graph.add_node(GraphNode::new(
        "linear2a",
        Layer::Linear(LinearLayer::new(w2a, None).expect("valid Linear2a")),
        vec!["relu1".to_string()],
    ));

    let w2b = arr2(&[[-0.4_f32, 0.7], [0.5, -0.1]]);
    graph.add_node(GraphNode::new(
        "linear2b",
        Layer::Linear(LinearLayer::new(w2b, None).expect("valid Linear2b")),
        vec!["relu1".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "add",
        Layer::Add(AddLayer),
        vec!["linear2a".to_string(), "linear2b".to_string()],
    ));
    graph.set_output("add");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -0.5_f32),
        ArrayD::from_elem(IxDyn(&[2]), 0.5_f32),
    )
    .expect("valid input bounds");

    (graph, input)
}

fn build_alpha_skip_merge_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    let trunk_w = arr2(&[[1.0_f32, -0.6], [0.4, 0.9]]);
    let trunk_b = arr1(&[0.1_f32, -0.15]);
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(trunk_w, Some(trunk_b)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let out_w = arr2(&[[0.7_f32, -0.3], [-0.2, 0.8]]);
    let out_b = arr1(&[0.05_f32, -0.02]);
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(out_w, Some(out_b)).unwrap()),
        vec!["relu1".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "merge",
        Layer::Add(AddLayer),
        vec![NETWORK_INPUT.to_string(), "linear2".to_string()],
    ));
    graph.set_output("merge");

    let input = BoundedTensor::new(
        arr1(&[-0.75_f32, -0.5]).into_dyn(),
        arr1(&[0.9_f32, 0.8]).into_dyn(),
    )
    .unwrap();

    (graph, input)
}

#[ntest::timeout(10000)]
#[test]
fn test_propagate_crown_to_node_with_alpha_uses_seeded_gpu_suffix_linear_tail_4023() {
    // Dispatch/routing test with a FAST (non-sound) mock engine: engage the gate to
    // OFF so the fast seeded GPU suffix is exercised (the production default is now
    // sound, which would mask this mock). #gpu-crown-sound-default.
    let _gate = sound_gpu_gate::test_lock::lock_gate();
    let (graph, input) = build_alpha_suffix_diamond_graph();
    let ibp_bounds = graph
        .collect_node_bounds(&input)
        .expect("IBP bounds should succeed");
    let alpha_state =
        init_relu_alpha_state(&graph, &input, &ibp_bounds, "relu1", "#4023 alpha init");

    let baseline = propagate_alpha_target(&graph, &input, "add", &ibp_bounds, &alpha_state, None);
    let engine = SeededSuffixScriptedEngine::new(
        baseline.lower().iter().copied().collect(),
        baseline.upper().iter().copied().collect(),
        baseline.len(),
        input.len(),
        input.lower().iter().copied().collect(),
        input.upper().iter().copied().collect(),
    );
    let with_engine = propagate_alpha_target(
        &graph,
        &input,
        "add",
        &ibp_bounds,
        &alpha_state,
        Some(&engine),
    );

    assert_bounded_tensor_close(
        &with_engine,
        &baseline,
        1e-6,
        "#4023 alpha target seeded GPU suffix parity",
    );
    assert_eq!(
        engine.seeded_calls(),
        1,
        "#4023 regression: alpha target backward should use one seeded GPU suffix call"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_propagate_crown_to_node_with_alpha_merges_pending_input_after_seeded_gpu_suffix_4023() {
    // Fast-mock dispatch test — gate OFF so the fast seeded GPU suffix runs (the
    // production default is now sound). #gpu-crown-sound-default.
    let _gate = sound_gpu_gate::test_lock::lock_gate();
    let (graph, input) = build_alpha_skip_merge_graph();
    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
    let alpha_state = init_relu_alpha_state(
        &graph,
        &input,
        &ibp_bounds,
        "relu1",
        "#4023 alpha skip merge",
    );

    let branch_only =
        propagate_alpha_target(&graph, &input, "linear2", &ibp_bounds, &alpha_state, None);
    let baseline_full =
        propagate_alpha_target(&graph, &input, "merge", &ibp_bounds, &alpha_state, None);
    let expected_full = BoundedTensor::new(
        branch_only.lower() + input.lower(),
        branch_only.upper() + input.upper(),
    )
    .expect("branch + skip contribution should build a valid BoundedTensor");
    assert_bounded_tensor_close(
        &baseline_full,
        &expected_full,
        1e-6,
        "#4023 baseline skip-merge decomposition",
    );
    assert!(
        baseline_full
            .lower()
            .iter()
            .zip(branch_only.lower().iter())
            .chain(baseline_full.upper().iter().zip(branch_only.upper().iter()))
            .any(|(merged, branch)| (merged - branch).abs() > 1e-6),
        "#4023 test fixture invalid: pending NETWORK_INPUT contribution should change final bounds"
    );

    let engine = SeededSuffixScriptedEngine::new(
        branch_only.lower().iter().copied().collect(),
        branch_only.upper().iter().copied().collect(),
        branch_only.len(),
        input.len(),
        input.lower().iter().copied().collect(),
        input.upper().iter().copied().collect(),
    );
    let with_engine = propagate_alpha_target(
        &graph,
        &input,
        "merge",
        &ibp_bounds,
        &alpha_state,
        Some(&engine),
    );

    assert_bounded_tensor_close(
        &with_engine,
        &baseline_full,
        1e-6,
        "#4023 alpha seeded GPU suffix pending-input merge parity",
    );
    assert_eq!(
        engine.seeded_calls(),
        1,
        "#4023 regression: skip-merge alpha target should use one seeded GPU suffix call"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_propagate_crown_to_node_with_equal_alpha_keeps_legacy_activation_seeded_gpu_suffix_4313() {
    // Fast-mock dispatch test — gate OFF so the fast seeded GPU suffix runs (the
    // production default is now sound). #gpu-crown-sound-default.
    let _gate = sound_gpu_gate::test_lock::lock_gate();
    let (graph, input) = build_alpha_suffix_diamond_graph();
    let ibp_bounds = graph
        .collect_node_bounds(&input)
        .expect("IBP bounds should succeed");
    let equal_alpha_state = init_relu_alpha_state(
        &graph,
        &input,
        &ibp_bounds,
        "relu1",
        "#4313 equal alpha init",
    );

    let baseline =
        propagate_alpha_target(&graph, &input, "add", &ibp_bounds, &equal_alpha_state, None);
    let engine = legacy_seeded_suffix_engine_for_bounds(&baseline, &input);
    let with_engine = propagate_alpha_target(
        &graph,
        &input,
        "add",
        &ibp_bounds,
        &equal_alpha_state,
        Some(&engine),
    );

    assert_bounded_tensor_close(
        &with_engine,
        &baseline,
        1e-6,
        "#4313 equal alpha seeded GPU suffix legacy fast-path parity",
    );
    assert_eq!(
        engine.seeded_calls(),
        1,
        "#4313 regression: equal alpha target should use one seeded GPU suffix call"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_propagate_crown_to_node_with_divergent_alpha_uses_dual_alpha_seeded_gpu_suffix_4313() {
    // Fast-mock dispatch test — gate OFF so the fast seeded GPU suffix runs (the
    // production default is now sound). #gpu-crown-sound-default.
    let _gate = sound_gpu_gate::test_lock::lock_gate();
    let (graph, input) = build_alpha_suffix_diamond_graph();
    let ibp_bounds = graph
        .collect_node_bounds(&input)
        .expect("IBP bounds should succeed");

    let equal_alpha_state = init_relu_alpha_state(
        &graph,
        &input,
        &ibp_bounds,
        "relu1",
        "#4313 equal alpha init",
    );
    let equal_alpha_bounds =
        propagate_alpha_target(&graph, &input, "add", &ibp_bounds, &equal_alpha_state, None);

    let divergent_alpha_state = init_divergent_relu_alpha_state(
        &graph,
        &input,
        &ibp_bounds,
        "relu1",
        "#4313 divergent alpha init",
    );

    let baseline = propagate_alpha_target(
        &graph,
        &input,
        "add",
        &ibp_bounds,
        &divergent_alpha_state,
        None,
    );
    assert_bounds_change_from_equal_alpha(&baseline, &equal_alpha_bounds, "#4313 fixture invalid");

    let engine = seeded_suffix_engine_for_bounds(&baseline, &input);
    let with_engine = propagate_alpha_target(
        &graph,
        &input,
        "add",
        &ibp_bounds,
        &divergent_alpha_state,
        Some(&engine),
    );

    assert_bounded_tensor_close(
        &with_engine,
        &baseline,
        1e-6,
        "#4313 divergent alpha seeded GPU suffix parity",
    );
    assert_eq!(
        engine.seeded_calls(),
        1,
        "#4313 regression: divergent alpha target should use one seeded GPU suffix call"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_try_finish_target_gpu_suffix_with_pending_input_skips_when_other_nodes_remain_4023() {
    let (graph, input) = build_alpha_skip_merge_graph();
    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
    let alpha_state = init_relu_alpha_state(
        &graph,
        &input,
        &ibp_bounds,
        "relu1",
        "#4023 alpha skip pending guard",
    );
    let target_contract = GraphTargetShapeContract::from_bounds(
        "merge",
        ibp_bounds.get("merge").expect("merge bounds should exist"),
    );
    let mut node_crown_bounds = CrownMergeAccumulator::new();
    node_crown_bounds.insert(
        NETWORK_INPUT.to_string(),
        CrownBounds::Dense(LinearBounds::identity(input.len())),
    );
    node_crown_bounds.insert(
        "sibling_pending".to_string(),
        CrownBounds::Dense(LinearBounds::identity(input.len())),
    );
    let node_lb = LinearBounds::identity(input.len());
    let engine = SeededSuffixScriptedEngine::new(
        vec![0.0; input.len()],
        vec![0.0; input.len()],
        input.len(),
        input.len(),
        input.lower().iter().copied().collect(),
        input.upper().iter().copied().collect(),
    );

    let plan = GpuSuffixPlan::build(
        &graph.ancestors("merge").expect("ancestors should succeed"),
        &graph,
        &input,
        &HashMap::new(),
        &ibp_bounds,
        Some(&alpha_state),
    );

    let result = try_finish_target_gpu_suffix_with_pending_input(
        &input,
        "linear2",
        &node_lb,
        &plan,
        Some(&engine),
        &target_contract,
        &mut node_crown_bounds,
    )
    .expect("other pending nodes should force CPU fallback without error");

    assert!(
        result.is_none(),
        "#4023 regression: seeded GPU suffix must not finish while other pending nodes remain"
    );
    assert_eq!(
        engine.seeded_calls(),
        0,
        "#4023 regression: guard should block any seeded GPU suffix dispatch"
    );
    assert!(
        node_crown_bounds.contains_key(NETWORK_INPUT)
            && node_crown_bounds.contains_key("sibling_pending"),
        "#4023 regression: fallback guard must preserve all pending contributions"
    );
}

// ---------------------------------------------------------------------------
// Packet C (#4340): plan-shape coverage
// ---------------------------------------------------------------------------

#[test]
fn test_gpu_suffix_plan_eligible_unary_tail_contains_expected_nodes_4340() {
    // Diamond graph: input -> linear1 -> relu1 -> {linear2a, linear2b} -> add
    // The unary tail from linear1 through relu1 to linear2a (or linear2b) is
    // eligible for GPU suffix. The plan should contain linear1, relu1, and the
    // linear branches — but NOT the add (binary merge).
    let (graph, input) = build_alpha_suffix_diamond_graph();
    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
    let alpha_state =
        init_relu_alpha_state(&graph, &input, &ibp_bounds, "relu1", "#4340 plan eligible");

    let relevant_nodes = graph.ancestors("add").unwrap();
    let plan = GpuSuffixPlan::build(
        &relevant_nodes,
        &graph,
        &input,
        &HashMap::new(),
        &ibp_bounds,
        Some(&alpha_state),
    );

    // linear1 is the first unary layer from NETWORK_INPUT — should be eligible.
    assert!(
        plan.contains("linear1"),
        "#4340: linear1 should be in the GPU suffix plan"
    );
    // relu1 has linear1 as its input which is in the plan — should be eligible.
    assert!(
        plan.contains("relu1"),
        "#4340: relu1 should be in the GPU suffix plan"
    );
    // linear2a and linear2b are both unary with relu1 as input — should be eligible.
    assert!(
        plan.contains("linear2a"),
        "#4340: linear2a should be in the GPU suffix plan"
    );
    assert!(
        plan.contains("linear2b"),
        "#4340: linear2b should be in the GPU suffix plan"
    );
    // add is binary (two inputs) — should NOT be eligible.
    assert!(
        !plan.contains("add"),
        "#4340: binary merge node 'add' should NOT be in the GPU suffix plan"
    );

    // Materialize from linear2a should produce layers for linear2a -> relu1 -> linear1.
    let layers = plan
        .materialize_suffix("linear2a")
        .expect("#4340: materialize from linear2a should succeed");
    assert_eq!(
        layers.len(),
        3,
        "#4340: suffix from linear2a should have 3 layers (linear2a, relu1, linear1)"
    );

    // Materialize from add should fail since add is not in the plan.
    assert!(
        plan.materialize_suffix("add").is_none(),
        "#4340: materialize from binary merge should be None"
    );
}

#[test]
fn test_gpu_suffix_plan_ineligible_merge_blocks_downstream_4340() {
    // Skip-merge graph: input -> linear1 -> relu1 -> linear2 -> merge(input, linear2)
    // The merge node is binary so it's NOT eligible. But linear1, relu1, linear2
    // are all unary and should be in the plan. However, the merge being at the
    // output means the backward loop would only probe from node names that ARE
    // in relevant_nodes. The key thing: looking up "merge" in the plan is O(1) miss.
    let (graph, input) = build_alpha_skip_merge_graph();
    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
    let alpha_state = init_relu_alpha_state(
        &graph,
        &input,
        &ibp_bounds,
        "relu1",
        "#4340 plan ineligible",
    );

    let relevant_nodes = graph.ancestors("merge").unwrap();
    let plan = GpuSuffixPlan::build(
        &relevant_nodes,
        &graph,
        &input,
        &HashMap::new(),
        &ibp_bounds,
        Some(&alpha_state),
    );

    // merge is binary — should NOT be in the plan.
    assert!(
        !plan.contains("merge"),
        "#4340: binary merge node should NOT be in the GPU suffix plan"
    );

    // The unary tail linear1 -> relu1 -> linear2 SHOULD all be eligible.
    assert!(
        plan.contains("linear1"),
        "#4340: linear1 in skip-merge should be eligible"
    );
    assert!(
        plan.contains("relu1"),
        "#4340: relu1 in skip-merge should be eligible"
    );
    assert!(
        plan.contains("linear2"),
        "#4340: linear2 in skip-merge should be eligible"
    );

    // Materialize from linear2 should produce 3 layers (linear2, relu1, linear1).
    let layers = plan
        .materialize_suffix("linear2")
        .expect("#4340: materialize from linear2 should succeed");
    assert_eq!(
        layers.len(),
        3,
        "#4340: suffix from linear2 should span 3 unary layers"
    );
}

#[test]
fn test_add_concrete_bounds_accepts_inf_inputs_4369() {
    use super::gpu_suffix::add_concrete_bounds;

    // GPU suffix result with Inf lower (from new_repaired(Widen))
    let lhs = BoundedTensor::new_allow_infinite(
        ArrayD::from_elem(IxDyn(&[3]), f32::NEG_INFINITY),
        ArrayD::from_elem(IxDyn(&[3]), 1.0_f32),
    )
    .expect("lhs with -Inf lower");

    // Pending input contribution with finite values
    let rhs = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[3]), -0.5_f32),
        ArrayD::from_elem(IxDyn(&[3]), 0.5_f32),
    )
    .expect("finite rhs");

    let result = add_concrete_bounds(lhs, &rhs, "test #4369 Inf repair");
    assert!(
        result.is_ok(),
        "#4369 regression: add_concrete_bounds must not abort on Inf inputs, got: {:?}",
        result.err()
    );
    let bt = result.unwrap();
    // -Inf + finite = -Inf stays -Inf: no proven lower bound, so no finite
    // substitute is sound. The finite upper sum passes through untouched.
    assert!(
        bt.lower().iter().all(|&v| v == f32::NEG_INFINITY),
        "#4369: -Inf lower plus finite must stay -Inf"
    );
    assert!(
        bt.upper().iter().all(|&v| (v - 1.5).abs() < 1e-6),
        "#4369: finite upper sum must stay 1.5"
    );
}

#[test]
fn test_add_concrete_bounds_repairs_nan_from_inf_cancellation_4369() {
    use super::gpu_suffix::add_concrete_bounds;

    // lhs.upper = +Inf, rhs.upper = -Inf → upper_sum = Inf + (-Inf) = NaN
    // lhs.lower = +Inf, rhs.lower = -Inf → lower_sum = Inf + (-Inf) = NaN
    // This exercises the actual NaN-from-cancellation path.
    let lhs = BoundedTensor::new_allow_infinite(
        ArrayD::from_elem(IxDyn(&[2]), f32::INFINITY),
        ArrayD::from_elem(IxDyn(&[2]), f32::INFINITY),
    )
    .expect("lhs with +Inf bounds");

    let rhs = BoundedTensor::new_allow_infinite(
        ArrayD::from_elem(IxDyn(&[2]), f32::NEG_INFINITY),
        ArrayD::from_elem(IxDyn(&[2]), f32::NEG_INFINITY),
    )
    .expect("rhs with -Inf bounds");

    // lower: Inf + (-Inf) = NaN, upper: Inf + (-Inf) = NaN
    // Conservative strategy widens NaN to -inf (lower) / +inf (upper): a NaN
    // sum proves nothing, so the repair must not fabricate a finite bound.
    let result = add_concrete_bounds(lhs, &rhs, "test #4369 NaN repair");
    assert!(
        result.is_ok(),
        "#4369 regression: add_concrete_bounds must handle Inf+Inf, got: {:?}",
        result.err()
    );
    let bt = result.unwrap();
    assert!(
        bt.lower().iter().all(|&v| v == f32::NEG_INFINITY),
        "#4369: NaN lower must widen to -Inf"
    );
    assert!(
        bt.upper().iter().all(|&v| v == f32::INFINITY),
        "#4369: NaN upper must widen to +Inf"
    );
}

#[test]
fn test_take_only_gpu_layer_rejects_empty_and_multi_layer_vectors_4411() {
    let single = GpuCrownLayer::Linear {
        weight: Arc::<[f32]>::from(vec![1.0_f32]),
        bias: None,
        out_features: 1,
        in_features: 1,
    };
    assert!(
        take_only_gpu_layer(Vec::new()).is_none(),
        "#4411: empty GPU extraction should be rejected"
    );
    assert!(
        matches!(
            take_only_gpu_layer(vec![single.clone()]),
            Some(GpuCrownLayer::Linear {
                out_features: 1,
                in_features: 1,
                ..
            })
        ),
        "#4411: singleton GPU extraction should succeed"
    );
    assert!(
        take_only_gpu_layer(vec![single.clone(), single]).is_none(),
        "#4411: multi-layer GPU extraction should be rejected instead of panicking"
    );
}
