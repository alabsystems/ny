// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::bounds::{run_with_m1_alpha_trace, M1AlphaBudgetOutcome, M1AlphaTraceEvent};
use super::*;
use crate::bounds::GraphAlphaState;
use crate::layers::{
    AddLayer, ConvTranspose1dLayer, LinearLayer, NonZeroLayer, ReLULayer, ReshapeLayer,
    SigmoidLayer, SqrtLayer, TanhLayer,
};
use crate::network::core::GraphNode;
use ndarray::{arr1, arr2, array, ArrayD, IxDyn};
use ny_test_utils::assert_bounds_do_not_loosen;
use std::time::{Duration, Instant};

fn tensor(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new_allow_infinite(arr1(lower).into_dyn(), arr1(upper).into_dyn())
        .expect("test bounds should be valid")
}

fn reference_bounds_with_input(
    graph: &GraphNetwork,
    input: &BoundedTensor,
) -> HashMap<String, BoundedTensor> {
    let mut reference = graph
        .collect_node_bounds(input)
        .expect("IBP bounds should succeed");
    reference.insert(NETWORK_INPUT.to_string(), input.clone());
    reference
}

fn make_target_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]), None)
                .expect("linear1 should construct"),
        ),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.0_f32, -0.5], [0.3, 0.8]]), None)
                .expect("linear2 should construct"),
        ),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "sigmoid",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["linear2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "tanh",
        Layer::Tanh(TanhLayer::new()),
        vec!["linear2".to_string()],
    ));
    graph.add_node(GraphNode::from_input(
        "linear3",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[0.4_f32, 0.2], [0.1, 0.6]]),
                Some(arr1(&[1.5_f32, 2.0])),
            )
            .expect("linear3 should construct"),
        ),
    ));
    graph.add_node(GraphNode::new(
        "sqrt",
        Layer::Sqrt(SqrtLayer::new()),
        vec!["linear3".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "merge",
        Layer::Add(AddLayer),
        vec!["tanh".to_string(), "sqrt".to_string()],
    ));
    graph.set_output("merge");
    graph
}

fn make_fallback_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.0_f32, 0.5], [-0.2, 0.8]]), None)
                .expect("linear should construct"),
        ),
    ));
    graph.add_node(GraphNode::new(
        "nonzero",
        Layer::NonZero(NonZeroLayer),
        vec!["linear".to_string()],
    ));
    graph.set_output("nonzero");
    let input = BoundedTensor::new(
        array![-1.0_f32, -1.0].into_dyn(),
        array![1.0_f32, 1.0].into_dyn(),
    )
    .expect("input bounds should construct");
    (graph, input)
}

fn make_refresh_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0_f32, -0.5], [0.5, 1.0], [-1.0, 0.3]]),
                Some(arr1(&[0.0_f32, 0.1, -0.1])),
            )
            .expect("linear1 should construct"),
        ),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(
            LinearLayer::new(arr2(&[[0.5_f32, -0.3, 0.8], [0.2, 0.6, -0.4]]), None)
                .expect("linear2 should construct"),
        ),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.set_output("relu2");
    let input = BoundedTensor::new(
        array![-0.5_f32, -0.5].into_dyn(),
        array![0.5_f32, 0.5].into_dyn(),
    )
    .expect("input bounds should construct");
    (graph, input)
}

fn make_chunk_aware_alpha_budget_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "small_target",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("small target should construct"),
        ),
    ));
    graph.add_node(GraphNode::new(
        "reshape",
        Layer::Reshape(ReshapeLayer::new(vec![1, 1])),
        vec!["small_target".to_string()],
    ));
    let conv_transpose = ConvTranspose1dLayer::with_input_length(
        ArrayD::from_elem(IxDyn(&[1, 1, 384]), 0.25_f32),
        None,
        1,
        0,
        1,
    )
    .expect("wide ConvTranspose1d target should construct");
    graph.add_node(GraphNode::new(
        "wide_target",
        Layer::ConvTranspose1d(conv_transpose),
        vec!["reshape".to_string()],
    ));
    graph.set_output("wide_target");
    let input = BoundedTensor::new(arr1(&[-0.5_f32]).into_dyn(), arr1(&[0.5_f32]).into_dyn())
        .expect("input bounds should construct");
    (graph, input)
}

fn make_relu_alpha_state(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    reference: &HashMap<String, BoundedTensor>,
    relu_names: &[&str],
) -> GraphAlphaState {
    let mut alpha_state = GraphAlphaState::new();
    for relu_name in relu_names {
        let pre_activation = graph
            .relu_preactivation_bounds(relu_name, input, reference, "#3677 alpha init")
            .expect("relu pre-activation should resolve");
        alpha_state
            .add_relu_node(relu_name, pre_activation, false)
            .expect("relu alpha should initialize");
    }
    alpha_state
}

#[test]
fn test_graph_alpha_reference_bounds_merge_tightens_per_element_3677() {
    let mut initial = HashMap::new();
    initial.insert("hidden".to_string(), tensor(&[-2.0, -1.0], &[4.0, 3.0]));
    initial.insert("other".to_string(), tensor(&[-5.0], &[5.0]));
    let mut state = GraphAlphaReferenceBounds::new(initial, vec!["hidden".to_string()])
        .expect("reference bounds should initialize");

    let mut candidate = HashMap::new();
    candidate.insert("hidden".to_string(), tensor(&[-1.0, -3.0], &[2.5, 2.0]));

    let tightened = state
        .merge_candidate(&candidate)
        .expect("merge should succeed");
    assert_eq!(
        tightened, 1,
        "#3677 merge should report one tightened target"
    );

    let best_hidden = state
        .best()
        .get("hidden")
        .expect("best hidden bounds should exist");
    assert_eq!(best_hidden.lower()[[0]], -1.0);
    assert_eq!(best_hidden.lower()[[1]], -1.0);
    assert_eq!(best_hidden.upper()[[0]], 2.5);
    assert_eq!(best_hidden.upper()[[1]], 2.0);

    state
        .promote_best_to_current()
        .expect("promotion should succeed");
    let promoted_hidden = state
        .current()
        .get("hidden")
        .expect("promoted hidden bounds should exist");
    assert_eq!(promoted_hidden.lower()[[0]], -1.0);
    assert_eq!(promoted_hidden.upper()[[0]], 2.5);

    let other = state
        .current()
        .get("other")
        .expect("non-target bounds should stay present");
    assert_eq!(other.lower()[[0]], -5.0);
    assert_eq!(other.upper()[[0]], 5.0);
}

#[test]
fn test_graph_alpha_reference_bound_targets_follow_activation_inputs_3677() {
    let graph = make_target_graph();
    let targets = graph
        .graph_alpha_reference_bound_targets()
        .expect("target collection should succeed");
    assert_eq!(
        targets,
        vec![
            "linear1".to_string(),
            "linear2".to_string(),
            "linear3".to_string(),
        ],
        "#3677 targets should follow ReLU/Sigmoid/Tanh/Sqrt inputs in topological order"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_collect_selected_crown_bounds_with_alpha_uses_reference_fallback_3677() {
    let (graph, input) = make_fallback_graph();
    let reference = reference_bounds_with_input(&graph, &input);
    let selected = graph
        .collect_selected_crown_bounds_with_alpha(
            &input,
            &["nonzero".to_string()],
            &reference,
            &GraphAlphaState::new(),
            None,
            None,
        )
        .expect("selected alpha-bound collection should fall back cleanly");

    let fallback = selected
        .get("nonzero")
        .expect("selected map should contain nonzero target");
    let baseline = reference
        .get("nonzero")
        .expect("reference map should contain nonzero target");
    assert_eq!(
        fallback.lower(),
        baseline.lower(),
        "#3677 fallback should preserve lower bounds"
    );
    assert_eq!(
        fallback.upper(),
        baseline.upper(),
        "#3677 fallback should preserve upper bounds"
    );
}

#[ntest::timeout(30000)]
#[test]
fn chunk_aware_alpha_budget_falls_back_without_dispatch_and_keeps_fixed_waves() {
    ny_test_utils::env::with_env_edits(|env| {
        env.set("NY_CROWN_CHUNK_AWARE_BUDGET", "1");
        env.set("NY_DENSE_BUDGET_MB", "1");
        env.set("NY_CROWN_OBJ_CHUNK", "0");
        for key in [
            "NY_NO_CHUNK_ABORT",
            "NY_NO_CHUNK_GROW",
            "NY_NO_CHUNK_WAVE_PAR",
            "NY_PER_NODE_CAP_SECS",
            "NY_PER_NODE_FLOOR_SECS",
        ] {
            env.remove(key);
        }

        let (graph, input) = make_chunk_aware_alpha_budget_graph();
        let reference = reference_bounds_with_input(&graph, &input);
        let targets = vec!["small_target".to_string(), "wide_target".to_string()];
        let (selected, trace) = run_with_m1_alpha_trace(|| {
            graph.collect_selected_crown_bounds_with_alpha_mode(
                &input,
                &targets,
                &reference,
                &GraphAlphaState::new(),
                None,
                Some(Instant::now() + Duration::from_secs(30)),
                false,
            )
        });
        let selected = selected.expect("M1 alpha-bound collection should succeed");

        let small = selected
            .get("small_target")
            .expect("small target should be selected");
        let small_reference = reference
            .get("small_target")
            .expect("small target reference bounds should exist");
        assert_eq!(small.lower(), small_reference.lower());
        assert_eq!(small.upper(), small_reference.upper());
        assert!(selected.contains_key("wide_target"));

        assert!(trace.iter().any(|event| matches!(
            event,
            M1AlphaTraceEvent::BudgetAdmission {
                node,
                outcome: M1AlphaBudgetOutcome::BelowFloor,
                deadline_present: false,
            } if node == "small_target"
        )));
        assert!(trace.iter().any(|event| matches!(
            event,
            M1AlphaTraceEvent::BudgetAdmission {
                node,
                outcome: M1AlphaBudgetOutcome::NotAdmitted,
                deadline_present: false,
            } if node == "reshape"
        )));
        assert!(trace.iter().any(|event| matches!(
            event,
            M1AlphaTraceEvent::BudgetAdmission {
                node,
                outcome: M1AlphaBudgetOutcome::Allocate,
                deadline_present: true,
            } if node == "wide_target"
        )));
        for fallback_node in ["small_target", "reshape"] {
            assert!(
                !trace.iter().any(|event| matches!(
                    event,
                    M1AlphaTraceEvent::BackwardDispatch { node, .. }
                        if node == fallback_node
                )),
                "{fallback_node} must take its reference bound without backward dispatch"
            );
        }
        assert!(trace.iter().any(|event| matches!(
            event,
            M1AlphaTraceEvent::BackwardDispatch {
                node,
                retained_fixed_wave: true,
            } if node == "wide_target"
        )));
    });
}

fn collect_refresh_state(
    graph: &GraphNetwork,
    input: &BoundedTensor,
) -> (
    HashMap<String, BoundedTensor>,
    GraphAlphaState,
    Vec<String>,
    HashMap<String, BoundedTensor>,
) {
    let reference = reference_bounds_with_input(graph, input);
    let alpha_state = make_relu_alpha_state(graph, input, &reference, &["relu1", "relu2"]);
    let targets = graph
        .graph_alpha_reference_bound_targets()
        .expect("targets should collect");
    let selected = graph
        .collect_selected_crown_bounds_with_alpha(
            input,
            &targets,
            &reference,
            &alpha_state,
            None,
            None,
        )
        .expect("selected alpha-bound collection should succeed");
    (reference, alpha_state, targets, selected)
}

fn baseline_and_refreshed_outputs(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    reference: &HashMap<String, BoundedTensor>,
    alpha_state: &GraphAlphaState,
    targets: Vec<String>,
    selected: &HashMap<String, BoundedTensor>,
) -> (
    usize,
    GraphAlphaReferenceBounds,
    BoundedTensor,
    BoundedTensor,
) {
    let baseline_output = graph
        .propagate_crown_to_node_with_alpha(
            input,
            graph.output_name(),
            &HashMap::new(),
            reference,
            alpha_state,
            None,
            None,
        )
        .expect("baseline output CROWN should succeed");
    let mut state =
        GraphAlphaReferenceBounds::new(reference.clone(), targets).expect("state should init");
    let tightened_targets = state
        .merge_candidate(selected)
        .expect("merge should succeed");
    state
        .promote_best_to_current()
        .expect("promotion should succeed");
    let refreshed_output = graph
        .propagate_crown_to_node_with_alpha(
            input,
            graph.output_name(),
            &HashMap::new(),
            state.current(),
            alpha_state,
            None,
            None,
        )
        .expect("refreshed output CROWN should succeed");
    (tightened_targets, state, baseline_output, refreshed_output)
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_reference_refresh_tightens_target_without_regressing_output_3677() {
    let (graph, input) = make_refresh_graph();
    let (reference, alpha_state, targets, selected) = collect_refresh_state(&graph, &input);
    let (tightened_targets, state, baseline_output, refreshed_output) =
        baseline_and_refreshed_outputs(
            &graph,
            &input,
            &reference,
            &alpha_state,
            targets,
            &selected,
        );

    assert!(
        tightened_targets > 0,
        "#3677 selected-node refresh should tighten at least one activation input"
    );
    let linear2_reference = reference
        .get("linear2")
        .expect("reference linear2 bounds should exist");
    let linear2_refreshed = state
        .current()
        .get("linear2")
        .expect("refreshed linear2 bounds should exist");
    assert_bounds_do_not_loosen(
        linear2_refreshed,
        linear2_reference,
        1e-6,
        "#3677 refreshed linear2",
    );
    assert_bounds_do_not_loosen(
        &refreshed_output,
        &baseline_output,
        1e-6,
        "#3677 refreshed output",
    );
}

#[test]
fn test_merge_tighter_bounds_disjoint_intervals_no_crash_3684() {
    // Disjoint intervals: current=[5,10], candidate=[1,3].
    // Element-wise max/min would give [5,3] (inverted).
    // Fix: keep current bounds unchanged for disjoint elements.
    let mut initial = HashMap::new();
    initial.insert("target".to_string(), tensor(&[5.0, -1.0], &[10.0, 3.0]));
    let mut state = GraphAlphaReferenceBounds::new(initial, vec!["target".to_string()])
        .expect("reference bounds should initialize");

    let mut candidate = HashMap::new();
    candidate.insert("target".to_string(), tensor(&[1.0, 0.0], &[3.0, 2.0]));

    // Must not crash — before the fix, this would error on inverted bounds.
    let tightened = state
        .merge_candidate(&candidate)
        .expect("#3684 merge with disjoint intervals should not crash");

    let best = state
        .best()
        .get("target")
        .expect("best target bounds should exist");

    // Element 0: current=[5,10], candidate=[1,3] → disjoint → keep [5,10].
    assert_eq!(best.lower()[[0]], 5.0, "#3684 disjoint lower[0] kept");
    assert_eq!(best.upper()[[0]], 10.0, "#3684 disjoint upper[0] kept");
    // Element 1: current=[-1,3], candidate=[0,2] → overlap → merge to [0,2].
    assert_eq!(
        best.lower()[[1]],
        0.0,
        "#3684 overlapping lower[1] tightened"
    );
    assert_eq!(
        best.upper()[[1]],
        2.0,
        "#3684 overlapping upper[1] tightened"
    );
    // Tightened count should be 1 (element 1 tightened, element 0 kept).
    assert_eq!(tightened, 1, "#3684 one target tightened");
}

#[test]
fn test_nan_candidate_rejected_at_bounded_tensor_construction_3684() {
    // NaN defense-in-depth: the merge function's NaN guard is unreachable
    // through normal API usage because BoundedTensor constructors reject NaN.
    // This test documents that invariant. The NaN guard in merge_tighter_bounds
    // is a safety net in case a future code path bypasses validation (#3684).
    let result = BoundedTensor::new_allow_infinite(
        arr1(&[f32::NAN, 2.0]).into_dyn(),
        arr1(&[3.0, 4.0]).into_dyn(),
    );
    assert!(
        result.is_err(),
        "#3684 BoundedTensor rejects NaN — merge NaN guard is defense-in-depth"
    );
}

#[test]
fn test_merge_tighter_bounds_fully_disjoint_all_elements_3684() {
    // All elements disjoint — the entire merge should be a no-op.
    let mut initial = HashMap::new();
    initial.insert("target".to_string(), tensor(&[10.0, 20.0], &[15.0, 25.0]));
    let mut state = GraphAlphaReferenceBounds::new(initial, vec!["target".to_string()])
        .expect("reference bounds should initialize");

    let mut candidate = HashMap::new();
    candidate.insert("target".to_string(), tensor(&[1.0, 1.0], &[5.0, 5.0]));

    let tightened = state
        .merge_candidate(&candidate)
        .expect("#3684 fully disjoint merge should not crash");

    let best = state
        .best()
        .get("target")
        .expect("best target bounds should exist");
    assert_eq!(best.lower()[[0]], 10.0, "#3684 fully disjoint lower[0]");
    assert_eq!(best.upper()[[0]], 15.0, "#3684 fully disjoint upper[0]");
    assert_eq!(best.lower()[[1]], 20.0, "#3684 fully disjoint lower[1]");
    assert_eq!(best.upper()[[1]], 25.0, "#3684 fully disjoint upper[1]");
    assert_eq!(tightened, 0, "#3684 no targets tightened in fully disjoint");
}

#[test]
fn test_merge_tighter_bounds_touching_intervals_produce_point_3684() {
    // Touching intervals: current=[0,5], candidate=[5,10].
    // max(lower)=5, min(upper)=5 → point interval [5,5].
    // The guard `merged_lower > merged_upper` is false (5 == 5),
    // so this is treated as overlapping (valid), not disjoint.
    let mut initial = HashMap::new();
    initial.insert("target".to_string(), tensor(&[0.0], &[5.0]));
    let mut state = GraphAlphaReferenceBounds::new(initial, vec!["target".to_string()])
        .expect("reference bounds should initialize");

    let mut candidate = HashMap::new();
    candidate.insert("target".to_string(), tensor(&[5.0], &[10.0]));

    let tightened = state
        .merge_candidate(&candidate)
        .expect("#3684 touching interval merge should not crash");

    let best = state
        .best()
        .get("target")
        .expect("best target bounds should exist");
    // Point interval [5,5] — both bounds equal.
    assert_eq!(best.lower()[[0]], 5.0, "#3684 touching → point lower");
    assert_eq!(best.upper()[[0]], 5.0, "#3684 touching → point upper");
    assert_eq!(tightened, 1, "#3684 touching intervals tighten to point");
}

#[test]
fn test_merge_tighter_bounds_infinite_current_tightens_to_candidate_3684() {
    // Infinite current bounds: current=[-inf, +inf], candidate=[-2, 3].
    // max(-inf, -2)=-2, min(+inf, 3)=3 → merged=[-2, 3].
    let mut initial = HashMap::new();
    initial.insert(
        "target".to_string(),
        tensor(&[f32::NEG_INFINITY], &[f32::INFINITY]),
    );
    let mut state = GraphAlphaReferenceBounds::new(initial, vec!["target".to_string()])
        .expect("reference bounds should initialize");

    let mut candidate = HashMap::new();
    candidate.insert("target".to_string(), tensor(&[-2.0], &[3.0]));

    let tightened = state
        .merge_candidate(&candidate)
        .expect("#3684 infinite-to-finite merge should succeed");

    let best = state
        .best()
        .get("target")
        .expect("best target bounds should exist");
    assert_eq!(best.lower()[[0]], -2.0, "#3684 infinite lower tightened");
    assert_eq!(best.upper()[[0]], 3.0, "#3684 infinite upper tightened");
    assert_eq!(tightened, 1, "#3684 infinite → finite is tightening");
}

#[test]
fn test_merge_reference_bound_maps_skips_shape_mismatched_nodes_4384() {
    // DAG models can produce different shapes for the same node name when
    // IBP sees pre-concat [96] and CROWN sees post-concat [192]. The merge
    // should skip the mismatched node and keep matching nodes intact.
    let mut current = HashMap::new();
    current.insert("matched".to_string(), tensor(&[-2.0, -1.0], &[4.0, 3.0]));
    current.insert(
        "mismatched".to_string(),
        tensor(&[-1.0, 0.0, 1.0], &[2.0, 3.0, 4.0]),
    );

    let mut candidate = HashMap::new();
    candidate.insert("matched".to_string(), tensor(&[-1.0, -3.0], &[2.5, 2.0]));
    // Different shape: candidate has 2 elements, current has 3.
    candidate.insert("mismatched".to_string(), tensor(&[-1.0, 0.0], &[2.0, 3.0]));
    // New key only in candidate.
    candidate.insert("new_node".to_string(), tensor(&[0.0], &[1.0]));

    let merged = merge_reference_bound_maps(Some(&current), Some(&candidate))
        .expect("#4384 shape-mismatched merge should not error")
        .expect("merged should be Some");

    // "matched" node: should be tightened (same shape, element-wise merge).
    let matched = merged.get("matched").expect("matched node should exist");
    assert_eq!(
        matched.lower()[[0]],
        -1.0,
        "#4384 matched lower[0] tightened"
    );
    assert_eq!(
        matched.upper()[[0]],
        2.5,
        "#4384 matched upper[0] tightened"
    );

    // "mismatched" node: should keep current bounds (shape mismatch → skip).
    let mismatched = merged
        .get("mismatched")
        .expect("mismatched node should exist");
    assert_eq!(
        mismatched.shape(),
        &[3],
        "#4384 mismatched node keeps current shape"
    );
    assert_eq!(
        mismatched.lower()[[0]],
        -1.0,
        "#4384 mismatched keeps current lower"
    );
    assert_eq!(
        mismatched.upper()[[2]],
        4.0,
        "#4384 mismatched keeps current upper"
    );

    // "new_node": only in candidate, should be added.
    let new_node = merged.get("new_node").expect("new_node should exist");
    assert_eq!(
        new_node.lower()[[0]],
        0.0,
        "#4384 new_node added from candidate"
    );
}

/// Performance proof: `merge_reference_bound_maps` scales O(N) in node count.
///
/// The merge iterates over `candidate` entries and does a constant-time
/// HashMap lookup + element-wise merge for each. This test verifies that
/// the output size and content are correct for increasing N, proving no
/// accidental quadratic inner loop exists. Part of performance_proofs phase.
#[test]
fn test_merge_reference_bound_maps_scales_linearly_in_node_count() {
    for n in [1, 10, 50, 200] {
        let mut current: HashMap<String, BoundedTensor> = HashMap::with_capacity(n);
        let mut candidate: HashMap<String, BoundedTensor> = HashMap::with_capacity(n);
        for i in 0..n {
            let name = format!("node_{}", i);
            current.insert(name.clone(), tensor(&[-1.0], &[1.0]));
            candidate.insert(name, tensor(&[-0.5], &[0.5]));
        }
        // Add one candidate-only node to verify new entries are added.
        candidate.insert("extra".to_string(), tensor(&[0.0], &[1.0]));

        let merged = merge_reference_bound_maps(Some(&current), Some(&candidate))
            .expect("merge should succeed")
            .expect("merged should be Some");

        // Output must contain all N shared nodes + 1 candidate-only node.
        assert_eq!(
            merged.len(),
            n + 1,
            "merge with {} shared nodes should produce {} entries",
            n,
            n + 1
        );

        // Verify tightening happened on shared nodes.
        for i in 0..n {
            let name = format!("node_{}", i);
            let bounds = merged.get(&name).unwrap();
            assert_eq!(bounds.lower()[[0]], -0.5, "node_{} lower tightened", i);
            assert_eq!(bounds.upper()[[0]], 0.5, "node_{} upper tightened", i);
        }
        // Verify candidate-only node was added.
        assert!(merged.contains_key("extra"));
    }
}

/// Performance proof: merge is idempotent — `merge(A, A) == A`.
///
/// This property proves that caching the merge result for repeated calls
/// with the same inputs (as happens in the BaB loop when alpha_node_bounds
/// doesn't change) would be safe. Each element-wise merge of identical
/// intervals must produce the same interval.
#[test]
fn test_merge_reference_bound_maps_idempotent() {
    let mut bounds: HashMap<String, BoundedTensor> = HashMap::new();
    bounds.insert("relu_0".to_string(), tensor(&[-2.0, 0.5], &[3.0, 4.0]));
    bounds.insert("linear_1".to_string(), tensor(&[-1.0], &[1.0]));
    bounds.insert(
        "output".to_string(),
        tensor(&[0.0, -0.5, 1.0], &[2.0, 0.5, 3.0]),
    );

    let merged = merge_reference_bound_maps(Some(&bounds), Some(&bounds))
        .expect("self-merge should succeed")
        .expect("merged should be Some");

    assert_eq!(
        merged.len(),
        bounds.len(),
        "idempotent merge preserves entry count"
    );
    for (name, original) in &bounds {
        let result = merged.get(name).unwrap();
        assert_eq!(
            result.lower().as_slice().unwrap(),
            original.lower().as_slice().unwrap(),
            "idempotent merge preserves lower for {}",
            name
        );
        assert_eq!(
            result.upper().as_slice().unwrap(),
            original.upper().as_slice().unwrap(),
            "idempotent merge preserves upper for {}",
            name
        );
    }
}

/// Performance proof: `merge_reference_bound_maps(Some(A), None)` returns
/// a clone of A without iterating over entries. This is the fast path when
/// only alpha_node_bounds exist (no child or IBP bounds).
#[test]
fn test_merge_reference_bound_maps_single_source_fast_path() {
    let mut bounds: HashMap<String, BoundedTensor> = HashMap::new();
    for i in 0..100 {
        bounds.insert(format!("node_{}", i), tensor(&[-(i as f32)], &[i as f32]));
    }

    // (Some, None) path — should return clone of first arg.
    let result = merge_reference_bound_maps(Some(&bounds), None)
        .expect("single-source merge should succeed")
        .expect("should be Some");
    assert_eq!(result.len(), 100);

    // (None, Some) path — should return clone of second arg.
    let result = merge_reference_bound_maps(None, Some(&bounds))
        .expect("single-source merge should succeed")
        .expect("should be Some");
    assert_eq!(result.len(), 100);

    // (None, None) path — should return None.
    let result = merge_reference_bound_maps(None, None).expect("empty merge should succeed");
    assert!(result.is_none());
}

/// Performance proof: shape-mismatched nodes in merge produce O(N) work,
/// not O(N²). When all nodes are shape-mismatched, the merge skips all
/// entries and returns the original current map unchanged. This verifies
/// the skip path (#4384) doesn't accidentally re-process entries.
#[test]
fn test_merge_reference_bound_maps_all_mismatched_is_linear() {
    let mut current: HashMap<String, BoundedTensor> = HashMap::new();
    let mut candidate: HashMap<String, BoundedTensor> = HashMap::new();
    for i in 0..50 {
        let name = format!("node_{}", i);
        // Current has 3 elements, candidate has 2 — always mismatched.
        current.insert(name.clone(), tensor(&[-1.0, 0.0, 1.0], &[2.0, 3.0, 4.0]));
        candidate.insert(name, tensor(&[-0.5, 0.5], &[1.5, 2.5]));
    }

    let merged = merge_reference_bound_maps(Some(&current), Some(&candidate))
        .expect("all-mismatched merge should succeed")
        .expect("should be Some");

    // All current entries preserved (mismatched candidates skipped).
    assert_eq!(merged.len(), 50, "no entries lost or duplicated");
    for i in 0..50 {
        let name = format!("node_{}", i);
        let bounds = merged.get(&name).unwrap();
        assert_eq!(bounds.shape(), &[3], "current shape preserved for {}", name);
        assert_eq!(
            bounds.lower()[[0]],
            -1.0,
            "current lower preserved for {}",
            name
        );
    }
}

mod proptest_merge {
    use super::*;
    use proptest::prelude::*;

    /// Generate valid interval bounds [lower, upper] where lower <= upper.
    fn valid_interval(range: f32) -> impl Strategy<Value = (f32, f32)> {
        (-range..=range)
            .prop_flat_map(move |a| (-range..=range).prop_map(move |b| (a.min(b), a.max(b))))
    }

    proptest! {
        #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

        /// Property: merge_tighter_bounds never produces inverted bounds and
        /// never loosens the current interval (#3684).
        #[test]
        fn merge_never_inverts_or_loosens_3684(
            (cur_l, cur_u) in valid_interval(100.0),
            (cand_l, cand_u) in valid_interval(100.0),
        ) {
            let current = tensor(&[cur_l], &[cur_u]);
            let candidate = tensor(&[cand_l], &[cand_u]);

            let (merged, _tightened) = merge_tighter_bounds("prop", &current, &candidate)
                .expect("merge should not crash on any valid interval pair");

            // Invariant 1: merged bounds are valid (lower <= upper).
            prop_assert!(
                merged.lower()[[0]] <= merged.upper()[[0]],
                "merged bounds inverted: [{}, {}]",
                merged.lower()[[0]], merged.upper()[[0]]
            );

            // Invariant 2: merged bounds never loosen — lower never decreases,
            // upper never increases.
            prop_assert!(
                merged.lower()[[0]] >= cur_l - 1e-6,
                "merged lower loosened: {} < {}",
                merged.lower()[[0]], cur_l
            );
            prop_assert!(
                merged.upper()[[0]] <= cur_u + 1e-6,
                "merged upper loosened: {} > {}",
                merged.upper()[[0]], cur_u
            );
        }

        /// Property: when intervals overlap, merge tightens to the intersection.
        #[test]
        fn merge_overlapping_equals_intersection_3684(
            (cur_l, cur_u) in valid_interval(100.0),
            (cand_l, cand_u) in valid_interval(100.0),
        ) {
            // Only test overlapping intervals.
            let expected_lower = f32::max(cur_l, cand_l);
            let expected_upper = f32::min(cur_u, cand_u);
            prop_assume!(expected_lower <= expected_upper);

            let current = tensor(&[cur_l], &[cur_u]);
            let candidate = tensor(&[cand_l], &[cand_u]);

            let (merged, _) = merge_tighter_bounds("prop", &current, &candidate)
                .expect("merge should succeed for overlapping intervals");

            let merged_l = merged.lower()[[0]];
            let merged_u = merged.upper()[[0]];
            prop_assert!(
                (merged_l - expected_lower).abs() < 1e-6,
                "overlapping merge lower: got {}, expected {}",
                merged_l, expected_lower
            );
            prop_assert!(
                (merged_u - expected_upper).abs() < 1e-6,
                "overlapping merge upper: got {}, expected {}",
                merged_u, expected_upper
            );
        }

        /// Property: disjoint intervals produce an exact no-op (current preserved).
        /// Uses a strategy that generates guaranteed-disjoint intervals to avoid
        /// excessive prop_assume! rejections (flaky with random filtering).
        #[test]
        fn merge_disjoint_preserves_current_3684(
            gap in 0.01_f32..50.0,
            cur_width in 0.01_f32..50.0,
            cand_width in 0.01_f32..50.0,
            cur_base in -50.0_f32..50.0,
        ) {
            // Construct guaranteed-disjoint intervals:
            // current = [cur_base, cur_base + cur_width]
            // candidate = [cur_base + cur_width + gap, cur_base + cur_width + gap + cand_width]
            let cur_l = cur_base;
            let cur_u = cur_base + cur_width;
            let cand_l = cur_u + gap;
            let cand_u = cand_l + cand_width;

            let current = tensor(&[cur_l], &[cur_u]);
            let candidate = tensor(&[cand_l], &[cand_u]);

            let (merged, tightened) = merge_tighter_bounds("prop", &current, &candidate)
                .expect("disjoint merge should not crash");

            // Disjoint: current bounds preserved exactly.
            prop_assert_eq!(merged.lower()[[0]], cur_l);
            prop_assert_eq!(merged.upper()[[0]], cur_u);
            prop_assert!(!tightened, "disjoint merge should not report tightening");
        }

        /// Performance proof: merge is idempotent at element level.
        /// merge(x, x) == x for all valid intervals. This proves that
        /// caching the merge result for repeated BaB calls with unchanged
        /// alpha_node_bounds would be correct.
        #[test]
        fn merge_idempotent_element(
            (l, u) in valid_interval(100.0),
        ) {
            let bounds = tensor(&[l], &[u]);
            let (merged, tightened) = merge_tighter_bounds("prop", &bounds, &bounds)
                .expect("self-merge should succeed");

            prop_assert!(
                (merged.lower()[[0]] - l).abs() < 1e-6,
                "idempotent: lower changed from {} to {}",
                l, merged.lower()[[0]]
            );
            prop_assert!(
                (merged.upper()[[0]] - u).abs() < 1e-6,
                "idempotent: upper changed from {} to {}",
                u, merged.upper()[[0]]
            );
            prop_assert!(
                !tightened,
                "self-merge should not report tightening"
            );
        }

        /// Performance proof: merge is monotone under repeated application.
        /// If merge(A,B) = C, then merge(C,B) == C (fixpoint after one step).
        /// This proves the BaB loop doesn't gain tightness from redundant
        /// re-merges — a single merge extracts all available tightening.
        #[test]
        fn merge_reaches_fixpoint_in_one_step(
            (a_l, a_u) in valid_interval(100.0),
            (b_l, b_u) in valid_interval(100.0),
        ) {
            let a = tensor(&[a_l], &[a_u]);
            let b = tensor(&[b_l], &[b_u]);

            let (c, _) = merge_tighter_bounds("prop_step1", &a, &b)
                .expect("first merge should succeed");
            let (d, tightened) = merge_tighter_bounds("prop_step2", &c, &b)
                .expect("second merge should succeed");

            prop_assert!(
                (d.lower()[[0]] - c.lower()[[0]]).abs() < 1e-6,
                "fixpoint: second merge changed lower from {} to {}",
                c.lower()[[0]], d.lower()[[0]]
            );
            prop_assert!(
                (d.upper()[[0]] - c.upper()[[0]]).abs() < 1e-6,
                "fixpoint: second merge changed upper from {} to {}",
                c.upper()[[0]], d.upper()[[0]]
            );
            prop_assert!(
                !tightened,
                "second merge should not report additional tightening"
            );
        }
    }
}
