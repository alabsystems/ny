// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for DomainList BaB verify_graph_gpu_domain_list entry point.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ndarray::{arr1, arr2, ArrayD, Axis, IxDyn};
use ny_core::Result;
use ny_tensor::TreeTraversal;
use ny_test_utils::CountingGemmEngine;

use super::check::BabLoopState;
use super::init::{cache_input_split_linear_bounds, InputSplitBootstrap};
use super::initial_bounds_deadline_status;
use super::input_split::{process_input_split_batch, InputSplitOutcome};
use super::input_split_support::build_parent_contexts;
use crate::batched_domain::{
    BatchedDomainOptions, DomainList, DomainListConfig, DomainMetadata, PickedDomains,
    ProcessedDomains,
};
use crate::beta_crown::branching::BranchingHeuristic;
use crate::beta_crown::config::{BetaCrownConfig, ConvMode, InputClipType, PhaseBudgetConfig};
use crate::beta_crown::engine::graph::input_split::adv_check::ADV_CHECK_INTERVAL;
use crate::beta_crown::engine::graph::input_split::shared::{
    compute_crown_or_ibp_bounds, graph_spec_ibp_fallback,
};
use crate::beta_crown::engine::GraphDomainBatchMetricsSink;
use crate::beta_crown::result::BabVerificationStatus;
use crate::beta_crown::{BetaCrownVerifier, GraphDomainBatchCallerLane, GraphDomainBatchRecord};
use crate::bounds::LinearBounds;
use crate::layers::{AddLayer, LinearLayer};
use crate::network::GraphNode;
use crate::{GraphNetwork, Layer, ReLULayer};

#[derive(Debug, Default)]
struct RecordingGraphDomainBatchMetricsSink {
    records: Mutex<Vec<GraphDomainBatchRecord>>,
}

impl RecordingGraphDomainBatchMetricsSink {
    fn snapshot(&self) -> Vec<GraphDomainBatchRecord> {
        self.records
            .lock()
            .expect("graph domain-batch metrics mutex should not be poisoned")
            .clone()
    }
}

impl GraphDomainBatchMetricsSink for RecordingGraphDomainBatchMetricsSink {
    fn record_batch_summary(&self, record: &GraphDomainBatchRecord) -> Result<()> {
        self.records
            .lock()
            .expect("graph domain-batch metrics mutex should not be poisoned")
            .push(record.clone());
        Ok(())
    }
}

/// Construct a graph with a linear layer + ReLU + linear layer whose IBP
/// bounds trivially satisfy the verification threshold. The weights are
/// chosen so that the output `y` satisfies `y >= 0` (lower > threshold)
/// when the input is non-negative.
fn easy_verify_graph() -> GraphNetwork {
    let linear1 = LinearLayer::new(arr2(&[[2.0_f32]]), None).expect("valid linear1");
    let linear2 = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("valid linear2");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

/// Construct a graph where spec-guided CROWN proves the correlation between
/// two identical outputs while plain output-interval IBP cannot.  The exact
/// objective is `x - x = 0`; IBP sees `[-1, 1] - [-1, 1] = [-2, 2]`.
fn reference_gap_graph_3870() -> GraphNetwork {
    let duplicate = LinearLayer::new(arr2(&[[1.0_f32], [1.0_f32]]), None)
        .expect("valid duplicate-output layer");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("duplicate", Layer::Linear(duplicate)));
    graph.set_output("duplicate");
    graph
}

fn identity_graph_3870() -> GraphNetwork {
    let linear = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("valid identity layer");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.set_output("linear");
    graph
}

fn difference_graph_3870() -> GraphNetwork {
    let linear =
        LinearLayer::new(arr2(&[[1.0_f32, -1.0_f32]]), None).expect("valid difference layer");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.set_output("linear");
    graph
}

fn residual_deadline_graph_4413() -> GraphNetwork {
    let linear = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("valid residual linear");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "residual",
        Layer::Add(AddLayer),
        "relu1",
        crate::NETWORK_INPUT,
    ));
    graph.set_output("residual");
    graph
}

fn complete_clip_override_graph_3870() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("out linear")),
        vec!["relu".to_string()],
    ));
    graph.set_output("out");
    graph
}

fn complete_clip_parent_linear_3870() -> LinearBounds {
    LinearBounds {
        lower_a: arr2(&[[1.0_f32]]),
        lower_b: arr1(&[0.0_f32]),
        upper_a: arr2(&[[1.0_f32]]),
        upper_b: arr1(&[0.0_f32]),
        lower_a_err: None,
        upper_a_err: None,
    }
}

fn complete_clip_reorder_fixture_3870(
    with_parent_linear: bool,
) -> Result<(
    BetaCrownVerifier,
    GraphNetwork,
    PickedDomains,
    InputSplitBootstrap,
    BabLoopState,
    DomainList,
)> {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Complete,
        reorder_bab: true,
        input_split_ibp_enhancement: true,
        use_alpha_crown: false,
        enable_cuts: false,
        max_domains: 16,
        max_depth: 4,
        timeout: Duration::from_secs(5),
        ..Default::default()
    });
    let graph = verifier.configured_graph_for_crown(&complete_clip_override_graph_3870());
    let parent_linear = complete_clip_parent_linear_3870();
    let mut parent_meta = DomainMetadata::root(-1.0, 1.0)?;
    if with_parent_linear {
        parent_meta.cached_la = Some(Arc::new(cache_input_split_linear_bounds(&parent_linear)));
    }
    let picked = PickedDomains {
        batch_size: 1,
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![-1.0_f32])
            .expect("valid picked lower bounds"),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0_f32])
            .expect("valid picked upper bounds"),
        global_lbs: vec![-1.0_f32],
        global_ubs: vec![1.0_f32],
        metadata: vec![parent_meta],
    };
    let bootstrap = InputSplitBootstrap {
        spec_matrix: arr2(&[[1.0_f32]]),
        fixed_node_bounds: None,
        root_alpha_state: None,
        root_linear_bounds: with_parent_linear.then_some(parent_linear),
        mul_binary_alphas: None,
        deadline: None,
    };
    Ok((
        verifier,
        graph,
        picked,
        bootstrap,
        BabLoopState::new(Instant::now()),
        empty_input_split_domain_list(vec![1])?,
    ))
}

fn input_split_verifier_config_3870() -> BetaCrownConfig {
    BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        use_alpha_crown: false,
        enable_cuts: false,
        max_domains: 16,
        max_depth: 4,
        timeout: Duration::from_secs(5),
        ..Default::default()
    }
}

fn empty_input_split_domain_list(input_shape: Vec<usize>) -> Result<DomainList> {
    DomainList::new(DomainListConfig {
        traversal: TreeTraversal::BreadthFirst,
        layer_names: Vec::new(),
        layer_shapes: HashMap::new(),
        input_shape,
        initial_capacity: 8,
        max_queue_size: 0,
    })
}

#[test]
fn domain_list_eviction_latch_prevents_verified_exhaustion() -> Result<()> {
    let mut domain_list = DomainList::new(DomainListConfig {
        traversal: TreeTraversal::BreadthFirst,
        layer_names: Vec::new(),
        layer_shapes: HashMap::new(),
        input_shape: vec![1],
        initial_capacity: 2,
        max_queue_size: 1,
    })?;
    domain_list.configure_queue_eviction(0, false)?;
    domain_list.add(ProcessedDomains {
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![-1.0, -0.5]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![1.0, 0.5]).unwrap(),
        global_lbs: vec![-1.0, -0.5],
        global_ubs: vec![1.0, 0.5],
        metadata: vec![
            DomainMetadata::root(-1.0, 1.0)?,
            DomainMetadata::root(-0.5, 0.5)?,
        ],
        keep_mask: vec![true, true],
    })?;
    assert_eq!(domain_list.evicted_count(), 1);

    let mut state = BabLoopState::new(Instant::now());
    super::latch_domain_list_eviction(&domain_list, &mut state);
    assert!(matches!(
        state.build_final_result().result,
        BabVerificationStatus::Unknown { .. }
    ));
    Ok(())
}

fn picked_domains_3870(
    input_lowers: Vec<f32>,
    input_uppers: Vec<f32>,
    input_shape: &[usize],
    lower_bound: f32,
    upper_bound: f32,
    needs_bounding: bool,
) -> Result<PickedDomains> {
    let mut metadata = DomainMetadata::root(lower_bound, upper_bound)?;
    metadata.set_needs_bounding(needs_bounding);
    let batch_shape: Vec<usize> = std::iter::once(1)
        .chain(input_shape.iter().copied())
        .collect();
    Ok(PickedDomains {
        batch_size: 1,
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: ArrayD::from_shape_vec(IxDyn(&batch_shape), input_lowers)
            .expect("valid picked lower bounds"),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&batch_shape), input_uppers)
            .expect("valid picked upper bounds"),
        global_lbs: vec![lower_bound],
        global_ubs: vec![upper_bound],
        metadata: vec![metadata],
    })
}

fn picked_input_bounds_3870(
    picked: &PickedDomains,
    idx: usize,
) -> Result<ny_tensor::BoundedTensor> {
    ny_tensor::BoundedTensor::new(
        picked
            .input_lowers
            .index_axis(Axis(0), idx)
            .to_owned()
            .into_dyn(),
        picked
            .input_uppers
            .index_axis(Axis(0), idx)
            .to_owned()
            .into_dyn(),
    )
}

fn reorder_child_boxes_3870(queued: &PickedDomains) -> Vec<(Vec<f32>, Vec<f32>)> {
    (0..queued.batch_size)
        .map(|idx| {
            let child = picked_input_bounds_3870(queued, idx).expect("child bounds");
            (
                child.lower().iter().copied().collect(),
                child.upper().iter().copied().collect(),
            )
        })
        .collect()
}

fn assert_reordered_child_partition_3870(child_boxes: &[(Vec<f32>, Vec<f32>)]) {
    let split_dims: Vec<usize> = (0..2)
        .filter(|&dim| {
            child_boxes
                .iter()
                .any(|(lower, upper)| lower[dim] != 0.0 || upper[dim] != 1.0)
        })
        .collect();
    assert_eq!(
        split_dims.len(),
        1,
        "input split should bisect exactly one input dimension"
    );
    let split_dim = split_dims[0];
    let unsplit_dim = 1 - split_dim;
    let mut split_segments: Vec<(f32, f32)> = child_boxes
        .iter()
        .map(|(lower, upper)| (lower[split_dim], upper[split_dim]))
        .collect();
    split_segments.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite split segment"));
    assert_eq!(split_segments, vec![(0.0_f32, 0.5_f32), (0.5_f32, 1.0_f32)]);
    assert!(
        child_boxes
            .iter()
            .all(|(lower, upper)| lower[unsplit_dim] == 0.0 && upper[unsplit_dim] == 1.0),
        "the non-split input dimension must stay unchanged across both queued children"
    );
}

fn sort_scalar_pairs_3870(mut pairs: Vec<(f32, f32)>) -> Vec<(f32, f32)> {
    pairs.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .expect("lower bounds are finite")
            .then_with(|| a.1.partial_cmp(&b.1).expect("upper bounds are finite"))
    });
    pairs
}

fn assert_deferred_rebound_matches_direct_crown_3870(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    queued: &PickedDomains,
    bootstrap: &InputSplitBootstrap,
) -> Result<()> {
    let rebound = build_parent_contexts(verifier, graph, queued, &[0, 1], bootstrap, None)?;
    let rebound = rebound.contexts;
    assert!(
        rebound.iter().all(|ctx| ctx.linear_bounds.is_some()),
        "deferred child picks must recover fresh linear bounds for SB scoring and clipping"
    );
    let rebound_pairs = sort_scalar_pairs_3870(
        rebound
            .iter()
            .map(|ctx| (ctx.lower_bound, ctx.upper_bound))
            .collect(),
    );
    let mut expected_pairs = Vec::new();
    for idx in 0..queued.batch_size {
        let child_input = picked_input_bounds_3870(queued, idx)?;
        let (expected_bounds, _) = compute_crown_or_ibp_bounds(
            graph,
            &child_input,
            &bootstrap.spec_matrix,
            None,
            None,
            None,
            None,
            None,
            None,
            false,
        )?;
        expected_pairs.push((expected_bounds.lower()[[0]], expected_bounds.upper()[[0]]));
    }
    for ((actual_lower, actual_upper), (expected_lower, expected_upper)) in rebound_pairs
        .iter()
        .zip(sort_scalar_pairs_3870(expected_pairs).iter())
    {
        assert!(
            (actual_lower - expected_lower).abs() <= 1e-5,
            "deferred lower bound mismatch: actual={} expected={}",
            actual_lower,
            expected_lower
        );
        assert!(
            (actual_upper - expected_upper).abs() <= 1e-5,
            "deferred upper bound mismatch: actual={} expected={}",
            actual_upper,
            expected_upper
        );
    }
    Ok(())
}

fn assert_scalar_bounds_close(
    actual: &ny_tensor::BoundedTensor,
    expected: &ny_tensor::BoundedTensor,
    tol: f32,
    label: &str,
) {
    assert!(
        (actual.lower()[[0]] - expected.lower()[[0]]).abs() <= tol,
        "{label}: lower mismatch actual={} expected={} (tol={})",
        actual.lower()[[0]],
        expected.lower()[[0]],
        tol
    );
    assert!(
        (actual.upper()[[0]] - expected.upper()[[0]]).abs() <= tol,
        "{label}: upper mismatch actual={} expected={} (tol={})",
        actual.upper()[[0]],
        expected.upper()[[0]],
        tol
    );
}

#[ntest::timeout(10000)]
#[test]
fn gpu_bab_ibp_prescreen_returns_verified_early_3870() -> Result<()> {
    let graph = easy_verify_graph();
    // Input [1.0, 2.0]: linear1 gives [2.0, 4.0], relu passes through,
    // linear2 gives [2.0, 4.0]. IBP lower = 2.0 > threshold -1.0 → verified.
    let input =
        ny_tensor::BoundedTensor::new(arr1(&[1.0_f32]).into_dyn(), arr1(&[2.0_f32]).into_dyn())?;
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        input_split_ibp_enhancement: true,
        timeout: Duration::from_secs(5),
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier.verify_graph_gpu_domain_list(
        &graph,
        &input,
        &[1.0],
        -1.0, // threshold: prove lower > -1.0
        None,
        None,
    )?;

    assert_eq!(
        result.result,
        BabVerificationStatus::Verified,
        "IBP pre-screen should verify this easy root domain without entering BaB"
    );
    assert_eq!(result.domains_explored, 1);
    assert_eq!(result.domains_verified, 1);

    Ok(())
}

/// The GPU DomainList input-split IBP pre-screen can return Verified before
/// its downstream bootstrap.  Configuration quarantine must run first.
#[ntest::timeout(10000)]
#[test]
fn gpu_ibp_prescreen_early_verified_rejects_cut_authority() -> Result<()> {
    let graph = easy_verify_graph();
    let input =
        ny_tensor::BoundedTensor::new(arr1(&[1.0_f32]).into_dyn(), arr1(&[2.0_f32]).into_dyn())?;
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        input_split_ibp_enhancement: true,
        enable_cuts: true,
        timeout: Duration::from_secs(5),
        ..Default::default()
    });

    let error = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &[1.0], -1.0, None, None)
        .expect_err("IBP would verify, but quarantined cut authority must reject first");
    assert!(
        error
            .to_string()
            .contains("cut proof authority is quarantined"),
        "expected quarantine error, got {error}"
    );
    Ok(())
}

#[ntest::timeout(60000)]
#[test]
fn domain_list_input_split_adaptive_route_matches_config_off_result() -> Result<()> {
    let _env_lock = ny_test_utils::env::lock_env();
    let gate_name =
        crate::beta_crown::engine::graph::adaptive_microbatch::ADAPTIVE_MICROBATCH_GATE_ENV;
    let graph = easy_verify_graph();
    let input =
        ny_tensor::BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())?;
    let base = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        input_split_ibp_enhancement: false,
        use_alpha_crown: false,
        enable_cuts: false,
        batch_size: 1,
        max_domains: 16,
        max_depth: 4,
        // This is a route-parity test, not a five-second throughput test.  A
        // short product deadline can race the adaptive route's larger CROWN
        // passes and turn the expected max-domain result into Timeout on a
        // loaded host.  Keep the verifier deadline outside the test harness's
        // bounded execution window so termination is governed by the explicit
        // domain/depth limits below.
        timeout: Duration::from_mins(2),
        ..Default::default()
    };
    let gate_dark = ny_test_utils::env::ScopedEnvVar::set(gate_name, "0");
    let preset_legacy = BetaCrownVerifier::new(BetaCrownConfig {
        auto_enlarge_batch_size: true,
        ..base.clone()
    })
    .verify_graph_gpu_domain_list(&graph, &input, &[1.0], 1.0, None, None)?;
    drop(gate_dark);

    let _gate_on = ny_test_utils::env::ScopedEnvVar::set(gate_name, "1");
    let legacy = BetaCrownVerifier::new(base.clone()).verify_graph_gpu_domain_list(
        &graph,
        &input,
        &[1.0],
        1.0,
        None,
        None,
    )?;
    let adaptive = BetaCrownVerifier::new(BetaCrownConfig {
        auto_enlarge_batch_size: true,
        ..base
    })
    .verify_graph_gpu_domain_list(&graph, &input, &[1.0], 1.0, None, None)?;

    assert_eq!(adaptive.result, legacy.result);
    assert_eq!(adaptive.domains_explored, legacy.domains_explored);
    assert_eq!(adaptive.domains_verified, legacy.domains_verified);
    assert_eq!(adaptive.max_depth_reached, legacy.max_depth_reached);
    assert_eq!(preset_legacy.result, legacy.result);
    assert_eq!(preset_legacy.domains_explored, legacy.domains_explored);
    assert_eq!(preset_legacy.domains_verified, legacy.domains_verified);
    assert_eq!(preset_legacy.max_depth_reached, legacy.max_depth_reached);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn gpu_bab_alpha_warmup_deadline_returns_unknown_4413() -> Result<()> {
    let graph = residual_deadline_graph_4413();
    let input =
        ny_tensor::BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())?;
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        use_alpha_crown: true,
        enable_cuts: false,
        timeout: Duration::from_secs(5),
        phase_budget: PhaseBudgetConfig {
            initial_bounds_fraction: 0.0,
            ..Default::default()
        },
        ..Default::default()
    });

    let result =
        verifier.verify_graph_gpu_domain_list(&graph, &input, &[1.0_f32], 0.0, None, None)?;

    match result.result {
        BabVerificationStatus::Unknown { ref reason } => {
            assert!(
                reason.contains("Initial-bound warmup exceeded its deadline cap"),
                "expected warmup-deadline reason, got {reason}"
            );
        }
        ref other => panic!("expected Unknown, got {other:?}"),
    }
    assert_eq!(
        result.domains_explored, 0,
        "deadline-capped warmup should stop before the root domain enters BaB"
    );

    Ok(())
}

#[test]
fn gpu_bab_initial_deadline_status_distinguishes_phase_and_global_expiry() {
    let now = Instant::now();
    let spent = now
        .checked_sub(Duration::from_millis(1))
        .expect("system uptime exceeds one millisecond");
    let live = now + Duration::from_secs(1);

    assert_eq!(
        initial_bounds_deadline_status(now, Some(spent), Some(spent)),
        BabVerificationStatus::Timeout,
        "an exhausted overall BaB budget is a verifier timeout"
    );
    assert!(matches!(
        initial_bounds_deadline_status(now, Some(spent), Some(live)),
        BabVerificationStatus::Unknown { reason }
            if reason.contains("Initial-bound warmup exceeded its deadline cap")
    ));
}

#[ntest::timeout(10000)]
#[test]
fn graph_spec_ibp_fallback_threads_engine_and_reuses_cache_4174() -> Result<()> {
    let graph = easy_verify_graph();
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        use_alpha_crown: false,
        timeout: Duration::from_secs(5),
        ..Default::default()
    });
    let configured_graph = verifier.configured_graph_for_crown(&graph);
    let input =
        ny_tensor::BoundedTensor::new(arr1(&[-0.25_f32]).into_dyn(), arr1(&[0.5_f32]).into_dyn())?;
    let spec_matrix = arr2(&[[1.0_f32]]);

    let (baseline_bounds, _) =
        graph_spec_ibp_fallback(&configured_graph, &input, &spec_matrix, None, None)?;

    let engine = CountingGemmEngine::new();
    let (engine_bounds, _) =
        graph_spec_ibp_fallback(&configured_graph, &input, &spec_matrix, Some(&engine), None)?;
    assert_scalar_bounds_close(&engine_bounds, &baseline_bounds, 1e-5, "engine fallback");
    assert!(
        engine.gemm_calls() > 0,
        "engine-backed fallback should dispatch GEMM when no node-bounds cache is supplied"
    );

    let mut cached_node_bounds = configured_graph.collect_node_bounds(&input)?;
    // Overwrite the output-node cache entry so the helper must visibly reuse
    // the caller's cache instead of silently recomputing IBP on CPU.
    let sentinel_output_bounds =
        ny_tensor::BoundedTensor::new(arr1(&[7.0_f32]).into_dyn(), arr1(&[9.0_f32]).into_dyn())?;
    cached_node_bounds.insert("linear2".to_string(), sentinel_output_bounds.clone());
    let cached_engine = CountingGemmEngine::new();
    let (cached_bounds, _) = graph_spec_ibp_fallback(
        &configured_graph,
        &input,
        &spec_matrix,
        Some(&cached_engine),
        Some(&cached_node_bounds),
    )?;
    assert_scalar_bounds_close(
        &cached_bounds,
        &sentinel_output_bounds,
        1e-5,
        "cached fallback",
    );
    assert_eq!(
        cached_engine.gemm_calls(),
        0,
        "pre-supplied node bounds should bypass fallback IBP recomputation"
    );

    Ok(())
}

/// Reference bounds for the IBP-enhancement toggle test.
struct IbpToggleReference {
    ibp_bounds: ny_tensor::BoundedTensor,
    threshold: f32,
}

fn build_ibp_toggle_reference(
    graph: &GraphNetwork,
    input: &ny_tensor::BoundedTensor,
    spec_matrix: &ndarray::Array2<f32>,
) -> Result<IbpToggleReference> {
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        input_split_ibp_enhancement: true,
        use_alpha_crown: false,
        conv_mode: ConvMode::Matrix,
        timeout: Duration::from_secs(5),
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);
    let configured_graph = verifier.configured_graph_for_crown(graph);
    let (ibp_bounds, _) =
        graph_spec_ibp_fallback(&configured_graph, input, spec_matrix, None, None)?;
    let threshold = ibp_bounds.lower()[[0]] - 1e-3;
    Ok(IbpToggleReference {
        ibp_bounds,
        threshold,
    })
}

#[ntest::timeout(10000)]
#[test]
fn gpu_bab_ibp_prescreen_skipped_when_enhancement_off_3870() -> Result<()> {
    let graph = reference_gap_graph_3870();
    let input =
        ny_tensor::BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())?;
    let objective = [1.0_f32, -1.0_f32];
    let spec_matrix = arr2(&[[1.0_f32, -1.0_f32]]);
    let reference = build_ibp_toggle_reference(&graph, &input, &spec_matrix)?;

    // IBP pre-screen path: ibp_enhancement=true should match IBP bounds.
    let pre_screen = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        input_split_ibp_enhancement: true,
        use_alpha_crown: false,
        conv_mode: ConvMode::Matrix,
        timeout: Duration::from_secs(5),
        ..Default::default()
    });
    let pre_screen_engine = CountingGemmEngine::new();
    let pre_result = pre_screen.verify_graph_gpu_domain_list(
        &graph,
        &input,
        &objective,
        reference.threshold,
        Some(&pre_screen_engine),
        None,
    )?;
    assert_eq!(pre_result.result, BabVerificationStatus::Verified);
    let pre_bounds = pre_result
        .output_bounds
        .as_ref()
        .expect("IBP pre-screen bounds");
    assert_scalar_bounds_close(pre_bounds, &reference.ibp_bounds, 1e-5, "pre-screen");
    let pre_screen_gemms = pre_screen_engine.gemm_calls();
    assert!(
        pre_screen_gemms > 0,
        "IBP pre-screen must execute the graph"
    );

    // With enhancement disabled, the entry must not return from the cheap IBP
    // pre-screen.  The engine-backed full bootstrap is deterministic for this
    // one-layer fixture and must preserve the exact duplicated-output
    // correlation, which distinguishes the route without a measured heuristic
    // tightness tolerance.
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        input_split_ibp_enhancement: false,
        use_alpha_crown: false,
        conv_mode: ConvMode::Matrix,
        timeout: Duration::from_secs(1),
        ..Default::default()
    });
    let full_engine = CountingGemmEngine::new();
    let result = verifier.verify_graph_gpu_domain_list(
        &graph,
        &input,
        &objective,
        reference.threshold,
        Some(&full_engine),
        None,
    )?;
    assert_eq!(result.result, BabVerificationStatus::Verified);
    let full_bounds = result
        .output_bounds
        .as_ref()
        .expect("full bootstrap bounds");
    assert!(
        full_bounds.lower()[[0]].abs() <= 1e-5 && full_bounds.upper()[[0]].abs() <= 1e-5,
        "enhancement-off route must preserve the exact x-x correlation through CROWN: [{}, {}]",
        full_bounds.lower()[[0]],
        full_bounds.upper()[[0]]
    );
    assert!(
        full_engine.gemm_calls() >= pre_screen_gemms,
        "full bootstrap unexpectedly performed less graph work than the IBP pre-screen"
    );

    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn gpu_bab_reorder_queue_marks_needs_bounding_and_rebounds_on_pick_3870() -> Result<()> {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        reorder_bab: true,
        adv_check: -1,
        ..input_split_verifier_config_3870()
    });
    let graph = verifier.configured_graph_for_crown(&difference_graph_3870());
    let picked = picked_domains_3870(
        vec![0.0_f32, 0.0_f32],
        vec![1.0_f32, 1.0_f32],
        &[2],
        -1.0,
        1.0,
        false,
    )?;
    let bootstrap = InputSplitBootstrap {
        spec_matrix: arr2(&[[1.0_f32]]),
        fixed_node_bounds: None,
        root_alpha_state: None,
        root_linear_bounds: None,
        mul_binary_alphas: None,
        deadline: None,
    };
    let mut state = BabLoopState::new(Instant::now());
    let mut domain_list = empty_input_split_domain_list(vec![2])?;

    let outcome = process_input_split_batch(
        &verifier,
        &graph,
        &picked,
        &[0],
        &[1.0_f32],
        &bootstrap,
        0.0,
        None,
        &mut state,
        &mut domain_list,
        0,
    )?;

    assert!(
        matches!(outcome, InputSplitOutcome::Continue),
        "reorder_bab should queue unresolved children instead of exiting early"
    );
    assert_eq!(state.domains_explored, 1);
    assert_eq!(domain_list.len(), 2);

    let queued = domain_list.pick_out_batched(2, BatchedDomainOptions::default())?;
    assert_eq!(queued.batch_size, 2);
    assert!(
        queued.metadata.iter().all(|meta| meta.needs_bounding()),
        "reordered children must be tagged for deferred parent-style bounding on the next pick"
    );
    assert!(
        queued
            .metadata
            .iter()
            .all(|meta| meta.cached_la().is_none()),
        "reordered children must not pretend the parent linear cache is fresh child data"
    );
    assert_eq!(
        queued.global_lbs,
        vec![-1.0_f32, -1.0_f32],
        "reordered children should inherit parent lower bounds until they are re-bounded"
    );
    assert_eq!(
        queued.global_ubs,
        vec![1.0_f32, 1.0_f32],
        "reordered children should inherit parent upper bounds until they are re-bounded"
    );
    assert_reordered_child_partition_3870(&reorder_child_boxes_3870(&queued));
    assert_deferred_rebound_matches_direct_crown_3870(&verifier, &graph, &queued, &bootstrap)?;

    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn gpu_bab_input_split_adv_check_short_circuits_batch_3870() -> Result<()> {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        adv_check: ADV_CHECK_INTERVAL as i32,
        ..input_split_verifier_config_3870()
    });
    let graph = verifier.configured_graph_for_crown(&identity_graph_3870());
    let picked = picked_domains_3870(vec![0.0_f32], vec![1.0_f32], &[1], 0.0, 1.0, false)?;
    let bootstrap = InputSplitBootstrap {
        spec_matrix: arr2(&[[1.0_f32]]),
        fixed_node_bounds: None,
        root_alpha_state: None,
        root_linear_bounds: None,
        mul_binary_alphas: None,
        deadline: None,
    };
    let mut state = BabLoopState::new(Instant::now());
    state.domains_explored = ADV_CHECK_INTERVAL;
    let mut domain_list = empty_input_split_domain_list(vec![1])?;

    let outcome = process_input_split_batch(
        &verifier,
        &graph,
        &picked,
        &[0],
        &[1.0_f32],
        &bootstrap,
        2.0,
        None,
        &mut state,
        &mut domain_list,
        0,
    )?;

    assert!(
        matches!(outcome, InputSplitOutcome::Violation),
        "adv_check should return an immediate violation before any split work once the interval gate opens"
    );
    assert_eq!(
        state.domains_explored,
        ADV_CHECK_INTERVAL,
        "adv_check short-circuit should happen before the per-parent exploration counter increments"
    );
    assert_eq!(state.domains_verified, 0);
    assert_eq!(domain_list.len(), 0);

    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn gpu_bab_reorder_complete_clip_carries_override_into_deferred_rebound_3870() -> Result<()> {
    let (verifier, graph, picked, bootstrap, mut state, mut domain_list) =
        complete_clip_reorder_fixture_3870(true)?;

    let outcome = process_input_split_batch(
        &verifier,
        &graph,
        &picked,
        &[0],
        &[1.0_f32],
        &bootstrap,
        0.2,
        None,
        &mut state,
        &mut domain_list,
        0,
    )?;

    assert!(
        matches!(outcome, InputSplitOutcome::Continue),
        "reordered complete-clip children should stay in the queue for deferred bounding"
    );
    assert_eq!(domain_list.len(), 2);

    let queued = domain_list.pick_out_batched(2, BatchedDomainOptions::default())?;
    assert!(
        queued.metadata.iter().all(|meta| meta.needs_bounding()),
        "complete-clipped reorder children must stay marked for deferred bounding"
    );
    assert!(
        queued
            .metadata
            .iter()
            .all(|meta| meta.node_bounds_override().is_some()),
        "complete-clipped reorder children must carry their child-local override through DomainList"
    );

    let rebound = build_parent_contexts(&verifier, &graph, &queued, &[0, 1], &bootstrap, None)?;
    let rebound = rebound.contexts;
    assert_eq!(rebound.len(), 2);
    assert!(
        rebound.iter().all(|ctx| ctx.upper_bound <= 0.21),
        "deferred rebound must consume the complete-clip override; got {:?}",
        rebound
            .iter()
            .map(|ctx| ctx.upper_bound)
            .collect::<Vec<_>>()
    );

    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn gpu_bab_reorder_complete_clip_without_parent_linear_recovers_override_3870() -> Result<()> {
    let (verifier, graph, picked, bootstrap, mut state, mut domain_list) =
        complete_clip_reorder_fixture_3870(false)?;

    let outcome = process_input_split_batch(
        &verifier,
        &graph,
        &picked,
        &[0],
        &[1.0_f32],
        &bootstrap,
        0.2,
        None,
        &mut state,
        &mut domain_list,
        0,
    )?;

    assert!(
        matches!(outcome, InputSplitOutcome::Continue),
        "complete-clip reorder should keep unresolved children queued even without cached parent linear bounds"
    );
    assert_eq!(domain_list.len(), 2);

    let queued = domain_list.pick_out_batched(2, BatchedDomainOptions::default())?;
    assert!(
        queued.metadata.iter().all(|meta| meta.needs_bounding()),
        "fallback complete-clip children must stay marked for deferred bounding"
    );
    assert!(
        queued
            .metadata
            .iter()
            .all(|meta| meta.node_bounds_override().is_some()),
        "fallback complete clipping must rebuild child-local overrides when parent linear cache is absent"
    );

    let rebound = build_parent_contexts(&verifier, &graph, &queued, &[0, 1], &bootstrap, None)?;
    let rebound = rebound.contexts;
    assert_eq!(rebound.len(), 2);
    assert!(
        rebound.iter().all(|ctx| ctx.upper_bound <= 0.21),
        "deferred rebound must consume the rebuilt complete-clip override; got {:?}",
        rebound
            .iter()
            .map(|ctx| ctx.upper_bound)
            .collect::<Vec<_>>()
    );

    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn gpu_bab_input_split_emits_domain_batch_metrics_4398() -> Result<()> {
    let metrics_sink = Arc::new(RecordingGraphDomainBatchMetricsSink::default());
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        reorder_bab: true,
        adv_check: -1,
        use_alpha_crown: false,
        enable_cuts: false,
        batch_size: 2,
        max_domains: 16,
        max_depth: 4,
        timeout: Duration::from_secs(5),
        ..Default::default()
    })
    .with_graph_domain_batch_metrics_sink(metrics_sink.clone());
    let graph = verifier.configured_graph_for_crown(&difference_graph_3870());
    let input = ny_tensor::BoundedTensor::new(
        arr1(&[0.0_f32, 0.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.0_f32]).into_dyn(),
    )?;
    let engine = CountingGemmEngine::new();

    let _result = verifier.verify_graph_gpu_domain_list(
        &graph,
        &input,
        &[1.0_f32],
        0.0,
        Some(&engine),
        None,
    )?;

    let records = metrics_sink.snapshot();
    assert!(
        !records.is_empty(),
        "GPU DomainList input split should emit at least one shared domain-batch record"
    );
    assert!(
        records
            .iter()
            .all(|record| record.caller_lane == GraphDomainBatchCallerLane::InputSplitDenseSpec),
        "single-objective GPU DomainList input split should reuse the dense-spec rebound lane"
    );
    assert!(
        records.iter().any(|record| record.domains_batched > 0),
        "at least one deferred rebound batch should report batched domains, got {records:?}"
    );

    Ok(())
}
