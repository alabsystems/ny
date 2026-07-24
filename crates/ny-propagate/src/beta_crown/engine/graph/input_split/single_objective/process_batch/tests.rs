// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ndarray::{arr1, arr2, Array2};
use ny_core::Result;
use ny_tensor::BoundedTensor;
use ny_test_utils::CountingGemmEngine;

use super::*;
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::engine::graph::input_split::shared::graph_spec_ibp_fallback;
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::engine::tensor_ext::BoundedTensorExt;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::bounds::LinearBounds;
use crate::layers::{Layer, LinearLayer};
use crate::{GraphNetwork, GraphNode};

fn build_single_objective_batch_graph_4353() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "out",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("valid output linear")),
    ));
    graph.set_output("out");
    graph
}

fn unresolved_parent_domain_4353() -> GraphInputDomain {
    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("finite parent bounds");
    GraphInputDomain {
        input_bounds: Arc::new(input_bounds),
        lower_bound: -1.0,
        upper_bound: 1.0,
        depth: 0,
        priority: 1.0,
        linear_bounds: None,
        needs_bounding: false,
        node_bounds_override: None,
        inherited_alpha_state: None,
    }
}

fn single_objective_baseline_gemm_calls_4353(
    graph: &GraphNetwork,
    spec_matrix: &Array2<f32>,
    threshold: f32,
) -> usize {
    let left_child = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[0.0_f32]).into_dyn())
        .expect("finite left child");
    let right_child = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("finite right child");
    let engine = CountingGemmEngine::new();
    let baseline_verified: Vec<bool> = [&left_child, &right_child]
        .into_iter()
        .map(|child| {
            let (bounds, _) =
                graph_spec_ibp_fallback(graph, child, spec_matrix, Some(&engine), None)
                    .expect("per-child IBP fallback should succeed");
            BetaCrownConfig::domain_is_verified_for_mode(
                false,
                bounds.lower_scalar(),
                bounds.upper_scalar(),
                threshold,
            )
        })
        .collect();
    assert_eq!(
        baseline_verified,
        vec![false, true],
        "baseline split children should leave exactly one unresolved domain"
    );
    engine.gemm_calls()
}

fn assert_queued_single_child_4353(queue: &mut BinaryHeap<GraphInputDomain>) {
    let child = queue.pop().expect("one unresolved child should be queued");
    assert!(child.needs_bounding);
    assert_eq!(child.depth, 1);
    assert_eq!(child.lower_bound, -1.0);
    assert_eq!(child.upper_bound, 1.0);
    assert!(child.linear_bounds.is_none());
    assert!(child.node_bounds_override.is_none());
    assert_eq!(child.input_bounds.lower()[[0]], -1.0);
    assert_eq!(child.input_bounds.upper()[[0]], 0.0);
}

#[test]
fn test_process_single_objective_domain_batch_reorder_batches_ibp_prescreen_4353() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        reorder_bab: true,
        input_split_ibp_enhancement: true,
        enable_relaxed_clip: false,
        ..Default::default()
    });
    let graph = build_single_objective_batch_graph_4353();
    let spec_matrix = arr2(&[[1.0_f32]]);
    let threshold = -0.1_f32;
    let baseline_calls = single_objective_baseline_gemm_calls_4353(&graph, &spec_matrix, threshold);

    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut domains_verified_by_ibp = 0usize;
    let mut domains_screened_by_crown = 0usize;
    let batched_engine = CountingGemmEngine::new();

    let result = process_single_objective_domain_batch(
        &verifier,
        &graph,
        vec![unresolved_parent_domain_4353()],
        &[1.0_f32],
        threshold,
        &spec_matrix,
        Some(&batched_engine),
        &|_input, _node_bounds| -> Result<(f32, f32, Option<LinearBounds>)> {
            panic!("reorder batched prescreen should not call compute_bounds")
        },
        None,
        Duration::from_secs(1),
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_ibp,
        &mut domains_screened_by_crown,
    )
    .expect("batched single-objective processing should not error");

    assert!(result.is_none());
    assert_eq!(domains_verified_by_ibp, 1);
    assert_eq!(domains_screened_by_crown, 0);
    assert_eq!(lifecycle.domains_verified, 1);
    assert_eq!(queue.len(), 1);
    assert_queued_single_child_4353(&mut queue);
    assert!(
        batched_engine.gemm_calls() < baseline_calls,
        "batched single-objective process path should reduce GEMM dispatches: batched={}, baseline={}",
        batched_engine.gemm_calls(),
        baseline_calls
    );
}
