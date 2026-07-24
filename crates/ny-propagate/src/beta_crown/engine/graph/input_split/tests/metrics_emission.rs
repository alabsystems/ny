// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};
use std::time::Duration;

use ndarray::{arr1, arr2};
use ny_core::Result;
use ny_tensor::BoundedTensor;

use super::*;
use crate::beta_crown::engine::GraphDomainBatchMetricsSink;
use crate::beta_crown::{
    BabVerificationStatus, BetaCrownConfig, BranchingHeuristic, DenseSpecReboundMode,
    GraphDomainBatchCallerLane, GraphDomainBatchRecord, InputSplitBatchRecord,
    InputSplitMetricsSink,
};
use crate::BetaCrownVerifier;

#[derive(Default)]
struct CollectingMetricsSink {
    records: Mutex<Vec<InputSplitBatchRecord>>,
}

impl InputSplitMetricsSink for CollectingMetricsSink {
    fn record_batch_summary(&self, record: &InputSplitBatchRecord) -> Result<()> {
        self.records
            .lock()
            .expect("collecting sink mutex")
            .push(record.clone());
        Ok(())
    }
}

#[derive(Default)]
struct CollectingGraphDomainBatchSink {
    records: Mutex<Vec<GraphDomainBatchRecord>>,
}

impl GraphDomainBatchMetricsSink for CollectingGraphDomainBatchSink {
    fn record_batch_summary(&self, record: &GraphDomainBatchRecord) -> Result<()> {
        self.records
            .lock()
            .expect("graph-domain-batch sink mutex")
            .push(record.clone());
        Ok(())
    }
}

fn build_disjunctive_metrics_graph_4357() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "out",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("identity linear")),
    ));
    graph.set_output("out");
    graph
}

fn build_single_objective_metrics_graph_4398() -> GraphNetwork {
    build_disjunctive_metrics_graph_4357()
}

#[test]
fn test_disjunctive_input_split_emits_metrics_records_4357() {
    let graph = build_disjunctive_metrics_graph_4357();
    let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid bounded input");
    let objectives = vec![vec![1.0_f32], vec![-1.0_f32]];
    let thresholds = vec![0.4_f32, 0.4_f32];
    let clause_sizes = vec![1usize, 1usize];
    let sink = Arc::new(CollectingMetricsSink::default());

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        enable_relaxed_clip: false,
        input_split_ibp_enhancement: false,
        max_domains: 64,
        max_depth: 1,
        batch_size: 4,
        timeout: Duration::from_secs(5),
        reorder_bab: true,
        ..Default::default()
    })
    .with_input_split_metrics_sink(sink.clone());

    let result = verifier
        .verify_graph_input_split_multi_clause_disjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            &clause_sizes,
            None,
            None,
        )
        .expect("grouped input split should complete");

    assert!(
        !matches!(result.result, BabVerificationStatus::Verified),
        "metrics harness should exercise the split path, not verify at the root"
    );

    let records = sink.records.lock().expect("collected records");
    assert!(
        !records.is_empty(),
        "direct grouped input split path should emit at least one metrics record"
    );
    assert_eq!(records[0].batch_index, 0);
    assert_eq!(records[0].queue_len_before_pop, 1);
    assert!(
        records.iter().any(|record| {
            matches!(
                record.rebound_mode,
                DenseSpecReboundMode::BatchedFastPath | DenseSpecReboundMode::RayonFallback
            )
        }),
        "expected a later record with real rebound timing, got {records:?}"
    );
}

#[test]
fn test_single_objective_input_split_emits_graph_domain_batch_records_4398() {
    let graph = build_single_objective_metrics_graph_4398();
    let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid bounded input");
    let sink = Arc::new(CollectingGraphDomainBatchSink::default());

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        enable_relaxed_clip: false,
        input_split_ibp_enhancement: false,
        max_domains: 64,
        max_depth: 2,
        batch_size: 4,
        timeout: Duration::from_secs(5),
        reorder_bab: true,
        ..Default::default()
    })
    .with_graph_domain_batch_metrics_sink(sink.clone());

    let result = verifier
        .verify_graph_input_split_with_engine(&graph, &input, &[1.0_f32], -0.5, None, None)
        .expect("single-objective input split should complete");

    assert!(
        !matches!(result.result, BabVerificationStatus::Verified),
        "metrics harness should exercise deferred rebound instead of verifying at the root"
    );

    let records = sink.records.lock().expect("collected graph-domain records");
    assert!(
        !records.is_empty(),
        "single-objective input split should emit at least one shared domain-batch record"
    );
    assert_eq!(records[0].batch_index, 1);
    assert!(
        records
            .iter()
            .all(|record| record.caller_lane == GraphDomainBatchCallerLane::InputSplitDenseSpec),
        "single-objective input split should reuse the dense-spec rebound lane"
    );
    assert!(
        records.iter().any(|record| record.domains_batched > 0),
        "expected at least one deferred rebound batch with batched domains, got {records:?}"
    );
}
