// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};
use std::time::Duration;

use ny_core::{NaiveCpuGemmEngine, Result};
use ny_tensor::BoundedTensor;

use crate::beta_crown::{
    BabVerificationStatus, BetaCrownConfig, GraphDomainBatchCallerLane,
    GraphDomainBatchMetricsSink, GraphDomainBatchRecord,
};
use crate::BetaCrownVerifier;
use crate::{GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer};

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

fn build_multi_objective_metrics_graph_4398() -> (GraphNetwork, BoundedTensor) {
    let linear1 = LinearLayer::new(ndarray::Array2::eye(1), None).expect("identity linear");
    let linear2 = LinearLayer::new(
        ndarray::arr2(&[[1.0_f32], [-1.0_f32]]),
        Some(ndarray::arr1(&[0.5_f32, 0.5_f32])),
    )
    .expect("two-output linear");

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

    let input = BoundedTensor::new(
        ndarray::arr1(&[-1.0_f32]).into_dyn(),
        ndarray::arr1(&[1.0_f32]).into_dyn(),
    )
    .expect("valid bounded input");

    (graph, input)
}

#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_emits_graph_domain_batch_records_4398() {
    let (graph, input) = build_multi_objective_metrics_graph_4398();
    let sink = Arc::new(CollectingGraphDomainBatchSink::default());
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        timeout: Duration::from_secs(5),
        max_domains: 100,
        max_depth: 10,
        batch_size: 4,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        ..Default::default()
    })
    .with_graph_domain_batch_metrics_sink(sink.clone());
    let engine = NaiveCpuGemmEngine;
    let objectives = vec![vec![-1.0_f32, 0.0_f32], vec![0.0_f32, -1.0_f32]];
    let thresholds = vec![-0.55_f32, -0.55_f32];

    let result = verifier
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            Some(&engine),
            None,
        )
        .expect("engine-backed multi-objective BaB should complete");

    assert!(
        matches!(result.result, BabVerificationStatus::Unknown { .. }),
        "metrics harness should exercise the disjunctive batch path, got {:?}",
        result.result
    );

    let records = sink
        .records
        .lock()
        .expect("graph-domain-batch sink records");
    assert!(
        !records.is_empty(),
        "multi-objective BaB should emit shared domain-batch records"
    );
    assert!(
        records
            .iter()
            .all(|record| record.caller_lane == GraphDomainBatchCallerLane::MultiObjective),
        "disjunctive multi-objective runs should report the multi_objective caller lane"
    );
    assert!(
        records.iter().any(|record| record.domains_batched > 0),
        "expected at least one shared-executor batch record, got {records:?}"
    );
}
