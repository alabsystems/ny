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

fn build_relu_split_metrics_graph_4398() -> (GraphNetwork, BoundedTensor) {
    let linear1 = LinearLayer::new(ndarray::arr2(&[[1.0_f32]]), None).expect("identity linear");
    let linear2 = LinearLayer::new(ndarray::arr2(&[[1.0_f32]]), None).expect("identity output");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        ndarray::arr1(&[-1.0_f32]).into_dyn(),
        ndarray::arr1(&[1.0_f32]).into_dyn(),
    )
    .expect("valid bounded input");

    (graph, input)
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_split_emits_graph_domain_batch_records_4398() {
    let (graph, input) = build_relu_split_metrics_graph_4398();
    let sink = Arc::new(CollectingGraphDomainBatchSink::default());
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        timeout: Duration::from_secs(5),
        max_domains: 32,
        max_depth: 4,
        batch_size: 2,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        ..Default::default()
    })
    .with_graph_domain_batch_metrics_sink(sink.clone());
    let engine = NaiveCpuGemmEngine;

    let result = verifier
        .verify_graph_relu_split_with_engine_gpu(
            &graph,
            &input,
            &[1.0_f32],
            0.0,
            Some(&engine),
            None,
        )
        .expect("engine-backed relu-split BaB should complete");

    assert!(
        !matches!(result.result, BabVerificationStatus::Verified),
        "metrics harness should exercise the relu-split batch path, got {:?}",
        result.result
    );

    let records = sink
        .records
        .lock()
        .expect("graph-domain-batch sink records");
    assert!(
        !records.is_empty(),
        "relu-split BaB should emit shared domain-batch records"
    );
    assert!(
        records
            .iter()
            .all(|record| record.caller_lane == GraphDomainBatchCallerLane::ReluSplit),
        "relu-split runs should report the relu_split caller lane"
    );
    assert!(
        records.iter().any(|record| record.domains_batched > 0),
        "expected at least one shared-executor batch record, got {records:?}"
    );
}
