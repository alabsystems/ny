// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Domain-batch metrics helpers for DomainList BaB.

use ny_core::Result;

use crate::beta_crown::engine::graph::domain_batch::{
    GraphDomainBatchEmitTiming, GraphDomainBatchMetricsSink, GraphDomainBatchPlan,
};
use crate::beta_crown::engine::graph::input_split::metrics::DenseSpecReboundTiming;

/// Emit the shared dense-spec rebound summary for one GPU DomainList input-split batch.
pub(crate) fn emit_input_split_rebound_metrics(
    metrics_sink: Option<&dyn GraphDomainBatchMetricsSink>,
    batch_index: usize,
    deferred_count: usize,
    batched_count: usize,
    override_count: usize,
    rebound_timing: &DenseSpecReboundTiming,
) -> Result<()> {
    if deferred_count == 0 {
        return Ok(());
    }

    GraphDomainBatchPlan::for_dense_spec_rebound(
        batch_index,
        deferred_count,
        batched_count,
        override_count,
        rebound_timing,
    )
    .emit_to_sink(
        metrics_sink,
        GraphDomainBatchEmitTiming::from_dense_spec(rebound_timing),
    )
}
