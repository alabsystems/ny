// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use ny_core::{NyError, Result as NyResult};
use ny_propagate::{GraphDomainBatchMetricsSink, GraphDomainBatchRecord};
use serde_json::json;

#[derive(Debug)]
pub(super) struct JsonlGraphDomainBatchMetricsSink {
    path: PathBuf,
    writer: Mutex<BufWriter<File>>,
}

impl JsonlGraphDomainBatchMetricsSink {
    fn open(path: &Path) -> Result<Self> {
        let file = File::create(path).with_context(|| {
            format!(
                "failed to create graph domain-batch metrics JSONL sidecar at {}",
                path.display()
            )
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            writer: Mutex::new(BufWriter::new(file)),
        })
    }

    pub(super) fn create(path: &Path) -> Result<Arc<dyn GraphDomainBatchMetricsSink>> {
        Ok(Arc::new(Self::open(path)?))
    }

    fn record_json(record: &GraphDomainBatchRecord) -> serde_json::Value {
        json!({
            "schema_version": GraphDomainBatchRecord::schema_version(),
            "record_kind": GraphDomainBatchRecord::record_kind(),
            "batch_index": record.batch_index,
            "caller_lane": record.caller_lane.as_str(),
            "domains_popped": record.domains_popped,
            "domains_batched": record.domains_batched,
            "domains_fallback": record.domains_fallback,
            "batch_width": record.batch_width,
            "forward_s": record.forward_s,
            "backward_s": record.backward_s,
            "materialize_s": record.materialize_s,
            "queue_update_s": record.queue_update_s,
            "total_s": record.total_s,
            "batch_share": record.batch_share(),
            "fallback_share": record.fallback_share(),
            "executor_other_s": record.executor_other_s(),
            "fallback_reason_counts": record.fallback_reason_counts,
        })
    }
}

impl GraphDomainBatchMetricsSink for JsonlGraphDomainBatchMetricsSink {
    fn record_batch_summary(&self, record: &GraphDomainBatchRecord) -> NyResult<()> {
        let mut writer = self.writer.lock().map_err(|err| {
            NyError::InternalError(format!(
                "graph domain-batch metrics writer mutex poisoned for {}: {}",
                self.path.display(),
                err
            ))
        })?;
        serde_json::to_writer(&mut *writer, &Self::record_json(record)).map_err(|err| {
            NyError::InvalidConfig(format!(
                "failed to serialize graph domain-batch metrics record for {}: {}",
                self.path.display(),
                err
            ))
        })?;
        writer.write_all(b"\n").map_err(|err| {
            NyError::InvalidConfig(format!(
                "failed to write graph domain-batch metrics record to {}: {}",
                self.path.display(),
                err
            ))
        })?;
        writer.flush().map_err(|err| {
            NyError::InvalidConfig(format!(
                "failed to flush graph domain-batch metrics record to {}: {}",
                self.path.display(),
                err
            ))
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ny_propagate::GraphDomainBatchCallerLane;
    use tempfile::tempdir;

    use super::*;

    fn sample_record() -> GraphDomainBatchRecord {
        let mut fallback_reason_counts = BTreeMap::new();
        fallback_reason_counts.insert("singleton_batch".to_string(), 1);
        GraphDomainBatchRecord {
            batch_index: 1,
            caller_lane: GraphDomainBatchCallerLane::ReluSplit,
            domains_popped: 2,
            domains_batched: 0,
            domains_fallback: 2,
            batch_width: 2,
            forward_s: None,
            backward_s: None,
            materialize_s: None,
            queue_update_s: Some(0.25),
            total_s: 1.0,
            fallback_reason_counts,
        }
    }

    #[test]
    fn test_jsonl_graph_domain_batch_metrics_sink_writes_batch_record_4398() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("graph_domain_batch_metrics.jsonl");
        let sink = JsonlGraphDomainBatchMetricsSink::open(&path).expect("sink should open");

        sink.record_batch_summary(&sample_record())
            .expect("sink should write record");

        let contents = std::fs::read_to_string(&path).expect("sidecar should exist");
        let value: serde_json::Value =
            serde_json::from_str(contents.trim()).expect("record should parse as JSON");

        assert_eq!(
            value["schema_version"],
            GraphDomainBatchRecord::schema_version()
        );
        assert_eq!(value["record_kind"], GraphDomainBatchRecord::record_kind());
        assert_eq!(value["caller_lane"], "relu_split");
        assert_eq!(value["fallback_reason_counts"]["singleton_batch"], 1);
    }

    #[test]
    fn test_jsonl_graph_domain_batch_metrics_sink_fails_fast_for_missing_parent_4398() {
        let dir = tempdir().expect("tempdir");
        let path = dir
            .path()
            .join("missing")
            .join("graph_domain_batch_metrics.jsonl");

        let err = JsonlGraphDomainBatchMetricsSink::open(&path)
            .expect_err("missing parent directory should fail immediately");

        assert!(
            err.to_string()
                .contains("failed to create graph domain-batch metrics JSONL sidecar"),
            "unexpected error: {err}"
        );
    }
}
