// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use ny_core::{NyError, Result as NyResult};
use ny_propagate::{InputSplitBatchRecord, InputSplitMetricsSink};
use serde_json::json;

#[derive(Debug)]
pub(super) struct JsonlInputSplitMetricsSink {
    path: PathBuf,
    writer: Mutex<BufWriter<File>>,
}

impl JsonlInputSplitMetricsSink {
    fn open(path: &Path) -> Result<Self> {
        let file = File::create(path).with_context(|| {
            format!(
                "failed to create input-split metrics JSONL sidecar at {}",
                path.display()
            )
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            writer: Mutex::new(BufWriter::new(file)),
        })
    }

    pub(super) fn create(path: &Path) -> Result<Arc<dyn InputSplitMetricsSink>> {
        Ok(Arc::new(Self::open(path)?))
    }

    fn record_json(record: &InputSplitBatchRecord) -> serde_json::Value {
        json!({
            "schema_version": InputSplitBatchRecord::schema_version(),
            "record_kind": InputSplitBatchRecord::record_kind(),
            "batch_index": record.batch_index,
            "queue_len_before_pop": record.queue_len_before_pop,
            "queue_len_after_batch": record.queue_len_after_batch,
            "popped_domains": record.popped_domains,
            "domains_explored_after_batch": record.domains_explored_after_batch,
            "domains_verified_in_batch": record.domains_verified_in_batch,
            "domains_clipped_in_batch": record.domains_clipped_in_batch,
            "rebound_mode": record.rebound_mode.as_str(),
            "rebound_total_s": record.rebound_total_s,
            "forward_s": record.forward_s,
            "backward_s": record.backward_s,
            "materialize_s": record.materialize_s,
            "rebound_other_s": record.rebound_other_s,
            "split_screen_s": record.split_screen_s,
            "batch_total_s": record.batch_total_s,
            "domains_per_second": record.domains_per_second,
        })
    }
}

impl InputSplitMetricsSink for JsonlInputSplitMetricsSink {
    fn record_batch_summary(&self, record: &InputSplitBatchRecord) -> NyResult<()> {
        let mut writer = self.writer.lock().map_err(|err| {
            NyError::InternalError(format!(
                "input-split metrics writer mutex poisoned for {}: {}",
                self.path.display(),
                err
            ))
        })?;
        serde_json::to_writer(&mut *writer, &Self::record_json(record)).map_err(|err| {
            NyError::InvalidConfig(format!(
                "failed to serialize input-split metrics record for {}: {}",
                self.path.display(),
                err
            ))
        })?;
        writer.write_all(b"\n").map_err(|err| {
            NyError::InvalidConfig(format!(
                "failed to write input-split metrics record to {}: {}",
                self.path.display(),
                err
            ))
        })?;
        writer.flush().map_err(|err| {
            NyError::InvalidConfig(format!(
                "failed to flush input-split metrics record to {}: {}",
                self.path.display(),
                err
            ))
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ny_propagate::DenseSpecReboundMode;
    use tempfile::tempdir;

    use super::*;

    fn sample_record() -> InputSplitBatchRecord {
        InputSplitBatchRecord {
            batch_index: 2,
            queue_len_before_pop: 5,
            queue_len_after_batch: 3,
            popped_domains: 4,
            domains_explored_after_batch: 12,
            domains_verified_in_batch: 1,
            domains_clipped_in_batch: 0,
            rebound_mode: DenseSpecReboundMode::RayonFallback,
            rebound_total_s: 0.75,
            forward_s: None,
            backward_s: None,
            materialize_s: None,
            rebound_other_s: 0.75,
            split_screen_s: 0.25,
            batch_total_s: 1.0,
            domains_per_second: 4.0,
        }
    }

    #[test]
    fn test_jsonl_input_split_metrics_sink_writes_batch_record_4357() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("input_split_metrics.jsonl");
        let sink = JsonlInputSplitMetricsSink::open(&path).expect("sink should open");

        sink.record_batch_summary(&sample_record())
            .expect("sink should write record");

        let contents = std::fs::read_to_string(&path).expect("sidecar should exist");
        let value: serde_json::Value =
            serde_json::from_str(contents.trim()).expect("record should parse as JSON");

        assert_eq!(
            value["schema_version"],
            InputSplitBatchRecord::schema_version()
        );
        assert_eq!(value["record_kind"], InputSplitBatchRecord::record_kind());
        assert_eq!(value["rebound_mode"], "rayon_fallback");
        assert_eq!(value["batch_total_s"], 1.0);
    }

    #[test]
    fn test_jsonl_input_split_metrics_sink_fails_fast_for_missing_parent_4357() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("missing").join("input_split_metrics.jsonl");

        let err = JsonlInputSplitMetricsSink::open(&path)
            .expect_err("missing parent directory should fail immediately");

        assert!(
            err.to_string()
                .contains("failed to create input-split metrics JSONL sidecar"),
            "unexpected error: {err}"
        );
    }
}
