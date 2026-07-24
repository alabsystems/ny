// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared batch-summary types for graph domain-batch executor observability.

use std::collections::BTreeMap;

use ny_core::Result;

/// Stable caller-lane taxonomy for shared graph domain-batch metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GraphDomainBatchCallerLane {
    /// ReLU-split graph BaB.
    ReluSplit,
    /// Dense-spec rebound in graph input split.
    InputSplitDenseSpec,
    /// Disjunctive multi-objective graph BaB.
    MultiObjective,
    /// Conjunctive multi-objective graph BaB.
    MultiObjectiveConjunctive,
}

impl GraphDomainBatchCallerLane {
    /// Stable schema string for JSONL sidecars.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReluSplit => "relu_split",
            Self::InputSplitDenseSpec => "input_split_dense_spec",
            Self::MultiObjective => "multi_objective",
            Self::MultiObjectiveConjunctive => "multi_objective_conjunctive",
        }
    }
}

/// Stable executor-local reasons for per-domain fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GraphDomainBatchFallbackReason {
    /// No `GemmEngine` was available.
    NoEngine,
    /// The popped batch was too small to amortize the shared executor.
    SingletonBatch,
    /// Cuts forced the batch through a non-shared path.
    CutsEnabled,
    /// The caller mode is not yet supported by the shared executor.
    UnsupportedCallerMode,
    /// Child-local node-bound overrides forced per-domain rebound.
    OverrideOnly,
}

impl GraphDomainBatchFallbackReason {
    /// Stable schema string for JSONL sidecars.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoEngine => "no_engine",
            Self::SingletonBatch => "singleton_batch",
            Self::CutsEnabled => "cuts_enabled",
            Self::UnsupportedCallerMode => "unsupported_caller_mode",
            Self::OverrideOnly => "override_only",
        }
    }
}

/// Durable record emitted to runtime sinks for one logged graph-domain batch.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphDomainBatchRecord {
    pub batch_index: usize,
    pub caller_lane: GraphDomainBatchCallerLane,
    pub domains_popped: usize,
    pub domains_batched: usize,
    pub domains_fallback: usize,
    pub batch_width: usize,
    pub forward_s: Option<f64>,
    pub backward_s: Option<f64>,
    pub materialize_s: Option<f64>,
    pub queue_update_s: Option<f64>,
    pub total_s: f64,
    pub fallback_reason_counts: BTreeMap<String, usize>,
}

impl GraphDomainBatchRecord {
    pub const SCHEMA_VERSION: &str = "graph_domain_batch_metrics_v1";
    pub const RECORD_KIND: &str = "batch_summary";

    #[must_use]
    pub const fn schema_version() -> &'static str {
        Self::SCHEMA_VERSION
    }

    #[must_use]
    pub const fn record_kind() -> &'static str {
        Self::RECORD_KIND
    }

    #[must_use]
    pub fn batch_share(&self) -> f64 {
        if self.domains_popped == 0 {
            0.0
        } else {
            self.domains_batched as f64 / self.domains_popped as f64
        }
    }

    #[must_use]
    pub fn fallback_share(&self) -> f64 {
        if self.domains_popped == 0 {
            0.0
        } else {
            self.domains_fallback as f64 / self.domains_popped as f64
        }
    }

    #[must_use]
    pub fn executor_other_s(&self) -> f64 {
        let accounted = self.forward_s.unwrap_or(0.0)
            + self.backward_s.unwrap_or(0.0)
            + self.materialize_s.unwrap_or(0.0)
            + self.queue_update_s.unwrap_or(0.0);
        (self.total_s - accounted).max(0.0)
    }
}

/// Runtime sink for durable graph domain-batch summaries.
pub trait GraphDomainBatchMetricsSink: Send + Sync {
    fn record_batch_summary(&self, record: &GraphDomainBatchRecord) -> Result<()>;
}

/// Increment one fallback-reason counter by `count`.
pub(crate) fn add_fallback_reason_count(
    counts: &mut BTreeMap<String, usize>,
    reason: impl Into<String>,
    count: usize,
) {
    if count == 0 {
        return;
    }
    *counts.entry(reason.into()).or_insert(0) += count;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record() -> GraphDomainBatchRecord {
        let mut fallback_reason_counts = BTreeMap::new();
        add_fallback_reason_count(
            &mut fallback_reason_counts,
            GraphDomainBatchFallbackReason::CutsEnabled.as_str(),
            1,
        );
        GraphDomainBatchRecord {
            batch_index: 2,
            caller_lane: GraphDomainBatchCallerLane::ReluSplit,
            domains_popped: 4,
            domains_batched: 3,
            domains_fallback: 1,
            batch_width: 3,
            forward_s: Some(0.2),
            backward_s: Some(0.3),
            materialize_s: Some(0.1),
            queue_update_s: Some(0.15),
            total_s: 1.0,
            fallback_reason_counts,
        }
    }

    #[test]
    fn test_graph_domain_batch_record_shares_are_safe_4398() {
        let mut record = sample_record();
        assert_eq!(record.batch_share(), 0.75);
        assert_eq!(record.fallback_share(), 0.25);

        record.domains_popped = 0;
        assert_eq!(record.batch_share(), 0.0);
        assert_eq!(record.fallback_share(), 0.0);
    }

    #[test]
    fn test_graph_domain_batch_record_executor_other_clamps_negative_4398() {
        let mut record = sample_record();
        record.total_s = 0.5;

        assert_eq!(record.executor_other_s(), 0.0);
    }

    #[test]
    fn test_add_fallback_reason_count_accumulates_4398() {
        let mut counts = BTreeMap::new();
        add_fallback_reason_count(
            &mut counts,
            GraphDomainBatchFallbackReason::NoEngine.as_str(),
            2,
        );
        add_fallback_reason_count(
            &mut counts,
            GraphDomainBatchFallbackReason::NoEngine.as_str(),
            3,
        );
        add_fallback_reason_count(&mut counts, "custom_reason", 0);

        assert_eq!(counts.get("no_engine"), Some(&5));
        assert!(!counts.contains_key("custom_reason"));
    }
}
