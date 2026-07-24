// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared batch-summary types for disjunctive input-split observability.

use ny_core::Result;

/// Dense-spec rebound execution mode for one input-split batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DenseSpecReboundMode {
    /// The popped batch already carried fresh bounds, so rebound did no work.
    NoDeferredDomains,
    /// Dense-spec rebound used the batched backward fast-path.
    BatchedFastPath,
    /// Dense-spec rebound fell back to rayon-parallel per-domain bounds.
    RayonFallback,
    /// Dense-spec rebound used only override-path recomputation.
    OverrideOnly,
}

impl DenseSpecReboundMode {
    /// Stable schema string for JSONL sidecars.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoDeferredDomains => "no_deferred_domains",
            Self::BatchedFastPath => "batched_fast_path",
            Self::RayonFallback => "rayon_fallback",
            Self::OverrideOnly => "override_only",
        }
    }
}

/// Timing data for the dense-spec rebound phase of one input-split batch.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub(crate) struct DenseSpecReboundTiming {
    pub(crate) mode: DenseSpecReboundMode,
    pub(crate) domains: usize,
    pub(crate) num_specs: usize,
    pub(crate) total_elapsed_s: f64,
    pub(crate) forward_elapsed_s: Option<f64>,
    pub(crate) backward_elapsed_s: Option<f64>,
    pub(crate) materialize_elapsed_s: Option<f64>,
}

impl DenseSpecReboundTiming {
    #[must_use]
    pub(crate) fn no_deferred_domains(num_specs: usize) -> Self {
        Self {
            mode: DenseSpecReboundMode::NoDeferredDomains,
            domains: 0,
            num_specs,
            total_elapsed_s: 0.0,
            forward_elapsed_s: None,
            backward_elapsed_s: None,
            materialize_elapsed_s: None,
        }
    }

    #[must_use]
    pub(crate) fn override_only(domains: usize, num_specs: usize, total_elapsed_s: f64) -> Self {
        Self {
            mode: DenseSpecReboundMode::OverrideOnly,
            domains,
            num_specs,
            total_elapsed_s,
            forward_elapsed_s: None,
            backward_elapsed_s: None,
            materialize_elapsed_s: None,
        }
    }

    #[must_use]
    pub(crate) fn with_total_elapsed(
        mut self,
        domains: usize,
        num_specs: usize,
        total_elapsed_s: f64,
    ) -> Self {
        self.domains = domains;
        self.num_specs = num_specs;
        self.total_elapsed_s = total_elapsed_s;
        self
    }

    #[must_use]
    pub(crate) fn rebound_other_elapsed_s(&self) -> f64 {
        let accounted = self.forward_elapsed_s.unwrap_or(0.0)
            + self.backward_elapsed_s.unwrap_or(0.0)
            + self.materialize_elapsed_s.unwrap_or(0.0);
        (self.total_elapsed_s - accounted).max(0.0)
    }
}

/// Human-readable batch summary emitted by the grouped disjunctive input-split loop.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub(crate) struct InputSplitBatchSummary {
    pub(crate) batch_index: usize,
    pub(crate) queue_len_before_pop: usize,
    pub(crate) queue_len_after_batch: usize,
    pub(crate) popped_domains: usize,
    pub(crate) domains_explored_after_batch: usize,
    pub(crate) domains_verified_in_batch: usize,
    pub(crate) domains_clipped_in_batch: usize,
    pub(crate) rebound: DenseSpecReboundTiming,
    pub(crate) split_screen_elapsed_s: f64,
}

impl InputSplitBatchSummary {
    #[must_use]
    pub(crate) fn batch_total_elapsed_s(&self) -> f64 {
        self.rebound.total_elapsed_s + self.split_screen_elapsed_s
    }

    #[must_use]
    pub(crate) fn domains_per_second(&self) -> f64 {
        let batch_total_s = self.batch_total_elapsed_s();
        if batch_total_s > 0.0 {
            self.popped_domains as f64 / batch_total_s
        } else {
            0.0
        }
    }

    #[must_use]
    pub(crate) fn to_record(&self) -> InputSplitBatchRecord {
        InputSplitBatchRecord {
            batch_index: self.batch_index,
            queue_len_before_pop: self.queue_len_before_pop,
            queue_len_after_batch: self.queue_len_after_batch,
            popped_domains: self.popped_domains,
            domains_explored_after_batch: self.domains_explored_after_batch,
            domains_verified_in_batch: self.domains_verified_in_batch,
            domains_clipped_in_batch: self.domains_clipped_in_batch,
            rebound_mode: self.rebound.mode,
            rebound_total_s: self.rebound.total_elapsed_s,
            forward_s: self.rebound.forward_elapsed_s,
            backward_s: self.rebound.backward_elapsed_s,
            materialize_s: self.rebound.materialize_elapsed_s,
            rebound_other_s: self.rebound.rebound_other_elapsed_s(),
            split_screen_s: self.split_screen_elapsed_s,
            batch_total_s: self.batch_total_elapsed_s(),
            domains_per_second: self.domains_per_second(),
        }
    }
}

/// Durable record emitted to runtime sinks for one logged input-split batch.
#[derive(Debug, Clone, PartialEq)]
pub struct InputSplitBatchRecord {
    pub batch_index: usize,
    pub queue_len_before_pop: usize,
    pub queue_len_after_batch: usize,
    pub popped_domains: usize,
    pub domains_explored_after_batch: usize,
    pub domains_verified_in_batch: usize,
    pub domains_clipped_in_batch: usize,
    pub rebound_mode: DenseSpecReboundMode,
    pub rebound_total_s: f64,
    pub forward_s: Option<f64>,
    pub backward_s: Option<f64>,
    pub materialize_s: Option<f64>,
    pub rebound_other_s: f64,
    pub split_screen_s: f64,
    pub batch_total_s: f64,
    pub domains_per_second: f64,
}

impl InputSplitBatchRecord {
    pub const SCHEMA_VERSION: &str = "input_split_batch_metrics_v1";
    pub const RECORD_KIND: &str = "batch_summary";

    #[must_use]
    pub const fn schema_version() -> &'static str {
        Self::SCHEMA_VERSION
    }

    #[must_use]
    pub const fn record_kind() -> &'static str {
        Self::RECORD_KIND
    }
}

/// Runtime sink for durable input-split batch summaries.
pub trait InputSplitMetricsSink: Send + Sync {
    fn record_batch_summary(&self, record: &InputSplitBatchRecord) -> Result<()>;
}

/// Log the first three batches and every 10th batch after that.
#[must_use]
pub(crate) const fn should_log_batch(batch_index: usize) -> bool {
    batch_index < 3 || (batch_index + 1).is_multiple_of(10)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fast_path_timing() -> DenseSpecReboundTiming {
        DenseSpecReboundTiming {
            mode: DenseSpecReboundMode::BatchedFastPath,
            domains: 4,
            num_specs: 2,
            total_elapsed_s: 1.5,
            forward_elapsed_s: Some(0.4),
            backward_elapsed_s: Some(0.8),
            materialize_elapsed_s: Some(0.2),
        }
    }

    #[test]
    fn test_domains_per_second_zero_elapsed_is_safe_4350() {
        let summary = InputSplitBatchSummary {
            batch_index: 0,
            queue_len_before_pop: 1,
            queue_len_after_batch: 0,
            popped_domains: 4,
            domains_explored_after_batch: 4,
            domains_verified_in_batch: 1,
            domains_clipped_in_batch: 0,
            rebound: DenseSpecReboundTiming::no_deferred_domains(2),
            split_screen_elapsed_s: 0.0,
        };

        assert_eq!(summary.domains_per_second(), 0.0);
    }

    #[test]
    fn test_rebound_other_elapsed_clamps_negative_drift_4350() {
        let timing = DenseSpecReboundTiming {
            total_elapsed_s: 1.0,
            forward_elapsed_s: Some(0.6),
            backward_elapsed_s: Some(0.3),
            materialize_elapsed_s: Some(0.2),
            ..fast_path_timing()
        };

        assert_eq!(timing.rebound_other_elapsed_s(), 0.0);
    }

    #[test]
    fn test_should_log_batch_matches_cadence_4350() {
        assert!(should_log_batch(0));
        assert!(should_log_batch(1));
        assert!(should_log_batch(2));
        assert!(!should_log_batch(3));
        assert!(!should_log_batch(8));
        assert!(should_log_batch(9));
        assert!(should_log_batch(19));
    }

    #[test]
    fn test_batch_summary_to_record_preserves_fast_path_timing_4357() {
        let summary = InputSplitBatchSummary {
            batch_index: 9,
            queue_len_before_pop: 8,
            queue_len_after_batch: 6,
            popped_domains: 4,
            domains_explored_after_batch: 20,
            domains_verified_in_batch: 2,
            domains_clipped_in_batch: 1,
            rebound: fast_path_timing(),
            split_screen_elapsed_s: 0.5,
        };

        let record = summary.to_record();

        assert_eq!(record.batch_index, 9);
        assert_eq!(record.rebound_mode, DenseSpecReboundMode::BatchedFastPath);
        assert_eq!(record.rebound_total_s, 1.5);
        assert_eq!(record.forward_s, Some(0.4));
        assert_eq!(record.backward_s, Some(0.8));
        assert_eq!(record.materialize_s, Some(0.2));
        assert!((record.rebound_other_s - 0.1).abs() < 1e-9);
        assert_eq!(record.batch_total_s, 2.0);
        assert_eq!(record.domains_per_second, 2.0);
    }

    #[test]
    fn test_override_only_record_4350() {
        let summary = InputSplitBatchSummary {
            batch_index: 3,
            queue_len_before_pop: 4,
            queue_len_after_batch: 3,
            popped_domains: 1,
            domains_explored_after_batch: 4,
            domains_verified_in_batch: 0,
            domains_clipped_in_batch: 0,
            rebound: DenseSpecReboundTiming::override_only(1, 2, 0.25),
            split_screen_elapsed_s: 0.75,
        };

        let record = summary.to_record();

        assert_eq!(record.rebound_mode, DenseSpecReboundMode::OverrideOnly);
        assert_eq!(record.forward_s, None);
        assert_eq!(record.backward_s, None);
        assert_eq!(record.materialize_s, None);
        assert_eq!(record.rebound_other_s, 0.25);
        assert_eq!(record.batch_total_s, 1.0);
    }

    #[test]
    fn test_rayon_fallback_record_4350() {
        let summary = InputSplitBatchSummary {
            batch_index: 4,
            queue_len_before_pop: 5,
            queue_len_after_batch: 2,
            popped_domains: 3,
            domains_explored_after_batch: 7,
            domains_verified_in_batch: 1,
            domains_clipped_in_batch: 0,
            rebound: DenseSpecReboundTiming {
                mode: DenseSpecReboundMode::RayonFallback,
                domains: 3,
                num_specs: 2,
                total_elapsed_s: 0.9,
                forward_elapsed_s: None,
                backward_elapsed_s: None,
                materialize_elapsed_s: None,
            },
            split_screen_elapsed_s: 0.1,
        };

        let record = summary.to_record();

        assert_eq!(record.rebound_mode, DenseSpecReboundMode::RayonFallback);
        assert_eq!(record.rebound_other_s, 0.9);
        assert_eq!(record.batch_total_s, 1.0);
        assert_eq!(record.domains_per_second, 3.0);
    }
}
