// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared ownership helpers for graph domain-batch execution.

use std::collections::BTreeMap;

use ny_core::Result;

use crate::beta_crown::branching::BranchingHeuristic;
use crate::beta_crown::engine::graph::input_split::metrics::{
    DenseSpecReboundMode, DenseSpecReboundTiming,
};

use super::metrics::{
    add_fallback_reason_count, GraphDomainBatchCallerLane, GraphDomainBatchFallbackReason,
    GraphDomainBatchMetricsSink, GraphDomainBatchRecord,
};

/// Shared execution mode for one graph-domain batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::beta_crown::engine::graph) enum GraphDomainBatchExecutionMode {
    SharedExecutor,
    ParallelFallback,
    SequentialFallback,
}

/// Branching semantics required by the ReLU-split loop's current configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReluSplitBranchingSemantics {
    Relu,
    GeneralNonlinear,
}

impl From<&BranchingHeuristic> for ReluSplitBranchingSemantics {
    fn from(heuristic: &BranchingHeuristic) -> Self {
        if matches!(heuristic, BranchingHeuristic::GenBaB(_)) {
            Self::GeneralNonlinear
        } else {
            Self::Relu
        }
    }
}

/// Complete eligibility context for one batch from the ReLU-split loop.
///
/// The shared single-objective executor implements only ReLU branching. Keeping
/// the branching semantics in this typed context prevents a GenBaB run from
/// silently entering that executor and dropping its general-nonlinearity split
/// candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::beta_crown::engine::graph) struct ReluSplitBatchContext {
    target_batch_size: usize,
    engine_available: bool,
    has_active_cuts: bool,
    branching: ReluSplitBranchingSemantics,
}

impl ReluSplitBatchContext {
    #[must_use]
    pub(in crate::beta_crown::engine::graph) fn new(
        target_batch_size: usize,
        engine_available: bool,
        has_active_cuts: bool,
        branching_heuristic: &BranchingHeuristic,
    ) -> Self {
        Self {
            target_batch_size,
            engine_available,
            has_active_cuts,
            branching: branching_heuristic.into(),
        }
    }
}

/// Timing payload used when emitting one graph-domain batch record.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(in crate::beta_crown::engine::graph) struct GraphDomainBatchEmitTiming {
    pub(in crate::beta_crown::engine::graph) forward_s: Option<f64>,
    pub(in crate::beta_crown::engine::graph) backward_s: Option<f64>,
    pub(in crate::beta_crown::engine::graph) materialize_s: Option<f64>,
    pub(in crate::beta_crown::engine::graph) queue_update_s: Option<f64>,
    pub(in crate::beta_crown::engine::graph) total_s: f64,
}

impl GraphDomainBatchEmitTiming {
    #[must_use]
    pub(in crate::beta_crown::engine::graph) const fn new(total_s: f64) -> Self {
        Self {
            forward_s: None,
            backward_s: None,
            materialize_s: None,
            queue_update_s: None,
            total_s,
        }
    }

    #[must_use]
    pub(in crate::beta_crown::engine::graph) const fn with_queue_update(
        mut self,
        queue_update_s: f64,
    ) -> Self {
        self.queue_update_s = Some(queue_update_s);
        self
    }

    #[must_use]
    pub(in crate::beta_crown::engine::graph) fn from_dense_spec(
        rebound_timing: &DenseSpecReboundTiming,
    ) -> Self {
        Self {
            forward_s: rebound_timing.forward_elapsed_s,
            backward_s: rebound_timing.backward_elapsed_s,
            materialize_s: rebound_timing.materialize_elapsed_s,
            queue_update_s: None,
            total_s: rebound_timing.total_elapsed_s,
        }
    }
}

/// Shared ownership surface for batch eligibility and record assembly.
#[derive(Debug, Clone, PartialEq)]
pub(in crate::beta_crown::engine::graph) struct GraphDomainBatchPlan {
    execution_mode: GraphDomainBatchExecutionMode,
    caller_lane: GraphDomainBatchCallerLane,
    batch_index: usize,
    domains_popped: usize,
    domains_batched: usize,
    domains_fallback: usize,
    batch_width: usize,
    fallback_reason_counts: BTreeMap<String, usize>,
}

impl GraphDomainBatchPlan {
    #[must_use]
    pub(in crate::beta_crown::engine::graph) fn for_relu_split(
        batch_index: usize,
        batch_width: usize,
        context: ReluSplitBatchContext,
    ) -> Self {
        let mut fallback_reason_counts = BTreeMap::new();
        let execution_mode = if context.target_batch_size > 1 && !context.has_active_cuts {
            match (context.engine_available, context.branching) {
                (true, ReluSplitBranchingSemantics::Relu) => {
                    GraphDomainBatchExecutionMode::SharedExecutor
                }
                (true, ReluSplitBranchingSemantics::GeneralNonlinear) => {
                    add_fallback_reason_count(
                        &mut fallback_reason_counts,
                        GraphDomainBatchFallbackReason::GeneralNonlinearBranching.as_str(),
                        batch_width,
                    );
                    GraphDomainBatchExecutionMode::SequentialFallback
                }
                (false, _) => {
                    add_fallback_reason_count(
                        &mut fallback_reason_counts,
                        GraphDomainBatchFallbackReason::NoEngine.as_str(),
                        batch_width,
                    );
                    GraphDomainBatchExecutionMode::ParallelFallback
                }
            }
        } else {
            let reason = if context.has_active_cuts {
                GraphDomainBatchFallbackReason::CutsEnabled
            } else {
                GraphDomainBatchFallbackReason::SingletonBatch
            };
            add_fallback_reason_count(&mut fallback_reason_counts, reason.as_str(), batch_width);
            GraphDomainBatchExecutionMode::SequentialFallback
        };

        Self {
            execution_mode,
            caller_lane: GraphDomainBatchCallerLane::ReluSplit,
            batch_index,
            domains_popped: batch_width,
            domains_batched: usize::from(
                execution_mode == GraphDomainBatchExecutionMode::SharedExecutor,
            ) * batch_width,
            domains_fallback: usize::from(
                execution_mode != GraphDomainBatchExecutionMode::SharedExecutor,
            ) * batch_width,
            batch_width,
            fallback_reason_counts,
        }
    }

    #[must_use]
    pub(in crate::beta_crown::engine::graph) fn for_multi_objective(
        batch_index: usize,
        batch_width: usize,
        batch_size: usize,
        engine_available: bool,
        conjunctive: bool,
    ) -> Self {
        let caller_lane = if conjunctive {
            GraphDomainBatchCallerLane::MultiObjectiveConjunctive
        } else {
            GraphDomainBatchCallerLane::MultiObjective
        };
        let mut fallback_reason_counts = BTreeMap::new();
        let execution_mode = if engine_available && batch_size > 1 && !conjunctive {
            GraphDomainBatchExecutionMode::SharedExecutor
        } else {
            let reason = if !engine_available {
                GraphDomainBatchFallbackReason::NoEngine
            } else if batch_size <= 1 {
                GraphDomainBatchFallbackReason::SingletonBatch
            } else {
                GraphDomainBatchFallbackReason::UnsupportedCallerMode
            };
            add_fallback_reason_count(&mut fallback_reason_counts, reason.as_str(), batch_width);
            GraphDomainBatchExecutionMode::SequentialFallback
        };

        Self {
            execution_mode,
            caller_lane,
            batch_index,
            domains_popped: batch_width,
            domains_batched: usize::from(
                execution_mode == GraphDomainBatchExecutionMode::SharedExecutor,
            ) * batch_width,
            domains_fallback: usize::from(
                execution_mode != GraphDomainBatchExecutionMode::SharedExecutor,
            ) * batch_width,
            batch_width,
            fallback_reason_counts,
        }
    }

    #[must_use]
    pub(in crate::beta_crown::engine::graph) fn for_dense_spec_rebound(
        batch_index: usize,
        deferred_count: usize,
        batched_count: usize,
        override_count: usize,
        rebound_timing: &DenseSpecReboundTiming,
    ) -> Self {
        let mut fallback_reason_counts = BTreeMap::new();
        let (execution_mode, domains_batched, domains_fallback, batch_width) =
            match rebound_timing.mode {
                DenseSpecReboundMode::BatchedFastPath => {
                    add_fallback_reason_count(
                        &mut fallback_reason_counts,
                        GraphDomainBatchFallbackReason::OverrideOnly.as_str(),
                        override_count,
                    );
                    (
                        GraphDomainBatchExecutionMode::SharedExecutor,
                        batched_count,
                        override_count,
                        batched_count,
                    )
                }
                DenseSpecReboundMode::RayonFallback => {
                    add_fallback_reason_count(
                        &mut fallback_reason_counts,
                        GraphDomainBatchFallbackReason::UnsupportedCallerMode.as_str(),
                        batched_count,
                    );
                    add_fallback_reason_count(
                        &mut fallback_reason_counts,
                        GraphDomainBatchFallbackReason::OverrideOnly.as_str(),
                        override_count,
                    );
                    (
                        GraphDomainBatchExecutionMode::SequentialFallback,
                        0,
                        batched_count + override_count,
                        batched_count,
                    )
                }
                DenseSpecReboundMode::OverrideOnly => {
                    add_fallback_reason_count(
                        &mut fallback_reason_counts,
                        GraphDomainBatchFallbackReason::OverrideOnly.as_str(),
                        override_count,
                    );
                    (
                        GraphDomainBatchExecutionMode::SequentialFallback,
                        0,
                        override_count,
                        0,
                    )
                }
                DenseSpecReboundMode::NoDeferredDomains => {
                    (GraphDomainBatchExecutionMode::SequentialFallback, 0, 0, 0)
                }
            };

        Self {
            execution_mode,
            caller_lane: GraphDomainBatchCallerLane::InputSplitDenseSpec,
            batch_index,
            domains_popped: deferred_count,
            domains_batched,
            domains_fallback,
            batch_width,
            fallback_reason_counts,
        }
    }

    #[must_use]
    pub(in crate::beta_crown::engine::graph) const fn execution_mode(
        &self,
    ) -> GraphDomainBatchExecutionMode {
        self.execution_mode
    }

    #[must_use]
    pub(in crate::beta_crown::engine::graph) fn build_record(
        &self,
        timing: GraphDomainBatchEmitTiming,
    ) -> GraphDomainBatchRecord {
        GraphDomainBatchRecord {
            batch_index: self.batch_index,
            caller_lane: self.caller_lane,
            domains_popped: self.domains_popped,
            domains_batched: self.domains_batched,
            domains_fallback: self.domains_fallback,
            batch_width: self.batch_width,
            forward_s: timing.forward_s,
            backward_s: timing.backward_s,
            materialize_s: timing.materialize_s,
            queue_update_s: timing.queue_update_s,
            total_s: timing.total_s,
            fallback_reason_counts: self.fallback_reason_counts.clone(),
        }
    }

    pub(in crate::beta_crown::engine::graph) fn emit_to_sink(
        &self,
        sink: Option<&dyn GraphDomainBatchMetricsSink>,
        timing: GraphDomainBatchEmitTiming,
    ) -> Result<()> {
        if let Some(sink) = sink {
            sink.record_batch_summary(&self.build_record(timing))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_split_plan_no_engine_preserves_parallel_fallback_4398() {
        let heuristic = BranchingHeuristic::LargestBoundWidth;
        let context = ReluSplitBatchContext::new(8, false, false, &heuristic);
        let plan = GraphDomainBatchPlan::for_relu_split(4, 3, context);

        assert_eq!(
            plan.execution_mode(),
            GraphDomainBatchExecutionMode::ParallelFallback
        );

        let record = plan.build_record(GraphDomainBatchEmitTiming::new(1.25));
        assert_eq!(record.caller_lane, GraphDomainBatchCallerLane::ReluSplit);
        assert_eq!(record.domains_batched, 0);
        assert_eq!(record.domains_fallback, 3);
        assert_eq!(record.fallback_reason_counts.get("no_engine"), Some(&3));
    }

    #[test]
    fn test_relu_split_plan_genbab_bypasses_relu_only_shared_executor() {
        let heuristic = BranchingHeuristic::GenBaB(Default::default());
        let context = ReluSplitBatchContext::new(8, true, false, &heuristic);
        let plan = GraphDomainBatchPlan::for_relu_split(0, 8, context);

        assert_eq!(
            plan.execution_mode(),
            GraphDomainBatchExecutionMode::SequentialFallback
        );

        let record = plan.build_record(GraphDomainBatchEmitTiming::new(0.5));
        assert_eq!(record.domains_batched, 0);
        assert_eq!(record.domains_fallback, 8);
        assert_eq!(
            record
                .fallback_reason_counts
                .get("general_nonlinear_branching"),
            Some(&8)
        );
    }

    #[test]
    fn test_relu_split_plan_relu_keeps_shared_executor() {
        let heuristic = BranchingHeuristic::LargestBoundWidth;
        let context = ReluSplitBatchContext::new(8, true, false, &heuristic);
        let plan = GraphDomainBatchPlan::for_relu_split(0, 8, context);

        assert_eq!(
            plan.execution_mode(),
            GraphDomainBatchExecutionMode::SharedExecutor
        );
        assert!(plan
            .build_record(GraphDomainBatchEmitTiming::new(0.5))
            .fallback_reason_counts
            .is_empty());
    }

    #[test]
    fn test_relu_split_plan_genbab_without_engine_keeps_parallel_fallback() {
        let heuristic = BranchingHeuristic::GenBaB(Default::default());
        let context = ReluSplitBatchContext::new(8, false, false, &heuristic);
        let plan = GraphDomainBatchPlan::for_relu_split(0, 8, context);

        assert_eq!(
            plan.execution_mode(),
            GraphDomainBatchExecutionMode::ParallelFallback
        );
        let record = plan.build_record(GraphDomainBatchEmitTiming::new(0.5));
        assert_eq!(record.fallback_reason_counts.get("no_engine"), Some(&8));
        assert!(!record
            .fallback_reason_counts
            .contains_key("general_nonlinear_branching"));
    }

    #[test]
    fn test_multi_objective_plan_conjunctive_uses_unsupported_reason_4398() {
        let plan = GraphDomainBatchPlan::for_multi_objective(2, 5, 8, true, true);

        assert_eq!(
            plan.execution_mode(),
            GraphDomainBatchExecutionMode::SequentialFallback
        );

        let record = plan.build_record(GraphDomainBatchEmitTiming::new(0.9).with_queue_update(0.2));
        assert_eq!(
            record.caller_lane,
            GraphDomainBatchCallerLane::MultiObjectiveConjunctive
        );
        assert_eq!(record.domains_batched, 0);
        assert_eq!(record.domains_fallback, 5);
        assert_eq!(
            record.fallback_reason_counts.get("unsupported_caller_mode"),
            Some(&5)
        );
        assert_eq!(record.queue_update_s, Some(0.2));
    }

    #[test]
    fn test_multi_objective_plan_disjunctive_with_engine_uses_shared_executor() {
        let plan = GraphDomainBatchPlan::for_multi_objective(0, 5, 8, true, false);

        assert_eq!(
            plan.execution_mode(),
            GraphDomainBatchExecutionMode::SharedExecutor
        );
        let record = plan.build_record(GraphDomainBatchEmitTiming::new(0.5));
        assert_eq!(record.domains_batched, 5);
        assert_eq!(record.domains_fallback, 0);
        assert!(record.fallback_reason_counts.is_empty());
    }

    #[test]
    fn test_multi_objective_plan_disjunctive_without_engine_uses_legacy_fallback() {
        let plan = GraphDomainBatchPlan::for_multi_objective(0, 5, 8, false, false);

        assert_eq!(
            plan.execution_mode(),
            GraphDomainBatchExecutionMode::SequentialFallback
        );
        let record = plan.build_record(GraphDomainBatchEmitTiming::new(0.5));
        assert_eq!(record.domains_batched, 0);
        assert_eq!(record.domains_fallback, 5);
        assert_eq!(record.fallback_reason_counts.get("no_engine"), Some(&5));
    }

    #[test]
    fn test_dense_spec_plan_fast_path_tracks_override_remainder_4398() {
        let rebound_timing = DenseSpecReboundTiming {
            mode: DenseSpecReboundMode::BatchedFastPath,
            domains: 5,
            num_specs: 2,
            total_elapsed_s: 1.4,
            forward_elapsed_s: Some(0.2),
            backward_elapsed_s: Some(0.3),
            materialize_elapsed_s: Some(0.1),
        };
        let plan = GraphDomainBatchPlan::for_dense_spec_rebound(7, 5, 3, 2, &rebound_timing);
        let record =
            plan.build_record(GraphDomainBatchEmitTiming::from_dense_spec(&rebound_timing));

        assert_eq!(
            plan.execution_mode(),
            GraphDomainBatchExecutionMode::SharedExecutor
        );
        assert_eq!(
            record.caller_lane,
            GraphDomainBatchCallerLane::InputSplitDenseSpec
        );
        assert_eq!(record.domains_popped, 5);
        assert_eq!(record.domains_batched, 3);
        assert_eq!(record.domains_fallback, 2);
        assert_eq!(record.batch_width, 3);
        assert_eq!(record.forward_s, Some(0.2));
        assert_eq!(record.backward_s, Some(0.3));
        assert_eq!(record.materialize_s, Some(0.1));
        assert_eq!(record.fallback_reason_counts.get("override_only"), Some(&2));
    }
}
