// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! CPU fallback execution path for DomainList BaB.
//!
//! When no GPU engine is available (`engine=None`), domains are materialized
//! and processed in parallel on the CPU using rayon. Each domain goes through
//! the sequential `process_graph_domain_parallel` path.
//!
//! Extracted from `verify_graph_gpu_domain_list` lines 728-813.

use ny_core::Result;
use tracing::warn;

use crate::batched_domain::PickedDomains;
use crate::beta_crown::engine::domain_results::GraphDomainResult;
use crate::beta_crown::engine::graph::domain_conversion::graph_domain_from_picked;
use crate::beta_crown::BetaCrownVerifier;
use crate::beta_crown::GraphBabDomain;
use crate::faer_parallelism::RayonTaskGuard;
use crate::GraphNetwork;

use super::check::{check_domain_bounds, BabLoopState, DomainCheckResult};

/// Outcome of CPU fallback processing.
pub(crate) enum CpuFallbackOutcome {
    /// Processing completed; child_domains contains surviving children.
    Children(Vec<GraphBabDomain>),
    /// A violation was found; return immediately.
    Violation,
}

fn handle_no_unstable_result(
    lower: f32,
    upper: f32,
    verified: bool,
    threshold: f32,
    verify_upper_bound: bool,
    state: &mut BabLoopState,
) -> Option<CpuFallbackOutcome> {
    if verified {
        state.domains_verified += 1;
        return None;
    }

    match check_domain_bounds(lower, upper, threshold, verify_upper_bound) {
        DomainCheckResult::Verified => {
            state.domains_verified += 1;
            None
        }
        DomainCheckResult::Violation => Some(CpuFallbackOutcome::Violation),
        DomainCheckResult::Undecided => {
            state.unresolved_due_to_no_unstable_neurons = true;
            None
        }
    }
}

/// Process a batch of picked domains using CPU parallel evaluation.
///
/// Materializes domains from `PickedDomains`, processes them in parallel
/// via `process_graph_domain_parallel`, then collects surviving children.
///
/// # Returns
/// `CpuFallbackOutcome::Violation` if any domain proves a violation,
/// `CpuFallbackOutcome::Children(children)` otherwise with unverified child domains.
// These parameters are the exact set of values needed from the caller's scope.
// Grouping into a context struct would just add indirection without reducing complexity.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_cpu_fallback_batch(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    picked: &PickedDomains,
    processable_picked_indices: &[usize],
    layer_names: &[String],
    relu_nodes: &[String],
    objective: &[f32],
    threshold: f32,
    state: &mut BabLoopState,
) -> Result<CpuFallbackOutcome> {
    use rayon::prelude::*;

    let domains_to_process: Vec<GraphBabDomain> = processable_picked_indices
        .iter()
        .map(|&idx| {
            graph_domain_from_picked(
                idx,
                picked,
                layer_names,
                verifier.config.verify_upper_bound,
                Some(graph),
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let results: Vec<_> = domains_to_process
        .par_iter()
        .map(|domain| {
            let _rayon_task_guard = RayonTaskGuard::new();
            verifier.process_graph_domain_parallel(
                graph, domain, relu_nodes, objective, threshold, None, 1,
            )
        })
        .collect();

    let mut child_domains: Vec<GraphBabDomain> = Vec::new();

    for result in results {
        match result {
            GraphDomainResult::AlreadyVerified => {
                state.domains_verified += 1;
            }
            GraphDomainResult::Violation => {
                return Ok(CpuFallbackOutcome::Violation);
            }
            GraphDomainResult::Children(children) => {
                for (child, verified) in children {
                    state.max_depth_reached = state.max_depth_reached.max(child.depth);
                    if child.depth >= verifier.config.max_depth {
                        state.unresolved_due_to_depth = true;
                        continue;
                    }
                    if verified {
                        state.domains_verified += 1;
                    } else {
                        child_domains.push(child);
                    }
                }
            }
            GraphDomainResult::NoUnstable {
                lower,
                upper,
                verified,
            } => {
                if let Some(outcome) = handle_no_unstable_result(
                    lower,
                    upper,
                    verified,
                    threshold,
                    verifier.config.verify_upper_bound,
                    state,
                ) {
                    return Ok(outcome);
                }
            }
            GraphDomainResult::PropagationFailure => {
                // #1852: propagation failed — sub-region unexplored.
                // Must not return Verified while any domain has this status.
                warn!(
                    "DomainList BaB: propagation failure in sequential fallback path — domain unresolved"
                );
                state.unresolved_due_to_propagation_failure = true;
            }
        }
    }

    Ok(CpuFallbackOutcome::Children(child_domains))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Instant;

    use ndarray::{arr2, ArrayD, IxDyn};

    use super::{
        check_domain_bounds, handle_no_unstable_result, process_cpu_fallback_batch, BabLoopState,
        CpuFallbackOutcome, DomainCheckResult,
    };
    use crate::batched_domain::{DomainMetadata, PickedDomains};
    use crate::beta_crown::result::BabVerificationStatus;
    use crate::beta_crown::{BetaCrownConfig, BetaCrownVerifier, BranchingHeuristic};
    use crate::{GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer};

    #[test]
    fn test_handle_no_unstable_verified_by_bound_check_increments_verified_count() {
        let mut state = BabLoopState::new(Instant::now());
        let outcome = handle_no_unstable_result(0.0, 0.5, false, 1.0, true, &mut state);

        assert!(
            outcome.is_none(),
            "verified domain should not return violation"
        );
        assert_eq!(
            state.domains_verified, 1,
            "bound-verified NoUnstable domain must increment domains_verified",
        );
        assert!(
            !state.unresolved_due_to_no_unstable_neurons,
            "bound-verified NoUnstable domain must not be marked unresolved",
        );
    }

    #[test]
    fn test_handle_no_unstable_undecided_marks_unresolved() {
        let mut state = BabLoopState::new(Instant::now());
        let outcome = handle_no_unstable_result(0.5, 1.5, false, 1.0, true, &mut state);

        assert!(
            outcome.is_none(),
            "undecided domain should not early-return"
        );
        assert_eq!(state.domains_verified, 0);
        assert!(state.unresolved_due_to_no_unstable_neurons);
    }

    #[test]
    fn test_handle_no_unstable_violation_returns_violation() {
        let mut state = BabLoopState::new(Instant::now());
        let outcome = handle_no_unstable_result(1.5, 2.0, false, 1.0, true, &mut state);

        assert!(
            matches!(outcome, Some(CpuFallbackOutcome::Violation)),
            "violating domain should return violation outcome",
        );
        assert_eq!(state.domains_verified, 0);
        assert!(!state.unresolved_due_to_no_unstable_neurons);
    }

    /// Regression test for #2004: PropagationFailure must set
    /// `unresolved_due_to_propagation_failure`, NOT `unresolved_due_to_no_unstable_neurons`.
    /// The bug caused build_final_result() to report "No unstable ReLU/Sign neurons"
    /// instead of "Child propagation failed" — misdirecting debugging.
    ///
    /// Drives `process_cpu_fallback_batch` with a picked batch that materializes
    /// to a domain with no stored node bounds under the `BoundImpact` heuristic,
    /// so branch-score computation inside `process_graph_domain_parallel` fails
    /// for real (#1915) and the batch's PropagationFailure arm runs.
    #[ntest::timeout(10000)]
    #[test]
    fn test_propagation_failure_sets_correct_unresolved_flag() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            branching_heuristic: BranchingHeuristic::BoundImpact,
            ..Default::default()
        });

        // relu(2 unstable input neurons) -> linear1(1x2). BoundImpact scoring
        // needs the producer bounds that the picked batch below does not carry.
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::new(
            "linear1",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0, 1.0]]), None).expect("valid linear layer")),
            vec!["relu".to_string()],
        ));
        graph.set_output("linear1");

        // Empty layer_names + empty layer bounds: the materialized domain has an
        // empty node-bounds map, forcing the branch-selection failure.
        let picked = PickedDomains {
            batch_size: 1,
            layer_lowers: HashMap::new(),
            layer_uppers: HashMap::new(),
            input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![-1.0_f32, -1.0])
                .expect("valid picked lower bounds"),
            input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0_f32, 1.0])
                .expect("valid picked upper bounds"),
            global_lbs: vec![-1.0],
            global_ubs: vec![1.0],
            metadata: vec![DomainMetadata::root(-1.0, 1.0).expect("valid root metadata")],
        };
        let mut state = BabLoopState::new(Instant::now());

        let outcome = process_cpu_fallback_batch(
            &verifier,
            &graph,
            &picked,
            &[0],
            &[],
            &["relu".to_string()],
            &[1.0_f32],
            0.0,
            &mut state,
        )
        .expect("a propagation failure must be recorded, not abort the batch");

        // A failed domain yields no children and must not short-circuit the loop.
        match outcome {
            CpuFallbackOutcome::Children(children) => {
                assert!(
                    children.is_empty(),
                    "a failed domain must not produce children, got {}",
                    children.len(),
                );
            }
            CpuFallbackOutcome::Violation => {
                unreachable!("a propagation failure must not be reported as a violation")
            }
        }
        assert_eq!(
            state.domains_verified, 0,
            "a failed domain must never count as verified",
        );

        // The correct flag must be set, not the no_branch flag.
        assert!(
            state.unresolved_due_to_propagation_failure,
            "PropagationFailure must set unresolved_due_to_propagation_failure",
        );
        assert!(
            !state.unresolved_due_to_no_unstable_neurons,
            "PropagationFailure must NOT set unresolved_due_to_no_unstable_neurons (#2004)",
        );

        // Verify the final result reason string matches the propagation failure path.
        let final_result = state.build_final_result();
        match final_result.result {
            BabVerificationStatus::Unknown { reason } => {
                assert!(
                    reason.contains("Child propagation failed"),
                    "reason must mention propagation failure, got: {reason}",
                );
                assert!(
                    !reason.contains("No unstable ReLU/Sign neurons"),
                    "reason must NOT mention no-branch (#2004 bug), got: {reason}",
                );
            }
            other => unreachable!("expected Unknown result, got {other:?}"),
        }
    }

    #[test]
    fn test_handle_no_unstable_matches_gpu_bound_check_semantics_for_verified_case() {
        let lower = 0.0;
        let upper = 0.5;
        let threshold = 1.0;
        let verify_upper_bound = true;
        let mut state = BabLoopState::new(Instant::now());

        // GPU path (`batched_gpu.rs`) checks bounds and increments verified count
        // when `DomainCheckResult::Verified`.
        let expected = check_domain_bounds(lower, upper, threshold, verify_upper_bound);
        assert!(matches!(expected, DomainCheckResult::Verified));

        let outcome = handle_no_unstable_result(
            lower,
            upper,
            false,
            threshold,
            verify_upper_bound,
            &mut state,
        );

        assert!(
            outcome.is_none(),
            "bound-verified NoUnstable domain should not return violation",
        );
        assert_eq!(
            state.domains_verified, 1,
            "CPU fallback must match GPU semantics for bound-verified NoUnstable domains",
        );
        assert!(
            !state.unresolved_due_to_no_unstable_neurons,
            "CPU fallback must not classify bound-verified NoUnstable domains as unresolved",
        );
    }
}
