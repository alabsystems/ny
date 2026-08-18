// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Input-split branch path for DomainList BaB.
//!
//! Input splitting branches on input dimensions (bisecting the input domain)
//! rather than fixing ReLU states. Each child gets new input bounds and
//! requires a fresh spec-guided CROWN backward pass to compute output bounds.
//!
//! This is fundamentally different from ReLU splitting:
//! - Children have empty layer bounds (recomputed from scratch each iteration)
//! - The spec matrix is used for a full CROWN backward pass per child
//! - This path does not carry per-domain alpha/beta state
//!
//! That last point is specific to ny's current DomainList input-split
//! implementation. The alpha-beta-CROWN reference only carries child alpha in
//! input-split mode when the config selects `solver.bound_prop_method:
//! alpha-crown` (or `solver.init_bound_prop_method: alpha-crown`). The shipped
//! ACAS-Xu / lsnc_relu / cifar_biasfield input-split configs use
//! `solver.bound_prop_method: crown`, so their reference runs also recompute
//! child bounds without per-child alpha re-optimization. See
//! `complete_verifier/input_split/batch_branch_and_bound.py` (`use_alpha`) and
//! the benchmark configs under `complete_verifier/exp_configs/vnncomp*/`.
//!
//! Extracted from `verify_graph_gpu_domain_list` lines 229-383.
//!
//! Part of #1891, Phase 1.
//! Reference: designs/2026-02-10-input-split-domain-list.md §1.4

use ny_core::{GemmEngine, Result};
use tracing::{info, warn};

use crate::batched_domain::{DomainList, PickedDomains, ProcessedDomains};
use crate::beta_crown::engine::graph::input_split::adv_check::{
    try_adv_check_on_input_bounds_batch, ADV_CHECK_INTERVAL,
};
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::GraphNetwork;

use super::check::{check_domain_bounds, BabLoopState, DomainCheckResult};
use super::init::InputSplitBootstrap;
use super::input_split_support::{build_parent_contexts, screen_child_domain, ChildDomainAction};

/// Outcome of processing one input-split iteration.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum InputSplitOutcome {
    /// All children processed; loop should continue.
    Continue,
    /// A violation was found; return immediately.
    Violation,
}

enum InputSplitChildSink<'a> {
    Immediate(&'a mut DomainList),
    Staged(&'a mut Vec<ProcessedDomains>),
}

impl InputSplitChildSink<'_> {
    fn queue(&mut self, child: ProcessedDomains) -> Result<()> {
        match self {
            Self::Immediate(domain_list) => domain_list.add(child),
            Self::Staged(children) => {
                children.push(child);
                Ok(())
            }
        }
    }
}

/// Staged effects of one input-split microbatch.
///
/// Allocation/dispatch errors during computation leave the caller's lifecycle
/// and DomainList untouched, so the same picked range can be retried after
/// controller backoff. Effects are committed only after the full computation
/// succeeds. `commit` intentionally uses `DomainList::add`'s historical
/// fail-closed behavior: it is not a rollback boundary for a later queue
/// storage allocation failure. Such a failure can leave earlier child appends
/// in place without installing the staged lifecycle, and is never retried.
pub(crate) struct InputSplitBatchEffects {
    pub(crate) outcome: InputSplitOutcome,
    state: BabLoopState,
    queued_children: Vec<ProcessedDomains>,
    batch_size: usize,
}

impl InputSplitBatchEffects {
    pub(crate) fn commit(
        self,
        state: &mut BabLoopState,
        domain_list: &mut DomainList,
    ) -> Result<InputSplitOutcome> {
        for child in self.queued_children {
            domain_list.add(child)?;
        }
        *state = self.state;
        if self.outcome == InputSplitOutcome::Continue {
            log_input_split_iteration(state, domain_list, self.batch_size);
        }
        Ok(self.outcome)
    }
}

fn log_input_split_iteration(state: &BabLoopState, domain_list: &DomainList, batch_size: usize) {
    info!(
        "DomainList BaB input split iteration: explored={}, verified={}, remaining={}, batch_size={}",
        state.domains_explored,
        state.domains_verified,
        domain_list.len(),
        batch_size
    );
}

/// Process a batch of picked domains using input-split branching.
///
/// For each processable domain:
/// 1. Reuse stored parent bounds or run the deferred parent bound pass
/// 2. Select an input dimension using SB scoring when linear bounds are available
/// 3. Create two children with bisected input bounds
/// 4. Screen children through clipping / IBP enhancement
/// 5. Either queue children for deferred bounding or compute fresh CROWN/IBP bounds
/// 6. Check verification/violation/depth and add surviving children to `DomainList`
///
/// # Returns
/// `InputSplitOutcome::Violation` if any child domain proves a violation,
/// `InputSplitOutcome::Continue` otherwise.
// These parameters are the exact set of values needed from the caller's scope.
// Grouping into a context struct would just add indirection without reducing complexity.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_input_split_batch(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    picked: &PickedDomains,
    processable_picked_indices: &[usize],
    objective: &[f32],
    bootstrap: &InputSplitBootstrap,
    threshold: f32,
    engine: Option<&dyn GemmEngine>,
    state: &mut BabLoopState,
    domain_list: &mut DomainList,
    batch_index: usize,
) -> Result<InputSplitOutcome> {
    // Keep the historical immediate state/queue mutation path intact. The
    // independently gated adaptive caller uses the staged attempt below.
    let outcome = {
        let mut child_sink = InputSplitChildSink::Immediate(domain_list);
        process_input_split_batch_inner(
            verifier,
            graph,
            picked,
            processable_picked_indices,
            objective,
            bootstrap,
            threshold,
            engine,
            state,
            &mut child_sink,
            batch_index,
        )?
    };
    if outcome == InputSplitOutcome::Continue {
        log_input_split_iteration(state, domain_list, picked.batch_size);
    }
    Ok(outcome)
}

/// Stage a complete input-split computation without mutating its queue or
/// lifecycle. This is the adaptive route's allocation/dispatch retry boundary;
/// queue-storage commit remains on the legacy fail-closed path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_input_split_batch_attempt(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    picked: &PickedDomains,
    processable_picked_indices: &[usize],
    objective: &[f32],
    bootstrap: &InputSplitBootstrap,
    threshold: f32,
    engine: Option<&dyn GemmEngine>,
    state: &BabLoopState,
    batch_index: usize,
) -> Result<InputSplitBatchEffects> {
    let mut staged_state = state.clone();
    let mut queued_children = Vec::new();
    let outcome = {
        let mut child_sink = InputSplitChildSink::Staged(&mut queued_children);
        process_input_split_batch_inner(
            verifier,
            graph,
            picked,
            processable_picked_indices,
            objective,
            bootstrap,
            threshold,
            engine,
            &mut staged_state,
            &mut child_sink,
            batch_index,
        )?
    };
    Ok(InputSplitBatchEffects {
        outcome,
        state: staged_state,
        queued_children,
        batch_size: processable_picked_indices.len(),
    })
}

#[allow(clippy::too_many_arguments)]
fn process_input_split_batch_inner(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    picked: &PickedDomains,
    processable_picked_indices: &[usize],
    objective: &[f32],
    bootstrap: &InputSplitBootstrap,
    threshold: f32,
    engine: Option<&dyn GemmEngine>,
    state: &mut BabLoopState,
    child_sink: &mut InputSplitChildSink<'_>,
    batch_index: usize,
) -> Result<InputSplitOutcome> {
    use super::super::domain_conversion::branch_input_split_from_picked;
    use super::metrics::emit_input_split_rebound_metrics;

    let parent_context_build = build_parent_contexts(
        verifier,
        graph,
        picked,
        processable_picked_indices,
        bootstrap,
        engine,
    )?;
    emit_input_split_rebound_metrics(
        verifier.graph_domain_batch_metrics_sink(),
        batch_index,
        parent_context_build.deferred_count,
        parent_context_build.batched_count,
        parent_context_build.override_count,
        &parent_context_build.rebound_timing,
    )?;
    let parent_contexts = parent_context_build.contexts;

    if verifier.config.adv_check >= 0
        && state.domains_explored >= verifier.config.adv_check as usize
        && state.domains_explored.is_multiple_of(ADV_CHECK_INTERVAL)
        && try_adv_check_on_input_bounds_batch(
            graph,
            parent_contexts
                .iter()
                .map(|parent_context| &parent_context.input_bounds),
            objective,
            threshold,
            verifier.config.verify_upper_bound,
            bootstrap.deadline,
            state.domains_explored as u64,
            engine,
        )?
        .is_some()
    {
        info!(
            "DomainList BaB input split: adv_check found counterexample from picked batch at domain {}",
            state.domains_explored
        );
        return Ok(InputSplitOutcome::Violation);
    }

    for (&picked_idx, parent_context) in processable_picked_indices
        .iter()
        .zip(parent_contexts.iter())
    {
        state.domains_explored += 1;
        let parent_depth = picked
            .metadata
            .get(picked_idx)
            .map(|meta| meta.depth())
            .unwrap_or(0);
        state.max_depth_reached = state.max_depth_reached.max(parent_depth);

        if !parent_context.lower_bound.is_finite() || !parent_context.upper_bound.is_finite() {
            warn!(
                picked_idx,
                lower = parent_context.lower_bound,
                upper = parent_context.upper_bound,
                "DomainList BaB input split: parent domain dropped — non-finite bounds"
            );
            state.unresolved_due_to_propagation_failure = true;
            continue;
        }

        match check_domain_bounds(
            parent_context.lower_bound,
            parent_context.upper_bound,
            threshold,
            verifier.config.verify_upper_bound,
        ) {
            DomainCheckResult::Verified => {
                state.domains_verified += 1;
                continue;
            }
            DomainCheckResult::Violation => {
                return Ok(InputSplitOutcome::Violation);
            }
            DomainCheckResult::Undecided => {}
        }

        if parent_depth >= verifier.config.max_depth {
            state.unresolved_due_to_depth = true;
            continue;
        }

        let active_bound = if verifier.config.verify_upper_bound {
            parent_context.upper_bound
        } else {
            parent_context.lower_bound
        };
        let split_dim = verifier.select_input_dimension_sb(
            &parent_context.input_bounds,
            parent_context.linear_bounds.as_ref(),
            Some(&[active_bound]),
            Some(&[threshold]),
        );

        let flat = parent_context.input_bounds.flatten();
        if split_dim >= flat.len() {
            warn!(
                picked_idx,
                split_dim,
                input_len = flat.len(),
                "DomainList BaB input split: split dimension out of range"
            );
            state.unresolved_due_to_unsplittable = true;
            continue;
        }
        let split_lower = flat.lower()[[split_dim]];
        let split_upper = flat.upper()[[split_dim]];
        if !split_lower.is_finite() || !split_upper.is_finite() || split_upper <= split_lower {
            state.unresolved_due_to_unsplittable = true;
            continue;
        }

        let (left_opt, right_opt) = match branch_input_split_from_picked(
            picked_idx,
            picked,
            split_dim,
            verifier.config.verify_upper_bound,
        ) {
            Ok(result) => result,
            Err(err) => {
                warn!(
                    picked_idx,
                    split_dim,
                    error = %err,
                    "DomainList BaB input split: failed to create children"
                );
                state.unresolved_due_to_propagation_failure = true;
                continue;
            }
        };

        for child_processed in [left_opt, right_opt].into_iter().flatten() {
            match screen_child_domain(
                verifier,
                graph,
                objective,
                bootstrap,
                threshold,
                engine,
                picked_idx,
                split_dim,
                parent_context.lower_bound,
                parent_context.upper_bound,
                parent_context.linear_bounds.as_ref(),
                state,
                child_processed,
            )? {
                ChildDomainAction::Skip => {}
                ChildDomainAction::Queue(child_processed) => child_sink.queue(*child_processed)?,
                ChildDomainAction::Violation => return Ok(InputSplitOutcome::Violation),
            }
        }
    }

    Ok(InputSplitOutcome::Continue)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    use ndarray::{arr2, ArrayD, IxDyn};
    use ny_tensor::{BoundedTensor, TreeTraversal};

    use super::super::check::BabLoopState;
    use super::super::init::InputSplitBootstrap;
    use super::super::input_split_support::{screen_child_domain, ChildDomainAction};
    use super::{process_input_split_batch, InputSplitOutcome};
    use crate::batched_domain::{
        DomainList, DomainListConfig, DomainMetadata, PickedDomains, ProcessedDomains,
    };
    use crate::beta_crown::branching::BranchingHeuristic;
    use crate::beta_crown::config::BetaCrownConfig;
    use crate::beta_crown::result::BabVerificationStatus;
    use crate::beta_crown::BetaCrownVerifier;
    use crate::layers::LinearLayer;
    use crate::network::GraphNode;
    use crate::{GraphNetwork, Layer};

    /// Minimal input-split verifier: no clipping, no IBP enhancement, no
    /// reordering, so `screen_child_domain` goes straight to the child
    /// CROWN/IBP bound computation.
    fn input_split_verifier() -> BetaCrownVerifier {
        BetaCrownVerifier::new(BetaCrownConfig {
            branching_heuristic: BranchingHeuristic::InputSplit,
            use_alpha_crown: false,
            enable_cuts: false,
            enable_relaxed_clip: false,
            input_split_ibp_enhancement: false,
            reorder_bab: false,
            max_domains: 16,
            max_depth: 4,
            timeout: Duration::from_secs(5),
            ..Default::default()
        })
    }

    /// 1 -> 1 identity graph (single input Linear node).
    fn identity_graph() -> GraphNetwork {
        let linear = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("valid identity layer");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
        graph.set_output("linear");
        graph
    }

    fn scalar_bootstrap(
        fixed_node_bounds: Option<HashMap<String, BoundedTensor>>,
    ) -> InputSplitBootstrap {
        InputSplitBootstrap {
            spec_matrix: arr2(&[[1.0_f32]]),
            fixed_node_bounds,
            root_alpha_state: None,
            root_linear_bounds: None,
            mul_binary_alphas: None,
            deadline: None,
        }
    }

    /// Single bisected child over a scalar input box `[lower, upper]`.
    fn scalar_child(lower: f32, upper: f32) -> ProcessedDomains {
        ProcessedDomains {
            layer_lowers: HashMap::new(),
            layer_uppers: HashMap::new(),
            input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![lower])
                .expect("valid child lower array"),
            input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![upper])
                .expect("valid child upper array"),
            global_lbs: vec![-1.0],
            global_ubs: vec![1.0],
            metadata: vec![DomainMetadata::root(-1.0, 1.0).expect("valid child metadata")],
            keep_mask: vec![true],
        }
    }

    /// A dropped domain must surface as `Unknown` with the propagation-failure
    /// reason — NOT the "no unstable neurons" reason (#2102's wrong-flag bug),
    /// and never as `Verified`.
    fn assert_unknown_due_to_propagation_failure(state: &BabLoopState) {
        assert!(
            state.unresolved_due_to_propagation_failure,
            "dropped domain must set unresolved_due_to_propagation_failure",
        );
        assert!(
            !state.unresolved_due_to_no_unstable_neurons,
            "dropped domain must NOT set unresolved_due_to_no_unstable_neurons",
        );
        let final_result = state.build_final_result();
        match final_result.result {
            BabVerificationStatus::Unknown { reason } => {
                assert!(
                    reason.contains("Child propagation failed"),
                    "reason must mention propagation failure, got: {reason}",
                );
                assert!(
                    !reason.contains("No unstable ReLU/Sign neurons"),
                    "reason must NOT mention no-branch, got: {reason}",
                );
            }
            other => unreachable!("expected Unknown result, got {other:?}"),
        }
    }

    /// Regression test for #2102: `BoundedTensor::new` failure on bisected
    /// child input bounds must set `unresolved_due_to_propagation_failure`,
    /// not `unresolved_due_to_no_unstable_neurons`.
    ///
    /// Drives `screen_child_domain` with an INVERTED child box (lower > upper)
    /// so its `BoundedTensor::new` error arm runs for real: the child must be
    /// skipped and the drop mapped to a propagation failure.
    #[ntest::timeout(10000)]
    #[test]
    fn test_input_split_invalid_child_bounds_sets_propagation_failure_flag_2102() {
        let verifier = input_split_verifier();
        let graph = verifier.configured_graph_for_crown(&identity_graph());
        let bootstrap = scalar_bootstrap(None);
        let mut state = BabLoopState::new(Instant::now());

        let action = screen_child_domain(
            &verifier,
            &graph,
            &[1.0_f32],
            &bootstrap,
            0.0,
            None,
            0,
            0,
            -1.0,
            1.0,
            None,
            &mut state,
            scalar_child(1.0, -1.0),
        )
        .expect("invalid child bounds must be dropped, not abort the batch");

        assert!(
            matches!(action, ChildDomainAction::Skip),
            "invalid child bounds must skip the child, not queue it",
        );
        assert_unknown_due_to_propagation_failure(&state);
    }

    /// Regression test for #2922: non-finite parent bounds must set
    /// `unresolved_due_to_propagation_failure` and NOT reach
    /// check_domain_bounds.
    ///
    /// Without the is_finite() guard on the parent context, non-finite bounds
    /// produce Undecided from check_domain_bounds (they fail both is_verified
    /// and is_violation checks), and the domain is re-queued — creating an
    /// infinite BaB loop. The guard mirrors batched_gpu.rs:431.
    ///
    /// Drives `process_input_split_batch` with stored ±inf parent bounds
    /// (`DomainMetadata` admits ±inf; NaN is rejected at construction), so the
    /// production guard runs for real: the domain must be dropped — never
    /// split, re-queued, or folded into a `Verified` result.
    #[ntest::timeout(10000)]
    #[test]
    fn test_input_split_non_finite_crown_bounds_sets_propagation_failure_2922() {
        use super::super::check::{check_domain_bounds, DomainCheckResult};

        // First, confirm the root cause: NaN/Inf always produce Undecided.
        // This is why the is_finite() guard is necessary.
        assert_eq!(
            check_domain_bounds(f32::NAN, 1.0, 0.5, false),
            DomainCheckResult::Undecided,
            "NaN lower must produce Undecided (not Verified or Violation)"
        );
        assert_eq!(
            check_domain_bounds(0.0, f32::NAN, 0.5, false),
            DomainCheckResult::Undecided,
            "NaN upper must produce Undecided"
        );
        assert_eq!(
            check_domain_bounds(f32::INFINITY, f32::INFINITY, 0.5, true),
            DomainCheckResult::Undecided,
            "Inf bounds must produce Undecided in verify_upper mode"
        );

        let verifier = input_split_verifier();
        let graph = verifier.configured_graph_for_crown(&identity_graph());
        let bootstrap = scalar_bootstrap(None);
        let picked = PickedDomains {
            batch_size: 1,
            layer_lowers: HashMap::new(),
            layer_uppers: HashMap::new(),
            input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![-1.0_f32])
                .expect("valid picked lower bounds"),
            input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0_f32])
                .expect("valid picked upper bounds"),
            global_lbs: vec![f32::NEG_INFINITY],
            global_ubs: vec![f32::INFINITY],
            metadata: vec![DomainMetadata::root(f32::NEG_INFINITY, f32::INFINITY)
                .expect("±inf parent bounds pass metadata validation")],
        };
        let mut state = BabLoopState::new(Instant::now());
        let mut domain_list = DomainList::new(DomainListConfig {
            traversal: TreeTraversal::BreadthFirst,
            layer_names: Vec::new(),
            layer_shapes: HashMap::new(),
            input_shape: vec![1],
            initial_capacity: 8,
            max_queue_size: 0,
        })
        .expect("valid domain list");

        let outcome = process_input_split_batch(
            &verifier,
            &graph,
            &picked,
            &[0],
            &[1.0_f32],
            &bootstrap,
            0.0,
            None,
            &mut state,
            &mut domain_list,
            0,
        )
        .expect("non-finite parent bounds must be dropped, not abort the batch");

        assert!(
            matches!(outcome, InputSplitOutcome::Continue),
            "a dropped parent domain must not short-circuit the loop",
        );
        assert_eq!(state.domains_explored, 1);
        assert_eq!(
            domain_list.len(),
            0,
            "a non-finite parent must be dropped before splitting, not re-queued",
        );
        assert_unknown_due_to_propagation_failure(&state);
    }

    /// Regression test for #2102: a failed child CROWN/IBP bound computation
    /// must set `unresolved_due_to_propagation_failure`.
    ///
    /// Drives `screen_child_domain` against a fixed node-bounds map that lacks
    /// the graph's output node, so `compute_crown_or_ibp_bounds_with_node_bounds`
    /// fails for real ("Output node ... not found") with no fallback producing
    /// bounds: the child must be skipped and the drop mapped to a propagation
    /// failure.
    #[ntest::timeout(10000)]
    #[test]
    fn test_input_split_crown_failure_sets_propagation_failure_flag_2102() {
        let verifier = input_split_verifier();
        let graph = verifier.configured_graph_for_crown(&identity_graph());
        let bootstrap = scalar_bootstrap(Some(HashMap::new()));
        let mut state = BabLoopState::new(Instant::now());

        let action = screen_child_domain(
            &verifier,
            &graph,
            &[1.0_f32],
            &bootstrap,
            0.0,
            None,
            0,
            0,
            -1.0,
            1.0,
            None,
            &mut state,
            scalar_child(-1.0, 1.0),
        )
        .expect("child bound-computation failure must be dropped, not abort the batch");

        assert!(
            matches!(action, ChildDomainAction::Skip),
            "a child whose bound computation failed must be skipped, not queued",
        );
        assert_unknown_due_to_propagation_failure(&state);

        // Combined reason when both depth and propagation failure occur
        // (possible in practice when some children hit max depth and others fail).
        state.unresolved_due_to_depth = true;
        state.max_depth_reached = 10;
        let final_result = state.build_final_result();
        match final_result.result {
            BabVerificationStatus::Unknown { reason } => {
                assert!(
                    reason.contains("Child propagation failed"),
                    "reason must mention propagation failure, got: {reason}",
                );
                assert!(
                    reason.contains("Max depth 10 reached"),
                    "reason must also mention depth limit, got: {reason}",
                );
            }
            other => unreachable!("expected Unknown result, got {other:?}"),
        }
    }
}
