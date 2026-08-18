// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Committed multi-depth expansion for multi-objective ReLU BaB.
//!
//! Branch scoring stays in `kfsb_multi`. This module only turns an ordered,
//! distinct decision plan into the complete feasible truth table. No
//! intermediate bound is computed here, so the returned leaves all enter the
//! ordinary authoritative child-bound path together.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use crate::beta_crown::branching::GraphNeuronConstraint;
use crate::beta_crown::domain::MultiObjectiveGraphBabDomain;
use crate::GraphNetwork;

/// Match the existing defensive cap in `GraphBabDomain::with_multi_constraints`.
pub(super) const MAX_MULTI_OBJECTIVE_SPLIT_DEPTH: usize = 10;

/// Do not start or continue optional multi-depth expansion unless one normal
/// authoritative child-bound chunk still has this much wall-clock headroom.
pub(super) const MULTI_DEPTH_AUTHORITY_RESERVE: Duration = Duration::from_secs(5);

/// Why a requested truth-table expansion declined. Every refusal leaves the
/// caller's existing depth-one children untouched for atomic fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MultiDepthRefusal {
    EmptyPlan,
    DuplicateDecision,
    LeafCap,
    DeadlineReserve,
    ChildConstruction,
}

/// Cap a wave's split depth so its final committed leaves never exceed the
/// existing depth-one worst-case capacity (`2 * batch_size`).
///
/// `parent_count` is the number of parents eligible for the wave selector.
/// Per-parent candidate shortages are applied by the caller after this common
/// wave cap, so mixed waves can only produce fewer leaves.
pub(super) fn cap_multi_objective_wave_depth(
    requested_depth: usize,
    parent_count: usize,
    candidate_count: usize,
    batch_size: usize,
) -> usize {
    if parent_count == 0 || candidate_count == 0 {
        return 0;
    }

    let mut depth = requested_depth
        .max(1)
        .min(candidate_count)
        .min(MAX_MULTI_OBJECTIVE_SPLIT_DEPTH);
    let wave_leaf_cap = batch_size.max(1).saturating_mul(2);
    while depth > 1 {
        let within_cap = 1usize
            .checked_shl(depth as u32)
            .and_then(|leaves_per_parent| parent_count.checked_mul(leaves_per_parent))
            .is_some_and(|wave_leaves| wave_leaves <= wave_leaf_cap);
        if within_cap {
            break;
        }
        depth -= 1;
    }
    depth
}

/// Cap a wave-selected split depth by the remaining configured depth budget
/// of one parent.
///
/// A wave can contain parents at different depths. Applying only the common
/// wave cap would let a shallow parent's selected depth push a deeper parent
/// past `max_depth`. The ordinary queue prefilter guarantees that an admitted
/// parent has at least one level left, but returning zero here keeps the helper
/// defensive and makes that boundary explicit.
#[inline]
pub(super) fn cap_multi_objective_parent_depth(
    wave_depth: usize,
    parent_depth: usize,
    max_depth: usize,
) -> usize {
    wave_depth.min(max_depth.saturating_sub(parent_depth))
}

/// Multi-depth commit is all-or-nothing at the wave-selected depth. A shorter
/// ranked plan must preserve the exact depth-one simulations for fallback.
#[inline]
pub(super) fn multi_depth_plan_is_complete(plan_len: usize, required_depth: usize) -> bool {
    required_depth > 1 && plan_len == required_depth
}

#[inline]
pub(super) fn multi_depth_authority_budget_available(
    now: Instant,
    authority_deadline: Option<Instant>,
) -> bool {
    authority_deadline.is_none_or(|deadline| {
        now.checked_add(MULTI_DEPTH_AUTHORITY_RESERVE)
            .is_some_and(|reserved_until| reserved_until < deadline)
    })
}

/// Build every feasible active/inactive combination for an ordered decision
/// plan. The bool carried beside each leaf is the phase of the first decision,
/// preserving the historical child tuple shape; downstream authoritative
/// bounding ignores that advisory label and reads the full split history.
///
/// The operation is atomic: a duplicate, deadline/cap refusal, or any
/// non-infeasibility construction error discards all locally-built leaves.
pub(super) fn expand_multi_objective_truth_table(
    graph: &GraphNetwork,
    parent: &MultiObjectiveGraphBabDomain,
    thresholds: &[f32],
    decisions: &[(String, usize, f32)],
    max_leaves: usize,
    authority_deadline: Option<Instant>,
) -> Result<Vec<(MultiObjectiveGraphBabDomain, bool)>, MultiDepthRefusal> {
    if decisions.is_empty() {
        return Err(MultiDepthRefusal::EmptyPlan);
    }
    if !multi_depth_authority_budget_available(Instant::now(), authority_deadline) {
        return Err(MultiDepthRefusal::DeadlineReserve);
    }

    let mut seen = HashSet::with_capacity(decisions.len());
    if decisions
        .iter()
        .any(|(node, neuron, _)| !seen.insert((node.as_str(), *neuron)))
    {
        return Err(MultiDepthRefusal::DuplicateDecision);
    }

    let expected_leaves = 1usize
        .checked_shl(decisions.len() as u32)
        .filter(|_| decisions.len() <= MAX_MULTI_OBJECTIVE_SPLIT_DEPTH)
        .ok_or(MultiDepthRefusal::LeafCap)?;
    if expected_leaves > max_leaves {
        return Err(MultiDepthRefusal::LeafCap);
    }

    let mut leaves: Vec<(MultiObjectiveGraphBabDomain, Option<bool>)> =
        vec![(parent.clone(), None)];
    for (node_name, neuron_idx, score) in decisions {
        if !multi_depth_authority_budget_available(Instant::now(), authority_deadline) {
            return Err(MultiDepthRefusal::DeadlineReserve);
        }
        let mut next = Vec::with_capacity(leaves.len().saturating_mul(2));
        for (domain, first_phase) in leaves {
            for is_active in [true, false] {
                let constraint =
                    GraphNeuronConstraint::new(node_name.clone(), *neuron_idx, is_active, *score)
                        .map_err(|_| MultiDepthRefusal::ChildConstruction)?;
                match domain.with_constraint(graph, constraint, false, thresholds) {
                    Ok(Some(child)) => {
                        next.push((child, first_phase.or(Some(is_active))));
                    }
                    Ok(None) => {}
                    Err(error) if error.is_infeasible_domain() => {}
                    Err(_) => return Err(MultiDepthRefusal::ChildConstruction),
                }
            }
        }
        leaves = next;
    }

    Ok(leaves
        .into_iter()
        .filter_map(|(domain, first_phase)| first_phase.map(|phase| (domain, phase)))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    use ny_tensor::BoundedTensor;

    use crate::beta_crown::domain::MultiObjectiveGraphBabDomain;
    use crate::{GraphNetwork, GraphNode, Layer, ReLULayer};

    fn direct_relu_fixture(
        lower: &[f32],
        upper: &[f32],
    ) -> (GraphNetwork, MultiObjectiveGraphBabDomain) {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
        graph.set_output("relu");
        let input =
            BoundedTensor::new(arr1(lower).into_dyn(), arr1(upper).into_dyn()).expect("valid box");
        let node_bounds = graph
            .collect_node_bounds(&input)
            .expect("direct ReLU bounds");
        let domain = MultiObjectiveGraphBabDomain::root(
            node_bounds,
            vec![(-1.0, 1.0)],
            &input,
            &[0.0],
            false,
        )
        .expect("valid root");
        (graph, domain)
    }

    #[test]
    fn depth_two_builds_exact_four_pattern_cover() {
        let (graph, parent) = direct_relu_fixture(&[-1.0, -2.0], &[1.0, 2.0]);
        let decisions = vec![("relu".to_string(), 0, 2.0), ("relu".to_string(), 1, 1.0)];
        let leaves =
            expand_multi_objective_truth_table(&graph, &parent, &[0.0], &decisions, 4, None)
                .expect("full truth table");
        assert_eq!(leaves.len(), 4);

        let patterns: HashSet<Vec<bool>> = leaves
            .iter()
            .map(|(leaf, first_phase)| {
                assert_eq!(leaf.depth(), 2);
                assert_eq!(
                    leaf.history().constraints.first().map(|c| c.is_active),
                    Some(*first_phase)
                );
                leaf.history()
                    .constraints
                    .iter()
                    .map(|constraint| constraint.is_active)
                    .collect()
            })
            .collect();
        assert_eq!(
            patterns,
            HashSet::from([
                vec![true, true],
                vec![true, false],
                vec![false, true],
                vec![false, false],
            ])
        );
    }

    #[test]
    fn cifar_root_depth_four_builds_sixteen_final_leaves() {
        let (graph, parent) = direct_relu_fixture(&[-1.0, -2.0, -3.0, -4.0], &[1.0, 2.0, 3.0, 4.0]);
        let decisions = (0..4)
            .map(|neuron| ("relu".to_string(), neuron, 4.0 - neuron as f32))
            .collect::<Vec<_>>();
        let leaves =
            expand_multi_objective_truth_table(&graph, &parent, &[0.0], &decisions, 16, None)
                .expect("CIFAR root d4 truth table");
        assert_eq!(leaves.len(), 16);
        assert!(leaves.iter().all(|(leaf, _)| leaf.depth() == 4));

        let patterns: HashSet<Vec<bool>> = leaves
            .iter()
            .map(|(leaf, _)| {
                leaf.history()
                    .constraints
                    .iter()
                    .map(|constraint| constraint.is_active)
                    .collect()
            })
            .collect();
        assert_eq!(patterns.len(), 16);
    }

    #[test]
    fn infeasible_halves_are_pruned_without_losing_feasible_cover() {
        let (graph, parent) = direct_relu_fixture(&[-1.0, 0.5], &[1.0, 1.0]);
        let decisions = vec![("relu".to_string(), 0, 2.0), ("relu".to_string(), 1, 1.0)];
        let leaves =
            expand_multi_objective_truth_table(&graph, &parent, &[0.0], &decisions, 4, None)
                .expect("stable inactive halves are empty");
        assert_eq!(leaves.len(), 2);
        assert!(leaves.iter().all(|(leaf, _)| {
            leaf.history()
                .constraints
                .last()
                .is_some_and(|constraint| constraint.is_active)
        }));
    }

    #[test]
    fn duplicate_plan_refuses_atomically() {
        let (graph, parent) = direct_relu_fixture(&[-1.0], &[1.0]);
        let decisions = vec![("relu".to_string(), 0, 2.0), ("relu".to_string(), 0, 1.0)];
        assert_eq!(
            expand_multi_objective_truth_table(&graph, &parent, &[0.0], &decisions, 4, None)
                .expect_err("duplicate cannot define a truth table"),
            MultiDepthRefusal::DuplicateDecision
        );
        assert!(parent.history().constraints.is_empty());
        assert_eq!(parent.depth(), 0);
    }

    #[test]
    fn depth_one_matches_manual_active_inactive_children() {
        let (graph, parent) = direct_relu_fixture(&[-1.0], &[1.0]);
        let decision = ("relu".to_string(), 0, 2.0);
        let expanded = expand_multi_objective_truth_table(
            &graph,
            &parent,
            &[0.0],
            std::slice::from_ref(&decision),
            2,
            None,
        )
        .expect("depth one");

        let mut manual = Vec::new();
        for is_active in [true, false] {
            let child = parent
                .with_constraint(
                    &graph,
                    GraphNeuronConstraint {
                        node_name: decision.0.clone(),
                        neuron_idx: decision.1,
                        is_active,
                        score: decision.2,
                    },
                    false,
                    &[0.0],
                )
                .expect("manual construction")
                .expect("manual side feasible");
            manual.push((child, is_active));
        }

        assert_eq!(expanded.len(), manual.len());
        for ((actual, actual_phase), (expected, expected_phase)) in expanded.iter().zip(&manual) {
            assert_eq!(actual_phase, expected_phase);
            assert_eq!(actual.depth(), expected.depth());
            assert_eq!(actual.history().constraints, expected.history().constraints);
            assert_eq!(
                actual.input_bounds().lower(),
                expected.input_bounds().lower()
            );
            assert_eq!(
                actual.input_bounds().upper(),
                expected.input_bounds().upper()
            );
        }
    }

    #[test]
    fn wave_cap_preserves_depth_one_capacity_and_allows_root_depth_four() {
        assert_eq!(cap_multi_objective_wave_depth(4, 1, 7, 256), 4);
        assert_eq!(cap_multi_objective_wave_depth(4, 64, 7, 256), 3);
        assert_eq!(cap_multi_objective_wave_depth(4, 256, 7, 256), 1);
        assert_eq!(cap_multi_objective_wave_depth(4, 0, 7, 256), 0);
    }

    #[test]
    fn parent_cap_never_crosses_configured_max_depth() {
        assert_eq!(cap_multi_objective_parent_depth(4, 6, 10), 4);
        assert_eq!(cap_multi_objective_parent_depth(4, 9, 10), 1);
        assert_eq!(cap_multi_objective_parent_depth(4, 10, 10), 0);
        assert_eq!(cap_multi_objective_parent_depth(1, 9, 10), 1);
    }

    #[test]
    fn partial_ranked_plan_preserves_depth_one_fallback() {
        assert!(!multi_depth_plan_is_complete(1, 4));
        assert!(!multi_depth_plan_is_complete(3, 4));
        assert!(multi_depth_plan_is_complete(4, 4));
        assert!(!multi_depth_plan_is_complete(1, 1));
    }

    #[test]
    fn five_second_authority_reserve_declines_without_partial_work() {
        let (graph, parent) = direct_relu_fixture(&[-1.0], &[1.0]);
        let deadline = Instant::now()
            .checked_add(Duration::from_secs(4))
            .expect("near deadline");
        assert_eq!(
            expand_multi_objective_truth_table(
                &graph,
                &parent,
                &[0.0],
                &[("relu".to_string(), 0, 1.0)],
                2,
                Some(deadline),
            )
            .expect_err("five-second reserve should decline"),
            MultiDepthRefusal::DeadlineReserve
        );
        assert!(parent.history().constraints.is_empty());
    }
}
