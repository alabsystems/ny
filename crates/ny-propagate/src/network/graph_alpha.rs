// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Alpha-CROWN methods for GraphNetwork.

mod alpha_projection;
pub(crate) mod atomic_cuda_margin_step;
pub(crate) mod atomic_cuda_rows;
mod backward;
mod binding_row_replay;
mod bounds;
// #gap-attribution: DARK diagnostics — exact per-neuron decomposition of a
// CROWN bound's looseness (Theorem 1,
// docs/THEORY_EXACT_GAP_ATTRIBUTION_AND_MAXMIN_ALPHA_2026-08-03.md). Observes a
// completed fold; never participates in a bound, verdict or branching decision.
pub(crate) mod gap_attribution;
pub(crate) mod invprop_backward;
mod propagate_dag;
pub(crate) use propagate_dag::root_alpha_margin_enabled_with;
// #graph-lane-search (M7): V2's GraphSearchNet lowering + G1 self-check.
pub(crate) use propagate_dag::search_lowering;
pub(crate) mod propagate_helpers;
mod propagate_sequential;
mod reference_bounds;
// pub(crate) so the network module can re-export the resnet decomposition for the
// beta_crown BaB engine's per-domain GPU beta backward (#unsat-keystone step 4).
pub(crate) mod resnet_decompose;
pub(crate) mod resnet_skeleton;
// #row-weights: the row-player of the max-min alpha objective (Sec 5 of the
// gap-attribution theory doc). Its only production caller is an exact-dark
// child of the typed atomic root-C margin optimizer.
mod row_weights;
mod runtime_state;
mod sequential_gradients;
mod spsa;
mod spsa_accumulate;
mod zonotope;

pub(crate) use bounds::budget_policy;
// #root-joint-demand-rank: demand selector re-export for the armed root-joint lane.
pub(crate) use bounds::nodes_requiring_crown_tightening;
// #cgan-unstable-rows: one source for abc's `minimum_sparsity` ceiling.
#[cfg(test)]
pub(crate) use bounds::CganCompleteCollectionEntryCounter;
pub(crate) use bounds::DEFAULT_SPARSE_RELU_MAX_PERMILLE;
pub(crate) use bounds::{GraphAlphaCollectionOutcome, PrecomputedAlphaReferenceBounds};
pub(crate) use reference_bounds::merge_reference_bound_maps;
// #fwd-root-screen: C1's lookup-table helpers for `zono_core` (re-export only).
pub(crate) use bounds::{check_lookup_shapes, pwl_point_enclosure, pwl_range_f64};

/// Test-only route evidence: `(iters, exit)` of the most recent DAG α ascent
/// on this thread (#inert-ascent-exit), as printed by `[alpha-ascent]`.
#[cfg(test)]
pub(crate) fn last_alpha_ascent_exit() -> Option<(usize, &'static str)> {
    propagate_dag::LAST_ALPHA_ASCENT_EXIT.with(|cell| *cell.borrow())
}

#[cfg(test)]
pub(crate) fn clear_last_alpha_ascent_exit() {
    propagate_dag::LAST_ALPHA_ASCENT_EXIT.with(|cell| *cell.borrow_mut() = None);
}

/// Process-wide retention policy for UNCONDITIONAL per-event observables
/// that can fire many times per row (one DAG ascent per clause, one refusal
/// per collection target): the first 16 occurrences and every power of two
/// afterwards are retained; `None` means this occurrence is suppressed. The
/// first occurrence is ALWAYS retained, which is the property that keeps a
/// null distinguishable from a negative. Returns the 1-based ordinal.
pub(crate) fn retained_occurrence(counter: &std::sync::atomic::AtomicUsize) -> Option<usize> {
    let occ = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    (occ <= 16 || occ.is_power_of_two()).then_some(occ)
}

#[cfg(test)]
mod retained_occurrence_tests {
    use super::retained_occurrence;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn retained_occurrence_keeps_the_first_sixteen_then_powers_of_two() {
        let counter = AtomicUsize::new(0);
        let kept: Vec<usize> = (1..=70)
            .filter_map(|_| retained_occurrence(&counter))
            .collect();
        let mut expected: Vec<usize> = (1..=16).collect();
        expected.extend([32, 64]);
        assert_eq!(kept, expected);
    }
}
