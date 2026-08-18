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
#[cfg(test)]
pub(crate) use bounds::CganCompleteCollectionEntryCounter;
pub(crate) use bounds::{GraphAlphaCollectionOutcome, PrecomputedAlphaReferenceBounds};
pub(crate) use reference_bounds::merge_reference_bound_maps;
