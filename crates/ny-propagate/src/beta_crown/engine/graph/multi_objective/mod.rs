// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Multi-objective graph β-CROWN verification.
//!
//! Verifies disjunctive properties (OR of linear constraints) by running all
//! objectives through a single branch-and-bound pass.

mod active_set_gpu_alpha;
mod batched;
mod bounded_shared_executor;
mod critical_gpu_alpha;
mod dd_zono_root;
mod domain_process;
mod finalize;
mod finalized_root_handoff;
#[cfg(test)]
mod metrics_emission_tests;
mod output_conditioned_head;
mod per_disjunct;
mod per_disjunct_eval;
mod post_c_survivor;
mod queue;
mod root;
// #root-phases: step 1 of decomposing evaluate_root -- the schedulable
// shrink-only tightener sequence, as data. Declarative only; see
// docs/DECOMPOSE_EVALUATE_ROOT_2026-08-10.md.
// Its items intentionally become live only in the extraction step that follows.
#[allow(dead_code)]
mod root_phases;
mod selective_root_alpha;
mod sequential;
mod shared;
mod stall_obbt_canary;
#[cfg(test)]
mod tests;
mod verify;

#[cfg(test)]
pub(crate) use shared::{merge_pruned_objective_bounds, prune_verified_multi_objective_targets};

/// #bab-monotone-inherit: the monotone parent-bound merge and its dark gate live
/// in `shared` so every BaB child lane in this module tree (batched, sequential,
/// per-disjunct) uses ONE implementation. Re-exported at module scope with the
/// helper's own visibility so the submodules reach it as `super::…`.
pub(in crate::beta_crown::engine::graph::multi_objective) use shared::{
    bab_monotone_inherit_enabled, inherit_parent_lower_only, tighten_child_bounds_with_parent,
};
