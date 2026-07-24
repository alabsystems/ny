// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Multi-objective graph β-CROWN verification.
//!
//! Verifies disjunctive properties (OR of linear constraints) by running all
//! objectives through a single branch-and-bound pass.

mod batched;
mod dd_zono_root;
mod domain_process;
mod finalize;
#[cfg(test)]
mod metrics_emission_tests;
mod per_disjunct;
mod per_disjunct_eval;
mod post_c_survivor;
mod queue;
mod root;
mod sequential;
mod shared;
#[cfg(test)]
mod tests;
mod verify;

#[cfg(test)]
pub(crate) use shared::{merge_pruned_objective_bounds, prune_verified_multi_objective_targets};
