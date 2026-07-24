// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Intermediate Domain Clipping: Tighten intermediate bounds using split constraints.
//!
//! This implements the `clip_interm_domain` feature from α,β-CROWN, which uses
//! split-derived linear constraints in input space to tighten intermediate
//! (pre-activation) bounds via constrained concretization.
//!
//! ## Algorithm Overview
//!
//! Given a domain's split history and CROWN linear bounds for intermediate neurons:
//!
//! 1. **Build split constraints**: Convert each split decision to an input-space
//!    linear constraint `A·x + b ≤ 0` using the CROWN linear bounds of the split neuron.
//!
//! 2. **Select objective neurons**: Pick the top-k most impactful unstable neurons
//!    per layer using a kFSB-style scoring heuristic.
//!
//! 3. **Constrained concretization**: For each objective neuron, solve a constrained
//!    optimization problem using the complete clipping solver to get tighter bounds.
//!
//! 4. **Merge bounds**: Update the domain's intermediate bounds with the tightened values.
//!
//! ## Soundness Model
//!
//! Split constraints are **necessary conditions** (relaxations) of the true split region,
//! so the LP feasible region is a superset of the true constrained region. This ensures
//! soundness: the tightened bounds still contain all reachable values.
//!
//! ## References
//!
//! - Design: `designs/2026-01-29-clip-interm-domain.md`
//! - Baseline: α,β-CROWN `complete_verifier/domain_clipper.py`
//! - Paper: Wei et al., "Clip and Verify" (arXiv:2512.11087)

use ndarray::{Array1, Array2};

mod constraints;
mod objectives;
mod tighten;

// Re-export the high-level integration API (used by engine/domain.rs).
pub use tighten::clip_interm_domain_full;

// Re-export submodule APIs for tests via `super::*`.
#[cfg(test)]
use crate::beta_crown::branching::GraphSplitHistory;
#[cfg(test)]
pub(crate) use constraints::split_constraints_from_store;
pub(crate) use constraints::{
    add_f32_down, build_split_constraints, build_split_constraints_with_deadline_check,
    sort_out_constraints, sort_out_constraints_with_deadline_check, sub_f32_down,
};
#[cfg(test)]
pub use objectives::select_objective_neurons;
#[cfg(test)]
pub(crate) use tighten::compute_unconstrained_bounds;
pub(crate) use tighten::tighten_with_constraints_with_deadline;
pub(crate) use tighten::{merge_bounds, tighten_with_constraints};

/// Result of building split constraints.
#[derive(Debug, Clone)]
pub struct SplitConstraints {
    /// Constraint matrix A, shape: `(n_constraints, x_dim)`.
    /// Each row is a constraint: A[k] · x + b[k] ≤ 0.
    pub a_matrix: Array2<f32>,
    /// Constraint offset vector b, shape: `(n_constraints,)`.
    pub b_vector: Array1<f32>,
    /// Number of valid constraints (may be less than allocated rows).
    pub num_constraints: usize,
}

impl SplitConstraints {
    /// Create empty constraints.
    pub fn empty(x_dim: usize) -> Self {
        Self {
            a_matrix: Array2::zeros((0, x_dim)),
            b_vector: Array1::zeros(0),
            num_constraints: 0,
        }
    }

    /// Check if constraints are empty.
    pub fn is_empty(&self) -> bool {
        self.num_constraints == 0
    }
}

/// Preprocessed constraints ready for constrained concretization.
#[derive(Debug, Clone)]
pub struct PreprocessedConstraints {
    /// Active constraint matrix A, shape: `(n_active, x_dim)`.
    pub a_active: Array2<f32>,
    /// Original active offsets `b` from `A*x+b <= 0`, shape `(n_active,)`.
    /// Retained exactly so a certified consumer never has to reconstruct `b`
    /// from the rounded centered offset `d`.
    pub b_active: Array1<f32>,
    /// Transformed offset d = A·x0 + b, shape: `(n_active,)`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub d_active: Array1<f32>,
    /// Mask of infeasible constraints (constraint can never be satisfied).
    /// Read by #[cfg(test)] assertions to verify preprocessing correctness.
    #[cfg_attr(not(test), allow(dead_code))]
    pub infeasible_mask: Vec<bool>,
    /// Mask of fully covered constraints (constraint always satisfied).
    /// Read by #[cfg(test)] assertions to verify preprocessing correctness.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fully_covered_mask: Vec<bool>,
}

#[cfg(test)]
mod tests;
