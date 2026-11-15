// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GenBaB: Branch-and-Bound for General Nonlinearities
//!
//! Implements branching heuristics for non-ReLU activations (GeLU, Sigmoid, Tanh,
//! Sine, etc.) following the GenBaB approach from alpha-beta-CROWN.
//!
//! # Overview
//!
//! Unlike ReLU which has a natural branching point at 0, general nonlinearities
//! can be split at any point within their domain. This module provides:
//!
//! - Uniform branching: split at midpoint(s) of the neuron's bounds
//! - BBPS heuristic: fast scoring based on linear bound coefficients
//!
//! # References
//!
//! - Shi et al., "Neural Network Verification with Branch-and-Bound for General
//!   Nonlinearities" (TACAS 2025)
//! - alpha-beta-CROWN: `complete_verifier/heuristics/nonlinear/bbps.py`

mod bilinear;
mod config;
mod decision;
mod scoring;
mod selector;
mod telemetry;

pub use config::{BranchingPointMethod, NonlinearBranchingConfig, NonlinearHeuristicMethod};
pub use decision::{points_strictly_partition, BranchingDecision};
pub(crate) use telemetry::{genbab_select_summary_line, record_genbab_empty_child};

/// Nonlinear branching heuristic for GenBaB.
///
/// This struct holds the configuration and provides methods for selecting
/// which neurons to branch and where to place the branching points.
pub struct NonlinearBranching {
    config: NonlinearBranchingConfig,
}

impl NonlinearBranching {
    /// Create a new nonlinear branching heuristic with the given configuration.
    pub fn new(config: NonlinearBranchingConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    #[deprecated(since = "0.1.0", note = "use NonlinearBranching::default() instead")]
    pub fn with_defaults() -> Self {
        Self::new(NonlinearBranchingConfig::default())
    }

    /// Get the configuration.
    pub fn config(&self) -> &NonlinearBranchingConfig {
        &self.config
    }
}

impl Default for NonlinearBranching {
    fn default() -> Self {
        Self::new(NonlinearBranchingConfig::default())
    }
}

#[cfg(test)]
mod tests;
