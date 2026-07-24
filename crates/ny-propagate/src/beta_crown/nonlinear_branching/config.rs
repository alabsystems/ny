// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

/// Configuration for nonlinear branching.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearBranchingConfig {
    /// Branching point method: how to select split points.
    #[serde(default)]
    pub point_method: BranchingPointMethod,

    /// Number of branches per split (2 for binary split).
    /// Binary splitting creates 2 branches with 1 branching point.
    #[serde(default = "default_num_branches")]
    pub num_branches: usize,

    /// Number of candidate neurons to evaluate for branching.
    #[serde(default = "default_num_candidates")]
    pub num_candidates: usize,

    /// Enable kFSB-like filtering (evaluate actual bounds for top candidates).
    #[serde(default)]
    pub filter: bool,

    /// Only branch ReLU neurons (ignore other nonlinearities).
    /// Useful for hybrid networks where ReLU dominates.
    #[serde(default)]
    pub relu_only: bool,

    /// Heuristic method for scoring candidate neurons.
    #[serde(default)]
    pub method: NonlinearHeuristicMethod,

    /// Minimum bound width for a neuron to be considered for branching.
    /// Neurons with tighter bounds than this are skipped.
    #[serde(default = "default_min_width")]
    pub min_branch_width: f32,
}

fn default_num_branches() -> usize {
    2
}

fn default_num_candidates() -> usize {
    1
}

fn default_min_width() -> f32 {
    1e-6
}

impl Default for NonlinearBranchingConfig {
    fn default() -> Self {
        Self {
            point_method: BranchingPointMethod::default(),
            num_branches: 2,
            num_candidates: 1,
            filter: false,
            relu_only: false,
            method: NonlinearHeuristicMethod::default(),
            min_branch_width: 1e-6,
        }
    }
}

/// Method for selecting branching points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum BranchingPointMethod {
    /// Split at uniform intervals (midpoint for binary).
    /// For num_branches=2: split at (lower + upper) / 2
    /// For num_branches=n: split at n-1 uniform points
    #[default]
    Uniform,
}

/// Heuristic method for scoring candidate neurons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum NonlinearHeuristicMethod {
    /// BBPS: fast heuristic using linear bound coefficients.
    /// Estimates impact of branching from existing linear bounds.
    #[default]
    Bbps,

    /// Simple bound width scoring.
    /// Selects neurons with largest bound width (similar to LargestBoundWidth).
    BoundWidth,
}
