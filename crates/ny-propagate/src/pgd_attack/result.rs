// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Result types for PGD attacks.

use ndarray::ArrayD;

/// Result of a PGD attack.
#[derive(Debug, Clone)]
pub struct PgdResult {
    /// Whether a counterexample was found.
    pub found_counterexample: bool,
    /// The counterexample input (if found).
    pub counterexample: Option<ArrayD<f32>>,
    /// Output at counterexample (if found).
    pub output: Option<ArrayD<f32>>,
    /// Best (most violating) output value found.
    pub best_output_value: f32,
    /// Number of restarts completed successfully.
    pub restarts_completed: usize,
    /// Number of restarts that failed with errors.
    /// When nonzero, some restarts errored and their counterexample search was lost.
    /// When equal to `restarts_completed + failed_restarts == num_restarts` and
    /// `found_counterexample` is false, the attack result is less reliable.
    pub failed_restarts: usize,
    /// Total network evaluations.
    pub total_evaluations: usize,
}

/// Internal result from a single PGD restart.
///
/// This struct replaces the complex tuple type `(ArrayD<f32>, ArrayD<f32>, f32, bool, usize)`
/// for better readability and self-documentation.
pub(crate) struct RestartResult {
    /// The input point found by this restart.
    pub input: ArrayD<f32>,
    /// Network output at the found input point.
    pub output: ArrayD<f32>,
    /// The objective value (output[idx], difference, or conjunctive max).
    pub value: f32,
    /// Whether this result represents a property violation.
    pub is_violation: bool,
    /// Number of network evaluations used in this restart.
    pub evaluations: usize,
}
