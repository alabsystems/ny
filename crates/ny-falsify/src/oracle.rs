// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The evaluator seam.
//!
//! A strategy never runs a network. It hands the caller a batch of input
//! vectors and gets back, per point, a steering margin and whether the caller's
//! own arithmetic says the property's assertions hold there.
//!
//! `holds` is not a verdict and cannot become one. It is the caller talking
//! about its own point; it never leaves the search; and the only thing a
//! strategy does with it is stop and return the INPUT VECTOR. Publication still
//! requires the caller's unchanged `gate_sat_with_trusted_oracle` under a real
//! ORT forward on the ORIGINAL graph, and a candidate that fails is dropped.

/// One point's score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    /// The steering margin the search hill-climbs. May be a lifted / pre-map
    /// view; it is the caller's choice and the search does not interpret it
    /// beyond "bigger is better".
    pub steer: f64,
    /// The caller's own check that every assertion of the property holds at
    /// this point.
    pub holds: bool,
}

/// The oracle failed. Not a verdict either way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleError(pub String);

impl core::fmt::Display for OracleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for OracleError {}

/// Scores a batch of candidate input vectors.
pub trait Oracle {
    /// Score `points`. Must return exactly one [`Score`] per point.
    fn evaluate_batch(&mut self, points: &[Vec<f64>]) -> Result<Vec<Score>, OracleError>;

    /// How many points the oracle will accept in one call. An oracle that
    /// cannot batch returns 1 and pays one forward per point; the strategies
    /// size their work units off this so that a non-batching oracle does not
    /// silently multiply the wall clock.
    fn batch_limit(&self) -> usize {
        256
    }
}
