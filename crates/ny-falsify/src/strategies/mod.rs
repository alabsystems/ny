// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The ported strategies.
//!
//! **Two, not six, and on calibration evidence rather than fresh rows.** The
//! E1 open-row measurement that gated this work returned 0 counterexamples on
//! 42 measurements at official and 10-12x budgets. What survives that zero is
//! the calibration attribution in
//! `reports/falsification_audit/selftest_calibration.json`: six strategies,
//! none dominating, each winning rows the others did not. Of those six,
//! `special` and `square` are the two with independent evidence that ny has no
//! equivalent, so they are the two ported.
//!
//! Deliberately NOT ported:
//!
//! - `spsa` (15 wins) and `nes` (3) — ny's incumbent lane is already an
//!   SPSA-class estimated-gradient attack (`vnncomp.rs:10949`: "The internal
//!   verifier's counterexample search is SPSA-based"), plus the exact-gradient
//!   APGD lane. Porting these would duplicate reach ny already has.
//! - `corners_random` (14 wins) — ny ships `low_dim_ort_corner_falsify`, a
//!   vertex-cover lane over the same geometry.
//! - `grid` (7 wins, `cctsdb_yolo` 5/5 and nothing else, ever) — the third
//!   candidate by evidence, and the one to do next. It is a total sweep of a
//!   <=3-free-input domain; it is trivially expressible on this chassis and its
//!   admission predicate is already written (`free_dims <= 3`). It is left out
//!   only to keep the first port to strategies with a demonstrated capability
//!   hole rather than a demonstrated family.

mod special_points;
mod square;

pub use special_points::{SpecialPoints, SPECIAL_MAX_FREE_DIMS, SPECIAL_PATTERNS};
pub use square::{
    Square, SQUARE_ANNEAL, SQUARE_INITIAL_FRACTION, SQUARE_ITERATIONS_PER_RESTART,
    SQUARE_MAX_FREE_DIMS, SQUARE_MIN_FREE_DIMS, SQUARE_STALL_BATCHES,
};

use crate::domain::SearchBox;
use crate::oracle::Oracle;
use crate::proposal::{Candidate, Effort, StrategyName};
use crate::registry::SearchState;

/// What one oracle batch produced.
pub(crate) enum Batch {
    /// A point whose assertions the caller's own arithmetic says all hold.
    /// The strategy stops and returns the INPUT VECTOR.
    Hit(Box<Candidate>),
    /// Steering margins, one per point, in input order.
    Margins(Vec<f64>),
    /// The oracle failed. Treated as end-of-strategy, never as a verdict.
    OracleFailed,
}

/// Materialise free-coordinate rows into declared input vectors, score them,
/// fold the best into the shared incumbent, and stop on a hit.
///
/// Every strategy goes through here, which is why "the candidate check happens
/// before the stall counter is consulted" is true once rather than per lane.
pub(crate) fn evaluate(
    domain: &SearchBox,
    oracle: &mut dyn Oracle,
    state: &mut SearchState,
    effort: &mut Effort,
    who: StrategyName,
    rows: &[Vec<f64>],
) -> Batch {
    if rows.is_empty() {
        return Batch::Margins(Vec::new());
    }
    let points: Vec<Vec<f64>> = rows.iter().map(|row| domain.materialise(row)).collect();
    let scores = match oracle.evaluate_batch(&points) {
        Ok(scores) if scores.len() == points.len() => scores,
        _ => return Batch::OracleFailed,
    };

    let mut best_index = 0usize;
    let mut best_steer = f64::NEG_INFINITY;
    for (index, score) in scores.iter().enumerate() {
        if score.steer > best_steer {
            best_steer = score.steer;
            best_index = index;
        }
    }
    effort.observe(points.len(), best_steer);

    // The incumbent is carried in FREE coordinates, and it must be the SNAPPED
    // ones: those are the values the oracle actually saw. Carrying the
    // pre-snap request would let a strategy hill-climb on a point that was
    // never evaluated.
    let best_free: Vec<f64> = domain.snap(&rows[best_index]);
    state.offer(&best_free, best_steer);

    if let Some(index) = scores.iter().position(|score| score.holds) {
        return Batch::Hit(Box::new(Candidate::new(
            points[index].clone(),
            who,
            effort.clone(),
        )));
    }
    Batch::Margins(scores.iter().map(|score| score.steer).collect())
}
