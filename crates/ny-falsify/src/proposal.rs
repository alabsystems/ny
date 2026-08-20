// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! What a strategy is allowed to return (design §8, M1 and M3).

use crate::admission::Decline;
use core::time::Duration;

/// The identity of a strategy. A plain string is deliberately NOT used: the
/// scheduler keys receipts and per-strategy ceilings off this, and a typo in a
/// string would silently create a second lane with no ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum StrategyName {
    /// S1 — declared corners, centre, parity patterns, bound/centre midpoints.
    SpecialPoints,
    /// S9 — random block sign-flip hill climbing.
    Square,
}

impl StrategyName {
    /// The name used in the Python portfolio's `strategies_run` accounting, so
    /// a receipt from this crate can be lined up against
    /// `reports/falsification_audit/selftest_calibration.json` directly.
    pub const fn calibration_key(self) -> &'static str {
        match self {
            Self::SpecialPoints => "special",
            Self::Square => "square",
        }
    }
}

impl core::fmt::Display for StrategyName {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.calibration_key())
    }
}

/// Work actually done. A `Decline` and an `Exhausted` are both receipts, never
/// silences (design §5 rule 4): a structural decline is only "free" if it is
/// measured to be.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Effort {
    /// Candidate input points constructed and handed to the oracle.
    pub points: usize,
    /// Oracle batch calls. Not the same as points: an oracle that refuses to
    /// batch pays one forward per point, and counting batches there would make
    /// a slow row look identical to a fast one.
    pub batches: usize,
    /// Wall time consumed by this strategy.
    pub wall: Duration,
    /// Best steering margin seen (`None` when nothing was evaluated).
    pub best_steer: Option<f64>,
    /// Work units spent since the last accepted progress, at the point of exit.
    /// This is what a stall rule reads.
    pub stalled_units: u64,
}

impl Effort {
    pub(crate) fn observe(&mut self, points: usize, best_in_batch: f64) {
        self.points += points;
        self.batches += 1;
        self.best_steer = Some(match self.best_steer {
            Some(previous) if previous >= best_in_batch => previous,
            _ => best_in_batch,
        });
    }
}

/// A proposed input vector. **Inputs only** (M3).
///
/// There is no output vector and no margin on this type on purpose. The `Y_j`
/// coordinates of a published witness must come from a real forward on the
/// ORIGINAL graph performed once, in the caller's publication path — not from
/// whatever arithmetic the search happened to do.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    inputs: Vec<f64>,
    found_by: StrategyName,
    effort: Effort,
}

impl Candidate {
    pub(crate) fn new(inputs: Vec<f64>, found_by: StrategyName, effort: Effort) -> Self {
        Self {
            inputs,
            found_by,
            effort,
        }
    }

    /// The candidate point, in declared spec input coordinates.
    pub fn inputs(&self) -> &[f64] {
        &self.inputs
    }

    /// Which strategy proposed it. Attribution only; carries no authority.
    pub const fn found_by(&self) -> StrategyName {
        self.found_by
    }

    /// The work that produced it.
    pub const fn effort(&self) -> &Effort {
        &self.effort
    }
}

/// Everything a strategy may return.
///
/// M1: this enum has no verdict-bearing variant, by construction. Adding one
/// would be a visible change to a public type in a crate that cannot see
/// `ny-cli`, and `the_crate_cannot_name_a_verdict` matches on it exhaustively.
#[derive(Debug, Clone, PartialEq)]
pub enum Proposal {
    /// An input vector worth handing to the caller's trusted-oracle gate.
    /// It is a *proposal*: the gate may, and often does, drop it.
    Candidate(Candidate),
    /// The strategy ran out of budget, work units or ideas. Not a claim about
    /// the property — a failed search is never a proof.
    Exhausted(Effort),
    /// The strategy was not admitted. Structural, typed, and cheap.
    Declined(Decline),
}

impl Proposal {
    /// True when this proposal cannot possibly influence a verdict on its own.
    /// Every variant satisfies it; that is the point.
    pub const fn is_falsification_only(&self) -> bool {
        match self {
            Self::Candidate(_) | Self::Exhausted(_) | Self::Declined(_) => true,
        }
    }
}
