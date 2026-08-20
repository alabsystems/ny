// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Stall rules, in units of each strategy's own work.
//!
//! This is the single most valuable lesson in the traffic session. Before
//! `#bnn-lp-stall`, the LP lane spent 216.98-218.30 s and accepted 0 flips on
//! all nine targeted rows, leaving the STE lane behind it only 108.5 s of its
//! 240 s cap. With the rule the lane yields in 0.03-0.05 s (refused) or
//! ~35-53 s (stalled) instead of 217 s -- and on the three `model_30` eps=1
//! rows it owns it changed the trajectory NOT AT ALL: 78 flips / 311 LP solves
//! / margin +2, identical to the run before the rule existed.
//!
//! Two properties make this a budget rule and never a correctness one:
//!
//! 1. the progress metric is monotone in **accepted progress**, not in effort;
//! 2. a stall rule may only bring `Exhausted` FORWARD. It can never turn a
//!    `Candidate` into anything else, because the candidate check happens on
//!    the batch before the stall counter is consulted.

/// The unit a strategy counts its own work in. A shared unit would be a
/// category error: 32 LP solves and 32 gradient steps are not comparable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkUnit {
    /// Realizability LPs solved without an accepted flip (S10).
    LpSolvesWithoutAcceptedFlip,
    /// Gradient steps without a margin improvement (S6/S7/S8).
    GradientStepsWithoutImprovement,
    /// Block-flip batches without a new best (S9).
    BlockBatchesWithoutImprovement,
    /// Sampling batches without a new best (S3).
    BatchesWithoutNewBest,
}

/// Abandon after `threshold` work units with no accepted progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StallRule {
    /// What is being counted.
    pub work_unit: WorkUnit,
    /// How many units without progress are tolerated.
    pub threshold: u64,
}

impl StallRule {
    /// Build a rule.
    pub const fn new(work_unit: WorkUnit, threshold: u64) -> Self {
        Self {
            work_unit,
            threshold,
        }
    }

    /// Start tracking.
    pub const fn tracker(self) -> StallTracker {
        StallTracker {
            rule: self,
            since_progress: 0,
        }
    }
}

/// Live stall state for one strategy run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StallTracker {
    rule: StallRule,
    since_progress: u64,
}

impl StallTracker {
    /// Record one work unit. `progressed` must mean *accepted* progress.
    pub fn observe(&mut self, progressed: bool) {
        if progressed {
            self.since_progress = 0;
        } else {
            self.since_progress += 1;
        }
    }

    /// Whether the strategy should abandon now.
    pub const fn stalled(&self) -> bool {
        self.since_progress >= self.rule.threshold
    }

    /// Units since the last accepted progress.
    pub const fn since_progress(&self) -> u64 {
        self.since_progress
    }

    /// The rule being applied.
    pub const fn rule(&self) -> StallRule {
        self.rule
    }
}
