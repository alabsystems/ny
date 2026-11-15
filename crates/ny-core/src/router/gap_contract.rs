// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r5 CR-G — GAP-ORDER CONTRACTS inside a multi-objective lane
//! (ROUTER_PROGRAM §4.2 CR-G, flaw table F8 / F9, §10 R5). PURE: integer
//! nanoseconds, no clock, no environment, no global.
//!
//! The frozen-dual tier already runs its classes in gap order, easiest first
//! (largest lane root bound, F9). What starved it was the funding: each class
//! got `left / (classes left)` of the tier's remaining wall, a FAIR SHARE that
//! need not fit even one state build — `t8853` refused 15/15 classes because
//! every ~1.3 s share was shorter than its build (MEASURED,
//! `op_fd_tier_census.out`). CR-G replaces the share with a CONTRACT:
//!
//! * a class is funded with at least its RUNG — one build plus
//!   [`CONTRACT_LEVELS`] BFS levels (F8: "contract >= build + 2 levels") —
//!   priced from what the tier MEASURED on the classes it already ran
//!   ([`ClassCost`]): the last build, the last per-level time, or, when a
//!   build was cut by its window before it finished, twice the longest cut
//!   window (a build cut at `w` needs more than `w`; doubling bounds the time
//!   spent on cut builds by twice the build that finally fits);
//! * with nothing measured yet (the first class) the class gets today's fair
//!   share, so the first class is funded exactly as today;
//! * a rung that the fair share already covers keeps the fair share (CR-G
//!   changes nothing where today's funding fits);
//! * a rung larger than the fair share is funded IN FULL when the tier still
//!   holds it (the easier class gets its rung, the harder classes behind it
//!   get less), and DECLINED when it does not: the class never half-runs, and
//!   the classes behind it (harder, priced the same) stay open in the tree,
//!   which takes back the tier's unspent wall.
//!
//! There is no ETA-based extension and no new constant beyond the two levels
//! the program names. Soundness: funding only decides how long a class's own
//! certified search may run; a declined or cut class closes nothing and stays
//! open in the tree.

use serde::{Deserialize, Serialize};

/// BFS levels a contract funds beyond one build (F8).
pub const CONTRACT_LEVELS: u64 = 2;

/// What the tier measured on the classes it already ran (nanoseconds).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct ClassCost {
    /// The most recent completed state build.
    pub build_ns: Option<u64>,
    /// The most recent search's time per BFS level (its wall over
    /// `max(depth, 1)`).
    pub level_ns: Option<u64>,
    /// The longest window a build was cut at before it completed.
    pub cut_ns: Option<u64>,
}

impl ClassCost {
    /// Record a class that built and searched.
    #[must_use]
    pub const fn with_run(self, build_ns: u64, search_ns: u64, depth: u64) -> Self {
        let levels = if depth == 0 { 1 } else { depth };
        Self {
            build_ns: Some(build_ns),
            level_ns: Some(search_ns / levels),
            cut_ns: self.cut_ns,
        }
    }

    /// Record a class whose build was cut by its window `window_ns`.
    #[must_use]
    pub const fn with_cut(self, window_ns: u64) -> Self {
        let cut = match self.cut_ns {
            Some(previous) if previous >= window_ns => previous,
            _ => window_ns,
        };
        Self {
            build_ns: self.build_ns,
            level_ns: self.level_ns,
            cut_ns: Some(cut),
        }
    }
}

/// The rung a class needs: one build plus [`CONTRACT_LEVELS`] levels, from
/// the most recent completed build; else twice the longest cut window; else
/// unknown (`None`). Saturating.
#[must_use]
pub const fn rung_ns(cost: &ClassCost) -> Option<u64> {
    match (cost.build_ns, cost.level_ns, cost.cut_ns) {
        (Some(build), Some(level), _) => {
            Some(build.saturating_add(level.saturating_mul(CONTRACT_LEVELS)))
        }
        (Some(build), None, _) => Some(build),
        (None, _, Some(cut)) => Some(cut.saturating_mul(2)),
        (None, _, None) => None,
    }
}

/// How a class is funded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClassWindow {
    /// Today's fair share (nothing priced, or the share covers the rung).
    Fair {
        /// The window, ns.
        window_ns: u64,
    },
    /// A contract: the rung, larger than the fair share, funded in full.
    Contract {
        /// The window (= the rung), ns.
        window_ns: u64,
        /// The fair share it replaced, ns.
        fair_ns: u64,
    },
    /// The rung does not fit what the tier has left: the class does not run.
    Decline {
        /// The rung, ns.
        rung_ns: u64,
        /// What the tier had left, ns.
        left_ns: u64,
    },
}

impl ClassWindow {
    /// The window to run the class for (`None` when declined).
    #[must_use]
    pub const fn window_ns(self) -> Option<u64> {
        match self {
            Self::Fair { window_ns } | Self::Contract { window_ns, .. } => Some(window_ns),
            Self::Decline { .. } => None,
        }
    }

    /// Stable token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Fair { .. } => "fair",
            Self::Contract { .. } => "contract",
            Self::Decline { .. } => "decline",
        }
    }
}

/// Fund the next class. `fair_ns` is today's share, computed by the caller
/// exactly as today (`left / classes_left`); `left_ns` is the tier's whole
/// remaining wall. Pure and total.
#[must_use]
pub const fn class_window(fair_ns: u64, left_ns: u64, cost: &ClassCost) -> ClassWindow {
    match rung_ns(cost) {
        None => ClassWindow::Fair { window_ns: fair_ns },
        Some(rung) if rung <= fair_ns => ClassWindow::Fair { window_ns: fair_ns },
        Some(rung) if rung <= left_ns => ClassWindow::Contract {
            window_ns: rung,
            fair_ns,
        },
        Some(rung) => ClassWindow::Decline {
            rung_ns: rung,
            left_ns,
        },
    }
}
