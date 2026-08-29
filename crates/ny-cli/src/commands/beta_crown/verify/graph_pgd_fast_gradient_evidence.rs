// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Versioned execution evidence for the governed sequential fast-gradient
//! selector.
//!
//! Wide exact-VJP and direct GAMA point-VJP routes deliberately do not own one
//! of these guards. The receipt therefore proves only a decision made by
//! `exact_graph_spec_gradient`, the sole historical consumer of
//! `NY_PGD_FAST_GRAD`.

use std::cell::Cell;

use crate::flight::FlightStatus;

const METHOD: &str = "pgd_fast_gradient_execution";
const SCHEMA: &str = "ny-cli/pgd-fast-gradient-execution/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PgdFastGradientRoute {
    Conjunctive,
    Disjunctive,
}

impl PgdFastGradientRoute {
    const fn token(self) -> &'static str {
        match self {
            Self::Conjunctive => "conjunctive",
            Self::Disjunctive => "disjunctive",
        }
    }
}

/// Per-invocation sequential-gradient evidence. Drop closes the receipt on
/// ordinary returns and `?` propagation alike; an interrupted decision remains
/// visibly inconsistent and cannot earn optimizer credit.
pub(super) struct PgdFastGradientExecution {
    route: PgdFastGradientRoute,
    configured: bool,
    decisions: Cell<usize>,
    fast_attempts: Cell<usize>,
    fast_selected: Cell<usize>,
    certified_selected: Cell<usize>,
    spsa_selected: Cell<usize>,
}

impl PgdFastGradientExecution {
    #[must_use]
    pub(super) const fn new(route: PgdFastGradientRoute, configured: bool) -> Self {
        Self {
            route,
            configured,
            decisions: Cell::new(0),
            fast_attempts: Cell::new(0),
            fast_selected: Cell::new(0),
            certified_selected: Cell::new(0),
            spsa_selected: Cell::new(0),
        }
    }

    #[must_use]
    pub(super) const fn configured(&self) -> bool {
        self.configured
    }

    /// Begin one governed sequential gradient decision.
    pub(super) fn begin_decision(&self) {
        self.decisions.set(self.decisions.get().saturating_add(1));
    }

    /// The configured fast point-Jacobian route was attempted.
    pub(super) fn attempted_fast(&self) {
        self.fast_attempts
            .set(self.fast_attempts.get().saturating_add(1));
    }

    /// The fast point-Jacobian supplied this decision's gradient.
    pub(super) fn selected_fast(&self) {
        self.fast_selected
            .set(self.fast_selected.get().saturating_add(1));
    }

    /// Certified two-sided CROWN supplied this decision's gradient.
    pub(super) fn selected_certified(&self) {
        self.certified_selected
            .set(self.certified_selected.get().saturating_add(1));
    }

    /// The caller selected SPSA after the exact routes declined or errored.
    /// This is recorded before SPSA starts so a propagated SPSA error still
    /// closes with the route decision that was actually made.
    pub(super) fn selected_spsa(&self) {
        self.spsa_selected
            .set(self.spsa_selected.get().saturating_add(1));
    }

    fn status(&self) -> FlightStatus {
        if self.decisions.get() == 0 {
            FlightStatus::Skipped
        } else {
            FlightStatus::Ran
        }
    }

    fn render(&self) -> String {
        format!(
            "schema={SCHEMA} phase=terminal route={} configured={} decisions={} \
             fast_attempts={} fast_selected={} certified_selected={} spsa_selected={}",
            self.route.token(),
            self.configured,
            self.decisions.get(),
            self.fast_attempts.get(),
            self.fast_selected.get(),
            self.certified_selected.get(),
            self.spsa_selected.get(),
        )
    }
}

impl Drop for PgdFastGradientExecution {
    fn drop(&mut self) {
        crate::flight::note(METHOD, self.status(), Some(self.render()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_reason_is_exact_for_enabled_and_disabled_decisions() {
        let enabled = PgdFastGradientExecution::new(PgdFastGradientRoute::Conjunctive, true);
        enabled.begin_decision();
        enabled.attempted_fast();
        enabled.selected_fast();
        assert_eq!(enabled.status(), FlightStatus::Ran);
        assert_eq!(
            enabled.render(),
            "schema=ny-cli/pgd-fast-gradient-execution/v1 phase=terminal \
             route=conjunctive configured=true decisions=1 fast_attempts=1 \
             fast_selected=1 certified_selected=0 spsa_selected=0",
        );

        let disabled = PgdFastGradientExecution::new(PgdFastGradientRoute::Disjunctive, false);
        disabled.begin_decision();
        disabled.selected_certified();
        assert_eq!(disabled.status(), FlightStatus::Ran);
        assert_eq!(
            disabled.render(),
            "schema=ny-cli/pgd-fast-gradient-execution/v1 phase=terminal \
             route=disjunctive configured=false decisions=1 fast_attempts=0 \
             fast_selected=0 certified_selected=1 spsa_selected=0",
        );
    }

    #[test]
    fn zero_decisions_are_explicitly_skipped() {
        let evidence = PgdFastGradientExecution::new(PgdFastGradientRoute::Conjunctive, true);
        assert_eq!(evidence.status(), FlightStatus::Skipped);
        assert!(evidence.render().contains("decisions=0 fast_attempts=0"));
    }
}
