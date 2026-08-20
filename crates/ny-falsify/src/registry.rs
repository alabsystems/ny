// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The registry: one admission ladder, one budget arithmetic, one place a new
//! strategy is added.
//!
//! Today ny has eleven falsification entry points in one file, each with its
//! own admission test, its own budget arithmetic, its own witness rendering and
//! its own comment explaining why it is sound. Adding a twelfth means writing
//! all four again. This is the fourth one, written once.

use crate::admission::{
    Admission, AdmissionContext, AdmissionReceipt, AdmissionStage, Arming, CostClass, Decline,
    FactLadder, ObjectiveQuality, ParamSpace, SpecFacts,
};
use crate::domain::SearchBox;
use crate::oracle::Oracle;
use crate::proposal::{Proposal, StrategyName};
use crate::stall::StallRule;
use core::time::Duration;
use std::time::Instant;

/// Incumbent state shared across the strategies of one row.
///
/// This is how `special` pays for `square`: `special` runs first, costs eight
/// points, and leaves behind the best free-coordinate vector it saw. `square`
/// starts every one of its restarts from that vector rather than from the box
/// centre. In the calibration, both `square` wins ran after `special` had
/// already contributed its eight points.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchState {
    best_free: Vec<f64>,
    best_steer: f64,
}

impl SearchState {
    /// Seed the state at the box centre, which is where the Python portfolio
    /// starts it.
    pub fn at_centre(domain: &SearchBox) -> Self {
        Self {
            best_free: domain.centre_free().to_vec(),
            best_steer: f64::NEG_INFINITY,
        }
    }

    /// The best free-coordinate vector seen so far.
    pub fn best_free(&self) -> &[f64] {
        &self.best_free
    }

    /// The best steering margin seen so far.
    pub const fn best_steer(&self) -> f64 {
        self.best_steer
    }

    /// Offer a new incumbent. Accepted only on a strict improvement, so the
    /// incumbent is monotone and a stall counter over it is meaningful.
    pub fn offer(&mut self, free: &[f64], steer: f64) -> bool {
        if steer > self.best_steer {
            self.best_steer = steer;
            self.best_free.clear();
            self.best_free.extend_from_slice(free);
            true
        } else {
            false
        }
    }
}

/// One strategy's allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slice {
    /// Which strategy.
    pub strategy: StrategyName,
    /// Wall-clock allowance.
    pub allowance: Duration,
    /// Ceilings and widths for this allowance.
    pub params: ParamSpace,
    /// When to abandon early.
    pub stall_rule: StallRule,
}

/// A slice, resolved against the clock and the oracle.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// Hard stop.
    pub deadline: Instant,
    /// Points per oracle call.
    pub batch: usize,
    /// Ceilings and widths.
    pub params: ParamSpace,
    /// When to abandon early.
    pub stall_rule: StallRule,
}

impl Budget {
    /// Whether the allowance is spent.
    pub fn expired(&self) -> bool {
        Instant::now() >= self.deadline
    }
}

/// A falsification strategy. Proposes inputs; cannot name a verdict.
pub trait Strategy {
    /// Which strategy this is.
    fn name(&self) -> StrategyName;

    /// The deepest evidence stage its admission decision needs. The registry
    /// will not descend the ladder past the deepest stage any surviving
    /// strategy asked for, which is what keeps refusals free.
    fn deepest_stage(&self) -> AdmissionStage;

    /// This strategy's share of the fractional budget plan, as in the Python
    /// portfolio's `DEFAULT_PLAN`. Shares are renormalised over the admitted
    /// set, so removing a strategy does not silently leave budget unspent.
    fn plan_share(&self) -> f64;

    /// Its own default stall rule, in its own work units.
    fn stall_rule(&self) -> StallRule;

    /// Admission. Pure in `(spec, graph, remaining budget, arming, objective)`.
    fn admit(&self, ctx: &AdmissionContext<'_>) -> Admission;

    /// Search. Returns inputs, never a verdict.
    fn search(
        &mut self,
        domain: &SearchBox,
        oracle: &mut dyn Oracle,
        budget: &Budget,
        state: &mut SearchState,
    ) -> Proposal;
}

/// What one full falsification phase produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Receipt {
    /// Per-strategy admission decisions, with elapsed times.
    pub admissions: Vec<AdmissionReceipt>,
    /// Per-strategy search outcomes, in the order they ran.
    pub proposals: Vec<(StrategyName, Proposal)>,
    /// Total wall time of the phase.
    pub elapsed: Duration,
}

impl Receipt {
    /// The first candidate produced, if any. The caller hands this, and only
    /// this, to its unchanged trusted-oracle gate.
    pub fn candidate(&self) -> Option<&crate::proposal::Candidate> {
        self.proposals
            .iter()
            .find_map(|(_, proposal)| match proposal {
                Proposal::Candidate(candidate) => Some(candidate),
                _ => None,
            })
    }
}

/// The strategy registry.
pub struct Registry {
    strategies: Vec<Box<dyn Strategy>>,
    arming: Arming,
}

impl core::fmt::Debug for Registry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Registry")
            .field(
                "strategies",
                &self.strategies.iter().map(|s| s.name()).collect::<Vec<_>>(),
            )
            .field("arming", &self.arming)
            .finish()
    }
}

impl Default for Registry {
    /// Empty and **dark**. Constructing a registry arms nothing.
    fn default() -> Self {
        Self {
            strategies: Vec::new(),
            arming: Arming::default(),
        }
    }
}

impl Registry {
    /// A registry holding the ported strategies, still dark.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a strategy.
    #[must_use]
    pub fn with(mut self, strategy: Box<dyn Strategy>) -> Self {
        self.strategies.push(strategy);
        self
    }

    /// Arm the registry. The caller does this from behind its own declared
    /// lever; this crate reads no process environment.
    #[must_use]
    pub const fn armed(mut self) -> Self {
        self.arming = Arming::Armed;
        self
    }

    /// Current arming.
    pub const fn arming(&self) -> Arming {
        self.arming
    }

    /// Registered strategy names, in registration order.
    pub fn names(&self) -> Vec<StrategyName> {
        self.strategies.iter().map(|s| s.name()).collect()
    }

    /// Run admission against a lazily-descended evidence ladder.
    ///
    /// The ladder is descended in fixed order and no deeper than needed:
    /// `spec_facts` always; `graph_facts` only if a strategy that survived the
    /// spec stages declares `GraphScan`; `load_model` only for `ModelLoad`.
    /// Neither ported strategy declares deeper than `SpecShape`, so on a
    /// refusal the ONNX graph is never touched.
    pub fn admit(
        &self,
        ladder: &mut dyn FactLadder,
        remaining: Duration,
        objective: ObjectiveQuality,
    ) -> Result<Vec<AdmissionReceipt>, Decline> {
        let spec: SpecFacts = ladder.spec_facts()?;

        let mut receipts = Vec::with_capacity(self.strategies.len());
        let mut deferred = Vec::new();
        for strategy in &self.strategies {
            if strategy.deepest_stage() <= AdmissionStage::SpecShape {
                let started = Instant::now();
                let ctx = AdmissionContext::new(&spec, remaining, self.arming, objective);
                let admission = strategy.admit(&ctx);
                receipts.push(AdmissionReceipt {
                    strategy: strategy.name(),
                    admission,
                    elapsed: started.elapsed(),
                    deepest_stage_reached: AdmissionStage::SpecShape,
                });
            } else {
                deferred.push(strategy);
            }
        }

        if !deferred.is_empty() {
            let graph = ladder.graph_facts()?;
            let mut needs_model = false;
            for strategy in &deferred {
                let started = Instant::now();
                let ctx = AdmissionContext::new(&spec, remaining, self.arming, objective)
                    .with_graph(&graph);
                let admission = strategy.admit(&ctx);
                if matches!(admission, Admission::Admitted(_))
                    && strategy.deepest_stage() == AdmissionStage::ModelLoad
                {
                    needs_model = true;
                }
                receipts.push(AdmissionReceipt {
                    strategy: strategy.name(),
                    admission,
                    elapsed: started.elapsed(),
                    deepest_stage_reached: AdmissionStage::GraphScan,
                });
            }
            if needs_model {
                ladder.load_model()?;
            }
        }

        Ok(receipts)
    }

    /// Allocate `total` across the admitted strategies.
    ///
    /// Ordering is `Instant` -> `Bounded` -> `Openended`, and that ordering is
    /// forced by arithmetic rather than preference: a stalled or refused cheap
    /// lane yields in fractions of a second, so putting it first costs nothing,
    /// while putting an open-ended lane first can overrun the instance budget
    /// outright (STE-first at 240 s + LP worst case 131.1 s + 45 s publication
    /// margin + 100 s downstream reserve = 517 s against 456 s available).
    pub fn schedule(&self, total: Duration, receipts: &[AdmissionReceipt]) -> Vec<Slice> {
        let mut admitted: Vec<(&Box<dyn Strategy>, ParamSpace, CostClass)> = Vec::new();
        for receipt in receipts {
            if let Admission::Admitted(profile) = &receipt.admission {
                if let Some(strategy) = self
                    .strategies
                    .iter()
                    .find(|s| s.name() == receipt.strategy)
                {
                    admitted.push((strategy, profile.params, profile.declared_cost));
                }
            }
        }
        admitted.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.name().cmp(&b.0.name())));

        let share_total: f64 = admitted.iter().map(|(s, _, _)| s.plan_share()).sum();
        if share_total <= 0.0 {
            return Vec::new();
        }
        admitted
            .into_iter()
            .map(|(strategy, params, _)| Slice {
                strategy: strategy.name(),
                allowance: total.mul_f64(strategy.plan_share() / share_total),
                params,
                stall_rule: strategy.stall_rule(),
            })
            .collect()
    }

    /// Admit, schedule, and run, stopping at the first candidate.
    pub fn run(
        &mut self,
        ladder: &mut dyn FactLadder,
        domain: &SearchBox,
        oracle: &mut dyn Oracle,
        total: Duration,
        objective: ObjectiveQuality,
    ) -> Result<Receipt, Decline> {
        let started = Instant::now();
        let admissions = self.admit(ladder, total, objective)?;
        let slices = self.schedule(total, &admissions);

        let batch = oracle.batch_limit().max(1);
        let mut state = SearchState::at_centre(domain);
        let mut proposals = Vec::with_capacity(slices.len());
        for slice in slices {
            let remaining = total.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                break;
            }
            let budget = Budget {
                deadline: Instant::now() + slice.allowance.min(remaining),
                batch,
                params: slice.params,
                stall_rule: slice.stall_rule,
            };
            let Some(strategy) = self
                .strategies
                .iter_mut()
                .find(|s| s.name() == slice.strategy)
            else {
                continue;
            };
            let proposal = strategy.search(domain, oracle, &budget, &mut state);
            let done = matches!(proposal, Proposal::Candidate(_));
            proposals.push((slice.strategy, proposal));
            if done {
                break;
            }
        }

        Ok(Receipt {
            admissions,
            proposals,
            elapsed: started.elapsed(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::{AdmissionProfile, ObjectiveRequirement, SpecShape};
    use crate::strategies::{SpecialPoints, Square};

    struct Ladder(SpecFacts);
    impl FactLadder for Ladder {
        fn spec_facts(&mut self) -> Result<SpecFacts, Decline> {
            Ok(self.0.clone())
        }
        fn graph_facts(&mut self) -> Result<crate::admission::GraphFacts, Decline> {
            panic!("no ported strategy needs the graph");
        }
        fn load_model(&mut self) -> Result<(), Decline> {
            panic!("no ported strategy needs the model");
        }
    }

    fn spec(free: usize) -> SpecFacts {
        SpecFacts {
            free_dims: free,
            pinned_dims: 0,
            shape: SpecShape::BoxInputs,
            disjunct_targets: 1,
            has_equality_atoms: false,
        }
    }

    #[test]
    fn the_schedule_puts_the_instant_lane_first_and_renormalises_the_plan() {
        // Registration order is deliberately the WRONG one, so the ordering
        // assertion is about CostClass and not about insertion.
        let registry = Registry::new()
            .with(Box::new(Square::default()))
            .with(Box::new(SpecialPoints))
            .armed();
        let mut ladder = Ladder(spec(128));
        let receipts = registry
            .admit(
                &mut ladder,
                Duration::from_secs(100),
                ObjectiveQuality::Flat,
            )
            .unwrap();
        let slices = registry.schedule(Duration::from_secs(100), &receipts);

        assert_eq!(slices.len(), 2);
        assert_eq!(slices[0].strategy, StrategyName::SpecialPoints);
        assert_eq!(slices[1].strategy, StrategyName::Square);

        // DEFAULT_PLAN gives special 0.02 and square 0.16 out of 1.00. With the
        // other nine strategies absent the shares are renormalised over 0.18 so
        // no budget is silently left unspent.
        let total: f64 = slices.iter().map(|s| s.allowance.as_secs_f64()).sum();
        assert!((total - 100.0).abs() < 1e-6, "allocated {total} s of 100");
        assert!((slices[0].allowance.as_secs_f64() - 100.0 * 0.02 / 0.18).abs() < 1e-6);
        assert!((slices[1].allowance.as_secs_f64() - 100.0 * 0.16 / 0.18).abs() < 1e-6);

        // And each carries its OWN stall rule and its OWN ceilings.
        assert_ne!(slices[0].stall_rule, slices[1].stall_rule);
        assert_ne!(
            slices[0].params.free_dims_ceiling,
            slices[1].params.free_dims_ceiling
        );
    }

    #[test]
    fn a_declined_strategy_gets_no_slice() {
        let registry = Registry::new()
            .with(Box::new(SpecialPoints))
            .with(Box::new(Square::default()))
            .armed();
        // One free dimension: square declines on its own floor.
        let mut ladder = Ladder(spec(1));
        let receipts = registry
            .admit(
                &mut ladder,
                Duration::from_mins(1),
                ObjectiveQuality::Informative,
            )
            .unwrap();
        let slices = registry.schedule(Duration::from_mins(1), &receipts);
        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].strategy, StrategyName::SpecialPoints);
        // The declined lane still left a receipt.
        assert!(receipts
            .iter()
            .any(|r| matches!(r.admission, Admission::Declined(_))));
    }

    #[test]
    fn the_incumbent_is_monotone_so_a_stall_counter_over_it_means_something() {
        let domain = SearchBox::new(&[0.0, 0.0], &[1.0, 1.0]).unwrap();
        let mut state = SearchState::at_centre(&domain);
        assert_eq!(state.best_steer(), f64::NEG_INFINITY);
        assert!(state.offer(&[0.25, 0.25], -3.0));
        assert!(
            !state.offer(&[0.75, 0.75], -3.0),
            "ties are not improvements"
        );
        assert_eq!(state.best_free(), &[0.25, 0.25]);
        assert!(state.offer(&[0.9, 0.9], -2.0));
        assert_eq!(state.best_free(), &[0.9, 0.9]);
    }

    #[test]
    fn an_admitted_profile_reports_the_reach_axes_the_scheduler_allocates_on() {
        let registry = Registry::new().with(Box::new(SpecialPoints)).armed();
        let mut ladder = Ladder(spec(6912));
        let receipts = registry
            .admit(&mut ladder, Duration::from_mins(8), ObjectiveQuality::Flat)
            .unwrap();
        let Admission::Admitted(AdmissionProfile {
            objective,
            needs_incumbent,
            needs_verifier_state,
            free_dims,
            declared_cost,
            ..
        }) = receipts[0].admission
        else {
            panic!("special should be admitted");
        };
        assert_eq!(objective, ObjectiveRequirement::ValueOnly);
        assert!(!needs_incumbent);
        assert!(!needs_verifier_state);
        assert_eq!(free_dims, 6912);
        assert_eq!(declared_cost, CostClass::Instant);
    }
}
