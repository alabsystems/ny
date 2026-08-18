// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #phase-scheduler — the marginal-value loop.
//!
//! Design: `docs/DESIGN_MARGINAL_VALUE_SCHEDULER_2026-08-08.md` §2.4. Composes
//! the three invariants already built:
//!
//! - **I1** [`crate::phase_window`] — a window derives from predicted cost and
//!   is admitted only inside a fraction of what remains; an unaffordable phase
//!   declines rather than half-running.
//! - **I2** [`crate::phase_yield`] — a phase cannot express "abort the
//!   instance". [`PhaseYield::Declined`] is not an error.
//! - **I3** — a phase whose *realised* yield is ~0 for `k` blocks is retired and
//!   its budget returns to the pool.
//!
//! ## The unit
//!
//! `value = expected increase in min_r slack_r`, `price = seconds`. The loop
//! spends the next block wherever `value/second` is highest. That choice of
//! unit is what makes root tightening, α ascent and branching comparable at
//! all — today's static fractions are claims about *time* that nothing converts
//! into claims about *bound*.
//!
//! ## Why phases report their own cost
//!
//! [`PhaseRun::actual_cost`] is returned by the phase, not measured by the
//! scheduler. That keeps the scheduler pure bookkeeping over reported numbers,
//! so it is deterministic and testable without a clock — and it is also what a
//! real phase already has, since it must poll its own deadline anyway.
//!
//! ## Soundness
//!
//! The scheduler chooses *what runs and for how long*. It never touches a
//! coefficient, a bound, or a verdict. This is the same argument that licenses
//! every existing deadline in the tree.

use std::time::Duration;

use crate::phase_window::{admit, Admission, WindowPolicy};
use crate::phase_yield::PhaseYield;

/// What a phase returns from one block.
#[derive(Debug, Clone)]
pub struct PhaseRun {
    /// Realised value: the increase in `min_r slack_r` this block actually
    /// produced. The scheduler compares it against the prediction.
    pub yielded: PhaseYield<f64>,
    /// Wall time the phase actually consumed. May exceed the granted window —
    /// a phase that overruns is charged for it rather than being disbelieved.
    pub actual_cost: Duration,
}

/// A unit of work the scheduler can choose to run.
pub trait SchedulablePhase {
    fn name(&self) -> &'static str;
    /// Predicted seconds **on this host**, from a measured rate. Never a
    /// constant (invariant I1).
    fn predicted_cost(&self) -> Duration;
    /// Predicted increase in `min_r slack_r`. May be a sound CEILING — for
    /// branch-and-bound this is `Σⱼ gⱼ` from the gap-attribution theorem, which
    /// is exactly what lets the scheduler conclude "this cannot pay".
    fn predicted_value(&self) -> f64;
    /// Run cooperatively for at most `window`.
    ///
    /// The scheduler cannot preempt an implementation. If work overruns, it
    /// must report the full elapsed time in [`PhaseRun::actual_cost`]; the
    /// scheduler charges that time and stops admitting work once exhausted.
    fn run(&mut self, window: Duration) -> PhaseRun;
    /// Whether the phase is eligible right now (dependencies satisfied).
    fn is_ready(&self) -> bool {
        true
    }
}

/// Why the loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Every phase is retired, declined, or not ready.
    NoRunnablePhase,
    /// The budget is spent.
    BudgetExhausted,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhaseRecord {
    pub name: &'static str,
    pub blocks: u32,
    pub spent: Duration,
    pub realised_value: f64,
    pub retired_for_zero_yield: bool,
}

#[derive(Debug, Clone)]
pub struct SchedulerReport {
    pub stop: StopReason,
    pub spent: Duration,
    pub total_value: f64,
    pub per_phase: Vec<PhaseRecord>,
}

impl SchedulerReport {
    #[must_use]
    pub fn phase(&self, name: &str) -> Option<&PhaseRecord> {
        self.per_phase.iter().find(|r| r.name == name)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SchedulerPolicy {
    pub window: WindowPolicy,
    /// Consecutive ~zero-yield blocks before a phase is retired (invariant I3).
    pub zero_yield_blocks: u32,
    /// Below this, a realised value counts as zero.
    pub zero_epsilon: f64,
    /// Stop when less than this remains.
    pub min_remaining: Duration,
}

impl Default for SchedulerPolicy {
    fn default() -> Self {
        Self {
            window: WindowPolicy::default(),
            zero_yield_blocks: 2,
            zero_epsilon: 1e-9,
            min_remaining: Duration::from_millis(50),
        }
    }
}

/// Run phases in decreasing predicted value-per-second until the budget or the
/// runnable set is exhausted.
///
/// A phase is never admitted after the budget is exhausted. Cooperative phases
/// therefore spend at most `budget`; an implementation that exceeds its granted
/// window can make [`SchedulerReport::spent`] exceed `budget`, and the overrun
/// is charged truthfully. A `Declined` result never aborts the loop: it counts
/// as zero realised value for that block and may retire under I3 (invariant I2).
pub fn run_schedule(
    phases: &mut [Box<dyn SchedulablePhase>],
    budget: Duration,
    policy: SchedulerPolicy,
) -> SchedulerReport {
    let mut records: Vec<PhaseRecord> = phases
        .iter()
        .map(|p| PhaseRecord {
            name: p.name(),
            blocks: 0,
            spent: Duration::ZERO,
            realised_value: 0.0,
            retired_for_zero_yield: false,
        })
        .collect();
    let mut zero_streak = vec![0u32; phases.len()];
    let mut retired = vec![false; phases.len()];
    let mut spent = Duration::ZERO;
    let mut total_value = 0.0f64;

    let stop = loop {
        let remaining = budget.saturating_sub(spent);
        if remaining < policy.min_remaining {
            break StopReason::BudgetExhausted;
        }

        // Score every eligible phase by predicted value per second, and admit
        // it under I1. A phase that cannot be afforded is skipped, not fatal.
        let mut best: Option<(usize, f64, Duration)> = None;
        for (i, p) in phases.iter().enumerate() {
            if retired[i] || !p.is_ready() {
                continue;
            }
            let cost = p.predicted_cost();
            let Admission::Admitted(window) = admit(cost, remaining, policy.window) else {
                continue;
            };
            let secs = cost.as_secs_f64();
            // Direct comparison rather than `!(secs > 0.0)`: the NaN-catching form
            // buys nothing here because `secs` comes from `Duration::as_secs_f64`,
            // which is always finite and non-negative. This guards the zero-cost
            // phase, whose score would otherwise divide by zero.
            if secs <= 0.0 {
                continue;
            }
            let score = p.predicted_value() / secs;
            if !score.is_finite() || score <= 0.0 {
                continue;
            }
            if best.is_none_or(|(_, b, _)| score > b) {
                best = Some((i, score, window));
            }
        }

        let Some((i, _, window)) = best else {
            break StopReason::NoRunnablePhase;
        };

        let run = phases[i].run(window);
        // Charge the ACTUAL cost, including an overrun. Believing the granted
        // window over the observed time is how a budget silently goes negative.
        spent = spent.saturating_add(run.actual_cost);
        records[i].blocks += 1;
        records[i].spent = records[i].spent.saturating_add(run.actual_cost);

        let realised = match run.yielded {
            PhaseYield::Complete(v) | PhaseYield::Partial(v) => v,
            // Declined is not an error and not a failure of the run — the phase
            // simply produced nothing this block (invariant I2).
            PhaseYield::Declined(_) => 0.0,
        };
        records[i].realised_value += realised;
        total_value += realised;

        // I3: retire on measured yield, not on a guess about iterations.
        if realised.abs() <= policy.zero_epsilon {
            zero_streak[i] += 1;
            if zero_streak[i] >= policy.zero_yield_blocks {
                retired[i] = true;
                records[i].retired_for_zero_yield = true;
            }
        } else {
            zero_streak[i] = 0;
        }
    };

    SchedulerReport {
        stop,
        spent,
        total_value,
        per_phase: records,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase_yield::DeclineReason;

    /// Scripted phase: fixed prediction, a queue of realised values.
    struct Mock {
        name: &'static str,
        cost: Duration,
        value: f64,
        script: Vec<f64>,
        cursor: usize,
        ready: bool,
        overrun: Duration,
    }

    impl Mock {
        /// Named `boxed`, not `new`: every caller wants the trait object, so
        /// returning `Self` here would just make all 11 of them write `Box::new`.
        fn boxed(
            name: &'static str,
            cost_s: u64,
            value: f64,
            script: Vec<f64>,
        ) -> Box<dyn SchedulablePhase> {
            Box::new(Self {
                name,
                cost: Duration::from_secs(cost_s),
                value,
                script,
                cursor: 0,
                ready: true,
                overrun: Duration::ZERO,
            })
        }
    }

    impl SchedulablePhase for Mock {
        fn name(&self) -> &'static str {
            self.name
        }
        fn predicted_cost(&self) -> Duration {
            self.cost
        }
        fn predicted_value(&self) -> f64 {
            self.value
        }
        fn is_ready(&self) -> bool {
            self.ready
        }
        fn run(&mut self, window: Duration) -> PhaseRun {
            let v = self.script.get(self.cursor).copied().unwrap_or(0.0);
            self.cursor += 1;
            PhaseRun {
                yielded: PhaseYield::Complete(v),
                actual_cost: window + self.overrun,
            }
        }
    }

    fn policy() -> SchedulerPolicy {
        SchedulerPolicy {
            window: WindowPolicy::default().with_max_frac(0.9),
            ..SchedulerPolicy::default()
        }
    }

    #[test]
    fn spends_on_the_highest_value_per_second_not_the_highest_value() {
        // `cheap` has less absolute value but 10x the rate. A scheduler that
        // ranked on value alone would pick `rich` and be wrong.
        let mut phases = vec![
            Mock::boxed("rich", 100, 50.0, vec![50.0]),  // 0.5 / s
            Mock::boxed("cheap", 1, 5.0, vec![5.0; 40]), // 5.0 / s
        ];
        let r = run_schedule(&mut phases, Duration::from_mins(1), policy());
        assert!(r.phase("cheap").unwrap().blocks > 0);
        assert_eq!(r.phase("rich").unwrap().blocks, 0, "rich is the wrong pick");
    }

    #[test]
    fn retires_a_zero_yield_phase_and_reallocates_to_a_productive_one() {
        // THE alpha-ascent case: a phase that predicts value and delivers none.
        // Measured: best_impr = 0.000e0 for seven consecutive iterations.
        let mut phases = vec![
            Mock::boxed("ascent", 1, 100.0, vec![0.0; 50]), // predicts high, yields nothing
            Mock::boxed("useful", 1, 1.0, vec![1.0; 50]),
        ];
        let r = run_schedule(&mut phases, Duration::from_secs(30), policy());
        let a = r.phase("ascent").unwrap();
        assert!(
            a.retired_for_zero_yield,
            "must be retired on measured yield"
        );
        assert_eq!(a.blocks, 2, "retired after zero_yield_blocks, not before");
        assert!(
            r.phase("useful").unwrap().blocks > 5,
            "the freed budget must go somewhere"
        );
    }

    #[test]
    fn an_unaffordable_phase_is_skipped_not_fatal() {
        // I1 + I2 together: the 82s forward-linear build against a 95s tier is
        // declined, and the loop continues with what it can afford.
        let mut phases = vec![
            Mock::boxed("fl_root", 82, 1000.0, vec![1000.0]),
            Mock::boxed("small", 1, 1.0, vec![1.0; 100]),
        ];
        let r = run_schedule(&mut phases, Duration::from_secs(95), policy());
        assert_eq!(r.phase("fl_root").unwrap().blocks, 0, "declined, not run");
        assert!(r.phase("small").unwrap().blocks > 0, "loop must continue");
    }

    #[test]
    fn cooperative_phase_does_not_overspend_the_budget() {
        let mut phases = vec![Mock::boxed("p", 1, 1.0, vec![1.0; 1000])];
        let budget = Duration::from_secs(10);
        let r = run_schedule(&mut phases, budget, policy());
        assert!(
            r.spent <= budget,
            "spent {:?} exceeds {:?}",
            r.spent,
            budget
        );
    }

    #[test]
    fn an_overrunning_phase_is_charged_what_it_actually_took() {
        // Believing the granted window over observed time is how a budget goes
        // silently negative. The margin-row lane overran a 45s slice by 113.1s.
        let p = Mock {
            name: "greedy",
            cost: Duration::from_secs(1),
            value: 1.0,
            script: vec![1.0; 100],
            cursor: 0,
            ready: true,
            overrun: Duration::from_secs(20),
        };
        let mut phases: Vec<Box<dyn SchedulablePhase>> = vec![Box::new(p)];
        let r = run_schedule(&mut phases, Duration::from_secs(20), policy());
        let rec = r.phase("greedy").unwrap();
        assert!(
            r.spent > Duration::from_secs(20),
            "an implementation overrun must remain visible in the report"
        );
        assert!(
            rec.spent.as_secs_f64() / f64::from(rec.blocks) > 20.0,
            "the block must be charged its real ~21.25s, not the 1.25s granted"
        );
        assert!(
            r.spent <= Duration::from_secs(26),
            "and the loop still stops"
        );
    }

    #[test]
    fn a_phase_that_is_not_ready_is_never_scheduled() {
        let mut blocked = Mock {
            name: "blocked",
            cost: Duration::from_secs(1),
            value: 1000.0,
            script: vec![1000.0; 10],
            cursor: 0,
            ready: false,
            overrun: Duration::ZERO,
        };
        blocked.ready = false;
        let mut phases: Vec<Box<dyn SchedulablePhase>> =
            vec![Box::new(blocked), Mock::boxed("ok", 1, 1.0, vec![1.0; 10])];
        let r = run_schedule(&mut phases, Duration::from_secs(5), policy());
        assert_eq!(r.phase("blocked").unwrap().blocks, 0);
    }

    #[test]
    fn stops_cleanly_when_nothing_is_runnable() {
        // Every phase declines: a huge budget must not spin.
        let mut phases = vec![Mock::boxed("huge", 10_000, 1.0, vec![1.0])];
        let r = run_schedule(&mut phases, Duration::from_secs(10), policy());
        assert_eq!(r.stop, StopReason::NoRunnablePhase);
        assert_eq!(r.spent, Duration::ZERO);
    }

    #[test]
    fn a_declined_yield_counts_as_zero_and_can_retire_the_phase() {
        struct Decliner;
        impl SchedulablePhase for Decliner {
            fn name(&self) -> &'static str {
                "decliner"
            }
            fn predicted_cost(&self) -> Duration {
                Duration::from_secs(1)
            }
            fn predicted_value(&self) -> f64 {
                10.0
            }
            fn run(&mut self, window: Duration) -> PhaseRun {
                PhaseRun {
                    yielded: PhaseYield::Declined(DeclineReason::Empty),
                    actual_cost: window,
                }
            }
        }
        let mut phases: Vec<Box<dyn SchedulablePhase>> = vec![Box::new(Decliner)];
        let r = run_schedule(&mut phases, Duration::from_mins(1), policy());
        assert!(r.phase("decliner").unwrap().retired_for_zero_yield);
        assert!(
            r.spent < Duration::from_secs(10),
            "must not burn the budget"
        );
    }

    #[test]
    fn the_measured_cifar100_shape_reallocates_from_the_ascent_to_search() {
        // The campaign's actual numbers: the alpha ascent predicts improvement,
        // measures best_impr = 0.000e0, and holds ~18s of a ~40s window while
        // branch-and-bound is starved. The scheduler must take that back.
        let mut phases = vec![
            Mock::boxed("alpha_ascent", 5, 10.0, vec![0.0; 20]),
            Mock::boxed("bab", 5, 1.0, vec![0.25; 20]),
        ];
        let r = run_schedule(&mut phases, Duration::from_secs(90), policy());
        let a = r.phase("alpha_ascent").unwrap();
        let b = r.phase("bab").unwrap();
        assert!(a.retired_for_zero_yield);
        assert!(
            b.spent > a.spent,
            "search must end up with more time than the phase that yields nothing \
             (ascent {:?}, bab {:?})",
            a.spent,
            b.spent
        );
        assert!(r.total_value > 0.0);
    }
}
