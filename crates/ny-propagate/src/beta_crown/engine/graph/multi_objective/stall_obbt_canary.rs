// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Measurement-only NeuralSAT-style stall telemetry for multi-objective BaB.
//!
//! Exact `NY_MO_STALL_OBBT_CANARY=1` enables this observer. Every other value,
//! including an unset variable, is OFF. The observer keeps only constant-size
//! counters and samples the heap's existing frontier reference immediately
//! before the normal batch pop. It never receives a mutable domain/queue,
//! returns no scheduling decision, and has no AY/MIP or proof-authority seam.
//!
//! Progress is `-domain.priority()`, not an unconditional lower bound:
//!
//! - lower-bound verification stores priority `-lower`, so progress is `lower`;
//! - upper-bound verification stores priority `upper`, so progress is `-upper`;
//! - conjunctive priority policy is already represented by the domain priority.
//!
//! Consequently, larger progress uniformly means a stronger frontier. The
//! patience update mirrors NeuralSAT: improvement or a non-growing queue
//! subtracts one, an equal frontier with queue growth adds one, and a regressing
//! frontier with queue growth adds three. Reaching ten emits a bounded
//! `would_trigger` record and resets the private score. This canary does not
//! tighten anything.

use std::collections::BinaryHeap;
use std::fmt;
use std::io::Write;

use crate::beta_crown::domain::MultiObjectiveGraphBabDomain;

const ENV_GATE: &str = "NY_MO_STALL_OBBT_CANARY";
const PATIENCE_LIMIT: usize = 10;
/// Bound stderr overhead independently of BaB depth/domain count.
const MAX_TRIGGER_LINES: usize = 8;

/// Exactly `"1"` enables. Keeping the predicate pure avoids environment races
/// in deterministic unit tests.
fn gate_on(raw: Option<&str>) -> bool {
    raw == Some("1")
}

fn emit(args: fmt::Arguments<'_>) {
    // Telemetry must not turn a closed/broken stderr into a verifier failure.
    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(stderr, "{args}");
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FrontierSnapshot {
    progress: f32,
    depth: usize,
    pure_relu_history: bool,
}

fn frontier_snapshot(domain: Option<&MultiObjectiveGraphBabDomain>) -> Option<FrontierSnapshot> {
    let domain = domain?;
    let progress = -domain.priority();
    progress.is_finite().then(|| FrontierSnapshot {
        progress,
        depth: domain.depth(),
        pure_relu_history: domain.history().is_pure_relu_at_zero(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct WouldTrigger {
    observation: usize,
    trigger: usize,
    progress: f32,
    previous_progress: f32,
    queue_len: usize,
    previous_queue_len: usize,
    batch_size: usize,
    patience_score: usize,
    depth: usize,
    pure_relu_history: bool,
}

/// Constant-space, print-only observer for one multi-objective BaB run.
pub(super) struct StallObbtCanary {
    emit_lines: bool,
    score: usize,
    peak_score: usize,
    last_progress: Option<f32>,
    last_queue_len: usize,
    observations: usize,
    invalid_frontiers: usize,
    triggers: usize,
    emitted_triggers: usize,
    max_queue_len: usize,
}

impl StallObbtCanary {
    fn configured(emit_lines: bool) -> Self {
        Self {
            emit_lines,
            score: 0,
            peak_score: 0,
            last_progress: None,
            last_queue_len: 0,
            observations: 0,
            invalid_frontiers: 0,
            triggers: 0,
            emitted_triggers: 0,
            max_queue_len: 0,
        }
    }

    /// `None` is the hard default-off path. The verification loop therefore
    /// does not even inspect the queue frontier when the exact gate is absent.
    pub(super) fn from_env() -> Option<Self> {
        gate_on(std::env::var(ENV_GATE).ok().as_deref()).then(|| Self::configured(true))
    }

    /// Observe the same immutable heap frontier that the next normal batch pop
    /// will consume. No iterator, allocation, graph scan, or clock read occurs.
    pub(super) fn observe_queue(
        &mut self,
        queue: &BinaryHeap<MultiObjectiveGraphBabDomain>,
        batch_size: usize,
    ) {
        let event = self.observe(frontier_snapshot(queue.peek()), queue.len(), batch_size);
        if let Some(event) = event {
            self.emit_trigger(event);
        }
    }

    fn observe(
        &mut self,
        snapshot: Option<FrontierSnapshot>,
        queue_len: usize,
        batch_size: usize,
    ) -> Option<WouldTrigger> {
        self.observations = self.observations.saturating_add(1);
        self.max_queue_len = self.max_queue_len.max(queue_len);

        let Some(snapshot) = snapshot else {
            self.invalid_frontiers = self.invalid_frontiers.saturating_add(1);
            self.score = 0;
            self.last_progress = None;
            self.last_queue_len = queue_len;
            return None;
        };

        let Some(previous_progress) = self.last_progress else {
            self.last_progress = Some(snapshot.progress);
            self.last_queue_len = queue_len;
            return None;
        };
        let previous_queue_len = self.last_queue_len;

        if snapshot.progress > previous_progress
            || queue_len <= batch_size
            || queue_len <= previous_queue_len
        {
            self.score = self.score.saturating_sub(1);
        } else if snapshot.progress == previous_progress {
            self.score = self.score.saturating_add(1);
        } else {
            self.score = self.score.saturating_add(3);
        }
        self.peak_score = self.peak_score.max(self.score);
        self.last_progress = Some(snapshot.progress);
        self.last_queue_len = queue_len;

        if queue_len <= batch_size || self.score < PATIENCE_LIMIT {
            return None;
        }

        let patience_score = self.score;
        self.score = 0;
        self.triggers = self.triggers.saturating_add(1);
        Some(WouldTrigger {
            observation: self.observations,
            trigger: self.triggers,
            progress: snapshot.progress,
            previous_progress,
            queue_len,
            previous_queue_len,
            batch_size,
            patience_score,
            depth: snapshot.depth,
            pure_relu_history: snapshot.pure_relu_history,
        })
    }

    fn emit_trigger(&mut self, event: WouldTrigger) {
        if !self.emit_lines || self.emitted_triggers >= MAX_TRIGGER_LINES {
            return;
        }
        self.emitted_triggers = self.emitted_triggers.saturating_add(1);
        emit(format_args!(
            "[mo-stall-obbt-canary] would_trigger=true observation={} trigger={} progress={:.9e} previous_progress={:.9e} queue_len={} previous_queue_len={} batch_size={} patience_score={} patience_limit={} depth={} pure_relu_history={} action=none",
            event.observation,
            event.trigger,
            event.progress,
            event.previous_progress,
            event.queue_len,
            event.previous_queue_len,
            event.batch_size,
            event.patience_score,
            PATIENCE_LIMIT,
            event.depth,
            event.pure_relu_history,
        ));
    }
}

impl Drop for StallObbtCanary {
    fn drop(&mut self) {
        if !self.emit_lines {
            return;
        }
        emit(format_args!(
            "[mo-stall-obbt-canary-summary] observations={} invalid_frontiers={} would_trigger_count={} emitted_triggers={} suppressed_triggers={} peak_patience_score={} max_queue_len={} patience_limit={} action=none",
            self.observations,
            self.invalid_frontiers,
            self.triggers,
            self.emitted_triggers,
            self.triggers.saturating_sub(self.emitted_triggers),
            self.peak_score,
            self.max_queue_len,
            PATIENCE_LIMIT,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use ndarray::arr1;
    use ny_tensor::BoundedTensor;

    fn snapshot(progress: f32) -> FrontierSnapshot {
        FrontierSnapshot {
            progress,
            depth: 4,
            pure_relu_history: true,
        }
    }

    fn domain(lower: f32, upper: f32, verify_upper: bool) -> MultiObjectiveGraphBabDomain {
        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
        MultiObjectiveGraphBabDomain::root(
            HashMap::new(),
            vec![(lower, upper)],
            &input,
            &[0.0],
            verify_upper,
        )
        .unwrap()
    }

    #[test]
    fn gate_is_default_off_and_only_exact_one_enables() {
        assert!(!gate_on(None));
        assert!(!gate_on(Some("")));
        assert!(!gate_on(Some("0")));
        assert!(!gate_on(Some("true")));
        assert!(!gate_on(Some("01")));
        assert!(gate_on(Some("1")));
    }

    #[test]
    fn frontier_progress_is_direction_correct() {
        let lower_mode = domain(-2.0, 4.0, false);
        let upper_mode = domain(-2.0, 4.0, true);
        assert_eq!(
            frontier_snapshot(Some(&lower_mode)).unwrap().progress,
            -2.0,
            "lower verification improves when the lower frontier increases"
        );
        assert_eq!(
            frontier_snapshot(Some(&upper_mode)).unwrap().progress,
            -4.0,
            "upper verification improves when the upper frontier decreases"
        );
    }

    #[test]
    fn equal_frontier_with_growing_queue_triggers_at_patience() {
        let mut canary = StallObbtCanary::configured(false);
        assert_eq!(canary.observe(Some(snapshot(-1.0)), 2, 1), None);
        for queue_len in 3..12 {
            assert_eq!(canary.observe(Some(snapshot(-1.0)), queue_len, 1), None);
        }
        let event = canary
            .observe(Some(snapshot(-1.0)), 12, 1)
            .expect("tenth equal/growing observation triggers");
        assert_eq!(event.patience_score, PATIENCE_LIMIT);
        assert_eq!(event.queue_len, 12);
        assert_eq!(canary.score, 0);
        assert_eq!(canary.triggers, 1);
    }

    #[test]
    fn regressing_frontier_adds_three_but_improvement_and_non_growth_subtract() {
        let mut canary = StallObbtCanary::configured(false);
        canary.observe(Some(snapshot(-1.0)), 2, 1);
        canary.observe(Some(snapshot(-2.0)), 3, 1);
        assert_eq!(canary.score, 3);
        canary.observe(Some(snapshot(-1.5)), 4, 1);
        assert_eq!(canary.score, 2, "frontier improvement subtracts one");
        canary.observe(Some(snapshot(-1.5)), 4, 1);
        assert_eq!(canary.score, 1, "non-growing queue subtracts one");
        canary.observe(Some(snapshot(-1.5)), 1, 1);
        assert_eq!(canary.score, 0, "one-batch frontier cannot trigger");
        assert_eq!(canary.triggers, 0);
    }

    #[test]
    fn invalid_frontier_resets_private_patience_state() {
        let mut canary = StallObbtCanary::configured(false);
        canary.observe(Some(snapshot(-1.0)), 2, 1);
        canary.observe(Some(snapshot(-2.0)), 3, 1);
        assert_eq!(canary.score, 3);
        assert_eq!(canary.observe(None, 3, 1), None);
        assert_eq!(canary.score, 0);
        assert_eq!(canary.last_progress, None);
        assert_eq!(canary.invalid_frontiers, 1);
    }

    #[test]
    fn loop_boundary_observation_cannot_pop_or_reorder_the_real_heap() {
        let mut queue = BinaryHeap::new();
        queue.push(domain(-1.0, 3.0, false));
        queue.push(domain(-2.0, 4.0, false));
        let before_len = queue.len();
        let before_priorities: Vec<u32> = queue.iter().map(|d| d.priority().to_bits()).collect();
        let before_frontier = queue.peek().unwrap().priority().to_bits();

        let mut canary = StallObbtCanary::configured(false);
        canary.observe_queue(&queue, 1);

        assert_eq!(queue.len(), before_len);
        assert_eq!(queue.peek().unwrap().priority().to_bits(), before_frontier);
        assert_eq!(
            queue
                .iter()
                .map(|d| d.priority().to_bits())
                .collect::<Vec<_>>(),
            before_priorities
        );
        assert_eq!(canary.observations, 1);
        assert_eq!(
            canary.last_progress.unwrap().to_bits(),
            (-queue.peek().unwrap().priority()).to_bits()
        );
    }

    #[test]
    fn trigger_output_is_hard_capped_without_capping_observation() {
        let mut canary = StallObbtCanary::configured(true);
        canary.triggers = MAX_TRIGGER_LINES + 3;
        canary.emitted_triggers = MAX_TRIGGER_LINES;
        canary.emit_trigger(WouldTrigger {
            observation: 1,
            trigger: canary.triggers,
            progress: -1.0,
            previous_progress: -1.0,
            queue_len: 12,
            previous_queue_len: 11,
            batch_size: 1,
            patience_score: PATIENCE_LIMIT,
            depth: 4,
            pure_relu_history: true,
        });
        assert_eq!(canary.emitted_triggers, MAX_TRIGGER_LINES);

        // Saturating bookkeeping remains defined even for a theoretical
        // usize-max run. Silence Drop so this unit test does not print.
        canary.emit_lines = false;
        canary.observations = usize::MAX;
        canary.observe(Some(snapshot(-1.0)), 2, 1);
        assert_eq!(canary.observations, usize::MAX);
    }
}
