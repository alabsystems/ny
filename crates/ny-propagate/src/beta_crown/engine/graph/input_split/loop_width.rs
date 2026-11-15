// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r6 `rule.trial_width` at the input-split site: the batch the
//! single-objective loop POPS per iteration, and the DORMANCY rule.
//!
//! `bab.batch_size` is DORMANT on the input-split route unless
//! `reorder_bab` is on: without it the loop discards the configured value
//! and pops ONE domain per iteration (CLAUDE.md, `preset_arrival_ratchet`'s
//! dormancy row). A width trial on that route would move a number no loop
//! reads — a placebo whose null is uninterpretable. So [`LoopWidth`] routes
//! around the dormancy explicitly:
//!
//! * `reorder_bab` off: the live width is the hardcoded 1, no machine is
//!   built, and an armed site records one `blocked` event
//!   (`why=dormant:reorder_bab_off`) so a measurement sees the refusal;
//! * `reorder_bab` on: the live width is the clamped
//!   `input_split_loop_batch_size` decision, and an armed site's AIMD machine
//!   moves THAT value inside `[eff / 4, eff x 4]`, every candidate re-clamped
//!   by the same memory / config cap (`MAX_INPUT_SPLIT_CHILD_WORKSET_BYTES`),
//!   so the value the machine reports is exactly the value the loop pops.
//!
//! Lease use is the pick's child workset against that same static cap.

use std::time::{Duration, Instant};

use ny_core::router::trial::{AimdTrial, WidthObs};
use ny_core::trial_authority::{TrialAuthority, TrialSite};
use ny_core::Result;

use super::loop_batch_size::{input_split_loop_batch_size, MAX_INPUT_SPLIT_CHILD_WORKSET_BYTES};

/// Why the site is dormant (the `blocked` event's reason).
pub(crate) const DORMANT_REASON: &str = "dormant:reorder_bab_off";

/// The live batch width of one input-split loop run.
#[derive(Debug)]
pub(crate) struct LoopWidth {
    /// The authority the site consults (the process's; a test's own).
    authority: &'static TrialAuthority,
    /// Today's width: the clamped decision under `reorder_bab`, else 1.
    base: usize,
    /// Whether the configured batch is dormant on this route.
    dormant: bool,
    /// Bytes one picked parent's child workset holds (the lease unit).
    bytes_per_parent: f64,
    /// The AIMD machine (`None` unless armed and live).
    trial: Option<AimdTrial>,
    /// The batch-iteration count (the trial's work unit).
    units: u64,
}

impl LoopWidth {
    /// Resolve the loop's width from today's configuration. Errors only where
    /// today's `input_split_loop_batch_size` errors.
    pub(crate) fn resolve(
        reorder_bab: bool,
        batch_size: usize,
        input_elems: usize,
    ) -> Result<Self> {
        Self::resolve_with(
            crate::trial_sites::global(),
            reorder_bab,
            batch_size,
            input_elems,
        )
    }

    /// [`Self::resolve`] against an explicit authority (tests arm their own).
    pub(crate) fn resolve_with(
        authority: &'static TrialAuthority,
        reorder_bab: bool,
        batch_size: usize,
        input_elems: usize,
    ) -> Result<Self> {
        let bytes_per_parent = (input_elems.max(1) as f64) * 2.0 * 4.0 * 2.0 * 2.0;
        if !reorder_bab {
            crate::trial_sites::width_dormant(
                authority,
                TrialSite::InputSplitWidth,
                1,
                DORMANT_REASON,
            );
            return Ok(Self {
                authority,
                base: 1,
                dormant: true,
                bytes_per_parent,
                trial: None,
                units: 0,
            });
        }
        let base = input_split_loop_batch_size(batch_size, input_elems)?.effective_batch_size;
        // The site's hard ceiling is the SAME clamp the loop applies to any
        // requested width (memory and config caps), so no trialled value can
        // exceed what the loop would pop.
        let hard_max = input_split_loop_batch_size(usize::MAX, input_elems)?.effective_batch_size;
        let trial = crate::trial_sites::width_trial(
            authority,
            TrialSite::InputSplitWidth,
            base as u64,
            1,
            hard_max as u64,
        );
        Ok(Self {
            authority,
            base,
            dormant: false,
            bytes_per_parent,
            trial,
            units: 0,
        })
    }

    /// The width the loop pops on its NEXT iteration: the machine's (always
    /// inside the clamp), else today's.
    pub(crate) fn width(&self) -> usize {
        if self.dormant {
            return 1;
        }
        crate::trial_sites::width_value(self.trial.as_ref(), self.base)
    }

    /// Whether the configured batch is dormant on this route.
    #[cfg(test)]
    pub(crate) const fn dormant(&self) -> bool {
        self.dormant
    }

    /// Whether a machine is live.
    #[cfg(test)]
    pub(crate) const fn trialling(&self) -> bool {
        self.trial.is_some()
    }

    /// One batch iteration finished (the loop's safe point): feed the
    /// machine the run's certified closes (`domains_verified`) and the
    /// loop's own clock.
    pub(crate) fn after_batch(&mut self, closes: u64, start: Instant, budget: Duration) {
        if self.trial.is_none() {
            self.units = self.units.saturating_add(1);
            return;
        }
        let elapsed = start.elapsed();
        let remaining = budget.saturating_sub(elapsed).as_secs_f64();
        self.observe_at(closes, elapsed.as_secs_f64(), Some(remaining));
    }

    /// [`Self::after_batch`] on an explicit lane clock (the loop's own
    /// seconds since its start, and the seconds left).
    pub(crate) fn observe_at(&mut self, closes: u64, t_s: f64, remaining_s: Option<f64>) {
        self.units = self.units.saturating_add(1);
        if self.trial.is_none() {
            return;
        }
        let lease_use = (self.width() as f64 * self.bytes_per_parent)
            / MAX_INPUT_SPLIT_CHILD_WORKSET_BYTES as f64;
        crate::trial_sites::observe_width(
            self.authority,
            &mut self.trial,
            WidthObs {
                units: self.units,
                closes,
                t_s,
                remaining_s,
                lease_use: Some(lease_use),
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unarmed (every unit-test process: nothing arms the global trial
    /// authority here), the loop pops exactly today's width on both routes:
    /// 1 without `reorder_bab` (the dormancy), the clamped decision with it.
    #[test]
    fn router_r6_input_split_width_is_todays_value_when_unarmed() {
        let dormant = LoopWidth::resolve(false, 4096, 6).expect("resolve");
        assert!(dormant.dormant());
        assert!(!dormant.trialling());
        assert_eq!(
            dormant.width(),
            1,
            "bab.batch_size is dormant without reorder_bab"
        );
        let live = LoopWidth::resolve(true, 4096, 6).expect("resolve");
        assert!(!live.dormant());
        assert!(!live.trialling());
        assert_eq!(
            live.width(),
            input_split_loop_batch_size(4096, 6)
                .expect("clamp")
                .effective_batch_size
        );
        // A large input: the memory cap binds and the live width is the cap.
        let capped = LoopWidth::resolve(true, 1 << 30, 784).expect("resolve");
        assert_eq!(
            capped.width(),
            input_split_loop_batch_size(1 << 30, 784)
                .expect("clamp")
                .effective_batch_size
        );
    }

    /// THE DORMANCY TRAP (CLAUDE.md): `bab.batch_size` is dormant on the
    /// input-split route unless `reorder_bab`. With the site ARMED (a local
    /// authority, never the global): without `reorder_bab` no machine is
    /// built, the loop keeps popping 1, and one `blocked` event names the
    /// dormancy; with `reorder_bab` a machine is built AROUND THE LIVE VALUE
    /// (the clamped decision the loop pops), and after a trial moves it the
    /// width the loop pops is exactly the machine's value, inside the clamp.
    #[test]
    fn router_r6_input_split_width_trials_only_the_live_batch() {
        use ny_core::router::trial::{TrialDecision, WidthRule};
        use ny_core::trial_authority::TrialDirectives;
        static AUTH: TrialAuthority = TrialAuthority::new();
        AUTH.arm(TrialDirectives {
            width: Some(WidthRule::SHIPPED),
            sites: TrialSite::InputSplitWidth.bit(),
            ..TrialDirectives::NONE
        })
        .expect("arm the site");

        let dormant = LoopWidth::resolve_with(&AUTH, false, 4096, 6).expect("resolve");
        assert!(dormant.dormant() && !dormant.trialling());
        assert_eq!(dormant.width(), 1);
        let snap = AUTH.snapshot();
        assert_eq!(snap.counts.blocked, 1, "{snap:?}");
        assert_eq!(snap.events[0].why, DORMANT_REASON);
        assert_eq!(snap.counts.started, 0, "a dormant batch is never trialled");

        let base = input_split_loop_batch_size(64, 6)
            .expect("clamp")
            .effective_batch_size;
        let mut live = LoopWidth::resolve_with(&AUTH, true, 64, 6).expect("resolve");
        assert!(live.trialling(), "the live batch IS trialled");
        assert_eq!(live.width(), base, "it starts at today's live value");
        // A synthetic stream on the machine's own clock (0.1 s per batch):
        // a window of 16 batches at 100 closes/s, then one at 120 closes/s (a
        // >= 5% rise, lease far under 70%, a far deadline): the machine
        // raises x2, and the loop's NEXT pop is that value.
        let mut closes = 0u64;
        let mut t = 0.0f64;
        for per_batch in [10u64; 17].into_iter().chain([12u64; 16]) {
            closes += per_batch;
            t += 0.1;
            live.observe_at(closes, t, Some(10_000.0));
        }
        let snap = AUTH.snapshot();
        assert!(
            snap.events
                .iter()
                .any(|e| e.decision == TrialDecision::Started && e.dir.token() == "up"),
            "{snap:?}"
        );
        assert_eq!(live.width(), base * 2, "the loop pops the machine's value");
        let hard_max = input_split_loop_batch_size(usize::MAX, 6)
            .expect("clamp")
            .effective_batch_size;
        assert!(live.width() <= hard_max, "never past the loop's own clamp");
        let start = Instant::now();
        let budget = Duration::from_secs(10_000);
        AUTH.disarm();
        live.after_batch(closes + 1, start, budget);
        assert_eq!(live.width(), base, "disarmed: today's value again");
    }
}
