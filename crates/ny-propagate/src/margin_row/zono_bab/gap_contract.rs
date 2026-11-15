// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r5 CR-G — the frozen-dual tier's class funding under the acting
//! router (ROUTER_PROGRAM §4.2 CR-G, F8 / F9, §10 R5).
//!
//! The tier runs its classes in gap order (easiest first) and, TODAY, gives
//! each `left / (classes left)` of its remaining wall. When the acting router
//! arms CR-G (`router.mode: on`, `router.rules.gap_contract` on; read once per
//! tier from `ny_core::route_authority`), each class is instead funded by the
//! pure rule `ny_core::router::gap_contract::class_window`: a contract of at
//! least one state build plus two BFS levels, priced from what THIS tier
//! measured on the classes it already ran; a class whose rung the tier can no
//! longer hold declines, and so do the (harder) classes behind it — they stay
//! open in the tree, which takes back the tier's unspent wall.
//!
//! Unarmed — every run without `router.mode: on` — [`Funding::window`] returns
//! the caller's own share unchanged and [`Funding::observe`] does nothing: the
//! tier is bit-identical to a binary without this module.
//!
//! Soundness: funding only decides how long a class's own certified search
//! may run. A class closes only through its own certificate checks; a
//! declined or cut class closes nothing and stays open in the tree.

use std::time::Duration;

use ny_core::route_authority::GapTier;
use ny_core::router::gap_contract::{class_window, ClassCost, ClassWindow};

use super::ClassRun;
use crate::zono_core::ZonoRefusal;

/// Nanoseconds of a duration, saturating.
fn ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// Nanoseconds of a millisecond / second measurement, saturating (NaN and
/// negatives are zero).
fn ns_of(value: f64, per_unit: f64) -> u64 {
    let scaled = value * per_unit;
    if scaled.is_finite() && scaled > 0.0 {
        // In range by the guard below; `as` saturates in any case.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let out = scaled.min(u64::MAX as f64) as u64;
        out
    } else {
        0
    }
}

/// One tier's class funding.
#[derive(Debug)]
pub(super) struct Funding {
    armed: bool,
    cost: ClassCost,
    tier: GapTier,
    /// The window the class in flight was given, ns.
    last_window_ns: u64,
    /// Whether the tier recorded itself (armed only).
    noted: bool,
}

impl Funding {
    /// The tier's funding, armed iff the acting router armed CR-G.
    pub(super) fn from_authority(classes: usize) -> Self {
        Self::new(ny_core::route_authority::global().gap_contract(), classes)
    }

    /// The tier's funding with an explicit arming (tests).
    pub(super) fn new(armed: bool, classes: usize) -> Self {
        Self {
            armed,
            cost: ClassCost::default(),
            tier: GapTier {
                classes: classes as u64,
                ..GapTier::default()
            },
            last_window_ns: 0,
            noted: false,
        }
    }

    /// The window of the next class (`class`, with `remaining` classes left
    /// including it), given today's `share` and the tier's `left` wall.
    /// Unarmed: `Some(share)`, unchanged. Armed: CR-G's window; `None` when
    /// the class (and every class behind it) declines.
    pub(super) fn window(
        &mut self,
        class: usize,
        remaining: usize,
        share: Duration,
        left: Duration,
    ) -> Option<Duration> {
        if !self.armed {
            return Some(share);
        }
        let decided = class_window(ns(share), ns(left), &self.cost);
        let window = match decided {
            ClassWindow::Fair { .. } => {
                self.tier.fair += 1;
                Some(share)
            }
            ClassWindow::Contract { window_ns, .. } => {
                self.tier.contract += 1;
                Some(Duration::from_nanos(window_ns))
            }
            ClassWindow::Decline { .. } => {
                self.tier.declined += remaining as u64;
                None
            }
        };
        let (rung_ms, left_ms) = match decided {
            ClassWindow::Decline { rung_ns, left_ns } => (rung_ns / 1_000_000, left_ns / 1_000_000),
            _ => (
                ny_core::router::gap_contract::rung_ns(&self.cost).map_or(0, |r| r / 1_000_000),
                ns(left) / 1_000_000,
            ),
        };
        eprintln!(
            "[frozen-dual] contract class={class} window={} window_ms={} fair_ms={} rung_ms={rung_ms} \
             left_ms={left_ms} remaining={remaining} rule=CR-G",
            decided.token(),
            window.map_or(0, |w| ns(w) / 1_000_000),
            ns(share) / 1_000_000,
        );
        self.last_window_ns = window.map_or(0, ns);
        window
    }

    /// Record what the class just run cost. A no-op unarmed.
    pub(super) fn observe(&mut self, result: &Result<ClassRun, ZonoRefusal>) {
        if !self.armed {
            return;
        }
        match result {
            Ok(run) => {
                if run.progressive.as_ref().is_some_and(|p| p.deadline_cut) {
                    self.tier.cut += 1;
                    self.cost = self.cost.with_cut(self.last_window_ns);
                    return;
                }
                self.cost = self.cost.with_run(
                    ns_of(run.build_ms, 1e6),
                    ns_of(
                        run.progressive
                            .as_ref()
                            .map_or(run.search.secs, |p| p.search_secs),
                        1e9,
                    ),
                    run.search.depth as u64,
                );
            }
            Err(ZonoRefusal::Deadline) => {
                self.tier.cut += 1;
                self.cost = self.cost.with_cut(self.last_window_ns);
            }
            Err(_) => {}
        }
    }

    /// Record the tier with the route authority and print its summary
    /// (armed only; once).
    pub(super) fn finish(&mut self) {
        if !self.armed || self.noted {
            return;
        }
        self.noted = true;
        ny_core::route_authority::global().note_tier(self.tier);
        eprintln!(
            "[frozen-dual] contract done classes={} fair={} contract={} declined={} cut={} \
             rule=CR-G",
            self.tier.classes,
            self.tier.fair,
            self.tier.contract,
            self.tier.declined,
            self.tier.cut,
        );
    }

    /// What this tier funded so far.
    #[cfg(test)]
    pub(super) const fn tier(&self) -> GapTier {
        self.tier
    }
}

impl Drop for Funding {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: Duration = Duration::from_millis(1);

    #[test]
    fn gap_contract_unarmed_funding_is_todays_share_bit_for_bit() {
        let mut funding = Funding::new(false, 3);
        for share in [
            MS * 1300,
            Duration::from_nanos(1_299_999_999),
            Duration::ZERO,
        ] {
            assert_eq!(funding.window(7, 3, share, share * 3), Some(share));
        }
        funding.observe(&Err(ZonoRefusal::Deadline));
        assert_eq!(funding.tier().cut, 0, "unarmed: nothing is recorded");
        assert_eq!(
            funding.window(7, 2, MS * 5, MS * 10),
            Some(MS * 5),
            "unarmed: a cut build never prices the next class"
        );
    }

    #[test]
    fn gap_contract_armed_funding_doubles_after_a_cut_build_and_declines_what_does_not_fit() {
        // 15 classes, a 20 s tier: today each class gets ~1.3 s and every
        // build that needs more is cut (the t8853 shape).
        let mut funding = Funding::new(true, 15);
        let left = MS * 20_000;
        let fair = left / 15;
        assert_eq!(
            funding.window(0, 15, fair, left),
            Some(fair),
            "nothing measured: the first class keeps today's share"
        );
        funding.observe(&Err(ZonoRefusal::Deadline));
        let left = left.checked_sub(fair).unwrap();
        let fair = left / 14;
        let next = funding.window(1, 14, fair, left).expect("funded");
        assert_eq!(
            next,
            Duration::from_nanos(2 * ns(MS * 20_000 / 15)),
            "a build cut at w is funded 2w next"
        );
        // A rung larger than what is left declines this class and the rest.
        funding.observe(&Err(ZonoRefusal::Deadline));
        funding.observe(&Err(ZonoRefusal::Deadline));
        let tiny_left = MS * 100;
        assert_eq!(funding.window(2, 13, tiny_left / 13, tiny_left), None);
        let tier = funding.tier();
        assert_eq!(
            (tier.fair, tier.contract, tier.declined, tier.cut),
            (1, 1, 13, 3)
        );
    }
}
