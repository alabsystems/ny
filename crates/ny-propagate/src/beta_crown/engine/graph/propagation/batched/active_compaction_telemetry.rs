// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dark, print-only telemetry for future active-domain compaction in the wide
//! GPU α/β optimization loop.
//!
//! `NY_WIDE_ACTIVE_COMPACTION_TELEMETRY=1` enables the observer. Every other
//! value, including an unset variable, is OFF. The observer samples the
//! skippable-domain flags immediately before the existing full-width GPU fold
//! and emits:
//!
//! - active and skippable domain-slot counts/ratios;
//! - whether the skippable ratio exceeds a `0.2` heuristic threshold borrowed
//!   from α,β-CROWN's in-iteration pruning trigger; and
//! - the per-iteration and cumulative domain folds a threshold-gated compactor
//!   would have avoided.
//!
//! This module is intentionally unable to alter the batch. It receives only
//! copied booleans, owns no verifier state, and returns no value to the caller.
//! A disabled observer does not consume the iterator. A malformed sample
//! (wrong domain count) or counter overflow permanently silences that observer
//! instance. No measured value feeds bounds, gradients, optimizer state,
//! domain order, scheduling, or verdict logic. As with any explicitly enabled
//! print telemetry, stderr I/O itself has measurement overhead.
//! FacetBank/Hydra trajectory collection deliberately consumes later folds for
//! completed domains, so the observer stays disabled when that collector is
//! armed rather than claiming those folds are avoidable.

const ENV_GATE: &str = "NY_WIDE_ACTIVE_COMPACTION_TELEMETRY";

/// Borrows auto_LiRPA's default in-iteration pruning threshold. NY's
/// `skippable` population is deliberately not described as winner parity:
/// auto_LiRPA's numerator is verified-positive domains, while NY can finish a
/// lane for additional fail-closed or optimization-specific reasons.
const COMPACTION_SKIPPABLE_RATIO_THRESHOLD: f64 = 0.2;

/// Exactly `"1"` enables. Keeping this predicate pure makes the default-off
/// and fail-closed semantics testable without process-environment races.
fn gate_on(raw: Option<&str>) -> bool {
    raw == Some("1")
}

fn enabled_from_env() -> bool {
    gate_on(std::env::var(ENV_GATE).ok().as_deref())
}

/// A lane can leave the next full-width fold only after its existing consumer
/// would reuse a cached bound. In particular, iteration-0 `!optimizable`
/// domains are `done` but still require their first evaluation.
pub(super) fn domain_is_skippable(done: bool, has_cached_bound: bool) -> bool {
    done && has_cached_bound
}

/// Print-only observer for one invocation of the wide α/β optimizer.
pub(super) struct WideActiveCompactionTelemetry {
    enabled: bool,
    n_domains: usize,
    cumulative_predicted_avoided_domain_folds: usize,
}

impl WideActiveCompactionTelemetry {
    fn configured(n_domains: usize, facet_bank_active: bool, gate_requested: bool) -> Self {
        Self {
            // A zero-width invocation cannot produce evidence. FacetBank
            // requires every trajectory fold, including completed lanes.
            enabled: n_domains > 0 && !facet_bank_active && gate_requested,
            n_domains,
            cumulative_predicted_avoided_domain_folds: 0,
        }
    }

    pub(super) fn from_env(n_domains: usize, facet_bank_active: bool) -> Self {
        // Short-circuit before the environment lookup in unsupported modes.
        let gate_requested = n_domains > 0 && !facet_bank_active && enabled_from_env();
        Self::configured(n_domains, facet_bank_active, gate_requested)
    }

    /// Observe immutable skippability flags immediately before one full-width
    /// GPU fold. The iterator is deliberately generic so the caller can lend
    /// `dstate.iter().map(...)` without allocating a shadow mask.
    pub(super) fn observe_iteration<I>(&mut self, iteration: usize, skippable_flags: I)
    where
        I: IntoIterator<Item = bool>,
    {
        if let Some(line) = self.iteration_line(skippable_flags, iteration) {
            eprintln!("{line}");
        }
    }

    /// Pure except for this observer's private counters. `None` is the silent
    /// path: gate off, malformed sample, or checked-arithmetic failure.
    fn iteration_line<I>(&mut self, skippable_flags: I, iteration: usize) -> Option<String>
    where
        I: IntoIterator<Item = bool>,
    {
        // Check before touching the iterator: default-off runs do not even
        // scan the domain flags and perform no formatting/allocation.
        if !self.enabled {
            return None;
        }

        let mut seen = 0usize;
        let mut skippable = 0usize;
        for is_skippable in skippable_flags {
            seen = match seen.checked_add(1) {
                Some(next) if next <= self.n_domains => next,
                _ => {
                    self.enabled = false;
                    return None;
                }
            };
            if is_skippable {
                skippable = match skippable.checked_add(1) {
                    Some(next) => next,
                    None => {
                        self.enabled = false;
                        return None;
                    }
                };
            }
        }
        if seen != self.n_domains {
            self.enabled = false;
            return None;
        }

        let active = match self.n_domains.checked_sub(skippable) {
            Some(active) => active,
            None => {
                self.enabled = false;
                return None;
            }
        };
        let skippable_ratio = skippable as f64 / self.n_domains as f64;
        let active_ratio = active as f64 / self.n_domains as f64;
        // Borrow the winner's executable strict `ratio > threshold` trigger
        // without claiming that NY's broader finished population is identical.
        let compaction_eligible = skippable_ratio > COMPACTION_SKIPPABLE_RATIO_THRESHOLD;
        let predicted_avoided = if compaction_eligible { skippable } else { 0 };
        self.cumulative_predicted_avoided_domain_folds = match self
            .cumulative_predicted_avoided_domain_folds
            .checked_add(predicted_avoided)
        {
            Some(total) => total,
            None => {
                self.enabled = false;
                return None;
            }
        };

        Some(format!(
            "[wide-active-compaction] iter={iteration} active_domain_slots={active}/{} active_ratio={active_ratio:.6} skippable_domain_slots={skippable}/{} skippable_ratio={skippable_ratio:.6} heuristic_threshold={COMPACTION_SKIPPABLE_RATIO_THRESHOLD:.6} eligible={compaction_eligible} predicted_avoided_domain_folds={predicted_avoided} cumulative_predicted_avoided_domain_folds={}",
            self.n_domains,
            self.n_domains,
            self.cumulative_predicted_avoided_domain_folds,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

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
    fn only_done_domains_with_cached_bounds_are_skippable() {
        assert!(!domain_is_skippable(false, false));
        assert!(!domain_is_skippable(false, true));
        assert!(
            !domain_is_skippable(true, false),
            "iteration-0 non-optimizable lanes still need their first fold"
        );
        assert!(domain_is_skippable(true, true));
    }

    #[test]
    fn unsupported_observer_modes_decline_even_when_gate_is_requested() {
        assert!(
            !WideActiveCompactionTelemetry::configured(0, false, true).enabled,
            "a zero-domain invocation cannot produce evidence"
        );
        assert!(
            !WideActiveCompactionTelemetry::configured(4, true, true).enabled,
            "FacetBank requires completed-domain trajectory folds"
        );
        assert!(
            WideActiveCompactionTelemetry::configured(4, false, true).enabled,
            "ordinary non-Facet telemetry can be explicitly enabled"
        );
    }

    #[test]
    fn disabled_observer_is_silent_without_consuming_domain_flags() {
        let consumed = Cell::new(0usize);
        let flags = [false, true].into_iter().inspect(|_| {
            consumed.set(consumed.get() + 1);
        });
        let mut telemetry = WideActiveCompactionTelemetry {
            enabled: false,
            n_domains: 2,
            cumulative_predicted_avoided_domain_folds: 0,
        };

        assert_eq!(telemetry.iteration_line(flags, 0), None);
        assert_eq!(consumed.get(), 0, "gate-off must not scan domain state");
        assert_eq!(telemetry.cumulative_predicted_avoided_domain_folds, 0);
    }

    #[test]
    fn reports_ratios_and_threshold_qualified_avoided_folds() {
        let mut telemetry = WideActiveCompactionTelemetry {
            enabled: true,
            n_domains: 10,
            cumulative_predicted_avoided_domain_folds: 0,
        };

        let initial = telemetry
            .iteration_line([false; 10], 0)
            .expect("valid enabled sample");
        assert_eq!(
            initial,
            "[wide-active-compaction] iter=0 active_domain_slots=10/10 active_ratio=1.000000 skippable_domain_slots=0/10 skippable_ratio=0.000000 heuristic_threshold=0.200000 eligible=false predicted_avoided_domain_folds=0 cumulative_predicted_avoided_domain_folds=0"
        );

        // Borrow the strict winner trigger: exactly 20% is not yet eligible.
        let boundary = telemetry
            .iteration_line(
                [
                    true, true, false, false, false, false, false, false, false, false,
                ],
                1,
            )
            .expect("valid enabled sample");
        assert_eq!(
            boundary,
            "[wide-active-compaction] iter=1 active_domain_slots=8/10 active_ratio=0.800000 skippable_domain_slots=2/10 skippable_ratio=0.200000 heuristic_threshold=0.200000 eligible=false predicted_avoided_domain_folds=0 cumulative_predicted_avoided_domain_folds=0"
        );

        let eligible = telemetry
            .iteration_line(
                [
                    true, true, true, false, false, false, false, false, false, false,
                ],
                2,
            )
            .expect("valid enabled sample");
        assert_eq!(
            eligible,
            "[wide-active-compaction] iter=2 active_domain_slots=7/10 active_ratio=0.700000 skippable_domain_slots=3/10 skippable_ratio=0.300000 heuristic_threshold=0.200000 eligible=true predicted_avoided_domain_folds=3 cumulative_predicted_avoided_domain_folds=3"
        );

        let later = telemetry
            .iteration_line(
                [
                    true, true, true, true, true, true, true, false, false, false,
                ],
                3,
            )
            .expect("valid enabled sample");
        assert_eq!(
            later,
            "[wide-active-compaction] iter=3 active_domain_slots=3/10 active_ratio=0.300000 skippable_domain_slots=7/10 skippable_ratio=0.700000 heuristic_threshold=0.200000 eligible=true predicted_avoided_domain_folds=7 cumulative_predicted_avoided_domain_folds=10"
        );
    }

    #[test]
    fn all_domain_slots_can_be_counted_as_avoidable_after_caching() {
        let mut telemetry = WideActiveCompactionTelemetry {
            enabled: true,
            n_domains: 4,
            cumulative_predicted_avoided_domain_folds: 0,
        };
        let line = telemetry
            .iteration_line([true; 4], 7)
            .expect("valid all-skippable sample");
        assert!(line.contains("active_domain_slots=0/4"));
        assert!(line.contains("skippable_domain_slots=4/4"));
        assert!(line.contains("predicted_avoided_domain_folds=4"));
    }

    #[test]
    fn malformed_domain_count_fails_closed_and_stays_silent() {
        let mut telemetry = WideActiveCompactionTelemetry {
            enabled: true,
            n_domains: 3,
            cumulative_predicted_avoided_domain_folds: 0,
        };

        assert_eq!(telemetry.iteration_line([false, true], 0), None);
        assert!(!telemetry.enabled, "malformed sample must poison observer");
        assert_eq!(
            telemetry.iteration_line([false, false, false], 1),
            None,
            "poisoned observer must remain silent"
        );
    }

    #[test]
    fn oversized_sample_fails_closed_and_stays_silent() {
        let mut telemetry = WideActiveCompactionTelemetry {
            enabled: true,
            n_domains: 2,
            cumulative_predicted_avoided_domain_folds: 0,
        };
        assert_eq!(telemetry.iteration_line([false, true, false], 0), None);
        assert!(!telemetry.enabled);
        assert_eq!(telemetry.iteration_line([false, false], 1), None);
    }

    #[test]
    fn cumulative_counter_overflow_fails_closed() {
        let mut telemetry = WideActiveCompactionTelemetry {
            enabled: true,
            n_domains: 1,
            cumulative_predicted_avoided_domain_folds: usize::MAX,
        };
        assert_eq!(telemetry.iteration_line([true], 0), None);
        assert!(!telemetry.enabled);
    }
}
