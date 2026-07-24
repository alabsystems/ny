// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! BICCOS cut-gate state machine for adaptive cut generation control.

use std::collections::VecDeque;

use tracing::info;

use crate::beta_crown::config::BetaCrownConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CutGatePhase {
    ColdStart,
    CutEnabled,
    CutGenerationDisabled,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CutGateBatchStats {
    pub(crate) total_domains: usize,
    pub(crate) verified_domains: usize,
    pub(crate) bound_gain_avg: Option<f32>,
    pub(crate) cut_pruned_domains: usize,
    pub(crate) cut_total_domains: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum CutGateEvent {
    EnableCuts { verified_rate: f32, bound_gain: f32 },
    DisableCutGeneration { cut_yield: f32 },
    DisableAfterWindow,
    ColdStartExhausted,
}

#[derive(Debug, Clone)]
pub(crate) struct CutGateState {
    pub(crate) phase: CutGatePhase,
    iterations: usize,
    cut_enabled_iters: usize,
    total_verified: usize,
    verified_window: VecDeque<usize>,
    total_window: VecDeque<usize>,
    bound_gain_window: VecDeque<f32>,
    cut_yield_window: VecDeque<(usize, usize)>,
    cut_yield_bad_windows: usize,
}

impl CutGateState {
    pub(crate) fn new(config: &BetaCrownConfig) -> Self {
        let phase = if config.enable_biccos_cold_start && config.enable_cuts {
            CutGatePhase::ColdStart
        } else {
            CutGatePhase::CutEnabled
        };
        Self {
            phase,
            iterations: 0,
            cut_enabled_iters: 0,
            total_verified: 0,
            verified_window: VecDeque::new(),
            total_window: VecDeque::new(),
            bound_gain_window: VecDeque::new(),
            cut_yield_window: VecDeque::new(),
            cut_yield_bad_windows: 0,
        }
    }

    pub(crate) fn is_cold_start(&self) -> bool {
        self.phase == CutGatePhase::ColdStart
    }

    pub(crate) fn record_batch(
        &mut self,
        config: &BetaCrownConfig,
        stats: CutGateBatchStats,
        cut_total_generated: usize,
    ) -> Option<CutGateEvent> {
        if !config.enable_biccos_cold_start || !config.enable_cuts {
            return None;
        }

        self.iterations += 1;
        self.total_verified += stats.verified_domains;

        self.verified_window.push_back(stats.verified_domains);
        self.total_window.push_back(stats.total_domains);
        if self.verified_window.len() > config.biccos_verified_rate_window.max(1) {
            self.verified_window.pop_front();
        }
        if self.total_window.len() > config.biccos_verified_rate_window.max(1) {
            self.total_window.pop_front();
        }

        if let Some(avg) = stats.bound_gain_avg {
            self.bound_gain_window.push_back(avg);
            if self.bound_gain_window.len() > config.biccos_bound_gain_window.max(1) {
                self.bound_gain_window.pop_front();
            }
        }

        if stats.cut_total_domains > 0 {
            self.cut_yield_window
                .push_back((stats.cut_pruned_domains, stats.cut_total_domains));
            if self.cut_yield_window.len() > config.biccos_cut_yield_window.max(1) {
                self.cut_yield_window.pop_front();
            }
        }

        match self.phase {
            CutGatePhase::ColdStart => {
                let total_in_window: usize = self.total_window.iter().sum();
                let verified_in_window: usize = self.verified_window.iter().sum();
                let verified_rate = if total_in_window > 0 {
                    verified_in_window as f32 / total_in_window as f32
                } else {
                    0.0
                };
                let bound_gain_avg = if self.bound_gain_window.is_empty() {
                    0.0
                } else {
                    self.bound_gain_window.iter().sum::<f32>() / self.bound_gain_window.len() as f32
                };

                if self.total_verified >= config.biccos_min_verified
                    || verified_rate >= config.biccos_min_verified_rate
                    || cut_total_generated >= config.biccos_min_cuts
                    || bound_gain_avg >= config.biccos_min_bound_gain
                {
                    self.phase = CutGatePhase::CutEnabled;
                    self.cut_enabled_iters = 0;
                    return Some(CutGateEvent::EnableCuts {
                        verified_rate,
                        bound_gain: bound_gain_avg,
                    });
                }

                if self.iterations >= config.biccos_cold_max_iters && self.total_verified == 0 {
                    return Some(CutGateEvent::ColdStartExhausted);
                }
            }
            CutGatePhase::CutEnabled => {
                self.cut_enabled_iters += 1;
                if config.biccos_cut_window > 0
                    && self.cut_enabled_iters >= config.biccos_cut_window
                {
                    self.phase = CutGatePhase::CutGenerationDisabled;
                    return Some(CutGateEvent::DisableAfterWindow);
                }

                if self.cut_yield_window.len() >= config.biccos_cut_yield_window.max(1) {
                    let (pruned, total): (usize, usize) = self
                        .cut_yield_window
                        .iter()
                        .copied()
                        .fold((0, 0), |acc, item| (acc.0 + item.0, acc.1 + item.1));
                    if total > 0 {
                        let cut_yield = pruned as f32 / total as f32;
                        if cut_yield < config.biccos_min_cut_yield {
                            self.cut_yield_bad_windows += 1;
                        } else {
                            self.cut_yield_bad_windows = 0;
                        }
                        if self.cut_yield_bad_windows >= config.biccos_cut_yield_patience.max(1) {
                            self.phase = CutGatePhase::CutGenerationDisabled;
                            return Some(CutGateEvent::DisableCutGeneration { cut_yield });
                        }
                    }
                }
            }
            CutGatePhase::CutGenerationDisabled => {}
        }

        None
    }
}

/// Apply a cut-gate event: update `cut_generation_enabled` and log the transition.
///
/// This deduplicates the two identical event-dispatch match blocks in `verify_impl`.
pub(crate) fn apply_event(
    event: &CutGateEvent,
    cut_generation_enabled: &mut bool,
    domains_verified: usize,
    cuts_generated: usize,
) {
    match event {
        CutGateEvent::EnableCuts {
            verified_rate,
            bound_gain,
        } => {
            *cut_generation_enabled = true;
            info!(
                "BICCOS cold-start complete: enabling cuts (verified_total={}, verified_rate={:.3}, avg_bound_gain={:.3e}, cuts_generated={})",
                domains_verified, verified_rate, bound_gain, cuts_generated
            );
        }
        CutGateEvent::DisableCutGeneration { cut_yield } => {
            *cut_generation_enabled = false;
            info!(
                "BICCOS cut generation disabled (cut_yield={:.3}, cuts_generated={})",
                cut_yield, cuts_generated
            );
        }
        CutGateEvent::DisableAfterWindow => {
            *cut_generation_enabled = false;
            info!(
                "BICCOS cut generation window complete (cuts_generated={})",
                cuts_generated
            );
        }
        CutGateEvent::ColdStartExhausted => {
            info!("BICCOS cold-start exhausted without verified domains; keeping cuts disabled");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cold_start_config() -> BetaCrownConfig {
        BetaCrownConfig {
            enable_cuts: true,
            enable_biccos_cold_start: true,
            biccos_min_verified: 3,
            biccos_min_verified_rate: 0.1,
            biccos_verified_rate_window: 5,
            biccos_min_cuts: 2,
            biccos_min_bound_gain: 1e-3,
            biccos_bound_gain_window: 5,
            biccos_cold_max_iters: 10,
            biccos_cut_window: 8,
            biccos_min_cut_yield: 0.05,
            biccos_cut_yield_window: 3,
            biccos_cut_yield_patience: 2,
            ..Default::default()
        }
    }

    fn batch_stats(
        total: usize,
        verified: usize,
        gain: Option<f32>,
        cut_pruned: usize,
        cut_total: usize,
    ) -> CutGateBatchStats {
        CutGateBatchStats {
            total_domains: total,
            verified_domains: verified,
            bound_gain_avg: gain,
            cut_pruned_domains: cut_pruned,
            cut_total_domains: cut_total,
        }
    }

    #[test]
    fn test_cut_gate_initial_phase_cold_start_when_enabled() {
        let config = cold_start_config();
        let gate = CutGateState::new(&config);
        assert!(gate.is_cold_start());
        assert_eq!(gate.phase, CutGatePhase::ColdStart);
        assert_eq!(gate.iterations, 0);
        assert_eq!(gate.total_verified, 0);
    }

    #[test]
    fn test_cut_gate_initial_phase_cut_enabled_when_cold_start_disabled() {
        let config = BetaCrownConfig {
            enable_cuts: true,
            enable_biccos_cold_start: false,
            ..Default::default()
        };
        let gate = CutGateState::new(&config);
        assert!(!gate.is_cold_start());
        assert_eq!(gate.phase, CutGatePhase::CutEnabled);
    }

    #[test]
    fn test_cut_gate_initial_phase_cut_enabled_when_cuts_disabled() {
        let config = BetaCrownConfig {
            enable_cuts: false,
            enable_biccos_cold_start: true,
            ..Default::default()
        };
        let gate = CutGateState::new(&config);
        assert_eq!(gate.phase, CutGatePhase::CutEnabled);
    }

    #[test]
    fn test_cut_gate_record_batch_noop_when_cold_start_disabled() {
        let config = BetaCrownConfig {
            enable_cuts: true,
            enable_biccos_cold_start: false,
            ..Default::default()
        };
        let mut gate = CutGateState::new(&config);
        let event = gate.record_batch(&config, batch_stats(10, 5, None, 0, 0), 0);
        assert!(event.is_none());
    }

    #[test]
    fn test_cut_gate_record_batch_noop_when_cuts_disabled() {
        let config = BetaCrownConfig {
            enable_cuts: false,
            enable_biccos_cold_start: true,
            ..Default::default()
        };
        let mut gate = CutGateState::new(&config);
        let event = gate.record_batch(&config, batch_stats(10, 5, None, 0, 0), 0);
        assert!(event.is_none());
    }

    #[test]
    fn test_cut_gate_cold_start_to_enabled_by_verified_count() {
        let mut config = cold_start_config();
        // Set rate threshold very high so verified_rate doesn't trigger first
        config.biccos_min_verified_rate = 0.99;
        let mut gate = CutGateState::new(&config);
        assert!(gate.is_cold_start());

        // Record batches with verified domains until min_verified (3) is reached.
        // Use large total_domains (100) so the rate stays below 0.99.
        let event1 = gate.record_batch(&config, batch_stats(100, 1, None, 0, 0), 0);
        assert!(event1.is_none());
        assert!(gate.is_cold_start());

        let event2 = gate.record_batch(&config, batch_stats(100, 1, None, 0, 0), 0);
        assert!(event2.is_none());

        // Third batch pushes total_verified to 3 >= min_verified
        let event3 = gate.record_batch(&config, batch_stats(100, 1, None, 0, 0), 0);
        assert!(matches!(event3, Some(CutGateEvent::EnableCuts { .. })));
        assert!(!gate.is_cold_start());
        assert_eq!(gate.phase, CutGatePhase::CutEnabled);
    }

    #[test]
    fn test_cut_gate_cold_start_to_enabled_by_verified_rate() {
        let config = cold_start_config();
        let mut gate = CutGateState::new(&config);

        // Single batch with high verified rate (2/10 = 0.20 >= 0.10)
        let event = gate.record_batch(&config, batch_stats(10, 2, None, 0, 0), 0);
        assert!(matches!(event, Some(CutGateEvent::EnableCuts { .. })));
        assert_eq!(gate.phase, CutGatePhase::CutEnabled);
    }

    #[test]
    fn test_cut_gate_cold_start_to_enabled_by_cut_count() {
        let config = cold_start_config();
        let mut gate = CutGateState::new(&config);

        // Pass cut_total_generated >= min_cuts (2) with zero verified
        let event = gate.record_batch(&config, batch_stats(10, 0, None, 0, 0), 2);
        assert!(matches!(event, Some(CutGateEvent::EnableCuts { .. })));
    }

    #[test]
    fn test_cut_gate_cold_start_to_enabled_by_bound_gain() {
        let config = cold_start_config();
        let mut gate = CutGateState::new(&config);

        // Pass bound_gain_avg >= min_bound_gain (1e-3)
        let event = gate.record_batch(&config, batch_stats(10, 0, Some(0.01), 0, 0), 0);
        assert!(matches!(event, Some(CutGateEvent::EnableCuts { .. })));
    }

    #[test]
    fn test_cut_gate_cold_start_exhausted() {
        let config = cold_start_config();
        let mut gate = CutGateState::new(&config);

        // Record max_iters (10) batches with 0 verified
        for i in 0..9 {
            let event = gate.record_batch(&config, batch_stats(10, 0, None, 0, 0), 0);
            assert!(event.is_none(), "unexpected event at iteration {i}");
        }

        // 10th iteration with still 0 total_verified → exhausted
        let event = gate.record_batch(&config, batch_stats(10, 0, None, 0, 0), 0);
        assert!(matches!(event, Some(CutGateEvent::ColdStartExhausted)));
    }

    #[test]
    fn test_cut_gate_cut_enabled_to_disabled_by_window() {
        // Both flags must be true for record_batch to proceed past the early return
        let config = cold_start_config();
        let mut gate = CutGateState::new(&config);
        // Manually advance to CutEnabled (simulating a cold-start-to-enabled transition)
        gate.phase = CutGatePhase::CutEnabled;

        // Record batches until biccos_cut_window (8) iterations
        for i in 0..7 {
            let event = gate.record_batch(&config, batch_stats(10, 5, None, 0, 0), 0);
            assert!(event.is_none(), "unexpected event at iteration {i}");
        }

        // 8th iteration: cut_enabled_iters reaches cut_window
        let event = gate.record_batch(&config, batch_stats(10, 5, None, 0, 0), 0);
        assert!(matches!(event, Some(CutGateEvent::DisableAfterWindow)));
        assert_eq!(gate.phase, CutGatePhase::CutGenerationDisabled);
    }

    #[test]
    fn test_cut_gate_cut_enabled_to_disabled_by_low_yield() {
        let mut config = cold_start_config();
        config.biccos_cut_window = 100; // Large window so it doesn't trigger first
        config.biccos_cut_yield_window = 2;
        config.biccos_cut_yield_patience = 2;
        config.biccos_min_cut_yield = 0.1; // 10% threshold
        let mut gate = CutGateState::new(&config);
        // Manually advance to CutEnabled phase
        gate.phase = CutGatePhase::CutEnabled;

        // Record batches with low cut yield (0/10 = 0.0 < 0.1)
        // Call 1: yield window = [(0,10)], len=1 < 2 → no yield check
        let e1 = gate.record_batch(&config, batch_stats(10, 0, None, 0, 10), 0);
        assert!(e1.is_none(), "call 1: window not yet full");

        // Call 2: yield window = [(0,10),(0,10)], len=2 >= 2 → yield=0.0 < 0.1 → bad_windows=1
        let e2 = gate.record_batch(&config, batch_stats(10, 0, None, 0, 10), 0);
        assert!(e2.is_none(), "call 2: bad_windows=1 < patience=2");

        // Call 3: yield window still len=2, yield=0.0 → bad_windows=2 >= patience=2 → disable
        let e3 = gate.record_batch(&config, batch_stats(10, 0, None, 0, 10), 0);
        assert!(
            matches!(e3, Some(CutGateEvent::DisableCutGeneration { .. })),
            "expected DisableCutGeneration at call 3, got {e3:?}"
        );
        assert_eq!(gate.phase, CutGatePhase::CutGenerationDisabled);
    }

    #[test]
    fn test_cut_gate_disabled_phase_is_terminal() {
        let config = cold_start_config();
        let mut gate = CutGateState::new(&config);
        gate.phase = CutGatePhase::CutGenerationDisabled;

        // No events emitted from the disabled state
        let event = gate.record_batch(&config, batch_stats(10, 10, Some(1.0), 10, 10), 100);
        assert!(event.is_none());
    }

    #[test]
    fn test_apply_event_enable_sets_flag_true() {
        let mut enabled = false;
        apply_event(
            &CutGateEvent::EnableCuts {
                verified_rate: 0.5,
                bound_gain: 1e-3,
            },
            &mut enabled,
            10,
            5,
        );
        assert!(enabled);
    }

    #[test]
    fn test_apply_event_disable_generation_clears_flag() {
        let mut enabled = true;
        apply_event(
            &CutGateEvent::DisableCutGeneration { cut_yield: 0.02 },
            &mut enabled,
            10,
            5,
        );
        assert!(!enabled);
    }

    #[test]
    fn test_apply_event_disable_after_window_clears_flag() {
        let mut enabled = true;
        apply_event(&CutGateEvent::DisableAfterWindow, &mut enabled, 10, 5);
        assert!(!enabled);
    }

    #[test]
    fn test_apply_event_cold_start_exhausted_preserves_flag() {
        let mut enabled = false;
        apply_event(&CutGateEvent::ColdStartExhausted, &mut enabled, 0, 0);
        assert!(!enabled, "ColdStartExhausted should not change the flag");
    }
}
