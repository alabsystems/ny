// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Host-side timing profile for GPU CROWN backward orchestration (#3599).

use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct CrownHostPhaseTiming {
    /// Host-side phase label recorded during `crown_backward_gpu()`.
    pub label: String,
    /// Wall-clock time spent in this host-side phase in seconds.
    pub seconds: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CrownHostPhaseTimingSummary {
    /// Host-side phase label recorded during `crown_backward_gpu()`.
    pub label: String,
    /// Sum of all durations with this label.
    pub total_seconds: f64,
    /// Number of times this label was recorded.
    pub phase_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CrownHostTimingProfile {
    /// Raw host-side timings in call order.
    pub phases: Vec<CrownHostPhaseTiming>,
}

impl CrownHostTimingProfile {
    #[must_use]
    pub fn total_seconds(&self) -> f64 {
        self.phases.iter().map(|phase| phase.seconds).sum()
    }

    #[must_use]
    pub fn summarize_by_label(&self) -> Vec<CrownHostPhaseTimingSummary> {
        let mut by_label: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        for phase in &self.phases {
            let entry = by_label.entry(phase.label.clone()).or_insert((0.0, 0));
            entry.0 += phase.seconds;
            entry.1 += 1;
        }

        by_label
            .into_iter()
            .map(
                |(label, (total_seconds, phase_count))| CrownHostPhaseTimingSummary {
                    label,
                    total_seconds,
                    phase_count,
                },
            )
            .collect()
    }

    pub(crate) fn record(&mut self, label: &'static str, seconds: f64) {
        self.phases.push(CrownHostPhaseTiming {
            label: label.to_string(),
            seconds,
        });
    }

    pub(crate) fn extend_from(&mut self, other: Self) {
        self.phases.extend(other.phases);
    }
}

#[derive(Default)]
pub(crate) struct CrownHostTimingProfileState {
    pub(crate) enabled: bool,
    pub(crate) last_profile: Option<CrownHostTimingProfile>,
}

impl CrownHostTimingProfileState {
    pub(crate) fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if enabled {
            self.last_profile = None;
        }
    }

    pub(crate) fn take_profile(&mut self) -> Option<CrownHostTimingProfile> {
        self.last_profile.take()
    }

    pub(crate) fn store_profile(&mut self, profile: Option<CrownHostTimingProfile>) {
        self.last_profile = profile;
    }
}

#[cfg(test)]
mod tests {
    use super::{CrownHostPhaseTiming, CrownHostTimingProfile, CrownHostTimingProfileState};

    #[test]
    fn test_crown_host_timing_profile_summarizes_by_label() {
        let profile = CrownHostTimingProfile {
            phases: vec![
                CrownHostPhaseTiming {
                    label: "encode_steps".to_string(),
                    seconds: 0.75,
                },
                CrownHostPhaseTiming {
                    label: "encode_steps".to_string(),
                    seconds: 0.50,
                },
                CrownHostPhaseTiming {
                    label: "readback_poll_wait".to_string(),
                    seconds: 1.25,
                },
            ],
        };

        let summaries = profile.summarize_by_label();
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].label, "encode_steps");
        assert_eq!(summaries[0].phase_count, 2);
        assert!(
            (summaries[0].total_seconds - 1.25).abs() < 1e-9,
            "encode_steps total should be 1.25s, got {}",
            summaries[0].total_seconds
        );
        assert_eq!(summaries[1].label, "readback_poll_wait");
        assert_eq!(summaries[1].phase_count, 1);
    }

    #[test]
    fn test_crown_host_timing_profile_state_disabling_preserves_last_profile() {
        let profile = CrownHostTimingProfile {
            phases: vec![CrownHostPhaseTiming {
                label: "queue_submit".to_string(),
                seconds: 0.125,
            }],
        };
        let mut state = CrownHostTimingProfileState {
            enabled: true,
            last_profile: Some(profile.clone()),
        };

        state.set_enabled(false);

        assert!(
            !state.enabled,
            "state should be disabled after set_enabled(false)"
        );
        assert_eq!(state.take_profile(), Some(profile));
    }

    #[test]
    fn test_crown_host_timing_profile_state_enabling_clears_stale_profile() {
        let mut state = CrownHostTimingProfileState {
            enabled: false,
            last_profile: Some(CrownHostTimingProfile {
                phases: vec![CrownHostPhaseTiming {
                    label: "plan_prepare".to_string(),
                    seconds: 0.25,
                }],
            }),
        };

        state.set_enabled(true);

        assert!(
            state.enabled,
            "state should be enabled after set_enabled(true)"
        );
        assert_eq!(state.take_profile(), None);
    }
}
