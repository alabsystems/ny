// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Timestamp-query profiling for GPU CROWN backward compute passes (#3599).

use std::collections::BTreeMap;

use ny_core::{NyError, Result};

use super::super::WgpuDevice;

#[derive(Clone, Debug, PartialEq)]
pub struct CrownGpuPassTiming {
    /// Compute-pass label recorded in the command encoder.
    pub label: String,
    /// GPU execution time for this pass in seconds.
    pub seconds: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CrownGpuPassTimingSummary {
    /// Compute-pass label recorded in the command encoder.
    pub label: String,
    /// Sum of all pass durations with this label.
    pub total_seconds: f64,
    /// Number of passes with this label.
    pub pass_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CrownGpuTimingProfile {
    /// Nanoseconds per timestamp tick reported by wgpu.
    pub timestamp_period_ns: f32,
    /// Raw per-pass GPU timings in encoder order.
    pub passes: Vec<CrownGpuPassTiming>,
}

impl CrownGpuTimingProfile {
    #[must_use]
    pub fn total_seconds(&self) -> f64 {
        self.passes.iter().map(|pass| pass.seconds).sum()
    }

    #[must_use]
    pub fn summarize_by_label(&self) -> Vec<CrownGpuPassTimingSummary> {
        let mut by_label: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        for pass in &self.passes {
            let entry = by_label.entry(pass.label.clone()).or_insert((0.0, 0));
            entry.0 += pass.seconds;
            entry.1 += 1;
        }

        by_label
            .into_iter()
            .map(
                |(label, (total_seconds, pass_count))| CrownGpuPassTimingSummary {
                    label,
                    total_seconds,
                    pass_count,
                },
            )
            .collect()
    }

    pub(crate) fn extend_from(&mut self, other: Self) -> Result<()> {
        if self.passes.is_empty() {
            self.timestamp_period_ns = other.timestamp_period_ns;
        } else if (self.timestamp_period_ns - other.timestamp_period_ns).abs() > f32::EPSILON {
            return Err(NyError::InternalError(format!(
                "timestamp period mismatch while merging GPU CROWN profiles: {} vs {}",
                self.timestamp_period_ns, other.timestamp_period_ns
            )));
        }
        self.passes.extend(other.passes);
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct CrownTimestampProfileState {
    pub(crate) enabled: bool,
    pub(crate) last_profile: Option<CrownGpuTimingProfile>,
}

impl CrownTimestampProfileState {
    pub(crate) fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if enabled {
            self.last_profile = None;
        }
    }

    pub(crate) fn take_profile(&mut self) -> Option<CrownGpuTimingProfile> {
        self.last_profile.take()
    }

    pub(crate) fn store_profile(&mut self, profile: Option<CrownGpuTimingProfile>) {
        self.last_profile = profile;
    }
}

pub(super) struct CrownTimestampProfiler {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    labels: Vec<&'static str>,
    query_count: u32,
    next_query: u32,
}

impl CrownTimestampProfiler {
    pub(super) fn new(device: &WgpuDevice, pass_count: u32) -> Result<Option<Self>> {
        if pass_count == 0 || !device.supports_timestamp_queries() {
            return Ok(None);
        }

        let query_count = pass_count.checked_mul(2).ok_or_else(|| {
            NyError::InternalError(format!(
                "GPU CROWN timestamp profiler overflowed query count for {pass_count} passes"
            ))
        })?;
        if query_count > wgpu::QUERY_SET_MAX_QUERIES {
            return Err(NyError::InternalError(format!(
                "GPU CROWN timestamp profiler needs {query_count} queries, exceeding wgpu limit {}",
                wgpu::QUERY_SET_MAX_QUERIES
            )));
        }

        let resolve_bytes = align_up(
            u64::from(query_count) * u64::from(wgpu::QUERY_SIZE),
            wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT,
        );

        Ok(Some(Self {
            query_set: device.device().create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("crown_backward_timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: query_count,
            }),
            resolve_buffer: device.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("crown_backward_timestamp_resolve"),
                size: resolve_bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback_buffer: device.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("crown_backward_timestamp_readback"),
                size: resolve_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            labels: Vec::with_capacity(pass_count as usize),
            query_count,
            next_query: 0,
        }))
    }

    pub(super) fn allocate_pass<'a>(
        &'a mut self,
        label: &'static str,
    ) -> Result<wgpu::ComputePassTimestampWrites<'a>> {
        let begin = self.next_query;
        let end = begin + 1;
        if end >= self.query_count {
            return Err(NyError::InternalError(format!(
                "GPU CROWN timestamp profiler exhausted query slots at pass `{label}` ({end} >= {})",
                self.query_count
            )));
        }
        self.next_query += 2;
        self.labels.push(label);

        Ok(wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        })
    }

    pub(super) fn encode_resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.next_query == 0 {
            return;
        }

        let resolved_bytes = align_up(
            u64::from(self.next_query) * u64::from(wgpu::QUERY_SIZE),
            wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT,
        );
        encoder.resolve_query_set(&self.query_set, 0..self.next_query, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback_buffer,
            0,
            resolved_bytes,
        );
    }

    pub(super) fn finish(self, device: &WgpuDevice) -> Result<CrownGpuTimingProfile> {
        if self.next_query == 0 {
            return Ok(CrownGpuTimingProfile::default());
        }

        let timestamps = WgpuDevice::read_u64_buffer(
            device.device(),
            &self.readback_buffer,
            self.next_query as usize,
        )?;
        let period_ns = device.queue().get_timestamp_period();
        let mut passes = Vec::with_capacity(self.labels.len());

        for (index, label) in self.labels.iter().enumerate() {
            let begin = timestamps[2 * index];
            let end = timestamps[2 * index + 1];
            let delta = end.checked_sub(begin).ok_or_else(|| {
                NyError::InternalError(format!(
                    "GPU CROWN timestamp profiler saw decreasing timestamps for pass `{label}`: start={begin} end={end}"
                ))
            })?;
            passes.push(CrownGpuPassTiming {
                label: (*label).to_string(),
                seconds: delta as f64 * f64::from(period_ns) / 1_000_000_000.0,
            });
        }

        Ok(CrownGpuTimingProfile {
            timestamp_period_ns: period_ns,
            passes,
        })
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    if value == 0 {
        return 0;
    }
    value.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::{CrownGpuPassTiming, CrownGpuTimingProfile, CrownTimestampProfileState};

    #[test]
    fn test_crown_gpu_timing_profile_summarizes_by_label() {
        let profile = CrownGpuTimingProfile {
            timestamp_period_ns: 1.0,
            passes: vec![
                CrownGpuPassTiming {
                    label: "crown_gemm_lower".to_string(),
                    seconds: 0.75,
                },
                CrownGpuPassTiming {
                    label: "crown_gemm_lower".to_string(),
                    seconds: 0.50,
                },
                CrownGpuPassTiming {
                    label: "crown_gemm_upper".to_string(),
                    seconds: 0.25,
                },
            ],
        };

        let summaries = profile.summarize_by_label();
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].label, "crown_gemm_lower");
        assert_eq!(summaries[0].pass_count, 2);
        assert!(
            (summaries[0].total_seconds - 1.25).abs() < 1e-9,
            "crown_gemm_lower total should be 1.25s, got {}",
            summaries[0].total_seconds
        );
        assert_eq!(summaries[1].label, "crown_gemm_upper");
        assert_eq!(summaries[1].pass_count, 1);
    }

    #[test]
    fn test_crown_timestamp_profile_state_disabling_preserves_last_profile() {
        let profile = CrownGpuTimingProfile {
            timestamp_period_ns: 2.5,
            passes: vec![CrownGpuPassTiming {
                label: "crown_conc".to_string(),
                seconds: 0.125,
            }],
        };
        let mut state = CrownTimestampProfileState::default();

        state.set_enabled(true);
        state.store_profile(Some(profile.clone()));
        state.set_enabled(false);

        assert!(
            !state.enabled,
            "state should be disabled after set_enabled(false)"
        );
        assert_eq!(state.take_profile(), Some(profile));
    }

    #[test]
    fn test_crown_timestamp_profile_state_enabling_clears_stale_profile() {
        let mut state = CrownTimestampProfileState::default();
        state.store_profile(Some(CrownGpuTimingProfile {
            timestamp_period_ns: 1.0,
            passes: vec![CrownGpuPassTiming {
                label: "stale".to_string(),
                seconds: 0.25,
            }],
        }));

        state.set_enabled(true);

        assert!(
            state.enabled,
            "state should be enabled after set_enabled(true)"
        );
        assert_eq!(state.take_profile(), None);
    }
}
