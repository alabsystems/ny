// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::Result;

use crate::benchmark_support::crown_backward_cases::BenchCase;
use crate::benchmark_support::crown_backward_measurements::{
    write_measured_phase, write_measured_phase_with_detail,
};
use crate::wgpu_device::{CrownGpuTimingProfile, CrownHostTimingProfile};

pub fn write_gpu_profile_rows(
    out: &mut impl std::io::Write,
    case: &BenchCase,
    cpu_budget: usize,
    profile: &CrownGpuTimingProfile,
) -> Result<()> {
    write_measured_phase_with_detail(
        out,
        case,
        "wgpu_production_profile_total",
        profile.total_seconds(),
        cpu_budget,
        &format!("timestamp_period_ns={:.3}", profile.timestamp_period_ns),
    )?;
    for summary in profile.summarize_by_label() {
        let phase = format!("wgpu_production_profile::{}", summary.label);
        let detail = format!("passes={}", summary.pass_count);
        write_measured_phase_with_detail(
            out,
            case,
            &phase,
            summary.total_seconds,
            cpu_budget,
            &detail,
        )?;
    }
    Ok(())
}

pub fn write_host_profile_rows(
    out: &mut impl std::io::Write,
    case: &BenchCase,
    cpu_budget: usize,
    profile: &CrownHostTimingProfile,
) -> Result<()> {
    write_measured_phase(
        out,
        case,
        "wgpu_production_host_profile_total",
        profile.total_seconds(),
        cpu_budget,
    )?;
    for summary in profile.summarize_by_label() {
        let phase = format!("wgpu_production_host_profile::{}", summary.label);
        let detail = format!("phases={}", summary.phase_count);
        write_measured_phase_with_detail(
            out,
            case,
            &phase,
            summary.total_seconds,
            cpu_budget,
            &detail,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::write_host_profile_rows;
    use crate::benchmark_support::crown_backward_cases::build_bench_cases;
    use crate::wgpu_device::{CrownHostPhaseTiming, CrownHostTimingProfile};

    #[test]
    fn test_write_host_profile_rows_writes_total_and_phase_counts() {
        let cases = build_bench_cases().expect("bench cases should build");
        let case = &cases[0];
        let mut out = Vec::new();
        let profile = CrownHostTimingProfile {
            phases: vec![
                CrownHostPhaseTiming {
                    label: "queue_submit".to_string(),
                    seconds: 0.125,
                },
                CrownHostPhaseTiming {
                    label: "queue_submit".to_string(),
                    seconds: 0.250,
                },
                CrownHostPhaseTiming {
                    label: "readback_poll_wait".to_string(),
                    seconds: 0.500,
                },
            ],
        };

        write_host_profile_rows(&mut out, case, 2048, &profile)
            .expect("host profile rows should write");

        let csv = String::from_utf8(out).expect("csv output must be utf-8");
        assert!(
            csv.contains("acasxu_like,wgpu_production_host_profile_total,0.875000"),
            "csv should contain total profile row: {csv}"
        );
        assert!(
            csv.contains("acasxu_like,wgpu_production_host_profile::queue_submit,0.375000"),
            "csv should contain queue_submit profile row: {csv}"
        );
        assert!(
            csv.contains("phases=2"),
            "queue_submit should show phases=2: {csv}"
        );
        assert!(
            csv.contains("acasxu_like,wgpu_production_host_profile::readback_poll_wait,0.500000"),
            "csv should contain readback_poll_wait profile row: {csv}"
        );
        assert!(
            csv.contains("phases=1"),
            "readback_poll_wait should show phases=1: {csv}"
        );
    }
}
