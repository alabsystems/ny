// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Timing estimation from static cost metadata plus a calibration profile.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::types::format_human_bytes;
use super::{CostError, CostResult};

const TIMING_PROFILE_SCHEMA_VERSION: u32 = 1;

/// Conservative timing calibration for a layer family on one backend/device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FamilyTimingCalibration {
    /// Minimum sustained arithmetic throughput in FLOPs/ns.
    pub min_effective_flops_per_ns: f64,
    /// Minimum sustained tensor traffic throughput in bytes/ns.
    pub min_effective_bytes_per_ns: f64,
    /// Per-layer launch or dispatch overhead in ns.
    pub launch_overhead_ns: u64,
}

/// Serialized hardware/backend timing profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingProfile {
    /// Schema version for forward-compatible profile loading.
    pub schema_version: u32,
    /// Human-readable calibration name.
    pub profile_name: String,
    /// Backend name, such as `wgpu` or `cpu`.
    pub backend: String,
    /// Device provenance string.
    pub device_info: String,
    /// Extra peak memory headroom for backend-specific workspace buffers.
    pub workspace_slack_bytes: u64,
    /// Calibration data keyed by `LayerCost::timing_family`.
    pub families: BTreeMap<String, FamilyTimingCalibration>,
}

/// Timing estimate for one layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerTimingEstimate {
    /// Layer name from the model.
    pub name: String,
    /// Timing family selected during static cost estimation.
    pub timing_family: String,
    /// Compute-bound portion of the latency ceiling in ns.
    pub compute_time_ns: u64,
    /// Memory-bound portion of the latency ceiling in ns.
    pub memory_time_ns: u64,
    /// Conservative total latency ceiling for this layer in ns.
    pub total_time_ns: u64,
}

/// Timing estimate for the full model under one calibration profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingEstimate {
    /// Profile name used for the estimate.
    pub profile_name: String,
    /// Backend named by the profile.
    pub backend: String,
    /// Device provenance named by the profile.
    pub device_info: String,
    /// Conservative end-to-end latency ceiling in ns.
    pub total_time_ns: u64,
    /// Peak total bytes plus backend workspace slack.
    pub peak_memory_bytes: u64,
    /// Per-layer timing breakdown.
    pub layers: Vec<LayerTimingEstimate>,
    /// Modeling assumptions callers should surface to users.
    pub assumptions: Vec<String>,
}

impl TimingEstimate {
    /// Human-readable summary for CLI output.
    pub fn summary(&self) -> String {
        let mut lines = vec![
            "Timing Estimate".to_string(),
            "===============".to_string(),
            format!("Profile: {}", self.profile_name),
            format!("Backend: {}", self.backend),
            format!("Device: {}", self.device_info),
            format!(
                "Total latency bound: {}",
                format_human_duration_ns(self.total_time_ns)
            ),
            format!(
                "Peak memory bound: {}",
                format_human_bytes(self.peak_memory_bytes)
            ),
        ];
        if !self.assumptions.is_empty() {
            lines.push("Assumptions:".to_string());
            for assumption in &self.assumptions {
                lines.push(format!("  - {assumption}"));
            }
        }
        lines.join("\n")
    }
}

/// Estimate conservative model timing from static cost plus a calibration profile.
pub fn estimate_model_timing(
    cost: &CostResult,
    profile: &TimingProfile,
) -> Result<TimingEstimate, CostError> {
    if cost.layers.is_empty() {
        return Err(CostError::no_layers("timing estimate"));
    }
    if profile.schema_version != TIMING_PROFILE_SCHEMA_VERSION {
        return Err(CostError::propagation_msg(
            "timing estimate",
            format!(
                "unsupported timing profile schema version {} (expected {})",
                profile.schema_version, TIMING_PROFILE_SCHEMA_VERSION
            ),
        ));
    }

    let mut layers = Vec::with_capacity(cost.layers.len());
    let mut total_time_ns = 0_u64;
    for layer in &cost.layers {
        let calibration = calibration_for_family(profile, &layer.timing_family)?;
        let compute_time_ns = ceil_time_ns(
            layer.flops,
            calibration.min_effective_flops_per_ns,
            "FLOPs",
            &layer.name,
            &layer.timing_family,
        )?;
        let memory_time_ns = ceil_time_ns(
            layer.total_tensor_traffic_bytes,
            calibration.min_effective_bytes_per_ns,
            "bytes",
            &layer.name,
            &layer.timing_family,
        )?;
        // Certificate-safe conservative composition: sum, not max.
        // max(compute, memory) is only sound if compute/memory fully overlap,
        // which requires a separate proof the current profile doesn't provide.
        let compute_plus_memory =
            checked_add(compute_time_ns, memory_time_ns, "layer timing overflow")?;
        let total_layer_time_ns = checked_add(
            calibration.launch_overhead_ns,
            compute_plus_memory,
            "layer timing overflow",
        )?;
        total_time_ns = checked_add(total_time_ns, total_layer_time_ns, "model timing overflow")?;
        layers.push(LayerTimingEstimate {
            name: layer.name.clone(),
            timing_family: layer.timing_family.clone(),
            compute_time_ns,
            memory_time_ns,
            total_time_ns: total_layer_time_ns,
        });
    }

    let peak_memory_bytes = checked_add(
        cost.peak_total_bytes,
        profile.workspace_slack_bytes,
        "timing peak memory overflow",
    )?;

    Ok(TimingEstimate {
        profile_name: profile.profile_name.clone(),
        backend: profile.backend.clone(),
        device_info: profile.device_info.clone(),
        total_time_ns,
        peak_memory_bytes,
        layers,
        assumptions: vec![
            "Per-layer latency bound uses launch_overhead_ns + compute_time_ns + memory_time_ns (conservative sum, no overlap assumed)."
                .to_string(),
            "Peak memory bound adds workspace_slack_bytes to the static peak_total_bytes estimate."
                .to_string(),
            "Family throughputs are treated as conservative minimum effective rates from the supplied profile."
                .to_string(),
        ],
    })
}

fn calibration_for_family<'a>(
    profile: &'a TimingProfile,
    family: &str,
) -> Result<&'a FamilyTimingCalibration, CostError> {
    let calibration = profile.families.get(family).ok_or_else(|| {
        CostError::propagation_msg(
            "timing estimate",
            format!(
                "timing profile '{}' is missing calibration for family '{}'",
                profile.profile_name, family
            ),
        )
    })?;
    validate_rate(
        calibration.min_effective_flops_per_ns,
        "min_effective_flops_per_ns",
        profile,
        family,
    )?;
    validate_rate(
        calibration.min_effective_bytes_per_ns,
        "min_effective_bytes_per_ns",
        profile,
        family,
    )?;
    Ok(calibration)
}

fn validate_rate(
    value: f64,
    field_name: &str,
    profile: &TimingProfile,
    family: &str,
) -> Result<(), CostError> {
    if value.is_finite() && value > 0.0 {
        return Ok(());
    }
    Err(CostError::propagation_msg(
        "timing estimate",
        format!(
            "timing profile '{}' has invalid {}={} for family '{}'",
            profile.profile_name, field_name, value, family
        ),
    ))
}

fn ceil_time_ns(
    work_units: u64,
    rate_per_ns: f64,
    unit_name: &str,
    layer_name: &str,
    family: &str,
) -> Result<u64, CostError> {
    if work_units == 0 {
        return Ok(0);
    }
    // Conservative f64 conversion: for work_units > 2^53, the cast to f64
    // may round down. Bias upward so we never underestimate.
    let mut numerator = work_units as f64;
    if work_units > (1_u64 << 53) && (numerator as u64) < work_units {
        numerator = next_up_f64(numerator);
    }
    let quotient = numerator / rate_per_ns;
    // Bias the quotient upward by one ULP before ceiling, ensuring the
    // division rounding direction never produces an underestimate.
    let biased = if quotient.is_finite() && quotient > 0.0 {
        next_up_f64(quotient)
    } else {
        quotient
    };
    let estimate = biased.ceil();
    if estimate.is_finite() && estimate >= 0.0 && estimate <= u64::MAX as f64 {
        return Ok(estimate as u64);
    }
    Err(CostError::propagation_msg(
        "timing estimate",
        format!(
            "timing estimate overflow for layer '{}' family '{}' ({}={}, rate={})",
            layer_name, family, unit_name, work_units, rate_per_ns
        ),
    ))
}

/// Return the next representable f64 greater than `x`.
/// Uses f64::next_up() semantics (stabilized in Rust 1.86, polyfilled here
/// for broader compatibility).
fn next_up_f64(x: f64) -> f64 {
    // next_up is stable since 1.86. Use bit manipulation for portability.
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == f64::NEG_INFINITY {
        return f64::MIN;
    }
    let bits = x.to_bits();
    let next_bits = if x >= 0.0 || x == 0.0 {
        bits + 1
    } else {
        bits - 1
    };
    f64::from_bits(next_bits)
}

fn checked_add(lhs: u64, rhs: u64, msg: &str) -> Result<u64, CostError> {
    lhs.checked_add(rhs)
        .ok_or_else(|| CostError::propagation_msg("timing estimate", msg))
}

fn format_human_duration_ns(value: u64) -> String {
    const US: f64 = 1_000.0;
    const MS: f64 = 1_000_000.0;
    const S: f64 = 1_000_000_000.0;

    if value >= S as u64 {
        format!("{:.2} s", value as f64 / S)
    } else if value >= MS as u64 {
        format!("{:.2} ms", value as f64 / MS)
    } else if value >= US as u64 {
        format!("{:.2} us", value as f64 / US)
    } else {
        format!("{value} ns")
    }
}
