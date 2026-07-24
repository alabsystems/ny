// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! System-level property checkers for multi-network verification.
//!
//! These operate on `BoundedTensor` gain/output bounds and check
//! properties like priority routing and ducking SNR.

use ny_tensor::BoundedTensor;

/// Result of checking a system-level property.
pub struct PropertyResult {
    /// Property name.
    pub name: String,
    /// Whether the property was verified to hold for all inputs in the bounded domain.
    pub verified: bool,
    /// The bound value that was checked (e.g., SNR lower bound in dB).
    pub bound_value: f64,
    /// The threshold required (e.g., 12.0 dB).
    pub threshold: f64,
}

/// Check that lead voice gain lower bound exceeds all backing voice
/// gain upper bounds.
///
/// Returns verified=true iff min(lead.lower) > max over all backing of max(backing.upper).
pub fn check_priority_routing(
    lead_gains: &BoundedTensor,
    backing_gains: &[&BoundedTensor],
) -> PropertyResult {
    let lead_min = lead_gains
        .lower()
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);

    let backing_max = backing_gains
        .iter()
        .flat_map(|bg| bg.upper().iter().copied())
        .fold(f32::NEG_INFINITY, f32::max);

    let margin = (lead_min - backing_max) as f64;
    let verified = lead_min > backing_max;

    PropertyResult {
        name: "priority_routing".to_string(),
        verified,
        bound_value: margin,
        threshold: 0.0,
    }
}

/// Compute SNR lower bound between lead and background.
///
/// SNR_lower = 20 * log10(abs_min(lead) / abs_max(background))
///
/// where:
///   abs_min([l, u]) = 0 if l <= 0 <= u, else min(|l|, |u|)
///   abs_max([l, u]) = max(|l|, |u|)
///
/// When abs_min(lead) = 0, SNR_lower = -∞ and verified=false.
pub fn check_ducking_snr(
    lead_bounds: &BoundedTensor,
    background_bounds: &BoundedTensor,
    threshold_db: f64,
) -> PropertyResult {
    let lead_abs_min = abs_min_of_bounds(lead_bounds);
    let bg_abs_max = abs_max_of_bounds(background_bounds);

    let snr_lower = if lead_abs_min <= 0.0 || bg_abs_max <= 0.0 {
        if bg_abs_max <= 0.0 && lead_abs_min > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        20.0 * (lead_abs_min as f64 / bg_abs_max as f64).log10()
    };

    PropertyResult {
        name: "ducking_snr".to_string(),
        verified: snr_lower >= threshold_db,
        bound_value: snr_lower,
        threshold: threshold_db,
    }
}

/// Minimum absolute value achievable within the bounded range.
/// If the range contains zero, abs_min = 0.
fn abs_min_of_bounds(bounds: &BoundedTensor) -> f32 {
    bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .map(|(&l, &u)| {
            if l <= 0.0 && u >= 0.0 {
                0.0
            } else {
                l.abs().min(u.abs())
            }
        })
        .fold(f32::INFINITY, f32::min)
}

/// Maximum absolute value achievable within the bounded range.
fn abs_max_of_bounds(bounds: &BoundedTensor) -> f32 {
    bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .map(|(&l, &u)| l.abs().max(u.abs()))
        .fold(0.0_f32, f32::max)
}

/// Check spatial separation (ILD) between two voices at each ear.
///
/// For each ear, computes the guaranteed level ratio between the voices:
///   level_ear_i = pan_ear_i * abs(voice_i_power)
///   level_ear_j = pan_ear_j * abs(voice_j_power)
///
/// ILD_min at each ear = max(
///   20 * log10(level_i_min / level_j_max),  // i dominates
///   20 * log10(level_j_min / level_i_max),  // j dominates
/// )
///
/// Voices are spatially separated if `max(ILD_left, ILD_right) >= threshold_db`.
///
/// Reference: Interaural level difference for spatial hearing (Blauert, 1997).
pub fn check_spatial_ild(
    voice_i_power: &BoundedTensor,
    voice_j_power: &BoundedTensor,
    pan_i: (f32, f32),
    pan_j: (f32, f32),
    threshold_db: f64,
) -> PropertyResult {
    let abs_min_i = abs_min_of_bounds(voice_i_power);
    let abs_max_i = abs_max_of_bounds(voice_i_power);
    let abs_min_j = abs_min_of_bounds(voice_j_power);
    let abs_max_j = abs_max_of_bounds(voice_j_power);

    let ild_left = ear_ild(abs_min_i, abs_max_i, abs_min_j, abs_max_j, pan_i.0, pan_j.0);
    let ild_right = ear_ild(abs_min_i, abs_max_i, abs_min_j, abs_max_j, pan_i.1, pan_j.1);

    let max_ild = ild_left.max(ild_right);

    PropertyResult {
        name: "spatial_ild".to_string(),
        verified: max_ild >= threshold_db,
        bound_value: max_ild,
        threshold: threshold_db,
    }
}

/// Guaranteed ILD at one ear between two voices.
///
/// Returns the maximum guaranteed level ratio in dB between the voices,
/// trying both directions (i over j and j over i). A positive value
/// means one voice is guaranteed louder at this ear.
fn ear_ild(
    abs_min_i: f32,
    abs_max_i: f32,
    abs_min_j: f32,
    abs_max_j: f32,
    pan_i: f32,
    pan_j: f32,
) -> f64 {
    // Promote to f64 before multiplication to avoid f32 precision loss,
    // consistent with the interval arithmetic pattern in mixer.rs.
    let level_i_min = (pan_i as f64) * (abs_min_i as f64);
    let level_i_max = (pan_i as f64) * (abs_max_i as f64);
    let level_j_min = (pan_j as f64) * (abs_min_j as f64);
    let level_j_max = (pan_j as f64) * (abs_max_j as f64);

    let ild_i_over_j = if level_i_min > 0.0 && level_j_max > 0.0 {
        20.0 * (level_i_min / level_j_max).log10()
    } else {
        f64::NEG_INFINITY
    };

    let ild_j_over_i = if level_j_min > 0.0 && level_i_max > 0.0 {
        20.0 * (level_j_min / level_i_max).log10()
    } else {
        f64::NEG_INFINITY
    };

    ild_i_over_j.max(ild_j_over_i)
}
