// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN output tightening via forward-bound intersection.
//!
//! After CROWN backward propagation produces output bounds, these helpers
//! intersect them with forward (IBP) bounds to produce tighter results.
//! NaN and non-finite values are handled defensively: NaN triggers full
//! fallback to IBP, ±Inf is handled by per-element intersection (#2681).
//!
//! Reference: alpha-beta-CROWN `optimized_bounds.py:937-947`.
//! Reference: #3043 — duplication of this pattern caused #2990 and #3037.

use crate::network::tighten_crown_with_forward_bounds;
use crate::types::{BoundsProvenance, CrownIbpFallbackReason};

use ndarray::{ArrayD, IxDyn};
use ny_core::{f32_to_f64_exact, NyError, Result};
use ny_tensor::BoundedTensor;
use std::mem::size_of;
use std::time::Instant;
use tracing::{debug, warn};

use super::bounds_validation::has_nan_bounds;

const TIGHTEN_POLL_STRIDE: usize = 4_096;

#[inline]
fn check_tighten_deadline(deadline: Option<Instant>, phase: &'static str) -> Result<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        Err(NyError::DeadlineExceeded(format!(
            "CROWN output tightening: deadline exceeded {phase}"
        )))
    } else {
        Ok(())
    }
}

fn has_nan_bounds_with_deadline(bounds: &BoundedTensor, deadline: Option<Instant>) -> Result<bool> {
    for endpoints in [bounds.lower(), bounds.upper()] {
        for (index, &value) in endpoints.iter().enumerate() {
            if index.is_multiple_of(TIGHTEN_POLL_STRIDE) {
                check_tighten_deadline(deadline, "while scanning endpoints")?;
            }
            if value.is_nan() {
                return Ok(true);
            }
        }
    }
    check_tighten_deadline(deadline, "after scanning endpoints")?;
    Ok(false)
}

pub(super) fn clone_forward_bounds_with_deadline(
    forward_bounds: &BoundedTensor,
    deadline: Option<Instant>,
) -> Result<BoundedTensor> {
    const SITE: &str = "CROWN output tightening forward fallback clone";
    check_tighten_deadline(deadline, "before forward fallback clone")?;
    let elements = forward_bounds.len();
    let source_bytes = elements.saturating_mul(2).saturating_mul(size_of::<f32>());
    let required_bytes = source_bytes.saturating_mul(2);
    let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
    if required_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: SITE,
        });
    }

    let mut lower = Vec::new();
    lower
        .try_reserve_exact(elements)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: SITE,
        })?;
    let lower_capacity = lower.capacity();
    let mut upper = Vec::new();
    upper
        .try_reserve_exact(elements)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: SITE,
        })?;
    let actual_required_bytes = source_bytes.saturating_add(
        lower_capacity
            .saturating_add(upper.capacity())
            .saturating_mul(size_of::<f32>()),
    );
    if actual_required_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: actual_required_bytes,
            budget_bytes,
            site: SITE,
        });
    }
    for (index, (&lower_value, &upper_value)) in forward_bounds
        .lower()
        .iter()
        .zip(forward_bounds.upper().iter())
        .enumerate()
    {
        if index.is_multiple_of(TIGHTEN_POLL_STRIDE) {
            check_tighten_deadline(deadline, "while cloning forward fallback")?;
        }
        lower.push(lower_value);
        upper.push(upper_value);
    }
    check_tighten_deadline(deadline, "after cloning forward fallback")?;
    let lower = ArrayD::from_shape_vec(IxDyn(forward_bounds.shape()), lower)
        .map_err(|error| NyError::InternalError(format!("{SITE}: lower shape: {error}")))?;
    let upper = ArrayD::from_shape_vec(IxDyn(forward_bounds.shape()), upper)
        .map_err(|error| NyError::InternalError(format!("{SITE}: upper shape: {error}")))?;
    BoundedTensor::new_allow_infinite_with_poll(lower, upper, || {
        check_tighten_deadline(deadline, "validating forward fallback clone")
    })
}

/// Reference: #3043 — duplication of this pattern caused #2990 and #3037.
/// Reference: alpha-beta-CROWN optimized_bounds.py:937-947.
pub(crate) fn tighten_crown_output(
    crown_output: BoundedTensor,
    forward_bounds: &BoundedTensor,
    label: &str,
) -> Result<BoundedTensor> {
    // NaN indicates a computational error — per-element intersection would
    // propagate NaN, so fall back entirely to IBP.
    // ±Inf is handled correctly by per-element intersection (#2681).
    if has_nan_bounds(&crown_output) {
        debug!(
            "{}: falling back to IBP — output contains NaN bounds",
            label
        );
        return Ok(forward_bounds.clone());
    }
    // Forward-bounds NaN would contaminate the intersection result (#3043).
    // Keep the CROWN output (known NaN-free from above) rather than introducing NaN.
    if has_nan_bounds(forward_bounds) {
        debug!(
            "{}: skipping intersection — forward bounds contain NaN",
            label
        );
        return Ok(crown_output);
    }
    // #3300: Shape-tolerant intersection. Reshape CROWN output to match forward
    // bounds shape when element counts match but shapes differ.
    let crown_output = if crown_output.shape() != forward_bounds.shape()
        && crown_output.len() == forward_bounds.len()
    {
        debug!(
            "{}: reshaping CROWN {:?} to match forward {:?} for intersection",
            label,
            crown_output.shape(),
            forward_bounds.shape()
        );
        match crown_output.reshape(forward_bounds.shape()) {
            Ok(reshaped) => reshaped,
            Err(_) => crown_output,
        }
    } else {
        crown_output
    };
    if crown_output.shape() == forward_bounds.shape() {
        let (tightened, disjoint_count) =
            tighten_crown_with_forward_bounds(&crown_output, forward_bounds)?;
        if disjoint_count > 0 {
            warn!(
                "{}: forward-bound tightening has {} disjoint intervals (out of {}); used union fallback",
                label, disjoint_count, tightened.len()
            );
        }
        Ok(tightened)
    } else {
        debug!(
            "{}: skipping CROWN-IBP intersection — shape mismatch: CROWN {:?} ({} elems) vs forward {:?} ({} elems)",
            label,
            crown_output.shape(),
            crown_output.len(),
            forward_bounds.shape(),
            forward_bounds.len(),
        );
        Ok(crown_output)
    }
}

/// Finite-authority counterpart to [`tighten_crown_output`]. All endpoint
/// scans and intersection writes are cooperatively polled; shape adjustment
/// moves standard-layout buffers without copying; the only required clone
/// (full forward fallback) is fallible and charged against the CROWN budget.
pub(crate) fn tighten_crown_output_with_deadline(
    crown_output: BoundedTensor,
    forward_bounds: &BoundedTensor,
    label: &str,
    deadline: Option<Instant>,
) -> Result<BoundedTensor> {
    tighten_crown_output_with_provenance_and_deadline(crown_output, forward_bounds, label, deadline)
        .map(|(bounds, _)| bounds)
}

/// Transactional, deadline-aware post-concretization tightening with
/// provenance. No partially scanned, reshaped, or intersected tensor is
/// published on deadline or allocation refusal.
pub(crate) fn tighten_crown_output_with_provenance_and_deadline(
    mut crown_output: BoundedTensor,
    forward_bounds: &BoundedTensor,
    label: &str,
    deadline: Option<Instant>,
) -> Result<(BoundedTensor, BoundsProvenance)> {
    check_tighten_deadline(deadline, "before tightening")?;
    let crown_nan = has_nan_bounds_with_deadline(&crown_output, deadline)?;
    let fwd_nan = has_nan_bounds_with_deadline(forward_bounds, deadline)?;

    if crown_nan && !fwd_nan {
        debug!(
            "{}: falling back to forward bounds — CROWN contains NaN",
            label
        );
        drop(crown_output);
        let fallback = clone_forward_bounds_with_deadline(forward_bounds, deadline)?;
        return Ok((
            fallback,
            BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::CrownPropagationError),
        ));
    }

    if crown_output.shape() != forward_bounds.shape()
        && crown_output.len() == forward_bounds.len()
        && crown_output.lower().is_standard_layout()
        && crown_output.upper().is_standard_layout()
    {
        debug!(
            "{}: moving CROWN {:?} to forward {:?} shape for finite intersection",
            label,
            crown_output.shape(),
            forward_bounds.shape()
        );
        crown_output = crown_output.into_reshape_with_poll(forward_bounds.shape(), || {
            check_tighten_deadline(deadline, "during shape adjustment")
        })?;
    }

    if crown_output.shape() != forward_bounds.shape() {
        debug!(
            "{}: skipping forward-bound tightening — shape mismatch \
             (crown={:?}, forward={:?})",
            label,
            crown_output.shape(),
            forward_bounds.shape()
        );
        check_tighten_deadline(deadline, "before mismatched publication")?;
        return Ok((crown_output, BoundsProvenance::Crown));
    }
    if fwd_nan || crown_nan {
        debug!(
            "{}: skipping intersection — NaN bounds present \
             (fwd_nan={}, crown_nan={})",
            label, fwd_nan, crown_nan
        );
        check_tighten_deadline(deadline, "before NaN-bearing publication")?;
        return Ok((crown_output, BoundsProvenance::Crown));
    }

    let (mut lower, mut upper) = crown_output.into_parts();
    let mut disjoint_count = 0usize;
    for (index, (((crown_lower, crown_upper), &forward_lower), &forward_upper)) in lower
        .iter_mut()
        .zip(upper.iter_mut())
        .zip(forward_bounds.lower().iter())
        .zip(forward_bounds.upper().iter())
        .enumerate()
    {
        if index.is_multiple_of(TIGHTEN_POLL_STRIDE) {
            check_tighten_deadline(deadline, "while intersecting endpoints")?;
        }
        let crown_lower_exact = f32_to_f64_exact(*crown_lower);
        let crown_upper_exact = f32_to_f64_exact(*crown_upper);
        let forward_lower_exact = f32_to_f64_exact(forward_lower);
        let forward_upper_exact = f32_to_f64_exact(forward_upper);
        let (tightened_lower, tightened_lower_exact) = if crown_lower_exact >= forward_lower_exact {
            (*crown_lower, crown_lower_exact)
        } else {
            (forward_lower, forward_lower_exact)
        };
        let (tightened_upper, tightened_upper_exact) = if crown_upper_exact <= forward_upper_exact {
            (*crown_upper, crown_upper_exact)
        } else {
            (forward_upper, forward_upper_exact)
        };
        if tightened_lower_exact <= tightened_upper_exact {
            *crown_lower = tightened_lower;
            *crown_upper = tightened_upper;
        } else {
            disjoint_count += 1;
            *crown_lower = if crown_lower_exact <= forward_lower_exact {
                *crown_lower
            } else {
                forward_lower
            };
            *crown_upper = if crown_upper_exact >= forward_upper_exact {
                *crown_upper
            } else {
                forward_upper
            };
        }
    }
    check_tighten_deadline(deadline, "after intersecting endpoints")?;
    let tightened = BoundedTensor::new_allow_infinite_with_poll(lower, upper, || {
        check_tighten_deadline(deadline, "validating tightened output")
    })?;
    if disjoint_count > 0 {
        warn!(
            "{}: forward-bound tightening has {} disjoint intervals (out of {}); used union fallback",
            label,
            disjoint_count,
            tightened.len()
        );
    }
    check_tighten_deadline(deadline, "before tightened publication")?;
    Ok((tightened, BoundsProvenance::Crown))
}

/// Post-concretization tightening with provenance tracking for graph CROWN.
///
/// Like [`tighten_crown_output`] but additionally:
/// - Detects inverted bounds (lower > upper) in addition to NaN/non-finite
/// - Validates forward bounds before using them for fallback or intersection
/// - Returns [`BoundsProvenance`] indicating whether fallback was used
///
/// Used by graph CROWN backward propagation where provenance tracking is needed
/// for diagnostic reporting. Replaces ~70 lines of inline logic in propagation.rs.
///
/// Reference: #3043 — graph_crown had its own `bounds_invalid_flags` + inline logic.
/// Reference: alpha-beta-CROWN optimized_bounds.py:937-947.
#[cfg(test)]
pub(crate) fn tighten_crown_output_with_provenance(
    crown_output: BoundedTensor,
    forward_bounds: &BoundedTensor,
    label: &str,
) -> Result<(BoundedTensor, BoundsProvenance)> {
    // NaN is the only condition that forces a full fallback: per-element
    // intersection (nan_propagating_{max,min}) would propagate NaN into healthy
    // elements. ±Inf and inverted (l>u) intervals are handled *soundly and more
    // tightly* by per-element intersection — see `tighten_crown_with_forward_bounds`
    // (±Inf row: max(-inf, ibp_l)=ibp_l; inverted: disjoint → per-element union).
    // Treating ±Inf as a full-fallback trigger (the old `has_degraded_bounds`
    // behaviour) discarded CROWN tightness on every healthy output whenever a
    // single output row overflowed to ±Inf — common on deep ResNets where the
    // certified A-matrix L1 explodes for a few outputs but stays tight for the
    // rest. Per-element intersection keeps those healthy rows CROWN-tight while
    // tightening the overflowed rows to IBP. (#2681 mechanism, applied here.)
    let crown_nan = has_nan_bounds(&crown_output);
    let fwd_nan = has_nan_bounds(forward_bounds);

    // Stage 1: Full fallback only when CROWN carries NaN but forward is NaN-free.
    // Reference: propagation.rs:615-631 (pre-dedup graph_crown logic).
    if crown_nan && !fwd_nan {
        debug!(
            "{}: falling back to forward bounds — CROWN contains NaN",
            label
        );
        let provenance =
            BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::CrownPropagationError);
        // After full fallback, output IS forward_bounds — intersection is identity, skip it.
        return Ok((forward_bounds.clone(), provenance));
    }

    // Stage 2: Intersection tightening.
    // #3300: Shape-tolerant intersection. Reshape CROWN output to match forward
    // bounds shape when element counts match but shapes differ.
    let crown_output = if crown_output.shape() != forward_bounds.shape()
        && crown_output.len() == forward_bounds.len()
    {
        debug!(
            "{}: reshaping CROWN {:?} to match forward {:?} for intersection",
            label,
            crown_output.shape(),
            forward_bounds.shape()
        );
        match crown_output.reshape(forward_bounds.shape()) {
            Ok(reshaped) => reshaped,
            Err(_) => crown_output,
        }
    } else {
        crown_output
    };
    // Skip the intersection only when NaN is present — NaN would contaminate
    // healthy elements through nan_propagating_{max,min}. ±Inf and inverted
    // intervals are handled per-element by `tighten_crown_with_forward_bounds`
    // (max/min for ±Inf, per-element union for disjoint/inverted), so they are
    // tightened in place rather than abandoned wholesale.
    if crown_output.shape() == forward_bounds.shape() {
        if fwd_nan || crown_nan {
            debug!(
                "{}: skipping intersection — NaN bounds present \
                 (fwd_nan={}, crown_nan={})",
                label, fwd_nan, crown_nan
            );
        } else {
            let (tightened, disjoint_count) =
                tighten_crown_with_forward_bounds(&crown_output, forward_bounds)?;
            if disjoint_count > 0 {
                warn!(
                    "{}: forward-bound tightening has {} disjoint intervals \
                     (out of {}); used union fallback",
                    label,
                    disjoint_count,
                    tightened.len()
                );
            }
            return Ok((tightened, BoundsProvenance::Crown));
        }
    } else {
        debug!(
            "{}: skipping forward-bound tightening — shape mismatch \
             (crown={:?}, forward={:?})",
            label,
            crown_output.shape(),
            forward_bounds.shape()
        );
    }

    Ok((crown_output, BoundsProvenance::Crown))
}

#[cfg(test)]
#[path = "tighten_tests.rs"]
mod tests;
