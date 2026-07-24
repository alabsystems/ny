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

use ny_core::Result;
use ny_tensor::BoundedTensor;
use tracing::{debug, warn};

use super::bounds_validation::has_nan_bounds;

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
