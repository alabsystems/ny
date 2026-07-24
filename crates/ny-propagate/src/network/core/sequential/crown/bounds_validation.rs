// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bounds validation utilities for CROWN propagation.
//!
//! Pure predicates that check [`BoundedTensor`] health after backward
//! propagation. Used by tightening heuristics to decide whether to
//! fall back to IBP.

use ny_tensor::BoundedTensor;

/// Check if bounds contain non-finite values (NaN/Inf).
///
/// Tightening heuristic: after `concretize_sound()` (#2287), NaN and inversions
/// are repaired internally, but repaired elements become `[-inf, +inf]` which is
/// sound but maximally loose. When CROWN degrades to non-finite bounds, IBP with
/// overflow clamping typically produces tighter results. This function detects
/// that degradation so callers can fall back to IBP.
///
/// Post-#2287: this function checks only for non-finite values (NaN, Inf).
/// Inversion detection was removed since `concretize_sound()` repairs
/// inversions internally via `new_repaired(Widen)` (#3423).
pub(crate) fn has_degraded_bounds(bounds: &BoundedTensor) -> bool {
    bounds
        .lower()
        .iter()
        .chain(bounds.upper().iter())
        .any(|&v| !v.is_finite())
}

/// Check whether bounds contain NaN values (but not ±Inf).
///
/// NaN indicates a computational error where per-element intersection cannot
/// produce meaningful results. ±Inf indicates maximally-loose bounds (e.g.,
/// from #2681 non-finite row fallback) that per-element intersection handles
/// correctly: `max(-inf, ibp_lower) = ibp_lower`.
///
/// Reference: #2681 — non-finite A-matrix row fallback produces ±inf bias,
/// which concretizes to [-inf, +inf]. Per-element IBP intersection tightens
/// only the affected rows while preserving CROWN tightness for healthy rows.
pub(crate) fn has_nan_bounds(bounds: &BoundedTensor) -> bool {
    bounds
        .lower()
        .iter()
        .chain(bounds.upper().iter())
        .any(|&v| v.is_nan())
}

/// Check whether bounds contain inverted intervals (lower > upper).
///
/// Inverted intervals indicate a CROWN propagation error where the linear
/// relaxation produced impossible bounds. NaN comparisons always return false,
/// so this function only detects numeric inversions — NaN is caught separately
/// by [`has_nan_bounds`] or [`has_degraded_bounds`].
///
/// Reference: #3043 — graph_crown had this as a local `bounds_invalid_flags` fn.
///
/// Retained as a diagnostic predicate: callers no longer use it to *block*
/// CROWN-IBP intersection (inverted intervals are handled soundly per-element
/// by `tighten_crown_with_forward_bounds`'s union path), but it documents the
/// invariant and is cheap to keep available.
#[allow(dead_code)]
pub(crate) fn has_inverted_bounds(bounds: &BoundedTensor) -> bool {
    bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .any(|(&l, &u)| l > u)
}
