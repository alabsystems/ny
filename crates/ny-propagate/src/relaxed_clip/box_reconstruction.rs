// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified centre/radius reconstruction for relaxed clipping.

use ny_core::{dd::next_up_f64, f32_to_f64_exact, f64_to_f32_up};

const F32_SIGN_MASK: u32 = 0x8000_0000;
const F32_MAGNITUDE_MASK: u32 = 0x7fff_ffff;
const F32_INFINITY_BITS: u32 = f32::INFINITY.to_bits();

/// Reconstruct an f32 centre/radius interval that contains `[x_l, x_u]`.
///
/// The centre deliberately preserves the historical binary32 expression. For
/// each endpoint, however, its distance from that centre is computed after an
/// exact bit-level binary32-to-binary64 lift, rounded upward in binary64, then
/// stored upward without emitting a binary32 subnormal. Thus the returned
/// radius is at least both exact-real endpoint distances. It is also never
/// smaller than the historical half-width.
///
/// The caller must establish [`ny_core::require_f64_interval_proof_environment`]
/// before entering this helper. All production entries in the parent module
/// do so before any proof arithmetic.
#[inline]
#[allow(clippy::manual_midpoint)]
pub(super) fn reconstruct_input_box(x_l: f32, x_u: f32) -> (f32, f32) {
    let centre = (x_l + x_u) / 2.0;
    let legacy_radius = (x_u - x_l) / 2.0;
    if !x_l.is_finite() || !x_u.is_finite() || !centre.is_finite() {
        return (0.0, f32::INFINITY);
    }

    let centre64 = f32_to_f64_exact(centre);
    let lower64 = f32_to_f64_exact(x_l);
    let upper64 = f32_to_f64_exact(x_u);

    // A difference of two binary32 values need not be exactly representable
    // in binary64 when their exponents are far apart. One binary64 bit-step
    // after round-to-nearest is therefore the required upper endpoint.
    let lower_gap = next_up_f64(centre64 - lower64);
    let upper_gap = next_up_f64(upper64 - centre64);

    let mut radius_bits = 0_u32;
    for gap in [lower_gap, upper_gap] {
        if gap > 0.0 {
            radius_bits = radius_bits.max(f64_to_f32_up(gap).to_bits());
        }
    }

    // Preserve the old half-width when it was a nonnegative number. Compare
    // representations, not f32 operands, so DAZ cannot erase a subnormal.
    let legacy_bits = legacy_radius.to_bits();
    let legacy_magnitude = legacy_bits & F32_MAGNITUDE_MASK;
    if legacy_magnitude > F32_INFINITY_BITS
        || (legacy_bits & F32_SIGN_MASK != 0 && legacy_magnitude != 0)
    {
        return (0.0, f32::INFINITY);
    }
    radius_bits = radius_bits.max(legacy_magnitude);
    (centre, f32::from_bits(radius_bits))
}
