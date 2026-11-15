// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared validation helpers for normalization layers.
//!
//! Extracted from per-layer `validated_eps()` and `MIN_EPS` constants
//! that were duplicated 4x (LayerNorm, RmsNorm, InstanceNorm, AdaIN).

use ny_core::{NyError, Result};

use super::NORMALIZATION_MIN_EPS;

/// Validate eps: must be finite and at least [`NORMALIZATION_MIN_EPS`].
///
/// `layer_name` is used in the error message (e.g., "LayerNorm", "RMSNorm").
///
/// # Scope: the DATA-VARIANCE family only
///
/// This is the gate for the normalizations whose variance is computed from
/// the input — LayerNorm, RMSNorm, InstanceNorm, GroupNorm, and AdaIN through
/// its InstanceNorm delegate. Their denominator is `sqrt(var + eps)` with
/// `var` data-dependent, so an input box containing a constant vector drives
/// `var` to exactly 0 and the whole denominator to `sqrt(eps)`: `eps == 0`
/// makes the represented function `0/0` at a reachable point, and no
/// enclosure of it exists. The floor is not merely `> 0` because below
/// [`NORMALIZATION_MIN_EPS`] these layers' derivations (the RMSNorm L2 sphere
/// argument, the CROWN slope bounds, the f32 fused paths) are not validated —
/// and eps is never silently rounded up to reach it, because changing eps
/// changes the represented function.
///
/// BatchNorm deliberately does NOT use this gate: its variance is a stored
/// constant, so `eps == 0` is sound whenever every channel's `var + eps` is
/// strictly positive, which is what `BatchNormLayer::new` checks per channel.
/// `ny-trace-bridge`'s `translate::validate_eps` /
/// `translate::validate_constant_variance_eps` mirror this same split at the
/// trace boundary, so a trace is refused where its op can be named rather
/// than deep in graph construction.
#[inline]
pub(crate) fn validate_norm_eps(eps: f32, layer_name: &str) -> Result<f32> {
    if !eps.is_finite() || eps < NORMALIZATION_MIN_EPS {
        return Err(NyError::InvalidSpec(format!(
            "{layer_name} eps invalid: {eps} (must be finite and at least {NORMALIZATION_MIN_EPS})"
        )));
    }
    Ok(eps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_epsilon_is_never_silently_clamped() {
        assert_eq!(
            validate_norm_eps(NORMALIZATION_MIN_EPS, "test").unwrap(),
            NORMALIZATION_MIN_EPS
        );
        assert!(
            validate_norm_eps(f32::from_bits(NORMALIZATION_MIN_EPS.to_bits() - 1), "test").is_err()
        );
        for invalid in [0.0, -1.0, f32::INFINITY, f32::NAN] {
            assert!(validate_norm_eps(invalid, "test").is_err());
        }
    }
}
