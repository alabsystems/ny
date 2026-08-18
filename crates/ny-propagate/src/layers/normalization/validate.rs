// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared validation helpers for normalization layers.
//!
//! Extracted from per-layer `validated_eps()` and `MIN_EPS` constants
//! that were duplicated 4x (LayerNorm, RmsNorm, InstanceNorm, AdaIN).
//!
//! Reference: designs/2026-02-27-normalization-trait-dedup.md

use ny_core::{NyError, Result};

use super::NORMALIZATION_MIN_EPS;

/// Validate eps: must be finite and at least [`NORMALIZATION_MIN_EPS`].
///
/// `layer_name` is used in the error message (e.g., "LayerNorm", "RMSNorm").
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
