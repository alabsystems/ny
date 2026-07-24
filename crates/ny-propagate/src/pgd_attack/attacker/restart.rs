// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Restart-when-stuck helpers for PGD attack (#4278).
//!
//! Detects projected no-op updates and resamples fresh points from the input
//! box when `PgdConfig::restart_when_stuck` is enabled.
//!
//! Reference: alpha-beta-CROWN `general_spec_attack.py:409-427`.

use ndarray::ArrayD;
use ny_tensor::BoundedTensor;
use rand::rngs::StdRng;

use super::PgdAttacker;

/// Check whether a projected step left the point unchanged.
///
/// Uses exact L1 equality: `sum(|projected - previous|) == 0.0`.
/// This matches the reference `(delta - old_delta).abs().sum() == 0` semantics.
pub(crate) fn projected_step_is_stuck(previous: &ArrayD<f32>, projected: &ArrayD<f32>) -> bool {
    previous.iter().zip(projected.iter()).all(|(&a, &b)| a == b)
}

/// Resample a fresh point uniformly from the input bounds.
pub(crate) fn resample_uniform_point(
    attacker: &PgdAttacker<'_>,
    input_bounds: &BoundedTensor,
    rng: &mut StdRng,
) -> ArrayD<f32> {
    attacker.sample_uniform(input_bounds, rng)
}
