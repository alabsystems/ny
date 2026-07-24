// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Statistical utilities for bound profiling.

use ndarray::{ArrayD, IxDyn};
use ny_core::NyError;
use ny_tensor::BoundedTensor;

/// Create deterministic input with unit variance for realistic LayerNorm/RMSNorm bounds.
///
/// Uses alternating ±1 pattern to ensure non-zero variance, avoiding artificial
/// amplification when the center point has variance near zero.
///
/// For RMSNorm/LayerNorm with zero-valued inputs:
/// - var(zeros) = 0, std = sqrt(eps) ≈ 0.003 → 300x amplification
///
/// For alternating ±1 inputs:
/// - mean ≈ 0, var = 1, std ≈ 1 → 1x amplification (realistic)
pub(super) fn make_unit_variance_input(
    shape: &[usize],
    epsilon: f32,
) -> Result<BoundedTensor, NyError> {
    let mut array = ArrayD::zeros(IxDyn(shape));
    for (i, value) in array.iter_mut().enumerate() {
        *value = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    BoundedTensor::from_epsilon(array, epsilon)
}

/// Calculate median of a slice.
pub(super) fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = values.iter().filter(|v| v.is_finite()).cloned().collect();
    if sorted.is_empty() {
        return f32::INFINITY;
    }
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        f32::midpoint(sorted[mid - 1], sorted[mid])
    } else {
        sorted[mid]
    }
}

/// Calculate verification difficulty score (0-100).
pub(super) fn difficulty_score(total_expansion: f32, max_growth: f32, overflow: bool) -> f32 {
    if overflow {
        return 100.0;
    }

    // Log-scale scoring
    let expansion_score = if total_expansion <= 1.0 {
        0.0
    } else {
        (total_expansion.log10() * 10.0).min(50.0)
    };

    let growth_score = if max_growth <= 1.0 {
        0.0
    } else {
        (max_growth.log10() * 20.0).min(50.0)
    };

    (expansion_score + growth_score).min(100.0)
}
