// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::LayerComparison;
use ndarray::ArrayD;

/// Compare two arrays element-wise and compute diff statistics.
pub(crate) fn compare_arrays(a: &ArrayD<f32>, b: &ArrayD<f32>, tolerance: f32) -> LayerComparison {
    let shape_a: Vec<usize> = a.shape().to_vec();
    let shape_b: Vec<usize> = b.shape().to_vec();

    if shape_a != shape_b {
        return LayerComparison {
            name: String::new(),
            name_b: None,
            max_diff: f32::INFINITY,
            mean_diff: f32::INFINITY,
            exceeds_tolerance: true,
            shape_a,
            shape_b,
        };
    }

    let diffs: Vec<f32> = a
        .iter()
        .zip(b.iter())
        .map(|(va, vb)| {
            let diff = (va - vb).abs();
            if diff.is_nan() && (va == vb || (va.is_nan() && vb.is_nan())) {
                // Matching special values (Inf-Inf, -Inf-(-Inf), NaN-NaN) produce
                // NaN from IEEE 754 arithmetic, but semantically both models agree
                // at this position. Treat as zero difference to avoid NaN poisoning
                // mean_diff and corrupting downstream diagnosis.
                0.0
            } else {
                diff
            }
        })
        .collect();

    let max_diff = diffs.iter().cloned().fold(0.0f32, f32::max);
    let mean_diff = diffs.iter().sum::<f32>() / diffs.len() as f32;

    LayerComparison {
        name: String::new(),
        name_b: None,
        max_diff,
        mean_diff,
        exceeds_tolerance: max_diff > tolerance,
        shape_a,
        shape_b,
    }
}
