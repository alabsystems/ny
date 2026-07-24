// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression test for #2489: verify that 2D softmax CROWN cross-group bias
//! accumulation uses f64 intermediates with directed rounding.

use ndarray::{Array1, Array2};
use ny_core::VerificationSoundnessMode;
use ny_propagate::layers::SoftmaxLayer;
use ny_propagate::LinearBounds;
use ny_tensor::BoundedTensor;
use ny_tensor::{next_down_f32, next_up_f32};

const ROWS: usize = 16;
const COLS: usize = 3;
const NUM_OUTPUTS: usize = 2;

/// Build [ROWS, COLS] pre-activation bounds with per-group variation.
fn build_pre_activation() -> (Array2<f32>, Array2<f32>) {
    let mut lower = Array2::<f32>::zeros((ROWS, COLS));
    let mut upper = Array2::<f32>::zeros((ROWS, COLS));
    for r in 0..ROWS {
        let base = (r as f32 * 0.017).sin() * 0.35 + (r as f32) * 0.002;
        for c in 0..COLS {
            let center = base + (c as f32 - 1.0) * 0.4;
            let radius = 0.12 + ((r + c) % 7) as f32 * 0.015;
            lower[[r, c]] = center - radius;
            upper[[r, c]] = center + radius;
        }
    }
    (lower, upper)
}

/// Build alternating-sign large-magnitude coefficient matrices.
fn build_bounds(bias_lower: &[f32], bias_upper: &[f32]) -> LinearBounds {
    let total = ROWS * COLS;
    let mut lower_a = Array2::<f32>::zeros((NUM_OUTPUTS, total));
    let mut upper_a = Array2::<f32>::zeros((NUM_OUTPUTS, total));
    for out_idx in 0..NUM_OUTPUTS {
        for r in 0..ROWS {
            for c in 0..COLS {
                let flat = r * COLS + c;
                let sign = if (r + c + out_idx) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                let scale = 4_000.0 + ((r * 13 + c * 7 + out_idx * 11) % 89) as f32;
                lower_a[[out_idx, flat]] = sign * scale;
                upper_a[[out_idx, flat]] = sign * (scale + 0.5);
            }
        }
    }
    LinearBounds::new(
        lower_a,
        Array1::from_vec(bias_lower.to_vec()),
        upper_a,
        Array1::from_vec(bias_upper.to_vec()),
    )
    .unwrap()
}

/// Compute reference f64/f32 bias sums by running each row as an independent 1D group.
fn compute_reference_bias_sums(
    layer: &SoftmaxLayer,
    bounds: &LinearBounds,
    pre_lower: &Array2<f32>,
    pre_upper: &Array2<f32>,
) -> (Array1<f32>, Array1<f32>, Array1<f32>, Array1<f32>) {
    let num_groups_f = ROWS as f32;
    let split_lower_b = bounds.lower_b().mapv(|v| v / num_groups_f);
    let split_upper_b = bounds.upper_b().mapv(|v| v / num_groups_f);

    let mut lower_f64 = Array1::<f64>::zeros(NUM_OUTPUTS);
    let mut upper_f64 = Array1::<f64>::zeros(NUM_OUTPUTS);
    let mut lower_f32 = Array1::<f32>::zeros(NUM_OUTPUTS);
    let mut upper_f32 = Array1::<f32>::zeros(NUM_OUTPUTS);

    for r in 0..ROWS {
        let mut group_la = Array2::<f32>::zeros((NUM_OUTPUTS, COLS));
        let mut group_ua = Array2::<f32>::zeros((NUM_OUTPUTS, COLS));
        for out_idx in 0..NUM_OUTPUTS {
            for c in 0..COLS {
                let flat = r * COLS + c;
                group_la[[out_idx, c]] = bounds.lower_a()[[out_idx, flat]];
                group_ua[[out_idx, c]] = bounds.upper_a()[[out_idx, flat]];
            }
        }
        let group_bounds = LinearBounds::new(
            group_la,
            split_lower_b.clone(),
            group_ua,
            split_upper_b.clone(),
        )
        .unwrap();
        let group_pre = BoundedTensor::new(
            pre_lower.row(r).to_owned().into_dyn(),
            pre_upper.row(r).to_owned().into_dyn(),
        )
        .expect("row BoundedTensor");

        let gr = layer
            .propagate_linear_with_bounds(
                &group_bounds,
                &group_pre,
                VerificationSoundnessMode::Sound,
            )
            .expect("1D sound propagation");

        lower_f64 += &gr.lower_b().mapv(|v| v as f64);
        upper_f64 += &gr.upper_b().mapv(|v| v as f64);
        lower_f32 += gr.lower_b();
        upper_f32 += gr.upper_b();
    }

    let expected_lower = lower_f64.mapv(|v| next_down_f32(v as f32));
    let expected_upper = upper_f64.mapv(|v| next_up_f32(v as f32));
    (expected_lower, expected_upper, lower_f32, upper_f32)
}

#[ntest::timeout(120000)]
#[test]
fn softmax_2d_cross_group_bias_uses_f64_accumulation_2489() {
    let layer = SoftmaxLayer::new(-1);
    let (pre_lower, pre_upper) = build_pre_activation();
    let pre = BoundedTensor::new(pre_lower.clone().into_dyn(), pre_upper.clone().into_dyn())
        .expect("BoundedTensor");
    let bounds = build_bounds(&[1.25, -2.5], &[2.0, -1.0]);

    let full_result = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
        .expect("2D softmax CROWN propagation");

    let (expected_lower_b, expected_upper_b, lower_f32, upper_f32) =
        compute_reference_bias_sums(&layer, &bounds, &pre_lower, &pre_upper);

    assert_eq!(
        *full_result.lower_b(),
        expected_lower_b,
        "lower bias must match f64 reference"
    );
    assert_eq!(
        *full_result.upper_b(),
        expected_upper_b,
        "upper bias must match f64 reference"
    );

    // Verify that pure f32 accumulation would produce different bit-patterns,
    // confirming the test exercises the f64 upgrade path.
    let f32_matches = lower_f32
        .iter()
        .zip(expected_lower_b.iter())
        .all(|(old, new)| old.to_bits() == new.to_bits())
        && upper_f32
            .iter()
            .zip(expected_upper_b.iter())
            .all(|(old, new)| old.to_bits() == new.to_bits());
    assert!(
        !f32_matches,
        "f32 accumulation must differ from f64 (test sensitivity check)"
    );
}
