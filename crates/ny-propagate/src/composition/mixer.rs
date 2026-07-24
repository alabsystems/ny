// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Linear audio mixer composition using interval arithmetic.
//!
//! Composes N voice certificates through a linear mixer, producing
//! per-ear output bounds using 4-corner interval multiplication.
//!
//! Reference: Moore, R.E. (1966). *Interval Analysis*. Prentice-Hall.

use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};
use ny_core::Result;
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::certificate::BoundCertificate;

/// Per-voice gain and spatial panning for a linear audio mixer.
pub struct MixerSpec {
    /// voice_id → gain bounds as BoundedTensor (scalar or per-sample).
    pub voice_gains: HashMap<String, BoundedTensor>,
    /// voice_id → (left_pan, right_pan) as constant coefficients.
    pub spatial_pan: HashMap<String, (f32, f32)>,
}

/// Compose N voice certificates through a linear mixer using
/// interval arithmetic.
///
/// For each voice i contributing to each ear:
///   contribution = gain_i * pan_coeff * voice_output_i
///
/// Uses 4-corner interval multiplication for each gain × voice product:
///   products = {g_l * v_l, g_l * v_u, g_u * v_l, g_u * v_u}
///   result = [min(products), max(products)]
///
/// System output per ear = sum of per-voice contributions (interval addition).
pub fn compose_linear_mix(
    certificates: &[BoundCertificate],
    spec: &MixerSpec,
) -> Result<(BoundedTensor, BoundedTensor)> {
    if certificates.is_empty() {
        return Err(ny_core::NyError::InvalidConfig(
            "compose_linear_mix requires at least one certificate".to_string(),
        ));
    }

    // Determine output dimension from first certificate.
    let output_shape = certificates[0].output_bounds().shape().to_vec();
    let n_dims = certificates[0].output_bounds().lower().len();

    let mut left_lower = vec![0.0_f64; n_dims];
    let mut left_upper = vec![0.0_f64; n_dims];
    let mut right_lower = vec![0.0_f64; n_dims];
    let mut right_upper = vec![0.0_f64; n_dims];

    for cert in certificates {
        if cert.output_bounds().shape() != &output_shape[..] {
            return Err(ny_core::NyError::InvalidConfig(format!(
                "voice '{}' output shape {:?} does not match mixer shape {:?}",
                cert.model_id(),
                cert.output_bounds().shape(),
                output_shape
            )));
        }
        let gain = spec.voice_gains.get(cert.model_id()).ok_or_else(|| {
            ny_core::NyError::InvalidConfig(format!(
                "mixer spec missing gain for voice '{}'",
                cert.model_id()
            ))
        })?;
        let &(left_pan, right_pan) = spec.spatial_pan.get(cert.model_id()).ok_or_else(|| {
            ny_core::NyError::InvalidConfig(format!(
                "mixer spec missing spatial pan for voice '{}'",
                cert.model_id()
            ))
        })?;
        if gain.shape() != &output_shape[..] && gain.lower().len() != 1 {
            return Err(ny_core::NyError::InvalidConfig(format!(
                "voice '{}' gain shape {:?} must be scalar or match output shape {:?}",
                cert.model_id(),
                gain.shape(),
                output_shape
            )));
        }
        if !left_pan.is_finite() || !right_pan.is_finite() {
            return Err(ny_core::NyError::InvalidConfig(format!(
                "mixer spec pan coefficients must be finite for voice '{}'",
                cert.model_id()
            )));
        }

        let voice_bounds = cert
            .output_bounds()
            .lower()
            .iter()
            .copied()
            .zip(cert.output_bounds().upper().iter().copied());

        if gain.lower().len() == 1 {
            let g_l = gain
                .lower()
                .iter()
                .next()
                .copied()
                .expect("scalar gain lower bound missing");
            let g_u = gain
                .upper()
                .iter()
                .next()
                .copied()
                .expect("scalar gain upper bound missing");
            for (j, (v_l, v_u)) in voice_bounds.enumerate() {
                accumulate_voice_contribution(
                    (&mut left_lower, &mut left_upper),
                    (&mut right_lower, &mut right_upper),
                    j,
                    (v_l, v_u),
                    (g_l, g_u),
                    (left_pan, right_pan),
                );
            }
        } else {
            for (j, ((v_l, v_u), (g_l, g_u))) in voice_bounds
                .zip(
                    gain.lower()
                        .iter()
                        .copied()
                        .zip(gain.upper().iter().copied()),
                )
                .enumerate()
            {
                accumulate_voice_contribution(
                    (&mut left_lower, &mut left_upper),
                    (&mut right_lower, &mut right_upper),
                    j,
                    (v_l, v_u),
                    (g_l, g_u),
                    (left_pan, right_pan),
                );
            }
        }
    }

    let left = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&output_shape),
            left_lower
                .into_iter()
                .map(|value| next_down_f32(value as f32))
                .collect(),
        )
        .map_err(|e| ny_core::NyError::InvalidConfig(e.to_string()))?,
        ArrayD::from_shape_vec(
            IxDyn(&output_shape),
            left_upper
                .into_iter()
                .map(|value| next_up_f32(value as f32))
                .collect(),
        )
        .map_err(|e| ny_core::NyError::InvalidConfig(e.to_string()))?,
    )?;
    let right = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&output_shape),
            right_lower
                .into_iter()
                .map(|value| next_down_f32(value as f32))
                .collect(),
        )
        .map_err(|e| ny_core::NyError::InvalidConfig(e.to_string()))?,
        ArrayD::from_shape_vec(
            IxDyn(&output_shape),
            right_upper
                .into_iter()
                .map(|value| next_up_f32(value as f32))
                .collect(),
        )
        .map_err(|e| ny_core::NyError::InvalidConfig(e.to_string()))?,
    )?;

    Ok((left, right))
}

fn scale_interval_by_constant(lower: f64, upper: f64, scalar: f32) -> (f64, f64) {
    debug_assert!(lower <= upper, "interval endpoints must be ordered");

    let scalar = scalar as f64;
    if scalar >= 0.0 {
        (scalar * lower, scalar * upper)
    } else {
        (scalar * upper, scalar * lower)
    }
}

fn accumulate_voice_contribution(
    left: (&mut [f64], &mut [f64]),
    right: (&mut [f64], &mut [f64]),
    index: usize,
    voice_bounds: (f32, f32),
    gain_bounds: (f32, f32),
    pan: (f32, f32),
) {
    let (left_lower, left_upper) = left;
    let (right_lower, right_upper) = right;
    let (voice_lower, voice_upper) = voice_bounds;
    let (gain_lower, gain_upper) = gain_bounds;
    let (left_pan, right_pan) = pan;

    // 4-corner interval multiplication for gain × voice.
    let products = [
        (gain_lower as f64) * (voice_lower as f64),
        (gain_lower as f64) * (voice_upper as f64),
        (gain_upper as f64) * (voice_lower as f64),
        (gain_upper as f64) * (voice_upper as f64),
    ];
    let prod_min = products.iter().copied().fold(f64::INFINITY, f64::min);
    let prod_max = products.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Signed constant scaling must preserve interval order.
    let (left_min, left_max) = scale_interval_by_constant(prod_min, prod_max, left_pan);
    let (right_min, right_max) = scale_interval_by_constant(prod_min, prod_max, right_pan);

    left_lower[index] += left_min;
    left_upper[index] += left_max;
    right_lower[index] += right_min;
    right_upper[index] += right_max;
}
