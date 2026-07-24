// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::Array1;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

/// Per-path tangent controls for one monotone S-shaped alpha parameter group.
///
/// Reference: alpha-beta-CROWN `auto_LiRPA/operators/s_shaped.py`
/// `_init_opt_parameters_impl()` and `bound_relax_impl()` for the 8 active
/// tangent-point slots used by Sigmoid/Tanh.
#[derive(Debug, Clone)]
pub(crate) struct MonotoneSShapedDualParams {
    pub(crate) lower_path: Array1<f32>,
    pub(crate) upper_path: Array1<f32>,
    pub(crate) velocity_lower: Array1<f32>,
    pub(crate) velocity_upper: Array1<f32>,
    pub(crate) adam_m_lower: Array1<f32>,
    pub(crate) adam_v_lower: Array1<f32>,
    pub(crate) adam_m_upper: Array1<f32>,
    pub(crate) adam_v_upper: Array1<f32>,
}

impl MonotoneSShapedDualParams {
    fn new(initial: &Array1<f32>) -> Self {
        let zeros = Array1::zeros(initial.len());
        Self {
            lower_path: initial.clone(),
            upper_path: initial.clone(),
            velocity_lower: zeros.clone(),
            velocity_upper: zeros.clone(),
            adam_m_lower: zeros.clone(),
            adam_v_lower: zeros.clone(),
            adam_m_upper: zeros.clone(),
            adam_v_upper: zeros,
        }
    }
}

/// One-path tangent-point bundle consumed by the Sigmoid/Tanh relaxation builder.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MonotoneSShapedPathAlpha {
    pub(crate) tp_pos: f32,
    pub(crate) tp_neg: f32,
    pub(crate) tp_both_lower: f32,
    pub(crate) tp_both_upper: f32,
    pub(crate) d_lower: f32,
    pub(crate) d_upper: f32,
}

/// Optimizable tangent-point state for monotone S-shaped activations.
///
/// The four parameter groups follow alpha-beta-CROWN's monotone Sigmoid/Tanh
/// port:
/// - `tp_pos`: positive intervals, upper relaxation tangent
/// - `tp_neg`: negative intervals, lower relaxation tangent
/// - `tp_both_lower`: crossing-zero intervals, lower relaxation tangent
/// - `tp_both_upper`: crossing-zero intervals, upper relaxation tangent
#[derive(Debug, Clone)]
pub(crate) struct MonotoneSShapedAlpha {
    pub(crate) tp_pos: MonotoneSShapedDualParams,
    pub(crate) tp_neg: MonotoneSShapedDualParams,
    pub(crate) tp_both_lower: MonotoneSShapedDualParams,
    pub(crate) tp_both_upper: MonotoneSShapedDualParams,
    pub(crate) mask_neg: Array1<bool>,
    pub(crate) mask_pos: Array1<bool>,
    pub(crate) mask_cross: Array1<bool>,
    pub(super) lower_bounds: Array1<f32>,
    pub(super) upper_bounds: Array1<f32>,
    pub(super) midpoint: Array1<f32>,
    pub(super) d_lower: Array1<f32>,
    pub(super) d_upper: Array1<f32>,
}

impl MonotoneSShapedAlpha {
    pub(crate) fn from_bounds(
        pre_activation: &BoundedTensor,
        crossing_tangents: fn(f32, f32) -> (f32, f32),
    ) -> Result<Self> {
        let flat = pre_activation.flatten();
        let lower = flat
            .lower()
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![flat.len()],
                got: flat.lower().shape().to_vec(),
            })?;
        let upper = flat
            .upper()
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![flat.len()],
                got: flat.upper().shape().to_vec(),
            })?;
        // Bit-identical tangent anchors: f32::midpoint rounds differently at overflow/subnormal edges.
        #[allow(clippy::manual_midpoint)]
        let midpoint = Array1::from_iter(
            lower
                .iter()
                .zip(upper.iter())
                .map(|(&l, &u)| 0.5_f32 * (l + u)),
        );
        let d_pairs: Vec<(f32, f32)> = lower
            .iter()
            .zip(upper.iter())
            .map(|(&l, &u)| crossing_tangents(l, u))
            .collect();
        let d_lower = Array1::from_iter(d_pairs.iter().map(|(dl, _)| *dl));
        let d_upper = Array1::from_iter(d_pairs.iter().map(|(_, du)| *du));
        let mask_neg = Array1::from_iter(
            lower
                .iter()
                .zip(upper.iter())
                .map(|(&l, &u)| u <= 0.0 || l > u),
        );
        let mask_pos = Array1::from_iter(
            lower
                .iter()
                .zip(upper.iter())
                .map(|(&l, &u)| l >= 0.0 && l <= u),
        );
        let mask_cross = Array1::from_iter(
            lower
                .iter()
                .zip(upper.iter())
                .map(|(&l, &u)| l < 0.0 && u > 0.0),
        );

        Ok(Self {
            tp_pos: MonotoneSShapedDualParams::new(&midpoint),
            tp_neg: MonotoneSShapedDualParams::new(&midpoint),
            tp_both_lower: MonotoneSShapedDualParams::new(&d_lower),
            tp_both_upper: MonotoneSShapedDualParams::new(&d_upper),
            mask_neg,
            mask_pos,
            mask_cross,
            lower_bounds: lower,
            upper_bounds: upper,
            midpoint,
            d_lower,
            d_upper,
        })
    }

    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.lower_bounds.len()
    }

    #[must_use]
    pub(crate) fn lower_path_alpha(&self, idx: usize) -> MonotoneSShapedPathAlpha {
        MonotoneSShapedPathAlpha {
            tp_pos: self.tp_pos.lower_path[idx],
            tp_neg: self.tp_neg.lower_path[idx],
            tp_both_lower: self.tp_both_lower.lower_path[idx],
            tp_both_upper: self.tp_both_upper.lower_path[idx],
            d_lower: self.d_lower[idx],
            d_upper: self.d_upper[idx],
        }
    }

    #[must_use]
    pub(crate) fn upper_path_alpha(&self, idx: usize) -> MonotoneSShapedPathAlpha {
        MonotoneSShapedPathAlpha {
            tp_pos: self.tp_pos.upper_path[idx],
            tp_neg: self.tp_neg.upper_path[idx],
            tp_both_lower: self.tp_both_lower.upper_path[idx],
            tp_both_upper: self.tp_both_upper.upper_path[idx],
            d_lower: self.d_lower[idx],
            d_upper: self.d_upper[idx],
        }
    }

    pub(crate) fn warm_start_from(&mut self, parent: &Self) {
        warm_start_dual_params(
            &mut self.tp_pos,
            &parent.tp_pos,
            &self.mask_pos,
            &self.midpoint,
            |idx, value| {
                clamp_or_reset(
                    value,
                    Some(self.lower_bounds[idx]),
                    Some(self.upper_bounds[idx]),
                    self.midpoint[idx],
                )
            },
        );
        warm_start_dual_params(
            &mut self.tp_neg,
            &parent.tp_neg,
            &self.mask_neg,
            &self.midpoint,
            |idx, value| {
                clamp_or_reset(
                    value,
                    Some(self.lower_bounds[idx]),
                    Some(self.upper_bounds[idx]),
                    self.midpoint[idx],
                )
            },
        );
        warm_start_dual_params(
            &mut self.tp_both_lower,
            &parent.tp_both_lower,
            &self.mask_cross,
            &self.d_lower,
            |idx, value| clamp_or_reset(value, None, Some(self.d_lower[idx]), self.d_lower[idx]),
        );
        warm_start_dual_params(
            &mut self.tp_both_upper,
            &parent.tp_both_upper,
            &self.mask_cross,
            &self.d_upper,
            |idx, value| clamp_or_reset(value, Some(self.d_upper[idx]), None, self.d_upper[idx]),
        );
    }
}

fn warm_start_dual_params<F>(
    values: &mut MonotoneSShapedDualParams,
    parent: &MonotoneSShapedDualParams,
    mask: &Array1<bool>,
    reset: &Array1<f32>,
    clamp: F,
) where
    F: Fn(usize, f32) -> f32,
{
    warm_start_array(
        &mut values.lower_path,
        &parent.lower_path,
        mask,
        reset,
        &clamp,
    );
    warm_start_array(
        &mut values.upper_path,
        &parent.upper_path,
        mask,
        reset,
        clamp,
    );
}

fn warm_start_array<F>(
    values: &mut Array1<f32>,
    parent: &Array1<f32>,
    mask: &Array1<bool>,
    reset: &Array1<f32>,
    clamp: F,
) where
    F: Fn(usize, f32) -> f32,
{
    let len = values
        .len()
        .min(parent.len())
        .min(mask.len())
        .min(reset.len());
    for i in 0..len {
        if mask[i] {
            values[i] = if parent[i].is_finite() {
                clamp(i, parent[i])
            } else {
                reset[i]
            };
        }
    }
}

fn clamp_or_reset(value: f32, lower: Option<f32>, upper: Option<f32>, reset: f32) -> f32 {
    if !value.is_finite() {
        return reset;
    }

    let clamped = value.clamp(
        lower.unwrap_or(f32::NEG_INFINITY),
        upper.unwrap_or(f32::INFINITY),
    );
    if clamped.is_nan() {
        reset
    } else {
        clamped
    }
}
