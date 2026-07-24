// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Adam momentum state for AdamClipping PGD optimizer.
//!
//! Reference: alpha-beta-CROWN `attack_utils.py:225-309`

use ndarray::{ArrayD, Zip};

use super::optimizer::{clamp_to_bounds, AdamClippingParams};

/// Adam-style momentum state for one PGD restart.
///
/// Implements the AdamClipping algorithm from alpha-beta-CROWN: standard Adam
/// moment tracking used only for sign-based direction, with uniform Linf step
/// size and exponential LR decay.
///
/// Reference: alpha-beta-CROWN `attack_utils.py:225-309`
pub(super) struct AdamClippingState {
    /// First moment estimate (exponential moving average of gradients).
    pub(super) exp_avg: ArrayD<f32>,
    /// Second moment estimate (exponential moving average of squared gradients).
    pub(super) exp_avg_sq: ArrayD<f32>,
    /// Step counter (for bias correction).
    pub(super) step_count: u32,
    /// Parameters for this optimizer instance.
    params: AdamClippingParams,
}

impl AdamClippingState {
    pub(super) fn new(shape: &[usize], params: AdamClippingParams) -> Self {
        Self {
            exp_avg: ArrayD::zeros(shape),
            exp_avg_sq: ArrayD::zeros(shape),
            step_count: 0,
            params,
        }
    }

    /// Compute one AdamClipping step: update moments, return the projected
    /// next point.
    ///
    /// The step rule uses Adam direction (sign of bias-corrected m/v ratio)
    /// with uniform Linf step size, matching the reference implementation.
    pub(super) fn step(
        &mut self,
        gradient: &ArrayD<f32>,
        current_x: &ArrayD<f32>,
        lower: &ArrayD<f32>,
        upper: &ArrayD<f32>,
    ) -> ArrayD<f32> {
        self.step_count += 1;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let eps = self.params.adam_eps;

        self.exp_avg
            .zip_mut_with(gradient, |m, &g| *m = beta1 * *m + (1.0 - beta1) * g);
        self.exp_avg_sq
            .zip_mut_with(gradient, |v, &g| *v = beta2 * *v + (1.0 - beta2) * g * g);

        let bias_correction1 = 1.0 - beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step_count as i32);

        // Match ExponentialLR ordering: step 1 uses initial_lr, then decays.
        let current_lr =
            self.params.initial_lr * self.params.lr_decay.powi(self.step_count as i32 - 1);
        let step_size = current_lr / bias_correction1;

        let mut result = current_x.clone();
        Zip::from(&mut result)
            .and(&self.exp_avg)
            .and(&self.exp_avg_sq)
            .and(lower)
            .and(upper)
            .for_each(|x, &m, &v, &lo, &hi| {
                let denom = (v / bias_correction2).sqrt() + eps;
                let direction = if denom.is_finite() && m.is_finite() {
                    (m / denom).signum()
                } else {
                    0.0
                };
                *x = clamp_to_bounds(*x + direction * step_size, lo, hi);
            });
        result
    }
}
