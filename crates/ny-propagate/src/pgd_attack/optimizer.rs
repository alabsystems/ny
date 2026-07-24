// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! AdamClipping optimizer for PGD attacks.
//!
//! Implements the AdamClipping algorithm from alpha-beta-CROWN: standard Adam
//! moment tracking used only for sign-based direction, with uniform Linf step
//! size and exponential LR decay.
//!
//! Reference: alpha-beta-CROWN `attack_utils.py:178-309`

use ndarray::{ArrayD, Zip};
use ny_tensor::BoundedTensor;
use serde::{Deserialize, Serialize};

use super::adam_state::AdamClippingState;

/// PGD optimizer strategy.
///
/// Reference: alpha-beta-CROWN `attack_pgd.py:253-260`
/// - `use_adam=True` uses AdamClipping with ExponentialLR decay
/// - `use_adam=False` uses vanilla signed gradient with constant step
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PgdOptimizer {
    /// Vanilla constant-step gradient update (legacy behavior).
    SignedGradient,
    /// AdamClipping: momentum-smoothed direction with Linf step and LR decay.
    /// Matches alpha-beta-CROWN reference.
    #[default]
    AdamClipping,
}

/// Parameters for the AdamClipping optimizer.
///
/// Reference: alpha-beta-CROWN `attack_utils.py:203-217`
/// Default values match the reference: beta1=0.9, beta2=0.999, eps=1e-8.
/// LR decay matches `torch.optim.lr_scheduler.ExponentialLR` with ny=0.99.
#[derive(Debug, Clone, Copy)]
pub struct AdamClippingParams {
    /// First moment decay rate. Default: 0.9 (reference default).
    pub beta1: f32,
    /// Second moment decay rate. Default: 0.999 (reference default).
    pub beta2: f32,
    /// Adam epsilon for numerical stability. Default: 1e-8.
    pub adam_eps: f32,
    /// Initial learning rate. Default: computed as max_eps / 4.
    ///
    /// Reference: alpha-beta-CROWN `attack_interface.py:102-114`
    pub initial_lr: f32,
    /// Per-step exponential decay factor. Default: 0.99.
    ///
    /// Reference: alpha-beta-CROWN `attack_pgd.py:255` uses
    /// `torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)`.
    pub lr_decay: f32,
}

impl Default for AdamClippingParams {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            adam_eps: 1e-8,
            initial_lr: 0.01,
            lr_decay: 0.99,
        }
    }
}

impl AdamClippingParams {
    /// Compute default params from the input perturbation budget,
    /// matching the reference `alpha = max_eps / 4` convention.
    ///
    /// Reference: alpha-beta-CROWN `attack_interface.py:102-114`
    pub fn from_perturbation_budget(input_bounds: &BoundedTensor) -> Self {
        Self {
            initial_lr: auto_alpha(input_bounds),
            ..Default::default()
        }
    }
}

/// Alpha (learning rate) mode for PGD attacks.
///
/// Reference: alpha-beta-CROWN `attack_interface.py:102-114`
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PgdAlphaMode {
    /// Automatically compute alpha from input perturbation budget.
    /// alpha = max_eps / 4, matching the reference.
    #[default]
    Auto,
    /// Fixed scalar step size (legacy compatibility).
    Scalar(f32),
    /// Scale the scalar alpha by the per-dimension input width.
    InputRangeScaled(f32),
}

fn max_input_width(input_bounds: &BoundedTensor) -> f32 {
    input_bounds
        .upper()
        .iter()
        .zip(input_bounds.lower().iter())
        .map(|(u, l)| (u - l).abs())
        .fold(0.0_f32, f32::max)
}

/// Compute the automatic alpha (learning rate) from input bounds.
///
/// Reference: alpha-beta-CROWN `attack_interface.py:102-114`
/// ```python
/// test_alpha = (prop_test['data_max'] - prop_test['data_min']) / 2
/// test_alpha = test_alpha.max().item() / 4
/// ```
pub fn auto_alpha(input_bounds: &BoundedTensor) -> f32 {
    let max_eps = max_input_width(input_bounds) / 2.0;
    (max_eps / 4.0).clamp(0.001, 1.0)
}

pub(super) fn clamp_to_bounds(value: f32, lower: f32, upper: f32) -> f32 {
    if value.is_nan() {
        lower
    } else {
        value.clamp(lower, upper)
    }
}

/// Project a point to the given lower/upper bounds, replacing NaNs with the
/// lower bound to keep downstream concrete evaluation well-defined.
pub fn project_to_bounds_in_place(
    point: &mut ArrayD<f32>,
    lower: &ArrayD<f32>,
    upper: &ArrayD<f32>,
) {
    Zip::from(point)
        .and(lower)
        .and(upper)
        .for_each(|value, &lo, &hi| *value = clamp_to_bounds(*value, lo, hi));
}

/// Return a projected copy of `point`.
pub fn project_to_bounds(
    point: &ArrayD<f32>,
    lower: &ArrayD<f32>,
    upper: &ArrayD<f32>,
) -> ArrayD<f32> {
    let mut projected = point.clone();
    project_to_bounds_in_place(&mut projected, lower, upper);
    projected
}

#[derive(Debug, Clone, Copy)]
enum SignedGradientScale {
    Scalar(f32),
    InputRangeScaled(f32),
}

fn resolve_signed_gradient_scale(
    alpha_mode: PgdAlphaMode,
    legacy_step_size: f32,
    input_bounds: &BoundedTensor,
) -> SignedGradientScale {
    match alpha_mode {
        PgdAlphaMode::Auto => SignedGradientScale::Scalar(auto_alpha(input_bounds)),
        PgdAlphaMode::Scalar(alpha) => SignedGradientScale::Scalar(alpha),
        PgdAlphaMode::InputRangeScaled(alpha) => {
            if alpha.is_finite() {
                SignedGradientScale::InputRangeScaled(alpha)
            } else {
                SignedGradientScale::Scalar(legacy_step_size)
            }
        }
    }
}

enum PgdStepKind {
    SignedGradient { scale: SignedGradientScale },
    Adam(Box<AdamClippingState>),
}

/// Unified step-state dispatcher for PGD.
///
/// Wraps either `SignedGradient` (vanilla constant/per-dimension step) or
/// `AdamClipping` (momentum-smoothed sign direction). Each restart creates one
/// `PgdStepState`.
pub struct PgdStepState {
    kind: PgdStepKind,
}

impl PgdStepState {
    /// Create a step state from config parameters.
    pub fn from_config(
        optimizer: PgdOptimizer,
        alpha_mode: PgdAlphaMode,
        legacy_step_size: f32,
        adam_params: AdamClippingParams,
        input_bounds: &BoundedTensor,
        shape: &[usize],
    ) -> Self {
        let kind = match optimizer {
            PgdOptimizer::SignedGradient => PgdStepKind::SignedGradient {
                scale: resolve_signed_gradient_scale(alpha_mode, legacy_step_size, input_bounds),
            },
            PgdOptimizer::AdamClipping => {
                let params = match alpha_mode {
                    PgdAlphaMode::Auto => {
                        AdamClippingParams::from_perturbation_budget(input_bounds)
                    }
                    PgdAlphaMode::Scalar(alpha) => AdamClippingParams {
                        initial_lr: alpha,
                        ..adam_params
                    },
                    PgdAlphaMode::InputRangeScaled(alpha) => AdamClippingParams {
                        initial_lr: alpha * max_input_width(input_bounds),
                        ..adam_params
                    },
                };
                PgdStepKind::Adam(Box::new(AdamClippingState::new(shape, params)))
            }
        };
        Self { kind }
    }

    /// Construct a signed-gradient step state directly.
    pub fn new_signed_gradient(
        alpha_mode: PgdAlphaMode,
        legacy_step_size: f32,
        input_bounds: &BoundedTensor,
    ) -> Self {
        Self::from_config(
            PgdOptimizer::SignedGradient,
            alpha_mode,
            legacy_step_size,
            AdamClippingParams::default(),
            input_bounds,
            input_bounds.shape(),
        )
    }

    /// Reset optimizer state for a new restart.
    pub fn reset(&mut self) {
        match &mut self.kind {
            PgdStepKind::SignedGradient { .. } => {}
            PgdStepKind::Adam(state) => {
                state.exp_avg.fill(0.0);
                state.exp_avg_sq.fill(0.0);
                state.step_count = 0;
            }
        }
    }

    /// Compute the next point after one gradient step.
    ///
    /// `verify_upper_bound=true` means ascend along the objective; `false`
    /// means descend. The result is always projected to input bounds.
    pub fn step(
        &mut self,
        gradient: &ArrayD<f32>,
        current_x: &ArrayD<f32>,
        input_bounds: &BoundedTensor,
        verify_upper_bound: bool,
    ) -> ArrayD<f32> {
        let lower = input_bounds.lower();
        let upper = input_bounds.upper();
        let direction = if verify_upper_bound {
            1.0_f32
        } else {
            -1.0_f32
        };

        match &mut self.kind {
            PgdStepKind::SignedGradient { scale } => {
                let mut result = current_x.clone();
                match scale {
                    SignedGradientScale::Scalar(alpha) => {
                        Zip::from(&mut result)
                            .and(gradient)
                            .and(lower)
                            .and(upper)
                            .for_each(|x, &g, &lo, &hi| {
                                *x = clamp_to_bounds(*x + direction * *alpha * g, lo, hi);
                            });
                    }
                    SignedGradientScale::InputRangeScaled(alpha) => {
                        Zip::from(&mut result)
                            .and(gradient)
                            .and(lower)
                            .and(upper)
                            .for_each(|x, &g, &lo, &hi| {
                                let scaled_alpha = *alpha * (hi - lo).abs();
                                *x = clamp_to_bounds(*x + direction * scaled_alpha * g, lo, hi);
                            });
                    }
                }
                result
            }
            PgdStepKind::Adam(state) => {
                let effective_gradient = if verify_upper_bound {
                    gradient.clone()
                } else {
                    gradient * -1.0
                };
                state.step(&effective_gradient, current_x, lower, upper)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_adam_degenerate_matches_signed_gradient() {
        let shape = &[3];
        let params = AdamClippingParams {
            beta1: 0.0,
            beta2: 0.0,
            adam_eps: 1e-8,
            initial_lr: 0.01,
            lr_decay: 1.0,
        };
        let mut adam_state = AdamClippingState::new(shape, params);

        let current_x = arr1(&[0.5_f32, 0.5, 0.5]).into_dyn();
        let gradient = arr1(&[1.0_f32, -2.0, 0.5]).into_dyn();
        let lower = arr1(&[0.0_f32, 0.0, 0.0]).into_dyn();
        let upper = arr1(&[1.0_f32, 1.0, 1.0]).into_dyn();

        let result = adam_state.step(&gradient, &current_x, &lower, &upper);
        let expected = arr1(&[0.51_f32, 0.49, 0.51]).into_dyn();
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "degenerate Adam mismatch: got {a}, expected {b}"
            );
        }
    }

    #[test]
    fn test_adam_lr_decay_uses_initial_lr_on_first_step() {
        let shape = &[1];
        let params = AdamClippingParams {
            beta1: 0.0,
            beta2: 0.0,
            adam_eps: 1e-8,
            initial_lr: 1.0,
            lr_decay: 0.5,
        };
        let mut state = AdamClippingState::new(shape, params);

        let gradient = arr1(&[1.0_f32]).into_dyn();
        let lower = arr1(&[-100.0_f32]).into_dyn();
        let upper = arr1(&[100.0_f32]).into_dyn();

        let x0 = arr1(&[0.0_f32]).into_dyn();
        let x1 = state.step(&gradient, &x0, &lower, &upper);
        let step1 = x1[[0]] - x0[[0]];

        let x2 = state.step(&gradient, &x1, &lower, &upper);
        let step2 = x2[[0]] - x1[[0]];

        assert!(
            (step1 - 1.0).abs() < 1e-6,
            "step1 should be ~1.0, got {step1}"
        );
        assert!(
            (step2 - 0.5).abs() < 1e-6,
            "step2 should be ~0.5, got {step2}"
        );
    }

    #[test]
    fn test_project_to_bounds_replaces_nan() {
        let mut point = arr1(&[f32::NAN, 2.0]).into_dyn();
        let lower = arr1(&[0.0_f32, -1.0]).into_dyn();
        let upper = arr1(&[1.0_f32, 1.0]).into_dyn();
        project_to_bounds_in_place(&mut point, &lower, &upper);
        assert_eq!(point[[0]], 0.0);
        assert_eq!(point[[1]], 1.0);
    }

    #[test]
    fn test_step_state_signed_gradient_legacy() {
        let bounds = BoundedTensor::new(
            arr1(&[0.0_f32, 0.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .unwrap();
        let mut state = PgdStepState::new_signed_gradient(PgdAlphaMode::Scalar(0.1), 0.1, &bounds);
        let x = arr1(&[0.5_f32, 0.5]).into_dyn();
        let gradient = arr1(&[1.0_f32, -1.0]).into_dyn();

        let result = state.step(&gradient, &x, &bounds, true);
        assert!((result[[0]] - 0.6).abs() < 1e-6);
        assert!((result[[1]] - 0.4).abs() < 1e-6);

        let result = state.step(&gradient, &x, &bounds, false);
        assert!((result[[0]] - 0.4).abs() < 1e-6);
        assert!((result[[1]] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_step_state_input_range_scaled() {
        let bounds = BoundedTensor::new(
            arr1(&[0.0_f32, -2.0]).into_dyn(),
            arr1(&[2.0_f32, 2.0]).into_dyn(),
        )
        .unwrap();
        let mut state =
            PgdStepState::new_signed_gradient(PgdAlphaMode::InputRangeScaled(0.1), 0.1, &bounds);
        let x = arr1(&[1.0_f32, 0.0]).into_dyn();
        let gradient = arr1(&[1.0_f32, 1.0]).into_dyn();
        let result = state.step(&gradient, &x, &bounds, true);
        assert!((result[[0]] - 1.2).abs() < 1e-6);
        assert!((result[[1]] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_step_state_adam_delegation() {
        let bounds = BoundedTensor::new(
            arr1(&[0.0_f32, 0.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .unwrap();
        let mut state = PgdStepState::from_config(
            PgdOptimizer::AdamClipping,
            PgdAlphaMode::Scalar(0.01),
            0.01,
            AdamClippingParams {
                beta1: 0.0,
                beta2: 0.0,
                adam_eps: 1e-8,
                initial_lr: 0.01,
                lr_decay: 1.0,
            },
            &bounds,
            &[2],
        );
        let x = arr1(&[0.5_f32, 0.5]).into_dyn();
        let gradient = arr1(&[1.0_f32, -1.0]).into_dyn();
        let result = state.step(&gradient, &x, &bounds, true);
        assert!((result[[0]] - 0.51).abs() < 1e-6);
        assert!((result[[1]] - 0.49).abs() < 1e-6);
    }
}
