// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::Array1;

use super::alpha_config::AdamParams;
use super::alpha_s_shaped::{MonotoneSShapedAlpha, MonotoneSShapedDualParams};

/// Gradient or perturbation values for one monotone S-shaped alpha parameter group.
#[derive(Debug, Clone)]
pub(crate) struct MonotoneSShapedGradientGroup {
    pub(crate) lower_path: Array1<f32>,
    pub(crate) upper_path: Array1<f32>,
}

impl MonotoneSShapedGradientGroup {
    fn zeros(len: usize) -> Self {
        Self {
            lower_path: Array1::zeros(len),
            upper_path: Array1::zeros(len),
        }
    }

    fn any_non_finite(&self) -> bool {
        self.lower_path.iter().any(|v| !v.is_finite())
            || self.upper_path.iter().any(|v| !v.is_finite())
    }

    fn negate(&self) -> Self {
        Self {
            lower_path: self.lower_path.mapv(|v| -v),
            upper_path: self.upper_path.mapv(|v| -v),
        }
    }

    fn scale_in_place(&mut self, scale: f32) {
        self.lower_path *= scale;
        self.upper_path *= scale;
    }
}

/// Gradient or SPSA perturbation container for all monotone S-shaped tangent groups.
#[derive(Debug, Clone)]
pub(crate) struct MonotoneSShapedGradients {
    pub(crate) tp_pos: MonotoneSShapedGradientGroup,
    pub(crate) tp_neg: MonotoneSShapedGradientGroup,
    pub(crate) tp_both_lower: MonotoneSShapedGradientGroup,
    pub(crate) tp_both_upper: MonotoneSShapedGradientGroup,
}

impl MonotoneSShapedGradients {
    pub(crate) fn any_non_finite(&self) -> bool {
        self.tp_pos.any_non_finite()
            || self.tp_neg.any_non_finite()
            || self.tp_both_lower.any_non_finite()
            || self.tp_both_upper.any_non_finite()
    }

    pub(crate) fn negate(&self) -> Self {
        Self {
            tp_pos: self.tp_pos.negate(),
            tp_neg: self.tp_neg.negate(),
            tp_both_lower: self.tp_both_lower.negate(),
            tp_both_upper: self.tp_both_upper.negate(),
        }
    }

    pub(crate) fn scale_in_place(&mut self, scale: f32) {
        self.tp_pos.scale_in_place(scale);
        self.tp_neg.scale_in_place(scale);
        self.tp_both_lower.scale_in_place(scale);
        self.tp_both_upper.scale_in_place(scale);
    }
}

impl MonotoneSShapedAlpha {
    #[must_use]
    pub(crate) fn zeros_gradients(&self) -> MonotoneSShapedGradients {
        let len = self.len();
        MonotoneSShapedGradients {
            tp_pos: MonotoneSShapedGradientGroup::zeros(len),
            tp_neg: MonotoneSShapedGradientGroup::zeros(len),
            tp_both_lower: MonotoneSShapedGradientGroup::zeros(len),
            tp_both_upper: MonotoneSShapedGradientGroup::zeros(len),
        }
    }

    pub(crate) fn spsa_perturbations<R>(&self, rng: &mut R) -> MonotoneSShapedGradients
    where
        R: rand::Rng + rand::RngExt + ?Sized,
    {
        fn bernoulli_masked<R>(rng: &mut R, mask: &Array1<bool>) -> MonotoneSShapedGradientGroup
        where
            R: rand::Rng + rand::RngExt + ?Sized,
        {
            let lower_path = Array1::from_iter(mask.iter().map(|&active| {
                if active {
                    if rng.random_bool(0.5) {
                        1.0
                    } else {
                        -1.0
                    }
                } else {
                    0.0
                }
            }));
            let upper_path = Array1::from_iter(mask.iter().map(|&active| {
                if active {
                    if rng.random_bool(0.5) {
                        1.0
                    } else {
                        -1.0
                    }
                } else {
                    0.0
                }
            }));
            MonotoneSShapedGradientGroup {
                lower_path,
                upper_path,
            }
        }

        MonotoneSShapedGradients {
            tp_pos: bernoulli_masked(rng, &self.mask_pos),
            tp_neg: bernoulli_masked(rng, &self.mask_neg),
            tp_both_lower: bernoulli_masked(rng, &self.mask_cross),
            tp_both_upper: bernoulli_masked(rng, &self.mask_cross),
        }
    }

    pub(crate) fn apply_perturbation(
        &mut self,
        perturbations: &MonotoneSShapedGradients,
        epsilon: f32,
    ) {
        Self::apply_group_perturbation(
            &mut self.tp_pos,
            &perturbations.tp_pos,
            &self.mask_pos,
            GroupClamp::Bounded {
                lower: &self.lower_bounds,
                upper: &self.upper_bounds,
                reset: &self.midpoint,
            },
            epsilon,
        );
        Self::apply_group_perturbation(
            &mut self.tp_neg,
            &perturbations.tp_neg,
            &self.mask_neg,
            GroupClamp::Bounded {
                lower: &self.lower_bounds,
                upper: &self.upper_bounds,
                reset: &self.midpoint,
            },
            epsilon,
        );
        Self::apply_group_perturbation(
            &mut self.tp_both_lower,
            &perturbations.tp_both_lower,
            &self.mask_cross,
            GroupClamp::UpperOnly {
                upper: &self.d_lower,
                reset: &self.d_lower,
            },
            epsilon,
        );
        Self::apply_group_perturbation(
            &mut self.tp_both_upper,
            &perturbations.tp_both_upper,
            &self.mask_cross,
            GroupClamp::LowerOnly {
                lower: &self.d_upper,
                reset: &self.d_upper,
            },
            epsilon,
        );
    }

    pub(crate) fn update_sgd(
        &mut self,
        gradients: &MonotoneSShapedGradients,
        learning_rate: f32,
        momentum: f32,
    ) {
        Self::update_group_sgd(
            &mut self.tp_pos,
            &gradients.tp_pos,
            &self.mask_pos,
            GroupClamp::Bounded {
                lower: &self.lower_bounds,
                upper: &self.upper_bounds,
                reset: &self.midpoint,
            },
            learning_rate,
            momentum,
        );
        Self::update_group_sgd(
            &mut self.tp_neg,
            &gradients.tp_neg,
            &self.mask_neg,
            GroupClamp::Bounded {
                lower: &self.lower_bounds,
                upper: &self.upper_bounds,
                reset: &self.midpoint,
            },
            learning_rate,
            momentum,
        );
        Self::update_group_sgd(
            &mut self.tp_both_lower,
            &gradients.tp_both_lower,
            &self.mask_cross,
            GroupClamp::UpperOnly {
                upper: &self.d_lower,
                reset: &self.d_lower,
            },
            learning_rate,
            momentum,
        );
        Self::update_group_sgd(
            &mut self.tp_both_upper,
            &gradients.tp_both_upper,
            &self.mask_cross,
            GroupClamp::LowerOnly {
                lower: &self.d_upper,
                reset: &self.d_upper,
            },
            learning_rate,
            momentum,
        );
    }

    pub(crate) fn update_adam(
        &mut self,
        gradients: &MonotoneSShapedGradients,
        params: &AdamParams,
    ) {
        Self::update_group_adam(
            &mut self.tp_pos,
            &gradients.tp_pos,
            &self.mask_pos,
            GroupClamp::Bounded {
                lower: &self.lower_bounds,
                upper: &self.upper_bounds,
                reset: &self.midpoint,
            },
            params,
        );
        Self::update_group_adam(
            &mut self.tp_neg,
            &gradients.tp_neg,
            &self.mask_neg,
            GroupClamp::Bounded {
                lower: &self.lower_bounds,
                upper: &self.upper_bounds,
                reset: &self.midpoint,
            },
            params,
        );
        Self::update_group_adam(
            &mut self.tp_both_lower,
            &gradients.tp_both_lower,
            &self.mask_cross,
            GroupClamp::UpperOnly {
                upper: &self.d_lower,
                reset: &self.d_lower,
            },
            params,
        );
        Self::update_group_adam(
            &mut self.tp_both_upper,
            &gradients.tp_both_upper,
            &self.mask_cross,
            GroupClamp::LowerOnly {
                lower: &self.d_upper,
                reset: &self.d_upper,
            },
            params,
        );
    }

    fn apply_group_perturbation(
        group: &mut MonotoneSShapedDualParams,
        perturbations: &MonotoneSShapedGradientGroup,
        mask: &Array1<bool>,
        clamp: GroupClamp<'_>,
        epsilon: f32,
    ) {
        apply_perturbation_to_array(
            &mut group.lower_path,
            &perturbations.lower_path,
            mask,
            clamp,
            epsilon,
        );
        apply_perturbation_to_array(
            &mut group.upper_path,
            &perturbations.upper_path,
            mask,
            clamp,
            epsilon,
        );
    }

    fn update_group_sgd(
        group: &mut MonotoneSShapedDualParams,
        gradients: &MonotoneSShapedGradientGroup,
        mask: &Array1<bool>,
        clamp: GroupClamp<'_>,
        learning_rate: f32,
        momentum: f32,
    ) {
        update_sgd_with_clamp(
            &mut group.lower_path,
            &gradients.lower_path,
            mask,
            &mut group.velocity_lower,
            learning_rate,
            momentum,
            clamp,
        );
        update_sgd_with_clamp(
            &mut group.upper_path,
            &gradients.upper_path,
            mask,
            &mut group.velocity_upper,
            learning_rate,
            momentum,
            clamp,
        );
    }

    fn update_group_adam(
        group: &mut MonotoneSShapedDualParams,
        gradients: &MonotoneSShapedGradientGroup,
        mask: &Array1<bool>,
        clamp: GroupClamp<'_>,
        params: &AdamParams,
    ) {
        update_adam_with_clamp(
            &mut group.lower_path,
            &gradients.lower_path,
            mask,
            &mut group.adam_m_lower,
            &mut group.adam_v_lower,
            params,
            clamp,
        );
        update_adam_with_clamp(
            &mut group.upper_path,
            &gradients.upper_path,
            mask,
            &mut group.adam_m_upper,
            &mut group.adam_v_upper,
            params,
            clamp,
        );
    }
}

#[derive(Clone, Copy)]
enum GroupClamp<'a> {
    Bounded {
        lower: &'a Array1<f32>,
        upper: &'a Array1<f32>,
        reset: &'a Array1<f32>,
    },
    UpperOnly {
        upper: &'a Array1<f32>,
        reset: &'a Array1<f32>,
    },
    LowerOnly {
        lower: &'a Array1<f32>,
        reset: &'a Array1<f32>,
    },
}

fn clamp_value_with_reset(value: f32, idx: usize, clamp: GroupClamp<'_>) -> f32 {
    let clamped = match clamp {
        GroupClamp::Bounded { lower, upper, .. } => value.clamp(lower[idx], upper[idx]),
        GroupClamp::UpperOnly { upper, .. } => value.min(upper[idx]),
        GroupClamp::LowerOnly { lower, .. } => value.max(lower[idx]),
    };
    if clamped.is_nan() {
        match clamp {
            GroupClamp::Bounded { reset, .. }
            | GroupClamp::UpperOnly { reset, .. }
            | GroupClamp::LowerOnly { reset, .. } => reset[idx],
        }
    } else {
        clamped
    }
}

fn apply_perturbation_to_array(
    values: &mut Array1<f32>,
    perturbations: &Array1<f32>,
    mask: &Array1<bool>,
    clamp: GroupClamp<'_>,
    epsilon: f32,
) {
    for i in 0..values.len() {
        if mask[i] && perturbations[i].abs() > 0.5 {
            values[i] = clamp_value_with_reset(values[i] + epsilon * perturbations[i], i, clamp);
        }
    }
}

fn update_sgd_with_clamp(
    values: &mut Array1<f32>,
    gradients: &Array1<f32>,
    mask: &Array1<bool>,
    velocity: &mut Array1<f32>,
    learning_rate: f32,
    momentum: f32,
    clamp: GroupClamp<'_>,
) {
    for i in 0..values.len() {
        if mask[i] {
            velocity[i] = momentum * velocity[i] - learning_rate * gradients[i];
            values[i] = clamp_value_with_reset(values[i] + velocity[i], i, clamp);
            if !values[i].is_finite() {
                values[i] = clamp_value_with_reset(f32::NAN, i, clamp);
                velocity[i] = 0.0;
            }
        }
    }
}

fn update_adam_with_clamp(
    values: &mut Array1<f32>,
    gradients: &Array1<f32>,
    mask: &Array1<bool>,
    adam_m: &mut Array1<f32>,
    adam_v: &mut Array1<f32>,
    params: &AdamParams,
    clamp: GroupClamp<'_>,
) {
    let t_f = params.t.max(1) as f32;
    let bias_correction1 = (1.0 - params.beta1.powf(t_f)).max(f32::EPSILON);
    let bias_correction2 = (1.0 - params.beta2.powf(t_f)).max(f32::EPSILON);

    for i in 0..values.len() {
        if mask[i] {
            let g = gradients[i];
            adam_m[i] = params.beta1 * adam_m[i] + (1.0 - params.beta1) * g;
            adam_v[i] = params.beta2 * adam_v[i] + (1.0 - params.beta2) * g * g;

            let m_hat = adam_m[i] / bias_correction1;
            let v_hat = adam_v[i] / bias_correction2;
            values[i] = clamp_value_with_reset(
                values[i] - params.learning_rate * m_hat / (v_hat.sqrt() + params.epsilon),
                i,
                clamp,
            );
            if !values[i].is_finite() {
                values[i] = clamp_value_with_reset(f32::NAN, i, clamp);
                adam_m[i] = 0.0;
                adam_v[i] = 0.0;
            }
        }
    }
}
