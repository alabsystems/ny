// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Reciprocal alpha parameters for DAG alpha-CROWN optimization.
//!
//! Mirrors the Sqrt alpha infrastructure (`alpha_sqrt.rs`). For positive-domain
//! Reciprocal (convex), the tangent point is the optimizable lower-bound
//! parameter. For negative-domain (concave), the tangent is the upper-bound
//! parameter.
//!
//! Reference: alpha-beta-CROWN `operators/convex_concave.py:66-196`

use ndarray::Array1;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use super::alpha_config::AdamParams;

const RECIPROCAL_TANGENT_GUARD: f32 = 1e-6;

#[derive(Debug, Clone)]
pub(crate) struct ReciprocalGradients {
    pub(crate) lower_path: Array1<f32>,
    pub(crate) upper_path: Array1<f32>,
}

impl ReciprocalGradients {
    #[must_use]
    pub(crate) fn any_non_finite(&self) -> bool {
        self.lower_path.iter().any(|v| !v.is_finite())
            || self.upper_path.iter().any(|v| !v.is_finite())
    }

    #[must_use]
    pub(crate) fn negate(&self) -> Self {
        Self {
            lower_path: self.lower_path.mapv(|v| -v),
            upper_path: self.upper_path.mapv(|v| -v),
        }
    }

    pub(crate) fn scale_in_place(&mut self, scale: f32) {
        self.lower_path *= scale;
        self.upper_path *= scale;
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ReciprocalAlpha {
    pub(crate) lower_path_mid: Array1<f32>,
    pub(crate) upper_path_mid: Array1<f32>,
    pub(crate) velocity_lower: Array1<f32>,
    pub(crate) velocity_upper: Array1<f32>,
    pub(crate) adam_m_lower: Array1<f32>,
    pub(crate) adam_v_lower: Array1<f32>,
    pub(crate) adam_m_upper: Array1<f32>,
    pub(crate) adam_v_upper: Array1<f32>,
    pub(crate) lower_bounds: Array1<f32>,
    pub(crate) upper_bounds: Array1<f32>,
    pub(crate) midpoint: Array1<f32>,
    pub(crate) active_mask: Array1<bool>,
}

impl ReciprocalAlpha {
    pub(crate) fn from_bounds(pre_activation: &BoundedTensor) -> Result<Self> {
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
        let midpoint = Array1::from_iter(
            lower
                .iter()
                .zip(upper.iter())
                .map(|(&l, &u)| reciprocal_midpoint(l, u)),
        );
        // Active only when both bounds are finite and the interval doesn't cross zero.
        let active_mask = Array1::from_iter(
            lower
                .iter()
                .zip(upper.iter())
                .map(|(&l, &u)| l.is_finite() && u.is_finite() && l <= u && (l > 0.0 || u < 0.0)),
        );
        let safe_lower = Array1::from_iter(lower.iter().zip(midpoint.iter()).map(|(&l, &mid)| {
            if l.is_finite() {
                l
            } else {
                mid
            }
        }));
        let safe_upper = Array1::from_iter(upper.iter().zip(midpoint.iter()).map(|(&u, &mid)| {
            if u.is_finite() {
                u
            } else {
                mid
            }
        }));
        let zeros = Array1::zeros(midpoint.len());

        Ok(Self {
            lower_path_mid: midpoint.clone(),
            upper_path_mid: midpoint.clone(),
            velocity_lower: zeros.clone(),
            velocity_upper: zeros.clone(),
            adam_m_lower: zeros.clone(),
            adam_v_lower: zeros.clone(),
            adam_m_upper: zeros.clone(),
            adam_v_upper: zeros,
            lower_bounds: safe_lower,
            upper_bounds: safe_upper,
            midpoint,
            active_mask,
        })
    }

    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.midpoint.len()
    }

    #[must_use]
    pub(crate) fn zeros_gradients(&self) -> ReciprocalGradients {
        ReciprocalGradients {
            lower_path: Array1::zeros(self.len()),
            upper_path: Array1::zeros(self.len()),
        }
    }

    pub(crate) fn warm_start_from(&mut self, parent: &Self) {
        let len = self.len().min(parent.len());
        for idx in 0..len {
            self.lower_path_mid[idx] = if self.active_mask[idx] {
                clamp_or_reset(
                    parent.lower_path_mid[idx],
                    self.lower_bounds[idx],
                    self.upper_bounds[idx],
                    self.midpoint[idx],
                )
            } else {
                self.midpoint[idx]
            };
            self.upper_path_mid[idx] = if self.active_mask[idx] {
                clamp_or_reset(
                    parent.upper_path_mid[idx],
                    self.lower_bounds[idx],
                    self.upper_bounds[idx],
                    self.midpoint[idx],
                )
            } else {
                self.midpoint[idx]
            };
        }
    }

    pub(crate) fn spsa_perturbations<R>(&self, rng: &mut R) -> ReciprocalGradients
    where
        R: rand::Rng + rand::RngExt + ?Sized,
    {
        let sample = |mask: &Array1<bool>, rng: &mut R| {
            Array1::from_iter(mask.iter().map(|&active| {
                if active {
                    if rng.random_bool(0.5) {
                        1.0
                    } else {
                        -1.0
                    }
                } else {
                    0.0
                }
            }))
        };

        ReciprocalGradients {
            lower_path: sample(&self.active_mask, rng),
            upper_path: sample(&self.active_mask, rng),
        }
    }

    pub(crate) fn apply_perturbation(&mut self, perturbation: &ReciprocalGradients, epsilon: f32) {
        Self::apply_update_to_path(
            &perturbation.lower_path,
            epsilon,
            &mut self.lower_path_mid,
            &mut self.velocity_lower,
            None,
            None,
            &self.lower_bounds,
            &self.upper_bounds,
            &self.midpoint,
            &self.active_mask,
        );
        Self::apply_update_to_path(
            &perturbation.upper_path,
            epsilon,
            &mut self.upper_path_mid,
            &mut self.velocity_upper,
            None,
            None,
            &self.lower_bounds,
            &self.upper_bounds,
            &self.midpoint,
            &self.active_mask,
        );
    }

    pub(crate) fn update_sgd(
        &mut self,
        gradients: &ReciprocalGradients,
        learning_rate: f32,
        momentum: f32,
    ) {
        Self::update_path_sgd(
            &gradients.lower_path,
            learning_rate,
            momentum,
            &mut self.lower_path_mid,
            &mut self.velocity_lower,
            &self.lower_bounds,
            &self.upper_bounds,
            &self.midpoint,
            &self.active_mask,
        );
        Self::update_path_sgd(
            &gradients.upper_path,
            learning_rate,
            momentum,
            &mut self.upper_path_mid,
            &mut self.velocity_upper,
            &self.lower_bounds,
            &self.upper_bounds,
            &self.midpoint,
            &self.active_mask,
        );
    }

    pub(crate) fn update_adam(&mut self, gradients: &ReciprocalGradients, params: &AdamParams) {
        Self::update_path_adam(
            &gradients.lower_path,
            params,
            &mut self.lower_path_mid,
            &mut self.adam_m_lower,
            &mut self.adam_v_lower,
            &self.lower_bounds,
            &self.upper_bounds,
            &self.midpoint,
            &self.active_mask,
        );
        Self::update_path_adam(
            &gradients.upper_path,
            params,
            &mut self.upper_path_mid,
            &mut self.adam_m_upper,
            &mut self.adam_v_upper,
            &self.lower_bounds,
            &self.upper_bounds,
            &self.midpoint,
            &self.active_mask,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_update_to_path(
        delta: &Array1<f32>,
        scale: f32,
        path: &mut Array1<f32>,
        velocity: &mut Array1<f32>,
        adam_m: Option<&mut Array1<f32>>,
        adam_v: Option<&mut Array1<f32>>,
        lower_bounds: &Array1<f32>,
        upper_bounds: &Array1<f32>,
        midpoint: &Array1<f32>,
        active_mask: &Array1<bool>,
    ) {
        let len = path
            .len()
            .min(delta.len())
            .min(lower_bounds.len())
            .min(upper_bounds.len())
            .min(midpoint.len())
            .min(active_mask.len())
            .min(velocity.len());
        let mut adam_m = adam_m;
        let mut adam_v = adam_v;

        for idx in 0..len {
            if active_mask[idx] {
                let next = path[idx] + scale * delta[idx];
                path[idx] =
                    clamp_or_reset(next, lower_bounds[idx], upper_bounds[idx], midpoint[idx]);
            } else {
                path[idx] = midpoint[idx];
                velocity[idx] = 0.0;
                if let Some(values) = adam_m.as_deref_mut() {
                    values[idx] = 0.0;
                }
                if let Some(values) = adam_v.as_deref_mut() {
                    values[idx] = 0.0;
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn update_path_sgd(
        gradients: &Array1<f32>,
        learning_rate: f32,
        momentum: f32,
        path: &mut Array1<f32>,
        velocity: &mut Array1<f32>,
        lower_bounds: &Array1<f32>,
        upper_bounds: &Array1<f32>,
        midpoint: &Array1<f32>,
        active_mask: &Array1<bool>,
    ) {
        let len = path
            .len()
            .min(gradients.len())
            .min(velocity.len())
            .min(lower_bounds.len())
            .min(upper_bounds.len())
            .min(midpoint.len())
            .min(active_mask.len());
        for idx in 0..len {
            if active_mask[idx] {
                velocity[idx] = momentum * velocity[idx] - learning_rate * gradients[idx];
                path[idx] = clamp_or_reset(
                    path[idx] + velocity[idx],
                    lower_bounds[idx],
                    upper_bounds[idx],
                    midpoint[idx],
                );
                if !path[idx].is_finite() {
                    path[idx] = midpoint[idx];
                    velocity[idx] = 0.0;
                }
            } else {
                path[idx] = midpoint[idx];
                velocity[idx] = 0.0;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn update_path_adam(
        gradients: &Array1<f32>,
        params: &AdamParams,
        path: &mut Array1<f32>,
        adam_m: &mut Array1<f32>,
        adam_v: &mut Array1<f32>,
        lower_bounds: &Array1<f32>,
        upper_bounds: &Array1<f32>,
        midpoint: &Array1<f32>,
        active_mask: &Array1<bool>,
    ) {
        let len = path
            .len()
            .min(gradients.len())
            .min(adam_m.len())
            .min(adam_v.len())
            .min(lower_bounds.len())
            .min(upper_bounds.len())
            .min(midpoint.len())
            .min(active_mask.len());
        let t_f = params.t.max(1) as f32;
        let bias_correction1 = (1.0 - params.beta1.powf(t_f)).max(f32::EPSILON);
        let bias_correction2 = (1.0 - params.beta2.powf(t_f)).max(f32::EPSILON);

        for idx in 0..len {
            if active_mask[idx] {
                let g = gradients[idx];
                adam_m[idx] = params.beta1 * adam_m[idx] + (1.0 - params.beta1) * g;
                adam_v[idx] = params.beta2 * adam_v[idx] + (1.0 - params.beta2) * g * g;
                let m_hat = adam_m[idx] / bias_correction1;
                let v_hat = adam_v[idx] / bias_correction2;
                path[idx] = clamp_or_reset(
                    path[idx] - params.learning_rate * m_hat / (v_hat.sqrt() + params.epsilon),
                    lower_bounds[idx],
                    upper_bounds[idx],
                    midpoint[idx],
                );
                if !path[idx].is_finite() {
                    path[idx] = midpoint[idx];
                    adam_m[idx] = 0.0;
                    adam_v[idx] = 0.0;
                }
            } else {
                path[idx] = midpoint[idx];
                adam_m[idx] = 0.0;
                adam_v[idx] = 0.0;
            }
        }
    }
}

fn reciprocal_midpoint(lower: f32, upper: f32) -> f32 {
    if lower.is_finite() && upper.is_finite() && lower <= upper && (lower > 0.0 || upper < 0.0) {
        // Bit-identical tangent anchor: f32::midpoint rounds differently at overflow/subnormal edges.
        #[allow(clippy::manual_midpoint)]
        let mid = 0.5 * (lower + upper);
        clamp_or_reset(mid, lower, upper, if lower > 0.0 { lower } else { upper })
    } else if lower.is_finite() && lower != 0.0 {
        lower
    } else if upper.is_finite() && upper != 0.0 {
        upper
    } else {
        1.0
    }
}

fn clamp_or_reset(value: f32, lower: f32, upper: f32, reset: f32) -> f32 {
    if value.is_finite() && lower.is_finite() && upper.is_finite() && lower <= upper {
        // Guard: keep tangent point away from zero to avoid 1/0 in slope computation.
        let effective_lower = if lower > 0.0 {
            lower.max(RECIPROCAL_TANGENT_GUARD.min(upper))
        } else {
            lower.min(-RECIPROCAL_TANGENT_GUARD.min(-lower))
        };
        let effective_upper = if upper < 0.0 {
            upper.min(-RECIPROCAL_TANGENT_GUARD.min(-lower))
        } else {
            upper
        };
        value.clamp(effective_lower, effective_upper)
    } else {
        reset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    fn unit_bounds(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        BoundedTensor::new(arr1(lower).into_dyn(), arr1(upper).into_dyn())
            .expect("test bounds should construct")
    }

    #[test]
    fn test_reciprocal_alpha_from_bounds_initializes_positive_midpoints() {
        let bounds = unit_bounds(&[0.5, 1.0, -2.0], &[2.0, 3.0, -0.5]);
        let alpha =
            ReciprocalAlpha::from_bounds(&bounds).expect("reciprocal alpha init should succeed");

        assert_eq!(alpha.lower_path_mid.len(), 3);
        assert!(alpha.active_mask[0], "positive interval should be active");
        assert!(alpha.active_mask[1], "positive interval should be active");
        assert!(alpha.active_mask[2], "negative interval should be active");
    }

    #[test]
    fn test_reciprocal_alpha_from_bounds_rejects_zero_crossing() {
        let bounds = unit_bounds(&[-1.0, 0.5], &[1.0, 2.0]);
        let alpha =
            ReciprocalAlpha::from_bounds(&bounds).expect("reciprocal alpha init should succeed");

        assert!(
            !alpha.active_mask[0],
            "zero-crossing interval must stay inactive"
        );
        assert!(alpha.active_mask[1], "positive interval should be active");
    }

    #[test]
    fn test_reciprocal_alpha_warm_start_clamps() {
        let parent_bounds = unit_bounds(&[0.5, 1.0], &[4.0, 3.0]);
        let child_bounds = unit_bounds(&[1.0, 1.5], &[2.0, 2.5]);
        let parent = ReciprocalAlpha::from_bounds(&parent_bounds).expect("parent init");
        let mut child = ReciprocalAlpha::from_bounds(&child_bounds).expect("child init");

        child.lower_path_mid.fill(0.0);
        child.upper_path_mid.fill(0.0);
        child.warm_start_from(&parent);

        for idx in 0..child.len() {
            assert!(
                child.lower_path_mid[idx] >= child.lower_bounds[idx]
                    && child.lower_path_mid[idx] <= child.upper_bounds[idx],
                "lower warm-start midpoint must stay within child bounds"
            );
            assert!(
                child.upper_path_mid[idx] >= child.lower_bounds[idx]
                    && child.upper_path_mid[idx] <= child.upper_bounds[idx],
                "upper warm-start midpoint must stay within child bounds"
            );
        }
    }
}
