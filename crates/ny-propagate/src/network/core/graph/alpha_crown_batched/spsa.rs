// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SPSA gradient estimation and Adam update for bilinear alpha optimization.
//!
//! Simultaneous Perturbation Stochastic Approximation (SPSA) estimates gradients
//! using random Bernoulli ±ε perturbations, requiring only 2 forward passes per
//! sample regardless of parameter count.

use std::collections::HashMap;

use crate::bounds::{AdamParams, AlphaCrownConfig};
use crate::network::alpha_crown_loop::finite_lower_sum;
use crate::network::core::graph::GraphNetwork;
use ndarray::Array4;
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;

/// SPSA perturbation magnitude for bilinear alpha gradients.
const BILINEAR_SPSA_EPS: f32 = 1e-3;

impl GraphNetwork {
    /// SPSA gradient estimation for bilinear alphas.
    ///
    /// Averages over `config.spsa_samples` samples to reduce variance.
    /// The `engine` parameter threads GPU GEMM acceleration into the
    /// ±ε batched CROWN evaluations (#3588).
    pub(super) fn spsa_bilinear_gradients(
        &self,
        input: &BoundedTensor,
        bilinear_alphas: &mut HashMap<String, Array4<f32>>,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<HashMap<String, Array4<f32>>> {
        use rand::RngExt;

        let mut rng = crate::random::rng();

        // Save original alphas.
        let originals: HashMap<String, Array4<f32>> = bilinear_alphas.clone();

        // Initialize gradient accumulators.
        let mut avg_grads: HashMap<String, Array4<f32>> = originals
            .iter()
            .map(|(name, alpha)| (name.clone(), Array4::zeros(alpha.raw_dim())))
            .collect();

        let result = (|| -> Result<()> {
            for _sample in 0..config.spsa_samples.max(1) {
                // Generate random Bernoulli perturbation for each alpha.
                let perturbations: HashMap<String, Array4<f32>> = originals
                    .iter()
                    .map(|(name, alpha)| {
                        let pert = Array4::from_shape_fn(alpha.raw_dim(), |_| {
                            if rng.random_bool(0.5) {
                                1.0_f32
                            } else {
                                -1.0_f32
                            }
                        });
                        (name.clone(), pert)
                    })
                    .collect();

                // Apply +ε perturbation.
                for (name, alpha) in bilinear_alphas.iter_mut() {
                    let orig = &originals[name];
                    let pert = &perturbations[name];
                    ndarray::Zip::from(alpha.view_mut())
                        .and(orig.view())
                        .and(pert.view())
                        .for_each(|a, &o, &p| {
                            *a = (o + BILINEAR_SPSA_EPS * p).clamp(0.0, 1.0);
                        });
                }
                let bounds_plus = self.propagate_crown_batched_with_bilinear_alphas_and_engine(
                    input,
                    bilinear_alphas,
                    engine,
                )?;
                let lower_plus = finite_lower_sum(bounds_plus.lower());

                // Apply -ε perturbation.
                for (name, alpha) in bilinear_alphas.iter_mut() {
                    let orig = &originals[name];
                    let pert = &perturbations[name];
                    ndarray::Zip::from(alpha.view_mut())
                        .and(orig.view())
                        .and(pert.view())
                        .for_each(|a, &o, &p| {
                            *a = (o - BILINEAR_SPSA_EPS * p).clamp(0.0, 1.0);
                        });
                }
                let bounds_minus = self.propagate_crown_batched_with_bilinear_alphas_and_engine(
                    input,
                    bilinear_alphas,
                    engine,
                )?;
                let lower_minus = finite_lower_sum(bounds_minus.lower());

                // SPSA gradient estimate.
                let diff = lower_plus - lower_minus;
                for (name, grad) in avg_grads.iter_mut() {
                    let pert = &perturbations[name];
                    ndarray::Zip::from(grad.view_mut())
                        .and(pert.view())
                        .for_each(|g, &p| {
                            if p.abs() > 0.5 {
                                *g += diff / (2.0 * BILINEAR_SPSA_EPS * p);
                            }
                        });
                }
            }
            Ok(())
        })();

        // Always restore original alphas, even on error.
        for (name, alpha) in bilinear_alphas.iter_mut() {
            if let Some(orig) = originals.get(name) {
                alpha.assign(orig);
            }
        }
        result?;

        // Average the gradients.
        let num_samples = config.spsa_samples.max(1) as f32;
        for grad in avg_grads.values_mut() {
            *grad /= num_samples;
        }

        Ok(avg_grads)
    }

    /// In-place Adam update for a single bilinear alpha.
    pub(super) fn update_bilinear_adam_inplace(
        &self,
        bilinear_alphas: &mut HashMap<String, Array4<f32>>,
        adam_m: &mut HashMap<String, Array4<f32>>,
        adam_v: &mut HashMap<String, Array4<f32>>,
        name: &str,
        gradient: &Array4<f32>,
        params: &AdamParams,
    ) {
        let Some(alpha) = bilinear_alphas.get_mut(name) else {
            return;
        };
        let Some(m) = adam_m.get_mut(name) else {
            return;
        };
        let Some(v) = adam_v.get_mut(name) else {
            return;
        };

        let t_f: f32 = params.t.max(1) as f32;
        let bias_correction1: f32 = (1.0_f32 - params.beta1.powf(t_f)).max(f32::EPSILON);
        let bias_correction2: f32 = (1.0_f32 - params.beta2.powf(t_f)).max(f32::EPSILON);

        ndarray::Zip::from(alpha.view_mut())
            .and(gradient.view())
            .and(m.view_mut())
            .and(v.view_mut())
            .for_each(|a, &g, m_val, v_val| {
                *m_val = params.beta1 * *m_val + (1.0 - params.beta1) * g;
                *v_val = params.beta2 * *v_val + (1.0 - params.beta2) * g * g;

                let m_hat = *m_val / bias_correction1;
                let v_hat = *v_val / bias_correction2;

                *a -= params.learning_rate * m_hat / (v_hat.sqrt() + params.epsilon);
                *a = a.clamp(0.0, 1.0);
                if a.is_nan() {
                    *a = 0.5;
                    *m_val = 0.0;
                    *v_val = 0.0;
                }
            });
    }
}
